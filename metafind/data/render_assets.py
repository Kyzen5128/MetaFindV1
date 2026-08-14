"""Fetch Objaverse GLBs, render 11 views, discard the mesh (paper sec. 2.3).

Graph node: SG1's render stage.

Objaverse GLBs for the 46,052 LVIS assets come to roughly 216 GB (measured: mean
4.69 MB, median 1.53 MB over a 60-asset sample -- the distribution is heavily
right-tailed). The renders are 2.1 GB. So the meshes are streamed:

    download a batch -> render 11 views -> write JPEGs -> delete the GLBs

Peak disk is one batch rather than 216 GB, and nothing is kept that is not used.

Rendering is not the bottleneck. Measured at 31 ms/asset for all 11 views at
224px, the full set takes ~0.4 hours of GPU; the download dominates by two
orders of magnitude. Batches are therefore sized for download parallelism, not
render throughput.

Deleting the mesh is irreversible, so it happens only after the renders are
written, fsynced and verified, and the sidecar records the source URI and the
sha256 of every image so an asset can be re-fetched and re-rendered later.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"

LVIS_JSON = DATA / "sources/objaverse-lvis/ULIP_Objaverse_Triplets/lvis.json"
OUT_DIR = DATA / "artifacts/renders"
SIDECAR_DIR = DATA / "runs/sidecars/renders"
PROGRESS = DATA / "runs/progress/renders.json"
GLB_CACHE = DATA / "work/objaverse"

CODE_REVISION = "render_assets/1"
JPEG_QUALITY = 90


def _redirect_objaverse_cache() -> None:
    """Point objaverse at the data volume.

    Its download paths are module-level constants resolved at import, defaulting
    to ~/.objaverse. That partition has ~100 GB free against a ~216 GB corpus,
    so leaving it alone would fill the root filesystem even though the meshes
    are deleted as we go.
    """
    import objaverse

    GLB_CACHE.mkdir(parents=True, exist_ok=True)
    objaverse.BASE_PATH = str(GLB_CACHE)
    objaverse._VERSIONED_PATH = str(GLB_CACHE / "hf-objaverse-v1")


OBJECT_PATHS_URL = (
    "https://huggingface.co/datasets/allenai/objaverse/resolve/main/object-paths.json.gz"
)


def _object_paths_ok(path: Path) -> bool:
    """Does this file actually gunzip into a usable uid->path map?

    objaverse decides with ``if not os.path.exists(local_path)``, so a failed
    download that leaves a zero-byte file counts as cached and the next run gets
    a baffling gzip error instead of retrying. Existence is not correctness.
    """
    import gzip

    if not path.exists() or path.stat().st_size < 1_000_000:
        return False
    try:
        with gzip.open(path, "rb") as fh:
            data = json.load(fh)
        return isinstance(data, dict) and len(data) > 100_000
    except Exception:  # noqa: BLE001 -- any failure means unusable
        return False


def _ensure_object_paths(max_attempts: int = 5) -> None:
    """Put a verified object-paths index where objaverse expects it.

    objaverse fetches this 19.8 MB index with a bare ``urlretrieve``: no retry,
    no size check, and a truncated result left on disk. Fetching it here instead
    means the index is downloaded atomically and verified before objaverse ever
    looks for it, so its own code path never runs.
    """
    import urllib.request

    import objaverse

    target = Path(objaverse._VERSIONED_PATH) / "object-paths.json.gz"
    if _object_paths_ok(target):
        return

    if target.exists():
        print(f"    移除損毀的索引快取 ({target.stat().st_size} bytes)", flush=True)
        target.unlink()

    # A valid copy may already exist in objaverse's default location from an
    # earlier run; reuse it rather than re-downloading 19.8 MB.
    legacy = Path.home() / ".objaverse/hf-objaverse-v1/object-paths.json.gz"
    if legacy != target and _object_paths_ok(legacy):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, target)
        print(f"    沿用既有索引 {legacy}", flush=True)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_attempts + 1):
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        os.close(fd)
        try:
            urllib.request.urlretrieve(OBJECT_PATHS_URL, tmp)
            if not _object_paths_ok(Path(tmp)):
                raise ValueError(f"index failed verification ({Path(tmp).stat().st_size} bytes)")
            os.replace(tmp, target)
            print(f"    索引已下載並驗證 ({target.stat().st_size / 1e6:.1f} MB)", flush=True)
            return
        except Exception as exc:  # noqa: BLE001
            Path(tmp).unlink(missing_ok=True)
            if attempt == max_attempts:
                raise RuntimeError(f"object-paths index unavailable after {max_attempts} attempts") from exc
            delay = min(2**attempt, 30) + random.uniform(0, 3)
            print(f"    索引下載失敗 ({type(exc).__name__})，{delay:.0f}s 後重試", flush=True)
            time.sleep(delay)


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def render_paths(uid: str, n_views: int) -> list[Path]:
    return [OUT_DIR / uid / f"view_{i:02d}.jpg" for i in range(n_views)]


def load_done(shard_name: str) -> dict[str, list[str]]:
    """Assets already rendered, keyed by uid, valued by per-view sha256.

    Completion is decided on CONTENT: every image must exist AND hash to what
    the sidecar recorded. A half-written JPEG from a killed run would otherwise
    be mistaken for a finished asset (L2-RESUME).
    """
    path = SIDECAR_DIR / f"{shard_name}.jsonl"
    if not path.exists():
        return {}
    done: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") != "admitted":
            continue
        hashes = rec.get("view_sha256") or []
        paths = render_paths(rec["uid"], len(hashes))
        if all(p.exists() for p in paths) and all(
            hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in zip(paths, hashes)
        ):
            done[rec["uid"]] = hashes
    return done


def write_views(uid: str, frames, quality: int = JPEG_QUALITY) -> list[str]:
    """Write the rendered views as JPEGs, returning their sha256 in view order."""
    from PIL import Image

    out_dir = OUT_DIR / uid
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes: list[str] = []
    for i, frame in enumerate(frames):
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
        payload = buf.getvalue()

        target = out_dir / f"view_{i:02d}.jpg"
        fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        hashes.append(hashlib.sha256(payload).hexdigest())
    return hashes


def process_batch(uids: list[str], cfg, shard_name: str, keep_glb: bool = False) -> dict:
    """Download, render and discard one batch of assets."""
    import objaverse

    from metafind.data.render import render_asset

    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        paths = objaverse.load_objects(uids=uids, download_processes=8)
    except Exception as exc:  # noqa: BLE001
        # all_settled: a failed batch is recorded and the run continues.
        with open(SIDECAR_DIR / f"{shard_name}.jsonl", "a") as sc:
            for uid in uids:
                sc.write(json.dumps({
                    "uid": uid, "status": "quarantined", "failure_class": "TRANSIENT",
                    "exception_type": type(exc).__name__, "exception_msg": str(exc)[:400],
                    "code_revision": CODE_REVISION,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }) + "\n")
        return {"admitted": 0, "quarantined": len(uids)}

    admitted = quarantined = 0
    with open(SIDECAR_DIR / f"{shard_name}.jsonl", "a") as sc:
        for uid in uids:
            rec = {
                "uid": uid,
                "n_views": cfg.n_views,
                "layout": cfg.layout,
                "resolution": cfg.resolution,
                "code_revision": CODE_REVISION,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            glb = paths.get(uid)
            try:
                if glb is None or not Path(glb).exists():
                    raise FileNotFoundError(f"objaverse did not return a mesh for {uid}")
                rec["source_path"] = str(glb)
                rec["source_sha256"] = hashlib.sha256(Path(glb).read_bytes()).hexdigest()
                rec["source_bytes"] = Path(glb).stat().st_size

                frames = render_asset(glb, cfg)
                rec["view_sha256"] = write_views(uid, frames)
                rec["status"] = "admitted"
                admitted += 1
            except Exception as exc:  # noqa: BLE001 -- the reason is the product
                rec |= {
                    "status": "quarantined",
                    "failure_class": "DETERMINISTIC_INPUT",
                    "exception_type": type(exc).__name__,
                    "exception_msg": str(exc)[:400],
                    "traceback": traceback.format_exc()[-600:],
                }
                shutil.rmtree(OUT_DIR / uid, ignore_errors=True)
                quarantined += 1
            sc.write(json.dumps(rec) + "\n")
            sc.flush()

    if not keep_glb:
        # Irreversible, so it runs only after every sidecar line is flushed. The
        # recorded source_sha256 is what makes a re-fetch verifiable.
        for glb in paths.values():
            Path(glb).unlink(missing_ok=True)

    return {"admitted": admitted, "quarantined": quarantined}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="only the first N assets (0 = all)")
    ap.add_argument("--batch", type=int, default=64, help="assets per download batch")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--layout", default="fibonacci", choices=["fibonacci", "axis_aligned"])
    ap.add_argument("--keep-glb", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(DATA / "cache/hf"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    _redirect_objaverse_cache()
    _ensure_object_paths()

    from metafind.data.render import RenderConfig

    cfg = RenderConfig(resolution=args.resolution, layout=args.layout)

    uids = sorted(json.loads(LVIS_JSON.read_text()))
    if args.limit:
        uids = uids[: args.limit]

    shard_name = "renders"
    done = load_done(shard_name)
    todo = [u for u in uids if u not in done]
    print(
        f"{len(uids)} assets referenced; {len(done)} already rendered, {len(todo)} to go "
        f"({cfg.n_views} views, {cfg.layout}, {cfg.resolution}px)"
    )

    progress = {"admitted": len(done), "quarantined": 0, "total": len(uids)}
    t0 = time.time()
    for start in range(0, len(todo), args.batch):
        batch = todo[start : start + args.batch]
        res = process_batch(batch, cfg, shard_name, keep_glb=args.keep_glb)
        progress["admitted"] += res["admitted"]
        progress["quarantined"] += res["quarantined"]
        _atomic_write_json(PROGRESS, progress)

        seen = start + len(batch)
        rate = seen / max(time.time() - t0, 1e-9)
        print(
            f"[{seen:6d}/{len(todo)}] admitted={progress['admitted']} "
            f"quar={progress['quarantined']}  "
            f"({rate * 60:.0f} 資產/分, 剩餘約 {(len(todo) - seen) / max(rate, 1e-9) / 60:.0f} 分)",
            flush=True,
        )

    print(f"\n完成: admitted={progress['admitted']} quarantined={progress['quarantined']} / {len(uids)}")
    return 0 if progress["admitted"] + progress["quarantined"] == len(uids) else 2


if __name__ == "__main__":
    raise SystemExit(main())
