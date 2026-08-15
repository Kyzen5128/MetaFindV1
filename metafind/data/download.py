"""Download everything MetaFind needs, and nothing else.

The paper names exactly two datasets (sec. 2.3):

* **Objaverse-LVIS** -- ~48K 3D assets. The raw asset is the GLB mesh; the 11
  renders and the structured captions are things we *produce* from it, not
  things to download. The meshes are kept, not discarded, because Table 2's
  iterative scene composition needs real geometry to place, not just embeddings.
* **ProcTHOR-10K** -- 10k+ house layouts, as JSON.

Plus the frozen weights those stages run on: ULIP-2, its open_clip ViT-bigG-14
backbone, and Qwen2.5-VL standing in for GPT-4o.

Deliberately not downloaded
---------------------------

ULIP-2 also publishes pre-sampled point clouds (185 GB) and its own renders
(474 GB) for all ~800k Objaverse objects. Neither is needed: we already fetch
the meshes, so point clouds can be sampled and views rendered from them, and
ULIP's renders are not the 11 orthographic views the paper specifies. Its
ShapeNet triplets (409 GB) belong to a different paper.

Everything here is resumable and verifies content rather than existence: a
truncated download that leaves a file behind must not be mistaken for a
finished one.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import time
from pathlib import Path

from metafind import paths


def _hf_snapshot(repo_id: str, dest: Path, repo_type: str = "model", allow: list[str] | None = None) -> Path:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id,
            repo_type=repo_type,
            local_dir=str(dest),
            allow_patterns=allow,
            max_workers=8,
        )
    )


# ------------------------------------------------------------------ datasets


def fetch_lvis_manifest() -> dict[str, str]:
    """The 46,052-uid manifest that defines Objaverse-LVIS.

    The paper says "approximately 48,000"; the released manifest holds 46,052.
    The manifest is the truth, so nothing downstream hardcodes 48000.
    """
    from huggingface_hub import hf_hub_download

    paths.OBJAVERSE.mkdir(parents=True, exist_ok=True)
    for name in ("lvis.json", "objaverse_lvis_metadata.json"):
        target = paths.OBJAVERSE / name
        if target.exists() and target.stat().st_size > 1000:
            continue
        src = hf_hub_download(
            "SFXX/ulip",
            f"ULIP_Objaverse_Triplets/{name}",
            repo_type="dataset",
            local_dir=str(paths.OBJAVERSE / "_tmp"),
        )
        target.write_bytes(Path(src).read_bytes())

    manifest = json.loads(paths.LVIS_MANIFEST.read_text())
    print(f"  manifest: {len(manifest)} assets")
    return manifest


OBJAVERSE_HF = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"


def _fetch_one_glb(uid: str, rel_path: str, dest_root: Path, attempts: int = 4) -> str:
    """One GLB, written atomically. Returns "" on success or the failure reason.

    Downloads to a .part file and renames only after the whole body arrives, so
    an interrupted fetch can never leave a truncated file that the resume scan
    would count as present.
    """
    import httpx

    dest = dest_root / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".glb.part")

    for attempt in range(attempts):
        try:
            with httpx.stream(
                "GET", f"{OBJAVERSE_HF}/{rel_path}", follow_redirects=True, timeout=60.0
            ) as r:
                r.raise_for_status()
                expected = int(r.headers.get("content-length", 0))
                written = 0
                with tmp.open("wb") as fh:
                    for block in r.iter_bytes(1 << 20):
                        fh.write(block)
                        written += len(block)
            if expected and written != expected:
                raise OSError(f"truncated: {written} of {expected} bytes")
            if written <= 1024:
                raise OSError(f"implausibly small: {written} bytes")
            tmp.replace(dest)
            return ""
        except Exception as exc:  # noqa: BLE001 -- retried, then recorded
            tmp.unlink(missing_ok=True)
            if attempt == attempts - 1:
                return f"{type(exc).__name__}: {exc}"
            time.sleep(2**attempt + random.random())
    return "unreachable"


def fetch_objaverse_glbs(uids: list[str], workers: int = 16) -> dict:
    """Fetch the GLB meshes, keeping them.

    Resumable: an asset counts as present only if its file is on disk and
    non-trivial in size, so a truncated fetch is redone.

    Deliberately does NOT use objaverse.load_objects. That helper runs a
    multiprocessing.Pool, and a single truncated HTTP response deadlocks the
    whole download permanently:

        worker raises urllib.error.ContentTooShortError
          -> the pool pickles it back to the parent
          -> ContentTooShortError.__init__(msg, content) needs two arguments,
             but unpickling supplies one
          -> TypeError inside the pool's _handle_results THREAD, which dies
          -> the parent blocks in futex_wait forever, waiting for results that
             will never be delivered

    Measured 2026-08-15: the run wedged after 2,047 of 46,052 assets and sat
    there for six hours. An enclosing try/except cannot catch it, because
    nothing is ever raised in the caller -- the call simply never returns.
    Threads keep the exception in-process where it can actually be handled.
    """
    import objaverse

    paths.OBJAVERSE_GLB.mkdir(parents=True, exist_ok=True)
    objaverse.BASE_PATH = str(paths.OBJAVERSE_GLB)
    objaverse._VERSIONED_PATH = str(paths.OBJAVERSE_GLB / "hf-objaverse-v1")
    object_paths = objaverse._load_object_paths()

    have = {p.stem for p in paths.OBJAVERSE_GLB.rglob("*.glb") if p.stat().st_size > 1024}
    todo = [u for u in uids if u not in have]
    print(f"  glbs: {len(have)} present, {len(todo)} to fetch, {workers} threads")

    unmapped = [u for u in todo if u not in object_paths]
    if unmapped:
        print(f"  {len(unmapped)} uids absent from object-paths.json.gz; skipped")

    failures: dict[str, str] = {}
    t0, done = time.time(), 0
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_one_glb, u, object_paths[u], paths.OBJAVERSE_GLB): u
            for u in todo
            if u in object_paths
        }
        for fut in cf.as_completed(futures):
            uid = futures[fut]
            if reason := fut.result():
                failures[uid] = reason
            done += 1
            if done % 200 == 0 or done == len(futures):
                rate = done / max(time.time() - t0, 1e-9)
                print(
                    f"  [{done:6d}/{len(futures)}] {rate * 60:.0f}/min, "
                    f"剩餘約 {(len(futures) - done) / max(rate, 1e-9) / 60:.0f} 分, "
                    f"失敗 {len(failures)}",
                    flush=True,
                )

    if failures:
        report = paths.OUTPUTS / "logs" / "glb_failures.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(failures, indent=2))
        print(f"  {len(failures)} failed; reasons in {report}")

    got = {p.stem for p in paths.OBJAVERSE_GLB.rglob("*.glb") if p.stat().st_size > 1024}
    return {"requested": len(uids), "present": len(got), "missing": len(set(uids) - got)}


def fetch_procthor() -> dict:
    """ProcTHOR-10K house layouts, written as one JSONL per split."""
    import prior

    paths.PROCTHOR.mkdir(parents=True, exist_ok=True)
    if all((paths.PROCTHOR / f"{s}.jsonl").exists() for s in ("train", "val", "test")):
        counts = {
            s: sum(1 for _ in (paths.PROCTHOR / f"{s}.jsonl").open())
            for s in ("train", "val", "test")
        }
        print(f"  procthor: already present {counts}")
        return counts

    dataset = prior.load_dataset("procthor-10k")
    counts = {}
    for split in ("train", "val", "test"):
        houses = dataset[split]
        out = paths.PROCTHOR / f"{split}.jsonl"
        with out.open("w") as fh:
            for i in range(len(houses)):
                fh.write(json.dumps(houses[i]) + "\n")
        counts[split] = len(houses)
        print(f"  procthor/{split}: {len(houses)} houses, {out.stat().st_size / 1e6:.0f} MB")
    return counts


# ------------------------------------------------------------------ models


def fetch_ulip2() -> Path:
    """The frozen ULIP-2 checkpoint: PointBERT weights plus the 1280-d projection."""
    dest = paths.MODELS / "ulip2"
    if paths.ULIP2_CKPT.exists() and paths.ULIP2_CKPT.stat().st_size > 3e8:
        print(f"  ulip2: already present ({paths.ULIP2_CKPT.stat().st_size / 1e6:.0f} MB)")
        return paths.ULIP2_CKPT
    _hf_snapshot(
        "SFXX/ulip",
        dest,
        repo_type="dataset",
        allow=["ULIP-2/pretrained_models/ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-*"],
    )
    print(f"  ulip2: {paths.ULIP2_CKPT.stat().st_size / 1e6:.0f} MB")
    return paths.ULIP2_CKPT


def fetch_qwen(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct") -> Path:
    """Qwen2.5-VL, standing in for GPT-4o as annotator and scene judge."""
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model_id, max_workers=8))
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print(f"  qwen: {total / 1e9:.1f} GB")
    return path


def fetch_openclip() -> None:
    """ViT-bigG-14, the frozen text/image half of ULIP-2.

    Pulled through open_clip rather than by URL so the file lands where the
    library will look for it.
    """
    import open_clip

    open_clip.create_model_and_transforms("ViT-bigG-14", pretrained="laion2b_s39b_b160k")
    print("  open_clip ViT-bigG-14: ready")


# ------------------------------------------------------------------ cli

STEPS = {
    "manifest": "Objaverse-LVIS uid manifest",
    "procthor": "ProcTHOR-10K house layouts",
    "ulip2": "ULIP-2 checkpoint",
    "openclip": "ViT-bigG-14",
    "qwen": "Qwen2.5-VL-7B",
    "glbs": "Objaverse-LVIS GLB meshes (~216 GB, slowest)",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=list(STEPS), help="run only these steps")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of GLBs, for a smoke run")
    ap.add_argument("--workers", type=int, default=16, help="download threads")
    args = ap.parse_args()

    paths.setup_env()
    paths.ensure_dirs()
    steps = args.only or list(STEPS)

    print("MetaFind 下載")
    print("=" * 60)
    for name in steps:
        print(f"\n[{name}] {STEPS[name]}", flush=True)
        if name == "manifest":
            fetch_lvis_manifest()
        elif name == "procthor":
            fetch_procthor()
        elif name == "ulip2":
            fetch_ulip2()
        elif name == "openclip":
            fetch_openclip()
        elif name == "qwen":
            fetch_qwen()
        elif name == "glbs":
            uids = sorted(json.loads(paths.LVIS_MANIFEST.read_text()))
            if args.limit:
                uids = uids[: args.limit]
            stats = fetch_objaverse_glbs(uids, workers=args.workers)
            print(f"  {stats}")

    print("\n" + "=" * 60)
    print(paths.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
