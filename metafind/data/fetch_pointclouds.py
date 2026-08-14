"""Stream the Objaverse-LVIS point clouds out of the ULIP-2 release.

Graph node: part of ``n02_acquire_sources`` / SG1 (see docs/graph/).

The ULIP-2 release ships ~800k point clouds in 160 tar.gz shards totalling
185 GB, but ``lvis.json`` only references 46,052 of them. The 46,052 are spread
across every shard, so shards cannot be skipped -- but they also do not need to
be kept. This module therefore streams:

    download shard -> extract only the referenced members -> verify -> drop tar

Peak disk is one shard (~1.2 GB) rather than 185 GB, and what survives is the
~11 GB we actually use.

Design rules this implements (docs/graph/01_GRAPH_SPEC.md):

* **B1 durable progress** -- progress is a checkpoint file written atomically
  (tmp -> fsync -> rename). stdout is a mirror; losing it must not fail the run.
* **B2 per-item sidecar** -- every asset gets a record with its sha256 and, on
  failure, the real exception. Collecting reasons only at the end loses them all
  when the collector dies.
* **Resume equivalence (L2-RESUME)** -- the skip decision is made on CONTENT
  hash, never on file existence, so a truncated write from a killed run is
  re-done rather than mistaken for complete.
* **all_settled** -- a bad shard or member is quarantined with its reason; it
  never halts the run. The admitted set is what downstream uses.

Usage::

    python -m metafind.data.fetch_pointclouds                  # all 160 shards
    python -m metafind.data.fetch_pointclouds --shards 3       # first 3, for a smoke run
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import tarfile
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"

LVIS_JSON = DATA / "sources/objaverse-lvis/ULIP_Objaverse_Triplets/lvis.json"
OUT_DIR = DATA / "artifacts/pointclouds"
SIDECAR_DIR = DATA / "runs/sidecars/pointclouds"
PROGRESS = DATA / "runs/progress/pointclouds.json"
WORK = DATA / "work/pc_shards"

HF_REPO = "SFXX/ulip"
HF_PREFIX = "ULIP-2/objaverse_lvis"

N_POINTS = 10000
CODE_REVISION = "fetch_pointclouds/1"


# ---------------------------------------------------------------- durable state


def _atomic_write_json(path: Path, obj) -> None:
    """Write JSON atomically so a crash mid-write cannot corrupt the checkpoint."""
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


def load_progress() -> dict:
    if PROGRESS.exists():
        try:
            return json.loads(PROGRESS.read_text())
        except json.JSONDecodeError:
            # A corrupt checkpoint must fail closed, not be silently treated as
            # "nothing done yet" -- that would redo work and could double-write.
            raise RuntimeError(f"progress file is corrupt: {PROGRESS}. Inspect it before rerunning.")
    return {"shards_done": {}, "admitted": 0, "skipped": 0, "quarantined": 0}


# ---------------------------------------------------------------- verification


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def verify_pointcloud(raw: bytes) -> tuple[str, dict]:
    """Assert the payload is a usable ULIP-2 colored point cloud.

    Returns (sha256, stats). Raises on anything malformed -- this is a content
    assertion, not an existence check: a file that loads but holds the wrong
    shape or NaNs is a failure, and would otherwise poison training silently.
    """
    digest = sha256_bytes(raw)
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as fh:
        fh.write(raw)
        tmp = fh.name
    try:
        obj = np.load(tmp, allow_pickle=True).item()
    finally:
        os.unlink(tmp)

    if not isinstance(obj, dict):
        raise ValueError(f"expected a dict payload, got {type(obj).__name__}")
    for key in ("xyz", "rgb"):
        if key not in obj:
            raise ValueError(f"missing key {key!r}; has {sorted(obj)}")

    xyz, rgb = np.asarray(obj["xyz"]), np.asarray(obj["rgb"])
    if xyz.shape != (N_POINTS, 3):
        raise ValueError(f"xyz shape {xyz.shape} != {(N_POINTS, 3)}")
    if rgb.shape != (N_POINTS, 3):
        raise ValueError(f"rgb shape {rgb.shape} != {(N_POINTS, 3)}")
    if not np.isfinite(xyz).all():
        raise ValueError("xyz contains non-finite values")
    if float(np.abs(xyz).max()) == 0.0:
        raise ValueError("xyz is all zeros")

    return digest, {
        "xyz_min": float(xyz.min()),
        "xyz_max": float(xyz.max()),
        "rgb_min": float(rgb.min()),
        "rgb_max": float(rgb.max()),
    }


# ---------------------------------------------------------------- main loop


def wanted_by_shard(lvis: dict[str, str]) -> dict[str, dict[str, str]]:
    """Group the referenced members by shard: {shard: {member_path: uid}}."""
    grouped: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for uid, rel in lvis.items():
        shard = str(rel).split("/")[0]
        grouped[shard][str(rel)] = uid
    return grouped


# TRANSIENT per SKILL section 11.1: same input, retrying may well succeed.
# A read timeout mid-download previously killed the whole run at shard 27 of 160.
TRANSIENT_ERRORS = (
    "ReadTimeout", "ConnectTimeout", "ConnectError", "ReadError", "WriteError",
    "PoolTimeout", "RemoteProtocolError", "ChunkedEncodingError",
    "ConnectionError", "IncompleteRead", "ProtocolError", "SSLError", "Timeout",
)
MAX_SHARD_ATTEMPTS = 5


def _is_transient(exc: BaseException) -> bool:
    """Network hiccups are retryable; a 404 or a disk-full error is not.

    Matching on class name rather than importing httpx/requests keeps this
    working whichever HTTP stack huggingface_hub is built on, and it walks the
    __cause__ chain because hub wraps the original error.
    """
    seen = 0
    cur: BaseException | None = exc
    while cur is not None and seen < 10:
        if type(cur).__name__ in TRANSIENT_ERRORS:
            return True
        cur = cur.__cause__ or cur.__context__
        seen += 1
    return False


def _download_with_retry(shard: str, max_attempts: int = MAX_SHARD_ATTEMPTS) -> str:
    """Fetch one shard, retrying transient network failures with backoff + jitter.

    Jitter matters: without it, several retries after a common upstream blip
    would resynchronise and hammer the server together.
    """
    from huggingface_hub import hf_hub_download

    last: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return hf_hub_download(
                HF_REPO,
                f"{HF_PREFIX}/{shard}.tar.gz",
                repo_type="dataset",
                local_dir=str(WORK),
            )
        except Exception as exc:  # noqa: BLE001 -- classify, then decide
            last = exc
            if not _is_transient(exc):
                raise
            if attempt == max_attempts:
                break
            delay = min(2**attempt, 60) + random.uniform(0, 5)
            print(
                f"    {shard}: {type(exc).__name__} (第 {attempt}/{max_attempts} 次)，"
                f"{delay:.0f}s 後重試",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(f"{shard}: {max_attempts} 次嘗試後仍失敗") from last


def process_shard(shard: str, members: dict[str, str], keep_tar: bool = False) -> dict:
    from huggingface_hub import hf_hub_download

    WORK.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)

    # Skip on CONTENT, not existence: an asset counts as done only if its bytes
    # hash to what the sidecar recorded (L2-RESUME).
    done: dict[str, str] = {}
    sidecar_path = SIDECAR_DIR / f"{shard}.jsonl"
    if sidecar_path.exists():
        for line in sidecar_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from a kill; that item is simply redone
            if rec.get("status") == "admitted":
                out = OUT_DIR / f"{rec['uid']}.npy"
                if out.exists() and sha256_bytes(out.read_bytes()) == rec["sha256"]:
                    done[rec["uid"]] = rec["sha256"]

    todo = {m: u for m, u in members.items() if u not in done}
    if not todo:
        # "admitted" means newly written by THIS run; reporting len(done) here as
        # well double-counted every resumed shard and inflated the completeness
        # equation, which is the one number that tells us the gallery
        # denominator is right.
        return {"shard": shard, "admitted": 0, "quarantined": 0, "skipped": len(done)}

    tar_path = _download_with_retry(shard)

    admitted = quarantined = 0
    seen: set[str] = set()

    # MUST be a single sequential pass in stream mode ("r|gz").
    #
    # A .tar.gz has no random access: every extractfile() by member re-inflates
    # the stream from the beginning. Measured on shard 000-000 (1228 wanted
    # members): random access 2184 ms/item -> 45 min/shard -> 119 h for 160
    # shards. One sequential pass does the same 1228 in 4.9 s. 542x.
    with open(sidecar_path, "a") as sc, tarfile.open(tar_path, "r|gz") as tf:
        for member in tf:
            if not member.isfile() or member.name not in todo:
                continue
            member_path = member.name
            uid = todo[member_path]
            seen.add(member_path)
            rec = {
                "uid": uid,
                "shard": shard,
                "member": member_path,
                "code_revision": CODE_REVISION,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            try:
                fh = tf.extractfile(member)
                if fh is None:
                    raise OSError(f"{member_path} is not a regular file")
                raw = fh.read()
                digest, stats = verify_pointcloud(raw)

                out = OUT_DIR / f"{uid}.npy"
                fd, tmp = tempfile.mkstemp(dir=OUT_DIR, suffix=".tmp")
                with os.fdopen(fd, "wb") as w:
                    w.write(raw)
                    w.flush()
                    os.fsync(w.fileno())
                os.replace(tmp, out)

                rec |= {"status": "admitted", "sha256": digest, "bytes": len(raw), **stats}
                admitted += 1
            except Exception as exc:  # noqa: BLE001 -- the reason IS the product here
                rec |= {
                    "status": "quarantined",
                    "failure_class": "DETERMINISTIC_INPUT",
                    "exception_type": type(exc).__name__,
                    "exception_msg": str(exc)[:400],
                    "traceback": traceback.format_exc()[-800:],
                }
                quarantined += 1
            sc.write(json.dumps(rec) + "\n")
            sc.flush()

        # Members lvis.json referenced but the archive does not contain. In
        # stream mode this can only be known after the pass, and it must be
        # recorded: silently missing assets would shrink the gallery denominator
        # without leaving any trace of why.
        for member_path, uid in todo.items():
            if member_path in seen:
                continue
            sc.write(
                json.dumps(
                    {
                        "uid": uid,
                        "shard": shard,
                        "member": member_path,
                        "status": "quarantined",
                        "failure_class": "DETERMINISTIC_INPUT",
                        "exception_type": "FileNotFoundError",
                        "exception_msg": f"{member_path} not present in {shard}.tar.gz",
                        "code_revision": CODE_REVISION,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                )
                + "\n"
            )
            quarantined += 1
        sc.flush()

    if not keep_tar:
        Path(tar_path).unlink(missing_ok=True)

    return {
        "shard": shard,
        "admitted": admitted,
        "quarantined": quarantined,
        "skipped": len(done),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=int, default=0, help="only the first N shards (0 = all)")
    ap.add_argument("--keep-tar", action="store_true", help="do not delete each tar after extraction")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(DATA / "cache/hf"))
    # Authenticating switches the Hub onto its xet transfer backend, which
    # measured at roughly 0.6 KB/s here -- effectively stalled -- against 6 MB/s
    # over plain HTTP. Opt out unless the caller insists.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    lvis = json.loads(LVIS_JSON.read_text())
    grouped = wanted_by_shard(lvis)
    shards = sorted(grouped)
    if args.shards:
        shards = shards[: args.shards]

    progress = load_progress()
    total_wanted = sum(len(grouped[s]) for s in shards)
    print(f"{len(shards)} shards, {total_wanted} point clouds referenced by lvis.json")

    # Rate over a rolling window of shards that actually did work. A cumulative
    # average is badly misleading on resume: shards already complete finish
    # instantly and inflate the estimate for the rest of the run.
    recent: collections.deque[float] = collections.deque(maxlen=8)

    for i, shard in enumerate(shards, 1):
        t_shard = time.time()
        try:
            res = process_shard(shard, grouped[shard], keep_tar=args.keep_tar)
        except Exception as exc:  # noqa: BLE001
            # all_settled: record the failure and carry on. One unreachable
            # shard must not discard the other 159, and the reason has to
            # survive so the shortfall stays diagnosable.
            n_lost = len(grouped[shard])
            print(f"[{i:3d}/{len(shards)}] {shard}  失敗: {type(exc).__name__}: {exc}", flush=True)
            with open(SIDECAR_DIR / f"{shard}.jsonl", "a") as sc:
                for member_path, uid in grouped[shard].items():
                    sc.write(json.dumps({
                        "uid": uid, "shard": shard, "member": member_path,
                        "status": "quarantined", "failure_class": "TRANSIENT",
                        "exception_type": type(exc).__name__,
                        "exception_msg": str(exc)[:400],
                        "code_revision": CODE_REVISION,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }) + "\n")
            res = {"shard": shard, "admitted": 0, "quarantined": n_lost, "skipped": 0}
        took = time.time() - t_shard
        if res["admitted"] or res["quarantined"]:
            recent.append(took)

        progress["shards_done"][shard] = res
        # admitted = newly written this run; skipped = already present. Their sum
        # is what must reconcile against lvis.json, so they are tracked apart.
        progress["admitted"] = sum(v["admitted"] for v in progress["shards_done"].values())
        progress["skipped"] = sum(v["skipped"] for v in progress["shards_done"].values())
        progress["quarantined"] = sum(v["quarantined"] for v in progress["shards_done"].values())
        _atomic_write_json(PROGRESS, progress)

        have = progress["admitted"] + progress["skipped"] + progress["quarantined"]
        if recent:
            eta = (len(shards) - i) * (sum(recent) / len(recent)) / 60
            eta_s = f"剩餘約 {eta:.0f} 分"
        else:
            eta_s = "剩餘 待測"
        print(
            f"[{i:3d}/{len(shards)}] {shard}  "
            f"admitted={res['admitted']:4d} skipped={res['skipped']:4d} quar={res['quarantined']:2d}  "
            f"完成 {have}/{total_wanted}  ({eta_s})",
            flush=True,
        )

    have = progress["admitted"] + progress["skipped"] + progress["quarantined"]
    print(
        f"\n完成: admitted={progress['admitted']} skipped={progress['skipped']} "
        f"quarantined={progress['quarantined']}  合計 {have}/{total_wanted}"
    )
    print("以 sidecar 為準的驗證請跑: python -m metafind.data.verify_pointclouds")
    # Completeness equation (L2-COMPLETE). Not an assertion here -- the gate owns
    # that judgement -- but the numbers are surfaced so a mismatch is visible.
    if have != total_wanted:
        print(f"警告: 合計 {have} != {total_wanted}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
