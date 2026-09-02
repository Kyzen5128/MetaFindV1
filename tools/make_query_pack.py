"""Build the Stage 1 QUERY-side observations: a second look at the same asset.

# SUPPORTS-NODE: n10_train_stage1
# SUPPORTS-NODE: n15_run_retrieval

DIAGNOSTIC ARTIFACTS. Nothing here writes under `pointclouds/` or `embeddings/`;
everything lands in `data/outputs/_probe/query_pack/`. The canonical corpus is
the GALLERY side and is not touched.

Why this exists
---------------
Stage 1 fed ONE `embeds` dict to both towers (`stage1.py:1792-1794`), so the
query text was the gallery's text -- the same cached vector, not merely an equal
one. Measured dev_val text R@1 96.42 against MetaFind's reported 13.8.

[PAPER 3experiments.tex:24] MetaFind names the mechanism itself: other models'
"'PC only' performance reflects retrieval using identical embeddings for both
query and gallery, leading to inflated accuracy", and credits its dual-tower
design as the cure. We have the dual tower and we fed it identical inputs. The
paper does not describe what else it does, so supplying a second observation is
an IMPLEMENTATION CHOICE whose basis is 3experiments.tex:24 -- NOT paper
silence. [MASTER ruling 2026-08-31, amending DL-050.]

The three arms, and what is NOT symmetric about them
-----------------------------------------------------
    query text    a non-canonical description candidate, re-serialised
    query image   ONE held-out view                     <- no artifact, a rule
    query pc      a second independent 10,000-point surface sample

    gallery       UNCHANGED on all three: canonical text, the 12-VIEW mean,
                  the canonical cloud.

[MASTER ruling 2026-08-31, option (a)] The gallery image stays the 12-view mean
and is NOT recomputed as the mean of the eleven views the query did not take.
Measured on dev_val at raw frozen-CLIP level, no tower:

    q = single view -> g = 12-view mean   R@1 0.9562
    q = single view -> g = 11-view mean   R@1 0.9054

Five points, against three costs: [PAPER 2methdology.tex:111] "all gallery asset
embeddings are precomputed and cached" stops being true once the gallery vector
depends on which view the query drew; `gallery_index.py:470` builds every
REPORTED protocol's gallery from `cached["image"]`, so the dev path and Table 1
A/B would silently use two different constructions; and
`gallery_index.gallery_encoder_sha256` hashes PARAMETERS, so G4 cannot see an
input-construction change at all. Recorded honestly: the held-out view still
sits in the gallery mean at weight 1/12, so exact identity is removed and a
twelfth of the leak is not.

Selection is FIXED per asset, never per epoch
----------------------------------------------
[MASTER ruling 2026-08-31] `uid_seed(uid)` decides every draw, so the selection
is re-derivable from the uid and needs no manifest to reproduce. Two of the
three arms cannot vary per epoch even in principle -- a varying image draw makes
the gallery uncacheable, and a varying pc draw multiplies a 7.7 GB artifact by
the epoch count -- and varying text ALONE would insert a second research
variable into a change whose single question is whether removing query/gallery
identity moves R@1. Per-epoch augmentation is a later arm, not this one.

Refusal, not substitution
--------------------------
An asset with no usable second observation is REFUSED and recorded. It is never
silently backed by the canonical vector: that would reintroduce exactly the
identity this artifact exists to remove, for an unknown subset, invisibly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metafind import paths, runlog  # noqa: E402

PACK = paths.OUTPUTS / "_probe" / "query_pack"
MANIFEST = PACK / "query_pack.json"

# Same offset the dev_val Protocol E sample already used, so the two shards are
# one construction rather than two. A different offset per shard would make the
# dev_train clouds a different treatment from the dev_val ones.
PC_SEED_OFFSET = 1000003
N_POINTS = 10000
N_VIEWS = 12


def held_out_view(uid: str) -> int:
    """Which of the 12 views is the QUERY's, and therefore not the query's alone.

    A rule, not an artifact: `uid_seed` is already this project's per-asset seed
    (`pointclouds.py:128`) and the dev_val pc probe already derives from it, so
    the view index needs nothing stored and cannot drift out of sync with a file.

    ⚠ Under option (a) the gallery mean still CONTAINS this view at weight 1/12.
    The name says held-out because it is held out of the query's alternatives,
    not because it is absent from the gallery.
    """
    from metafind.data.pointclouds import uid_seed

    return uid_seed(uid) % N_VIEWS


def pick_alternate(annotation: dict) -> tuple[int, str] | None:
    """The non-canonical caption this asset's query will use, or None.

    Walks `description_candidates` in rank order from rank 1 and returns the
    FIRST whose serialised form fits CLIP's 77-token context. Returns None when
    no candidate qualifies -- 55 assets in the train pool, measured.

    [OBSERVED DATA 2026-08-30, full train pool of 36,554]
        assets with <2 candidates                        14
        alternates over 77 tokens          2,086 / 146,047  (1.43%, max 87)
        assets with ZERO usable alternate                55

    The token gate is NOT bypassed. `refuse_if_overlong` exists because CLIP
    truncates at 77 SILENTLY and the template puts the description first, so an
    overlong string loses its tail -- the placement clause. Encoding a knowingly
    truncated caption would put a degraded vector in the query pool with nothing
    downstream able to tell. Falling through to the next candidate is the only
    move that keeps both the token budget and the independence.
    """
    from metafind.data.encode_text_image import TEXT_CONTEXT_LENGTH, true_token_count
    from metafind.models.resolve_stage1 import serialize_annotation

    canonical = (annotation.get("description") or "").strip()
    for cand in (annotation.get("description_candidates") or []):
        if cand.get("rank") == 0:
            continue        # the canonical one IS the gallery's; that is the leak
        # A rank>=1 candidate can be the canonical string again (the generator
        # repeated itself; 68 rows in the first pack). Skipping only rank 0 let
        # those through as "second observations" that were byte-equal to the
        # gallery's text -- the exact identity this pack exists to remove.
        if (cand.get("text") or "").strip() == canonical:
            continue
        text = serialize_annotation({**annotation, "description": cand["text"]})
        if true_token_count(text) <= TEXT_CONTEXT_LENGTH:
            return int(cand["rank"]), text
    return None


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _file_sha(path: Path) -> str:
    """Streamed, because the pc array is 7.7 GB.

    `hashlib.sha256(path.read_bytes())` reads the whole file into RAM. With 12
    sampling workers already resident that is what took this machine to 10 GB
    free and got a concurrent job OOM-killed -- and it would have been the LAST
    line of a 13-minute build. Same block loop as
    `encode_text_image.ulip2_ckpt_sha`.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------- pc arm

_MM = None
_WRITES = 0
# Dirty memmap pages accumulate until the kernel decides to write them back.
# Twelve workers filling a 7.7 GB mapping took this machine from 49 GB free to
# 10 GB and the OOM killer took 18 python processes, including a concurrent
# encode. Flushing on a fixed stride bounds the dirty set instead of hoping.
_FLUSH_EVERY = 32


def _pc_init(out_path):
    """Map the output ONCE per worker process, not once per asset.

    [MEASURED 2026-08-31] The first build opened `np.lib.format.open_memmap`
    inside the worker body, so every one of 31,985 assets mapped and unmapped a
    7.7 GB file. Throughput fell from 76/s to 5/s as the run went on and the
    page cache filled -- the sampling was never the bottleneck, the mapping was.
    """
    global _MM
    _MM = np.lib.format.open_memmap(out_path, mode="r+")


def _pc_worker(job):
    """Sample one asset into the shared memmap. Runs in a forked child.

    Writes THROUGH the memmap rather than returning the array: 31,985 x 240 KB
    back over a pipe is 7.7 GB of IPC to move data that is already destined for
    a file both processes have mapped. Rows are disjoint by index, so no lock.
    """
    row, uid, glb = job
    from metafind.data.pointclouds import pc_norm, sample_mesh, uid_seed

    try:
        xyz, rgb, *_ = sample_mesh(Path(glb), uid_seed(uid) + PC_SEED_OFFSET,
                                   N_POINTS)
    except Exception as exc:                      # a mesh this sampler cannot read
        return uid, row, None, f"{type(exc).__name__}: {exc}"
    normed = pc_norm(xyz.astype(np.float64)).astype(np.float32)
    if not np.isfinite(normed).all() or not np.isfinite(rgb).all():
        return uid, row, None, "non-finite values after normalisation"
    cloud = np.concatenate([normed, rgb.astype(np.float32)], axis=1)
    if cloud.shape != (N_POINTS, 6):
        return uid, row, None, f"shape {cloud.shape}, expected ({N_POINTS}, 6)"
    global _WRITES
    _MM[row] = cloud
    _WRITES += 1
    if _WRITES % _FLUSH_EVERY == 0:
        _MM.flush()
    return uid, row, _sha(cloud), None


def build_pc(uids: list[str], tag: str, workers: int) -> dict:
    """A second independent surface sample per asset. CPU only.

    The seed is `uid_seed(uid) + PC_SEED_OFFSET`, so this cloud is a different
    draw from the SAME mesh at the same density -- another observation, not
    another object and not a perturbation of the first.

    RESUMABLE, because this build was killed once at 21,124 of 31,985 assets and
    a 7.7 GB array is not something to redo from zero on a second interruption.
    Completed rows are appended to a `.done.jsonl` sidecar as they land; a
    restart maps the existing array `r+` and skips them. The sidecar is the
    record of what is actually written -- the array's own zeros are NOT used as
    the resume signal, because a legitimately all-zero row and an unwritten one
    are indistinguishable.
    """
    import multiprocessing as mp

    out_path = PACK / f"query_pc_{tag}_offset{PC_SEED_OFFSET}.npy"
    done_path = out_path.with_suffix(".npy.done.jsonl")
    # ASSERT the destination, do not merely intend it. `pointclouds/` is the
    # canonical corpus and a stray tag here would overwrite the gallery's own
    # clouds -- the one write that would make every number in the project wrong
    # and self-consistent at the same time.
    canonical = paths.POINTCLOUDS.resolve()
    if canonical == out_path.parent.resolve() or canonical in out_path.resolve().parents:
        raise SystemExit(f"refusing to write inside the canonical pointclouds/: {out_path}")

    glb_by_uid = {p.stem: p for p in paths.OBJAVERSE_GLB.rglob("*.glb")}
    missing = [u for u in uids if u not in glb_by_uid]
    if missing:
        raise SystemExit(f"{len(missing)} uid(s) have no GLB, e.g. {missing[:3]}")

    PACK.mkdir(parents=True, exist_ok=True)
    shape = (len(uids), N_POINTS, 6)
    shas, refused = {}, {}
    resume = out_path.exists() and done_path.exists()
    if resume:
        existing = np.load(out_path, mmap_mode="r")
        if existing.shape != shape:
            raise SystemExit(
                f"{out_path} is {existing.shape}, this build wants {shape}. "
                "Different pool: move the old array aside rather than resuming "
                "onto it -- the uid order is a positional index into it.")
        del existing
        uid_index = {u: i for i, u in enumerate(uids)}
        for line in done_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                # The done-list records the ROW each uid was written to. A
                # resume against a pool of the same size but different order
                # would otherwise serve one asset's cloud as another's.
                if uid_index.get(rec["uid"]) != rec["row"]:
                    raise SystemExit(
                        f"{done_path}: {rec['uid']} was written at row "
                        f"{rec['row']} but the current pool puts it at "
                        f"{uid_index.get(rec['uid'])}. Different pool order; "
                        "move the old array and done-list aside.")
                shas[rec["uid"]] = rec["sha256"]
        print(f"  resuming: {len(shas):,} already written", flush=True)
    else:
        mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                       shape=shape)
        del mm
        done_path.write_text("")

    jobs = [(i, u, str(glb_by_uid[u])) for i, u in enumerate(uids)
            if u not in shas]
    print(f"  {len(jobs):,} to sample", flush=True)
    t0 = time.time()
    with done_path.open("a") as done_fh, \
            mp.get_context("fork").Pool(workers, _pc_init, (str(out_path),)) as pool:
        # `.next(timeout=)` rather than plain iteration. A worker killed by the
        # OOM killer takes its task's result with it and the pool blocks
        # forever; a timeout converts that into a refusal naming the resume
        # command. The bound is per RESULT and the slowest mesh measured is
        # 0.35 s, so 300 s cannot fire on a legitimately slow asset.
        # chunksize=1 DELIBERATELY. With chunksize > 1 CPython wraps the
        # IMapUnorderedIterator in a plain generator to unpack the chunks, and a
        # generator has no `.next(timeout=)` -- the guard below silently stops
        # being available. The payload here is a 4-tuple of scalars (the cloud
        # itself goes through the memmap, not the pipe), so the chunking bought
        # almost nothing and cost the only thing standing between an OOM-killed
        # worker and an indefinite hang.
        results = pool.imap_unordered(_pc_worker, jobs)
        for k in range(1, len(jobs) + 1):
            try:
                uid, row, sha, err = results.next(timeout=300)
            except mp.TimeoutError:
                raise SystemExit(
                    f"no result for 300 s at {k - 1}/{len(jobs)}. A worker was "
                    "probably killed and its task lost, which stalls the pool "
                    "silently. Progress is in the done-list; re-run the same "
                    "command to resume, with fewer --workers.")
            if err:
                refused[uid] = err
                continue
            shas[uid] = sha
            # Flushed per record. A crash between the memmap write and this line
            # loses one asset and re-samples it; the reverse order would record
            # an asset the array does not hold.
            done_fh.write(json.dumps({"uid": uid, "row": row, "sha256": sha}) + "\n")
            if k % 2000 == 0:
                done_fh.flush()
                print(f"  [{k:6d}/{len(jobs)}] {k / (time.time() - t0):.0f}/s  "
                      f"refused {len(refused)}", flush=True)
    covered = [u for u in uids if u in shas]
    print(f"  {len(covered):,}/{len(uids):,} sampled, {len(refused)} refused, "
          f"{time.time() - t0:.0f}s -> {out_path}", flush=True)
    if len(covered) != len(uids):
        # The shard describes ONLY the rows that exist. Listing all `uids` here
        # while some rows are zeros is the failure this build already produced
        # once: a manifest that overstates its array serves unwritten rows as
        # observations, and `QueryPack` can only catch the count, not the gap.
        raise SystemExit(
            f"{len(uids) - len(covered):,} asset(s) did not sample; refusing to "
            f"write a shard that claims them. Re-run to resume: {refused}")
    return {"array": str(out_path), "uid_order": uids,
            "sha256_per_uid": shas, "refused": refused,
            "seed_offset": PC_SEED_OFFSET, "n_points": N_POINTS,
            "array_sha256": _file_sha(out_path)}


# ---------------------------------------------------------------- text arm

def build_text(uids: list[str], tag: str, device: str, batch: int) -> dict:
    """Encode the alternate caption through the SAME frozen tower as the gallery.

    The gallery's text vector came from n06's ViT-bigG-14 via
    `ULIPBackbone.encode_text`. If the query's came from anywhere else the two
    would live in different spaces and the comparison would measure the encoder
    difference instead of the caption difference -- the exact seam this whole
    change exists to close. So this reuses `encode_text_image.Encoder`, which is
    the same class n06 ran, rather than building a second text path.
    """
    from metafind.data.encode_text_image import Encoder

    rows, texts, ranks, refused = [], [], {}, {}
    for uid in uids:
        ann = json.loads((paths.ANNOTATIONS / f"{uid}.json").read_text())
        picked = pick_alternate(ann)
        if picked is None:
            refused[uid] = ("no description_candidate outside rank 0 fits CLIP's "
                            "77-token context")
            continue
        rank, text = picked
        rows.append(uid)
        texts.append(text)
        ranks[uid] = rank
    print(f"  {len(rows):,} encodable, {len(refused)} refused "
          f"(no usable alternate)", flush=True)

    enc = Encoder(device=device)
    out = np.empty((len(rows), 1280), dtype=np.float16)
    t0 = time.time()
    for i in range(0, len(rows), batch):
        chunk = texts[i:i + batch]
        with enc.torch.no_grad():
            vecs = enc.backbone.encode_text(chunk).float().cpu().numpy()
        out[i:i + len(chunk)] = vecs.astype(np.float16)
        if (i // batch) % 50 == 0:
            done = i + len(chunk)
            print(f"  [{done:6d}/{len(rows)}] "
                  f"{done / max(time.time() - t0, 1e-9) * 60:.0f}/min", flush=True)

    PACK.mkdir(parents=True, exist_ok=True)
    out_path = PACK / f"query_text_{tag}.npy"
    # Same dtype as the gallery's cached text (`embeddings/<uid>.npz` stores
    # float16). Storing the query side wider would make the two sides differ by
    # a precision as well as by a caption.
    np.save(out_path, out)
    print(f"  {len(rows):,} encoded in {time.time() - t0:.0f}s -> {out_path}",
          flush=True)
    return {"array": str(out_path), "uid_order": rows,
            "candidate_rank_per_uid": ranks, "refused": refused,
            "array_sha256": _file_sha(out_path)}


# ----------------------------------------------------------------- manifest

def merge(arm: str, shard: dict) -> None:
    """Add one shard to the pack, leaving the other arms alone.

    Read-modify-write rather than a fresh file: the arms are produced by
    separate jobs (one CPU, one GPU) and the dev_val pc shard predates this
    script. Rewriting the manifest per arm would drop whichever arm ran first.
    """
    PACK.mkdir(parents=True, exist_ok=True)
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {
        "what": "Stage 1 query-side observations: a second look at each asset.",
        "status": "DIAGNOSTIC ONLY -- not a canonical artifact",
        "gallery": "UNCHANGED: canonical text, 12-view mean image, canonical pc",
        "image": {"rule": "views[uid_seed(uid) % 12]",
                  "note": "a rule, not an array; the gallery mean still "
                          "contains this view at weight 1/12 [option (a)]"},
        "text": {"shards": []}, "pc": {"shards": []},
    }
    man.setdefault(arm, {"shards": []})
    tag = shard.get("tag")
    man[arm]["shards"] = [s for s in man[arm]["shards"] if s.get("tag") != tag]
    man[arm]["shards"].append(shard)
    man["written_at"] = time.time()
    man["code_revision"] = runlog.code_revision()
    man["code_dirty"] = runlog.code_dirty()
    tmp = MANIFEST.with_suffix(".json.part")
    with tmp.open("w") as fh:
        json.dump(man, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(MANIFEST)
    print(f"manifest -> {MANIFEST}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, choices=("pc", "text"))
    ap.add_argument("--split", required=True,
                    help="a key in splits.json's `object` block, e.g. dev_train")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    # SIX, not `cpu_count - 2`. Twelve was measured to exhaust 61 GB against a
    # 7.7 GB output and get workers OOM-killed; `mp.Pool` then repopulates the
    # dead worker but its in-flight task is GONE, so `imap_unordered` waits for
    # a result that will never arrive. The build hung at 31,891 of 31,985 with
    # every worker at zero CPU, no exception and no log line -- a silent stall
    # at 99.7%, which reads exactly like slow progress.
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    if args.split == "test":
        raise SystemExit("the test split is sealed; --split test is refused here")
    splits = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    if args.split not in splits:
        raise SystemExit(f"{args.split!r} is not in splits.json: {sorted(splits)}")
    uids = sorted(splits[args.split])
    if args.limit:
        uids = uids[: args.limit]
    print(f"{args.arm} arm, {args.split}, {len(uids):,} assets", flush=True)

    shard = (build_pc(uids, args.split, args.workers) if args.arm == "pc"
             else build_text(uids, args.split, args.device, args.batch))
    # A --limit run must not replace the full shard under the split's own tag:
    # it used to write the same array path and the same manifest tag, so a
    # ten-asset smoke overwrote a 31,931-row artifact. The limited run gets
    # its own tag and is never mistaken for coverage of the split.
    shard["tag"] = f"{args.split}_limit{args.limit}" if args.limit else args.split
    shard["n_assets"] = len(uids)
    merge(args.arm, shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
