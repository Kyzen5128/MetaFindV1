"""Cross-modal alignment pretraining: train the point encoder and the fusers.

# IMPLEMENTS-NODE: n10_train_stage1

Writes ``stage1_ckpt``, ``run_progress``, ``cost_ledger`` and ``degraded_flags``.

What trains and what does not
------------------------------

[PAPER 2.6] "Both query and gallery encoders are trained on large-scale
object-level data from Objaverse-LVIS." Under U-34's resolved reading that means
the point encoder, its projection and both fusion modules; OpenCLIP stays
frozen, so its text and image outputs come from n06's cache rather than being
recomputed 46,052 times per epoch.

The point cloud is the one modality that CANNOT be cached: PointBERT is in the
optimizer, so a cached point embedding would be the output of a network that is
about to change. Caching all three is what made an earlier draft Table 3's
"train fuser only" row -- the row the paper reports as WORSE (8.7 against 11.4).

[L1-LOSS-STAGE1-UNIDIRECTIONAL] Eq. 5 is query->gallery only. Eq. 7a/7b's
symmetric form belongs to Stage 2, and the paper is explicit about the
difference, so ``bidirectional`` stays False here.

Why the checkpoint saves so little
-----------------------------------

[F27, L1-CKPT-TRAINABLE-ONLY] ``torch.save(model.state_dict())`` would write
ViT-bigG-14 as well: 2.5B frozen parameters, 10.2 GB, reconstructible from the
pinned OpenCLIP weights and moved by not one step. Across Table 3's ten variants
plus the main line that is 112 GB against 1.9 GB, on a shared volume. Only
parameters with ``requires_grad`` are saved, and the count travels with the file
so the claim is checkable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import contextlib
import fcntl
import json
import math
import os
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import shutil
import sys

from metafind import paths, runlog
from metafind.models.stage1_config import UnsupportedProtocol

# Before torch or open_clip: HF_HOME is read at import time and ViT-bigG-14 is
# 9.5 GB.
paths.setup_env()

from metafind.models.stage1_config import (  # noqa: E402
    PER_VIEW_AGGREGATIONS,
    PRECOMPUTABLE_AGGREGATIONS,
    REQUIRED_HYPERPARAMETERS,
)

NODE = "n10_train_stage1"
TRAINER_VERSION = 2   # v2 saves backbone + tower + loss; v1 dropped the point encoder

# Open lock file descriptors, kept referenced so the flock survives. Closing
# an fd releases the lock, and nothing else holds these.
_HELD_LOCKS: list[int] = []

CKPT_PATH = paths.CHECKPOINTS / "stage1.pt"
CKPT_RECORD = paths.CHECKPOINTS / "stage1_ckpt.json"
# Kept beside the per-epoch file rather than replacing it: the last epoch is
# what `--phase final` produces, and the development phase needs both -- the
# curve comes from every epoch, the answer comes from one of them.
BEST_CKPT_PATH = paths.CHECKPOINTS / "stage1_best.pt"
BEST_CKPT_RECORD = paths.CHECKPOINTS / "stage1_best_ckpt.json"


def weight_decay_groups(named, weight_decay: float) -> list[dict]:
    """Two optimizer groups; the rule reads the parameter NAME.

    [UPSTREAM-OFFICIAL-IMPL upstream/ULIP/main.py:129-135] verbatim predicate:
    `p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n` goes to the group with
    no decay. A single flat weight_decay pulls LayerNorm scales and every bias
    toward zero -- decay applied to parameters that have nothing to overfit
    with. The MECHANISM is inherited; `weight_decay` itself is USER-APPROVED
    separately (resolve_stage1.py).

    Frozen parameters are dropped here rather than by the caller, so an
    optimizer can never receive one.

    ANY GROUP ADDED HERE MUST STATE ITS OWN `weight_decay`. Upstream passes
    `weight_decay=args.wd` to AdamW as well as per-group; that top-level value
    is dropped here so nothing implicit competes with the groups -- which also
    removes the fallback, so a third group that omits the key would silently get
    torch's own default of 1e-2, an order of magnitude off.
    """
    p_wd, p_non_wd = [], []
    for n, p in named:
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)
    return [{"params": p_wd, "weight_decay": weight_decay},
            {"params": p_non_wd, "weight_decay": 0.0}]


def cosine_schedule(base: float, final: float, epochs: int, niter_per_ep: int,
                    warmup_epochs: int, start_warmup: float) -> np.ndarray:
    """One learning rate per optimizer step: linear warmup, then cosine to `final`.

    [UPSTREAM-OFFICIAL-IMPL upstream/ULIP/utils/utils.py:215-226] The shape is
    reproduced rather than imported -- that module pulls in ULIP's distributed
    training stack -- and it is a SHAPE, which is what may be inherited; the four
    numbers that parameterise it are USER-APPROVED separately.

    `torch.optim.lr_scheduler.CosineAnnealingLR`, which stood here, cannot
    express it. It has no warmup, and its floor is `eta_min`, default 0, not
    `lr_end`. A run using it spent step 0 at the full base rate and finished at
    exactly 0.

    Measured against a verbatim reimplementation of upstream's function: on
    every input upstream survives, this returns a BIT-IDENTICAL array (5x100,
    250x10, no-warmup, warmup==epochs, niter==0 -- exact equality, not
    approximate). The two guards below change only inputs upstream refuses.

      * `warmup_epochs > epochs` -- upstream raises AssertionError here, because
        `warmup_iters` exceeds the total and its concatenation comes out the
        wrong length. Warmup is truncated to the run instead. NOTE: the case
        `warmup_epochs == epochs` needs no guard -- upstream handles it, and so
        does this, identically. An earlier version of this comment named that
        case, and named it wrongly.
      * `niter_per_ep == 0` -- upstream ALSO returns an empty array here; this
        does not repair anything, it makes the empty return explicit instead of
        incidental.
    """
    total = epochs * niter_per_ep
    if total <= 0:
        return np.zeros(0, dtype=np.float64)
    warmup_iters = min(max(warmup_epochs, 0) * niter_per_ep, total)
    warmup = (np.linspace(start_warmup, base, warmup_iters)
              if warmup_iters > 0 else np.zeros(0, dtype=np.float64))
    n_cos = total - warmup_iters
    if n_cos <= 0:
        return warmup
    iters = np.arange(n_cos)
    cos = final + 0.5 * (base - final) * (1 + np.cos(np.pi * iters / n_cos))
    schedule = np.concatenate((warmup, cos))
    assert len(schedule) == total
    return schedule


def load_protocols() -> tuple[dict, dict, dict]:
    """Everything n10 is not allowed to decide for itself."""
    def read(name: str, writer: str) -> dict:
        path = paths.OUTPUTS / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run {writer} first")
        return json.loads(path.read_text())

    encoding = read("stage1_encoding_protocol.json", "n05b_resolve_stage1_encoding")
    training = read("stage1_protocol.json", "n09_build_splits")
    hyperparameters = read("stage1_hyperparameters.json", "n05b_resolve_stage1_encoding")

    for name, proto in (("stage1_encoding_protocol", encoding),
                        ("stage1_protocol", training)):
        if proto.get("status") != "resolved":
            raise ValueError(f"{name} is {proto.get('status')!r}")
    if training["hyperparameter_config_hash"] != hyperparameters["sha256"]:
        raise ValueError(
            "stage1_protocol points at a different hyperparameter artifact "
            f"({training['hyperparameter_config_hash'][:12]}) than the one on "
            f"disk ({hyperparameters['sha256'][:12]}). G3 dereferences this."
        )
    missing = [f for f in REQUIRED_HYPERPARAMETERS if f not in hyperparameters["values"]]
    if missing:
        raise ValueError(f"hyperparameters missing {', '.join(missing)}")
    return encoding, training, hyperparameters


# Arms a query pack can supply. `image` needs no array -- it is a rule over the
# `views` matrix n06 already stores -- but it is named here anyway so that
# "which arms are independent" is one list rather than three implicit branches.
QUERY_ARMS = ("text", "image", "pc")


class QueryPack:
    """The QUERY side's own observations of each asset. Gallery untouched.

    Stage 1 built ONE `embeds` dict and handed it to both towers, so the query
    text was not merely equal to the gallery's text -- it was the same cached
    vector. Measured dev_val text R@1 96.42 against MetaFind's reported 13.8.

    [PAPER 3experiments.tex:24] is the basis, and it is NOT paper silence:
    MetaFind writes that other models' "'PC only' performance reflects retrieval
    using identical embeddings for both query and gallery, leading to inflated
    accuracy", and credits its dual-tower design as the cure. We have a real
    dual tower -- `model.query` and `model.gallery` are distinct -- and feeding
    it identical inputs still gives 96.42, so the paper's stated mechanism is
    implemented and does not produce the paper's behaviour. Supplying a second
    observation is an IMPLEMENTATION CHOICE against that sentence.
    [MASTER ruling 2026-08-31, amending DL-050, which recorded "paper silence".]

    ⚠ This removes a MEASURED LEAK. It does not close the gap to Table 1, and
    must never be described as doing so: an independent caption moved text
    96.42 -> 74.98, still 5.4x the paper's 13.8, and the no-tower control scores
    99.56 -- higher than the model. Gallery size is separately eliminated
    (4,569 -> 9,138 costs 1.5-3.5 pp). Three candidate explanations are down and
    the gap is not explained.

    What each arm draws
    --------------------
        text   a non-canonical `description_candidates` entry, re-serialised
               through the same template and encoded by the same frozen tower
        image  ONE view, `views[uid_seed(uid) % self.n_views]`, where
               `n_views` is the encoding protocol's count (12 on the
               corpus this pack was built for), not a literal
        pc     a second independent 10,000-point sample of the same mesh

    Fixed per asset, never per epoch [MASTER ruling 2026-08-31]: `uid_seed`
    decides every draw, so a selection is re-derivable from the uid alone. Two
    arms could not vary anyway -- a varying image draw makes the gallery
    uncacheable and a varying pc draw multiplies a 7.7 GB artifact by the epoch
    count -- and varying text alone would put a second research variable inside
    a change whose single question is whether removing query/gallery identity
    moves R@1.

    ⚠ THE IMAGE ARM DOES NOT REMOVE THE WHOLE IMAGE LEAK. [MASTER ruling, option
    (a)] The gallery image stays the 12-VIEW mean, so the query's view is still
    inside it at weight 1/12. The alternative -- an 11-view mean excluding it --
    was measured to buy 5 points at raw CLIP level (0.9562 -> 0.9054) and to
    cost three things: [PAPER 2methdology.tex:111] "all gallery asset embeddings
    are precomputed and cached" stops being true when the gallery vector depends
    on which view the query drew; `gallery_index.py` builds every REPORTED
    protocol's gallery from `cached["image"]`, so Table 1 A/B and the dev path
    would silently diverge; and `gallery_index.gallery_encoder_sha256` hashes
    PARAMETERS, so no gate could see the difference. Exact identity is removed;
    a twelfth of the leak is not.

    Refusal, never substitution
    ----------------------------
    `require()` fails at CONSTRUCTION, listing what is missing, rather than
    letting `__getitem__` fall back to the canonical vector for an uncovered
    uid. A silent fallback would restore the identity this class exists to
    remove, for an unknown subset, with nothing downstream able to tell -- the
    same defect class as a silently truncated caption. 55 assets in the train
    pool have no alternate caption that fits CLIP's 77-token context and 14 have
    no second candidate at all, so the uncovered set is real, not theoretical.
    """

    def __init__(self, manifest_path: str | Path, n_views: int) -> None:
        # `n_views` comes from the encoding protocol (`protocol_n_views`), not
        # from this manifest and not from a constant: the image arm is a RULE
        # over the corpus's `views` matrix, so the modulus belongs to whatever
        # produced that matrix. Required, because a default here would be the
        # compile-time constant again under another name.
        self.n_views = int(n_views)
        self.path = Path(manifest_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found -- build it with tools/make_query_pack.py")
        self.manifest = json.loads(self.path.read_text())
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.rows: dict[str, dict[str, tuple[int, int]]] = {}
        self.arrays: dict[str, list] = {}
        self.refused: dict[str, dict[str, str]] = {}
        for arm in ("text", "pc"):
            shards = (self.manifest.get(arm) or {}).get("shards") or []
            if not shards:
                continue
            # mmap: the pc shards are 7.7 GB for dev_train alone, and a training
            # run touches each row once per epoch. Reading them eagerly here
            # would cost the memory whether or not --preload was asked for.
            self.arrays[arm] = [np.load(sh["array"], mmap_mode="r") for sh in shards]
            index, refused = {}, {}
            for si, sh in enumerate(shards):
                # [FOUND 2026-08-31, the hard way] A killed build left a 31,985-
                # row array on disk under a manifest still describing the
                # 8-asset smoke shard that preceded it. Nothing raised: the
                # first eight uids happened to be the same eight, in the same
                # order, so `vector()` returned CORRECT clouds -- correct by
                # coincidence of sort order, and it would have gone on being
                # correct until the day the pools differed. Rows and uids are a
                # positional correspondence and nothing else enforced it.
                rows = self.arrays[arm][si].shape[0]
                if rows != len(sh["uid_order"]):
                    raise ValueError(
                        f"{self.path}: {arm!r} shard {sh.get('tag')!r} lists "
                        f"{len(sh['uid_order']):,} uids but {sh['array']} has "
                        f"{rows:,} rows. The uid order is a POSITIONAL index "
                        "into that array, so a mismatch silently serves one "
                        "asset's observation as another's. Rebuild the shard.")
                for ri, uid in enumerate(sh["uid_order"]):
                    index[uid] = (si, ri)
                refused.update(sh.get("refused") or {})
            self.rows[arm] = index
            self.refused[arm] = refused
        # `image` is a rule, so it is available for every asset with a `views`
        # matrix and carries no index. It is active whenever the manifest
        # declares it, which is what keeps "declared but never consumed" from
        # being possible: an arm absent here does not silently fall back to the
        # gallery vector, it simply is not claimed.
        self.arms = tuple(a for a in QUERY_ARMS
                          if a in self.arrays or (a == "image" and self.manifest.get("image")))
        if not self.arms:
            raise ValueError(f"{self.path} declares no usable query arm")

    def require(self, uids: list[str]) -> None:
        """Every uid must be covered by every active array-backed arm, or stop.

        Reported per arm with the recorded refusal reason where there is one, so
        the operator can tell "this asset has no usable alternate caption" from
        "this shard was built for a different split".
        """
        for arm, index in self.rows.items():
            missing = [u for u in uids if u not in index]
            if not missing:
                continue
            why = [f"{u}: {self.refused.get(arm, {}).get(u, 'not in any shard')}"
                   for u in missing[:3]]
            raise ValueError(
                f"query pack {self.path} covers {len(index):,} uids for the "
                f"{arm!r} arm but {len(missing):,} of the {len(uids):,} "
                f"requested are absent, e.g. {why}. Refusing: falling back to "
                f"the canonical {arm} vector for these would put the query and "
                "the gallery back on the same observation for an unrecorded "
                "subset. Filter the pool, or rebuild the shard.")

    def covered(self, uids: list[str]) -> tuple[list[str], list[str]]:
        """Split a pool into (kept, dropped) WITHOUT relaxing `require`.

        The two are deliberately separate. `require` is the GUARD and lives in
        the callee: it refuses any pool handed to a dataset that the pack cannot
        cover, and it stays a refusal. This is the POLICY and is called only
        from an entry point that has the authority to change a pool and the
        obligation to record it -- so a caller cannot quietly obtain a filtered
        pool by calling the dataset, and the guard cannot be softened into a
        warning by anyone who merely wants their run to start.

        [MASTER ruling 2026-08-31] The 55 uncovered assets are DROPPED. Keeping
        them would leave the query text equal to the gallery's for a silent
        0.15% minority, inside the very run that exists to measure that
        equality's removal -- a leak surviving in the experiment about the leak.
        """
        keep = [u for u in uids
                if all(u in idx for idx in self.rows.values())]
        return keep, [u for u in uids if u not in set(keep)]

    def vector(self, arm: str, uid: str) -> np.ndarray:
        si, ri = self.rows[arm][uid]
        return np.asarray(self.arrays[arm][si][ri], dtype=np.float32)

    def view_index(self, uid: str) -> int:
        """Which view the query takes. A rule over `uid_seed`, not a stored map.

        `uid_seed` is already the project's per-asset seed (`pointclouds.py:128`)
        and the pc arm derives from it too, so the image draw cannot drift out
        of sync with a file that would have to be kept in step.

        The modulus is `self.n_views`, the protocol's count, and was a module
        constant. It is an instance attribute rather than an argument so that
        the only way to obtain an index is from a pack that was told what the
        corpus holds.
        """
        from metafind.data.pointclouds import uid_seed

        return uid_seed(uid) % self.n_views

    def identity(self) -> dict:
        """What goes into the arm hash: which arms, and which bytes supplied them.

        The BYTES of the consumed arrays, not the manifest: `merge()` rewrites
        the manifest (written_at, code_revision) every time another shard is
        added, so hashing the manifest gave byte-identical dev_train arrays two
        different arm hashes before and after the dev_val shards were merged.
        Shard digests are computed once per pack construction.
        """
        shards = {}
        for arm in self.rows:
            digests = []
            for sh in self.manifest[arm]["shards"]:
                sha = sh.get("array_sha256")
                if not sha:
                    sha = hashlib.sha256(Path(sh["array"]).read_bytes()).hexdigest()
                digests.append({"tag": sh.get("tag"), "array_sha256": sha})
            shards[arm] = sorted(digests, key=lambda d: (d["tag"] or "", d["array_sha256"]))
        return {"arms": list(self.arms), "shards": shards,
                "image_rule": (self.manifest.get("image") or {}).get("rule")}


def protocol_n_views(encoding: dict) -> int:
    """How many views an asset holds, as the ENCODING PROTOCOL records it.

    [WAS `N_VIEWS_PER_ASSET = 11`, a module constant.] It was compared against
    what each cached sidecar records about ITSELF, and every one of the 45,692
    says 12, so `check_embedding_sidecars` refused the whole corpus and `main`
    raised SystemExit before the first batch -- 200 of 200 sampled assets
    mismatched. The guard's purpose was right and its comparison source was
    not: the view count is a protocol value. n05b resolves it, n06 stamps what
    it actually encoded into every sidecar, and the two get compared. A
    compile-time constant can be neither side of that comparison.

    The safety property the deleted constant's comment claimed is NOT dropped;
    it moves to `_query_side`, which still asserts the loaded `views` matrix
    against this number rather than trusting it, so a corpus re-rendered at
    another count cannot silently change which view a query draws.

    No default, on purpose. A protocol resolved before `view_aggregation`
    existed does not know its own view count, and substituting one here would
    restore exactly the defect this replaces.
    """
    try:
        n = encoding["view_aggregation"]["n_views"]
    except (KeyError, TypeError):
        raise SystemExit(
            "stage1_encoding_protocol.json carries no view_aggregation.n_views. "
            "Re-run n05b_resolve_stage1_encoding: the view count is a protocol "
            "value and this trainer will not substitute a constant for it."
        ) from None
    return int(n)


class Stage1Dataset:
    """One admitted asset: cached text/image vectors plus a LIVE point cloud.

    The asymmetry is the whole design. Text and image come from n06 because
    OpenCLIP is frozen; the point cloud is loaded raw because PointBERT is in
    the optimizer and a cached embedding would be stale after step one.

    [U-14] `aggregation` decides which image vector this returns, and until
    2026-08-27 it decided nothing: the constructor stored it and `__getitem__`
    read `cached["image"]` -- the pooled vector -- unconditionally. Setting the
    protocol to `random_single_view` therefore trained on the 12-view mean while
    the protocol recorded per-view sampling, with nothing raising. The value was
    a name in `stage1_config.PER_VIEW_AGGREGATIONS` with no consumer anywhere.

    This is NOT a change of protocol. `n05b` still resolves `mean`, and `mean`
    still returns the pooled vector byte for byte. What changed is that the other
    value now does what its name says instead of being silently ignored.
    """

    def __init__(self, uids: list[str], aggregation: str,
                 preload: bool = False,
                 query_pack: "QueryPack | None" = None) -> None:
        if aggregation not in PRECOMPUTABLE_AGGREGATIONS + PER_VIEW_AGGREGATIONS:
            raise ValueError(
                f"unknown image_aggregation {aggregation!r}; "
                f"stage1_config knows {PRECOMPUTABLE_AGGREGATIONS + PER_VIEW_AGGREGATIONS}")
        self.uids = uids
        self.aggregation = aggregation
        self.per_view = aggregation in PER_VIEW_AGGREGATIONS
        # `query_pack=None` is the pre-2026-08-31 construction, byte for byte:
        # no `q_*` key is emitted, `collate` produces the same three tensors,
        # and every call site takes the `query is gallery` branch. That is what
        # keeps `gallery_index.py`, `tools/measure_dtype_effect.py` and every
        # existing arm hash unaffected by this change.
        self.query_pack = query_pack
        # Checked HERE, against this dataset's own uid list, not once by the
        # caller: `encode_pools` builds two datasets over two different pools,
        # and a pack covering only one of them would otherwise be discovered on
        # the pool it covers and never on the pool it does not.
        if query_pack is not None:
            query_pack.require(uids)
        self.cache: dict[str, dict] | None = None
        if preload:
            self._preload()

    def _needs_views(self) -> bool:
        """`views` is 12x the pooled vector, so it is kept only when read."""
        return self.per_view or (self.query_pack is not None
                                 and "image" in self.query_pack.arms)

    def _query_side(self, uid: str, entry: dict) -> dict:
        """The `q_*` half of one item. Empty dict when query IS gallery.

        Each arm falls back to the GALLERY's own vector only when that arm is
        not active in the pack -- never when it is active and the asset is
        missing, which `QueryPack.require` has already made impossible.
        """
        pack = self.query_pack
        if pack is None:
            return {}
        out = {}
        if "text" in pack.arms:
            out["q_text"] = pack.vector("text", uid)
        if "image" in pack.arms:
            views = entry["views"]
            if views.shape[0] != pack.n_views:
                raise ValueError(
                    f"{uid} has {views.shape[0]} views, not {pack.n_views} as "
                    "stage1_encoding_protocol.view_aggregation.n_views says; "
                    "the held-out index is taken modulo that count, so a "
                    "re-rendered corpus would silently change which view every "
                    "query drew. Re-run n05b and n06 together, or use the "
                    "protocol the cache was built under.")
            out["q_image"] = np.asarray(views[pack.view_index(uid)],
                                        dtype=np.float32)
        if "pc" in pack.arms:
            out["q_pc"] = pack.vector("pc", uid)
        # Arms the pack does not supply emit no key at all, so those modalities
        # keep the gallery's observation. The active arm list travels into the
        # checkpoint's arm hash, so a partial pack is recorded as partial rather
        # than mistaken for a full one.
        return out

    def _preload(self) -> None:
        """Read every asset once, up front, and keep it in RAM.

        [KYZEN 2026-08-29] Not an optimisation -- a hypothesis about why this
        machine hard-resets. It survived four DAYS of VLM annotation and 45,692
        encodes, then died within twenty minutes of the first training run, and
        has now done so eight times. The difference is not average power:
        measured, annotation and training both sit near 550 W. It is that
        training opens 128 compressed files, decompresses them, and allocates
        and frees their buffers EVERY step -- ~240 file opens a second across
        four worker processes -- so the GPU alternates between waiting on I/O
        and running flat out twice a second, and the memory allocator never
        settles. Inference does neither.

        Preloading removes all of it: no per-step open, no decompress, no
        allocation churn, and `num_workers` drops to 0 so the worker processes
        stop existing. Whether that stops the resets is the experiment; it is
        not assumed.

        The DATA IS IDENTICAL -- same arrays, same dtypes, read from the same
        files -- so this cannot change a result. Only where the bytes are when
        the step asks for them.

        Cost: ~8.8 GB for the 31,985-asset dev pool (measured: 35 KB of
        embedding + 234 KB of cloud per asset, unpacked) against 52 GB free,
        and two to three minutes at the start.
        """
        import sys
        n = len(self.uids)
        print(f"preloading {n:,} assets into RAM "
              f"(no per-step file I/O, no dataloader workers)", flush=True)
        self.cache = {}
        for k, uid in enumerate(self.uids):
            cached = np.load(paths.EMBEDDINGS / f"{uid}.npz")
            cloud = np.load(paths.POINTCLOUDS / f"{uid}.npz")
            entry = {"text": cached["text"].astype(np.float32),
                     "image": cached["image"].astype(np.float32)}
            if self._needs_views():
                # Kept under per-view sampling OR the query pack's image arm:
                # `views` is 12x the size of the pooled vector, and under plain
                # `mean` with no pack nothing ever reads it.
                entry["views"] = cached["views"]
            xyz = cloud["xyz"].astype(np.float32)
            rgb = cloud["rgb"].astype(np.float32) if "rgb" in cloud else None
            entry["pc"] = xyz if rgb is None else np.concatenate([xyz, rgb], axis=1)
            self.cache[uid] = entry
            if (k + 1) % 5000 == 0:
                print(f"  {k + 1:,}/{n:,}", flush=True)
        gb = sum(a.nbytes for e in self.cache.values() for a in e.values()) / 1e9
        print(f"preloaded {n:,} assets, {gb:.1f} GB resident", flush=True)

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int) -> dict:
        uid = self.uids[i]
        if self.cache is not None:
            e = self.cache[uid]
            if self.per_view:
                v = e["views"]
                image = v[random.randrange(v.shape[0])]
            else:
                image = e["image"]
            # Copies, not views: the collate stacks these and torch would
            # otherwise alias the cache. A training step that wrote through one
            # of these would corrupt the asset for every later epoch, silently.
            item = {"uid": uid, "text": e["text"].copy(),
                    "image": np.asarray(image, dtype=np.float32).copy(),
                    "pc": e["pc"].copy()}
            item.update({k: np.asarray(v, dtype=np.float32).copy()
                         for k, v in self._query_side(uid, e).items()})
            return item
        cached = np.load(paths.EMBEDDINGS / f"{uid}.npz")
        cloud = np.load(paths.POINTCLOUDS / f"{uid}.npz")
        xyz = cloud["xyz"].astype(np.float32)
        rgb = cloud["rgb"].astype(np.float32) if "rgb" in cloud else None
        pc = xyz if rgb is None else np.concatenate([xyz, rgb], axis=1)
        if self.per_view:
            # [UPSTREAM ulip2 main.tex:612] "randomly sample its 2D rendered
            # image I ~ render(O)"; OpenShape method.tex:77 and ULIP-1
            # main.tex:236 do the same. A fresh view per step, not per asset --
            # `views` is the per-view matrix n06 stores for exactly this.
            #
            # `random` rather than a dataset-owned RNG because torch seeds each
            # dataloader worker's `random` from the generator passed to
            # DataLoader, so the draw is reproducible from `seed` for a given
            # worker count. It is NOT reproducible across a change in
            # `num_workers`; that is upstream's behaviour too and it is recorded
            # rather than hidden.
            #
            # AND `num_workers` IS NOT IN THE RECORDED HYPERPARAMETERS. It is a
            # literal at the DataLoader below and appears in neither
            # `resolve_stage1.DEFAULT_HYPERPARAMETERS` nor
            # `stage1_config.REQUIRED_HYPERPARAMETERS`. Under `mean` that costs
            # nothing -- nothing draws. Under `random_single_view` it decides
            # WHICH IMAGES THE MODEL SAW, so the honest statement is:
            #
            #     random_single_view is NOT reproducible from
            #     stage1_hyperparameters.json alone.
            #
            # Adding the field would change the canonical hash and is therefore
            # a protocol change, not a code change; it is raised rather than
            # taken here. Written down now because the moment it starts to
            # matter -- someone selecting per-view sampling -- is exactly the
            # moment nobody re-reads this comment.
            views = cached["views"]
            image = views[random.randrange(views.shape[0])]
        else:
            image = cached["image"]
        item = {
            "uid": uid,
            "text": cached["text"].astype(np.float32),
            "image": image.astype(np.float32),
            "pc": pc,
        }
        if self.query_pack is not None:
            entry = {"views": cached["views"]} if self._needs_views() else {}
            item.update({k: np.asarray(v, dtype=np.float32)
                         for k, v in self._query_side(uid, entry).items()})
        return item


def collate(batch: list[dict]):
    """Stack the gallery triple, and the query triple when there is one.

    The `q_*` keys are emitted only by a dataset carrying a query pack, so a
    batch either has all of its active query arms or none. Callers branch on
    presence rather than on a flag, which is what stops a pack from being
    configured and then not read -- there is no key to ignore.
    """
    import torch

    out = {
        "uid": [b["uid"] for b in batch],
        "text": torch.from_numpy(np.stack([b["text"] for b in batch])),
        "image": torch.from_numpy(np.stack([b["image"] for b in batch])),
        "pc": torch.from_numpy(np.stack([b["pc"] for b in batch])),
    }
    for key in ("q_text", "q_image", "q_pc"):
        if key in batch[0]:
            out[key] = torch.from_numpy(np.stack([b[key] for b in batch]))
    return out


def split_embeds(batch, backbone, device):
    """One batch -> (query embeds, gallery embeds).

    THE SEAM. Before 2026-08-31 all three call sites built a single `embeds`
    dict and passed it to both towers, so every invariant about query/gallery
    independence was supplied by the caller and guaranteed by nobody. It lives
    in one function now so the trainer and the two evaluators cannot drift:
    training on one construction and scoring on the other would make every
    number uninterpretable, and nothing would have raised.

    `gallery is query` by IDENTITY when no pack is present -- the same dict
    object, not an equal one -- so the point encoder runs once and the
    pre-2026-08-31 numerics are unchanged.

    With a pc arm the point encoder necessarily runs TWICE per step: the query's
    cloud is a different cloud. That is a real cost on the one encoder Stage 1
    trains, and it is the price of the arm rather than an oversight.
    """
    pc = backbone.encode_pc(batch["pc"].to(device))
    gallery = {"text": batch["text"].to(device),
               "image": batch["image"].to(device),
               "pc": pc}
    if not any(k in batch for k in ("q_text", "q_image", "q_pc")):
        return gallery, gallery
    query = dict(gallery)
    if "q_text" in batch:
        query["text"] = batch["q_text"].to(device)
    if "q_image" in batch:
        query["image"] = batch["q_image"].to(device)
    if "q_pc" in batch:
        query["pc"] = backbone.encode_pc(batch["q_pc"].to(device))
    return query, gallery


def trainable_state_dict(module) -> dict:
    """[L1-CKPT-TRAINABLE-ONLY] What training CHANGED: trainable parameters,
    plus the buffers of the modules that hold them.

    Keyed off ``requires_grad`` rather than off a name prefix: a prefix list
    goes stale the moment a module is renamed, and it would go stale silently --
    the checkpoint would still save, still load, and quietly omit a trained
    tensor.

    [Codex 2026-08-28, ratified by MASTER] The buffers half was missing and it
    is not a refinement. MEASURED:

        trainable_state_dict(nn.Sequential(nn.BatchNorm1d(4)))
            saved      ['0.bias', '0.weight']
            buffers    ['0.running_mean', '0.running_var', '0.num_batches_tracked']
            running_mean saved?  False

    Stage 1 puts `point_encoder` in train() (ulip_backbone.py:235) and PointBERT's
    stack holds BatchNorm1d (vendor/.../dvae.py:191-201), so those running
    statistics MOVE during training and were thrown away at every save. Reloading
    gave the trained weights sitting on the ORIGINAL ULIP-2 statistics -- and eval
    mode reads exactly those statistics, so the restored encoder is neither the
    one that trained nor the one that was scored. `gallery_index.py:90` hashes
    parameters only, so the mismatch was invisible to encoder identity too.

    This does not weaken the rule. The rule exists so the frozen half, which can
    be rebuilt byte-for-byte from upstream, is not copied into every checkpoint.
    A trained module's running statistics CANNOT be rebuilt from upstream -- they
    are what the training did. So the buffers are filtered by the same predicate
    as the parameters: a frozen submodule's buffers are still not saved.

    NOT DONE, and recorded because it is the other way to make this correct:
    putting BatchNorm in eval for the whole run would freeze the statistics and
    make "do not save them" right. That changes training dynamics, which is a
    research decision and Kyzen's, not this node's.
    """
    out = {name: p.detach().cpu()
           for name, p in module.named_parameters() if p.requires_grad}
    out.update({name: b.detach().cpu()
                for name, b in module.named_buffers()
                if _owner(name) in _submodules_with_trainable_params(module)})
    return out


def _owner(qualified_name: str) -> str:
    """The submodule path a `named_buffers()` key belongs to (`""` for the root)."""
    return qualified_name.rsplit(".", 1)[0] if "." in qualified_name else ""


def _submodules_with_trainable_params(module) -> set[str]:
    return {name for name, sub in module.named_modules()
            if any(p.requires_grad for p in sub.parameters(recurse=False))}


# The three modules Stage 1's optimizer touches. Naming them here rather than in
# save_checkpoint's body is what OPTIMIZER_COVERS_CHECKPOINT can assert against.
CKPT_SECTIONS = ("backbone_trainable_state", "tower_trainable_state",
                 "loss_trainable_state")


# [USER 2026-08-28] Kyzen: 「一 要」（每輪跑 dev-val）「二 對啊挑最好的」.
# The mechanism was already ratified: METAFIND_NOTEBOOK.md:435-440, DEVIATION D-3,
# Kyzen 2026-08-27 --
#
#     開發期在 80% training pool 內部切 dev-val，用它的 Mean R@1（跨模態平均，
#     平手用 Mean R@5）定下 lr、輪數與 checkpoint 政策；正式期設定鎖死、
#     從頭重訓完整 80%、不中途挑 checkpoint、最後才第一次打開 20% test 考一次。
#     20% test 全程不參與任何選擇。
#
# So the metric is not an invention here: Mean R@1 across the seven Table 1 query
# conditions, ties broken by Mean R@5. What this function adds is a consumer for
# a split that had none -- `splits.py` has written `dev_val` since 2026-08-27 and
# `grep dev_val metafind/train/stage1.py` returned nothing until today.
#
# THE PHASE MATTERS AND IT IS NOT COSMETIC. `--phase dev` trains on `dev_train`
# (70%) so that dev-val is genuinely held out. Validating on `dev_val` while
# training on the full `train` (80%) would score the model on assets it had just
# fitted -- dev_val is a SUBSET of train, so the contamination is total, not
# partial. `--phase final` trains on the full 80% and does not select at all,
# which is the other half of D-3.
@contextlib.contextmanager
def modules_in_eval(*modules):
    """Put every module in eval for the block, then RESTORE what was there.

    [MASTER 2026-08-28, REJECT] `evaluate_dev_val` called `model.eval()` and
    nothing else. Stage 1's train_scope is `point_encoder_and_fuser`, and
    `ulip_backbone.py:235` puts `self.model.point_encoder` into train() -- so
    the backbone stayed in TRAIN mode through the whole evaluation.

    MEASURED by MASTER on CPU, no files changed:

        DropPath with drop_prob > 0   17 modules, 0.0059 .. rising per block
        BatchNorm1d                    2 modules, momentum 0.1,
                                       track_running_stats=True, training=True
        one encode_pc inside torch.no_grad()
            -> running_mean CHANGED: [True, True]

    `no_grad()` stops gradients. It does NOT stop BatchNorm from updating its
    running statistics. So every dev-val pass wrote dev-val's statistics into
    the model and training then continued from there. D-3 exists so that dev-val
    DECIDES without being FITTED; that is the exact thing it was doing, and
    nothing would have said so -- the loss looks normal, the metric has a value,
    the tests are green. The DropPath randomness on top means the same
    checkpoint would score differently on two runs, which is a selection metric
    that partly selects on noise.

    Restoring rather than calling `.train()` afterwards, because unconditional
    `.train()` would flip a module that was already in eval when the caller
    handed it over -- the same class of bug one level down.
    """
    # [MASTER 2026-08-28, REJECT #2] EVERY submodule's flag, and restored by
    # assignment rather than by `.train()`. The first version of this recorded
    # only the roots and restored with `m.train(was)` -- and `nn.Module.train`
    # RECURSES, while the state it had to restore is split:
    #
    #     ulip_backbone.py:228  self.model.eval()                  -> False
    #     ulip_backbone.py:235  self.model.point_encoder.train()   -> True
    #
    # so restoring the root to False drove point_encoder to False with it.
    # MEASURED on a two-level toy: point_encoder goes in True and comes out
    # False, and nothing sets it back. That is worse than the bug being fixed --
    # the original leaked dev-val statistics into a buffer, this one would put
    # the point encoder in eval for the whole remaining run: BatchNorm on frozen
    # running stats, all 17 DropPath off, every forward and every gradient
    # different from the epoch before, with no signal anywhere.
    #
    # Assignment, not `.train(was)`, precisely because `.train()` recurses.
    previous = [(m, m.training)
                for root in modules if root is not None
                for m in root.modules()]
    try:
        for m, _ in previous:
            m.training = False
        yield
    finally:
        for m, was_training in previous:
            m.training = was_training


@dataclass(frozen=True)
class Stage1RunPaths:
    """Where ONE run writes. Frozen, so nothing can move it after training starts.

    [CODEX 2026-08-30, BLOCKER] Replaces module globals reassigned by an
    `apply_out_dir` call. The globals worked, but Codex refused them as the
    provenance seam and the reason holds: a mutable module-level destination is
    reachable from anywhere, and the failure it guards against is precisely one
    run's output landing where another run's belongs.

    `smoke` carries the `--limit` suffix rule that `best_paths` used to own:
    a 200-asset smoke must never write the canonical `stage1_best.pt`, because a
    smoke started after a development run would replace a multi-hour selection
    with one chosen over a 200-asset gallery and the file would look identical.
    """
    root: Path
    latest_checkpoint: Path
    latest_record: Path
    best_checkpoint: Path
    best_record: Path
    reservation: Path
    lock: Path
    lock_fd: int = -1

    def release(self) -> None:
        """Drop the live-run lock. FOR TESTS AND SHORT TOOLS ONLY.

        A training run must hold its lock until the process exits, which the
        kernel does for it -- so production never calls this. A test process
        runs many resolutions in one interpreter and would otherwise lock itself
        out of the paths it just claimed.
        """
        if self.lock_fd >= 0:
            with contextlib.suppress(OSError):
                os.close(self.lock_fd)
            if self.lock_fd in _HELD_LOCKS:
                _HELD_LOCKS.remove(self.lock_fd)

    def targets(self) -> tuple[Path, ...]:
        return (self.latest_checkpoint, self.latest_record,
                self.best_checkpoint, self.best_record)


def resolve_run_paths(out_dir: str | None, limit: int | None = None,
                      overwrite: bool = False) -> Stage1RunPaths:
    """Decide this run's four destinations, and refuse to start on top of a run.

    Three defects Codex confirmed in the first version, all of which survive a
    passing test suite and only show up as a lost run:

    * **`--out-dir` was optional and silent.** A sweep arm that forgot the flag
      wrote the canonical names, and the arm before it was gone. There is no
      flag discipline that fixes this; the guard has to be on the write.
    * **`mkdir(exist_ok=True)` accepted an occupied directory.** Two arms given
      the same name overwrote each other exactly as the shared canonical names
      did, one directory further down.
    * **Absolute paths and `..` escaped the checkpoint root.** A typo could
      write outside the data tree entirely.

    So: relative paths only, resolved under `paths.CHECKPOINTS` and re-checked
    after resolution (`..` survives string inspection, it does not survive
    `resolve()` plus `is_relative_to`), and FAIL CLOSED if any of the four
    targets already exists. `--overwrite` is the deliberate way past it, which
    is what `data/outputs/ladder/e5_RECOVERED/` needed and did not have: it
    holds 8 KB of metrics and no weights because a later run reached
    `stage1_best.pt` first, in silence.

    The check is here, before the first batch, not at the first save -- an hour
    of training that ends in a refusal to write is the same lost run.
    """
    root = paths.CHECKPOINTS
    if out_dir is not None:
        if Path(out_dir).is_absolute():
            raise SystemExit(
                f"--out-dir must be relative to {paths.CHECKPOINTS}; got an "
                f"absolute path {out_dir!r}")
        root = (paths.CHECKPOINTS / out_dir).resolve()
        if not root.is_relative_to(paths.CHECKPOINTS.resolve()):
            raise SystemExit(
                f"--out-dir {out_dir!r} resolves to {root}, outside "
                f"{paths.CHECKPOINTS}. Checkpoints do not leave the data root.")

    suffix = f".smoke{limit}" if limit else ""
    rp = Stage1RunPaths(
        root=root,
        latest_checkpoint=root / f"stage1{suffix}.pt",
        latest_record=root / f"stage1{suffix}_ckpt.json",
        best_checkpoint=root / f"stage1_best{suffix}.pt",
        best_record=root / f"stage1_best{suffix}_ckpt.json",
        reservation=root / f"stage1{suffix}_run.json",
        lock=root / f"stage1{suffix}_run.lock")

    if not overwrite:
        if occupied := [t for t in rp.targets() if t.exists()]:
            raise SystemExit(
                "refusing to start: " + ", ".join(str(t) for t in occupied)
                + " already exist(s). Give --out-dir a fresh name, or pass "
                  "--overwrite if losing that run is intended.")
    root.mkdir(parents=True, exist_ok=True)

    # [CODEX 2026-08-30, BLOCKER 2] The existence check above is not a
    # reservation. Codex demonstrated it: two processes both pass it -- nothing
    # is written until the end of epoch one, minutes later -- and then share the
    # same `.part` and the same final names. Checking is not claiming.
    #
    # `O_CREAT | O_EXCL` is the claim, and it is atomic in the kernel: exactly
    # one of any number of simultaneous callers creates the file, the rest get
    # FileExistsError. It also survives a crash, which the four-target check
    # cannot: a run killed before its first save leaves no checkpoint at all, so
    # the next run saw an empty directory and started on top of an attempt that
    # may still have been alive. This machine hard-reset nine times on
    # 2026-08-29, so that is not hypothetical here.
    #
    # Written at reservation time, before any training, so an interrupted
    # attempt still says what it was.
    # [CODEX MAJOR 2026-08-30] The previous attempt kept `O_EXCL` but let
    # `--overwrite` unlink the reservation first -- so two --overwrite processes
    # still both succeeded, which Codex demonstrated directly and my own test
    # missed because its second call did not pass `overwrite=True`. A file whose
    # existence is the lock cannot also be the thing `--overwrite` deletes.
    #
    # Two separate mechanisms now, because they answer two different questions:
    #
    #   stage1_run.lock   IS ANOTHER PROCESS RUNNING HERE RIGHT NOW?
    #       `flock(LOCK_EX | LOCK_NB)`. Held for the life of the process and
    #       released BY THE KERNEL when it dies -- including a hard reset, which
    #       this machine did nine times on 2026-08-29. A crashed run therefore
    #       does not block the retry, and `--overwrite` never touches it.
    #
    #   stage1_run.json   WHAT PRODUCED THE OUTPUTS ALREADY HERE?
    #       provenance, and one of the outputs `--overwrite` may replace.
    lock_fd = os.open(rp.lock, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(lock_fd)
        raise SystemExit(
            f"refusing to start: another process holds {rp.lock}. It is "
            f"training into this directory right now. Give --out-dir a fresh "
            f"name. (--overwrite replaces an OLD run's outputs; it never joins "
            f"a live one.)") from None
    # Kept alive for the process lifetime. A closed fd releases the lock, and
    # the fd is otherwise unreferenced the moment this function returns.
    _HELD_LOCKS.append(lock_fd)
    rp = replace(rp, lock_fd=lock_fd)

    if overwrite:
        rp.reservation.unlink(missing_ok=True)
    try:
        fd = os.open(rp.reservation,
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise SystemExit(
            f"refusing to start: {rp.reservation} exists, so another run has "
            f"claimed this directory (it may still be running, or it may have "
            f"died before its first checkpoint). Give --out-dir a fresh name, "
            f"or pass --overwrite if losing that run is intended.") from None
    with os.fdopen(fd, "w") as fh:
        json.dump({"run_id": runlog.run_id(), "argv": sys.argv[1:],
                   "started_at": time.time(),
                   "code_revision": runlog.code_revision()}, fh, indent=1)
        fh.flush()
        # fsync, because the claim's whole purpose is to survive a machine that
        # hard-resets -- which this one did nine times on 2026-08-29. A claim
        # sitting in the page cache when the power goes is not a claim.
        os.fsync(fh.fileno())
    return rp


def save_best(record: dict, epoch: int, scores: dict, args,
              train_uids: list, dev_val_uids: list,
              rp: Stage1RunPaths) -> None:
    """Copy the epoch's checkpoint aside, atomically, with what selected it.

    Temp-and-rename for BOTH files, matching `save_checkpoint`. `copyfile`
    straight onto the destination stood here: an interrupt mid-copy leaves a
    truncated `stage1_best.pt` that still has the right name, and the JSON
    beside it still describes the run that was interrupted.

    The record carries the phase, the limit, the two pool sizes and a digest of
    the two uid lists, because "best" is only meaningful against the pools it
    was chosen over -- a number without its denominators is the U-09 problem
    again, one directory down.
    """
    ckpt_path, record_path = rp.best_checkpoint, rp.best_record
    digest = hashlib.sha256(
        json.dumps([sorted(train_uids), sorted(dev_val_uids)]).encode()).hexdigest()

    tmp_ckpt = ckpt_path.with_suffix(ckpt_path.suffix + ".part")
    shutil.copyfile(rp.latest_checkpoint, tmp_ckpt)
    tmp_ckpt.replace(ckpt_path)

    payload = {**record, "uri": str(ckpt_path), "epoch": epoch,
               "selected_on": "dev_val mean_R@1, tie mean_R@5 [D-3]",
               "phase": args.phase, "limit": args.limit,
               "n_train": len(train_uids), "n_dev_val": len(dev_val_uids),
               "pools_sha256": digest, "dev_val": scores}
    tmp_rec = record_path.with_suffix(record_path.suffix + ".part")
    tmp_rec.write_text(json.dumps(payload, indent=1))
    tmp_rec.replace(record_path)


def better_checkpoint(candidate: tuple[float, float],
                      incumbent: tuple[float, float] | None) -> bool:
    """[D-3] Is `candidate` a better selection than `incumbent`?

    Both are ``(mean_R@1, mean_R@5)``. Strictly greater, so an epoch that ties
    on both keeps the EARLIER one -- otherwise "best" drifts forward through
    epochs that improved on nothing, and the reported best epoch stops meaning
    the epoch where the model was best.

    A separate function because the rule is the decision (Kyzen 2026-08-27) and
    the loop around it is not; inline, the only way to test it is to train.
    """
    return incumbent is None or candidate > incumbent


def flatten_condition_scores(scores: dict) -> dict:
    """The seven per-condition recalls as flat scalars, for the runlog.

    [ULIP2 ENGINEER 2026-08-30, approved by Kyzen] `evaluate_dev_val` computes
    `recall_at_k` for all seven Table 1 conditions and the caller then THREW
    THEM AWAY: `runlog.train_metrics` filtered `scores` to `isinstance(v, (int,
    float))`, and `out[cond]` is a dict. So every run recorded to date carries
    `mean_R@1` and no way to see which condition produced it.

    That is not cosmetic. `DL-047`'s open finding is that `text` R@1 DEGRADES
    under training (-15.3 pp at lr 2.5e-4 on protocol D) while the seven-
    condition mean rises, because the mean is diluted by four cells at their
    ceiling (`DL-044`: an untrained model scores `full` R@1 = 0.9989). A metric
    that hides its own components cannot show that.

    ZERO extra computation -- the values already exist in `scores`. Nothing
    about the evaluator, the gallery, `ks=(1, 5)`, the ranking or the selection
    rule is touched by this function; `key` in the training loop still reads
    `mean_R@1` and `mean_R@5` and nothing else.

    A function rather than a comprehension inline at the call site, for the
    reason `better_checkpoint` above gives: inside `main()` the only way to
    test it is to train.

    Naming: `cond_` prefix so that neither a prefix nor a suffix match can
    confuse a condition with the aggregate -- `mean_R@1` does not begin with
    `cond_`, and it is the only other key ending in `R@1`. The middle segment is
    the `QUERY_CONDITIONS` key verbatim, `+` included, so `cond_text+pc_R@5`
    names one cell of Table 1 and cannot name any other.

    `recall_at_k` also puts `n_query` and `n_gallery` in all seven dicts; they
    are NOT flattened. Every one carries the value the row already holds once as
    top-level `n_gallery`, so seven more copies would be noise, not denominators.
    The `R@` test is on the metric key, so a future `ks` follows automatically
    rather than raising KeyError at the end of epoch 0 of an eight-hour run.
    """
    return {f"cond_{cond}_{metric}": value
            for cond, per_k in scores.items() if isinstance(per_k, dict)
            for metric, value in per_k.items() if metric.startswith("R@")}


def evaluate_dev_val(backbone, model, dev_val_uids, aggregation, device,
                     batch_size, query_pack=None, num_workers: int = 4):
    """Mean R@1 / R@5 over the seven Table 1 conditions, on the dev-val gallery.

    Gallery is dev_val itself (`splits.build_eval_protocols`, protocol
    ``C_dev_selection``: query_split and gallery_split are both ``dev_val``).
    Ranking a dev-val query against the whole training pool would make the
    gallery size differ from every reported protocol and the number would not
    compare to anything.

    Returns ``{}`` when there is nothing to evaluate, so a caller can tell
    "no dev-val" from "dev-val scored 0.0" -- an empty gallery silently scoring
    zero is how a selection metric starts choosing checkpoints at random.

    **Scored in float64** [ULIP2 REVIEWER 2026-08-30, PASS]. Two reasons, one
    measured here and one measured in `run_retrieval`:

    * `run_retrieval` (n15) scores in float64 because at production shape the
      collapse diagnostic `tie_count` moved with the caller's block size in
      float32 (7-9 of 12 trials) and not at all in float64. Two evaluators with
      different numerical semantics is the thing this project already refused
      once, so this one follows.
    * The obvious objection -- "then the ladder numbers become incomparable" --
      was a guess, and it was measured and is **false, in the opposite
      direction**. `tools/measure_dtype_effect.py`, 2026-08-30, encoding
      dev_val's 4,569 assets ONCE with `ladder/e25_500w/stage1_best.pt` and
      scoring the same embeddings twice::

          condition     recorded run    rescored f32    rescored f64
          text          0.713066316     0.713066316     0.713066316
          image         0.945283432     0.945283432     0.945283432
          pc            0.896476253     0.896476253     0.896476253
          text+image    0.985335960     0.985335960     0.985335960
          text+pc       0.996498140     0.996498140     0.996498140
          image+pc      0.987743489     0.987962355     0.987743489
          full          1.000000000     1.000000000     1.000000000
          mean_R@1      0.932057656     0.932088922     0.932057656

          bit-exact agreement with the recorded run:
              float64  7/7 conditions        float32  6/7

      **float64 reproduces the recorded run exactly; float32 rescored in numpy
      does not**, differing by one query in 4,569 on `image+pc`. WHY the
      original torch-float32 run agrees with numpy-float64 and not with
      numpy-float32 has NOT been measured, and no mechanism is asserted here.

    If a future dev_val score ever disagrees with an earlier value at the same
    arm, this table is the first thing to re-read. The evidence is kept at
    `output/look/dtype_effect.json`.

    ⚠ Scope: one checkpoint, one split. It can find a disagreement; it cannot
    prove the two dtypes agree for every model.

    🎁 One thing this also settled, on the way past: `full` R@1 is
    **1.000000000 in float64 as well**, bit-exact. "The 1.0000 is float32
    rounding" is therefore **eliminated** -- the first candidate explanation for
    that cell to be closed by measurement rather than argument. The rest still
    need n15's negative controls.
    """
    import torch
    from torch.utils.data import DataLoader

    from metafind.eval.retrieval import (
        QUERY_CONDITIONS,
        condition_mask,
        normalize_for_scoring,
        recall_at_k,
    )

    if not dev_val_uids:
        return {}

    # shuffle=False and drop_last=False are load-bearing, not defaults: they are
    # what makes row i of the query stack and row i of the gallery stack the same
    # asset, which is what `targets = np.arange(...)` below relies on.
    #
    # `num_workers=4` is a HARDCODED LITERAL and is NOT in any protocol. It is
    # the same open item raised on 2026-08-27 for the training loader: adding it
    # to `stage1_hyperparameters.json` would change that artifact's hash, which
    # is a protocol change and not this node's to make. Recorded here so it is
    # not mistaken for a resolved value.
    # The SAME pack the training loader was given. Selecting a checkpoint on the
    # shared construction while training on the independent one would pick the
    # epoch that best exploits a leak the run exists to remove, and the two
    # numbers would look comparable. `main` passes one object to both.
    loader = DataLoader(Stage1Dataset(dev_val_uids, aggregation,
                                      query_pack=query_pack),
                        batch_size=batch_size, shuffle=False, collate_fn=collate,
                        num_workers=num_workers, drop_last=False)
    per_condition_q = {c: [] for c in QUERY_CONDITIONS}
    gallery = []
    # `backbone.model` as well as `model`: the tower is what `model.eval()`
    # reaches, and the point encoder lives on the backbone. See `modules_in_eval`.
    with modules_in_eval(model, getattr(backbone, "model", None)), torch.no_grad():
        for batch in loader:
            query_embeds, gallery_embeds = split_embeds(batch, backbone, device)
            n = gallery_embeds["text"].size(0)
            gallery.append(model.gallery(gallery_embeds).float().cpu())
            for cond in QUERY_CONDITIONS:
                mask = condition_mask(cond, n).to(device)
                per_condition_q[cond].append(
                    model.query(query_embeds, present=mask).float().cpu())

    # The shared helper is also used by n15 and the dtype measurement harness.
    # Keeping the normalisation as well as the GEMM in NumPy float64 is what
    # makes those three paths the same numerical experiment rather than three
    # implementations that happen to carry the same dtype label.
    g = normalize_for_scoring(torch.cat(gallery).numpy())
    # The loader is `shuffle=False` and `drop_last=False`, so row i of the query
    # stack and row i of the gallery stack are the same asset. That is what makes
    # arange the target column; it is stated rather than assumed because
    # `rank_of_target` exists precisely so callers do not assume the diagonal.
    targets = np.arange(g.shape[0])
    out = {}
    for cond, chunks in per_condition_q.items():
        q = normalize_for_scoring(torch.cat(chunks).numpy())
        sim = q @ g.T
        out[cond] = recall_at_k(sim, targets, ks=(1, 5))
    out["mean_R@1"] = float(np.mean([out[c]["R@1"] for c in QUERY_CONDITIONS]))
    out["mean_R@5"] = float(np.mean([out[c]["R@5"] for c in QUERY_CONDITIONS]))
    out["n_gallery"] = int(g.shape[0])
    return out


# Excluded from the arm on purpose, each for a different reason.
#   seed        -- two seeds are REPEATS of one treatment [R-33]; folding it in
#                  would make every repeat its own experiment and leave the
#                  paired-difference analysis nothing to pair.
#   preload,
#   num_workers,
#   device      -- [CODEX 2026-08-30] execution facts, not treatment. Under the
#                  resolved `mean` aggregation nothing in `__getitem__` draws, so
#                  the worker count cannot change what the model sees. THIS IS
#                  CONDITIONAL: under `random_single_view` `random` is seeded per
#                  worker and these become arm-effective. The assertion below
#                  fails if that protocol is ever resolved, rather than letting
#                  the exclusion quietly stop being true.
#                  These three are named here even though nothing merges them
#                  into `values` today: [ULIP2 REVIEWER 2026-08-30] the error
#                  message below sends the reader to ARM_EXCLUDED, and this
#                  batch established `values["learning_rate"] = args.lr` as the
#                  way to fold a flag in. The next `values["preload"] =
#                  args.preload` would enter every arm hash with nothing to stop
#                  it. Absent keys are harmless; an untrue declaration is not.
#   max_epochs  -- [ULIP2 REVIEWER 2026-08-30] the operator's approved CEILING,
#                  not a treatment. `stage1.py:1187` only WARNS when a run
#                  exceeds it and `resolve_stage1.py:304` states outright that
#                  "no production code reads max_epochs". Raising it from 250 to
#                  500 leaves training bit-identical -- and, while it sat in
#                  `values`, changed every arm hash. That is the mirror of the
#                  enumeration hole: under-inclusion gives two experiments one
#                  identity, over-inclusion gives one experiment two, and the
#                  second silently strips comparability from every arm already
#                  run.
ARM_EXCLUDED = ("seed", "preload", "num_workers", "device", "max_epochs")


# The two protocol artifacts are hashed WHOLE, minus these. Same discipline as
# ARM_EXCLUDED and for the same reason: an enumeration of what to include is a
# list of what someone remembered, and `allow_all_masked` is what it forgot --
# it reaches `sample_modality_mask` at stage1.py:1310 and decides whether an
# all-masked query can occur, and two runs differing in it shared one arm hash.
# [CODEX 2026-08-30, BLOCKER 1]
TRAINING_EXCLUDED = (
    "status", "decided_by", "decided_at",
    # Lifted out explicitly below as `train_scope`, so it must not ALSO enter
    # as `training.train_scope`: with `--train-scope` given, the key exists in
    # the dict and was hashed twice under two names, and a flag-less run (no
    # key) hashed it once -- one treatment, two arm hashes.
    "train_scope",
    # Recorded separately and verbatim as `base_hyperparameter_sha256`. Hashing
    # it here too would make the arm change whenever the artifact's digest does,
    # which is already covered and is not itself a treatment.
    "hyperparameter_config_hash",
)
ENCODING_EXCLUDED = (
    "status", "decided_by", "decided_at",
    # Prose ABOUT the paper's claim, not about what this run does. Rewording the
    # basis note would otherwise give one experiment a second identity -- the
    # over-inclusion failure, which strips comparability from every arm already
    # run. `actual_clip_train_scope` is what runs and IS hashed.
    "paper_clip_train_scope", "paper_clip_train_scope_basis",
    "paper_clip_train_scope_confidence",
)

# Fields the protocols DECLARE but the trainer does not branch on. Declaring a
# value the code cannot honour is the failure Codex named: a run could set
# `scheduler: linear`, change its arm hash, and still be annealed by
# `cosine_schedule`. Refusing is the only way the declaration stays true.
ENFORCED_SINGLETONS = {
    "values": {
        "optimizer": "adamw",   # torch.optim.AdamW is constructed unconditionally
        "scheduler": "cosine",  # cosine_schedule() is called unconditionally
    },
    "training": {
        "similarity": "cosine",  # normalize + matmul, no other branch exists
        # [CODEX BLOCKER 2026-08-30, RESOLVED 2026-09-01] The trainer used to
        # hardcode `train_scope="point_encoder_and_fuser"` when building the
        # backbone, so the protocol value was hashed into the arm and then
        # ignored -- a checkpoint could name an ablation the run did not
        # perform. Refusing was the right response to that. The trainer now
        # READS the resolved scope (see `scope = training.get(...)` before
        # `ULIPBackbone(...)`), so `fuser_only` and `point_encoder_and_fuser`
        # both do what they say and the singleton is lifted for them.
        #
        # `full` stays refused, and not for tidiness: it unfreezes ViT-bigG-14,
        # whose AdamW state alone is ~30 GB against a 32.6 GB card. RA-3 records
        # the attempt. Allowing it here would let a run start and die mid-epoch
        # having already written its recipe.
    },
    "encoding": {
        # [CODEX BLOCKER 2026-08-30] Stage 1 consumes text and image embeddings
        # that were computed once, offline, by a FROZEN ViT-bigG-14 (n06). No
        # value of this field can change that inside a Stage 1 run: the CLIP
        # towers are never in the optimizer and the cached vectors are read from
        # disk. `trainable` and `finetuned` were accepted and hashed.
        "actual_clip_train_scope": "frozen",
    },
}


def arm_config_hash(values: dict, training: dict, encoding: dict,
                    phase: str, query_construction: dict | None = None
                    ) -> tuple[str, dict]:
    """Identity of the EXPERIMENTAL CONDITION, and the resolved recipe behind it.

    [R-33 ratified by Kyzen; corrected twice by CODEX, 2026-08-30] The first
    version hashed eight enumerated fields. The second took `values` whole but
    still enumerated what it pulled out of the two protocols, and Codex found
    the field that enumeration had missed: `allow_all_masked`, which reaches
    `sample_modality_mask` and decides whether an all-masked query is legal.
    Two genuinely different maskings shared one arm hash.

    So all three sources are now taken WHOLE and the exclusions are enumerated
    instead, because those are the ones that need a stated reason. See
    `ARM_EXCLUDED`, `TRAINING_EXCLUDED`, `ENCODING_EXCLUDED`.

    Rules kept literally from R-33:

    * **Hash the merged effective values, never the override patch.** No `--lr`
      and `--lr 5e-4` train the same model and MUST hash equal.
    * **`seed` is excluded**, and lives in the run record.

    Returns the digest AND the dict it was taken over. A digest alone is
    unreadable provenance; the checkpoint stores both.
    """
    if encoding["image_aggregation"] != "mean":
        raise UnsupportedProtocol(
            f"image_aggregation is {encoding['image_aggregation']!r}, not "
            "'mean'. Under a per-view draw the dataloader worker count and "
            "--preload change what the model sees, so they stop being "
            "execution details and must enter the arm. See ARM_EXCLUDED.")
    # [CODEX 2026-08-30] A declared value the trainer ignores is worse than an
    # unrecorded one: the record asserts a recipe that did not run.
    # Checked per SOURCE, not against a merged dict: `train_scope` lives in the
    # training protocol and `actual_clip_train_scope` in the encoding protocol,
    # and a merged lookup silently misses whichever source it did not include --
    # which is how both of them stayed unchecked.
    for source, fields in ENFORCED_SINGLETONS.items():
        d = {"values": values, "training": training, "encoding": encoding}[source]
        for field, only in fields.items():
            got = d.get(field, only)
            if got != only:
                raise UnsupportedProtocol(
                    f"{source}.{field} is {got!r}; this trainer implements "
                    f"{only!r} only and does not branch on it. Running would "
                    f"record a recipe that did not happen. Implement the branch "
                    f"or correct the protocol.")

    # Not a singleton -- two of the three scopes now run -- but `full` must not
    # start. It unfreezes ViT-bigG-14, whose AdamW state alone is about 30 GB
    # against a 32.6 GB card, so the run would write its recipe and then die
    # partway through an epoch. Refusing before the first batch is the honest
    # failure. See RA-3 for the measurement.
    if training.get("train_scope") == "full":
        raise UnsupportedProtocol(
            "training.train_scope is 'full', which unfreezes ViT-bigG-14. Its "
            "optimizer state alone is ~30 GB against a 32.6 GB card (RA-3), so "
            "the run would fail mid-epoch after recording its recipe. Use "
            "'fuser_only' or 'point_encoder_and_fuser'.")
    # build_model honours exactly two values of this field: "zero_pad" sets
    # FusionConfig.zero_pad and anything else trains learned tokens. A protocol
    # naming a third reading would hash that reading into the arm and train
    # learned tokens -- a recorded recipe that did not happen.
    missing_repr = encoding.get("missing_modality_representation", "learned_token")
    if missing_repr not in ("learned_token", "zero_pad"):
        raise UnsupportedProtocol(
            f"encoding.missing_modality_representation is {missing_repr!r}; "
            "this trainer implements 'learned_token' and 'zero_pad' only.")

    resolved = {k: v for k, v in values.items() if k not in ARM_EXCLUDED}
    # Underscore-prefixed keys are THIS RUN's facts riding on the protocol dict
    # (see main); they are run record, not treatment -- except the two lifted
    # out explicitly below, which are command-line and belong to the arm.
    resolved.update({f"training.{k}": v for k, v in training.items()
                     if k not in TRAINING_EXCLUDED and not k.startswith("_")})
    resolved.update({f"encoding.{k}": v for k, v in encoding.items()
                     if k not in ENCODING_EXCLUDED})
    resolved.update({
        # Command-line only, so absent from every artifact and from
        # `base_hyperparameter_sha256`. This is the hole that hash had: e5, e10
        # and e25 all quoted one digest while training three different things.
        "epochs": training["_epoch_count"],
        "lr_horizon": training["_lr_horizon"],
        "phase": phase,
        "train_scope": training.get("train_scope", "point_encoder_and_fuser"),
    })
    # ADDED ONLY WHEN NON-DEFAULT, and that asymmetry is deliberate. Which
    # observation the query sees is unambiguously a TREATMENT -- it is the whole
    # of this change -- so a run using a pack must not share an arm with one
    # that does not. But adding the key unconditionally would alter the digest
    # of every arm already run, for runs whose behaviour did not change: the
    # over-inclusion failure this docstring already names, which "strips
    # comparability from every arm already run". Absence therefore MEANS the
    # pre-2026-08-31 shared construction, and means it for the runs that were
    # recorded before the key existed as well as for new ones.
    if query_construction:
        resolved["query_construction"] = query_construction
    blob = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest(), resolved


def pool_provenance(train_uids: list, selection_uids: list) -> dict:
    """Which assets this run trained on, in which ORDER, for BOTH phases.

    [CODEX MAJOR 4, 2026-08-30] `save_best` already hashed the sorted uid lists,
    but only into the dev-phase best record: a `--phase final` checkpoint --
    the one a paper number would come from -- carried no pool identity at all.

    Two digests, because they answer different questions and one of them was
    missing:

    * **sequence** -- the uid list in the order the DataLoader received it. The
      shuffle is a permutation of POSITIONS, so at a fixed seed a different
      input order produces a different sequence of batches, and therefore
      different in-batch negatives for the contrastive loss. Sorted membership
      cannot see that.
    * **set** -- sorted membership, which is what "the same pool" usually means
      and what makes two runs comparable regardless of enumeration order.

    Computed AFTER `--limit` is applied, so a smoke's digests describe the smoke.
    """
    def seq(u): return hashlib.sha256(json.dumps(u).encode()).hexdigest()
    def st(u): return hashlib.sha256(json.dumps(sorted(u)).encode()).hexdigest()
    return {"train_uid_sequence_sha256": seq(train_uids),
            "train_uid_set_sha256": st(train_uids),
            "selection_uid_sequence_sha256": seq(selection_uids),
            "selection_uid_set_sha256": st(selection_uids),
            "n_train": len(train_uids), "n_selection": len(selection_uids)}


def input_content_digest(uids: list[str]) -> dict:
    """What the uids POINTED AT, not merely which uids were used.

    [CODEX MAJOR 2026-08-30] `pool_provenance` proves which assets and in what
    order. It cannot prove what those assets contained: re-run n06 or n07 and
    the same uid list, at the same arm hash, reads different `.npz` bytes and
    nothing recorded would differ. That is the same failure as `code_revision`
    without `runtime_source_sha256`, one layer down in the data.

    Point-cloud digests are READ from the sidecars n07 already wrote
    (`data/outputs/pointclouds/<uid>.json:sha256`), so they cost no I/O over the
    7.7 GB of clouds. Embedding sidecars carry no digest, so those `.npz` are
    hashed -- 33 KB each, measured at ~3 s for the 32 k dev pool, against a run
    of hours.

    `n_missing` is reported rather than raised on: a missing sidecar is a real
    condition to record, and refusing to start a multi-hour run over one absent
    provenance file would be the guard doing more harm than the gap.
    """
    pc_dir, emb_dir = paths.POINTCLOUDS, paths.EMBEDDINGS
    h = hashlib.sha256()
    missing_pc = missing_emb = mismatched = 0
    for uid in sorted(uids):
        h.update(uid.encode())
        # [CODEX MAJOR 2026-08-30] The first version read the sidecar's claimed
        # sha256 and hashed THAT. A sidecar is a claim about a file, and a claim
        # is exactly what the rest of this batch stopped trusting -- an edited
        # `.npz` beside an untouched sidecar produced a complete-looking digest
        # that described the old bytes. The digest is now taken over the ACTUAL
        # bytes, and the sidecar's claim is checked against them.
        # MEASURED: 1,500 clouds in 0.2 s, so ~4 s for the 32 k dev pool.
        npz = pc_dir / f"{uid}.npz"
        if npz.exists():
            digest = hashlib.sha256(npz.read_bytes()).hexdigest()
            h.update(digest.encode())
            side = pc_dir / f"{uid}.json"
            if side.exists():
                claimed = json.loads(side.read_text()).get("sha256")
                if claimed and claimed != digest:
                    mismatched += 1
            else:
                missing_pc += 1
        else:
            missing_pc += 1
        npz = emb_dir / f"{uid}.npz"
        if npz.exists():
            h.update(hashlib.sha256(npz.read_bytes()).digest())
        else:
            missing_emb += 1
    return {"content_sha256": h.hexdigest(), "n_assets": len(uids),
            "n_missing_pointcloud_sidecar": missing_pc,
            "n_missing_embedding_npz": missing_emb,
            # Reported, not raised on: a run of hours should not be refused over
            # a stale sidecar, but a nonzero count here means n07's record and
            # its clouds have diverged and every claim about them is suspect.
            "n_pointcloud_sidecar_mismatch": mismatched,
            "pointcloud_digest_source": "sha256 of the .npz bytes; the n07 "
                                        "sidecar's claim is checked against it",
            "embedding_digest_source": "sha256 of the .npz bytes"}


def check_embedding_sidecars(uids: list[str], encoding: dict) -> None:
    """Refuse cached embeddings produced under a different encoding protocol.

    The trainer reads `embeddings/<uid>.npz` and hashes the encoding protocol
    into the arm; nothing compared the two. Re-resolving the text template or
    the image aggregation without re-running the encoder would train on the
    old vectors under the new arm keys, with a self-consistent and wrong
    record. Each sidecar names what produced its vectors; four fields are
    compared, as strings, because the sidecar stores them as strings.
    """
    from metafind.data.encode_text_image import ENCODER_VERSION

    want = {"text_serialization": str(encoding["text_serialization"]),
            "aggregation": str(encoding["image_aggregation"]),
            "encoder_version": str(ENCODER_VERSION),
            # The PROTOCOL's count, not a module constant. Comparing a
            # sidecar's self-report against a number compiled into the trainer
            # made this guard refuse the entire corpus while the cache and the
            # protocol agreed with each other.
            "n_views": str(protocol_n_views(encoding))}
    bad: dict[str, int] = {}
    missing = 0
    for uid in uids:
        side = paths.EMBEDDINGS / f"{uid}.json"
        if not side.exists():
            missing += 1
            continue
        rec = json.loads(side.read_text())
        for k, v in want.items():
            if str(rec.get(k)) != v:
                bad[k] = bad.get(k, 0) + 1
    if bad or missing:
        raise SystemExit(
            f"cached embeddings disagree with the encoding protocol: "
            f"{bad} sidecar field mismatches, {missing} sidecars missing, over "
            f"{len(uids):,} assets. Expected {want}. Re-run the encoder under "
            "this protocol, or restore the protocol the cache was built under.")


def initializer_provenance(backbone) -> dict:
    """The weights this run STARTED from, which no artifact recorded.

    [CODEX MAJOR 4, 2026-08-30] Two separate initializers reach Stage 1 and
    neither was written down:

    * the ULIP-2 PointBERT checkpoint, loaded from disk;
    * **OpenCLIP ViT-bigG-14 `laion2b_s39b_b160k`, which is NOT inside that
      checkpoint** -- `ulip_backbone.py:225` builds it through
      `open_clip.create_model_and_transforms` and its weights arrive from
      OpenCLIP's own cache. A reader holding our checkpoint and the ULIP-2
      checkpoint still could not reconstruct the run.

    Hashing the ULIP-2 file costs a few seconds once per run, against a run
    measured in hours.
    """
    out: dict = {}
    try:
        ckpt = Path(getattr(backbone.cfg, "checkpoint", "")) if hasattr(
            backbone, "cfg") else None
        if ckpt and ckpt.exists():
            out["ulip2"] = {"uri": str(ckpt),
                            "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                            "size_bytes": ckpt.stat().st_size}
    except Exception:  # noqa: BLE001 -- provenance is best-effort, never fatal
        out["ulip2"] = {"status": "unavailable"}
    try:
        import open_clip
        oc = {"model": "ViT-bigG-14", "pretrained": "laion2b_s39b_b160k",
              "package_version": getattr(open_clip, "__version__", None)}
        oc.update(_open_clip_weight_identity())
        out["open_clip"] = oc
    except Exception:  # noqa: BLE001
        out["open_clip"] = {"status": "unavailable"}
    return out


def _open_clip_weight_identity() -> dict:
    """The actual bytes behind `laion2b_s39b_b160k`, not just its name.

    [CODEX MAJOR 2026-08-30] A model name, a pretrained tag and a package
    version identify a REQUEST, not a file. The tag resolves through a Hugging
    Face repo whose `main` can move, so two runs could record identical
    initializer provenance and start from different weights.

    Read out of the HF cache rather than hashed: the snapshot entry is a symlink
    into `blobs/`, and for an LFS file the blob's FILENAME is its sha256 --
    which is why this costs a `readlink` instead of digesting 10.2 GB on every
    run. `refs/main` gives the resolved commit.

    `ULIP_models.py:354-355` is what actually downloads this
    (`create_model_and_transforms(..., pretrained='laion2b_s39b_b160k')`);
    `ulip_backbone.py:225` builds a second copy with `pretrained=None` purely to
    recover the preprocess transform, and contributes no weights.
    """
    import os as _os

    repo = "models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k"
    roots = [_os.environ.get("HF_HUB_CACHE"),
             (Path(_os.environ["HF_HOME"]) / "hub") if _os.environ.get("HF_HOME")
             else None,
             Path.home() / ".cache" / "huggingface" / "hub"]
    for root in roots:
        if not root:
            continue
        d = Path(root) / repo
        if not d.is_dir():
            continue
        out: dict = {"hf_repo": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                     "hf_cache": str(d)}
        ref = d / "refs" / "main"
        rev = ref.read_text().strip() if ref.exists() else None
        if rev:
            out["hf_revision"] = rev
        # [CODEX MINOR 2026-08-30] Was `glob("snapshots/*/*")`, which with more
        # than one cached revision would pair `refs/main`'s revision with some
        # OTHER snapshot's blob -- a provenance record whose two halves describe
        # different downloads. The snapshot for THIS revision, or nothing.
        snap_root = (d / "snapshots" / rev) if rev else None
        if snap_root is None or not snap_root.is_dir():
            out["weight_identity"] = (
                f"unavailable -- no snapshot directory for revision {rev!r}")
            return out
        for snap in sorted(snap_root.iterdir()):
            if snap.name.startswith("open_clip"):
                blob = snap.resolve()
                out["weight_file"] = snap.name
                # The blob name IS the git-lfs sha256 for an LFS file. Recorded
                # under a name that says where it came from, so nobody mistakes
                # it for something this code computed.
                out["weight_blob_sha256"] = blob.name
                if blob.exists():
                    out["weight_size_bytes"] = blob.stat().st_size
                break
        return out
    return {"weight_identity": "unavailable -- HF cache not found"}


def save_checkpoint(backbone, model, loss_fn, hyperparameters: dict,
                    encoding: dict, training: dict, seed: int, epoch: int,
                    rp: Stage1RunPaths) -> dict:
    """Save EVERY trainable module, not just the dual tower.

    The optimizer is built from three modules::

        backbone.trainable_parameters()   # PointBERT + pc_projection
        model.parameters()                # query/gallery fusion
        loss_fn.parameters()              # logit_scale, if learnable

    An earlier version passed only ``model`` here. Everything still ran: the
    save succeeded, the reload succeeded, the shapes were right, and the
    fine-tuned PointBERT was silently discarded at the end of every epoch --
    downstream code rebuilt the backbone from the ORIGINAL ULIP-2 weights, so
    Stage 1's point-tower training had no effect on anything that followed. It
    is the same failure shape as the checkpoint-bloat and placeholder-tensor
    bugs: correct-looking artifact, no error, wrong contents.

    ``assert_checkpoint_covers_optimizer`` below is the check that would have
    caught it, and it runs on every save.
    """
    import torch

    paths.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    sections = {
        "backbone_trainable_state": trainable_state_dict(backbone.model),
        "tower_trainable_state": trainable_state_dict(model),
        "loss_trainable_state": trainable_state_dict(loss_fn),
    }
    assert_checkpoint_covers_optimizer(backbone, model, loss_fn, sections)
    n_params = sum(v.numel() for s in sections.values() for v in s.values())

    tmp = rp.latest_checkpoint.with_suffix(".part")
    # [CODEX MAJOR 4, 2026-08-30] The metadata is built BEFORE the save and
    # embedded in the `.pt` itself, not only written to the sidecar. Weights and
    # sidecar are two files: one gets copied, renamed, or moved without the
    # other, and until now the weights alone could not answer which run produced
    # them. The `.pt` deliberately does NOT carry its own output sha256 -- a
    # file cannot contain its own digest -- so that stays in the sidecar, which
    # is also what binds the two.
    metadata = {
        # [CODEX D 2026-08-30] Was `config_hash`. It and
        # `base_hyperparameter_sha256` are the SAME value, and two hashes side
        # by side invited a reader to treat one of them as the experiment's
        # identity, which neither is -- `arm_config_hash` below is.
        "base_hyperparameter_sha256": training["hyperparameter_config_hash"],
        "checkpoint_schema": 4,
        "seed": seed,
        "epoch": epoch,
        # [ULIP2 ENGINEER 2026-08-29, approved by MASTER as a bug fix] This
        # record named a config_hash and a seed and no code state at all, so
        # `stage1_best.pt` from the 06:17 dev run -- produced by a working tree
        # carrying gradient checkpointing that does not exist at HEAD fdfd6a8 --
        # could not be tied to any commit. `code_dirty` is carried beside the
        # revision because the revision ALONE is the false claim: it names a
        # commit at which this checkpoint could not have been produced (batch 64
        # OOMs without checkpointing). `run_id` matches the value stamped on the
        # `train_stage1.jsonl` rows, so a checkpoint and its loss curve can be
        # joined even though six runs share that file.
        "run_id": runlog.run_id(),
        "code_revision": runlog.code_revision(),
        "code_dirty": runlog.code_dirty(),
        # [R-33] `code_revision` + `code_dirty` was measured insufficient: it
        # called e25_400w and e25_500w a clean repeat when the tree had been
        # edited between them. See runlog.dirty_patch_sha256.
        "runtime_source_sha256": runlog.runtime_source_sha256(),
        "runtime_source_status": runlog.runtime_source_status(),
        # [R-33] The experiment's identity. `config_hash` above names the
        # ARTIFACT the run started from; this names what it actually trained,
        # after --lr / --epochs / --lr-horizon. Resolved values stored beside
        # the digest so the run is readable without recomputing it.
        "arm_config_hash": training["_arm_config_hash"],
        "arm_config": training["_arm_config"],
        # [CODEX MAJOR 2026-08-30] R-33's run record asked for these and the
        # first version shipped without them. `repeat_index` is what makes two
        # runs a declared pair rather than two runs that happen to share an arm;
        # `argv` is the only field that survives a reader who does not trust any
        # of the others.
        "repeat_index": training.get("_repeat_index"),
        "argv": training.get("_argv"),
        "hardware": training.get("_hardware"),
        # [KYZEN 2026-08-29] Which ladder construction produced this checkpoint.
        # Equal to `epoch_count` -> the cosine curve finished inside this run;
        # larger -> the run is a PREFIX of a longer curve and is comparable with
        # other rungs sharing the horizon. Without it, e5 and e10 look like the
        # same experiment at two lengths, which is what they were reported as.
        # `.get` with a None default: runs before this field existed are
        # UNKNOWN, and that is different from "the horizon equalled the epochs".
        "lr_horizon": training.get("_lr_horizon"),
        "epoch_count": training.get("_epoch_count"),
        # `num_workers` was a hardcoded 4 and appeared in NO artifact, so no run
        # before today recorded how many processes fed it. Under `mean` that
        # costs nothing scientifically, but it is the variable this machine's
        # crash investigation now turns on, and it was invisible.
        "preload": training.get("_preload"),
        "num_workers": training.get("_num_workers"),
        # `null` is the pre-2026-08-31 construction, where the query read the
        # gallery's own cached vectors. Recorded on the checkpoint as well as in
        # the arm hash so a loose .pt can still answer which construction
        # trained it, without the arm dict to dereference.
        "query_construction": training.get("_query_construction"),
        "query_observation": training.get("_query_observation"),
        "train_scope": training.get("train_scope", "point_encoder_and_fuser"),
        "trainable_only": True,
        "n_params_saved": int(n_params),
        "n_params_by_section": {k: int(sum(v.numel() for v in s.values()))
                                for k, s in sections.items()},
        "clip_train_scope": encoding["actual_clip_train_scope"],
        # [CODEX MAJOR 4] Which assets, in which order, and which weights this
        # run started from. `pool_provenance` covers a `--phase final` run,
        # which previously recorded no pool at all; `initializer_provenance`
        # covers OpenCLIP, which is loaded separately from the ULIP-2
        # checkpoint and appeared in no artifact.
        "inputs": training.get("_pools"),
        "initializers": training.get("_initializers"),
    }

    torch.save({**sections,
                "trainer_version": TRAINER_VERSION,
                "epoch": epoch,
                "metadata": metadata,
                "train_scope": training.get("train_scope", "point_encoder_and_fuser")},
               tmp)
    tmp.replace(rp.latest_checkpoint)

    record = {**metadata,
              "uri": str(rp.latest_checkpoint),
              "sha256": hashlib.sha256(rp.latest_checkpoint.read_bytes()).hexdigest(),
              "size_bytes": rp.latest_checkpoint.stat().st_size}
    tmp = rp.latest_record.with_suffix(".json.part")
    with tmp.open("w") as fh:
        json.dump(record, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(rp.latest_record)
    return record


def assert_checkpoint_covers_optimizer(backbone, model, loss_fn, sections: dict) -> None:
    """Every tensor the optimizer moves must appear in the checkpoint.

    Compares by IDENTITY, not by name or count. A name comparison would pass if
    two modules happened to expose the same key, and a count comparison would
    pass if one tensor were swapped for another of equal size.
    """
    in_opt = {id(p) for p in
              list(backbone.trainable_parameters()) + list(model.parameters())
              + list(loss_fn.parameters()) if p.requires_grad}
    if not in_opt:
        raise RuntimeError("the optimizer has no trainable parameters at all")

    # Compared against WHAT IS IN `sections`, not against what the modules
    # happen to expose. An earlier version rebuilt `saved` by walking the same
    # three modules again, which proves only that the optimizer's tensors live
    # in modules we intend to serialise -- a tautology, since `sections` was
    # built from those very modules. It would have passed even if
    # trainable_state_dict had silently dropped a key.
    #
    # Names, not identities, on this side: a state dict holds detached CPU
    # copies, so `id()` cannot match across the copy. The names come from the
    # same `named_parameters()` call that produced the dict, so a rename cannot
    # desynchronise them either.
    saved_names = {f"{sec}:{n}" for sec, state in sections.items() for n in state}
    expected = set()
    for sec, module in (("backbone_trainable_state", backbone.model),
                        ("tower_trainable_state", model),
                        ("loss_trainable_state", loss_fn)):
        expected |= {f"{sec}:{n}" for n, p in module.named_parameters()
                     if p.requires_grad and id(p) in in_opt}
        # The buffers of every module that owns a trainable parameter, on the
        # same footing as the parameters: dropping `running_mean` is as silent
        # as dropping a weight and, unlike a weight, nothing downstream notices.
        trainable_mods = _submodules_with_trainable_params(module)
        expected |= {f"{sec}:{n}" for n, _ in module.named_buffers()
                     if _owner(n) in trainable_mods}
    if missing := expected - saved_names:
        raise RuntimeError(
            f"{len(missing)} parameter(s) are in the optimizer but absent from "
            f"the checkpoint sections, e.g. {sorted(missing)[:3]}; Stage 1 would "
            "train them and throw the result away")

    covered = {id(p) for _, module in (("b", backbone.model), ("t", model),
                                       ("l", loss_fn))
               for p in module.parameters() if p.requires_grad}
    if orphaned := in_opt - covered:
        raise RuntimeError(
            f"{len(orphaned)} parameter(s) are in the optimizer but belong to no "
            f"module the checkpoint serialises {CKPT_SECTIONS}")


def load_stage1_checkpoint(backbone, model, loss_fn, path=None,
                           new_prefixes: tuple[str, ...] = ()) -> dict:
    """Restore all three trainable modules, refusing anything partial.

    ``strict=False`` is required because each section holds only the trainable
    subset -- but the missing keys are then CHECKED against what should have
    been there, so "legitimately new" and "silently dropped" stay distinguished.

    Args:
        new_prefixes: parameter-name prefixes the caller KNOWS this checkpoint
            cannot contain. Stage 2 passes the ESSGNN and Eq. 6's lambda, which
            do not exist in Stage 1 by design (2.6 introduces them in Stage 2).
            Anything trainable and not covered by a declared prefix is an error,
            so the escape hatch has to be opened deliberately and per name --
            a blanket `strict=False` is what let the point encoder vanish.
    """
    import torch

    path = path or CKPT_PATH
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if absent := [s for s in CKPT_SECTIONS if s not in ckpt]:
        raise ValueError(
            f"{path} is missing {absent}. Checkpoints written before "
            f"trainer_version {TRAINER_VERSION} stored only the dual tower and "
            "silently dropped the fine-tuned point encoder; they cannot be "
            "upgraded, only retrained.")

    for section, module in (("backbone_trainable_state", backbone.model),
                            ("tower_trainable_state", model),
                            ("loss_trainable_state", loss_fn)):
        state = ckpt[section]
        _, unexpected = module.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"{section} holds keys {module} does not have: "
                             f"{sorted(unexpected)[:5]}")
        trainable = {n for n, p in module.named_parameters() if p.requires_grad}
        # The buffers of every submodule that has a trainable parameter are
        # saved (see trainable_state_dict: BatchNorm running statistics MOVE
        # during training) and must be restored too. A checkpoint from before
        # that fix lacks them and used to load silently, leaving trained
        # weights on ULIP-2's original statistics.
        owners = _submodules_with_trainable_params(module)
        expected = trainable | {n for n, _ in module.named_buffers()
                                if _owner(n) in owners}
        gap = {n for n in expected - set(state)
               if not n.startswith(new_prefixes)} if new_prefixes else \
            expected - set(state)
        if gap:
            raise ValueError(
                f"{section} does not cover {len(gap)} trainable parameter(s), "
                f"e.g. {sorted(gap)[:3]} -- restoring would leave them at their "
                "freshly-initialised values")
    return ckpt


def build_model(encoding: dict, training: dict, hyperparameters: dict):
    import torch
    from metafind.models.dual_tower import DualTowerConfig, MetaFindDualTower
    from metafind.models.fusion import FusionConfig
    from metafind.models.losses import ContrastiveConfig, MetaFindContrastiveLoss
    from metafind.models.ulip_backbone import EMBED_DIM

    values = hyperparameters["values"]
    # [U-11] 2.6 rules out zero-padding and names no replacement; n05b chose a
    # learned token. FusionConfig expresses that as `zero_pad`, so the mapping
    # is made here rather than letting a dataclass default decide -- Table 3's
    # "Padding missing modalities with 0" is the row that sets it True.
    zero_pad = encoding["missing_modality_representation"] == "zero_pad"
    fusion = FusionConfig(kind=training["fusion"], dim=EMBED_DIM, zero_pad=zero_pad)
    model = MetaFindDualTower(DualTowerConfig(
        dim=EMBED_DIM, tower_sharing=training["tower_sharing"],
        query_fusion=fusion, gallery_fusion=fusion,
        # [PAPER 2.6] Stage 1 is object-level pretraining "without spatial
        # context"; the ESSGNN branch and Eq. 6's lambda term belong to Stage 2.
        # Building it here would put an untrained layout module in the optimizer
        # and in the checkpoint.
        use_layout=False))
    loss = MetaFindContrastiveLoss(ContrastiveConfig(
        # [L1-LOSS-STAGE1-UNIDIRECTIONAL] Eq. 5 is query->gallery only; the
        # symmetric form is Stage 2's Eq. 7a/7b.
        bidirectional=False,
        learnable_temperature=values["learnable_temperature"],
        init_temperature=values["init_temperature"],
        max_logit_scale=values["max_logit_scale"]))
    return model, loss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, help="override the recorded value")
    ap.add_argument("--preload", action="store_true",
                    help="read every asset into RAM once instead of opening two "
                         ".npz per sample per step; sets num_workers=0. Same data, "
                         "same results; ~8.8 GB for the dev pool.")
    ap.add_argument("--lr-horizon", type=int, default=None,
                    help="epochs the cosine curve spans; training still stops at "
                         "--epochs. Omit to anneal fully within --epochs (the "
                         "behaviour of every run before 2026-08-29).")
    # [R-33, ratified by Kyzen 2026-08-29] The three values a comparison needs
    # to vary and could not. `learning_rate` and `seed` live in a signed
    # artifact, so varying them meant editing `stage1_hyperparameters.json` --
    # rewriting the ratified recipe to run one arm of a sweep. These override
    # for THIS RUN only; the artifact is never touched, its sha256 is recorded
    # as `base_hyperparameter_sha256`, and what actually ran is recorded as
    # `arm_config_hash` + `arm_config`.
    ap.add_argument("--lr", type=float, default=None,
                    help="override learning_rate for this run. The artifact is "
                         "not modified; the effective value goes into "
                         "arm_config_hash.")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the seed for this run. NOT part of "
                         "arm_config_hash -- two seeds are repeats of one "
                         "treatment, not two experiments.")
    # [Kyzen 2026-09-01] Which parameters train was reachable only by editing
    # the training protocol, so all 73 runs to date used the recorded
    # `point_encoder_and_fuser` and `fuser_only` was never once exercised. It is
    # Table 3's "Train fuser only" row (8.7 against the full setting's 11.4) and
    # it is also the only scope under which PointBERT's output norm cannot move
    # -- measured 27.9 -> 200.5 after training, against text and image frozen at
    # 37 and 40, which is what lets the point cloud dominate the unweighted
    # readout. UNLIKE --seed this DOES enter arm_config_hash, because a
    # different set of trainable parameters is a different treatment.
    ap.add_argument("--train-scope", default=None,
                    choices=("fuser_only", "point_encoder_and_fuser", "full"),
                    help="override the training protocol's train_scope for this "
                         "run. The artifact is not modified; the effective value "
                         "is what enters arm_config_hash and the checkpoint "
                         "record. `full` also unfreezes ViT-bigG-14.")
    ap.add_argument("--out-dir", default=None,
                    help="directory for this run's checkpoints, RELATIVE to "
                         "data/outputs/checkpoints. Omit and the run writes the "
                         "canonical names. Either way the run refuses to start "
                         "on top of existing checkpoints.")
    ap.add_argument("--overwrite", action="store_true",
                    help="permit writing over checkpoints that already exist. "
                         "Without it an occupied destination stops the run "
                         "before the first batch.")
    ap.add_argument("--repeat-index", type=int, default=None,
                    help="which repeat of this arm this run is. Recorded, never "
                         "hashed: repeats share an arm by definition.")
    ap.add_argument("--query-pack", default=None,
                    help="query_pack.json from tools/make_query_pack.py. The "
                         "query side then trains on a SECOND observation of "
                         "each asset (alternate caption, one held-out view, a "
                         "second point sample) instead of the gallery's own "
                         "cached vectors. Enters arm_config_hash. Omit for the "
                         "pre-2026-08-31 construction; the gallery is unchanged "
                         "either way.")
    # Which observation the QUERY side trains on is a research axis, and it
    # used to be decided by whether --query-pack happened to be present: no
    # pack, and both towers silently read the same cached vectors of the same
    # asset (the identity that let a zero-parameter control score 99.56). It
    # must now be declared, and the declaration is checked against what the
    # dataset will actually feed:
    #   same_record         both towers read the asset's own canonical text,
    #                       12-view mean image and canonical point cloud. This
    #                       is the paper's literal Stage 1 (section 2.7: one
    #                       full-modality record per asset, masking only).
    #                       Reproduction runs only; the self-match leak is
    #                       audited, not hidden.
    #   second_observation  the query side reads a second observation of each
    #                       asset from --query-pack (another caption, one
    #                       held-out view, a second point sample) while the
    #                       gallery keeps the canonical record. Required for
    #                       every run whose numbers will be compared.
    ap.add_argument("--query-observation", required=True,
                    choices=("same_record", "second_observation"),
                    help="what the query tower sees during training; "
                         "second_observation requires --query-pack, "
                         "same_record forbids it. Recorded in the checkpoint.")
    ap.add_argument("--limit", type=int, help="assets, for a smoke run")
    ap.add_argument("--device", default="cuda")
    # [D-3] `dev` is the development phase: train on dev_train, score dev_val
    # every epoch, keep the best. `final` is the locked run: train on the whole
    # 80%, no scoring, no selection -- the epoch count is already decided by
    # then, and picking mid-run there would put a selection back on a pool the
    # protocol says is not available for one.
    ap.add_argument("--phase", choices=("dev", "final"), default="dev",
                    help="dev: train dev_train, select on dev_val (D-3). "
                         "final: train the full 80%%, no selection.")
    args = ap.parse_args()
    # Checked here, before protocols, splits or CUDA: an argument that can never
    # produce a run should cost a second, not a dataset load. The resolution of
    # `epochs` still happens below, where the artifact's value is available.
    if args.epochs is not None and args.epochs <= 0:
        raise SystemExit(f"--epochs {args.epochs} must be positive")

    import torch
    from torch.utils.data import DataLoader
    from metafind.models.fusion import sample_modality_mask
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

    encoding, training, hyperparameters = load_protocols()
    # Two ULIP-2 backbones (one per tower) is a registered protocol value that
    # this trainer does not implement: stage1_config builds a second
    # BackboneConfig for it, but main below constructs ONE ULIPBackbone and
    # routes both towers' point clouds through it. Until the second backbone
    # is wired in, a protocol asking for it must stop here, not train the
    # shared reading under the other name.
    if training["tower_sharing"] == "fully_separate":
        raise SystemExit(
            "stage1_protocol.tower_sharing = 'fully_separate' asks for two "
            "ULIP-2 backbones, and this trainer builds one. The run would "
            "silently train the shared-backbone reading and record the other. "
            "Either set tower_sharing to 'shared_backbone_separate_fusion' "
            "(Figure 1 prints 'ULIP-2 (Shared)'), or implement the second "
            "backbone first.")
    # A COPY. `hyperparameters` stays exactly as loaded so its recorded sha256
    # keeps describing the file on disk; the overrides below apply to this run's
    # working values only. Mutating in place would leave the artifact's digest
    # attached to values the artifact does not contain.
    values = copy.deepcopy(hyperparameters["values"])
    if args.lr is not None:
        values["learning_rate"] = args.lr
    if args.seed is not None:
        values["seed"] = args.seed
    # `training` is the in-memory protocol dict; the file on disk is untouched.
    # Both places that read the scope -- `resolved["train_scope"]` (which feeds
    # arm_config_hash) and the checkpoint record -- take it from here, so
    # overriding once is enough and cannot leave the two disagreeing.
    if args.train_scope is not None:
        training["train_scope"] = args.train_scope
    # Refused HERE, before the 9.5 GB backbone is built and the pool digested;
    # arm_config_hash refuses it too, but by then minutes have been spent and
    # the run directory has been claimed.
    if training.get("train_scope") == "full":
        raise SystemExit("train_scope 'full' unfreezes ViT-bigG-14, whose AdamW "
                         "state alone exceeds this card. Use 'fuser_only' or "
                         "'point_encoder_and_fuser'.")
    # [CODEX F 2026-08-30] No arbitrary upper bound -- a ceiling on lr would be
    # an invented hyperparameter. These three are internal consistency: a
    # non-finite base rate poisons the whole cosine array, and a base below
    # either endpoint inverts the schedule into a curve that WARMS UP to the
    # floor and anneals upward, which trains happily and looks like a bad
    # learning rate rather than a broken one.
    lr = values["learning_rate"]
    if not math.isfinite(lr) or lr <= 0:
        raise SystemExit(f"learning_rate {lr} must be finite and positive")
    if lr < values["lr_end"] or lr < values["lr_start"]:
        raise SystemExit(
            f"learning_rate {lr} is below lr_end {values['lr_end']} or "
            f"lr_start {values['lr_start']}: the cosine schedule would run "
            "backwards. Check --lr.")
    run_paths = resolve_run_paths(args.out_dir, args.limit, args.overwrite)
    # Snapshot the program NOW, before any training, so a worktree edited at
    # hour two cannot change what this run reports it ran.
    #
    # [CODEX MINOR 2026-08-30] Fail closed rather than record a null. A run of
    # this cost whose produced-by-what cannot be answered is a run that will be
    # argued about later and cannot be settled -- exactly the position the
    # withdrawn e25 noise floor left us in. Nothing normal reaches this: the
    # package is beside the module doing the asking.
    if runlog.runtime_source_status() != "ok":
        raise SystemExit(
            "cannot fingerprint the source tree under "
            f"{paths.REPO / 'metafind'}, so this run could not say what code "
            "produced it. Refusing to start.")

    splits_path = paths.OUTPUTS / "splits.json"
    if not splits_path.exists():
        print(f"{splits_path} not found -- run n09_build_splits first", flush=True)
        return 2
    splits = json.loads(splits_path.read_text())["object"]
    # [D-3] In the development phase the training pool is dev_train, NOT train.
    # dev_val is a subset of train (`splits.split_dev`), so training on `train`
    # and scoring `dev_val` would score the model on assets it had just fitted.
    if args.phase == "dev":
        # No fallback to `train` here, and the test above exists to keep it out:
        # `... or splits["train"]` stood here for one revision, and it would have
        # trained on the full 80% while still scoring dev_val -- dev_val is a
        # subset of train, so that is not a degraded run, it is a meaningless
        # one, and nothing would have said so. Missing dev_train is a reason to
        # stop, not a reason to substitute the one pool D-3 forbids.
        missing = [k for k in ("dev_train", "dev_val") if not splits.get(k)]
        if missing:
            raise ValueError(
                f"--phase dev needs {' and '.join(missing)} in splits.json. "
                "Re-run n09_build_splits, or use --phase final, which selects "
                "nothing and needs no held-out pool.")
        train_uids = splits["dev_train"]
        dev_val_uids = splits["dev_val"]
        # [Codex 2026-08-28] `splits.py:292` enforces disjointness when it
        # WRITES the file. This node reads it, and a stale, hand-repaired or
        # truncated splits.json satisfies every check above while overlapping.
        # The AST test cannot see this -- it forbids reading `splits["train"]`,
        # not an overlap that arrives inside `dev_train`. The invariant is cheap
        # here and the failure it prevents is silent contamination of every
        # checkpoint decision, so it is checked where it is used and not only
        # where it is produced.
        overlap = set(train_uids) & set(dev_val_uids)
        if overlap:
            raise ValueError(
                f"{len(overlap):,} uid(s) are in BOTH dev_train and dev_val, "
                f"e.g. {sorted(overlap)[:3]}. Selection on dev_val would then "
                "score assets the run fitted. Re-run n09_build_splits.")
    else:
        train_uids = splits["train"]
        dev_val_uids = []
    # Built HERE, before `--limit` and before `_pools`, because it CHANGES THE
    # POOL and the recorded digests have to describe what actually ran.
    query_pack = (QueryPack(args.query_pack, protocol_n_views(encoding))
                  if args.query_pack else None)
    # The declared observation and the dataset's construction must agree, or
    # the checkpoint would record one thing and the towers would have seen
    # another. Checked before any data is loaded.
    if args.query_observation == "second_observation" and query_pack is None:
        raise SystemExit("--query-observation second_observation needs "
                         "--query-pack; without one the query tower reads the "
                         "gallery's own record.")
    if args.query_observation == "same_record" and query_pack is not None:
        raise SystemExit("--query-observation same_record was declared but "
                         "--query-pack was given; the query tower would read "
                         "a second observation. Declare second_observation "
                         "or drop the pack.")
    if args.query_observation == "same_record":
        print("query observation: SAME RECORD on both towers (paper-literal "
              "Stage 1; the query text/image/pc are the gallery's own cached "
              "vectors, so the self-match leak is present and must be "
              "audited, not compared).", flush=True)
    dropped = {"train": [], "selection": []}
    if query_pack is not None:
        print(f"query pack {query_pack.path}\n"
              f"  arms {list(query_pack.arms)}  sha256 {query_pack.sha256[:12]}\n"
              f"  gallery UNCHANGED (canonical text, 12-view mean, canonical pc)",
              flush=True)
        # [MASTER ruling 2026-08-31] Assets with no second observation are
        # DROPPED, not carried with the gallery's own vector. `require` below
        # still refuses -- this is the one place allowed to change a pool, and
        # it records the uids rather than a count.
        train_uids, dropped["train"] = query_pack.covered(train_uids)
        dev_val_uids, dropped["selection"] = query_pack.covered(dev_val_uids)
        n = len(dropped["train"]) + len(dropped["selection"])
        if n:
            print(f"  DROPPED {n} asset(s) with no second observation "
                  f"({len(dropped['train'])} train, "
                  f"{len(dropped['selection'])} selection). This run is NOT "
                  f"pool-comparable to any arm without a query pack.", flush=True)

    if args.limit:
        train_uids = train_uids[: args.limit]
        # A smoke run keeps the gallery the same size as the query pool it is
        # given; a 200-asset run against a 4,602 gallery reports a number that
        # belongs to neither the smoke nor the real protocol.
        dev_val_uids = dev_val_uids[: args.limit]

    seed = values["seed"]
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # [MEASURED 2026-08-29] `batch_size: 64` does not fit on this card without
    # recomputing PointBERT's blocks: measured OOM at 48 and at 64, 23.8 GiB at
    # 32. Checkpointing is exact -- same batch, same in-batch negatives, same
    # gradients -- so this keeps the ratified hyperparameter instead of halving
    # it. Off on CPU, where there is no memory pressure and the recompute is
    # pure cost. See BackboneConfig.grad_checkpointing.
    # [FIXED 2026-09-01] This was the literal `"point_encoder_and_fuser"`, which
    # is the bug the CODEX BLOCKER above describes: the protocol's `train_scope`
    # was hashed into the arm and then ignored here, so a protocol declaring
    # `fuser_only` would have produced a checkpoint whose recorded recipe named
    # an ablation the run did not perform. Reading the resolved value is what
    # makes the recorded recipe true, and it is what implements Table 3's
    # "Train fuser only" row. `ULIPBackbone._apply_train_scope` already handles
    # all three scopes; nothing downstream needs a branch because
    # `named_trainable_parameters()` and `trainable_state_dict()` both key off
    # `requires_grad`, so the optimizer and the checkpoint follow automatically.
    scope = training.get("train_scope", "point_encoder_and_fuser")
    backbone = ULIPBackbone(BackboneConfig(device=args.device,
                                           train_scope=scope,
                                           grad_checkpointing=args.device.startswith("cuda")))
    model, loss_fn = build_model(encoding, training, hyperparameters)
    model.to(args.device)
    loss_fn.to(args.device)

    # [UPSTREAM-OFFICIAL-IMPL upstream/ULIP/main.py:129-135] Two groups, and the
    # rule reads the parameter NAME. A single flat `weight_decay=0.1` pulls
    # LayerNorm scales and every bias toward zero, which is decay applied to
    # parameters that have nothing to overfit with. The mechanism is inherited;
    # the number 0.1 is USER-APPROVED separately (see resolve_stage1.py).
    named = (list(backbone.named_trainable_parameters())
             + list(model.named_parameters())
             + list(loss_fn.named_parameters()))
    groups = weight_decay_groups(named, values["weight_decay"])
    opt = torch.optim.AdamW(groups,
                            lr=values["learning_rate"],
                            betas=tuple(values["betas"]),
                            eps=values["eps"])
    p_wd, p_non_wd = groups[0]["params"], groups[1]["params"]
    print(f"optimizer: {len(p_wd):,} decayed / {len(p_non_wd):,} not decayed, "
          f"lr {values['learning_rate']}, betas {tuple(values['betas'])}, "
          f"eps {values['eps']}", flush=True)

    # [KYZEN 2026-08-29] `num_workers` goes to 0 under --preload. With the data
    # already resident the workers have nothing to do, and keeping them would be
    # actively worse: fork gives each of the four a view of an 8.8 GB cache, and
    # they are the per-step process churn this flag exists to remove.
    #
    # Under `mean` this changes no result -- nothing in `__getitem__` draws, so
    # the worker count cannot affect what the model sees (stage1.py:230 says why,
    # and `stage1_encoding_protocol.json` resolves `mean`). Under
    # `random_single_view` it WOULD, because `random` is seeded per worker; that
    # protocol is not in use and this flag must be re-examined before it is.
    workers = 0 if args.preload else 4
    # ONE pack object for the training loader and the selection loader (built
    # above, with the pools). Two constructions in one run -- train independent,
    # select shared -- would choose the epoch that best exploits the leak, and
    # nothing in the numbers would say so. `evaluate_dev_val` gets this object.
    loader = DataLoader(
        Stage1Dataset(train_uids, encoding["image_aggregation"],
                      preload=args.preload, query_pack=query_pack),
        batch_size=values["batch_size"], shuffle=True, collate_fn=collate,
        num_workers=workers, drop_last=True, generator=generator)

    # [CODEX F 2026-08-30] `args.epochs or values["epochs"]` stood here, and
    # `--epochs 0` fell through to the artifact's value -- the operator asked
    # for nothing and got a full run. Same shape as the `--lr-horizon 0` defect
    # the Reviewer found on 2026-08-29: `or` erases the input before any guard
    # can see it.
    epochs = values["epochs"] if args.epochs is None else args.epochs
    if epochs <= 0:
        raise SystemExit(f"--epochs {epochs} must be positive")
    if epochs > values["max_epochs"]:
        # max_epochs is a ceiling the operator enforces, so say so rather than
        # silently clamping: clamping would run a DIFFERENT experiment than the
        # command asked for, which is worse than refusing.
        print(f"WARNING: {epochs} epochs exceeds the approved ceiling "
              f"{values['max_epochs']} (resolve_stage1.py). Nothing stops you; "
              f"record it as a deviation.", flush=True)
    # [KYZEN 2026-08-29] The pilot ladder asks "how many epochs before it stops
    # improving". It could not answer that, because the cosine curve spans
    # `epochs * niter_per_ep` -- so raising --epochs also SLOWS THE ANNEAL, and
    # each rung was a different training rather than a longer one. Measured on
    # the first two rungs: e5 finished its anneal and peaked at 0.9571; e10 was
    # still near peak lr at epoch 4, dipped to 0.8873, and had only climbed back
    # to 0.9471 by epoch 9. "10 is worse than 5" was not a statement about
    # epochs, and nothing in the run recorded that.
    #
    # `--lr-horizon 250` pins the curve to the full approved ceiling and stops
    # early, so every rung is a PREFIX of one trajectory and the rungs become
    # comparable. Kyzen chose this construction after being shown both.
    #
    # It is opt-in and defaults to the old behaviour: silently repointing the
    # schedule would have changed what every past run means without changing any
    # recorded value. The horizon is written into the checkpoint instead, so a
    # run says which construction produced it.
    # `is None`, not `or`: [ULIP2 REVIEWER 2026-08-29] traced the failure --
    # `--lr-horizon 0` under `or` becomes `epochs`, the guard below then tests
    # `epochs < epochs` and passes, and the run silently uses the OLD
    # construction while the operator asked for something else. The `or` erases
    # the input before the guard can see it. One word turns a silent
    # wrong-experiment into a refusal.
    horizon = epochs if args.lr_horizon is None else args.lr_horizon
    if horizon < epochs:
        raise SystemExit(
            f"--lr-horizon {horizon} is shorter than --epochs {epochs}: the "
            f"schedule would run out and the last {epochs - horizon} epochs "
            f"would train at the floor rate. Raise the horizon or lower epochs.")
    # Underscore-prefixed: `training` is the stage1_protocol.json artifact, and
    # these two are facts about THIS RUN, not fields of that protocol. They ride
    # along so `save_checkpoint` can record them without a seventh parameter
    # threaded through two call sites.
    training["_lr_horizon"] = horizon
    training["_epoch_count"] = epochs
    training["_preload"] = bool(args.preload)
    training["_num_workers"] = workers
    # [CODEX MAJOR 4] Computed AFTER --limit and AFTER the phase chose the
    # pools, so the digests describe what this run actually iterated.
    training["_pools"] = pool_provenance(train_uids, dev_val_uids)
    # BY UID, not as a count [MASTER ruling 2026-08-31]. A count tells a later
    # reader that something was removed; the uids tell them WHICH, which is what
    # it takes to reconstruct the pool or to check whether two runs dropped the
    # same assets. And the comparability statement is written down rather than
    # left for someone to infer from a digest mismatch they were not expecting.
    if query_pack is not None:
        training["_pools"]["dropped_uncovered"] = dropped
        training["_pools"]["pool_comparable_to_packless_arms"] = not (
            dropped["train"] or dropped["selection"])
        training["_pools"]["pool_note"] = (
            f"{len(dropped['train'])} train and {len(dropped['selection'])} "
            "selection asset(s) were dropped for having no second observation, "
            "so train_uid_set_sha256 DIFFERS from every arm run without a query "
            "pack. The difference is intended and this run is not pool-"
            "comparable to those arms.")
    print("  digesting input contents...", flush=True)
    check_embedding_sidecars(train_uids + dev_val_uids, encoding)
    training["_pools"]["train_content"] = input_content_digest(train_uids)
    if training["_pools"]["train_content"]["n_missing_embedding_npz"]:
        raise SystemExit(
            f"{training['_pools']['train_content']['n_missing_embedding_npz']} "
            "training asset(s) have no cached embedding; the run would die in a "
            "worker at an unpredictable step. Re-run the encoder first.")
    if dev_val_uids:
        training["_pools"]["selection_content"] = input_content_digest(dev_val_uids)
    training["_initializers"] = initializer_provenance(backbone)
    training["_repeat_index"] = args.repeat_index
    training["_argv"] = sys.argv[1:]
    training["_hardware"] = {
        "device": args.device,
        "gpu": (torch.cuda.get_device_name(0)
                if args.device.startswith("cuda") and torch.cuda.is_available()
                else None),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        # Recorded because it changes the numbers, not because it changes the
        # experiment: TF32 matmuls are a different arithmetic than the default.
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    training["_query_construction"] = (query_pack.identity() if query_pack
                                       else None)
    training["_query_observation"] = args.query_observation
    training["_arm_config_hash"], training["_arm_config"] = arm_config_hash(
        values, training, encoding, args.phase,
        query_construction=training["_query_construction"])
    # [CODEX E 2026-08-30] Stamped once, onto every later metrics row, so a loss
    # curve says which arm and seed produced it without a join.
    runlog.set_run_context(arm_config_hash=training["_arm_config_hash"],
                           seed=seed, repeat_index=args.repeat_index)
    print(f"arm {training['_arm_config_hash'][:12]} -> {run_paths.root}\n"
          f"  {json.dumps(training['_arm_config'], sort_keys=True)}", flush=True)
    lr_schedule = cosine_schedule(
        base=values["learning_rate"], final=values["lr_end"],
        epochs=horizon, niter_per_ep=len(loader),
        warmup_epochs=values["warmup_epochs"], start_warmup=values["lr_start"])
    # Without this the `min()` in the loop is not a guard but a mask: a schedule
    # shorter than the loop would pin the lr at the floor and train on happily,
    # discoverable only in the curve thousands of steps later. Upstream raises
    # IndexError, which at least names the step it happened at; this names the
    # condition before any step happens.
    assert len(lr_schedule) == horizon * len(loader), (
        f"schedule has {len(lr_schedule)} entries for "
        f"{horizon} x {len(loader)} = {horizon * len(loader)} steps")
    # The loop indexes with min(step, len-1). Under a longer horizon that min
    # never clamps, so the run reads a PREFIX and the clamp stays a guard rather
    # than becoming the mechanism -- assert that too, or a future off-by-one in
    # the loop would be absorbed silently by the same min().
    assert epochs * len(loader) <= len(lr_schedule)
    if horizon != epochs:
        print(f"lr horizon {horizon} epochs, training stops at {epochs}: this run "
              f"uses the first {epochs / horizon:.1%} of the cosine curve "
              f"(lr {lr_schedule[0]:.3g} -> {lr_schedule[epochs * len(loader) - 1]:.3g}, "
              f"floor {values['lr_end']:.3g} reached at epoch {horizon})", flush=True)

    print(f"{len(train_uids):,} train assets, batch {values['batch_size']}, "
          f"{epochs} epochs, {len(loader):,} steps/epoch", flush=True)

    started, step, best = time.time(), 0, None
    with runlog.run_progress(NODE):
        for epoch in range(epochs):
            model.train()
            for batch in loader:
                # [PAPER 2.6] masking is unchanged by the query pack: the pack
                # decides WHICH observation each modality is, the mask decides
                # WHETHER it is present. The two are independent and 2.6
                # constrains only the second.
                query_embeds, gallery_embeds = split_embeds(
                    batch, backbone, args.device)

                # [PAPER 2.6] each modality masked INDEPENDENTLY at 30%, on the
                # query side only -- 2.6 also says the gallery encoder is
                # trained to be modality-complete.
                present = sample_modality_mask(
                    gallery_embeds["text"].size(0), p_mask=values["p_mask"],
                    allow_empty=training["allow_all_masked"],
                    device=args.device, generator=generator)

                q = model.query(query_embeds, present=present)
                g = model.gallery(gallery_embeds)

                out = loss_fn(q, g)
                opt.zero_grad(set_to_none=True)
                out["loss"].backward()
                # [UPSTREAM-OFFICIAL-IMPL upstream/ULIP/main.py:292] the lr is
                # WRITTEN per iteration from a precomputed array, not stepped by
                # a torch scheduler. Set before opt.step() so this iteration
                # uses this iteration's rate.
                lr_now = float(lr_schedule[min(step, len(lr_schedule) - 1)])
                for group in opt.param_groups:
                    group["lr"] = lr_now
                opt.step()
                step += 1

                if step % 20 == 0:
                    # Written every 20 steps, printed every 100. The file is
                    # what gets plotted, and a curve sampled at 100 hides the
                    # early collapse that matters most -- the first few hundred
                    # steps are where a broken loss or a runaway temperature
                    # shows, and by step 100 it is one point.
                    runlog.train_metrics(
                        "stage1", epoch=epoch, step=step,
                        loss=round(out["loss"].item(), 6),
                        acc_q2g=round(out.get("acc_q2g", torch.tensor(0.0)).item(), 6),
                        tau=round(loss_fn.temperature.item(), 6),
                        # [FIXED 2026-08-28] `sched.get_last_lr()[0]` and
                        # `params` stood here and NEITHER NAME EXISTS. They were
                        # left behind on 2026-08-27 when the torch scheduler was
                        # replaced by the precomputed `lr_schedule` array
                        # (upstream writes the lr per iteration; see the comment
                        # at the assignment below). This block runs at step 20,
                        # so a real run raised NameError twenty steps into the
                        # first epoch -- Stage 1 could not train at all.
                        #
                        # The 13 unit tests did not catch it because they test
                        # `cosine_schedule` and `weight_decay_groups` as pure
                        # functions and never enter the loop. A smoke run would
                        # have; none had been run since the rewrite.
                        #
                        # `lr` is now the value actually written to the param
                        # groups three lines below, not a second opinion about
                        # it -- the pilot exists to look at this curve, so the
                        # logged rate has to BE the applied rate.
                        lr=round(lr_now, 8),
                        grad_norm=round(float(sum(
                            p.grad.norm().item() ** 2
                            for grp in opt.param_groups for p in grp["params"]
                            if p.grad is not None) ** 0.5), 6))
                if step % 100 == 0:
                    print(f"  epoch {epoch} step {step}: loss {out['loss'].item():.4f}, "
                          f"acc {out.get('acc_q2g', torch.tensor(0.0)).item():.3f}, "
                          f"tau {loss_fn.temperature.item():.4f}", flush=True)

            record = save_checkpoint(backbone, model, loss_fn, hyperparameters,
                                     encoding, training, seed, epoch, run_paths)
            print(f"  epoch {epoch} saved: {record['n_params_saved']:,} params, "
                  f"{record['size_bytes'] / 1e6:.0f} MB", flush=True)

            # [ULIP2 REVIEWER 2026-08-28] The phase is tested here as well as
            # at the assignment, and the redundancy is the point. `if
            # dev_val_uids:` alone is safe ONLY because the assignment twelve
            # lines up is phase-bound; a later edit that populates the list for
            # any other reason would silently put a dev_val score inside a
            # final-phase run -- and dev_val is a subset of the final phase's
            # training pool, so that is not a degraded number, it is a
            # meaningless one printed under a name that looks trustworthy.
            # That edit is the same shape as the `or splits["train"]` fallback
            # already removed above. Two independent conditions, so reopening
            # the path takes two mistakes instead of one.
            if args.phase == "dev" and dev_val_uids:
                scores = evaluate_dev_val(
                    backbone, model, dev_val_uids, encoding["image_aggregation"],
                    args.device, values["batch_size"], query_pack,
                    # the same worker count the training loader uses, so a
                    # --preload run really has no worker processes at all
                    num_workers=workers)
                runlog.train_metrics("stage1_dev_val", epoch=epoch, step=step,
                                     **{k: v for k, v in scores.items()
                                        if isinstance(v, (int, float))},
                                     **flatten_condition_scores(scores))
                print(f"  epoch {epoch} dev-val: mean R@1 {scores['mean_R@1']:.4f}  "
                      f"mean R@5 {scores['mean_R@5']:.4f}  "
                      f"gallery {scores['n_gallery']:,}", flush=True)
                # [D-3] Mean R@1 across the seven conditions, ties broken by
                # Mean R@5 -- the rule Kyzen ratified 2026-08-27, not a choice
                # made here. `>` rather than `>=` on the tie-break too: an equal
                # pair keeps the EARLIER epoch, so "best" never drifts forward
                # through epochs that did not improve on anything.
                key = (scores["mean_R@1"], scores["mean_R@5"])
                if better_checkpoint(key, best[0] if best else None):
                    best = (key, epoch)
                    save_best(record, epoch, scores, args, train_uids,
                              dev_val_uids, run_paths)
                    print(f"  epoch {epoch} is the best so far -> "
                          f"{run_paths.best_checkpoint.name}", flush=True)

    runlog.cost_ledger(wallclock_s=round(time.time() - started, 1),
                       steps=step, epochs=epochs)
    print(f"\nStage 1 done: {step:,} steps -> {run_paths.latest_checkpoint}")
    if best is not None:
        print(f"best epoch {best[1]} (dev-val mean R@1 {best[0][0]:.4f}) "
              f"-> {run_paths.best_checkpoint}")
    elif args.phase == "final":
        # Saying it out loud: in the locked run the answer is the LAST epoch,
        # because the epoch count was already chosen in the development phase.
        # A reader who expects a `stage1_best.pt` here should not have to infer
        # from its absence that nothing selected.
        print("phase=final: no selection was performed [D-3]; "
              f"the run's result is the last epoch in {run_paths.latest_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
