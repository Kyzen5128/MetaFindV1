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
import hashlib
import contextlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import shutil

from metafind import paths, runlog

# Before torch or open_clip: HF_HOME is read at import time and ViT-bigG-14 is
# 9.5 GB.
paths.setup_env()

from metafind.models.stage1_config import (  # noqa: E402
    PAPER_P_MASK,
    PER_VIEW_AGGREGATIONS,
    PRECOMPUTABLE_AGGREGATIONS,
    REQUIRED_HYPERPARAMETERS,
)

NODE = "n10_train_stage1"
TRAINER_VERSION = 2   # v2 saves backbone + tower + loss; v1 dropped the point encoder

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

    def __init__(self, uids: list[str], aggregation: str) -> None:
        if aggregation not in PRECOMPUTABLE_AGGREGATIONS + PER_VIEW_AGGREGATIONS:
            raise ValueError(
                f"unknown image_aggregation {aggregation!r}; "
                f"stage1_config knows {PRECOMPUTABLE_AGGREGATIONS + PER_VIEW_AGGREGATIONS}")
        self.uids = uids
        self.aggregation = aggregation
        self.per_view = aggregation in PER_VIEW_AGGREGATIONS

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int) -> dict:
        uid = self.uids[i]
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
        return {
            "uid": uid,
            "text": cached["text"].astype(np.float32),
            "image": image.astype(np.float32),
            "pc": pc,
        }


def collate(batch: list[dict]):
    import torch

    return {
        "uid": [b["uid"] for b in batch],
        "text": torch.from_numpy(np.stack([b["text"] for b in batch])),
        "image": torch.from_numpy(np.stack([b["image"] for b in batch])),
        "pc": torch.from_numpy(np.stack([b["pc"] for b in batch])),
    }


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


def best_paths(args) -> tuple[Path, Path]:
    """[Codex 2026-08-28] A `--limit` run does NOT write the canonical name.

    `chain_to_stage1.sh` runs Stage 1 as a 200-asset smoke, and the smoke went
    through the same dev-val selection and the same `stage1_best.pt`. So a smoke
    started after a real development run would silently replace the selected
    checkpoint with one chosen from a 200-asset gallery, and the file would look
    exactly the same. The suffix is not tidiness: it is the only thing that keeps
    a 10-minute run from overwriting a multi-hour one.
    """
    if args.limit:
        return (BEST_CKPT_PATH.with_suffix(f".smoke{args.limit}.pt"),
                BEST_CKPT_RECORD.with_suffix(f".smoke{args.limit}.json"))
    return BEST_CKPT_PATH, BEST_CKPT_RECORD


def save_best(record: dict, epoch: int, scores: dict, args,
              train_uids: list, dev_val_uids: list) -> None:
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
    ckpt_path, record_path = best_paths(args)
    digest = hashlib.sha256(
        json.dumps([sorted(train_uids), sorted(dev_val_uids)]).encode()).hexdigest()

    tmp_ckpt = ckpt_path.with_suffix(ckpt_path.suffix + ".part")
    shutil.copyfile(CKPT_PATH, tmp_ckpt)
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


def evaluate_dev_val(backbone, model, dev_val_uids, aggregation, device,
                     batch_size):
    """Mean R@1 / R@5 over the seven Table 1 conditions, on the dev-val gallery.

    Gallery is dev_val itself (`splits.build_eval_protocols`, protocol
    ``C_dev_selection``: query_split and gallery_split are both ``dev_val``).
    Ranking a dev-val query against the whole training pool would make the
    gallery size differ from every reported protocol and the number would not
    compare to anything.

    Returns ``{}`` when there is nothing to evaluate, so a caller can tell
    "no dev-val" from "dev-val scored 0.0" -- an empty gallery silently scoring
    zero is how a selection metric starts choosing checkpoints at random.
    """
    import torch
    from torch.utils.data import DataLoader

    from metafind.eval.retrieval import QUERY_CONDITIONS, condition_mask, recall_at_k

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
    loader = DataLoader(Stage1Dataset(dev_val_uids, aggregation),
                        batch_size=batch_size, shuffle=False, collate_fn=collate,
                        num_workers=4, drop_last=False)
    per_condition_q = {c: [] for c in QUERY_CONDITIONS}
    gallery = []
    # `backbone.model` as well as `model`: the tower is what `model.eval()`
    # reaches, and the point encoder lives on the backbone. See `modules_in_eval`.
    with modules_in_eval(model, getattr(backbone, "model", None)), torch.no_grad():
        for batch in loader:
            embeds = {"text": batch["text"].to(device),
                      "image": batch["image"].to(device),
                      "pc": backbone.encode_pc(batch["pc"].to(device))}
            n = embeds["text"].size(0)
            gallery.append(model.gallery(embeds).float().cpu())
            for cond in QUERY_CONDITIONS:
                mask = condition_mask(cond, n).to(device)
                per_condition_q[cond].append(
                    model.query(embeds, present=mask).float().cpu())

    g = torch.cat(gallery)
    g = torch.nn.functional.normalize(g, dim=-1)
    # The loader is `shuffle=False` and `drop_last=False`, so row i of the query
    # stack and row i of the gallery stack are the same asset. That is what makes
    # arange the target column; it is stated rather than assumed because
    # `rank_of_target` exists precisely so callers do not assume the diagonal.
    targets = np.arange(g.size(0))
    out = {}
    for cond, chunks in per_condition_q.items():
        q = torch.nn.functional.normalize(torch.cat(chunks), dim=-1)
        sim = (q @ g.T).numpy()
        out[cond] = recall_at_k(sim, targets, ks=(1, 5))
    out["mean_R@1"] = float(np.mean([out[c]["R@1"] for c in QUERY_CONDITIONS]))
    out["mean_R@5"] = float(np.mean([out[c]["R@5"] for c in QUERY_CONDITIONS]))
    out["n_gallery"] = int(g.size(0))
    return out


def save_checkpoint(backbone, model, loss_fn, hyperparameters: dict,
                    encoding: dict, training: dict, seed: int, epoch: int) -> dict:
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

    tmp = CKPT_PATH.with_suffix(".part")
    torch.save({**sections,
                "trainer_version": TRAINER_VERSION,
                "epoch": epoch,
                "train_scope": training.get("train_scope", "point_encoder_and_fuser")},
               tmp)
    tmp.replace(CKPT_PATH)

    digest = hashlib.sha256(CKPT_PATH.read_bytes()).hexdigest()
    record = {
        "uri": str(CKPT_PATH),
        "sha256": digest,
        "config_hash": training["hyperparameter_config_hash"],
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
        "train_scope": training.get("train_scope", "point_encoder_and_fuser"),
        "trainable_only": True,
        "n_params_saved": int(n_params),
        "n_params_by_section": {k: int(sum(v.numel() for v in s.values()))
                                for k, s in sections.items()},
        "size_bytes": CKPT_PATH.stat().st_size,
        "clip_train_scope": encoding["actual_clip_train_scope"],
    }
    tmp = CKPT_RECORD.with_suffix(".json.part")
    with tmp.open("w") as fh:
        json.dump(record, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(CKPT_RECORD)
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
        gap = {n for n in trainable - set(state)
               if not n.startswith(new_prefixes)} if new_prefixes else \
            trainable - set(state)
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

    import torch
    from torch.utils.data import DataLoader
    from metafind.models.fusion import sample_modality_mask
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

    encoding, training, hyperparameters = load_protocols()
    values = hyperparameters["values"]

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
    backbone = ULIPBackbone(BackboneConfig(device=args.device,
                                           train_scope="point_encoder_and_fuser",
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

    loader = DataLoader(
        Stage1Dataset(train_uids, encoding["image_aggregation"]),
        batch_size=values["batch_size"], shuffle=True, collate_fn=collate,
        num_workers=4, drop_last=True, generator=generator)

    epochs = args.epochs or values["epochs"]
    if epochs > values["max_epochs"]:
        # max_epochs is a ceiling the operator enforces, so say so rather than
        # silently clamping: clamping would run a DIFFERENT experiment than the
        # command asked for, which is worse than refusing.
        print(f"WARNING: {epochs} epochs exceeds the approved ceiling "
              f"{values['max_epochs']} (resolve_stage1.py). Nothing stops you; "
              f"record it as a deviation.", flush=True)
    lr_schedule = cosine_schedule(
        base=values["learning_rate"], final=values["lr_end"],
        epochs=epochs, niter_per_ep=len(loader),
        warmup_epochs=values["warmup_epochs"], start_warmup=values["lr_start"])
    # Without this the `min()` in the loop is not a guard but a mask: a schedule
    # shorter than the loop would pin the lr at the floor and train on happily,
    # discoverable only in the curve thousands of steps later. Upstream raises
    # IndexError, which at least names the step it happened at; this names the
    # condition before any step happens.
    assert len(lr_schedule) == epochs * len(loader), (
        f"schedule has {len(lr_schedule)} entries for "
        f"{epochs} x {len(loader)} = {epochs * len(loader)} steps")

    print(f"{len(train_uids):,} train assets, batch {values['batch_size']}, "
          f"{epochs} epochs, {len(loader):,} steps/epoch", flush=True)

    started, step, best = time.time(), 0, None
    with runlog.run_progress(NODE):
        for epoch in range(epochs):
            model.train()
            for batch in loader:
                text = batch["text"].to(args.device)
                image = batch["image"].to(args.device)
                pc = backbone.encode_pc(batch["pc"].to(args.device))

                # [PAPER 2.6] each modality masked INDEPENDENTLY at 30%, on the
                # query side only -- 2.6 also says the gallery encoder is
                # trained to be modality-complete.
                present = sample_modality_mask(
                    text.size(0), p_mask=values["p_mask"],
                    allow_empty=training["allow_all_masked"],
                    device=args.device, generator=generator)

                embeds = {"text": text, "image": image, "pc": pc}
                q = model.query(embeds, present=present)
                g = model.gallery(embeds)

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
                                     encoding, training, seed, epoch)
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
                    args.device, values["batch_size"])
                runlog.train_metrics("stage1_dev_val", epoch=epoch, step=step,
                                     **{k: v for k, v in scores.items()
                                        if isinstance(v, (int, float))})
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
                              dev_val_uids)
                    print(f"  epoch {epoch} is the best so far -> "
                          f"{best_paths(args)[0].name}", flush=True)

    runlog.cost_ledger(wallclock_s=round(time.time() - started, 1),
                       steps=step, epochs=epochs)
    print(f"\nStage 1 done: {step:,} steps -> {CKPT_PATH}")
    if best is not None:
        print(f"best epoch {best[1]} (dev-val mean R@1 {best[0][0]:.4f}) "
              f"-> {BEST_CKPT_PATH}")
    elif args.phase == "final":
        # Saying it out loud: in the locked run the answer is the LAST epoch,
        # because the epoch count was already chosen in the development phase.
        # A reader who expects a `stage1_best.pt` here should not have to infer
        # from its absence that nothing selected.
        print("phase=final: no selection was performed [D-3]; "
              f"the run's result is the last epoch in {CKPT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
