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
import json
import os
import time
from pathlib import Path

import numpy as np

from metafind import paths, runlog

# Before torch or open_clip: HF_HOME is read at import time and ViT-bigG-14 is
# 9.5 GB.
paths.setup_env()

from metafind.models.stage1_config import (  # noqa: E402
    PAPER_P_MASK,
    REQUIRED_HYPERPARAMETERS,
)

NODE = "n10_train_stage1"
TRAINER_VERSION = 2   # v2 saves backbone + tower + loss; v1 dropped the point encoder

CKPT_PATH = paths.CHECKPOINTS / "stage1.pt"
CKPT_RECORD = paths.CHECKPOINTS / "stage1_ckpt.json"


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
    """

    def __init__(self, uids: list[str], aggregation: str) -> None:
        self.uids = uids
        self.aggregation = aggregation

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int) -> dict:
        uid = self.uids[i]
        cached = np.load(paths.EMBEDDINGS / f"{uid}.npz")
        cloud = np.load(paths.POINTCLOUDS / f"{uid}.npz")
        xyz = cloud["xyz"].astype(np.float32)
        rgb = cloud["rgb"].astype(np.float32) if "rgb" in cloud else None
        pc = xyz if rgb is None else np.concatenate([xyz, rgb], axis=1)
        return {
            "uid": uid,
            "text": cached["text"].astype(np.float32),
            "image": cached["image"].astype(np.float32),
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
    """[L1-CKPT-TRAINABLE-ONLY] Only what the optimizer moves.

    Keyed off ``requires_grad`` rather than off a name prefix: a prefix list
    goes stale the moment a module is renamed, and it would go stale silently --
    the checkpoint would still save, still load, and quietly omit a trained
    tensor.
    """
    return {name: p.detach().cpu()
            for name, p in module.named_parameters() if p.requires_grad}


# The three modules Stage 1's optimizer touches. Naming them here rather than in
# save_checkpoint's body is what OPTIMIZER_COVERS_CHECKPOINT can assert against.
CKPT_SECTIONS = ("backbone_trainable_state", "tower_trainable_state",
                 "loss_trainable_state")


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
    train_uids = json.loads(splits_path.read_text())["object"]["train"]
    if args.limit:
        train_uids = train_uids[: args.limit]

    seed = values["seed"]
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    backbone = ULIPBackbone(BackboneConfig(device=args.device,
                                           train_scope="point_encoder_and_fuser"))
    model, loss_fn = build_model(encoding, training, hyperparameters)
    model.to(args.device)
    loss_fn.to(args.device)

    params = [p for p in list(backbone.trainable_parameters())
              + list(model.parameters()) + list(loss_fn.parameters())
              if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=values["learning_rate"],
                            weight_decay=values["weight_decay"])

    loader = DataLoader(
        Stage1Dataset(train_uids, encoding["image_aggregation"]),
        batch_size=values["batch_size"], shuffle=True, collate_fn=collate,
        num_workers=4, drop_last=True, generator=generator)

    epochs = args.epochs or values["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs * len(loader), 1))

    print(f"{len(train_uids):,} train assets, batch {values['batch_size']}, "
          f"{epochs} epochs, {len(loader):,} steps/epoch", flush=True)

    started, step = time.time(), 0
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
                opt.step()
                sched.step()
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
                        lr=round(sched.get_last_lr()[0], 8),
                        grad_norm=round(float(sum(
                            p.grad.norm().item() ** 2 for p in params
                            if p.grad is not None) ** 0.5), 6))
                if step % 100 == 0:
                    print(f"  epoch {epoch} step {step}: loss {out['loss'].item():.4f}, "
                          f"acc {out.get('acc_q2g', torch.tensor(0.0)).item():.3f}, "
                          f"tau {loss_fn.temperature.item():.4f}", flush=True)

            record = save_checkpoint(backbone, model, loss_fn, hyperparameters,
                                     encoding, training, seed, epoch)
            print(f"  epoch {epoch} saved: {record['n_params_saved']:,} params, "
                  f"{record['size_bytes'] / 1e6:.0f} MB", flush=True)

    runlog.cost_ledger(wallclock_s=round(time.time() - started, 1),
                       steps=step, epochs=epochs)
    print(f"\nStage 1 done: {step:,} steps -> {CKPT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
