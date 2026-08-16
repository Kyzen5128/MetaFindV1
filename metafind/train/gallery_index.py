"""Encode every admitted asset with the FROZEN gallery tower.

# IMPLEMENTS-NODE: n11_gallery_index_staging
# IMPLEMENTS-NODE: n12_promote_index
# IMPLEMENTS-NODE: n11b_stage2_gallery_index

Writes ``gallery_index_staging`` (n11), ``gallery_index`` (n12),
``stage2_gallery_index`` (n11b), and ``run_progress`` / ``cost_ledger``.

Three nodes, one file, because they are the same operation over different
corpora and the thing that must not drift between them is the ENCODER. Splitting
them would mean three copies of "load the checkpoint, freeze the tower, hash the
weights", and the hash is the whole point.

Why staging and promotion are separate
---------------------------------------

n11 writes a staging index, G4 verifies it, n12 promotes exactly that artifact.
The alternative -- write the live index directly -- means a failed verification
leaves a partially-written index that every downstream evaluation would read as
authoritative. Promotion copies nothing: it records the digest G4 saw, and a
second differing write for the same checkpoint is an error rather than an
update.

Why the encoder hash travels with the index
--------------------------------------------

[2.6] The gallery encoder is frozen during Stage 2. That is only checkable if
the weights that produced an index are pinned to it. An index built by a drifted
encoder trains and evaluates the model against embeddings it will never produce
at inference, and nothing in the loss or the metrics would reveal it -- the
numbers would simply be wrong in a self-consistent way.

Stage 1's Objaverse index and Stage 2's ProcTHOR index NEVER merge
-------------------------------------------------------------------

[U-08a] Stage 2 draws its positives from a ProcTHOR gallery; Table 1 retrieves
from Objaverse. Merging them would change Table 1's denominator, which is
already an open question (U-09) and must not acquire a second one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from metafind import paths, runlog

paths.setup_env()

STAGING_PATH = paths.OUTPUTS / "gallery_index_staging.json"
PROMOTED_PATH = paths.OUTPUTS / "gallery_index.json"
STAGE2_PATH = paths.OUTPUTS / "stage2_gallery_index.json"


def load_checkpoint_record() -> dict:
    path = paths.CHECKPOINTS / "stage1_ckpt.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run n10_train_stage1 first")
    return json.loads(path.read_text())


def gallery_encoder_sha256(backbone, model) -> str:
    """A digest over EVERYTHING that produces a gallery embedding.

    The gallery path is::

        text  -> OpenCLIP  --+
        image -> OpenCLIP  --+--> gallery fusion -> e_gallery
        pc    -> PointBERT -> pc_projection --+

    so hashing the fusion alone is not an encoder identity. An earlier version
    did exactly that, and the name `gallery_encoder_sha256` made it read as more
    than it was: two runs with DIFFERENT fine-tuned PointBERTs and the same
    fusion produced the same digest, and G4's "gallery encoder matches Stage 1"
    would have passed while the embeddings differed.

    Sorted by name because Python's parameter iteration order is stable but not
    guaranteed across refactors, and a hash that changes when a module is
    reordered would report drift that did not happen -- worse than no hash,
    because it teaches everyone to ignore it.
    """
    h = hashlib.sha256()
    for tag, module in (("backbone", backbone.model), ("gallery", model.gallery)):
        for name, p in sorted(module.named_parameters()):
            h.update(f"{tag}.{name}".encode())
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _write(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w") as fh:
        json.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def build_index(embeddings: np.ndarray, ids: list[str], out: Path) -> dict:
    """Write the vectors and return the record that describes them."""
    tmp = out.with_suffix(".part.npz")
    np.savez_compressed(tmp, ids=np.array(ids), embeddings=embeddings.astype(np.float32))
    tmp.replace(out)
    return {
        "uri": str(out),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
    }


def promote(gate_passed: bool) -> int:
    """[n12] Late commit: publish the artifact G4 actually verified.

    The digest is compared rather than trusted. A staging index rebuilt between
    verification and promotion would otherwise be published under G4's verdict
    without G4 having seen it -- which is the failure the two-step exists to
    prevent, so promotion cannot be the step that reintroduces it.
    """
    if not STAGING_PATH.exists():
        print(f"{STAGING_PATH} not found -- run n11 first", flush=True)
        return 2
    staging = json.loads(STAGING_PATH.read_text())
    if not gate_passed:
        print("G4_gallery_freeze has not passed; refusing to promote", flush=True)
        return 3

    for stage1_sha, record in staging.items():
        on_disk = hashlib.sha256(Path(record["uri"]).read_bytes()).hexdigest()
        if on_disk != record["sha256"]:
            print(f"{record['uri']} changed since staging "
                  f"({record['sha256'][:12]} -> {on_disk[:12]}); refusing", flush=True)
            return 2

    promoted = json.loads(PROMOTED_PATH.read_text()) if PROMOTED_PATH.exists() else {}
    for stage1_sha, record in staging.items():
        existing = promoted.get(stage1_sha)
        if existing and existing["sha256"] != record["sha256"]:
            # write_once: an index for a given checkpoint is a fact, and a
            # second differing one means two different artifacts claim the same
            # provenance.
            print(f"gallery_index[{stage1_sha[:12]}] already published with a "
                  f"different digest; refusing to overwrite", flush=True)
            return 2
        promoted[stage1_sha] = {**record,
                                "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    _write(PROMOTED_PATH, promoted)
    print(f"promoted {len(staging)} index/indices -> {PROMOTED_PATH}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("stage1", "promote", "stage2"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gate-passed", action="store_true",
                    help="promote only: assert G4_gallery_freeze returned PASS")
    args = ap.parse_args()

    if args.mode == "promote":
        return promote(args.gate_passed)

    import torch
    from metafind.train.stage1 import (
        build_model, load_protocols, load_stage1_checkpoint)
    from metafind.models.ulip_backbone import (
        BackboneConfig, ULIPBackbone, prepare_depth_shell)

    encoding, training, hyperparameters = load_protocols()
    ckpt_record = load_checkpoint_record()

    # train_scope="point_encoder_and_fuser", NOT "fuser_only": the scope decides
    # which parameters carry requires_grad, and load_stage1_checkpoint checks the
    # backbone section against exactly that set. Building with "fuser_only" here
    # would freeze the point encoder before restoring it, so Stage 1's
    # fine-tuned PointBERT would be declared "not expected" and dropped -- the
    # original bug, moved one line down. Freezing for inference happens after
    # the restore, via freeze_gallery / eval.
    backbone = ULIPBackbone(BackboneConfig(device=args.device,
                                           train_scope="point_encoder_and_fuser"))
    model, loss_fn = build_model(encoding, training, hyperparameters)
    load_stage1_checkpoint(backbone, model, loss_fn, Path(ckpt_record["uri"]))
    # Restore first, THEN freeze. The checkpoint's point-encoder section can only
    # land in a backbone whose point encoder is trainable, but the index must be
    # built with it frozen and in eval: ULIP-2's PointBERT config sets
    # drop_path_rate=0.1, so a point encoder left in train() applies stochastic
    # depth and the index comes out non-deterministic. Nothing downstream would
    # catch that -- the vectors would have the right shape and the wrong values.
    backbone.set_train_scope("fuser_only")
    assert backbone.is_frozen(), "the backbone is still trainable or in train mode"
    model.to(args.device).eval()
    model.freeze_gallery(True)
    encoder_sha = gallery_encoder_sha256(backbone, model)

    node = "n11_gallery_index_staging" if args.mode == "stage1" else "n11b_stage2_gallery_index"
    started = time.time()

    with runlog.run_progress(node):
        if args.mode == "stage1":
            ids = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
            ids = sorted(ids["train"] + ids["test"])
            if args.limit:
                ids = ids[: args.limit]
            vectors = []
            for i, uid in enumerate(ids):
                cached = np.load(paths.EMBEDDINGS / f"{uid}.npz")
                cloud = np.load(paths.POINTCLOUDS / f"{uid}.npz")
                pc = np.concatenate([cloud["xyz"], cloud["rgb"]], axis=1)[None]
                with torch.no_grad():
                    pc_vec = backbone.encode_pc(torch.from_numpy(pc.astype(np.float32)))
                    embeds = {
                        "text": torch.from_numpy(cached["text"].astype(np.float32))[None].to(args.device),
                        "image": torch.from_numpy(cached["image"].astype(np.float32))[None].to(args.device),
                        "pc": pc_vec,
                    }
                    # [2.6] the gallery encoder is modality-complete: no mask.
                    vectors.append(model.gallery(embeds)[0].cpu().numpy())
                if (i + 1) % 2000 == 0:
                    print(f"  [{i + 1:6d}/{len(ids)}]", flush=True)
            record = build_index(np.stack(vectors), ids,
                                 paths.OUTPUTS / "gallery_index.npz")
            record["gallery_encoder_sha256"] = encoder_sha
            _write(STAGING_PATH, {ckpt_record["sha256"]: record})
            print(f"\nstaged {record['count']:,} x {record['dim']} "
                  f"-> {STAGING_PATH}")
        else:
            # [2.6] "The gallery encoder is trained to be modality-complete."
            # F26 measured that 24 of the 1,467 ProcTHOR assets have no depth at
            # all -- transparent materials are absent from Unity's depth prepass
            # -- so they have text and images and no point cloud.
            #
            # They are EXCLUDED from the Stage 2 gallery rather than admitted
            # with a gap. Admitting them would mean the gallery side runs a
            # presence mask, which 2.6 rules out, and the alternative of
            # zero-filling is the failure L1-SEMEDGE-NO-ZEROFILL exists to name.
            # The cost is that those assets cannot be Stage 2 positives, hence
            # cannot be targets: 24 of 1,467, 1.6%, recorded here rather than
            # discovered as a mysterious KeyError inside n13.
            mods = sorted(paths.PROCTHOR_MODALITIES.glob("*.json"))
            if args.limit:
                mods = mods[: args.limit]
            ids, vectors, excluded = [], [], []
            for path in mods:
                rec = json.loads(path.read_text())
                if rec["pointcloud_uri"] is None:
                    excluded.append({"asset_id": rec["asset_id"],
                                     "reason": rec["pointcloud_missing_reason"]})
                    continue
                # [P0-4] pc_norm happens INSIDE prepare_depth_shell: n07b stores
                # world-frame points (asset lifted to y=40 m), n03 stores
                # unit-normalised ones, and the checkpoint was trained on the
                # latter. The grey channel is there because the shell has no
                # colour, not because grey is a measurement.
                cloud = np.load(rec["pointcloud_uri"])["xyz"].astype(np.float32)
                pc = prepare_depth_shell(cloud)
                with torch.no_grad():
                    text_vec = backbone.encode_text([rec["text"]])
                    view_vecs = backbone.encode_image(torch.stack([
                        backbone.preprocess(Image.open(v).convert("RGB"))
                        for v in rec["view_paths"]]))
                    image_vec = view_vecs.mean(dim=0, keepdim=True)
                    pc_vec = backbone.encode_pc(torch.from_numpy(pc))
                    vectors.append(model.gallery(
                        {"text": text_vec, "image": image_vec, "pc": pc_vec}
                    )[0].cpu().numpy())
                ids.append(rec["asset_id"])
                if len(ids) % 200 == 0:
                    print(f"  [{len(ids):5d}/{len(mods)}]", flush=True)

            record = build_index(np.stack(vectors), ids,
                                 paths.OUTPUTS / "stage2_gallery.npz")
            record.update({
                "asset_ids": ids,
                "embedding_dim": record["dim"],
                "n_assets": record["count"],
                "gallery_encoder_sha256": encoder_sha,
                "modality_completeness": {
                    "complete": len(ids),
                    "excluded_no_pointcloud": excluded,
                },
            })
            _write(STAGE2_PATH, record)
            print(f"\nstage2 index: {record['n_assets']:,} assets, "
                  f"{len(excluded)} excluded for having no point cloud "
                  f"-> {STAGE2_PATH}")

    runlog.cost_ledger(wallclock_s=round(time.time() - started, 1),
                       assets_encoded=record["count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
