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
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from metafind import paths, runlog

paths.setup_env()

STAGING_PATH = paths.OUTPUTS / "gallery_index_staging.json"
PROMOTED_PATH = paths.OUTPUTS / "gallery_index.json"
STAGE2_PATH = paths.OUTPUTS / "stage2_gallery_index.json"


def load_checkpoint_record(record_path: str | Path | None = None) -> dict:
    """Which Stage 1 checkpoint this index is built from.

    [CODEX MAJOR 2026-08-30] Was a fixed `paths.CHECKPOINTS / "stage1_ckpt.json"`.
    Stage 1 gained `--out-dir` so a sweep's arms stop overwriting each other, and
    with the path fixed here, a run-specific checkpoint could reach downstream
    only by being copied back over the canonical name -- which destroys the
    provenance the out-dir was added to create.

    The default is unchanged, so every existing command keeps working; naming
    the record is how a sweep's selected arm is promoted without a copy.
    """
    path = Path(record_path) if record_path else paths.CHECKPOINTS / "stage1_ckpt.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run n10_train_stage1 first")
    record = json.loads(path.read_text())
    # [CODEX MAJOR 2026-08-30] The record is a CLAIM about a file, and until now
    # nothing checked it. Codex demonstrated the consequence: a record naming
    # checkpoint A's provenance can be pointed at checkpoint B's bytes and every
    # downstream artifact inherits the wrong identity, silently. `--out-dir`
    # makes several checkpoints exist at once, so this stops being theoretical.
    weights = Path(record["uri"])
    if not weights.exists():
        raise FileNotFoundError(
            f"{path} names {weights}, which does not exist. The record and its "
            "weights have been separated.")
    actual = hashlib.sha256(weights.read_bytes()).hexdigest()
    if actual != record["sha256"]:
        raise ValueError(
            f"{path} records sha256 {record['sha256'][:16]}... but {weights} "
            f"hashes to {actual[:16]}.... Refusing: an index built from these "
            "bytes would carry the other checkpoint's provenance.")
    return record


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


def _write(path: Path, obj, dump=json.dump) -> None:
    """Atomic, fsynced write. ``dump(obj, fh)`` -- json by default.

    ``dump`` exists so G4's YAML gate record goes through THIS writer rather
    than a second one: the temp-and-rename plus fsync is the property that
    matters, and it should not have to be re-implemented per serialisation.
    ``yaml.safe_dump`` has the same ``(obj, stream)`` signature as ``json.dump``.
    """
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w") as fh:
        dump(obj, fh)
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


INDEX_RECORD_FIELDS = ("uri", "sha256", "dim", "count",
                       "stage1_checkpoint_sha256", "gallery_encoder_sha256")


class IndexUnreadable(ValueError):
    """The .npz cannot be opened, or does not carry ``ids`` and ``embeddings``.

    A subclass so the documented contract ("raises ValueError") stays true, and
    a distinct type so G4 can tell MISSING EVIDENCE from a FAILED CHECK without
    matching on an exception message. Those are rc 3 and rc 2 and they are not
    interchangeable: a corrupt archive means nobody knows whether the index was
    good, while a record that misdescribes its index means somebody knows it was
    not.
    """


def verified_index(record: dict, source: str) -> tuple[list[str], np.ndarray]:
    """Hash the bytes, then read THOSE bytes. Both halves, one call.

    Splitting them -- verify here, open there -- is two separate opens of one
    path, and the file can change in between. A verification that does not hand
    back the bytes it verified has verified something other than what gets used.

    ``source`` names whoever is making the claim (the promoted registry, the
    staging record) so the error says which document is wrong. Returns
    ``(ids, embeddings)`` with ``embeddings`` exactly as stored: float32,
    unnormalised, in the index's own row order.
    """
    if missing := [k for k in INDEX_RECORD_FIELDS if k not in record]:
        raise ValueError(f"{source} record is missing {missing}; it cannot "
                         "identify the index it describes")
    uri = Path(record["uri"])
    if not uri.exists():
        raise FileNotFoundError(
            f"{source} names {uri}, which does not exist. The record and its "
            "vectors have been separated.")
    # ONE read. Hashing `uri.read_bytes()` and then handing `uri` to np.load is
    # two opens of one path, and the docstring above promises the bytes hashed
    # are the bytes returned -- a promise the caller cannot keep and the callee
    # was not enforcing. The window is small and the failure is silent, which is
    # the combination this repository keeps being bitten by.
    raw = uri.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != record["sha256"]:
        raise ValueError(
            f"{uri} hashes to {actual[:16]}... but {source} records "
            f"{record['sha256'][:16]}.... These are not the verified vectors.")

    try:
        npz = np.load(io.BytesIO(raw))
        ids = [str(x) for x in npz["ids"]]
        embeddings = npz["embeddings"]
    except Exception as exc:  # noqa: BLE001 -- any read failure is unreadable
        raise IndexUnreadable(f"{uri} cannot be read as an index: {exc}") from exc
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got {embeddings.shape}")
    if len(ids) != embeddings.shape[0]:
        raise ValueError(f"{len(ids)} ids for {embeddings.shape[0]} vectors "
                         f"in {uri}")
    if (embeddings.shape[0], embeddings.shape[1]) != (record["count"], record["dim"]):
        raise ValueError(f"{uri} is {embeddings.shape} but {source} records "
                         f"count={record['count']} dim={record['dim']}")
    return ids, embeddings


def load_promoted_index_for_checkpoint(
    checkpoint_sha: str,
    promoted_path: Path | None = None,
) -> tuple[dict, list[str], np.ndarray]:
    """The promoted gallery index for one Stage 1 checkpoint. CONTRACT.

    Every consumer of ``gallery_index`` goes through here. n15 does not parse
    the registry itself, because the registry is a map of CLAIMS about files --
    ``{stage1_sha: {uri, sha256, dim, count, ...}}`` -- and a claim nobody
    re-checks is how an index built by one encoder gets read under another
    one's provenance.

    Re-verified on EVERY call, not once at import: the file named by the record
    lives on a shared volume, and "it was correct when this process started" is
    not the question a reader is asking.

    Args:
        checkpoint_sha: the Stage 1 checkpoint sha256 the index must belong to.
            This is the registry key AND the identity being asserted.
        promoted_path: the registry. Defaults to ``gallery_index.json``.

    Returns:
        ``(record, ids, embeddings)``. ``record`` is the registry entry, whose
        provenance fields the caller carries into its own output.
        ``embeddings`` is ``(N, D)`` **exactly as stored** -- float32,
        unnormalised, in the index's own row order -- and ``ids[i]`` names row
        ``i``. Row selection, uid ordering and the float64 normalisation are
        n15's semantics, not the registry's, and none of them happen here.

    Raises:
        FileNotFoundError: no registry, or the record names a file that is gone.
        KeyError: the registry holds no index for this checkpoint.
        ValueError: the record cannot identify its index, the bytes do not hash
            to the recorded digest, the array disagrees with the record's own
            ``count``/``dim``, or an asset id repeats. Never returns None, an
            empty result, or a partially-verified array.
    """
    path = Path(promoted_path) if promoted_path else PROMOTED_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- n12 has not promoted an index. A missing "
            "registry is not an empty one; refusing to score against nothing.")
    promoted = json.loads(path.read_text())
    if checkpoint_sha not in promoted:
        raise KeyError(
            f"{path} holds no index for checkpoint {checkpoint_sha[:16]}...; "
            f"it has {sorted(k[:16] for k in promoted)}")
    record = promoted[checkpoint_sha]
    ids, embeddings = verified_index(record, str(path))
    if len(set(ids)) != len(ids):
        # A consumer building a uid -> column map from a list with repeats
        # silently loses every earlier duplicate, and the loss shows up as a
        # slightly lower R@k that nothing explains. G4 rejects duplicates too;
        # this is here because the loader is what n15 trusts, and a trust
        # boundary that relies on an upstream gate having run is not one.
        raise ValueError(f"{record['uri']} repeats "
                         f"{len(ids) - len(set(ids))} asset id(s)")
    return record, ids, embeddings


GATE_RECORD_PATH = paths.LOGS / "gates" / "G4_gallery_freeze.yaml"


def promote(gate_record_path: Path | None = None) -> int:
    """[n12] Late commit: publish the artifact G4 actually verified.

    The digest is compared rather than trusted. A staging index rebuilt between
    verification and promotion would otherwise be published under G4's verdict
    without G4 having seen it -- which is the failure the two-step exists to
    prevent, so promotion cannot be the step that reintroduces it.

    [2026-08-30] This took ``gate_passed: bool`` from a ``--gate-passed`` CLI
    flag, so an operator typing a word stood in for the evidence the gate exists
    to produce -- and nothing in the promoted registry recorded which verdict,
    if any, had been asserted. The flag is GONE rather than ignored: an argument
    that no longer decides anything still reads as "the gate is asserted here",
    and the next person to add a caller would pass it.

    Promotion now reads G4's own record and re-verifies all three identities it
    contains -- staging record bytes, index bytes, checkpoint sha -- against
    what is on disk NOW. Verifying one and inferring the others is exactly the
    hole this closes: the staging-record digest proves the record is the one G4
    read, and the index digest proves the vectors are the ones G4 scored, and
    neither implies the other.

    [L1-GATE-NORECORD] A missing gate record is NOT PASSED, never pass-by-default.

    Returns 0 on success, 3 when the gate evidence is absent or does not say
    PASS, 2 when an artifact disagrees with what the gate recorded.
    """
    if not STAGING_PATH.exists():
        print(f"{STAGING_PATH} not found -- run n11 first", flush=True)
        return 2
    staging = json.loads(STAGING_PATH.read_text())

    gate_path = Path(gate_record_path) if gate_record_path else GATE_RECORD_PATH
    if not gate_path.exists():
        print(f"{gate_path} not found -- G4_gallery_freeze has not run; "
              "a missing gate record is not a pass", flush=True)
        return 3
    try:
        gate = yaml.safe_load(gate_path.read_text())
    except yaml.YAMLError as exc:
        print(f"{gate_path} is not readable YAML: {exc}", flush=True)
        return 3
    if not isinstance(gate, dict) or gate.get("gate_id") != "G4_gallery_freeze":
        print(f"{gate_path} is not a G4_gallery_freeze record "
              f"(gate_id={gate.get('gate_id') if isinstance(gate, dict) else None})",
              flush=True)
        return 3
    if gate.get("verdict") != "PASS":
        print(f"{gate_path} records verdict {gate.get('verdict')!r} "
              f"(rc {gate.get('rc')}); refusing to promote", flush=True)
        return 3
    if gate.get("is_terminal") is not True:
        print(f"{gate_path} is not a terminal record; refusing to promote",
              flush=True)
        return 3

    # The record G4 read must be the record on disk now, byte for byte. This
    # subsumes every field inside it -- and the two digests below are still
    # compared, because "the file did not change" and "these bytes are the
    # index G4 scored" are different claims and only one of them is about the
    # .npz.
    staging_now = hashlib.sha256(STAGING_PATH.read_bytes()).hexdigest()
    if gate.get("staging_record_sha256") != staging_now:
        print(f"{STAGING_PATH} changed since G4 verified it "
              f"({str(gate.get('staging_record_sha256'))[:12]} -> "
              f"{staging_now[:12]}); refusing", flush=True)
        return 2

    for stage1_sha, record in staging.items():
        if gate.get("stage1_checkpoint_sha256") != record["stage1_checkpoint_sha256"]:
            print(f"G4 verified an index built from checkpoint "
                  f"{str(gate.get('stage1_checkpoint_sha256'))[:12]}... but the "
                  f"staging record names "
                  f"{record['stage1_checkpoint_sha256'][:12]}...; refusing",
                  flush=True)
            return 2
        # Hashed ONCE, compared against BOTH authorities. They are different
        # questions -- "is this what G4 scored?" and "is this what the staging
        # record describes?" -- and only the second one existed before.
        on_disk = hashlib.sha256(Path(record["uri"]).read_bytes()).hexdigest()
        if gate.get("index_sha256") != on_disk:
            print(f"{record['uri']} is not the artifact G4 verified "
                  f"({str(gate.get('index_sha256'))[:12]} -> {on_disk[:12]}); "
                  "refusing", flush=True)
            return 2
        if on_disk != record["sha256"]:
            print(f"{record['uri']} changed since staging "
                  f"({record['sha256'][:12]} -> {on_disk[:12]}); refusing", flush=True)
            return 2

    gate_sha = hashlib.sha256(gate_path.read_bytes()).hexdigest()

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
        promoted[stage1_sha] = {
            **record,
            # Which verdict published this, and the bytes of that verdict. A
            # promoted index whose record cannot name the gate record that
            # cleared it is back to being an assertion.
            "gate_record_uri": str(gate_path),
            "gate_record_sha256": gate_sha,
            "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    _write(PROMOTED_PATH, promoted)
    print(f"promoted {len(staging)} index/indices -> {PROMOTED_PATH}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("stage1", "promote", "stage2"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stage1-ckpt-record", default=None,
                    help="Stage 1 checkpoint record to build from. Defaults to "
                         "the canonical stage1_ckpt.json; name a run's own "
                         "record to promote a sweep arm without copying it "
                         "over the canonical file.")
    args = ap.parse_args()

    if args.mode == "promote":
        # No flag. `--gate-passed` used to let the operator assert the verdict;
        # promotion reads G4's record instead, and an unrecognised argument is
        # the loudest possible way to tell an old command that it no longer
        # means what it says.
        return promote()

    import torch
    from metafind.train.stage1 import (
        build_model, load_protocols, load_stage1_checkpoint)
    from metafind.models.ulip_backbone import (
        BackboneConfig, ULIPBackbone, prepare_depth_shell)

    encoding, training, hyperparameters = load_protocols()
    ckpt_record = load_checkpoint_record(args.stage1_ckpt_record)

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
            # Named by the checkpoint that produced it, NOT a fixed live path.
            # `gallery_index.npz` was overwritten by every rebuild while the
            # promoted registry kept a record per stage1_sha -- so an older
            # record's `sha256` stopped matching the bytes at its own `uri`, and
            # the write-once guarantee that promotion exists to provide was not
            # actually held by anything on disk.
            record = build_index(
                np.stack(vectors), ids,
                paths.OUTPUTS / f"gallery_index_{ckpt_record['sha256'][:16]}.npz")
            record["gallery_encoder_sha256"] = encoder_sha
            # [CODEX MAJOR 2026-08-30] Stated in the record, not only in the
            # dict key: Stage 2 reads the record, and a key is not a field.
            record["stage1_checkpoint_sha256"] = ckpt_record["sha256"]
            record["stage1_ckpt_record"] = str(
                args.stage1_ckpt_record
                or paths.CHECKPOINTS / "stage1_ckpt.json")
            _write(STAGING_PATH, {ckpt_record["sha256"]: record})
            print(f"\nstaged {record['count']:,} x {record['dim']} "
                  f"-> {STAGING_PATH}")
        else:
            # [2.6] "The gallery encoder is trained to be modality-complete."
            # 28 of the 1,467 ProcTHOR assets have no depth at all -- transparent
            # materials are absent from Unity's depth prepass -- so they have
            # text and images and no point cloud.
            #
            # 28 is defined by the SAME condition the loop below tests, and the
            # condition is the definition: a record whose `pointcloud_uri` key is
            # present and null. Rescanned 2026-08-30 over all 1,467 files in
            # `procthor_modalities/` (OBSERVED DATA): 28 null, 0 with the key
            # absent, 0 with an empty string, 1,439 non-null -- and all 1,439 of
            # those .npz files exist on disk, so "missing file" is a different
            # (currently empty) failure mode and is NOT what this counts. All 28
            # carry the same `pointcloud_missing_reason`: "every view was empty;
            # the asset never entered frame". The 24 that stood here was an
            # earlier measurement and is superseded.
            #
            # They are EXCLUDED from the Stage 2 gallery rather than admitted
            # with a gap. Admitting them would mean the gallery side runs a
            # presence mask, which 2.6 rules out, and the alternative of
            # zero-filling is the failure L1-SEMEDGE-NO-ZEROFILL exists to name.
            # The cost is that those assets cannot be Stage 2 positives, hence
            # cannot be targets: 28 of 1,467, 1.9%, recorded here rather than
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

            record = build_index(
                np.stack(vectors), ids,
                paths.OUTPUTS / f"stage2_gallery_{ckpt_record['sha256'][:16]}.npz")
            record.update({
                "asset_ids": ids,
                "embedding_dim": record["dim"],
                "n_assets": record["count"],
                "gallery_encoder_sha256": encoder_sha,
                "modality_completeness": {
                    "complete": len(ids),
                    "excluded_no_pointcloud": excluded,
                },
                # [CODEX MAJOR 2026-08-30] Stage 2's [G6] comment claimed the
                # index and the checkpoint were compared. Nothing compared them,
                # because the index never said which checkpoint made it.
                "stage1_checkpoint_sha256": ckpt_record["sha256"],
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
