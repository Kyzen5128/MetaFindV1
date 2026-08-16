"""Materialise the resolved Stage 2 protocols into the channels n13 reads.

# IMPLEMENTS-NODE: n09b_resolve_stage2_protocol

Writes ``stage2_protocol``, ``stage2_positive_map``, ``essgnn_edge_protocol``,
``essgnn_arch_protocol`` and ``run_progress``.

This node does not DECIDE. U-08a, U-08b, U-08d and U-08e are resolved and
recorded in the registry; what was missing is the artifact n13 can read, and a
write_once channel does not fill itself.

The ESSGNN protocols are different: their unknowns are non-blocking with
"record the choice" resolutions, and the choices are made here rather than by
whatever ``essgnn.py``'s dataclass defaults happen to be. That distinction is
the point of the node. A default is a decision nobody wrote down.

What each ESSGNN field settles
-------------------------------

[U-29] ``physical_relation_encoding`` -- 2.3 defines physical and semantic
edges; 2.5's f_h and f_x take exactly ONE edge argument, e_ij, which Appendix C
defines as the LLM sentence. So nothing says how a physical edge enters at all.
We use it to determine the neighbourhood N(i) and nothing else: the edge feature
stays purely semantic, which is the only reading under which the published
tensor signature is literally correct.

[U-30] ``semantic_missing_representation`` -- f_h's input width is fixed, so an
edge that reaches the network without an e_ij still needs those slots filled.
Zero is forbidden (indistinguishable from a real embedding), so a learned
missing-edge token, the same mechanism U-11 uses for absent modalities.

[U-19] ``directionality`` -- symmetric, matching n07's stored edges. The paper
never says, and a directed reading would give different h updates.

[U-17] ``distance`` -- squared. Appendix C's Eq. 10-12 and the reference EGNN
both use it; 2.5's prose says the L2 norm. The two disagree and Appendix C is
the one that carries the proof.
"""

from __future__ import annotations

import argparse
import getpass
import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from metafind import paths, runlog

NODE = "n09b_resolve_stage2_protocol"

STAGE2_PROTOCOL = paths.OUTPUTS / "stage2_protocol.json"
POSITIVE_MAP = paths.OUTPUTS / "stage2_positive_map.json"
EDGE_PROTOCOL = paths.OUTPUTS / "essgnn_edge_protocol.json"
ARCH_PROTOCOL = paths.OUTPUTS / "essgnn_arch_protocol.json"

# [U-08a/b/d/e] Recorded in the registry; materialised here.
STAGE2_DECISIONS = {
    "gallery_scope": "procthor",
    "positive_identity": "same_asset_id",
    "modality_source": "ai2thor_isolated",
    "image_protocol": "n04_compatible",
    "pointcloud_source": "multiview_depth_shell",
    "query_pointcloud": "optional",
    "sampling_unit": "object_instance",
    "target_eligibility": "has_modalities_and_pointcloud",
    "target_removed_before_essgnn": True,
    "samples_per_house": "all_eligible",
    "instance_resampling": "fixed",
    "epoch_definition": "one pass over the enumerated leave-one-out samples",
    "batch_positive_uniqueness": True,
    # [U-32] 2.6 says the layout vector is "omitted in 30% of BATCHES", which
    # reads as one draw per batch. `sample` stays selectable as a variant.
    "scene_dropout_granularity": "batch",
}

EDGE_DECISIONS = {
    "topology": "support_union_adjacency",
    "physical_relation_encoding": "neighbourhood_only",
    "semantic_missing_representation": "learned_missing_token",
    "directionality": "symmetric",
}

ARCH_DECISIONS = {
    # [C1 / U-26] WHICH ESSGNN. The paper describes two, and until this key
    # existed the code silently implemented one of them: ESSGCL builds f_h and
    # f_x unconditionally, so `essgnn_arch_protocol.status = "resolved"` and
    # "U-26 is blocking and unresolved" were both true at once and G6 could not
    # gate the thing it exists to gate.
    #
    #   sec25_two_mlp        2.5's f_h and f_x, each reading (d, h_i, h_j, e_ij)
    #   appendix_shared_msg  the appendix's phi_e -> m_ij -> {phi_x, phi_h}
    #
    # NOT a free knob and NOT a value to guess: it selects between two things
    # the authors themselves wrote, and the two are different models. Written
    # here as UNRESOLVED so G6 blocks Stage 2 until a person decides.
    #
    # Note what is NOT part of this fork. `j in N(i)` versus `j != i` looks like
    # it belongs here -- 2.5 uses one and the appendix the other -- but EGNN's
    # own model.tex settles it: "in this work we choose to aggregate messages
    # from all other nodes j != i, but we could limit the message exchange to a
    # given neighborhood j in N(i) if desired in both equations". It is an
    # option upstream states outright, available to EITHER family, so pairing
    # the appendix's messages with N(i) is not cross-mixing.
    "architecture_family": None,
    "use_io_projections": True,
    "distance": "squared",
    # ESSGNNConfig's vocabulary is "updated"|"current", not "h_next".
    # [U-26] 2.5 writes f_x taking h^{l+1}, so the coordinate head sees the
    # UPDATED features -- which is what "updated" names.
    "coord_feat": "updated",
    "layer_sharing": "independent",
    "pooling": "mean",
    "hidden_dim": 128,
    "n_layers": 4,
    # [U-35] Recorded because G6 requires it and the report must state it,
    # but ESSGNNConfig has no such field: metafind/models/essgnn.py builds one
    # Linear-SiLU-Linear for every MLP. So this string DESCRIBES the code
    # rather than configuring it, and _assert_matches_code below checks the
    # two have not drifted -- a recorded value nothing reads is how U-14 and
    # U-11 were being decided by dataclass defaults before n05b existed.
    "mlp_structure": "linear_silu_linear",
}


def build_positive_map() -> dict:
    """[U-08a] Identity. Every eligible ProcTHOR asset is its own positive.

    Eligibility requires a POINT CLOUD, not merely a modality record: F26 found
    24 transparent assets that AI2-THOR's depth prepass omits, and 2.6 needs a
    modality-complete gallery, so those are excluded from the gallery and cannot
    be positives. Writing them here would name a positive that
    stage2_gallery_index does not contain -- a loss with a name and no vector.
    """
    mapping, skipped = {}, []
    for path in sorted(glob.glob(str(paths.PROCTHOR_MODALITIES / "*.json"))):
        rec = json.loads(Path(path).read_text())
        if rec.get("pointcloud_uri") is None:
            skipped.append(rec["asset_id"])
            continue
        mapping[rec["asset_id"]] = {
            "positive_asset_id": rec["asset_id"],
            "gallery_scope": "procthor",
            "method": "identity",
            "confidence": "certain",
        }
    return mapping, skipped


def assert_matches_code(arch: dict) -> None:
    """Refuse to record a protocol the code cannot honour.

    ESSGNNConfig.from_protocol validates the fields it consumes, but it runs at
    Stage 2 -- hours after this node. Checking here means a typo in a Literal
    value fails in a second rather than after Stage 1 has trained.
    """
    import typing

    from metafind.models import essgnn as mod
    from metafind.models.essgnn import ESSGNNConfig

    # MEASURED: comparing the built config's fields against the protocol proves
    # NOTHING. ESSGNNConfig annotates them with Literal, and a dataclass does not
    # enforce annotations at runtime -- it stores whatever it is handed, so the
    # two always match and the check always passes. The first version of this
    # function did exactly that and accepted distance="l2".
    #
    # The vocabularies have to be read off the annotations and compared.
    hints = typing.get_type_hints(ESSGNNConfig)
    for field, value in arch.items():
        hint = hints.get(field)
        allowed = typing.get_args(hint) if typing.get_origin(hint) is typing.Literal else None
        if allowed and value not in allowed:
            raise ValueError(
                f"essgnn_arch_protocol.{field} = {value!r} is not one of "
                f"{list(allowed)}. ESSGNNConfig would accept it silently and "
                "the failure would surface inside Stage 2."
            )

    # from_protocol names its own required set; check it HERE rather than by
    # letting the call below raise, because that call is skipped while C1 is
    # undecided and a missing field would then go unreported until Stage 2.
    required = {"architecture_family", "use_io_projections", "distance",
                "coord_feat", "layer_sharing", "pooling", "hidden_dim",
                "n_layers"}
    if missing := required - arch.keys():
        raise ValueError(f"essgnn_arch_protocol is missing {sorted(missing)}")

    # C1 undecided: the round trip through ESSGNNConfig cannot run, because
    # from_protocol correctly refuses an undecided family. Skipping only that
    # step -- rather than substituting a placeholder family -- keeps this
    # function from being the place that quietly picks one.
    if arch["architecture_family"] is not None:
        cfg = ESSGNNConfig.from_protocol(
            {**arch, "status": "resolved"},
            node_feat_dim=512, edge_feat_dim=512, out_dim=1280)
        for field in ("distance", "coord_feat", "layer_sharing", "pooling",
                      "hidden_dim", "n_layers", "use_io_projections"):
            if getattr(cfg, field) != arch[field]:
                raise ValueError(
                    f"essgnn_arch_protocol.{field} = {arch[field]!r} did not "
                    f"survive ESSGNNConfig, which holds {getattr(cfg, field)!r}"
                )
    # [U-35] mlp_structure has no config field, so the string describes the
    # code. Assert the description is still true, and refuse any other value --
    # recording one essgnn.py does not implement is worse than recording none.
    import inspect

    src = inspect.getsource(mod)
    if arch["mlp_structure"] != "linear_silu_linear":
        raise ValueError(
            f"mlp_structure = {arch['mlp_structure']!r}, but essgnn.py builds "
            "one Linear-SiLU-Linear per MLP and has no field to configure "
            "anything else"
        )
    if "nn.SiLU()" not in src:
        raise ValueError(
            "mlp_structure says linear_silu_linear but essgnn.py no longer "
            "builds SiLU MLPs"
        )


def _write(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w") as fh:
        json.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Materialise the resolved Stage 2 protocols. Decides nothing.")
    ap.add_argument("--decided-by", default=None)
    args = ap.parse_args()

    if not paths.PROCTHOR_MODALITIES.exists():
        print(f"{paths.PROCTHOR_MODALITIES} not found -- run "
              "n07b_procthor_asset_modalities first", flush=True)
        return 2

    decided_by = args.decided_by or getpass.getuser()
    stamp = {"decided_by": decided_by,
             "decided_at": datetime.now(timezone.utc).isoformat()}

    with runlog.run_progress(NODE):
        mapping, skipped = build_positive_map()
        if not mapping:
            print("no eligible ProcTHOR asset has a point cloud", flush=True)
            return 2

        _write(STAGE2_PROTOCOL, {"status": "resolved", **STAGE2_DECISIONS, **stamp})
        _write(POSITIVE_MAP, mapping)
        _write(EDGE_PROTOCOL, {"status": "resolved", **EDGE_DECISIONS, **stamp})
        assert_matches_code(ARCH_DECISIONS)
        # [C1 / U-26] The arch protocol is "resolved" ONLY when the ESSGNN
        # family has been chosen. While it is None the file is written -- every
        # other decision in it is real and worth recording -- but marked
        # unresolved, so G6 blocks Stage 2 and ESSGNNConfig.from_protocol
        # refuses. Previously this said "resolved" unconditionally, which is how
        # "U-26 is blocking" and "the arch protocol is resolved" coexisted.
        arch_status = ("resolved" if ARCH_DECISIONS["architecture_family"]
                       else "unresolved_c1_architecture_family")
        _write(ARCH_PROTOCOL, {"status": arch_status, **ARCH_DECISIONS, **stamp})

    print(f"Stage 2 protocols materialised by {decided_by}")
    print(f"  positives      {len(mapping):,} assets, identity mapping")
    print(f"  excluded       {len(skipped)} with no point cloud (F26)")
    print(f"  gallery_scope  {STAGE2_DECISIONS['gallery_scope']}")
    print(f"  sampling       {STAGE2_DECISIONS['sampling_unit']}, "
          f"target removed = {STAGE2_DECISIONS['target_removed_before_essgnn']}, "
          f"unique positive per batch = {STAGE2_DECISIONS['batch_positive_uniqueness']}")
    print(f"  edges          {EDGE_DECISIONS['physical_relation_encoding']}, "
          f"{EDGE_DECISIONS['directionality']}, "
          f"missing -> {EDGE_DECISIONS['semantic_missing_representation']}")
    print(f"  architecture   d^2, {ARCH_DECISIONS['n_layers']} layers x "
          f"{ARCH_DECISIONS['hidden_dim']}, {ARCH_DECISIONS['layer_sharing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
