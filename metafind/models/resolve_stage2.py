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
    # Whether Stage 1's per-modality query masking is also applied in Stage 2.
    # The paper states scene dropout for Stage 2 and is silent on modality
    # masking there. "none" records what every Stage 2 run has done (all
    # three query modalities always present); "p_mask" is a possible ablation
    # and is not implemented. Recorded so the choice is visible, not inferred
    # from the absence of code.
    "query_modality_masking": "none",
    # [DL-103, 2026-09-06] The unit of a scene graph and of the Stage 2 context.
    # PAPER FACT: "layout-aware room-level datasets", "Each room configuration
    # provides precise spatial coordinates", "single-room indoor scenes"
    # (2methdology.tex:16,28; 3experiments.tex:8). scene_graphs BUILDER_VERSION 2
    # draws adjacency within a room and stage2.build_context_graph keeps only the
    # target's room; graphs without `graph_unit` (v1) are read as whole houses.
    "graph_unit": "room",
    # [MASTER DECISION 2026-09-03, under Kyzen's explicit delegation of the four
    # open items. IMPLEMENTATION CHOICE. Reversible; he can pin a literal.]
    #
    # WHAT KYZEN RULED, restored. DL-077 item 8: lambda init **0.1**. DL-078
    # then recorded 9.0 = 0.1 x 91.4, where 91.4 was the fused-query norm
    # measured once, on one smoke batch, through the checkpoint
    # `qpack_ti_lr2.50e-04_s20260816/stage1_best.pt`. The reasoning was sound --
    # a literal 0.1 makes the layout term 0.1% of a query whose norm is ~91, so
    # it is invisible, and the ruling's intent is plainly a tenth -- but the
    # ARTEFACT it produced was a constant welded to a retired measurement.
    #
    # That checkpoint is twelve-view-era and archived as non-comparable, and
    # DL-079 schedules a Stage 1 retrain before the reproduction-line Stage 2.
    # After it, 9.0 is a tenth of a norm that no longer exists, and nothing in
    # the code would notice: the fusion output is not normalised before the
    # loss, so its scale is a free property of whichever checkpoint is loaded.
    # The ESSGNN Block Reviewer raised this as MAJOR-1 and it is right.
    #
    # SO THE RATIO IS THE DECISION, NOT THE PRODUCT. 0.1 is Kyzen's own number,
    # stored as what he actually said. lambda_0 is derived at Stage 2 start from
    # the fused-query norm of the first kept batch, under no_grad, and BOTH the
    # ratio and the derived value are written into the run record. That makes
    # the intent survive a retrain instead of silently expiring with one
    # measurement, and it makes the number reproducible from the checkpoint
    # rather than from a ledger entry.
    #
    # The paper gives NO initial value (Eq. 6: "lambda is a learnable scalar"),
    # so choosing 0.1-of-the-query is an implementation choice either way. This
    # one is merely the version that says how it was obtained.
    "init_lambda_ratio": 0.1,
    # Kept so an existing artifact still loads and so a literal remains
    # expressible: when this is not null it WINS and no measurement is taken.
    # Null means "derive from the ratio", which is now the default.
    "init_lambda": None,
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
    # the authors themselves wrote, and the two are different models.
    #
    # DECIDED 2026-08-17: appendix_shared_msg. 2.5 is the normative Method
    # section and that counts for something, but the same paragraph carries
    # three transcription errors -- h^(0) = Concat(x, t) against the appendix's
    # own invariance premise, a width that does not close, and f_x typed to R^3
    # where the proof needs a scalar. The appendix is internally coherent, is
    # what the equivariance proof is written about, and matches the "semantic
    # extension of EGNN" the paper claims for itself.
    #
    # This is an INFERENCE about the most likely implemented architecture, NOT
    # a paper fact. The paper specifies both and never says which it ran. 2.5
    # stays implemented as `sec25_two_mlp` and must be measured against this
    # one; it is a competing hypothesis, not a fallback.
    #
    # Note what is NOT part of this fork. `j in N(i)` versus `j != i` looks like
    # it belongs here -- 2.5 uses one and the appendix the other -- but EGNN's
    # own model.tex settles it: "in this work we choose to aggregate messages
    # from all other nodes j != i, but we could limit the message exchange to a
    # given neighborhood j in N(i) if desired in both equations". It is an
    # option upstream states outright, available to EITHER family, so pairing
    # the appendix's messages with N(i) is not cross-mixing.
    "architecture_family": "appendix_shared_msg",
    "use_io_projections": True,
    "distance": "squared",
    # ESSGNNConfig's vocabulary is "updated"|"current", not "h_next".
    # [C1] Was "updated", which is 2.5's reading -- Eq. 3 passes h^{l+1} to f_x.
    # Under appendix_shared_msg there is no such choice: phi_x reads m_ij, and
    # phi_e built it from h^l. ESSGNN refuses "updated" beside this family
    # rather than ignoring it, so the protocol cannot describe a model nobody
    # ran. Switching families means switching this back.
    "coord_feat": "current",
    "layer_sharing": "independent",
    # [Kyzen 2026-09-02, notebook section 12.6 item 2: 「可以 先按照你的評估去設計」]
    # Three values change to EGNN's QM9 configuration, the task MetaFind's
    # own citation of EGNN points at (drug design). The paper gives none of
    # these numbers; each is an UPSTREAM FACT adopted with his approval, not
    # a paper fact:
    #   pooling        sum   EGNN's QM9 readout is sum-pooling
    #                        (egnn appendix "Implementation details for QM9";
    #                        repo qm9/models.py:83 torch.sum). Was mean.
    #   n_layers       7     EGNN's QM9 depth (main_qm9.py:34). The previous
    #                        4 was the N-body default, a different task.
    #   mlp_structure  egnn_appendix  phi_e ends with a Swish and phi_x's last
    #                        Linear has no bias, as EGNN's appendix specifies;
    #                        phi_h keeps MetaFind's Eq. 14 form (residual
    #                        outside). Was linear_silu_linear, which had been
    #                        recorded as "what the code happened to build".
    # hidden_dim 128 already equals QM9's; distance, projections and
    # layer_sharing are unchanged.
    # [Kyzen 2026-09-02, item 8: 甲] normalised_sum, not the plain sum
    # approved a few hours earlier: the plain sum scales with the room's
    # object count and was 27x the fused query at init. See essgnn.Pool.
    "pooling": "normalised_sum",
    "hidden_dim": 128,
    "n_layers": 7,
    "mlp_structure": "egnn_appendix",
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

    def literal_values(hint):
        """Unwrap `Literal[...] | None` as well as a bare `Literal[...]`.

        `coord_feat` became optional so its default could be resolved per
        architecture family, which turned its hint into `Optional[Literal[...]]`
        -- and `get_origin(...) is Literal` is False for that, so this check
        silently stopped validating the field it was written for. A vocabulary
        check that quietly covers less than it did is worse than one that errors.
        """
        if typing.get_origin(hint) is typing.Literal:
            return typing.get_args(hint)
        for arg in typing.get_args(hint):          # Union / Optional members
            if typing.get_origin(arg) is typing.Literal:
                return typing.get_args(arg)
        return None

    for field, value in arch.items():
        allowed = literal_values(hints.get(field))
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
                "n_layers", "mlp_structure"}
    if missing := required - arch.keys():
        raise ValueError(f"essgnn_arch_protocol is missing {sorted(missing)}")

    # C1 undecided: the round trip through ESSGNNConfig cannot run, because
    # from_protocol correctly refuses an undecided family. Skipping only that
    # step -- rather than substituting a placeholder family -- keeps this
    # function from being the place that quietly picks one.
    if arch["architecture_family"] is not None:
        # SCAFFOLDING, not a decision. This call exists to check that the seven
        # protocol fields below survive ESSGNNConfig; the three widths are
        # arbitrary inputs to that probe. The live path never reads them --
        # stage2.py takes node_feat_dim/edge_feat_dim off the artifacts
        # (procthor_node_embeddings.json, sem_edge_cache), not off this protocol.
        #
        # [U-20] Kyzen decided 2026-08-27: both encoders are OpenCLIP
        # ViT-bigG-14, 1280-d (METAFIND_NOTEBOOK.md:937). The 512s below are
        # updated to match so this probe stops advertising a superseded width --
        # but **that is cosmetic and pays no debt**: neither width is a key of
        # essgnn_arch_protocol.json, so U-20 has no machine-checked enforcement
        # point anywhere. `01_GRAPH_SPEC.md:1123` item 201 required both widths
        # in the protocol BEFORE n13 was implemented; n13 is 680 lines and they
        # are still absent. Do not read this edit as the debt being paid.
        cfg = ESSGNNConfig.from_protocol(
            {**arch, "status": "resolved"},
            node_feat_dim=1280, edge_feat_dim=1280, out_dim=1280)
        for field in ("distance", "coord_feat", "layer_sharing", "pooling",
                      "hidden_dim", "n_layers", "use_io_projections",
                      "mlp_structure"):
            if getattr(cfg, field) != arch[field]:
                raise ValueError(
                    f"essgnn_arch_protocol.{field} = {arch[field]!r} did not "
                    f"survive ESSGNNConfig, which holds {getattr(cfg, field)!r}"
                )
    # mlp_structure is now a real ESSGNNConfig field (the vocabulary check
    # above refuses values essgnn.py cannot build). What remains to assert is
    # that the code still implements the shapes the names promise: both
    # readings are built by `_mlp`, whose SiLU and optional trailing
    # activation are the whole difference between them.
    import inspect

    src = inspect.getsource(mod)
    for needle in ("nn.SiLU()", "trailing_act", "out_bias"):
        if needle not in src:
            raise ValueError(
                f"essgnn.py no longer contains {needle!r}; the mlp_structure "
                "vocabulary describes shapes the code does not build")


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

    with runlog.run_progress(NODE) as progress:
        mapping, skipped = build_positive_map()
        if not mapping:
            print("no eligible ProcTHOR asset has a point cloud", flush=True)
            progress.rc = 2
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
