"""Tests for n09b_resolve_stage2_protocol.

The one that matters is assert_matches_code. A protocol value the code cannot
honour is not caught until Stage 2 builds an ESSGNNConfig -- hours after Stage 1
finished training. Writing "h_next" instead of "updated" was exactly that, and
it was caught here rather than there.
"""

from __future__ import annotations

import json

import pytest

from metafind.models.resolve_stage2 import (
    ARCH_DECISIONS,
    EDGE_DECISIONS,
    STAGE2_DECISIONS,
    assert_matches_code,
    build_positive_map,
)


# --- the protocol must be one the code can consume -------------------------

def test_the_recorded_architecture_survives_essgnn_config():
    assert_matches_code(ARCH_DECISIONS)


@pytest.mark.parametrize("field,bad", [
    ("coord_feat", "h_next"),        # the real mistake: right meaning, wrong word
    ("distance", "l2"),              # 2.5's prose word, not the config's
    ("layer_sharing", "tied"),
    ("pooling", "average"),
])
def test_a_value_outside_the_configs_vocabulary_is_refused(field, bad):
    """[the injection] Each of these reads correctly to a human and is rejected
    by ESSGNNConfig -- at Stage 2, if nothing checks here."""
    with pytest.raises(Exception):
        assert_matches_code({**ARCH_DECISIONS, field: bad})


def test_every_field_from_protocol_requires_is_present():
    """from_protocol names its required set; a protocol missing one of them
    fails at Stage 2, so it must fail here."""
    for field in ("use_io_projections", "distance", "coord_feat",
                  "layer_sharing", "pooling", "hidden_dim", "n_layers"):
        partial = {k: v for k, v in ARCH_DECISIONS.items() if k != field}
        with pytest.raises(Exception):
            assert_matches_code(partial)


def test_mlp_structure_is_checked_against_the_code_not_just_recorded():
    """[U-35] ESSGNNConfig has no such field, so the string describes the code.
    A recorded value nothing reads is how U-14 and U-11 were being decided by
    dataclass defaults before n05b existed."""
    with pytest.raises(ValueError):
        assert_matches_code({**ARCH_DECISIONS, "mlp_structure": "linear_relu_linear"})


# --- the decisions themselves ---------------------------------------------

def test_the_distance_follows_appendix_c_not_the_prose():
    """[U-17] 2.5's prose says the L2 norm; Appendix C's Eq. 10-12 and the
    reference EGNN square it. They disagree and Appendix C carries the proof."""
    assert ARCH_DECISIONS["distance"] == "squared"


def test_physical_edges_only_set_the_neighbourhood():
    """[U-29] f_h and f_x take exactly ONE edge argument, and Appendix C defines
    it as the LLM sentence. Neighbourhood-only is the reading under which the
    published tensor signature is literally correct."""
    assert EDGE_DECISIONS["physical_relation_encoding"] == "neighbourhood_only"


def test_a_missing_semantic_edge_is_not_zero():
    """[U-30] f_h's width is fixed, so the slots need filling; zero is
    indistinguishable from a real embedding, which is L1-SEMEDGE-NO-ZEROFILL."""
    assert EDGE_DECISIONS["semantic_missing_representation"] == "learned_missing_token"
    assert "zero" not in EDGE_DECISIONS["semantic_missing_representation"]


def test_edges_are_symmetric_matching_what_n07_stored():
    """[U-19] n07 writes unordered pairs; a directed reading here would ask
    ESSGNN for edges the scene graphs do not contain."""
    from metafind.data.scene_graphs import build_scene_graph

    assert EDGE_DECISIONS["directionality"] == "symmetric"
    g = build_scene_graph(
        {"rooms": [{"id": "room|0", "roomType": "Bedroom"}],
         "objects": [{"assetId": "Bed_1", "id": "Bed|0|0",
                      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                      "children": [{"assetId": "Pillow_1", "id": "Pillow|s|0|0",
                                    "position": {"x": 0.1, "y": 0.3, "z": 0.0}}]}]},
        "h0")
    assert all(a < b for a, b in g["phys_edges"]["support"])


def test_the_target_is_removed_before_the_layout_encoder_sees_the_graph():
    """[U-08d] The load-bearing one: leaving the target in lets ESSGNN see the
    answer, the loss falls, and nothing distinguishes that from learning."""
    assert STAGE2_DECISIONS["target_removed_before_essgnn"] is True


def test_a_batch_admits_each_positive_once():
    """[U-08e] 665,320 instances over 1,467 assets; at batch 64, 99.3% of
    batches would otherwise carry a duplicate whose gallery embedding is
    bit-identical to the positive's."""
    assert STAGE2_DECISIONS["batch_positive_uniqueness"] is True


def test_the_positive_is_the_asset_itself():
    """[U-08a] No ProcTHOR-to-Objaverse correspondence exists or is needed."""
    assert STAGE2_DECISIONS["positive_identity"] == "same_asset_id"
    assert STAGE2_DECISIONS["gallery_scope"] == "procthor"


def test_scene_dropout_is_per_batch():
    """[U-32] 2.6 says the layout vector is omitted in 30% of BATCHES."""
    assert STAGE2_DECISIONS["scene_dropout_granularity"] == "batch"


# --- the positive map ------------------------------------------------------

def modality(tmp_path, asset_id: str, with_cloud: bool):
    rec = {"asset_id": asset_id,
           "pointcloud_uri": str(tmp_path / f"{asset_id}.npz") if with_cloud else None,
           "pointcloud_missing_reason": None if with_cloud else "no depth returned"}
    (tmp_path / f"{asset_id}.json").write_text(json.dumps(rec))


def test_every_eligible_asset_maps_to_itself(monkeypatch, tmp_path):
    import metafind.models.resolve_stage2 as r

    for a in ("Bed_1", "Chair_7"):
        modality(tmp_path, a, True)
    monkeypatch.setattr(r.paths, "PROCTHOR_MODALITIES", tmp_path)
    mapping, skipped = build_positive_map()
    assert skipped == []
    assert all(v["positive_asset_id"] == k for k, v in mapping.items())
    assert all(v["method"] == "identity" for v in mapping.values())


def test_an_asset_without_a_point_cloud_gets_no_positive(monkeypatch, tmp_path):
    """[F26] 2.6 needs a modality-complete gallery, so n11b excludes these.
    Writing a positive here would name an id stage2_gallery_index does not
    contain -- a loss with a name and no vector behind it."""
    import metafind.models.resolve_stage2 as r

    modality(tmp_path, "Bed_1", True)
    modality(tmp_path, "Bowl_11", False)
    monkeypatch.setattr(r.paths, "PROCTHOR_MODALITIES", tmp_path)
    mapping, skipped = build_positive_map()
    assert "Bowl_11" not in mapping
    assert skipped == ["Bowl_11"]


def test_the_exclusion_is_reported_not_silent(monkeypatch, tmp_path):
    """Dropping assets quietly is how a corpus shrinks without anyone noticing."""
    import metafind.models.resolve_stage2 as r

    for i in range(3):
        modality(tmp_path, f"Bowl_{i}", False)
    monkeypatch.setattr(r.paths, "PROCTHOR_MODALITIES", tmp_path)
    mapping, skipped = build_positive_map()
    assert mapping == {}
    assert len(skipped) == 3


# --- C1 / U-26: the protocol must carry the architecture choice --------------

def test_the_architecture_family_is_recorded_and_is_the_appendix():
    """[C1 / U-26] DECIDED 2026-08-17: the appendix's shared-message form.

    A value here is a claim, so it has to be the one the audit records. Before
    this key existed, ESSGCL built the two-MLP layer unconditionally while C
    called U-26 unresolved -- both true at once, and G6 could not gate it.
    """
    assert ARCH_DECISIONS["architecture_family"] == "appendix_shared_msg", (
        "changing the primary family means updating "
        "docs/audit/C_PAPER_CONTRADICTIONS.md#c1 and saying who decided and why")


def test_the_appendix_family_forces_coord_feat_current():
    """phi_x reads m_ij, which phi_e built from h^l. 2.5's `updated` names a
    choice this family does not have, so recording it would describe a model
    nobody ran."""
    assert ARCH_DECISIONS["coord_feat"] == "current"


def test_an_undecided_family_still_refuses_to_build_a_config():
    from metafind.models.essgnn import ESSGNNConfig

    proto = {**ARCH_DECISIONS, "status": "resolved", "architecture_family": None}
    with pytest.raises(ValueError, match="architecture_family"):
        ESSGNNConfig.from_protocol(proto, node_feat_dim=8, edge_feat_dim=8, out_dim=8)


def test_an_unknown_family_is_refused():
    from metafind.models.essgnn import ESSGNNConfig

    proto = {**ARCH_DECISIONS, "status": "resolved",
             "architecture_family": "some_third_thing"}
    with pytest.raises(ValueError, match="architecture_family"):
        ESSGNNConfig.from_protocol(proto, node_feat_dim=8, edge_feat_dim=8, out_dim=8)


def test_the_competing_hypothesis_is_still_buildable():
    """2.5 is a competing hypothesis, not a fallback. If it stops building, the
    comparison C1 turns on cannot be run."""
    from metafind.models.essgnn import ESSGNNConfig

    proto = {**ARCH_DECISIONS, "status": "resolved",
             "architecture_family": "sec25_two_mlp", "coord_feat": "updated"}
    cfg = ESSGNNConfig.from_protocol(proto, node_feat_dim=8, edge_feat_dim=8, out_dim=8)
    assert cfg.architecture_family == "sec25_two_mlp"
    assert cfg.coord_feat == "updated"


def test_assert_matches_code_still_checks_everything_else_while_c1_is_open():
    """The early return must skip ONLY the config round trip."""
    with pytest.raises(ValueError, match="mlp_structure|not one of"):
        assert_matches_code({**ARCH_DECISIONS, "mlp_structure": "linear_relu_linear"})
    with pytest.raises(ValueError, match="not one of"):
        assert_matches_code({**ARCH_DECISIONS, "distance": "l2"})
