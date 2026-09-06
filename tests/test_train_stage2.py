"""Tests for n13_train_stage2's pure half.

n13 cannot be smoke-tested end to end yet: it needs stage1_ckpt (n10 has not
run) and sem_edge_cache (n08 has not run). What IS testable is the part that
encodes the decisions -- the batcher that enforces U-08e and the context graph
that enforces U-08d -- and those are the two that would fail silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from metafind.train.stage2 import (
    build_context_graph,
    unique_positive_batches,
    verify_recorded_artifact,
)


def samples(pattern: list[str]) -> list[tuple[str, int, str]]:
    """`pattern` gives each sample's assetId; house and index are incidental."""
    return [(f"h{i//5:03d}", i, a) for i, a in enumerate(pattern)]


# --- downstream artifact integrity -----------------------------------------

def test_stage2_refuses_an_index_record_pointed_at_wrong_bytes(tmp_path):
    import hashlib

    index = tmp_path / "index.npz"
    index.write_bytes(b"the bytes that were recorded")
    record = {"uri": str(index),
              "sha256": hashlib.sha256(index.read_bytes()).hexdigest()}
    index.write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="changed since it was recorded"):
        verify_recorded_artifact(record, "gallery index", "Rebuild n11.")


def test_stage2_refuses_an_unverifiable_or_missing_index(tmp_path):
    index = tmp_path / "index.npz"
    index.write_bytes(b"x")
    with pytest.raises(ValueError, match="no sha256"):
        verify_recorded_artifact({"uri": str(index)}, "gallery index", "Rebuild n11.")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        verify_recorded_artifact(
            {"uri": str(tmp_path / "gone.npz"), "sha256": "0" * 64},
            "gallery index", "Rebuild n11.")


def test_stage2_accepts_index_bytes_matching_the_record(tmp_path):
    import hashlib

    index = tmp_path / "index.npz"
    index.write_bytes(b"stable")
    record = {"uri": str(index),
              "sha256": hashlib.sha256(index.read_bytes()).hexdigest()}
    assert verify_recorded_artifact(record, "gallery index", "Rebuild n11.") == index


# --- U-08e: no duplicate positive in a batch -------------------------------

def test_no_batch_contains_the_same_asset_twice():
    """[U-08e] A frozen encoder gives one assetId ONE embedding, so a duplicate
    is a negative bit-identical to the positive: the gradient would ask the
    model to separate two identical vectors."""
    s = samples(["A", "B", "C", "A", "B", "C", "A", "D"])
    for batch in unique_positive_batches(s, 3, np.random.default_rng(0)):
        assets = [s[i][2] for i in batch]
        assert len(assets) == len(set(assets)), assets


def test_it_holds_on_the_real_frequency_shape():
    """The measured corpus is 665,320 instances over 1,467 assets, so a few
    assets recur thousands of times. A batcher that works on a uniform pattern
    and not on a skewed one would pass a weaker test than reality."""
    skewed = ["A"] * 60 + ["B"] * 30 + [f"X{i}" for i in range(40)]
    s = samples(skewed)
    for batch in unique_positive_batches(s, 16, np.random.default_rng(1)):
        assets = [s[i][2] for i in batch]
        assert len(assets) == len(set(assets))


def test_every_sample_is_used_not_dropped():
    """Deferring a colliding sample rather than discarding it. Dropping would
    silently reweight the corpus toward rare assets -- a different experiment,
    reported under the same name."""
    s = samples(["A", "B", "A", "B", "A", "C"])
    used = [i for batch in unique_positive_batches(s, 2, np.random.default_rng(2))
            for i in batch]
    assert sorted(used) == list(range(len(s)))


def test_a_single_asset_yields_single_sample_batches():
    """The degenerate case: if every sample shares one asset, uniqueness forces
    batch size 1 rather than an infinite loop."""
    s = samples(["A"] * 5)
    batches = unique_positive_batches(s, 4, np.random.default_rng(3))
    assert all(len(b) == 1 for b in batches)
    assert sum(len(b) for b in batches) == 5


def test_batching_is_seeded():
    s = samples(["A", "B", "C", "D", "E", "F", "A", "B"])
    a = unique_positive_batches(s, 3, np.random.default_rng(7))
    b = unique_positive_batches(s, 3, np.random.default_rng(7))
    assert a == b


def test_a_different_seed_gives_a_different_order():
    s = samples([f"X{i}" for i in range(40)])
    a = unique_positive_batches(s, 8, np.random.default_rng(1))
    b = unique_positive_batches(s, 8, np.random.default_rng(2))
    assert a != b


# --- U-08d: the target leaves the graph ------------------------------------

def graph() -> dict:
    return {
        "nodes": [{"index": i, "asset_id": f"A{i}", "position": [float(i), 0.0, 0.0]}
                  for i in range(4)],
        "sem_edge_ids": [[0, 1], [1, 2], [2, 3], [0, 3]],
    }


def data_bits():
    text_map = {f"A{i}": f"a thing {i}" for i in range(4)}
    text_map["_meta"] = {"prompt_version": 1, "llm_model": "m",
                         "text_encoder_version": "v"}
    return {}, text_map


def _key(text_map: dict, a: str, b: str) -> str:
    from metafind.train.stage2 import _edge_key
    return _edge_key(a, b, text_map)


def test_the_target_node_is_absent_from_the_context():
    """[U-08d] The load-bearing one. Leaving the target in lets ESSGNN read the
    answer off its own input: the loss falls and nothing distinguishes that from
    learning."""
    sem, text = data_bits()
    keep, pos, edge_index, edge_attr, edge_missing = build_context_graph(graph(), 2, 4, sem, text)
    assert [n["index"] for n in keep] == [0, 1, 3]
    assert pos.shape == (3, 3)


def test_every_edge_touching_the_target_is_removed():
    sem, text = data_bits()
    _, _, edge_index, _, edge_missing = build_context_graph(graph(), 2, 4, sem, text)
    # [1,2] and [2,3] go; [0,1] and [0,3] remain, each stored both ways
    assert edge_index.shape[1] == 4


def test_node_indices_are_remapped_not_left_with_holes():
    """ESSGNN indexes edges positionally. A hole would make every edge past the
    target point at the wrong node, and every shape would still be valid."""
    sem, text = data_bits()
    keep, _, edge_index, _, edge_missing = build_context_graph(graph(), 1, 4, sem, text)
    assert edge_index.max() < len(keep)
    assert set(edge_index.flatten().tolist()) <= set(range(len(keep)))


def test_edges_are_stored_symmetrically():
    """[U-19] Matching what n07 wrote. A directed reading would ask ESSGNN for
    edges the scene graphs do not contain."""
    sem, text = data_bits()
    _, _, edge_index, _, edge_missing = build_context_graph(graph(), 2, 4, sem, text)
    pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    assert all((b, a) in pairs for a, b in pairs)


def test_a_missing_semantic_edge_is_MARKED_not_filled_here():
    """[U-30] The builder reports absence; ESSGNN substitutes a LEARNED token.

    This used to assert that the builder wrote a token into `edge_attr`. It did,
    and the token was a seeded numpy constant -- so `essgnn_edge_protocol`'s
    `semantic_missing_representation = learned_missing_token` described
    something no optimizer ever touched. The mask is what lets the substitution
    happen against an nn.Parameter instead; the zeros left behind never reach an
    MLP, and test_marked_edges_actually_use_the_token_not_the_passed_row in
    test_essgnn.py is what proves it.
    """
    sem, text = data_bits()
    _, _, _, edge_attr, edge_missing = build_context_graph(graph(), 2, 4, sem, text)
    assert edge_attr.shape[1] == 4
    assert edge_missing.shape == (edge_attr.shape[0],)
    assert edge_missing.all(), "sem_cache is empty here, so every edge is missing"


def test_the_missing_mask_lines_up_with_edge_attr_rows():
    """A mask one row out would swap a real relation for the token silently."""
    sem, text = data_bits()
    sem[_key(text, "A0", "A1")] = np.arange(4, dtype=np.float32)
    _, _, edge_index, edge_attr, edge_missing = build_context_graph(
        graph(), 2, 4, sem, text)
    assert edge_missing.shape == (edge_index.shape[1],)
    present = ~edge_missing
    assert present.sum() == 2, "the cached A0-A1 edge should be present both ways"
    assert np.allclose(edge_attr[present][0], np.arange(4, dtype=np.float32))


def test_edge_attr_rows_match_edge_count():
    """A mismatch here is a silent misalignment: ESSGNN would pair edge k with
    attribute k and both arrays would look fine."""
    sem, text = data_bits()
    for target in range(4):
        _, _, edge_index, edge_attr, edge_missing = build_context_graph(
            graph(), target, 4, sem, text)
        assert edge_attr.shape[0] == edge_index.shape[1]


def test_removing_every_neighbour_leaves_an_empty_edge_set_not_a_crash():
    g = {"nodes": [{"index": 0, "asset_id": "A0", "position": [0.0, 0.0, 0.0]},
                   {"index": 1, "asset_id": "A1", "position": [1.0, 0.0, 0.0]}],
         "sem_edge_ids": [[0, 1]]}
    sem, text = data_bits()
    keep, _, edge_index, edge_attr, edge_missing = build_context_graph(g, 1, 4, sem, text)
    assert len(keep) == 1
    assert edge_index.shape == (2, 0)
    assert edge_attr.shape == (0, 4)


# --- Table 3 variants: --variant must reach the model, not just the filename --

def registry_file(tmp_path, monkeypatch, rows):
    import json as _json
    from metafind import paths as _paths
    monkeypatch.setattr(_paths, "OUTPUTS", tmp_path)
    (tmp_path / "variant_registry.json").write_text(_json.dumps(rows))


FULL = {"variant_id": "full", "requires_training": True, "reuses_ckpt": None,
        "train_scope": "point_encoder_and_fuser", "fusion": None,
        "dropout": 0.30, "layout_encoder": "essgnn",
        "composition_mode": "iterative"}
CKPT = {"train_scope": "point_encoder_and_fuser"}


def test_a_known_variant_resolves(tmp_path, monkeypatch):
    from metafind.train.stage2 import load_variant
    registry_file(tmp_path, monkeypatch, [FULL])
    assert load_variant("full", CKPT)["layout_encoder"] == "essgnn"


def test_no_layout_turns_the_branch_off(tmp_path, monkeypatch):
    """The row that made the old bug invisible: it trained the FULL model and
    saved it as stage2_no_layout.pt."""
    from metafind.train.stage2 import load_variant
    row = {**FULL, "variant_id": "no_layout", "layout_encoder": None}
    registry_file(tmp_path, monkeypatch, [row])
    assert load_variant("no_layout", CKPT)["layout_encoder"] is None


def test_an_unknown_variant_is_refused(tmp_path, monkeypatch):
    from metafind.train.stage2 import load_variant
    registry_file(tmp_path, monkeypatch, [FULL])
    with pytest.raises(ValueError, match="unknown variant"):
        load_variant("typo_here", CKPT)


def test_the_inference_only_row_refuses_to_train(tmp_path, monkeypatch):
    """[L1-ABLATION-INFERENCE-ONLY] Algorithm 1 runs at inference, so
    'w/o iterative retrieval' is the Full checkpoint evaluated differently."""
    from metafind.train.stage2 import load_variant
    row = {**FULL, "variant_id": "no_iterative", "requires_training": False,
           "reuses_ckpt": "full", "train_scope": None,
           "composition_mode": "parallel"}
    registry_file(tmp_path, monkeypatch, [row])
    with pytest.raises(ValueError, match="INFERENCE setting"):
        load_variant("no_iterative", CKPT)


def test_the_gat_row_is_refused_rather_than_run_with_essgnn(tmp_path, monkeypatch):
    from metafind.train.stage2 import load_variant
    row = {**FULL, "variant_id": "layout_gat", "layout_encoder": "gat"}
    registry_file(tmp_path, monkeypatch, [row])
    with pytest.raises(NotImplementedError, match="GAT"):
        load_variant("layout_gat", CKPT)


def test_a_stage1_field_mismatch_is_caught_not_applied(tmp_path, monkeypatch):
    """`Train fuser only` is a STAGE 1 setting. Stage 2 cannot apply it after
    the fact -- it can only refuse a checkpoint trained the other way."""
    from metafind.train.stage2 import load_variant
    row = {**FULL, "variant_id": "fuser_only", "train_scope": "fuser_only"}
    registry_file(tmp_path, monkeypatch, [row])
    with pytest.raises(ValueError, match="Re-run n10"):
        load_variant("fuser_only", CKPT)


def test_the_degenerate_tail_of_unique_positive_batching_is_dropped():
    """[MEASURED 2026-09-04] the first real Stage 2 run ended with 300 steps of
    'loss 0.0000': batches of one or two samples left over from the most-placed
    assets. InfoNCE over one class is identically zero."""
    from metafind.train.stage2 import MIN_BATCH, usable_batches
    batches = [list(range(64))] * 3 + [[1], [2, 3], list(range(7)), list(range(8))]
    kept, n_b, n_s = usable_batches(batches)
    assert MIN_BATCH == 8
    assert len(kept) == 4 and all(len(b) >= 8 for b in kept)
    assert (n_b, n_s) == (3, 1 + 2 + 7)


def test_the_stage2_tower_uses_the_stage1_fusion_config():
    """[BUG 2026-09-04] build_stage2_model dropped prefusion_norm / image_tokens,
    so a parent trained with prefusion_norm=True got its fusion fed raw vectors
    in Stage 2 and in the w/ ESSGNN evaluation."""
    import json
    from metafind.train.stage1 import build_model
    from metafind.train.stage2 import build_stage2_model
    encoding = {"missing_modality_representation": "learned_token"}
    training = {"fusion": "transformer", "tower_sharing": "shared_backbone_separate_fusion",
                "prefusion_norm": True, "image_tokens": 1}
    hyper = {"values": {"learnable_temperature": False, "init_temperature": 0.5,
                        "max_logit_scale": 100.0}}
    arch = {"status": "resolved", "architecture_family": "appendix_shared_msg",
            "use_io_projections": True, "distance": "squared", "coord_feat": "current",
            "layer_sharing": "independent", "pooling": "normalised_sum", "hidden_dim": 16,
            "n_layers": 1, "mlp_structure": "egnn_appendix"}
    s1, _ = build_model(encoding, training, hyper)
    s2 = build_stage2_model(encoding, training, hyper, arch, node_feat_dim=8,
                            edge_feat_dim=8, use_layout=True, init_lambda=1.0)
    for tower in ("query", "gallery"):
        c1, c2 = getattr(s1, tower).fusion.cfg, getattr(s2, tower).fusion.cfg
        assert (c1.prefusion_norm, c1.image_tokens, c1.kind, c1.zero_pad) == \
               (c2.prefusion_norm, c2.image_tokens, c2.kind, c2.zero_pad), tower
    assert s2.query.fusion.cfg.prefusion_norm is True


def test_query_present_modes():
    import numpy as np
    from metafind.train.stage2 import query_present, QUERY_MASKING_MODES
    assert QUERY_MASKING_MODES == ("none", "text_only", "stage1")
    rng = np.random.default_rng(0)
    assert query_present("none", rng) is None
    t = query_present("text_only", rng)
    assert t.shape == (1, 3) and t.tolist() == [[True, False, False]]
    masks = [query_present("stage1", rng) for _ in range(300)]
    assert all(m.shape == (1, 3) and m.any() for m in masks)
    rate = 1 - sum(m.float().mean().item() for m in masks) / len(masks)
    assert 0.15 < rate < 0.45          # ~30% absent, minus the at-least-one rule


def test_room_unit_graph_limits_the_context_to_the_targets_room():
    """[DL-103] The paper's scenes are single rooms; the context is the room, not the house."""
    sem, text = data_bits()
    g = graph()
    g["graph_unit"] = "room"
    for n in g["nodes"]:
        n["room_id"] = "room|0" if n["index"] in (0, 2) else "room|1"
    keep, pos, edge_index, _, _ = build_context_graph(g, 2, 4, sem, text)
    assert [n["index"] for n in keep] == [0]
    assert pos.shape == (1, 3)
    assert edge_index.max() < len(keep) if edge_index.size else True


def test_legacy_house_graph_keeps_the_whole_house():
    sem, text = data_bits()
    keep, *_ = build_context_graph(graph(), 2, 4, sem, text)
    assert [n["index"] for n in keep] == [0, 1, 3]
