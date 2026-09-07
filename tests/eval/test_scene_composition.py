"""CPU Algorithm 1 integration: real QueryTower, fusion, and tiny ESSGNN."""
from dataclasses import asdict
import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from metafind.data.semantic_edges import cache_key
from metafind.models.dual_tower import DualTowerConfig, QueryTower
from metafind.models.essgnn import ESSGNNConfig
from metafind.models.fusion import FusionConfig
from metafind.scene import compose as module


def slot(name, x, *, on=None):
    return {"new_object_id": name, "position": {"x": x, "y": 10., "z": -3.},
            "rotation": {"z_angle": 45.}, "size_in_meters": {"length": 1., "width": 2., "height": 3.},
            "placement": {"objects_in_room": ([] if on is None else
                          [{"object_id": on, "preposition": "on"}]), "room_layout_elements": []}}


@pytest.fixture
def case():
    fusion = FusionConfig(dim=2, kind="mean", include_absent_slots=False)
    arch = ESSGNNConfig(node_feat_dim=2, edge_feat_dim=1, out_dim=2,
                       use_io_projections=False, architecture_family="appendix_shared_msg",
                       hidden_dim=2, n_layers=1)
    tower = QueryTower(DualTowerConfig(dim=2, tower_sharing="fully_separate",
                                      query_fusion=fusion, gallery_fusion=fusion,
                                      essgnn=arch, init_lambda=1.)).eval()
    # Identity messages make a hand-computable real ESSGNN: h'=h, pool=mean.
    with torch.no_grad():
        for parameter in tower.layout_encoder.layers.parameters():
            parameter.zero_()
    meta = {"prompt_version": 1, "llm_model": "fixture-llm", "text_encoder_version": "fixture-text"}
    texts = {a: {"text": f"canonical node {a}", "relation_text": f"relation {a}"} for a in ("A", "B")}
    cache = {cache_key(f"relation {a}", f"relation {b}", **meta): torch.tensor([.5])
             for a in texts for b in texts}
    inputs = {"gallery_ids": ["A", "B"], "gallery": torch.eye(2),
              "asset_node_embeddings": {"A": torch.tensor([0., 4.]), "B": torch.tensor([4., 0.])},
              "asset_texts": texts, "semantic_cache": cache, "semantic_meta": meta,
              "initial_graph": {"room_id": "explicit-room", "nodes": []},
              "queries": [{"slot": slot("first", 20.), "embeds": {"text": torch.tensor([1., 0.])}, "query_text": "wanted object, not node text"},
                          {"slot": slot("second", 21., on="first"), "embeds": {"text": torch.tensor([1., 0.])}},
                          {"slot": slot("third", 22.), "embeds": {"text": torch.tensor([1., 0.])}}]}
    return tower, inputs


def test_iterative_retrieval_changes_after_placing_actual_asset(case):
    tower, inputs = case
    original = copy.deepcopy(inputs)
    state = {k: v.clone() for k, v in tower.state_dict().items()}
    seen = []
    hook = tower.layout_encoder.register_forward_pre_hook(lambda _m, args: seen.append(args[0].clone()))
    result = module.compose(tower, **inputs)
    hook.remove()
    assert [t["selected_asset_id"] for t in result["trace"]] == ["A", "B", "A"]
    assert [t["context"]["asset_ids"] for t in result["trace"]] == [[], ["A"], ["A", "B"]]
    torch.testing.assert_close(seen[0], torch.tensor([[0., 4.]]))
    torch.testing.assert_close(seen[1], torch.tensor([[0., 4.], [4., 0.]]))
    assert result["trace"][1]["lambda_layout_norm"] == 4.
    assert result["trace"][0]["selected_node_text"] == "canonical node A"
    assert result["trace"][1]["context"]["positions"] == [[20., 10., -3.]]
    assert result["final_graph"]["support"] == [[0, 1]]
    assert [n["slot"] for n in result["final_graph"]["nodes"]] == [q["slot"] for q in inputs["queries"]]
    torch.testing.assert_close(inputs["gallery"], original["gallery"], rtol=0, atol=0)
    assert inputs["initial_graph"] == original["initial_graph"]
    for key, value in tower.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert all(p.grad is None for p in tower.parameters())


def test_parallel_keeps_g0_and_same_weights(case):
    tower, inputs = case
    result = module.compose(tower, **inputs, mode="parallel")
    assert [t["selected_asset_id"] for t in result["trace"]] == ["A"] * 3
    assert all(t["context"]["node_ids"] == [] for t in result["trace"])


def test_layout_off_updates_graph_but_does_not_apply_layout(case):
    tower, inputs = case
    result = module.compose(tower, **inputs, use_layout=False)
    assert [t["selected_asset_id"] for t in result["trace"]] == ["A"] * 3
    assert [len(t["context"]["node_ids"]) for t in result["trace"]] == [0, 1, 2]
    assert all(t["lambda_layout_norm"] == 0. for t in result["trace"])


def no_layout_case(case):
    tower, inputs = case
    config = copy.deepcopy(tower.cfg)
    config.use_layout = False
    config.essgnn = None
    tower = QueryTower(config).eval()
    inputs = copy.deepcopy(inputs)
    for name in ("asset_node_embeddings", "semantic_cache", "semantic_meta"):
        inputs.pop(name)
    inputs["use_layout"] = False
    return tower, inputs


def test_no_layout_tower_keeps_physical_graph_and_text_without_semantic_inputs(case):
    tower, inputs = no_layout_case(case)
    result = module.compose(tower, **inputs)
    assert tower.layout_weight is None
    assert result["semantic_status"] == "not_used"
    assert [t["selected_asset_id"] for t in result["trace"]] == ["A"] * 3
    assert [len(t["context"]["node_ids"]) for t in result["trace"]] == [0, 1, 2]
    assert all(t["selected_node_text"] == "canonical node A" for t in result["trace"])
    assert all(t["lambda"] == t["layout_norm"] == t["lambda_layout_norm"] == 0. for t in result["trace"])
    final = result["final_graph"]
    assert final["semantic_status"] == "not_used"
    assert final["semantic_keys"] == final["semantic_missing"] == []
    assert final["support"] == [[0, 1]]
    assert final["adjacency"] == [[0, 2], [1, 2]]
    assert [n["slot"] for n in final["nodes"]] == [q["slot"] for q in inputs["queries"]]


def test_no_layout_tower_does_not_read_supplied_unused_vectors_or_cache(case):
    tower, inputs = no_layout_case(case)

    class Unused(dict):
        def __contains__(self, key):
            raise AssertionError("unused semantic data was inspected")

        def __getitem__(self, key):
            raise AssertionError("unused semantic data was read")

        def __iter__(self):
            raise AssertionError("unused semantic metadata was inspected")

    result = module.compose(tower, **inputs, asset_node_embeddings=Unused(),
                            semantic_cache=Unused(), semantic_meta=Unused())
    assert result["status"] == "complete"


def test_no_layout_tower_requires_explicit_layout_off_and_keeps_text_validation(case):
    tower, inputs = no_layout_case(case)
    inputs["use_layout"] = True
    with pytest.raises(ValueError, match="use_layout=True.*ESSGNN branch"):
        module.compose(tower, **inputs)
    inputs["use_layout"] = False
    inputs["asset_texts"]["A"]["text"] = ""
    with pytest.raises(ValueError, match="canonical node text"):
        module.compose(tower, **inputs)


@pytest.mark.parametrize("field,value,match", [
    ("asset_node_embeddings", {}, "node semantics"),
    ("semantic_cache", {}, "semantic cache lacks"),
    ("semantic_meta", {}, "semantic_meta requires"),
])
def test_full_tower_layout_off_preserves_semantic_input_checks(case, field, value, match):
    tower, inputs = case
    inputs[field] = value
    with pytest.raises((ValueError, KeyError), match=match):
        module.compose(tower, **inputs, use_layout=False)


def test_initial_graph_singleton_is_encoded_and_future_slot_not_preloaded(case):
    tower, inputs = case
    inputs["initial_graph"]["nodes"] = [{"asset_id": "B", "slot": slot("existing", 100.)}]
    result = module.compose(tower, **inputs)
    assert result["trace"][0]["context"]["node_ids"] == ["existing"]
    assert result["trace"][0]["layout_norm"] == 4.


GEOMETRY_FIELDS = ([("position", axis) for axis in ("x", "y", "z")]
                   + [("rotation", "z_angle")]
                   + [("size_in_meters", axis) for axis in ("length", "width", "height")])


@pytest.mark.parametrize("number_type", [int, float, np.float64])
def test_composed_numeric_slots_pass_the_real_placement_contract_unchanged(case, number_type):
    from metafind.scene.placement import _slot as placement_slot

    tower, inputs = case
    inputs["initial_graph"]["nodes"] = [{"asset_id": "B", "slot": slot("existing", 100.)}]
    planned = [inputs["initial_graph"]["nodes"][0]["slot"], *(q["slot"] for q in inputs["queries"])]
    for item in planned:
        for group, field in GEOMETRY_FIELDS:
            item[group][field] = number_type(item[group][field])
    result = module.compose(tower, **inputs)
    for node, original in zip(result["final_graph"]["nodes"], planned):
        placement_slot(node["slot"])
        assert node["slot"] == original
        for group, field in GEOMETRY_FIELDS:
            assert type(node["slot"][group][field]) is number_type


@pytest.mark.parametrize("group,field", GEOMETRY_FIELDS)
@pytest.mark.parametrize("value", ["1.0", True, float("nan"), float("inf"), -float("inf")])
def test_composition_refuses_unplaceable_geometry_before_encoding(case, group, field, value):
    from metafind.scene.placement import _slot as placement_slot

    tower, inputs = case
    bad = inputs["queries"][0]["slot"]
    bad[group][field] = value
    with pytest.raises(ValueError, match="finite numeric"):
        placement_slot(bad)
    # A bad future slot must be rejected before any layout or fusion forward.
    calls = []
    handles = [part.register_forward_pre_hook(lambda *_: calls.append(True))
               for part in (tower.fusion, tower.layout_encoder)]
    try:
        with pytest.raises(ValueError, match="finite numeric"):
            module.compose(tower, **inputs)
    finally:
        for handle in handles:
            handle.remove()
    assert not calls


@pytest.mark.parametrize("value", [np.float32(1), np.int64(1), torch.tensor(1.)])
def test_initial_slot_scalar_types_follow_the_existing_placement_contract(case, value):
    from metafind.scene.placement import _slot as placement_slot

    tower, inputs = case
    existing = slot("existing", value)
    inputs["initial_graph"]["nodes"] = [{"asset_id": "B", "slot": existing}]
    with pytest.raises(ValueError, match="finite numeric"):
        placement_slot(existing)
    with pytest.raises(ValueError, match="finite numeric"):
        module.compose(tower, **inputs)


def test_explicit_missing_semantic_edge_uses_learned_token(case):
    tower, inputs = case
    inputs["semantic_cache"] = dict.fromkeys(inputs["semantic_cache"], None)
    with torch.no_grad():
        tower.layout_encoder.missing_edge_token.fill_(2.5)
    seen = []
    hook = tower.layout_encoder.layers[0].register_forward_pre_hook(lambda _m, args: seen.append(args[3].clone()))
    result = module.compose(tower, **inputs)
    hook.remove()
    assert result["trace"][2]["context"]["semantic_missing"] == [True, True]
    torch.testing.assert_close(seen[-1], torch.full((2, 1), 2.5))


@pytest.mark.parametrize("fault,match", [
    ("cache", "semantic cache lacks"), ("asset", "node semantics"),
    ("slot", "slot requires explicit"), ("duplicate", "slot IDs"),
    ("future", "slot IDs"), ("relation", "unknown or identical"),
    ("gallery", "gallery IDs"), ("nan", "finite nonzero"),
    ("query", "finite floating vector"), ("train", "eval mode")])
def test_invalid_inputs_fail_explicitly(case, fault, match):
    tower, inputs = case
    if fault == "cache": inputs["semantic_cache"] = {}
    if fault == "asset": del inputs["asset_node_embeddings"]["B"]
    if fault == "slot": del inputs["queries"][0]["slot"]["position"]
    if fault == "duplicate": inputs["queries"][1]["slot"]["new_object_id"] = "first"
    if fault == "future": inputs["initial_graph"]["nodes"] = [{"asset_id": "A", "slot": inputs["queries"][0]["slot"]}]
    if fault == "relation": inputs["queries"][1]["slot"]["placement"]["objects_in_room"][0]["object_id"] = "unknown"
    if fault == "gallery": inputs["gallery_ids"] = ["A", "A"]
    if fault == "nan": inputs["gallery"][0, 0] = float("nan")
    if fault == "query": inputs["queries"][0]["embeds"]["text"] = torch.ones(3)
    if fault == "train": tower.train()
    with pytest.raises((ValueError, KeyError), match=match):
        module.compose(tower, **inputs)


def frozen_manifest(tmp_path, tower, inputs):
    def save(name, value):
        path = tmp_path / name
        torch.save(value, path)
        return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    record = {"schema": module.SCHEMA, "status": "frozen", "provenance": {"kind": "CPU synthetic fixture"},
              "model": {"config": asdict(tower.cfg), "dtype": "float32", "weights": save("tower.pt", tower.state_dict())},
              "inputs": save("inputs.pt", inputs)}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(record))
    return path


def test_cli_loads_real_tower_and_publishes_complete_result(case, tmp_path):
    tower, inputs = case
    manifest = frozen_manifest(tmp_path, tower, inputs)
    out = tmp_path / "composition.json"
    assert module.main(["--manifest", str(manifest), "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["status"] == "complete"
    assert [t["selected_asset_id"] for t in result["trace"]] == ["A", "B", "A"]
    assert result["provenance"]["device"] == "cpu"
    assert not (tmp_path / "composition.json.part").exists()
    with pytest.raises(FileExistsError):
        module.main(["--manifest", str(manifest), "--out", str(out)])


def test_manifest_refuses_changed_input_bytes(case, tmp_path):
    manifest = frozen_manifest(tmp_path, *case)
    (tmp_path / "inputs.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.run_manifest(manifest)


def test_stable_ties_keep_explicit_gallery_order(case):
    tower, inputs = case
    inputs["gallery"] = torch.ones((2, 2))
    inputs["gallery_ids"] = ["B", "A"]
    result = module.compose(tower, **inputs)
    assert result["trace"][0]["selected_asset_id"] == "B"
    assert result["trace"][0]["top5"][0]["score"] == result["trace"][0]["top5"][1]["score"]


def test_top5_trace_has_exact_cosines_and_no_sixth_candidate(case):
    tower, inputs = case
    inputs["queries"] = inputs["queries"][:1]
    inputs["gallery_ids"] = list("ABCDEF")
    inputs["gallery"] = torch.tensor([[1., 0.], [.8, .2], [.6, .4], [.4, .6], [.2, .8], [0., 1.]])
    for asset in "CDEF":
        inputs["asset_node_embeddings"][asset] = torch.ones(2)
        inputs["asset_texts"][asset] = {"text": asset}
    result = module.compose(tower, **inputs)
    top = result["trace"][0]["top5"]
    assert [row["asset_id"] for row in top] == list("ABCDE")
    for row, vector in zip(top, inputs["gallery"].double()):
        assert row["score"] == pytest.approx(float(vector[0] / vector.norm()), abs=1e-14)


def test_missing_prefusion_norm_cannot_silently_change_retrieved_asset(case, tmp_path):
    tower, inputs = case
    tower.cfg.query_fusion.prefusion_norm = True
    inputs["queries"] = inputs["queries"][:1]
    inputs["queries"][0]["embeds"] = {"text": torch.tensor([100., 0.]), "image": torch.tensor([0., 1.])}
    inputs["gallery"] = torch.tensor([[1., 1.], [1., 0.]])
    path = frozen_manifest(tmp_path, tower, inputs)
    assert module.run_manifest(path)["trace"][0]["selected_asset_id"] == "A"
    manifest = json.loads(path.read_text())
    del manifest["model"]["config"]["query_fusion"]["prefusion_norm"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete dataclass.*prefusion_norm"):
        module.run_manifest(path)


@pytest.mark.parametrize("section,field", [(None, "init_lambda"), ("essgnn", "pooling"),
                                          ("gallery_fusion", "include_absent_slots")])
def test_all_nested_configs_require_complete_fields(case, tmp_path, section, field):
    path = frozen_manifest(tmp_path, *case)
    manifest = json.loads(path.read_text())
    config = manifest["model"]["config"]
    del (config[section] if section else config)[field]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete dataclass"):
        module.run_manifest(path)


def test_cli_refuses_float64_state_before_lambda_narrowing(case, tmp_path):
    tower, inputs = case
    tower.double()
    with torch.no_grad():
        tower.layout_weight.fill_(1. + 1e-8)
    inputs["initial_graph"]["nodes"] = [{"asset_id": "B", "slot": slot("existing", 100.)}]
    inputs["queries"] = inputs["queries"][:1]
    inputs["queries"][0]["embeds"] = {"text": torch.tensor([-4., 1e-10], dtype=torch.float64)}
    assert module.compose(tower, **inputs)["trace"][0]["selected_asset_id"] == "A"
    path = frozen_manifest(tmp_path, tower, inputs)
    with pytest.raises(ValueError, match="float32 state only"):
        module.run_manifest(path)


def test_cli_requires_explicit_model_dtype(case, tmp_path):
    path = frozen_manifest(tmp_path, *case)
    manifest = json.loads(path.read_text())
    del manifest["model"]["dtype"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="model.dtype"):
        module.run_manifest(path)


def test_cosine_does_not_clip_tiny_valid_norms(case):
    tower, inputs = case
    inputs["queries"] = inputs["queries"][:1]
    inputs["queries"][0]["embeds"] = {"text": torch.tensor([1e-15, 0.])}
    inputs["gallery"] = torch.tensor([[1e-15, 0.], [.6, .8]])
    result = module.compose(tower, **inputs)
    top = result["trace"][0]["top5"]
    assert top[0] == {"asset_id": "A", "score": 1.}
    assert top[1]["score"] == pytest.approx(.6, abs=5e-8)


def existing_checkpoint_records(case, tmp_path, monkeypatch):
    """Real loaders, metadata, states, and overlays; only model width is tiny."""
    from metafind.models import ulip_backbone
    from metafind.models.dual_tower import MetaFindDualTower
    from metafind.train import stage1

    tower, inputs = case
    monkeypatch.setattr(ulip_backbone, "EMBED_DIM", 2)
    monkeypatch.setattr(stage1, "load_protocols", lambda: ({}, {}, {"values": {}}))
    def forbid_backbone(*args, **kwargs):
        raise AssertionError("model-only export must not construct a backbone")
    monkeypatch.setattr(ulip_backbone, "ULIPBackbone", forbid_backbone)
    parent_model = MetaFindDualTower(DualTowerConfig(dim=2, tower_sharing="fully_separate",
                           use_layout=False, query_fusion=tower.cfg.query_fusion,
                           gallery_fusion=tower.cfg.gallery_fusion))
    encoding = {"actual_clip_train_scope": "frozen", "missing_modality_representation": "learned_token"}
    training = {"fusion": "mean", "tower_sharing": "fully_separate", "train_scope": "fuser_only"}
    hp = {"values": {"learnable_temperature": False, "init_temperature": .5, "max_logit_scale": 100.}}
    snapshot = stage1.model_input_snapshot(encoding, training, hp, parent_model)
    parent_meta = {"model_inputs": snapshot}
    parent_path = tmp_path / "s1.pt"
    torch.save({"tower_trainable_state": parent_model.state_dict(), "metadata": parent_meta}, parent_path)
    parent_rec = {**parent_meta, "uri": str(parent_path), "sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest()}
    parent_json = tmp_path / "s1.json"
    parent_json.write_text(json.dumps(parent_rec))
    child_meta = {"stage1_checkpoint_sha256": parent_rec["sha256"],
                  "lambda_init": {"init_lambda": 1.} if tower.cfg.use_layout else None,
                  "stage1_model_inputs": snapshot, "stage2_protocol": {"asset_modalities": ["text", "image"]},
                  "variant_id": "full" if tower.cfg.use_layout else "no_layout",
                  "use_layout": tower.cfg.use_layout}
    if tower.cfg.use_layout:
        child_meta.update(arch_protocol={**asdict(tower.cfg.essgnn), "status": "resolved"},
                          layout_input_dims={"node_feat_dim": 2, "edge_feat_dim": 1})
    child_path = tmp_path / "s2.pt"
    torch.save({"trainable_state": {"query." + k: v for k, v in tower.state_dict().items()},
                "metadata": child_meta}, child_path)
    child_json = tmp_path / "s2.json"
    child_json.write_text(json.dumps({**child_meta, "uri": str(child_path),
                                     "sha256": hashlib.sha256(child_path.read_bytes()).hexdigest()}))
    return parent_json, child_json


def test_export_existing_s1_s2_model_then_real_manifest_composition(case, tmp_path, monkeypatch):
    parent, child = existing_checkpoint_records(case, tmp_path, monkeypatch)
    export_dir = tmp_path / "export"
    assert module.main(["export-model", "--stage1-record", str(parent), "--stage2-record", str(child),
                        "--out-dir", str(export_dir)]) == 0
    model = json.loads((export_dir / "model.json").read_text())
    assert model["dtype"] == "float32"
    assert model["provenance"]["stage1_checkpoint"]["sha256"] == json.loads(parent.read_text())["sha256"]
    assert model["provenance"]["records"][1]["sha256"] == hashlib.sha256(child.read_bytes()).hexdigest()
    path = frozen_manifest(tmp_path, *case)
    manifest = json.loads(path.read_text())
    manifest["model"] = model
    path.write_text(json.dumps(manifest))
    result = module.run_manifest(path)
    assert [t["selected_asset_id"] for t in result["trace"]] == ["A", "B", "A"]
    exported = torch.load(model["weights"]["path"], weights_only=True)
    for key, value in case[0].state_dict().items():
        torch.testing.assert_close(exported[key], value, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        module.export_query_model(parent, child, export_dir)


def test_model_export_refuses_wrong_parent(case, tmp_path, monkeypatch):
    parent, child = existing_checkpoint_records(case, tmp_path, monkeypatch)
    record = json.loads(child.read_text())
    record["stage1_checkpoint_sha256"] = "f" * 64
    child.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="fine-tuned from"):
        module.export_query_model(parent, child, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_export_no_layout_checkpoint_roundtrips_manifest_and_composition(case, tmp_path, monkeypatch):
    case = no_layout_case(case)
    parent, child = existing_checkpoint_records(case, tmp_path, monkeypatch)
    export_dir = tmp_path / "export"
    assert module.main(["export-model", "--stage1-record", str(parent), "--stage2-record", str(child),
                        "--variant", "no_layout", "--out-dir", str(export_dir)]) == 0
    model = json.loads((export_dir / "model.json").read_text())
    assert model["config"]["use_layout"] is False
    assert model["config"]["essgnn"] is None
    exported = torch.load(model["weights"]["path"], weights_only=True)
    assert not any(k.startswith("layout_encoder.") or k == "layout_weight" for k in exported)
    for key, value in case[0].state_dict().items():
        torch.testing.assert_close(exported[key], value, rtol=0, atol=0)
    path = frozen_manifest(tmp_path, *case)
    manifest = json.loads(path.read_text())
    manifest["model"] = model
    path.write_text(json.dumps(manifest))
    result = module.run_manifest(path)
    assert result["semantic_status"] == "not_used"
    assert [t["selected_asset_id"] for t in result["trace"]] == ["A"] * 3
    assert result["final_graph"]["support"] == [[0, 1]]
    assert all(t["lambda"] == 0. for t in result["trace"])


def test_no_layout_manifest_cannot_enable_missing_branch(case, tmp_path):
    tower, inputs = no_layout_case(case)
    inputs["use_layout"] = True
    path = frozen_manifest(tmp_path, tower, inputs)
    with pytest.raises(ValueError, match="use_layout=True.*ESSGNN branch"):
        module.run_manifest(path)
    manifest = json.loads(path.read_text())
    manifest["model"]["config"]["use_layout"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="use_layout=True.*complete ESSGNNConfig"):
        module.run_manifest(path)
