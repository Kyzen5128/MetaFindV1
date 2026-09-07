"""Small CPU checks for the custom retrieval protocol's executed observations.

No pretrained models, network, corpus, or GPU are used. Arrays and manifests
are synthetic and temporary; numerical expectations are fixed independently.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from metafind.eval import custom_table1 as custom
from metafind.models.dual_tower import DualTowerConfig, MetaFindDualTower
from metafind.models.fusion import FusionConfig


def source(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class FakeBackbone:
    def __init__(self, pc_offset=0.):
        self.pc_offset = pc_offset
        self.pc_inputs = []
        self.text_inputs = []

    def tokenizer(self, texts):
        # The final token represents the alternative description. Distinct
        # strings remain distinct here; the truncation negative overrides this.
        return torch.tensor([[ord(t[0]), int(t.startswith("query"))] for t in texts])

    def encode_text(self, texts):
        self.text_inputs.extend(texts)
        return torch.tensor([[float(ord(t[-1]) - ord("a") + 1),
                              2. if t.startswith("query") else 1., 1.] for t in texts])

    def encode_pc(self, clouds):
        self.pc_inputs.append(clouds.clone())
        return clouds[:, 0, :3] + self.pc_offset


@pytest.fixture
def observation_protocol(tmp_path):
    """Intentionally different query/gallery orders and one gallery distractor."""
    records, expected = {}, {}
    for i, uid in enumerate(("b", "c", "a")):
        views = np.array([[1 + i, 2, 3], [101, 202 + i, 303], [5, 6, 7 + i]], np.float32)
        path = tmp_path / f"{uid}_emb.npz"
        np.savez(path, text=np.array([-999, -999, -999], np.float32),
                 image=views.mean(0), views=views)
        cloud = np.tile(np.array([i + 1., 2., 3., .2, .3, .4], np.float32), (10000, 1))
        cp = tmp_path / f"{uid}_pc.npz"
        np.savez(cp, xyz=cloud[:, :3], rgb=cloud[:, 3:])
        rec = {"embedding": source(path), "cloud": source(cp),
               "canonical_text": f"canonical-{uid}", "query_view": 1,
               "gallery_views": [0, 2]}
        if uid != "c":
            qp = tmp_path / f"{uid}.npy"
            query_cloud = cloud.copy()
            query_cloud[:, 0] += 10
            np.save(qp, query_cloud)
            rec.update(query_cloud=source(qp), query_text=f"query-{uid}")
        records[uid] = rec
        expected[uid] = {"views": views, "cloud": cloud}
    return {"query_uids": ["a", "b"], "gallery_uids": ["b", "c", "a"],
            "records": records}, expected


@pytest.mark.parametrize("condition", tuple(custom.QUERY_CONDITIONS))
def test_available_mean_has_only_the_named_modalities(condition):
    # Unequal raw norms make averaging before per-modality normalization wrong.
    embeds = {"text": np.array([[9., 0, 0], [4., 0, 0]]),
              "image": np.array([[0, 2., 0], [0, 5., 0]]),
              "pc": np.array([[0, 0, 7.], [0, 0, 3.]])}
    flags = np.array(custom.QUERY_CONDITIONS[condition])
    expected = np.tile(flags.astype(float) / np.sqrt(flags.sum()), (2, 1))
    np.testing.assert_allclose(custom.available_mean(embeds, condition), expected)
    # An absent modality is never normalized, included, or used as a mask token.
    for modality, present in zip(custom.MODALITIES, flags):
        if not present:
            embeds[modality] = np.full((2, 3), np.nan)
    np.testing.assert_allclose(custom.available_mean(embeds, condition), expected)


def test_observations_keep_pool_order_and_exclude_query_view(observation_protocol):
    protocol, expected = observation_protocol
    before = copy.deepcopy(protocol)
    bb = FakeBackbone()
    obs = custom.encode_observations(protocol, bb, device="cpu", batch_size=2)
    np.testing.assert_array_equal(obs["targets"], [2, 0])
    assert protocol == before
    for arm in custom.ARMS:
        assert all(v.shape == (3, 3) for v in obs["gallery"][arm].values())
        assert all(v.shape == (2, 3) for v in obs["query"][arm].values())
    for row, uid in enumerate(protocol["gallery_uids"]):
        views = expected[uid]["views"]
        np.testing.assert_allclose(obs["gallery"]["same_observation"]["image"][row], views.mean(0))
        np.testing.assert_allclose(obs["gallery"]["different_observations"]["image"][row],
                                   (views[0] + views[2]) / 2)
        # The high-valued held-out view would visibly change the gallery mean.
        assert not np.allclose(views.mean(0), (views[0] + views[2]) / 2)
    for row, uid in enumerate(protocol["query_uids"]):
        np.testing.assert_allclose(obs["query"]["different_observations"]["image"][row],
                                   expected[uid]["views"][1])
    for modality in custom.MODALITIES:
        np.testing.assert_allclose(obs["query"]["same_observation"][modality],
                                   obs["gallery"]["same_observation"][modality][[2, 0]])
    assert bb.text_inputs == ["canonical-b", "canonical-c", "canonical-a", "query-a", "query-b"]
    assert not np.any(obs["gallery"]["same_observation"]["text"] == -999)


def test_separate_query_point_path_encodes_both_query_observations(observation_protocol):
    protocol, expected = observation_protocol
    bb, qbb = FakeBackbone(), FakeBackbone(pc_offset=100.)
    obs = custom.encode_observations(protocol, bb, query_backbone=qbb, device="cpu", batch_size=1)
    assert sum(len(x) for x in bb.pc_inputs) == 3  # only gallery canonical clouds
    assert sum(len(x) for x in qbb.pc_inputs) == 4  # both observations of two queries
    assert qbb.text_inputs == []
    for row, uid in enumerate(protocol["query_uids"]):
        canonical = expected[uid]["cloud"][0, :3]
        np.testing.assert_array_equal(obs["query"]["same_observation"]["pc"][row], canonical + 100)
        np.testing.assert_array_equal(obs["query"]["different_observations"]["pc"][row],
                                      canonical + np.array([110, 100, 100]))
    np.testing.assert_array_equal(obs["gallery"]["same_observation"]["pc"],
                                  [[1, 2, 3], [2, 2, 3], [3, 2, 3]])


def test_both_image_arms_derive_from_stored_views_not_a_separately_rounded_pool(observation_protocol):
    protocol, expected = observation_protocol
    for uid, record in protocol["records"].items():
        path = Path(record["embedding"]["path"])
        # Historical caches round a precomputed image mean separately from the
        # individual views. A strong sentinel ensures neither arm reads it.
        np.savez(path, text=np.array([-999, -999, -999], np.float32),
                 image=np.array([999, -999, 999], np.float16),
                 views=expected[uid]["views"].astype(np.float16))
        record["embedding"] = source(path)
    obs = custom.encode_observations(protocol, FakeBackbone(), device="cpu", batch_size=2)
    for row, uid in enumerate(protocol["gallery_uids"]):
        views = expected[uid]["views"].astype(np.float16).astype(np.float32)
        np.testing.assert_array_equal(obs["gallery"]["same_observation"]["image"][row], views.mean(0))
        np.testing.assert_array_equal(obs["gallery"]["different_observations"]["image"][row],
                                      views[[0, 2]].mean(0))
    np.testing.assert_array_equal(obs["query"]["same_observation"]["image"],
                                  obs["gallery"]["same_observation"]["image"][[2, 0]])


def test_distinct_strings_collapsing_to_the_same_tokens_are_refused(observation_protocol):
    protocol, _ = observation_protocol
    bb = FakeBackbone()
    # Simulate differing suffixes beyond the actual tokenizer's context window.
    bb.tokenizer = lambda texts: torch.ones((len(texts), 4), dtype=torch.long)
    with pytest.raises(ValueError, match="identical effective CLIP tokens"):
        custom.encode_observations(protocol, bb, device="cpu", batch_size=2)


def test_replaced_observation_bytes_are_refused(observation_protocol):
    protocol, _ = observation_protocol
    Path(protocol["records"]["b"]["embedding"]["path"]).write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="input changed after protocol preparation"):
        custom.encode_observations(protocol, FakeBackbone(), device="cpu", batch_size=2)


def tiny_observations(order=custom.MODALITIES):
    parts = {"text": np.array([[2., 0, 0], [1., 2, 0]], np.float32),
             "image": np.array([[0, 3., 0], [0, 1., 2]], np.float32),
             "pc": np.array([[0, 0, 4.], [3., 0, 1]], np.float32)}
    parts = {m: parts[m] for m in order}
    return {side: {arm: copy.deepcopy(parts) for arm in custom.ARMS}
            for side in ("gallery", "query")}


@pytest.mark.parametrize("order", [custom.MODALITIES, ("pc", "text", "image")])
def test_real_query_tower_receives_none_for_absent_modalities_and_no_layout(order):
    cfg = FusionConfig(dim=3, kind="mean", prefusion_norm=False, include_absent_slots=True)
    model = MetaFindDualTower(DualTowerConfig(dim=3,
        tower_sharing="shared_backbone_separate_fusion", query_fusion=cfg,
        gallery_fusion=replace(cfg), use_layout=False)).eval()
    tokens = np.array([[.2, .3, .4], [.5, .6, .7], [.8, .9, 1.]], np.float32)
    with torch.no_grad():
        model.query.fusion.mask_tokens.copy_(torch.from_numpy(tokens))
    calls = []

    def inspect_input(module, args, kwargs):
        inputs = args[0]
        present = kwargs["present"]
        assert kwargs["layout"] is None
        for m, yes in zip(custom.MODALITIES, present[0].tolist()):
            assert (inputs[m] is not None) == yes, m
        calls.append(tuple(present[0].tolist()))

    hook = model.query.register_forward_pre_hook(inspect_input, with_kwargs=True)
    observations = tiny_observations(order)
    fused = custom.fuse_observations(observations, model=model, device="cpu", batch_size=1)
    hook.remove()
    assert len(calls) == 2 * 7 * 2
    for arm in custom.ARMS:
        for condition, flags in custom.QUERY_CONDITIONS.items():
            total = sum(observations["query"][arm][m] if yes else tokens[j]
                        for j, (m, yes) in enumerate(zip(custom.MODALITIES, flags)))
            expected = total / np.linalg.norm(total, axis=-1, keepdims=True)
            np.testing.assert_allclose(fused[arm]["query"][condition], expected, atol=1e-7)


def ranking_fixture():
    protocol = {"gallery_uids": ["b", "c", "a", "d", "e", "f"], "query_uids": ["a", "b"]}
    gallery = np.eye(6)
    normal = np.array([gallery[2], [.8, 1, 0, 0, 0, 0]])
    vectors = {}
    for arm in custom.ARMS:
        query = {c: normal.copy() for c in custom.QUERY_CONDITIONS}
        query["image"] = gallery[[2, 0]]
        query["full"] = gallery[[2, 1]]
        vectors[arm] = {"gallery": gallery.copy(), "query": query}
    return protocol, vectors


@pytest.mark.parametrize("block", [1, 4, 100])
def test_ranking_uses_uid_mapping_percentages_and_fixed_pools(tmp_path, block):
    protocol, vectors = ranking_fixture()
    before = copy.deepcopy(protocol)
    result = custom.evaluate_vectors(vectors, protocol, tmp_path, "fixture", block=block)
    assert protocol == before
    assert len(list(tmp_path.glob("*.jsonl"))) == 14
    for arm in custom.ARMS:
        assert set(result[arm]) == set(custom.QUERY_CONDITIONS)
        assert result[arm]["text"] == {"R@1": 50., "R@5": 100.}
        assert result[arm]["image"] == {"R@1": 100., "R@5": 100.}
        assert result[arm]["full"]["R@1"] == 50.
        assert result[arm]["full"]["R@5"] == 50.
        assert result[arm]["text+image"]["gain_over_best_constituent_R@1_pp"] == -50.
        rows = [json.loads(x) for x in (tmp_path / f"fixture__{arm}__text.jsonl").read_text().splitlines()]
        assert [r["query_uid"] for r in rows] == ["a", "b"]
        assert [r["target_uid"] for r in rows] == ["a", "b"]
        assert [r["target_column"] for r in rows] == [2, 0]
        assert [r["top1_uid"] for r in rows] == ["a", "c"]
        assert [r["rank"] for r in rows] == [1, 2]


def test_collapsed_gallery_ranks_last_and_cannot_claim_retrieval(tmp_path):
    protocol, vectors = ranking_fixture()
    for arm in custom.ARMS:
        vectors[arm]["gallery"] = np.tile([1., 0, 0], (6, 1))
        vectors[arm]["query"] = {c: np.tile([1., 0, 0], (2, 1)) for c in custom.QUERY_CONDITIONS}
    result = custom.evaluate_vectors(vectors, protocol, tmp_path, "collapse", block=2)
    for arm in custom.ARMS:
        assert all(cell["R@1"] == cell["R@5"] == 0. for cell in result[arm].values())
    rows = [json.loads(x) for x in (tmp_path / "collapse__same_observation__full.jsonl").read_text().splitlines()]
    assert [r["rank"] for r in rows] == [6, 6]
    assert [r["tie_count"] for r in rows] == [5, 5]


@pytest.mark.parametrize("key,value,match", [
    ("query_uids", [], "nonempty and unique"),
    ("query_uids", ["a", "a"], "nonempty and unique"),
    ("gallery_uids", ["a", "a"], "nonempty and unique"),
    ("query_uids", ["missing"], "every query target"),
])
def test_wrong_uid_pools_refused_before_output(tmp_path, key, value, match):
    protocol, vectors = ranking_fixture()
    protocol[key] = value
    with pytest.raises(ValueError, match=match):
        custom.evaluate_vectors(vectors, protocol, tmp_path, "invalid")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("side", ["gallery", "query"])
def test_vector_count_cannot_change_the_declared_pool(tmp_path, side):
    protocol, vectors = ranking_fixture()
    if side == "gallery":
        vectors[custom.ARMS[0]][side] = vectors[custom.ARMS[0]][side][:-1]
    else:
        vectors[custom.ARMS[0]][side]["text"] = vectors[custom.ARMS[0]][side]["text"][:-1]
    with pytest.raises(ValueError, match="vectors do not match the fixed UID list"):
        custom.evaluate_vectors(vectors, protocol, tmp_path, "invalid")
    assert not list(tmp_path.iterdir())


def test_per_query_evidence_is_never_overwritten(tmp_path):
    protocol, vectors = ranking_fixture()
    custom.evaluate_vectors(vectors, protocol, tmp_path, "once")
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        custom.evaluate_vectors(vectors, protocol, tmp_path, "once")
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == original


@pytest.fixture
def cli_seam(monkeypatch, tmp_path, observation_protocol):
    """Stub checkpoint/manifest boundaries; execute real encoding/fusion/scoring.

    Source/configuration authentication is covered by its dedicated tests. This
    fixture isolates orchestration without loading any pretrained model weights.
    """
    from metafind import runlog
    from metafind.eval import custom_models, custom_protocol
    from metafind.train import gallery_index, stage1

    protocol, _ = observation_protocol
    encoding = {"image_aggregation": "mean", "text_serialization": "fixture",
                "text_template": "{text}", "view_aggregation": {"n_views": 3},
                "actual_clip_train_scope": "frozen",
                "missing_modality_representation": "learned_mask"}
    training = {"fusion": "mean", "train_scope": "fuser_only",
                "tower_sharing": "shared_backbone_separate_fusion"}
    protocol.update(n_views=3, encoding_protocol={"content": encoding})
    protocol_path = tmp_path / "prepared.json"
    # Deliberate whitespace proves provenance preserves the actual source bytes.
    protocol_bytes = (json.dumps(protocol, indent=3) + "\n\n").encode()
    protocol_path.write_bytes(protocol_bytes)
    s1 = {"uri": "unused-s1.pt", "sha256": "s1-bytes"}
    s2 = {"uri": "unused-s2.pt", "sha256": "s2-bytes"}
    monkeypatch.setattr(custom_protocol, "load_protocol", lambda p: json.loads(p.read_bytes()))
    monkeypatch.setattr(gallery_index, "load_checkpoint_record", lambda p: copy.deepcopy(s1))
    monkeypatch.setattr(gallery_index, "load_stage2_checkpoint_record",
                        lambda *a, **kw: copy.deepcopy(s2))
    monkeypatch.setattr(stage1, "load_stage1_model_config", lambda p, r: r)
    monkeypatch.setattr(stage1, "load_protocols", lambda: ({}, {}, {}))
    monkeypatch.setattr(stage1, "effective_stage1_model_inputs",
                        lambda *a: (encoding, training, {}))
    monkeypatch.setattr(stage1, "_open_clip_weight_identity", lambda: {"fixture": True})
    monkeypatch.setattr(stage1, "stage1_backbone_kwargs", lambda r: {"checkpoint": "unused.pt"})
    monkeypatch.setattr(custom_models, "verify_mean_initializer", lambda: Path("unused.pt"))
    monkeypatch.setattr(runlog, "runtime_source_sha256", lambda: "fixture-source")
    monkeypatch.setattr(runlog, "code_revision", lambda: "fixture-revision")
    monkeypatch.setattr(runlog, "code_dirty", lambda: True)

    cfg = FusionConfig(dim=3, kind="mean", prefusion_norm=False, include_absent_slots=True)
    model = MetaFindDualTower(DualTowerConfig(dim=3,
        tower_sharing="shared_backbone_separate_fusion", query_fusion=cfg,
        gallery_fusion=replace(cfg), use_layout=False)).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    loaded = SimpleNamespace(backbone=FakeBackbone(pc_offset=2.), query_backbone=None,
                             model=model, record=copy.deepcopy(s1))
    calls = []

    def load_mean(device):
        calls.append("mean")
        return FakeBackbone()

    def load_stage1(path, device):
        calls.append("stage1")
        return loaded

    def apply_stage2(loaded, path, variant):
        calls.append("stage2")
        with torch.no_grad():
            loaded.model.query.fusion.mask_tokens.add_(.5)
        return copy.deepcopy(s2)

    monkeypatch.setattr(custom_models, "load_mean", load_mean)
    monkeypatch.setattr(custom_models, "load_stage1", load_stage1)
    monkeypatch.setattr(custom_models, "apply_stage2", apply_stage2)
    output = tmp_path / "output"
    args = ["--protocol", str(protocol_path), "--stage1-record", str(tmp_path / "s1.json"),
            "--stage2-record", str(tmp_path / "s2.json"), "--out-dir", str(output),
            "--device", "cpu", "--batch-size", "2", "--block", "2"]
    return SimpleNamespace(args=args, output=output, calls=calls, loaded=loaded,
                           s1=s1, s2=s2, protocol=protocol, protocol_path=protocol_path,
                           protocol_bytes=protocol_bytes, encoding=encoding, training=training,
                           custom_models=custom_models, custom_protocol=custom_protocol, stage1=stage1)


def test_cli_three_method_outputs_execute_the_same_protocol(cli_seam):
    f = cli_seam
    assert custom.main(f.args) == 0
    assert f.calls == ["mean", "stage1", "stage2"]
    assert (f.output / "protocol.json").read_bytes() == f.protocol_bytes
    result = json.loads((f.output / "results.json").read_text())
    assert result["provenance"]["protocol_sha256"] == hashlib.sha256(f.protocol_bytes).hexdigest()
    assert result["claim"] == "custom_same_uid_retrieval"
    assert result["independent_test_status"] == "not_certified"
    assert (result["n_query"], result["n_gallery"]) == (2, 3)
    assert set(result["methods"]) == {"ulip2_available_mean", "stage1", "stage2_layout_off"}
    assert result["validation"]["stage2_parent_gallery"] == "unchanged"
    for method, arms in result["methods"].items():
        for arm, conditions in arms.items():
            assert set(conditions) == set(custom.QUERY_CONDITIONS)
            for condition, metrics in conditions.items():
                assert 0 <= metrics["R@1"] <= metrics["R@5"] <= 100
                baseline = result["methods"]["ulip2_available_mean"][arm][condition]["R@1"]
                assert metrics["gain_over_mean_R@1_pp"] == metrics["R@1"] - baseline
    evidence = list(f.output.glob("*.jsonl"))
    assert len(evidence) == 3 * 2 * 7
    for path in evidence:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row["target_column"] for row in rows] == [2, 0]
        assert [row["query_uid"] for row in rows] == ["a", "b"]
    assert (f.output / "table.md").is_file()
    assert not (f.output / "failure.json").exists()


def test_cli_check_only_never_loads_models_or_writes_outputs(cli_seam, capsys):
    f = cli_seam
    assert custom.main(f.args + ["--check-only"]) == 0
    assert f.calls == []
    assert not f.output.exists()
    report = json.loads(capsys.readouterr().out)
    assert report["model_execution"] == "not_performed"
    assert report["effective_token_check"] == "required during model encoding"


def test_cli_protocol_mutation_during_preflight_is_refused(cli_seam, monkeypatch):
    f = cli_seam

    def mutate(path):
        data = json.loads(path.read_bytes())
        path.write_bytes(path.read_bytes() + b"\n")
        return data

    monkeypatch.setattr(f.custom_protocol, "load_protocol", mutate)
    with pytest.raises(ValueError, match="protocol changed while"):
        custom.main(f.args)
    assert f.calls == []
    assert not f.output.exists()


@pytest.mark.parametrize("stage", [1, 2])
def test_cli_checkpoint_replaced_after_preflight_cannot_publish_results(cli_seam, monkeypatch, stage):
    f = cli_seam
    if stage == 1:
        f.loaded.record["sha256"] = "replaced"
    else:
        monkeypatch.setattr(f.custom_models, "apply_stage2", lambda *a: {"sha256": "replaced"})
    with pytest.raises(ValueError, match=f"Stage {stage} checkpoint changed after preflight"):
        custom.main(f.args)
    assert not (f.output / "results.json").exists()
    assert not (f.output / "table.md").exists()
    failure = json.loads((f.output / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert failure["exception"] == "ValueError"


def test_cli_stage2_gallery_change_cannot_publish_results(cli_seam, monkeypatch):
    f = cli_seam

    def wrong_overlay(loaded, *args):
        # Change the executed output: mean-fusion mask changes alone cannot
        # affect an all-present gallery and would be an ineffective negative.
        loaded.model.gallery.register_forward_hook(lambda m, a, output: output.flip(-1))
        return copy.deepcopy(f.s2)

    monkeypatch.setattr(f.custom_models, "apply_stage2", wrong_overlay)
    with pytest.raises(ValueError, match="changed the parent gallery"):
        custom.main(f.args)
    assert not (f.output / "results.json").exists()
    assert not (f.output / "table.md").exists()
    assert json.loads((f.output / "failure.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("field,value", [
    ("actual_clip_train_scope", "full"), ("train_scope", "full"), ("image_tokens", 3),
])
def test_cli_unsupported_cache_forward_configuration_fails_before_execution(cli_seam, field, value):
    f = cli_seam
    (f.encoding if field == "actual_clip_train_scope" else f.training)[field] = value
    with pytest.raises(ValueError, match="requires frozen CLIP and one image token"):
        custom.main(f.args)
    assert f.calls == []
    assert not f.output.exists()


@pytest.mark.parametrize("initializer", ["stage1", "mean"])
def test_cli_check_only_propagates_initializer_validation_failures(cli_seam, monkeypatch, initializer):
    f = cli_seam

    def invalid(*args):
        raise ValueError(f"{initializer} initializer bytes differ")

    if initializer == "stage1":
        monkeypatch.setattr(f.stage1, "stage1_backbone_kwargs", invalid)
    else:
        monkeypatch.setattr(f.custom_models, "verify_mean_initializer", invalid)
    with pytest.raises(ValueError, match=f"{initializer} initializer bytes differ"):
        custom.main(f.args + ["--check-only"])
    assert f.calls == []
    assert not f.output.exists()


def test_cli_check_only_rejects_stage2_with_tied_fusion(cli_seam):
    f = cli_seam
    f.training["tower_sharing"] = "fully_shared"
    with pytest.raises(ValueError, match="Stage 2 requires separate query/gallery fusion weights"):
        custom.main(f.args + ["--check-only"])
    assert f.calls == []
    assert not f.output.exists()
