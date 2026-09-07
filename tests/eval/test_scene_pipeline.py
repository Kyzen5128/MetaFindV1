"""Real CPU scene pipelines across the new module boundaries.

Only the large ULIP constructor/output width and gallery encoder-identity
verification are substituted. Model loaders, checkpoint metadata/byte checks,
index loading, raw observation handling, export, composition and Blender run
unchanged. Synthetic weights/observations do not measure paper retrieval quality.
The full-layout case also substitutes the large relation-writer constructor;
SG2 validation/repair, semantic encoding and encoder identity remain real.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest
import torch
import trimesh

from metafind import paths
from metafind.eval import custom_models
from metafind.models import ulip_backbone
from metafind.models.dual_tower import DualTowerConfig, MetaFindDualTower
from metafind.models.essgnn import ESSGNNConfig, ESSGCLShared
from metafind.models.fusion import FusionConfig
from metafind.models.losses import MetaFindContrastiveLoss
from metafind.scene import compose, idesign, placement, prepare, semantics
from metafind.data.semantic_provenance import (
    build_node_source, build_edge_source, source_identity_sha256, text_encoder_identity,
)
from metafind.train import gallery_index, stage1, stage2
from .test_idesign_scene import inputs as idesign_inputs
from .test_scene_placement import inputs as placement_inputs
from .test_scene_composition import slot


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


class _TinyFrozenBackbone:
    """Deterministic two-dimensional constructor substitute, never pretrained."""
    constructed = []

    def __init__(self, cfg):
        assert cfg.device == "cpu" and cfg.train_scope == "fuser_only"
        self.cfg = cfg
        self.model = torch.nn.Linear(2, 2, bias=False).requires_grad_(False).eval()
        self.texts, self.clouds = [], []
        self.constructed.append(self)

    def set_train_scope(self, scope):
        assert scope == "fuser_only"
        self.model.requires_grad_(False).eval()

    def encode_text(self, texts):
        self.texts.extend(texts)
        return torch.tensor([[3., 0.] if text.startswith("A ") else [0., 4.]
                             for text in texts])

    def preprocess(self, image):
        return torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float()

    def encode_image(self, batch):
        return batch[:, :2].mean((2, 3))

    def encode_pc(self, points):
        self.clouds.append(points.clone())
        return torch.stack((points[:, :, :3].mean((1, 2)),
                            points[:, :, 3:].mean((1, 2)) * 5), dim=1)


def _checkpoints_and_index(tmp_path, monkeypatch):
    """Real recipe files and partial S1/S2 payloads for the existing loaders."""
    monkeypatch.setattr(ulip_backbone, "EMBED_DIM", 2)
    monkeypatch.setattr(custom_models, "ULIPBackbone", _TinyFrozenBackbone)
    monkeypatch.setattr(ulip_backbone, "ULIPBackbone", _TinyFrozenBackbone)
    monkeypatch.setattr(_TinyFrozenBackbone, "constructed", [])
    protocol_dir = tmp_path / "protocols"
    monkeypatch.setattr(paths, "OUTPUTS", protocol_dir)
    fusion = FusionConfig(dim=2, kind="mean", include_absent_slots=False)
    model = MetaFindDualTower(DualTowerConfig(
        dim=2, tower_sharing="shared_backbone_separate_fusion", use_layout=False,
        query_fusion=fusion, gallery_fusion=fusion))
    encoding = {"status": "resolved", "actual_clip_train_scope": "frozen",
                "missing_modality_representation": "learned_token", "image_aggregation": "mean"}
    values = {"optimizer": "adamw", "learning_rate": 1e-4, "weight_decay": .01,
              "scheduler": "cosine", "batch_size": 2, "epochs": 1, "max_epochs": 1,
              "p_mask": .3, "decay_mask_tokens": False, "init_temperature": .5,
              "learnable_temperature": False, "max_logit_scale": 100.,
              "betas": [.9, .999], "eps": 1e-8, "warmup_epochs": 0,
              "lr_start": 1e-4, "lr_end": 0., "seed": 17}
    hp = {"values": values, "sha256": hashlib.sha256(json.dumps(values).encode()).hexdigest()}
    training = {"status": "resolved", "fusion": "mean", "train_scope": "fuser_only",
                "tower_sharing": "shared_backbone_separate_fusion",
                "hyperparameter_config_hash": hp["sha256"]}
    for name, value in (("stage1_encoding_protocol.json", encoding),
                        ("stage1_protocol.json", training), ("stage1_hyperparameters.json", hp)):
        _json(protocol_dir / name, value)
    initializer = tmp_path / "tiny_initializer.bin"
    initializer.write_bytes(b"synthetic CPU constructor fixture, not ULIP weights")
    snapshot = stage1.model_input_snapshot(encoding, training, hp, model)
    metadata = {"model_inputs": snapshot,
                "initializers": {"ulip2": {"uri": str(initializer), "sha256": _sha(initializer)}},
                "fixture": "untrained tiny model; no pretrained performance claim"}
    parent_path = tmp_path / "s1.pt"
    torch.save({"metadata": metadata, "backbone_trainable_state": {},
                "tower_trainable_state": stage1.trainable_state_dict(model),
                "loss_trainable_state": {}}, parent_path)
    parent = {**metadata, "uri": str(parent_path), "sha256": _sha(parent_path)}
    parent_json = _json(tmp_path / "records/s1.json", parent)
    model.gallery.requires_grad_(False)
    with torch.no_grad():
        model.query.fusion.mask_tokens.fill_(.125)
    child = {"uri": str(tmp_path / "s2.pt"), "variant_id": "no_layout", "use_layout": False,
             "lambda_init": None, "stage1_checkpoint_sha256": parent["sha256"],
             "stage1_model_inputs": snapshot,
             "stage2_protocol": {"asset_modalities": ["text", "image"]}}
    torch.save(stage2.stage2_checkpoint_payload(model, MetaFindContrastiveLoss(), child), child["uri"])
    child["sha256"] = _sha(child["uri"])
    child_json = _json(tmp_path / "records/s2.json", child)
    index_path = tmp_path / "gallery.npz"
    # Deliberately reversed rows exercise the real registry and selected-UID reordering.
    np.savez(index_path, ids=np.array(["B", "A"]), embeddings=np.array([[0., 1.], [1., 0.]], dtype=np.float32))
    index = {"uri": str(index_path), "sha256": _sha(index_path), "count": 2, "dim": 2,
             "stage1_checkpoint_sha256": parent["sha256"],
             "gallery_encoder_sha256": hashlib.sha256(b"explicit fixture verification seam").hexdigest(),
             "fixture": "encoder identity helper substituted; not an actual ULIP hash"}
    registry = _json(tmp_path / "records/gallery.json", {parent["sha256"]: index})
    verified = []

    def verify_encoder(record, backbone, restored, *, parent_checkpoint, allow_legacy):
        assert record == index and parent_checkpoint["sha256"] == parent["sha256"]
        assert allow_legacy is False and isinstance(backbone, _TinyFrozenBackbone)
        assert not any(p.requires_grad for p in restored.parameters())
        assert restored.cfg.tower_sharing == "shared_backbone_separate_fusion"
        verified.append(backbone)
        return record["gallery_encoder_sha256"]

    monkeypatch.setattr(gallery_index, "verify_gallery_encoder", verify_encoder)
    return parent_json, child_json, registry, index, verified


def test_raw_idesign_to_real_cpu_blender_scene_pipeline(
        tmp_path, monkeypatch, idesign_inputs, placement_inputs):
    try:
        blender = placement.default_blender()
    except FileNotFoundError:
        pytest.skip("installed Blender binary unavailable; no dependency installation")
    if not blender.is_file() or not os.access(blender, os.X_OK):
        pytest.skip("installed Blender binary unavailable; no dependency installation")
    parent, child, registry, index, verified = _checkpoints_and_index(tmp_path, monkeypatch)
    texts = {"A": {"text": "canonical retrieved A"}, "B": {"text": "canonical retrieved B"}}
    text_path = _json(tmp_path / "records/texts.json", texts)
    mapping = json.loads(idesign_inputs["mapping"].read_text())
    mapping["chair_2"]["text"] = "A requested chair; not its retrieved annotation"
    mapping["table_1"]["text"] = "B requested table"
    _json(idesign_inputs["mapping"], mapping)
    base = {"schema": prepare.REQUEST_SCHEMA,
            "provenance": {"fixture": "two-slot CPU cross-module smoke; untrained model"},
            "stage1_record": str(parent), "stage2_record": str(child), "variant": "no_layout",
            "gallery_registry": str(registry), "gallery_ids": ["A", "B"], "asset_texts": str(text_path),
            "mode": "iterative", "use_layout": False}
    _json(idesign_inputs["base"], base)
    request_path = idesign.build_request(idesign_inputs["scene"], idesign_inputs["sidecar"],
        idesign_inputs["mapping"], idesign_inputs["base"], tmp_path / "request.json")
    request = json.loads(request_path.read_text())
    original_scene = json.loads(idesign_inputs["scene"].read_text())
    planned = [original_scene[0], original_scene[2]]
    assert [q["slot"] for q in request["queries"]] == planned
    assert request["provenance"]["idesign_adapter"]["query_order"] == ["chair_2", "table_1"]
    for source in request["provenance"]["idesign_adapter"]["sources"].values():
        assert source["sha256"] == _sha(source["path"])

    manifest_path = prepare.prepare(request_path, tmp_path / "encoded", device="cpu")
    assert len(verified) == len(_TinyFrozenBackbone.constructed) == 1
    backbone = verified[0]
    assert backbone.texts == [mapping["chair_2"]["text"], mapping["table_1"]["text"]]
    assert backbone.clouds[0].shape == (1, 10000, 6)
    torch.testing.assert_close(backbone.clouds[0][:, :, :3], torch.zeros(1, 10000, 3))
    torch.testing.assert_close(backbone.clouds[0][:, :, 3:], torch.ones(1, 10000, 3))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["model"]["config"]["essgnn"] is None
    assert manifest["provenance"]["semantic_source"]["status"] == "not_used"
    sources = manifest["provenance"]["sources"]
    required_sources = [request_path, parent, child, registry, Path(index["uri"]), text_path,
                        tmp_path / "observations/image.png", tmp_path / "observations/pc.npz"]
    assert set(map(str, required_sources)) <= sources.keys()
    assert all(_sha(path) == digest for path, digest in sources.items())
    exported = torch.load(manifest_path.parent / manifest["model"]["weights"]["path"], weights_only=True)
    torch.testing.assert_close(exported["fusion.mask_tokens"], torch.full((3, 2), .125), rtol=0, atol=0)

    result = compose.run_manifest(manifest_path)
    assert result["gallery_ids"] == ["A", "B"]
    assert [step["slot_id"] for step in result["trace"]] == ["chair_2", "table_1"]
    assert [step["selected_asset_id"] for step in result["trace"]] == ["A", "B"]
    assert [step["selected_node_text"] for step in result["trace"]] == [texts[a]["text"] for a in ("A", "B")]
    assert [step["context"]["asset_ids"] for step in result["trace"]] == [[], ["A"]]
    assert result["final_graph"]["support"] == [[0, 1]]
    assert [node["slot"] for node in result["final_graph"]["nodes"]] == planned
    np.testing.assert_allclose(result["trace"][0]["query_embedding"], [3., 0.])
    np.testing.assert_allclose(result["trace"][1]["query_embedding"], [10/3, 29/3], atol=1e-6)
    assert _sha(index["uri"]) == index["sha256"]
    composition_path = _json(tmp_path / "actual_composition.json", result)

    # Two distinct raw GLBs make incorrect selected-asset handoff observable.
    second = trimesh.creation.icosphere(subdivisions=0, radius=.5)
    second.visual.vertex_colors = [20, 210, 40, 255]
    second_glb = tmp_path / "B.glb"
    second_glb.write_bytes(second.export(file_type="glb"))
    meshes = {"A": placement_inputs["glb"], "B": second_glb}
    assets = {}
    for uid, mesh in meshes.items():
        annotation = _json(tmp_path / f"{uid}_annotation.json", {"uid": uid, "text": texts[uid]["text"]})
        assets[uid] = {"mesh": {"path": str(mesh), "sha256": _sha(mesh), "frame": placement.FRAME},
                       "annotation": {"path": str(annotation), "sha256": _sha(annotation)}}
    asset_path = _json(tmp_path / "selected_assets.json", {"provenance": {"fixture": "static raw GLBs"}, "assets": assets})
    sidecar = json.loads(idesign_inputs["sidecar"].read_text())
    room = {"room_id": sidecar["scene_id"], "dimensions": sidecar["room_dimensions"],
            "provenance": {"sidecar_sha256": _sha(idesign_inputs["sidecar"])},
            "surfaces": [{"id": "floor", "vertices": [[0., 0., 0.], [4., 0., 0.], [4., 5., 0.], [0., 5., 0.]],
                          "rgba": [.6, .6, .6, 1.], "roughness": .7}]}
    room_path = _json(tmp_path / "actual_room.json", room)
    render = json.loads(placement_inputs["render"].read_text())
    pose = [[1., 0., 0., 2.], [0., 1., 0., 2.5], [0., 0., 1., 7.], [0., 0., 0., 1.]]
    render["cameras"][0]["matrix_world"] = pose
    render["lights"][0]["matrix_world"] = pose
    render_path = _json(tmp_path / "actual_render.json", render)
    placement_path = placement.prepare_placement(composition_path, asset_path, room_path,
        tmp_path / "placement", render_config_path=render_path)
    frozen = placement.load_placement(placement_path)
    assert frozen["instances"] == result["final_graph"]["nodes"]
    assert frozen["sources"]["composition"]["sha256"] == _sha(composition_path)
    assert frozen["render"]["device"] == "CPU" and frozen["render"]["resolution"] == [64, 64]
    assert frozen["render"]["samples"] == 2 and frozen["render"]["threads"] == 1
    rendered_path = placement.run_placement(placement_path, tmp_path / "rendered", blender=blender, timeout=120)
    rendered = json.loads(rendered_path.read_text())
    assert rendered["status"] == "complete" and rendered["device"] == "CPU"
    assert rendered["manifest_sha256"] == _sha(placement_path)
    assert [(node["slot_id"], node["asset_id"]) for node in rendered["instances"]] == [("chair_2", "A"), ("table_1", "B")]
    assert [len(node["mesh_objects"]) for node in rendered["instances"]] == [2, 1]
    expected_bounds = {"chair_2": [[.5, 1.5, 0.], [1.5, 2.5, 1.]],
                       "table_1": [[2.5, 1.5, 0.], [3.5, 2.5, 1.]]}
    for node in rendered["instances"]:
        np.testing.assert_allclose(node["actual_bbox"], expected_bounds[node["slot_id"]], atol=1e-5)
    for record in [rendered["blend"], *rendered["renders"]]:
        assert record["sha256"] == _sha(rendered_path.parent / record["path"])
    assert len(rendered["renders"]) == 1
    with Image.open(rendered_path.parent / rendered["renders"][0]["path"]) as image:
        pixels = np.array(image)
    assert pixels.shape == (64, 64, 4) and pixels[..., :3].std() > 10

    # Reopen the actual saved blend and derive bounds from its mesh vertices,
    # independently of placement's bounds helper and reported JSON.
    inspection_path = tmp_path / "blend_geometry.json"
    inspector = tmp_path / "inspect_pipeline_blend.py"
    inspector.write_text(
        "import bpy,json\nfrom pathlib import Path\ns=bpy.context.scene\n"
        "assert s.render.engine=='CYCLES' and s.cycles.device=='CPU'\n"
        "assert s.cycles.samples==2 and s.render.threads==1\n"
        "assert (s.render.resolution_x,s.render.resolution_y)==(64,64)\n"
        "result={}\n"
        "for root in s.objects:\n"
        " if 'slot_id' not in root: continue\n"
        " points=[o.matrix_world@v.co for o in root.children_recursive if o.type=='MESH' for v in o.data.vertices]\n"
        " result[root['slot_id']]={'asset_id':root['asset_id'],'bbox':[[min(p[a] for p in points) for a in range(3)],[max(p[a] for p in points) for a in range(3)]]}\n"
        f"Path({str(inspection_path)!r}).write_text(json.dumps(result))\n")
    proc = subprocess.run([str(blender), "--background", "--threads", "1",
        str(rendered_path.parent / rendered["blend"]["path"]), "--python-exit-code", "3", "--python", str(inspector)],
        capture_output=True, text=True, timeout=60, cwd=tmp_path,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    inspection = json.loads(inspection_path.read_text())
    assert set(inspection) == set(expected_bounds)
    for sid, uid in (("chair_2", "A"), ("table_1", "B")):
        assert inspection[sid]["asset_id"] == uid
        np.testing.assert_allclose(inspection[sid]["bbox"], expected_bounds[sid], atol=1e-5)
    assert all(_sha(path) == digest for path, digest in sources.items())
    assert all(_sha(meshes[uid]) == assets[uid]["mesh"]["sha256"] for uid in assets)


class _TinySemanticModel(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.open_clip_model = clip

    def encode_text(self, tokens):
        return self.open_clip_model.encode_text(tokens, normalize=False)


class _TinySemanticBackbone(_TinyFrozenBackbone):
    """Actual tiny OpenCLIP forward/tokenizer, with no pretrained download."""
    def __init__(self, cfg):
        from open_clip.model import CLIP
        from open_clip.tokenizer import SimpleTokenizer

        super().__init__(cfg)
        self.tokenizer = SimpleTokenizer(context_length=16)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(29)
            clip = CLIP(embed_dim=2,
                vision_cfg={"layers": 1, "width": 8, "head_width": 4,
                            "patch_size": 2, "image_size": 4},
                text_cfg={"context_length": 16, "vocab_size": self.tokenizer.vocab_size,
                          "width": 8, "heads": 2, "layers": 1})
        self.model = _TinySemanticModel(clip).requires_grad_(False).eval()

    def encode_text(self, texts):
        self.texts.extend(texts)
        with torch.no_grad():
            return self.model.encode_text(self.tokenizer(texts))


def _full_layout_reference(tmp_path, monkeypatch):
    """Synthetic training evidence, bound to an actual loaded tiny text tower."""
    parent, child, registry, index, verified = _checkpoints_and_index(tmp_path, monkeypatch)
    monkeypatch.setattr(custom_models, "ULIPBackbone", _TinySemanticBackbone)
    monkeypatch.setattr(ulip_backbone, "ULIPBackbone", _TinySemanticBackbone)
    monkeypatch.setattr(semantics.n08, "EDGE_DIM", 2)
    monkeypatch.setattr(semantics.n08, "TEXT_ENCODER_VERSION", "tiny-openclip-fixture")
    monkeypatch.setattr(semantics.n08, "LLM_MODEL", "fixture-writer")
    monkeypatch.setattr(semantics.n08, "LLM_MODEL_PATH", str(tmp_path / "fixture-writer"))
    loaded = custom_models.load_stage1(parent, "cpu")
    identity = text_encoder_identity(loaded.backbone)
    texts = {"A": {"text": "A canonical chair annotation", "relation_text": "chair"},
             "B": {"text": "B canonical table annotation", "relation_text": "table"}}
    text_path = _json(tmp_path / "reference/texts.json", texts)
    node_source = build_node_source(texts, identity)
    node_path = tmp_path / "reference/nodes.npz"
    np.savez(node_path, ids=np.array(node_source["asset_ids"]),
             embeddings=semantics.n08.encode_sentences([texts[a]["text"] for a in node_source["asset_ids"]],
                                                       backbone=loaded.backbone),
             source_identity_sha256=np.array(source_identity_sha256(node_source)))
    node = {"uri": str(node_path), "sha256": _sha(node_path)}
    node_record = _json(tmp_path / "reference/nodes.json", {**node, "source_identity": node_source})
    edge_path = tmp_path / "reference/edges.npz"
    edge_source = build_edge_source({}, identity)
    np.savez(edge_path, keys=np.array([], dtype=str), embeddings=np.empty((0, 2), dtype=np.float32),
             source_identity_sha256=np.array(source_identity_sha256(edge_source)))
    edge = {"uri": str(edge_path), "sha256": _sha(edge_path)}
    cache_path = _json(tmp_path / "reference/cache.json", {
        "prompt_version": semantics.n08.PROMPT_VERSION,
        "llm_model": semantics.n08.LLM_MODEL,
        "text_encoder_version": semantics.n08.TEXT_ENCODER_VERSION,
        "entries": {}, "source_identity": edge_source, "embedding_artifact": edge})
    arch = ESSGNNConfig(node_feat_dim=2, edge_feat_dim=2, out_dim=2,
        hidden_dim=2, n_layers=1, use_io_projections=True,
        architecture_family="appendix_shared_msg", pooling="mean")
    fusion = FusionConfig(dim=2, kind="mean", include_absent_slots=False)
    model = MetaFindDualTower(DualTowerConfig(dim=2,
        tower_sharing="shared_backbone_separate_fusion", use_layout=True,
        query_fusion=fusion, gallery_fusion=fusion, essgnn=arch, init_lambda=1.))
    model.gallery.requires_grad_(False)
    with torch.no_grad():
        graph = model.query.layout_encoder
        for parameter in graph.parameters():
            parameter.zero_()
        graph.embed_in.weight.copy_(torch.eye(2))
        graph.embed_out.weight.copy_(torch.eye(2))
        graph.embed_out.bias.copy_(torch.tensor([8., 0.]))
        graph.missing_edge_token.fill_(-4.)
        layer = graph.layers[0]
        # A nonzero relation path, not an identity-only graph: success vectors
        # and the learned missing token must yield different query embeddings.
        layer.phi_e[0].weight[0, -2:] = torch.tensor([.3, .4])
        layer.phi_e[0].bias[0] = 1.
        layer.phi_e[2].weight[0, 0] = 1.
        layer.phi_h[0].weight[0, 0] = 1.
        layer.phi_h[2].weight[1, 0] = 4.
    rec = json.loads(child.read_text())
    rec.pop("sha256")
    rec.update(variant_id="full", use_layout=True, lambda_init={"init_lambda": 1.},
               arch_protocol={**asdict(arch), "status": "resolved"},
               layout_input_dims={"node_feat_dim": 2, "edge_feat_dim": 2},
               input_identity={"artifacts": {
                   "node_record": {"uri": str(node_record), "sha256": _sha(node_record)},
                   "node_embeddings": node, "object_text": {"uri": str(text_path), "sha256": _sha(text_path)},
                   "semantic_cache": {"uri": str(cache_path), "sha256": _sha(cache_path)},
                   "semantic_embeddings": edge}})
    torch.save(stage2.stage2_checkpoint_payload(model, MetaFindContrastiveLoss(), rec), rec["uri"])
    rec["sha256"] = _sha(rec["uri"])
    _json(child, rec)
    return parent, child, registry, index, verified, text_path, cache_path, identity


def test_full_layout_sg2_cache_reaches_real_checkpoint_and_iterative_retrieval(tmp_path, monkeypatch):
    parent, child, registry, index, verified, text_path, empty_cache, identity = \
        _full_layout_reference(tmp_path, monkeypatch)
    image_path = tmp_path / "query.png"
    Image.new("RGB", (2, 2), (4, 0, 0)).save(image_path)
    spec = {"schema": prepare.REQUEST_SCHEMA, "provenance": {"fixture": "full-layout CPU integration"},
            "stage1_record": str(parent), "stage2_record": str(child), "variant": "full",
            "gallery_registry": str(registry), "gallery_ids": ["A", "B"], "asset_texts": str(text_path),
            "initial_graph": {"room_id": "fixture-room", "nodes": []},
            "queries": [{"slot": slot(f"query-{i}", i, on="query-0" if i == 1 else None),
                         "images": [str(image_path)]} for i in range(3)],
            "mode": "iterative", "use_layout": True, "semantic_cache": str(empty_cache)}
    request = _json(tmp_path / "request.json", spec)
    incomplete = tmp_path / "incomplete"
    with pytest.raises(compose.MissingSemanticPairsError) as error:
        prepare.prepare(request, incomplete, device="cpu")
    assert error.value.pairs == [["A", "A"]]
    assert not (incomplete / "manifest.json").exists()
    required = incomplete / "required_pairs.json"
    assert json.loads(required.read_text()) == error.value.pairs

    answers, prompts = [], []

    class Writer:
        def __init__(self, model_id, device):
            assert device == "cpu"
            self.model_id = model_id
            self.answers = iter(answers)

        def generate(self, prompt):
            prompts.append(prompt)
            return next(self.answers)

    monkeypatch.setattr(semantics.n08, "RelationWriter", Writer)
    results, caches = {}, {}
    for name, last_answer in (("success", "Sure: The chairs accompany each other."),
                              ("degraded", "1234")):
        answers[:] = ["", last_answer]
        prompts.clear()
        cache = semantics.generate(text_path, required, parent, child, tmp_path / f"sg2-{name}", device="cpu")
        assert len(prompts) == semantics.n08.MAX_ATTEMPTS == 2
        assert "YOUR PREVIOUS ANSWER WAS REJECTED" in prompts[1]
        assert "chair" in prompts[0] and "canonical chair annotation" not in prompts[0]
        cache_record = json.loads(cache.read_text())
        key = error.value.keys[0]
        assert set(cache_record["entries"]) == {key}
        assert cache_record["entries"][key]["degraded"] == (name == "degraded")
        proof = json.loads((cache.parent / "record.json").read_text())
        assert proof["text_encoder_identity"] == identity
        assert proof["generated_successes"] == int(name == "success")
        assert proof["generated_degraded"] == int(name == "degraded")
        assert proof["llm"]["historical_weight_byte_equivalence"] == "UNKNOWN"
        assert all(_sha(path) == digest for path, digest in proof["sources"].items())
        caches[name] = cache
        request = _json(tmp_path / f"request-{name}.json", {**spec, "semantic_cache": str(cache)})
        manifest_path = prepare.prepare(request, tmp_path / f"prepared-{name}", device="cpu")
        manifest = json.loads(manifest_path.read_text())
        assert manifest["model"]["config"]["use_layout"] is True
        assert manifest["model"]["config"]["essgnn"]["architecture_family"] == "appendix_shared_msg"
        assert manifest["provenance"]["semantic_source"]["encoder_identity"] == identity
        assert str(cache) in manifest["provenance"]["sources"]
        assert all(_sha(path) == digest for path, digest in manifest["provenance"]["sources"].items())
        layers = []

        def observe(module, args):
            if isinstance(module, ESSGCLShared):
                layers.append(tuple(value.detach().clone() for value in args))

        hook = torch.nn.modules.module.register_module_forward_pre_hook(observe)
        try:
            result = compose.run_manifest(manifest_path)
        finally:
            hook.remove()
        results[name] = result
        assert result["trace"] == json.loads((manifest_path.parent / "validation_composition.json").read_text())["trace"]
        assert [row["context"]["asset_ids"] for row in result["trace"]] == [[], ["A"], ["A", "A"]]
        assert [row["selected_asset_id"] for row in result["trace"]] == ["A"] * 3
        assert [row["selected_node_text"] for row in result["trace"]] == ["A canonical chair annotation"] * 3
        assert result["trace"][2]["context"]["semantic_missing"] == [name == "degraded"] * 2
        assert result["trace"][2]["context"]["support"] == [[0, 1]]
        assert [len(row[0]) for row in layers] == [1, 2]
        bundle = torch.load(manifest_path.parent / "inputs.pt", weights_only=True)
        torch.testing.assert_close(layers[1][0], bundle["asset_node_embeddings"]["A"].repeat(2, 1))
        if name == "success":
            torch.testing.assert_close(layers[1][3], bundle["semantic_cache"][key].repeat(2, 1))
        else:
            assert bundle["semantic_cache"][key] is None
            torch.testing.assert_close(layers[1][3], torch.full((2, 2), -4.))
        assert [row["slot_id"] for row in result["trace"]] == [f"query-{i}" for i in range(3)]
    # Same actual checkpoint/gallery/queries; the generated relation versus
    # exhausted SG2 evidence changes the layout branch after two placements.
    assert results["success"]["trace"][:2] == results["degraded"]["trace"][:2]
    assert np.linalg.norm(np.subtract(results["success"]["trace"][2]["query_embedding"],
                                      results["degraded"]["trace"][2]["query_embedding"])) > 1.
    off_request = _json(tmp_path / "request-off.json", {**spec, "use_layout": False,
                                                       "semantic_cache": str(caches["success"])})
    off = compose.run_manifest(prepare.prepare(off_request, tmp_path / "prepared-off", device="cpu"))
    for row in off["trace"]:
        np.testing.assert_array_equal(row["query_embedding"], [4., 0.])
        assert row["layout_norm"] == row["lambda_layout_norm"] == 0.
    assert results["success"]["trace"][1]["lambda_layout_norm"] > 7.
    assert _sha(index["uri"]) == index["sha256"] and len(verified) == 4
    assert all(backbone.texts == ["A canonical chair annotation", "B canonical table annotation"]
               for backbone in verified)
