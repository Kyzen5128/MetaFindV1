"""Isolated raw-input/judgment fixtures; no labels, corpus, or models generated."""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from metafind.eval import intent_protocol as m


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    data = tmp_path / "data"
    for key, rel in {"DATA": "", "OUTPUTS": "outputs", "EMBEDDINGS": "outputs/embeddings",
                     "POINTCLOUDS": "outputs/pointclouds", "ANNOTATIONS": "outputs/annotations",
                     "OBJAVERSE_GLB": "datasets/objaverse-lvis/glbs"}.items():
        root = data / rel
        root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(m.paths, key, root)
    encoding = {"status": "resolved", "actual_clip_train_scope": "frozen",
                "image_aggregation": "mean", "view_aggregation": {"n_views": 11},
                "text_template": "__json__", "text_serialization": "fixture-json"}
    write_json(m.paths.OUTPUTS / "stage1_encoding_protocol.json", encoding)
    for uid in ("chair-a", "chair-b"):
        ann = {"uid": uid, "description": "canonical " + uid, "category": "chair",
               "width": 10, "length": 20, "height": 30, "volume": 6000, "mass": 2}
        write_json(m.paths.ANNOTATIONS / f"{uid}.json", ann)
        text = m.canonical.serialize_annotation(ann, template="__json__")
        views = np.arange(44, dtype=np.float32).reshape(11, 4) + 1
        np.savez(m.paths.EMBEDDINGS / f"{uid}.npz", text=np.ones(4), image=views.mean(0), views=views)
        write_json(m.paths.EMBEDDINGS / f"{uid}.json",
                   {"uid": uid, "text": text, "n_views": 11, "aggregation": "mean",
                    "text_serialization": "fixture-json", "encoder_identity": "test-only"})
        xyz = np.resize(np.eye(3), (10000, 3)).astype(np.float32)
        np.savez(m.paths.POINTCLOUDS / f"{uid}.npz", xyz=xyz, rgb=np.ones((10000, 3)))
    incoming = tmp_path / "external"
    incoming.mkdir()
    Image.new("RGB", (12, 10), color=(20, 40, 60)).save(incoming / "reference.png")
    # Deliberately not pre-normalized: the runner, not this producer, does pc_norm.
    raw_pc = np.ones((10000, 6), dtype=np.float32)
    raw_pc[:, :3] = np.resize(np.eye(3) * 17 + 8, (10000, 3))
    np.save(incoming / "reference.npy", raw_pc)
    query = {"query_id": "intent-dining-seating", "text": "Comfortable seats for a family dining room.",
             "conditions": ["text"],
             "provenance": {"source": "fixture brief", "creator": "test assessor", "method": "synthetic test only"},
             "qrels": {"text": {"chair-a": 1, "chair-b": True}},
             "judgments": {"text": {"assessor": "test assessor", "method": "manual fixture",
                                    "criteria": "both fit the textual fixture request"}}}
    spec = {"schema": m.SPEC_SCHEMA, "gallery_uids": ["chair-b", "chair-a"],
            "provenance": {"purpose": "synthetic CPU test"}, "queries": [query]}
    source = incoming / "spec.json"
    write_json(source, spec)
    def no_sampling(*args, **kwargs):
        raise AssertionError("intent preparation must not sample meshes")
    monkeypatch.setattr(m.canonical.pointclouds, "sample_mesh", no_sampling)
    return source, tmp_path / "prepared", spec


def prepare(corpus, spec=None):
    source, out, original = corpus
    if spec is not None:
        write_json(source, spec)
    return m.prepare_protocol(source, out)


def multimodal(spec):
    spec = copy.deepcopy(spec)
    q = spec["queries"][0]
    q["images"] = ["reference.png"]
    q["pointcloud"] = "reference.npy"
    q["conditions"] = list(m.QUERY_CONDITIONS)
    q["qrels"] = {c: {"chair-a": c == "text", "chair-b": 1} for c in q["conditions"]}
    q["judgments"] = {c: {"assessor": "test assessor", "method": "per-condition fixture",
                          "criteria": f"inspect only {c} inputs"} for c in q["conditions"]}
    return spec


def test_text_intent_id_is_independent_and_multiple_assets_can_be_relevant(corpus):
    source, out, spec = corpus
    before = {p: digest(p) for p in source.parent.parent.rglob("*") if p.is_file()}
    dest = prepare(corpus)
    result = m.load_protocol(dest)
    assert result["schema"] == m.SCHEMA and result["status"] == "complete"
    assert result["query_ids"] == ["intent-dining-seating"]
    assert not set(result["query_ids"]) & set(result["gallery_uids"])
    assert result["gallery_uids"] == ["chair-b", "chair-a"]
    assert result["queries"] == spec["queries"]
    assert result["queries"][0]["qrels"]["text"] == {"chair-a": 1, "chair-b": True}
    assert all(digest(p) == sha for p, sha in before.items())
    assert not dest.with_suffix(".json.part").exists()
    assert not list(m.paths.OBJAVERSE_GLB.glob("*.glb"))  # meshes are not required
    assert all("query_view" not in r and "gallery_views" not in r for r in result["records"].values())


def test_each_condition_keeps_its_own_complete_judgments_and_raw_sources(corpus):
    spec = multimodal(corpus[2])
    raw_path = corpus[0].parent / "reference.npy"
    before = raw_path.read_bytes()
    dest = prepare(corpus, spec)
    result = m.load_protocol(dest)
    q = result["queries"][0]
    assert q["conditions"] == list(m.QUERY_CONDITIONS)
    assert q["qrels"]["text"]["chair-a"] is True
    assert q["qrels"]["image"]["chair-a"] is False
    assert q["judgments"] == spec["queries"][0]["judgments"]
    assert q["pointcloud"]["path"] == str(raw_path.resolve())
    assert m.verified_source_bytes(q["pointcloud"]) == before == raw_path.read_bytes()
    assert q["images"][0]["path"] == str((corpus[0].parent / "reference.png").resolve())
    assert "pc_norm(float64 xyz)" in result["choices"]["query_pc_normalization"]


@pytest.mark.parametrize("value", [0.0, 1.0, 0.5, "1", None, [], {}, 2, -1])
def test_nonbinary_or_wrong_label_types_are_not_accepted(corpus, value):
    spec = corpus[2]
    spec["queries"][0]["qrels"]["text"]["chair-a"] = value
    with pytest.raises(ValueError, match="booleans or integer"):
        prepare(corpus, spec)
    assert not corpus[1].exists()


@pytest.mark.parametrize("change", ["unjudged", "foreign_uid", "no_positive", "wrong_condition",
                                     "missing_judgment", "missing_assessor", "missing_criteria"])
def test_coverage_and_assessment_are_explicit(corpus, change):
    spec = corpus[2]
    q = spec["queries"][0]
    if change == "unjudged":
        del q["qrels"]["text"]["chair-a"]
    elif change == "foreign_uid":
        q["qrels"]["text"]["chair-c"] = 0
    elif change == "no_positive":
        q["qrels"]["text"] = dict.fromkeys(spec["gallery_uids"], 0)
    elif change == "wrong_condition":
        q["qrels"]["image"] = q["qrels"].pop("text")
    elif change == "missing_judgment":
        q["judgments"] = {}
    else:
        del q["judgments"]["text"][change.removeprefix("missing_")]
    with pytest.raises(ValueError):
        prepare(corpus, spec)
    assert not corpus[1].exists()


@pytest.mark.parametrize("conditions", [[True], [1], [[]], ["unknown"], ["text", "text"], [], "text",
                                         ["image"], ["pc"], ["full"]])
def test_no_unsupported_or_missing_modality_fallback(corpus, conditions):
    spec = corpus[2]
    spec["queries"][0]["conditions"] = conditions
    with pytest.raises(ValueError):
        prepare(corpus, spec)
    assert not corpus[1].exists()


@pytest.mark.parametrize("key", ["target_uid", "canonical_uid", "embedding", "embeddings", "image_embedding", "pc_embedding"])
def test_no_implicit_target_or_external_embedding_shortcut(corpus, key):
    spec = corpus[2]
    spec["queries"][0][key] = "chair-a"
    with pytest.raises(ValueError, match="unsupported fields"):
        prepare(corpus, spec)


@pytest.mark.parametrize("change", ["duplicate_query", "duplicate_gallery", "empty_text", "empty_images", "null_pc",
                                     "missing_provenance", "empty_creator", "no_queries", "no_gallery"])
def test_ids_and_raw_metadata_validated(corpus, change):
    spec = corpus[2]
    q = spec["queries"][0]
    if change == "duplicate_query": spec["queries"].append(copy.deepcopy(q))
    elif change == "duplicate_gallery": spec["gallery_uids"].append("chair-a")
    elif change == "empty_text": q["text"] = " "
    elif change == "empty_images": q["images"] = []
    elif change == "null_pc": q["pointcloud"] = None
    elif change == "missing_provenance": del q["provenance"]["source"]
    elif change == "empty_creator": q["provenance"]["creator"] = ""
    elif change == "no_queries": spec["queries"] = []
    elif change == "no_gallery": spec["gallery_uids"] = []
    with pytest.raises(ValueError):
        prepare(corpus, spec)


@pytest.mark.parametrize("which,whitespace", [("query", " "), ("query", "\n"),
                                              ("gallery", " "), ("gallery", "\t")])
def test_ids_match_scorer_contract_before_reading_canonical_sources(corpus, which, whitespace):
    spec = corpus[2]
    if which == "query":
        spec["queries"][0]["query_id"] += whitespace
    else:
        spec["gallery_uids"][0] = whitespace + spec["gallery_uids"][0]
    with pytest.raises(ValueError, match="surrounding whitespace"):
        prepare(corpus, spec)
    assert not corpus[1].exists()


@pytest.mark.parametrize("kind", ["shape", "dtype", "nan", "rgb", "degenerate", "constant_offset", "archive"])
def test_invalid_external_pointcloud_refused(corpus, kind):
    spec = multimodal(corpus[2])
    p = corpus[0].parent / "reference.npy"
    a = np.load(p)
    if kind == "shape": a = a[:5]
    elif kind == "dtype": a = a.astype(np.float64)
    elif kind == "nan": a[0, 0] = np.nan
    elif kind == "rgb": a[0, 3] = 1.1
    elif kind == "degenerate": a[:, :3] = 0
    elif kind == "constant_offset": a[:, :3] = 8
    if kind == "archive":
        with p.open("wb") as fh: np.savez(fh, embeddings=np.ones(4))
    else:
        np.save(p, a)
    with pytest.raises(ValueError):
        prepare(corpus, spec)
    assert not corpus[1].exists()


def test_invalid_image_is_rejected_before_publication(corpus):
    spec = multimodal(corpus[2])
    (corpus[0].parent / "reference.png").write_bytes(b"not an image")
    with pytest.raises((ValueError, OSError)):
        prepare(corpus, spec)
    assert not corpus[1].exists()


@pytest.mark.parametrize("key", ["embedding", "cloud", "annotation", "embedding_sidecar", "image", "pointcloud", "spec", "encoding"])
def test_modified_bound_source_is_rejected(corpus, key):
    dest = prepare(corpus, multimodal(corpus[2]))
    obj = json.loads(dest.read_text())
    source = (obj["queries"][0]["images"][0] if key == "image" else
              obj["queries"][0]["pointcloud"] if key == "pointcloud" else
              obj["spec_source"] if key == "spec" else
              obj["encoding_protocol"] if key == "encoding" else obj["records"]["chair-a"][key])
    with Path(source["path"]).open("ab") as fh: fh.write(b"\nchanged")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        m.load_protocol(dest)


@pytest.mark.parametrize("change", ["qrels", "text", "condition", "image_path", "gallery_order", "query_order", "status", "choices", "uid_record"])
def test_manifest_cannot_silently_change_spec_interpretation(corpus, change):
    dest = prepare(corpus, multimodal(corpus[2]))
    obj = json.loads(dest.read_text())
    q = obj["queries"][0]
    if change == "qrels": q["qrels"]["text"]["chair-a"] = 0
    elif change == "text": q["text"] = "a different intended room"
    elif change == "condition": q["conditions"] = ["text"]
    elif change == "image_path": q["images"][0]["path"] = str(corpus[0])
    elif change == "gallery_order": obj["gallery_uids"].reverse()
    elif change == "query_order": obj["query_ids"] = ["different-query"]
    elif change == "status": obj["status"] = "partial"
    elif change == "choices": obj["choices"]["unjudged_policy"] = "negative"
    elif change == "uid_record": obj["records"]["chair-a"] = obj["records"]["chair-b"]
    write_json(dest, obj)
    with pytest.raises(ValueError):
        m.load_protocol(dest)


@pytest.mark.parametrize("malformation", ["duplicate", "nan", "overflow"])
def test_spec_json_is_strict(corpus, malformation):
    p, _, spec = corpus
    text = json.dumps(spec)
    if malformation == "duplicate": text = text.replace('"schema":', '"schema": "ignored", "schema":', 1)
    elif malformation == "nan": text = text.replace('"chair-a": 1', '"chair-a": NaN')
    else: text = text.replace('"chair-a": 1', '"chair-a": 1e999')
    p.write_text(text)
    with pytest.raises(ValueError):
        prepare(corpus)


@pytest.mark.parametrize("missing", ["embedding", "cloud", "annotation", "embedding_sidecar"])
def test_canonical_input_missing_is_not_silently_dropped(corpus, missing):
    root, suffix = {"embedding": (m.paths.EMBEDDINGS, ".npz"), "cloud": (m.paths.POINTCLOUDS, ".npz"),
                    "annotation": (m.paths.ANNOTATIONS, ".json"),
                    "embedding_sidecar": (m.paths.EMBEDDINGS, ".json")}[missing]
    (root / ("chair-a" + suffix)).unlink()
    with pytest.raises(FileNotFoundError):
        prepare(corpus)
    assert not corpus[1].exists()


@pytest.mark.parametrize("change", ["text", "view_count", "source_uid", "nonfinite_embedding", "pc_size"])
def test_canonical_consistency_uses_actual_sources(corpus, change):
    p = m.paths.EMBEDDINGS / "chair-a.json"
    sidecar = json.loads(p.read_text())
    if change == "text": sidecar["text"] = "stale annotation"
    elif change == "view_count": sidecar["n_views"] = 12
    elif change == "source_uid": sidecar["uid"] = "chair-b"
    elif change == "nonfinite_embedding":
        np.savez(m.paths.EMBEDDINGS / "chair-a.npz", text=np.ones(4), image=np.ones(4), views=np.full((11, 4), np.nan))
    elif change == "pc_size":
        np.savez(m.paths.POINTCLOUDS / "chair-a.npz", xyz=np.ones((5, 3)), rgb=np.ones((5, 3)))
    write_json(p, sidecar)
    with pytest.raises(ValueError):
        prepare(corpus)


def test_existing_directory_and_symlink_are_never_overwritten(corpus):
    dest = prepare(corpus)
    before = dest.read_bytes()
    with pytest.raises(FileExistsError): prepare(corpus)
    assert dest.read_bytes() == before
    alias = corpus[1].parent / "alias"
    alias.symlink_to(corpus[1], target_is_directory=True)
    with pytest.raises(FileExistsError): m.prepare_protocol(corpus[0], alias)
    assert dest.read_bytes() == before


def test_verified_bytes_are_the_bytes_that_passed_hash_check(tmp_path):
    p = tmp_path / "raw"
    p.write_bytes(b"verified image or pointcloud bytes")
    source = m._capture(p)
    assert m.verified_source_bytes(source) == p.read_bytes()
    p.write_bytes(b"other")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        m.verified_source_bytes(source)


def test_real_cli_uses_spec_relative_paths_and_publishes_a_loadable_manifest(corpus):
    source, out, _ = corpus
    env = dict(os.environ, METAFIND_DATA=str(m.paths.DATA), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-m", "metafind.eval.intent_protocol", "--spec", str(source),
                             "--out-dir", str(out)], cwd=m.paths.REPO, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert m.load_protocol(out / "protocol.json")["query_ids"] == ["intent-dining-seating"]
