"""CPU fixture tests for custom observation inputs; no corpus or model writes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from metafind.eval import custom_protocol as m


def write_json(path, obj):
    path.write_text(json.dumps(obj))


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    data = tmp_path / "data"
    for key, rel in {"DATA": "", "OUTPUTS": "outputs", "EMBEDDINGS": "outputs/embeddings",
                     "POINTCLOUDS": "outputs/pointclouds", "ANNOTATIONS": "outputs/annotations",
                     "OBJAVERSE_GLB": "datasets/glbs"}.items():
        p = data / rel
        p.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(m.paths, key, p)
    enc = {"status": "resolved", "actual_clip_train_scope": "frozen",
           "image_aggregation": "mean", "view_aggregation": {"n_views": 11},
           "text_template": "__json__", "text_serialization": "fixture-json"}
    write_json(m.paths.OUTPUTS / "stage1_encoding_protocol.json", enc)
    for uid in ("a", "b"):
        ann = {"uid": uid, "description": "canonical " + uid, "category": "chair",
               "width": 10, "length": 20, "height": 30, "volume": 6000, "mass": 2}
        write_json(m.paths.ANNOTATIONS / f"{uid}.json", ann)
        canonical = m.serialize_annotation(ann, template="__json__")
        views = np.arange(44, dtype=np.float32).reshape(11, 4) + 1
        np.savez(m.paths.EMBEDDINGS / f"{uid}.npz", text=np.ones(4), image=views.mean(0), views=views)
        write_json(m.paths.EMBEDDINGS / f"{uid}.json",
                   {"uid": uid, "text": canonical, "n_views": 11, "aggregation": "mean",
                    "text_serialization": "fixture-json", "ulip2_ckpt_sha": "test-encoder"})
        xyz = np.resize(np.eye(3), (10000, 3)).astype(np.float32)
        np.savez(m.paths.POINTCLOUDS / f"{uid}.npz", xyz=xyz, rgb=np.ones((10000, 3)))
        (m.paths.OBJAVERSE_GLB / f"{uid}.glb").write_bytes(uid.encode())
    calls = []
    def sample(path, seed, n):
        calls.append((path, seed, n))
        rng = np.random.default_rng(seed)
        return rng.normal(size=(n, 3)), rng.random((n, 3)), None, "fixture"
    monkeypatch.setattr(m.pointclouds, "sample_mesh", sample)
    return tmp_path / "prepared", calls


def prepare(corpus, **kwargs):
    out, _ = corpus
    return m.prepare_protocol(["a"], ["a", "b"], out,
                              query_texts={"a": "a different observation"},
                              provenance={"split": "fixture"}, **kwargs)


def test_prepares_disjoint_11_view_sources_and_resampled_cloud(corpus):
    manifest = prepare(corpus)
    obj = m.load_protocol(manifest)
    assert obj["status"] == "complete"
    assert not manifest.with_suffix(".json.part").exists()
    assert obj["n_views"] == 11
    assert obj["encoding_protocol"]["content"]["text_template"] == "__json__"
    for uid, rec in obj["records"].items():
        assert len(rec["gallery_views"]) == 10
        assert set(rec["gallery_views"]) | {rec["query_view"]} == set(range(11))
        assert rec["query_view"] not in rec["gallery_views"]
        assert rec["embedding_sidecar"]["content"]["ulip2_ckpt_sha"] == "test-encoder"
    assert "query_cloud" not in obj["records"]["b"]
    cloud = np.load(obj["records"]["a"]["query_cloud"]["path"])
    assert cloud.shape == (10000, 6) and cloud.dtype == np.float32
    assert np.allclose(cloud[:, :3].mean(0), 0, atol=1e-6)
    assert np.isclose(np.linalg.norm(cloud[:, :3], axis=1).max(), 1)
    assert corpus[1][0][1:] == (m.pointclouds.uid_seed("a") + 1000003, 10000)
    assert "runtime_required" in obj["choices"]["effective_text_tokens"]


@pytest.mark.parametrize("missing", ["embedding", "annotation", "cloud", "sidecar", "mesh"])
def test_missing_source_refused_before_output_or_sampling(corpus, missing):
    roots = {"embedding": (m.paths.EMBEDDINGS, ".npz"), "sidecar": (m.paths.EMBEDDINGS, ".json"),
             "annotation": (m.paths.ANNOTATIONS, ".json"), "cloud": (m.paths.POINTCLOUDS, ".npz"),
             "mesh": (m.paths.OBJAVERSE_GLB, ".glb")}
    root, suffix = roots[missing]
    (root / ("b" + suffix)).unlink()
    with pytest.raises(FileNotFoundError):
        prepare(corpus)
    assert not corpus[0].exists() and not corpus[1]


@pytest.mark.parametrize("q,g,texts", [(["a", "a"], ["a", "b"], {"a": "x"}),
                                     (["a"], ["a", "a"], {"a": "x"}),
                                     (["c"], ["a", "b"], {"c": "x"}),
                                     (["a"], ["a", "b"], {}),
                                     (["a"], ["a", "b"], {"a": "x", "b": "y"}),
                                     (["../a"], ["../a"], {"../a": "x"})])
def test_pool_and_text_coverage_strict(corpus, q, g, texts):
    with pytest.raises(ValueError):
        m.prepare_protocol(q, g, corpus[0], query_texts=texts, provenance={})
    assert not corpus[0].exists()


def test_existing_destination_is_never_overwritten(corpus):
    dest = prepare(corpus)
    before = dest.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(corpus)
    assert dest.read_bytes() == before
    assert len(corpus[1]) == 1


@pytest.mark.parametrize("key", ["embedding", "cloud", "annotation", "embedding_sidecar", "mesh", "query_cloud"])
def test_load_refuses_modified_file(corpus, key):
    dest = prepare(corpus)
    obj = json.loads(dest.read_text())
    path = Path(obj["records"]["a"][key]["path"])
    with path.open("ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        m.load_protocol(dest)


@pytest.mark.parametrize("change", ["uid", "views", "text", "query_cloud_missing"])
def test_load_checks_semantics_not_only_file_hashes(corpus, change):
    dest = prepare(corpus)
    obj = json.loads(dest.read_text())
    rec = obj["records"]["a"]
    if change == "uid":
        rec["embedding"] = obj["records"]["b"]["embedding"]
    elif change == "views":
        rec["gallery_views"] = list(range(11))
    elif change == "text":
        rec["query_text"] = rec["canonical_text"]
    else:
        del rec["query_cloud"]
    write_json(dest, obj)
    with pytest.raises(ValueError):
        m.load_protocol(dest)


def test_mixed_view_counts_fail_preflight(corpus):
    np.savez(m.paths.EMBEDDINGS / "b.npz", text=np.ones(4), image=np.ones(4), views=np.ones((12, 4)))
    with pytest.raises(ValueError, match="view count"):
        prepare(corpus)
    assert not corpus[0].exists()


def test_annotation_change_cannot_reuse_old_text_sidecar(corpus):
    p = m.paths.ANNOTATIONS / "a.json"
    ann = json.loads(p.read_text())
    ann["description"] = "changed"
    write_json(p, ann)
    with pytest.raises(ValueError, match="canonical text"):
        prepare(corpus)


def test_cloud_sidecar_identity_is_preserved_and_checked(corpus):
    path = m.paths.POINTCLOUDS / "a.npz"
    write_json(path.with_suffix(".json"), {"uid": "a", "sampler_version": 8,
                                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    obj = m.load_protocol(prepare(corpus))
    assert obj["records"]["a"]["cloud_sidecar"]["content"]["sampler_version"] == 8
    assert obj["choices"]["query_pc"]["sampler_version"] == m.pointclouds.SAMPLER_VERSION


def test_cli_binds_input_file_provenance(corpus, tmp_path):
    q, g, t = [tmp_path / name for name in ("q.json", "g.json", "t.json")]
    for p, data in ((q, ["a"]), (g, ["a", "b"]), (t, {"a": "new raw caption"})):
        write_json(p, data)
    assert m.main(["--query-uids", str(q), "--gallery-uids", str(g),
                   "--query-texts", str(t), "--out-dir", str(corpus[0])]) == 0
    rec = m.load_protocol(corpus[0] / "protocol.json")["provenance"]["input_files"]["query_texts"]
    assert rec == {"path": str(t), "sha256": hashlib.sha256(t.read_bytes()).hexdigest()}
    t.write_text('{"a": "edited"}')
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        m.load_protocol(corpus[0] / "protocol.json")


def test_12_view_corpus_uses_1_and_11_without_literal_modulus(corpus):
    ep = m.paths.OUTPUTS / "stage1_encoding_protocol.json"
    enc = json.loads(ep.read_text())
    enc["view_aggregation"]["n_views"] = 12
    write_json(ep, enc)
    for uid in ("a", "b"):
        np.savez(m.paths.EMBEDDINGS / f"{uid}.npz", text=np.ones(4), image=np.ones(4), views=np.ones((12, 4)))
        p = m.paths.EMBEDDINGS / f"{uid}.json"
        sc = json.loads(p.read_text())
        sc["n_views"] = 12
        write_json(p, sc)
    obj = m.load_protocol(prepare(corpus))
    for uid, rec in obj["records"].items():
        assert rec["query_view"] == m.pointclouds.uid_seed(uid) % 12
        assert len(rec["gallery_views"]) == 11


@pytest.mark.parametrize("n", [0, 1, 11.0, True])
def test_invalid_protocol_view_count_rejected_before_sampling(corpus, n):
    p = m.paths.OUTPUTS / "stage1_encoding_protocol.json"
    enc = json.loads(p.read_text())
    enc["view_aggregation"]["n_views"] = n
    write_json(p, enc)
    with pytest.raises(ValueError, match="n_views"):
        prepare(corpus)
    assert not corpus[0].exists() and not corpus[1]


def test_failed_sampling_never_publishes_complete_manifest(corpus, monkeypatch):
    def fail(*args):
        raise RuntimeError("fixture sampler failure")
    monkeypatch.setattr(m.pointclouds, "sample_mesh", fail)
    with pytest.raises(RuntimeError, match="sampler failure"):
        prepare(corpus)
    assert corpus[0].is_dir()
    assert not (corpus[0] / "protocol.json").exists()


def test_nonfinite_sampler_output_is_not_published(corpus, monkeypatch):
    def bad(*args):
        return np.full((10000, 3), np.nan), np.ones((10000, 3)), None
    monkeypatch.setattr(m.pointclouds, "sample_mesh", bad)
    with pytest.raises(ValueError, match="finite"):
        prepare(corpus)
    assert not (corpus[0] / "protocol.json").exists()


def test_duplicate_mesh_uid_is_ambiguous(corpus):
    dup = m.paths.OBJAVERSE_GLB / "other"
    dup.mkdir()
    (dup / "a.glb").write_bytes(b"another mesh")
    with pytest.raises(ValueError, match="duplicate mesh UID"):
        prepare(corpus)
    assert not corpus[0].exists()


@pytest.mark.parametrize("n", [3, 9999, 10001])
def test_canonical_point_count_is_preflighted_before_sampling(corpus, n):
    np.savez(m.paths.POINTCLOUDS / "b.npz", xyz=np.ones((n, 3)), rgb=np.ones((n, 3)))
    with pytest.raises(ValueError, match="canonical xyz/rgb"):
        prepare(corpus)
    assert not corpus[0].exists() and not corpus[1]


@pytest.mark.parametrize("status", [None, "preparing", "failed"])
def test_incomplete_manifest_is_rejected_even_without_hash_checks(corpus, status):
    dest = prepare(corpus)
    obj = json.loads(dest.read_text())
    obj["status"] = status
    write_json(dest, obj)
    with pytest.raises(ValueError, match="not complete"):
        m.load_protocol(dest, verify=False)


def test_sampler_source_bytes_are_bound(corpus, tmp_path):
    dest = prepare(corpus)
    obj = json.loads(dest.read_text())
    sampler = obj["choices"]["query_pc"]
    assert sampler["sampler_version"] == m.pointclouds.SAMPLER_VERSION
    assert sampler["sampler_source"]["sha256"] == hashlib.sha256(Path(m.pointclouds.__file__).read_bytes()).hexdigest()
    fake = tmp_path / "sampler.py"
    fake.write_text("changed implementation")
    sampler["sampler_source"]["path"] = str(fake)
    write_json(dest, obj)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        m.load_protocol(dest)


def test_manifest_write_failure_leaves_only_partial_file(corpus, monkeypatch):
    def fail_dump(obj, fh, **kwargs):
        fh.write('{"schema":')
        raise OSError("fixture interrupted write")
    monkeypatch.setattr(m.json, "dump", fail_dump)
    with pytest.raises(OSError, match="interrupted write"):
        prepare(corpus)
    assert not (corpus[0] / "protocol.json").exists()
    assert (corpus[0] / "protocol.json.part").exists()


def test_atomic_publication_does_not_replace_a_concurrent_file(corpus, monkeypatch):
    original = m.os.link
    def competing_writer(src, dest):
        Path(dest).write_text("other writer")
        return original(src, dest)
    monkeypatch.setattr(m.os, "link", competing_writer)
    with pytest.raises(FileExistsError):
        prepare(corpus)
    assert (corpus[0] / "protocol.json").read_text() == "other writer"
    assert (corpus[0] / "protocol.json.part").exists()


def test_prepare_reports_each_phase(corpus, capsys):
    prepare(corpus)
    output = capsys.readouterr().out
    assert "preflight canonical sources: 0/2" in output
    assert "preflight canonical sources: 2/2" in output
    assert "sample query clouds: 0/1" in output
    assert "sample query clouds: 1/1" in output


def test_sampler_version_cannot_be_changed_without_implementation(corpus):
    dest = prepare(corpus)
    obj = json.loads(dest.read_text())
    obj["choices"]["query_pc"]["sampler_version"] += 1
    write_json(dest, obj)
    with pytest.raises(ValueError, match="sampler version"):
        m.load_protocol(dest)
