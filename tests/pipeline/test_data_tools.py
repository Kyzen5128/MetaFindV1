"""Filesystem regressions for diagnostic packs and generated split views."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def split_view(monkeypatch, tmp_path):
    from tools import materialize_split_dirs as m

    src = tmp_path / "source"
    src.mkdir()
    for uid in ("a", "b"):
        (src / f"{uid}.json").write_text(uid)
    monkeypatch.setattr(m, "KINDS", {"annotations": (src, ".json")})
    splits, out = tmp_path / "splits.json", tmp_path / "view"
    splits.write_text(json.dumps({"object": {"train": ["a"], "val": ["b"]}}))
    monkeypatch.setattr("sys.argv", ["materialize_split_dirs", "--splits", str(splits), "--out", str(out)])
    assert m.main() == 0
    return m, splits, out, src


def test_split_view_rebuild_accepts_its_metadata_and_updates_membership(split_view):
    m, splits, out, src = split_view
    splits.write_text(json.dumps({"object": {"train": ["b"], "test": ["a"]}}))
    assert m.main() == 0
    assert not (out / "val").exists()
    assert (out / "train" / "annotations" / "b.json").resolve() == src / "b.json"
    assert (out / "train" / "uids.txt").read_text() == "b\n"
    assert (src / "a.json").read_text() == "a"


def test_split_view_checks_all_real_files_before_unlinking(split_view):
    m, _, out, src = split_view
    foreign = out / "train" / "annotations" / "notes.txt"
    foreign.write_text("user notes")
    link = out / "val" / "annotations" / "b.json"
    with pytest.raises(SystemExit, match="real file"):
        m.main()
    assert foreign.read_text() == "user notes"
    assert link.is_symlink() and link.resolve() == src / "b.json"
    assert (out / "README.txt").exists()


def test_split_view_does_not_claim_an_unmarked_readme(split_view):
    m, _, out, _ = split_view
    (out / "README.txt").write_text("my own directory")
    with pytest.raises(SystemExit, match="real file"):
        m.main()
    assert (out / "train" / "annotations" / "a.json").is_symlink()


@pytest.mark.parametrize("arm", ["text", "pc"])
def test_limited_pack_keeps_full_array_and_manifest_shard(monkeypatch, tmp_path, arm):
    from tools import make_query_pack as m
    from metafind.data import encode_text_image

    pack = tmp_path / "pack"
    pack.mkdir()
    monkeypatch.setattr(m, "PACK", pack)
    monkeypatch.setattr(m, "MANIFEST", pack / "query_pack.json")
    monkeypatch.setattr(m.paths, "OUTPUTS", tmp_path)
    monkeypatch.setattr(m.paths, "ANNOTATIONS", tmp_path)
    monkeypatch.setattr(m.runlog, "code_revision", lambda: "test")
    monkeypatch.setattr(m.runlog, "code_dirty", lambda: True)
    (tmp_path / "splits.json").write_text(json.dumps({"object": {"train": ["a", "b", "c"]}}))
    for uid in ("a", "b", "c"):
        (tmp_path / f"{uid}.json").write_text(json.dumps({"description": uid}))
    suffix = "" if arm == "text" else f"_offset{m.PC_SEED_OFFSET}"
    full = pack / f"query_{arm}_train{suffix}.npy"
    np.save(full, np.arange(6).reshape(3, 2))
    original = full.read_bytes()
    full_shard = {"array": str(full), "uid_order": ["a", "b", "c"], "tag": "train"}
    m.MANIFEST.write_text(json.dumps({arm: {"shards": [full_shard]}}))

    if arm == "text":
        # Use the actual text builder and publication code; only the model is fake.
        import torch
        monkeypatch.setattr(m, "pick_alternate", lambda ann: (1, ann["description"]))
        backbone = SimpleNamespace(encode_text=lambda texts: torch.ones((len(texts), 1280)))
        monkeypatch.setattr(encode_text_image, "Encoder", lambda **kw: SimpleNamespace(torch=torch, backbone=backbone))
    else:
        # The same tag also names the PC builder's resumable array/done-list.
        def build_pc(uids, tag, workers):
            dest = pack / f"query_pc_{tag}{suffix}.npy"
            np.save(dest, np.ones((len(uids), 2)))
            dest.with_suffix(".npy.done.jsonl").write_text("limited progress\n")
            return {"array": str(dest), "uid_order": uids}
        monkeypatch.setattr(m, "build_pc", build_pc)
        full.with_suffix(".npy.done.jsonl").write_text("full progress\n")
    monkeypatch.setattr("sys.argv", ["make_query_pack", "--arm", arm, "--split", "train", "--limit", "1", "--device", "cpu"])
    assert m.main() == 0
    assert full.read_bytes() == original
    shards = json.loads(m.MANIFEST.read_text())[arm]["shards"]
    assert shards[0] == full_shard
    assert shards[1]["tag"] == "train_limit1"
    assert Path(shards[1]["array"]) != full
    assert np.load(shards[1]["array"]).shape[0] == 1
    if arm == "pc":
        assert full.with_suffix(".npy.done.jsonl").read_text() == "full progress\n"


@pytest.mark.parametrize("versions, expected", [([8, 8], "sampler_version 8"),
                                               ([8, 9], "MIXED"),
                                               ([8, None], "MIXED"),
                                               ([None, None], "UNKNOWN")])
def test_dataset_manifest_records_versions_of_included_clouds(monkeypatch, tmp_path, versions, expected):
    from tools import build_dataset_manifest as m

    for key, value in {"LOGS": tmp_path, "OUTPUTS": tmp_path,
                       "LVIS_MANIFEST": tmp_path / "lvis.json",
                       "ANNOTATIONS": tmp_path / "ann", "EMBEDDINGS": tmp_path / "emb",
                       "POINTCLOUDS": tmp_path / "pc", "OBJAVERSE_GLB": tmp_path / "glbs"}.items():
        monkeypatch.setattr(m.paths, key, value)
    monkeypatch.setattr(m, "_vendor_sha", lambda: "test")
    (tmp_path / "lvis.json").write_text(json.dumps(["a", "b", "c"]))
    (tmp_path / "splits.json").write_text(json.dumps({"object": {"train": ["a", "b"], "test": ["c"]}}))
    # The third row is outside --limit and must not contaminate its version/count.
    records = [{"uid": uid, "sampler_version": version}
               for uid, version in zip(("a", "b", "c"), [*versions, 99])]
    (tmp_path / "pointclouds_index.jsonl").write_text("\n".join(map(json.dumps, records)))
    out = tmp_path / "manifest"
    m.build(out, limit=2)
    pc = next(c for c in json.loads((out / "caches.json").read_text())["caches"] if c["name"] == "pointclouds")
    assert pc["preprocessing_version"] == expected
    assert pc["n"] == 2
    assert sum(pc["sampler_version_counts"].values()) == 2
    assert "99" not in pc["sampler_version_counts"]


@pytest.mark.parametrize("supply_index", [False, True])
def test_archive_note_requires_artifact_evidence_for_sampler_version(monkeypatch, tmp_path, supply_index):
    from tools import write_archive_manifest as m

    archive = tmp_path / "archive"
    archive.mkdir()
    argv = ["write_archive_manifest", str(archive)]
    if supply_index:
        index = tmp_path / "pointclouds_index.jsonl"
        index.write_text(json.dumps({"uid": "a", "sampler_version": 8}) + "\n")
        argv += ["--pointcloud-index", str(index)]
    monkeypatch.setattr("sys.argv", argv)
    assert m.main() == 0
    note = (archive / "ARCHIVED.md").read_text()
    assert ("sampler_version 8" if supply_index else "sampler version: UNKNOWN") in note
    assert "sampler_version 9" not in note
