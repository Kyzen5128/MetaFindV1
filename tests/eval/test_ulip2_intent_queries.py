"""Isolated real tar/NPY extraction: source joins, captions, and draft boundaries."""
from __future__ import annotations

import hashlib
import io
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest

from tools import prepare_ulip2_intent_queries as adapter


A, B, C = "a" * 32, "b" * 32, "c" * 32


def _npy(value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=True)
    return stream.getvalue()


def _record(**changes):
    result = {"dataset": "Objaverse", "group": "000-000", "id": A,
              "blip_caption": "  a wooden chair  ", "msft_caption": "a brown seat",
              "text": ["Chair"], "xyz": np.array([[1., 2., 3.], [-1., 0., 2.]]),
              "rgb": np.array([[0., .5, 1.], [.25, .75, 0.]]),
              "image_feat": np.zeros((12, 1280), dtype=np.float16),
              "thumbnail_feat": np.zeros(1280, dtype=np.float16),
              "blip_caption_feat": {"original": np.zeros(1280, dtype=np.float16)}}
    result.update(changes)
    return result


def _tar(path, records):
    with tarfile.open(path, "w:gz") as archive:
        for name, raw in records:
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))


def _fixture(tmp_path, *, uids=None, records=None, entries=None):
    uids = uids if uids is not None else [A]
    uid_path, metadata_path = tmp_path / "uids.json", tmp_path / "metadata.json"
    uid_path.write_text(json.dumps(uids))
    metadata_path.write_text(json.dumps({"entries": entries if entries is not None else
                                         [{"u": uid, "glb": f"glbs/000-000/{uid}.glb"} for uid in uids]}))
    shards = tmp_path / "shards"
    shards.mkdir()
    records = records if records is not None else [(f"000-000/{A}.npy", _npy(_record()))]
    _tar(shards / "000-000.tar.gz", records)
    return dict(shards=shards, asset_uids=uid_path, metadata=metadata_path,
                caption_field="blip_caption", out_dir=tmp_path / "draft")


def _keys_recursive(value):
    if isinstance(value, dict):
        return set(value).union(*(_keys_recursive(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys_recursive(v) for v in value))
    return set()


def test_real_tar_draft_preserves_raw_caption_provenance_without_embeddings_or_qrels(tmp_path):
    raw = _npy(_record())
    kwargs = _fixture(tmp_path, records=[(f"000-000/{A}.npy", raw)])
    # An unrelated shard is intentionally unreadable as a tar. Its existence
    # must not lead to scanning every shard under the supplied directory.
    (kwargs["shards"] / "000-159.tar.gz").write_bytes(b"not a tar")
    result = adapter.prepare_queries(**kwargs)
    assert json.loads((kwargs["out_dir"] / "draft_queries.json").read_text()) == result
    assert result["status"] == "needs_relevance_review"
    assert result["failures"] == []
    draft = result["records"][0]
    assert draft["id"] == draft["query_id"] == f"ulip2-{A}"
    assert draft["source_uid"] == A
    assert draft["text"] == "  a wooden chair  "  # no rewriting or whitespace stripping
    assert draft["conditions"] == ["text"]
    assert not {"qrels", "judgments", "relevant_uids", "positive_uids", "images",
                "image_feat", "thumbnail_feat", "blip_caption_feat"} & _keys_recursive(result)
    assert list(result["sources"]["archives"]) == ["000-000"]
    for key in ("asset_uids", "metadata"):
        source = result["sources"][key]
        assert source["sha256"] == hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest()
    provenance = draft["provenance"]
    assert provenance["member"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert provenance["member"]["name"] == f"000-000/{A}.npy"
    assert provenance["archive"]["stat"]["size_bytes"] == (kwargs["shards"] / "000-000.tar.gz").stat().st_size
    assert provenance["caption"]["field"] == "blip_caption"
    assert provenance["caption"]["utf8_sha256"] == hashlib.sha256(draft["text"].encode()).hexdigest()
    assert not (kwargs["out_dir"] / "pointclouds").exists()


def test_optional_pointcloud_is_raw_xyzrgb_float32_with_no_resampling(tmp_path):
    record = _record()
    kwargs = _fixture(tmp_path, records=[(f"000-000/{A}.npy", _npy(record))])
    result = adapter.prepare_queries(**kwargs, include_pointcloud=True)
    draft = result["records"][0]
    pc = np.load(draft["pointcloud"], allow_pickle=False)
    np.testing.assert_array_equal(pc, np.concatenate((record["xyz"], record["rgb"]), axis=1).astype(np.float32))
    assert pc.shape == (2, 6)  # downstream protocol, not this draft, enforces 10,000
    assert pc.dtype == np.float32
    assert draft["conditions"] == ["text", "pc", "text+pc"]
    provenance = draft["provenance"]["pointcloud"]
    assert provenance["sha256"] == hashlib.sha256(Path(draft["pointcloud"]).read_bytes()).hexdigest()


@pytest.mark.parametrize("field,value,expected", [("msft_caption", "the MSFT caption", "the MSFT caption"),
                                                 ("text", ["the name"], "the name"),
                                                 ("text", "literal name", "literal name")])
def test_exact_field_selection_does_not_replace_it_with_blip(tmp_path, field, value, expected):
    kwargs = _fixture(tmp_path, records=[(f"000-000/{A}.npy", _npy(_record(**{field: value})))])
    kwargs["caption_field"] = field
    result = adapter.prepare_queries(**kwargs)
    assert result["records"][0]["text"] == expected
    assert result["records"][0]["provenance"]["caption"]["field"] == field


@pytest.mark.parametrize("field,value", [("blip_caption", ""), ("blip_caption", " \n"),
                                        ("blip_caption", None), ("blip_caption", ["caption"]),
                                        ("text", []), ("text", ["one", "two"]), ("text", [""])])
def test_empty_or_ambiguous_caption_is_a_failure_without_fallback_or_qrels(tmp_path, field, value):
    kwargs = _fixture(tmp_path, records=[(f"000-000/{A}.npy", _npy(_record(**{field: value})))])
    kwargs["caption_field"] = field
    result = adapter.prepare_queries(**kwargs)
    assert result["records"] == []
    assert result["failures"][0]["source_uid"] == A
    assert result["failures"][0]["reason"] == "invalid_record"
    assert "caption" in result["failures"][0]["detail"]
    assert not {"qrels", "judgments"} & _keys_recursive(result)


def test_missing_caption_field_is_reported_despite_available_alternative(tmp_path):
    record = _record()
    del record["blip_caption"]
    result = adapter.prepare_queries(**_fixture(tmp_path, records=[(f"000-000/{A}.npy", _npy(record))]))
    assert result["records"] == []
    assert "missing caption field: blip_caption" in result["failures"][0]["detail"]


def test_requested_uid_order_and_source_member_join_are_explicit(tmp_path):
    kwargs = _fixture(tmp_path, uids=[B, A], records=[(f"000-000/{A}.npy", _npy(_record(uid=A))),
                                                   (f"000-000/{B}.npy", _npy(_record(id=B, uid=B, blip_caption="B")))])
    result = adapter.prepare_queries(**kwargs)
    assert [r["source_uid"] for r in result["records"]] == [B, A]
    assert [r["text"] for r in result["records"]] == ["B", "  a wooden chair  "]


@pytest.mark.parametrize("mismatch", ["metadata", "record", "member", "duplicate_metadata", "duplicate_member"])
def test_source_join_mismatch_cannot_produce_a_valid_draft_record(tmp_path, mismatch):
    kwargs = _fixture(tmp_path)
    archive = kwargs["shards"] / "000-000.tar.gz"
    if mismatch in ("metadata", "duplicate_metadata"):
        entries = [{"u": A, "glb": f"glbs/000-000/{B if mismatch == 'metadata' else A}.glb"}]
        if mismatch == "duplicate_metadata": entries *= 2
        kwargs["metadata"].write_text(json.dumps({"entries": entries}))
    elif mismatch == "record":
        _tar(archive, [(f"000-000/{A}.npy", _npy(_record(uid=B)))])
    elif mismatch == "member":
        _tar(archive, [(f"../000-123/{A}.npy", _npy(_record()))])
    else:
        _tar(archive, [(f"000-000/{A}.npy", _npy(_record()))] * 2)
    if mismatch == "record":
        result = adapter.prepare_queries(**kwargs)
        assert result["records"] == []
        assert "disagrees" in result["failures"][0]["detail"]
    else:
        with pytest.raises(ValueError, match="join mismatch|duplicate"):
            adapter.prepare_queries(**kwargs)
        assert not (kwargs["out_dir"] / "draft_queries.json").exists()


def test_missing_metadata_archive_and_member_are_distinct_failures(tmp_path):
    kwargs = _fixture(tmp_path, uids=[C, B, A], records=[], entries=[
        {"u": A, "glb": f"glbs/000-000/{A}.glb"}, {"u": B, "glb": f"glbs/000-001/{B}.glb"}])
    result = adapter.prepare_queries(**kwargs)
    assert [(r["source_uid"], r["reason"]) for r in result["failures"]] == [
        (C, "missing_metadata"), (B, "missing_archive"), (A, "missing_member")]
    assert result["records"] == []


@pytest.mark.parametrize("field,value", [("id", B), ("id", None), ("group", "000-001"),
                                        ("dataset", "ShapeNet")])
def test_embedded_identity_must_match_filename_and_metadata(tmp_path, field, value):
    kwargs = _fixture(tmp_path, records=[(f"000-000/{A}.npy", _npy(_record(**{field: value})))])
    result = adapter.prepare_queries(**kwargs)
    assert result["records"] == []
    assert f"record {field}" in result["failures"][0]["detail"]


def test_numpy_only_decoder_rejects_unknown_globals_even_in_unused_extra_field(tmp_path):
    sentinel = tmp_path / "must_not_exist"

    class Unexpected:
        def __reduce__(self):
            return eval, (f"open({str(sentinel)!r}, 'w').write('unsafe')",)

    raw = _npy(_record(unused=Unexpected()))
    with pytest.raises(pickle.UnpicklingError, match="forbidden pickle global"):
        adapter.decode_numpy_record(raw)
    kwargs = _fixture(tmp_path, records=[(f"000-000/{A}.npy", raw)])
    result = adapter.prepare_queries(**kwargs)
    assert result["records"] == []
    assert "forbidden pickle global" in result["failures"][0]["detail"]
    assert not sentinel.exists()


@pytest.mark.parametrize("version", [(1, 0), (2, 0)])
def test_numpy_decoder_accepts_only_expected_header_and_numpy_constructors(version):
    buffer = io.BytesIO()
    value = np.array(_record(scalar=np.float32(.5)), dtype=object)
    np.lib.format.write_array(buffer, value, version=version, allow_pickle=True)
    decoded = adapter.decode_numpy_record(buffer.getvalue())
    assert decoded["scalar"] == .5
    np.testing.assert_array_equal(decoded["xyz"], _record()["xyz"])


@pytest.mark.parametrize("raw", [_npy(np.zeros(3)), _npy(np.array([{}], dtype=object)),
                                 _npy(_record()) + b"unexpected trailing payload"])
def test_numpy_decoder_rejects_wrong_container_or_trailing_data(raw):
    with pytest.raises(ValueError):
        adapter.decode_numpy_record(raw)


@pytest.mark.parametrize("change", ["missing", "nan", "shape", "unequal", "range", "zero_extent"])
def test_requested_pointcloud_invalidity_does_not_silently_drop_modality(tmp_path, change):
    record = _record()
    if change == "missing": del record["xyz"]
    elif change == "nan": record["xyz"][0, 0] = np.nan
    elif change == "shape": record["xyz"] = np.zeros((2, 4))
    elif change == "unequal": record["rgb"] = record["rgb"][:1]
    elif change == "range": record["rgb"][0, 0] = 255
    else: record["xyz"][:] = 0
    result = adapter.prepare_queries(**_fixture(tmp_path, records=[(f"000-000/{A}.npy", _npy(record))]),
                                     include_pointcloud=True)
    assert result["records"] == []
    assert len(result["failures"]) == 1


def test_source_mutation_before_publication_prevents_manifest(tmp_path, monkeypatch):
    kwargs = _fixture(tmp_path)
    original = adapter.decode_numpy_record

    def mutate_metadata(raw):
        kwargs["metadata"].write_text('{"entries": []}')
        return original(raw)

    monkeypatch.setattr(adapter, "decode_numpy_record", mutate_metadata)
    with pytest.raises(ValueError, match="source changed"):
        adapter.prepare_queries(**kwargs)
    assert not (kwargs["out_dir"] / "draft_queries.json").exists()


def test_existing_output_is_never_reused_or_overwritten(tmp_path):
    kwargs = _fixture(tmp_path)
    kwargs["out_dir"].mkdir()
    prior = kwargs["out_dir"] / "draft_queries.json"
    prior.write_bytes(b"existing draft")
    with pytest.raises(FileExistsError, match="must be fresh"):
        adapter.prepare_queries(**kwargs)
    assert prior.read_bytes() == b"existing draft"


@pytest.mark.parametrize("uids", [[], [A, A], ["../unsafe"], [123], {"uids": [A]}])
def test_uid_list_is_validated_before_output_or_shard_reads(tmp_path, uids):
    kwargs = _fixture(tmp_path)
    kwargs["asset_uids"].write_text(json.dumps(uids))
    with pytest.raises(ValueError, match="unique lowercase 32-hex"):
        adapter.prepare_queries(**kwargs)
    assert not kwargs["out_dir"].exists()


def test_successful_cli_selects_name_and_exports_explicit_pointcloud(tmp_path):
    kwargs = _fixture(tmp_path)
    kwargs["caption_field"] = "text"
    argv = []
    for key, value in kwargs.items():
        argv.extend(("--" + key.replace("_", "-"), str(value)))
    assert adapter.main(argv + ["--include-pointcloud"]) == 0
    result = json.loads((kwargs["out_dir"] / "draft_queries.json").read_text())
    assert result["records"][0]["text"] == "Chair"
    assert Path(result["records"][0]["pointcloud"]).is_file()
    assert result["status"] == "needs_relevance_review"


def test_real_cli_flags_and_partial_status(tmp_path, capsys):
    kwargs = _fixture(tmp_path, uids=[A, B])
    argv = []
    for key, value in kwargs.items():
        argv.extend(("--" + key.replace("_", "-"), str(value)))
    assert adapter.main(argv) == 2
    result = json.loads((kwargs["out_dir"] / "draft_queries.json").read_text())
    assert len(result["records"]) == len(result["failures"]) == 1
    assert "relevance review required" in capsys.readouterr().out
