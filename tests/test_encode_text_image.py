"""Tests for n06_encode_text_image's GPU-free half.

The check that matters is the refusal: n06 must not run under
actual_clip_train_scope == trainable. A cache is the output of a network that is
not being updated, and an earlier draft cached all three modalities -- which made
the main line Table 3's "train fuser only" row, the one the paper reports as
worse (8.7 against 11.4).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from metafind.data.encode_text_image import (
    TEXT_CONTEXT_LENGTH,
    aggregate,
    is_complete,
    load_protocol,
)
from metafind.models.stage1_config import (
    PER_VIEW_AGGREGATIONS,
    PRECOMPUTABLE_AGGREGATIONS,
)


def protocol(**over) -> dict:
    base = {
        "status": "resolved",
        "text_serialization": "metafind_v1_natural",
        "image_aggregation": "mean",
        "paper_clip_train_scope": "frozen",
        "actual_clip_train_scope": "frozen",
        "missing_modality_representation": "learned_token",
    }
    base.update(over)
    return base


def write_protocol(monkeypatch, tmp_path, proto):
    import metafind.data.encode_text_image as m

    (tmp_path / "stage1_encoding_protocol.json").write_text(json.dumps(proto))
    monkeypatch.setattr(m.paths, "OUTPUTS", tmp_path)


# --- the refusal ----------------------------------------------------------

def test_a_frozen_protocol_is_accepted(monkeypatch, tmp_path):
    write_protocol(monkeypatch, tmp_path, protocol())
    assert load_protocol()["actual_clip_train_scope"] == "frozen"


def test_encoding_is_refused_under_a_trainable_clip(monkeypatch, tmp_path):
    """[the injection] Under `trainable` gradient must reach OpenCLIP, so there
    is nothing a cache could contribute and everything it could corrupt."""
    write_protocol(monkeypatch, tmp_path,
                   protocol(actual_clip_train_scope="trainable"))
    with pytest.raises(ValueError) as exc:
        load_protocol()
    assert "trainable" in str(exc.value)


def test_an_unresolved_protocol_is_refused(monkeypatch, tmp_path):
    write_protocol(monkeypatch, tmp_path, protocol(status="unresolved"))
    with pytest.raises(ValueError):
        load_protocol()


def test_a_missing_protocol_names_the_node_that_writes_it(monkeypatch, tmp_path):
    """Encoding before the protocol exists is what n05b was split out to stop,
    so the error has to point at n05b rather than at a missing file."""
    import metafind.data.encode_text_image as m

    monkeypatch.setattr(m.paths, "OUTPUTS", tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        load_protocol()
    assert "n05b" in str(exc.value)


def test_a_per_view_aggregation_is_refused(monkeypatch, tmp_path):
    """[U-14] `random_single_view` cannot be honoured by one cached vector, and
    accepting it would produce a cache that silently answers a different rule."""
    assert "random_single_view" in PER_VIEW_AGGREGATIONS
    write_protocol(monkeypatch, tmp_path,
                   protocol(image_aggregation="random_single_view"))
    with pytest.raises(ValueError) as exc:
        load_protocol()
    assert "per-view" in str(exc.value)


# --- aggregation ----------------------------------------------------------

def views() -> np.ndarray:
    return np.array([[1.0, 0.0], [3.0, 4.0], [5.0, 2.0]])


def test_mean_is_the_arithmetic_mean_over_views():
    assert np.allclose(aggregate(views(), "mean"), [3.0, 2.0])


def test_max_and_fixed_view_differ_from_mean():
    """If two rules ever agree on real data, recording which was applied stops
    meaning anything."""
    v = views()
    assert not np.allclose(aggregate(v, "max"), aggregate(v, "mean"))
    assert not np.allclose(aggregate(v, "fixed_view"), aggregate(v, "mean"))


def test_fixed_view_takes_the_first_view_not_a_random_one():
    assert np.allclose(aggregate(views(), "fixed_view"), [1.0, 0.0])


@pytest.mark.parametrize("rule", PRECOMPUTABLE_AGGREGATIONS)
def test_every_precomputable_rule_is_implemented(rule):
    """The protocol validates against this vocabulary, so a rule it accepts and
    this function rejects would fail at asset 1 of 46,052."""
    assert aggregate(views(), rule).shape == (2,)


def test_an_unimplemented_rule_raises_rather_than_defaulting():
    with pytest.raises(ValueError):
        aggregate(views(), "median")


def test_aggregation_preserves_the_embedding_width():
    v = np.random.default_rng(0).normal(size=(11, 1280))
    for rule in PRECOMPUTABLE_AGGREGATIONS:
        assert aggregate(v, rule).shape == (1280,)


# --- completion -----------------------------------------------------------

def test_an_asset_with_a_sidecar_but_no_npz_is_not_complete(monkeypatch, tmp_path):
    """The vectors are the artifact; a record pointing at nothing is not a
    finished asset, and treating it as one loses 11 view embeddings silently."""
    import metafind.data.encode_text_image as m

    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path)
    (tmp_path / "abc.json").write_text(json.dumps({
        "uid": "abc", "encoder_version": m.ENCODER_VERSION,
        "embedding_uri": str(tmp_path / "abc.npz")}))
    assert is_complete("abc") is False

    np.savez_compressed(tmp_path / "abc.npz", text=np.zeros(4))
    assert is_complete("abc") is True


def test_a_stale_encoder_version_forces_a_re_encode(monkeypatch, tmp_path):
    import metafind.data.encode_text_image as m

    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path)
    np.savez_compressed(tmp_path / "abc.npz", text=np.zeros(4))
    (tmp_path / "abc.json").write_text(json.dumps({
        "uid": "abc", "encoder_version": m.ENCODER_VERSION + 1,
        "embedding_uri": str(tmp_path / "abc.npz")}))
    assert is_complete("abc") is False


# --- the token budget -----------------------------------------------------

def test_the_context_length_matches_clips():
    """[L1-TEXT-TOKEN-BUDGET] 77 is CLIP's, not ours to choose."""
    assert TEXT_CONTEXT_LENGTH == 77
