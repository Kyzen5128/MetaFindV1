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
    expected_text_for,
    is_complete,
    load_protocol,
)
from metafind.models.resolve_stage1 import (
    TEXT_TEMPLATE,
    serialize_annotation,
    text_serialization_id,
)
from metafind.models.stage1_config import (
    PER_VIEW_AGGREGATIONS,
    PRECOMPUTABLE_AGGREGATIONS,
)

ANNOTATION = {
    "category": "dining chair",
    "synset": "chair.n.01",
    "description": "A wooden dining chair with a slatted back and four tapered legs",
    "width": 50.0, "length": 45.0, "height": 90.0,
    "mass": 6.0,
    "materials": ["wood", "fabric"],
    "onCeiling": False, "onWall": False, "onFloor": True, "onObject": False,
}


def protocol(**over) -> dict:
    base = {
        "status": "resolved",
        # [B-2] The identity and the template are both re-derived from the
        # imported serializer, so a fixture that hardcoded either would start
        # failing the moment the serializer moved -- which is the point.
        "text_serialization": text_serialization_id(),
        "text_template": TEXT_TEMPLATE,
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

def sidecar(tmp_path, **over) -> str:
    """A complete sidecar for `abc`, minus whatever the caller overrides."""
    rec = {"uid": "abc",
           "encoder_version": 1,
           "embedding_uri": str(tmp_path / "abc.npz"),
           "text": serialize_annotation(ANNOTATION)}
    rec.update(over)
    (tmp_path / "abc.json").write_text(json.dumps(rec, ensure_ascii=False))
    return rec["text"]


def test_an_asset_with_a_sidecar_but_no_npz_is_not_complete(monkeypatch, tmp_path):
    """The vectors are the artifact; a record pointing at nothing is not a
    finished asset, and treating it as one loses 11 view embeddings silently."""
    import metafind.data.encode_text_image as m

    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path)
    text = sidecar(tmp_path, encoder_version=m.ENCODER_VERSION)
    assert is_complete("abc", text) is False

    np.savez_compressed(tmp_path / "abc.npz", text=np.zeros(4))
    assert is_complete("abc", text) is True


def test_a_stale_encoder_version_forces_a_re_encode(monkeypatch, tmp_path):
    import metafind.data.encode_text_image as m

    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path)
    np.savez_compressed(tmp_path / "abc.npz", text=np.zeros(4))
    text = sidecar(tmp_path, encoder_version=m.ENCODER_VERSION + 1)
    assert is_complete("abc", text) is False


# --- the token budget -----------------------------------------------------

def test_the_context_length_matches_clips():
    """[L1-TEXT-TOKEN-BUDGET] 77 is CLIP's, not ours to choose."""
    assert TEXT_CONTEXT_LENGTH == 77


# --- B-1: cache invalidation ----------------------------------------------

def test_a_sidecar_whose_text_came_from_another_serializer_is_not_complete(
        monkeypatch, tmp_path):
    """[B-1, D0-008 §11.2] THE blocker this contract exists to close.

    5,276 embeddings on disk were built from the metre-based template. The old
    is_complete() compared sidecar existence, encoder_version and NPZ existence
    and NOTHING about the text, so a resumed run skipped every one of them as
    "complete" and encoded the rest in centimetres: a gallery from two text
    distributions, no error, no warning, and the same text_serialization label
    on both halves.
    """
    import metafind.data.encode_text_image as m

    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path)
    np.savez_compressed(tmp_path / "abc.npz", text=np.zeros(4))
    stale = ("A wooden dining chair with a slatted back and four tapered legs. "
             "A dining chair made of wood, fabric, "
             "roughly 0.45 by 0.50 by 0.90 metres, typically placed floor.")
    sidecar(tmp_path, encoder_version=m.ENCODER_VERSION, text=stale)

    assert is_complete("abc", serialize_annotation(ANNOTATION)) is False


def test_a_sidecar_with_no_text_field_at_all_is_not_complete(monkeypatch, tmp_path):
    """Absence is not agreement. A pre-D10 sidecar that predates the field must
    re-encode rather than be given the benefit of the doubt."""
    import metafind.data.encode_text_image as m

    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path)
    np.savez_compressed(tmp_path / "abc.npz", text=np.zeros(4))
    text = sidecar(tmp_path, encoder_version=m.ENCODER_VERSION)
    rec = json.loads((tmp_path / "abc.json").read_text())
    del rec["text"]
    (tmp_path / "abc.json").write_text(json.dumps(rec))

    assert is_complete("abc", text) is False


def test_a_record_this_serializer_cannot_serialize_is_never_complete(
        monkeypatch, tmp_path):
    """The 3 prompt_version:1 residuals raise KeyError: 'width'.

    They must land in the work list and be quarantined by the encode loop, not
    be silently declared finished because no expected text could be computed.
    """
    import metafind.data.encode_text_image as m

    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path)
    np.savez_compressed(tmp_path / "abc.npz", text=np.zeros(4))
    sidecar(tmp_path, encoder_version=m.ENCODER_VERSION)
    assert is_complete("abc", "") is False


def test_expected_text_for_returns_empty_on_an_unserialisable_record(tmp_path):
    v1 = tmp_path / "abc.json"
    v1.write_text(json.dumps({"category": "chair", "description": "a chair",
                              "dimensions": {"w": 1}, "materials": ["wood"],
                              "prompt_version": 1}))
    assert expected_text_for(v1) == ""


def test_expected_text_for_matches_the_serializer(tmp_path):
    good = tmp_path / "abc.json"
    good.write_text(json.dumps(ANNOTATION))
    assert expected_text_for(good) == serialize_annotation(ANNOTATION)


# --- B-2: the protocol must describe the serializer -------------------------

def test_a_protocol_from_another_serializer_is_refused(monkeypatch, tmp_path):
    """[B-2] Before this check, load_protocol() validated status, CLIP scope and
    aggregation only. The artifact on disk recorded the metre template while the
    code emitted centimetres, and line 233 stamped the artifact's label on the
    code's output: encode with serializer X, label it Y, no error anywhere."""
    write_protocol(monkeypatch, tmp_path,
                   protocol(text_serialization="metafind_v1_natural"))
    with pytest.raises(ValueError) as exc:
        load_protocol()
    assert "text_serialization" in str(exc.value)


def test_a_protocol_recording_the_old_metre_template_is_refused(
        monkeypatch, tmp_path):
    """The identity matches but the recorded template does not: someone edited
    the artifact by hand. A record that can drift from its own code is a record
    that certifies nothing."""
    write_protocol(monkeypatch, tmp_path, protocol(
        text_template=("{description} A {category} made of {materials}, roughly "
                       "{length:.2f} by {width:.2f} by {height:.2f} metres, "
                       "typically placed {placement}.")))
    with pytest.raises(ValueError) as exc:
        load_protocol()
    assert "text_template" in str(exc.value)


def test_a_protocol_with_no_serialization_identity_at_all_is_refused(
        monkeypatch, tmp_path):
    proto = protocol()
    del proto["text_serialization"]
    write_protocol(monkeypatch, tmp_path, proto)
    with pytest.raises(ValueError):
        load_protocol()


# --- B-3: the retired identifier -------------------------------------------

def test_the_accepted_identity_is_content_addressed(monkeypatch, tmp_path):
    """[B-3] "metafind_v1_natural" is retired because it named two different
    transformations. What replaces it is derived from the emitted string."""
    write_protocol(monkeypatch, tmp_path, protocol())
    accepted = load_protocol()["text_serialization"]
    assert accepted == text_serialization_id()
    assert accepted != "metafind_v1_natural"
    assert "@" in accepted


def test_n06_never_overrides_the_serializer_template():
    """[B-2] The identity binds this module's serializer, not an argument.

    `serialize_annotation(annotation, template=...)` would slip past every check
    in load_protocol() while still stamping the protocol's label on the result,
    so n06's call sites are pinned to the no-override form here.
    """
    import inspect
    import re

    import metafind.data.encode_text_image as m

    calls = re.findall(r"serialize_annotation\((.*?)\)\)?", inspect.getsource(m))
    assert calls, "n06 no longer calls serialize_annotation at all"
    for call in calls:
        assert "template" not in call, call
        assert "," not in call, call


def test_a_sidecar_pointing_at_some_other_file_is_not_complete(monkeypatch, tmp_path):
    """[B-1] Adversarial review round 2 pointed embedding_uri at a source file in
    the repository and is_complete() returned true.

    "A file exists there" is not the invariant. "THIS asset's vectors are there"
    is, so the URI must resolve to the canonical <uid>.npz.
    """
    import metafind.data.encode_text_image as m

    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path)
    np.savez_compressed(tmp_path / "abc.npz", text=np.zeros(4))
    other = tmp_path / "something_else.txt"
    other.write_text("not vectors")

    text = sidecar(tmp_path, encoder_version=m.ENCODER_VERSION,
                   embedding_uri=str(other))
    assert is_complete("abc", text) is False

    text = sidecar(tmp_path, encoder_version=m.ENCODER_VERSION,
                   embedding_uri=str(tmp_path / "a_different_uid.npz"))
    assert is_complete("abc", text) is False

    text = sidecar(tmp_path, encoder_version=m.ENCODER_VERSION)
    assert is_complete("abc", text) is True


# --- P-4: the 77-token budget is enforced, not merely recorded ---------------

def test_the_token_count_is_untruncated():
    """[P-4] `Encoder.token_count` counts non-zero slots in a tokenizer that pads
    to exactly 77, so it CANNOT return a number above 77 -- the corpus's 89-token
    record reported 77 and looked like a boundary case rather than a 12-token
    loss. This is Codex finding C-3 from D0-008, confirmed."""
    from metafind.data.encode_text_image import true_token_count

    assert true_token_count("wooden " * 100) > TEXT_CONTEXT_LENGTH
    assert true_token_count("a chair") == 4       # SOT + 2 BPE + EOT


def test_an_overlong_text_is_refused_rather_than_encoded():
    """[P-4] The previous code set `text_truncated=True` and encoded the asset
    anyway, so a knowingly-degraded embedding entered the gallery behind a flag
    nothing downstream reads."""
    from metafind.data.encode_text_image import refuse_if_overlong

    with pytest.raises(ValueError) as exc:
        refuse_if_overlong("wooden " * 100)
    assert "true BPE tokens" in str(exc.value)
    assert "placement clause" in str(exc.value)


def test_a_text_inside_the_budget_passes_and_returns_its_count():
    from metafind.data.encode_text_image import refuse_if_overlong

    assert refuse_if_overlong(serialize_annotation(ANNOTATION)) < TEXT_CONTEXT_LENGTH


def test_exactly_the_context_length_is_allowed():
    """77 is the context, not the first illegal value: SOT + 75 + EOT fits."""
    from metafind.data.encode_text_image import refuse_if_overlong, true_token_count

    text = "wooden " * 100
    while true_token_count(text) > TEXT_CONTEXT_LENGTH:
        text = text[:-7]
    assert refuse_if_overlong(text) <= TEXT_CONTEXT_LENGTH
