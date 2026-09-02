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


from metafind.models.resolve_stage1 import view_aggregation  # noqa: E402


def protocol(**over) -> dict:
    base = {
        "status": "resolved",
        # [B-2] The identity and the template are both re-derived from the
        # imported serializer, so a fixture that hardcoded either would start
        # failing the moment the serializer moved -- which is the point.
        "text_serialization": text_serialization_id(),
        "text_template": TEXT_TEMPLATE,
        "image_aggregation": "mean",
        # Re-derived, for the same reason as the two lines above: `load_protocol`
        # compares the whole block against what this code would produce, so a
        # hardcoded copy here would start failing the moment the rule moved --
        # which is the point.
        "view_aggregation": view_aggregation(),
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

# [ADDED 2026-08-24] The image half of the cache key. Any fixed string works
# here -- the production value is a digest of the render generation and the 12
# view hashes, and what these tests check is that completion COMPARES it.
IMAGE_ID = "0123456789abcdef"
# [ADDED 2026-08-24, Codex CHANGES REQUIRED] These tests used to write
# `np.savez_compressed(..., text=np.zeros(4))` -- an NPZ with no `image` and no
# `views` -- and assert it was COMPLETE. That is not a fixture shortcut, it is
# the test suite certifying the defect: n06 would skip the asset and n10 would
# fail mid-epoch, or train on whatever it found. A "complete" artifact in these
# tests now looks like a complete artifact on disk.
DIM = 4
N_VIEWS = 12
CKPT_SHA = "0" * 64


def complete_npz(tmp_path, **over):
    arrays = {"text": np.zeros(DIM), "image": np.zeros(DIM),
              "views": np.zeros((N_VIEWS, DIM))}
    arrays.update(over)
    np.savez_compressed(tmp_path / "abc.npz", **arrays)


def sidecar(tmp_path, **over) -> str:
    """A complete sidecar for `abc`, minus whatever the caller overrides."""
    rec = {"uid": "abc",
           "encoder_version": 1,
           # [registry:328] Part of this node's cache key. A v2 sidecar without
           # it was not written by this code, so a fixture without it is not a
           # complete sidecar -- see the tests at the end of this file.
           "ulip2_ckpt_sha": CKPT_SHA,
           "embedding_uri": str(tmp_path / "abc.npz"),
           "image_identity": IMAGE_ID,
           "embedding_dim": DIM,
           "n_views": N_VIEWS,
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
    assert is_complete("abc", text, IMAGE_ID) is False

    complete_npz(tmp_path)
    assert is_complete("abc", text, IMAGE_ID) is True


def test_a_re_render_forces_a_re_encode(monkeypatch, tmp_path):
    """[ADDED 2026-08-24] The images are half of this artifact and were not in
    the cache key at all.

    On 2026-08-24 the corpus was re-rendered with a different denoiser
    (`RENDERER_VERSION` 5 -> 6, different pixels). Completion compared the text
    exactly and the images not at all, so every existing embedding stayed
    "complete" while its view vectors came from images that had been deleted.

    The negative injection is the real one: the sidecar is untouched and only
    the render it claims to describe has moved.
    """
    import metafind.data.encode_text_image as m

    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path)
    complete_npz(tmp_path)
    text = sidecar(tmp_path, encoder_version=m.ENCODER_VERSION)
    assert is_complete("abc", text, IMAGE_ID) is True
    assert is_complete("abc", text, "a different render") is False
    # An n04 record that cannot say what it rendered is a reason to re-encode.
    assert m.image_identity({"renderer_version": 6}) == ""
    assert is_complete("abc", text, "") is False


def test_the_image_identity_moves_with_the_renderer_and_the_pixels():
    """It has to catch BOTH a version bump and a silent re-render."""
    import metafind.data.encode_text_image as m

    base = {"renderer_version": 6, "view_sha256": ["a", "b", "c"]}
    assert m.image_identity(base) == m.image_identity(dict(base))
    assert m.image_identity({**base, "renderer_version": 5}) != m.image_identity(base)
    assert m.image_identity({**base, "view_sha256": ["a", "b", "d"]}) != m.image_identity(base)


def test_a_stale_encoder_version_forces_a_re_encode(monkeypatch, tmp_path):
    import metafind.data.encode_text_image as m

    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path)
    complete_npz(tmp_path)
    text = sidecar(tmp_path, encoder_version=m.ENCODER_VERSION + 1)
    assert is_complete("abc", text, IMAGE_ID) is False


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
    complete_npz(tmp_path)
    stale = ("A wooden dining chair with a slatted back and four tapered legs. "
             "A dining chair made of wood, fabric, "
             "roughly 0.45 by 0.50 by 0.90 metres, typically placed floor.")
    sidecar(tmp_path, encoder_version=m.ENCODER_VERSION, text=stale)

    assert is_complete("abc", serialize_annotation(ANNOTATION), IMAGE_ID) is False


def test_a_sidecar_with_no_text_field_at_all_is_not_complete(monkeypatch, tmp_path):
    """Absence is not agreement. A pre-D10 sidecar that predates the field must
    re-encode rather than be given the benefit of the doubt."""
    import metafind.data.encode_text_image as m

    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path)
    complete_npz(tmp_path)
    text = sidecar(tmp_path, encoder_version=m.ENCODER_VERSION)
    rec = json.loads((tmp_path / "abc.json").read_text())
    del rec["text"]
    (tmp_path / "abc.json").write_text(json.dumps(rec))

    assert is_complete("abc", text, IMAGE_ID) is False


def test_a_record_this_serializer_cannot_serialize_is_never_complete(
        monkeypatch, tmp_path):
    """The 3 prompt_version:1 residuals raise KeyError: 'width'.

    They must land in the work list and be quarantined by the encode loop, not
    be silently declared finished because no expected text could be computed.
    """
    import metafind.data.encode_text_image as m

    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path)
    complete_npz(tmp_path)
    sidecar(tmp_path, encoder_version=m.ENCODER_VERSION)
    assert is_complete("abc", "", IMAGE_ID) is False


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
    complete_npz(tmp_path)
    other = tmp_path / "something_else.txt"
    other.write_text("not vectors")

    text = sidecar(tmp_path, encoder_version=m.ENCODER_VERSION,
                   embedding_uri=str(other))
    assert is_complete("abc", text, IMAGE_ID) is False

    text = sidecar(tmp_path, encoder_version=m.ENCODER_VERSION,
                   embedding_uri=str(tmp_path / "a_different_uid.npz"))
    assert is_complete("abc", text, IMAGE_ID) is False

    text = sidecar(tmp_path, encoder_version=m.ENCODER_VERSION)
    assert is_complete("abc", text, IMAGE_ID) is True


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


# --- 2026-08-24, Codex CHANGES REQUIRED: the gaps it named ------------------

def test_encode_views_composites_alpha_instead_of_dropping_it(monkeypatch, tmp_path):
    """[Codex coverage gap] Nothing reached `Encoder.encode_views()`, so reverting
    the alpha fix would have left the whole selection green.

    Expected truth is the source-over formula, not any model output: a 50%-alpha
    white pixel over black is 128. `Image.open(p).convert("RGB")` returns 255 --
    that is the bug, and the assertion below fails if it comes back.
    """
    from PIL import Image

    import metafind.data.encode_text_image as m

    a = np.zeros((2, 2, 4), dtype=np.uint8)
    a[..., :3] = 255
    a[0, 0, 3] = 128          # half transparent white
    p = tmp_path / "view_00.png"
    Image.fromarray(a, mode="RGBA").save(p)

    seen = []

    class _NoGrad:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Torch:
        @staticmethod
        def stack(xs): return xs
        @staticmethod
        def no_grad(): return _NoGrad()

    class _Backbone:
        @staticmethod
        def encode_image(batch):
            seen.extend(batch)
            class _Out:
                def float(self): return self
                def cpu(self): return self
                def numpy(self): return np.zeros((len(batch), 4))
            return _Out()

    enc = m.Encoder.__new__(m.Encoder)
    enc.torch = _Torch()
    enc.backbone = _Backbone()
    enc.preprocess = lambda im: np.asarray(im)

    enc.encode_views([str(p)])
    assert tuple(seen[0][0, 0]) == (128, 128, 128), (
        f"got {tuple(seen[0][0, 0])}; 255 means alpha was dropped, not composited")


def test_an_npz_missing_its_image_vectors_is_not_complete(monkeypatch, tmp_path):
    """[Codex] `return npz.is_file()` accepted a file with only `text` in it."""
    import metafind.data.encode_text_image as m

    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path)
    text = sidecar(tmp_path, encoder_version=m.ENCODER_VERSION)
    complete_npz(tmp_path)
    assert is_complete("abc", text, IMAGE_ID) is True

    np.savez_compressed(tmp_path / "abc.npz", text=np.zeros(DIM))
    assert is_complete("abc", text, IMAGE_ID) is False, "missing image/views"

    complete_npz(tmp_path, image=np.zeros(DIM + 1))
    assert is_complete("abc", text, IMAGE_ID) is False, "wrong width"

    complete_npz(tmp_path, views=np.zeros((N_VIEWS - 1, DIM)))
    assert is_complete("abc", text, IMAGE_ID) is False, "wrong view count"


def test_changing_the_aggregation_forces_a_re_encode(monkeypatch, tmp_path):
    """[Codex] `image` IS the aggregate. Switching mean -> max scheduled no work
    at all, and Stage 1 then trained on the old pooled vector under the new
    label."""
    import metafind.data.encode_text_image as m

    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path)
    complete_npz(tmp_path)
    text = sidecar(tmp_path, encoder_version=m.ENCODER_VERSION, aggregation="mean")
    assert is_complete("abc", text, IMAGE_ID, "mean") is True
    assert is_complete("abc", text, IMAGE_ID, "max") is False


def test_an_annotation_from_another_render_is_not_encodable(tmp_path):
    """[Codex] n06 joined stale TEXT to new PIXELS and cached the pair forever."""
    import metafind.data.encode_text_image as m

    a = tmp_path / "abc.json"
    a.write_text(json.dumps({**ANNOTATION, "image_identity": "old-render"}))
    assert m._annotation_image_identity(a) == "old-render"

    b = tmp_path / "def.json"
    b.write_text(json.dumps(ANNOTATION))
    assert m._annotation_image_identity(b) == "", "a record with no identity is not current"


def _render_rec(uid: str, tag: str) -> dict:
    """An n04 record whose `image_identity` is stable and distinguishable by `tag`."""
    return {"uid": uid, "renderer_version": 6,
            "view_sha256": [f"{tag}{i:02d}" for i in range(12)]}


def test_one_stale_annotation_halts_even_when_there_is_fresh_work(monkeypatch, tmp_path):
    """[N-1] `partial_failure_semantics: halt` means halt, not "halt if nothing is left".

    The bug this pins was a rank-5 registry violation that could only be seen
    with a MIXED corpus. `return 3 if stale_text else 0` sat under `if not todo`,
    so rc 3 was reachable only when the stale set was EVERY asset. One stale
    annotation among 45,500 fresh ones excluded the stale asset, retired its
    artifacts, encoded the rest, and returned **0** -- the chain read success
    while the corpus had silently shrunk. `chain_to_stage1.sh:71` admits that.

    So the assertion has to be made with `todo` NON-EMPTY, which is exactly the
    case the old guard skipped. `Encoder` is replaced by a tripwire rather than
    mocked out: on the old code `todo` is non-empty and the encoder IS
    constructed, so the tripwire is what distinguishes "halted" from "returned a
    3 it happened to reach anyway".
    """
    import metafind.data.encode_text_image as m
    from metafind.data import view_io

    monkeypatch.setattr(m.paths, "LOGS", tmp_path)
    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path / "emb")
    monkeypatch.setattr(m.paths, "ANNOTATIONS", tmp_path / "ann")
    (tmp_path / "ann").mkdir()
    monkeypatch.setattr("sys.argv", ["encode_text_image"])

    def _tripwire(*a, **k):
        raise AssertionError(
            "n06 built its Encoder with a stale annotation present -- the "
            "registry declares partial_failure_semantics: halt")

    monkeypatch.setattr(m, "Encoder", _tripwire)

    fresh, stale = _render_rec("fresh", "aa"), _render_rec("stale", "bb")
    (tmp_path / "renders_index.jsonl").write_text(
        json.dumps(fresh) + "\n" + json.dumps(stale) + "\n")

    # `fresh` agrees with what is on disk; `stale` names a render that is gone.
    (tmp_path / "ann" / "fresh.json").write_text(json.dumps(
        {**ANNOTATION, "uid": "fresh", "image_identity": view_io.image_identity(fresh)}))
    (tmp_path / "ann" / "stale.json").write_text(json.dumps(
        {**ANNOTATION, "uid": "stale", "image_identity": "an identity from a dead render"}))

    assert m.main() == 3, "one stale annotation among fresh work must halt the node"


def test_the_stale_artifacts_are_retired_before_the_halt(monkeypatch, tmp_path):
    """[N-1] Order matters: retire, THEN halt.

    A halt that returns before retiring would leave the old `<uid>.npz` in place,
    and n09 admits a uid on the FILE, not on this node's exit code -- so the
    stale embedding would survive a run that had already refused it. The halt is
    the reason retirement cannot be deferred to the next pass.
    """
    import metafind.data.encode_text_image as m
    from metafind.data import view_io

    monkeypatch.setattr(m.paths, "LOGS", tmp_path)
    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path / "emb")
    monkeypatch.setattr(m.paths, "ANNOTATIONS", tmp_path / "ann")
    (tmp_path / "ann").mkdir()
    (tmp_path / "emb").mkdir()
    monkeypatch.setattr("sys.argv", ["encode_text_image"])
    monkeypatch.setattr(m, "Encoder", lambda *a, **k: pytest.fail("must not encode"))

    rec = _render_rec("stale", "bb")
    (tmp_path / "renders_index.jsonl").write_text(json.dumps(rec) + "\n")
    (tmp_path / "ann" / "stale.json").write_text(json.dumps(
        {**ANNOTATION, "uid": "stale", "image_identity": "an identity from a dead render"}))
    npz = tmp_path / "emb" / "stale.npz"
    npz.write_bytes(b"the embedding of images that no longer exist")

    assert m.main() == 3
    assert not npz.exists(), "the stale .npz was left where n09 would still admit it"
    assert (tmp_path / "emb" / "stale.npz.stale").exists(), "retired, not deleted"


def test_a_clean_corpus_with_nothing_to_do_still_returns_zero(monkeypatch, tmp_path):
    """[N-1] The halt must not swallow the ordinary "already complete" exit.

    Guards against fixing the silent-0 by making rc 3 unconditional: with no
    stale annotations and nothing left to encode, n06 is complete and must say
    so. Without this the fix would halt every resumed run of a finished corpus.
    """
    import metafind.data.encode_text_image as m
    from metafind.data import view_io

    monkeypatch.setattr(m.paths, "LOGS", tmp_path)
    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path / "emb")
    monkeypatch.setattr(m.paths, "ANNOTATIONS", tmp_path / "ann")
    (tmp_path / "ann").mkdir()
    monkeypatch.setattr("sys.argv", ["encode_text_image"])
    monkeypatch.setattr(m, "Encoder", lambda *a, **k: pytest.fail("nothing to encode"))
    monkeypatch.setattr(m, "is_complete", lambda *a, **k: True)

    rec = _render_rec("fresh", "aa")
    (tmp_path / "renders_index.jsonl").write_text(json.dumps(rec) + "\n")
    (tmp_path / "ann" / "fresh.json").write_text(json.dumps(
        {**ANNOTATION, "uid": "fresh", "image_identity": view_io.image_identity(rec)}))

    assert m.main() == 0


# --- [registry:328] the checkpoint half of the cache key ----------------------

def test_a_sidecar_without_a_checkpoint_sha_is_not_complete(monkeypatch, tmp_path):
    """A weights swap without an ENCODER_VERSION bump used to resume across two
    encoders, and every sidecar written by either passed forever.

    The field is required for its own sake, not only when the caller supplies a
    sha to compare against: a record claiming this encoder_version while lacking
    the field was not written by this code.
    """
    import metafind.data.encode_text_image as m

    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path)
    complete_npz(tmp_path)
    text = sidecar(tmp_path, encoder_version=m.ENCODER_VERSION)
    assert is_complete("abc", text, IMAGE_ID) is True

    rec = json.loads((tmp_path / "abc.json").read_text())
    del rec["ulip2_ckpt_sha"]
    (tmp_path / "abc.json").write_text(json.dumps(rec))
    assert is_complete("abc", text, IMAGE_ID) is False


def test_a_different_checkpoint_forces_a_re_encode(monkeypatch, tmp_path):
    """The failure this exists to stop: new weights, same ENCODER_VERSION, and a
    gallery built in two halves that is self-consistent and wrong."""
    import metafind.data.encode_text_image as m

    monkeypatch.setattr(m.paths, "EMBEDDINGS", tmp_path)
    complete_npz(tmp_path)
    text = sidecar(tmp_path, encoder_version=m.ENCODER_VERSION)

    assert is_complete("abc", text, IMAGE_ID, None, CKPT_SHA) is True
    assert is_complete("abc", text, IMAGE_ID, None, "f" * 64) is False


def test_the_checkpoint_sha_is_the_file_not_a_label(tmp_path, monkeypatch):
    """`ulip2_ckpt_sha` must read the bytes; a name or a path would not change
    when the weights inside it do."""
    import metafind.data.encode_text_image as m

    ckpt = tmp_path / "w.pt"
    ckpt.write_bytes(b"first weights")
    monkeypatch.setattr(m.paths, "ULIP2_CKPT", ckpt)
    m.ulip2_ckpt_sha.cache_clear()
    first = m.ulip2_ckpt_sha()

    ckpt.write_bytes(b"second weights")     # same path, same name
    m.ulip2_ckpt_sha.cache_clear()
    assert m.ulip2_ckpt_sha() != first


# --- retiring an artifact must not destroy the previous retirement -----------

def test_a_second_retirement_does_not_overwrite_the_first(tmp_path):
    """`art.replace(art.suffix + ".stale")` overwrites, on the exact
    re-annotate path the retirement was written for. Keeping evidence and then
    deleting the older evidence is the same as not keeping it."""
    import metafind.data.encode_text_image as m

    art = tmp_path / "abc.json"
    art.write_text("first")
    m._retire(art)
    assert (tmp_path / "abc.json.stale").read_text() == "first"

    art.write_text("second")
    m._retire(art)
    assert (tmp_path / "abc.json.stale").read_text() == "first"
    assert (tmp_path / "abc.json.stale.2").read_text() == "second"


def test_an_edited_view_aggregation_block_is_refused(monkeypatch, tmp_path):
    """[ULIP2 REVIEWER MAJOR 1] Six of the block's seven fields had no consumer.

    `POST_NORMALIZE = True` was a one-line edit that changed every recorded
    recipe, changed every arm hash, changed nothing about `aggregate()` -- an
    unconditional `views.mean(axis=0)` -- and raised nothing. A recorded recipe
    that did not happen.

    The comparison is TOTAL rather than per field, so a field added later
    inherits the check instead of having to be remembered. Each case below is a
    single edited field, which is how the defect would actually arrive.
    """
    from metafind.data.encode_text_image import load_protocol

    for field, wrong in (("post_normalize", True),
                         ("pre_normalize_each_view", True),
                         ("view_selection_policy", "first_eleven"),
                         ("selected_view_ids", list(range(11))),
                         ("aggregation_version", 2),
                         ("n_views", 11)):
        va = dict(view_aggregation())
        va[field] = wrong
        write_protocol(monkeypatch, tmp_path, protocol(view_aggregation=va))
        with pytest.raises(ValueError, match="view_aggregation"):
            load_protocol()

    write_protocol(monkeypatch, tmp_path, protocol(view_aggregation=None))
    with pytest.raises(ValueError, match="view_aggregation"):
        load_protocol()
