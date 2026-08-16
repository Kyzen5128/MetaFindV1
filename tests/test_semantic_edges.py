"""Tests for n08_semantic_edges' deterministic half.

The three that matter are L1-SEMEDGE-KEY, L1-SEMEDGE-POS-INDEPENDENT and
L1-SEMEDGE-NO-ZEROFILL. Each is written with its negative injection beside it.
No GPU is touched here, which is the reason the module is split at all.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from metafind.data.scene_graphs import build_scene_graph
from metafind.data.semantic_edges import (
    MAX_SENTENCE_CHARS,
    PROMPT_VERSION,
    SemanticEdgeError,
    build_relation_prompt,
    build_repair_prompt,
    cache_key,
    iter_pair_descriptions,
    parse_sentence,
    validate_sentence,
)

LLM = "Qwen/Qwen2.5-7B-Instruct"
ENC = "clip-vit-b32-laion2b-s34b-b79k"


def key(a: str, b: str, **over) -> str:
    kw = {"prompt_version": PROMPT_VERSION, "llm_model": LLM,
          "text_encoder_version": ENC}
    kw.update(over)
    return cache_key(a, b, **kw)


# --- L1-SEMEDGE-KEY --------------------------------------------------------

def test_different_descriptions_of_the_same_categories_do_not_collide():
    """[L1-SEMEDGE-KEY] The exact collision the check names.

    "office chair + desk" and "dining chair + dining table" are both (chair,
    table) at category granularity. A key that hashed categories would serve one
    relation for both.
    """
    a = key("an office chair", "a desk")
    b = key("a dining chair", "a dining table")
    assert a != b


def test_keying_on_categories_would_collide():
    """[L1-SEMEDGE-KEY negative injection] Reduce both pairs to categories."""
    def category_key(d1: str, d2: str) -> str:
        # the injection: drop the modifier, keep the head noun
        return key(d1.split()[-1], d2.split()[-1])

    assert category_key("an office chair", "a desk") != category_key(
        "a dining chair", "a dining table")
    # ... but once "desk" and "table" are the same head noun, which is what a
    # real category vocabulary does, the injected key collides and the honest
    # one does not:
    assert category_key("an office chair", "a dining table") == category_key(
        "a dining chair", "a dining table")
    assert key("an office chair", "a dining table") != key(
        "a dining chair", "a dining table")


def test_the_key_is_order_independent():
    assert key("a lamp", "a nightstand") == key("a nightstand", "a lamp")


@pytest.mark.parametrize("field,value", [
    ("prompt_version", 2),
    ("llm_model", "some/other-model"),
    ("text_encoder_version", "bert-base-uncased"),
])
def test_changing_any_version_invalidates_the_key(field, value):
    """A cached sentence from a different pipeline must not be served."""
    assert key("a lamp", "a nightstand") != key("a lamp", "a nightstand",
                                                **{field: value})


def test_the_key_is_stable_across_calls():
    assert key("a lamp", "a nightstand") == key("a lamp", "a nightstand")


# --- L1-SEMEDGE-POS-INDEPENDENT -------------------------------------------

def two_object_house(offset: float = 0.0) -> dict:
    return {
        "rooms": [{"id": "room|0", "roomType": "Bedroom"}],
        "objects": [
            {"assetId": "Bed_1", "id": "Bed|0|0",
             "position": {"x": 1.0 + offset, "y": 0.3 + offset, "z": 2.0 + offset}},
            {"assetId": "Lamp_2", "id": "Lamp|0|1",
             "position": {"x": 2.0 + offset, "y": 0.8 + offset, "z": 2.0 + offset}},
        ],
    }


TEXT_MAP = {"Bed_1": {"text": "a bed", "source": "procthor_category"},
            "Lamp_2": {"text": "a lamp", "source": "procthor_category"}}


def test_a_rigidly_transformed_house_produces_the_same_cache_keys():
    """[L1-SEMEDGE-POS-INDEPENDENT] Move the house 100 m; the keys do not move.

    Appendix C's proof assumes e_ij is independent of x. This is that assumption
    stated as a test.
    """
    def keys_of(house):
        g = build_scene_graph(house, "h0")
        return sorted(key(a, b) for _, a, b in iter_pair_descriptions(g, TEXT_MAP))

    assert keys_of(two_object_house()) == keys_of(two_object_house(100.0))


def test_folding_positions_into_the_description_breaks_key_stability():
    """[L1-SEMEDGE-POS-INDEPENDENT negative injection] Put coordinates in.

    The injection is applied where a coordinate would actually get added -- to
    the description that feeds the key -- rather than to the key function, so it
    exercises the path a well-meaning "richer descriptions" edit would take.
    """
    def injected_keys(house):
        g = build_scene_graph(house, "h0")
        out = []
        for (i, j), a, b in iter_pair_descriptions(g, TEXT_MAP):
            pos_i, pos_j = g["positions"][i], g["positions"][j]
            out.append(key(f"{a} at {pos_i}", f"{b} at {pos_j}"))
        return sorted(out)

    assert injected_keys(two_object_house()) != injected_keys(two_object_house(100.0))


def test_the_relation_prompt_carries_no_spatial_term():
    prompt = build_relation_prompt("a bed", "a lamp").lower()
    for banned in ("coordinate", "x =", "position of", "metres away",
                   "rotation", "world-frame"):
        assert banned not in prompt


def test_the_relation_prompt_contains_both_descriptions_and_nothing_numeric():
    prompt = build_relation_prompt("a microscope", "a lab bench")
    assert "a microscope" in prompt and "a lab bench" in prompt
    assert not any(ch.isdigit() for ch in prompt)


def test_a_prompt_that_gained_a_coordinate_trips_its_own_assertion(monkeypatch):
    """The guard inside build_relation_prompt fires on ITS OWN output."""
    import metafind.data.semantic_edges as se

    original = se.build_relation_prompt

    def leaky(desc_i, desc_j):
        return original(desc_i, desc_j) + "\nThe position of A is (1, 2, 3)."

    # the injected text is what the real assertion inspects
    assert se._COORDINATE_HINT.search(leaky("a bed", "a lamp"))
    assert not se._COORDINATE_HINT.search(original("a bed", "a lamp"))


# --- sentence handling -----------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("A pillow rests on a bed.", "A pillow rests on a bed."),
    ('  "A pillow rests on a bed."  ', "A pillow rests on a bed."),
    ("Sure, here it is: A pillow rests on a bed.", "A pillow rests on a bed."),
    ("Here is the sentence: A rug lies under a table.",
     "A rug lies under a table."),
    ("```\nA pillow rests on a bed.\n```", "A pillow rests on a bed."),
    ("A pillow rests on a bed.\nThey are both bedroom items.",
     "A pillow rests on a bed."),
])
def test_parse_sentence_strips_the_wrapping(raw, expected):
    assert parse_sentence(raw) == expected


@pytest.mark.parametrize("raw", [
    "A lamp: it stands on a nightstand.",
    "A desk lamp: used for reading at a desk.",
])
def test_a_legitimate_colon_keeps_its_subject(raw):
    """Preamble stripping must not eat the object the relation is about.

    The first version matched any short prefix ending in a colon, so
    "A lamp: it stands on a nightstand." became "it stands on a nightstand."
    and the lamp vanished from the embedding. Carrying a constant preamble is
    the lesser harm -- it shifts every edge alike; losing content does not.
    """
    assert parse_sentence(raw) == raw


@pytest.mark.parametrize("bad", ["", "   ", "\n", "123 456", "!!! ???"])
def test_validate_rejects_empty_or_wordless(bad):
    with pytest.raises(SemanticEdgeError):
        validate_sentence(bad)


def test_validate_rejects_an_essay():
    with pytest.raises(SemanticEdgeError) as exc:
        validate_sentence("x " * MAX_SENTENCE_CHARS)
    assert str(MAX_SENTENCE_CHARS) in str(exc.value)


def test_validate_returns_the_stripped_sentence():
    assert validate_sentence("  A lamp stands on a nightstand.  ") == \
        "A lamp stands on a nightstand."


def test_a_repair_prompt_must_name_the_failure():
    original = build_relation_prompt("a bed", "a lamp")
    repair = build_repair_prompt(original, "the sentence contains no words", "!!!")
    assert repair != original
    assert "the sentence contains no words" in repair


def test_a_repair_prompt_with_no_error_is_refused():
    with pytest.raises(ValueError):
        build_repair_prompt(build_relation_prompt("a bed", "a lamp"), "", "!!!")


# --- L1-SEMEDGE-NO-ZEROFILL ------------------------------------------------

def build_cache(settled: dict) -> dict:
    """The production assembler. Importing it here is the whole point: a second
    copy written for the test would pass whatever the shipped code did."""
    from metafind.data.semantic_edges_run import build_cache as production

    return production(settled, "emb")


def test_an_exhausted_edge_is_flagged_and_carries_no_embedding():
    """[L1-SEMEDGE-NO-ZEROFILL] The entry exists, says missing, holds no vector."""
    cache = build_cache({
        "k1": {"sentence": "A lamp stands on a nightstand.", "degraded": False},
        "k2": {"sentence": None, "degraded": True, "reason": "no words"},
    })
    assert cache["k2"]["semantic_edge_missing"] is True
    assert cache["k2"]["embedding_uri"] is None
    assert cache["k1"].get("semantic_edge_missing") is None


def test_zero_filling_would_lose_the_distinction():
    """[L1-SEMEDGE-NO-ZEROFILL negative injection] Fill zeros instead of flagging.

    The point is not that zeros are an unusual value -- it is that downstream
    there is no way left to ask which edges were real.
    """
    dim = 512
    settled = {"k1": {"sentence": "A lamp stands on a nightstand.", "degraded": False},
               "k2": {"sentence": None, "degraded": True, "reason": "no words"}}

    injected = {k: {"embedding": np.zeros(dim) if r["degraded"] else np.ones(dim),
                    "sentence": r["sentence"]} for k, r in settled.items()}
    assert all("semantic_edge_missing" not in v for v in injected.values())
    # a real relation whose embedding happens to be near zero is now
    # indistinguishable from an exhausted one
    injected["k3"] = {"embedding": np.zeros(dim), "sentence": "A rug lies under a table."}
    zeros = [k for k, v in injected.items() if not v["embedding"].any()]
    assert set(zeros) == {"k2", "k3"}  # the flag would have separated them

    honest = build_cache(settled)
    assert [k for k, v in honest.items() if v["degraded"]] == ["k2"]


# --- pair enumeration ------------------------------------------------------

def test_iter_pair_descriptions_covers_every_selected_pair():
    g = build_scene_graph(two_object_house(), "h0")
    pairs = list(iter_pair_descriptions(g, TEXT_MAP))
    assert len(pairs) == len(g["sem_edge_ids"])
    assert {p[0] for p in pairs} == {tuple(e) for e in g["sem_edge_ids"]}


def test_a_missing_object_text_is_raised_not_skipped():
    g = build_scene_graph(two_object_house(), "h0")
    with pytest.raises(KeyError):
        list(iter_pair_descriptions(g, {"Bed_1": TEXT_MAP["Bed_1"]}))
