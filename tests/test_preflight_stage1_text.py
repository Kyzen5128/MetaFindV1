"""The gate that runs before n06 spends GPU time had two bugs and no tests.

Both survived because nothing imported this tool. Each test below states the
bug it pins and then verifies it can actually fail, by feeding the OLD rule the
same input: a test for a guard is worth nothing until you have seen it go red.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import pytest

from metafind.data.encode_text_image import is_complete
from metafind.models.resolve_stage1 import serialize_annotation

TOOL = Path(__file__).resolve().parents[1] / "tools" / "preflight_stage1_text.py"


def _tool():
    spec = importlib.util.spec_from_file_location("preflight_stage1_text", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The exact shape that broke it: Gemma writes "roughly" inside prose, so the
# serialized string carries ", roughly " twice. Both of these are real
# descriptions from the corpus, trimmed.
PROSE_WITH_ROUGHLY = [
    "This milestone is a weathered, roughly hewn stone pillar with a gray surface",
    "This 3D model is an axe with a dark grey, roughly forged metal head",
]


def _annotation(description: str) -> dict:
    return {
        "description": description,
        "category": "milestone",
        "materials": ["wood", "paint"],
        "width": 69.90990825760335,
        "length": 69.07735228303062,
        "height": 100.0,
        "onCeiling": False, "onWall": False, "onFloor": True, "onObject": False,
    }


@pytest.mark.parametrize("description", PROSE_WITH_ROUGHLY)
def test_dimension_parse_survives_roughly_in_the_description(description):
    text = serialize_annotation(_annotation(description))
    assert text.count(", roughly ") == 2, "the fixture must reproduce the collision"
    assert _tool().rendered_dimensions(text) == ["69.9", "69.1", "100"]


@pytest.mark.parametrize("description", PROSE_WITH_ROUGHLY)
def test_the_old_first_occurrence_rule_would_have_failed(description):
    """The false-negative check. Without this, the test above passes on any
    parser, including one that never looks at ", roughly " at all."""
    text = serialize_annotation(_annotation(description))
    old = text.split(", roughly ", 1)[1].split(" centimetres,", 1)[0].split(" by ")
    assert old != ["69.9", "69.1", "100"]
    with pytest.raises(ValueError):
        float(old[0])          # this is what put 21 uids in `zero_dim`


def test_the_gate_calls_is_complete_the_way_n06_does():
    """n06 decides the run; a gate that asks an easier question reports a
    cache-valid count n06 will not honour. Pins ARITY, which is what actually
    broke -- the tool passed two arguments to a five-parameter function and
    died with TypeError before printing anything."""
    required = [
        p.name for p in inspect.signature(is_complete).parameters.values()
        if p.default is inspect.Parameter.empty
    ]
    calls = [
        node for node in ast.walk(ast.parse(TOOL.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "is_complete"
    ]
    assert len(calls) == 1, f"expected one is_complete call, found {len(calls)}"
    supplied = len(calls[0].args) + len(calls[0].keywords)
    assert supplied >= len(required), (
        f"the gate passes {supplied} argument(s); is_complete requires "
        f"{len(required)}: {required}")
    # And the optional ones too -- passing only the required three would still
    # skip the aggregation and checkpoint halves of the cache key, which is the
    # drift `ulip2_ckpt_sha` was added to catch.
    assert supplied == len(inspect.signature(is_complete).parameters)
