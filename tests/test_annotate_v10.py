"""PROMPT_VERSION 10: the paper's Figure 2 object in one call (DL-103)."""
import json

import pytest

from metafind.data import annotate_v10 as v10
from metafind.data.annotate import AnnotationError, FIGURE_2_EXAMPLE, PLACEMENT_FLAGS

FIG2 = json.loads(FIGURE_2_EXAMPLE)


def test_prompt_names_every_figure2_field_and_shows_the_example():
    p = v10.build_prompt(11, "robot")
    for f in v10.FIGURE2_FIELDS:
        assert f in p, f
    assert FIGURE_2_EXAMPLE in p
    assert "11 rendered views" in p
    assert "Objaverse-LVIS as: \"robot\"" in p


def test_figure2_example_is_admitted_and_volume_is_computed():
    out = v10.validate(dict(FIG2), lvis_category="robot")
    assert out["category"] == "robot" and out["synset"] == "robot.n.01"
    assert out["synset_source"] == "model"
    assert (out["width"], out["length"], out["height"]) == (30.0, 30.0, 40.0)
    assert out["volume"] == 36000.0 and out["mass"] == 2.5
    assert out["materials"] == ["metal", "glass", "plastic"]
    assert all(f in out for f in PLACEMENT_FLAGS)
    assert out["category_relation"] == "exact"
    for f in v10.RECORD_REQUIRED_FIELDS:
        assert f in out


@pytest.mark.parametrize("drop", ["category", "width", "mass", "materials", "onFloor", "description"])
def test_missing_required_field_is_refused(drop):
    obj = dict(FIG2); obj.pop(drop)
    with pytest.raises(AnnotationError, match="missing"):
        v10.validate(obj, lvis_category="robot")


def test_unphysical_mass_is_refused_with_a_unit_hint():
    obj = dict(FIG2); obj["mass"] = 90000.0         # 90 tonnes in a 30x30x40 cm box: 2.5 kg/cm^3, above the 1.0 bound
    with pytest.raises(AnnotationError, match="not physical"):
        v10.validate(obj, lvis_category="robot")


def test_string_numbers_and_string_booleans_are_normalised():
    obj = dict(FIG2); obj.update({"width": "30", "onFloor": "true", "materials": "metal, wood"})
    out = v10.validate(obj, lvis_category="robot")
    assert out["width"] == 30.0 and out["onFloor"] is True and out["materials"] == ["metal", "wood"]


def test_malformed_synset_falls_back_to_the_lookup():
    obj = dict(FIG2); obj["synset"] = "not a synset"; obj["category"] = "chair"
    out = v10.validate(obj, lvis_category="chair")
    assert out["synset"].endswith(".n.01") or "." in out["synset"]
    assert out["synset_source"].endswith("_after_malformed_model_synset")


def test_non_english_description_is_refused():
    obj = dict(FIG2); obj["description"] = "一個小機器人"
    with pytest.raises(AnnotationError, match="English"):
        v10.validate(obj, lvis_category="robot")


def test_contract_id_is_versioned_and_stable():
    a, b = v10.contract_id(), v10.contract_id()
    assert a == b and a.startswith("metafind_annot_v10@") and len(a.split("@")[1]) == 16
