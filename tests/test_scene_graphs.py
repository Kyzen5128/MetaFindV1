"""Tests for n07_scene_graphs.

The two that matter are L1-SCENE-SUPPORT and L1-SCENE-COORDS-RAW; each is
written with its negative injection right beside it, so a test that would pass
on a broken builder is visible rather than merely absent.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from metafind.data.scene_graphs import (
    ADJACENCY_K,
    build_scene_graph,
    humanise,
    object_text,
    _knn_pairs,
)


def house_with_cup_on_table() -> dict:
    """The paper's own example: 'cup on table', as ProcTHOR would store it."""
    return {
        "rooms": [{"id": "room|0", "roomType": "Kitchen"}],
        "objects": [
            {
                "assetId": "DiningTable_3", "id": "DiningTable|0|0",
                "position": {"x": 1.0, "y": 0.5, "z": 2.0},
                "children": [
                    {"assetId": "Cup_1", "id": "Cup|surface|0|0",
                     "position": {"x": 1.1, "y": 0.9, "z": 2.1}},
                ],
            },
            {"assetId": "Fridge_19", "id": "Fridge|0|1",
             "position": {"x": 4.0, "y": 0.9, "z": 0.5}},
            {"assetId": "Chair_7", "id": "Chair|0|2",
             "position": {"x": 1.6, "y": 0.5, "z": 2.0}},
        ],
    }


def index_of(graph: dict, category: str) -> int:
    return next(n["index"] for n in graph["nodes"] if n["category"] == category)


# --- L1-SCENE-SUPPORT ------------------------------------------------------

def test_support_edge_from_children_tree_is_symmetric():
    """[L1-SCENE-SUPPORT] A cup nested under a table becomes a support edge.

    Stored once as an unordered pair; symmetry is the reading convention, and
    [U-19] it is OUR choice -- the paper never says whether either edge type is
    directed.
    """
    g = build_scene_graph(house_with_cup_on_table(), "h0")
    table, cup = index_of(g, "DiningTable"), index_of(g, "Cup")
    pair = [min(table, cup), max(table, cup)]
    assert pair in g["phys_edges"]["support"]


def test_support_edges_vanish_when_the_children_tree_is_ignored():
    """[L1-SCENE-SUPPORT negative injection] Flatten the tree, lose the support.

    This is the injection the check names: if support edges survived a house
    whose nesting has been stripped, they would be coming from somewhere other
    than ProcTHOR's containment relation.
    """
    injected = house_with_cup_on_table()
    cup = injected["objects"][0]["children"][0]
    injected["objects"][0]["children"] = []
    injected["objects"].append(cup)  # same object, no longer nested

    g = build_scene_graph(injected, "h0")
    assert g["phys_edges"]["support"] == []
    # and the object itself is still there, so this is a lost RELATION, not a
    # lost node -- otherwise the assertion above would pass for the wrong reason
    assert index_of(g, "Cup") >= 0


def test_support_survives_two_levels_of_nesting():
    h = house_with_cup_on_table()
    h["objects"][0]["children"][0]["children"] = [
        {"assetId": "Spoon_2", "id": "Spoon|surface|0|0|0",
         "position": {"x": 1.12, "y": 0.95, "z": 2.11}},
    ]
    g = build_scene_graph(h, "h0")
    cup, spoon = index_of(g, "Cup"), index_of(g, "Spoon")
    assert [min(cup, spoon), max(cup, spoon)] in g["phys_edges"]["support"]


# --- L1-SCENE-COORDS-RAW ---------------------------------------------------

def translate(house: dict, offset: float) -> dict:
    moved = copy.deepcopy(house)

    def walk(objs):
        for o in objs:
            for axis in ("x", "y", "z"):
                o["position"][axis] += offset
            walk(o.get("children") or [])

    walk(moved["objects"])
    return moved


def test_translating_a_house_translates_stored_positions_unchanged_structure():
    """[L1-SCENE-COORDS-RAW] Move the house 100 m; the graph moves with it.

    Paper 2.5 states the setting as "large and often unnormalized coordinate
    systems, with no guarantee that scenes are aligned or centered". Centring
    here would not break equivariance -- it would remove the global translation
    the design claims robustness to, so the claim would never be exercised.
    """
    h = house_with_cup_on_table()
    a = build_scene_graph(h, "h0")
    b = build_scene_graph(translate(h, 100.0), "h0")

    assert np.allclose(np.array(b["positions"]), np.array(a["positions"]) + 100.0)
    assert b["phys_edges"] == a["phys_edges"]
    assert b["sem_edge_ids"] == a["sem_edge_ids"]


def test_recentring_would_break_the_offset_assertion():
    """[L1-SCENE-COORDS-RAW negative injection] Centre the positions.

    The injection is applied to the OUTPUT of the production builder, so this
    exercises the same code path the previous test does -- a test that built its
    own centred array would only prove that centring changes numbers.
    """
    h = house_with_cup_on_table()
    a = np.array(build_scene_graph(h, "h0")["positions"])
    b = np.array(build_scene_graph(translate(h, 100.0), "h0")["positions"])

    centred_a = a - a.mean(axis=0)
    centred_b = b - b.mean(axis=0)
    # Centring makes the two indistinguishable, so the +100 assertion cannot
    # fail and cannot pass -- it has nothing left to measure.
    assert np.allclose(centred_a, centred_b)
    assert not np.allclose(centred_b, centred_a + 100.0)


# --- adjacency [U-05] ------------------------------------------------------

def test_knn_pairs_are_unordered_and_exclude_self():
    pts = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    pairs = _knn_pairs(pts, k=2)
    assert all(i < j for i, j in pairs)
    assert all(i != j for i, j in pairs)
    assert (0, 1) in pairs and (2, 3) in pairs


def test_knn_k_is_clamped_below_the_node_count():
    pts = np.array([[0.0, 0, 0], [1, 0, 0]])
    assert _knn_pairs(pts, k=ADJACENCY_K) == [(0, 1)]


def test_a_single_object_house_has_no_edges():
    h = {"rooms": [{"id": "room|0", "roomType": "Bedroom"}],
         "objects": [{"assetId": "Bed_1", "id": "Bed|0|0",
                      "position": {"x": 0.0, "y": 0.0, "z": 0.0}}]}
    g = build_scene_graph(h, "h0")
    assert g["phys_edges"]["support"] == []
    assert g["phys_edges"]["adjacency"] == []
    assert g["sem_edge_ids"] == []


def test_support_and_adjacency_are_disjoint():
    """A pair is one kind of physical edge or the other, never counted twice."""
    g = build_scene_graph(house_with_cup_on_table(), "h0")
    sup = {tuple(p) for p in g["phys_edges"]["support"]}
    adj = {tuple(p) for p in g["phys_edges"]["adjacency"]}
    assert sup & adj == set()
    assert {tuple(p) for p in g["sem_edge_ids"]} == sup | adj


# --- room assignment -------------------------------------------------------

def test_children_inherit_their_parents_room():
    g = build_scene_graph(house_with_cup_on_table(), "h0")
    table = next(n for n in g["nodes"] if n["category"] == "DiningTable")
    cup = next(n for n in g["nodes"] if n["category"] == "Cup")
    # "Cup|surface|0|0" parses to room|surface, which is not a room; inheritance
    # is what saves it.
    assert cup["room_id"] == table["room_id"] == "room|0"


def test_an_unparseable_room_index_yields_none_not_a_wrong_room():
    h = house_with_cup_on_table()
    h["objects"][1]["id"] = "Fridge|99|1"  # room|99 does not exist
    g = build_scene_graph(h, "h0")
    assert next(n for n in g["nodes"] if n["category"] == "Fridge")["room_id"] is None


# --- t_i text [U-12] -------------------------------------------------------

def test_humanise_splits_camel_case():
    assert humanise("CounterTop") == "counter top"
    assert humanise("HousePlant") == "house plant"
    assert humanise("Fridge") == "fridge"


def test_object_text_is_instance_independent():
    """[U-12] Keyed by assetId, so it must not carry per-instance context."""
    assert object_text("DiningTable") == "a dining table"
    assert "kitchen" not in object_text("DiningTable").lower()


# --- failure handling ------------------------------------------------------

@pytest.mark.parametrize("missing", ["objects", "rooms"])
def test_a_house_missing_a_required_field_raises(missing):
    h = house_with_cup_on_table()
    del h[missing]
    with pytest.raises(KeyError):
        build_scene_graph(h, "h0")


def test_a_house_with_no_objects_raises():
    with pytest.raises(KeyError):
        build_scene_graph({"rooms": [], "objects": []}, "h0")
