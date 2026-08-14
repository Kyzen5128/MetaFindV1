"""L1 tests for ProcTHOR scene-graph extraction (sec. 2.3/2.5).

Runs against the real downloaded houses where available, and against a
hand-built fixture where the assertion needs an exactly known answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from metafind.data.scene_graph import (
    ADJACENCY,
    SUPPORT,
    SceneGraphConfig,
    build_scene_graph,
)

REAL_HOUSES = Path(__file__).resolve().parents[1] / "data/sources/procthor/train.jsonl"


def obj(oid: str, asset: str, xyz, children=None) -> dict:
    return {
        "id": oid,
        "assetId": asset,
        "position": {"x": xyz[0], "y": xyz[1], "z": xyz[2]},
        "rotation": {"x": 0, "y": 0, "z": 0},
        "children": children or [],
    }


def fixture_house() -> dict:
    """A table at the origin with a cup on it, and a chair two metres away."""
    return {
        "objects": [
            obj("Table|0", "Table_1", (0, 0.5, 0), [obj("Cup|surface|0", "Cup_1", (0, 1.0, 0))]),
            obj("Chair|0", "Chair_1", (2.0, 0.5, 0)),
        ],
        "rooms": [{"roomType": "Kitchen", "id": "room|0"}],
    }


def load_real(n: int = 3) -> list[dict]:
    if not REAL_HOUSES.exists():
        pytest.skip("ProcTHOR not downloaded yet")
    out = []
    with open(REAL_HOUSES) as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            out.append(json.loads(line))
    return out


# --------------------------------------------------------------- structure


def test_children_become_support_edges():
    """The paper's own example: a cup on a table must be a support edge."""
    g = build_scene_graph(fixture_house(), "fix")
    assert g.node_ids == ["Table|0", "Cup|surface|0", "Chair|0"]

    support = {
        (g.node_ids[i], g.node_ids[j])
        for (i, j), k in zip(g.edge_index.T.tolist(), g.edge_kind.tolist())
        if k == SUPPORT
    }
    assert ("Table|0", "Cup|surface|0") in support
    assert ("Cup|surface|0", "Table|0") in support, "support must be symmetric for message passing"
    assert not any("Chair|0" in pair for pair in support), "the chair supports nothing"


def test_support_survives_when_geometry_would_also_connect():
    """A cup sits on its table, so the pair is adjacent too; support must win."""
    g = build_scene_graph(fixture_house(), "fix", SceneGraphConfig(adjacency="radius", radius=10.0))
    kinds = {
        (g.node_ids[i], g.node_ids[j]): k
        for (i, j), k in zip(g.edge_index.T.tolist(), g.edge_kind.tolist())
    }
    assert kinds[("Table|0", "Cup|surface|0")] == SUPPORT
    assert kinds[("Table|0", "Chair|0")] == ADJACENCY


def test_disabling_support_removes_exactly_those_edges():
    with_s = build_scene_graph(fixture_house(), "fix", SceneGraphConfig(include_support=True))
    without = build_scene_graph(fixture_house(), "fix", SceneGraphConfig(include_support=False))
    assert with_s.kind_counts()["support"] > 0
    assert without.kind_counts()["support"] == 0


def test_nested_children_are_flattened():
    """Objects nest arbitrarily deep; a shallow walk would drop the deepest ones."""
    house = {
        "objects": [
            obj("A|0", "A", (0, 0, 0), [obj("B|0", "B", (0, 1, 0), [obj("C|0", "C", (0, 2, 0))])])
        ]
    }
    g = build_scene_graph(house, "nest")
    assert g.node_ids == ["A|0", "B|0", "C|0"]
    support = {tuple(e) for e, k in zip(g.edge_index.T.tolist(), g.edge_kind.tolist()) if k == SUPPORT}
    assert (0, 1) in support and (1, 2) in support
    assert (0, 2) not in support, "grandchild must not be linked directly to grandparent"


# --------------------------------------------------------------- adjacency


def test_knn_bounds_the_degree():
    """kNN exists to keep degree independent of room size."""
    house = {"objects": [obj(f"O|{i}", f"A{i}", (i * 0.1, 0, 0)) for i in range(40)]}
    g = build_scene_graph(house, "many", SceneGraphConfig(adjacency="knn", k=4, include_support=False))
    deg = np.bincount(g.edge_index[0], minlength=g.n_nodes)
    # Symmetrisation can raise degree above k, but it must stay bounded.
    assert deg.max() <= 12, f"degree ran to {deg.max()} despite k=4"
    assert deg.min() >= 4


def test_radius_mode_respects_the_threshold():
    house = {
        "objects": [
            obj("A|0", "A", (0, 0, 0)),
            obj("B|0", "B", (1.0, 0, 0)),
            obj("C|0", "C", (50.0, 0, 0)),
        ]
    }
    g = build_scene_graph(house, "r", SceneGraphConfig(adjacency="radius", radius=1.5, include_support=False))
    pairs = {tuple(e) for e in g.edge_index.T.tolist()}
    assert (0, 1) in pairs and (1, 0) in pairs
    assert not any(2 in p for p in pairs), "an object 50 m away must not be adjacent"


def test_no_self_loops_and_no_duplicates():
    for house in load_real(3):
        g = build_scene_graph(house, "real")
        pairs = g.edge_index.T.tolist()
        assert all(i != j for i, j in pairs), "self loop present"
        assert len(pairs) == len({tuple(p) for p in pairs}), "duplicate edge present"


# --------------------------------------------------------------- coordinates


def test_positions_are_left_unnormalised():
    """Normalising would defeat the equivariant encoder it feeds."""
    house = fixture_house()
    shifted = json.loads(json.dumps(house))
    for o in shifted["objects"]:
        o["position"]["x"] += 100.0
        for c in o["children"]:
            c["position"]["x"] += 100.0

    a = build_scene_graph(house, "a")
    b = build_scene_graph(shifted, "b")
    assert np.allclose(b.positions[:, 0] - a.positions[:, 0], 100.0), (
        "positions were re-centred; ESSGNN's equivariance would then be untestable"
    )


def test_edge_structure_is_translation_invariant():
    """Which objects connect must not depend on where the house sits."""
    house = fixture_house()
    shifted = json.loads(json.dumps(house))
    for o in shifted["objects"]:
        o["position"]["x"] += 1000.0
        for c in o["children"]:
            c["position"]["x"] += 1000.0

    a = build_scene_graph(house, "a")
    b = build_scene_graph(shifted, "b")
    assert np.array_equal(a.edge_index, b.edge_index)
    assert np.array_equal(a.edge_kind, b.edge_kind)


# --------------------------------------------------------------- contracts


def test_truncation_is_recorded_never_silent():
    house = {"objects": [obj(f"O|{i}", f"A{i}", (i, 0, 0)) for i in range(300)]}
    g = build_scene_graph(house, "big", SceneGraphConfig(max_nodes=64))
    assert g.n_nodes == 64
    assert g.truncated is True, "truncation must be visible downstream"

    small = build_scene_graph(fixture_house(), "fix", SceneGraphConfig(max_nodes=64))
    assert small.truncated is False


def test_empty_house_is_handled():
    g = build_scene_graph({"objects": []}, "empty")
    assert g.n_nodes == 0 and g.n_edges == 0
    assert g.positions.shape == (0, 3)
    assert g.edge_index.shape == (2, 0)


def test_real_houses_produce_usable_graphs():
    houses = load_real(5)
    for i, h in enumerate(houses):
        g = build_scene_graph(h, f"train_{i}")
        assert g.n_nodes > 0, "a real ProcTHOR house should have objects"
        assert g.n_edges > 0
        assert g.positions.shape == (g.n_nodes, 3)
        assert g.positions.dtype == np.float32
        assert np.isfinite(g.positions).all()
        assert g.edge_index.max() < g.n_nodes, "edge index out of range"
        assert len(g.asset_ids) == g.n_nodes
        counts = g.kind_counts()
        assert counts["support"] > 0, "real houses put objects on surfaces"
        assert counts["adjacency"] > 0


def test_edge_index_is_valid_for_torch_scatter():
    """ESSGNN indexes h[row] and scatters into num_segments=N; both must be in range."""
    g = build_scene_graph(load_real(1)[0], "t")
    assert g.edge_index.dtype == np.int64
    assert g.edge_index.min() >= 0
    assert g.edge_index.max() <= g.n_nodes - 1
    assert g.edge_kind.shape == (g.n_edges,)
