"""Turn ProcTHOR house JSON into the scene graphs ESSGNN consumes (sec. 2.3, 2.5).

Paper sec. 2.3 describes two edge types::

    1. Physical-relation edges that capture spatial dependencies (e.g. "cup on table")
    2. Semantic-relation edges that capture functional or contextual associations
       (e.g. "microscope-lab bench"), obtained by prompting an LLM on object pairs

and sec. 2.5 adds that spatial edges come from "physical layout constraints
(e.g., adjacency, support)".

Support comes free
------------------

ProcTHOR already encodes support in its object tree: ``Apple_24`` appears as a
child of ``Countertop_I_8x2``. That is literally the paper's "cup on table", so
support edges are read off the tree rather than inferred from geometry, which
would be both slower and less reliable.

Adjacency does not
------------------

The paper names adjacency but gives no criterion -- no radius, no neighbour
count (U-16). Rooms here hold 69 objects on average and up to 245, so a fully
connected graph would reach 60k edges and the choice materially changes both
cost and what the model can express. kNN is the default because it bounds degree
regardless of room size; radius is available and the parameter is recorded with
the artifact rather than hidden in code.

Coordinates stay raw
--------------------

Positions are world coordinates and are deliberately **not** normalised or
centred. Normalising would defeat the entire purpose of an SE(3)-equivariant
encoder, whose stated motivation (sec. 2.5) is "large and often unnormalized
coordinate systems, with no guarantee that scenes are aligned or centered".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

__all__ = ["SceneGraphConfig", "SceneGraph", "build_scene_graph", "SUPPORT", "ADJACENCY"]

SUPPORT = 0
ADJACENCY = 1

AdjacencyMode = Literal["knn", "radius"]


@dataclass
class SceneGraphConfig:
    """How physical edges are drawn.

    Attributes:
        adjacency: ``"knn"`` bounds degree, ``"radius"`` is metric. The paper
            specifies neither (U-16).
        k: neighbours per node under ``"knn"``.
        radius: metres under ``"radius"``.
        include_support: draw parent/child support edges from the object tree.
        symmetric: add both directions for every edge. ESSGNN aggregates over
            ``N(i)``, so a one-way support edge would let the table inform the
            cup but not the reverse.
        max_nodes: cap on objects per scene. Houses reach 245 objects; a cap
            keeps memory predictable. Truncation is recorded, never silent.
    """

    adjacency: AdjacencyMode = "knn"
    k: int = 8
    radius: float = 1.5
    include_support: bool = True
    symmetric: bool = True
    max_nodes: int = 256


@dataclass
class SceneGraph:
    """A single room-level scene graph."""

    house_id: str
    node_ids: list[str]
    asset_ids: list[str]
    types: list[str]
    room_types: list[str]
    positions: np.ndarray  # (N, 3) float32, world frame, unnormalised
    edge_index: np.ndarray  # (2, E) int64
    edge_kind: np.ndarray  # (E,) int64 -- SUPPORT or ADJACENCY
    truncated: bool = False
    config: SceneGraphConfig = field(default_factory=SceneGraphConfig)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def kind_counts(self) -> dict[str, int]:
        return {
            "support": int((self.edge_kind == SUPPORT).sum()),
            "adjacency": int((self.edge_kind == ADJACENCY).sum()),
        }


def _flatten_objects(house: dict) -> tuple[list[dict], list[tuple[int, int]]]:
    """Walk the object tree, returning records plus (parent, child) index pairs.

    Children are objects resting on or inside their parent, so the tree edges are
    the support relations. Dropping children would discard the very relation the
    paper cites as its example.
    """
    records: list[dict] = []
    support: list[tuple[int, int]] = []

    def visit(obj: dict, parent: int | None, room: str) -> None:
        idx = len(records)
        records.append({**obj, "_room": room})
        if parent is not None:
            support.append((parent, idx))
        for child in obj.get("children") or []:
            visit(child, idx, room)

    # Room membership is only available for objects reachable from a room; the
    # top-level `objects` list is house-wide, so fall back to "unknown".
    for obj in house.get("objects", []) or []:
        visit(obj, None, obj.get("_room", "unknown"))
    return records, support


def _adjacency_edges(pos: np.ndarray, cfg: SceneGraphConfig) -> list[tuple[int, int]]:
    n = pos.shape[0]
    if n < 2:
        return []
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)

    edges: list[tuple[int, int]] = []
    if cfg.adjacency == "knn":
        k = min(cfg.k, n - 1)
        for i in range(n):
            for j in np.argpartition(d[i], k - 1)[:k]:
                edges.append((i, int(j)))
    else:
        for i, j in zip(*np.where(d <= cfg.radius)):
            edges.append((int(i), int(j)))
    return edges


def build_scene_graph(
    house: dict, house_id: str, cfg: SceneGraphConfig | None = None
) -> SceneGraph:
    """Build one scene graph from a ProcTHOR house.

    Args:
        house: a decoded house JSON record.
        house_id: stable identifier, used for provenance.
        cfg: edge-construction settings.

    Returns:
        A :class:`SceneGraph`. Deduplicated edges, positions untouched.
    """
    cfg = cfg or SceneGraphConfig()
    records, support = _flatten_objects(house)

    truncated = False
    if len(records) > cfg.max_nodes:
        keep = set(range(cfg.max_nodes))
        support = [(p, c) for p, c in support if p in keep and c in keep]
        records = records[: cfg.max_nodes]
        truncated = True

    if not records:
        return SceneGraph(
            house_id=house_id,
            node_ids=[],
            asset_ids=[],
            types=[],
            room_types=[],
            positions=np.zeros((0, 3), dtype=np.float32),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_kind=np.zeros((0,), dtype=np.int64),
            truncated=truncated,
            config=cfg,
        )

    pos = np.array(
        [[r["position"]["x"], r["position"]["y"], r["position"]["z"]] for r in records],
        dtype=np.float32,
    )
    node_ids = [str(r.get("id", f"obj_{i}")) for i, r in enumerate(records)]
    asset_ids = [str(r.get("assetId", "")) for r in records]
    types = [nid.split("|")[0] for nid in node_ids]
    room_types = [str(r.get("_room", "unknown")) for r in records]

    pairs: dict[tuple[int, int], int] = {}
    if cfg.include_support:
        for p, c in support:
            pairs[(p, c)] = SUPPORT
            if cfg.symmetric:
                pairs[(c, p)] = SUPPORT
    for i, j in _adjacency_edges(pos, cfg):
        # Support is the stronger, semantically explicit relation, so it wins
        # when the same pair is also geometrically adjacent -- which it usually
        # is, since a cup sits right on its table.
        pairs.setdefault((i, j), ADJACENCY)
        if cfg.symmetric:
            pairs.setdefault((j, i), ADJACENCY)

    if pairs:
        keys = sorted(pairs)
        edge_index = np.array(keys, dtype=np.int64).T
        edge_kind = np.array([pairs[k] for k in keys], dtype=np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_kind = np.zeros((0,), dtype=np.int64)

    return SceneGraph(
        house_id=house_id,
        node_ids=node_ids,
        asset_ids=asset_ids,
        types=types,
        room_types=room_types,
        positions=pos,
        edge_index=edge_index,
        edge_kind=edge_kind,
        truncated=truncated,
        config=cfg,
    )
