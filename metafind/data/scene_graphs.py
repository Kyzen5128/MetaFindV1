"""ProcTHOR houses into spatial-semantic scene graphs.

# IMPLEMENTS-NODE: n07_scene_graphs

Writes ``scene_graphs`` (one sidecar per house, plus the derived index) and
``procthor_object_text`` (one map, keyed by ProcTHOR assetId), and
``quarantine`` / ``run_progress`` via runlog.

What the paper actually specifies
---------------------------------

Paper 2.3 names two edge types and gives one example of each:

  1. physical-relation edges, "spatial dependencies (e.g., 'cup on table')"
  2. semantic-relation edges, "functional or contextual associations
     (e.g., 'microscope-lab bench'), obtained by prompting an LLM on object
     pairs"

That is the whole specification. Everything below marked [U-nn] is our choice,
recorded here because the paper does not make it.

Support edges come free
-----------------------

ProcTHOR nests a cup as a ``children`` entry of its table, which is exactly the
paper's example. Reading the tree is not an inference -- it is the dataset's own
containment relation. [U-19] We store support edges in BOTH directions; the
paper never says whether either edge type is directed. L1-SCENE-SUPPORT pins
that convention so it cannot drift, and asserts our choice, not the paper's.

[U-05] Adjacency has no stated criterion at all. kNN with k=8 on raw positions,
recorded in every artifact so a reader can see what produced the graph.

[U-06] Which pairs get a semantic edge is likewise unstated. We take the unique
unordered pairs of the physical edges: an LLM asked about two objects that are
neither stacked nor near each other is being asked about a relation the scene
does not contain. n08 turns each pair into a sentence and an embedding.

[U-12] How ProcTHOR metadata becomes the sentence behind ``t_i`` is unstated.
ProcTHOR's per-asset semantics are category-level -- the object id carries the
category ("CounterTop|2|0"), the assetId carries only a variant suffix
("Countertop_I_8x2"), and there is no description field -- so the sentence is
category-derived and every instance of a category shares it. That is a ceiling
imposed by the dataset, not a simplification we chose.

Coordinates stay raw
--------------------

[L1-SCENE-COORDS-RAW] Positions are stored exactly as ProcTHOR gives them. Not
because centring would break equivariance -- x'_i = x_i - mean(x) leaves an EGNN
equivariant -- but because paper 2.5 states the setting as "large and often
unnormalized coordinate systems, with no guarantee that scenes are aligned or
centered". Pre-centring would remove the very global translation the design
claims robustness to, so the claim would never be exercised.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from metafind import paths, runlog

NODE = "n07_scene_graphs"
BUILDER_VERSION = 1

# [U-05] The adjacency criterion. Travels with every artifact.
ADJACENCY_MODE = "knn"
ADJACENCY_K = 8

SPLITS = ("train", "val", "test")

# Its own file outside SCENE_GRAPHS/. It lived beside the house sidecars for one
# revision, and rebuild_index's `*.json` glob promptly tried to read it as a
# house. A directory whose every entry is one channel record is a directory a
# glob can be trusted on.
OBJECT_TEXT_PATH = paths.OUTPUTS / "procthor_object_text.json"
REQUIRED_HOUSE_FIELDS = ("objects", "rooms")

# "CounterTop|2|0" -> "CounterTop"; "Apple|surface|2|0" -> "Apple".
# Split at a lower->upper boundary, or before the last capital of a run of
# capitals ("TVStand" -> "TV Stand"). Splitting before EVERY capital turned
# "CD" into "c d" and "TVStand" into "t v stand"; those strings were the node
# text t_i and the semantic-edge prompt input for 48,577 of 827,730 nodes
# (5.9%, measured 2026-09-02). Changing this changes the semantic-edge cache
# keys, so the fix takes effect only when the object-text map, the
# semantic-edge job and the node embeddings are regenerated together.
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def humanise(category: str) -> str:
    """"CounterTop" -> "counter top", "TVStand" -> "tv stand", "CD" -> "cd"."""
    return _CAMEL.sub(" ", category).lower().replace("_", " ").strip()


def object_text(category: str) -> str:
    """The sentence behind t_i, from ProcTHOR's category alone.

    Deliberately not enriched with room type or material: this map is keyed by
    assetId, and a chair's room varies per instance while its assetId does not.
    Folding instance context into an instance-independent key would make the
    same asset carry whichever room happened to be seen last.
    """
    noun = humanise(category)
    article = "an" if noun[:1] in "aeiou" else "a"
    return f"{article} {noun}"


def _category(obj: dict) -> str:
    """ProcTHOR puts the category in the id, not in a field of its own."""
    return str(obj.get("id", "")).split("|")[0] or obj.get("assetId", "unknown")


def _room_id(oid: str, known: set[str]) -> str | None:
    """MEASURED: room membership lives in the object id, not on the room.

    ``rooms[*].children`` is empty in every house checked (0 entries across 200
    houses), so the obvious join has nothing on one side of it. What ProcTHOR
    does carry is the room index as the second field of a top-level object id:
    ``CounterTop|2|0`` belongs to ``room|2``. Nested objects break the pattern
    (``Apple|surface|2|0``), which is why children inherit their parent's room
    in the caller instead of being parsed.
    """
    parts = oid.split("|")
    if len(parts) < 2:
        return None
    candidate = f"room|{parts[1]}"
    return candidate if candidate in known else None


def _flatten(objects: list[dict], rooms: set[str]) -> tuple[list[dict], list[tuple[int, int]]]:
    """Depth-first walk of the object tree.

    Returns the flat node list and the support pairs found on the way down, as
    ``(parent_index, child_index)``. Symmetrising happens in the caller so that
    the tree walk stays a faithful reading of the dataset.
    """
    nodes: list[dict] = []
    support: list[tuple[int, int]] = []

    def visit(obj: dict, parent: int | None) -> None:
        idx = len(nodes)
        oid = str(obj.get("id", ""))
        pos = obj.get("position") or {}
        nodes.append({
            "index": idx,
            "id": oid,
            "asset_id": obj.get("assetId"),
            "category": _category(obj),
            # A cup is in whatever room its table is in.
            "room_id": nodes[parent]["room_id"] if parent is not None
                       else _room_id(oid, rooms),
            "position": [float(pos.get("x", 0.0)), float(pos.get("y", 0.0)),
                         float(pos.get("z", 0.0))],
        })
        if parent is not None:
            support.append((parent, idx))
        for child in obj.get("children") or []:
            visit(child, idx)

    for obj in objects:
        visit(obj, None)
    return nodes, support


def _knn_pairs(positions: np.ndarray, k: int) -> list[tuple[int, int]]:
    """[U-05] Unordered kNN pairs on raw world coordinates.

    A house has tens of objects, so the full distance matrix is cheaper than any
    index. Self-distance is masked rather than sorted around, because argsort on
    a tie would otherwise let an object be its own neighbour.
    """
    n = len(positions)
    if n < 2:
        return []
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    kk = min(k, n - 1)
    nbrs = np.argpartition(d, kk - 1, axis=1)[:, :kk]
    return sorted({(min(i, int(j)), max(i, int(j)))
                   for i in range(n) for j in nbrs[i]})


def build_scene_graph(house: dict, house_id: str) -> dict:
    """One ProcTHOR house into one scene graph. Pure: no I/O, no globals.

    Purity is what makes L1-SCENE-COORDS-RAW testable -- the test translates a
    house by 100 m and calls this again, which only means anything if the same
    input always gives the same output.
    """
    missing = [f for f in REQUIRED_HOUSE_FIELDS if f not in house]
    if missing:
        raise KeyError(f"house is missing {', '.join(missing)}")

    room_type = {str(r.get("id", "")): r.get("roomType", "unknown")
                 for r in house["rooms"]}

    nodes, support_directed = _flatten(house["objects"], set(room_type))
    if not nodes:
        raise KeyError("house has no objects")

    positions = np.asarray([n["position"] for n in nodes], dtype=np.float64)

    # [U-19] Symmetric. Our convention, pinned by L1-SCENE-SUPPORT.
    support = sorted({(min(a, b), max(a, b)) for a, b in support_directed})
    adjacency = [p for p in _knn_pairs(positions, ADJACENCY_K) if p not in set(support)]

    # [U-06] Semantic candidates are the physical pairs. n08 fills in the
    # sentence and the embedding; the identity of a semantic edge at this stage
    # is its node pair, since n08's cache key needs an llm_model and a text
    # encoder version that do not exist yet here.
    sem_edge_ids = sorted(set(support) | set(adjacency))

    return {
        "house_id": house_id,
        "builder_version": BUILDER_VERSION,
        "room_types": room_type,
        "nodes": nodes,
        "positions": positions.tolist(),
        "phys_edges": {
            "support": [list(p) for p in support],
            "adjacency": [list(p) for p in adjacency],
        },
        "adjacency_criterion": {"mode": ADJACENCY_MODE, "k": ADJACENCY_K},
        "sem_edge_ids": [list(p) for p in sem_edge_ids],
    }


def sidecar_path(house_id: str) -> Path:
    return paths.SCENE_GRAPHS / f"{house_id}.json"


def is_complete(house_id: str) -> bool:
    sc = sidecar_path(house_id)
    if not sc.exists():
        return False
    try:
        rec = json.loads(sc.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return rec.get("builder_version") == BUILDER_VERSION and bool(rec.get("nodes"))


def iter_houses(limit: int | None = None) -> Iterator[tuple[str, dict]]:
    """Every house across the three splits, with a synthetic id.

    ProcTHOR ships no house id, so the id is (split, line number). It has to be
    stable: it is the key of the scene_graphs channel and the join key for
    n09c's splits.
    """
    seen = 0
    for split in SPLITS:
        path = paths.PROCTHOR / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open() as fh:
            for lineno, line in enumerate(fh):
                if not line.strip():
                    continue
                yield f"{split}_{lineno:05d}", json.loads(line)
                seen += 1
                if limit and seen >= limit:
                    return


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w") as fh:
        json.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def rebuild_index(index_path: Path) -> int:
    """Derive scene_graphs_index.jsonl from the sidecars. Never appended to."""
    tmp = index_path.with_suffix(".jsonl.part")
    n = 0
    with tmp.open("w") as fh:
        for sc in sorted(paths.SCENE_GRAPHS.glob("*.json")):
            try:
                rec = json.loads(sc.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            fh.write(json.dumps({
                "house_id": rec["house_id"],
                "uri": str(sc),
                "n_nodes": len(rec["nodes"]),
                "n_support": len(rec["phys_edges"]["support"]),
                "n_adjacency": len(rec["phys_edges"]["adjacency"]),
                "n_sem_candidates": len(rec["sem_edge_ids"]),
            }) + "\n")
            n += 1
    tmp.replace(index_path)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not paths.PROCTHOR.exists():
        print(f"{paths.PROCTHOR} not found -- run n02_download first", flush=True)
        return 2
    paths.SCENE_GRAPHS.mkdir(parents=True, exist_ok=True)

    done, skipped, quarantined, started = 0, 0, 0, time.time()
    object_text_map: dict[str, dict[str, str]] = {}

    with runlog.run_progress(NODE):
        for house_id, house in iter_houses(args.limit):
            if not args.force and is_complete(house_id):
                skipped += 1
                # The text map is DERIVED from every house, not only the ones
                # built this run, so a resumed run must still walk the skipped
                # houses' assets or it would publish a map missing whatever the
                # earlier run contributed.
                try:
                    graph = json.loads(sidecar_path(house_id).read_text())
                except (OSError, json.JSONDecodeError):
                    graph = None
                if graph:
                    for n in graph["nodes"]:
                        object_text_map.setdefault(str(n["asset_id"]), {
                            "text": object_text(n["category"]),
                            "source": "procthor_category",
                        })
                continue

            try:
                graph = build_scene_graph(house, house_id)
            except (KeyError, TypeError, ValueError) as exc:
                runlog.quarantine(NODE, [{
                    "house_id": house_id,
                    "failure_class": "DETERMINISTIC_INPUT",
                    "missing_fields": str(exc)[:200],
                    "exception_type": type(exc).__name__,
                }])
                quarantined += 1
                continue

            for n in graph["nodes"]:
                object_text_map.setdefault(str(n["asset_id"]), {
                    "text": object_text(n["category"]),
                    "source": "procthor_category",
                })
            _write_json(sidecar_path(house_id), graph)
            done += 1
            if done % 2000 == 0:
                rate = done / max(time.time() - started, 1e-9) * 60
                print(f"  [{done:6d}] {rate:.0f}/min, quarantine {quarantined}",
                      flush=True)

    _write_json(OBJECT_TEXT_PATH, object_text_map)
    n_indexed = rebuild_index(paths.LOGS / "scene_graphs_index.jsonl")
    print(f"\n{done:,} built, {skipped:,} already present, {n_indexed:,} on disk, "
          f"{quarantined:,} quarantined; {len(object_text_map):,} assetIds "
          f"-> {paths.SCENE_GRAPHS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
