#!/usr/bin/env python3
"""Regenerate ProcTHOR's node text and its semantic edges, or change nothing.

THE HAZARD THIS TOOL EXISTS FOR
-------------------------------
`procthor_object_text.json` on disk was written before commit `2f255f5`, which
fixed two things at once: the camel-case split (`"CD"` became `"a c d"`, `
"TVStand"` became `"a t v stand"`) and the article rule (`"a apple"` should be
`"an apple"`). 146 of 1,467 assetIds change under the current code, which is
48,577 of 827,730 node instances, 5.87%.

Regenerating the text alone is silently destructive. `_edge_key` hashes the
DESCRIPTIONS, so a changed description is a changed cache key; the lookup misses;
`vec is None`; `edge_missing = True`; and the learned missing token substitutes
for the edge. Nothing raises. Training proceeds. Table 2 and Table 3 are wrong.

Measured over 200 random houses and 66,603 edges: **7,702 edges, 11.56%, would
silently become "missing"**.

So the text and the edges are one operation or they are no operation. Every
output is staged and nothing is moved into place until all of them exist:

    procthor_object_text.json
    sem_edge_cache.json
    sem_edge_sentences.jsonl
    sem_edge_embeddings.npz
    procthor_node_embeddings.npz + .json   (the record carries the npz's sha256)

An interruption at any point leaves the corpus exactly as it was, because the
only mutation is the final sequence of `Path.replace` calls, and each source is
already complete on the same filesystem.

WHAT THIS DOES NOT DECIDE
-------------------------
`REPRODUCTION_PROTOCOL_20260903.md` 問題 10 calls ProcTHOR node-text
construction **UNRESOLVED**: the paper does not say whether `t_i` comes from a
category name, a metadata template, a description, or an LLM. This tool does not
answer that. It only makes the CURRENT rule -- a category string through
`scene_graphs.object_text` -- reproducible without corrupting the edges. If the
node-text question is later answered differently, the same atomicity applies and
this tool is where that change goes.

USAGE
-----
    python tools/repair_procthor_node_text.py            # dry run, reads only
    python tools/repair_procthor_node_text.py --apply    # needs the GPU

The dry run is the default because the apply path loads a 23 GB VLM and
overwrites five artifacts. §十九 forbids running it without Kyzen's decision.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

from metafind import paths
from metafind.data.scene_graphs import object_text
from metafind.data.semantic_edges import PROMPT_VERSION, cache_key

TEXT_PATH = paths.OUTPUTS / "procthor_object_text.json"
CACHE_PATH = paths.OUTPUTS / "sem_edge_cache.json"
SENTENCES_PATH = paths.OUTPUTS / "sem_edge_sentences.jsonl"
EMB_PATH = paths.OUTPUTS / "sem_edge_embeddings.npz"
NODE_EMB_PATH = paths.OUTPUTS / "procthor_node_embeddings.npz"
NODE_EMB_RECORD = paths.OUTPUTS / "procthor_node_embeddings.json"

OUTPUTS = (TEXT_PATH, CACHE_PATH, SENTENCES_PATH, EMB_PATH,
           NODE_EMB_PATH, NODE_EMB_RECORD)


def current_text() -> dict:
    return json.loads(TEXT_PATH.read_text())


def scan_scene_graphs() -> tuple[dict, set]:
    """One walk of the 12,000 graphs: asset -> category, and the pair set.

    The category comes from the SCENE GRAPHS, not from `procthor_object_text`.
    The stored map holds only `{text, source}`, and the first version of this
    tool tried to recover the category from the text -- that is inverting
    `object_text`, which is exactly the lossy direction: `"a c d"` and `"a cd"`
    both invert to something, and only one of them is right. Refusing to guess
    was correct; taking the category from where it is actually recorded is
    better.

    Both products come from one scan because walking 12,000 files twice to
    answer one question invites the two answers to disagree.
    """
    cats, pairs = {}, set()
    for f in sorted(glob.glob(str(paths.SCENE_GRAPHS / "*.json"))):
        g = json.loads(Path(f).read_text())
        by_index = {}
        for n in g["nodes"]:
            cats.setdefault(n["asset_id"], n["category"])
            by_index[n["index"]] = n["asset_id"]
        for i, j in g.get("sem_edge_ids") or []:
            a, b = by_index.get(i), by_index.get(j)
            if a is not None and b is not None:
                pairs.add((a, b))
    return cats, pairs


def rebuilt_text(cats: dict) -> dict:
    """The map as today's `object_text` would write it, from the real categories."""
    return {a: {"text": object_text(c), "source": "procthor_category"}
            for a, c in cats.items()}


def pair_keys(text_map: dict, pairs: set, llm_model: str,
              encoder_version: str) -> dict:
    """Every distinct description pair the corpus asks for, keyed as n08 keys it.

    The pair set is the scene graphs' `sem_edge_ids`, the union of support and
    adjacency -- the corpus's question, not this tool's.
    """
    want = {}
    for a, b in pairs:
        da = (text_map.get(a) or {}).get("text")
        db = (text_map.get(b) or {}).get("text")
        if da is None or db is None:
            continue
        k = cache_key(da, db, PROMPT_VERSION, llm_model, encoder_version)
        want.setdefault(k, (da, db))
    return want


def plan() -> dict:
    """What an apply would change. Reads only; touches no GPU."""
    old = current_text()
    cats, pairs = scan_scene_graphs()
    new = rebuilt_text(cats)
    changed = {a: (old[a]["text"], new[a]["text"])
               for a in old if a in new and old[a]["text"] != new[a]["text"]}

    cache = json.loads(CACHE_PATH.read_text())
    llm_model = cache["llm_model"]
    encoder_version = cache["text_encoder_version"]

    have = set(cache["entries"])
    want_old = pair_keys(old, pairs, llm_model, encoder_version)
    want_new = pair_keys(new, pairs, llm_model, encoder_version)

    missing = sorted(set(want_new) - have)
    orphaned = sorted(have - set(want_new))
    return {
        "assets_total": len(old),
        "assets_changed": len(changed),
        "examples": [f"{o!r} -> {n!r}" for o, n in list(changed.values())[:8]],
        "pairs_before": len(want_old), "pairs_after": len(want_new),
        "cache_entries": len(have),
        "pairs_needing_a_new_llm_call": len(missing),
        "cache_entries_that_would_be_orphaned": len(orphaned),
        "llm_model": llm_model, "text_encoder_version": encoder_version,
        "prompt_version": PROMPT_VERSION,
        # The number that makes this a repair and not an edit.
        "edges_that_would_silently_go_missing_if_text_alone_were_regenerated":
            "measured 11.56% over 200 houses (PHASE1_AUDIT_20260903 section C)",
    }


def apply(dry: bool) -> int:
    """Stage every output, then move them in together, or move nothing."""
    p = plan()
    if not p["assets_changed"]:
        print("nothing to repair: the on-disk text already matches this code.")
        return 0
    if dry:
        return 0

    missing_before = [f for f in OUTPUTS if not f.exists()]
    if missing_before:
        raise SystemExit(
            f"refusing: {[str(f) for f in missing_before]} do not exist, so "
            "there is nothing to repair atomically. Run n08 first.")

    stage = Path(tempfile.mkdtemp(prefix="procthor_repair_", dir=paths.OUTPUTS))
    try:
        # Everything below writes into `stage` only. The GPU work -- the LLM for
        # the new pairs and the frozen text tower for the vectors -- belongs to
        # `semantic_edges_run`, which is imported here rather than reimplemented
        # so this tool cannot drift from the node that owns the protocol.
        from metafind.data import semantic_edges_run as n08

        raise SystemExit(
            "APPLY IS NOT WIRED. The staging skeleton, the refusal path and the "
            "dry run are complete and tested; the generation step deliberately "
            "is not, because it needs the GPU and a decision Kyzen has not "
            "made: REPRODUCTION_PROTOCOL_20260903 問題 10 calls node-text "
            "construction UNRESOLVED, so repairing the CURRENT rule may not be "
            "what is wanted. Wiring it means calling semantic_edges_run's "
            "generation and encoding into `stage`, then the replace loop below. "
            f"The dry run says: {p['assets_changed']} assets change, "
            f"{p['pairs_needing_a_new_llm_call']} pairs need a new call.")
        # for src, dst in ...: src.replace(dst)   # the only mutation
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually repair. Needs the GPU and Kyzen's decision.")
    args = ap.parse_args()

    p = plan()
    print("ProcTHOR node-text repair -- DRY RUN" if not args.apply else
          "ProcTHOR node-text repair -- APPLY")
    for k in ("assets_total", "assets_changed", "pairs_before", "pairs_after",
              "cache_entries", "pairs_needing_a_new_llm_call",
              "cache_entries_that_would_be_orphaned"):
        print(f"  {k:<42} {p[k]:,}" if isinstance(p[k], int)
              else f"  {k:<42} {p[k]}")
    print(f"  llm_model                                  {p['llm_model']}")
    for e in p["examples"]:
        print(f"    {e}")
    print("\n  if the text were regenerated WITHOUT the edges: "
          + p["edges_that_would_silently_go_missing_if_text_alone_were_regenerated"])
    return apply(dry=not args.apply)


def demo() -> None:
    """Self-check: the plan is read-only and the refusal path is the easy one."""
    before = {f: (f.stat().st_mtime_ns if f.exists() else None) for f in OUTPUTS}
    p = plan()
    after = {f: (f.stat().st_mtime_ns if f.exists() else None) for f in OUTPUTS}
    assert before == after, "plan() modified an artifact; it must read only"
    assert p["assets_changed"] >= 0
    # The apply path must refuse rather than half-write. Asserted, because
    # "it is not wired yet" is a property that must FAIL LOUDLY, not silently
    # succeed the day someone deletes the raise.
    try:
        apply(dry=False)
    except SystemExit as e:
        assert "NOT WIRED" in str(e) or "refusing" in str(e), e
    else:
        raise AssertionError("apply() returned without generating or refusing")
    after2 = {f: (f.stat().st_mtime_ns if f.exists() else None) for f in OUTPUTS}
    assert before == after2, "the refused apply still touched an artifact"
    print(f"repair demo ok: plan() is read-only, apply() refuses, "
          f"{p['assets_changed']} assets would change")


if __name__ == "__main__":
    raise SystemExit(main())
