#!/usr/bin/env python3
"""Every data artifact on disk, counted and checked against what the MetaFind
paper says the data is. Read-only. One JSON report, one printed table.

Kyzen, 2026-09-02: 「所有資料也幫我掃一下 順便告訴我資料前處理我需要做什麼改變 一切以論文為主」

The paper's statements about the data (all from docs/paper/metafind_source):
  2methdology.tex:28  Objaverse-LVIS ~48,000 assets; each rendered from ELEVEN
                      orthogonal viewpoints; annotated with GPT-4o; descriptions
                      cover category, size dimensions, materials, placement
                      constraints. ProcTHOR: >10,000 houses, >3,000 unique
                      assets, per-asset coordinates and semantic metadata;
                      physical edges (adjacency, support) and semantic edges
                      (LLM relation sentences, frozen text encoder).
  3experiments.tex:8  80/20 split on both datasets; 48K unique assets;
                      ProcTHOR-10K.
  2methdology.tex:75  Stage 1: each asset has full modality inputs (text,
                      images, point clouds).
Everything else about the data (point count, image resolution, projection,
which eleven cameras, how eleven views become one vector) the paper does not
state, and this scan says so rather than inventing a target.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths  # noqa: E402

OUT = REPO / "output" / "look" / "data_scan_against_paper.json"

PAPER = {
    "objaverse_assets": "~48,000",
    "views_per_asset": 11,
    "views_kind": "orthogonal (camera set unspecified)",
    "annotator": "GPT-4o",
    "annotation_fields": ["category", "size dimensions", "materials", "placement constraints"],
    "procthor_houses": ">10,000",
    "procthor_assets": ">3,000",
    "split": "80/20 on both datasets",
    "stage1_inputs": "text + images + point cloud per asset (modality-complete)",
}


def count(path, pat):
    try:
        return sum(1 for _ in pathlib.Path(path).glob(pat))
    except Exception:
        return None


def main() -> int:
    R = {"paper": PAPER, "found": {}, "checks": []}

    def check(name, ok, detail):
        R["checks"].append({"name": name, "ok": bool(ok), "detail": detail})

    # ---- Objaverse-LVIS corpus ------------------------------------------
    lvis = json.loads((paths.DATASETS / "objaverse-lvis" / "lvis.json").read_text())
    n_pc = count(paths.POINTCLOUDS, "*.npz")
    n_ann = count(paths.ANNOTATIONS, "*.json")
    n_emb = count(paths.EMBEDDINGS, "*.npz")
    splits = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    train, test = set(splits["train"]), set(splits["test"])
    corpus = train | test
    R["found"]["objaverse"] = {
        "official_lvis_list": len(lvis), "pointclouds": n_pc, "annotations": n_ann,
        "embeddings": n_emb, "corpus": len(corpus), "train": len(train), "test": len(test),
        "train_fraction": round(len(train) / len(corpus), 4)}
    check("corpus vs paper 48K", False,
          f"official LVIS list {len(lvis):,}; corpus {len(corpus):,}; paper ~48,000 -- not "
          "reachable from the official list; 360 assets short of the list (28 render "
          "failures + 332 annotation failures)")
    check("80/20 split, disjoint", not (train & test) and abs(len(train) / len(corpus) - 0.8) < 0.01,
          f"train {len(train):,} test {len(test):,} overlap {len(train & test)}")

    # ---- renders: views per asset, protocol --------------------------------
    rend_idx = paths.LOGS / "renders_index.jsonl"
    views = collections.Counter()
    renderer = collections.Counter()
    n_rend = 0
    for line in rend_idx.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        n_rend += 1
        # the count lives in the nested renderer block (a string), and the
        # view list is the ground truth when it is present
        nv = len(r.get("view_paths") or []) or int((r.get("renderer") or {}).get("n_views", 0))
        views[nv] += 1
        renderer[str(r.get("renderer_version"))] += 1
    R["found"]["renders"] = {"assets": n_rend, "views_per_asset": dict(views),
                             "renderer_version": dict(renderer),
                             "protocol": "Blender, 12 views = three polar rings (60/90/120 deg) "
                                         "x four azimuths, 512 px, perspective, black composite"}
    check("views per asset vs paper 11", set(views) == {11},
          f"found {dict(views)}; paper says 11 orthogonal viewpoints. Camera set for "
          "'orthogonal' is unspecified in the paper")

    # ---- embeddings sidecars ----------------------------------------------
    side = collections.Counter()
    sample = sorted(paths.EMBEDDINGS.glob("*.json"))
    for p in sample[::max(1, len(sample) // 2000)]:
        d = json.loads(p.read_text())
        side[(d.get("n_views"), d.get("aggregation"), d.get("text_serialization"),
              d.get("encoder_version"))] += 1
    R["found"]["embedding_sidecars_sampled"] = {str(k): v for k, v in side.items()}
    check("embedding cache built under ONE protocol", len(side) == 1, str(dict(side)))

    # ---- annotations: fields the paper names -----------------------------
    fields = {"category": 0, "dimensions": 0, "materials": 0, "placement": 0,
              "description": 0}
    annot_model = collections.Counter()
    n_cand = []
    ann_files = sorted(paths.ANNOTATIONS.glob("*.json"))
    step = max(1, len(ann_files) // 3000)
    n_seen = 0
    for p in ann_files[::step]:
        a = json.loads(p.read_text())
        n_seen += 1
        annot_model[str(a.get("annotator_model", "")).rsplit("/", 1)[-1]] += 1
        if a.get("category"):
            fields["category"] += 1
        if all(a.get(k) is not None for k in ("height", "length")) or a.get("width") is not None:
            fields["dimensions"] += 1
        if a.get("materials"):
            fields["materials"] += 1
        if any(k in a for k in ("onFloor", "onWall", "onCeiling", "onObject")):
            fields["placement"] += 1
        if a.get("description"):
            fields["description"] += 1
        n_cand.append(len(a.get("description_candidates") or []))
    R["found"]["annotations_sampled"] = {
        "n_sampled": n_seen, "fields_present": {k: round(v / n_seen, 4) for k, v in fields.items()},
        "annotator_model": dict(annot_model),
        "description_candidates_per_asset": {"min": min(n_cand), "median": int(np.median(n_cand)),
                                             "max": max(n_cand)}}
    check("annotation fields vs paper (category, dimensions, materials, placement)",
          all(v == n_seen for k, v in fields.items() if k != "description"),
          {k: round(v / n_seen, 4) for k, v in fields.items()})
    check("annotator vs paper GPT-4o", False,
          f"found {dict(annot_model)}; paper GPT-4o; DEVIATION ruled by Kyzen (gemma)")

    # ---- point clouds -----------------------------------------------------
    pc_idx = paths.LOGS / "pointclouds_index.jsonl"
    pts = collections.Counter()
    ver = collections.Counter()
    for line in pc_idx.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pts[int(r.get("n_points", 0))] += 1
        ver[str(r.get("sampler_version"))] += 1
    R["found"]["pointclouds"] = {"n_points": dict(pts), "sampler_version": dict(ver),
                                 "columns": "xyz + rgb, pc_norm on xyz (ULIP-2 upstream; "
                                            "paper does not state a point count)"}
    check("point clouds uniform", len(pts) == 1 and len(ver) == 1, f"{dict(pts)} {dict(ver)}")

    # ---- ProcTHOR ---------------------------------------------------------
    graphs = sorted((paths.OUTPUTS / "scene_graphs").glob("*.json"))
    asset_ids, n_nodes, n_sup, n_adj, n_sem = set(), 0, 0, 0, 0
    for p in graphs:
        g = json.loads(p.read_text())
        for n in g["nodes"]:
            asset_ids.add(str(n["asset_id"]))
        n_nodes += len(g["nodes"])
        n_sup += len(g["phys_edges"].get("support", []))
        n_adj += len(g["phys_edges"].get("adjacency", []))
        n_sem += len(g.get("sem_edge_ids", []))
    mods = sorted(paths.PROCTHOR_MODALITIES.glob("*.json"))
    mod_views = collections.Counter()
    mod_proto = collections.Counter()
    no_pc = 0
    for p in mods:
        r = json.loads(p.read_text())
        ip = r.get("image_protocol", {})
        mod_views[ip.get("n_views")] += 1
        mod_proto[(ip.get("resolution"), ip.get("projection"))] += 1
        if r.get("pointcloud_uri") is None:
            no_pc += 1
    scene_splits = json.loads((paths.OUTPUTS / "scene_splits.json").read_text())
    text_map = json.loads((paths.OUTPUTS / "procthor_object_text.json").read_text())
    distinct_text = len({v["text"] for v in text_map.values()})
    sem = json.loads((paths.OUTPUTS / "sem_edge_cache.json").read_text())
    R["found"]["procthor"] = {
        "houses": len(graphs), "train_houses": len(scene_splits["train_houses"]),
        "test_houses": len(scene_splits["test_houses"]),
        "distinct_asset_ids_in_graphs": len(asset_ids), "modality_records": len(mods),
        "no_pointcloud": no_pc, "views_per_asset": dict(mod_views),
        "render_protocol": {str(k): v for k, v in mod_proto.items()},
        "nodes_total": n_nodes, "support_edges": n_sup, "adjacency_edges": n_adj,
        "semantic_edge_slots": n_sem, "semantic_sentences_cached": len(sem.get("entries", {})),
        "distinct_node_texts": distinct_text}
    check("ProcTHOR houses vs paper >10,000", len(graphs) >= 10000, f"{len(graphs):,}")
    check("ProcTHOR assets vs paper >3,000", len(asset_ids) > 3000,
          f"{len(asset_ids):,} distinct asset ids across all houses; {len(mods):,} modality "
          "records; paper says more than 3,000 curated assets")
    check("ProcTHOR render protocol = Objaverse render protocol", False,
          f"ProcTHOR {dict(mod_views)} views, {dict(mod_proto)}; Objaverse 12 views 512 px "
          "perspective. The paper describes one rendering (11 orthogonal) for Objaverse and "
          "none for ProcTHOR")
    check("node text carries semantic metadata (paper) vs category name only",
          distinct_text > 300, f"{distinct_text} distinct node sentences for {len(text_map):,} "
          "assets; ProcTHOR's metadata has no material/colour/style, only the category")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    print(f"{'check':<66s} {'ok':>4s}")
    for c in R["checks"]:
        print(f"{c['name']:<66s} {'yes' if c['ok'] else 'NO':>4s}")
        print(f"    {c['detail']}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
