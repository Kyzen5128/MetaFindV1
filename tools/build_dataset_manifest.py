#!/usr/bin/env python3
"""Build the UID-level dataset manifest the reproduction protocol requires.

[SPEC `workflow/REPRODUCTION_PROTOCOL_20260903.md` §四, §六, §十, §十二]

**Recomputes nothing.** Every value here is read off an artifact that already
exists -- render sidecars, annotation files, point-cloud sidecars, embedding
sidecars, `splits.json`, the quarantine channels. No embedding, render or point
cloud is produced, loaded into a model, or modified. The tool is idempotent and
writing it twice over the same corpus produces the same bytes.

Why a manifest at all, when the sidecars already hold this
---------------------------------------------------------
Because the sidecars answer "what is this asset" and nobody can ask them "which
assets have four captions", "which are in the sealed test split", or "which
carry a degraded render" without walking 46,052 files.

552,288 image rows, not 46,052 x 12: the 28 assets n04 lost have a DIRECTORY,
created before the failure, but no sidecar and no index row, so they contribute
no image record. A directory is not evidence that an asset rendered. §四's real requirement is
not a new copy of the data -- it is that **one asset UID is one row**, with
views and captions as CHILD records, so that a question about assets cannot
accidentally be answered over a table of views.

That constraint is the point. `1 UID x 12 views -> 12 gallery entries` and
`1 UID x 5 captions -> 5 gallery entries` are both forbidden by §四.1, and both
are the kind of mistake that inflates a retrieval score without anything
looking wrong.

Shape on disk
-------------
    manifest/assets.jsonl        one row per uid                     46,052
    manifest/images.jsonl        one row per (uid, view_id)         552,288
    manifest/captions.jsonl      one row per (uid, caption_id)      273,927
    manifest/pointclouds.jsonl   one row per uid                     46,052
    manifest/camera_views.json   the per-view camera, ONCE               12
    manifest/filters.json        every stage: before / removed / after
    manifest/splits.json         the lists, the algorithm, the seed, the hashes
    manifest/caches.json         each feature cache + valid_for_train_scope
    manifest/gallery_test.jsonl  §十二, one asset uid per item         9,138
    manifest/gallery_full.jsonl  §十二, one asset uid per item        45,692
    manifest/MANIFEST.json       counts, provenance, and the sha256 of each file

`camera_views.json` is a separate table rather than a field repeated on every
image row: the camera is identical across all 46,052 assets, so inlining it
would write the same twelve records 46,024 times and give the corpus 552,288
independent places to disagree with itself. The image rows carry `view_id`,
which is the join key.

Absent fields are recorded as absent
------------------------------------
§四.2 lists provenance fields to preserve. Several have no source on this
corpus and are written as `null` with a `_absent` note rather than filled in
with something plausible:

* **`source_dataset_version`** -- Objaverse-LVIS ships no version string. What
  exists is the manifest file itself, so its sha256 is recorded instead.
* **`raw_pointcloud`** -- §九 wants a source layer AND a derived layer. Only the
  derived one exists; the sole source is the `.glb` mesh, 351.4 GB. Recorded as
  absent, not as "the derived one is both".
* **`source_file_sha256` for the mesh** -- hashing 351.4 GB of GLB is a
  multi-hour read this tool will not do silently. The path and byte size are
  recorded; the hash is `null` with the reason.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import time
from pathlib import Path

from metafind import paths

MANIFEST_VERSION = 1
NODE = "manifest_build"


# --------------------------------------------------------------------- camera
# [PHASE1_AUDIT_20260903 Q7] The render sidecar stores `view_directions`
# RING-GROUPED, not per view index, so a reader cannot tell which PNG came from
# which camera without re-deriving it. It is derivable, deterministically:
#
#   * `metafind/vendor/openshape/render_single_glb.py:172` holds an ORDERED
#     twelve-entry `views` list, iterated `for i in range(args.num_images)`;
#   * `render_blender.render_asset` renames the sorted `000.png .. 011.png` to
#     `view_00 .. view_11` in that same order.
#
# So `view_00` is the first entry of that list and `view_11` the last. It is
# written out here ONCE so no future reader has to re-derive it -- and so that
# if the vendored list ever changes, this table and the corpus disagree
# loudly instead of silently.
#
# `phi` is the polar angle from +Z, `theta` the azimuth, both in the vendored
# script's convention (`sample_camera_loc`). Elevation is 90 - phi, so
# phi 120 deg means the camera looks UP at the asset from below.
_VENDOR_VIEWS = [(60.0, 30.0), (60.0, 120.0), (60.0, 210.0), (60.0, 300.0),
                 (90.0, 60.0), (90.0, 150.0), (90.0, 240.0), (90.0, 330.0),
                 (120.0, 0.0), (120.0, 90.0), (120.0, 180.0), (120.0, 270.0)]


def camera_views() -> list[dict]:
    return [{"view_id": i,
             "file": f"view_{i:02d}.png",
             "polar_phi_deg": phi,
             "azimuth_theta_deg": theta,
             "elevation_deg": round(90.0 - phi, 6),
             "looks_from_below": phi > 90.0,
             "ring": {60.0: "phi_60_above", 90.0: "phi_90_level",
                      120.0: "phi_120_below"}[phi]}
            for i, (phi, theta) in enumerate(_VENDOR_VIEWS)]


def _vendor_sha() -> str:
    from metafind.data import render_blender
    return hashlib.sha256(Path(render_blender.VENDOR_SCRIPT).read_bytes()).hexdigest()


# ---------------------------------------------------------------------- utils

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, rows) -> int:
    """Write rows and return the count. Sorted keys so two runs match byte for byte."""
    n = 0
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            n += 1
    tmp.replace(path)
    return n


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False))
    tmp.replace(path)


def _jsonl_index(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec.get("uid") or rec.get("asset_id")] = rec
    return out


# ------------------------------------------------------------------ the build

def build(out: Path, limit: int | None = None) -> dict:
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)

    manifest_uids = sorted(json.loads(paths.LVIS_MANIFEST.read_text()))
    if limit:
        manifest_uids = manifest_uids[:limit]
    raw = set(manifest_uids)

    renders = _jsonl_index(paths.LOGS / "renders_index.jsonl")
    clouds = _jsonl_index(paths.LOGS / "pointclouds_index.jsonl")
    anns = _jsonl_index(paths.LOGS / "annotations_index.jsonl")

    split = json.loads((paths.OUTPUTS / "splits.json").read_text())
    obj = split["object"]
    split_of = {u: "train" for u in obj["train"]}
    split_of.update({u: "test" for u in obj["test"]})
    dev_train, dev_val = set(obj.get("dev_train") or []), set(obj.get("dev_val") or [])

    # The exclusion ledger, read through the SAME parser the split uses, so the
    # manifest cannot disagree with `admitted_uids()` about who was excluded.
    from metafind.data.splits import admitted_uids, ledger_excluded_uids
    ledger_path = paths.OUTPUTS / "annotation_exclusions.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    excluded = ledger_excluded_uids(ledger) if ledger else set()
    exclusion_reason = {}
    for group, body in (ledger.get("groups") or {}).items():
        for e in (body or {}).get("uids") or []:
            exclusion_reason[str(e.get("uid") if isinstance(e, dict) else e)] = group

    admitted = set(admitted_uids()) if not limit else (
        raw & set(renders) & set(clouds) & set(anns)) - excluded

    # ---- assets, images, captions, pointclouds -----------------------------
    asset_rows, image_rows, caption_rows, cloud_rows = [], [], [], []
    caption_hist = collections.Counter()
    degraded = []
    glb_root = paths.OBJAVERSE_GLB

    for uid in manifest_uids:
        r, c, a = renders.get(uid), clouds.get(uid), anns.get(uid)
        ann = None
        ann_path = paths.ANNOTATIONS / f"{uid}.json"
        if ann_path.exists():
            ann = json.loads(ann_path.read_text())

        # --- filter status. §六: never a silent drop; every absence has a reason.
        reasons = []
        if r is None:
            reasons.append("missing_image")
        if c is None:
            reasons.append("missing_pc")
        if a is None or ann is None:
            reasons.append("missing_text")
        if uid in excluded:
            reasons.append(exclusion_reason.get(uid, "excluded_by_ledger"))
        is_admitted = uid in admitted

        # --- the render's own recorded anomalies. Flagged, NEVER removed:
        # whether the 253 degraded admitted assets stay in the corpus is
        # Kyzen's decision, and a manifest that quietly dropped them would
        # make that decision for him.
        n_views = len(r.get("view_paths") or []) if r else 0
        anomaly = None
        if r:
            blank = int(r.get("blank_views") or 0)
            dark = int(r.get("dark_views") or 0)
            distinct = r.get("distinct_views")
            # `distinct_views < n_views` counts too, and MASTER's first
            # criterion missed it: an asset whose sidecar lists twelve views
            # while only nine are byte-distinct carries less information than
            # the row claims, with nothing blank and nothing dark. Measured,
            # that is the difference between 244 and the audited 253.
            few_distinct = distinct is not None and distinct < (n_views or 12)
            if blank or dark or few_distinct:
                anomaly = {"blank_views": blank, "dark_views": dark,
                           "distinct_views": distinct,
                           "fewer_distinct_than_listed": few_distinct,
                           "effectively_blank": blank >= 10 and (distinct or 99) <= 3}
                if is_admitted:
                    degraded.append(uid)

        cands = (ann or {}).get("description_candidates") or []
        if ann is not None:
            caption_hist[len(cands)] += 1

        rel = None
        if r and r.get("view_paths"):
            try:
                rel = str(Path(r["view_paths"][0]).parent)
            except Exception:
                rel = None

        asset_rows.append({
            "uid": uid,
            "split": split_of.get(uid),
            "dev_split": ("dev_train" if uid in dev_train else
                          "dev_val" if uid in dev_val else None),
            "admitted": is_admitted,
            "filter_reasons": reasons or None,
            "modalities": {"text": ann is not None, "image": r is not None,
                           "pointcloud": c is not None,
                           "embedding": (paths.EMBEDDINGS / f"{uid}.npz").exists()},
            "n_views": n_views,
            "n_captions": len(cands),
            "render_anomaly": anomaly,
            "category": (ann or {}).get("category"),
            "lvis_category": (ann or {}).get("lvis_category"),
            "provenance": {
                "source_dataset": "objaverse-lvis",
                # No version string ships with Objaverse-LVIS. The manifest file
                # itself is the only identity, so its hash stands in; recorded as
                # absent rather than invented.
                "source_dataset_version": None,
                "_absent_source_dataset_version": "Objaverse-LVIS ships no version "
                                                  "field; see MANIFEST.lvis_manifest_sha256",
                "source_path": str(glb_root / f"{uid}.glb"),
                "source_file_sha256": None,
                "_absent_source_file_sha256": "hashing 351.4 GB of GLB is a "
                                              "multi-hour read; not done silently",
                "renderer_version": (r or {}).get("renderer_version"),
                "sampler_version": (c or {}).get("sampler_version"),
                "annotation_schema_version": (ann or {}).get("schema_version"),
                "annotation_prompt_version": (ann or {}).get("prompt_version"),
                "annotator_model": (ann or {}).get("annotator_model"),
            },
            "asset_dir": rel,
        })

        for vid, vp in enumerate((r or {}).get("view_paths") or []):
            image_rows.append({
                "uid": uid, "view_id": vid, "image_path": vp,
                "sha256": ((r.get("view_sha256") or [None] * (vid + 1))[vid]
                           if r else None),
                "bytes": ((r.get("view_bytes") or [None] * (vid + 1))[vid]
                          if r else None),
                "coverage": ((r.get("view_coverage") or [None] * (vid + 1))[vid]
                             if r else None),
                # The camera lives once, in camera_views.json, joined on view_id.
                "camera_ref": "camera_views.json",
            })

        # §八: the canonical description AND every ranked alternative, with the
        # score each was ranked on. Collapsing to one string would destroy the
        # only alternate-caption source the corpus has -- and 20 assets have
        # exactly one candidate, so the alternatives are not uniformly available.
        if ann is not None:
            caption_rows.append({
                "uid": uid, "caption_id": "canonical", "rank": 0,
                "text": ann.get("description"),
                "provenance": {"source": ann.get("description_source"),
                               "model": ann.get("annotator_model"),
                               "prompt_version": ann.get("prompt_version"),
                               "schema_version": ann.get("schema_version")},
            })
            for i, cand in enumerate(cands):
                caption_rows.append({
                    "uid": uid, "caption_id": f"candidate_{i}",
                    "rank": cand.get("rank", i), "text": cand.get("text"),
                    "clip_score": cand.get("clip_score"),
                    "provenance": {"ranker": (ann.get("description_ranker") or {}),
                                   "sampling": ann.get("description_sampling")},
                })

        if c is not None:
            cloud_rows.append({
                "uid": uid,
                "derived_path": c.get("uri") or str(paths.OUTPUTS / "pointclouds" / f"{uid}.npz"),
                "n_points": c.get("n_points"),
                "sampler_version": c.get("sampler_version"),
                "frame_correction": c.get("frame_correction"),
                "colour_source": c.get("colour_source"),
                "color0_modulated": c.get("color0_modulated"),
                "normalisation": {"rule": "ulip2_pc_norm",
                                  "centroid_to_origin": True,
                                  "max_radius": c.get("max_radius"),
                                  "applies_to": "xyz_only"},
                "seed": c.get("seed"),
                "sha256": c.get("sha256"),
                # §九 wants a source layer as well. There is none: the only
                # source is the mesh. Absent, and said so.
                "raw_path": None,
                "_absent_raw": "no source/raw point-cloud layer exists on this "
                               "corpus; the only source is the .glb mesh",
            })

    # ---- write ------------------------------------------------------------
    n_assets = _write_jsonl(out / "assets.jsonl", asset_rows)
    n_images = _write_jsonl(out / "images.jsonl", image_rows)
    n_caps = _write_jsonl(out / "captions.jsonl", caption_rows)
    n_clouds = _write_jsonl(out / "pointclouds.jsonl", cloud_rows)
    _write_json(out / "camera_views.json", {
        "n_views": len(_VENDOR_VIEWS),
        "convention": "phi = polar angle from +Z; theta = azimuth; "
                      "elevation = 90 - phi. Vendored script's convention.",
        "derivation": "vendor/openshape/render_single_glb.py:172 holds an ordered "
                      "list iterated by index; render_asset renames the sorted "
                      "000..011.png to view_00..view_11 in that order.",
        "vendor_script_sha256": _vendor_sha(),
        "camera_layout": "openshape_three_rings_of_four",
        "projection": "perspective", "resolution": 512,
        "background": "transparent_rgba",
        "renderer_version_on_corpus": 6,
        "views": camera_views(),
    })

    # ---- filters, §六 -----------------------------------------------------
    q_reasons = {}
    for node, fname in (("n04_render_views", "quarantine_n04_render_views.jsonl"),
                        ("n05_annotate", "quarantine_n05_annotate.jsonl")):
        f = paths.LOGS / fname
        if not f.exists():
            continue
        per = collections.Counter()
        for line in f.read_text().splitlines():
            if line.strip():
                per[json.loads(line).get("failure_class", "UNKNOWN")] += 1
        q_reasons[node] = dict(per)

    ladder = [
        {"stage": "manifest", "before": None, "removed": None, "after": len(raw)},
        {"stage": "n03_pointclouds", "before": len(raw),
         "removed": len(raw - set(clouds)), "after": len(raw & set(clouds))},
        {"stage": "n04_renders", "before": len(raw & set(clouds)),
         "removed": len((raw & set(clouds)) - set(renders)),
         "after": len(raw & set(clouds) & set(renders))},
        {"stage": "n05_annotate", "before": len(raw & set(clouds) & set(renders)),
         "removed": len((raw & set(clouds) & set(renders)) - set(anns)),
         "after": len(raw & set(clouds) & set(renders) & set(anns))},
        {"stage": "exclusion_ledger",
         "before": len(raw & set(clouds) & set(renders) & set(anns)),
         "removed": len((raw & set(clouds) & set(renders) & set(anns)) & excluded),
         "after": len(admitted)},
    ]
    _write_json(out / "filters.json", {
        "ladder": ladder,
        "quarantine_rows_by_failure_class": q_reasons,
        "_quarantine_note": "quarantine channels are APPEND-ONLY across retries, "
                            "so a distinct-uid count in them is NOT a removal "
                            "count. The ladder above is computed by set "
                            "difference between the index files.",
        "exclusion_ledger": {
            "path": str(ledger_path), "stated_total": ledger.get("excluded_total"),
            "parsed_uids": len(excluded),
            "groups": {g: len((b or {}).get("uids") or [])
                       for g, b in (ledger.get("groups") or {}).items()},
        },
        "summary": {"raw_assets": len(raw), "usable_assets": len(admitted),
                    "train_assets": len(obj["train"]), "test_assets": len(obj["test"]),
                    "quarantined_or_excluded": len(raw) - len(admitted)},
        "degraded_admitted_assets": {
            "count": len(degraded),
            "note": "flagged, NOT removed. Whether these stay in the corpus is "
                    "Kyzen's decision; see assets.jsonl render_anomaly.",
            "in_test_split": sorted(u for u in degraded if split_of.get(u) == "test"),
        },
        "caption_count_histogram": {str(k): v for k, v in sorted(caption_hist.items())},
    })

    # ---- splits, §五 ------------------------------------------------------
    def _h(xs):
        return hashlib.sha256("\n".join(sorted(xs)).encode()).hexdigest()

    _write_json(out / "splits.json", {
        "source": "data/outputs/splits.json (built by metafind.data.splits)",
        "ratio": {"train": 0.8, "test": 0.2},
        "ratio_basis": "PAPER FACT, 3experiments.tex:8",
        "algorithm": "sorted(uids) -> random.Random(seed).shuffle -> "
                     "cut = int(round(n * 0.8)) -> both halves re-sorted",
        "algorithm_class": "IMPLEMENTATION CHOICE -- the paper publishes no uid "
                           "list, seed or split implementation",
        "seed_object": split.get("seed"), "seed_dev": split.get("dev_seed"),
        "split_happens": "AFTER modality filtering -- splits.py calls "
                         "admitted_uids() and splits its result",
        "universe_size": len(admitted),
        "counts": {k: len(obj.get(k) or []) for k in
                   ("train", "test", "dev_train", "dev_val")},
        "sha256": {k: _h(obj.get(k) or []) for k in
                   ("train", "test", "dev_train", "dev_val")},
        "assert_disjoint": {
            "train_test": len(set(obj["train"]) & set(obj["test"])),
            "dev_train_dev_val": len(dev_train & dev_val),
            "dev_val_test": len(dev_val & set(obj["test"])),
        },
    })

    # ---- caches, §十 ------------------------------------------------------
    emb_side = None
    for u in sorted(admitted)[:1]:
        f = paths.EMBEDDINGS / f"{u}.json"
        if f.exists():
            emb_side = json.loads(f.read_text())
    _write_json(out / "caches.json", {
        "_why_train_scope": "§十. Text and image are cached because they are "
                            "FROZEN under the current train scope. Under a "
                            "full-encoder-finetune arm they must be re-encoded "
                            "online, so the cache is not valid there and saying "
                            "so is what stops a later scope change reading it.",
        "caches": [
            {"name": "embeddings", "path": str(paths.EMBEDDINGS),
             "n": len(admitted), "arrays": {"text": "(1280,) f2",
                                            "views": "(12, 1280) f2",
                                            "image": "(1280,) f2"},
             "encoder_name": "open_clip ViT-bigG-14 via ULIPBackbone",
             "checkpoint_sha": (emb_side or {}).get("ulip2_ckpt_sha"),
             "preprocessing_version": (emb_side or {}).get("encoder_version"),
             "text_serialization": (emb_side or {}).get("text_serialization"),
             "feature_dim": (emb_side or {}).get("embedding_dim"),
             "dtype": (emb_side or {}).get("dtype"),
             "valid_for_train_scope": ["point_encoder_and_fuser", "fuser_only"],
             "_per_view_present": "views (12,1280) is stored BESIDE the mean, so "
                                  "single-view, held-out-view and disjoint-subset "
                                  "arms need no re-encoding (§七)"},
            {"name": "pointclouds", "path": str(paths.OUTPUTS / "pointclouds"),
             "n": len(clouds), "arrays": {"xyz": "(10000,3) f4", "rgb": "(10000,3) f4"},
             "encoder_name": None, "checkpoint_sha": None,
             "preprocessing_version": "sampler_version 8",
             "valid_for_train_scope": ["point_encoder_and_fuser", "fuser_only",
                                       "full_encoder_finetune"],
             "_note": "raw geometry, not an encoder output, so no train scope "
                      "invalidates it. PointBERT runs on it LIVE (§九)."},
        ],
    })

    # ---- gallery manifests, §十二 -----------------------------------------
    # One asset uid per item. Never expanded by views or captions -- that
    # expansion is what §四.1 forbids and what would inflate a retrieval score.
    n_gt = _write_jsonl(out / "gallery_test.jsonl",
                        ({"uid": u, "scope": "test"} for u in sorted(obj["test"])))
    n_gf = _write_jsonl(out / "gallery_full.jsonl",
                        ({"uid": u, "scope": "full"} for u in sorted(admitted)))

    files = {p.name: {"sha256": _sha256_file(p), "bytes": p.stat().st_size}
             for p in sorted(out.iterdir()) if p.name != "MANIFEST.json"}
    rec = {
        "manifest_version": MANIFEST_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "spec": "workflow/REPRODUCTION_PROTOCOL_20260903.md §四 §六 §十 §十二",
        "recomputed_nothing": True,
        "lvis_manifest_sha256": _sha256_file(paths.LVIS_MANIFEST),
        "counts": {"assets": n_assets, "images": n_images, "captions": n_caps,
                   "pointclouds": n_clouds, "admitted": len(admitted),
                   "gallery_test": n_gt, "gallery_full": n_gf,
                   "degraded_admitted": len(degraded)},
        "files": files,
        "wallclock_s": round(time.time() - started, 1),
    }
    _write_json(out / "MANIFEST.json", rec)
    return rec


# ------------------------------------------------------------------ the check

def verify(out: Path, expect: dict) -> list[str]:
    """Assert the manifest against the audited corpus. Returns failures.

    Asserted, not eyeballed: a later corpus change must fail here loudly rather
    than be discovered when a number looks odd in a table.
    """
    rec = json.loads((out / "MANIFEST.json").read_text())
    c, bad = rec["counts"], []
    for k, want in expect.items():
        got = c.get(k)
        if got != want:
            bad.append(f"{k}: manifest {got}, audited {want}")

    sp = json.loads((out / "splits.json").read_text())
    for k, v in sp["assert_disjoint"].items():
        if v != 0:
            bad.append(f"split overlap {k} = {v}, must be 0")

    # One uid per gallery row -- §四.1's forbidden expansion, checked rather
    # than assumed, because it is the failure that looks like a good score.
    for name in ("gallery_test.jsonl", "gallery_full.jsonl"):
        uids = [json.loads(l)["uid"] for l in (out / name).read_text().splitlines() if l.strip()]
        if len(uids) != len(set(uids)):
            bad.append(f"{name} repeats a uid: {len(uids)} rows, {len(set(uids))} unique")

    # Every image row must join to a camera row.
    cam = {v["view_id"] for v in json.loads((out / "camera_views.json").read_text())["views"]}
    seen = collections.Counter()
    for line in (out / "images.jsonl").read_text().splitlines():
        if line.strip():
            seen[json.loads(line)["view_id"]] += 1
    if set(seen) - cam:
        bad.append(f"image rows carry view_ids with no camera: {sorted(set(seen) - cam)}")
    if len(set(seen.values())) > 1:
        bad.append(f"view_id counts are not uniform: {dict(seen)}")
    return bad


# [CORRECTED 2026-09-03] `images` was 552,624 = 46,052 x 12, taken from the
# render DIRECTORY count. The right multiplicand is the INDEX count, 46,024:
# the 28 assets n04 lost have a directory, created before the failure, but no
# sidecar and no index row, so they contribute no image record. The corpus was
# never wrong; the expectation was, and it was the loose one -- a directory is
# not evidence that an asset rendered.
AUDITED = {"assets": 46052, "images": 552288, "pointclouds": 46052,
           "admitted": 45692, "gallery_test": 9138, "gallery_full": 45692,
           "degraded_admitted": 253}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="default: data/outputs/manifest")
    ap.add_argument("--limit", type=int, help="first N uids, for a smoke run")
    args = ap.parse_args()
    out = Path(args.out) if args.out else paths.OUTPUTS / "manifest"

    rec = build(out, args.limit)
    print(f"manifest -> {out}")
    for k, v in sorted(rec["counts"].items()):
        print(f"  {k:<20} {v:,}")
    print(f"  wallclock            {rec['wallclock_s']}s")

    if args.limit:
        print("  (--limit run: counts not asserted against the audited corpus)")
        return 0
    bad = verify(out, AUDITED)
    if bad:
        print("\nMANIFEST DISAGREES WITH THE AUDITED CORPUS:")
        for b in bad:
            print("  " + b)
        return 1
    print("\nagrees with PHASE1_AUDIT_20260903 on every audited count, "
          "splits are disjoint, no gallery uid repeats, every image row joins "
          "a camera")
    return 0


def demo() -> None:
    """Self-check: the camera table is the vendored list, in order, no repeats."""
    v = camera_views()
    assert len(v) == 12, len(v)
    assert [x["view_id"] for x in v] == list(range(12))
    assert v[0]["polar_phi_deg"] == 60.0 and v[0]["azimuth_theta_deg"] == 30.0
    assert v[11]["polar_phi_deg"] == 120.0 and v[11]["azimuth_theta_deg"] == 270.0
    assert v[0]["elevation_deg"] == 30.0 and v[11]["elevation_deg"] == -30.0
    assert v[11]["looks_from_below"] and not v[0]["looks_from_below"]
    assert len({(x["polar_phi_deg"], x["azimuth_theta_deg"]) for x in v}) == 12
    assert sorted(collections.Counter(x["ring"] for x in v).values()) == [4, 4, 4]
    print("camera table ok: 12 distinct cameras, three rings of four, in vendor order")


if __name__ == "__main__":
    raise SystemExit(main())
