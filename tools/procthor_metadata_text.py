#!/usr/bin/env python3
"""ProcTHOR node / gallery text from ProcTHOR's own asset metadata (DL-103 R6).

Paper 2.3: ProcTHOR "provides precise spatial coordinates and comprehensive semantic
metadata for each asset"; 2.5: node feature t_i is text-derived. The 2026-09-02 plan
assumed that metadata did not exist and ruled gemma captions instead (DL-077 Q10);
the ESSGNN reviewer (2026-09-06) found it shipped with the procthor package:
`asset-database.json` (bounding box, materials, objectType, properties) and
`placement-annotations.json` (onFloor / onWall / room types / pickupable). UPSTREAM
FACT, read from /home/kyzen/upstream/procthor/procthor/databases/.

Every ProcTHOR asset gets a Figure-2-shaped record built from that metadata, so the
Stage 2 gallery text and the node text t_i are the same construction as the
Objaverse gallery text (`figure2_json`):

    category   objectType split into words                              UPSTREAM
    synset     WordNet lookup of the category (annotate.resolve_synset) IMPLEMENTATION CHOICE
    width / length / height   boundingBox x / z / y in cm               UPSTREAM (metres x 100)
    volume     width x length x height                                  Figure 2 arithmetic
    mass       not in the metadata -> null                              UNKNOWN
    description  gemma caption of the unified renders when available   DL-077 Q10 (R6b), else null
    materials  material words recognised in the Unity material names   IMPLEMENTATION CHOICE
    onCeiling  false (no ProcTHOR asset hangs from a ceiling)           INFERENCE
    onWall     placement-annotations onWall                             UPSTREAM
    onFloor    placement-annotations onFloor                            UPSTREAM
    onObject   not onFloor and not onWall (it sits on a receptacle)     INFERENCE

Writes `outputs/procthor_asset_annotations.json` and rewrites
`outputs/procthor_object_text.json` with, per assetId, `text` (the figure2_json string,
what the text tower encodes) and `relation_text` (one TYPE-level sentence -- category and placement -- what the
relation-sentence LLM prompt sees; instance details would multiply the distinct pairs). `--captions <json>` merges gemma descriptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from metafind import paths
from metafind.data.annotate import resolve_synset
from metafind.models.resolve_stage1 import figure2_json_string

DB = Path("/home/kyzen/upstream/procthor/procthor/databases")
MATERIAL_WORDS = ("metal", "wood", "glass", "plastic", "fabric", "cloth", "marble", "ceramic",
                  "leather", "paper", "stone", "rubber", "concrete", "steel", "chrome", "brick",
                  "cardboard", "foam", "porcelain", "granite", "tile", "wicker", "cotton", "wool")
SYNONYM = {"steel": "metal", "chrome": "metal", "cloth": "fabric", "cotton": "fabric", "wool": "fabric",
           "porcelain": "ceramic", "tile": "ceramic", "granite": "stone", "wicker": "wood"}
SOURCE_VERSION = 1


def split_camel(name: str) -> str:
    """AlarmClock -> alarm clock; TVStand -> tv stand; Countertop_I -> countertop i."""
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    return " ".join(w.lower() for w in s.replace("_", " ").split())


def synset_for(category: str) -> tuple[str, str]:
    """WordNet id for the category; the last word, then a generic noun, when WordNet has no entry."""
    from metafind.data.annotate import AnnotationError
    for cand, tag in ((category, "wordnet"), (category.split()[-1], "wordnet_last_word")):
        try:
            synset, src = resolve_synset(cand, None)
            return synset, f"{src}:{tag}"
        except AnnotationError:
            continue
    return "object.n.01", "fallback_generic"


def materials_from(names: list) -> list[str]:
    out: list[str] = []
    for m in names or []:
        s = (m[1] if isinstance(m, list) else str(m)).lower()
        for w in MATERIAL_WORDS:
            if w in s:
                w = SYNONYM.get(w, w)
                if w not in out:
                    out.append(w)
    return out[:6]


def build_records(asset_ids, captions: dict | None = None) -> dict[str, dict]:
    ad = json.loads((DB / "asset-database.json").read_text())
    pa = json.loads((DB / "placement-annotations.json").read_text())
    by_id = {e["assetId"]: e for lst in ad.values() for e in lst}
    out = {}
    for aid in sorted(asset_ids):
        e = by_id[aid]
        t = e["objectType"]
        category = split_camel(t)
        bb = e["boundingBox"]
        width, length, height = (round(bb["x"] * 100, 1), round(bb["z"] * 100, 1), round(bb["y"] * 100, 1))
        on_floor = bool(pa["onFloor"].get(t, False))
        on_wall = bool(pa["onWall"].get(t, False))
        synset, synset_source = synset_for(category)
        rec = {
            "category": category, "synset": synset, "synset_source": synset_source,
            "width": width, "length": length, "height": height, "volume": round(width * length * height, 1),
            "mass": None,
            "description": (captions or {}).get(aid),
            "materials": materials_from(e.get("materials")),
            "onCeiling": False, "onWall": on_wall, "onFloor": on_floor,
            "onObject": (not on_floor) and (not on_wall),
            "objectType": t, "primaryProperty": e.get("primaryProperty"),
            "room_counts": {k: pa[k].get(t) for k in ("inKitchens", "inLivingRooms", "inBedrooms", "inBathrooms")},
            "isPickupable": pa["isPickupable"].get(t),
            "source": f"procthor_metadata_v{SOURCE_VERSION}",
        }
        out[aid] = rec
    return out


def relation_sentence(rec: dict) -> str:
    """TYPE-level, on purpose: the paper's semantic edges relate object kinds
    ("microscope-lab bench"), and n08 asks the LLM once per distinct sentence pair. A
    per-asset sentence (dimensions, caption) would turn ~4K distinct pairs into hundreds
    of thousands of LLM calls for relations that do not depend on the instance."""
    place = ("on the wall" if rec["onWall"] else "on the floor" if rec["onFloor"]
             else "on top of other objects")
    return f"a {rec['category']}, typically placed {place}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--captions", type=Path, default=None, help="assetId -> description JSON (gemma, R6b)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    text_path = paths.OUTPUTS / "procthor_object_text.json"
    old = json.loads(text_path.read_text())
    captions = json.loads(args.captions.read_text()) if args.captions else None
    recs = build_records(old.keys(), captions)
    db_sha = hashlib.sha256((DB / "asset-database.json").read_bytes() + (DB / "placement-annotations.json").read_bytes()).hexdigest()[:16]
    new_map = {aid: {"text": figure2_json_string(r), "relation_text": relation_sentence(r),
                     "source": f"procthor_metadata_v{SOURCE_VERSION}@{db_sha}"
                               + ("+gemma_caption" if r.get("description") else "")}
               for aid, r in recs.items()}
    n_desc = sum(1 for r in recs.values() if r.get("description"))
    n_mat = sum(1 for r in recs.values() if r["materials"])
    print(f"{len(recs):,} assets; {n_mat:,} with recognised materials; {n_desc:,} with captions; "
          f"e.g. {next(iter(new_map.values()))['text'][:160]}")
    if args.dry_run:
        return 0
    (paths.OUTPUTS / "procthor_asset_annotations.json").write_text(json.dumps(recs, indent=1, ensure_ascii=False))
    backup = text_path.with_name(f"procthor_object_text.before_metadata_v{SOURCE_VERSION}.json")
    if not backup.exists():
        backup.write_text(json.dumps(old, indent=1, ensure_ascii=False))
    text_path.write_text(json.dumps(new_map, indent=1, ensure_ascii=False))
    print(f"-> {text_path} (previous map kept at {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
