#!/usr/bin/env python3
"""Resolve each paper's arXiv TeX source into a dependency tree and manifest.

    python3 tools/build_source_manifest.py

The point is to know WHICH .tex files the compiled document actually contains.
An archive is not a document: EGNN's ships a complete duplicate of itself in a
subdirectory, and a formula census that walked every .tex on disk would double
every equation in that paper and report a count nobody could reconcile.

So `\\input` / `\\include` / `\\subfile` are followed from the main entrypoint and
nothing else is trusted. Files that exist but are never reached are listed under
`orphans` -- kept visible rather than deleted, because "unreferenced" is a claim
about the include graph, not about the authors' intent.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "docs" / "paper"

# arXiv ids: from the archives' own metadata where present, else the canonical
# published identifier. Recorded so a later reader can re-download and compare.
PAPERS = {
    "metafind": {"dir": "metafind_source", "archive": "MetaFind.gz",
                 "main": "neurips_2025.tex", "arxiv_id": "2510.05057"},
    "ulip2":    {"dir": "ulip2_source", "archive": "Ulip2.gz",
                 "main": "main.tex", "arxiv_id": "2305.08275"},
    "egnn":     {"dir": "egnn_source", "archive": "EGNN.gz",
                 "main": "example_paper.tex", "arxiv_id": "2102.09844"},
    "idesign":  {"dir": "idesign_source", "archive": "I-Design.gz",
                 "main": "main.tex", "arxiv_id": "2404.02838"},
}

INCLUDE = re.compile(r"\\(?:input|include|subfile)\s*\{([^}]+)\}")


def resolve(root: Path, ref: str) -> Path | None:
    ref = ref.strip()
    for cand in (root / ref, root / (ref + ".tex")):
        if cand.is_file():
            return cand
    return None


def walk(root: Path, main: Path) -> tuple[list[str], dict]:
    """Depth-first over the include graph. Returns reached files and the tree."""
    order, tree, seen = [], {}, set()

    def visit(p: Path) -> dict:
        rel = str(p.relative_to(root))
        if rel in seen:                       # a cycle, or a file included twice
            return {"file": rel, "repeat": True}
        seen.add(rel)
        order.append(rel)
        kids = []
        for m in INCLUDE.finditer(p.read_text(errors="replace")):
            child = resolve(root, m.group(1))
            if child is not None:
                kids.append(visit(child))
            else:
                kids.append({"missing": m.group(1)})
        node = {"file": rel}
        if kids:
            node["includes"] = kids
        return node

    tree = visit(main)
    return order, tree


def main() -> int:
    out = {}
    for name, meta in PAPERS.items():
        root = PAPER / meta["dir"]
        archive = PAPER / meta["archive"]
        main_tex = root / meta["main"]
        if not main_tex.is_file():
            print(f"!! {name}: {meta['main']} not found")
            continue

        order, tree = walk(root, main_tex)
        every = sorted(str(p.relative_to(root))
                       for p in root.rglob("*.tex") if p.is_file())
        orphans = [f for f in every if f not in order]

        # Appendix / supplement are named by convention, not by structure, so
        # they are TAGGED, never inferred -- a file called `appendix.tex` that
        # main.tex never includes is an orphan, and saying otherwise would let
        # an unpublished draft become evidence.
        def tagged(kind: str) -> list[str]:
            return [f for f in order if kind in Path(f).name.lower()]

        manifest = {
            "paper_name": name,
            "arxiv_id": meta["arxiv_id"],
            "archive_path": str(archive.relative_to(PAPER.parents[1])),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "main_tex": meta["main"],
            "included_tex_files": order,
            "include_tree": tree,
            "appendix_files": tagged("appendix"),
            "supplement_files": tagged("suppl"),
            "macro_files": [f for f in order if "preamble" in f or "macro" in f],
            "orphan_tex_files": orphans,
            "tex_sha256": {f: hashlib.sha256((root / f).read_bytes()).hexdigest()
                           for f in order},
        }
        (root / "SOURCE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=1, ensure_ascii=False))
        out[name] = manifest

        print(f"{name:9} main={meta['main']:20} reached={len(order):2}  "
              f"orphan={len(orphans):2}  appendix={len(manifest['appendix_files'])}")
        for f in order:
            print(f"           + {f}")
        if orphans:
            print(f"           orphans (present, never included): {orphans}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
