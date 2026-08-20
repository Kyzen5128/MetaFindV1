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
import tarfile
import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "docs" / "paper"

# arXiv ids: from the archives' own metadata where present, else the canonical
# published identifier. Recorded so a later reader can re-download and compare.
PAPERS = {
    "metafind": {"dir": "metafind_source", "archive": "MetaFind.gz",
                 "main": "neurips_2025.tex", "arxiv_id": "2510.04057"},
    "ulip2":    {"dir": "ulip2_source", "archive": "Ulip2.gz",
                 "main": "main.tex", "arxiv_id": "2305.08275"},
    "egnn":     {"dir": "egnn_source", "archive": "EGNN.gz",
                 "main": "example_paper.tex", "arxiv_id": "2102.09844"},
    "idesign":  {"dir": "idesign_source", "archive": "I-Design.gz",
                 "main": "main.tex", "arxiv_id": "2404.02838"},
}

INCLUDE = re.compile(r"\\(?:input|include|subfile)\s*\{([^}]+)\}")


def strip_comments(tex: str) -> str:
    """Drop TeX comments BEFORE walking the include tree.

    Without this, a commented-out `%\\input{sections/template}` counts as part of
    the document. EGNN comments out two includes and ULIP-2 one, so the manifest
    claimed 10 and 4 reached files against a true 8 and 3 -- and `X_suppl.tex`
    was additionally classified as a supplement the paper does not contain.

    The formula extractor already did this, which is why three commented-out
    EGNN equations were correctly excluded from the census. Only the manifest
    was missing it, so the two tools disagreed about what the document IS.

    A percent sign preceded by a backslash is a literal, not a comment.
    """
    out = []
    for line in tex.split("\n"):
        i, esc = 0, False
        while i < len(line):
            if line[i] == "\\":
                esc = not esc
            elif line[i] == "%" and not esc:
                line = line[:i]
                break
            else:
                esc = False
            i += 1
        out.append(line)
    return "\n".join(out)


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
        for m in INCLUDE.finditer(strip_comments(p.read_text(errors="replace"))):
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


GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def figures(root: Path, archive: Path, order: list[str]) -> dict:
    """Which images the TeX references, and whether we actually have them.

    This exists because we did not have them, and nothing said so. The manifest
    recorded only `.tex` hashes, the extraction pulled only `.tex`, and six
    figures sat unread inside the archive for the whole project. One of them --
    MetaFind's `data-preprocess.png` -- prints the annotation schema, so n05 was
    built against an invented schema while the paper's own was on disk. Every
    audit inherited the same blind spot, because "we read the paper" was true of
    the text and false of the figures.

    A figure named by the TeX and missing from disk is now a recorded fact.
    """
    referenced = sorted({m for f in order
                         for m in GRAPHIC.findall(strip_comments((root / f).read_text()))})
    in_archive: list[str] = []
    try:
        with tarfile.open(archive, "r:gz") as tf:
            in_archive = sorted(n for n in tf.getnames()
                                if Path(n).suffix.lower() in IMAGE_SUFFIXES)
    except (tarfile.TarError, OSError):
        pass  # single-file gzip or unreadable: the on-disk check below still runs

    def present(ref: str) -> str | None:
        # TeX may omit the extension; graphicx resolves it.
        for cand in ([root / ref] if Path(ref).suffix else
                     [root / (ref + s) for s in sorted(IMAGE_SUFFIXES)]):
            if cand.is_file():
                return str(cand.relative_to(root))
        return None

    found = {ref: present(ref) for ref in referenced}
    missing = sorted(r for r, p in found.items() if p is None)
    return {
        "referenced_figures": referenced,
        "figure_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                          for p in sorted(v for v in found.values() if v)},
        "images_in_archive": in_archive,
        # Non-empty means the paper cannot be read in full from this checkout.
        "missing_figures": missing,
    }


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"}


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
            **figures(root, archive, order),
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
