#!/usr/bin/env python3
"""Extract every formula from the authors' arXiv TeX source.

    python3 tools/build_source_manifest.py     # first: resolve the include tree
    python3 tools/build_formula_inventory.py   # then: this

This replaces an earlier version that read converted Markdown. That approach was
abandoned, not improved: the PDF-to-Markdown converter interpreted LaTeX
backslash sequences as C string escapes, so `\\frac` arrived as a form feed plus
"rac" and `\\neq` as a REAL newline -- which is a legal character, so a control-
byte census skipped it and the whole `\\n`-prefixed command class stayed broken
through two rounds of "repaired". No amount of repair makes a lossy conversion
authoritative. The TeX is what the authors wrote.

Only files the main document actually includes are read, per SOURCE_MANIFEST.
EGNN's archive ships a complete duplicate of itself in a subdirectory; walking
every .tex on disk would double that paper's equation count.

Equation numbers are ASSIGNED HERE by counting numbered environments in document
order, because TeX does not store them -- LaTeX computes them at compile time.
They are therefore OURS, and any disagreement with the published PDF is a bug in
this counter, not in the paper. `\\label`s are extracted separately and are the
authors' own; prefer them when citing.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "paper"
AUDIT = ROOT / "docs" / "audit"

PAPERS = {"metafind": "MetaFind", "ulip2": "ULIP-2",
          "egnn": "EGNN", "idesign": "I-Design"}
SHORT = {"metafind": "MF", "ulip2": "U2", "egnn": "EG", "idesign": "ID"}

# Numbered environments produce an equation number; starred ones do not.
NUMBERED = ("equation", "align", "gather", "multline", "eqnarray", "flalign")
STARRED = tuple(e + r"\*" for e in NUMBERED)
ENVS = NUMBERED + STARRED + ("displaymath",)

ENV_RE = re.compile(
    r"\\begin\{(" + "|".join(ENVS) + r")\}(.*?)\\end\{\1\}", re.S)
BRACKET_RE = re.compile(r"(?<!\\)\\\[(.*?)(?<!\\)\\\]", re.S)
# Plain TeX display math. Only EGNN's appendix uses it -- six blocks, including
# the EGCL equivariance target and the velocity-variant statements -- and an
# earlier version of this file did not scan for it at all. The census then
# reported `failures: []`, which proved only that the formulas it HAD found were
# uncorrupted; it said nothing about the ones it never looked for. Completeness
# and integrity are different claims and the validation now makes both.
DOLLAR_RE = re.compile(r"(?<![\\$])\$\$(.+?)\$\$", re.S)
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
SECTION_RE = re.compile(r"\\(sub)*section\*?\{([^}]*)\}")


def strip_comments(tex: str) -> str:
    """Drop TeX comments so a commented-out equation never enters the inventory.

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


def sections_before(tex: str, pos: int) -> str:
    last = ""
    for m in SECTION_RE.finditer(tex):
        if m.start() < pos:
            last = m.group(2)
        else:
            break
    return last


def rows_matching(clean: str, raw: str, rel: str, short: str,
                  counter: list[int]) -> list[dict]:
    found = []
    for m in ENV_RE.finditer(clean):
        found.append((m.start(), m.group(1), m.group(2)))
    for m in BRACKET_RE.finditer(clean):
        found.append((m.start(), r"\[\]", m.group(1)))
    for m in DOLLAR_RE.finditer(clean):
        found.append((m.start(), "$$", m.group(1)))
    found.sort()

    rows = []
    for start, env, body in found:
        numbered = env in NUMBERED
        base = env.rstrip(r"\*")
        # A multi-row `align` takes one number PER ROW. Splitting matters twice
        # over: counting the block as one would desynchronise every later
        # equation in the paper, and MetaFind's Eq. (2) and Eq. (3) -- the two
        # ESSGNN update rules, which differ in ways the whole audit turns on --
        # live in a single align. Each row is stored separately and is still a
        # literal substring of the source, so exactness is not traded away.
        splittable = base in ("align", "gather", "eqnarray", "flalign")
        parts = ([p for p in re.split(r"(?<!\\)\\\\", body) if p.strip()]
                 if splittable else [body])

        parent = hashlib.sha256(body.encode()).hexdigest() if len(parts) > 1 else None
        for part in parts:
            num = None
            if numbered:
                counter[0] += 1
                num = counter[0]
            rows.append({
                "paper": short,
                "source_tex": rel,
                "source_line": clean[:start].count("\n") + 1,
                "section": sections_before(clean, start),
                "environment": env,
                "equation_label": LABEL_RE.findall(part) or None,
                "equation_number": num,
                "numbered": numbered,
                "exact_latex": part,
                "sha256": hashlib.sha256(part.encode()).hexdigest(),
                "parent_environment_sha256": parent,
            })
    return rows


def inline_formulas(clean: str) -> list[str]:
    """Inline math, for the symbol/definition checks -- not part of the census."""
    out = re.findall(r"(?<!\\)\\\((.+?)(?<!\\)\\\)", clean, re.S)
    out += re.findall(r"(?<![\\$])\$([^$\n]+?)\$(?!\$)", clean)
    return out


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    all_rows, inline, failures, summary = [], {}, [], {}

    for name, title in PAPERS.items():
        src = PAPER / f"{name}_source"
        manifest_path = src / "SOURCE_MANIFEST.json"
        if not manifest_path.is_file():
            failures.append(f"{name}: no SOURCE_MANIFEST.json -- run "
                            "tools/build_source_manifest.py first")
            continue
        manifest = json.loads(manifest_path.read_text())

        counter = [0]
        rows, inl = [], []
        for rel in manifest["included_tex_files"]:
            raw = (src / rel).read_text(errors="replace")
            clean = strip_comments(raw)
            rows += rows_matching(clean, raw, rel, SHORT[name], counter)
            inl += inline_formulas(clean)

        for i, r in enumerate(rows):
            base = f"{SHORT[name]}-{r['equation_number']}" \
                if r["equation_number"] else f"{SHORT[name]}-U{i}"
            r["id"] = base
        all_rows += rows
        inline[name] = inl
        summary[name] = {
            "title": title, "arxiv_id": manifest["arxiv_id"],
            "archive_sha256": manifest["archive_sha256"],
            "main_tex": manifest["main_tex"],
            "tex_files_read": len(manifest["included_tex_files"]),
            "orphan_tex_ignored": len(manifest["orphan_tex_files"]),
            "display_formulas": len(rows),
            "numbered": sum(1 for r in rows if r["numbered"]),
            "unnumbered": sum(1 for r in rows if not r["numbered"]),
            "inline_formulas": len(inl),
        }

    # ---- validation: the inventory must be a faithful copy --------------------
    for r in all_rows:
        src = PAPER / f"{[k for k, v in SHORT.items() if v == r['paper']][0]}_source"
        text = strip_comments((src / r["source_tex"]).read_text(errors="replace"))
        if r["exact_latex"] not in text:
            failures.append(f"{r['id']}: exact_latex is not a substring of its source")
        if hashlib.sha256(r["exact_latex"].encode()).hexdigest() != r["sha256"]:
            failures.append(f"{r['id']}: sha256 does not match exact_latex")
        if "\u2026" in r["exact_latex"] or "..." in r["exact_latex"]:
            # `\dots` and `\cdots` are legitimate LaTeX and are NOT this check;
            # a literal ellipsis character means a serializer truncated us.
            failures.append(f"{r['id']}: literal ellipsis inside exact_latex")

    for fid, n in _counter(r["id"] for r in all_rows).items():
        if n > 1:
            failures.append(f"formula id {fid} is not unique ({n} rows)")

    RECORDS = AUDIT / "formula_inventory.json"
    RECORDS.write_text(json.dumps(
        {"summary": summary, "formulas": all_rows}, indent=1, ensure_ascii=False))

    # round trip: read back from disk and re-hash
    back = json.loads(RECORDS.read_text())["formulas"]
    for a, b in zip(all_rows, back):
        if hashlib.sha256(b["exact_latex"].encode()).hexdigest() != a["sha256"]:
            failures.append(f"{a['id']}: sha256 changed through JSON round trip")

    write_markdown(all_rows, summary)
    md = (AUDIT / "A_FORMULA_INVENTORY.md").read_text()
    for r in all_rows:
        if r["exact_latex"].strip() not in md:
            failures.append(f"{r['id']}: exact_latex absent from A_FORMULA_INVENTORY.md")
    if "\u2026" in md:
        failures.append("A_FORMULA_INVENTORY.md contains a literal ellipsis")

    (AUDIT / "formula_inventory_validation.json").write_text(json.dumps({
        "summary": summary, "formulas_total": len(all_rows),
        "checks": ["exact_latex is a substring of its source TeX",
                   "sha256 matches exact_latex",
                   "no literal ellipsis in exact_latex",
                   "formula ids are unique",
                   "sha256 survives the JSON round trip",
                   "every exact_latex appears verbatim in the markdown",
                   "no literal ellipsis in the markdown"],
        "failures": failures}, indent=1))

    for name, s in summary.items():
        print(f"{name:9} {s['display_formulas']:3} display "
              f"({s['numbered']} numbered, {s['unnumbered']} unnumbered)  "
              f"{s['inline_formulas']:4} inline  "
              f"from {s['tex_files_read']} tex, {s['orphan_tex_ignored']} orphans ignored")
    print(f"\n{len(all_rows)} display formulas total")
    if failures:
        print(f"\n{len(failures)} FAILURES")
        for f in failures[:40]:
            print("  " + f)
        return 1
    print("round-trip, sha256, uniqueness and no-ellipsis all pass")
    return 0


def _counter(it):
    d = {}
    for x in it:
        d[x] = d.get(x, 0) + 1
    return d


def write_markdown(rows: list[dict], summary: dict) -> None:
    L = ["# A. FORMULA_INVENTORY", "",
         "Every display formula in the four papers, taken from the authors' arXiv",
         "TeX source. Generated by `tools/build_formula_inventory.py`; do not edit",
         "by hand -- the validation compares this file against the TeX and fails.", "",
         "## Authority", "",
         "**arXiv TeX source > published PDF > converted Markdown.** The Markdown",
         "under `docs/paper/*.md` is a convenience copy for prose search and is NOT",
         "authoritative for any formula, dimension, symbol or equation number.", "",
         "| paper | arXiv | main tex | files read | orphans ignored | display | inline |",
         "|---|---|---|---|---|---|---|"]
    for name, s in summary.items():
        L.append(f"| {s['title']} | `{s['arxiv_id']}` | `{s['main_tex']}` | "
                 f"{s['tex_files_read']} | {s['orphan_tex_ignored']} | "
                 f"{s['display_formulas']} | {s['inline_formulas']} |")
    L += ["",
          "Orphans are `.tex` files present in an archive that the main document",
          "never includes. EGNN's archive ships a complete duplicate of itself in a",
          "subdirectory; reading every file on disk would double its equation count.",
          "", "## Equation numbers", "",
          "TeX does not store equation numbers -- LaTeX computes them at compile",
          "time. The numbers here are **assigned by this tool** by counting numbered",
          "environments in document order, with each row of a multi-row `align`",
          "taking its own number. They are ours; a disagreement with the published",
          "PDF is a bug in the counter, not in the paper. `equation_label` holds the",
          "authors' own `\\label`s and is the stable way to cite.", "",
          "## What is validated", "",
          "1. every `exact_latex` is a literal substring of its source `.tex`",
          "2. SHA256 matches `exact_latex`, and survives the JSON round trip",
          "3. no literal ellipsis character inside any formula (`\\dots` is fine --",
          "   the check is for a serializer having truncated us)",
          "4. formula ids are unique",
          "5. every formula appears verbatim in this markdown", "",
          "Formulas live in fenced blocks, never in table cells: a pipe ends a cell",
          "and a width cap inserts an ellipsis.", ""]

    for name, s in summary.items():
        short = SHORT[name]
        sub = [r for r in rows if r["paper"] == short]
        L += [f"## {s['title']} ({len(sub)})", ""]
        cur = None
        for r in sub:
            if r["section"] != cur:
                cur = r["section"]
                L += [f"### {cur or '(front matter)'}", ""]
            num = f"({r['equation_number']})" if r["equation_number"] else "unnumbered"
            L += [f"**`{r['id']}`** — {num} — `{r['source_tex']}` line {r['source_line']}"
                  f" — `{r['environment']}`"]
            if r["equation_label"]:
                L.append(f"- label: " + ", ".join(f"`{x}`" for x in r["equation_label"]))
            L += [f"- sha256: `{r['sha256'][:16]}`", "",
                  "```latex", r["exact_latex"].strip(), "```", ""]
    (AUDIT / "A_FORMULA_INVENTORY.md").write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
