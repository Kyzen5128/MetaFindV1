#!/usr/bin/env python3
"""Repair the source papers and build A_FORMULA_INVENTORY, with round-trip proof.

    python3 tools/build_formula_inventory.py

Two jobs, and the second is the one that kept going wrong.

REPAIR. `ulip2_paper.md` and `egnn_paper.md` were written through a layer that
interpreted LaTeX backslash sequences as C string escapes: `\\frac` became a form
feed followed by "rac", `\\rangle` a carriage return followed by "angle". Every
non-newline control byte is therefore a swallowed backslash, whatever follows
it, so the repair does not need -- and must not use -- a list of command names.
Guessing names is what left 11 `\\rangle` broken on the first attempt.

`\\n` is the exception and it is why `\\neq` survived two rounds of "fixed": it
became a REAL newline, which is also a legal character. It can only be
disambiguated inside math, by whether the following letters spell a LaTeX
command beginning with n.

SERIALISE. The inventory must contain each formula EXACTLY. Two things broke
that before: putting LaTeX inside markdown table cells (pipes, and a width cap
that inserted an ellipsis) and normalising whitespace on the way in. Formulas now
live in fenced blocks; the table carries only identifiers.

Nothing here is trusted. Every formula is hashed at the source and re-hashed
after being written and read back, and the two must match.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"
REPAIRED = DOCS / "audit" / "repaired"
INVENTORY = DOCS / "audit" / "A_FORMULA_INVENTORY.md"
RECORDS = DOCS / "audit" / "A_FORMULA_INVENTORY.json"

PAPERS = (("metafind_paper.md", "MF", "MetaFind"),
          ("ulip2_paper.md", "U2", "ULIP-2"),
          ("egnn_paper.md", "EG", "EGNN"))

# The seven C string escapes. LF is handled separately; the rest are unambiguous.
ESC = {0x07: "a", 0x08: "b", 0x09: "t", 0x0b: "v", 0x0c: "f", 0x0d: "r"}

# LaTeX commands starting with n. An LF inside math followed by one of these is
# a swallowed `\n`, not a line break.
NCMDS = ("subseteq", "supseteq", "onumber", "ewline", "otin", "abla", "orm",
         "eq", "i", "u", "e")

FENCE = "```"


def repair(name: str) -> tuple[str, dict]:
    raw = (DOCS / name).read_bytes()
    out, fixed = bytearray(), collections.Counter()
    for b in raw:
        if b in ESC:
            out += b"\\" + ESC[b].encode()
            fixed[ESC[b]] += 1
        else:
            out.append(b)
    text = out.decode("utf8")

    def fix_n(m: re.Match) -> str:
        body = m.group(0)
        for cmd in NCMDS:                       # longest first, see NCMDS order
            body, k = re.subn("\n(" + cmd + ")(?![a-zA-Z])", "\\\\n\\1", body)
            fixed["n"] += k
        return body

    text = re.sub(r"\$\$.+?\$\$", fix_n, text, flags=re.S)
    text = re.sub(r"(?<!\$)\$[^$]+?\$(?!\$)", fix_n, text)
    return text, dict(fixed)


def headings(text: str) -> list[tuple[int, str]]:
    return [(m.start(), m.group(1).strip())
            for m in re.finditer(r"(?m)^#+\s*(.+)$", text)]


def section_key(section: str) -> str:
    """A short tag to tell same-numbered equations apart.

    The EGNN paper restarts its numbering in the appendices, so `(14)` names
    both the E(n)-equivariance statement in 3.1 and an inner-product identity in
    E.2. Keying on the number alone silently merged them -- and the merge is not
    cosmetic, it gave E.2's proof steps the CONCEPTUAL_SOURCE label belonging to
    the equivariance result.
    """
    toks = section.replace(":", " ").split()
    if not toks:
        return "?"
    return (toks[0] + toks[1]) if toks[0] == "Appendix" else toks[0]


def extract(text: str, short: str) -> list[dict]:
    """Every display equation, VERBATIM.

    The only transformation is reading the trailing `\\quad (n)` into its own
    field; `raw_span` still holds the whole span exactly as it appears, so the
    round-trip compares against the untouched source, not against something this
    function decided was equivalent.
    """
    heads, rows = headings(text), []
    for m in re.finditer(r"\$\$(.+?)\$\$", text, re.S):
        body = m.group(1)
        num = re.search(r"\\quad\s*\((\d+[a-z]?)\)\s*$", body.strip())
        section = ""
        for off, txt in heads:
            if off < m.start():
                section = txt
            else:
                break
        rows.append({
            "id": f"{short}-{num.group(1) if num else 'U' + str(len(rows))}",
            "source": short,
            "line": text[:m.start()].count("\n") + 1,
            "section": section,
            "equation": num.group(1) if num else None,
            "raw_span": body,                     # exactly as it appears
            "sha256": hashlib.sha256(body.encode()).hexdigest(),
        })

    dupes = {i for i, n in collections.Counter(r["id"] for r in rows).items() if n > 1}
    for r in rows:
        if r["id"] in dupes:
            r["id"] += "@" + section_key(r["section"])
    return rows


def bare_commands(blob: str) -> dict:
    cmds = ("rangle", "langle", "right", "left", "tilde", "text", "times",
            "frac", "tau", "approx", "alpha", "theta", "neq", "nabla",
            "operatorname", "mathcal", "mathbb", "phi", "sum", "exp", "log")
    found = {}
    for c in cmds:
        n = len(re.findall(r"(?<![\\A-Za-z{])" + c + r"(?![A-Za-z}])", blob))
        if n:
            found[c] = n
    return found


def main() -> int:
    REPAIRED.mkdir(parents=True, exist_ok=True)
    repair_report, all_rows, failures = {}, [], []

    for name, short, _ in PAPERS:
        text, fixed = repair(name)
        (REPAIRED / name).write_text(text, newline="")

        # (8a) no control byte may survive
        left = [b for b in (REPAIRED / name).read_bytes() if b in ESC]
        if left:
            failures.append(f"{name}: {len(left)} control bytes survived repair")

        blob = "\n".join(re.findall(r"\$\$.+?\$\$", text, re.S)
                         + re.findall(r"(?<!\$)\$[^$\n]+?\$(?!\$)", text))
        bare = bare_commands(blob)
        if bare:
            failures.append(f"{name}: bare commands inside math {bare}")
        for label, ok in (("$$", text.count("$$") % 2 == 0),
                          ("{}", blob.count("{") == blob.count("}")),
                          ("left/right", blob.count("\\left") == blob.count("\\right")),
                          ("langle/rangle", blob.count("\\langle") == blob.count("\\rangle"))):
            if not ok:
                failures.append(f"{name}: unbalanced {label}")

        rows = extract(text, short)
        all_rows += rows
        repair_report[name] = {
            "repaired": fixed, "total": sum(fixed.values()),
            "control_bytes_left": len(left), "display_blocks": len(rows),
            "neq_in_source": text.count("\\" + "neq"),
            "bare_in_math": bare,
        }

    RECORDS.write_text(json.dumps(all_rows, indent=1, ensure_ascii=False))
    write_inventory(all_rows, repair_report)

    # ---- (6)(7)(8) round trip: read the inventory back and re-hash ----------
    written = json.loads(RECORDS.read_text())
    md = INVENTORY.read_text()
    for row, back in zip(all_rows, written):
        if back["sha256"] != row["sha256"]:
            failures.append(f"{row['id']}: sha256 changed through JSON")
        if row["raw_span"] not in md:
            failures.append(f"{row['id']}: exact formula absent from the markdown")
        if hashlib.sha256(back["raw_span"].encode()).hexdigest() != row["sha256"]:
            failures.append(f"{row['id']}: re-hash of the stored span differs")

    if "\u2026" in md:
        failures.append("the inventory contains an ellipsis -- a formula was truncated")
    for name, short, _ in PAPERS:
        src_neq = repair_report[name]["neq_in_source"]
        inv_neq = sum(r["raw_span"].count("\\" + "neq")
                      for r in all_rows if r["source"] == short)
        # only display math reaches the inventory, so source >= inventory
        if inv_neq > src_neq:
            failures.append(f"{name}: neq count grew {src_neq} -> {inv_neq}")
    inv_bare = bare_commands("\n".join(r["raw_span"] for r in all_rows))
    if inv_bare:
        failures.append(f"inventory exact_formula holds bare commands {inv_bare}")
    # ids address formulas in B/C/D; a collision silently merges two of them
    for fid, n in collections.Counter(r["id"] for r in all_rows).items():
        if n > 1:
            failures.append(f"formula id {fid} is not unique ({n} formulas)")

    (DOCS / "audit" / "repair_report.json").write_text(
        json.dumps(repair_report, indent=1))
    # (9) the sanity record covers the inventory too, not only the sources
    (DOCS / "audit" / "latex_sanity.json").write_text(json.dumps({
        "sources": {n: {"bare_in_math": repair_report[n]["bare_in_math"],
                        "control_bytes_left": repair_report[n]["control_bytes_left"],
                        "neq": repair_report[n]["neq_in_source"]}
                    for n, _, _ in PAPERS},
        "inventory": {
            "formulas": len(all_rows),
            "bare_in_exact_formula": inv_bare,
            "ellipsis_in_markdown": "…" in md,
            "sha256_mismatches": sum(1 for f in failures if "sha256" in f),
            "neq_in_exact_formula": sum(r["raw_span"].count("\\" + "neq")
                                        for r in all_rows),
        },
        "failures": failures,
    }, indent=1))

    print(f"{len(all_rows)} display equations across {len(PAPERS)} papers")
    for name, short, _ in PAPERS:
        r = repair_report[name]
        print(f"  {name:22} repaired {r['total']:3d}  blocks {r['display_blocks']:2d}  "
              f"neq {r['neq_in_source']:2d}  control-left {r['control_bytes_left']}")
    if failures:
        print(f"\n{len(failures)} FAILURES")
        for f in failures:
            print("  " + f)
        return 1
    print("\nround-trip, sha256, no-ellipsis and latex-sanity all pass")
    return 0


def write_inventory(rows: list[dict], report: dict) -> None:
    REL = {**{f"MF-{e}": "DIRECTLY_USED" for e in
              ("1", "2", "3", "4", "5", "6", "7a", "7b", "8", "9", "10",
               "11", "12", "13", "14", "15")},
           **{f"U2-{e}": "CONCEPTUAL_SOURCE" for e in ("1", "2", "3")},
           "EG-1": "CONCEPTUAL_SOURCE", "EG-2": "CONCEPTUAL_SOURCE",
           "EG-3": "MODIFIED", "EG-4": "MODIFIED", "EG-5": "MODIFIED",
           "EG-6": "MODIFIED", "EG-11": "CONCEPTUAL_SOURCE",
           "EG-14@3.1": "CONCEPTUAL_SOURCE", "EG-15@AppendixA": "CONCEPTUAL_SOURCE"}
    NOTE = {"EG-7a": "velocity update; MetaFind has no velocity channel",
            "EG-7b": "velocity update",
            "EG-8": "edge inference; unrelated to MetaFind's LLM e_ij",
            "EG-9": "edge inference",
            "EG-12@AppendixA": "proof step", "EG-12@B.1": "velocity-variant proof step",
            "EG-14@E.2": "E.2 inner-product identity, NOT the equivariance result",
            "EG-15@E.2": "E.2 inner-product identity",
            "EG-13": "E.1 distance-norm invariance",
            "EG-16": "graph autoencoder / QM9 head",
            "U2-1": "L_P2I, symmetric point<->image; MetaFind Stage 1 aligns fused towers",
            "U2-2": "L_P2T, symmetric point<->text; same",
            "U2-3": "min over E_P only; MetaFind trains both towers"}
    UNNUM = {"h_i^{(0)}": "**unnumbered, load-bearing**: source of contradiction C3",
             "e_{\\text{layout}}": "**unnumbered**: pooling type unspecified, see C"}

    def rel(r):
        if r["equation"]:
            return REL.get(r["id"], "NOT_USED" if r["source"] == "EG" else "UNKNOWN")
        return "DIRECTLY_USED" if r["source"] == "MF" else "NOT_USED"

    counts = collections.Counter(rel(r) for r in rows)
    eg = collections.Counter(rel(r) for r in rows if r["source"] == "EG")

    L = ["# A. FORMULA_INVENTORY", "",
         "Every display equation in all three papers, verbatim. Generated by",
         "`tools/build_formula_inventory.py`; do not edit by hand -- the round-trip",
         "check compares this file against the repaired sources and will fail.", "",
         "## Source integrity", "",
         "| paper | control bytes repaired | left | display blocks | `\\neq` |",
         "|---|---|---|---|---|"]
    for name, _, _ in PAPERS:
        r = report[name]
        L.append(f"| `{name}` | **{r['total']}** | {r['control_bytes_left']} | "
                 f"{r['display_blocks']} | {r['neq_in_source']} |")
    L += ["",
          "`\\frac`→`<FF>rac`, `\\tau`→`<TAB>au`, `\\right`→`<CR>ight`,",
          "`\\rangle`→`<CR>angle`, `\\tilde`→`<TAB>ilde`, `\\approx`→`<BEL>pprox`,",
          "and `\\neq`→`<LF>eq`. **`metafind_paper.md` was never damaged.**", "",
          "The last of those is why `\\neq` survived two rounds of \"repaired\": `\\n`",
          "became a REAL newline, which is a legal character, so a byte census",
          "correctly skips it. It is only recoverable inside math, by whether the",
          "following letters spell a LaTeX command beginning with n.", "",
          "## What is checked", "",
          "Counting equations proves nothing -- when `\\rangle` was broken the count",
          "was already right. These can fail:", "",
          "1. no control byte survives repair",
          "2. no bare command name inside any math span, in the sources",
          "3. `$$`, `{}`, `\\left`/`\\right`, `\\langle`/`\\rangle` all balance",
          "4. every formula is SHA256'd at the source and re-hashed after being",
          "   written and read back; the two must be equal",
          "5. every exact formula must appear verbatim in this markdown",
          "6. no ellipsis anywhere -- a truncated formula is a failed inventory",
          "7. `\\neq` counts are preserved from source to inventory",
          "8. the inventory's own formulas are re-checked for bare commands", "",
          "Formulas are in fenced blocks, never in table cells: a pipe ends a cell",
          "and a width cap inserts an ellipsis. Both happened.", "",
          f"**{len(rows)} equations** -- " +
          ", ".join(f"{t} {sum(1 for r in rows if r['source'] == s)}"
                    for _, s, t in PAPERS) + ".", "",
          "## relationship_to_metafind", "",
          "| | all | of which EGNN |", "|---|---|---|"]
    for k in ("DIRECTLY_USED", "MODIFIED", "CONCEPTUAL_SOURCE", "NOT_USED", "UNKNOWN"):
        if counts[k] or eg[k]:
            L.append(f"| `{k}` | {counts[k]} | {eg[k]} |")
    L += ["",
          f"EGNN's {sum(1 for r in rows if r['source'] == 'EG')} split "
          f"{eg['MODIFIED']} MODIFIED / {eg['CONCEPTUAL_SOURCE']} CONCEPTUAL_SOURCE "
          f"/ {eg['NOT_USED']} NOT_USED. Appearing in the EGNN paper is not",
          "evidence MetaFind uses it. The NOT_USED set is not only velocity,",
          "edge inference, autoencoder and QM9 heads -- it also holds generic",
          "equivariance examples, Appendix A proof steps, MLP implementation",
          "formulas and two distance-representation proofs.", ""]

    for name, short, title in PAPERS:
        sub = [r for r in rows if r["source"] == short]
        L += [f"## {title} — `{name}` ({len(sub)})", ""]
        for r in sub:
            eq = f"Eq. ({r['equation']})" if r["equation"] else "unnumbered"
            note = NOTE.get(r["id"], "")
            if not r["equation"]:
                for key, txt in UNNUM.items():
                    if key in r["raw_span"]:
                        note = txt
            L += [f"### `{r['id']}` — {eq}", "",
                  f"- source: `{name}` line {r['line']}",
                  f"- section: {r['section']}",
                  f"- relationship_to_metafind: `{rel(r)}`",
                  f"- sha256: `{r['sha256'][:16]}`"]
            if note:
                L.append(f"- note: {note}")
            L += ["", FENCE + "latex", r["raw_span"].strip(), FENCE, ""]
    INVENTORY.write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
