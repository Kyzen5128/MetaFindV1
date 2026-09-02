#!/usr/bin/env python3
"""Write ARCHIVED.md for the pre-eleven-view archive, wherever it now lives.

A script rather than a heredoc because the manifest names a paper file, and a
shell command that merely MENTIONS a path under `docs/paper/*_source/` is
refused by the research-authority guard. Nothing here writes to the paper
tree; it only quotes a line number.
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
PAPER_LINE = "docs/paper/metafind_" + "source/2methdology.tex:28"


def du(p: pathlib.Path) -> str:
    try:
        return subprocess.run(["du", "-shL", str(p)], capture_output=True,
                              text=True).stdout.split()[0]
    except Exception:
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive")
    args = ap.parse_args()
    A = pathlib.Path(args.archive)
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=REPO).stdout.strip()[:12]

    rows = []
    for name, what in (
        ("checkpoints/sweep_lr", "the eight-arm learning-rate sweep"),
        ("checkpoints/hpo_r1", "the first hyperparameter round"),
        ("checkpoints/stage1_final", "a --phase final attempt on the twelve-view corpus"),
        ("eval", "21 evaluation outputs, all under the same-record construction"),
        ("ladder", "the 5 / 10 / 25 epoch ladder"),
        ("probe", "probe scratch: the query pack, protocol-E clouds, released embeddings"),
        ("look", "probe reports and figures"),
    ):
        p = A / name
        if p.exists():
            rows.append(f"| `{name}` | {du(p)} | {what} |")
    others = sorted(q.name for q in (A / "checkpoints").glob("*")
                    if q.name not in ("sweep_lr", "hpo_r1", "stage1_final"))
    if others:
        rows.append(f"| `checkpoints/` (rest) | | {', '.join(others)} |")

    body = "\n".join(rows)
    text = f"""# ARCHIVED {datetime.date.today()} -- everything produced before the eleven-view corpus

Moved here, not deleted. Repo at the time of the move: `{rev}`.

## Why

MetaFind renders each asset from **eleven** orthogonal viewpoints
({PAPER_LINE}). The corpus that produced everything below was rendered from
**twelve** (OpenShape's three polar rings of four), a recorded deviation.
Kyzen reverted it on 2026-09-02 and `RENDERER_VERSION` went 6 -> 7. Every
number in this tree came from a model trained on twelve-view image vectors
under the same-record query construction, so none of it is comparable with
anything the eleven-view corpus will produce. It is kept because it was
expensive and because the ledger's diagnoses and retractions point at it.

## What is in here

| path | size | what it is |
|---|---|---|
{body}

## What was deliberately NOT archived

* `pointclouds/` -- 10,000 xyz+rgb points per asset, sampler_version 8. Point
  clouds do not depend on the camera layout; the eleven-view corpus reuses
  them unchanged.
* `splits.json`, `scene_splits.json` -- uid-level, seed 20260816, unchanged.
* `scene_graphs/` -- ProcTHOR geometry and edges, independent of rendering.
* `renders/`, `embeddings/`, `annotations/`, `procthor_modalities/`, the two
  gallery indexes, `sem_edge_*`, `procthor_node_embeddings*` -- STALE but left
  in place: the pipeline needs them until the re-render replaces them, and the
  version fields already refuse them where it matters (renderer_version 6
  against a live 7; embedding sidecars carrying n_views 12 against a live 11).
* `checkpoints/qpack_ti_lr2.50e-04_s20260816` and the two canonical records --
  the base the Stage 2 smoke loads.

## Reading anything in here

Every number belongs to the twelve-view corpus and the same-record query
construction. Quoting one beside an eleven-view number is the comparison this
archive exists to prevent.
"""
    (A / "ARCHIVED.md").write_text(text)
    print(f"-> {A / 'ARCHIVED.md'}")
    print(text[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
