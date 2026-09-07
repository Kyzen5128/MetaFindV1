#!/usr/bin/env python3
"""A per-split FOLDER view of the corpus, built from symlinks.

[KYZEN 2026-09-04] 「我現在嚴重懷疑你拆分沒有分資料夾放 那些是練哪些是測試」
Membership has always been a uid LIST (`outputs/splits.json`, same shape as
OpenShape's `meta_data/split/*.json` and ULIP-2's `lvis.json`), and the data
directories (`annotations/`, `pointclouds/`, `embeddings/`, `renders/`) are flat
by uid. That is auditable only with Python. This tool writes

    <OUTPUTS>/split_dirs/<split>/<kind>/<uid>.<ext>  ->  ../../../../<kind>/<uid>.<ext>

so `ls | wc -l` per split answers "which are trained on, which are tested on".
It is a VIEW: nothing is copied or moved, the code keeps reading the lists, and
re-running it after a new split rebuilds the view from scratch.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from metafind import paths

KINDS = {"annotations": (paths.ANNOTATIONS, ".json"),
         "pointclouds": (paths.POINTCLOUDS, ".npz"),
         "embeddings": (paths.EMBEDDINGS, ".npz")}
SPLIT_NAMES = ("train", "val", "test", "holdout")
VIEW_MARKER = ("symlink view of outputs/splits.json; "
               "rebuild with tools/materialize_split_dirs.py\n")


def _remove_previous_view(out: Path) -> None:
    """Preflight the entire view before removing any generated entry."""
    if out.is_symlink() or not out.is_dir():
        raise SystemExit(f"{out} is not a real directory, refusing to rebuild")
    entries = sorted(out.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    readme = out / "README.txt"
    owned = (readme.is_file() and not readme.is_symlink()
             and VIEW_MARKER in readme.read_text())
    metadata = {readme, *(out / name / "uids.txt" for name in SPLIT_NAMES)}
    for p in entries:
        if p.is_symlink() or p.is_dir():
            continue
        if not (p.is_file() and owned and p in metadata):
            raise SystemExit(f"{p} is a real file, refusing to remove")
    for p in entries:
        if p.is_symlink() or p.is_file():
            p.unlink()
        else:
            p.rmdir()
    out.rmdir()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default=str(paths.OUTPUTS / "splits.json"))
    ap.add_argument("--out", default=str(paths.OUTPUTS / "split_dirs"))
    args = ap.parse_args()
    raw = json.loads(Path(args.splits).read_text())
    obj = raw["object"]
    names = [n for n in SPLIT_NAMES if n in obj]
    out = Path(args.out)
    if out.exists() or out.is_symlink():
        _remove_previous_view(out)
    counts = {}
    for name in names:
        ids = obj[name]
        for kind, (src_dir, ext) in KINDS.items():
            d = out / name / kind
            d.mkdir(parents=True, exist_ok=True)
            n = 0
            for u in ids:
                src = src_dir / f"{u}{ext}"
                if src.exists():
                    os.symlink(os.path.relpath(src, d), d / f"{u}{ext}")
                    n += 1
            counts[f"{name}/{kind}"] = n
        (out / name / "uids.txt").write_text("\n".join(ids) + "\n")
    (out / "README.txt").write_text(
        f"scheme {raw.get('scheme', '70/10/20')}; split_seed {raw.get('split_seed')}; "
        f"val_seed {raw.get('val_seed')}\n" + VIEW_MARKER
        + "\n".join(f"{k}: {v}" for k, v in counts.items()) + "\n")
    for k, v in counts.items():
        print(f"  {k:<24} {v:>7,}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
