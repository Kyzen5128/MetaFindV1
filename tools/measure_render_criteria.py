#!/usr/bin/env python
"""Measure `SPEC_M1`'s `S-1`, `S-2` and `S-4` on a render corpus that exists.

# SUPPORTS-NODE: n04_render_views

These three criteria are all statements about **where the object lands in the
frame**, so they share one pass over the PNGs and one foreground detector. The
detector is the only thing here that could quietly be wrong, so it is checked
against a synthetic frame in `demo()` rather than trusted.

    python tools/measure_render_criteria.py --renders data/outputs/renders --n 400
    python tools/measure_render_criteria.py --demo

`S-1` — tall assets render upright
    Image aspect must track the mesh's **y/x** extents, not its **z/y**. Under
    renderer v2 the camera orbited `+Z` while the meshes are Y-up, which is
    exactly the swap this correlates for. The recorded v2 figures are `+0.893`
    for the wrong model and `-0.671` for the right one, **and the signs must
    swap**.

`S-2` — the orbit is about the up axis
    A body of revolution seen from a ring of cameras at constant elevation
    presents the same silhouette from every azimuth, so its per-view aspect is
    constant. Orbiting the wrong axis makes that aspect swing. Reported as the
    spread of per-view aspect within an asset; the criterion is about the
    rotationally symmetric ones, so the distribution's **low tail** is where
    the evidence is, not its mean.

`S-4` — framing at `xmag 1.10` clips nothing
    `S-4` was rewritten because its original `0.60 +- 0.03` was fitted to a
    different `xmag` on 8 assets. The property that would actually corrupt data
    is clipping, so that is what is asserted; the size ratio is measured and
    recorded rather than targeted.

**This measures our own renders against our own criteria.** It says nothing
about agreement with ULIP-2 -- that is `verify_renders_against_ulip.py`, whose
expected-truth source is an artifact we did not produce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The renderer's own constant, not a literal repeated here: if the background
# changes again this follows it, and a stale copy would silently make every
# pixel foreground.
from metafind.data.renders import BACKGROUND_RGBA  # noqa: E402


def foreground_bbox(img: np.ndarray, tol: int = 8) -> tuple[int, int, int, int] | None:
    """(top, left, height, width) of the non-background pixels, or None if empty.

    `tol` absorbs PNG-level rounding and the renderer's own antialiasing at the
    silhouette edge. It is deliberately small: a large tolerance would eat pale
    parts of the asset itself and shrink every measured bbox.
    """
    bg = np.asarray(BACKGROUND_RGBA[:3], dtype=np.int16)
    fg = (np.abs(img[..., :3].astype(np.int16) - bg).max(axis=-1) > tol)
    if img.shape[-1] == 4:  # a transparent pixel is background whatever its RGB
        fg &= img[..., 3] > 0
    rows, cols = np.any(fg, axis=1), np.any(fg, axis=0)
    if not rows.any() or not cols.any():
        return None
    r = np.where(rows)[0]
    c = np.where(cols)[0]
    return int(r[0]), int(c[0]), int(r[-1] - r[0] + 1), int(c[-1] - c[0] + 1)


def measure_asset(rec: dict) -> dict | None:
    """Per-view aspect, clipping, and size, for one sidecar's views."""
    aspects, longest, clipped, blank = [], [], 0, 0
    h_img = w_img = None
    for p in rec["view_paths"]:
        img = np.asarray(Image.open(p).convert("RGBA"))
        h_img, w_img = img.shape[0], img.shape[1]
        box = foreground_bbox(img)
        if box is None:
            blank += 1
            continue
        top, left, h, w = box
        aspects.append(h / max(w, 1))
        longest.append(max(h, w))
        if top == 0 or left == 0 or top + h == h_img or left + w == w_img:
            clipped += 1
    if not aspects:
        return None
    return {
        "uid": rec["uid"],
        "aspect_median": float(np.median(aspects)),
        "aspect_spread": float(np.max(aspects) - np.min(aspects)),
        "aspect_cv": float(np.std(aspects) / max(np.mean(aspects), 1e-9)),
        "longest_px_median": float(np.median(longest)),
        "size_ratio": float(np.median(longest)) / max(w_img or 1, 1),
        "clipped_views": clipped,
        "blank_views": blank,
        "extents": [float(v) for v in rec["raw_bbox_extents"]],
    }


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(Pearson, Spearman) in log space, over the pairs where both are finite.

    Log space because both quantities are ratios: an asset twice as tall and an
    asset half as tall are equally far from square, and only logs treat them so.

    Spearman is reported because Pearson on these log ratios is dominated by a
    handful of extreme assets -- a decal is 1e-8 thick, so its log ratio is
    around -18 while a typical asset sits near 0, and a few such assets move r
    by tenths. Rank correlation cannot be swung that way. `S-1` is stated as a
    sign test, which both statistics answer; the pair is reported so the sign
    is not resting on the statistic that a dozen decals can move.
    """
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan"), float("nan")
    x, y = a[m], b[m]
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(x, y)[0, 1]), float(np.corrcoef(rx, ry)[0, 1])


def report(rows: list[dict], n_total: int) -> dict:
    ext = np.array([r["extents"] for r in rows])
    asp = np.log(np.array([r["aspect_median"] for r in rows]))
    with np.errstate(divide="ignore", invalid="ignore"):
        right = np.log(ext[:, 1] / ext[:, 0])   # y / x  -- upright, Y-up mesh
        wrong = np.log(ext[:, 2] / ext[:, 1])   # z / y  -- what a +Z orbit produces
        # Diagnostic, not a criterion. A ring of cameras at constant elevation
        # about Y sees a horizontal width that sweeps between the x and z
        # extents as the azimuth turns, so `y/x` alone is a partial model of
        # what the image can show and is diluted on any asset where z differs
        # from x. The geometric mean is the model that matches the orbit. It is
        # reported beside `S-1`, never in place of it: `S-1` was pre-registered
        # as y/x and swapping the model after seeing the number is the move
        # `S-3` records the USER forbidding.
        orbit = np.log(ext[:, 1] / np.sqrt(ext[:, 0] * ext[:, 2]))
    (r_right, s_right) = _corr(asp, right)
    (r_wrong, s_wrong) = _corr(asp, wrong)
    (r_orbit, s_orbit) = _corr(asp, orbit)

    cv = np.array([r["aspect_cv"] for r in rows])
    ratio = np.array([r["size_ratio"] for r in rows])
    clipped = [r for r in rows if r["clipped_views"]]

    print(f"\nS-1  tall assets render upright                         n={len(rows)}")
    print(f"                                              pearson   spearman")
    print(f"     log corr( image aspect , mesh y/x )       {r_right:+.3f}    {s_right:+.3f}   <-- the right model")
    print(f"     log corr( image aspect , mesh z/y )       {r_wrong:+.3f}    {s_wrong:+.3f}   <-- the wrong model")
    print(f"     log corr( image aspect , mesh y/sqrt(xz)) {r_orbit:+.3f}    {s_orbit:+.3f}   <-- diagnostic, not a criterion")
    print(f"     v2 recorded: right -0.671, wrong +0.893.  The signs must have swapped.")
    verdict1 = r_right > 0 and r_wrong < r_right and s_right > 0 and s_wrong < s_right
    print(f"     -> {'PASS' if verdict1 else 'FAIL'}")

    print(f"\nS-2  the orbit is about the up axis")
    print(f"     per-asset spread of view aspect (cv): "
          f"p05 {np.percentile(cv, 5):.4f}  median {np.median(cv):.4f}  p95 {np.percentile(cv, 95):.4f}")
    print(f"     assets with cv < 0.02 (near-constant silhouette): "
          f"{int((cv < 0.02).sum()):,} / {len(cv):,}")
    print(f"     a +Z orbit of a Y-up mesh cannot hold any asset near-constant; the low tail is the evidence")

    print(f"\nS-4  framing clips nothing")
    print(f"     assets with any view touching the frame edge: {len(clipped)} / {len(rows)}")
    print(f"     longest foreground side / image width: "
          f"min {ratio.min():.3f}  median {np.median(ratio):.3f}  max {ratio.max():.3f}")
    print(f"     -> {'PASS' if not clipped else 'FAIL'}")
    for r in clipped[:5]:
        print(f"        {r['uid']}  {r['clipped_views']}/11 views  ratio {r['size_ratio']:.3f}")

    return {
        "population_scored": len(rows), "population_available": n_total,
        "s1": {"corr_right_yx": r_right, "corr_wrong_zy": r_wrong,
               "corr_orbit_y_over_sqrt_xz": r_orbit,
               "spearman_right_yx": s_right, "spearman_wrong_zy": s_wrong,
               "spearman_orbit": s_orbit,
               "v2_right": -0.671, "v2_wrong": 0.893, "pass": bool(verdict1)},
        "s2": {"cv_p05": float(np.percentile(cv, 5)), "cv_median": float(np.median(cv)),
               "cv_p95": float(np.percentile(cv, 95)),
               "n_cv_below_0.02": int((cv < 0.02).sum())},
        "s4": {"assets_clipped": len(clipped), "clipped_uids": [r["uid"] for r in clipped],
               "size_ratio_min": float(ratio.min()), "size_ratio_median": float(np.median(ratio)),
               "size_ratio_max": float(ratio.max()), "pass": not clipped},
    }


def demo() -> None:
    """The foreground detector, checked against frames whose answer is known."""
    bg = np.array(BACKGROUND_RGBA, dtype=np.uint8)
    blank = np.tile(bg, (64, 64, 1))
    assert foreground_bbox(blank) is None, "an all-background frame must report no foreground"

    img = blank.copy()
    img[10:30, 20:25, :3] = 0          # 20 tall, 5 wide, top-left at (10, 20)
    img[10:30, 20:25, 3] = 255
    assert foreground_bbox(img) == (10, 20, 20, 5), foreground_bbox(img)

    edge = blank.copy()                 # touches the top edge -> S-4 clipping
    edge[0:10, 5:9, :3] = 0
    edge[0:10, 5:9, 3] = 255
    assert foreground_bbox(edge)[0] == 0

    faint = blank.copy()                # within tol of the background -> not foreground
    faint[10:30, 20:25, :3] = np.array(BACKGROUND_RGBA[:3], dtype=np.uint8) - 3
    assert foreground_bbox(faint) is None, "a near-background patch must not count as the asset"
    print("demo ok: blank, bbox, edge-touch and near-background all as expected")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", type=Path, default=None)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo()
        return 0
    if not args.renders:
        print("--renders is required (or --demo)")
        return 2

    demo()  # never measure with an unchecked detector
    scs = sorted(args.renders.glob("*.json"))
    if not scs:
        print(f"no sidecars in {args.renders}")
        return 2
    # Ordered by hash(seed + uid), NOT by rng.choice(len(scs)). The first
    # version of this drew indices into the sidecar list, so the sample moved
    # when the corpus grew: measuring twice while `n04` was still writing gave
    # S-1 correlations of +0.433 and +0.842 from the same seed and the same n.
    # Hashing the uid makes the sample depend only on the uid and the seed, so
    # the same n on a growing, a finished, or a differently ordered corpus
    # selects the same assets -- which is what `S-1`'s pre-registered
    # "default_rng(20260822)" population is supposed to mean.
    take = min(args.n, len(scs))
    picked = sorted(scs, key=lambda p: hashlib.sha256(
        f"{args.seed}:{p.stem}".encode()).hexdigest())[:take]
    print(f"{len(scs):,} sidecars available; measuring {take}", flush=True)

    rows = []
    for i, sc in enumerate(picked):
        rec = json.loads(sc.read_text())
        if not all(Path(p).exists() for p in rec.get("view_paths", [])):
            continue
        r = measure_asset(rec)
        if r:
            rows.append(r)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{take}", flush=True)

    out = report(rows, len(scs))
    if args.out:
        args.out.write_text(json.dumps({"seed": args.seed, **out}, indent=1))
        print(f"\nwritten {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
