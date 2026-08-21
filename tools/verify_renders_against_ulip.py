#!/usr/bin/env python
"""Score our renders against ULIP-2's released `image_feat`. Rebuilds FIND-9.

# SUPPORTS-NODE: n04_render_views

`evidence/n03_n04_upstream_verification.md` FIND-9 measured our renders at
**R@1 83.5%** against ULIP-2's official per-asset image features, while FIND-7
measured our point clouds at **98.0%** against the same target. That 14.5-point
gap is the hypothesis this tool tests: it is what the renderer-v2 defects --
orbiting the wrong up axis, a white background where upstream is black, and
throwing away half the frame -- would be expected to cost.

Why this is the criterion and a silhouette statistic is not
-----------------------------------------------------------

The expected-truth source is an **official upstream artifact**: `image_feat`
inside ULIP's released `.npy`, which we did not produce and cannot tune. A
silhouette match can be improved by fitting a camera parameter to it; a
retrieval score against features computed by someone else on someone else's
renders cannot be fitted the same way, because the target never moves.

It is still a comparison against ULIP, not against MetaFind. A higher score
means our renders sit closer to ULIP-2's, which is what `U-O` asks for where
MetaFind is silent. **It is not evidence about what MetaFind rendered**, and no
result from this tool may be written up as paper fidelity.

The control
-----------

Mismatched pairs. With a pool of N assets, chance R@1 is 1/N, so the gap
between matched and mismatched cosine is what says the measurement is alive
rather than that everything scores high against everything.

    python tools/verify_renders_against_ulip.py --n 200
    python tools/verify_renders_against_ulip.py --n 60 --elevations 5,10,15,20,25
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metafind import paths  # noqa: E402

DEFAULT_ULIP_PC = "*/ulip_pc"


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)


def load_ulip_features(root: Path) -> dict[str, np.ndarray]:
    """uid -> mean of the released ``image_feat`` (12, 1280).

    Meaned to a single vector because our gallery entry is a single vector; the
    per-view features are kept by upstream and discarded here only for the
    comparison, not in the pipeline.
    """
    out: dict[str, np.ndarray] = {}
    for f in glob.glob(str(root / "**" / "*.npy"), recursive=True):
        uid = os.path.basename(f)[:-4]
        try:
            rec = np.load(f, allow_pickle=True).item()
        except Exception:  # noqa: BLE001 -- a corrupt reference file is not our asset's fault
            continue
        feat = rec.get("image_feat")
        if feat is None:
            continue
        out[uid] = np.asarray(feat, dtype=np.float32).mean(axis=0)
    return out


def score(ours: np.ndarray, theirs: np.ndarray) -> dict:
    """R@1, R@5 and the matched/mismatched cosines over a pool of N assets."""
    a, b = _unit(ours), _unit(theirs)
    sim = a @ b.T
    n = len(a)
    matched = np.diag(sim)
    off = sim[~np.eye(n, dtype=bool)]
    order = np.argsort(-sim, axis=1)
    rank = np.array([int(np.where(order[i] == i)[0][0]) for i in range(n)])
    return {
        "n": n,
        "r@1": float((rank == 0).mean()),
        "r@5": float((rank < 5).mean()),
        "median_rank": float(np.median(rank) + 1),
        "matched_cos": float(matched.mean()),
        "mismatched_cos": float(off.mean()),
        "gap": float(matched.mean() - off.mean()),
        "chance_r@1": 1.0 / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ulip-pc", type=Path, required=True,
                    help="directory holding ULIP's extracted objaverse_lvis .npy files")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--elevations", default="",
                    help="comma-separated degrees to sweep; empty uses the module default")
    ap.add_argument("--n-views", type=int, default=0, help="0 = the pipeline's N_VIEWS")
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from metafind.data.encode_text_image import Encoder, aggregate
    from metafind.data.renders import N_VIEWS, ORBIT_ELEVATION_DEG, render_views

    ulip = load_ulip_features(args.ulip_pc)
    have = {p.stem for p in paths.OBJAVERSE_GLB.rglob("*.glb")}
    pool = sorted(set(ulip) & have)
    if not pool:
        print(f"no overlap between {args.ulip_pc} and the GLB corpus", flush=True)
        return 2
    rng = np.random.default_rng(args.seed)
    take = min(args.n, len(pool))
    uids = list(rng.choice(pool, take, replace=False))
    print(f"{len(pool):,} assets overlap; scoring {take}", flush=True)

    n_views = args.n_views or N_VIEWS
    elevations = ([float(x) for x in args.elevations.split(",")]
                  if args.elevations else [ORBIT_ELEVATION_DEG])

    encoder = Encoder()
    theirs = np.stack([ulip[u] for u in uids])
    results = {}
    for elev in elevations:
        vecs = []
        for i, uid in enumerate(uids):
            glb = next(paths.OBJAVERSE_GLB.rglob(f"{uid}.glb"))
            images, _ = render_views(glb, n_views=n_views, elevation_deg=elev)
            # Written to disk rather than encoded in memory so the encoder sees
            # exactly the PNG bytes the pipeline would hand it -- PNG round-trip
            # included, not an in-memory array the real path never produces.
            tmp = paths.OUTPUTS / "_render_probe"
            tmp.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            view_paths = []
            for k, img in enumerate(images):
                p = tmp / f"{uid}_{k:02d}.png"
                Image.fromarray(img).save(p)
                view_paths.append(str(p))
            vecs.append(aggregate(encoder.encode_views(view_paths), "mean"))
            for p in view_paths:
                os.unlink(p)
            if (i + 1) % 25 == 0:
                print(f"  elevation {elev:g}: {i + 1}/{take}", flush=True)
        results[elev] = score(np.stack(vecs), theirs)
        r = results[elev]
        print(f"  elevation {elev:>5g}   R@1 {r['r@1'] * 100:5.1f}%  R@5 {r['r@5'] * 100:5.1f}%  "
              f"median rank {r['median_rank']:.0f}  matched {r['matched_cos']:.4f}  "
              f"mismatched {r['mismatched_cos']:.4f}  gap {r['gap']:.4f}", flush=True)

    best = max(results, key=lambda e: results[e]["r@1"])
    print(f"\nbest elevation {best:g}  R@1 {results[best]['r@1'] * 100:.1f}%")
    print(f"reference: FIND-9 renderer v2 = 83.5% R@1; FIND-7 point clouds = 98.0%")
    print(f"chance R@1 over this pool = {results[best]['chance_r@1'] * 100:.2f}%")
    if args.out:
        args.out.write_text(json.dumps(
            {"seed": args.seed, "n": take, "n_views": n_views,
             "results": {str(k): v for k, v in results.items()}}, indent=1))
        print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
