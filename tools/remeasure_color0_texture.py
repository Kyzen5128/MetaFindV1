#!/usr/bin/env python
"""V1.0 -- does `D-12` survive the per-point colour rule?

# SUPPORTS-NODE: n03_sample_pointclouds

`D-12` withdraws COLOR_0 from the `texture` class. It was decided on a
measurement taken under `SAMPLER_VERSION 6`, where a point's colour was the mean
of its triangle's three BAKED vertex colours:

    over all 37 texture assets carrying COLOR_0 that ULIP also publishes
      modulated is DARKER on 37 of 37, by 0.2076 mean
      cosine against ULIP's own cloud, through the frozen ULIP-2 encoder:
        0.8980 modulated   vs   0.9005 not modulated

`SAMPLER_VERSION 7` reads the texture at the sampled point instead. **The
measurement does not carry over**: baking to vertices and averaging is a
low-pass filter, so the two rules disagree most exactly where the texture has
detail -- which is the population this decision is about. A decision inherited
across a change to the thing it was measured on is not a decision, it is a
habit.

So: re-run it, same population, same criterion, new sampler. Whichever rule the
frozen encoder prefers is the one that ships, and if the two are inside noise
the tie-break stays what `R-11` says it is -- agree with upstream, register only
a deliberate divergence.

    python tools/remeasure_color0_texture.py --ulip-npy DIR

`--ulip-npy` is a directory of ULIP's released `.npy` triplets (from
`SFXX/ULIP`, `ULIP-2/objaverse_lvis/*.tar.gz`). Only assets present in BOTH that
release and our corpus can be scored, because the criterion is agreement with
upstream's own cloud.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metafind import paths  # noqa: E402

paths.setup_env()


def our_cloud(glb: Path, uid: str, *, modulate_texture: bool):
    """Sample one asset under SAMPLER_VERSION 7, with COLOR_0 on or off.

    The switch is applied by monkeypatching `_texture_is_samplable`, which is
    the single predicate that decides whether a textured part keeps its UVs
    (COLOR_0 excluded, the `D-12` behaviour) or is baked through `to_color()`
    where `_commit` multiplies COLOR_0 in. Patching the predicate rather than
    duplicating the sampler means both arms run the SHIPPING code path.
    """
    from metafind.data import pointclouds as pc

    original = pc._texture_is_samplable
    try:
        if modulate_texture:
            pc._texture_is_samplable = lambda vis: False
        xyz, rgb, *_ = pc.sample_mesh(glb, pc.uid_seed(uid))
    finally:
        pc._texture_is_samplable = original
    return pc.pc_norm(xyz.astype(np.float64)).astype(np.float32), rgb.astype(np.float32)


def encode(backbone, xyz: np.ndarray, rgb: np.ndarray):
    import torch

    cloud = np.concatenate([xyz, rgb], axis=1)[None]
    with torch.no_grad():
        return backbone.encode_pc(torch.from_numpy(cloud).float()).squeeze(0).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ulip-npy", type=Path, required=True,
                    help="directory of ULIP's released .npy triplets")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=paths.OUTPUTS / "v1_0_color0_texture.json")
    # `cpu` exists so this can be measured WHILE n04 owns the GPU. Hiding the
    # card with CUDA_VISIBLE_DEVICES does not work -- BackboneConfig.device
    # defaults to "cuda" and .to() then raises. Ask for CPU explicitly instead.
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    import torch

    from metafind.data import pointclouds as pc
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

    released = {p.stem: p for p in args.ulip_npy.rglob("*.npy")}
    print(f"ULIP released clouds available: {len(released)}", flush=True)

    # The population is fixed by the decision being re-tested: textured assets
    # that carry COLOR_0. Anything else cannot discriminate the two rules.
    from metafind.data import meshload

    # One walk, not one per uid. The GLBs are sharded, so `rglob(f"{uid}.glb")`
    # inside the loop re-walks 46k files 4,999 times.
    on_disk = {p.stem: p for p in paths.OBJAVERSE_GLB.rglob("*.glb")}
    print(f"GLBs on disk: {len(on_disk)}", flush=True)

    rows = []
    for uid, npy in sorted(released.items()):
        glb = on_disk.get(uid)
        if glb is None:
            continue
        try:
            if not meshload.has_color0(glb):
                continue
            _parts, worst, _srcs, _mod = pc.load_parts(glb)
        except Exception:  # noqa: BLE001 -- a GLB we cannot open is not evidence
            continue
        if worst != "texture":
            continue
        rows.append((uid, glb, npy))
        if args.limit and len(rows) >= args.limit:
            break
    print(f"population (texture + COLOR_0 + released by ULIP): {len(rows)}", flush=True)
    if not rows:
        print("EMPTY POPULATION -- cannot re-measure; D-12 stays as it is")
        return 2

    backbone = ULIPBackbone(BackboneConfig(train_scope="fuser_only",
                                          device=args.device))

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    out = []
    for uid, glb, npy in rows:
        d = np.load(npy, allow_pickle=True).item()
        ref = encode(backbone,
                     pc.pc_norm(np.asarray(d["xyz"]).astype(np.float64)).astype(np.float32),
                     np.asarray(d["rgb"]).astype(np.float32))
        rec = {"uid": uid}
        for name, mod in (("not_modulated", False), ("modulated", True)):
            xyz, rgb = our_cloud(glb, uid, modulate_texture=mod)
            rec[f"cos_{name}"] = cos(encode(backbone, xyz, rgb), ref)
            rec[f"brightness_{name}"] = float(rgb.mean())
        out.append(rec)
        print(f"  {uid[:12]}  not_mod {rec['cos_not_modulated']:.4f}  "
              f"mod {rec['cos_modulated']:.4f}", flush=True)
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # [ULIP2 Reviewer, 2026-08-23] Selection bias, and it is not hypothetical.
    # The argument for re-measuring at all is that averaging is a low-pass
    # filter, so the two rules diverge most where the TEXTURE HAS DETAIL. These
    # 37 were selected by overlapping with ULIP's release, not by detail. If
    # they happen to be low-detail, the run returns "no difference" and the
    # cause is the sample, not the rule -- and nothing in the output would say
    # so. Reporting the detail distribution of the 37 against the whole texture
    # class is what makes a null result readable.
    detail = [float(np.var(our_cloud(glb, uid, modulate_texture=False)[1]))
              for uid, glb, _ in rows]
    print(f"\ntexture detail (variance of sampled RGB) over the {len(rows)} scored:")
    print(f"  mean {statistics.mean(detail):.5f}  median {statistics.median(detail):.5f}  "
          f"min {min(detail):.5f}  max {max(detail):.5f}")
    print("  ^ compare against the whole `texture` class with "
          "tools/texture_detail_baseline.py before reading a null result")

    a = [r["cos_not_modulated"] for r in out]
    b = [r["cos_modulated"] for r in out]
    wins_mod = sum(1 for x, y in zip(a, b) if y > x)
    print("\n" + "=" * 66)
    print(f"n = {len(out)}")
    print(f"  NOT modulated (D-12, current)  mean {statistics.mean(a):.4f}  "
          f"median {statistics.median(a):.4f}")
    print(f"  modulated (glTF 2.0)           mean {statistics.mean(b):.4f}  "
          f"median {statistics.median(b):.4f}")
    print(f"  modulated wins on {wins_mod} of {len(out)}")
    print(f"  brightness delta (mod - not)   "
          f"{statistics.mean(r['brightness_modulated'] - r['brightness_not_modulated'] for r in out):+.4f}")
    # A margin smaller than the standard error of a coin flip is not a result.
    # Saying so here costs one line and stops a tie being written up as a finding.
    se = (len(out) ** 0.5) / 2
    if abs(wins_mod - len(out) / 2) < se:
        print(f"  ^ WITHIN NOISE: {wins_mod}/{len(out)} is inside +/-{se:.1f} of a "
              f"coin flip. Report as undecided; R-11 keeps D-12 as it stands.")
    print("=" * 66)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"sampler_version": pc.SAMPLER_VERSION, "n": len(out), "rows": out,
         "texture_detail_of_scored": {
             "mean": statistics.mean(detail), "median": statistics.median(detail),
             "min": min(detail), "max": max(detail),
             "note": "variance of sampled RGB; compare with the whole texture "
                     "class before reading a null result as 'the rules agree'"}},
        indent=1))
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
