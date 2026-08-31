#!/usr/bin/env python3
"""Is our image->pc score bought by the checkpoint having memorised these assets?

THE QUESTION
------------
Raw ULIP-2, no tower, pc-only gallery, our Objaverse corpus:  image -> pc = 56.8 %.
CLIP-GS reports 5.6 % for ULIP-2 image->3D, and MetaFind's ULIP row says 0.1 %.
Unlike the text cell, this one has no "our query is richer" explanation: a render
is a render.

What our corpus does have is CONTAMINATION. Our checkpoint is ULIP-2's
`objaverse_shapenet` release, whose pretraining pool contains every Objaverse-LVIS
asset we evaluate on -- and P<->I alignment on exactly those pairs is what it was
optimised for. CLIP-GS states its compared models exclude Objaverse-LVIS.

⚠ AN EARLIER ARGUMENT OF MINE WAS WRONG AND THIS PROBE EXISTS BECAUSE OF IT.
I dismissed contamination on the grounds that ULIP-2's own clean-vs-contaminated
gap is 4.3 points (46.3 vs 50.6). That gap is CLASSIFICATION. Classification only
needs the category, which ~40 assets share, so memorising one instance buys
almost nothing. INSTANCE retrieval needs to pick this asset out of 45,691 others,
which is exactly what memorising a (render, cloud) pair buys. The 4.3 does not
transfer, and using it to rule contamination out was a bad analogy.

THE HELD-OUT SET
----------------
`outputs/procthor_modalities/` -- 1,467 AI2-THOR assets with 11 renders and a
point cloud each. AI2-THOR ships its own Unity asset library; it is not Objaverse
and not ShapeNet, so ULIP-2 has not seen these. (INFERENCE, not a released
exclusion list: no `common_ids` manifest exists on this machine.)

THE CONTROLS, BECAUSE THE TWO SETS DIFFER IN MORE THAN CONTAMINATION
---------------------------------------------------------------------
1. GALLERY SIZE. ProcTHOR gives 1,467 candidates; our corpus gives 45,692, and
   pool size was already measured at 29 pp. So the Objaverse control is drawn
   down to the SAME 1,467.
2. COLOUR. A ProcTHOR cloud is a depth shell: geometry only, no rgb, so
   `prepare_depth_shell` fills a flat grey 0.5. Our Objaverse clouds carry real
   texture colour. That alone could move the score, so the Objaverse control runs
   BOTH -- real colour and the same flat grey.
3. VIEW COUNT. 11 renders against our 12. Single-view is reported beside the
   mean so the comparison does not rest on the aggregate.

NOT CONTROLLED, AND DECLARED: a ProcTHOR cloud is the VISIBLE DEPTH SHELL, an
Objaverse cloud is the full mesh surface. That is a real domain gap and it is not
removable here, so a ProcTHOR drop is an upper bound on the contamination effect,
not a measurement of it.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from metafind import paths                                          # noqa: E402
from metafind.models.ulip_backbone import (BackboneConfig, DEPTH_SHELL_GREY,  # noqa: E402
                                           ULIPBackbone, pc_norm)
from metafind.eval.retrieval import normalize_for_scoring, rank_of_target  # noqa: E402

OUT = REPO / "output" / "look" / "contamination_holdout.json"
N = 1467          # cap; the usable count is 1,439 and the control matches THAT


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    import torch
    bb = ULIPBackbone(BackboneConfig(device="cuda", train_scope="fuser_only"))
    assert bb.is_frozen()

    def enc_pc(clouds, tag, batch=32):
        out = []
        with torch.no_grad():
            for i in range(0, len(clouds), batch):
                out.append(bb.encode_pc(np.stack(clouds[i:i+batch]).astype(np.float32))
                           .float().cpu().numpy())
                if i and i % (batch*10) == 0: log(f"  pc {tag} {i}/{len(clouds)}")
        return np.concatenate(out)

    def enc_views(paths_per_asset, tag):
        out = []
        with torch.no_grad():
            for i, ps in enumerate(paths_per_asset):
                ims = torch.stack([bb.preprocess(Image.open(p).convert("RGB")) for p in ps])
                out.append(bb.encode_image(ims).float().cpu().numpy())
                if i and i % 200 == 0: log(f"  img {tag} {i}/{len(paths_per_asset)}")
        return out

    def scores(img_per_asset, pcv):
        gal = normalize_for_scoring(pcv)
        col = np.arange(len(pcv))
        r = {}
        for name, q in (("mean_views", np.stack([v.mean(0) for v in img_per_asset])),
                        ("single_view_0", np.stack([v[0] for v in img_per_asset]))):
            a = normalize_for_scoring(q)
            rk = rank_of_target(a @ gal.T, col)
            r[name] = {"R@1": 100.0*float((rk <= 1).mean()),
                       "R@5": 100.0*float((rk <= 5).mean())}
        return r

    res = {}

    # ---------- held-out: ProcTHOR --------------------------------------
    root = paths.OUTPUTS / "procthor_modalities"
    # 28 of the 1,467 folders carry renders but no cloud, so require both rather
    # than assume completeness -- the run that assumed it died on `Bottle_1`.
    ids = sorted(p.name for p in root.iterdir()
                 if p.is_dir() and (p / "pointcloud.npz").exists()
                 and len(list(p.glob("view_*.png"))) == 11)[:N]
    log(f"ProcTHOR: {len(ids):,} assets (AI2-THOR, not in ULIP-2's pool)")
    pc_p = [np.load(root / a / "pointcloud.npz")["xyz"] for a in ids]
    pc_p = [np.concatenate([pc_norm(x.astype(np.float64)).astype(np.float32),
                            np.full((len(x), 3), DEPTH_SHELL_GREY, np.float32)], 1)
            for x in pc_p]
    v_p = enc_views([sorted((root / a).glob("view_*.png")) for a in ids], "procthor")
    res["procthor_heldout_grey"] = scores(v_p, enc_pc(pc_p, "procthor"))
    log(f"  ProcTHOR image->pc R@1 {res['procthor_heldout_grey']['mean_views']['R@1']:.2f}")

    # ---------- control: our Objaverse, same pool size -------------------
    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    rng = np.random.default_rng(0)
    ouids = list(rng.choice(sorted(sp["test"]), size=len(ids), replace=False))
    log(f"Objaverse control: {len(ouids):,} assets, same pool size")
    raw = [np.load(paths.OUTPUTS / "pointclouds" / f"{u}.npz") for u in ouids]
    colour = [np.concatenate([z["xyz"], z["rgb"]], 1).astype(np.float32) for z in raw]
    grey = [np.concatenate([z["xyz"], np.full_like(z["xyz"], DEPTH_SHELL_GREY)], 1)
            .astype(np.float32) for z in raw]
    v_o = enc_views([[paths.OUTPUTS / "renders" / u / f"view_{v:02d}.png" for v in range(12)]
                     for u in ouids], "objaverse")
    res["objaverse_control_colour"] = scores(v_o, enc_pc(colour, "obj-colour"))
    res["objaverse_control_grey"] = scores(v_o, enc_pc(grey, "obj-grey"))

    print(f"\n{'='*72}\nimage -> pc, gallery = {len(ids):,} in every row")
    print(f"{'':38}{'12/11-view mean':>17}{'single view':>13}")
    for k, label in (("objaverse_control_colour", "Objaverse, real colour   (SEEN)"),
                     ("objaverse_control_grey",   "Objaverse, flat grey     (SEEN)"),
                     ("procthor_heldout_grey",    "ProcTHOR, flat grey  (UNSEEN)")):
        print(f"  {label:36}{res[k]['mean_views']['R@1']:17.2f}"
              f"{res[k]['single_view_0']['R@1']:13.2f}")
    c = res["objaverse_control_grey"]["mean_views"]["R@1"]
    p = res["procthor_heldout_grey"]["mean_views"]["R@1"]
    print(f"\n  colour effect  : {res['objaverse_control_colour']['mean_views']['R@1'] - c:+.2f} pp")
    print(f"  seen vs unseen : {p - c:+.2f} pp   <- contamination + depth-shell gap, "
          f"an UPPER BOUND on contamination alone")
    print("  published for reference: CLIP-GS ULIP-2 image->3D 5.6 (20K pool); "
          "MetaFind ULIP row 0.1")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "question": "is image->pc bought by the checkpoint having seen these assets?",
        "n_gallery": len(ids), "results": res,
        "colour_effect_pp": res["objaverse_control_colour"]["mean_views"]["R@1"] - c,
        "seen_minus_unseen_pp": p - c,
        "caveat": ("ProcTHOR clouds are visible depth shells, Objaverse clouds are "
                   "full mesh surface. That gap is inside `seen_minus_unseen`, so the "
                   "figure bounds contamination from above rather than measuring it."),
    }, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
