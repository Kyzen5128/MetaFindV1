#!/usr/bin/env python3
"""Is our FPS substitution load-bearing? Test it without building the CUDA extension.

WHAT ACTUALLY DIFFERS FROM UPSTREAM
------------------------------------
All 18 vendored ULIP source files are BYTE-IDENTICAL to
`github.com/salesforce/ULIP@95d480f` (verified by diff). What differs is four
runtime shims in `metafind/compat/ulip_patch.py`, and only ONE of them touches a
number:

    torch._six          import plumbing, no computation
    knn_cuda stub       dvae.py builds KNN(k=4) at import and never calls it;
                        the live path is dvae.py's own pure-torch knn_point
    pointnet2_ops stub  raises if reached; its only real consumer was misc.fps
    misc.fps REPLACED   pure-torch greedy FPS instead of pointnet2's CUDA kernel

So "run upstream completely unmodified" reduces to one question: does the FPS
implementation change the answer? FPS picks the 512 group centres that PointBERT
sees, so if it does, every point-cloud embedding moves.

WHY NOT JUST BUILD pointnet2_ops
---------------------------------
No nvcc on this machine. `nvidia-cuda-nvcc-cu12` ships only `ptxas`, and torch
here is built against CUDA 13.2 while the available toolkit wheels are 12.9 --
mixing them to build an old extension is a bigger risk to the environment than
the question is worth.

WHAT THIS DOES INSTEAD
-----------------------
Perturb the one free parameter of greedy FPS -- the seed index -- and see
whether the published number moves. Our implementation seeds at index 0, which
is what pointnet2's kernel does too. If starting somewhere ELSE entirely leaves
zero-shot accuracy where it was, then FPS tie-breaking and kernel-level float
differences cannot matter either, and our 50.6 is upstream's 50.6.

If the number DOES move with the seed, this shim is load-bearing and the CUDA
build becomes worth the trouble.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from metafind import paths                                        # noqa: E402
from metafind.eval.retrieval import normalize_for_scoring         # noqa: E402

UPSTREAM = Path("/home/kyzen/upstream/ULIP")
OUT = REPO / "output" / "look" / "fps_sensitivity.json"
N = 8000     # assets; the class set is always all 1,156


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    import torch
    from metafind.compat import ulip_patch
    from metafind.models.ulip_backbone import BackboneConfig, ULIPBackbone

    base = ulip_patch.farthest_point_sample_idx

    def make_fps(start):
        """Same greedy FPS, different seed index. `start=None` = per-cloud random."""
        def _fps(data, number):
            b, n, _ = data.shape
            if start is None:
                first = torch.randint(0, n, (b,), device=data.device)
            else:
                first = torch.full((b,), start, dtype=torch.long, device=data.device)
            centroids = torch.zeros(b, number, dtype=torch.long, device=data.device)
            distance = torch.full((b, n), float("inf"), dtype=data.dtype, device=data.device)
            farthest, bi = first, torch.arange(b, device=data.device)
            for i in range(number):
                centroids[:, i] = farthest
                d = ((data - data[bi, farthest].view(b, 1, 3)) ** 2).sum(-1)
                distance = torch.minimum(distance, d)
                farthest = distance.argmax(-1)
            return torch.gather(data, 1, centroids.unsqueeze(-1).expand(-1, -1, 3))
        return _fps

    meta = json.loads(paths.LVIS_METADATA.read_text())
    templates = json.loads((UPSTREAM / "data" / "templates.json").read_text())["modelnet40_64"]
    sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    corpus = sorted(set(sp["train"]) | set(sp["test"]))
    rng = np.random.default_rng(0)
    uids = [corpus[i] for i in sorted(rng.choice(len(corpus), N, replace=False))]
    tgt = np.array([meta["key_to_id"][meta["value_to_key_mapping"][u]] for u in uids])
    log(f"{N:,} assets · {len(meta['all_keys']):,} classes x {len(templates)} templates")

    bb = ULIPBackbone(BackboneConfig(device="cuda", train_scope="fuser_only"))

    log("class prompts (encoded once, shared by every arm) ...")
    flat = [t.format(l) for l in meta["all_keys"] for t in templates]
    ch = []
    with torch.no_grad():
        for i in range(0, len(flat), 256):
            ch.append(bb.encode_text(flat[i:i+256]).float().cpu().numpy())
    e = np.concatenate(ch).reshape(len(meta["all_keys"]), len(templates), -1)
    e /= np.linalg.norm(e, axis=-1, keepdims=True)
    e = e.mean(1); cls = normalize_for_scoring(e / np.linalg.norm(e, axis=-1, keepdims=True))

    root = paths.OUTPUTS / "pointclouds"
    clouds = [np.concatenate([z["xyz"], z["rgb"]], 1).astype(np.float32)
              for z in (np.load(root / f"{u}.npz") for u in uids)]

    from models.pointbert import misc
    res, centres = {}, {}
    for label, start in (("seed_index_0 (ours = pointnet2's)", 0),
                         ("seed_index_5000", 5000),
                         ("seed_random_per_cloud", None)):
        torch.manual_seed(0)
        misc.fps = make_fps(start)
        out, first_centres = [], None
        with torch.no_grad():
            for i in range(0, len(clouds), 64):
                x = torch.tensor(np.stack(clouds[i:i+64])).cuda()
                if first_centres is None:
                    first_centres = misc.fps(x[:8, :, :3], 512).cpu().numpy()
                out.append(bb.encode_pc(x).float().cpu().numpy())
                if i and i % 2560 == 0: log(f"  {label} {i}/{len(clouds)}")
        q = normalize_for_scoring(np.concatenate(out))
        order = np.argsort(-(q @ cls.T), axis=1)[:, :5]
        res[label] = {"top1": 100.0*float((order[:, :1] == tgt[:, None]).any(1).mean()),
                      "top5": 100.0*float((order == tgt[:, None]).any(1).mean())}
        centres[label] = first_centres
        log(f"  {label}: top-1 {res[label]['top1']:.2f}")
    misc.fps = ulip_patch.fps    # restore

    a = centres["seed_index_0 (ours = pointnet2's)"]
    same = {k: float(np.abs(v - a).max()) for k, v in centres.items()}

    print(f"\n{'='*70}\nFPS seed sensitivity · zero-shot Objaverse-LVIS · n={N:,}")
    print(f"{'FPS 起點':<34}{'top-1':>9}{'top-5':>9}{'中心點與 seed0 最大差':>24}")
    for k, v in res.items():
        print(f"  {k:<32}{v['top1']:9.2f}{v['top5']:9.2f}{same[k]:24.4f}")
    print(f"\n  官方公開值 (全 45,692)：top-1 50.6 / top-5 79.1")
    print("  若三行 top-1 幾乎相同，FPS 的實作細節不影響結論，")
    print("  我們的純 torch 版就等同上游未修改版。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "question": "does the FPS substitution change the published number?",
        "n_assets": N, "results": res,
        "max_centre_coord_diff_vs_seed0": same,
        "note": "all 18 vendored ULIP files are byte-identical to upstream; FPS is "
                "the only shim that touches a number",
    }, indent=1, default=float))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
