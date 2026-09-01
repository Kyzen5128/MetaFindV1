#!/usr/bin/env python3
"""Can view averaging alone take the SLIP checkpoint from ~5% to 0.1%?

Codex, 2026-09-01, naming the last externally-testable branch of the image
question:

> Per-view R@1 for all 12, normalize(mean(normalize(v))), normalize(mean(raw
> v)), max-view score, centroid norm, mean view-to-view cosine. Each single
> view ~5% and the mean still ~5% -> pooling eliminated. Each ~5% and the mean
> collapsing to ~0.1% -> SLIP view averaging becomes a real explanation.

He also settles a question this probe therefore does not need to ask: for a
normalised gallery, `normalize(mean(z_v)) @ G.T` and `mean(z_v @ G.T)` produce
the SAME ranking, because the query's norm is a constant across all candidates
of that query. Early and late averaging are not two hypotheses.

The centroid norm is the instrument:

    centroid_norm = || mean_v( normalize(z_v) ) ||

near 1 means the twelve views agree and no mean can destroy the signal; low
means they scatter and averaging could plausibly cancel it.

Run on the 8k-xyz SLIP ViT-B checkpoint, because that is the one whose endpoint
is already low enough (5.53 against the paper's 0.1) for pooling to be able to
finish the job. On ViT-bigG the same comparison was 52.2 single view against
52.7 for the 12-view mean, so pooling was already eliminated there.

Gallery is all 45,692 clouds encoded by this checkpoint, cached on first run
since it costs ten minutes and this probe wants it several times.

Upstream's own unmodified `ULIP_PointBERT` builder; only the environment shims
the 50.576 reproduction used.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
UPSTREAM = pathlib.Path("/home/kyzen/upstream/ULIP_run")

from metafind import paths  # noqa: E402

OUT = REPO / "output" / "look" / "slip_view_pooling.json"
CACHE = paths.OUTPUTS / "_probe" / "slip_vitb_pc_gallery.npy"
CKPT = (REPO / "data/models/ulip2/ULIP-2/pretrained_models/"
        "ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt")
THEIRS = ("/tmp/claude-1002/-home-kyzen-MetaFindV1/"
          "ffe254ed-7700-49b8-99f2-29bde6c5e0be/scratchpad/their_renders")


class Args:
    evaluate_3d = True
    npoints = 8192
    use_height = False


def build():
    from metafind.compat import ulip_patch
    ulip_patch.apply(patch_fps=False)
    os.chdir(UPSTREAM)
    sys.path.insert(0, str(UPSTREAM))
    from models.ULIP_models import ULIP_PointBERT
    ulip_patch.apply(patch_fps=True)
    m = ULIP_PointBERT(Args())
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in sd.get("state_dict", sd).items()}
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"loaded {CKPT.name}  missing {len(miss)}  unexpected {len(unexp)}",
          flush=True)
    return m.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--their-renders", default=THEIRS)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    split = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    corpus = sorted(set(split["train"]) | set(split["test"]))
    pos = {u: i for i, u in enumerate(corpus)}
    have = {pathlib.Path(p).parent.name for p in
            glob.glob(os.path.join(args.their_renders, "**", "*.png"),
                      recursive=True)}
    q_uids = sorted(u for u in have if u in pos and (paths.RENDERS / u).is_dir())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"queries {len(q_uids)}  gallery {len(corpus):,}", flush=True)

    model = build().to(dev)
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    sys.path.insert(0, str(REPO))
    from metafind.data.view_io import load_views_rgb

    def views_of(ps):
        b = torch.stack([tf(im) for im in load_views_rgb(ps)]).to(dev)
        with torch.no_grad():
            return model.encode_image(b).float().cpu().numpy()   # (12, 512)

    V = np.stack([views_of([str(paths.RENDERS / u / f"view_{i:02d}.png")
                            for i in range(12)]) for u in q_uids])
    print(f"encoded views {V.shape}", flush=True)

    rng = np.random.default_rng(args.seed)
    if CACHE.exists():
        G = np.load(CACHE)
        print(f"gallery from cache {CACHE.name} {G.shape}", flush=True)
    else:
        G = np.empty((len(corpus), 512), np.float32)
        sub = rng.permutation(10000)[:8192]
        buf = []
        for k, u in enumerate(corpus):
            x = np.load(paths.POINTCLOUDS / f"{u}.npz")["xyz"].astype(np.float32)[sub]
            x = x - x.mean(0)
            buf.append(x / (np.sqrt((x ** 2).sum(1)).max() + 1e-9))
            if len(buf) == args.batch or k == len(corpus) - 1:
                with torch.no_grad():
                    G[k - len(buf) + 1:k + 1] = model.encode_pc(
                        torch.from_numpy(np.stack(buf)).to(dev)).float().cpu().numpy()
                buf = []
            if (k + 1) % 8000 == 0:
                print(f"  clouds {k + 1:,}/{len(corpus):,}", flush=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(CACHE, G)
        print(f"cached -> {CACHE}", flush=True)

    n = lambda a: torch.nn.functional.normalize(a, dim=-1)
    Gt = n(torch.from_numpy(G).to(dev))
    tgt = torch.tensor([pos[u] for u in q_uids], device=dev)
    Vt = torch.from_numpy(V).to(dev)
    Vn = n(Vt)
    m = len(q_uids)

    def rank_of(q):
        s = n(q) @ Gt.t()
        own = s.gather(1, tgt.unsqueeze(1))
        return (s > own).sum(1) + 1

    def rep(name, q):
        r = rank_of(q)
        d = {"R@1": round((r == 1).sum().item() / m * 100, 2),
             "R@5": round((r <= 5).sum().item() / m * 100, 2),
             "median_rank": int(r.median())}
        print(f"{name:<38s}{d['R@1']:8.2f}{d['R@5']:8.2f}{d['median_rank']:10d}")
        return d

    res = {"checkpoint": CKPT.name, "n_query": m, "n_gallery": len(corpus),
           "chance_R1": round(100.0 / len(corpus), 5), "paper": 0.1,
           "per_view": {}, "pooled": {}}

    print(f"\n{'':<38s}{'R@1':>8s}{'R@5':>8s}{'排名中位':>10s}")
    for k in range(12):
        res["per_view"][k] = rep(f"單一視角 {k:02d}", Vt[:, k])
    single = [res["per_view"][k]["R@1"] for k in range(12)]
    print(f"{'單張平均 / 最好 / 最差':<38s}"
          f"{np.mean(single):8.2f}{max(single):8.2f}{min(single):10.2f}")

    print()
    res["pooled"]["mean_of_normalised"] = rep(
        "normalize(mean(normalize(v)))", Vn.mean(1))
    res["pooled"]["mean_of_raw"] = rep("normalize(mean(raw v))", Vt.mean(1))
    s_all = torch.stack([n(Vt[:, k]) @ Gt.t() for k in range(12)])
    r_max = (s_all.max(0).values > s_all.max(0).values.gather(
        1, tgt.unsqueeze(1))).sum(1) + 1
    res["pooled"]["max_over_views"] = {
        "R@1": round((r_max == 1).sum().item() / m * 100, 2),
        "R@5": round((r_max <= 5).sum().item() / m * 100, 2),
        "median_rank": int(r_max.median())}
    d = res["pooled"]["max_over_views"]
    print(f"{'max over views':<38s}{d['R@1']:8.2f}{d['R@5']:8.2f}"
          f"{d['median_rank']:10d}")

    cn = torch.linalg.norm(Vn.mean(1), dim=-1)
    vv = (Vn @ Vn.transpose(1, 2))
    off = vv.sum((1, 2)) - vv.diagonal(dim1=1, dim2=2).sum(1)
    res["centroid_norm"] = {"mean": round(float(cn.mean()), 4),
                            "p05": round(float(cn.quantile(0.05)), 4),
                            "p95": round(float(cn.quantile(0.95)), 4)}
    res["mean_view_to_view_cos"] = round(float((off / (12 * 11)).mean()), 4)
    print(f"\ncentroid norm ||mean(normalize(v))||  平均 {cn.mean():.4f}  "
          f"p05 {cn.quantile(0.05):.4f}  p95 {cn.quantile(0.95):.4f}")
    print(f"視角之間平均 cosine  {float((off / (12 * 11)).mean()):.4f}")

    mn = res["pooled"]["mean_of_normalised"]["R@1"]
    res["reading"] = (
        f"單張平均 {np.mean(single):.2f}，12 視角平均 {mn:.2f}。"
        + ("pooling 排除 -- 平均沒有讓它崩" if mn > np.mean(single) * 0.5 else
           "平均確實造成崩潰，pooling 是真的解釋"))
    print(f"\n判讀: {res['reading']}")
    print(f"論文 0.1，此池隨機 {100.0/len(corpus):.5f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
