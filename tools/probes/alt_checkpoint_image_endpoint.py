#!/usr/bin/env python3
"""Does ULIP-2's OTHER released checkpoint put image-only near chance?

Codex, 2026-09-01: `xue2024ulip` cites the ULIP-2 PAPER and names no weights,
so reproducing 50.576 proves we loaded a checkpoint that reproduces 50.6 -- not
that MetaFind loaded the same one.

Searched the paper for anything that would pin it. Across all five .tex files:
`2048` `8192` `10000` `colored` `colour` `xyzrgb` `vit_g` `ViT-B` `SLIP`
`OpenCLIP` `npoints` `checkpoint` `1280` `512` all appear ZERO times. Every
mention is "the ULIP-2 embedding backbone", seven times, with no specification.
So the checkpoint is UNKNOWN, and ULIP-2 released two with different
architectures:

  ULIP-2-PointBERT-10k-xyzrgb-pc-vit_g-...   OpenCLIP ViT-bigG, 10k coloured,
                                             1280-d   <- what we use
  ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-...  SLIP ViT-B, 8,192 xyz-only,
                                             512-d    <- this probe

The second is a materially weaker vision-language space. If MetaFind used it,
the paper's image-only 0.1 might be a property of the checkpoint rather than of
the query construction, and the whole image branch of the investigation closes.

Built with UPSTREAM's own unmodified `ULIP_PointBERT` (ULIP_models.py:314),
which is the builder that checkpoint's architecture requires: timm
`vit_base_patch16_224`, `PointTransformer_8192point.yaml`, embed_dim 512.
`evaluate_3d=True` so it does not try to reload SLIP's initialisation over the
trained weights.

Both the query images and the gallery clouds go through THIS model, since its
512-d space has nothing to do with the 1280-d one everything else in this
project lives in. Image preprocessing follows `main.py:176-181` -- ImageNet
statistics, which is what this checkpoint was trained against -- with
Resize+CenterCrop instead of RandomResizedCrop because evaluation must be
deterministic.

Clouds are our stored 10,000-point xyz, subsampled to 8,192 and re-normalised.
That is a DEVIATION from however the checkpoint's own loader would sample, and
it is recorded rather than hidden; the question here is an order of magnitude,
not a decimal.
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

OUT = REPO / "output" / "look" / "alt_checkpoint_image_endpoint.json"
CKPT = (REPO / "data/models/ulip2/ULIP-2/pretrained_models/"
        "ULIP-2-PointBERT-8k-xyz-pc-slip_vit_b-objaverse-pretrained.pt")
THEIRS = ("/tmp/claude-1002/-home-kyzen-MetaFindV1/"
          "ffe254ed-7700-49b8-99f2-29bde6c5e0be/scratchpad/their_renders")


class Args:
    evaluate_3d = True
    npoints = 8192
    use_height = False


def build():
    # The same environment shims the 50.576 reproduction used: torch._six was
    # removed in PyTorch 2.0 and dataset_3d.py still imports it, plus the CUDA
    # extension stubs. Environment only -- no upstream program logic is
    # touched. [Kyzen, verbatim: 「Torch這些影響不大吧？ 我指的是程式碼」]
    from metafind.compat import ulip_patch
    ulip_patch.apply(patch_fps=False)
    os.chdir(UPSTREAM)
    sys.path.insert(0, str(UPSTREAM))
    from models.ULIP_models import ULIP_PointBERT
    ulip_patch.apply(patch_fps=True)
    m = ULIP_PointBERT(Args())
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"loaded {CKPT.name}\n  missing {len(missing)}  "
          f"unexpected {len(unexpected)}", flush=True)
    for tag, ks in (("missing", missing), ("unexpected", unexpected)):
        if ks:
            print(f"  {tag}[:6] {list(ks)[:6]}", flush=True)
    return m.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gallery", type=int, default=45692)
    ap.add_argument("--their-renders", default=THEIRS)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    split = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    corpus = sorted(set(split["train"]) | set(split["test"]))
    have = {pathlib.Path(p).parent.name for p in
            glob.glob(os.path.join(args.their_renders, "**", "*.png"),
                      recursive=True)}
    q_uids = sorted(u for u in have if u in set(corpus)
                    and (paths.RENDERS / u).is_dir())
    rng = np.random.default_rng(args.seed)
    if args.gallery < len(corpus):
        rest = [u for u in corpus if u not in set(q_uids)]
        keep = set(q_uids) | {rest[i] for i in
                              rng.choice(len(rest),
                                         size=args.gallery - len(q_uids),
                                         replace=False)}
        corpus = [u for u in corpus if u in keep]
    pos = {u: i for i, u in enumerate(corpus)}
    print(f"queries {len(q_uids)}  gallery {len(corpus):,}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build().to(dev)

    from PIL import Image
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    sys.path.insert(0, str(REPO))
    from metafind.data.view_io import load_views_rgb

    def enc_views(ps):
        b = torch.stack([tf(im) for im in load_views_rgb(ps)]).to(dev)
        with torch.no_grad():
            return model.encode_image(b).float().mean(0).cpu().numpy()

    ours, theirs = [], []
    for k, u in enumerate(q_uids):
        ours.append(enc_views([str(paths.RENDERS / u / f"view_{i:02d}.png")
                               for i in range(12)]))
        tp = sorted(glob.glob(os.path.join(args.their_renders, "**", u, "*.png"),
                              recursive=True))
        theirs.append(enc_views(tp[:12]))
        if (k + 1) % 50 == 0:
            print(f"  images {k + 1}/{len(q_uids)}", flush=True)

    G = np.empty((len(corpus), 512), np.float32)
    sub = rng.permutation(10000)[:8192]
    buf = []
    for k, u in enumerate(corpus):
        xyz = np.load(paths.POINTCLOUDS / f"{u}.npz")["xyz"].astype(np.float32)[sub]
        xyz = xyz - xyz.mean(0)
        xyz = xyz / (np.sqrt((xyz ** 2).sum(1)).max() + 1e-9)
        buf.append(xyz)
        if len(buf) == args.batch or k == len(corpus) - 1:
            with torch.no_grad():
                v = model.encode_pc(torch.from_numpy(np.stack(buf)).to(dev))
            G[k - len(buf) + 1:k + 1] = v.float().cpu().numpy()
            buf = []
        if (k + 1) % 4000 == 0:
            print(f"  clouds {k + 1:,}/{len(corpus):,}", flush=True)

    n = lambda a: torch.nn.functional.normalize(a, dim=-1)
    Gt = n(torch.from_numpy(G).to(dev))
    tgt = torch.tensor([pos[u] for u in q_uids], device=dev)
    perm = rng.permutation(len(q_uids))

    res = {"checkpoint": CKPT.name, "embed_dim": 512, "npoints": 8192,
           "n_query": len(q_uids), "n_gallery": len(corpus),
           "chance_R1": round(100.0 / len(corpus), 5),
           "paper_ulip_image_R1": 0.1,
           "vit_bigG_reference": {"ours": 54.84, "theirs": 58.53,
                                  "pool": 45692},
           "arms": {}}
    print(f"\n{'來源':<26s}{'R@1':>8s}{'R@5':>8s}{'排名中位':>10s}{'正例 cos':>10s}")
    for name, Q in (("我們的 render", ours), ("ULIP-2 官方 render", theirs),
                    ("我們的 render, 打亂", [ours[i] for i in perm])):
        q = n(torch.from_numpy(np.stack(Q)).to(dev))
        s = q @ Gt.t()
        own = s.gather(1, tgt.unsqueeze(1)).squeeze(1)
        rank = (s > own.unsqueeze(1)).sum(1) + 1
        m = len(q_uids)
        r = {"R@1": round((rank == 1).sum().item() / m * 100, 2),
             "R@5": round((rank <= 5).sum().item() / m * 100, 2),
             "median_target_rank": int(rank.median()),
             "positive_cos": round(float(own.mean()), 4)}
        res["arms"][name] = r
        print(f"{name:<26s}{r['R@1']:8.2f}{r['R@5']:8.2f}"
              f"{r['median_target_rank']:10d}{r['positive_cos']:10.4f}")
    print(f"{'論文 ULIP image':<26s}{0.1:8.1f}")
    print(f"{'隨機':<26s}{100.0/len(corpus):8.5f}")

    # [CORRECTED] The first version thresholded R@1 at 15 and printed "close to
    # chance" for 5.53, which is 2,525x chance and 55x the paper. The scale that
    # matters is the RATIO to chance and to the paper, not an absolute cut.
    a = res["arms"]["ULIP-2 官方 render"]["R@1"]
    ch = 100.0 / len(corpus)
    res["x_chance"] = round(a / ch, 1)
    res["x_paper"] = round(a / 0.1, 1)
    res["reading"] = (
        f"{a:.2f} 是隨機的 {a/ch:,.0f} 倍、論文 0.1 的 {a/0.1:.0f} 倍。"
        + ("checkpoint 解釋不了" if a / 0.1 > 3 else
           "已落到論文量級，checkpoint 是主因"))
    print(f"\n判讀: {res['reading']}")
    print(f"參考 ViT-bigG 同池: 我們 54.84 / 官方 58.53 -> 換這顆掉了約 "
          f"{54.84 / max(res['arms']['我們的 render']['R@1'], 1e-9):.0f} 倍")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
