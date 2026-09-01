#!/usr/bin/env python3
"""Everything both datasets hold for one asset, laid out to be read.

Kyzen, 2026-09-01: 「我要的是 12 張擺出來直接比較 每個資料集有提供的都給我 我都要比較」

`side_by_side_dump` printed the text and the cloud statistics. This adds the
part that was missing -- all twelve views from each side, in index order, above
the twelve-by-twelve cosine matrix between them -- and drops nothing else, so
every field either dataset stores for an asset appears somewhere on the page.

WHERE EACH SIDE'S PIXELS COME FROM
----------------------------------
Ours are on disk: `renders/<uid>/view_NN.png`, twelve per asset.

Theirs are NOT in the shards we downloaded. `ULIP-2/objaverse_lvis/*.npy` holds
`image_feat`, a (12, 1280) matrix of already-encoded vectors, and no pixels at
all. The pixels live in a different directory of the same HuggingFace repo,
`ULIP_Objaverse_Triplets/render_images_resized_224/`, as 193 chunks of about
2.5 GB. `--their-renders` points at an extracted chunk; without it the page
still renders and says so per asset rather than leaving a silent gap.

The two sets of pixels are NOT the same renders. Ours are Blender at three
polar rings of four (`render_blender.py:103`), verified in DL-065 to be
OpenShape's layout with our index order rotated 180 degrees. Theirs are
whatever `ULIP_Objaverse_Triplets` shipped. That is exactly what the cosine
matrix is for: a diagonal means the same camera at the same index, an
off-diagonal maximum means the same cameras in a different order.

Reads only. Encodes nothing, scores nothing, trains nothing.
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import io
import json
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths  # noqa: E402
from metafind.models import resolve_stage1 as R  # noqa: E402

OUT = REPO / "output" / "look" / "full_asset_compare.html"
SHARDS = ("/tmp/claude-1002/-home-kyzen-MetaFindV1/"
          "ffe254ed-7700-49b8-99f2-29bde6c5e0be/scratchpad/ulip2_shards")
E = html.escape


def uri(path: pathlib.Path, px: int = 116) -> str | None:
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        im.thumbnail((px, px))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    except Exception:
        return None


def their_views(root: str | None, uid: str) -> dict[int, pathlib.Path]:
    if not root:
        return {}
    out = {}
    for p in glob.glob(os.path.join(root, "**", uid, "*.png"), recursive=True):
        stem = pathlib.Path(p).stem
        digits = "".join(c for c in stem if c.isdigit())
        if digits:
            out[int(digits)] = pathlib.Path(p)
    return out


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--shards", default=SHARDS)
    ap.add_argument("--their-renders", default=None,
                    help="an extracted objaverse_rgb_chunk_*.tar.gz")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--uids", default=None, help="comma-separated, overrides --seed")
    args = ap.parse_args()

    theirs = {os.path.basename(f)[:-4]: f
              for f in glob.glob(os.path.join(args.shards, "*", "*.npy"))}
    split = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    corpus = set(split["train"]) | set(split["test"])
    both = sorted(u for u in theirs if u in corpus)

    if args.uids:
        pick = [u for u in args.uids.split(",") if u in theirs]
    elif args.their_renders:
        have = {pathlib.Path(p).parent.name
                for p in glob.glob(os.path.join(args.their_renders, "**", "*.png"),
                                   recursive=True)}
        pick = sorted(set(both) & have)[:args.n]
        print(f"他們的 render chunk 裡有 {len(have):,} 個物件，"
              f"跟我們的重疊 {len(set(both) & have):,} 個", flush=True)
    else:
        rng = np.random.default_rng(args.seed)
        pick = [both[i] for i in rng.choice(len(both), size=min(args.n, len(both)),
                                            replace=False)]
    if not pick:
        sys.exit("沒有可比的資產")
    print(f"重疊 {len(both):,}，這頁畫 {len(pick)} 個", flush=True)

    P = ["<meta charset='utf-8'><title>12 views, both datasets</title><style>"
         "body{font:14px/1.65 system-ui,'Noto Sans TC',sans-serif;margin:0;"
         "padding:26px;background:#f6f6f4;color:#24262a;max-width:1500px}"
         "h1{font-size:23px;margin:0 0 4px}.sub{color:#6b7078;margin:0 0 24px}"
         ".a{background:#fff;border:1px solid #e1e1dd;border-radius:6px;"
         "padding:20px 22px;margin:0 0 26px}"
         ".uid{font:12px ui-monospace,monospace;color:#8b9098}"
         ".cat{font-size:18px;font-weight:650;margin:2px 0 14px}"
         ".hd{font-weight:650;font-size:12px;letter-spacing:.05em;"
         "text-transform:uppercase;color:#8b9098;border-bottom:2px solid #d8d8d2;"
         "padding-bottom:5px;margin:20px 0 10px}"
         ".strip{display:flex;gap:5px;flex-wrap:nowrap;overflow-x:auto;"
         "padding-bottom:6px}"
         ".v{flex:0 0 auto;text-align:center}"
         ".v img{border:1px solid #e1e1dd;border-radius:3px;display:block}"
         ".v span{font:11px ui-monospace,monospace;color:#8b9098}"
         ".miss{width:116px;height:116px;border:1px dashed #ccccc6;border-radius:3px;"
         "display:flex;align-items:center;justify-content:center;font-size:10px;"
         "color:#a8aeb5;text-align:center;padding:4px}"
         "table{border-collapse:collapse;margin:0 0 6px;font-size:13px}"
         "td,th{border-bottom:1px solid #efefeb;padding:5px 9px;text-align:left;"
         "vertical-align:top}"
         "th{color:#6b7078;font-weight:600;font-size:12px;white-space:nowrap}"
         ".two{display:grid;grid-template-columns:1fr 1fr;gap:0 24px}"
         ".m{border-collapse:collapse;font:11px ui-monospace,monospace}"
         ".m td,.m th{border:1px solid #ececE8;padding:3px 5px;text-align:center}"
         ".m th{background:#f3f3f0;color:#6b7078}"
         ".s{font:12.5px ui-monospace,monospace;background:#f3f3f0;padding:2px 5px;"
         "border-radius:3px;display:inline-block}"
         ".num{font:12.5px ui-monospace,monospace}"
         "</style>",
         "<h1>一個資產，兩邊全部攤開</h1>"]
    P.append(f"<p class='sub'>兩邊都有的資產 {len(both):,} 個。"
             + ("他們的渲染圖已載入。" if args.their_renders else
                "<b>他們的渲染圖沒有載入</b>——那 185 GB 的包只存編碼後的向量，"
                "像素在 <code>ULIP_Objaverse_Triplets/render_images_resized_224</code>。")
             + "</p>")

    for uid in pick:
        d = np.load(theirs[uid], allow_pickle=True).item()
        a = json.loads((paths.ANNOTATIONS / f"{uid}.json").read_text())
        pc = np.load(paths.POINTCLOUDS / f"{uid}.npz")
        em = np.load(paths.EMBEDDINGS / f"{uid}.npz")
        tv = np.asarray(d["image_feat"], np.float32)
        ov = em["views"].astype(np.float32)
        trend = their_views(args.their_renders, uid)

        P.append("<div class='a'>")
        P.append(f"<div class='uid'>{E(uid)}</div>")
        P.append(f"<div class='cat'>{E(a['category'])} "
                 f"<span class='uid'>{E(a['synset'])}</span></div>")

        P.append("<div class='hd'>我們的 12 個視角 · Blender · phi 60/90/120 三環</div>")
        P.append("<div class='strip'>")
        for k in range(12):
            u = uri(paths.RENDERS / uid / f"view_{k:02d}.png")
            P.append("<div class='v'>" + (f"<img src='{u}' width='116'>" if u
                     else "<div class='miss'>缺</div>")
                     + f"<span>{k:02d}</span></div>")
        P.append("</div>")

        P.append("<div class='hd'>官方 ULIP-2 的 12 個視角</div>")
        P.append("<div class='strip'>")
        for k in range(12):
            p = trend.get(k)
            u = uri(p) if p else None
            P.append("<div class='v'>" + (f"<img src='{u}' width='116'>" if u
                     else "<div class='miss'>只有向量<br>沒有像素</div>")
                     + f"<span>{k:02d}</span></div>")
        P.append("</div>")

        M = unit(tv) @ unit(ov).T
        P.append("<div class='hd'>12×12 cosine　列=官方　行=我們　"
                 "（對角線亮 = 相機順序一致）</div>")
        P.append("<table class='m'><tr><th></th>"
                 + "".join(f"<th>{j:02d}</th>" for j in range(12)) + "</tr>")
        for i in range(12):
            best = int(M[i].argmax())
            P.append(f"<tr><th>{i:02d}</th>" + "".join(
                (f"<td style='background:#d9ecd9;font-weight:700'>{M[i, j]:.2f}</td>"
                 if j == best else f"<td>{M[i, j]:.2f}</td>")
                for j in range(12)) + "</tr>")
        P.append("</table>")
        P.append(f"<p class='num'>每列最大值落在：{M.argmax(1).tolist()}　"
                 f"對角線平均 {np.mean(np.diag(M)):.3f}</p>")

        P.append("<div class='two'>")
        P.append("<div><div class='hd'>官方這包提供的全部欄位</div><table>")
        rt = list(d.get("retrieval_text") or [])
        for lab, val in (
                ("dataset / group", f"{d.get('dataset')} / {d.get('group')}"),
                ("text（物件名）", (d["text"] or [""])[0]),
                ("text_feat", "original (1,1280) + prompt_avg (1,1280)"),
                ("blip_caption", d.get("blip_caption") or ""),
                ("blip_caption_feat", "original + prompt_avg"),
                ("msft_caption", d.get("msft_caption") or ""),
                ("msft_caption_feat", "original + prompt_avg"),
                ("thumbnail_feat", "(1280,) float32"),
                ("retrieval_text", f"{len(rt)} 句：" + " ／ ".join(rt[:4])),
                ("retrieval_text_feat", f"{len(rt)} × original"),
                ("xyz", f"{np.asarray(d['xyz']).shape} {np.asarray(d['xyz']).dtype}"),
                ("rgb", f"{np.asarray(d['rgb']).shape} {np.asarray(d['rgb']).dtype}"),
                ("image_feat", f"{tv.shape} {tv.dtype}")):
            P.append(f"<tr><th>{E(lab)}</th><td>{E(str(val))}</td></tr>")
        P.append("</table></div>")

        P.append("<div><div class='hd'>我們提供的全部欄位</div><table>")
        for lab, val in (
                ("category / synset", f"{a['category']} / {a['synset']}"),
                ("lvis_category", a["lvis_category"]),
                ("identity_confirmed", a["identity_confirmed"]),
                ("description", a["description"]),
                ("候選句數", f"{len(a.get('description_candidates') or [])}"
                             f"（CLIP 排序，取 rank 0）"),
                ("materials", ", ".join(a["materials"])),
                ("尺寸 cm", f"{a['width']:.2f} × {a['length']:.2f} × "
                            f"{a['height']:.2f}　體積 {a['volume']:.1f}"),
                ("mass kg", a["mass"]),
                ("擺放", ", ".join(k for k in ("onCeiling", "onWall", "onFloor",
                                              "onObject") if a[k])),
                ("annotator", a["annotator_model"].split("/")[-1]),
                ("xyz", f"{pc['xyz'].shape} {pc['xyz'].dtype}"),
                ("rgb", f"{pc['rgb'].shape} {pc['rgb'].dtype}"),
                ("views", f"{ov.shape} {em['views'].dtype}"),
                ("text 向量", f"{em['text'].shape} {em['text'].dtype}"),
                ("image 向量", f"{em['image'].shape}（12 視角平均）")):
            P.append(f"<tr><th>{E(lab)}</th><td>{E(str(val))}</td></tr>")
        P.append("</table></div></div>")

        P.append("<div class='hd'>真正餵進文字編碼器的字串</div>")
        P.append(f"<p><span class='s'>{E(R.serialize_annotation(a))}</span></p>")
        P.append("</div>")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(P))
    print(f"-> {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
