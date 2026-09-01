#!/usr/bin/env python3
"""One asset, both sides, everything printed. For reading, not for scoring.

Kyzen, 2026-09-01: 「還是說你直接把你解壓的結果整理給我看 我直接看比對」

Every probe so far has reduced the two datasets to a number. This reduces
nothing: for N assets that appear in both our corpus and ULIP-2's published
shards, it prints what each side actually holds, in the units each side stores
it in, side by side.

    text        their four sources' raw strings, ours as the encoder sees it
    point cloud shape, dtype, coordinate range, radius, colour range
    image       how many views, what the encoder produced
    geometry    per-asset cosine between the two sides' clouds and views

It computes nothing that any other probe reports as a finding. It exists so the
comparison can be checked by eye rather than taken on trust.

Writes a self-contained HTML page next to the JSON, with our own render
thumbnails inlined so the object being described is visible while its
descriptions are read. Theirs are not shown: the published shards carry
`image_feat`, the encoded vectors, not the pixels.
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

OUT_JSON = REPO / "output" / "look" / "side_by_side.json"
OUT_HTML = REPO / "output" / "look" / "side_by_side.html"
SHARDS = ("/tmp/claude-1002/-home-kyzen-MetaFindV1/"
          "ffe254ed-7700-49b8-99f2-29bde6c5e0be/scratchpad/ulip2_shards")


def cloud_stats(xyz: np.ndarray, rgb: np.ndarray) -> dict:
    r = np.sqrt((xyz.astype(np.float64) ** 2).sum(1))
    return {
        "n_points": int(xyz.shape[0]),
        "dtype": str(xyz.dtype),
        "xyz_min": [round(float(v), 4) for v in xyz.min(0)],
        "xyz_max": [round(float(v), 4) for v in xyz.max(0)],
        "radius_max": round(float(r.max()), 4),
        "radius_mean": round(float(r.mean()), 4),
        "centroid": [round(float(v), 5) for v in xyz.mean(0)],
        "rgb_min": round(float(rgb.min()), 4),
        "rgb_max": round(float(rgb.max()), 4),
        "rgb_mean": round(float(rgb.mean()), 4),
    }


def thumb(uid: str, view: int, px: int = 150) -> str | None:
    """Our render, as a data URI. None if the file is not on disk."""
    for pat in (f"{uid}/{view:03d}.png", f"{uid}/view_{view:02d}.png",
                f"{uid}_{view:02d}.png", f"{uid}/{view}.png"):
        p = paths.RENDERS / pat
        if p.exists():
            try:
                from PIL import Image
                im = Image.open(p).convert("RGB")
                im.thumbnail((px, px))
                b = io.BytesIO()
                im.save(b, "JPEG", quality=80)
                return "data:image/jpeg;base64," + base64.b64encode(
                    b.getvalue()).decode()
            except Exception:
                return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=12, help="how many assets")
    ap.add_argument("--shards", default=SHARDS)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    theirs = {os.path.basename(f)[:-4]: f
              for f in glob.glob(os.path.join(args.shards, "*", "*.npy"))}
    split = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
    corpus = set(split["train"]) | set(split["test"])
    both = sorted(u for u in theirs if u in corpus)
    print(f"overlap {len(both):,}, showing {args.n}", flush=True)
    rng = np.random.default_rng(args.seed)
    pick = [both[i] for i in rng.choice(len(both), size=min(args.n, len(both)),
                                        replace=False)]

    rows = []
    for uid in pick:
        d = np.load(theirs[uid], allow_pickle=True).item()
        a = json.loads((paths.ANNOTATIONS / f"{uid}.json").read_text())
        ours_pc = np.load(paths.POINTCLOUDS / f"{uid}.npz")
        ours_emb = np.load(paths.EMBEDDINGS / f"{uid}.npz")

        t_xyz = np.asarray(d["xyz"], np.float32)
        t_rgb = np.asarray(d["rgb"], np.float32)
        o_xyz = ours_pc["xyz"].astype(np.float32)
        o_rgb = ours_pc["rgb"].astype(np.float32)
        t_views = np.asarray(d["image_feat"], np.float32)
        o_views = ours_emb["views"].astype(np.float32)

        def unit(x):
            return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

        rows.append({
            "uid": uid,
            "their_text": {
                "name": (d["text"] or [""])[0],
                "blip_caption": d.get("blip_caption") or "",
                "msft_caption": d.get("msft_caption") or "",
                "retrieval_text": list(d.get("retrieval_text") or [])[:3],
            },
            "our_text": {
                "category": a["category"],
                "synset": a["synset"],
                "description": a["description"],
                "materials": a["materials"],
                "dims_cm": [round(float(a[k]), 2)
                            for k in ("width", "length", "height")],
                "mass_kg": a["mass"],
                "placement": {k: a[k] for k in ("onCeiling", "onWall",
                                                "onFloor", "onObject")},
                "encoded_string": R.serialize_annotation(a),
            },
            "their_cloud": cloud_stats(t_xyz, t_rgb),
            "our_cloud": cloud_stats(o_xyz, o_rgb),
            "their_views": {"n": int(t_views.shape[0]),
                            "dim": int(t_views.shape[1]),
                            "note": "already encoded by OpenCLIP ViT-bigG; "
                                    "the pixels are not in the shard"},
            "our_views": {"n": int(o_views.shape[0]), "dim": int(o_views.shape[1])},
            "view_cosine_same_index": [
                round(float((unit(t_views[k]) * unit(o_views[k])).sum()), 4)
                for k in range(min(t_views.shape[0], o_views.shape[0]))],
            "thumbnails": [thumb(uid, k) for k in (0, 3, 6, 9)],
        })
        print(f"  {uid}  {a['category']}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    slim = json.loads(json.dumps(rows))
    for r in slim:
        r.pop("thumbnails", None)
    OUT_JSON.write_text(json.dumps(
        {"overlap": len(both), "shown": len(rows), "assets": slim},
        indent=1, ensure_ascii=False))

    e = html.escape
    parts = [
        "<meta charset='utf-8'><title>Side by side</title><style>"
        "body{font:14px/1.6 system-ui,sans-serif;margin:0;padding:28px;"
        "background:#f7f7f5;color:#25272b;max-width:1180px}"
        "h1{font-size:22px;margin:0 0 4px}"
        ".sub{color:#6b7078;margin:0 0 26px}"
        ".a{background:#fff;border:1px solid #e2e2de;border-radius:6px;"
        "padding:18px 20px;margin:0 0 22px}"
        ".uid{font:12px ui-monospace,monospace;color:#8b9098}"
        ".cat{font-size:17px;font-weight:650;margin:2px 0 12px}"
        "img{border:1px solid #e2e2de;border-radius:4px;margin:0 6px 10px 0}"
        "table{border-collapse:collapse;width:100%;margin:8px 0 0}"
        "td,th{border-bottom:1px solid #eeeeea;padding:6px 10px;"
        "vertical-align:top;text-align:left}"
        "th{width:150px;color:#6b7078;font-weight:600;font-size:12.5px}"
        ".two{display:grid;grid-template-columns:1fr 1fr;gap:0 22px}"
        ".hd{font-weight:650;font-size:13px;letter-spacing:.03em;"
        "text-transform:uppercase;color:#8b9098;border-bottom:2px solid #d8d8d2;"
        "padding-bottom:5px;margin:14px 0 0}"
        ".s{font:12.5px ui-monospace,monospace;background:#f3f3f0;padding:2px 5px;"
        "border-radius:3px}"
        ".num{font:12.5px ui-monospace,monospace}"
        "</style>",
        f"<h1>官方 ULIP-2 的資料 vs 我們的資料</h1>",
        f"<p class='sub'>兩邊都有的資產 {len(both):,} 個，隨機取 {len(rows)} 個。"
        "縮圖是<b>我們</b>渲的；官方那包只存編碼後的向量，沒有像素。</p>",
    ]
    for r in rows:
        t, o = r["their_text"], r["our_text"]
        parts.append("<div class='a'>")
        parts.append(f"<div class='uid'>{e(r['uid'])}</div>")
        parts.append(f"<div class='cat'>{e(o['category'])}"
                     f" <span class='uid'>{e(o['synset'])}</span></div>")
        for src in r["thumbnails"]:
            if src:
                parts.append(f"<img src='{src}' width='150'>")
        parts.append("<div class='two'>")
        parts.append("<div><div class='hd'>官方 ULIP-2 的文字</div><table>")
        parts.append(f"<tr><th>物件名</th><td>{e(t['name'])}</td></tr>")
        parts.append(f"<tr><th>BLIP</th><td>{e(t['blip_caption'])}</td></tr>")
        parts.append(f"<tr><th>微軟</th><td>{e(t['msft_caption'])}</td></tr>")
        parts.append("<tr><th>LAION 撈的</th><td>"
                     + "<br>".join(e(x) for x in t["retrieval_text"])
                     + "</td></tr></table></div>")
        parts.append("<div><div class='hd'>我們的文字</div><table>")
        parts.append(f"<tr><th>description</th><td>{e(o['description'])}</td></tr>")
        parts.append(f"<tr><th>材質</th><td>{e(', '.join(o['materials']))}</td></tr>")
        parts.append(f"<tr><th>尺寸 cm</th><td class='num'>"
                     f"{o['dims_cm'][0]} × {o['dims_cm'][1]} × {o['dims_cm'][2]}"
                     f"　質量 {o['mass_kg']} kg</td></tr>")
        parts.append(f"<tr><th>擺放</th><td class='num'>"
                     + ", ".join(k for k, v in o["placement"].items() if v)
                     + "</td></tr>")
        parts.append(f"<tr><th>真正餵進編碼器的</th><td><span class='s'>"
                     f"{e(o['encoded_string'])}</span></td></tr></table></div>")
        parts.append("</div>")

        tc, oc = r["their_cloud"], r["our_cloud"]
        parts.append("<div class='hd'>點雲</div><table>"
                     "<tr><th></th><th>官方</th><th>我們</th></tr>")
        for k, lab in (("n_points", "點數"), ("dtype", "型別"),
                       ("radius_max", "最大半徑"), ("radius_mean", "平均半徑"),
                       ("rgb_min", "色彩最小"), ("rgb_max", "色彩最大"),
                       ("rgb_mean", "色彩平均")):
            parts.append(f"<tr><th>{lab}</th><td class='num'>{tc[k]}</td>"
                         f"<td class='num'>{oc[k]}</td></tr>")
        parts.append("</table>")

        cv = r["view_cosine_same_index"]
        parts.append("<div class='hd'>視角</div><table>")
        parts.append(f"<tr><th>張數</th><td class='num'>官方 {r['their_views']['n']}"
                     f"　我們 {r['our_views']['n']}</td></tr>")
        parts.append("<tr><th>同索引 cosine</th><td class='num'>"
                     + " ".join(f"{v:.2f}" for v in cv)
                     + f"　<b>平均 {np.mean(cv):.3f}</b></td></tr></table>")
        parts.append("</div>")

    OUT_HTML.write_text("\n".join(parts))
    print(f"\n-> {OUT_JSON}\n-> {OUT_HTML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
