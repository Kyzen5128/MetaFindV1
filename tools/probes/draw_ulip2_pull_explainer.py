#!/usr/bin/env python3
"""Draw, for Kyzen, how ULIP-2 pulls the three modalities and why a fused
query of the SAME observations still lands on its own gallery entry.

Numbers come from output/look/ulip2_geometry_1024.json (released ULIP-2,
1,024 dev_val assets, measured 2026-09-04). Panels A and D are schematics.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

g = json.loads(Path("output/look/ulip2_geometry_1024.json").read_text())
fig, ax = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("ULIP-2 的「拉近」是什麼，以及為什麼 query 用同一份觀測時，線索越多分數越高", fontsize=15)

# ---- A. schematic: ULIP-2 training ------------------------------------------
a = ax[0, 0]
a.set_title("A. ULIP-2 訓練：只有點雲編碼器會動", fontsize=13)
th = np.linspace(0, 2 * np.pi, 400)
a.plot(np.cos(th), np.sin(th), color="#bbb", lw=1)
def pt(x, y, label, color, marker="o", size=140):
    a.scatter([x], [y], s=size, color=color, marker=marker, zorder=5)
    a.annotate(label, (x, y), textcoords="offset points", xytext=(8, 8), fontsize=11, color=color)
pt(np.cos(1.1), np.sin(1.1), "text t（CLIP，鎖住）", "#1f77b4", "s")
pt(np.cos(-0.35), np.sin(-0.35), "image i（CLIP，鎖住）", "#2ca02c", "^")
p0 = (np.cos(2.6), np.sin(2.6)); p1 = (np.cos(0.4), np.sin(0.4))
pt(*p0, "pc 起點", "#999")
pt(*p1, "pc p（被拉到 t、i 之間）", "#d62728")
a.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="->", lw=2, color="#d62728"))
q0 = (np.cos(3.6), np.sin(3.6))
pt(*q0, "別的物件的 t、i", "#7f7f7f", "x")
a.annotate("推開", xy=(np.cos(3.0) * 0.6, np.sin(3.0) * 0.6), fontsize=10, color="#7f7f7f")
a.text(-1.55, -1.45,
       "損失：p 對自己的 i 和 t 拉近、對別人的 i 和 t 推開（P2I + P2T）。\n"
       "t 和 i 是 CLIP 給的，不動。\n"
       f"拉完的樣子（實測）：cos(t,p)={g['cos_text_own_pc']:.2f}、cos(i,p)={g['cos_image_own_pc']:.2f}，\n"
       f"對別人的 pc 只有 {g['cos_text_other_pc']:.2f} / {g['cos_image_other_pc']:.2f}。\n"
       "「拉近」= 自己的比別人的近，不是三點疊成一點。",
       fontsize=10, va="top")
a.set_xlim(-1.6, 1.9); a.set_ylim(-2.6, 1.4); a.set_aspect("equal"); a.axis("off")

# ---- B. measured: fused query vs own pc and nearest other pc -------------------
b = ax[0, 1]
b.set_title("B. 實測：query 加線索後，離自己多遠、離最近的別人多遠", fontsize=13)
conds = ["只有 pc", "text + pc", "text + image + pc"]
own = [1.0, g["cos_query_tp_to_own_pc"], g["cos_query_full_to_own_pc"]]
oth = [g["cos_pc_nearest_other_pc"], g["cos_query_tp_to_nearest_other"], g["cos_query_full_to_nearest_other"]]
x = np.arange(3); w = 0.36
b.bar(x - w / 2, own, w, color="#d62728", label="query 到「自己的」gallery pc")
b.bar(x + w / 2, oth, w, color="#7f7f7f", label="query 到「最近的別人」gallery pc")
for i in range(3):
    b.text(x[i] - w / 2, own[i] + 0.02, f"{own[i]:.2f}", ha="center", fontsize=11)
    b.text(x[i] + w / 2, oth[i] + 0.02, f"{oth[i]:.2f}", ha="center", fontsize=11)
b.set_xticks(x); b.set_xticklabels(conds, fontsize=11); b.set_ylim(0, 1.15); b.set_ylabel("cosine")
b.legend(loc="upper right", fontsize=10)
b.text(-0.45, -0.22,
       "拉扯真的發生：query 離自己從 1.00 掉到 0.76。\n"
       "但別人也跟著遠（0.59 → 0.52），差距一直有 0.24 以上，所以排名不會翻。\n"
       "gallery 的 pc 和 query 的 pc 是同一個檔（cos 1.00），這是關鍵。",
       fontsize=10, va="top", transform=b.transAxes)

# ---- C. measured: why text alone / image alone are low --------------------------
c = ax[1, 0]
c.set_title("C. 實測：單獨用 text 或 image 為什麼低", fontsize=13)
labels = ["text → 自己的 pc", "text → 別人的 pc（平均）", "image → 自己的 pc", "image → 別人的 pc（平均）", "pc → 最近的別人 pc"]
vals = [g["cos_text_own_pc"], g["cos_text_other_pc"], g["cos_image_own_pc"], g["cos_image_other_pc"], g["cos_pc_nearest_other_pc"]]
cols = ["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a", "#7f7f7f"]
c.barh(labels, vals, color=cols)
for i, v in enumerate(vals):
    c.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=11)
c.set_xlim(0, 1.0); c.invert_yaxis(); c.set_xlabel("cosine")
c.text(0.0, -0.3,
       f"text 只有 0.29 比 0.07：差一點點，1,024 個裡面 text 單獨只對 {g['R1_text_vs_pc']:.0f}%，\n"
       f"自己減最近的別人平均是 {g['text_gap_own_minus_nearest_other']:+.3f}（常常輸給某個很像的別人）。\n"
       f"image 單獨對 {g['R1_image_vs_pc']:.0f}%。pc 自己對自己 100%。\n"
       "所以 text/image 弱是「差距小」，不是「指向別人」。差距小的線索加進去，拉不動 pc 的大差距。",
       fontsize=10, va="top", transform=c.transAxes)

# ---- D. schematic: what the paper's falling row needs -------------------------------
d = ax[1, 1]
d.set_title("D. 論文那列要「越加越低」，需要什麼", fontsize=13)
d.plot(np.cos(th), np.sin(th), color="#bbb", lw=1)
def pt2(x, y, label, color, marker="o", size=140):
    d.scatter([x], [y], s=size, color=color, marker=marker, zorder=5)
    d.annotate(label, (x, y), textcoords="offset points", xytext=(8, 8), fontsize=11, color=color)
pt2(np.cos(-0.3), np.sin(-0.3), "gallery 的 p（自己）", "#d62728")
pt2(np.cos(2.35), np.sin(2.35), "別人的 p_j", "#7f7f7f")
pt2(np.cos(0.45), np.sin(0.45), "query 的 p′（另一份觀測）", "#ff9896")
pt2(np.cos(1.25), np.sin(1.25), "query 的 t′、i′（偏向別人）", "#1f77b4", "s")
qf = (np.cos(1.35) * 0.7, np.sin(1.35) * 0.7)
d.scatter([qf[0]], [qf[1]], s=260, color="black", marker="*", zorder=6)
d.annotate("平均後的 query\n→ 找到別人", qf, textcoords="offset points", xytext=(-95, -30), fontsize=11)
d.text(-1.75, -1.45,
       "論文 w/o ESSGNN：pc 75.1 → text+pc 44.5 → full 51.7（越加越低）。\n"
       "要這樣，query 的三份觀測必須跟 gallery 的不是同一份，\n而且 text/image 要偏向別人。\n"
       "我們現在：p′ = p（同一檔）、t′ = t（同一句）、只有 image 換一張。\n"
       "所以我們只會越加越高；論文沒寫它的 query 是怎麼來的。",
       fontsize=10, va="top")
d.set_xlim(-1.8, 2.3); d.set_ylim(-2.6, 1.4); d.set_aspect("equal"); d.axis("off")

fig.tight_layout(rect=(0, 0, 1, 0.96))
out = Path("output/look/ulip2_pull_explainer.png")
fig.savefig(out, dpi=130)
print(f"-> {out}")
