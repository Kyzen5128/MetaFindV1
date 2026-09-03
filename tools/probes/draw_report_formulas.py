#!/usr/bin/env python3
"""Render the paper's equations and two simple explainer charts as PNGs for the
report deck (fonts in PowerPoint garbled the Unicode math). Output: output/look/pptx/slides/imgs/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "dejavusans"
OUT = Path("output/look/pptx/slides/imgs"); OUT.mkdir(parents=True, exist_ok=True)


def eq(name, tex, w=9.0, h=1.15, size=22):
    fig = plt.figure(figsize=(w, h))
    fig.text(0.02, 0.5, tex, fontsize=size, va="center", ha="left")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


# Eq. 5 -- Stage 1, one-directional InfoNCE, tau = 0.5
eq("eq5", r"$\mathcal{L}_{\mathrm{pre}} = -\log \frac{\exp\left(\mathrm{sim}\left(f_{\mathrm{query}}(Q),\, f_{\mathrm{gallery}}(A)\right)/\tau\right)}"
          r"{\sum_{A' \in \mathcal{B}} \exp\left(\mathrm{sim}\left(f_{\mathrm{query}}(Q),\, f_{\mathrm{gallery}}(A')\right)/\tau\right)}$"
          r"$\qquad \tau = 0.5$", w=11, h=1.6)
# Eq. 6 -- Stage 2 query vector
eq("eq6", r"$e_{\mathrm{query}} = \mathrm{Fusion}\left(e_{\mathrm{text}},\, e_{\mathrm{img}},\, e_{\mathrm{pc}}\right) + \lambda \, e_{\mathrm{layout}}$", w=9, h=1.0)
# Eq. 7/8 -- bidirectional
eq("eq78", r"$\mathcal{L}_{\mathrm{layout}} = \frac{1}{2}\left(\mathcal{L}^{\,\mathrm{q2g}} + \mathcal{L}^{\,\mathrm{g2q}}\right),\quad"
           r"\mathcal{L}^{\,\mathrm{q2g}} = -\log \frac{\exp(\mathrm{sim}(e_{\mathrm{query}}, e_{\mathrm{gallery}})/\tau)}{\sum_{e'} \exp(\mathrm{sim}(e_{\mathrm{query}}, e')/\tau)}$", w=12, h=1.6, size=20)
# the trivial solution
eq("eq_trivial", r"$\text{if } Q \equiv A \text{ (same record)}:\; f_{\mathrm{query}} \equiv f_{\mathrm{gallery}} \Rightarrow \mathrm{sim} = 1 \text{ for the pair},\; \mathcal{L}_{\mathrm{pre}} \to \min \Rightarrow \mathrm{R@1} \to 100$", w=12, h=1.0, size=18)
# mean-of-unit-vectors score, why text cannot flip the ranking
eq("eq_mean", r"$q = \frac{p + t}{\|p + t\|},\qquad \mathrm{own\ score} \propto 1 + p\!\cdot\! t,\qquad \mathrm{other}_j \propto p\!\cdot\! p_j + t\!\cdot\! p_j$"
              r"$\qquad \Rightarrow\ \mathrm{flip\ only\ if}\ t\!\cdot\!p_j - t\!\cdot\!p > 1 - p\!\cdot\!p_j$", w=13, h=1.1, size=18)

# ---- simple chart: query-to-own vs query-to-nearest-other (released ULIP-2, 1,024 assets)
own = [1.00, 0.80, 0.76]; oth = [0.59, 0.53, 0.52]
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(3); w = 0.36
ax.bar(x - w / 2, own, w, color="#e76f51", label="query 到「自己的」gallery 點雲")
ax.bar(x + w / 2, oth, w, color="#8d99ae", label="query 到「最像的別人」gallery 點雲")
for i in range(3):
    ax.text(x[i] - w / 2, own[i] + 0.02, f"{own[i]:.2f}", ha="center", fontsize=13, fontweight="bold")
    ax.text(x[i] + w / 2, oth[i] + 0.02, f"{oth[i]:.2f}", ha="center", fontsize=13)
ax.set_xticks(x); ax.set_xticklabels(["只給點雲", "點雲 + 文字", "點雲 + 文字 + 圖片"], fontsize=13)
ax.set_ylim(0, 1.15); ax.set_ylabel("相似度（cosine）", fontsize=12); ax.legend(fontsize=11, loc="upper right")
ax.set_title("加線索後，query 離自己遠了一點，但離別人更遠：名次不會翻", fontsize=14)
fig.tight_layout(); fig.savefig(OUT / "fig_own_vs_other.png", dpi=160); plt.close(fig)

# ---- simple chart: paper vs P1, only pc and the three fused cells
lab = ["只給點雲", "點雲+文字", "點雲+圖片", "三個都給"]
paper = [75.1, 44.5, 45.8, 51.7]; p1 = [66.6, 95.6, 77.8, 98.1]
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(4); w = 0.36
ax.bar(x - w / 2, paper, w, color="#264653", label="論文")
ax.bar(x + w / 2, p1, w, color="#e76f51", label="我們 P1")
for i in range(4):
    ax.text(x[i] - w / 2, paper[i] + 1.5, f"{paper[i]:.1f}", ha="center", fontsize=12)
    ax.text(x[i] + w / 2, p1[i] + 1.5, f"{p1[i]:.1f}", ha="center", fontsize=12)
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=13); ax.set_ylim(0, 110); ax.set_ylabel("R@1 (%)", fontsize=12)
ax.legend(fontsize=11, loc="upper left")
ax.set_title("論文：加線索反而掉。我們：加線索一路升。", fontsize=14)
fig.tight_layout(); fig.savefig(OUT / "fig_shape_paper_vs_p1.png", dpi=160); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
