#!/usr/bin/env python3
"""Report charts for Kyzen's Stage 1 / Stage 2 presentation (2026-09-04).

Numbers are copied from output/look/ARMS_TABLE.md (official evaluator, D and C
protocols), the P1 per-epoch dev_val log, exp_query_pc_observation.json, the
Stage 2 arm logs, and exp_ulip2_zero_shot_lvis.json. Writes output/look/report_*.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUT = Path("output/look")
CONDS = ["text", "image", "pc", "T+I", "T+PC", "I+PC", "full"]
PAPER = [13.8, 11.7, 75.1, 17.2, 44.5, 45.8, 51.7]
PAPER_R5 = [23.1, 19.2, 78.0, 21.8, 71.3, 73.1, 76.5]

# ---- Fig 1: Stage 1 arms on protocol D (4,569 dev_val queries vs 36,554 train gallery), R@1 ----
ARMS_D = {
    "pilot10b（舊構法）": [58.0, 84.6, 78.8, 96.5, 99.6, 94.1, 100.0],
    "P1 填表+單視角+L2": [11.6, 29.7, 66.6, 67.5, 95.6, 77.8, 98.1],
    "P3 12 視角 token": [10.4, 24.6, 61.1, 59.7, 94.6, 72.7, 98.0],
    "P4 共用一份 Fusion": [12.0, 25.0, 52.3, 58.5, 94.4, 65.8, 98.0],
    "P5 三模態第二觀測": [14.3, 49.4, 88.5, 59.5, 92.8, 94.0, 95.5],
    "P6 隨機視角": [12.8, 28.8, 67.3, 67.1, 96.9, 77.3, 98.2],
    "P7 關掉 L2": [9.5, 36.1, 69.1, 64.6, 95.6, 81.7, 98.5],
}
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(7)
ax.plot(x, PAPER, "k-o", lw=3, ms=9, label="論文 w/o ESSGNN", zorder=10)
for (name, v), c in zip(ARMS_D.items(), plt.cm.tab10.colors):
    ax.plot(x, v, "-o", color=c, lw=1.6 if "P1" not in name else 3, ms=6, label=name, alpha=0.95)
ax.set_xticks(x); ax.set_xticklabels(CONDS, fontsize=12); ax.set_ylabel("R@1 (%)", fontsize=12)
ax.set_ylim(0, 105); ax.grid(alpha=0.3)
ax.set_title("圖 1　Stage 1 七個版本 vs 論文（D 協定：4,569 個 dev_val query 對 36,554 個 gallery）", fontsize=14)
ax.legend(fontsize=10, ncol=2, loc="lower right")
ax.text(0.01, 0.97, "論文：pc 最高，加了 text/image 反而掉。\n我們：只要含 pc 的組合全部 95+，越加越高。",
        transform=ax.transAxes, va="top", fontsize=11, bbox=dict(fc="white", ec="#999"))
fig.tight_layout(); fig.savefig(OUT / "report_fig1_stage1_arms_D.png", dpi=130); plt.close(fig)

# ---- Fig 2: level / shape ----
LS = {"pilot10b": (0.91, 0.57), "P1": (0.59, 0.41), "P3": (0.56, 0.40), "P4": (0.54, 0.41),
      "P5": (0.66, 0.40), "P6": (0.58, 0.40), "P7": (0.61, 0.44)}
fig, ax = plt.subplots(figsize=(9, 4.8))
names = list(LS); lv = [LS[n][0] for n in names]; sh = [LS[n][1] for n in names]
x = np.arange(len(names)); w = 0.38
ax.bar(x - w / 2, lv, w, color="#9ecae1", label="level：14 格 |ln(我們/論文)| 平均（整體高低）")
ax.bar(x + w / 2, sh, w, color="#3182bd", label="shape：扣掉整體高低後的形狀差")
for i in range(len(names)):
    ax.text(x[i] - w / 2, lv[i] + 0.01, f"{lv[i]:.2f}", ha="center", fontsize=10)
    ax.text(x[i] + w / 2, sh[i] + 0.01, f"{sh[i]:.2f}", ha="center", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11); ax.set_ylim(0, 1.05); ax.legend(fontsize=10)
ax.set_title("圖 2　離論文多遠：level 與 shape（越小越像；D 協定）", fontsize=14)
ax.text(0.01, 0.97, "七個新版本 shape 都在 0.40–0.44：換架構軸不改形狀。\n舊構法 0.57。", transform=ax.transAxes,
        va="top", fontsize=11, bbox=dict(fc="white", ec="#999"))
fig.tight_layout(); fig.savefig(OUT / "report_fig2_level_shape.png", dpi=130); plt.close(fig)

# ---- Fig 3: P1 per-epoch dev_val (C protocol) ----
rows = [json.loads(l) for l in open("/home/kyzen/metafind_data_attrs/outputs/logs/train_stage1_dev_val.jsonl")]
rows = [r for r in rows if "cond_full_R@1" in r]
ep = [r["epoch"] for r in rows]
keys = ["cond_text_R@1", "cond_image_R@1", "cond_pc_R@1", "cond_text+image_R@1",
        "cond_text+pc_R@1", "cond_image+pc_R@1", "cond_full_R@1"]
fig, ax = plt.subplots(figsize=(10, 5.5))
for k, lab, c in zip(keys, CONDS, plt.cm.tab10.colors):
    ax.plot(ep, [r[k] * 100 for r in rows], "-o", label=lab, color=c, ms=5)
ax.set_xlabel("epoch", fontsize=12); ax.set_ylabel("dev_val R@1 (%)", fontsize=12); ax.set_ylim(0, 105)
ax.grid(alpha=0.3); ax.legend(fontsize=10, ncol=4, loc="lower right")
ax.set_title("圖 3　P1 每一代的 dev_val（C 協定，4,569 對 4,569）：full 從第 0 代就高於 pc", fontsize=14)
fig.tight_layout(); fig.savefig(OUT / "report_fig3_P1_epochs.png", dpi=130); plt.close(fig)

# ---- Fig 4: query cloud observation probe (P1 re-scored) ----
d = json.load(open(OUT / "exp_query_pc_observation.json"))
B = d["B"]; order = ["canonical", "resample", "rotz", "nocolor", "jitter02", "half", "half_nocolor", "sparse1k"]
lab = {"canonical": "原檔 (cos 1.00)", "resample": "重取樣 (0.997)", "rotz": "轉向 (0.99)", "nocolor": "去顏色 (0.83)",
       "jitter02": "加雜訊 (0.82)", "half": "單邊掃描 (0.81)", "half_nocolor": "單邊+去色 (0.59)", "sparse1k": "1k 點 (0.37)"}
CK = ["text", "image", "pc", "text+image", "text+pc", "image+pc", "full"]
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(7)
ax.plot(x, PAPER, "k-o", lw=3, ms=9, label="論文 w/o ESSGNN", zorder=10)
for p, c in zip(order, plt.cm.viridis(np.linspace(0, 0.95, len(order)))):
    v = [B[p]["cells"][k]["R@1"] * 100 for k in CK]
    ax.plot(x, v, "-o", color=c, ms=5, label=lab[p])
ax.set_xticks(x); ax.set_xticklabels(CONDS, fontsize=12); ax.set_ylabel("R@1 (%)"); ax.set_ylim(0, 105); ax.grid(alpha=0.3)
ax.legend(fontsize=9, ncol=3, loc="upper left")
ax.set_title("圖 4　只換 query 的點雲觀測，P1 直接重評（D 協定）：pc 掉很快，但 full 永遠高於 pc", fontsize=14)
fig.tight_layout(); fig.savefig(OUT / "report_fig4_query_pc_observation.png", dpi=130); plt.close(fig)

# ---- Fig 5: Stage 2 w/ ESSGNN row (C protocol) ----
S2 = {"P1 父（w/o）": [34.7, 56.9, 86.1, 87.6, 99.0, 92.7, 99.7],
      "先導 2：全模態 query，平坦 5e-4": [10.1, 15.2, 49.2, 22.2, 56.4, 50.5, 58.9],
      "S2-C：只給文字，5e-5 warmup cosine": [22.5, 37.7, 74.3, 62.8, 86.1, 80.1, 88.2],
      "S2-D：全模態，5e-5 warmup cosine": [24.9, 36.1, 71.2, 58.2, 80.9, 73.0, 80.1]}
PW = [11.3, 10.5, 63.2, 15.9, 41.2, 42.0, 48.2]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
x = np.arange(7)
for (n, v), c in zip(S2.items(), ["#444", "#d62728", "#2ca02c", "#1f77b4"]):
    a1.plot(x, v, "-o", color=c, label=n, lw=3 if "S2-C" in n else 1.6)
a1.set_xticks(x); a1.set_xticklabels(CONDS); a1.set_ylim(0, 105); a1.grid(alpha=0.3); a1.set_ylabel("R@1 (%)")
a1.set_title("Stage 2 頭疊在 P1 上、layout 關閉（C 協定）", fontsize=12); a1.legend(fontsize=9, loc="lower right")
ratio_paper = np.array(PW) / np.array(PAPER)
parent = np.array(S2["P1 父（w/o）"])
a2.plot(x, ratio_paper, "k-o", lw=3, label="論文 w/ ÷ w/o")
for n, c in (("S2-C：只給文字，5e-5 warmup cosine", "#2ca02c"), ("S2-D：全模態，5e-5 warmup cosine", "#1f77b4"),
             ("先導 2：全模態 query，平坦 5e-4", "#d62728")):
    a2.plot(x, np.array(S2[n]) / parent, "-o", color=c, label=n.split("：")[0] + " ÷ 父")
a2.axhline(1, color="#999", ls="--"); a2.set_xticks(x); a2.set_xticklabels(CONDS); a2.set_ylim(0, 1.1); a2.grid(alpha=0.3)
a2.set_title("加了 ESSGNN 之後掉多少（比值，越接近黑線越像論文）", fontsize=12); a2.legend(fontsize=9, loc="lower right")
fig.suptitle("圖 5　Stage 2：Table 1 的 w/ ESSGNN 列", fontsize=14)
fig.tight_layout(); fig.savefig(OUT / "report_fig5_stage2.png", dpi=130); plt.close(fig)

# ---- Fig 6: ULIP-2 integrity ----
z = json.load(open(OUT / "exp_ulip2_zero_shot_lvis.json"))
fig, ax = plt.subplots(figsize=(6.5, 4.2))
x = np.arange(2); w = 0.36
ax.bar(x - w / 2, [z["paper_top1"], z["paper_top5"]], w, color="#999", label="ULIP-2 論文 Table 10")
ax.bar(x + w / 2, [z["top1"] * 100, z["top5"] * 100], w, color="#d62728", label="我們（釋出權重，dev_val 4,569 朵，1,156 類）")
for i, (a, b) in enumerate(zip([z["paper_top1"], z["paper_top5"]], [z["top1"] * 100, z["top5"] * 100])):
    ax.text(x[i] - w / 2, a + 1, f"{a:.1f}", ha="center"); ax.text(x[i] + w / 2, b + 1, f"{b:.1f}", ha="center")
ax.set_xticks(x); ax.set_xticklabels(["top-1", "top-5"], fontsize=12); ax.set_ylim(0, 100); ax.legend(fontsize=9)
ax.set_title("圖 6　ULIP-2 沒有被動：零樣本 Objaverse-LVIS 分類", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "report_fig6_ulip2_zero_shot.png", dpi=130); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("report_fig*.png")))
