#!/usr/bin/env python3
"""Turn train_*.jsonl into a page you can actually look at.

    python3 tools/plot_training.py                    # every run found
    python3 tools/plot_training.py stage1             # one run
    python3 tools/plot_training.py --out /tmp/x.html

A printed line is not data. "loss 3.9214" scrolls past and cannot be compared
against step 40, or against yesterday's run. These curves are the difference
between "training ran" and "training worked", and the failure they exist to
catch is the one this project keeps finding: the run completes, the checkpoint
saves, every number is finite, and the model learned nothing.

Small multiples, one metric per panel, because the metrics have different units
and a shared y-axis would either flatten the small ones or need a second scale.
Single-series panels carry no legend -- the panel title names the series.

Self-contained HTML: no CDN, no build step, opens from a file:// path.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metafind import paths  # noqa: E402

# Panels are declared, not discovered, so the order is stable across runs and a
# metric that stops being written leaves a visible gap rather than silently
# reflowing the page. `series` lists the keys drawn together on one axis; more
# than one is only allowed where the units genuinely match.
PANELS = [
    ("loss", ["loss"], "對比損失", "越低越好。前 100 步就該明顯往下掉。"),
    ("loss_split", ["loss_q2g", "loss_g2q"], "兩個方向的損失",
     "Stage 2 才有。兩條應該貼在一起 —— 差太多表示某個方向學不動。"),
    ("acc_q2g", ["acc_q2g"], "批次內檢索命中率",
     "在這一批裡有沒有選對。越高越好，但它跟 batch size 有關，不是最終準確率。"),
    ("tau", ["tau"], "溫度 τ",
     "可學習的。它決定「差一點」和「差很多」的懲罰差距。跑掉會讓 softmax 飽和。"),
    ("lam", ["lam"], "λ（場景權重）",
     "Stage 2 才有。Eq. 6 裡乘在場景向量上的係數。掉到 0 表示模型學會忽略場景。"),
    ("grad_norm", ["grad_norm"], "梯度大小",
     "突然衝高 = 不穩；掉到 0 = 沒有東西在學。"),
    ("lr", ["lr"], "學習率", "照 cosine 排程走，應該是一條平滑下降的線。"),
]

# Categorical slots 1-3 of the reference palette. Only the two-series panel uses
# more than one, and that pair validates all-pairs in both modes.
SERIES = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70")]

W, H = 520, 190          # panel plot box
PAD = {"l": 58, "r": 16, "t": 14, "b": 30}


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass          # a torn final line after a kill -9; skip it
    return rows


def downsample(rows: list[dict], limit: int = 600) -> list[dict]:
    """Keep the page light without lying about the shape.

    Stride sampling, and the LAST row is always kept: the final value is the one
    a reader checks against the log, and dropping it would make the chart and
    the log disagree at the only point both show.
    """
    if len(rows) <= limit:
        return rows
    stride = len(rows) // limit + 1
    out = rows[::stride]
    if out[-1] is not rows[-1]:
        out.append(rows[-1])
    return out


def nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** (len(f"{raw:.0e}".split("e")[1]) and int(f"{raw:.0e}".split("e")[1]))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    first = (lo // step) * step
    return [first + i * step for i in range(n + 2) if first + i * step <= hi + step * 0.01]


def fmt(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a < 0.001 or a >= 100000:
        return f"{v:.1e}"
    if a < 1:
        return f"{v:.4g}"
    return f"{v:,.4g}".rstrip("0").rstrip(".") if "." in f"{v:,.4g}" else f"{v:,.4g}"


def panel(key: str, keys: list[str], title: str, note: str, rows: list[dict]) -> str:
    present = [k for k in keys if any(k in r and r[k] is not None for r in rows)]
    if not present:
        return ""

    xs = [r.get("step", i) for i, r in enumerate(rows)]
    x_lo, x_hi = min(xs), max(xs)
    vals = [r[k] for k in present for r in rows if k in r and r[k] is not None]
    y_lo, y_hi = min(vals), max(vals)
    if y_hi == y_lo:                       # a constant series still needs a box
        y_lo, y_hi = y_lo - 0.5, y_hi + 0.5
    pad_y = (y_hi - y_lo) * 0.08
    y_lo, y_hi = y_lo - pad_y, y_hi + pad_y

    def sx(v: float) -> float:
        return PAD["l"] + (0 if x_hi == x_lo else (v - x_lo) / (x_hi - x_lo)) * W

    def sy(v: float) -> float:
        return PAD["t"] + H - (v - y_lo) / (y_hi - y_lo) * H

    parts = []
    for t in nice_ticks(y_lo, y_hi):
        y = sy(t)
        if not (PAD["t"] - 1 <= y <= PAD["t"] + H + 1):
            continue
        parts.append(f'<line class="grid" x1="{PAD["l"]}" y1="{y:.1f}" '
                     f'x2="{PAD["l"] + W}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{PAD["l"] - 8}" y="{y + 3.5:.1f}" '
                     f'text-anchor="end">{fmt(t)}</text>')
    for t in nice_ticks(x_lo, x_hi, 5):
        if not (x_lo <= t <= x_hi):
            continue
        parts.append(f'<text class="tick" x="{sx(t):.1f}" '
                     f'y="{PAD["t"] + H + 18}" text-anchor="middle">{fmt(t)}</text>')

    for i, k in enumerate(present):
        pts = [(sx(r.get("step", j)), sy(r[k]))
               for j, r in enumerate(rows) if k in r and r[k] is not None]
        d = " ".join(f"{'M' if n == 0 else 'L'}{x:.1f} {y:.1f}" for n, (x, y) in enumerate(pts))
        parts.append(f'<path class="s{i}" d="{d}" fill="none"/>')
        if pts:
            parts.append(f'<circle class="s{i}f" cx="{pts[-1][0]:.1f}" '
                         f'cy="{pts[-1][1]:.1f}" r="3.5"/>')

    legend = ""
    if len(present) > 1:
        chips = "".join(
            f'<span class="chip"><i class="s{i}b"></i>{html.escape(k)}</span>'
            for i, k in enumerate(present))
        legend = f'<div class="legend">{chips}</div>'

    last = {k: next((r[k] for r in reversed(rows) if k in r and r[k] is not None), None)
            for k in present}
    readout = " · ".join(f"{html.escape(k)} <b>{fmt(v)}</b>"
                         for k, v in last.items() if v is not None)

    svg_w, svg_h = PAD["l"] + W + PAD["r"], PAD["t"] + H + PAD["b"]
    return f'''<figure class="panel">
  <figcaption>
    <h3>{html.escape(title)}</h3>
    <p class="note">{html.escape(note)}</p>
  </figcaption>
  {legend}
  <div class="plot"><svg viewBox="0 0 {svg_w} {svg_h}" role="img"
       aria-label="{html.escape(title)}">
    <line class="axis" x1="{PAD["l"]}" y1="{PAD["t"] + H}" x2="{PAD["l"] + W}"
          y2="{PAD["t"] + H}"/>
    {"".join(parts)}
  </svg></div>
  <p class="readout">最新　{readout}</p>
</figure>'''


def table(rows: list[dict], keys: list[str]) -> str:
    """The accessibility fallback, and the thing to read when a curve looks odd."""
    show = rows[-25:]
    head = "".join(f"<th>{html.escape(k)}</th>" for k in keys)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{fmt(r[k]) if isinstance(r.get(k), (int, float)) else html.escape(str(r.get(k, '')))}</td>"
            for k in keys) + "</tr>"
        for r in show)
    return (f'<details class="tablewrap"><summary>最後 {len(show)} 筆的數字</summary>'
            f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div></details>')


CSS = """
:root{
  --ground:#f4f7f6; --surface:#fcfcfb; --surface-2:#eaefed;
  --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#8b9995;
  --rule:#dce3e0; --grid:#e6ebe9;
  --s0:#2a78d6; --s1:#eb6834; --s2:#1baf7a;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"PingFang TC","Noto Sans TC","Microsoft JhengHei",sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0f1413; --surface:#1a1a19; --surface-2:#20211f;
  --ink:#fff; --ink-2:#c3c2b7; --ink-3:#8a8a80;
  --rule:#2b2c29; --grid:#262723;
  --s0:#3987e5; --s1:#d95926; --s2:#199e70;
}}
:root[data-theme="dark"]{
  --ground:#0f1413; --surface:#1a1a19; --surface-2:#20211f;
  --ink:#fff; --ink-2:#c3c2b7; --ink-3:#8a8a80;
  --rule:#2b2c29; --grid:#262723;
  --s0:#3987e5; --s1:#d95926; --s2:#199e70;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
     font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:78rem;margin:0 auto;padding:2.5rem 1.25rem 5rem}
h1{font-family:var(--mono);font-size:1.5rem;font-weight:600;margin:0 0 .3rem}
.sub{color:var(--ink-2);margin:0 0 1.8rem;font-size:.95rem}
.runhead{display:flex;flex-wrap:wrap;align-items:baseline;gap:.5rem 1.2rem;
  border-bottom:2px solid var(--rule);padding-bottom:.5rem;margin:2.4rem 0 1.2rem}
.runhead h2{font-family:var(--mono);font-size:.8rem;letter-spacing:.14em;
  text-transform:uppercase;margin:0}
.runhead .meta{color:var(--ink-2);font-size:.85rem;font-variant-numeric:tabular-nums}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(21rem,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule);border-radius:3px;overflow:hidden}
.panel{background:var(--surface);margin:0;padding:1rem 1.1rem 1.15rem}
.panel h3{margin:0 0 .15rem;font-size:.95rem;font-weight:600}
.note{margin:0 0 .5rem;font-size:.82rem;color:var(--ink-2)}
.plot{overflow-x:auto}
svg{display:block;width:100%;height:auto}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--rule);stroke-width:1}
.tick{fill:var(--ink-3);font-family:var(--mono);font-size:10px;
  font-variant-numeric:tabular-nums}
path.s0{stroke:var(--s0);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
path.s1{stroke:var(--s1);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
path.s2{stroke:var(--s2);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
circle.s0f{fill:var(--s0)} circle.s1f{fill:var(--s1)} circle.s2f{fill:var(--s2)}
.legend{display:flex;gap:.9rem;margin:0 0 .35rem;font-size:.8rem;color:var(--ink-2)}
.chip{display:inline-flex;align-items:center;gap:.35rem;font-family:var(--mono)}
.chip i{width:.7rem;height:.16rem;border-radius:1px;display:inline-block}
.s0b{background:var(--s0)} .s1b{background:var(--s1)} .s2b{background:var(--s2)}
.readout{margin:.55rem 0 0;font-family:var(--mono);font-size:.78rem;
  color:var(--ink-2);font-variant-numeric:tabular-nums}
.readout b{color:var(--ink)}
.tablewrap{margin:1rem 0 0;font-size:.85rem}
summary{cursor:pointer;color:var(--ink-2);font-family:var(--mono);font-size:.8rem}
.scroll{overflow-x:auto;margin-top:.6rem}
table{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:.75rem;
  font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:.3rem .6rem;border-bottom:1px solid var(--rule);
  white-space:nowrap}
th{color:var(--ink-3);font-weight:600;text-align:right}
.empty{background:var(--surface);border:1px solid var(--rule);border-radius:3px;
  padding:2rem 1.5rem;color:var(--ink-2)}
.empty code{font-family:var(--mono);font-size:.85rem;background:var(--surface-2);
  padding:.15rem .4rem;border-radius:2px}
"""


def build(runs: dict[str, list[dict]], title: str) -> str:
    if not runs:
        return (f"<title>{title}</title><style>{CSS}</style><div class='wrap'>"
                f"<h1>{title}</h1><div class='empty'>"
                "還沒有任何訓練指標。訓練跑起來之後，每 20 步會寫一筆到 "
                "<code>logs/train_&lt;run&gt;.jsonl</code>，再跑一次這個工具就會有圖。"
                "</div></div>")

    body = []
    for name, rows in runs.items():
        rows = sorted(rows, key=lambda r: (r.get("epoch", 0), r.get("step", 0)))
        keys = [k for k in ("epoch", "step", "loss", "acc_q2g", "tau", "lam",
                            "grad_norm", "lr") if any(k in r for r in rows)]
        thin = downsample(rows)
        last = rows[-1]
        meta = (f"{len(rows):,} 筆　·　最後 epoch {last.get('epoch', '?')} "
                f"step {last.get('step', '?')}")
        if len(thin) < len(rows):
            meta += f"　·　圖上畫 {len(thin):,} 點（等距抽樣，末點保留）"
        panels = "".join(panel(k, ks, t, n, thin) for k, ks, t, n in PANELS)
        body.append(
            f'<div class="runhead"><h2>{html.escape(name)}</h2>'
            f'<span class="meta">{meta}</span></div>'
            f'<div class="grid2">{panels}</div>{table(rows, keys)}')

    return (f"<title>{title}</title><style>{CSS}</style><div class='wrap'>"
            f"<h1>{title}</h1>"
            "<p class='sub'>每 20 步記錄一筆。曲線是判斷「訓練有沒有真的在學」"
            "的依據 —— 存檔成功、數字有限、卻什麼都沒學到，是這個專案一再遇到的失敗。</p>"
            + "".join(body) + "</div>")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="*", help="run names; default = all found")
    ap.add_argument("--out", default=str(paths.OUTPUTS / "training_curves.html"))
    args = ap.parse_args()

    found = sorted(paths.LOGS.glob("train_*.jsonl"))
    if args.run:
        found = [p for p in found if p.stem[len("train_"):] in args.run]

    runs = {}
    for p in found:
        rows = load(p)
        if rows:
            runs[p.stem[len("train_"):]] = rows

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(runs, "MetaFind 訓練曲線"))

    for name, rows in runs.items():
        print(f"  {name:24} {len(rows):,} 筆")
    print(f"\n{len(runs)} 個 run -> {out}")
    if not runs:
        print("  （還沒有訓練指標；訓練跑起來後再執行一次）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
