"""Contact sheets for the CLIP-score bands, ten assets per sheet.

[USER 2026-08-28] 「低分數區間你每十張做成一張圖給我 我先看區間呈現的結果」

Read-only. Reads annotations_index.jsonl for the winning description and its
clip_score, and the n04 render sidecar for the view paths -- the paths are never
rebuilt from a naming convention.

[CORRECTED 2026-08-28, Codex finding 5] This said an asset whose render is
incomplete "simply drops out". FALSE: an asset with no sidecar, or with fewer
views than VIEWS asks for, still gets its row; the missing view is drawn as a
red rectangle. That is the intended behaviour -- a reviewer should see that a
view is missing rather than have the asset vanish -- but the docstring claimed
the opposite of what the code does.

Pixels come from `view_io.load_view_rgb`, the same compositor n05, n06 and the
CLIP ranker use. Earlier versions composited onto WHITE while every model
consumer sees BACKGROUND_RGB = (0, 0, 0) (USER DECISION U-BG, 2026-08-23). A
sheet on white shows a dark asset clearly that CLIP saw against black, so any
conclusion drawn from it about "what a low score looks like" was drawn from
different pixels than the score was.

The MEDIAN band is included on purpose and labelled as such. Without it a low
score has nothing to be low RELATIVE TO: a sheet of 0.19s looks bad until you
see that the 0.29s look much the same, at which point the finding is about the
instrument, not the corpus.

    python tools/build_lowclip_sheets.py            # 5 bands x 10
    python tools/build_lowclip_sheets.py --per 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metafind import paths  # noqa: E402

BANDS = [
    ("1", 0.00, 0.15, "最爛的一批"),
    ("2", 0.15, 0.18, "很低"),
    ("3", 0.18, 0.20, "低"),
    ("4", 0.20, 0.22, "偏低"),
    ("5", 0.285, 0.295, "中位數 (對照組)"),
]
VIEWS = (0, 4, 8)          # three angles out of the twelve
THUMB = 240
PAD = 12
TEXT_W = 760
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
# DejaVu has no CJK, so the Chinese header rendered as tofu boxes on the first
# pass. The header is the only part that has to say anything in Chinese.
CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def views_for(uid: str) -> list[str]:
    sc = paths.OUTPUTS / "renders" / f"{uid}.json"
    if not sc.is_file():
        return []
    try:
        return json.loads(sc.read_text()).get("view_paths", [])
    except (OSError, json.JSONDecodeError):
        return []


def thumb(p: str) -> tuple[Image.Image, float]:
    """The thumbnail as the MODEL saw it, and how much of the frame it fills.

    Compositing goes through `view_io.load_view_rgb` rather than being repeated
    here: that module is the single place the background decision lives, and a
    sheet that composites differently from the encoder is showing the reviewer
    different pixels than the score was computed from.

    Coverage is measured from the ALPHA channel, not from colour -- against
    BACKGROUND_RGB = (0, 0, 0) a dark object is nearly invisible to a colour
    threshold and perfectly separable by alpha. It is here because the first
    sheet made it look obvious that the bottom band is mostly assets rendered as
    a speck, and "it looks tiny" is an impression until it is a number.
    """
    import numpy as np

    from metafind.data.view_io import BACKGROUND_RGB, load_view_rgb

    alpha = Image.open(p)
    if alpha.mode == "RGBA":
        a = np.asarray(alpha.split()[3])
        cover = float((a > 16).mean())
    else:
        cover = float("nan")      # no alpha: coverage is not defined, not zero
    bg = load_view_rgb(p)
    bg.thumbnail((THUMB, THUMB))
    out = Image.new("RGB", (THUMB, THUMB), BACKGROUND_RGB)
    out.paste(bg, ((THUMB - bg.width) // 2, (THUMB - bg.height) // 2))
    return out, cover


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--out", type=Path, default=Path("output/look/lowclip"))
    # [USER 2026-08-28] 「clip分數低不代表錯誤 你應該要給我的是標籤不同分數又低的」
    # The bands above answer "what does a score look like". This answers the
    # different question "where is an error likely": the LABEL disagreeing is
    # the error signal, the score is only a way to order the disagreements.
    ap.add_argument("--relation", choices=["divergent", "refined", "exact"],
                    help="only this category_relation; sheets run in ascending "
                         "score order instead of by band")
    ap.add_argument("--max-score", type=float, default=0.20)
    ap.add_argument("--sheets", type=int, default=10**6)
    # `identity_confirmed` is the model's own yes/no on "are the images the
    # catalogued object" -- unrelated to clip_score, and NOT a gate: false is
    # written to the card exactly like true (annotate.py:1090).
    ap.add_argument("--confirmed", choices=["true", "false"],
                    help="only rows whose identity_confirmed is this")
    # [USER 2026-08-28] 「把我輸的記錄下來 我人工驗證」 -- the duel's losers are
    # an explicit uid list, in the order the duel ranked them, so the sheets
    # must follow that list rather than re-derive an order from the corpus.
    ap.add_argument("--uids-file", type=Path,
                    help="exactly these uids, in this order, 10 per sheet")
    ap.add_argument("--shuffle", action="store_true",
                    help="sample at random instead of walking up the score, "
                         "which is the honest order when the score is known "
                         "not to rank quality")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    idx = paths.OUTPUTS / "logs" / "annotations_index.jsonl"
    for line in idx.open():
        if not line.strip():
            continue
        r = json.loads(line)
        w = next((c for c in (r.get("description_candidates") or [])
                  if c.get("rank") == 0), None)
        if not w or "clip_score" not in w:
            continue
        rows.append((float(w["clip_score"]), r))
    rows.sort(key=lambda t: t[0])
    print(f"{len(rows):,} scored")

    CHUNKS = {}
    if args.uids_file:
        want = [u for u in args.uids_file.read_text().split() if u]
        by_uid = {r["uid"]: (s_, r) for s_, r in rows}
        BANDS[:] = []      # the five default score bands are not wanted here
        # [Codex finding 5] A uid the index does not carry used to be counted
        # and skipped, so a 21-uid review could quietly render 20 rows with the
        # numbering closed up behind the gap. A review set that is not the set
        # asked for is not a review set.
        absent = [u for u in want if u not in by_uid]
        if absent:
            raise SystemExit(
                f"{len(absent)} of {len(want)} uid(s) are not in "
                f"annotations_index.jsonl, e.g. {absent[:3]}. They may have been "
                "excluded from the corpus -- point --uids-file at a list that "
                "matches the index, or read them from annotations_excluded/.")
        sel = [by_uid[u] for u in want]
        for i in range(0, len(sel), args.per):
            chunk = sel[i:i + args.per]
            tag = f"{i // args.per + 1:03d}"
            CHUNKS[tag] = chunk
            BANDS.append((tag, chunk[0][0], chunk[-1][0] + 1e-9,
                          f"CLIP 判 LVIS 贏、模型輸 · 第 {i + 1}-{i + len(chunk)} 筆 "
                          f"(共 {len(sel)})"))
        # `missing` used to be printed here. It cannot be non-zero any more --
        # the guard above raises instead -- and removing its assignment left this
        # line referencing a name that no longer exists.
        print(f"{len(sel):,} uid -> {len(BANDS)} 張")
        args.relation = args.relation or "loss"
        args.shuffle = True          # keep the file's order, do not re-sort
    elif args.relation:
        sel = [(s_, r) for s_, r in rows
               if r.get("category_relation") == args.relation and s_ < args.max_score
               and (args.confirmed is None
                    or r.get("identity_confirmed") is (args.confirmed == "true"))]
        if args.shuffle:
            random.Random(args.seed).shuffle(sel)
        BANDS[:] = []
        for i in range(0, min(len(sel), args.sheets * args.per), args.per):
            chunk = sel[i:i + args.per]
            CHUNKS[f"{i // args.per + 1:03d}"] = chunk
            BANDS.append((f"{i // args.per + 1:03d}", chunk[0][0], chunk[-1][0] + 1e-9,
                          f"{args.relation}"
                          + ("" if args.confirmed is None
                             else f" · identity_confirmed={args.confirmed}")
                          + f" · 第 {i + 1}-{i + len(chunk)} 筆 (共 {len(sel)})"))
        print(f"{args.relation} 且 < {args.max_score}: {len(sel):,} 筆 -> {len(BANDS)} 張")

    f_hdr = ImageFont.truetype(CJK, 28)
    f_lbl = ImageFont.truetype(MONO, 19)
    f_txt = ImageFont.truetype(CJK, 17)   # the two row labels are Chinese too

    for tag, lo, hi, name in BANDS:
        if args.relation:
            pool = CHUNKS[tag]
            pick = pool if args.shuffle else sorted(pool, key=lambda t: t[0])[:args.per]
        else:
            pool = [(s, r) for s, r in rows if lo <= s < hi]
            if not pool:
                print(f"band {tag}: empty")
                continue
            rng = random.Random(args.seed + int(tag))
            pick = sorted(rng.sample(pool, min(args.per, len(pool))), key=lambda t: t[0])
        if not pick:
            continue

        rh = THUMB + PAD
        W = PAD + len(VIEWS) * (THUMB + PAD) + TEXT_W + PAD
        H = 74 + len(pick) * rh + PAD
        sheet = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(sheet)
        d.rectangle([0, 0, W, 66], fill=(28, 28, 34))
        head = (name if args.relation else
                f"區間 {tag}   CLIP {lo:.3f} – {hi:.3f}   {name}"
                f"      全體共 {len(pool):,} 筆，抽 {len(pick)} 筆")
        d.text((PAD, 16), head, font=f_hdr, fill="white")

        for i, (score, r) in enumerate(pick):
            y = 74 + i * rh
            if i % 2:
                d.rectangle([0, y - PAD // 2, W, y + THUMB + PAD // 2], fill=(246, 246, 248))
            vp = views_for(r["uid"])
            covers = []
            for j, vi in enumerate(VIEWS):
                x = PAD + j * (THUMB + PAD)
                if vi < len(vp) and Path(vp[vi]).is_file():
                    im, cv = thumb(vp[vi])
                    sheet.paste(im, (x, y))
                    covers.append(cv)
                else:
                    d.rectangle([x, y, x + THUMB, y + THUMB], outline=(200, 60, 60))
            cover = 100.0 * sum(covers) / len(covers) if covers else float("nan")
            tx = PAD + len(VIEWS) * (THUMB + PAD)
            rel = r.get("category_relation")
            colour = {"exact": (20, 120, 40), "refined": (150, 110, 0),
                      "divergent": (190, 40, 40)}.get(rel, (60, 60, 60))
            d.text((tx, y + 2), f"{score:.4f}   {r['uid'][:16]}", font=f_lbl, fill=(20, 20, 20))
            # [ULIP2 REVIEWER 2026-08-28] `nan < 5` is False, so an UNDEFINED
            # coverage used to be painted the calm grey that means "this one is
            # fine". The docstring says undefined is not zero; the colour said
            # otherwise, and on a sheet whose whole purpose is judging by eye,
            # red is what the eye catches. The case where the tool does not know
            # must not look like the case where it is happy.
            if cover != cover:                      # NaN: no alpha channel
                d.text((tx + 330, y + 2), "fill    n/a", font=f_lbl,
                       fill=(190, 120, 20))         # amber: unknown, not fine
            else:
                d.text((tx + 330, y + 2), f"fill {cover:5.1f}%", font=f_lbl,
                       fill=(190, 40, 40) if cover < 5 else (60, 60, 60))
            d.text((tx, y + 28), f"LVIS 標籤 : {r.get('lvis_category')}", font=f_txt, fill=(60, 60, 60))
            d.text((tx, y + 52), f"模型認為 : {r.get('category')}   [{rel}]", font=f_txt, fill=colour)
            body = textwrap.wrap(r.get("description") or "", width=68)[:6]
            for k, ln in enumerate(body):
                d.text((tx, y + 84 + k * 24), ln, font=f_txt, fill=(30, 30, 30))

        p = (args.out / f"{args.relation}{'_unconf' if args.confirmed == 'false' else ''}_{tag}.png"
             if args.relation or args.uids_file
             else args.out / f"band{tag}_{lo:.2f}-{hi:.2f}.png")
        sheet.save(p)
        print(f"band {tag}: {len(pool):,} in band -> {p}  ({sheet.width}x{sheet.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
