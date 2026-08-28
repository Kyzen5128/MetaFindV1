"""Which label do the RENDERS support: the LVIS one, or the model's?

[USER 2026-08-28] 「好可以」 -- approved after the sheets showed, by eye, that
the model looked right and the catalogue looked wrong on the `divergent` rows.
This turns that impression into a rate.

Why this is a different question from `clip_score`
--------------------------------------------------
The recorded `clip_score` scores a whole DESCRIPTION against the views, and
measured 2026-08-28 it does not rank annotation quality at all: the proportion
of `divergent` rows RISES with the score (6.0% in 0.00-0.15, 19.4% in
0.30-0.34). A two-way forced choice between two short LABELS is CLIP's actual
zero-shot classification setting, which is a different and much better-supported
use of the same model.

The instrument is measured before the objects
---------------------------------------------
`--control N` duels the agreed label against a random other LVIS label on rows
where the model AGREED with the catalogue. If CLIP cannot win that, it cannot
adjudicate anything here and the divergent numbers must be thrown away. Run it
first; it is reported first.

THREE LIMITS ON WHAT THE WIN RATE MEANS. Reviewed 2026-08-28 by Codex, ULIP2
REVIEWER and MASTER; all three found the same shape and none of them is a bug in
this file. They bound the CLAIM, so they are recorded here rather than in a
message that scrolls away.

1. The win rate is NOT a catalogue-error rate. `category_relation == "divergent"`
   is a string comparison (`annotate.category_relation`, annotate.py:378) and it
   deliberately mixes real mislabels with legitimate refinements that change
   vocabulary: "motor vehicle" -> "pickup truck" is divergent, and the annotation
   prompt (annotate.py:788) explicitly ASKS for that refinement. CLIP reliably
   prefers the more specific true phrase, so every such row counts as a model win
   while both labels are correct. Measured 2026-08-28: 947 of 8,313 divergent
   rows (11.4%) have one label as a substring of the other.

2. The control is EASIER than the test, in two independent ways.
   (a) The decoy is drawn uniformly from the 1,156-term vocabulary, so it is
       usually unrelated -- "wheel vs parasol". The test is "wheel vs alarm
       clock", a competitor a VLM produced after looking at these very images,
       i.e. plausible by construction.
   (b) The control rows are `exact` ones, which are by definition the
       unambiguous objects. The instrument is calibrated on the easy subgroup
       and then applied to the hard one.
   A high control number therefore establishes only that CLIP is not broken. It
   does NOT establish that CLIP can separate confusable pairs, which is the only
   thing the test asks of it. A harder control -- decoy drawn as the nearest
   other label by CLIP text embedding, or from the same supercategory -- costs
   the same forward passes and is the honest version; it is not implemented here.
   `exact` also does not imply `identity_confirmed`: Codex found a row where the
   catalogue label was repeated with `identity_confirmed=false`, so a "true"
   label in the control may itself be wrong.

3. A tie goes to LVIS (`s[1] > s[0]` is strict). Because the ACTION taken on this
   output is exclusion, a tie counts toward deleting an asset rather than keeping
   it, which is the wrong direction for a destructive step. Float32 cosines make
   exact ties vanishingly rare, so this is recorded rather than changed -- but it
   is recorded, because "rare" is not "impossible" and the next reader should not
   have to re-derive it.

Both prompt forms are scored in the same forward pass ("a photo of a X" and the
bare "X"), so a result that depends on the template is visible instead of being
a choice made silently.

Read-only. Writes one JSON of per-asset verdicts; changes no annotation.
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metafind import paths                                    # noqa: E402
from metafind.data.describe_rank import (                      # noqa: E402
    RANKER_MODEL, RANKER_VERSION, score_candidates,
)

PAREN = re.compile(r"\s*\(.*?\)")
TEMPLATE = "a photo of a {}"


def clean(name: str) -> str:
    """LVIS parenthetical disambiguators are annotation apparatus, not the name
    -- `annotate._tokens` already treats them that way, so the duel must too or
    the catalogue side is handicapped by text no human would say."""
    return PAREN.sub("", name or "").strip()


def views_for(uid: str) -> list[str]:
    sc = paths.OUTPUTS / "renders" / f"{uid}.json"
    if not sc.is_file():
        return []
    try:
        return json.loads(sc.read_text()).get("view_paths", [])
    except (OSError, json.JSONDecodeError):
        return []


def duel(uid: str, a: str, b: str) -> dict | None:
    """`a` = LVIS side, `b` = model side. One image forward, four texts."""
    vp = views_for(uid)
    if not vp:
        return None
    ca, cb = clean(a), clean(b)
    if not ca or not cb:
        return None
    s = score_candidates(vp, [TEMPLATE.format(ca), TEMPLATE.format(cb), ca, cb])
    return {"uid": uid, "lvis": a, "model": b,
            "tmpl_lvis": s[0], "tmpl_model": s[1],
            "bare_lvis": s[2], "bare_model": s[3],
            "tmpl_winner": "model" if s[1] > s[0] else "lvis",
            "bare_winner": "model" if s[3] > s[2] else "lvis"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--relation", default="divergent")
    ap.add_argument("--confirmed", choices=["true", "false"])
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--control", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--out", type=Path,
                    default=paths.OUTPUTS / "logs" / "label_duel.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            (paths.OUTPUTS / "logs" / "annotations_index.jsonl").open() if l.strip()]
    vocab = sorted({r["lvis_category"] for r in rows if r.get("lvis_category")})
    rng = random.Random(args.seed)

    # [Codex finding 3] The artifact used to carry only the two result lists, so
    # nothing in it said which model scored them, with which seed, over which
    # population -- and `--out` defaults to one path, so a second run with a
    # different `--control` silently replaced the first with a file of identical
    # shape. A number that decided 21 exclusions has to say where it came from.
    #
    # `control` is in here because the RNG is shared: the control rows and their
    # decoys are drawn BEFORE `rng.shuffle(sel)`, so changing `--control` changes
    # which test rows `--limit` lands on. The seed alone does not reproduce this
    # file.
    out = {
        "provenance": {
            "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                         capture_output=True, text=True,
                                         cwd=Path(__file__).resolve().parents[1]
                                         ).stdout.strip() or None,
            "ranker_model": RANKER_MODEL,
            "ranker_version": RANKER_VERSION,
            "template": TEMPLATE,
            "views_scored": "all views in the n04 sidecar, mean cosine per candidate",
            "seed": args.seed,
            "control_n_requested": args.control,
            "limit": args.limit,
            "relation": args.relation,
            "confirmed_filter": args.confirmed,
            "index": str(paths.OUTPUTS / "logs" / "annotations_index.jsonl"),
            "index_rows": len(rows),
            "vocab_size": len(vocab),
            "tie_goes_to": "lvis",
            "control_decoy": "uniform over the LVIS vocabulary, excluding the "
                             "true label -- see LIMIT 2 in the module docstring",
            "control_population": "category_relation == 'exact'",
        },
        "control": [], "test": [], "vocab_size": len(vocab),
    }

    # ---- the instrument, first -------------------------------------------
    agreed = [r for r in rows if r.get("category_relation") == "exact"]
    for r in rng.sample(agreed, min(args.control, len(agreed))):
        true = r["lvis_category"]
        wrong = rng.choice([v for v in vocab if v != true])
        d = duel(r["uid"], true, wrong)          # `model` slot holds the DECOY
        if d:
            d["decoy"] = True
            out["control"].append(d)
    c = out["control"]
    if c:
        t = 100 * sum(1 for d in c if d["tmpl_winner"] == "lvis") / len(c)
        b = 100 * sum(1 for d in c if d["bare_winner"] == "lvis") / len(c)
        print(f"[control] {len(c)} 筆：真標籤 vs 隨機錯標籤")
        print(f"          真標籤勝率  樣板 {t:.1f}%   裸詞 {b:.1f}%   (亂猜=50%)",
              flush=True)

    # ---- the actual question ---------------------------------------------
    sel = [r for r in rows if r.get("category_relation") == args.relation
           and (args.confirmed is None
                or r.get("identity_confirmed") is (args.confirmed == "true"))]
    rng.shuffle(sel)
    sel = sel[:args.limit]
    t0 = time.time()
    for i, r in enumerate(sel, 1):
        d = duel(r["uid"], r["lvis_category"], r["category"])
        if d:
            d["identity_confirmed"] = r.get("identity_confirmed")
            out["test"].append(d)
        if i % 100 == 0:
            w = 100 * sum(1 for x in out["test"] if x["tmpl_winner"] == "model") / len(out["test"])
            print(f"  [{i}/{len(sel)}] 模型勝率 {w:.1f}%  "
                  f"{(time.time() - t0) / i:.2f}s/筆", flush=True)
    t_ = out["test"]
    if t_:
        mt = 100 * sum(1 for d in t_ if d["tmpl_winner"] == "model") / len(t_)
        mb = 100 * sum(1 for d in t_ if d["bare_winner"] == "model") / len(t_)
        agree = 100 * sum(1 for d in t_ if d["tmpl_winner"] == d["bare_winner"]) / len(t_)
        print(f"\n[test] {len(t_)} 筆 {args.relation}"
              + ("" if args.confirmed is None else f" · identity_confirmed={args.confirmed}"))
        print(f"       模型標籤勝率  樣板 {mt:.1f}%   裸詞 {mb:.1f}%")
        print(f"       兩種寫法給出同一個贏家: {agree:.1f}%")
    args.out.write_text(json.dumps(out))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
