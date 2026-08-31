#!/usr/bin/env python3
"""n09: is the 80/20 split real, reproducible, and what the paper describes?

`3experiments.tex:10` says only this: "In both datasets, we allocate 80% of the
data for training and reserve the remaining 20% for testing." One sentence, no
seed, no mention of stratification, no statement about whether the 20% is drawn
uniformly or per category. So this checks the things the sentence DOES pin down,
and reports the things it leaves open rather than assuming them.

Checked:
  1. the counts, and whether 80/20 is exact or rounded
  2. disjointness and duplicate uids
  3. reproducibility -- re-derive the split from the recorded seed and compare
  4. category balance, since a uniform draw over 1,156 LVIS categories leaves
     some of them with few or no test items and the paper does not say whether
     that was allowed
  5. the same for the dev_train / dev_val sub-split, which is ours and not the
     paper's
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from metafind import paths  # noqa: E402
from metafind.data import splits as S  # noqa: E402

LVIS = pathlib.Path("/home/kyzen/upstream/ULIP_run/data/objaverse-lvis/"
                    "objaverse_lvis_metadata.json")
OUT = REPO / "output" / "look" / "audit_splits.json"


def main() -> int:
    d = json.loads((paths.OUTPUTS / "splits.json").read_text())
    o = d["object"]
    tr, te = o["train"], o["test"]
    dtr, dva = o["dev_train"], o["dev_val"]
    total = d["admitted_total"]
    res = {"recorded": {k: d[k] for k in d if k != "object"}}

    print("=== 1. 數量 ===")
    print(f"  admitted_total   {total:,}")
    print(f"  train            {len(tr):,}   {len(tr) / total * 100:.4f}%")
    print(f"  test             {len(te):,}   {len(te) / total * 100:.4f}%")
    print(f"  train + test     {len(tr) + len(te):,}")
    exact = total * d["train_fraction"]
    print(f"  80% of {total:,} = {exact:.1f}  -> train is {len(tr):,} "
          f"({'ceil' if len(tr) > exact else 'floor'})")
    res["counts"] = {"total": total, "train": len(tr), "test": len(te),
                     "train_pct": round(len(tr) / total * 100, 4),
                     "exact_80pct": exact}

    print("\n=== 2. 互斥與重複 ===")
    checks = {
        "train ∩ test": len(set(tr) & set(te)),
        "train 內重複": len(tr) - len(set(tr)),
        "test 內重複": len(te) - len(set(te)),
        "train+test 是否等於 admitted": int(set(tr) | set(te) == set(S.admitted_uids())),
        "dev_train ∩ dev_val": len(set(dtr) & set(dva)),
        "dev_train+dev_val 是否等於 train": int(set(dtr) | set(dva) == set(tr)),
        "dev 池 ∩ test": len((set(dtr) | set(dva)) & set(te)),
    }
    for k, v in checks.items():
        print(f"  {k:34s} {v}")
    res["integrity"] = checks

    print("\n=== 3. 可重現性（用記錄的 seed 重算一次）===")
    uids = S.admitted_uids()
    rng = np.random.default_rng(d["split_seed"])
    perm = rng.permutation(len(uids))
    n_tr = int(round(len(uids) * d["train_fraction"]))
    redo_tr = sorted(uids[i] for i in perm[:n_tr])
    redo_te = sorted(uids[i] for i in perm[n_tr:])
    same = (redo_tr == sorted(tr)) and (redo_te == sorted(te))
    print(f"  seed {d['split_seed']}, fraction {d['train_fraction']}")
    print(f"  重算 train {len(redo_tr):,} / test {len(redo_te):,}")
    print(f"  跟記錄的完全相同: {same}")
    if not same:
        print("  !! 不同 -- 這支腳本猜的重建方式跟 splits.py 不一樣，")
        print("     不代表切分有問題。以 splits.py 的實作為準。")
    res["reproducible_by_this_script"] = bool(same)

    print("\n=== 4. 類別平衡（LVIS 1,156 類）===")
    meta = json.loads(LVIS.read_text())
    v2k = meta["value_to_key_mapping"]
    ctr = collections.Counter(v2k[u] for u in tr if u in v2k)
    cte = collections.Counter(v2k[u] for u in te if u in v2k)
    cats = set(ctr) | set(cte)
    zero_te = [c for c in cats if cte.get(c, 0) == 0]
    zero_tr = [c for c in cats if ctr.get(c, 0) == 0]
    ratios = np.array([cte.get(c, 0) / (ctr.get(c, 0) + cte.get(c, 0))
                       for c in cats])
    print(f"  出現在語料中的類別      {len(cats):,} / 1,156")
    print(f"  test 裡一個都沒有的類別  {len(zero_te):,}")
    print(f"  train 裡一個都沒有的類別 {len(zero_tr):,}")
    print(f"  每類 test 佔比  平均 {ratios.mean() * 100:.2f}%  "
          f"中位 {np.median(ratios) * 100:.2f}%  "
          f"最低 {ratios.min() * 100:.2f}%  最高 {ratios.max() * 100:.2f}%")
    small = sorted(cats, key=lambda c: ctr.get(c, 0) + cte.get(c, 0))[:5]
    print("  最小的 5 個類別 (train/test):")
    for c in small:
        print(f"    {c[:36]:38s} {ctr.get(c, 0):4d} / {cte.get(c, 0):3d}")
    res["category_balance"] = {
        "categories_present": len(cats),
        "zero_in_test": len(zero_te), "zero_in_train": len(zero_tr),
        "per_category_test_share": {
            "mean": round(float(ratios.mean()) * 100, 3),
            "median": round(float(np.median(ratios)) * 100, 3),
            "min": round(float(ratios.min()) * 100, 3),
            "max": round(float(ratios.max()) * 100, 3)},
        "note": "a uniform draw is NOT stratified; the paper does not say which",
    }

    print("\n=== 5. dev 子切分（我們的，論文沒有）===")
    print(f"  dev_train {len(dtr):,}   dev_val {len(dva):,}   "
          f"dev_val 佔 train 的 {len(dva) / len(tr) * 100:.3f}% "
          f"(記錄值 {d['dev_val_fraction'] * 100:.1f}%)")
    res["dev"] = {"dev_train": len(dtr), "dev_val": len(dva),
                  "dev_val_share_of_train": round(len(dva) / len(tr) * 100, 3),
                  "recorded_fraction": d["dev_val_fraction"]}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
