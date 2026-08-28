#!/bin/bash
# splits -> smoke -> dev ladder 5/10/25, each gated on the last.
#
#   nohup bash tools/chain_overnight.sh > .../logs/chain_overnight.log 2>&1 &
#
# [Kyzen 2026-08-29] 「每個階段做完自動去審 設定個監控自動去看
#  我希望我起床後看到訓練結果」 -- he waived the per-stage ✅ for this chain.
#
# DO NOT EDIT WHILE RUNNING. bash reads a script incrementally by byte offset;
# an edit mid-run shifts the offset and it resumes in the middle of a word.
# That is not hypothetical -- it killed the previous chain with
# `line 36: ath: command not found`.

set -uo pipefail
REPO=/home/kyzen/MetaFindV1
PY=$HOME/miniconda3/envs/MetaFind/bin/python
cd "$REPO" || exit 1
eval "$(${METAFIND_PYTHON:-$PY} -m metafind.paths)"
OUT=$METAFIND_OUTPUTS
LOGS=$OUT/logs
CK=$METAFIND_CHECKPOINTS
RUNS=$OUT/ladder            # one directory per rung; nothing overwrites anything
mkdir -p "$LOGS" "$RUNS"

say() { echo "[$(date '+%F %H:%M:%S')] $*"; }
die() { say "STOPPED: $*"; say "nothing after this point ran"; exit 1; }

if pgrep -f "metafind\.(data|train)\." > /dev/null; then
    die "another metafind stage is already running; refusing to compete for the GPU"
fi

# ---------------------------------------------------------------- 1. splits
say "=== splits ==="
$PY -m metafind.data.splits >> "$LOGS/splits.log" 2>&1 \
    || die "splits exited $? -- read $LOGS/splits.log"
for f in splits.json stage1_protocol.json eval_protocols.json; do
    [ -f "$OUT/$f" ] || die "splits returned 0 but did not write $f"
done
# G3: the protocol names the hyperparameter artifact by hash. Checked HERE so a
# mismatch is reported before the 9.5 GB backbone is loaded.
$PY - <<'CHECK' || die "stage1_protocol does not match the hyperparameter artifact"
import json
from metafind import paths
tr = json.loads((paths.OUTPUTS / "stage1_protocol.json").read_text())
hp = json.loads((paths.OUTPUTS / "stage1_hyperparameters.json").read_text())
assert tr["hyperparameter_config_hash"] == hp["sha256"], (
    tr["hyperparameter_config_hash"][:12], hp["sha256"][:12])
sp = json.loads((paths.OUTPUTS / "splits.json").read_text())["object"]
print(f"  train {len(sp['train']):,}  test {len(sp['test']):,}  "
      f"dev_train {len(sp.get('dev_train', [])):,}  "
      f"dev_val {len(sp.get('dev_val', [])):,}")
CHECK
say "splits OK"

# ----------------------------------------------------------------- 2. smoke
# 200 assets, one epoch. Its job is to prove the wiring holds against a REAL
# loaded backbone -- not that training is correct. Writes stage1_best.smoke200.pt,
# a different name, so it cannot replace a real run's checkpoint.
# [FIXED 2026-08-29] Was `--limit 200`. 200 assets at batch 64 is THREE steps,
# and `stage1.py:900` logs every 20th, so `train_stage1.jsonl` was never written
# and the loss check below died on a missing file -- after a smoke that had
# actually succeeded. 1,500 assets is 24 steps, so at least one metrics row
# lands and the curve check has something to check. A smoke that cannot produce
# a training curve cannot be gated on one.
say "=== smoke: 1,500 assets, 1 epoch ==="
$PY -m metafind.train.stage1 --limit 1500 --epochs 1 >> "$LOGS/stage1_smoke.log" 2>&1 \
    || die "smoke exited $? -- read $LOGS/stage1_smoke.log"
$PY - <<'CHECK' || die "the smoke ran but its checkpoint or its loss is wrong"
import json, torch
from metafind import paths
from metafind.train.stage1 import CKPT_SECTIONS
ck = torch.load(paths.CHECKPOINTS / "stage1.pt", map_location="cpu", weights_only=False)
missing = [s for s in CKPT_SECTIONS if s not in ck]
assert not missing, missing
n = {s: sum(v.numel() for v in ck[s].values()) for s in CKPT_SECTIONS}
print("  " + "  ".join(f"{k.split('_')[0]}={v:,}" for k, v in n.items()))
assert n["backbone_trainable_state"] > 1_000_000, (
    f"backbone section holds only {n['backbone_trainable_state']:,} parameters "
    "-- PointBERT is not in the checkpoint")
# The dev-val file proves the EVALUATION path ran, which the training log does
# not. Both are checked because they fail independently.
dv = [json.loads(l) for l in
      (paths.LOGS / "train_stage1_dev_val.jsonl").read_text().splitlines() if l.strip()]
assert dv, "no dev-val row -- evaluate_dev_val never ran"
print(f"  dev-val R@1 {dv[-1]['mean_R@1']:.4f} over {dv[-1]['n_gallery']} gallery items "
      "(a tiny gallery inflates this; it is a wiring check, not a score)")
rows = [json.loads(l) for l in
        (paths.LOGS / "train_stage1.jsonl").read_text().splitlines() if l.strip()]
assert rows, "no training metric rows -- the smoke was too short to log any"
first, last = rows[0]["loss"], rows[-1]["loss"]
print(f"  loss {first:.4f} -> {last:.4f} over {len(rows)} logged steps")
# One epoch is too short to demand convergence; what it CAN rule out is a loss
# that is NaN or diverging -- each of which means something is wired wrong.
assert last == last, "loss is NaN"
assert last < first * 3, f"loss grew from {first:.4f} to {last:.4f}"
CHECK
say "smoke OK"

# ------------------------------------------------------------- 3. dev ladder
# Each rung is a SEPARATE run and its artifacts are archived under its own name.
# `stage1_best.pt` is a fixed path, so rung 10 would otherwise overwrite rung 5
# and the comparison the ladder exists for would be gone.
for E in 5 10 25; do
    say "=== dev ladder: $E epochs ==="
    $PY -m metafind.train.stage1 --phase dev --epochs "$E" \
        >> "$LOGS/stage1_dev_e$E.log" 2>&1 \
        || die "dev --epochs $E exited $? -- read $LOGS/stage1_dev_e$E.log"
    D=$RUNS/e$E
    mkdir -p "$D"
    for f in stage1_best.pt stage1_best_ckpt.json stage1.pt stage1_ckpt.json; do
        [ -f "$CK/$f" ] && cp -p "$CK/$f" "$D/$f"
    done
    [ -f "$LOGS/train_stage1.jsonl" ] && cp -p "$LOGS/train_stage1.jsonl" "$D/train_stage1.jsonl"
    $PY - "$E" "$D" <<'CHECK' || die "rung $E finished but its record does not hold up"
import json, sys
from pathlib import Path
E, D = int(sys.argv[1]), Path(sys.argv[2])
rec = json.loads((D / "stage1_best_ckpt.json").read_text())
print(f"  rung {E}: phase={rec.get('phase')} n_train={rec.get('n_train'):,} "
      f"n_dev_val={rec.get('n_dev_val'):,} best_epoch={rec.get('epoch')}")
assert rec.get("phase") == "dev", rec.get("phase")
assert rec.get("limit") in (None, 0), f"a --limit run leaked into the ladder: {rec.get('limit')}"
rows = [json.loads(l) for l in (D / "train_stage1.jsonl").read_text().splitlines() if l.strip()]
losses = [r["loss"] for r in rows if "loss" in r]
assert losses and losses[-1] == losses[-1], "final loss is NaN"
print(f"  loss {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} logged steps")
CHECK
    say "rung $E archived -> $D"
done

say "=== splits, smoke and the 5/10/25 dev ladder all finished and were verified ==="
say "next, and NOT automated: pick a rung, then --phase final, then the gallery index"
