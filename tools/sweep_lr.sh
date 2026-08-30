#!/bin/bash
# Stage 1 LR sweep. One GPU, one arm at a time, halt on the first failure.
# Arm 1 (lr7.50e-4_s20260830) already ran; it is in the table so that its line
# lands in the summary too, and the `already trained` branch means it is never
# retrained. That is a guarantee, not a coincidence: the REF check below exits
# if arm 1's record is absent, so the loop cannot reach arm 1 without it.
#
#   bash tools/sweep_lr.sh --i-have-kyzens-approval
#
# It re-execs itself under `setsid nohup` and returns immediately, so the run
# does not die with the shell, the ssh session or the agent that launched it.
#
# DO NOT EDIT WHILE RUNNING. bash reads a script incrementally by byte offset;
# an edit mid-run shifts the offset and it resumes in the middle of a word.
# That killed a previous chain with `line 36: ath: command not found`.

set -uo pipefail

# ---------------------------------------------------------------- THE GUARD
# `workflow/blocks/ULIP2/evidence/lr_sweep_plan.md` heads its own text with
# "NOT AUTHORISED TO RUN": Reviewer R-34 passed the code, execution needs a
# separate check-mark from Kyzen. Roughly 25 min of GPU per arm x 7 arms is not
# something a stray tab-completion should be able to start.
if [ "${1:-}" != "--i-have-kyzens-approval" ]; then
    echo "REFUSING TO RUN: this launches ~3 hours of GPU training (7 arms x ~25 min)."
    echo "It needs Kyzen's explicit approval. When you have it:"
    echo "    bash $0 --i-have-kyzens-approval"
    exit 64
fi

REPO=/home/kyzen/MetaFindV1
PY=$HOME/miniconda3/envs/MetaFind/bin/python
cd "$REPO" || exit 1
eval "$(${METAFIND_PYTHON:-$PY} -m metafind.paths)"
LOGS=$METAFIND_LOGS
CK=$METAFIND_CHECKPOINTS
SUMMARY=$LOGS/sweep_lr_summary.jsonl
RUNNER_LOG=$LOGS/sweep_lr_runner.log
mkdir -p "$LOGS"

# ------------------------------------------------------------- DETACH FIRST
# Requirement from the ULIP-2 handoff, section 8 item 4: a multi-hour job must
# not hang on the lifecycle of the process that started it. `setsid` makes this
# a new session leader, so it survives the terminal, and `nohup` detaches it
# from SIGHUP. Done here rather than left to whoever types the command, because
# "remember to prefix nohup" is the kind of instruction that gets forgotten
# once and costs three hours.
if [ -z "${SWEEP_LR_DETACHED:-}" ]; then
    export SWEEP_LR_DETACHED=1
    setsid nohup bash "$0" "$@" >> "$RUNNER_LOG" 2>&1 < /dev/null &
    echo "sweep detached: pid $!, own session. It will outlive this shell."
    echo "  runner log   tail -f $RUNNER_LOG"
    echo "  per-arm log  $LOGS/sweep_arm<N>.log"
    echo "  summaries    $SUMMARY"
    exit 0
fi

say() { echo "[$(date '+%F %H:%M:%S')] $*"; }
die() { say "STOPPED: $*"; say "every arm after this point did NOT run"; exit 1; }

if pgrep -f "metafind\.(data|train)\." > /dev/null; then
    die "another metafind stage is already running; refusing to compete for the GPU"
fi

# Arm 1's record is the reference every later arm's data pools are checked
# against. DL-033: the invariant is `same arm_config_hash AND same pools_sha256
# => same treatment`, not the hash alone -- so a paired difference computed
# across two different dev_val pools would not be a paired difference.
REF=$CK/sweep_lr/lr7.50e-4_s20260830/stage1_best_ckpt.json
[ -f "$REF" ] || die "arm 1's record is missing at $REF; there is nothing to pair against"

# lr | seed | repeat_index | out-dir tag.
#
# Arms 2..8 in the order `random.Random(20260830)` produced over
# [(lr, seed) for lr in LRS for seed in SEEDS], recorded in lr_sweep_plan.md
# BEFORE any arm ran. Randomised because this machine crashed nine times on
# 2026-08-29 and 500 W is a mitigation, not a repair: a correlated failure must
# not land entirely on one learning rate.
#
# DO NOT REORDER. The order is part of the experimental design.
ARMS=(
  "0.00075 20260830 1 lr7.50e-4_s20260830"
  "0.001   20260816 0 lr1.00e-3_s20260816"
  "0.0005  20260816 0 lr5.00e-4_s20260816"
  "0.00075 20260816 0 lr7.50e-4_s20260816"
  "0.001   20260830 1 lr1.00e-3_s20260830"
  "0.0005  20260830 1 lr5.00e-4_s20260830"
  "0.00025 20260830 1 lr2.50e-4_s20260830"
  "0.00025 20260816 0 lr2.50e-4_s20260816"
)

say "=== Stage 1 LR sweep, arms 1..8, sequential on one GPU ==="
say "reference arm 1: $REF"

N=0
for a in "${ARMS[@]}"; do
    N=$((N + 1))
    read -r LR SEED RI TAG <<< "$a"
    D=$CK/sweep_lr/$TAG

    # Restartable. `stage1.resolve_run_paths` refuses to write into a directory
    # that already holds a checkpoint, so without this branch a restart after a
    # crash would fail on the first finished arm and never reach the unfinished
    # ones. Summarising still runs, so an interrupted sweep converges to a
    # complete jsonl instead of one missing whatever finished before the crash.
    if [ -f "$D/stage1_best_ckpt.json" ]; then
        say "arm $N ($TAG) already has a best-checkpoint record -- not retraining"
    else
        say "=== arm $N/8: lr $LR  seed $SEED  repeat_index $RI  -> sweep_lr/$TAG ==="
        $PY -m metafind.train.stage1 --phase dev --epochs 5 --preload \
            --lr "$LR" --seed "$SEED" --repeat-index "$RI" \
            --out-dir "sweep_lr/$TAG" >> "$LOGS/sweep_arm$N.log" 2>&1 \
            || die "arm $N exited $? -- read $LOGS/sweep_arm$N.log"
    fi

    # A finished process is not a finished arm. The record is read back, the
    # pools are checked against arm 1, and only then does a summary line exist.
    $PY - "$N" "$LR" "$SEED" "$RI" "$TAG" "$D/stage1_best_ckpt.json" "$REF" "$SUMMARY" <<'SUMMARISE' || die "arm $N trained but its record did not hold up -- nothing after it ran"
import json, sys, time
from pathlib import Path

n, lr, seed, ri, tag, rec_p, ref_p, out_p = sys.argv[1:9]
rec = json.loads(Path(rec_p).read_text())
ref = json.loads(Path(ref_p).read_text())

# Appending is not idempotent, and a restart re-walks every arm. Keyed on
# run_id rather than on the arm number, so a genuine re-run of an arm under a
# new run_id is still recorded rather than swallowed.
if Path(out_p).exists():
    seen = {json.loads(l)["run_id"] for l in Path(out_p).read_text().splitlines() if l.strip()}
    if rec["run_id"] in seen:
        print(f"  arm {n} {tag}: already in {out_p}, not appended again")
        raise SystemExit(0)

# `same arm_config_hash AND same pools_sha256 => same treatment` (DL-033). A
# differing pool makes the paired difference this sweep exists to measure
# meaningless, so it halts the sweep rather than being noted afterwards.
for field, get in (
    ("pools_sha256", lambda d: d.get("pools_sha256")),
    ("train_uid_sequence_sha256", lambda d: d["inputs"]["train_uid_sequence_sha256"]),
    ("selection_uid_sequence_sha256", lambda d: d["inputs"]["selection_uid_sequence_sha256"]),
):
    a, b = get(rec), get(ref)
    assert a == b, (
        f"{field} differs from arm 1: {a} vs {b}. The two arms did not train "
        "or select on the same data, so no paired difference between them is valid.")

dv = rec["dev_val"]
# [DL-033] The selector is the mean R@1 over {text, image, pc} on protocol C.
# The other four conditions sit at >=0.98 and divide any real difference by
# seven before it can be read. The seven-cell mean is recorded beside it as the
# guardrail, never as the selector.
THREE = ("text", "image", "pc")
missing = [c for c in THREE if c not in dv]
assert not missing, f"dev_val is missing {missing}; the three-cell mean cannot be formed"
three = sum(dv[c]["R@1"] for c in THREE) / len(THREE)

row = {
    "arm": int(n), "tag": tag, "lr": float(lr), "seed": int(seed),
    "repeat_index": int(ri), "recorded_at": time.time(),
    "run_id": rec["run_id"],
    "arm_config_hash": rec["arm_config_hash"],
    "code_revision": rec["code_revision"],
    "code_dirty": rec["code_dirty"],
    "runtime_source_sha256": rec["runtime_source_sha256"],
    "runtime_source_status": rec["runtime_source_status"],
    "best_epoch": rec["epoch"],
    "mean_R@1_three_cell": three,
    "mean_R@1_seven_cell": dv["mean_R@1"],
    "mean_R@5_seven_cell": dv["mean_R@5"],
    "per_condition_R@1": {c: dv[c]["R@1"] for c in dv if isinstance(dv[c], dict)},
    "n_gallery": dv["n_gallery"],
    "pools_sha256": rec.get("pools_sha256"),
    "checkpoint": rec["uri"],
    "checkpoint_sha256": rec["sha256"],
}
with open(out_p, "a") as f:
    f.write(json.dumps(row) + "\n")
    f.flush()
print(f"  arm {n} {tag}: best epoch {row['best_epoch']}  "
      f"three-cell R@1 {three:.4f}  seven-cell {dv['mean_R@1']:.4f}")
SUMMARISE
    say "arm $N recorded -> $SUMMARY"
done

say "=== all 8 arms accounted for; $SUMMARY holds one line per arm ==="
say "NEXT, and not automated: the paired analysis against delta = 1.0 pp (DL-033)."
say "n=2 per arm. The stopping rule does not run on n=2 -- this round MEASURES."
