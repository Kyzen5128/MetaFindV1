#!/bin/bash
# ULIP-2 block: run n03 -> n04 -> n05 unattended, to annotation completion.
#
#   nohup bash tools/run_ulip_full.sh > data/outputs/logs/ulip_full.log 2>&1 &
#
# Every stage is gated on the one before it FINISHING AND BEING CHECKED. A chain
# that runs the next step on a failed predecessor turns one bad run into three.
#
# `set -e` is deliberately NOT used: each step reports its own verdict and the
# script decides. An implicit exit would leave no record of what stopped it.
#
# D12_POLICY controls the one research decision this chain can meet:
#   measurement  -- let V1.0 decide the COLOR_0 rule and keep going (needs USER)
#   stop         -- halt and wait for the USER if V1.0 challenges D-12 (default)

set -uo pipefail

REPO=/home/kyzen/MetaFindV1
PY=/home/kyzen/miniconda3/envs/MetaFind/bin/python
D12_POLICY="${D12_POLICY:-stop}"
TOTAL=46052
# Each runner skips work it already has, so re-entering is cheap and safe. This
# is the whole unattended-recovery story: a crash at hour 40 costs one asset,
# not the run.
MAX_PASSES="${MAX_PASSES:-6}"

cd "$REPO" || exit 1
eval "$($PY -m metafind.paths)"
# `data/` is a symlink, and GNU find silently returns NOTHING for a path whose
# start point traverses one -- no error, just an empty result. Every counter
# below would read 0, the stall detector would call n03 "done" after two empty
# passes, and the 90% gate would kill a run that had actually succeeded.
# Resolve once, here, rather than remembering `-L` at four call sites.
OUT=$(readlink -f "$METAFIND_OUTPUTS")
LOGS=$OUT/logs
mkdir -p "$LOGS"

say()  { echo "[$(date '+%F %H:%M:%S')] $*"; }
die()  { say "STOP -- $*"; say "chain halted; nothing after this ran"; exit 1; }

# -L is REQUIRED, not defensive: outputs/{pointclouds,renders,annotations} are
# each a symlink to /home/kyzen/metafind_out/. Plain `find` refuses to descend a
# symlinked start point and returns EMPTY WITH NO ERROR, so every counter reads
# 0, the stall detector calls n03 finished after two empty passes, and the 90%
# gate kills a run that actually succeeded. Verified: find 0, find -L 7942.
count_n03() { find -L "$OUT/pointclouds" -name '*.npz' 2>/dev/null | wc -l; }
count_n04() { find -L "$OUT/renders" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
count_n05() { find -L "$OUT/annotations" -name '*.json' 2>/dev/null | wc -l; }

# Re-enter a runner until its count stops climbing. Stalling twice in a row is
# the terminating condition, not a fixed pass budget: assets that fail every
# time are quarantined, and a pass that recovers nothing will never recover
# anything.
drive() {
    local name=$1 counter=$2 log=$3; shift 3
    local before after stalled=0
    for ((pass=1; pass<=MAX_PASSES; pass++)); do
        before=$($counter)
        say "$name pass $pass -- starting at $before/$TOTAL"
        "$@" >> "$log" 2>&1
        after=$($counter)
        say "$name pass $pass -- finished at $after/$TOTAL (+$((after-before)))"
        [[ $after -ge $TOTAL ]] && { say "$name COMPLETE"; return 0; }
        if [[ $after -eq $before ]]; then
            stalled=$((stalled+1))
            [[ $stalled -ge 2 ]] && { say "$name stalled twice at $after; rest are quarantined"; return 0; }
        else
            stalled=0
        fi
    done
    say "$name exhausted $MAX_PASSES passes at $(  $counter)/$TOTAL"
    return 0
}

say "=========================================================="
say "ULIP-2 full run   git $(git rev-parse --short HEAD)   D12_POLICY=$D12_POLICY"
say "=========================================================="

# ---- 0a  tests -------------------------------------------------------------
say "0a  pytest"
$PY -m pytest -q > "$LOGS/00_pytest.log" 2>&1 \
    || die "tests failed -- see $LOGS/00_pytest.log"
say "0a  OK  $(tail -1 "$LOGS/00_pytest.log")"

# ---- 0b  vendored renderer is byte-identical to upstream -------------------
say "0b  vendored OpenShape script"
UPSTREAM=$(find /home/kyzen/upstream/OpenShape -name render_single_glb.py 2>/dev/null | head -1)
if [[ -n "$UPSTREAM" ]]; then
    diff -q metafind/vendor/openshape/render_single_glb.py "$UPSTREAM" \
        || die "vendored renderer differs from $UPSTREAM"
    say "0b  OK  byte-identical to $UPSTREAM"
else
    say "0b  SKIP  upstream copy not on disk; COMMIT=$(cat metafind/vendor/openshape/COMMIT)"
fi

# ---- 0c  V1.0  COLOR_0 re-measure (decides the n03 colour rule) ------------
V10=$OUT/v1_0_color0_texture.json
if [[ -f $V10 ]]; then
    say "0c  V1.0 already measured; reusing $V10"
else
    say "0c  V1.0 re-measuring COLOR_0"
    $PY tools/remeasure_color0_texture.py \
        --ulip-npy "$OUT/reference/ulip_npy/000-009" > "$LOGS/01_v1_0.log" 2>&1
    rc=$?
    [[ $rc -eq 2 ]] && say "0c  empty population -- D-12 stands unchanged"
    [[ $rc -ne 0 && $rc -ne 2 ]] && die "V1.0 failed rc=$rc -- see $LOGS/01_v1_0.log"
fi

if [[ -f $V10 ]]; then
    VERDICT=$($PY - "$V10" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
rows = d["rows"]; n = len(rows)
wins = sum(1 for r in rows if r["cos_modulated"] > r["cos_not_modulated"])
se = (n ** 0.5) / 2
TARGET = 37   # the pre-registered population, PIPELINE_FINAL.md:159
if n < TARGET:
    # Only one of ULIP's 160 shards is on this machine, so the overlap that
    # feeds this population is a fraction of what the measurement was designed
    # for. Printing CONFIRMED off n=1 would turn missing data into a result.
    print(f"UNDERPOWERED {wins}/{n} (needs {TARGET}) -- D-12 stands UNRESOLVED")
elif abs(wins - n / 2) < se:
    print(f"NOISE {wins}/{n}")          # R-11: D-12 stands
elif wins > n / 2:
    print(f"CHALLENGED {wins}/{n}")     # modulated wins -- D-12 is wrong
else:
    print(f"CONFIRMED {wins}/{n}")      # not-modulated wins -- D-12 holds
PY
)
    say "0c  V1.0 verdict: $VERDICT"
    if [[ $VERDICT == CHALLENGED* ]]; then
        if [[ $D12_POLICY == measurement ]]; then
            say "0c  D-12 CHALLENGED and policy=measurement."
            say "0c  !! the sampler still ships the D-12 rule; switching it is a"
            say "0c  !! code change, not a flag. Halting so it is made deliberately."
            die "D-12 challenged -- USER must approve the sampler change"
        fi
        die "D-12 challenged by V1.0 -- USER decision required (see $LOGS/01_v1_0.log)"
    fi
fi

# ---- 1  n03  point clouds --------------------------------------------------
say "1  n03 point clouds"
drive n03 count_n03 "$LOGS/n03_full.log" \
      $PY -m metafind.data.pointclouds --workers 8
N03=$(count_n03); say "1  n03 at $N03/$TOTAL"
[[ $N03 -lt $((TOTAL * 90 / 100)) ]] && die "n03 only $N03/$TOTAL (<90%)"

# ---- 2  n04  renders -------------------------------------------------------
say "2  n04 renders"
drive n04 count_n04 "$LOGS/n04_full.log" \
      $PY -m metafind.data.renders --workers 8
N04=$(count_n04); say "2  n04 at $N04/$TOTAL"
[[ $N04 -lt $((TOTAL * 95 / 100)) ]] && die "n04 only $N04/$TOTAL (<95%, gate is 5% failures)"

# ---- 3  n05  annotation ----------------------------------------------------
say "3  n05 annotation"
drive n05 count_n05 "$LOGS/n05_full.log" \
      $PY -m metafind.data.annotate_run
N05=$(count_n05); say "3  n05 at $N05/$TOTAL"
[[ $N05 -lt $((N04 * 90 / 100)) ]] && die "n05 only $N05 of $N04 rendered (<90%, gate is 10% failures)"

# ---- 4  hand-off report ----------------------------------------------------
say "4  building the divergent / refined lists for the USER"
$PY tools/report_annotation_divergence.py > "$LOGS/04_report.log" 2>&1 \
    || say "4  report failed -- see $LOGS/04_report.log (the corpus itself is fine)"

say "=========================================================="
say "DONE   n03 $N03   n04 $N04   n05 $N05"
say "review list: $OUT/annotation_review.json"
say "=========================================================="
