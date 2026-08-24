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
# Must fit on the card. See the note at the n05 stage.
N05_MODEL="${N05_MODEL:-/home/kyzen/metafind_out/gemma-4-12B-it}"

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
#
# [FIXED 2026-08-24, Codex] Two defects, both silent:
#   * no `-type f`, so a DIRECTORY named `*.json` counted as an artifact;
#   * `2>/dev/null | wc -l` swallowed find's failure while `wc` still printed a
#     number, so an unreadable or transiently missing symlink produced a
#     confident 0 or a partial count. A single transient 0 at the final n04
#     count makes the completion gate KILL a healthy finished run.
# `count()` now fails loudly instead of returning a plausible number.
# [REWRITTEN 2026-08-24, ULIP2 Reviewer BLOCKER] The previous version called
# `die` from inside here. Every caller reaches this through `x=$(count_nXX)`,
# which is a COMMAND SUBSTITUTION -- a subshell. `exit 1` ended the subshell,
# not the chain, and `say` writes to stdout, so `die`'s two lines were CAPTURED
# INTO THE VARIABLE. Reproduced verbatim: the chain ran to the end with rc=0,
# `before` holding the halt notice, and `$((after-before))` throwing an
# arithmetic error that `set -e`'s deliberate absence let pass.
#
# That was strictly worse than the bug it replaced: the old code returned a
# plausible `0`; this returned a multi-line error string and kept going.
#
# The status has to cross the substitution boundary instead. This function
# complains on STDERR and RETURNS non-zero; a plain assignment propagates the
# substitution's status, so every caller is `x=$(count_nXX) || die`.
#
# Any stderr from find is also a failure, not a warning: one unreadable
# subdirectory makes find exit 0 with a SHORT count, and a short count at the
# final gate kills a healthy finished run.
count() {
    local n err rc
    err=$(mktemp) || return 1
    n=$(find -L "$@" -type f 2>"$err"); rc=$?
    if [[ $rc -ne 0 || -s $err ]]; then
        printf 'counting %s failed (rc=%s): %s\n' "$1" "$rc" "$(head -c 300 "$err")" >&2
        rm -f "$err"
        return 1
    fi
    rm -f "$err"
    printf '%s\n' "$n" | grep -c . || true
}
count_n03() { count "$OUT/pointclouds" -name '*.npz'; }
count_n04() { count "$OUT/renders" -maxdepth 1 -name '*.json'; }
count_n05() { count "$OUT/annotations" -name '*.json'; }

# Re-enter a runner until its count stops climbing. Stalling twice in a row is
# the terminating condition, not a fixed pass budget: assets that fail every
# time are quarantined, and a pass that recovers nothing will never recover
# anything.
drive() {
    local name=$1 counter=$2 log=$3; shift 3
    local before after stalled=0
    for ((pass=1; pass<=MAX_PASSES; pass++)); do
        before=$($counter) || die "$name: cannot count what is on disk; refusing to guess"
        say "$name pass $pass -- starting at $before/$TOTAL"
        "$@" >> "$log" 2>&1
        rc=$?
        after=$($counter) || die "$name: cannot count what is on disk; refusing to guess"
        # [AUDIT 2026-08-23] The exit code used to be discarded, so a stage
        # killed by OOM or dead on an ImportError at second one produced
        # before==after, tripped the stall counter twice, and was announced
        # COMPLETE. A stage that exits non-zero AND produced nothing has not
        # finished, it has failed, and the two must not read the same.
        say "$name pass $pass -- finished at $after/$TOTAL (+$((after-before))) rc=$rc"
        # [FIXED 2026-08-24, Codex BLOCKER] `rc != 0 && after == before` meant a
        # runner that reported a SYSTEMIC failure was ignored whenever the count
        # had moved at all -- and a systemic failure usually happens after some
        # assets succeeded. n04 now returns 3 for exactly that, and the chain
        # would have started another pass, then announced COMPLETE.
        # rc=3 is the runner saying "this run is broken". It is never advisory.
        if [[ $rc -eq 3 ]]; then
            die "$name exited rc=3 -- the runner declared the RUN broken, not the asset. See $log"
        fi
        if [[ $rc -ne 0 && $after -eq $before ]]; then
            die "$name exited rc=$rc having produced nothing -- see $log"
        fi
        [[ $after -ge $TOTAL ]] && { say "$name COMPLETE"; return 0; }
        if [[ $after -eq $before ]]; then
            stalled=$((stalled+1))
            [[ $stalled -ge 2 ]] && { say "$name stalled twice at $after; rest are quarantined"; return 0; }
        else
            stalled=0
        fi
    done
    # [FIXED 2026-08-24, Codex] This returned 0. The stated terminal conditions
    # are "reached TOTAL" and "stalled twice"; running out of passes is NEITHER,
    # and returning success let a stage that was still climbing -- 43,750 of
    # 46,052 with a positive delta -- hand a short corpus to the next stage.
    local final; final=$($counter) || final="unknown"
    say "$name exhausted $MAX_PASSES passes at $final/$TOTAL without reaching TOTAL or stalling"
    say "$name this is not completion. Raise MAX_PASSES or find out why it is still climbing."
    return 1
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
# [AUDIT 2026-08-23] pointed at the stale sampler_version-7, n=1 result, so the
# gate printed UNDERPOWERED forever and waved the chain through. The v8
# measurement (n=138) lives in the file below.
V10=$OUT/v1_0_color0_texture_v8.json
if [[ -f $V10 ]]; then
    say "0c  V1.0 already measured; reusing $V10"
else
    say "0c  V1.0 re-measuring COLOR_0"
    $PY tools/remeasure_color0_texture.py \
        --ulip-npy "$OUT/reference/ulip_npy" --device cpu > "$LOGS/01_v1_0.log" 2>&1
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
    if [[ $VERDICT == CHALLENGED* && $D12_POLICY == user_retained ]]; then
        # [USER DECISION 2026-08-23] He was shown this exact verdict -- 94/138,
        # 4.26 sigma on n=138 (5.30 sigma if ties are dropped as a sign test
        # normally does), two-tailed binomial p = 2.5e-05, and the brightness
        # direction REVERSING against the v6 measurement D-12 rests on -- and
        # answered: 「好啦 沒關係我不想重跑了 同意好了 你跟Master解釋」.
        #
        # So the gate fired, a human read it, and a human overrode it. That is
        # a different thing from the measurement supporting D-12, and it must
        # never be written up as the latter.
        say "0c  D-12 CHALLENGED and the USER has retained it anyway (2026-08-23)."
        say "0c  The corpus keeps the carve-out. The measurement says otherwise."
        say "0c  This is a USER override of a 4.26-sigma contrary result, not agreement."
    elif [[ $VERDICT == CHALLENGED* ]]; then
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
N03=$(count_n03) || die "cannot count n03 output"
say "1  n03 at $N03/$TOTAL"
# [FIXED 2026-08-24, Codex] `$((TOTAL * 90 / 100))` floors: 90% of 46,052 is
# 41,446.8 and the gate admitted 41,446. Compare by multiplying out so the
# stated percentage is the percentage enforced.
[[ $((N03 * 100)) -lt $((TOTAL * 90)) ]] && die "n03 only $N03/$TOTAL (<90%)"

# ---- 2  n04  renders -------------------------------------------------------
say "2  n04 renders"
drive n04 count_n04 "$LOGS/n04_full.log" \
      $PY -m metafind.data.renders --workers 8
N04=$(count_n04) || die "cannot count n04 output"
say "2  n04 at $N04/$TOTAL"
[[ $((N04 * 100)) -lt $((TOTAL * 95)) ]] && die "n04 only $N04/$TOTAL (<95%, gate is 5% failures)"

# ---- 3  n05  annotation ----------------------------------------------------
say "3  n05 annotation"
# [AUDIT 2026-08-23] This called annotate_run with NO --model, so it inherited
# `annotate_run.MODEL_ID` = /mnt/data1/kyzen/models/Qwen3.8-27B. Measured: that
# checkpoint is 56 GB and this card is 32,607 MiB. After 30 hours of n04 the
# chain would have loaded 56 GB of bf16 onto a 32 GB card, OOM'd, retried until
# `drive` stalled twice, and reported n05 "complete" at 0 assets.
#
# The 5 annotations already in the corpus were produced with the 23 GB
# gemma-4-12B-it copy on NVMe -- named here so a resumed run cannot silently
# annotate the rest of the corpus with a different model than the first five.
# Override with N05_MODEL=... if the USER picks the other bake-off arm.
drive n05 count_n05 "$LOGS/n05_full.log" \
      $PY -m metafind.data.annotate_run --model "$N05_MODEL"
N05=$(count_n05) || die "cannot count n05 output"
say "3  n05 at $N05/$TOTAL"
[[ $((N05 * 100)) -lt $((N04 * 90)) ]] && die "n05 only $N05 of $N04 rendered (<90%, gate is 10% failures)"

# ---- 4  hand-off report ----------------------------------------------------
say "4  building the divergent / refined lists for the USER"
$PY tools/report_annotation_divergence.py > "$LOGS/04_report.log" 2>&1 \
    || say "4  report failed -- see $LOGS/04_report.log (the corpus itself is fine)"

say "=========================================================="
say "DONE   n03 $N03   n04 $N04   n05 $N05"
say "review list: $OUT/annotation_review.json"
say "=========================================================="
