#!/bin/bash
# One screen: what is running, what is queued, what is done.
#
#     bash tools/status.sh
#     watch -n 60 bash tools/status.sh     # refresh every minute
#
# chain.log only ever shows the step it is ON, which reads as "nothing is
# happening" while n05 still has hours to go. This reads the actual artifacts
# instead of a log tail.

# Roots come from metafind/paths.py, never spelled here. Six scripts used
# to hardcode the previous machine's /mnt/data1/kyzen/MetaFind, so on any
# other checkout they silently observed an empty directory.
eval "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && ${METAFIND_PYTHON:-python3} -m metafind.paths)"
OUT="$METAFIND_OUTPUTS"
LOGS=$OUT/logs

printf '\n=== %s ===\n\n' "$(date '+%F %H:%M:%S')"

# ---- what is running now
if pgrep -f "metafind.data.annotate_run" > /dev/null; then
    # The log NAME is chosen by whoever launches the run, so naming one here
    # goes stale the next time it changes -- and it had: this tailed
    # n05_v3_full.log, last written 2026-08-20 07:09, and printed its final
    # line "45,955 complete on disk" under 跑中 while the live n05_full.log
    # said [ 32000/46024 ]. Two lines on one screen contradicting each other,
    # and the false one overstated completion, which is the direction that
    # invites someone to start the next stage. Take the newest n05 log instead.
    printf '  跑中   n05 標註   %s\n' "$(tail -1 "$(ls -t "$LOGS"/n05*.log 2>/dev/null | head -1)" 2>/dev/null | tr -s ' ')"
elif pgrep -f "metafind.data.semantic_edges_run" > /dev/null; then
    printf '  跑中   n08 語意邊  %s\n' "$(tail -1 "$LOGS/n08_full.log" 2>/dev/null | tr -s ' ')"
elif pgrep -f "metafind.train.stage1" > /dev/null; then
    printf '  跑中   第一階段訓練 %s\n' "$(tail -1 "$LOGS/n10_smoke.log" 2>/dev/null | tr -s ' ')"
else
    printf '  跑中   （沒有）\n'
fi

# ---- is the queue still armed
# Both names: the script was renamed chain_after_n05 -> chain_to_stage1, and
# matching only the old one made this branch print "腳本不在了" unconditionally,
# which is indistinguishable from the true answer.
#
# The pattern is ANCHORED to an actual invocation -- `bash <path>/chain_*.sh` --
# and not to the bare script name. `pgrep -f` matches whole command lines, so a
# bare name reports "armed" for any process that merely MENTIONS the script: an
# editor, a grep, another agent's tool call. That is not hypothetical; the loose
# version reported 還在 during this very edit, matched against the shell that was
# writing this comment. A false 還在 is the dangerous direction -- it claims a
# GPU chain is queued when nothing is.
QUEUE_PAT='^(/[^ ]*/)?bash +[^ ]*chain_(to_stage1|after_n05)\.sh'
if pgrep -f "$QUEUE_PAT" > /dev/null; then
    printf '  排隊   還在（n05 結束後自動跑 n08 → 第一階段測試）\n'
else
    printf '  排隊   !! 腳本不在了，後面不會自動接\n'
fi

printf '\n  --- 產出 ---\n'
count() { printf '  %-22s %s\n' "$1" "$2"; }
# Count the RECORD (.json), not every file. Each cloud is .npz + .json, each
# render is a directory + .json, each ProcTHOR asset likewise -- so a naive
# `ls | wc -l` doubles them, which is exactly the kind of number that reads as
# plausible and is wrong.
#
# -L because `data/outputs/{pointclouds,renders,annotations,embeddings}` are
# symlinks into /home/kyzen/metafind_out. Without it find never descends: it
# tests the link itself against *.json, fails, and reports 0 -- a zero that
# reads as "this stage has produced nothing" while 46,052 records sit on disk.
# The two callers that pass a REAL directory (scene_graphs 12,000 and
# procthor_modalities 1,467) count identically with and without -L, so this is
# one fix at the fork rather than two at the call sites.
recs() { find -L "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
# Count the records n05 ITSELF calls current. Keying on prompt_version was
# wrong twice over: it was pinned to 3 while the live corpus writes 8, so this
# read 0 against 32,000 finished records; and prompt_version alone cannot
# express the validator and schema axes -- annotate_run.py:195-214 keys
# completeness on the annotation contract id for exactly that reason. Import
# the id rather than spelling it, so this cannot go stale a third time.
# glob.glob resolves the directory component through symlinks, so this helper
# was never affected by the -L bug above; its zero had a separate cause.
current_count() { (cd "$METAFIND_REPO" && "${METAFIND_PYTHON:-python3}" -c '
import json, glob, sys
from metafind.data.annotate import annotation_contract_id
want = annotation_contract_id()
print(sum(1 for f in glob.glob(sys.argv[1] + "/*.json")
          if json.load(open(f)).get("annotation_contract") == want))' "$1"); }
count "n02 模型下載"   "$(find "$OUT/../datasets" -name '*.glb' 2>/dev/null | wc -l) 個"
count "n03 點雲"       "$(recs "$OUT/pointclouds") 個"
count "n04 渲染"       "$(recs "$OUT/renders") 個"
# n05 的 sidecar 有兩代 schema（prompt_version 1 = 舊的 5 欄位，
# 3 = 論文 Figure 2 的 13 欄位）。光數檔案會把還沒重跑的 v1 算成完成，
# 這正是「看起來做完了、其實還有幾小時」的那種讀數。
# Denominator DERIVED from n04's index, which IS n05's work list
# (annotate_run.py reads renders_index.jsonl to build it). The literal 45,955
# that stood here was a previous corpus size; the current target is 46,024. A
# finished run printed against the old number reads as "nearly done" when it is
# done -- or, if the quarantine count falls, as more than 100%.
# This and `recs "$OUT/renders"` above are two DERIVATIONS of one source, not
# two sources: the index is rebuilt from the sidecars filtered by is_complete,
# so index <= sidecars by construction. If the two ever disagree on screen that
# is a real condition being reported, not a bug to reconcile.
# The braces matter: `wc -l < missing 2>/dev/null` does NOT suppress the
# error, because the redirect fails in the SHELL before wc starts, so the
# message is bash's and the 2>/dev/null attached to wc never sees it. The
# fallback still worked -- only the suppression was theatre.
N05_TARGET=$({ wc -l < "$LOGS/renders_index.jsonl"; } 2>/dev/null || echo '?')
count "n05 標註"       "$(current_count "$OUT/annotations") / $N05_TARGET"
count "n07 場景圖"     "$(recs "$OUT/scene_graphs") 個"
count "n07b 資產模態"  "$(recs "$OUT/procthor_modalities") / 1,467"
[ -f "$OUT/sem_edge_cache.json" ] \
    && count "n08 語意邊" "已完成" || count "n08 語意邊" "還沒跑（排隊中）"
[ -f "$OUT/../checkpoints/stage1.pt" ] \
    && count "n10 第一階段" "已有 checkpoint" || count "n10 第一階段" "還沒跑（排隊中）"

printf '\n  --- GPU ---\n'
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
           --format=csv,noheader | sed 's/^/  /'

printf '\n  --- 排隊腳本最後 3 行 ---\n'
tail -3 "$LOGS/chain.log" 2>/dev/null | sed 's/^/  /'
printf '\n'
