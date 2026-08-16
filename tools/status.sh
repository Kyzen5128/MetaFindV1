#!/bin/bash
# One screen: what is running, what is queued, what is done.
#
#     bash tools/status.sh
#     watch -n 60 bash tools/status.sh     # refresh every minute
#
# chain.log only ever shows the step it is ON, which reads as "nothing is
# happening" while n05 still has hours to go. This reads the actual artifacts
# instead of a log tail.

OUT=/mnt/data1/kyzen/MetaFind/outputs
LOGS=$OUT/logs

printf '\n=== %s ===\n\n' "$(date '+%F %H:%M:%S')"

# ---- what is running now
if pgrep -f "metafind.data.annotate_run" > /dev/null; then
    printf '  跑中   n05 標註   %s\n' "$(tail -1 "$LOGS/n05_full.log" 2>/dev/null | tr -s ' ')"
elif pgrep -f "metafind.data.semantic_edges_run" > /dev/null; then
    printf '  跑中   n08 語意邊  %s\n' "$(tail -1 "$LOGS/n08_full.log" 2>/dev/null | tr -s ' ')"
elif pgrep -f "metafind.train.stage1" > /dev/null; then
    printf '  跑中   第一階段訓練 %s\n' "$(tail -1 "$LOGS/n10_smoke.log" 2>/dev/null | tr -s ' ')"
else
    printf '  跑中   （沒有）\n'
fi

# ---- is the queue still armed
if pgrep -f "chain_after_n05" > /dev/null; then
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
recs() { find "$1" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l; }
count "n02 模型下載"   "$(find "$OUT/../datasets" -name '*.glb' 2>/dev/null | wc -l) 個"
count "n03 點雲"       "$(recs "$OUT/pointclouds") 個"
count "n04 渲染"       "$(recs "$OUT/renders") 個"
count "n05 標註"       "$(recs "$OUT/annotations") / 45,955"
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
