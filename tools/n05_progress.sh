#!/bin/bash
# n05 全量標註進度。唯讀，隨時可跑。
# 路徑經 metafind.paths 解析，因此跟著 data/ symlink 走 —— 資料搬遷後不會壞。
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$("$REPO/../miniconda3/envs/MetaFind/bin/python" -c 'from metafind import paths; print(paths.LOGS)' 2>/dev/null)"
[ -d "$LOGS" ] || LOGS="$REPO/data/outputs/logs"
L="$LOGS/${N05_LOG:-n05_v3_full.log}"
printf '=== %s ===\n' "$(date '+%m-%d %H:%M')"
if pgrep -f metafind.data.annotate_run >/dev/null; then
  echo "  狀態   跑中 (PID $(pgrep -f metafind.data.annotate_run | head -1))"
else
  echo "  狀態   !! 沒在跑"
fi
tail -c 4000 "$L" 2>/dev/null | tr '\r' '\n' | grep -oE '\[ *[0-9,]+/[0-9,]+\][^|]*' | tail -1 | sed 's/^/  進度   /'
grep -c . "$LOGS/quarantine_n05_annotate.jsonl" 2>/dev/null | sed 's/^/  隔離   /'
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  GPU    /'
