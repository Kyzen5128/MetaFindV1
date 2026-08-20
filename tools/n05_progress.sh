#!/bin/bash
# n05 v3 全量標註進度。唯讀，隨時可跑。
L=/home/kyzen/data/MetaFind/outputs/logs/n05_v3_full.log
printf '=== %s ===\n' "$(date '+%m-%d %H:%M')"
if pgrep -f metafind.data.annotate_run >/dev/null; then
  echo "  狀態   跑中 (PID $(pgrep -f metafind.data.annotate_run | head -1))"
else
  echo "  狀態   !! 沒在跑"
fi
tail -c 4000 "$L" 2>/dev/null | tr '\r' '\n' | grep -oE '\[ *[0-9,]+/[0-9,]+\][^|]*' | tail -1 | sed 's/^/  進度   /'
grep -c . /home/kyzen/data/MetaFind/outputs/logs/quarantine_n05_annotate.jsonl 2>/dev/null | sed 's/^/  隔離   /'
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  GPU    /'
