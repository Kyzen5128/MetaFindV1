#!/bin/bash
# n03 v8 has now died silently twice -- no traceback, no OOM line, the log just
# stops mid-progress. Until the cause is known, re-enter it: the runner skips
# what it already has, so a restart costs seconds and the corpus converges.
# The exit status is recorded so the next death says WHY instead of nothing.
cd /home/kyzen/MetaFindV1 || exit 1
PY=/home/kyzen/miniconda3/envs/MetaFind/bin/python
LOG=data/outputs/logs/n03_v8.log
for attempt in $(seq 1 12); do
    left=$($PY -c "
import json,glob
n=sum(1 for f in glob.glob('/home/kyzen/metafind_out/pointclouds/*.json')
      if json.load(open(f)).get('sampler_version')==8)
print(46052-n)")
    echo "[$(date '+%F %H:%M:%S')] attempt $attempt -- $left still on v7" >> "$LOG"
    [ "$left" -le 0 ] && { echo "[$(date '+%F %H:%M:%S')] n03 v8 COMPLETE" >> "$LOG"; exit 0; }
    nice -n10 $PY -m metafind.data.pointclouds --workers 8 >> "$LOG" 2>&1
    echo "[$(date '+%F %H:%M:%S')] attempt $attempt exited rc=$?" >> "$LOG"
done
echo "[$(date '+%F %H:%M:%S')] gave up after 12 attempts" >> "$LOG"
