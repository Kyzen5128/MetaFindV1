#!/bin/bash
# Sample the machine twice a second and FSYNC every sample, so the last line
# written survives a hard power-off.
#
# [Kyzen 2026-08-29] Three hard crashes during Stage 1 (boots -3/-2/-1 in
# `journalctl --list-boots`, all unclean). Two hypotheses were argued and
# NEITHER was measured while a crash happened: host RAM exhaustion, and a GPU
# power transient. Every reading so far was taken during runs that SURVIVED,
# which is the wrong sample. This records the ones that do not.
#
# `sync` per line is the whole point. A buffered log loses the seconds before a
# power-off -- exactly the seconds that answer the question.
OUT=${1:-/home/kyzen/MetaFindV1/data/outputs/logs/crash_recorder.csv}
[ -f "$OUT" ] || echo "ts,ram_used_mb,ram_avail_mb,swap_used_mb,gpu_mem_mb,gpu_power_w,gpu_temp_c,proc_rss_mb" > "$OUT"
while true; do
    read -r used avail <<< "$(free -m | awk '/^Mem/{print $3, $7}')"
    swp=$(free -m | awk '/^Swap|^置換/{print $3}')
    read -r gmem gpow gtmp <<< "$(nvidia-smi --query-gpu=memory.used,power.draw,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ',')"
    rss=$(ps -eo rss,args --no-headers 2>/dev/null | grep -a "[m]etafind.train.stage1" | awk '{s+=$1} END{print int(s/1024)}')
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$(date +%s.%N)" "$used" "$avail" "${swp:-0}" \
        "${gmem:-0}" "${gpow:-0}" "${gtmp:-0}" "${rss:-0}" >> "$OUT"
    sync "$OUT" 2>/dev/null
    sleep 0.5
done
