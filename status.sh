#!/usr/bin/env bash
# MetaFind 進度總覽
#
#   bash status.sh          一次性快照
#   bash status.sh -w       每 30 秒自動更新
#
# 「完成度」以 sidecar 為準（metafind.data.verify_pointclouds），不看 log 的摘要 ——
# 摘要是衍生資料，摘要程式有 bug 時不該讓真實的缺件隱形。

set -uo pipefail
ROOT=/mnt/data1/kyzen/MetaFind
LOG="$ROOT/runs/progress/fetch_pointclouds.log"

snapshot() {
  echo "════════ MetaFind 進度  $(date '+%H:%M:%S') ════════"

  # ---- 點雲
  local pid n_files
  pid=$(pgrep -f "metafind.data.fetch_pointclouds" | head -1)
  n_files=$(ls "$ROOT/artifacts/pointclouds" 2>/dev/null | wc -l)
  printf "\n點雲   %s\n" "$([ -n "$pid" ] && echo "下載中 (PID $pid)" || echo "未執行")"
  printf "  已抽出  %s / 46052  (%.1f%%)\n" "$n_files" "$(echo "$n_files*100/46052" | bc -l)"
  [ -f "$LOG" ] && grep -v Warning "$LOG" | tail -1 | sed 's/^/  最新    /'

  # ---- 實際傳輸速率（量位元組；抽取是突發式的，量檔案數會誤導）
  if [ -n "$pid" ]; then
    local a b
    a=$(find "$ROOT/work" -name "*.incomplete" -printf "%s\n" 2>/dev/null | paste -sd+ | bc)
    sleep 5
    b=$(find "$ROOT/work" -name "*.incomplete" -printf "%s\n" 2>/dev/null | paste -sd+ | bc)
    python3 -c "
a=${a:-0}; b=${b:-0}; n=$n_files
d=(b-a)/5/1048576
if d>0.05:
    rem=(46052-n)/288*1160/d/60
    print(f'  速率    {d:.1f} MB/s   剩餘約 {rem:.0f} 分鐘')
else:
    print('  速率    (正在解壓或切換 shard)')
" 2>/dev/null
  fi

  # ---- 其他資料
  printf "\n其他資料\n"
  for item in "sources/ulip2:ULIP-2 權重" "sources/procthor:ProcTHOR" "cache/hf:模型快取"; do
    local p="${item%%:*}" label="${item##*:}"
    printf "  %-14s %s\n" "$label" "$(du -sh "$ROOT/$p" 2>/dev/null | cut -f1 || echo '-')"
  done
  printf "  %-14s %s\n" "磁碟可用" "$(df -h "$ROOT" | tail -1 | awk '{print $4}')"

  # ---- 程式碼
  printf "\n程式碼\n"
  printf "  測試          %s\n" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && python -m pytest tests/ -q 2>/dev/null | tail -1 | tr -d '\n')"
  printf "  最新 commit   %s\n" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && git log --oneline -1)"
}

if [[ "${1:-}" == "-w" ]]; then
  while true; do clear; snapshot; sleep 30; done
else
  snapshot
fi
