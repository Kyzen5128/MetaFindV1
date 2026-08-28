#!/bin/bash
# Watch the Stage 1 trainer and record what the state was WHEN it died.
#
# [Kyzen 2026-08-29] Three runs were SIGKILLed (exit 137) with no traceback and
# no OOM record: the kernel log has no "Killed process", the cgroup reports
# oom_kill 0, host RAM peaked at 10 GB of 61, and cgroup memory sat flat at
# 26.7 GB for nine minutes. So the cause is not memory, and the question is who
# sends the signal.
#
# The standing suspect is this session's own shell churn: every Bash tool call
# writes a new ~/.claude/shell-snapshots/snapshot-bash-*.sh, and the deaths
# clustered around times when calls were frequent. That is a CORRELATION nobody
# has measured, which is exactly what this records: snapshot file count and
# newest mtime, sampled beside the trainer's liveness, so "a snapshot appeared
# in the same second the trainer died" becomes checkable instead of plausible.
#
# Read-only. Touches no GPU, writes one log.
set -uo pipefail
OUT=${1:-/home/kyzen/MetaFindV1/data/outputs/logs/watch_trainer.log}
SNAP=$HOME/.claude/shell-snapshots
CG=/sys/fs/cgroup$(awk -F: '/^0::/{print $3}' /proc/self/cgroup)

pid=""
say() { printf '%s %s\n' "$(date '+%F %H:%M:%S')" "$*" >> "$OUT"; }

say "watcher started, pid $$"
while true; do
    # `pgrep -f` would match this script's own command line, so match the
    # python binary's argv instead and exclude anything holding this filename.
    new=$(pgrep -f "metafind\.train\.stage1" 2>/dev/null | while read -r p; do
              grep -aq "watch_trainer" "/proc/$p/cmdline" 2>/dev/null || echo "$p"
          done | head -1)

    if [ -n "$new" ] && [ "$new" != "$pid" ]; then
        pid=$new
        say "TRAINER APPEARED pid=$pid ppid=$(awk '{print $4}' /proc/$pid/stat 2>/dev/null)"
    fi

    if [ -n "$pid" ] && [ ! -d "/proc/$pid" ]; then
        # The moment after. Everything that could name the killer, captured
        # before it ages out.
        say "=== TRAINER GONE pid=$pid ==="
        say "  gpu        $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
        say "  sys mem    $(free -m | awk '/^Mem/{print "used "$3" MB avail "$7" MB"}')"
        say "  cgroup     $(cat $CG/memory.current 2>/dev/null) bytes"
        say "  oom_kill   $(awk '/^oom_kill /{print $2}' $CG/memory.events 2>/dev/null)"
        say "  snapshots  $(ls -1 $SNAP 2>/dev/null | wc -l) files, newest $(ls -t $SNAP 2>/dev/null | head -1)"
        say "  newest age $(( $(date +%s) - $(stat -c %Y "$SNAP/$(ls -t $SNAP | head -1)" 2>/dev/null || echo 0) )) s"
        say "  claude procs $(pgrep -c -f 'claude' 2>/dev/null)"
        say "  journal    $(journalctl --user -n 3 --no-pager 2>/dev/null | tail -2 | tr '\n' '|')"
        say "=== end ==="
        pid=""
    fi

    # Sampled beside liveness so the two series share a clock. A snapshot
    # written in the same second as a death is the signal being looked for.
    printf '%s alive=%s snaps=%s newest_age=%s gpu=%s\n' \
        "$(date +%s)" \
        "$([ -n "$pid" ] && echo 1 || echo 0)" \
        "$(ls -1 $SNAP 2>/dev/null | wc -l)" \
        "$(( $(date +%s) - $(stat -c %Y "$SNAP/$(ls -t $SNAP 2>/dev/null | head -1)" 2>/dev/null || echo 0) ))" \
        "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" \
        >> "${OUT%.log}.series"
    sleep 1
done
