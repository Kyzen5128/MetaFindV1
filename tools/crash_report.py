#!/usr/bin/env python3
"""Merge the trainer's own log with the system black box into one timeline.

[KYZEN 2026-08-29] "你能不能在跑的時候有個log能看到時候爆掉時是卡在哪".
Both halves already survive a hard reset -- `runlog._append` fsyncs every
metrics row and `crash_recorder.sh` fsyncs every sample -- but they are two
files on two different clocks, so "where was it when it died" took a manual
join every time. This does the join.

The machine hard-reset five times on 2026-08-29 with no kernel record of any
kind, so the only account of the final seconds is what was fsynced before the
power went. This prints that account.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from metafind import paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="window before the last record")
    a = ap.parse_args()

    trn = paths.LOGS / "train_stage1.jsonl"
    # [CODEX 2026-08-29] This read `crash_recorder.csv` only. When the 20 Hz
    # recorder landed and started writing `crash_fast.csv`, this tool kept
    # reading the OLD file -- which had stopped at 10:09:25 -- and printed
    # "系統黑盒子比訓練列多活了 0.0 秒" for the 17:04 crash. That line was FALSE:
    # the fast recorder had 1.3 more seconds of data that this never opened.
    #
    # The failure shape is the one this whole investigation keeps hitting: a
    # tool that answers from a source that stopped, with no sign that it did.
    # Both files are read now, and the freshest is named in the output so the
    # reader can see which one the answer came from.
    candidates = [paths.LOGS / "crash_fast.csv", paths.LOGS / "crash_recorder.csv"]
    csvs = [c for c in candidates if c.exists()]

    rows = [json.loads(l) for l in trn.read_text().splitlines() if l.strip()] \
        if trn.exists() else []
    # Sorted, because three recorders once wrote this file at the same time and
    # left it out of order; the last LINE is not always the last SAMPLE.
    sys_rows, src = [], None
    for c in csvs:
        rs = [l.split(",") for l in c.read_text().splitlines()[1:] if l.strip()]
        rs = [r for r in rs if len(r) >= 6]
        if rs and (not sys_rows or float(rs[-1][0]) > float(sys_rows[-1][0])):
            sys_rows, src = sorted(rs, key=lambda r: float(r[0])), c.name

    if not rows and not sys_rows:
        print("no records", file=sys.stderr)
        return 1

    end = max([rows[-1]["ts"] if rows else 0,
               float(sys_rows[-1][0]) if sys_rows else 0])
    start = end - a.seconds

    merged = []
    for r in rows:
        if r["ts"] >= start:
            merged.append((r["ts"], "TRAIN",
                           f"epoch {r['epoch']:>2} step {r['step']:>6}  "
                           f"loss {r['loss']:.4f}  grad {r['grad_norm']:.3f}"))
    # The two recorders write different columns, so the source decides the
    # format. Guessing from field count would break silently the next time a
    # column is added -- which is how this tool got here.
    fast = src == "crash_fast.csv"
    for r in sys_rows:
        if float(r[0]) < start:
            continue
        if fast:   # ts,gpu_w,gpu_c,sm_mhz,vram_mb,util,ram_used,ram_avail
            merged.append((float(r[0]), "sys",
                           f"{float(r[1]):>6.1f} W  {r[2]:>2}C  {r[3]:>4} MHz  "
                           f"util {r[5]:>3}%  VRAM {r[4]:>5} MB  "
                           f"RAM {r[6]:>6} used / {r[7]:>6} free"))
        else:      # ts,ram_used,ram_avail,swap,vram,gpu_w,gpu_c,rss
            merged.append((float(r[0]), "sys",
                           f"{float(r[5]):>6.1f} W  {r[6]:>2}C  "
                           f"VRAM {r[4]:>5} MB  "
                           f"RAM {r[1]:>6} used / {r[2]:>6} free  swap {r[3]}"))
    merged.sort()

    print(f"最後 {a.seconds:.0f} 秒，止於 {time.strftime('%H:%M:%S', time.localtime(end))}")
    print(f"訓練列每 20 步一筆；系統列來自 {src or '(無)'}"
          f"{'（20 Hz）' if fast else '（2 Hz）'}。兩者都 fsync 過，硬重置也留得住。\n")
    for ts, kind, text in merged:
        mark = ">>" if kind == "TRAIN" else "  "
        print(f"{mark} {time.strftime('%H:%M:%S', time.localtime(ts))}  {kind:<5} {text}")

    last_train = next((r for r in reversed(rows)), None)
    if last_train:
        print(f"\n最後跑到：epoch {last_train['epoch']} step {last_train['step']}"
              f"  於 {time.strftime('%H:%M:%S', time.localtime(last_train['ts']))}")
        if sys_rows:
            gap = end - last_train["ts"]
            print(f"系統黑盒子比訓練列多活了 {gap:.1f} 秒"
                  f"（訓練每 20 步才寫一次，所以真正的死亡點在最後那列之後 0~{gap:.0f} 秒內）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
