#!/usr/bin/env python3
"""Stress RAM and CPU with the GPU idle, to separate two crash hypotheses.

The machine hard-reset four times on 2026-08-29, every time during Stage 1
training, with no kernel panic, no MCE and no Xid -- the journal stops
mid-stream and the next entry is a fresh boot. Two candidates survive:

  H1  power delivery
  H2  memory / memory-controller instability

The measurement that already exists cannot separate them, and that is the whole
reason this file is here: a GPU burn loop held 599.8 W for 6.5 minutes and
survived, while training averaged 525.1 W and crashed four times -- but the burn
loop also left CPU and RAM nearly idle, so "GPU-only vs everything" and "steady
vs jittery power" moved together and neither can be credited.

This run moves exactly one of them. The GPU is never touched -- no torch, no
CUDA, no import of either -- so power stays at idle (measured 8-10 W) while RAM
and CPU go to full load.

  crashes  ->  the GPU is not required to reproduce the fault. H2, and the
               remaining suspect is DDR5-6000 XMP on a 265KF whose validated
               speed is 5600, with 2x32GB dual-rank modules.
  survives ->  RAM and CPU alone do not reproduce it in this window. H2 is
               weakened; the GPU is back in the picture.

Neither outcome is proof. A survival bounds nothing -- the fault is
intermittent and this is one window.

Integrity is checked as well as liveness. A memory fault that does NOT reset
the machine corrupts silently, and that is the worse outcome for a training
run: it would land in a checkpoint rather than in a crash log.

Run `tools/crash_recorder.sh` alongside this so there is a black box.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

_K = np.uint64(0x5DEECE66D)


def _log(path: str, line: str) -> None:
    """Append one line, flushed to the OS, O_APPEND so workers cannot interleave.

    [ULIP2 REVIEWER 2026-08-29, defect 4] Everything used to live in an
    in-memory `mp.Queue` printed only after `join()`, so the CRASH case -- the
    outcome this tool exists to produce -- left nothing at all. Died at minute 4
    of 10? "It died" was the entire result: no elapsed time, no pass count, no
    partial mismatches, no per-worker state, all of which are evidence about the
    fault. That is the same shape this block has spent two days cleaning up
    after: the run that fails is the one that leaves nothing behind.
    """
    with open(path, "a") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def worker(wid: int, gib: float, seconds: float, log: str, every: float) -> None:
    """Churn `gib` of RAM against a pristine reference until `seconds` elapse."""
    n = int(gib * (1 << 30) // 8)          # float64 elements
    rng = np.random.default_rng(20260829 + wid)
    try:
        a = rng.random(n)
        # The second buffer is what makes this a BANDWIDTH test rather than a
        # residency test: copying across it moves data through the controller,
        # where a residency test leaves it sitting in DRAM after the first fill.
        b = np.empty_like(a)
    except MemoryError:
        _log(log, f"worker {wid} alloc_failed"); return

    passes = bad = 0
    t0 = last = time.time()
    while time.time() - t0 < seconds:
        np.copyto(b, a)
        # [ULIP2 REVIEWER 2026-08-29 defect 1, then CORRECTED by running it]
        # This was `b *= 1.0000001; b /= 1.0000001` compared with `> 1e-9`. The
        # Reviewer showed 1e-9 blinds the bottom 21 of 52 mantissa bits (a flip
        # at bit 20 is 1.16e-10, under the threshold) -- where single-bit DRAM
        # errors mostly land -- and measured the multiply/divide round-trip
        # EXACT over 2,000,000 elements, concluding `==` was safe.
        #
        # The first half is right. The second is an under-powered sample, and
        # the fixed version's own smoke test found it: 210 "mismatches", all at
        # ONE index, THE SAME index every pass, differing by exactly 1 ULP
        # (0.4999999940109797 -> ...98). Measured across seeds: 1 mismatching
        # element per ~16,000,000, so a 2,000,000 sample sees zero most of the
        # time. IEEE 754 does not guarantee x*c/c == x, and a repeatable
        # 1-ULP difference at a fixed index is rounding, not DRAM.
        #
        # So neither `>1e-9` nor `==` works on a float round-trip: one is blind,
        # the other cries wolf. The operation is what has to change. XOR against
        # a constant, applied twice on the uint64 view, is bit-exact by
        # CONSTRUCTION rather than by measurement -- verified 0 mismatches over
        # 6,710,886 elements -- so `==` is now exact and every bit is covered.
        v = b.view(np.uint64)
        v ^= _K
        v ^= _K
        diff = b != a
        n_bad = int(diff.sum())
        if n_bad:
            bad += n_bad
            i = int(np.argmax(diff))
            _log(log, f"worker {wid} MISMATCH pass {passes} index {i} "
                      f"expected {a[i]!r} got {b[i]!r} n_bad {n_bad}")
        # [ULIP2 REVIEWER 2026-08-29, defect 2] The buffers used to be swapped
        # here, which made a corrupted `b` the next reference: after one fault
        # every later pass compared against corrupt data, so `bad` stopped
        # incrementing while the memory was still faulty. `a` is now written
        # once and never again -- the reference stays pristine for the whole
        # run, and the copy into `b` supplies the churn the swap was there for.
        passes += 1
        now = time.time()
        if now - last >= every:
            _log(log, f"worker {wid} alive t {now - t0:.0f}s passes {passes} bad {bad}")
            last = now
    _log(log, f"worker {wid} done t {time.time() - t0:.0f}s passes {passes} bad {bad}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gib", type=float, default=1.0, help="per worker, per buffer")
    p.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    p.add_argument("--seconds", type=float, default=600.0)
    p.add_argument("--headroom", type=float, default=0.6,
                   help="fraction of MemAvailable this run may claim")
    p.add_argument("--heartbeat", type=float, default=30.0, help="seconds between alive lines")
    p.add_argument("--log", default="data/outputs/logs/ram_soak.log")
    a = p.parse_args()

    # [KYZEN 2026-08-29] "你在執行時這樣跑是不是會導致OOM" -- he asked, I said it
    # was guarded, and the FIRST REAL RUN WAS OOM-KILLED: 20 workers x 1.4 GiB
    # was admitted as 28 GiB and peaked at 52.2 GiB. The guard counted `a` and
    # not `b`, so it approved half the true footprint. BUFFERS_PER_WORKER exists
    # so the count and the allocation cannot drift apart again -- if a third
    # buffer is ever added, this constant is what has to change with it.
    #
    # A diagnostic that OOMs proves nothing about the fault it was written to
    # find, and worse, an OOM kill LOOKS like the crash under investigation
    # while having an entirely different cause.
    BUFFERS_PER_WORKER = 2
    total = a.gib * a.workers * BUFFERS_PER_WORKER

    avail = 0.0
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) / (1 << 20)     # kB -> GiB
            break
    ceiling = avail * a.headroom
    if total > ceiling:
        # Refuse rather than silently shrink: a quietly shrunk soak reports
        # "survived" for a test that was never run.
        print(f"REFUSED: {a.workers} workers x {a.gib} GiB x {BUFFERS_PER_WORKER} buffers "
              f"= {total:.1f} GiB, but MemAvailable is {avail:.1f} GiB and the cap is "
              f"{a.headroom:.0%} of it ({ceiling:.1f} GiB).\n"
              f"         Suggested: --gib {ceiling / (a.workers * BUFFERS_PER_WORKER):.2f}",
              flush=True, file=sys.stderr)
        return 2

    print(f"{a.workers} workers x {a.gib} GiB x {BUFFERS_PER_WORKER} buffers = {total:.1f} GiB "
          f"(MemAvailable {avail:.1f}, cap {ceiling:.1f}), {a.seconds:.0f}s, GPU untouched",
          flush=True)
    assert "torch" not in sys.modules, "this must not touch CUDA"

    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    _log(a.log, f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"workers {a.workers} gib {a.gib} total {total:.1f} seconds {a.seconds:.0f}")

    # [ULIP2 REVIEWER 2026-08-29, defects 3 and minor] `mp.Queue` is gone.
    # `q.put` in the hot loop could fill the pipe buffer and block a worker
    # while the parent was blocked in `join()` -- a deadlock that triggered
    # precisely when memory WAS faulty, i.e. the positive case, and read to the
    # operator as a wedged machine: the very symptom under investigation, from a
    # different cause. `while not q.empty()` was also not a synchronisation
    # primitive and could under-report living workers. An append-only file has
    # neither problem and survives a reset, which the queue did not.
    ps = [mp.Process(target=worker, args=(i, a.gib, a.seconds, a.log, a.heartbeat))
          for i in range(a.workers)]
    t0 = time.time()
    for x in ps:
        x.start()
    for x in ps:
        x.join()

    lines = open(a.log).read().splitlines()
    done = [l for l in lines if " done " in l]
    bad = sum(int(l.rsplit(" bad ", 1)[1]) for l in done)
    dead = [x.exitcode for x in ps if x.exitcode not in (0, None)]
    _log(a.log, f"=== end survived {time.time() - t0:.0f}s done {len(done)}/{a.workers} "
                f"mismatches {bad} nonzero_exits {dead}")
    print(f"survived {time.time() - t0:.0f}s · workers done {len(done)}/{a.workers} · "
          f"mismatches {bad} · nonzero exits {dead} · log {a.log}", flush=True)
    return 1 if (bad or dead or len(done) != a.workers) else 0


if __name__ == "__main__":
    raise SystemExit(main())
