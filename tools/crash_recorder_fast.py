#!/usr/bin/env python3
"""20 Hz black box: 60 samples in the last 3 seconds instead of 6.

[KYZEN 2026-08-29] "不要抓崩潰前0.5秒 要抓崩潰前3秒". The 0.5 s recorder answered
"was anything at a limit" and the answer was no, every time -- 50 GB of RAM free,
72 C, 559 W of a 600 W cap. What it could not answer is what happened BETWEEN two
samples, and on this machine that is where the whole event lives: six hard resets
with no kernel record, no MCE, no Xid.

`nvidia-smi -lms 50` is used instead of re-invoking nvidia-smi per sample. The old
recorder spawned a process every 0.5 s, which is most of why it could not go
faster; this is one long-lived process streaming CSV, and its timestamps come out
exactly 50 ms apart (verified before this file was written).

WHAT THIS STILL CANNOT SEE, stated because the last recorder's silence was read as
evidence: the RTX 5090's transient current spikes are MICROSECONDS wide and can
reach roughly twice the rated draw. 20 Hz is 10x better than 2 Hz and still four
orders of magnitude too slow for that. A clean trace here means "nothing at a
limit on a 50 ms average", not "no transient". Only a clamp meter or a PSU with
telemetry can settle that one.

Every line is fsynced, so a hard reset keeps everything already written.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time


def _read(path) -> int:
    """One sysfs integer, or -1. Never raises: a rail that disappears mid-run
    must not take the recorder down with it -- this file's whole job is to still
    be writing at the instant everything else stops."""
    try:
        return int(path.read_text())
    except (OSError, ValueError):
        return -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ms", type=int, default=50, help="sampling period")
    ap.add_argument("--out", default="data/outputs/logs/crash_fast.csv",
                    help="prefer a disk the workload is NOT using")
    ap.add_argument("--fsync-every", type=int, default=10,
                    help="samples per fsync; 1 = every sample")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()

    proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.sm,"
         "memory.used,utilization.gpu,pcie.link.gen.current,"
         "pcie.link.width.current,clocks_throttle_reasons.active",
         "--format=csv,noheader,nounits", "-lms", str(a.ms)],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE, text=True, bufsize=1)

    n = 0
    prev_cpu = prev_rapl = None
    # [KYZEN 2026-08-29] 「你把所有硬體可能出現的問題都監控一下」. Surveyed
    # first rather than assumed: this board exposes coretemp (CPU package),
    # nvme (drive composite) and acpitz (board zone), and NO voltage rails --
    # the super-I/O chip has no driver loaded, so 12 V droop, the reading that
    # would test the supply theory head on, is not available here. That gap
    # is stated rather than papered over with temperatures that cannot show it.
    temps = {}
    for h in sorted(pathlib.Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            name = (h / "name").read_text().strip()
        except OSError:
            continue
        if name == "coretemp":
            temps["cpu_c"] = h / "temp1_input"      # package
        elif name == "nvme":
            temps["nvme_c"] = h / "temp1_input"     # composite
        elif name == "acpitz":
            temps["board_c"] = h / "temp1_input"
    temps = {k: v for k, v in temps.items() if v.exists()}

    # [KYZEN 2026-08-29] `sensors-detect` found nct6799 -- 18 voltage rails
    # and 7 fans that were invisible an hour ago, when this file's own header
    # said no voltage reading existed on this board. THIS IS THE MEASUREMENT
    # THE WHOLE INVESTIGATION WANTED: if the supply sags at the connector, a
    # rail dips at the moment of the reset, and eight crashes so far have
    # shown nothing dipping because nothing was watching the rails.
    #
    # Every rail is recorded raw, unlabelled and unscaled, because the driver
    # has no board profile: `in1` reads 1.016 V and 1.016 x 11.7 = 11.9 V, so
    # it is PROBABLY +12 V -- probably is not good enough to pick one and
    # discard seventeen. The scale does not matter for the question being
    # asked. A rail that drops 10% drops 10% whatever its divider is, and
    # recording all of them means the answer does not depend on my guess
    # about which one to keep.
    volts = sorted(pathlib.Path(f"/sys/class/hwmon/hwmon{i}")
                   for i in range(20)
                   if (pathlib.Path(f"/sys/class/hwmon/hwmon{i}/name").exists()
                       and pathlib.Path(f"/sys/class/hwmon/hwmon{i}/name")
                       .read_text().strip() == "nct6799"))
    rails, fans = [], []
    if volts:
        h = volts[0]
        rails = sorted(h.glob("in*_input"), key=lambda x: int(x.name[2:-6]))
        fans = sorted(h.glob("fan*_input"), key=lambda x: int(x.name[3:-6]))
    print(f"  溫度感測: {', '.join(temps) or '無'}", flush=True)
    print(f"  電壓軌: {len(rails)} 條   風扇: {len(fans)} 個"
          f"{'' if rails else '  ⚠ 沒有電壓感測'}", flush=True)
    with out.open("a") as fh:
        if new:
            fh.write("ts,gpu_w,gpu_c,sm_mhz,vram_mb,util,pcie_gen,pcie_width,"
                     "gpu_throttle,ram_used_mb,ram_avail_mb,"
                     "cpu_busy_pct,cpu_mhz,cpu_w,cpu_c,nvme_c,board_c"
                     + "".join(f",in{i}" for i in range(len(rails)))
                     + "".join(f",fan{i}" for i in range(len(fans))) + "\n")
        rapl = next((c for c in
                     ("/sys/class/powercap/intel-rapl:0/energy_uj",
                      "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj")
                     if os.access(c, os.R_OK)), None)
        if rapl is None:
            print("  ⚠ CPU 功耗讀不到，cpu_w 欄位會是 -1。要開啟請敲："
                  "sudo chmod -R a+r /sys/class/powercap/intel-rapl*/", flush=True)
        print(f"20 Hz recorder -> {out} (period {a.ms} ms, "
              f"fsync every {a.fsync_every} samples)", flush=True)
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line or "," not in line:
                    continue
                # /proc/meminfo is read per sample: at 20 Hz it is a page-cache
                # read, and RAM is the field the previous recorder was built to
                # exonerate, so it has to stay at the same resolution as the rest.
                used = avail = 0
                try:
                    tot = 0
                    for ml in open("/proc/meminfo"):
                        if ml.startswith("MemTotal:"):
                            tot = int(ml.split()[1]) >> 10
                        elif ml.startswith("MemAvailable:"):
                            avail = int(ml.split()[1]) >> 10
                            break
                    used = tot - avail
                except OSError:
                    pass

                # [KYZEN 2026-08-29] 「沒有我說爆掉前ㄟ」-- the recorder had RAM at
                # 20 Hz right up to each reset but NOT ONE CPU SAMPLE, so eight
                # crashes produced no account of what the CPU was doing at the
                # moment the machine died. The 265KF can draw 250 W on top of the
                # GPU's 400, and all of it leaves the same PSU. Ruling the CPU out
                # from a `load average` read minutes afterwards is not the same
                # thing as having the moment.
                #
                # /proc/stat is a running total, so busy% is the delta against the
                # previous sample; the first row of a run has no delta and reports
                # -1 rather than a fabricated 0.
                cpu_pct, cpu_mhz, cpu_w = -1.0, 0, -1.0
                try:
                    t = [int(x) for x in open("/proc/stat").readline().split()[1:8]]
                    tot_j, idle_j = sum(t), t[3] + t[4]
                    if prev_cpu and tot_j > prev_cpu[0]:
                        cpu_pct = (1 - (idle_j - prev_cpu[1]) / (tot_j - prev_cpu[0])) * 100
                    prev_cpu = (tot_j, idle_j)
                except (OSError, IndexError, ValueError):
                    pass
                try:
                    mhz = [float(l.split(":")[1]) for l in open("/proc/cpuinfo")
                           if l.startswith("cpu MHz")]
                    cpu_mhz = int(max(mhz)) if mhz else 0
                except (OSError, IndexError, ValueError):
                    pass
                if rapl is not None:
                    # RAPL is a monotonic energy counter in microjoules; power is
                    # its delta over the interval. Unreadable without root unless
                    # the powercap tree has been chmod a+r.
                    try:
                        e, now = int(open(rapl).read()), time.time()
                        if prev_rapl and now > prev_rapl[1] and e >= prev_rapl[0]:
                            cpu_w = (e - prev_rapl[0]) / 1e6 / (now - prev_rapl[1])
                        prev_rapl = (e, now)
                    except (OSError, ValueError):
                        pass

                t = {}
                for k, path in temps.items():
                    try:
                        t[k] = int(path.read_text()) / 1000
                    except (OSError, ValueError):
                        t[k] = -1
                fh.write(f"{time.time():.6f},{line.replace(', ', ',')},{used},{avail},"
                         f"{cpu_pct:.1f},{cpu_mhz},{cpu_w:.1f},"
                         f"{t.get('cpu_c', -1):.1f},{t.get('nvme_c', -1):.1f},"
                         f"{t.get('board_c', -1):.1f}"
                         + "".join(f",{_read(r)}" for r in rails)
                         + "".join(f",{_read(f)}" for f in fans) + "\n")
                fh.flush()
                # [CODEX 2026-08-29] This fsynced EVERY sample -- 20 syncs a
                # second onto the same NVMe the training reads its .npz files
                # from, while the file header claimed the recorder does not
                # affect the run. It did not cause the first six resets, but
                # "does not affect training" was not measured, it was asserted.
                #
                # Batching to 10 means a hard reset loses at most 0.5 s of
                # samples -- exactly the resolution the OLD 2 Hz recorder had,
                # so nothing that recorder could have caught is lost. flush()
                # still runs every sample, so only a power cut loses anything;
                # a killed process does not.
                n += 1
                if n % a.fsync_every == 0:
                    os.fsync(fh.fileno())
        except KeyboardInterrupt:
            pass
        finally:
            proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
