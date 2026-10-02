#!/usr/bin/env python3
"""Log the host's *delivered* CPU speed and GPU activity together.

`scaling_cur_freq` reports the requested P-state and reads the nominal 2000 MHz
across states that differ ~9x in measured speed, so the only usable instrument
is a fixed CPU-bound workload. One sample = one microbenchmark run plus the
GPU busy value at that instant, so host speed can be correlated with whether
the GPU is in a prefill or a decode window.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

OUT = Path("/tmp/q4x-long/cpu-speed-watch.jsonl")
DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 7200.0
ITERATIONS = 60_000  # ~0.1 s at full speed, ~0.9 s at the slowest state

ENV = os.environ
KEYS = ("HIPENGINE_HIP_ARCH", "PATH", "HOME", "LD_LIBRARY_PATH", "NOPE_MISSING")
TABLE = {f"k{i}": i for i in range(200)}


def work(n: int) -> float:
    started = time.perf_counter()
    for i in range(n):
        for key in KEYS:
            ENV.get(key)
        TABLE.get("k42")
        "some/weight/name/for/a/layer".strip().split("/")[-1]
    return (time.perf_counter() - started) * 1000.0


def cpu_max_mhz() -> int:
    values = []
    for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"):
        try:
            values.append(int(Path(path).read_text()) // 1000)
        except OSError:
            pass
    return max(values) if values else 0


def gpu_busy() -> int:
    try:
        return int(Path("/sys/class/drm/card1/device/gpu_busy_percent").read_text().strip())
    except OSError:
        return -1


def rss_top_mb() -> int:
    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
                    break
        except (OSError, ValueError, IndexError):
            continue
    return total // 1024


work(10_000)  # interpreter warmup
started = time.time()
with OUT.open("a") as fh:
    while time.time() - started < DURATION:
        busy_before = gpu_busy()
        speed = work(ITERATIONS)
        row = {
            "t": time.strftime("%H:%M:%S", time.localtime()),
            "elapsed_s": round(time.time() - started, 1),
            "work_ms": round(speed, 1),
            "gpu_busy_before_pct": busy_before,
            "gpu_busy_after_pct": gpu_busy(),
            "cpu_max_mhz": cpu_max_mhz(),
            "rss_total_mb": rss_top_mb(),
        }
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        time.sleep(8)
