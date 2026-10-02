#!/usr/bin/env python3
"""How long does the host package stay at its 2.0 GHz base clock after load stops?

DE-5 (worklog 2026-10-02) established that a 4 s prefill pins all 32 cores at
exactly 2000 MHz for ~2 s of the following decode. The X-8 long-context ladder
showed the same state persisting for many minutes of continuous load and
halving decode. This measures the recovery time constant so the ladder's decode
rates can be attributed to the host state rather than to context depth.
"""
import glob
import json
import subprocess
import time
from pathlib import Path

OUT = Path("/tmp/q4x-long/clock-recovery.jsonl")
DURATION = float(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 300.0


def freqs() -> list[int]:
    values = []
    for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"):
        try:
            values.append(int(Path(path).read_text()) // 1000)
        except OSError:
            pass
    return values


def gpu_busy() -> int:
    try:
        return int(Path("/sys/class/drm/card1/device/gpu_busy_percent").read_text().strip())
    except OSError:
        return -1


started = time.time()
rows = []
with OUT.open("w") as fh:
    while time.time() - started < DURATION:
        f = freqs()
        row = {
            "t_s": round(time.time() - started, 1),
            "cpu_max_mhz": max(f) if f else 0,
            "cpu_p50_mhz": sorted(f)[len(f) // 2] if f else 0,
            "n_above_3000": sum(1 for v in f if v > 3000),
            "gpu_busy_pct": gpu_busy(),
        }
        rows.append(row)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        time.sleep(5)

first_boost = next((r["t_s"] for r in rows if r["cpu_max_mhz"] > 3000), None)
print(json.dumps({
    "samples": len(rows),
    "first_sample": rows[0] if rows else None,
    "first_boost_s": first_boost,
    "last_sample": rows[-1] if rows else None,
}, indent=1))
