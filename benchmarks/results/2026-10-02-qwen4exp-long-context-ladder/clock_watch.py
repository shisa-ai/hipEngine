#!/usr/bin/env python3
"""Log the host CPU clock state and GPU busy while the ladder runs."""
import glob, json, subprocess, time, sys
from pathlib import Path

out = Path("/tmp/q4x-long/host-clock-watch.jsonl")
stop = time.time() + float(sys.argv[1] if len(sys.argv) > 1 else 7200)
busy = Path("/sys/class/drm/card1/device/gpu_busy_percent")
with out.open("a") as fh:
    while time.time() < stop:
        freqs = []
        for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"):
            try: freqs.append(int(Path(p).read_text()) // 1000)
            except OSError: pass
        try: gb = int(busy.read_text().strip())
        except OSError: gb = -1
        row = {
            "t": time.strftime("%H:%M:%S", time.localtime()),
            "epoch": round(time.time(), 1),
            "cpu_max_mhz": max(freqs) if freqs else 0,
            "cpu_min_mhz": min(freqs) if freqs else 0,
            "cpu_p50_mhz": sorted(freqs)[len(freqs)//2] if freqs else 0,
            "n_at_min": sum(1 for v in freqs if v == (min(freqs) if freqs else 0)),
            "gpu_busy_pct": gb,
        }
        fh.write(json.dumps(row) + "\n"); fh.flush()
        time.sleep(15)
