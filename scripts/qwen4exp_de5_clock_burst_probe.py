#!/usr/bin/env python3
"""DE-5 control: which load class clamps the host CPU package clock?

DE-5's loop-start transient is a host CPU clock clamp: every decode that follows
a prefill runs its first ~30 steps with every core pinned at the 2.0 GHz base
clock (the warmup decode of the same loop holds 5.1 GHz), and the step time
tracks the clock rather than the work.  This probe is the control for the
*trigger*.  It applies each load class in isolation and samples the same clock,
so a clamp can be attributed to the class that produces it instead of inferred
from the engine's own run.

Arms:

* ``cpu``  - one thread at 100% CPU.  The burner is a subprocess so the sampler
  thread is not starved of the GIL, which is what made the first attempt at this
  measurement unreadable.
* ``gpu``  - device-to-device copies over 1 GiB buffers: a bandwidth-bound GPU
  load with the host almost idle.
* ``cpu_mem`` - one thread driving host memory traffic (64 MiB copies plus a
  strided walk over a 256 MiB buffer), the one load class the plain CPU arm does
  not exercise and the one the prefill's staging does.
* ``both`` / ``mem_gpu`` - the pairs that a prefill actually is.

Clocks are read as the maximum and mean across every core's
``scaling_cur_freq``.  Reading all cores costs ~0.2 ms, so the series is sampled
at 100 ms.  ``cpuinfo_avg_freq`` is per-CPU, not a package average: with one
thread pinned to cpu4 it reads 5118 MHz on cpu4 while cpu0 and cpu8 read 2000.

Usage:
    python3 scripts/qwen4exp_de5_clock_burst_probe.py --seconds 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time

FREQ_TEMPLATE = "/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_cur_freq"
AVG_FREQ_TEMPLATE = "/sys/devices/system/cpu/cpu{core}/cpufreq/cpuinfo_avg_freq"


def core_paths() -> list[pathlib.Path]:
    count = pathlib.Path("/sys/devices/system/cpu/present").read_text().strip()
    if "-" in count:
        count = count.split("-")[-1]
    return [
        pathlib.Path(FREQ_TEMPLATE.format(core=core))
        for core in range(int(count) + 1)
    ]


def sample(paths: list[pathlib.Path]) -> tuple[float, float]:
    values = []
    for path in paths:
        try:
            values.append(int(path.read_text().strip()))
        except Exception:
            continue
    if not values:
        return float("nan"), float("nan")
    return max(values) / 1000.0, statistics.fmean(values) / 1000.0


def sample_series(
    paths: list[pathlib.Path], seconds: float, interval_s: float = 0.1
) -> dict:
    started = time.perf_counter()
    series: list[tuple[float, float]] = []
    while time.perf_counter() - started < seconds:
        series.append(sample(paths))
        time.sleep(interval_s)
    maxima = [m for m, _ in series if m == m]
    means = [mean for _, mean in series if mean == mean]
    return {
        "samples": len(series),
        "max_mhz": [round(m, 1) for m, _ in series],
        "mean_mhz": [round(mean, 1) for _, mean in series],
        "max_mhz_median": statistics.median(maxima) if maxima else None,
        "max_mhz_min": min(maxima) if maxima else None,
        "max_mhz_max": max(maxima) if maxima else None,
        "mean_mhz_median": statistics.median(means) if means else None,
    }


CPU_BURNER = "while True: pass"
MEM_BURNER = (
    "import array\n"
    "src = array.array('d', bytes(8 * (1 << 23)))\n"
    "dst = array.array('d', bytes(8 * (1 << 23)))\n"
    "walk = array.array('d', bytes(8 * (1 << 25)))\n"
    "stride = 64 * 1024\n"
    "while True:\n"
    "    dst[:] = src\n"
    "    total = 0.0\n"
    "    for index in range(0, len(walk), stride):\n"
    "        total += walk[index]\n"
)


def start_burner(program: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def gpu_series(
    runtime, paths: list[pathlib.Path], seconds: float, sample_fn
) -> dict:
    from hipengine.core.hip import HipMemcpyKind
    from hipengine.core.memory import free, malloc

    nbytes = 1 << 30
    source = malloc(nbytes, runtime=runtime)
    target = malloc(nbytes, runtime=runtime)
    started = time.perf_counter()
    series: list[tuple[float, float]] = []
    copies = 0
    try:
        while time.perf_counter() - started < seconds:
            runtime.memcpy(
                target.ptr, source.ptr, nbytes, HipMemcpyKind.DEVICE_TO_DEVICE
            )
            runtime.device_synchronize()
            copies += 1
            series.append(sample_fn(paths))
    finally:
        free(target, runtime=runtime)
        free(source, runtime=runtime)
    return summarize(series, copies)


def summarize(series: list[tuple[float, float]], copies: int = 0) -> dict:
    maxima = [m for m, _ in series]
    means = [mean for _, mean in series]
    return {
        "samples": len(series),
        "copies": copies,
        "max_mhz": [round(m, 1) for m, _ in series],
        "mean_mhz": [round(mean, 1) for _, mean in series],
        "max_mhz_median": statistics.median(maxima) if maxima else None,
        "max_mhz_min": min(maxima) if maxima else None,
        "max_mhz_max": max(maxima) if maxima else None,
        "mean_mhz_median": statistics.median(means) if means else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--interval-ms", type=float, default=100.0)
    ap.add_argument("--json-out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    paths = core_paths()
    interval_s = args.interval_ms / 1000.0
    results: dict[str, dict] = {}

    print(f"cores sampled: {len(paths)}", flush=True)
    results["idle"] = sample_series(paths, 2.0, interval_s)
    print(
        f"idle             max core {results['idle']['max_mhz_median']:6.0f} MHz "
        f"(min {results['idle']['max_mhz_min']:6.0f})",
        flush=True,
    )

    burner = start_burner(CPU_BURNER)
    try:
        results["cpu"] = sample_series(paths, args.seconds, interval_s)
    finally:
        burner.terminate()
        burner.wait()
    print(
        f"cpu burst        max core {results['cpu']['max_mhz_median']:6.0f} MHz "
        f"(min {results['cpu']['max_mhz_min']:6.0f})",
        flush=True,
    )

    burner = start_burner(MEM_BURNER)
    try:
        results["cpu_mem"] = sample_series(paths, args.seconds, interval_s)
    finally:
        burner.terminate()
        burner.wait()
    print(
        f"cpu mem burst    max core {results['cpu_mem']['max_mhz_median']:6.0f} MHz "
        f"(min {results['cpu_mem']['max_mhz_min']:6.0f})",
        flush=True,
    )

    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    results["gpu"] = gpu_series(runtime, paths, args.seconds, sample)
    print(
        f"gpu d2d burst    max core {results['gpu']['max_mhz_median']:6.0f} MHz "
        f"(min {results['gpu']['max_mhz_min']:6.0f}, "
        f"{results['gpu']['copies']} GiB copies)",
        flush=True,
    )

    burner = start_burner(CPU_BURNER)
    try:
        results["both"] = gpu_series(runtime, paths, args.seconds, sample)
    finally:
        burner.terminate()
        burner.wait()
    print(
        f"cpu+gpu burst    max core {results['both']['max_mhz_median']:6.0f} MHz "
        f"(min {results['both']['max_mhz_min']:6.0f}, "
        f"{results['both']['copies']} GiB copies)",
        flush=True,
    )

    burner = start_burner(MEM_BURNER)
    try:
        results["mem_gpu"] = gpu_series(runtime, paths, args.seconds, sample)
    finally:
        burner.terminate()
        burner.wait()
    print(
        f"mem+gpu burst    max core {results['mem_gpu']['max_mhz_median']:6.0f} MHz "
        f"(min {results['mem_gpu']['max_mhz_min']:6.0f}, "
        f"{results['mem_gpu']['copies']} GiB copies)",
        flush=True,
    )

    results["release"] = sample_series(paths, 3.0, interval_s)
    print(
        f"after release    max core {results['release']['max_mhz_median']:6.0f} MHz",
        flush=True,
    )

    payload = {
        "schema": 1,
        "kind": "qwen4exp_de5_clock_burst_probe",
        "host": pathlib.Path("/etc/hostname").read_text().strip()
        if pathlib.Path("/etc/hostname").exists()
        else None,
        "seconds": float(args.seconds),
        "interval_ms": float(args.interval_ms),
        "cores": len(paths),
        "arms": results,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
