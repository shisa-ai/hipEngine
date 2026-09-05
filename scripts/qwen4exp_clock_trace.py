#!/usr/bin/env python3
"""Sample hardware outside a child benchmark, then join its phase windows."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time


def read_sensors(root):
    result = {}
    for field, filename in (
        ("sclk_hz", "freq1_input"), ("temperature_millic", "temp1_input"),
        ("power_uw", "power1_average"),
    ):
        try:
            result[field] = int((root / filename).read_text().strip())
        except (OSError, ValueError):
            pass
    return result


def summarize_window(samples, start, end):
    rows = [row for row in samples if start <= row["time_ns"] <= end]
    result = {"samples": len(rows)}
    for key in ("sclk_hz", "temperature_millic", "power_uw"):
        values = [row[key] for row in rows if key in row]
        if values:
            result[key] = {"min": min(values), "max": max(values), "mean": statistics.mean(values)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--benchmark-output", type=Path, required=True)
    p.add_argument("--hwmon", type=Path, required=True)
    p.add_argument("--interval", type=float, default=0.25)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.interval < 0.05:
        p.error("child command and interval >=0.05 required")
    if (args.hwmon / "freq1_label").read_text().strip() != "sclk":
        p.error("freq1 sensor must report sclk")
    samples = []
    child = subprocess.Popen(command)
    try:
        while child.poll() is None:
            samples.append({"time_ns": time.perf_counter_ns(), **read_sensors(args.hwmon)})
            time.sleep(args.interval)
    finally:
        if child.poll() is None:
            child.terminate()
        returncode = child.wait()
        report = {
            "command": command, "returncode": returncode, "hwmon": str(args.hwmon),
            "interval_s": args.interval, "clock": "perf_counter_ns",
            "samples": samples, "phases": [],
            "scope": "external sampler, diagnostic only; no injected delays in benchmark",
        }
        if args.benchmark_output.exists():
            benchmark = json.loads(args.benchmark_output.read_text())
            for row in benchmark.get("samples", []):
                phase = {k: row[k] for k in ("case_id", "mode", "repetition")}
                for name, window in row.get("phase_windows_ns", {}).items():
                    phase[name] = summarize_window(samples, *window)
                report["phases"].append(phase)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
