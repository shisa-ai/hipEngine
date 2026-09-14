#!/usr/bin/env python3
"""A/B the native and Python orchestration of the two-rank staged exchange.

Both arms implement the same protocol with the same dependency structure and
completion boundaries: both device-to-host copies submitted before either wait,
two host waits per reduction, a host sum, both return copies submitted with no
wait, and one drain per chain. The Python arm is
``scripts/tp_collective_bench.py``'s ``staged_exchange_batched``; the native arm
is ``benchmarks/micro/runners/hip_staged_exchange.hip``.

The comparison is on total latency and on the same ladder slope, not on presumed
savings. Phase counters from each arm are reported side by side because they are
*not* transferable: earlier submission changes the exposed wait, so a phase that
shrinks in one arm can grow in the other.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE = REPO_ROOT / "benchmarks" / "micro" / "runners" / "hip_staged_exchange.hip"
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-tp2-native-staged-exchange")
DEFAULT_DEPTHS = (1, 4, 16, 32, 64, 128)


def _build(source: Path, build_dir: Path, arch: str, *, require_cached: bool) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    exe = build_dir / "hip_staged_exchange"
    if require_cached and not exe.exists():
        raise SystemExit(f"require_cached set but {exe} is missing; build it first")
    if not exe.exists() or exe.stat().st_mtime < source.stat().st_mtime:
        command = [
            "hipcc",
            "-O2",
            f"--offload-arch={arch}",
            str(source),
            "-o",
            str(exe),
        ]
        print(f"$ {shlex.join(command)}", file=sys.stderr)
        subprocess.run(command, check=True)
    return exe


def _run_native(
    exe: Path,
    *,
    depth: int,
    count: int,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    command = [
        str(exe),
        "--depth",
        str(depth),
        "--count",
        str(count),
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmup),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {
            "error": f"exit {completed.returncode}",
            "stderr": completed.stderr[-2000:],
        }
    return json.loads(completed.stdout)


def _marginal(points: list[tuple[int, float]]) -> dict[str, Any]:
    """The same ladder statistic the Python arm's chain report uses."""

    ordered = sorted(points)
    if len(ordered) < 2:
        return {"overall_us_per_step": None}
    (first_depth, first_us), (last_depth, last_us) = ordered[0], ordered[-1]
    if last_depth == first_depth:
        return {"overall_us_per_step": None}
    return {
        "from_depth": first_depth,
        "to_depth": last_depth,
        "overall_us_per_step": (last_us - first_us) / (last_depth - first_depth),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", default=",".join(str(d) for d in DEFAULT_DEPTHS))
    parser.add_argument("--count", type=int, default=5120)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--arch", default="gfx1100")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument(
        "--python-artifact",
        type=Path,
        default=REPO_ROOT / "benchmarks/results/2026-09-14-w7900-tp2-dependent-reduction-chain.json",
        help="the Python arm's chain report to compare against",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    depths = [int(value) for value in args.depths.split(",") if value.strip()]
    exe = _build(NATIVE_SOURCE, args.build_dir, args.arch, require_cached=args.require_cached)

    native_points: list[tuple[int, float]] = []
    native: dict[str, Any] = {}
    for depth in depths:
        result = _run_native(
            exe,
            depth=depth,
            count=args.count,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        native[str(depth)] = result
        if "error" in result:
            print(f"native depth {depth} failed: {result['error']}", file=sys.stderr)
            continue
        native_points.append((depth, float(result["total_median_us"])))

    native_marginal = _marginal(native_points)

    python_arm: dict[str, Any] = {}
    if args.python_artifact.exists():
        artifact = json.loads(args.python_artifact.read_text())
        try:
            chain = artifact["collective"]["cases"]["all_reduce:rows1:fp32"]["dependent_chain"]
            python_arm = chain["modes"]["staged_exchange_batched"]
            # The performance arm is deliberately uninstrumented, so the phase
            # breakdown comes from the attribution arm, which is a different and
            # slower measurement of the same protocol.
            python_instrumented = chain["modes"]["staged_exchange_instrumented"]
        except KeyError as error:
            python_arm = {"error": f"could not read the batched arm: {error}"}
            python_instrumented = {}

    python_marginal = (python_arm.get("marginal") or {}).get("overall_us_per_step")
    native_marginal_us = native_marginal.get("overall_us_per_step")
    ratio = None
    if python_marginal and native_marginal_us:
        ratio = python_marginal / native_marginal_us

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "tp2-staged-exchange-native-ab",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": "batched: both D2H before either wait, no return wait, one drain per chain",
        "count": args.count,
        "payload_bytes": args.count * 4,
        "depths": depths,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "arch": args.arch,
        "native": native,
        "native_marginal": native_marginal,
        "python_arm": {
            "source": str(args.python_artifact.relative_to(REPO_ROOT))
            if args.python_artifact.exists()
            else None,
            "marginal_us_per_step": python_marginal,
            "instrumented_marginal_us_per_step": (python_instrumented.get("marginal") or {}).get(
                "overall_us_per_step"
            ),
            "phases_per_step_us": python_instrumented.get("phases_per_step_us"),
            "vector_check": python_arm.get("vector_check"),
            "copy_probe": python_arm.get("copy_probe"),
        },
        "comparison": {
            "python_us_per_step": python_marginal,
            "native_us_per_step": native_marginal_us,
            "python_over_native": ratio,
            "basis": "ladder slope over the same depths, same protocol, same payload",
            "phase_counters_are_not_transferable": (
                "earlier submission changes the exposed wait, so a phase that "
                "shrinks in one arm can grow in the other; compare totals"
            ),
        },
    }

    text = json.dumps(report, indent=2, sort_keys=False) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
        print(f"wrote {args.json}", file=sys.stderr)
    else:
        print(text)

    print(
        f"python {python_marginal} us/step vs native {native_marginal_us} us/step"
        + (f" -> {ratio:.3f}x" if ratio else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
