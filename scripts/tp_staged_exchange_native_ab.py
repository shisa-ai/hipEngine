#!/usr/bin/env python3
"""A/B the native and Python orchestration of the two-rank staged exchange.

Both arms run in this process, at the same depths and payload, with balanced
repetitions, and the comparison is only reported as matched when both arms agree
on the payload, the depth ladder, the protocol and the physical devices, and when
each arm's source is pinned by hash. Until then the native figure is a screen and
the ratio is provisional.

The Python arm is ``scripts/tp_collective_bench.py``'s
``staged_exchange_batched``; the native arm is
``benchmarks/micro/runners/hip_staged_exchange.hip``. They implement the same
protocol: both device-to-host copies submitted before either wait, two host waits
per reduction, a host reduction, both return copies submitted with no wait, and
one drain per chain.

Phase counters from the two arms are reported side by side because they are *not*
transferable: earlier submission changes the exposed wait, so a phase that
shrinks in one arm can grow in the other. Compare totals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE = REPO_ROOT / "benchmarks" / "micro" / "runners" / "hip_staged_exchange.hip"
PYTHON_SOURCE = REPO_ROOT / "scripts" / "tp_collective_bench.py"
DEFAULT_BUILD_DIR = Path("/tmp/hipengine-tp2-native-staged-exchange")
DEFAULT_DEPTHS = (1, 4, 16, 32, 64, 128)

#: The native runner reports its protocol as this string; the Python arm reports
#: ``batched``. They must describe the same structure before a comparison is
#: reported as matched.
NATIVE_PROTOCOL = "batched: both D2H submitted before either wait, no return wait"
PYTHON_PROTOCOL = "batched"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    return completed.stdout.strip() or "unknown"


def _visible_device_names(environment: dict[str, str]) -> list[str]:
    """The physical cards this process can see, in visible order."""

    import ctypes

    try:
        library = ctypes.CDLL("libamdhip64.so")
    except OSError:
        return []
    count = ctypes.c_int()
    if library.hipGetDeviceCount(ctypes.byref(count)) != 0:
        return []
    names: list[str] = []
    for index in range(count.value):
        buffer = ctypes.create_string_buffer(256)
        if library.hipDeviceGetName(buffer, 256, index) == 0:
            names.append(buffer.value.decode("utf-8", "replace"))
    return names


def _build(source: Path, build_dir: Path, arch: str, *, require_cached: bool) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    exe = build_dir / "hip_staged_exchange"
    if require_cached and not exe.exists():
        raise SystemExit(f"require_cached set but {exe} is missing; build it first")
    if not exe.exists() or exe.stat().st_mtime < source.stat().st_mtime:
        command = ["hipcc", "-O2", f"--offload-arch={arch}", str(source), "-o", str(exe)]
        print(f"$ {shlex.join(command)}", file=sys.stderr)
        subprocess.run(command, check=True)
    return exe


def _run_native(
    exe: Path, *, depth: int, count: int, iterations: int, warmup: int
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
        payload: dict[str, Any] = {
            "error": f"exit {completed.returncode}",
            "stderr": completed.stderr[-2000:],
        }
        # The runner prints its report before it exits non-zero on a failed check,
        # so keep the verification detail when there is one.
        try:
            payload["report"] = json.loads(completed.stdout)
        except json.JSONDecodeError:
            pass
        return payload
    return json.loads(completed.stdout)


def _native_verification_ok(entry: dict[str, Any]) -> bool:
    """Both recurrences must hold: bounded always, sum wherever it is exact."""

    verification = entry.get("verification") or {}
    bounded = verification.get("bounded") or {}
    summed = verification.get("sum") or {}
    if not bounded.get("exact") or not bounded.get("finite"):
        return False
    if summed.get("informative"):
        return bool(summed.get("finite") and summed.get("exact"))
    # Otherwise the closed form overflows fp32 at this depth and the nonfinite
    # values are expected rather than a failure.
    return bool(summed.get("saturation_expected"))


def _run_python_arm(
    *, depths: list[int], count: int, iterations: int, warmup: int, seed: float, timeout_s: float
) -> dict[str, Any]:
    """Run the Python arm in this process, at the same depths and payload."""

    import importlib.util

    from hipengine.core.device import Device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import free, malloc
    from hipengine.distributed.plan import DistributedPlan
    from hipengine.distributed.rccl import RcclTransport

    spec = importlib.util.spec_from_file_location("tp_collective_bench_arm", PYTHON_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load the Python arm from {PYTHON_SOURCE}")
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)

    runtime = get_hip_runtime()
    case = bench.Case(op="all_reduce", rows=1, dtype="fp32", hidden_size=count)
    if case.count != count:
        raise RuntimeError(f"case count {case.count} does not match --count {count}")
    plan = DistributedPlan.resolve([0, 1], hidden_size=count, algorithm="rccl")
    transport = RcclTransport(
        [rank.device for rank in plan.ranks], runtime=runtime, init_timeout_s=timeout_s
    )
    world = transport.world_size
    work = [malloc(case.payload_bytes, device=Device("hip", rank)) for rank in range(world)]
    scratch = [malloc(case.payload_bytes, device=Device("hip", rank)) for rank in range(world)]
    try:
        # The whole ladder in one call: the dependency verdict and the deepest
        # informative depth are properties of the ladder, so a per-depth invocation
        # would report them against a single point (and depth 1 can never
        # demonstrate compounding, since its value *is* the single-step value).
        entry = bench._measure_staged_exchange_chain(
            transport=transport,
            runtime=runtime,
            case=case,
            buffers=(work, scratch),
            depths=tuple(depths),
            iterations=iterations,
            warmup=warmup,
            seed=seed,
            protocol=PYTHON_PROTOCOL,
            instrumented=False,
        )
        results: dict[str, Any] = {
            "protocol": entry.get("protocol"),
            "host_waits_per_reduction": entry.get("host_waits_per_reduction"),
            "depends_on_every_step": entry.get("depends_on_every_step"),
            "dependency_carried_by": entry.get("dependency_carried_by"),
            "dependency_verdict_depth": entry.get("dependency_verdict_depth"),
            "vector_check": entry.get("vector_check"),
            "marginal": entry.get("marginal"),
            "depths": {},
        }
        for depth in depths:
            measured = entry.get("depths", {}).get(str(depth), {})
            # ``p50_ms`` is the median whole-chain time at this depth, which is the
            # quantity the native runner reports as ``total_median_us``; the entry's
            # ``per_step_us`` is that divided by the depth.
            p50_ms = measured.get("p50_ms")
            results["depths"][str(depth)] = {
                "value_check_informative": measured.get("value_check_informative"),
                "final_value_matches": measured.get("final_value_matches"),
                "observed_final_value": measured.get("observed_final_value"),
                "total_median_us": (float(p50_ms) * 1e3) if p50_ms is not None else None,
                "per_step_us": measured.get("per_step_us"),
            }
        return results
    finally:
        # Free the buffers first, then the transport: a live RCCL communicator
        # holds device handles, and leaving it open leaks them across repetitions.
        for buffer in (*work, *scratch):
            try:
                free(buffer)
            except Exception:  # noqa: BLE001 - cleanup must not mask the result
                pass
        try:
            transport.close()
        except Exception as error:  # noqa: BLE001 - reported, not raised
            results.setdefault("errors", []).append(f"transport close: {error!r}")


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


def _depth_total(arm: dict[str, Any], depth: int, key: str) -> float | None:
    entry = arm.get("depths", {}).get(str(depth)) or {}
    value = entry.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", default=",".join(str(d) for d in DEFAULT_DEPTHS))
    parser.add_argument("--count", type=int, default=5120)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="balanced repetitions; each rep runs both arms at every depth, alternating order",
    )
    parser.add_argument("--seed", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--arch", default="gfx1100")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--require-cached", action="store_true")
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="run only the Python arm, for a provenance-matched control on its own",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    depths = [int(value) for value in args.depths.split(",") if value.strip()]
    if args.reps < 1:
        raise SystemExit("--reps must be at least 1")
    environment = dict(os.environ)
    device_names = _visible_device_names(environment)

    exe = None if args.python_only else _build(
        NATIVE_SOURCE, args.build_dir, args.arch, require_cached=args.require_cached
    )

    # Balanced: each repetition runs both arms at every depth, alternating which
    # goes first, so a slow drift in machine state cannot land on one arm.
    python_samples: dict[int, list[float]] = {depth: [] for depth in depths}
    native_samples: dict[int, list[float]] = {depth: [] for depth in depths}
    python_entries: dict[int, dict[str, Any]] = {}
    native_entries: dict[int, dict[str, Any]] = {}
    python_ladders: list[dict[str, Any]] = []
    errors: list[str] = []

    def run_python_ladder() -> None:
        try:
            result = _run_python_arm(
                depths=depths,
                count=args.count,
                iterations=args.iterations,
                warmup=args.warmup,
                seed=args.seed,
                timeout_s=args.timeout_s,
            )
        except Exception as error:  # noqa: BLE001 - reported, not raised
            errors.append(f"python: {error!r}")
            return
        python_ladders.append(result)
        for ladder_depth in depths:
            python_entries[ladder_depth] = {
                **result["depths"][str(ladder_depth)],
                "protocol": result.get("protocol"),
                "host_waits_per_reduction": result.get("host_waits_per_reduction"),
                "depends_on_every_step": result.get("depends_on_every_step"),
                "vector_check": result.get("vector_check"),
            }
            total = _depth_total(result, ladder_depth, "total_median_us")
            if total is not None:
                python_samples[ladder_depth].append(total)

    def run_native_ladder() -> None:
        if exe is None:
            return
        for ladder_depth in depths:
            result = _run_native(
                exe,
                depth=ladder_depth,
                count=args.count,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            if "error" in result:
                errors.append(f"native depth {ladder_depth}: {result['error']}")
                continue
            native_entries[ladder_depth] = result
            total = result.get("total_median_us")
            if isinstance(total, (int, float)):
                native_samples[ladder_depth].append(float(total))

    # Balanced: each repetition runs both arms over the whole ladder, alternating
    # which goes first, so a slow drift in machine state cannot land on one arm.
    for rep in range(args.reps):
        order = (run_python_ladder, run_native_ladder)
        if rep % 2:
            order = tuple(reversed(order))
        for runner in order:
            runner()

    python_medians = {
        depth: statistics.median(values) for depth, values in python_samples.items() if values
    }
    # Every contributing repetition must pass its own correctness check. Keeping
    # only the last verdict would let a failed earlier repetition contribute its
    # timings while a later passing run supplied the verdict.
    python_ladder = python_ladders[-1] if python_ladders else {}
    python_verdicts = [
        {
            "depends_on_every_step": ladder.get("depends_on_every_step"),
            "dependency_carried_by": ladder.get("dependency_carried_by"),
            "dependency_verdict_depth": ladder.get("dependency_verdict_depth"),
            # The Python arm's vector check reports its three findings
            # separately, so read them rather than a single aggregate field.
            "vector_check": {
                key: (ladder.get("vector_check") or {}).get(key)
                for key in ("ranks_agree", "full_vector_matches", "rank_seeds_differ")
            }
            if isinstance(ladder.get("vector_check"), dict)
            else None,
            "final_values_match": all(
                bool((ladder.get("depths", {}).get(str(depth)) or {}).get("final_value_matches"))
                for depth in depths
            ),
        }
        for ladder in python_ladders
    ]
    python_dependency = (
        bool(python_verdicts) and all(bool(v["depends_on_every_step"]) for v in python_verdicts)
    )
    python_repetitions_passed = (
        len(python_verdicts) == args.reps
        and all(
            bool(v["depends_on_every_step"])
            and bool(v["final_values_match"])
            # A missing vector check is a failure, not an unknown: every
            # repetition must have verified the whole vector on both ranks.
            and all(
                bool((v["vector_check"] or {}).get(key))
                for key in ("ranks_agree", "full_vector_matches", "rank_seeds_differ")
            )
            for v in python_verdicts
        )
    )
    # The native runner exits non-zero on a failed check, so a recorded native
    # depth already passed its own verification; a depth with no record did not.
    native_repetitions_passed = all(
        len(native_samples[depth]) == args.reps for depth in depths
    )
    native_medians = {
        depth: statistics.median(values) for depth, values in native_samples.items() if values
    }
    python_marginal = _marginal(sorted(python_medians.items()))
    native_marginal = _marginal(sorted(native_medians.items()))

    # Provenance and protocol agreement, checked rather than assumed.
    python_count = None
    python_payload = None
    for entry in python_entries.values():
        summary = entry.get("summary") or {}
        if isinstance(summary, dict):
            python_payload = python_payload or summary.get("payload_bytes")
    native_counts = {entry.get("count") for entry in native_entries.values()}
    native_payloads = {entry.get("payload_bytes") for entry in native_entries.values()}
    native_protocols = {entry.get("protocol") for entry in native_entries.values()}
    python_protocols = {entry.get("protocol") for entry in python_entries.values()}
    native_verified = {
        _native_verification_ok(entry) for entry in native_entries.values()
    }
    match = {
        "count": (
            bool(native_counts) and native_counts == {args.count}
        ),
        "payload_bytes": (
            bool(native_payloads) and native_payloads == {args.count * 4}
        ),
        "depths": (
            sorted(python_samples) == sorted(native_samples) == sorted(depths)
            and all(python_samples[depth] and native_samples[depth] for depth in depths)
        ),
        "protocol": (
            python_protocols == {PYTHON_PROTOCOL} and native_protocols == {NATIVE_PROTOCOL}
        ),
        "python_dependency_check": python_dependency,
        "python_repetitions_passed": python_repetitions_passed,
        "native_repetitions_passed": native_repetitions_passed,
        "native_verification": native_verified == {True},
        "balanced_repetitions": all(
            len(python_samples[depth]) == args.reps and len(native_samples[depth]) == args.reps
            for depth in depths
        ),
        "two_devices": len(device_names) >= 2,
    }
    matched = all(match.values())
    python_us = python_marginal.get("overall_us_per_step")
    native_us = native_marginal.get("overall_us_per_step")
    ratio = (python_us / native_us) if python_us and native_us else None

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "tp2-staged-exchange-native-ab",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": NATIVE_PROTOCOL,
        "count": args.count,
        "payload_bytes": args.count * 4,
        "depths": depths,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "reps": args.reps,
        "arch": args.arch,
        "provenance": {
            "git_commit": _git_commit(),
            "python_arm": {
                "path": str(PYTHON_SOURCE.relative_to(REPO_ROOT)),
                "sha256": _sha256(PYTHON_SOURCE),
                "entry_point": "staged_exchange_batched",
            },
            "native_arm": {
                "path": str(NATIVE_SOURCE.relative_to(REPO_ROOT)),
                "sha256": _sha256(NATIVE_SOURCE),
                "binary_sha256": _sha256(exe) if exe else None,
                "compile_flags": ["-O2", f"--offload-arch={args.arch}"],
            },
            "devices": {
                "hip_visible_devices": environment.get("HIP_VISIBLE_DEVICES", ""),
                "visible_names": device_names,
            },
        },
        "arms": {
            "python": {
                "ladder": {
                    "depends_on_every_step": python_dependency,
                    "dependency_carried_by": python_ladder.get("dependency_carried_by"),
                    "dependency_verdict_depth": python_ladder.get("dependency_verdict_depth"),
                    "vector_check": python_ladder.get("vector_check"),
                    "marginal": python_ladder.get("marginal"),
                    "repetitions": python_verdicts,
                    "every_repetition_passed": python_repetitions_passed,
                },
                "depths": {
                    str(depth): {
                        "total_median_us": python_medians.get(depth),
                        "samples_us": python_samples[depth],
                        **python_entries.get(depth, {}),
                    }
                    for depth in depths
                },
                "marginal": python_marginal,
            },
            "native": {
                "depths": {
                    str(depth): {
                        "total_median_us": native_medians.get(depth),
                        "samples_us": native_samples[depth],
                        **native_entries.get(depth, {}),
                    }
                    for depth in depths
                },
                "marginal": native_marginal,
            },
        },
        "provenance_match": match,
        "comparison": {
            "python_us_per_step": python_us,
            "native_us_per_step": native_us,
            "python_over_native": ratio,
            "balanced_reps": args.reps,
            "matched": matched,
            "provisional": not matched,
            "provisional_reason": (
                None
                if matched
                else "unmatched: " + ", ".join(sorted(k for k, v in match.items() if not v))
            ),
            "basis": "both arms rerun in one session, same depths, same payload, "
            "alternating order per repetition, ladder slope over the depth medians",
            "phase_counters_are_not_transferable": (
                "earlier submission changes the exposed wait, so a phase that "
                "shrinks in one arm can grow in the other; compare totals"
            ),
        },
        "errors": errors,
    }

    text = json.dumps(report, indent=2, sort_keys=False) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
        print(f"wrote {args.json}", file=sys.stderr)
    else:
        print(text)

    label = "matched" if matched else "PROVISIONAL"
    print(
        f"[{label}] python {python_us and round(python_us, 2)} us/step vs "
        f"native {native_us and round(native_us, 2)} us/step"
        + (f" -> {ratio:.3f}x" if ratio else "")
        + ("" if matched else f" ({report['comparison']['provisional_reason']})"),
        file=sys.stderr,
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
