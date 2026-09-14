#!/usr/bin/env python3
"""Localize a hang in the dependent-chain modes by printing per-mode progress.

Not a retained harness: it drives the same functions the bench uses, one depth
at a time, so a stall names the mode and depth that caused it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hipengine.core.device import Device
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_host_array_to_device, free, malloc
from hipengine.distributed.plan import DistributedPlan
from hipengine.distributed.rccl import RcclTransport
from scripts.tp_collective_bench import Case, encode_values, _measure_dependent_chain

DEVICES = [0, 1]
HIDDEN = 5120
DEPTHS = [int(chunk) for chunk in sys.argv[1].split(",")]
ITERATIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
MODES = [chunk for chunk in sys.argv[3].split(",")] if len(sys.argv) > 3 else None


def main() -> int:
    runtime = get_hip_runtime()
    case = Case(op="all_reduce", rows=1, dtype="fp32", hidden_size=HIDDEN)
    plan = DistributedPlan.resolve(DEVICES, hidden_size=HIDDEN, algorithm="rccl")
    transport = RcclTransport([spec.device for spec in plan.ranks], runtime=runtime, init_timeout_s=60.0)
    world = transport.world_size
    work = [malloc(case.payload_bytes, device=Device("hip", rank)) for rank in range(world)]
    scratch = [malloc(case.payload_bytes, device=Device("hip", rank)) for rank in range(world)]
    for rank in range(world):
        copy_host_array_to_device(work[rank], encode_values([1.0] * case.count, case.dtype))
    started = time.time()
    try:
        for depth in DEPTHS:
            print(f"[{time.time() - started:7.1f}s] depth {depth} start", flush=True)
            report = _measure_dependent_chain(
                transport=transport,
                runtime=runtime,
                case=case,
                work=work,
                scratch=scratch,
                depths=(depth,),
                iterations=ITERATIONS,
                warmup=1,
                seed=1.0,
                timeout_s=30.0,
                modes=tuple(MODES) if MODES else None,
            )
            for mode, mode_report in report["modes"].items():
                depth_report = (mode_report.get("depths") or {}).get(str(depth))
                if depth_report is None:
                    print(
                        f"[{time.time() - started:7.1f}s]   {mode}: captured="
                        f"{mode_report.get('captured')} error={mode_report.get('error')}",
                        flush=True,
                    )
                    continue
                print(
                    f"[{time.time() - started:7.1f}s]   {mode}: p50={depth_report['p50_ms']:.3f} ms "
                    f"({depth_report['per_step_us']:.1f} us/step) matches="
                    f"{depth_report['final_value_matches']} observed="
                    f"{depth_report['observed_final_value']} expected="
                    f"{depth_report['expected_final_value']}",
                    flush=True,
                )
            print(f"[{time.time() - started:7.1f}s] depth {depth} done", flush=True)
    finally:
        for buffer in work + scratch:
            free(buffer, runtime=runtime)
        transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
