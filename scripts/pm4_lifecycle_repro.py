#!/usr/bin/env python3
"""Standalone same-HSACO HIP/direct-AQL/retained-PM4 lifecycle reproducer.

Safe defaults reuse one queue and one executable for four tiny submissions.
Every native submit arm that recreates packet resources is rejected unless the
operator passes ``--ack-reset-risk``. The script never retries a failed cycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

Transport = Literal["hipgraph", "aql", "pm4"]
QueueMode = Literal["reuse", "recreate"]
ResourceMode = Literal["reuse", "recreate"]
AllocationMode = Literal["hip", "hsa"]
BufferMode = Literal["reuse", "recreate"]


@dataclass(frozen=True, slots=True)
class ReproConfig:
    transport: Transport = "pm4"
    cycles: int = 4
    queue_mode: QueueMode = "reuse"
    resource_mode: ResourceMode = "reuse"
    allocation_mode: AllocationMode = "hip"
    buffer_mode: BufferMode = "reuse"
    submit: bool = True
    timestamps: bool = False
    quarantine_generations: int = 0
    reset_risk_acknowledged: bool = False
    n: int = 257
    seed: int = 20260807
    timeout_seconds: float = 5.0
    destructive: bool = False

    def validated(self) -> "ReproConfig":
        if self.transport not in {"hipgraph", "aql", "pm4"}:
            raise ValueError("transport must be hipgraph, aql, or pm4")
        if not 1 <= self.cycles <= 1_000_000:
            raise ValueError("cycles must be in 1..1000000")
        if self.queue_mode not in {"reuse", "recreate"}:
            raise ValueError("queue-mode must be reuse or recreate")
        if self.resource_mode not in {"reuse", "recreate"}:
            raise ValueError("resource-mode must be reuse or recreate")
        if self.allocation_mode not in {"hip", "hsa"}:
            raise ValueError("allocation-mode must be hip or hsa")
        if self.buffer_mode not in {"reuse", "recreate"}:
            raise ValueError("buffer-mode must be reuse or recreate")
        if self.queue_mode == "recreate" and self.resource_mode != "recreate":
            raise ValueError("queue-mode=recreate requires resource-mode=recreate")
        if self.quarantine_generations < 0 or self.quarantine_generations > 4096:
            raise ValueError("quarantine-generations must be in 0..4096")
        if self.quarantine_generations and self.queue_mode != "recreate":
            raise ValueError("quarantine requires queue-mode=recreate")
        if self.timestamps and self.transport != "pm4":
            raise ValueError("timestamps are supported only by retained PM4")
        if (
            self.transport != "hipgraph"
            and self.buffer_mode == "recreate"
            and self.resource_mode != "recreate"
        ):
            raise ValueError("native buffer-mode=recreate requires resource-mode=recreate")
        if self.transport == "hipgraph" and self.queue_mode == "recreate" and self.allocation_mode == "hip":
            raise ValueError("hipgraph+HIP allocation has no HSA queue to recreate")
        if self.n <= 0 or self.n > (1 << 30):
            raise ValueError("n must be positive and bounded")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout-seconds must be positive")

        config = self
        if self.allocation_mode == "hsa" and self.queue_mode == "recreate":
            config = replace(config, buffer_mode="recreate")
        destructive = bool(
            config.submit
            and config.transport in {"aql", "pm4"}
            and config.resource_mode == "recreate"
        )
        config = replace(config, destructive=destructive)
        if destructive and not config.reset_risk_acknowledged:
            raise ValueError(
                "reset-risk native submit/resource-recreate is blocked; re-run only after "
                "arranging journal/coredump collection and pass --ack-reset-risk"
            )
        return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("hipgraph", "aql", "pm4"), default="pm4")
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--queue-mode", choices=("reuse", "recreate"), default="reuse")
    parser.add_argument("--resource-mode", choices=("reuse", "recreate"), default="reuse")
    parser.add_argument("--allocation-mode", choices=("hip", "hsa"), default="hip")
    parser.add_argument("--buffer-mode", choices=("reuse", "recreate"), default="reuse")
    submit = parser.add_mutually_exclusive_group()
    submit.add_argument("--submit", dest="submit", action="store_true", default=True)
    submit.add_argument("--no-submit", dest="submit", action="store_false")
    parser.add_argument("--timestamps", action="store_true")
    parser.add_argument("--quarantine-generations", type=int, default=0)
    parser.add_argument(
        "--ack-reset-risk",
        action="store_true",
        help="acknowledge that native submit/resource recreation may VM-fault/reset the GPU",
    )
    parser.add_argument("--stress", action="store_true", help="select 128 recreate/submit cycles")
    parser.add_argument("--n", type=int, default=257)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--print-plan", action="store_true")
    return parser


def plan_from_args(args: argparse.Namespace) -> ReproConfig:
    cycles = 128 if args.stress and args.cycles == 4 else args.cycles
    queue_mode = "recreate" if args.stress else args.queue_mode
    resource_mode = "recreate" if args.stress else args.resource_mode
    return ReproConfig(
        transport=args.transport,
        cycles=cycles,
        queue_mode=queue_mode,
        resource_mode=resource_mode,
        allocation_mode=args.allocation_mode,
        buffer_mode=args.buffer_mode,
        submit=args.submit,
        timestamps=args.timestamps,
        quarantine_generations=args.quarantine_generations,
        reset_risk_acknowledged=args.ack_reset_risk,
        n=args.n,
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
    ).validated()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _compiler_version(path: Path | None) -> str | None:
    return None if path is None else path.expanduser().read_text(encoding="utf-8").strip()


def _attempt_teardown(label: str, operation: Callable[[], None]) -> dict[str, Any]:
    try:
        operation()
    except Exception as exc:
        return {
            "operation": label,
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {"operation": label, "status": "pass"}


def _close_generation(generation: dict[str, Any]) -> list[dict[str, Any]]:
    """Close one retired generation in pointee-safe order without losing errors."""

    statuses: list[dict[str, Any]] = []

    def attempt(label: str, key: str, operation: Callable[[], None]) -> bool:
        status = _attempt_teardown(label, operation)
        statuses.append(status)
        if status["status"] == "pass":
            generation[key] = None
            return True
        return False

    executable = generation.get("executable")
    if executable is not None and not attempt("executable.close", "executable", executable.close):
        statuses.append(
            {
                "operation": "generation.remaining_resources",
                "status": "skipped",
                "reason": "executable close did not prove packet-pointee release",
            }
        )
        return statuses

    runtime = generation.get("runtime")
    graph_exec = int(generation.get("graph_exec") or 0)
    if graph_exec and not attempt(
        "hip_graph_exec.destroy",
        "graph_exec",
        lambda: runtime.graph_exec_destroy(graph_exec),
    ):
        statuses.append(
            {
                "operation": "generation.remaining_resources",
                "status": "skipped",
                "reason": "HIP graph executable destruction failed",
            }
        )
        return statuses
    graph = int(generation.get("graph") or 0)
    if graph and not attempt(
        "hip_graph.destroy",
        "graph",
        lambda: runtime.graph_destroy(graph),
    ):
        statuses.append(
            {
                "operation": "generation.remaining_resources",
                "status": "skipped",
                "reason": "HIP graph destruction failed",
            }
        )
        return statuses

    hsa_buffers = list(generation.get("hsa_buffers", []))
    for reverse_index, buffer in enumerate(reversed(hsa_buffers)):
        index = len(hsa_buffers) - 1 - reverse_index
        status = _attempt_teardown(f"hsa_buffer[{index}].close", buffer.close)
        statuses.append(status)
        if status["status"] != "pass":
            statuses.append(
                {
                    "operation": "generation.remaining_resources",
                    "status": "skipped",
                    "reason": "HSA buffer release failed",
                }
            )
            return statuses
    generation["hsa_buffers"] = []

    hip_buffers = list(generation.get("hip_buffers", []))
    for reverse_index, buffer in enumerate(reversed(hip_buffers)):
        index = len(hip_buffers) - 1 - reverse_index
        status = _attempt_teardown(
            f"hip_buffer[{index}].free",
            lambda buffer=buffer: generation["free_hip_buffer"](buffer),
        )
        statuses.append(status)
        if status["status"] != "pass":
            statuses.append(
                {
                    "operation": "generation.remaining_resources",
                    "status": "skipped",
                    "reason": "HIP buffer release failed",
                }
            )
            return statuses
    generation["hip_buffers"] = []

    context = generation.get("context")
    if context is not None:
        attempt("context.close", "context", context.close)
    return statuses


def _capture_graph(runtime, library, pointers: tuple[int, int, int], n: int, stream: int):
    from hipengine.core.pm4 import inspect_hip_graph
    from hipengine.kernels.hip_gfx1100.smoke import smoke_add_f32

    smoke_add_f32(*pointers, n, library=library, runtime=runtime)
    runtime.device_synchronize()
    runtime.stream_begin_capture(stream, 2)
    smoke_add_f32(*pointers, n, stream=stream, library=library, runtime=runtime)
    graph = runtime.stream_end_capture(stream)
    manifest = inspect_hip_graph(runtime, graph, gfx_arch="gfx1100", stream=stream)
    return graph, manifest


def run_reproducer(
    config: ReproConfig,
    *,
    compiler_version: str | None = None,
    require_cached_build: bool = False,
) -> dict[str, Any]:
    """Run the declared matrix arm once; never retry a failed generation."""

    config = config.validated()
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.core.pm4 import NativePm4Context
    from hipengine.kernels.hip_gfx1100.smoke import build_smoke_add

    runtime = get_hip_runtime()
    pci_bdf = runtime.device_pci_bus_id()
    library = build_smoke_add(
        load=True,
        compiler_version=compiler_version,
        require_cached=require_cached_build,
    )
    stream = runtime.stream_create(nonblocking=True)
    nbytes = config.n * np.dtype(np.float32).itemsize
    rng = np.random.default_rng(config.seed)
    cycles: list[dict[str, Any]] = []
    quarantine: deque[dict[str, Any]] = deque()
    reusable_context = None
    reusable_executable = None
    reusable_hsa_buffers: list[Any] = []
    hip_buffers: list[Any] = []
    reusable_graph = 0
    reusable_graph_exec = 0
    reusable_manifest = None
    failure: Exception | None = None
    unsafe_native_failure = False
    result: dict[str, Any] | None = None
    started_ns = time.monotonic_ns()

    def make_context():
        return NativePm4Context.create(
            pci_bdf=pci_bdf,
            gfx_arch="gfx1100",
            compiler_version=compiler_version,
            require_cached=require_cached_build,
        )

    def make_hsa_buffers(context):
        return [context.allocate_buffer(nbytes) for _ in range(3)]

    def addresses(hip, hsa) -> tuple[int, int, int]:
        if config.allocation_mode == "hip":
            return tuple(buffer.ptr for buffer in hip)  # type: ignore[return-value]
        return tuple(buffer.address for buffer in hsa)  # type: ignore[return-value]

    try:
        if config.allocation_mode == "hip" and config.buffer_mode == "reuse":
            hip_buffers = [malloc(nbytes) for _ in range(3)]

        needs_context = config.transport != "hipgraph" or config.allocation_mode == "hsa"
        if config.queue_mode == "reuse" and needs_context:
            reusable_context = make_context()
            if config.allocation_mode == "hsa":
                reusable_hsa_buffers = make_hsa_buffers(reusable_context)

        if config.buffer_mode == "reuse":
            reusable_graph, reusable_manifest = _capture_graph(
                runtime,
                library,
                addresses(hip_buffers, reusable_hsa_buffers),
                config.n,
                stream,
            )
            if config.transport == "hipgraph":
                reusable_graph_exec = runtime.graph_instantiate(reusable_graph)
            elif config.resource_mode == "reuse":
                reusable_executable = reusable_context.instantiate(
                    reusable_manifest, timestamps=config.timestamps
                )

        for cycle_index in range(config.cycles):
            cycle_start = time.monotonic_ns()
            context = reusable_context
            hsa_buffers = reusable_hsa_buffers
            local_hip_buffers: list[Any] = []
            graph = reusable_graph
            graph_exec = reusable_graph_exec
            manifest = reusable_manifest
            executable = reusable_executable
            owns_graph = False
            owns_graph_exec = False
            owns_context = False
            owns_executable = False
            owns_hsa_buffers = False
            owns_hip_buffers = False
            cycle: dict[str, Any] = {
                "cycle": cycle_index,
                "status": "started",
                "transport": config.transport,
            }
            try:
                if config.queue_mode == "recreate" and needs_context:
                    context = make_context()
                    owns_context = True
                if config.allocation_mode == "hsa" and config.buffer_mode == "recreate":
                    hsa_buffers = make_hsa_buffers(context)
                    owns_hsa_buffers = True
                if config.allocation_mode == "hip" and config.buffer_mode == "recreate":
                    local_hip_buffers = [malloc(nbytes) for _ in range(3)]
                    owns_hip_buffers = True
                active_hip = local_hip_buffers if local_hip_buffers else hip_buffers
                pointers = addresses(active_hip, hsa_buffers)

                if config.buffer_mode == "recreate":
                    graph, manifest = _capture_graph(runtime, library, pointers, config.n, stream)
                    owns_graph = True
                    if config.transport == "hipgraph":
                        graph_exec = runtime.graph_instantiate(graph)
                        owns_graph_exec = True
                if config.transport != "hipgraph" and (
                    config.resource_mode == "recreate" or executable is None
                ):
                    executable = context.instantiate(manifest, timestamps=config.timestamps)
                    owns_executable = True

                before = None if context is None else context.provenance()
                exec_before = None if executable is None else executable.provenance()
                a = np.ascontiguousarray(rng.standard_normal(config.n), dtype=np.float32)
                b = np.ascontiguousarray(rng.standard_normal(config.n), dtype=np.float32)
                expected = np.ascontiguousarray(a + b)
                if config.allocation_mode == "hip":
                    copy_host_to_device(active_hip[0], host_array_ptr(a), a.nbytes, runtime=runtime)
                    copy_host_to_device(active_hip[1], host_array_ptr(b), b.nbytes, runtime=runtime)
                    runtime.memset(active_hip[2].ptr, 0, nbytes)
                    runtime.device_synchronize()
                else:
                    hsa_buffers[0].write(a.tobytes())
                    hsa_buffers[1].write(b.tobytes())
                    hsa_buffers[2].write(bytes(nbytes))

                submit_start = time.monotonic_ns()
                if config.submit:
                    if config.transport == "hipgraph":
                        runtime.graph_launch(graph_exec, stream)
                        runtime.stream_synchronize(stream)
                    else:
                        runtime.stream_synchronize(stream)
                        executable.launch(config.transport, timeout_seconds=config.timeout_seconds)
                submit_end = time.monotonic_ns()

                if config.submit:
                    if config.allocation_mode == "hip":
                        output = np.empty(config.n, dtype=np.float32)
                        copy_device_to_host(
                            host_array_ptr(output), active_hip[2], output.nbytes, runtime=runtime
                        )
                    else:
                        output = np.frombuffer(hsa_buffers[2].read(), dtype=np.float32).copy()
                    correct = bool(np.array_equal(output, expected))
                    if not correct:
                        raise RuntimeError("GPU output differs from the exact CPU addition oracle")
                    output_hash = _sha256(output.tobytes())
                else:
                    correct = None
                    output_hash = None

                after = None if context is None else context.provenance()
                exec_after = None if executable is None else executable.provenance()
                cycle.update(
                    {
                        "status": "pass",
                        "submit": config.submit,
                        "correct": correct,
                        "output_sha256": output_hash,
                        "expected_sha256": _sha256(expected.tobytes()) if config.submit else None,
                        "buffer_addresses": list(pointers),
                        "graph_handle": graph,
                        "graph_fingerprint": manifest.fingerprint,
                        "node_count": len(manifest.nodes),
                        "hsaco_sha256": sorted({node.hsaco_sha256 for node in manifest.nodes}),
                        "context_before": before,
                        "context_after": after,
                        "executable_before": exec_before,
                        "executable_after": exec_after,
                        "submit_ns": submit_end - submit_start,
                    }
                )
            except Exception as exc:
                failure = exc
                unsafe_native_failure = bool(config.transport != "hipgraph" and config.submit)
                cycle.update(
                    {
                        "status": "fail",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "fail_stop": unsafe_native_failure,
                    }
                )
                for label, owner in (("context_failure", context), ("executable_failure", executable)):
                    if owner is not None:
                        try:
                            cycle[label] = owner.provenance()
                        except Exception as provenance_error:
                            cycle[label] = {"error": str(provenance_error)}
                cycles.append(cycle)
            finally:
                cycle["cycle_ns"] = time.monotonic_ns() - cycle_start
                if not cycles or cycles[-1] is not cycle:
                    cycles.append(cycle)
                if unsafe_native_failure:
                    cycle["cleanup"] = "quarantined_until_process_exit"
                else:
                    package = {
                        "context": context if owns_context else None,
                        "executable": executable if owns_executable else None,
                        "hsa_buffers": hsa_buffers if owns_hsa_buffers else [],
                        "hip_buffers": local_hip_buffers if owns_hip_buffers else [],
                        "graph_exec": graph_exec if owns_graph_exec else 0,
                        "graph": graph if owns_graph else 0,
                        "runtime": runtime,
                        "free_hip_buffer": lambda buffer: free(buffer, runtime=runtime),
                        "retired_cycle": cycle_index,
                    }
                    cycle_teardown: list[dict[str, Any]] = []
                    if owns_context and config.quarantine_generations:
                        cycle["queue_before_retire"] = context.provenance()
                        retirement = _attempt_teardown(
                            "context.retire_queue", context.retire_queue
                        )
                        cycle_teardown.append(retirement)
                        if retirement["status"] == "pass":
                            cycle["context_after_queue_retire"] = context.provenance()
                            quarantine.append(package)
                            while len(quarantine) > config.quarantine_generations:
                                cycle_teardown.extend(_close_generation(quarantine.popleft()))
                        else:
                            unsafe_native_failure = True
                            cycle["cleanup"] = "quarantined_until_process_exit"
                    else:
                        cycle_teardown.extend(_close_generation(package))
                    if cycle_teardown:
                        cycle["teardown"] = cycle_teardown
                    teardown_failures = [
                        status
                        for status in cycle_teardown
                        if status["status"] in {"fail", "skipped"}
                    ]
                    if teardown_failures:
                        cycle["status"] = "fail"
                        cycle.setdefault("error_type", "LifecycleTeardownError")
                        cycle.setdefault(
                            "error",
                            f"lifecycle teardown failed at {teardown_failures[0]['operation']}",
                        )
                        if failure is None:
                            failure = RuntimeError(cycle["error"])
                        if config.transport != "hipgraph":
                            unsafe_native_failure = True
                            cycle["cleanup"] = "quarantined_until_process_exit"
            if failure is not None:
                break

        result = {
            "schema_version": 1,
            "kind": "hipengine_pm4_lifecycle_reproducer",
            "status": "fail" if failure is not None else "pass",
            "config": asdict(config),
            "hardware": {
                "process_id": os.getpid(),
                "hip_ordinal": runtime.current_device(),
                "pci_bdf": pci_bdf,
                "gfx_arch": "gfx1100",
                "platform": platform.platform(),
                "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
                "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
                "gpu_max_hw_queues": os.environ.get("GPU_MAX_HW_QUEUES"),
            },
            "cycles": cycles,
            "summary": {
                "cycles_requested": config.cycles,
                "cycles_passed": sum(cycle["status"] == "pass" for cycle in cycles),
                "first_failed_cycle": next(
                    (cycle["cycle"] for cycle in cycles if cycle["status"] != "pass"), None
                ),
                "elapsed_ns": time.monotonic_ns() - started_ns,
                "quarantine_depth": config.quarantine_generations,
            },
        }
        return result
    finally:
        final_cleanup: list[dict[str, Any]] = []
        if unsafe_native_failure:
            final_cleanup.append(
                {
                    "operation": "process_exit_quarantine",
                    "status": "skipped",
                    "reason": "native failure left GPU-visible resources quarantined",
                }
            )
        else:
            while quarantine:
                final_cleanup.extend(_close_generation(quarantine.popleft()))
            final_cleanup.extend(
                _close_generation(
                    {
                        "context": reusable_context,
                        "executable": reusable_executable,
                        "hsa_buffers": reusable_hsa_buffers,
                        "hip_buffers": hip_buffers,
                        "graph_exec": reusable_graph_exec,
                        "graph": reusable_graph,
                        "runtime": runtime,
                        "free_hip_buffer": lambda buffer: free(buffer, runtime=runtime),
                    }
                )
            )
            final_cleanup.append(
                _attempt_teardown("stream.destroy", lambda: runtime.stream_destroy(stream))
            )
        if result is not None:
            result["final_cleanup"] = final_cleanup
            cleanup_failures = [
                status
                for status in final_cleanup
                if status["status"] in {"fail", "skipped"}
            ]
            if cleanup_failures:
                result["status"] = "fail"
                result["summary"]["final_cleanup_passed"] = False
                result["summary"]["final_cleanup_first_failure"] = cleanup_failures[0]
            else:
                result["summary"]["final_cleanup_passed"] = True


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = plan_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.print_plan:
        print(json.dumps(asdict(config), sort_keys=True, indent=2))
        return 0

    result: dict[str, Any]
    try:
        result = run_reproducer(
            config,
            compiler_version=_compiler_version(args.compiler_version_file),
            require_cached_build=args.require_cached_build,
        )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "kind": "hipengine_pm4_lifecycle_reproducer",
            "status": "fail",
            "config": asdict(config),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    encoded = json.dumps(result, sort_keys=True, indent=2)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
