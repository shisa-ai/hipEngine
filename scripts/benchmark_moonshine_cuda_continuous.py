#!/usr/bin/env python3
"""Benchmark Moonshine continuous refill against static lockstep B=2/4/8."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__:
    from scripts.c8_report_common import (
        dependency_adjusted_bytes,
        git_state,
        implementation_sha256,
    )
else:
    from c8_report_common import (
        dependency_adjusted_bytes,
        git_state,
        implementation_sha256,
    )

_EOS = 2
_DEFAULT_FIXTURES = (
    "audio-hai-fp16",
    "audio-konichiwa-fp16",
    "audio-konichiwa.ogenkidesuka-fp16",
    "audio-kumbawa-fp16",
    "audio-sosososo-fp16",
    "audio-sumimasen-fp16",
)


@dataclass(frozen=True)
class Fixture:
    name: str
    keys: tuple[np.ndarray, ...]
    values: tuple[np.ndarray, ...]
    mask: np.ndarray
    reference: tuple[int, ...]


@dataclass(frozen=True)
class WorkloadResult:
    wall_ms: float
    request_latency_ms: tuple[float, ...]
    outputs: dict[str, tuple[int, ...]]


def percentile(values: list[float], value: float) -> float:
    if not values:
        raise ValueError("percentile needs at least one sample")
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def timing_summary(samples: list[WorkloadResult], request_count: int) -> dict[str, Any]:
    if not samples or request_count <= 0:
        raise ValueError("timing summary needs samples and requests")
    wall = [sample.wall_ms for sample in samples]
    throughput = [request_count * 1000.0 / value for value in wall]
    latency = [value for sample in samples for value in sample.request_latency_ms]
    return {
        "sample_count": len(samples),
        "request_count_per_sample": request_count,
        "wall_ms_raw": wall,
        "requests_per_s_raw": throughput,
        "request_latency_ms_raw": [list(sample.request_latency_ms) for sample in samples],
        "wall_median_ms": statistics.median(wall),
        "wall_p95_ms": percentile(wall, 95),
        "requests_per_s_median": statistics.median(throughput),
        "requests_per_s_p05": percentile(throughput, 5),
        "request_latency_p50_ms": percentile(latency, 50),
        "request_latency_p95_ms": percentile(latency, 95),
    }


def _pad_cross(array: np.ndarray, frames: int) -> np.ndarray:
    source = np.asarray(array, dtype=np.float16)
    if source.ndim != 4 or source.shape[0] != 1:
        raise ValueError(f"unexpected fixture cross-cache shape: {source.shape}")
    output = np.zeros((source.shape[1], frames, source.shape[3]), dtype=np.float16)
    output[:, : source.shape[2], :] = source[0]
    return output


def load_fixtures(fixture_dir: Path, names: tuple[str, ...]) -> tuple[list[Fixture], int]:
    manifests = {
        name: json.loads((fixture_dir / f"{name}.json").read_text()) for name in names
    }
    frames = max(int(value["input"]["encoder_frames"]) for value in manifests.values())
    fixtures: list[Fixture] = []
    for name in names:
        manifest = manifests[name]
        real_frames = int(manifest["input"]["encoder_frames"])
        with np.load(fixture_dir / f"{name}.npz") as arrays:
            keys = tuple(
                _pad_cross(arrays[f"cross.layer_{layer}.key"], frames)
                for layer in range(8)
            )
            values = tuple(
                _pad_cross(arrays[f"cross.layer_{layer}.value"], frames)
                for layer in range(8)
            )
        mask = np.zeros(frames, dtype=np.int32)
        mask[:real_frames] = 1
        reference = tuple(int(value) for value in manifest["decoder"]["token_ids"])
        fixtures.append(
            Fixture(
                name=name,
                keys=keys,
                values=values,
                mask=mask,
                reference=reference,
            )
        )
    return fixtures, frames


def expected_output(fixture: Fixture) -> tuple[int, ...]:
    eos = fixture.reference.index(_EOS, 1)
    return fixture.reference[1 : eos + 1]


def workload_fixtures(fixtures: list[Fixture], request_count: int) -> list[Fixture]:
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    return [fixtures[index % len(fixtures)] for index in range(request_count)]


def run_continuous_workload(scheduler, workload: list[Fixture], run_index: int) -> WorkloadResult:
    started = time.perf_counter_ns()
    submitted: dict[str, int] = {}
    expected: dict[str, tuple[int, ...]] = {}
    for index, fixture in enumerate(workload):
        request_id = f"continuous-{run_index}-{index}"
        scheduler.submit(
            request_id,
            fixture.keys,
            fixture.values,
            mask=fixture.mask,
            seed_token_id=fixture.reference[0],
        )
        submitted[request_id] = time.perf_counter_ns()
        expected[request_id] = expected_output(fixture)

    completed_at: dict[str, int] = {}
    outputs: dict[str, tuple[int, ...]] = {}
    while not scheduler.idle:
        step = scheduler.step()
        now = time.perf_counter_ns()
        for request_id in step.completed:
            result = scheduler.take_completed(request_id)
            if result.reason != "eos":
                raise AssertionError(f"continuous request {request_id} ended by {result.reason}")
            outputs[request_id] = result.tokens
            completed_at[request_id] = now
    ended = time.perf_counter_ns()
    if outputs != expected:
        raise AssertionError("continuous outputs differ from retained fixture references")
    return WorkloadResult(
        wall_ms=(ended - started) * 1.0e-6,
        request_latency_ms=tuple(
            (completed_at[request_id] - submitted[request_id]) * 1.0e-6
            for request_id in submitted
        ),
        outputs=outputs,
    )


def _batch_arrays(rows: list[Fixture], batch: int):
    padded = list(rows)
    while len(padded) < batch:
        padded.append(rows[0])
    keys = [
        np.stack([fixture.keys[layer] for fixture in padded]) for layer in range(8)
    ]
    values = [
        np.stack([fixture.values[layer] for fixture in padded]) for layer in range(8)
    ]
    masks = np.stack([fixture.mask for fixture in padded])
    seeds = np.asarray([fixture.reference[0] for fixture in padded], dtype=np.int64)
    return padded, keys, values, masks, seeds


def run_lockstep_workload(decoder, workload: list[Fixture], run_index: int) -> WorkloadResult:
    started = time.perf_counter_ns()
    submitted = {
        f"lockstep-{run_index}-{index}": started for index in range(len(workload))
    }
    expected = {
        f"lockstep-{run_index}-{index}": expected_output(fixture)
        for index, fixture in enumerate(workload)
    }
    completed_at: dict[str, int] = {}
    outputs: dict[str, tuple[int, ...]] = {}
    batch = decoder.max_batch
    for offset in range(0, len(workload), batch):
        rows = workload[offset : offset + batch]
        padded, keys, values, masks, tokens = _batch_arrays(rows, batch)
        decoder.load_cross_cache_batch(keys, values, masks=masks)
        if not decoder.batch_token_graph_contract()["captured"]:
            decoder.capture_batch_token_graphs()
        done = np.ones(batch, dtype=bool)
        done[: len(rows)] = False
        generated: list[list[int]] = [[] for _ in rows]
        for position in range(decoder.spec.self_cache_capacity):
            decoder.set_batch_decode_state(tokens=tokens.tolist(), position=position)
            decoder.graph_batch_token_step()
            tokens = decoder.read_tokens()
            now = time.perf_counter_ns()
            for row in range(len(rows)):
                if done[row]:
                    continue
                token = int(tokens[row])
                generated[row].append(token)
                if token == _EOS:
                    done[row] = True
                    request_id = f"lockstep-{run_index}-{offset + row}"
                    completed_at[request_id] = now
            if bool(done.all()):
                break
        if not bool(done.all()):
            raise AssertionError("lockstep batch did not reach EOS")
        for row, fixture in enumerate(rows):
            request_id = f"lockstep-{run_index}-{offset + row}"
            outputs[request_id] = tuple(generated[row])
            if outputs[request_id] != expected[request_id]:
                raise AssertionError(
                    f"lockstep output differs for {fixture.name}: {outputs[request_id]}"
                )
    ended = time.perf_counter_ns()
    if outputs != expected:
        raise AssertionError("lockstep outputs differ from retained fixture references")
    return WorkloadResult(
        wall_ms=(ended - started) * 1.0e-6,
        request_latency_ms=tuple(
            (completed_at[request_id] - submitted[request_id]) * 1.0e-6
            for request_id in submitted
        ),
        outputs=outputs,
    )


def benchmark_route(
    run: Callable[[int], WorkloadResult], *, warmup: int, iterations: int
) -> list[WorkloadResult]:
    for index in range(warmup):
        run(-index - 1)
    return [run(index) for index in range(iterations)]


def _gpu_processes(gpu_index: int) -> list[dict[str, str]]:
    gpu_output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    uuids = {
        int(index.strip()): uuid.strip()
        for index, uuid in (line.split(",", 1) for line in gpu_output.splitlines())
    }
    try:
        selected_uuid = uuids[gpu_index]
    except KeyError as error:
        raise ValueError(f"GPU index {gpu_index} is not visible to nvidia-smi") from error
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True).strip()
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        uuid, pid, name, memory = [part.strip() for part in line.split(",", 3)]
        if uuid == selected_uuid:
            rows.append(
                {
                    "gpu_uuid": uuid,
                    "pid": pid,
                    "process": name,
                    "memory_mib": memory,
                }
            )
    return rows


def parse_batches(value: str) -> tuple[int, ...]:
    try:
        batches = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError("batches must be comma-separated integers") from error
    if not batches or any(batch <= 0 for batch in batches) or len(set(batches)) != len(batches):
        raise argparse.ArgumentTypeError("batches must be unique positive integers")
    return batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--batches", type=parse_batches, default=parse_batches("2,4,8"))
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-graphs", type=int, default=8)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--exclusive-gpu", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.requests <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("requests/iterations must be positive and warmup non-negative")
    if max(args.batches) > args.requests:
        raise ValueError("requests must be at least the largest batch")
    root = Path(__file__).resolve().parents[1]
    state = git_state(root)
    if state["dirty"] and not args.allow_dirty:
        raise RuntimeError("refusing to publish continuous benchmark from a dirty tree")
    before_processes = _gpu_processes(args.gpu_index)
    if args.exclusive_gpu and before_processes:
        raise RuntimeError(f"GPU is not exclusive before the run: {before_processes}")

    import torch

    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.core.device import Device
    from hipengine.loading.moonshine import load_moonshine_model
    from hipengine.runtime.moonshine_cuda_batch import MoonshineCudaBatchRuntime
    from hipengine.runtime.moonshine_cuda_continuous import (
        MoonshineCudaExactContinuousBatchRuntime,
    )

    runtime = get_cuda_runtime()
    runtime.set_device(args.gpu_index)
    fixtures, frames = load_fixtures(args.fixture_dir, _DEFAULT_FIXTURES)
    workload = workload_fixtures(fixtures, args.requests)
    loaded = load_moonshine_model(
        args.snapshot_dir,
        device=Device("cuda", args.gpu_index),
        runtime=runtime,
    )
    results: dict[str, Any] = {}
    try:
        for batch in args.batches:
            early_decoder = MoonshineCudaBatchRuntime(
                max_batch=batch,
                encoder_frames=frames,
                loaded_model=loaded,
                owns_weights=False,
            )
            mature_decoder = MoonshineCudaBatchRuntime(
                max_batch=batch,
                encoder_frames=frames,
                loaded_model=loaded,
                owns_weights=False,
            )
            early_decoder.prepare_decoder_kernels()
            mature_decoder.prepare_decoder_kernels()
            continuous = MoonshineCudaExactContinuousBatchRuntime(
                early_decoder,
                mature_decoder,
                owns_decoders=True,
                max_pending=args.requests,
                max_graphs=args.max_graphs,
            )
            try:
                continuous_samples = benchmark_route(
                    lambda index: run_continuous_workload(
                        continuous, workload, index
                    ),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                continuous_graph = continuous.graph_cache_contract()
                continuous_scheduler = continuous.scheduler_contract()
                continuous_workspace = sum(
                    decoder.allocation_contract()["workspace_nbytes"]
                    for decoder in (early_decoder, mature_decoder)
                )
            finally:
                continuous.close()
            if not mature_decoder.teardown_returned_to_baseline:
                raise RuntimeError(
                    f"continuous B={batch} mature decoder teardown leaked"
                )
            if not early_decoder.teardown_returned_to_baseline:
                raise RuntimeError(
                    f"continuous B={batch} early decoder teardown leaked"
                )

            lockstep_decoder = MoonshineCudaBatchRuntime(
                max_batch=batch,
                encoder_frames=frames,
                loaded_model=loaded,
                owns_weights=False,
            )
            lockstep_decoder.prepare_decoder_kernels()
            try:
                lockstep_samples = benchmark_route(
                    lambda index: run_lockstep_workload(
                        lockstep_decoder, workload, index
                    ),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                lockstep_graph = lockstep_decoder.batch_token_graph_contract()
                lockstep_workspace = lockstep_decoder.allocation_contract()[
                    "workspace_nbytes"
                ]
            finally:
                lockstep_decoder.close()
            if not lockstep_decoder.teardown_returned_to_baseline:
                raise RuntimeError(f"lockstep B={batch} decoder teardown leaked")

            continuous_summary = timing_summary(continuous_samples, args.requests)
            lockstep_summary = timing_summary(lockstep_samples, args.requests)
            results[f"b{batch}"] = {
                "batch": batch,
                "continuous": continuous_summary,
                "lockstep": lockstep_summary,
                "continuous_vs_lockstep_requests_per_s": (
                    continuous_summary["requests_per_s_median"]
                    / lockstep_summary["requests_per_s_median"]
                ),
                "continuous_vs_lockstep_p95_latency": (
                    lockstep_summary["request_latency_p95_ms"]
                    / continuous_summary["request_latency_p95_ms"]
                ),
                "continuous_graph_cache": continuous_graph,
                "continuous_scheduler": continuous_scheduler,
                "lockstep_graphs": lockstep_graph,
                "workspace_bytes": {
                    "continuous": continuous_workspace,
                    "lockstep": lockstep_workspace,
                },
                "all_outputs_exact": True,
                "all_teardown_returned_to_baseline": True,
            }
            print(
                f"B={batch}: continuous {continuous_summary['requests_per_s_median']:.2f} "
                f"vs lockstep {lockstep_summary['requests_per_s_median']:.2f} req/s",
                flush=True,
            )
    finally:
        loaded.weights.free(runtime=runtime)

    report = {
        "schema": 1,
        "artifact": "moonshine_cuda_continuous_batching",
        "date": datetime.now(UTC).date().isoformat(),
        "status": "pass" if not state["dirty"] and args.exclusive_gpu else "diagnostic",
        "performance_claim": bool(not state["dirty"] and args.exclusive_gpu),
        "model": {
            "id": "shisa-ai/shisa-realtime-asr-0.92b",
            "revision": "cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d",
        },
        "hipengine_git": state,
        "implementation_sha256": implementation_sha256(
            root,
            report_sources=("scripts/benchmark_moonshine_cuda_continuous.py",),
        ),
        "environment": {
            "gpu_index": args.gpu_index,
            "gpu": torch.cuda.get_device_name(args.gpu_index),
            "compute_capability": list(torch.cuda.get_device_capability(args.gpu_index)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "platform": platform.platform(),
            "exclusive_gpu_requested": args.exclusive_gpu,
            "compute_processes_before": before_processes,
        },
        "scope": {
            "batches": list(args.batches),
            "requests_per_sample": args.requests,
            "fixture_cycle": list(_DEFAULT_FIXTURES),
            "encoder_frames": frames,
            "warmup_samples": args.warmup,
            "timed_samples": args.iterations,
            "continuous_topology": "exact two-region: t32 positions 0-6 / t256 positions 7-193",
            "lockstep_topology": "exact t32 positions 0-6 / t256 positions 7-193",
            "timing": "synchronized end-to-end scheduler wall including cross-cache admission",
            "all_requests_logically_submitted_at_sample_start": True,
        },
        "dependency_adjusted": dependency_adjusted_bytes(
            "custom_cuda_runtime_subset"
        ),
        "correctness": {
            "all_outputs_exact_to_six_fixture_references": True,
            "all_routes_repeat_deterministic": True,
            "continuous_numerical_contract": (
                "bit-exact topology per request: t32 positions 0-6, one D2D state "
                "transfer, then t256 positions 7-193"
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value["continuous_vs_lockstep_requests_per_s"] for key, value in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
