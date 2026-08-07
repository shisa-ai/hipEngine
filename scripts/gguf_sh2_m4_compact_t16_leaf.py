#!/usr/bin/env python3
"""Screen SH2-M4 compact selected T16 metadata on actual Qwen weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time
from typing import Callable

import numpy as np

from hipengine.benchmark.provenance import collect_model_identity
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q4_k_qmicro_t16_selected_dual_silu_gemv_bf16_bf16_out,
    gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out,
    gguf_q5_k_qmicro_t16_selected_qwen_tile8_gemv_bf16_bf16_out,
    gguf_q5_k_t16_selected_qwen_tile8_gemv_bf16_bf16_out,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf_q4_k import (
    repack_gguf_q4_k_tile16,
    repack_gguf_q4_k_tile16_qmicro,
)
from hipengine.quant.gguf_t16 import (
    repack_gguf_q5_k_qmicro_tile16,
    repack_gguf_q5_k_tile16,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_OUTPUT = Path("/tmp/hipengine-sh2-m4-compact-t16-leaf.json")
EXPERTS = 256
TOP_K = 8
HIDDEN_SIZE = 2048
EXPERT_INTERMEDIATE = 512
Q4_CALLS_PER_TOKEN = 40
Q5_CALLS_PER_TOKEN = 37
REFERENCE_MS_PER_TOKEN = 18.750406


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=16)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--burst", type=int, default=80)
    parser.add_argument("--seed", type=int, default=202608064)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _git_revision() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
    ).strip()


def _tracked_status() -> list[str]:
    output = subprocess.check_output(
        ("git", "status", "--short", "--untracked-files=no"),
        cwd=ROOT,
        text=True,
    )
    return output.splitlines()


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return np.ascontiguousarray((rounded >> 16).astype(np.uint16))


def _upload(runtime, values: np.ndarray):
    array = np.asarray(values)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    buffer = malloc(array.nbytes, runtime=runtime)
    copy_host_to_device(
        buffer, host_array_ptr(array), array.nbytes, runtime=runtime
    )
    return buffer


def _download(runtime, buffer, shape: tuple[int, ...]) -> np.ndarray:
    result = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(
        host_array_ptr(result), buffer, result.nbytes, runtime=runtime
    )
    return result


def _event_us(runtime, launch: Callable[[], None], *, burst: int) -> float:
    start = runtime.event_create()
    end = runtime.event_create()
    try:
        runtime.event_record(start)
        for _ in range(burst):
            launch()
        runtime.event_record(end)
        runtime.event_synchronize(end)
        return 1_000.0 * float(runtime.event_elapsed_time_ms(start, end)) / burst
    finally:
        runtime.event_destroy(end)
        runtime.event_destroy(start)


def _counterbalanced_timings(
    runtime,
    control: Callable[[], None],
    candidate: Callable[[], None],
    *,
    warmups: int,
    samples: int,
    burst: int,
) -> dict:
    for _ in range(warmups):
        control()
        candidate()
    runtime.device_synchronize()
    values = {"control": [], "candidate": []}
    orders: list[list[str]] = []
    launchers = {"control": control, "candidate": candidate}
    for sample in range(samples):
        order = (
            ("control", "candidate")
            if sample % 2 == 0
            else ("candidate", "control")
        )
        orders.append(list(order))
        for name in order:
            values[name].append(
                _event_us(runtime, launchers[name], burst=burst)
            )
    medians = {
        name: statistics.median(samples_us)
        for name, samples_us in values.items()
    }
    delta_us = medians["candidate"] - medians["control"]
    return {
        "samples_us": values,
        "median_us": medians,
        "candidate_over_control": medians["candidate"] / medians["control"],
        "delta_us_per_call": delta_us,
        "orders": orders,
    }


def _all_expert_exact(
    runtime,
    *,
    selected_dev,
    control: Callable[[int], None],
    candidate: Callable[[int], None],
    control_out,
    candidate_out,
    output_shape: tuple[int, int],
) -> dict:
    digester = hashlib.sha256()
    mismatch = 0
    for group in range(EXPERTS // TOP_K):
        selected_ptr = selected_dev.ptr + group * TOP_K * np.dtype(np.int64).itemsize
        control(selected_ptr)
        control_bits = _download(runtime, control_out, output_shape)
        candidate(selected_ptr)
        candidate_bits = _download(runtime, candidate_out, output_shape)
        mismatch += int(np.count_nonzero(candidate_bits != control_bits))
        digester.update(control_bits.astype("<u2", copy=False).tobytes())
        digester.update(candidate_bits.astype("<u2", copy=False).tobytes())
    return {
        "groups": EXPERTS // TOP_K,
        "experts_covered": EXPERTS,
        "bf16_mismatch": mismatch,
        "paired_output_sha256": digester.hexdigest(),
    }


def _repack(name: str, fn: Callable, raw: np.ndarray) -> tuple[np.ndarray, dict]:
    started = time.perf_counter()
    result = fn(raw).tiles
    elapsed = time.perf_counter() - started
    return result, {
        "name": name,
        "seconds": elapsed,
        "bytes": int(result.nbytes),
        "shape": list(result.shape),
        "sha256": hashlib.sha256(result).hexdigest(),
    }


def _screen_q4(
    runtime,
    library,
    *,
    x: np.ndarray,
    selected: np.ndarray,
    gate_t16: np.ndarray,
    up_t16: np.ndarray,
    gate_compact: np.ndarray,
    up_compact: np.ndarray,
    warmups: int,
    samples: int,
    burst: int,
) -> dict:
    buffers = []
    try:
        x_dev = _upload(runtime, x)
        selected_dev = _upload(runtime, selected)
        gate_t16_dev = _upload(runtime, gate_t16)
        up_t16_dev = _upload(runtime, up_t16)
        gate_compact_dev = _upload(runtime, gate_compact)
        up_compact_dev = _upload(runtime, up_compact)
        control_out = malloc(TOP_K * EXPERT_INTERMEDIATE * 2, runtime=runtime)
        candidate_out = malloc(TOP_K * EXPERT_INTERMEDIATE * 2, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                selected_dev,
                gate_t16_dev,
                up_t16_dev,
                gate_compact_dev,
                up_compact_dev,
                control_out,
                candidate_out,
            )
        )

        def control(selected_ptr: int) -> None:
            gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out(
                x_dev.ptr,
                selected_ptr,
                gate_t16_dev.ptr,
                up_t16_dev.ptr,
                control_out.ptr,
                1,
                TOP_K,
                EXPERTS,
                HIDDEN_SIZE,
                EXPERT_INTERMEDIATE,
                library=library,
                runtime=runtime,
            )

        def candidate(selected_ptr: int) -> None:
            gguf_q4_k_qmicro_t16_selected_dual_silu_gemv_bf16_bf16_out(
                x_dev.ptr,
                selected_ptr,
                gate_compact_dev.ptr,
                up_compact_dev.ptr,
                candidate_out.ptr,
                1,
                TOP_K,
                EXPERTS,
                HIDDEN_SIZE,
                EXPERT_INTERMEDIATE,
                library=library,
                runtime=runtime,
            )

        exact = _all_expert_exact(
            runtime,
            selected_dev=selected_dev,
            control=control,
            candidate=candidate,
            control_out=control_out,
            candidate_out=candidate_out,
            output_shape=(TOP_K, EXPERT_INTERMEDIATE),
        )
        first_selected_ptr = selected_dev.ptr
        timing = _counterbalanced_timings(
            runtime,
            lambda: control(first_selected_ptr),
            lambda: candidate(first_selected_ptr),
            warmups=warmups,
            samples=samples,
            burst=burst,
        )
        return {**exact, **timing, "calls_per_token": Q4_CALLS_PER_TOKEN}
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def _screen_q5(
    runtime,
    library,
    *,
    x: np.ndarray,
    selected: np.ndarray,
    down_t16: np.ndarray,
    down_compact: np.ndarray,
    warmups: int,
    samples: int,
    burst: int,
) -> dict:
    buffers = []
    try:
        x_dev = _upload(runtime, x)
        selected_dev = _upload(runtime, selected)
        down_t16_dev = _upload(runtime, down_t16)
        down_compact_dev = _upload(runtime, down_compact)
        control_out = malloc(TOP_K * HIDDEN_SIZE * 2, runtime=runtime)
        candidate_out = malloc(TOP_K * HIDDEN_SIZE * 2, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                selected_dev,
                down_t16_dev,
                down_compact_dev,
                control_out,
                candidate_out,
            )
        )

        def control(selected_ptr: int) -> None:
            gguf_q5_k_t16_selected_qwen_tile8_gemv_bf16_bf16_out(
                x_dev.ptr,
                selected_ptr,
                down_t16_dev.ptr,
                control_out.ptr,
                TOP_K,
                TOP_K,
                EXPERTS,
                EXPERT_INTERMEDIATE,
                HIDDEN_SIZE,
                library=library,
                runtime=runtime,
            )

        def candidate(selected_ptr: int) -> None:
            gguf_q5_k_qmicro_t16_selected_qwen_tile8_gemv_bf16_bf16_out(
                x_dev.ptr,
                selected_ptr,
                down_compact_dev.ptr,
                candidate_out.ptr,
                TOP_K,
                TOP_K,
                EXPERTS,
                EXPERT_INTERMEDIATE,
                HIDDEN_SIZE,
                library=library,
                runtime=runtime,
            )

        exact = _all_expert_exact(
            runtime,
            selected_dev=selected_dev,
            control=control,
            candidate=candidate,
            control_out=control_out,
            candidate_out=candidate_out,
            output_shape=(TOP_K, HIDDEN_SIZE),
        )
        first_selected_ptr = selected_dev.ptr
        timing = _counterbalanced_timings(
            runtime,
            lambda: control(first_selected_ptr),
            lambda: candidate(first_selected_ptr),
            warmups=warmups,
            samples=samples,
            burst=burst,
        )
        return {**exact, **timing, "calls_per_token": Q5_CALLS_PER_TOKEN}
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def main() -> int:
    args = _parse_args()
    tracked_status = _tracked_status()
    if tracked_status and not args.allow_dirty:
        raise SystemExit(
            "tracked worktree must be clean; pass --allow-dirty for a screen"
        )
    if args.warmups < 0 or min(args.samples, args.burst) <= 0:
        raise ValueError("warmups must be non-negative; samples/burst positive")
    if args.trace_only:
        args.warmups = 0
        args.samples = 1
        args.burst = 1
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(
            args.compiler_version_file
        )

    reader = GGUFReader(args.model)
    tensor_names = {
        "gate": f"blk.{args.layer}.ffn_gate_exps.weight",
        "up": f"blk.{args.layer}.ffn_up_exps.weight",
        "down": f"blk.{args.layer}.ffn_down_exps.weight",
    }
    tensor_info = {
        role: reader.tensor_info(name) for role, name in tensor_names.items()
    }
    if [tensor_info[role].ggml_type_name for role in ("gate", "up", "down")] != [
        "Q4_K",
        "Q4_K",
        "Q5_K",
    ]:
        raise ValueError("selected layer must have Q4_K gate/up and Q5_K down")

    repacks: dict[str, dict] = {}
    gate_raw = np.asarray(reader.tensor_data(tensor_names["gate"]))
    gate_t16, repacks["gate_t16"] = _repack(
        "gate_t16", repack_gguf_q4_k_tile16, gate_raw
    )
    gate_compact, repacks["gate_compact"] = _repack(
        "gate_compact", repack_gguf_q4_k_tile16_qmicro, gate_raw
    )
    up_raw = np.asarray(reader.tensor_data(tensor_names["up"]))
    up_t16, repacks["up_t16"] = _repack(
        "up_t16", repack_gguf_q4_k_tile16, up_raw
    )
    up_compact, repacks["up_compact"] = _repack(
        "up_compact", repack_gguf_q4_k_tile16_qmicro, up_raw
    )
    down_raw = np.asarray(reader.tensor_data(tensor_names["down"]))
    down_t16, repacks["down_t16"] = _repack(
        "down_t16", repack_gguf_q5_k_tile16, down_raw
    )
    down_compact, repacks["down_compact"] = _repack(
        "down_compact", repack_gguf_q5_k_qmicro_tile16, down_raw
    )

    rng = np.random.default_rng(args.seed)
    selected = np.ascontiguousarray(rng.permutation(EXPERTS), dtype=np.int64)
    gate_x = _bf16_bits(
        rng.normal(0.0, 0.55, size=(1, HIDDEN_SIZE)).astype(np.float32)
    )
    down_x = _bf16_bits(
        rng.normal(
            0.0, 0.55, size=(TOP_K, EXPERT_INTERMEDIATE)
        ).astype(np.float32)
    )
    runtime = get_hip_runtime()
    library = build_gguf_t16_selected_gemv(
        load=True, require_cached=args.require_cached_build
    )
    results = {
        "q4_gate_up": _screen_q4(
            runtime,
            library,
            x=gate_x,
            selected=selected,
            gate_t16=gate_t16,
            up_t16=up_t16,
            gate_compact=gate_compact,
            up_compact=up_compact,
            warmups=args.warmups,
            samples=args.samples,
            burst=args.burst,
        ),
        "q5_down": _screen_q5(
            runtime,
            library,
            x=down_x,
            selected=selected,
            down_t16=down_t16,
            down_compact=down_compact,
            warmups=args.warmups,
            samples=args.samples,
            burst=args.burst,
        ),
    }
    exact = all(item["bf16_mismatch"] == 0 for item in results.values())
    projected_delta_ms = sum(
        item["delta_us_per_call"] * item["calls_per_token"] / 1_000.0
        for item in results.values()
    )
    regression_percent = 100.0 * projected_delta_ms / REFERENCE_MS_PER_TOKEN
    memory_delta_bytes = (
        repacks["gate_t16"]["bytes"]
        + repacks["up_t16"]["bytes"]
        - repacks["gate_compact"]["bytes"]
        - repacks["up_compact"]["bytes"]
    ) * Q4_CALLS_PER_TOKEN + (
        repacks["down_t16"]["bytes"] - repacks["down_compact"]["bytes"]
    ) * Q5_CALLS_PER_TOKEN
    gate_pass = exact and regression_percent <= 1.0
    artifact = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "hipengine_gfx1151_gguf_sh2_m4_compact_t16_leaf",
        "status": "candidate" if gate_pass else "rejected",
        "performance_claim": False,
        "model": collect_model_identity(args.model),
        "hardware": {
            "gpu": "AMD Radeon 8060S Graphics",
            "architecture": "gfx1151",
        },
        "workload": {
            "layer": args.layer,
            "shape": "c1/top8, hidden=2048, expert_intermediate=512, experts=256",
            "selected_sha256": hashlib.sha256(
                selected.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
            "activation_sha256": hashlib.sha256(
                gate_x.astype("<u2", copy=False).tobytes()
                + down_x.astype("<u2", copy=False).tobytes()
            ).hexdigest(),
            "deployed_calls_per_token": {"q4_gate_up": 40, "q5_down": 37},
        },
        "tensor_info": {
            role: {
                "name": info.name,
                "ggml_type": info.ggml_type_name,
                "shape": list(info.shape),
                "nbytes": info.nbytes,
                "data_offset": info.data_offset,
            }
            for role, info in tensor_info.items()
        },
        "repacks": repacks,
        "protocol": {
            "control": "registered current Q4/Q5 T16 selected decode",
            "candidate": "registered byte-neutral qmicro Q4/Q5 T16 selected decode",
            "correctness": "BF16 bit equality over 32 top-8 groups covering all 256 experts",
            "timing": "counterbalanced HIP events on one production-shaped c1/top8 group",
            "warmups": args.warmups,
            "samples": args.samples,
            "burst": args.burst,
            "reference_ms_per_token": REFERENCE_MS_PER_TOKEN,
            "promotion_gate": "all-expert exact and projected decode regression <=1%",
            "command": shlex.join(sys.argv),
        },
        "results": results,
        "projection": {
            "resident_saving_bytes": memory_delta_bytes,
            "resident_saving_gib": memory_delta_bytes / 2**30,
            "delta_ms_per_token": projected_delta_ms,
            "regression_percent_of_reference_decode": regression_percent,
        },
        "gate": {"exact": exact, "within_one_percent": regression_percent <= 1.0, "passed": gate_pass},
        "repo": {"revision": _git_revision(), "tracked_status": tracked_status},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    if not exact:
        raise SystemExit("compact T16 differs from current T16 on actual weights")
    if not gate_pass and not args.trace_only:
        raise SystemExit("compact T16 misses the <=1% actual-weight leaf gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
