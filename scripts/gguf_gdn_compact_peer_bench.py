#!/usr/bin/env python3
"""Counterbalanced leaf screen for compact peer-wave32 GDN prefill.

The control materializes normalized Q/K per V head.  The candidate materializes
those tensors once per K head and changes only the read stride in the admitted
wave32/XOR recurrence.  Timing covers prepare + recurrent + RMSNorm/gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Callable

import numpy as np


def _f32_to_bf16_u16(array: np.ndarray) -> np.ndarray:
    f32 = np.asarray(array, dtype=np.float32, order="C")
    u32 = f32.view(np.uint32).copy()
    rounded = (u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16
    return rounded.astype(np.uint16).reshape(f32.shape)


def _upload(runtime, array: np.ndarray, buffers: list):
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

    host = np.ascontiguousarray(array)
    buffer = malloc(host.nbytes, runtime=runtime)
    copy_host_to_device(buffer, host_array_ptr(host), host.nbytes, runtime=runtime)
    buffers.append(buffer)
    return buffer


def _allocate(runtime, nbytes: int, buffers: list):
    from hipengine.core.memory import malloc

    buffer = malloc(nbytes, runtime=runtime)
    buffers.append(buffer)
    return buffer


def _download(runtime, buffer, shape: tuple[int, ...], dtype) -> np.ndarray:
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), buffer, host.nbytes, runtime=runtime)
    return host


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _event_ms(runtime, launch: Callable[[], None]) -> float:
    start = runtime.event_create()
    stop = runtime.event_create()
    try:
        runtime.event_record(start)
        launch()
        runtime.event_record(stop)
        runtime.event_synchronize(stop)
        return float(runtime.event_elapsed_time_ms(start, stop))
    finally:
        runtime.event_destroy(stop)
        runtime.event_destroy(start)


def _time_counterbalanced(
    runtime,
    control: Callable[[], None],
    candidate: Callable[[], None],
    *,
    warmups: int,
    samples: int,
) -> dict[str, object]:
    for index in range(warmups):
        first, second = (control, candidate) if index % 2 == 0 else (candidate, control)
        first()
        second()
    runtime.device_synchronize()
    values: dict[str, list[float]] = {"control": [], "candidate": []}
    named = {"control": control, "candidate": candidate}
    for index in range(samples):
        order = ("control", "candidate") if index % 2 == 0 else ("candidate", "control")
        for name in order:
            values[name].append(_event_ms(runtime, named[name]))
    control_median = float(statistics.median(values["control"]))
    candidate_median = float(statistics.median(values["candidate"]))
    return {
        "samples_ms": values,
        "median_ms": {"control": control_median, "candidate": candidate_median},
        "speedup": control_median / candidate_median,
        "candidate_delta_percent": (candidate_median / control_median - 1.0) * 100.0,
        "candidate_wins": sum(
            candidate_value < control_value
            for control_value, candidate_value in zip(
                values["control"], values["candidate"], strict=True
            )
        ),
        "pair_count": samples,
    }


def _screen_row(runtime, library, rows: int, warmups: int, samples: int) -> dict[str, object]:
    from hipengine.core.memory import free
    from hipengine.core.runtime import MemcpyKind
    from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
        qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_f32,
        qwen35_gdn_prefill_recurrent_normalized_wave32_xor_f32,
        qwen35_gdn_prefill_rmsnorm_gate_bf16,
        qwen35_linear_attn_prefill_prepare_compact_peer_normalized_f32_bf16,
        qwen35_linear_attn_prefill_prepare_peer_normalized_f32_bf16,
    )

    num_k_heads = 4
    num_v_heads = 32
    head_k_dim = 128
    head_v_dim = 128
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_width = 2 * key_dim + value_dim
    rng = np.random.default_rng(36_027 + rows)
    conv = rng.normal(0.0, 0.18, (rows, conv_width)).astype(np.float32)
    alpha = _f32_to_bf16_u16(rng.normal(0.0, 0.25, (rows, num_v_heads)))
    beta_lowp = _f32_to_bf16_u16(rng.normal(0.0, 0.25, (rows, num_v_heads)))
    gate = _f32_to_bf16_u16(rng.normal(0.0, 0.2, (rows, value_dim)))
    dt_bias = rng.normal(0.0, 0.1, (num_v_heads,)).astype(np.float32)
    a_log = rng.normal(0.0, 0.1, (num_v_heads,)).astype(np.float32)
    norm = rng.normal(1.0, 0.05, (head_v_dim,)).astype(np.float32)
    initial_state = rng.normal(
        0.0, 0.02, (num_v_heads, head_k_dim, head_v_dim)
    ).astype(np.float32)

    buffers: list = []
    try:
        conv_dev = _upload(runtime, conv, buffers)
        alpha_dev = _upload(runtime, alpha, buffers)
        beta_lowp_dev = _upload(runtime, beta_lowp, buffers)
        gate_dev = _upload(runtime, gate, buffers)
        dt_bias_dev = _upload(runtime, dt_bias, buffers)
        a_log_dev = _upload(runtime, a_log, buffers)
        norm_dev = _upload(runtime, norm, buffers)
        initial_state_dev = _upload(runtime, initial_state, buffers)
        control_state = _allocate(runtime, initial_state.nbytes, buffers)
        candidate_state = _allocate(runtime, initial_state.nbytes, buffers)

        f32 = np.dtype(np.float32).itemsize
        control_q = _allocate(runtime, rows * num_v_heads * head_k_dim * f32, buffers)
        control_k = _allocate(runtime, rows * num_v_heads * head_k_dim * f32, buffers)
        candidate_q = _allocate(runtime, rows * num_k_heads * head_k_dim * f32, buffers)
        candidate_k = _allocate(runtime, rows * num_k_heads * head_k_dim * f32, buffers)
        control_v = _allocate(runtime, rows * value_dim * f32, buffers)
        candidate_v = _allocate(runtime, rows * value_dim * f32, buffers)
        scalar_bytes = rows * num_v_heads * f32
        control_beta = _allocate(runtime, scalar_bytes, buffers)
        control_decay = _allocate(runtime, scalar_bytes, buffers)
        candidate_beta = _allocate(runtime, scalar_bytes, buffers)
        candidate_decay = _allocate(runtime, scalar_bytes, buffers)
        recurrent_bytes = rows * value_dim * f32
        control_recurrent = _allocate(runtime, recurrent_bytes, buffers)
        candidate_recurrent = _allocate(runtime, recurrent_bytes, buffers)
        out_bytes = rows * value_dim * np.dtype(np.uint16).itemsize
        control_out = _allocate(runtime, out_bytes, buffers)
        candidate_out = _allocate(runtime, out_bytes, buffers)

        def reset_state(destination) -> None:
            runtime.memcpy(
                destination.ptr,
                initial_state_dev.ptr,
                initial_state.nbytes,
                MemcpyKind.DEVICE_TO_DEVICE,
            )

        def control() -> None:
            reset_state(control_state)
            qwen35_linear_attn_prefill_prepare_peer_normalized_f32_bf16(
                conv_dev.ptr, alpha_dev.ptr, beta_lowp_dev.ptr,
                dt_bias_dev.ptr, a_log_dev.ptr, control_q.ptr, control_k.ptr,
                control_v.ptr, control_beta.ptr, control_decay.ptr, rows,
                num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                library=library, runtime=runtime,
            )
            qwen35_gdn_prefill_recurrent_normalized_wave32_xor_f32(
                control_q.ptr, control_k.ptr, control_v.ptr, control_beta.ptr,
                control_decay.ptr, control_state.ptr, control_recurrent.ptr,
                rows, num_v_heads, head_k_dim, head_v_dim,
                library=library, runtime=runtime,
            )
            qwen35_gdn_prefill_rmsnorm_gate_bf16(
                control_recurrent.ptr, gate_dev.ptr, norm_dev.ptr,
                control_out.ptr, 1.0e-6, rows, num_v_heads, head_v_dim,
                library=library, runtime=runtime,
            )

        def candidate() -> None:
            reset_state(candidate_state)
            qwen35_linear_attn_prefill_prepare_compact_peer_normalized_f32_bf16(
                conv_dev.ptr, alpha_dev.ptr, beta_lowp_dev.ptr,
                dt_bias_dev.ptr, a_log_dev.ptr, candidate_q.ptr, candidate_k.ptr,
                candidate_v.ptr, candidate_beta.ptr, candidate_decay.ptr, rows,
                num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                library=library, runtime=runtime,
            )
            qwen35_gdn_prefill_recurrent_compact_normalized_wave32_xor_f32(
                candidate_q.ptr, candidate_k.ptr, candidate_v.ptr,
                candidate_beta.ptr, candidate_decay.ptr, candidate_state.ptr,
                candidate_recurrent.ptr, rows, num_k_heads, num_v_heads,
                head_k_dim, head_v_dim, library=library, runtime=runtime,
            )
            qwen35_gdn_prefill_rmsnorm_gate_bf16(
                candidate_recurrent.ptr, gate_dev.ptr, norm_dev.ptr,
                candidate_out.ptr, 1.0e-6, rows, num_v_heads, head_v_dim,
                library=library, runtime=runtime,
            )

        control()
        candidate()
        runtime.device_synchronize()
        out_shape = (rows, num_v_heads, head_v_dim)
        state_shape = (num_v_heads, head_k_dim, head_v_dim)
        control_out_host = _download(runtime, control_out, out_shape, np.uint16)
        candidate_out_host = _download(runtime, candidate_out, out_shape, np.uint16)
        control_state_host = _download(runtime, control_state, state_shape, np.float32)
        candidate_state_host = _download(runtime, candidate_state, state_shape, np.float32)
        output_exact = bool(np.array_equal(control_out_host, candidate_out_host))
        state_exact = bool(
            np.array_equal(control_state_host.view(np.uint32), candidate_state_host.view(np.uint32))
        )
        timing = _time_counterbalanced(
            runtime, control, candidate, warmups=warmups, samples=samples
        )
        timing.update(
            {
                "output_bf16_exact": output_exact,
                "state_f32_bits_exact": state_exact,
                "output_mismatches": int(np.count_nonzero(control_out_host != candidate_out_host)),
                "state_mismatches": int(
                    np.count_nonzero(
                        control_state_host.view(np.uint32)
                        != candidate_state_host.view(np.uint32)
                    )
                ),
                "output_sha256": {
                    "control": _sha256(control_out_host),
                    "candidate": _sha256(candidate_out_host),
                },
                "state_sha256": {
                    "control": _sha256(control_state_host),
                    "candidate": _sha256(candidate_state_host),
                },
                "workspace_bytes": {
                    "control_qk": 2 * rows * num_v_heads * head_k_dim * f32,
                    "candidate_qk": 2 * rows * num_k_heads * head_k_dim * f32,
                    "candidate_qk_saved": 2 * rows * (num_v_heads - num_k_heads) * head_k_dim * f32,
                    "persistent_added": 0,
                },
            }
        )
        return timing
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if any(row <= 0 for row in args.rows):
        parser.error("rows must be positive")
    if args.warmups < 0 or args.samples <= 0:
        parser.error("warmups must be non-negative and samples positive")

    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.linear_attn.gdn import build_qwen35_linear_attn_gdn

    runtime = get_hip_runtime()
    library = build_qwen35_linear_attn_gdn(
        load=True,
        require_cached=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD", "").lower()
        in {"1", "true", "yes", "on"},
    )
    results = {
        str(rows): _screen_row(runtime, library, rows, args.warmups, args.samples)
        for rows in args.rows
    }
    quality_pass = all(
        result["output_bf16_exact"] and result["state_f32_bits_exact"]
        for result in results.values()
    )
    performance_pass = all(
        result["candidate_wins"] >= 13 and result["speedup"] > 1.0
        for result in results.values()
    )
    artifact = {
        "schema_version": 1,
        "status": "leaf_admitted" if quality_pass and performance_pass else "leaf_rejected",
        "performance_claim": False,
        "hardware": {
            "selector": os.environ.get("HIP_VISIBLE_DEVICES"),
            "arch": os.environ.get("HIPENGINE_HIP_ARCH", "gfx1100"),
        },
        "shape": {
            "num_k_heads": 4,
            "num_v_heads": 32,
            "head_k_dim": 128,
            "head_v_dim": 128,
        },
        "protocol": {
            "timing": "counterbalanced HIP events",
            "warmups": args.warmups,
            "samples": args.samples,
            "control": "peer-normalized per-V-head Q/K prepare + admitted wave32/XOR recurrence + RMSNorm/gate",
            "candidate": "compact per-K-head Q/K prepare + stride-adjusted bit-equivalent wave32/XOR recurrence + RMSNorm/gate",
            "gate": "output/state bit exact; candidate wins >=13/15 at each row",
        },
        "results": results,
        "quality_pass": quality_pass,
        "performance_pass": performance_pass,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    raise SystemExit(0 if quality_pass and performance_pass else 1)


if __name__ == "__main__":
    main()
