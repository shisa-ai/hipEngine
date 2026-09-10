from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.hip import HIP_SUCCESS, get_hip_runtime


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _bf16_to_float(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def _make_q5_1_weight(rng: np.random.Generator, out_features: int, in_features: int):
    blocks = out_features * (in_features // 32)
    inter = np.empty(blocks, dtype=[("d", "<f2"), ("m", "<f2"), ("qh", "<u4"), ("qs", "u1", (16,))])
    inter["d"] = (rng.standard_normal(blocks) * 0.03).astype(np.float16)
    inter["m"] = (rng.standard_normal(blocks) * 0.01).astype(np.float16)
    inter["qh"] = rng.integers(0, 1 << 32, size=blocks, dtype=np.uint32)
    inter["qs"] = rng.integers(0, 256, size=(blocks, 16), dtype=np.uint8)
    assert inter.dtype.itemsize == 24, inter.dtype.itemsize
    raw = inter.tobytes()
    # Dequantized reference (GGML Q5_1 plane order): values 0..15 are the
    # low nibbles of qs bytes 0..15, values 16..31 the high nibbles; qh bit v
    # belongs to value v; w = (nibble + 16*bit) * d + m.
    nibbles = np.empty((blocks, 32), dtype=np.float64)
    qs = inter["qs"]
    nibbles[:, 0:16] = qs & 0x0F
    nibbles[:, 16:32] = qs >> 4
    bits = np.empty((blocks, 32), dtype=np.float64)
    for lane in range(32):
        bits[:, lane] = (inter["qh"] >> np.uint32(lane)) & np.uint32(1)
    weights = (nibbles + 16.0 * bits) * inter["d"].astype(np.float64)[:, None] + inter["m"].astype(np.float64)[:, None]
    return raw, weights.reshape(out_features, in_features)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q5_1_mmq_ds4_selected_prefill_bounded_and_deterministic() -> None:
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
        gguf_q8_1_mmq_ds4_pack_bf16_d4x3 as gguf_q8_1_mmq_ds4_pack_bf16,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q5_1_mmq_selected_prefill import (
        build_gguf_q5_1_mmq_selected_prefill,
        ds4_workspace_nbytes,
        gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out,
    )

    rng = np.random.default_rng(31)
    experts, in_features, out_features = 8, 640, 512
    counts = np.array([4, 7, 0, 12, 1, 3, 0, 6], dtype=np.int64)
    compact_rows = int(counts.sum())
    expert_start = np.zeros(experts + 1, dtype=np.int64)
    expert_start[1:] = np.cumsum(counts)

    raw_weights, weights = _make_q5_1_weight(rng, experts * out_features, in_features)
    # The kernel indexes weight rows as (expert * out_features + out).
    host_w = np.frombuffer(raw_weights, dtype=np.uint8)
    rows_f32 = (rng.standard_normal((compact_rows, in_features)) * 0.4).astype(np.float32)
    rows_bf16 = _bf16_bits(rows_f32).reshape(compact_rows, in_features)
    row_owner = np.empty((compact_rows, out_features), dtype=np.uint16)

    runtime = get_hip_runtime()
    library = build_gguf_q5_1_mmq_selected_prefill(load=True)
    allocations = []
    try:
        w_dev = malloc(host_w.nbytes, runtime=runtime)
        rows_dev = malloc(rows_bf16.nbytes, runtime=runtime)
        ds4_dev = malloc(ds4_workspace_nbytes(compact_rows, in_features, 3), runtime=runtime)
        start_dev = malloc(expert_start.nbytes, runtime=runtime)
        out_owner = malloc(row_owner.nbytes, runtime=runtime)
        out_mmq = malloc(row_owner.nbytes, runtime=runtime)
        allocations += [w_dev, rows_dev, ds4_dev, start_dev, out_owner, out_mmq]
        copy_host_to_device(w_dev, host_array_ptr(host_w), runtime=runtime)
        copy_host_to_device(rows_dev, host_array_ptr(np.ascontiguousarray(rows_bf16)), runtime=runtime)
        copy_host_to_device(start_dev, host_array_ptr(expert_start), runtime=runtime)

        # Strict grouped owner (float dequant, exact contract reference).
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out(
            rows_dev.ptr,
            start_dev.ptr,
            w_dev.ptr,
            out_owner.ptr,
            compact_rows,
            experts,
            in_features,
            out_features,
            runtime=runtime,
        )

        def run_mmq() -> None:
            gguf_q8_1_mmq_ds4_pack_bf16(
                rows_dev.ptr,
                ds4_dev.ptr,
                compact_rows,
                in_features,
                runtime=runtime,
            )
            gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out(
                ds4_dev.ptr,
                start_dev.ptr,
                w_dev.ptr,
                out_mmq.ptr,
                compact_rows,
                experts,
                in_features,
                out_features,
                3,
                runtime=runtime,
                library=library,
            )

        run_mmq()
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(row_owner), out_owner, runtime=runtime)
        first = np.empty_like(row_owner)
        copy_device_to_host(host_array_ptr(first), out_mmq, runtime=runtime)
        run_mmq()
        runtime.device_synchronize()
        second = np.empty_like(row_owner)
        copy_device_to_host(host_array_ptr(second), out_mmq, runtime=runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(first, second)

    owner = _bf16_to_float(row_owner)
    mmq = _bf16_to_float(first)
    # Oracle: exact dequant weights times the original activation rows.
    oracle = np.empty_like(owner)
    for expert in range(experts):
        for row in range(expert_start[expert], expert_start[expert + 1]):
            weight = weights[expert * out_features : (expert + 1) * out_features]
            oracle[row] = _bf16_to_float(rows_bf16)[row] @ weight.T
    # Q8_1 quantization noise is absolute, not relative to each output cell:
    # bound it against the per-row output scale instead of per-cell magnitude.
    row_scale = np.maximum(
        np.abs(oracle).max(axis=1, keepdims=True), 1e-3
    )
    abs_owner = np.abs(owner - oracle) / row_scale
    abs_mmq = np.abs(mmq - oracle) / row_scale
    assert float(abs_owner.max()) < 1e-2
    assert float(abs_mmq.max()) < 2e-2
    assert float(abs_mmq.mean()) < 2e-3
    # Against the strict owner (production-variant envelope).
    abs_vs_owner = np.abs(mmq - owner) / row_scale
    assert float(abs_vs_owner.max()) < 5e-2
    assert int((mmq.argmax(1) == owner.argmax(1)).sum()) >= compact_rows - 1
