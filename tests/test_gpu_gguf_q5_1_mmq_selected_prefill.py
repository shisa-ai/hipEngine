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


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q5_1_mmq_ds4_serves_in_features_not_divisible_by_128() -> None:
    """The down projection's K=704 is not a multiple of the 128-value DS4 block.

    ``blk.27.ffn_down_exps.weight`` is ``(128, 2816, 704)``: 22 Q5_1 blocks of
    32, but 5.5 DS4 blocks of 128. The pack, the workspace sizing and the
    consumer all floor that division, which would cover 640 of 704 inputs and
    drop the trailing 64 values silently -- which is why the route's own gate
    refuses the shape outright today.

    Zero-filling the trailing lanes is exact: a zero activation contributes 0
    to ``dot4``, to the block sum, and therefore to the ``m`` offset, so a
    partial final block must reproduce the strict float-dequant owner rather
    than merely stay inside its buffer.
    """
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
        # Match the production route: the consumer runs planes=3, so the pack
        # must write all three planes (d4x3 = residual_passes 3).
        gguf_q8_1_mmq_ds4_pack_bf16_d4x3 as gguf_q8_1_mmq_ds4_pack_bf16,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q5_1_mmq_selected_prefill import (
        build_gguf_q5_1_mmq_selected_prefill,
        ds4_workspace_nbytes,
        gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out,
    )

    rng = np.random.default_rng(704)
    experts, in_features, out_features = 4, 704, 256
    assert in_features % 32 == 0 and in_features % 128 != 0
    counts = np.array([5, 3, 0, 6], dtype=np.int64)
    compact_rows = int(counts.sum())
    expert_start = np.zeros(experts + 1, dtype=np.int64)
    expert_start[1:] = np.cumsum(counts)

    raw_weights, weights = _make_q5_1_weight(rng, experts * out_features, in_features)
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
        # Raises today: in_features % 128 != 0 is rejected outright.
        ds4_dev = malloc(
            ds4_workspace_nbytes(compact_rows, in_features, 3), runtime=runtime
        )
        start_dev = malloc(expert_start.nbytes, runtime=runtime)
        out_dev = malloc(row_owner.nbytes, runtime=runtime)
        allocations += [w_dev, rows_dev, ds4_dev, start_dev, out_dev]
        copy_host_to_device(w_dev, host_array_ptr(host_w), runtime=runtime)
        copy_host_to_device(
            rows_dev, host_array_ptr(np.ascontiguousarray(rows_bf16)), runtime=runtime
        )
        copy_host_to_device(start_dev, host_array_ptr(expert_start), runtime=runtime)

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
            out_dev.ptr,
            compact_rows,
            experts,
            in_features,
            out_features,
            3,
            runtime=runtime,
            library=library,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(row_owner), out_dev, runtime=runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    mmq = _bf16_to_float(row_owner)
    oracle = np.empty_like(mmq)
    row_f32 = _bf16_to_float(rows_bf16)
    for expert in range(experts):
        for row in range(expert_start[expert], expert_start[expert + 1]):
            weight = weights[expert * out_features : (expert + 1) * out_features]
            oracle[row] = row_f32[row] @ weight.T

    # Dropping the trailing 64 inputs moves every output cell, so this is the
    # assertion that pins the contract rather than a tolerance on noise.
    row_scale = np.maximum(np.abs(oracle).max(axis=1, keepdims=True), 1e-3)
    abs_err = np.abs(mmq - oracle) / row_scale
    assert float(abs_err.max()) < 2e-2, (
        f"max scaled error {float(abs_err.max()):.4f} -- the final 64 inputs "
        f"of K={in_features} are not reaching the kernel"
    )
    assert float(abs_err.mean()) < 2e-3
    assert int((mmq.argmax(1) == oracle.argmax(1)).sum()) >= compact_rows - 1


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q5_1_mmq_wmma_prefill_agrees_with_dp4a_leaf_at_k704() -> None:
    """P10: the int8 matrix-core twin must reproduce the leaf it targets.

    K=704 is the shape that matters -- 22 Q5_1 blocks, 5.5 DS4 blocks -- so
    this exercises the partial-final-block path through the new kernel too.
    """
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
        gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out as run_dp4a,
        gguf_q5_1_mmq_ds4_wmma_prefill_bf16_bf16_out as run_wmma,
    )

    rng = np.random.default_rng(41)
    experts, in_features, out_features = 8, 704, 512
    counts = np.array([4, 7, 0, 12, 1, 3, 0, 6], dtype=np.int64)
    compact_rows = int(counts.sum())
    expert_start = np.zeros(experts + 1, dtype=np.int64)
    expert_start[1:] = np.cumsum(counts)

    raw_weights, weights = _make_q5_1_weight(rng, experts * out_features, in_features)
    host_w = np.frombuffer(raw_weights, dtype=np.uint8)
    rows_f32 = (rng.standard_normal((compact_rows, in_features)) * 0.4).astype(np.float32)
    rows_bf16 = _bf16_bits(rows_f32).reshape(compact_rows, in_features)

    out_dp4a = np.empty((compact_rows, out_features), dtype=np.uint16)
    out_wmma = np.empty_like(out_dp4a)

    runtime = get_hip_runtime()
    library = build_gguf_q5_1_mmq_selected_prefill(load=True)
    allocations = []
    try:
        w_dev = malloc(host_w.nbytes, runtime=runtime)
        rows_dev = malloc(rows_bf16.nbytes, runtime=runtime)
        ds4_dev = malloc(ds4_workspace_nbytes(compact_rows, in_features, 3), runtime=runtime)
        start_dev = malloc(expert_start.nbytes, runtime=runtime)
        dp4a_dev = malloc(out_dp4a.nbytes, runtime=runtime)
        wmma_dev = malloc(out_wmma.nbytes, runtime=runtime)
        allocations += [w_dev, rows_dev, ds4_dev, start_dev, dp4a_dev, wmma_dev]
        copy_host_to_device(w_dev, host_array_ptr(host_w), runtime=runtime)
        copy_host_to_device(
            rows_dev, host_array_ptr(np.ascontiguousarray(rows_bf16)), runtime=runtime
        )
        copy_host_to_device(start_dev, host_array_ptr(expert_start), runtime=runtime)

        gguf_q8_1_mmq_ds4_pack_bf16(
            rows_dev.ptr, ds4_dev.ptr, compact_rows, in_features, runtime=runtime
        )
        run_dp4a(
            ds4_dev.ptr, start_dev.ptr, w_dev.ptr, dp4a_dev.ptr,
            compact_rows, experts, in_features, out_features, 3,
            runtime=runtime, library=library,
        )
        run_wmma(
            ds4_dev.ptr, start_dev.ptr, w_dev.ptr, wmma_dev.ptr,
            compact_rows, experts, in_features, out_features, 3,
            runtime=runtime, library=library,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_dp4a), dp4a_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(out_wmma), wmma_dev, runtime=runtime)
        runtime.device_synchronize()
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    dp4a = _bf16_to_float(out_dp4a)
    wmma = _bf16_to_float(out_wmma)

    # Guard against a silently uninitialised output: an unwritten plane or a
    # dropped tile shows up as NaN first, and as agreement failure second.
    assert np.isfinite(wmma).all(), "wmma leaf produced non-finite output"
    assert np.isfinite(dp4a).all(), "dp4a leaf produced non-finite output"

    oracle = np.empty_like(dp4a)
    rows_bf16_f = _bf16_to_float(rows_bf16)
    for expert in range(experts):
        for row in range(expert_start[expert], expert_start[expert + 1]):
            weight = weights[expert * out_features : (expert + 1) * out_features]
            oracle[row] = rows_bf16_f[row] @ weight.T

    row_scale = np.maximum(np.abs(oracle).max(axis=1, keepdims=True), 1e-3)
    abs_dp4a = np.abs(dp4a - oracle) / row_scale
    abs_wmma = np.abs(wmma - oracle) / row_scale
    abs_pair = np.abs(wmma - dp4a) / row_scale

    assert float(abs_wmma.max()) < 2e-2, f"wmma vs oracle max {float(abs_wmma.max()):.4f}"
    assert float(abs_wmma.mean()) < 2e-3, f"wmma vs oracle mean {float(abs_wmma.mean()):.4f}"
    assert float(abs_dp4a.max()) < 2e-2, f"dp4a vs oracle max {float(abs_dp4a.max()):.4f}"
    # The two leaves compute the same arithmetic contract, so they should be
    # far closer to each other than either is to the float oracle.
    assert float(abs_pair.max()) < 5e-2, f"wmma vs dp4a max {float(abs_pair.max()):.4f}"
    assert int((wmma.argmax(1) == oracle.argmax(1)).sum()) >= compact_rows - 1
    # K=704's partial final DS4 block must contribute: without it the last 64
    # inputs vanish and every output cell moves.
    assert float(abs_wmma.mean()) < float(abs_dp4a.mean()) * 5 + 1e-4, (
        "wmma leaf's error is disproportionate to the DP4A leaf's"
    )
