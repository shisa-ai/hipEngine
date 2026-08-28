from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import (
    GGMLQuantizationType,
    bf16_to_float32,
    dequantize_gguf_data,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_qwen4_exp_q5_1_selected_build_and_registry_contract() -> None:
    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        plan_qwen4_exp_q5_1_build,
        qwen4_exp_q5_1_selected_gemv_bf16_bf16_out,
        qwen4_exp_q5_1_selected_gemv_logical256_t128_bf16_bf16_out,
        qwen4_exp_q5_1_selected_gemv_logical256_t64_bf16_bf16_out,
        qwen4_exp_q5_1_selected_gemv_wave64_bf16_bf16_out,
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out,
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_bf16_bf16_out,
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out,
        qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out,
        register_qwen4_exp_q5_1_kernels,
    )
    from hipengine.kernels.registry import resolve

    artifact = plan_qwen4_exp_q5_1_build()
    assert artifact.output_path.name == "qwen4_exp_q5_1.so"
    register_qwen4_exp_q5_1_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q5_1",
            variant="selected_gemv_bf16_bf16_out",
        )
        is qwen4_exp_q5_1_selected_gemv_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q5_1",
            variant="selected_gemv_wave64_bf16_bf16_out",
        )
        is qwen4_exp_q5_1_selected_gemv_wave64_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q5_1",
            variant="selected_gemv_logical256_t128_bf16_bf16_out",
        )
        is qwen4_exp_q5_1_selected_gemv_logical256_t128_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q5_1",
            variant="selected_gemv_logical256_t64_bf16_bf16_out",
        )
        is qwen4_exp_q5_1_selected_gemv_logical256_t64_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q5_1",
            variant="selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out",
        )
        is qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q5_1",
            variant="selected_grouped_prefill_compact_rowbatch8_out8_bf16_bf16_out",
        )
        is qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q5_1",
            variant="selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
        )
        is qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q5_1",
            variant="selected_grouped_wmma_prefill_compact_bf16_bf16_out",
        )
        is qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_gather_bf16_lanes_matches_host_permutation() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        qwen4_exp_gather_bf16_lanes,
    )

    runtime = get_hip_runtime()
    source = np.arange(24, dtype=np.uint16).reshape(6, 4)
    lanes = np.asarray([5, 2, 0, 4, 1, 3], dtype=np.int64)
    allocations = []
    try:
        d_source = _upload(source, runtime, allocations)
        d_lanes = _upload(lanes, runtime, allocations)
        d_output = _alloc(source.shape, np.uint16, runtime, allocations)
        qwen4_exp_gather_bf16_lanes(
            d_source.ptr,
            d_lanes.ptr,
            d_output.ptr,
            source.shape[0],
            source.shape[1],
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(d_output, source.shape, np.uint16, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
    np.testing.assert_array_equal(actual, source[lanes])


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_q5_1_selected_matches_cpu_dequant_oracle() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        build_qwen4_exp_q5_1,
        qwen4_exp_q5_1_selected_gemv_bf16_bf16_out,
        qwen4_exp_q5_1_selected_gemv_logical256_t128_bf16_bf16_out,
        qwen4_exp_q5_1_selected_gemv_logical256_t64_bf16_bf16_out,
        qwen4_exp_q5_1_selected_gemv_wave64_bf16_bf16_out,
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out,
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_bf16_bf16_out,
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out,
        qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out,
    )

    runtime = get_hip_runtime()
    library = build_qwen4_exp_q5_1(load=True)
    rng = np.random.default_rng(51)
    rows, experts, in_features, out_features = 3, 68, 64, 48
    blocks_per_row = in_features // 32
    raw = np.empty((experts, out_features, blocks_per_row * 24), dtype=np.uint8)
    for expert in range(experts):
        for output in range(out_features):
            for block in range(blocks_per_row):
                offset = block * 24
                raw[expert, output, offset : offset + 2] = np.asarray(
                    [0.02 * (1 + expert + block)], dtype=np.float16
                ).view(np.uint8)
                raw[expert, output, offset + 2 : offset + 4] = np.asarray(
                    [-0.1 + 0.01 * output], dtype=np.float16
                ).view(np.uint8)
                raw[expert, output, offset + 4 : offset + 8] = rng.integers(
                    0, 256, size=4, dtype=np.uint8
                )
                raw[expert, output, offset + 8 : offset + 24] = rng.integers(
                    0, 256, size=16, dtype=np.uint8
                )
    # Router selection is an int64 contract shared by all selected GGUF kernels.
    selected = np.array([67, 0, 2], dtype=np.int64)
    x_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    x = bf16_to_float32(x_bits)
    expected = np.empty((rows, out_features), dtype=np.float32)
    for row, expert in enumerate(selected):
        weight = dequantize_gguf_data(
            raw[expert],
            GGMLQuantizationType.Q5_1,
        )
        expected[row] = x[row] @ weight.T
    expected_bits = float_array_to_bf16_bits(expected)

    order = np.argsort(selected, kind="stable")
    grouped_x_bits = x_bits[order]
    grouped_expected_bits = expected_bits[order]
    counts = np.bincount(selected, minlength=experts)
    expert_start = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    padded = ((counts + 15) // 16) * 16
    expert_start_wmma = np.concatenate(([0], np.cumsum(padded))).astype(np.int64)
    tile_expert = np.concatenate(
        [np.full(value // 16, expert, dtype=np.int64) for expert, value in enumerate(padded)]
    )

    allocations = []
    try:
        d_x = _upload(x_bits, runtime, allocations)
        d_selected = _upload(selected, runtime, allocations)
        d_weight = _upload(raw, runtime, allocations)
        d_output = _alloc(expected_bits.shape, np.uint16, runtime, allocations)
        d_exact128_output = _alloc(expected_bits.shape, np.uint16, runtime, allocations)
        d_exact64_output = _alloc(expected_bits.shape, np.uint16, runtime, allocations)
        d_wave64_output = _alloc(expected_bits.shape, np.uint16, runtime, allocations)
        d_grouped_x = _upload(grouped_x_bits, runtime, allocations)
        d_expert_start = _upload(expert_start, runtime, allocations)
        d_grouped_output = _alloc(
            grouped_expected_bits.shape, np.uint16, runtime, allocations
        )
        d_grouped_out8 = _alloc(
            grouped_expected_bits.shape, np.uint16, runtime, allocations
        )
        d_grouped_expertgrid64 = _alloc(
            grouped_expected_bits.shape, np.uint16, runtime, allocations
        )
        d_expert_start_wmma = _upload(expert_start_wmma, runtime, allocations)
        d_tile_expert = _upload(tile_expert, runtime, allocations)
        d_wmma_output = _alloc(
            grouped_expected_bits.shape, np.uint16, runtime, allocations
        )
        qwen4_exp_q5_1_selected_gemv_bf16_bf16_out(
            d_x.ptr,
            d_selected.ptr,
            d_weight.ptr,
            d_output.ptr,
            rows,
            rows,
            experts,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        qwen4_exp_q5_1_selected_gemv_logical256_t128_bf16_bf16_out(
            d_x.ptr,
            d_selected.ptr,
            d_weight.ptr,
            d_exact128_output.ptr,
            rows,
            rows,
            experts,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        qwen4_exp_q5_1_selected_gemv_logical256_t64_bf16_bf16_out(
            d_x.ptr,
            d_selected.ptr,
            d_weight.ptr,
            d_exact64_output.ptr,
            rows,
            rows,
            experts,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        qwen4_exp_q5_1_selected_gemv_wave64_bf16_bf16_out(
            d_x.ptr,
            d_selected.ptr,
            d_weight.ptr,
            d_wave64_output.ptr,
            rows,
            rows,
            experts,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out(
            d_grouped_x.ptr,
            d_expert_start.ptr,
            d_weight.ptr,
            d_grouped_output.ptr,
            rows,
            experts,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_bf16_bf16_out(
            d_grouped_x.ptr,
            d_expert_start.ptr,
            d_weight.ptr,
            d_grouped_out8.ptr,
            rows,
            experts,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out(
            d_grouped_x.ptr,
            d_expert_start.ptr,
            d_weight.ptr,
            d_grouped_expertgrid64.ptr,
            rows,
            experts,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out(
            d_grouped_x.ptr,
            d_expert_start.ptr,
            d_expert_start_wmma.ptr,
            d_tile_expert.ptr,
            d_weight.ptr,
            d_wmma_output.ptr,
            rows,
            experts,
            in_features,
            out_features,
            int(expert_start_wmma[-1]),
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(d_output, expected_bits.shape, np.uint16, runtime)
        exact128_actual = _download(
            d_exact128_output, expected_bits.shape, np.uint16, runtime
        )
        exact64_actual = _download(
            d_exact64_output, expected_bits.shape, np.uint16, runtime
        )
        wave64_actual = _download(
            d_wave64_output, expected_bits.shape, np.uint16, runtime
        )
        grouped_actual = _download(
            d_grouped_output, grouped_expected_bits.shape, np.uint16, runtime
        )
        grouped_out8 = _download(
            d_grouped_out8, grouped_expected_bits.shape, np.uint16, runtime
        )
        grouped_expertgrid64 = _download(
            d_grouped_expertgrid64, grouped_expected_bits.shape, np.uint16, runtime
        )
        wmma_actual = _download(
            d_wmma_output, grouped_expected_bits.shape, np.uint16, runtime
        )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(actual, expected_bits)
    np.testing.assert_array_equal(exact128_actual, expected_bits)
    np.testing.assert_array_equal(exact64_actual, expected_bits)
    np.testing.assert_allclose(
        bf16_to_float32(wave64_actual), expected, rtol=2e-2, atol=2e-2
    )
    np.testing.assert_array_equal(grouped_actual, grouped_expected_bits)
    np.testing.assert_array_equal(grouped_out8, grouped_expected_bits)
    np.testing.assert_array_equal(grouped_expertgrid64, grouped_expected_bits)
    np.testing.assert_allclose(
        bf16_to_float32(wmma_actual),
        bf16_to_float32(grouped_expected_bits),
        rtol=2e-2,
        atol=2e-2,
    )


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _alloc(shape, dtype, runtime, allocations):
    device = malloc(int(np.prod(shape)) * np.dtype(dtype).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
