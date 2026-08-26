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


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_q5_1_selected_matches_cpu_dequant_oracle() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        build_qwen4_exp_q5_1,
        qwen4_exp_q5_1_selected_gemv_bf16_bf16_out,
    )

    runtime = get_hip_runtime()
    library = build_qwen4_exp_q5_1(load=True)
    rng = np.random.default_rng(51)
    rows, experts, in_features, out_features = 3, 4, 64, 48
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
    selected = np.array([3, 0, 2], dtype=np.int64)
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

    allocations = []
    try:
        d_x = _upload(x_bits, runtime, allocations)
        d_selected = _upload(selected, runtime, allocations)
        d_weight = _upload(raw, runtime, allocations)
        d_output = _alloc(expected_bits.shape, np.uint16, runtime, allocations)
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
        runtime.device_synchronize()
        actual = _download(d_output, expected_bits.shape, np.uint16, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(actual, expected_bits)


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
