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
from hipengine.kernels.cpu_reference.qwen4_exp import (
    PLEConvState,
    dilated_depthwise_conv,
    ple_signed_sqrt_gate,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_qwen4_exp_ple_build_and_registry_contract() -> None:
    from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_ple import (
        plan_qwen4_exp_ple_build,
        qwen4_exp_ple_signed_sqrt_gate_f32,
        register_qwen4_exp_ple_kernels,
    )
    from hipengine.kernels.registry import resolve

    artifact = plan_qwen4_exp_ple_build()
    assert artifact.output_path.name == "qwen4_exp_ple.so"
    register_qwen4_exp_ple_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="ple_signed_sqrt_gate",
            quant="f32",
            variant="strict",
        )
        is qwen4_exp_ple_signed_sqrt_gate_f32
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_ple_native_primitives_match_cpu_at_production_width() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_ple import (
        build_qwen4_exp_ple,
        qwen4_exp_ple_dilated_depthwise_conv_f32,
        qwen4_exp_ple_repeat_gated_value_f32,
        qwen4_exp_ple_signed_sqrt_gate_f32,
    )

    runtime = get_hip_runtime()
    library = build_qwen4_exp_ple(load=True)
    rng = np.random.default_rng(4_038)
    rows, branches, hidden = 5, 4, 2_560
    channels = branches * hidden
    key = rng.normal(0.0, 0.2, size=(rows, branches, hidden)).astype(np.float32)
    query = rng.normal(0.0, 0.2, size=(rows, branches, hidden)).astype(np.float32)
    scores = np.sum(key * query, axis=-1, dtype=np.float32) / np.float32(np.sqrt(hidden))
    expected_gate = ple_signed_sqrt_gate(scores)
    value = rng.normal(0.0, 0.2, size=(rows, hidden)).astype(np.float32)
    expected_gated = (value[:, None, :] * expected_gate[:, :, None]).astype(np.float32)

    conv_input = rng.normal(0.0, 0.1, size=(rows, channels)).astype(np.float32)
    kernel = rng.normal(0.0, 0.1, size=(channels, 4)).astype(np.float32)
    history = rng.normal(0.0, 0.1, size=(9, channels)).astype(np.float32)
    expected_conv, expected_state = dilated_depthwise_conv(
        conv_input,
        kernel,
        dilation=3,
        positions=np.arange(10, 10 + rows),
        state=PLEConvState(history.copy(), 10),
    )

    allocations = []
    try:
        d_key = _upload(key, runtime, allocations)
        d_query = _upload(query, runtime, allocations)
        d_gate = _alloc(expected_gate.shape, np.float32, runtime, allocations)
        qwen4_exp_ple_signed_sqrt_gate_f32(
            d_key.ptr,
            d_query.ptr,
            d_gate.ptr,
            rows,
            branches,
            hidden,
            library=library,
            runtime=runtime,
        )
        d_value = _upload(value, runtime, allocations)
        d_gated = _alloc(expected_gated.shape, np.float32, runtime, allocations)
        qwen4_exp_ple_repeat_gated_value_f32(
            d_value.ptr,
            d_gate.ptr,
            d_gated.ptr,
            rows,
            branches,
            hidden,
            library=library,
            runtime=runtime,
        )
        d_conv_input = _upload(conv_input, runtime, allocations)
        d_kernel = _upload(kernel, runtime, allocations)
        d_history = _upload(history, runtime, allocations)
        d_conv = _alloc(expected_conv.shape, np.float32, runtime, allocations)
        qwen4_exp_ple_dilated_depthwise_conv_f32(
            d_conv_input.ptr,
            d_kernel.ptr,
            d_history.ptr,
            d_conv.ptr,
            rows,
            channels,
            4,
            3,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual_gate = _download(d_gate, expected_gate.shape, np.float32, runtime)
        actual_gated = _download(d_gated, expected_gated.shape, np.float32, runtime)
        actual_conv = _download(d_conv, expected_conv.shape, np.float32, runtime)
        actual_state = _download(d_history, history.shape, np.float32, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual_gate, expected_gate, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual_gated, expected_gated, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual_conv, expected_conv, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(actual_state, expected_state.history)


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
