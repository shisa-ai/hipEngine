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
from hipengine.kernels.cpu_reference.qwen4_exp import PLEConvState, ple_injection
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.qwen4_exp_runner import (
    Qwen4ExpPLEScratch,
    run_qwen4_exp_ple,
)
from tests.test_qwen4_exp_runner_gr import _dense_f32_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_runner_ple_matches_reduced_cpu_oracle() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    rng = np.random.default_rng(40381)
    rows, branches, hidden, kernel_size, dilation = 3, 2, 4, 4, 3
    channels = branches * hidden
    residual_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, branches, hidden)).astype(np.float32)
    )
    residual = bf16_to_float32(residual_bits)
    embedding = rng.normal(0.0, 0.2, size=(rows, hidden)).astype(np.float32)
    key = rng.normal(0.0, 0.15, size=(channels, hidden)).astype(np.float32)
    value = rng.normal(0.0, 0.15, size=(hidden, hidden)).astype(np.float32)
    norm_key = rng.normal(1.0, 0.05, size=(branches, hidden)).astype(np.float32)
    norm_query = rng.normal(1.0, 0.05, size=(branches, hidden)).astype(np.float32)
    norm_conv = rng.normal(1.0, 0.05, size=(branches, hidden)).astype(np.float32)
    conv = rng.normal(0.0, 0.1, size=(channels, kernel_size)).astype(np.float32)
    history = rng.normal(
        0.0,
        0.05,
        size=((kernel_size - 1) * dilation, channels),
    ).astype(np.float32)
    expected = ple_injection(
        residual,
        embedding,
        key,
        value,
        norm_key,
        norm_query,
        norm_conv,
        conv,
        positions=np.arange(10, 10 + rows),
        state=PLEConvState(history.copy(), 10),
        dilation=dilation,
    )
    expected_bits = float_array_to_bf16_bits(expected.residual)

    allocations = []
    scratch = None
    try:
        d_residual = _upload(residual_bits, runtime, allocations)
        d_embedding = _upload(embedding, runtime, allocations)
        weights = {
            "ple_key": _dense_f32_weight("ple_key", key, runtime, allocations),
            "ple_value": _dense_f32_weight("ple_value", value, runtime, allocations),
        }
        d_norm_key = _upload(norm_key, runtime, allocations)
        d_norm_query = _upload(norm_query, runtime, allocations)
        d_norm_conv = _upload(norm_conv, runtime, allocations)
        d_conv = _upload(conv, runtime, allocations)
        d_history = _upload(history, runtime, allocations)
        scratch = Qwen4ExpPLEScratch.allocate(
            rows=rows,
            branches=branches,
            hidden=hidden,
            runtime=runtime,
        )
        output = run_qwen4_exp_ple(
            d_residual.ptr,
            d_embedding.ptr,
            weights,
            norm_key_ptr=d_norm_key.ptr,
            norm_query_ptr=d_norm_query.ptr,
            norm_conv_ptr=d_norm_conv.ptr,
            conv_weight_ptr=d_conv.ptr,
            conv_history_ptr=d_history.ptr,
            scratch=scratch,
            rows=rows,
            branches=branches,
            hidden=hidden,
            conv_kernel=kernel_size,
            dilation=dilation,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(output, expected_bits.shape, np.uint16, runtime)
        actual_history = _download(d_history, history.shape, np.float32, runtime)
    finally:
        if scratch is not None:
            scratch.close()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(actual, expected_bits)
    np.testing.assert_allclose(
        actual_history,
        expected.state.history,
        rtol=1e-6,
        atol=1e-6,
    )


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
