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
    gr_write,
    grouped_zero_centered_rmsnorm,
    sigmoid_gated_rmsnorm,
)
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_qwen4_exp_gr_build_and_registry_contract() -> None:
    from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
        plan_qwen4_exp_gr_build,
        qwen4_exp_gated_mean_sigmoid_f32,
        qwen4_exp_gated_mean_sigmoid_unfused_f32,
        qwen4_exp_gr_write_bf16_f32,
        register_qwen4_exp_gr_kernels,
    )
    from hipengine.kernels.registry import resolve

    artifact = plan_qwen4_exp_gr_build()
    assert artifact.output_path.name == "qwen4_exp_gr.so"
    assert artifact.sources[0].name == "qwen4_exp_gr.hip"
    register_qwen4_exp_gr_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gr_gated_mean_sigmoid",
            quant="f32",
            variant="strict",
        )
        is qwen4_exp_gated_mean_sigmoid_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gr_gated_mean_sigmoid",
            quant="f32",
            variant="strict_unfused",
        )
        is qwen4_exp_gated_mean_sigmoid_unfused_f32
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="gr_write",
            quant="bf16_f32",
            variant="strict",
        )
        is qwen4_exp_gr_write_bf16_f32
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_qwen4_exp_gr_native_primitives_match_cpu_reference() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
        build_qwen4_exp_gr,
        qwen4_exp_gated_mean_f32,
        qwen4_exp_gated_mean_sigmoid_f32,
        qwen4_exp_gr_write_bf16_f32,
        qwen4_exp_grouped_rmsnorm_bf16_f32,
        qwen4_exp_sigmoid_f32,
        qwen4_exp_sigmoid_gated_rmsnorm_f32,
    )

    runtime = get_hip_runtime()
    library = build_qwen4_exp_gr(load=True)
    rng = np.random.default_rng(38)
    rows, branches, hidden = 2, 4, 2560
    residual_f32 = rng.normal(0.0, 0.3, size=(rows, branches, hidden)).astype(np.float32)
    residual_bits = float_array_to_bf16_bits(residual_f32)
    residual = bf16_to_float32(residual_bits)
    gamma = rng.normal(1.0, 0.05, size=(branches, hidden)).astype(np.float32)
    expected_norm = grouped_zero_centered_rmsnorm(residual, gamma)
    gate = rng.normal(0.0, 0.5, size=residual.shape).astype(np.float32)
    expected_mean = np.mean(expected_norm * gate, axis=1, dtype=np.float32)
    block = rng.normal(0.0, 0.2, size=(rows, hidden)).astype(np.float32)
    inject = rng.normal(0.0, 0.4, size=(rows, branches)).astype(np.float32)
    expected_write_bits = float_array_to_bf16_bits(gr_write(residual, block, inject))

    heads, head_dim = rows * 3, 128
    norm_input = rng.normal(0.0, 0.3, size=(heads, head_dim)).astype(np.float32)
    norm_gate = rng.normal(0.0, 1.0, size=(heads, head_dim)).astype(np.float32)
    norm_gamma = rng.normal(1.0, 0.05, size=(head_dim,)).astype(np.float32)
    expected_sigmoid_norm = sigmoid_gated_rmsnorm(
        norm_input,
        norm_gamma,
        norm_gate,
    )

    allocations = []
    try:
        d_residual = _upload(residual_bits, runtime, allocations)
        d_gamma = _upload(gamma, runtime, allocations)
        d_norm = _alloc(expected_norm.shape, np.float32, runtime, allocations)
        qwen4_exp_grouped_rmsnorm_bf16_f32(
            d_residual.ptr,
            d_gamma.ptr,
            d_norm.ptr,
            rows,
            branches,
            hidden,
            library=library,
            runtime=runtime,
        )
        d_gate = _upload(gate, runtime, allocations)
        d_mean = _alloc(expected_mean.shape, np.float32, runtime, allocations)
        qwen4_exp_gated_mean_f32(
            d_norm.ptr,
            d_gate.ptr,
            d_mean.ptr,
            rows,
            branches,
            hidden,
            library=library,
            runtime=runtime,
        )
        gate_logits = rng.normal(0.0, 0.5, size=residual.shape).astype(np.float32)
        d_gate_logits = _upload(gate_logits, runtime, allocations)
        d_gate_sigmoid = _alloc(gate_logits.shape, np.float32, runtime, allocations)
        d_mean_chain = _alloc(expected_mean.shape, np.float32, runtime, allocations)
        d_mean_fused = _alloc(expected_mean.shape, np.float32, runtime, allocations)
        qwen4_exp_sigmoid_f32(
            d_gate_logits.ptr,
            d_gate_sigmoid.ptr,
            gate_logits.size,
            library=library,
            runtime=runtime,
        )
        qwen4_exp_gated_mean_f32(
            d_norm.ptr,
            d_gate_sigmoid.ptr,
            d_mean_chain.ptr,
            rows,
            branches,
            hidden,
            library=library,
            runtime=runtime,
        )
        qwen4_exp_gated_mean_sigmoid_f32(
            d_norm.ptr,
            d_gate_logits.ptr,
            d_mean_fused.ptr,
            rows,
            branches,
            hidden,
            library=library,
            runtime=runtime,
        )
        d_block = _upload(block, runtime, allocations)
        d_inject = _upload(inject, runtime, allocations)
        d_written = _alloc(residual_bits.shape, np.uint16, runtime, allocations)
        qwen4_exp_gr_write_bf16_f32(
            d_residual.ptr,
            d_block.ptr,
            d_inject.ptr,
            d_written.ptr,
            rows,
            branches,
            hidden,
            library=library,
            runtime=runtime,
        )
        d_norm_input = _upload(norm_input, runtime, allocations)
        d_norm_gamma = _upload(norm_gamma, runtime, allocations)
        d_norm_gate = _upload(norm_gate, runtime, allocations)
        d_sigmoid_norm = _alloc(
            expected_sigmoid_norm.shape,
            np.float32,
            runtime,
            allocations,
        )
        qwen4_exp_sigmoid_gated_rmsnorm_f32(
            d_norm_input.ptr,
            d_norm_gamma.ptr,
            d_norm_gate.ptr,
            d_sigmoid_norm.ptr,
            heads,
            head_dim,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual_norm = _download(d_norm, expected_norm.shape, np.float32, runtime)
        actual_mean = _download(d_mean, expected_mean.shape, np.float32, runtime)
        actual_mean_chain = _download(
            d_mean_chain, expected_mean.shape, np.float32, runtime
        )
        actual_mean_fused = _download(
            d_mean_fused, expected_mean.shape, np.float32, runtime
        )
        actual_gate_chain = _download(
            d_gate_sigmoid, gate_logits.shape, np.float32, runtime
        )
        actual_gate_fused = _download(
            d_gate_logits, gate_logits.shape, np.float32, runtime
        )
        actual_written = _download(d_written, residual_bits.shape, np.uint16, runtime)
        actual_sigmoid_norm = _download(
            d_sigmoid_norm,
            expected_sigmoid_norm.shape,
            np.float32,
            runtime,
        )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual_norm, expected_norm, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(actual_gate_fused, actual_gate_chain)
    np.testing.assert_array_equal(actual_mean_fused, actual_mean_chain)
    np.testing.assert_array_equal(actual_written, expected_write_bits)
    np.testing.assert_allclose(
        actual_sigmoid_norm,
        expected_sigmoid_norm,
        rtol=2e-5,
        atol=2e-5,
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
