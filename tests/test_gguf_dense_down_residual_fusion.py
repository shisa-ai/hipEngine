"""Bit-exact c1 dense-down projection plus rounded-BF16 residual siblings."""

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
from hipengine.kernels.hip_gfx1100.fused import gguf_bf16_add
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (
    build_dense_gemv,
    dense_gemv_out_bf16,
    dense_gemv_out_bf16_residual_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_pack8_prefill_bf16_bf16_out,
    gguf_q4_k_pack8_prefill_bf16_residual_bf16_out,
)
from hipengine.kernels.registry import KernelKey, resolve
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_pack8
from tests.test_gguf_q4_k_gemv import make_q4_k_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    words = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).copy()
    words += 0x7FFF + ((words >> 16) & 1)
    return (words >> 16).astype(np.uint16)


def _bf16_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def _cpu_rounded_residual(
    projection_bits: np.ndarray,
    residual_bits: np.ndarray,
) -> np.ndarray:
    return _bf16_bits(_bf16_f32(projection_bits) + _bf16_f32(residual_bits))


def test_dense_down_residual_registry_contract() -> None:
    q4_key = KernelKey(
        "hip_gfx1100",
        "linear+residual",
        "gguf_q4_k",
        "pack8_prefill_bf16_residual_bf16_out",
    )
    dense_key = KernelKey(
        "hip_gfx1100",
        "linear+residual",
        "bf16",
        "out_bf16_residual_bf16_out",
    )
    assert resolve(
        backend=q4_key.backend,
        layer=q4_key.layer,
        quant=q4_key.quant,
        variant=q4_key.variant,
    ) is gguf_q4_k_pack8_prefill_bf16_residual_bf16_out
    assert resolve(
        backend=dense_key.backend,
        layer=dense_key.layer,
        quant=dense_key.quant,
        variant=dense_key.variant,
    ) is dense_gemv_out_bf16_residual_bf16_out


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q4_pack8_down_residual_is_bit_exact() -> None:
    rows, in_features, out_features = 1, 256, 16
    packed = repack_gguf_q4_k_pack8(make_q4_k_weight(out_features, in_features))
    rng = np.random.default_rng(0xD3B4)
    x = _bf16_bits(rng.normal(0.0, 0.2, size=(rows, in_features)))
    residual = _bf16_bits(rng.normal(0.0, 0.2, size=(rows, out_features)))
    arrays = (x, packed.qweight, packed.scales, packed.mins, residual)
    inputs = [malloc(array.nbytes) for array in arrays]
    projection_d = malloc(residual.nbytes)
    control_d = malloc(residual.nbytes)
    candidate_d = malloc(residual.nbytes)
    projection = np.empty_like(residual)
    control = np.empty_like(residual)
    candidate = np.empty_like(residual)
    library = build_gguf_q4_k_gemv(load=True)
    try:
        for array, allocation in zip(arrays, inputs, strict=True):
            copy_host_to_device(allocation, host_array_ptr(array), array.nbytes)
        x_d, qweight_d, scales_d, mins_d, residual_d = inputs
        gguf_q4_k_pack8_prefill_bf16_bf16_out(
            x_d.ptr,
            qweight_d.ptr,
            scales_d.ptr,
            mins_d.ptr,
            projection_d.ptr,
            rows,
            in_features,
            out_features,
            threads=32,
            library=library,
        )
        gguf_bf16_add(
            residual_d.ptr,
            projection_d.ptr,
            control_d.ptr,
            residual.size,
        )
        gguf_q4_k_pack8_prefill_bf16_residual_bf16_out(
            x_d.ptr,
            qweight_d.ptr,
            scales_d.ptr,
            mins_d.ptr,
            residual_d.ptr,
            candidate_d.ptr,
            rows,
            in_features,
            out_features,
            threads=32,
            library=library,
        )
        copy_device_to_host(host_array_ptr(projection), projection_d, projection.nbytes)
        copy_device_to_host(host_array_ptr(control), control_d, control.nbytes)
        copy_device_to_host(host_array_ptr(candidate), candidate_d, candidate.nbytes)
    finally:
        for allocation in (candidate_d, control_d, projection_d, *inputs):
            free(allocation)
    np.testing.assert_array_equal(candidate, control)
    np.testing.assert_array_equal(candidate, _cpu_rounded_residual(projection, residual))


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_dense_bf16_down_residual_is_bit_exact() -> None:
    rows, in_features, out_features = 1, 256, 16
    rng = np.random.default_rng(0xD3B6)
    x = _bf16_bits(rng.normal(0.0, 0.1, size=(rows, in_features)))
    weight = _bf16_bits(
        rng.normal(0.0, 0.1, size=(out_features, in_features))
    )
    residual = _bf16_bits(rng.normal(0.0, 0.2, size=(rows, out_features)))
    arrays = (x, weight, residual)
    inputs = [malloc(array.nbytes) for array in arrays]
    projection_d = malloc(residual.nbytes)
    control_d = malloc(residual.nbytes)
    candidate_d = malloc(residual.nbytes)
    projection = np.empty_like(residual)
    control = np.empty_like(residual)
    candidate = np.empty_like(residual)
    library = build_dense_gemv(load=True)
    try:
        for array, allocation in zip(arrays, inputs, strict=True):
            copy_host_to_device(allocation, host_array_ptr(array), array.nbytes)
        x_d, weight_d, residual_d = inputs
        dense_gemv_out_bf16(
            x_d.ptr,
            weight_d.ptr,
            projection_d.ptr,
            rows,
            in_features,
            out_features,
            threads=256,
            library=library,
        )
        gguf_bf16_add(
            residual_d.ptr,
            projection_d.ptr,
            control_d.ptr,
            residual.size,
        )
        dense_gemv_out_bf16_residual_bf16_out(
            x_d.ptr,
            weight_d.ptr,
            residual_d.ptr,
            candidate_d.ptr,
            rows,
            in_features,
            out_features,
            threads=256,
            library=library,
        )
        copy_device_to_host(host_array_ptr(projection), projection_d, projection.nbytes)
        copy_device_to_host(host_array_ptr(control), control_d, control.nbytes)
        copy_device_to_host(host_array_ptr(candidate), candidate_d, candidate.nbytes)
    finally:
        for allocation in (candidate_d, control_d, projection_d, *inputs):
            free(allocation)
    np.testing.assert_array_equal(candidate, control)
    np.testing.assert_array_equal(candidate, _cpu_rounded_residual(projection, residual))
