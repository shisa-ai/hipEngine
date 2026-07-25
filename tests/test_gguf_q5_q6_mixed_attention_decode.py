"""Exact mixed Q5/Q6 c=1 attention-projection contraction for Laguna."""

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
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_f32_out,
    gguf_q5_q6_attention_q5_qg_mixed_gemv_decode_bf16_f32_out,
    gguf_q5_q6_attention_q5_qg_mixed_local32_fixed_meta_gemv_decode_bf16_f32_out,
    gguf_q5_q6_attention_q5_qg_mixed_q6_fixed_meta_gemv_decode_bf16_f32_out,
    gguf_q6_k_pair_pack8_gemv_decode_bf16_f32_out,
    gguf_q6_q8_attention_q6_qg_mixed_gemv_decode_bf16_f32_out,
    gguf_q6_q8_attention_q6_qg_mixed_q6_fixed_meta_gemv_decode_bf16_f32_out,
    gguf_q8_0_pack8_gemv_bf16_f32_out,
    register_gguf_k_gemv_kernels,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from tests._gguf_synthetic_weights import (
    make_q5_k_weight,
    make_q6_k_weight,
    make_q8_0_weight,
)

_VARIANT = "mixed_pack8_gemv_decode_bf16_f32_out"
_Q6_FIXED_META_VARIANT = "mixed_q6_fixed_meta_pack8_gemv_decode_bf16_f32_out"
_LOCAL32_FIXED_META_VARIANT = (
    "mixed_local32_fixed_meta_pack8_gemv_decode_bf16_f32_out"
)
_Q5_QG_QUANT = "gguf_q5_k+gguf_q6_k+gguf_q6_k+gguf_q5_k"
_Q6_Q8_QUANT = "gguf_q6_k+gguf_q8_0+gguf_q8_0+gguf_q6_k"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _f32_to_bf16_u16(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    rounded = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)
    rounded[np.isnan(f32)] = 0x7FC0
    return rounded.reshape(f32.shape)


def _run_pair(
    fn,
    x: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
    *,
    library,
) -> tuple[np.ndarray, np.ndarray]:
    rows, in_features = x.shape
    out_a = np.empty((rows, weight_a.shape[0]), dtype=np.float32)
    out_b = np.empty((rows, weight_b.shape[0]), dtype=np.float32)
    buffers = [
        malloc(x.nbytes),
        malloc(weight_a.nbytes),
        malloc(weight_b.nbytes),
        malloc(out_a.nbytes),
        malloc(out_b.nbytes),
    ]
    x_buf, weight_a_buf, weight_b_buf, out_a_buf, out_b_buf = buffers
    try:
        copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
        copy_host_to_device(weight_a_buf, host_array_ptr(weight_a), weight_a.nbytes)
        copy_host_to_device(weight_b_buf, host_array_ptr(weight_b), weight_b.nbytes)
        fn(
            x_buf.ptr,
            weight_a_buf.ptr,
            weight_b_buf.ptr,
            out_a_buf.ptr,
            out_b_buf.ptr,
            rows,
            in_features,
            weight_a.shape[0],
            weight_b.shape[0],
            library=library,
        )
        copy_device_to_host(host_array_ptr(out_a), out_a_buf, out_a.nbytes)
        copy_device_to_host(host_array_ptr(out_b), out_b_buf, out_b.nbytes)
        return out_a, out_b
    finally:
        for buffer in reversed(buffers):
            free(buffer)


def _run_singleton(fn, x: np.ndarray, weight: np.ndarray, *, library) -> np.ndarray:
    rows, in_features = x.shape
    output = np.empty((rows, weight.shape[0]), dtype=np.float32)
    buffers = [malloc(x.nbytes), malloc(weight.nbytes), malloc(output.nbytes)]
    x_buf, weight_buf, output_buf = buffers
    try:
        copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
        copy_host_to_device(weight_buf, host_array_ptr(weight), weight.nbytes)
        fn(
            x_buf.ptr,
            weight_buf.ptr,
            output_buf.ptr,
            rows,
            in_features,
            weight.shape[0],
            library=library,
        )
        copy_device_to_host(host_array_ptr(output), output_buf, output.nbytes)
        return output
    finally:
        for buffer in reversed(buffers):
            free(buffer)


def _run_mixed(
    fn,
    x: np.ndarray,
    weights: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    library,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, in_features = x.shape
    outputs = tuple(
        np.empty((rows, weight.shape[0]), dtype=np.float32) for weight in weights
    )
    x_buf = malloc(x.nbytes)
    weight_buffers = [malloc(weight.nbytes) for weight in weights]
    output_buffers = [malloc(output.nbytes) for output in outputs]
    try:
        copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
        for weight, buffer in zip(weights, weight_buffers, strict=True):
            copy_host_to_device(buffer, host_array_ptr(weight), weight.nbytes)
        fn(
            x_buf.ptr,
            *(buffer.ptr for buffer in weight_buffers),
            *(buffer.ptr for buffer in output_buffers),
            rows,
            in_features,
            *(weight.shape[0] for weight in weights),
            library=library,
        )
        for output, buffer in zip(outputs, output_buffers, strict=True):
            copy_device_to_host(host_array_ptr(output), buffer, output.nbytes)
        return outputs
    finally:
        for buffer in reversed(output_buffers):
            free(buffer)
        for buffer in reversed(weight_buffers):
            free(buffer)
        free(x_buf)


def test_mixed_q5_q6_attention_registry_is_role_specific_and_gfx1100_only() -> None:
    register_gguf_k_gemv_kernels()
    assert resolve(
        backend="hip_gfx1100",
        layer="attention_projection_quad",
        quant=_Q5_QG_QUANT,
        variant=_VARIANT,
    ) is gguf_q5_q6_attention_q5_qg_mixed_gemv_decode_bf16_f32_out
    assert resolve(
        backend="hip_gfx1100",
        layer="attention_projection_quad",
        quant=_Q6_Q8_QUANT,
        variant=_VARIANT,
    ) is gguf_q6_q8_attention_q6_qg_mixed_gemv_decode_bf16_f32_out
    assert resolve(
        backend="hip_gfx1100",
        layer="attention_projection_quad",
        quant=_Q5_QG_QUANT,
        variant=_Q6_FIXED_META_VARIANT,
    ) is gguf_q5_q6_attention_q5_qg_mixed_q6_fixed_meta_gemv_decode_bf16_f32_out
    assert resolve(
        backend="hip_gfx1100",
        layer="attention_projection_quad",
        quant=_Q6_Q8_QUANT,
        variant=_Q6_FIXED_META_VARIANT,
    ) is gguf_q6_q8_attention_q6_qg_mixed_q6_fixed_meta_gemv_decode_bf16_f32_out
    assert resolve(
        backend="hip_gfx1100",
        layer="attention_projection_quad",
        quant=_Q5_QG_QUANT,
        variant=_LOCAL32_FIXED_META_VARIANT,
    ) is gguf_q5_q6_attention_q5_qg_mixed_local32_fixed_meta_gemv_decode_bf16_f32_out
    assert not is_registered(
        KernelKey("hip_gfx1151", "attention_projection_quad", _Q5_QG_QUANT, _VARIANT)
    )


def test_mixed_q5_q6_attention_wrapper_rejects_out_of_scope_shapes() -> None:
    with pytest.raises(ValueError, match="rows must be exactly 1"):
        gguf_q5_q6_attention_q5_qg_mixed_gemv_decode_bf16_f32_out(
            *range(1, 10),
            rows=2,
            in_features=3072,
            q_features=6144,
            k_features=1024,
            v_features=1024,
            gate_features=48,
        )
    with pytest.raises(ValueError, match="divisible by 8"):
        gguf_q6_q8_attention_q6_qg_mixed_gemv_decode_bf16_f32_out(
            *range(1, 10),
            rows=1,
            in_features=3072,
            q_features=9216,
            k_features=1023,
            v_features=1024,
            gate_features=72,
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "family,q_features,k_features,v_features,gate_features",
    [
        ("q5_q6", 6144, 1024, 1024, 48),
        ("q6_q8", 9216, 1024, 1024, 72),
    ],
)
def test_mixed_q5_q6_attention_is_bit_exact_at_laguna_role_shapes(
    family: str,
    q_features: int,
    k_features: int,
    v_features: int,
    gate_features: int,
) -> None:
    rows, in_features = 1, 3072
    rng = np.random.default_rng(20260726 + q_features)
    x = _f32_to_bf16_u16(rng.normal(0.0, 0.2, size=(rows, in_features)))
    dimensions = (q_features, k_features, v_features, gate_features)
    factories = (
        (make_q5_k_weight, make_q6_k_weight, make_q6_k_weight, make_q5_k_weight)
        if family == "q5_q6"
        else (make_q6_k_weight, make_q8_0_weight, make_q8_0_weight, make_q6_k_weight)
    )
    weights = tuple(
        factory(features, in_features)
        for factory, features in zip(factories, dimensions, strict=True)
    )
    library = build_gguf_k_gemv(load=True)

    if family == "q5_q6":
        expected_q, expected_gate = _run_pair(
            gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_f32_out,
            x,
            weights[0],
            weights[3],
            library=library,
        )
        expected_k, expected_v = _run_pair(
            gguf_q6_k_pair_pack8_gemv_decode_bf16_f32_out,
            x,
            weights[1],
            weights[2],
            library=library,
        )
        candidate = gguf_q5_q6_attention_q5_qg_mixed_gemv_decode_bf16_f32_out
    else:
        expected_q, expected_gate = _run_pair(
            gguf_q6_k_pair_pack8_gemv_decode_bf16_f32_out,
            x,
            weights[0],
            weights[3],
            library=library,
        )
        expected_k = _run_singleton(
            gguf_q8_0_pack8_gemv_bf16_f32_out,
            x,
            weights[1],
            library=library,
        )
        expected_v = _run_singleton(
            gguf_q8_0_pack8_gemv_bf16_f32_out,
            x,
            weights[2],
            library=library,
        )
        candidate = gguf_q6_q8_attention_q6_qg_mixed_gemv_decode_bf16_f32_out

    actual = _run_mixed(candidate, x, weights, library=library)
    for observed, expected in zip(
        actual,
        (expected_q, expected_k, expected_v, expected_gate),
        strict=True,
    ):
        np.testing.assert_array_equal(observed.view(np.uint32), expected.view(np.uint32))


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_mixed_q5_q6_fixed_meta_q6_blocks_match_retained_mixed_bits() -> None:
    rows, in_features = 1, 3072
    dimensions = (9216, 1024, 1024, 72)
    rng = np.random.default_rng(20260727)
    x = _f32_to_bf16_u16(rng.normal(0.0, 0.2, size=(rows, in_features)))
    weights = (
        make_q5_k_weight(dimensions[0], in_features),
        make_q6_k_weight(dimensions[1], in_features),
        make_q6_k_weight(dimensions[2], in_features),
        make_q5_k_weight(dimensions[3], in_features),
    )
    library = build_gguf_k_gemv(load=True)
    expected = _run_mixed(
        gguf_q5_q6_attention_q5_qg_mixed_gemv_decode_bf16_f32_out,
        x,
        weights,
        library=library,
    )
    actual = _run_mixed(
        gguf_q5_q6_attention_q5_qg_mixed_q6_fixed_meta_gemv_decode_bf16_f32_out,
        x,
        weights,
        library=library,
    )
    for observed, retained in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            observed.view(np.uint32), retained.view(np.uint32)
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "dimensions",
    [
        (6144, 1024, 1024, 48),
        (9216, 1024, 1024, 72),
    ],
)
def test_mixed_q5_q6_local32_matches_fixed_meta_bits(
    dimensions: tuple[int, int, int, int],
) -> None:
    rows, in_features = 1, 3072
    rng = np.random.default_rng(20260729 + dimensions[0])
    x = _f32_to_bf16_u16(rng.normal(0.0, 0.2, size=(rows, in_features)))
    weights = (
        make_q5_k_weight(dimensions[0], in_features),
        make_q6_k_weight(dimensions[1], in_features),
        make_q6_k_weight(dimensions[2], in_features),
        make_q5_k_weight(dimensions[3], in_features),
    )
    library = build_gguf_k_gemv(load=True)
    expected = _run_mixed(
        gguf_q5_q6_attention_q5_qg_mixed_q6_fixed_meta_gemv_decode_bf16_f32_out,
        x,
        weights,
        library=library,
    )
    actual = _run_mixed(
        gguf_q5_q6_attention_q5_qg_mixed_local32_fixed_meta_gemv_decode_bf16_f32_out,
        x,
        weights,
        library=library,
    )
    for observed, retained in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            observed.view(np.uint32), retained.view(np.uint32)
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_mixed_q6_q8_fixed_meta_q6_blocks_match_retained_mixed_bits() -> None:
    rows, in_features = 1, 3072
    dimensions = (9216, 1024, 1024, 72)
    rng = np.random.default_rng(20260728)
    x = _f32_to_bf16_u16(rng.normal(0.0, 0.2, size=(rows, in_features)))
    weights = (
        make_q6_k_weight(dimensions[0], in_features),
        make_q8_0_weight(dimensions[1], in_features),
        make_q8_0_weight(dimensions[2], in_features),
        make_q6_k_weight(dimensions[3], in_features),
    )
    library = build_gguf_k_gemv(load=True)
    expected = _run_mixed(
        gguf_q6_q8_attention_q6_qg_mixed_gemv_decode_bf16_f32_out,
        x,
        weights,
        library=library,
    )
    actual = _run_mixed(
        gguf_q6_q8_attention_q6_qg_mixed_q6_fixed_meta_gemv_decode_bf16_f32_out,
        x,
        weights,
        library=library,
    )
    for observed, retained in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            observed.view(np.uint32), retained.view(np.uint32)
        )
