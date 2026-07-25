"""Exact dense GGUF Q5_K pack8 decode coverage for Laguna Q2 XL."""

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
    gguf_q5_k_pack8_gemv_bf16_bf16_out,
    gguf_q5_k_pack8_gemv_bf16_f32_out,
    gguf_q5_k_pack8_gemv_decode_bf16_bf16_out,
    gguf_q5_k_pack8_gemv_decode_bf16_f32_out,
    gguf_q5_k_pair_pack8_gemv_decode_bf16_bf16_out,
    gguf_q5_k_pair_pack8_gemv_decode_bf16_f32_out,
    gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out,
    gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_f32_out,
    gguf_q5_k_pair_wave32x2_gemv_decode_bf16_f32_out,
    gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out,
    gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_f32_out,
    gguf_q5_k_wave32x2_gemv_decode_bf16_bf16_out,
    gguf_q5_k_wave32x2_gemv_decode_bf16_f32_out,
    register_gguf_k_gemv_kernels,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from tests._gguf_synthetic_weights import make_q5_k_weight


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


def _run(
    fn,
    x: np.ndarray,
    qweight: np.ndarray,
    *,
    out_dtype: np.dtype,
    library,
) -> np.ndarray:
    rows, in_features = x.shape
    out_features = qweight.shape[0]
    x_buf = malloc(x.nbytes)
    qweight_buf = malloc(qweight.nbytes)
    out = np.empty((rows, out_features), dtype=out_dtype)
    out_buf = malloc(out.nbytes)
    try:
        copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
        copy_host_to_device(qweight_buf, host_array_ptr(qweight), qweight.nbytes)
        fn(
            x_buf.ptr,
            qweight_buf.ptr,
            out_buf.ptr,
            rows,
            in_features,
            out_features,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), out_buf, out.nbytes)
        return out
    finally:
        for buffer in (out_buf, qweight_buf, x_buf):
            free(buffer)


def test_q5_k_pack8_decode_registry_keys_resolve() -> None:
    register_gguf_k_gemv_kernels()
    assert resolve(
        backend="hip_gfx1100",
        layer="linear_pair",
        quant="gguf_q5_k",
        variant="pack8_gemv_decode_bf16_bf16_out",
    ) is gguf_q5_k_pair_pack8_gemv_decode_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear_pair",
        quant="gguf_q5_k",
        variant="pack8_gemv_decode_bf16_f32_out",
    ) is gguf_q5_k_pair_pack8_gemv_decode_bf16_f32_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q5_k",
        variant="pack8_gemv_decode_bf16_bf16_out",
    ) is gguf_q5_k_pack8_gemv_decode_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q5_k",
        variant="pack8_gemv_decode_bf16_f32_out",
    ) is gguf_q5_k_pack8_gemv_decode_bf16_f32_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q5_k",
        variant="wave32x2_gemv_decode_bf16_bf16_out",
    ) is gguf_q5_k_wave32x2_gemv_decode_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear_pair",
        quant="gguf_q5_k",
        variant="wave32x2_gemv_decode_bf16_f32_out",
    ) is gguf_q5_k_pair_wave32x2_gemv_decode_bf16_f32_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q5_k",
        variant="wave32x2_fixed_meta_gemv_decode_bf16_bf16_out",
    ) is gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear_pair",
        quant="gguf_q5_k",
        variant="wave32x2_fixed_meta_gemv_decode_bf16_bf16_out",
    ) is gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="linear_pair",
        quant="gguf_q5_k",
        variant="wave32x2_fixed_meta_gemv_decode_bf16_f32_out",
    ) is gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_f32_out
    assert not is_registered(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q5_k",
            "wave32x2_gemv_decode_bf16_f32_out",
        )
    )
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            "linear",
            "gguf_q5_k",
            "wave32x2_gemv_decode_bf16_bf16_out",
        )
    )


def test_q5_k_pack8_decode_wrapper_rejects_unaligned_shape() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        gguf_q5_k_pair_pack8_gemv_decode_bf16_bf16_out(
            1,
            2,
            3,
            4,
            5,
            rows=1,
            in_features=256,
            out_features=7,
        )
    with pytest.raises(ValueError, match="divisible by 8"):
        gguf_q5_k_pack8_gemv_decode_bf16_bf16_out(
            1,
            2,
            3,
            rows=1,
            in_features=256,
            out_features=7,
        )
    with pytest.raises(ValueError, match="divisible by 8"):
        gguf_q5_k_pair_pack8_gemv_decode_bf16_f32_out(
            1,
            2,
            3,
            4,
            5,
            rows=1,
            in_features=256,
            out_features=8,
            out_features_b=7,
        )


def test_q5_k_wave32x2_wrapper_rejects_out_of_scope_shapes() -> None:
    with pytest.raises(ValueError, match="rows must be exactly 1"):
        gguf_q5_k_wave32x2_gemv_decode_bf16_bf16_out(
            1,
            2,
            3,
            rows=2,
            in_features=256,
            out_features=2,
        )
    with pytest.raises(ValueError, match="divisible by 2"):
        gguf_q5_k_wave32x2_gemv_decode_bf16_bf16_out(
            1,
            2,
            3,
            rows=1,
            in_features=256,
            out_features=3,
        )
    with pytest.raises(ValueError, match="threads must be 32"):
        gguf_q5_k_wave32x2_gemv_decode_bf16_f32_out(
            1,
            2,
            3,
            rows=1,
            in_features=256,
            out_features=2,
            threads=64,
        )
    with pytest.raises(ValueError, match="divisible by 2"):
        gguf_q5_k_pair_wave32x2_gemv_decode_bf16_f32_out(
            1,
            2,
            3,
            4,
            5,
            rows=1,
            in_features=256,
            out_features=2,
            out_features_b=3,
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("in_features,out_features", [(256, 8), (512, 16)])
def test_q5_k_wave32x2_is_bit_exact_to_pack8_for_synthetic_blocks(
    in_features: int,
    out_features: int,
) -> None:
    rng = np.random.default_rng(20260730 + in_features + out_features)
    values = rng.normal(0.0, 0.2, size=(1, in_features)).astype(np.float32)
    values[0, ::17] *= np.float32(2.0**8)
    values[0, 1::19] *= np.float32(2.0**-8)
    values[0, 2::23] = -values[0, 2::23]
    x = _f32_to_bf16_u16(values)
    qweight = make_q5_k_weight(out_features, in_features)
    library = build_gguf_k_gemv(load=True)

    expected_bf16 = _run(
        gguf_q5_k_pack8_gemv_decode_bf16_bf16_out,
        x,
        qweight,
        out_dtype=np.uint16,
        library=library,
    )
    actual_bf16 = _run(
        gguf_q5_k_wave32x2_gemv_decode_bf16_bf16_out,
        x,
        qweight,
        out_dtype=np.uint16,
        library=library,
    )
    expected_f32 = _run(
        gguf_q5_k_pack8_gemv_decode_bf16_f32_out,
        x,
        qweight,
        out_dtype=np.float32,
        library=library,
    )
    actual_f32 = _run(
        gguf_q5_k_wave32x2_gemv_decode_bf16_f32_out,
        x,
        qweight,
        out_dtype=np.float32,
        library=library,
    )

    np.testing.assert_array_equal(actual_bf16, expected_bf16)
    np.testing.assert_array_equal(actual_f32.view(np.uint32), expected_f32.view(np.uint32))


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("in_features,out_features", [(256, 8), (512, 16)])
def test_q5_k_wave32x2_fixed_meta_is_bit_exact_to_retained(
    in_features: int,
    out_features: int,
) -> None:
    rng = np.random.default_rng(20260725 + in_features + out_features)
    x = _f32_to_bf16_u16(rng.normal(0.0, 0.2, size=(1, in_features)))
    qweight = make_q5_k_weight(out_features, in_features)
    library = build_gguf_k_gemv(load=True)

    expected_bf16 = _run(
        gguf_q5_k_wave32x2_gemv_decode_bf16_bf16_out,
        x,
        qweight,
        out_dtype=np.uint16,
        library=library,
    )
    actual_bf16 = _run(
        gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out,
        x,
        qweight,
        out_dtype=np.uint16,
        library=library,
    )
    expected_f32 = _run(
        gguf_q5_k_wave32x2_gemv_decode_bf16_f32_out,
        x,
        qweight,
        out_dtype=np.float32,
        library=library,
    )
    actual_f32 = _run(
        gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_f32_out,
        x,
        qweight,
        out_dtype=np.float32,
        library=library,
    )

    np.testing.assert_array_equal(actual_bf16, expected_bf16)
    np.testing.assert_array_equal(actual_f32.view(np.uint32), expected_f32.view(np.uint32))


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q5_k_wave32x2_preserves_adversarial_nonfinite_classes_and_defined_bits() -> None:
    in_features, out_features = 256, 8
    x = _f32_to_bf16_u16(
        np.linspace(-0.5, 0.5, in_features, dtype=np.float32).reshape(1, -1)
    )
    x[0, :8] = np.asarray(
        [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x3F80, 0xBF80, 0x0001],
        dtype=np.uint16,
    )
    qweight = make_q5_k_weight(out_features, in_features)
    # Exercise retained FP16 coefficient edge behavior independently by row.
    qweight[0, 0:2] = np.asarray([0x00, 0x80], dtype=np.uint8)  # d = -0
    qweight[1, 2:4] = np.asarray([0x00, 0x7C], dtype=np.uint8)  # dmin = +inf
    qweight[2, 0:2] = np.asarray([0x01, 0x7E], dtype=np.uint8)  # d = qNaN
    library = build_gguf_k_gemv(load=True)

    expected_bf16 = _run(
        gguf_q5_k_pack8_gemv_decode_bf16_bf16_out,
        x,
        qweight,
        out_dtype=np.uint16,
        library=library,
    )
    actual_bf16 = _run(
        gguf_q5_k_wave32x2_gemv_decode_bf16_bf16_out,
        x,
        qweight,
        out_dtype=np.uint16,
        library=library,
    )
    expected_f32 = _run(
        gguf_q5_k_pack8_gemv_decode_bf16_f32_out,
        x,
        qweight,
        out_dtype=np.float32,
        library=library,
    )
    actual_f32 = _run(
        gguf_q5_k_wave32x2_gemv_decode_bf16_f32_out,
        x,
        qweight,
        out_dtype=np.float32,
        library=library,
    )
    fixed_meta_bf16 = _run(
        gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out,
        x,
        qweight,
        out_dtype=np.uint16,
        library=library,
    )
    fixed_meta_f32 = _run(
        gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_f32_out,
        x,
        qweight,
        out_dtype=np.float32,
        library=library,
    )

    expected_bf16_nan = (expected_bf16 & 0x7FFF) > 0x7F80
    for observed_bf16 in (actual_bf16, fixed_meta_bf16):
        observed_bf16_nan = (observed_bf16 & 0x7FFF) > 0x7F80
        np.testing.assert_array_equal(observed_bf16_nan, expected_bf16_nan)
        np.testing.assert_array_equal(
            observed_bf16[~expected_bf16_nan], expected_bf16[~expected_bf16_nan]
        )
    expected_f32_bits = expected_f32.view(np.uint32)
    expected_f32_nan = (expected_f32_bits & 0x7FFFFFFF) > 0x7F800000
    for observed_f32 in (actual_f32, fixed_meta_f32):
        observed_f32_bits = observed_f32.view(np.uint32)
        observed_f32_nan = (observed_f32_bits & 0x7FFFFFFF) > 0x7F800000
        np.testing.assert_array_equal(observed_f32_nan, expected_f32_nan)
        np.testing.assert_array_equal(
            observed_f32_bits[~expected_f32_nan],
            expected_f32_bits[~expected_f32_nan],
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("in_features", [6144, 9216])
def test_q5_k_wave32x2_is_bit_exact_at_laguna_output_shapes(in_features: int) -> None:
    rows, out_features = 1, 3072
    rng = np.random.default_rng(20260731 + in_features)
    x = _f32_to_bf16_u16(rng.normal(0.0, 0.2, size=(rows, in_features)))
    eight_rows = make_q5_k_weight(8, in_features)
    qweight = np.tile(eight_rows, (out_features // 8, 1))
    library = build_gguf_k_gemv(load=True)

    expected = _run(
        gguf_q5_k_pack8_gemv_decode_bf16_bf16_out,
        x,
        qweight,
        out_dtype=np.uint16,
        library=library,
    )
    actual = _run(
        gguf_q5_k_wave32x2_gemv_decode_bf16_bf16_out,
        x,
        qweight,
        out_dtype=np.uint16,
        library=library,
    )
    fixed_meta = _run(
        gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out,
        x,
        qweight,
        out_dtype=np.uint16,
        library=library,
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(fixed_meta, expected)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("out_features,in_features", [(48, 3072), (72, 3072), (1024, 3072)])
def test_q5_k_pack8_decode_is_bit_exact_to_existing_raw_pack8(
    out_features: int,
    in_features: int,
) -> None:
    rng = np.random.default_rng(20260724 + out_features)
    x = _f32_to_bf16_u16(rng.normal(0.0, 0.2, size=(1, in_features)))
    qweight = make_q5_k_weight(out_features, in_features)
    library = build_gguf_k_gemv(load=True)

    baseline_bf16 = _run(
        gguf_q5_k_pack8_gemv_bf16_bf16_out,
        x,
        qweight,
        out_dtype=np.uint16,
        library=library,
    )
    actual_bf16 = _run(
        gguf_q5_k_pack8_gemv_decode_bf16_bf16_out,
        x,
        qweight,
        out_dtype=np.uint16,
        library=library,
    )
    baseline_f32 = _run(
        gguf_q5_k_pack8_gemv_bf16_f32_out,
        x,
        qweight,
        out_dtype=np.float32,
        library=library,
    )
    actual_f32 = _run(
        gguf_q5_k_pack8_gemv_decode_bf16_f32_out,
        x,
        qweight,
        out_dtype=np.float32,
        library=library,
    )

    np.testing.assert_array_equal(actual_bf16, baseline_bf16)
    np.testing.assert_array_equal(actual_f32, baseline_f32)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q5_k_pack8_decode_pair_is_bit_exact_to_two_singletons() -> None:
    rows, in_features, out_features = 1, 3072, 1024
    rng = np.random.default_rng(20260724)
    x = _f32_to_bf16_u16(rng.normal(0.0, 0.2, size=(rows, in_features)))
    qweight_a = make_q5_k_weight(out_features, in_features)
    qweight_b = np.roll(qweight_a, 17, axis=0).copy()
    library = build_gguf_k_gemv(load=True)

    expected_a = _run(
        gguf_q5_k_pack8_gemv_decode_bf16_bf16_out,
        x,
        qweight_a,
        out_dtype=np.uint16,
        library=library,
    )
    expected_b = _run(
        gguf_q5_k_pack8_gemv_decode_bf16_bf16_out,
        x,
        qweight_b,
        out_dtype=np.uint16,
        library=library,
    )

    x_buf = malloc(x.nbytes)
    qweight_a_buf = malloc(qweight_a.nbytes)
    qweight_b_buf = malloc(qweight_b.nbytes)
    actual_a = np.empty((rows, out_features), dtype=np.uint16)
    actual_b = np.empty((rows, out_features), dtype=np.uint16)
    fixed_meta_a = np.empty((rows, out_features), dtype=np.uint16)
    fixed_meta_b = np.empty((rows, out_features), dtype=np.uint16)
    actual_a_buf = malloc(actual_a.nbytes)
    actual_b_buf = malloc(actual_b.nbytes)
    fixed_meta_a_buf = malloc(fixed_meta_a.nbytes)
    fixed_meta_b_buf = malloc(fixed_meta_b.nbytes)
    try:
        copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
        copy_host_to_device(qweight_a_buf, host_array_ptr(qweight_a), qweight_a.nbytes)
        copy_host_to_device(qweight_b_buf, host_array_ptr(qweight_b), qweight_b.nbytes)
        gguf_q5_k_pair_pack8_gemv_decode_bf16_bf16_out(
            x_buf.ptr,
            qweight_a_buf.ptr,
            qweight_b_buf.ptr,
            actual_a_buf.ptr,
            actual_b_buf.ptr,
            rows,
            in_features,
            out_features,
            library=library,
        )
        gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out(
            x_buf.ptr,
            qweight_a_buf.ptr,
            qweight_b_buf.ptr,
            fixed_meta_a_buf.ptr,
            fixed_meta_b_buf.ptr,
            rows,
            in_features,
            out_features,
            out_features,
            library=library,
        )
        copy_device_to_host(host_array_ptr(actual_a), actual_a_buf, actual_a.nbytes)
        copy_device_to_host(host_array_ptr(actual_b), actual_b_buf, actual_b.nbytes)
        copy_device_to_host(
            host_array_ptr(fixed_meta_a), fixed_meta_a_buf, fixed_meta_a.nbytes
        )
        copy_device_to_host(
            host_array_ptr(fixed_meta_b), fixed_meta_b_buf, fixed_meta_b.nbytes
        )
    finally:
        for buffer in (
            fixed_meta_b_buf,
            fixed_meta_a_buf,
            actual_b_buf,
            actual_a_buf,
            qweight_b_buf,
            qweight_a_buf,
            x_buf,
        ):
            free(buffer)

    np.testing.assert_array_equal(actual_a, expected_a)
    np.testing.assert_array_equal(actual_b, expected_b)
    np.testing.assert_array_equal(fixed_meta_a, expected_a)
    np.testing.assert_array_equal(fixed_meta_b, expected_b)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("out_features_a,out_features_b", [(9216, 72), (6144, 48)])
def test_q5_k_pack8_decode_f32_pair_is_bit_exact_at_laguna_attention_shapes(
    out_features_a: int,
    out_features_b: int,
) -> None:
    rows, in_features = 1, 3072
    rng = np.random.default_rng(20260725 + out_features_a)
    x = _f32_to_bf16_u16(rng.normal(0.0, 0.2, size=(rows, in_features)))
    eight_rows = make_q5_k_weight(8, in_features)
    qweight_a = np.tile(eight_rows, (out_features_a // 8, 1))
    qweight_b = np.tile(np.roll(eight_rows, 1, axis=0), (out_features_b // 8, 1))
    library = build_gguf_k_gemv(load=True)

    expected_a = _run(
        gguf_q5_k_pack8_gemv_decode_bf16_f32_out,
        x,
        qweight_a,
        out_dtype=np.float32,
        library=library,
    )
    expected_b = _run(
        gguf_q5_k_pack8_gemv_decode_bf16_f32_out,
        x,
        qweight_b,
        out_dtype=np.float32,
        library=library,
    )

    x_buf = malloc(x.nbytes)
    qweight_a_buf = malloc(qweight_a.nbytes)
    qweight_b_buf = malloc(qweight_b.nbytes)
    actual_a = np.empty((rows, out_features_a), dtype=np.float32)
    actual_b = np.empty((rows, out_features_b), dtype=np.float32)
    candidate_a = np.empty((rows, out_features_a), dtype=np.float32)
    candidate_b = np.empty((rows, out_features_b), dtype=np.float32)
    fixed_meta_a = np.empty((rows, out_features_a), dtype=np.float32)
    fixed_meta_b = np.empty((rows, out_features_b), dtype=np.float32)
    actual_a_buf = malloc(actual_a.nbytes)
    actual_b_buf = malloc(actual_b.nbytes)
    candidate_a_buf = malloc(candidate_a.nbytes)
    candidate_b_buf = malloc(candidate_b.nbytes)
    fixed_meta_a_buf = malloc(fixed_meta_a.nbytes)
    fixed_meta_b_buf = malloc(fixed_meta_b.nbytes)
    try:
        copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
        copy_host_to_device(qweight_a_buf, host_array_ptr(qweight_a), qweight_a.nbytes)
        copy_host_to_device(qweight_b_buf, host_array_ptr(qweight_b), qweight_b.nbytes)
        gguf_q5_k_pair_pack8_gemv_decode_bf16_f32_out(
            x_buf.ptr,
            qweight_a_buf.ptr,
            qweight_b_buf.ptr,
            actual_a_buf.ptr,
            actual_b_buf.ptr,
            rows,
            in_features,
            out_features_a,
            out_features_b,
            library=library,
        )
        gguf_q5_k_pair_wave32x2_gemv_decode_bf16_f32_out(
            x_buf.ptr,
            qweight_a_buf.ptr,
            qweight_b_buf.ptr,
            candidate_a_buf.ptr,
            candidate_b_buf.ptr,
            rows,
            in_features,
            out_features_a,
            out_features_b,
            library=library,
        )
        gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_f32_out(
            x_buf.ptr,
            qweight_a_buf.ptr,
            qweight_b_buf.ptr,
            fixed_meta_a_buf.ptr,
            fixed_meta_b_buf.ptr,
            rows,
            in_features,
            out_features_a,
            out_features_b,
            library=library,
        )
        copy_device_to_host(host_array_ptr(actual_a), actual_a_buf, actual_a.nbytes)
        copy_device_to_host(host_array_ptr(actual_b), actual_b_buf, actual_b.nbytes)
        copy_device_to_host(host_array_ptr(candidate_a), candidate_a_buf, candidate_a.nbytes)
        copy_device_to_host(host_array_ptr(candidate_b), candidate_b_buf, candidate_b.nbytes)
        copy_device_to_host(host_array_ptr(fixed_meta_a), fixed_meta_a_buf, fixed_meta_a.nbytes)
        copy_device_to_host(host_array_ptr(fixed_meta_b), fixed_meta_b_buf, fixed_meta_b.nbytes)
    finally:
        for buffer in (
            fixed_meta_b_buf,
            fixed_meta_a_buf,
            candidate_b_buf,
            candidate_a_buf,
            actual_b_buf,
            actual_a_buf,
            qweight_b_buf,
            qweight_a_buf,
            x_buf,
        ):
            free(buffer)

    np.testing.assert_array_equal(actual_a.view(np.uint32), expected_a.view(np.uint32))
    np.testing.assert_array_equal(actual_b.view(np.uint32), expected_b.view(np.uint32))
    np.testing.assert_array_equal(candidate_a.view(np.uint32), expected_a.view(np.uint32))
    np.testing.assert_array_equal(candidate_b.view(np.uint32), expected_b.view(np.uint32))
    np.testing.assert_array_equal(fixed_meta_a.view(np.uint32), expected_a.view(np.uint32))
    np.testing.assert_array_equal(fixed_meta_b.view(np.uint32), expected_b.view(np.uint32))
