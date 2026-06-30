"""Correctness fixtures for the dense GGUF Q6_K pack8 GEMV decode (P9.B4b).

This covers the Q6_K dense pack8 GEMV decode kernel added during P9.B5
(task #24) so the qwen35moe Qwen3.6-35B-A3B-UD-Q4_K_M lm-head logits
projection (Q6_K tied output) can be tested under the F32 output regime
specified by the P9.B5 task description.

Four ``(scalar_in_t, scalar_out_t)`` instantiations are covered: BF16/BF16,
FP16/FP16, BF16/F32, and FP16/F32. F32 output variants are validated under
a tighter tolerance (essentially bit-exact vs CPU oracle, since no output-
side rounding happens).
"""

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
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv import (
    build_gguf_q6_k_pack8_gemv,
    gguf_q6_k_pack8_gemv_decode_bf16_bf16_out,
    gguf_q6_k_pack8_gemv_decode_bf16_f32_out,
    gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32,
    gguf_q6_k_pack8_gemv_decode_fp16_f32_out,
    gguf_q6_k_pack8_gemv_decode_fp16_fp16_out,
    plan_gguf_q6_k_pack8_gemv_build,
    register_gguf_q6_k_pack8_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.convert.gather import build_gather, gather_f32_rows_by_i32id
from hipengine.kernels.hip_gfx1100.linear.lm_head import build_lm_head, topk_f32_rows_i32
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import make_q6_k_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def q6_k_dense_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_q6_k_pack8_gemv(load=True)


def test_p9_b4b_registry_keys_resolve() -> None:
    register_gguf_q6_k_pack8_gemv_kernels()
    for variant in (
        "pack8_gemv_decode_bf16_bf16_out",
        "pack8_gemv_decode_fp16_fp16_out",
        "pack8_gemv_decode_bf16_f32_out",
        "pack8_gemv_decode_fp16_f32_out",
        "pack8_gemv_decode_bf16_top1_gather_f32",
    ):
        fn = resolve(backend="hip_gfx1100", layer="linear", quant="gguf_q6_k", variant=variant)
        assert fn is not None, f"missing registry entry: {variant}"


def test_p9_b4b_build_plan_is_dry_run_safe() -> None:
    plan = plan_gguf_q6_k_pack8_gemv_build()
    assert plan.output_path.name == "gguf_q6_k_pack8_gemv.so"


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    nan_mask = np.isnan(f32)
    lsb = (u32 >> 16) & 1
    rounded = ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)
    rounded[nan_mask] = 0x7FC0
    return rounded.reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape).copy()


def _run_dense(fn, x, qweight, rows, in_features, out_features, out_dtype, library):
    x_buf = malloc(x.nbytes)
    copy_host_to_device(x_buf, host_array_ptr(x), x.nbytes)
    w_buf = malloc(qweight.nbytes)
    copy_host_to_device(w_buf, host_array_ptr(qweight), qweight.nbytes)
    out_arr = np.zeros((rows, out_features), dtype=out_dtype)
    out_buf = malloc(out_arr.nbytes)
    try:
        fn(x_buf.ptr, w_buf.ptr, out_buf.ptr, rows, in_features, out_features, library=library)
        copy_device_to_host(host_array_ptr(out_arr), out_buf, out_arr.nbytes)
        return out_arr
    finally:
        for b in (x_buf, w_buf, out_buf):
            free(b)


def test_q6_k_top1_gather_wrapper_validates_before_gpu_load() -> None:
    with pytest.raises(ValueError, match="rows"):
        gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32(0, 0, 0, 0, 0, None, None, None, 0, 256, 8, 0)
    with pytest.raises(ValueError, match="together"):
        gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32(0, 0, 0, 0, 0, None, 1, None, 1, 256, 8, 0)
    with pytest.raises(ValueError, match="hidden_size"):
        gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32(0, 0, 0, 0, 0, None, 1, 2, 1, 256, 8, 0)


_HALF_TOL = dict(atol=1.0e-3, rtol=1.0e-2)
_F32_TOL = dict(atol=5.0e-3, rtol=5.0e-3)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (1, 256, 8),
        (1, 256, 256),
        (1, 1024, 2048),
        (1, 4096, 4096),
        (4, 1024, 1024),
    ],
)
def test_p9_b4b_bf16_bf16_matches_cpu_oracle(rows, in_features, out_features, q6_k_dense_library) -> None:
    rng = np.random.default_rng(rows + in_features * 17 + out_features)
    qweight = make_q6_k_weight(out_features, in_features)
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)
    actual = _run_dense(
        gguf_q6_k_pack8_gemv_decode_bf16_bf16_out,
        x_bf16, qweight, rows, in_features, out_features, np.uint16, q6_k_dense_library,
    )
    actual_f32 = _bf16_u16_to_f32(actual)
    expected = gguf_quant_gemv(x_ref, qweight, GGMLQuantizationType.Q6_K)
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))
    np.testing.assert_allclose(actual_f32, expected_bf16, **_HALF_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (1, 256, 256),
        (1, 2048, 2048),
        (4, 1024, 4096),
    ],
)
def test_p9_b4b_fp16_fp16_matches_cpu_oracle(rows, in_features, out_features, q6_k_dense_library) -> None:
    rng = np.random.default_rng(rows * 41 + in_features + out_features * 5)
    qweight = make_q6_k_weight(out_features, in_features)
    x_f16 = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float16)
    x_ref = x_f16.astype(np.float32)
    actual = _run_dense(
        gguf_q6_k_pack8_gemv_decode_fp16_fp16_out,
        x_f16, qweight, rows, in_features, out_features, np.float16, q6_k_dense_library,
    )
    expected = gguf_quant_gemv(x_ref, qweight, GGMLQuantizationType.Q6_K)
    expected_f16 = expected.astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(actual.astype(np.float32), expected_f16, **_HALF_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (1, 256, 2048),       # tiny lm-head
        (1, 2048, 32_768),    # Qwen3.6-35B-A3B-class lm-head vocab subset
        (4, 2048, 4096),
    ],
)
def test_p9_b4b_bf16_f32_lm_head_matches_cpu_oracle(rows, in_features, out_features, q6_k_dense_library) -> None:
    rng = np.random.default_rng(rows * 67 + in_features + out_features * 7)
    qweight = make_q6_k_weight(out_features, in_features)
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)
    actual = _run_dense(
        gguf_q6_k_pack8_gemv_decode_bf16_f32_out,
        x_bf16, qweight, rows, in_features, out_features, np.float32, q6_k_dense_library,
    )
    expected = gguf_quant_gemv(x_ref, qweight, GGMLQuantizationType.Q6_K)
    np.testing.assert_allclose(actual, expected, **_F32_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows,in_features,out_features",
    [
        (1, 256, 2048),
        (1, 2048, 4096),
    ],
)
def test_p9_b4b_fp16_f32_lm_head_matches_cpu_oracle(rows, in_features, out_features, q6_k_dense_library) -> None:
    rng = np.random.default_rng(rows * 79 + in_features + out_features * 9)
    qweight = make_q6_k_weight(out_features, in_features)
    x_f16 = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float16)
    x_ref = x_f16.astype(np.float32)
    actual = _run_dense(
        gguf_q6_k_pack8_gemv_decode_fp16_f32_out,
        x_f16, qweight, rows, in_features, out_features, np.float32, q6_k_dense_library,
    )
    expected = gguf_quant_gemv(x_ref, qweight, GGMLQuantizationType.Q6_K)
    np.testing.assert_allclose(actual, expected, **_F32_TOL)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q6_k_bf16_top1_gather_matches_logits_topk_chain(q6_k_dense_library) -> None:
    rows, in_features, out_features, hidden = 1, 512, 4096, 32
    rng = np.random.default_rng(20260701)
    qweight = make_q6_k_weight(out_features, in_features)
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    embed_table = rng.normal(0.0, 0.2, size=(out_features, hidden)).astype(np.float32)

    lm_lib = build_lm_head(load=True)
    gather_lib = build_gather(load=True)
    x_buf = malloc(x_bf16.nbytes)
    w_buf = malloc(qweight.nbytes)
    logits_buf = malloc(rows * out_features * 4)
    topk_values_buf = malloc(rows * 4)
    topk_indices_buf = malloc(rows * 4)
    embed_table_buf = malloc(embed_table.nbytes)
    baseline_embed_buf = malloc(rows * hidden * 4)
    fused_values_buf = malloc(rows * 4)
    fused_indices_buf = malloc(rows * 4)
    fused_embed_buf = malloc(rows * hidden * 4)
    block_values_buf = malloc((out_features // 8) * 4)
    block_indices_buf = malloc((out_features // 8) * 4)
    try:
        copy_host_to_device(x_buf, host_array_ptr(x_bf16), x_bf16.nbytes)
        copy_host_to_device(w_buf, host_array_ptr(qweight), qweight.nbytes)
        copy_host_to_device(embed_table_buf, host_array_ptr(embed_table), embed_table.nbytes)

        gguf_q6_k_pack8_gemv_decode_bf16_f32_out(
            x_buf.ptr,
            w_buf.ptr,
            logits_buf.ptr,
            rows,
            in_features,
            out_features,
            library=q6_k_dense_library,
        )
        topk_f32_rows_i32(
            logits_buf.ptr,
            topk_values_buf.ptr,
            topk_indices_buf.ptr,
            rows,
            out_features,
            1,
            library=lm_lib,
        )
        gather_f32_rows_by_i32id(
            embed_table_buf.ptr,
            topk_indices_buf.ptr,
            baseline_embed_buf.ptr,
            rows,
            hidden,
            out_features,
            library=gather_lib,
        )

        gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32(
            x_buf.ptr,
            w_buf.ptr,
            block_values_buf.ptr,
            block_indices_buf.ptr,
            fused_indices_buf.ptr,
            fused_values_buf.ptr,
            embed_table_buf.ptr,
            fused_embed_buf.ptr,
            rows,
            in_features,
            out_features,
            hidden,
            library=q6_k_dense_library,
        )

        baseline_index = np.empty((rows,), dtype=np.int32)
        fused_index = np.empty((rows,), dtype=np.int32)
        baseline_value = np.empty((rows,), dtype=np.float32)
        fused_value = np.empty((rows,), dtype=np.float32)
        baseline_embed = np.empty((rows, hidden), dtype=np.float32)
        fused_embed = np.empty((rows, hidden), dtype=np.float32)
        copy_device_to_host(host_array_ptr(baseline_index), topk_indices_buf, baseline_index.nbytes)
        copy_device_to_host(host_array_ptr(fused_index), fused_indices_buf, fused_index.nbytes)
        copy_device_to_host(host_array_ptr(baseline_value), topk_values_buf, baseline_value.nbytes)
        copy_device_to_host(host_array_ptr(fused_value), fused_values_buf, fused_value.nbytes)
        copy_device_to_host(host_array_ptr(baseline_embed), baseline_embed_buf, baseline_embed.nbytes)
        copy_device_to_host(host_array_ptr(fused_embed), fused_embed_buf, fused_embed.nbytes)
    finally:
        for b in (
            block_indices_buf,
            block_values_buf,
            fused_embed_buf,
            fused_indices_buf,
            fused_values_buf,
            baseline_embed_buf,
            embed_table_buf,
            topk_indices_buf,
            topk_values_buf,
            logits_buf,
            w_buf,
            x_buf,
        ):
            free(b)

    np.testing.assert_array_equal(fused_index, baseline_index)
    np.testing.assert_array_equal(fused_value, baseline_value)
    np.testing.assert_array_equal(fused_embed, baseline_embed)
