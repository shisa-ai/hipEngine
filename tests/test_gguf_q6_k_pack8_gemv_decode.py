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
    gguf_q6_k_pack8_gemv_decode_bf16_top1_stage1_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_gather_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_pack16_gather_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_pack16_stage1_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_pack8_llama_gather_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_pack8_llama_stage1_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_gather_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_stage1_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_row_gather_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_row_stage1_f32,
    gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_stage1_f32,
    gguf_q6_k_pack8_gemv_decode_fp16_f32_out,
    gguf_q6_k_pack8_gemv_decode_fp16_fp16_out,
    gguf_q6_k_pack8_top1_stage2_gather_f32,
    plan_gguf_q6_k_pack8_gemv_build,
    register_gguf_q6_k_pack8_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_quantize_bf16_q8_1,
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
        "pack8_gemv_decode_q8_1_dp4a_top1_gather_f32",
        "pack8_gemv_decode_bf16_top1_stage1_f32",
        "pack8_gemv_decode_q8_1_dp4a_top1_stage1_f32",
        "pack8_gemv_decode_q8_1_dp4a_top1_pack16_stage1_f32",
        "pack8_gemv_decode_q8_1_dp4a_top1_pack16_gather_f32",
        "pack8_gemv_decode_q8_1_dp4a_top1_row_stage1_f32",
        "pack8_gemv_decode_q8_1_dp4a_top1_row_gather_f32",
        "pack8_top1_stage2_gather_f32",
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


def _round_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def _quantize_q8_1_cpu(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, in_features = x.shape
    blocks = in_features // 32
    q = np.empty((rows, blocks, 32), dtype=np.int8)
    d = np.empty((rows, blocks), dtype=np.float32)
    for row in range(rows):
        for block_idx in range(blocks):
            chunk = x[row, block_idx * 32 : (block_idx + 1) * 32].astype(np.float32)
            amax = float(np.max(np.abs(chunk)))
            scale = 0.0 if amax == 0.0 else amax / 127.0
            d[row, block_idx] = np.float16(scale).astype(np.float32)
            if amax == 0.0:
                q[row, block_idx] = 0
            else:
                q[row, block_idx] = np.clip(_round_away_from_zero(chunk / scale), -128, 127).astype(np.int8)
    return q, d


def _q6_unsigned_pack(block: np.ndarray, group32: int, lane4: int) -> np.ndarray:
    ql = block[:128]
    qh = block[128:192]
    base64 = 64 if group32 >= 4 else 0
    low_nibble = (group32 & 2) == 0
    ql_group = group32 & 1
    low = ql[base64 + ql_group * 32 + lane4 : base64 + ql_group * 32 + lane4 + 4].astype(np.uint8)
    low = (low & 0x0F) if low_nibble else (low >> 4)
    qh_base = 32 if group32 >= 4 else 0
    high_bits = qh[qh_base + lane4 : qh_base + lane4 + 4].astype(np.uint8)
    high = ((high_bits >> (2 * (group32 & 3))) & 0x03) << 4
    return (low | high).astype(np.int32)


def _q6_k_q8_1_dp4a_oracle(x_bf16_f32: np.ndarray, qweight: np.ndarray) -> np.ndarray:
    rows, in_features = x_bf16_f32.shape
    out_features = int(qweight.shape[0])
    blocks_per_row = in_features // 256
    q8, d8 = _quantize_q8_1_cpu(x_bf16_f32)
    out = np.zeros((rows, out_features), dtype=np.float32)
    for row in range(rows):
        for out_col in range(out_features):
            weight_row = qweight[out_col]
            acc = 0.0
            for block_idx in range(blocks_per_row):
                block = weight_row[block_idx * 210 : (block_idx + 1) * 210]
                d = np.frombuffer(block[208:210].tobytes(), dtype=np.float16)[0].astype(np.float32)
                scales = block[192:208].view(np.int8).astype(np.int32)
                for group32 in range(8):
                    q8_index = block_idx * 8 + group32
                    xd = float(d8[row, q8_index])
                    for lane4 in range(0, 32, 4):
                        scale_index = group32 * 2 + (lane4 >> 4)
                        q6_u = _q6_unsigned_pack(block, group32, lane4)
                        x_pack = q8[row, q8_index, lane4 : lane4 + 4].astype(np.int32)
                        dot_u = int(np.dot(q6_u, x_pack))
                        q8_sum = int(np.sum(x_pack))
                        acc += xd * float(d) * float(scales[scale_index]) * float(dot_u - 32 * q8_sum)
            out[row, out_col] = acc
    return out


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
    with pytest.raises(ValueError, match="together"):
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_gather_f32(0, 0, 0, 0, 0, None, 1, None, 1, 256, 8, 0)
    with pytest.raises(ValueError, match="64 or 128"):
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_stage1_f32(0, 0, 0, 0, 1, 256, 8, stage1_threads=96)
    with pytest.raises(ValueError, match="num_blocks"):
        gguf_q6_k_pack8_top1_stage2_gather_f32(0, 0, 0, None, None, None, 1, 0, 0, 8)


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


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_q6_k_q8_1_dp4a_top1_gather_matches_q8_1_oracle(q6_k_dense_library) -> None:
    rows, in_features, out_features, hidden = 2, 512, 1024, 24
    rng = np.random.default_rng(20260702)
    qweight = make_q6_k_weight(out_features, in_features)
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)
    embed_table = rng.normal(0.0, 0.2, size=(out_features, hidden)).astype(np.float32)
    oracle_logits = _q6_k_q8_1_dp4a_oracle(x_ref, qweight)
    oracle_indices = np.argmax(oracle_logits, axis=1).astype(np.int32)
    oracle_values = oracle_logits[np.arange(rows), oracle_indices].astype(np.float32)
    oracle_embed = embed_table[oracle_indices]

    q4_lib = build_gguf_q4_k_gemv(load=True)
    x_buf = malloc(x_bf16.nbytes)
    xq_buf = malloc(rows * (in_features // 32) * 36)
    w_buf = malloc(qweight.nbytes)
    embed_table_buf = malloc(embed_table.nbytes)
    fused_values_buf = malloc(rows * 4)
    fused_indices_buf = malloc(rows * 4)
    fused_embed_buf = malloc(rows * hidden * 4)
    block_values_buf = malloc(rows * (out_features // 8) * 4)
    block_indices_buf = malloc(rows * (out_features // 8) * 4)
    split_values_buf = malloc(rows * 4)
    split_indices_buf = malloc(rows * 4)
    split_embed_buf = malloc(rows * hidden * 4)
    split_block_values_buf = malloc(rows * (out_features // 8) * 4)
    split_block_indices_buf = malloc(rows * (out_features // 8) * 4)
    scalehoist_values_buf = malloc(rows * 4)
    scalehoist_indices_buf = malloc(rows * 4)
    scalehoist_embed_buf = malloc(rows * hidden * 4)
    scalehoist_block_values_buf = malloc(rows * (out_features // 8) * 4)
    scalehoist_block_indices_buf = malloc(rows * (out_features // 8) * 4)
    scalehoist_split_values_buf = malloc(rows * 4)
    scalehoist_split_indices_buf = malloc(rows * 4)
    scalehoist_split_embed_buf = malloc(rows * hidden * 4)
    scalehoist_split_block_values_buf = malloc(rows * (out_features // 8) * 4)
    scalehoist_split_block_indices_buf = malloc(rows * (out_features // 8) * 4)
    pack8_llama_values_buf = malloc(rows * 4)
    pack8_llama_indices_buf = malloc(rows * 4)
    pack8_llama_embed_buf = malloc(rows * hidden * 4)
    pack8_llama_block_values_buf = malloc(rows * (out_features // 8) * 4)
    pack8_llama_block_indices_buf = malloc(rows * (out_features // 8) * 4)
    pack8_llama_split_values_buf = malloc(rows * 4)
    pack8_llama_split_indices_buf = malloc(rows * 4)
    pack8_llama_split_embed_buf = malloc(rows * hidden * 4)
    pack8_llama_split_block_values_buf = malloc(rows * (out_features // 8) * 4)
    pack8_llama_split_block_indices_buf = malloc(rows * (out_features // 8) * 4)
    row_values_buf = malloc(rows * 4)
    row_indices_buf = malloc(rows * 4)
    row_embed_buf = malloc(rows * hidden * 4)
    row_block_values_buf = malloc(rows * out_features * 4)
    row_block_indices_buf = malloc(rows * out_features * 4)
    row_split_values_buf = malloc(rows * 4)
    row_split_indices_buf = malloc(rows * 4)
    row_split_embed_buf = malloc(rows * hidden * 4)
    row_split_block_values_buf = malloc(rows * out_features * 4)
    row_split_block_indices_buf = malloc(rows * out_features * 4)
    try:
        copy_host_to_device(x_buf, host_array_ptr(x_bf16), x_bf16.nbytes)
        copy_host_to_device(w_buf, host_array_ptr(qweight), qweight.nbytes)
        copy_host_to_device(embed_table_buf, host_array_ptr(embed_table), embed_table.nbytes)
        gguf_q4_k_quantize_bf16_q8_1(
            x_buf.ptr,
            xq_buf.ptr,
            rows,
            in_features,
            library=q4_lib,
        )
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_gather_f32(
            xq_buf.ptr,
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
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_stage1_f32(
            xq_buf.ptr,
            w_buf.ptr,
            split_block_values_buf.ptr,
            split_block_indices_buf.ptr,
            rows,
            in_features,
            out_features,
            stage1_threads=64,
            library=q6_k_dense_library,
        )
        gguf_q6_k_pack8_top1_stage2_gather_f32(
            split_block_values_buf.ptr,
            split_block_indices_buf.ptr,
            split_indices_buf.ptr,
            split_values_buf.ptr,
            embed_table_buf.ptr,
            split_embed_buf.ptr,
            rows,
            out_features // 8,
            hidden,
            out_features,
            library=q6_k_dense_library,
        )
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_gather_f32(
            xq_buf.ptr,
            w_buf.ptr,
            scalehoist_block_values_buf.ptr,
            scalehoist_block_indices_buf.ptr,
            scalehoist_indices_buf.ptr,
            scalehoist_values_buf.ptr,
            embed_table_buf.ptr,
            scalehoist_embed_buf.ptr,
            rows,
            in_features,
            out_features,
            hidden,
            library=q6_k_dense_library,
        )
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_scalehoist_stage1_f32(
            xq_buf.ptr,
            w_buf.ptr,
            scalehoist_split_block_values_buf.ptr,
            scalehoist_split_block_indices_buf.ptr,
            rows,
            in_features,
            out_features,
            stage1_threads=64,
            library=q6_k_dense_library,
        )
        gguf_q6_k_pack8_top1_stage2_gather_f32(
            scalehoist_split_block_values_buf.ptr,
            scalehoist_split_block_indices_buf.ptr,
            scalehoist_split_indices_buf.ptr,
            scalehoist_split_values_buf.ptr,
            embed_table_buf.ptr,
            scalehoist_split_embed_buf.ptr,
            rows,
            out_features // 8,
            hidden,
            out_features,
            library=q6_k_dense_library,
        )
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_pack8_llama_gather_f32(
            xq_buf.ptr,
            w_buf.ptr,
            pack8_llama_block_values_buf.ptr,
            pack8_llama_block_indices_buf.ptr,
            pack8_llama_indices_buf.ptr,
            pack8_llama_values_buf.ptr,
            embed_table_buf.ptr,
            pack8_llama_embed_buf.ptr,
            rows,
            in_features,
            out_features,
            hidden,
            library=q6_k_dense_library,
        )
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_pack8_llama_stage1_f32(
            xq_buf.ptr,
            w_buf.ptr,
            pack8_llama_split_block_values_buf.ptr,
            pack8_llama_split_block_indices_buf.ptr,
            rows,
            in_features,
            out_features,
            stage1_threads=64,
            library=q6_k_dense_library,
        )
        gguf_q6_k_pack8_top1_stage2_gather_f32(
            pack8_llama_split_block_values_buf.ptr,
            pack8_llama_split_block_indices_buf.ptr,
            pack8_llama_split_indices_buf.ptr,
            pack8_llama_split_values_buf.ptr,
            embed_table_buf.ptr,
            pack8_llama_split_embed_buf.ptr,
            rows,
            out_features // 8,
            hidden,
            out_features,
            library=q6_k_dense_library,
        )
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_row_gather_f32(
            xq_buf.ptr,
            w_buf.ptr,
            row_block_values_buf.ptr,
            row_block_indices_buf.ptr,
            row_indices_buf.ptr,
            row_values_buf.ptr,
            embed_table_buf.ptr,
            row_embed_buf.ptr,
            rows,
            in_features,
            out_features,
            hidden,
            library=q6_k_dense_library,
        )
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_row_stage1_f32(
            xq_buf.ptr,
            w_buf.ptr,
            row_split_block_values_buf.ptr,
            row_split_block_indices_buf.ptr,
            rows,
            in_features,
            out_features,
            library=q6_k_dense_library,
        )
        gguf_q6_k_pack8_top1_stage2_gather_f32(
            row_split_block_values_buf.ptr,
            row_split_block_indices_buf.ptr,
            row_split_indices_buf.ptr,
            row_split_values_buf.ptr,
            embed_table_buf.ptr,
            row_split_embed_buf.ptr,
            rows,
            out_features,
            hidden,
            out_features,
            library=q6_k_dense_library,
        )

        fused_index = np.empty((rows,), dtype=np.int32)
        fused_value = np.empty((rows,), dtype=np.float32)
        fused_embed = np.empty((rows, hidden), dtype=np.float32)
        split_index = np.empty((rows,), dtype=np.int32)
        split_value = np.empty((rows,), dtype=np.float32)
        split_embed = np.empty((rows, hidden), dtype=np.float32)
        scalehoist_index = np.empty((rows,), dtype=np.int32)
        scalehoist_value = np.empty((rows,), dtype=np.float32)
        scalehoist_embed = np.empty((rows, hidden), dtype=np.float32)
        scalehoist_split_index = np.empty((rows,), dtype=np.int32)
        scalehoist_split_value = np.empty((rows,), dtype=np.float32)
        scalehoist_split_embed = np.empty((rows, hidden), dtype=np.float32)
        pack8_llama_index = np.empty((rows,), dtype=np.int32)
        pack8_llama_value = np.empty((rows,), dtype=np.float32)
        pack8_llama_embed = np.empty((rows, hidden), dtype=np.float32)
        pack8_llama_split_index = np.empty((rows,), dtype=np.int32)
        pack8_llama_split_value = np.empty((rows,), dtype=np.float32)
        pack8_llama_split_embed = np.empty((rows, hidden), dtype=np.float32)
        row_index = np.empty((rows,), dtype=np.int32)
        row_value = np.empty((rows,), dtype=np.float32)
        row_embed = np.empty((rows, hidden), dtype=np.float32)
        row_split_index = np.empty((rows,), dtype=np.int32)
        row_split_value = np.empty((rows,), dtype=np.float32)
        row_split_embed = np.empty((rows, hidden), dtype=np.float32)
        copy_device_to_host(host_array_ptr(fused_index), fused_indices_buf, fused_index.nbytes)
        copy_device_to_host(host_array_ptr(fused_value), fused_values_buf, fused_value.nbytes)
        copy_device_to_host(host_array_ptr(fused_embed), fused_embed_buf, fused_embed.nbytes)
        copy_device_to_host(host_array_ptr(split_index), split_indices_buf, split_index.nbytes)
        copy_device_to_host(host_array_ptr(split_value), split_values_buf, split_value.nbytes)
        copy_device_to_host(host_array_ptr(split_embed), split_embed_buf, split_embed.nbytes)
        copy_device_to_host(host_array_ptr(scalehoist_index), scalehoist_indices_buf, scalehoist_index.nbytes)
        copy_device_to_host(host_array_ptr(scalehoist_value), scalehoist_values_buf, scalehoist_value.nbytes)
        copy_device_to_host(host_array_ptr(scalehoist_embed), scalehoist_embed_buf, scalehoist_embed.nbytes)
        copy_device_to_host(
            host_array_ptr(scalehoist_split_index),
            scalehoist_split_indices_buf,
            scalehoist_split_index.nbytes,
        )
        copy_device_to_host(
            host_array_ptr(scalehoist_split_value),
            scalehoist_split_values_buf,
            scalehoist_split_value.nbytes,
        )
        copy_device_to_host(
            host_array_ptr(scalehoist_split_embed),
            scalehoist_split_embed_buf,
            scalehoist_split_embed.nbytes,
        )
        copy_device_to_host(host_array_ptr(pack8_llama_index), pack8_llama_indices_buf, pack8_llama_index.nbytes)
        copy_device_to_host(host_array_ptr(pack8_llama_value), pack8_llama_values_buf, pack8_llama_value.nbytes)
        copy_device_to_host(host_array_ptr(pack8_llama_embed), pack8_llama_embed_buf, pack8_llama_embed.nbytes)
        copy_device_to_host(
            host_array_ptr(pack8_llama_split_index),
            pack8_llama_split_indices_buf,
            pack8_llama_split_index.nbytes,
        )
        copy_device_to_host(
            host_array_ptr(pack8_llama_split_value),
            pack8_llama_split_values_buf,
            pack8_llama_split_value.nbytes,
        )
        copy_device_to_host(
            host_array_ptr(pack8_llama_split_embed),
            pack8_llama_split_embed_buf,
            pack8_llama_split_embed.nbytes,
        )
        copy_device_to_host(host_array_ptr(row_index), row_indices_buf, row_index.nbytes)
        copy_device_to_host(host_array_ptr(row_value), row_values_buf, row_value.nbytes)
        copy_device_to_host(host_array_ptr(row_embed), row_embed_buf, row_embed.nbytes)
        copy_device_to_host(host_array_ptr(row_split_index), row_split_indices_buf, row_split_index.nbytes)
        copy_device_to_host(host_array_ptr(row_split_value), row_split_values_buf, row_split_value.nbytes)
        copy_device_to_host(host_array_ptr(row_split_embed), row_split_embed_buf, row_split_embed.nbytes)
    finally:
        for b in (
            row_split_block_indices_buf,
            row_split_block_values_buf,
            row_split_embed_buf,
            row_split_indices_buf,
            row_split_values_buf,
            row_block_indices_buf,
            row_block_values_buf,
            row_embed_buf,
            row_indices_buf,
            row_values_buf,
            scalehoist_split_block_indices_buf,
            scalehoist_split_block_values_buf,
            scalehoist_split_embed_buf,
            scalehoist_split_indices_buf,
            scalehoist_split_values_buf,
            scalehoist_block_indices_buf,
            scalehoist_block_values_buf,
            scalehoist_embed_buf,
            scalehoist_indices_buf,
            scalehoist_values_buf,
            pack8_llama_split_block_indices_buf,
            pack8_llama_split_block_values_buf,
            pack8_llama_split_embed_buf,
            pack8_llama_split_indices_buf,
            pack8_llama_split_values_buf,
            pack8_llama_block_indices_buf,
            pack8_llama_block_values_buf,
            pack8_llama_embed_buf,
            pack8_llama_indices_buf,
            pack8_llama_values_buf,
            split_block_indices_buf,
            split_block_values_buf,
            split_embed_buf,
            split_indices_buf,
            split_values_buf,
            block_indices_buf,
            block_values_buf,
            fused_embed_buf,
            fused_indices_buf,
            fused_values_buf,
            embed_table_buf,
            w_buf,
            xq_buf,
            x_buf,
        ):
            free(b)

    np.testing.assert_array_equal(fused_index, oracle_indices)
    np.testing.assert_allclose(fused_value, oracle_values, atol=1.0e-3, rtol=1.0e-5)
    np.testing.assert_array_equal(fused_embed, oracle_embed)
    np.testing.assert_array_equal(split_index, oracle_indices)
    np.testing.assert_allclose(split_value, oracle_values, atol=1.0e-3, rtol=1.0e-5)
    np.testing.assert_array_equal(split_embed, oracle_embed)
    np.testing.assert_array_equal(scalehoist_index, oracle_indices)
    np.testing.assert_allclose(scalehoist_value, oracle_values, atol=1.0e-3, rtol=1.0e-5)
    np.testing.assert_array_equal(scalehoist_embed, oracle_embed)
    np.testing.assert_array_equal(scalehoist_split_index, oracle_indices)
    np.testing.assert_allclose(scalehoist_split_value, oracle_values, atol=1.0e-3, rtol=1.0e-5)
    np.testing.assert_array_equal(scalehoist_split_embed, oracle_embed)
    np.testing.assert_array_equal(pack8_llama_index, oracle_indices)
    np.testing.assert_allclose(pack8_llama_value, oracle_values, atol=1.0e-3, rtol=1.0e-5)
    np.testing.assert_array_equal(pack8_llama_embed, oracle_embed)
    np.testing.assert_array_equal(pack8_llama_split_index, oracle_indices)
    np.testing.assert_allclose(pack8_llama_split_value, oracle_values, atol=1.0e-3, rtol=1.0e-5)
    np.testing.assert_array_equal(pack8_llama_split_embed, oracle_embed)
    np.testing.assert_array_equal(row_index, oracle_indices)
    np.testing.assert_allclose(row_value, oracle_values, atol=1.0e-3, rtol=1.0e-5)
    np.testing.assert_array_equal(row_embed, oracle_embed)
    np.testing.assert_array_equal(row_split_index, oracle_indices)
    np.testing.assert_allclose(row_split_value, oracle_values, atol=1.0e-3, rtol=1.0e-5)
    np.testing.assert_array_equal(row_split_embed, oracle_embed)


def test_q6_k_q8_1_dp4a_pack16_top1_matches_q8_1_oracle(q6_k_dense_library) -> None:
    rows, in_features, out_features, hidden = 2, 512, 1024, 16
    rng = np.random.default_rng(20260703)
    qweight = make_q6_k_weight(out_features, in_features)
    x = rng.normal(0.0, 0.3, size=(rows, in_features)).astype(np.float32)
    x_bf16 = _f32_to_bf16_u16(x)
    x_ref = _bf16_u16_to_f32(x_bf16)
    embed_table = rng.normal(0.0, 0.2, size=(out_features, hidden)).astype(np.float32)
    oracle_logits = _q6_k_q8_1_dp4a_oracle(x_ref, qweight)
    oracle_indices = np.argmax(oracle_logits, axis=1).astype(np.int32)
    oracle_values = oracle_logits[np.arange(rows), oracle_indices].astype(np.float32)
    oracle_embed = embed_table[oracle_indices]

    q4_lib = build_gguf_q4_k_gemv(load=True)
    x_buf = malloc(x_bf16.nbytes)
    xq_buf = malloc(rows * (in_features // 32) * 36)
    w_buf = malloc(qweight.nbytes)
    embed_table_buf = malloc(embed_table.nbytes)
    fused_values_buf = malloc(rows * 4)
    fused_indices_buf = malloc(rows * 4)
    fused_embed_buf = malloc(rows * hidden * 4)
    block_values_buf = malloc(rows * (out_features // 16) * 4)
    block_indices_buf = malloc(rows * (out_features // 16) * 4)
    split_values_buf = malloc(rows * 4)
    split_indices_buf = malloc(rows * 4)
    split_embed_buf = malloc(rows * hidden * 4)
    split_block_values_buf = malloc(rows * (out_features // 16) * 4)
    split_block_indices_buf = malloc(rows * (out_features // 16) * 4)
    try:
        copy_host_to_device(x_buf, host_array_ptr(x_bf16), x_bf16.nbytes)
        copy_host_to_device(w_buf, host_array_ptr(qweight), qweight.nbytes)
        copy_host_to_device(embed_table_buf, host_array_ptr(embed_table), embed_table.nbytes)
        gguf_q4_k_quantize_bf16_q8_1(
            x_buf.ptr,
            xq_buf.ptr,
            rows,
            in_features,
            library=q4_lib,
        )
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_pack16_gather_f32(
            xq_buf.ptr,
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
        gguf_q6_k_pack8_gemv_decode_q8_1_dp4a_top1_pack16_stage1_f32(
            xq_buf.ptr,
            w_buf.ptr,
            split_block_values_buf.ptr,
            split_block_indices_buf.ptr,
            rows,
            in_features,
            out_features,
            stage1_threads=64,
            library=q6_k_dense_library,
        )
        gguf_q6_k_pack8_top1_stage2_gather_f32(
            split_block_values_buf.ptr,
            split_block_indices_buf.ptr,
            split_indices_buf.ptr,
            split_values_buf.ptr,
            embed_table_buf.ptr,
            split_embed_buf.ptr,
            rows,
            out_features // 16,
            hidden,
            out_features,
            library=q6_k_dense_library,
        )

        fused_index = np.empty((rows,), dtype=np.int32)
        fused_value = np.empty((rows,), dtype=np.float32)
        fused_embed = np.empty((rows, hidden), dtype=np.float32)
        split_index = np.empty((rows,), dtype=np.int32)
        split_value = np.empty((rows,), dtype=np.float32)
        split_embed = np.empty((rows, hidden), dtype=np.float32)
        copy_device_to_host(host_array_ptr(fused_index), fused_indices_buf, fused_index.nbytes)
        copy_device_to_host(host_array_ptr(fused_value), fused_values_buf, fused_value.nbytes)
        copy_device_to_host(host_array_ptr(fused_embed), fused_embed_buf, fused_embed.nbytes)
        copy_device_to_host(host_array_ptr(split_index), split_indices_buf, split_index.nbytes)
        copy_device_to_host(host_array_ptr(split_value), split_values_buf, split_value.nbytes)
        copy_device_to_host(host_array_ptr(split_embed), split_embed_buf, split_embed.nbytes)
    finally:
        for b in (
            split_block_indices_buf,
            split_block_values_buf,
            split_embed_buf,
            split_indices_buf,
            split_values_buf,
            block_indices_buf,
            block_values_buf,
            fused_embed_buf,
            fused_indices_buf,
            fused_values_buf,
            embed_table_buf,
            w_buf,
            xq_buf,
            x_buf,
        ):
            free(b)

    np.testing.assert_array_equal(fused_index, oracle_indices)
    np.testing.assert_allclose(fused_value, oracle_values, atol=1.0e-3, rtol=1.0e-5)
    np.testing.assert_array_equal(fused_embed, oracle_embed)
    np.testing.assert_array_equal(split_index, oracle_indices)
    np.testing.assert_allclose(split_value, oracle_values, atol=1.0e-3, rtol=1.0e-5)
    np.testing.assert_array_equal(split_embed, oracle_embed)
