"""POC gate for raw GGUF Q5_K/Q6_K selected-pack8 q8_1 + sudot4 GEMV."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    build_gguf_k_gemv,
    gguf_q5_k_selected_pack8_gemv_bf16_bf16_out,
    gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q6_k_selected_pack8_gemv_bf16_bf16_out,
    gguf_q6_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out,
    register_gguf_k_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_quantize_bf16_q8_1,
)
from hipengine.kernels.registry import resolve
from tests._gguf_synthetic_weights import make_q5_k_weight, make_q6_k_weight

QK_K = 256
Q8_1_BLOCK = 32
Q8_1_BLOCK_BYTES = 36
Q5_K_BLOCK_BYTES = 176
Q6_K_BLOCK_BYTES = 210


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32)
    lsb = (u32 >> 16) & 1
    return ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16).reshape(f32.shape)


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(bits.shape).copy()


def _round_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def _quantize_q8_1_cpu(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, in_features = x.shape
    blocks = in_features // Q8_1_BLOCK
    q = np.empty((rows, blocks, Q8_1_BLOCK), dtype=np.int8)
    d = np.empty((rows, blocks), dtype=np.float32)
    for row in range(rows):
        for block_idx in range(blocks):
            chunk = x[row, block_idx * Q8_1_BLOCK : (block_idx + 1) * Q8_1_BLOCK].astype(np.float32)
            amax = float(np.max(np.abs(chunk)))
            scale = 0.0 if amax == 0.0 else amax / 127.0
            d[row, block_idx] = np.float16(scale).astype(np.float32)
            if amax == 0.0:
                q[row, block_idx] = 0
            else:
                q[row, block_idx] = np.clip(_round_away_from_zero(chunk / scale), -128, 127).astype(np.int8)
    return q, d


def _q4_k_scale(scales: np.ndarray, subblock: int) -> int:
    if subblock < 4:
        return int(scales[subblock] & 0x3F)
    idx = subblock - 4
    return int((scales[8 + idx] & 0x0F) | ((scales[idx] >> 2) & 0x30))


def _q4_k_min(scales: np.ndarray, subblock: int) -> int:
    if subblock < 4:
        return int(scales[4 + subblock] & 0x3F)
    idx = subblock - 4
    return int((scales[8 + idx] >> 4) | ((scales[4 + idx] >> 2) & 0x30))


def _q5_value(block: np.ndarray, k_in_block: int) -> int:
    subblock = k_in_block >> 5
    lane = k_in_block & 31
    qh = block[16:48]
    qs = block[48:176]
    packed = int(qs[(subblock >> 1) * 32 + lane])
    low = (packed >> 4) if (subblock & 1) else (packed & 0x0F)
    high = (int(qh[lane]) >> subblock) & 0x01
    return int(low | (high << 4))


def _q6_value(block: np.ndarray, k_in_block: int) -> int:
    group32 = k_in_block >> 5
    lane = k_in_block & 31
    ql = block[:128]
    qh = block[128:192]
    base64 = 64 if group32 >= 4 else 0
    ql_group = group32 & 1
    low_byte = int(ql[base64 + ql_group * 32 + lane])
    low = (low_byte & 0x0F) if ((group32 & 2) == 0) else (low_byte >> 4)
    qh_base = 32 if group32 >= 4 else 0
    high = (int(qh[qh_base + lane]) >> (2 * (group32 & 3))) & 0x03
    return int(low | (high << 4)) - 32


def _selected_pack8_q8_1_cpu(
    quant: str,
    x_f32: np.ndarray,
    selected: np.ndarray,
    qweight: np.ndarray,
) -> np.ndarray:
    q8, d8 = _quantize_q8_1_cpu(x_f32)
    x_rows, in_features = x_f32.shape
    rows = int(selected.size)
    num_experts, out_features = qweight.shape[:2]
    lanes_per_x_row = rows // x_rows
    out = np.zeros((rows, out_features), dtype=np.float32)
    block_bytes = Q5_K_BLOCK_BYTES if quant == "q5" else Q6_K_BLOCK_BYTES
    for row, expert in enumerate(selected.astype(np.int64)):
        assert 0 <= expert < num_experts
        x_row = 0 if x_rows == 1 else row // lanes_per_x_row
        for out_col in range(out_features):
            acc = 0.0
            for block_idx in range(in_features // QK_K):
                block = qweight[expert, out_col, block_idx * block_bytes : (block_idx + 1) * block_bytes]
                if quant == "q5":
                    d = block[0:2].view(np.float16).astype(np.float32)[0]
                    dmin = block[2:4].view(np.float16).astype(np.float32)[0]
                    scales = block[4:16]
                else:
                    d = block[208:210].view(np.float16).astype(np.float32)[0]
                    scales_i8 = block[192:208].view(np.int8)
                for k in range(QK_K):
                    q8_idx = block_idx * 8 + (k >> 5)
                    xv = float(d8[x_row, q8_idx]) * float(q8[x_row, q8_idx, k & 31])
                    if quant == "q5":
                        sb = k >> 5
                        wv = float(d) * _q4_k_scale(scales, sb) * _q5_value(block, k)
                        wv -= float(dmin) * _q4_k_min(scales, sb)
                    else:
                        wv = float(d) * int(scales_i8[k >> 4]) * _q6_value(block, k)
                    acc += xv * wv
            out[row, out_col] = acc
    return out


def _softmax_kl(ref: np.ndarray, cand: np.ndarray) -> tuple[float, float]:
    ref64 = ref.astype(np.float64)
    cand64 = cand.astype(np.float64)

    def logsm(x: np.ndarray) -> np.ndarray:
        shifted = x - x.max(axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    log_ref = logsm(ref64)
    log_cand = logsm(cand64)
    row_kl = np.sum(np.exp(log_ref) * (log_ref - log_cand), axis=-1)
    return float(np.mean(row_kl)), float(np.max(row_kl))


def _top1(ref: np.ndarray, cand: np.ndarray) -> float:
    return float(np.mean(ref.argmax(axis=-1) == cand.argmax(axis=-1)))


def _fixture(quant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows = 2
    rows = 4
    num_experts = 3
    in_features = 512
    out_features = 24
    rng = np.random.default_rng(20260627 + (5 if quant == "q5" else 6))
    x_f32 = (rng.standard_normal((x_rows, in_features)).astype(np.float32) * 0.1) + 0.002
    x_bits = _bf16_bits(x_f32)
    make_weight = make_q5_k_weight if quant == "q5" else make_q6_k_weight
    base = make_weight(out_features, in_features)
    weights = np.ascontiguousarray(
        np.stack([np.roll(base, shift=expert + 1, axis=0) for expert in range(num_experts)], axis=0)
    )
    selected = np.asarray([0, 2, 1, 2], dtype=np.int64)
    assert rows == selected.size
    return x_bits, selected, weights


def _run(wrapper, x_bits, selected, qweight, *, prequantized: bool) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc

    runtime = get_hip_runtime()
    library = build_gguf_k_gemv(load=True)
    q4_library = build_gguf_q4_k_gemv(load=True)
    x_rows, in_features = x_bits.shape
    rows = int(selected.size)
    num_experts, out_features = qweight.shape[:2]
    out = np.zeros((rows, out_features), dtype=np.uint16)
    bufs = []
    try:
        x_buf = malloc(x_bits.nbytes, runtime=runtime)
        selected_buf = malloc(selected.nbytes, runtime=runtime)
        qweight_buf = malloc(qweight.nbytes, runtime=runtime)
        out_buf = malloc(out.nbytes, runtime=runtime)
        bufs.extend((x_buf, selected_buf, qweight_buf, out_buf))
        copy_host_to_device(x_buf, host_array_ptr(np.ascontiguousarray(x_bits)), runtime=runtime)
        copy_host_to_device(selected_buf, host_array_ptr(np.ascontiguousarray(selected)), runtime=runtime)
        copy_host_to_device(qweight_buf, host_array_ptr(np.ascontiguousarray(qweight)), runtime=runtime)
        x_arg = x_buf.ptr
        if prequantized:
            xq_buf = malloc(x_rows * (in_features // Q8_1_BLOCK) * Q8_1_BLOCK_BYTES, runtime=runtime)
            bufs.append(xq_buf)
            gguf_q4_k_quantize_bf16_q8_1(
                x_buf.ptr, xq_buf.ptr, x_rows, in_features, library=q4_library, runtime=runtime
            )
            x_arg = xq_buf.ptr
        wrapper(
            x_arg,
            selected_buf.ptr,
            qweight_buf.ptr,
            out_buf.ptr,
            x_rows,
            rows,
            num_experts,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_buf, runtime=runtime)
        return out
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)


def test_selected_pack8_dp4a_registry_and_contract() -> None:
    register_gguf_k_gemv_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q5_k",
            variant="selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out",
        )
        is gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q6_k",
            variant="selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out",
        )
        is gguf_q6_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out
    )
    with pytest.raises(ValueError, match="divisible"):
        gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out(1, 2, 3, 4, 1, 1, 1, 256, 10)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "quant,float_wrapper,dp4a_wrapper",
    [
        ("q5", gguf_q5_k_selected_pack8_gemv_bf16_bf16_out, gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out),
        ("q6", gguf_q6_k_selected_pack8_gemv_bf16_bf16_out, gguf_q6_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out),
    ],
)
def test_selected_pack8_dp4a_matches_cpu_q8_1_oracle_and_float_gate(quant, float_wrapper, dp4a_wrapper) -> None:
    x_bits, selected, qweight = _fixture(quant)
    ref_bits = _run(float_wrapper, x_bits, selected, qweight, prequantized=False)
    got_bits = _run(dp4a_wrapper, x_bits, selected, qweight, prequantized=True)

    cpu = _selected_pack8_q8_1_cpu(quant, _bf16_to_f32(x_bits), selected, qweight)
    got = _bf16_to_f32(got_bits)
    np.testing.assert_allclose(got, cpu, rtol=2e-2, atol=2e-2)

    ref = _bf16_to_f32(ref_bits)
    kl_mean, kl_max = _softmax_kl(ref, got)
    assert kl_mean <= 0.05
    assert kl_max <= 0.10
    assert _top1(ref, got) >= 0.90
