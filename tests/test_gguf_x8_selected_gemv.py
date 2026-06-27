"""Correctness gates for GGUF X8 selected-down q8_1+sudot4 kernels."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_quantize_bf16_q8_1,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_x8_selected_gemv import (
    build_gguf_x8_selected_gemv,
    gguf_q4_k_x8_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q5_k_x8_selected_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out,
    gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out,
    gguf_q6_k_x8_selected_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out,
    register_gguf_x8_selected_gemv_kernels,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_x8 import repack_gguf_q4_k_x8, repack_gguf_q5_k_x8, repack_gguf_q6_k_x8
from tests._gguf_synthetic_weights import make_q4_k_weight, make_q5_k_weight, make_q6_k_weight

QK_K = 256
Q8_1_BLOCK = 32
Q8_1_BLOCK_BYTES = 36
Q5_K_BLOCK_BYTES = 176
Q6_K_BLOCK_BYTES = 210
Q4_K_BLOCK_BYTES = 144


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


def _q4_value(block: np.ndarray, k_in_block: int) -> int:
    subblock = k_in_block >> 5
    lane = k_in_block & 31
    qs = block[16:144]
    packed = int(qs[(subblock >> 1) * 32 + lane])
    return (packed >> 4) if (subblock & 1) else (packed & 0x0F)


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


def _q8_oracle(
    quant: str,
    x_f32: np.ndarray,
    x_row_for_output: np.ndarray,
    expert_for_output: np.ndarray,
    qweight: np.ndarray,
) -> np.ndarray:
    q8, d8 = _quantize_q8_1_cpu(x_f32)
    rows = int(expert_for_output.size)
    out_features = int(qweight.shape[1])
    in_features = int(x_f32.shape[1])
    block_bytes = {"q4": Q4_K_BLOCK_BYTES, "q5": Q5_K_BLOCK_BYTES, "q6": Q6_K_BLOCK_BYTES}[quant]
    out = np.zeros((rows, out_features), dtype=np.float32)
    for row in range(rows):
        expert = int(expert_for_output[row])
        x_row = int(x_row_for_output[row])
        for out_col in range(out_features):
            acc = 0.0
            for block_idx in range(in_features // QK_K):
                block = qweight[expert, out_col, block_idx * block_bytes : (block_idx + 1) * block_bytes]
                if quant == "q4":
                    d = block[0:2].view(np.float16).astype(np.float32)[0]
                    dmin = block[2:4].view(np.float16).astype(np.float32)[0]
                    scales = block[4:16]
                elif quant == "q5":
                    d = block[0:2].view(np.float16).astype(np.float32)[0]
                    dmin = block[2:4].view(np.float16).astype(np.float32)[0]
                    scales = block[4:16]
                else:
                    d = block[208:210].view(np.float16).astype(np.float32)[0]
                    scales_i8 = block[192:208].view(np.int8)
                for k in range(QK_K):
                    q8_idx = block_idx * 8 + (k >> 5)
                    xv = float(d8[x_row, q8_idx]) * float(q8[x_row, q8_idx, k & 31])
                    if quant == "q4":
                        sb = k >> 5
                        wv = float(d) * _q4_k_scale(scales, sb) * _q4_value(block, k)
                        wv -= float(dmin) * _q4_k_min(scales, sb)
                    elif quant == "q5":
                        sb = k >> 5
                        wv = float(d) * _q4_k_scale(scales, sb) * _q5_value(block, k)
                        wv -= float(dmin) * _q4_k_min(scales, sb)
                    else:
                        wv = float(d) * int(scales_i8[k >> 4]) * _q6_value(block, k)
                    acc += xv * wv
            out[row, out_col] = acc
    return out


def _exact_oracle(
    quant: str,
    x_f32: np.ndarray,
    x_row_for_output: np.ndarray,
    expert_for_output: np.ndarray,
    qweight: np.ndarray,
) -> np.ndarray:
    qtype = {
        "q4": GGMLQuantizationType.Q4_K,
        "q5": GGMLQuantizationType.Q5_K,
        "q6": GGMLQuantizationType.Q6_K,
    }[quant]
    rows = int(expert_for_output.size)
    out = np.zeros((rows, qweight.shape[1]), dtype=np.float32)
    for row in range(rows):
        out[row] = gguf_quant_gemv(
            x_f32[int(x_row_for_output[row]) : int(x_row_for_output[row]) + 1],
            qweight[int(expert_for_output[row])],
            qtype,
        )[0]
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


def _weights(quant: str, *, out_features: int = 24, in_features: int = 512, experts: int = 4) -> np.ndarray:
    make_weight = {"q4": make_q4_k_weight, "q5": make_q5_k_weight, "q6": make_q6_k_weight}[quant]
    base = make_weight(out_features, in_features)
    return np.ascontiguousarray(
        np.stack([np.roll(base, shift=expert + 1, axis=0) for expert in range(experts)], axis=0)
    )


def _tiles(quant: str, qweight: np.ndarray) -> np.ndarray:
    if quant == "q4":
        return repack_gguf_q4_k_x8(qweight).tiles
    return repack_gguf_q5_k_x8(qweight).tiles if quant == "q5" else repack_gguf_q6_k_x8(qweight).tiles


def _run_direct(wrapper, x_bits: np.ndarray, selected: np.ndarray, tiles: np.ndarray) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    x8_library = build_gguf_x8_selected_gemv(load=True)
    q4_library = build_gguf_q4_k_gemv(load=True)
    x_rows, in_features = x_bits.shape
    rows = int(selected.size)
    num_experts = int(tiles.shape[0])
    out_features = int(tiles.shape[1] * 8)
    out = np.zeros((rows, out_features), dtype=np.uint16)
    bufs = []
    try:
        x_buf = malloc(x_bits.nbytes, runtime=runtime)
        selected_buf = malloc(selected.nbytes, runtime=runtime)
        tiles_buf = malloc(tiles.nbytes, runtime=runtime)
        xq_buf = malloc(x_rows * (in_features // Q8_1_BLOCK) * Q8_1_BLOCK_BYTES, runtime=runtime)
        out_buf = malloc(out.nbytes, runtime=runtime)
        bufs.extend((x_buf, selected_buf, tiles_buf, xq_buf, out_buf))
        copy_host_to_device(x_buf, host_array_ptr(np.ascontiguousarray(x_bits)), runtime=runtime)
        copy_host_to_device(selected_buf, host_array_ptr(np.ascontiguousarray(selected)), runtime=runtime)
        copy_host_to_device(tiles_buf, host_array_ptr(np.ascontiguousarray(tiles)), runtime=runtime)
        gguf_q4_k_quantize_bf16_q8_1(
            x_buf.ptr, xq_buf.ptr, x_rows, in_features, library=q4_library, runtime=runtime
        )
        wrapper(
            xq_buf.ptr,
            selected_buf.ptr,
            tiles_buf.ptr,
            out_buf.ptr,
            x_rows,
            rows,
            num_experts,
            in_features,
            out_features,
            library=x8_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_buf, runtime=runtime)
        return out
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)


def _run_dual_direct(
    wrapper,
    x_bits: np.ndarray,
    selected: np.ndarray,
    tiles_a: np.ndarray,
    tiles_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    x8_library = build_gguf_x8_selected_gemv(load=True)
    q4_library = build_gguf_q4_k_gemv(load=True)
    x_rows, in_features = x_bits.shape
    rows = int(selected.size)
    num_experts = int(tiles_a.shape[0])
    out_features = int(tiles_a.shape[1] * 8)
    out_a = np.zeros((rows, out_features), dtype=np.uint16)
    out_b = np.zeros((rows, out_features), dtype=np.uint16)
    bufs = []
    try:
        x_buf = malloc(x_bits.nbytes, runtime=runtime)
        selected_buf = malloc(selected.nbytes, runtime=runtime)
        tiles_a_buf = malloc(tiles_a.nbytes, runtime=runtime)
        tiles_b_buf = malloc(tiles_b.nbytes, runtime=runtime)
        xq_buf = malloc(x_rows * (in_features // Q8_1_BLOCK) * Q8_1_BLOCK_BYTES, runtime=runtime)
        out_a_buf = malloc(out_a.nbytes, runtime=runtime)
        out_b_buf = malloc(out_b.nbytes, runtime=runtime)
        bufs.extend((x_buf, selected_buf, tiles_a_buf, tiles_b_buf, xq_buf, out_a_buf, out_b_buf))
        copy_host_to_device(x_buf, host_array_ptr(np.ascontiguousarray(x_bits)), runtime=runtime)
        copy_host_to_device(selected_buf, host_array_ptr(np.ascontiguousarray(selected)), runtime=runtime)
        copy_host_to_device(tiles_a_buf, host_array_ptr(np.ascontiguousarray(tiles_a)), runtime=runtime)
        copy_host_to_device(tiles_b_buf, host_array_ptr(np.ascontiguousarray(tiles_b)), runtime=runtime)
        gguf_q4_k_quantize_bf16_q8_1(
            x_buf.ptr, xq_buf.ptr, x_rows, in_features, library=q4_library, runtime=runtime
        )
        wrapper(
            xq_buf.ptr,
            selected_buf.ptr,
            tiles_a_buf.ptr,
            tiles_b_buf.ptr,
            out_a_buf.ptr,
            out_b_buf.ptr,
            x_rows,
            rows,
            num_experts,
            in_features,
            out_features,
            library=x8_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_a), out_a_buf, runtime=runtime)
        copy_device_to_host(host_array_ptr(out_b), out_b_buf, runtime=runtime)
        return out_a, out_b
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)


def _run_compact(wrapper, x_bits: np.ndarray, expert_start: np.ndarray, tiles: np.ndarray) -> np.ndarray:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    x8_library = build_gguf_x8_selected_gemv(load=True)
    q4_library = build_gguf_q4_k_gemv(load=True)
    compact_rows, in_features = x_bits.shape
    num_experts = int(tiles.shape[0])
    out_features = int(tiles.shape[1] * 8)
    out = np.zeros((compact_rows, out_features), dtype=np.uint16)
    bufs = []
    try:
        x_buf = malloc(x_bits.nbytes, runtime=runtime)
        expert_start_buf = malloc(expert_start.nbytes, runtime=runtime)
        tiles_buf = malloc(tiles.nbytes, runtime=runtime)
        xq_buf = malloc(compact_rows * (in_features // Q8_1_BLOCK) * Q8_1_BLOCK_BYTES, runtime=runtime)
        out_buf = malloc(out.nbytes, runtime=runtime)
        bufs.extend((x_buf, expert_start_buf, tiles_buf, xq_buf, out_buf))
        copy_host_to_device(x_buf, host_array_ptr(np.ascontiguousarray(x_bits)), runtime=runtime)
        copy_host_to_device(expert_start_buf, host_array_ptr(np.ascontiguousarray(expert_start)), runtime=runtime)
        copy_host_to_device(tiles_buf, host_array_ptr(np.ascontiguousarray(tiles)), runtime=runtime)
        gguf_q4_k_quantize_bf16_q8_1(
            x_buf.ptr, xq_buf.ptr, compact_rows, in_features, library=q4_library, runtime=runtime
        )
        wrapper(
            xq_buf.ptr,
            expert_start_buf.ptr,
            tiles_buf.ptr,
            out_buf.ptr,
            compact_rows,
            in_features,
            out_features,
            num_experts,
            library=x8_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_buf, runtime=runtime)
        return out
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)


def test_x8_selected_registry_and_contract() -> None:
    register_gguf_x8_selected_gemv_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q4_k_x8_v1",
            variant="selected_dual_x8_q8_1_dp4a_gemv_decode_bf16_bf16_out",
        )
        is gguf_q4_k_x8_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q5_k_x8_v1",
            variant="selected_x8_q8_1_dp4a_gemv_decode_bf16_bf16_out",
        )
        is gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_q6_k_x8_v1",
            variant="selected_x8_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out",
        )
        is gguf_q6_k_x8_selected_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out
    )
    with pytest.raises(ValueError, match="divisible by 8"):
        gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out(1, 2, 3, 4, 1, 1, 1, 256, 10)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q4_x8_selected_dual_direct_matches_q8_1_oracle_and_cpu_gate() -> None:
    rng = np.random.default_rng(20260628)
    x_f32 = (rng.standard_normal((2, 512)).astype(np.float32) * 0.1) + 0.002
    x_bits = _bf16_bits(x_f32)
    qweight_a = _weights("q4")
    qweight_b = np.ascontiguousarray(np.roll(qweight_a, shift=3, axis=1))
    selected = np.asarray([0, 2, 1, 3], dtype=np.int64)
    rows = int(selected.size)
    x_row_for_output = np.arange(rows, dtype=np.int64) // (rows // x_f32.shape[0])

    got_a_bits, got_b_bits = _run_dual_direct(
        gguf_q4_k_x8_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out,
        x_bits,
        selected,
        _tiles("q4", qweight_a),
        _tiles("q4", qweight_b),
    )
    got_a = _bf16_to_f32(got_a_bits)
    got_b = _bf16_to_f32(got_b_bits)
    x_bf16 = _bf16_to_f32(x_bits)
    q8_ref_a = _q8_oracle("q4", x_bf16, x_row_for_output, selected, qweight_a)
    q8_ref_b = _q8_oracle("q4", x_bf16, x_row_for_output, selected, qweight_b)
    np.testing.assert_allclose(got_a, q8_ref_a, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(got_b, q8_ref_b, rtol=2e-2, atol=2e-2)

    exact_a = _exact_oracle("q4", x_bf16, x_row_for_output, selected, qweight_a)
    exact_b = _exact_oracle("q4", x_bf16, x_row_for_output, selected, qweight_b)
    for exact, got in ((exact_a, got_a), (exact_b, got_b)):
        kl_mean, kl_max = _softmax_kl(exact, got)
        assert kl_mean <= 0.05
        assert kl_max <= 0.10
        assert _top1(exact, got) >= 0.90


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "quant,wrapper",
    [
        ("q5", gguf_q5_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out),
        ("q6", gguf_q6_k_x8_selected_q8_1_dp4a_gemv_bf16_bf16_out),
    ],
)
def test_x8_selected_direct_matches_q8_1_oracle_and_cpu_gate(quant, wrapper) -> None:
    rng = np.random.default_rng(20260627 + (50 if quant == "q5" else 60))
    x_f32 = (rng.standard_normal((2, 512)).astype(np.float32) * 0.1) + 0.002
    x_bits = _bf16_bits(x_f32)
    qweight = _weights(quant)
    selected = np.asarray([0, 2, 1, 3], dtype=np.int64)
    rows = int(selected.size)
    x_row_for_output = np.arange(rows, dtype=np.int64) // (rows // x_f32.shape[0])

    got = _bf16_to_f32(_run_direct(wrapper, x_bits, selected, _tiles(quant, qweight)))
    q8_ref = _q8_oracle(quant, _bf16_to_f32(x_bits), x_row_for_output, selected, qweight)
    np.testing.assert_allclose(got, q8_ref, rtol=2e-2, atol=2e-2)

    exact = _exact_oracle(quant, _bf16_to_f32(x_bits), x_row_for_output, selected, qweight)
    kl_mean, kl_max = _softmax_kl(exact, got)
    assert kl_mean <= 0.05
    assert kl_max <= 0.10
    assert _top1(exact, got) >= 0.90


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "quant,wrapper",
    [
        ("q5", gguf_q5_k_x8_selected_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out),
        ("q6", gguf_q6_k_x8_selected_q8_1_dp4a_gemv_decode_compact_bf16_bf16_out),
    ],
)
def test_x8_selected_compact_matches_q8_1_oracle_and_cpu_gate(quant, wrapper) -> None:
    rng = np.random.default_rng(20260628 + (50 if quant == "q5" else 60))
    x_f32 = (rng.standard_normal((4, 512)).astype(np.float32) * 0.1) + 0.002
    x_bits = _bf16_bits(x_f32)
    qweight = _weights(quant)
    expert_start = np.asarray([0, 1, 1, 3, 4], dtype=np.int64)
    expert_for_output = np.asarray([0, 2, 2, 3], dtype=np.int64)
    x_row_for_output = np.arange(4, dtype=np.int64)

    got = _bf16_to_f32(_run_compact(wrapper, x_bits, expert_start, _tiles(quant, qweight)))
    q8_ref = _q8_oracle(quant, _bf16_to_f32(x_bits), x_row_for_output, expert_for_output, qweight)
    np.testing.assert_allclose(got, q8_ref, rtol=2e-2, atol=2e-2)

    exact = _exact_oracle(quant, _bf16_to_f32(x_bits), x_row_for_output, expert_for_output, qweight)
    kl_mean, kl_max = _softmax_kl(exact, got)
    assert kl_mean <= 0.05
    assert kl_max <= 0.10
    assert _top1(exact, got) >= 0.90
