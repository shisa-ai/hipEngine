"""POC gate for GGUF Q4_K selected-dual q8_1 + sudot4 GEMV."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_quantize_bf16_q8_1,
    gguf_q4_k_selected_dual_dp4a_gemv_bf16_bf16_out,
    gguf_q4_k_selected_dual_gemv_bf16_bf16_out,
    gguf_q4_k_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out,
)
from hipengine.kernels.registry import resolve
from tests.test_gguf_q4_k_gemv import Q4_K_BLOCK_BYTES, QK_K, make_q4_k_weight

Q8_1_BLOCK = 32
Q8_1_BLOCK_BYTES = 36


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(arr: np.ndarray) -> np.ndarray:
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    lsb = (u32 >> 16) & 1
    u32 = u32 + 0x7FFF + lsb
    return (u32 >> 16).astype(np.uint16)


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


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


def _q4_k_subblock_qs(block: np.ndarray, subblock: int) -> np.ndarray:
    qs = block[16:144]
    pair = subblock >> 1
    packed = qs[pair * 32 : pair * 32 + 32]
    if subblock & 1:
        return (packed >> 4).astype(np.int32)
    return (packed & 0x0F).astype(np.int32)


def _round_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def _softmax_kl(ref: np.ndarray, cand: np.ndarray) -> np.ndarray:
    ref64 = ref.astype(np.float64)
    cand64 = cand.astype(np.float64)

    def logsm(x: np.ndarray) -> np.ndarray:
        shifted = x - x.max(axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    log_ref = logsm(ref64)
    log_cand = logsm(cand64)
    return np.sum(np.exp(log_ref) * (log_ref - log_cand), axis=-1)


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


def _selected_dual_q8_1_cpu(
    x_f32: np.ndarray,
    selected: np.ndarray,
    qa: np.ndarray,
    qb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q8, d8 = _quantize_q8_1_cpu(x_f32)
    rows = int(selected.size)
    x_rows, in_features = x_f32.shape
    out_features = qa.shape[1]
    lanes_per_x_row = rows // x_rows
    out_a = np.zeros((rows, out_features), dtype=np.float32)
    out_b = np.zeros_like(out_a)

    for row, expert in enumerate(selected.astype(np.int64)):
        x_row = 0 if x_rows == 1 else row // lanes_per_x_row
        for out_col in range(out_features):
            acc_a = 0.0
            acc_b = 0.0
            for block_idx in range(in_features // QK_K):
                for subblock in range(8):
                    q8_idx = block_idx * 8 + subblock
                    q8_vals = q8[x_row, q8_idx].astype(np.int32)
                    q8_sum = int(np.sum(q8_vals))
                    xd = float(d8[x_row, q8_idx])
                    for weights, accum in ((qa, "a"), (qb, "b")):
                        block = weights[expert, out_col, block_idx * Q4_K_BLOCK_BYTES : (block_idx + 1) * Q4_K_BLOCK_BYTES]
                        d = block[0:2].view(np.float16).astype(np.float32)[0]
                        dmin = block[2:4].view(np.float16).astype(np.float32)[0]
                        scales = block[4:16]
                        q4_vals = _q4_k_subblock_qs(block, subblock)
                        dot = int(np.sum(q4_vals * q8_vals))
                        term = xd * (
                            float(d) * _q4_k_scale(scales, subblock) * dot
                            - float(dmin) * _q4_k_min(scales, subblock) * q8_sum
                        )
                        if accum == "a":
                            acc_a += term
                        else:
                            acc_b += term
            out_a[row, out_col] = acc_a
            out_b[row, out_col] = acc_b
    return out_a, out_b


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_rows = 2
    rows = 4
    num_experts = 3
    in_features = 512
    out_features = 24
    rng = np.random.default_rng(20260627)
    x_f32 = (rng.standard_normal((x_rows, in_features)).astype(np.float32) * 0.1) + 0.003
    x_bits = _bf16_bits(x_f32).reshape(x_rows, in_features)
    qa = np.stack([make_q4_k_weight(out_features, in_features) for _ in range(num_experts)], axis=0)
    qb = np.stack([make_q4_k_weight(out_features, in_features) for _ in range(num_experts)], axis=0)
    qb = np.roll(qb, shift=3, axis=1).copy()
    selected = np.asarray([0, 2, 1, 2], dtype=np.int64)
    assert rows == selected.size
    return x_bits, selected, qa, qb


def _run_dual(wrapper, x_bits, selected, qa, qb, *, prequantized: bool = False) -> tuple[np.ndarray, np.ndarray]:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc

    runtime = get_hip_runtime()
    library = build_gguf_q4_k_gemv(load=True)
    x_rows, in_features = x_bits.shape
    rows = int(selected.size)
    num_experts, out_features = qa.shape[:2]
    out_a = np.zeros((rows, out_features), dtype=np.uint16)
    out_b = np.zeros_like(out_a)
    bufs = []
    try:
        x_buf = malloc(x_bits.nbytes, runtime=runtime)
        sel_buf = malloc(selected.nbytes, runtime=runtime)
        qa_buf = malloc(qa.nbytes, runtime=runtime)
        qb_buf = malloc(qb.nbytes, runtime=runtime)
        out_a_buf = malloc(out_a.nbytes, runtime=runtime)
        out_b_buf = malloc(out_b.nbytes, runtime=runtime)
        bufs.extend((x_buf, sel_buf, qa_buf, qb_buf, out_a_buf, out_b_buf))
        copy_host_to_device(x_buf, host_array_ptr(np.ascontiguousarray(x_bits)), runtime=runtime)
        copy_host_to_device(sel_buf, host_array_ptr(np.ascontiguousarray(selected)), runtime=runtime)
        copy_host_to_device(qa_buf, host_array_ptr(np.ascontiguousarray(qa)), runtime=runtime)
        copy_host_to_device(qb_buf, host_array_ptr(np.ascontiguousarray(qb)), runtime=runtime)
        if prequantized:
            xq_buf = malloc(x_rows * (in_features // Q8_1_BLOCK) * Q8_1_BLOCK_BYTES, runtime=runtime)
            bufs.append(xq_buf)
            gguf_q4_k_quantize_bf16_q8_1(
                x_buf.ptr, xq_buf.ptr, x_rows, in_features, library=library, runtime=runtime
            )
            wrapper(
                xq_buf.ptr,
                sel_buf.ptr,
                qa_buf.ptr,
                qb_buf.ptr,
                out_a_buf.ptr,
                out_b_buf.ptr,
                x_rows,
                rows,
                num_experts,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
        else:
            wrapper(
                x_buf.ptr,
                sel_buf.ptr,
                qa_buf.ptr,
                qb_buf.ptr,
                out_a_buf.ptr,
                out_b_buf.ptr,
                x_rows,
                rows,
                num_experts,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_a), out_a_buf, runtime=runtime)
        copy_device_to_host(host_array_ptr(out_b), out_b_buf, runtime=runtime)
        return out_a, out_b
    finally:
        for buf in reversed(bufs):
            free(buf, runtime=runtime)


def test_selected_dual_dp4a_registry_and_contract() -> None:
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="gguf_q4_k",
            variant="selected_dual_dp4a_gemv_bf16_bf16_out",
        )
        is gguf_q4_k_selected_dual_dp4a_gemv_bf16_bf16_out
    )

    with pytest.raises(ValueError, match="divisible"):
        gguf_q4_k_selected_dual_dp4a_gemv_bf16_bf16_out(1, 2, 3, 4, 5, 6, 1, 1, 1, 255, 1)
    with pytest.raises(ValueError, match="threads"):
        gguf_q4_k_selected_dual_dp4a_gemv_bf16_bf16_out(
            1, 2, 3, 4, 5, 6, 1, 1, 1, 256, 1, threads=96
        )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_selected_dual_dp4a_matches_q8_1_cpu_oracle() -> None:
    x_bits, selected, qa, qb = _fixture()
    x_f32 = _bf16_to_f32(x_bits)
    exp_a, exp_b = _selected_dual_q8_1_cpu(x_f32, selected, qa, qb)

    got_a_bits, got_b_bits = _run_dual(
        gguf_q4_k_selected_dual_dp4a_gemv_bf16_bf16_out, x_bits, selected, qa, qb
    )
    got_a = _bf16_to_f32(got_a_bits)
    got_b = _bf16_to_f32(got_b_bits)

    np.testing.assert_allclose(got_a, exp_a, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(got_b, exp_b, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_selected_dual_dp4a_prequantized_path_matches_convenience_wrapper() -> None:
    x_bits, selected, qa, qb = _fixture()
    got = _run_dual(gguf_q4_k_selected_dual_dp4a_gemv_bf16_bf16_out, x_bits, selected, qa, qb)
    preq = _run_dual(
        gguf_q4_k_selected_dual_q8_1_dp4a_gemv_bf16_bf16_out,
        x_bits,
        selected,
        qa,
        qb,
        prequantized=True,
    )

    np.testing.assert_array_equal(preq[0], got[0])
    np.testing.assert_array_equal(preq[1], got[1])


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_selected_dual_dp4a_stays_close_to_float_dequant_kernel() -> None:
    x_bits, selected, qa, qb = _fixture()
    ref_a_bits, ref_b_bits = _run_dual(gguf_q4_k_selected_dual_gemv_bf16_bf16_out, x_bits, selected, qa, qb)
    got_a_bits, got_b_bits = _run_dual(
        gguf_q4_k_selected_dual_dp4a_gemv_bf16_bf16_out, x_bits, selected, qa, qb
    )

    ref_a = _bf16_to_f32(ref_a_bits)
    ref_b = _bf16_to_f32(ref_b_bits)
    got_a = _bf16_to_f32(got_a_bits)
    got_b = _bf16_to_f32(got_b_bits)

    np.testing.assert_allclose(got_a, ref_a, rtol=8e-2, atol=8e-2)
    np.testing.assert_allclose(got_b, ref_b, rtol=8e-2, atol=8e-2)
    for ref, got in ((ref_a, got_a), (ref_b, got_b)):
        assert float(np.mean(_softmax_kl(ref, got))) <= 0.05
        assert float(np.mean(ref.argmax(axis=-1) == got.argmax(axis=-1))) >= 0.90
