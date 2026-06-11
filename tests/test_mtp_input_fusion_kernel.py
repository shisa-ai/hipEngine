from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.speculative.mtp import (
    build_mtp_speculative,
    mtp_accumulate_route_bf16_to_f32,
    mtp_accumulate_sigmoid_gate_bf16_to_f32,
    mtp_add_rmsnorm_bf16_oneplus,
    mtp_finalize_f32_to_bf16,
    mtp_fuse_inputs_f16_bf16,
    mtp_gate_mul_bf16,
    mtp_rmsnorm_bf16_oneplus,
    mtp_softmax_topk_f32,
    mtp_split_q_gate_f32_bf16,
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def _f32_to_bf16_bits(array: np.ndarray) -> np.ndarray:
    f32 = np.asarray(array, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    lsb = (u32 >> 16) & np.uint32(1)
    u32 += np.uint32(0x7FFF) + lsb
    return (u32 >> 16).astype(np.uint16)


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    u32 = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


@pytest.mark.skipif(not _hip_available(), reason="ROCm runtime not available")
def test_mtp_fuse_inputs_f16_bf16_matches_cpu_reference() -> None:
    rows = 2
    hidden = 8
    vocab = 6
    eps = 1.0e-6
    token_ids = np.array([1, 4], dtype=np.int64)
    embedding = (np.arange(vocab * hidden, dtype=np.float32).reshape(vocab, hidden) / 17.0 - 0.8).astype(np.float16)
    target_hidden = np.array(
        [
            [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
            [0.9, -0.7, 0.5, -0.3, 0.1, 0.2, -0.4, 0.6],
        ],
        dtype=np.float32,
    )
    embed_weight = np.linspace(-0.05, 0.08, hidden, dtype=np.float32)
    hidden_weight = np.linspace(0.07, -0.04, hidden, dtype=np.float32)

    target_bits = _f32_to_bf16_bits(target_hidden)
    embed_weight_bits = _f32_to_bf16_bits(embed_weight)
    hidden_weight_bits = _f32_to_bf16_bits(hidden_weight)
    out = np.zeros((rows, 2 * hidden), dtype=np.uint16)

    buffers = []
    try:
        for arr in (token_ids, embedding, target_bits, embed_weight_bits, hidden_weight_bits, out):
            buf = malloc(arr.nbytes)
            copy_host_to_device(buf, host_array_ptr(np.ascontiguousarray(arr)), arr.nbytes)
            buffers.append(buf)
        library = build_mtp_speculative(load=True)
        mtp_fuse_inputs_f16_bf16(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            buffers[3].ptr,
            buffers[4].ptr,
            buffers[5].ptr,
            rows,
            hidden,
            vocab,
            eps=eps,
            threads=64,
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), buffers[5], out.nbytes)
    finally:
        for buf in reversed(buffers):
            free(buf)

    expected = np.zeros((rows, 2 * hidden), dtype=np.float32)
    for row, token in enumerate(token_ids):
        embed = embedding[token].astype(np.float32)
        hidden_row = target_hidden[row].astype(np.float32)
        embed_norm = embed * (1.0 / np.sqrt(np.mean(embed * embed) + eps)) * (1.0 + _bf16_bits_to_f32(embed_weight_bits))
        hidden_norm = hidden_row * (1.0 / np.sqrt(np.mean(hidden_row * hidden_row) + eps)) * (1.0 + _bf16_bits_to_f32(hidden_weight_bits))
        expected[row, :hidden] = embed_norm
        expected[row, hidden:] = hidden_norm
    expected_bf16 = _f32_to_bf16_bits(expected)

    actual = _bf16_bits_to_f32(out)
    expected_after_round = _bf16_bits_to_f32(expected_bf16)
    assert np.max(np.abs(actual - expected_after_round)) <= 1.0e-2
    assert np.array_equal(out.shape, expected_bf16.shape)


@pytest.mark.skipif(not _hip_available(), reason="ROCm runtime not available")
def test_mtp_decoder_helper_kernels_match_cpu_reference() -> None:
    rows = 1
    hidden = 8
    eps = 1.0e-6
    x_f32 = np.linspace(-0.7, 0.9, hidden, dtype=np.float32).reshape(rows, hidden)
    residual_f32 = np.linspace(0.5, -0.4, hidden, dtype=np.float32).reshape(rows, hidden)
    weight_f32 = np.linspace(-0.03, 0.06, hidden, dtype=np.float32)
    x_bits = _f32_to_bf16_bits(x_f32)
    residual_bits = _f32_to_bf16_bits(residual_f32)
    weight_bits = _f32_to_bf16_bits(weight_f32)
    norm_bits = np.zeros_like(x_bits)
    add_norm_bits = np.zeros_like(x_bits)
    residual_out_bits = np.zeros_like(x_bits)

    q_proj = np.array([[0.1, 0.2, 1.0, 1.2, -0.3, -0.4, -1.0, -1.2]], dtype=np.float32)
    query = np.zeros((1, 4), dtype=np.float32)
    gate = np.zeros((1, 4), dtype=np.uint16)
    attn_bits = _f32_to_bf16_bits(np.array([[0.25, -0.5, 0.75, 1.0]], dtype=np.float32))
    gated_bits = np.zeros_like(attn_bits)
    topk_values = np.array([[2.0, 1.0, -1.0]], dtype=np.float32)
    routing = np.zeros_like(topk_values)
    accum = np.full((1, 4), 7.0, dtype=np.float32)
    finalized = np.zeros((1, 4), dtype=np.uint16)

    arrays = [
        x_bits,
        residual_bits,
        weight_bits,
        norm_bits,
        add_norm_bits,
        residual_out_bits,
        q_proj,
        query,
        gate,
        attn_bits,
        gated_bits,
        topk_values,
        routing,
        accum,
        finalized,
    ]
    buffers = []
    try:
        for arr in arrays:
            buf = malloc(arr.nbytes)
            copy_host_to_device(buf, host_array_ptr(np.ascontiguousarray(arr)), arr.nbytes)
            buffers.append(buf)
        library = build_mtp_speculative(load=True)
        mtp_rmsnorm_bf16_oneplus(buffers[0].ptr, buffers[2].ptr, buffers[3].ptr, rows, hidden, eps=eps, threads=64, library=library)
        mtp_add_rmsnorm_bf16_oneplus(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            buffers[4].ptr,
            buffers[5].ptr,
            rows,
            hidden,
            eps=eps,
            threads=64,
            library=library,
        )
        mtp_split_q_gate_f32_bf16(buffers[6].ptr, buffers[7].ptr, buffers[8].ptr, 1, 2, 2, threads=64, library=library)
        mtp_gate_mul_bf16(buffers[9].ptr, buffers[8].ptr, buffers[10].ptr, 4, threads=64, library=library)
        mtp_softmax_topk_f32(buffers[11].ptr, buffers[12].ptr, 1, 3, library=library)
        mtp_accumulate_route_bf16_to_f32(
            buffers[10].ptr,
            buffers[12].ptr,
            buffers[13].ptr,
            4,
            0,
            reset_output=True,
            threads=64,
            library=library,
        )
        mtp_accumulate_sigmoid_gate_bf16_to_f32(buffers[10].ptr, buffers[6].ptr, buffers[13].ptr, 4, threads=64, library=library)
        mtp_finalize_f32_to_bf16(buffers[13].ptr, buffers[14].ptr, 4, threads=64, library=library)
        for arr, buf in ((norm_bits, buffers[3]), (add_norm_bits, buffers[4]), (residual_out_bits, buffers[5]), (query, buffers[7]), (gate, buffers[8]), (gated_bits, buffers[10]), (routing, buffers[12]), (accum, buffers[13]), (finalized, buffers[14])):
            copy_device_to_host(host_array_ptr(arr), buf, arr.nbytes)
    finally:
        for buf in reversed(buffers):
            free(buf)

    x_rounded = _bf16_bits_to_f32(x_bits)
    residual_rounded = _bf16_bits_to_f32(residual_bits)
    weight_rounded = _bf16_bits_to_f32(weight_bits)
    norm_ref = x_rounded * (1.0 / np.sqrt(np.mean(x_rounded * x_rounded, axis=1, keepdims=True) + eps)) * (1.0 + weight_rounded.reshape(1, -1))
    residual_ref_bits = _f32_to_bf16_bits(x_rounded + residual_rounded)
    residual_ref = _bf16_bits_to_f32(residual_ref_bits)
    add_norm_ref = residual_ref * (1.0 / np.sqrt(np.mean(residual_ref * residual_ref, axis=1, keepdims=True) + eps)) * (1.0 + weight_rounded.reshape(1, -1))
    assert np.array_equal(norm_bits, _f32_to_bf16_bits(norm_ref))
    assert np.array_equal(residual_out_bits, residual_ref_bits)
    assert np.array_equal(add_norm_bits, _f32_to_bf16_bits(add_norm_ref))
    assert np.allclose(query, np.array([[0.1, 0.2, -0.3, -0.4]], dtype=np.float32), atol=0.0)
    assert np.array_equal(gate, _f32_to_bf16_bits(np.array([[1.0, 1.2, -1.0, -1.2]], dtype=np.float32)))
    gated_ref = _bf16_bits_to_f32(attn_bits) * (1.0 / (1.0 + np.exp(-_bf16_bits_to_f32(gate))))
    assert np.array_equal(gated_bits, _f32_to_bf16_bits(gated_ref))
    routing_ref = np.exp(topk_values - np.max(topk_values, axis=1, keepdims=True))
    routing_ref = routing_ref / np.sum(routing_ref, axis=1, keepdims=True)
    assert np.allclose(routing, routing_ref, atol=1.0e-6)
    accum_ref = routing_ref[0, 0] * _bf16_bits_to_f32(gated_bits) + (1.0 / (1.0 + np.exp(-q_proj[0, 0]))) * _bf16_bits_to_f32(gated_bits)
    assert np.allclose(accum, accum_ref, atol=1.0e-6)
    assert np.array_equal(finalized, _f32_to_bf16_bits(accum_ref))
