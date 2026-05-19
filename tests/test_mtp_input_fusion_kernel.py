from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.speculative.mtp import build_mtp_speculative, mtp_fuse_inputs_f16_bf16


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
