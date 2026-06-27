"""Bit-lossless fixtures for GGUF selected-down X8 replacement layouts."""

from __future__ import annotations

import numpy as np

from hipengine.quant.registry import resolve_quant
from hipengine.quant.gguf_x8 import (
    GGUF_Q5_K_X8_BLOCK_BYTES,
    GGUF_Q6_K_X8_BLOCK_BYTES,
    GGUF_X8_COLS,
    repack_gguf_q5_k_x8,
    repack_gguf_q6_k_x8,
    unpack_gguf_q5_k_x8,
    unpack_gguf_q6_k_x8,
)
from tests._gguf_synthetic_weights import make_q5_k_weight, make_q6_k_weight


def _stack(builder, *, out_features: int = 24, in_features: int = 512, experts: int = 3) -> np.ndarray:
    base = builder(out_features, in_features)
    return np.ascontiguousarray(
        np.stack([np.roll(base, shift=expert + 1, axis=0) for expert in range(experts)], axis=0)
    )


def test_q5_x8_repack_is_byte_lossless() -> None:
    raw = _stack(make_q5_k_weight)
    packed = repack_gguf_q5_k_x8(raw)

    assert packed.tiles.shape == (3, raw.shape[1] // GGUF_X8_COLS, 2, GGUF_Q5_K_X8_BLOCK_BYTES)
    np.testing.assert_array_equal(unpack_gguf_q5_k_x8(packed), raw)


def test_q6_x8_repack_is_byte_lossless() -> None:
    raw = _stack(make_q6_k_weight)
    packed = repack_gguf_q6_k_x8(raw)

    assert packed.tiles.shape == (3, raw.shape[1] // GGUF_X8_COLS, 2, GGUF_Q6_K_X8_BLOCK_BYTES)
    np.testing.assert_array_equal(unpack_gguf_q6_k_x8(packed), raw)


def test_x8_quant_keys_are_registered() -> None:
    assert resolve_quant("gguf_q5_k_x8_v1").kernel_family == "gguf_x8_gemv"
    assert resolve_quant("gguf_q6_k_x8_v1").kernel_family == "gguf_x8_gemv"
