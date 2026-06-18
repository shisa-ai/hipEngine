from __future__ import annotations

import numpy as np

from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from hipengine.runtime.qwen35_gguf_runner import _q8_0_embedding_rows_to_bf16


def _q8_row(seed: int, *, blocks: int) -> np.ndarray:
    row = np.empty((blocks, 34), dtype=np.uint8)
    for block in range(blocks):
        scale = np.float16(0.125 * (1 + ((seed + block) % 4)))
        row[block, :2] = np.frombuffer(scale.tobytes(), dtype=np.uint8)
        q = ((np.arange(32, dtype=np.int16) + seed * 7 + block * 3) % 127 - 63).astype(np.int8)
        row[block, 2:] = q.view(np.uint8)
    return row.reshape(blocks * 34)


def test_q8_0_host_token_embedding_matches_reference_bf16() -> None:
    hidden_size = 64
    blocks = hidden_size // 32
    raw = np.stack([_q8_row(seed, blocks=blocks) for seed in range(5)], axis=0)
    token_ids = np.asarray([3, 1, 3, 4], dtype=np.int64)

    actual = _q8_0_embedding_rows_to_bf16(raw, token_ids, hidden_size=hidden_size, cache={})
    reference = float_array_to_bf16_bits(dequantize_gguf_data(raw[token_ids], GGMLQuantizationType.Q8_0))

    assert actual.dtype == np.uint16
    assert actual.shape == (4, hidden_size)
    np.testing.assert_array_equal(actual, reference)


def test_q8_0_host_token_embedding_uses_supplied_cache() -> None:
    hidden_size = 64
    blocks = hidden_size // 32
    raw = np.stack([_q8_row(seed, blocks=blocks) for seed in range(3)], axis=0)
    cache: dict[int, np.ndarray] = {}

    first = _q8_0_embedding_rows_to_bf16(raw, np.asarray([2, 2], dtype=np.int64), hidden_size=hidden_size, cache=cache)
    cached_row = cache[2]
    second = _q8_0_embedding_rows_to_bf16(raw, np.asarray([2], dtype=np.int64), hidden_size=hidden_size, cache=cache)

    assert cache[2] is cached_row
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[0])
