from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.ops import (
    step_gqa_attention_decode,
    step_gqa_attention_prefill,
    step_kv_live_span_bounds,
)


def _manual_gqa_decode(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    start: int,
    count: int,
) -> np.ndarray:
    group = query.shape[0] // key.shape[1]
    out = np.zeros_like(query, dtype=np.float32)
    scale = query.shape[-1] ** -0.5
    for q_head in range(query.shape[0]):
        kv_head = q_head // group
        keys = key[start : start + count, kv_head]
        values = value[start : start + count, kv_head]
        logits = keys @ query[q_head] * scale
        logits = logits - np.max(logits)
        weights = np.exp(logits) / np.sum(np.exp(logits))
        out[q_head] = weights @ values
    return out


def test_step_kv_live_span_bounds_full_and_sliding() -> None:
    counts = np.asarray([1, 511, 512, 513, 600], dtype=np.int64)

    full_start, full_count = step_kv_live_span_bounds(counts)
    sliding_start, sliding_count = step_kv_live_span_bounds(counts, sliding_window=512)

    np.testing.assert_array_equal(full_start, [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(full_count, counts)
    np.testing.assert_array_equal(sliding_start, [0, 0, 0, 1, 88])
    np.testing.assert_array_equal(sliding_count, [1, 511, 512, 512, 512])
    with pytest.raises(ValueError, match="sliding_window"):
        step_kv_live_span_bounds(counts, sliding_window=0)


def test_step_full_attention_decode_matches_manual_gqa_shape() -> None:
    rng = np.random.default_rng(123)
    query = rng.normal(size=(1, 64, 128)).astype(np.float32)
    key = rng.normal(size=(1, 7, 8, 128)).astype(np.float32)
    value = rng.normal(size=(1, 7, 8, 128)).astype(np.float32)

    out = step_gqa_attention_decode(query, key, value, np.asarray([7], dtype=np.int64))
    expected = _manual_gqa_decode(query[0], key[0], value[0], start=0, count=7)

    assert out.shape == (1, 64, 128)
    np.testing.assert_allclose(out[0], expected, rtol=1e-6, atol=1e-6)


def test_step_sliding_attention_decode_uses_window_boundary() -> None:
    rng = np.random.default_rng(456)
    query = rng.normal(size=(1, 96, 128)).astype(np.float32)
    key = rng.normal(size=(1, 600, 8, 128)).astype(np.float32)
    value = rng.normal(size=(1, 600, 8, 128)).astype(np.float32)

    out = step_gqa_attention_decode(
        query,
        key,
        value,
        np.asarray([600], dtype=np.int64),
        sliding_window=512,
    )
    expected = _manual_gqa_decode(query[0], key[0], value[0], start=88, count=512)
    full = step_gqa_attention_decode(query, key, value, np.asarray([600], dtype=np.int64))

    assert out.shape == (1, 96, 128)
    np.testing.assert_allclose(out[0], expected, rtol=1e-6, atol=1e-6)
    assert not np.allclose(out, full)


def test_step_gqa_prefill_matches_repeated_decode_for_full_and_sliding() -> None:
    rng = np.random.default_rng(789)
    query = rng.normal(size=(6, 4, 16)).astype(np.float32)
    key = rng.normal(size=(6, 2, 16)).astype(np.float32)
    value = rng.normal(size=(6, 2, 16)).astype(np.float32)

    for window in (None, 3):
        prefill = step_gqa_attention_prefill(query, key, value, sliding_window=window)
        rows = []
        for pos in range(query.shape[0]):
            rows.append(
                step_gqa_attention_decode(
                    query[pos : pos + 1],
                    key[None, : pos + 1],
                    value[None, : pos + 1],
                    np.asarray([pos + 1], dtype=np.int64),
                    sliding_window=window,
                )[0]
            )
        expected = np.stack(rows, axis=0)
        np.testing.assert_allclose(prefill, expected, rtol=1e-6, atol=1e-6)
