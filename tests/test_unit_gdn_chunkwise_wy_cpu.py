"""CPU algebra/oracle tests for chunkwise/WY Gated Delta Rule prefill."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import (
    gdn_prefill_chunkwise_wy_segments,
    gdn_prefill_recurrent_segments,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "gdn_chunkwise_wy_tiny.json"


def _serial_f64(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    beta: np.ndarray,
    decay: np.ndarray,
    recurrent_state: np.ndarray,
    cu_seqlens: np.ndarray,
    state_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent token-serial, float64 definition of the GDN recurrence."""

    q = np.asarray(query, dtype=np.float64)
    k = np.asarray(key, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    b = np.asarray(beta, dtype=np.float64)
    d = np.asarray(decay, dtype=np.float64)
    state = np.asarray(recurrent_state, dtype=np.float64).copy()
    out = np.empty_like(v)
    for segment, slot in enumerate(np.asarray(state_indices, dtype=np.int64)):
        start = int(cu_seqlens[segment])
        end = int(cu_seqlens[segment + 1])
        for token in range(start, end):
            for head in range(q.shape[1]):
                decayed = d[token, head] * state[slot, head]
                residual = v[token, head] - k[token, head] @ decayed
                delta = b[token, head] * residual
                state[slot, head] = decayed + np.outer(k[token, head], delta)
                out[token, head] = q[token, head] @ state[slot, head]
    return out.astype(np.float32), state.astype(np.float32)


def _random_case(
    *, tokens: int, heads: int, key_dim: int, value_dim: int, slots: int, seed: int
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    query = rng.normal(0.0, 0.25, size=(tokens, heads, key_dim)).astype(np.float32)
    key = rng.normal(0.0, 0.25, size=(tokens, heads, key_dim)).astype(np.float32)
    key /= np.linalg.norm(key, axis=-1, keepdims=True)
    value = rng.normal(0.0, 0.4, size=(tokens, heads, value_dim)).astype(np.float32)
    beta = rng.uniform(0.1, 0.9, size=(tokens, heads)).astype(np.float32)
    decay = rng.uniform(0.7, 0.999, size=(tokens, heads)).astype(np.float32)
    state = rng.normal(0.0, 0.05, size=(slots, heads, key_dim, value_dim)).astype(
        np.float32
    )
    return query, key, value, beta, decay, state


def test_chunkwise_wy_matches_hand_checked_fixture() -> None:
    payload = json.loads(_FIXTURE.read_text())
    arrays = {
        name: np.asarray(value["data"], dtype=np.dtype(value["dtype"]))
        for name, value in payload["inputs"].items()
    }
    out, state = gdn_prefill_chunkwise_wy_segments(
        arrays["query"],
        arrays["key"],
        arrays["value"],
        arrays["beta"],
        arrays["decay"],
        arrays["recurrent_state"],
        arrays["cu_seqlens"],
        arrays["state_indices"],
        chunk_size=payload["chunk_size"],
    )
    expected_out = np.asarray(
        payload["expected"]["out"]["data"],
        dtype=np.dtype(payload["expected"]["out"]["dtype"]),
    )
    expected_state = np.asarray(
        payload["expected"]["state"]["data"],
        dtype=np.dtype(payload["expected"]["state"]["dtype"]),
    )
    atol = float(payload["tolerances"]["atol"])
    rtol = float(payload["tolerances"]["rtol"])
    np.testing.assert_allclose(out, expected_out, atol=atol, rtol=rtol)
    np.testing.assert_allclose(state, expected_state, atol=atol, rtol=rtol)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 8, 16])
def test_chunkwise_wy_is_algebraically_equal_to_serial(chunk_size: int) -> None:
    arrays = _random_case(
        tokens=17, heads=2, key_dim=7, value_dim=5, slots=3, seed=chunk_size
    )
    cu_seqlens = np.asarray([0, 9, 17], dtype=np.int32)
    state_indices = np.asarray([2, 0], dtype=np.int64)
    expected_out, expected_state = _serial_f64(
        *arrays, cu_seqlens, state_indices
    )
    actual_out, actual_state = gdn_prefill_chunkwise_wy_segments(
        *arrays,
        cu_seqlens,
        state_indices,
        chunk_size=chunk_size,
    )
    # The oracle evaluates the same recurrence in float64 through two different
    # algebraic forms before the common float32 boundary.
    np.testing.assert_allclose(actual_out, expected_out, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(actual_state, expected_state, atol=2e-6, rtol=2e-6)
    # State slot 1 is not selected by either packed segment.
    np.testing.assert_array_equal(actual_state[1], arrays[-1][1])


def test_chunkwise_wy_stays_inside_fp32_primitive_budget() -> None:
    arrays = _random_case(
        tokens=33, heads=2, key_dim=128, value_dim=16, slots=1, seed=8128
    )
    cu_seqlens = np.asarray([0, 33], dtype=np.int32)
    state_indices = np.asarray([0], dtype=np.int64)
    serial_out, serial_state = gdn_prefill_recurrent_segments(
        *arrays, cu_seqlens, state_indices
    )
    chunk_out, chunk_state = gdn_prefill_chunkwise_wy_segments(
        *arrays, cu_seqlens, state_indices, chunk_size=8
    )
    assert np.isfinite(chunk_out).all()
    assert np.isfinite(chunk_state).all()
    np.testing.assert_allclose(chunk_out, serial_out, atol=3e-5, rtol=3e-4)
    np.testing.assert_allclose(chunk_state, serial_state, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_chunkwise_wy_rejects_nonpositive_chunk_size(chunk_size: int) -> None:
    arrays = _random_case(
        tokens=1, heads=1, key_dim=2, value_dim=2, slots=1, seed=0
    )
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        gdn_prefill_chunkwise_wy_segments(
            *arrays,
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([0], dtype=np.int64),
            chunk_size=chunk_size,
        )
