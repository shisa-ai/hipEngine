"""Unit tests for the chunked GDN drift screen (ledger row GD-4).

The screen exists to answer whether a chunked GDN scan drifts, and its committed result
is that it does not (fp32 NMSE ~5-7e-14, fp16 state carry ~1e-10, no chunk/tail boundary
effect). Two things must stay true for that conclusion to keep meaning anything:

1. The chunked form must remain **algebraically identical** to the sequential recurrence.
   If a refactor breaks the UT transform, every drift figure silently becomes a
   measurement of a different algorithm.
2. The decay-ratio matrix must stay **bounded by 1**. The first revision of the screen
   computed ``D_t/D_j`` as a ratio of prefix products, which explodes when the numerator's
   and denominator's underflows are not simultaneous - larger chunks got *worse* (max_abs
   1.05e1 at chunk 256 vs 2.26e-01 at 64), which is the signature of the bug. That is the
   finding a kernel author most needs, so it gets a direct assertion rather than a comment.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from gguf_gdn_replay_compare import GDNShape, replay_gdn  # noqa: E402
from gdn_chunk_drift_screen import (  # noqa: E402
    DEFAULT_EPS,
    _ratio_matrix,
    chunked_reference,
    make_inputs,
)

_SHAPE = GDNShape(num_k_heads=16, num_v_heads=48, head_k_dim=128, head_v_dim=128)


def _drift(tokens: int, chunk: int, *, state_dtype=np.float32, seed: int = 0):
    inputs = make_inputs(tokens, _SHAPE, seed=seed)
    ref = replay_gdn(shape=_SHAPE, eps=DEFAULT_EPS, **inputs)
    cand = chunked_reference(
        shape=_SHAPE, eps=DEFAULT_EPS, chunk_size=chunk, state_dtype=state_dtype, **inputs
    )
    diff = cand.astype(np.float64) - ref.astype(np.float64)
    return diff, ref


@pytest.mark.parametrize("tokens,chunk", [(8, 8), (32, 32), (64, 32), (64, 64)])
def test_chunked_form_matches_sequential_to_fp32_roundoff(tokens: int, chunk: int) -> None:
    """RED: the chunked form is the same arithmetic, so it must land on fp32 epsilon."""
    diff, ref = _drift(tokens, chunk)
    nmse = np.sum(diff * diff) / np.sum(ref.astype(np.float64) ** 2)
    assert nmse < 1.0e-12, f"chunked scan diverged from the sequential reference: nmse={nmse:.3e}"


def test_smaller_chunks_are_not_worse_than_larger_ones() -> None:
    """A stable reordering must not degrade as the prefix horizon shrinks."""
    small, ref = _drift(64, 16)
    large, _ = _drift(64, 64)
    ref_energy = np.sum(ref.astype(np.float64) ** 2)
    assert np.sum(small * small) / ref_energy <= 10.0 * np.sum(large * large) / ref_energy


def test_ratio_matrix_is_bounded_and_lower_triangular() -> None:
    """RED: the decay-ratio matrix must never exceed 1.

    This is the guard on the explosion the first revision hit. A ratio of prefix products
    can exceed 1 when the denominator underflows first; a direct suffix product cannot,
    because every factor is a decay in (0, 1].
    """
    decay = np.full((16,), 0.5, dtype=np.float32)
    ratio = _ratio_matrix(decay, 16)
    assert np.all(ratio >= 0.0)
    assert np.all(ratio <= 1.0 + 1.0e-6), f"ratio exceeded 1: max={float(np.max(ratio)):.3e}"
    assert np.allclose(np.triu(ratio, 1), 0.0), "ratio must be lower triangular"
    assert np.allclose(np.diag(ratio), 1.0), "diagonal must be 1 (D_i/D_i)"


def test_ratio_matrix_survives_underflow() -> None:
    """A long, strongly decaying chunk must underflow to zero, never to a huge value."""
    decay = np.full((256,), 0.5, dtype=np.float32)
    ratio = _ratio_matrix(decay, 256)
    assert np.all(np.isfinite(ratio))
    assert np.all(ratio <= 1.0 + 1.0e-6)
    # the far corner is genuinely decayed away, and it is exactly 0 rather than inf/nan
    assert ratio[-1, 0] == 0.0


def test_fp16_state_carry_stays_far_below_the_reference_claim() -> None:
    """fp16 carry is the case the row cares about; PR #91 cites NMSE ~2e-7."""
    diff, ref = _drift(256, 128, state_dtype=np.float16)
    nmse = np.sum(diff * diff) / np.sum(ref.astype(np.float64) ** 2)
    assert nmse < 2.0e-7, f"fp16 state carry worse than the cited reference: nmse={nmse:.3e}"


def test_partial_final_chunk_costs_nothing() -> None:
    """A tail that is not a multiple of the chunk must behave like the exact multiple."""
    exact, ref_exact = _drift(256, 128)
    tail, ref_tail = _drift(255, 128)
    nmse_exact = np.sum(exact * exact) / np.sum(ref_exact.astype(np.float64) ** 2)
    nmse_tail = np.sum(tail * tail) / np.sum(ref_tail.astype(np.float64) ** 2)
    assert nmse_tail < 1.0e-12
    assert nmse_tail < 10.0 * nmse_exact
