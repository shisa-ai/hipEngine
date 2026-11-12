"""Unit validation for partial-selection top-p sampling (CPU only).

The top-p-only path replaces the full-vocabulary sort with adaptive top-K
partitioning and canonical-ID-order softmax normalization. Per-token
probabilities drift at ulp level vs the legacy score-sorted sum; the contract
is bounded drift + same-build determinism (EXECUTION-PROFILES production),
not cross-version bit stability. These tests pin:

- distribution parity vs the legacy full-sort implementation (raw probs to
  1e-12 relative, identical or ulp-boundary-shifted support),
- legacy paths (top_p=1, top_k>0, constraints) byte-identical,
- edge shapes (single token, uniform, ties, tiny vocab, extreme top_p),
- seeded determinism and empirical draw frequencies through select_token.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.generation import sampling as S
from hipengine.generation.registry import GenerationRequest


def _params(top_p=0.9, top_k=0, min_p=0.0, temperature=0.7):
    return GenerationRequest(
        prompts=((1, 2, 3),), max_tokens=8, temperature=temperature,
        top_p=top_p, top_k=top_k, min_p=min_p, ignore_eos=True,
    )


def _legacy_support(processed, params, *, temperature, token_allowed=None):
    """The pre-change full-sort implementation, as the parity oracle."""
    scaled = processed / float(temperature)
    top_k = int(getattr(params, "top_k", 0))
    candidate_ids = (
        S._constraint_candidate_ids(scaled, token_allowed, limit=top_k)
        if token_allowed is not None
        else S._top_k_candidate_ids(scaled, top_k)
    )
    candidate_logits = scaled[candidate_ids]
    candidate_probs = S._softmax(candidate_logits)
    retained_ids, retained_probs = S._apply_probability_filters(
        candidate_ids, candidate_probs,
        top_p=float(getattr(params, "top_p", 1.0)),
        min_p=float(getattr(params, "min_p", 0.0)),
    )
    probs_sum = float(retained_probs.sum())
    return retained_ids, retained_probs / probs_sum


def _raw_probs(processed, temperature, ids):
    scaled = np.asarray(processed, dtype=np.float64) / float(temperature)
    finite = scaled[np.isfinite(scaled)]
    shifted = scaled[ids] - float(np.max(finite))
    exp = np.exp(shifted)
    return exp / float(np.exp(finite - float(np.max(finite))).sum())


def _parity_case(logits, params, temperature):
    new_ids, new_probs = S.processed_support(logits, params, temperature=temperature)
    ref_ids, ref_probs = _legacy_support(logits, params, temperature=temperature)
    assert new_ids[0] == ref_ids[0], "top-1 token must be identical"
    assert abs(new_probs.sum() - 1.0) < 1e-12
    if np.array_equal(new_ids, ref_ids):
        np.testing.assert_allclose(new_probs, ref_probs, rtol=1e-10, atol=1e-14)
        return
    # Rare ulp-boundary shift: supports may differ by the cut token only.
    assert abs(new_ids.size - ref_ids.size) <= 1
    common = np.intersect1d(new_ids, ref_ids)
    pn = _raw_probs(logits, temperature, common)
    pr = _raw_probs(logits, temperature, common)
    np.testing.assert_allclose(pn, pr, rtol=1e-12)
    new_mass = float(_raw_probs(logits, temperature, new_ids).sum())
    ref_mass = float(_raw_probs(logits, temperature, ref_ids).sum())
    assert abs(new_mass - ref_mass) < 1e-12


@pytest.mark.parametrize("vocab", [64, 4096, 50000])
@pytest.mark.parametrize("std", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("top_p", [0.5, 0.9, 0.99])
def test_partial_top_p_distribution_parity(vocab, std, top_p):
    rng = np.random.default_rng(vocab + int(std * 10) + int(top_p * 100))
    logits = (rng.standard_normal(vocab) * std).astype(np.float32)
    for min_p in (0.0, 0.05):
        _parity_case(logits, _params(top_p=top_p, min_p=min_p), 0.7)


@pytest.mark.parametrize("shape", [
    "single", "two", "uniform", "ties", "tiny_top_p", "near_one", "masked_tail",
])
def test_partial_top_p_edge_shapes(shape):
    rng = np.random.default_rng(5)
    if shape == "single":
        logits = np.full(100, -np.inf, dtype=np.float32); logits[42] = 1.0
    elif shape == "two":
        logits = np.full(100, -np.inf, dtype=np.float32); logits[7] = 2.0; logits[90] = 1.0
    elif shape == "uniform":
        logits = np.zeros(5000, dtype=np.float32)
    elif shape == "ties":
        logits = np.round(rng.standard_normal(5000), 1).astype(np.float32)
    elif shape == "tiny_top_p":
        logits = (rng.standard_normal(5000) * 5).astype(np.float32)
    elif shape == "near_one":
        logits = (rng.standard_normal(5000) * 5).astype(np.float32)
    else:
        logits = (rng.standard_normal(5000) * 5).astype(np.float32)
        logits[100:] = -np.inf
    top_p = {"tiny_top_p": 1e-6, "near_one": 0.999999}.get(shape, 0.9)
    _parity_case(logits, _params(top_p=top_p), 0.7)


def test_flat_distribution_bails_out_to_legacy_byte_identical():
    rng = np.random.default_rng(17)
    logits = (rng.standard_normal(50000) * 0.1).astype(np.float32)
    params = _params(top_p=0.9)
    new_ids, new_probs = S.processed_support(logits, params, temperature=0.7)
    ref_ids, ref_probs = _legacy_support(logits, params, temperature=0.7)
    np.testing.assert_array_equal(new_ids, ref_ids)
    np.testing.assert_array_equal(new_probs, ref_probs)


def test_legacy_paths_byte_identical():
    rng = np.random.default_rng(9)
    logits = (rng.standard_normal(4096) * 5).astype(np.float32)
    cases = [
        _params(top_p=1.0),
        _params(top_p=0.9, top_k=50),
        _params(top_p=1.0, top_k=10, min_p=0.01),
    ]
    for params in cases:
        new_ids, new_probs = S.processed_support(logits, params, temperature=0.7)
        ref_ids, ref_probs = _legacy_support(logits, params, temperature=0.7)
        np.testing.assert_array_equal(new_ids, ref_ids)
        np.testing.assert_array_equal(new_probs, ref_probs)
    allowed = lambda tid: tid % 2 == 0
    new_ids, new_probs = S.processed_support(logits, _params(top_p=0.9), temperature=0.7, token_allowed=allowed)
    ref_ids, ref_probs = _legacy_support(logits, _params(top_p=0.9), temperature=0.7, token_allowed=allowed)
    np.testing.assert_array_equal(new_ids, ref_ids)
    np.testing.assert_array_equal(new_probs, ref_probs)


def test_select_token_seeded_determinism():
    rng = np.random.default_rng(13)
    logits = (rng.standard_normal(8192) * 5).astype(np.float32)
    params = _params(top_p=0.9)
    toks_a, toks_b = [], []
    for seq in (toks_a, toks_b):
        state = S.RowSamplingState(prompt_tokens=(1, 2, 3), seed=77, row_index=0)
        for _ in range(32):
            seq.append(S.select_token(logits.copy(), params, state, token_text_for_id=lambda t: "").token_id)
    assert toks_a == toks_b


def test_select_token_empirical_frequencies():
    rng = np.random.default_rng(21)
    logits = (rng.standard_normal(64) * 2).astype(np.float32)
    params = _params(top_p=0.9, temperature=1.0)
    ids, probs = S.processed_support(logits, params, temperature=1.0)
    ref_ids, ref_probs = _legacy_support(logits, params, temperature=1.0)
    np.testing.assert_array_equal(ids, ref_ids)
    counts = np.zeros(64, dtype=np.int64)
    state = S.RowSamplingState(prompt_tokens=(1,), seed=3, row_index=0)
    n = 20000
    for _ in range(n):
        tok = S.select_token(logits.copy(), params, state, token_text_for_id=lambda t: "").token_id
        counts[tok] += 1
    freq = counts[ids] / n
    assert np.max(np.abs(freq - probs)) < 0.02, (freq, probs)