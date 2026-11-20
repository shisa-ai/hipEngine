"""Unit parity for the top-p tie-repair optimization (CPU only).

_apply_probability_filters receives (logit-desc, lower-ID) order; float32
shift rounding in _softmax creates new probability ties, which used to trigger
a full-vocabulary lexsort. The optimization repairs only equal-probability
groups to lower-ID order. These tests pin byte-identical output against the
previous full-lexsort implementation on adversarial tie shapes, and a RED test
asserts the tie path no longer needs np.lexsort.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.generation import sampling as S


def _reference_filter(token_ids, probs, *, top_p, min_p):
    """The previous implementation: full lexsort whenever order is broken."""
    ordered = np.all(
        (probs[:-1] > probs[1:])
        | ((probs[:-1] == probs[1:]) & (token_ids[:-1] <= token_ids[1:]))
    )
    if ordered:
        sorted_ids = token_ids.copy()
        sorted_probs = probs.copy()
    else:
        order = np.lexsort((token_ids, -probs))
        sorted_ids = token_ids[order]
        sorted_probs = probs[order]
    if sorted_ids.size == 0:
        return sorted_ids, sorted_probs
    if top_p < 1.0:
        if top_p <= 0.0:
            keep_count = 1
        else:
            keep_count = int(np.searchsorted(np.cumsum(sorted_probs), top_p, side="left")) + 1
            keep_count = max(1, min(keep_count, sorted_ids.size))
        sorted_ids = sorted_ids[:keep_count]
        sorted_probs = sorted_probs[:keep_count]
    if min_p > 0.0:
        threshold = float(probs.max()) * min_p
        mask = sorted_probs >= threshold
        if np.any(mask):
            sorted_ids = sorted_ids[mask]
            sorted_probs = sorted_probs[mask]
        else:
            sorted_ids = sorted_ids[:1]
            sorted_probs = sorted_probs[:1]
    return sorted_ids.astype(np.int64, copy=False), sorted_probs.astype(np.float64, copy=False)


def _cases():
    rng = np.random.default_rng(20261108)
    vocab = 4096
    # float32 shift-collision ties (the production trigger)
    f32 = rng.normal(0, 5, vocab).astype(np.float32)
    ids32 = S._top_k_candidate_ids(f32 / 0.7, 0)
    probs32 = S._softmax((f32 / 0.7)[ids32])
    # many exact logit ties
    quant = np.round(rng.normal(0, 5, vocab), 1).astype(np.float32)
    idsq = S._top_k_candidate_ids(quant, 0)
    probsq = S._softmax(quant[idsq])
    # uniform
    uni = np.zeros(vocab, dtype=np.float32)
    idsu = S._top_k_candidate_ids(uni, 0)
    probsu = S._softmax(uni[idsu])
    # boundary cut through an exact tie group: ids must be the lowest in the group
    crafted_scores = np.array([10.0, 9.0, 9.0, 9.0, 9.0, 8.0, 7.0], dtype=np.float32)
    idsc = S._top_k_candidate_ids(crafted_scores, 0)
    probsc = S._softmax(crafted_scores[idsc])
    # float64 (no shift ties)
    f64 = rng.normal(0, 5, vocab)
    ids64 = S._top_k_candidate_ids(f64, 0)
    probs64 = S._softmax(f64[ids64])
    bases = [
        ("f32_shift_ties", ids32, probs32),
        ("quantized_ties", idsq, probsq),
        ("uniform", idsu, probsu),
        ("crafted_boundary", idsc, probsc),
        ("f64", ids64, probs64),
        ("single", np.array([7], dtype=np.int64), np.array([1.0])),
        ("empty", np.array([], dtype=np.int64), np.array([], dtype=np.float64)),
    ]
    for name, ids, probs in bases:
        for top_p in (0.0, 0.3, 0.9, 0.999, 1.0):
            for min_p in (0.0, 0.05):
                yield f"{name}_p{top_p}_mp{min_p}", ids, probs, top_p, min_p


@pytest.mark.parametrize("name,ids,probs,top_p,min_p", list(_cases()))
def test_probability_filters_byte_identical_to_lexsort_reference(name, ids, probs, top_p, min_p):
    got_ids, got_probs = S._apply_probability_filters(ids, probs, top_p=top_p, min_p=min_p)
    ref_ids, ref_probs = _reference_filter(ids, probs, top_p=top_p, min_p=min_p)
    np.testing.assert_array_equal(got_ids, ref_ids, err_msg=name)
    np.testing.assert_array_equal(got_probs, ref_probs, err_msg=name)


def test_tie_repair_does_not_need_full_lexsort(monkeypatch):
    """RED before the optimization: the float32 tie path called np.lexsort."""
    rng = np.random.default_rng(7)
    f32 = rng.normal(0, 5, 4096).astype(np.float32)
    ids = S._top_k_candidate_ids(f32 / 0.7, 0)
    probs = S._softmax((f32 / 0.7)[ids])
    assert not np.all(
        (probs[:-1] > probs[1:])
        | ((probs[:-1] == probs[1:]) & (ids[:-1] <= ids[1:]))
    ), "fixture must actually trigger the broken-order path"

    def forbidden(*args, **kwargs):
        raise AssertionError("full-vocabulary lexsort must not run for tie repair")

    monkeypatch.setattr(S.np, "lexsort", forbidden)
    got_ids, got_probs = S._apply_probability_filters(ids, probs, top_p=0.9, min_p=0.0)
    assert got_ids.size >= 1
    assert np.all(np.isfinite(got_probs))


def test_inverted_probs_fall_back_to_full_sort():
    """Robustness: inputs that violate weak monotonicity still sort fully."""
    ids = np.array([5, 3, 9, 1], dtype=np.int64)
    probs = np.array([0.2, 0.5, 0.1, 0.2], dtype=np.float64)  # inversion present
    got_ids, got_probs = S._apply_probability_filters(ids, probs, top_p=1.0, min_p=0.0)
    ref_ids, ref_probs = _reference_filter(ids, probs, top_p=1.0, min_p=0.0)
    np.testing.assert_array_equal(got_ids, ref_ids)
    np.testing.assert_array_equal(got_probs, ref_probs)


def test_select_token_top_p_deterministic_with_float32_ties():
    from hipengine.generation.registry import GenerationRequest
    from hipengine.generation.sampling import RowSamplingState, select_token

    rng = np.random.default_rng(11)
    logits = rng.normal(0, 5, 4096).astype(np.float32)
    req = GenerationRequest(prompts=((1, 2, 3),), max_tokens=8, temperature=0.7,
                            top_p=0.9, ignore_eos=True)
    state = RowSamplingState(prompt_tokens=(1, 2, 3), seed=0, row_index=0)
    first = select_token(logits.copy(), req, state, token_text_for_id=lambda t: "")
    state2 = RowSamplingState(prompt_tokens=(1, 2, 3), seed=0, row_index=0)
    second = select_token(logits.copy(), req, state2, token_text_for_id=lambda t: "")
    assert first.token_id == second.token_id
    assert first.mode != S.SamplingMode.PROCESSED_ARGMAX