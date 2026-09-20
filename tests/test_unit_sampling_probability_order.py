"""Avoid a second vocabulary sort without changing probability or tie order."""

import numpy as np
import pytest

from hipengine.generation import sampling


def test_ordered_probabilities_do_not_repeat_the_full_vocabulary_sort(monkeypatch):
    ids = np.array([9, 1, 7, 4], dtype=np.int64)
    probs = np.array([0.5, 0.2, 0.2, 0.1], dtype=np.float64)

    def unexpected_sort(*args, **kwargs):
        raise AssertionError("already ordered probabilities must not repeat lexsort")

    monkeypatch.setattr(np, "lexsort", unexpected_sort)
    selected, weights = sampling._apply_probability_filters(ids, probs, top_p=0.8, min_p=0.0)
    np.testing.assert_array_equal(selected, [9, 1, 7])
    np.testing.assert_array_equal(weights, probs[:3])
    assert not np.shares_memory(selected, ids)
    assert not np.shares_memory(weights, probs)


@pytest.mark.parametrize("top_p", [0.0, 0.5, 0.95, 1.0])
@pytest.mark.parametrize("min_p", [0.0, 0.3, 1.0])
def test_probability_order_is_invariant_to_input_permutations(top_p, min_p):
    ids = np.array([7, 1, 9, 4], dtype=np.int64)
    probs = np.array([0.2, 0.2, 0.5, 0.1], dtype=np.float64)
    expected_order = np.lexsort((ids, -probs))
    expected = sampling._apply_probability_filters(
        ids, probs, top_p=top_p, min_p=min_p,
    )
    actual = sampling._apply_probability_filters(
        ids[expected_order], probs[expected_order], top_p=top_p, min_p=min_p,
    )
    for lhs, rhs in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(lhs, rhs)


def test_equal_probabilities_still_use_lower_id_even_after_logit_sorting():
    # Distinct logits can collapse to equal float64 probabilities. Logit order
    # then differs from the binding probability/lower-ID order.
    scaled = np.array([0.0, 1e-20, -1.0], dtype=np.float64)
    ids = sampling._top_k_candidate_ids(scaled, 0)
    assert ids.tolist() == [1, 0, 2]
    probs = sampling._softmax(scaled[ids])
    assert probs[0] == probs[1]
    selected, _ = sampling._apply_probability_filters(ids, probs, top_p=0.1, min_p=0.0)
    assert selected.tolist() == [0]


@pytest.mark.parametrize("size", [0, 1, 2, 257, 4096])
def test_sorted_probability_fast_path_preserves_exact_arrays_and_input_ownership(size):
    rng = np.random.default_rng(23)
    ids = rng.permutation(size).astype(np.int64)
    # Many ties, including zeros, exercise the complete lexicographic order.
    probs = rng.integers(0, 10, size=size).astype(np.float64)
    if size and probs.sum():
        probs /= probs.sum()
    order = np.lexsort((ids, -probs))
    ids, probs = ids[order], probs[order]
    selected, weights = sampling._apply_probability_filters(ids, probs, top_p=1.0, min_p=0.0)
    np.testing.assert_array_equal(selected, ids)
    np.testing.assert_array_equal(weights, probs)
    assert not np.shares_memory(selected, ids)
    assert not np.shares_memory(weights, probs)
