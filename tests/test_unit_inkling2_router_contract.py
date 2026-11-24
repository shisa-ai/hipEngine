"""Contract tests for the Inkling 2 expert-route selector.

These pin the semantics the GGUF loader and the device router both rely on:

* selection ranks by sigmoid(logits) + bias (bias is ranking-only);
* exact ties break toward the lower expert id;
* the returned expert ids are sorted ascending;
* the mixing weights are the *unbiased* sigmoid scores, normalized to
  ``route_scale`` (with a 1e-20 floor) in id order;
* the function is deterministic for repeated calls (repeated physical layers).

The function takes logits already clipped to the real expert count; padding
below that boundary is the caller's responsibility and is not re-tested here.
"""

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.inkling2_mini import select_expert_routes


def reference_select(logits, bias, *, top_k, route_scale):
    """Independent, deliberately simple reimplementation used as the oracle."""
    logits = np.asarray(logits, dtype=np.float64)
    scores = 1.0 / (1.0 + np.exp(-logits))
    ranking = scores if bias is None else scores + np.asarray(bias, dtype=np.float64)
    chosen = []
    remaining = list(range(len(ranking)))
    for _ in range(top_k):
        best = max(remaining, key=lambda i: (ranking[i], -i))
        chosen.append(best)
        remaining.remove(best)
    ids = np.array(sorted(chosen), dtype=np.int64)
    mix = scores[ids].astype(np.float64)
    mix = mix / (mix.sum() + 1e-20) * route_scale
    return ids, mix


def test_unbiased_selection_matches_manual_reference():
    logits = np.array([0.4, -1.2, 2.0, 0.1, -3.0, 1.1], dtype=np.float32)
    ids, mix = select_expert_routes(logits, None, top_k=2, route_scale=8.0)
    exp_ids, exp_mix = reference_select(logits, None, top_k=2, route_scale=8.0)
    np.testing.assert_array_equal(ids, exp_ids)
    np.testing.assert_allclose(mix, exp_mix, rtol=1e-6, atol=1e-6)


def test_bias_changes_ranking_but_not_unbiased_mix_weights():
    logits = np.array([0.0, 3.0, 0.0], dtype=np.float32)
    bias = np.array([5.0, 0.0, 0.0], dtype=np.float32)  # pulls expert 0 into the top-2
    ids, mix = select_expert_routes(logits, bias, top_k=2, route_scale=4.0)
    assert list(ids) == [0, 1]  # biased ranking picks experts 0 and 1
    raw = np.array([1 / (1 + np.exp(-0.0)), 1 / (1 + np.exp(-3.0))], dtype=np.float64)
    np.testing.assert_allclose(mix, 4.0 * raw / (raw.sum() + 1e-20), rtol=1e-6)


def test_exact_ties_prefer_lower_expert_id():
    logits = np.zeros(6, dtype=np.float32)
    ids, mix = select_expert_routes(logits, None, top_k=3, route_scale=8.0)
    assert list(ids) == [0, 1, 2]
    # Bias that re-ties two mid-ranking experts must still favor the lower id.
    bias = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ids, _ = select_expert_routes(logits, bias, top_k=2, route_scale=8.0)
    assert list(ids) == [0, 1]


def test_mix_sums_to_route_scale_and_ids_are_sorted():
    rng = np.random.default_rng(20261124)
    for _ in range(64):
        experts = int(rng.integers(4, 24))
        top_k = int(rng.integers(1, experts))
        logits = rng.normal(0, 3, size=experts).astype(np.float32)
        bias = rng.normal(0, 1, size=experts).astype(np.float32)
        ids, mix = select_expert_routes(logits, bias, top_k=top_k, route_scale=8.0)
        assert np.all(np.diff(ids) > 0), "ids must be strictly increasing"
        np.testing.assert_allclose(mix.sum(), 8.0, rtol=1e-5, atol=1e-5)


def test_extreme_logits_saturate_without_nan():
    logits = np.array([1e4, -1e4, 0.0, 1e4, -1e4], dtype=np.float32)
    ids, mix = select_expert_routes(logits, None, top_k=2, route_scale=8.0)
    assert np.all(np.isfinite(mix))
    np.testing.assert_allclose(mix.sum(), 8.0, rtol=1e-4)


def test_repeated_calls_are_deterministic():
    logits = np.linspace(-3, 3, 17, dtype=np.float32)
    bias = np.sin(np.arange(17, dtype=np.float32))
    first = select_expert_routes(logits, bias, top_k=4, route_scale=8.0)
    for _ in range(8):
        ids, mix = select_expert_routes(logits, bias, top_k=4, route_scale=8.0)
        np.testing.assert_array_equal(ids, first[0])
        np.testing.assert_array_equal(mix, first[1])


def test_top_k_equal_to_expert_count_selects_everything():
    logits = np.array([2.0, -1.0, 0.5, 3.0], dtype=np.float32)
    ids, mix = select_expert_routes(logits, None, top_k=4, route_scale=8.0)
    np.testing.assert_array_equal(ids, np.array([0, 1, 2, 3]))
    np.testing.assert_allclose(mix.sum(), 8.0, rtol=1e-6)


def test_batched_rows_match_single_row_calls():
    rng = np.random.default_rng(7)
    logits = rng.normal(0, 2, size=(5, 12)).astype(np.float32)
    bias = rng.normal(0, 0.5, size=12).astype(np.float32)
    ids, mix = select_expert_routes(logits, bias, top_k=3, route_scale=8.0)
    assert ids.shape == (5, 3) and mix.shape == (5, 3)
    for row in range(5):
        rids, rmix = select_expert_routes(logits[row], bias, top_k=3, route_scale=8.0)
        np.testing.assert_array_equal(ids[row], rids)
        np.testing.assert_allclose(mix[row], rmix, rtol=1e-6, atol=1e-6)


def test_fuzz_against_reference():
    rng = np.random.default_rng(41)
    for _ in range(200):
        experts = int(rng.integers(2, 33))
        top_k = int(rng.integers(1, experts + 1))
        logits = rng.normal(0, 4, size=experts).astype(np.float32)
        bias = None if rng.random() < 0.3 else rng.normal(0, 2, size=experts).astype(np.float32)
        ids, mix = select_expert_routes(logits, bias, top_k=top_k, route_scale=8.0)
        exp_ids, exp_mix = reference_select(logits, bias, top_k=top_k, route_scale=8.0)
        np.testing.assert_array_equal(ids, exp_ids)
        np.testing.assert_allclose(mix, exp_mix, rtol=1e-5, atol=1e-5)


def test_invalid_top_k_rejected():
    logits = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError):
        select_expert_routes(logits, None, top_k=0, route_scale=8.0)
    with pytest.raises(ValueError):
        select_expert_routes(logits, None, top_k=5, route_scale=8.0)
