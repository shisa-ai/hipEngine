"""Unit: host sampler history penalties stay exact after the incremental-count
+ vectorized-application optimization (MUSE-GLIMMER §11.1 host sampler track).

Guards two contracts:
  * history_counts() equals Counter(prompt + generated) at every step, including
    after observe() and clone() (the cache is maintained incrementally).
  * _apply_history_penalties produces bit-identical logits to the original
    per-token scalar loop for repetition/presence/frequency combinations,
    including out-of-vocab history tokens and negative/positive logits.
"""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np

from hipengine.generation.sampling import RowSamplingState, _apply_history_penalties


def _reference_scalar_penalties(logits, params, state):
    """The original per-token scalar implementation (ground truth)."""
    repetition_penalty = float(getattr(params, "repetition_penalty", 1.0))
    presence_penalty = float(getattr(params, "presence_penalty", 0.0))
    frequency_penalty = float(getattr(params, "frequency_penalty", 0.0))
    if repetition_penalty == 1.0 and presence_penalty == 0.0 and frequency_penalty == 0.0:
        return
    counts = Counter(int(t) for t in (*state.prompt_tokens, *state.generated_tokens))
    if not counts:
        return
    vocab = int(logits.size)
    for token_id, count in counts.items():
        if token_id < 0 or token_id >= vocab:
            continue
        if repetition_penalty != 1.0:
            if logits[token_id] < 0.0:
                logits[token_id] *= repetition_penalty
            else:
                logits[token_id] /= repetition_penalty
        if presence_penalty != 0.0:
            logits[token_id] -= presence_penalty
        if frequency_penalty != 0.0:
            logits[token_id] -= frequency_penalty * int(count)


def test_history_counts_incremental_matches_full_rebuild() -> None:
    prompt = (5, 5, 9, 12, 5, 999999)  # includes an out-of-vocab id
    state = RowSamplingState(prompt_tokens=prompt, seed=0)
    # Before any history_counts() call (cache unbuilt), observe a few tokens.
    state.observe(9)
    state.observe(5)
    assert state.history_counts() == Counter(prompt + (9, 5))
    # Cache now built; further observes must increment it.
    for tok in (12, 12, 7, 999999, 5):
        state.observe(tok)
        expected = Counter(prompt + tuple(state.generated_tokens))
        assert state.history_counts() == expected
    # A clone rebuilds independently and stays consistent under more observes.
    cloned = state.clone()
    assert cloned.history_counts() == state.history_counts()
    cloned.observe(42)
    assert cloned.history_counts() == Counter(prompt + tuple(cloned.generated_tokens))
    assert state.history_counts() == Counter(prompt + tuple(state.generated_tokens))


def test_history_counts_cache_not_shared_with_clone() -> None:
    state = RowSamplingState(prompt_tokens=(1, 2, 2), seed=0)
    state.history_counts()  # build cache
    cloned = state.clone()
    cloned.observe(3)
    assert state.history_counts() == Counter((1, 2, 2))
    assert cloned.history_counts() == Counter((1, 2, 2, 3))


def test_vectorized_penalties_bit_identical_to_scalar() -> None:
    rng = np.random.default_rng(0)
    vocab = 4096
    combos = [
        dict(repetition_penalty=1.1),
        dict(presence_penalty=0.5),
        dict(frequency_penalty=0.25),
        dict(repetition_penalty=1.3, presence_penalty=0.2, frequency_penalty=0.1),
        dict(repetition_penalty=0.8, presence_penalty=-0.4, frequency_penalty=0.7),
    ]
    for combo in combos:
        params = SimpleNamespace(**combo)
        # History with repeats (frequency>1) and out-of-vocab ids.
        prompt = tuple(int(x) for x in rng.integers(0, vocab + 50, 300))
        state = RowSamplingState(prompt_tokens=prompt, seed=0)
        for _ in range(20):
            state.observe(int(rng.integers(0, vocab + 50)))
        state.history_counts()  # exercise the incremental cache path
        base = rng.standard_normal(vocab).astype(np.float32)
        # Ensure both negative and positive logits are present.
        got = base.copy()
        ref = base.copy()
        _apply_history_penalties(got, params, state)
        _reference_scalar_penalties(ref, params, state)
        assert np.array_equal(got, ref), f"penalty mismatch for {combo}"


def test_neutral_penalties_leave_logits_unchanged() -> None:
    params = SimpleNamespace(repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0)
    state = RowSamplingState(prompt_tokens=tuple(range(100)), seed=0)
    logits = np.random.default_rng(1).standard_normal(256).astype(np.float32)
    before = logits.copy()
    _apply_history_penalties(logits, params, state)
    assert np.array_equal(logits, before)