"""Sampler checkpoints preserve draws independently of emitted/forced tokens."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.sampling import RowSamplingState, select_token


def params(temperature=1):
    return SimpleNamespace(temperature=temperature, top_p=1, top_k=0, min_p=0)


@pytest.mark.parametrize('schedule', ['forced', 'greedy', 'mixed', 'draw_only'])
def test_clone_preserves_exact_rng_without_counting_output_steps(schedule):
    state = RowSamplingState(seed=42, prompt_tokens=(2,), forced_tokens_pending=(1, 0))
    if schedule == 'draw_only':
        state.random_unit()
        state.random_unit()
    else:
        for i in range(6):
            select_token([.5, 1, -.5], params(0 if schedule == 'greedy' or (schedule == 'mixed' and i % 2) else 1), state)
    checkpoint = state.clone()
    assert checkpoint._rng.bit_generator.state == state._rng.bit_generator.state
    assert checkpoint.generated_tokens == state.generated_tokens
    assert checkpoint.step_index == state.step_index
    before = deepcopy(checkpoint._rng.bit_generator.state)
    expected = [state.random_unit() for _ in range(8)]
    assert checkpoint._rng.bit_generator.state == before  # independent ownership
    assert [checkpoint.random_unit() for _ in range(8)] == expected


def test_draw_counter_counts_only_uniforms_and_survives_restore():
    state = RowSamplingState(seed=7, forced_tokens_pending=(1, 0))
    for _ in range(2):
        assert select_token([0., 1.], params(), state).forced
    select_token([0., 1.], params(0), state)
    assert state.step_index == 3 and state.rng_draw_count == 0
    expected = np.random.Generator(np.random.PCG64(7))
    assert state.random_unit() == expected.random()
    assert state.step_index == 3 and state.rng_draw_count == 1
    checkpoint = state.clone()
    # Discard a transaction that consumed both uniforms and forced output.
    state.queue_forced_tokens((0,))
    select_token([0., 1.], params(), state)
    select_token([0., 1.], params(), state)
    assert checkpoint.rng_draw_count == 1
    assert checkpoint.step_index == 3 and checkpoint.forced_tokens == ()
    state = checkpoint.clone()
    assert state.random_unit() == expected.random()
    assert state.rng_draw_count == 2


@pytest.mark.parametrize('step,draws', [(0, 5), (8, 0), (8, 3), (8, 8)])
def test_explicit_reconstruction_uses_draw_count_not_step(step, draws):
    state = RowSamplingState(seed=7, step_index=step, rng_draw_count=draws)
    expected = np.random.Generator(np.random.PCG64(7))
    expected.random(draws)
    assert state.rng_draw_count == draws
    assert state.random_unit() == expected.random()
    assert state.rng_draw_count == draws + 1 and state.step_index == step


def test_legacy_step_only_reconstruction_keeps_its_existing_sequence():
    state = RowSamplingState(seed=7, step_index=5)
    expected = np.random.Generator(np.random.PCG64(7))
    expected.random(5)
    assert state.random_unit() == expected.random()
    assert state.rng_draw_count == 6


def test_negative_draw_count_is_rejected():
    with pytest.raises(ValueError, match='rng_draw_count'):
        RowSamplingState(rng_draw_count=-1)
