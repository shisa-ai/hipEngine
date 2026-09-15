"""The solver's alpha-bar tables are cached without changing their values.

The cumulative product in ``_alpha_bar_tables`` is a Python-level loop with
fp32 scalar arithmetic that mirrors torch's sequential fp32 accumulation, so it
cannot be vectorised. It depends only on ``num_train_timesteps``; rebuilding it
per scheduler instance cost the TTS session ~524 us on every diffusion frame.
These tests pin that the cached tables are bit-identical to a fresh computation
and that the scheduler's observable state is unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.vibevoice_tts_diffusion import (
    _ALPHA_BAR_TABLES,
    DPMSolverMultistepScheduler,
    _alpha_bar_tables,
    betas_for_alpha_bar,
)


class _Spec:
    num_train_timesteps = 1000
    num_inference_steps = 20
    prediction_type = "v_prediction"
    algorithm_type = "dpmsolver++"
    solver_order = 2
    solver_type = "midpoint"
    lower_order_final = True


def _reference_tables(num_train_timesteps: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent recomputation of the same values, never touching the cache."""
    betas = betas_for_alpha_bar(num_train_timesteps)
    alphas = 1.0 - betas
    ac = np.empty_like(alphas)
    acc = np.float32(1.0)
    for i, a in enumerate(alphas):
        acc = np.float32(acc * a)
        ac[i] = acc
    return ac, (((1.0 - ac) / ac) ** 0.5).astype(np.float32)


@pytest.fixture(autouse=True)
def _clear_cache():
    _ALPHA_BAR_TABLES.clear()
    yield
    _ALPHA_BAR_TABLES.clear()


def test_cached_tables_are_bit_identical_to_a_fresh_computation():
    ref_ac, ref_sigmas = _reference_tables(1000)
    ac, sigmas = _alpha_bar_tables(1000)
    assert np.array_equal(ac, ref_ac)
    assert np.array_equal(sigmas, ref_sigmas)
    # A second call must return the same objects, not recompute them.
    ac2, sigmas2 = _alpha_bar_tables(1000)
    assert ac2 is ac and sigmas2 is sigmas


def test_scheduler_state_matches_an_uncached_construction():
    ref_ac, ref_sigmas = _reference_tables(1000)
    sched = DPMSolverMultistepScheduler(_Spec())
    assert np.array_equal(sched.alphas_cumprod, ref_ac)
    assert np.array_equal(sched.init_sigmas, ref_sigmas)
    sched.set_timesteps(20)
    assert sched._step_index == 0
    assert sched.lower_order_nums == 0
    assert sched.model_outputs == [None] * 2


def test_repeated_construction_reuses_one_table_pair():
    first = _alpha_bar_tables(1000)
    for _ in range(25):  # one 25-frame TTS request
        DPMSolverMultistepScheduler(_Spec())
    assert len(_ALPHA_BAR_TABLES) == 1
    assert _alpha_bar_tables(1000)[0] is first[0]


def test_a_different_schedule_gets_its_own_tables():
    ac_a, _ = _alpha_bar_tables(1000)
    ac_b, _ = _alpha_bar_tables(500)
    assert ac_a.shape != ac_b.shape or not np.array_equal(ac_a, ac_b)
    assert len(_ALPHA_BAR_TABLES) == 2
