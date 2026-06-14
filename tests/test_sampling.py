from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.sampling import (
    RowSamplingState,
    SamplingMode,
    derive_row_seed,
    normalize_logit_bias_pairs,
    normalize_stop_token_sequences,
    plan_sampler,
    row_seed_for_index,
    select_token,
    supports_native_gpu_sampling,
)


def _params(**overrides):
    values = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "logit_bias": (),
        "seed": None,
        "row_seeds": (),
        "stop_token_ids": (),
        "stop_token_sequences": (),
        "logprobs": False,
        "top_logprobs": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_sampler_plan_keeps_inert_top_p_top_k_on_greedy_fast_path() -> None:
    plan = plan_sampler(_params(temperature=0.0, top_p=0.5, top_k=4, min_p=0.5))

    assert plan.mode is SamplingMode.GREEDY_FAST
    assert plan.active_processors == ()


def test_sampler_plan_uses_processed_argmax_for_active_processors() -> None:
    plan = plan_sampler(_params(temperature=0.0, presence_penalty=1.0))

    assert plan.mode is SamplingMode.PROCESSED_ARGMAX
    assert plan.active_processors == ("presence_penalty",)


def test_sampler_plan_uses_processed_argmax_for_logprobs() -> None:
    plan = plan_sampler(_params(temperature=0.0, logprobs=True, top_logprobs=2))

    assert plan.mode is SamplingMode.PROCESSED_ARGMAX


def test_stop_token_sequences_are_active_processors() -> None:
    plan = plan_sampler(_params(temperature=0.0, stop_token_sequences=((10, 11),)))

    assert plan.mode is SamplingMode.PROCESSED_ARGMAX
    assert plan.active_processors == ("stop_token_sequences",)
    assert normalize_stop_token_sequences([[10, 11], [10, 11], []]) == ((10, 11),)


def test_sampler_plan_uses_host_logits_for_non_greedy_without_gpu_sampler() -> None:
    plan = plan_sampler(_params(temperature=0.7, top_p=0.9))

    assert plan.mode is SamplingMode.HOST_LOGITS_SAMPLE


def test_sampler_plan_uses_gpu_sample_for_native_supported_request() -> None:
    params = _params(temperature=0.7, top_k=64, logprobs=True)
    plan = plan_sampler(params, native_gpu_available=True)

    assert supports_native_gpu_sampling(params) is True
    assert plan.mode is SamplingMode.GPU_SAMPLE
    assert plan.native_gpu_available is True


def test_sampler_plan_allows_native_gpu_sample_with_supported_processors() -> None:
    params = _params(
        temperature=0.7,
        top_k=4,
        logit_bias=((1, 2.0),),
        repetition_penalty=1.2,
        presence_penalty=0.25,
        frequency_penalty=0.1,
    )
    plan = plan_sampler(params, native_gpu_available=True)

    assert supports_native_gpu_sampling(params) is True
    assert plan.mode is SamplingMode.GPU_SAMPLE
    assert plan.active_processors == (
        "logit_bias",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    )


def test_native_gpu_sampler_support_rejects_unwired_shapes() -> None:
    assert supports_native_gpu_sampling(_params(temperature=0.0)) is False
    assert supports_native_gpu_sampling(_params(temperature=0.7, top_k=65)) is False
    assert supports_native_gpu_sampling(_params(temperature=0.7, top_k=4, top_p=0.9)) is False
    assert supports_native_gpu_sampling(_params(temperature=0.7, top_logprobs=1)) is False
    assert plan_sampler(_params(temperature=0.7, top_k=65), native_gpu_available=True).mode is SamplingMode.HOST_LOGITS_SAMPLE


def test_greedy_tie_break_selects_lower_token_id() -> None:
    result = select_token(np.array([1.0, 3.0, 3.0, 2.0], dtype=np.float32), _params())

    assert result.token_id == 1
    assert result.logit == 3.0
    assert result.logprob is None
    assert result.mode is SamplingMode.GREEDY_FAST


def test_processed_argmax_reports_requested_logprobs() -> None:
    result = select_token(
        np.array([1.0, 3.0, 3.0, 2.0], dtype=np.float32),
        _params(logprobs=True, top_logprobs=2),
    )

    assert result.token_id == 1
    assert result.logprob is not None
    assert result.mode is SamplingMode.PROCESSED_ARGMAX
    assert result.top_logprobs[0][0] == 1
    assert len(result.top_logprobs) == 2


def test_logit_bias_and_penalties_apply_before_processed_argmax() -> None:
    state = RowSamplingState(prompt_tokens=(0, 0))
    result = select_token(
        np.array([5.0, 4.0], dtype=np.float32),
        _params(temperature=0.0, presence_penalty=1.0, frequency_penalty=1.0),
        state,
    )

    assert result.token_id == 1
    assert result.mode is SamplingMode.PROCESSED_ARGMAX
    assert state.generated_tokens == [1]


def test_repetition_penalty_and_logit_bias_share_documented_order() -> None:
    state = RowSamplingState(prompt_tokens=(0,))
    result = select_token(
        np.array([2.0, 1.5], dtype=np.float32),
        _params(temperature=0.0, repetition_penalty=2.0, logit_bias={"1": 1.0}),
        state,
    )

    assert result.token_id == 1
    assert normalize_logit_bias_pairs({"1": 1.0}) == ((1, 1.0),)


def test_temperature_sampling_is_fixed_seed_deterministic() -> None:
    logits = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    params = _params(temperature=0.8, seed=123)
    first = select_token(logits, params, RowSamplingState(seed=123))
    second = select_token(logits, params, RowSamplingState(seed=123))

    assert first == second
    assert first.logprob is not None


def test_top_k_filter_limits_candidate_set() -> None:
    result = select_token(
        np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        _params(temperature=1.0, top_k=2, seed=5),
        RowSamplingState(seed=5),
    )

    assert result.token_id in {2, 3}
    assert result.candidate_count == 2


def test_top_p_and_min_p_retain_at_least_one_candidate() -> None:
    result = select_token(
        np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32),
        _params(temperature=1.0, top_p=0.0, min_p=1.0, seed=7),
        RowSamplingState(seed=7),
    )

    assert result.token_id == 0
    assert result.candidate_count == 1


def test_top_p_keeps_minimal_nucleus() -> None:
    result = select_token(
        np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32),
        _params(temperature=1.0, top_p=0.7, seed=11),
        RowSamplingState(seed=11),
    )

    assert result.token_id in {0, 1}
    assert result.candidate_count == 2


def test_nonfinite_logits_are_rejected_when_no_finite_values_remain() -> None:
    with pytest.raises(ValueError, match="no finite"):
        select_token(np.array([float("nan"), float("inf")], dtype=np.float32), _params())


def test_row_seed_derivation_is_stable_and_uses_explicit_row_seed_first() -> None:
    params = _params(seed=123, row_seeds=(99,))

    assert row_seed_for_index(params, 0) == 99
    assert row_seed_for_index(params, 1) == derive_row_seed(123, 1)
    assert derive_row_seed(123, 1) == derive_row_seed(123, 1)
    assert derive_row_seed(123, 1) != derive_row_seed(123, 2)
