from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.sampling import (
    ForcedTokenQueue,
    JsonObjectConstraintState,
    NATIVE_GPU_SAMPLER_UNSUPPORTED_CAPABILITIES,
    RowSamplingState,
    SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS,
    SPECULATIVE_MTP_INCOMPATIBLE_FIELDS,
    SamplingMode,
    ThinkingBudgetState,
    derive_row_seed,
    normalize_logit_bias_pairs,
    normalize_stop_token_sequences,
    plan_sampler,
    row_seed_for_index,
    sampler_fast_path_blockers,
    select_token,
    speculative_mtp_sampling_blockers,
    thinking_budget_active,
    thinking_budget_state_from_params,
    supports_native_gpu_sampling,
    supports_speculative_mtp_sampling,
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
        "suppress_token_ids": (),
        "min_tokens": 0,
        "eos_token_id": None,
        "ignore_eos": False,
        "seed": None,
        "row_seeds": (),
        "stop_token_ids": (),
        "stop_token_sequences": (),
        "forced_tokens_pending": (),
        "forced_token_reason": None,
        "post_thinking_forced_tokens_pending": (),
        "post_thinking_forced_token_reason": None,
        "force_sequence_completion_token_sequences": (),
        "force_sequence_completion_reason": None,
        "json_object_close_forcing": False,
        "thinking_close_token_ids": (),
        "thinking_hard_token_cap": None,
        "thinking_soft_close_window": 0,
        "logprobs": False,
        "top_logprobs": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _speculative_mtp_blocker_cases():
    return {
        "temperature": _params(temperature=0.7),
        "logit_bias": _params(logit_bias={1: 10.0}),
        "repetition_penalty": _params(repetition_penalty=1.1),
        "presence_penalty": _params(presence_penalty=0.1),
        "frequency_penalty": _params(frequency_penalty=0.1),
        "suppress_token_ids": _params(suppress_token_ids=(7,)),
        "min_tokens": _params(min_tokens=2, eos_token_id=9),
        "eos_token_id": _params(eos_token_id=9),
        "ignore_eos": _params(ignore_eos=True),
        "stop_token_ids": _params(stop_token_ids=(99,)),
        "stop_token_sequences": _params(stop_token_sequences=((10, 11),)),
        "forced_tokens_pending": _params(forced_tokens_pending=(10, 11)),
        "post_thinking_forced_tokens_pending": _params(
            thinking_close_token_ids=(10, 11),
            thinking_hard_token_cap=4,
            post_thinking_forced_tokens_pending=(12,),
        ),
        "force_sequence_completion_token_sequences": _params(
            force_sequence_completion_token_sequences=((10, 11),),
        ),
        "json_object_close_forcing": _params(json_object_close_forcing=True),
        "thinking_budget": _params(thinking_close_token_ids=(10, 11), thinking_hard_token_cap=2),
        "logprobs": _params(logprobs=True),
        "top_logprobs": _params(top_logprobs=2),
    }


def _native_gpu_sampler_guard_cases():
    return {
        "top_logprobs_exceed_native_limit": _params(temperature=0.7, top_logprobs=65),
        "top_logprobs_exceed_top_k": _params(temperature=0.7, top_k=4, top_logprobs=5),
        "forced_tokens_pending": _params(temperature=0.7, forced_tokens_pending=(10,)),
        "post_thinking_forced_tokens_pending": _params(
            temperature=0.7,
            thinking_close_token_ids=(10, 11),
            thinking_hard_token_cap=4,
            post_thinking_forced_tokens_pending=(12,),
        ),
        "force_sequence_completion_token_sequences": _params(
            temperature=0.7,
            force_sequence_completion_token_sequences=((10, 11),),
        ),
        "json_object_close_forcing": _params(temperature=0.7, json_object_close_forcing=True),
        "thinking_budget": _params(
            temperature=0.7,
            thinking_close_token_ids=(10, 11),
            thinking_hard_token_cap=2,
        ),
    }


def test_sampler_plan_keeps_inert_top_p_top_k_on_greedy_fast_path() -> None:
    plan = plan_sampler(_params(temperature=0.0, top_p=0.5, top_k=4, min_p=0.5))

    assert plan.mode is SamplingMode.GREEDY_FAST
    assert plan.active_processors == ()
    assert plan.fast_path_blockers == ()
    assert plan.fallback_reason is None


def test_sampler_plan_uses_processed_argmax_for_active_processors() -> None:
    plan = plan_sampler(_params(temperature=0.0, presence_penalty=1.0))

    assert plan.mode is SamplingMode.PROCESSED_ARGMAX
    assert plan.active_processors == ("presence_penalty",)
    assert plan.fast_path_blockers == ("presence_penalty",)
    assert plan.fallback_reason == "processed_logits_required"


def test_sampler_plan_uses_processed_argmax_for_logprobs() -> None:
    plan = plan_sampler(_params(temperature=0.0, logprobs=True, top_logprobs=2))

    assert plan.mode is SamplingMode.PROCESSED_ARGMAX
    assert plan.fast_path_blockers == ("logprobs", "top_logprobs")
    assert plan.fallback_reason == "processed_logits_required"


def test_stop_token_sequences_are_active_processors() -> None:
    plan = plan_sampler(_params(temperature=0.0, stop_token_sequences=((10, 11),)))

    assert plan.mode is SamplingMode.PROCESSED_ARGMAX
    assert plan.active_processors == ("stop_token_sequences",)
    assert normalize_stop_token_sequences([[10, 11], [10, 11], []]) == ((10, 11),)


def test_forced_tokens_are_active_processors() -> None:
    plan = plan_sampler(_params(temperature=0.0, forced_tokens_pending=(10, 11)))

    assert plan.mode is SamplingMode.PROCESSED_ARGMAX
    assert plan.active_processors == ("forced_tokens_pending",)
    assert sampler_fast_path_blockers(_params(temperature=0.7, logit_bias={1: 2.0})) == (
        "temperature",
        "logit_bias",
    )


def test_force_sequence_completion_is_active_processor() -> None:
    plan = plan_sampler(
        _params(
            temperature=0.0,
            force_sequence_completion_token_sequences=((10, 11),),
        )
    )

    assert plan.mode is SamplingMode.PROCESSED_ARGMAX
    assert plan.active_processors == ("force_sequence_completion_token_sequences",)
    assert plan.fast_path_blockers == ("force_sequence_completion_token_sequences",)


def test_suppressions_and_min_tokens_are_active_processors() -> None:
    params = _params(temperature=0.0, suppress_token_ids=(3,), min_tokens=2, eos_token_id=4)
    plan = plan_sampler(params)

    assert plan.mode is SamplingMode.PROCESSED_ARGMAX
    assert plan.active_processors == ("suppress_token_ids", "min_tokens")
    assert plan.fast_path_blockers == ("suppress_token_ids", "min_tokens")


def test_sampler_plan_uses_host_logits_for_non_greedy_without_gpu_sampler() -> None:
    plan = plan_sampler(_params(temperature=0.7, top_p=0.9))

    assert plan.mode is SamplingMode.HOST_LOGITS_SAMPLE
    assert plan.fallback_reason == "host_sampling_required"


def test_sampler_plan_uses_gpu_sample_for_native_supported_request() -> None:
    params = _params(temperature=0.7, top_k=64, logprobs=True)
    plan = plan_sampler(params, native_gpu_available=True)

    assert supports_native_gpu_sampling(params) is True
    assert plan.mode is SamplingMode.GPU_SAMPLE
    assert plan.native_gpu_available is True
    assert plan.fallback_reason is None


def test_sampler_plan_uses_gpu_sample_for_bounded_top_logprobs() -> None:
    params = _params(temperature=0.7, top_k=4, top_logprobs=2)
    plan = plan_sampler(params, native_gpu_available=True)

    assert supports_native_gpu_sampling(params) is True
    assert plan.mode is SamplingMode.GPU_SAMPLE
    assert plan.fast_path_blockers == ("temperature", "top_logprobs")


def test_sampler_plan_uses_gpu_sample_for_full_vocab_top_logprobs() -> None:
    params = _params(temperature=0.7, top_k=0, top_logprobs=2)
    plan = plan_sampler(params, native_gpu_available=True)

    assert supports_native_gpu_sampling(params) is True
    assert plan.mode is SamplingMode.GPU_SAMPLE
    assert plan.fast_path_blockers == ("temperature", "top_logprobs")


def test_sampler_plan_uses_gpu_sample_for_bounded_top_k_probability_filters() -> None:
    params = _params(temperature=0.7, top_k=4, top_p=0.9, min_p=0.05)
    plan = plan_sampler(params, native_gpu_available=True)

    assert supports_native_gpu_sampling(params) is True
    assert plan.mode is SamplingMode.GPU_SAMPLE
    assert plan.fast_path_blockers == ("temperature",)


def test_sampler_plan_allows_native_gpu_sample_with_supported_processors() -> None:
    params = _params(
        temperature=0.7,
        top_k=4,
        logit_bias=((1, 2.0),),
        suppress_token_ids=(3,),
        min_tokens=2,
        eos_token_id=4,
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
        "suppress_token_ids",
        "min_tokens",
    )


def test_native_gpu_sampler_support_rejects_unwired_shapes() -> None:
    assert supports_native_gpu_sampling(_params(temperature=0.0)) is False
    assert supports_native_gpu_sampling(_params(temperature=0.7, top_k=65)) is False
    assert supports_native_gpu_sampling(_params(temperature=0.7, top_k=4, top_p=0.9)) is True
    assert supports_native_gpu_sampling(_params(temperature=0.7, top_logprobs=1)) is True
    assert supports_native_gpu_sampling(_params(temperature=0.7, top_logprobs=65)) is False
    assert supports_native_gpu_sampling(_params(temperature=0.7, top_k=4, top_logprobs=5)) is False
    assert supports_native_gpu_sampling(_params(temperature=0.7, suppress_token_ids=(1,))) is True
    assert supports_native_gpu_sampling(_params(temperature=0.7, min_tokens=1, eos_token_id=2)) is True
    assert supports_native_gpu_sampling(_params(temperature=0.7, forced_tokens_pending=(1,))) is False
    assert (
        supports_native_gpu_sampling(_params(temperature=0.7, force_sequence_completion_token_sequences=((1, 2),)))
        is False
    )
    assert supports_native_gpu_sampling(_params(temperature=0.7, json_object_close_forcing=True)) is False
    assert (
        supports_native_gpu_sampling(
            _params(temperature=0.7, thinking_close_token_ids=(4, 5), thinking_hard_token_cap=8)
        )
        is False
    )
    assert plan_sampler(_params(temperature=0.7, top_k=65), native_gpu_available=True).mode is SamplingMode.HOST_LOGITS_SAMPLE
    assert (
        plan_sampler(_params(temperature=0.7, top_k=65), native_gpu_available=True).fallback_reason
        == "native_gpu_unsupported_request"
    )
    assert (
        plan_sampler(_params(temperature=0.7, forced_tokens_pending=(1,)), native_gpu_available=True).mode
        is SamplingMode.HOST_LOGITS_SAMPLE
    )


def test_native_gpu_unsupported_capabilities_match_guard_policy() -> None:
    cases = _native_gpu_sampler_guard_cases()
    advertised = set(NATIVE_GPU_SAMPLER_UNSUPPORTED_CAPABILITIES)

    assert set(cases) <= advertised
    assert {"true_batched_c_gt_1", "gguf"} <= advertised
    assert "logit_bias" not in advertised
    assert "repetition_penalty" not in advertised

    for params in cases.values():
        assert supports_native_gpu_sampling(params) is False
        plan = plan_sampler(params, native_gpu_available=True)
        assert plan.mode is SamplingMode.HOST_LOGITS_SAMPLE
        assert plan.fallback_reason == "native_gpu_unsupported_request"


def test_sampler_plan_reports_requested_native_gpu_unavailable() -> None:
    plan = plan_sampler(_params(temperature=0.7, top_k=4), native_gpu_requested=True)

    assert plan.mode is SamplingMode.HOST_LOGITS_SAMPLE
    assert plan.native_gpu_available is False
    assert plan.fallback_reason == "native_gpu_unsupported_request"


def test_speculative_mtp_sampling_allows_only_greedy_fast_policy() -> None:
    assert tuple(SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS) == SPECULATIVE_MTP_INCOMPATIBLE_FIELDS
    assert SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS["temperature"] == "temperature > 0"

    greedy_inert = _params(temperature=0.0, top_p=0.1, top_k=4, min_p=0.5)
    assert supports_speculative_mtp_sampling(greedy_inert) is True
    assert speculative_mtp_sampling_blockers(greedy_inert) == ()

    assert supports_speculative_mtp_sampling(_params(logit_bias={1: 10.0})) is False
    assert speculative_mtp_sampling_blockers(_params(logit_bias={1: 10.0})) == (
        "logit_bias",
    )
    assert speculative_mtp_sampling_blockers(_params(presence_penalty=0.1)) == (
        "presence_penalty",
    )
    assert speculative_mtp_sampling_blockers(_params(suppress_token_ids=(7,))) == (
        "suppress_token_ids",
    )
    assert speculative_mtp_sampling_blockers(_params(min_tokens=2, eos_token_id=9)) == (
        "min_tokens",
        "eos_token_id",
    )
    assert speculative_mtp_sampling_blockers(_params(eos_token_id=9)) == (
        "eos_token_id",
    )
    assert speculative_mtp_sampling_blockers(_params(ignore_eos=True)) == (
        "ignore_eos",
    )
    assert speculative_mtp_sampling_blockers(_params(stop_token_sequences=((10, 11),))) == (
        "stop_token_sequences",
    )
    assert speculative_mtp_sampling_blockers(_params(forced_tokens_pending=(10, 11))) == (
        "forced_tokens_pending",
    )
    assert speculative_mtp_sampling_blockers(_params(force_sequence_completion_token_sequences=((10, 11),))) == (
        "force_sequence_completion_token_sequences",
    )
    assert speculative_mtp_sampling_blockers(_params(json_object_close_forcing=True)) == (
        "json_object_close_forcing",
    )
    assert speculative_mtp_sampling_blockers(
        _params(
            thinking_close_token_ids=(10, 11),
            thinking_hard_token_cap=4,
            post_thinking_forced_tokens_pending=(12,),
        )
    ) == (
        "thinking_budget",
        "post_thinking_forced_tokens_pending",
    )
    thinking_budget = _params(thinking_close_token_ids=(10, 11), thinking_hard_token_cap=2)
    assert thinking_budget_active(thinking_budget) is True
    assert thinking_budget_state_from_params(thinking_budget).close_sequence == (10, 11)
    assert speculative_mtp_sampling_blockers(thinking_budget) == ("thinking_budget",)
    assert speculative_mtp_sampling_blockers(
        SimpleNamespace(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            logit_bias=(),
            seed=None,
            row_seeds=(),
            stop_tokens=(99,),
            stop_token_sequences=(),
            ignore_eos=False,
            thinking_close_token_ids=(),
            thinking_hard_token_cap=None,
            thinking_soft_close_window=0,
            logprobs=False,
            top_logprobs=0,
        )
    ) == ("stop_token_ids",)
    assert speculative_mtp_sampling_blockers(_params(temperature=0.7, logprobs=True, top_logprobs=2)) == (
        "temperature",
        "logprobs",
        "top_logprobs",
    )


def test_speculative_mtp_incompatible_fields_match_blocker_policy() -> None:
    cases = _speculative_mtp_blocker_cases()
    advertised = set(SPECULATIVE_MTP_INCOMPATIBLE_FIELDS)

    assert set(cases) == advertised
    assert tuple(SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS) == SPECULATIVE_MTP_INCOMPATIBLE_FIELDS

    for field, params in cases.items():
        blockers = speculative_mtp_sampling_blockers(params)
        assert field in blockers
        assert set(blockers) <= advertised
        assert supports_speculative_mtp_sampling(params) is False


def test_greedy_tie_break_selects_lower_token_id() -> None:
    result = select_token(np.array([1.0, 3.0, 3.0, 2.0], dtype=np.float32), _params())

    assert result.token_id == 1
    assert result.logit == 3.0
    assert result.logprob is None
    assert result.mode is SamplingMode.GREEDY_FAST
    assert result.active_processors == ()
    assert result.fast_path_blockers == ()


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


def test_forced_token_queue_overrides_argmax_and_updates_history() -> None:
    state = RowSamplingState(
        forced_tokens_pending=ForcedTokenQueue((2, 1), reason="close_think"),
    )

    first = select_token(
        np.array([1.0, 10.0, 0.5], dtype=np.float32),
        _params(temperature=0.0),
        state,
    )
    second = select_token(
        np.array([1.0, 10.0, 0.5], dtype=np.float32),
        _params(temperature=0.0),
        state,
    )

    assert first.token_id == 2
    assert first.forced is True
    assert first.forced_reason == "close_think"
    assert first.forced_tokens_remaining == 1
    assert first.active_processors == ("forced_tokens_pending",)
    assert first.fast_path_blockers == ("forced_tokens_pending",)
    assert second.token_id == 1
    assert second.forced is True
    assert second.forced_reason == "close_think"
    assert second.forced_tokens_remaining == 0
    assert state.generated_tokens == [2, 1]
    assert state.forced_tokens == ()


def test_request_forced_tokens_seed_default_row_state() -> None:
    result = select_token(
        np.array([10.0, 9.0, 1.0], dtype=np.float32),
        _params(forced_tokens_pending=(2,), forced_token_reason="tool_choice_required"),
    )

    assert result.token_id == 2
    assert result.forced is True
    assert result.forced_reason == "tool_choice_required"
    assert result.active_processors == ("forced_tokens_pending",)
    assert result.fast_path_blockers == ("forced_tokens_pending",)


def test_row_sampling_state_tracks_stop_suffix_state() -> None:
    params = _params(stop_token_sequences=((5, 6), (5, 7)))
    state = RowSamplingState(stop_token_sequences=params.stop_token_sequences)

    first = select_token(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 0.0, 0.0], dtype=np.float32),
        params,
        state,
    )
    assert first.token_id == 5
    assert state.stop_suffix_state == {
        "partial_suffix": [5],
        "candidate_sequences": [[5, 6], [5, 7]],
    }

    second = select_token(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 10.0, 0.0], dtype=np.float32),
        params,
        state,
    )
    assert second.token_id == 6
    assert state.stop_suffix_state == {"matched_sequence": [5, 6]}


def test_forced_token_queue_overrides_sampling() -> None:
    forced = RowSamplingState(seed=123, forced_tokens_pending=(3,), forced_token_reason="grammar")
    forced_result = select_token(
        np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        _params(temperature=0.8, seed=123),
        forced,
    )

    assert forced_result.token_id == 3
    assert forced_result.mode is SamplingMode.HOST_LOGITS_SAMPLE
    assert forced_result.forced is True
    assert forced_result.forced_reason == "grammar"
    assert forced.generated_tokens == [3]


def test_forced_token_queue_overrides_suppression() -> None:
    state = RowSamplingState(forced_tokens_pending=(1,), forced_token_reason="close")
    result = select_token(
        np.array([0.0, 10.0, 2.0], dtype=np.float32),
        _params(temperature=0.0, suppress_token_ids=(1,)),
        state,
    )

    assert result.token_id == 1
    assert result.forced is True
    assert result.logprob is None
    assert result.active_processors == ("suppress_token_ids", "forced_tokens_pending")


def test_json_object_close_forcing_queues_suffix_at_budget_boundary() -> None:
    state = RowSamplingState(json_object_close_forcing=True)
    state.observe_text_for_json_object_close(
        "{",
        remaining_tokens=1,
        encode_text=lambda text: (3,) if text == "}" else (),
    )

    result = select_token(
        np.array([10.0, 9.0, 8.0, 0.0], dtype=np.float32),
        _params(json_object_close_forcing=True),
        state,
    )

    assert result.token_id == 3
    assert result.forced is True
    assert result.forced_reason == "json_object_close_forcing"
    assert result.active_processors == ("json_object_close_forcing", "forced_tokens_pending")
    assert result.fast_path_blockers == ("json_object_close_forcing", "forced_tokens_pending")


def test_json_object_close_forcing_queues_value_string_repair_at_budget_boundary() -> None:
    state = RowSamplingState(json_object_close_forcing=True)
    state.observe_text_for_json_object_close(
        '{"path":"README',
        remaining_tokens=2,
        encode_text=lambda text: (3, 4) if text == '"}' else (),
    )

    first = select_token(
        np.array([10.0, 9.0, 8.0, 0.0, -1.0], dtype=np.float32),
        _params(json_object_close_forcing=True),
        state,
    )
    second = select_token(
        np.array([10.0, 9.0, 8.0, 0.0, -1.0], dtype=np.float32),
        _params(json_object_close_forcing=True),
        state,
    )

    assert [first.token_id, second.token_id] == [3, 4]
    assert first.forced_reason == "json_object_close_forcing"
    assert first.forced_tokens_remaining == 1
    assert second.forced_reason == "json_object_close_forcing"
    assert second.forced_tokens_remaining == 0


def test_json_object_close_forcing_does_not_queue_unparseable_suffix() -> None:
    state = RowSamplingState(json_object_close_forcing=True)
    state.observe_text_for_json_object_close(
        '{"path":',
        remaining_tokens=1,
        encode_text=lambda text: (3,) if text == "}" else (),
    )

    result = select_token(
        np.array([10.0, 9.0, 8.0, 0.0], dtype=np.float32),
        _params(json_object_close_forcing=True),
        state,
    )

    assert result.token_id == 0
    assert result.forced is False
    assert result.active_processors == ("json_object_close_forcing",)
    assert result.fast_path_blockers == ("json_object_close_forcing",)


def test_json_object_close_forcing_suppresses_eos_until_complete() -> None:
    result = select_token(
        np.array([10.0, 9.0, 8.0], dtype=np.float32),
        _params(json_object_close_forcing=True, eos_token_id=0),
        RowSamplingState(json_object_close_forcing=True),
    )

    assert result.token_id == 1
    assert result.active_processors == ("json_object_close_forcing",)
    assert result.fast_path_blockers == ("json_object_close_forcing",)


def test_thinking_budget_hard_close_queues_forced_tokens_before_selection() -> None:
    state = RowSamplingState(
        thinking_budget=ThinkingBudgetState(close_sequence=(2, 1), hard_token_cap=1),
    )

    natural = select_token(np.array([5.0, 1.0, 0.0], dtype=np.float32), _params(), state)
    first_close = select_token(np.array([5.0, 1.0, 0.0], dtype=np.float32), _params(), state)
    second_close = select_token(np.array([5.0, 1.0, 0.0], dtype=np.float32), _params(), state)

    assert natural.token_id == 0
    assert natural.forced is False
    assert first_close.token_id == 2
    assert first_close.forced is True
    assert first_close.forced_reason == "thinking_hard_close"
    assert first_close.active_processors == ("forced_tokens_pending",)
    assert second_close.token_id == 1
    assert second_close.forced is True
    assert state.generated_tokens == [0, 2, 1]
    assert state.forced_tokens == ()
    assert state.thinking_budget is not None
    assert state.thinking_budget.phase == "answer"
    assert state.thinking_budget.reasoning_tokens == 3


def test_thinking_budget_manual_force_close_queues_controller_tokens() -> None:
    budget = ThinkingBudgetState(close_sequence=(2, 1), hard_token_cap=None)
    state = RowSamplingState(thinking_budget=budget)

    assert budget.force_close(reason="controller_close") is True
    first_close = select_token(np.array([5.0, 1.0, 0.0], dtype=np.float32), _params(), state)
    second_close = select_token(np.array([5.0, 1.0, 0.0], dtype=np.float32), _params(), state)

    assert first_close.token_id == 2
    assert first_close.forced is True
    assert first_close.forced_reason == "controller_close"
    assert first_close.forced_tokens_remaining == 1
    assert second_close.token_id == 1
    assert second_close.forced is True
    assert second_close.forced_reason == "controller_close"
    assert second_close.forced_tokens_remaining == 0
    assert state.generated_tokens == [2, 1]
    assert budget.phase == "answer"
    assert budget.reasoning_tokens == 2
    assert budget.force_close(reason="late_controller_close") is False


def test_thinking_budget_soft_close_bias_changes_argmax_and_forces_suffix() -> None:
    state = RowSamplingState(
        thinking_budget=ThinkingBudgetState(
            close_sequence=(2, 1),
            hard_token_cap=5,
            soft_close_window=2,
        ),
    )
    state.observe(7)
    state.observe(8)
    state.observe(9)

    first_close = select_token(np.array([2.0, 1.0, -1.0], dtype=np.float32), _params(), state)
    second_close = select_token(np.array([2.0, 0.0, -1.0], dtype=np.float32), _params(), state)

    assert first_close.token_id == 2
    assert first_close.logit == 3.0
    assert first_close.forced is False
    assert first_close.mode is SamplingMode.PROCESSED_ARGMAX
    assert first_close.active_processors == ("thinking_budget",)
    assert first_close.fast_path_blockers == ("thinking_budget",)
    assert second_close.token_id == 1
    assert second_close.forced is True
    assert second_close.forced_reason == "thinking_soft_close"
    assert state.generated_tokens == [7, 8, 9, 2, 1]
    assert state.forced_tokens == ()
    assert state.thinking_budget is not None
    assert state.thinking_budget.phase == "answer"


def test_thinking_budget_suppresses_eos_until_answer_phase() -> None:
    params = _params(
        eos_token_id=1,
        thinking_close_token_ids=(2,),
        thinking_hard_token_cap=5,
    )
    state = RowSamplingState(thinking_budget=thinking_budget_state_from_params(params))

    reasoning = select_token(np.array([4.0, 5.0, 3.0], dtype=np.float32), params, state)
    state.observe(2)
    answer = select_token(np.array([4.0, 5.0, 3.0], dtype=np.float32), params, state)

    assert reasoning.token_id == 0
    assert reasoning.active_processors == ("thinking_budget",)
    assert reasoning.fast_path_blockers == ("thinking_budget",)
    assert state.thinking_budget is not None
    assert state.thinking_budget.phase == "answer"
    assert answer.token_id == 1


def test_post_thinking_forced_tokens_queue_after_close_sequence() -> None:
    params = _params(
        thinking_close_token_ids=(2,),
        thinking_hard_token_cap=8,
        post_thinking_forced_tokens_pending=(3, 4),
        post_thinking_forced_token_reason="tool_choice_required",
    )
    state = RowSamplingState(
        thinking_budget=thinking_budget_state_from_params(params),
        post_thinking_forced_tokens_pending=params.post_thinking_forced_tokens_pending,
        post_thinking_forced_token_reason=params.post_thinking_forced_token_reason,
    )

    close = select_token(np.array([0.0, 1.0, 5.0, 0.0, 0.0], dtype=np.float32), params, state)
    first_tool = select_token(np.array([10.0, 9.0, 8.0, 0.0, 0.0], dtype=np.float32), params, state)
    second_tool = select_token(np.array([10.0, 9.0, 8.0, 0.0, 0.0], dtype=np.float32), params, state)

    assert close.token_id == 2
    assert close.forced is False
    assert first_tool.token_id == 3
    assert first_tool.forced is True
    assert first_tool.forced_reason == "tool_choice_required"
    assert first_tool.forced_tokens_remaining == 1
    assert second_tool.token_id == 4
    assert second_tool.forced is True
    assert second_tool.forced_tokens_remaining == 0
    assert state.generated_tokens == [2, 3, 4]


def test_force_sequence_completion_queues_remaining_suffix() -> None:
    params = _params(
        force_sequence_completion_token_sequences=((5, 6, 7),),
        force_sequence_completion_reason="tool_call_close_repair",
    )
    state = RowSamplingState(
        force_sequence_completion_token_sequences=params.force_sequence_completion_token_sequences,
        force_sequence_completion_reason=params.force_sequence_completion_reason,
    )

    first = select_token(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 0.0, 0.0], dtype=np.float32), params, state)
    second = select_token(np.array([10.0, 9.0, 8.0, 7.0, 6.0, 0.0, 0.0, 0.0], dtype=np.float32), params, state)
    third = select_token(np.array([10.0, 9.0, 8.0, 7.0, 6.0, 0.0, 0.0, 0.0], dtype=np.float32), params, state)

    assert first.token_id == 5
    assert first.forced is False
    assert first.active_processors == ("force_sequence_completion_token_sequences",)
    assert second.token_id == 6
    assert second.forced is True
    assert second.forced_reason == "tool_call_close_repair"
    assert second.forced_tokens_remaining == 1
    assert third.token_id == 7
    assert third.forced is True
    assert third.forced_reason == "tool_call_close_repair"
    assert third.forced_tokens_remaining == 0
    assert state.generated_tokens == [5, 6, 7]


def test_force_sequence_completion_extends_overlapping_forced_prefix_once() -> None:
    params = _params(
        forced_tokens_pending=(77, 78),
        forced_token_reason="tool_choice_required",
        force_sequence_completion_token_sequences=((77, 78, 90, 91, 92),),
        force_sequence_completion_reason="tool_call_sequence_completion",
    )
    state = RowSamplingState(
        forced_tokens_pending=params.forced_tokens_pending,
        forced_token_reason=params.forced_token_reason,
        force_sequence_completion_token_sequences=params.force_sequence_completion_token_sequences,
        force_sequence_completion_reason=params.force_sequence_completion_reason,
    )
    logits = np.zeros(100, dtype=np.float32)

    selected = [select_token(logits, params, state) for _ in range(5)]

    assert [result.token_id for result in selected] == [77, 78, 90, 91, 92]
    assert selected[0].forced_reason == "tool_choice_required"
    assert selected[0].forced_tokens_remaining == 1
    assert selected[1].forced_reason == "tool_choice_required"
    assert selected[1].forced_tokens_remaining == 3
    assert [result.forced_reason for result in selected[2:]] == [
        "tool_call_sequence_completion",
        "tool_call_sequence_completion",
        "tool_call_sequence_completion",
    ]
    assert selected[-1].forced_tokens_remaining == 0
    assert state.generated_tokens == [77, 78, 90, 91, 92]


def test_json_object_constraint_accepts_complete_root_object() -> None:
    state = JsonObjectConstraintState().observe_text('  {"path": ["README.md"], "ok": true}  ')

    assert state.started is True
    assert state.complete is True
    assert state.invalid is False
    assert state.forced_close_suffix == ""
    assert state.to_json_dict() == {
        "started": True,
        "complete": True,
        "invalid": False,
    }


def test_json_object_constraint_reports_safe_forced_close_suffix() -> None:
    state = JsonObjectConstraintState().observe_text('{"outer": [{"inner": 1')

    assert state.complete is False
    assert state.invalid is False
    assert state.forced_close_suffix == "}]}"
    assert state.needs_close is True
    assert state.to_json_dict()["expected_close_stack"] == ["}", "]", "}"]

    state.observe_text(state.forced_close_suffix)

    assert state.complete is True
    assert state.invalid is False


def test_json_object_constraint_ignores_delimiters_inside_strings() -> None:
    state = JsonObjectConstraintState().observe_text(r'{"text": "literal } and escaped \" brace {", "open": [')

    assert state.complete is False
    assert state.invalid is False
    assert state.in_string is False
    assert state.escaping is False
    assert state.forced_close_suffix == "]}"


def test_json_object_constraint_repairs_open_value_string_when_parseable() -> None:
    state = JsonObjectConstraintState().observe_text('{"path":"README')

    assert state.complete is False
    assert state.invalid is False
    assert state.in_string is True
    assert state.forced_close_suffix == '"}'
    assert state.to_json_dict()["forced_close_suffix"] == '"}'

    state.observe_text(state.forced_close_suffix)

    assert state.complete is True
    assert state.invalid is False


def test_json_object_constraint_repairs_nested_array_value_string_when_parseable() -> None:
    state = JsonObjectConstraintState().observe_text('{"paths":["README')

    assert state.complete is False
    assert state.invalid is False
    assert state.in_string is True
    assert state.forced_close_suffix == '"]}'

    state.observe_text(state.forced_close_suffix)

    assert state.complete is True
    assert state.invalid is False


@pytest.mark.parametrize("text", ['{"path"', '{"path":', '{"path": [1,'])
def test_json_object_constraint_refuses_unparseable_close_suffix(text: str) -> None:
    state = JsonObjectConstraintState().observe_text(text)

    assert state.complete is False
    assert state.invalid is False
    assert state.forced_close_suffix == ""


def test_json_object_constraint_refuses_escape_state_string_repair() -> None:
    state = JsonObjectConstraintState().observe_text('{"path":"README' + "\\")

    assert state.in_string is True
    assert state.escaping is True
    assert state.forced_close_suffix == ""


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("[]", "root_must_be_object"),
        ('{"a": [1}', "mismatched_closing_delimiter"),
        ('{"a": 1} trailing', "trailing_content"),
    ],
)
def test_json_object_constraint_reports_invalid_states(text: str, reason: str) -> None:
    state = JsonObjectConstraintState().observe_text(text)

    assert state.invalid is True
    assert state.error_reason == reason
    assert state.forced_close_suffix == ""


def test_thinking_budget_hard_close_overrides_logit_bias_and_sampling() -> None:
    state = RowSamplingState(
        seed=123,
        thinking_budget=ThinkingBudgetState(close_sequence=(3,), hard_token_cap=0),
    )

    result = select_token(
        np.array([10.0, 9.0, 8.0, 0.0], dtype=np.float32),
        _params(temperature=0.8, seed=123, logit_bias={0: 100.0}),
        state,
    )

    assert result.token_id == 3
    assert result.mode is SamplingMode.HOST_LOGITS_SAMPLE
    assert result.forced is True
    assert result.forced_reason == "thinking_hard_close"
    assert result.active_processors == ("logit_bias", "forced_tokens_pending")
    assert result.fast_path_blockers == ("temperature", "logit_bias", "forced_tokens_pending")


def test_forced_token_outside_vocab_does_not_consume_queue() -> None:
    state = RowSamplingState(forced_tokens_pending=(5,), forced_token_reason="bad")

    with pytest.raises(ValueError, match="outside vocab"):
        select_token(np.array([1.0, 2.0], dtype=np.float32), _params(), state)

    assert state.forced_tokens == (5,)
    assert state.generated_tokens == []


def test_suppress_token_ids_apply_after_bias_before_argmax() -> None:
    result = select_token(
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
        _params(temperature=0.0, logit_bias={1: 10.0}, suppress_token_ids=(1,)),
    )

    assert result.token_id == 2
    assert result.active_processors == ("logit_bias", "suppress_token_ids")
    assert result.fast_path_blockers == ("logit_bias", "suppress_token_ids")


def test_min_tokens_suppresses_eos_until_step_threshold() -> None:
    first = select_token(
        np.array([0.0, 5.0, 4.0], dtype=np.float32),
        _params(temperature=0.0, min_tokens=1, eos_token_id=1),
        RowSamplingState(step_index=0),
    )
    second = select_token(
        np.array([0.0, 5.0, 4.0], dtype=np.float32),
        _params(temperature=0.0, min_tokens=1, eos_token_id=1),
        RowSamplingState(step_index=1),
    )

    assert first.token_id == 2
    assert first.active_processors == ("min_tokens",)
    assert second.token_id == 1


def test_suppressions_validate_vocab_and_cannot_remove_every_token() -> None:
    with pytest.raises(ValueError, match="outside vocab"):
        select_token(np.array([1.0, 2.0], dtype=np.float32), _params(suppress_token_ids=(2,)))
    with pytest.raises(ValueError, match="no finite logits"):
        select_token(np.array([1.0, 2.0], dtype=np.float32), _params(suppress_token_ids=(0, 1)))
    with pytest.raises(ValueError, match="requires eos_token_id"):
        select_token(np.array([1.0, 2.0], dtype=np.float32), _params(min_tokens=1))
    with pytest.raises(ValueError, match="forced_tokens_pending"):
        select_token(np.array([1.0, 2.0], dtype=np.float32), _params(forced_tokens_pending=(-1,)))
    with pytest.raises(ValueError, match="post_thinking_forced_tokens_pending"):
        select_token(
            np.array([1.0, 2.0], dtype=np.float32),
            _params(post_thinking_forced_tokens_pending=(1,)),
        )
    with pytest.raises(ValueError, match="force_sequence_completion_token_sequences"):
        select_token(
            np.array([1.0, 2.0], dtype=np.float32),
            _params(force_sequence_completion_token_sequences=((-1, 2),)),
        )


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
