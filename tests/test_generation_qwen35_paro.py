from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hipengine.generation.qwen35_paro as qwen35
from hipengine.generation import (
    GenerationCancellationToken,
    GenerationCancelled,
    GenerationDeadlineExceeded,
    GenerationRequest,
    GenerationStreamChunk,
)
from hipengine.generation.sampling import select_token
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoAutoregressiveStepResult,
    estimate_qwen35_paro_kv_capacity,
    qwen35_paro_kv_bytes_per_token,
)


def _request(prompts=("hello",), max_tokens=1, *, ignore_eos=False, **overrides) -> GenerationRequest:
    values = {
        "prompts": tuple(prompts),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": ignore_eos,
    }
    values.update(overrides)
    return GenerationRequest(**values)


def _result(
    token_id: int,
    text: str,
    *,
    logprob: float | None = None,
    top_logprobs: tuple[tuple[int, float], ...] = (),
    forced: bool = False,
    forced_reason: str | None = None,
    forced_tokens_remaining: int = 0,
) -> Qwen35ParoAutoregressiveStepResult:
    return Qwen35ParoAutoregressiveStepResult(
        token_id=token_id,
        token_text=text,
        logit=float(token_id),
        logprob=logprob,
        top_logprobs=top_logprobs,
        forced=forced,
        forced_reason=forced_reason,
        forced_tokens_remaining=forced_tokens_remaining,
    )


def _decode_state(output):
    assert output.telemetry is not None
    return output.telemetry.to_json_dict()["decode_state"]


def test_qwen35_paro_row_sampling_state_binds_thinking_budget() -> None:
    request = _request(
        thinking_close_token_ids=(42, 43),
        thinking_hard_token_cap=1,
        thinking_soft_close_window=1,
    )

    state = qwen35._row_sampling_state(request, [1, 2], row_index=0)
    assert state.thinking_budget is not None
    assert state.thinking_budget.close_sequence == (42, 43)
    assert state.thinking_budget.hard_token_cap == 1

    state.observe(10)
    cloned = qwen35._clone_row_sampling_state(state)
    cloned.prepare_for_selection()

    assert cloned.forced_tokens == (42, 43)
    assert cloned.forced_token_reason == "thinking_hard_close"


def test_qwen35_paro_row_sampling_state_binds_request_forced_tokens() -> None:
    request = _request(
        forced_tokens_pending=(77, 78),
        forced_token_reason="tool_choice_required",
    )

    state = qwen35._row_sampling_state(request, [1, 2], row_index=0)

    assert state.forced_tokens == (77, 78)
    assert state.forced_token_reason == "tool_choice_required"


def test_qwen35_paro_row_sampling_state_queues_post_thinking_forced_tokens() -> None:
    request = _request(
        thinking_close_token_ids=(42, 43),
        thinking_hard_token_cap=8,
        post_thinking_forced_tokens_pending=(77, 78),
        post_thinking_forced_token_reason="tool_choice_required",
    )

    state = qwen35._row_sampling_state(request, [1, 2], row_index=0)
    state.observe(42)
    state.observe(43)
    state.prepare_for_selection()

    assert state.forced_tokens == (77, 78)
    assert state.forced_token_reason == "tool_choice_required"
    assert state.post_thinking_forced_tokens_pending.pending_tokens == ()


def test_qwen35_paro_row_sampling_state_repairs_tool_close_suffix() -> None:
    request = _request(
        force_sequence_completion_token_sequences=((70, 71),),
        force_sequence_completion_reason="tool_call_close_repair",
    )

    state = qwen35._row_sampling_state(request, [1, 2], row_index=0)
    state.observe(70)

    assert state.forced_tokens == (71,)
    assert state.forced_token_reason == "tool_call_close_repair"


def test_qwen35_paro_host_sampler_resolves_tokenizer_eos_for_thinking_budget(monkeypatch) -> None:
    calls = []

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: 99 if token == "<|endoftext|>" else None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            pass

        def configure_host_sampler(self, params, state):
            calls.append(("configure_host_sampler", None if params is None else params.eos_token_id))

        def prefill_native(self, token_ids, *, sample: bool = True):
            return _result(2, "C")

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    out = generator.generate(
        _request(
            max_tokens=1,
            thinking_close_token_ids=(2,),
            thinking_hard_token_cap=5,
        )
    )

    assert out == ["C"]
    assert calls == [("configure_host_sampler", 99), ("configure_host_sampler", None)]


def test_qwen35_paro_sampled_request_forced_token_overrides_logits(monkeypatch) -> None:
    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)
        vocab_size = 3

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            self.params = None
            self.state = None

        def configure_host_sampler(self, params, state):
            self.params = params
            self.state = state

        def prefill_native(self, token_ids, *, sample: bool = True):
            assert self.params is not None
            assert self.state is not None
            sample_result = select_token(
                np.array([0.0, 10.0, 1.0], dtype=np.float32),
                self.params,
                self.state,
            )
            return _result(
                sample_result.token_id,
                "C",
                forced=sample_result.forced,
                forced_reason=sample_result.forced_reason,
                forced_tokens_remaining=sample_result.forced_tokens_remaining,
            )

        def step(self, token_id: int, *, position: int, sample: bool = True):  # pragma: no cover - max_tokens=1
            raise AssertionError("forced-token fixture should finish after prefill")

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    out = generator.generate(
        _request(
            max_tokens=1,
            forced_tokens_pending=(2,),
            forced_token_reason="tool_choice_required",
        )
    )

    assert out == ["C"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict()["sampler_mode"] == "processed_argmax"
    decode_state = _decode_state(generator.last_generation_outputs[0])
    assert decode_state["active_processors"] == ["forced_tokens_pending"]
    assert decode_state["forced_token_id"] == 2
    assert decode_state["forced_token_reason"] == "tool_choice_required"
    assert decode_state["forced_tokens_remaining"] == 0


def test_qwen35_paro_kv_capacity_estimate_reports_int8_max_below_model_context() -> None:
    config = SimpleNamespace(
        layer_types=("linear_attention",) * 30 + ("full_attention",) * 10,
        num_key_value_heads=2,
        head_dim=256,
        max_position_embeddings=262144,
    )
    bytes_per_token = qwen35_paro_kv_bytes_per_token(
        config,
        storage_dtype="int8_per_token_head",
        scale_dtype="fp16",
    )
    estimate = estimate_qwen35_paro_kv_capacity(
        config,
        available_bytes=bytes_per_token * 131072 + 512 * 1024**2,
        requested_context_tokens=8192,
        storage_dtype="int8_per_token_head",
        scale_dtype="fp16",
        reserve_bytes=512 * 1024**2,
    )

    assert bytes_per_token == 10320
    assert 0 < estimate.allocatable_context_tokens < 131072
    assert estimate.requested_context_overhead_bytes > 0
    assert estimate.requested_total_bytes == estimate.requested_kv_bytes + estimate.requested_context_overhead_bytes
    assert estimate.model_max_context_tokens == 262144
    assert estimate.fits_requested is True
    assert estimate.fits_model_max is False


def test_qwen35_paro_prepare_allocates_configured_resident_session(monkeypatch) -> None:
    calls = []

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            calls.append(("init", runner, max_sequence_length, kwargs["kv_policy"].storage_dtype.value))

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    runner = object()
    generator._runner = runner

    generator.prepare(
        max_sequence_length=131072,
        sampling_params=SimpleNamespace(
            kv_storage="int8_per_token_head",
            kv_scale_dtype="fp16",
            kv_scale_granularity="per_token_head",
        ),
    )

    assert calls == [("init", runner, 131072, "int8_per_token_head")]


def test_qwen35_paro_generator_runs_multi_token_resident_decode_graph(monkeypatch) -> None:
    calls = []

    class FakeGraph:
        def __enter__(self):
            calls.append(("graph_enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("graph_close",))

        def replay(self, steps: int):
            calls.append(("graph_replay", steps))

        def read_generated_token_ids(self, count: int):
            calls.append(("graph_read", count))
            return [101, 102]

    class FakeSession:
        tokenizer = SimpleNamespace(
            token_to_id=lambda token: 999 if token == "<|endoftext|>" else None,
            decode=lambda ids: {101: "B", 102: "C"}[int(ids[0])],
        )

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            calls.append(("init", runner, max_sequence_length))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("close",))

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

        def capture_decode_graph(self, *, position, steps_per_replay, max_replay_steps, record_steps):
            calls.append(
                (
                    "capture_decode_graph",
                    position,
                    steps_per_replay,
                    max_replay_steps,
                    record_steps,
                )
            )
            return FakeGraph()

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)

    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    runner = object()
    generator._runner = runner

    out = generator.generate(_request(max_tokens=3))

    assert out == ["ABC"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "length",
        "length_limit": 3,
        "sampler_mode": "greedy_fast",
    }
    assert _decode_state(generator.last_generation_outputs[0]) == {
        "row_index": 0,
        "step_index": 3,
        "prompt_tokens": 2,
        "generated_tokens": 3,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_mode": "greedy_fast",
    }
    assert calls == [
        ("init", runner, 4096),
        ("prefill_native", (10, 11), True),
        ("capture_decode_graph", 2, 1, 2, 2),
        ("graph_enter",),
        ("graph_replay", 2),
        ("graph_read", 2),
        ("graph_close",),
    ]


def test_qwen35_paro_generator_allows_inert_greedy_filters(monkeypatch) -> None:
    calls = []

    class FakeGraph:
        def __enter__(self):
            calls.append(("graph_enter",))
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("graph_close",))

        def replay(self, steps: int):
            calls.append(("graph_replay", steps))

        def read_generated_token_ids(self, count: int):
            return [101]

    class FakeSession:
        tokenizer = SimpleNamespace(
            token_to_id=lambda token: None,
            decode=lambda ids: {101: "B"}[int(ids[0])],
        )

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            pass

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

        def capture_decode_graph(self, **kwargs):
            calls.append(("capture_decode_graph", kwargs["position"]))
            return FakeGraph()

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    out = generator.generate(_request(max_tokens=2, top_p=0.5, top_k=4, min_p=0.5))

    assert out == ["AB"]
    assert ("graph_replay", 1) in calls



def test_qwen35_paro_generator_uses_host_sampler_for_non_greedy(monkeypatch) -> None:
    calls = []

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)
        vocab_size = 128

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            pass

        def configure_host_sampler(self, params, state):
            calls.append(
                (
                    "configure_host_sampler",
                    None if params is None else params.temperature,
                    None if state is None else state.seed,
                    None if state is None else state.prompt_tokens,
                )
            )

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

        def step(self, token_id: int, *, position: int, sample: bool = True):
            calls.append(("step", token_id, position, sample))
            return _result(101, "B") if sample else None

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    out = generator.generate(_request(max_tokens=2, temperature=0.7, seed=5))

    assert out == ["AB"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "length",
        "length_limit": 2,
        "sampler_mode": "host_logits_sample",
    }
    assert _decode_state(generator.last_generation_outputs[0]) == {
        "row_index": 0,
        "step_index": 2,
        "prompt_tokens": 2,
        "generated_tokens": 2,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_fast_path_blockers": ["temperature"],
        "sampler_fallback_reason": "host_sampling_required",
        "sampler_mode": "host_logits_sample",
        "full_vocab_logits_d2h": True,
        "logits_d2h_bytes": 512,
    }
    assert calls[0][0] == "configure_host_sampler"
    assert calls[0][1] == 0.7
    assert calls[0][3] == (10, 11)
    assert ("step", 100, 2, True) in calls
    assert calls[-1] == ("configure_host_sampler", None, None, None)


def test_qwen35_paro_stream_detailed_emits_live_greedy_telemetry(monkeypatch) -> None:
    calls = []

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            calls.append(("init", runner, max_sequence_length))

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

        def step(self, token_id: int, *, position: int, sample: bool = True):
            calls.append(("step", token_id, position, sample))
            return _result(101, "B") if sample else None

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    runner = object()
    generator._runner = runner

    chunks = list(generator.stream_detailed(_request(max_tokens=2)))

    assert [chunk.text for chunk in chunks] == ["A", "B"]
    assert all(isinstance(chunk, GenerationStreamChunk) for chunk in chunks)
    assert [_decode_state(chunk) for chunk in chunks] == [
        {
            "row_index": 0,
            "step_index": 1,
            "prompt_tokens": 2,
            "generated_tokens": 1,
            "phase": "answer",
            "continuation_eligible": False,
            "sampler_mode": "greedy_fast",
        },
        {
            "row_index": 0,
            "step_index": 2,
            "prompt_tokens": 2,
            "generated_tokens": 2,
            "phase": "answer",
            "continuation_eligible": False,
            "sampler_mode": "greedy_fast",
        },
    ]
    assert calls == [
        ("init", runner, 4096),
        ("prefill_native", (10, 11), True),
        ("step", 100, 2, True),
    ]


def test_qwen35_paro_stream_text_wrapper_preserves_plain_chunks(monkeypatch) -> None:
    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            pass

        def prefill_native(self, token_ids, *, sample: bool = True):
            return _result(100, "A") if sample else None

        def step(self, token_id: int, *, position: int, sample: bool = True):
            return _result(101, "B") if sample else None

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    assert list(generator.stream(_request(max_tokens=2))) == ["A", "B"]


def test_qwen35_paro_stream_detailed_emits_live_sampled_telemetry(monkeypatch) -> None:
    calls = []

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            pass

        def configure_host_sampler(self, params, state):
            calls.append(
                (
                    "configure_host_sampler",
                    None if params is None else params.temperature,
                    None if state is None else state.seed,
                    None if state is None else state.prompt_tokens,
                )
            )

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

        def step(self, token_id: int, *, position: int, sample: bool = True):
            calls.append(("step", token_id, position, sample))
            return _result(101, "B") if sample else None

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    chunks = list(generator.stream_detailed(_request(max_tokens=2, temperature=0.7, seed=5)))

    assert [chunk.text for chunk in chunks] == ["A", "B"]
    assert [_decode_state(chunk) for chunk in chunks] == [
        {
            "row_index": 0,
            "step_index": 1,
            "prompt_tokens": 2,
            "generated_tokens": 1,
            "phase": "answer",
            "continuation_eligible": False,
            "sampler_fast_path_blockers": ["temperature"],
            "sampler_fallback_reason": "host_sampling_required",
            "sampler_mode": "host_logits_sample",
        },
        {
            "row_index": 0,
            "step_index": 2,
            "prompt_tokens": 2,
            "generated_tokens": 2,
            "phase": "answer",
            "continuation_eligible": False,
            "sampler_fast_path_blockers": ["temperature"],
            "sampler_fallback_reason": "host_sampling_required",
            "sampler_mode": "host_logits_sample",
        },
    ]
    assert calls[0][0] == "configure_host_sampler"
    assert calls[0][1] == 0.7
    assert isinstance(calls[0][2], int)
    assert calls[0][3] == (10, 11)
    assert calls[1:] == [
        ("prefill_native", (10, 11), True),
        ("step", 100, 2, True),
        ("configure_host_sampler", None, None, None),
    ]


def test_qwen35_paro_stream_detailed_reports_thinking_budget_pressure(monkeypatch) -> None:
    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)
        vocab_size = 3

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            self.params = None
            self.state = None

        def configure_host_sampler(self, params, state):
            self.params = params
            self.state = state

        def prefill_native(self, token_ids, *, sample: bool = True):
            assert self.params is not None
            assert self.state is not None
            sample_result = select_token(
                np.array([0.0, 5.0, 1.0], dtype=np.float32),
                self.params,
                self.state,
            )
            return _result(
                sample_result.token_id,
                "C",
                forced=sample_result.forced,
                forced_reason=sample_result.forced_reason,
                forced_tokens_remaining=sample_result.forced_tokens_remaining,
            )

        def step(self, token_id: int, *, position: int, sample: bool = True):  # pragma: no cover - max_tokens=1
            raise AssertionError("hard-close stream fixture should finish after prefill")

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    chunks = list(
        generator.stream_detailed(
            _request(
                max_tokens=1,
                thinking_close_token_ids=(2,),
                thinking_hard_token_cap=0,
            )
        )
    )

    assert [chunk.text for chunk in chunks] == ["C"]
    assert _decode_state(chunks[0]) == {
        "row_index": 0,
        "step_index": 1,
        "prompt_tokens": 2,
        "generated_tokens": 1,
        "phase": "answer",
        "continuation_eligible": False,
        "reasoning_tokens": 1,
        "active_processors": ["thinking_budget"],
        "sampler_fast_path_blockers": ["thinking_budget"],
        "sampler_fallback_reason": "processed_logits_required",
        "forced_token_id": 2,
        "forced_token_reason": "thinking_hard_close",
        "forced_tokens_remaining": 0,
        "budget_pressure": "hard_close",
        "sampler_mode": "processed_argmax",
        "full_vocab_logits_d2h": True,
        "logits_d2h_bytes": 12,
    }


def test_qwen35_paro_generator_checks_deadline_after_prefill(monkeypatch) -> None:
    calls = []

    def check_deadline(value, **kwargs) -> None:
        calls.append(("deadline", None if value is None else getattr(value, "deadline_at", value)))
        if ("prefill_native", (10, 11), True) in calls:
            raise GenerationDeadlineExceeded(deadline_at=getattr(value, "deadline_at", value))

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            calls.append(("init", runner, max_sequence_length))

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

        def capture_decode_graph(self, **kwargs):  # pragma: no cover - deadline should stop first
            calls.append(("capture_decode_graph", kwargs))
            raise AssertionError("deadline should stop before graph replay")

    monkeypatch.setattr(qwen35, "raise_if_generation_deadline_expired", check_deadline)
    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    with pytest.raises(GenerationDeadlineExceeded):
        generator.generate(_request(max_tokens=2, deadline_at=123.0))

    assert ("prefill_native", (10, 11), True) in calls
    assert not any(call[0] == "capture_decode_graph" for call in calls)


def test_qwen35_paro_generator_checks_cancellation_after_prefill(monkeypatch) -> None:
    calls = []
    token = GenerationCancellationToken()

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            calls.append(("init", runner, max_sequence_length))

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            token.cancel()
            return _result(100, "A") if sample else None

        def capture_decode_graph(self, **kwargs):  # pragma: no cover - cancellation should stop first
            calls.append(("capture_decode_graph", kwargs))
            raise AssertionError("cancellation should stop before graph replay")

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    with pytest.raises(GenerationCancelled) as raised:
        generator.generate(_request(max_tokens=2, cancellation_token=token))

    assert raised.value.finish_details.to_json_dict() == {"reason": "cancelled", "cancelled": True}
    assert ("prefill_native", (10, 11), True) in calls
    assert not any(call[0] == "capture_decode_graph" for call in calls)


def test_qwen35_paro_finish_details_report_forced_thinking_close(monkeypatch) -> None:
    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            self.params = None
            self.state = None

        def configure_host_sampler(self, params, state):
            self.params = params
            self.state = state

        def prefill_native(self, token_ids, *, sample: bool = True):
            assert self.params is not None
            assert self.state is not None
            sample_result = select_token(
                np.array([0.0, 5.0, 1.0], dtype=np.float32),
                self.params,
                self.state,
            )
            return _result(
                sample_result.token_id,
                "C",
                forced=sample_result.forced,
                forced_reason=sample_result.forced_reason,
                forced_tokens_remaining=sample_result.forced_tokens_remaining,
            )

        def step(self, token_id: int, *, position: int, sample: bool = True):  # pragma: no cover - max_tokens=1
            raise AssertionError("hard-close fixture should finish after prefill")

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    out = generator.generate(
        _request(
            max_tokens=1,
            thinking_close_token_ids=(2,),
            thinking_hard_token_cap=0,
        )
    )

    assert out == ["C"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "thinking_budget_exhausted",
        "length_limit": 1,
        "forced_close": True,
        "reasoning_tokens": 1,
        "budget_pressure": "hard_close",
        "sampler_mode": "processed_argmax",
        "phase": "answer",
    }
    decode_state = _decode_state(generator.last_generation_outputs[0])
    assert decode_state["phase"] == "answer"
    assert decode_state["reasoning_tokens"] == 1
    assert decode_state["forced_token_id"] == 2
    assert decode_state["forced_token_reason"] == "thinking_hard_close"
    assert decode_state["forced_tokens_remaining"] == 0
    assert decode_state["budget_pressure"] == "hard_close"


def test_qwen35_paro_generator_env_routes_supported_c1_request_to_native_sampler(monkeypatch) -> None:
    calls = []

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            pass

        def configure_native_sampler(self, params, state):
            calls.append(
                (
                    "configure_native_sampler",
                    None if params is None else params.temperature,
                    None if state is None else state.seed,
                    None if state is None else state.prompt_tokens,
                )
            )

        def configure_host_sampler(self, params, state):  # pragma: no cover - this path must not be used
            calls.append(("configure_host_sampler", params is None))

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

        def step(self, token_id: int, *, position: int, sample: bool = True):
            calls.append(("step", token_id, position, sample))
            return _result(101, "B") if sample else None

    monkeypatch.setenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", "1")
    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    out = generator.generate(_request(max_tokens=2, temperature=0.7, top_k=4, seed=5))

    assert out == ["AB"]
    assert calls[0][0] == "configure_native_sampler"
    assert calls[0][1] == 0.7
    assert calls[0][3] == (10, 11)
    assert not any(call[0] == "configure_host_sampler" for call in calls)
    assert ("step", 100, 2, True) in calls
    assert calls[-1] == ("configure_native_sampler", None, None, None)
    assert _decode_state(generator.last_generation_outputs[0]) == {
        "row_index": 0,
        "step_index": 2,
        "prompt_tokens": 2,
        "generated_tokens": 2,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_fast_path_blockers": ["temperature"],
        "sampler_mode": "gpu_sample",
        "full_vocab_logits_d2h": False,
        "logits_d2h_bytes": 0,
    }


def test_qwen35_paro_native_opt_in_reports_unsupported_top_logprobs_fallback(monkeypatch) -> None:
    calls = []

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)
        vocab_size = 128

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            pass

        def configure_host_sampler(self, params, state):
            calls.append(
                (
                    "configure_host_sampler",
                    None if params is None else params.temperature,
                    None if params is None else params.top_logprobs,
                    None if state is None else state.prompt_tokens,
                )
            )

        def configure_native_sampler(self, params, state):  # pragma: no cover - unsupported request must not use native
            calls.append(("configure_native_sampler", params is None))

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

        def step(self, token_id: int, *, position: int, sample: bool = True):
            calls.append(("step", token_id, position, sample))
            return _result(101, "B") if sample else None

    monkeypatch.setenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", "1")
    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    out = generator.generate(_request(max_tokens=2, temperature=0.7, top_logprobs=1, seed=5))

    assert out == ["AB"]
    assert calls[0] == ("configure_host_sampler", 0.7, 1, (10, 11))
    assert not any(call[0] == "configure_native_sampler" for call in calls)
    assert _decode_state(generator.last_generation_outputs[0]) == {
        "row_index": 0,
        "step_index": 2,
        "prompt_tokens": 2,
        "generated_tokens": 2,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_fast_path_blockers": ["temperature", "top_logprobs"],
        "sampler_fallback_reason": "native_gpu_unsupported_request",
        "sampler_mode": "host_logits_sample",
        "full_vocab_logits_d2h": True,
        "logits_d2h_bytes": 512,
    }


def test_qwen35_paro_native_opt_in_stream_reports_unsupported_top_k_fallback(monkeypatch) -> None:
    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)
        vocab_size = 128

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            pass

        def configure_host_sampler(self, params, state):
            pass

        def configure_native_sampler(self, params, state):  # pragma: no cover - unsupported request must not use native
            raise AssertionError("unsupported native stream request should use host sampler")

        def prefill_native(self, token_ids, *, sample: bool = True):
            return _result(100, "A") if sample else None

        def step(self, token_id: int, *, position: int, sample: bool = True):
            return _result(101, "B") if sample else None

    monkeypatch.setenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", "1")
    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    chunks = list(generator.stream_detailed(_request(max_tokens=2, temperature=0.7, top_k=65, seed=5)))

    assert [chunk.text for chunk in chunks] == ["A", "B"]
    assert _decode_state(chunks[-1]) == {
        "row_index": 0,
        "step_index": 2,
        "prompt_tokens": 2,
        "generated_tokens": 2,
        "phase": "answer",
        "continuation_eligible": False,
        "sampler_fast_path_blockers": ["temperature"],
        "sampler_fallback_reason": "native_gpu_unsupported_request",
        "sampler_mode": "host_logits_sample",
        "full_vocab_logits_d2h": True,
        "logits_d2h_bytes": 512,
    }



def test_qwen35_paro_host_sampler_stops_on_stop_token_id(monkeypatch) -> None:
    calls = []

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            pass

        def configure_host_sampler(self, params, state):
            calls.append(("configure_host_sampler", params is None))

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

        def step(self, token_id: int, *, position: int, sample: bool = True):
            calls.append(("step", token_id, position, sample))
            return _result(101, "B") if sample else None

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    out = generator.generate(_request(max_tokens=2, stop_token_ids=(100,)))

    assert out == ["A"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "stop",
        "stop_sequence": [100],
        "sampler_mode": "processed_argmax",
    }
    assert not any(call[0] == "step" for call in calls)



def test_qwen35_paro_host_sampler_stops_on_multi_token_stop_sequence(monkeypatch) -> None:
    calls = []

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            pass

        def configure_host_sampler(self, params, state):
            calls.append(("configure_host_sampler", params is None))

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

        def step(self, token_id: int, *, position: int, sample: bool = True):
            calls.append(("step", token_id, position, sample))
            return _result(101, "B") if sample else None

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    out = generator.generate(_request(max_tokens=3, stop_token_sequences=((100, 101),)))

    assert out == ["AB"]
    assert generator.last_generation_outputs[0].finish_details is not None
    assert generator.last_generation_outputs[0].finish_details.to_json_dict() == {
        "reason": "stop",
        "stop_sequence": [100, 101],
        "sampler_mode": "processed_argmax",
    }
    assert _decode_state(generator.last_generation_outputs[0]) == {
        "row_index": 0,
        "step_index": 2,
        "prompt_tokens": 2,
        "generated_tokens": 2,
        "phase": "done",
        "continuation_eligible": False,
        "stop_suffix_state": {"matched_sequence": [100, 101]},
        "active_processors": ["stop_token_sequences"],
        "sampler_fast_path_blockers": ["stop_token_sequences"],
        "sampler_fallback_reason": "processed_logits_required",
        "sampler_mode": "processed_argmax",
    }
    assert len([call for call in calls if call[0] == "step"]) == 1



def test_qwen35_paro_generator_uses_scheduler_packed_prefill_for_prompt_batch(monkeypatch) -> None:
    calls = []
    token_rows = {"alpha": [10, 11], "beta": [20]}

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)
        block_size = 256

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            calls.append(
                (
                    "init",
                    runner,
                    max_sequence_length,
                    kwargs.get("max_batch_size"),
                    kwargs["kv_policy"].storage_dtype.value,
                )
            )
            self.max_sequence_length = max_sequence_length
            self.max_batch_size = kwargs.get("max_batch_size", 1)

        def prefill_native_packed(self, slab, *, sample: bool = True):
            calls.append(
                (
                    "prefill_native_packed",
                    slab.request_ids,
                    slab.token_rows,
                    slab.physical_slot_ids,
                    sample,
                )
            )
            return tuple(
                _result(100 + request_id, {0: "A", 1: "B"}[request_id])
                for request_id in slab.request_ids
            )

        def step_batch_serial(self, token_ids, *, positions, slots, sample: bool = True):
            calls.append(
                ("step_batch_serial", tuple(token_ids), tuple(positions), tuple(slots), sample)
            )
            return (_result(200, "C"), _result(201, "D"))

        def batch_execution_metadata(
            self, *, scheduler_owned: bool = False, native_decode: bool = False
        ):
            calls.append(("batch_execution_metadata", scheduler_owned, native_decode))
            return SimpleNamespace(
                native_compact_prefill=True,
                native_caware_decode=False,
                throughput_claim_eligible=False,
            )

    monkeypatch.setattr(
        qwen35,
        "_select_token",
        lambda model, prompt, token_id: (token_rows[prompt][-1], token_rows[prompt]),
    )
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    runner = object()
    generator._runner = runner

    out = generator.generate(_request(prompts=("alpha", "beta"), max_tokens=2))

    assert out == ["AC", "BD"]
    assert calls == [
        ("init", runner, 4096, 2, "bf16"),
        ("prefill_native_packed", (0, 1), ((10, 11), (20,)), (0, 1), True),
        ("step_batch_serial", (100, 101), (2, 1), (0, 1), True),
        ("batch_execution_metadata", True, False),
    ]
    assert generator.last_batch_generation == {
        "path": "scheduler_native_packed_prefill_serial_decode",
        "batch_size": 2,
        "request_ids": [0, 1],
        "prompt_lengths": [2, 1],
        "packed_prefill_slabs": [
            {
                "request_ids": [0, 1],
                "slot_ids": [0, 1],
                "rows": 3,
                "request_count": 2,
                "block_count": 1,
            }
        ],
        "decode_steps": 1,
        "native_decode_steps": 0,
        "serial_decode_fallback": True,
        "native_compact_prefill": True,
        "native_caware_decode": False,
        "throughput_claim_eligible": False,
    }
    assert [_decode_state(output)["row_index"] for output in generator.last_generation_outputs] == [0, 1]
    assert [_decode_state(output)["request_id"] for output in generator.last_generation_outputs] == ["0", "1"]
    assert [_decode_state(output)["prompt_tokens"] for output in generator.last_generation_outputs] == [2, 1]
    assert [_decode_state(output)["sampler_mode"] for output in generator.last_generation_outputs] == [
        "greedy_fast",
        "greedy_fast",
    ]


@pytest.mark.parametrize(
    ("native_requested", "expected_fallback"),
    [
        (False, "host_sampling_required"),
        (True, "native_gpu_unsupported_request"),
    ],
)
def test_qwen35_paro_sampled_batch_uses_scheduler_packed_prefill(
    monkeypatch,
    native_requested: bool,
    expected_fallback: str,
) -> None:
    calls = []
    token_rows = {"alpha": [10, 11], "beta": [20]}

    class FakeSession:
        tokenizer = SimpleNamespace(
            token_to_id=lambda token: None,
            decode=lambda ids: {100: "A", 101: "B", 200: "C", 201: "D"}.get(int(ids[0]), "?"),
        )
        block_size = 256
        vocab_size = 512

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            calls.append(("init", runner, max_sequence_length, kwargs.get("max_batch_size")))
            self.max_sequence_length = max_sequence_length
            self.max_batch_size = kwargs.get("max_batch_size", 1)

        def configure_host_sampler_rows(self, params, states_by_slot):
            calls.append(
                (
                    "configure_host_sampler_rows",
                    None if params is None else params.temperature,
                    None if params is None else params.logit_bias,
                    None
                    if states_by_slot is None
                    else {slot: tuple(state.generated_tokens) for slot, state in states_by_slot.items()},
                )
            )

        def prefill_native_packed(self, slab, *, sample: bool = True):
            calls.append(("prefill_native_packed", slab.request_ids, slab.physical_slot_ids, sample))
            return tuple(
                _result(
                    100 + request_id,
                    {0: "A", 1: "B"}[request_id],
                    logprob={0: -0.1, 1: -0.2}[request_id],
                    top_logprobs=(
                        (100 + request_id, {0: -0.1, 1: -0.2}[request_id]),
                        (300 + request_id, -1.5),
                    ),
                )
                for request_id in slab.request_ids
            )

        def step_batch_serial(self, token_ids, *, positions, slots, sample: bool = True):
            calls.append(("step_batch_serial", tuple(token_ids), tuple(positions), tuple(slots), sample))
            return (
                _result(200, "C", logprob=-0.3, top_logprobs=((200, -0.3), (400, -1.7))),
                _result(201, "D", logprob=-0.4, top_logprobs=((201, -0.4), (401, -1.8))),
            )

        def batch_execution_metadata(self, *, scheduler_owned: bool = False, native_decode: bool = False):
            calls.append(("batch_execution_metadata", scheduler_owned, native_decode))
            return SimpleNamespace(native_compact_prefill=True)

    monkeypatch.setattr(
        qwen35,
        "_select_token",
        lambda model, prompt, token_id: (token_rows[prompt][-1], token_rows[prompt]),
    )
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    runner = object()
    generator._runner = runner
    if native_requested:
        monkeypatch.setenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", "1")
    else:
        monkeypatch.delenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", raising=False)

    out = generator.generate(
        _request(
            prompts=("alpha", "beta"),
            max_tokens=2,
            temperature=0.7,
            seed=5,
            logit_bias={42: 1.5},
        )
    )

    assert out == ["AC", "BD"]
    assert calls == [
        ("init", runner, 4096, 2),
        ("configure_host_sampler_rows", 0.7, ((42, 1.5),), {0: (), 1: ()}),
        ("prefill_native_packed", (0, 1), (0, 1), True),
        ("configure_host_sampler_rows", 0.7, ((42, 1.5),), {0: (100,), 1: (101,)}),
        ("step_batch_serial", (100, 101), (2, 1), (0, 1), True),
        ("configure_host_sampler_rows", None, None, None),
        ("batch_execution_metadata", True, False),
    ]
    assert generator.last_batch_generation["path"] == "scheduler_native_packed_prefill_serial_host_sampler_decode"
    assert generator.last_batch_generation["native_compact_prefill"] is True
    assert [_decode_state(output)["row_index"] for output in generator.last_generation_outputs] == [0, 1]
    assert [_decode_state(output)["sampler_mode"] for output in generator.last_generation_outputs] == [
        "host_logits_sample",
        "host_logits_sample",
    ]
    assert [_decode_state(output)["active_processors"] for output in generator.last_generation_outputs] == [
        ["logit_bias"],
        ["logit_bias"],
    ]
    assert [_decode_state(output)["sampler_fast_path_blockers"] for output in generator.last_generation_outputs] == [
        ["temperature", "logit_bias"],
        ["temperature", "logit_bias"],
    ]
    assert [_decode_state(output)["sampler_fallback_reason"] for output in generator.last_generation_outputs] == [
        expected_fallback,
        expected_fallback,
    ]
    assert [_decode_state(output)["full_vocab_logits_d2h"] for output in generator.last_generation_outputs] == [
        True,
        True,
    ]
    assert [_decode_state(output)["logits_d2h_bytes"] for output in generator.last_generation_outputs] == [
        2048,
        2048,
    ]
    assert [
        [(token.token_id, token.token_text, token.logprob) for token in output.token_logprobs]
        for output in generator.last_generation_outputs
    ] == [
        [(100, "A", -0.1), (200, "C", -0.3)],
        [(101, "B", -0.2), (201, "D", -0.4)],
    ]
    assert [
        [token.top_logprobs for token in output.token_logprobs]
        for output in generator.last_generation_outputs
    ] == [
        [
            ((100, "A", -0.1), (300, "?", -1.5)),
            ((200, "C", -0.3), (400, "?", -1.7)),
        ],
        [
            ((101, "B", -0.2), (301, "?", -1.5)),
            ((201, "D", -0.4), (401, "?", -1.8)),
        ],
    ]


def test_qwen35_paro_generator_reuses_resident_session(monkeypatch) -> None:
    calls = []

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            calls.append(("init", runner, max_sequence_length))

        def reset(self):
            calls.append(("reset",))

        def close(self):
            calls.append(("close",))

        def prefill_native(self, token_ids, *, sample: bool = True):
            calls.append(("prefill_native", tuple(token_ids), sample))
            return _result(100, "A") if sample else None

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    runner = object()
    generator._runner = runner

    assert generator.generate(_request(max_tokens=1)) == ["A"]
    assert generator.generate(_request(max_tokens=1)) == ["A"]
    assert calls == [
        ("init", runner, 4096),
        ("prefill_native", (10, 11), True),
        ("reset",),
        ("prefill_native", (10, 11), True),
    ]


def test_qwen35_paro_generator_passes_int8_kv_policy_to_session(monkeypatch) -> None:
    captured = {}

    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: None)

        def __init__(self, runner, *, max_sequence_length, kv_policy, kv_scale_dtype, kv_scale_granularity):
            captured["storage_dtype"] = kv_policy.storage_dtype.value
            captured["scale_dtype"] = kv_scale_dtype.value
            captured["scale_granularity"] = kv_scale_granularity

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill_native(self, token_ids, *, sample: bool = True):
            return _result(100, "A") if sample else None

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (11, [10, 11]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    out = generator.generate(_request(max_tokens=1).__class__(
        prompts=("hello",),
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
        kv_storage="int8_per_token_head",
        kv_scale_dtype="fp32",
        kv_scale_granularity="per_token_head",
    ))

    assert out == ["A"]
    assert captured == {
        "storage_dtype": "int8_per_token_head",
        "scale_dtype": "fp32",
        "scale_granularity": "per_token_head",
    }



def test_qwen35_paro_generator_handles_zero_tokens_without_loading(monkeypatch) -> None:
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not load")))
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )

    assert generator.generate(_request(prompts=("a", "b"), max_tokens=0)) == ["", ""]


def test_qwen35_paro_generator_stops_on_eos(monkeypatch) -> None:
    class FakeSession:
        tokenizer = SimpleNamespace(token_to_id=lambda token: 100 if token == "<|endoftext|>" else None)

        def __init__(self, runner, *, max_sequence_length, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def prefill_native(self, token_ids, *, sample: bool = True):
            return _result(100, "<eos>") if sample else None

        def step(self, token_id: int, *, position: int, sample: bool = True):
            return _result(100, "<eos>") if sample else None

    monkeypatch.setattr(qwen35, "_select_token", lambda model, prompt, token_id: (1, [1]))
    monkeypatch.setattr(qwen35, "Qwen35ParoResidentSession", FakeSession)
    generator = qwen35.Qwen35ParoOneTokenGenerator(
        model_path="/tmp/model",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
    )
    generator._runner = object()

    assert generator.generate(_request(max_tokens=4, ignore_eos=False)) == ["<eos>"]
