from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hipengine import SamplingParams
from hipengine.generation import (
    FinishDetails,
    GenerationCancelled,
    GenerationDeadlineExceeded,
    GenerationOutput,
    TokenLogprob,
)
from hipengine.server import ServerConfig, create_app
from hipengine.server.api import OpenAIHTTPError, _RequestControl, _await_with_request_control


TRACE_PATH = Path(__file__).resolve().parent / "fixtures" / "agentic_traces" / "golden_traces.json"

_REQUIRED_AGENTIC_TRACE_COVERAGE: dict[str, frozenset[str]] = {
    "tool_loop": frozenset(
        {
            "tool_loop_turn_1_non_streaming",
            "tool_loop_turn_2_non_streaming_final_answer",
            "tool_call_streaming",
        }
    ),
    "reasoning_controls": frozenset(
        {
            "reasoning_tool_call_non_streaming",
            "reasoning_tool_call_streaming",
            "reasoning_split_non_streaming",
            "no_think_prompt_rendering",
            "reasoning_effort_low_prompt_budget",
        }
    ),
    "tool_validation": frozenset(
        {
            "strict_malformed_tool_call_non_streaming",
            "duplicated_tool_call_start_compat_non_streaming",
            "auto_unknown_tool_name_non_streaming",
            "required_tool_missing_non_streaming",
            "specific_tool_wrong_function_non_streaming",
            "tool_choice_none_rejects_tool_call_non_streaming",
            "parallel_tool_calls_without_opt_in_non_streaming",
            "parallel_tool_calls_streaming",
        }
    ),
    "structured_agent_outputs": frozenset(
        {
            "json_schema_violation_chat",
            "guided_json_schema_chat_reasoning_success",
            "guided_json_schema_local_ref_chat_reasoning_success",
            "guided_choice_chat_reasoning_success",
            "guided_regex_completion_rejects_mismatch",
            "guided_patch_chat_reasoning_fenced_diff",
            "guided_diff_completion_rejects_prefaced_patch",
        }
    ),
    "sessions_and_continuations": frozenset(
        {
            "stateless_session_append_none_completion",
            "session_reasoning_tool_loop_sequence",
            "session_snapshot_restore_reasoning_tool_loop_sequence",
            "session_rollback_visible_transcript_sequence",
            "length_finish_chat_answer_continuation",
            "length_finish_chat_structured_continuation",
            "continuation_resume_chat_answer_sequence",
            "session_continuation_resume_chat_sequence",
            "invalid_continuation_id_chat",
            "continuation_expired_completion_sequence",
        }
    ),
    "finish_phase_and_sampling": frozenset(
        {
            "length_finish_completion",
            "length_finish_chat_reasoning_phase",
            "length_finish_chat_closing_think_phase",
            "length_finish_chat_partial_tool_call",
            "completion_logprobs_success",
            "chat_logprobs_success",
            "completion_logprobs_omitted_reason",
            "completion_echo_logprobs_prompt_omission_reason",
            "chat_logprobs_omitted_reason",
            "completion_logprobs_missing_backend_metadata_error",
        }
    ),
    "server_error_paths": frozenset(
        {
            "context_overflow_completion_error",
            "context_overflow_completion_stream_error",
            "model_unavailable_completion_error",
            "unsupported_parameter_completion_error",
            "schema_violation_missing_prompt_error",
            "unsupported_feature_tokenize_error",
            "engine_busy_chat_session_cap_sequence",
            "deadline_error_completion",
            "backend_cancelled_completion_error",
            "backend_cancelled_completion_stream_error",
            "backend_deadline_chat_stream_error",
            "backend_cancelled_chat_stream_error",
            "request_control_cancelled",
        }
    ),
}


class TraceLLM:
    def __init__(self, trace: dict[str, Any]) -> None:
        self.outputs = list(trace.get("fake_outputs", ()))
        self.stream_chunks = list(trace.get("fake_stream_chunks", ()))
        self.detailed_outputs = [_trace_generation_output(item) for item in trace.get("fake_detailed_outputs", ())]
        self.fake_exception = str(trace.get("fake_exception") or "")
        self.generate_delay_s = float(trace.get("generate_delay_s", 0.0))
        self.calls: list[tuple[tuple[str, ...], SamplingParams]] = []
        self.stream_calls: list[tuple[str, SamplingParams]] = []

    def generate(self, prompts, sampling_params: SamplingParams) -> list[Any]:
        prompts = tuple(str(prompt) for prompt in prompts)
        self.calls.append((prompts, sampling_params))
        if self.generate_delay_s > 0.0:
            time.sleep(self.generate_delay_s)
        if self.fake_exception == "cancelled":
            if sampling_params.cancellation_token is None:
                raise AssertionError("cancelled trace expected a cancellation token")
            sampling_params.cancellation_token.cancel()
            raise GenerationCancelled(sampling_params.cancellation_token.finish_details)
        if self.fake_exception == "deadline":
            if sampling_params.deadline_at is None:
                raise AssertionError("deadline trace expected a deadline")
            raise GenerationDeadlineExceeded(deadline_at=sampling_params.deadline_at)
        if self.detailed_outputs:
            if len(self.detailed_outputs) < len(prompts):
                raise AssertionError("not enough fake detailed outputs for trace request")
            outputs = self.detailed_outputs[: len(prompts)]
            del self.detailed_outputs[: len(prompts)]
            return outputs
        if self.outputs:
            if len(self.outputs) < len(prompts):
                raise AssertionError("not enough fake outputs for trace request")
            outputs = self.outputs[: len(prompts)]
            del self.outputs[: len(prompts)]
            return outputs
        return [f"generated:{prompt}" for prompt in prompts]

    def stream(self, prompt: str, sampling_params: SamplingParams):
        self.stream_calls.append((str(prompt), sampling_params))
        self.calls.append(((str(prompt),), sampling_params))
        if self.fake_exception == "cancelled":
            if sampling_params.cancellation_token is None:
                raise AssertionError("cancelled trace expected a cancellation token")
            sampling_params.cancellation_token.cancel()
            raise GenerationCancelled(sampling_params.cancellation_token.finish_details)
        if self.fake_exception == "deadline":
            if sampling_params.deadline_at is None:
                raise AssertionError("deadline trace expected a deadline")
            raise GenerationDeadlineExceeded(deadline_at=sampling_params.deadline_at)
        yield from self.stream_chunks or self.outputs or [f"generated:{prompt}"]

    def count_tokens(self, text: str) -> int:
        return len(str(text).split())


def _trace_generation_output(item: dict[str, Any]) -> GenerationOutput:
    finish_details = item.get("finish_details")
    return GenerationOutput(
        text=str(item["text"]),
        token_logprobs=tuple(
            TokenLogprob(
                token_id=token["token_id"],
                token_text=token["token_text"],
                logprob=token.get("logprob"),
                top_logprobs=tuple(tuple(top) for top in token.get("top_logprobs", ())),
            )
            for token in item.get("token_logprobs", ())
        ),
        finish_details=None if finish_details is None else FinishDetails(**dict(finish_details)),
    )


def _load_traces() -> list[dict[str, Any]]:
    with TRACE_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["schema"] == "hipengine.agentic_traces.v1"
    return list(payload["traces"])


def test_agentic_golden_traces_cover_required_server_patterns() -> None:
    trace_names = {trace["name"] for trace in _load_traces()}
    missing = {
        category: sorted(required - trace_names)
        for category, required in _REQUIRED_AGENTIC_TRACE_COVERAGE.items()
        if not required.issubset(trace_names)
    }
    assert not missing


@pytest.mark.parametrize("trace", _load_traces(), ids=lambda trace: trace["name"])
def test_agentic_golden_trace(trace: dict[str, Any]) -> None:
    if trace["kind"] == "request_control_cancelled":
        _assert_request_control_cancelled(trace)
        return
    if trace["kind"] == "http_sequence":
        _assert_http_sequence_trace(trace)
        return
    assert trace["kind"] == "http"
    fake = TraceLLM(trace)
    app = create_app(_server_config(trace), llm=fake)
    _assert_http_exchange(
        TestClient(app),
        fake,
        endpoint=trace["endpoint"],
        request_payload=trace["request"],
        expected=trace["expected"],
    )


def _assert_http_sequence_trace(trace: dict[str, Any]) -> None:
    fake = TraceLLM(trace)
    app = create_app(_server_config(trace), llm=fake)
    client = TestClient(app)
    context: dict[str, Any] = {}
    for step in trace["steps"]:
        action = str(step.get("action") or "")
        if action == "expire_continuation":
            _expire_continuation(app, context[str(step.get("continuation_id", "continuation_id")).removeprefix("$")])
            continue
        if action:
            raise AssertionError(f"unsupported trace action {action!r}")
        endpoint = str(step.get("endpoint") or trace.get("endpoint"))
        method = str(step.get("method", "POST")).upper()
        request_payload = _resolve_trace_values(step.get("request", {}), context)
        expected = _resolve_trace_values(step["expected"], context)
        payload = _assert_http_exchange(
            client,
            fake,
            method=method,
            endpoint=endpoint,
            request_payload=request_payload,
            expected=expected,
        )
        _capture_trace_values(payload, expected=expected, context=context)


def _expire_continuation(app: Any, continuation_id: str) -> None:
    record = app.state.hipengine_continuations[continuation_id]
    app.state.hipengine_continuations[continuation_id] = replace(record, expires_at=0.0)


def _server_config(trace: dict[str, Any]) -> ServerConfig:
    options = {
        "model": "fake-path",
        "served_model_name": "fake-model",
        "eager_load": False,
    }
    options.update(dict(trace.get("server_config") or {}))
    return ServerConfig(**options)


def _assert_http_exchange(
    client: TestClient,
    fake: TraceLLM,
    *,
    method: str = "POST",
    endpoint: str,
    request_payload: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any] | None:
    call_index = len(fake.calls)
    if method == "POST":
        response = client.post(endpoint, json=request_payload)
    elif method == "GET":
        response = client.get(endpoint)
    elif method == "DELETE":
        response = client.delete(endpoint)
    else:  # pragma: no cover - fixture schema guard
        raise AssertionError(f"unsupported trace HTTP method {method!r}")

    assert response.status_code == expected["status_code"]
    _assert_response_exclusions(response.text, expected)
    if response.status_code >= 400:
        payload = response.json()
        _assert_error_response(payload, expected)
        return payload
    if request_payload.get("stream"):
        _assert_stream_response(response.text, expected)
        payload = None
    else:
        payload = response.json()
        if endpoint.endswith("/chat/completions"):
            _assert_chat_response(payload, expected)
        elif endpoint.endswith("/completions"):
            _assert_completion_response(payload, expected)
        else:
            _assert_generic_response(payload, expected)
    _assert_prompt_expectations(fake, expected, call_index=call_index)
    return payload


def _resolve_trace_values(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        return context.get(value[1:], value)
    if isinstance(value, list):
        return [_resolve_trace_values(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_trace_values(item, context) for key, item in value.items()}
    return value


def _capture_trace_values(
    payload: dict[str, Any] | None,
    *,
    expected: dict[str, Any],
    context: dict[str, Any],
) -> None:
    if payload is None:
        return
    if expected.get("continuation_id"):
        context["continuation_id"] = payload["choices"][0]["continuation_id"]
    if expected.get("capture_tool_call_id"):
        context["tool_call_id"] = payload["choices"][0]["message"]["tool_calls"][0]["id"]
    if expected.get("capture_snapshot"):
        context["snapshot"] = payload


def _assert_chat_response(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    choice = payload["choices"][0]
    if "finish_reason" in expected:
        assert choice["finish_reason"] == expected["finish_reason"]
    if "finish_details" in expected:
        assert choice["finish_details"] == _expected_finish_details(choice, expected)
    if expected.get("continuation_id"):
        assert choice["continuation_id"].startswith("gen_")
    if "logprobs" in expected:
        assert choice["logprobs"] == expected["logprobs"]
    message = choice["message"]
    if "message_content" in expected:
        assert message["content"] == expected["message_content"]
    if "reasoning_content" in expected:
        assert message["reasoning_content"] == expected["reasoning_content"]
    if expected.get("no_tool_calls"):
        assert "tool_calls" not in message
    if "tool_call" in expected:
        tool_call = message["tool_calls"][0]
        _assert_tool_call(tool_call, expected["tool_call"])
    if "tool_calls" in expected:
        assert len(message["tool_calls"]) == len(expected["tool_calls"])
        for actual, expected_call in zip(message["tool_calls"], expected["tool_calls"], strict=True):
            _assert_tool_call(actual, expected_call)


def _assert_completion_response(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    choice = payload["choices"][0]
    assert choice["text"] == expected["text"]
    assert choice["finish_reason"] == expected["finish_reason"]
    assert choice["finish_details"] == _expected_finish_details(choice, expected)
    if expected.get("continuation_id"):
        assert choice["continuation_id"].startswith("gen_")
    if "logprobs" in expected:
        assert choice["logprobs"] == expected["logprobs"]


def _assert_generic_response(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    if "object" in expected:
        assert payload["object"] == expected["object"]
    if "deleted" in expected:
        assert payload["deleted"] is expected["deleted"]
    if "restored" in expected:
        assert payload["restored"] is expected["restored"]
    if "rolled_back" in expected:
        assert payload["rolled_back"] is expected["rolled_back"]
    if "resident_state_reuse" in expected:
        assert payload["resident_state_reuse"] is expected["resident_state_reuse"]
    if "previous_message_count" in expected:
        assert payload["previous_message_count"] == expected["previous_message_count"]
    if "message_count" in expected:
        assert payload["message_count"] == expected["message_count"]
    if "snapshot_messages" in expected:
        assert payload["messages"] == expected["snapshot_messages"]


def _expected_finish_details(choice: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    details = dict(expected["finish_details"])
    if details.get("continuation_id") == "$continuation_id":
        details["continuation_id"] = choice["continuation_id"]
    return details


def _assert_stream_response(text: str, expected: dict[str, Any]) -> None:
    payloads = _sse_payloads(text)
    final = payloads[-1]
    assert final["choices"][0]["finish_reason"] == expected["finish_reason"]
    if "error_code" in expected:
        _assert_error_response(final, expected)
    if "hipengine_event" in expected:
        assert final["hipengine"]["event"] == expected["hipengine_event"]
    if "finish_details" in expected:
        assert final["choices"][0]["finish_details"] == expected["finish_details"]
    if "reasoning_content" in expected:
        reasoning_delta = next(
            payload
            for payload in payloads
            if payload["choices"][0]["delta"].get("reasoning_content")
        )
        assert reasoning_delta["choices"][0]["delta"]["reasoning_content"] == expected["reasoning_content"]
    if "tool_call" in expected:
        tool_delta = next(
            payload for payload in payloads if payload["choices"][0]["delta"].get("tool_calls")
        )
        tool_call = tool_delta["choices"][0]["delta"]["tool_calls"][0]
        _assert_tool_call(tool_call, expected["tool_call"])
    if "tool_calls" in expected:
        actual_calls = [
            call
            for payload in payloads
            for call in payload["choices"][0]["delta"].get("tool_calls", ())
        ]
        assert len(actual_calls) == len(expected["tool_calls"])
        for actual, expected_call in zip(actual_calls, expected["tool_calls"], strict=True):
            _assert_tool_call(actual, expected_call)


def _assert_error_response(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    error = payload["error"]
    assert error["code"] == expected["error_code"]
    if "error_param" in expected:
        assert error["param"] == expected["error_param"]
    assert error["hipengine"]["code"] == expected["hipengine_error_code"]
    if "fit_context" in expected:
        assert error["fit_context"] == expected["fit_context"]
    if "finish_details" in expected:
        assert error["finish_details"] == expected["finish_details"]


def _assert_response_exclusions(text: str, expected: dict[str, Any]) -> None:
    for needle in expected.get("response_excludes", ()):
        assert str(needle) not in text


def _assert_prompt_expectations(fake: TraceLLM, expected: dict[str, Any], *, call_index: int = 0) -> None:
    if call_index >= len(fake.calls):
        return
    prompt = fake.calls[call_index][0][0]
    for needle in expected.get("prompt_contains", ()):
        assert str(needle) in prompt
    for needle in expected.get("prompt_excludes", ()):
        assert str(needle) not in prompt
    if "prompt_endswith" in expected:
        assert prompt.endswith(str(expected["prompt_endswith"]))
    if "sampling" in expected:
        sampling = fake.calls[call_index][1]
        for field, value in expected["sampling"].items():
            actual = getattr(sampling, field)
            assert actual == _expected_sampling_value(actual, value)


def _expected_sampling_value(actual: Any, value: Any) -> Any:
    if isinstance(actual, tuple) and isinstance(value, list):
        return tuple(
            _expected_sampling_value(item_actual, item_value)
            for item_actual, item_value in zip(actual, value, strict=True)
        )
    return value


def _assert_tool_call(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    if "index" in expected:
        assert actual["index"] == expected["index"]
    assert actual["type"] == "function"
    assert actual["function"]["name"] == expected["name"]
    assert json.loads(actual["function"]["arguments"]) == expected["arguments"]


def _assert_request_control_cancelled(trace: dict[str, Any]) -> None:
    async def disconnected() -> bool:
        return True

    async def run() -> None:
        control = _RequestControl(disconnected=disconnected)
        with pytest.raises(OpenAIHTTPError) as raised:
            await _await_with_request_control(asyncio.sleep(0), control)
        exc = raised.value
        expected = trace["expected"]
        assert exc.status_code == expected["status_code"]
        assert exc.code == expected["error_code"]
        assert exc.finish_details == expected["finish_details"]

    asyncio.run(run())


def _sse_payloads(text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        if not raw_line.startswith("data: "):
            continue
        data = raw_line[6:]
        if data == "[DONE]":
            continue
        payloads.append(json.loads(data))
    return payloads
