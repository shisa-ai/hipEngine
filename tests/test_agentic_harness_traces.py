from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hipengine import SamplingParams
from hipengine.generation import FinishDetails, GenerationOutput
from hipengine.server import ServerConfig, create_app
from hipengine.server.api import OpenAIHTTPError, _RequestControl, _await_with_request_control


TRACE_PATH = Path(__file__).resolve().parent / "fixtures" / "agentic_traces" / "golden_traces.json"


class TraceLLM:
    def __init__(self, trace: dict[str, Any]) -> None:
        self.outputs = list(trace.get("fake_outputs", ()))
        self.stream_chunks = list(trace.get("fake_stream_chunks", ()))
        self.detailed_outputs = [
            GenerationOutput(
                text=str(item["text"]),
                finish_details=FinishDetails(**dict(item["finish_details"])),
            )
            for item in trace.get("fake_detailed_outputs", ())
        ]
        self.generate_delay_s = float(trace.get("generate_delay_s", 0.0))
        self.calls: list[tuple[tuple[str, ...], SamplingParams]] = []
        self.stream_calls: list[tuple[str, SamplingParams]] = []

    def generate(self, prompts, sampling_params: SamplingParams) -> list[Any]:
        prompts = tuple(str(prompt) for prompt in prompts)
        self.calls.append((prompts, sampling_params))
        if self.generate_delay_s > 0.0:
            time.sleep(self.generate_delay_s)
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
        yield from self.stream_chunks or self.outputs or [f"generated:{prompt}"]

    def count_tokens(self, text: str) -> int:
        return len(str(text).split())


def _load_traces() -> list[dict[str, Any]]:
    with TRACE_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["schema"] == "hipengine.agentic_traces.v1"
    return list(payload["traces"])


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
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=fake,
    )
    _assert_http_exchange(
        TestClient(app),
        fake,
        endpoint=trace["endpoint"],
        request_payload=trace["request"],
        expected=trace["expected"],
    )


def _assert_http_sequence_trace(trace: dict[str, Any]) -> None:
    fake = TraceLLM(trace)
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=fake,
    )
    client = TestClient(app)
    context: dict[str, Any] = {}
    for step in trace["steps"]:
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


def _assert_generic_response(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    if "object" in expected:
        assert payload["object"] == expected["object"]
    if "deleted" in expected:
        assert payload["deleted"] is expected["deleted"]
    if "restored" in expected:
        assert payload["restored"] is expected["restored"]
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
