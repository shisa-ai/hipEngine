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
            return self.detailed_outputs[: len(prompts)]
        if self.outputs:
            return self.outputs[: len(prompts)]
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
    assert trace["kind"] == "http"
    fake = TraceLLM(trace)
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=fake,
    )
    response = TestClient(app).post(trace["endpoint"], json=trace["request"])
    expected = trace["expected"]

    assert response.status_code == expected["status_code"]
    if response.status_code >= 400:
        _assert_error_response(response.json(), expected)
        return
    if trace["request"].get("stream"):
        _assert_stream_response(response.text, expected)
    elif trace["endpoint"].endswith("/chat/completions"):
        _assert_chat_response(response.json(), expected)
    else:
        _assert_completion_response(response.json(), expected)
    _assert_prompt_expectations(fake, expected)


def _assert_chat_response(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    choice = payload["choices"][0]
    if "finish_reason" in expected:
        assert choice["finish_reason"] == expected["finish_reason"]
    message = choice["message"]
    if "message_content" in expected:
        assert message["content"] == expected["message_content"]
    if "reasoning_content" in expected:
        assert message["reasoning_content"] == expected["reasoning_content"]
    if "tool_call" in expected:
        tool_call = message["tool_calls"][0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == expected["tool_call"]["name"]
        assert json.loads(tool_call["function"]["arguments"]) == expected["tool_call"]["arguments"]


def _assert_completion_response(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    choice = payload["choices"][0]
    assert choice["text"] == expected["text"]
    assert choice["finish_reason"] == expected["finish_reason"]
    assert choice["finish_details"] == expected["finish_details"]


def _assert_stream_response(text: str, expected: dict[str, Any]) -> None:
    payloads = _sse_payloads(text)
    final = payloads[-1]
    assert final["choices"][0]["finish_reason"] == expected["finish_reason"]
    if "tool_call" in expected:
        tool_delta = next(
            payload for payload in payloads if payload["choices"][0]["delta"].get("tool_calls")
        )
        tool_call = tool_delta["choices"][0]["delta"]["tool_calls"][0]
        assert tool_call["function"]["name"] == expected["tool_call"]["name"]
        assert json.loads(tool_call["function"]["arguments"]) == expected["tool_call"]["arguments"]


def _assert_error_response(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    error = payload["error"]
    assert error["code"] == expected["error_code"]
    assert error["hipengine"]["code"] == expected["hipengine_error_code"]
    assert error["finish_details"] == expected["finish_details"]


def _assert_prompt_expectations(fake: TraceLLM, expected: dict[str, Any]) -> None:
    if not fake.calls:
        return
    prompt = fake.calls[0][0][0]
    for needle in expected.get("prompt_contains", ()):
        assert str(needle) in prompt
    if "prompt_endswith" in expected:
        assert prompt.endswith(str(expected["prompt_endswith"]))


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
