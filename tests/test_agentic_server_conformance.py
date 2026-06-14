from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from hipengine import SamplingParams
from hipengine.generation import GenerationOutput
from hipengine.server import ServerConfig, create_app


class AgenticFakeLLM:
    def __init__(
        self,
        *,
        outputs: list[str] | None = None,
        stream_chunks: list[str] | None = None,
    ) -> None:
        self.outputs = list(outputs or ())
        self.stream_chunks = list(stream_chunks or ())
        self.calls: list[tuple[tuple[str, ...], SamplingParams]] = []
        self.stream_calls: list[tuple[str, SamplingParams]] = []

    def generate(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
        prompts = tuple(str(prompt) for prompt in prompts)
        self.calls.append((prompts, sampling_params))
        if self.outputs:
            return [GenerationOutput(text=output) for output in self.outputs[: len(prompts)]]
        return [GenerationOutput(text=f"generated:{prompt}") for prompt in prompts]

    def stream(self, prompt: str, sampling_params: SamplingParams):
        prompt = str(prompt)
        self.stream_calls.append((prompt, sampling_params))
        self.calls.append(((prompt,), sampling_params))
        yield from self.stream_chunks or self.outputs or [f"generated:{prompt}"]

    def count_tokens(self, text: str) -> int:
        return len(str(text).split())


def _client(llm: AgenticFakeLLM) -> TestClient:
    return TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
            llm=llm,
        )
    )


def _read_tool(*, strict: bool = True) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a repository file.",
            "strict": strict,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "mode": {"type": "string", "enum": ["raw", "summary"]},
                },
                "required": ["path", "mode"],
                "additionalProperties": False,
            },
        },
    }


def test_agentic_conformance_strict_reasoning_tool_call_response_shape() -> None:
    llm = AgenticFakeLLM(
        outputs=[
            (
                "<think>inspect first</think>"
                '<tool_call>{"name":"read","arguments":{"path":"README.md","mode":"summary"}}</tool_call>'
            )
        ]
    )
    response = _client(llm).post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [
                {"role": "developer", "content": "Use tools carefully."},
                {"role": "user", "content": "Summarize README.md."},
            ],
            "tools": [_read_tool()],
            "tool_choice": {"type": "function", "function": {"name": "read"}},
            "parallel_tool_calls": False,
            "reasoning": {"enabled": True, "effort": "low"},
            "max_tokens": 2048,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["finish_details"] == {
        "reason": "tool_calls",
        "cache_action": "append_none",
        "reasoning_tokens": 2,
        "tool_call_tokens": 1,
        "phase": "tool_call",
    }
    message = choice["message"]
    assert message["role"] == "assistant"
    assert message["content"] == ""
    assert message["reasoning_content"] == "inspect first"
    assert "<tool_call>" not in json.dumps(message)
    tool_call = message["tool_calls"][0]
    assert tool_call["id"].startswith("call_")
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "read"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "path": "README.md",
        "mode": "summary",
    }

    prompt = llm.calls[0][0][0]
    assert prompt.count("\n<tools>\n") == 1
    assert "You must call the function named 'read'." in prompt
    assert "close </think> before exceeding 512 hidden reasoning tokens" in prompt
    assert "reserve at least 512 tokens for the final answer or tool call" in prompt


def test_agentic_conformance_tool_result_replay_renders_once() -> None:
    llm = AgenticFakeLLM(outputs=["README summary: hello."])
    response = _client(llm).post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [
                {"role": "user", "content": "Read the README."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": '{"path":"README.md","mode":"summary"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "hello"},
            ],
            "tools": [_read_tool()],
            "enable_thinking": False,
            "session": {"commit": "append_none"},
            "max_tokens": 64,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == {"reason": "stop", "cache_action": "append_none"}
    assert choice["message"] == {"role": "assistant", "content": "README summary: hello."}

    prompt = llm.calls[0][0][0]
    rendered_call = (
        '<tool_call>{"name":"read","arguments":{"path":"README.md","mode":"summary"}}</tool_call>'
    )
    assert prompt.count(rendered_call) == 1
    assert prompt.count("<tool_response>\nhello\n</tool_response>") == 1
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_agentic_conformance_permissive_duplicated_tool_start_recovers_call() -> None:
    raw_tool_markup = (
        "<tool_call>\n"
        '<tool_call>{"name":"read","arguments":{"path":"README.md","mode":"raw"}}</tool_call>'
    )
    llm = AgenticFakeLLM(outputs=[raw_tool_markup])
    response = _client(llm).post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Read README.md."}],
            "tools": [_read_tool(strict=False)],
            "max_tokens": 64,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["finish_details"] == {
        "reason": "tool_calls",
        "cache_action": "append_none",
        "tool_call_tokens": 2,
        "phase": "tool_call",
    }
    message = choice["message"]
    assert message["content"] == ""
    assert "<tool_call>" not in json.dumps(message)
    tool_call = message["tool_calls"][0]
    assert tool_call["function"]["name"] == "read"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "path": "README.md",
        "mode": "raw",
    }


def test_agentic_conformance_permissive_malformed_tool_json_remains_text() -> None:
    raw_tool_markup = '<tool_call>{"name":"read","arguments":</tool_call>'
    llm = AgenticFakeLLM(outputs=[raw_tool_markup])
    response = _client(llm).post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Read README.md."}],
            "tools": [_read_tool(strict=False)],
            "max_tokens": 64,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == {"reason": "stop", "cache_action": "append_none"}
    assert choice["message"] == {"role": "assistant", "content": raw_tool_markup}


def test_agentic_conformance_streaming_tool_call_matches_non_streaming_shape() -> None:
    llm = AgenticFakeLLM(
        stream_chunks=[
            (
                "<think>need file</think>"
                '<tool_call>{"name":"read","arguments":{"path":"README.md","mode":"raw"}}</tool_call>'
            )
        ]
    )
    response = _client(llm).post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Read README.md."}],
            "tools": [_read_tool()],
            "parallel_tool_calls": False,
            "stream": True,
            "stream_options": {"include_hipengine": True},
            "max_tokens": 64,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(
        "<tool_call>" in json.dumps(payload["choices"][0]["delta"])
        for payload in payloads
        if payload.get("choices") and payload["choices"][0].get("delta")
    )

    reasoning = next(payload for payload in payloads if payload["choices"][0]["delta"].get("reasoning_content"))
    assert reasoning["choices"][0]["delta"]["reasoning_content"] == "need file"

    tool_payload = next(payload for payload in payloads if payload["choices"][0]["delta"].get("tool_calls"))
    tool_call = tool_payload["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "read"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "path": "README.md",
        "mode": "raw",
    }

    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_reason"] == "tool_calls"
    assert done["choices"][0]["finish_details"] == {
        "reason": "tool_calls",
        "cache_action": "append_none",
        "reasoning_tokens": 2,
        "tool_call_tokens": 1,
        "phase": "tool_call",
    }
    assert done["choices"][0]["hipengine"]["phase"] == "done"
    assert done["choices"][0]["hipengine"]["finish_details"] == {
        "reason": "tool_calls",
        "cache_action": "append_none",
        "reasoning_tokens": 2,
        "tool_call_tokens": 1,
        "phase": "tool_call",
    }
    assert llm.stream_calls


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
