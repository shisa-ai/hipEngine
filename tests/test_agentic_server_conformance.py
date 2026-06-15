from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from hipengine import SamplingParams
from hipengine.generation import FinishDetails, GenerationOutput
from hipengine.server import ServerConfig, create_app


class AgenticFakeLLM:
    def __init__(
        self,
        *,
        outputs: list[str] | None = None,
        stream_chunks: list[str] | None = None,
        detailed_outputs: list[GenerationOutput] | None = None,
    ) -> None:
        self.outputs = list(outputs or ())
        self.stream_chunks = list(stream_chunks or ())
        self.detailed_outputs = list(detailed_outputs or ())
        self.calls: list[tuple[tuple[str, ...], SamplingParams]] = []
        self.stream_calls: list[tuple[str, SamplingParams]] = []

    def generate(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
        prompts = tuple(str(prompt) for prompt in prompts)
        self.calls.append((prompts, sampling_params))
        if self.detailed_outputs:
            if len(self.detailed_outputs) < len(prompts):
                raise AssertionError("not enough fake detailed generation output left")
            outputs = self.detailed_outputs[: len(prompts)]
            del self.detailed_outputs[: len(prompts)]
            return outputs
        if self.outputs:
            if len(self.outputs) < len(prompts):
                raise AssertionError("not enough fake generation output left")
            outputs = self.outputs[: len(prompts)]
            del self.outputs[: len(prompts)]
            return [GenerationOutput(text=output) for output in outputs]
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


def test_agentic_conformance_reasoning_final_answer_response_shape() -> None:
    llm = AgenticFakeLLM(outputs=["<think>plan first</think>final answer"])
    response = _client(llm).post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Answer directly."}],
            "reasoning": {"enabled": True, "effort": "minimal"},
            "max_tokens": 64,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == {"reason": "stop", "cache_action": "append_none"}
    message = choice["message"]
    assert message == {
        "role": "assistant",
        "content": "final answer",
        "reasoning_content": "plan first",
    }
    assert "<think>" not in json.dumps(message)
    prompt = llm.calls[0][0][0]
    assert "close </think> before exceeding 32 hidden reasoning tokens" in prompt
    assert "reserve at least 32 tokens for the final answer or tool call" in prompt


def test_agentic_conformance_reasoning_structured_json_response_shape() -> None:
    llm = AgenticFakeLLM(outputs=['<think>shape it</think>{"result":"ok"}'])
    response = _client(llm).post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Return status."}],
            "response_format": {"type": "json_object"},
            "reasoning": {"enabled": True, "effort": "minimal"},
            "max_tokens": 64,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == {"reason": "stop", "cache_action": "append_none"}
    message = choice["message"]
    assert message["role"] == "assistant"
    assert message["reasoning_content"] == "shape it"
    assert json.loads(message["content"]) == {"result": "ok"}
    assert "<think>" not in json.dumps(message)
    prompt = llm.calls[0][0][0]
    assert "Return only one valid JSON object in the final answer." in prompt
    assert "close </think> before exceeding 32 hidden reasoning tokens" in prompt
    assert "reserve at least 32 tokens for the final answer or tool call" in prompt


def test_agentic_conformance_streaming_reasoning_structured_json_shape() -> None:
    llm = AgenticFakeLLM(
        outputs=['<think>shape it</think>{"result":"ok"}'],
        stream_chunks=["should-not-stream-raw"],
    )
    response = _client(llm).post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Return status."}],
            "response_format": {"type": "json_object"},
            "reasoning": {"enabled": True, "effort": "minimal"},
            "stream": True,
            "stream_options": {"include_hipengine": True},
            "max_tokens": 64,
        },
    )

    assert response.status_code == 200
    assert "should-not-stream-raw" not in response.text
    assert "<think>" not in response.text
    payloads = _sse_payloads(response.text)
    reasoning = next(payload for payload in payloads if payload["choices"][0]["delta"].get("reasoning_content"))
    assert reasoning["choices"][0]["delta"] == {"reasoning_content": "shape it"}
    assert reasoning["choices"][0]["hipengine"]["phase"] == "think"

    structured = next(payload for payload in payloads if payload["choices"][0]["delta"].get("content"))
    assert structured["choices"][0]["delta"] == {"content": '{"result":"ok"}'}
    assert json.loads(structured["choices"][0]["delta"]["content"]) == {"result": "ok"}
    assert structured["choices"][0]["hipengine"]["phase"] == "structured"
    assert structured["choices"][0]["hipengine"]["decode_state"]["structured_tokens"] == 1

    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_reason"] == "stop"
    assert done["choices"][0]["finish_details"] == {"reason": "stop", "cache_action": "append_none"}
    assert done["choices"][0]["hipengine"]["phase"] == "structured"
    assert done["choices"][0]["hipengine"]["finish_details"] == {
        "reason": "stop",
        "cache_action": "append_none",
    }
    assert llm.calls
    assert not llm.stream_calls


def test_agentic_conformance_continuation_resume_answer_shape() -> None:
    llm = AgenticFakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="partial answer",
                finish_details=FinishDetails(reason="length", length_limit=5),
            ),
            GenerationOutput(
                text=" complete.",
                finish_details=FinishDetails(reason="eos", eos_token_id=151645),
            ),
        ]
    )
    client = _client(llm)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "answer briefly"}],
            "max_tokens": 5,
            "temperature": 0.0,
        },
    )

    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    continuation_id = first_choice["continuation_id"]
    assert continuation_id.startswith("gen_")
    assert first_choice["finish_reason"] == "length"
    assert first_choice["message"] == {"role": "assistant", "content": "partial answer"}
    assert first_choice["finish_details"] == {
        "reason": "length",
        "length_limit": 5,
        "cache_action": "append_none",
        "phase": "answer",
        "continuation_eligible": True,
        "continuation_id": continuation_id,
    }

    second = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert second.status_code == 200
    second_choice = second.json()["choices"][0]
    assert second_choice["finish_reason"] == "stop"
    assert second_choice["message"] == {"role": "assistant", "content": "partial answer complete."}
    assert second_choice["finish_details"] == {
        "reason": "eos",
        "eos_token_id": 151645,
        "cache_action": "append_none",
    }
    assert llm.calls[1][0][0].endswith("partial answer")

    reused = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 1},
    )

    assert reused.status_code == 400
    assert reused.json()["error"]["code"] == "invalid_continuation"
    assert reused.json()["error"]["param"] == "continuation_id"
    assert len(llm.calls) == 2


def test_agentic_conformance_visible_session_replays_tool_loop_without_reasoning() -> None:
    llm = AgenticFakeLLM(
        outputs=[
            (
                "<think>need file</think>"
                '<tool_call>{"name":"read","arguments":{"path":"README.md","mode":"raw"}}</tool_call>'
            ),
            "README says hello.",
        ]
    )
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=llm,
    )
    client = TestClient(app)
    tools = [_read_tool()]

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Read README.md."}],
            "tools": tools,
            "session": {"id": "sess_agentic_tool", "commit": "append_visible_only"},
            "max_tokens": 64,
        },
    )
    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    assert first_choice["finish_reason"] == "tool_calls"
    assert first_choice["finish_details"] == {
        "reason": "tool_calls",
        "cache_action": "append_visible_only",
        "reasoning_tokens": 2,
        "tool_call_tokens": 1,
        "phase": "tool_call",
    }
    first_message = first_choice["message"]
    assert first_message["reasoning_content"] == "need file"
    tool_call = first_message["tool_calls"][0]

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": "hello",
                }
            ],
            "tools": tools,
            "session": {"id": "sess_agentic_tool", "commit": "append_none"},
            "max_tokens": 64,
        },
    )

    assert second.status_code == 200
    second_choice = second.json()["choices"][0]
    assert second_choice["finish_reason"] == "stop"
    assert second_choice["message"] == {"role": "assistant", "content": "README says hello."}
    assert second_choice["finish_details"] == {"reason": "stop", "cache_action": "append_none"}

    rendered_call = (
        '<tool_call>{"name":"read","arguments":{"path":"README.md","mode":"raw"}}</tool_call>'
    )
    prompt = llm.calls[1][0][0]
    assert prompt.count(rendered_call) == 1
    assert prompt.count("<tool_response>\nhello\n</tool_response>") == 1
    assert "need file" not in prompt
    assert "<think>need file</think>" not in prompt


def test_agentic_conformance_snapshot_restore_replays_tool_loop_without_reasoning() -> None:
    llm = AgenticFakeLLM(
        outputs=[
            (
                "<think>need file</think>"
                '<tool_call>{"name":"read","arguments":{"path":"README.md","mode":"raw"}}</tool_call>'
            ),
            "README says hello after restore.",
        ]
    )
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=llm,
    )
    client = TestClient(app)
    tools = [_read_tool()]

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Read README.md."}],
            "tools": tools,
            "session": {"id": "sess_agentic_restore", "commit": "append_visible_only"},
            "max_tokens": 64,
        },
    )
    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    assert first_choice["finish_reason"] == "tool_calls"
    first_message = first_choice["message"]
    assert first_message["reasoning_content"] == "need file"
    tool_call = first_message["tool_calls"][0]

    exported = client.get("/v1/hipengine/sessions/sess_agentic_restore/snapshot")
    deleted = client.delete("/v1/hipengine/sessions/sess_agentic_restore")
    restored = client.post(
        "/v1/hipengine/sessions/sess_agentic_restore/snapshot",
        json=exported.json(),
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": "hello",
                }
            ],
            "tools": tools,
            "session": {"id": "sess_agentic_restore", "commit": "append_none"},
            "max_tokens": 64,
        },
    )

    assert exported.status_code == 200
    snapshot = exported.json()
    assert snapshot["messages"] == [
        {"role": "user", "content": "Read README.md."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call],
        },
    ]
    assert "need file" not in json.dumps(snapshot)
    assert deleted.json()["deleted"] is True
    assert restored.status_code == 200
    assert restored.json()["message_count"] == 2
    assert second.status_code == 200
    second_choice = second.json()["choices"][0]
    assert second_choice["finish_reason"] == "stop"
    assert second_choice["message"] == {
        "role": "assistant",
        "content": "README says hello after restore.",
    }

    rendered_call = (
        '<tool_call>{"name":"read","arguments":{"path":"README.md","mode":"raw"}}</tool_call>'
    )
    prompt = llm.calls[1][0][0]
    assert prompt.count(rendered_call) == 1
    assert prompt.count("<tool_response>\nhello\n</tool_response>") == 1
    assert "need file" not in prompt
    assert "<think>need file</think>" not in prompt


def test_agentic_conformance_strict_duplicated_tool_start_recovers_call() -> None:
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
            "tools": [_read_tool(strict=True)],
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


def test_agentic_conformance_permissive_malformed_tool_json_fails_closed() -> None:
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
    assert choice["finish_details"] == {
        "reason": "invalid_tool_call",
        "cache_action": "append_none",
    }
    assert choice["message"] == {"role": "assistant", "content": ""}
    assert "<tool_call>" not in response.text


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
    tool_delta = tool_payload["choices"][0]["delta"]
    assert set(tool_delta) == {"tool_calls"}
    assert len(tool_delta["tool_calls"]) == 1
    tool_call = tool_delta["tool_calls"][0]
    assert set(tool_call) == {"index", "id", "type", "function"}
    assert tool_call["index"] == 0
    assert tool_call["id"].startswith("call_")
    assert tool_call["type"] == "function"
    assert set(tool_call["function"]) == {"name", "arguments"}
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


def test_agentic_conformance_streaming_parallel_tool_loop_continues_from_tool_results() -> None:
    raw_calls = (
        '<tool_call>{"name":"read","arguments":{"path":"README.md","mode":"summary"}}</tool_call>'
        '<tool_call>{"name":"read","arguments":{"path":"WORKLOG.md","mode":"summary"}}</tool_call>'
    )
    llm = AgenticFakeLLM(
        outputs=["Both files summarized."],
        stream_chunks=[raw_calls],
    )
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=llm,
    )
    client = TestClient(app)
    tools = [_read_tool()]

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Read README.md and WORKLOG.md."}],
            "tools": tools,
            "parallel_tool_calls": True,
            "stream": True,
            "stream_options": {"include_hipengine": True},
            "max_tokens": 64,
        },
    )

    assert first.status_code == 200
    assert "<tool_call>" not in first.text
    first_payloads = _sse_payloads(first.text)
    tool_delta_payloads = [
        payload
        for payload in first_payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("tool_calls")
    ]
    assert [set(payload["choices"][0]["delta"]) for payload in tool_delta_payloads] == [
        {"tool_calls"},
        {"tool_calls"},
    ]
    tool_deltas = [payload["choices"][0]["delta"]["tool_calls"][0] for payload in tool_delta_payloads]
    assert [set(delta) for delta in tool_deltas] == [
        {"index", "id", "type", "function"},
        {"index", "id", "type", "function"},
    ]
    assert [delta["index"] for delta in tool_deltas] == [0, 1]
    assert [delta["type"] for delta in tool_deltas] == ["function", "function"]
    assert [delta["function"]["name"] for delta in tool_deltas] == ["read", "read"]
    assert [json.loads(delta["function"]["arguments"]) for delta in tool_deltas] == [
        {"path": "README.md", "mode": "summary"},
        {"path": "WORKLOG.md", "mode": "summary"},
    ]
    assert tool_deltas[0]["id"] != tool_deltas[1]["id"]
    first_done = next(payload for payload in first_payloads if payload["choices"][0]["finish_reason"])
    assert first_done["choices"][0]["finish_reason"] == "tool_calls"
    assert first_done["choices"][0]["finish_details"] == {
        "reason": "tool_calls",
        "cache_action": "append_none",
        "tool_call_tokens": 2,
        "phase": "tool_call",
    }

    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [
                {"role": "user", "content": "Read README.md and WORKLOG.md."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tool_deltas[0]["id"],
                            "type": "function",
                            "function": {
                                "name": tool_deltas[0]["function"]["name"],
                                "arguments": tool_deltas[0]["function"]["arguments"],
                            },
                        },
                        {
                            "id": tool_deltas[1]["id"],
                            "type": "function",
                            "function": {
                                "name": tool_deltas[1]["function"]["name"],
                                "arguments": tool_deltas[1]["function"]["arguments"],
                            },
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": tool_deltas[0]["id"], "content": "README summary"},
                {"role": "tool", "tool_call_id": tool_deltas[1]["id"], "content": "WORKLOG summary"},
            ],
            "tools": tools,
            "session": {"commit": "append_none"},
            "max_tokens": 64,
        },
    )

    assert second.status_code == 200
    second_choice = second.json()["choices"][0]
    assert second_choice["finish_reason"] == "stop"
    assert second_choice["message"] == {"role": "assistant", "content": "Both files summarized."}
    assert second_choice["finish_details"] == {"reason": "stop", "cache_action": "append_none"}

    prompt = llm.calls[1][0][0]
    readme_call = (
        '<tool_call>{"name":"read","arguments":{"path":"README.md","mode":"summary"}}</tool_call>'
    )
    worklog_call = (
        '<tool_call>{"name":"read","arguments":{"path":"WORKLOG.md","mode":"summary"}}</tool_call>'
    )
    assert prompt.count(readme_call) == 1
    assert prompt.count(worklog_call) == 1
    assert prompt.count("<tool_response>\nREADME summary\n</tool_response>") == 1
    assert prompt.count("<tool_response>\nWORKLOG summary\n</tool_response>") == 1


def test_agentic_conformance_streaming_malformed_tool_json_fails_closed() -> None:
    llm = AgenticFakeLLM(
        stream_chunks=[
            '<tool_call>{"name":"read","arguments":</tool_call>',
        ]
    )
    response = _client(llm).post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Read README.md."}],
            "tools": [_read_tool()],
            "stream": True,
            "stream_options": {"include_hipengine": True},
            "max_tokens": 64,
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0]["delta"].get("tool_calls") for payload in payloads)

    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_reason"] == "stop"
    assert done["choices"][0]["finish_details"] == {
        "reason": "invalid_tool_call",
        "cache_action": "append_none",
    }
    assert done["choices"][0]["hipengine"]["finish_details"] == {
        "reason": "invalid_tool_call",
        "cache_action": "append_none",
    }


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
