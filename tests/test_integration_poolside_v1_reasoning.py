from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from hipengine import SamplingParams
from hipengine.generation import GenerationOutput
from hipengine.server import ServerConfig, create_app
from hipengine.server.api import _ReasoningSplitter
from hipengine.chat.poolside_v1 import (
    PoolsideV1ReasoningParser,
    render_poolside_v1_chat,
)

FIXTURES = Path(__file__).parent / "fixtures"
TEMPLATE_FIXTURE = FIXTURES / "laguna_poolside_v1_template.json"
REASONING_FIXTURE = FIXTURES / "laguna_poolside_v1_reasoning.json"


class _TinyPoolsideTokenizer:
    token_to_id = {"<think>": 18, "</think>": 19, "<assistant>": 23}

    def encode(self, text: str) -> list[int]:
        markers = tuple(self.token_to_id)
        ids: list[int] = []
        cursor = 0
        while cursor < len(text):
            marker = next((item for item in markers if text.startswith(item, cursor)), None)
            if marker is None:
                ids.append(700 + ord(text[cursor]) % 97)
                cursor += 1
                continue
            ids.append(self.token_to_id[marker])
            cursor += len(marker)
        return ids


class _PoolsideFakeLLM:
    chat_template_family = "poolside_v1"
    reasoning_parser_name = "poolside_v1"

    def __init__(self, *, output: str, stream_chunks: list[str] | None = None) -> None:
        self.output = str(output)
        self.stream_chunks = list(stream_chunks or [output])
        self.tokenizer = _TinyPoolsideTokenizer()
        self.chat_reasoning_parser = PoolsideV1ReasoningParser(self.tokenizer)
        self.calls: list[tuple[tuple[str, ...], SamplingParams]] = []
        self.stream_calls: list[tuple[str, SamplingParams]] = []
        self.max_sequence_length = 4096

    def render_chat_prompt(
        self,
        messages,
        *,
        tools=None,
        enable_thinking: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        return render_poolside_v1_chat(
            messages,
            tools=tools,
            enable_thinking=enable_thinking,
            add_generation_prompt=add_generation_prompt,
        )

    def prepare(
        self,
        *,
        max_sequence_length: int | None = None,
        sampling_params: SamplingParams,
    ) -> int:
        del sampling_params
        if max_sequence_length is not None:
            self.max_sequence_length = int(max_sequence_length)
        return self.max_sequence_length

    def generate_detailed(self, prompts, sampling_params: SamplingParams):
        prompt_tuple = tuple(str(prompt) for prompt in prompts)
        self.calls.append((prompt_tuple, sampling_params))
        return [GenerationOutput(text=self.output) for _prompt in prompt_tuple]

    def stream(self, prompt: str, sampling_params: SamplingParams):
        self.stream_calls.append((str(prompt), sampling_params))
        self.calls.append(((str(prompt),), sampling_params))
        yield from self.stream_chunks

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(str(text)))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))


def _fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sse_payloads(text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads


def test_poolside_v1_renderer_matches_frozen_template_cases() -> None:
    fixture = _fixture(TEMPLATE_FIXTURE)
    for case in fixture["cases"]:
        assert render_poolside_v1_chat(
            case["messages"],
            tools=case["tools"],
            enable_thinking=case["enable_thinking"],
            add_generation_prompt=case["add_generation_prompt"],
        ) == case["rendered"]


def test_poolside_v1_reasoning_scope_stops_at_current_assistant() -> None:
    fixture = _fixture(REASONING_FIXTURE)
    parser = PoolsideV1ReasoningParser(_TinyPoolsideTokenizer())

    for case in fixture["scope_cases"]:
        assert parser.is_reasoning_end(case["token_ids"]) is case["reasoning_end"]
        assert parser.initially_open_ids(case["token_ids"]) is case["initially_open"]


def test_poolside_v1_reasoning_fixture_splits_fragmented_markers() -> None:
    fixture = _fixture(REASONING_FIXTURE)
    for case in fixture["output_cases"]:
        splitter = _ReasoningSplitter(initially_open=case["initially_open"])
        parts = []
        for chunk in case["chunks"]:
            parts.extend(splitter.feed(chunk))
        parts.extend(splitter.finish())
        assert "".join(text for field, text in parts if field == "content") == case["content"]
        assert "".join(
            text for field, text in parts if field == "reasoning_content"
        ) == case["reasoning_content"]
        assert "<think>" not in "".join(text for _field, text in parts)
        assert "</think>" not in "".join(text for _field, text in parts)


def test_poolside_v1_server_capabilities_name_chat_and_reasoning_contract() -> None:
    fake = _PoolsideFakeLLM(output="Direct answer.")
    client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    )

    response = client.get("/v1/hipengine/capabilities")

    assert response.status_code == 200
    assert response.json()["chat_template"] == {
        "family": "poolside_v1",
        "reasoning_parser": "poolside_v1",
        "reasoning_tags": True,
        "tool_call_tags": True,
    }


def test_poolside_v1_server_blocking_uses_prompt_initial_reasoning_state() -> None:
    template = {case["name"]: case for case in _fixture(TEMPLATE_FIXTURE)["cases"]}
    fake = _PoolsideFakeLLM(output="Inspect state.</think>Final answer.")
    client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": template["thinking"]["messages"],
            "enable_thinking": True,
        },
    )

    assert response.status_code == 200
    assert fake.calls[0][0] == (template["thinking"]["rendered"],)
    assert response.json()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Final answer.",
        "reasoning_content": "Inspect state.",
    }
    assert "<think>" not in response.text
    assert "</think>" not in response.text


def test_poolside_v1_server_no_thinking_and_stop_boundary_do_not_leak_markers() -> None:
    template = {case["name"]: case for case in _fixture(TEMPLATE_FIXTURE)["cases"]}
    direct = _PoolsideFakeLLM(output="Direct answer.")
    direct_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=direct)
    )
    direct_response = direct_client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": template["no_thinking"]["messages"],
            "enable_thinking": False,
        },
    )
    assert direct_response.status_code == 200
    assert direct.calls[0][0] == (template["no_thinking"]["rendered"],)
    assert direct_response.json()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Direct answer.",
    }

    stopped = _PoolsideFakeLLM(output="Stopped plan.</think>Never visible")
    stopped_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=stopped)
    )
    stopped_response = stopped_client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": template["thinking"]["messages"],
            "enable_thinking": True,
            "stop": "</think>",
        },
    )
    assert stopped_response.status_code == 200
    assert stopped_response.json()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Stopped plan.",
    }
    assert "</think>" not in stopped_response.text
    assert "Never visible" not in stopped_response.text


def test_poolside_v1_server_buffered_multi_stream_keeps_prompt_open_state() -> None:
    fake = _PoolsideFakeLLM(output="Inspect state.</think>Final answer.")
    client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "enable_thinking": True,
            "stream": True,
            "n": 2,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    visible = [
        (payload["choices"][0]["index"], payload["choices"][0]["delta"])
        for payload in payloads
        if payload.get("choices")
        and (
            payload["choices"][0]["delta"].get("content")
            or payload["choices"][0]["delta"].get("reasoning_content")
        )
    ]
    assert visible == [
        (0, {"reasoning_content": "Inspect state."}),
        (0, {"content": "Final answer."}),
        (1, {"reasoning_content": "Inspect state."}),
        (1, {"content": "Final answer."}),
    ]


def test_poolside_v1_server_streaming_splits_fragmented_close_marker() -> None:
    fake = _PoolsideFakeLLM(
        output="Inspect state.</think>Final answer.",
        stream_chunks=["Inspect state.", "</th", "ink>", "Final ", "answer."],
    )
    client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "enable_thinking": True,
            "stream": True,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    deltas = [
        payload["choices"][0]["delta"]
        for payload in payloads
        if payload.get("choices")
        and (
            payload["choices"][0]["delta"].get("content")
            or payload["choices"][0]["delta"].get("reasoning_content")
        )
    ]
    assert deltas == [
        {"reasoning_content": "Inspect state."},
        {"content": "Final "},
        {"content": "answer."},
    ]
    assert "<think>" not in response.text
    assert "</think>" not in response.text
