from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from hipengine import SamplingParams
from hipengine.chat.poolside_v1 import PoolsideV1ToolParser, render_poolside_v1_chat
from hipengine.generation import GenerationOutput
from hipengine.server import ServerConfig, create_app

FIXTURE = Path(__file__).parent / "fixtures" / "laguna_poolside_v1_tools.json"


class _PoolsideToolFakeLLM:
    chat_template_family = "poolside_v1"
    reasoning_parser_name = "poolside_v1"
    tool_parser_name = "poolside_v1"

    def __init__(self, output: str, *, stream_chunks: list[str] | None = None) -> None:
        self.output = str(output)
        self.stream_chunks = list(stream_chunks or [self.output])
        self.chat_tool_parser = PoolsideV1ToolParser()
        self.calls: list[tuple[tuple[str, ...], SamplingParams]] = []
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
        self.calls.append(((str(prompt),), sampling_params))
        yield from self.stream_chunks

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(str(text))



def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))



def _sse_payloads(text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads



def test_poolside_v1_tool_parser_matches_frozen_contract() -> None:
    fixture = _fixture()
    parser = PoolsideV1ToolParser()

    for case in fixture["cases"]:
        parsed = parser.parse(case["output"], tools=fixture["tools"])
        assert parsed.content == case["content"], case["name"]
        assert [
            {"name": call.name, "arguments": json.loads(call.arguments)}
            for call in parsed.tool_calls
        ] == case["calls"], case["name"]
        assert bool(parsed.invalid_blocks) is case["invalid"], case["name"]



def test_poolside_v1_blocking_emits_openai_tool_call_and_finish_reason() -> None:
    fixture = _fixture()
    case = next(item for item in fixture["cases"] if item["name"] == "newline_less_single")
    fake = _PoolsideToolFakeLLM(case["output"])
    client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": fixture["tools"],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["finish_details"]["reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    assert len(choice["message"]["tool_calls"]) == 1
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["id"].startswith("call_")
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "city": "Paris",
        "days": 3,
    }
    assert "<available_tools>" in fake.calls[0][0][0]



def test_poolside_v1_streaming_fragments_arguments_with_stable_call_id() -> None:
    fixture = _fixture()
    content = "    " + "x = 'quoted \\\" value'\n" * 20
    output = (
        "<tool_call>write_file"
        "<arg_key>content</arg_key><arg_value>"
        f"{content}</arg_value>"
        "<arg_key>mode</arg_key><arg_value>420</arg_value>"
        "</tool_call>"
    )
    chunks = [output[:7], output[7:31], output[31:83], output[83:197], output[197:]]
    fake = _PoolsideToolFakeLLM(output, stream_chunks=chunks)
    client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Write the file."}],
            "tools": fixture["tools"],
            "stream": True,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    call_deltas = [
        choice["delta"]["tool_calls"][0]
        for payload in payloads
        for choice in payload.get("choices", [])
        if choice.get("delta", {}).get("tool_calls")
    ]
    assert len(call_deltas) >= 2
    assert len({delta["id"] for delta in call_deltas}) == 1
    assert call_deltas[0]["function"]["name"] == "write_file"
    assert all("name" not in delta["function"] for delta in call_deltas[1:])
    arguments = "".join(delta["function"]["arguments"] for delta in call_deltas)
    assert json.loads(arguments) == {"content": content, "mode": 420}
    done = next(
        choice
        for payload in payloads
        for choice in payload.get("choices", [])
        if choice.get("finish_reason") is not None
    )
    assert done["finish_reason"] == "tool_calls"
    assert done["finish_details"]["reason"] == "tool_calls"
    assert "<tool_call>" not in response.text
    assert "<arg_key>" not in response.text



def test_poolside_v1_multiple_calls_require_parallel_opt_in_and_have_distinct_ids() -> None:
    fixture = _fixture()
    case = next(item for item in fixture["cases"] if item["name"] == "multiple_adjacent_calls")

    rejected = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=_PoolsideToolFakeLLM(case["output"]),
        )
    ).post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Weather in both cities?"}],
            "tools": fixture["tools"],
        },
    )
    assert rejected.status_code == 200
    assert rejected.json()["choices"][0]["finish_details"]["reason"] == "invalid_tool_call"
    assert "tool_calls" not in rejected.json()["choices"][0]["message"]

    accepted = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=_PoolsideToolFakeLLM(case["output"]),
        )
    ).post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "Weather in both cities?"}],
            "tools": fixture["tools"],
            "parallel_tool_calls": True,
        },
    )
    assert accepted.status_code == 200
    calls = accepted.json()["choices"][0]["message"]["tool_calls"]
    assert [call["function"]["name"] for call in calls] == [
        "get_weather",
        "get_weather",
    ]
    assert len({call["id"] for call in calls}) == 2
    assert [json.loads(call["function"]["arguments"])["city"] for call in calls] == [
        "Paris",
        "Tokyo",
    ]



def test_poolside_v1_partial_and_malformed_calls_fail_closed() -> None:
    fixture = _fixture()
    invalid_cases = [case for case in fixture["cases"] if case["invalid"]]

    for case in invalid_cases:
        fake = _PoolsideToolFakeLLM(case["output"])
        client = TestClient(
            create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-model",
                "messages": [{"role": "user", "content": "Use the tool."}],
                "tools": fixture["tools"],
            },
        )

        assert response.status_code == 200, case["name"]
        choice = response.json()["choices"][0]
        assert choice["finish_reason"] == "stop", case["name"]
        assert choice["finish_details"]["reason"] == "invalid_tool_call", case["name"]
        assert choice["message"] == {"role": "assistant", "content": ""}, case["name"]
        assert case["output"] not in response.text, case["name"]



def test_poolside_v1_capabilities_advertise_xml_parser() -> None:
    fake = _PoolsideToolFakeLLM("No tool.")
    client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    )

    response = client.get("/v1/hipengine/capabilities")

    assert response.status_code == 200
    tools = response.json()["features"]["tools"]
    assert tools["format"] == "poolside_v1_xml"
    assert tools["parser"] == "poolside_v1"
    assert tools["string_argument_whitespace"] == "schema_typed_verbatim"
    assert tools["incremental_string_values"] is True
