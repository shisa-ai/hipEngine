from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hipengine.generation import GenerationOutput
from hipengine.server.api import ServerConfig, create_app


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_local_agent_config.py"
PI_SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_pi_agent_models.py"
CONFIG_PATH = REPO_ROOT / "docs" / "examples" / "local-agent" / "openai-compatible.json"
PI_CONFIG_PATH = REPO_ROOT / "docs" / "examples" / "pi-agent" / "models.json"

_SPEC = importlib.util.spec_from_file_location("validate_local_agent_config", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
validate_local_agent_config = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validate_local_agent_config)

_PI_SPEC = importlib.util.spec_from_file_location("validate_pi_agent_models", PI_SCRIPT_PATH)
assert _PI_SPEC is not None and _PI_SPEC.loader is not None
validate_pi_agent_models = importlib.util.module_from_spec(_PI_SPEC)
_PI_SPEC.loader.exec_module(validate_pi_agent_models)


class _FakeLLM:
    tokenizer = None

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(range(len(text.split())))

    def detokenize(self, token_ids) -> str:
        return " ".join(str(token_id) for token_id in token_ids)

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class _PiToolSmokeLLM(_FakeLLM):
    def __init__(self) -> None:
        self.calls = []

    def generate(self, prompts, sampling_params):
        prompts = tuple(str(prompt) for prompt in prompts)
        self.calls.append((prompts, sampling_params))
        return [
            GenerationOutput(
                text='<tool_call>{"name":"record_result","arguments":{"result":"ok"}}</tool_call>'
            )
            for _ in prompts
        ]

    def stream(self, prompt, sampling_params):
        self.calls.append(((str(prompt),), sampling_params))
        yield '<tool_call>{"name":"record_result","arguments":{"result":"ok"}}</tool_call>'


class _PiReasoningSmokeLLM(_FakeLLM):
    def __init__(self) -> None:
        self.calls = []

    def generate(self, prompts, sampling_params):
        prompts = tuple(str(prompt) for prompt in prompts)
        self.calls.append((prompts, sampling_params))
        return [GenerationOutput(text="<think>brief check</think>OK") for _ in prompts]


def _capabilities(**overrides):
    payload = {
        "model": {"id": "fake-model"},
        "features": {
            "chat_completions": True,
            "streaming": True,
            "stream_options": {"include_usage": True, "include_hipengine": True},
            "structured_outputs": {
                "response_format": True,
                "json_object": True,
                "json_schema": True,
                "guided_json": True,
                "guided_regex": True,
                "guided_choice": True,
                "strict_decoding": False,
                "strict_result_validation": True,
            },
            "tools": {
                "enabled": True,
                "strict_decoding": False,
                "strict_result_validation": True,
                "parallel_tool_calls": True,
                "streaming_argument_chunks": True,
                "streaming_argument_chunk_chars": 128,
                "no_tool_start_suppression": True,
                "required_tool_start_forcing": True,
                "required_tool_start_forcing_scope": "initial_or_after_tokenized_thinking_close",
                "specific_tool_name_prefix_forcing": True,
                "tool_call_close_repair": True,
            },
            "reasoning_controls": {
                "enabled": True,
                "fields": [
                    "reasoning_effort",
                    "enable_thinking",
                    "max_think_tokens",
                    "min_answer_tokens",
                    "hard_think_cap",
                    "soft_close_window",
                    "hard_close_message",
                    "hard_close_sequence",
                    "thinking_token_budget",
                    "chat_template_kwargs",
                    "thinking",
                    "reasoning",
                ],
                "budget_policy": "prompt_hint_plus_tokenized_soft_and_hard_close",
                "token_budget": True,
                "token_budget_enforced": True,
                "effort_defaults": {
                    "minimal": {"hard_think_cap": 256, "soft_close_window": 64, "min_answer_tokens": 256},
                    "low": {"hard_think_cap": 512, "soft_close_window": 128, "min_answer_tokens": 512},
                    "medium": {"hard_think_cap": 4096, "soft_close_window": 512, "min_answer_tokens": 1024},
                    "high": {"hard_think_cap": 16384, "soft_close_window": 1024, "min_answer_tokens": 2048},
                    "xhigh": {"hard_think_cap": 32768, "soft_close_window": 2048, "min_answer_tokens": 4096},
                    "max": {"hard_think_cap": 32768, "soft_close_window": 2048, "min_answer_tokens": 4096},
                },
                "effort_default_clamp": "request_max_tokens_chat_default_or_remaining_context",
                "hard_close_token_forcing": True,
                "soft_close_bias": True,
                "eos_suppression": True,
            },
            "request_timeouts": {
                "timeout_ms": True,
                "cooperative_backend_deadline": True,
                "cooperative_backend_cancel": True,
                "preemptive_decode_cancel": False,
            },
            "token_diagnostics": {
                "tokenize": True,
                "detokenize": True,
                "count_tokens": True,
                "fit_context": True,
                "session_aware_chat": True,
            },
        },
        "unsupported_fields": [],
    }
    payload.update(overrides)
    return payload


def test_local_agent_config_matches_capabilities() -> None:
    config = validate_local_agent_config.load_config(CONFIG_PATH)

    summary = validate_local_agent_config.validate_config_against_capabilities(
        config, _capabilities()
    )

    assert summary["model"] == "fake-model"
    assert summary["streaming"] is True
    assert summary["tools"] is True
    assert "parallel_tool_calls" in summary["blocked_fields"]
    assert "top_logprobs" in summary["blocked_fields"]
    assert "grammar" in summary["blocked_fields"]
    assert "guided_json" not in summary["blocked_fields"]
    for unsupported in _capabilities()["unsupported_fields"]:
        assert unsupported in summary["blocked_fields"]


def test_local_agent_config_matches_server_capabilities_manifest() -> None:
    config = validate_local_agent_config.load_config(CONFIG_PATH)
    app = create_app(
        ServerConfig(
            model="/models/fake",
            served_model_name="fake-model",
            eager_load=False,
            request_timeout_ms=250.0,
        ),
        llm=_FakeLLM(),
    )
    capabilities = TestClient(app).get("/v1/hipengine/capabilities").json()

    summary = validate_local_agent_config.validate_config_against_capabilities(
        config, capabilities
    )

    assert summary["model"] == "fake-model"
    for unsupported in capabilities["unsupported_fields"]:
        assert unsupported in summary["blocked_fields"]


def test_local_agent_chat_smoke_payload_avoids_unsupported_fields() -> None:
    config = validate_local_agent_config.load_config(CONFIG_PATH)
    payload = validate_local_agent_config.build_chat_smoke_payload(config, _capabilities())

    assert payload["model"] == "fake-model"
    assert payload["stream"] is False
    assert payload["max_tokens"] == 8
    assert payload["session"] == {"commit": "append_none"}
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "record_result"}}
    assert payload["tools"][0]["function"]["name"] == "record_result"
    for unsupported in config["chat_completions"]["do_not_send"]:
        assert unsupported not in payload


def test_local_agent_chat_smoke_response_requires_tool_call_when_enabled() -> None:
    summary = validate_local_agent_config.validate_chat_smoke_response(
        {
            "object": "chat.completion",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "record_result",
                                    "arguments": json.dumps({"result": "ok"}),
                                },
                            }
                        ],
                    },
                }
            ],
        },
        expect_tool_call=True,
    )

    assert summary == {
        "finish_reason": "tool_calls",
        "tool_name": "record_result",
        "argument_keys": ["result"],
        "result": "ok",
    }


def test_local_agent_chat_smoke_response_allows_text_when_tools_disabled() -> None:
    summary = validate_local_agent_config.validate_chat_smoke_response(
        {
            "object": "chat.completion",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
        },
        expect_tool_call=False,
    )

    assert summary == {"finish_reason": "stop"}


def test_local_agent_chat_smoke_response_rejects_missing_tool_call() -> None:
    with pytest.raises(validate_local_agent_config.ConfigValidationError, match="tool_calls"):
        validate_local_agent_config.validate_chat_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            },
            expect_tool_call=True,
        )


def test_local_agent_chat_smoke_response_rejects_raw_tool_call_markup() -> None:
    with pytest.raises(validate_local_agent_config.ConfigValidationError, match="raw <tool_call> text"):
        validate_local_agent_config.validate_chat_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '<tool_call>{"name":"record_result","arguments":{"result":"ok"}}</tool_call>',
                        },
                    }
                ],
            },
            expect_tool_call=True,
        )


def test_local_agent_chat_smoke_response_rejects_raw_tool_call_content_leak() -> None:
    with pytest.raises(validate_local_agent_config.ConfigValidationError, match="raw <tool_call> text"):
        validate_local_agent_config.validate_chat_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": '<tool_call>{"name":"record_result","arguments":{"result":"ok"}}</tool_call>',
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "record_result",
                                        "arguments": json.dumps({"result": "ok"}),
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            expect_tool_call=True,
        )


def test_local_agent_chat_smoke_response_rejects_raw_tool_call_reasoning_leak() -> None:
    with pytest.raises(
        validate_local_agent_config.ConfigValidationError,
        match=r"message\.reasoning_content",
    ):
        validate_local_agent_config.validate_chat_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": (
                                '<tool_call>{"name":"record_result","arguments":{"result":"ok"}}</tool_call>'
                            ),
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "record_result",
                                        "arguments": json.dumps({"result": "ok"}),
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            expect_tool_call=True,
        )


def test_local_agent_chat_smoke_response_rejects_wrong_tool_argument() -> None:
    with pytest.raises(validate_local_agent_config.ConfigValidationError, match="result.*'ok'"):
        validate_local_agent_config.validate_chat_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "record_result",
                                        "arguments": json.dumps({"result": "not ok"}),
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            expect_tool_call=True,
        )


def test_local_agent_chat_smoke_response_rejects_non_assistant_message() -> None:
    with pytest.raises(validate_local_agent_config.ConfigValidationError, match="message.role"):
        validate_local_agent_config.validate_chat_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {"role": "tool", "tool_calls": []},
                    }
                ],
            },
            expect_tool_call=True,
        )


def test_local_agent_chat_smoke_response_rejects_content_with_tool_call() -> None:
    with pytest.raises(validate_local_agent_config.ConfigValidationError, match="assistant content"):
        validate_local_agent_config.validate_chat_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "I will call the tool.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "record_result",
                                        "arguments": json.dumps({"result": "ok"}),
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
            expect_tool_call=True,
        )


@pytest.mark.parametrize(
    ("tool_call", "match"),
    [
        (
            {
                "type": "function",
                "function": {"name": "record_result", "arguments": json.dumps({"result": "ok"})},
            },
            "missing=.*id",
        ),
        (
            {
                "id": "call_1",
                "type": "custom",
                "function": {"name": "record_result", "arguments": json.dumps({"result": "ok"})},
            },
            r"\.type must be 'function'",
        ),
        (
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "record_result",
                    "arguments": json.dumps({"result": "ok"}),
                    "extra": True,
                },
            },
            "extra=.*extra",
        ),
        (
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "record_result", "arguments": {"result": "ok"}},
            },
            r"\.function\.arguments must be a JSON string",
        ),
    ],
)
def test_local_agent_chat_smoke_response_rejects_malformed_tool_call_shape(
    tool_call: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(validate_local_agent_config.ConfigValidationError, match=match):
        validate_local_agent_config.validate_chat_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "tool_calls": [tool_call],
                        },
                    }
                ],
            },
            expect_tool_call=True,
        )


def test_local_agent_config_rejects_missing_unsupported_blocklist() -> None:
    config = validate_local_agent_config.load_config(CONFIG_PATH)
    config["chat_completions"]["do_not_send"] = []

    with pytest.raises(validate_local_agent_config.ConfigValidationError, match="do_not_send"):
        validate_local_agent_config.validate_config_against_capabilities(
            config,
            _capabilities(unsupported_fields=["session.id"]),
        )


def test_local_agent_config_rejects_strict_tool_decoding_when_unavailable() -> None:
    config = validate_local_agent_config.load_config(CONFIG_PATH)
    config["chat_completions"]["tool_calling"]["strict_decoding_required"] = True

    with pytest.raises(
        validate_local_agent_config.ConfigValidationError,
        match="strict tool decoding",
    ):
        validate_local_agent_config.validate_config_against_capabilities(config, _capabilities())


def test_pi_agent_models_config_enables_qwen_thinking() -> None:
    config = json.loads(PI_CONFIG_PATH.read_text(encoding="utf-8"))
    provider = config["providers"]["hipengine-local"]
    model = provider["models"][0]

    assert provider["baseUrl"].endswith("/v1")
    assert provider["compat"]["thinkingFormat"] == "qwen"
    assert provider["compat"]["supportsReasoningEffort"] is False
    assert model["reasoning"] is True
    assert model["input"] == ["text"]


def test_pi_agent_models_config_validates_with_helper() -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)

    summary = validate_pi_agent_models.validate_pi_models_config(config)

    assert summary["provider_count"] == 1
    assert summary["model_count"] == 1
    assert summary["providers"][0]["provider"] == "hipengine-local"
    assert summary["providers"][0]["models"][0]["reasoning"] is True


def test_pi_agent_models_config_matches_server_capabilities_manifest() -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)
    model_id = config["providers"]["hipengine-local"]["models"][0]["id"]
    app = create_app(
        ServerConfig(
            model="/models/fake",
            served_model_name=model_id,
            eager_load=False,
            max_context_tokens=131072,
        ),
        llm=_FakeLLM(),
    )
    capabilities = TestClient(app).get("/v1/hipengine/capabilities").json()

    summary = validate_pi_agent_models.validate_pi_models_against_capabilities(
        config,
        capabilities,
    )

    assert summary["capability_model"] == model_id
    assert summary["qwen_thinking"] is True
    assert summary["tools"] is True
    assert summary["streaming_usage"] is True


def test_pi_agent_models_validator_rejects_live_model_mismatch() -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)

    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="not listed"):
        validate_pi_agent_models.validate_pi_models_against_capabilities(
            config,
            {
                "model": {"id": "other-model"},
                "context": {"effective_max_context_tokens": 131072},
                "features": {
                    "chat_completions": True,
                    "streaming": True,
                    "stream_options": {"include_usage": True},
                    "tools": {"enabled": True},
                    "reasoning_controls": {"enabled": True, "fields": ["enable_thinking"]},
                },
            },
        )


def test_pi_agent_chat_smoke_payload_uses_qwen_tool_shape() -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)
    model_id = config["providers"]["hipengine-local"]["models"][0]["id"]

    payload = validate_pi_agent_models.build_pi_chat_smoke_payload(
        config,
        {"model": {"id": model_id}},
    )

    assert payload["model"] == model_id
    assert payload["temperature"] == 0
    assert payload["enable_thinking"] is False
    assert payload["session"] == {"commit": "append_none"}
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "record_result"}}
    assert payload["tools"][0]["function"]["name"] == "record_result"


def test_pi_agent_streaming_chat_smoke_payload_uses_usage_sse() -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)
    model_id = config["providers"]["hipengine-local"]["models"][0]["id"]

    payload = validate_pi_agent_models.build_pi_streaming_chat_smoke_payload(
        config,
        {"model": {"id": model_id}},
    )

    assert payload["model"] == model_id
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["temperature"] == 0
    assert payload["enable_thinking"] is False
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "record_result"}}
    assert payload["tools"][0]["function"]["name"] == "record_result"


def test_pi_agent_reasoning_smoke_payload_uses_qwen_thinking() -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)
    model_id = config["providers"]["hipengine-local"]["models"][0]["id"]

    payload = validate_pi_agent_models.build_pi_reasoning_smoke_payload(
        config,
        {"model": {"id": model_id}},
    )

    assert payload["model"] == model_id
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 96
    assert payload["enable_thinking"] is True
    assert payload["session"] == {"commit": "append_none"}
    assert "tools" not in payload
    assert "tool_choice" not in payload


def test_pi_agent_chat_smoke_payload_round_trips_through_server() -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)
    model_id = config["providers"]["hipengine-local"]["models"][0]["id"]
    llm = _PiToolSmokeLLM()
    client = TestClient(
        create_app(
            ServerConfig(
                model="/models/fake",
                served_model_name=model_id,
                eager_load=False,
            ),
            llm=llm,
        )
    )
    capabilities = client.get("/v1/hipengine/capabilities").json()
    payload = validate_pi_agent_models.build_pi_chat_smoke_payload(config, capabilities)

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert validate_pi_agent_models.validate_pi_chat_smoke_response(body) == {
        "finish_reason": "tool_calls",
        "tool_name": "record_result",
        "argument_keys": ["result"],
        "result": "ok",
    }
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["finish_details"] == {
        "reason": "tool_calls",
        "cache_action": "append_none",
        "tool_call_tokens": 1,
        "phase": "tool_call",
    }
    assert choice["message"]["content"] == ""
    assert "<tool_call>" not in json.dumps(choice["message"])
    assert len(llm.calls) == 1
    prompt = llm.calls[0][0][0]
    assert "record_result" in prompt
    assert "You must call the function named 'record_result'." in prompt
    assert "<think>\n\n</think>" in prompt


def test_pi_agent_streaming_chat_smoke_payload_round_trips_through_server() -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)
    model_id = config["providers"]["hipengine-local"]["models"][0]["id"]
    llm = _PiToolSmokeLLM()
    client = TestClient(
        create_app(
            ServerConfig(
                model="/models/fake",
                served_model_name=model_id,
                eager_load=False,
            ),
            llm=llm,
        )
    )
    capabilities = client.get("/v1/hipengine/capabilities").json()
    payload = validate_pi_agent_models.build_pi_streaming_chat_smoke_payload(config, capabilities)

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert validate_pi_agent_models.validate_pi_streaming_chat_smoke_response(response.text) == {
        "finish_reason": "tool_calls",
        "tool_name": "record_result",
        "argument_keys": ["result"],
        "result": "ok",
        "sse_payloads": 4,
        "usage_chunk": True,
        "done": True,
    }
    assert "<tool_call>" not in response.text
    assert len(llm.calls) == 1
    prompt = llm.calls[0][0][0]
    assert "record_result" in prompt
    assert "You must call the function named 'record_result'." in prompt
    assert "<think>\n\n</think>" in prompt


def test_pi_agent_reasoning_smoke_payload_round_trips_through_server() -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)
    model_id = config["providers"]["hipengine-local"]["models"][0]["id"]
    llm = _PiReasoningSmokeLLM()
    client = TestClient(
        create_app(
            ServerConfig(
                model="/models/fake",
                served_model_name=model_id,
                eager_load=False,
            ),
            llm=llm,
        )
    )
    capabilities = client.get("/v1/hipengine/capabilities").json()
    payload = validate_pi_agent_models.build_pi_reasoning_smoke_payload(config, capabilities)

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert validate_pi_agent_models.validate_pi_reasoning_smoke_response(body) == {
        "finish_reason": "stop",
        "reasoning_chars": len("brief check"),
        "answer_chars": len("OK"),
    }
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"] == {
        "role": "assistant",
        "content": "OK",
        "reasoning_content": "brief check",
    }
    assert "<think>" not in json.dumps(choice["message"])
    assert len(llm.calls) == 1
    prompt = llm.calls[0][0][0]
    assert "Think briefly, then answer with exactly OK." in prompt
    assert "<think>\n\n</think>" not in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_pi_agent_chat_smoke_response_requires_tool_call() -> None:
    summary = validate_pi_agent_models.validate_pi_chat_smoke_response(
        {
            "object": "chat.completion",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "record_result",
                                    "arguments": json.dumps({"result": "ok"}),
                                },
                            }
                        ],
                    },
                }
            ],
        }
    )

    assert summary == {
        "finish_reason": "tool_calls",
        "tool_name": "record_result",
        "argument_keys": ["result"],
        "result": "ok",
    }


def test_pi_agent_streaming_chat_smoke_response_reconstructs_tool_call() -> None:
    summary = validate_pi_agent_models.validate_pi_streaming_chat_smoke_response(
        "\n".join(
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"record_result","arguments":"{\\"result\\""}}]},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\\"ok\\"}"}}]},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
                "data: [DONE]",
            ]
        )
    )

    assert summary == {
        "finish_reason": "tool_calls",
        "tool_name": "record_result",
        "argument_keys": ["result"],
        "result": "ok",
        "sse_payloads": 4,
        "usage_chunk": True,
        "done": True,
    }


def test_pi_agent_streaming_chat_smoke_runner_posts_sse_request(monkeypatch) -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)
    model_id = config["providers"]["hipengine-local"]["models"][0]["id"]
    capabilities = {"model": {"id": model_id}}
    calls: dict[str, Any] = {}

    def fake_request_text(
        method,
        url,
        *,
        api_key=None,
        payload=None,
        timeout,
        accept="text/plain",
    ):
        calls["request"] = {
            "method": method,
            "url": url,
            "api_key": api_key,
            "payload": payload,
            "timeout": timeout,
            "accept": accept,
        }
        return "\n".join(
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"record_result","arguments":"{\\"result\\""}}]},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\\"ok\\"}"}}]},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
                "data: [DONE]",
            ]
        )

    monkeypatch.setattr(validate_pi_agent_models, "_request_text", fake_request_text)

    summary = validate_pi_agent_models.run_pi_streaming_chat_smoke(
        "http://server.test/v1",
        config,
        capabilities,
        api_key="secret",
        timeout=12.5,
    )

    assert summary == {
        "finish_reason": "tool_calls",
        "tool_name": "record_result",
        "argument_keys": ["result"],
        "result": "ok",
        "sse_payloads": 4,
        "usage_chunk": True,
        "done": True,
    }
    request = calls["request"]
    assert request["method"] == "POST"
    assert request["url"] == "http://server.test/v1/chat/completions"
    assert request["api_key"] == "secret"
    assert request["timeout"] == 12.5
    assert request["accept"] == "text/event-stream"
    payload = request["payload"]
    assert payload["model"] == model_id
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "record_result"}}


def test_pi_agent_reasoning_smoke_response_requires_reasoning_content() -> None:
    summary = validate_pi_agent_models.validate_pi_reasoning_smoke_response(
        {
            "object": "chat.completion",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "OK",
                        "reasoning_content": "brief check",
                    },
                }
            ],
        }
    )

    assert summary == {
        "finish_reason": "stop",
        "reasoning_chars": len("brief check"),
        "answer_chars": len("OK"),
    }


def test_pi_agent_chat_smoke_response_rejects_missing_tool_call() -> None:
    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="tool_calls"):
        validate_pi_agent_models.validate_pi_chat_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
            }
        )


def test_pi_agent_streaming_chat_smoke_response_rejects_missing_usage() -> None:
    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="usage SSE payload"):
        validate_pi_agent_models.validate_pi_streaming_chat_smoke_response(
            "\n".join(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"record_result","arguments":"{\\"result\\":\\"ok\\"}"}}]},"finish_reason":null}]}',
                    'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        )


def test_pi_agent_streaming_chat_smoke_response_rejects_raw_tool_call_markup() -> None:
    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="raw <tool_call> text"):
        validate_pi_agent_models.validate_pi_streaming_chat_smoke_response(
            "\n".join(
                [
                    'data: {"choices":[{"delta":{"content":"<tool_call>{}"},"finish_reason":null}]}',
                    'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
                    "data: [DONE]",
                ]
            )
        )


def test_pi_agent_streaming_smoke_cli_enters_live_mode_without_base_url(monkeypatch, capsys) -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)
    provider = config["providers"]["hipengine-local"]
    model_id = provider["models"][0]["id"]
    capabilities = {
        "model": {"id": model_id},
        "context": {"effective_max_context_tokens": provider["models"][0]["contextWindow"]},
        "features": {
            "chat_completions": True,
            "streaming": True,
            "stream_options": {"include_usage": True},
            "tools": {"enabled": True},
            "reasoning_controls": {"enabled": True, "fields": ["enable_thinking"]},
        },
    }
    calls: dict[str, object] = {}

    def fake_fetch_capabilities(base_url, *, api_key=None, timeout=10.0):
        calls["fetch"] = {"base_url": base_url, "api_key": api_key, "timeout": timeout}
        return capabilities

    def fake_streaming_smoke(base_url, config, capabilities, *, api_key=None, timeout=30.0):
        calls["streaming"] = {
            "base_url": base_url,
            "api_key": api_key,
            "timeout": timeout,
            "model": capabilities["model"]["id"],
        }
        return {
            "finish_reason": "tool_calls",
            "tool_name": "record_result",
            "argument_keys": ["result"],
            "result": "ok",
            "sse_payloads": 4,
            "usage_chunk": True,
            "done": True,
        }

    monkeypatch.setattr(validate_pi_agent_models, "fetch_capabilities", fake_fetch_capabilities)
    monkeypatch.setattr(validate_pi_agent_models, "run_pi_streaming_chat_smoke", fake_streaming_smoke)

    status = validate_pi_agent_models.main(
        ["--config", str(PI_CONFIG_PATH), "--streaming-smoke"]
    )

    assert status == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["streaming_smoke"]["done"] is True
    assert "chat_smoke" not in out
    assert "reasoning_smoke" not in out
    assert calls == {
        "fetch": {
            "base_url": provider["baseUrl"],
            "api_key": provider["apiKey"],
            "timeout": 10.0,
        },
        "streaming": {
            "base_url": provider["baseUrl"],
            "api_key": provider["apiKey"],
            "timeout": 30.0,
            "model": model_id,
        },
    }


@pytest.mark.parametrize(
    "content",
    [
        '<tool_call>{"name":"record_result","arguments":{"result":"ok"}}</tool_call>',
        '<tool_call>\n<tool_call>{"name":"record_result","arguments":{"result":"ok"}}</tool_call>',
    ],
)
def test_pi_agent_chat_smoke_response_rejects_raw_tool_call_markup(content: str) -> None:
    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="raw <tool_call> text"):
        validate_pi_agent_models.validate_pi_chat_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                    }
                ],
            }
        )


def test_pi_agent_chat_smoke_response_rejects_raw_tool_call_content_leak() -> None:
    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="raw <tool_call> text"):
        validate_pi_agent_models.validate_pi_chat_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": '<tool_call>{"name":"record_result","arguments":{"result":"ok"}}</tool_call>',
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "record_result",
                                        "arguments": json.dumps({"result": "ok"}),
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        )


def test_pi_agent_chat_smoke_response_rejects_raw_tool_call_reasoning_leak() -> None:
    with pytest.raises(
        validate_pi_agent_models.PiConfigValidationError,
        match=r"message\.reasoning_content",
    ):
        validate_pi_agent_models.validate_pi_chat_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": (
                                '<tool_call>{"name":"record_result","arguments":{"result":"ok"}}</tool_call>'
                            ),
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "record_result",
                                        "arguments": json.dumps({"result": "ok"}),
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        )


def test_pi_agent_reasoning_smoke_response_rejects_missing_reasoning_content() -> None:
    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="reasoning_content"):
        validate_pi_agent_models.validate_pi_reasoning_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "OK"},
                    }
                ],
            }
        )


def test_pi_agent_reasoning_smoke_response_rejects_raw_think_markup() -> None:
    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="raw <think> text"):
        validate_pi_agent_models.validate_pi_reasoning_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "<think>brief</think>OK"},
                    }
                ],
            }
        )


def test_pi_agent_reasoning_smoke_response_rejects_reasoning_content_with_tags() -> None:
    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="think tags"):
        validate_pi_agent_models.validate_pi_reasoning_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "OK",
                            "reasoning_content": "<think>brief</think>",
                        },
                    }
                ],
            }
        )


def test_pi_agent_chat_smoke_response_rejects_wrong_tool_argument() -> None:
    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="result.*'ok'"):
        validate_pi_agent_models.validate_pi_chat_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "record_result",
                                        "arguments": json.dumps({"result": "not ok"}),
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        )


def test_pi_agent_chat_smoke_response_rejects_non_assistant_message() -> None:
    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="message.role"):
        validate_pi_agent_models.validate_pi_chat_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {"role": "tool", "tool_calls": []},
                    }
                ],
            }
        )


def test_pi_agent_chat_smoke_response_rejects_content_with_tool_call() -> None:
    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="assistant content"):
        validate_pi_agent_models.validate_pi_chat_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "I will call the tool.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "record_result",
                                        "arguments": json.dumps({"result": "ok"}),
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    ("tool_call", "match"),
    [
        (
            {
                "type": "function",
                "function": {"name": "record_result", "arguments": json.dumps({"result": "ok"})},
            },
            "missing=.*id",
        ),
        (
            {
                "id": "call_1",
                "type": "custom",
                "function": {"name": "record_result", "arguments": json.dumps({"result": "ok"})},
            },
            r"\.type must be 'function'",
        ),
        (
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "record_result",
                    "arguments": json.dumps({"result": "ok"}),
                    "extra": True,
                },
            },
            "extra=.*extra",
        ),
        (
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "record_result", "arguments": {"result": "ok"}},
            },
            r"\.function\.arguments must be a JSON string",
        ),
    ],
)
def test_pi_agent_chat_smoke_response_rejects_malformed_tool_call_shape(
    tool_call: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match=match):
        validate_pi_agent_models.validate_pi_chat_smoke_response(
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "tool_calls": [tool_call],
                        },
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    ("tool_delta", "match"),
    [
        (
            {
                "index": 0,
                "type": "function",
                "function": {"name": "record_result", "arguments": '{"result":"ok"}'},
            },
            "first fragment.*id",
        ),
        (
            {
                "index": 0,
                "id": "call_1",
                "type": "custom",
                "function": {"name": "record_result", "arguments": '{"result":"ok"}'},
            },
            r"\.type must be 'function'",
        ),
        (
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"arguments": '{"result":"ok"}'},
            },
            r"\.function\.name is required",
        ),
        (
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "record_result", "arguments": {"result": "ok"}},
            },
            r"\.function\.arguments must be a string",
        ),
        (
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "record_result", "arguments": '{"result":"ok"}'},
                "unexpected": True,
            },
            "unsupported OpenAI tool-call fields",
        ),
    ],
)
def test_pi_agent_streaming_chat_smoke_response_rejects_malformed_tool_call_delta(
    tool_delta: dict[str, Any],
    match: str,
) -> None:
    response = "\n".join(
        [
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"tool_calls": [tool_delta]},
                            "finish_reason": None,
                        }
                    ]
                },
                separators=(",", ":"),
            ),
            'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
            "data: [DONE]",
        ]
    )
    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match=match):
        validate_pi_agent_models.validate_pi_streaming_chat_smoke_response(response)


def test_pi_agent_models_validator_rejects_reasoning_disabled() -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)
    config["providers"]["hipengine-local"]["models"][0]["reasoning"] = False

    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="reasoning must be true"):
        validate_pi_agent_models.validate_pi_models_config(config)


def test_pi_agent_models_validator_reports_multiple_thinking_misconfigs() -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)
    config["providers"] = {"epyc": config["providers"]["hipengine-local"]}
    provider = config["providers"]["epyc"]
    provider["baseUrl"] = "http://epyc:8000/v1"
    provider["compat"].pop("thinkingFormat")
    provider["compat"]["supportsUsageInStreaming"] = False
    provider["models"][0]["reasoning"] = False
    provider["models"][0]["contextWindow"] = 262144

    with pytest.raises(validate_pi_agent_models.PiConfigValidationError) as raised:
        validate_pi_agent_models.validate_pi_models_config(config)

    message = str(raised.value)
    assert "providers.epyc.compat.thinkingFormat must be 'qwen'" in message
    assert "providers.epyc.compat.supportsUsageInStreaming must be true" in message
    assert "providers.epyc.models[0].reasoning must be true" in message


def test_pi_agent_models_validator_rejects_missing_qwen_thinking_format() -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)
    config["providers"]["hipengine-local"]["compat"].pop("thinkingFormat")

    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="thinkingFormat"):
        validate_pi_agent_models.validate_pi_models_config(config)


def test_pi_agent_models_validator_rejects_streaming_usage_disabled() -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)
    config["providers"]["hipengine-local"]["compat"]["supportsUsageInStreaming"] = False

    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="supportsUsageInStreaming"):
        validate_pi_agent_models.validate_pi_models_config(config)


def test_pi_agent_models_validator_rejects_context_window_above_server() -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)
    model_id = config["providers"]["hipengine-local"]["models"][0]["id"]

    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="contextWindow exceeds"):
        validate_pi_agent_models.validate_pi_models_against_capabilities(
            config,
            {
                "model": {"id": model_id},
                "context": {"effective_max_context_tokens": 65536},
                "features": {
                    "chat_completions": True,
                    "streaming": True,
                    "stream_options": {"include_usage": True},
                    "tools": {"enabled": True},
                    "reasoning_controls": {"enabled": True, "fields": ["enable_thinking"]},
                },
            },
        )
