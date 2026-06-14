from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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
                "strict_decoding": False,
                "strict_result_validation": True,
            },
            "tools": {
                "enabled": True,
                "strict_decoding": False,
                "strict_result_validation": True,
                "parallel_tool_calls": True,
                "no_tool_start_suppression": True,
                "required_tool_start_forcing": True,
                "required_tool_start_forcing_scope": "no_tokenized_thinking_budget",
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
            },
        },
        "unsupported_fields": [
            "continuation_id",
            "session.id",
        ],
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
    assert payload["tool_choice"] == "auto"
    assert payload["tools"][0]["function"]["name"] == "record_result"
    for unsupported in config["chat_completions"]["do_not_send"]:
        assert unsupported not in payload


def test_local_agent_config_rejects_missing_unsupported_blocklist() -> None:
    config = validate_local_agent_config.load_config(CONFIG_PATH)
    config["chat_completions"]["do_not_send"] = []

    with pytest.raises(validate_local_agent_config.ConfigValidationError, match="do_not_send"):
        validate_local_agent_config.validate_config_against_capabilities(config, _capabilities())


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


def test_pi_agent_models_validator_rejects_reasoning_disabled() -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)
    config["providers"]["hipengine-local"]["models"][0]["reasoning"] = False

    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="reasoning must be true"):
        validate_pi_agent_models.validate_pi_models_config(config)


def test_pi_agent_models_validator_rejects_missing_qwen_thinking_format() -> None:
    config = validate_pi_agent_models.load_config(PI_CONFIG_PATH)
    config["providers"]["hipengine-local"]["compat"].pop("thinkingFormat")

    with pytest.raises(validate_pi_agent_models.PiConfigValidationError, match="thinkingFormat"):
        validate_pi_agent_models.validate_pi_models_config(config)
