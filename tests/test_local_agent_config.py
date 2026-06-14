from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hipengine.server.api import ServerConfig, create_app


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_local_agent_config.py"
CONFIG_PATH = REPO_ROOT / "docs" / "examples" / "local-agent" / "openai-compatible.json"

_SPEC = importlib.util.spec_from_file_location("validate_local_agent_config", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
validate_local_agent_config = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validate_local_agent_config)


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
            "tools": {"enabled": True, "strict_decoding": False},
            "reasoning_controls": {
                "enabled": True,
                "fields": [
                    "reasoning_effort",
                    "enable_thinking",
                    "chat_template_kwargs",
                    "thinking",
                    "reasoning",
                ],
            },
            "request_timeouts": {"timeout_ms": True},
            "token_diagnostics": {
                "tokenize": True,
                "detokenize": True,
                "count_tokens": True,
                "fit_context": True,
            },
        },
        "unsupported_fields": [
            "continuation_id",
            "session.commit",
            "response_format",
            "parallel_tool_calls",
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
