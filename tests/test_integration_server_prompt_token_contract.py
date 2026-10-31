"""The prompt-count contract holds on the wire, including for legacy plugins.

``GenerationOutput.prompt_tokens`` owns ``usage.prompt_tokens``; a producer that
reports nothing falls back to the server counting the prepared prompt. These
tests drive a real app over HTTP so the contract is exercised through the route
a user reaches, not only through ``_usage`` directly.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from hipengine import SamplingParams
from hipengine.server.api import ServerConfig, create_app

_PROMPT = "one two three four"


class _PluginOutput(SimpleNamespace):
    """A duck-typed plugin output: not a ``GenerationOutput`` instance."""


class _LegacyPluginLLM:
    """An engine with only ``generate()``, like an out-of-tree plugin.

    ``prompt_tokens`` models the exact length the plugin prefilled (text plus
    image tokens, for a vision plugin). ``None`` means it cannot say.
    """

    def __init__(self, *, prompt_tokens: int | None) -> None:
        self.prompt_tokens = prompt_tokens
        self.max_sequence_length: int | None = None

    def count_tokens(self, text: str) -> int:
        return len(str(text).split())

    def generate(self, prompts, sampling_params: SamplingParams) -> list[object]:
        return [
            _PluginOutput(
                text="plugin reply",
                generated_token_ids=(901, 902),
                prompt_tokens=self.prompt_tokens,
            )
            for _ in prompts
        ]


def _client(prompt_tokens: int | None) -> TestClient:
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            startup_chat_smoke=False,
            startup_scratch_probe=False,
        ),
        llm=_LegacyPluginLLM(prompt_tokens=prompt_tokens),
    )
    return TestClient(app)


def test_plugin_reported_prompt_count_reaches_the_usage_payload() -> None:
    response = _client(11).post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": _PROMPT, "max_tokens": 2},
    )

    assert response.status_code == 200, response.text
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] == 2
    assert usage["total_tokens"] == 13


def test_plugin_without_a_count_keeps_the_prepared_prompt_fallback() -> None:
    response = _client(None).post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": _PROMPT, "max_tokens": 2},
    )

    assert response.status_code == 200, response.text
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 4
    assert usage["total_tokens"] == 6


def test_plugin_explicit_zero_is_reported_as_zero() -> None:
    response = _client(0).post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": _PROMPT, "max_tokens": 2},
    )

    assert response.status_code == 200, response.text
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 0
    assert usage["total_tokens"] == 2
