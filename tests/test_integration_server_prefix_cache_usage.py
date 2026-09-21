"""Prefix-cache reuse reaches the usage payload on the wire.

``_usage`` reads ``prefix_cache.reused_tokens`` from the same generation
telemetry that already carries MTP accounting, so these tests drive a real app
over HTTP and assert the OpenAI/vLLM-compatible field for a cache hit, a
measured miss, and a backend that reported no prefix telemetry at all.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from hipengine import SamplingParams
from hipengine.generation.registry import (
    GenerationOutput,
    GenerationStreamChunk,
    GenerationTelemetry,
)
from hipengine.server.api import ServerConfig, create_app

_PROMPT = "one two three four"


class _PrefixCacheFakeLLM:
    """Engine whose outputs report the prompt's prefix-cache reuse.

    ``reused_tokens=None`` models a backend that publishes no prefix-cache
    diagnostics at all, which is the case the usage field must stay silent about.
    """

    def __init__(self, *, reused_tokens: int | None) -> None:
        self.reused_tokens = reused_tokens
        self.max_sequence_length: int | None = None
        self.stream_calls: list[tuple[str, SamplingParams]] = []

    def prepare(self, *, max_sequence_length: int | None = None, sampling_params: SamplingParams) -> int:
        selected = 4096 if max_sequence_length is None else int(max_sequence_length)
        self.max_sequence_length = selected
        return selected

    def count_tokens(self, text: str) -> int:
        return len(str(text).split())

    def _telemetry(self, prompt_tokens: int) -> GenerationTelemetry:
        diagnostics: dict[str, Any] | None = None
        if self.reused_tokens is not None:
            diagnostics = {
                "prefix_cache": {
                    "mode": "radix",
                    "block_size_tokens": 256,
                    "eligible": True,
                    "lookup": True,
                    "hit": bool(self.reused_tokens),
                    "reused_tokens": self.reused_tokens,
                    "executed_prefill_tokens": max(0, prompt_tokens - self.reused_tokens),
                    "fallback_reason": None if self.reused_tokens else "miss",
                }
            }
        return GenerationTelemetry.from_decode_counts(
            prompt_tokens=prompt_tokens,
            generated_tokens=2,
            row_index=0,
            request_id="0",
            phase="answer",
            diagnostics=diagnostics,
        )

    def stream(self, prompt: str, sampling_params: SamplingParams):
        self.stream_calls.append((str(prompt), sampling_params))
        yield GenerationStreamChunk(
            text="cached reply",
            generated_token_ids=(901, 902),
            finish_details={"reason": "length"},
            telemetry=self._telemetry(self.count_tokens(prompt)),
        )

    def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
        return [
            GenerationOutput(
                text="cached reply",
                generated_token_ids=(901, 902),
                telemetry=self._telemetry(self.count_tokens(prompt)),
            )
            for prompt in prompts
        ]

    def generate(self, prompts, sampling_params: SamplingParams) -> list[str]:
        return [output.text for output in self.generate_detailed(prompts, sampling_params)]


def _client(reused_tokens: int | None) -> TestClient:
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=_PrefixCacheFakeLLM(reused_tokens=reused_tokens),
    )
    return TestClient(app)


def _sse_payloads(text: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def _streamed_usage(client: TestClient) -> dict:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": _PROMPT}],
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_usage": True, "include_hipengine": True},
        },
    )

    assert response.status_code == 200
    usage = next(payload["usage"] for payload in _sse_payloads(response.text) if payload.get("usage"))
    # The chat path counts the rendered prompt, so the four content words arrive
    # with the template's own tokens in front of them.
    assert usage["prompt_tokens"] == 6
    assert usage["completion_tokens"] == 2
    return usage


def test_streamed_usage_reports_cached_prompt_tokens() -> None:
    usage = _streamed_usage(_client(3))

    assert usage["prompt_tokens_details"] == {"cached_tokens": 3}


def test_streamed_usage_reports_a_measured_miss_as_zero() -> None:
    usage = _streamed_usage(_client(0))

    assert usage["prompt_tokens_details"] == {"cached_tokens": 0}


def test_streamed_usage_omits_cached_tokens_without_prefix_telemetry() -> None:
    usage = _streamed_usage(_client(None))

    assert "prompt_tokens_details" not in usage


def test_blocking_usage_reports_cached_prompt_tokens() -> None:
    response = _client(3).post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": _PROMPT}],
            "max_tokens": 2,
        },
    )

    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 6
    assert usage["prompt_tokens_details"] == {"cached_tokens": 3}
