from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hipengine import SamplingParams
from hipengine.generation import FinishDetails, GenerationOutput, GenerationStreamChunk
from hipengine.server import ServerConfig, create_app
from hipengine.server.__main__ import build_parser


class _SpeculativeFakeLLM:
    supports_speculative = True

    def __init__(self) -> None:
        self.ar_calls: list[tuple[tuple[str, ...], SamplingParams]] = []
        self.speculative_calls: list[tuple[tuple[str, ...], SamplingParams]] = []
        self.stream_calls: list[tuple[str, SamplingParams]] = []
        self.speculative_stream_calls: list[tuple[str, SamplingParams]] = []
        self.prepare_calls = 0
        self.max_sequence_length: int | None = None

    @property
    def speculative_capabilities(self) -> dict[str, Any]:
        return {
            "provider": "dflash",
            "policy": "explicit_only",
            "default_enabled": False,
            "streaming_compatible": True,
            "candidate_budget": 4,
            "exactness_mode": "target_corrected_greedy",
            "processed_target_verification": False,
            "target": {"model": "target", "sha256": "a" * 64},
            "drafter": {
                "model": "drafter",
                "revision": "revision",
                "sha256": "b" * 64,
            },
            "fallback_reason": "d4_full_suite_speedup_0p9469x_below_1p10",
            "performance_claim": False,
            "economics_evidence": "benchmarks/results/laguna-d4.json",
        }

    def prepare(
        self,
        *,
        max_sequence_length: int | None = None,
        sampling_params: SamplingParams,
    ) -> int:
        del sampling_params
        self.prepare_calls += 1
        self.max_sequence_length = 4096 if max_sequence_length is None else int(max_sequence_length)
        return self.max_sequence_length

    def generate_detailed(
        self,
        prompts,
        sampling_params: SamplingParams,
    ) -> list[GenerationOutput]:
        rows = tuple(str(prompt) for prompt in prompts)
        self.ar_calls.append((rows, sampling_params))
        return [
            GenerationOutput(
                text=f"ar:{prompt}",
                finish_details=FinishDetails(reason="length", length_limit=sampling_params.max_tokens),
            )
            for prompt in rows
        ]

    def generate_speculative_detailed(
        self,
        prompts,
        sampling_params: SamplingParams,
    ) -> list[GenerationOutput]:
        rows = tuple(str(prompt) for prompt in prompts)
        self.speculative_calls.append((rows, sampling_params))
        return [
            GenerationOutput(
                text=f"spec:{prompt}",
                finish_details=FinishDetails(reason="length", length_limit=sampling_params.max_tokens),
                generated_token_ids=(10, 11),
            )
            for prompt in rows
        ]

    def stream_detailed(self, prompt: str, sampling_params: SamplingParams):
        self.stream_calls.append((str(prompt), sampling_params))
        yield GenerationStreamChunk(
            text=f"ar:{prompt}",
            finish_details=FinishDetails(reason="length", length_limit=sampling_params.max_tokens),
        )

    def stream_speculative_detailed(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ):
        self.speculative_stream_calls.append((str(prompt), sampling_params))
        yield GenerationStreamChunk(text="spec:", generated_token_ids=(10,))
        yield GenerationStreamChunk(
            text=str(prompt),
            finish_details=FinishDetails(reason="length", length_limit=sampling_params.max_tokens),
            generated_token_ids=(10, 11),
        )

    def render_chat_prompt(self, messages, **kwargs) -> str:
        del kwargs
        return "chat:" + "|".join(
            str(
                message.get("content", "")
                if isinstance(message, dict)
                else getattr(message, "content", "")
            )
            for message in messages
        )

    def count_tokens(self, text: str) -> int:
        return max(1, len(str(text).split()))


def _config(**overrides: Any) -> ServerConfig:
    values: dict[str, Any] = {
        "model": "fake-path",
        "served_model_name": "fake-model",
        "eager_load": False,
        "startup_chat_smoke": False,
        "startup_scratch_probe": False,
        "speculative_provider": "dflash",
        "draft_model": "/models/drafter",
        "speculative_candidate_budget": 4,
    }
    values.update(overrides)
    return ServerConfig(**values)


def _sse_payloads(text: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: {")
    ]


def test_server_config_requires_complete_explicit_provider_owner() -> None:
    with pytest.raises(ValueError, match="draft_model requires speculative_provider"):
        ServerConfig(model="target", draft_model="drafter")
    with pytest.raises(ValueError, match="speculative_provider requires draft_model"):
        ServerConfig(model="target", speculative_provider="dflash")
    with pytest.raises(ValueError, match="speculative_candidate_budget"):
        ServerConfig(
            model="target",
            speculative_provider="dflash",
            draft_model="drafter",
            speculative_candidate_budget=0,
        )


def test_capabilities_report_truthful_generic_speculative_provider() -> None:
    fake = _SpeculativeFakeLLM()
    client = TestClient(create_app(_config(), llm=fake))

    capability = client.get("/v1/hipengine/capabilities").json()["sampling"]["speculative"]

    assert capability["serving_route"] is True
    assert capability["request_field"] == "speculative"
    assert capability["configured_provider"] == "dflash"
    assert capability["policy"] == "explicit_only"
    assert capability["default_enabled"] is False
    assert capability["streaming_compatible"] is True
    assert capability["candidate_budget"] == 4
    assert capability["exactness_mode"] == "target_corrected_greedy"
    assert capability["processed_target_verification"] is False
    assert capability["target"]["sha256"] == "a" * 64
    assert capability["drafter"]["sha256"] == "b" * 64
    assert capability["fallback_reason"] == "d4_full_suite_speedup_0p9469x_below_1p10"
    assert capability["performance_claim"] is False
    models = client.get("/v1/models").json()
    assert models["data"][0]["hipengine"]["capabilities"]["speculative"] is True


def test_completion_default_stays_ar_and_explicit_request_routes_provider() -> None:
    fake = _SpeculativeFakeLLM()
    client = TestClient(create_app(_config(), llm=fake))

    default = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "one", "max_tokens": 2},
    )
    explicit = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "two",
            "max_tokens": 2,
            "speculative": {
                "enabled": True,
                "provider": "dflash",
                "candidate_budget": 4,
            },
        },
    )

    assert default.status_code == 200
    assert default.json()["choices"][0]["text"] == "ar:one"
    assert explicit.status_code == 200
    assert explicit.json()["choices"][0]["text"] == "spec:two"
    assert len(fake.ar_calls) == 1
    assert len(fake.speculative_calls) == 1
    assert explicit.json()["hipengine"]["generation_shape"]["route"] == "speculative"
    assert explicit.json()["hipengine"]["generation_shape"]["route_cap"] == {
        "scope": "queue_requests",
        "value": 1,
        "applied": True,
    }


def test_chat_streaming_explicit_request_uses_provider_stream() -> None:
    fake = _SpeculativeFakeLLM()
    client = TestClient(create_app(_config(), llm=fake))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "stream": True,
            "enable_thinking": False,
            "speculative": True,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content = "".join(
        str(payload["choices"][0]["delta"].get("content", ""))
        for payload in payloads
        if payload.get("choices")
    )
    assert content == "spec:chat:hello"
    assert len(fake.speculative_stream_calls) == 1
    assert fake.stream_calls == []
    assert "data: [DONE]" in response.text


@pytest.mark.parametrize(
    ("request_update", "expected_blocker"),
    [
        ({"temperature": 0.2}, "temperature"),
        ({"top_p": 0.9}, "top_p"),
        ({"top_k": 2}, "top_k"),
        ({"min_p": 0.1}, "min_p"),
        ({"logit_bias": {"10": 1.0}}, "logit_bias"),
        ({"ignore_eos": True}, "ignore_eos"),
        ({"eos_token_id": 2}, "eos_token_id"),
        ({"logprobs": 1}, "logprobs"),
        ({"n": 2}, "c"),
    ],
)
def test_explicit_provider_rejects_unsupported_request_before_prepare(
    request_update: dict[str, Any],
    expected_blocker: str,
) -> None:
    fake = _SpeculativeFakeLLM()
    client = TestClient(create_app(_config(), llm=fake))
    request = {
        "model": "fake-model",
        "prompt": "one",
        "max_tokens": 2,
        "speculative": True,
    }
    request.update(request_update)

    response = client.post("/v1/completions", json=request)

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unsupported_parameter"
    assert error["param"] == "speculative"
    assert expected_blocker in error["hipengine"]["speculative"]["blockers"]
    assert fake.prepare_calls == 0
    assert fake.ar_calls == []
    assert fake.speculative_calls == []


def test_chat_provider_rejects_sampling_before_target_prepare() -> None:
    fake = _SpeculativeFakeLLM()
    client = TestClient(create_app(_config(), llm=fake))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "temperature": 0.2,
            "enable_thinking": False,
            "speculative": True,
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "speculative"
    assert "temperature" in error["hipengine"]["speculative"]["blockers"]
    assert fake.prepare_calls == 0
    assert fake.ar_calls == []
    assert fake.speculative_calls == []


def test_explicit_provider_rejects_mismatch_budget_and_mtp_conflict() -> None:
    fake = _SpeculativeFakeLLM()
    client = TestClient(create_app(_config(), llm=fake))
    base = {"model": "fake-model", "prompt": "one", "max_tokens": 2}

    wrong_provider = client.post(
        "/v1/completions",
        json={**base, "speculative": {"provider": "other"}},
    )
    wrong_budget = client.post(
        "/v1/completions",
        json={**base, "speculative": {"candidate_budget": 2}},
    )
    conflict = client.post(
        "/v1/completions",
        json={**base, "speculative": True, "speculative_mtp": True},
    )

    assert wrong_provider.status_code == 400
    assert wrong_provider.json()["error"]["param"] == "speculative"
    assert wrong_budget.status_code == 400
    assert wrong_budget.json()["error"]["param"] == "speculative"
    assert conflict.status_code == 400
    assert conflict.json()["error"]["param"] == "speculative"
    assert fake.prepare_calls == 0


def test_lazy_server_forwards_speculative_owner_to_llm(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    fake = _SpeculativeFakeLLM()

    def build_llm(model: str, **kwargs: Any):
        captured["model"] = model
        captured.update(kwargs)
        return fake

    monkeypatch.setattr("hipengine.server.api.LLM", build_llm)
    client = TestClient(
        create_app(_config(execution_profile="strict"), llm=None)
    )

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "one", "max_tokens": 1},
    )

    assert response.status_code == 200
    assert captured["execution_profile"] == "strict"
    assert captured["speculative_provider"] == "dflash"
    assert captured["draft_model"] == "/models/drafter"
    assert captured["speculative_candidate_budget"] == 4


def test_server_cli_accepts_speculative_provider_owner(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_SPECULATIVE_PROVIDER", "dflash")
    monkeypatch.setenv("HIPENGINE_DRAFT_MODEL", "/models/env-drafter")
    monkeypatch.setenv("HIPENGINE_SPECULATIVE_CANDIDATE_BUDGET", "4")

    env_args = build_parser().parse_args(["--model", "target"])
    cli_args = build_parser().parse_args(
        [
            "--model",
            "target",
            "--speculative-provider",
            "other",
            "--draft-model",
            "/models/cli-drafter",
            "--speculative-candidate-budget",
            "7",
        ]
    )

    assert env_args.speculative_provider == "dflash"
    assert env_args.draft_model == "/models/env-drafter"
    assert env_args.speculative_candidate_budget == 4
    assert cli_args.speculative_provider == "other"
    assert cli_args.draft_model == "/models/cli-drafter"
    assert cli_args.speculative_candidate_budget == 7
