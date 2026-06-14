from __future__ import annotations

import asyncio
import json
import logging
import time
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from hipengine import SamplingParams
from hipengine.generation import (
    GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS,
    FinishDetails,
    GenerationCancellationToken,
    GenerationCancelled,
    GenerationDeadlineExceeded,
    GenerationOutput,
    GenerationTelemetry,
    TokenLogprob,
)
from hipengine.server import ServerConfig, create_app, render_chat_prompt
from hipengine.server.__main__ import build_parser
from hipengine.server.api import (
    ChatCompletionRequest,
    CompletionRequest,
    OpenAIHTTPError,
    _await_with_request_control,
    _coerce_generation_output,
    _GenerationBatcher,
    _request_control,
    _startup_memory_summary,
)


class FakeLLM:
    def __init__(
        self,
        outputs: list[str] | None = None,
        stream_chunks: list[str] | None = None,
        token_map: dict[str, list[int]] | None = None,
        detailed_outputs: list[GenerationOutput] | None = None,
    ) -> None:
        self.outputs = outputs
        self.detailed_outputs = detailed_outputs
        self.stream_chunks = stream_chunks
        self.token_map = token_map
        self.calls: list[tuple[tuple[str, ...], SamplingParams]] = []
        self.stream_calls: list[tuple[str, SamplingParams]] = []
        self.prepares: list[tuple[int | None, SamplingParams]] = []
        self.scratch_prepares: list[dict[str, Any]] = []
        self.tokenize_calls: list[str] = []
        self.max_sequence_length: int | None = None
        self.kv_capacity_estimate = None
        self.kv_capacity_int8_estimate = None

    def prepare(self, *, max_sequence_length: int | None = None, sampling_params: SamplingParams) -> int:
        self.prepares.append((None if max_sequence_length is None else int(max_sequence_length), sampling_params))
        requested = 262144 if max_sequence_length is None else int(max_sequence_length)
        selected = min(262144, 131072) if max_sequence_length is None else requested
        self.max_sequence_length = selected
        self.kv_capacity_estimate = _fake_kv_estimate(
            max_sequence_length=selected,
            storage="bf16" if sampling_params.kv_storage == "auto" else sampling_params.kv_storage,
        )
        self.kv_capacity_int8_estimate = _fake_kv_estimate(
            max_sequence_length=selected,
            storage="int8_per_token_head",
        )
        return selected

    def prepare_request_scratch(
        self,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int = 0,
        sampling_params: SamplingParams | None = None,
        max_batch_size: int = 1,
        release_after_probe: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "max_prompt_tokens": int(max_prompt_tokens),
            "max_new_tokens": int(max_new_tokens),
            "sampling_params": sampling_params,
            "max_batch_size": int(max_batch_size),
            "release_after_probe": bool(release_after_probe),
        }
        self.scratch_prepares.append(payload)
        return {key: value for key, value in payload.items() if key != "sampling_params"}

    def generate(self, prompts, sampling_params: SamplingParams) -> list[str]:
        return [output.text for output in self.generate_detailed(prompts, sampling_params)]

    def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
        prompts = tuple(prompts)
        self.calls.append((prompts, sampling_params))
        if self.detailed_outputs is not None:
            return self.detailed_outputs[: len(prompts)]
        if self.outputs is not None:
            return [GenerationOutput(text=output) for output in self.outputs[: len(prompts)]]
        return [GenerationOutput(text=f"generated:{prompt}") for prompt in prompts]

    def stream(self, prompt: str, sampling_params: SamplingParams):
        self.stream_calls.append((str(prompt), sampling_params))
        self.calls.append(((prompt,), sampling_params))
        if self.stream_chunks is not None:
            yield from self.stream_chunks
        elif self.outputs is not None:
            yield self.outputs[0]
        else:
            yield f"generated:{prompt}"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def tokenize(self, text: str) -> tuple[int, ...]:
        self.tokenize_calls.append(str(text))
        if self.token_map is None:
            raise NotImplementedError("fake tokenization is not configured")
        return tuple(self.token_map[str(text)])

    def detokenize(self, token_ids, *, skip_special: bool = False) -> str:
        return " ".join(f"T{int(token)}" for token in token_ids)


class DetailedGenerateFakeLLM(FakeLLM):
    def generate(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
        return self.generate_detailed(prompts, sampling_params)


class DelayedFakeLLM(FakeLLM):
    def __init__(
        self,
        *args,
        generate_delay_s: float = 0.0,
        stream_delay_s: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.generate_delay_s = float(generate_delay_s)
        self.stream_delay_s = float(stream_delay_s)
        self.completed_generations = 0

    def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
        if self.generate_delay_s > 0.0:
            time.sleep(self.generate_delay_s)
        try:
            return super().generate_detailed(prompts, sampling_params)
        finally:
            self.completed_generations += 1

    def stream(self, prompt: str, sampling_params: SamplingParams):
        for chunk in super().stream(prompt, sampling_params):
            if self.stream_delay_s > 0.0:
                time.sleep(self.stream_delay_s)
            yield chunk


class BackendDeadlineFakeLLM(FakeLLM):
    def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
        prompts = tuple(prompts)
        self.calls.append((prompts, sampling_params))
        assert sampling_params.deadline_at is not None
        raise GenerationDeadlineExceeded(deadline_at=sampling_params.deadline_at)

    def stream(self, prompt: str, sampling_params: SamplingParams):
        self.stream_calls.append((str(prompt), sampling_params))
        self.calls.append(((prompt,), sampling_params))
        assert sampling_params.deadline_at is not None
        raise GenerationDeadlineExceeded(deadline_at=sampling_params.deadline_at)
        yield  # pragma: no cover - keeps this method a generator


class BackendCancelledFakeLLM(FakeLLM):
    def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
        prompts = tuple(prompts)
        self.calls.append((prompts, sampling_params))
        assert sampling_params.cancellation_token is not None
        sampling_params.cancellation_token.cancel()
        raise GenerationCancelled(sampling_params.cancellation_token.finish_details)

    def stream(self, prompt: str, sampling_params: SamplingParams):
        self.stream_calls.append((str(prompt), sampling_params))
        self.calls.append(((prompt,), sampling_params))
        assert sampling_params.cancellation_token is not None
        sampling_params.cancellation_token.cancel()
        raise GenerationCancelled(sampling_params.cancellation_token.finish_details)
        yield  # pragma: no cover - keeps this method a generator


def _fake_kv_estimate(*, max_sequence_length: int, storage: str):
    bytes_per_token = 8192
    rounded_tokens = ((int(max_sequence_length) + 255) // 256) * 256
    return SimpleNamespace(
        requested_context_tokens=int(max_sequence_length),
        model_max_context_tokens=262144,
        allocatable_context_tokens=131072,
        requested_kv_bytes=rounded_tokens * bytes_per_token,
        bytes_per_token=bytes_per_token,
        usable_bytes=4 * 1024**3,
        reserve_bytes=512 * 1024**2,
        kv_storage_dtype=storage,
        kv_scale_dtype="fp16" if storage == "int8_per_token_head" else None,
        fits_model_max=False,
    )


def test_coerce_generation_output_preserves_telemetry() -> None:
    raw = SimpleNamespace(
        text="answer",
        telemetry=GenerationTelemetry.from_decode_counts(
            prompt_tokens=3,
            generated_tokens=2,
            sampler_mode="host_logits_sample",
        ),
    )

    output = _coerce_generation_output(raw)

    assert output.text == "answer"
    assert output.telemetry is not None
    assert output.telemetry.to_json_dict()["decode_state"] == {
        "row_index": 0,
        "step_index": 2,
        "prompt_tokens": 3,
        "generated_tokens": 2,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_mode": "host_logits_sample",
    }


def test_models_endpoint_reports_served_model_name_and_auth() -> None:
    fake = FakeLLM()
    app = create_app(
        ServerConfig(model="/models/fake", served_model_name="fake-model", api_key="secret"),
        llm=fake,
    )
    client = TestClient(app)

    unauthorized = client.get("/v1/models")
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["type"] == "authentication_error"

    response = client.get("/v1/models", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    model = body["data"][0]
    assert model["id"] == "fake-model"
    assert model["object"] == "model"
    assert model["created"] == app.state.hipengine_config.created
    assert model["owned_by"] == "hipengine"
    assert model["hipengine"] == {
        "path": "/models/fake",
        "backend": "auto",
        "quant": "w4_paro",
        "loaded": True,
        "resident_context": True,
        "context": {
            "configured_max_context_tokens": None,
            "effective_max_context_tokens": None,
            "chat_default_max_tokens": 4096,
        },
        "kv_capacity": {
            "storage": "auto",
            "scale_dtype": "fp16",
            "scale_granularity": "per_token_head",
            "estimate": None,
        },
        "capabilities_url": "/v1/hipengine/capabilities",
        "routing": {"loaded_model_count": 1, "multiple_models": False},
    }


def test_models_endpoint_reports_lazy_model_not_loaded() -> None:
    app = create_app(
        ServerConfig(model="/models/fake", served_model_name="fake-model", eager_load=False),
        llm=None,
    )
    client = TestClient(app)

    response = client.get("/v1/models")

    assert response.status_code == 200
    model = response.json()["data"][0]
    assert model["id"] == "fake-model"
    assert model["hipengine"]["loaded"] is False
    assert model["hipengine"]["routing"] == {"loaded_model_count": 0, "multiple_models": False}
    assert model["hipengine"]["kv_capacity"]["estimate"] is None


def test_capabilities_endpoint_reports_manifest_and_auth(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_QWEN35_NATIVE_SAMPLER", raising=False)
    fake = FakeLLM()
    app = create_app(
        ServerConfig(
            model="/models/fake",
            served_model_name="fake-model",
            api_key="secret",
            eager_load=False,
            max_context_tokens=2048,
            request_timeout_ms=250.0,
        ),
        llm=fake,
    )
    client = TestClient(app)

    unauthorized = client.get("/v1/hipengine/capabilities")
    assert unauthorized.status_code == 401

    response = client.get("/v1/hipengine/capabilities", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "hipengine.capabilities"
    assert body["model"] == {
        "id": "fake-model",
        "path": "/models/fake",
        "backend": "auto",
        "quant": "w4_paro",
    }
    assert body["context"] == {
        "configured_max_context_tokens": 2048,
        "effective_max_context_tokens": 2048,
        "chat_default_max_tokens": 4096,
        "chat_default_mode": "bounded",
    }
    assert body["tokenizer"]["tokenize"] is True
    assert body["tokenizer"]["detokenize"] is True
    assert body["tokenizer"]["count_tokens"] is True
    assert body["features"]["stream_options"] == {"include_usage": True, "include_hipengine": True}
    assert body["features"]["structured_outputs"] == {
        "response_format": True,
        "json_object": True,
        "json_schema": True,
        "strict_decoding": False,
        "strict_result_validation": True,
    }
    assert body["features"]["token_diagnostics"] == {
        "tokenize": True,
        "detokenize": True,
        "count_tokens": True,
        "fit_context": True,
    }
    assert body["features"]["tools"] == {
        "enabled": True,
        "strict_decoding": False,
        "strict_result_validation": True,
        "schema_validation": "function_strict",
        "schema_subset": [
            "type",
            "enum",
            "const",
            "object.properties",
            "object.required",
            "object.additionalProperties=false",
            "array.items",
            "array.minItems",
            "array.maxItems",
            "string.minLength",
            "string.maxLength",
            "number.minimum",
            "number.maximum",
            "number.exclusiveMinimum",
            "number.exclusiveMaximum",
        ],
        "format": "qwen_tool_call_json",
        "parallel_tool_calls": True,
        "no_tool_start_suppression": True,
        "required_tool_start_forcing": True,
        "required_tool_start_forcing_scope": "initial_or_after_tokenized_thinking_close",
    }
    assert body["features"]["reasoning_controls"] == {
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
        "hard_close_validation": True,
        "hard_close_token_forcing": True,
        "soft_close_bias": True,
        "eos_suppression": True,
        "hard_close_marker": "</think>",
        "diagnostic_close_token_lowering": True,
        "diagnostic_initial_state": True,
    }
    assert body["features"]["logprobs"]["streaming"] == "buffered"
    assert body["features"]["request_timeouts"] == {
        "timeout_ms": True,
        "default_timeout_ms": 250.0,
        "client_disconnect": True,
        "cooperative_backend_deadline": True,
        "cooperative_backend_cancel": True,
        "preemptive_decode_cancel": False,
    }
    assert body["errors"]["schema"] == "hipengine.error_taxonomy.v1"
    errors_by_code = {item["code"]: item for item in body["errors"]["codes"]}
    for code in (
        "unsupported_parameter",
        "invalid_tool_call",
        "schema_violation",
        "context_overflow",
        "deadline_exceeded",
        "cancelled",
        "engine_busy",
        "model_unavailable",
        "routing_failed",
    ):
        assert code in errors_by_code
    assert errors_by_code["engine_busy"]["emitted"] is True
    assert errors_by_code["engine_busy"]["status_code"] == 429
    assert errors_by_code["invalid_tool_call"]["emitted"] is True
    assert "finish_details.reason" in errors_by_code["invalid_tool_call"]["description"]
    assert "response_format result" in errors_by_code["schema_violation"]["description"]
    assert {
        "legacy_code": "model_not_found",
        "code": "model_unavailable",
    } in body["errors"]["aliases"]
    assert body["sampling"]["execution_modes"] == [
        "greedy_fast",
        "processed_argmax",
        "host_logits_sample",
        "gpu_sample",
    ]
    assert "suppress_token_ids" in body["sampling"]["parameters"]
    assert "min_tokens" in body["sampling"]["parameters"]
    assert "eos_token_id" in body["sampling"]["parameters"]
    assert body["sampling"]["native_gpu"] == {
        "enabled": False,
        "env": "HIPENGINE_QWEN35_NATIVE_SAMPLER",
        "scope": "paro_c1_only",
        "default_path": False,
        "top_k_max": 64,
        "top_p_min_p": "exact_full_vocab_top_k_0",
        "selected_logprobs": True,
        "top_logprobs": False,
        "processors": [
            "logit_bias",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
        ],
        "unsupported": [
            "c_gt_1",
            "gguf",
            "top_logprobs",
            "suppress_token_ids",
            "min_tokens",
            "thinking_budget",
            "combined_top_k_with_top_p_or_min_p",
        ],
    }
    assert body["sampling"]["speculative_mtp"] == {
        "serving_route": False,
        "sampling_compatible": False,
        "compatibility_guard": "supports_speculative_mtp_sampling",
        "allowed_execution_modes": ["greedy_fast"],
        "incompatible_fields": [
            "temperature",
            "logit_bias",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "suppress_token_ids",
            "min_tokens",
            "stop_token_ids",
            "stop_token_sequences",
            "forced_tokens_pending",
            "post_thinking_forced_tokens_pending",
            "thinking_budget",
            "logprobs",
            "top_logprobs",
        ],
        "incompatible_conditions": {
            "temperature": "temperature > 0",
            "logit_bias": "non-empty logit_bias",
            "repetition_penalty": "repetition_penalty != 1.0",
            "presence_penalty": "presence_penalty != 0.0",
            "frequency_penalty": "frequency_penalty != 0.0",
            "suppress_token_ids": "one or more suppressed token ids",
            "min_tokens": "min_tokens > 0",
            "stop_token_ids": "one or more token stop ids",
            "stop_token_sequences": "one or more multi-token stop sequences",
            "forced_tokens_pending": "one or more forced tokens pending",
            "post_thinking_forced_tokens_pending": "one or more post-thinking forced tokens pending",
            "thinking_budget": "thinking budget soft-close, EOS suppression, or hard-close control",
            "logprobs": "logprobs requested",
            "top_logprobs": "top_logprobs > 0",
        },
        "processed_target_verification": False,
    }
    assert body["sessions"] == {
        "resident_context": True,
        "commit_policy": {
            "supported": True,
            "stateful": False,
            "default": "append_none",
            "modes": ["append_none"],
            "unsupported_stateful_modes": [
                "append_all",
                "append_visible_only",
                "append_prompt_only",
            ],
        },
        "continuations": False,
    }
    assert body["routing"] == {"loaded_model_count": 1, "multiple_models": False}
    assert "timeout_ms" not in body["unsupported_fields"]
    assert "parallel_tool_calls" not in body["unsupported_fields"]
    assert "response_format" not in body["unsupported_fields"]


def test_capabilities_endpoint_reports_auto_chat_default_and_cache_config() -> None:
    fake = FakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            chat_default_max_tokens=None,
            kv_storage="int8_per_token_head",
            kv_scale_dtype="fp32",
            prefix_cache="radix",
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.get("/v1/hipengine/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["context"]["chat_default_max_tokens"] is None
    assert body["context"]["chat_default_mode"] == "auto"
    assert body["cache"]["prefix_cache"] == "radix"
    assert body["cache"]["kv_storage"] == "int8_per_token_head"
    assert body["cache"]["kv_scale_dtype"] == "fp32"


def test_token_diagnostics_endpoints_handle_text_and_chat() -> None:
    fake = FakeLLM(token_map={"hello": [10, 11], "closing now</think>\n": [42, 43, 44]})
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=512,
            chat_default_max_tokens=7,
        ),
        llm=fake,
    )
    client = TestClient(app)

    tokenize = client.post("/v1/hipengine/tokenize", json={"text": "hello"})
    assert tokenize.status_code == 200
    assert tokenize.json() == {
        "object": "hipengine.tokens",
        "text": "hello",
        "token_ids": [10, 11],
        "token_count": 2,
    }

    detokenize = client.post("/v1/hipengine/detokenize", json={"token_ids": [10, 11]})
    assert detokenize.status_code == 200
    assert detokenize.json() == {
        "object": "hipengine.text",
        "text": "T10 T11",
        "token_ids": [10, 11],
    }

    count_text = client.post("/v1/hipengine/count_tokens", json={"text": "one two three"})
    assert count_text.status_code == 200
    assert count_text.json()["token_count"] == 3
    assert count_text.json()["input_type"] == "text"

    chat_payload = {
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"query":"hello"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "tool result"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        "reasoning_effort": "low",
        "hard_close_sequence": "closing now</think>\n",
        "soft_close_window": 4,
        "max_tokens": 32,
    }
    count_chat = client.post("/v1/hipengine/count_tokens", json=chat_payload)
    assert count_chat.status_code == 200
    chat_body = count_chat.json()
    assert chat_body["input_type"] == "chat"
    assert "<|im_start|>user\nhello<|im_end|>" in chat_body["text"]
    assert "<tools>" in chat_body["text"]
    assert '<tool_call>{"name":"lookup","arguments":{"query":"hello"}}</tool_call>' in chat_body["text"]
    assert "<tool_response>\ntool result\n</tool_response>" in chat_body["text"]
    assert "use 'closing now</think>\\n' as the close sequence" in chat_body["text"]
    assert chat_body["token_count"] == fake.count_tokens(chat_body["text"])
    assert chat_body["thinking_budget"]["close_text"] == "closing now</think>\n"
    assert chat_body["thinking_budget"]["close_token_ids"] == [42, 43, 44]
    assert chat_body["thinking_budget"]["initial_state"] == {
        "phase": "think",
        "reasoning_tokens": 0,
        "answer_tokens": 0,
        "hard_token_cap": 16,
        "remaining_think_tokens": 16,
        "soft_close_window": 4,
        "close_sequence": [42, 43, 44],
    }

    fit = client.post("/v1/hipengine/fit_context", json=chat_payload)
    assert fit.status_code == 200
    fit_body = fit.json()
    expected_max_tokens = 32
    assert fit_body["input_type"] == "chat"
    assert fit_body["prompt_tokens"] == chat_body["token_count"]
    assert fit_body["max_context_tokens"] == 512
    assert fit_body["requested_max_tokens"] == 32
    assert fit_body["effective_max_tokens"] == expected_max_tokens
    assert fit_body["required_context_tokens"] == chat_body["token_count"] + expected_max_tokens + 1
    assert fit_body["fits"] is True
    assert fit_body["chat_default_max_tokens"] == 7
    assert fit_body["clear_policy"] == "reject"
    assert fit_body["would_drop"] == []
    assert fit_body["thinking_budget"]["lowering_supported"] is True
    assert fit_body["thinking_budget"]["close_token_ids"] == [42, 43, 44]


def test_token_diagnostics_reject_ambiguous_inputs() -> None:
    fake = FakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/hipengine/count_tokens",
        json={"text": "hello", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_server_eager_loads_model_on_startup(caplog) -> None:
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    fake = FakeLLM(outputs=["warm"])
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        eager_load_prompt="one two three four",
        eager_load_max_tokens=2,
    )
    app = create_app(config, llm=fake)

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert "Config: model=fake-path" in caplog.text
    assert "max_context_tokens=131072" in caplog.text
    assert "chat_default_max_tokens=4096" in caplog.text
    assert "kv_storage=auto" in caplog.text
    assert "KVCache: storage=bf16" in caplog.text
    assert "model_max_context_tokens=262144" in caplog.text
    assert "WARMUP: prompt_tokens<=131072 max_tokens=2" in caplog.text
    assert "STARTUP_SCRATCH_PROBE: max_prompt_tokens=131071" in caplog.text
    assert "WARMUP_CHAT: prompt_tokens=" in caplog.text
    assert "LOAD_TIMING: phase=startup resident_prepare_s=" in caplog.text
    assert "LOAD_TIMING: model=fake-model engine_create_s=" in caplog.text
    assert "warmup_s=" in caplog.text
    assert "scratch_probe_s=" in caplog.text
    assert "chat_smoke_s=" in caplog.text
    assert "hipEngine is ready." in caplog.text
    assert fake.prepares == [
        (
            None,
            SamplingParams(max_tokens=2, temperature=0.0, top_p=1.0, ignore_eos=True),
        )
    ]
    assert fake.scratch_prepares == [
        {
            "max_prompt_tokens": 131071,
            "max_new_tokens": 0,
            "sampling_params": SamplingParams(max_tokens=2, temperature=0.0, top_p=1.0, ignore_eos=True),
            "max_batch_size": 1,
            "release_after_probe": True,
        }
    ]
    assert fake.calls == [
        (
            ("one two three four",),
            SamplingParams(max_tokens=2, temperature=0.0, top_p=1.0, ignore_eos=True),
        ),
        (
            ("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",),
            SamplingParams(max_tokens=2, temperature=0.0, top_p=1.0),
        ),
    ]


def test_startup_memory_summary_counts_live_scratch_probe_peak() -> None:
    summary = _startup_memory_summary(
        {
            "startup_begin": {"free_bytes": 900, "used_bytes": 100, "total_bytes": 1000},
            "after_raw_warmup": {"free_bytes": 500, "used_bytes": 500, "total_bytes": 1000},
            "guard": {"free_bytes": 600, "used_bytes": 400, "total_bytes": 1000},
        },
        {
            "scratch_probe": {
                "status": "passed",
                "result": {
                    "live_memory": {
                        "stage": "linear_prefill_scratch_live",
                        "free_bytes": 250,
                        "used_bytes": 750,
                        "total_bytes": 1000,
                    },
                },
            },
        },
    )

    assert summary == {
        "sample_count": 4,
        "final_stage": "guard",
        "final_free_bytes": 600,
        "final_used_bytes": 400,
        "peak_stage": "scratch_probe:linear_prefill_scratch_live",
        "peak_used_bytes": 750,
        "min_free_stage": "scratch_probe:linear_prefill_scratch_live",
        "min_free_bytes": 250,
        "total_bytes": 1000,
    }


def test_health_and_ready_report_eager_startup_diagnostics() -> None:
    fake = FakeLLM(outputs=["private warmup output"])
    config = ServerConfig(
        model="fake-path",
        served_model_name="fake-model",
        eager_load_prompt="private startup prompt",
        eager_load_max_tokens=2,
    )
    app = create_app(config, llm=fake)

    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")

    assert health.status_code == 200
    assert health.json() == {
        "object": "hipengine.health",
        "status": "ok",
        "model": "fake-model",
    }
    assert ready.status_code == 200
    body = ready.json()
    assert body["object"] == "hipengine.readiness"
    assert body["ready"] is True
    assert body["status"] == "ready"
    assert body["model"] == {
        "id": "fake-model",
        "backend": "auto",
        "quant": "w4_paro",
        "loaded": True,
        "loaded_model_count": 1,
    }
    assert body["startup"]["eager_load"] is True
    assert body["startup"]["warmup_complete"] is True
    assert body["startup"]["last_timings_s"]["warmup_s"] >= 0.0
    assert body["startup"]["last_timings_s"]["scratch_probe_s"] >= 0.0
    assert body["startup"]["last_timings_s"]["chat_smoke_s"] >= 0.0
    assert body["startup"]["checks"]["scratch_probe"]["status"] == "passed"
    assert body["startup"]["checks"]["scratch_probe"]["max_prompt_tokens"] == 131071
    assert body["startup"]["checks"]["chat_smoke"]["status"] == "passed"
    for snapshot in body["startup"]["memory"].values():
        assert set(snapshot) >= {"free_bytes", "total_bytes", "used_bytes"}
    assert body["context"]["effective_max_context_tokens"] == 131072
    assert body["kv_capacity"]["estimate"]["allocatable_context_tokens"] == 131072
    assert body["kv_capacity"]["storage"] == "auto"
    assert body["graph_cache"]["entries"] == 0.0
    assert body["queue"]["depth"] == 0
    assert body["queue"]["max_depth"] is None
    assert body["sessions"] == {"resident_context": True, "active": 0}
    serialized = json.dumps(body)
    assert "private startup prompt" not in serialized
    assert "private warmup output" not in serialized


def test_ready_reports_lazy_server_ready_without_loaded_model() -> None:
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False)
    )

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["status"] == "ready"
    assert body["startup"]["eager_load"] is False
    assert body["startup"]["warmup_complete"] is True
    assert body["startup"]["last_timings_s"]["startup_total_s"] >= 0.0
    assert body["model"]["loaded"] is False
    assert body["model"]["loaded_model_count"] == 0


def test_chat_default_max_tokens_is_dynamic_when_omitted() -> None:
    request = ChatCompletionRequest(
        model="fake-model",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert request.max_tokens is None


def test_generation_batcher_coalesces_compatible_submissions() -> None:
    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        first, second = await asyncio.gather(
            batcher.submit(("one",), sampling),
            batcher.submit(("two", "three"), sampling),
        )

        assert first == ["generated:one"]
        assert second == ["generated:two", "generated:three"]
        assert fake.calls == [(("one", "two", "three"), sampling)]

    asyncio.run(run())


def test_generation_batcher_default_zero_window_queues_without_lifetime_lock() -> None:
    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.0,
        )

        first, second = await asyncio.gather(
            batcher.submit(("one",), sampling),
            batcher.submit(("two",), sampling),
        )

        streamed = [chunk async for chunk in batcher.stream(("three",), sampling)]

        assert first == ["generated:one"]
        assert second == ["generated:two"]
        assert streamed == ["generated:three"]
        assert fake.calls == [(("one", "two"), sampling), (("three",), sampling)]

    asyncio.run(run())


def test_generation_batcher_stream_uses_per_request_queue_and_coalesces() -> None:
    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        async def collect_stream() -> list[str]:
            return [chunk async for chunk in batcher.stream(("stream",), sampling)]

        streamed, submitted = await asyncio.gather(
            collect_stream(),
            batcher.submit(("batch",), sampling),
        )

        assert streamed == ["generated:stream"]
        assert submitted == ["generated:batch"]
        assert len(fake.calls) == 1
        assert set(fake.calls[0][0]) == {"stream", "batch"}
        assert fake.calls[0][1] == sampling

    asyncio.run(run())


def test_generation_batcher_skips_cancelled_queued_submit() -> None:
    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        cancelled = asyncio.create_task(batcher.submit(("cancelled",), sampling))
        await asyncio.sleep(0)
        cancelled.cancel()
        try:
            await cancelled
        except asyncio.CancelledError:
            pass
        else:  # pragma: no cover - defensive guard for cancellation semantics
            raise AssertionError("cancelled submit task did not raise CancelledError")

        live = await batcher.submit(("live",), sampling)

        assert live == ["generated:live"]
        assert fake.calls == [(("live",), sampling)]

    asyncio.run(run())


def test_generation_batcher_rejects_when_queue_cap_is_full() -> None:
    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.01,
            max_queue_size=1,
            retry_after_seconds=2,
        )

        first = asyncio.create_task(batcher.submit(("one",), sampling))
        await asyncio.sleep(0)
        with pytest.raises(OpenAIHTTPError) as exc_info:
            await batcher.submit(("two",), sampling)

        exc = exc_info.value
        assert exc.status_code == 429
        assert exc.code == "engine_busy"
        assert exc.headers == {"Retry-After": "2"}
        assert await first == ["generated:one"]
        assert fake.calls == [(("one",), sampling)]

    asyncio.run(run())


def test_generation_batcher_keeps_incompatible_sampling_separate() -> None:
    async def run() -> None:
        fake = FakeLLM()
        first_sampling = SamplingParams(max_tokens=1)
        second_sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        first, second = await asyncio.gather(
            batcher.submit(("one",), first_sampling),
            batcher.submit(("two",), second_sampling),
        )

        assert first == ["generated:one"]
        assert second == ["generated:two"]
        assert fake.calls == [(("one",), first_sampling), (("two",), second_sampling)]

    asyncio.run(run())


def test_generation_batcher_keeps_different_deadlines_separate() -> None:
    async def run() -> None:
        fake = FakeLLM()
        first_sampling = SamplingParams(max_tokens=2, deadline_at=100.0)
        second_sampling = SamplingParams(max_tokens=2, deadline_at=101.0)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        first, second = await asyncio.gather(
            batcher.submit(("one",), first_sampling),
            batcher.submit(("two",), second_sampling),
        )

        assert first == ["generated:one"]
        assert second == ["generated:two"]
        assert fake.calls == [(("one",), first_sampling), (("two",), second_sampling)]

    asyncio.run(run())


def test_generation_batcher_keeps_different_cancellation_tokens_separate() -> None:
    async def run() -> None:
        fake = FakeLLM()
        first_sampling = SamplingParams(max_tokens=2, cancellation_token=GenerationCancellationToken())
        second_sampling = SamplingParams(max_tokens=2, cancellation_token=GenerationCancellationToken())
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        first, second = await asyncio.gather(
            batcher.submit(("one",), first_sampling),
            batcher.submit(("two",), second_sampling),
        )

        assert first == ["generated:one"]
        assert second == ["generated:two"]
        assert fake.calls == [(("one",), first_sampling), (("two",), second_sampling)]

    asyncio.run(run())


def test_request_control_maps_http_disconnect_to_cancelled_error() -> None:
    async def run() -> None:
        async def receive() -> dict[str, object]:
            return {"type": "http.disconnect"}

        raw_request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/completions",
                "headers": [],
                "query_string": b"",
            },
            receive,
        )
        control = _request_control(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            CompletionRequest(model="fake-model", prompt="hello"),
            raw_request,
        )

        async def work() -> str:
            await asyncio.sleep(1.0)
            return "late"

        try:
            await _await_with_request_control(work(), control)
        except OpenAIHTTPError as exc:
            assert exc.status_code == 499
            assert exc.error_type == "cancelled_error"
            assert exc.code == "cancelled"
            assert exc.finish_details == {"reason": "cancelled", "cancelled": True}
            assert control.cancellation_token.cancelled is True
            assert control.cancellation_token.finish_details.to_json_dict() == exc.finish_details
        else:  # pragma: no cover - defensive guard for cancellation semantics
            raise AssertionError("disconnect did not cancel request")

    asyncio.run(run())


def test_completions_endpoint_calls_llm_and_applies_stop() -> None:
    fake = FakeLLM(outputs=["alpha<stop>tail", "beta<stop>tail"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            kv_storage="int8_per_token_head",
            kv_scale_dtype="fp32",
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": ["one", "two"],
            "max_tokens": 3,
            "temperature": 0.0,
            "top_p": 1.0,
            "stop": "<stop>",
            "suppress_token_ids": [12, 13],
            "min_tokens": 2,
            "eos_token_id": 151645,
            "kv_storage": "int8_per_token_head",
            "kv_scale_dtype": "fp32",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "fake-model"
    assert [choice["text"] for choice in body["choices"]] == ["alpha", "beta"]
    assert [choice["finish_reason"] for choice in body["choices"]] == ["stop", "stop"]
    assert [choice["finish_details"] for choice in body["choices"]] == [{"reason": "stop"}, {"reason": "stop"}]
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
    assert fake.calls == [
        (
            ("one", "two"),
            SamplingParams(
                max_tokens=3,
                temperature=0.0,
                top_p=1.0,
                suppress_token_ids=(12, 13),
                min_tokens=2,
                eos_token_id=151645,
                ignore_eos=False,
                kv_storage="int8_per_token_head",
                kv_scale_dtype="fp32",
            ),
        )
    ]


def test_completions_preserve_structured_finish_details() -> None:
    fake = DetailedGenerateFakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="alpha",
                finish_details=FinishDetails(reason="eos", eos_token_id=151645, sampler_mode="greedy_fast"),
            ),
            GenerationOutput(
                text="beta",
                finish_details=FinishDetails(reason="length", length_limit=2, budget_pressure="answer_budget"),
            ),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": ["one", "two"], "max_tokens": 2},
    )

    assert response.status_code == 200
    choices = response.json()["choices"]
    assert [choice["finish_reason"] for choice in choices] == ["stop", "length"]
    assert choices[0]["finish_details"] == {
        "reason": "eos",
        "eos_token_id": 151645,
        "sampler_mode": "greedy_fast",
    }
    assert choices[1]["finish_details"] == {
        "reason": "length",
        "length_limit": 2,
        "budget_pressure": "answer_budget",
    }


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        (
            "/v1/completions",
            {"model": "fake-model", "prompt": "hello", "max_tokens": 1},
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
            },
        ),
    ],
)
def test_session_append_none_reports_cache_action(endpoint, payload) -> None:
    fake = FakeLLM(outputs=["reply"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(endpoint, json={**payload, "session": {"commit": "append_none"}})

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_details"]["cache_action"] == "append_none"


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        (
            "/v1/completions",
            {"model": "fake-model", "prompt": "hello", "max_tokens": 1},
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
            },
        ),
    ],
)
def test_streaming_session_append_none_reports_cache_action(endpoint, payload) -> None:
    fake = FakeLLM(outputs=["should-not-buffer"], stream_chunks=["reply"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        endpoint,
        json={**payload, "stream": True, "session": {"commit": "append_none"}},
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    done = next(item for item in payloads if item["choices"][0]["finish_reason"] == "stop")
    assert done["choices"][0]["finish_details"]["cache_action"] == "append_none"


def test_completions_response_format_json_object_validates_result() -> None:
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=["not json"]),
        )
    )

    valid = valid_client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {"type": "json_object"},
        },
    )
    invalid = invalid_client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {"type": "json_object"},
        },
    )

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"ok":true}'
    assert valid.json()["choices"][0]["finish_details"] == {"reason": "stop"}
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == {"reason": "schema_violation"}


def _response_json_schema() -> dict[str, Any]:
    return {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "path": {"type": "string", "minLength": 1},
            },
            "required": ["ok", "path"],
            "additionalProperties": False,
        },
    }


def test_completions_response_format_json_schema_validates_result() -> None:
    schema = _response_json_schema()
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"path":"README.md"}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":"yes","path":"README.md"}']),
        )
    )

    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"ok":true,"path":"README.md"}'
    assert valid.json()["choices"][0]["finish_details"] == {"reason": "stop"}
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == {"reason": "schema_violation"}


def test_completions_response_format_rejects_unsupported_modes() -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    unsupported = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {"type": "xml"},
        },
    )
    missing_schema = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
        },
    )
    echo = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "echo": True,
            "response_format": {"type": "json_object"},
        },
    )

    assert unsupported.status_code == 400
    assert unsupported.json()["error"]["code"] == "unsupported_parameter"
    assert unsupported.json()["error"]["param"] == "response_format"
    assert missing_schema.status_code == 400
    assert missing_schema.json()["error"]["code"] == "invalid_request"
    assert missing_schema.json()["error"]["param"] == "response_format.json_schema.schema"
    assert echo.status_code == 400
    assert echo.json()["error"]["code"] == "invalid_request"
    assert echo.json()["error"]["param"] == "echo"
    assert fake.calls == []


def test_server_lowers_single_token_stop_strings_to_stop_token_ids() -> None:
    fake = FakeLLM(outputs=["alpha!tail"], token_map={"!": [99], "two tokens": [10, 11]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "one",
            "max_tokens": 4,
            "stop": ["!", "two tokens"],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "alpha"
    sampling = fake.calls[0][1]
    assert sampling.stop_token_ids == (99,)
    assert sampling.stop_token_sequences == ((10, 11),)
    assert fake.tokenize_calls == ["!", "two tokens"]


def test_chat_completion_uses_bounded_default_max_tokens() -> None:
    fake = FakeLLM(outputs=["assistant reply"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            max_context_tokens=100,
            chat_default_max_tokens=7,
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert fake.calls[0][1].max_tokens == 7


def test_chat_completion_auto_default_max_tokens_uses_remaining_context() -> None:
    fake = FakeLLM(outputs=["assistant reply"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            max_context_tokens=12,
            chat_default_max_tokens=None,
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[0][0][0]
    assert fake.calls[0][1].max_tokens == 12 - fake.count_tokens(prompt) - 1


def test_completions_endpoint_plumbs_sampling_parameters() -> None:
    fake = FakeLLM(outputs=["sampled"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "one",
            "max_tokens": 2,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.3,
            "logit_bias": {"12": -1.5},
            "seed": 123,
        },
    )

    assert response.status_code == 200
    sampling = fake.calls[0][1]
    assert sampling.temperature == 0.8
    assert sampling.top_p == 0.9
    assert sampling.top_k == 40
    assert sampling.min_p == 0.05
    assert sampling.repetition_penalty == 1.1
    assert sampling.presence_penalty == 0.2
    assert sampling.frequency_penalty == 0.3
    assert sampling.logit_bias == ((12, -1.5),)
    assert sampling.seed == 123


def test_completion_timeout_returns_deadline_error_and_server_reuses() -> None:
    fake = DelayedFakeLLM(outputs=["late", "ok"], generate_delay_s=0.03)
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    timed_out = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "slow", "max_tokens": 1, "timeout_ms": 1},
    )

    assert timed_out.status_code == 408
    error = timed_out.json()["error"]
    assert error["type"] == "timeout_error"
    assert error["code"] == "deadline_exceeded"
    assert error["param"] == "timeout_ms"
    assert error["hipengine"] == {
        "code": "deadline_exceeded",
        "status_code": 408,
        "retryable": True,
    }
    assert error["finish_details"] == {"reason": "deadline_exceeded", "deadline_exceeded": True}

    # With very short deadlines the worker may time out before the threadpool
    # starts it, or it may unwind later. The public contract is server reuse.
    fake.generate_delay_s = 0.0
    fake.outputs = ["ok"]
    reused = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "after", "max_tokens": 1},
    )

    assert reused.status_code == 200
    assert reused.json()["choices"][0]["text"] == "ok"


def test_backend_deadline_exception_maps_to_completion_408() -> None:
    fake = BackendDeadlineFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "slow", "max_tokens": 4, "timeout_ms": 5000},
    )

    assert response.status_code == 408
    assert fake.calls[0][1].deadline_at is not None
    error = response.json()["error"]
    assert error["type"] == "timeout_error"
    assert error["code"] == "deadline_exceeded"
    assert error["param"] == "timeout_ms"
    assert error["finish_details"] == {"reason": "deadline_exceeded", "deadline_exceeded": True}


def test_backend_cancelled_exception_maps_to_completion_499() -> None:
    fake = BackendCancelledFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "cancel", "max_tokens": 4},
    )

    assert response.status_code == 499
    assert fake.calls[0][1].cancellation_token is not None
    assert fake.calls[0][1].cancellation_token.cancelled is True
    error = response.json()["error"]
    assert error["type"] == "cancelled_error"
    assert error["code"] == "cancelled"
    assert error["finish_details"] == {"reason": "cancelled", "cancelled": True}


def test_backend_deadline_finish_detail_maps_to_chat_408() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="deadline_exceeded", deadline_exceeded=True),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "slow"}],
            "max_tokens": 4,
            "timeout_ms": 5000,
        },
    )

    assert response.status_code == 408
    assert fake.calls[0][1].deadline_at is not None
    error = response.json()["error"]
    assert error["code"] == "deadline_exceeded"
    assert error["finish_details"] == {"reason": "deadline_exceeded", "deadline_exceeded": True}


def test_streaming_completion_timeout_emits_error_and_done() -> None:
    fake = DelayedFakeLLM(outputs=["ok"], stream_chunks=["late"], stream_delay_s=0.03)
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "slow", "max_tokens": 1, "stream": True, "timeout_ms": 1},
    )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    payloads = _sse_payloads(response.text)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["choices"][0]["finish_reason"] == "error"
    assert payload["choices"][0]["finish_details"] == {
        "reason": "deadline_exceeded",
        "deadline_exceeded": True,
    }
    assert payload["error"]["type"] == "timeout_error"
    assert payload["error"]["code"] == "deadline_exceeded"
    assert payload["error"]["param"] == "timeout_ms"
    assert payload["error"]["hipengine"] == {
        "code": "deadline_exceeded",
        "status_code": 408,
        "retryable": True,
    }
    assert payload["error"]["finish_details"] == payload["choices"][0]["finish_details"]


def test_streaming_completion_timeout_can_include_hipengine_error_metadata() -> None:
    fake = DelayedFakeLLM(outputs=["ok"], stream_chunks=["late"], stream_delay_s=0.03)
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "slow",
            "max_tokens": 1,
            "stream": True,
            "timeout_ms": 1,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["hipengine"]["event"] == "error"
    assert isinstance(payload["hipengine"]["timing"]["elapsed_ms"], float)
    assert payload["choices"][0]["hipengine"] == {
        "phase": "done",
        "finish_details": {"reason": "deadline_exceeded", "deadline_exceeded": True},
    }
    assert payload["error"]["finish_details"] == payload["choices"][0]["finish_details"]


def test_streaming_chat_backend_deadline_exception_emits_error_and_done() -> None:
    fake = BackendDeadlineFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "slow"}],
            "max_tokens": 4,
            "stream": True,
            "timeout_ms": 5000,
        },
    )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert fake.calls[0][1].deadline_at is not None
    payloads = _sse_payloads(response.text)
    payload = next(item for item in payloads if item.get("error"))
    assert payload["choices"][0]["finish_reason"] == "error"
    assert payload["choices"][0]["finish_details"] == {
        "reason": "deadline_exceeded",
        "deadline_exceeded": True,
    }
    assert payload["error"]["type"] == "timeout_error"
    assert payload["error"]["code"] == "deadline_exceeded"
    assert payload["error"]["param"] == "timeout_ms"


def test_streaming_completion_backend_cancelled_exception_emits_error_and_done() -> None:
    fake = BackendCancelledFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "cancel", "max_tokens": 4, "stream": True},
    )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text
    assert fake.calls[0][1].cancellation_token is not None
    assert fake.calls[0][1].cancellation_token.cancelled is True
    payloads = _sse_payloads(response.text)
    payload = next(item for item in payloads if item.get("error"))
    assert payload["choices"][0]["finish_reason"] == "error"
    assert payload["choices"][0]["finish_details"] == {
        "reason": "cancelled",
        "cancelled": True,
    }
    assert payload["error"]["type"] == "cancelled_error"
    assert payload["error"]["code"] == "cancelled"


def test_completions_endpoint_returns_openai_logprobs() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="alpha beta",
                token_logprobs=(
                    TokenLogprob(
                        token_id=1,
                        token_text="alpha",
                        logprob=-0.25,
                        top_logprobs=((1, "alpha", -0.25), (2, "omega", -1.5)),
                    ),
                    TokenLogprob(
                        token_id=3,
                        token_text=" beta",
                        logprob=-0.5,
                        top_logprobs=((3, " beta", -0.5),),
                    ),
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 2, "logprobs": 2},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == "alpha beta"
    assert choice["logprobs"]["tokens"] == ["alpha", " beta"]
    assert choice["logprobs"]["token_logprobs"] == [-0.25, -0.5]
    assert choice["logprobs"]["top_logprobs"][0] == {"alpha": -0.25, "omega": -1.5}
    assert fake.calls[0][1].logprobs is True
    assert fake.calls[0][1].top_logprobs == 2


def test_completions_endpoint_echo_logprobs_shift_generated_offsets() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text=" world",
                token_logprobs=(
                    TokenLogprob(token_id=7, token_text=" world", logprob=-0.125),
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 1, "echo": True, "logprobs": 0},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == "hello world"
    assert choice["logprobs"]["tokens"] == ["hello", " world"]
    assert choice["logprobs"]["token_logprobs"] == [None, -0.125]
    assert choice["logprobs"]["text_offset"] == [0, 5]


def test_streaming_completion_returns_logprobs_from_buffered_path() -> None:
    fake = FakeLLM(
        outputs=["should-not-stream"],
        stream_chunks=["wrong"],
        detailed_outputs=[
            GenerationOutput(
                text="alpha",
                token_logprobs=(
                    TokenLogprob(token_id=1, token_text="alpha", logprob=-0.25, top_logprobs=((1, "alpha", -0.25),)),
                ),
            )
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 1, "stream": True, "logprobs": 1},
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert payloads[0]["choices"][0]["text"] == "alpha"
    assert payloads[0]["choices"][0]["logprobs"]["token_logprobs"] == [-0.25]
    assert payloads[-1]["choices"][0]["finish_details"] == {"reason": "stop"}
    assert fake.stream_calls == []
    assert fake.calls[0][1].logprobs is True


def test_streaming_completion_response_format_buffers_validation() -> None:
    fake = FakeLLM(outputs=["not json"], stream_chunks=["wrong"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "stream": True,
            "response_format": {"type": "json_object"},
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0].get("text") for payload in payloads if payload.get("choices"))
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == {"reason": "schema_violation"}
    assert done["choices"][0]["hipengine"] == {
        "phase": "done",
        "finish_details": {"reason": "schema_violation"},
    }
    assert fake.stream_calls == []


def test_streaming_completion_response_format_json_schema_buffers_validation() -> None:
    fake = FakeLLM(outputs=['{"ok":"yes","path":"README.md"}'], stream_chunks=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "stream": True,
            "response_format": {"type": "json_schema", "json_schema": _response_json_schema()},
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0].get("text") for payload in payloads if payload.get("choices"))
    done = next(
        payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"]
    )
    assert done["choices"][0]["finish_details"] == {"reason": "schema_violation"}
    assert done["choices"][0]["hipengine"] == {
        "phase": "done",
        "finish_details": {"reason": "schema_violation"},
    }
    assert fake.stream_calls == []


def test_chat_completion_returns_openai_logprobs() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="assistant reply",
                token_logprobs=(
                    TokenLogprob(
                        token_id=4,
                        token_text="assistant",
                        logprob=-0.1,
                        top_logprobs=((4, "assistant", -0.1),),
                    ),
                    TokenLogprob(token_id=5, token_text=" reply", logprob=-0.2),
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "logprobs": True,
            "top_logprobs": 1,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"]["content"] == "assistant reply"
    assert choice["logprobs"]["content"][0]["token"] == "assistant"
    assert choice["logprobs"]["content"][0]["top_logprobs"] == [
        {"token": "assistant", "logprob": -0.1, "bytes": None}
    ]
    assert fake.calls[0][1].logprobs is True
    assert fake.calls[0][1].top_logprobs == 1


def test_streaming_chat_completion_returns_logprobs_from_buffered_path() -> None:
    fake = FakeLLM(
        outputs=["should-not-stream"],
        stream_chunks=["wrong"],
        detailed_outputs=[
            GenerationOutput(
                text="assistant",
                token_logprobs=(
                    TokenLogprob(token_id=4, token_text="assistant", logprob=-0.1),
                ),
            )
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "stream": True,
            "logprobs": True,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content_chunks = [payload for payload in payloads if payload.get("choices") and payload["choices"][0]["delta"].get("content")]
    assert content_chunks[0]["choices"][0]["delta"] == {"content": "assistant"}
    assert content_chunks[0]["choices"][0]["logprobs"]["content"][0]["logprob"] == -0.1
    assert payloads[-1]["choices"][0]["finish_details"] == {"reason": "stop"}
    assert fake.stream_calls == []
    assert fake.calls[0][1].logprobs is True


def test_chat_completion_renders_messages_to_prompt() -> None:
    fake = FakeLLM(outputs=["assistant reply"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
            "max_tokens": 4,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "assistant reply"},
            "finish_reason": "stop",
            "finish_details": {"reason": "stop"},
        }
    ]
    assert fake.calls[0][0] == (
        "<|im_start|>system\nbe concise<|im_end|>\n"
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n",
    )
    assert fake.calls[0][1].max_tokens == 4


def test_chat_completion_segregates_reasoning_content() -> None:
    fake = FakeLLM(outputs=["<think>scratch pad</think>assistant reply"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message == {
        "role": "assistant",
        "content": "assistant reply",
        "reasoning_content": "scratch pad",
    }


def test_chat_completion_accepts_qwen_no_think_controls() -> None:
    fake = FakeLLM(outputs=["direct answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "answer directly"}],
            "enable_thinking": False,
        },
    )

    assert response.status_code == 200
    assert fake.calls[0][0][0].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_chat_completion_accepts_reasoning_effort_controls() -> None:
    fake = FakeLLM(outputs=["reasoned answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    low = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think briefly"}],
            "reasoning_effort": "low",
        },
    )
    assert low.status_code == 200
    assert "keep it very brief" in fake.calls[-1][0][0]
    assert "close </think> before exceeding 512 hidden reasoning tokens" in fake.calls[-1][0][0]
    assert "reserve at least 512 tokens for the final answer or tool call" in fake.calls[-1][0][0]
    assert "begin closing during the final 128 hidden reasoning tokens" in fake.calls[-1][0][0]
    assert not fake.calls[-1][0][0].endswith("<think>\n\n</think>\n\n")

    none = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "do not think"}],
            "reasoning_effort": "none",
        },
    )
    assert none.status_code == 200
    assert fake.calls[-1][0][0].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_chat_completion_clamps_reasoning_effort_defaults_to_generation_budget() -> None:
    fake = FakeLLM(outputs=["reasoned answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "debug"}],
            "reasoning_effort": "medium",
            "max_tokens": 100,
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[-1][0][0]
    assert "keep it concise" in prompt
    assert "close </think> before exceeding 50 hidden reasoning tokens" in prompt
    assert "reserve at least 50 tokens for the final answer or tool call" in prompt
    assert "begin closing during the final 50 hidden reasoning tokens" in prompt


def test_chat_completion_clamps_reasoning_effort_defaults_to_remaining_context() -> None:
    fake = FakeLLM(outputs=["reasoned answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            max_context_tokens=120,
            chat_default_max_tokens=4096,
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "debug"}],
            "reasoning_effort": "medium",
        },
    )

    assert response.status_code == 200
    prompt, sampling = fake.calls[-1]
    admitted_budget = int(sampling.max_tokens)
    expected_reserved = admitted_budget // 2
    expected_think_cap = admitted_budget - expected_reserved
    assert admitted_budget < 4096
    assert f"close </think> before exceeding {expected_think_cap} hidden reasoning tokens" in prompt[0]
    assert f"reserve at least {expected_reserved} tokens for the final answer or tool call" in prompt[0]
    assert f"begin closing during the final {expected_think_cap} hidden reasoning tokens" in prompt[0]
    assert "4096 hidden reasoning tokens" not in prompt[0]
    assert "1024 tokens for the final answer" not in prompt[0]


def test_chat_completion_clamps_explicit_thinking_budget_hints() -> None:
    fake = FakeLLM(outputs=["bounded answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think within budget"}],
            "max_tokens": 100,
            "hard_think_cap": 90,
            "min_answer_tokens": 80,
            "soft_close_window": 200,
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[-1][0][0]
    assert "close </think> before exceeding 50 hidden reasoning tokens" in prompt
    assert "reserve at least 50 tokens for the final answer or tool call" in prompt
    assert "begin closing during the final 50 hidden reasoning tokens" in prompt


def test_chat_completion_accepts_thinking_budget_prompt_hints() -> None:
    fake = FakeLLM(outputs=["bounded answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think, then answer"}],
            "chat_template_kwargs": {"thinking_budget": 99, "reasoning_effort": "low"},
            "thinking_token_budget": 123,
            "max_think_tokens": 32,
            "min_answer_tokens": 8,
            "soft_close_window": 4,
            "hard_close_message": "closing now",
            "thinking": {
                "budget_tokens": 456,
                "hard_close_sequence": "closing now</think>\n",
            },
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[-1][0][0]
    assert "keep it very brief" in prompt
    assert "aim to close hidden reasoning within 32 tokens" in prompt
    assert "close </think> before exceeding 456 hidden reasoning tokens" in prompt
    assert "reserve at least 8 tokens for the final answer or tool call" in prompt
    assert "begin closing during the final 4 hidden reasoning tokens" in prompt
    assert "use the close message 'closing now' only if budget pressure requires it" in prompt
    assert "use 'closing now</think>\\n' as the close sequence if budget pressure requires it" in prompt
    assert "exceeding 99 hidden reasoning tokens" not in prompt
    assert "exceeding 123 hidden reasoning tokens" not in prompt


def test_chat_completion_lowers_thinking_budget_into_sampling_params() -> None:
    fake = FakeLLM(outputs=["bounded answer"], token_map={"</think>": [42, 43]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think, then answer"}],
            "max_tokens": 20,
            "hard_think_cap": 12,
            "soft_close_window": 3,
        },
    )

    assert response.status_code == 200
    assert fake.tokenize_calls == ["</think>"]
    sampling = fake.calls[-1][1]
    assert sampling.thinking_close_token_ids == (42, 43)
    assert sampling.thinking_hard_token_cap == 12
    assert sampling.thinking_soft_close_window == 3


def test_chat_completion_preserves_string_thinking_budget_effort_alias() -> None:
    fake = FakeLLM(outputs=["answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "chat_template_kwargs": {"thinking_budget": "high"},
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[-1][0][0]
    assert "keep it focused but complete" in prompt
    assert "close </think> before exceeding 2048 hidden reasoning tokens" in prompt
    assert "reserve at least 2048 tokens for the final answer or tool call" in prompt
    assert "begin closing during the final 1024 hidden reasoning tokens" in prompt


def test_chat_completion_rejects_hard_close_without_think_marker() -> None:
    fake = FakeLLM(outputs=["unused"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "hard_close_sequence": "DONE\n",
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "hard_close_sequence"
    assert fake.calls == []


def test_chat_completion_rejects_invalid_thinking_budget_value() -> None:
    fake = FakeLLM(outputs=["unused"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "thinking": {"max_tokens": "soon"},
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "thinking.max_tokens"
    assert fake.calls == []


@pytest.mark.parametrize(
    ("output", "phase", "content", "reasoning_content"),
    [
        ("<think>scratch", "reasoning", "", "scratch"),
        ("<think>scratch</thi", "closing_think", "", "scratch</thi"),
        ('<tool_call>{"name":"read"', "tool_call", '<tool_call>{"name":"read"', None),
        ('{"status":', "structured", '{"status":', None),
        ("partial answer", "answer", "partial answer", None),
    ],
)
def test_chat_completion_length_finish_details_include_phase(
    output: str,
    phase: str,
    content: str,
    reasoning_content: str | None,
) -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text=output,
                finish_details=FinishDetails(reason="length", length_limit=5),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "continue"}]},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == {
        "reason": "length",
        "length_limit": 5,
        "phase": phase,
        "continuation_eligible": False,
    }
    assert choice["message"]["content"] == content
    if reasoning_content is None:
        assert "reasoning_content" not in choice["message"]
    else:
        assert choice["message"]["reasoning_content"] == reasoning_content


def test_chat_completion_response_format_json_object_validates_visible_content() -> None:
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['<think>check</think>{"ok":true}']),
        )
    )
    invalid_fake = FakeLLM(outputs=["not json"])
    invalid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=invalid_fake)
    )

    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "return json"}],
        "response_format": {"type": "json_object"},
    }
    valid = valid_client.post("/v1/chat/completions", json=payload)
    invalid = invalid_client.post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["message"]["content"] == '{"ok":true}'
    assert valid_choice["message"]["reasoning_content"] == "check"
    assert valid_choice["finish_details"] == {"reason": "stop"}
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["message"] == {"role": "assistant", "content": ""}
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == {"reason": "schema_violation"}
    assert "Return only one valid JSON object" in invalid_fake.calls[0][0][0]


def test_chat_completion_response_format_json_schema_validates_visible_content() -> None:
    valid_fake = FakeLLM(outputs=['<think>check</think>{"ok":true,"path":"README.md"}'])
    invalid_fake = FakeLLM(outputs=['{"ok":true,"path":""}'])
    valid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=valid_fake)
    )
    invalid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=invalid_fake)
    )

    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "return json"}],
        "response_format": {"type": "json_schema", "json_schema": _response_json_schema()},
    }
    valid = valid_client.post("/v1/chat/completions", json=payload)
    invalid = invalid_client.post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["message"]["content"] == '{"ok":true,"path":"README.md"}'
    assert valid_choice["message"]["reasoning_content"] == "check"
    assert valid_choice["finish_details"] == {"reason": "stop"}
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["message"] == {"role": "assistant", "content": ""}
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == {"reason": "schema_violation"}
    assert "Return only JSON that satisfies this JSON schema" in invalid_fake.calls[0][0][0]


def test_chat_completion_response_format_length_keeps_partial_json() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text='{"ok":',
                finish_details=FinishDetails(reason="length", length_limit=6),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_object"},
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": '{"ok":'}
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == {
        "reason": "length",
        "length_limit": 6,
        "phase": "structured",
        "continuation_eligible": False,
    }


def test_streaming_chat_completion_response_format_buffers_validation() -> None:
    fake = FakeLLM(outputs=["not json"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_object"},
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0]["delta"].get("content") for payload in payloads if payload.get("choices"))
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == {"reason": "schema_violation"}
    assert done["choices"][0]["hipengine"] == {
        "phase": "done",
        "finish_details": {"reason": "schema_violation"},
    }


def test_streaming_chat_completion_response_format_json_schema_buffers_validation() -> None:
    fake = FakeLLM(outputs=['{"ok":true,"path":""}'], stream_chunks=['{"ok":true,"path":"README.md"}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_schema", "json_schema": _response_json_schema()},
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(
        payload["choices"][0]["delta"].get("content")
        for payload in payloads
        if payload.get("choices")
    )
    done = next(
        payload
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"]
    )
    assert done["choices"][0]["finish_details"] == {"reason": "schema_violation"}
    assert done["choices"][0]["hipengine"] == {
        "phase": "done",
        "finish_details": {"reason": "schema_violation"},
    }
    assert fake.stream_calls == []


def test_render_chat_prompt_includes_qwen_tool_blocks() -> None:
    prompt = render_chat_prompt(
        [
            {"role": "developer", "content": "Use tools carefully."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path":"README.md"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "file text"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "read"}},
    )

    assert "<tools>" in prompt
    assert '"name":"read"' in prompt
    assert "You must call the function named 'read'." in prompt
    assert "<|im_start|>system\nUse tools carefully.<|im_end|>" in prompt
    assert '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>' in prompt
    assert "<tool_response>\nfile text\n</tool_response>" in prompt


def test_chat_completion_returns_openai_tool_calls() -> None:
    fake = FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["finish_details"] == {"reason": "tool_calls"}
    message = choice["message"]
    assert message["content"] == ""
    tool_call = message["tool_calls"][0]
    assert tool_call["id"].startswith("call_")
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "read"
    assert json.loads(tool_call["function"]["arguments"]) == {"path": "README.md"}
    assert "<tools>" in fake.calls[0][0][0]


def test_chat_completion_preserves_reasoning_with_openai_tool_call() -> None:
    fake = FakeLLM(
        outputs=['<think>need file</think><tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>']
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["finish_details"] == {"reason": "tool_calls"}
    message = choice["message"]
    assert message["content"] == ""
    assert message["reasoning_content"] == "need file"
    tool_call = message["tool_calls"][0]
    assert tool_call["function"]["name"] == "read"
    assert json.loads(tool_call["function"]["arguments"]) == {"path": "README.md"}
    assert "<tool_call>" not in json.dumps(message)


def test_chat_completion_strict_validation_rejects_doubled_tool_call_tag() -> None:
    fake = FakeLLM(
        outputs=['<tool_call>\n<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>']
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "strict": True,
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == {"reason": "invalid_tool_call"}
    assert choice["message"] == {"role": "assistant", "content": ""}
    assert "<tool_call>" not in response.text


def test_chat_completion_required_tool_reports_missing_call() -> None:
    fake = FakeLLM(outputs=["ordinary answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tool_choice": "required",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == {"reason": "tool_required_not_satisfied"}
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_specific_tool_rejects_wrong_function() -> None:
    fake = FakeLLM(outputs=['<tool_call>{"name":"write","arguments":{"path":"README.md"}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tool_choice": {"type": "function", "function": {"name": "read"}},
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "write", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == {"reason": "invalid_tool_call"}
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_tool_choice_none_rejects_tool_call() -> None:
    fake = FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "answer without tools"}],
            "tool_choice": "none",
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == {"reason": "invalid_tool_call"}
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_tool_choice_none_suppresses_tool_call_start_token() -> None:
    fake = FakeLLM(outputs=["plain answer"], token_map={"<tool_call>": [77, 78]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "answer without tools"}],
            "tool_choice": "none",
            "suppress_token_ids": [13],
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert fake.tokenize_calls == ["<tool_call>"]
    assert fake.calls[-1][1].suppress_token_ids == (13, 77)
    assert response.json()["choices"][0]["message"]["content"] == "plain answer"


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        {"type": "function", "function": {"name": "read"}},
    ],
)
def test_chat_completion_required_tool_choice_forces_tool_call_start_tokens(tool_choice) -> None:
    fake = FakeLLM(outputs=["ordinary answer"], token_map={"<tool_call>": [77, 78]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tool_choice": tool_choice,
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert fake.tokenize_calls == ["<tool_call>"]
    assert fake.calls[-1][1].forced_tokens_pending == (77, 78)
    assert fake.calls[-1][1].forced_token_reason == "tool_choice_required"
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == {"reason": "tool_required_not_satisfied"}
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_required_tool_choice_queues_tool_start_after_thinking_budget() -> None:
    fake = FakeLLM(outputs=["ordinary answer"], token_map={"</think>": [91, 92], "<tool_call>": [77, 78]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tool_choice": {"type": "function", "function": {"name": "read"}},
            "reasoning": {"enabled": True, "effort": "low"},
            "max_tokens": 2048,
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert fake.tokenize_calls == ["</think>", "<tool_call>"]
    params = fake.calls[-1][1]
    assert params.forced_tokens_pending == ()
    assert params.post_thinking_forced_tokens_pending == (77, 78)
    assert params.post_thinking_forced_token_reason == "tool_choice_required"
    assert params.thinking_close_token_ids == (91, 92)
    assert params.thinking_hard_token_cap == 512
    choice = response.json()["choices"][0]
    assert choice["finish_details"] == {"reason": "tool_required_not_satisfied"}


def test_chat_completion_strict_tool_schema_reports_schema_violation() -> None:
    fake = FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":123,"mode":"raw"}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == {"reason": "schema_violation"}
    assert "tool_calls" not in choice["message"]


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"path": "README.md", "mode": "raw"},
    ],
)
def test_chat_completion_strict_tool_schema_rejects_missing_and_extra_arguments(arguments) -> None:
    fake = FakeLLM(outputs=[f'<tool_call>{{"name":"read","arguments":{json.dumps(arguments)}}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_details"] == {"reason": "schema_violation"}


def _bounded_tool_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {"const": "file"},
            "path": {"type": "string", "minLength": 1, "maxLength": 64},
            "mode": {"type": "string", "enum": ["raw", "summary"]},
            "tags": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {"type": "string", "minLength": 1, "maxLength": 16},
            },
            "filters": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string", "enum": ["ext", "name"]},
                        "value": {"type": "string", "minLength": 1},
                    },
                    "required": ["field", "value"],
                    "additionalProperties": False,
                },
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 3},
        },
        "required": ["kind", "path", "mode", "tags", "limit"],
        "additionalProperties": False,
    }


def test_chat_completion_strict_tool_schema_accepts_bounded_subset() -> None:
    arguments = {
        "kind": "file",
        "path": "README.md",
        "mode": "summary",
        "tags": ["docs"],
        "filters": [{"field": "ext", "value": "md"}],
        "limit": 2,
    }
    fake = FakeLLM(outputs=[f'<tool_call>{{"name":"read","arguments":{json.dumps(arguments)}}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "strict": True,
                        "parameters": _bounded_tool_schema(),
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["finish_details"] == {"reason": "tool_calls"}
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == arguments


@pytest.mark.parametrize(
    "arguments",
    [
        {"kind": "directory", "path": "README.md", "mode": "summary", "tags": ["docs"], "limit": 2},
        {"kind": "file", "path": "", "mode": "summary", "tags": ["docs"], "limit": 2},
        {"kind": "file", "path": "README.md", "mode": "binary", "tags": ["docs"], "limit": 2},
        {"kind": "file", "path": "README.md", "mode": "summary", "tags": [], "limit": 2},
        {"kind": "file", "path": "README.md", "mode": "summary", "tags": ["a", "b", "c"], "limit": 2},
        {"kind": "file", "path": "README.md", "mode": "summary", "tags": ["docs"], "limit": 4},
        {
            "kind": "file",
            "path": "README.md",
            "mode": "summary",
            "tags": ["docs"],
            "filters": [{"field": "suffix", "value": "md"}],
            "limit": 2,
        },
        {
            "kind": "file",
            "path": "README.md",
            "mode": "summary",
            "tags": ["docs"],
            "filters": [{"field": "ext"}],
            "limit": 2,
        },
    ],
)
def test_chat_completion_strict_tool_schema_rejects_bounded_subset_violations(arguments) -> None:
    fake = FakeLLM(outputs=[f'<tool_call>{{"name":"read","arguments":{json.dumps(arguments)}}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "strict": True,
                        "parameters": _bounded_tool_schema(),
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == {"reason": "schema_violation"}
    assert "tool_calls" not in choice["message"]


def test_chat_completion_parallel_tool_calls_require_explicit_opt_in() -> None:
    output = (
        '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'
        '<tool_call>{"name":"read","arguments":{"path":"WORKLOG.md"}}</tool_call>'
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    rejected = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=[output]),
    )
    accepted = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=[output]),
    )

    rejected_response = TestClient(rejected).post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "read files"}], "tools": tools},
    )
    accepted_response = TestClient(accepted).post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read files"}],
            "tools": tools,
            "parallel_tool_calls": True,
        },
    )

    assert rejected_response.json()["choices"][0]["finish_details"] == {"reason": "invalid_tool_call"}
    accepted_choice = accepted_response.json()["choices"][0]
    assert accepted_choice["finish_reason"] == "tool_calls"
    assert len(accepted_choice["message"]["tool_calls"]) == 2


def test_streaming_chat_completion_returns_tool_call_deltas() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=['<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run pwd"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    payloads = _sse_payloads(response.text)
    tool_delta = next(payload for payload in payloads if payload["choices"][0]["delta"].get("tool_calls"))
    tool_call = tool_delta["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call["index"] == 0
    assert tool_call["id"].startswith("call_")
    assert tool_call["function"]["name"] == "bash"
    assert json.loads(tool_call["function"]["arguments"]) == {"command": "pwd"}
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert payloads[-1]["choices"][0]["finish_details"] == {"reason": "tool_calls"}


def test_streaming_chat_completion_preserves_reasoning_with_tool_call() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            '<think>need shell</think><tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run pwd"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a command",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    payloads = _sse_payloads(response.text)
    reasoning = next(payload for payload in payloads if payload["choices"][0]["delta"].get("reasoning_content"))
    assert reasoning["choices"][0]["delta"] == {"reasoning_content": "need shell"}
    tool_delta = next(payload for payload in payloads if payload["choices"][0]["delta"].get("tool_calls"))
    tool_call = tool_delta["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "bash"
    assert json.loads(tool_call["function"]["arguments"]) == {"command": "pwd"}
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert payloads[-1]["choices"][0]["finish_details"] == {"reason": "tool_calls"}


def test_streaming_chat_completion_strict_validation_rejects_doubled_tool_call_tag() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=['<tool_call>\n<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>'],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run pwd"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "strict": True,
                        "parameters": {"type": "object"},
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert "<tool_call>" not in response.text
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0]["delta"].get("tool_calls") for payload in payloads)
    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_reason"] == "stop"
    assert done["choices"][0]["finish_details"] == {"reason": "invalid_tool_call"}


def test_streaming_chat_completion_preserves_parallel_tool_call_indexes() -> None:
    output = (
        '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'
        '<tool_call>{"name":"read","arguments":{"path":"WORKLOG.md"}}</tool_call>'
    )
    fake = FakeLLM(outputs=["should-not-buffer"], stream_chunks=[output])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read files"}],
            "stream": True,
            "parallel_tool_calls": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    tool_calls = [
        payload["choices"][0]["delta"]["tool_calls"][0]
        for payload in payloads
        if payload["choices"][0]["delta"].get("tool_calls")
    ]
    assert [call["index"] for call in tool_calls] == [0, 1]
    assert [call["function"]["name"] for call in tool_calls] == ["read", "read"]
    assert [json.loads(call["function"]["arguments"]) for call in tool_calls] == [
        {"path": "README.md"},
        {"path": "WORKLOG.md"},
    ]
    assert tool_calls[0]["id"] != tool_calls[1]["id"]
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert payloads[-1]["choices"][0]["finish_details"] == {"reason": "tool_calls"}


def test_streaming_chat_completion_reports_strict_tool_schema_failure() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=['<tool_call>{"name":"bash","arguments":{"command":7}}</tool_call>'],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run pwd"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0]["delta"].get("tool_calls") for payload in payloads)
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert payloads[-1]["choices"][0]["finish_details"] == {"reason": "schema_violation"}


def test_streaming_chat_timeout_can_include_hipengine_error_metadata() -> None:
    fake = DelayedFakeLLM(outputs=["ok"], stream_chunks=["late"], stream_delay_s=0.03)
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "timeout_ms": 1,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert payloads[0]["hipengine"]["event"] == "role"
    payload = next(item for item in payloads if item.get("error"))
    assert payload["hipengine"]["event"] == "error"
    assert isinstance(payload["hipengine"]["timing"]["elapsed_ms"], float)
    assert payload["choices"][0]["hipengine"] == {
        "phase": "done",
        "finish_details": {"reason": "deadline_exceeded", "deadline_exceeded": True},
    }
    assert payload["error"]["finish_details"] == payload["choices"][0]["finish_details"]


def test_streaming_chat_completion_returns_token_sse_and_usage() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=["<think>scratch pad</think>streamed ", "reply"],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"object":"chat.completion.chunk"' in response.text
    assert "data: [DONE]" in response.text
    payloads = _sse_payloads(response.text)
    assert all("hipengine" not in payload for payload in payloads)
    assert all("hipengine" not in payload["choices"][0] for payload in payloads if payload.get("choices"))
    deltas = [payload["choices"][0]["delta"] for payload in payloads if payload.get("choices")]
    assert deltas[:4] == [
        {"role": "assistant"},
        {"reasoning_content": "scratch pad"},
        {"content": "streamed "},
        {"content": "reply"},
    ]
    assert len(fake.stream_calls) == 1
    prompt = fake.calls[0][0][0]
    completion_tokens = fake.count_tokens("<think>scratch pad</think>streamed reply")
    assert payloads[-1]["usage"] == {
        "prompt_tokens": fake.count_tokens(prompt),
        "completion_tokens": completion_tokens,
        "total_tokens": fake.count_tokens(prompt) + completion_tokens,
    }
    assert fake.calls[0][1].max_tokens == 4096


def test_streaming_chat_completion_can_include_hipengine_metadata() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=["<think>scratch pad</think>streamed ", "reply"],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert payloads[0]["hipengine"]["metadata_version"] == 1
    assert payloads[0]["hipengine"]["event"] == "role"
    assert isinstance(payloads[0]["hipengine"]["timing"]["elapsed_ms"], float)

    reasoning = next(payload for payload in payloads if payload.get("choices") and "reasoning_content" in payload["choices"][0]["delta"])
    assert reasoning["hipengine"]["event"] == "delta"
    assert reasoning["choices"][0]["hipengine"] == {
        "phase": "think",
        "tokens": {
            "streamed_tokens": 2,
            "delta_tokens": 2,
            "reasoning_tokens": 2,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 2,
            "prompt_tokens": 0,
            "generated_tokens": 2,
            "phase": "think",
            "continuation_eligible": False,
            "reasoning_tokens": 2,
        },
    }

    content = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["delta"].get("content") == "streamed ")
    assert content["choices"][0]["hipengine"] == {
        "phase": "answer",
        "tokens": {
            "streamed_tokens": 3,
            "delta_tokens": 1,
            "reasoning_tokens": 2,
            "answer_tokens": 1,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 3,
            "prompt_tokens": 0,
            "generated_tokens": 3,
            "phase": "answer",
            "continuation_eligible": False,
            "reasoning_tokens": 2,
            "answer_tokens": 1,
        },
    }

    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"] == "stop")
    assert done["hipengine"]["event"] == "done"
    assert done["choices"][0]["hipengine"] == {
        "phase": "done",
        "finish_details": {"reason": "stop"},
        "tokens": {
            "prompt_tokens": payloads[-1]["usage"]["prompt_tokens"],
            "completion_tokens": payloads[-1]["usage"]["completion_tokens"],
            "total_tokens": payloads[-1]["usage"]["total_tokens"],
            "streamed_tokens": 4,
            "reasoning_tokens": 2,
            "answer_tokens": 2,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 4,
            "prompt_tokens": payloads[-1]["usage"]["prompt_tokens"],
            "generated_tokens": payloads[-1]["usage"]["completion_tokens"],
            "phase": "done",
            "continuation_eligible": False,
            "reasoning_tokens": 2,
            "answer_tokens": 2,
        },
    }
    assert payloads[-1]["hipengine"]["event"] == "usage"
    assert payloads[-1]["hipengine"]["usage"] == payloads[-1]["usage"]


def test_streaming_completion_uses_engine_stream_and_usage() -> None:
    fake = FakeLLM(outputs=["should-not-buffer"], stream_chunks=["alpha", " beta"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in response.text
    payloads = _sse_payloads(response.text)
    assert all("hipengine" not in payload for payload in payloads)
    assert all("hipengine" not in payload["choices"][0] for payload in payloads if payload.get("choices"))
    text_chunks = [payload["choices"][0]["text"] for payload in payloads if payload.get("choices")]
    assert text_chunks == ["alpha", " beta", ""]
    assert payloads[-1]["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    assert fake.stream_calls == [("hello", SamplingParams(max_tokens=2))]


def test_streaming_completion_can_include_hipengine_metadata() -> None:
    fake = FakeLLM(outputs=["should-not-buffer"], stream_chunks=["alpha", " beta"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert [payload["hipengine"]["event"] for payload in payloads] == ["delta", "delta", "done", "usage"]
    assert payloads[0]["choices"][0]["hipengine"] == {
        "phase": "answer",
        "tokens": {
            "streamed_tokens": 1,
            "delta_tokens": 1,
            "answer_tokens": 1,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 1,
            "prompt_tokens": 0,
            "generated_tokens": 1,
            "phase": "answer",
            "continuation_eligible": False,
            "answer_tokens": 1,
        },
    }
    assert isinstance(payloads[0]["hipengine"]["timing"]["elapsed_ms"], float)
    assert payloads[2]["choices"][0]["hipengine"] == {
        "phase": "done",
        "finish_details": {"reason": "stop"},
        "tokens": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "streamed_tokens": 2,
            "answer_tokens": 2,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 2,
            "prompt_tokens": 1,
            "generated_tokens": 2,
            "phase": "done",
            "continuation_eligible": False,
            "answer_tokens": 2,
        },
    }
    assert payloads[-1]["hipengine"]["usage"] == payloads[-1]["usage"]


def test_metrics_prefix_cache_and_generation_batch_cli_env_defaults(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GENERATION_BATCH_WINDOW_MS", raising=False)
    monkeypatch.delenv("HIPENGINE_DEBUG", raising=False)
    monkeypatch.delenv("HIPENGINE_CHAT_DEFAULT_MAX_TOKENS", raising=False)
    monkeypatch.delenv("HIPENGINE_STARTUP_CHAT_SMOKE", raising=False)
    monkeypatch.delenv("HIPENGINE_STARTUP_SCRATCH_PROBE", raising=False)
    monkeypatch.delenv("HIPENGINE_STARTUP_MIN_FREE_MIB", raising=False)
    monkeypatch.delenv("HIPENGINE_REQUEST_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("HIPENGINE_MAX_QUEUED_REQUESTS", raising=False)
    monkeypatch.delenv("HIPENGINE_REPLAY_DIR", raising=False)
    monkeypatch.delenv("HIPENGINE_REPLAY_REDACTION", raising=False)
    default_args = build_parser().parse_args(["--model", "fake-path"])
    assert default_args.generation_batch_window_ms == 0.0
    assert default_args.debug is False
    assert default_args.chat_default_max_tokens == 4096
    assert default_args.startup_chat_smoke is True
    assert default_args.startup_scratch_probe is True
    assert default_args.startup_min_free_mib is None
    assert default_args.request_timeout_ms is None
    assert default_args.max_queued_requests is None
    assert default_args.replay_dir is None
    assert default_args.replay_redaction == "hash"

    monkeypatch.setenv("HIPENGINE_METRICS", "prometheus")
    monkeypatch.setenv("HIPENGINE_PREFIX_CACHE", "radix")
    monkeypatch.setenv("HIPENGINE_GENERATION_BATCH_WINDOW_MS", "3.5")
    monkeypatch.setenv("HIPENGINE_DEBUG", "1")
    monkeypatch.setenv("HIPENGINE_CHAT_DEFAULT_MAX_TOKENS", "auto")
    monkeypatch.setenv("HIPENGINE_STARTUP_CHAT_SMOKE", "0")
    monkeypatch.setenv("HIPENGINE_STARTUP_SCRATCH_PROBE", "0")
    monkeypatch.setenv("HIPENGINE_STARTUP_MIN_FREE_MIB", "512")
    monkeypatch.setenv("HIPENGINE_REQUEST_TIMEOUT_MS", "250.5")
    monkeypatch.setenv("HIPENGINE_MAX_QUEUED_REQUESTS", "7")
    monkeypatch.setenv("HIPENGINE_REPLAY_DIR", "/tmp/hipengine-replay")
    monkeypatch.setenv("HIPENGINE_REPLAY_REDACTION", "none")
    env_args = build_parser().parse_args(["--model", "fake-path"])
    assert env_args.metrics == "prometheus"
    assert env_args.prefix_cache == "radix"
    assert env_args.generation_batch_window_ms == 3.5
    assert env_args.debug is True
    assert env_args.chat_default_max_tokens is None
    assert env_args.startup_chat_smoke is False
    assert env_args.startup_scratch_probe is False
    assert env_args.startup_min_free_mib == 512
    assert env_args.request_timeout_ms == 250.5
    assert env_args.max_queued_requests == 7
    assert env_args.replay_dir == "/tmp/hipengine-replay"
    assert env_args.replay_redaction == "none"

    cli_args = build_parser().parse_args(
        [
            "--model",
            "fake-path",
            "--metrics",
            "off",
            "--prefix-cache",
            "off",
            "--generation-batch-window-ms",
            "0",
            "--request-timeout-ms",
            "123.5",
            "--max-queued-requests",
            "3",
            "--chat-default-max-tokens",
            "123",
            "--replay-dir",
            "/tmp/hipengine-cli-replay",
            "--replay-redaction",
            "hash",
            "--startup-chat-smoke",
            "--startup-scratch-probe",
            "--startup-min-free-mib",
            "256",
            "--no-debug",
        ]
    )
    assert cli_args.metrics == "off"
    assert cli_args.prefix_cache == "off"
    assert cli_args.generation_batch_window_ms == 0.0
    assert cli_args.request_timeout_ms == 123.5
    assert cli_args.max_queued_requests == 3
    assert cli_args.chat_default_max_tokens == 123
    assert cli_args.replay_dir == "/tmp/hipengine-cli-replay"
    assert cli_args.replay_redaction == "hash"
    assert cli_args.startup_chat_smoke is True
    assert cli_args.startup_scratch_probe is True
    assert cli_args.startup_min_free_mib == 256
    assert cli_args.debug is False

    app = create_app(ServerConfig(model="fake-path", eager_load=False, prefix_cache="radix"), llm=FakeLLM())
    assert app.state.hipengine_prefix_cache_mode == "radix"


def test_replay_artifacts_are_default_off(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=FakeLLM(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "secret prompt", "typical_p": 0.9},
    )

    assert response.status_code == 400
    assert not replay_dir.exists()


def test_replay_artifact_redacts_failed_request(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "secret prompt",
            "max_tokens": 1,
            "top_k": 8,
            "logit_bias": {"12": -2.0},
            "top_logprobs": 2,
            "response_format": {"type": "json_object"},
            "seed": 123,
            "typical_p": 0.9,
        },
    )

    assert response.status_code == 400
    artifacts = list(replay_dir.glob("*.json"))
    assert len(artifacts) == 1
    artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
    serialized = json.dumps(artifact, sort_keys=True)
    assert artifact["schema"] == "hipengine.replay.v1"
    assert artifact["redaction"] == {"mode": "hash", "hash": "sha256"}
    assert artifact["request"]["method"] == "POST"
    assert artifact["request"]["path"] == "/v1/completions"
    assert artifact["request"]["json"]["prompt"]["redacted"] == "sha256"
    assert artifact["request"]["prompt_hashes"] == [
        {
            "path": "$.prompt",
            "sha256": artifact["request"]["json"]["prompt"]["sha256"],
            "length": len("secret prompt"),
        }
    ]
    assert artifact["model"]["id"] == "fake-model"
    assert artifact["sampling"]["max_tokens"] == 1
    assert artifact["sampling"]["top_k"] == 8
    assert artifact["sampling"]["logit_bias"] == {"12": -2.0}
    assert artifact["sampling"]["top_logprobs"] == 2
    assert artifact["sampling"]["response_format"] == {"type": "json_object"}
    assert artifact["seeds"] == {"seed": 123, "row_seeds": []}
    assert artifact["token_counts"] == {
        "prompt_tokens": 2,
        "completion_tokens": None,
        "total_tokens": None,
        "available": True,
        "source": "completion_prompt",
        "entries": [{"path": "$.prompt", "token_count": 2}],
    }
    assert artifact["error"]["code"] == "unsupported_parameter"
    assert artifact["error"]["param"] == "top_logprobs"
    assert artifact["error"]["hipengine"]["code"] == "unsupported_parameter"
    assert artifact["capabilities"]["model"]["id"] == "fake-model"
    assert artifact["capabilities"]["sampling"]["speculative_mtp"] == {
        "serving_route": False,
        "sampling_compatible": False,
        "compatibility_guard": "supports_speculative_mtp_sampling",
        "allowed_execution_modes": ["greedy_fast"],
        "incompatible_fields": [
            "temperature",
            "logit_bias",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "suppress_token_ids",
            "min_tokens",
            "stop_token_ids",
            "stop_token_sequences",
            "forced_tokens_pending",
            "post_thinking_forced_tokens_pending",
            "thinking_budget",
            "logprobs",
            "top_logprobs",
        ],
        "incompatible_conditions": {
            "temperature": "temperature > 0",
            "logit_bias": "non-empty logit_bias",
            "repetition_penalty": "repetition_penalty != 1.0",
            "presence_penalty": "presence_penalty != 0.0",
            "frequency_penalty": "frequency_penalty != 0.0",
            "suppress_token_ids": "one or more suppressed token ids",
            "min_tokens": "min_tokens > 0",
            "stop_token_ids": "one or more token stop ids",
            "stop_token_sequences": "one or more multi-token stop sequences",
            "forced_tokens_pending": "one or more forced tokens pending",
            "post_thinking_forced_tokens_pending": "one or more post-thinking forced tokens pending",
            "thinking_budget": "thinking budget soft-close, EOS suppression, or hard-close control",
            "logprobs": "logprobs requested",
            "top_logprobs": "top_logprobs > 0",
        },
        "processed_target_verification": False,
    }
    assert artifact["capabilities"]["sessions"] == {
        "resident_context": True,
        "commit_policy": {
            "supported": True,
            "stateful": False,
            "default": "append_none",
            "modes": ["append_none"],
            "unsupported_stateful_modes": [
                "append_all",
                "append_visible_only",
                "append_prompt_only",
            ],
        },
        "continuations": False,
    }
    assert "secret prompt" not in serialized


def test_replay_artifact_counts_chat_prompt_when_engine_loaded(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "secret chat prompt"}],
            "top_logprobs": 1,
        },
    )

    assert response.status_code == 400
    artifacts = list(replay_dir.glob("*.json"))
    assert len(artifacts) == 1
    artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
    serialized = json.dumps(artifact, sort_keys=True)

    assert artifact["request"]["path"] == "/v1/chat/completions"
    assert artifact["request"]["prompt_hashes"] == [
        {
            "path": "$.messages[0].content",
            "sha256": artifact["request"]["json"]["messages"][0]["content"]["sha256"],
            "length": len("secret chat prompt"),
        }
    ]
    assert artifact["token_counts"]["available"] is True
    assert artifact["token_counts"]["source"] == "chat_prompt"
    assert artifact["token_counts"]["entries"][0]["path"] == "$.messages"
    assert artifact["token_counts"]["entries"][0]["token_count"] == artifact["token_counts"]["prompt_tokens"]
    assert artifact["token_counts"]["prompt_tokens"] > 0
    assert artifact["error"]["param"] == "top_logprobs"
    assert "secret chat prompt" not in serialized


def test_debug_mode_logs_full_request_and_response_payloads(caplog) -> None:
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    fake = FakeLLM(outputs=["debug reply"])
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False, debug=True),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 1},
    )

    assert response.status_code == 200
    assert "DEBUG_PAYLOAD REQUEST POST /v1/completions" in caplog.text
    assert '"prompt":"hello"' in caplog.text
    assert "DEBUG_PAYLOAD RESPONSE POST /v1/completions status=200" in caplog.text
    assert '"text":"debug reply"' in caplog.text


def test_metrics_endpoint_is_opt_in_and_additive() -> None:
    disabled = create_app(ServerConfig(model="fake-path", eager_load=False), llm=FakeLLM())
    assert TestClient(disabled).get("/metrics").status_code == 404

    fake = FakeLLM(outputs=["alpha beta"])
    fake.kv_pool_stats = SimpleNamespace(
        current_bytes=4096,
        high_water_observed_bytes=8192,
        grow_events=2,
        grow_failures=1,
        shrink_events=3,
        free_pages=4,
        refcounted_pages=5,
    )
    fake.graph_bucket_stats = SimpleNamespace(
        entries=6,
        hits=7,
        misses=8,
        replay_hit_rate=0.0,
        miss_reasons={"cache_absent": 5, "shape_changed": 3, "bool_bad": True, "nan_bad": float("nan")},
        kernel_time_histogram_ns={
            "le_10us": 2,
            "le_100us": 4,
            "le_1ms": True,
            "le_10ms": float("inf"),
            "gt_10ms": -1,
            "lt_1us": 9,
        },
    )
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            metrics="prometheus",
            max_queued_requests=3,
        ),
        llm=fake,
    )
    client = TestClient(app)

    before = client.get("/metrics")
    assert before.status_code == 200
    assert _metric_value(before.text, "hipengine_requests_total") == 0
    assert _metric_value(before.text, "hipengine_generation_queue_depth") == 0
    assert _metric_value(before.text, "hipengine_generation_queue_max_depth") == 3
    assert _metric_value(before.text, "hipengine_generation_worker_active") == 0

    for prompt in ["one", "two three"]:
        response = client.post(
            "/v1/completions",
            json={"model": "fake-model", "prompt": prompt, "max_tokens": 2},
        )
        assert response.status_code == 200

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain")
    assert _metric_value(metrics.text, "hipengine_requests_total") == 2
    assert _metric_value(metrics.text, "hipengine_request_completed_total") == 2
    assert _metric_value(metrics.text, "hipengine_request_failed_total") == 0
    assert _metric_value(metrics.text, "hipengine_request_rejected_total") == 0
    assert _metric_value(metrics.text, "hipengine_generation_queue_depth") == 0
    assert _metric_value(metrics.text, "hipengine_generation_queue_max_depth") == 3
    assert _metric_value(metrics.text, "hipengine_generation_worker_active") == 0
    assert _metric_value(metrics.text, "hipengine_prompt_tokens_total") == 3
    assert _metric_value(metrics.text, "hipengine_completion_tokens_total") == 4
    assert _metric_value(metrics.text, "hipengine_kv_pool_current_bytes") == 4096
    assert _metric_value(metrics.text, "hipengine_kv_pool_grow_events_total") == 2
    assert _metric_value(metrics.text, "hipengine_kv_pool_grow_failures_total") == 1
    assert _metric_value(metrics.text, "hipengine_kv_pool_shrink_events_total") == 3
    assert _metric_value(metrics.text, "hipengine_kv_pool_free_pages") == 4
    assert _metric_value(metrics.text, "hipengine_kv_pool_refcounted_pages") == 5
    assert _metric_value(metrics.text, "hipengine_graph_bucket_entries") == 6
    assert _metric_value(metrics.text, "hipengine_graph_bucket_hits_total") == 7
    assert _metric_value(metrics.text, "hipengine_graph_bucket_misses_total") == 8
    assert _metric_value(metrics.text, "hipengine_graph_bucket_replay_hit_rate") == 7 / 15
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_miss_reason_total", reason="cache_absent") == 5
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_miss_reason_total", reason="shape_changed") == 3
    assert 'hipengine_graph_bucket_miss_reason_total{reason="bool_bad"}' not in metrics.text
    assert 'hipengine_graph_bucket_miss_reason_total{reason="nan_bad"}' not in metrics.text
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_kernel_time_bucket_total", bucket="le_10us") == 2
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_kernel_time_bucket_total", bucket="le_100us") == 4
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_kernel_time_bucket_total", bucket="le_1ms") == 0
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_kernel_time_bucket_total", bucket="le_10ms") == 0
    assert _labeled_metric_value(metrics.text, "hipengine_graph_bucket_kernel_time_bucket_total", bucket="gt_10ms") == 0
    for bucket in GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS:
        assert f'hipengine_graph_bucket_kernel_time_bucket_total{{bucket="{bucket}"}}' in metrics.text
    assert 'hipengine_graph_bucket_kernel_time_bucket_total{bucket="lt_1us"}' not in metrics.text


def test_metrics_endpoint_filters_malformed_graph_bucket_scalars() -> None:
    fake = FakeLLM()
    fake.graph_bucket_stats = SimpleNamespace(
        entries=True,
        hits=float("nan"),
        misses=float("inf"),
        replay_hit_rate=1.0,
        miss_reasons={},
        kernel_time_histogram_ns={},
    )
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False, metrics="prometheus"),
        llm=fake,
    )
    client = TestClient(app)

    malformed = client.get("/metrics")

    assert malformed.status_code == 200
    assert _metric_value(malformed.text, "hipengine_graph_bucket_entries") == 0
    assert _metric_value(malformed.text, "hipengine_graph_bucket_hits_total") == 0
    assert _metric_value(malformed.text, "hipengine_graph_bucket_misses_total") == 0
    assert _metric_value(malformed.text, "hipengine_graph_bucket_replay_hit_rate") == 0

    fake.graph_bucket_stats = SimpleNamespace(
        entries=-1,
        hits=3,
        misses=-4,
        replay_hit_rate=0.0,
        miss_reasons={},
        kernel_time_histogram_ns={},
    )
    partially_valid = client.get("/metrics")

    assert partially_valid.status_code == 200
    assert _metric_value(partially_valid.text, "hipengine_graph_bucket_entries") == 0
    assert _metric_value(partially_valid.text, "hipengine_graph_bucket_hits_total") == 3
    assert _metric_value(partially_valid.text, "hipengine_graph_bucket_misses_total") == 0
    assert _metric_value(partially_valid.text, "hipengine_graph_bucket_replay_hit_rate") == 1


def test_metrics_endpoint_filters_malformed_kv_pool_scalars() -> None:
    fake = FakeLLM()
    fake.kv_pool_stats = SimpleNamespace(
        current_bytes=True,
        high_water_observed_bytes=float("nan"),
        grow_events=float("inf"),
        grow_failures=-1,
        shrink_events="bad",
        free_pages=3,
        refcounted_pages=-4,
    )
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False, metrics="prometheus"),
        llm=fake,
    )
    client = TestClient(app)

    metrics = client.get("/metrics")

    assert metrics.status_code == 200
    assert _metric_value(metrics.text, "hipengine_kv_pool_current_bytes") == 0
    assert _metric_value(metrics.text, "hipengine_kv_pool_high_water_observed_bytes") == 0
    assert _metric_value(metrics.text, "hipengine_kv_pool_grow_events_total") == 0
    assert _metric_value(metrics.text, "hipengine_kv_pool_grow_failures_total") == 0
    assert _metric_value(metrics.text, "hipengine_kv_pool_shrink_events_total") == 0
    assert _metric_value(metrics.text, "hipengine_kv_pool_free_pages") == 3
    assert _metric_value(metrics.text, "hipengine_kv_pool_refcounted_pages") == 0


def test_streaming_chat_completion_lowers_n_to_seeded_rows() -> None:
    fake = FakeLLM(outputs=["alpha", "beta"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "n": 2,
            "seed": 5,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    indices = [payload["choices"][0]["index"] for payload in payloads]
    assert 0 in indices and 1 in indices
    assert "data: [DONE]" in response.text
    assert fake.calls[0][0] == (fake.calls[0][0][0], fake.calls[0][0][0])
    assert len(fake.calls[0][1].row_seeds) == 2
    assert len(set(fake.calls[0][1].row_seeds)) == 2


def test_server_rejects_requests_beyond_preallocated_context() -> None:
    fake = FakeLLM()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=5,
        ),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "one two three four", "max_tokens": 2},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert error["hipengine"] == {
        "code": "context_overflow",
        "status_code": 400,
        "legacy_code": "context_length_exceeded",
        "retryable": False,
    }
    assert error["fit_context"] == {
        "prompt_tokens": 4,
        "max_context_tokens": 5,
        "effective_max_tokens": 2,
        "required_context_tokens": 7,
        "fits": False,
        "clear_policy": "reject",
        "would_truncate": False,
        "would_drop": [],
    }
    fit = client.post(
        "/v1/hipengine/fit_context",
        json={"text": "one two three four", "max_tokens": 2},
    )
    assert fit.status_code == 200
    fit_body = fit.json()
    for key, value in error["fit_context"].items():
        assert fit_body[key] == value
    assert fake.calls == []


def test_server_rejects_request_kv_policy_mismatch() -> None:
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            kv_storage="int8_per_token_head",
        ),
        llm=FakeLLM(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "kv_storage": "bf16"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_kv_policy"


def test_server_rejects_wrong_model_and_unsupported_options(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="uvicorn.error")
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=FakeLLM())
    client = TestClient(app)

    wrong_model = client.post(
        "/v1/completions",
        json={"model": "other", "prompt": "hello"},
    )
    assert wrong_model.status_code == 404
    assert wrong_model.json()["error"]["code"] == "model_not_found"
    assert wrong_model.json()["error"]["hipengine"] == {
        "code": "model_unavailable",
        "status_code": 404,
        "legacy_code": "model_not_found",
        "retryable": False,
    }

    schema_violation = client.post(
        "/v1/completions",
        json={"model": "fake-model", "max_tokens": 1},
    )
    assert schema_violation.status_code == 422
    assert schema_violation.json()["error"]["code"] == "validation_error"
    assert schema_violation.json()["error"]["param"] == "prompt"
    assert schema_violation.json()["error"]["hipengine"] == {
        "code": "schema_violation",
        "status_code": 422,
        "legacy_code": "validation_error",
        "retryable": False,
    }

    unsupported_chat_top_logprobs = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "top_logprobs": 1,
        },
    )
    assert unsupported_chat_top_logprobs.status_code == 400
    assert unsupported_chat_top_logprobs.json()["error"]["param"] == "top_logprobs"

    unsupported_extra = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "typical_p": 0.9},
    )
    assert unsupported_extra.status_code == 400
    assert unsupported_extra.json()["error"]["code"] == "unsupported_parameter"
    assert unsupported_extra.json()["error"]["hipengine"] == {
        "code": "unsupported_parameter",
        "status_code": 400,
        "retryable": False,
    }
    assert unsupported_extra.json()["error"]["param"] == "typical_p"
    assert "REQUEST_FAILED: POST /v1/completions status=404 code=model_not_found" in caplog.text
    assert "REQUEST_FAILED: POST /v1/completions status=400 code=unsupported_parameter" in caplog.text
    assert "param=typical_p" in caplog.text


@pytest.mark.parametrize(
    ("endpoint", "payload", "param"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "continuation_id": "gen_123",
            },
            "continuation_id",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "session": {"commit": "append_visible_only"},
            },
            "session.commit",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "session": {"id": "session_123"},
            },
            "session.id",
        ),
    ],
)
def test_server_rejects_known_unsupported_agentic_fields(endpoint, payload, param) -> None:
    fake = FakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(endpoint, json=payload)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert response.json()["error"]["param"] == param
    assert fake.calls == []


def test_completions_endpoint_lowers_n_to_distinct_seeded_rows() -> None:
    fake = FakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": ["one", "two"], "max_tokens": 1, "n": 2, "seed": 123},
    )

    assert response.status_code == 200
    body = response.json()
    assert [choice["text"] for choice in body["choices"]] == [
        "generated:one",
        "generated:one",
        "generated:two",
        "generated:two",
    ]
    assert [choice["index"] for choice in body["choices"]] == [0, 1, 2, 3]
    assert len({choice["request_id"] for choice in body["choices"]}) == 4
    assert fake.calls[0][0] == ("one", "one", "two", "two")
    assert len(fake.calls[0][1].row_seeds) == 4
    assert len(set(fake.calls[0][1].row_seeds)) == 4


def test_chat_endpoint_lowers_n_to_distinct_seeded_rows() -> None:
    fake = FakeLLM(outputs=["first", "second"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [{"role": "user", "content": "hello"}], "n": 2, "seed": 9},
    )

    assert response.status_code == 200
    body = response.json()
    assert [choice["index"] for choice in body["choices"]] == [0, 1]
    assert [choice["message"]["content"] for choice in body["choices"]] == ["first", "second"]
    assert len({choice["request_id"] for choice in body["choices"]}) == 2
    assert fake.calls[0][0] == (fake.calls[0][0][0], fake.calls[0][0][0])
    assert len(fake.calls[0][1].row_seeds) == 2
    assert len(set(fake.calls[0][1].row_seeds)) == 2


def test_render_chat_prompt_accepts_plain_message_mappings() -> None:
    assert render_chat_prompt([{"role": "user", "content": "hello"}]) == (
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    )


def test_chat_endpoint_rejects_non_text_content_parts() -> None:
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=FakeLLM())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_content_type"


def _metric_value(text: str, name: str) -> float:
    prefix = f"{name} "
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line.removeprefix(prefix))
    raise AssertionError(f"metric {name} not found in:\n{text}")


def _labeled_metric_value(text: str, name: str, **labels: str) -> int:
    encoded_labels = ",".join(f'{key}="{value}"' for key, value in sorted(labels.items()))
    prefix = f"{name}{{{encoded_labels}}} "
    for line in text.splitlines():
        if line.startswith(prefix):
            return int(float(line.removeprefix(prefix)))
    raise AssertionError(f"metric {name} with labels {labels} not found in:\n{text}")


def _sse_payloads(text: str) -> list[dict]:
    payloads = []
    for line in text.splitlines():
        if line == "data: [DONE]" or not line.startswith("data: "):
            continue
        payloads.append(json.loads(line.removeprefix("data: ")))
    return payloads
