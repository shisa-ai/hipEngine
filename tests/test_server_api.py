from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

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
    GenerationStreamChunk,
    GenerationTelemetry,
    TokenLogprob,
)
from hipengine.generation.sampling import (
    NATIVE_GPU_SAMPLER_UNSUPPORTED_CAPABILITIES,
    SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS,
    SPECULATIVE_MTP_INCOMPATIBLE_FIELDS,
)
from hipengine.server import ServerConfig, create_app, render_chat_prompt
from hipengine.server.__main__ import build_parser
from hipengine.server.api import (
    ChatCompletionRequest,
    CompletionRequest,
    OpenAIHTTPError,
    _AGENTIC_REPLAY_FAILURE_REASONS,
    _STRUCTURED_OUTPUT_RESULT_VALIDATION_FAILURE_REASONS,
    _TOOL_RESULT_VALIDATION_FAILURE_REASONS,
    _await_with_request_control,
    _backend_scheduler_token_chunks,
    _chat_session_message_copy,
    _coerce_generation_output,
    _GenerationBatcher,
    _QueuedBatchResult,
    _request_control,
    _startup_memory_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _api_error_taxonomy_table() -> dict[str, dict[str, Any]]:
    lines = (REPO_ROOT / "docs" / "API.md").read_text(encoding="utf-8").splitlines()
    start = lines.index("| Code | Status | Retry | Current emission |") + 2
    table: dict[str, dict[str, Any]] = {}
    for line in lines[start:]:
        if not line.startswith("| `"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        code = cells[0].strip("`")
        table[code] = {
            "status_code": int(cells[1]),
            "retryable": cells[2] == "yes",
            "current_emission": cells[3],
        }
    return table


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


class SchedulerChunkRowsFakeLLM(FakeLLM):
    def __init__(
        self,
        raw_outputs: tuple[str, ...],
        chunk_rows: tuple[tuple[str, ...], ...],
        *,
        execution_path: str = "scheduler_tool_call_chunks",
    ) -> None:
        super().__init__()
        self.raw_outputs = raw_outputs
        self.chunk_rows = chunk_rows
        self.execution_path = str(execution_path)

    def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
        prompt_tuple = tuple(str(prompt) for prompt in prompts)
        self.calls.append((prompt_tuple, sampling_params))
        self.last_batch_generation = {
            "scheduler_token_chunks": [
                {
                    "request_id": request_id,
                    "token_index": token_index,
                    "token_id": 600 + request_id * 10 + token_index,
                    "finished": token_index == len(row) - 1,
                    "chunk": {
                        "text": text,
                        "telemetry": GenerationTelemetry.from_decode_counts(
                            prompt_tokens=1,
                            generated_tokens=token_index + 1,
                            row_index=request_id,
                            request_id=str(request_id),
                            phase="answer",
                            sampler_mode="greedy_fast",
                            execution_path=self.execution_path,
                        ).to_json_dict(),
                    },
                }
                for request_id, row in enumerate(self.chunk_rows)
                for token_index, text in enumerate(row)
            ]
        }
        return [
            GenerationOutput(
                text=self.raw_outputs[index],
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=len(self.chunk_rows[index]),
                    row_index=index,
                    phase="done",
                    sampler_mode="greedy_fast",
                ),
            )
            for index, _prompt in enumerate(prompt_tuple)
        ]


class ScratchProbeFailureFakeLLM(FakeLLM):
    def prepare_request_scratch(
        self,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int = 0,
        sampling_params: SamplingParams | None = None,
        max_batch_size: int = 1,
        release_after_probe: bool = True,
    ) -> dict[str, Any]:
        super().prepare_request_scratch(
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
            max_batch_size=max_batch_size,
            release_after_probe=release_after_probe,
        )
        raise RuntimeError("scratch failed near private startup prompt")


class SequentialFakeLLM(FakeLLM):
    def __init__(self, outputs: list[str | GenerationOutput]) -> None:
        super().__init__()
        self.sequence = list(outputs)

    def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
        prompts = tuple(prompts)
        self.calls.append((prompts, sampling_params))
        outputs: list[GenerationOutput] = []
        for _prompt in prompts:
            if not self.sequence:
                raise AssertionError("no fake generation output left")
            output = self.sequence.pop(0)
            if isinstance(output, GenerationOutput):
                outputs.append(output)
            else:
                outputs.append(GenerationOutput(text=str(output)))
        return outputs


class NoTokenizerFakeLLM:
    def generate(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
        return [GenerationOutput(text=str(prompt)) for prompt in prompts]


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


class BackendErrorThenSequentialFakeLLM(SequentialFakeLLM):
    def __init__(self, error: str, outputs: list[str | GenerationOutput]) -> None:
        super().__init__(outputs)
        self.error = str(error)
        self.raised = False

    def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
        if not self.raised:
            self.raised = True
            prompts = tuple(prompts)
            self.calls.append((prompts, sampling_params))
            if self.error == "deadline":
                assert sampling_params.deadline_at is not None
                raise GenerationDeadlineExceeded(deadline_at=sampling_params.deadline_at)
            if self.error == "cancelled":
                assert sampling_params.cancellation_token is not None
                sampling_params.cancellation_token.cancel()
                raise GenerationCancelled(sampling_params.cancellation_token.finish_details)
            raise AssertionError(f"unsupported fake backend error {self.error!r}")
        return super().generate_detailed(prompts, sampling_params)


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


def _stateless_finish_details(reason: str, **extra: Any) -> dict[str, Any]:
    return {"reason": reason, **extra, "cache_action": "append_none"}


def _assert_openai_tool_call_shape(
    tool_call: Mapping[str, Any],
    *,
    name: str,
    arguments: Mapping[str, Any],
) -> None:
    assert set(tool_call) == {"id", "type", "function"}
    assert isinstance(tool_call["id"], str)
    assert tool_call["id"].startswith("call_")
    assert tool_call["type"] == "function"
    function = tool_call["function"]
    assert isinstance(function, Mapping)
    assert set(function) == {"name", "arguments"}
    assert function["name"] == name
    assert isinstance(function["arguments"], str)
    assert json.loads(function["arguments"]) == dict(arguments)


def _assert_openai_stream_tool_call_delta_shape(
    choice: Mapping[str, Any],
    *,
    name: str,
    arguments: Mapping[str, Any],
    index: int = 0,
    tool_index: int = 0,
) -> None:
    assert choice["index"] == index
    assert choice["finish_reason"] is None
    assert set(choice["delta"]) == {"tool_calls"}
    tool_calls = choice["delta"]["tool_calls"]
    assert isinstance(tool_calls, list)
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert set(tool_call) == {"index", "id", "type", "function"}
    assert tool_call["index"] == tool_index
    _assert_openai_tool_call_shape(
        {
            "id": tool_call["id"],
            "type": tool_call["type"],
            "function": tool_call["function"],
        },
        name=name,
        arguments=arguments,
    )


def _routing_metadata(
    *,
    requested_model: str = "fake-model",
    served_model: str = "fake-model",
    loaded_model_count: int = 1,
) -> dict[str, Any]:
    return {
        "requested_model": requested_model,
        "served_model": served_model,
        "fallback_used": False,
        "policy": "single_model_exact",
        "loaded_model_count": loaded_model_count,
        "multiple_models": False,
    }


def _fake_kv_pool_stats() -> SimpleNamespace:
    return SimpleNamespace(
        current_bytes=4096,
        high_water_observed_bytes=8192,
        grow_events=2,
        grow_failures=1,
        shrink_events=3,
        free_pages=4,
        refcounted_pages=5,
    )


def _fake_kv_pool_metadata() -> dict[str, float]:
    return {
        "current_bytes": 4096.0,
        "high_water_observed_bytes": 8192.0,
        "grow_events": 2.0,
        "grow_failures": 1.0,
        "shrink_events": 3.0,
        "free_pages": 4.0,
        "refcounted_pages": 5.0,
    }


def _continuation_capability() -> dict[str, Any]:
    return {
        "supported": True,
        "stateful": False,
        "resident_state_reuse": False,
        "single_use": True,
        "ttl_seconds": 900,
        "scoped_to": ["served_model", "endpoint", "tokenizer", "auth_principal", "session_id"],
        "supported_endpoints": ["completions", "chat_completions"],
        "supported_finishes": ["length"],
        "supported_streaming": False,
        "supported_sampling": "deterministic_buffered_only",
        "ineligible_when": [
            "non_length_finish",
            "length_phase_not_answer_or_structured",
            "stream",
            "n_not_1",
            "logprobs",
            "completion_echo",
            "non_deterministic_sampling",
            "logit_processors",
            "stop",
            "chat_tools",
            "thinking_budget_controls",
            "session_id_without_commit",
        ],
        "unsupported_resume_fields": [
            "prompt",
            "messages",
            "stream",
            "n",
            "logprobs",
            "echo",
            "temperature",
            "logit_bias",
            "suppress_token_ids",
            "min_tokens",
            "ignore_eos",
            "stop",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "response_format",
            "guided_json",
            "guided_regex",
            "guided_choice",
            "guided_patch",
            "guided_diff",
            "reasoning_effort",
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
    }


def _session_commit_policy_capability() -> dict[str, Any]:
    return {
        "supported": True,
        "stateful": True,
        "resident_state_reuse": False,
        "storage": "app_local_transcript",
        "default": "append_none",
        "stateful_default": "append_visible_only",
        "modes": [
            "append_none",
            "append_prompt_only",
            "append_visible_only",
            "append_all",
        ],
        "supported_endpoints": ["chat_completions"],
        "supported_streaming": False,
        "resident_kv_commit": False,
        "visible_only_reprefill": False,
        "visible_only_replay": "rerender_app_local_transcript",
        "downgrade_visible_only_on": [
            "cancelled",
            "deadline_exceeded",
            "invalid_tool_call",
            "length",
            "schema_violation",
            "tool_required_not_satisfied",
        ],
        "context_overflow_policy": {
            "field": "session.context_overflow_policy",
            "default": "reject",
            "modes": ["reject", "auto_clear_transient", "new_session", "truncate_oldest_visible"],
            "aliases": {"fail": "reject"},
            "auto_clear_transient": {
                "scope": "transient_session_segments",
                "drops_committed_visible_turns": False,
                "drops_request_content": False,
                "current_transient_segment_count": 0,
                "metadata": ["clear_policy", "would_clear_transient", "transient_message_count"],
            },
            "new_session": {
                "scope": "app_local_chat_transcript_prefix",
                "drops_request_content": False,
                "requires_request_only_fit": True,
                "metadata": ["clear_policy", "would_reset_session", "would_drop", "kept_segments"],
            },
            "truncate_oldest_visible": {
                "scope": "app_local_chat_transcript_prefix",
                "drops_request_content": False,
                "requires_valid_rendered_suffix": True,
                "commit": "replace_stored_prefix_on_success",
                "metadata": ["clear_policy", "would_truncate", "would_drop", "kept_segments"],
            },
        },
    }


def _session_metadata_capability(max_active: int | None = None) -> dict[str, Any]:
    return {
        "supported": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "includes_transcript": False,
        "transcript_message_copy": "json_deep_copy",
        "max_active": max_active,
        "list_endpoint": "/v1/hipengine/sessions",
        "delete_endpoint": "/v1/hipengine/sessions/{session_id}",
        "fork_endpoint": "/v1/hipengine/sessions/{session_id}/fork",
        "fork_resident_state_reuse": False,
        "fork_deep_copies_transcript": True,
        "rollback_endpoint": "/v1/hipengine/sessions/{session_id}/rollback",
        "rollback_target": "message_count",
        "rollback_resident_state_reuse": False,
        "rollback_deep_copies_retained_transcript": True,
        "snapshot_schema": "hipengine.chat_session_snapshot.v1",
        "snapshot_export_endpoint": "/v1/hipengine/sessions/{session_id}/snapshot",
        "snapshot_restore_endpoint": "/v1/hipengine/sessions/{session_id}/snapshot",
        "snapshot_includes_transcript": True,
        "snapshot_resident_state_reuse": False,
        "snapshot_export_deep_copies_transcript": True,
        "snapshot_includes_tokenizer_metadata": True,
        "snapshot_tokenizer_validation": "when_model_loaded",
    }


def _parallelism_capability() -> dict[str, Any]:
    return {
        "tensor_parallel": {
            "enabled": False,
            "topology": {
                "mode": "single_process",
                "world_size": 1,
                "local_world_size": 1,
                "rank": 0,
                "local_rank": 0,
            },
            "collectives": {
                "available": False,
                "backend": None,
            },
            "unsupported_features": [
                "world_size_gt_1",
                "weight_sharding",
                "kv_cache_sharding",
                "collective_reduce_scatter_all_gather",
                "multi_gpu_graph_capture",
                "cross_rank_session_snapshots",
            ],
        },
    }


def test_tensor_parallel_design_doc_matches_capability_manifest() -> None:
    doc = (REPO_ROOT / "docs" / "TENSOR_PARALLEL.md").read_text(encoding="utf-8")
    tensor_parallel = _parallelism_capability()["tensor_parallel"]

    assert "`parallelism.tensor_parallel.enabled=false`" in doc
    assert '`topology.mode="single_process"`' in doc
    assert "`world_size=1`" in doc
    assert "Rank 0 owns sampling." in doc
    assert "Phase 1 replicates KV cache on every rank." in doc
    assert "No TP code may affect the default single-GPU path without hardware validation" in doc
    for feature_id in tensor_parallel["unsupported_features"]:
        assert f"`{feature_id}`" in doc


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
        "capabilities": {
            "completions": True,
            "chat_completions": True,
            "streaming": True,
            "tools": True,
            "reasoning_controls": True,
            "structured_outputs": True,
            "continuations": True,
            "sessions": True,
            "grammars": False,
            "speculative_mtp": False,
            "tensor_parallel": False,
            "multiple_models": False,
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
    assert model["hipengine"]["capabilities"]["chat_completions"] is True
    assert model["hipengine"]["capabilities"]["tools"] is True
    assert model["hipengine"]["capabilities"]["multiple_models"] is False
    assert model["hipengine"]["routing"] == {"loaded_model_count": 0, "multiple_models": False}
    assert model["hipengine"]["kv_capacity"]["estimate"] is None


def test_agentic_replay_failure_reasons_match_capability_contract() -> None:
    advertised = frozenset(
        (
            *_TOOL_RESULT_VALIDATION_FAILURE_REASONS,
            *_STRUCTURED_OUTPUT_RESULT_VALIDATION_FAILURE_REASONS,
        )
    )
    assert _AGENTIC_REPLAY_FAILURE_REASONS == advertised


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
            max_queued_requests=5,
            max_active_requests=2,
            max_chat_sessions=3,
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
        "overflow_policy_field": "session.context_overflow_policy",
        "default_overflow_policy": "reject",
        "overflow_policies": ["reject", "auto_clear_transient", "new_session", "truncate_oldest_visible"],
    }
    assert body["tokenizer"]["tokenize"] is True
    assert body["tokenizer"]["detokenize"] is True
    assert body["tokenizer"]["count_tokens"] is True
    assert body["features"]["stream_options"] == {"include_usage": True, "include_hipengine": True}
    assert body["features"]["stream_metadata"] == {
        "metadata_version": 1,
        "events": ["role", "delta", "tool_call", "done", "usage", "error"],
        "timing": [
            "elapsed_ms",
            "ttft_ms",
            "decode_elapsed_ms",
            "decode_tokens_per_second",
            "backend_*",
        ],
        "server_wall_timing": True,
        "backend_prefill_timing": "GenerationTelemetry.timing_when_available",
        "choice_phase": True,
        "choice_finish_details": True,
        "choice_token_accounting": True,
        "choice_token_accounting_scopes": ["live_delta", "buffered_delta", "final_choice"],
        "choice_decode_state": True,
        "choice_decode_state_scopes": ["live_delta", "buffered_delta", "final_choice"],
        "backend_telemetry_scopes": [
            "live_chunk",
            "buffered_delta_safe_decode_state",
            "buffered_done",
        ],
        "live_many_chunks": {
            "available": False,
            "source": "engine.stream_many_detailed",
            "capability": "engine.supports_stream_many",
            "requires_row_index": "GenerationTelemetry.decode_state.row_index",
            "public_surfaces": [
                "chat_answer_delta",
                "chat_reasoning_delta",
            ],
            "safe_request_shape": {
                "chat_n_gt_1": True,
                "tools": False,
                "structured_outputs": False,
                "logprobs": False,
                "stop": False,
                "continuation": False,
            },
            "fallback": "buffered_scheduler_chunks_or_buffered_choice",
        },
        "buffered_scheduler_chunks": {
            "source": "engine_or_wrapped_generator.last_batch_generation.scheduler_token_chunks",
            "requires_single_http_request_batch": True,
            "text_must_reconstruct_public_choice": True,
            "public_surfaces": [
                "completion_delta",
                "chat_answer_delta",
                "chat_reasoning_delta_without_logprobs",
                "chat_reasoning_delta_with_private_logprobs",
                "chat_content_logprob_delta",
                "chat_structured_delta_validated",
                "chat_tool_argument_delta_validated",
            ],
            "fallback_surfaces": [
                "coalesced_multi_request_batch",
                "raw_text_mismatch",
                "invalid_tool_call",
                "unmappable_tool_arguments",
                "structured_validation_failure",
                "unmappable_logprobs",
            ],
            "fallback_diagnostics": [
                "choices[].hipengine.withheld_scheduler_tool_chunks",
                "choices[].hipengine.withheld_scheduler_logprob_chunks",
            ],
        },
        "routing": "stream_options.include_hipengine",
        "kv_pool": "done_and_usage_events_when_engine_exposes_kv_pool_stats",
    }
    assert body["features"]["choice_telemetry"] == {
        "non_streaming": True,
        "streaming": "stream_options.include_hipengine",
        "decode_state": True,
        "decode_state_fields": [
            "row_index",
            "step_index",
            "prompt_tokens",
            "generated_tokens",
            "phase",
            "continuation_eligible",
            "request_id",
            "reasoning_tokens",
            "answer_tokens",
            "tool_call_tokens",
            "structured_tokens",
            "stop_suffix_state",
            "forced_tokens_pending",
            "forced_token_id",
            "forced_token_reason",
            "forced_tokens_remaining",
            "post_thinking_forced_tokens_pending",
            "post_thinking_forced_token_reason",
            "force_sequence_completion_token_sequences",
            "force_sequence_completion_reason",
            "active_processors",
            "sampler_fast_path_blockers",
            "sampler_fallback_reason",
            "budget_pressure",
            "sampler_mode",
            "full_vocab_logits_d2h",
            "logits_d2h_bytes",
            "execution_path",
            "native_compact_prefill",
            "native_caware_decode",
            "serial_decode_fallback",
            "native_sampler_rows",
        ],
        "timing": "backend_generation_telemetry_when_available",
        "usage": "backend_generation_telemetry_when_available",
        "source": "backend_generation_telemetry_when_available",
    }
    assert body["features"]["structured_outputs"] == {
        "response_format": True,
        "json_object": True,
        "json_schema": True,
        "guided_json": True,
        "guided_json_modes": ["json_object", "json_schema"],
        "guided_regex": True,
        "guided_regex_match": "fullmatch_after_strip",
        "guided_choice": True,
        "guided_patch": True,
        "guided_diff": True,
        "guided_patch_formats": ["unified_diff"],
        "guided_diff_formats": ["unified_diff"],
        "guided_patch_fenced_policies": ["optional", "required", "forbidden"],
        "guided_diff_fenced_policies": ["optional", "required", "forbidden"],
        "guided_patch_default_fenced_policy": "optional",
        "guided_diff_default_fenced_policy": "optional",
        "guided_patch_fence_labels": ["diff", "patch"],
        "guided_diff_fence_labels": ["diff", "patch"],
        "strict_decoding": False,
        "strict_result_validation": True,
        "decode_time_close_forcing": "host_json_object_parse_validated_suffix",
        "length_finish_structural_validation": "root_object_json_prefix",
        "result_validation_failure_reasons": ["schema_violation"],
        "schema_validation": "json_schema_subset",
        "schema_subset": [
            "type",
            "enum",
            "const",
            "references.local_ref",
            "references.$defs",
            "references.definitions",
            "composition.allOf",
            "composition.anyOf",
            "composition.oneOf",
            "composition.not",
            "conditional.if",
            "conditional.then",
            "conditional.else",
            "object.properties",
            "object.patternProperties",
            "object.propertyNames",
            "object.required",
            "object.dependentRequired",
            "object.dependentSchemas",
            "object.additionalProperties=false",
            "object.additionalProperties=schema",
            "object.minProperties",
            "object.maxProperties",
            "array.items",
            "array.contains",
            "array.minItems",
            "array.maxItems",
            "array.minContains",
            "array.maxContains",
            "array.uniqueItems",
            "string.minLength",
            "string.maxLength",
            "string.pattern",
            "number.minimum",
            "number.maximum",
            "number.exclusiveMinimum",
            "number.exclusiveMaximum",
            "number.multipleOf",
        ],
        "unsupported_schema_keywords_rejected": True,
        "annotation_keywords_ignored": [
            "title",
            "description",
            "default",
            "examples",
            "deprecated",
            "readOnly",
            "writeOnly",
            "format",
        ],
    }
    assert body["features"]["grammars"] == {
        "enabled": False,
        "strict_decoding": False,
        "supported": [],
        "unsupported_fields": [
            "grammar",
            "guided_grammar",
            "guided_decoding_backend",
        ],
        "result_validation_only": [
            "json_object",
            "json_schema",
            "guided_json",
            "guided_regex",
            "guided_choice",
            "guided_patch",
            "guided_diff",
        ],
    }
    assert body["features"]["token_diagnostics"] == {
        "tokenize": True,
        "detokenize": True,
        "count_tokens": True,
        "fit_context": True,
        "session_aware_chat": True,
    }
    assert body["features"]["tools"] == {
        "enabled": True,
        "strict_decoding": False,
        "strict_result_validation": True,
        "result_validation_failure_reasons": [
            "invalid_tool_call",
            "tool_required_not_satisfied",
            "schema_violation",
        ],
        "invalid_tool_call_error_mode_field": "invalid_tool_call_error_mode",
        "invalid_tool_call_error_modes": ["finish_details", "hard_error"],
        "default_invalid_tool_call_error_mode": "finish_details",
        "hard_error_surfaces": ["http", "sse"],
        "schema_validation": "function_strict",
        "schema_subset": [
            "type",
            "enum",
            "const",
            "references.local_ref",
            "references.$defs",
            "references.definitions",
            "composition.allOf",
            "composition.anyOf",
            "composition.oneOf",
            "composition.not",
            "conditional.if",
            "conditional.then",
            "conditional.else",
            "object.properties",
            "object.patternProperties",
            "object.propertyNames",
            "object.required",
            "object.dependentRequired",
            "object.dependentSchemas",
            "object.additionalProperties=false",
            "object.additionalProperties=schema",
            "object.minProperties",
            "object.maxProperties",
            "array.items",
            "array.contains",
            "array.minItems",
            "array.maxItems",
            "array.minContains",
            "array.maxContains",
            "array.uniqueItems",
            "string.minLength",
            "string.maxLength",
            "string.pattern",
            "number.minimum",
            "number.maximum",
            "number.exclusiveMinimum",
            "number.exclusiveMaximum",
            "number.multipleOf",
        ],
        "unsupported_schema_keywords_rejected": True,
        "annotation_keywords_ignored": [
            "title",
            "description",
            "default",
            "examples",
            "deprecated",
            "readOnly",
            "writeOnly",
            "format",
        ],
        "format": "qwen_tool_call_json",
        "compatibility_parser_repairs": ["duplicated_tool_call_start"],
        "malformed_json_compatibility": "invalid_tool_call_when_tools_enabled",
        "strict_malformed_blocks_rejected": True,
        "declared_tool_name_validation": True,
        "transcript_validation": {
            "role_specific_fields": True,
            "assistant_tool_call_ids_unique": True,
            "tool_results_reference_prior_call_ids": True,
            "tool_results_must_resolve_pending_calls_before_non_tool_messages": True,
            "allows_pending_tool_calls_at_transcript_end": True,
            "applies_to_session_snapshots": True,
        },
        "parallel_tool_calls_requires_opt_in": True,
        "parallel_tool_calls": True,
        "streaming_argument_chunks": True,
        "streaming_argument_chunk_chars": 128,
        "no_tool_start_suppression": True,
        "required_tool_start_forcing": True,
        "required_tool_start_forcing_scope": "initial_or_after_tokenized_thinking_close",
        "specific_tool_name_prefix_forcing": True,
        "tool_call_close_repair": True,
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
            "thinking.allow_unbounded",
            "reasoning.allow_unbounded",
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
    assert body["features"]["logprobs"] == {
        "completions": True,
        "chat": True,
        "top_logprobs_max": 20,
        "streaming": "buffered",
        "live_chunk_metadata": False,
        "live_chunk_metadata_capability": "engine.supports_stream_logprobs",
        "chat_reasoning_private_stream_metadata": "choices[].hipengine.reasoning_logprobs",
        "requires_backend_token_metadata": True,
        "omission_reasons": ["backend_omitted_logprob", "prompt_logprob_unavailable"],
        "missing_backend_metadata_error": {
            "code": "unsupported_feature",
            "status_code": 501,
            "param": "logprobs",
        },
    }
    assert body["features"]["request_timeouts"] == {
        "timeout_ms": True,
        "default_timeout_ms": 250.0,
        "client_disconnect": True,
        "cooperative_backend_deadline": True,
        "cooperative_backend_cancel": True,
        "preemptive_decode_cancel": False,
    }
    assert body["admission"] == {
        "queue": {
            "max_queued_requests": 5,
            "retry_after_seconds": 1,
            "rejects_when_full": True,
        },
        "active_requests": {
            "max_active_requests": 2,
            "limits_backend_batch_width": True,
        },
        "chat_sessions": {
            "max_active": 3,
            "rejects_new_sessions_when_full": True,
        },
        "scheduler_fairness": {
            "policy": "fifo_compatible_sampling_key",
            "compatible_sampling_coalescing": True,
            "continuous_decode": False,
            "preemptive_fairness": False,
        },
    }
    assert body["errors"]["schema"] == "hipengine.error_taxonomy.v1"
    errors_by_code = {item["code"]: item for item in body["errors"]["codes"]}
    for code in (
        "unsupported_parameter",
        "unsupported_feature",
        "invalid_tool_call",
        "schema_violation",
        "invalid_continuation",
        "continuation_expired",
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
    assert errors_by_code["unsupported_feature"]["status_code"] == 501
    assert errors_by_code["invalid_continuation"]["status_code"] == 400
    assert errors_by_code["continuation_expired"]["status_code"] == 410
    assert errors_by_code["routing_failed"]["status_code"] == 502
    assert errors_by_code["routing_failed"]["retryable"] is True
    assert errors_by_code["routing_failed"]["emitted"] is False
    assert errors_by_code["invalid_tool_call"]["emitted"] is True
    assert "finish_details.reason" in errors_by_code["invalid_tool_call"]["description"]
    assert "structured-output result" in errors_by_code["schema_violation"]["description"]
    assert {
        "legacy_code": "model_not_found",
        "code": "model_unavailable",
    } in body["errors"]["aliases"]
    assert {
        "legacy_code": "unsupported_content_type",
        "code": "unsupported_parameter",
    } in body["errors"]["aliases"]
    assert {
        "legacy_code": "invalid_request",
        "code": "schema_violation",
    } in body["errors"]["aliases"]
    assert {
        "legacy_code": "validation_error",
        "code": "schema_violation",
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
    assert "json_object_close_forcing" in body["sampling"]["parameters"]
    assert body["sampling"]["native_gpu"] == {
        "enabled": True,
        "env": "HIPENGINE_QWEN35_NATIVE_SAMPLER",
        "disable_env": "HIPENGINE_QWEN35_NATIVE_SAMPLER=0",
        "scope": "paro_c1_and_serial_per_slot_c_gt_1",
        "c_gt_1": "serial_per_slot_when_all_rows_supported",
        "true_batched_c_gt_1": False,
        "default_path": True,
        "top_k_max": 64,
        "top_p_min_p": "exact_full_vocab_top_k_0_and_bounded_top_k",
        "selected_logprobs": True,
        "top_logprobs": {
            "full_vocab_top_k_0": True,
            "bounded_top_k": True,
            "max": 64,
            "constraint": "top_k=0 or top_logprobs <= top_k <= 64",
        },
        "processors": [
            "logit_bias",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "suppress_token_ids",
            "min_tokens",
        ],
        "post_selection_controls": [
            "stop_token_ids",
            "stop_token_sequences",
        ],
        "unsupported": list(NATIVE_GPU_SAMPLER_UNSUPPORTED_CAPABILITIES),
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
            "eos_token_id",
            "ignore_eos",
            "stop_token_ids",
            "stop_token_sequences",
            "forced_tokens_pending",
            "post_thinking_forced_tokens_pending",
            "force_sequence_completion_token_sequences",
            "json_object_close_forcing",
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
            "eos_token_id": "eos_token_id set",
            "ignore_eos": "ignore_eos=true",
            "stop_token_ids": "one or more token stop ids",
            "stop_token_sequences": "one or more multi-token stop sequences",
            "forced_tokens_pending": "one or more forced tokens pending",
            "post_thinking_forced_tokens_pending": "one or more post-thinking forced tokens pending",
            "force_sequence_completion_token_sequences": "one or more token sequence completion repairs",
            "json_object_close_forcing": "JSON object close forcing active",
            "thinking_budget": "thinking budget soft-close, EOS suppression, or hard-close control",
            "logprobs": "logprobs requested",
            "top_logprobs": "top_logprobs > 0",
        },
        "processed_target_verification": False,
    }
    assert body["sampling"]["speculative_mtp"]["incompatible_fields"] == list(
        SPECULATIVE_MTP_INCOMPATIBLE_FIELDS
    )
    assert body["sampling"]["speculative_mtp"]["incompatible_conditions"] == dict(
        SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS
    )
    mtp_incompatible = set(body["sampling"]["speculative_mtp"]["incompatible_fields"])
    assert {
        "response_format",
        "guided_json",
        "guided_regex",
        "guided_choice",
        "guided_patch",
        "guided_diff",
    }.isdisjoint(mtp_incompatible)
    assert body["sessions"] == {
        "resident_context": True,
        "commit_policy": _session_commit_policy_capability(),
        "continuations": _continuation_capability(),
        "metadata": _session_metadata_capability(3),
    }
    assert body["routing"] == {"loaded_model_count": 1, "multiple_models": False}
    assert body["parallelism"] == _parallelism_capability()
    assert "session.id" not in body["unsupported_fields"]
    assert "timeout_ms" not in body["unsupported_fields"]
    assert "parallel_tool_calls" not in body["unsupported_fields"]
    assert "response_format" not in body["unsupported_fields"]
    assert "continuation_id" not in body["unsupported_fields"]
    assert "grammar" in body["unsupported_fields"]
    assert "guided_json" not in body["unsupported_fields"]
    assert "guided_json" in body["features"]["grammars"]["result_validation_only"]
    assert "guided_patch" not in body["unsupported_fields"]
    assert "guided_diff" not in body["unsupported_fields"]


def test_capabilities_endpoint_advertises_live_stream_logprobs_when_engine_supports_metadata() -> None:
    fake = FakeLLM(outputs=["ok"])
    fake.supports_stream_logprobs = True
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False), llm=fake)
    client = TestClient(app)

    response = client.get("/v1/hipengine/capabilities")

    assert response.status_code == 200
    assert response.json()["features"]["logprobs"] == {
        "completions": True,
        "chat": True,
        "top_logprobs_max": 20,
        "streaming": "live_chunk_metadata",
        "live_chunk_metadata": True,
        "live_chunk_metadata_capability": "engine.supports_stream_logprobs",
        "chat_reasoning_private_stream_metadata": "choices[].hipengine.reasoning_logprobs",
        "requires_backend_token_metadata": True,
        "omission_reasons": ["backend_omitted_logprob", "prompt_logprob_unavailable"],
        "missing_backend_metadata_error": {
            "code": "unsupported_feature",
            "status_code": 501,
            "param": "logprobs",
        },
    }


def test_api_error_taxonomy_table_matches_capabilities_manifest() -> None:
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False), llm=FakeLLM())
    client = TestClient(app)

    response = client.get("/v1/hipengine/capabilities")

    assert response.status_code == 200
    api_table = _api_error_taxonomy_table()
    manifest_by_code = {item["code"]: item for item in response.json()["errors"]["codes"]}
    assert set(api_table) == set(manifest_by_code)
    for code, item in manifest_by_code.items():
        assert api_table[code]["status_code"] == item["status_code"]
        assert api_table[code]["retryable"] == item["retryable"]
        assert api_table[code]["current_emission"]


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
            max_chat_sessions=6,
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
    assert body["sessions"]["metadata"]["max_active"] == 6


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
    assert fit_body["max_allowed_max_tokens"] == 512 - chat_body["token_count"] - 1
    assert fit_body["recommended_max_tokens"] == expected_max_tokens
    assert fit_body["required_context_tokens"] == chat_body["token_count"] + expected_max_tokens + 1
    assert fit_body["overflow_tokens"] == 0
    assert fit_body["fits"] is True
    assert fit_body["chat_default_max_tokens"] == 7
    assert fit_body["clear_policy"] == "reject"
    assert fit_body["would_drop"] == []
    assert fit_body["thinking_budget"]["lowering_supported"] is True
    assert fit_body["thinking_budget"]["close_token_ids"] == [42, 43, 44]


def test_token_diagnostics_report_unbounded_nested_reasoning_control() -> None:
    fake = FakeLLM(token_map={"</think>": [42, 43]})
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=512,
            chat_default_max_tokens=128,
        ),
        llm=fake,
    )
    client = TestClient(app)
    payload = {
        "messages": [{"role": "user", "content": "think without hard cap"}],
        "max_tokens": 64,
        "reasoning": {"effort": "low", "allow_unbounded": True},
    }

    count = client.post("/v1/hipengine/count_tokens", json=payload)
    fit = client.post("/v1/hipengine/fit_context", json=payload)

    assert count.status_code == 200
    assert fit.status_code == 200
    for body in (count.json(), fit.json()):
        budget = body["thinking_budget"]
        assert budget["enabled"] is True
        assert budget["effort"] == "low"
        assert budget["allow_unbounded"] is True
        assert budget["min_answer_tokens"] == 32
        assert budget["close_text"] == "</think>"
        assert budget["close_token_ids"] == [42, 43]
        assert "hard_think_cap" not in budget
        assert budget["initial_state"] == {
            "phase": "think",
            "reasoning_tokens": 0,
            "answer_tokens": 0,
            "close_sequence": [42, 43],
        }


def test_token_diagnostics_use_session_prefix_for_chat() -> None:
    fake = SequentialFakeLLM(["stored answer", "follow-up answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=512,
            chat_default_max_tokens=9,
        ),
        llm=fake,
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "remember alpha"}],
            "max_tokens": 2,
            "session": {"id": "diag_session"},
        },
    )
    diagnostic_payload = {
        "messages": [{"role": "user", "content": "now beta"}],
        "session": {"id": "diag_session"},
    }
    count = client.post("/v1/hipengine/count_tokens", json=diagnostic_payload)
    fit = client.post("/v1/hipengine/fit_context", json=diagnostic_payload)
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "now beta"}],
            "max_tokens": 2,
            "session": {"id": "diag_session", "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert count.status_code == 200
    assert fit.status_code == 200
    assert second.status_code == 200
    count_body = count.json()
    assert count_body["input_type"] == "chat"
    assert "remember alpha" in count_body["text"]
    assert "stored answer" in count_body["text"]
    assert "now beta" in count_body["text"]
    assert count_body["session"] == {
        "id": "diag_session",
        "stateful": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "prefix_message_count": 2,
        "request_message_count": 1,
        "rendered_message_count": 3,
        "cache_action": "append_visible_only",
    }
    assert count_body["token_count"] == fake.count_tokens(count_body["text"])

    fit_body = fit.json()
    assert fit_body["prompt_tokens"] == count_body["token_count"]
    assert fit_body["effective_max_tokens"] == 9
    assert fit_body["max_allowed_max_tokens"] == 512 - count_body["token_count"] - 1
    assert fit_body["recommended_max_tokens"] == 9
    assert fit_body["required_context_tokens"] == count_body["token_count"] + 9 + 1
    assert fit_body["overflow_tokens"] == 0
    assert fit_body["session"] == count_body["session"]
    assert fake.calls[1][0][0] == count_body["text"]


def test_chat_context_overflow_reports_session_fit_context() -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=20,
        ),
        llm=fake,
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "remember alpha"}],
            "max_tokens": 1,
            "session": {"id": "overflow_session"},
        },
    )
    diagnostic_payload = {
        "messages": [{"role": "user", "content": "now beta"}],
        "max_tokens": 64,
        "session": {"id": "overflow_session"},
    }
    fit = client.post("/v1/hipengine/fit_context", json=diagnostic_payload)
    overflow = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", **diagnostic_payload},
    )

    assert first.status_code == 200
    assert fit.status_code == 200
    assert overflow.status_code == 400
    fit_body = fit.json()
    error = overflow.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert error["hipengine"]["code"] == "context_overflow"
    assert error["param"] == "max_tokens"
    for key in (
        "prompt_tokens",
        "max_context_tokens",
        "effective_max_tokens",
        "max_allowed_max_tokens",
        "recommended_max_tokens",
        "required_context_tokens",
        "overflow_tokens",
        "fits",
        "clear_policy",
        "would_truncate",
        "would_drop",
        "session",
    ):
        assert error["fit_context"][key] == fit_body[key]
    assert error["fit_context"]["session"] == {
        "id": "overflow_session",
        "stateful": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "prefix_message_count": 2,
        "request_message_count": 1,
        "rendered_message_count": 3,
        "cache_action": "append_visible_only",
    }
    assert len(fake.calls) == 1


def test_fit_context_new_session_policy_reports_reset_when_prefix_overflows() -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=10,
        ),
        llm=fake,
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "remember alpha"}],
            "max_tokens": 1,
            "session": {"id": "fit_reset_session"},
        },
    )
    fit = client.post(
        "/v1/hipengine/fit_context",
        json={
            "messages": [{"role": "user", "content": "now beta"}],
            "max_tokens": 5,
            "session": {"id": "fit_reset_session", "context_overflow_policy": "new_session"},
        },
    )

    assert first.status_code == 200
    assert fit.status_code == 200
    body = fit.json()
    assert body["fits"] is True
    assert body["prompt_tokens"] == 4
    assert body["effective_max_tokens"] == 5
    assert body["required_context_tokens"] == 10
    assert body["clear_policy"] == "new_session"
    assert body["would_truncate"] is False
    assert body["would_reset_session"] is True
    assert body["would_drop"] == [
        {
            "kind": "session_prefix",
            "session_id": "fit_reset_session",
            "storage": "app_local_transcript",
            "message_count": 2,
        }
    ]
    assert body["kept_segments"] == [{"kind": "request_messages", "message_count": 1}]
    assert body["session"] == {
        "id": "fit_reset_session",
        "stateful": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "prefix_message_count": 0,
        "request_message_count": 1,
        "rendered_message_count": 1,
        "cache_action": "append_visible_only",
    }
    serialized = json.dumps(body)
    assert "now beta" in serialized
    assert "remember alpha" not in serialized
    assert "stored answer" not in serialized


def test_chat_context_new_session_policy_resets_transcript_on_success() -> None:
    fake = SequentialFakeLLM(["stored answer", "fresh answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=10,
        ),
        llm=fake,
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "remember alpha"}],
            "max_tokens": 1,
            "session": {"id": "chat_reset_session"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "now beta"}],
            "max_tokens": 5,
            "session": {"id": "chat_reset_session", "context_overflow_policy": "new_session"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(fake.calls) == 2
    prompt = fake.calls[1][0][0]
    assert "now beta" in prompt
    assert "remember alpha" not in prompt
    assert "stored answer" not in prompt
    record = app.state.hipengine_chat_sessions["chat_reset_session"]
    assert record.messages == (
        {"role": "user", "content": "now beta"},
        {"role": "assistant", "content": "fresh answer"},
    )


def test_chat_context_new_session_policy_does_not_hide_request_overflow() -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=10,
        ),
        llm=fake,
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "remember alpha"}],
            "max_tokens": 1,
            "session": {"id": "too_large_session"},
        },
    )
    payload = {
        "messages": [{"role": "user", "content": "one two three four five six seven eight nine ten"}],
        "max_tokens": 1,
        "session": {"id": "too_large_session", "context_overflow_policy": "new_session"},
    }
    fit = client.post("/v1/hipengine/fit_context", json=payload)
    overflow = client.post("/v1/chat/completions", json={"model": "fake-model", **payload})

    assert first.status_code == 200
    assert fit.status_code == 200
    assert overflow.status_code == 400
    fit_body = fit.json()
    error = overflow.json()["error"]
    assert fit_body["fits"] is False
    assert fit_body["clear_policy"] == "new_session"
    assert fit_body["would_reset_session"] is False
    assert fit_body["would_drop"] == []
    assert fit_body["kept_segments"] == [
        {
            "kind": "session_prefix",
            "session_id": "too_large_session",
            "storage": "app_local_transcript",
            "message_count": 2,
        },
        {"kind": "request_messages", "message_count": 1},
    ]
    assert fit_body["session"]["prefix_message_count"] == 2
    assert error["code"] == "context_length_exceeded"
    assert error["fit_context"]["clear_policy"] == "new_session"
    assert error["fit_context"]["would_reset_session"] is False
    assert len(fake.calls) == 1
    assert app.state.hipengine_chat_sessions["too_large_session"].messages == (
        {"role": "user", "content": "remember alpha"},
        {"role": "assistant", "content": "stored answer"},
    )


def test_chat_context_auto_clear_transient_preserves_committed_prefix() -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=10,
        ),
        llm=fake,
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "remember alpha"}],
            "max_tokens": 1,
            "session": {"id": "auto_clear_session"},
        },
    )
    payload = {
        "messages": [{"role": "user", "content": "now beta"}],
        "max_tokens": 5,
        "session": {"id": "auto_clear_session", "context_overflow_policy": "auto_clear_transient"},
    }
    fit = client.post("/v1/hipengine/fit_context", json=payload)
    overflow = client.post("/v1/chat/completions", json={"model": "fake-model", **payload})

    assert first.status_code == 200
    assert fit.status_code == 200
    assert overflow.status_code == 400
    fit_body = fit.json()
    assert fit_body["fits"] is False
    assert fit_body["clear_policy"] == "auto_clear_transient"
    assert fit_body["would_clear_transient"] is False
    assert fit_body["transient_message_count"] == 0
    assert fit_body["would_drop"] == []
    assert fit_body["kept_segments"] == [
        {
            "kind": "session_prefix",
            "session_id": "auto_clear_session",
            "storage": "app_local_transcript",
            "message_count": 2,
        },
        {"kind": "request_messages", "message_count": 1},
    ]
    assert fit_body["session"] == {
        "id": "auto_clear_session",
        "stateful": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "prefix_message_count": 2,
        "request_message_count": 1,
        "rendered_message_count": 3,
        "cache_action": "append_visible_only",
    }
    error = overflow.json()["error"]
    assert error["code"] == "context_length_exceeded"
    assert error["fit_context"]["clear_policy"] == "auto_clear_transient"
    assert error["fit_context"]["would_clear_transient"] is False
    assert error["fit_context"]["transient_message_count"] == 0
    assert error["fit_context"]["would_drop"] == []
    assert len(fake.calls) == 1
    assert app.state.hipengine_chat_sessions["auto_clear_session"].messages == (
        {"role": "user", "content": "remember alpha"},
        {"role": "assistant", "content": "stored answer"},
    )


def test_chat_context_truncate_oldest_visible_policy_keeps_fitting_suffix() -> None:
    fake = SequentialFakeLLM(["answer alpha", "answer beta", "answer gamma"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=12,
        ),
        llm=fake,
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "one alpha"}],
            "max_tokens": 1,
            "session": {"id": "truncate_session"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "two beta"}],
            "max_tokens": 1,
            "session": {"id": "truncate_session"},
        },
    )
    payload = {
        "messages": [{"role": "user", "content": "now gamma"}],
        "max_tokens": 1,
        "session": {"id": "truncate_session", "context_overflow_policy": "truncate_oldest_visible"},
    }
    fit = client.post("/v1/hipengine/fit_context", json=payload)
    third = client.post("/v1/chat/completions", json={"model": "fake-model", **payload})

    assert first.status_code == 200
    assert second.status_code == 200
    assert fit.status_code == 200
    assert third.status_code == 200
    fit_body = fit.json()
    assert fit_body["fits"] is True
    assert fit_body["prompt_tokens"] == 10
    assert fit_body["effective_max_tokens"] == 1
    assert fit_body["required_context_tokens"] == 12
    assert fit_body["clear_policy"] == "truncate_oldest_visible"
    assert fit_body["would_truncate"] is True
    assert fit_body["would_reset_session"] is False
    assert fit_body["would_drop"] == [
        {
            "kind": "session_prefix",
            "session_id": "truncate_session",
            "storage": "app_local_transcript",
            "message_count": 2,
        }
    ]
    assert fit_body["kept_segments"] == [
        {
            "kind": "session_prefix",
            "session_id": "truncate_session",
            "storage": "app_local_transcript",
            "message_count": 2,
        },
        {"kind": "request_messages", "message_count": 1},
    ]
    assert fit_body["session"] == {
        "id": "truncate_session",
        "stateful": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "prefix_message_count": 2,
        "request_message_count": 1,
        "rendered_message_count": 3,
        "cache_action": "append_visible_only",
    }
    serialized = json.dumps(fit_body)
    assert "two beta" in serialized
    assert "answer beta" in serialized
    assert "now gamma" in serialized
    assert "one alpha" not in serialized
    assert "answer alpha" not in serialized
    prompt = fake.calls[2][0][0]
    assert "two beta" in prompt
    assert "answer beta" in prompt
    assert "now gamma" in prompt
    assert "one alpha" not in prompt
    assert "answer alpha" not in prompt
    assert app.state.hipengine_chat_sessions["truncate_session"].messages == (
        {"role": "user", "content": "two beta"},
        {"role": "assistant", "content": "answer beta"},
        {"role": "user", "content": "now gamma"},
        {"role": "assistant", "content": "answer gamma"},
    )


def test_chat_context_truncate_oldest_visible_policy_skips_orphan_tool_suffix() -> None:
    class WeightedToolTranscriptFakeLLM(SequentialFakeLLM):
        def count_tokens(self, text: str) -> int:
            prompt = str(text)
            if "new beta" not in prompt:
                return 1
            score = 1
            if "README.md" in prompt:
                score += 20
            if "alpha file text" in prompt:
                score += 1
            if "alpha summary" in prompt:
                score += 1
            return score

    fake = WeightedToolTranscriptFakeLLM(
        [
            '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>',
            "alpha summary",
            "after truncate",
        ]
    )
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_context_tokens=10,
        ),
        llm=fake,
    )
    client = TestClient(app)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
        }
    ]

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "alpha read"}],
            "tools": tools,
            "max_tokens": 1,
            "session": {"id": "truncate_tool_session"},
        },
    )
    assert first.status_code == 200
    tool_call_id = first.json()["choices"][0]["message"]["tool_calls"][0]["id"]
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "tool", "tool_call_id": tool_call_id, "content": "alpha file text"}],
            "tools": tools,
            "max_tokens": 1,
            "session": {"id": "truncate_tool_session"},
        },
    )
    assert second.status_code == 200
    original_prefix = app.state.hipengine_chat_sessions["truncate_tool_session"].messages
    payload = {
        "messages": [{"role": "user", "content": "new beta"}],
        "max_tokens": 1,
        "session": {
            "id": "truncate_tool_session",
            "context_overflow_policy": "truncate_oldest_visible",
        },
    }

    fit = client.post("/v1/hipengine/fit_context", json=payload)
    third = client.post("/v1/chat/completions", json={"model": "fake-model", **payload})

    assert fit.status_code == 200
    assert third.status_code == 200
    fit_body = fit.json()
    assert fit_body["fits"] is True
    assert fit_body["clear_policy"] == "truncate_oldest_visible"
    assert fit_body["would_truncate"] is True
    assert fit_body["would_reset_session"] is False
    dropped_messages = fit_body["would_drop"][0]["message_count"]
    assert dropped_messages >= 3
    assert fit_body["kept_segments"] == [
        {
            "kind": "session_prefix",
            "session_id": "truncate_tool_session",
            "storage": "app_local_transcript",
            "message_count": len(original_prefix) - dropped_messages,
        },
        {"kind": "request_messages", "message_count": 1},
    ]
    serialized = json.dumps(fit_body)
    assert "new beta" in serialized
    assert "alpha file text" not in serialized
    prompt = fake.calls[2][0][0]
    assert "new beta" in prompt
    assert "alpha file text" not in prompt
    assert "<tool_response>" not in prompt
    expected_messages = (
        *original_prefix[dropped_messages:],
        {"role": "user", "content": "new beta"},
        {"role": "assistant", "content": "after truncate"},
    )
    assert app.state.hipengine_chat_sessions["truncate_tool_session"].messages == expected_messages


def test_chat_context_overflow_policy_requires_known_stateful_mode() -> None:
    fake = FakeLLM(outputs=["unused"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    invalid = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "session": {"id": "bad_policy", "context_overflow_policy": "compact_summary"},
        },
    )
    stateless = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "session": {"context_overflow_policy": "new_session"},
        },
    )

    assert invalid.status_code == 400
    assert invalid.json()["error"]["code"] == "invalid_request"
    assert invalid.json()["error"]["param"] == "session.context_overflow_policy"
    assert stateless.status_code == 400
    assert stateless.json()["error"]["code"] == "unsupported_parameter"
    assert stateless.json()["error"]["param"] == "session.context_overflow_policy"
    assert fake.calls == []


def test_token_diagnostics_report_unsupported_model_hooks() -> None:
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=NoTokenizerFakeLLM(),
    )
    client = TestClient(app)

    capabilities = client.get("/v1/hipengine/capabilities")
    tokenize = client.post("/v1/hipengine/tokenize", json={"text": "hello"})
    detokenize = client.post("/v1/hipengine/detokenize", json={"token_ids": [1, 2]})
    count = client.post("/v1/hipengine/count_tokens", json={"text": "hello"})
    fit = client.post("/v1/hipengine/fit_context", json={"text": "hello"})

    assert capabilities.status_code == 200
    tokenizer = capabilities.json()["tokenizer"]
    assert tokenizer["tokenize"] is False
    assert tokenizer["detokenize"] is False
    assert tokenizer["count_tokens"] is False
    assert capabilities.json()["features"]["token_diagnostics"] == {
        "tokenize": False,
        "detokenize": False,
        "count_tokens": False,
        "fit_context": False,
        "session_aware_chat": False,
    }
    for response, param in (
        (tokenize, "text"),
        (detokenize, "token_ids"),
        (count, "text"),
        (fit, "text"),
    ):
        assert response.status_code == 501
        assert response.json()["error"]["code"] == "unsupported_feature"
        assert response.json()["error"]["param"] == param
        assert response.json()["error"]["hipengine"] == {
            "code": "unsupported_feature",
            "status_code": 501,
            "retryable": False,
        }


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
    assert response.json()["error"]["hipengine"] == {
        "code": "schema_violation",
        "status_code": 400,
        "legacy_code": "invalid_request",
        "retryable": False,
    }


def test_token_diagnostics_reject_session_for_raw_text() -> None:
    fake = FakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/hipengine/count_tokens",
        json={"text": "hello", "session": {"id": "diag_session"}},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    assert response.json()["error"]["param"] == "session"


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
    assert body["queue"]["active_requests"] == 0
    assert body["queue"]["max_active_requests"] is None
    assert body["queue"]["scheduler_fairness"] == {
        "policy": "fifo_compatible_sampling_key",
        "compatible_sampling_coalescing": True,
        "continuous_decode": False,
        "preemptive_fairness": False,
    }
    assert body["sessions"] == {
        "resident_context": True,
        "active": 0,
        "pending_creations": 0,
        "max_active": None,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "total_messages": 0,
        "continuations": {"active": 0, "ttl_seconds": 900},
    }
    serialized = json.dumps(body)
    assert "private startup prompt" not in serialized
    assert "private warmup output" not in serialized


def test_ready_reports_selected_visible_gpu_from_rocm_env(monkeypatch) -> None:
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0, 2")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3")
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=FakeLLM(),
    )
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["device"] == {
        "backend": "auto",
        "hip_visible_devices": "0, 2",
        "rocr_visible_devices": "3",
        "visible_devices": ["0", "2"],
        "selected_visible_device": "0",
        "selection_source": "HIP_VISIBLE_DEVICES",
    }


def test_ready_reports_startup_failure_diagnostics_without_payload_text() -> None:
    fake = ScratchProbeFailureFakeLLM(outputs=["private warmup output"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load_prompt="private startup prompt",
            eager_load_max_tokens=2,
        ),
        llm=fake,
    )

    with TestClient(app) as client:
        ready = client.get("/ready")

    assert ready.status_code == 503
    body = ready.json()
    assert body["object"] == "hipengine.readiness"
    assert body["ready"] is False
    assert body["status"] == "error"
    assert body["diagnostics"] == [
        "server startup is not ready; check startup.error and server logs",
        "startup scratch_probe failed: Try a lower --max-context-tokens or a higher scratch/headroom reserve.",
    ]
    assert body["startup"]["warmup_complete"] is False
    assert body["startup"]["error"] == {
        "stage": "scratch_probe",
        "type": "RuntimeError",
        "message": "startup scratch_probe failed",
        "guidance": "Try a lower --max-context-tokens or a higher scratch/headroom reserve.",
    }
    assert body["startup"]["checks"]["raw_warmup"] == {"status": "passed", "max_tokens": 2}
    assert body["startup"]["checks"]["scratch_probe"] == {
        "enabled": True,
        "status": "failed",
        "max_prompt_tokens": 131071,
        "exception_type": "RuntimeError",
    }
    assert body["startup"]["last_timings_s"]["warmup_s"] >= 0.0
    assert body["startup"]["last_timings_s"]["scratch_probe_s"] >= 0.0
    assert body["startup"]["last_timings_s"]["startup_total_s"] >= 0.0
    serialized = json.dumps(body)
    assert "private startup prompt" not in serialized
    assert "private warmup output" not in serialized


def test_ready_reports_chat_session_counts_without_payload_text() -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=fake,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "secret session prompt"}],
            "session": {"id": "sess_ready"},
            "max_tokens": 4,
        },
    )
    ready = client.get("/ready")

    assert response.status_code == 200
    assert ready.status_code == 200
    body = ready.json()
    assert body["sessions"] == {
        "resident_context": True,
        "active": 1,
        "pending_creations": 0,
        "max_active": None,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "total_messages": 2,
        "continuations": {"active": 0, "ttl_seconds": 900},
    }
    serialized = json.dumps(body)
    assert "secret session prompt" not in serialized
    assert "stored answer" not in serialized


def test_session_metadata_list_and_delete_are_authenticated() -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            api_key="secret",
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    unauthorized = client.get("/v1/hipengine/sessions")
    created = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "secret list prompt"}],
            "session": {"id": "sess_list"},
            "max_tokens": 4,
        },
    )
    listed = client.get("/v1/hipengine/sessions", headers=headers)
    deleted = client.delete("/v1/hipengine/sessions/sess_list", headers=headers)
    deleted_again = client.delete("/v1/hipengine/sessions/sess_list", headers=headers)
    listed_after_delete = client.get("/v1/hipengine/sessions", headers=headers)

    assert unauthorized.status_code == 401
    assert created.status_code == 200
    assert listed.status_code == 200
    body = listed.json()
    assert body["object"] == "hipengine.sessions"
    assert body["storage"] == "app_local_transcript"
    assert body["resident_state_reuse"] is False
    assert body["includes_transcript"] is False
    assert body["active"] == 1
    assert body["pending_creations"] == 0
    assert body["max_active"] is None
    assert body["continuations"] == {"active": 0, "ttl_seconds": 900}
    assert len(body["sessions"]) == 1
    metadata = body["sessions"][0]
    assert metadata["id"] == "sess_list"
    assert metadata["storage"] == "app_local_transcript"
    assert metadata["resident_state_reuse"] is False
    assert metadata["message_count"] == 2
    assert isinstance(metadata["created"], int)
    assert isinstance(metadata["updated"], int)
    serialized = json.dumps(body)
    assert "secret list prompt" not in serialized
    assert "stored answer" not in serialized
    assert deleted.json() == {
        "object": "hipengine.session.deleted",
        "id": "sess_list",
        "deleted": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
    }
    assert deleted_again.json()["deleted"] is False
    assert listed_after_delete.json()["active"] == 0
    assert listed_after_delete.json()["sessions"] == []


def test_chat_session_message_copy_deep_copies_nested_tool_calls() -> None:
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path":"README.md"}'},
            }
        ],
    }

    copied = _chat_session_message_copy(message)

    assert copied == message
    assert copied is not message
    assert copied["tool_calls"] is not message["tool_calls"]
    assert copied["tool_calls"][0] is not message["tool_calls"][0]
    assert copied["tool_calls"][0]["function"] is not message["tool_calls"][0]["function"]
    copied["tool_calls"][0]["function"]["arguments"] = '{"path":"MUTATED.md"}'
    assert message["tool_calls"][0]["function"]["arguments"] == '{"path":"README.md"}'


def test_chat_session_fork_branches_visible_transcript_without_cross_contamination() -> None:
    fake = SequentialFakeLLM(["base answer", "left answer", "right answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            api_key="secret",
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    created = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "base prompt"}],
            "session": {"id": "sess_root"},
            "max_tokens": 4,
        },
    )
    unauthorized = client.post("/v1/hipengine/sessions/sess_root/fork", json={"id": "sess_branch"})
    forked = client.post(
        "/v1/hipengine/sessions/sess_root/fork",
        headers=headers,
        json={"id": "sess_branch"},
    )
    continued_root = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "left turn"}],
            "session": {"id": "sess_root"},
            "max_tokens": 4,
        },
    )
    continued_branch = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "right turn"}],
            "session": {"id": "sess_branch"},
            "max_tokens": 4,
        },
    )
    listed = client.get("/v1/hipengine/sessions", headers=headers)

    assert created.status_code == 200
    assert unauthorized.status_code == 401
    assert forked.status_code == 200
    assert forked.json() == {
        "object": "hipengine.session.forked",
        "source_id": "sess_root",
        "id": "sess_branch",
        "forked": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "message_count": 2,
    }
    assert continued_root.status_code == 200
    assert continued_branch.status_code == 200
    root_prompt = fake.calls[1][0][0]
    branch_prompt = fake.calls[2][0][0]
    assert "base prompt" in root_prompt
    assert "base answer" in root_prompt
    assert "left turn" in root_prompt
    assert "right turn" not in root_prompt
    assert "base prompt" in branch_prompt
    assert "base answer" in branch_prompt
    assert "right turn" in branch_prompt
    assert "left turn" not in branch_prompt
    body = listed.json()
    assert body["active"] == 2
    counts = {session["id"]: session["message_count"] for session in body["sessions"]}
    assert counts == {"sess_root": 4, "sess_branch": 4}
    assert "base prompt" not in json.dumps(body)
    assert "base answer" not in json.dumps(body)


def test_chat_session_fork_deep_copies_nested_tool_calls() -> None:
    fake = SequentialFakeLLM(
        ['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>']
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
        }
    ]

    created = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read it"}],
            "tools": tools,
            "session": {"id": "sess_tool_root", "commit": "append_visible_only"},
            "max_tokens": 4,
        },
    )
    forked = client.post(
        "/v1/hipengine/sessions/sess_tool_root/fork",
        json={"id": "sess_tool_branch"},
    )

    assert created.status_code == 200
    assert forked.status_code == 200
    root_call = app.state.hipengine_chat_sessions["sess_tool_root"].messages[1]["tool_calls"][0]
    branch_call = app.state.hipengine_chat_sessions["sess_tool_branch"].messages[1]["tool_calls"][0]
    assert branch_call is not root_call
    assert branch_call["function"] is not root_call["function"]

    branch_call["function"]["arguments"] = '{"path":"BRANCH.md"}'

    assert root_call["function"]["arguments"] == '{"path":"README.md"}'
    assert branch_call["function"]["arguments"] == '{"path":"BRANCH.md"}'


def test_chat_session_fork_rejects_existing_target_and_session_cap() -> None:
    fake = SequentialFakeLLM(["root answer", "target answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            api_key="secret",
            max_chat_sessions=2,
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    root = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "root"}],
            "session": {"id": "sess_root"},
            "max_tokens": 4,
        },
    )
    target = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "target"}],
            "session": {"id": "sess_target"},
            "max_tokens": 4,
        },
    )
    existing = client.post(
        "/v1/hipengine/sessions/sess_root/fork",
        headers=headers,
        json={"id": "sess_target"},
    )
    full = client.post(
        "/v1/hipengine/sessions/sess_root/fork",
        headers=headers,
        json={"id": "sess_new"},
    )
    missing = client.post(
        "/v1/hipengine/sessions/sess_missing/fork",
        headers=headers,
        json={"id": "sess_other"},
    )

    assert root.status_code == 200
    assert target.status_code == 200
    assert existing.status_code == 400
    assert existing.json()["error"]["param"] == "id"
    assert full.status_code == 429
    assert full.json()["error"]["code"] == "engine_busy"
    assert full.json()["error"]["message"] == "chat session limit is full"
    assert full.json()["error"]["hipengine"] == {
        "code": "engine_busy",
        "status_code": 429,
        "retryable": True,
        "routing": {
            **_routing_metadata(),
            "matched": True,
            "reason": "engine_busy",
            "overload_source": "chat_session_cap",
            "max_active_chat_sessions": 2,
        },
    }
    assert full.headers["retry-after"] == "1"
    assert missing.status_code == 404
    assert missing.json()["error"]["param"] == "session_id"
    assert set(app.state.hipengine_chat_sessions) == {"sess_root", "sess_target"}
    assert app.state.hipengine_server_metrics.request_rejected_total == 1


def test_chat_session_rollback_trims_visible_transcript_for_next_turn() -> None:
    fake = SequentialFakeLLM(["base answer", "second answer", "after rollback"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            api_key="secret",
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    first = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "base prompt"}],
            "session": {"id": "sess_rollback"},
            "max_tokens": 4,
        },
    )
    second = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "second turn"}],
            "session": {"id": "sess_rollback"},
            "max_tokens": 4,
        },
    )
    unauthorized = client.post(
        "/v1/hipengine/sessions/sess_rollback/rollback",
        json={"message_count": 2},
    )
    rollback = client.post(
        "/v1/hipengine/sessions/sess_rollback/rollback",
        headers=headers,
        json={"message_count": 2},
    )
    continued = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "after rollback turn"}],
            "session": {"id": "sess_rollback", "commit": "append_none"},
            "max_tokens": 4,
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert unauthorized.status_code == 401
    assert rollback.status_code == 200
    assert rollback.json() == {
        "object": "hipengine.session.rolled_back",
        "id": "sess_rollback",
        "rolled_back": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "previous_message_count": 4,
        "message_count": 2,
    }
    assert continued.status_code == 200
    record = app.state.hipengine_chat_sessions["sess_rollback"]
    assert len(record.messages) == 2
    prompt = fake.calls[2][0][0]
    assert "base prompt" in prompt
    assert "base answer" in prompt
    assert "after rollback turn" in prompt
    assert "second turn" not in prompt
    assert "second answer" not in prompt
    listed = client.get("/v1/hipengine/sessions", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["sessions"][0]["message_count"] == 2
    assert "base prompt" not in json.dumps(listed.json())
    assert "base answer" not in json.dumps(listed.json())


def test_chat_session_rollback_deep_copies_retained_tool_calls() -> None:
    fake = SequentialFakeLLM(
        [
            '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>',
            "done",
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
        }
    ]

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read it"}],
            "tools": tools,
            "session": {"id": "sess_tool_rollback", "commit": "append_visible_only"},
            "max_tokens": 4,
        },
    )
    assert first.status_code == 200
    tool_call_id = first.json()["choices"][0]["message"]["tool_calls"][0]["id"]
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "tool", "tool_call_id": tool_call_id, "content": "file text"}],
            "tools": tools,
            "session": {"id": "sess_tool_rollback", "commit": "append_visible_only"},
            "max_tokens": 4,
        },
    )
    previous = app.state.hipengine_chat_sessions["sess_tool_rollback"]
    rollback = client.post(
        "/v1/hipengine/sessions/sess_tool_rollback/rollback",
        json={"message_count": 2},
    )

    assert second.status_code == 200
    assert rollback.status_code == 200
    assert rollback.json()["rolled_back"] is True
    rolled_back = app.state.hipengine_chat_sessions["sess_tool_rollback"]
    previous_call = previous.messages[1]["tool_calls"][0]
    rolled_back_call = rolled_back.messages[1]["tool_calls"][0]
    assert rolled_back_call is not previous_call
    assert rolled_back_call["function"] is not previous_call["function"]

    rolled_back_call["function"]["arguments"] = '{"path":"ROLLED.md"}'

    assert previous_call["function"]["arguments"] == '{"path":"README.md"}'
    assert rolled_back_call["function"]["arguments"] == '{"path":"ROLLED.md"}'


def test_chat_session_rollback_rejects_missing_and_out_of_range() -> None:
    fake = SequentialFakeLLM(["root answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            api_key="secret",
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    created = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "root prompt"}],
            "session": {"id": "sess_rollback_errors"},
            "max_tokens": 4,
        },
    )
    too_far = client.post(
        "/v1/hipengine/sessions/sess_rollback_errors/rollback",
        headers=headers,
        json={"message_count": 3},
    )
    missing = client.post(
        "/v1/hipengine/sessions/sess_missing/rollback",
        headers=headers,
        json={"message_count": 0},
    )
    noop = client.post(
        "/v1/hipengine/sessions/sess_rollback_errors/rollback",
        headers=headers,
        json={"message_count": 2},
    )

    assert created.status_code == 200
    assert too_far.status_code == 400
    assert too_far.json()["error"]["param"] == "message_count"
    assert missing.status_code == 404
    assert missing.json()["error"]["param"] == "session_id"
    assert noop.status_code == 200
    assert noop.json()["rolled_back"] is False
    assert noop.json()["previous_message_count"] == 2
    assert noop.json()["message_count"] == 2
    assert len(app.state.hipengine_chat_sessions["sess_rollback_errors"].messages) == 2


def test_chat_session_snapshot_export_restore_round_trips_visible_transcript() -> None:
    fake = SequentialFakeLLM(["stored answer", "after restore"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            backend="test-backend",
            quant="test-quant",
            eager_load=False,
            api_key="secret",
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    unauthorized = client.get("/v1/hipengine/sessions/sess_snap/snapshot")
    unauthorized_restore = client.post("/v1/hipengine/sessions/sess_snap/snapshot", json={})
    created = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "snapshot prompt"}],
            "session": {"id": "sess_snap"},
            "max_tokens": 4,
        },
    )
    exported = client.get("/v1/hipengine/sessions/sess_snap/snapshot", headers=headers)
    deleted = client.delete("/v1/hipengine/sessions/sess_snap", headers=headers)
    restored = client.post(
        "/v1/hipengine/sessions/sess_snap/snapshot",
        headers=headers,
        json=exported.json(),
    )
    continued = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "session": {"id": "sess_snap", "commit": "append_none"},
            "max_tokens": 4,
        },
    )

    assert unauthorized.status_code == 401
    assert unauthorized_restore.status_code == 401
    assert created.status_code == 200
    assert exported.status_code == 200
    snapshot = exported.json()
    assert snapshot["object"] == "hipengine.session.snapshot"
    assert snapshot["schema"] == "hipengine.chat_session_snapshot.v1"
    assert snapshot["model"] == {"id": "fake-model", "backend": "test-backend", "quant": "test-quant"}
    assert snapshot["tokenizer"] == {
        "name": "SequentialFakeLLM",
        "tokenize": True,
        "detokenize": True,
        "count_tokens": True,
    }
    assert snapshot["resident_state_reuse"] is False
    assert snapshot["session"]["id"] == "sess_snap"
    assert snapshot["session"]["storage"] == "app_local_transcript"
    assert snapshot["session"]["includes_transcript"] is True
    assert snapshot["messages"] == [
        {"role": "user", "content": "snapshot prompt"},
        {"role": "assistant", "content": "stored answer"},
    ]
    assert deleted.json()["deleted"] is True
    assert restored.json() == {
        "object": "hipengine.session.restored",
        "id": "sess_snap",
        "restored": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "message_count": 2,
    }
    assert continued.status_code == 200
    prompt = fake.calls[1][0][0]
    assert prompt.index("snapshot prompt") < prompt.index("stored answer") < prompt.index("continue")


def test_chat_session_snapshot_restore_rejects_incompatible_model() -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            backend="test-backend",
            quant="test-quant",
            eager_load=False,
            api_key="secret",
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    created = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "snapshot prompt"}],
            "session": {"id": "sess_bad"},
            "max_tokens": 4,
        },
    )
    snapshot = client.get("/v1/hipengine/sessions/sess_bad/snapshot", headers=headers).json()
    snapshot["model"]["quant"] = "other-quant"
    client.delete("/v1/hipengine/sessions/sess_bad", headers=headers)

    restored = client.post(
        "/v1/hipengine/sessions/sess_bad/snapshot",
        headers=headers,
        json=snapshot,
    )

    assert created.status_code == 200
    assert restored.status_code == 400
    assert restored.json()["error"]["code"] == "invalid_request"
    assert restored.json()["error"]["param"] == "model.quant"
    assert "sess_bad" not in app.state.hipengine_chat_sessions


def test_chat_session_snapshot_restore_rejects_incompatible_tokenizer() -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            api_key="secret",
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    created = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "snapshot prompt"}],
            "session": {"id": "sess_bad_tokenizer"},
            "max_tokens": 4,
        },
    )
    snapshot = client.get(
        "/v1/hipengine/sessions/sess_bad_tokenizer/snapshot",
        headers=headers,
    ).json()
    snapshot["tokenizer"]["name"] = "OtherTokenizer"
    client.delete("/v1/hipengine/sessions/sess_bad_tokenizer", headers=headers)

    restored = client.post(
        "/v1/hipengine/sessions/sess_bad_tokenizer/snapshot",
        headers=headers,
        json=snapshot,
    )

    assert created.status_code == 200
    assert restored.status_code == 400
    assert restored.json()["error"]["code"] == "invalid_request"
    assert restored.json()["error"]["param"] == "tokenizer.name"
    assert "sess_bad_tokenizer" not in app.state.hipengine_chat_sessions


@pytest.mark.parametrize(
    ("corruption", "param"),
    [
        ("object", "object"),
        ("resident_state_reuse", "resident_state_reuse"),
        ("session_resident_state_reuse", "session.resident_state_reuse"),
        ("session_includes_transcript", "session.includes_transcript"),
    ],
)
def test_chat_session_snapshot_restore_rejects_corrupted_envelope(
    corruption: str,
    param: str,
) -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            api_key="secret",
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    created = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "snapshot prompt"}],
            "session": {"id": "sess_corrupt_envelope"},
            "max_tokens": 4,
        },
    )
    snapshot = client.get(
        "/v1/hipengine/sessions/sess_corrupt_envelope/snapshot",
        headers=headers,
    ).json()
    if corruption == "object":
        snapshot["object"] = "hipengine.session.other"
    elif corruption == "resident_state_reuse":
        snapshot["resident_state_reuse"] = True
    elif corruption == "session_resident_state_reuse":
        snapshot["session"]["resident_state_reuse"] = True
    elif corruption == "session_includes_transcript":
        snapshot["session"]["includes_transcript"] = False
    else:
        raise AssertionError(f"unhandled corruption case: {corruption}")
    client.delete("/v1/hipengine/sessions/sess_corrupt_envelope", headers=headers)

    restored = client.post(
        "/v1/hipengine/sessions/sess_corrupt_envelope/snapshot",
        headers=headers,
        json=snapshot,
    )

    assert created.status_code == 200
    assert restored.status_code == 400
    assert restored.json()["error"]["code"] == "invalid_request"
    assert restored.json()["error"]["param"] == param
    assert "sess_corrupt_envelope" not in app.state.hipengine_chat_sessions


def test_chat_session_snapshot_restore_rejects_corrupted_message_shape() -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            api_key="secret",
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    created = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "snapshot prompt"}],
            "session": {"id": "sess_corrupt"},
            "max_tokens": 4,
        },
    )
    snapshot = client.get("/v1/hipengine/sessions/sess_corrupt/snapshot", headers=headers).json()
    snapshot["messages"][0]["unexpected"] = True
    client.delete("/v1/hipengine/sessions/sess_corrupt", headers=headers)

    restored = client.post(
        "/v1/hipengine/sessions/sess_corrupt/snapshot",
        headers=headers,
        json=snapshot,
    )

    assert created.status_code == 200
    assert restored.status_code == 400
    assert restored.json()["error"]["code"] == "invalid_request"
    assert restored.json()["error"]["param"] == "messages[0].unexpected"
    assert "sess_corrupt" not in app.state.hipengine_chat_sessions


@pytest.mark.parametrize(
    ("corruption", "param"),
    [
        ("content_object", "messages[0].content"),
        ("content_part_non_object", "messages[0].content[0]"),
        ("content_part_wrong_type", "messages[0].content[0]"),
        ("content_part_non_string_text", "messages[0].content[0].text"),
        ("unsupported_role", "messages[0].role"),
        ("user_tool_calls", "messages[0].tool_calls"),
        ("assistant_tool_call_id", "messages[1].tool_call_id"),
        ("tool_missing_tool_call_id", "messages[2].tool_call_id"),
        ("tool_with_tool_calls", "messages[2].tool_calls"),
        ("tool_unknown_tool_call_id", "messages[2].tool_call_id"),
        ("duplicate_tool_result", "messages[3].tool_call_id"),
        ("duplicate_tool_call_id", "messages[1].tool_calls[1].id"),
        ("tool_result_skipped_by_user", "messages[2].role"),
        ("empty_name", "messages[0].name"),
        ("non_string_tool_call_id", "messages[0].tool_call_id"),
    ],
)
def test_chat_session_snapshot_restore_rejects_corrupted_message_fields(
    corruption: str,
    param: str,
) -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            api_key="secret",
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    created = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [
                {"role": "user", "content": "snapshot prompt"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_snap",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_snap", "content": "snapshot tool result"},
            ],
            "session": {"id": "sess_corrupt_fields"},
            "max_tokens": 4,
        },
    )
    snapshot = client.get("/v1/hipengine/sessions/sess_corrupt_fields/snapshot", headers=headers).json()

    if corruption == "content_object":
        snapshot["messages"][0]["content"] = {"text": "not an accepted content shape"}
    elif corruption == "content_part_non_object":
        snapshot["messages"][0]["content"] = [7]
    elif corruption == "content_part_wrong_type":
        snapshot["messages"][0]["content"] = [{"type": "image_url", "text": "not text"}]
    elif corruption == "content_part_non_string_text":
        snapshot["messages"][0]["content"] = [{"type": "text", "text": 7}]
    elif corruption == "unsupported_role":
        snapshot["messages"][0]["role"] = "critic"
    elif corruption == "user_tool_calls":
        snapshot["messages"][0]["tool_calls"] = snapshot["messages"][1]["tool_calls"]
    elif corruption == "assistant_tool_call_id":
        snapshot["messages"][1]["tool_call_id"] = "call_snap"
    elif corruption == "tool_missing_tool_call_id":
        del snapshot["messages"][2]["tool_call_id"]
    elif corruption == "tool_with_tool_calls":
        snapshot["messages"][2]["tool_calls"] = snapshot["messages"][1]["tool_calls"]
    elif corruption == "tool_unknown_tool_call_id":
        snapshot["messages"][2]["tool_call_id"] = "call_missing"
    elif corruption == "duplicate_tool_result":
        snapshot["messages"].insert(
            3,
            {"role": "tool", "tool_call_id": "call_snap", "content": "duplicate tool result"},
        )
    elif corruption == "duplicate_tool_call_id":
        snapshot["messages"][1]["tool_calls"].append(
            {
                "id": "call_snap",
                "type": "function",
                "function": {"name": "write", "arguments": "{}"},
            }
        )
    elif corruption == "tool_result_skipped_by_user":
        snapshot["messages"].insert(2, {"role": "user", "content": "skip the tool"})
    elif corruption == "empty_name":
        snapshot["messages"][0]["name"] = ""
    elif corruption == "non_string_tool_call_id":
        snapshot["messages"][0]["tool_call_id"] = 7
    else:
        raise AssertionError(f"unhandled corruption case: {corruption}")
    client.delete("/v1/hipengine/sessions/sess_corrupt_fields", headers=headers)

    restored = client.post(
        "/v1/hipengine/sessions/sess_corrupt_fields/snapshot",
        headers=headers,
        json=snapshot,
    )

    assert created.status_code == 200
    assert restored.status_code == 400
    assert restored.json()["error"]["code"] == "invalid_request"
    assert restored.json()["error"]["param"] == param
    assert "sess_corrupt_fields" not in app.state.hipengine_chat_sessions


@pytest.mark.parametrize(
    ("corruption", "param"),
    [
        ("non_object", "messages[1].tool_calls[0]"),
        ("extra_key", "messages[1].tool_calls[0].unexpected"),
        ("missing_id", "messages[1].tool_calls[0].id"),
        ("wrong_type", "messages[1].tool_calls[0].type"),
        ("missing_function", "messages[1].tool_calls[0].function"),
        ("function_extra_key", "messages[1].tool_calls[0].function.unexpected"),
        ("empty_name", "messages[1].tool_calls[0].function.name"),
        ("non_string_arguments", "messages[1].tool_calls[0].function.arguments"),
        ("invalid_json_arguments", "messages[1].tool_calls[0].function.arguments"),
    ],
)
def test_chat_session_snapshot_restore_rejects_corrupted_tool_call_shape(
    corruption: str,
    param: str,
) -> None:
    fake = SequentialFakeLLM(
        ['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>']
    )
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            api_key="secret",
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    created = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "snapshot tool call"}],
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
            "session": {"id": "sess_corrupt_tool"},
            "max_tokens": 4,
        },
    )
    snapshot = client.get("/v1/hipengine/sessions/sess_corrupt_tool/snapshot", headers=headers).json()
    tool_call = snapshot["messages"][1]["tool_calls"][0]

    if corruption == "non_object":
        snapshot["messages"][1]["tool_calls"][0] = "not-object"
    elif corruption == "extra_key":
        tool_call["unexpected"] = True
    elif corruption == "missing_id":
        tool_call["id"] = ""
    elif corruption == "wrong_type":
        tool_call["type"] = "custom"
    elif corruption == "missing_function":
        del tool_call["function"]
    elif corruption == "function_extra_key":
        tool_call["function"]["unexpected"] = True
    elif corruption == "empty_name":
        tool_call["function"]["name"] = ""
    elif corruption == "non_string_arguments":
        tool_call["function"]["arguments"] = {"path": "README.md"}
    elif corruption == "invalid_json_arguments":
        tool_call["function"]["arguments"] = '{"path":'
    else:
        raise AssertionError(f"unhandled corruption case: {corruption}")
    client.delete("/v1/hipengine/sessions/sess_corrupt_tool", headers=headers)

    restored = client.post(
        "/v1/hipengine/sessions/sess_corrupt_tool/snapshot",
        headers=headers,
        json=snapshot,
    )

    assert created.status_code == 200
    assert restored.status_code == 400
    assert restored.json()["error"]["code"] == "invalid_request"
    assert restored.json()["error"]["param"] == param
    assert "sess_corrupt_tool" not in app.state.hipengine_chat_sessions


def test_chat_session_snapshot_restore_rejects_new_session_when_cap_full() -> None:
    fake = SequentialFakeLLM(["stored answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            api_key="secret",
            max_chat_sessions=1,
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    created = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "snapshot prompt"}],
            "session": {"id": "sess_one"},
            "max_tokens": 4,
        },
    )
    snapshot = client.get("/v1/hipengine/sessions/sess_one/snapshot", headers=headers).json()
    snapshot["session"]["id"] = "sess_two"

    restored = client.post(
        "/v1/hipengine/sessions/sess_two/snapshot",
        headers=headers,
        json=snapshot,
    )

    assert created.status_code == 200
    assert restored.status_code == 429
    assert restored.headers["Retry-After"] == "1"
    assert restored.json()["error"]["code"] == "engine_busy"
    assert restored.json()["error"]["message"] == "chat session limit is full"
    assert restored.json()["error"]["hipengine"] == {
        "code": "engine_busy",
        "status_code": 429,
        "retryable": True,
        "routing": {
            **_routing_metadata(),
            "matched": True,
            "reason": "engine_busy",
            "overload_source": "chat_session_cap",
            "max_active_chat_sessions": 1,
        },
    }
    assert set(app.state.hipengine_chat_sessions) == {"sess_one"}
    assert app.state.hipengine_server_metrics.request_rejected_total == 1


def test_chat_session_cap_rejects_new_sessions_before_generation() -> None:
    fake = SequentialFakeLLM(["first answer", "existing answer", "after delete answer"])
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            max_chat_sessions=1,
        ),
        llm=fake,
    )
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "first"}],
            "session": {"id": "sess_one"},
            "max_tokens": 4,
        },
    )
    rejected = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "second"}],
            "session": {"id": "sess_two"},
            "max_tokens": 4,
        },
    )

    assert first.status_code == 200
    assert rejected.status_code == 429
    assert rejected.headers["retry-after"] == "1"
    assert rejected.json()["error"]["code"] == "engine_busy"
    assert rejected.json()["error"]["message"] == "chat session limit is full"
    assert rejected.json()["error"]["hipengine"] == {
        "code": "engine_busy",
        "status_code": 429,
        "retryable": True,
        "routing": {
            **_routing_metadata(),
            "matched": True,
            "reason": "engine_busy",
            "overload_source": "chat_session_cap",
            "max_active_chat_sessions": 1,
        },
    }
    assert "sess_two" not in app.state.hipengine_chat_sessions
    assert app.state.hipengine_server_metrics.request_rejected_total == 1
    assert len(fake.calls) == 1

    existing = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "again"}],
            "session": {"id": "sess_one"},
            "max_tokens": 4,
        },
    )
    assert existing.status_code == 200
    assert len(fake.calls) == 2

    deleted = client.delete("/v1/hipengine/sessions/sess_one")
    assert deleted.json()["deleted"] is True

    after_delete = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "after delete"}],
            "session": {"id": "sess_two"},
            "max_tokens": 4,
        },
    )
    assert after_delete.status_code == 200
    assert len(fake.calls) == 3
    ready = client.get("/ready").json()
    assert ready["sessions"]["active"] == 1
    assert ready["sessions"]["max_active"] == 1
    assert ready["sessions"]["pending_creations"] == 0


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


def test_generation_batcher_returns_scheduler_chunks_for_single_metadata_submission() -> None:
    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            outputs = super().generate_detailed(prompts, sampling_params)
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {"request_id": index, "token_index": 0, "token_id": 100 + index, "chunk": {"text": output.text}}
                    for index, output in enumerate(outputs)
                ]
            }
            return outputs

    async def run() -> None:
        fake = SchedulerChunkFakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.0,
        )

        result = await batcher.submit(
            ("one", "two"),
            sampling,
            detailed=True,
            include_batch_metadata=True,
        )

        assert isinstance(result, _QueuedBatchResult)
        assert [output.text for output in result.outputs] == ["generated:one", "generated:two"]
        assert result.scheduler_token_chunks == [
            {"request_id": 0, "token_index": 0, "token_id": 100, "chunk": {"text": "generated:one"}},
            {"request_id": 1, "token_index": 0, "token_id": 101, "chunk": {"text": "generated:two"}},
        ]

    asyncio.run(run())


def test_backend_scheduler_token_chunks_reads_wrapped_text_generator() -> None:
    source_chunks = [
        {
            "request_id": 0,
            "token_index": 0,
            "token_id": 101,
            "chunk": {"text": "A", "telemetry": {"decode_state": {"execution_path": "gguf_serial_host_sampler_decode"}}},
        }
    ]
    inner = SimpleNamespace(last_batch_generation={"scheduler_token_chunks": source_chunks})
    engine = SimpleNamespace(_text_generator=SimpleNamespace(inner=inner))

    chunks = _backend_scheduler_token_chunks(engine)

    assert chunks == source_chunks
    assert chunks is not None
    assert chunks[0] is not source_chunks[0]
    assert chunks[0]["chunk"] is not source_chunks[0]["chunk"]


def test_generation_batcher_drops_scheduler_chunks_for_coalesced_metadata_submissions() -> None:
    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            outputs = super().generate_detailed(prompts, sampling_params)
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {"request_id": index, "token_index": 0, "token_id": 100 + index, "chunk": {"text": output.text}}
                    for index, output in enumerate(outputs)
                ]
            }
            return outputs

    async def run() -> None:
        fake = SchedulerChunkFakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.001,
        )

        first, second = await asyncio.gather(
            batcher.submit(("one",), sampling, detailed=True, include_batch_metadata=True),
            batcher.submit(("two",), sampling, detailed=True, include_batch_metadata=True),
        )

        assert isinstance(first, _QueuedBatchResult)
        assert isinstance(second, _QueuedBatchResult)
        assert [output.text for output in first.outputs] == ["generated:one"]
        assert [output.text for output in second.outputs] == ["generated:two"]
        assert first.scheduler_token_chunks is None
        assert second.scheduler_token_chunks is None
        assert fake.calls == [(("one", "two"), sampling)]

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
        assert [chunk.text for chunk in streamed] == ["generated:three"]
        assert all(isinstance(chunk, GenerationStreamChunk) for chunk in streamed)
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

        async def collect_stream() -> list[GenerationStreamChunk]:
            return [chunk async for chunk in batcher.stream(("stream",), sampling)]

        streamed, submitted = await asyncio.gather(
            collect_stream(),
            batcher.submit(("batch",), sampling),
        )

        assert [chunk.text for chunk in streamed] == ["generated:stream"]
        assert all(isinstance(chunk, GenerationStreamChunk) for chunk in streamed)
        assert submitted == ["generated:batch"]
        assert len(fake.calls) == 1
        assert set(fake.calls[0][0]) == {"stream", "batch"}
        assert fake.calls[0][1] == sampling

    asyncio.run(run())


def test_generation_batcher_stream_generate_only_fallback_yields_chunks() -> None:
    class GenerateOnlyLLM:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[str, ...], SamplingParams]] = []

        def generate(self, prompts, sampling_params: SamplingParams) -> list[str]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            return [f"generated:{prompt}" for prompt in prompt_tuple]

    async def run() -> None:
        fake = GenerateOnlyLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.0,
        )

        streamed = [chunk async for chunk in batcher.stream(("fallback",), sampling)]

        assert [chunk.text for chunk in streamed] == ["generated:fallback"]
        assert all(isinstance(chunk, GenerationStreamChunk) for chunk in streamed)
        assert fake.calls == [(("fallback",), sampling)]

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
            await batcher.submit(
                ("two",),
                sampling,
                error_extra={
                    "hipengine": {
                        "routing": {
                            "reason": "engine_busy",
                            "overload_source": "generation_queue_cap",
                        }
                    }
                },
            )

        exc = exc_info.value
        assert exc.status_code == 429
        assert exc.code == "engine_busy"
        assert exc.extra == {
            "hipengine": {
                "routing": {
                    "reason": "engine_busy",
                    "overload_source": "generation_queue_cap",
                }
            }
        }
        assert exc.headers == {"Retry-After": "2"}
        assert await first == ["generated:one"]
        assert fake.calls == [(("one",), sampling)]

    asyncio.run(run())


def test_generation_batcher_limits_active_request_group_size() -> None:
    async def run() -> None:
        fake = FakeLLM()
        sampling = SamplingParams(max_tokens=2)
        batcher = _GenerationBatcher(
            engine_factory=lambda: fake,
            batch_window_seconds=0.01,
            max_active_requests=1,
        )

        first = asyncio.create_task(batcher.submit(("one",), sampling))
        await asyncio.sleep(0)
        second = asyncio.create_task(batcher.submit(("two",), sampling))

        first_result, second_result = await asyncio.gather(first, second)

        assert first_result == ["generated:one"]
        assert second_result == ["generated:two"]
        assert fake.calls == [(("one",), sampling), (("two",), sampling)]
        assert batcher.active_requests() == 0
        assert batcher.max_active_requests() == 1

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
    assert body["hipengine"]["routing"] == _routing_metadata()
    assert [choice["text"] for choice in body["choices"]] == ["alpha", "beta"]
    assert [choice["finish_reason"] for choice in body["choices"]] == ["stop", "stop"]
    assert [choice["finish_details"] for choice in body["choices"]] == [_stateless_finish_details("stop"), _stateless_finish_details("stop")]
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
        json={"model": "fake-model", "prompt": ["one", "two"], "max_tokens": 2, "logit_bias": {"12": -1.0}},
    )

    assert response.status_code == 200
    choices = response.json()["choices"]
    assert [choice["finish_reason"] for choice in choices] == ["stop", "length"]
    assert choices[0]["finish_details"] == _stateless_finish_details(
        "eos",
        eos_token_id=151645,
        sampler_mode="greedy_fast",
    )
    assert choices[1]["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=2,
        budget_pressure="answer_budget",
        continuation_eligible=False,
    )


def test_completions_expose_backend_generation_telemetry() -> None:
    fake = DetailedGenerateFakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="answer",
                finish_details=FinishDetails(reason="eos", eos_token_id=151645, sampler_mode="greedy_fast"),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=3,
                    generated_tokens=1,
                    row_index=0,
                    phase="done",
                    sampler_mode="greedy_fast",
                    forced_token_id=42,
                    forced_token_reason="tool_choice_required",
                    forced_tokens_remaining=0,
                    active_processors=("logit_bias",),
                    sampler_fast_path_blockers=("logit_bias",),
                    timing={"prefill_ms": 2.5, "decode_ms": 1.25},
                    usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "one two three", "max_tokens": 1},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == "answer"
    assert choice["hipengine"] == {
        "decode_state": {
            "row_index": 0,
            "step_index": 1,
            "prompt_tokens": 3,
            "generated_tokens": 1,
            "phase": "done",
            "continuation_eligible": False,
            "forced_token_id": 42,
            "forced_token_reason": "tool_choice_required",
            "forced_tokens_remaining": 0,
            "active_processors": ["logit_bias"],
            "sampler_fast_path_blockers": ["logit_bias"],
            "sampler_mode": "greedy_fast",
        },
        "timing": {"prefill_ms": 2.5, "decode_ms": 1.25},
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        "finish_details": _stateless_finish_details(
            "eos",
            eos_token_id=151645,
            sampler_mode="greedy_fast",
        ),
    }


def test_streaming_completion_n_uses_scheduler_token_chunks_for_buffered_deltas() -> None:
    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            outputs = [
                GenerationOutput(
                    text="AB",
                    finish_details=FinishDetails(reason="length", length_limit=2, sampler_mode="greedy_fast"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=2,
                        row_index=0,
                        phase="done",
                        sampler_mode="greedy_fast",
                    ),
                ),
                GenerationOutput(
                    text="CD",
                    finish_details=FinishDetails(reason="length", length_limit=2, sampler_mode="greedy_fast"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=2,
                        row_index=1,
                        phase="done",
                        sampler_mode="greedy_fast",
                    ),
                ),
            ]
            chunk_texts = (("A", "B"), ("C", "D"))
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {
                        "request_id": request_id,
                        "token_index": token_index,
                        "token_id": 100 + request_id * 10 + token_index,
                        "finished": token_index == 1,
                        "chunk": {
                            "text": text,
                            "telemetry": GenerationTelemetry.from_decode_counts(
                                prompt_tokens=1,
                                generated_tokens=token_index + 1,
                                row_index=request_id,
                                request_id=str(request_id),
                                phase="answer",
                                sampler_mode="greedy_fast",
                                execution_path="scheduler_native_packed_prefill_serial_decode",
                            ).to_json_dict(),
                        },
                    }
                    for request_id, row in enumerate(chunk_texts)
                    for token_index, text in enumerate(row)
                ]
            }
            return outputs[: len(prompt_tuple)]

    fake = SchedulerChunkFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "one",
            "n": 2,
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    deltas = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] is None
    ]
    assert [(choice["index"], choice["text"]) for choice in deltas] == [
        (0, "A"),
        (0, "B"),
        (1, "C"),
        (1, "D"),
    ]
    assert [
        choice["hipengine"]["decode_state"]["generated_tokens"]
        for choice in deltas
    ] == [1, 2, 1, 2]
    assert [
        choice["hipengine"]["decode_state"]["execution_path"]
        for choice in deltas
    ] == [
        "scheduler_native_packed_prefill_serial_decode",
        "scheduler_native_packed_prefill_serial_decode",
        "scheduler_native_packed_prefill_serial_decode",
        "scheduler_native_packed_prefill_serial_decode",
    ]
    done = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] == "length"
    ]
    assert [(choice["index"], choice["finish_details"]) for choice in done] == [
        (
            0,
            {
                "reason": "length",
                "length_limit": 2,
                "cache_action": "append_none",
                "sampler_mode": "greedy_fast",
                "continuation_eligible": False,
            },
        ),
        (
            1,
            {
                "reason": "length",
                "length_limit": 2,
                "cache_action": "append_none",
                "sampler_mode": "greedy_fast",
                "continuation_eligible": False,
            },
        ),
    ]
    assert payloads[-1]["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
    assert "data: [DONE]" in response.text
    assert fake.calls[0][0] == ("one", "one")
    assert fake.stream_calls == []


def test_completion_continuation_resumes_buffered_length_finish_once() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="alpha",
                finish_details=FinishDetails(reason="length", length_limit=1),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=2,
                    generated_tokens=1,
                    sampler_mode="greedy_fast",
                ),
            ),
            GenerationOutput(text=" beta", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "temperature": 0.0},
    )

    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    continuation_id = first_choice["continuation_id"]
    assert continuation_id.startswith("gen_")
    assert first_choice["text"] == "alpha"
    assert first_choice["finish_reason"] == "length"
    assert first_choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=1,
        continuation_eligible=True,
        continuation_id=continuation_id,
    )
    assert first_choice["hipengine"]["decode_state"] == {
        "row_index": 0,
        "step_index": 1,
        "prompt_tokens": 2,
        "generated_tokens": 1,
        "phase": "done",
        "continuation_eligible": True,
        "sampler_mode": "greedy_fast",
    }

    second = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert second.status_code == 200
    second_choice = second.json()["choices"][0]
    assert second_choice["text"] == "alpha beta"
    assert second_choice["finish_reason"] == "stop"
    assert second_choice["finish_details"] == _stateless_finish_details("eos", eos_token_id=151645)
    assert fake.calls[0][0] == ("Say: ",)
    assert fake.calls[1][0] == ("Say: alpha",)

    reused = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 1},
    )

    assert reused.status_code == 400
    assert reused.json()["error"]["code"] == "invalid_continuation"
    assert len(fake.calls) == 2


def test_completion_continuation_is_scoped_to_auth_principal() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="alpha",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" beta", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            api_key="secret",
        ),
        llm=fake,
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}

    first = client.post(
        "/v1/completions",
        headers=headers,
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "temperature": 0.0},
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]
    record = app.state.hipengine_continuations[continuation_id]
    assert record.auth_principal.startswith("bearer_sha256:")
    assert "secret" not in record.auth_principal

    app.state.hipengine_continuations[continuation_id] = replace(
        record,
        auth_principal="bearer_sha256:other",
    )
    rejected = client.post(
        "/v1/completions",
        headers=headers,
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "invalid_continuation"
    assert rejected.json()["error"]["param"] == "continuation_id"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    app.state.hipengine_continuations[continuation_id] = record
    resumed = client.post(
        "/v1/completions",
        headers=headers,
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert resumed.status_code == 200
    assert resumed.json()["choices"][0]["text"] == "alpha beta"
    assert continuation_id not in app.state.hipengine_continuations
    assert len(fake.calls) == 2


def test_completion_continuation_is_scoped_to_session_id() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="alpha",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" beta", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "temperature": 0.0},
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]
    record = app.state.hipengine_continuations[continuation_id]
    assert record.session_id is None

    app.state.hipengine_continuations[continuation_id] = replace(record, session_id="other-session")
    rejected = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "invalid_continuation"
    assert rejected.json()["error"]["param"] == "continuation_id"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    app.state.hipengine_continuations[continuation_id] = record
    resumed = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert resumed.status_code == 200
    assert resumed.json()["choices"][0]["text"] == "alpha beta"
    assert continuation_id not in app.state.hipengine_continuations
    assert len(fake.calls) == 2


def test_completion_continuation_is_scoped_to_tokenizer_metadata() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="alpha",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" beta", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "temperature": 0.0},
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]
    record = app.state.hipengine_continuations[continuation_id]
    assert record.tokenizer == {
        "name": "SequentialFakeLLM",
        "tokenize": True,
        "detokenize": True,
        "count_tokens": True,
    }

    app.state.hipengine_continuations[continuation_id] = replace(
        record,
        tokenizer={**record.tokenizer, "name": "OtherTokenizer"},
    )
    rejected = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "invalid_continuation"
    assert rejected.json()["error"]["param"] == "continuation_id"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    app.state.hipengine_continuations[continuation_id] = record
    resumed = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert resumed.status_code == 200
    assert resumed.json()["choices"][0]["text"] == "alpha beta"
    assert continuation_id not in app.state.hipengine_continuations
    assert len(fake.calls) == 2


def test_completion_continuation_resume_rejects_explicit_response_format_override() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text='{"ok":',
                finish_details=FinishDetails(reason="length", length_limit=6),
            ),
            GenerationOutput(text="true}", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "Return JSON: ",
            "response_format": {"type": "json_object"},
            "max_tokens": 6,
        },
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    override = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "response_format": {"type": "text"},
            "max_tokens": 4,
        },
    )

    assert override.status_code == 400
    assert override.json()["error"]["code"] == "unsupported_parameter"
    assert override.json()["error"]["param"] == "response_format"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    inherited = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert inherited.status_code == 200
    assert inherited.json()["choices"][0]["text"] == '{"ok":true}'
    assert len(fake.calls) == 2


def test_completion_continuation_resume_rejects_prompt_without_consuming_handle() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="alpha",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" beta", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "temperature": 0.0},
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    with_prompt = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "prompt": "ignored prompt",
            "max_tokens": 4,
        },
    )

    assert with_prompt.status_code == 400
    assert with_prompt.json()["error"]["code"] == "unsupported_parameter"
    assert with_prompt.json()["error"]["param"] == "prompt"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    resumed = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert resumed.status_code == 200
    assert resumed.json()["choices"][0]["text"] == "alpha beta"
    assert len(fake.calls) == 2


def test_completion_length_finish_with_stop_is_continuation_ineligible() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="length", length_limit=1, continuation_eligible=True),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=2,
                    generated_tokens=1,
                    sampler_mode="greedy_fast",
                    continuation_eligible=True,
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "Say: ",
            "stop": "<end>",
            "max_tokens": 1,
            "temperature": 0.0,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert "continuation_id" not in choice
    assert choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=1,
        continuation_eligible=False,
    )
    assert choice["hipengine"]["decode_state"] == {
        "row_index": 0,
        "step_index": 1,
        "prompt_tokens": 2,
        "generated_tokens": 1,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_mode": "greedy_fast",
    }


def test_completion_length_finish_honors_backend_continuation_ineligible() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="length", length_limit=1, continuation_eligible=False),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=2,
                    generated_tokens=1,
                    sampler_mode="greedy_fast",
                    continuation_eligible=True,
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "temperature": 0.0},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == "partial"
    assert "continuation_id" not in choice
    assert choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=1,
        continuation_eligible=False,
    )
    assert choice["hipengine"]["decode_state"] == {
        "row_index": 0,
        "step_index": 1,
        "prompt_tokens": 2,
        "generated_tokens": 1,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_mode": "greedy_fast",
    }


def test_completion_length_finish_with_ignore_eos_is_continuation_ineligible() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="length", length_limit=1),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=2,
                    generated_tokens=1,
                    sampler_mode="greedy_fast",
                    continuation_eligible=True,
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "Say: ",
            "ignore_eos": True,
            "max_tokens": 1,
            "temperature": 0.0,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert "continuation_id" not in choice
    assert choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=1,
        continuation_eligible=False,
    )
    assert choice["hipengine"]["decode_state"] == {
        "row_index": 0,
        "step_index": 1,
        "prompt_tokens": 2,
        "generated_tokens": 1,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_mode": "greedy_fast",
    }


def test_completion_continuation_resume_rejects_explicit_stop_without_consuming_handle() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="alpha",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" beta", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "temperature": 0.0},
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    with_stop = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "stop": "<end>",
            "max_tokens": 4,
        },
    )

    assert with_stop.status_code == 400
    assert with_stop.json()["error"]["code"] == "unsupported_parameter"
    assert with_stop.json()["error"]["param"] == "stop"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    resumed = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert resumed.status_code == 200
    assert resumed.json()["choices"][0]["text"] == "alpha beta"
    assert len(fake.calls) == 2


def test_completion_continuation_resume_rejects_ignore_eos_without_consuming_handle() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="alpha",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" beta", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "temperature": 0.0},
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    with_ignore_eos = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "ignore_eos": True,
            "max_tokens": 4,
        },
    )

    assert with_ignore_eos.status_code == 400
    assert with_ignore_eos.json()["error"]["code"] == "unsupported_parameter"
    assert with_ignore_eos.json()["error"]["param"] == "ignore_eos"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    resumed = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert resumed.status_code == 200
    assert resumed.json()["choices"][0]["text"] == "alpha beta"
    assert len(fake.calls) == 2


def test_completion_length_finish_marks_sampled_continuation_ineligible() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="sampled partial",
                finish_details=FinishDetails(reason="length", length_limit=2),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "sample", "max_tokens": 2, "temperature": 0.7},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert "continuation_id" not in choice
    assert choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=2,
        continuation_eligible=False,
    )


def test_completion_length_finish_with_n_gt_one_is_continuation_ineligible() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="partial one",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(
                text="partial two",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "Say: ", "max_tokens": 1, "n": 2},
    )

    assert response.status_code == 200
    choices = response.json()["choices"]
    assert len(choices) == 2
    for choice in choices:
        assert "continuation_id" not in choice
        assert choice["finish_details"] == _stateless_finish_details(
            "length",
            length_limit=1,
            continuation_eligible=False,
        )


def test_chat_length_finish_with_n_gt_one_is_continuation_ineligible() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="partial one",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(
                text="partial two",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "max_tokens": 1,
            "n": 2,
        },
    )

    assert response.status_code == 200
    choices = response.json()["choices"]
    assert len(choices) == 2
    for choice in choices:
        assert "continuation_id" not in choice
        assert choice["finish_details"] == _stateless_finish_details(
            "length",
            length_limit=1,
            phase="answer",
            continuation_eligible=False,
        )


def test_chat_length_finish_with_reasoning_effort_is_continuation_ineligible() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="partial answer",
                finish_details=FinishDetails(reason="length", length_limit=2),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "max_tokens": 16,
            "reasoning_effort": "low",
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert "continuation_id" not in choice
    assert choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=2,
        phase="answer",
        continuation_eligible=False,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reasoning_effort", "low"),
        ("max_think_tokens", 16),
        ("min_answer_tokens", 4),
        ("hard_think_cap", 16),
        ("soft_close_window", 4),
        ("hard_close_message", "answer now"),
        ("hard_close_sequence", "</think>\n"),
        ("thinking_token_budget", 16),
        ("chat_template_kwargs", {"thinking_budget": 16}),
        ("thinking", {"budget_tokens": 16}),
        ("reasoning", {"allow_unbounded": True}),
    ],
)
def test_chat_continuation_resume_rejects_thinking_controls_without_consuming_handle(
    field: str,
    value: Any,
) -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="partial answer",
                finish_details=FinishDetails(reason="length", length_limit=2),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "max_tokens": 2,
        },
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    capabilities = client.get("/v1/hipengine/capabilities").json()
    assert field in capabilities["sessions"]["continuations"]["unsupported_resume_fields"]
    resumed = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "max_tokens": 4,
            field: value,
        },
    )

    assert resumed.status_code == 400
    assert resumed.json()["error"]["code"] == "unsupported_parameter"
    assert resumed.json()["error"]["param"] == field
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1


def test_chat_continuation_resume_rejects_messages_without_consuming_handle() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" answer", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "max_tokens": 1,
        },
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    with_messages = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "messages": [{"role": "user", "content": "ignored follow-up"}],
            "max_tokens": 4,
        },
    )

    assert with_messages.status_code == 400
    assert with_messages.json()["error"]["code"] == "unsupported_parameter"
    assert with_messages.json()["error"]["param"] == "messages"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    resumed = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert resumed.status_code == 200
    assert resumed.json()["choices"][0]["message"]["content"] == "partial answer"
    assert len(fake.calls) == 2


@pytest.mark.parametrize(
    ("field", "value", "expected_param"),
    [
        ("stream", True, "stream"),
        ("n", 2, "n"),
    ],
)
def test_chat_continuation_resume_rejects_lower_loop_fields_without_consuming_handle(
    field: str,
    value: Any,
    expected_param: str,
) -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" answer", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "max_tokens": 1,
        },
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    capabilities = client.get("/v1/hipengine/capabilities").json()
    assert expected_param in capabilities["sessions"]["continuations"]["unsupported_resume_fields"]
    rejected = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "max_tokens": 4,
            field: value,
        },
    )

    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "unsupported_parameter"
    assert rejected.json()["error"]["param"] == expected_param
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    resumed = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert resumed.status_code == 200
    assert resumed.json()["choices"][0]["message"]["content"] == "partial answer"
    assert continuation_id not in app.state.hipengine_continuations
    assert len(fake.calls) == 2


def test_chat_session_continuation_resumes_buffered_length_finish_once() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" answer", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "session": {"id": "sess_continue"},
            "max_tokens": 1,
            "temperature": 0.0,
        },
    )

    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    continuation_id = first_choice["continuation_id"]
    assert continuation_id.startswith("gen_")
    assert first_choice["finish_reason"] == "length"
    assert first_choice["finish_details"] == {
        "reason": "length",
        "length_limit": 1,
        "cache_action": "append_prompt_only",
        "phase": "answer",
        "continuation_eligible": True,
        "continuation_id": continuation_id,
    }
    assert app.state.hipengine_continuations[continuation_id].session_id == "sess_continue"
    assert app.state.hipengine_chat_sessions["sess_continue"].messages == (
        {"role": "user", "content": "continue"},
    )

    resumed = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "session": {"id": "sess_continue"},
            "max_tokens": 4,
        },
    )

    assert resumed.status_code == 200
    resumed_choice = resumed.json()["choices"][0]
    assert resumed_choice["message"]["content"] == "partial answer"
    assert resumed_choice["finish_reason"] == "stop"
    assert resumed_choice["finish_details"]["cache_action"] == "append_visible_only"
    assert continuation_id not in app.state.hipengine_continuations
    assert app.state.hipengine_chat_sessions["sess_continue"].messages == (
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "partial answer"},
    )
    assert "partial" in fake.calls[1][0][0]

    reused = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "session": {"id": "sess_continue"},
            "max_tokens": 1,
        },
    )

    assert reused.status_code == 400
    assert reused.json()["error"]["code"] == "invalid_continuation"
    assert len(fake.calls) == 2


def test_chat_session_continuation_rejects_deleted_session_without_generation() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="length", length_limit=1),
            ),
            GenerationOutput(text=" should not run", finish_details=FinishDetails(reason="eos")),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "session": {"id": "sess_deleted"},
            "max_tokens": 1,
        },
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    deleted = client.delete("/v1/hipengine/sessions/sess_deleted")
    rejected = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "session": {"id": "sess_deleted"},
            "max_tokens": 4,
        },
    )

    assert deleted.json()["deleted"] is True
    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "invalid_continuation"
    assert rejected.json()["error"]["param"] == "continuation_id"
    assert continuation_id in app.state.hipengine_continuations
    assert "sess_deleted" not in app.state.hipengine_chat_sessions
    assert len(fake.calls) == 1


def test_completion_continuation_expiration_reports_stable_error() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="partial",
                finish_details=FinishDetails(reason="length", length_limit=1),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "start", "max_tokens": 1},
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]
    record = app.state.hipengine_continuations[continuation_id]
    app.state.hipengine_continuations[continuation_id] = replace(record, expires_at=time.time() - 1)

    expired = client.post(
        "/v1/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 1},
    )

    assert expired.status_code == 410
    assert expired.json()["error"]["code"] == "continuation_expired"
    assert expired.json()["error"]["param"] == "continuation_id"
    assert len(fake.calls) == 1


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
def test_stateless_session_default_reports_append_none_cache_action(endpoint, payload) -> None:
    fake = FakeLLM(outputs=["reply"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(endpoint, json=payload)

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


def test_chat_session_visible_only_prepends_stored_transcript_and_commits_visible_answer() -> None:
    fake = SequentialFakeLLM(["first answer", "second answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "session": {"id": "sess_visible", "commit": "append_visible_only"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "again"}],
            "max_tokens": 2,
            "session": {"id": "sess_visible", "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["choices"][0]["finish_details"]["cache_action"] == "append_visible_only"
    assert second.json()["choices"][0]["finish_details"]["cache_action"] == "append_none"
    prompt = fake.calls[1][0][0]
    assert prompt.index("hello") < prompt.index("first answer") < prompt.index("again")
    record = app.state.hipengine_chat_sessions["sess_visible"]
    assert record.messages == (
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "first answer"},
    )


def test_chat_session_visible_only_strips_hidden_reasoning_from_next_prompt() -> None:
    fake = SequentialFakeLLM(["<think>secret plan</think>visible answer", "done"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "max_tokens": 2,
            "session": {"id": "sess_reasoning"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "max_tokens": 2,
            "session": {"id": "sess_reasoning", "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    choice = first.json()["choices"][0]
    assert choice["message"] == {
        "role": "assistant",
        "content": "visible answer",
        "reasoning_content": "secret plan",
    }
    assert choice["finish_details"]["cache_action"] == "append_visible_only"
    prompt = fake.calls[1][0][0]
    assert "visible answer" in prompt
    assert "secret plan" not in prompt
    assert "<think>secret plan</think>" not in prompt


def test_chat_session_append_all_retains_raw_generated_text_for_debug() -> None:
    fake = SequentialFakeLLM(["<think>secret plan</think>visible answer", "done"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "debug"}],
            "max_tokens": 2,
            "session": {"id": "sess_all", "commit": "append_all"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "max_tokens": 2,
            "session": {"id": "sess_all", "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["choices"][0]["finish_details"]["cache_action"] == "append_all"
    assert "<think>secret plan</think>visible answer" in fake.calls[1][0][0]


def test_chat_session_visible_only_downgrades_length_finish_to_prompt_only() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(text="partial answer", finish_details=FinishDetails(reason="length")),
            "done",
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "start"}],
            "max_tokens": 1,
            "session": {"id": "sess_length", "commit": "append_visible_only"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "next"}],
            "max_tokens": 2,
            "session": {"id": "sess_length", "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_choice = first.json()["choices"][0]
    continuation_id = first_choice["continuation_id"]
    assert first_choice["finish_details"]["cache_action"] == "append_prompt_only"
    assert first_choice["finish_details"]["continuation_eligible"] is True
    assert first_choice["finish_details"]["continuation_id"] == continuation_id
    prompt = fake.calls[1][0][0]
    assert "start" in prompt
    assert "next" in prompt
    assert "partial answer" not in prompt
    record = app.state.hipengine_chat_sessions["sess_length"]
    assert record.messages == ({"role": "user", "content": "start"},)


@pytest.mark.parametrize(
    ("case", "first_output", "request_extra", "expected_reason", "rejected_text"),
    [
        (
            "invalid_tool_call",
            '<tool_call>{"name":"write","arguments":{"path":"README.md"}}</tool_call>',
            {},
            "invalid_tool_call",
            '"name":"write"',
        ),
        (
            "missing_required_tool",
            "ordinary answer",
            {"tool_choice": "required"},
            "tool_required_not_satisfied",
            "ordinary answer",
        ),
        (
            "schema_violation",
            '<tool_call>{"name":"read","arguments":{"path":7}}</tool_call>',
            {},
            "schema_violation",
            '"path":7',
        ),
    ],
)
def test_chat_session_visible_only_downgrades_strict_tool_failures_to_prompt_only(
    case: str,
    first_output: str,
    request_extra: dict[str, Any],
    expected_reason: str,
    rejected_text: str,
) -> None:
    fake = SequentialFakeLLM([first_output, "done"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
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
    session_id = f"sess_tool_failure_{case}"

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "try tool"}],
            "tools": tools,
            "max_tokens": 4,
            "session": {"id": session_id, "commit": "append_visible_only"},
            **request_extra,
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "tools": tools,
            "max_tokens": 4,
            "session": {"id": session_id, "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_choice = first.json()["choices"][0]
    assert first_choice["finish_details"]["reason"] == expected_reason
    assert first_choice["finish_details"]["cache_action"] == "append_prompt_only"
    prompt = fake.calls[1][0][0]
    assert "try tool" in prompt
    assert "continue" in prompt
    assert rejected_text not in prompt
    record = app.state.hipengine_chat_sessions[session_id]
    assert record.messages == ({"role": "user", "content": "try tool"},)


def test_chat_session_visible_only_downgrades_unparseable_tool_markup_to_prompt_only() -> None:
    raw_tool_markup = '<tool_call>{"name":"read","arguments":</tool_call>'
    fake = SequentialFakeLLM([raw_tool_markup, "done"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
        }
    ]

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "try tool"}],
            "tools": tools,
            "max_tokens": 4,
            "session": {"id": "sess_unparseable_tool", "commit": "append_visible_only"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "tools": tools,
            "max_tokens": 4,
            "session": {"id": "sess_unparseable_tool", "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_choice = first.json()["choices"][0]
    assert first_choice["finish_details"]["reason"] == "invalid_tool_call"
    assert first_choice["finish_details"]["cache_action"] == "append_prompt_only"
    prompt = fake.calls[1][0][0]
    assert "try tool" in prompt
    assert "continue" in prompt
    assert raw_tool_markup not in prompt
    record = app.state.hipengine_chat_sessions["sess_unparseable_tool"]
    assert record.messages == ({"role": "user", "content": "try tool"},)


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_finish_details", "request_extra"),
    [
        (
            "deadline",
            408,
            {
                "reason": "deadline_exceeded",
                "deadline_exceeded": True,
                "cache_action": "append_prompt_only",
            },
            {"timeout_ms": 5000},
        ),
        (
            "cancelled",
            499,
            {"reason": "cancelled", "cancelled": True, "cache_action": "append_prompt_only"},
            {},
        ),
    ],
)
def test_chat_session_visible_only_downgrades_backend_errors_to_prompt_only(
    error: str,
    expected_status: int,
    expected_finish_details: dict[str, Any],
    request_extra: dict[str, Any],
) -> None:
    fake = BackendErrorThenSequentialFakeLLM(error, ["done"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)
    session_id = f"sess_backend_error_{error}"

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "unsafe turn"}],
            "max_tokens": 4,
            "session": {"id": session_id, "commit": "append_visible_only"},
            **request_extra,
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "follow up"}],
            "max_tokens": 4,
            "session": {"id": session_id, "commit": "append_none"},
        },
    )

    assert first.status_code == expected_status
    assert first.json()["error"]["finish_details"] == expected_finish_details
    assert second.status_code == 200
    prompt = fake.calls[1][0][0]
    assert "unsafe turn" in prompt
    assert "follow up" in prompt
    assert "done" not in prompt
    record = app.state.hipengine_chat_sessions[session_id]
    assert record.messages == ({"role": "user", "content": "unsafe turn"},)


def test_chat_session_visible_only_commits_tool_calls_without_reasoning() -> None:
    fake = SequentialFakeLLM(
        [
            '<think>need file</think><tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>',
            "done",
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
        }
    ]

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read it"}],
            "tools": tools,
            "max_tokens": 4,
            "session": {"id": "sess_tool", "commit": "append_visible_only"},
        },
    )
    assert first.status_code == 200
    tool_call_id = first.json()["choices"][0]["message"]["tool_calls"][0]["id"]
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "tool", "tool_call_id": tool_call_id, "content": "file text"}],
            "tools": tools,
            "max_tokens": 4,
            "session": {"id": "sess_tool", "commit": "append_none"},
        },
    )

    assert second.status_code == 200
    assert first.json()["choices"][0]["finish_reason"] == "tool_calls"
    prompt = fake.calls[1][0][0]
    assert '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>' in prompt
    assert "<tool_response>\nfile text\n</tool_response>" in prompt
    assert "need file" not in prompt
    assert first.json()["choices"][0]["finish_details"]["cache_action"] == "append_visible_only"


@pytest.mark.parametrize(
    "response_format_mode",
    ["json_object", "json_schema"],
)
def test_chat_session_visible_only_downgrades_structured_output_failures_to_prompt_only(
    response_format_mode: str,
) -> None:
    response_format = (
        {"type": "json_object"}
        if response_format_mode == "json_object"
        else {"type": "json_schema", "json_schema": _response_json_schema()}
    )
    fake = SequentialFakeLLM(["not json", "done"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": response_format,
            "max_tokens": 4,
            "session": {"id": "sess_structured_failure", "commit": "append_visible_only"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "max_tokens": 4,
            "session": {"id": "sess_structured_failure", "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_choice = first.json()["choices"][0]
    assert first_choice["message"] == {"role": "assistant", "content": ""}
    assert first_choice["finish_details"]["reason"] == "schema_violation"
    assert first_choice["finish_details"]["cache_action"] == "append_prompt_only"
    prompt = fake.calls[1][0][0]
    assert "return json" in prompt
    assert "continue" in prompt
    assert "not json" not in prompt
    record = app.state.hipengine_chat_sessions["sess_structured_failure"]
    assert record.messages == ({"role": "user", "content": "return json"},)


def test_chat_session_visible_only_downgrades_synthetic_output_to_prompt_only() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="synthetic repair",
                finish_details=FinishDetails(reason="stop", synthetic_tokens=2),
            ),
            "done",
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "repair"}],
            "max_tokens": 4,
            "session": {"id": "sess_synthetic", "commit": "append_visible_only"},
        },
    )
    second = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "continue"}],
            "max_tokens": 4,
            "session": {"id": "sess_synthetic", "commit": "append_none"},
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_choice = first.json()["choices"][0]
    assert first_choice["message"] == {"role": "assistant", "content": "synthetic repair"}
    assert first_choice["finish_details"]["reason"] == "stop"
    assert first_choice["finish_details"]["synthetic_tokens"] == 2
    assert first_choice["finish_details"]["cache_action"] == "append_prompt_only"
    prompt = fake.calls[1][0][0]
    assert "repair" in prompt
    assert "continue" in prompt
    assert "synthetic repair" not in prompt
    record = app.state.hipengine_chat_sessions["sess_synthetic"]
    assert record.messages == ({"role": "user", "content": "repair"},)


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
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


@pytest.mark.parametrize(
    ("payload", "output", "expected"),
    [
        ({"response_format": {"type": "json_object"}}, '{"ok":true}', True),
        (
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "object_result",
                        "schema": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}},
                            "required": ["ok"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
            '{"ok":true}',
            True,
        ),
        (
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "array_result",
                        "schema": {"type": "array", "items": {"type": "integer"}},
                    },
                },
            },
            "[1]",
            False,
        ),
        ({"guided_json": True}, '{"ok":true}', True),
        (
            {
                "guided_json": {
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                },
            },
            '{"ok":true}',
            True,
        ),
    ],
)
def test_completions_structured_json_lowers_close_forcing(
    payload: dict[str, Any],
    output: str,
    expected: bool,
) -> None:
    fake = FakeLLM(outputs=[output])
    client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=fake,
        )
    )

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            **payload,
        },
    )

    assert response.status_code == 200
    assert fake.calls[0][1].json_object_close_forcing is expected
    assert fake.calls[0][1].max_tokens == 16


def test_completions_response_format_length_rejects_invalid_json_continuation() -> None:
    client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(
                detailed_outputs=[
                    GenerationOutput(
                        text='{"ok": [1}',
                        finish_details=FinishDetails(reason="length", length_limit=9),
                    )
                ]
            ),
        )
    )

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {"type": "json_object"},
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == '{"ok": [1}'
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "schema_violation",
        length_limit=9,
        phase="structured",
        continuation_eligible=False,
    )
    assert "continuation_id" not in choice


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


def _unified_diff_text() -> str:
    return "\n".join(
        [
            "diff --git a/README.md b/README.md",
            "index 1111111..2222222 100644",
            "--- a/README.md",
            "+++ b/README.md",
            "@@ -1,2 +1,2 @@",
            " hello",
            "-old",
            "+new",
        ]
    )


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
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_string_pattern() -> None:
    schema = _response_json_schema()
    schema["schema"]["properties"]["path"]["pattern"] = r"^README[.]md$"
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"path":"README.md"}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"path":"WORKLOG.md"}']),
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
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_object_property_count() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "minProperties": 2,
            "maxProperties": 2,
            "properties": {
                "ok": {"type": "boolean"},
                "path": {"type": "string"},
                "extra": {"type": "string"},
            },
        },
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"path":"README.md"}']),
        )
    )
    too_few_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true}']),
        )
    )
    too_many_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"path":"README.md","extra":"x"}']),
        )
    )

    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid = valid_client.post("/v1/completions", json=payload)
    too_few = too_few_client.post("/v1/completions", json=payload)
    too_many = too_many_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"ok":true,"path":"README.md"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    for response in (too_few, too_many):
        assert response.status_code == 200
        choice = response.json()["choices"][0]
        assert choice["text"] == ""
        assert choice["finish_reason"] == "stop"
        assert choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_additional_properties_schema() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": {"type": "string"},
        },
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"note":"done"}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"note":7}']),
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
    assert valid.json()["choices"][0]["text"] == '{"ok":true,"note":"done"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_pattern_properties() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "patternProperties": {r"^x-[a-z]+$": {"type": "integer"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"x-count":2}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true,"x-count":"two"}']),
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
    assert valid.json()["choices"][0]["text"] == '{"ok":true,"x-count":2}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_object_name_keywords() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "propertyNames": {"pattern": r"^(id|name|x-[a-z]+)$"},
            "dependentRequired": {"id": ["name"]},
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
            },
        },
    }
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"id":"1","name":"doc","x-extra":"ok"}']),
        )
    )
    invalid_outputs = [
        '{"id":"1","name":"doc","bad-key":"no"}',
        '{"id":"1","x-extra":"ok"}',
    ]

    valid = valid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"id":"1","name":"doc","x-extra":"ok"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    for generated in invalid_outputs:
        invalid = TestClient(
            create_app(
                ServerConfig(model="fake-path", served_model_name="fake-model"),
                llm=FakeLLM(outputs=[generated]),
            )
        ).post("/v1/completions", json=payload)
        assert invalid.status_code == 200
        invalid_choice = invalid.json()["choices"][0]
        assert invalid_choice["text"] == ""
        assert invalid_choice["finish_reason"] == "stop"
        assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_dependent_schemas() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "properties": {
                "kind": {"enum": ["file", "url"]},
                "path": {"type": "string"},
                "href": {"type": "string", "pattern": r"^https://"},
            },
            "dependentSchemas": {
                "path": {
                    "required": ["kind"],
                    "properties": {"kind": {"const": "file"}},
                },
                "href": {
                    "required": ["kind"],
                    "properties": {"kind": {"const": "url"}},
                },
            },
            "additionalProperties": False,
        },
    }
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"kind":"file","path":"README.md"}']),
        )
    )
    invalid_outputs = [
        '{"kind":"url","path":"README.md"}',
        '{"href":"https://example.test"}',
    ]

    valid = valid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"kind":"file","path":"README.md"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    for generated in invalid_outputs:
        invalid = TestClient(
            create_app(
                ServerConfig(model="fake-path", served_model_name="fake-model"),
                llm=FakeLLM(outputs=[generated]),
            )
        ).post("/v1/completions", json=payload)
        assert invalid.status_code == 200
        invalid_choice = invalid.json()["choices"][0]
        assert invalid_choice["text"] == ""
        assert invalid_choice["finish_reason"] == "stop"
        assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_conditional_schemas() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "properties": {
                "kind": {"enum": ["file", "url"]},
                "path": {"type": "string"},
                "href": {"type": "string", "pattern": r"^https://"},
            },
            "if": {"required": ["kind"], "properties": {"kind": {"const": "file"}}},
            "then": {"required": ["path"], "not": {"required": ["href"]}},
            "else": {"required": ["href"], "not": {"required": ["path"]}},
            "additionalProperties": False,
        },
    }
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"kind":"file","path":"README.md"}']),
        )
    )
    invalid_outputs = [
        '{"kind":"file","href":"https://example.test"}',
        '{"kind":"url","path":"README.md"}',
    ]

    valid = valid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"kind":"file","path":"README.md"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    for generated in invalid_outputs:
        invalid = TestClient(
            create_app(
                ServerConfig(model="fake-path", served_model_name="fake-model"),
                llm=FakeLLM(outputs=[generated]),
            )
        ).post("/v1/completions", json=payload)
        assert invalid.status_code == 200
        invalid_choice = invalid.json()["choices"][0]
        assert invalid_choice["text"] == ""
        assert invalid_choice["finish_reason"] == "stop"
        assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_local_refs() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "$defs": {
                "file_ref": {"type": "string", "pattern": r"^[A-Z]+[.]md$"},
                "result_tag": {"enum": ["ok", "skip"]},
            },
            "properties": {
                "path": {"$ref": "#/$defs/file_ref"},
                "result": {"$ref": "#/$defs/result_tag"},
            },
            "required": ["path", "result"],
            "additionalProperties": False,
        },
    }
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"path":"README.md","result":"ok"}']),
        )
    )
    invalid_outputs = [
        '{"path":"readme.md","result":"ok"}',
        '{"path":"README.md","result":"done"}',
    ]

    valid = valid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"path":"README.md","result":"ok"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    for generated in invalid_outputs:
        invalid = TestClient(
            create_app(
                ServerConfig(model="fake-path", served_model_name="fake-model"),
                llm=FakeLLM(outputs=[generated]),
            )
        ).post("/v1/completions", json=payload)
        assert invalid.status_code == 200
        invalid_choice = invalid.json()["choices"][0]
        assert invalid_choice["text"] == ""
        assert invalid_choice["finish_reason"] == "stop"
        assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_numeric_multiple_of() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "properties": {"score": {"type": "number", "multipleOf": 0.25}},
            "required": ["score"],
            "additionalProperties": False,
        },
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"score":1.5}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"score":1.3}']),
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
    assert valid.json()["choices"][0]["text"] == '{"score":1.5}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_unique_items() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}},
            "required": ["tags"],
            "additionalProperties": False,
        },
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"tags":["docs","api"]}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"tags":["docs","docs"]}']),
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
    assert valid.json()["choices"][0]["text"] == '{"tags":["docs","api"]}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_array_contains() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "contains": {"type": "number", "minimum": 0.9},
                    "minContains": 1,
                    "maxContains": 2,
                }
            },
            "required": ["scores"],
            "additionalProperties": False,
        },
    }
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"scores":[0.1,0.95,0.99]}']),
        )
    )
    too_few_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"scores":[0.1,0.2]}']),
        )
    )
    too_many_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"scores":[0.9,0.95,0.99]}']),
        )
    )

    valid = valid_client.post("/v1/completions", json=payload)
    too_few = too_few_client.post("/v1/completions", json=payload)
    too_many = too_many_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"scores":[0.1,0.95,0.99]}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    for response in (too_few, too_many):
        assert response.status_code == 200
        choice = response.json()["choices"][0]
        assert choice["text"] == ""
        assert choice["finish_reason"] == "stop"
        assert choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_uses_json_typed_equality() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "properties": {
                "const_num": {"const": 1},
                "enum_num": {"enum": [1]},
                "mixed": {"type": "array", "uniqueItems": True},
            },
            "required": ["const_num", "enum_num", "mixed"],
            "additionalProperties": False,
        },
    }
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"const_num":1,"enum_num":1,"mixed":[true,1,"1"]}']),
        )
    )
    invalid_outputs = [
        '{"const_num":true,"enum_num":1,"mixed":[true,1,"1"]}',
        '{"const_num":1,"enum_num":true,"mixed":[true,1,"1"]}',
        '{"const_num":1,"enum_num":1,"mixed":[1,1.0]}',
    ]

    valid = valid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"const_num":1,"enum_num":1,"mixed":[true,1,"1"]}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    for generated in invalid_outputs:
        invalid = TestClient(
            create_app(
                ServerConfig(model="fake-path", served_model_name="fake-model"),
                llm=FakeLLM(outputs=[generated]),
            )
        ).post("/v1/completions", json=payload)
        assert invalid.status_code == 200
        invalid_choice = invalid.json()["choices"][0]
        assert invalid_choice["text"] == ""
        assert invalid_choice["finish_reason"] == "stop"
        assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_validates_composition_keywords() -> None:
    schema = {
        "name": "agent_result",
        "schema": {
            "type": "object",
            "properties": {
                "target": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
                "score": {"type": "number", "allOf": [{"minimum": 0}, {"maximum": 1}]},
                "label": {"type": "string", "allOf": [{"minLength": 2}, {"maxLength": 4}]},
                "ambiguous": {"oneOf": [{"type": "number"}, {"type": "integer"}]},
                "debug": {"not": {"const": True}},
            },
            "required": ["target", "score", "label", "ambiguous", "debug"],
            "additionalProperties": False,
        },
    }
    payload = {
        "model": "fake-model",
        "prompt": "json",
        "response_format": {"type": "json_schema", "json_schema": schema},
    }
    valid = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"target":null,"score":0.5,"label":"ok","ambiguous":1.5,"debug":false}']),
        )
    ).post("/v1/completions", json=payload)
    invalid_outputs = [
        '{"target":7,"score":0.5,"label":"ok","ambiguous":1.5,"debug":false}',
        '{"target":null,"score":2,"label":"ok","ambiguous":1.5,"debug":false}',
        '{"target":null,"score":0.5,"label":"x","ambiguous":1.5,"debug":false}',
        '{"target":null,"score":0.5,"label":"ok","ambiguous":1,"debug":false}',
        '{"target":null,"score":0.5,"label":"ok","ambiguous":1.5,"debug":true}',
    ]

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == (
        '{"target":null,"score":0.5,"label":"ok","ambiguous":1.5,"debug":false}'
    )
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    for generated in invalid_outputs:
        invalid = TestClient(
            create_app(
                ServerConfig(model="fake-path", served_model_name="fake-model"),
                llm=FakeLLM(outputs=[generated]),
            )
        ).post("/v1/completions", json=payload)
        assert invalid.status_code == 200
        invalid_choice = invalid.json()["choices"][0]
        assert invalid_choice["text"] == ""
        assert invalid_choice["finish_reason"] == "stop"
        assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_response_format_json_schema_length_rejects_invalid_json_continuation() -> None:
    client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(
                detailed_outputs=[
                    GenerationOutput(
                        text='{"ok": [1}',
                        finish_details=FinishDetails(reason="length", length_limit=9),
                    )
                ]
            ),
        )
    )

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {"type": "json_schema", "json_schema": _response_json_schema()},
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == '{"ok": [1}'
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "schema_violation",
        length_limit=9,
        phase="structured",
        continuation_eligible=False,
    )
    assert "continuation_id" not in choice


def test_completions_guided_json_true_validates_object_result() -> None:
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"ok":true}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=["[true]"]),
        )
    )
    payload = {"model": "fake-model", "prompt": "return json", "guided_json": True}

    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"ok":true}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_guided_json_length_rejects_invalid_json_continuation() -> None:
    client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(
                detailed_outputs=[
                    GenerationOutput(
                        text='{"ok": [1}',
                        finish_details=FinishDetails(reason="length", length_limit=9),
                    )
                ]
            ),
        )
    )

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "return json", "guided_json": True},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == '{"ok": [1}'
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "schema_violation",
        length_limit=9,
        phase="structured",
        continuation_eligible=False,
    )
    assert "continuation_id" not in choice


def test_completions_guided_json_schema_validates_result() -> None:
    schema = _response_json_schema()["schema"]
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
    payload = {"model": "fake-model", "prompt": "return json", "guided_json": schema}

    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"ok":true,"path":"README.md"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_guided_json_schema_validates_local_refs() -> None:
    schema = {
        "type": "object",
        "$defs": {
            "file_ref": {"type": "string", "pattern": r"^[A-Z]+[.]md$"},
            "result_tag": {"enum": ["ok", "skip"]},
        },
        "properties": {
            "path": {"$ref": "#/$defs/file_ref"},
            "result": {"$ref": "#/$defs/result_tag"},
        },
        "required": ["path", "result"],
        "additionalProperties": False,
    }
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"path":"README.md","result":"ok"}']),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=['{"path":"readme.md","result":"ok"}']),
        )
    )
    payload = {
        "model": "fake-model",
        "prompt": "return json",
        "guided_json": {"schema": schema},
    }

    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == '{"path":"README.md","result":"ok"}'
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_guided_json_schema_length_rejects_invalid_json_continuation() -> None:
    schema = _response_json_schema()["schema"]
    client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(
                detailed_outputs=[
                    GenerationOutput(
                        text='{"ok": [1}',
                        finish_details=FinishDetails(reason="length", length_limit=9),
                    )
                ]
            ),
        )
    )

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "return json", "guided_json": schema},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == '{"ok": [1}'
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "schema_violation",
        length_limit=9,
        phase="structured",
        continuation_eligible=False,
    )
    assert "continuation_id" not in choice


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
    assert missing_schema.json()["error"]["hipengine"]["code"] == "schema_violation"
    assert missing_schema.json()["error"]["hipengine"]["legacy_code"] == "invalid_request"
    assert echo.status_code == 400
    assert echo.json()["error"]["code"] == "invalid_request"
    assert echo.json()["error"]["param"] == "echo"
    assert echo.json()["error"]["hipengine"]["code"] == "schema_violation"
    assert echo.json()["error"]["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_rejects_unsupported_schema_keywords() -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {
                        "type": "object",
                        "$anchor": "agent_result",
                    },
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "response_format.json_schema.schema.$anchor"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_rejects_invalid_composition_schema() -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"anyOf": []}},
                    },
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "response_format.json_schema.schema.properties.ok.anyOf"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


@pytest.mark.parametrize(
    ("schema", "param"),
    [
        (
            {"type": "object", "if": "file"},
            "response_format.json_schema.schema.if",
        ),
        (
            {"type": "object", "if": {"type": "object"}, "then": "file"},
            "response_format.json_schema.schema.then",
        ),
        (
            {"type": "object", "if": {"type": "object"}, "else": "url"},
            "response_format.json_schema.schema.else",
        ),
    ],
)
def test_completions_response_format_rejects_invalid_conditional_schema(
    schema: dict[str, Any],
    param: str,
) -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": schema,
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == param
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


@pytest.mark.parametrize(
    ("schema", "param"),
    [
        (
            {"type": "object", "$ref": "https://example.test/schema.json"},
            "response_format.json_schema.schema.$ref",
        ),
        (
            {"type": "object", "$defs": {}, "$ref": "#/$defs/missing"},
            "response_format.json_schema.schema.$ref",
        ),
        (
            {
                "type": "object",
                "$defs": {"node": {"$ref": "#/$defs/node"}},
                "properties": {"node": {"$ref": "#/$defs/node"}},
            },
            "response_format.json_schema.schema.$defs.node.$ref",
        ),
        (
            {"type": "object", "$defs": {"name": "schema"}},
            "response_format.json_schema.schema.$defs.name",
        ),
    ],
)
def test_completions_response_format_rejects_invalid_schema_refs(
    schema: dict[str, Any],
    param: str,
) -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": schema,
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == param
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_rejects_invalid_schema_pattern() -> None:
    fake = FakeLLM(outputs=['{"ok":true,"path":"README.md"}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string", "pattern": "["}},
                    },
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "response_format.json_schema.schema.properties.path.pattern"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_rejects_invalid_property_count_bound() -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {"type": "object", "minProperties": -1},
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "response_format.json_schema.schema.minProperties"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_rejects_invalid_additional_properties_schema() -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {"type": "object", "additionalProperties": ["string"]},
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "response_format.json_schema.schema.additionalProperties"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_rejects_invalid_pattern_properties_schema() -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {"type": "object", "patternProperties": {"[": {"type": "string"}}},
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "response_format.json_schema.schema.patternProperties.["
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


@pytest.mark.parametrize(
    ("schema", "param"),
    [
        (
            {"type": "object", "propertyNames": "name"},
            "response_format.json_schema.schema.propertyNames",
        ),
        (
            {"type": "object", "dependentRequired": {"id": "name"}},
            "response_format.json_schema.schema.dependentRequired.id",
        ),
        (
            {"type": "object", "dependentSchemas": "path"},
            "response_format.json_schema.schema.dependentSchemas",
        ),
        (
            {"type": "object", "dependentSchemas": {"path": "schema"}},
            "response_format.json_schema.schema.dependentSchemas.path",
        ),
    ],
)
def test_completions_response_format_rejects_invalid_object_keyword_schema(
    schema: dict[str, Any],
    param: str,
) -> None:
    fake = FakeLLM(outputs=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": schema,
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == param
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_rejects_invalid_multiple_of_bound() -> None:
    fake = FakeLLM(outputs=['{"score":1}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {
                        "type": "object",
                        "properties": {"score": {"type": "number", "multipleOf": 0}},
                    },
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "response_format.json_schema.schema.properties.score.multipleOf"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_rejects_invalid_unique_items_bound() -> None:
    fake = FakeLLM(outputs=['{"tags":["docs"]}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {
                        "type": "object",
                        "properties": {"tags": {"type": "array", "uniqueItems": "yes"}},
                    },
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "response_format.json_schema.schema.properties.tags.uniqueItems"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


@pytest.mark.parametrize(
    ("schema", "param"),
    [
        (
            {"type": "array", "contains": "number"},
            "response_format.json_schema.schema.contains",
        ),
        (
            {"type": "array", "contains": {"type": "number"}, "minContains": -1},
            "response_format.json_schema.schema.minContains",
        ),
    ],
)
def test_completions_response_format_rejects_invalid_contains_schema(
    schema: dict[str, Any],
    param: str,
) -> None:
    fake = FakeLLM(outputs=["[]"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": schema,
                },
            },
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == param
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_completions_response_format_accepts_annotation_schema_keywords() -> None:
    fake = FakeLLM(outputs=['{"ok":true,"path":"README.md"}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_result",
                    "schema": {
                        "title": "Agent result",
                        "description": "Annotation-only metadata is ignored by validation.",
                        "default": {"ok": True, "path": "README.md"},
                        "type": "object",
                        "properties": {
                            "ok": {"type": "boolean", "description": "success flag"},
                            "path": {
                                "type": "string",
                                "examples": ["README.md"],
                                "format": "uri-reference",
                            },
                        },
                        "required": ["ok", "path"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == '{"ok":true,"path":"README.md"}'
    assert choice["finish_details"] == _stateless_finish_details("stop")
    assert len(fake.calls) == 1


def test_completions_guided_choice_validates_result() -> None:
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=["yes"]),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=["maybe"]),
        )
    )
    payload = {
        "model": "fake-model",
        "prompt": "answer yes or no",
        "guided_choice": ["yes", "no"],
    }

    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == "yes"
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_guided_regex_validates_result() -> None:
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=["AB-12"]),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=["AB"]),
        )
    )
    payload = {
        "model": "fake-model",
        "prompt": "return an id",
        "guided_regex": r"[A-Z]{2}-\d{2}",
    }

    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == "AB-12"
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_completions_guided_diff_validates_unified_diff_result() -> None:
    diff_text = _unified_diff_text()
    valid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=[diff_text]),
        )
    )
    invalid_client = TestClient(
        create_app(
            ServerConfig(model="fake-path", served_model_name="fake-model"),
            llm=FakeLLM(outputs=["Here is the patch:\n" + diff_text]),
        )
    )

    payload = {
        "model": "fake-model",
        "prompt": "patch",
        "guided_diff": {"type": "unified_diff"},
    }
    valid = valid_client.post("/v1/completions", json=payload)
    invalid = invalid_client.post("/v1/completions", json=payload)

    assert valid.status_code == 200
    assert valid.json()["choices"][0]["text"] == diff_text
    assert valid.json()["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["text"] == ""
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


@pytest.mark.parametrize(
    ("guided_diff", "generated", "expected_text"),
    [
        ({"type": "unified_diff", "fenced": False}, _unified_diff_text(), _unified_diff_text()),
        ({"type": "unified_diff", "fenced": "forbidden"}, f"```diff\n{_unified_diff_text()}\n```", ""),
        ({"type": "unified_diff", "fenced": True}, f"```patch\n{_unified_diff_text()}\n```", f"```patch\n{_unified_diff_text()}\n```"),
        ({"type": "unified_diff", "fenced": "required"}, _unified_diff_text(), ""),
        ({"type": "unified_diff", "fenced": "optional"}, f"```diff\n{_unified_diff_text()}\n```", f"```diff\n{_unified_diff_text()}\n```"),
    ],
)
def test_completions_guided_diff_enforces_fenced_policy(guided_diff, generated, expected_text) -> None:
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=[generated]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "patch", "guided_diff": guided_diff},
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["text"] == expected_text
    expected_reason = "stop" if expected_text else "schema_violation"
    assert choice["finish_details"] == _stateless_finish_details(expected_reason)


def test_streaming_completion_guided_diff_buffers_validation_failure() -> None:
    fake = FakeLLM(outputs=["not a diff"], stream_chunks=[_unified_diff_text()])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "patch",
            "guided_diff": True,
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0].get("text") for payload in payloads if payload.get("choices"))
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("schema_violation"),
    }
    assert fake.stream_calls == []


def test_streaming_completion_guided_choice_buffers_validation_failure() -> None:
    fake = FakeLLM(outputs=["maybe"], stream_chunks=["yes"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "answer yes or no",
            "guided_choice": ["yes", "no"],
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0].get("text") for payload in payloads if payload.get("choices"))
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("schema_violation"),
    }
    assert fake.stream_calls == []


def test_streaming_completion_guided_regex_buffers_validation_failure() -> None:
    fake = FakeLLM(outputs=["AB"], stream_chunks=["AB-12"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "return an id",
            "guided_regex": r"[A-Z]{2}-\d{2}",
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0].get("text") for payload in payloads if payload.get("choices"))
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("schema_violation"),
    }
    assert fake.stream_calls == []


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


def test_completion_logprobs_omitted_selected_score_reports_reason() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="alpha beta",
                token_logprobs=(
                    TokenLogprob(token_id=1, token_text="alpha", logprob=None),
                    TokenLogprob(token_id=3, token_text=" beta", logprob=-0.5),
                ),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 2, "logprobs": 1},
    )

    assert response.status_code == 200
    logprobs = response.json()["choices"][0]["logprobs"]
    assert logprobs["tokens"] == ["alpha", " beta"]
    assert logprobs["token_logprobs"] == [None, -0.5]
    assert logprobs["hipengine"] == {
        "omitted_token_logprobs": [
            {
                "index": 0,
                "token": "alpha",
                "token_id": 1,
                "reason": "backend_omitted_logprob",
            }
        ]
    }


def test_completion_logprobs_missing_backend_metadata_returns_unsupported_feature() -> None:
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=["alpha"]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 1, "logprobs": 1},
    )

    assert response.status_code == 501
    error = response.json()["error"]
    assert error["code"] == "unsupported_feature"
    assert error["param"] == "logprobs"
    assert error["hipengine"] == {
        "code": "unsupported_feature",
        "status_code": 501,
        "retryable": False,
    }


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
    assert choice["logprobs"]["hipengine"] == {
        "omitted_token_logprobs": [
            {
                "index": 0,
                "token": "hello",
                "token_id": None,
                "reason": "prompt_logprob_unavailable",
            }
        ]
    }


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
    assert payloads[-1]["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert fake.stream_calls == []
    assert fake.calls[0][1].logprobs is True


def test_streaming_completion_returns_live_chunk_logprobs_when_backend_supports_metadata() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                "alpha",
                token_logprobs=(
                    TokenLogprob(
                        token_id=1,
                        token_text="alpha",
                        logprob=-0.25,
                        top_logprobs=((1, "alpha", -0.25),),
                    ),
                ),
            )
        ],
    )
    fake.supports_stream_logprobs = True
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 1, "stream": True, "logprobs": 1},
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    delta = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0].get("text"))
    choice = delta["choices"][0]
    assert choice["text"] == "alpha"
    assert choice["logprobs"] == {
        "tokens": ["alpha"],
        "token_logprobs": [-0.25],
        "top_logprobs": [{"alpha": -0.25}],
        "text_offset": [0],
    }
    assert fake.stream_calls
    assert fake.calls[0][1].logprobs is True


def test_streaming_completion_live_logprobs_omitted_selected_score_reports_reason() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                "alpha",
                token_logprobs=(TokenLogprob(token_id=1, token_text="alpha", logprob=None),),
            )
        ],
    )
    fake.supports_stream_logprobs = True
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 1, "stream": True, "logprobs": 1},
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    delta = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0].get("text"))
    assert delta["choices"][0]["logprobs"]["token_logprobs"] == [None]
    assert delta["choices"][0]["logprobs"]["hipengine"] == {
        "omitted_token_logprobs": [
            {
                "index": 0,
                "token": "alpha",
                "token_id": 1,
                "reason": "backend_omitted_logprob",
            }
        ]
    }


def test_buffered_streaming_completion_preserves_backend_done_decode_state() -> None:
    fake = FakeLLM(
        outputs=["should-not-stream"],
        stream_chunks=["wrong"],
        detailed_outputs=[
            GenerationOutput(
                text="alpha",
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=7,
                    generated_tokens=3,
                    phase="done",
                    sampler_mode="processed_argmax",
                    forced_token_id=77,
                    forced_token_reason="thinking_hard_close",
                    forced_tokens_remaining=0,
                    active_processors=("logit_bias",),
                    sampler_fallback_reason="processed_logits_required",
                    sampler_fast_path_blockers=("logit_bias",),
                    full_vocab_logits_d2h=False,
                    logits_d2h_bytes=0,
                    execution_path="native_sampler",
                    native_compact_prefill=True,
                    native_caware_decode=True,
                    serial_decode_fallback=False,
                    native_sampler_rows=True,
                    timing={"prefill_ms": 3.5, "decode_ms": 1.25},
                ),
            )
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 1,
            "stream": True,
            "echo": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert payloads[0]["choices"][0]["text"] == "helloalpha"
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
            "active_processors": ["logit_bias"],
            "sampler_fast_path_blockers": ["logit_bias"],
            "sampler_fallback_reason": "processed_logits_required",
            "sampler_mode": "processed_argmax",
            "full_vocab_logits_d2h": False,
            "logits_d2h_bytes": 0,
            "execution_path": "native_sampler",
            "native_compact_prefill": True,
            "native_caware_decode": True,
            "serial_decode_fallback": False,
            "native_sampler_rows": True,
        },
    }
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["hipengine"] == {
        "phase": "done",
        "decode_state": {
            "row_index": 0,
            "step_index": 3,
            "prompt_tokens": 7,
            "generated_tokens": 3,
            "phase": "done",
            "continuation_eligible": False,
            "forced_token_id": 77,
            "forced_token_reason": "thinking_hard_close",
            "forced_tokens_remaining": 0,
            "active_processors": ["logit_bias"],
            "sampler_fast_path_blockers": ["logit_bias"],
            "sampler_fallback_reason": "processed_logits_required",
            "sampler_mode": "processed_argmax",
            "full_vocab_logits_d2h": False,
            "logits_d2h_bytes": 0,
            "execution_path": "native_sampler",
            "native_compact_prefill": True,
            "native_caware_decode": True,
            "serial_decode_fallback": False,
            "native_sampler_rows": True,
        },
        "timing": {"prefill_ms": 3.5, "decode_ms": 1.25},
        "finish_details": _stateless_finish_details("stop"),
    }
    assert done["hipengine"]["timing"]["backend_prefill_ms"] == 3.5
    assert done["hipengine"]["timing"]["backend_decode_ms"] == 1.25
    assert fake.stream_calls == []


def test_streaming_completion_logprobs_missing_backend_metadata_returns_unsupported_feature() -> None:
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=["alpha"], stream_chunks=["alpha"]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "hello", "max_tokens": 1, "stream": True, "logprobs": 1},
    )

    assert response.status_code == 501
    error = response.json()["error"]
    assert error["code"] == "unsupported_feature"
    assert error["param"] == "logprobs"
    assert error["hipengine"] == {
        "code": "unsupported_feature",
        "status_code": 501,
        "retryable": False,
    }


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
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("schema_violation"),
    }
    assert fake.stream_calls == []


def test_streaming_completion_guided_json_buffers_validation() -> None:
    fake = FakeLLM(outputs=["not json"], stream_chunks=['{"ok":true}'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "json",
            "stream": True,
            "guided_json": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0].get("text") for payload in payloads if payload.get("choices"))
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("schema_violation"),
    }
    assert fake.stream_calls == []


def test_streaming_completion_response_format_emits_structured_metadata() -> None:
    structured_text = '{"ok": true, "path": "README.md"}'
    fake = FakeLLM(outputs=[structured_text], stream_chunks=["wrong"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "return json",
            "stream": True,
            "response_format": {"type": "json_object"},
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    delta = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0].get("text"))
    assert delta["choices"][0]["text"] == structured_text
    assert delta["choices"][0]["hipengine"] == {
        "phase": "structured",
        "tokens": {
            "streamed_tokens": 4,
            "delta_tokens": 4,
            "structured_tokens": 4,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 4,
            "prompt_tokens": 0,
            "generated_tokens": 4,
            "phase": "structured",
            "continuation_eligible": False,
            "structured_tokens": 4,
        },
    }
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("stop"),
        "tokens": {
            "prompt_tokens": 2,
            "completion_tokens": 4,
            "total_tokens": 6,
            "streamed_tokens": 4,
            "structured_tokens": 4,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 4,
            "prompt_tokens": 2,
            "generated_tokens": 4,
            "phase": "structured",
            "continuation_eligible": False,
            "structured_tokens": 4,
        },
    }
    usage = next(payload for payload in payloads if payload.get("usage"))
    assert usage["usage"] == {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6}
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
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("schema_violation"),
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


def test_chat_logprobs_use_visible_content_tokens_after_reasoning() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="<think>plan</think>answer",
                token_logprobs=(
                    TokenLogprob(token_id=3, token_text="<think>plan</think>", logprob=-0.1),
                    TokenLogprob(
                        token_id=4,
                        token_text="answer",
                        logprob=-0.2,
                        top_logprobs=((4, "answer", -0.2),),
                    ),
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
    assert choice["message"] == {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "plan",
    }
    assert choice["logprobs"] == {
        "content": [
            {
                "token": "answer",
                "logprob": -0.2,
                "bytes": None,
                "top_logprobs": [{"token": "answer", "logprob": -0.2, "bytes": None}],
            }
        ],
        "refusal": None,
    }


def test_chat_logprobs_omitted_selected_score_reports_reason() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="assistant reply",
                token_logprobs=(
                    TokenLogprob(token_id=4, token_text="assistant", logprob=None),
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
        },
    )

    assert response.status_code == 200
    logprobs = response.json()["choices"][0]["logprobs"]
    assert [entry["logprob"] for entry in logprobs["content"]] == [None, -0.2]
    assert logprobs["hipengine"] == {
        "omitted_token_logprobs": [
            {
                "index": 0,
                "token": "assistant",
                "token_id": 4,
                "reason": "backend_omitted_logprob",
            }
        ]
    }


def test_chat_logprobs_missing_backend_metadata_returns_unsupported_feature() -> None:
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=["assistant"]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "logprobs": True,
        },
    )

    assert response.status_code == 501
    error = response.json()["error"]
    assert error["code"] == "unsupported_feature"
    assert error["param"] == "logprobs"
    assert error["hipengine"] == {
        "code": "unsupported_feature",
        "status_code": 501,
        "retryable": False,
    }


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
    assert payloads[-1]["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    assert fake.stream_calls == []
    assert fake.calls[0][1].logprobs is True


def test_streaming_chat_completion_returns_live_chunk_logprobs_when_backend_supports_metadata() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                "assistant",
                token_logprobs=(
                    TokenLogprob(
                        token_id=4,
                        token_text="assistant",
                        logprob=-0.1,
                        top_logprobs=((4, "assistant", -0.1),),
                    ),
                ),
            )
        ],
    )
    fake.supports_stream_logprobs = True
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
            "top_logprobs": 1,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content = next(
        payload for payload in payloads if payload.get("choices") and payload["choices"][0]["delta"].get("content")
    )
    choice = content["choices"][0]
    assert choice["delta"] == {"content": "assistant"}
    assert choice["logprobs"] == {
        "content": [
            {
                "token": "assistant",
                "logprob": -0.1,
                "bytes": None,
                "top_logprobs": [{"token": "assistant", "logprob": -0.1, "bytes": None}],
            }
        ],
        "refusal": None,
    }
    assert fake.stream_calls
    assert fake.calls[0][1].logprobs is True
    assert fake.calls[0][1].top_logprobs == 1


def test_streaming_chat_completion_n_uses_scheduler_token_chunks_for_buffered_logprobs() -> None:
    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            outputs = [
                GenerationOutput(
                    text="AB",
                    token_logprobs=(
                        TokenLogprob(token_id=100, token_text="A", logprob=-0.1),
                        TokenLogprob(token_id=101, token_text="B", logprob=-0.2),
                    ),
                    finish_details=FinishDetails(reason="length", length_limit=2, sampler_mode="host_logits_sample"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=2,
                        row_index=0,
                        phase="done",
                        sampler_mode="host_logits_sample",
                    ),
                ),
                GenerationOutput(
                    text="CD",
                    token_logprobs=(
                        TokenLogprob(token_id=200, token_text="C", logprob=-0.3),
                        TokenLogprob(token_id=201, token_text="D", logprob=-0.4),
                    ),
                    finish_details=FinishDetails(reason="length", length_limit=2, sampler_mode="host_logits_sample"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=2,
                        row_index=1,
                        phase="done",
                        sampler_mode="host_logits_sample",
                    ),
                ),
            ]
            chunk_texts = (("A", "B"), ("C", "D"))
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {
                        "request_id": request_id,
                        "token_index": token_index,
                        "token_id": 100 + request_id * 100 + token_index,
                        "finished": token_index == 1,
                        "chunk": {
                            "text": text,
                            "token_logprobs": [
                                {
                                    "token_id": 100 + request_id * 100 + token_index,
                                    "token_text": text,
                                    "logprob": -(request_id * 2 + token_index + 1) / 10,
                                    "top_logprobs": [
                                        {
                                            "token_id": 100 + request_id * 100 + token_index,
                                            "token_text": text,
                                            "logprob": -(request_id * 2 + token_index + 1) / 10,
                                        }
                                    ],
                                }
                            ],
                            "telemetry": GenerationTelemetry.from_decode_counts(
                                prompt_tokens=1,
                                generated_tokens=token_index + 1,
                                row_index=request_id,
                                request_id=str(request_id),
                                phase="answer",
                                sampler_mode="host_logits_sample",
                                execution_path="scheduler_native_packed_prefill_serial_host_sampler_decode",
                            ).to_json_dict(),
                        },
                    }
                    for request_id, row in enumerate(chunk_texts)
                    for token_index, text in enumerate(row)
                ]
            }
            return outputs[: len(prompt_tuple)]

    fake = SchedulerChunkFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "n": 2,
            "max_tokens": 2,
            "stream": True,
            "logprobs": True,
            "top_logprobs": 1,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    deltas = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] is None
    ]
    content_deltas = [choice for choice in deltas if "content" in choice["delta"]]
    assert [(choice["index"], choice["delta"]["content"]) for choice in content_deltas] == [
        (0, "A"),
        (0, "B"),
        (1, "C"),
        (1, "D"),
    ]
    assert [
        choice["logprobs"]["content"][0]["logprob"]
        for choice in content_deltas
    ] == pytest.approx([-0.1, -0.2, -0.3, -0.4])
    assert [
        choice["logprobs"]["content"][0]["top_logprobs"][0]["token"]
        for choice in content_deltas
    ] == ["A", "B", "C", "D"]
    assert {
        choice["hipengine"]["decode_state"]["execution_path"]
        for choice in content_deltas
    } == {"scheduler_native_packed_prefill_serial_host_sampler_decode"}
    done = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] == "length"
    ]
    assert [choice["index"] for choice in done] == [0, 1]
    assert fake.stream_calls == []


def test_streaming_chat_completion_uses_scheduler_reasoning_private_logprobs() -> None:
    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            text = "<think>r0</think>A"
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {
                        "request_id": 0,
                        "token_index": token_index,
                        "token_id": 300 + token_index,
                        "finished": token_index == 1,
                        "chunk": {
                            "text": chunk_text,
                            "token_logprobs": [
                                {
                                    "token_id": 300 + token_index,
                                    "token_text": chunk_text,
                                    "logprob": -(token_index + 1) / 10,
                                }
                            ],
                            "telemetry": GenerationTelemetry.from_decode_counts(
                                prompt_tokens=1,
                                generated_tokens=token_index + 1,
                                row_index=0,
                                request_id="0",
                                phase="answer",
                                sampler_mode="host_logits_sample",
                                execution_path="scheduler_should_not_surface",
                            ).to_json_dict(),
                        },
                    }
                    for token_index, chunk_text in enumerate(("<think>r0</think>", "A"))
                ]
            }
            return [
                GenerationOutput(
                    text=text,
                    token_logprobs=(
                        TokenLogprob(token_id=300, token_text="<think>r0</think>", logprob=-0.1),
                        TokenLogprob(token_id=301, token_text="A", logprob=-0.2),
                    ),
                    finish_details=FinishDetails(reason="stop", sampler_mode="host_logits_sample"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=2,
                        row_index=0,
                        phase="done",
                        sampler_mode="host_logits_sample",
                        execution_path="buffered_final_metadata",
                    ),
                )
            ][: len(prompt_tuple)]

    fake = SchedulerChunkFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "stream": True,
            "logprobs": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    deltas = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] is None
    ]
    assert [(choice["index"], choice["delta"]) for choice in deltas] == [
        (0, {"role": "assistant"}),
        (0, {"reasoning_content": "r0"}),
        (0, {"content": "A"}),
    ]
    assert "<think>" not in json.dumps([choice["delta"] for choice in deltas])
    assert {
        choice.get("hipengine", {}).get("decode_state", {}).get("execution_path")
        for choice in deltas
        if choice["delta"].get("content") or choice["delta"].get("reasoning_content")
    } == {"scheduler_should_not_surface"}
    reasoning = next(choice for choice in deltas if choice["delta"].get("reasoning_content"))
    assert "logprobs" not in reasoning
    assert reasoning["hipengine"]["decode_state"]["phase"] == "think"
    assert reasoning["hipengine"]["reasoning_logprobs"] == {
        "content": [
            {
                "token_id": 300,
                "token": "<think>r0</think>",
                "logprob": -0.1,
                "bytes": None,
                "top_logprobs": [],
            }
        ],
        "public_text": "r0",
        "refusal": None,
    }
    content = next(choice for choice in deltas if choice["delta"].get("content"))
    assert content["hipengine"]["decode_state"]["phase"] == "answer"
    assert content["logprobs"]["content"] == [
        {"token": "A", "logprob": -0.2, "bytes": None, "top_logprobs": []}
    ]
    assert fake.stream_calls == []


def test_streaming_chat_completion_uses_live_reasoning_private_logprobs() -> None:
    fake = FakeLLM(
        stream_chunks=[
            GenerationStreamChunk(
                text="<think>r0</think>",
                token_logprobs=(
                    TokenLogprob(
                        token_id=300,
                        token_text="<think>r0</think>",
                        logprob=-0.1,
                        top_logprobs=((300, "<think>r0</think>", -0.1),),
                    ),
                ),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=1,
                    row_index=0,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    execution_path="live_host_sampler_decode",
                ),
            ),
            GenerationStreamChunk(
                text="A",
                token_logprobs=(TokenLogprob(token_id=301, token_text="A", logprob=-0.2),),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=2,
                    row_index=0,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    execution_path="live_host_sampler_decode",
                ),
            ),
        ],
    )
    fake.supports_stream_logprobs = True
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "stream": True,
            "logprobs": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    reasoning = next(
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("reasoning_content")
    )
    content = next(
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("content")
    )
    assert reasoning["delta"] == {"reasoning_content": "r0"}
    assert "logprobs" not in reasoning
    assert reasoning["hipengine"]["decode_state"]["phase"] == "think"
    assert reasoning["hipengine"]["decode_state"]["execution_path"] == "live_host_sampler_decode"
    assert reasoning["hipengine"]["reasoning_logprobs"] == {
        "content": [
            {
                "token_id": 300,
                "token": "<think>r0</think>",
                "logprob": -0.1,
                "bytes": None,
                "top_logprobs": [
                    {
                        "token_id": 300,
                        "token": "<think>r0</think>",
                        "logprob": -0.1,
                        "bytes": None,
                    }
                ],
            }
        ],
        "public_text": "r0",
        "refusal": None,
    }
    assert content["delta"] == {"content": "A"}
    assert content["logprobs"]["content"] == [
        {"token": "A", "logprob": -0.2, "bytes": None, "top_logprobs": []}
    ]
    assert fake.stream_calls
    assert fake.calls[0][1].logprobs is True
    assert fake.calls[0][1].top_logprobs == 0


def test_streaming_chat_completion_maps_live_middle_reasoning_span_logprobs() -> None:
    fake = FakeLLM(
        stream_chunks=[
            GenerationStreamChunk(
                text="<think>plan</think>A<thi",
                token_logprobs=(
                    TokenLogprob(token_id=330, token_text="<think>", logprob=-0.1),
                    TokenLogprob(token_id=331, token_text="plan", logprob=-0.2),
                    TokenLogprob(token_id=332, token_text="</think>", logprob=-0.3),
                    TokenLogprob(token_id=333, token_text="A", logprob=-0.4),
                    TokenLogprob(token_id=334, token_text="<thi", logprob=-0.5),
                ),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=5,
                    row_index=0,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    execution_path="live_host_sampler_decode",
                ),
            ),
        ],
    )
    fake.supports_stream_logprobs = True
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "logprobs": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    reasoning = next(
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("reasoning_content")
    )
    content = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("content")
    ]
    assert reasoning["delta"] == {"reasoning_content": "plan"}
    assert reasoning["hipengine"]["reasoning_logprobs"] == {
        "content": [
            {
                "token_id": 331,
                "token": "plan",
                "logprob": -0.2,
                "bytes": None,
                "top_logprobs": [],
            }
        ],
        "public_text": "plan",
        "refusal": None,
    }
    assert [choice["delta"] for choice in content] == [{"content": "A"}, {"content": "<thi"}]
    assert content[0]["logprobs"]["content"] == [
        {"token": "A", "logprob": -0.4, "bytes": None, "top_logprobs": []}
    ]
    assert content[1]["logprobs"]["content"] == [
        {"token": "<thi", "logprob": -0.5, "bytes": None, "top_logprobs": []}
    ]


def test_streaming_chat_completion_maps_live_final_content_logprobs() -> None:
    fake = FakeLLM(
        stream_chunks=[
            GenerationStreamChunk(
                text="<t",
                token_logprobs=(TokenLogprob(token_id=310, token_text="<t", logprob=-0.7),),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=1,
                    row_index=0,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    execution_path="live_host_sampler_decode",
                ),
            ),
            GenerationStreamChunk(
                text="hi",
                token_logprobs=(TokenLogprob(token_id=311, token_text="hi", logprob=-0.8),),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=2,
                    row_index=0,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    execution_path="live_host_sampler_decode",
                ),
            ),
        ],
    )
    fake.supports_stream_logprobs = True
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "logprobs": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content = next(
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("content")
    )
    assert content["delta"] == {"content": "<thi"}
    assert content["logprobs"]["content"] == [
        {"token": "<t", "logprob": -0.7, "bytes": None, "top_logprobs": []},
        {"token": "hi", "logprob": -0.8, "bytes": None, "top_logprobs": []},
    ]
    assert content["hipengine"]["decode_state"]["phase"] == "answer"
    assert content["hipengine"]["decode_state"]["execution_path"] == "live_host_sampler_decode"


def test_streaming_chat_completion_maps_live_final_reasoning_logprobs() -> None:
    fake = FakeLLM(
        stream_chunks=[
            GenerationStreamChunk(
                text="<think>plan",
                token_logprobs=(TokenLogprob(token_id=320, token_text="<think>plan", logprob=-0.1),),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=1,
                    row_index=0,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    execution_path="live_host_sampler_decode",
                ),
            ),
            GenerationStreamChunk(
                text="</t",
                token_logprobs=(TokenLogprob(token_id=321, token_text="</t", logprob=-0.2),),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=2,
                    row_index=0,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    execution_path="live_host_sampler_decode",
                ),
            ),
            GenerationStreamChunk(
                text="hi",
                token_logprobs=(TokenLogprob(token_id=322, token_text="hi", logprob=-0.3),),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=3,
                    row_index=0,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    execution_path="live_host_sampler_decode",
                ),
            ),
        ],
    )
    fake.supports_stream_logprobs = True
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "logprobs": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    reasoning_deltas = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("reasoning_content")
    ]
    assert [choice["delta"] for choice in reasoning_deltas] == [
        {"reasoning_content": "plan"},
        {"reasoning_content": "</thi"},
    ]
    assert [choice["hipengine"]["decode_state"]["phase"] for choice in reasoning_deltas] == [
        "think",
        "think",
    ]
    assert reasoning_deltas[-1]["hipengine"]["reasoning_logprobs"] == {
        "content": [
            {
                "token_id": 321,
                "token": "</t",
                "logprob": -0.2,
                "bytes": None,
                "top_logprobs": [],
            },
            {
                "token_id": 322,
                "token": "hi",
                "logprob": -0.3,
                "bytes": None,
                "top_logprobs": [],
            }
        ],
        "public_text": "</thi",
        "refusal": None,
    }


def test_streaming_chat_completion_falls_back_for_unmappable_reasoning_logprobs() -> None:
    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            text = "<think>r0</think>A"
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {
                        "request_id": 0,
                        "token_index": 0,
                        "token_id": 300,
                        "finished": True,
                        "chunk": {
                            "text": text,
                            "token_logprobs": [
                                {
                                    "token_id": 300,
                                    "token_text": text,
                                    "logprob": -0.1,
                                }
                            ],
                            "telemetry": GenerationTelemetry.from_decode_counts(
                                prompt_tokens=1,
                                generated_tokens=1,
                                row_index=0,
                                request_id="0",
                                phase="answer",
                                sampler_mode="host_logits_sample",
                                execution_path="scheduler_should_not_surface",
                            ).to_json_dict(),
                        },
                    }
                ]
            }
            return [
                GenerationOutput(
                    text=text,
                    token_logprobs=(TokenLogprob(token_id=300, token_text=text, logprob=-0.1),),
                    finish_details=FinishDetails(reason="stop", sampler_mode="host_logits_sample"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=1,
                        row_index=0,
                        phase="done",
                        sampler_mode="host_logits_sample",
                        execution_path="buffered_final_metadata",
                    ),
                )
            ][: len(prompt_tuple)]

    fake = SchedulerChunkFakeLLM()
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
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    deltas = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] is None
    ]
    assert [(choice["index"], choice["delta"]) for choice in deltas] == [
        (0, {"role": "assistant"}),
        (0, {"reasoning_content": "r0"}),
        (0, {"content": "A"}),
    ]
    assert {
        choice.get("hipengine", {}).get("decode_state", {}).get("execution_path")
        for choice in deltas
        if choice["delta"].get("content") or choice["delta"].get("reasoning_content")
    } == {"buffered_final_metadata"}
    content = next(choice for choice in deltas if choice["delta"].get("content"))
    assert content["logprobs"]["content"] == []
    done = next(
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] == "stop"
    )
    diagnostic = done["hipengine"]["withheld_scheduler_logprob_chunks"]
    assert diagnostic == {
        "surface": "chat_logprob_delta",
        "reason": "unmappable_logprobs",
        "public_delta": "buffered_without_scheduler_logprobs",
        "chunk_count": 1,
        "malformed_chunk_count": 0,
        "chunks_with_logprobs": 1,
        "token_logprob_count": 1,
        "raw_text_matches": True,
        "text_bytes": len("<think>r0</think>A".encode("utf-8")),
        "text_sha256": hashlib.sha256("<think>r0</think>A".encode("utf-8")).hexdigest(),
        "chunk_text_bytes": [len("<think>r0</think>A".encode("utf-8"))],
        "execution_paths": ["scheduler_should_not_surface"],
    }
    assert "text" not in diagnostic
    assert fake.stream_calls == []


def test_streaming_chat_live_logprobs_omitted_selected_score_reports_reason() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                "assistant",
                token_logprobs=(TokenLogprob(token_id=4, token_text="assistant", logprob=None),),
            )
        ],
    )
    fake.supports_stream_logprobs = True
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
    content = next(
        payload for payload in payloads if payload.get("choices") and payload["choices"][0]["delta"].get("content")
    )
    assert content["choices"][0]["logprobs"]["content"][0]["logprob"] is None
    assert content["choices"][0]["logprobs"]["hipengine"] == {
        "omitted_token_logprobs": [
            {
                "index": 0,
                "token": "assistant",
                "token_id": 4,
                "reason": "backend_omitted_logprob",
            }
        ]
    }


def test_buffered_streaming_chat_preserves_backend_done_decode_state() -> None:
    fake = FakeLLM(
        outputs=["should-not-stream"],
        stream_chunks=["wrong"],
        detailed_outputs=[
            GenerationOutput(
                text="first",
                telemetry=GenerationTelemetry.from_decode_counts(
                    row_index=0,
                    prompt_tokens=11,
                    generated_tokens=1,
                    phase="done",
                    sampler_mode="processed_argmax",
                    sampler_fallback_reason="processed_logits_required",
                    sampler_fast_path_blockers=("logit_bias",),
                    timing={"prefill_ms": 2.0},
                ),
            ),
            GenerationOutput(
                text="second",
                telemetry=GenerationTelemetry.from_decode_counts(
                    row_index=1,
                    prompt_tokens=11,
                    generated_tokens=2,
                    phase="done",
                    sampler_mode="host_logits_sample",
                    sampler_fallback_reason="host_sampling_required",
                    sampler_fast_path_blockers=("temperature",),
                    timing={"prefill_ms": 2.5},
                ),
            ),
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "n": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content_payloads = [
        payload
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("content")
    ]
    assert [payload["choices"][0]["index"] for payload in content_payloads] == [0, 1]
    assert content_payloads[0]["choices"][0]["hipengine"] == {
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
            "sampler_fast_path_blockers": ["logit_bias"],
            "sampler_fallback_reason": "processed_logits_required",
            "sampler_mode": "processed_argmax",
        },
    }
    assert content_payloads[1]["choices"][0]["hipengine"] == {
        "phase": "answer",
        "tokens": {
            "streamed_tokens": 1,
            "delta_tokens": 1,
            "answer_tokens": 1,
        },
        "decode_state": {
            "row_index": 1,
            "step_index": 1,
            "prompt_tokens": 0,
            "generated_tokens": 1,
            "phase": "answer",
            "continuation_eligible": False,
            "answer_tokens": 1,
            "sampler_fast_path_blockers": ["temperature"],
            "sampler_fallback_reason": "host_sampling_required",
            "sampler_mode": "host_logits_sample",
        },
    }
    done_payloads = [
        payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"]
    ]
    assert [payload["choices"][0]["index"] for payload in done_payloads] == [0, 1]
    assert done_payloads[0]["choices"][0]["hipengine"] == {
        "phase": "done",
        "decode_state": {
            "row_index": 0,
            "step_index": 1,
            "prompt_tokens": 11,
            "generated_tokens": 1,
            "phase": "done",
            "continuation_eligible": False,
            "sampler_fast_path_blockers": ["logit_bias"],
            "sampler_fallback_reason": "processed_logits_required",
            "sampler_mode": "processed_argmax",
        },
        "timing": {"prefill_ms": 2.0},
        "finish_details": _stateless_finish_details("stop"),
        "tokens": {
            "streamed_tokens": 1,
            "answer_tokens": 1,
        },
    }
    assert done_payloads[1]["choices"][0]["hipengine"] == {
        "phase": "done",
        "decode_state": {
            "row_index": 1,
            "step_index": 2,
            "prompt_tokens": 11,
            "generated_tokens": 2,
            "phase": "done",
            "continuation_eligible": False,
            "sampler_fast_path_blockers": ["temperature"],
            "sampler_fallback_reason": "host_sampling_required",
            "sampler_mode": "host_logits_sample",
        },
        "timing": {"prefill_ms": 2.5},
        "finish_details": _stateless_finish_details("stop"),
        "tokens": {
            "streamed_tokens": 1,
            "answer_tokens": 1,
        },
    }
    assert done_payloads[0]["hipengine"]["timing"]["backend_prefill_ms"] == 2.0
    assert done_payloads[1]["hipengine"]["timing"]["backend_prefill_ms"] == 2.5
    assert fake.stream_calls == []


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
    assert body["model"] == "fake-model"
    assert body["hipengine"]["routing"] == _routing_metadata()
    assert body["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "assistant reply"},
            "finish_reason": "stop",
            "finish_details": _stateless_finish_details("stop"),
        }
    ]
    assert fake.calls[0][0] == (
        "<|im_start|>system\nbe concise<|im_end|>\n"
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n",
    )
    assert fake.calls[0][1].max_tokens == 4


def test_chat_completion_rejects_unsupported_message_role_before_generation() -> None:
    fake = FakeLLM(outputs=["should not generate"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "critic", "content": "hello"}],
            "max_tokens": 4,
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "messages[0].role"
    assert "assistant, developer, system, tool, user" in error["message"]
    assert fake.calls == []


@pytest.mark.parametrize(
    ("message", "param"),
    [
        (
            {
                "role": "user",
                "content": "hello",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            },
            "messages[0].tool_calls",
        ),
        (
            {"role": "assistant", "content": "hello", "tool_call_id": "call_1"},
            "messages[0].tool_call_id",
        ),
        ({"role": "tool", "content": "tool result"}, "messages[0].tool_call_id"),
    ],
)
def test_chat_completion_rejects_invalid_role_specific_tool_fields_before_generation(
    message: dict[str, Any],
    param: str,
) -> None:
    fake = FakeLLM(outputs=["should not generate"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": [message], "max_tokens": 4},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == param
    assert fake.calls == []


@pytest.mark.parametrize(
    ("messages", "param"),
    [
        (
            [{"role": "tool", "tool_call_id": "call_missing", "content": "orphan tool result"}],
            "messages[0].tool_call_id",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        },
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "write", "arguments": "{}"},
                        },
                    ],
                }
            ],
            "messages[0].tool_calls[1].id",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "first result"},
                {"role": "tool", "tool_call_id": "call_1", "content": "duplicate result"},
            ],
            "messages[2].tool_call_id",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "user", "content": "skip the tool"},
            ],
            "messages[1].role",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "write", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "first result"},
                {"role": "user", "content": "skip the second tool"},
            ],
            "messages[2].role",
        ),
    ],
)
def test_chat_completion_rejects_invalid_tool_transcript_before_generation(
    messages: list[dict[str, Any]],
    param: str,
) -> None:
    fake = FakeLLM(outputs=["should not generate"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "messages": messages, "max_tokens": 4},
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == param
    assert fake.calls == []


@pytest.mark.parametrize(
    ("corruption", "param"),
    [
        ("extra_key", "messages[0].tool_calls[0].unexpected"),
        ("missing_id", "messages[0].tool_calls[0].id"),
        ("wrong_type", "messages[0].tool_calls[0].type"),
        ("missing_function", "messages[0].tool_calls[0].function"),
        ("function_extra_key", "messages[0].tool_calls[0].function.unexpected"),
        ("empty_name", "messages[0].tool_calls[0].function.name"),
        ("non_string_arguments", "messages[0].tool_calls[0].function.arguments"),
        ("invalid_json_arguments", "messages[0].tool_calls[0].function.arguments"),
    ],
)
def test_chat_completion_rejects_invalid_prior_assistant_tool_call_shape_before_generation(
    corruption: str,
    param: str,
) -> None:
    tool_call: dict[str, Any] = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "read", "arguments": '{"path":"README.md"}'},
    }
    if corruption == "extra_key":
        tool_call["unexpected"] = True
    elif corruption == "missing_id":
        del tool_call["id"]
    elif corruption == "wrong_type":
        tool_call["type"] = "custom"
    elif corruption == "missing_function":
        del tool_call["function"]
    elif corruption == "function_extra_key":
        tool_call["function"]["unexpected"] = True
    elif corruption == "empty_name":
        tool_call["function"]["name"] = ""
    elif corruption == "non_string_arguments":
        tool_call["function"]["arguments"] = {"path": "README.md"}
    elif corruption == "invalid_json_arguments":
        tool_call["function"]["arguments"] = '{"path":'
    else:
        raise AssertionError(f"unhandled corruption case: {corruption}")

    fake = FakeLLM(outputs=["should not generate"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "assistant", "content": "", "tool_calls": [tool_call]}],
            "max_tokens": 4,
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == param
    assert fake.calls == []


def test_chat_completion_exposes_backend_generation_telemetry() -> None:
    fake = DetailedGenerateFakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="assistant reply",
                finish_details=FinishDetails(reason="eos", eos_token_id=151645, sampler_mode="processed_argmax"),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=4,
                    generated_tokens=2,
                    row_index=0,
                    phase="answer",
                    sampler_mode="processed_argmax",
                    active_processors=("min_tokens",),
                    sampler_fast_path_blockers=("min_tokens",),
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
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "assistant reply"}
    assert choice["hipengine"] == {
        "decode_state": {
            "row_index": 0,
            "step_index": 2,
            "prompt_tokens": 4,
            "generated_tokens": 2,
            "phase": "answer",
            "continuation_eligible": False,
            "active_processors": ["min_tokens"],
            "sampler_fast_path_blockers": ["min_tokens"],
            "sampler_mode": "processed_argmax",
        },
        "finish_details": _stateless_finish_details(
            "eos",
            eos_token_id=151645,
            sampler_mode="processed_argmax",
        ),
    }


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


def test_chat_completion_allow_unbounded_skips_default_hard_thinking_cap() -> None:
    fake = FakeLLM(outputs=["answer"], token_map={"</think>": [42, 43]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "reasoning_effort": "low",
            "thinking": {"allow_unbounded": True},
            "max_tokens": 100,
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[-1][0][0]
    assert "keep it very brief" in prompt
    assert "reserve at least 50 tokens for the final answer or tool call" in prompt
    assert "close </think> before exceeding" not in prompt
    assert "begin closing during the final" not in prompt
    sampling = fake.calls[-1][1]
    assert sampling.thinking_hard_token_cap is None
    assert sampling.thinking_close_token_ids == ()
    assert fake.tokenize_calls == []


def test_chat_completion_allow_unbounded_preserves_explicit_hard_cap() -> None:
    fake = FakeLLM(outputs=["answer"], token_map={"</think>": [42, 43]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "reasoning_effort": "low",
            "thinking": {"allow_unbounded": True, "hard_think_cap": 12},
            "max_tokens": 100,
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[-1][0][0]
    assert "close </think> before exceeding 12 hidden reasoning tokens" in prompt
    assert "begin closing during the final 12 hidden reasoning tokens" in prompt
    sampling = fake.calls[-1][1]
    assert sampling.thinking_hard_token_cap == 12
    assert sampling.thinking_close_token_ids == (42, 43)
    assert fake.tokenize_calls == ["</think>"]


def test_chat_completion_reasoning_object_budget_overrides_top_level_controls() -> None:
    fake = FakeLLM(outputs=["answer"], token_map={"closing</think>\n": [42, 43, 44]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "max_tokens": 100,
            "chat_template_kwargs": {"thinking_budget": 99},
            "thinking_token_budget": 123,
            "hard_think_cap": 80,
            "min_answer_tokens": 70,
            "soft_close_window": 60,
            "thinking": {"budget_tokens": 50, "min_answer_tokens": 30},
            "reasoning": {
                "effort": "high",
                "max_tokens": 12,
                "min_answer_tokens": 4,
                "soft_close_window": 3,
                "hard_close_sequence": "closing</think>\n",
            },
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[-1][0][0]
    assert "keep it focused but complete" in prompt
    assert "close </think> before exceeding 12 hidden reasoning tokens" in prompt
    assert "reserve at least 4 tokens for the final answer or tool call" in prompt
    assert "begin closing during the final 3 hidden reasoning tokens" in prompt
    assert "use 'closing</think>\\n' as the close sequence if budget pressure requires it" in prompt
    assert "exceeding 99 hidden reasoning tokens" not in prompt
    assert "exceeding 123 hidden reasoning tokens" not in prompt
    assert "exceeding 80 hidden reasoning tokens" not in prompt
    assert "exceeding 50 hidden reasoning tokens" not in prompt
    sampling = fake.calls[-1][1]
    assert sampling.thinking_hard_token_cap == 12
    assert sampling.thinking_soft_close_window == 3
    assert sampling.thinking_close_token_ids == (42, 43, 44)
    assert fake.tokenize_calls == ["closing</think>\n"]


def test_chat_completion_reasoning_type_disabled_wins_over_enabled_true() -> None:
    fake = FakeLLM(outputs=["answer"], token_map={"</think>": [42, 43]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "think"}],
            "reasoning_effort": "low",
            "reasoning": {"type": "none", "enabled": True, "hard_think_cap": 12},
        },
    )

    assert response.status_code == 200
    prompt = fake.calls[-1][0][0]
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n" in prompt
    assert "Do not include hidden reasoning" in prompt
    assert "close </think> before exceeding" not in prompt
    sampling = fake.calls[-1][1]
    assert sampling.thinking_hard_token_cap is None
    assert sampling.thinking_close_token_ids == ()
    assert fake.tokenize_calls == []


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
    if phase in {"answer", "structured"}:
        continuation_id = choice["continuation_id"]
        assert continuation_id.startswith("gen_")
        assert choice["finish_details"] == _stateless_finish_details(
            "length",
            length_limit=5,
            phase=phase,
            continuation_eligible=True,
            continuation_id=continuation_id,
        )
    else:
        assert "continuation_id" not in choice
        assert choice["finish_details"] == _stateless_finish_details(
            "length",
            length_limit=5,
            phase=phase,
            continuation_eligible=False,
        )
    assert choice["message"]["content"] == content
    if reasoning_content is None:
        assert "reasoning_content" not in choice["message"]
    else:
        assert choice["message"]["reasoning_content"] == reasoning_content


def test_chat_length_finish_honors_backend_continuation_ineligible() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="partial answer",
                finish_details=FinishDetails(reason="length", length_limit=2, continuation_eligible=False),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=2,
                    generated_tokens=2,
                    sampler_mode="greedy_fast",
                    continuation_eligible=True,
                ),
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
    assert "continuation_id" not in choice
    assert choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=2,
        phase="answer",
        continuation_eligible=False,
    )
    assert choice["message"] == {"role": "assistant", "content": "partial answer"}
    assert choice["hipengine"]["decode_state"] == {
        "row_index": 0,
        "step_index": 2,
        "prompt_tokens": 2,
        "generated_tokens": 2,
        "phase": "done",
        "continuation_eligible": False,
        "sampler_mode": "greedy_fast",
    }


def test_chat_completion_thinking_budget_exhausted_maps_to_length_finish() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="<think>closed</think>",
                finish_details=FinishDetails(
                    reason="thinking_budget_exhausted",
                    length_limit=1,
                    forced_close=True,
                    reasoning_tokens=1,
                    budget_pressure="hard_close",
                    phase="answer",
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
            "messages": [{"role": "user", "content": "think briefly"}],
            "reasoning_effort": "low",
            "max_tokens": 1,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "length"
    assert "continuation_id" not in choice
    assert choice["finish_details"] == _stateless_finish_details(
        "thinking_budget_exhausted",
        length_limit=1,
        forced_close=True,
        reasoning_tokens=1,
        budget_pressure="hard_close",
        phase="answer",
        continuation_eligible=False,
    )


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
    assert valid_choice["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["message"] == {"role": "assistant", "content": ""}
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
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
    assert valid_choice["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["message"] == {"role": "assistant", "content": ""}
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "Return only JSON that satisfies this JSON schema" in invalid_fake.calls[0][0][0]


def test_chat_completion_response_format_json_schema_length_rejects_invalid_json_continuation() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text='{"ok": [1}',
                finish_details=FinishDetails(reason="length", length_limit=9),
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
            "response_format": {"type": "json_schema", "json_schema": _response_json_schema()},
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": '{"ok": [1}'}
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "schema_violation",
        length_limit=9,
        phase="structured",
        continuation_eligible=False,
    )
    assert "continuation_id" not in choice


def test_chat_completion_guided_json_schema_validates_visible_content() -> None:
    valid_fake = FakeLLM(outputs=['<think>check</think>{"ok":true,"path":"README.md"}'])
    invalid_fake = FakeLLM(outputs=['{"ok":true,"path":""}'])
    valid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=valid_fake)
    )
    invalid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=invalid_fake)
    )
    schema_text = json.dumps(_response_json_schema()["schema"])

    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "return json"}],
        "guided_json": schema_text,
    }
    valid = valid_client.post("/v1/chat/completions", json=payload)
    invalid = invalid_client.post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["message"]["content"] == '{"ok":true,"path":"README.md"}'
    assert valid_choice["message"]["reasoning_content"] == "check"
    assert valid_choice["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["message"] == {"role": "assistant", "content": ""}
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "Return only JSON that satisfies this JSON schema" in valid_fake.calls[0][0][0]


def test_chat_completion_guided_json_schema_validates_local_refs() -> None:
    schema = {
        "type": "object",
        "$defs": {
            "file_ref": {"type": "string", "pattern": r"^[A-Z]+[.]md$"},
            "result_tag": {"enum": ["ok", "skip"]},
        },
        "properties": {
            "path": {"$ref": "#/$defs/file_ref"},
            "result": {"$ref": "#/$defs/result_tag"},
        },
        "required": ["path", "result"],
        "additionalProperties": False,
    }
    valid_fake = FakeLLM(outputs=['<think>check</think>{"path":"README.md","result":"ok"}'])
    invalid_fake = FakeLLM(outputs=['{"path":"README.md","result":"done"}'])
    valid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=valid_fake)
    )
    invalid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=invalid_fake)
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "return json"}],
        "guided_json": json.dumps(schema),
    }

    valid = valid_client.post("/v1/chat/completions", json=payload)
    invalid = invalid_client.post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["message"]["content"] == '{"path":"README.md","result":"ok"}'
    assert valid_choice["message"]["reasoning_content"] == "check"
    assert valid_choice["finish_details"] == _stateless_finish_details("stop")
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["message"] == {"role": "assistant", "content": ""}
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "Return only JSON that satisfies this JSON schema" in valid_fake.calls[0][0][0]


def test_chat_guided_json_schema_length_rejects_invalid_json_continuation() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text='{"ok": [1}',
                finish_details=FinishDetails(reason="length", length_limit=9),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)
    schema_text = json.dumps(_response_json_schema()["schema"])

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "guided_json": schema_text,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": '{"ok": [1}'}
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "schema_violation",
        length_limit=9,
        phase="structured",
        continuation_eligible=False,
    )
    assert "continuation_id" not in choice


def test_chat_completion_guided_choice_validates_visible_content() -> None:
    valid_fake = FakeLLM(outputs=["<think>choose</think>no"])
    invalid_fake = FakeLLM(outputs=["<think>choose</think>maybe"])
    valid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=valid_fake)
    )
    invalid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=invalid_fake)
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "answer yes or no"}],
        "guided_choice": ["yes", "no"],
    }

    valid = valid_client.post("/v1/chat/completions", json=payload)
    invalid = invalid_client.post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["message"]["content"] == "no"
    assert valid_choice["message"]["reasoning_content"] == "choose"
    assert valid_choice["finish_details"] == _stateless_finish_details("stop")
    assert 'Return exactly one of these choices and no other text: ["yes","no"]' in valid_fake.calls[0][0][0]
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["message"] == {"role": "assistant", "content": ""}
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_chat_completion_guided_regex_validates_visible_content() -> None:
    valid_fake = FakeLLM(outputs=["<think>format</think>AB-12"])
    invalid_fake = FakeLLM(outputs=["<think>format</think>AB"])
    valid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=valid_fake)
    )
    invalid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=invalid_fake)
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "return an id"}],
        "guided_regex": r"[A-Z]{2}-\d{2}",
    }

    valid = valid_client.post("/v1/chat/completions", json=payload)
    invalid = invalid_client.post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["message"]["content"] == "AB-12"
    assert valid_choice["message"]["reasoning_content"] == "format"
    assert valid_choice["finish_details"] == _stateless_finish_details("stop")
    assert (
        'Return text that fully matches this regular expression and no other text: "[A-Z]{2}-\\\\d{2}"'
        in valid_fake.calls[0][0][0]
    )
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["message"] == {"role": "assistant", "content": ""}
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


def test_chat_completion_guided_patch_validates_visible_unified_diff() -> None:
    diff_text = _unified_diff_text()
    fenced_diff = f"<think>patch</think>```diff\n{diff_text}\n```"
    valid_fake = FakeLLM(outputs=[fenced_diff])
    invalid_fake = FakeLLM(outputs=[f"<think>patch</think>Here is the patch:\n{diff_text}"])
    valid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=valid_fake)
    )
    invalid_client = TestClient(
        create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=invalid_fake)
    )

    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "edit README"}],
        "guided_patch": {"format": "unified-diff"},
    }
    valid = valid_client.post("/v1/chat/completions", json=payload)
    invalid = invalid_client.post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["message"]["content"] == f"```diff\n{diff_text}\n```"
    assert valid_choice["message"]["reasoning_content"] == "patch"
    assert valid_choice["finish_details"] == _stateless_finish_details("stop")
    assert "Return only a valid unified diff patch" in valid_fake.calls[0][0][0]
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["message"] == {"role": "assistant", "content": ""}
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")


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
    continuation_id = choice["continuation_id"]
    assert choice["message"] == {"role": "assistant", "content": '{"ok":'}
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=6,
        phase="structured",
        continuation_eligible=True,
        continuation_id=continuation_id,
    )


def test_chat_completion_response_format_length_rejects_invalid_json_continuation() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text='{"ok": [1}',
                finish_details=FinishDetails(reason="length", length_limit=9),
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
    assert choice["message"] == {"role": "assistant", "content": '{"ok": [1}'}
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "schema_violation",
        length_limit=9,
        phase="structured",
        continuation_eligible=False,
    )
    assert "continuation_id" not in choice


def test_chat_completion_response_format_length_marks_complete_json_structured() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text='{"ok":true}',
                finish_details=FinishDetails(reason="length", length_limit=11),
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
    continuation_id = choice["continuation_id"]
    assert choice["message"] == {"role": "assistant", "content": '{"ok":true}'}
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=11,
        phase="structured",
        continuation_eligible=True,
        continuation_id=continuation_id,
    )


def test_chat_continuation_resumes_partial_guided_patch_and_inherits_validation() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old",
                finish_details=FinishDetails(reason="length", length_limit=12),
            ),
            GenerationOutput(text="\n+new", finish_details=FinishDetails(reason="stop")),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "edit README"}],
            "guided_patch": True,
            "max_tokens": 12,
            "temperature": 0.0,
        },
    )

    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    continuation_id = first_choice["continuation_id"]
    assert first_choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=12,
        phase="structured",
        continuation_eligible=True,
        continuation_id=continuation_id,
    )

    second = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert second.status_code == 200
    second_choice = second.json()["choices"][0]
    assert second_choice["message"]["content"].endswith("-old\n+new")
    assert second_choice["finish_reason"] == "stop"
    assert second_choice["finish_details"] == _stateless_finish_details("stop")


def test_chat_continuation_resumes_partial_json_and_inherits_response_format() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text='{"ok":',
                finish_details=FinishDetails(reason="length", length_limit=6),
            ),
            GenerationOutput(text="true}", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_object"},
            "max_tokens": 6,
            "temperature": 0.0,
        },
    )

    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    continuation_id = first_choice["continuation_id"]
    assert first_choice["message"] == {"role": "assistant", "content": '{"ok":'}
    assert first_choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=6,
        phase="structured",
        continuation_eligible=True,
        continuation_id=continuation_id,
    )

    second = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert second.status_code == 200
    second_choice = second.json()["choices"][0]
    assert second_choice["message"] == {"role": "assistant", "content": '{"ok":true}'}
    assert second_choice["finish_reason"] == "stop"
    assert second_choice["finish_details"] == _stateless_finish_details("eos", eos_token_id=151645)
    assert fake.calls[1][0][0].endswith('{"ok":')


def test_chat_continuation_resumes_partial_guided_json_and_inherits_validation() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text='{"ok":',
                finish_details=FinishDetails(reason="length", length_limit=6),
            ),
            GenerationOutput(text="true}", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "guided_json": True,
            "max_tokens": 6,
            "temperature": 0.0,
        },
    )

    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    continuation_id = first_choice["continuation_id"]
    assert first_choice["message"] == {"role": "assistant", "content": '{"ok":'}
    assert first_choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=6,
        phase="structured",
        continuation_eligible=True,
        continuation_id=continuation_id,
    )

    second = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert second.status_code == 200
    second_choice = second.json()["choices"][0]
    assert second_choice["message"] == {"role": "assistant", "content": '{"ok":true}'}
    assert second_choice["finish_reason"] == "stop"
    assert second_choice["finish_details"] == _stateless_finish_details("eos", eos_token_id=151645)
    assert fake.calls[1][0][0].endswith('{"ok":')


def test_chat_guided_json_length_rejects_invalid_json_continuation() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text='{"ok": [1}',
                finish_details=FinishDetails(reason="length", length_limit=9),
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
            "guided_json": True,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": '{"ok": [1}'}
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "schema_violation",
        length_limit=9,
        phase="structured",
        continuation_eligible=False,
    )
    assert "continuation_id" not in choice


def test_chat_continuation_resumes_partial_guided_regex_and_inherits_validation() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="AB-",
                finish_details=FinishDetails(reason="length", length_limit=3),
            ),
            GenerationOutput(text="12", finish_details=FinishDetails(reason="stop")),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return an id"}],
            "guided_regex": r"[A-Z]{2}-\d{2}",
            "max_tokens": 3,
            "temperature": 0.0,
        },
    )

    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    continuation_id = first_choice["continuation_id"]
    assert first_choice["message"] == {"role": "assistant", "content": "AB-"}
    assert first_choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=3,
        phase="structured",
        continuation_eligible=True,
        continuation_id=continuation_id,
    )

    second = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 2},
    )

    assert second.status_code == 200
    second_choice = second.json()["choices"][0]
    assert second_choice["message"] == {"role": "assistant", "content": "AB-12"}
    assert second_choice["finish_reason"] == "stop"
    assert second_choice["finish_details"] == _stateless_finish_details("stop")


def test_chat_continuation_resumes_partial_guided_choice_and_inherits_validation() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text="ye",
                finish_details=FinishDetails(reason="length", length_limit=2),
            ),
            GenerationOutput(text="s", finish_details=FinishDetails(reason="stop")),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "answer yes or no"}],
            "guided_choice": ["yes", "no"],
            "max_tokens": 2,
            "temperature": 0.0,
        },
    )

    assert first.status_code == 200
    first_choice = first.json()["choices"][0]
    continuation_id = first_choice["continuation_id"]
    assert first_choice["message"] == {"role": "assistant", "content": "ye"}
    assert first_choice["finish_details"] == _stateless_finish_details(
        "length",
        length_limit=2,
        phase="structured",
        continuation_eligible=True,
        continuation_id=continuation_id,
    )

    second = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 1},
    )

    assert second.status_code == 200
    second_choice = second.json()["choices"][0]
    assert second_choice["message"] == {"role": "assistant", "content": "yes"}
    assert second_choice["finish_reason"] == "stop"
    assert second_choice["finish_details"] == _stateless_finish_details("stop")


def test_chat_continuation_resume_rejects_explicit_response_format_override() -> None:
    fake = SequentialFakeLLM(
        [
            GenerationOutput(
                text='{"ok":',
                finish_details=FinishDetails(reason="length", length_limit=6),
            ),
            GenerationOutput(text="true}", finish_details=FinishDetails(reason="eos", eos_token_id=151645)),
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_object"},
            "max_tokens": 6,
        },
    )
    assert first.status_code == 200
    continuation_id = first.json()["choices"][0]["continuation_id"]

    override = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "continuation_id": continuation_id,
            "response_format": {"type": "text"},
            "max_tokens": 4,
        },
    )

    assert override.status_code == 400
    assert override.json()["error"]["code"] == "unsupported_parameter"
    assert override.json()["error"]["param"] == "response_format"
    assert continuation_id in app.state.hipengine_continuations
    assert len(fake.calls) == 1

    inherited = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": continuation_id, "max_tokens": 4},
    )

    assert inherited.status_code == 200
    assert inherited.json()["choices"][0]["message"] == {"role": "assistant", "content": '{"ok":true}'}
    assert len(fake.calls) == 2


def test_streaming_chat_completion_response_format_buffers_validation() -> None:
    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {
                        "request_id": 0,
                        "token_index": 0,
                        "token_id": 400,
                        "finished": True,
                        "chunk": {
                            "text": "not json",
                            "telemetry": GenerationTelemetry.from_decode_counts(
                                prompt_tokens=1,
                                generated_tokens=1,
                                row_index=0,
                                request_id="0",
                                phase="answer",
                                sampler_mode="greedy_fast",
                                execution_path="scheduler_invalid_should_not_surface",
                            ).to_json_dict(),
                        },
                    }
                ]
            }
            return [GenerationOutput(text="not json") for _prompt in prompt_tuple]

    fake = SchedulerChunkFakeLLM()
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
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("schema_violation"),
    }
    assert "scheduler_invalid_should_not_surface" not in response.text


def test_streaming_chat_completion_response_format_emits_structured_metadata() -> None:
    structured_text = '{"ok": true, "path": "README.md"}'
    fake = FakeLLM(outputs=[structured_text], stream_chunks=["wrong"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_object"},
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content = next(
        payload
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("content")
    )
    assert content["choices"][0]["delta"] == {"content": structured_text}
    assert content["choices"][0]["hipengine"] == {
        "phase": "structured",
        "tokens": {
            "streamed_tokens": 4,
            "delta_tokens": 4,
            "structured_tokens": 4,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 4,
            "prompt_tokens": 0,
            "generated_tokens": 4,
            "phase": "structured",
            "continuation_eligible": False,
            "structured_tokens": 4,
        },
    }
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("stop")
    done_hipengine = done["choices"][0]["hipengine"]
    assert done_hipengine["phase"] == "structured"
    assert done_hipengine["finish_details"] == _stateless_finish_details("stop")
    assert done_hipengine["tokens"] == {
        "streamed_tokens": 4,
        "structured_tokens": 4,
    }
    assert done_hipengine["decode_state"]["generated_tokens"] == 4
    assert done_hipengine["decode_state"]["phase"] == "structured"
    assert done_hipengine["decode_state"]["structured_tokens"] == 4
    usage = next(payload for payload in payloads if payload.get("usage"))
    assert usage["usage"]["completion_tokens"] == 4
    assert fake.stream_calls == []


def test_streaming_chat_completion_response_format_uses_scheduler_structured_chunks() -> None:
    structured_text = '{"ok": true}'
    chunk_texts = ('{"ok":', " true}")

    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {
                        "request_id": 0,
                        "token_index": token_index,
                        "token_id": 500 + token_index,
                        "finished": token_index == len(chunk_texts) - 1,
                        "chunk": {
                            "text": chunk_text,
                            "telemetry": GenerationTelemetry.from_decode_counts(
                                prompt_tokens=1,
                                generated_tokens=token_index + 1,
                                row_index=0,
                                request_id="0",
                                phase="answer",
                                sampler_mode="greedy_fast",
                                execution_path="scheduler_structured_chunks",
                            ).to_json_dict(),
                        },
                    }
                    for token_index, chunk_text in enumerate(chunk_texts)
                ]
            }
            return [
                GenerationOutput(
                    text=structured_text,
                    finish_details=FinishDetails(reason="stop", sampler_mode="greedy_fast"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=len(chunk_texts),
                        row_index=0,
                        phase="done",
                        sampler_mode="greedy_fast",
                    ),
                )
                for _prompt in prompt_tuple
            ]

    fake = SchedulerChunkFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": {"type": "json_object"},
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("content")
    ]
    assert [choice["delta"]["content"] for choice in content] == list(chunk_texts)
    assert [choice["hipengine"]["phase"] for choice in content] == ["structured", "structured"]
    assert {
        choice["hipengine"]["decode_state"]["execution_path"]
        for choice in content
    } == {"scheduler_structured_chunks"}
    assert {
        choice["hipengine"]["decode_state"]["phase"]
        for choice in content
    } == {"structured"}
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_details"] == {
        "reason": "stop",
        "cache_action": "append_none",
        "sampler_mode": "greedy_fast",
    }
    assert done["choices"][0]["hipengine"]["phase"] == "structured"
    assert done["choices"][0]["hipengine"]["decode_state"]["execution_path"] == "scheduler_structured_chunks"
    assert fake.stream_calls == []


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
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"] == {
        "phase": "structured",
        "finish_details": _stateless_finish_details("schema_violation"),
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
    assert choice["finish_details"] == _stateless_finish_details(
        "tool_calls",
        tool_call_tokens=1,
        phase="tool_call",
    )
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
    assert choice["finish_details"] == _stateless_finish_details(
        "tool_calls",
        reasoning_tokens=2,
        tool_call_tokens=1,
        phase="tool_call",
    )
    message = choice["message"]
    assert message["content"] == ""
    assert message["reasoning_content"] == "need file"
    tool_call = message["tool_calls"][0]
    assert tool_call["function"]["name"] == "read"
    assert json.loads(tool_call["function"]["arguments"]) == {"path": "README.md"}
    assert "<tool_call>" not in json.dumps(message)


def test_chat_completion_parses_tool_call_after_literal_marker_text() -> None:
    fake = FakeLLM(
        outputs=[
            'Mention <tool_call> literally, then call. '
            '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'
        ]
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
    message = choice["message"]
    assert message["content"] == "Mention <tool_call> literally, then call."
    assert len(message["tool_calls"]) == 1
    _assert_openai_tool_call_shape(message["tool_calls"][0], name="read", arguments={"path": "README.md"})


def test_chat_completion_parses_tool_call_with_marker_text_in_arguments() -> None:
    fake = FakeLLM(
        outputs=[
            '<tool_call>{"name":"read","arguments":{"path":"README.md",'
            '"needle":"</tool_call>"}}</tool_call>'
        ]
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
    message = choice["message"]
    assert message["content"] == ""
    _assert_openai_tool_call_shape(
        message["tool_calls"][0],
        name="read",
        arguments={"path": "README.md", "needle": "</tool_call>"},
    )


def test_chat_completion_auto_tool_rejects_literal_paired_tool_markup() -> None:
    raw_tool_markup = "Use <tool_call> to invoke tools.</tool_call>"
    fake = FakeLLM(outputs=[raw_tool_markup])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "explain tool syntax"}],
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
    assert raw_tool_markup not in response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("invalid_tool_call")
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_auto_tool_recovers_duplicated_start_marker() -> None:
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
    assert choice["finish_details"] == _stateless_finish_details(
        "tool_calls",
        tool_call_tokens=2,
        phase="tool_call",
    )
    message = choice["message"]
    assert set(message) == {"role", "content", "tool_calls"}
    assert message["role"] == "assistant"
    assert message["content"] == ""
    assert "<tool_call>" not in json.dumps(message)
    assert isinstance(message["tool_calls"], list)
    assert len(message["tool_calls"]) == 1
    tool_call = message["tool_calls"][0]
    _assert_openai_tool_call_shape(tool_call, name="read", arguments={"path": "README.md"})


def test_chat_completion_auto_tool_rejects_unparseable_tool_markup() -> None:
    raw_tool_markup = '<tool_call>{"name":"read","arguments":</tool_call>'
    fake = FakeLLM(outputs=[raw_tool_markup])
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
    assert raw_tool_markup not in response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("invalid_tool_call")
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_auto_tool_rejects_undeclared_function() -> None:
    fake = FakeLLM(
        outputs=['<tool_call>{"name":"write","arguments":{"path":"README.md"}}</tool_call>']
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
    assert "<tool_call>" not in response.text
    assert "write" not in response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("invalid_tool_call")
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_invalid_tool_call_can_return_hard_error() -> None:
    fake = FakeLLM(
        outputs=['<tool_call>{"name":"write","arguments":{"path":"README.md"}}</tool_call>']
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "invalid_tool_call_error_mode": "hard_error",
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

    assert response.status_code == 400
    assert "<tool_call>" not in response.text
    assert "write" not in response.text
    error = response.json()["error"]
    assert error["code"] == "invalid_tool_call"
    assert error["param"] == "tool_calls"
    assert error["finish_details"] == _stateless_finish_details("invalid_tool_call")
    assert error["hipengine"] == {
        "code": "invalid_tool_call",
        "status_code": 400,
        "retryable": False,
    }
    assert len(fake.calls) == 1


def test_chat_completion_rejects_invalid_tool_call_error_mode_before_generation() -> None:
    fake = FakeLLM(outputs=["unused"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "invalid_tool_call_error_mode": "sometimes",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["error"]["param"] == "invalid_tool_call_error_mode"
    assert fake.calls == []


def test_chat_completion_strict_validation_recovers_doubled_tool_call_tag() -> None:
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
    assert choice["finish_reason"] == "tool_calls"
    assert choice["finish_details"] == _stateless_finish_details(
        "tool_calls",
        tool_call_tokens=2,
        phase="tool_call",
    )
    message = choice["message"]
    assert set(message) == {"role", "content", "tool_calls"}
    assert message["role"] == "assistant"
    assert message["content"] == ""
    assert isinstance(message["tool_calls"], list)
    assert len(message["tool_calls"]) == 1
    tool_call = message["tool_calls"][0]
    _assert_openai_tool_call_shape(tool_call, name="read", arguments={"path": "README.md"})
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
    assert choice["finish_details"] == _stateless_finish_details("tool_required_not_satisfied")
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_required_tool_choice_rejects_missing_tools_without_generation() -> None:
    fake = FakeLLM(outputs=["ordinary answer"])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tool_choice": "required",
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "tool_choice"
    assert error["hipengine"]["code"] == "schema_violation"
    assert fake.calls == []


def test_chat_completion_required_tool_missing_call_preserves_length_context() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text="ordinary answer",
                finish_details=FinishDetails(reason="length", length_limit=7),
            )
        ]
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tool_choice": "required",
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "tool_required_not_satisfied",
        length_limit=7,
        phase="answer",
        continuation_eligible=False,
    )
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_required_tool_partial_call_preserves_length_context() -> None:
    fake = FakeLLM(
        detailed_outputs=[
            GenerationOutput(
                text='<tool_call>{"name":"read"',
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
            "messages": [{"role": "user", "content": "read the readme"}],
            "tool_choice": "required",
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "length"
    assert choice["finish_details"] == _stateless_finish_details(
        "invalid_tool_call",
        length_limit=6,
        phase="tool_call",
        continuation_eligible=False,
    )
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
    assert choice["finish_details"] == _stateless_finish_details("invalid_tool_call")
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_specific_tool_choice_rejects_unknown_function_without_generation() -> None:
    fake = FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'])
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "tool_choice": {"type": "function", "function": {"name": "write"}},
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "tool_choice.function.name"
    assert error["hipengine"]["code"] == "schema_violation"
    assert fake.calls == []


@pytest.mark.parametrize(
    ("tool_choice", "param"),
    [
        ({"type": "custom", "function": {"name": "read"}}, "tool_choice.type"),
        ({"type": "function"}, "tool_choice.function"),
        ({"type": "function", "function": "read"}, "tool_choice.function"),
        ({"type": "function", "function": {}}, "tool_choice.function.name"),
        ({"type": "function", "function": {"name": ""}}, "tool_choice.function.name"),
    ],
)
def test_chat_completion_malformed_tool_choice_rejects_without_generation(
    tool_choice: dict[str, Any],
    param: str,
) -> None:
    fake = FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>'])
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

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == param
    assert error["hipengine"]["code"] == "schema_violation"
    assert fake.calls == []


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
    assert choice["finish_details"] == _stateless_finish_details("invalid_tool_call")
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
    fake = FakeLLM(
        outputs=["ordinary answer"],
        token_map={
            "<tool_call>": [77, 78],
            '<tool_call>{"name":"read","arguments":': [77, 78, 90, 91, 92],
            "</tool_call>": [88, 89],
        },
    )
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
    assert fake.tokenize_calls == ["<tool_call>", '<tool_call>{"name":"read","arguments":', "</tool_call>"]
    params = fake.calls[-1][1]
    assert params.forced_tokens_pending == (77, 78)
    assert params.forced_token_reason == "tool_choice_required"
    assert params.force_sequence_completion_token_sequences == ((77, 78, 90, 91, 92), (88, 89))
    assert params.force_sequence_completion_reason == "tool_call_sequence_completion"
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("tool_required_not_satisfied")
    assert choice["message"] == {"role": "assistant", "content": ""}


def test_chat_completion_required_tool_choice_queues_tool_start_after_thinking_budget() -> None:
    fake = FakeLLM(
        outputs=["ordinary answer"],
        token_map={
            "</think>": [91, 92],
            "<tool_call>": [77, 78],
            '<tool_call>{"name":"read","arguments":': [77, 78, 90, 91, 92],
            "</tool_call>": [88, 89],
        },
    )
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
    assert fake.tokenize_calls == [
        "</think>",
        "<tool_call>",
        '<tool_call>{"name":"read","arguments":',
        "</tool_call>",
    ]
    params = fake.calls[-1][1]
    assert params.forced_tokens_pending == ()
    assert params.post_thinking_forced_tokens_pending == (77, 78)
    assert params.post_thinking_forced_token_reason == "tool_choice_required"
    assert params.force_sequence_completion_token_sequences == ((77, 78, 90, 91, 92), (88, 89))
    assert params.force_sequence_completion_reason == "tool_call_sequence_completion"
    assert params.thinking_close_token_ids == (91, 92)
    assert params.thinking_hard_token_cap == 512
    choice = response.json()["choices"][0]
    assert choice["finish_details"] == _stateless_finish_details("tool_required_not_satisfied")


def test_chat_completion_required_tool_choice_skips_name_prefix_with_multiple_tools() -> None:
    fake = FakeLLM(outputs=["ordinary answer"], token_map={"<tool_call>": [77, 78], "</tool_call>": [88, 89]})
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read or write"}],
            "tool_choice": "required",
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "write", "parameters": {"type": "object"}}},
            ],
        },
    )

    assert response.status_code == 200
    assert fake.tokenize_calls == ["<tool_call>", "</tool_call>"]
    params = fake.calls[-1][1]
    assert params.forced_tokens_pending == (77, 78)
    assert params.force_sequence_completion_token_sequences == ((88, 89),)
    assert params.force_sequence_completion_reason == "tool_call_sequence_completion"


def test_chat_completion_specific_tool_choice_skips_noncomposable_name_prefix() -> None:
    fake = FakeLLM(
        outputs=["ordinary answer"],
        token_map={
            "<tool_call>": [77, 78],
            '<tool_call>{"name":"read","arguments":': [90, 91, 92],
            "</tool_call>": [88, 89],
        },
    )
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
            ],
        },
    )

    assert response.status_code == 200
    assert fake.tokenize_calls == ["<tool_call>", '<tool_call>{"name":"read","arguments":', "</tool_call>"]
    params = fake.calls[-1][1]
    assert params.forced_tokens_pending == (77, 78)
    assert params.force_sequence_completion_token_sequences == ((88, 89),)
    assert params.force_sequence_completion_reason == "tool_call_sequence_completion"


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
    assert choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in choice["message"]


def test_chat_completion_strict_tool_schema_rejects_unsupported_keywords() -> None:
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
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "$anchor": "read_args",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "tools[0].function.parameters.$anchor"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_chat_completion_strict_tool_schema_validates_string_pattern() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "pattern": r"^README[.]md$"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>']),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"read","arguments":{"path":"WORKLOG.md"}}</tool_call>']),
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "read the readme"}],
        "tools": tools,
    }

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert json.loads(valid_choice["message"]["tool_calls"][0]["function"]["arguments"]) == {
        "path": "README.md"
    }
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_rejects_invalid_pattern_without_generation() -> None:
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
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string", "pattern": "["}},
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["param"] == "tools[0].function.parameters.properties.path.pattern"
    assert error["hipengine"]["code"] == "schema_violation"
    assert error["hipengine"]["legacy_code"] == "invalid_request"
    assert fake.calls == []


def test_chat_completion_strict_tool_schema_validates_property_count() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "minProperties": 2,
                    "maxProperties": 2,
                    "properties": {
                        "path": {"type": "string"},
                        "mode": {"type": "string"},
                        "extra": {"type": "string"},
                    },
                },
            },
        }
    ]
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=['<tool_call>{"name":"read","arguments":{"path":"README.md","mode":"raw"}}</tool_call>']
        ),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=[
                '<tool_call>{"name":"read","arguments":{"path":"README.md","mode":"raw","extra":"x"}}</tool_call>'
            ]
        ),
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "read the readme"}],
        "tools": tools,
    }

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_additional_properties_schema() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": {"type": "string"},
                },
            },
        }
    ]
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"ok":true,"note":"done"}}</tool_call>']),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"ok":true,"note":7}}</tool_call>']),
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record result"}],
        "tools": tools,
    }

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_pattern_properties() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "patternProperties": {r"^x-[a-z]+$": {"type": "integer"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"ok":true,"x-count":2}}</tool_call>']),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"ok":true,"x-count":"two"}}</tool_call>']),
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record result"}],
        "tools": tools,
    }

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_object_name_keywords() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "propertyNames": {"pattern": r"^(id|name|x-[a-z]+)$"},
                    "dependentRequired": {"id": ["name"]},
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                    },
                },
            },
        }
    ]
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record result"}],
        "tools": tools,
    }
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=['<tool_call>{"name":"record","arguments":{"id":"1","name":"doc","x-extra":"ok"}}</tool_call>']
        ),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"id":"1","x-extra":"ok"}}</tool_call>']),
    )

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_dependent_schemas() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": ["file", "url"]},
                        "path": {"type": "string"},
                        "href": {"type": "string", "pattern": r"^https://"},
                    },
                    "dependentSchemas": {
                        "path": {
                            "required": ["kind"],
                            "properties": {"kind": {"const": "file"}},
                        },
                        "href": {
                            "required": ["kind"],
                            "properties": {"kind": {"const": "url"}},
                        },
                    },
                    "additionalProperties": False,
                },
            },
        }
    ]
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record result"}],
        "tools": tools,
    }
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=['<tool_call>{"name":"record","arguments":{"kind":"file","path":"README.md"}}</tool_call>']
        ),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=['<tool_call>{"name":"record","arguments":{"kind":"url","path":"README.md"}}</tool_call>']
        ),
    )

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_conditional_schemas() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {"enum": ["file", "url"]},
                        "path": {"type": "string"},
                        "href": {"type": "string", "pattern": r"^https://"},
                    },
                    "if": {"required": ["kind"], "properties": {"kind": {"const": "file"}}},
                    "then": {"required": ["path"], "not": {"required": ["href"]}},
                    "else": {"required": ["href"], "not": {"required": ["path"]}},
                    "additionalProperties": False,
                },
            },
        }
    ]
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record result"}],
        "tools": tools,
    }
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=['<tool_call>{"name":"record","arguments":{"kind":"file","path":"README.md"}}</tool_call>']
        ),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=[
                '<tool_call>{"name":"record","arguments":{"kind":"file","href":"https://example.test"}}</tool_call>'
            ]
        ),
    )

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_local_refs() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "definitions": {
                        "file_ref": {"type": "string", "pattern": r"^[A-Z]+[.]md$"},
                        "result_tag": {"enum": ["ok", "skip"]},
                    },
                    "properties": {
                        "path": {"$ref": "#/definitions/file_ref"},
                        "result": {"$ref": "#/definitions/result_tag"},
                    },
                    "required": ["path", "result"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record result"}],
        "tools": tools,
    }
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=['<tool_call>{"name":"record","arguments":{"path":"README.md","result":"ok"}}</tool_call>']
        ),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=['<tool_call>{"name":"record","arguments":{"path":"readme.md","result":"ok"}}</tool_call>']
        ),
    )

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_multiple_of() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"score": {"type": "number", "multipleOf": 0.5}},
                    "required": ["score"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"score":1.5}}</tool_call>']),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"score":1.25}}</tool_call>']),
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record score"}],
        "tools": tools,
    }

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_unique_items() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tags": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
                    },
                    "required": ["tags"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"tags":["docs","api"]}}</tool_call>']),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"tags":["docs","docs"]}}</tool_call>']),
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record tags"}],
        "tools": tools,
    }

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_array_contains() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scores": {
                            "type": "array",
                            "contains": {"type": "number", "minimum": 0.9},
                            "minContains": 1,
                            "maxContains": 2,
                        }
                    },
                    "required": ["scores"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record score"}],
        "tools": tools,
    }
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"scores":[0.1,0.95]}}</tool_call>']),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(outputs=['<tool_call>{"name":"record","arguments":{"scores":[0.1,0.2]}}</tool_call>']),
    )

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_uses_json_typed_equality() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "const_num": {"const": 1},
                        "enum_num": {"enum": [1]},
                        "mixed": {"type": "array", "uniqueItems": True},
                    },
                    "required": ["const_num", "enum_num", "mixed"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record result"}],
        "tools": tools,
    }
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=[
                '<tool_call>{"name":"record","arguments":{"const_num":1,"enum_num":1,'
                '"mixed":[true,1,"1"]}}</tool_call>'
            ]
        ),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=[
                '<tool_call>{"name":"record","arguments":{"const_num":true,"enum_num":1,'
                '"mixed":[true,1,"1"]}}</tool_call>'
            ]
        ),
    )

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_validates_composition_keywords() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "record",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
                        "score": {
                            "type": "number",
                            "allOf": [{"minimum": 0}, {"maximum": 1}],
                        },
                        "label": {
                            "type": "string",
                            "allOf": [{"minLength": 2}, {"maxLength": 4}],
                        },
                        "ambiguous": {"oneOf": [{"type": "number"}, {"type": "integer"}]},
                        "debug": {"not": {"const": True}},
                    },
                    "required": ["target", "score", "label", "ambiguous", "debug"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    valid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=[
                (
                    '<tool_call>{"name":"record","arguments":'
                    '{"target":null,"score":0.5,"label":"ok","ambiguous":1.5,"debug":false}}</tool_call>'
                )
            ]
        ),
    )
    invalid_app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model"),
        llm=FakeLLM(
            outputs=[
                (
                    '<tool_call>{"name":"record","arguments":'
                    '{"target":null,"score":0.5,"label":"x","ambiguous":1.5,"debug":false}}</tool_call>'
                )
            ]
        ),
    )
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "record result"}],
        "tools": tools,
    }

    valid = TestClient(valid_app).post("/v1/chat/completions", json=payload)
    invalid = TestClient(invalid_app).post("/v1/chat/completions", json=payload)

    assert valid.status_code == 200
    valid_choice = valid.json()["choices"][0]
    assert valid_choice["finish_reason"] == "tool_calls"
    assert valid_choice["finish_details"]["reason"] == "tool_calls"
    assert invalid.status_code == 200
    invalid_choice = invalid.json()["choices"][0]
    assert invalid_choice["finish_reason"] == "stop"
    assert invalid_choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert "tool_calls" not in invalid_choice["message"]


def test_chat_completion_strict_tool_schema_accepts_annotation_keywords() -> None:
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
                        "strict": True,
                        "parameters": {
                            "title": "Read arguments",
                            "description": "Annotation-only metadata is ignored by validation.",
                            "default": {"path": "README.md"},
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Repository path",
                                    "examples": ["README.md"],
                                    "format": "uri-reference",
                                }
                            },
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
    assert choice["finish_reason"] == "tool_calls"
    assert choice["finish_details"]["reason"] == "tool_calls"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"path": "README.md"}


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
    assert response.json()["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")


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
    assert choice["finish_details"]["reason"] == "tool_calls"
    assert choice["finish_details"]["phase"] == "tool_call"
    assert choice["finish_details"]["tool_call_tokens"] > 0
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
    assert choice["finish_details"] == _stateless_finish_details("schema_violation")
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

    assert rejected_response.json()["choices"][0]["finish_details"] == _stateless_finish_details("invalid_tool_call")
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
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    tool_delta = next(payload for payload in payloads if payload["choices"][0]["delta"].get("tool_calls"))
    tool_call = tool_delta["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call["index"] == 0
    assert tool_call["id"].startswith("call_")
    assert tool_call["function"]["name"] == "bash"
    assert json.loads(tool_call["function"]["arguments"]) == {"command": "pwd"}
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert payloads[-1]["choices"][0]["finish_details"] == _stateless_finish_details(
        "tool_calls",
        tool_call_tokens=1,
        phase="tool_call",
    )


def test_streaming_chat_completion_parses_tool_call_after_literal_marker_text() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            "Mention <tool_call> literally, then call. ",
            '<tool_call>{"name":"read","arguments":{"path":"README.md"}}</tool_call>',
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "stream": True,
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
    assert '<tool_call>{"name"' not in response.text
    payloads = _sse_payloads(response.text)
    content = "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
        if payload.get("choices")
    )
    assert content == "Mention <tool_call> literally, then call."
    tool_choice = next(payload["choices"][0] for payload in payloads if payload["choices"][0]["delta"].get("tool_calls"))
    _assert_openai_stream_tool_call_delta_shape(
        tool_choice,
        name="read",
        arguments={"path": "README.md"},
    )
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_streaming_chat_completion_n_uses_scheduler_chunks_for_tool_call_arguments() -> None:
    raw_outputs = (
        '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>',
        '<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>',
    )
    chunk_texts = (
        (
            '<tool_call>{"name":"bash","arguments":',
            '{"command":',
            '"pwd"}',
            "}</tool_call>",
        ),
        (
            '<tool_call>{"name":"bash","arguments":',
            '{"command":',
            '"ls"}',
            "}</tool_call>",
        ),
    )

    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {
                        "request_id": request_id,
                        "token_index": token_index,
                        "token_id": 600 + request_id * 10 + token_index,
                        "finished": token_index == len(row) - 1,
                        "chunk": {
                            "text": text,
                            "telemetry": GenerationTelemetry.from_decode_counts(
                                prompt_tokens=1,
                                generated_tokens=token_index + 1,
                                row_index=request_id,
                                request_id=str(request_id),
                                phase="answer",
                                sampler_mode="greedy_fast",
                                execution_path="scheduler_tool_call_chunks",
                            ).to_json_dict(),
                        },
                    }
                    for request_id, row in enumerate(chunk_texts)
                    for token_index, text in enumerate(row)
                ]
            }
            return [
                GenerationOutput(
                    text=raw_outputs[index],
                    finish_details=FinishDetails(reason="stop", sampler_mode="greedy_fast"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=len(chunk_texts[index]),
                        row_index=index,
                        phase="done",
                        sampler_mode="greedy_fast",
                    ),
                )
                for index, _prompt in enumerate(prompt_tuple)
            ]

    fake = SchedulerChunkFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run shell commands"}],
            "n": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True},
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
    tool_choices = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("tool_calls")
    ]
    assert [
        (
            choice["index"],
            choice["delta"]["tool_calls"][0]["function"].get("name"),
            choice["delta"]["tool_calls"][0]["function"]["arguments"],
        )
        for choice in tool_choices
    ] == [
        (0, "bash", '{"command":'),
        (0, None, '"pwd"}'),
        (1, "bash", '{"command":'),
        (1, None, '"ls"}'),
    ]
    arguments_by_choice: dict[int, str] = {}
    for choice in tool_choices:
        call = choice["delta"]["tool_calls"][0]
        arguments_by_choice.setdefault(choice["index"], "")
        arguments_by_choice[choice["index"]] += call["function"]["arguments"]
    assert {index: json.loads(arguments) for index, arguments in arguments_by_choice.items()} == {
        0: {"command": "pwd"},
        1: {"command": "ls"},
    }
    assert {
        choice["hipengine"]["phase"]
        for choice in tool_choices
    } == {"tool_call"}
    assert {
        choice["hipengine"]["decode_state"]["phase"]
        for choice in tool_choices
    } == {"tool_call"}
    assert {
        choice["hipengine"]["decode_state"]["execution_path"]
        for choice in tool_choices
    } == {"scheduler_tool_call_chunks"}
    done = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] == "tool_calls"
    ]
    assert [choice["index"] for choice in done] == [0, 1]
    assert {
        choice["hipengine"]["decode_state"]["execution_path"]
        for choice in done
    } == {"scheduler_tool_call_chunks"}
    assert fake.stream_calls == []


def test_streaming_chat_completion_n_reports_withheld_scheduler_tool_chunks_for_invalid_tool_call() -> None:
    raw_outputs = (
        '<tool_call>{"name":"write","arguments":{"path":"README.md"}}</tool_call>',
        '<tool_call>{"name":"write","arguments":{"path":"WORKLOG.md"}}</tool_call>',
    )
    fake = SchedulerChunkRowsFakeLLM(
        raw_outputs,
        (
            (
                '<tool_call>{"name":"write","arguments":',
                '{"path":"README.md"}}</tool_call>',
            ),
            (
                '<tool_call>{"name":"write","arguments":',
                '{"path":"WORKLOG.md"}}</tool_call>',
            ),
        ),
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read files"}],
            "n": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True},
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
    assert "<tool_call>" not in response.text
    assert "write" not in response.text
    assert "README.md" not in response.text
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0]["delta"].get("tool_calls") for payload in payloads)
    done = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"]
    ]
    assert [choice["index"] for choice in done] == [0, 1]
    for index, choice in enumerate(done):
        assert choice["finish_reason"] == "stop"
        assert choice["finish_details"] == _stateless_finish_details("invalid_tool_call")
        diagnostic = choice["hipengine"]["withheld_scheduler_tool_chunks"]
        assert diagnostic == {
            "surface": "chat_tool_argument_delta",
            "reason": "invalid_tool_call",
            "public_delta": "withheld",
            "chunk_count": 2,
            "malformed_chunk_count": 0,
            "raw_text_matches": True,
            "text_bytes": len(raw_outputs[index].encode("utf-8")),
            "text_sha256": hashlib.sha256(raw_outputs[index].encode("utf-8")).hexdigest(),
            "chunk_text_bytes": [
                len(chunk.encode("utf-8")) for chunk in fake.chunk_rows[index]
            ],
            "execution_paths": ["scheduler_tool_call_chunks"],
        }
    assert fake.stream_calls == []


def test_streaming_chat_completion_n_reports_unmappable_scheduler_tool_chunks() -> None:
    raw_output = '<tool_call>{"name":"bash","arguments":"{\\"command\\":\\"pwd\\"}"}</tool_call>'
    fake = SchedulerChunkRowsFakeLLM(
        (raw_output, raw_output),
        (
            (
                '<tool_call>{"name":"bash","arguments":',
                '"{\\"command\\":\\"pwd\\"}"}</tool_call>',
            ),
            (
                '<tool_call>{"name":"bash","arguments":',
                '"{\\"command\\":\\"pwd\\"}"}</tool_call>',
            ),
        ),
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run pwd"}],
            "n": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True},
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
    tool_choices = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["delta"].get("tool_calls")
    ]
    assert [choice["index"] for choice in tool_choices] == [0, 1]
    for choice in tool_choices:
        _assert_openai_stream_tool_call_delta_shape(
            choice,
            name="bash",
            arguments={"command": "pwd"},
            index=choice["index"],
        )
        assert choice["hipengine"]["phase"] == "tool_call"
        assert choice["hipengine"]["decode_state"]["phase"] == "tool_call"
        assert choice["hipengine"]["decode_state"].get("execution_path") != "scheduler_tool_call_chunks"

    done = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"]
    ]
    assert [choice["index"] for choice in done] == [0, 1]
    for choice in done:
        assert choice["finish_reason"] == "tool_calls"
        assert choice["finish_details"] == _stateless_finish_details(
            "tool_calls",
            tool_call_tokens=1,
            phase="tool_call",
        )
        diagnostic = choice["hipengine"]["withheld_scheduler_tool_chunks"]
        assert diagnostic["reason"] == "unmappable_tool_arguments"
        assert diagnostic["public_delta"] == "withheld"
        assert diagnostic["chunk_count"] == 2
        assert diagnostic["raw_text_matches"] is True
        assert diagnostic["text_sha256"] == hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
        assert diagnostic["execution_paths"] == ["scheduler_tool_call_chunks"]
        assert "text" not in diagnostic
    assert fake.stream_calls == []


def test_streaming_chat_completion_recovers_duplicated_tool_start_marker() -> None:
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
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    tool_delta = next(payload for payload in payloads if payload["choices"][0]["delta"].get("tool_calls"))
    _assert_openai_stream_tool_call_delta_shape(
        tool_delta["choices"][0],
        name="bash",
        arguments={"command": "pwd"},
    )
    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["delta"] == {}
    assert done["choices"][0]["finish_reason"] == "tool_calls"
    assert done["choices"][0]["finish_details"] == _stateless_finish_details(
        "tool_calls",
        tool_call_tokens=2,
        phase="tool_call",
    )


def test_streaming_chat_completion_rejects_undeclared_auto_tool_name() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=['<tool_call>{"name":"write","arguments":{"path":"README.md"}}</tool_call>'],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "stream": True,
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
    assert "<tool_call>" not in response.text
    assert "write" not in response.text
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0]["delta"].get("tool_calls") for payload in payloads)
    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_reason"] == "stop"
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("invalid_tool_call")


def test_streaming_chat_completion_invalid_tool_call_can_return_sse_error() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=['<tool_call>{"name":"write","arguments":{"path":"README.md"}}</tool_call>'],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "read the readme"}],
            "stream": True,
            "stream_options": {"include_hipengine": True},
            "invalid_tool_call_error_mode": "hard_error",
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
    assert "<tool_call>" not in response.text
    assert "write" not in response.text
    payloads = _sse_payloads(response.text)
    error_payload = next(payload for payload in payloads if payload.get("error"))
    assert error_payload["choices"][0]["finish_reason"] == "error"
    assert error_payload["choices"][0]["finish_details"] == _stateless_finish_details("invalid_tool_call")
    assert error_payload["error"]["code"] == "invalid_tool_call"
    assert error_payload["error"]["param"] == "tool_calls"
    assert error_payload["error"]["finish_details"] == _stateless_finish_details("invalid_tool_call")
    assert error_payload["error"]["hipengine"] == {
        "code": "invalid_tool_call",
        "status_code": 400,
        "retryable": False,
    }
    assert error_payload["hipengine"]["event"] == "error"
    assert "data: [DONE]" in response.text


def test_streaming_chat_completion_auto_tool_rejects_unparseable_tool_markup() -> None:
    raw_tool_markup = '<tool_call>{"name":"bash","arguments":</tool_call>'
    fake = FakeLLM(outputs=["should-not-buffer"], stream_chunks=[raw_tool_markup])
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
    assert raw_tool_markup not in response.text
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0]["delta"].get("tool_calls") for payload in payloads)
    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_reason"] == "stop"
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("invalid_tool_call")


def test_streaming_chat_completion_chunks_long_tool_call_arguments() -> None:
    command = "x" * 300
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[f'<tool_call>{{"name":"bash","arguments":{{"command":{json.dumps(command)}}}}}</tool_call>'],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "run a long command"}],
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
    chunks = [
        payload["choices"][0]["delta"]["tool_calls"][0]
        for payload in payloads
        if payload["choices"][0]["delta"].get("tool_calls")
    ]
    assert len(chunks) > 1
    assert {chunk["id"] for chunk in chunks} == {chunks[0]["id"]}
    assert {chunk["index"] for chunk in chunks} == {0}
    assert chunks[0]["function"]["name"] == "bash"
    assert all("name" not in chunk["function"] for chunk in chunks[1:])
    argument_text = "".join(chunk["function"]["arguments"] for chunk in chunks)
    assert json.loads(argument_text) == {"command": command}
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"


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
            "stream_options": {"include_hipengine": True},
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
    tool_delta = next(payload for payload in payloads if payload["choices"][0]["delta"].get("tool_calls"))
    tool_call = tool_delta["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "bash"
    assert json.loads(tool_call["function"]["arguments"]) == {"command": "pwd"}
    assert tool_delta["choices"][0]["hipengine"] == {
        "phase": "tool_call",
        "tokens": {
            "streamed_tokens": 3,
            "delta_tokens": 1,
            "reasoning_tokens": 2,
            "tool_call_tokens": 1,
        },
        "decode_state": {
            "row_index": 0,
            "step_index": 3,
            "prompt_tokens": 0,
            "generated_tokens": 3,
            "phase": "tool_call",
            "continuation_eligible": False,
            "reasoning_tokens": 2,
            "tool_call_tokens": 1,
        },
    }
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert payloads[-1]["choices"][0]["finish_details"] == _stateless_finish_details(
        "tool_calls",
        reasoning_tokens=2,
        tool_call_tokens=1,
        phase="tool_call",
    )
    done_hipengine = payloads[-1]["choices"][0]["hipengine"]
    assert done_hipengine["phase"] == "done"
    assert done_hipengine["finish_details"] == _stateless_finish_details(
        "tool_calls",
        reasoning_tokens=2,
        tool_call_tokens=1,
        phase="tool_call",
    )
    assert done_hipengine["tokens"]["streamed_tokens"] == 3
    assert done_hipengine["tokens"]["reasoning_tokens"] == 2
    assert done_hipengine["tokens"]["tool_call_tokens"] == 1
    assert done_hipengine["tokens"]["prompt_tokens"] == fake.count_tokens(fake.calls[0][0][0])
    assert done_hipengine["decode_state"]["step_index"] == 3
    assert done_hipengine["decode_state"]["prompt_tokens"] == done_hipengine["tokens"]["prompt_tokens"]
    assert done_hipengine["decode_state"]["generated_tokens"] == done_hipengine["tokens"]["completion_tokens"]
    assert done_hipengine["decode_state"]["phase"] == "done"
    assert done_hipengine["decode_state"]["reasoning_tokens"] == 2
    assert done_hipengine["decode_state"]["tool_call_tokens"] == 1


def test_streaming_chat_completion_strict_validation_recovers_doubled_tool_call_tag() -> None:
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
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    tool_payload = next(payload for payload in payloads if payload["choices"][0]["delta"].get("tool_calls"))
    _assert_openai_stream_tool_call_delta_shape(
        tool_payload["choices"][0],
        name="bash",
        arguments={"command": "pwd"},
    )
    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["delta"] == {}
    assert done["choices"][0]["finish_reason"] == "tool_calls"
    assert done["choices"][0]["finish_details"] == _stateless_finish_details(
        "tool_calls",
        tool_call_tokens=2,
        phase="tool_call",
    )


def test_streaming_chat_completion_strict_validation_rejects_malformed_tool_json() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=['<tool_call>{"name":"bash","arguments":</tool_call>'],
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
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("invalid_tool_call")


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
    assert payloads[-1]["choices"][0]["finish_details"] == _stateless_finish_details(
        "tool_calls",
        tool_call_tokens=2,
        phase="tool_call",
    )


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
    assert "<tool_call>" not in response.text
    assert '"command":7' not in response.text
    payloads = _sse_payloads(response.text)
    assert not any(payload["choices"][0]["delta"].get("tool_calls") for payload in payloads)
    assert not any(payload["choices"][0]["delta"].get("content") for payload in payloads)
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert payloads[-1]["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")


def test_streaming_chat_timeout_can_include_hipengine_error_metadata() -> None:
    fake = DelayedFakeLLM(outputs=["ok"], stream_chunks=["late"], stream_delay_s=0.1)
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "timeout_ms": 50,
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
    assert "ttft_ms" not in payloads[0]["hipengine"]["timing"]
    assert all(payload["hipengine"]["routing"] == _routing_metadata() for payload in payloads)

    reasoning = next(payload for payload in payloads if payload.get("choices") and "reasoning_content" in payload["choices"][0]["delta"])
    assert reasoning["hipengine"]["event"] == "delta"
    assert isinstance(reasoning["hipengine"]["timing"]["ttft_ms"], float)
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
    assert isinstance(done["hipengine"]["timing"]["ttft_ms"], float)
    assert isinstance(done["hipengine"]["timing"]["decode_elapsed_ms"], float)
    assert isinstance(done["hipengine"]["timing"]["decode_tokens_per_second"], float)
    assert done["choices"][0]["hipengine"] == {
        "phase": "done",
        "finish_details": _stateless_finish_details("stop"),
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
    assert isinstance(payloads[-1]["hipengine"]["timing"]["ttft_ms"], float)
    assert isinstance(payloads[-1]["hipengine"]["timing"]["decode_elapsed_ms"], float)
    assert isinstance(payloads[-1]["hipengine"]["timing"]["decode_tokens_per_second"], float)


def test_streaming_chat_completion_n_uses_scheduler_token_chunks_for_buffered_answer_deltas() -> None:
    class SchedulerChunkFakeLLM(FakeLLM):
        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.calls.append((prompt_tuple, sampling_params))
            outputs = [
                GenerationOutput(
                    text="<think>r0</think>A",
                    finish_details=FinishDetails(reason="length", length_limit=2, sampler_mode="greedy_fast"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=4,
                        row_index=0,
                        phase="done",
                        sampler_mode="greedy_fast",
                    ),
                ),
                GenerationOutput(
                    text="BC",
                    finish_details=FinishDetails(reason="length", length_limit=2, sampler_mode="greedy_fast"),
                    telemetry=GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=2,
                        row_index=1,
                        phase="done",
                        sampler_mode="greedy_fast",
                    ),
                ),
            ]
            chunk_texts = (("<thi", "nk>r0</think>", "A"), ("B", "C"))
            self.last_batch_generation = {
                "scheduler_token_chunks": [
                    {
                        "request_id": request_id,
                        "token_index": token_index,
                        "token_id": 200 + request_id * 10 + token_index,
                        "finished": token_index == len(row) - 1,
                        "chunk": {
                            "text": text,
                            "telemetry": GenerationTelemetry.from_decode_counts(
                                prompt_tokens=1,
                                generated_tokens=token_index + 1,
                                row_index=request_id,
                                request_id=str(request_id),
                                phase="answer",
                                sampler_mode="greedy_fast",
                                execution_path="scheduler_native_packed_prefill_serial_decode",
                            ).to_json_dict(),
                        },
                    }
                    for request_id, row in enumerate(chunk_texts)
                    for token_index, text in enumerate(row)
                ]
            }
            return outputs[: len(prompt_tuple)]

    fake = SchedulerChunkFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "n": 2,
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    deltas = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] is None
    ]
    assert [(choice["index"], choice["delta"]) for choice in deltas] == [
        (0, {"role": "assistant"}),
        (0, {"reasoning_content": "r0"}),
        (0, {"content": "A"}),
        (1, {"role": "assistant"}),
        (1, {"content": "B"}),
        (1, {"content": "C"}),
    ]
    chunk_deltas = [choice for choice in deltas if "role" not in choice["delta"]]
    assert [
        choice["hipengine"]["decode_state"]["generated_tokens"]
        for choice in chunk_deltas
    ] == [2, 3, 1, 2]
    assert {
        choice["hipengine"]["decode_state"]["execution_path"]
        for choice in chunk_deltas
    } == {"scheduler_native_packed_prefill_serial_decode"}
    assert [
        choice["hipengine"]["phase"]
        for choice in chunk_deltas
    ] == ["think", "answer", "answer", "answer"]
    done = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] == "length"
    ]
    assert [(choice["index"], choice["finish_details"]) for choice in done] == [
        (
            0,
            {
                "reason": "length",
                "length_limit": 2,
                "cache_action": "append_none",
                "sampler_mode": "greedy_fast",
                "phase": "answer",
                "continuation_eligible": False,
            },
        ),
        (
            1,
            {
                "reason": "length",
                "length_limit": 2,
                "cache_action": "append_none",
                "sampler_mode": "greedy_fast",
                "phase": "answer",
                "continuation_eligible": False,
            },
        ),
    ]
    prompt_tokens = sum(fake.count_tokens(prompt) for prompt in fake.calls[0][0])
    assert payloads[-1]["usage"] == {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 2,
        "total_tokens": prompt_tokens + 2,
    }
    assert "data: [DONE]" in response.text
    assert len(fake.calls) == 1
    assert fake.stream_calls == []


def test_streaming_chat_completion_n_forwards_runtime_native_live_chunks() -> None:
    class LiveManyFakeLLM(FakeLLM):
        supports_stream_many = True

        def __init__(self) -> None:
            super().__init__(outputs=["should-not-buffer"])
            self.stream_many_calls: list[tuple[tuple[str, ...], SamplingParams]] = []

        def stream_many_detailed(self, prompts, sampling_params: SamplingParams):
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.stream_many_calls.append((prompt_tuple, sampling_params))
            chunks = (
                (
                    0,
                    "<think>r0</think>",
                    GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=1,
                        row_index=0,
                        request_id="0",
                        phase="think",
                        sampler_mode="greedy_fast",
                        execution_path="runtime_native_live_many",
                    ),
                    None,
                ),
                (
                    1,
                    "B",
                    GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=1,
                        row_index=1,
                        request_id="1",
                        phase="answer",
                        sampler_mode="greedy_fast",
                        execution_path="runtime_native_live_many",
                    ),
                    None,
                ),
                (
                    0,
                    "A",
                    GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=2,
                        row_index=0,
                        request_id="0",
                        phase="answer",
                        sampler_mode="greedy_fast",
                        execution_path="runtime_native_live_many",
                    ),
                    FinishDetails(reason="length", length_limit=2, sampler_mode="greedy_fast"),
                ),
                (
                    1,
                    "C",
                    GenerationTelemetry.from_decode_counts(
                        prompt_tokens=1,
                        generated_tokens=2,
                        row_index=1,
                        request_id="1",
                        phase="answer",
                        sampler_mode="greedy_fast",
                        execution_path="runtime_native_live_many",
                    ),
                    FinishDetails(reason="length", length_limit=2, sampler_mode="greedy_fast"),
                ),
            )
            assert tuple(row for row, _text, _telemetry, _finish in chunks) == (0, 1, 0, 1)
            for _row, text, telemetry, finish_details in chunks:
                yield GenerationStreamChunk(
                    text=text,
                    telemetry=telemetry,
                    finish_details=finish_details,
                )

    fake = LiveManyFakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    capabilities = client.get("/v1/hipengine/capabilities")
    assert capabilities.status_code == 200
    assert capabilities.json()["features"]["stream_metadata"]["live_many_chunks"]["available"] is True

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            "n": 2,
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True, "include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    choices = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] is None
    ]
    assert [(choice["index"], choice["delta"]) for choice in choices] == [
        (0, {"role": "assistant"}),
        (1, {"role": "assistant"}),
        (0, {"reasoning_content": "r0"}),
        (1, {"content": "B"}),
        (0, {"content": "A"}),
        (1, {"content": "C"}),
    ]
    token_choices = [choice for choice in choices if "role" not in choice["delta"]]
    assert [
        (
            choice["index"],
            choice["hipengine"]["phase"],
            choice["hipengine"]["decode_state"]["row_index"],
            choice["hipengine"]["decode_state"]["execution_path"],
        )
        for choice in token_choices
    ] == [
        (0, "think", 0, "runtime_native_live_many"),
        (1, "answer", 1, "runtime_native_live_many"),
        (0, "answer", 0, "runtime_native_live_many"),
        (1, "answer", 1, "runtime_native_live_many"),
    ]
    done = [
        payload["choices"][0]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0]["finish_reason"] == "length"
    ]
    assert [(choice["index"], choice["finish_details"]) for choice in done] == [
        (
            0,
            {
                "reason": "length",
                "length_limit": 2,
                "cache_action": "append_none",
                "sampler_mode": "greedy_fast",
                "phase": "answer",
                "continuation_eligible": False,
            },
        ),
        (
            1,
            {
                "reason": "length",
                "length_limit": 2,
                "cache_action": "append_none",
                "sampler_mode": "greedy_fast",
                "phase": "answer",
                "continuation_eligible": False,
            },
        ),
    ]
    prompt_tokens = sum(fake.count_tokens(prompt) for prompt in fake.stream_many_calls[0][0])
    assert payloads[-1]["usage"] == {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 2,
        "total_tokens": prompt_tokens + 2,
    }
    assert len(fake.stream_many_calls) == 1
    assert fake.calls == []
    assert fake.stream_calls == []
    assert "data: [DONE]" in response.text


def test_streaming_chat_completion_n_buffers_live_many_unsupported_surfaces() -> None:
    class LiveManyCapableFakeLLM(FakeLLM):
        supports_stream_many = True

        def __init__(self, outputs: list[str], *, include_logprobs: bool = False) -> None:
            super().__init__(outputs=outputs)
            self.include_logprobs = bool(include_logprobs)
            self.stream_many_calls: list[tuple[tuple[str, ...], SamplingParams]] = []

        def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
            if not self.include_logprobs:
                return super().generate_detailed(prompts, sampling_params)
            prompt_tuple = tuple(prompts)
            self.calls.append((prompt_tuple, sampling_params))
            return [
                GenerationOutput(
                    text=output,
                    token_logprobs=(
                        TokenLogprob(
                            token_id=700 + index,
                            token_text=output,
                            logprob=-0.1,
                            top_logprobs=((700 + index, output, -0.1),),
                        ),
                    ),
                )
                for index, output in enumerate(self.outputs or ())
            ][: len(prompt_tuple)]

        def stream_many_detailed(self, prompts, sampling_params: SamplingParams):
            prompt_tuple = tuple(str(prompt) for prompt in prompts)
            self.stream_many_calls.append((prompt_tuple, sampling_params))
            raise AssertionError("unsupported live-many surfaces must use the buffered path")

    tool_request = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a command",
                    "parameters": {"type": "object"},
                },
            }
        ]
    }
    cases: tuple[tuple[str, dict[str, Any], list[str], bool], ...] = (
        (
            "tools",
            tool_request,
            [
                '<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>',
                '<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>',
            ],
            False,
        ),
        ("logprobs", {"logprobs": True}, ["alpha", "beta"], True),
        ("response_format", {"response_format": {"type": "json_object"}}, ['{"ok":true}', '{"ok":false}'], False),
        ("stop", {"stop": "<stop>"}, ["alpha<stop>tail", "beta<stop>tail"], False),
    )

    for label, request_extra, outputs, include_logprobs in cases:
        fake = LiveManyCapableFakeLLM(outputs, include_logprobs=include_logprobs)
        app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
        client = TestClient(app)

        capabilities = client.get("/v1/hipengine/capabilities")
        assert capabilities.status_code == 200
        assert capabilities.json()["features"]["stream_metadata"]["live_many_chunks"]["available"] is True

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-model",
                "messages": [{"role": "user", "content": f"{label} request"}],
                "n": 2,
                "stream": True,
                "stream_options": {"include_hipengine": True},
                **request_extra,
            },
        )

        assert response.status_code == 200, (label, response.text)
        assert fake.stream_many_calls == []
        assert len(fake.calls) == 1
        assert len(fake.calls[0][0]) == 2
        assert "data: [DONE]" in response.text
        payloads = _sse_payloads(response.text)
        choices = [
            payload["choices"][0]
            for payload in payloads
            if payload.get("choices")
        ]
        done = [choice for choice in choices if choice["finish_reason"] is not None]
        assert [choice["index"] for choice in done] == [0, 1]
        if label == "tools":
            assert "<tool_call>" not in response.text
            tool_choices = [
                choice for choice in choices if choice["delta"].get("tool_calls")
            ]
            assert [choice["index"] for choice in tool_choices] == [0, 1]
            for choice, command in zip(tool_choices, ("pwd", "ls"), strict=True):
                _assert_openai_stream_tool_call_delta_shape(
                    choice,
                    name="bash",
                    arguments={"command": command},
                    index=choice["index"],
                )
            assert [choice["finish_reason"] for choice in done] == ["tool_calls", "tool_calls"]
        elif label == "logprobs":
            assert any(choice.get("logprobs") is not None for choice in choices)
            assert [choice["finish_reason"] for choice in done] == ["stop", "stop"]
        elif label == "response_format":
            assert [choice["finish_details"]["reason"] for choice in done] == ["stop", "stop"]
            content = "".join(
                choice["delta"].get("content", "")
                for choice in choices
                if choice["index"] == 0
            )
            assert json.loads(content) == {"ok": True}
        elif label == "stop":
            content_by_index = {
                index: "".join(
                    choice["delta"].get("content", "")
                    for choice in choices
                    if choice["index"] == index
                )
                for index in (0, 1)
            }
            assert content_by_index == {0: "alpha", 1: "beta"}
            assert [choice["finish_reason"] for choice in done] == ["stop", "stop"]


def test_streaming_chat_completion_can_include_kv_pool_metadata() -> None:
    fake = FakeLLM(outputs=["should-not-buffer"], stream_chunks=["streamed reply"])
    fake.kv_pool_stats = _fake_kv_pool_stats()
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
    assert "kv_pool" not in payloads[0]["hipengine"]
    done = next(payload for payload in payloads if payload["hipengine"]["event"] == "done")
    usage = next(payload for payload in payloads if payload["hipengine"]["event"] == "usage")
    assert done["hipengine"]["kv_pool"] == _fake_kv_pool_metadata()
    assert usage["hipengine"]["kv_pool"] == _fake_kv_pool_metadata()


def test_streaming_chat_completion_prefers_backend_chunk_decode_state() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                "backend reply",
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=7,
                    generated_tokens=3,
                    phase="answer",
                    sampler_mode="processed_argmax",
                    sampler_fallback_reason="processed_logits_required",
                    forced_token_id=78,
                    forced_token_reason="tool_choice_required",
                    forced_tokens_remaining=1,
                    active_processors=("logit_bias",),
                    sampler_fast_path_blockers=("logit_bias",),
                    timing={"prefill_ms": 4.0, "decode_ms": 2.0},
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
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    content = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["delta"].get("content"))
    assert content["choices"][0]["delta"] == {"content": "backend reply"}
    assert content["choices"][0]["hipengine"] == {
        "phase": "answer",
        "decode_state": {
            "row_index": 0,
            "step_index": 3,
            "prompt_tokens": 7,
            "generated_tokens": 3,
            "phase": "answer",
            "continuation_eligible": False,
            "forced_token_id": 78,
            "forced_token_reason": "tool_choice_required",
            "forced_tokens_remaining": 1,
            "active_processors": ["logit_bias"],
            "sampler_fast_path_blockers": ["logit_bias"],
            "sampler_fallback_reason": "processed_logits_required",
            "sampler_mode": "processed_argmax",
        },
        "timing": {"prefill_ms": 4.0, "decode_ms": 2.0},
        "tokens": {
            "streamed_tokens": 2,
            "delta_tokens": 2,
            "answer_tokens": 2,
        },
    }
    assert content["hipengine"]["timing"]["backend_prefill_ms"] == 4.0
    assert content["hipengine"]["timing"]["backend_decode_ms"] == 2.0
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["hipengine"]["timing"]["backend_prefill_ms"] == 4.0
    assert done["hipengine"]["timing"]["backend_decode_ms"] == 2.0


def test_streaming_chat_completion_prefers_backend_chunk_finish_details() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                "backend reply",
                finish_details=FinishDetails(
                    reason="length",
                    length_limit=4,
                    sampler_mode="processed_argmax",
                    phase="answer",
                    continuation_eligible=False,
                ),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=7,
                    generated_tokens=4,
                    phase="answer",
                    sampler_mode="processed_argmax",
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
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    expected_finish = {
        "reason": "length",
        "length_limit": 4,
        "cache_action": "append_none",
        "sampler_mode": "processed_argmax",
        "phase": "answer",
        "continuation_eligible": False,
    }
    assert done["choices"][0]["finish_reason"] == "length"
    assert done["choices"][0]["finish_details"] == expected_finish
    assert done["choices"][0]["hipengine"]["finish_details"] == expected_finish
    assert done["choices"][0]["hipengine"]["decode_state"] == {
        "row_index": 0,
        "step_index": 4,
        "prompt_tokens": 7,
        "generated_tokens": 4,
        "phase": "answer",
        "continuation_eligible": False,
        "sampler_mode": "processed_argmax",
    }


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
    assert all(payload["hipengine"]["routing"] == _routing_metadata() for payload in payloads)
    assert isinstance(payloads[0]["hipengine"]["timing"]["ttft_ms"], float)
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
    assert isinstance(payloads[2]["hipengine"]["timing"]["ttft_ms"], float)
    assert isinstance(payloads[2]["hipengine"]["timing"]["decode_elapsed_ms"], float)
    assert isinstance(payloads[2]["hipengine"]["timing"]["decode_tokens_per_second"], float)
    assert payloads[2]["choices"][0]["hipengine"] == {
        "phase": "done",
        "finish_details": _stateless_finish_details("stop"),
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
    assert isinstance(payloads[-1]["hipengine"]["timing"]["ttft_ms"], float)
    assert isinstance(payloads[-1]["hipengine"]["timing"]["decode_elapsed_ms"], float)
    assert isinstance(payloads[-1]["hipengine"]["timing"]["decode_tokens_per_second"], float)


def test_streaming_completion_can_include_kv_pool_metadata() -> None:
    fake = FakeLLM(outputs=["should-not-buffer"], stream_chunks=["alpha", " beta"])
    fake.kv_pool_stats = _fake_kv_pool_stats()
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
    assert "kv_pool" not in payloads[0]["hipengine"]
    done = next(payload for payload in payloads if payload["hipengine"]["event"] == "done")
    usage = next(payload for payload in payloads if payload["hipengine"]["event"] == "usage")
    assert done["hipengine"]["kv_pool"] == _fake_kv_pool_metadata()
    assert usage["hipengine"]["kv_pool"] == _fake_kv_pool_metadata()


def test_streaming_completion_prefers_backend_chunk_decode_state() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                "alpha",
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=5,
                    generated_tokens=4,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                    sampler_fallback_reason="host_sampling_required",
                    forced_token_id=79,
                    forced_token_reason="thinking_hard_close",
                    forced_tokens_remaining=0,
                    sampler_fast_path_blockers=("temperature",),
                    timing={"prefill_ms": 4.0, "decode_ms": 2.0},
                    usage={"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
                ),
            )
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 1,
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert payloads[0]["choices"][0]["text"] == "alpha"
    assert payloads[0]["choices"][0]["hipengine"] == {
        "phase": "answer",
        "decode_state": {
            "row_index": 0,
            "step_index": 4,
            "prompt_tokens": 5,
            "generated_tokens": 4,
            "phase": "answer",
            "continuation_eligible": False,
            "forced_token_id": 79,
            "forced_token_reason": "thinking_hard_close",
            "forced_tokens_remaining": 0,
            "sampler_fast_path_blockers": ["temperature"],
            "sampler_fallback_reason": "host_sampling_required",
            "sampler_mode": "host_logits_sample",
        },
        "timing": {"prefill_ms": 4.0, "decode_ms": 2.0},
        "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
        "tokens": {
            "streamed_tokens": 1,
            "delta_tokens": 1,
            "answer_tokens": 1,
        },
    }
    assert payloads[0]["hipengine"]["timing"]["backend_prefill_ms"] == 4.0
    assert payloads[0]["hipengine"]["timing"]["backend_decode_ms"] == 2.0
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["hipengine"]["timing"] == {"prefill_ms": 4.0, "decode_ms": 2.0}
    assert done["hipengine"]["timing"]["backend_prefill_ms"] == 4.0
    assert done["hipengine"]["timing"]["backend_decode_ms"] == 2.0


def test_streaming_completion_prefers_backend_chunk_finish_details() -> None:
    fake = FakeLLM(
        outputs=["should-not-buffer"],
        stream_chunks=[
            GenerationStreamChunk(
                "alpha",
                finish_details=FinishDetails(
                    reason="length",
                    length_limit=4,
                    sampler_mode="host_logits_sample",
                ),
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=5,
                    generated_tokens=4,
                    phase="answer",
                    sampler_mode="host_logits_sample",
                ),
            )
        ],
    )
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "hello",
            "max_tokens": 1,
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    done = next(payload for payload in payloads if payload.get("choices") and payload["choices"][0]["finish_reason"])
    expected_finish = {
        "reason": "length",
        "length_limit": 4,
        "cache_action": "append_none",
        "sampler_mode": "host_logits_sample",
    }
    assert done["choices"][0]["finish_reason"] == "length"
    assert done["choices"][0]["finish_details"] == expected_finish
    assert done["choices"][0]["hipengine"]["finish_details"] == expected_finish
    assert done["choices"][0]["hipengine"]["decode_state"] == {
        "row_index": 0,
        "step_index": 4,
        "prompt_tokens": 5,
        "generated_tokens": 4,
        "phase": "answer",
        "continuation_eligible": False,
        "sampler_mode": "host_logits_sample",
    }


def test_metrics_prefix_cache_and_generation_batch_cli_env_defaults(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_GENERATION_BATCH_WINDOW_MS", raising=False)
    monkeypatch.delenv("HIPENGINE_DEBUG", raising=False)
    monkeypatch.delenv("HIPENGINE_CHAT_DEFAULT_MAX_TOKENS", raising=False)
    monkeypatch.delenv("HIPENGINE_STARTUP_CHAT_SMOKE", raising=False)
    monkeypatch.delenv("HIPENGINE_STARTUP_SCRATCH_PROBE", raising=False)
    monkeypatch.delenv("HIPENGINE_STARTUP_MIN_FREE_MIB", raising=False)
    monkeypatch.delenv("HIPENGINE_REQUEST_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("HIPENGINE_MAX_QUEUED_REQUESTS", raising=False)
    monkeypatch.delenv("HIPENGINE_MAX_ACTIVE_REQUESTS", raising=False)
    monkeypatch.delenv("HIPENGINE_MAX_CHAT_SESSIONS", raising=False)
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
    assert default_args.max_active_requests is None
    assert default_args.max_chat_sessions is None
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
    monkeypatch.setenv("HIPENGINE_MAX_ACTIVE_REQUESTS", "6")
    monkeypatch.setenv("HIPENGINE_MAX_CHAT_SESSIONS", "5")
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
    assert env_args.max_active_requests == 6
    assert env_args.max_chat_sessions == 5
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
            "--max-active-requests",
            "4",
            "--max-chat-sessions",
            "2",
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
    assert cli_args.max_active_requests == 4
    assert cli_args.max_chat_sessions == 2
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


@pytest.mark.parametrize(
    ("payload", "output", "expected_finish_details"),
    [
        (
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "secret no replay tool task"}],
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
                "max_tokens": 16,
            },
            "ordinary answer",
            _stateless_finish_details("tool_required_not_satisfied"),
        ),
        (
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "secret no replay structured task"}],
                "response_format": {"type": "json_object"},
                "max_tokens": 16,
            },
            "not json",
            _stateless_finish_details("schema_violation"),
        ),
    ],
)
def test_replay_artifacts_are_default_off_for_agentic_result_failures(
    tmp_path,
    payload: dict[str, Any],
    output: str,
    expected_finish_details: dict[str, Any],
) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(model="fake-path", served_model_name="fake-model", eager_load=False),
        llm=FakeLLM(outputs=[output]),
    )
    client = TestClient(app)

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_details"] == expected_finish_details
    assert choice["message"] == {"role": "assistant", "content": ""}
    assert not replay_dir.exists()


def _load_single_replay_artifact(replay_dir: Path) -> tuple[dict[str, Any], str]:
    artifacts = list(replay_dir.glob("*.json"))
    assert len(artifacts) == 1
    artifact = json.loads(artifacts[0].read_text(encoding="utf-8"))
    serialized = json.dumps(artifact, sort_keys=True, allow_nan=False)
    return artifact, serialized


def test_replay_artifact_captures_streaming_completion_deadline_error(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=BackendDeadlineFakeLLM(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "secret streaming deadline prompt",
            "max_tokens": 4,
            "stream": True,
            "timeout_ms": 5000,
        },
    )

    assert response.status_code == 200
    payload = next(item for item in _sse_payloads(response.text) if item.get("error"))
    assert payload["error"]["code"] == "deadline_exceeded"
    assert payload["choices"][0]["finish_details"] == {
        "reason": "deadline_exceeded",
        "deadline_exceeded": True,
    }

    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["schema"] == "hipengine.replay.v1"
    assert artifact["request"]["path"] == "/v1/completions"
    assert artifact["request"]["prompt_hashes"] == [
        {
            "path": "$.prompt",
            "sha256": artifact["request"]["json"]["prompt"]["sha256"],
            "length": len("secret streaming deadline prompt"),
        }
    ]
    assert artifact["error"]["type"] == "timeout_error"
    assert artifact["error"]["code"] == "deadline_exceeded"
    assert artifact["error"]["param"] == "timeout_ms"
    assert artifact["error"]["hipengine"] == {
        "code": "deadline_exceeded",
        "status_code": 408,
        "retryable": True,
    }
    assert artifact["finish_details"] == {"reason": "deadline_exceeded", "deadline_exceeded": True}
    assert "secret streaming deadline prompt" not in serialized


def test_replay_artifact_captures_streaming_chat_cancelled_error(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=BackendCancelledFakeLLM(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "secret streaming cancel task"}],
            "max_tokens": 4,
            "stream": True,
        },
    )

    assert response.status_code == 200
    payload = next(item for item in _sse_payloads(response.text) if item.get("error"))
    assert payload["error"]["code"] == "cancelled"
    assert payload["choices"][0]["finish_details"] == {
        "reason": "cancelled",
        "cancelled": True,
    }

    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["schema"] == "hipengine.replay.v1"
    assert artifact["request"]["path"] == "/v1/chat/completions"
    assert artifact["request"]["prompt_hashes"] == [
        {
            "path": "$.messages[0].content",
            "sha256": artifact["request"]["json"]["messages"][0]["content"]["sha256"],
            "length": len("secret streaming cancel task"),
        }
    ]
    assert artifact["error"]["type"] == "cancelled_error"
    assert artifact["error"]["code"] == "cancelled"
    assert artifact["error"]["param"] is None
    assert artifact["error"]["hipengine"] == {
        "code": "cancelled",
        "status_code": 499,
        "retryable": True,
    }
    assert artifact["finish_details"] == {"reason": "cancelled", "cancelled": True}
    assert "secret streaming cancel task" not in serialized


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
    artifact, serialized = _load_single_replay_artifact(replay_dir)
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
    assert artifact["sampling"]["response_format"]["type"]["redacted"] == "sha256"
    assert artifact["sampling"]["response_format"]["type"]["length"] == len("json_object")
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
    assert artifact["capabilities"]["features"]["structured_outputs"][
        "result_validation_failure_reasons"
    ] == ["schema_violation"]
    assert artifact["capabilities"]["features"]["tools"]["result_validation_failure_reasons"] == [
        "invalid_tool_call",
        "tool_required_not_satisfied",
        "schema_violation",
    ]
    assert artifact["capabilities"]["features"]["tools"]["no_tool_start_suppression"] is True
    assert artifact["capabilities"]["features"]["tools"]["required_tool_start_forcing"] is True
    assert artifact["capabilities"]["features"]["tools"]["required_tool_start_forcing_scope"] == (
        "initial_or_after_tokenized_thinking_close"
    )
    assert artifact["capabilities"]["features"]["tools"]["specific_tool_name_prefix_forcing"] is True
    assert artifact["capabilities"]["features"]["tools"]["tool_call_close_repair"] is True
    assert artifact["capabilities"]["features"]["tools"]["compatibility_parser_repairs"] == [
        "duplicated_tool_call_start"
    ]
    assert (
        artifact["capabilities"]["features"]["tools"]["malformed_json_compatibility"]
        == "invalid_tool_call_when_tools_enabled"
    )
    assert artifact["capabilities"]["features"]["tools"]["strict_malformed_blocks_rejected"] is True
    assert artifact["capabilities"]["features"]["tools"]["declared_tool_name_validation"] is True
    assert artifact["capabilities"]["features"]["tools"]["transcript_validation"] == {
        "role_specific_fields": True,
        "assistant_tool_call_ids_unique": True,
        "tool_results_reference_prior_call_ids": True,
        "tool_results_must_resolve_pending_calls_before_non_tool_messages": True,
        "allows_pending_tool_calls_at_transcript_end": True,
        "applies_to_session_snapshots": True,
    }
    assert artifact["capabilities"]["features"]["tools"]["parallel_tool_calls_requires_opt_in"] is True
    assert artifact["capabilities"]["features"]["tools"]["streaming_argument_chunks"] is True
    assert artifact["capabilities"]["features"]["tools"]["streaming_argument_chunk_chars"] == 128
    assert artifact["capabilities"]["features"]["reasoning_controls"]["token_budget_enforced"] is True
    assert artifact["capabilities"]["features"]["reasoning_controls"]["hard_close_token_forcing"] is True
    assert artifact["capabilities"]["features"]["reasoning_controls"]["soft_close_bias"] is True
    assert artifact["capabilities"]["features"]["reasoning_controls"]["eos_suppression"] is True
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
            "eos_token_id",
            "ignore_eos",
            "stop_token_ids",
            "stop_token_sequences",
            "forced_tokens_pending",
            "post_thinking_forced_tokens_pending",
            "force_sequence_completion_token_sequences",
            "json_object_close_forcing",
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
            "eos_token_id": "eos_token_id set",
            "ignore_eos": "ignore_eos=true",
            "stop_token_ids": "one or more token stop ids",
            "stop_token_sequences": "one or more multi-token stop sequences",
            "forced_tokens_pending": "one or more forced tokens pending",
            "post_thinking_forced_tokens_pending": "one or more post-thinking forced tokens pending",
            "force_sequence_completion_token_sequences": "one or more token sequence completion repairs",
            "json_object_close_forcing": "JSON object close forcing active",
            "thinking_budget": "thinking budget soft-close, EOS suppression, or hard-close control",
            "logprobs": "logprobs requested",
            "top_logprobs": "top_logprobs > 0",
        },
        "processed_target_verification": False,
    }
    assert artifact["capabilities"]["sampling"]["speculative_mtp"]["incompatible_fields"] == list(
        SPECULATIVE_MTP_INCOMPATIBLE_FIELDS
    )
    assert artifact["capabilities"]["sampling"]["speculative_mtp"]["incompatible_conditions"] == dict(
        SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS
    )
    assert artifact["capabilities"]["sessions"] == {
        "resident_context": True,
        "commit_policy": _session_commit_policy_capability(),
        "continuations": _continuation_capability(),
        "metadata": _session_metadata_capability(),
    }
    assert artifact["capabilities"]["parallelism"] == _parallelism_capability()
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
            "chat_template_kwargs": {"thinking_budget": "low"},
            "thinking": {"budget_tokens": 32},
            "reasoning": {
                "allow_unbounded": True,
                "max_tokens": 12,
                "min_answer_tokens": 4,
                "hard_close_sequence": "closing</think>\n",
            },
            "hard_close_message": "closing",
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "top_logprobs": 1,
        },
    )

    assert response.status_code == 400
    artifact, serialized = _load_single_replay_artifact(replay_dir)

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
    assert artifact["sampling"]["chat_template_kwargs"]["thinking_budget"]["redacted"] == "sha256"
    assert artifact["sampling"]["chat_template_kwargs"]["thinking_budget"]["length"] == len("low")
    assert artifact["sampling"]["thinking"] == {"budget_tokens": 32}
    assert artifact["sampling"]["reasoning"]["allow_unbounded"] is True
    assert artifact["sampling"]["reasoning"]["max_tokens"] == 12
    assert artifact["sampling"]["reasoning"]["min_answer_tokens"] == 4
    assert artifact["sampling"]["reasoning"]["hard_close_sequence"]["redacted"] == "sha256"
    assert artifact["sampling"]["reasoning"]["hard_close_sequence"]["length"] == len("closing</think>\n")
    assert artifact["sampling"]["hard_close_message"]["redacted"] == "sha256"
    assert artifact["sampling"]["hard_close_message"]["length"] == len("closing")
    assert artifact["sampling"]["tool_choice"]["redacted"] == "sha256"
    assert artifact["sampling"]["tool_choice"]["length"] == len("auto")
    assert artifact["sampling"]["parallel_tool_calls"] is False
    assert artifact["error"]["param"] == "top_logprobs"
    assert "secret chat prompt" not in serialized
    assert "closing</think>" not in serialized


def test_replay_artifact_captures_completion_structured_result_validation_failure(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(outputs=["not json"]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "secret structured prompt",
            "response_format": {"type": "json_object"},
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert choice["text"] == ""

    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["schema"] == "hipengine.replay.v1"
    assert artifact["request"]["path"] == "/v1/completions"
    assert artifact["request"]["json"]["prompt"]["redacted"] == "sha256"
    assert artifact["request"]["prompt_hashes"] == [
        {
            "path": "$.prompt",
            "sha256": artifact["request"]["json"]["prompt"]["sha256"],
            "length": len("secret structured prompt"),
        }
    ]
    assert artifact["sampling"]["response_format"]["type"]["redacted"] == "sha256"
    assert artifact["sampling"]["response_format"]["type"]["length"] == len("json_object")
    assert artifact["finish_details"] == _stateless_finish_details("schema_violation")
    assert artifact["error"] is None
    assert artifact["result"] == {
        "type": "agentic_result_validation",
        "finish_details": _stateless_finish_details("schema_violation"),
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "finish_details": _stateless_finish_details("schema_violation"),
            }
        ],
    }
    assert "secret structured prompt" not in serialized
    assert "not json" not in serialized


@pytest.mark.parametrize(
    ("endpoint", "prompt_text", "request_extra", "generated", "sampling_assertion"),
    [
        (
            "/v1/completions",
            "secret guided json prompt",
            {"guided_json": True},
            "[true]",
            ("guided_json", True),
        ),
        (
            "/v1/chat/completions",
            "secret guided regex task",
            {"guided_regex": r"[A-Z]{2}-\d{2}"},
            "AB",
            ("guided_regex", {"length": len(r"[A-Z]{2}-\d{2}")}),
        ),
        (
            "/v1/chat/completions",
            "secret guided choice task",
            {"guided_choice": ["yes", "no"]},
            "maybe",
            ("guided_choice", [{"length": 3}, {"length": 2}]),
        ),
    ],
)
def test_replay_artifact_captures_guided_result_validation_failure(
    tmp_path,
    endpoint: str,
    prompt_text: str,
    request_extra: dict[str, Any],
    generated: str,
    sampling_assertion: tuple[str, Any],
) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(outputs=[generated]),
    )
    client = TestClient(app)
    is_chat = endpoint.endswith("/chat/completions")
    if not is_chat:
        payload: dict[str, Any] = {"model": "fake-model", "prompt": prompt_text, "max_tokens": 16}
    else:
        payload = {
            "model": "fake-model",
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": 16,
        }
    payload.update(request_extra)

    response = client.post(endpoint, json=payload)

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("schema_violation")
    if is_chat:
        assert choice["message"] == {"role": "assistant", "content": ""}
        prompt_path = "$.messages[0].content"
    else:
        assert choice["text"] == ""
        prompt_path = "$.prompt"

    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["schema"] == "hipengine.replay.v1"
    assert artifact["request"]["path"] == endpoint
    redacted_prompt = (
        artifact["request"]["json"]["messages"][0]["content"]
        if is_chat
        else artifact["request"]["json"]["prompt"]
    )
    assert artifact["request"]["prompt_hashes"] == [
        {
            "path": prompt_path,
            "sha256": redacted_prompt["sha256"],
            "length": len(prompt_text),
        }
    ]
    field, expected_sampling = sampling_assertion
    actual_sampling = artifact["sampling"][field]
    if isinstance(expected_sampling, list):
        assert [item["length"] for item in actual_sampling] == [
            item["length"] for item in expected_sampling
        ]
        assert all(item["redacted"] == "sha256" for item in actual_sampling)
    elif isinstance(expected_sampling, dict):
        assert actual_sampling["redacted"] == "sha256"
        assert actual_sampling["length"] == expected_sampling["length"]
    else:
        assert actual_sampling is expected_sampling
    assert artifact["finish_details"] == _stateless_finish_details("schema_violation")
    assert artifact["error"] is None
    assert artifact["result"] == {
        "type": "agentic_result_validation",
        "finish_details": _stateless_finish_details("schema_violation"),
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "finish_details": _stateless_finish_details("schema_violation"),
            }
        ],
    }
    assert prompt_text not in serialized
    assert generated not in serialized


def test_replay_artifact_captures_guided_patch_result_validation_failure(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    invalid_patch = "Here is the patch:\n" + _unified_diff_text()
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(outputs=[invalid_patch]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "secret patch task"}],
            "guided_patch": True,
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("schema_violation")
    assert choice["message"] == {"role": "assistant", "content": ""}

    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["schema"] == "hipengine.replay.v1"
    assert artifact["request"]["path"] == "/v1/chat/completions"
    assert artifact["request"]["json"]["messages"][0]["content"]["redacted"] == "sha256"
    assert artifact["request"]["prompt_hashes"] == [
        {
            "path": "$.messages[0].content",
            "sha256": artifact["request"]["json"]["messages"][0]["content"]["sha256"],
            "length": len("secret patch task"),
        }
    ]
    assert artifact["sampling"]["guided_patch"] is True
    assert artifact["finish_details"] == _stateless_finish_details("schema_violation")
    assert artifact["error"] is None
    assert artifact["result"] == {
        "type": "agentic_result_validation",
        "finish_details": _stateless_finish_details("schema_violation"),
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "finish_details": _stateless_finish_details("schema_violation"),
            }
        ],
    }
    assert "secret patch task" not in serialized
    assert "Here is the patch" not in serialized
    assert "diff --git" not in serialized


def test_replay_artifact_captures_agentic_result_validation_failure(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    prior_tool_arguments = '{"path":"secret previous path"}'
    prior_tool_result = "secret previous tool result"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(outputs=["ordinary answer"]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [
                {"role": "user", "content": "secret tool task"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_secret_prev",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": prior_tool_arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_secret_prev",
                    "content": prior_tool_result,
                },
            ],
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
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["finish_details"] == _stateless_finish_details("tool_required_not_satisfied")
    assert choice["message"] == {"role": "assistant", "content": ""}

    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["schema"] == "hipengine.replay.v1"
    assert artifact["request"]["path"] == "/v1/chat/completions"
    assert artifact["request"]["json"]["messages"][0]["content"]["redacted"] == "sha256"
    assert (
        artifact["request"]["json"]["messages"][1]["tool_calls"][0]["function"]["arguments"][
            "redacted"
        ]
        == "sha256"
    )
    assert artifact["request"]["json"]["messages"][2]["content"]["redacted"] == "sha256"
    assert artifact["request"]["prompt_hashes"] == [
        {
            "path": "$.messages[0].content",
            "sha256": artifact["request"]["json"]["messages"][0]["content"]["sha256"],
            "length": len("secret tool task"),
        },
        {
            "path": "$.messages[1].tool_calls[0].function.arguments",
            "sha256": artifact["request"]["json"]["messages"][1]["tool_calls"][0]["function"][
                "arguments"
            ]["sha256"],
            "length": len(prior_tool_arguments),
        },
        {
            "path": "$.messages[2].content",
            "sha256": artifact["request"]["json"]["messages"][2]["content"]["sha256"],
            "length": len(prior_tool_result),
        },
    ]
    assert artifact["finish_details"] == _stateless_finish_details("tool_required_not_satisfied")
    assert artifact["error"] is None
    assert artifact["result"] == {
        "type": "agentic_result_validation",
        "finish_details": _stateless_finish_details("tool_required_not_satisfied"),
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "finish_details": _stateless_finish_details("tool_required_not_satisfied"),
            }
        ],
    }
    assert artifact["token_counts"]["available"] is True
    assert artifact["token_counts"]["source"] == "chat_prompt"
    assert "secret tool task" not in serialized
    assert "secret previous path" not in serialized
    assert prior_tool_arguments not in serialized
    assert prior_tool_result not in serialized
    assert "ordinary answer" not in serialized


def test_replay_artifact_captures_streaming_structured_result_validation_failure(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(stream_chunks=["not json"]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "secret streaming structured task"}],
            "response_format": {"type": "json_object"},
            "stream": True,
            "stream_options": {"include_hipengine": True},
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_reason"] == "stop"
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"]["finish_details"] == _stateless_finish_details("schema_violation")

    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["request"]["path"] == "/v1/chat/completions"
    assert artifact["sampling"]["response_format"]["type"]["redacted"] == "sha256"
    assert artifact["sampling"]["response_format"]["type"]["length"] == len("json_object")
    assert artifact["finish_details"] == _stateless_finish_details("schema_violation")
    assert artifact["error"] is None
    assert artifact["result"]["choices"] == [
        {
            "index": 0,
            "finish_reason": "stop",
            "finish_details": _stateless_finish_details("schema_violation"),
        }
    ]
    assert "secret streaming structured task" not in serialized
    assert "not json" not in serialized


def test_replay_artifact_captures_streaming_guided_diff_result_validation_failure(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(outputs=["not a diff"], stream_chunks=["not a diff"]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "fake-model",
            "prompt": "secret streaming patch prompt",
            "guided_diff": True,
            "stream": True,
            "stream_options": {"include_hipengine": True},
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_reason"] == "stop"
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("schema_violation")
    assert done["choices"][0]["hipengine"]["finish_details"] == _stateless_finish_details("schema_violation")

    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["request"]["path"] == "/v1/completions"
    assert artifact["sampling"]["guided_diff"] is True
    assert artifact["finish_details"] == _stateless_finish_details("schema_violation")
    assert artifact["error"] is None
    assert artifact["result"]["choices"] == [
        {
            "index": 0,
            "finish_reason": "stop",
            "finish_details": _stateless_finish_details("schema_violation"),
        }
    ]
    assert "secret streaming patch prompt" not in serialized
    assert "not a diff" not in serialized


def test_replay_artifact_captures_streaming_agentic_result_validation_failure(tmp_path) -> None:
    replay_dir = tmp_path / "replay"
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            replay_dir=str(replay_dir),
            replay_redaction="hash",
        ),
        llm=FakeLLM(stream_chunks=["ordinary stream answer"]),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "secret streaming tool task"}],
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
            "stream": True,
            "stream_options": {"include_hipengine": True},
            "max_tokens": 16,
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    done = next(payload for payload in payloads if payload["choices"][0]["finish_reason"])
    assert done["choices"][0]["finish_reason"] == "stop"
    assert done["choices"][0]["finish_details"] == _stateless_finish_details("tool_required_not_satisfied")
    assert done["choices"][0]["hipengine"]["finish_details"] == _stateless_finish_details(
        "tool_required_not_satisfied"
    )

    artifact, serialized = _load_single_replay_artifact(replay_dir)
    assert artifact["request"]["path"] == "/v1/chat/completions"
    assert artifact["finish_details"] == _stateless_finish_details("tool_required_not_satisfied")
    assert artifact["error"] is None
    assert artifact["result"]["choices"] == [
        {
            "index": 0,
            "finish_reason": "stop",
            "finish_details": _stateless_finish_details("tool_required_not_satisfied"),
        }
    ]
    assert "secret streaming tool task" not in serialized
    assert "ordinary stream answer" not in serialized


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
            max_active_requests=2,
            max_chat_sessions=4,
        ),
        llm=fake,
    )
    client = TestClient(app)

    before = client.get("/metrics")
    assert before.status_code == 200
    assert _metric_value(before.text, "hipengine_requests_total") == 0
    assert _metric_value(before.text, "hipengine_generation_queue_depth") == 0
    assert _metric_value(before.text, "hipengine_request_cancelled_total") == 0
    assert _metric_value(before.text, "hipengine_generation_queue_max_depth") == 3
    assert _metric_value(before.text, "hipengine_generation_worker_active") == 0
    assert _metric_value(before.text, "hipengine_generation_requests_active") == 0
    assert _metric_value(before.text, "hipengine_generation_requests_max_active") == 2
    assert (
        'hipengine_generation_scheduler_fairness_policy_info{policy="fifo_compatible_sampling_key",'
        'compatible_sampling_coalescing="true",continuous_decode="false",preemptive_fairness="false"} 1'
        in before.text
    )
    assert _metric_value(before.text, "hipengine_chat_sessions_active") == 0
    assert _metric_value(before.text, "hipengine_chat_sessions_pending") == 0
    assert _metric_value(before.text, "hipengine_chat_sessions_max_active") == 4

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
    assert _metric_value(metrics.text, "hipengine_request_cancelled_total") == 0
    assert _metric_value(metrics.text, "hipengine_generation_queue_depth") == 0
    assert _metric_value(metrics.text, "hipengine_generation_queue_max_depth") == 3
    assert _metric_value(metrics.text, "hipengine_generation_worker_active") == 0
    assert _metric_value(metrics.text, "hipengine_generation_requests_active") == 0
    assert _metric_value(metrics.text, "hipengine_generation_requests_max_active") == 2
    assert _metric_value(metrics.text, "hipengine_chat_sessions_active") == 0
    assert _metric_value(metrics.text, "hipengine_chat_sessions_pending") == 0
    assert _metric_value(metrics.text, "hipengine_chat_sessions_max_active") == 4
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


def test_metrics_endpoint_counts_cancelled_requests() -> None:
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
            metrics="prometheus",
        ),
        llm=BackendCancelledFakeLLM(),
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "fake-model", "prompt": "cancel", "max_tokens": 1},
    )
    metrics = client.get("/metrics")

    assert response.status_code == 499
    assert _metric_value(metrics.text, "hipengine_requests_total") == 1
    assert _metric_value(metrics.text, "hipengine_request_failed_total") == 1
    assert _metric_value(metrics.text, "hipengine_request_cancelled_total") == 1
    assert _metric_value(metrics.text, "hipengine_request_rejected_total") == 0


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
        "routing": {
            **_routing_metadata(),
            "matched": True,
            "reason": "context_overflow",
        },
    }
    assert error["fit_context"] == {
        "prompt_tokens": 4,
        "max_context_tokens": 5,
        "effective_max_tokens": 2,
        "max_allowed_max_tokens": 0,
        "recommended_max_tokens": 0,
        "required_context_tokens": 7,
        "overflow_tokens": 2,
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


def test_streaming_completion_context_overflow_preserves_error_diagnostics() -> None:
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
        json={
            "model": "fake-model",
            "prompt": "one two three four",
            "max_tokens": 2,
            "stream": True,
            "stream_options": {"include_hipengine": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["choices"][0]["finish_reason"] == "error"
    assert payload["error"]["code"] == "context_length_exceeded"
    assert payload["error"]["hipengine"] == {
        "code": "context_overflow",
        "status_code": 400,
        "legacy_code": "context_length_exceeded",
        "retryable": False,
        "routing": {
            **_routing_metadata(),
            "matched": True,
            "reason": "context_overflow",
        },
    }
    assert payload["error"]["fit_context"] == {
        "prompt_tokens": 4,
        "max_context_tokens": 5,
        "effective_max_tokens": 2,
        "max_allowed_max_tokens": 0,
        "recommended_max_tokens": 0,
        "required_context_tokens": 7,
        "overflow_tokens": 2,
        "fits": False,
        "clear_policy": "reject",
        "would_truncate": False,
        "would_drop": [],
    }
    assert payload["hipengine"]["event"] == "error"
    assert payload["hipengine"]["routing"] == _routing_metadata()
    assert "data: [DONE]" in response.text
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
    fake = FakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
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
        "routing": {
            "requested_model": "other",
            "served_model": None,
            "configured_model": "fake-model",
            "fallback_used": False,
            "policy": "single_model_exact",
            "loaded_model_count": 1,
            "multiple_models": False,
            "matched": False,
            "reason": "model_unavailable",
        },
    }

    chat_wrong_model = client.post(
        "/v1/chat/completions",
        json={"model": "other", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert chat_wrong_model.status_code == 404
    assert chat_wrong_model.json()["error"]["code"] == "model_not_found"
    assert chat_wrong_model.json()["error"]["hipengine"]["routing"] == wrong_model.json()["error"]["hipengine"]["routing"]
    assert fake.calls == []

    schema_violation = client.post(
        "/v1/completions",
        json={"model": "fake-model", "max_tokens": 1},
    )
    assert schema_violation.status_code == 400
    assert schema_violation.json()["error"]["code"] == "invalid_request"
    assert schema_violation.json()["error"]["param"] == "prompt"
    assert schema_violation.json()["error"]["hipengine"] == {
        "code": "schema_violation",
        "status_code": 400,
        "legacy_code": "invalid_request",
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
            "/v1/completions",
            {
                "model": "fake-model",
                "prompt": "hello",
                "session": {"id": "session_123"},
            },
            "session.id",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "session": {"id": "session_123"},
            },
            "stream",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "n": 2,
                "session": {"id": "session_123"},
            },
            "n",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "continuation_id": "gen_missing",
                "session": {"id": "session_123"},
            },
            "messages",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "grammar": {"type": "json"},
            },
            "grammar",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "guided_grammar": "root ::= 'ok'",
            },
            "guided_grammar",
        ),
        (
            "/v1/chat/completions",
            {
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "guided_decoding_backend": "outlines",
            },
            "guided_decoding_backend",
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
    if param in {"grammar", "guided_grammar", "guided_decoding_backend"}:
        assert response.json()["error"]["hipengine"] == {
            "code": "unsupported_parameter",
            "status_code": 400,
            "retryable": False,
            "routing": {
                **_routing_metadata(),
                "matched": True,
                "reason": "unsupported_grammar",
                "unsupported_field": param,
                "unsupported_capability": "grammar",
            },
        }
    assert fake.calls == []


def test_capabilities_advertised_unsupported_fields_are_rejected_before_generation() -> None:
    fake = FakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)
    capabilities = client.get("/v1/hipengine/capabilities").json()

    unsupported_fields = capabilities["unsupported_fields"]
    assert unsupported_fields == capabilities["features"]["grammars"]["unsupported_fields"]

    field_values = {
        "grammar": {"type": "json"},
        "guided_grammar": "root ::= 'ok'",
        "guided_decoding_backend": "outlines",
    }
    assert set(unsupported_fields) == set(field_values)

    for endpoint in ("/v1/completions", "/v1/chat/completions"):
        for field in unsupported_fields:
            payload = (
                {"model": "fake-model", "prompt": "hello"}
                if endpoint == "/v1/completions"
                else {"model": "fake-model", "messages": [{"role": "user", "content": "hello"}]}
            )
            payload[field] = field_values[field]

            response = client.post(endpoint, json=payload)

            assert response.status_code == 400
            assert response.json()["error"]["code"] == "unsupported_parameter"
            assert response.json()["error"]["param"] == field
            assert response.json()["error"]["hipengine"] == {
                "code": "unsupported_parameter",
                "status_code": 400,
                "retryable": False,
                "routing": {
                    **_routing_metadata(),
                    "matched": True,
                    "reason": "unsupported_grammar",
                    "unsupported_field": field,
                    "unsupported_capability": "grammar",
                },
            }

    assert fake.calls == []


@pytest.mark.parametrize(
    ("payload", "param", "code"),
    [
        (
            {
                "guided_json": 7,
            },
            "guided_json",
            "invalid_request",
        ),
        (
            {
                "guided_json": "{not json}",
            },
            "guided_json",
            "invalid_request",
        ),
        (
            {
                "guided_json": {"$ref": "#/$defs/result"},
            },
            "guided_json.$ref",
            "invalid_request",
        ),
        (
            {
                "guided_json": {"schema": {"$ref": "#/$defs/result"}},
            },
            "guided_json.schema.$ref",
            "invalid_request",
        ),
        (
            {
                "guided_json": True,
                "response_format": {"type": "json_object"},
            },
            "guided_json",
            "invalid_request",
        ),
        (
            {
                "guided_json": True,
                "guided_regex": "[a-z]+",
            },
            "guided_json",
            "invalid_request",
        ),
        (
            {
                "guided_json": True,
                "guided_choice": ["yes", "no"],
            },
            "guided_json",
            "invalid_request",
        ),
        (
            {
                "guided_json": True,
                "guided_patch": True,
            },
            "guided_json",
            "invalid_request",
        ),
        (
            {
                "guided_regex": "",
            },
            "guided_regex",
            "invalid_request",
        ),
        (
            {
                "guided_regex": ["[a-z]+"],
            },
            "guided_regex",
            "invalid_request",
        ),
        (
            {
                "guided_regex": "[",
            },
            "guided_regex",
            "invalid_request",
        ),
        (
            {
                "guided_regex": "[a-z]+",
                "response_format": {"type": "json_object"},
            },
            "guided_regex",
            "invalid_request",
        ),
        (
            {
                "guided_regex": "[a-z]+",
                "guided_choice": ["yes", "no"],
            },
            "guided_regex",
            "invalid_request",
        ),
        (
            {
                "guided_regex": "[a-z]+",
                "guided_patch": True,
            },
            "guided_regex",
            "invalid_request",
        ),
        (
            {
                "guided_choice": [],
            },
            "guided_choice",
            "invalid_request",
        ),
        (
            {
                "guided_choice": ["yes", 1],
            },
            "guided_choice[1]",
            "invalid_request",
        ),
        (
            {
                "guided_choice": ["yes", "no"],
                "response_format": {"type": "json_object"},
            },
            "guided_choice",
            "invalid_request",
        ),
        (
            {
                "guided_choice": ["yes", "no"],
                "guided_patch": True,
            },
            "guided_choice",
            "invalid_request",
        ),
        (
            {
                "guided_patch": {"type": "regex"},
            },
            "guided_patch.type",
            "unsupported_parameter",
        ),
        (
            {
                "guided_patch": True,
                "guided_diff": True,
            },
            "guided_patch",
            "invalid_request",
        ),
        (
            {
                "guided_patch": True,
                "response_format": {"type": "json_object"},
            },
            "guided_patch",
            "invalid_request",
        ),
        (
            {
                "guided_patch": {"fenced": "sometimes"},
            },
            "guided_patch.fenced",
            "invalid_request",
        ),
        (
            {
                "guided_patch": {"fenced": None},
            },
            "guided_patch.fenced",
            "invalid_request",
        ),
    ],
)
def test_guided_output_request_validation_fails_before_generation(payload, param, code) -> None:
    fake = FakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "fake-model",
            "messages": [{"role": "user", "content": "hello"}],
            **payload,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["param"] == param
    assert fake.calls == []


def test_server_rejects_unknown_continuation_id_without_generation() -> None:
    fake = FakeLLM()
    app = create_app(ServerConfig(model="fake-path", served_model_name="fake-model"), llm=fake)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "fake-model", "continuation_id": "gen_123"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_continuation"
    assert response.json()["error"]["param"] == "continuation_id"
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
    error = response.json()["error"]
    assert error["code"] == "unsupported_content_type"
    assert error["hipengine"] == {
        "code": "unsupported_parameter",
        "status_code": 400,
        "legacy_code": "unsupported_content_type",
        "retryable": False,
    }


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
