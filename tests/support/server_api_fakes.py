"""Fake engine doubles shared by the OpenAI server integration tests.

Extracted from ``tests/test_integration_server_api.py`` by RF-12. These are
test doubles, not fixtures: tests construct them directly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import anyio
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from hipengine import ResidentPreparationError, SamplingParams
from hipengine.generation import (
    GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS,
    EngineCommandTimeout,
    EngineServiceClosed,
    FinishDetails,
    GenerationAdmissionRejected,
    GenerationCancellationToken,
    GenerationCancelled,
    GenerationDeadlineExceeded,
    GenerationExecutionFailed,
    GenerationOutput,
    GenerationStreamChunk,
    GenerationTelemetry,
    PreparedPromptInput,
    TokenLogprob,
)
from hipengine.generation.qwen4_exp_ar_batch import QWEN4_EXP_MAX_BATCH_ROWS
from hipengine.generation.sampling import (
    NATIVE_GPU_SAMPLER_UNSUPPORTED_CAPABILITIES,
    SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS,
    SPECULATIVE_MTP_INCOMPATIBLE_FIELDS,
)
from hipengine.runtime.memory_admission import (
    ADMISSION_REFUSAL_CODE,
    MemoryAdmissionRefused,
    MemoryConsumer,
    price_memory_admission,
)
from hipengine.server import ServerConfig, create_app, render_chat_prompt
from hipengine.server.__main__ import build_parser
from hipengine.speculative import (
    SpeculativeMTPStaticEligibility,
    SpeculativeMTPStaticState,
)
from hipengine.models.qwen35 import Qwen35GGUFModel
from hipengine.speculative.policy import DEFAULT_AUTO_DEPTH_POLICY
from hipengine.speculative.serving import SpeculativeMTPServingKey
from hipengine.server.api import (
    OpenAIHTTPError,
    StartupFailure,
    _AGENTIC_REPLAY_FAILURE_REASONS,
    _STRUCTURED_OUTPUT_RESULT_VALIDATION_FAILURE_REASONS,
    _TOOL_RESULT_VALIDATION_FAILURE_REASONS,
    _THINKING_CLOSE_MARKER,
    _await_with_request_control,
    _backend_scheduler_token_chunks,
    _coerce_generation_output,
    _execution_route_for_static_intent,
    _generation_route_for_request,
    _GenerationBatcher,
    _MTPCircuitBreaker,
    _QueuedBatchResult,
    _QueuedGeneration,
    _SPECULATIVE_MTP_AUTO_ROUTE,
    _SPECULATIVE_MTP_BATCH_ROUTE,
    _SPECULATIVE_MTP_DEFAULT_ROUTE,
    _SPECULATIVE_MTP_K0_ROUTE,
    _prepared_context_tokens,
    _request_completion_cap,
    _request_control,
    _realized_model_serving_plan,
    _resolve_realized_generation_route,
    _sampling_for_realized_generation_route,
    _sampling_key,
    _serving_plan_route_decision,
    _log_pretty_startup_summary,
    _startup_memory_summary,
    _mtp_accepted_rejected_counts,
    _mtp_response_summary,
)
from hipengine.server.chat_output import _parse_chat_tool_calls
from hipengine.server.metrics import _ServerMetrics
from hipengine.server.schemas import ChatCompletionRequest, CompletionRequest


class FakeLLM:
    def __init__(
        self,
        outputs: list[str] | None = None,
        stream_chunks: list[str] | None = None,
        token_map: dict[str, list[int]] | None = None,
        detailed_outputs: list[GenerationOutput] | None = None,
        speculative_mtp_serving: str | None = None,
        **_unused_kwargs: Any,
    ) -> None:
        self.speculative_mtp_serving = speculative_mtp_serving
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
        if self.token_map is None:
            raise NotImplementedError("fake tokenization is not configured")
        prompt = str(text)
        token_ids = tuple(self.token_map[prompt])
        self.tokenize_calls.append(prompt)
        return token_ids

    def detokenize(self, token_ids, *, skip_special: bool = False) -> str:
        return " ".join(f"T{int(token)}" for token in token_ids)


class PromptPreparingFakeLLM(FakeLLM):
    def __init__(self, *, stream: bool = False) -> None:
        details = [
            GenerationOutput(text="prepared reply", generated_token_ids=(901, 902))
        ]
        super().__init__(
            stream_chunks=["prepared reply"] if stream else None,
            detailed_outputs=None if stream else details,
        )
        self.count_token_calls: list[str] = []

    def tokenize(self, text: str) -> tuple[int, ...]:
        prompt = str(text)
        self.tokenize_calls.append(prompt)
        pieces = prompt.split()
        return tuple(range(100, 100 + max(1, len(pieces))))

    def count_tokens(self, text: str) -> int:
        prompt = str(text)
        self.count_token_calls.append(prompt)
        return max(1, len(prompt.split()))


class ResidentSessionPromptFakeLLM(PromptPreparingFakeLLM):
    supports_resident_session_kv = True


class SpeculativeMTPFakeLLM(FakeLLM):
    supports_speculative_mtp = True
    # The loaded engine resolves an omitted budget from model-plugin evidence.
    speculative_candidate_budget_requested: int | None = None
    speculative_candidate_budget: int | None = 3
    speculative_candidate_budget_source = "model_plugin_evidence"

    @property
    def speculative_candidate_budget_resolution(self) -> dict[str, Any]:
        return {
            "requested_candidate_budget": self.speculative_candidate_budget_requested,
            "candidate_budget": self.speculative_candidate_budget,
            "candidate_budget_source": self.speculative_candidate_budget_source,
        }

    def __init__(self, token_map: dict[str, list[int]] | None = None) -> None:
        super().__init__(token_map=token_map)
        self.mtp_calls: list[tuple[tuple[str, ...], SamplingParams]] = []

    def stream_speculative_mtp_detailed(
        self,
        prompts,
        sampling_params: SamplingParams,
    ):
        prompt_tuple = (
            (str(prompts),)
            if isinstance(prompts, str)
            else tuple(str(prompt) for prompt in prompts)
        )
        self.mtp_calls.append((prompt_tuple, sampling_params))
        yield GenerationStreamChunk(
            text=f"mtp:{prompt_tuple[0]}",
            generated_token_ids=(901,),
            telemetry=GenerationTelemetry.from_decode_counts(
                prompt_tokens=1,
                generated_tokens=1,
                row_index=0,
                request_id="0",
                phase="answer",
                sampler_mode="greedy_fast",
                execution_path="speculative_mtp_server",
            ),
        )

    def generate_speculative_mtp_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
        prompt_tuple = tuple(str(prompt) for prompt in prompts)
        self.mtp_calls.append((prompt_tuple, sampling_params))
        self.last_batch_generation = {
            "path": "speculative_mtp_server",
            "batch_size": len(prompt_tuple),
            "group_widths": [len(prompt_tuple)],
            "speculative_mtp": {
                "target_verify_rows": len(prompt_tuple) * 3,
            },
            "scheduler_token_chunks": [
                {
                    "request_id": request_id,
                    "token_index": 0,
                    "token_id": 900 + request_id,
                    "finished": True,
                    "chunk": {
                        "text": f"mtp:{prompt}",
                        "telemetry": GenerationTelemetry.from_decode_counts(
                            prompt_tokens=1,
                            generated_tokens=1,
                            row_index=request_id,
                            request_id=str(request_id),
                            phase="answer",
                            sampler_mode="greedy_fast",
                            execution_path="speculative_mtp_server",
                        ).to_json_dict(),
                    },
                }
                for request_id, prompt in enumerate(prompt_tuple)
            ],
        }
        return [
            GenerationOutput(
                text=f"mtp:{prompt}",
                telemetry=GenerationTelemetry.from_decode_counts(
                    prompt_tokens=1,
                    generated_tokens=1,
                    row_index=index,
                    phase="done",
                    sampler_mode="greedy_fast",
                    execution_path="speculative_mtp_server",
                    timing={
                        "mtp_cycles_count": 1.0,
                        "mtp_generated_draft_tokens": 3.0,
                        "mtp_accepted_draft_tokens": 2.0,
                    },
                ),
            )
            for index, prompt in enumerate(prompt_tuple)
        ]


class LateResolvingMTPFakeLLM(SpeculativeMTPFakeLLM):
    """An engine whose MTP route exists only once ``prepare`` loads the model.

    Real engines behave this way: ``LLM.__init__`` builds no generator, and the
    model plugin's speculative evidence is read during ``prepare``. So the route
    is genuinely unknown between construction and preparation, which is the
    window where a startup line must not claim a resolved state.
    """

    def __init__(self, token_map: dict[str, list[int]] | None = None) -> None:
        super().__init__(token_map=token_map)
        self.supports_speculative_mtp = False
        self._text_generator = None

    def prepare(
        self,
        *,
        max_sequence_length: int | None = None,
        sampling_params: SamplingParams,
    ) -> int:
        selected = super().prepare(
            max_sequence_length=max_sequence_length,
            sampling_params=sampling_params,
        )
        self.supports_speculative_mtp = True
        self._text_generator = object()
        return selected


class AccountingSpeculativeMTPFakeLLM(SpeculativeMTPFakeLLM):
    """Streaming MTP fake that reports per-request execution accounting.

    Mirrors the resident GGUF MTP2 path: three depth-3 cycles emit nine tokens
    (six accepted drafts plus three target bonus tokens) out of twelve, and the
    response carries the same accounting in its streamed final metadata.
    """

    def stream_speculative_mtp_detailed(self, prompts, sampling_params):
        prompt_tuple = (
            (str(prompts),)
            if isinstance(prompts, str)
            else tuple(str(prompt) for prompt in prompts)
        )
        self.mtp_calls.append((prompt_tuple, sampling_params))
        yield GenerationStreamChunk(
            text=f"mtp:{prompt_tuple[0]}",
            generated_token_ids=tuple(range(901, 913)),
            finish_details={"reason": "length"},
            telemetry=GenerationTelemetry.from_decode_counts(
                prompt_tokens=1,
                generated_tokens=12,
                row_index=0,
                request_id="0",
                phase="answer",
                sampler_mode="greedy_fast",
                execution_path="gguf_specdec2_mtp2",
                # Resident MTP2 publishes these counters only in diagnostics;
                # the server must not need a scalar timing mirror.
                diagnostics={
                    "specdec2_mtp2": {
                        "requested_budget": 3,
                        "candidate_budget": 3,
                        "prompt_streaming": True,
                        "prompt_fallback_reason": None,
                        "cycles": 3,
                        "generated_draft_tokens": 9,
                        "accepted_draft_tokens": 6,
                        "selected_depth_histogram": {"3": 3},
                        "mtp_output_tokens": 9,
                        "ar_output_tokens_in_cycles": 0,
                        "first_fallback_position": None,
                        "ar_step_reason_counts": {},
                        "recoverable_failures": 0,
                        "failure_reason_counts": {},
                        "k0_catchups": 0,
                    }
                },
            ),
        )


class ArtifactScopedSpeculativeMTPFakeLLM(SpeculativeMTPFakeLLM):
    def __init__(self) -> None:
        super().__init__()
        self.plan_calls: list[dict[str, Any]] = []

    @staticmethod
    def _plan(**kwargs: Any) -> dict[str, Any]:
        admitted = bool(
            kwargs["realized_group_rows"] == 1
            and kwargs["sampling_mode"] == "greedy_fast"
            and kwargs["kv_storage"] in {"auto", "bf16"}
            and kwargs["memory_fit"]
        )
        reason = (
            "qualified_explicit_c1_b3"
            if admitted
            else (
                "sampling_mode_not_qualified"
                if kwargs["sampling_mode"] != "greedy_fast"
                else "physical_group_not_qualified"
            )
        )
        return {
            "schema_version": 1,
            "plan_fingerprint": "sha256:" + ("a" if admitted else "b") * 64,
            "key": {
                "realized_group_rows": kwargs["realized_group_rows"],
            },
            "admitted": admitted,
            "selected_route": "speculative_mtp" if admitted else "default",
            "selected_candidate_count": 3 if admitted else 0,
            "reason": reason,
            "strict_fallback_key": "gguf_target_ar",
            "evidence_key": "fake-qwen38-q4km-c1-b3",
            "evidence_artifacts": ["benchmarks/results/fake-qwen38-q4km-s0.json"],
            "automatic_eligible": False,
        }

    def resolve_speculative_mtp_serving_plan(self, **kwargs: Any):
        self.plan_calls.append(dict(kwargs))
        return self._plan(**kwargs)

    @property
    def speculative_mtp_serving_capability(self):
        return self._plan(
            realized_group_rows=1,
            sampling_mode="greedy_fast",
            kv_storage="bf16",
            memory_fit=True,
        )


class DeclaredCellSpeculativeMTPFakeLLM(SpeculativeMTPFakeLLM):
    """Resolves plans through the real Qwen3.8 dense capability declarations.

    The server layer only decides whether typed intent survives; whether a cell
    is executable is the model plugin's answer. Driving these tests through the
    real declaration keeps them honest about which cells the kernels implement.
    """

    backend = "hip_gfx1100"
    target_arch = "gfx1100"
    resident_capacity = 8

    def resolve_speculative_mtp_serving_plan(
        self,
        *,
        realized_group_rows: int,
        sampling_mode: str,
        kv_storage: str = "auto",
        memory_fit: bool = True,
        request_mode: str = "automatic",
    ):
        key = SpeculativeMTPServingKey(
            artifact_sha256=None,
            artifact_size_bytes=None,
            content_verified=False,
            backend=self.backend,
            target_arch=self.target_arch,
            weight_quant="gguf_q4_k_m",
            kv_storage="bf16" if str(kv_storage) == "auto" else str(kv_storage),
            kv_layout="uniform",
            realized_group_rows=int(realized_group_rows),
            resident_capacity=int(self.resident_capacity),
            candidate_budget=int(self.speculative_candidate_budget or 3),
            sampling_mode=str(sampling_mode),
            memory_fit=bool(memory_fit),
            kv_scale_dtype=None,
            kv_scale_granularity=None,
            artifact_execution_fingerprint=None,
        )
        return Qwen35GGUFModel().resolve_speculative_mtp_serving_plan(
            key=key,
            request_mode=str(request_mode),
        )


class DefaultMTPDeclaredCellFakeLLM(DeclaredCellSpeculativeMTPFakeLLM):
    """The same declaration-backed plans, with automatic MTP advertised.

    A real artifact answers `speculative_mtp_serving_capability` through the
    model plugin's declaration, which is automatic-eligible, so an `enabled`
    server routes an implicit request to the MTP batch route. Reproducing that
    answer is what makes the two intents eligible to coalesce, which is the
    grouping question these tests ask.
    """

    @property
    def speculative_mtp_serving_capability(self):
        return Qwen35GGUFModel().resolve_speculative_mtp_serving_plan(
            key=SpeculativeMTPServingKey(
                artifact_sha256=None,
                artifact_size_bytes=None,
                content_verified=False,
                backend=self.backend,
                target_arch=self.target_arch,
                weight_quant="gguf_q4_k_m",
                kv_storage="bf16",
                kv_layout="uniform",
                realized_group_rows=1,
                resident_capacity=int(self.resident_capacity),
                candidate_budget=int(self.speculative_candidate_budget or 3),
                sampling_mode="greedy_fast",
                memory_fit=True,
                kv_scale_dtype=None,
                kv_scale_granularity=None,
                artifact_execution_fingerprint=None,
            ),
            request_mode="automatic",
        ).as_dict()


class PromotedArtifactScopedSpeculativeMTPFakeLLM(
    ArtifactScopedSpeculativeMTPFakeLLM
):
    @staticmethod
    def _plan(**kwargs: Any) -> dict[str, Any]:
        plan = ArtifactScopedSpeculativeMTPFakeLLM._plan(**kwargs)
        promoted = bool(plan["admitted"])
        plan["automatic_eligible"] = promoted
        plan["plan_fingerprint"] = "sha256:" + ("c" if promoted else "d") * 64
        return plan


class RealizedGroupArtifactScopedSpeculativeMTPFakeLLM(
    PromotedArtifactScopedSpeculativeMTPFakeLLM
):
    @staticmethod
    def _plan(**kwargs: Any) -> dict[str, Any]:
        rows = int(kwargs["realized_group_rows"])
        admitted = bool(
            rows in {1, 2}
            and kwargs["sampling_mode"] == "greedy_fast"
            and kwargs["kv_storage"] in {"auto", "bf16"}
            and kwargs["memory_fit"]
        )
        reason = f"qualified_automatic_c{rows}_b3" if admitted else "scope_not_qualified"
        return {
            "schema_version": 1,
            "plan_fingerprint": "sha256:" + ("e" if rows == 1 else "f") * 64,
            "key": {
                "realized_group_rows": rows,
            },
            "admitted": admitted,
            "selected_route": "speculative_mtp" if admitted else "default",
            "selected_candidate_count": 3 if admitted else 0,
            "reason": reason,
            "strict_fallback_key": "gguf_target_ar",
            "evidence_key": f"fake-qwen38-q4km-c{rows}-b3",
            "evidence_artifacts": [f"benchmarks/results/fake-qwen38-q4km-c{rows}.json"],
            "automatic_eligible": admitted,
        }


class C2OnlyArtifactScopedSpeculativeMTPFakeLLM(
    RealizedGroupArtifactScopedSpeculativeMTPFakeLLM
):
    @staticmethod
    def _plan(**kwargs: Any) -> dict[str, Any]:
        plan = RealizedGroupArtifactScopedSpeculativeMTPFakeLLM._plan(**kwargs)
        if int(kwargs["realized_group_rows"]) == 1 and bool(plan["admitted"]):
            plan["admitted"] = False
            plan["selected_route"] = "default"
            plan["selected_candidate_count"] = 0
            plan["reason"] = "physical_group_not_qualified"
            plan["automatic_eligible"] = False
        return plan


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


class WideScratchProbeFakeLLM(FakeLLM):
    """Fails the scratch probe above width one, the way a HIP allocation does."""

    def prepare_request_scratch(
        self,
        *,
        max_prompt_tokens: int,
        max_new_tokens: int = 0,
        sampling_params: SamplingParams | None = None,
        max_batch_size: int = 1,
        release_after_probe: bool = True,
    ) -> dict[str, Any]:
        if int(max_batch_size) > 1:
            self.scratch_prepares.append({"max_batch_size": int(max_batch_size), "failed": True})
            raise MemoryError("resident session allocation failed (HIP error 2: out of memory)")
        return super().prepare_request_scratch(
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
            max_batch_size=max_batch_size,
            release_after_probe=release_after_probe,
        )


class _FakeCapacityEstimate:
    """The fields the startup probe preflight reads off a capacity estimate."""

    def __init__(
        self,
        *,
        allocatable_context_tokens: int,
        requested_total_bytes: int,
        usable_bytes: int,
    ) -> None:
        self.allocatable_context_tokens = int(allocatable_context_tokens)
        self.requested_total_bytes = int(requested_total_bytes)
        self.usable_bytes = int(usable_bytes)
        self.available_bytes = int(usable_bytes)
        self.reserve_bytes = 0


class NarrowCapacityFakeLLM(FakeLLM):
    """Prices one probe session at 20 GiB against 8 GiB usable."""

    def resident_capacity_estimate(
        self,
        *,
        max_batch_size: int = 1,
        requested_context_tokens: int | None = None,
    ) -> _FakeCapacityEstimate:
        del max_batch_size, requested_context_tokens
        return _FakeCapacityEstimate(
            allocatable_context_tokens=4096,
            requested_total_bytes=20 * 1024**3,
            usable_bytes=8 * 1024**3,
        )


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


class BlockingPrepareFakeLLM(PromptPreparingFakeLLM):
    """A fake whose ``tokenize`` blocks and whose stream runs until stopped.

    ``EngineService`` runs the engine on one driver thread, so a control command
    (the server's prompt tokenization) waits behind whatever that thread is doing:
    a context prepare that grows the KV pool, or one prefill tick of a large
    prompt. The wait is real; what matters is which thread absorbs it.
    """

    supports_controlled_streaming = True
    supports_stream_many = True

    def __init__(self, *, stream_delay_s: float = 0.05) -> None:
        super().__init__(stream=True)
        self.stream_delay_s = float(stream_delay_s)
        self.tokenize_entered = threading.Event()
        self.tokenize_release = threading.Event()
        self.stream_stop = threading.Event()
        self.tokenize_threads: list[int] = []

    def tokenize(self, text: str) -> tuple[int, ...]:
        self.tokenize_threads.append(threading.get_ident())
        if "SLOW-PREP" in str(text):
            self.tokenize_entered.set()
            assert self.tokenize_release.wait(timeout=10.0)
        return super().tokenize(text)

    def stream(self, prompt: str, sampling_params: SamplingParams):
        """Keep yielding until the test stops it, so the stream outlives a wait."""

        index = 0
        while not self.stream_stop.is_set():
            yield f"chunk-{index}"
            index += 1
            time.sleep(self.stream_delay_s)

    def stream_many_detailed(self, prompts, sampling_params):
        """One row: the incremental chat path gates on this being callable."""

        yield from self.stream(prompts[0], sampling_params)


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


class AdmissionRejectedFakeLLM(FakeLLM):
    def _reject(self, prompts, sampling_params: SamplingParams) -> None:
        prompts = tuple(prompts)
        self.calls.append((prompts, sampling_params))
        raise GenerationAdmissionRejected(
            "device KV pool high-water rejection",
            resource="device_kv_pool",
            requested_units=17,
            current_units=129,
            capacity_units=129,
        )

    def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
        self._reject(prompts, sampling_params)
        raise AssertionError("unreachable")

    def stream(self, prompt: str, sampling_params: SamplingParams):
        self.stream_calls.append((str(prompt), sampling_params))
        self._reject((prompt,), sampling_params)
        yield  # pragma: no cover - keeps this method a generator


class BackendExecutionFailedFakeLLM(FakeLLM):
    """A contained execution failure: this request died, the engine did not."""

    def _fail(self, prompts, sampling_params: SamplingParams) -> None:
        self.calls.append((tuple(prompts), sampling_params))
        raise GenerationExecutionFailed(
            phase="decode",
            request_ids=(7,),
            work_kind="decode_batch",
            mutation="none",
            reason="sampler row 1 exceeds the packed capacity of 0",
            error=ValueError("capacity"),
        )

    def generate_detailed(self, prompts, sampling_params: SamplingParams) -> list[GenerationOutput]:
        self._fail(prompts, sampling_params)
        raise AssertionError("unreachable")

    def stream(self, prompt: str, sampling_params: SamplingParams):
        self.stream_calls.append((str(prompt), sampling_params))
        self._fail((prompt,), sampling_params)
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


class FailingSpeculativeMTPFakeLLM(SpeculativeMTPFakeLLM):
    """MTP route that always raises, so the scope breaker opens from real failures."""

    def generate_speculative_mtp_detailed(self, prompts, sampling_params):
        self.mtp_calls.append((tuple(str(prompt) for prompt in prompts), sampling_params))
        raise RuntimeError("injected MTP runtime failure")

    def stream_speculative_mtp_detailed(self, prompts, sampling_params):
        self.mtp_calls.append((tuple(str(prompt) for prompt in prompts), sampling_params))
        raise RuntimeError("injected MTP runtime failure")
        yield  # pragma: no cover - makes this a generator, like the real streamer


class OverDeclaredDepthSpeculativeMTPFakeLLM(DeclaredCellSpeculativeMTPFakeLLM):
    """Requests K4 while the model declaration covers K3."""

    speculative_candidate_budget = 4
