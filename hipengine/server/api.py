"""OpenAI-compatible FastAPI surface for hipEngine.

The server layer is optional and intentionally thin: it adapts OpenAI-style JSON
requests to the torch-free ``hipengine.LLM.generate()`` library API.  HTTP
requests are routed through the generation batcher; the remaining async lock is
limited to short model/session preparation mutations.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import json
import logging
import math
import os
import re
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from anyio import CancelScope
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

try:  # Pydantic v2; FastAPI's current default.
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - Pydantic v1 compatibility
    ConfigDict = None  # type: ignore[assignment]

from starlette.concurrency import run_in_threadpool

from hipengine import LLM, SamplingParams
from hipengine.generation import (
    DecodeState,
    FinishDetails,
    GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS,
    GenerationAdmissionRejected,
    GenerationCancellationToken,
    GenerationCancelled,
    GenerationDeadlineExceeded,
    GenerationOutput,
    GenerationStreamChunk,
    GenerationTelemetry,
    NATIVE_GPU_SAMPLER_UNSUPPORTED_CAPABILITIES,
    PreparedPromptInput,
    PromptInput,
    SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS,
    SPECULATIVE_MTP_INCOMPATIBLE_FIELDS,
    ThinkingBudgetState,
    TokenLogprob,
    derive_row_seed,
    relax_thinking_budget_for_mtp,
    speculative_mtp_sampling_blockers,
    supports_speculative_mtp_sampling,
)
from hipengine.generation.constraints import JsonObjectConstraintState, ToolCallConstraintSpec
from hipengine.generation.registry import normalize_prompt_input
from hipengine.kvcache import resolve_prefix_cache_mode
from hipengine.speculative.policy import (
    DEFAULT_AUTO_DEPTH_POLICY,
    select_offline_speculative_depth,
)
from hipengine.speculative.serving import SpeculativeMTPStaticEligibility
from hipengine.tokenization.identity import token_ids_sha256


_LOGGER = logging.getLogger("uvicorn.error")
# Solo-idle batch dispatch slice for the arrival-aware generation window.
_SOLO_DISPATCH_SLICE_SECONDS = 0.002
_GRAPH_KERNEL_TIME_HISTOGRAM_BUCKET_SET = frozenset(GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS)
_THINKING_START_MARKER = "<think>"
_THINKING_CLOSE_MARKER = "</think>"
_TOOL_CALL_START_MARKER = "<tool_call>"
_TOOL_CALL_END_MARKER = "</tool_call>"
_CHAT_TEMPLATE_TERMINAL_MARKERS = ("<|im_end|>",)
_CHAT_TEMPLATE_RESIDUE_MARKERS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")
_CHAT_TEMPLATE_RESIDUE_ROLES = ("assistant", "developer", "system", "tool", "user")
_TOOL_CALL_ARGUMENT_STREAM_CHARS = 128
_CHAT_MESSAGE_ROLES = ("assistant", "developer", "system", "tool", "user")
_CHAT_MESSAGE_ROLE_SET = frozenset(_CHAT_MESSAGE_ROLES)
_LOGPROB_OMISSION_REASON = "backend_omitted_logprob"
_PROMPT_LOGPROB_OMISSION_REASON = "prompt_logprob_unavailable"
_CONTINUATION_TTL_SECONDS = 15 * 60
_SESSION_COMMIT_MODES = (
    "append_none",
    "append_prompt_only",
    "append_visible_only",
    "append_all",
)
_SESSION_STATEFUL_DEFAULT_COMMIT = "append_visible_only"
_SESSION_CONTEXT_OVERFLOW_POLICIES = (
    "reject",
    "auto_clear_transient",
    "new_session",
    "truncate_oldest_visible",
)
_SESSION_CONTEXT_OVERFLOW_POLICY_ALIASES = {"fail": "reject"}
_SESSION_ALLOWED_STATEFUL_KEYS = frozenset({"id", "commit", "context_overflow_policy"})
_SESSION_UNSAFE_VISIBLE_REASONS = frozenset(
    {
        "length",
        "cancelled",
        "deadline_exceeded",
        "invalid_tool_call",
        "schema_violation",
        "tool_required_not_satisfied",
    }
)
_CONTINUATION_INELIGIBLE_WHEN = (
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
)
_CONTINUATION_UNSUPPORTED_RESUME_FIELDS = (
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
)
_TOOL_RESULT_VALIDATION_FAILURE_REASONS = (
    "invalid_tool_call",
    "tool_required_not_satisfied",
    "schema_violation",
)
_INVALID_TOOL_CALL_ERROR_MODES = ("finish_details", "hard_error")
_STRUCTURED_OUTPUT_RESULT_VALIDATION_FAILURE_REASONS = ("schema_violation",)
_JSON_SCHEMA_ANNOTATION_KEYWORDS = (
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "format",
)
_JSON_SCHEMA_SUPPORTED_TYPES = frozenset(
    {
        "array",
        "boolean",
        "integer",
        "null",
        "number",
        "object",
        "string",
    }
)
_JSON_SCHEMA_SUPPORTED_KEYS = frozenset(
    {
        "type",
        "enum",
        "const",
        "$ref",
        "$defs",
        "definitions",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "properties",
        "patternProperties",
        "propertyNames",
        "required",
        "dependentRequired",
        "dependentSchemas",
        "additionalProperties",
        "minProperties",
        "maxProperties",
        "items",
        "contains",
        "minItems",
        "maxItems",
        "minContains",
        "maxContains",
        "uniqueItems",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        *_JSON_SCHEMA_ANNOTATION_KEYWORDS,
    }
)
_AGENTIC_REPLAY_FAILURE_REASONS = frozenset(
    (
        *_TOOL_RESULT_VALIDATION_FAILURE_REASONS,
        *_STRUCTURED_OUTPUT_RESULT_VALIDATION_FAILURE_REASONS,
    )
)
_GENERATION_SCHEDULER_FAIRNESS_POLICY = "fifo_compatible_sampling_key"
_CHAT_SESSION_SNAPSHOT_SCHEMA = "hipengine.chat_session_snapshot.v1"
_SPECULATIVE_MTP_SERVING_MODES = ("off", "opt_in", "auto", "enabled")
_SPECULATIVE_MTP_THINKING_MODES = ("hint", "hard")
_SPECULATIVE_MTP_DEFAULT_ROUTE = "default"
_SPECULATIVE_MTP_BATCH_ROUTE = "speculative_mtp"
_SPECULATIVE_MTP_AUTO_ROUTE = "speculative_mtp_auto"
_SPECULATIVE_MTP_K0_ROUTE = "speculative_mtp_k0"


class _SpeculativeMTPRouteReason(str, Enum):
    AUTOMATIC_SCOPE_NOT_PROMOTED = "automatic_mtp_scope_not_promoted"


_SPECULATIVE_MTP_AUTO_REJECTION_REASON = (
    _SpeculativeMTPRouteReason.AUTOMATIC_SCOPE_NOT_PROMOTED.value
)
_SPECULATIVE_MTP_AUTO_EVIDENCE = (
    "benchmarks/results/2026-08-26-gfx1151-specdec2-perf-p9-fixed-policy.json"
)
_SPECULATIVE_PROVIDER_ROUTE = "speculative"
_SPECULATIVE_PROVIDER_ALLOWED_REQUEST_KEYS = frozenset(
    {"enabled", "provider", "candidate_budget"}
)
_GGUF_DEFAULT_AR_MAX_ACTIVE_REQUESTS = 4
_GGUF_MTP_MAX_ACTIVE_REQUESTS = 4
_UNSUPPORTED_GRAMMAR_FIELDS = (
    "grammar",
    "guided_grammar",
    "guided_decoding_backend",
)
_GUIDED_JSON_FIELD = "guided_json"
_GUIDED_REGEX_FIELD = "guided_regex"
_GUIDED_CHOICE_FIELD = "guided_choice"
_GUIDED_PATCH_FIELDS = ("guided_patch", "guided_diff")
_GUIDED_PATCH_FORMATS = ("unified_diff",)
_GUIDED_PATCH_FENCED_POLICIES = ("optional", "required", "forbidden")
_GUIDED_PATCH_FENCE_LABELS = ("diff", "patch")
_PATCH_FENCE_RE = re.compile(r"\A\s*```(?P<label>[^\n`]*)\n(?P<body>.*?)\n```\s*\Z", re.DOTALL)
_UNIFIED_DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@(?: .*)?$")
_UNIFIED_DIFF_METADATA_PREFIXES = (
    "diff --git ",
    "index ",
    "new file mode ",
    "deleted file mode ",
    "old mode ",
    "new mode ",
    "similarity index ",
    "dissimilarity index ",
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
)
_UNIFIED_DIFF_BINARY_PREFIXES = ("Binary files ", "GIT binary patch")


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for the optional OpenAI-compatible server."""

    model: str
    backend: str = "auto"
    quant: str = "auto"
    served_model_name: str | None = None
    api_key: str | None = None
    eager_load: bool = True
    eager_load_prompt: str = "one two three four"
    eager_load_max_tokens: int = 1
    startup_chat_smoke: bool = True
    startup_scratch_probe: bool = True
    startup_min_free_mib: int | None = None
    max_context_tokens: int | None = None
    chat_default_max_tokens: int | None = 4096
    kv_storage: str = "auto"
    kv_scale_dtype: str = "fp16"
    kv_scale_granularity: str = "per_token_head"
    generation_batch_window_ms: float = 0.0
    stream_queue_max_chunks: int = 16
    shutdown_grace_seconds: float = 5.0
    request_timeout_ms: float | None = None
    metrics: str = "off"
    prefix_cache: str = "off"
    debug: bool = False
    replay_dir: str | None = None
    replay_redaction: str = "hash"
    max_queued_requests: int | None = None
    max_active_requests: int | None = None
    max_chat_sessions: int | None = None
    queue_retry_after_seconds: int = 1
    speculative_mtp_serving: str = "auto"
    speculative_mtp_thinking: str = "hint"
    speculative_provider: str | None = None
    draft_model: str | None = None
    speculative_candidate_budget: int = 4
    created: int = field(default_factory=lambda: int(time.time()))
    execution_profile: str | None = None

    def __post_init__(self) -> None:
        from hipengine.execution_profiles import resolve_requested_execution_profile

        profile = resolve_requested_execution_profile(self.execution_profile)
        object.__setattr__(
            self,
            "execution_profile",
            None if profile is None else profile.value,
        )
        mode = str(self.speculative_mtp_serving).strip().lower().replace("-", "_")
        if mode not in _SPECULATIVE_MTP_SERVING_MODES:
            raise ValueError(
                "speculative_mtp_serving must be one of: "
                + ", ".join(_SPECULATIVE_MTP_SERVING_MODES)
            )
        object.__setattr__(self, "speculative_mtp_serving", mode)
        thinking = str(self.speculative_mtp_thinking).strip().lower().replace("-", "_")
        if thinking not in _SPECULATIVE_MTP_THINKING_MODES:
            raise ValueError(
                "speculative_mtp_thinking must be one of: "
                + ", ".join(_SPECULATIVE_MTP_THINKING_MODES)
            )
        object.__setattr__(self, "speculative_mtp_thinking", thinking)
        provider = (
            None
            if self.speculative_provider is None
            else str(self.speculative_provider).strip()
        )
        drafter = None if self.draft_model is None else str(self.draft_model).strip()
        if provider == "":
            raise ValueError("speculative_provider must be non-empty when set")
        if drafter == "":
            raise ValueError("draft_model must be non-empty when set")
        if provider is None and drafter is not None:
            raise ValueError("draft_model requires speculative_provider")
        if provider is not None and drafter is None:
            raise ValueError("speculative_provider requires draft_model")
        candidate_budget = int(self.speculative_candidate_budget)
        if candidate_budget <= 0:
            raise ValueError("speculative_candidate_budget must be positive")
        object.__setattr__(self, "speculative_provider", provider)
        object.__setattr__(self, "draft_model", drafter)
        object.__setattr__(self, "speculative_candidate_budget", candidate_budget)
        if int(self.stream_queue_max_chunks) < 2:
            raise ValueError("stream_queue_max_chunks must be at least 2")
        if float(self.shutdown_grace_seconds) < 0.0:
            raise ValueError("shutdown_grace_seconds must be non-negative")
        object.__setattr__(self, "stream_queue_max_chunks", int(self.stream_queue_max_chunks))
        object.__setattr__(self, "shutdown_grace_seconds", float(self.shutdown_grace_seconds))

    @property
    def model_id(self) -> str:
        """Public model identifier exposed through the OpenAI API."""

        if self.served_model_name:
            return self.served_model_name
        path = Path(self.model)
        if path.exists() and path.name:
            return path.name
        return self.model


def _server_model_identity(config: ServerConfig, engine: Any | None = None) -> dict[str, str]:
    """Return requested model identity, replacing selectors after engine resolution."""

    backend = str(config.backend)
    quant = str(config.quant)
    if engine is not None:
        resolved_backend = str(getattr(engine, "_resolved_backend", "") or "").strip()
        resolved_quant = str(getattr(engine, "_resolved_quant", "") or "").strip()
        if resolved_backend and resolved_backend != "auto":
            backend = resolved_backend
        if resolved_quant and resolved_quant != "auto":
            quant = resolved_quant
    return {"id": config.model_id, "backend": backend, "quant": quant}


def _kv_capability_provenance(engine: Any | None) -> dict[str, Any]:
    """Return loaded-engine approximate-KV capability provenance."""

    pending = [engine]
    seen: set[int] = set()
    while pending:
        target = pending.pop(0)
        if target is None or id(target) in seen:
            continue
        seen.add(id(target))
        raw = getattr(target, "kv_capability_provenance", None)
        payload = raw() if callable(raw) else raw
        if isinstance(payload, Mapping):
            return dict(payload)
        pending.extend(
            (
                getattr(target, "_text_generator", None),
                getattr(target, "inner", None),
            )
        )
    return {
        "schema_version": 1,
        "status": "unavailable",
        "runtime_action": "not_applicable",
        "promotion_eligible": False,
        "diagnostic_override": False,
        "requested": None,
        "effective_kv_storage": None,
        "artifact": None,
        "evidence": None,
        "reason": "loaded engine does not expose artifact-scoped KV capability provenance",
    }


def _server_execution_profile(
    config: ServerConfig,
    engine: Any | None = None,
) -> dict[str, Any]:
    requested = config.execution_profile
    if engine is None:
        return {
            "requested": requested,
            "resolved": None,
            "manifest_sha256": None,
            "strict_manifest_sha256": None,
            "fell_back_to_strict": None,
            "migration_default_preserved": requested is None,
        }
    resolved = getattr(engine, "resolved_execution_profile", requested)
    manifest_hash = getattr(engine, "execution_profile_manifest_sha256", None)
    strict_manifest_hash = getattr(
        engine, "execution_profile_strict_manifest_sha256", None
    )
    fell_back = getattr(engine, "execution_profile_fell_back_to_strict", None)
    return {
        "requested": requested,
        "resolved": resolved,
        "manifest_sha256": manifest_hash,
        "strict_manifest_sha256": strict_manifest_hash,
        "fell_back_to_strict": fell_back,
        "migration_default_preserved": resolved is None,
    }


def _server_model_uses_gguf(config: ServerConfig, engine: Any | None = None) -> bool:
    """Return whether server grouping must use the retained GGUF width caps."""

    quant = _server_model_identity(config, engine)["quant"]
    if quant != "auto":
        return quant.startswith("gguf_")

    path = Path(config.model).expanduser()
    if path.suffix.lower() == ".gguf":
        return True
    from hipengine.loading import resolve_model_path

    path = resolve_model_path(config.model)
    if path.is_file():
        return path.suffix.lower() == ".gguf"
    return path.is_dir() and any(path.glob("*.gguf"))


class OpenAIHTTPError(Exception):
    """Exception converted to an OpenAI-style error JSON body."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
        finish_details: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ):
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code
        self.param = param
        self.finish_details = None if finish_details is None else dict(finish_details)
        self.extra = {} if extra is None else dict(extra)
        self.headers = {} if headers is None else dict(headers)
        super().__init__(message)


def _generation_admission_http_error(
    exc: GenerationAdmissionRejected,
    *,
    retry_after_seconds: int,
    error_extra: Mapping[str, Any] | None = None,
) -> OpenAIHTTPError:
    """Convert a bounded resource miss into the public retryable admission error."""

    extra = deepcopy(dict(error_extra or {}))
    hipengine = extra.get("hipengine")
    if not isinstance(hipengine, dict):
        hipengine = {}
        extra["hipengine"] = hipengine
    routing = hipengine.get("routing")
    if not isinstance(routing, dict):
        routing = {}
        hipengine["routing"] = routing
    routing["reason"] = "engine_busy"
    routing["overload_source"] = "kv_pool_capacity"
    routing["admission"] = exc.to_json_dict()
    return OpenAIHTTPError(
        429,
        str(exc),
        error_type="rate_limit_error",
        code="engine_busy",
        extra=extra,
        headers={"Retry-After": str(max(1, int(retry_after_seconds)))},
    )


_ERROR_TAXONOMY: dict[str, dict[str, Any]] = {
    "unsupported_parameter": {
        "status_code": 400,
        "retryable": False,
        "emitted": True,
        "description": "A request field or value is not supported by this server.",
    },
    "unsupported_feature": {
        "status_code": 501,
        "retryable": False,
        "emitted": True,
        "description": "The served model lacks a requested optional runtime feature.",
    },
    "invalid_tool_call": {
        "status_code": 400,
        "retryable": False,
        "emitted": True,
        "description": (
            "Emitted as finish_details.reason for strict tool result-validation failures; "
            "also available as an opt-in HTTP/SSE hard error via invalid_tool_call_error_mode."
        ),
    },
    "schema_violation": {
        "status_code": 422,
        "retryable": False,
        "emitted": True,
        "description": "A request, structured-output result, or strict tool result schema was violated.",
    },
    "invalid_continuation": {
        "status_code": 400,
        "retryable": False,
        "emitted": True,
        "description": "A continuation_id is unknown, already consumed, or incompatible with this request.",
    },
    "continuation_expired": {
        "status_code": 410,
        "retryable": False,
        "emitted": True,
        "description": "A continuation_id existed but expired before it was resumed.",
    },
    "context_overflow": {
        "status_code": 400,
        "retryable": False,
        "emitted": True,
        "description": "Prompt plus generation budget exceeds the admitted context.",
    },
    "deadline_exceeded": {
        "status_code": 408,
        "retryable": True,
        "emitted": True,
        "description": "The request exceeded its timeout_ms or server default deadline.",
    },
    "cancelled": {
        "status_code": 499,
        "retryable": True,
        "emitted": True,
        "description": "The client disconnected or the queued request was cancelled.",
    },
    "engine_busy": {
        "status_code": 429,
        "retryable": True,
        "emitted": True,
        "description": "The server admission queue or chat-session cap is full.",
    },
    "model_unavailable": {
        "status_code": 404,
        "retryable": False,
        "emitted": True,
        "description": "The requested model is not served by this process.",
    },
    "routing_failed": {
        "status_code": 502,
        "retryable": True,
        "emitted": False,
        "description": "Reserved for future multi-model or multi-worker routing failures.",
    },
}

_ERROR_CODE_ALIASES = {
    "context_length_exceeded": "context_overflow",
    "invalid_request": "schema_violation",
    "model_not_found": "model_unavailable",
    "unsupported_content_type": "unsupported_parameter",
    "validation_error": "schema_violation",
}


def _canonical_error_code(code: str | None) -> str | None:
    if code is None:
        return None
    return _ERROR_CODE_ALIASES.get(str(code), str(code))


def _error_taxonomy_manifest() -> dict[str, Any]:
    return {
        "schema": "hipengine.error_taxonomy.v1",
        "codes": [
            {"code": code, **dict(metadata)}
            for code, metadata in sorted(_ERROR_TAXONOMY.items())
        ],
        "aliases": [
            {"legacy_code": legacy, "code": canonical}
            for legacy, canonical in sorted(_ERROR_CODE_ALIASES.items())
        ],
    }


def _error_extension(status_code: int, code: str | None) -> dict[str, Any] | None:
    canonical = _canonical_error_code(code)
    if canonical is None:
        return None
    payload: dict[str, Any] = {
        "code": canonical,
        "status_code": int(status_code),
    }
    legacy = None if code is None else str(code)
    if legacy and legacy != canonical:
        payload["legacy_code"] = legacy
    metadata = _ERROR_TAXONOMY.get(canonical)
    if metadata is not None:
        payload["retryable"] = bool(metadata["retryable"])
    return payload


def _error_payload(
    *,
    message: str,
    error_type: str,
    code: str | None,
    param: str | None,
    status_code: int,
    finish_details: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "message": message,
        "type": error_type,
        "param": param,
        "code": code,
    }
    extension = _error_extension(status_code, code)
    if extension is not None:
        payload["hipengine"] = extension
    if finish_details is not None:
        payload["finish_details"] = dict(finish_details)
    if extra is not None:
        extra_payload = dict(extra)
        extra_hipengine = extra_payload.pop("hipengine", None)
        if isinstance(extra_hipengine, Mapping):
            existing = payload.get("hipengine")
            if isinstance(existing, Mapping):
                payload["hipengine"] = {**dict(existing), **dict(extra_hipengine)}
            else:
                payload["hipengine"] = dict(extra_hipengine)
        payload.update(extra_payload)
    return payload


async def _maybe_write_replay_artifact(
    config: ServerConfig,
    request: Request,
    error_payload: Mapping[str, Any] | None,
    *,
    engine: Any | None = None,
    result_payload: Mapping[str, Any] | None = None,
) -> None:
    if not config.replay_dir:
        return
    try:
        body_bytes = await request.body()
        artifact = _build_replay_artifact(
            config,
            request,
            error_payload,
            body_bytes,
            engine=engine,
            result_payload=result_payload,
        )
        _write_replay_artifact(Path(config.replay_dir), artifact)
    except Exception as exc:  # pragma: no cover - best-effort diagnostic path
        _LOGGER.warning("failed to write replay artifact: %s", exc)


async def _maybe_write_agentic_result_replay_artifact(
    config: ServerConfig,
    request: Request,
    response_payload: Mapping[str, Any],
    *,
    engine: Any | None = None,
) -> None:
    if not config.replay_dir:
        return
    result_payload = _agentic_result_replay_payload(response_payload)
    if result_payload is None:
        return
    await _maybe_write_replay_artifact(
        config,
        request,
        None,
        engine=engine,
        result_payload=result_payload,
    )


async def _maybe_write_stream_error_replay_artifact(
    config: ServerConfig,
    request: Request,
    *,
    message: str,
    status_code: int,
    code: str | None,
    param: str | None = None,
    error_type: str = "server_error",
    finish_details: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    engine: Any | None = None,
) -> None:
    if not config.replay_dir:
        return
    await _maybe_write_replay_artifact(
        config,
        request,
        _error_payload(
            message=message,
            error_type=error_type,
            code=code,
            param=param,
            status_code=status_code,
            finish_details=finish_details,
            extra=extra,
        ),
        engine=engine,
    )


def _agentic_result_replay_payload(response_payload: Mapping[str, Any]) -> dict[str, Any] | None:
    choices = response_payload.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes, bytearray)):
        return None
    failed_choices: list[dict[str, Any]] = []
    first_details: dict[str, Any] | None = None
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        finish_details = choice.get("finish_details")
        if not isinstance(finish_details, Mapping):
            continue
        reason = finish_details.get("reason")
        if reason not in _AGENTIC_REPLAY_FAILURE_REASONS:
            continue
        details = dict(finish_details)
        if first_details is None:
            first_details = details
        failed_choices.append(
            {
                "index": choice.get("index"),
                "finish_reason": choice.get("finish_reason"),
                "finish_details": details,
            }
        )
    if not failed_choices or first_details is None:
        return None
    return {
        "type": "agentic_result_validation",
        "finish_details": first_details,
        "choices": failed_choices,
    }


def _build_replay_artifact(
    config: ServerConfig,
    request: Request,
    error_payload: Mapping[str, Any] | None,
    body_bytes: bytes,
    *,
    engine: Any | None = None,
    result_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body_json = _decode_replay_body(body_bytes)
    redaction = _replay_redaction_mode(config.replay_redaction)
    finish_details = None
    if error_payload is not None:
        finish_details = error_payload.get("finish_details")
    if result_payload is not None and result_payload.get("finish_details") is not None:
        finish_details = result_payload.get("finish_details")
    artifact = {
        "schema": "hipengine.replay.v1",
        "created": int(time.time()),
        "redaction": {
            "mode": redaction,
            "hash": "sha256" if redaction == "hash" else None,
        },
        "request": {
            "method": request.method,
            "path": _request_target(request),
            "body_sha256": _sha256_text(body_bytes.decode("utf-8", errors="replace")),
            "json": _redact_replay_value(body_json, redaction=redaction),
            "prompt_hashes": _collect_prompt_hashes(body_json),
        },
        "model": _server_model_identity(config, engine),
        "sampling": _replay_sampling_payload(body_json, redaction=redaction),
        "seeds": _replay_seed_payload(body_json),
        "token_counts": _replay_token_counts(body_json, engine, config),
        "finish_details": finish_details,
        "error": (
            None
            if error_payload is None
            else {
                "type": error_payload.get("type"),
                "code": error_payload.get("code"),
                "param": error_payload.get("param"),
                "hipengine": error_payload.get("hipengine"),
            }
        ),
        "capabilities": _replay_capability_snapshot(config, engine=engine),
    }
    if result_payload is not None:
        artifact["result"] = dict(result_payload)
    return artifact


def _write_replay_artifact(directory: Path, artifact: Mapping[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{int(time.time() * 1000)}-{uuid.uuid4().hex}.json"
    destination = directory / filename
    payload = json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def _decode_replay_body(body: bytes) -> Any:
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return {"_non_json_body_sha256": _sha256_text(body.decode("utf-8", errors="replace"))}


def _replay_redaction_mode(raw: str | None) -> str:
    mode = "hash" if raw is None or str(raw).strip() == "" else str(raw).strip().lower()
    if mode not in {"hash", "none"}:
        raise ValueError("replay_redaction must be one of: hash, none")
    return mode


def _redact_replay_value(value: Any, *, redaction: str) -> Any:
    if isinstance(value, str):
        if redaction == "none":
            return value
        return {"redacted": "sha256", "sha256": _sha256_text(value), "length": len(value)}
    if isinstance(value, list):
        return [_redact_replay_value(item, redaction=redaction) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _redact_replay_value(item, redaction=redaction)
            for key, item in value.items()
        }
    return value


def _collect_prompt_hashes(value: Any, path: str = "$") -> list[dict[str, Any]]:
    hashes: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key in {"prompt", "content", "arguments"} and isinstance(item, str):
                hashes.append(_prompt_hash(child_path, item))
            hashes.extend(_collect_prompt_hashes(item, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hashes.extend(_collect_prompt_hashes(item, f"{path}[{index}]"))
    elif path.startswith("$.prompt[") and isinstance(value, str):
        hashes.append(_prompt_hash(path, value))
    return hashes


def _prompt_hash(path: str, text: str) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": _sha256_text(text),
        "length": len(text),
    }


def _replay_token_counts(body_json: Any, engine: Any | None, config: ServerConfig) -> dict[str, Any]:
    unavailable = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "available": False,
    }
    if engine is None:
        return {**unavailable, "unavailable_reason": "engine_not_loaded"}
    count_entries = _replay_completion_prompt_texts(body_json)
    source = "completion_prompt"
    if count_entries is None:
        chat_prompt = _replay_chat_prompt_text(body_json, engine, config)
        if chat_prompt is not None:
            count_entries = [("$.messages", chat_prompt)]
            source = "chat_prompt"
    if count_entries is None:
        return {**unavailable, "unavailable_reason": "unsupported_request_shape"}
    try:
        entries = [
            {"path": path, "token_count": _count_tokens_strict(engine, text)}
            for path, text in count_entries
        ]
    except Exception:
        return {**unavailable, "unavailable_reason": "token_count_failed"}
    prompt_tokens = sum(int(item["token_count"]) for item in entries)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": None,
        "total_tokens": None,
        "available": True,
        "source": source,
        "entries": entries,
    }


def _replay_completion_prompt_texts(body_json: Any) -> list[tuple[str, str]] | None:
    if not isinstance(body_json, Mapping) or "prompt" not in body_json:
        return None
    prompt = body_json.get("prompt")
    if isinstance(prompt, str):
        return [("$.prompt", prompt)]
    if isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
        return [(f"$.prompt[{index}]", item) for index, item in enumerate(prompt)]
    return None


def _replay_chat_prompt_text(body_json: Any, engine: Any, config: ServerConfig) -> str | None:
    if not isinstance(body_json, Mapping) or "messages" not in body_json:
        return None
    try:
        request = ChatCompletionRequest(**dict(body_json))
        prompt, _thinking = _render_chat_prompt_for_request(
            request,
            chat_default_max_tokens=config.chat_default_max_tokens,
            engine=engine,
            max_context_tokens=config.max_context_tokens,
        )
    except Exception:
        return None
    return prompt


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _replay_sampling_payload(body_json: Any, *, redaction: str) -> dict[str, Any]:
    if not isinstance(body_json, dict):
        return {}
    keys = (
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "suppress_token_ids",
        "min_tokens",
        "eos_token_id",
        "n",
        "stream",
        "timeout_ms",
        "stop",
        "ignore_eos",
        "reasoning_effort",
        "enable_thinking",
        "max_think_tokens",
        "min_answer_tokens",
        "hard_think_cap",
        "soft_close_window",
        "hard_close_message",
        "hard_close_sequence",
        "chat_template_kwargs",
        "thinking",
        "reasoning",
        "tool_choice",
        "parallel_tool_calls",
        "logprobs",
        "top_logprobs",
        "response_format",
        "guided_json",
        "guided_regex",
        "guided_choice",
        "guided_patch",
        "guided_diff",
        "thinking_token_budget",
    )
    payload = {key: body_json[key] for key in keys if key in body_json}
    return _redact_replay_value(payload, redaction=redaction)


def _replay_seed_payload(body_json: Any) -> dict[str, Any]:
    if not isinstance(body_json, dict):
        return {"seed": None, "row_seeds": []}
    return {"seed": body_json.get("seed"), "row_seeds": []}


def _reasoning_control_fields() -> list[str]:
    return [
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
    ]


_THINKING_EFFORT_DEFAULTS: dict[str, dict[str, int]] = {
    "minimal": {"hard_think_cap": 256, "soft_close_window": 64, "min_answer_tokens": 256},
    "low": {"hard_think_cap": 512, "soft_close_window": 128, "min_answer_tokens": 512},
    "medium": {"hard_think_cap": 4096, "soft_close_window": 512, "min_answer_tokens": 1024},
    "high": {"hard_think_cap": 16384, "soft_close_window": 1024, "min_answer_tokens": 2048},
    "xhigh": {"hard_think_cap": 32768, "soft_close_window": 2048, "min_answer_tokens": 4096},
    "max": {"hard_think_cap": 32768, "soft_close_window": 2048, "min_answer_tokens": 4096},
}
_THINKING_BUDGET_UNSET = object()


def _thinking_effort_defaults_capability() -> dict[str, dict[str, int]]:
    return {name: dict(values) for name, values in _THINKING_EFFORT_DEFAULTS.items()}


def _tool_schema_subset() -> list[str]:
    return [
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
    ]


def _tools_capability(
    *,
    tokenizer_backed: bool,
    strict_decoding_backed: bool | None = None,
    tool_parser: Any | None = None,
) -> dict[str, Any]:
    strict_backed = (
        bool(tokenizer_backed)
        if strict_decoding_backed is None
        else bool(strict_decoding_backed)
    )
    capability = {
        "enabled": True,
        "strict_decoding": strict_backed,
        "strict_decoding_scope": (
            "tokenizer_aware_canonical_envelope_and_root_json_syntax"
            if strict_backed
            else "none"
        ),
        "strict_result_validation": True,
        "result_validation_failure_reasons": list(_TOOL_RESULT_VALIDATION_FAILURE_REASONS),
        "invalid_tool_call_error_mode_field": "invalid_tool_call_error_mode",
        "invalid_tool_call_error_modes": list(_INVALID_TOOL_CALL_ERROR_MODES),
        "default_invalid_tool_call_error_mode": "finish_details",
        "hard_error_surfaces": ["http", "sse"],
        "schema_validation": "function_strict",
        "schema_subset": _tool_schema_subset(),
        "unsupported_schema_keywords_rejected": True,
        "annotation_keywords_ignored": list(_JSON_SCHEMA_ANNOTATION_KEYWORDS),
        "format": "qwen_tool_call_json",
        "compatibility_parser_repairs": [
            "duplicated_tool_call_start",
            "incomplete_duplicate_tool_prefix_control_residue",
            "outer_qwen_template_control_residue",
        ],
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
        "streaming_argument_chunk_chars": _TOOL_CALL_ARGUMENT_STREAM_CHARS,
        "no_tool_start_suppression": tokenizer_backed,
        "required_tool_start_forcing": tokenizer_backed,
        "required_tool_start_forcing_scope": (
            "initial_or_after_tokenized_thinking_close" if tokenizer_backed else "none"
        ),
        "specific_tool_name_prefix_forcing": tokenizer_backed,
        "specific_tool_name_prefix_forcing_scope": (
            "atomic_tool_call_name_and_arguments_key" if tokenizer_backed else "none"
        ),
        "strict_tool_schema_prefix_anchor": tokenizer_backed,
        "strict_tool_schema_prefix_anchor_scope": (
            "selected_closed_object_first_required_string_key" if tokenizer_backed else "none"
        ),
        "tool_call_close_repair": tokenizer_backed,
        "tool_call_close_repair_scope": (
            "tokenizer_safe_marker_or_object_envelope_then_stop" if tokenizer_backed else "none"
        ),
    }
    parser_capability = getattr(tool_parser, "capabilities", None)
    if isinstance(parser_capability, Mapping):
        capability.update(dict(parser_capability))
    return capability


def _structured_outputs_capability(*, tokenizer_backed: bool) -> dict[str, Any]:
    return {
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
        "guided_patch_formats": list(_GUIDED_PATCH_FORMATS),
        "guided_diff_formats": list(_GUIDED_PATCH_FORMATS),
        "guided_patch_fenced_policies": list(_GUIDED_PATCH_FENCED_POLICIES),
        "guided_diff_fenced_policies": list(_GUIDED_PATCH_FENCED_POLICIES),
        "guided_patch_default_fenced_policy": "optional",
        "guided_diff_default_fenced_policy": "optional",
        "guided_patch_fence_labels": list(_GUIDED_PATCH_FENCE_LABELS),
        "guided_diff_fence_labels": list(_GUIDED_PATCH_FENCE_LABELS),
        "strict_decoding": tokenizer_backed,
        "strict_decoding_scope": "root_object_json_syntax" if tokenizer_backed else "none",
        "strict_result_validation": True,
        "decode_time_close_forcing": (
            "host_json_object_parse_validated_suffix" if tokenizer_backed else "none"
        ),
        "length_finish_structural_validation": "root_object_json_prefix",
        "result_validation_failure_reasons": list(
            _STRUCTURED_OUTPUT_RESULT_VALIDATION_FAILURE_REASONS
        ),
        "schema_validation": "json_schema_subset",
        "schema_subset": _tool_schema_subset(),
        "unsupported_schema_keywords_rejected": True,
        "annotation_keywords_ignored": list(_JSON_SCHEMA_ANNOTATION_KEYWORDS),
    }


def _grammar_capability() -> dict[str, Any]:
    return {
        "enabled": False,
        "strict_decoding": False,
        "supported": [],
        "unsupported_fields": list(_UNSUPPORTED_GRAMMAR_FIELDS),
        "result_validation_only": [
            "json_object",
            "json_schema",
            _GUIDED_JSON_FIELD,
            _GUIDED_REGEX_FIELD,
            _GUIDED_CHOICE_FIELD,
            *_GUIDED_PATCH_FIELDS,
        ],
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


def _known_unsupported_fields() -> list[str]:
    return list(_UNSUPPORTED_GRAMMAR_FIELDS)


def _admission_capability(config: ServerConfig, *, engine: Any | None = None) -> dict[str, Any]:
    controlled_streaming = _engine_supports_controlled_streaming(engine)
    return {
        "queue": {
            "max_queued_requests": config.max_queued_requests,
            "retry_after_seconds": int(config.queue_retry_after_seconds),
            "rejects_when_full": config.max_queued_requests is not None,
        },
        "active_requests": {
            "max_active_requests": config.max_active_requests,
            "limits_backend_batch_width": config.max_active_requests is not None,
        },
        "chat_sessions": {
            "max_active": config.max_chat_sessions,
            "rejects_new_sessions_when_full": config.max_chat_sessions is not None,
        },
        "streaming": {
            "controlled_model_loop": controlled_streaming,
            "queue_max_chunks": int(config.stream_queue_max_chunks),
            "backpressure_scope": "per_request",
            "shutdown_grace_seconds": float(config.shutdown_grace_seconds),
        },
        "scheduler_fairness": _scheduler_fairness_capability(
            continuous_decode=controlled_streaming
        ),
    }


def _scheduler_fairness_capability(*, continuous_decode: bool = False) -> dict[str, Any]:
    return {
        "policy": _GENERATION_SCHEDULER_FAIRNESS_POLICY,
        "compatible_sampling_coalescing": True,
        "continuous_decode": bool(continuous_decode),
        "preemptive_fairness": False,
    }


def _speculative_mtp_capability(config: ServerConfig, *, engine: Any | None = None) -> dict[str, Any]:
    engine_supported = _engine_supports_speculative_mtp(engine)
    configured_mode = str(config.speculative_mtp_serving)
    serving_route = configured_mode != "off" and engine_supported
    serving_plan = (
        _engine_speculative_mtp_serving_capability(engine)
        if serving_route
        else None
    )
    automatic_promoted = bool(
        serving_plan is not None
        and serving_plan.get("admitted")
        and serving_plan.get("automatic_eligible")
    )
    payload = {
        "serving_route": bool(serving_route),
        "configured_policy": configured_mode,
        "engine_supported": bool(engine_supported),
        "model_has_mtp_tensors": bool(engine_supported),
        "certified_default_scopes": (
            [deepcopy(serving_plan)] if automatic_promoted else []
        ),
        "automatic_route_promoted": automatic_promoted,
        "sampling_compatible": bool(serving_route),
        "compatibility_guard": "supports_speculative_mtp_sampling",
        "allowed_execution_modes": ["greedy_fast"],
        "incompatible_fields": list(SPECULATIVE_MTP_INCOMPATIBLE_FIELDS),
        "incompatible_conditions": dict(SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS),
        "thinking_policy": str(config.speculative_mtp_thinking),
        "processed_target_verification": False,
    }
    if serving_plan is not None:
        payload["certified_explicit_scope"] = deepcopy(serving_plan)
    if not serving_route:
        return payload
    default_enabled = bool(
        configured_mode in {"auto", "enabled"} and automatic_promoted
    )
    payload.update(
        {
            "policy": configured_mode,
            "request_field": "speculative_mtp",
            "default_enabled": default_enabled,
            "streaming_compatible": True,
            "batch_route": _SPECULATIVE_MTP_BATCH_ROUTE,
            "physical_concurrency": "generation2_target_frontier",
            "max_physical_target_slots": 4,
            "route_coalescing_is_physical_concurrency": True,
            "circuit_breaker": deepcopy(
                getattr(
                    engine,
                    "mtp_circuit_breaker_state",
                    {
                        "state": "closed",
                        "operator_disabled": False,
                        "operator_reason": None,
                        "threshold": 3,
                        "open_scopes": [],
                        "failures": [],
                        "transitions": [],
                        "reset_policy": "server_restart",
                    },
                )
            ),
        }
    )
    if configured_mode == "auto":
        plan_artifacts = (
            None if serving_plan is None else serving_plan.get("evidence_artifacts")
        )
        plan_evidence = (
            plan_artifacts[0]
            if isinstance(plan_artifacts, list) and plan_artifacts
            else _SPECULATIVE_MTP_AUTO_EVIDENCE
        )
        payload["auto_route"] = {
            "selected_route": (
                _SPECULATIVE_MTP_BATCH_ROUTE
                if automatic_promoted
                else _SPECULATIVE_MTP_DEFAULT_ROUTE
            ),
            "reason": (
                str(serving_plan.get("reason"))
                if automatic_promoted and serving_plan is not None
                else _SPECULATIVE_MTP_AUTO_REJECTION_REASON
            ),
            "selected_candidate_count": (
                int(serving_plan.get("selected_candidate_count", 0))
                if automatic_promoted and serving_plan is not None
                else 0
            ),
            "policy_key": (
                DEFAULT_AUTO_DEPTH_POLICY.policy_key
                if serving_plan is None
                else serving_plan.get("evidence_key")
            ),
            "policy_fingerprint": (
                DEFAULT_AUTO_DEPTH_POLICY.fingerprint
                if serving_plan is None
                else serving_plan.get("plan_fingerprint")
            ),
            "exact_default_required": True,
            "compatibility_mtp_explicit_only": not automatic_promoted,
            "evidence": plan_evidence,
        }
    return payload


def _speculative_provider_capability(
    config: ServerConfig,
    *,
    engine: Any | None = None,
) -> dict[str, Any]:
    """Describe the generic explicit provider without implying default routing."""

    configured = config.speculative_provider is not None
    supported = configured and _engine_supports_speculative(engine)
    payload: dict[str, Any] = {
        "configured": bool(configured),
        "serving_route": bool(supported),
        "request_field": "speculative",
        "configured_provider": config.speculative_provider,
        "policy": "explicit_only",
        "default_enabled": False,
        "streaming_compatible": bool(
            supported and _engine_supports_speculative_streaming(engine)
        ),
        "candidate_budget": int(config.speculative_candidate_budget),
        "compatibility_guard": "raw_greedy_bf16_c1",
        "processed_target_verification": False,
    }
    if not supported:
        return payload
    capabilities = _engine_speculative_capabilities(engine)
    payload.update(capabilities)
    payload.update(
        {
            "configured": True,
            "serving_route": True,
            "request_field": "speculative",
            "configured_provider": config.speculative_provider,
            "policy": "explicit_only",
            "default_enabled": False,
            "streaming_compatible": _engine_supports_speculative_streaming(engine),
            "candidate_budget": int(config.speculative_candidate_budget),
            "compatibility_guard": "raw_greedy_bf16_c1",
        }
    )
    return payload


def _replay_capability_snapshot(config: ServerConfig, *, engine: Any | None = None) -> dict[str, Any]:
    tokenizer_caps = _tokenizer_capability_flags(engine)
    tokenizer_backed = tokenizer_caps["tokenize"]
    strict_decoding_backed = tokenizer_backed and tokenizer_caps["detokenize"]
    return {
        "model": _server_model_identity(config, engine),
        "context": {
            "configured_max_context_tokens": config.max_context_tokens,
            "chat_default_max_tokens": config.chat_default_max_tokens,
            "chat_default_mode": "auto" if config.chat_default_max_tokens is None else "bounded",
        },
        "features": {
            "chat_completions": True,
            "completions": True,
            "streaming": True,
            "choice_telemetry": _choice_telemetry_capability(),
            "structured_outputs": _structured_outputs_capability(
                tokenizer_backed=strict_decoding_backed
            ),
            "grammars": _grammar_capability(),
            "tools": _tools_capability(
                tokenizer_backed=tokenizer_backed,
                strict_decoding_backed=strict_decoding_backed,
            ),
            "reasoning_controls": {
                "enabled": True,
                "fields": _reasoning_control_fields(),
                "budget_policy": "prompt_hint_plus_tokenized_soft_and_hard_close",
                "token_budget": tokenizer_backed,
                "token_budget_enforced": tokenizer_backed,
                "effort_defaults": _thinking_effort_defaults_capability(),
                "effort_default_clamp": "request_max_tokens_chat_default_or_remaining_context",
                "hard_close_validation": True,
                "hard_close_token_forcing": tokenizer_backed,
                "soft_close_bias": tokenizer_backed,
                "eos_suppression": tokenizer_backed,
            },
            "request_timeouts": {
                "timeout_ms": True,
                "cooperative_backend_deadline": True,
                "cooperative_backend_cancel": True,
                "preemptive_decode_cancel": False,
            },
        },
        "sampling": {
            "execution_modes": [
                "greedy_fast",
                "processed_argmax",
                "host_logits_sample",
                "gpu_sample",
            ],
            "parameters": [
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "repetition_penalty",
                "presence_penalty",
                "frequency_penalty",
                "logit_bias",
                "suppress_token_ids",
                "min_tokens",
                "eos_token_id",
                "json_object_close_forcing",
                "seed",
                "n",
                "stop",
            ],
            "speculative_mtp": _speculative_mtp_capability(config, engine=engine),
            "speculative": _speculative_provider_capability(config, engine=engine),
        },
        "cache": {
            "prefix_cache": config.prefix_cache,
            "kv_storage": config.kv_storage,
            "kv_scale_dtype": config.kv_scale_dtype,
            "kv_scale_granularity": config.kv_scale_granularity,
        },
        "sessions": {
            "resident_context": True,
            "commit_policy": _session_commit_policy_capability(engine=engine),
            "continuations": _session_continuation_capability(),
            "metadata": _session_metadata_capability(config.max_chat_sessions),
        },
        "admission": _admission_capability(config, engine=engine),
        "parallelism": _parallelism_capability(),
        "unsupported_fields": _known_unsupported_fields(),
    }


def _session_commit_policy_capability(*, engine: Any | None = None) -> dict[str, Any]:
    target = _model_chat_protocol_target(engine)
    resident_kv = bool(getattr(target, "supports_resident_session_kv", False))
    return {
        "supported": True,
        "stateful": True,
        "resident_state_reuse": resident_kv,
        "storage": (
            "app_local_transcript_plus_single_resident_kv"
            if resident_kv
            else "app_local_transcript"
        ),
        "default": "append_none",
        "stateful_default": _SESSION_STATEFUL_DEFAULT_COMMIT,
        "modes": list(_SESSION_COMMIT_MODES),
        "supported_endpoints": ["chat_completions"],
        "supported_streaming": resident_kv,
        "resident_kv_commit": resident_kv,
        **({"resident_kv_capacity": 1} if resident_kv else {}),
        "visible_only_reprefill": False,
        "visible_only_replay": "rerender_app_local_transcript",
        "downgrade_visible_only_on": sorted(_SESSION_UNSAFE_VISIBLE_REASONS),
        "context_overflow_policy": {
            "field": "session.context_overflow_policy",
            "default": "reject",
            "modes": list(_SESSION_CONTEXT_OVERFLOW_POLICIES),
            "aliases": dict(_SESSION_CONTEXT_OVERFLOW_POLICY_ALIASES),
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


def _session_continuation_capability() -> dict[str, Any]:
    return {
        "supported": True,
        "stateful": False,
        "resident_state_reuse": False,
        "single_use": True,
        "ttl_seconds": _CONTINUATION_TTL_SECONDS,
        "scoped_to": ["served_model", "endpoint", "tokenizer", "auth_principal", "session_id"],
        "supported_endpoints": ["completions", "chat_completions"],
        "supported_finishes": ["length"],
        "supported_streaming": False,
        "supported_sampling": "deterministic_buffered_only",
        "ineligible_when": list(_CONTINUATION_INELIGIBLE_WHEN),
        "unsupported_resume_fields": list(_CONTINUATION_UNSUPPORTED_RESUME_FIELDS),
    }


def _session_metadata_capability(max_active: int | None = None) -> dict[str, Any]:
    return {
        "supported": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "includes_transcript": False,
        "transcript_message_copy": "json_deep_copy",
        "max_active": None if max_active is None else int(max_active),
        "list_endpoint": "/v1/hipengine/sessions",
        "delete_endpoint": "/v1/hipengine/sessions/{session_id}",
        "fork_endpoint": "/v1/hipengine/sessions/{session_id}/fork",
        "fork_resident_state_reuse": False,
        "fork_deep_copies_transcript": True,
        "rollback_endpoint": "/v1/hipengine/sessions/{session_id}/rollback",
        "rollback_target": "message_count",
        "rollback_resident_state_reuse": False,
        "rollback_deep_copies_retained_transcript": True,
        "snapshot_schema": _CHAT_SESSION_SNAPSHOT_SCHEMA,
        "snapshot_export_endpoint": "/v1/hipengine/sessions/{session_id}/snapshot",
        "snapshot_restore_endpoint": "/v1/hipengine/sessions/{session_id}/snapshot",
        "snapshot_includes_transcript": True,
        "snapshot_resident_state_reuse": False,
        "snapshot_export_deep_copies_transcript": True,
        "snapshot_includes_tokenizer_metadata": True,
        "snapshot_tokenizer_validation": "when_model_loaded",
    }


def _model_capability_summary(config: ServerConfig | None = None, *, engine: Any | None = None) -> dict[str, Any]:
    return {
        "completions": True,
        "chat_completions": True,
        "streaming": True,
        "tools": True,
        "reasoning_controls": True,
        "structured_outputs": True,
        "continuations": True,
        "sessions": True,
        "grammars": False,
        "speculative_mtp": (
            False
            if config is None
            else bool(_speculative_mtp_capability(config, engine=engine)["serving_route"])
        ),
        "speculative": (
            False
            if config is None
            else bool(_speculative_provider_capability(config, engine=engine)["serving_route"])
        ),
        "tensor_parallel": False,
        "multiple_models": False,
    }


def _routing_response_metadata(
    config: ServerConfig,
    *,
    requested_model: str | None,
    engine: Any | None,
) -> dict[str, Any]:
    return {
        "requested_model": requested_model or config.model_id,
        "served_model": config.model_id,
        "fallback_used": False,
        "policy": "single_model_exact",
        "loaded_model_count": 0 if engine is None else 1,
        "multiple_models": False,
    }


def _routing_failure_metadata(
    config: ServerConfig,
    *,
    requested_model: str | None,
    reason: str,
    engine: Any | None = None,
) -> dict[str, Any]:
    return {
        **_routing_response_metadata(
            config,
            requested_model=requested_model,
            engine=engine,
        ),
        "served_model": None,
        "configured_model": config.model_id,
        "matched": False,
        "reason": str(reason),
    }


def _routing_rejection_metadata(
    config: ServerConfig,
    *,
    requested_model: str | None,
    reason: str,
    engine: Any | None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        **_routing_response_metadata(
            config,
            requested_model=requested_model,
            engine=engine,
        ),
        "matched": True,
        "reason": str(reason),
    }
    if details is not None:
        payload.update(dict(details))
    return payload


def _choice_telemetry_capability() -> dict[str, Any]:
    return {
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
        "diagnostics": "backend_generation_telemetry_when_available",
        "source": "backend_generation_telemetry_when_available",
    }


if ConfigDict is not None:

    class _OpenAIBaseModel(BaseModel):
        model_config = ConfigDict(extra="allow")

else:  # pragma: no cover - Pydantic v1 compatibility

    class _OpenAIBaseModel(BaseModel):
        class Config:
            extra = "allow"


class CompletionRequest(_OpenAIBaseModel):
    model: str | None = None
    prompt: str | list[str] | list[int] | list[list[int]] | None = None
    max_tokens: int | None = Field(default=16, ge=0)
    max_completion_tokens: int | None = Field(default=None, ge=0)
    temperature: float | None = Field(default=0.0, ge=0.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int | None = Field(default=0, ge=0)
    min_p: float | None = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=1.0, gt=0.0)
    presence_penalty: float | None = Field(default=0.0)
    frequency_penalty: float | None = Field(default=0.0)
    logit_bias: dict[str, float] | None = None
    suppress_token_ids: list[int] | None = None
    min_tokens: int | None = Field(default=0, ge=0)
    eos_token_id: int | None = Field(default=None, ge=0)
    n: int | None = Field(default=1, ge=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    timeout_ms: float | None = Field(default=None, gt=0.0)
    stop: str | list[str] | None = None
    seed: int | None = Field(default=None, ge=0)
    echo: bool = False
    logprobs: int | None = Field(default=None, ge=0, le=20)
    ignore_eos: bool = False
    kv_storage: str | None = None
    kv_scale_dtype: str | None = None
    kv_scale_granularity: str | None = None
    response_format: Any | None = None
    guided_json: Any | None = None
    guided_regex: Any | None = None
    guided_choice: Any | None = None
    guided_patch: Any | None = None
    guided_diff: Any | None = None
    continuation_id: Any | None = None
    session: Any | None = None
    speculative_mtp: bool | dict[str, Any] | None = None
    speculative: bool | dict[str, Any] | None = None


class ChatMessage(_OpenAIBaseModel):
    role: str
    content: str | list[Any] | None = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(_OpenAIBaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = Field(default=None, ge=0)
    max_completion_tokens: int | None = Field(default=None, ge=0)
    temperature: float | None = Field(default=0.0, ge=0.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int | None = Field(default=0, ge=0)
    min_p: float | None = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(default=1.0, gt=0.0)
    presence_penalty: float | None = Field(default=0.0)
    frequency_penalty: float | None = Field(default=0.0)
    logit_bias: dict[str, float] | None = None
    suppress_token_ids: list[int] | None = None
    min_tokens: int | None = Field(default=0, ge=0)
    eos_token_id: int | None = Field(default=None, ge=0)
    n: int | None = Field(default=1, ge=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    timeout_ms: float | None = Field(default=None, gt=0.0)
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    invalid_tool_call_error_mode: str | None = None
    reasoning_effort: str | None = None
    enable_thinking: bool | None = None
    max_think_tokens: int | None = Field(default=None, ge=0)
    min_answer_tokens: int | None = Field(default=None, ge=0)
    hard_think_cap: int | None = Field(default=None, ge=0)
    soft_close_window: int | None = Field(default=None, ge=0)
    hard_close_message: str | None = None
    hard_close_sequence: str | None = None
    thinking_token_budget: int | None = Field(default=None, ge=0)
    chat_template_kwargs: dict[str, Any] | None = None
    thinking: str | dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    stop: str | list[str] | None = None
    seed: int | None = Field(default=None, ge=0)
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    ignore_eos: bool = False
    kv_storage: str | None = None
    kv_scale_dtype: str | None = None
    kv_scale_granularity: str | None = None
    response_format: Any | None = None
    guided_json: Any | None = None
    guided_regex: Any | None = None
    guided_choice: Any | None = None
    guided_patch: Any | None = None
    guided_diff: Any | None = None
    continuation_id: Any | None = None
    session: Any | None = None
    speculative_mtp: bool | dict[str, Any] | None = None
    speculative: bool | dict[str, Any] | None = None


class SessionForkRequest(_OpenAIBaseModel):
    id: str


class SessionRollbackRequest(_OpenAIBaseModel):
    message_count: int = Field(..., ge=0)


class TokenizeRequest(_OpenAIBaseModel):
    text: str


class DetokenizeRequest(_OpenAIBaseModel):
    token_ids: list[int]
    skip_special: bool = False


class TokenDiagnosticRequest(_OpenAIBaseModel):
    text: str | None = None
    messages: list[ChatMessage] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    reasoning_effort: str | None = None
    enable_thinking: bool | None = None
    max_think_tokens: int | None = Field(default=None, ge=0)
    min_answer_tokens: int | None = Field(default=None, ge=0)
    hard_think_cap: int | None = Field(default=None, ge=0)
    soft_close_window: int | None = Field(default=None, ge=0)
    hard_close_message: str | None = None
    hard_close_sequence: str | None = None
    thinking_token_budget: int | None = Field(default=None, ge=0)
    chat_template_kwargs: dict[str, Any] | None = None
    thinking: str | dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    guided_json: Any | None = None
    guided_regex: Any | None = None
    guided_choice: Any | None = None
    guided_patch: Any | None = None
    guided_diff: Any | None = None
    session: Any | None = None


class FitContextRequest(TokenDiagnosticRequest):
    max_tokens: int | None = Field(default=None, ge=0)


@dataclass(frozen=True)
class _GeneratedBatch:
    outputs: list[str]
    usage: dict[str, int]
    details: list[GenerationOutput]
    prepared_prompts: tuple[PromptInput, ...] = ()
    scheduler_token_chunks: list[dict[str, Any]] | None = None
    generation_shape: dict[str, Any] | None = None


@dataclass(frozen=True)
class _QueuedBatchResult:
    outputs: list[Any]
    scheduler_token_chunks: list[dict[str, Any]] | None = None
    backend_groups: tuple[dict[str, Any], ...] = ()
    generation_shape: dict[str, Any] | None = None


@dataclass(frozen=True)
class _ChatContextRender:
    prompt: str
    render_request: ChatCompletionRequest
    prefix_messages: tuple[dict[str, Any], ...]
    prepared_prompt: PromptInput | None = None
    thinking: _ThinkingControl | None = None
    session_payload: dict[str, Any] | None = None
    fit_context_extra: dict[str, Any] | None = None
    reset_session_on_commit: bool = False
    commit_base_messages: tuple[dict[str, Any], ...] | None = None


@dataclass
class _ServerMetrics:
    """Additive server counters rendered by the opt-in Prometheus endpoint."""

    request_total: int = 0
    request_completed_total: int = 0
    request_failed_total: int = 0
    request_rejected_total: int = 0
    request_cancelled_total: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    mtp_requests_total: int = 0
    mtp_draft_tokens_accepted_total: int = 0
    mtp_draft_tokens_rejected_total: int = 0

    def record_success(self, usage: Mapping[str, Any]) -> None:
        self.request_total += 1
        self.request_completed_total += 1
        self.prompt_tokens_total += int(usage.get("prompt_tokens", 0))
        self.completion_tokens_total += int(usage.get("completion_tokens", 0))
        completion_details = usage.get("completion_tokens_details")
        if isinstance(completion_details, Mapping) and (
            "accepted_prediction_tokens" in completion_details
            or "rejected_prediction_tokens" in completion_details
        ):
            accepted = int(completion_details.get("accepted_prediction_tokens", 0) or 0)
            rejected = int(completion_details.get("rejected_prediction_tokens", 0) or 0)
            self.mtp_requests_total += 1
            self.mtp_draft_tokens_accepted_total += accepted
            self.mtp_draft_tokens_rejected_total += rejected

    def record_failure(self) -> None:
        self.request_total += 1
        self.request_failed_total += 1

    def record_rejected(self) -> None:
        self.request_total += 1
        self.request_failed_total += 1
        self.request_rejected_total += 1

    def record_cancelled(self) -> None:
        self.request_total += 1
        self.request_failed_total += 1
        self.request_cancelled_total += 1


def _record_openai_error(metrics: _ServerMetrics, exc: OpenAIHTTPError) -> None:
    if exc.code == "engine_busy":
        metrics.record_rejected()
    elif exc.code == "cancelled":
        metrics.record_cancelled()
    else:
        metrics.record_failure()


@dataclass(frozen=True)
class _RequestControl:
    """Server-side cancellation/deadline state for one HTTP generation."""

    deadline_at: float | None = None
    disconnected: Callable[[], Awaitable[bool]] | None = None
    poll_interval_s: float = 0.01
    cancellation_token: GenerationCancellationToken = field(default_factory=GenerationCancellationToken)


_STREAM_DONE = object()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _serving_plan_route_decision(
    plan: Mapping[str, Any],
    *,
    requested_route: str,
    group_rows: int,
    output_horizon_tokens: int,
) -> tuple[str, dict[str, Any]]:
    """Resolve one pre-mutation model-plugin plan against realized grouping."""

    copied = deepcopy(dict(plan))
    key = copied.get("key") if isinstance(copied.get("key"), Mapping) else {}
    request_admitted = bool(copied.get("admitted"))
    admitted = request_admitted
    planned_rows = int(key.get("realized_group_rows", 0) or 0)
    automatic_eligible = bool(copied.get("automatic_eligible"))
    route = str(requested_route)
    reported_requested_route = (
        _SPECULATIVE_MTP_BATCH_ROUTE
        if route == _SPECULATIVE_MTP_K0_ROUTE
        else route
    )
    reason = str(copied.get("reason") or "model_plugin_scope_not_qualified")
    static_payload = copied.get("static_eligibility")
    if not isinstance(static_payload, Mapping):
        static_payload = {
            "state": (
                "speculative_capable" if request_admitted else "permanent_ar"
            ),
            "eligible": request_admitted,
            "reason": reason,
            "max_candidate_count": (
                int(copied.get("selected_candidate_count", 0))
                if request_admitted
                else 0
            ),
            "max_realized_group_rows": planned_rows if request_admitted else 0,
            "automatic_eligible": automatic_eligible if request_admitted else False,
            "strict_fallback_key": str(
                copied.get("strict_fallback_key") or "gguf_target_ar"
            ),
            "evidence_key": (
                copied.get("evidence_key")
                or ("legacy-model-plugin-plan" if request_admitted else None)
            ),
            "evidence_fingerprint": (
                copied.get("evidence_fingerprint")
                or copied.get("plan_fingerprint")
                or ("legacy-plan" if request_admitted else None)
            ),
            "evidence_artifacts": list(copied.get("evidence_artifacts") or ()),
        }
    static_intent_allowed = bool(
        copied.get(
            "static_intent_allowed",
            request_admitted
            and (
                route == _SPECULATIVE_MTP_BATCH_ROUTE
                or (
                    route == _SPECULATIVE_MTP_AUTO_ROUTE
                    and automatic_eligible
                )
            ),
        )
    )
    static_max_rows = int(static_payload.get("max_realized_group_rows", 0) or 0)
    if (
        route == _SPECULATIVE_MTP_AUTO_ROUTE
        and int(group_rows) > static_max_rows
    ):
        # Losing/unqualified automatic widths must select true AR before provider
        # mutation. Explicit requests retain typed intent for functional routes.
        static_intent_allowed = False
    if admitted and planned_rows != int(group_rows):
        admitted = False
        reason = "physical_group_not_qualified"
    if route == _SPECULATIVE_MTP_AUTO_ROUTE:
        if admitted and not automatic_eligible:
            reason = _SPECULATIVE_MTP_AUTO_REJECTION_REASON
        admitted = admitted and automatic_eligible
    elif (
        route == _SPECULATIVE_MTP_K0_ROUTE
        and admitted
        and not automatic_eligible
    ):
        admitted = False
        reason = _SPECULATIVE_MTP_AUTO_REJECTION_REASON
    selected_route = (
        _SPECULATIVE_MTP_BATCH_ROUTE
        if admitted
        and route in {
            _SPECULATIVE_MTP_BATCH_ROUTE,
            _SPECULATIVE_MTP_AUTO_ROUTE,
        }
        else _SPECULATIVE_MTP_DEFAULT_ROUTE
    )
    artifacts = copied.get("evidence_artifacts")
    evidence = (
        artifacts[0]
        if isinstance(artifacts, list) and artifacts
        else "model_plugin_speculative_mtp_serving_plan"
    )
    k0_class = (
        "not_k0"
        if selected_route == _SPECULATIVE_MTP_BATCH_ROUTE
        else "transitional_k0"
        if static_intent_allowed
        else "pure_k0"
    )
    return selected_route, {
        "requested_route": reported_requested_route,
        "selected_route": selected_route,
        "reason": reason,
        "k0_class": k0_class,
        "static_eligibility": deepcopy(dict(static_payload)),
        "static_eligibility_fingerprint": SpeculativeMTPStaticEligibility.from_mapping(
            static_payload
        ).fingerprint,
        "static_intent_allowed": static_intent_allowed,
        "policy_cell": copied.get("evidence_key"),
        "selected_candidate_count": (
            int(copied.get("selected_candidate_count", 0))
            if selected_route == _SPECULATIVE_MTP_BATCH_ROUTE
            else 0
        ),
        "policy_reason": str(copied.get("reason") or reason),
        "policy_fingerprint": copied.get("plan_fingerprint"),
        "realized_group_rows": int(group_rows),
        "output_horizon_tokens": int(output_horizon_tokens),
        "exact_default_required": True,
        "automatic_eligible": automatic_eligible,
        "evidence": evidence,
    }


def _static_eligibility_from_route_decision(
    route_decision: Mapping[str, Any] | None,
) -> SpeculativeMTPStaticEligibility | None:
    if not isinstance(route_decision, Mapping) or not bool(
        route_decision.get("static_intent_allowed")
    ):
        return None
    payload = route_decision.get("static_eligibility")
    if not isinstance(payload, Mapping):
        return None
    eligibility = SpeculativeMTPStaticEligibility.from_mapping(payload)
    return eligibility if eligibility.eligible else None


def _sampling_for_realized_generation_route(
    sampling: SamplingParams,
    route_decision: Mapping[str, Any] | None,
) -> SamplingParams:
    """Attach typed static provider intent without selecting future C_due/K."""

    eligibility = _static_eligibility_from_route_decision(route_decision)
    if sampling.speculative_mtp_static_eligibility == eligibility:
        return sampling
    return replace(
        sampling,
        speculative_mtp_static_eligibility=eligibility,
    )


def _execution_route_for_static_intent(
    selected_route: str,
    route_decision: Mapping[str, Any] | None,
) -> str:
    """Enter Generation-2 for eligible intent even when this group predicts K0."""

    if _static_eligibility_from_route_decision(route_decision) is not None:
        return _SPECULATIVE_MTP_BATCH_ROUTE
    return str(selected_route)


def _resolve_realized_generation_route(
    requested_route: str,
    *,
    group_rows: int,
    sampling: SamplingParams,
    precomputed_decision: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Resolve automatic routing only after the batcher's group is known.

    Exact staged MTP has one current fixed C1/K2 performance premise, but
    automatic requests remain on the default route until its complete product
    packet passes. Losing and unqualified widths stay pre-mutation K0.
    """

    route = str(requested_route)
    if (
        isinstance(precomputed_decision, Mapping)
        and precomputed_decision.get("plan_fingerprint") is not None
    ):
        return _serving_plan_route_decision(
            precomputed_decision,
            requested_route=route,
            group_rows=int(group_rows),
            output_horizon_tokens=int(sampling.max_tokens),
        )
    if route == _SPECULATIVE_MTP_K0_ROUTE:
        blockers = tuple(speculative_mtp_sampling_blockers(sampling))
        reason = (
            "unsupported_sampling_k0"
            if blockers
            else "model_plugin_scope_not_qualified"
        )
        return (
            _SPECULATIVE_MTP_DEFAULT_ROUTE,
            {
                "requested_route": _SPECULATIVE_MTP_BATCH_ROUTE,
                "selected_route": _SPECULATIVE_MTP_DEFAULT_ROUTE,
                "reason": reason,
                "policy_cell": "compatibility-pre-mutation-k0",
                "selected_candidate_count": 0,
                "policy_reason": (
                    "unsupported_sampling"
                    if blockers
                    else "no_automatic_model_plugin_evidence"
                ),
                "sampling_blockers": list(blockers),
                "realized_group_rows": int(group_rows),
                "output_horizon_tokens": int(sampling.max_tokens),
                "exact_default_required": True,
                "evidence": "docs/SPECDEC2.md#12-s6--gfx1151-product-closure",
            },
        )
    if route != _SPECULATIVE_MTP_AUTO_ROUTE:
        return route, None
    rows = int(group_rows)
    if rows < 1:
        raise ValueError("automatic generation route requires a positive realized group")
    decision = select_offline_speculative_depth(
        DEFAULT_AUTO_DEPTH_POLICY,
        concurrency=rows,
        output_horizon_tokens=int(sampling.max_tokens),
    )
    if decision.selected_k != 0:
        return (
            _SPECULATIVE_MTP_BATCH_ROUTE,
            {
                "requested_route": _SPECULATIVE_MTP_AUTO_ROUTE,
                "selected_route": _SPECULATIVE_MTP_BATCH_ROUTE,
                "reason": decision.reason,
                "policy_cell": decision.cell_key,
                "selected_candidate_count": decision.selected_k,
                "policy_fingerprint": decision.policy_fingerprint,
                "realized_group_rows": rows,
                "output_horizon_tokens": int(sampling.max_tokens),
                "exact_default_required": True,
                "evidence": decision.evidence,
            },
        )
    return (
        _SPECULATIVE_MTP_DEFAULT_ROUTE,
        {
            "requested_route": _SPECULATIVE_MTP_AUTO_ROUTE,
            "selected_route": _SPECULATIVE_MTP_DEFAULT_ROUTE,
            "reason": _SPECULATIVE_MTP_AUTO_REJECTION_REASON,
            "policy_cell": decision.cell_key,
            "selected_candidate_count": 0,
            "policy_reason": decision.reason,
            "policy_fingerprint": decision.policy_fingerprint,
            "realized_group_rows": rows,
            "output_horizon_tokens": int(sampling.max_tokens),
            "exact_default_required": True,
            "evidence": decision.evidence,
        },
    )


def _next_stream_item(iterator: Iterator[Any]) -> object:
    try:
        return next(iterator)
    except StopIteration:
        return _STREAM_DONE


class _DebugPayloadMiddleware:
    """ASGI middleware that logs full HTTP payloads when explicitly enabled."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", ""))
        path = str(scope.get("path", ""))
        query = scope.get("query_string", b"")
        if isinstance(query, bytes) and query:
            target = f"{path}?{query.decode('utf-8', 'replace')}"
        else:
            target = path
        request_chunks: list[bytes] = []
        response_chunks: list[bytes] = []
        request_logged = False
        response_status: int | None = None

        def log_request_once() -> None:
            nonlocal request_logged
            if request_logged:
                return
            request_logged = True
            _LOGGER.info(
                "DEBUG_PAYLOAD REQUEST %s %s body=%s",
                method,
                target,
                _debug_payload_text(request_chunks),
            )

        async def receive_wrapper() -> dict[str, Any]:
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                if body:
                    request_chunks.append(bytes(body))
                if not message.get("more_body", False):
                    log_request_once()
            return message

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal response_status
            if message.get("type") == "http.response.start":
                response_status = int(message.get("status", 0))
            elif message.get("type") == "http.response.body":
                body = message.get("body", b"")
                if body:
                    response_chunks.append(bytes(body))
                await send(message)
                if not message.get("more_body", False):
                    log_request_once()
                    _LOGGER.info(
                        "DEBUG_PAYLOAD RESPONSE %s %s status=%s body=%s",
                        method,
                        target,
                        "unknown" if response_status is None else str(response_status),
                        _debug_payload_text(response_chunks),
                    )
                return
            await send(message)

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        except Exception:
            log_request_once()
            _LOGGER.exception("DEBUG_PAYLOAD RESPONSE %s %s raised before completion", method, target)
            raise


def _debug_payload_text(chunks: Sequence[bytes]) -> str:
    data = b"".join(chunks)
    if not data:
        return "<empty>"
    text = data.decode("utf-8", "replace")
    try:
        parsed = json.loads(text)
    except Exception:
        return text
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def _request_target(request: Request) -> str:
    query = request.url.query
    return request.url.path if not query else f"{request.url.path}?{query}"


def _log_request_failure(
    request: Request,
    *,
    status_code: int,
    code: str | None,
    param: str | None,
    message: str,
) -> None:
    logger = _LOGGER.error if int(status_code) >= 500 else _LOGGER.warning
    logger(
        "REQUEST_FAILED: %s %s status=%d code=%s param=%s message=%s",
        request.method,
        _request_target(request),
        int(status_code),
        code,
        param,
        message,
    )


def _log_stream_failure(
    endpoint: str,
    *,
    status_code: int,
    code: str | None,
    param: str | None,
    message: str,
) -> None:
    logger = _LOGGER.error if int(status_code) >= 500 else _LOGGER.warning
    logger(
        "REQUEST_FAILED: %s status=%d code=%s param=%s message=%s",
        endpoint,
        int(status_code),
        code,
        param,
        message,
    )


class _MTPCircuitBreaker:
    """Restart-scoped protection for repeated MTP backend/runtime failures."""

    def __init__(self, *, threshold: int = 3) -> None:
        if int(threshold) <= 0:
            raise ValueError("MTP circuit-breaker threshold must be positive")
        self.threshold = int(threshold)
        self._failures: dict[tuple[str, str, str, str], int] = {}
        self._opened: set[tuple[str, str, str, str]] = set()
        self._transitions: list[dict[str, Any]] = []
        self._operator_disabled = False
        self._operator_reason: str | None = None

    def scope(
        self,
        engine: Any,
        prompts: Sequence[PromptInput],
    ) -> tuple[str, str, str, str]:
        model = str(getattr(engine, "model_id", None) or getattr(engine, "model_path", "unknown"))
        backend = str(
            getattr(engine, "_resolved_backend", None)
            or getattr(engine, "backend", "unknown")
        )
        profile = str(getattr(engine, "resolved_execution_profile", None) or "strict")
        lengths = [
            int(length)
            for length in (_prepared_context_tokens(prompt) for prompt in prompts)
            if length is not None and int(length) > 0
        ]
        context_bucket = (
            "unknown"
            if not lengths
            else str(((max(lengths) + 255) // 256) * 256)
        )
        return model, backend, profile, context_bucket

    @property
    def operator_disabled(self) -> bool:
        return bool(self._operator_disabled)

    def disable_new_mtp(self, *, reason: str = "mtp_operator_rollback") -> None:
        if not self._operator_disabled:
            self._operator_disabled = True
            self._operator_reason = str(reason)
            self._transitions.append(
                {"scope": ["all"], "state": "operator_disabled", "reason": str(reason)}
            )

    def is_open(self, scope: tuple[str, str, str, str]) -> bool:
        return scope in self._opened

    def record_failure(
        self,
        scope: tuple[str, str, str, str],
        exc: BaseException,
    ) -> None:
        failures = self._failures.get(scope, 0) + 1
        self._failures[scope] = failures
        if failures >= self.threshold and scope not in self._opened:
            self._opened.add(scope)
            self._transitions.append(
                {
                    "scope": list(scope),
                    "state": "open",
                    "failures": failures,
                    "error_type": type(exc).__name__,
                }
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": (
                "operator_disabled"
                if self._operator_disabled
                else ("open" if self._opened else "closed")
            ),
            "operator_disabled": bool(self._operator_disabled),
            "operator_reason": self._operator_reason,
            "threshold": self.threshold,
            "open_scopes": [list(scope) for scope in sorted(self._opened)],
            "failures": [
                {"scope": list(scope), "count": count}
                for scope, count in sorted(self._failures.items())
            ],
            "transitions": deepcopy(self._transitions),
            "reset_policy": "server_restart",
        }

    def attach(self, engine: Any) -> None:
        setattr(engine, "mtp_circuit_breaker_state", self.snapshot())


@dataclass
class _QueuedGeneration:
    prompts: tuple[PromptInput, ...]
    sampling: SamplingParams
    future: asyncio.Future[Any] | None = None
    stream_queue: asyncio.Queue[object] | None = None
    detailed: bool = False
    include_batch_metadata: bool = False
    route: str = _SPECULATIVE_MTP_DEFAULT_ROUTE
    route_decision: dict[str, Any] | None = None
    cancelled: bool = False
    finished: bool = False
    producer_task: asyncio.Task[None] | None = None


class _CompositeGenerationCancellationToken:
    """Cancellation view that trips only when every grouped request is gone.

    Coalesced requests share one backend call, but one disconnected client must
    not abort unrelated rows in that call.  Until the model runners consume
    row-scoped cancellation tokens and can reclaim individual slots, a
    cancelled member is allowed to finish as discarded work; the shared call
    is cancelled once no live member remains.
    """

    def __init__(self, tokens: Sequence[GenerationCancellationToken]) -> None:
        self._tokens = tuple(tokens)

    @property
    def cancelled(self) -> bool:
        return bool(self._tokens) and all(bool(token.cancelled) for token in self._tokens)

    @property
    def finish_details(self) -> FinishDetails:
        for token in self._tokens:
            if token.cancelled:
                return token.finish_details
        return FinishDetails(reason="cancelled", cancelled=True)

    def cancel(self, finish_details: FinishDetails | None = None) -> None:
        for token in self._tokens:
            token.cancel(finish_details)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise GenerationCancelled(self.finish_details)


class _GenerationBatcher:
    """Coalesce compatible HTTP generations into prompt-list calls."""

    def __init__(
        self,
        *,
        engine_factory: Callable[[], Any],
        batch_window_seconds: float,
        max_queue_size: int | None = None,
        max_active_requests: int | None = None,
        route_max_active_requests: Mapping[str, int] | None = None,
        retry_after_seconds: int = 1,
        stream_queue_max_chunks: int = 16,
        mtp_circuit_breaker: _MTPCircuitBreaker | None = None,
    ) -> None:
        self._engine_factory = engine_factory
        self._batch_window_seconds = max(0.0, float(batch_window_seconds))
        self._max_queue_size = None if max_queue_size is None else int(max_queue_size)
        if self._max_queue_size is not None and self._max_queue_size < 1:
            raise ValueError("max_queue_size must be positive when set")
        self._max_active_requests = None if max_active_requests is None else int(max_active_requests)
        if self._max_active_requests is not None and self._max_active_requests < 1:
            raise ValueError("max_active_requests must be positive when set")
        self._route_max_active_requests: dict[str, int] = {}
        for route, limit in dict(route_max_active_requests or {}).items():
            route_limit = int(limit)
            if route_limit < 1:
                raise ValueError("route max_active_requests limits must be positive")
            self._route_max_active_requests[str(route)] = route_limit
        self._retry_after_seconds = max(1, int(retry_after_seconds))
        self._stream_queue_max_chunks = int(stream_queue_max_chunks)
        self._mtp_circuit_breaker = mtp_circuit_breaker
        if self._stream_queue_max_chunks < 2:
            raise ValueError("stream_queue_max_chunks must be at least 2")
        self._queue: deque[_QueuedGeneration] = deque()
        self._worker: asyncio.Task[None] | None = None
        self._stream_tasks: set[asyncio.Task[None]] = set()
        self._independent_tasks: set[asyncio.Task[None]] = set()
        self._stream_items: dict[int, _QueuedGeneration] = {}
        self._active_items: dict[int, _QueuedGeneration] = {}
        self._active_requests = 0
        self._accepting = True

    def queue_depth(self) -> int:
        return len(self._queue)

    def max_queue_size(self) -> int | None:
        return self._max_queue_size

    def active_requests(self) -> int:
        return self._active_requests

    def max_active_requests(self) -> int | None:
        return self._max_active_requests

    def active(self) -> bool:
        worker_active = self._worker is not None and not self._worker.done()
        producers = (*self._stream_tasks, *self._independent_tasks)
        return worker_active or any(not task.done() for task in producers)

    def stream_queue_max_chunks(self) -> int:
        return self._stream_queue_max_chunks

    def stream_queue_depths(self) -> tuple[int, ...]:
        items = [*self._queue, *self._stream_items.values()]
        queues = {
            id(item.stream_queue): item.stream_queue
            for item in items
            if item.stream_queue is not None
        }
        return tuple(queue.qsize() for queue in queues.values())

    def _route_request_cap(self, route: str) -> int | None:
        route_name = str(route)
        limit = self._max_active_requests
        route_limit = self._route_max_active_requests.get(route_name)
        if route_name == _SPECULATIVE_MTP_BATCH_ROUTE:
            # The physical MTP width is capability-owned: a resident adapter
            # publishes its certified bound; engines without one retain the
            # registered GGUF fallback constant.
            registered_limit = getattr(
                self._engine_factory(),
                "server_mtp_batch_max_active_requests",
                None,
            )
            if registered_limit is not None:
                route_limit = int(registered_limit)
                if route_limit < 1:
                    raise ValueError(
                        "server_mtp_batch_max_active_requests must be positive"
                    )
        if route_name in {
            _SPECULATIVE_MTP_DEFAULT_ROUTE,
            _SPECULATIVE_MTP_AUTO_ROUTE,
        }:
            # Automatic requests must reach their normal AR group width before
            # static evidence decides MTP or pure K0. Explicit MTP stays bounded
            # by its registered physical owner.
            engine = self._engine_factory()
            independent = _engine_supports_independent_generation(engine)
            if independent:
                route_limit = None
            registered_limit = (
                None
                if independent
                else getattr(engine, "server_plain_ar_max_active_requests", None)
            )
            if registered_limit is not None:
                route_limit = int(registered_limit)
                if route_limit < 1:
                    raise ValueError(
                        "server_plain_ar_max_active_requests must be positive"
                    )
        if route_limit is not None:
            limit = route_limit if limit is None else min(limit, route_limit)
        return limit

    def _group_has_capacity(self, group: Sequence[_QueuedGeneration]) -> bool:
        limit = self._route_request_cap(group[0].route) if group else self._max_active_requests
        return limit is None or len(group) < limit

    def _group_key(self, item: _QueuedGeneration) -> tuple[str, tuple[Any, ...]]:
        route = str(item.route)
        decision = item.route_decision or {}
        return (
            route,
            (
                decision.get("plan_fingerprint"),
                _sampling_key(
                    item.sampling,
                    include_cancellation_token=False,
                ),
            ),
        )

    @staticmethod
    def _sampling_for_group(group: Sequence[_QueuedGeneration]) -> SamplingParams:
        sampling = group[0].sampling
        if len(group) <= 1:
            return sampling
        tokens = [
            item.sampling.cancellation_token
            for item in group
            if item.sampling.cancellation_token is not None
        ]
        unique_tokens = tuple(dict.fromkeys(tokens))
        if len(unique_tokens) <= 1:
            return sampling
        return replace(
            sampling,
            cancellation_token=_CompositeGenerationCancellationToken(unique_tokens),
        )

    def _raise_if_full(self, *, error_extra: Mapping[str, Any] | None = None) -> None:
        if not self._accepting:
            raise OpenAIHTTPError(
                429,
                "generation engine is draining",
                error_type="rate_limit_error",
                code="engine_busy",
                extra=error_extra,
                headers={"Retry-After": str(self._retry_after_seconds)},
            )
        if self._max_queue_size is None:
            return
        if len(self._queue) < self._max_queue_size:
            return
        raise OpenAIHTTPError(
            429,
            "generation queue is full",
            error_type="rate_limit_error",
            code="engine_busy",
            extra=error_extra,
            headers={"Retry-After": str(self._retry_after_seconds)},
        )

    async def submit(
        self,
        prompts: Sequence[PromptInput],
        sampling: SamplingParams,
        *,
        detailed: bool = False,
        include_batch_metadata: bool = False,
        route: str = _SPECULATIVE_MTP_DEFAULT_ROUTE,
        route_decision: Mapping[str, Any] | None = None,
        error_extra: Mapping[str, Any] | None = None,
    ) -> list[Any] | _QueuedBatchResult:
        prompt_tuple = tuple(normalize_prompt_input(prompt) for prompt in prompts)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._raise_if_full(error_extra=error_extra)
        self._queue.append(
            _QueuedGeneration(
                prompts=prompt_tuple,
                sampling=sampling,
                future=future,
                detailed=bool(detailed),
                include_batch_metadata=bool(include_batch_metadata),
                route=str(route),
                route_decision=(
                    None if route_decision is None else deepcopy(dict(route_decision))
                ),
            )
        )
        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._run())
        try:
            return await future
        except GenerationAdmissionRejected as exc:
            raise _generation_admission_http_error(
                exc,
                retry_after_seconds=self._retry_after_seconds,
                error_extra=error_extra,
            ) from exc

    async def stream(
        self,
        prompts: Sequence[PromptInput],
        sampling: SamplingParams,
        *,
        route: str = _SPECULATIVE_MTP_DEFAULT_ROUTE,
        route_decision: Mapping[str, Any] | None = None,
        error_extra: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[GenerationStreamChunk]:
        """Yield generated stream chunks through a per-request queue owned by the batcher."""

        prompt_tuple = tuple(normalize_prompt_input(prompt) for prompt in prompts)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[object] = asyncio.Queue(maxsize=self._stream_queue_max_chunks)
        self._raise_if_full(error_extra=error_extra)
        item = _QueuedGeneration(
            prompts=prompt_tuple,
            sampling=sampling,
            stream_queue=queue,
            route=str(route),
            route_decision=(
                None if route_decision is None else deepcopy(dict(route_decision))
            ),
        )
        self._queue.append(item)
        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._run())
        try:
            while True:
                event = await queue.get()
                if event is _STREAM_DONE:
                    break
                if isinstance(event, GenerationAdmissionRejected):
                    raise _generation_admission_http_error(
                        event,
                        retry_after_seconds=self._retry_after_seconds,
                        error_extra=error_extra,
                    ) from event
                if isinstance(event, BaseException):
                    raise event
                yield event
        finally:
            if not item.finished:
                _cancel_queued_generation(item, reason="disconnect")
                # The client no longer consumes this queue.  Release a producer
                # blocked on backpressure, then let the backend observe its
                # cooperative cancellation token before closing its iterator.
                # Cancelling the producer task here can race a synchronous
                # generator still executing in a worker thread and raise
                # ``aclose(): asynchronous generator is already running``.
                while not queue.empty():
                    with suppress(asyncio.QueueEmpty):
                        queue.get_nowait()
                producer = item.producer_task
                if producer is not None and producer is not asyncio.current_task() and not producer.done():
                    with suppress(asyncio.CancelledError):
                        await producer

    async def _run(self) -> None:
        try:
            if self._batch_window_seconds > 0.0:
                # Arrival-aware window: a solo submission on an idle engine
                # dispatches after one short slice instead of paying the full
                # window (measured 2026-08-29: the fixed 50 ms window was the
                # largest single slice of C1 request latency, and no second
                # submission can join an idle solo batch). Two or more queued
                # submissions, or a busy engine, still honor the full window
                # so concurrent-arrival batching behavior is unchanged.
                solo_slice = min(
                    _SOLO_DISPATCH_SLICE_SECONDS, self._batch_window_seconds
                )
                await asyncio.sleep(solo_slice)
                if not (
                    len(self._queue) <= 1 and self._active_requests == 0
                ):
                    remaining = self._batch_window_seconds - solo_slice
                    if remaining > 0.0:
                        await asyncio.sleep(remaining)
            while self._queue:
                first = self._queue.popleft()
                if _queued_generation_cancelled(first):
                    continue
                engine = self._engine_factory()
                if first.stream_queue is not None:
                    if _engine_supports_controlled_streaming(engine):
                        active_limit = self._route_request_cap(first.route)
                        if (
                            active_limit is not None
                            and self._active_requests >= active_limit
                        ):
                            self._queue.appendleft(first)
                            active_streams = tuple(
                                task for task in self._stream_tasks if not task.done()
                            )
                            if not active_streams:  # pragma: no cover - defensive invariant
                                raise RuntimeError(
                                    "controlled stream capacity is full without an active producer"
                                )
                            await asyncio.wait(
                                active_streams,
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                            continue
                        self._launch_controlled_stream(first, engine)
                        continue
                elif (
                    str(first.route) == _SPECULATIVE_MTP_DEFAULT_ROUTE
                    and _engine_supports_independent_generation(engine)
                ):
                    active_limit = self._route_request_cap(first.route)
                    child_rows = len(first.prompts)
                    if active_limit is not None and child_rows > active_limit:
                        _finish_queued_generation(
                            first,
                            exception=GenerationAdmissionRejected(
                                "independent child group exceeds the resident request limit",
                                resource="resident_child_limit",
                                requested_units=child_rows,
                                current_units=self._active_requests,
                                capacity_units=active_limit,
                            ),
                        )
                        continue
                    if active_limit is not None and self._active_requests + child_rows > active_limit:
                        self._queue.appendleft(first)
                        active_tasks = tuple(
                            task
                            for task in (*self._stream_tasks, *self._independent_tasks)
                            if not task.done()
                        )
                        if not active_tasks:  # pragma: no cover - defensive invariant
                            raise RuntimeError(
                                "independent generation capacity is full without an active producer"
                            )
                        await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                        continue
                    self._launch_independent_generation(first)
                    continue
                key = self._group_key(first)
                group = [first]
                deferred: deque[_QueuedGeneration] = deque()
                while self._queue:
                    item = self._queue.popleft()
                    if _queued_generation_cancelled(item):
                        continue
                    if self._group_key(item) == key and self._group_has_capacity(group):
                        group.append(item)
                    else:
                        deferred.append(item)
                self._queue.extendleft(reversed(deferred))
                await self._run_group(group)
                if self._queue and self._batch_window_seconds > 0.0:
                    await asyncio.sleep(self._batch_window_seconds)
        finally:
            self._worker = None
            if self._queue:
                self._worker = asyncio.create_task(self._run())

    def _launch_controlled_stream(self, item: _QueuedGeneration, engine: Any) -> None:
        self._active_requests += 1
        self._active_items[id(item)] = item
        task = asyncio.create_task(self._run_controlled_stream(item, engine))
        item.producer_task = task
        self._stream_tasks.add(task)
        self._stream_items[id(item)] = item

        def finished(done: asyncio.Task[None]) -> None:
            self._release_controlled_stream(item)
            self._stream_tasks.discard(done)
            self._stream_items.pop(id(item), None)

        task.add_done_callback(finished)

    def _release_controlled_stream(self, item: _QueuedGeneration) -> None:
        if self._active_items.pop(id(item), None) is not None:
            self._active_requests = max(0, self._active_requests - 1)

    def _launch_independent_generation(self, item: _QueuedGeneration) -> None:
        child_rows = len(item.prompts)
        self._active_requests += child_rows
        self._active_items[id(item)] = item
        task = asyncio.create_task(self._run_group((item,), manage_active=False))
        item.producer_task = task
        self._independent_tasks.add(task)

        def finished(done: asyncio.Task[None]) -> None:
            if self._active_items.pop(id(item), None) is not None:
                self._active_requests = max(0, self._active_requests - child_rows)
            self._independent_tasks.discard(done)

        task.add_done_callback(finished)

    async def _run_controlled_stream(self, item: _QueuedGeneration, engine: Any) -> None:
        try:
            if len(item.prompts) == 1:
                await self._stream_single(item, engine=engine)
            elif _engine_supports_stream_many(engine):
                await self._stream_many(item, engine)
            else:
                raise NotImplementedError("controlled multi-row streaming is not supported by this generator")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await _finish_stream_queued_generation(item, exception=exc)
        finally:
            token = item.sampling.cancellation_token
            if (
                token is not None
                and bool(getattr(token, "cancel_requested", False))
                and not bool(getattr(token, "cancelled", False))
            ):
                drain_cancellations = getattr(engine, "drain_generation_cancellations", None)
                if callable(drain_cancellations):
                    await run_in_threadpool(drain_cancellations)
            self._release_controlled_stream(item)

    async def _run_group(
        self,
        group: Sequence[_QueuedGeneration],
        *,
        manage_active: bool = True,
    ) -> None:
        if not group:
            return
        if manage_active:
            self._active_requests += len(group)
            for item in group:
                self._active_items[id(item)] = item
        try:
            if len(group) == 1 and group[0].stream_queue is not None:
                if len(group[0].prompts) == 1:
                    await self._stream_single(group[0])
                    return
                engine = self._engine_factory()
                if _engine_supports_stream_many(engine):
                    await self._stream_many(group[0], engine)
                    return
            prompts: list[PromptInput] = []
            slices: list[tuple[_QueuedGeneration, int, int]] = []
            for item in group:
                start = len(prompts)
                prompts.extend(item.prompts)
                slices.append((item, start, len(prompts)))
            queue_group_id = f"queue-{uuid.uuid4().hex}"
            requested_route = str(group[0].route)
            group_sampling = self._sampling_for_group(group)
            route, route_decision = _resolve_realized_generation_route(
                requested_route,
                group_rows=len(prompts),
                sampling=group_sampling,
                precomputed_decision=group[0].route_decision,
            )
            breaker = self._mtp_circuit_breaker
            if (
                route == _SPECULATIVE_MTP_BATCH_ROUTE
                and breaker is not None
                and breaker.operator_disabled
            ):
                route = _SPECULATIVE_MTP_DEFAULT_ROUTE
                route_decision = {
                    "requested_route": requested_route,
                    "selected_route": route,
                    "reason": "mtp_operator_rollback",
                    "realized_group_rows": len(prompts),
                    "output_horizon_tokens": int(group_sampling.max_tokens),
                    "exact_default_required": True,
                    "evidence": "operator_runtime_rollback",
                }
            route_cap = self._route_request_cap(requested_route)
            group_sampling = _sampling_for_realized_generation_route(
                group_sampling,
                route_decision,
            )
            execution_route = _execution_route_for_static_intent(
                route,
                route_decision,
            )
            try:
                batch_result = await self._generate_prompts(
                    tuple(prompts),
                    group_sampling,
                    route=execution_route,
                )
            except Exception as exc:
                for item in group:
                    _finish_queued_generation(item, exception=exc)
                return
            outputs = batch_result.outputs
            for item_index, (item, start, end) in enumerate(slices):
                item_outputs: Sequence[Any] = outputs[start:end]
                if not item.detailed:
                    item_outputs = [_coerce_generation_output(output).text for output in item_outputs]
                if item.include_batch_metadata:
                    scheduler_token_chunks = (
                        _copy_scheduler_token_chunks(batch_result.scheduler_token_chunks)
                        if len(group) == 1
                        else None
                    )
                    _finish_queued_generation(
                        item,
                        result=_QueuedBatchResult(
                            outputs=list(item_outputs),
                            scheduler_token_chunks=scheduler_token_chunks,
                            backend_groups=batch_result.backend_groups,
                            generation_shape=_queue_generation_shape(
                                route=route,
                                route_cap=route_cap,
                                queue_group_id=queue_group_id,
                                queue_request_count=len(group),
                                queue_prompt_rows=len(prompts),
                                item_index=item_index,
                                item_prompt_offset=start,
                                item_prompt_rows=end - start,
                                backend_groups=batch_result.backend_groups,
                                route_decision=route_decision,
                            ),
                        ),
                    )
                else:
                    _finish_queued_generation(item, outputs=item_outputs)
        finally:
            if manage_active:
                for item in group:
                    self._active_items.pop(id(item), None)
                self._active_requests = max(0, self._active_requests - len(group))

    async def _generate_prompts(
        self,
        prompts: tuple[PromptInput, ...],
        sampling: SamplingParams,
        *,
        route: str = _SPECULATIVE_MTP_DEFAULT_ROUTE,
    ) -> _QueuedBatchResult:
        engine = self._engine_factory()
        if str(route) == _SPECULATIVE_MTP_BATCH_ROUTE:
            breaker = self._mtp_circuit_breaker
            scope = None if breaker is None else breaker.scope(engine, prompts)
            if breaker is not None and scope is not None and breaker.is_open(scope):
                breaker.attach(engine)
                raise OpenAIHTTPError(
                    503,
                    "speculative_mtp circuit breaker is open for this scope",
                    error_type="service_unavailable",
                    code="mtp_circuit_breaker_open",
                    param="speculative_mtp",
                    extra={
                        "hipengine": {
                            "speculative_mtp": {
                                "decision_reason": "mtp_circuit_breaker_open",
                                "circuit_breaker": breaker.snapshot(),
                            }
                        }
                    },
                )
            # The raw-argmax MTP verifier is exact only for the greedy fast
            # path.  Under the hint thinking policy, host-sampler thinking
            # enforcement is relaxed (prompt hints stay) so thinking requests
            # can still use the MTP route.
            mtp_sampling = relax_thinking_budget_for_mtp(sampling)
            try:
                raw_outputs = await _generate_speculative_mtp_detailed(
                    engine,
                    prompts,
                    mtp_sampling,
                )
            except (GenerationCancelled, GenerationDeadlineExceeded, OpenAIHTTPError):
                raise
            except Exception as exc:
                if breaker is not None and scope is not None:
                    breaker.record_failure(scope, exc)
                    breaker.attach(engine)
                raise
            else:
                if breaker is not None:
                    breaker.attach(engine)
        elif str(route) == _SPECULATIVE_PROVIDER_ROUTE:
            raw_outputs = await _generate_speculative_detailed(engine, prompts, sampling)
        else:
            raw_outputs = await _generate_detailed(engine, prompts, sampling)
        outputs = list(raw_outputs)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"generator returned {len(outputs)} outputs for {len(prompts)} prompts"
            )
        return _QueuedBatchResult(
            outputs=outputs,
            scheduler_token_chunks=_backend_scheduler_token_chunks(engine),
            backend_groups=(
                _backend_generation_group_shape(
                    engine,
                    input_rows=len(prompts),
                ),
            ),
        )

    async def _stream_single(self, item: _QueuedGeneration, *, engine: Any | None = None) -> None:
        assert item.stream_queue is not None
        try:
            async for chunk in _stream_engine_text(
                self._engine_factory() if engine is None else engine,
                item.prompts[0],
                _sampling_for_realized_generation_route(
                    item.sampling,
                    item.route_decision,
                ),
                route=item.route,
            ):
                if _queued_generation_cancelled(item):
                    raise GenerationCancelled(_queued_generation_finish_details(item))
                await item.stream_queue.put(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await _finish_stream_queued_generation(item, exception=exc)
            return
        await _finish_stream_queued_generation(item)

    async def _stream_many(self, item: _QueuedGeneration, engine: Any) -> None:
        assert item.stream_queue is not None
        try:
            async for chunk in _stream_engine_many(engine, item.prompts, item.sampling):
                if _queued_generation_cancelled(item):
                    raise GenerationCancelled(_queued_generation_finish_details(item))
                await item.stream_queue.put(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await _finish_stream_queued_generation(item, exception=exc)
            return
        await _finish_stream_queued_generation(item)

    async def shutdown(self, *, grace_seconds: float = 5.0) -> dict[str, Any]:
        """Stop admission, drain active work, then cooperatively force-cancel."""

        grace = max(0.0, float(grace_seconds))
        self._accepting = False
        forced = False
        try:
            await asyncio.wait_for(self._wait_until_idle(), timeout=grace)
        except asyncio.TimeoutError:
            forced = True
            finish_details = FinishDetails(reason="cancelled", cancelled=True)
            error = GenerationCancelled(finish_details)
            queued = tuple(self._queue)
            active = tuple(self._active_items.values())
            self._queue.clear()
            for item in queued:
                _cancel_queued_generation(item, reason="cancel")
                _finish_queued_generation(item, exception=error)
            for item in active:
                _cancel_queued_generation(item, reason="cancel")
                _finish_queued_generation(item, exception=error)
            # Do not cancel the asyncio task awaiting run_in_threadpool: Python
            # cannot cancel its model thread, and engine.close() would race live
            # GPU ownership. Request tokens now force the dense MTP loop to stop
            # at its next safe post-cycle boundary. Wait until that owned work
            # retires before allowing model teardown.
            await self._wait_until_idle()
        return {
            "forced": forced,
            "grace_seconds": grace,
            "active_requests": self._active_requests,
            "queued_requests": len(self._queue),
        }

    async def _wait_until_idle(self) -> None:
        # Poll rather than awaiting producer tasks directly: wait_for() must not
        # cancel a model thread before the force path trips its cooperative token.
        while self._queue or self.active() or self._active_requests:
            await asyncio.sleep(0.001)


def _queued_generation_finish_details(item: _QueuedGeneration) -> FinishDetails:
    token = item.sampling.cancellation_token
    if token is not None and bool(getattr(token, "cancelled", False)):
        return FinishDetails.from_value(getattr(token, "finish_details", None))
    return FinishDetails(reason="cancelled", cancelled=True)


def _cancel_queued_generation(item: _QueuedGeneration, *, reason: str) -> None:
    item.cancelled = True
    token = item.sampling.cancellation_token
    if token is None:
        return
    if reason == "timeout":
        details = FinishDetails(reason="deadline_exceeded", deadline_exceeded=True)
    else:
        details = FinishDetails(reason="cancelled", cancelled=True)
    cancel = getattr(token, "cancel", None)
    if callable(cancel):
        cancel(details)


def _queued_generation_cancelled(item: _QueuedGeneration) -> bool:
    token = item.sampling.cancellation_token
    token_cancelled = token is not None and bool(getattr(token, "cancelled", False))
    return item.cancelled or token_cancelled or (item.future is not None and item.future.cancelled())


async def _finish_stream_queued_generation(
    item: _QueuedGeneration,
    *,
    exception: Exception | None = None,
) -> None:
    queue = item.stream_queue
    if queue is None or item.finished:
        return
    item.finished = True
    if item.cancelled:
        return
    if exception is not None:
        await queue.put(exception)
    await queue.put(_STREAM_DONE)


def _finish_queued_generation(
    item: _QueuedGeneration,
    *,
    outputs: Sequence[Any] | None = None,
    result: Any | None = None,
    exception: Exception | None = None,
) -> None:
    if item.finished:
        return
    item.finished = True
    if item.future is not None and not item.future.done():
        if exception is not None:
            item.future.set_exception(exception)
        elif result is not None:
            item.future.set_result(result)
        else:
            item.future.set_result(list(outputs or ()))
    if item.stream_queue is None:
        return
    terminal: list[object] = []
    if exception is not None:
        terminal.append(exception)
    else:
        terminal.extend(_coerce_generation_stream_chunk(output) for output in outputs or ())
    terminal.append(_STREAM_DONE)
    if len(terminal) > item.stream_queue.maxsize:
        terminal = [GenerationCancelled(FinishDetails(reason="cancelled", cancelled=True)), _STREAM_DONE]
    if item.stream_queue.qsize() + len(terminal) > item.stream_queue.maxsize:
        while not item.stream_queue.empty():
            with suppress(asyncio.QueueEmpty):
                item.stream_queue.get_nowait()
        terminal = [GenerationCancelled(FinishDetails(reason="cancelled", cancelled=True)), _STREAM_DONE]
    for event in terminal:
        item.stream_queue.put_nowait(event)


async def _stream_engine_text(
    engine: Any,
    prompt: PromptInput,
    sampling: SamplingParams,
    *,
    route: str = _SPECULATIVE_MTP_DEFAULT_ROUTE,
) -> AsyncIterator[Any]:
    if str(route) == _SPECULATIVE_PROVIDER_ROUTE:
        detailed_streamer = _engine_speculative_stream_callable(engine)
        if detailed_streamer is None:
            raise NotImplementedError(
                "speculative provider streaming is not supported by this engine"
            )
    elif str(route) == _SPECULATIVE_MTP_BATCH_ROUTE:
        detailed_streamer = getattr(
            engine,
            "stream_speculative_mtp_detailed",
            None,
        )
        if not callable(detailed_streamer):
            raise NotImplementedError(
                "speculative MTP streaming is not supported by this engine"
            )
    else:
        detailed_streamer = getattr(engine, "stream_detailed", None)
    if callable(detailed_streamer):
        iterator = iter(detailed_streamer(prompt, sampling))
        done = False
        try:
            while True:
                item = await run_in_threadpool(_next_stream_item, iterator)
                if item is _STREAM_DONE:
                    done = True
                    break
                yield _coerce_generation_stream_chunk(item)
        finally:
            if not done:
                closer = getattr(iterator, "close", None)
                if callable(closer):
                    await run_in_threadpool(closer)
        return
    streamer = getattr(engine, "stream", None)
    if not callable(streamer):
        for output in await run_in_threadpool(engine.generate, (prompt,), sampling):
            yield _coerce_generation_stream_chunk(output)
        return
    iterator = iter(streamer(prompt, sampling))
    done = False
    try:
        while True:
            item = await run_in_threadpool(_next_stream_item, iterator)
            if item is _STREAM_DONE:
                done = True
                break
            yield _coerce_generation_stream_chunk(item)
    finally:
        if not done:
            closer = getattr(iterator, "close", None)
            if callable(closer):
                await run_in_threadpool(closer)


async def _stream_engine_many(
    engine: Any,
    prompts: Sequence[PromptInput],
    sampling: SamplingParams,
) -> AsyncIterator[GenerationStreamChunk]:
    streamer = _engine_stream_many_callable(engine)
    if streamer is None:
        raise NotImplementedError("multi-row streaming is not supported by this generator")
    iterator = iter(streamer(tuple(prompts), sampling))
    done = False
    try:
        while True:
            item = await run_in_threadpool(_next_stream_item, iterator)
            if item is _STREAM_DONE:
                done = True
                break
            yield _coerce_generation_stream_chunk(item)
    finally:
        if not done:
            closer = getattr(iterator, "close", None)
            if callable(closer):
                await run_in_threadpool(closer)


@dataclass(frozen=True)
class _ReasoningSplit:
    content: str
    reasoning_content: str


@dataclass(frozen=True)
class _ParsedToolCall:
    id: str
    name: str
    arguments: str
    raw_text: str = ""


@dataclass(frozen=True)
class _ParsedChatOutput:
    text: str
    tool_calls: tuple[_ParsedToolCall, ...]
    reasoning_initially_open: bool = False
    tool_parser_name: str | None = None
    invalid_tool_call_blocks: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ToolValidationResult:
    parsed: _ParsedChatOutput
    failure_reason: str | None = None

    @property
    def failed(self) -> bool:
        return self.failure_reason is not None


@dataclass(frozen=True)
class _SchedulerToolArgumentFragment:
    tool_index: int
    call: _ParsedToolCall
    text: str
    stream_chunk: GenerationStreamChunk


@dataclass(frozen=True)
class _ReasoningPart:
    field: str
    text: str
    source_start: int
    source_end: int


@dataclass(frozen=True)
class _LiveSourceChunk:
    source_start: int
    source_end: int
    stream_chunk: GenerationStreamChunk


@dataclass(frozen=True)
class _ThinkingControl:
    enabled: bool | None = None
    effort: str | None = None
    allow_unbounded: bool = False
    max_think_tokens: int | None = None
    min_answer_tokens: int | None = None
    hard_think_cap: int | None = None
    soft_close_window: int | None = None
    hard_close_message: str | None = None
    hard_close_sequence: str | None = None


@dataclass
class _ReadinessState:
    ready: bool
    status: str
    eager_load: bool
    model_loaded: bool
    warmup_complete: bool
    startup_error: dict[str, Any] | None = None
    last_startup_timings: dict[str, float | None] = field(default_factory=dict)
    last_startup_memory: dict[str, Any] = field(default_factory=dict)
    last_startup_checks: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ContinuationRecord:
    id: str
    endpoint: str
    model_id: str
    auth_principal: str
    session_id: str | None
    tokenizer: dict[str, Any]
    prompts: tuple[str, ...]
    generated_texts: tuple[str, ...]
    created: float
    expires_at: float
    response_format: Any | None = None
    guided_json: Any | None = None
    guided_regex: Any | None = None
    guided_choice: Any | None = None
    guided_patch: Any | None = None
    guided_diff: Any | None = None

    def resume_prompts(self) -> tuple[str, ...]:
        return tuple(
            f"{prompt}{generated}"
            for prompt, generated in zip(self.prompts, self.generated_texts, strict=True)
        )


@dataclass(frozen=True)
class _ChatSessionRecord:
    id: str
    messages: tuple[dict[str, Any], ...]
    created: float
    updated: float


def create_app(config: ServerConfig, *, llm: Any | None = None) -> FastAPI:
    """Create a FastAPI app for OpenAI-compatible local inference.

    ``llm`` is injectable for tests and must expose ``generate(prompts,
    sampling_params)``.  Startup eagerly warms the configured model by default;
    disabling ``ServerConfig.eager_load`` keeps construction lazy until the first
    generation request.
    """

    app = FastAPI(title="hipEngine OpenAI-compatible API", version="0.2.1")
    metrics_mode = _metrics_mode(config.metrics)
    prefix_cache_mode = resolve_prefix_cache_mode(config.prefix_cache)
    app.state.hipengine_config = config
    app.state.hipengine_llm = llm
    app.state.hipengine_effective_max_context_tokens = config.max_context_tokens
    app.state.hipengine_prefix_cache_mode = prefix_cache_mode
    app.state.hipengine_server_metrics = _ServerMetrics()
    app.state.hipengine_readiness = _ReadinessState(
        ready=not bool(config.eager_load),
        status="ready" if not bool(config.eager_load) else "starting",
        eager_load=bool(config.eager_load),
        model_loaded=llm is not None,
        warmup_complete=not bool(config.eager_load),
        last_startup_timings={
            "engine_create_s": None,
            "resident_prepare_s": None,
            "warmup_s": None,
            "scratch_probe_s": None,
            "chat_smoke_s": None,
            "startup_total_s": None,
        },
    )
    if config.debug:
        app.add_middleware(_DebugPayloadMiddleware)
    session_lock = asyncio.Lock()
    continuation_lock = asyncio.Lock()
    continuations: dict[str, _ContinuationRecord] = {}
    app.state.hipengine_continuations = continuations
    app.state.hipengine_continuation_ttl_seconds = _CONTINUATION_TTL_SECONDS
    chat_session_lock = asyncio.Lock()
    chat_sessions: dict[str, _ChatSessionRecord] = {}
    chat_session_pending: set[str] = set()
    app.state.hipengine_chat_sessions = chat_sessions
    app.state.hipengine_chat_session_pending = chat_session_pending

    def chat_session_metadata(record: _ChatSessionRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "storage": "app_local_transcript",
            "resident_state_reuse": False,
            "message_count": len(record.messages),
            "created": int(record.created),
            "updated": int(record.updated),
        }

    def chat_session_snapshot(record: _ChatSessionRecord) -> dict[str, Any]:
        return {
            "object": "hipengine.session.snapshot",
            "schema": _CHAT_SESSION_SNAPSHOT_SCHEMA,
            "model": {
                "id": config.model_id,
                "backend": config.backend,
                "quant": config.quant,
            },
            "tokenizer": _tokenizer_compatibility_metadata(
                getattr(app.state, "hipengine_llm", None)
            ),
            "session": {
                **chat_session_metadata(record),
                "includes_transcript": True,
            },
            "resident_state_reuse": False,
            "messages": [_chat_session_message_copy(message) for message in record.messages],
        }

    def restore_chat_session_from_snapshot(session_id: str, snapshot: Mapping[str, Any]) -> _ChatSessionRecord:
        if snapshot.get("object") != "hipengine.session.snapshot":
            raise OpenAIHTTPError(
                400,
                "session snapshot object must be hipengine.session.snapshot",
                code="invalid_request",
                param="object",
            )
        if snapshot.get("schema") != _CHAT_SESSION_SNAPSHOT_SCHEMA:
            raise OpenAIHTTPError(
                400,
                f"session snapshot schema must be {_CHAT_SESSION_SNAPSHOT_SCHEMA!r}",
                code="invalid_request",
                param="schema",
            )
        if snapshot.get("resident_state_reuse") is not False:
            raise OpenAIHTTPError(
                400,
                "session snapshot resident_state_reuse must be false",
                code="invalid_request",
                param="resident_state_reuse",
            )
        model = snapshot.get("model")
        if not isinstance(model, Mapping):
            raise OpenAIHTTPError(400, "session snapshot model must be an object", code="invalid_request", param="model")
        expected_model = {
            "id": config.model_id,
            "backend": config.backend,
            "quant": config.quant,
        }
        for key, expected in expected_model.items():
            if model.get(key) != expected:
                raise OpenAIHTTPError(
                    400,
                    f"session snapshot model.{key} is incompatible with this server",
                    code="invalid_request",
                    param=f"model.{key}",
                )
        _validate_chat_session_snapshot_tokenizer_metadata(
            snapshot.get("tokenizer"),
            current_engine=getattr(app.state, "hipengine_llm", None),
        )
        session = snapshot.get("session")
        if not isinstance(session, Mapping):
            raise OpenAIHTTPError(
                400,
                "session snapshot session must be an object",
                code="invalid_request",
                param="session",
            )
        if session.get("storage") != "app_local_transcript":
            raise OpenAIHTTPError(
                400,
                "session snapshot storage must be app_local_transcript",
                code="invalid_request",
                param="session.storage",
            )
        if session.get("resident_state_reuse") is not False:
            raise OpenAIHTTPError(
                400,
                "session snapshot resident_state_reuse must be false",
                code="invalid_request",
                param="session.resident_state_reuse",
            )
        if session.get("includes_transcript") is not True:
            raise OpenAIHTTPError(
                400,
                "session snapshot includes_transcript must be true",
                code="invalid_request",
                param="session.includes_transcript",
            )
        snapshot_id = session.get("id")
        if snapshot_id != session_id:
            raise OpenAIHTTPError(
                400,
                "session snapshot id must match the restore path",
                code="invalid_request",
                param="session.id",
            )
        messages = _chat_session_snapshot_messages(snapshot.get("messages"))
        created = _chat_session_snapshot_time(session.get("created"), param="session.created")
        updated = _chat_session_snapshot_time(session.get("updated"), param="session.updated")
        return _ChatSessionRecord(
            id=session_id,
            messages=messages,
            created=created,
            updated=updated,
        )

    def chat_session_summary() -> dict[str, Any]:
        return {
            "resident_context": True,
            "active": len(chat_sessions),
            "pending_creations": len(chat_session_pending),
            "max_active": None if config.max_chat_sessions is None else int(config.max_chat_sessions),
            "storage": "app_local_transcript",
            "resident_state_reuse": False,
            "total_messages": sum(len(record.messages) for record in chat_sessions.values()),
            "continuations": {
                "active": len(continuations),
                "ttl_seconds": _CONTINUATION_TTL_SECONDS,
            },
        }

    def route_rejection_extra(
        *,
        requested_model: str | None,
        reason: str,
        engine: Any | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "hipengine": {
                "routing": _routing_rejection_metadata(
                    config,
                    requested_model=requested_model,
                    reason=reason,
                    engine=getattr(app.state, "hipengine_llm", None) if engine is None else engine,
                    details=details,
                )
            }
        }

    async def reserve_chat_session_if_needed(request: ChatCompletionRequest) -> str | None:
        session_id = _session_id(request)
        if session_id is None:
            return None
        if request.continuation_id is not None:
            return None
        cache_action = _session_cache_action(request)
        if cache_action in (None, "append_none"):
            return None
        async with chat_session_lock:
            if session_id in chat_sessions:
                return None
            if session_id in chat_session_pending:
                exc = OpenAIHTTPError(
                    429,
                    "chat session is being created",
                    error_type="rate_limit_error",
                    code="engine_busy",
                    extra=route_rejection_extra(
                        requested_model=request.model,
                        reason="engine_busy",
                        details={"overload_source": "chat_session_pending"},
                    ),
                    headers={"Retry-After": str(config.queue_retry_after_seconds)},
                )
                _record_openai_error(app.state.hipengine_server_metrics, exc)
                raise exc
            if config.max_chat_sessions is not None and (
                len(chat_sessions) + len(chat_session_pending)
            ) >= int(config.max_chat_sessions):
                exc = OpenAIHTTPError(
                    429,
                    "chat session limit is full",
                    error_type="rate_limit_error",
                    code="engine_busy",
                    extra=route_rejection_extra(
                        requested_model=request.model,
                        reason="engine_busy",
                        details={
                            "overload_source": "chat_session_cap",
                            "max_active_chat_sessions": int(config.max_chat_sessions),
                        },
                    ),
                    headers={"Retry-After": str(config.queue_retry_after_seconds)},
                )
                _record_openai_error(app.state.hipengine_server_metrics, exc)
                raise exc
            chat_session_pending.add(session_id)
            return session_id

    async def release_chat_session_reservation(session_id: str | None) -> None:
        if session_id is None:
            return
        async with chat_session_lock:
            chat_session_pending.discard(session_id)

    def cleanup_expired_continuations(now: float | None = None) -> None:
        current = time.time() if now is None else float(now)
        expired = [
            continuation_id
            for continuation_id, record in continuations.items()
            if record.expires_at <= current
        ]
        for continuation_id in expired:
            continuations.pop(continuation_id, None)

    async def pop_continuation(
        request: CompletionRequest | ChatCompletionRequest,
        *,
        endpoint: str,
        auth_principal: str,
    ) -> _ContinuationRecord | None:
        raw_id = getattr(request, "continuation_id", None)
        if raw_id is None:
            return None
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise OpenAIHTTPError(
                400,
                "continuation_id must be a non-empty string",
                code="invalid_request",
                param="continuation_id",
            )
        continuation_id = raw_id.strip()
        async with continuation_lock:
            record = continuations.get(continuation_id)
            if record is None:
                cleanup_expired_continuations()
                raise OpenAIHTTPError(
                    400,
                    "continuation_id is invalid or has already been consumed",
                    code="invalid_continuation",
                    param="continuation_id",
                )
            if record.expires_at <= time.time():
                continuations.pop(continuation_id, None)
                cleanup_expired_continuations()
                raise OpenAIHTTPError(
                    410,
                    "continuation_id has expired",
                    code="continuation_expired",
                    param="continuation_id",
                )
            if record.endpoint != endpoint:
                raise OpenAIHTTPError(
                    400,
                    f"continuation_id was created for {record.endpoint}, not {endpoint}",
                    code="invalid_continuation",
                    param="continuation_id",
                )
            if record.model_id != config.model_id:
                raise OpenAIHTTPError(
                    400,
                    "continuation_id is scoped to a different served model",
                    code="invalid_continuation",
                    param="continuation_id",
                )
            if record.auth_principal != auth_principal:
                raise OpenAIHTTPError(
                    400,
                    "continuation_id is scoped to a different auth principal",
                    code="invalid_continuation",
                    param="continuation_id",
                )
            if record.session_id != _session_id(request):
                raise OpenAIHTTPError(
                    400,
                    "continuation_id is scoped to a different session",
                    code="invalid_continuation",
                    param="continuation_id",
                )
            if endpoint == "chat" and record.session_id is not None:
                async with chat_session_lock:
                    if record.session_id not in chat_sessions:
                        raise OpenAIHTTPError(
                            400,
                            "continuation_id is scoped to a deleted session",
                            code="invalid_continuation",
                            param="continuation_id",
                        )
            current_engine = getattr(app.state, "hipengine_llm", None)
            if (
                current_engine is not None
                and record.tokenizer != _tokenizer_compatibility_metadata(current_engine)
            ):
                raise OpenAIHTTPError(
                    400,
                    "continuation_id is scoped to a different tokenizer",
                    code="invalid_continuation",
                    param="continuation_id",
                )
            cleanup_expired_continuations()
            return continuations.pop(continuation_id)

    async def store_continuation(
        *,
        endpoint: str,
        prompts: Sequence[str],
        generated_texts: Sequence[str],
        response_format: Any | None,
        guided_json: Any | None,
        guided_regex: Any | None,
        guided_choice: Any | None,
        guided_patch: Any | None,
        guided_diff: Any | None,
        auth_principal: str,
        session_id: str | None,
    ) -> _ContinuationRecord:
        now = time.time()
        record = _ContinuationRecord(
            id=f"gen_{uuid.uuid4().hex}",
            endpoint=endpoint,
            model_id=config.model_id,
            auth_principal=auth_principal,
            session_id=session_id,
            tokenizer=_tokenizer_compatibility_metadata(getattr(app.state, "hipengine_llm", None)),
            prompts=tuple(str(prompt) for prompt in prompts),
            generated_texts=tuple(str(text) for text in generated_texts),
            created=now,
            expires_at=now + _CONTINUATION_TTL_SECONDS,
            response_format=response_format,
            guided_json=guided_json,
            guided_regex=guided_regex,
            guided_choice=guided_choice,
            guided_patch=guided_patch,
            guided_diff=guided_diff,
        )
        async with continuation_lock:
            cleanup_expired_continuations(now)
            continuations[record.id] = record
        return record

    async def chat_session_prefix_messages(request: ChatCompletionRequest) -> tuple[dict[str, Any], ...]:
        session_id = _session_id(request)
        if session_id is None:
            return ()
        async with chat_session_lock:
            record = chat_sessions.get(session_id)
            if record is None:
                return ()
            return tuple(_chat_session_message_copy(message) for message in record.messages)

    async def commit_chat_session(
        request: ChatCompletionRequest,
        *,
        request_messages: Sequence[ChatMessage | Mapping[str, Any]],
        raw_output: str,
        visible_message: Mapping[str, Any],
        cache_action: str | None,
        reset_session: bool = False,
        commit_base_messages: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        session_id = _session_id(request)
        if session_id is None or cache_action in (None, "append_none"):
            return
        prompt_messages = tuple(_chat_message_to_session_dict(message) for message in request_messages)
        if cache_action == "append_prompt_only":
            appended = prompt_messages
        elif cache_action == "append_visible_only":
            appended = (*prompt_messages, _assistant_visible_session_message(visible_message))
        elif cache_action == "append_all":
            appended = (*prompt_messages, {"role": "assistant", "content": str(raw_output)})
        else:
            return
        now = time.time()
        async with chat_session_lock:
            previous = chat_sessions.get(session_id)
            created = now if previous is None or reset_session else previous.created
            if commit_base_messages is not None:
                base = tuple(_chat_session_message_copy(message) for message in commit_base_messages)
            else:
                base = () if previous is None or reset_session else previous.messages
            chat_sessions[session_id] = _ChatSessionRecord(
                id=session_id,
                messages=(*base, *appended),
                created=created,
                updated=now,
            )

    async def commit_chat_session_error(
        request: ChatCompletionRequest,
        *,
        request_messages: Sequence[ChatMessage | Mapping[str, Any]],
        exc: OpenAIHTTPError,
    ) -> None:
        if not isinstance(exc.finish_details, Mapping):
            return
        requested_cache_action = _session_cache_action(request)
        effective_cache_action = _effective_session_cache_action(
            requested_cache_action,
            exc.finish_details,
        )
        if effective_cache_action != "append_prompt_only":
            return
        exc.finish_details = {**dict(exc.finish_details), "cache_action": effective_cache_action}
        await commit_chat_session(
            request,
            request_messages=request_messages,
            raw_output="",
            visible_message={"role": "assistant", "content": ""},
            cache_action=effective_cache_action,
        )

    def chat_context_fit_payload(
        chat_request: ChatCompletionRequest,
        prompt: PromptInput,
        engine: Any,
    ) -> dict[str, Any]:
        max_context = effective_max_context_tokens(engine)
        max_tokens = _request_max_tokens(
            chat_request,
            (prompt,),
            engine,
            max_context,
            chat_default_max_tokens=config.chat_default_max_tokens,
        )
        return _context_fit_payload(
            prompt_tokens=_prompt_token_count(engine, prompt),
            max_context_tokens=max_context,
            max_tokens=max_tokens,
        )

    def chat_context_candidate(
        request: ChatCompletionRequest,
        prefix_messages: Sequence[Mapping[str, Any]],
        engine: Any,
    ) -> tuple[ChatCompletionRequest, str, PromptInput, _ThinkingControl]:
        render_request = request
        if prefix_messages:
            render_request = _chat_request_with_messages(
                request,
                (*prefix_messages, *request.messages),
            )
        prompt, thinking, prepared_prompt = _render_prepared_chat_prompt_for_request(
            render_request,
            chat_default_max_tokens=config.chat_default_max_tokens,
            engine=engine,
            max_context_tokens=effective_max_context_tokens(engine),
        )
        return render_request, prompt, prepared_prompt, thinking

    async def render_chat_context_for_request(
        request: ChatCompletionRequest,
        engine: Any,
        *,
        apply_context_policy: bool,
    ) -> _ChatContextRender:
        prefix_messages = tuple(await chat_session_prefix_messages(request))
        render_request, prompt, prepared_prompt, thinking = chat_context_candidate(
            request,
            prefix_messages,
            engine,
        )
        effective_prefix = prefix_messages
        session_payload = _diagnostic_session_payload(request, effective_prefix)
        fit_context_extra = None if session_payload is None else {"session": session_payload}
        reset_session = False
        commit_base_messages = None
        policy = _session_context_overflow_policy(request)
        if (
            apply_context_policy
            and policy in {"auto_clear_transient", "new_session", "truncate_oldest_visible"}
            and session_payload is not None
        ):
            dropped_message_count = 0
            if (
                policy != "auto_clear_transient"
                and prefix_messages
                and effective_max_context_tokens(engine) is not None
            ):
                prefixed_fit = chat_context_fit_payload(
                    render_request,
                    prepared_prompt,
                    engine,
                )
                if not prefixed_fit["fits"]:
                    candidate_drop_counts = (
                        (len(prefix_messages),)
                        if policy == "new_session"
                        else tuple(range(1, len(prefix_messages) + 1))
                    )
                    for drop_count in candidate_drop_counts:
                        candidate_prefix = prefix_messages[drop_count:]
                        try:
                            (
                                candidate_request,
                                candidate_prompt,
                                candidate_prepared_prompt,
                                candidate_thinking,
                            ) = chat_context_candidate(
                                request,
                                candidate_prefix,
                                engine,
                            )
                        except OpenAIHTTPError:
                            continue
                        candidate_fit = chat_context_fit_payload(
                            candidate_request,
                            candidate_prepared_prompt,
                            engine,
                        )
                        if candidate_fit["fits"]:
                            prompt = candidate_prompt
                            prepared_prompt = candidate_prepared_prompt
                            thinking = candidate_thinking
                            render_request = candidate_request
                            effective_prefix = candidate_prefix
                            dropped_message_count = drop_count
                            reset_session = policy == "new_session"
                            commit_base_messages = tuple(dict(item) for item in candidate_prefix)
                            session_payload = _diagnostic_session_payload(request, effective_prefix)
                            break
            policy_extra = _context_policy_session_prefix_payload(
                request,
                session_id=_session_id(request),
                clear_policy=policy,
                dropped_message_count=dropped_message_count,
                kept_prefix_message_count=len(effective_prefix),
                reset=reset_session,
            )
            fit_context_extra = {"session": session_payload, **policy_extra}
        return _ChatContextRender(
            prompt=prompt,
            render_request=render_request,
            prefix_messages=tuple(dict(item) for item in effective_prefix),
            prepared_prompt=prepared_prompt,
            thinking=thinking,
            session_payload=session_payload,
            fit_context_extra=fit_context_extra,
            reset_session_on_commit=reset_session,
            commit_base_messages=commit_base_messages,
        )

    async def diagnostic_render_for_request(
        request: TokenDiagnosticRequest,
        engine: Any,
        *,
        apply_context_policy: bool = False,
    ) -> dict[str, Any]:
        has_text = request.text is not None
        has_messages = request.messages is not None
        if has_text == has_messages:
            raise OpenAIHTTPError(
                400,
                "provide exactly one of text or messages",
                code="invalid_request",
                param="text",
            )
        if has_text:
            if request.session is not None:
                raise OpenAIHTTPError(
                    400,
                    "session diagnostics are only supported for chat messages",
                    code="unsupported_parameter",
                    param="session",
                )
            return {
                "text": str(request.text),
                "input_type": "text",
                "chat_request": None,
                "session": None,
            }
        chat_request = _chat_request_from_diagnostic(config, request)
        _validate_session_request(chat_request)
        unsupported_param = _unsupported_agentic_request_param(
            chat_request,
            engine=engine,
        )
        if unsupported_param is not None:
            raise OpenAIHTTPError(
                400,
                f"{unsupported_param} is not supported by this server",
                code="unsupported_parameter",
                param=unsupported_param,
            )
        rendered = await render_chat_context_for_request(
            chat_request,
            engine,
            apply_context_policy=apply_context_policy,
        )
        return {
            "text": rendered.prompt,
            "input_type": "chat",
            "chat_request": rendered.render_request,
            "session": rendered.session_payload,
            "fit_context_extra": rendered.fit_context_extra,
        }

    mtp_circuit_breaker = _MTPCircuitBreaker()
    app.state.hipengine_mtp_circuit_breaker = mtp_circuit_breaker

    def get_llm() -> Any:
        if app.state.hipengine_llm is None:
            app.state.hipengine_llm = LLM(
                config.model,
                backend=config.backend,
                quant=config.quant,
                execution_profile=config.execution_profile,
                max_active_requests=config.max_active_requests,
                max_sequence_length=config.max_context_tokens,
                prefix_cache=prefix_cache_mode,
                speculative_provider=config.speculative_provider,
                draft_model=config.draft_model,
                speculative_candidate_budget=config.speculative_candidate_budget,
            )
            _log_effective_mtp_config(config, engine=app.state.hipengine_llm)
        mtp_circuit_breaker.attach(app.state.hipengine_llm)
        app.state.hipengine_readiness.model_loaded = True
        return app.state.hipengine_llm

    generation_batcher = _GenerationBatcher(
        engine_factory=get_llm,
        batch_window_seconds=float(config.generation_batch_window_ms) / 1000.0,
        max_queue_size=config.max_queued_requests,
        max_active_requests=config.max_active_requests,
        route_max_active_requests={
            **(
                {_SPECULATIVE_PROVIDER_ROUTE: 1}
                if config.speculative_provider is not None
                else {}
            ),
            **(
                {
                    _SPECULATIVE_MTP_DEFAULT_ROUTE: _GGUF_DEFAULT_AR_MAX_ACTIVE_REQUESTS,
                    _SPECULATIVE_MTP_BATCH_ROUTE: _GGUF_MTP_MAX_ACTIVE_REQUESTS,
                    _SPECULATIVE_MTP_AUTO_ROUTE: _GGUF_DEFAULT_AR_MAX_ACTIVE_REQUESTS,
                }
                if _server_model_uses_gguf(config, llm)
                else {}
            ),
        },
        retry_after_seconds=config.queue_retry_after_seconds,
        stream_queue_max_chunks=config.stream_queue_max_chunks,
        mtp_circuit_breaker=mtp_circuit_breaker,
    )
    app.state.hipengine_generation_batcher = generation_batcher

    def configured_max_context_tokens() -> int | None:
        if config.max_context_tokens is None:
            return None
        return max(1, int(config.max_context_tokens))

    def effective_max_context_tokens(engine: Any) -> int | None:
        configured = configured_max_context_tokens()
        if configured is not None:
            return configured
        cached = getattr(app.state, "hipengine_effective_max_context_tokens", None)
        if cached is not None:
            return max(1, int(cached))
        prepared = _prepared_context_tokens(engine)
        if prepared is not None:
            app.state.hipengine_effective_max_context_tokens = prepared
            return prepared
        return None

    def preparation_sampling(request: CompletionRequest | ChatCompletionRequest | None = None) -> SamplingParams:
        return SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
            kv_storage=(request.kv_storage if request is not None and request.kv_storage else config.kv_storage),
            kv_scale_dtype=(
                request.kv_scale_dtype if request is not None and request.kv_scale_dtype else config.kv_scale_dtype
            ),
            kv_scale_granularity=(
                request.kv_scale_granularity
                if request is not None and request.kv_scale_granularity
                else config.kv_scale_granularity
            ),
        )

    async def ensure_resident_context(
        engine: Any,
        sampling: SamplingParams,
        *,
        phase: str,
    ) -> int | None:
        requested_context = configured_max_context_tokens()
        prepared = _prepared_context_tokens(engine)
        if prepared is not None and (requested_context is None or prepared >= requested_context):
            app.state.hipengine_effective_max_context_tokens = prepared
            return effective_max_context_tokens(engine)
        preparer = getattr(engine, "prepare", None)
        if not callable(preparer):
            return effective_max_context_tokens(engine)
        prepare_started = time.perf_counter()
        try:
            prepared_result = await run_in_threadpool(
                lambda: preparer(
                    max_sequence_length=requested_context,
                    sampling_params=sampling,
                )
            )
        except MemoryError as exc:
            _LOGGER.error(
                "hipEngine %s failed to allocate resident KV cache: %s. "
                "Try a lower --max-context-tokens or --kv-storage int8_per_token_head.",
                phase,
                exc,
            )
            raise
        except Exception as exc:
            _LOGGER.error(
                "hipEngine %s failed to prepare resident session/KV cache: %s. "
                "Try a lower --max-context-tokens or --kv-storage int8_per_token_head.",
                phase,
                exc,
            )
            raise
        if prepared_result is not None:
            app.state.hipengine_effective_max_context_tokens = max(1, int(prepared_result))
        else:
            prepared = _prepared_context_tokens(engine)
            if prepared is not None:
                app.state.hipengine_effective_max_context_tokens = prepared
        effective = effective_max_context_tokens(engine)
        _LOGGER.info(
            "LOAD_TIMING: phase=%s resident_prepare_s=%.3f max_context_tokens=%s",
            phase,
            time.perf_counter() - prepare_started,
            "unknown" if effective is None else str(effective),
        )
        return effective

    def mark_startup_failed(
        readiness: _ReadinessState,
        exc: BaseException,
        *,
        startup_started: float,
        stage: str,
        guidance: str,
        timings: Mapping[str, float | None] | None = None,
    ) -> None:
        readiness.ready = False
        readiness.status = "error"
        readiness.model_loaded = app.state.hipengine_llm is not None
        readiness.warmup_complete = False
        readiness.startup_error = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": f"startup {stage} failed",
            "guidance": guidance,
        }
        startup_total_s = time.perf_counter() - startup_started
        updated_timings = dict(readiness.last_startup_timings)
        updated_timings.update(dict(timings or {}))
        updated_timings["startup_total_s"] = round(startup_total_s, 6)
        readiness.last_startup_timings = updated_timings

    async def eager_load_model() -> None:
        readiness = app.state.hipengine_readiness
        readiness.status = "starting"
        readiness.ready = not bool(config.eager_load)
        readiness.startup_error = None
        startup_started = time.perf_counter()
        startup_memory: dict[str, Any] = {}
        startup_checks: dict[str, Any] = {}
        readiness.last_startup_memory = startup_memory
        readiness.last_startup_checks = startup_checks
        _record_startup_memory_snapshot(startup_memory, "startup_begin")
        max_tokens = max(1, int(config.eager_load_max_tokens))
        if not config.eager_load:
            max_context = configured_max_context_tokens()
            _LOGGER.info(
                "Config: model=%s served_model=%s max_context_tokens=%s "
                "chat_default_max_tokens=%s kv_storage=%s kv_scale_dtype=%s "
                "kv_scale_granularity=%s eager_load=False",
                config.model,
                config.model_id,
                "auto" if max_context is None else str(max_context),
                _chat_default_max_tokens_label(config),
                config.kv_storage,
                config.kv_scale_dtype,
                config.kv_scale_granularity,
            )
            startup_total_s = time.perf_counter() - startup_started
            readiness.ready = True
            readiness.status = "ready"
            readiness.model_loaded = app.state.hipengine_llm is not None
            readiness.warmup_complete = True
            readiness.last_startup_timings = {
                "engine_create_s": None,
                "resident_prepare_s": None,
                "warmup_s": None,
                "scratch_probe_s": None,
                "chat_smoke_s": None,
                "startup_total_s": round(startup_total_s, 6),
            }
            _LOGGER.info("LOAD_TIMING: eager_load=False startup_total_s=%.3f", startup_total_s)
            _log_effective_mtp_config(config, engine=app.state.hipengine_llm)
            _LOGGER.info("hipEngine is ready (lazy load).")
            return
        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
            kv_storage=config.kv_storage,
            kv_scale_dtype=config.kv_scale_dtype,
            kv_scale_granularity=config.kv_scale_granularity,
        )
        async with session_lock:
            engine_started = time.perf_counter()
            try:
                engine = get_llm()
            except Exception as exc:
                startup_checks["engine_create"] = {
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                }
                mark_startup_failed(
                    readiness,
                    exc,
                    startup_started=startup_started,
                    stage="engine_create",
                    guidance="Check the configured model path, backend, quantization, and server logs.",
                    timings={"engine_create_s": round(time.perf_counter() - engine_started, 6)},
                )
                _LOGGER.exception("STARTUP_ENGINE_CREATE: failed")
                return
            engine_create_s = time.perf_counter() - engine_started
            prepare_started = time.perf_counter()
            try:
                max_context = await ensure_resident_context(engine, sampling, phase="startup")
            except Exception as exc:
                startup_checks["resident_prepare"] = {
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                }
                mark_startup_failed(
                    readiness,
                    exc,
                    startup_started=startup_started,
                    stage="resident_prepare",
                    guidance="Try a lower --max-context-tokens or --kv-storage int8_per_token_head.",
                    timings={
                        "engine_create_s": round(engine_create_s, 6),
                        "resident_prepare_s": round(time.perf_counter() - prepare_started, 6),
                    },
                )
                return
            resident_prepare_s = time.perf_counter() - prepare_started
            _LOGGER.info(
                "Config: model=%s served_model=%s max_context_tokens=%s "
                "chat_default_max_tokens=%s kv_storage=%s kv_scale_dtype=%s "
                "kv_scale_granularity=%s eager_load=True",
                config.model,
                config.model_id,
                "unknown" if max_context is None else str(max_context),
                _chat_default_max_tokens_label(config),
                config.kv_storage,
                config.kv_scale_dtype,
                config.kv_scale_granularity,
            )
            _log_kv_capacity_summary(engine)
            _validate_context_budget(max_context, engine, (config.eager_load_prompt,), sampling)
        _record_startup_memory_snapshot(startup_memory, "after_resident_prepare")
        _LOGGER.info(
            "WARMUP: prompt_tokens<=%s max_tokens=%d",
            "unknown" if max_context is None else str(max_context),
            max_tokens,
        )
        warmup_started = time.perf_counter()
        try:
            await run_in_threadpool(engine.generate, (config.eager_load_prompt,), sampling)
        except Exception as exc:
            startup_checks["raw_warmup"] = {
                "status": "failed",
                "max_tokens": max_tokens,
                "exception_type": type(exc).__name__,
            }
            mark_startup_failed(
                readiness,
                exc,
                startup_started=startup_started,
                stage="raw_warmup",
                guidance="Check backend generation logs and lower --eager-load-max-tokens if needed.",
                timings={
                    "engine_create_s": round(engine_create_s, 6),
                    "resident_prepare_s": round(resident_prepare_s, 6),
                    "warmup_s": round(time.perf_counter() - warmup_started, 6),
                },
            )
            _LOGGER.exception("WARMUP: failed during eager startup")
            return
        warmup_s = time.perf_counter() - warmup_started
        startup_checks["raw_warmup"] = {"status": "passed", "max_tokens": max_tokens}
        _record_startup_memory_snapshot(startup_memory, "after_raw_warmup")

        scratch_probe_s: float | None = None
        scratch_probe_started = time.perf_counter()
        max_prompt_tokens = _startup_max_prompt_tokens(max_context)
        if config.startup_scratch_probe:
            scratch_preparer = getattr(engine, "prepare_request_scratch", None)
            scratch_probe_context_unknown = max_prompt_tokens is None
            scratch_probe_prompt_tokens = 64 if max_prompt_tokens is None else int(max_prompt_tokens)
            scratch_probe_batch_size = max(1, int(config.max_active_requests or 1))
            mtp_startup_warmup_requested = str(config.speculative_mtp_serving) != "off"
            mtp_route_enabled = (
                mtp_startup_warmup_requested
                and _engine_supports_speculative_mtp(engine)
            )
            if not mtp_route_enabled:
                plain_ar_limit = getattr(
                    engine,
                    "server_plain_ar_max_active_requests",
                    None,
                )
                if plain_ar_limit is not None:
                    plain_ar_limit = int(plain_ar_limit)
                    if plain_ar_limit < 1:
                        raise ValueError(
                            "server_plain_ar_max_active_requests must be positive"
                        )
                    scratch_probe_batch_size = min(
                        scratch_probe_batch_size,
                        plain_ar_limit,
                    )
            if max_prompt_tokens is None and scratch_probe_batch_size <= 1:
                startup_checks["scratch_probe"] = {"enabled": True, "status": "skipped", "reason": "unknown_context"}
            elif not callable(scratch_preparer):
                startup_checks["scratch_probe"] = {"enabled": True, "status": "skipped", "reason": "backend_hook_unavailable"}
                _LOGGER.warning("STARTUP_SCRATCH_PROBE: skipped; backend does not expose prepare_request_scratch")
            else:
                _LOGGER.info(
                    "STARTUP_SCRATCH_PROBE: max_prompt_tokens=%s probe_prompt_tokens=%d max_new_tokens=0 max_batch_size=%d release_after_probe=True",
                    "unknown" if scratch_probe_context_unknown else str(max_prompt_tokens),
                    scratch_probe_prompt_tokens,
                    scratch_probe_batch_size,
                )
                try:
                    def run_scratch_probe() -> Any:
                        mtp_warmup_env = "HIPENGINE_GGUF_MTP_SERVER_STARTUP_WARMUP"
                        previous_mtp_warmup = os.environ.get(mtp_warmup_env)
                        os.environ[mtp_warmup_env] = "1" if mtp_startup_warmup_requested else "0"
                        try:
                            return scratch_preparer(
                                max_prompt_tokens=scratch_probe_prompt_tokens,
                                max_new_tokens=0,
                                sampling_params=sampling,
                                max_batch_size=scratch_probe_batch_size,
                                release_after_probe=True,
                            )
                        finally:
                            if previous_mtp_warmup is None:
                                os.environ.pop(mtp_warmup_env, None)
                            else:
                                os.environ[mtp_warmup_env] = previous_mtp_warmup

                    scratch_result = await run_in_threadpool(
                        run_scratch_probe
                    )
                except Exception as exc:
                    startup_checks["scratch_probe"] = {
                        "enabled": True,
                        "status": "failed",
                        "max_prompt_tokens": None if scratch_probe_context_unknown else scratch_probe_prompt_tokens,
                        "probe_prompt_tokens": scratch_probe_prompt_tokens,
                        "context_unknown": scratch_probe_context_unknown,
                        "exception_type": type(exc).__name__,
                    }
                    _LOGGER.error(
                        "STARTUP_SCRATCH_PROBE: failed at probe_prompt_tokens=%d: %s. "
                        "Try a lower --max-context-tokens or a higher scratch/headroom reserve.",
                        scratch_probe_prompt_tokens,
                        exc,
                    )
                    mark_startup_failed(
                        readiness,
                        exc,
                        startup_started=startup_started,
                        stage="scratch_probe",
                        guidance="Try a lower --max-context-tokens or a higher scratch/headroom reserve.",
                        timings={
                            "engine_create_s": round(engine_create_s, 6),
                            "resident_prepare_s": round(resident_prepare_s, 6),
                            "warmup_s": round(warmup_s, 6),
                            "scratch_probe_s": round(time.perf_counter() - scratch_probe_started, 6),
                        },
                    )
                    return
                startup_checks["scratch_probe"] = {
                    "enabled": True,
                    "status": "passed",
                    "max_prompt_tokens": None if scratch_probe_context_unknown else scratch_probe_prompt_tokens,
                    "probe_prompt_tokens": scratch_probe_prompt_tokens,
                    "context_unknown": scratch_probe_context_unknown,
                    "result": scratch_result,
                }
        else:
            startup_checks["scratch_probe"] = {"enabled": False, "status": "disabled"}
        scratch_probe_s = time.perf_counter() - scratch_probe_started
        _record_startup_memory_snapshot(startup_memory, "after_scratch_probe")

        chat_smoke_s: float | None = None
        if config.startup_chat_smoke:
            smoke_started = time.perf_counter()
            smoke_request = ChatCompletionRequest(
                model=config.model_id,
                messages=[ChatMessage(role="user", content="hello")],
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
            )
            async with session_lock:
                smoke_prompt = chat_prompt_for_request(smoke_request, engine)
                smoke_sampling = sampling_params(smoke_request, (smoke_prompt,), engine)
                smoke_prompt_tokens = _count_tokens_for_admission(engine, smoke_prompt)
                _validate_context_budget(effective_max_context_tokens(engine), engine, (smoke_prompt,), smoke_sampling)
            _LOGGER.info(
                "WARMUP_CHAT: prompt_tokens=%d max_tokens=%d",
                smoke_prompt_tokens,
                int(smoke_sampling.max_tokens),
            )
            await generation_batcher.submit((smoke_prompt,), smoke_sampling, detailed=True)
            chat_smoke_s = time.perf_counter() - smoke_started
            startup_checks["chat_smoke"] = {
                "enabled": True,
                "status": "passed",
                "prompt_tokens": int(smoke_prompt_tokens),
                "max_tokens": int(smoke_sampling.max_tokens),
            }
            _record_startup_memory_snapshot(startup_memory, "after_chat_smoke")
        else:
            startup_checks["chat_smoke"] = {"enabled": False, "status": "disabled"}

        guard_memory = _device_memory_snapshot()
        if guard_memory is not None:
            startup_memory["guard"] = guard_memory
        _log_startup_memory_summary(startup_memory, startup_checks)
        _startup_free_memory_guard(memory=guard_memory, min_free_mib=config.startup_min_free_mib)
        startup_total_s = time.perf_counter() - startup_started
        readiness.ready = True
        readiness.status = "ready"
        readiness.model_loaded = True
        readiness.warmup_complete = True
        readiness.last_startup_timings = {
            "engine_create_s": round(engine_create_s, 6),
            "resident_prepare_s": round(resident_prepare_s, 6),
            "warmup_s": round(warmup_s, 6),
            "scratch_probe_s": None if scratch_probe_s is None else round(scratch_probe_s, 6),
            "chat_smoke_s": None if chat_smoke_s is None else round(chat_smoke_s, 6),
            "startup_total_s": round(startup_total_s, 6),
        }
        _LOGGER.info(
            "LOAD_TIMING: model=%s engine_create_s=%.3f resident_prepare_s=%.3f warmup_s=%.3f scratch_probe_s=%s chat_smoke_s=%s startup_total_s=%.3f",
            config.model_id,
            engine_create_s,
            resident_prepare_s,
            warmup_s,
            "skipped" if scratch_probe_s is None else f"{scratch_probe_s:.3f}",
            "skipped" if chat_smoke_s is None else f"{chat_smoke_s:.3f}",
            startup_total_s,
        )
        _log_effective_mtp_config(config, engine=app.state.hipengine_llm)
        _LOGGER.info("hipEngine is ready.")

    async def shutdown_model() -> None:
        readiness: _ReadinessState = app.state.hipengine_readiness
        readiness.ready = False
        readiness.status = "draining"
        shutdown_result = await generation_batcher.shutdown(
            grace_seconds=config.shutdown_grace_seconds
        )
        engine = getattr(app.state, "hipengine_llm", None)
        closer = None if engine is None else getattr(engine, "close", None)
        if callable(closer):
            await run_in_threadpool(closer)
        app.state.hipengine_shutdown = shutdown_result
        readiness.status = "stopped"

    if hasattr(app, "add_event_handler"):
        app.add_event_handler("startup", eager_load_model)
        app.add_event_handler("shutdown", shutdown_model)
    else:  # FastAPI-lite compatibility in minimal test/runtime environments.
        app.router.on_startup.append(eager_load_model)
        app.router.on_shutdown.append(shutdown_model)

    async def require_auth(request: Request) -> str:
        if not config.api_key:
            return "anonymous"
        expected = f"Bearer {config.api_key}"
        if request.headers.get("authorization") != expected:
            raise OpenAIHTTPError(
                401,
                "missing or invalid bearer token",
                error_type="authentication_error",
                code="invalid_api_key",
            )
        return f"bearer_sha256:{_sha256_text(config.api_key)}"

    @app.exception_handler(OpenAIHTTPError)
    async def openai_error_handler(request: Request, exc: OpenAIHTTPError) -> JSONResponse:
        _log_request_failure(
            request,
            status_code=exc.status_code,
            code=exc.code,
            param=exc.param,
            message=exc.message,
        )
        headers = dict(exc.headers)
        if exc.status_code == 401:
            headers.setdefault("WWW-Authenticate", "Bearer")
        error_payload = _error_payload(
            message=exc.message,
            error_type=exc.error_type,
            code=exc.code,
            param=exc.param,
            status_code=exc.status_code,
            finish_details=exc.finish_details,
            extra=exc.extra,
        )
        await _maybe_write_replay_artifact(
            config,
            request,
            error_payload,
            engine=getattr(app.state, "hipengine_llm", None),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": error_payload},
            headers=headers or None,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        message = _format_validation_error(exc)
        param = _validation_error_param(exc)
        _log_request_failure(
            request,
            status_code=422,
            code="validation_error",
            param=param,
            message=message,
        )
        error_payload = _error_payload(
            message=message,
            error_type="invalid_request_error",
            param=param,
            code="validation_error",
            status_code=422,
        )
        await _maybe_write_replay_artifact(
            config,
            request,
            error_payload,
            engine=getattr(app.state, "hipengine_llm", None),
        )
        return JSONResponse(
            status_code=422,
            content={"error": error_payload},
        )

    def sampling_params(
        request: CompletionRequest | ChatCompletionRequest,
        prompts: Sequence[PromptInput],
        engine: Any,
        *,
        deadline_at: float | None = None,
        cancellation_token: GenerationCancellationToken | None = None,
        prepared_thinking: _ThinkingControl | None = None,
        resident_session_key: str | None = None,
    ) -> SamplingParams:
        stop_token_ids, stop_token_sequences = _stop_tokens_from_stop(request.stop, engine)
        uses_stored_chat_prompt = isinstance(request, ChatCompletionRequest) and request.continuation_id is not None
        thinking_budget = (
            _thinking_budget_sampling_kwargs(
                config,
                request,
                engine=engine,
                max_context_tokens=effective_max_context_tokens(engine),
                prepared_thinking=prepared_thinking,
            )
            if isinstance(request, ChatCompletionRequest) and not uses_stored_chat_prompt
            else {}
        )
        no_tool_suppress_token_ids = (
            _no_tool_sampling_suppress_token_ids(request, engine)
            if isinstance(request, ChatCompletionRequest) and not uses_stored_chat_prompt
            else ()
        )
        forced_tool_token_ids, strict_tool_schema_prefix_anchored = (
            _required_tool_sampling_forced_prefix(request, engine)
            if isinstance(request, ChatCompletionRequest) and not uses_stored_chat_prompt
            else ((), False)
        )
        tool_call_constraint = (
            _tool_call_sampling_constraint(
                request,
                engine,
                chat_default_max_tokens=config.chat_default_max_tokens,
            )
            if isinstance(request, ChatCompletionRequest) and not uses_stored_chat_prompt
            else None
        )
        force_sequence_completion_token_sequences = (
            _tool_call_sequence_completion_token_sequences(
                request,
                engine,
                forced_tool_token_ids,
                include_object_close=strict_tool_schema_prefix_anchored,
            )
            if isinstance(request, ChatCompletionRequest)
            else ()
        )
        stop_token_sequences = tuple(
            dict.fromkeys((*stop_token_sequences, *force_sequence_completion_token_sequences))
        )
        initial_forced_tool_token_ids = () if thinking_budget else forced_tool_token_ids
        post_thinking_forced_tool_token_ids = forced_tool_token_ids if thinking_budget else ()
        suppress_token_ids = tuple(
            dict.fromkeys(
                (
                    *(int(token) for token in (request.suppress_token_ids or ())),
                    *no_tool_suppress_token_ids,
                )
            )
        )
        return SamplingParams(
            max_tokens=_request_max_tokens(
                request,
                prompts,
                engine,
                effective_max_context_tokens(engine),
                chat_default_max_tokens=config.chat_default_max_tokens,
            ),
            logprobs=_request_logprobs_enabled(request),
            top_logprobs=_request_top_logprobs(request),
            temperature=float(request.temperature if request.temperature is not None else 0.0),
            top_p=float(request.top_p if request.top_p is not None else 1.0),
            top_k=int(request.top_k if request.top_k is not None else 0),
            min_p=float(request.min_p if request.min_p is not None else 0.0),
            repetition_penalty=float(request.repetition_penalty if request.repetition_penalty is not None else 1.0),
            presence_penalty=float(request.presence_penalty if request.presence_penalty is not None else 0.0),
            frequency_penalty=float(request.frequency_penalty if request.frequency_penalty is not None else 0.0),
            logit_bias=request.logit_bias or (),
            suppress_token_ids=suppress_token_ids,
            min_tokens=int(request.min_tokens if request.min_tokens is not None else 0),
            eos_token_id=None if request.eos_token_id is None else int(request.eos_token_id),
            stop_token_ids=stop_token_ids,
            stop_token_sequences=stop_token_sequences,
            forced_tokens_pending=initial_forced_tool_token_ids,
            forced_token_reason="tool_choice_required" if initial_forced_tool_token_ids else None,
            post_thinking_forced_tokens_pending=post_thinking_forced_tool_token_ids,
            post_thinking_forced_token_reason="tool_choice_required" if post_thinking_forced_tool_token_ids else None,
            force_sequence_completion_token_sequences=force_sequence_completion_token_sequences,
            force_sequence_completion_reason=(
                "tool_call_sequence_completion" if force_sequence_completion_token_sequences else None
            ),
            json_object_close_forcing=_json_object_close_forcing(request),
            tool_call_constraint=tool_call_constraint,
            ignore_eos=bool(request.ignore_eos),
            kv_storage=request.kv_storage or config.kv_storage,
            kv_scale_dtype=request.kv_scale_dtype or config.kv_scale_dtype,
            kv_scale_granularity=request.kv_scale_granularity or config.kv_scale_granularity,
            seed=request.seed,
            deadline_at=deadline_at,
            cancellation_token=cancellation_token,
            resident_session_key=resident_session_key,
            resident_session_cache_action=(
                _session_cache_action(request)
                if resident_session_key is not None
                and isinstance(request, ChatCompletionRequest)
                else None
            ),
            **thinking_budget,
        )

    def chat_prompt_for_request(request: ChatCompletionRequest, engine: Any) -> str:
        max_context = effective_max_context_tokens(engine)
        prompt, _thinking = _render_chat_prompt_for_request(
            request,
            chat_default_max_tokens=config.chat_default_max_tokens,
            engine=engine,
            max_context_tokens=max_context,
        )
        return prompt

    async def generate(
        prompts: Sequence[PromptInput],
        request: CompletionRequest | ChatCompletionRequest,
        *,
        deadline_at: float | None = None,
        cancellation_token: GenerationCancellationToken | None = None,
        fit_context_extra: Mapping[str, Any] | None = None,
        prepared_thinking: _ThinkingControl | None = None,
        resident_session_key: str | None = None,
    ) -> _GeneratedBatch:
        generation_shape: dict[str, Any] | None = None
        try:
            _validate_generation_request(
                config,
                request,
                engine=getattr(app.state, "hipengine_llm", None),
                route_unsupported_grammar=True,
            )
            async with session_lock:
                engine = get_llm()
                prompts = _prepare_prompt_inputs(engine, prompts)
                admission_started = time.perf_counter()
                sampling = sampling_params(
                    request,
                    prompts,
                    engine,
                    deadline_at=deadline_at,
                    cancellation_token=cancellation_token,
                    prepared_thinking=prepared_thinking,
                    resident_session_key=resident_session_key,
                )
                if _request_n(request) > 1:
                    sampling = replace(
                        sampling,
                        row_seeds=_row_seeds_for_request(request.seed, len(prompts)),
                    )
                generation_route, generation_route_decision = (
                    _generation_route_for_request(
                        config,
                        request,
                        engine=engine,
                        sampling=sampling,
                        prompts=prompts,
                    )
                )
                await ensure_resident_context(
                    engine,
                    preparation_sampling(request),
                    phase="preparation",
                )
                _validate_context_budget(
                    effective_max_context_tokens(engine),
                    engine,
                    prompts,
                    sampling,
                    fit_context_extra=fit_context_extra,
                    error_extra={
                        "hipengine": {
                            "routing": _routing_rejection_metadata(
                                config,
                                requested_model=request.model,
                                reason="context_overflow",
                                engine=engine,
                            )
                        }
                    },
                )
                prompts = _with_admission_prepare_ms(
                    prompts,
                    (time.perf_counter() - admission_started) * 1_000.0,
                )
            if _request_logprobs_enabled(request):
                generation_route, generation_route_decision = (
                    _resolve_realized_generation_route(
                        generation_route,
                        group_rows=len(prompts),
                        sampling=sampling,
                        precomputed_decision=generation_route_decision,
                    )
                )
                raw_outputs = await _generate_detailed(engine, tuple(prompts), sampling)
                scheduler_token_chunks = _backend_scheduler_token_chunks(engine)
                direct_backend_groups = (
                    _backend_generation_group_shape(engine, input_rows=len(prompts)),
                )
                generation_shape = _queue_generation_shape(
                    route=generation_route,
                    route_cap=generation_batcher._route_request_cap(generation_route),
                    route_cap_applied=False,
                    queue_group_id=f"direct-{uuid.uuid4().hex}",
                    queue_request_count=1,
                    queue_prompt_rows=len(prompts),
                    item_index=0,
                    item_prompt_offset=0,
                    item_prompt_rows=len(prompts),
                    backend_groups=direct_backend_groups,
                    route_decision=generation_route_decision,
                )
            else:
                queued_result = await generation_batcher.submit(
                    tuple(prompts),
                    sampling,
                    detailed=True,
                    include_batch_metadata=True,
                    route=generation_route,
                    route_decision=generation_route_decision,
                    error_extra=route_rejection_extra(
                        requested_model=request.model,
                        reason="engine_busy",
                        engine=engine,
                        details={
                            "overload_source": "generation_queue_cap",
                            "max_queued_requests": config.max_queued_requests,
                        },
                    ),
                )
                if isinstance(queued_result, _QueuedBatchResult):
                    raw_outputs = queued_result.outputs
                    scheduler_token_chunks = queued_result.scheduler_token_chunks
                    generation_shape = queued_result.generation_shape
                else:
                    raw_outputs = queued_result
                    scheduler_token_chunks = None
        except GenerationAdmissionRejected as exc:
            openai_exc = _generation_admission_http_error(
                exc,
                retry_after_seconds=config.queue_retry_after_seconds,
                error_extra=route_rejection_extra(
                    requested_model=request.model,
                    reason="engine_busy",
                    engine=getattr(app.state, "hipengine_llm", None),
                    details={"overload_source": "kv_pool_capacity"},
                ),
            )
            _record_openai_error(app.state.hipengine_server_metrics, openai_exc)
            raise openai_exc from exc
        except GenerationDeadlineExceeded as exc:
            raise _deadline_exceeded_error(exc.finish_details) from exc
        except GenerationCancelled as exc:
            raise _request_cancelled_error(exc.finish_details) from exc
        except OpenAIHTTPError as exc:
            _record_openai_error(app.state.hipengine_server_metrics, exc)
            raise
        except NotImplementedError as exc:
            app.state.hipengine_server_metrics.record_failure()
            raise OpenAIHTTPError(400, str(exc), code="unsupported_parameter") from exc
        except ValueError as exc:
            app.state.hipengine_server_metrics.record_failure()
            raise OpenAIHTTPError(400, str(exc), code="invalid_request") from exc
        except Exception as exc:  # pragma: no cover - exercised by real runtime failures
            app.state.hipengine_server_metrics.record_failure()
            raise OpenAIHTTPError(
                500,
                f"generation failed: {exc}",
                error_type="server_error",
                code="generation_failed",
            ) from exc

        details = [_coerce_generation_output(item) for item in raw_outputs]
        deadline_detail = _deadline_detail_from_outputs(details)
        if deadline_detail is not None:
            raise _deadline_exceeded_error(deadline_detail)
        outputs = [item.text for item in details]
        if len(outputs) != len(prompts):
            app.state.hipengine_server_metrics.record_failure()
            raise OpenAIHTTPError(
                500,
                f"generator returned {len(outputs)} outputs for {len(prompts)} prompts",
                error_type="server_error",
                code="bad_generator_output",
            )
        if _request_logprobs_enabled(request):
            _validate_logprob_details(details, outputs)
        batch = _GeneratedBatch(
            outputs=outputs,
            usage=_usage(engine, prompts, outputs, details=details),
            details=details,
            prepared_prompts=tuple(prompts),
            scheduler_token_chunks=scheduler_token_chunks,
            generation_shape=generation_shape,
        )
        app.state.hipengine_server_metrics.record_success(batch.usage)
        return batch

    async def generate_with_request_control(
        prompts: Sequence[PromptInput],
        request: CompletionRequest | ChatCompletionRequest,
        control: _RequestControl | None = None,
        *,
        fit_context_extra: Mapping[str, Any] | None = None,
        prepared_thinking: _ThinkingControl | None = None,
        resident_session_key: str | None = None,
    ) -> _GeneratedBatch:
        active_control = control or _request_control(config, request)
        try:
            return await _await_with_request_control(
                generate(
                    prompts,
                    request,
                    deadline_at=active_control.deadline_at,
                    cancellation_token=active_control.cancellation_token,
                    fit_context_extra=fit_context_extra,
                    prepared_thinking=prepared_thinking,
                    resident_session_key=resident_session_key,
                ),
                active_control,
            )
        except OpenAIHTTPError as exc:
            if exc.code in {"deadline_exceeded", "cancelled"}:
                _record_openai_error(app.state.hipengine_server_metrics, exc)
            raise

    async def stream_completion_one(
        prompt: PromptInput,
        request: CompletionRequest,
        control: _RequestControl,
        raw_request: Request,
    ) -> AsyncIterator[str]:
        response_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        include_hipengine = _stream_include_hipengine(request)
        stream_started_at = _StreamTimingTracker.start() if include_hipengine else time.perf_counter()
        routing_metadata = (
            _routing_response_metadata(
                config,
                requested_model=request.model,
                engine=getattr(app.state, "hipengine_llm", None),
            )
            if include_hipengine
            else None
        )
        async def write_error_artifact(
            message: str,
            *,
            status_code: int,
            code: str | None,
            param: str | None = None,
            error_type: str = "server_error",
            finish_details: Mapping[str, Any] | None = None,
            extra: Mapping[str, Any] | None = None,
        ) -> None:
            await _maybe_write_stream_error_replay_artifact(
                config,
                raw_request,
                message=message,
                status_code=status_code,
                code=code,
                param=param,
                error_type=error_type,
                finish_details=finish_details,
                extra=extra,
                engine=getattr(app.state, "hipengine_llm", None),
            )

        full_text: list[str] = []
        last_stream_chunk: GenerationStreamChunk | None = None
        try:
            _validate_generation_request(
                config,
                request,
                engine=getattr(app.state, "hipengine_llm", None),
                route_unsupported_grammar=True,
            )

            async def prepare_stream() -> tuple[
                Any,
                SamplingParams,
                str,
                dict[str, Any] | None,
                PromptInput,
            ]:
                async with session_lock:
                    engine = get_llm()
                    generation_prompt = _prepare_prompt_input(engine, prompt)
                    admission_started = time.perf_counter()
                    sampling = sampling_params(
                        request,
                        (generation_prompt,),
                        engine,
                        deadline_at=control.deadline_at,
                        cancellation_token=control.cancellation_token,
                    )
                    generation_route, generation_route_decision = (
                        _generation_route_for_request(
                            config,
                            request,
                            engine=engine,
                            sampling=sampling,
                            prompts=(generation_prompt,),
                        )
                    )
                    generation_route, generation_route_decision = (
                        _resolve_realized_generation_route(
                            generation_route,
                            group_rows=1,
                            sampling=sampling,
                            precomputed_decision=generation_route_decision,
                        )
                    )
                    await ensure_resident_context(
                        engine,
                        preparation_sampling(request),
                        phase="preparation",
                    )
                    _validate_context_budget(
                        effective_max_context_tokens(engine),
                        engine,
                        (generation_prompt,),
                        sampling,
                        error_extra={
                            "hipengine": {
                                "routing": _routing_rejection_metadata(
                                    config,
                                    requested_model=request.model,
                                    reason="context_overflow",
                                    engine=engine,
                                )
                            }
                        },
                    )
                    generation_prompt = _with_admission_prepare_ms(
                        (generation_prompt,),
                        (time.perf_counter() - admission_started) * 1_000.0,
                    )[0]
                    return (
                        engine,
                        sampling,
                        generation_route,
                        generation_route_decision,
                        generation_prompt,
                    )

            (
                engine,
                sampling,
                generation_route,
                generation_route_decision,
                generation_prompt,
            ) = await _await_with_request_control(
                prepare_stream(),
                control,
            )
            if include_hipengine:
                routing_metadata = _routing_response_metadata(
                    config,
                    requested_model=request.model,
                    engine=engine,
                )
            token_accounting = _StreamTokenAccounting.for_engine(engine) if include_hipengine else None
            async for token in _iterate_with_request_control(
                generation_batcher.stream(
                    (generation_prompt,),
                    sampling,
                    route=generation_route,
                    route_decision=generation_route_decision,
                    error_extra=route_rejection_extra(
                        requested_model=request.model,
                        reason="engine_busy",
                        engine=engine,
                        details={
                            "overload_source": "generation_queue_cap",
                            "max_queued_requests": config.max_queued_requests,
                        },
                    ),
                ),
                control,
            ):
                stream_chunk = _coerce_generation_stream_chunk(token)
                last_stream_chunk = stream_chunk
                text = stream_chunk.text
                if not text:
                    continue
                full_text.append(text)
                logprobs = (
                    _completion_stream_logprobs(stream_chunk)
                    if _request_logprobs_enabled(request)
                    else None
                )
                token_payload = (
                    token_accounting.observe_stream_chunk("answer", stream_chunk)
                    if token_accounting is not None
                    else None
                )
                yield _completion_stream_delta(
                    response_id,
                    created,
                    config.model_id,
                    text,
                    logprobs=logprobs,
                    tokens=token_payload,
                    stream_chunk=stream_chunk,
                    include_hipengine=include_hipengine,
                    stream_started_at=stream_started_at,
                    routing=routing_metadata,
                )
        except GenerationDeadlineExceeded as exc:
            openai_exc = _deadline_exceeded_error(exc.finish_details)
            _record_openai_error(app.state.hipengine_server_metrics, openai_exc)
            _log_stream_failure(
                "POST /v1/completions stream",
                status_code=openai_exc.status_code,
                code=openai_exc.code,
                param=openai_exc.param,
                message=openai_exc.message,
            )
            await write_error_artifact(
                openai_exc.message,
                status_code=openai_exc.status_code,
                code=openai_exc.code,
                param=openai_exc.param,
                error_type=openai_exc.error_type,
                finish_details=openai_exc.finish_details,
            )
            yield _completion_stream_error(
                response_id,
                created,
                config.model_id,
                openai_exc.message,
                status_code=openai_exc.status_code,
                code=openai_exc.code,
                param=openai_exc.param,
                error_type=openai_exc.error_type,
                finish_details=openai_exc.finish_details,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
            )
            yield "data: [DONE]\n\n"
            return
        except GenerationCancelled as exc:
            openai_exc = _request_cancelled_error(exc.finish_details)
            _record_openai_error(app.state.hipengine_server_metrics, openai_exc)
            _log_stream_failure(
                "POST /v1/completions stream",
                status_code=openai_exc.status_code,
                code=openai_exc.code,
                param=openai_exc.param,
                message=openai_exc.message,
            )
            await write_error_artifact(
                openai_exc.message,
                status_code=openai_exc.status_code,
                code=openai_exc.code,
                param=openai_exc.param,
                error_type=openai_exc.error_type,
                finish_details=openai_exc.finish_details,
            )
            yield _completion_stream_error(
                response_id,
                created,
                config.model_id,
                openai_exc.message,
                status_code=openai_exc.status_code,
                code=openai_exc.code,
                param=openai_exc.param,
                error_type=openai_exc.error_type,
                finish_details=openai_exc.finish_details,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
            )
            yield "data: [DONE]\n\n"
            return
        except OpenAIHTTPError as exc:
            _record_openai_error(app.state.hipengine_server_metrics, exc)
            _log_stream_failure(
                "POST /v1/completions stream",
                status_code=exc.status_code,
                code=exc.code,
                param=exc.param,
                message=exc.message,
            )
            await write_error_artifact(
                exc.message,
                status_code=exc.status_code,
                code=exc.code,
                param=exc.param,
                error_type=exc.error_type,
                finish_details=exc.finish_details,
                extra=exc.extra,
            )
            yield _completion_stream_error(
                response_id,
                created,
                config.model_id,
                exc.message,
                status_code=exc.status_code,
                code=exc.code,
                param=exc.param,
                error_type=exc.error_type,
                finish_details=exc.finish_details,
                extra=exc.extra,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
            )
            yield "data: [DONE]\n\n"
            return
        except NotImplementedError as exc:
            app.state.hipengine_server_metrics.record_failure()
            message = str(exc)
            _log_stream_failure(
                "POST /v1/completions stream",
                status_code=400,
                code="unsupported_parameter",
                param=None,
                message=message,
            )
            await write_error_artifact(
                message,
                status_code=400,
                code="unsupported_parameter",
                error_type="invalid_request_error",
            )
            yield _completion_stream_error(
                response_id,
                created,
                config.model_id,
                message,
                status_code=400,
                code="unsupported_parameter",
                error_type="invalid_request_error",
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
            )
            yield "data: [DONE]\n\n"
            return
        except ValueError as exc:
            app.state.hipengine_server_metrics.record_failure()
            message = str(exc)
            _log_stream_failure(
                "POST /v1/completions stream",
                status_code=400,
                code="invalid_request",
                param=None,
                message=message,
            )
            await write_error_artifact(
                message,
                status_code=400,
                code="invalid_request",
                error_type="invalid_request_error",
            )
            yield _completion_stream_error(
                response_id,
                created,
                config.model_id,
                message,
                status_code=400,
                code="invalid_request",
                error_type="invalid_request_error",
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
            )
            yield "data: [DONE]\n\n"
            return
        except Exception as exc:  # pragma: no cover - real runtime failures
            app.state.hipengine_server_metrics.record_failure()
            message = f"generation failed: {exc}"
            _log_stream_failure(
                "POST /v1/completions stream",
                status_code=500,
                code="generation_failed",
                param=None,
                message=message,
            )
            await write_error_artifact(
                message,
                status_code=500,
                code="generation_failed",
            )
            yield _completion_stream_error(
                response_id,
                created,
                config.model_id,
                message,
                status_code=500,
                code="generation_failed",
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
            )
            yield "data: [DONE]\n\n"
            return

        raw_text = "".join(full_text)
        text, finish_reason = _apply_stop(raw_text, request.stop)
        server_stop = text != raw_text
        backend_detail = None if server_stop else _output_from_stream_chunk(last_stream_chunk, raw_text)
        done_stream_chunk = None if server_stop else last_stream_chunk
        finish_reason = _finish_reason_for_output(backend_detail, finish_reason, server_stop=server_stop)
        usage = _usage(
            engine,
            (generation_prompt,),
            [text],
            details=None if backend_detail is None else (backend_detail,),
            prefer_telemetry_counts=True,
        )
        app.state.hipengine_server_metrics.record_success(usage)
        cache_action = _session_cache_action(request)
        final_tokens = (
            _stream_usage_token_payload(usage, token_accounting)
            if include_hipengine and token_accounting is not None
            else None
        )
        final_kv_pool = _kv_pool_stream_payload(engine) if include_hipengine else None
        yield _completion_stream_done(
            response_id,
            created,
            config.model_id,
            finish_reason,
            finish_details=_finish_details_payload(
                backend_detail,
                finish_reason,
                cache_action=cache_action,
            ),
            tokens=final_tokens,
            stream_chunk=done_stream_chunk,
            include_hipengine=include_hipengine,
            extra_hipengine=_stream_speculative_mtp_metadata(
                generation_route,
                backend_detail,
                route_decision=generation_route_decision,
                thinking_policy=_request_speculative_mtp_thinking(config, request),
            ),
            stream_started_at=stream_started_at,
            routing=routing_metadata,
            kv_pool=final_kv_pool,
        )
        if _stream_include_usage(request):
            yield _completion_stream_usage(
                response_id,
                created,
                config.model_id,
                usage,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
                kv_pool=final_kv_pool,
            )
        yield "data: [DONE]\n\n"

    def readiness_payload() -> dict[str, Any]:
        engine = getattr(app.state, "hipengine_llm", None)
        model_identity = _server_model_identity(config, engine)
        readiness: _ReadinessState = app.state.hipengine_readiness
        effective_context = getattr(app.state, "hipengine_effective_max_context_tokens", None)
        live_snapshot = _live_loop_snapshot(engine)
        graph = _graph_bucket_metric_values(engine, live_snapshot=live_snapshot)
        pool = _pool_metric_values(engine, live_snapshot=live_snapshot)
        prefix_cache = _prefix_cache_metric_values(live_snapshot)
        diagnostics: list[str] = []
        if not readiness.ready:
            diagnostics.append("server startup is not ready; check startup.error and server logs")
            if readiness.startup_error:
                diagnostics.append(
                    f"{readiness.startup_error.get('message')}: {readiness.startup_error.get('guidance')}"
                )
        return {
            "object": "hipengine.readiness",
            "status": readiness.status,
            "ready": bool(readiness.ready),
            "diagnostics": diagnostics,
            "model": {
                **model_identity,
                "loaded": bool(readiness.model_loaded),
                "loaded_model_count": 0 if engine is None else 1,
                "kv_capability": _kv_capability_provenance(engine),
            },
            "startup": {
                "eager_load": bool(readiness.eager_load),
                "warmup_complete": bool(readiness.warmup_complete),
                "last_timings_s": dict(readiness.last_startup_timings),
                "checks": dict(readiness.last_startup_checks),
                "memory": dict(readiness.last_startup_memory),
                "error": readiness.startup_error,
            },
            "context": {
                "configured_max_context_tokens": configured_max_context_tokens(),
                "effective_max_context_tokens": (
                    None if effective_context is None else int(effective_context)
                ),
                "chat_default_max_tokens": config.chat_default_max_tokens,
            },
            "kv_capacity": {
                "storage": config.kv_storage,
                "scale_dtype": config.kv_scale_dtype,
                "scale_granularity": config.kv_scale_granularity,
                "estimate": _kv_capacity_estimate_payload(engine),
                "pool": pool,
            },
            "prefix_cache": prefix_cache,
            "graph_cache": {
                "entries": graph["entries"],
                "hits": graph["hits"],
                "misses": graph["misses"],
                "replay_hit_rate": graph["replay_hit_rate"],
            },
            "device": _selected_device_payload(config, engine),
            "queue": {
                "depth": generation_batcher.queue_depth(),
                "max_depth": generation_batcher.max_queue_size(),
                "worker_active": generation_batcher.active(),
                "active_requests": generation_batcher.active_requests(),
                "max_active_requests": generation_batcher.max_active_requests(),
                "scheduler_fairness": _scheduler_fairness_capability(
                    continuous_decode=_engine_supports_controlled_streaming(engine)
                ),
                "batch_window_ms": float(config.generation_batch_window_ms),
            },
            "sessions": chat_session_summary(),
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "object": "hipengine.health",
            "status": "ok",
            "model": config.model_id,
        }

    @app.get("/ready")
    async def ready() -> JSONResponse:
        payload = readiness_payload()
        return JSONResponse(status_code=200 if payload["ready"] else 503, content=payload)

    @app.get("/v1/hipengine/sessions")
    async def list_sessions(_auth: None = Depends(require_auth)) -> dict[str, Any]:
        async with chat_session_lock:
            session_payloads = [
                chat_session_metadata(record)
                for record in sorted(
                    chat_sessions.values(),
                    key=lambda item: (-item.updated, item.id),
                )
            ]
        async with continuation_lock:
            cleanup_expired_continuations()
            active_continuations = len(continuations)
        return {
            "object": "hipengine.sessions",
            "storage": "app_local_transcript",
            "resident_state_reuse": False,
            "includes_transcript": False,
            "active": len(session_payloads),
            "pending_creations": len(chat_session_pending),
            "max_active": None if config.max_chat_sessions is None else int(config.max_chat_sessions),
            "sessions": session_payloads,
            "continuations": {
                "active": active_continuations,
                "ttl_seconds": _CONTINUATION_TTL_SECONDS,
            },
        }

    @app.delete("/v1/hipengine/sessions/{session_id}")
    async def delete_session(session_id: str, _auth: None = Depends(require_auth)) -> dict[str, Any]:
        async with chat_session_lock:
            deleted = chat_sessions.pop(session_id, None) is not None
        return {
            "object": "hipengine.session.deleted",
            "id": session_id,
            "deleted": deleted,
            "storage": "app_local_transcript",
            "resident_state_reuse": False,
        }

    @app.post("/v1/hipengine/sessions/{session_id}/fork")
    async def fork_session(
        session_id: str,
        request: SessionForkRequest,
        _auth: None = Depends(require_auth),
    ) -> dict[str, Any]:
        target_id = str(request.id).strip()
        if not target_id:
            raise OpenAIHTTPError(
                400,
                "target session id must be a non-empty string",
                code="invalid_request",
                param="id",
            )
        if target_id == session_id:
            raise OpenAIHTTPError(
                400,
                "target session id must differ from source session id",
                code="invalid_request",
                param="id",
            )
        async with chat_session_lock:
            source = chat_sessions.get(session_id)
            if source is None:
                raise OpenAIHTTPError(
                    404,
                    "chat session does not exist",
                    code="invalid_request",
                    param="session_id",
                )
            if target_id in chat_sessions:
                raise OpenAIHTTPError(
                    400,
                    "target chat session already exists",
                    code="invalid_request",
                    param="id",
                )
            if (
                config.max_chat_sessions is not None
                and len(chat_sessions) >= int(config.max_chat_sessions)
            ):
                exc = OpenAIHTTPError(
                    429,
                    "chat session limit is full",
                    error_type="rate_limit_error",
                    code="engine_busy",
                    extra=route_rejection_extra(
                        requested_model=None,
                        reason="engine_busy",
                        details={
                            "overload_source": "chat_session_cap",
                            "max_active_chat_sessions": int(config.max_chat_sessions),
                        },
                    ),
                    headers={"Retry-After": str(config.queue_retry_after_seconds)},
                )
                _record_openai_error(app.state.hipengine_server_metrics, exc)
                raise exc
            now = time.time()
            forked = _ChatSessionRecord(
                id=target_id,
                messages=tuple(_chat_session_message_copy(message) for message in source.messages),
                created=now,
                updated=now,
            )
            chat_sessions[target_id] = forked
        return {
            "object": "hipengine.session.forked",
            "source_id": session_id,
            "id": target_id,
            "forked": True,
            "storage": "app_local_transcript",
            "resident_state_reuse": False,
            "message_count": len(forked.messages),
        }

    @app.post("/v1/hipengine/sessions/{session_id}/rollback")
    async def rollback_session(
        session_id: str,
        request: SessionRollbackRequest,
        _auth: None = Depends(require_auth),
    ) -> dict[str, Any]:
        target_count = int(request.message_count)
        async with chat_session_lock:
            record = chat_sessions.get(session_id)
            if record is None:
                raise OpenAIHTTPError(
                    404,
                    "chat session does not exist",
                    code="invalid_request",
                    param="session_id",
                )
            previous_count = len(record.messages)
            if target_count > previous_count:
                raise OpenAIHTTPError(
                    400,
                    "rollback message_count cannot exceed current session message_count",
                    code="invalid_request",
                    param="message_count",
                )
            rolled_back = target_count != previous_count
            if rolled_back:
                now = time.time()
                record = _ChatSessionRecord(
                    id=session_id,
                    messages=tuple(
                        _chat_session_message_copy(message) for message in record.messages[:target_count]
                    ),
                    created=record.created,
                    updated=now,
                )
                chat_sessions[session_id] = record
        return {
            "object": "hipengine.session.rolled_back",
            "id": session_id,
            "rolled_back": rolled_back,
            "storage": "app_local_transcript",
            "resident_state_reuse": False,
            "previous_message_count": previous_count,
            "message_count": target_count,
        }

    @app.get("/v1/hipengine/sessions/{session_id}/snapshot")
    async def export_session_snapshot(session_id: str, _auth: None = Depends(require_auth)) -> dict[str, Any]:
        async with chat_session_lock:
            record = chat_sessions.get(session_id)
            if record is None:
                raise OpenAIHTTPError(
                    404,
                    "chat session does not exist",
                    code="invalid_request",
                    param="session_id",
                )
            return chat_session_snapshot(record)

    @app.post("/v1/hipengine/sessions/{session_id}/snapshot")
    async def restore_session_snapshot(
        session_id: str,
        snapshot: dict[str, Any],
        _auth: None = Depends(require_auth),
    ) -> dict[str, Any]:
        record = restore_chat_session_from_snapshot(session_id, snapshot)
        async with chat_session_lock:
            is_new = session_id not in chat_sessions
            if (
                is_new
                and config.max_chat_sessions is not None
                and len(chat_sessions) >= int(config.max_chat_sessions)
            ):
                exc = OpenAIHTTPError(
                    429,
                    "chat session limit is full",
                    error_type="rate_limit_error",
                    code="engine_busy",
                    extra=route_rejection_extra(
                        requested_model=None,
                        reason="engine_busy",
                        details={
                            "overload_source": "chat_session_cap",
                            "max_active_chat_sessions": int(config.max_chat_sessions),
                        },
                    ),
                    headers={"Retry-After": str(config.queue_retry_after_seconds)},
                )
                _record_openai_error(app.state.hipengine_server_metrics, exc)
                raise exc
            chat_sessions[session_id] = record
        return {
            "object": "hipengine.session.restored",
            "id": session_id,
            "restored": True,
            "storage": "app_local_transcript",
            "resident_state_reuse": False,
            "message_count": len(record.messages),
        }

    if metrics_mode == "prometheus":

        @app.get("/metrics", response_class=PlainTextResponse)
        async def prometheus_metrics() -> PlainTextResponse:
            return PlainTextResponse(
                _render_prometheus_metrics(
                    app.state.hipengine_server_metrics,
                    engine=getattr(app.state, "hipengine_llm", None),
                    generation_batcher=getattr(app.state, "hipengine_generation_batcher", None),
                    chat_sessions=chat_sessions,
                    pending_chat_sessions=chat_session_pending,
                    max_chat_sessions=config.max_chat_sessions,
                    config=config,
                ),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

    @app.get("/v1/models")
    async def list_models(_auth: None = Depends(require_auth)) -> dict[str, Any]:
        engine = getattr(app.state, "hipengine_llm", None)
        model_identity = _server_model_identity(config, engine)
        readiness: _ReadinessState = app.state.hipengine_readiness
        configured_context = configured_max_context_tokens()
        effective_context = getattr(app.state, "hipengine_effective_max_context_tokens", None)
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model_id,
                    "object": "model",
                    "created": config.created,
                    "owned_by": "hipengine",
                    "hipengine": {
                        "path": config.model,
                        "backend": model_identity["backend"],
                        "quant": model_identity["quant"],
                        "execution_profile": _server_execution_profile(config, engine),
                        "loaded": bool(readiness.model_loaded),
                        "resident_context": True,
                        "context": {
                            "configured_max_context_tokens": configured_context,
                            "effective_max_context_tokens": (
                                None if effective_context is None else int(effective_context)
                            ),
                            "chat_default_max_tokens": config.chat_default_max_tokens,
                        },
                        "kv_capacity": {
                            "storage": config.kv_storage,
                            "scale_dtype": config.kv_scale_dtype,
                            "scale_granularity": config.kv_scale_granularity,
                            "estimate": _kv_capacity_estimate_payload(engine),
                            "capability": _kv_capability_provenance(engine),
                        },
                        "capabilities": _model_capability_summary(config, engine=engine),
                        "capabilities_url": "/v1/hipengine/capabilities",
                        "routing": {
                            "loaded_model_count": 0 if engine is None else 1,
                            "multiple_models": False,
                        },
                    },
                }
            ],
        }

    @app.get("/v1/hipengine/capabilities")
    async def capabilities(_auth: None = Depends(require_auth)) -> dict[str, Any]:
        engine = getattr(app.state, "hipengine_llm", None)
        model_identity = _server_model_identity(config, engine)
        tokenizer_caps = _tokenizer_capability_flags(engine)
        chat_protocol_target = _model_chat_protocol_target(engine)
        chat_template_family = str(
            getattr(chat_protocol_target, "chat_template_family", "qwen")
        )
        reasoning_parser_name = str(
            getattr(chat_protocol_target, "reasoning_parser_name", "qwen_tags")
        )
        chat_tool_parser = getattr(chat_protocol_target, "chat_tool_parser", None)
        stream_logprobs = _engine_supports_stream_logprobs(engine)
        configured_context = configured_max_context_tokens()
        cached_context = getattr(app.state, "hipengine_effective_max_context_tokens", None)
        effective_context = configured_context if configured_context is not None else cached_context
        gguf_native_candidate = Path(str(config.model)).suffix.lower() == ".gguf"
        native_gpu_capability = {
            "enabled": _env_flag(
                "HIPENGINE_QWEN35_NATIVE_SAMPLER",
                default=not gguf_native_candidate,
            ),
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
        if gguf_native_candidate:
            native_gpu_capability.update(
                {
                    "enable_env": "HIPENGINE_QWEN35_NATIVE_SAMPLER=1",
                    "scope": "gguf_resident_c1_and_dense_compatible_c_gt_1",
                    "c_gt_1": (
                        "single_batched_launch_when_compatible_else_native_rows"
                    ),
                    "true_batched_c_gt_1": True,
                    "default_path": False,
                    "unsupported": [
                        capability
                        for capability in NATIVE_GPU_SAMPLER_UNSUPPORTED_CAPABILITIES
                        if capability not in {"true_batched_c_gt_1", "gguf"}
                    ],
                }
            )
        return {
            "object": "hipengine.capabilities",
            "model": {
                "id": model_identity["id"],
                "path": config.model,
                "backend": model_identity["backend"],
                "quant": model_identity["quant"],
                "kv_capability": _kv_capability_provenance(engine),
                "execution_profile": _server_execution_profile(config, engine),
            },
            "context": {
                "configured_max_context_tokens": configured_context,
                "effective_max_context_tokens": None if effective_context is None else int(effective_context),
                "chat_default_max_tokens": config.chat_default_max_tokens,
                "chat_default_mode": "auto" if config.chat_default_max_tokens is None else "bounded",
                "overflow_policy_field": "session.context_overflow_policy",
                "default_overflow_policy": "reject",
                "overflow_policies": list(_SESSION_CONTEXT_OVERFLOW_POLICIES),
            },
            "tokenizer": {
                "tokenize": tokenizer_caps["tokenize"],
                "detokenize": tokenizer_caps["detokenize"],
                "count_tokens": tokenizer_caps["count_tokens"],
                "name": None if engine is None else type(engine).__name__,
            },
            "chat_template": {
                "family": chat_template_family,
                "reasoning_parser": reasoning_parser_name,
                "reasoning_tags": True,
                "tool_call_tags": True,
            },
            "features": {
                "completions": True,
                "chat_completions": True,
                "streaming": True,
                "stream_options": {
                    "include_usage": True,
                    "include_hipengine": True,
                },
                "stream_metadata": {
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
                    "choice_generated_token_ids": (
                        "done_event_when_backend_supplies_exact_ids"
                    ),
                    "choice_token_accounting": tokenizer_caps["count_tokens"],
                    "choice_token_accounting_scopes": (
                        ["live_delta", "buffered_delta", "final_choice"]
                        if tokenizer_caps["count_tokens"]
                        else []
                    ),
                    "choice_decode_state": tokenizer_caps["count_tokens"],
                    "choice_decode_state_scopes": (
                        ["live_delta", "buffered_delta", "final_choice"]
                        if tokenizer_caps["count_tokens"]
                        else []
                    ),
                    "backend_telemetry_scopes": [
                        "live_chunk",
                        "buffered_delta_safe_decode_state",
                        "buffered_done",
                    ],
                    "live_many_chunks": {
                        "available": _engine_supports_stream_many(engine),
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
                },
                "choice_telemetry": _choice_telemetry_capability(),
                "generation_shape": {
                    "non_streaming": True,
                    "response_path": "hipengine.generation_shape",
                    "schema_version": 1,
                    "route_cap_scope": "queue_requests",
                    "separates": [
                        "automatic_route_decision",
                        "route_cap",
                        "queue_group_requests",
                        "queue_group_prompt_rows",
                        "backend_group_rows",
                        "verifier_rows",
                    ],
                    "streaming": False,
                },
                "exact_token_prompts": {
                    "completions": True,
                    "request_forms": ["token_ids", "token_id_rows"],
                    "direct_api": "LLM.generate_detailed/stream_detailed(token_id_rows, sampling_params)",
                    "streaming": True,
                    "streaming_mode": {
                        "single_row": "live",
                        "multiple_rows": "buffered",
                    },
                    "echo": False,
                    "response_identity": "hipengine.prompt_token_accounting",
                    "generated_id_oracle": "hipengine.token_accounting.choice_generated_token_ids",
                },
                "structured_outputs": _structured_outputs_capability(
                    tokenizer_backed=(
                        tokenizer_caps["tokenize"] and tokenizer_caps["detokenize"]
                    )
                ),
                "grammars": _grammar_capability(),
                "finish_details": True,
                "token_diagnostics": {
                    "tokenize": tokenizer_caps["tokenize"],
                    "detokenize": tokenizer_caps["detokenize"],
                    "count_tokens": tokenizer_caps["count_tokens"],
                    "fit_context": tokenizer_caps["count_tokens"],
                    "session_aware_chat": tokenizer_caps["count_tokens"],
                },
                "tools": _tools_capability(
                    tokenizer_backed=tokenizer_caps["tokenize"],
                    strict_decoding_backed=(
                        tokenizer_caps["tokenize"] and tokenizer_caps["detokenize"]
                    ),
                    tool_parser=chat_tool_parser,
                ),
                "reasoning_controls": {
                    "enabled": True,
                    "fields": _reasoning_control_fields(),
                    "budget_policy": "prompt_hint_plus_tokenized_soft_and_hard_close",
                    "token_budget": tokenizer_caps["tokenize"],
                    "token_budget_enforced": tokenizer_caps["tokenize"],
                    "effort_defaults": _thinking_effort_defaults_capability(),
                    "effort_default_clamp": "request_max_tokens_chat_default_or_remaining_context",
                    "hard_close_validation": True,
                    "hard_close_token_forcing": tokenizer_caps["tokenize"],
                    "soft_close_bias": tokenizer_caps["tokenize"],
                    "eos_suppression": tokenizer_caps["tokenize"],
                    "hard_close_marker": _THINKING_CLOSE_MARKER,
                    "diagnostic_close_token_lowering": tokenizer_caps["tokenize"],
                    "diagnostic_initial_state": tokenizer_caps["tokenize"],
                },
                "logprobs": {
                    "completions": True,
                    "chat": True,
                    "top_logprobs_max": 20,
                    "streaming": "live_chunk_metadata" if stream_logprobs else "buffered",
                    "live_chunk_metadata": stream_logprobs,
                    "live_chunk_metadata_capability": "engine.supports_stream_logprobs",
                    "chat_reasoning_private_stream_metadata": "choices[].hipengine.reasoning_logprobs",
                    "requires_backend_token_metadata": True,
                    "omission_reasons": [_LOGPROB_OMISSION_REASON, _PROMPT_LOGPROB_OMISSION_REASON],
                    "missing_backend_metadata_error": {
                        "code": "unsupported_feature",
                        "status_code": 501,
                        "param": "logprobs",
                    },
                },
                "request_timeouts": {
                    "timeout_ms": True,
                    "default_timeout_ms": config.request_timeout_ms,
                    "client_disconnect": True,
                    "cooperative_backend_deadline": True,
                    "cooperative_backend_cancel": True,
                    "preemptive_decode_cancel": False,
                },
            },
            "sampling": {
                "modes": ["greedy", "temperature"],
                "execution_modes": [
                    "greedy_fast",
                    "processed_argmax",
                    "host_logits_sample",
                    "gpu_sample",
                ],
                "parameters": [
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "repetition_penalty",
                    "presence_penalty",
                    "frequency_penalty",
                    "logit_bias",
                    "suppress_token_ids",
                    "min_tokens",
                    "eos_token_id",
                    "json_object_close_forcing",
                    "seed",
                    "n",
                    "stop",
                ],
                "native_gpu": native_gpu_capability,
                "speculative_mtp": _speculative_mtp_capability(config, engine=engine),
                "speculative": _speculative_provider_capability(config, engine=engine),
            },
            "cache": {
                "prefix_cache": prefix_cache_mode,
                "kv_storage": config.kv_storage,
                "kv_scale_dtype": config.kv_scale_dtype,
                "kv_scale_granularity": config.kv_scale_granularity,
            },
            "sessions": {
                "resident_context": True,
                "commit_policy": _session_commit_policy_capability(engine=engine),
                "continuations": _session_continuation_capability(),
                "metadata": _session_metadata_capability(config.max_chat_sessions),
            },
            "admission": _admission_capability(config, engine=engine),
            "routing": {
                "loaded_model_count": 0 if engine is None else 1,
                "multiple_models": False,
            },
            "parallelism": _parallelism_capability(),
            "errors": _error_taxonomy_manifest(),
            "unsupported_fields": _known_unsupported_fields(),
        }

    @app.post("/v1/hipengine/speculative_mtp/rollback")
    async def rollback_speculative_mtp(
        _auth: None = Depends(require_auth),
    ) -> dict[str, Any]:
        """Route every new MTP request to AR until the server restarts."""

        mtp_circuit_breaker.disable_new_mtp(reason="mtp_operator_rollback")
        engine = getattr(app.state, "hipengine_llm", None)
        if engine is not None:
            mtp_circuit_breaker.attach(engine)
        return {
            "object": "hipengine.speculative_mtp.rollback",
            "new_requests_route": _SPECULATIVE_MTP_DEFAULT_ROUTE,
            "in_flight_policy": "retire_owned_cycle_then_cancel_or_complete",
            "serving_plan": _engine_speculative_mtp_serving_capability(engine),
            "circuit_breaker": mtp_circuit_breaker.snapshot(),
        }

    @app.post("/v1/hipengine/tokenize")
    async def tokenize(request: TokenizeRequest, _auth: None = Depends(require_auth)) -> dict[str, Any]:
        engine = get_llm()
        tokenize_started = time.perf_counter()
        token_ids = await run_in_threadpool(_tokenize_text, engine, request.text)
        tokenize_ms = max(0.0, (time.perf_counter() - tokenize_started) * 1_000.0)
        return {
            "object": "hipengine.tokens",
            "text": request.text,
            "token_ids": list(token_ids),
            "token_count": len(token_ids),
            "timing": {"tokenize_ms": round(tokenize_ms, 3)},
        }

    @app.post("/v1/hipengine/detokenize")
    async def detokenize(request: DetokenizeRequest, _auth: None = Depends(require_auth)) -> dict[str, Any]:
        engine = get_llm()
        token_ids = tuple(int(token) for token in request.token_ids)
        text = await run_in_threadpool(_detokenize_ids, engine, token_ids, request.skip_special)
        return {
            "object": "hipengine.text",
            "text": text,
            "token_ids": list(token_ids),
        }

    @app.post("/v1/hipengine/count_tokens")
    async def count_tokens(request: TokenDiagnosticRequest, _auth: None = Depends(require_auth)) -> dict[str, Any]:
        engine = get_llm()
        max_context = effective_max_context_tokens(engine)
        rendered = await diagnostic_render_for_request(request, engine)
        text = str(rendered["text"])
        input_type = str(rendered["input_type"])
        tokenize_started = time.perf_counter()
        token_count = await run_in_threadpool(_count_tokens_strict, engine, text)
        tokenize_ms = max(0.0, (time.perf_counter() - tokenize_started) * 1_000.0)
        response = {
            "object": "hipengine.count_tokens",
            "input_type": input_type,
            "text": text,
            "token_count": token_count,
            "timing": {"tokenize_ms": round(tokenize_ms, 3)},
        }
        if rendered["session"] is not None:
            response["session"] = rendered["session"]
        thinking_budget = _diagnostic_thinking_budget_payload(
            config,
            request,
            engine=engine,
            max_context_tokens=max_context,
            chat_request=rendered["chat_request"],
        )
        if thinking_budget is not None:
            response["thinking_budget"] = thinking_budget
        return response

    @app.post("/v1/hipengine/fit_context")
    async def fit_context(request: FitContextRequest, _auth: None = Depends(require_auth)) -> dict[str, Any]:
        engine = get_llm()
        max_context = effective_max_context_tokens(engine)
        rendered = await diagnostic_render_for_request(request, engine, apply_context_policy=True)
        text = str(rendered["text"])
        input_type = str(rendered["input_type"])
        prompt_tokens = await run_in_threadpool(_count_tokens_strict, engine, text)
        if request.max_tokens is not None:
            effective_max_tokens = max(0, int(request.max_tokens))
        elif input_type == "chat":
            effective_max_tokens = _request_max_tokens(
                rendered["chat_request"],
                (text,),
                engine,
                max_context,
                chat_default_max_tokens=config.chat_default_max_tokens,
            )
        else:
            effective_max_tokens = 16
        fit_payload = _context_fit_payload(
            prompt_tokens=int(prompt_tokens),
            max_context_tokens=max_context,
            max_tokens=int(effective_max_tokens),
        )
        fit_context_extra = rendered.get("fit_context_extra")
        if fit_context_extra is not None:
            fit_payload.update(dict(fit_context_extra))
        response = {
            "object": "hipengine.fit_context",
            "input_type": input_type,
            "text": text,
            "requested_max_tokens": request.max_tokens,
            **fit_payload,
            "chat_default_max_tokens": config.chat_default_max_tokens if input_type == "chat" else None,
            "chat_default_mode": "auto" if config.chat_default_max_tokens is None else "bounded",
        }
        if rendered["session"] is not None:
            response["session"] = rendered["session"]
        thinking_budget = _diagnostic_thinking_budget_payload(
            config,
            request,
            engine=engine,
            max_context_tokens=max_context,
            chat_request=rendered["chat_request"],
        )
        if thinking_budget is not None:
            response["thinking_budget"] = thinking_budget
        return response

    async def account_active_stream_cancellation(
        iterator: AsyncIterator[str],
        control: _RequestControl,
    ) -> AsyncIterator[str]:
        """Record ASGI task cancellation after the active stream unwinds."""

        try:
            async for item in iterator:
                yield item
        except asyncio.CancelledError:
            finish_details = FinishDetails(reason="cancelled", cancelled=True)
            control.cancellation_token.cancel(finish_details)
            _record_openai_error(
                app.state.hipengine_server_metrics,
                _request_cancelled_error(finish_details),
            )
            raise

    @app.post("/v1/completions", response_model=None)
    async def completions(
        request: CompletionRequest,
        raw_request: Request,
        auth_principal: str = Depends(require_auth),
    ) -> dict[str, Any] | StreamingResponse:
        _validate_model(config, request.model, engine=getattr(app.state, "hipengine_llm", None))
        _validate_generation_request(
            config,
            request,
            engine=getattr(app.state, "hipengine_llm", None),
            route_unsupported_grammar=True,
        )
        _validate_continuation_resume_request(request)
        continuation = await pop_continuation(
            request,
            endpoint="completion",
            auth_principal=auth_principal,
        )
        _apply_continuation_defaults(request, continuation)
        prompts = continuation.resume_prompts() if continuation is not None else _normalize_prompts(request.prompt)
        n = _request_n(request)
        expanded_prompts = _expand_prompts_for_n(prompts, n)
        control = _request_control(config, request, raw_request)
        live_completion_logprobs = _engine_supports_stream_logprobs(getattr(app.state, "hipengine_llm", None))
        if (
            request.stream
            and len(expanded_prompts) == 1
            and not request.echo
            and (not _request_logprobs_enabled(request) or live_completion_logprobs)
            and not _structured_result_validation(request)
        ):
            # StreamingResponse owns the ASGI receive channel and cancels its
            # body iterator on disconnect. Avoid a competing receive through
            # Request.is_disconnected() while that response is active.
            control = replace(control, disconnected=None)
            return StreamingResponse(
                account_active_stream_cancellation(
                    stream_completion_one(expanded_prompts[0], request, control, raw_request),
                    control,
                ),
                media_type="text/event-stream",
            )
        batch = await generate_with_request_control(expanded_prompts, request, control)
        response_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        choices = []
        final_texts: list[str] = []
        visible_generated_texts: list[str] = []
        cache_action = _session_cache_action(request)
        for index, (prompt, output, detail) in enumerate(zip(expanded_prompts, batch.outputs, batch.details, strict=True)):
            previous_text = "" if continuation is None else continuation.generated_texts[index]
            output = f"{previous_text}{output}"
            generated_text, finish_reason = _apply_stop(output, request.stop)
            server_stop = generated_text != output
            finish_reason = _finish_reason_for_output(detail, finish_reason, server_stop=server_stop)
            text = prompt + generated_text if request.echo else generated_text
            structured_failure = _structured_output_failure_reason(request, generated_text, finish_reason)
            if structured_failure is not None:
                finish_reason = "stop"
                text = ""
            structured_length_failure = _structured_length_failure_reason(
                request,
                generated_text,
                finish_reason,
            )
            final_texts.append(text)
            visible_generated_texts.append(generated_text)
            choice = {
                "text": text,
                "index": index,
                "logprobs": (
                    _completion_logprobs(detail, generated_text, echo_text=prompt if request.echo else "")
                    if request.logprobs is not None
                    else None
                ),
                "finish_reason": finish_reason,
                "finish_details": _finish_details_payload(
                    detail,
                    finish_reason,
                    reason_override=structured_failure
                    or structured_length_failure
                    or ("stop" if server_stop else None),
                    cache_action=cache_action,
                ),
            }
            _mark_structured_length_failure(request, structured_length_failure, choice["finish_details"])
            if _continuation_can_create(
                request,
                finish_reason=finish_reason,
                finish_details=choice["finish_details"],
                backend_continuation_eligible=_backend_continuation_eligible(detail),
            ):
                base_prompt = prompt if continuation is None else continuation.prompts[index]
                record = await store_continuation(
                    endpoint="completion",
                    prompts=(base_prompt,),
                    generated_texts=(generated_text,),
                    response_format=request.response_format,
                    guided_json=request.guided_json,
                    guided_regex=request.guided_regex,
                    guided_choice=request.guided_choice,
                    guided_patch=request.guided_patch,
                    guided_diff=request.guided_diff,
                    auth_principal=auth_principal,
                    session_id=_session_id(request),
                )
                _attach_continuation_metadata(choice, continuation_id=record.id)
            else:
                _mark_continuation_unavailable(choice["finish_details"])
            _attach_choice_telemetry(choice, detail)
            if n > 1:
                choice["request_id"] = _choice_request_id(response_id, index // n, index % n)
            choices.append(choice)
        response = {
            "id": response_id,
            "object": "text_completion",
            "created": created,
            "model": config.model_id,
            "choices": choices,
            "usage": batch.usage,
            "hipengine": {
                "routing": _routing_response_metadata(
                    config,
                    requested_model=request.model,
                    engine=getattr(app.state, "hipengine_llm", None),
                ),
            },
        }
        token_accounting = _exact_generated_token_accounting(
            getattr(app.state, "hipengine_llm", None),
            batch.details,
            visible_generated_texts,
        )
        if token_accounting is not None:
            response["hipengine"]["token_accounting"] = token_accounting
        prompt_token_accounting = _exact_prompt_token_accounting(batch.prepared_prompts)
        if prompt_token_accounting is not None:
            response["hipengine"]["prompt_token_accounting"] = prompt_token_accounting
        if batch.generation_shape is not None:
            response["hipengine"]["generation_shape"] = deepcopy(batch.generation_shape)
        response["hipengine"]["speculative_mtp"] = _mtp_response_summary(
            None if batch.generation_shape is None else batch.generation_shape.get("route"),
            batch.details,
            route_decision=(
                None
                if batch.generation_shape is None
                else batch.generation_shape.get("route_decision")
            ),
            thinking_policy=(
                _request_speculative_mtp_thinking(config, request)
                if batch.generation_shape is not None
                and batch.generation_shape.get("route")
                == _SPECULATIVE_MTP_BATCH_ROUTE
                else None
            ),
        )
        await _maybe_write_agentic_result_replay_artifact(
            config,
            raw_request,
            response,
            engine=getattr(app.state, "hipengine_llm", None),
        )
        if request.stream:
            include_hipengine = _stream_include_hipengine(request)
            stream_started_at = _StreamTimingTracker.start() if include_hipengine else time.perf_counter()
            engine = getattr(app.state, "hipengine_llm", None)
            token_accounting = _StreamTokenAccounting.for_engine(engine) if include_hipengine else None
            return StreamingResponse(
                _completion_stream(
                    response_id,
                    created,
                    config.model_id,
                    final_texts,
                    choices,
                    details=batch.details,
                    usage=batch.usage if _stream_include_usage(request) else None,
                    token_accounting=token_accounting,
                    include_hipengine=include_hipengine,
                    stream_started_at=stream_started_at,
                    routing=_routing_response_metadata(
                        config,
                        requested_model=request.model,
                        engine=engine,
                    )
                    if include_hipengine
                    else None,
                    kv_pool=(
                        _kv_pool_stream_payload(engine)
                        if include_hipengine
                        else None
                    ),
                    done_phase="structured" if _structured_result_validation(request) else "done",
                    scheduler_token_chunks=batch.scheduler_token_chunks,
                ),
                media_type="text/event-stream",
            )
        return response

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: ChatCompletionRequest,
        raw_request: Request,
        auth_principal: str = Depends(require_auth),
    ) -> dict[str, Any] | StreamingResponse:
        _validate_model(config, request.model, engine=getattr(app.state, "hipengine_llm", None))
        _validate_generation_request(
            config,
            request,
            engine=getattr(app.state, "hipengine_llm", None),
            route_unsupported_grammar=True,
        )
        _validate_continuation_resume_request(request)
        admitted_session_id = await reserve_chat_session_if_needed(request)
        try:
            continuation = await pop_continuation(
                request,
                endpoint="chat",
                auth_principal=auth_principal,
            )
            _apply_continuation_defaults(request, continuation)
            control = _request_control(config, request, raw_request)

            async def prepare_prompt() -> _ChatContextRender:
                async with session_lock:
                    engine = get_llm()
                    speculative_enabled, _provider, _budget = (
                        _request_speculative_provider_config(request)
                    )
                    if not speculative_enabled:
                        await ensure_resident_context(
                            engine,
                            preparation_sampling(request),
                            phase="preparation",
                        )
                    if continuation is not None:
                        continuation_prompt = continuation.resume_prompts()[0]
                        return _ChatContextRender(
                            prompt=continuation_prompt,
                            render_request=request,
                            prefix_messages=(),
                            prepared_prompt=_prepare_prompt_input(
                                engine,
                                continuation_prompt,
                            ),
                        )
                    if not request.messages:
                        raise OpenAIHTTPError(
                            400,
                            "messages must not be empty",
                            code="invalid_request",
                            param="messages",
                        )
                    return await render_chat_context_for_request(
                        request,
                        engine,
                        apply_context_policy=True,
                    )

            prepared_prompt = await _await_with_request_control(prepare_prompt(), control)
            prompt = prepared_prompt.prompt
            generation_prompt = (
                prepared_prompt.prepared_prompt
                if prepared_prompt.prepared_prompt is not None
                else _prepare_prompt_input(get_llm(), prompt)
            )
            fit_context_extra = prepared_prompt.fit_context_extra
            resident_session_key = (
                _resident_chat_session_key(auth_principal, request)
                if _request_n(request) == 1
                else None
            )
            if request.stream:
                control = replace(control, disconnected=None)
                live_chat_logprobs = (
                    bool(request.logprobs)
                    and not request.tools
                    and _engine_supports_stream_logprobs(getattr(app.state, "hipengine_llm", None))
                )
                streamer = (
                    stream_chat_completion_many
                    if _request_n(request) > 1
                    or (request.logprobs and not live_chat_logprobs)
                    or _structured_result_validation(request)
                    else stream_chat_completion
                )
                return StreamingResponse(
                    account_active_stream_cancellation(
                        streamer(
                            prompt,
                            generation_prompt,
                            prepared_prompt.thinking,
                            resident_session_key,
                            request,
                            control,
                            raw_request,
                        ),
                        control,
                    ),
                    media_type="text/event-stream",
                )
            n = _request_n(request)
            prompts = tuple(generation_prompt for _ in range(n))
            try:
                batch = await generate_with_request_control(
                    prompts,
                    request,
                    control,
                    fit_context_extra=fit_context_extra,
                    prepared_thinking=prepared_prompt.thinking,
                    resident_session_key=resident_session_key,
                )
            except OpenAIHTTPError as exc:
                await commit_chat_session_error(
                    request,
                    request_messages=request.messages,
                    exc=exc,
                )
                raise
            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            choices = []
            visible_generated_texts: list[str] = []
            requested_cache_action = _session_cache_action(request)
            reasoning_initially_open = _reasoning_initially_open_for_prompt(
                getattr(app.state, "hipengine_llm", None),
                generation_prompt,
            )
            for index, (output, detail) in enumerate(zip(batch.outputs, batch.details, strict=True)):
                previous_text = "" if continuation is None else continuation.generated_texts[index]
                output = f"{previous_text}{output}"
                text, finish_reason = _apply_stop(output, request.stop)
                server_stop = text != output
                raw_parsed = _parse_chat_tool_calls_for_engine(
                    getattr(app.state, "hipengine_llm", None),
                    text,
                    tools=request.tools,
                )
                tool_validation = _validate_chat_tool_result(request, raw_parsed, text)
                parsed = replace(
                    tool_validation.parsed,
                    reasoning_initially_open=reasoning_initially_open,
                )
                message, parsed_finish_reason = _chat_message_from_parsed(parsed)
                if tool_validation.failed:
                    finish_reason = _strict_tool_failure_finish_reason(
                        detail,
                        finish_reason,
                        server_stop=server_stop,
                    )
                else:
                    finish_reason = _finish_reason_for_output(
                        detail,
                        parsed_finish_reason if parsed.tool_calls else finish_reason,
                        server_stop=server_stop,
                        tool_calls=bool(parsed.tool_calls),
                    )
                structured_failure = _structured_output_failure_reason(
                    request,
                    _chat_response_format_text(message),
                    finish_reason,
                )
                if structured_failure is not None:
                    finish_reason = "stop"
                    message = {"role": "assistant", "content": ""}
                structured_length_failure = _structured_length_failure_reason(
                    request,
                    _chat_response_format_text(message),
                    finish_reason,
                )
                if tool_validation.failed:
                    finish_reason_override = tool_validation.failure_reason
                elif structured_failure is not None:
                    finish_reason_override = structured_failure
                elif structured_length_failure is not None:
                    finish_reason_override = structured_length_failure
                elif parsed.tool_calls:
                    finish_reason_override = "tool_calls"
                elif server_stop:
                    finish_reason_override = "stop"
                else:
                    finish_reason_override = None
                choice = {
                    "index": index,
                    "message": message,
                    "finish_reason": finish_reason,
                    "finish_details": _chat_finish_details_payload(
                        detail,
                        finish_reason,
                        text,
                        reason_override=finish_reason_override,
                        cache_action=requested_cache_action,
                        parsed=parsed,
                        token_counter=getattr(getattr(app.state, "hipengine_llm", None), "count_tokens", None),
                    ),
                }
                _mark_structured_length_failure(request, structured_length_failure, choice["finish_details"])
                _mark_structured_length_phase(request, choice["finish_details"])
                _raise_invalid_tool_call_hard_error_if_requested(
                    request,
                    tool_validation,
                    finish_details=choice["finish_details"],
                )
                effective_cache_action = _effective_session_cache_action(
                    requested_cache_action,
                    choice["finish_details"],
                )
                if effective_cache_action != requested_cache_action:
                    choice["finish_details"]["cache_action"] = effective_cache_action
                if request.logprobs:
                    choice["logprobs"] = _chat_visible_content_logprobs(
                        detail,
                        text,
                        initially_open=reasoning_initially_open,
                    )
                if _continuation_can_create(
                    request,
                    finish_reason=finish_reason,
                    finish_details=choice["finish_details"],
                    backend_continuation_eligible=_backend_continuation_eligible(detail),
                ):
                    base_prompt = prompt if continuation is None else continuation.prompts[index]
                    record = await store_continuation(
                        endpoint="chat",
                        prompts=(base_prompt,),
                        generated_texts=(text,),
                        response_format=request.response_format,
                        guided_json=request.guided_json,
                        guided_regex=request.guided_regex,
                        guided_choice=request.guided_choice,
                        guided_patch=request.guided_patch,
                        guided_diff=request.guided_diff,
                        auth_principal=auth_principal,
                        session_id=_session_id(request),
                    )
                    _attach_continuation_metadata(choice, continuation_id=record.id)
                else:
                    _mark_continuation_unavailable(choice["finish_details"])
                _attach_choice_telemetry(choice, detail)
                if n > 1:
                    choice["request_id"] = _choice_request_id(response_id, 0, index)
                choices.append(choice)
                visible_generated_texts.append(_chat_response_format_text(message))
                await commit_chat_session(
                    request,
                    request_messages=request.messages,
                    raw_output=output,
                    visible_message=message,
                    cache_action=effective_cache_action,
                    reset_session=prepared_prompt.reset_session_on_commit,
                    commit_base_messages=prepared_prompt.commit_base_messages,
                )
            response = {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": config.model_id,
                "choices": choices,
                "usage": batch.usage,
                "hipengine": {
                    "routing": _routing_response_metadata(
                        config,
                        requested_model=request.model,
                        engine=getattr(app.state, "hipengine_llm", None),
                    ),
                },
            }
            token_accounting = _exact_generated_token_accounting(
                getattr(app.state, "hipengine_llm", None),
                batch.details,
                visible_generated_texts,
            )
            if token_accounting is not None:
                response["hipengine"]["token_accounting"] = token_accounting
            if batch.generation_shape is not None:
                response["hipengine"]["generation_shape"] = deepcopy(batch.generation_shape)
            response["hipengine"]["speculative_mtp"] = _mtp_response_summary(
                None if batch.generation_shape is None else batch.generation_shape.get("route"),
                batch.details,
                route_decision=(
                    None
                    if batch.generation_shape is None
                    else batch.generation_shape.get("route_decision")
                ),
                thinking_policy=(
                    _request_speculative_mtp_thinking(config, request)
                    if batch.generation_shape is not None
                    and batch.generation_shape.get("route")
                    == _SPECULATIVE_MTP_BATCH_ROUTE
                    else None
                ),
            )
            await _maybe_write_agentic_result_replay_artifact(
                config,
                raw_request,
                response,
                engine=getattr(app.state, "hipengine_llm", None),
            )
            return response
        finally:
            await release_chat_session_reservation(admitted_session_id)

    async def stream_chat_completion_many(
        prompt: str,
        generation_prompt: PromptInput,
        prepared_thinking: _ThinkingControl | None,
        resident_session_key: str | None,
        request: ChatCompletionRequest,
        control: _RequestControl,
        raw_request: Request,
    ) -> AsyncIterator[str]:
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        include_hipengine = _stream_include_hipengine(request)
        stream_started_at = _StreamTimingTracker.start() if include_hipengine else time.perf_counter()
        routing_metadata = (
            _routing_response_metadata(
                config,
                requested_model=request.model,
                engine=getattr(app.state, "hipengine_llm", None),
            )
            if include_hipengine
            else None
        )
        try:
            n = _request_n(request)
            prompts = tuple(generation_prompt for _ in range(n))
            if _chat_live_many_streaming_allowed(request):

                async def prepare_many_stream() -> tuple[Any, SamplingParams] | None:
                    async with session_lock:
                        engine = get_llm()
                        await ensure_resident_context(engine, preparation_sampling(request), phase="preparation")
                        if not _engine_supports_stream_many(engine):
                            return None
                        sampling = sampling_params(
                            request,
                            prompts,
                            engine,
                            deadline_at=control.deadline_at,
                            cancellation_token=control.cancellation_token,
                            prepared_thinking=prepared_thinking,
                            resident_session_key=resident_session_key,
                        )
                        sampling = replace(
                            sampling,
                            row_seeds=_row_seeds_for_request(request.seed, len(prompts)),
                        )
                        _validate_context_budget(
                            effective_max_context_tokens(engine),
                            engine,
                            prompts,
                            sampling,
                            error_extra={
                                "hipengine": {
                                    "routing": _routing_rejection_metadata(
                                        config,
                                        requested_model=request.model,
                                        reason="context_overflow",
                                        engine=engine,
                                    )
                                }
                            },
                        )
                        return engine, sampling

                prepared_stream = await _await_with_request_control(prepare_many_stream(), control)
                if prepared_stream is not None:
                    engine, sampling = prepared_stream
                    if include_hipengine:
                        routing_metadata = _routing_response_metadata(
                            config,
                            requested_model=request.model,
                            engine=engine,
                        )
                    token_accounting_by_index = [
                        _StreamTokenAccounting.for_engine(engine) if include_hipengine else None
                        for _index in range(n)
                    ]
                    splitters = [
                        _ReasoningSplitter(
                            initially_open=_reasoning_initially_open_for_prompt(
                                engine,
                                row_prompt,
                            )
                        )
                        for row_prompt in prompts
                    ]
                    full_text = ["" for _index in range(n)]
                    last_stream_chunks: list[GenerationStreamChunk | None] = [None for _index in range(n)]
                    for index in range(n):
                        yield _chat_stream_role(
                            response_id,
                            created,
                            config.model_id,
                            index=index,
                            include_hipengine=include_hipengine,
                            stream_started_at=stream_started_at,
                            routing=routing_metadata,
                        )
                    async for token in _iterate_with_request_control(
                        generation_batcher.stream(
                            prompts,
                            sampling,
                            error_extra=route_rejection_extra(
                                requested_model=request.model,
                                reason="engine_busy",
                                engine=engine,
                                details={
                                    "overload_source": "generation_queue_cap",
                                    "max_queued_requests": config.max_queued_requests,
                                },
                            ),
                        ),
                        control,
                    ):
                        stream_chunk = _coerce_generation_stream_chunk(token)
                        row_index = _stream_chunk_row_index(stream_chunk, row_count=n)
                        if row_index is None:
                            raise ValueError(
                                "multi-row stream chunks must carry telemetry.decode_state.row_index"
                            )
                        last_stream_chunks[row_index] = stream_chunk
                        text = stream_chunk.text
                        if not text:
                            continue
                        full_text[row_index] += text
                        splitter = splitters[row_index]
                        token_accounting = token_accounting_by_index[row_index]
                        for part in splitter.feed_parts(text):
                            phase = "think" if part.field == "reasoning_content" else "answer"
                            token_payload = (
                                token_accounting.observe(phase, part.text)
                                if token_accounting is not None
                                else None
                            )
                            yield _chat_stream_delta(
                                response_id,
                                created,
                                config.model_id,
                                part.field,
                                part.text,
                                index=row_index,
                                tokens=token_payload,
                                stream_chunk=_stream_chunk_with_phase(stream_chunk, phase),
                                include_hipengine=include_hipengine,
                                stream_started_at=stream_started_at,
                                routing=routing_metadata,
                                phase=phase,
                            )
                    for index, splitter in enumerate(splitters):
                        token_accounting = token_accounting_by_index[index]
                        for part in splitter.finish_parts():
                            phase = "think" if part.field == "reasoning_content" else "answer"
                            token_payload = (
                                token_accounting.observe(phase, part.text)
                                if token_accounting is not None
                                else None
                            )
                            final_stream_chunk = last_stream_chunks[index]
                            yield _chat_stream_delta(
                                response_id,
                                created,
                                config.model_id,
                                part.field,
                                part.text,
                                index=index,
                                tokens=token_payload,
                                stream_chunk=(
                                    None
                                    if final_stream_chunk is None
                                    else _stream_chunk_with_phase(final_stream_chunk, phase)
                                ),
                                include_hipengine=include_hipengine,
                                stream_started_at=stream_started_at,
                                routing=routing_metadata,
                                phase=phase,
                            )
                    usage = _usage(engine, prompts, full_text)
                    app.state.hipengine_server_metrics.record_success(usage)
                    final_kv_pool = _kv_pool_stream_payload(engine) if include_hipengine else None
                    for index, text in enumerate(full_text):
                        backend_detail = _output_from_stream_chunk(last_stream_chunks[index], text)
                        finish_reason = _finish_reason_for_output(backend_detail, "stop")
                        token_accounting = token_accounting_by_index[index]
                        row_usage = _usage(engine, (prompts[index],), (text,))
                        final_tokens = (
                            _stream_usage_token_payload(row_usage, token_accounting)
                            if include_hipengine and token_accounting is not None
                            else None
                        )
                        yield _chat_stream_done(
                            response_id,
                            created,
                            config.model_id,
                            finish_reason,
                            index=index,
                            finish_details=_chat_finish_details_payload(
                                backend_detail,
                                finish_reason,
                                text,
                                cache_action=_session_cache_action(request),
                            ),
                            tokens=final_tokens,
                            stream_chunk=last_stream_chunks[index],
                            include_hipengine=include_hipengine,
                            stream_started_at=stream_started_at,
                            routing=routing_metadata,
                            kv_pool=final_kv_pool,
                        )
                    if _stream_include_usage(request):
                        yield _chat_stream_usage(
                            response_id,
                            created,
                            config.model_id,
                            usage,
                            include_hipengine=include_hipengine,
                            stream_started_at=stream_started_at,
                            routing=routing_metadata,
                            kv_pool=final_kv_pool,
                        )
                    yield "data: [DONE]\n\n"
                    return

            batch = await generate_with_request_control(
                prompts,
                request,
                control,
                prepared_thinking=prepared_thinking,
                resident_session_key=resident_session_key,
            )
            scheduler_chunks_by_index = _scheduler_token_chunks_by_request(batch.scheduler_token_chunks)
            if include_hipengine:
                routing_metadata = _routing_response_metadata(
                    config,
                    requested_model=request.model,
                    engine=getattr(app.state, "hipengine_llm", None),
                )
            for index, output in enumerate(batch.outputs):
                text, finish_reason = _apply_stop(output, request.stop)
                server_stop = text != output
                raw_parsed = _parse_chat_tool_calls_for_engine(
                    getattr(app.state, "hipengine_llm", None),
                    text,
                    tools=request.tools,
                )
                tool_validation = _validate_chat_tool_result(request, raw_parsed, text)
                reasoning_initially_open = _reasoning_initially_open_for_prompt(
                    getattr(app.state, "hipengine_llm", None),
                    prompts[index],
                )
                parsed = replace(
                    tool_validation.parsed,
                    reasoning_initially_open=reasoning_initially_open,
                )
                detail = batch.details[index]
                if tool_validation.failed:
                    finish_reason = _strict_tool_failure_finish_reason(
                        detail,
                        finish_reason,
                        server_stop=server_stop,
                    )
                else:
                    finish_reason = _finish_reason_for_output(
                        detail,
                        finish_reason,
                        server_stop=server_stop,
                        tool_calls=bool(parsed.tool_calls),
                    )
                structured_failure = _structured_output_failure_reason(
                    request,
                    _chat_response_format_text_from_parsed(parsed),
                    finish_reason,
                )
                if structured_failure is not None:
                    finish_reason = "stop"
                    parsed = _ParsedChatOutput(
                        text="",
                        tool_calls=(),
                        reasoning_initially_open=reasoning_initially_open,
                    )
                structured_length_failure = _structured_length_failure_reason(
                    request,
                    _chat_response_format_text_from_parsed(parsed),
                    finish_reason,
                )
                if tool_validation.failed:
                    finish_reason_override = tool_validation.failure_reason
                elif structured_failure is not None:
                    finish_reason_override = structured_failure
                elif structured_length_failure is not None:
                    finish_reason_override = structured_length_failure
                elif parsed.tool_calls:
                    finish_reason_override = "tool_calls"
                elif server_stop:
                    finish_reason_override = "stop"
                else:
                    finish_reason_override = None
                finish_details = _chat_finish_details_payload(
                    detail,
                    finish_reason,
                    text,
                    reason_override=finish_reason_override,
                    cache_action=_session_cache_action(request),
                    parsed=parsed,
                    token_counter=getattr(getattr(app.state, "hipengine_llm", None), "count_tokens", None),
                )
                _mark_structured_length_failure(request, structured_length_failure, finish_details)
                _mark_structured_length_phase(request, finish_details)
                _raise_invalid_tool_call_hard_error_if_requested(
                    request,
                    tool_validation,
                    finish_details=finish_details,
                )
                await _maybe_write_agentic_result_replay_artifact(
                    config,
                    raw_request,
                    {
                        "choices": [
                            {
                                "index": index,
                                "finish_reason": finish_reason,
                                "finish_details": finish_details,
                            }
                        ]
                    },
                    engine=getattr(app.state, "hipengine_llm", None),
                )
                logprobs = (
                    _chat_visible_content_logprobs(
                        batch.details[index],
                        text,
                        initially_open=reasoning_initially_open,
                    )
                    if request.logprobs
                    else None
                )
                token_accounting = (
                    _StreamTokenAccounting.for_engine(getattr(app.state, "hipengine_llm", None))
                    if include_hipengine
                    else None
                )
                yield _chat_stream_role(
                    response_id,
                    created,
                    config.model_id,
                    index=index,
                    include_hipengine=include_hipengine,
                    stream_started_at=stream_started_at,
                    routing=routing_metadata,
                )
                scheduler_chunks = scheduler_chunks_by_index.get(index, ())
                use_scheduler_logprobs = bool(request.logprobs)
                scheduler_logprob_safe_text = (
                    not use_scheduler_logprobs
                    or _scheduler_chunks_support_chat_logprob_stream(
                        text,
                        scheduler_chunks,
                        initially_open=reasoning_initially_open,
                    )
                )
                scheduler_done_phase = (
                    "structured"
                    if _structured_result_validation(request) and not parsed.tool_calls
                    else "done"
                )
                scheduler_content_phase = "structured" if scheduler_done_phase == "structured" else "answer"
                scheduler_tool_fragments = ()
                if parsed.tool_calls and not request.logprobs:
                    scheduler_tool_fragments = _scheduler_tool_call_argument_fragments(
                        parsed,
                        text,
                        scheduler_chunks,
                    )
                use_scheduler_tool_chunks = (
                    bool(scheduler_tool_fragments)
                    and not tool_validation.failed
                    and structured_failure is None
                    and structured_length_failure is None
                )
                withheld_scheduler_tool_chunks = None
                withheld_tool_chunk_reason = None
                if scheduler_chunks and (tool_validation.failed or raw_parsed.tool_calls):
                    if tool_validation.failed:
                        withheld_tool_chunk_reason = tool_validation.failure_reason
                    elif raw_parsed.tool_calls and not use_scheduler_tool_chunks:
                        withheld_tool_chunk_reason = "unmappable_tool_arguments"
                if withheld_tool_chunk_reason is not None:
                    withheld_scheduler_tool_chunks = _withheld_scheduler_tool_chunks_payload(
                        str(withheld_tool_chunk_reason),
                        raw_text=text,
                        scheduler_chunks=scheduler_chunks,
                    )
                withheld_scheduler_logprob_chunks = None
                if scheduler_chunks and use_scheduler_logprobs and not scheduler_logprob_safe_text:
                    withheld_scheduler_logprob_chunks = _withheld_scheduler_logprob_chunks_payload(
                        "unmappable_logprobs",
                        raw_text=text,
                        scheduler_chunks=scheduler_chunks,
                    )
                final_hipengine = {}
                for payload in (withheld_scheduler_tool_chunks, withheld_scheduler_logprob_chunks):
                    if payload is not None:
                        final_hipengine.update(payload)
                use_scheduler_chunks = (
                    not request.tools
                    and not parsed.tool_calls
                    and not tool_validation.failed
                    and structured_failure is None
                    and structured_length_failure is None
                    and scheduler_logprob_safe_text
                    and _scheduler_chunks_match_completion_text(
                        text,
                        scheduler_chunks,
                        require_logprobs=use_scheduler_logprobs,
                    )
                )
                if use_scheduler_tool_chunks:
                    for event in _chat_stream_scheduler_tool_call_chunks(
                        response_id,
                        created,
                        config.model_id,
                        scheduler_tool_fragments,
                        index=index,
                        finish_details=finish_details,
                        token_accounting=token_accounting,
                        final_stream_chunk=_stream_chunk_from_detail("", detail),
                        include_hipengine=include_hipengine,
                        stream_started_at=stream_started_at,
                        routing=routing_metadata,
                        kv_pool=(
                            _kv_pool_stream_payload(getattr(app.state, "hipengine_llm", None))
                            if include_hipengine
                            else None
                        ),
                    ):
                        yield event
                elif use_scheduler_chunks:
                    for event in _chat_stream_scheduler_text_chunks(
                        response_id,
                        created,
                        config.model_id,
                        scheduler_chunks,
                        finish_reason,
                        index=index,
                        include_logprobs=use_scheduler_logprobs,
                        content_phase=scheduler_content_phase,
                        reasoning_initially_open=reasoning_initially_open,
                        finish_details=finish_details,
                        token_accounting=token_accounting,
                        include_hipengine=include_hipengine,
                        stream_started_at=stream_started_at,
                        routing=routing_metadata,
                        kv_pool=(
                            _kv_pool_stream_payload(getattr(app.state, "hipengine_llm", None))
                            if include_hipengine
                            else None
                        ),
                        done_phase=scheduler_done_phase,
                    ):
                        yield event
                else:
                    for event in _chat_stream_parsed(
                        response_id,
                        created,
                        config.model_id,
                        parsed,
                        finish_reason,
                        index=index,
                        logprobs=logprobs,
                        finish_details=finish_details,
                        token_accounting=token_accounting,
                        stream_chunk=_stream_chunk_from_detail("", detail),
                        include_hipengine=include_hipengine,
                        final_hipengine=final_hipengine or None,
                        stream_started_at=stream_started_at,
                        routing=routing_metadata,
                        kv_pool=(
                            _kv_pool_stream_payload(getattr(app.state, "hipengine_llm", None))
                            if include_hipengine
                            else None
                        ),
                        done_phase=scheduler_done_phase,
                    ):
                        yield event
            if _stream_include_usage(request):
                yield _chat_stream_usage(
                    response_id,
                    created,
                    config.model_id,
                    batch.usage,
                    include_hipengine=include_hipengine,
                    stream_started_at=stream_started_at,
                    routing=routing_metadata,
                    kv_pool=(
                        _kv_pool_stream_payload(getattr(app.state, "hipengine_llm", None))
                        if include_hipengine
                        else None
                    ),
                )
        except OpenAIHTTPError as exc:
            _log_stream_failure(
                "POST /v1/chat/completions stream",
                status_code=exc.status_code,
                code=exc.code,
                param=exc.param,
                message=exc.message,
            )
            yield _chat_stream_error(
                response_id,
                created,
                config.model_id,
                exc.message,
                status_code=exc.status_code,
                code=exc.code,
                param=exc.param,
                error_type=exc.error_type,
                finish_details=exc.finish_details,
                extra=exc.extra,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
            )
        except Exception as exc:  # pragma: no cover - real runtime failures
            app.state.hipengine_server_metrics.record_failure()
            message = f"generation failed: {exc}"
            _log_stream_failure(
                "POST /v1/chat/completions stream",
                status_code=500,
                code="generation_failed",
                param=None,
                message=message,
            )
            yield _chat_stream_error(
                response_id,
                created,
                config.model_id,
                message,
                status_code=500,
                code="generation_failed",
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
            )
        yield "data: [DONE]\n\n"

    async def stream_chat_completion(
        prompt: str,
        generation_prompt: PromptInput,
        prepared_thinking: _ThinkingControl | None,
        resident_session_key: str | None,
        request: ChatCompletionRequest,
        control: _RequestControl,
        raw_request: Request,
    ) -> AsyncIterator[str]:
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        include_hipengine = _stream_include_hipengine(request)
        stream_started_at = _StreamTimingTracker.start() if include_hipengine else time.perf_counter()
        routing_metadata = (
            _routing_response_metadata(
                config,
                requested_model=request.model,
                engine=getattr(app.state, "hipengine_llm", None),
            )
            if include_hipengine
            else None
        )
        async def write_error_artifact(
            message: str,
            *,
            status_code: int,
            code: str | None,
            param: str | None = None,
            error_type: str = "server_error",
            finish_details: Mapping[str, Any] | None = None,
            extra: Mapping[str, Any] | None = None,
        ) -> None:
            await _maybe_write_stream_error_replay_artifact(
                config,
                raw_request,
                message=message,
                status_code=status_code,
                code=code,
                param=param,
                error_type=error_type,
                finish_details=finish_details,
                extra=extra,
                engine=getattr(app.state, "hipengine_llm", None),
            )

        try:
            _validate_generation_request(
                config,
                request,
                engine=getattr(app.state, "hipengine_llm", None),
                route_unsupported_grammar=True,
            )
        except OpenAIHTTPError as exc:
            _record_openai_error(app.state.hipengine_server_metrics, exc)
            _log_stream_failure(
                "POST /v1/chat/completions stream",
                status_code=exc.status_code,
                code=exc.code,
                param=exc.param,
                message=exc.message,
            )
            await write_error_artifact(
                exc.message,
                status_code=exc.status_code,
                code=exc.code,
                param=exc.param,
                error_type=exc.error_type,
                finish_details=exc.finish_details,
                extra=exc.extra,
            )
            yield _chat_stream_error(
                response_id,
                created,
                config.model_id,
                exc.message,
                status_code=exc.status_code,
                code=exc.code,
                param=exc.param,
                error_type=exc.error_type,
                finish_details=exc.finish_details,
                extra=exc.extra,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
            )
            yield "data: [DONE]\n\n"
            return
        full_text: list[str] = []
        last_stream_chunk: GenerationStreamChunk | None = None
        splitter_source_chunks: list[_LiveSourceChunk] = []
        splitter_source_offset = 0
        splitter = _ReasoningSplitter(
            initially_open=_reasoning_initially_open_for_prompt(
                getattr(app.state, "hipengine_llm", None),
                generation_prompt,
            )
        )
        buffer_tool_output = bool(request.tools)

        try:
            async def prepare_stream() -> tuple[
                Any,
                SamplingParams,
                str,
                dict[str, Any] | None,
                PromptInput,
            ]:
                async with session_lock:
                    engine = get_llm()
                    admission_started = time.perf_counter()
                    sampling = sampling_params(
                        request,
                        (generation_prompt,),
                        engine,
                        deadline_at=control.deadline_at,
                        cancellation_token=control.cancellation_token,
                        prepared_thinking=prepared_thinking,
                        resident_session_key=resident_session_key,
                    )
                    generation_route, generation_route_decision = (
                        _generation_route_for_request(
                            config,
                            request,
                            engine=engine,
                            sampling=sampling,
                            prompts=(generation_prompt,),
                        )
                    )
                    generation_route, generation_route_decision = (
                        _resolve_realized_generation_route(
                            generation_route,
                            group_rows=1,
                            sampling=sampling,
                            precomputed_decision=generation_route_decision,
                        )
                    )
                    await ensure_resident_context(
                        engine,
                        preparation_sampling(request),
                        phase="preparation",
                    )
                    _validate_context_budget(
                        effective_max_context_tokens(engine),
                        engine,
                        (generation_prompt,),
                        sampling,
                        error_extra={
                            "hipengine": {
                                "routing": _routing_rejection_metadata(
                                    config,
                                    requested_model=request.model,
                                    reason="context_overflow",
                                    engine=engine,
                                )
                            }
                        },
                    )
                    admitted_prompt = _with_admission_prepare_ms(
                        (generation_prompt,),
                        (time.perf_counter() - admission_started) * 1_000.0,
                    )[0]
                    return (
                        engine,
                        sampling,
                        generation_route,
                        generation_route_decision,
                        admitted_prompt,
                    )

            (
                engine,
                sampling,
                generation_route,
                generation_route_decision,
                generation_prompt,
            ) = await _await_with_request_control(
                prepare_stream(),
                control,
            )
            if include_hipengine:
                routing_metadata = _routing_response_metadata(
                    config,
                    requested_model=request.model,
                    engine=engine,
                )
            token_accounting = _StreamTokenAccounting.for_engine(engine) if include_hipengine else None
            yield _chat_stream_role(
                response_id,
                created,
                config.model_id,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
            )
            async for token in _iterate_with_request_control(
                generation_batcher.stream(
                    (generation_prompt,),
                    sampling,
                    route=generation_route,
                    route_decision=generation_route_decision,
                    error_extra=route_rejection_extra(
                        requested_model=request.model,
                        reason="engine_busy",
                        engine=engine,
                        details={
                            "overload_source": "generation_queue_cap",
                            "max_queued_requests": config.max_queued_requests,
                        },
                    ),
                ),
                control,
            ):
                stream_chunk = _coerce_generation_stream_chunk(token)
                last_stream_chunk = stream_chunk
                text = stream_chunk.text
                if not text:
                    continue
                if request.logprobs:
                    _validate_stream_logprob_chunk(stream_chunk)
                full_text.append(text)
                if buffer_tool_output:
                    continue
                source_start = splitter_source_offset
                source_end = source_start + len(text)
                splitter_source_offset = source_end
                splitter_source_chunks.append(
                    _LiveSourceChunk(
                        source_start=source_start,
                        source_end=source_end,
                        stream_chunk=stream_chunk,
                    )
                )
                for part in splitter.feed_parts(text):
                    phase = "think" if part.field == "reasoning_content" else "answer"
                    phase_stream_chunk = _live_splitter_stream_chunk_for_delta(
                        splitter_source_chunks,
                        stream_chunk,
                        part.text,
                        source_start=part.source_start,
                        source_end=part.source_end,
                        phase=phase,
                    )
                    logprobs = (
                        _chat_stream_logprobs(phase_stream_chunk, part.text)
                        if request.logprobs and part.field == "content"
                        else None
                    )
                    reasoning_logprobs = (
                        _chat_reasoning_stream_logprobs(phase_stream_chunk, part.text)
                        if request.logprobs
                        and include_hipengine
                        and part.field == "reasoning_content"
                        else None
                    )
                    token_payload = (
                        token_accounting.observe(phase, part.text) if token_accounting is not None else None
                    )
                    yield _chat_stream_delta(
                        response_id,
                        created,
                        config.model_id,
                        part.field,
                        part.text,
                        logprobs=logprobs,
                        reasoning_logprobs=reasoning_logprobs,
                        tokens=token_payload,
                        stream_chunk=phase_stream_chunk,
                        include_hipengine=include_hipengine,
                        stream_started_at=stream_started_at,
                        routing=routing_metadata,
                        phase=phase,
                    )
                _trim_live_splitter_source_chunks(
                    splitter_source_chunks,
                    min_source_start=splitter.pending_source_start,
                )
            if not buffer_tool_output:
                for part in splitter.finish_parts():
                    phase = "think" if part.field == "reasoning_content" else "answer"
                    finish_stream_chunk = _live_splitter_stream_chunk_for_delta(
                        splitter_source_chunks,
                        last_stream_chunk,
                        part.text,
                        source_start=part.source_start,
                        source_end=part.source_end,
                        phase=phase,
                    )
                    logprobs = (
                        _chat_stream_logprobs(finish_stream_chunk, part.text)
                        if request.logprobs and part.field == "content" and finish_stream_chunk is not None
                        else None
                    )
                    reasoning_logprobs = (
                        _chat_reasoning_stream_logprobs(finish_stream_chunk, part.text)
                        if request.logprobs
                        and include_hipengine
                        and part.field == "reasoning_content"
                        and finish_stream_chunk is not None
                        else None
                    )
                    token_payload = (
                        token_accounting.observe(phase, part.text) if token_accounting is not None else None
                    )
                    yield _chat_stream_delta(
                        response_id,
                        created,
                        config.model_id,
                        part.field,
                        part.text,
                        logprobs=logprobs,
                        reasoning_logprobs=reasoning_logprobs,
                        tokens=token_payload,
                        stream_chunk=finish_stream_chunk,
                        include_hipengine=include_hipengine,
                        stream_started_at=stream_started_at,
                        routing=routing_metadata,
                        phase=phase,
                    )
        except GenerationDeadlineExceeded as exc:
            openai_exc = _deadline_exceeded_error(exc.finish_details)
            _record_openai_error(app.state.hipengine_server_metrics, openai_exc)
            _log_stream_failure(
                "POST /v1/chat/completions stream",
                status_code=openai_exc.status_code,
                code=openai_exc.code,
                param=openai_exc.param,
                message=openai_exc.message,
            )
            await write_error_artifact(
                openai_exc.message,
                status_code=openai_exc.status_code,
                code=openai_exc.code,
                param=openai_exc.param,
                error_type=openai_exc.error_type,
                finish_details=openai_exc.finish_details,
            )
            yield _chat_stream_error(
                response_id,
                created,
                config.model_id,
                openai_exc.message,
                status_code=openai_exc.status_code,
                code=openai_exc.code,
                param=openai_exc.param,
                error_type=openai_exc.error_type,
                finish_details=openai_exc.finish_details,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
            )
            yield "data: [DONE]\n\n"
            return
        except GenerationCancelled as exc:
            openai_exc = _request_cancelled_error(exc.finish_details)
            _record_openai_error(app.state.hipengine_server_metrics, openai_exc)
            _log_stream_failure(
                "POST /v1/chat/completions stream",
                status_code=openai_exc.status_code,
                code=openai_exc.code,
                param=openai_exc.param,
                message=openai_exc.message,
            )
            await write_error_artifact(
                openai_exc.message,
                status_code=openai_exc.status_code,
                code=openai_exc.code,
                param=openai_exc.param,
                error_type=openai_exc.error_type,
                finish_details=openai_exc.finish_details,
            )
            yield _chat_stream_error(
                response_id,
                created,
                config.model_id,
                openai_exc.message,
                status_code=openai_exc.status_code,
                code=openai_exc.code,
                param=openai_exc.param,
                error_type=openai_exc.error_type,
                finish_details=openai_exc.finish_details,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
            )
            yield "data: [DONE]\n\n"
            return
        except OpenAIHTTPError as exc:
            _record_openai_error(app.state.hipengine_server_metrics, exc)
            _log_stream_failure(
                "POST /v1/chat/completions stream",
                status_code=exc.status_code,
                code=exc.code,
                param=exc.param,
                message=exc.message,
            )
            await write_error_artifact(
                exc.message,
                status_code=exc.status_code,
                code=exc.code,
                param=exc.param,
                error_type=exc.error_type,
                finish_details=exc.finish_details,
                extra=exc.extra,
            )
            yield _chat_stream_error(
                response_id,
                created,
                config.model_id,
                exc.message,
                status_code=exc.status_code,
                code=exc.code,
                param=exc.param,
                error_type=exc.error_type,
                finish_details=exc.finish_details,
                extra=exc.extra,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
            )
            yield "data: [DONE]\n\n"
            return
        except NotImplementedError as exc:
            app.state.hipengine_server_metrics.record_failure()
            message = str(exc)
            _log_stream_failure(
                "POST /v1/chat/completions stream",
                status_code=400,
                code="unsupported_parameter",
                param=None,
                message=message,
            )
            await write_error_artifact(
                message,
                status_code=400,
                code="unsupported_parameter",
                error_type="invalid_request_error",
            )
            yield _chat_stream_error(
                response_id,
                created,
                config.model_id,
                message,
                status_code=400,
                code="unsupported_parameter",
                error_type="invalid_request_error",
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
            )
            yield "data: [DONE]\n\n"
            return
        except ValueError as exc:
            app.state.hipengine_server_metrics.record_failure()
            message = str(exc)
            _log_stream_failure(
                "POST /v1/chat/completions stream",
                status_code=400,
                code="invalid_request",
                param=None,
                message=message,
            )
            await write_error_artifact(
                message,
                status_code=400,
                code="invalid_request",
                error_type="invalid_request_error",
            )
            yield _chat_stream_error(
                response_id,
                created,
                config.model_id,
                message,
                status_code=400,
                code="invalid_request",
                error_type="invalid_request_error",
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
            )
            yield "data: [DONE]\n\n"
            return
        except Exception as exc:  # pragma: no cover - real runtime failures
            app.state.hipengine_server_metrics.record_failure()
            message = f"generation failed: {exc}"
            _log_stream_failure(
                "POST /v1/chat/completions stream",
                status_code=500,
                code="generation_failed",
                param=None,
                message=message,
            )
            await write_error_artifact(
                message,
                status_code=500,
                code="generation_failed",
            )
            yield _chat_stream_error(
                response_id,
                created,
                config.model_id,
                message,
                status_code=500,
                code="generation_failed",
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
            )
            yield "data: [DONE]\n\n"
            return

        raw_text = "".join(full_text)
        text, finish_reason = _apply_stop(raw_text, request.stop)
        server_stop = text != raw_text
        backend_detail = None if server_stop else _output_from_stream_chunk(last_stream_chunk, raw_text)
        done_stream_chunk = None if server_stop else last_stream_chunk
        if server_stop:
            # Stop strings can split across yielded chunks; current streaming keeps
            # transport simple and reports the stop after generation completes.
            finish_reason = "stop"
        else:
            finish_reason = _finish_reason_for_output(backend_detail, finish_reason)
        usage = _usage(
            engine,
            (generation_prompt,),
            [text],
            details=None if backend_detail is None else (backend_detail,),
            prefer_telemetry_counts=True,
        )
        app.state.hipengine_server_metrics.record_success(usage)
        cache_action = _session_cache_action(request)
        final_tokens = (
            _stream_usage_token_payload(usage, token_accounting)
            if include_hipengine and token_accounting is not None
            else None
        )
        final_kv_pool = _kv_pool_stream_payload(engine) if include_hipengine else None
        if buffer_tool_output:
            parsed = _parse_chat_tool_calls_for_engine(
                engine,
                text,
                tools=request.tools,
            )
            tool_validation = _validate_chat_tool_result(request, parsed, text)
            parsed = replace(
                tool_validation.parsed,
                reasoning_initially_open=_reasoning_initially_open_for_prompt(
                    engine,
                    generation_prompt,
                ),
            )
            stream_finish_reason = "stop" if tool_validation.failed else "tool_calls" if parsed.tool_calls else finish_reason
            finish_details = _chat_finish_details_payload(
                None,
                stream_finish_reason,
                text,
                reason_override=tool_validation.failure_reason if tool_validation.failed else None,
                cache_action=cache_action,
                parsed=parsed,
                token_counter=getattr(engine, "count_tokens", None),
            )
            _mark_structured_length_phase(request, finish_details)
            hard_error = _invalid_tool_call_hard_error(
                request,
                tool_validation,
                finish_details=finish_details,
            )
            if hard_error is not None:
                _record_openai_error(app.state.hipengine_server_metrics, hard_error)
                _log_stream_failure(
                    "POST /v1/chat/completions stream",
                    status_code=hard_error.status_code,
                    code=hard_error.code,
                    param=hard_error.param,
                    message=hard_error.message,
                )
                await write_error_artifact(
                    hard_error.message,
                    status_code=hard_error.status_code,
                    code=hard_error.code,
                    param=hard_error.param,
                    error_type=hard_error.error_type,
                    finish_details=hard_error.finish_details,
                    extra=hard_error.extra,
                )
                yield _chat_stream_error(
                    response_id,
                    created,
                    config.model_id,
                    hard_error.message,
                    status_code=hard_error.status_code,
                    code=hard_error.code,
                    param=hard_error.param,
                    error_type=hard_error.error_type,
                    finish_details=hard_error.finish_details,
                    extra=hard_error.extra,
                    include_hipengine=include_hipengine,
                    stream_started_at=stream_started_at,
                    routing=routing_metadata,
                )
                yield "data: [DONE]\n\n"
                return
            await _maybe_write_agentic_result_replay_artifact(
                config,
                raw_request,
                {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": stream_finish_reason,
                            "finish_details": finish_details,
                        }
                    ]
                },
                engine=engine,
            )
            for event in _chat_stream_parsed(
                response_id,
                created,
                config.model_id,
                parsed,
                stream_finish_reason,
                finish_details=finish_details,
                done_tokens=final_tokens,
                token_accounting=token_accounting,
                stream_chunk=done_stream_chunk,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
                kv_pool=final_kv_pool,
            ):
                yield event
        else:
            yield _chat_stream_done(
                response_id,
                created,
                config.model_id,
                finish_reason,
                finish_details=_chat_finish_details_payload(
                    backend_detail,
                    finish_reason,
                    text,
                    cache_action=cache_action,
                ),
                tokens=final_tokens,
                stream_chunk=done_stream_chunk,
                include_hipengine=include_hipengine,
                extra_hipengine=_stream_speculative_mtp_metadata(
                    generation_route,
                    backend_detail,
                    route_decision=generation_route_decision,
                    thinking_policy=_request_speculative_mtp_thinking(
                        config,
                        request,
                    ),
                ),
                stream_started_at=stream_started_at,
                routing=routing_metadata,
                kv_pool=final_kv_pool,
            )
        if _stream_include_usage(request):
            yield _chat_stream_usage(
                response_id,
                created,
                config.model_id,
                usage,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing_metadata,
                kv_pool=final_kv_pool,
            )
        yield "data: [DONE]\n\n"

    return app


def _metrics_mode(raw: str | None) -> str:
    value = "off" if raw is None or raw == "" else str(raw).strip().lower()
    if value not in {"off", "prometheus"}:
        raise ValueError("metrics must be one of: off, prometheus")
    return value


def _live_loop_snapshot(engine: Any | None) -> Mapping[str, Any] | None:
    if engine is None:
        return None
    getter = getattr(engine, "live_loop_snapshot", None)
    if not callable(getter):
        return None
    try:
        payload = getter()
    except Exception:
        return None
    return payload if isinstance(payload, Mapping) else None


def _nested_mapping(owner: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    if owner is None:
        return {}
    value = owner.get(key)
    return value if isinstance(value, Mapping) else {}


def _prefix_cache_metric_values(
    snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    runner = _nested_mapping(snapshot, "runner")
    prefix = _nested_mapping(runner, "prefix_cache")
    if not prefix:
        return {
            "mode": "off",
            "block_size_tokens": 256,
            "snapshot_entries": 0,
            "snapshot_limit": 0,
            "retained_snapshot_entries": 0,
            "snapshot_bytes": 0,
            "retained_kv_pages": 0,
            "retained_kv_bytes": 0,
            "resident_bytes": 0,
            "resident_limit_bytes": 0,
        }
    return deepcopy(dict(prefix))


def _resident_loop_metric_values(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    loop = _nested_mapping(snapshot, "loop")
    requests = _nested_mapping(loop, "requests")
    bucket = _nested_mapping(loop, "physical_bucket")
    work = _nested_mapping(loop, "work_counts")
    policy = _nested_mapping(loop, "scheduler_policy")
    latency = _nested_mapping(loop, "latency_seconds")
    runner = _nested_mapping(snapshot, "runner")
    model_runner = _nested_mapping(runner, "model_runner")
    routes = _nested_mapping(runner, "routes")

    active_mask_value = bucket.get("active_mask")
    if isinstance(active_mask_value, Sequence) and not isinstance(active_mask_value, (str, bytes)):
        active_mask = "".join("1" if value is True else "0" for value in active_mask_value)
    else:
        active_mask = ""
    latency_rows: dict[str, dict[str, float]] = {}
    for kind in ("queue", "time_to_first_token", "inter_token", "service", "completion"):
        row = _nested_mapping(latency, kind)
        latency_rows[kind] = {
            "count": _non_negative_metric_value(row.get("count")),
            "sum": _non_negative_metric_value(row.get("sum")),
            "max": _non_negative_metric_value(row.get("max")),
            "p50": _non_negative_metric_value(row.get("p50")),
            "p95": _non_negative_metric_value(row.get("p95")),
        }
    return {
        "requests": {
            "pending": _non_negative_metric_value(requests.get("pending")),
            "admitted": _non_negative_metric_value(requests.get("admitted_current")),
            "active": _non_negative_metric_value(requests.get("active")),
            "admitted_total": _non_negative_metric_value(requests.get("admitted_total")),
            "reclaimed_total": _non_negative_metric_value(requests.get("reclaimed_total")),
        },
        "bucket": {
            "capacity": _non_negative_metric_value(bucket.get("capacity")),
            "active_rows": _non_negative_metric_value(bucket.get("active_c")),
            "occupied_slots": _non_negative_metric_value(bucket.get("occupied_slots")),
            "free_slots": _non_negative_metric_value(bucket.get("free_slots")),
            "occupancy_ratio": _non_negative_metric_value(bucket.get("occupancy_ratio")),
            "active_mask": active_mask,
        },
        "work": {
            "prefill": _non_negative_metric_value(work.get("prefill")),
            "decode": _non_negative_metric_value(work.get("decode")),
            "reclaim": _non_negative_metric_value(work.get("reclaim")),
        },
        "packed_workspace": {
            "current_bytes": _non_negative_metric_value(
                model_runner.get("packed_workspace_current_bytes")
            ),
            "release_events": _non_negative_metric_value(
                model_runner.get("packed_workspace_release_events")
            ),
            "released_bytes": _non_negative_metric_value(
                model_runner.get("packed_workspace_released_bytes")
            ),
        },
        "persistent_kv": {
            "int8_payload_bytes": _non_negative_metric_value(
                model_runner.get("persistent_int8_payload_bytes")
            ),
            "bf16_payload_bytes": _non_negative_metric_value(
                model_runner.get("persistent_bf16_payload_bytes")
            ),
            "scale_bytes": _non_negative_metric_value(
                model_runner.get("persistent_scale_bytes")
            ),
            "bf16_mirror_bytes": _non_negative_metric_value(
                model_runner.get("persistent_bf16_mirror_bytes")
            ),
            "total_bytes": _non_negative_metric_value(
                model_runner.get("persistent_kv_total_bytes")
            ),
        },
        "policy": {
            "name": str(policy.get("prefill_decode_policy") or "unavailable"),
            "prefill_chunk_tokens": _non_negative_metric_value(
                policy.get("prefill_chunk_tokens")
            ),
            "fair_prefill_burst_chunks": max(
                1.0,
                _non_negative_metric_value(policy.get("fair_prefill_burst_chunks", 1)),
            ),
            "consecutive_prefill_chunks": _non_negative_metric_value(
                policy.get("consecutive_prefill_chunks")
            ),
            "last_work_kind": str(policy.get("last_work_kind") or "none"),
        },
        "latency": latency_rows,
        "route_counts": _non_negative_metric_mapping(routes.get("counts")),
        "fallback_reasons": _non_negative_metric_mapping(routes.get("fallback_reasons")),
        "last_execution_manifest": _nested_mapping(routes, "last_execution_manifest"),
    }


def _mtp_acceptance_rate_metric(metrics: _ServerMetrics) -> float:
    """Cumulative MTP draft acceptance rate across recorded requests."""

    total = metrics.mtp_draft_tokens_accepted_total + metrics.mtp_draft_tokens_rejected_total
    if total <= 0:
        return 0.0
    return metrics.mtp_draft_tokens_accepted_total / total


def _mtp_serving_metric(config: ServerConfig | None, engine: Any | None) -> float:
    """Effective MTP serving state for the prometheus gauge (1 enabled / 0 off)."""

    if config is None:
        return 0.0
    return 1.0 if _speculative_mtp_capability(config, engine=engine)["serving_route"] else 0.0


def _render_prometheus_metrics(
    metrics: _ServerMetrics,
    *,
    engine: Any | None,
    generation_batcher: Any | None = None,
    chat_sessions: Mapping[str, Any] | None = None,
    pending_chat_sessions: set[str] | None = None,
    max_chat_sessions: int | None = None,
    config: ServerConfig | None = None,
) -> str:
    live_snapshot = _live_loop_snapshot(engine)
    resident = _resident_loop_metric_values(live_snapshot)
    pool = _pool_metric_values(engine, live_snapshot=live_snapshot)
    graph = _graph_bucket_metric_values(engine, live_snapshot=live_snapshot)
    queue = _generation_queue_metric_values(generation_batcher)
    values = {
        "hipengine_requests_total": metrics.request_total,
        "hipengine_request_completed_total": metrics.request_completed_total,
        "hipengine_request_failed_total": metrics.request_failed_total,
        "hipengine_request_rejected_total": metrics.request_rejected_total,
        "hipengine_request_cancelled_total": metrics.request_cancelled_total,
        "hipengine_prompt_tokens_total": metrics.prompt_tokens_total,
        "hipengine_completion_tokens_total": metrics.completion_tokens_total,
        "hipengine_resident_requests_pending": resident["requests"]["pending"],
        "hipengine_resident_requests_admitted": resident["requests"]["admitted"],
        "hipengine_resident_requests_active": resident["requests"]["active"],
        "hipengine_resident_requests_admitted_total": resident["requests"]["admitted_total"],
        "hipengine_resident_requests_reclaimed_total": resident["requests"]["reclaimed_total"],
        "hipengine_resident_bucket_capacity": resident["bucket"]["capacity"],
        "hipengine_resident_bucket_active_rows": resident["bucket"]["active_rows"],
        "hipengine_resident_bucket_occupied_slots": resident["bucket"]["occupied_slots"],
        "hipengine_resident_bucket_free_slots": resident["bucket"]["free_slots"],
        "hipengine_resident_bucket_occupancy_ratio": resident["bucket"]["occupancy_ratio"],
        "hipengine_resident_work_prefill_total": resident["work"]["prefill"],
        "hipengine_resident_work_decode_total": resident["work"]["decode"],
        "hipengine_resident_work_reclaim_total": resident["work"]["reclaim"],
        "hipengine_resident_prefill_chunk_tokens": resident["policy"]["prefill_chunk_tokens"],
        "hipengine_resident_fair_prefill_burst_chunks": resident["policy"]["fair_prefill_burst_chunks"],
        "hipengine_resident_consecutive_prefill_chunks": resident["policy"]["consecutive_prefill_chunks"],
        "hipengine_resident_packed_workspace_current_bytes": resident["packed_workspace"]["current_bytes"],
        "hipengine_resident_packed_workspace_release_events_total": resident["packed_workspace"]["release_events"],
        "hipengine_resident_packed_workspace_released_bytes_total": resident["packed_workspace"]["released_bytes"],
        "hipengine_resident_kv_int8_payload_bytes": resident["persistent_kv"]["int8_payload_bytes"],
        "hipengine_resident_kv_bf16_payload_bytes": resident["persistent_kv"]["bf16_payload_bytes"],
        "hipengine_resident_kv_scale_bytes": resident["persistent_kv"]["scale_bytes"],
        "hipengine_resident_kv_bf16_mirror_bytes": resident["persistent_kv"]["bf16_mirror_bytes"],
        "hipengine_resident_kv_total_bytes": resident["persistent_kv"]["total_bytes"],
        "hipengine_kv_pool_current_bytes": pool["current_bytes"],
        "hipengine_kv_pool_high_water_observed_bytes": pool["high_water_observed_bytes"],
        "hipengine_kv_pool_grow_events_total": pool["grow_events"],
        "hipengine_kv_pool_grow_failures_total": pool["grow_failures"],
        "hipengine_kv_pool_shrink_events_total": pool["shrink_events"],
        "hipengine_kv_pool_current_pages": pool["current_pages"],
        "hipengine_kv_pool_high_water_observed_pages": pool["high_water_observed_pages"],
        "hipengine_kv_pool_free_pages": pool["free_pages"],
        "hipengine_kv_pool_refcounted_pages": pool["refcounted_pages"],
        "hipengine_kv_pool_pinned_pages": pool["pinned_pages"],
        "hipengine_graph_bucket_entries": graph["entries"],
        "hipengine_graph_bucket_hits_total": graph["hits"],
        "hipengine_graph_bucket_misses_total": graph["misses"],
        "hipengine_graph_bucket_replay_hit_rate": graph["replay_hit_rate"],
        "hipengine_graph_bucket_captures_total": graph["captures"],
        "hipengine_graph_bucket_replays_total": graph["replays"],
        "hipengine_graph_bucket_invalidations_total": graph["invalidations"],
        "hipengine_generation_queue_depth": queue["depth"],
        "hipengine_generation_queue_max_depth": queue["max_depth"],
        "hipengine_generation_worker_active": queue["worker_active"],
        "hipengine_generation_requests_active": queue["active_requests"],
        "hipengine_generation_requests_max_active": queue["max_active_requests"],
        "hipengine_chat_sessions_active": 0.0 if chat_sessions is None else float(len(chat_sessions)),
        "hipengine_chat_sessions_pending": (
            0.0 if pending_chat_sessions is None else float(len(pending_chat_sessions))
        ),
        "hipengine_chat_sessions_max_active": (
            0.0 if max_chat_sessions is None else float(max_chat_sessions)
        ),
        "hipengine_mtp_requests_total": metrics.mtp_requests_total,
        "hipengine_mtp_draft_tokens_accepted_total": metrics.mtp_draft_tokens_accepted_total,
        "hipengine_mtp_draft_tokens_rejected_total": metrics.mtp_draft_tokens_rejected_total,
        "hipengine_mtp_draft_acceptance_rate": _mtp_acceptance_rate_metric(metrics),
        "hipengine_mtp_serving": _mtp_serving_metric(config, engine),
    }
    help_text = {
        "hipengine_requests_total": "Total generation requests observed by the server.",
        "hipengine_request_completed_total": "Generation requests that completed successfully.",
        "hipengine_request_failed_total": "Generation requests that failed after reaching generation validation.",
        "hipengine_request_rejected_total": "Generation requests rejected by admission/backpressure.",
        "hipengine_request_cancelled_total": "Generation requests cancelled after client disconnect or backend cancellation.",
        "hipengine_prompt_tokens_total": "Prompt tokens counted for successful requests.",
        "hipengine_completion_tokens_total": "Completion tokens counted for successful requests.",
        "hipengine_resident_requests_pending": "Current requests pending resident-loop admission.",
        "hipengine_resident_requests_admitted": "Current requests admitted to physical resident slots.",
        "hipengine_resident_requests_active": "Current active resident-loop requests.",
        "hipengine_resident_requests_admitted_total": "Cumulative resident-loop admissions.",
        "hipengine_resident_requests_reclaimed_total": "Cumulative resident-loop reclaims.",
        "hipengine_resident_bucket_capacity": "Physical resident bucket capacity.",
        "hipengine_resident_bucket_active_rows": "Current active rows in the physical resident bucket.",
        "hipengine_resident_bucket_occupied_slots": "Current occupied physical resident slots.",
        "hipengine_resident_bucket_free_slots": "Current free physical resident slots.",
        "hipengine_resident_bucket_occupancy_ratio": "Current occupied/capacity ratio for the resident bucket.",
        "hipengine_resident_work_prefill_total": "Executed resident prefill work items.",
        "hipengine_resident_work_decode_total": "Executed resident decode work items.",
        "hipengine_resident_work_reclaim_total": "Executed resident reclaim transitions.",
        "hipengine_resident_prefill_chunk_tokens": "Configured maximum tokens in one resident prefill work item.",
        "hipengine_resident_fair_prefill_burst_chunks": "Configured maximum consecutive prefill chunks while fair scheduling also has decode work.",
        "hipengine_resident_consecutive_prefill_chunks": "Current consecutive resident prefill chunks since the last decode work item.",
        "hipengine_resident_packed_workspace_current_bytes": "Current owner-only packed GGUF workspace bytes.",
        "hipengine_resident_packed_workspace_release_events_total": "Reclaimed packed GGUF workspace owners.",
        "hipengine_resident_packed_workspace_released_bytes_total": "Cumulative owner-only packed GGUF workspace bytes reclaimed.",
        "hipengine_resident_kv_int8_payload_bytes": "Current resident request-owned INT8 KV payload bytes.",
        "hipengine_resident_kv_bf16_payload_bytes": "Current resident request-owned BF16 KV payload bytes.",
        "hipengine_resident_kv_scale_bytes": "Current resident request-owned KV scale bytes.",
        "hipengine_resident_kv_bf16_mirror_bytes": "Current resident request-owned persistent BF16 mirror bytes.",
        "hipengine_resident_kv_total_bytes": "Current resident request-owned persistent KV bytes.",
        "hipengine_kv_pool_current_bytes": "Current dynamic KV pool bytes, or 0 when unavailable.",
        "hipengine_kv_pool_high_water_observed_bytes": "Peak observed dynamic KV pool bytes, or 0 when unavailable.",
        "hipengine_kv_pool_grow_events_total": "Dynamic KV pool grow events, or 0 when unavailable.",
        "hipengine_kv_pool_grow_failures_total": "Dynamic KV pool grow failures, or 0 when unavailable.",
        "hipengine_kv_pool_shrink_events_total": "Dynamic KV pool shrink events, or 0 when unavailable.",
        "hipengine_kv_pool_current_pages": "Current dynamic KV pool pages, or 0 when unavailable.",
        "hipengine_kv_pool_high_water_observed_pages": "Peak observed dynamic KV pool pages, or 0 when unavailable.",
        "hipengine_kv_pool_free_pages": "Current dynamic KV pool free pages, or 0 when unavailable.",
        "hipengine_kv_pool_refcounted_pages": "Current dynamic KV pool refcounted pages, or 0 when unavailable.",
        "hipengine_kv_pool_pinned_pages": "Current graph-pinned dynamic KV pages, or 0 when unavailable.",
        "hipengine_graph_bucket_entries": "Current graph bucket cache entries, or 0 when unavailable.",
        "hipengine_graph_bucket_hits_total": "Graph bucket cache hits, or 0 when unavailable.",
        "hipengine_graph_bucket_misses_total": "Graph bucket cache misses, or 0 when unavailable.",
        "hipengine_graph_bucket_replay_hit_rate": "Graph bucket replay hit rate, or 0 when unavailable.",
        "hipengine_graph_bucket_captures_total": "Resident graph captures across all physical buckets.",
        "hipengine_graph_bucket_replays_total": "Resident graph replay calls across all physical buckets.",
        "hipengine_graph_bucket_invalidations_total": "Resident graph invalidations across all physical buckets.",
        "hipengine_generation_queue_depth": "Current generation-batcher queue depth.",
        "hipengine_generation_queue_max_depth": "Configured generation-batcher queue cap, or 0 when unset.",
        "hipengine_generation_worker_active": "Whether the generation-batcher worker is active, as 0 or 1.",
        "hipengine_generation_requests_active": "Current number of HTTP requests in the active backend batch.",
        "hipengine_generation_requests_max_active": "Configured active backend request cap, or 0 when unset.",
        "hipengine_chat_sessions_active": "Current app-local chat transcript session count.",
        "hipengine_chat_sessions_pending": "Current app-local chat session creations in flight.",
        "hipengine_chat_sessions_max_active": "Configured app-local chat session cap, or 0 when unset.",
        "hipengine_mtp_requests_total": "Completed requests that served through the speculative MTP route.",
        "hipengine_mtp_draft_tokens_accepted_total": "Draft tokens proposed by MTP that the target accepted.",
        "hipengine_mtp_draft_tokens_rejected_total": "Draft tokens proposed by MTP that the target rejected.",
        "hipengine_mtp_draft_acceptance_rate": "Cumulative MTP draft acceptance rate (accepted/(accepted+rejected)).",
        "hipengine_mtp_serving": "Whether the server's effective MTP serving route is enabled (1) or off (0).",
    }
    counter_names = {
        "hipengine_requests_total",
        "hipengine_request_completed_total",
        "hipengine_request_failed_total",
        "hipengine_request_rejected_total",
        "hipengine_request_cancelled_total",
        "hipengine_prompt_tokens_total",
        "hipengine_completion_tokens_total",
        "hipengine_resident_requests_admitted_total",
        "hipengine_resident_requests_reclaimed_total",
        "hipengine_resident_work_prefill_total",
        "hipengine_resident_work_decode_total",
        "hipengine_resident_work_reclaim_total",
        "hipengine_resident_packed_workspace_release_events_total",
        "hipengine_resident_packed_workspace_released_bytes_total",
        "hipengine_kv_pool_grow_events_total",
        "hipengine_kv_pool_grow_failures_total",
        "hipengine_kv_pool_shrink_events_total",
        "hipengine_graph_bucket_hits_total",
        "hipengine_graph_bucket_misses_total",
        "hipengine_graph_bucket_captures_total",
        "hipengine_graph_bucket_replays_total",
        "hipengine_graph_bucket_invalidations_total",
    }
    lines: list[str] = []
    for name, value in values.items():
        lines.append(f"# HELP {name} {help_text[name]}")
        lines.append(f"# TYPE {name} {'counter' if name in counter_names else 'gauge'}")
        lines.append(f"{name} {_format_metric_value(value)}")
    scheduler = _scheduler_fairness_capability(
        continuous_decode=_engine_supports_controlled_streaming(engine)
    )
    lines.append("# HELP hipengine_generation_scheduler_fairness_policy_info Generation scheduler fairness policy.")
    lines.append("# TYPE hipengine_generation_scheduler_fairness_policy_info gauge")
    lines.append(
        "hipengine_generation_scheduler_fairness_policy_info{"
        f'policy="{_escape_prometheus_label_value(str(scheduler["policy"]))}",'
        f'compatible_sampling_coalescing="{str(bool(scheduler["compatible_sampling_coalescing"])).lower()}",'
        f'continuous_decode="{str(bool(scheduler["continuous_decode"])).lower()}",'
        f'preemptive_fairness="{str(bool(scheduler["preemptive_fairness"])).lower()}"'
        "} 1"
    )
    lines.append("# HELP hipengine_resident_bucket_info Resident physical bucket and scheduler policy.")
    lines.append("# TYPE hipengine_resident_bucket_info gauge")
    lines.append(
        "hipengine_resident_bucket_info{"
        f'active_mask="{_escape_prometheus_label_value(str(resident["bucket"]["active_mask"]))}",'
        f'fair_prefill_burst_chunks="{int(resident["policy"]["fair_prefill_burst_chunks"])}",'
        f'last_work_kind="{_escape_prometheus_label_value(str(resident["policy"]["last_work_kind"]))}",'
        f'policy="{_escape_prometheus_label_value(str(resident["policy"]["name"]))}"'
        "} 1"
    )
    _append_resident_latency_metrics(lines, resident["latency"])
    graph_buckets = graph.get("buckets", {})
    for graph_stat, metric_name in (
        ("entries", "hipengine_graph_bucket_entries_by_bucket"),
        ("captures", "hipengine_graph_bucket_captures_by_bucket_total"),
        ("hits", "hipengine_graph_bucket_hits_by_bucket_total"),
        ("replays", "hipengine_graph_bucket_replays_by_bucket_total"),
        ("invalidations", "hipengine_graph_bucket_invalidations_by_bucket_total"),
    ):
        bucket_values = {
            str(bucket): _non_negative_metric_value(row.get(graph_stat))
            for bucket, row in graph_buckets.items()
            if isinstance(row, Mapping)
        }
        if graph_stat == "entries":
            _append_labeled_gauge_metrics(
                lines,
                metric_name,
                "Current resident graph entries by stable physical bucket.",
                "bucket",
                bucket_values,
            )
        else:
            _append_labeled_counter_metrics(
                lines,
                metric_name,
                f"Resident graph {graph_stat} by stable physical bucket.",
                "bucket",
                bucket_values,
            )
    _append_labeled_counter_metrics(
        lines,
        "hipengine_resident_route_total",
        "Resident GGUF execution transitions by declared route.",
        "route",
        resident["route_counts"],
    )
    _append_labeled_counter_metrics(
        lines,
        "hipengine_resident_fallback_total",
        "Resident GGUF explicit fallbacks by reason.",
        "reason",
        resident["fallback_reasons"],
    )
    manifest = resident["last_execution_manifest"]
    if manifest:
        lines.append("# HELP hipengine_resident_route_manifest_info Last resident execution manifest identity.")
        lines.append("# TYPE hipengine_resident_route_manifest_info gauge")
        lines.append(
            "hipengine_resident_route_manifest_info{"
            f'claim_level="{_escape_prometheus_label_value(str(manifest.get("claim_level") or "unavailable"))}",'
            f'kind="{_escape_prometheus_label_value(str(manifest.get("kind") or "unavailable"))}",'
            f'kv_attention_source="{_escape_prometheus_label_value(str(manifest.get("kv_attention_source") or "unavailable"))}",'
            f'logical_c="{_escape_prometheus_label_value(str(manifest.get("logical_c") or manifest.get("rows") or 0))}",'
            f'mode="{_escape_prometheus_label_value(str(manifest.get("mode") or "unavailable"))}",'
            f'physical_execution_width="{_escape_prometheus_label_value(str(manifest.get("physical_execution_width") or manifest.get("physical_rows") or 0))}",'
            f'rows="{_escape_prometheus_label_value(str(manifest.get("rows") or 0))}",'
            f'serial_decode_fallback="{str(bool(manifest.get("serial_decode_fallback", False))).lower()}",'
            f'throughput_claim_eligible="{str(bool(manifest.get("throughput_claim_eligible", False))).lower()}"'
            "} 1"
        )
    _append_labeled_counter_metrics(
        lines,
        "hipengine_graph_bucket_miss_reason_total",
        "Graph bucket cache misses by reason, or empty when unavailable.",
        "reason",
        graph["miss_reasons"],
    )
    _append_labeled_counter_metrics(
        lines,
        "hipengine_graph_bucket_kernel_time_bucket_total",
        "Graph bucket kernel-time observations by duration bucket, or empty when unavailable.",
        "bucket",
        graph["kernel_time_histogram_ns"],
    )
    return "\n".join(lines) + "\n"


_KV_POOL_STATS_ATTRS = ("kv_pool", "kv_cache_pool", "pool", "kv_pool_stats")
_KV_POOL_METRIC_DEFAULTS = {
    "current_bytes": 0.0,
    "high_water_observed_bytes": 0.0,
    "current_pages": 0.0,
    "high_water_observed_pages": 0.0,
    "grow_events": 0.0,
    "grow_failures": 0.0,
    "shrink_events": 0.0,
    "free_pages": 0.0,
    "refcounted_pages": 0.0,
    "pinned_pages": 0.0,
}


def _pool_metric_values(
    engine: Any | None,
    *,
    live_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    stats = _kv_pool_stats_object(engine, live_snapshot=live_snapshot)
    if stats is None:
        return dict(_KV_POOL_METRIC_DEFAULTS)
    return _kv_pool_metric_values_from_stats(stats)


def _kv_pool_stats_object(
    engine: Any | None,
    *,
    live_snapshot: Mapping[str, Any] | None = None,
) -> Any | None:
    snapshot = live_snapshot if live_snapshot is not None else _live_loop_snapshot(engine)
    runner = _nested_mapping(snapshot, "runner")
    live_pool = runner.get("kv_pool")
    if isinstance(live_pool, Mapping):
        return live_pool
    return _first_stats_object(engine, _KV_POOL_STATS_ATTRS)


def _kv_pool_metric_values_from_stats(stats: Any) -> dict[str, float]:
    values = dict(_KV_POOL_METRIC_DEFAULTS)
    data = _stats_to_mapping(stats)
    for key in values:
        values[key] = _non_negative_metric_value(data.get(key))
    return values


def _kv_pool_stream_payload(engine: Any | None) -> dict[str, float] | None:
    stats = _kv_pool_stats_object(engine)
    if stats is None:
        return None
    return _kv_pool_metric_values_from_stats(stats)


def _generation_queue_metric_values(generation_batcher: Any | None) -> dict[str, float]:
    if generation_batcher is None:
        return {
            "depth": 0.0,
            "max_depth": 0.0,
            "worker_active": 0.0,
            "active_requests": 0.0,
            "max_active_requests": 0.0,
        }
    depth = _non_negative_metric_value(_call_metric_getter(generation_batcher, "queue_depth"))
    max_depth_raw = _call_metric_getter(generation_batcher, "max_queue_size")
    max_depth = 0.0 if max_depth_raw is None else _non_negative_metric_value(max_depth_raw)
    active = _call_metric_getter(generation_batcher, "active")
    active_requests = _non_negative_metric_value(_call_metric_getter(generation_batcher, "active_requests"))
    max_active_raw = _call_metric_getter(generation_batcher, "max_active_requests")
    max_active = 0.0 if max_active_raw is None else _non_negative_metric_value(max_active_raw)
    return {
        "depth": depth,
        "max_depth": max_depth,
        "worker_active": 1.0 if bool(active) else 0.0,
        "active_requests": active_requests,
        "max_active_requests": max_active,
    }


def _call_metric_getter(owner: Any, name: str) -> Any:
    value = getattr(owner, name, None)
    return value() if callable(value) else value


def _graph_bucket_metric_values(
    engine: Any | None,
    *,
    live_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "entries": 0.0,
        "hits": 0.0,
        "misses": 0.0,
        "replay_hit_rate": 0.0,
        "captures": 0.0,
        "replays": 0.0,
        "invalidations": 0.0,
        "buckets": {},
        "miss_reasons": {},
        "kernel_time_histogram_ns": {},
    }
    snapshot = live_snapshot if live_snapshot is not None else _live_loop_snapshot(engine)
    runner = _nested_mapping(snapshot, "runner")
    live_graph = _nested_mapping(runner, "graph_buckets")
    if live_graph:
        values["entries"] = _non_negative_metric_value(live_graph.get("entries"))
        values["hits"] = _non_negative_metric_value(live_graph.get("hits_total"))
        values["captures"] = _non_negative_metric_value(live_graph.get("captures_total"))
        values["misses"] = values["captures"]
        values["replays"] = _non_negative_metric_value(live_graph.get("replays_total"))
        values["invalidations"] = _non_negative_metric_value(
            live_graph.get("invalidations_total")
        )
        lookups = values["hits"] + values["misses"]
        values["replay_hit_rate"] = values["hits"] / lookups if lookups > 0.0 else 0.0
        bucket_rows = _nested_mapping(live_graph, "buckets")
        values["buckets"] = {
            str(bucket): {
                field: _non_negative_metric_value(row.get(field))
                for field in ("entries", "captures", "hits", "replays", "invalidations")
            }
            for bucket, row in bucket_rows.items()
            if isinstance(row, Mapping)
        }
        return values
    stats = _first_stats_object(engine, ("graph_buckets", "graph_bucket_cache", "graph_bucket_stats"))
    if stats is None:
        return values
    data = _stats_to_mapping(stats)
    for key in ("entries", "hits", "misses"):
        values[key] = _non_negative_metric_value(data.get(key))
    lookups = values["hits"] + values["misses"]
    values["replay_hit_rate"] = values["hits"] / lookups if lookups > 0.0 else 0.0
    values["miss_reasons"] = _non_negative_metric_mapping(data.get("miss_reasons"))
    kernel_time_histogram = _non_negative_metric_mapping(data.get("kernel_time_histogram_ns"))
    known_kernel_time_histogram = {bucket: 0.0 for bucket in GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS}
    for bucket, value in kernel_time_histogram.items():
        if bucket in _GRAPH_KERNEL_TIME_HISTOGRAM_BUCKET_SET:
            known_kernel_time_histogram[bucket] = value
    values["kernel_time_histogram_ns"] = known_kernel_time_histogram
    return values


def _first_stats_object(engine: Any | None, names: Sequence[str]) -> Any | None:
    if engine is None:
        return None
    session = _resident_session_for_engine(engine)
    for owner in (engine, session):
        if owner is None:
            continue
        for name in names:
            candidate = getattr(owner, name, None)
            if candidate is None:
                continue
            stats = getattr(candidate, "stats", candidate)
            return stats() if callable(stats) else stats
    return None


def _stats_to_mapping(stats: Any) -> Mapping[str, Any]:
    if isinstance(stats, Mapping):
        return stats
    to_json = getattr(stats, "to_json_dict", None)
    if callable(to_json):
        data = to_json()
        if isinstance(data, Mapping):
            return data
    keys = (
        "current_bytes",
        "high_water_observed_bytes",
        "current_pages",
        "high_water_observed_pages",
        "grow_events",
        "grow_failures",
        "shrink_events",
        "free_pages",
        "refcounted_pages",
        "pinned_pages",
        "entries",
        "hits",
        "misses",
        "miss_reasons",
        "kernel_time_histogram_ns",
    )
    return {key: getattr(stats, key) for key in keys if hasattr(stats, key)}


def _non_negative_metric_value(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        return default
    return numeric


def _non_negative_metric_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for key, raw in value.items():
        numeric = _non_negative_metric_value(raw, default=-1.0)
        if numeric < 0:
            continue
        metrics[str(key)] = numeric
    return metrics


def _append_resident_latency_metrics(
    lines: list[str],
    values: Mapping[str, Mapping[str, float]],
) -> None:
    name = "hipengine_resident_request_latency_seconds"
    lines.append(f"# HELP {name} Bounded resident request latency summaries by timing kind.")
    lines.append(f"# TYPE {name} summary")
    max_name = "hipengine_resident_request_latency_max_seconds"
    lines.append(f"# HELP {max_name} Maximum bounded resident request latency by timing kind.")
    lines.append(f"# TYPE {max_name} gauge")
    for kind, row in sorted(values.items()):
        label = _escape_prometheus_label_value(str(kind))
        lines.append(
            f'{name}{{kind="{label}",quantile="0.5"}} '
            f'{_format_metric_value(row.get("p50", 0.0))}'
        )
        lines.append(
            f'{name}{{kind="{label}",quantile="0.95"}} '
            f'{_format_metric_value(row.get("p95", 0.0))}'
        )
        lines.append(
            f'{name}_sum{{kind="{label}"}} '
            f'{_format_metric_value(row.get("sum", 0.0))}'
        )
        lines.append(
            f'{name}_count{{kind="{label}"}} '
            f'{_format_metric_value(row.get("count", 0.0))}'
        )
        lines.append(
            f'{max_name}{{kind="{label}"}} '
            f'{_format_metric_value(row.get("max", 0.0))}'
        )


def _append_labeled_gauge_metrics(
    lines: list[str],
    name: str,
    help_text: str,
    label: str,
    values: Mapping[str, float],
) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} gauge")
    for key, value in sorted(values.items()):
        lines.append(f'{name}{{{label}="{_escape_prometheus_label_value(key)}"}} {_format_metric_value(value)}')


def _append_labeled_counter_metrics(lines: list[str], name: str, help_text: str, label: str, values: Mapping[str, float]) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} counter")
    for key, value in sorted(values.items()):
        lines.append(f'{name}{{{label}="{_escape_prometheus_label_value(key)}"}} {_format_metric_value(value)}')


def _escape_prometheus_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_metric_value(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return repr(numeric)


def _thinking_control_from_request(
    request: ChatCompletionRequest,
    *,
    chat_default_max_tokens: int | None = None,
    generation_budget: Any = _THINKING_BUDGET_UNSET,
) -> _ThinkingControl:
    enabled: bool | None = None
    effort: str | None = None
    max_think_tokens: int | None = None
    min_answer_tokens: int | None = None
    hard_think_cap: int | None = None
    soft_close_window: int | None = None
    hard_close_message: str | None = None
    hard_close_sequence: str | None = None
    allow_unbounded = False

    if isinstance(request.chat_template_kwargs, Mapping):
        enabled = _maybe_bool(request.chat_template_kwargs.get("enable_thinking"), enabled)
        effort = _maybe_effort(request.chat_template_kwargs.get("reasoning_effort"), effort)
        thinking_budget = request.chat_template_kwargs.get("thinking_budget")
        hard_think_cap = _maybe_budget_alias(
            thinking_budget,
            hard_think_cap,
            param="chat_template_kwargs.thinking_budget",
        )
        if hard_think_cap is None:
            effort = _maybe_effort(thinking_budget, effort)

    enabled = _maybe_bool(request.enable_thinking, enabled)
    effort = _maybe_effort(request.reasoning_effort, effort)
    hard_think_cap = _maybe_budget_alias(
        request.thinking_token_budget,
        hard_think_cap,
        param="thinking_token_budget",
    )
    max_think_tokens = _maybe_nonnegative_int(
        request.max_think_tokens,
        max_think_tokens,
        param="max_think_tokens",
    )
    min_answer_tokens = _maybe_nonnegative_int(
        request.min_answer_tokens,
        min_answer_tokens,
        param="min_answer_tokens",
    )
    hard_think_cap = _maybe_nonnegative_int(
        request.hard_think_cap,
        hard_think_cap,
        param="hard_think_cap",
    )
    soft_close_window = _maybe_nonnegative_int(
        request.soft_close_window,
        soft_close_window,
        param="soft_close_window",
    )
    hard_close_message = _maybe_text(request.hard_close_message, hard_close_message)
    hard_close_sequence = _maybe_text(request.hard_close_sequence, hard_close_sequence)
    if _effort_disables_thinking(effort):
        enabled = False

    if isinstance(request.thinking, Mapping):
        thinking_type = str(request.thinking.get("type", "")).strip().lower()
        requested_enabled = _maybe_bool(request.thinking.get("enabled"), enabled)
        if thinking_type in {"disabled", "disable", "off", "none"}:
            enabled = False
        elif requested_enabled is False:
            enabled = False
        elif thinking_type in {"enabled", "enable", "on"}:
            enabled = True
        else:
            enabled = requested_enabled
        allow_unbounded = bool(_maybe_bool(request.thinking.get("allow_unbounded"), allow_unbounded))
        effort = _maybe_effort(request.thinking.get("effort"), effort)
        budget_tokens = request.thinking.get("budget_tokens")
        budget_cap = _coerce_nonnegative_int(
            budget_tokens,
            param="thinking.budget_tokens",
            allow_text_alias=True,
        )
        if budget_cap is None:
            effort = _maybe_effort(budget_tokens, effort)
        else:
            hard_think_cap = budget_cap
        hard_think_cap = _maybe_nonnegative_int(
            request.thinking.get("max_tokens"),
            hard_think_cap,
            param="thinking.max_tokens",
        )
        max_think_tokens = _maybe_nonnegative_int(
            request.thinking.get("max_think_tokens"),
            max_think_tokens,
            param="thinking.max_think_tokens",
        )
        min_answer_tokens = _maybe_nonnegative_int(
            request.thinking.get("min_answer_tokens"),
            min_answer_tokens,
            param="thinking.min_answer_tokens",
        )
        hard_think_cap = _maybe_nonnegative_int(
            request.thinking.get("hard_think_cap"),
            hard_think_cap,
            param="thinking.hard_think_cap",
        )
        soft_close_window = _maybe_nonnegative_int(
            request.thinking.get("soft_close_window"),
            soft_close_window,
            param="thinking.soft_close_window",
        )
        hard_close_message = _maybe_text(
            request.thinking.get("hard_close_message"),
            hard_close_message,
        )
        hard_close_sequence = _maybe_text(
            request.thinking.get("hard_close_sequence"),
            hard_close_sequence,
        )
    elif isinstance(request.thinking, str):
        effort = _maybe_effort(request.thinking, effort)
        if _effort_disables_thinking(effort):
            enabled = False

    if isinstance(request.reasoning, Mapping):
        reasoning_type = str(request.reasoning.get("type", "")).strip().lower()
        requested_enabled = _maybe_bool(request.reasoning.get("enabled"), enabled)
        if reasoning_type in {"disabled", "disable", "off", "none"}:
            enabled = False
        elif requested_enabled is False:
            enabled = False
        elif reasoning_type in {"enabled", "enable", "on"}:
            enabled = True
        else:
            enabled = requested_enabled
        allow_unbounded = bool(_maybe_bool(request.reasoning.get("allow_unbounded"), allow_unbounded))
        effort = _maybe_effort(request.reasoning.get("effort"), effort)
        budget_tokens = request.reasoning.get("budget_tokens")
        budget_cap = _coerce_nonnegative_int(
            budget_tokens,
            param="reasoning.budget_tokens",
            allow_text_alias=True,
        )
        if budget_cap is None:
            effort = _maybe_effort(budget_tokens, effort)
        else:
            hard_think_cap = budget_cap
        hard_think_cap = _maybe_nonnegative_int(
            request.reasoning.get("max_tokens"),
            hard_think_cap,
            param="reasoning.max_tokens",
        )
        max_think_tokens = _maybe_nonnegative_int(
            request.reasoning.get("max_think_tokens"),
            max_think_tokens,
            param="reasoning.max_think_tokens",
        )
        min_answer_tokens = _maybe_nonnegative_int(
            request.reasoning.get("min_answer_tokens"),
            min_answer_tokens,
            param="reasoning.min_answer_tokens",
        )
        hard_think_cap = _maybe_nonnegative_int(
            request.reasoning.get("hard_think_cap"),
            hard_think_cap,
            param="reasoning.hard_think_cap",
        )
        soft_close_window = _maybe_nonnegative_int(
            request.reasoning.get("soft_close_window"),
            soft_close_window,
            param="reasoning.soft_close_window",
        )
        hard_close_message = _maybe_text(
            request.reasoning.get("hard_close_message"),
            hard_close_message,
        )
        hard_close_sequence = _maybe_text(
            request.reasoning.get("hard_close_sequence"),
            hard_close_sequence,
        )

    if _effort_disables_thinking(effort):
        enabled = False
    if enabled is not False and not _effort_disables_thinking(effort):
        if generation_budget is _THINKING_BUDGET_UNSET:
            generation_budget = _thinking_generation_budget(
                request,
                chat_default_max_tokens=chat_default_max_tokens,
            )
        if allow_unbounded and hard_think_cap is None:
            defaults = _THINKING_EFFORT_DEFAULTS.get(str(effort or "").strip().lower())
            if defaults is not None and min_answer_tokens is None:
                min_answer_tokens = int(defaults["min_answer_tokens"])
            hard_think_cap, min_answer_tokens, soft_close_window = _clamp_thinking_budget_hints(
                generation_budget=None if generation_budget is None else int(generation_budget),
                hard_think_cap=None,
                min_answer_tokens=min_answer_tokens,
                soft_close_window=None,
            )
        else:
            hard_think_cap, min_answer_tokens, soft_close_window = _apply_thinking_effort_defaults(
                effort,
                generation_budget=None if generation_budget is None else int(generation_budget),
                hard_think_cap=hard_think_cap,
                min_answer_tokens=min_answer_tokens,
                soft_close_window=soft_close_window,
            )
    control = _ThinkingControl(
        enabled=enabled,
        effort=effort,
        allow_unbounded=allow_unbounded,
        max_think_tokens=max_think_tokens,
        min_answer_tokens=min_answer_tokens,
        hard_think_cap=hard_think_cap,
        soft_close_window=soft_close_window,
        hard_close_message=hard_close_message,
        hard_close_sequence=hard_close_sequence,
    )
    _validate_thinking_control(control)
    return control


def _model_chat_protocol_target(engine: Any | None) -> Any | None:
    if engine is None:
        return None
    return getattr(engine, "_text_generator", None) or engine


def _model_template_thinking_enabled(thinking: _ThinkingControl) -> bool:
    if thinking.enabled is not None:
        return bool(thinking.enabled)
    return bool(
        thinking.effort
        or thinking.allow_unbounded
        or thinking.max_think_tokens is not None
        or thinking.min_answer_tokens is not None
        or thinking.hard_think_cap is not None
        or thinking.soft_close_window is not None
        or thinking.hard_close_message
        or thinking.hard_close_sequence
    )


def _render_chat_prompt_with_model_protocol(
    request: ChatCompletionRequest,
    *,
    thinking: _ThinkingControl,
    engine: Any | None,
    validate_tool_transcript: bool,
) -> str:
    target = _model_chat_protocol_target(engine)
    renderer = getattr(target, "render_chat_prompt", None)
    if not callable(renderer):
        return render_chat_prompt(
            request.messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            thinking=thinking,
            response_format=request.response_format,
            guided_json=request.guided_json,
            guided_regex=request.guided_regex,
            guided_choice=request.guided_choice,
            guided_patch=request.guided_patch,
            guided_diff=request.guided_diff,
            validate_tool_transcript=validate_tool_transcript,
        )

    if validate_tool_transcript:
        # Keep the generic API's role/tool-transcript validation in front of a
        # model-owned renderer. The discarded rendering has no side effects.
        render_chat_prompt(
            request.messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            thinking=thinking,
            response_format=request.response_format,
            guided_json=request.guided_json,
            guided_regex=request.guided_regex,
            guided_choice=request.guided_choice,
            guided_patch=request.guided_patch,
            guided_diff=request.guided_diff,
            validate_tool_transcript=True,
        )
    tools = request.tools if _tool_choice_name(request.tool_choice) != "none" else None
    return str(
        renderer(
            request.messages,
            tools=tools,
            enable_thinking=_model_template_thinking_enabled(thinking),
            add_generation_prompt=True,
        )
    )


def _render_prepared_chat_prompt_for_request(
    request: ChatCompletionRequest,
    *,
    chat_default_max_tokens: int | None,
    engine: Any | None = None,
    max_context_tokens: int | None = None,
    validate_tool_transcript: bool = True,
) -> tuple[str, _ThinkingControl, PromptInput]:
    render_ms = 0.0
    encode_ms = 0.0

    def render(thinking: _ThinkingControl) -> str:
        nonlocal render_ms
        started = time.perf_counter()
        value = _render_chat_prompt_with_model_protocol(
            request,
            thinking=thinking,
            engine=engine,
            validate_tool_transcript=validate_tool_transcript,
        )
        render_ms += (time.perf_counter() - started) * 1_000.0
        return value

    def prepare(prompt: str) -> PromptInput:
        nonlocal encode_ms
        if engine is None:
            return prompt
        prepared = _prepare_prompt_input(engine, prompt)
        if isinstance(prepared, PreparedPromptInput):
            encode_ms += prepared.tokenize_ms
        return prepared

    def finish(
        prompt: str,
        thinking: _ThinkingControl,
        prepared: PromptInput,
    ) -> tuple[str, _ThinkingControl, PromptInput]:
        if isinstance(prepared, PreparedPromptInput):
            prepared = replace(
                prepared,
                tokenize_ms=encode_ms,
                render_ms=render_ms,
            )
        return prompt, thinking, prepared

    thinking = _thinking_control_from_request(
        request,
        chat_default_max_tokens=chat_default_max_tokens,
    )
    prompt = render(thinking)
    if engine is None:
        return prompt, thinking, prompt

    prepared = prepare(prompt)
    for _ in range(4):
        generation_budget = _thinking_generation_budget_for_prompt(
            request,
            prepared,
            engine,
            max_context_tokens,
            chat_default_max_tokens=chat_default_max_tokens,
        )
        adjusted = _thinking_control_from_request(
            request,
            chat_default_max_tokens=chat_default_max_tokens,
            generation_budget=generation_budget,
        )
        if adjusted == thinking:
            return finish(prompt, thinking, prepared)
        adjusted_prompt = render(adjusted)
        if adjusted_prompt == prompt:
            return finish(prompt, adjusted, prepared)
        thinking = adjusted
        prompt = adjusted_prompt
        prepared = prepare(prompt)
    return finish(prompt, thinking, prepared)


def _render_chat_prompt_for_request(
    request: ChatCompletionRequest,
    *,
    chat_default_max_tokens: int | None,
    engine: Any | None = None,
    max_context_tokens: int | None = None,
    validate_tool_transcript: bool = True,
) -> tuple[str, _ThinkingControl]:
    prompt, thinking, _prepared = _render_prepared_chat_prompt_for_request(
        request,
        chat_default_max_tokens=chat_default_max_tokens,
        engine=engine,
        max_context_tokens=max_context_tokens,
        validate_tool_transcript=validate_tool_transcript,
    )
    return prompt, thinking


def _thinking_generation_budget_for_prompt(
    request: ChatCompletionRequest,
    prompt: PromptInput,
    engine: Any,
    max_context_tokens: int | None,
    *,
    chat_default_max_tokens: int | None,
) -> int | None:
    budget = _thinking_generation_budget(
        request,
        chat_default_max_tokens=chat_default_max_tokens,
    )
    remaining = _remaining_context_tokens((prompt,), engine, max_context_tokens)
    if remaining is None:
        return budget
    remaining_budget = max(0, int(remaining))
    if budget is None:
        return remaining_budget
    return min(max(0, int(budget)), remaining_budget)


def _thinking_generation_budget(
    request: ChatCompletionRequest,
    *,
    chat_default_max_tokens: int | None,
) -> int | None:
    cap = _request_completion_cap(request)
    if cap is not None:
        return cap
    if chat_default_max_tokens is not None:
        return max(0, int(chat_default_max_tokens))
    return None


def _apply_thinking_effort_defaults(
    effort: str | None,
    *,
    generation_budget: int | None,
    hard_think_cap: int | None,
    min_answer_tokens: int | None,
    soft_close_window: int | None,
) -> tuple[int | None, int | None, int | None]:
    defaults = _THINKING_EFFORT_DEFAULTS.get(str(effort or "").strip().lower())
    if defaults is not None:
        hard_think_cap = int(defaults["hard_think_cap"]) if hard_think_cap is None else hard_think_cap
        min_answer_tokens = int(defaults["min_answer_tokens"]) if min_answer_tokens is None else min_answer_tokens
        soft_close_window = int(defaults["soft_close_window"]) if soft_close_window is None else soft_close_window
    return _clamp_thinking_budget_hints(
        generation_budget=generation_budget,
        hard_think_cap=hard_think_cap,
        min_answer_tokens=min_answer_tokens,
        soft_close_window=soft_close_window,
    )


def _clamp_thinking_budget_hints(
    *,
    generation_budget: int | None,
    hard_think_cap: int | None,
    min_answer_tokens: int | None,
    soft_close_window: int | None,
) -> tuple[int | None, int | None, int | None]:
    if generation_budget is None:
        return hard_think_cap, min_answer_tokens, soft_close_window
    budget = max(0, int(generation_budget))
    if hard_think_cap == 0:
        min_answer_tokens = budget if min_answer_tokens is None else min(int(min_answer_tokens), budget)
        return 0, min_answer_tokens, 0 if soft_close_window is not None else soft_close_window
    if min_answer_tokens is not None:
        min_answer_tokens = min(int(min_answer_tokens), max(0, budget // 2))
    reserved = 0 if min_answer_tokens is None else int(min_answer_tokens)
    if hard_think_cap is not None:
        hard_think_cap = min(int(hard_think_cap), max(0, budget - reserved))
    if soft_close_window is not None:
        soft_limit = budget if hard_think_cap is None else int(hard_think_cap)
        soft_close_window = min(int(soft_close_window), max(0, soft_limit))
    return (
        hard_think_cap,
        min_answer_tokens,
        soft_close_window,
    )


def _maybe_bool(value: Any, current: bool | None) -> bool | None:
    if value is None:
        return current
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled", "enable"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled", "disable", "none"}:
            return False
    return current


def _maybe_effort(value: Any, current: str | None) -> str | None:
    if value is None:
        return current
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "low" if float(value) <= 1024 else "medium" if float(value) <= 4096 else "high"
    text = str(value).strip().lower()
    return text or current


def _maybe_budget_alias(value: Any, current: int | None, *, param: str) -> int | None:
    budget = _coerce_nonnegative_int(value, param=param, allow_text_alias=True)
    return current if budget is None else budget


def _maybe_nonnegative_int(value: Any, current: int | None, *, param: str) -> int | None:
    number = _coerce_nonnegative_int(value, param=param, allow_text_alias=False)
    return current if number is None else number


def _coerce_nonnegative_int(value: Any, *, param: str, allow_text_alias: bool) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        _raise_invalid_nonnegative_int(param)
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not value.is_integer():
            _raise_invalid_nonnegative_int(param)
        number = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            if allow_text_alias:
                return None
            _raise_invalid_nonnegative_int(param)
        try:
            number = int(text, 10)
        except ValueError:
            if allow_text_alias:
                return None
            _raise_invalid_nonnegative_int(param)
    else:
        _raise_invalid_nonnegative_int(param)
    if number < 0:
        _raise_invalid_nonnegative_int(param)
    return number


def _raise_invalid_nonnegative_int(param: str) -> None:
    raise OpenAIHTTPError(
        400,
        f"{param} must be a non-negative integer",
        code="invalid_request",
        param=param,
    )


def _maybe_text(value: Any, current: str | None) -> str | None:
    if value is None:
        return current
    return str(value)


def _effort_disables_thinking(effort: str | None) -> bool:
    return effort in {"0", "false", "none", "off", "disabled", "disable", "nothink", "no_think"}


def _validate_thinking_control(control: _ThinkingControl) -> None:
    sequence = control.hard_close_sequence
    if sequence is not None and _THINKING_CLOSE_MARKER not in sequence:
        raise OpenAIHTTPError(
            400,
            f"hard_close_sequence must contain {_THINKING_CLOSE_MARKER!r}",
            code="invalid_request",
            param="hard_close_sequence",
        )


def _render_thinking_prompt(thinking: _ThinkingControl | None) -> str:
    if thinking is None:
        return ""
    if thinking.enabled is False:
        return "Do not include hidden reasoning. Answer directly after the pre-closed <think></think> block."
    effort = thinking.effort
    hints: list[str] = []
    if effort and not _effort_disables_thinking(effort):
        if effort in {"minimal", "low"}:
            limit = "very brief"
        elif effort == "medium":
            limit = "concise"
        elif effort in {"high", "xhigh", "max"}:
            limit = "focused but complete"
        else:
            limit = "concise"
        hints.append(f"keep it {limit}")
    if thinking.max_think_tokens is not None:
        hints.append(f"aim to close hidden reasoning within {thinking.max_think_tokens} tokens")
    if thinking.hard_think_cap is not None:
        hints.append(
            f"close {_THINKING_CLOSE_MARKER} before exceeding "
            f"{thinking.hard_think_cap} hidden reasoning tokens"
        )
    if thinking.min_answer_tokens is not None:
        hints.append(
            f"reserve at least {thinking.min_answer_tokens} tokens for "
            "the final answer or tool call"
        )
    if thinking.soft_close_window is not None:
        hints.append(
            f"begin closing during the final {thinking.soft_close_window} "
            "hidden reasoning tokens"
        )
    if thinking.hard_close_message:
        hints.append(
            f"use the close message {thinking.hard_close_message!r} "
            "only if budget pressure requires it"
        )
    if thinking.hard_close_sequence:
        hints.append(
            f"use {thinking.hard_close_sequence!r} as the close sequence "
            "if budget pressure requires it"
        )
    if not hints:
        return ""
    hint_text = "; ".join(hints)
    return (
        f"If you use <think> reasoning, {hint_text}; when ready, close {_THINKING_CLOSE_MARKER} "
        "before emitting the final answer or any <tool_call> block."
    )


def _assistant_prefix_for_thinking(thinking: _ThinkingControl | None) -> str:
    prefix = "<|im_start|>assistant\n"
    if thinking is not None and thinking.enabled is False:
        return prefix + "<think>\n\n</think>\n\n"
    return prefix


def _render_response_format_prompt(response_format: Any | None) -> str:
    if response_format is None:
        return ""
    response_type = ""
    if isinstance(response_format, str):
        response_type = response_format.strip().lower()
    elif isinstance(response_format, Mapping):
        response_type = str(response_format.get("type", "")).strip().lower()
    if response_type == "json_object":
        return (
            "Return only one valid JSON object in the final answer. Do not wrap it in "
            "Markdown, prose, arrays, or scalar JSON values."
        )
    if response_type == "json_schema":
        schema = _response_format_json_schema_from_value(response_format)
        if schema is None:
            return "Return only JSON that satisfies the requested JSON schema."
        schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        return f"Return only JSON that satisfies this JSON schema: {schema_text}"
    return ""


def _render_guided_json_prompt(guided_json: Any | None) -> str:
    mode = _guided_json_mode_from_value(guided_json, validate=False)
    if mode is None:
        return ""
    if mode == "json_object":
        return (
            "Return only one valid JSON object in the final answer. Do not wrap it in "
            "Markdown, prose, arrays, or scalar JSON values."
        )
    schema = _guided_json_schema_from_value(guided_json)
    if schema is None:
        return "Return only JSON that satisfies the requested JSON schema."
    schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return f"Return only JSON that satisfies this JSON schema: {schema_text}"


def _render_guided_choice_prompt(guided_choice: Any | None) -> str:
    if guided_choice is None:
        return ""
    choices = _guided_choice_values(guided_choice, validate=False)
    if not choices:
        return ""
    choices_text = json.dumps(list(choices), ensure_ascii=False, separators=(",", ":"))
    return f"Return exactly one of these choices and no other text: {choices_text}"


def _render_guided_regex_prompt(guided_regex: Any | None) -> str:
    if guided_regex is None:
        return ""
    pattern = _guided_regex_pattern(guided_regex, validate=False)
    if pattern is None:
        return ""
    pattern_text = json.dumps(pattern, ensure_ascii=False, separators=(",", ":"))
    return f"Return text that fully matches this regular expression and no other text: {pattern_text}"


def _render_guided_patch_prompt(guided_patch: Any | None, guided_diff: Any | None) -> str:
    mode = _guided_patch_field(guided_patch, guided_diff)
    if mode is None:
        return ""
    return (
        "Return only a valid unified diff patch. Do not include prose outside "
        "the patch. A single fenced diff or patch code block is allowed."
    )


def _render_tools_prompt(
    tools: Sequence[Mapping[str, Any]] | None,
    tool_choice: str | Mapping[str, Any] | None,
) -> str:
    if not tools or _tool_choice_name(tool_choice) == "none":
        if _tool_choice_name(tool_choice) == "none":
            return "Do not call tools for this response."
        return ""
    tool_lines = [json.dumps(dict(tool), ensure_ascii=False, separators=(",", ":")) for tool in tools]
    directive = _tool_choice_directive(tool_choice)
    return "\n".join(
        [
            "You may call one or more functions to assist with the user request.",
            "Available functions are provided in JSON schema form inside <tools></tools> tags:",
            "<tools>",
            *tool_lines,
            "</tools>",
            directive,
            "For each function call, respond with a JSON object inside <tool_call></tool_call> tags and no extra prose:",
            '<tool_call>{"name":"function_name","arguments":{"arg":"value"}}</tool_call>',
        ]
    )


def _tool_choice_name(tool_choice: str | Mapping[str, Any] | None) -> str | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower()
    choice_type = str(tool_choice.get("type", "")).strip().lower()
    if choice_type == "function":
        function = tool_choice.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            return None if name is None else str(name)
    return choice_type or None


def _tool_choice_directive(tool_choice: str | Mapping[str, Any] | None) -> str:
    name = _tool_choice_name(tool_choice)
    if name == "required":
        return "You must call at least one function."
    if name and name not in {"auto", "none"}:
        return f"You must call the function named {name!r}."
    return "Call a function only when it is useful; otherwise answer normally."


def _render_tool_call_for_prompt(tool_call: Mapping[str, Any]) -> str:
    function = tool_call.get("function")
    if isinstance(function, Mapping):
        name = str(function.get("name", ""))
        raw_arguments = function.get("arguments", {})
    else:
        name = str(tool_call.get("name", ""))
        raw_arguments = tool_call.get("arguments", {})
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except Exception:
            arguments = raw_arguments
    else:
        arguments = raw_arguments
    payload = {"name": name, "arguments": arguments if arguments is not None else {}}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</tool_call>"


def render_chat_prompt(
    messages: Sequence[ChatMessage | Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    tool_choice: str | Mapping[str, Any] | None = None,
    thinking: _ThinkingControl | None = None,
    response_format: Any | None = None,
    guided_json: Any | None = None,
    guided_regex: Any | None = None,
    guided_choice: Any | None = None,
    guided_patch: Any | None = None,
    guided_diff: Any | None = None,
    validate_tool_transcript: bool = True,
) -> str:
    """Render OpenAI chat messages to a Qwen-style text prompt.

    This is intentionally tokenizer-independent so the API can stay a thin
    adapter around ``LLM.generate()``.  Model-specific chat-template rendering
    can replace this helper once tokenizers are exposed by the runtime.
    """

    if not messages:
        raise OpenAIHTTPError(400, "messages must contain at least one item", param="messages")
    rendered: list[str] = []
    control_prompts = [
        item
        for item in (
            _render_thinking_prompt(thinking),
            _render_response_format_prompt(response_format),
            _render_guided_json_prompt(guided_json),
            _render_guided_regex_prompt(guided_regex),
            _render_guided_choice_prompt(guided_choice),
            _render_guided_patch_prompt(guided_patch, guided_diff),
            _render_tools_prompt(tools, tool_choice),
        )
        if item
    ]
    if control_prompts:
        control_block = "\n\n".join(control_prompts)
        rendered.append(f"<|im_start|>system\n{control_block}<|im_end|>")
    pending_tool_call_ids: set[str] = set()
    seen_tool_call_ids: set[str] = set()
    for index, message in enumerate(messages):
        if isinstance(message, Mapping):
            role_value = message.get("role", "")
            content_value = message.get("content", "")
            tool_calls = message.get("tool_calls")
            tool_call_id = message.get("tool_call_id")
        else:
            role_value = message.role
            content_value = message.content
            tool_calls = message.tool_calls
            tool_call_id = message.tool_call_id
        role = _chat_message_role(role_value, param=f"messages[{index}].role")
        _validate_chat_message_tool_fields(
            role,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            param_prefix=f"messages[{index}]",
        )
        normalized_tool_calls: list[dict[str, Any]] = []
        if role == "assistant" and tool_calls is not None:
            normalized_tool_calls = _chat_message_tool_calls(
                tool_calls,
                message_index=index,
                context="chat message",
            )
        if validate_tool_transcript:
            _validate_chat_message_tool_transcript_entry(
                role,
                tool_calls=normalized_tool_calls,
                tool_call_id=tool_call_id,
                message_index=index,
                context="chat message",
                pending_tool_call_ids=pending_tool_call_ids,
                seen_tool_call_ids=seen_tool_call_ids,
            )
        content = _message_content_text(content_value, index)
        if role == "developer":
            role = "system"
        if role == "tool":
            rendered.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>")
            continue
        if role == "assistant" and normalized_tool_calls:
            tool_call_text = "\n".join(_render_tool_call_for_prompt(item) for item in normalized_tool_calls)
            content = "\n".join(part for part in (content, tool_call_text) if part)
        rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    rendered.append(_assistant_prefix_for_thinking(thinking))
    return "\n".join(rendered)


def _chat_message_role(value: Any, *, param: str) -> str:
    if not isinstance(value, str):
        raise OpenAIHTTPError(
            400,
            f"{param} must be one of: {', '.join(_CHAT_MESSAGE_ROLES)}",
            code="invalid_request",
            param=param,
        )
    role = value.strip()
    if role not in _CHAT_MESSAGE_ROLE_SET:
        raise OpenAIHTTPError(
            400,
            f"{param} must be one of: {', '.join(_CHAT_MESSAGE_ROLES)}",
            code="invalid_request",
            param=param,
        )
    return role


def _validate_chat_message_tool_fields(
    role: str,
    *,
    tool_calls: Any,
    tool_call_id: Any,
    param_prefix: str,
) -> None:
    if tool_calls is not None and role != "assistant":
        raise OpenAIHTTPError(
            400,
            f"{param_prefix}.tool_calls is only supported on assistant messages",
            code="invalid_request",
            param=f"{param_prefix}.tool_calls",
        )
    if tool_call_id is not None and role != "tool":
        raise OpenAIHTTPError(
            400,
            f"{param_prefix}.tool_call_id is only supported on tool messages",
            code="invalid_request",
            param=f"{param_prefix}.tool_call_id",
        )
    if role == "tool" and (not isinstance(tool_call_id, str) or not tool_call_id.strip()):
        raise OpenAIHTTPError(
            400,
            f"{param_prefix}.tool_call_id is required for tool messages",
            code="invalid_request",
            param=f"{param_prefix}.tool_call_id",
        )


def _chat_message_tool_calls(
    value: Any,
    *,
    message_index: int,
    context: str,
) -> list[dict[str, Any]]:
    param = f"messages[{message_index}].tool_calls"
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise OpenAIHTTPError(
            400,
            f"{context} {param} must be an array",
            code="invalid_request",
            param=param,
        )
    return [
        _chat_message_tool_call(
            call,
            message_index=message_index,
            tool_index=tool_index,
            context=context,
        )
        for tool_index, call in enumerate(value)
    ]


def _validate_chat_message_tool_transcript(
    messages: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> None:
    pending_tool_call_ids: set[str] = set()
    seen_tool_call_ids: set[str] = set()
    for index, message in enumerate(messages):
        tool_calls = message.get("tool_calls")
        _validate_chat_message_tool_transcript_entry(
            str(message.get("role") or ""),
            tool_calls=tool_calls if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes)) else [],
            tool_call_id=message.get("tool_call_id"),
            message_index=index,
            context=context,
            pending_tool_call_ids=pending_tool_call_ids,
            seen_tool_call_ids=seen_tool_call_ids,
        )


def _validate_chat_message_tool_transcript_entry(
    role: str,
    *,
    tool_calls: Sequence[Mapping[str, Any]],
    tool_call_id: Any,
    message_index: int,
    context: str,
    pending_tool_call_ids: set[str],
    seen_tool_call_ids: set[str],
) -> None:
    if pending_tool_call_ids and role != "tool":
        raise OpenAIHTTPError(
            400,
            f"{context} messages[{message_index}].role must be 'tool' until prior assistant tool calls are resolved",
            code="invalid_request",
            param=f"messages[{message_index}].role",
        )
    if role == "assistant":
        for tool_index, tool_call in enumerate(tool_calls):
            call_id = str(tool_call.get("id") or "")
            if call_id in seen_tool_call_ids:
                param = f"messages[{message_index}].tool_calls[{tool_index}].id"
                raise OpenAIHTTPError(
                    400,
                    f"{context} {param} duplicates a prior assistant tool call id",
                    code="invalid_request",
                    param=param,
                )
            seen_tool_call_ids.add(call_id)
            pending_tool_call_ids.add(call_id)
        return
    if role != "tool":
        return
    call_id = str(tool_call_id or "").strip()
    if call_id not in pending_tool_call_ids:
        raise OpenAIHTTPError(
            400,
            f"{context} messages[{message_index}].tool_call_id must reference a prior unconsumed assistant tool call",
            code="invalid_request",
            param=f"messages[{message_index}].tool_call_id",
        )
    pending_tool_call_ids.remove(call_id)


def _validate_model(config: ServerConfig, requested: str | None, *, engine: Any | None = None) -> None:
    if requested is not None and requested != config.model_id:
        raise OpenAIHTTPError(
            404,
            f"model {requested!r} is not served by this hipEngine instance",
            code="model_not_found",
            param="model",
            extra={
                "hipengine": {
                    "routing": _routing_failure_metadata(
                        config,
                        requested_model=requested,
                        reason="model_unavailable",
                        engine=engine,
                    )
                }
            },
        )


def _log_kv_capacity_summary(engine: Any) -> None:
    session = _resident_session_for_engine(engine)
    if session is None:
        return
    estimate = getattr(session, "kv_capacity_estimate", None)
    if estimate is not None:
        model_max = int(getattr(estimate, "model_max_context_tokens", 0) or 0)
        _LOGGER.info(
            "KVCache: storage=%s scale=%s max_context_tokens=%d model_max_context_tokens=%s "
            "allocatable_context_tokens=%d requested_kv=%s metadata=%s total=%s "
            "bytes_per_token=%d usable=%s reserve=%s",
            getattr(estimate, "kv_storage_dtype", "unknown"),
            getattr(estimate, "kv_scale_dtype", None) or "none",
            int(getattr(estimate, "requested_context_tokens", 0) or 0),
            "unknown" if model_max <= 0 else str(model_max),
            int(getattr(estimate, "allocatable_context_tokens", 0) or 0),
            _format_bytes(int(getattr(estimate, "requested_kv_bytes", 0) or 0)),
            _format_bytes(int(getattr(estimate, "requested_context_overhead_bytes", 0) or 0)),
            _format_bytes(int(getattr(estimate, "requested_total_bytes", 0) or 0)),
            int(getattr(estimate, "bytes_per_token", 0) or 0),
            _format_bytes(int(getattr(estimate, "usable_bytes", 0) or 0)),
            _format_bytes(int(getattr(estimate, "reserve_bytes", 0) or 0)),
        )
        if model_max > 0 and not bool(getattr(estimate, "fits_model_max", True)):
            _LOGGER.warning(
                "KVCache: selected policy can fit allocatable_context_tokens=%d, "
                "below model_max_context_tokens=%d",
                int(getattr(estimate, "allocatable_context_tokens", 0) or 0),
                model_max,
            )
    int8_estimate = getattr(session, "kv_capacity_int8_estimate", None)
    if int8_estimate is None:
        return
    int8_model_max = int(getattr(int8_estimate, "model_max_context_tokens", 0) or 0)
    if int8_model_max > 0 and not bool(getattr(int8_estimate, "fits_model_max", True)):
        _LOGGER.warning(
            "KVCache: int8_per_token_head can fit allocatable_context_tokens=%d, "
            "below model_max_context_tokens=%d",
            int(getattr(int8_estimate, "allocatable_context_tokens", 0) or 0),
            int8_model_max,
        )


def _kv_capacity_estimate_payload(engine: Any | None) -> dict[str, Any] | None:
    if engine is None:
        return None
    session = _resident_session_for_engine(engine)
    estimate = None
    for owner in (session, engine):
        if owner is None:
            continue
        estimate = getattr(owner, "kv_capacity_estimate", None)
        if estimate is not None:
            break
    if estimate is None:
        return None
    payload: dict[str, Any] = {}
    for name in (
        "requested_context_tokens",
        "model_max_context_tokens",
        "allocatable_context_tokens",
        "requested_kv_bytes",
        "bytes_per_token",
        "usable_bytes",
        "reserve_bytes",
        "fits_model_max",
        "kv_storage_dtype",
        "kv_scale_dtype",
    ):
        if not hasattr(estimate, name):
            continue
        value = getattr(estimate, name)
        if isinstance(value, bool) or value is None:
            payload[name] = value
        elif isinstance(value, (int, float, str)):
            payload[name] = value
    return payload or None


def _selected_device_payload(config: ServerConfig, engine: Any | None = None) -> dict[str, Any]:
    hip_visible = os.environ.get("HIP_VISIBLE_DEVICES")
    rocr_visible = os.environ.get("ROCR_VISIBLE_DEVICES")
    source, visible_devices = _visible_device_selection(
        hip_visible_devices=hip_visible,
        rocr_visible_devices=rocr_visible,
    )
    return {
        "backend": _server_model_identity(config, engine)["backend"],
        "hip_visible_devices": hip_visible,
        "rocr_visible_devices": rocr_visible,
        "visible_devices": visible_devices,
        "selected_visible_device": None if not visible_devices else visible_devices[0],
        "selection_source": source,
    }


def _visible_device_selection(
    *,
    hip_visible_devices: str | None,
    rocr_visible_devices: str | None,
) -> tuple[str | None, list[str]]:
    for source, raw in (
        ("HIP_VISIBLE_DEVICES", hip_visible_devices),
        ("ROCR_VISIBLE_DEVICES", rocr_visible_devices),
    ):
        devices = _parse_visible_devices(raw)
        if devices:
            return source, devices
    return None, []


def _parse_visible_devices(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _resident_session_for_engine(engine: Any) -> Any | None:
    if hasattr(engine, "kv_capacity_estimate"):
        return engine
    generator = getattr(engine, "_text_generator", None)
    if generator is not None:
        session = getattr(generator, "_session", None)
        if session is not None:
            return session
        resident_owner = getattr(generator, "_resident_model_runner", None)
        session = getattr(resident_owner, "_session", None)
        if session is not None:
            return session
    session = getattr(engine, "_session", None)
    if session is not None:
        return session
    return None


def _prepared_context_tokens(engine: Any) -> int | None:
    session = _resident_session_for_engine(engine)
    if session is None:
        return None
    value = getattr(session, "max_sequence_length", None)
    if value is None:
        return None
    return max(1, int(value))


def _format_bytes(value: int) -> str:
    return f"{int(value) / 1024**3:.2f} GiB"


def _device_memory_snapshot() -> dict[str, Any] | None:
    try:
        from hipengine.core.hip import get_hip_runtime

        runtime = get_hip_runtime()
        free_bytes, total_bytes = runtime.mem_get_info()
    except Exception:
        return None
    free = int(free_bytes)
    total = int(total_bytes)
    return {
        "free_bytes": free,
        "total_bytes": total,
        "used_bytes": max(0, total - free),
        "free_gib": round(free / 1024**3, 6),
        "used_gib": round(max(0, total - free) / 1024**3, 6),
    }


def _startup_max_prompt_tokens(max_context_tokens: int | None) -> int | None:
    if max_context_tokens is None:
        return None
    return max(1, int(max_context_tokens) - 1)


def _record_startup_memory_snapshot(target: dict[str, Any], stage: str) -> dict[str, Any] | None:
    snapshot = _device_memory_snapshot()
    if snapshot is not None:
        target[str(stage)] = snapshot
        _LOGGER.debug(
            "STARTUP_MEMORY_SAMPLE: stage=%s free=%s used=%s total=%s",
            stage,
            _format_bytes(int(snapshot["free_bytes"])),
            _format_bytes(int(snapshot["used_bytes"])),
            _format_bytes(int(snapshot["total_bytes"])),
        )
    return snapshot


def _startup_memory_samples(
    memory: Mapping[str, Any],
    checks: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    samples: list[tuple[str, Mapping[str, Any]]] = []
    for stage, snapshot in memory.items():
        if isinstance(snapshot, Mapping) and "free_bytes" in snapshot and "used_bytes" in snapshot:
            samples.append((str(stage), snapshot))
    scratch = checks.get("scratch_probe")
    if isinstance(scratch, Mapping):
        result = scratch.get("result")
        if isinstance(result, Mapping):
            live_memory = result.get("live_memory")
            if isinstance(live_memory, Mapping) and "free_bytes" in live_memory and "used_bytes" in live_memory:
                inner_stage = str(live_memory.get("stage") or "live")
                samples.append((f"scratch_probe:{inner_stage}", live_memory))
    return samples


def _startup_memory_summary(
    memory: Mapping[str, Any],
    checks: Mapping[str, Any],
) -> dict[str, Any] | None:
    memory_samples = _startup_memory_samples(memory, {})
    samples = _startup_memory_samples(memory, checks)
    if not samples:
        return None
    final_stage, final_snapshot = (memory_samples[-1] if memory_samples else samples[-1])
    peak_stage, peak_snapshot = max(samples, key=lambda item: int(item[1].get("used_bytes", 0) or 0))
    min_free_stage, min_free_snapshot = min(samples, key=lambda item: int(item[1].get("free_bytes", 0) or 0))
    total_bytes = int(final_snapshot.get("total_bytes", 0) or peak_snapshot.get("total_bytes", 0) or 0)
    return {
        "sample_count": len(samples),
        "final_stage": final_stage,
        "final_free_bytes": int(final_snapshot.get("free_bytes", 0) or 0),
        "final_used_bytes": int(final_snapshot.get("used_bytes", 0) or 0),
        "peak_stage": peak_stage,
        "peak_used_bytes": int(peak_snapshot.get("used_bytes", 0) or 0),
        "min_free_stage": min_free_stage,
        "min_free_bytes": int(min_free_snapshot.get("free_bytes", 0) or 0),
        "total_bytes": total_bytes,
    }


def _log_effective_mtp_config(config: ServerConfig, *, engine: Any | None) -> None:
    """Log configured or effective MTP serving state.

    Lazy startup has no engine to inspect yet, so it reports a pending state;
    ``get_llm`` emits the effective state immediately after creating the model.
    """

    if engine is None:
        _LOGGER.info(
            "EFFECTIVE_MTP: serving=pending engine_supported=unknown "
            "default_enabled=unknown policy=%s thinking=%s",
            config.speculative_mtp_serving,
            config.speculative_mtp_thinking,
        )
        return
    capability = _speculative_mtp_capability(config, engine=engine)
    _LOGGER.info(
        "EFFECTIVE_MTP: serving=%s engine_supported=%s default_enabled=%s policy=%s thinking=%s",
        "enabled" if capability["serving_route"] else "off",
        _engine_supports_speculative_mtp(engine),
        capability.get("default_enabled", False),
        capability.get("policy", str(config.speculative_mtp_serving)),
        capability.get("thinking_policy", str(config.speculative_mtp_thinking)),
    )


def _log_startup_memory_summary(memory: Mapping[str, Any], checks: Mapping[str, Any]) -> None:
    summary = _startup_memory_summary(memory, checks)
    if summary is None:
        return
    _LOGGER.info(
        "STARTUP_MEMORY: final_stage=%s final_free=%s final_used=%s peak_stage=%s peak_used=%s min_free_stage=%s min_free=%s total=%s samples=%d",
        summary["final_stage"],
        _format_bytes(int(summary["final_free_bytes"])),
        _format_bytes(int(summary["final_used_bytes"])),
        summary["peak_stage"],
        _format_bytes(int(summary["peak_used_bytes"])),
        summary["min_free_stage"],
        _format_bytes(int(summary["min_free_bytes"])),
        _format_bytes(int(summary["total_bytes"])),
        int(summary["sample_count"]),
    )


def _startup_free_memory_guard(
    *,
    memory: Mapping[str, Any] | None,
    min_free_mib: int | None,
) -> None:
    if min_free_mib is None:
        return
    if memory is None:
        _LOGGER.warning("STARTUP_MEMORY: minimum free-memory guard skipped; HIP memory snapshot unavailable")
        return
    required = max(0, int(min_free_mib)) * 1024**2
    free = int(memory.get("free_bytes", 0) or 0)
    if free < required:
        raise MemoryError(
            f"startup free-memory guard failed: free={_format_bytes(free)} below "
            f"required={_format_bytes(required)}"
        )


def _request_completion_cap(
    request: CompletionRequest | ChatCompletionRequest,
) -> int | None:
    """Return the total completion cap (reasoning + answer) for a request.

    Prefers the legacy ``max_tokens`` and falls back to OpenAI's newer
    ``max_completion_tokens`` when the legacy field is unset.
    """

    if getattr(request, "max_tokens", None) is not None:
        return max(0, int(request.max_tokens))
    max_completion_tokens = getattr(request, "max_completion_tokens", None)
    if max_completion_tokens is not None:
        return max(0, int(max_completion_tokens))
    return None


def _request_max_tokens(
    request: CompletionRequest | ChatCompletionRequest,
    prompts: Sequence[PromptInput],
    engine: Any,
    max_context_tokens: int | None,
    *,
    chat_default_max_tokens: int | None = 4096,
) -> int:
    cap = _request_completion_cap(request)
    if cap is not None:
        return cap
    if isinstance(request, ChatCompletionRequest):
        remaining = _remaining_context_tokens(prompts, engine, max_context_tokens)
        if chat_default_max_tokens is None:
            return 8192 if remaining is None else max(0, int(remaining))
        default_tokens = max(0, int(chat_default_max_tokens))
        if remaining is None:
            return default_tokens
        return max(0, min(default_tokens, int(remaining)))
    return 16


def _remaining_context_tokens(
    prompts: Sequence[PromptInput],
    engine: Any,
    max_context_tokens: int | None,
) -> int | None:
    if max_context_tokens is None:
        return None
    return min(
        int(max_context_tokens) - _prompt_token_count(engine, prompt) - 1
        for prompt in prompts
    )


def _chat_default_max_tokens_label(config: ServerConfig) -> str:
    return "auto" if config.chat_default_max_tokens is None else str(int(config.chat_default_max_tokens))


def _request_logprobs_enabled(request: CompletionRequest | ChatCompletionRequest) -> bool:
    if isinstance(request, CompletionRequest):
        return request.logprobs is not None
    return bool(request.logprobs)


def _engine_supports_stream_logprobs(engine: Any | None) -> bool:
    if engine is None:
        return False
    target = getattr(engine, "_text_generator", None) or engine
    return bool(
        getattr(target, "supports_stream_logprobs", False)
        or getattr(target, "supports_stream_token_logprobs", False)
    )


def _engine_stream_many_callable(engine: Any | None) -> Callable[[tuple[str, ...], SamplingParams], Iterator[Any]] | None:
    if engine is None:
        return None
    streamer = getattr(engine, "stream_many_detailed", None)
    if callable(streamer):
        return streamer
    return None


def _engine_supports_controlled_streaming(engine: Any | None) -> bool:
    if engine is None:
        return False
    for target in (engine, getattr(engine, "_text_generator", None)):
        if target is not None and bool(getattr(target, "supports_controlled_streaming", False)):
            return True
    return False


def _engine_supports_independent_generation(engine: Any | None) -> bool:
    """Whether one model service independently owns child progress/output."""

    if engine is None:
        return False
    for target in (engine, getattr(engine, "_text_generator", None)):
        if target is not None and bool(
            getattr(target, "supports_independent_generation", False)
        ):
            return True
    return False


def _engine_supports_stream_many(engine: Any | None) -> bool:
    if engine is None:
        return False
    if _engine_stream_many_callable(engine) is None:
        return False
    for target in (engine, getattr(engine, "_text_generator", None)):
        if target is None:
            continue
        if bool(getattr(target, "supports_stream_many", False)):
            return True
        if bool(getattr(target, "supports_stream_many_detailed", False)):
            return True
    return False


def _engine_speculative_callable(
    engine: Any | None,
) -> Callable[[tuple[PromptInput, ...], SamplingParams], Any] | None:
    if engine is None:
        return None
    supports = getattr(engine, "supports_speculative", None)
    if supports is not None and not bool(supports):
        return None
    runner = getattr(engine, "generate_speculative_detailed", None)
    return runner if callable(runner) else None


def _engine_speculative_stream_callable(
    engine: Any | None,
) -> Callable[[PromptInput, SamplingParams], Any] | None:
    if engine is None:
        return None
    supports = getattr(engine, "supports_speculative", None)
    if supports is not None and not bool(supports):
        return None
    streamer = getattr(engine, "stream_speculative_detailed", None)
    return streamer if callable(streamer) else None


def _engine_supports_speculative(engine: Any | None) -> bool:
    return _engine_speculative_callable(engine) is not None


def _engine_supports_speculative_streaming(engine: Any | None) -> bool:
    return _engine_speculative_stream_callable(engine) is not None


def _engine_speculative_capabilities(engine: Any | None) -> dict[str, Any]:
    if engine is None:
        return {}
    raw = getattr(engine, "speculative_capabilities", {})
    capabilities = raw() if callable(raw) else raw
    if capabilities is None:
        return {}
    if not isinstance(capabilities, Mapping):
        raise TypeError("engine speculative_capabilities must be a mapping")
    return deepcopy(dict(capabilities))


def _engine_speculative_mtp_callable(engine: Any | None) -> Callable[[tuple[str, ...], SamplingParams], Any] | None:
    if engine is None:
        return None
    supports = getattr(engine, "supports_speculative_mtp", None)
    if supports is not None and not bool(supports):
        return None
    runner = getattr(engine, "generate_speculative_mtp_detailed", None)
    if callable(runner):
        return runner
    return None


def _engine_supports_speculative_mtp(engine: Any | None) -> bool:
    return _engine_speculative_mtp_callable(engine) is not None


def _engine_speculative_mtp_serving_capability(
    engine: Any | None,
) -> dict[str, Any] | None:
    if engine is None:
        return None
    raw = getattr(engine, "speculative_mtp_serving_capability", None)
    value = raw() if callable(raw) else raw
    if value is None:
        return None
    payload = value.as_dict() if callable(getattr(value, "as_dict", None)) else value
    if not isinstance(payload, Mapping):
        raise TypeError("engine speculative_mtp_serving_capability must be a mapping")
    return deepcopy(dict(payload))


def _engine_supports_default_mtp(engine: Any | None) -> bool:
    """Whether immutable model-plugin evidence admits automatic MTP."""

    plan = _engine_speculative_mtp_serving_capability(engine)
    return bool(
        plan is not None
        and plan.get("admitted")
        and plan.get("automatic_eligible")
    )


def _chat_live_many_streaming_allowed(request: ChatCompletionRequest) -> bool:
    return (
        _request_n(request) > 1
        and request.continuation_id is None
        and not request.tools
        and not _request_logprobs_enabled(request)
        and not _structured_result_validation(request)
        and not _request_has_stop_strings(request)
    )


def _request_has_stop_strings(request: CompletionRequest | ChatCompletionRequest) -> bool:
    stop = request.stop
    if stop is None:
        return False
    if isinstance(stop, str):
        return bool(stop)
    return any(bool(str(item)) for item in stop)


def _request_top_logprobs(request: CompletionRequest | ChatCompletionRequest) -> int:
    if isinstance(request, CompletionRequest):
        return 0 if request.logprobs is None else int(request.logprobs)
    return 0 if request.top_logprobs is None else int(request.top_logprobs)


async def _generate_detailed(
    engine: Any,
    prompts: tuple[PromptInput, ...],
    sampling: SamplingParams,
) -> list[Any]:
    detailed = getattr(engine, "generate_detailed", None)
    if callable(detailed):
        return list(await run_in_threadpool(detailed, prompts, sampling))
    return [_coerce_generation_output(item) for item in await run_in_threadpool(engine.generate, prompts, sampling)]


async def _generate_speculative_detailed(
    engine: Any,
    prompts: tuple[PromptInput, ...],
    sampling: SamplingParams,
) -> list[Any]:
    runner = _engine_speculative_callable(engine)
    if runner is None:
        raise NotImplementedError(
            "speculative provider serving is not supported by this engine"
        )
    return list(await run_in_threadpool(runner, prompts, sampling))


async def _generate_speculative_mtp_detailed(
    engine: Any,
    prompts: tuple[PromptInput, ...],
    sampling: SamplingParams,
) -> list[Any]:
    runner = _engine_speculative_mtp_callable(engine)
    if runner is None:
        raise NotImplementedError("speculative MTP serving is not supported by this engine")
    return list(await run_in_threadpool(runner, prompts, sampling))


def _coerce_generation_output(value: Any) -> GenerationOutput:
    if isinstance(value, GenerationOutput):
        return value
    token_logprobs = getattr(value, "token_logprobs", None)
    finish_details = getattr(value, "finish_details", None)
    telemetry = getattr(value, "telemetry", None)
    generated_token_ids = getattr(value, "generated_token_ids", None)
    if (
        token_logprobs is not None
        or finish_details is not None
        or telemetry is not None
        or generated_token_ids is not None
    ):
        return GenerationOutput(
            text=str(getattr(value, "text", value)),
            token_logprobs=_coerce_token_logprobs(token_logprobs),
            finish_details=finish_details,
            telemetry=telemetry,
            generated_token_ids=generated_token_ids,
        )
    return GenerationOutput(text=str(value))


def _coerce_token_logprobs(value: Any) -> tuple[TokenLogprob, ...]:
    if not value:
        return ()
    tokens: list[TokenLogprob] = []
    for item in value:
        if isinstance(item, TokenLogprob):
            tokens.append(item)
            continue
        if isinstance(item, Mapping):
            raw_top = item.get("top_logprobs", ()) or ()
            top_logprobs = tuple(
                (
                    int(top.get("token_id")),
                    str(top.get("token_text", top.get("token", ""))),
                    float(top.get("logprob")),
                )
                if isinstance(top, Mapping)
                else (int(top[0]), str(top[1]), float(top[2]))
                for top in raw_top
            )
            tokens.append(
                TokenLogprob(
                    token_id=int(item.get("token_id")),
                    token_text=str(item.get("token_text", item.get("token", ""))),
                    logprob=(None if item.get("logprob") is None else float(item.get("logprob"))),
                    top_logprobs=top_logprobs,
                )
            )
            continue
        tokens.append(
            TokenLogprob(
                token_id=int(getattr(item, "token_id")),
                token_text=str(getattr(item, "token_text", getattr(item, "token", ""))),
                logprob=getattr(item, "logprob", None),
                top_logprobs=tuple(getattr(item, "top_logprobs", ()) or ()),
            )
        )
    return tuple(tokens)


def _coerce_generation_stream_chunk(value: Any) -> GenerationStreamChunk:
    if isinstance(value, GenerationStreamChunk):
        return value
    token_logprobs = getattr(value, "token_logprobs", None)
    finish_details = getattr(value, "finish_details", None)
    telemetry = getattr(value, "telemetry", None)
    generated_token_ids = getattr(value, "generated_token_ids", None)
    if (
        token_logprobs is not None
        or finish_details is not None
        or telemetry is not None
        or generated_token_ids is not None
    ):
        return GenerationStreamChunk(
            text=str(getattr(value, "text", value)),
            token_logprobs=_coerce_token_logprobs(token_logprobs),
            finish_details=finish_details,
            telemetry=telemetry,
            generated_token_ids=generated_token_ids,
        )
    if isinstance(value, Mapping) and any(
        key in value
        for key in (
            "text",
            "token_logprobs",
            "finish_details",
            "telemetry",
            "generated_token_ids",
        )
    ):
        return GenerationStreamChunk(
            text=str(value.get("text", "")),
            token_logprobs=_coerce_token_logprobs(value.get("token_logprobs", ())),
            finish_details=value.get("finish_details"),
            telemetry=value.get("telemetry"),
            generated_token_ids=value.get("generated_token_ids"),
        )
    return GenerationStreamChunk(text=str(value))


def _stream_chunk_from_detail(text: str, detail: GenerationOutput | None) -> GenerationStreamChunk | None:
    if detail is None or (
        detail.finish_details is None
        and detail.telemetry is None
        and detail.generated_token_ids is None
    ):
        return None
    return GenerationStreamChunk(
        text=str(text),
        finish_details=detail.finish_details,
        telemetry=detail.telemetry,
        generated_token_ids=detail.generated_token_ids,
    )


def _output_from_stream_chunk(chunk: GenerationStreamChunk | None, text: str) -> GenerationOutput | None:
    if chunk is None or (
        chunk.finish_details is None
        and chunk.telemetry is None
        and chunk.generated_token_ids is None
    ):
        return None
    return GenerationOutput(
        text=str(text),
        finish_details=chunk.finish_details,
        telemetry=chunk.telemetry,
        generated_token_ids=chunk.generated_token_ids,
    )


def _backend_generation_targets(engine: Any) -> tuple[Any, ...]:
    targets: list[Any] = [engine]
    generator = getattr(engine, "_text_generator", None)
    if generator is not None:
        targets.append(generator)
        inner = getattr(generator, "inner", None)
        if inner is not None:
            targets.append(inner)
    return tuple(targets)


def _backend_last_batch_generation(engine: Any) -> Mapping[str, Any] | None:
    for target in _backend_generation_targets(engine):
        batch_generation = getattr(target, "last_batch_generation", None)
        if isinstance(batch_generation, Mapping):
            return batch_generation
    return None


def _backend_scheduler_token_chunks(engine: Any) -> list[dict[str, Any]] | None:
    for target in _backend_generation_targets(engine):
        batch_generation = getattr(target, "last_batch_generation", None)
        if not isinstance(batch_generation, Mapping):
            continue
        raw_chunks = batch_generation.get("scheduler_token_chunks")
        if isinstance(raw_chunks, Sequence) and not isinstance(raw_chunks, (str, bytes, bytearray)):
            chunks = [
                deepcopy(dict(raw_chunk))
                for raw_chunk in raw_chunks
                if isinstance(raw_chunk, Mapping)
            ]
            if chunks:
                return chunks
    return None


def _shape_int(value: Any, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= minimum else None


def _backend_actual_group_rows(
    batch_generation: Mapping[str, Any] | None,
    *,
    input_rows: int,
) -> list[int]:
    if batch_generation is not None:
        candidates = batch_generation.get("actual_backend_group_rows")
        if candidates is None:
            candidates = batch_generation.get("group_widths")
        if isinstance(candidates, Sequence) and not isinstance(
            candidates,
            (str, bytes, bytearray),
        ):
            widths = [
                width
                for raw in candidates
                if (width := _shape_int(raw, minimum=1)) is not None
            ]
            if widths and sum(widths) == int(input_rows):
                return widths
        groups = batch_generation.get("groups")
        if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes, bytearray)):
            widths = [
                width
                for group in groups
                if isinstance(group, Mapping)
                and (width := _shape_int(group.get("width"), minimum=1)) is not None
            ]
            if widths and sum(widths) == int(input_rows):
                return widths
    return [int(input_rows)]


def _backend_verifier_rows(batch_generation: Mapping[str, Any] | None) -> int:
    if batch_generation is None:
        return 0
    for key in ("verifier_rows", "target_verify_rows"):
        value = _shape_int(batch_generation.get(key))
        if value is not None:
            return value
    speculative = batch_generation.get("speculative_mtp")
    if isinstance(speculative, Mapping):
        for key in ("verifier_rows", "target_verify_rows", "linear_state_captured_rows"):
            value = _shape_int(speculative.get(key))
            if value is not None:
                return value
    batch_timing = batch_generation.get("batch_timing")
    if isinstance(batch_timing, Mapping):
        value = _shape_int(batch_timing.get("mtp_target_verify_rows"))
        if value is not None:
            return value
    return 0


def _backend_generation_group_shape(engine: Any, *, input_rows: int) -> dict[str, Any]:
    rows = max(1, int(input_rows))
    batch_generation = _backend_last_batch_generation(engine)
    batch_id = None if batch_generation is None else batch_generation.get("batch_id")
    group_id = str(batch_id).strip() if batch_id is not None else ""
    if not group_id:
        group_id = f"backend-{uuid.uuid4().hex}"
    actual_group_rows = _backend_actual_group_rows(batch_generation, input_rows=rows)
    verifier_rows = _backend_verifier_rows(batch_generation)
    return {
        "id": group_id,
        "call_index": 0,
        "prompt_offset": 0,
        "input_rows": rows,
        "actual_group_rows": actual_group_rows,
        "max_actual_group_rows": max(actual_group_rows),
        "verifier_rows": verifier_rows,
    }


def _queue_generation_shape(
    *,
    route: str,
    route_cap: int | None,
    route_cap_applied: bool = True,
    queue_group_id: str,
    queue_request_count: int,
    queue_prompt_rows: int,
    item_index: int,
    item_prompt_offset: int,
    item_prompt_rows: int,
    backend_groups: Sequence[Mapping[str, Any]],
    route_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    copied_groups = [deepcopy(dict(group)) for group in backend_groups]
    payload = {
        "schema_version": 1,
        "route": str(route),
        "route_cap": {
            "scope": "queue_requests",
            "value": None if route_cap is None else int(route_cap),
            "applied": bool(route_cap_applied),
        },
        "queue_group": {
            "id": str(queue_group_id),
            "request_count": int(queue_request_count),
            "prompt_rows": int(queue_prompt_rows),
            "item_index": int(item_index),
            "item_prompt_offset": int(item_prompt_offset),
            "item_prompt_rows": int(item_prompt_rows),
        },
        "backend_groups": copied_groups,
        "verifier_rows": sum(int(group.get("verifier_rows", 0)) for group in copied_groups),
    }
    if route_decision is not None:
        payload["route_decision"] = deepcopy(dict(route_decision))
    return payload


def _copy_scheduler_token_chunks(chunks: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not chunks:
        return None
    copied = [deepcopy(dict(chunk)) for chunk in chunks if isinstance(chunk, Mapping)]
    return copied or None


def _buffered_delta_stream_chunk(
    text: str,
    final_chunk: GenerationStreamChunk | None,
    *,
    phase: str,
    tokens: Mapping[str, int] | None,
) -> GenerationStreamChunk | None:
    telemetry = None if final_chunk is None else final_chunk.telemetry
    if telemetry is None or tokens is None:
        return None
    backend_state = telemetry.decode_state
    token_state = DecodeState.from_stream_tokens(
        phase=phase,
        tokens=tokens,
        row_index=backend_state.row_index,
    )
    return GenerationStreamChunk(
        text=str(text),
        telemetry=GenerationTelemetry(
            decode_state=DecodeState(
                request_id=backend_state.request_id,
                row_index=backend_state.row_index,
                step_index=token_state.step_index,
                prompt_tokens=token_state.prompt_tokens,
                generated_tokens=token_state.generated_tokens,
                phase=token_state.phase,
                reasoning_tokens=token_state.reasoning_tokens,
                answer_tokens=token_state.answer_tokens,
                tool_call_tokens=token_state.tool_call_tokens,
                structured_tokens=token_state.structured_tokens,
                active_processors=backend_state.active_processors,
                sampler_fast_path_blockers=backend_state.sampler_fast_path_blockers,
                sampler_fallback_reason=backend_state.sampler_fallback_reason,
                sampler_mode=backend_state.sampler_mode,
                full_vocab_logits_d2h=backend_state.full_vocab_logits_d2h,
                logits_d2h_bytes=backend_state.logits_d2h_bytes,
                execution_path=backend_state.execution_path,
                native_compact_prefill=backend_state.native_compact_prefill,
                native_caware_decode=backend_state.native_caware_decode,
                serial_decode_fallback=backend_state.serial_decode_fallback,
                native_sampler_rows=backend_state.native_sampler_rows,
                continuation_eligible=token_state.continuation_eligible,
            )
        ),
        generated_token_ids=(
            None if final_chunk is None else final_chunk.generated_token_ids
        ),
    )


def _validate_logprob_details(details: Sequence[GenerationOutput], outputs: Sequence[str]) -> None:
    for output, text in zip(details, outputs, strict=True):
        if text and not output.token_logprobs:
            raise OpenAIHTTPError(
                501,
                "logprobs are not supported by this backend response path",
                error_type="server_error",
                code="unsupported_feature",
                param="logprobs",
            )


def _request_deadline_at(config: ServerConfig, request: CompletionRequest | ChatCompletionRequest) -> float | None:
    timeout_ms = request.timeout_ms if request.timeout_ms is not None else config.request_timeout_ms
    if timeout_ms is None or float(timeout_ms) <= 0.0:
        return None
    return time.perf_counter() + float(timeout_ms) / 1000.0


def _request_control(
    config: ServerConfig,
    request: CompletionRequest | ChatCompletionRequest,
    raw_request: Request | None = None,
) -> _RequestControl:
    disconnected: Callable[[], Awaitable[bool]] | None = None
    if raw_request is not None:

        async def is_disconnected() -> bool:
            try:
                return bool(await raw_request.is_disconnected())
            except Exception:
                return False

        disconnected = is_disconnected
    return _RequestControl(deadline_at=_request_deadline_at(config, request), disconnected=disconnected)


def _deadline_finish_details() -> dict[str, Any]:
    return FinishDetails(reason="deadline_exceeded", deadline_exceeded=True).to_json_dict()


def _cancelled_finish_details() -> dict[str, Any]:
    return FinishDetails(reason="cancelled", cancelled=True).to_json_dict()


def _deadline_exceeded_error(finish_details: FinishDetails | Mapping[str, Any] | None = None) -> OpenAIHTTPError:
    if isinstance(finish_details, FinishDetails):
        details = finish_details.to_json_dict()
    elif finish_details is not None:
        details = dict(finish_details)
    else:
        details = _deadline_finish_details()
    return OpenAIHTTPError(
        408,
        "request deadline exceeded",
        error_type="timeout_error",
        code="deadline_exceeded",
        param="timeout_ms",
        finish_details=details,
    )


def _request_cancelled_error(finish_details: FinishDetails | Mapping[str, Any] | None = None) -> OpenAIHTTPError:
    if isinstance(finish_details, FinishDetails):
        details = finish_details.to_json_dict()
    elif finish_details is not None:
        details = dict(finish_details)
    else:
        details = _cancelled_finish_details()
    return OpenAIHTTPError(
        499,
        "request cancelled",
        error_type="cancelled_error",
        code="cancelled",
        finish_details=details,
    )


async def _raise_for_request_control(control: _RequestControl) -> None:
    if control.deadline_at is not None and float(control.deadline_at) - time.perf_counter() <= 0.0:
        control.cancellation_token.cancel(FinishDetails(reason="deadline_exceeded", deadline_exceeded=True))
        raise _deadline_exceeded_error()
    if control.disconnected is not None and await control.disconnected():
        control.cancellation_token.cancel(FinishDetails(reason="cancelled", cancelled=True))
        raise _request_cancelled_error()


def _request_control_poll_interval(control: _RequestControl) -> float:
    interval = max(0.001, float(control.poll_interval_s))
    if control.deadline_at is None:
        return interval
    remaining = float(control.deadline_at) - time.perf_counter()
    if remaining <= 0.0:
        return 0.0
    return min(interval, remaining)


async def _await_controlled_task_exit(task: asyncio.Task[Any], *, timeout_s: float = 5.0) -> None:
    """Wait for one controlled child to acknowledge cancellation, then force it."""

    # Starlette cancels response bodies through an AnyIO cancel scope. AnyIO's
    # level cancellation would otherwise interrupt every cleanup await again,
    # leaving ``__anext__`` live when the iterator wrapper calls ``aclose()``.
    with CancelScope(shield=True):
        done, _pending = await asyncio.wait({task}, timeout=max(0.0, float(timeout_s)))
        if task not in done:
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _await_with_request_control(awaitable, control: _RequestControl | None = None):
    if control is None:
        return await awaitable
    if control.deadline_at is None and control.disconnected is None:
        # Active StreamingResponse bodies deliberately leave request polling to
        # Starlette. Shield the in-flight child so Starlette's caller
        # cancellation can become a cooperative row-token signal first.
        task = asyncio.ensure_future(awaitable)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            control.cancellation_token.cancel(FinishDetails(reason="cancelled", cancelled=True))
            await _await_controlled_task_exit(task)
            raise
    try:
        await _raise_for_request_control(control)
    except OpenAIHTTPError:
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise
    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            done, _pending = await asyncio.wait({task}, timeout=_request_control_poll_interval(control))
            if task in done:
                return await task
            await _raise_for_request_control(control)
    except OpenAIHTTPError:
        # The control token is already tripped. Give controlled generation a
        # bounded opportunity to acknowledge it and reclaim its own row before
        # forcing task cancellation; abrupt ``__anext__`` cancellation closes
        # the generator and can interfere with concurrently subscribed rows.
        await _await_controlled_task_exit(task)
        raise
    except asyncio.CancelledError:
        # ASGI servers cancel the response task when a streaming client closes
        # its socket. Signal only this row, then let the child acknowledge and
        # unwind before the iterator wrapper calls ``aclose()``. This avoids an
        # abrupt ``__anext__`` cancellation propagating into peer subscriptions.
        control.cancellation_token.cancel(FinishDetails(reason="cancelled", cancelled=True))
        await _await_controlled_task_exit(task)
        raise


async def _iterate_with_request_control(
    iterator: AsyncIterator[str],
    control: _RequestControl | None = None,
) -> AsyncIterator[str]:
    async_iterator = iterator.__aiter__()
    try:
        while True:
            try:
                item = await _await_with_request_control(async_iterator.__anext__(), control)
            except StopAsyncIteration:
                break
            yield item
    finally:
        closer = getattr(async_iterator, "aclose", None)
        if callable(closer):
            await closer()


def _deadline_detail_from_outputs(details: Sequence[GenerationOutput]) -> FinishDetails | None:
    for item in details:
        finish = item.finish_details
        if finish is not None and finish.deadline_exceeded:
            return finish
    return None


def _validate_generation_request(
    config: ServerConfig,
    request: CompletionRequest | ChatCompletionRequest,
    *,
    engine: Any | None = None,
    route_unsupported_grammar: bool = False,
) -> None:
    _request_n(request)
    if _completion_prompt_uses_token_ids(request):
        for param in ("echo", "continuation_id", "session"):
            value = getattr(request, param, None)
            if value not in (None, False):
                raise OpenAIHTTPError(
                    400,
                    f"{param} is not supported with exact token-ID prompts",
                    code="unsupported_parameter",
                    param=param,
                )
    extra_keys = _request_extra_keys(request)
    if extra_keys:
        param = sorted(extra_keys)[0]
        extra = None
        if route_unsupported_grammar and param in _UNSUPPORTED_GRAMMAR_FIELDS:
            extra = {
                "hipengine": {
                    "routing": _routing_rejection_metadata(
                        config,
                        requested_model=request.model,
                        reason="unsupported_grammar",
                        engine=engine,
                        details={
                            "unsupported_field": param,
                            "unsupported_capability": "grammar",
                        },
                    )
                }
            }
        raise OpenAIHTTPError(
            400,
            f"unsupported request parameter {param!r}",
            code="unsupported_parameter",
            param=param,
            extra=extra,
        )
    _validate_session_request(request)
    unsupported_param = _unsupported_agentic_request_param(request, engine=engine)
    if unsupported_param is not None:
        raise OpenAIHTTPError(
            400,
            f"{unsupported_param} is not supported by this server",
            code="unsupported_parameter",
            param=unsupported_param,
        )
    _validate_response_format_request(request)
    _validate_guided_json_request(request)
    _validate_guided_regex_request(request)
    _validate_guided_choice_request(request)
    _validate_guided_patch_request(request)
    _validate_tool_schema_requests(request)
    if isinstance(request, ChatCompletionRequest):
        _invalid_tool_call_error_mode(request)
    if isinstance(request, ChatCompletionRequest) and request.top_logprobs is not None and not request.logprobs:
        raise OpenAIHTTPError(
            400,
            "top_logprobs requires logprobs=true",
            code="invalid_request",
            param="top_logprobs",
        )
    try:
        from hipengine.kvcache import resolve_kv_policy

        server_policy = resolve_kv_policy(
            config.kv_storage,
            scale_dtype=config.kv_scale_dtype,
            scale_granularity=config.kv_scale_granularity,
        )
        request_policy = resolve_kv_policy(
            request.kv_storage or config.kv_storage,
            scale_dtype=request.kv_scale_dtype or config.kv_scale_dtype,
            scale_granularity=request.kv_scale_granularity or config.kv_scale_granularity,
        )
    except ValueError as exc:
        raise OpenAIHTTPError(400, str(exc), code="invalid_kv_policy", param="kv_storage") from exc
    if (
        request.kv_storage is not None
        or request.kv_scale_dtype is not None
        or request.kv_scale_granularity is not None
    ) and (
        request_policy.storage_dtype != server_policy.storage_dtype
        or request_policy.scale_dtype != server_policy.scale_dtype
        or request_policy.scale_granularity != server_policy.scale_granularity
    ):
        raise OpenAIHTTPError(
            400,
            "this server preallocates a fixed KV cache policy; restart with matching "
            "--kv-storage/--kv-scale-dtype to use different KV settings",
            code="unsupported_kv_policy",
            param="kv_storage",
        )


def _validate_context_budget(
    max_context_tokens: int | None,
    engine: Any,
    prompts: Sequence[PromptInput],
    sampling: SamplingParams,
    *,
    fit_context_extra: Mapping[str, Any] | None = None,
    error_extra: Mapping[str, Any] | None = None,
) -> None:
    if max_context_tokens is None:
        return
    max_context = max(1, int(max_context_tokens))
    max_tokens = max(0, int(sampling.max_tokens))
    for index, prompt in enumerate(prompts):
        prompt_tokens = _prompt_token_count(engine, prompt)
        required = prompt_tokens + max_tokens + 1
        if required > max_context:
            fit_context = _context_fit_payload(
                prompt_tokens=prompt_tokens,
                max_context_tokens=max_context,
                max_tokens=max_tokens,
            )
            if fit_context_extra is not None:
                fit_context.update(dict(fit_context_extra))
            extra = {"fit_context": fit_context}
            if error_extra is not None:
                extra.update(dict(error_extra))
            raise OpenAIHTTPError(
                400,
                f"request requires {required} context tokens (prompt {prompt_tokens} + "
                f"max_tokens {max_tokens} + 1), exceeding this server's "
                f"preallocated max_context_tokens={max_context}",
                code="context_length_exceeded",
                param=f"prompts[{index}].max_tokens" if len(prompts) > 1 else "max_tokens",
                extra=extra,
            )


def _context_fit_payload(
    *,
    prompt_tokens: int,
    max_context_tokens: int | None,
    max_tokens: int,
) -> dict[str, Any]:
    prompt_count = max(0, int(prompt_tokens))
    effective_max_tokens = max(0, int(max_tokens))
    required_context = prompt_count + effective_max_tokens + 1
    max_context = None if max_context_tokens is None else max(1, int(max_context_tokens))
    max_allowed = None if max_context is None else max(0, max_context - prompt_count - 1)
    overflow_tokens = 0 if max_context is None else max(0, required_context - max_context)
    recommended = effective_max_tokens if max_allowed is None else min(effective_max_tokens, max_allowed)
    return {
        "prompt_tokens": prompt_count,
        "max_context_tokens": max_context,
        "effective_max_tokens": effective_max_tokens,
        "max_allowed_max_tokens": max_allowed,
        "recommended_max_tokens": recommended,
        "required_context_tokens": required_context,
        "overflow_tokens": overflow_tokens,
        "fits": True if max_context is None else required_context <= max_context,
        "clear_policy": "reject",
        "would_truncate": False,
        "would_drop": [],
    }


def _count_tokens_for_admission(engine: Any, text: str) -> int:
    counter = getattr(engine, "count_tokens", None)
    if not callable(counter):
        return 0
    try:
        return max(0, int(counter(text)))
    except NotImplementedError:
        return 0


def _prompt_token_count(engine: Any, prompt: PromptInput) -> int:
    if isinstance(prompt, str):
        return _count_tokens_for_admission(engine, prompt)
    return len(prompt)


def _prompt_tokenizer_identity(engine: Any) -> str | None:
    target = _model_chat_protocol_target(engine)
    tokenizer = getattr(target, "tokenizer", None)
    owner = tokenizer if tokenizer is not None else target
    if owner is None:
        return None
    cls = type(owner)
    return f"{cls.__module__}.{cls.__qualname__}"


def _prepare_prompt_input(
    engine: Any,
    prompt: PromptInput,
    *,
    render_ms: float = 0.0,
) -> PromptInput:
    if isinstance(prompt, PreparedPromptInput):
        if render_ms <= 0.0 or prompt.render_ms == float(render_ms):
            return prompt
        return replace(prompt, render_ms=max(0.0, float(render_ms)))
    if not isinstance(prompt, str):
        return normalize_prompt_input(prompt)
    tokenizer = getattr(engine, "tokenize", None)
    if not callable(tokenizer):
        return prompt
    started = time.perf_counter()
    try:
        token_ids = tuple(int(token) for token in tokenizer(prompt))
    except (KeyError, NotImplementedError, TypeError, ValueError):
        return prompt
    tokenize_ms = (time.perf_counter() - started) * 1_000.0
    return PreparedPromptInput(
        source_text=prompt,
        token_ids=token_ids,
        tokenize_ms=tokenize_ms,
        render_ms=max(0.0, float(render_ms)),
        tokenizer_identity=_prompt_tokenizer_identity(engine),
    )


def _prepare_prompt_inputs(
    engine: Any,
    prompts: Sequence[PromptInput],
) -> tuple[PromptInput, ...]:
    prepared_by_text: dict[str, PromptInput] = {}
    prepared: list[PromptInput] = []
    for prompt in prompts:
        if isinstance(prompt, str):
            value = prepared_by_text.get(prompt)
            if value is None:
                value = _prepare_prompt_input(engine, prompt)
                prepared_by_text[prompt] = value
            prepared.append(value)
        else:
            prepared.append(_prepare_prompt_input(engine, prompt))
    return tuple(prepared)


def _with_admission_prepare_ms(
    prompts: Sequence[PromptInput],
    admission_prepare_ms: float,
) -> tuple[PromptInput, ...]:
    elapsed = max(0.0, float(admission_prepare_ms))
    replaced: dict[int, PreparedPromptInput] = {}
    result: list[PromptInput] = []
    for prompt in prompts:
        if not isinstance(prompt, PreparedPromptInput):
            result.append(prompt)
            continue
        value = replaced.get(id(prompt))
        if value is None:
            value = replace(prompt, admission_prepare_ms=elapsed)
            replaced[id(prompt)] = value
        result.append(value)
    return tuple(result)


def _count_tokens_strict(engine: Any, text: str) -> int:
    counter = getattr(engine, "count_tokens", None)
    if not callable(counter):
        raise OpenAIHTTPError(
            501,
            "token counting is not supported by this model",
            error_type="server_error",
            code="unsupported_feature",
            param="text",
        )
    try:
        return max(0, int(counter(str(text))))
    except NotImplementedError as exc:
        raise OpenAIHTTPError(
            501,
            "token counting is not supported by this model",
            error_type="server_error",
            code="unsupported_feature",
            param="text",
        ) from exc


def _tokenizer_capability_flags(engine: Any | None) -> dict[str, bool]:
    if engine is None:
        return {"tokenize": False, "detokenize": False, "count_tokens": False}
    target = getattr(engine, "_text_generator", None) or engine
    tokenizer = getattr(target, "tokenizer", None)
    return {
        "tokenize": callable(getattr(target, "tokenize", None)),
        "detokenize": callable(getattr(target, "detokenize", None)) or callable(getattr(tokenizer, "decode", None)),
        "count_tokens": callable(getattr(target, "count_tokens", None)),
    }


def _tokenizer_compatibility_metadata(engine: Any | None) -> dict[str, Any]:
    flags = _tokenizer_capability_flags(engine)
    target = getattr(engine, "_text_generator", None) or engine
    return {
        "name": None if target is None else type(target).__name__,
        "tokenize": flags["tokenize"],
        "detokenize": flags["detokenize"],
        "count_tokens": flags["count_tokens"],
    }


def _validate_chat_session_snapshot_tokenizer_metadata(
    value: Any,
    *,
    current_engine: Any | None,
) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise OpenAIHTTPError(
            400,
            "session snapshot tokenizer must be an object",
            code="invalid_request",
            param="tokenizer",
        )
    if current_engine is None:
        return
    expected = _tokenizer_compatibility_metadata(current_engine)
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise OpenAIHTTPError(
                400,
                f"session snapshot tokenizer.{key} is incompatible with this server",
                code="invalid_request",
                param=f"tokenizer.{key}",
            )


def _tokenize_text(engine: Any, text: str) -> tuple[int, ...]:
    tokenizer = getattr(engine, "tokenize", None)
    if not callable(tokenizer):
        raise OpenAIHTTPError(
            501,
            "tokenization is not supported by this model",
            error_type="server_error",
            code="unsupported_feature",
            param="text",
        )
    try:
        return tuple(int(token) for token in tokenizer(str(text)))
    except NotImplementedError as exc:
        raise OpenAIHTTPError(
            501,
            "tokenization is not supported by this model",
            error_type="server_error",
            code="unsupported_feature",
            param="text",
        ) from exc


def _detokenize_ids(engine: Any, token_ids: Sequence[int], skip_special: bool = False) -> str:
    detokenizer = getattr(engine, "detokenize", None)
    if callable(detokenizer):
        try:
            return str(detokenizer(tuple(int(token) for token in token_ids), skip_special=bool(skip_special)))
        except TypeError:
            return str(detokenizer(tuple(int(token) for token in token_ids)))
        except NotImplementedError as exc:
            raise OpenAIHTTPError(
                501,
                "detokenization is not supported by this model",
                error_type="server_error",
                code="unsupported_feature",
                param="token_ids",
            ) from exc
    tokenizer = getattr(engine, "tokenizer", None)
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        try:
            return str(decode(tuple(int(token) for token in token_ids), skip_special=bool(skip_special)))
        except TypeError:
            return str(decode(tuple(int(token) for token in token_ids)))
    raise OpenAIHTTPError(
        501,
        "detokenization is not supported by this model",
        error_type="server_error",
        code="unsupported_feature",
        param="token_ids",
    )


def _chat_request_from_diagnostic(config: ServerConfig, request: TokenDiagnosticRequest) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=config.model_id,
        messages=list(request.messages or ()),
        tools=request.tools,
        tool_choice=request.tool_choice,
        reasoning_effort=request.reasoning_effort,
        enable_thinking=request.enable_thinking,
        max_think_tokens=request.max_think_tokens,
        min_answer_tokens=request.min_answer_tokens,
        hard_think_cap=request.hard_think_cap,
        soft_close_window=request.soft_close_window,
        hard_close_message=request.hard_close_message,
        hard_close_sequence=request.hard_close_sequence,
        thinking_token_budget=request.thinking_token_budget,
        chat_template_kwargs=request.chat_template_kwargs,
        thinking=request.thinking,
        reasoning=request.reasoning,
        guided_json=request.guided_json,
        guided_regex=request.guided_regex,
        guided_choice=request.guided_choice,
        guided_patch=request.guided_patch,
        guided_diff=request.guided_diff,
        max_tokens=getattr(request, "max_tokens", None),
        session=request.session,
    )


def _diagnostic_text_from_request(
    config: ServerConfig,
    request: TokenDiagnosticRequest,
    *,
    engine: Any | None = None,
    max_context_tokens: int | None = None,
) -> tuple[str, str]:
    has_text = request.text is not None
    has_messages = request.messages is not None
    if has_text == has_messages:
        raise OpenAIHTTPError(
            400,
            "provide exactly one of text or messages",
            code="invalid_request",
            param="text",
        )
    if has_text:
        return str(request.text), "text"
    chat_request = _chat_request_from_diagnostic(config, request)
    prompt, _thinking = _render_chat_prompt_for_request(
        chat_request,
        chat_default_max_tokens=config.chat_default_max_tokens,
        engine=engine,
        max_context_tokens=max_context_tokens,
    )
    return prompt, "chat"


def _diagnostic_thinking_budget_payload(
    config: ServerConfig,
    request: TokenDiagnosticRequest,
    *,
    engine: Any,
    max_context_tokens: int | None,
    chat_request: ChatCompletionRequest | None = None,
) -> dict[str, Any] | None:
    if request.messages is None and chat_request is None:
        return None
    if chat_request is None:
        chat_request = _chat_request_from_diagnostic(config, request)
    _prompt, thinking = _render_chat_prompt_for_request(
        chat_request,
        chat_default_max_tokens=config.chat_default_max_tokens,
        engine=engine,
        max_context_tokens=max_context_tokens,
    )
    has_budget_policy = any(
        value is not None
        for value in (
            thinking.effort,
            thinking.max_think_tokens,
            thinking.min_answer_tokens,
            thinking.hard_think_cap,
            thinking.soft_close_window,
            thinking.hard_close_message,
            thinking.hard_close_sequence,
        )
    )
    if thinking.enabled is False or not has_budget_policy:
        return None
    close_text = thinking.hard_close_sequence or _THINKING_CLOSE_MARKER
    payload: dict[str, Any] = {
        "enabled": True,
        "effort": thinking.effort,
        "allow_unbounded": True if thinking.allow_unbounded else None,
        "max_think_tokens": thinking.max_think_tokens,
        "min_answer_tokens": thinking.min_answer_tokens,
        "hard_think_cap": thinking.hard_think_cap,
        "soft_close_window": thinking.soft_close_window,
        "hard_close_message": thinking.hard_close_message,
        "close_text": close_text,
    }
    try:
        close_token_ids = _tokenize_text(engine, close_text)
    except OpenAIHTTPError:
        payload["lowering_supported"] = False
        return {key: value for key, value in payload.items() if value is not None}
    state = ThinkingBudgetState(
        close_sequence=close_token_ids,
        hard_token_cap=thinking.hard_think_cap,
        soft_close_window=0 if thinking.soft_close_window is None else thinking.soft_close_window,
    )
    payload.update(
        {
            "lowering_supported": True,
            "close_token_ids": list(close_token_ids),
            "initial_state": state.to_json_dict(),
        }
    )
    return {key: value for key, value in payload.items() if value is not None}


def _diagnostic_session_payload(
    request: ChatCompletionRequest,
    prefix_messages: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    session_id = _session_id(request)
    if session_id is None:
        return None
    prefix_count = len(prefix_messages)
    request_count = len(request.messages)
    return {
        "id": session_id,
        "stateful": True,
        "storage": "app_local_transcript",
        "resident_state_reuse": False,
        "prefix_message_count": prefix_count,
        "request_message_count": request_count,
        "rendered_message_count": prefix_count + request_count,
        "cache_action": _session_cache_action(request),
    }


def _context_policy_session_prefix_payload(
    request: ChatCompletionRequest,
    *,
    session_id: str | None,
    clear_policy: str,
    dropped_message_count: int,
    kept_prefix_message_count: int,
    reset: bool,
) -> dict[str, Any]:
    kept_segments: list[dict[str, Any]] = []
    if kept_prefix_message_count > 0:
        kept_segments.append(
            {
                "kind": "session_prefix",
                "session_id": session_id,
                "storage": "app_local_transcript",
                "message_count": int(kept_prefix_message_count),
            }
        )
    kept_segments.append(
        {
            "kind": "request_messages",
            "message_count": len(request.messages),
        }
    )
    would_drop: list[dict[str, Any]] = []
    if dropped_message_count > 0:
        would_drop.append(
            {
                "kind": "session_prefix",
                "session_id": session_id,
                "storage": "app_local_transcript",
                "message_count": int(dropped_message_count),
            }
        )
    payload = {
        "clear_policy": clear_policy,
        "would_truncate": clear_policy == "truncate_oldest_visible" and dropped_message_count > 0,
        "would_reset_session": bool(reset),
        "would_drop": would_drop,
        "kept_segments": kept_segments,
    }
    if clear_policy == "auto_clear_transient":
        payload["would_clear_transient"] = False
        payload["transient_message_count"] = 0
    return payload


def _thinking_budget_sampling_kwargs(
    config: ServerConfig,
    request: ChatCompletionRequest,
    *,
    engine: Any,
    max_context_tokens: int | None,
    prepared_thinking: _ThinkingControl | None = None,
) -> dict[str, Any]:
    if prepared_thinking is None:
        _prompt, thinking = _render_chat_prompt_for_request(
            request,
            chat_default_max_tokens=config.chat_default_max_tokens,
            engine=engine,
            max_context_tokens=max_context_tokens,
            validate_tool_transcript=False,
        )
    else:
        thinking = prepared_thinking
    if thinking.enabled is False or thinking.hard_think_cap is None:
        return {}
    close_text = thinking.hard_close_sequence or _THINKING_CLOSE_MARKER
    try:
        close_token_ids = _tokenize_text(engine, close_text)
    except OpenAIHTTPError:
        return {}
    if not close_token_ids:
        return {}
    return {
        "thinking_close_token_ids": close_token_ids,
        "thinking_hard_token_cap": thinking.hard_think_cap,
        "thinking_soft_close_window": 0 if thinking.soft_close_window is None else thinking.soft_close_window,
    }


def _no_tool_sampling_suppress_token_ids(
    request: ChatCompletionRequest,
    engine: Any,
) -> tuple[int, ...]:
    if not request.tools:
        return ()
    mode, _name = _tool_choice_mode(request.tool_choice)
    if mode != "none":
        return ()
    try:
        token_ids = _tokenize_text(engine, _TOOL_CALL_START_MARKER)
    except OpenAIHTTPError:
        return ()
    if not token_ids:
        return ()
    return (int(token_ids[0]),)


def _tool_call_sampling_constraint(
    request: ChatCompletionRequest,
    engine: Any,
    *,
    chat_default_max_tokens: int | None,
) -> ToolCallConstraintSpec | None:
    tokenizer_caps = _tokenizer_capability_flags(engine)
    if not request.tools or not tokenizer_caps["tokenize"] or not tokenizer_caps["detokenize"]:
        return None
    mode, requested_name = _tool_choice_mode(request.tool_choice)
    if mode == "none":
        return None
    if mode == "function":
        names = () if requested_name is None else (str(requested_name),)
        constraint_mode = "required"
    else:
        names = tuple(
            str(name)
            for tool in request.tools
            if isinstance((name := _tool_function(tool).get("name")), str) and name
        )
        constraint_mode = "required" if mode == "required" else "auto"
    if not names:
        return None
    thinking = _thinking_control_from_request(
        request,
        chat_default_max_tokens=chat_default_max_tokens,
    )
    thinking_enabled = thinking.enabled is not False
    forbidden = (_THINKING_START_MARKER,) if not thinking_enabled else ()
    return ToolCallConstraintSpec(
        tool_names=names,
        mode=constraint_mode,
        forbidden_text_prefixes=forbidden,
        thinking_start_marker=_THINKING_START_MARKER if thinking_enabled else None,
        thinking_end_marker=_THINKING_CLOSE_MARKER if thinking_enabled else None,
    )


def _required_tool_sampling_forced_prefix(
    request: ChatCompletionRequest,
    engine: Any,
) -> tuple[tuple[int, ...], bool]:
    if not request.tools:
        return (), False
    mode, _name = _tool_choice_mode(request.tool_choice)
    if mode not in {"required", "function"}:
        return (), False
    try:
        start_ids = _tokenize_text(engine, _TOOL_CALL_START_MARKER)
    except OpenAIHTTPError:
        return (), False
    start = tuple(int(token_id) for token_id in start_ids)
    name = _specific_tool_name_prefix_target(request)
    if name is None:
        return start, False
    try:
        prefix_ids = _tokenize_text(engine, _tool_call_name_prefix_text(name))
    except OpenAIHTTPError:
        return start, False
    prefix = tuple(int(token_id) for token_id in prefix_ids)
    if not prefix:
        return start, False
    schema_prefix = _specific_tool_first_required_string_prefix(request, name)
    if schema_prefix is None:
        return prefix, False
    try:
        schema_prefix_ids = _tokenize_text(engine, schema_prefix)
    except OpenAIHTTPError:
        return prefix, False
    schema_prefix_tokens = tuple(int(token_id) for token_id in schema_prefix_ids)
    if not schema_prefix_tokens:
        return prefix, False
    return schema_prefix_tokens, True


def _tool_call_sequence_completion_token_sequences(
    request: ChatCompletionRequest,
    engine: Any,
    forced_tool_token_ids: Sequence[int],
    *,
    include_object_close: bool,
) -> tuple[tuple[int, ...], ...]:
    return _tool_call_close_repair_token_sequences(
        request,
        engine,
        forced_tool_token_ids,
        include_object_close=include_object_close,
    )


def _specific_tool_name_prefix_target(request: ChatCompletionRequest) -> str | None:
    if not request.tools:
        return None
    mode, requested_name = _tool_choice_mode(request.tool_choice)
    if mode == "function":
        return requested_name
    if mode != "required":
        return None
    names: list[str] = []
    for tool in request.tools:
        name = _tool_function(tool).get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names[0] if len(names) == 1 else None


def _tool_call_name_prefix_text(name: str) -> str:
    return (
        f"{_TOOL_CALL_START_MARKER}"
        f'{{"name":{json.dumps(str(name), ensure_ascii=False, separators=(",", ":"))},"arguments":'
    )


def _specific_tool_first_required_string_prefix(
    request: ChatCompletionRequest,
    name: str,
) -> str | None:
    tool = _tool_map_by_name(request.tools).get(str(name))
    if tool is None:
        return None
    function = _tool_function(tool)
    if function.get("strict") is not True:
        return None
    schema = _tool_parameters_schema(tool)
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        return None
    if schema.get("additionalProperties") is not False:
        return None
    required = schema.get("required")
    properties = schema.get("properties")
    if (
        not isinstance(required, Sequence)
        or isinstance(required, (str, bytes))
        or not required
        or not isinstance(properties, Mapping)
    ):
        return None
    key = required[0]
    property_schema = properties.get(key) if isinstance(key, str) else None
    if not isinstance(property_schema, Mapping) or property_schema.get("type") != "string":
        return None
    encoded_key = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
    return f"{_tool_call_name_prefix_text(name)}{{{encoded_key}:\""


def _tool_call_close_repair_token_sequences(
    request: ChatCompletionRequest,
    engine: Any,
    forced_tool_token_ids: Sequence[int],
    *,
    include_object_close: bool,
) -> tuple[tuple[int, ...], ...]:
    if not request.tools:
        return ()
    mode, _name = _tool_choice_mode(request.tool_choice)
    if mode not in {"auto", "required", "function"}:
        return ()
    try:
        marker_ids = _tokenize_text(engine, _TOOL_CALL_END_MARKER)
    except OpenAIHTTPError:
        return ()
    object_close_ids: tuple[int, ...] = ()
    if include_object_close:
        try:
            object_close_ids = _tokenize_text(engine, f"}}}}{_TOOL_CALL_END_MARKER}")
        except OpenAIHTTPError:
            pass
    full = tuple(int(token_id) for token_id in marker_ids)
    object_close = tuple(int(token_id) for token_id in object_close_ids)
    forced = tuple(int(token_id) for token_id in forced_tool_token_ids)
    sequences: list[tuple[int, ...]] = []
    for sequence in (object_close, full):
        if sequence and sequence not in sequences and not _token_sequence_occurs(forced, sequence):
            sequences.append(sequence)
    return tuple(sequences)


def _token_sequence_occurs(token_ids: Sequence[int], sequence: Sequence[int]) -> bool:
    tokens = tuple(int(token_id) for token_id in token_ids)
    target = tuple(int(token_id) for token_id in sequence)
    if not target or len(target) > len(tokens):
        return False
    return any(
        tokens[index : index + len(target)] == target
        for index in range(len(tokens) - len(target) + 1)
    )


def _normalize_prompts(
    prompt: str | list[str] | list[int] | list[list[int]] | None,
) -> tuple[PromptInput, ...]:
    if prompt is None:
        raise OpenAIHTTPError(400, "prompt is required", code="invalid_request", param="prompt")
    if isinstance(prompt, str):
        return (prompt,)
    if not prompt:
        raise OpenAIHTTPError(400, "prompt must not be empty", param="prompt")
    try:
        if all(isinstance(item, int) and not isinstance(item, bool) for item in prompt):
            return (normalize_prompt_input(prompt),)
        if all(isinstance(item, str) for item in prompt):
            return tuple(str(item) for item in prompt)
        if all(
            isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray))
            for item in prompt
        ):
            return tuple(normalize_prompt_input(item) for item in prompt)
    except (TypeError, ValueError) as exc:
        raise OpenAIHTTPError(400, str(exc), code="invalid_request", param="prompt") from exc
    raise OpenAIHTTPError(
        400,
        "prompt must be text, text rows, one token-ID row, or token-ID rows",
        code="invalid_request",
        param="prompt",
    )


def _completion_prompt_uses_token_ids(request: CompletionRequest | ChatCompletionRequest) -> bool:
    if not isinstance(request, CompletionRequest):
        return False
    prompt = request.prompt
    if not isinstance(prompt, list) or not prompt:
        return False
    return all(isinstance(item, int) and not isinstance(item, bool) for item in prompt) or all(
        isinstance(item, list)
        and bool(item)
        and all(isinstance(token, int) and not isinstance(token, bool) for token in item)
        for item in prompt
    )


def _request_n(request: CompletionRequest | ChatCompletionRequest) -> int:
    n = 1 if request.n is None else int(request.n)
    if n < 1:
        raise OpenAIHTTPError(400, "n must be at least 1", code="invalid_request", param="n")
    return n


def _request_speculative_provider_config(
    request: CompletionRequest | ChatCompletionRequest,
) -> tuple[bool | None, str | None, int | None]:
    raw = getattr(request, "speculative", None)
    if raw is None:
        return None, None, None
    if isinstance(raw, bool):
        return raw, None, None
    if not isinstance(raw, Mapping):
        raise OpenAIHTTPError(
            400,
            "speculative must be a boolean or object",
            code="unsupported_parameter",
            param="speculative",
        )
    unknown = set(raw) - _SPECULATIVE_PROVIDER_ALLOWED_REQUEST_KEYS
    if unknown:
        raise OpenAIHTTPError(
            400,
            "speculative contains unsupported fields: "
            + ", ".join(sorted(str(key) for key in unknown)),
            code="unsupported_parameter",
            param="speculative",
        )
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise OpenAIHTTPError(
            400,
            "speculative.enabled must be a boolean",
            code="unsupported_parameter",
            param="speculative",
        )
    provider = raw.get("provider")
    if provider is not None:
        provider = str(provider).strip()
        if not provider:
            raise OpenAIHTTPError(
                400,
                "speculative.provider must be non-empty",
                code="unsupported_parameter",
                param="speculative",
            )
    budget = raw.get("candidate_budget")
    if budget is not None:
        if isinstance(budget, bool):
            raise OpenAIHTTPError(
                400,
                "speculative.candidate_budget must be a positive integer",
                code="unsupported_parameter",
                param="speculative",
            )
        try:
            budget = int(budget)
        except (TypeError, ValueError) as exc:
            raise OpenAIHTTPError(
                400,
                "speculative.candidate_budget must be a positive integer",
                code="unsupported_parameter",
                param="speculative",
            ) from exc
        if budget <= 0:
            raise OpenAIHTTPError(
                400,
                "speculative.candidate_budget must be a positive integer",
                code="unsupported_parameter",
                param="speculative",
            )
    return enabled, provider, budget


def _request_speculative_mtp_enabled(request: CompletionRequest | ChatCompletionRequest) -> bool | None:
    raw = getattr(request, "speculative_mtp", None)
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, Mapping):
        enabled = raw.get("enabled", True)
        if isinstance(enabled, bool):
            return enabled
        raise OpenAIHTTPError(
            400,
            "speculative_mtp.enabled must be a boolean",
            code="unsupported_parameter",
            param="speculative_mtp",
        )
    raise OpenAIHTTPError(
        400,
        "speculative_mtp must be a boolean or object",
        code="unsupported_parameter",
        param="speculative_mtp",
    )


def _request_speculative_mtp_thinking(
    config: ServerConfig,
    request: CompletionRequest | ChatCompletionRequest,
) -> str:
    """Return the per-request MTP thinking policy ("hint" or "hard").

    ``hint`` keeps the thinking hints in the rendered prompt but relaxes the
    host-sampler thinking-budget enforcement (soft-close bias, EOS
    suppression, hard-close forcing) so the request can use the exact
    raw-argmax MTP route.  ``hard`` keeps full host-sampler enforcement and
    treats the thinking budget as a hard MTP blocker.
    """

    raw = getattr(request, "speculative_mtp", None)
    if isinstance(raw, Mapping):
        thinking = raw.get("thinking")
        if thinking is not None:
            mode = str(thinking).strip().lower().replace("-", "_")
            if mode not in _SPECULATIVE_MTP_THINKING_MODES:
                raise OpenAIHTTPError(
                    400,
                    "speculative_mtp.thinking must be 'hint' or 'hard'",
                    code="unsupported_parameter",
                    param="speculative_mtp",
                )
            return mode
    return str(config.speculative_mtp_thinking)


def _speculative_mtp_route_for_request(
    config: ServerConfig,
    request: CompletionRequest | ChatCompletionRequest,
    *,
    engine: Any | None,
    sampling: SamplingParams,
) -> str:
    explicit = _request_speculative_mtp_enabled(request)
    configured_mode = str(config.speculative_mtp_serving)
    if explicit is False:
        return _SPECULATIVE_MTP_DEFAULT_ROUTE
    if explicit is None and configured_mode not in {"auto", "enabled"}:
        return _SPECULATIVE_MTP_DEFAULT_ROUTE
    explicit_requested = explicit is True
    if configured_mode == "off":
        if explicit_requested:
            raise OpenAIHTTPError(
                400,
                "speculative_mtp is disabled for this server",
                code="unsupported_parameter",
                param="speculative_mtp",
            )
        return _SPECULATIVE_MTP_DEFAULT_ROUTE
    if not _engine_supports_speculative_mtp(engine):
        if explicit_requested:
            raise OpenAIHTTPError(
                501,
                "speculative_mtp is not supported by this model/backend",
                error_type="unsupported_feature",
                code="unsupported_feature",
                param="speculative_mtp",
            )
        return _SPECULATIVE_MTP_DEFAULT_ROUTE
    if (
        configured_mode == "enabled"
        and not explicit_requested
        and not _engine_supports_default_mtp(engine)
    ):
        # Missing typed automatic evidence stays K0 so enabled/capabilities/
        # response reporting agree.
        return _SPECULATIVE_MTP_K0_ROUTE
    thinking_policy = _request_speculative_mtp_thinking(config, request)
    if thinking_policy == "hint":
        # The host-sampler thinking budget is relaxed to prompt-hint-only for
        # the raw-argmax MTP route: the rendered prompt keeps its thinking
        # hints, but the sampler-level enforcement fields are cleared so the
        # request is raw-greedy-exact again.  Under the hard policy the
        # thinking budget stays a hard blocker and MTP falls back to AR.
        sampling = relax_thinking_budget_for_mtp(sampling)
    blockers = speculative_mtp_sampling_blockers(sampling)
    if blockers:
        return _SPECULATIVE_MTP_K0_ROUTE
    if not supports_speculative_mtp_sampling(sampling):
        return _SPECULATIVE_MTP_DEFAULT_ROUTE
    if explicit_requested:
        return _SPECULATIVE_MTP_BATCH_ROUTE
    if configured_mode == "enabled":
        return _SPECULATIVE_MTP_BATCH_ROUTE
    return _SPECULATIVE_MTP_AUTO_ROUTE


def _speculative_provider_route_for_request(
    config: ServerConfig,
    request: CompletionRequest | ChatCompletionRequest,
    *,
    engine: Any | None,
    sampling: SamplingParams,
    prompt_count: int,
) -> str | None:
    enabled, requested_provider, requested_budget = (
        _request_speculative_provider_config(request)
    )
    if not enabled:
        return None
    if _request_speculative_mtp_enabled(request):
        raise OpenAIHTTPError(
            400,
            "speculative and speculative_mtp cannot both be enabled",
            code="unsupported_parameter",
            param="speculative",
        )
    configured_provider = config.speculative_provider
    if configured_provider is None:
        raise OpenAIHTTPError(
            400,
            "speculative provider serving is not configured for this server",
            code="unsupported_parameter",
            param="speculative",
        )
    selected_provider = requested_provider or configured_provider
    if selected_provider != configured_provider:
        raise OpenAIHTTPError(
            400,
            "requested speculative provider does not match the configured provider",
            code="unsupported_parameter",
            param="speculative",
        )
    candidate_budget = (
        int(config.speculative_candidate_budget)
        if requested_budget is None
        else int(requested_budget)
    )
    if candidate_budget != int(config.speculative_candidate_budget):
        raise OpenAIHTTPError(
            400,
            "requested speculative candidate budget does not match the configured owner",
            code="unsupported_parameter",
            param="speculative",
        )
    blockers = [
        blocker
        for blocker in speculative_mtp_sampling_blockers(sampling)
        if blocker not in {"stop_token_ids", "stop_token_sequences"}
    ]
    if float(sampling.top_p) != 1.0:
        blockers.append("top_p")
    if int(sampling.top_k) != 0:
        blockers.append("top_k")
    if float(sampling.min_p) != 0.0:
        blockers.append("min_p")
    if str(sampling.kv_storage) not in {"auto", "bf16"}:
        blockers.append("kv_storage")
    if str(sampling.kv_scale_dtype) != "fp16":
        blockers.append("kv_scale_dtype")
    if str(sampling.kv_scale_granularity) != "per_token_head":
        blockers.append("kv_scale_granularity")
    if int(prompt_count) != 1 or _request_n(request) != 1:
        blockers.append("c")
    blockers = list(dict.fromkeys(blockers))
    if blockers:
        raise OpenAIHTTPError(
            400,
            "speculative provider requires raw greedy BF16 c=1 generation",
            code="unsupported_parameter",
            param="speculative",
            extra={
                "hipengine": {
                    "speculative": {
                        "provider": selected_provider,
                        "candidate_budget": candidate_budget,
                        "blockers": blockers,
                        "compatibility_guard": "raw_greedy_bf16_c1",
                    }
                }
            },
        )
    if not _engine_supports_speculative(engine):
        raise OpenAIHTTPError(
            501,
            "speculative provider is not supported by this model/backend",
            error_type="unsupported_feature",
            code="unsupported_feature",
            param="speculative",
        )
    if bool(getattr(request, "stream", False)) and not _engine_supports_speculative_streaming(engine):
        raise OpenAIHTTPError(
            501,
            "speculative provider does not support streaming",
            error_type="unsupported_feature",
            code="unsupported_feature",
            param="speculative",
        )
    return _SPECULATIVE_PROVIDER_ROUTE


def _engine_speculative_mtp_serving_plan(
    config: ServerConfig,
    request: CompletionRequest | ChatCompletionRequest,
    *,
    engine: Any | None,
    sampling: SamplingParams,
    prompts: Sequence[PromptInput],
) -> dict[str, Any] | None:
    """Return a model-plugin plan using mechanical request identity only."""

    resolver = getattr(engine, "resolve_speculative_mtp_serving_plan", None)
    if not callable(resolver) or not prompts:
        return None
    effective_sampling = sampling
    if _request_speculative_mtp_thinking(config, request) == "hint":
        effective_sampling = relax_thinking_budget_for_mtp(sampling)
    sampling_mode = (
        "greedy_fast"
        if supports_speculative_mtp_sampling(effective_sampling)
        else "processed_argmax"
    )
    decision = resolver(
        realized_group_rows=len(prompts),
        sampling_mode=sampling_mode,
        context_tokens=max(_prompt_token_count(engine, prompt) for prompt in prompts),
        output_horizon_tokens=int(effective_sampling.max_tokens),
        kv_storage=str(effective_sampling.kv_storage),
        memory_fit=True,
    )
    if decision is None:
        return None
    payload = decision.as_dict() if callable(getattr(decision, "as_dict", None)) else decision
    if not isinstance(payload, Mapping):
        raise TypeError("speculative MTP serving plan must be a mapping")
    return deepcopy(dict(payload))


def _generation_route_for_request(
    config: ServerConfig,
    request: CompletionRequest | ChatCompletionRequest,
    *,
    engine: Any | None,
    sampling: SamplingParams,
    prompts: Sequence[PromptInput],
) -> tuple[str, dict[str, Any] | None]:
    provider_route = _speculative_provider_route_for_request(
        config,
        request,
        engine=engine,
        sampling=sampling,
        prompt_count=len(prompts),
    )
    if provider_route is not None:
        return provider_route, None
    route = _speculative_mtp_route_for_request(
        config,
        request,
        engine=engine,
        sampling=sampling,
    )
    if route not in {
        _SPECULATIVE_MTP_BATCH_ROUTE,
        _SPECULATIVE_MTP_AUTO_ROUTE,
        _SPECULATIVE_MTP_K0_ROUTE,
    }:
        return route, None
    plan = _engine_speculative_mtp_serving_plan(
        config,
        request,
        engine=engine,
        sampling=sampling,
        prompts=prompts,
    )
    if plan is None:
        return route, None
    explicit_requested = _request_speculative_mtp_enabled(request) is True
    static_payload = plan.get("static_eligibility")
    static_eligible = bool(
        isinstance(static_payload, Mapping)
        and static_payload.get("eligible") is True
    )
    plan["static_intent_allowed"] = bool(
        static_eligible
        and (
            explicit_requested
            or (
                route in {
                    _SPECULATIVE_MTP_AUTO_ROUTE,
                    _SPECULATIVE_MTP_BATCH_ROUTE,
                }
                and bool(plan.get("automatic_eligible"))
            )
        )
    )
    if route == _SPECULATIVE_MTP_BATCH_ROUTE and not bool(plan.get("admitted")):
        route = _SPECULATIVE_MTP_K0_ROUTE
    elif (
        route == _SPECULATIVE_MTP_BATCH_ROUTE
        and _request_speculative_mtp_enabled(request) is not True
        and not bool(plan.get("automatic_eligible"))
    ):
        route = _SPECULATIVE_MTP_K0_ROUTE
    return route, plan


def _unsupported_agentic_request_param(
    request: CompletionRequest | ChatCompletionRequest,
    *,
    engine: Any | None = None,
) -> str | None:
    session = getattr(request, "session", None)
    if session is not None:
        if isinstance(session, Mapping):
            if "id" in session:
                if not isinstance(request, ChatCompletionRequest):
                    return "session.id"
                if request.stream and not bool(
                    getattr(
                        _model_chat_protocol_target(engine),
                        "supports_resident_session_kv",
                        False,
                    )
                ):
                    return "stream"
                if _request_n(request) != 1:
                    return "n"
                if set(session.keys()) - _SESSION_ALLOWED_STATEFUL_KEYS:
                    return "session"
                if _session_cache_action(request) is not None:
                    return None
                return "session.commit"
            if "context_overflow_policy" in session:
                return "session.context_overflow_policy"
            if _session_cache_action(request) is not None:
                return None
            if "commit" in session:
                return "session.commit"
        return "session"
    return None


def _validate_continuation_resume_request(request: CompletionRequest | ChatCompletionRequest) -> None:
    if getattr(request, "continuation_id", None) is None:
        return
    unsupported_param = _continuation_resume_unsupported_param(request)
    if unsupported_param is not None:
        raise OpenAIHTTPError(
            400,
            f"continuation_id resume does not support {unsupported_param} yet",
            code="unsupported_parameter",
            param=unsupported_param,
        )
    if request.stream:
        raise OpenAIHTTPError(
            400,
            "continuation_id resume does not support stream=true yet",
            code="unsupported_parameter",
            param="stream",
        )
    if _request_n(request) != 1:
        raise OpenAIHTTPError(
            400,
            "continuation_id resume requires n=1",
            code="unsupported_parameter",
            param="n",
        )
    if _request_logprobs_enabled(request):
        raise OpenAIHTTPError(
            400,
            "continuation_id resume does not support logprobs yet",
            code="unsupported_parameter",
            param="logprobs",
        )
    if isinstance(request, CompletionRequest) and request.echo:
        raise OpenAIHTTPError(
            400,
            "continuation_id resume does not support echo=true",
            code="unsupported_parameter",
            param="echo",
        )


def _continuation_resume_unsupported_param(request: CompletionRequest | ChatCompletionRequest) -> str | None:
    if isinstance(request, CompletionRequest) and request.prompt is not None:
        return "prompt"
    if isinstance(request, ChatCompletionRequest) and request.messages:
        return "messages"
    if request.temperature not in (None, 0, 0.0):
        return "temperature"
    if request.logit_bias:
        return "logit_bias"
    if request.suppress_token_ids:
        return "suppress_token_ids"
    if int(request.min_tokens or 0) != 0:
        return "min_tokens"
    if bool(request.ignore_eos):
        return "ignore_eos"
    if request.stop is not None:
        return "stop"
    if float(request.repetition_penalty or 1.0) != 1.0:
        return "repetition_penalty"
    if float(request.presence_penalty or 0.0) != 0.0:
        return "presence_penalty"
    if float(request.frequency_penalty or 0.0) != 0.0:
        return "frequency_penalty"
    if request.response_format is not None:
        return "response_format"
    if getattr(request, "guided_json", None) is not None:
        return "guided_json"
    if getattr(request, "guided_regex", None) is not None:
        return "guided_regex"
    if getattr(request, "guided_choice", None) is not None:
        return "guided_choice"
    if getattr(request, "guided_patch", None) is not None:
        return "guided_patch"
    if getattr(request, "guided_diff", None) is not None:
        return "guided_diff"
    if isinstance(request, ChatCompletionRequest):
        if request.tools:
            return "tools"
        if request.tool_choice is not None:
            return "tool_choice"
        if request.parallel_tool_calls is not None:
            return "parallel_tool_calls"
        thinking_param = _thinking_budget_sampling_unsupported_param(request)
        if thinking_param is not None:
            return thinking_param
    return None


def _continuation_can_create(
    request: CompletionRequest | ChatCompletionRequest,
    *,
    finish_reason: str,
    finish_details: Mapping[str, Any],
    backend_continuation_eligible: bool | None = None,
) -> bool:
    if _completion_prompt_uses_token_ids(request):
        return False
    if backend_continuation_eligible is False:
        return False
    if str(finish_details.get("reason") or "") in (_SESSION_UNSAFE_VISIBLE_REASONS - {"length"}):
        return False
    if not _is_length_finish(finish_reason, finish_details):
        return False
    session_id = _session_id(request)
    if session_id is not None:
        if not isinstance(request, ChatCompletionRequest):
            return False
        cache_action = str(finish_details.get("cache_action") or _session_cache_action(request) or "")
        if cache_action in {"", "append_none"}:
            return False
    if request.stream or _request_n(request) != 1 or _request_logprobs_enabled(request):
        return False
    if isinstance(request, CompletionRequest) and request.echo:
        return False
    if isinstance(request, ChatCompletionRequest):
        phase = str(finish_details.get("phase") or "")
        if phase and phase not in {"answer", "structured"}:
            return False
        if request.tools or request.tool_choice is not None or request.parallel_tool_calls is not None:
            return False
        if _thinking_budget_sampling_unsupported_param(request) is not None:
            return False
    return _continuation_sampling_is_deterministic(request)


def _backend_continuation_eligible(detail: GenerationOutput | None) -> bool | None:
    finish_details = None if detail is None else detail.finish_details
    if finish_details is None:
        return None
    return finish_details.continuation_eligible


def _thinking_budget_sampling_kwargs_present(request: ChatCompletionRequest) -> bool:
    return _thinking_budget_sampling_unsupported_param(request) is not None


def _thinking_budget_sampling_unsupported_param(request: ChatCompletionRequest) -> str | None:
    for name in (
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
    ):
        if getattr(request, name) is not None:
            return name
    return None


def _continuation_sampling_is_deterministic(request: CompletionRequest | ChatCompletionRequest) -> bool:
    if request.temperature not in (None, 0, 0.0):
        return False
    if request.logit_bias:
        return False
    if request.suppress_token_ids:
        return False
    if int(request.min_tokens or 0) != 0:
        return False
    if bool(request.ignore_eos):
        return False
    if request.stop is not None:
        return False
    if float(request.repetition_penalty or 1.0) != 1.0:
        return False
    if float(request.presence_penalty or 0.0) != 0.0:
        return False
    if float(request.frequency_penalty or 0.0) != 0.0:
        return False
    return True


def _mark_continuation_unavailable(finish_details: dict[str, Any]) -> None:
    if _is_length_finish(str(finish_details.get("reason", "")), finish_details):
        finish_details["continuation_eligible"] = False


def _mark_structured_length_phase(
    request: CompletionRequest | ChatCompletionRequest,
    finish_details: dict[str, Any],
) -> None:
    if not _structured_result_validation(request):
        return
    if not _is_length_finish(str(finish_details.get("reason", "")), finish_details):
        return
    if finish_details.get("phase") in (None, "", "answer"):
        finish_details["phase"] = "structured"


def _mark_structured_length_failure(
    request: CompletionRequest | ChatCompletionRequest,
    structured_length_failure: str | None,
    finish_details: dict[str, Any],
) -> None:
    if structured_length_failure is None:
        return
    if (
        _structured_result_validation(request)
        and finish_details.get("phase") in (None, "", "answer")
    ):
        finish_details["phase"] = "structured"
    finish_details["continuation_eligible"] = False


def _attach_continuation_metadata(
    choice: dict[str, Any],
    *,
    continuation_id: str,
) -> None:
    choice["continuation_id"] = continuation_id
    details = choice.setdefault("finish_details", {})
    if isinstance(details, dict):
        details["continuation_eligible"] = True
        details["continuation_id"] = continuation_id


def _apply_continuation_defaults(
    request: CompletionRequest | ChatCompletionRequest,
    record: _ContinuationRecord | None,
) -> None:
    if record is None:
        return
    if getattr(request, "response_format", None) is None and record.response_format is not None:
        request.response_format = record.response_format
    if getattr(request, "guided_json", None) is None and record.guided_json is not None:
        request.guided_json = record.guided_json
    if getattr(request, "guided_regex", None) is None and record.guided_regex is not None:
        request.guided_regex = record.guided_regex
    if getattr(request, "guided_choice", None) is None and record.guided_choice is not None:
        request.guided_choice = record.guided_choice
    if getattr(request, "guided_patch", None) is None and record.guided_patch is not None:
        request.guided_patch = record.guided_patch
    if getattr(request, "guided_diff", None) is None and record.guided_diff is not None:
        request.guided_diff = record.guided_diff


def _validate_session_request(request: CompletionRequest | ChatCompletionRequest) -> None:
    session = getattr(request, "session", None)
    if session is None or not isinstance(session, Mapping):
        return
    if "id" in session:
        _session_id(request)
    if "context_overflow_policy" in session:
        _session_context_overflow_policy(request)
    if "commit" not in session:
        return
    raw_commit = session.get("commit")
    if not isinstance(raw_commit, str) or not raw_commit.strip():
        raise OpenAIHTTPError(
            400,
            "session.commit must be a non-empty string",
            code="invalid_request",
            param="session.commit",
        )
    mode = raw_commit.strip().lower()
    if "id" in session and mode not in _SESSION_COMMIT_MODES:
        raise OpenAIHTTPError(
            400,
            "session.commit must be one of: " + ", ".join(_SESSION_COMMIT_MODES),
            code="invalid_request",
            param="session.commit",
        )


def _session_context_overflow_policy(request: CompletionRequest | ChatCompletionRequest) -> str:
    session = getattr(request, "session", None)
    if not isinstance(session, Mapping):
        return "reject"
    raw_policy = session.get("context_overflow_policy")
    if raw_policy is None:
        return "reject"
    if not isinstance(raw_policy, str) or not raw_policy.strip():
        raise OpenAIHTTPError(
            400,
            "session.context_overflow_policy must be a non-empty string",
            code="invalid_request",
            param="session.context_overflow_policy",
        )
    policy = raw_policy.strip().lower()
    policy = _SESSION_CONTEXT_OVERFLOW_POLICY_ALIASES.get(policy, policy)
    if policy not in _SESSION_CONTEXT_OVERFLOW_POLICIES:
        raise OpenAIHTTPError(
            400,
            "session.context_overflow_policy must be one of: "
            + ", ".join(_SESSION_CONTEXT_OVERFLOW_POLICIES),
            code="invalid_request",
            param="session.context_overflow_policy",
        )
    return policy


def _session_id(request: CompletionRequest | ChatCompletionRequest) -> str | None:
    session = getattr(request, "session", None)
    if not isinstance(session, Mapping) or "id" not in session:
        return None
    raw_id = session.get("id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise OpenAIHTTPError(
            400,
            "session.id must be a non-empty string",
            code="invalid_request",
            param="session.id",
        )
    return raw_id.strip()


def _resident_chat_session_key(
    auth_principal: str,
    request: ChatCompletionRequest,
) -> str | None:
    session_id = _session_id(request)
    if session_id is None:
        return None
    payload = f"{str(auth_principal)}\0{session_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _session_cache_action(request: CompletionRequest | ChatCompletionRequest) -> str | None:
    """Return the effective requested session commit/cache action."""

    session = getattr(request, "session", None)
    if session is None:
        return "append_none"
    if not isinstance(session, Mapping):
        return None
    has_id = "id" in session
    commit = session.get("commit")
    if commit is None and has_id:
        return _SESSION_STATEFUL_DEFAULT_COMMIT
    if commit is None:
        return None
    mode = str(commit).strip().lower()
    if has_id:
        if mode in _SESSION_COMMIT_MODES and not (set(session.keys()) - _SESSION_ALLOWED_STATEFUL_KEYS):
            return mode
        return None
    if mode == "append_none" and set(session.keys()) == {"commit"}:
        return mode
    return None


def _effective_session_cache_action(
    requested_action: str | None,
    finish_details: Mapping[str, Any],
) -> str | None:
    if requested_action != "append_visible_only":
        return requested_action
    reason = str(finish_details.get("reason") or "")
    synthetic_tokens = int(finish_details.get("synthetic_tokens") or 0)
    if reason in _SESSION_UNSAFE_VISIBLE_REASONS or synthetic_tokens > 0:
        return "append_prompt_only"
    return requested_action


def _chat_message_to_session_dict(message: ChatMessage | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(message, Mapping):
        payload = dict(message)
    elif hasattr(message, "model_dump"):
        payload = message.model_dump(exclude_none=True)
    else:
        payload = message.dict(exclude_none=True)
    return _chat_session_message_copy(
        {str(key): value for key, value in payload.items() if value is not None}
    )


def _assistant_visible_session_message(message: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": "assistant"}
    if "content" in message:
        payload["content"] = deepcopy(message.get("content"))
    else:
        payload["content"] = ""
    tool_calls = message.get("tool_calls")
    if tool_calls:
        payload["tool_calls"] = deepcopy(tool_calls)
    return payload


def _chat_session_message_copy(message: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(message))


def _chat_session_snapshot_messages(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise OpenAIHTTPError(
            400,
            "session snapshot messages must be an array",
            code="invalid_request",
            param="messages",
        )
    messages = tuple(
        _chat_session_snapshot_message(item, index=index)
        for index, item in enumerate(value)
    )
    _validate_chat_message_tool_transcript(messages, context="session snapshot")
    return messages


def _chat_session_snapshot_message(value: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OpenAIHTTPError(
            400,
            f"session snapshot messages[{index}] must be an object",
            code="invalid_request",
            param=f"messages[{index}]",
        )
    allowed = {"role", "content", "name", "tool_call_id", "tool_calls"}
    extra = sorted(str(key) for key in value if str(key) not in allowed)
    if extra:
        raise OpenAIHTTPError(
            400,
            f"session snapshot messages[{index}].{extra[0]} is not supported",
            code="invalid_request",
            param=f"messages[{index}].{extra[0]}",
        )
    role = _chat_message_role(value.get("role"), param=f"messages[{index}].role")
    _validate_chat_message_tool_fields(
        role,
        tool_calls=value.get("tool_calls"),
        tool_call_id=value.get("tool_call_id"),
        param_prefix=f"messages[{index}]",
    )
    payload: dict[str, Any] = {"role": role}
    if "content" in value:
        payload["content"] = _chat_session_snapshot_content(value.get("content"), message_index=index)
    for key in ("name", "tool_call_id"):
        if key in value:
            payload[key] = _chat_session_snapshot_string(value.get(key), param=f"messages[{index}].{key}")
    if "tool_calls" in value:
        payload["tool_calls"] = _chat_message_tool_calls(
            value.get("tool_calls"),
            message_index=index,
            context="session snapshot",
        )
    return payload


def _chat_session_snapshot_string(value: Any, *, param: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpenAIHTTPError(
            400,
            f"session snapshot {param} must be a non-empty string",
            code="invalid_request",
            param=param,
        )
    return value


def _chat_session_snapshot_content(value: Any, *, message_index: int) -> str | list[Any] | None:
    param = f"messages[{message_index}].content"
    if value is None or isinstance(value, str):
        return value
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise OpenAIHTTPError(
            400,
            f"session snapshot {param} must be text, null, or text content parts",
            code="invalid_request",
            param=param,
        )
    parts: list[Any] = []
    for part_index, part in enumerate(value):
        part_param = f"{param}[{part_index}]"
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, Mapping):
            raise OpenAIHTTPError(
                400,
                f"session snapshot {part_param} must be a text object",
                code="invalid_request",
                param=part_param,
            )
        part_type = part.get("type", "text")
        if part_type != "text":
            raise OpenAIHTTPError(
                400,
                f"session snapshot {part_param}.type must be 'text'",
                code="invalid_request",
                param=part_param,
            )
        text = part.get("text", "")
        if not isinstance(text, str):
            raise OpenAIHTTPError(
                400,
                f"session snapshot {part_param}.text must be a string",
                code="invalid_request",
                param=f"{part_param}.text",
            )
        parts.append(dict(part))
    return parts


def _chat_message_tool_call(
    value: Any,
    *,
    message_index: int,
    tool_index: int,
    context: str,
) -> dict[str, Any]:
    param = f"messages[{message_index}].tool_calls[{tool_index}]"
    if not isinstance(value, Mapping):
        raise OpenAIHTTPError(
            400,
            f"{context} {param} must be an object",
            code="invalid_request",
            param=param,
        )
    allowed = {"id", "type", "function"}
    extra = sorted(str(key) for key in value if str(key) not in allowed)
    if extra:
        raise OpenAIHTTPError(
            400,
            f"{context} {param}.{extra[0]} is not supported",
            code="invalid_request",
            param=f"{param}.{extra[0]}",
        )
    call_id = value.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise OpenAIHTTPError(
            400,
            f"{context} {param}.id must be a non-empty string",
            code="invalid_request",
            param=f"{param}.id",
        )
    call_type = value.get("type")
    if call_type != "function":
        raise OpenAIHTTPError(
            400,
            f"{context} {param}.type must be 'function'",
            code="invalid_request",
            param=f"{param}.type",
        )
    function = value.get("function")
    function_param = f"{param}.function"
    if not isinstance(function, Mapping):
        raise OpenAIHTTPError(
            400,
            f"{context} {function_param} must be an object",
            code="invalid_request",
            param=function_param,
        )
    function_allowed = {"name", "arguments"}
    function_extra = sorted(str(key) for key in function if str(key) not in function_allowed)
    if function_extra:
        raise OpenAIHTTPError(
            400,
            f"{context} {function_param}.{function_extra[0]} is not supported",
            code="invalid_request",
            param=f"{function_param}.{function_extra[0]}",
        )
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise OpenAIHTTPError(
            400,
            f"{context} {function_param}.name must be a non-empty string",
            code="invalid_request",
            param=f"{function_param}.name",
        )
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise OpenAIHTTPError(
            400,
            f"{context} {function_param}.arguments must be a string",
            code="invalid_request",
            param=f"{function_param}.arguments",
        )
    try:
        json.loads(arguments)
    except Exception as exc:
        raise OpenAIHTTPError(
            400,
            f"{context} {function_param}.arguments must be valid JSON",
            code="invalid_request",
            param=f"{function_param}.arguments",
        ) from exc
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _chat_session_snapshot_time(value: Any, *, param: str) -> float:
    if isinstance(value, bool):
        value = None
    if not isinstance(value, (int, float)) or float(value) < 0:
        raise OpenAIHTTPError(
            400,
            f"session snapshot {param} must be a non-negative timestamp",
            code="invalid_request",
            param=param,
        )
    return float(value)


def _chat_request_with_messages(
    request: ChatCompletionRequest,
    messages: Sequence[ChatMessage | Mapping[str, Any]],
) -> ChatCompletionRequest:
    if hasattr(request, "model_copy"):
        return request.model_copy(update={"messages": list(messages)})
    return request.copy(update={"messages": list(messages)})


def _validate_response_format_request(request: CompletionRequest | ChatCompletionRequest) -> None:
    mode = _response_format_mode(request)
    if mode is None:
        return
    if mode in {"json_object", "json_schema"} and isinstance(request, CompletionRequest) and bool(request.echo):
        raise OpenAIHTTPError(
            400,
            f"response_format {mode} is incompatible with echo=true",
            code="invalid_request",
            param="echo",
        )
    if mode == "json_schema" and _response_format_json_schema(request) is None:
        raise OpenAIHTTPError(
            400,
            "response_format json_schema requires json_schema.schema",
            code="invalid_request",
            param="response_format.json_schema.schema",
        )
    if mode == "json_schema":
        schema = _response_format_json_schema(request)
        if schema is not None:
            schema_error = _validate_json_schema_subset(
                schema,
                path="response_format.json_schema.schema",
            )
            if schema_error is not None:
                param, message = schema_error
                raise OpenAIHTTPError(400, message, code="invalid_request", param=param)


def _validate_guided_json_request(request: CompletionRequest | ChatCompletionRequest) -> None:
    mode = _guided_json_mode(request)
    if mode is None:
        return
    if mode in {"json_object", "json_schema"} and isinstance(request, CompletionRequest) and bool(request.echo):
        raise OpenAIHTTPError(
            400,
            f"guided_json {mode} is incompatible with echo=true",
            code="invalid_request",
            param="echo",
        )
    response_mode = _response_format_mode(request)
    if response_mode not in {None, "text"}:
        raise OpenAIHTTPError(
            400,
            f"guided_json is incompatible with response_format {response_mode}",
            code="invalid_request",
            param="guided_json",
        )
    if getattr(request, "guided_regex", None) is not None:
        raise OpenAIHTTPError(
            400,
            "guided_json is incompatible with guided_regex",
            code="invalid_request",
            param="guided_json",
        )
    if getattr(request, "guided_choice", None) is not None:
        raise OpenAIHTTPError(
            400,
            "guided_json is incompatible with guided_choice",
            code="invalid_request",
            param="guided_json",
        )
    guided_patch_field = _guided_patch_field(
        getattr(request, "guided_patch", None),
        getattr(request, "guided_diff", None),
    )
    if guided_patch_field is not None:
        raise OpenAIHTTPError(
            400,
            f"guided_json is incompatible with {guided_patch_field}",
            code="invalid_request",
            param="guided_json",
        )
    if mode == "json_schema":
        schema = _guided_json_schema(request)
        if schema is None:
            raise OpenAIHTTPError(
                400,
                "guided_json schema must be an object",
                code="invalid_request",
                param="guided_json",
            )
        schema_error = _validate_json_schema_subset(schema, path=_guided_json_schema_param(request.guided_json))
        if schema_error is not None:
            param, message = schema_error
            raise OpenAIHTTPError(400, message, code="invalid_request", param=param)


def _guided_json_mode(request: CompletionRequest | ChatCompletionRequest) -> str | None:
    return _guided_json_mode_from_value(getattr(request, "guided_json", None), validate=True)


def _guided_json_mode_from_value(value: Any, *, validate: bool) -> str | None:
    if value is None or value is False:
        return None
    if value is True:
        return "json_object"
    if _guided_json_schema_from_value(value) is not None:
        return "json_schema"
    if validate:
        raise OpenAIHTTPError(
            400,
            "guided_json must be true, a JSON schema object, or a JSON schema string",
            code="invalid_request",
            param="guided_json",
        )
    return None


def _guided_json_schema(request: CompletionRequest | ChatCompletionRequest) -> Mapping[str, Any] | None:
    return _guided_json_schema_from_value(getattr(request, "guided_json", None))


def _guided_json_schema_from_value(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        wrapped = value.get("schema")
        if isinstance(wrapped, Mapping):
            return wrapped
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        if isinstance(parsed, Mapping):
            wrapped = parsed.get("schema")
            if isinstance(wrapped, Mapping):
                return wrapped
            return parsed
    return None


def _guided_json_schema_param(value: Any) -> str:
    if isinstance(value, Mapping) and isinstance(value.get("schema"), Mapping):
        return "guided_json.schema"
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return "guided_json"
        if isinstance(parsed, Mapping) and isinstance(parsed.get("schema"), Mapping):
            return "guided_json.schema"
    return "guided_json"


def _validate_guided_choice_request(request: CompletionRequest | ChatCompletionRequest) -> None:
    if getattr(request, "guided_choice", None) is None:
        return
    _guided_choice_values(request.guided_choice, validate=True)
    response_mode = _response_format_mode(request)
    if response_mode not in {None, "text"}:
        raise OpenAIHTTPError(
            400,
            f"guided_choice is incompatible with response_format {response_mode}",
            code="invalid_request",
            param="guided_choice",
        )
    guided_patch_field = _guided_patch_field(
        getattr(request, "guided_patch", None),
        getattr(request, "guided_diff", None),
    )
    if guided_patch_field is not None:
        raise OpenAIHTTPError(
            400,
            f"guided_choice is incompatible with {guided_patch_field}",
            code="invalid_request",
            param="guided_choice",
        )


def _validate_guided_regex_request(request: CompletionRequest | ChatCompletionRequest) -> None:
    if getattr(request, "guided_regex", None) is None:
        return
    pattern = _guided_regex_pattern(request.guided_regex, validate=True)
    assert pattern is not None
    try:
        re.compile(pattern)
    except re.error as exc:
        raise OpenAIHTTPError(
            400,
            f"guided_regex is not a valid regular expression: {exc}",
            code="invalid_request",
            param="guided_regex",
        ) from exc
    response_mode = _response_format_mode(request)
    if response_mode not in {None, "text"}:
        raise OpenAIHTTPError(
            400,
            f"guided_regex is incompatible with response_format {response_mode}",
            code="invalid_request",
            param="guided_regex",
        )
    if getattr(request, "guided_choice", None) is not None:
        raise OpenAIHTTPError(
            400,
            "guided_regex is incompatible with guided_choice",
            code="invalid_request",
            param="guided_regex",
        )
    guided_patch_field = _guided_patch_field(
        getattr(request, "guided_patch", None),
        getattr(request, "guided_diff", None),
    )
    if guided_patch_field is not None:
        raise OpenAIHTTPError(
            400,
            f"guided_regex is incompatible with {guided_patch_field}",
            code="invalid_request",
            param="guided_regex",
        )


def _guided_regex_pattern(value: Any, *, validate: bool) -> str | None:
    if not isinstance(value, str):
        if validate:
            raise OpenAIHTTPError(
                400,
                "guided_regex must be a non-empty string",
                code="invalid_request",
                param="guided_regex",
            )
        return None
    pattern = value.strip()
    if not pattern:
        if validate:
            raise OpenAIHTTPError(
                400,
                "guided_regex must be a non-empty string",
                code="invalid_request",
                param="guided_regex",
            )
        return None
    return pattern


def _guided_choice_values(value: Any, *, validate: bool) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        if validate:
            raise OpenAIHTTPError(
                400,
                "guided_choice must be a non-empty array of strings",
                code="invalid_request",
                param="guided_choice",
            )
        return ()
    choices: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            if validate:
                raise OpenAIHTTPError(
                    400,
                    f"guided_choice[{index}] must be a string",
                    code="invalid_request",
                    param=f"guided_choice[{index}]",
                )
            return ()
        choice = item.strip()
        if not choice:
            if validate:
                raise OpenAIHTTPError(
                    400,
                    f"guided_choice[{index}] must be a non-empty string",
                    code="invalid_request",
                    param=f"guided_choice[{index}]",
                )
            return ()
        choices.append(choice)
    if not choices:
        if validate:
            raise OpenAIHTTPError(
                400,
                "guided_choice must contain at least one choice",
                code="invalid_request",
                param="guided_choice",
            )
        return ()
    return tuple(choices)


def _validate_guided_patch_request(request: CompletionRequest | ChatCompletionRequest) -> None:
    active_fields = [
        field
        for field in _GUIDED_PATCH_FIELDS
        if _guided_patch_value_enabled(getattr(request, field, None))
    ]
    if len(active_fields) > 1:
        raise OpenAIHTTPError(
            400,
            "guided_patch and guided_diff cannot both be set",
            code="invalid_request",
            param="guided_patch",
        )
    if not active_fields:
        return
    field = active_fields[0]
    response_mode = _response_format_mode(request)
    if response_mode not in {None, "text"}:
        raise OpenAIHTTPError(
            400,
            f"{field} is incompatible with response_format {response_mode}",
            code="invalid_request",
            param=field,
        )
    value = getattr(request, field)
    if isinstance(value, bool):
        return
    if isinstance(value, str):
        if _normalize_guided_patch_format(value) in _GUIDED_PATCH_FORMATS:
            return
        raise OpenAIHTTPError(
            400,
            f"{field} format {value!r} is not supported",
            code="unsupported_parameter",
            param=field,
        )
    if not isinstance(value, Mapping):
        raise OpenAIHTTPError(
            400,
            f"{field} must be true, a format string, or an object",
            code="invalid_request",
            param=field,
        )
    allowed = {"type", "format", "fenced"}
    extra = sorted(str(key) for key in value.keys() if str(key) not in allowed)
    if extra:
        raise OpenAIHTTPError(
            400,
            f"{field}.{extra[0]} is not supported",
            code="unsupported_parameter",
            param=f"{field}.{extra[0]}",
        )
    requested_format = value.get("type", value.get("format", "unified_diff"))
    if _normalize_guided_patch_format(requested_format) not in _GUIDED_PATCH_FORMATS:
        raise OpenAIHTTPError(
            400,
            f"{field} format {requested_format!r} is not supported",
            code="unsupported_parameter",
            param=f"{field}.type",
        )
    _guided_patch_fenced_policy_from_value(value.get("fenced", "optional"), field=field)


def _guided_patch_fenced_policy_from_value(value: Any, *, field: str) -> str:
    if value is None:
        raise OpenAIHTTPError(
            400,
            f"{field}.fenced must be one of optional, required, or forbidden",
            code="invalid_request",
            param=f"{field}.fenced",
        )
    if isinstance(value, bool):
        return "required" if value else "forbidden"
    text = str(value).strip().lower().replace("-", "_")
    if text in {"", "optional", "allow", "allowed"}:
        return "optional"
    if text in {"required", "require", "only", "fenced"}:
        return "required"
    if text in {"forbidden", "forbid", "disallow", "disallowed", "none", "raw", "unfenced"}:
        return "forbidden"
    raise OpenAIHTTPError(
        400,
        f"{field}.fenced must be one of optional, required, or forbidden",
        code="invalid_request",
        param=f"{field}.fenced",
    )


def _normalize_guided_patch_format(value: Any) -> str:
    text = str(value or "unified_diff").strip().lower().replace("-", "_")
    if text in {"diff", "patch"}:
        return "unified_diff"
    return text


def _guided_patch_value_enabled(value: Any) -> bool:
    return value is not None and value is not False


def _guided_patch_field(guided_patch: Any | None, guided_diff: Any | None) -> str | None:
    if _guided_patch_value_enabled(guided_patch):
        return "guided_patch"
    if _guided_patch_value_enabled(guided_diff):
        return "guided_diff"
    return None


def _guided_patch_result_validation(request: CompletionRequest | ChatCompletionRequest) -> bool:
    return _guided_patch_field(
        getattr(request, "guided_patch", None),
        getattr(request, "guided_diff", None),
    ) is not None


def _guided_json_result_validation(request: CompletionRequest | ChatCompletionRequest) -> bool:
    return _guided_json_mode_from_value(getattr(request, "guided_json", None), validate=False) in {
        "json_object",
        "json_schema",
    }


def _guided_regex_result_validation(request: CompletionRequest | ChatCompletionRequest) -> bool:
    return getattr(request, "guided_regex", None) is not None


def _guided_choice_result_validation(request: CompletionRequest | ChatCompletionRequest) -> bool:
    return getattr(request, "guided_choice", None) is not None


def _guided_patch_fenced_policy(request: CompletionRequest | ChatCompletionRequest) -> str:
    field = _guided_patch_field(
        getattr(request, "guided_patch", None),
        getattr(request, "guided_diff", None),
    )
    if field is None:
        return "optional"
    value = getattr(request, field)
    if isinstance(value, Mapping):
        return _guided_patch_fenced_policy_from_value(value.get("fenced", "optional"), field=field)
    return "optional"


def _validate_tool_schema_requests(request: CompletionRequest | ChatCompletionRequest) -> None:
    if not isinstance(request, ChatCompletionRequest):
        return
    _validate_tool_choice_request(request)
    if not request.tools:
        return
    if not _strict_tool_validation_enabled(request):
        return
    for index, tool in enumerate(request.tools):
        schema = _tool_parameters_schema(tool)
        if schema is None:
            continue
        schema_error = _validate_json_schema_subset(
            schema,
            path=f"tools[{index}].function.parameters",
        )
        if schema_error is None:
            continue
        param, message = schema_error
        raise OpenAIHTTPError(400, message, code="invalid_request", param=param)


def _validate_tool_choice_request(request: ChatCompletionRequest) -> None:
    raw_choice = request.tool_choice
    if raw_choice is None:
        return
    if isinstance(raw_choice, Mapping):
        choice_type = str(raw_choice.get("type", "")).strip().lower()
        if choice_type != "function":
            raise OpenAIHTTPError(
                400,
                "tool_choice object must have type='function'",
                code="invalid_request",
                param="tool_choice.type",
            )
        function = raw_choice.get("function")
        if not isinstance(function, Mapping):
            raise OpenAIHTTPError(
                400,
                "tool_choice.function must be an object",
                code="invalid_request",
                param="tool_choice.function",
            )
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise OpenAIHTTPError(
                400,
                "tool_choice.function.name must be a non-empty string",
                code="invalid_request",
                param="tool_choice.function.name",
            )
        mode = "function"
        requested_name = name.strip()
    else:
        mode, requested_name = _tool_choice_mode(raw_choice)

    if mode in {"required", "function"} and not request.tools:
        raise OpenAIHTTPError(
            400,
            "tool_choice requires at least one tool",
            code="invalid_request",
            param="tool_choice",
        )
    if mode == "function" and requested_name not in _tool_map_by_name(request.tools or ()):
        raise OpenAIHTTPError(
            400,
            f"tool_choice function {requested_name!r} is not declared in tools",
            code="invalid_request",
            param="tool_choice.function.name",
        )


def _response_format_mode(request: CompletionRequest | ChatCompletionRequest) -> str | None:
    value = getattr(request, "response_format", None)
    if value is None:
        return None
    if isinstance(value, str):
        response_type = value.strip().lower()
    elif isinstance(value, Mapping):
        response_type = str(value.get("type", "")).strip().lower()
    else:
        raise OpenAIHTTPError(
            400,
            "response_format must be an object with type",
            code="invalid_request",
            param="response_format",
        )
    if response_type in {"", "text"}:
        return "text"
    if response_type == "json_object":
        return "json_object"
    if response_type == "json_schema":
        return "json_schema"
    raise OpenAIHTTPError(
        400,
        f"response_format type {response_type!r} is not supported",
        code="unsupported_parameter",
        param="response_format",
    )


def _response_format_json_object(request: CompletionRequest | ChatCompletionRequest) -> bool:
    return _response_format_mode(request) == "json_object"


def _response_format_result_validation(request: CompletionRequest | ChatCompletionRequest) -> bool:
    return _response_format_mode(request) in {"json_object", "json_schema"}


def _structured_result_validation(request: CompletionRequest | ChatCompletionRequest) -> bool:
    return (
        _response_format_result_validation(request)
        or _guided_json_result_validation(request)
        or _guided_regex_result_validation(request)
        or _guided_choice_result_validation(request)
        or _guided_patch_result_validation(request)
    )


def _response_format_json_schema(
    request: CompletionRequest | ChatCompletionRequest,
) -> Mapping[str, Any] | None:
    return _response_format_json_schema_from_value(getattr(request, "response_format", None))


def _response_format_json_schema_from_value(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    if str(value.get("type", "")).strip().lower() != "json_schema":
        return None
    json_schema = value.get("json_schema")
    if not isinstance(json_schema, Mapping):
        return None
    schema = json_schema.get("schema")
    return schema if isinstance(schema, Mapping) else None


def _request_extra_keys(request: CompletionRequest | ChatCompletionRequest) -> set[str]:
    extra = getattr(request, "model_extra", None)
    if isinstance(extra, Mapping):
        return {str(key) for key in extra}
    extra = getattr(request, "__pydantic_extra__", None)
    if isinstance(extra, Mapping):
        return {str(key) for key in extra}
    fields = getattr(request, "__fields__", None)
    if isinstance(fields, Mapping):
        return {str(key) for key in vars(request) if str(key) not in fields}
    return set()


def _expand_prompts_for_n(prompts: Sequence[PromptInput], n: int) -> tuple[PromptInput, ...]:
    return tuple(prompt for prompt in prompts for _ in range(int(n)))


def _choice_request_id(response_id: str, prompt_index: int, choice_index: int) -> str:
    return f"{response_id}:prompt-{int(prompt_index)}:choice-{int(choice_index)}"


def _row_seeds_for_request(seed: int | None, row_count: int) -> tuple[int, ...]:
    return tuple(derive_row_seed(seed, index) for index in range(int(row_count)))


def _message_content_text(content: str | list[Any] | None, message_index: int) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise OpenAIHTTPError(
            400,
            "message content must be text or text content parts",
            param=f"messages[{message_index}].content",
        )
    text_parts: list[str] = []
    for part_index, part in enumerate(content):
        if isinstance(part, str):
            text_parts.append(part)
            continue
        if not isinstance(part, dict):
            raise OpenAIHTTPError(
                400,
                "message content parts must be text objects",
                param=f"messages[{message_index}].content[{part_index}]",
            )
        part_type = part.get("type", "text")
        if part_type != "text":
            raise OpenAIHTTPError(
                400,
                f"unsupported content part type {part_type!r}; text only for now",
                code="unsupported_content_type",
                param=f"messages[{message_index}].content[{part_index}]",
            )
        text = part.get("text", "")
        if not isinstance(text, str):
            raise OpenAIHTTPError(
                400,
                "text content part must contain a string text field",
                param=f"messages[{message_index}].content[{part_index}].text",
            )
        text_parts.append(text)
    return "".join(text_parts)


def _apply_stop(text: str, stop: str | list[str] | None) -> tuple[str, str]:
    stops = _stop_strings(stop)
    if not stops:
        return text, "stop"
    earliest: int | None = None
    for item in stops:
        if not item:
            continue
        index = text.find(item)
        if index >= 0 and (earliest is None or index < earliest):
            earliest = index
    if earliest is None:
        return text, "stop"
    return text[:earliest], "stop"


def _finish_reason_for_output(
    detail: GenerationOutput | None,
    fallback: str,
    *,
    server_stop: bool = False,
    tool_calls: bool = False,
) -> str:
    if tool_calls:
        return "tool_calls"
    if server_stop:
        return "stop"
    finish = None if detail is None else detail.finish_details
    if finish is None:
        return str(fallback)
    reason = finish.reason.strip().lower()
    if reason in {
        "length",
        "max_length",
        "max_tokens",
        "token_budget_exhausted",
        "budget_exhausted",
        "thinking_budget_exhausted",
    }:
        return "length"
    if reason in {"tool_call", "tool_calls"}:
        return "tool_calls"
    if reason == "content_filter":
        return "content_filter"
    return str(fallback)


def _strict_tool_failure_finish_reason(
    detail: GenerationOutput | None,
    fallback: str,
    *,
    server_stop: bool = False,
) -> str:
    finish_reason = _finish_reason_for_output(detail, fallback, server_stop=server_stop)
    return "length" if finish_reason == "length" else "stop"


def _finish_details_payload(
    detail: GenerationOutput | None,
    finish_reason: str,
    *,
    reason_override: str | None = None,
    cache_action: str | None = None,
) -> dict[str, Any]:
    finish = None if detail is None else detail.finish_details
    if finish is None:
        finish = FinishDetails(reason=finish_reason)
    payload = finish.to_json_dict(reason=reason_override)
    if cache_action is not None:
        payload.setdefault("cache_action", str(cache_action))
    return payload


def _chat_finish_details_payload(
    detail: GenerationOutput | None,
    finish_reason: str,
    text: str,
    *,
    reason_override: str | None = None,
    cache_action: str | None = None,
    parsed: _ParsedChatOutput | None = None,
    token_counter: Any | None = None,
) -> dict[str, Any]:
    payload = _finish_details_payload(
        detail,
        finish_reason,
        reason_override=reason_override,
        cache_action=cache_action,
    )
    _enrich_chat_tool_finish_details(payload, parsed=parsed, token_counter=token_counter)
    if _is_length_finish(finish_reason, payload):
        payload.setdefault("phase", _classify_chat_length_phase(text))
        payload.setdefault("continuation_eligible", False)
    return payload


def _enrich_chat_tool_finish_details(
    payload: dict[str, Any],
    *,
    parsed: _ParsedChatOutput | None,
    token_counter: Any | None,
) -> None:
    if parsed is None or not parsed.tool_calls or not callable(token_counter):
        return
    split = _split_reasoning(
        parsed.text,
        initially_open=parsed.reasoning_initially_open,
    )
    reasoning_tokens = _safe_count(token_counter, split.reasoning_content)
    answer_tokens = _safe_count(token_counter, split.content)
    tool_call_tokens = sum(
        _safe_count(token_counter, call.raw_text or _tool_call_count_text(call))
        for call in parsed.tool_calls
    )
    payload.setdefault("phase", "tool_call")
    if reasoning_tokens:
        payload.setdefault("reasoning_tokens", reasoning_tokens)
    if answer_tokens:
        payload.setdefault("answer_tokens", answer_tokens)
    if tool_call_tokens:
        payload.setdefault("tool_call_tokens", tool_call_tokens)


def _tool_call_count_text(call: _ParsedToolCall) -> str:
    return f'<tool_call>{{"name":{json.dumps(call.name)},"arguments":{call.arguments}}}</tool_call>'


def _response_format_failure_reason(
    request: CompletionRequest | ChatCompletionRequest,
    text: str,
    finish_reason: str,
) -> str | None:
    mode = _response_format_mode(request)
    if mode not in {"json_object", "json_schema"}:
        return None
    if str(finish_reason).strip().lower() == "length":
        return None
    try:
        value = json.loads(str(text).strip())
    except Exception:
        return "schema_violation"
    if mode == "json_object":
        return None if isinstance(value, dict) else "schema_violation"
    schema = _response_format_json_schema(request)
    if schema is None:
        return "schema_violation"
    return None if _validate_json_schema_value(value, schema, path="$") is None else "schema_violation"


def _structured_output_failure_reason(
    request: CompletionRequest | ChatCompletionRequest,
    text: str,
    finish_reason: str,
) -> str | None:
    response_format_failure = _response_format_failure_reason(request, text, finish_reason)
    if response_format_failure is not None:
        return response_format_failure
    guided_json_failure = _guided_json_failure_reason(request, text, finish_reason)
    if guided_json_failure is not None:
        return guided_json_failure
    guided_regex_failure = _guided_regex_failure_reason(request, text, finish_reason)
    if guided_regex_failure is not None:
        return guided_regex_failure
    guided_choice_failure = _guided_choice_failure_reason(request, text, finish_reason)
    if guided_choice_failure is not None:
        return guided_choice_failure
    return _guided_patch_failure_reason(request, text, finish_reason)


def _structured_length_failure_reason(
    request: CompletionRequest | ChatCompletionRequest,
    text: str,
    finish_reason: str,
) -> str | None:
    if str(finish_reason).strip().lower() != "length":
        return None
    if not _structured_json_object_prefix_validation(request, text):
        return None
    state = JsonObjectConstraintState().observe_text(text)
    return "schema_violation" if state.invalid else None


def _structured_json_object_prefix_validation(
    request: CompletionRequest | ChatCompletionRequest,
    text: str,
) -> bool:
    del text  # The constraint state decides whether the observed root prefix is valid.
    response_mode = _response_format_mode(request)
    guided_mode = _guided_json_mode_from_value(getattr(request, "guided_json", None), validate=False)
    if response_mode == "json_object" or guided_mode == "json_object":
        return True
    if response_mode == "json_schema":
        return _json_schema_root_object(_response_format_json_schema(request))
    if guided_mode == "json_schema":
        return _json_schema_root_object(_guided_json_schema(request))
    return False


def _json_object_close_forcing(request: CompletionRequest | ChatCompletionRequest) -> bool:
    response_mode = _response_format_mode(request)
    guided_mode = _guided_json_mode_from_value(getattr(request, "guided_json", None), validate=False)
    if response_mode == "json_object" or guided_mode == "json_object":
        return True
    if response_mode == "json_schema" and _json_schema_root_object(_response_format_json_schema(request)):
        return True
    return bool(guided_mode == "json_schema" and _json_schema_root_object(_guided_json_schema(request)))


def _json_schema_root_object(schema: Mapping[str, Any] | None) -> bool:
    if not isinstance(schema, Mapping):
        return False
    schema_type = schema.get("type")
    if schema_type == "object":
        return True
    return any(key in schema for key in ("properties", "required", "additionalProperties"))


def _guided_json_failure_reason(
    request: CompletionRequest | ChatCompletionRequest,
    text: str,
    finish_reason: str,
) -> str | None:
    mode = _guided_json_mode_from_value(getattr(request, "guided_json", None), validate=False)
    if mode not in {"json_object", "json_schema"}:
        return None
    if str(finish_reason).strip().lower() == "length":
        return None
    try:
        value = json.loads(str(text).strip())
    except Exception:
        return "schema_violation"
    if mode == "json_object":
        return None if isinstance(value, dict) else "schema_violation"
    schema = _guided_json_schema(request)
    if schema is None:
        return "schema_violation"
    return None if _validate_json_schema_value(value, schema, path="$") is None else "schema_violation"


def _guided_regex_failure_reason(
    request: CompletionRequest | ChatCompletionRequest,
    text: str,
    finish_reason: str,
) -> str | None:
    if not _guided_regex_result_validation(request):
        return None
    if str(finish_reason).strip().lower() == "length":
        return None
    pattern = _guided_regex_pattern(getattr(request, "guided_regex", None), validate=False)
    if pattern is None:
        return "schema_violation"
    try:
        valid = re.fullmatch(pattern, str(text).strip()) is not None
    except re.error:
        valid = False
    return None if valid else "schema_violation"


def _guided_choice_failure_reason(
    request: CompletionRequest | ChatCompletionRequest,
    text: str,
    finish_reason: str,
) -> str | None:
    if not _guided_choice_result_validation(request):
        return None
    if str(finish_reason).strip().lower() == "length":
        return None
    choices = _guided_choice_values(getattr(request, "guided_choice", ()), validate=False)
    return None if str(text).strip() in choices else "schema_violation"


def _guided_patch_failure_reason(
    request: CompletionRequest | ChatCompletionRequest,
    text: str,
    finish_reason: str,
) -> str | None:
    if not _guided_patch_result_validation(request):
        return None
    if str(finish_reason).strip().lower() == "length":
        return None
    valid = _is_valid_guided_patch_text(
        text,
        fenced_policy=_guided_patch_fenced_policy(request),
    )
    return None if valid else "schema_violation"


def _is_valid_guided_patch_text(text: str, *, fenced_policy: str = "optional") -> bool:
    extracted = _extract_guided_patch_body(str(text))
    if extracted is None:
        return False
    patch, is_fenced = extracted
    if fenced_policy == "required" and not is_fenced:
        return False
    if fenced_policy == "forbidden" and is_fenced:
        return False
    return _is_valid_unified_diff(patch)


def _extract_guided_patch_body(text: str) -> tuple[str, bool] | None:
    stripped = str(text).strip()
    if not stripped:
        return None
    match = _PATCH_FENCE_RE.match(stripped)
    if match is None:
        return stripped, False
    label = match.group("label").strip().lower()
    if label and label not in {"diff", "patch"}:
        return None
    return match.group("body").strip(), True


def _is_valid_unified_diff(text: str) -> bool:
    lines = str(text).strip().splitlines()
    if not lines:
        return False
    index = 0
    file_count = 0
    hunk_count = 0
    while index < len(lines):
        while index < len(lines) and _is_unified_diff_metadata_line(lines[index]):
            index += 1
        if index < len(lines) and lines[index].startswith(_UNIFIED_DIFF_BINARY_PREFIXES):
            return False
        if index >= len(lines) or not lines[index].startswith("--- "):
            return False
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            return False
        file_count += 1
        index += 1
        file_hunks = 0
        while index < len(lines) and _UNIFIED_DIFF_HUNK_RE.match(lines[index]):
            file_hunks += 1
            hunk_count += 1
            index += 1
            body_lines = 0
            while index < len(lines):
                line = lines[index]
                if _UNIFIED_DIFF_HUNK_RE.match(line):
                    break
                if line.startswith("--- ") or _is_unified_diff_metadata_line(line):
                    break
                if not _is_unified_diff_body_line(line):
                    return False
                body_lines += 1
                index += 1
            if body_lines == 0:
                return False
        if file_hunks == 0:
            return False
    return file_count > 0 and hunk_count > 0


def _is_unified_diff_metadata_line(line: str) -> bool:
    return line.startswith(_UNIFIED_DIFF_METADATA_PREFIXES)


def _is_unified_diff_body_line(line: str) -> bool:
    return (
        bool(line)
        and line[0] in {" ", "+", "-"}
        or line.startswith("\\ No newline at end of file")
    )


def _is_json_object_text(text: str) -> bool:
    try:
        value = json.loads(str(text).strip())
    except Exception:
        return False
    return isinstance(value, dict)


def _chat_response_format_text(message: Mapping[str, Any]) -> str:
    content = message.get("content", "")
    return content if isinstance(content, str) else ""


def _chat_response_format_text_from_parsed(parsed: _ParsedChatOutput) -> str:
    if parsed.tool_calls:
        return ""
    return _split_reasoning(
        parsed.text,
        initially_open=parsed.reasoning_initially_open,
    ).content


def _is_length_finish_payload(payload: Mapping[str, Any]) -> bool:
    reason = str(payload.get("reason", "")).strip().lower()
    return reason in {
        "length",
        "max_length",
        "max_tokens",
        "token_budget_exhausted",
        "budget_exhausted",
        "thinking_budget_exhausted",
    }


def _is_length_finish(finish_reason: str, payload: Mapping[str, Any]) -> bool:
    return str(finish_reason).strip().lower() == "length" or _is_length_finish_payload(payload)


def _stop_strings(stop: str | list[str] | None) -> tuple[str, ...]:
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,)
    return tuple(str(item) for item in stop)


def _stop_tokens_from_stop(
    stop: str | list[str] | None,
    engine: Any,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Lower tokenizable OpenAI stop strings to token stop metadata."""

    stops = tuple(item for item in _stop_strings(stop) if item)
    tokenizer = getattr(engine, "tokenize", None)
    if not stops or not callable(tokenizer):
        return (), ()
    token_ids: list[int] = []
    sequences: list[tuple[int, ...]] = []
    for item in stops:
        try:
            ids = tuple(int(token) for token in tokenizer(item))
        except (KeyError, NotImplementedError, TypeError, ValueError):
            continue
        if len(ids) == 1:
            if ids[0] not in token_ids:
                token_ids.append(ids[0])
        elif len(ids) > 1 and ids not in sequences:
            sequences.append(ids)
    return tuple(token_ids), tuple(sequences)


def _sampling_key(
    sampling: SamplingParams,
    *,
    include_cancellation_token: bool = True,
) -> tuple[Any, ...]:
    return (
        int(sampling.max_tokens),
        float(sampling.temperature),
        float(sampling.top_p),
        int(sampling.top_k),
        float(sampling.min_p),
        float(sampling.repetition_penalty),
        float(sampling.presence_penalty),
        float(sampling.frequency_penalty),
        tuple((int(token), float(bias)) for token, bias in sampling.logit_bias),
        tuple(int(token) for token in sampling.suppress_token_ids),
        int(sampling.min_tokens),
        None if sampling.eos_token_id is None else int(sampling.eos_token_id),
        tuple(int(token) for token in sampling.stop_token_ids),
        tuple(tuple(int(token) for token in row) for row in sampling.stop_token_sequences),
        bool(sampling.ignore_eos),
        str(sampling.kv_storage),
        str(sampling.kv_scale_dtype),
        str(sampling.kv_scale_granularity),
        None if sampling.seed is None else int(sampling.seed),
        tuple(int(seed) for seed in sampling.row_seeds),
        None if sampling.deadline_at is None else float(sampling.deadline_at),
        (
            None
            if not include_cancellation_token or sampling.cancellation_token is None
            else id(sampling.cancellation_token)
        ),
        (
            None
            if sampling.speculative_mtp_static_eligibility is None
            else sampling.speculative_mtp_static_eligibility.fingerprint
        ),
        bool(sampling.logprobs),
        int(sampling.top_logprobs),
    )


def _completion_logprobs(detail: GenerationOutput, text: str, *, echo_text: str = "") -> dict[str, Any]:
    tokens = list(_trim_token_logprobs(detail.token_logprobs, text))
    response_tokens: list[str] = []
    token_logprobs: list[float | None] = []
    top_logprobs: list[dict[str, float] | None] = []
    offsets: list[int] = []
    omitted: list[dict[str, Any]] = []
    cursor = 0
    if echo_text:
        response_tokens.append(echo_text)
        token_logprobs.append(None)
        top_logprobs.append(None)
        offsets.append(0)
        omitted.append(_prompt_logprob_omission(echo_text, 0))
        cursor = len(echo_text)
    for token in tokens:
        response_index = len(response_tokens)
        response_tokens.append(token.token_text)
        token_logprobs.append(token.logprob)
        top_logprobs.append(_completion_top_logprobs(token))
        offsets.append(cursor)
        if token.logprob is None:
            omitted.append(_logprob_omission(token, response_index))
        cursor += len(token.token_text)
    payload: dict[str, Any] = {
        "tokens": response_tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": offsets,
    }
    if omitted:
        payload["hipengine"] = {"omitted_token_logprobs": omitted}
    return payload


def _completion_stream_logprobs(stream_chunk: GenerationStreamChunk) -> dict[str, Any]:
    _validate_stream_logprob_chunk(stream_chunk)
    return _completion_logprobs(
        GenerationOutput(text=stream_chunk.text, token_logprobs=stream_chunk.token_logprobs),
        stream_chunk.text,
    )


def _completion_top_logprobs(token: TokenLogprob) -> dict[str, float] | None:
    if not token.top_logprobs:
        return None
    return {text: float(logprob) for _token_id, text, logprob in token.top_logprobs}


def _chat_logprobs(detail: GenerationOutput, text: str) -> dict[str, Any]:
    tokens = _trim_token_logprobs(detail.token_logprobs, text)
    content = [
        {
            "token": token.token_text,
            "logprob": token.logprob,
            "bytes": None,
            "top_logprobs": [
                {"token": top_text, "logprob": float(top_logprob), "bytes": None}
                for _top_id, top_text, top_logprob in token.top_logprobs
            ],
        }
        for token in tokens
    ]
    omitted = [_logprob_omission(token, index) for index, token in enumerate(tokens) if token.logprob is None]
    payload: dict[str, Any] = {
        "content": content,
        "refusal": None,
    }
    if omitted:
        payload["hipengine"] = {"omitted_token_logprobs": omitted}
    return payload


def _chat_visible_content_logprobs(
    detail: GenerationOutput,
    text: str,
    *,
    initially_open: bool = False,
) -> dict[str, Any]:
    split = _split_reasoning(text, initially_open=initially_open)
    if split.content == text:
        return _chat_logprobs(detail, text)
    tokens = _visible_content_token_logprobs(detail.token_logprobs, text, split.content)
    return _chat_logprobs(GenerationOutput(text=split.content, token_logprobs=tokens), split.content)


def _visible_content_token_logprobs(
    tokens: Sequence[TokenLogprob],
    raw_text: str,
    visible_text: str,
) -> tuple[TokenLogprob, ...]:
    if not tokens or not visible_text:
        return ()
    spans = tuple(
        (start, end)
        for field, start, end in _reasoning_text_segments(raw_text)
        if field == "content" and start < end
    )
    selected = _token_logprobs_for_spans(tokens, spans)
    if "".join(token.token_text for token in selected) == visible_text:
        return selected
    return ()


def _reasoning_text_segments(text: str) -> tuple[tuple[str, int, int], ...]:
    segments: list[tuple[str, int, int]] = []
    cursor = 0
    in_reasoning = False
    while cursor < len(text):
        tag = _REASONING_CLOSE_TAG if in_reasoning else _REASONING_OPEN_TAG
        index = text.find(tag, cursor)
        if index < 0:
            if cursor < len(text):
                field = "reasoning_content" if in_reasoning else "content"
                segments.append((field, cursor, len(text)))
            break
        if index > cursor:
            field = "reasoning_content" if in_reasoning else "content"
            segments.append((field, cursor, index))
        cursor = index + len(tag)
        in_reasoning = not in_reasoning
    return tuple(segments)


def _token_logprobs_for_spans(
    tokens: Sequence[TokenLogprob],
    spans: Sequence[tuple[int, int]],
) -> tuple[TokenLogprob, ...]:
    if not tokens or not spans:
        return ()
    selected: list[TokenLogprob] = []
    span_index = 0
    cursor = 0
    for token in tokens:
        token_start = cursor
        token_end = token_start + len(token.token_text)
        cursor = token_end
        while span_index < len(spans) and spans[span_index][1] <= token_start:
            span_index += 1
        if span_index >= len(spans):
            continue
        span_start, span_end = spans[span_index]
        if token_start >= span_start and token_end <= span_end:
            selected.append(token)
            continue
        if token_end <= span_start or token_start >= span_end:
            continue
        return ()
    return tuple(selected)


def _logprob_omission(token: TokenLogprob, index: int) -> dict[str, Any]:
    return {
        "index": int(index),
        "token": token.token_text,
        "token_id": int(token.token_id),
        "reason": _LOGPROB_OMISSION_REASON,
    }


def _prompt_logprob_omission(token_text: str, index: int) -> dict[str, Any]:
    return {
        "index": int(index),
        "token": str(token_text),
        "token_id": None,
        "reason": _PROMPT_LOGPROB_OMISSION_REASON,
    }


def _chat_stream_logprobs(stream_chunk: GenerationStreamChunk, text: str) -> dict[str, Any]:
    tokens = _stream_token_logprobs_for_text(stream_chunk, text)
    if text and not tokens:
        raise _unsupported_stream_logprobs_error()
    return _chat_logprobs(GenerationOutput(text=text, token_logprobs=tokens), text)


def _chat_reasoning_stream_logprobs(
    stream_chunk: GenerationStreamChunk,
    public_text: str,
) -> dict[str, Any]:
    tokens = _stream_token_logprobs_for_text(stream_chunk, stream_chunk.text)
    if public_text and not tokens:
        raise _unsupported_stream_logprobs_error()
    content = [
        {
            "token_id": int(token.token_id),
            "token": token.token_text,
            "logprob": token.logprob,
            "bytes": None,
            "top_logprobs": [
                {
                    "token_id": int(top_id),
                    "token": top_text,
                    "logprob": float(top_logprob),
                    "bytes": None,
                }
                for top_id, top_text, top_logprob in token.top_logprobs
            ],
        }
        for token in tokens
    ]
    payload: dict[str, Any] = {
        "content": content,
        "public_text": str(public_text),
        "refusal": None,
    }
    omitted = [_logprob_omission(token, index) for index, token in enumerate(tokens) if token.logprob is None]
    if omitted:
        payload["omitted_token_logprobs"] = omitted
    return payload


def _validate_stream_logprob_chunk(stream_chunk: GenerationStreamChunk) -> None:
    if stream_chunk.text and not stream_chunk.token_logprobs:
        raise _unsupported_stream_logprobs_error()


def _live_splitter_stream_chunk_for_delta(
    source_chunks: Sequence[_LiveSourceChunk],
    fallback_chunk: GenerationStreamChunk | None,
    text: str,
    *,
    source_start: int,
    source_end: int,
    phase: str,
) -> GenerationStreamChunk | None:
    source_chunk = _live_splitter_span_stream_chunk(
        source_chunks,
        source_start=source_start,
        source_end=source_end,
        text=text,
        phase=phase,
    )
    if source_chunk is not None:
        return source_chunk
    if fallback_chunk is None:
        return None
    return _stream_chunk_with_phase(fallback_chunk, phase)


def _trim_live_splitter_source_chunks(
    source_chunks: list[_LiveSourceChunk],
    *,
    min_source_start: int,
) -> None:
    while source_chunks and source_chunks[0].source_end <= min_source_start:
        del source_chunks[0]


def _live_splitter_span_stream_chunk(
    source_chunks: Sequence[_LiveSourceChunk],
    *,
    source_start: int,
    source_end: int,
    text: str,
    phase: str,
) -> GenerationStreamChunk | None:
    if not text or not source_chunks or source_end <= source_start:
        return None
    pieces: list[str] = []
    token_logprobs: list[TokenLogprob] = []
    tail_chunk: GenerationStreamChunk | None = None
    for source_chunk in source_chunks:
        overlap_start = max(source_start, source_chunk.source_start)
        overlap_end = min(source_end, source_chunk.source_end)
        if overlap_start >= overlap_end:
            continue
        stream_chunk = source_chunk.stream_chunk
        local_start = overlap_start - source_chunk.source_start
        local_end = overlap_end - source_chunk.source_start
        piece = stream_chunk.text[local_start:local_end]
        tokens = _stream_token_logprobs_for_text_span(stream_chunk, local_start, local_end)
        if piece and stream_chunk.token_logprobs and not tokens:
            return None
        pieces.append(piece)
        token_logprobs.extend(tokens)
        tail_chunk = stream_chunk
    if tail_chunk is None:
        return None
    if "".join(pieces) != text:
        return None
    chunk = GenerationStreamChunk(
        text=text,
        token_logprobs=tuple(token_logprobs),
        finish_details=tail_chunk.finish_details,
        telemetry=tail_chunk.telemetry,
        generated_token_ids=tail_chunk.generated_token_ids,
    )
    return _stream_chunk_with_phase(chunk, phase)


def _unsupported_stream_logprobs_error() -> OpenAIHTTPError:
    return OpenAIHTTPError(
        501,
        "streaming logprobs are not supported by this backend response path",
        error_type="server_error",
        code="unsupported_feature",
        param="logprobs",
    )


def _stream_token_logprobs_for_text(
    stream_chunk: GenerationStreamChunk,
    text: str,
) -> tuple[TokenLogprob, ...]:
    if not stream_chunk.token_logprobs or not text:
        return ()
    source_text = stream_chunk.text
    if source_text == text:
        return _trim_token_logprobs(stream_chunk.token_logprobs, text)
    start = source_text.find(text)
    if start < 0:
        return ()
    end = start + len(text)
    return _stream_token_logprobs_for_text_span(stream_chunk, start, end)


def _stream_token_logprobs_for_text_span(
    stream_chunk: GenerationStreamChunk,
    start: int,
    end: int,
) -> tuple[TokenLogprob, ...]:
    if (
        not stream_chunk.token_logprobs
        or start < 0
        or end < start
        or end > len(stream_chunk.text)
        or start == end
    ):
        return ()
    cursor = 0
    selected: list[TokenLogprob] = []
    for token in stream_chunk.token_logprobs:
        token_start = cursor
        token_end = token_start + len(token.token_text)
        cursor = token_end
        if token_end <= start:
            continue
        if token_start < start or token_end > end:
            return ()
        selected.append(token)
        if token_end == end:
            break
    if "".join(token.token_text for token in selected) != stream_chunk.text[start:end]:
        return ()
    return tuple(selected)


def _trim_token_logprobs(tokens: Sequence[TokenLogprob], text: str) -> tuple[TokenLogprob, ...]:
    if not tokens or not text:
        return ()
    selected: list[TokenLogprob] = []
    cursor = 0
    for token in tokens:
        next_cursor = cursor + len(token.token_text)
        if next_cursor > len(text):
            break
        selected.append(token)
        cursor = next_cursor
        if cursor >= len(text):
            break
    return tuple(selected)


def _usage(
    engine: Any,
    prompts: Sequence[PromptInput],
    outputs: Sequence[str],
    *,
    details: Sequence[GenerationOutput] | None = None,
    prefer_telemetry_counts: bool = False,
) -> dict[str, int]:
    counter = getattr(engine, "count_tokens", None)
    prompt_tokens = sum(_prompt_token_count(engine, prompt) for prompt in prompts)
    exact_rows = _exact_generated_token_rows(details, expected_rows=len(outputs))
    telemetry_counts = (
        _telemetry_generated_token_counts(details, expected_rows=len(outputs))
        if prefer_telemetry_counts
        else None
    )
    if exact_rows is not None:
        completion_tokens = sum(len(row) for row in exact_rows)
    elif telemetry_counts is not None:
        completion_tokens = sum(telemetry_counts)
    elif callable(counter):
        completion_tokens = sum(_safe_count(counter, text) for text in outputs)
    else:
        # Compatibility placeholder for legacy generators without token IDs.
        completion_tokens = 0
    reasoning_tokens = _finish_reasoning_token_total(details)
    usage: dict[str, int] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    completion_details: dict[str, int] = {}
    if reasoning_tokens:
        usage["reasoning_tokens"] = reasoning_tokens
        completion_details["reasoning_tokens"] = reasoning_tokens
    mtp_counts = _mtp_accepted_rejected_counts(details)
    if mtp_counts is not None:
        accepted, rejected = mtp_counts
        completion_details["accepted_prediction_tokens"] = accepted
        completion_details["rejected_prediction_tokens"] = rejected
    if completion_details:
        usage["completion_tokens_details"] = completion_details
    return usage


def _mtp_telemetry_parts(detail: GenerationOutput) -> tuple[Mapping[str, Any] | None, Any, bool]:
    """Return timing, decode state, and timing ownership from one output."""

    telemetry = getattr(detail, "telemetry", None)
    if isinstance(telemetry, Mapping):
        timing = telemetry.get("timing")
        decode_state = telemetry.get("decode_state")
        timing_owner = telemetry.get("timing_owner")
    else:
        timing = None if telemetry is None else getattr(telemetry, "timing", None)
        decode_state = None if telemetry is None else getattr(telemetry, "decode_state", None)
        timing_owner = None if telemetry is None else getattr(telemetry, "timing_owner", None)
    return (timing if isinstance(timing, Mapping) else None), decode_state, timing_owner is not False


def _mtp_accepted_rejected_counts(
    details: Sequence[GenerationOutput] | None,
) -> tuple[int, int] | None:
    """Sum owned MTP draft acceptance from generation telemetry.

    Mirrors vLLM's ``usage.completion_tokens_details.accepted_prediction_tokens``
    and ``rejected_prediction_tokens`` fields while honoring batch timing
    ownership so copied non-owner payloads are not counted more than once.
    Returns ``None`` when no owned payload carries MTP telemetry.
    """

    accepted = 0
    generated = 0
    found = False
    for detail in (details or ()):
        timing, _, timing_owner = _mtp_telemetry_parts(detail)
        if timing is None or not timing_owner:
            continue
        row_accepted = timing.get("mtp_accepted_draft_tokens")
        row_generated = timing.get("mtp_generated_draft_tokens")
        if row_accepted is None and row_generated is None:
            continue
        found = True
        accepted += max(0, int(row_accepted or 0))
        generated += max(0, int(row_generated or 0))
    if not found:
        return None
    return accepted, max(0, generated - accepted)


def _mtp_response_summary(
    route: str | None,
    details: Sequence[GenerationOutput] | None,
    *,
    route_decision: Mapping[str, Any] | None = None,
    thinking_policy: str | None = None,
) -> dict[str, Any]:
    """Compact per-request effective MTP state for the hipengine extension.

    ``route`` is the realized generation route from the batch's generation
    shape (``default`` / ``speculative_mtp`` / ``speculative``); MTP usage and
    draft counts are derived from each row's backend telemetry.
    """

    accepted = 0
    generated = 0
    cycles = 0
    used = False
    selected_route = "unknown" if route is None else str(route)
    for detail in (details or ()):
        timing, decode_state, timing_owner = _mtp_telemetry_parts(detail)
        if isinstance(decode_state, Mapping):
            execution_path = decode_state.get("execution_path")
        else:
            execution_path = None if decode_state is None else getattr(decode_state, "execution_path", None)
        if execution_path and "mtp" in str(execution_path).lower():
            used = True
        if timing is None or not timing_owner:
            continue
        row_accepted = timing.get("mtp_accepted_draft_tokens")
        row_generated = timing.get("mtp_generated_draft_tokens")
        row_cycles = timing.get("mtp_cycles_count")
        if any(
            value is not None
            for value in (row_accepted, row_generated, row_cycles)
        ):
            used = True
        if row_accepted is not None:
            accepted += max(0, int(row_accepted))
        if row_generated is not None:
            generated += max(0, int(row_generated))
        if row_cycles is not None:
            cycles += max(0, int(row_cycles))
    used = used or generated > 0
    backend_k0_fallback = (
        selected_route == _SPECULATIVE_MTP_BATCH_ROUTE and not used
    )
    summary = {
        "effective_route": (
            _SPECULATIVE_MTP_BATCH_ROUTE
            if used
            else _SPECULATIVE_MTP_DEFAULT_ROUTE
            if backend_k0_fallback
            else selected_route
        ),
        "used": bool(used),
        "draft_tokens": generated,
        "accepted_draft_tokens": accepted,
        "rejected_draft_tokens": max(0, generated - accepted),
        "acceptance_rate": (float(accepted) / float(generated)) if generated else None,
        "draft_cycles": cycles,
    }
    if backend_k0_fallback:
        summary["selected_route"] = selected_route
        summary["decision_reason"] = "backend_k0_fallback"
    if isinstance(route_decision, Mapping):
        requested_route = route_decision.get("requested_route")
        decision_reason = route_decision.get("reason")
        if requested_route is not None:
            summary["requested_route"] = str(requested_route)
        if decision_reason is not None:
            summary[
                "selection_reason" if backend_k0_fallback else "decision_reason"
            ] = str(decision_reason)
    if used and thinking_policy is not None:
        policy = str(thinking_policy)
        summary["thinking_policy"] = policy
        summary["thinking_controls"] = (
            "prompt_hint_only" if policy == "hint" else "hard_controls_preserved"
        )
    return summary


def _stream_speculative_mtp_metadata(
    route: str,
    detail: GenerationOutput | None,
    *,
    route_decision: Mapping[str, Any] | None,
    thinking_policy: str,
) -> dict[str, Any]:
    if str(route) == _SPECULATIVE_MTP_DEFAULT_ROUTE and route_decision is None:
        return {}
    generation_shape: dict[str, Any] = {"route": str(route)}
    if route_decision is not None:
        generation_shape["route_decision"] = deepcopy(dict(route_decision))
    return {
        "generation_shape": generation_shape,
        "speculative_mtp": _mtp_response_summary(
            route,
            None if detail is None else (detail,),
            route_decision=route_decision,
            thinking_policy=thinking_policy,
        ),
    }


def _finish_reasoning_token_total(details: Sequence[GenerationOutput] | None) -> int:
    """Sum reasoning tokens reported by each completed generation's finish details."""

    total = 0
    for detail in (details or ()):
        finish = getattr(detail, "finish_details", None)
        if finish is not None:
            total += int(getattr(finish, "reasoning_tokens", 0) or 0)
    return total


def _exact_prompt_token_accounting(prompts: Sequence[PromptInput]) -> dict[str, Any] | None:
    if not prompts or any(
        isinstance(prompt, (str, PreparedPromptInput)) for prompt in prompts
    ):
        return None
    rows = tuple(tuple(int(token) for token in prompt) for prompt in prompts)
    return {
        "schema_version": 1,
        "input_type": "token_ids",
        "prompt_token_ids_sha256": [token_ids_sha256(row) for row in rows],
        "prompt_tokens": [len(row) for row in rows],
        "total_prompt_tokens": sum(len(row) for row in rows),
    }


def _exact_generated_token_rows(
    details: Sequence[GenerationOutput] | None,
    *,
    expected_rows: int,
) -> tuple[tuple[int, ...], ...] | None:
    if details is None or len(details) != int(expected_rows):
        return None
    rows: list[tuple[int, ...]] = []
    for detail in details:
        if detail.generated_token_ids is None:
            return None
        rows.append(tuple(detail.generated_token_ids))
    return tuple(rows)


def _telemetry_generated_token_count(telemetry: Any) -> int | None:
    if telemetry is None:
        return None
    if isinstance(telemetry, Mapping):
        decode_state = telemetry.get("decode_state")
        if not isinstance(decode_state, Mapping):
            return None
        value = decode_state.get("generated_tokens")
    else:
        decode_state = getattr(telemetry, "decode_state", None)
        value = getattr(decode_state, "generated_tokens", None)
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _telemetry_generated_token_counts(
    details: Sequence[GenerationOutput] | None,
    *,
    expected_rows: int,
) -> tuple[int, ...] | None:
    if details is None or len(details) != int(expected_rows):
        return None
    counts: list[int] = []
    for detail in details:
        count = _telemetry_generated_token_count(detail.telemetry)
        if count is None:
            return None
        counts.append(count)
    return tuple(counts)


def _exact_generated_token_accounting(
    engine: Any,
    details: Sequence[GenerationOutput],
    visible_outputs: Sequence[str],
) -> dict[str, Any] | None:
    rows = _exact_generated_token_rows(details, expected_rows=len(visible_outputs))
    if rows is None:
        return None
    payload: dict[str, Any] = {
        "choice_generated_token_ids": [list(row) for row in rows],
        "choice_generated_tokens": [len(row) for row in rows],
        "total_generated_tokens": sum(len(row) for row in rows),
    }
    counter = getattr(engine, "count_tokens", None)
    if callable(counter):
        payload["retokenized_visible_tokens"] = sum(
            _safe_count(counter, text) for text in visible_outputs
        )
    return payload


def _safe_count(counter: Any, text: str) -> int:
    try:
        return max(0, int(counter(text)))
    except Exception:
        return 0


_REASONING_OPEN_TAG = "<think>"
_REASONING_CLOSE_TAG = "</think>"


class _ReasoningSplitter:
    """Incrementally split thinking tags, including prompt-opened reasoning."""

    def __init__(self, *, initially_open: bool = False) -> None:
        self._buffer = ""
        self._buffer_start = 0
        self._input_length = 0
        self._in_reasoning = bool(initially_open)

    @property
    def pending_text_length(self) -> int:
        return len(self._buffer)

    @property
    def pending_source_start(self) -> int:
        return self._buffer_start

    def feed(self, text: str) -> list[tuple[str, str]]:
        return [(part.field, part.text) for part in self.feed_parts(text)]

    def finish(self) -> list[tuple[str, str]]:
        return [(part.field, part.text) for part in self.finish_parts()]

    def feed_parts(self, text: str) -> list[_ReasoningPart]:
        if not text:
            return []
        if not self._buffer:
            self._buffer_start = self._input_length
        self._buffer += text
        self._input_length += len(text)
        return self._drain(final=False)

    def finish_parts(self) -> list[_ReasoningPart]:
        parts = self._drain(final=True)
        if parts and parts[-1].field == "content":
            stripped = _strip_chat_terminal_markers(parts[-1].text)
            if stripped != parts[-1].text:
                parts[-1] = _ReasoningPart(
                    field="content",
                    text=stripped,
                    source_start=parts[-1].source_start,
                    source_end=parts[-1].source_start + len(stripped),
                )
        return parts

    def _drain(self, *, final: bool) -> list[_ReasoningPart]:
        outputs: list[_ReasoningPart] = []
        while self._buffer:
            markers = (
                (self._buffer.find(_REASONING_OPEN_TAG), _REASONING_OPEN_TAG, True),
                (self._buffer.find(_REASONING_CLOSE_TAG), _REASONING_CLOSE_TAG, False),
            )
            found = min((item for item in markers if item[0] >= 0), default=None)
            if found is not None:
                index, tag, next_state = found
                self._append(outputs, self._buffer[:index], self._buffer_start)
                self._buffer = self._buffer[index + len(tag) :]
                self._buffer_start += index + len(tag)
                self._in_reasoning = next_state
                continue
            if final:
                self._append(outputs, self._buffer, self._buffer_start)
                self._buffer_start += len(self._buffer)
                self._buffer = ""
                break
            keep = max(
                _tag_suffix_len(self._buffer, _REASONING_OPEN_TAG),
                _tag_suffix_len(self._buffer, _REASONING_CLOSE_TAG),
            )
            emit_len = len(self._buffer) - keep
            if emit_len > 0:
                self._append(outputs, self._buffer[:emit_len], self._buffer_start)
                self._buffer = self._buffer[emit_len:]
                self._buffer_start += emit_len
            break
        return outputs

    def _append(self, outputs: list[_ReasoningPart], text: str, source_start: int) -> None:
        if text:
            field = "reasoning_content" if self._in_reasoning else "content"
            outputs.append(
                _ReasoningPart(
                    field=field,
                    text=text,
                    source_start=int(source_start),
                    source_end=int(source_start) + len(text),
                )
            )


def _tag_suffix_len(text: str, tag: str) -> int:
    max_len = min(len(tag) - 1, len(text))
    for length in range(max_len, 0, -1):
        if tag.startswith(text[-length:]):
            return length
    return 0


def _classify_chat_length_phase(text: str) -> str:
    if _tag_suffix_len(text, _REASONING_OPEN_TAG):
        return "reasoning"
    in_reasoning = text.rfind(_REASONING_OPEN_TAG) > text.rfind(_REASONING_CLOSE_TAG)
    if in_reasoning:
        if _tag_suffix_len(text, _REASONING_CLOSE_TAG):
            return "closing_think"
        return "reasoning"
    if _has_unclosed_tool_call(text):
        return "tool_call"
    if _looks_like_partial_json(text):
        return "structured"
    return "answer"


def _has_unclosed_tool_call(text: str) -> bool:
    lowered = text.lower()
    if _tag_suffix_len(lowered, "<tool_call>") or _tag_suffix_len(lowered, "</tool_call>"):
        return True
    open_index = lowered.rfind("<tool_call>")
    close_index = lowered.rfind("</tool_call>")
    return open_index > close_index


def _looks_like_partial_json(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return False
    try:
        json.loads(stripped)
    except Exception:
        return True
    return False


def _reasoning_initially_open_for_prompt(
    engine: Any | None,
    prompt: PromptInput,
) -> bool:
    target = _model_chat_protocol_target(engine)
    parser = getattr(target, "chat_reasoning_parser", None)
    if not isinstance(prompt, str):
        ids_detector = getattr(parser, "initially_open_ids", None)
        if callable(ids_detector):
            return bool(ids_detector(tuple(int(token) for token in prompt)))
    detector = getattr(parser, "initially_open", None)
    if not callable(detector):
        return False
    return bool(detector(str(prompt)))


def _split_reasoning(text: str, *, initially_open: bool = False) -> _ReasoningSplit:
    splitter = _ReasoningSplitter(initially_open=initially_open)
    parts = splitter.feed(text) + splitter.finish()
    content = "".join(part for field, part in parts if field == "content")
    reasoning = "".join(part for field, part in parts if field == "reasoning_content")
    return _ReasoningSplit(content=content, reasoning_content=reasoning)


@dataclass(frozen=True)
class _ToolCallBlock:
    start: int
    end: int
    parsed: _ParsedToolCall


def _find_marker(lowered: str, marker: str, start: int, end: int | None = None) -> int:
    if end is None:
        return lowered.find(marker, start)
    return lowered.find(marker, start, end)


def _find_valid_tool_call_block(text: str, lowered: str, start_at: int) -> _ToolCallBlock | None:
    start = _find_marker(lowered, _TOOL_CALL_START_MARKER, start_at)
    while start >= 0:
        body_start = start + len(_TOOL_CALL_START_MARKER)
        end = _find_marker(lowered, _TOOL_CALL_END_MARKER, body_start)
        while end >= 0:
            raw_text = text[start : end + len(_TOOL_CALL_END_MARKER)]
            parsed = _parsed_tool_call_from_block_body(text[body_start:end], raw_text=raw_text)
            if parsed is not None:
                return _ToolCallBlock(
                    start=start,
                    end=end + len(_TOOL_CALL_END_MARKER),
                    parsed=parsed,
                )
            end = _find_marker(lowered, _TOOL_CALL_END_MARKER, end + len(_TOOL_CALL_END_MARKER))
        start = _find_marker(lowered, _TOOL_CALL_START_MARKER, body_start)
    return None


def _valid_tool_call_blocks(text: str) -> tuple[_ToolCallBlock, ...]:
    lowered = text.lower()
    blocks: list[_ToolCallBlock] = []
    cursor = 0
    while cursor < len(text):
        block = _find_valid_tool_call_block(text, lowered, cursor)
        if block is None:
            break
        blocks.append(block)
        cursor = block.end
    return tuple(blocks)


def _is_chat_template_control_residue(text: str) -> bool:
    """Return whether text contains only leaked Qwen template controls/roles."""

    residue = str(text)
    cursor = 0
    saw_marker = False
    role_allowed = False
    while cursor < len(residue):
        if residue[cursor].isspace():
            cursor += 1
            continue
        marker = next(
            (item for item in _CHAT_TEMPLATE_RESIDUE_MARKERS if residue.startswith(item, cursor)),
            None,
        )
        if marker is not None:
            cursor += len(marker)
            saw_marker = True
            role_allowed = role_allowed or marker == "<|im_start|>"
            continue
        if role_allowed:
            role = next(
                (item for item in _CHAT_TEMPLATE_RESIDUE_ROLES if residue.startswith(item, cursor)),
                None,
            )
            if role is not None:
                role_end = cursor + len(role)
                if role_end == len(residue) or residue[role_end] in "\r\n":
                    cursor = role_end
                    role_allowed = False
                    continue
        return False
    return saw_marker


def _strip_chat_template_terminal_edges(text: str) -> str:
    """Remove chat-template residue only around a parsed tool envelope."""

    stripped = str(text).strip()
    while stripped:
        previous = stripped
        for marker in _CHAT_TEMPLATE_TERMINAL_MARKERS:
            if stripped.startswith(marker):
                stripped = stripped[len(marker) :].lstrip()
            if stripped.endswith(marker):
                stripped = stripped[: -len(marker)].rstrip()
        if stripped == previous:
            break
    if _is_chat_template_control_residue(stripped):
        return ""
    return stripped


def _is_incomplete_duplicate_tool_prefix_control_residue(text: str, *, tool_name: str) -> bool:
    """Return whether text is an unfinished duplicate prefix plus Qwen controls."""

    residue = str(text).strip()
    prefix = _tool_call_name_prefix_text(tool_name)
    if not residue.startswith(prefix):
        return False
    residue = residue[len(prefix) :].lstrip()
    if residue.startswith("{"):
        residue = residue[1:].lstrip()
    return _is_chat_template_control_residue(residue)


def _parse_chat_tool_calls_for_engine(
    engine: Any | None,
    text: str,
    *,
    tools: Sequence[Mapping[str, Any]] | None,
) -> _ParsedChatOutput:
    target = _model_chat_protocol_target(engine)
    parser = getattr(target, "chat_tool_parser", None)
    parse = getattr(parser, "parse", None)
    if not callable(parse):
        return _parse_chat_tool_calls(text)
    parser_name = str(getattr(parser, "name", type(parser).__name__))
    try:
        result = parse(str(text), tools=tools)
        content = getattr(result, "content")
        raw_calls = getattr(result, "tool_calls")
        invalid_blocks = getattr(result, "invalid_blocks", ())
        if not isinstance(content, str):
            raise TypeError("model tool parser content must be a string")
        calls: list[_ParsedToolCall] = []
        for raw_call in raw_calls:
            name = getattr(raw_call, "name")
            arguments = getattr(raw_call, "arguments")
            raw_text = getattr(raw_call, "raw_text", "")
            if not isinstance(name, str) or not name:
                raise TypeError("model tool parser call name must be a non-empty string")
            calls.append(
                _ParsedToolCall(
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    name=name,
                    arguments=(
                        arguments
                        if isinstance(arguments, str)
                        else _tool_arguments_json(arguments)
                    ),
                    raw_text=str(raw_text),
                )
            )
        return _ParsedChatOutput(
            text=content,
            tool_calls=tuple(calls),
            tool_parser_name=parser_name,
            invalid_tool_call_blocks=tuple(str(block) for block in invalid_blocks),
        )
    except Exception:
        _LOGGER.exception("model-owned chat tool parser failed")
        return _ParsedChatOutput(
            text=str(text),
            tool_calls=(),
            tool_parser_name=parser_name,
            invalid_tool_call_blocks=(str(text),),
        )


def _parse_chat_tool_calls(text: str) -> _ParsedChatOutput:
    calls: list[_ParsedToolCall] = []
    text_parts: list[str] = []
    last_end = 0
    for block in _valid_tool_call_blocks(text):
        text_parts.append(text[last_end : block.start])
        calls.append(block.parsed)
        last_end = block.end
    text_parts.append(text[last_end:])
    if not calls:
        stripped = text.strip()
        parsed = _parsed_tool_call_from_json(stripped, raw_text=stripped)
        if parsed is not None:
            return _ParsedChatOutput(text="", tool_calls=(parsed,))
    visible_text = "".join(text_parts).strip()
    if calls:
        if _is_incomplete_duplicate_tool_prefix_control_residue(
            visible_text,
            tool_name=calls[0].name,
        ):
            visible_text = ""
        else:
            visible_text = _strip_chat_template_terminal_edges(visible_text)
    return _ParsedChatOutput(text=visible_text, tool_calls=tuple(calls))


def _parsed_tool_call_from_block_body(raw: str, *, raw_text: str = "") -> _ParsedToolCall | None:
    stripped = raw.strip()
    parsed = _parsed_tool_call_from_json(stripped, raw_text=raw_text)
    if parsed is not None:
        return parsed
    repaired = _strip_duplicate_tool_call_start(stripped)
    if repaired == stripped:
        return None
    return _parsed_tool_call_from_json(repaired, raw_text=raw_text)


def _strip_duplicate_tool_call_start(text: str) -> str:
    stripped = text.lstrip()
    marker_len = len(_TOOL_CALL_START_MARKER)
    if not stripped[:marker_len].lower() == _TOOL_CALL_START_MARKER:
        return text
    return stripped[marker_len:].lstrip()


def _parsed_tool_call_from_json(raw: str, *, raw_text: str = "") -> _ParsedToolCall | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    return _parsed_tool_call_from_mapping(payload, raw_text=raw_text)


def _parsed_tool_call_from_mapping(payload: Mapping[str, Any], *, raw_text: str = "") -> _ParsedToolCall | None:
    function = payload.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = payload.get("name")
        arguments = payload.get("arguments", {})
    if not isinstance(name, str) or not name:
        return None
    return _ParsedToolCall(
        id=f"call_{uuid.uuid4().hex[:24]}",
        name=name,
        arguments=_tool_arguments_json(arguments),
        raw_text=str(raw_text),
    )


def _tool_arguments_json(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except Exception:
            return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    if arguments is None:
        arguments = {}
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def _strip_chat_terminal_markers(text: str) -> str:
    """Remove trailing chat-template terminal markers (e.g. the EOS token).

    The decoded generation includes the terminating special token (for Qwen
    ``<|im_end|>``) in visible content. This strips trailing terminal markers
    and trailing whitespace so clients never see them.
    """

    stripped = str(text)
    changed = True
    while changed:
        changed = False
        for marker in _CHAT_TEMPLATE_TERMINAL_MARKERS:
            if marker and stripped.endswith(marker):
                stripped = stripped[: -len(marker)]
                changed = True
    return stripped.rstrip()


def _chat_message_from_parsed(parsed: _ParsedChatOutput) -> tuple[dict[str, Any], str]:
    split = _split_reasoning(
        parsed.text,
        initially_open=parsed.reasoning_initially_open,
    )
    message: dict[str, Any] = {"role": "assistant", "content": _strip_chat_terminal_markers(split.content)}
    if split.reasoning_content:
        message["reasoning_content"] = split.reasoning_content
    if parsed.tool_calls:
        message["tool_calls"] = [_openai_tool_call(call) for call in parsed.tool_calls]
        return message, "tool_calls"
    return message, "stop"


def _validate_chat_tool_result(
    request: ChatCompletionRequest,
    parsed: _ParsedChatOutput,
    raw_text: str,
) -> _ToolValidationResult:
    if not request.tools:
        return _ToolValidationResult(parsed)
    mode, required_name = _tool_choice_mode(request.tool_choice)
    strict = _strict_tool_validation_enabled(request)
    if parsed.tool_parser_name is not None:
        malformed_blocks = parsed.invalid_tool_call_blocks if strict else ()
        unparseable_blocks = parsed.invalid_tool_call_blocks
    else:
        malformed_blocks = _malformed_tool_call_blocks(raw_text) if strict else ()
        unparseable_blocks = _unparseable_tool_call_blocks(raw_text)
    if strict:
        if mode == "none":
            if parsed.tool_calls or malformed_blocks:
                return _tool_validation_failure("invalid_tool_call")
            return _ToolValidationResult(parsed)
        if malformed_blocks:
            return _tool_validation_failure("invalid_tool_call")
    if unparseable_blocks:
        return _tool_validation_failure("invalid_tool_call")
    if not parsed.tool_calls:
        if strict and mode in {"required", "function"}:
            return _tool_validation_failure("tool_required_not_satisfied")
        return _ToolValidationResult(parsed)
    if len(parsed.tool_calls) > 1 and not bool(request.parallel_tool_calls):
        return _tool_validation_failure("invalid_tool_call")

    tools_by_name = _tool_map_by_name(request.tools)
    for call in parsed.tool_calls:
        if mode == "function" and required_name is not None and call.name != required_name:
            return _tool_validation_failure("invalid_tool_call")
        tool = tools_by_name.get(call.name)
        if tool is None:
            return _tool_validation_failure("invalid_tool_call")
        if strict:
            schema = _tool_parameters_schema(tool)
            if schema is not None:
                schema_error = _validate_json_schema_value(
                    _tool_call_arguments_value(call),
                    schema,
                    path=f"{call.name}.arguments",
                )
                if schema_error is not None:
                    return _tool_validation_failure("schema_violation")
    if (
        strict
        and mode == "function"
        and required_name is not None
        and not any(call.name == required_name for call in parsed.tool_calls)
    ):
        return _tool_validation_failure("tool_required_not_satisfied")
    return _ToolValidationResult(parsed)


def _tool_validation_failure(reason: str) -> _ToolValidationResult:
    return _ToolValidationResult(_ParsedChatOutput(text="", tool_calls=()), str(reason))


def _invalid_tool_call_error_mode(request: ChatCompletionRequest) -> str:
    raw_mode = request.invalid_tool_call_error_mode
    if raw_mode is None:
        return "finish_details"
    if not isinstance(raw_mode, str) or not raw_mode.strip():
        raise OpenAIHTTPError(
            400,
            "invalid_tool_call_error_mode must be a non-empty string",
            code="invalid_request",
            param="invalid_tool_call_error_mode",
        )
    mode = raw_mode.strip().lower()
    aliases = {
        "normal": "finish_details",
        "response": "finish_details",
        "http_error": "hard_error",
        "sse_error": "hard_error",
    }
    mode = aliases.get(mode, mode)
    if mode not in _INVALID_TOOL_CALL_ERROR_MODES:
        raise OpenAIHTTPError(
            400,
            "invalid_tool_call_error_mode must be one of: "
            + ", ".join(_INVALID_TOOL_CALL_ERROR_MODES),
            code="invalid_request",
            param="invalid_tool_call_error_mode",
        )
    return mode


def _invalid_tool_call_hard_error(
    request: ChatCompletionRequest,
    tool_validation: _ToolValidationResult,
    *,
    finish_details: Mapping[str, Any],
) -> OpenAIHTTPError | None:
    if not tool_validation.failed or tool_validation.failure_reason != "invalid_tool_call":
        return None
    if _invalid_tool_call_error_mode(request) != "hard_error":
        return None
    return OpenAIHTTPError(
        400,
        "generated tool call failed validation",
        code="invalid_tool_call",
        param="tool_calls",
        finish_details=finish_details,
    )


def _raise_invalid_tool_call_hard_error_if_requested(
    request: ChatCompletionRequest,
    tool_validation: _ToolValidationResult,
    *,
    finish_details: Mapping[str, Any],
) -> None:
    error = _invalid_tool_call_hard_error(
        request,
        tool_validation,
        finish_details=finish_details,
    )
    if error is not None:
        raise error


def _strict_tool_validation_enabled(request: ChatCompletionRequest) -> bool:
    if not request.tools:
        return False
    mode, _name = _tool_choice_mode(request.tool_choice)
    if mode in {"none", "required", "function"}:
        return True
    if request.parallel_tool_calls is not None:
        return True
    return any(_tool_function(tool).get("strict") is True for tool in request.tools)


def _tool_choice_mode(tool_choice: str | Mapping[str, Any] | None) -> tuple[str, str | None]:
    if tool_choice is None:
        return "auto", None
    if isinstance(tool_choice, str):
        value = tool_choice.strip().lower()
        if value in {"none", "auto", "required"}:
            return value, None
        return "function", tool_choice.strip()
    if isinstance(tool_choice, Mapping):
        choice_type = str(tool_choice.get("type", "")).strip().lower()
        function = tool_choice.get("function")
        if choice_type == "function" and isinstance(function, Mapping):
            name = function.get("name")
            if isinstance(name, str) and name:
                return "function", name
    return "auto", None


def _tool_map_by_name(tools: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    mapped: dict[str, Mapping[str, Any]] = {}
    for tool in tools:
        function = _tool_function(tool)
        name = function.get("name")
        if isinstance(name, str) and name:
            mapped[name] = tool
    return mapped


def _tool_function(tool: Mapping[str, Any]) -> Mapping[str, Any]:
    function = tool.get("function")
    return function if isinstance(function, Mapping) else {}


def _tool_parameters_schema(tool: Mapping[str, Any]) -> Mapping[str, Any] | None:
    function = _tool_function(tool)
    parameters = function.get("parameters")
    return parameters if isinstance(parameters, Mapping) else None


def _tool_call_arguments_value(call: _ParsedToolCall) -> Any:
    try:
        return json.loads(call.arguments)
    except Exception:
        return call.arguments


def _malformed_tool_call_blocks(text: str) -> tuple[str, ...]:
    return _invalid_tool_call_blocks(text)


def _unparseable_tool_call_blocks(text: str) -> tuple[str, ...]:
    return _invalid_tool_call_blocks(text)


def _invalid_tool_call_blocks(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    invalid: list[str] = []
    cursor = 0
    blocks = _valid_tool_call_blocks(text)
    for block in blocks:
        invalid.extend(_invalid_tool_call_blocks_between(text, lowered, cursor, block.start, flag_unclosed=False))
        cursor = block.end
    invalid.extend(_invalid_tool_call_blocks_between(text, lowered, cursor, len(text), flag_unclosed=True))
    return tuple(invalid)


def _invalid_tool_call_blocks_between(
    text: str,
    lowered: str,
    start: int,
    end: int,
    *,
    flag_unclosed: bool,
) -> tuple[str, ...]:
    invalid: list[str] = []
    cursor = start
    while cursor < end:
        open_index = _find_marker(lowered, _TOOL_CALL_START_MARKER, cursor, end)
        if open_index < 0:
            break
        close_index = _find_marker(
            lowered,
            _TOOL_CALL_END_MARKER,
            open_index + len(_TOOL_CALL_START_MARKER),
            end,
        )
        if close_index < 0:
            if flag_unclosed:
                invalid.append(text[open_index:end])
            break
        close_end = close_index + len(_TOOL_CALL_END_MARKER)
        invalid.append(text[open_index:close_end])
        cursor = close_end
    return tuple(invalid)


def _json_schema_pointer_token(raw_token: str) -> tuple[str | None, str | None]:
    token: list[str] = []
    index = 0
    while index < len(raw_token):
        char = raw_token[index]
        if char != "~":
            token.append(char)
            index += 1
            continue
        if index + 1 >= len(raw_token) or raw_token[index + 1] not in {"0", "1"}:
            return None, f"invalid JSON Pointer escape in {raw_token!r}"
        token.append("~" if raw_token[index + 1] == "0" else "/")
        index += 2
    return "".join(token), None


def _json_schema_resolve_local_ref(
    root: Mapping[str, Any],
    ref: str,
    *,
    root_path: str,
) -> tuple[Mapping[str, Any] | None, str, tuple[str, ...], str | None]:
    text = ref.strip()
    if not text:
        return None, root_path, (), "$ref must be a non-empty string"
    if not text.startswith("#"):
        return None, root_path, (), "only local JSON schema $ref values are supported"
    fragment = unquote(text[1:])
    if fragment == "":
        return root, root_path, (), None
    if not fragment.startswith("/"):
        return None, root_path, (), "$ref must be # or a JSON Pointer starting with #/"

    current: Any = root
    current_path = root_path
    tokens: list[str] = []
    for raw_token in fragment[1:].split("/"):
        token, error = _json_schema_pointer_token(raw_token)
        if error is not None or token is None:
            return None, current_path, tuple(tokens), error
        tokens.append(token)
        if isinstance(current, Mapping):
            if token not in current:
                return None, current_path, tuple(tokens), f"$ref target {text!r} does not exist"
            current = current[token]
            current_path = f"{current_path}.{token}"
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            if not token.isdecimal():
                return None, current_path, tuple(tokens), f"$ref target {text!r} does not exist"
            item_index = int(token)
            if item_index >= len(current):
                return None, current_path, tuple(tokens), f"$ref target {text!r} does not exist"
            current = current[item_index]
            current_path = f"{current_path}[{item_index}]"
        else:
            return None, current_path, tuple(tokens), f"$ref target {text!r} does not exist"
    if not isinstance(current, Mapping):
        return None, current_path, tuple(tokens), "$ref target must be a schema object"
    return current, current_path, tuple(tokens), None


def _validate_json_schema_subset(
    schema: Mapping[str, Any],
    *,
    path: str,
    root: Mapping[str, Any] | None = None,
    root_path: str | None = None,
    ref_stack: tuple[tuple[str, ...], ...] = (),
) -> tuple[str, str] | None:
    root_schema = schema if root is None else root
    root_param = path if root_path is None else root_path
    for raw_key in schema:
        key = str(raw_key)
        if key not in _JSON_SCHEMA_SUPPORTED_KEYS:
            param = f"{path}.{key}"
            return (param, f"{param} is not supported by hipEngine JSON schema subset")

    for key in ("$defs", "definitions"):
        definitions = schema.get(key)
        if definitions is None:
            continue
        if not isinstance(definitions, Mapping):
            return (f"{path}.{key}", f"{path}.{key} must be an object")
        for raw_name, subschema in definitions.items():
            definition_path = f"{path}.{key}.{raw_name}"
            if not isinstance(raw_name, str):
                return (f"{path}.{key}", f"{path}.{key} keys must be strings")
            if not isinstance(subschema, Mapping):
                return (definition_path, f"{definition_path} must be an object")
            error = _validate_json_schema_subset(
                subschema,
                path=definition_path,
                root=root_schema,
                root_path=root_param,
                ref_stack=ref_stack,
            )
            if error is not None:
                return error

    ref = schema.get("$ref")
    if "$ref" in schema:
        if not isinstance(ref, str):
            return (f"{path}.$ref", f"{path}.$ref must be a string")
        target, target_path, pointer, error_message = _json_schema_resolve_local_ref(
            root_schema,
            ref,
            root_path=root_param,
        )
        if error_message is not None or target is None:
            return (f"{path}.$ref", f"{path}.$ref {error_message}")
        if pointer in ref_stack:
            return (f"{path}.$ref", f"{path}.$ref cycle is not supported")
        error = _validate_json_schema_subset(
            target,
            path=target_path,
            root=root_schema,
            root_path=root_param,
            ref_stack=(*ref_stack, pointer),
        )
        if error is not None:
            return error

    expected = schema.get("type")
    if expected is not None:
        if isinstance(expected, str):
            expected_types = (expected,)
        elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
            expected_types = tuple(expected)
        else:
            return (f"{path}.type", f"{path}.type must be a string or list of strings")
        for item in expected_types:
            if not isinstance(item, str) or item not in _JSON_SCHEMA_SUPPORTED_TYPES:
                return (f"{path}.type", f"{path}.type contains unsupported type {item!r}")

    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, Sequence) or isinstance(enum, (str, bytes))):
        return (f"{path}.enum", f"{path}.enum must be an array")

    for key in ("allOf", "anyOf", "oneOf"):
        subschemas = schema.get(key)
        if subschemas is None:
            continue
        if not isinstance(subschemas, Sequence) or isinstance(subschemas, (str, bytes)) or not subschemas:
            return (f"{path}.{key}", f"{path}.{key} must be a non-empty array of schema objects")
        for index, subschema in enumerate(subschemas):
            subschema_path = f"{path}.{key}[{index}]"
            if not isinstance(subschema, Mapping):
                return (subschema_path, f"{subschema_path} must be an object")
            error = _validate_json_schema_subset(
                subschema,
                path=subschema_path,
                root=root_schema,
                root_path=root_param,
                ref_stack=ref_stack,
            )
            if error is not None:
                return error

    not_schema = schema.get("not")
    if "not" in schema:
        if not isinstance(not_schema, Mapping):
            return (f"{path}.not", f"{path}.not must be an object")
        error = _validate_json_schema_subset(
            not_schema,
            path=f"{path}.not",
            root=root_schema,
            root_path=root_param,
            ref_stack=ref_stack,
        )
        if error is not None:
            return error

    for key in ("if", "then", "else"):
        conditional_schema = schema.get(key)
        if key not in schema:
            continue
        if not isinstance(conditional_schema, Mapping):
            return (f"{path}.{key}", f"{path}.{key} must be an object")
        error = _validate_json_schema_subset(
            conditional_schema,
            path=f"{path}.{key}",
            root=root_schema,
            root_path=root_param,
            ref_stack=ref_stack,
        )
        if error is not None:
            return error

    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            return (f"{path}.properties", f"{path}.properties must be an object")
        for raw_key, subschema in properties.items():
            property_path = f"{path}.properties.{raw_key}"
            if not isinstance(subschema, Mapping):
                return (property_path, f"{property_path} must be an object")
            error = _validate_json_schema_subset(
                subschema,
                path=property_path,
                root=root_schema,
                root_path=root_param,
                ref_stack=ref_stack,
            )
            if error is not None:
                return error

    pattern_properties = schema.get("patternProperties")
    if pattern_properties is not None:
        if not isinstance(pattern_properties, Mapping):
            return (f"{path}.patternProperties", f"{path}.patternProperties must be an object")
        for raw_pattern, subschema in pattern_properties.items():
            pattern_path = f"{path}.patternProperties.{raw_pattern}"
            if not isinstance(raw_pattern, str):
                return (f"{path}.patternProperties", f"{path}.patternProperties keys must be strings")
            try:
                re.compile(raw_pattern)
            except re.error as exc:
                return (pattern_path, f"{pattern_path} must be a valid regular expression: {exc}")
            if not isinstance(subschema, Mapping):
                return (pattern_path, f"{pattern_path} must be an object")
            error = _validate_json_schema_subset(
                subschema,
                path=pattern_path,
                root=root_schema,
                root_path=root_param,
                ref_stack=ref_stack,
            )
            if error is not None:
                return error

    property_names = schema.get("propertyNames")
    if property_names is not None:
        if not isinstance(property_names, Mapping):
            return (f"{path}.propertyNames", f"{path}.propertyNames must be an object")
        error = _validate_json_schema_subset(
            property_names,
            path=f"{path}.propertyNames",
            root=root_schema,
            root_path=root_param,
            ref_stack=ref_stack,
        )
        if error is not None:
            return error

    required = schema.get("required")
    if required is not None:
        if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
            return (f"{path}.required", f"{path}.required must be an array of strings")
        if any(not isinstance(item, str) for item in required):
            return (f"{path}.required", f"{path}.required must contain only strings")

    dependent_required = schema.get("dependentRequired")
    if dependent_required is not None:
        if not isinstance(dependent_required, Mapping):
            return (f"{path}.dependentRequired", f"{path}.dependentRequired must be an object")
        for raw_key, dependencies in dependent_required.items():
            dependency_path = f"{path}.dependentRequired.{raw_key}"
            if not isinstance(raw_key, str):
                return (f"{path}.dependentRequired", f"{path}.dependentRequired keys must be strings")
            if not isinstance(dependencies, Sequence) or isinstance(dependencies, (str, bytes)):
                return (dependency_path, f"{dependency_path} must be an array of strings")
            if any(not isinstance(item, str) for item in dependencies):
                return (dependency_path, f"{dependency_path} must contain only strings")

    dependent_schemas = schema.get("dependentSchemas")
    if dependent_schemas is not None:
        if not isinstance(dependent_schemas, Mapping):
            return (f"{path}.dependentSchemas", f"{path}.dependentSchemas must be an object")
        for raw_key, subschema in dependent_schemas.items():
            dependency_path = f"{path}.dependentSchemas.{raw_key}"
            if not isinstance(raw_key, str):
                return (f"{path}.dependentSchemas", f"{path}.dependentSchemas keys must be strings")
            if not isinstance(subschema, Mapping):
                return (dependency_path, f"{dependency_path} must be an object")
            error = _validate_json_schema_subset(
                subschema,
                path=dependency_path,
                root=root_schema,
                root_path=root_param,
                ref_stack=ref_stack,
            )
            if error is not None:
                return error

    additional_properties = schema.get("additionalProperties")
    if "additionalProperties" in schema:
        if isinstance(additional_properties, bool):
            pass
        elif isinstance(additional_properties, Mapping):
            error = _validate_json_schema_subset(
                additional_properties,
                path=f"{path}.additionalProperties",
                root=root_schema,
                root_path=root_param,
                ref_stack=ref_stack,
            )
            if error is not None:
                return error
        else:
            return (
                f"{path}.additionalProperties",
                f"{path}.additionalProperties must be a boolean or object in hipEngine JSON schema subset",
            )

    items = schema.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            return (f"{path}.items", f"{path}.items must be an object")
        error = _validate_json_schema_subset(
            items,
            path=f"{path}.items",
            root=root_schema,
            root_path=root_param,
            ref_stack=ref_stack,
        )
        if error is not None:
            return error

    contains = schema.get("contains")
    if contains is not None:
        if not isinstance(contains, Mapping):
            return (f"{path}.contains", f"{path}.contains must be an object")
        error = _validate_json_schema_subset(
            contains,
            path=f"{path}.contains",
            root=root_schema,
            root_path=root_param,
            ref_stack=ref_stack,
        )
        if error is not None:
            return error

    for key in (
        "minProperties",
        "maxProperties",
        "minItems",
        "maxItems",
        "minContains",
        "maxContains",
        "minLength",
        "maxLength",
    ):
        if key in schema and _schema_nonnegative_int(schema.get(key)) is None:
            return (f"{path}.{key}", f"{path}.{key} must be a non-negative integer")
    if "uniqueItems" in schema and not isinstance(schema.get("uniqueItems"), bool):
        return (f"{path}.uniqueItems", f"{path}.uniqueItems must be a boolean")
    if "pattern" in schema:
        pattern = schema.get("pattern")
        if not isinstance(pattern, str):
            return (f"{path}.pattern", f"{path}.pattern must be a string")
        try:
            re.compile(pattern)
        except re.error as exc:
            return (f"{path}.pattern", f"{path}.pattern must be a valid regular expression: {exc}")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in schema and _schema_finite_number(schema.get(key)) is None:
            return (f"{path}.{key}", f"{path}.{key} must be a finite number")
    if "multipleOf" in schema and _schema_positive_finite_number(schema.get("multipleOf")) is None:
        return (f"{path}.multipleOf", f"{path}.multipleOf must be a positive finite number")
    return None


def _validate_json_schema_value(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str,
    root: Mapping[str, Any] | None = None,
    ref_stack: tuple[tuple[str, ...], ...] = (),
) -> str | None:
    root_schema = schema if root is None else root
    ref = schema.get("$ref")
    if isinstance(ref, str):
        target, _, pointer, error_message = _json_schema_resolve_local_ref(
            root_schema,
            ref,
            root_path="$",
        )
        if error_message is not None or target is None:
            return f"{path} has invalid schema reference"
        if pointer in ref_stack:
            return f"{path} has cyclic schema reference"
        error = _validate_json_schema_value(
            value,
            target,
            path=path,
            root=root_schema,
            ref_stack=(*ref_stack, pointer),
        )
        if error is not None:
            return error
    enum = schema.get("enum")
    if (
        isinstance(enum, Sequence)
        and not isinstance(enum, (str, bytes))
        and not any(_schema_json_values_equal(value, item) for item in enum)
    ):
        return f"{path} is not one of the allowed enum values"
    if "const" in schema and not _schema_json_values_equal(value, schema.get("const")):
        return f"{path} does not match const"
    expected = schema.get("type")
    if expected is not None and not _json_schema_type_matches(value, expected):
        return f"{path} does not match schema type {expected!r}"
    all_of = schema.get("allOf")
    if isinstance(all_of, Sequence) and not isinstance(all_of, (str, bytes)):
        for subschema in all_of:
            if not isinstance(subschema, Mapping):
                continue
            error = _validate_json_schema_value(
                value,
                subschema,
                path=path,
                root=root_schema,
                ref_stack=ref_stack,
            )
            if error is not None:
                return error
    any_of = schema.get("anyOf")
    if isinstance(any_of, Sequence) and not isinstance(any_of, (str, bytes)):
        matches = 0
        for subschema in any_of:
            if (
                isinstance(subschema, Mapping)
                and _validate_json_schema_value(
                    value,
                    subschema,
                    path=path,
                    root=root_schema,
                    ref_stack=ref_stack,
                )
                is None
            ):
                matches += 1
                break
        if matches == 0:
            return f"{path} does not match any allowed schema"
    one_of = schema.get("oneOf")
    if isinstance(one_of, Sequence) and not isinstance(one_of, (str, bytes)):
        matches = sum(
            1
            for subschema in one_of
            if isinstance(subschema, Mapping)
            and _validate_json_schema_value(
                value,
                subschema,
                path=path,
                root=root_schema,
                ref_stack=ref_stack,
            )
            is None
        )
        if matches != 1:
            return f"{path} must match exactly one allowed schema"
    not_schema = schema.get("not")
    if (
        isinstance(not_schema, Mapping)
        and _validate_json_schema_value(
            value,
            not_schema,
            path=path,
            root=root_schema,
            ref_stack=ref_stack,
        )
        is None
    ):
        return f"{path} matches a disallowed schema"
    if_schema = schema.get("if")
    if isinstance(if_schema, Mapping):
        branch_key = (
            "then"
            if _validate_json_schema_value(
                value,
                if_schema,
                path=path,
                root=root_schema,
                ref_stack=ref_stack,
            )
            is None
            else "else"
        )
        branch_schema = schema.get(branch_key)
        if isinstance(branch_schema, Mapping):
            error = _validate_json_schema_value(
                value,
                branch_schema,
                path=path,
                root=root_schema,
                ref_stack=ref_stack,
            )
            if error is not None:
                return error
    schema_type = _primary_json_schema_type(expected, value)
    if schema_type == "object":
        if not isinstance(value, Mapping):
            return f"{path} must be an object"
        min_properties = _schema_nonnegative_int(schema.get("minProperties"))
        if min_properties is not None and len(value) < min_properties:
            return f"{path} must have at least {min_properties} properties"
        max_properties = _schema_nonnegative_int(schema.get("maxProperties"))
        if max_properties is not None and len(value) > max_properties:
            return f"{path} must have at most {max_properties} properties"
        required = schema.get("required", ())
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
            for key in required:
                if isinstance(key, str) and key not in value:
                    return f"{path}.{key} is required"
        property_names = schema.get("propertyNames")
        if isinstance(property_names, Mapping):
            for raw_key in value:
                key = str(raw_key)
                error = _validate_json_schema_value(
                    key,
                    property_names,
                    path=f"{path}.{key}",
                    root=root_schema,
                    ref_stack=ref_stack,
                )
                if error is not None:
                    return f"{path}.{key} property name is invalid"
        dependent_required = schema.get("dependentRequired")
        if isinstance(dependent_required, Mapping):
            for raw_key, dependencies in dependent_required.items():
                if not isinstance(raw_key, str) or raw_key not in value:
                    continue
                if isinstance(dependencies, Sequence) and not isinstance(dependencies, (str, bytes)):
                    for dependency in dependencies:
                        if isinstance(dependency, str) and dependency not in value:
                            return f"{path}.{dependency} is required when {raw_key} is present"
        dependent_schemas = schema.get("dependentSchemas")
        if isinstance(dependent_schemas, Mapping):
            for raw_key, subschema in dependent_schemas.items():
                if not isinstance(raw_key, str) or raw_key not in value or not isinstance(subschema, Mapping):
                    continue
                error = _validate_json_schema_value(
                    value,
                    subschema,
                    path=path,
                    root=root_schema,
                    ref_stack=ref_stack,
                )
                if error is not None:
                    return error
        properties = schema.get("properties")
        property_map = properties if isinstance(properties, Mapping) else {}
        for key, subschema in property_map.items():
            if key in value and isinstance(key, str) and isinstance(subschema, Mapping):
                error = _validate_json_schema_value(
                    value[key],
                    subschema,
                    path=f"{path}.{key}",
                    root=root_schema,
                    ref_stack=ref_stack,
                )
                if error is not None:
                    return error
        pattern_properties = schema.get("patternProperties")
        pattern_map = pattern_properties if isinstance(pattern_properties, Mapping) else {}
        matched_by_pattern: set[str] = set()
        for raw_pattern, subschema in pattern_map.items():
            if not isinstance(raw_pattern, str) or not isinstance(subschema, Mapping):
                continue
            for raw_key, item in value.items():
                key = str(raw_key)
                try:
                    matches = re.search(raw_pattern, key) is not None
                except re.error:
                    matches = False
                if not matches:
                    continue
                matched_by_pattern.add(key)
                error = _validate_json_schema_value(
                    item,
                    subschema,
                    path=f"{path}.{key}",
                    root=root_schema,
                    ref_stack=ref_stack,
                )
                if error is not None:
                    return error
        additional_properties = schema.get("additionalProperties")
        if additional_properties is False:
            allowed = {str(key) for key in property_map}
            extra = sorted(
                str(key)
                for key in value.keys()
                if str(key) not in allowed and str(key) not in matched_by_pattern
            )
            if extra:
                return f"{path}.{extra[0]} is not allowed"
        elif isinstance(additional_properties, Mapping):
            allowed = {str(key) for key in property_map}
            extra_items = sorted(
                (str(key), key)
                for key in value.keys()
                if str(key) not in allowed and str(key) not in matched_by_pattern
            )
            for key, raw_key in extra_items:
                error = _validate_json_schema_value(
                    value[raw_key],
                    additional_properties,
                    path=f"{path}.{key}",
                    root=root_schema,
                    ref_stack=ref_stack,
                )
                if error is not None:
                    return error
    elif schema_type == "array":
        if not isinstance(value, list):
            return f"{path} must be an array"
        min_items = _schema_nonnegative_int(schema.get("minItems"))
        if min_items is not None and len(value) < min_items:
            return f"{path} must have at least {min_items} items"
        max_items = _schema_nonnegative_int(schema.get("maxItems"))
        if max_items is not None and len(value) > max_items:
            return f"{path} must have at most {max_items} items"
        if schema.get("uniqueItems") is True and not _schema_array_items_unique(value):
            return f"{path} must contain unique items"
        contains = schema.get("contains")
        if isinstance(contains, Mapping):
            match_count = sum(
                1
                for item in value
                if _validate_json_schema_value(
                    item,
                    contains,
                    path=path,
                    root=root_schema,
                    ref_stack=ref_stack,
                )
                is None
            )
            min_contains = _schema_nonnegative_int(schema.get("minContains"))
            min_contains = 1 if min_contains is None else min_contains
            if match_count < min_contains:
                return f"{path} must contain at least {min_contains} matching items"
            max_contains = _schema_nonnegative_int(schema.get("maxContains"))
            if max_contains is not None and match_count > max_contains:
                return f"{path} must contain at most {max_contains} matching items"
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                error = _validate_json_schema_value(
                    item,
                    items,
                    path=f"{path}[{index}]",
                    root=root_schema,
                    ref_stack=ref_stack,
                )
                if error is not None:
                    return error
    elif schema_type == "string":
        if isinstance(value, str):
            min_length = _schema_nonnegative_int(schema.get("minLength"))
            if min_length is not None and len(value) < min_length:
                return f"{path} must have at least {min_length} characters"
            max_length = _schema_nonnegative_int(schema.get("maxLength"))
            if max_length is not None and len(value) > max_length:
                return f"{path} must have at most {max_length} characters"
            pattern = schema.get("pattern")
            if isinstance(pattern, str) and re.search(pattern, value) is None:
                return f"{path} does not match pattern"
    elif schema_type in {"integer", "number"}:
        numeric = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        if numeric is not None:
            minimum = _schema_finite_number(schema.get("minimum"))
            if minimum is not None and numeric < minimum:
                return f"{path} must be >= {minimum:g}"
            maximum = _schema_finite_number(schema.get("maximum"))
            if maximum is not None and numeric > maximum:
                return f"{path} must be <= {maximum:g}"
            exclusive_minimum = _schema_finite_number(schema.get("exclusiveMinimum"))
            if exclusive_minimum is not None and numeric <= exclusive_minimum:
                return f"{path} must be > {exclusive_minimum:g}"
            exclusive_maximum = _schema_finite_number(schema.get("exclusiveMaximum"))
            if exclusive_maximum is not None and numeric >= exclusive_maximum:
                return f"{path} must be < {exclusive_maximum:g}"
            multiple_of = schema.get("multipleOf")
            if not _schema_number_is_multiple_of(value, multiple_of):
                assert isinstance(multiple_of, (int, float))
                return f"{path} must be a multiple of {float(multiple_of):g}"
    return None


def _schema_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _schema_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _schema_positive_finite_number(value: Any) -> float | None:
    numeric = _schema_finite_number(value)
    if numeric is None or numeric <= 0:
        return None
    return numeric


def _schema_decimal_number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return numeric if numeric.is_finite() else None


def _schema_number_is_multiple_of(value: Any, multiple_of: Any) -> bool:
    divisor = _schema_decimal_number(multiple_of)
    if divisor is None or divisor <= 0:
        return True
    numeric = _schema_decimal_number(value)
    if numeric is None:
        return True
    return numeric % divisor == 0


def _schema_array_items_unique(values: Sequence[Any]) -> bool:
    for index, item in enumerate(values):
        for other in values[index + 1 :]:
            if _schema_json_values_equal(item, other):
                return False
    return True


def _schema_json_values_equal(left: Any, right: Any) -> bool:
    left_kind = _schema_json_value_kind(left)
    right_kind = _schema_json_value_kind(right)
    if left_kind != right_kind:
        return False
    if left_kind == "number":
        left_number = _schema_decimal_number(left)
        right_number = _schema_decimal_number(right)
        return left_number is not None and right_number is not None and left_number == right_number
    if left_kind == "array":
        return len(left) == len(right) and all(
            _schema_json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if left_kind == "object":
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_schema_json_values_equal(left[key], right[key]) for key in left)
    return left == right


def _schema_json_value_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if _schema_decimal_number(value) is not None:
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return "unknown"


def _json_schema_type_matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return any(_json_schema_type_matches(value, item) for item in expected)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    return True


def _primary_json_schema_type(expected: Any, value: Any) -> str | None:
    if isinstance(expected, str):
        return expected
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        for item in expected:
            if _json_schema_type_matches(value, item):
                return str(item)
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float) and math.isfinite(value):
        return "number"
    if value is None:
        return "null"
    return None


def _openai_tool_call(call: _ParsedToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": call.arguments,
        },
    }


def _stream_include_usage(request: CompletionRequest | ChatCompletionRequest) -> bool:
    options = request.stream_options
    return isinstance(options, Mapping) and bool(options.get("include_usage"))


def _stream_include_hipengine(request: CompletionRequest | ChatCompletionRequest) -> bool:
    options = request.stream_options
    return isinstance(options, Mapping) and bool(options.get("include_hipengine"))


def _stream_chunk_generated_token_total(chunk: GenerationStreamChunk) -> int | None:
    return _telemetry_generated_token_count(chunk.telemetry)


@dataclass(slots=True)
class _StreamTokenAccounting:
    counter: Any
    streamed_tokens: int = 0
    reasoning_tokens: int = 0
    answer_tokens: int = 0
    tool_call_tokens: int = 0
    structured_tokens: int = 0

    @classmethod
    def for_engine(cls, engine: Any) -> "_StreamTokenAccounting | None":
        counter = getattr(engine, "count_tokens", None)
        return cls(counter) if callable(counter) else None

    def observe(
        self,
        phase: str,
        text: str,
        *,
        cumulative_tokens: int | None = None,
    ) -> dict[str, int]:
        if cumulative_tokens is None or int(cumulative_tokens) < self.streamed_tokens:
            delta_tokens = _safe_count(self.counter, text)
            self.streamed_tokens += delta_tokens
        else:
            exact_total = max(0, int(cumulative_tokens))
            delta_tokens = exact_total - self.streamed_tokens
            self.streamed_tokens = exact_total
        phase_name = str(phase)
        if phase_name == "think":
            self.reasoning_tokens += delta_tokens
        elif phase_name == "tool_call":
            self.tool_call_tokens += delta_tokens
        elif phase_name == "structured":
            self.structured_tokens += delta_tokens
        else:
            self.answer_tokens += delta_tokens
        return self.snapshot(delta_tokens=delta_tokens)

    def observe_stream_chunk(
        self,
        phase: str,
        chunk: GenerationStreamChunk,
    ) -> dict[str, int]:
        return self.observe(
            phase,
            chunk.text,
            cumulative_tokens=_stream_chunk_generated_token_total(chunk),
        )

    def snapshot(self, *, delta_tokens: int | None = None) -> dict[str, int]:
        payload: dict[str, int] = {"streamed_tokens": self.streamed_tokens}
        if delta_tokens is not None:
            payload["delta_tokens"] = max(0, int(delta_tokens))
        if self.reasoning_tokens:
            payload["reasoning_tokens"] = self.reasoning_tokens
        if self.answer_tokens:
            payload["answer_tokens"] = self.answer_tokens
        if self.tool_call_tokens:
            payload["tool_call_tokens"] = self.tool_call_tokens
        if self.structured_tokens:
            payload["structured_tokens"] = self.structured_tokens
        return payload


@dataclass(slots=True)
class _StreamTimingTracker:
    started_at: float
    first_token_at: float | None = None
    observed_tokens: int = 0

    @classmethod
    def start(cls) -> "_StreamTimingTracker":
        return cls(started_at=time.perf_counter())

    def observe(
        self,
        *,
        event: str,
        token_event: bool = False,
        tokens: Mapping[str, int] | None = None,
        usage: Mapping[str, int] | None = None,
    ) -> dict[str, float]:
        now = time.perf_counter()
        token_total = _stream_timing_token_total(tokens=tokens, usage=usage)
        if token_event:
            if self.first_token_at is None:
                self.first_token_at = now
        if token_total > self.observed_tokens:
            self.observed_tokens = token_total
        return self.snapshot(now=now, event=event)

    def snapshot(self, *, now: float, event: str) -> dict[str, float]:
        elapsed_s = max(0.0, now - self.started_at)
        payload: dict[str, float] = {"elapsed_ms": round(elapsed_s * 1000.0, 3)}
        if self.first_token_at is None:
            return payload
        ttft_s = max(0.0, self.first_token_at - self.started_at)
        payload["ttft_ms"] = round(ttft_s * 1000.0, 3)
        if str(event) in {"done", "usage"}:
            decode_elapsed_s = max(0.0, now - self.first_token_at)
            payload["decode_elapsed_ms"] = round(decode_elapsed_s * 1000.0, 3)
            if self.observed_tokens > 0:
                payload["decode_tokens_per_second"] = round(self.observed_tokens / max(decode_elapsed_s, 1e-9), 3)
        return payload


_StreamTimingSource = float | _StreamTimingTracker | None


def _stream_timing_token_total(
    *,
    tokens: Mapping[str, int] | None = None,
    usage: Mapping[str, int] | None = None,
) -> int:
    for source in (tokens, usage):
        if source is None:
            continue
        for key in ("streamed_tokens", "completion_tokens"):
            raw_value = source.get(key)
            if isinstance(raw_value, int) and raw_value > 0:
                return raw_value
    return 0


def _stream_usage_token_payload(
    usage: Mapping[str, int],
    accounting: _StreamTokenAccounting | None,
) -> dict[str, int]:
    payload = {
        "prompt_tokens": max(0, int(usage.get("prompt_tokens", 0))),
        "completion_tokens": max(0, int(usage.get("completion_tokens", 0))),
        "total_tokens": max(0, int(usage.get("total_tokens", 0))),
    }
    if accounting is not None:
        payload.update(accounting.snapshot())
    return payload


def _stream_hipengine_payload(
    event: str,
    *,
    stream_started_at: _StreamTimingSource = None,
    usage: Mapping[str, int] | None = None,
    tokens: Mapping[str, int] | None = None,
    token_event: bool = False,
    backend_timing: Mapping[str, float] | None = None,
    routing: Mapping[str, Any] | None = None,
    kv_pool: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"metadata_version": 1, "event": str(event)}
    if stream_started_at is not None:
        if isinstance(stream_started_at, _StreamTimingTracker):
            payload["timing"] = stream_started_at.observe(
                event=event,
                token_event=token_event,
                tokens=tokens,
                usage=usage,
            )
        else:
            payload["timing"] = {
                "elapsed_ms": round(max(0.0, (time.perf_counter() - stream_started_at) * 1000.0), 3)
            }
    backend_timing_payload = _backend_stream_timing_payload(backend_timing)
    if backend_timing_payload:
        payload.setdefault("timing", {}).update(backend_timing_payload)
    if usage is not None:
        payload["usage"] = dict(usage)
    if routing is not None:
        payload["routing"] = dict(routing)
    if kv_pool is not None:
        payload["kv_pool"] = dict(kv_pool)
    return payload


def _backend_stream_timing_payload(timing: Mapping[str, float] | None) -> dict[str, float]:
    if not timing:
        return {}
    payload: dict[str, float] = {}
    for raw_key, raw_value in timing.items():
        key = str(raw_key).strip()
        if not key:
            continue
        value = float(raw_value)
        if not math.isfinite(value):
            continue
        payload[key if key.startswith("backend_") else f"backend_{key}"] = value
    return payload


def _stream_chunk_backend_timing(stream_chunk: GenerationStreamChunk | None) -> Mapping[str, float] | None:
    telemetry = None if stream_chunk is None else stream_chunk.telemetry
    return None if telemetry is None else telemetry.timing


def _choice_hipengine_payload(
    phase: str,
    *,
    finish_details: Mapping[str, Any] | None = None,
    tokens: Mapping[str, int] | None = None,
    stream_chunk: GenerationStreamChunk | None = None,
    include_generated_token_ids: bool = False,
) -> dict[str, Any]:
    telemetry = None if stream_chunk is None else stream_chunk.telemetry
    payload: dict[str, Any] = {"phase": str(phase)}
    if telemetry is not None:
        payload.update(telemetry.to_json_dict())
        payload["phase"] = payload.get("decode_state", {}).get("phase", str(phase))
    if finish_details is not None:
        payload["finish_details"] = dict(finish_details)
    if (
        include_generated_token_ids
        and stream_chunk is not None
        and stream_chunk.generated_token_ids is not None
    ):
        payload["generated_token_ids"] = list(stream_chunk.generated_token_ids)
        payload["generated_tokens"] = len(stream_chunk.generated_token_ids)
    if tokens is not None:
        token_payload = {str(key): max(0, int(value)) for key, value in tokens.items()}
        payload["tokens"] = token_payload
        if "decode_state" not in payload:
            payload["decode_state"] = DecodeState.from_stream_tokens(
                phase=phase,
                tokens=token_payload,
            ).to_json_dict()
    return payload


def _attach_choice_telemetry(choice: dict[str, Any], detail: GenerationOutput | None) -> None:
    if detail is None:
        return
    telemetry = detail.telemetry
    payload = {} if telemetry is None else telemetry.to_json_dict()
    if detail.generated_token_ids is not None:
        payload["generated_token_ids"] = list(detail.generated_token_ids)
        payload["generated_tokens"] = len(detail.generated_token_ids)
    if not payload:
        return
    finish_details = choice.get("finish_details")
    if isinstance(finish_details, Mapping):
        payload["finish_details"] = dict(finish_details)
        decode_state = payload.get("decode_state")
        if isinstance(decode_state, Mapping) and "continuation_eligible" in finish_details:
            decode_state = dict(decode_state)
            decode_state["continuation_eligible"] = bool(finish_details["continuation_eligible"])
            payload["decode_state"] = decode_state
    existing = choice.get("hipengine")
    if isinstance(existing, Mapping):
        merged = dict(existing)
        merged.update(payload)
        choice["hipengine"] = merged
        return
    choice["hipengine"] = payload


def _attach_stream_hipengine(
    payload: dict[str, Any],
    *,
    include_hipengine: bool,
    event: str,
    stream_started_at: _StreamTimingSource,
    usage: Mapping[str, int] | None = None,
    tokens: Mapping[str, int] | None = None,
    token_event: bool = False,
    backend_timing: Mapping[str, float] | None = None,
    routing: Mapping[str, Any] | None = None,
    kv_pool: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if include_hipengine:
        payload["hipengine"] = _stream_hipengine_payload(
            event,
            stream_started_at=stream_started_at,
            usage=usage,
            tokens=tokens,
            token_event=token_event,
            backend_timing=backend_timing,
            routing=routing,
            kv_pool=kv_pool,
        )
    return payload


def _completion_stream_delta(
    response_id: str,
    created: int,
    model: str,
    text: str,
    *,
    index: int = 0,
    logprobs: Mapping[str, Any] | None = None,
    tokens: Mapping[str, int] | None = None,
    stream_chunk: GenerationStreamChunk | None = None,
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
    phase: str = "answer",
) -> str:
    choice = {
        "text": text,
        "index": int(index),
        "logprobs": None if logprobs is None else dict(logprobs),
        "finish_reason": None,
    }
    if include_hipengine:
        choice["hipengine"] = _choice_hipengine_payload(phase, tokens=tokens, stream_chunk=stream_chunk)
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [choice],
            },
            include_hipengine=include_hipengine,
            event="delta",
            stream_started_at=stream_started_at,
            tokens=tokens,
            token_event=bool(text),
            backend_timing=_stream_chunk_backend_timing(stream_chunk),
            routing=routing,
        )
    )


def _completion_stream_done(
    response_id: str,
    created: int,
    model: str,
    finish_reason: str,
    *,
    index: int = 0,
    finish_details: Mapping[str, Any] | None = None,
    tokens: Mapping[str, int] | None = None,
    stream_chunk: GenerationStreamChunk | None = None,
    include_hipengine: bool = False,
    extra_hipengine: Mapping[str, Any] | None = None,
    stream_started_at: _StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
    kv_pool: Mapping[str, float] | None = None,
    phase: str = "done",
) -> str:
    finish_payload = _finish_details_payload(None, finish_reason) if finish_details is None else dict(finish_details)
    choice = {
        "text": "",
        "index": int(index),
        "logprobs": None,
        "finish_reason": finish_reason,
        "finish_details": finish_payload,
    }
    if include_hipengine:
        hipengine_payload = _choice_hipengine_payload(
            phase,
            finish_details=finish_payload,
            tokens=tokens,
            stream_chunk=stream_chunk,
            include_generated_token_ids=True,
        )
        if extra_hipengine is not None:
            hipengine_payload.update(deepcopy(dict(extra_hipengine)))
        choice["hipengine"] = hipengine_payload
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [choice],
            },
            include_hipengine=include_hipengine,
            event="done",
            stream_started_at=stream_started_at,
            tokens=tokens,
            backend_timing=_stream_chunk_backend_timing(stream_chunk),
            routing=routing,
            kv_pool=kv_pool,
        )
    )


def _completion_stream_usage(
    response_id: str,
    created: int,
    model: str,
    usage: Mapping[str, int],
    *,
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
    kv_pool: Mapping[str, float] | None = None,
) -> str:
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [],
                "usage": dict(usage),
            },
            include_hipengine=include_hipengine,
            event="usage",
            stream_started_at=stream_started_at,
            usage=usage,
            routing=routing,
            kv_pool=kv_pool,
        )
    )


def _completion_stream_error(
    response_id: str,
    created: int,
    model: str,
    message: str,
    *,
    status_code: int = 500,
    code: str | None = None,
    param: str | None = None,
    error_type: str = "server_error",
    finish_details: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
) -> str:
    choice: dict[str, Any] = {
        "text": "",
        "index": 0,
        "logprobs": None,
        "finish_reason": "error",
    }
    error = _error_payload(
        message=message,
        error_type=error_type,
        code=code,
        param=param,
        status_code=status_code,
        finish_details=finish_details,
        extra=extra,
    )
    if finish_details is not None:
        details = dict(finish_details)
        choice["finish_details"] = details
    if include_hipengine:
        choice["hipengine"] = _choice_hipengine_payload(
            "done",
            finish_details=choice.get("finish_details"),
        )
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [choice],
                "error": error,
            },
            include_hipengine=include_hipengine,
            event="error",
            stream_started_at=stream_started_at,
            routing=routing,
        )
    )


def _completion_stream(
    response_id: str,
    created: int,
    model: str,
    texts: Sequence[str],
    choices: Sequence[dict[str, Any]],
    *,
    details: Sequence[GenerationOutput] | None = None,
    usage: Mapping[str, int] | None = None,
    token_accounting: _StreamTokenAccounting | None = None,
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
    kv_pool: Mapping[str, float] | None = None,
    done_phase: str = "done",
    scheduler_token_chunks: Sequence[Mapping[str, Any]] | None = None,
) -> Iterator[str]:
    choices_by_index = {int(choice["index"]): choice for choice in choices}
    details_by_index = (
        {index: detail for index, detail in enumerate(details)}
        if details is not None
        else {}
    )
    scheduler_chunks_by_index = _scheduler_token_chunks_by_request(scheduler_token_chunks)
    for index, text in enumerate(texts):
        choice = choices_by_index.get(index, {})
        scheduler_chunks = scheduler_chunks_by_index.get(index, ())
        if _scheduler_chunks_match_completion_text(
            text,
            scheduler_chunks,
            require_logprobs=choice.get("logprobs") is not None,
        ):
            phase = "structured" if done_phase == "structured" else "answer"
            for raw_chunk in scheduler_chunks:
                stream_chunk = _scheduler_payload_stream_chunk(raw_chunk)
                if stream_chunk is None:
                    continue
                token_payload = (
                    token_accounting.observe_stream_chunk(phase, stream_chunk)
                    if include_hipengine and token_accounting is not None and len(choices) == 1
                    else None
                )
                yield _completion_stream_delta(
                    response_id,
                    created,
                    model,
                    stream_chunk.text,
                    index=index,
                    logprobs=(
                        _completion_stream_logprobs(stream_chunk)
                        if choice.get("logprobs") is not None
                        else None
                    ),
                    tokens=token_payload,
                    stream_chunk=stream_chunk,
                    include_hipengine=include_hipengine,
                    stream_started_at=stream_started_at,
                    routing=routing,
                    phase=phase,
                )
            continue
        final_stream_chunk = _stream_chunk_from_detail("", details_by_index.get(index))
        phase = "structured" if done_phase == "structured" else "answer"
        token_payload = (
            token_accounting.observe(phase, text)
            if include_hipengine and token_accounting is not None and len(choices) == 1
            else None
        )
        yield _completion_stream_delta(
            response_id,
            created,
            model,
            text,
            index=index,
            logprobs=choice.get("logprobs"),
            tokens=token_payload,
            stream_chunk=_buffered_delta_stream_chunk(
                text,
                final_stream_chunk,
                phase=phase,
                tokens=token_payload,
            ),
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
            routing=routing,
            phase=phase,
        )
    final_tokens = (
        _stream_usage_token_payload(usage, token_accounting)
        if include_hipengine and usage is not None and len(choices) == 1
        else None
    )
    for choice in choices:
        yield _completion_stream_done(
            response_id,
            created,
            model,
            str(choice["finish_reason"]),
            index=choice["index"],
            finish_details=choice.get("finish_details"),
            tokens=final_tokens,
            stream_chunk=_stream_chunk_from_detail("", details_by_index.get(int(choice["index"]))),
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
            routing=routing,
            kv_pool=kv_pool,
            phase=done_phase,
        )
    if usage is not None:
        yield _completion_stream_usage(
            response_id,
            created,
            model,
            usage,
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
            routing=routing,
            kv_pool=kv_pool,
        )
    yield "data: [DONE]\n\n"


def _scheduler_token_chunks_by_request(
    chunks: Sequence[Mapping[str, Any]] | None,
) -> dict[int, list[Mapping[str, Any]]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    if not chunks:
        return grouped
    for chunk in chunks:
        try:
            request_id = int(chunk.get("request_id"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        grouped.setdefault(request_id, []).append(chunk)
    for request_chunks in grouped.values():
        request_chunks.sort(key=lambda item: int(item.get("token_index", 0)))
    return grouped


def _scheduler_payload_stream_chunk(payload: Mapping[str, Any]) -> GenerationStreamChunk | None:
    raw_chunk = payload.get("chunk")
    if not isinstance(raw_chunk, Mapping):
        return None
    return _coerce_generation_stream_chunk(raw_chunk)


def _stream_chunk_with_phase(
    stream_chunk: GenerationStreamChunk,
    phase: str,
) -> GenerationStreamChunk:
    if stream_chunk.telemetry is None:
        return stream_chunk
    return GenerationStreamChunk(
        text=stream_chunk.text,
        token_logprobs=stream_chunk.token_logprobs,
        finish_details=stream_chunk.finish_details,
        telemetry=replace(
            stream_chunk.telemetry,
            decode_state=replace(stream_chunk.telemetry.decode_state, phase=phase),
        ),
        generated_token_ids=stream_chunk.generated_token_ids,
    )


def _stream_chunk_row_index(
    stream_chunk: GenerationStreamChunk,
    *,
    row_count: int,
) -> int | None:
    telemetry = stream_chunk.telemetry
    if telemetry is None:
        return None
    row_index = int(telemetry.decode_state.row_index)
    if 0 <= row_index < int(row_count):
        return row_index
    return None


def _scheduler_chunks_match_completion_text(
    text: str,
    chunks: Sequence[Mapping[str, Any]],
    *,
    require_logprobs: bool,
) -> bool:
    if not chunks:
        return False
    pieces: list[str] = []
    for payload in chunks:
        raw_chunk = payload.get("chunk")
        if not isinstance(raw_chunk, Mapping):
            return False
        chunk_text = raw_chunk.get("text")
        if not isinstance(chunk_text, str):
            return False
        if require_logprobs and not raw_chunk.get("token_logprobs"):
            return False
        pieces.append(chunk_text)
    return "".join(pieces) == str(text)


def _withheld_scheduler_tool_chunks_payload(
    reason: str,
    *,
    raw_text: str,
    scheduler_chunks: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not scheduler_chunks:
        return None
    pieces: list[str] = []
    chunk_text_bytes: list[int] = []
    execution_paths: set[str] = set()
    malformed_chunk_count = 0
    for payload in scheduler_chunks:
        stream_chunk = _scheduler_payload_stream_chunk(payload)
        if stream_chunk is None:
            malformed_chunk_count += 1
            continue
        chunk_text = str(stream_chunk.text)
        pieces.append(chunk_text)
        chunk_text_bytes.append(len(chunk_text.encode("utf-8")))
        telemetry = stream_chunk.telemetry
        execution_path = None if telemetry is None else telemetry.decode_state.execution_path
        if execution_path:
            execution_paths.add(str(execution_path))
    if not pieces and malformed_chunk_count == 0:
        return None
    text = "".join(pieces)
    return {
        "withheld_scheduler_tool_chunks": {
            "surface": "chat_tool_argument_delta",
            "reason": str(reason),
            "public_delta": "withheld",
            "chunk_count": len(pieces),
            "malformed_chunk_count": malformed_chunk_count,
            "raw_text_matches": malformed_chunk_count == 0 and text == str(raw_text),
            "text_bytes": len(text.encode("utf-8")),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "chunk_text_bytes": chunk_text_bytes,
            "execution_paths": sorted(execution_paths),
        }
    }


def _withheld_scheduler_logprob_chunks_payload(
    reason: str,
    *,
    raw_text: str,
    scheduler_chunks: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not scheduler_chunks:
        return None
    pieces: list[str] = []
    chunk_text_bytes: list[int] = []
    execution_paths: set[str] = set()
    malformed_chunk_count = 0
    chunks_with_logprobs = 0
    token_logprob_count = 0
    for payload in scheduler_chunks:
        stream_chunk = _scheduler_payload_stream_chunk(payload)
        if stream_chunk is None:
            malformed_chunk_count += 1
            continue
        chunk_text = str(stream_chunk.text)
        pieces.append(chunk_text)
        chunk_text_bytes.append(len(chunk_text.encode("utf-8")))
        if stream_chunk.token_logprobs:
            chunks_with_logprobs += 1
            token_logprob_count += len(stream_chunk.token_logprobs)
        telemetry = stream_chunk.telemetry
        execution_path = None if telemetry is None else telemetry.decode_state.execution_path
        if execution_path:
            execution_paths.add(str(execution_path))
    if not pieces and malformed_chunk_count == 0:
        return None
    text = "".join(pieces)
    return {
        "withheld_scheduler_logprob_chunks": {
            "surface": "chat_logprob_delta",
            "reason": str(reason),
            "public_delta": "buffered_without_scheduler_logprobs",
            "chunk_count": len(pieces),
            "malformed_chunk_count": malformed_chunk_count,
            "chunks_with_logprobs": chunks_with_logprobs,
            "token_logprob_count": token_logprob_count,
            "raw_text_matches": malformed_chunk_count == 0 and text == str(raw_text),
            "text_bytes": len(text.encode("utf-8")),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "chunk_text_bytes": chunk_text_bytes,
            "execution_paths": sorted(execution_paths),
        }
    }


def _scheduler_chunks_support_chat_logprob_stream(
    text: str,
    chunks: Sequence[Mapping[str, Any]],
    *,
    initially_open: bool = False,
) -> bool:
    if not _scheduler_chunks_match_completion_text(text, chunks, require_logprobs=True):
        return False
    splitter = _ReasoningSplitter(initially_open=initially_open)
    last_stream_chunk: GenerationStreamChunk | None = None
    for raw_chunk in chunks:
        stream_chunk = _scheduler_payload_stream_chunk(raw_chunk)
        if stream_chunk is None:
            return False
        last_stream_chunk = stream_chunk
        for delta_field, chunk in splitter.feed(stream_chunk.text):
            if delta_field == "content" and not _stream_token_logprobs_for_text(stream_chunk, chunk):
                return False
            if delta_field == "reasoning_content" and not _stream_token_logprobs_for_text(
                stream_chunk,
                stream_chunk.text,
            ):
                return False
    for delta_field, chunk in splitter.finish():
        if last_stream_chunk is None:
            return False
        if delta_field == "content" and not _stream_token_logprobs_for_text(last_stream_chunk, chunk):
            return False
        if delta_field == "reasoning_content" and not _stream_token_logprobs_for_text(
            last_stream_chunk,
            last_stream_chunk.text,
        ):
            return False
    return True


def _chat_stream(
    response_id: str,
    created: int,
    model: str,
    text: str,
    finish_reason: str,
) -> Iterator[str]:
    yield _chat_stream_role(response_id, created, model)
    split = _split_reasoning(text)
    if split.reasoning_content:
        yield _chat_stream_delta(
            response_id,
            created,
            model,
            "reasoning_content",
            split.reasoning_content,
        )
    if split.content:
        yield _chat_stream_delta(response_id, created, model, "content", split.content)
    yield _chat_stream_done(response_id, created, model, finish_reason)
    yield "data: [DONE]\n\n"


def _chat_stream_role(
    response_id: str,
    created: int,
    model: str,
    *,
    index: int = 0,
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
) -> str:
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": int(index), "delta": {"role": "assistant"}, "finish_reason": None}],
            },
            include_hipengine=include_hipengine,
            event="role",
            stream_started_at=stream_started_at,
            routing=routing,
        )
    )


def _chat_stream_delta(
    response_id: str,
    created: int,
    model: str,
    field: str,
    text: str,
    *,
    index: int = 0,
    logprobs: Mapping[str, Any] | None = None,
    reasoning_logprobs: Mapping[str, Any] | None = None,
    tokens: Mapping[str, int] | None = None,
    stream_chunk: GenerationStreamChunk | None = None,
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
    phase: str | None = None,
) -> str:
    choice: dict[str, Any] = {"index": int(index), "delta": {field: text}, "finish_reason": None}
    if logprobs is not None:
        choice["logprobs"] = dict(logprobs)
    if include_hipengine:
        choice_phase = phase if phase is not None else "think" if field == "reasoning_content" else "answer"
        choice["hipengine"] = _choice_hipengine_payload(choice_phase, tokens=tokens, stream_chunk=stream_chunk)
        if reasoning_logprobs is not None:
            choice["hipengine"]["reasoning_logprobs"] = dict(reasoning_logprobs)
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [choice],
            },
            include_hipengine=include_hipengine,
            event="delta",
            stream_started_at=stream_started_at,
            tokens=tokens,
            token_event=bool(text),
            backend_timing=_stream_chunk_backend_timing(stream_chunk),
            routing=routing,
        )
    )


def _chat_stream_tool_call(
    response_id: str,
    created: int,
    model: str,
    call: _ParsedToolCall,
    *,
    index: int = 0,
    tool_index: int = 0,
    argument_chunk: str | None = None,
    include_name: bool = True,
    tokens: Mapping[str, int] | None = None,
    stream_chunk: GenerationStreamChunk | None = None,
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
) -> str:
    function: dict[str, Any] = {"arguments": call.arguments if argument_chunk is None else str(argument_chunk)}
    if include_name:
        function["name"] = call.name
    choice = {
        "index": int(index),
        "delta": {
            "tool_calls": [
                {
                    "index": int(tool_index),
                    "id": call.id,
                    "type": "function",
                    "function": function,
                }
            ]
        },
        "finish_reason": None,
    }
    if include_hipengine:
        choice["hipengine"] = _choice_hipengine_payload("tool_call", tokens=tokens, stream_chunk=stream_chunk)
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [choice],
            },
            include_hipengine=include_hipengine,
            event="tool_call",
            stream_started_at=stream_started_at,
            tokens=tokens,
            token_event=True,
            backend_timing=_stream_chunk_backend_timing(stream_chunk),
            routing=routing,
        )
    )


def _tool_call_argument_stream_chunks(arguments: str) -> tuple[str, ...]:
    text = str(arguments)
    if len(text) <= _TOOL_CALL_ARGUMENT_STREAM_CHARS:
        return (text,)
    return tuple(
        text[start : start + _TOOL_CALL_ARGUMENT_STREAM_CHARS]
        for start in range(0, len(text), _TOOL_CALL_ARGUMENT_STREAM_CHARS)
    )


def _chat_stream_parsed(
    response_id: str,
    created: int,
    model: str,
    parsed: _ParsedChatOutput,
    finish_reason: str,
    *,
    index: int = 0,
    logprobs: Mapping[str, Any] | None = None,
    finish_details: Mapping[str, Any] | None = None,
    done_tokens: Mapping[str, int] | None = None,
    token_accounting: _StreamTokenAccounting | None = None,
    stream_chunk: GenerationStreamChunk | None = None,
    include_hipengine: bool = False,
    final_hipengine: Mapping[str, Any] | None = None,
    stream_started_at: _StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
    kv_pool: Mapping[str, float] | None = None,
    done_phase: str = "done",
) -> Iterator[str]:
    split = _split_reasoning(
        parsed.text,
        initially_open=parsed.reasoning_initially_open,
    )
    if split.reasoning_content:
        token_payload = (
            token_accounting.observe("think", split.reasoning_content)
            if token_accounting is not None
            else None
        )
        yield _chat_stream_delta(
            response_id,
            created,
            model,
            "reasoning_content",
            split.reasoning_content,
            index=index,
            tokens=token_payload,
            stream_chunk=_buffered_delta_stream_chunk(
                split.reasoning_content,
                stream_chunk,
                phase="think",
                tokens=token_payload,
            ),
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
            routing=routing,
        )
    if split.content:
        content_phase = "structured" if done_phase == "structured" and not parsed.tool_calls else "answer"
        token_payload = (
            token_accounting.observe(content_phase, split.content)
            if token_accounting is not None
            else None
        )
        yield _chat_stream_delta(
            response_id,
            created,
            model,
            "content",
            split.content,
            index=index,
            logprobs=logprobs,
            tokens=token_payload,
            stream_chunk=_buffered_delta_stream_chunk(
                split.content,
                stream_chunk,
                phase=content_phase,
                tokens=token_payload,
            ),
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
            routing=routing,
            phase=content_phase,
        )
    for tool_index, call in enumerate(parsed.tool_calls):
        for chunk_index, argument_chunk in enumerate(_tool_call_argument_stream_chunks(call.arguments)):
            token_payload = (
                token_accounting.observe("tool_call", argument_chunk)
                if token_accounting is not None
                else None
            )
            yield _chat_stream_tool_call(
                response_id,
                created,
                model,
                call,
                index=index,
                tool_index=tool_index,
                argument_chunk=argument_chunk,
                include_name=chunk_index == 0,
                tokens=token_payload,
                stream_chunk=_buffered_delta_stream_chunk(
                    argument_chunk,
                    stream_chunk,
                    phase="tool_call",
                    tokens=token_payload,
                ),
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing,
            )
    final_tokens = done_tokens
    if token_accounting is not None and token_accounting.streamed_tokens > 0:
        accounting_snapshot = token_accounting.snapshot()
        if final_tokens is None:
            final_tokens = accounting_snapshot
        else:
            final_tokens = {**dict(final_tokens), **accounting_snapshot}
    done_reason = "tool_calls" if parsed.tool_calls else finish_reason
    yield _chat_stream_done(
        response_id,
        created,
        model,
        done_reason,
        index=index,
        finish_details=finish_details,
        tokens=final_tokens,
        stream_chunk=stream_chunk,
        include_hipengine=include_hipengine,
        extra_hipengine=final_hipengine,
        stream_started_at=stream_started_at,
        routing=routing,
        kv_pool=kv_pool,
        phase=done_phase,
    )


def _scheduler_tool_call_argument_fragments(
    parsed: _ParsedChatOutput,
    raw_text: str,
    scheduler_chunks: Sequence[Mapping[str, Any]],
) -> tuple[_SchedulerToolArgumentFragment, ...]:
    if not parsed.tool_calls or parsed.text.strip():
        return ()
    chunk_ranges: list[tuple[int, int, Mapping[str, Any], GenerationStreamChunk]] = []
    pieces: list[str] = []
    cursor = 0
    for payload in scheduler_chunks:
        stream_chunk = _scheduler_payload_stream_chunk(payload)
        if stream_chunk is None:
            return ()
        text = stream_chunk.text
        pieces.append(text)
        next_cursor = cursor + len(text)
        chunk_ranges.append((cursor, next_cursor, payload, stream_chunk))
        cursor = next_cursor
    reconstructed = "".join(pieces)
    if reconstructed != str(raw_text):
        return ()

    fragments: list[_SchedulerToolArgumentFragment] = []
    search_start = 0
    for tool_index, call in enumerate(parsed.tool_calls):
        if not call.raw_text or not call.arguments:
            return ()
        call_start = reconstructed.find(call.raw_text, search_start)
        if call_start < 0:
            return ()
        call_end = call_start + len(call.raw_text)
        argument_offset = call.raw_text.find(call.arguments)
        if argument_offset < 0:
            return ()
        argument_start = call_start + argument_offset
        argument_end = argument_start + len(call.arguments)
        call_fragments: list[_SchedulerToolArgumentFragment] = []
        for chunk_start, chunk_end, _payload, stream_chunk in chunk_ranges:
            overlap_start = max(chunk_start, argument_start)
            overlap_end = min(chunk_end, argument_end)
            if overlap_start >= overlap_end:
                continue
            call_fragments.append(
                _SchedulerToolArgumentFragment(
                    tool_index=tool_index,
                    call=call,
                    text=reconstructed[overlap_start:overlap_end],
                    stream_chunk=stream_chunk,
                )
            )
        if "".join(fragment.text for fragment in call_fragments) != call.arguments:
            return ()
        fragments.extend(call_fragments)
        search_start = call_end
    return tuple(fragments)


def _chat_stream_scheduler_tool_call_chunks(
    response_id: str,
    created: int,
    model: str,
    fragments: Sequence[_SchedulerToolArgumentFragment],
    *,
    index: int = 0,
    finish_details: Mapping[str, Any] | None = None,
    token_accounting: _StreamTokenAccounting | None = None,
    final_stream_chunk: GenerationStreamChunk | None = None,
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
    kv_pool: Mapping[str, float] | None = None,
) -> Iterator[str]:
    last_stream_chunk: GenerationStreamChunk | None = None
    emitted_by_tool: set[int] = set()
    for fragment in fragments:
        last_stream_chunk = _stream_chunk_with_phase(fragment.stream_chunk, "tool_call")
        token_payload = (
            token_accounting.observe("tool_call", fragment.text)
            if token_accounting is not None
            else None
        )
        include_name = fragment.tool_index not in emitted_by_tool
        emitted_by_tool.add(fragment.tool_index)
        yield _chat_stream_tool_call(
            response_id,
            created,
            model,
            fragment.call,
            index=index,
            tool_index=fragment.tool_index,
            argument_chunk=fragment.text,
            include_name=include_name,
            tokens=token_payload,
            stream_chunk=last_stream_chunk,
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
            routing=routing,
        )
    final_tokens = None
    if token_accounting is not None and token_accounting.streamed_tokens > 0:
        final_tokens = token_accounting.snapshot()
    done_stream_chunk = last_stream_chunk
    if final_stream_chunk is not None:
        if done_stream_chunk is None:
            done_stream_chunk = final_stream_chunk
        elif final_stream_chunk.generated_token_ids is not None:
            done_stream_chunk = replace(
                done_stream_chunk,
                generated_token_ids=final_stream_chunk.generated_token_ids,
            )
    yield _chat_stream_done(
        response_id,
        created,
        model,
        "tool_calls",
        index=index,
        finish_details=finish_details,
        tokens=final_tokens,
        stream_chunk=done_stream_chunk,
        include_hipengine=include_hipengine,
        stream_started_at=stream_started_at,
        routing=routing,
        kv_pool=kv_pool,
        phase="tool_call",
    )


def _chat_stream_scheduler_text_chunks(
    response_id: str,
    created: int,
    model: str,
    scheduler_chunks: Sequence[Mapping[str, Any]],
    finish_reason: str,
    *,
    index: int = 0,
    include_logprobs: bool = False,
    content_phase: str = "answer",
    reasoning_initially_open: bool = False,
    finish_details: Mapping[str, Any] | None = None,
    token_accounting: _StreamTokenAccounting | None = None,
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
    kv_pool: Mapping[str, float] | None = None,
    done_phase: str = "done",
) -> Iterator[str]:
    splitter = _ReasoningSplitter(initially_open=reasoning_initially_open)
    last_stream_chunk: GenerationStreamChunk | None = None
    for raw_chunk in scheduler_chunks:
        stream_chunk = _scheduler_payload_stream_chunk(raw_chunk)
        if stream_chunk is None:
            continue
        last_stream_chunk = stream_chunk
        for delta_field, chunk in splitter.feed(stream_chunk.text):
            phase = "think" if delta_field == "reasoning_content" else content_phase
            token_payload = (
                token_accounting.observe(phase, chunk) if token_accounting is not None else None
            )
            logprobs = (
                _chat_stream_logprobs(stream_chunk, chunk)
                if include_logprobs and delta_field == "content"
                else None
            )
            reasoning_logprobs = (
                _chat_reasoning_stream_logprobs(stream_chunk, chunk)
                if include_logprobs and delta_field == "reasoning_content"
                else None
            )
            yield _chat_stream_delta(
                response_id,
                created,
                model,
                delta_field,
                chunk,
                index=index,
                logprobs=logprobs,
                reasoning_logprobs=reasoning_logprobs,
                tokens=token_payload,
                stream_chunk=_stream_chunk_with_phase(stream_chunk, phase),
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
                routing=routing,
                phase=phase,
            )
    for delta_field, chunk in splitter.finish():
        phase = "think" if delta_field == "reasoning_content" else content_phase
        token_payload = (
            token_accounting.observe(phase, chunk) if token_accounting is not None else None
        )
        logprobs = None
        reasoning_logprobs = None
        if include_logprobs and last_stream_chunk is not None:
            if delta_field == "content":
                logprobs = _chat_stream_logprobs(last_stream_chunk, chunk)
            elif delta_field == "reasoning_content":
                reasoning_logprobs = _chat_reasoning_stream_logprobs(last_stream_chunk, chunk)
        yield _chat_stream_delta(
            response_id,
            created,
            model,
            delta_field,
            chunk,
            index=index,
            logprobs=logprobs,
            reasoning_logprobs=reasoning_logprobs,
            tokens=token_payload,
            stream_chunk=(
                None if last_stream_chunk is None else _stream_chunk_with_phase(last_stream_chunk, phase)
            ),
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
            routing=routing,
            phase=phase,
        )
    final_tokens = None
    if token_accounting is not None and token_accounting.streamed_tokens > 0:
        final_tokens = token_accounting.snapshot()
    done_stream_chunk = (
        None if last_stream_chunk is None else _stream_chunk_with_phase(last_stream_chunk, done_phase)
    )
    yield _chat_stream_done(
        response_id,
        created,
        model,
        finish_reason,
        index=index,
        finish_details=finish_details,
        tokens=final_tokens,
        stream_chunk=done_stream_chunk,
        include_hipengine=include_hipengine,
        stream_started_at=stream_started_at,
        routing=routing,
        kv_pool=kv_pool,
        phase=done_phase,
    )


def _chat_stream_done(
    response_id: str,
    created: int,
    model: str,
    finish_reason: str,
    *,
    index: int = 0,
    finish_details: Mapping[str, Any] | None = None,
    tokens: Mapping[str, int] | None = None,
    stream_chunk: GenerationStreamChunk | None = None,
    include_hipengine: bool = False,
    extra_hipengine: Mapping[str, Any] | None = None,
    stream_started_at: _StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
    kv_pool: Mapping[str, float] | None = None,
    phase: str = "done",
) -> str:
    finish_payload = _finish_details_payload(None, finish_reason) if finish_details is None else dict(finish_details)
    choice = {
        "index": int(index),
        "delta": {},
        "finish_reason": finish_reason,
        "finish_details": finish_payload,
    }
    if include_hipengine:
        hipengine_payload = _choice_hipengine_payload(
            phase,
            finish_details=finish_payload,
            tokens=tokens,
            stream_chunk=stream_chunk,
            include_generated_token_ids=True,
        )
        if extra_hipengine is not None:
            hipengine_payload.update(deepcopy(dict(extra_hipengine)))
        choice["hipengine"] = hipengine_payload
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [choice],
            },
            include_hipengine=include_hipengine,
            event="done",
            stream_started_at=stream_started_at,
            tokens=tokens,
            backend_timing=_stream_chunk_backend_timing(stream_chunk),
            routing=routing,
            kv_pool=kv_pool,
        )
    )


def _chat_stream_usage(
    response_id: str,
    created: int,
    model: str,
    usage: Mapping[str, int],
    *,
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
    kv_pool: Mapping[str, float] | None = None,
) -> str:
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": dict(usage),
            },
            include_hipengine=include_hipengine,
            event="usage",
            stream_started_at=stream_started_at,
            usage=usage,
            routing=routing,
            kv_pool=kv_pool,
        )
    )


def _chat_stream_error(
    response_id: str,
    created: int,
    model: str,
    message: str,
    *,
    status_code: int = 500,
    code: str | None = None,
    param: str | None = None,
    error_type: str = "server_error",
    finish_details: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
) -> str:
    choice: dict[str, Any] = {
        "index": 0,
        "delta": {"content": ""},
        "finish_reason": "error",
    }
    error = _error_payload(
        message=message,
        error_type=error_type,
        code=code,
        param=param,
        status_code=status_code,
        finish_details=finish_details,
        extra=extra,
    )
    if finish_details is not None:
        details = dict(finish_details)
        choice["finish_details"] = details
    if include_hipengine:
        choice["hipengine"] = _choice_hipengine_payload(
            "done",
            finish_details=choice.get("finish_details"),
        )
    return _sse(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [choice],
                "error": error,
            },
            include_hipengine=include_hipengine,
            event="error",
            stream_started_at=stream_started_at,
            routing=routing,
        )
    )


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _format_validation_error(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "invalid request"
    first = errors[0]
    loc = _validation_error_location(first.get("loc", ()))
    msg = str(first.get("msg", "invalid value"))
    return f"{loc}: {msg}" if loc else msg


def _validation_error_param(exc: RequestValidationError) -> str | None:
    for error in exc.errors():
        loc = _validation_error_location(error.get("loc", ()))
        if loc:
            return loc
    return None


def _validation_error_location(value: Any) -> str | None:
    if not isinstance(value, (list, tuple)):
        return None
    parts = [str(item) for item in value if item != "body"]
    return ".".join(parts) or None
