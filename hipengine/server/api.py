"""OpenAI-compatible FastAPI surface for hipEngine.

The server layer is optional and intentionally thin: it adapts OpenAI-style JSON
requests to the torch-free ``hipengine.LLM.generate()`` library API.  HTTP
requests are routed through the generation batcher; the remaining async lock is
limited to short model/session preparation mutations.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
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
    GenerationCancellationToken,
    GenerationCancelled,
    GenerationDeadlineExceeded,
    GenerationOutput,
    SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS,
    SPECULATIVE_MTP_INCOMPATIBLE_FIELDS,
    ThinkingBudgetState,
    TokenLogprob,
    derive_row_seed,
)
from hipengine.kvcache import resolve_prefix_cache_mode


_LOGGER = logging.getLogger("uvicorn.error")
_GRAPH_KERNEL_TIME_HISTOGRAM_BUCKET_SET = frozenset(GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS)
_THINKING_CLOSE_MARKER = "</think>"
_TOOL_CALL_START_MARKER = "<tool_call>"
_TOOL_CALL_END_MARKER = "</tool_call>"
_CONTINUATION_TTL_SECONDS = 15 * 60
_SESSION_COMMIT_MODES = (
    "append_none",
    "append_prompt_only",
    "append_visible_only",
    "append_all",
)
_SESSION_STATEFUL_DEFAULT_COMMIT = "append_visible_only"
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
    "chat_tools",
    "thinking_budget_controls",
    "session_id",
)
_CONTINUATION_UNSUPPORTED_RESUME_FIELDS = (
    "stream",
    "n",
    "logprobs",
    "echo",
    "temperature",
    "logit_bias",
    "suppress_token_ids",
    "min_tokens",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "thinking",
)
_TOOL_RESULT_VALIDATION_FAILURE_REASONS = (
    "invalid_tool_call",
    "tool_required_not_satisfied",
    "schema_violation",
)
_STRUCTURED_OUTPUT_RESULT_VALIDATION_FAILURE_REASONS = ("schema_violation",)
_JSON_SCHEMA_ANNOTATION_KEYWORDS = (
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
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
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
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
_UNSUPPORTED_GRAMMAR_FIELDS = (
    "grammar",
    "guided_json",
    "guided_regex",
    "guided_choice",
    "guided_grammar",
    "guided_decoding_backend",
    "guided_patch",
    "guided_diff",
)


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for the optional OpenAI-compatible server."""

    model: str
    backend: str = "auto"
    quant: str = "w4_paro"
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
    created: int = field(default_factory=lambda: int(time.time()))

    @property
    def model_id(self) -> str:
        """Public model identifier exposed through the OpenAI API."""

        if self.served_model_name:
            return self.served_model_name
        path = Path(self.model)
        if path.exists() and path.name:
            return path.name
        return self.model


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


_ERROR_TAXONOMY: dict[str, dict[str, Any]] = {
    "unsupported_parameter": {
        "status_code": 400,
        "retryable": False,
        "emitted": True,
        "description": "A request field or value is not supported by this server.",
    },
    "invalid_tool_call": {
        "status_code": 400,
        "retryable": False,
        "emitted": True,
        "description": (
            "Emitted as finish_details.reason for strict tool result-validation failures; "
            "reserved as an HTTP error for future strict decode-time failures."
        ),
    },
    "schema_violation": {
        "status_code": 422,
        "retryable": False,
        "emitted": True,
        "description": "A request, response_format result, or strict tool result schema was violated.",
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
        payload.update(dict(extra))
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
        "model": {
            "id": config.model_id,
            "backend": config.backend,
            "quant": config.quant,
        },
        "sampling": _replay_sampling_payload(body_json),
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


def _replay_sampling_payload(body_json: Any) -> dict[str, Any]:
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
        "hard_close_sequence",
        "logprobs",
        "top_logprobs",
        "response_format",
        "thinking_token_budget",
    )
    return {key: body_json[key] for key in keys if key in body_json}


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
    ]


def _tools_capability(*, tokenizer_backed: bool) -> dict[str, Any]:
    return {
        "enabled": True,
        "strict_decoding": False,
        "strict_result_validation": True,
        "result_validation_failure_reasons": list(_TOOL_RESULT_VALIDATION_FAILURE_REASONS),
        "schema_validation": "function_strict",
        "schema_subset": _tool_schema_subset(),
        "unsupported_schema_keywords_rejected": True,
        "annotation_keywords_ignored": list(_JSON_SCHEMA_ANNOTATION_KEYWORDS),
        "format": "qwen_tool_call_json",
        "parallel_tool_calls": True,
        "no_tool_start_suppression": tokenizer_backed,
        "required_tool_start_forcing": tokenizer_backed,
        "required_tool_start_forcing_scope": (
            "initial_or_after_tokenized_thinking_close" if tokenizer_backed else "none"
        ),
        "specific_tool_name_prefix_forcing": tokenizer_backed,
        "tool_call_close_repair": tokenizer_backed,
    }


def _structured_outputs_capability() -> dict[str, Any]:
    return {
        "response_format": True,
        "json_object": True,
        "json_schema": True,
        "strict_decoding": False,
        "strict_result_validation": True,
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
        "result_validation_only": ["json_object", "json_schema"],
    }


def _known_unsupported_fields() -> list[str]:
    return list(_UNSUPPORTED_GRAMMAR_FIELDS)


def _admission_capability(config: ServerConfig) -> dict[str, Any]:
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
        "scheduler_fairness": _scheduler_fairness_capability(),
    }


def _scheduler_fairness_capability() -> dict[str, Any]:
    return {
        "policy": _GENERATION_SCHEDULER_FAIRNESS_POLICY,
        "compatible_sampling_coalescing": True,
        "continuous_decode": False,
        "preemptive_fairness": False,
    }


def _replay_capability_snapshot(config: ServerConfig, *, engine: Any | None = None) -> dict[str, Any]:
    tokenizer_caps = _tokenizer_capability_flags(engine)
    tokenizer_backed = tokenizer_caps["tokenize"]
    return {
        "model": {
            "id": config.model_id,
            "backend": config.backend,
            "quant": config.quant,
        },
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
            "structured_outputs": _structured_outputs_capability(),
            "grammars": _grammar_capability(),
            "tools": _tools_capability(tokenizer_backed=tokenizer_backed),
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
                "seed",
                "n",
                "stop",
            ],
            "speculative_mtp": {
                "serving_route": False,
                "sampling_compatible": False,
                "compatibility_guard": "supports_speculative_mtp_sampling",
                "allowed_execution_modes": ["greedy_fast"],
                "incompatible_fields": list(SPECULATIVE_MTP_INCOMPATIBLE_FIELDS),
                "incompatible_conditions": dict(SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS),
                "processed_target_verification": False,
            },
        },
        "cache": {
            "prefix_cache": config.prefix_cache,
            "kv_storage": config.kv_storage,
            "kv_scale_dtype": config.kv_scale_dtype,
            "kv_scale_granularity": config.kv_scale_granularity,
        },
        "sessions": {
            "resident_context": True,
            "commit_policy": _session_commit_policy_capability(),
            "continuations": _session_continuation_capability(),
            "metadata": _session_metadata_capability(config.max_chat_sessions),
        },
        "admission": _admission_capability(config),
        "unsupported_fields": _known_unsupported_fields(),
    }


def _session_commit_policy_capability() -> dict[str, Any]:
    return {
        "supported": True,
        "stateful": True,
        "resident_state_reuse": False,
        "storage": "app_local_transcript",
        "default": "append_none",
        "stateful_default": _SESSION_STATEFUL_DEFAULT_COMMIT,
        "modes": list(_SESSION_COMMIT_MODES),
        "supported_endpoints": ["chat_completions"],
        "supported_streaming": False,
        "downgrade_visible_only_on": sorted(_SESSION_UNSAFE_VISIBLE_REASONS),
    }


def _session_continuation_capability() -> dict[str, Any]:
    return {
        "supported": True,
        "stateful": False,
        "resident_state_reuse": False,
        "single_use": True,
        "ttl_seconds": _CONTINUATION_TTL_SECONDS,
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
        "max_active": None if max_active is None else int(max_active),
        "list_endpoint": "/v1/hipengine/sessions",
        "delete_endpoint": "/v1/hipengine/sessions/{session_id}",
    }


def _choice_telemetry_capability() -> dict[str, Any]:
    return {
        "non_streaming": True,
        "streaming": "stream_options.include_hipengine",
        "decode_state": True,
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
    prompt: str | list[str] | None = None
    max_tokens: int | None = Field(default=16, ge=0)
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
    continuation_id: Any | None = None
    session: Any | None = None


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
    continuation_id: Any | None = None
    session: Any | None = None


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
    session: Any | None = None


class FitContextRequest(TokenDiagnosticRequest):
    max_tokens: int | None = Field(default=None, ge=0)


@dataclass(frozen=True)
class _GeneratedBatch:
    outputs: list[str]
    usage: dict[str, int]
    details: list[GenerationOutput]


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

    def record_success(self, usage: Mapping[str, int]) -> None:
        self.request_total += 1
        self.request_completed_total += 1
        self.prompt_tokens_total += int(usage.get("prompt_tokens", 0))
        self.completion_tokens_total += int(usage.get("completion_tokens", 0))

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


@dataclass
class _QueuedGeneration:
    prompts: tuple[str, ...]
    sampling: SamplingParams
    future: asyncio.Future[list[Any]] | None = None
    stream_queue: asyncio.Queue[object] | None = None
    detailed: bool = False
    cancelled: bool = False


class _GenerationBatcher:
    """Coalesce compatible HTTP generations into prompt-list calls."""

    def __init__(
        self,
        *,
        engine_factory: Callable[[], Any],
        batch_window_seconds: float,
        max_queue_size: int | None = None,
        max_active_requests: int | None = None,
        retry_after_seconds: int = 1,
    ) -> None:
        self._engine_factory = engine_factory
        self._batch_window_seconds = max(0.0, float(batch_window_seconds))
        self._max_queue_size = None if max_queue_size is None else int(max_queue_size)
        if self._max_queue_size is not None and self._max_queue_size < 1:
            raise ValueError("max_queue_size must be positive when set")
        self._max_active_requests = None if max_active_requests is None else int(max_active_requests)
        if self._max_active_requests is not None and self._max_active_requests < 1:
            raise ValueError("max_active_requests must be positive when set")
        self._retry_after_seconds = max(1, int(retry_after_seconds))
        self._queue: deque[_QueuedGeneration] = deque()
        self._worker: asyncio.Task[None] | None = None
        self._active_requests = 0

    def queue_depth(self) -> int:
        return len(self._queue)

    def max_queue_size(self) -> int | None:
        return self._max_queue_size

    def active_requests(self) -> int:
        return self._active_requests

    def max_active_requests(self) -> int | None:
        return self._max_active_requests

    def active(self) -> bool:
        return self._worker is not None and not self._worker.done()

    def _group_has_capacity(self, group: Sequence[_QueuedGeneration]) -> bool:
        return self._max_active_requests is None or len(group) < self._max_active_requests

    def _raise_if_full(self) -> None:
        if self._max_queue_size is None:
            return
        if len(self._queue) < self._max_queue_size:
            return
        raise OpenAIHTTPError(
            429,
            "generation queue is full",
            error_type="rate_limit_error",
            code="engine_busy",
            headers={"Retry-After": str(self._retry_after_seconds)},
        )

    async def submit(
        self,
        prompts: Sequence[str],
        sampling: SamplingParams,
        *,
        detailed: bool = False,
    ) -> list[Any]:
        prompt_tuple = tuple(str(prompt) for prompt in prompts)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[Any]] = loop.create_future()
        self._raise_if_full()
        self._queue.append(
            _QueuedGeneration(
                prompts=prompt_tuple,
                sampling=sampling,
                future=future,
                detailed=bool(detailed),
            )
        )
        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._run())
        return await future

    async def stream(self, prompts: Sequence[str], sampling: SamplingParams) -> AsyncIterator[str]:
        """Yield generated text through a per-request queue owned by the batcher."""

        prompt_tuple = tuple(str(prompt) for prompt in prompts)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[object] = asyncio.Queue()
        self._raise_if_full()
        item = _QueuedGeneration(
            prompts=prompt_tuple,
            sampling=sampling,
            stream_queue=queue,
        )
        self._queue.append(item)
        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._run())
        try:
            while True:
                event = await queue.get()
                if event is _STREAM_DONE:
                    break
                if isinstance(event, BaseException):
                    raise event
                yield str(event)
        finally:
            item.cancelled = True

    async def _run(self) -> None:
        try:
            if self._batch_window_seconds > 0.0:
                await asyncio.sleep(self._batch_window_seconds)
            while self._queue:
                first = self._queue.popleft()
                if _queued_generation_cancelled(first):
                    continue
                key = _sampling_key(first.sampling)
                group = [first]
                deferred: deque[_QueuedGeneration] = deque()
                while self._queue:
                    item = self._queue.popleft()
                    if _queued_generation_cancelled(item):
                        continue
                    if _sampling_key(item.sampling) == key and self._group_has_capacity(group):
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

    async def _run_group(self, group: Sequence[_QueuedGeneration]) -> None:
        if not group:
            return
        try:
            self._active_requests = len(group)
            if len(group) == 1 and group[0].stream_queue is not None and len(group[0].prompts) == 1:
                await self._stream_single(group[0])
                return
            prompts: list[str] = []
            slices: list[tuple[_QueuedGeneration, int, int]] = []
            for item in group:
                start = len(prompts)
                prompts.extend(item.prompts)
                slices.append((item, start, len(prompts)))
            try:
                outputs = await self._generate_prompts(tuple(prompts), group[0].sampling)
            except Exception as exc:
                for item in group:
                    _finish_queued_generation(item, exception=exc)
                return
            for item, start, end in slices:
                item_outputs: Sequence[Any] = outputs[start:end]
                if not item.detailed:
                    item_outputs = [_coerce_generation_output(output).text for output in item_outputs]
                _finish_queued_generation(item, outputs=item_outputs)
        finally:
            self._active_requests = 0

    async def _generate_prompts(self, prompts: tuple[str, ...], sampling: SamplingParams) -> list[Any]:
        raw_outputs = await _generate_detailed(self._engine_factory(), prompts, sampling)
        outputs = list(raw_outputs)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"generator returned {len(outputs)} outputs for {len(prompts)} prompts"
            )
        return outputs

    async def _stream_single(self, item: _QueuedGeneration) -> None:
        assert item.stream_queue is not None
        try:
            async for chunk in _stream_engine_text(self._engine_factory(), item.prompts[0], item.sampling):
                if _queued_generation_cancelled(item):
                    break
                item.stream_queue.put_nowait(str(chunk))
        except Exception as exc:
            _finish_queued_generation(item, exception=exc)
            return
        _finish_queued_generation(item, outputs=())


def _queued_generation_cancelled(item: _QueuedGeneration) -> bool:
    return item.cancelled or (item.future is not None and item.future.cancelled())


def _finish_queued_generation(
    item: _QueuedGeneration,
    *,
    outputs: Sequence[Any] | None = None,
    exception: Exception | None = None,
) -> None:
    if item.future is not None and not item.future.done():
        if exception is not None:
            item.future.set_exception(exception)
        else:
            item.future.set_result(list(outputs or ()))
    if item.stream_queue is None:
        return
    if exception is not None:
        item.stream_queue.put_nowait(exception)
    else:
        for output in outputs or ():
            item.stream_queue.put_nowait(str(output))
    item.stream_queue.put_nowait(_STREAM_DONE)


async def _stream_engine_text(engine: Any, prompt: str, sampling: SamplingParams) -> AsyncIterator[str]:
    streamer = getattr(engine, "stream", None)
    if not callable(streamer):
        for output in await run_in_threadpool(engine.generate, (prompt,), sampling):
            yield str(output)
        return
    iterator = iter(streamer(prompt, sampling))
    done = False
    try:
        while True:
            item = await run_in_threadpool(_next_stream_item, iterator)
            if item is _STREAM_DONE:
                done = True
                break
            yield str(item)
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


@dataclass(frozen=True)
class _ToolValidationResult:
    parsed: _ParsedChatOutput
    failure_reason: str | None = None

    @property
    def failed(self) -> bool:
        return self.failure_reason is not None


@dataclass(frozen=True)
class _ThinkingControl:
    enabled: bool | None = None
    effort: str | None = None
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
    prompts: tuple[str, ...]
    generated_texts: tuple[str, ...]
    created: float
    expires_at: float
    response_format: Any | None = None

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

    async def reserve_chat_session_if_needed(request: ChatCompletionRequest) -> str | None:
        session_id = _session_id(request)
        if session_id is None:
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
            cleanup_expired_continuations()
            return continuations.pop(continuation_id)

    async def store_continuation(
        *,
        endpoint: str,
        prompts: Sequence[str],
        generated_texts: Sequence[str],
        response_format: Any | None,
    ) -> _ContinuationRecord:
        now = time.time()
        record = _ContinuationRecord(
            id=f"gen_{uuid.uuid4().hex}",
            endpoint=endpoint,
            model_id=config.model_id,
            prompts=tuple(str(prompt) for prompt in prompts),
            generated_texts=tuple(str(text) for text in generated_texts),
            created=now,
            expires_at=now + _CONTINUATION_TTL_SECONDS,
            response_format=response_format,
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
            return tuple(dict(message) for message in record.messages)

    async def commit_chat_session(
        request: ChatCompletionRequest,
        *,
        request_messages: Sequence[ChatMessage | Mapping[str, Any]],
        raw_output: str,
        visible_message: Mapping[str, Any],
        cache_action: str | None,
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
            created = now if previous is None else previous.created
            base = () if previous is None else previous.messages
            chat_sessions[session_id] = _ChatSessionRecord(
                id=session_id,
                messages=(*base, *appended),
                created=created,
                updated=now,
            )

    async def diagnostic_render_for_request(
        request: TokenDiagnosticRequest,
        engine: Any,
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
        unsupported_param = _unsupported_agentic_request_param(chat_request)
        if unsupported_param is not None:
            raise OpenAIHTTPError(
                400,
                f"{unsupported_param} is not supported by this server",
                code="unsupported_parameter",
                param=unsupported_param,
            )
        prefix_messages = await chat_session_prefix_messages(chat_request)
        render_request = chat_request
        if prefix_messages:
            render_request = _chat_request_with_messages(
                chat_request,
                (*prefix_messages, *chat_request.messages),
            )
        return {
            "text": chat_prompt_for_request(render_request, engine),
            "input_type": "chat",
            "chat_request": render_request,
            "session": _diagnostic_session_payload(chat_request, prefix_messages),
        }

    def get_llm() -> Any:
        if app.state.hipengine_llm is None:
            app.state.hipengine_llm = LLM(config.model, backend=config.backend, quant=config.quant)
        app.state.hipengine_readiness.model_loaded = True
        return app.state.hipengine_llm

    generation_batcher = _GenerationBatcher(
        engine_factory=get_llm,
        batch_window_seconds=float(config.generation_batch_window_ms) / 1000.0,
        max_queue_size=config.max_queued_requests,
        max_active_requests=config.max_active_requests,
        retry_after_seconds=config.queue_retry_after_seconds,
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
            engine = get_llm()
            engine_create_s = time.perf_counter() - engine_started
            prepare_started = time.perf_counter()
            max_context = await ensure_resident_context(engine, sampling, phase="startup")
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
        await run_in_threadpool(engine.generate, (config.eager_load_prompt,), sampling)
        warmup_s = time.perf_counter() - warmup_started
        startup_checks["raw_warmup"] = {"max_tokens": max_tokens}
        _record_startup_memory_snapshot(startup_memory, "after_raw_warmup")

        scratch_probe_s: float | None = None
        scratch_probe_started = time.perf_counter()
        max_prompt_tokens = _startup_max_prompt_tokens(max_context)
        if config.startup_scratch_probe:
            scratch_preparer = getattr(engine, "prepare_request_scratch", None)
            if max_prompt_tokens is None:
                startup_checks["scratch_probe"] = {"enabled": True, "status": "skipped", "reason": "unknown_context"}
            elif not callable(scratch_preparer):
                startup_checks["scratch_probe"] = {"enabled": True, "status": "skipped", "reason": "backend_hook_unavailable"}
                _LOGGER.warning("STARTUP_SCRATCH_PROBE: skipped; backend does not expose prepare_request_scratch")
            else:
                _LOGGER.info(
                    "STARTUP_SCRATCH_PROBE: max_prompt_tokens=%d max_new_tokens=0 max_batch_size=1 release_after_probe=True",
                    max_prompt_tokens,
                )
                try:
                    scratch_result = await run_in_threadpool(
                        lambda: scratch_preparer(
                            max_prompt_tokens=max_prompt_tokens,
                            max_new_tokens=0,
                            sampling_params=sampling,
                            max_batch_size=1,
                            release_after_probe=True,
                        )
                    )
                except Exception as exc:
                    startup_checks["scratch_probe"] = {
                        "enabled": True,
                        "status": "failed",
                        "max_prompt_tokens": max_prompt_tokens,
                        "error": str(exc),
                    }
                    _LOGGER.error(
                        "STARTUP_SCRATCH_PROBE: failed at max_prompt_tokens=%d: %s. "
                        "Try a lower --max-context-tokens or a higher scratch/headroom reserve.",
                        max_prompt_tokens,
                        exc,
                    )
                    raise
                startup_checks["scratch_probe"] = {
                    "enabled": True,
                    "status": "passed",
                    "max_prompt_tokens": max_prompt_tokens,
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
        _LOGGER.info("hipEngine is ready.")

    if hasattr(app, "add_event_handler"):
        app.add_event_handler("startup", eager_load_model)
    else:  # FastAPI-lite compatibility in minimal test/runtime environments.
        app.router.on_startup.append(eager_load_model)

    async def require_auth(request: Request) -> None:
        if not config.api_key:
            return
        expected = f"Bearer {config.api_key}"
        if request.headers.get("authorization") != expected:
            raise OpenAIHTTPError(
                401,
                "missing or invalid bearer token",
                error_type="authentication_error",
                code="invalid_api_key",
            )

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
        prompts: Sequence[str],
        engine: Any,
        *,
        deadline_at: float | None = None,
        cancellation_token: GenerationCancellationToken | None = None,
    ) -> SamplingParams:
        stop_token_ids, stop_token_sequences = _stop_tokens_from_stop(request.stop, engine)
        uses_stored_chat_prompt = isinstance(request, ChatCompletionRequest) and request.continuation_id is not None
        thinking_budget = (
            _thinking_budget_sampling_kwargs(
                config,
                request,
                engine=engine,
                max_context_tokens=effective_max_context_tokens(engine),
            )
            if isinstance(request, ChatCompletionRequest) and not uses_stored_chat_prompt
            else {}
        )
        no_tool_suppress_token_ids = (
            _no_tool_sampling_suppress_token_ids(request, engine)
            if isinstance(request, ChatCompletionRequest) and not uses_stored_chat_prompt
            else ()
        )
        forced_tool_token_ids = (
            _required_tool_sampling_forced_token_ids(request, engine)
            if isinstance(request, ChatCompletionRequest) and not uses_stored_chat_prompt
            else ()
        )
        force_sequence_completion_token_sequences = (
            _tool_call_sequence_completion_token_sequences(request, engine, forced_tool_token_ids)
            if isinstance(request, ChatCompletionRequest)
            else ()
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
            ignore_eos=bool(request.ignore_eos),
            kv_storage=request.kv_storage or config.kv_storage,
            kv_scale_dtype=request.kv_scale_dtype or config.kv_scale_dtype,
            kv_scale_granularity=request.kv_scale_granularity or config.kv_scale_granularity,
            seed=request.seed,
            deadline_at=deadline_at,
            cancellation_token=cancellation_token,
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
        prompts: Sequence[str],
        request: CompletionRequest | ChatCompletionRequest,
        *,
        deadline_at: float | None = None,
        cancellation_token: GenerationCancellationToken | None = None,
    ) -> _GeneratedBatch:
        try:
            _validate_generation_request(config, request)
            async with session_lock:
                engine = get_llm()
                await ensure_resident_context(engine, preparation_sampling(request), phase="preparation")
                sampling = sampling_params(
                    request,
                    prompts,
                    engine,
                    deadline_at=deadline_at,
                    cancellation_token=cancellation_token,
                )
                if _request_n(request) > 1:
                    sampling = replace(
                        sampling,
                        row_seeds=_row_seeds_for_request(request.seed, len(prompts)),
                    )
                _validate_context_budget(effective_max_context_tokens(engine), engine, prompts, sampling)
            if _request_logprobs_enabled(request):
                raw_outputs = await _generate_detailed(engine, tuple(prompts), sampling)
            else:
                raw_outputs = await generation_batcher.submit(tuple(prompts), sampling, detailed=True)
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
        batch = _GeneratedBatch(outputs=outputs, usage=_usage(engine, prompts, outputs), details=details)
        app.state.hipengine_server_metrics.record_success(batch.usage)
        return batch

    async def generate_with_request_control(
        prompts: Sequence[str],
        request: CompletionRequest | ChatCompletionRequest,
        control: _RequestControl | None = None,
    ) -> _GeneratedBatch:
        active_control = control or _request_control(config, request)
        try:
            return await _await_with_request_control(
                generate(
                    prompts,
                    request,
                    deadline_at=active_control.deadline_at,
                    cancellation_token=active_control.cancellation_token,
                ),
                active_control,
            )
        except OpenAIHTTPError as exc:
            if exc.code in {"deadline_exceeded", "cancelled"}:
                _record_openai_error(app.state.hipengine_server_metrics, exc)
            raise

    async def stream_completion_one(
        prompt: str,
        request: CompletionRequest,
        control: _RequestControl,
    ) -> AsyncIterator[str]:
        response_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        include_hipengine = _stream_include_hipengine(request)
        stream_started_at = _StreamTimingTracker.start() if include_hipengine else time.perf_counter()
        full_text: list[str] = []
        try:
            _validate_generation_request(config, request)

            async def prepare_stream() -> tuple[Any, SamplingParams]:
                async with session_lock:
                    engine = get_llm()
                    await ensure_resident_context(engine, preparation_sampling(request), phase="preparation")
                    sampling = sampling_params(
                        request,
                        (prompt,),
                        engine,
                        deadline_at=control.deadline_at,
                        cancellation_token=control.cancellation_token,
                    )
                    _validate_context_budget(effective_max_context_tokens(engine), engine, (prompt,), sampling)
                    return engine, sampling

            engine, sampling = await _await_with_request_control(prepare_stream(), control)
            token_accounting = _StreamTokenAccounting.for_engine(engine) if include_hipengine else None
            async for token in _iterate_with_request_control(generation_batcher.stream((prompt,), sampling), control):
                text = str(token)
                if not text:
                    continue
                full_text.append(text)
                token_payload = token_accounting.observe("answer", text) if token_accounting is not None else None
                yield _completion_stream_delta(
                    response_id,
                    created,
                    config.model_id,
                    text,
                    tokens=token_payload,
                    include_hipengine=include_hipengine,
                    stream_started_at=stream_started_at,
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
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
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
            yield _completion_stream_error(
                response_id,
                created,
                config.model_id,
                message,
                status_code=500,
                code="generation_failed",
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
            )
            yield "data: [DONE]\n\n"
            return

        raw_text = "".join(full_text)
        text, finish_reason = _apply_stop(raw_text, request.stop)
        usage = _usage(engine, (prompt,), [text])
        app.state.hipengine_server_metrics.record_success(usage)
        cache_action = _session_cache_action(request)
        final_tokens = (
            _stream_usage_token_payload(usage, token_accounting)
            if include_hipengine and token_accounting is not None
            else None
        )
        yield _completion_stream_done(
            response_id,
            created,
            config.model_id,
            finish_reason,
            finish_details=_finish_details_payload(
                None,
                finish_reason,
                cache_action=cache_action,
            ),
            tokens=final_tokens,
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
        )
        if _stream_include_usage(request):
            yield _completion_stream_usage(
                response_id,
                created,
                config.model_id,
                usage,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
            )
        yield "data: [DONE]\n\n"

    def readiness_payload() -> dict[str, Any]:
        engine = getattr(app.state, "hipengine_llm", None)
        readiness: _ReadinessState = app.state.hipengine_readiness
        effective_context = getattr(app.state, "hipengine_effective_max_context_tokens", None)
        graph = _graph_bucket_metric_values(engine)
        pool = _pool_metric_values(engine)
        diagnostics: list[str] = []
        if not readiness.ready:
            diagnostics.append("server startup is not ready; check startup.error and server logs")
        return {
            "object": "hipengine.readiness",
            "status": readiness.status,
            "ready": bool(readiness.ready),
            "diagnostics": diagnostics,
            "model": {
                "id": config.model_id,
                "backend": config.backend,
                "quant": config.quant,
                "loaded": bool(readiness.model_loaded),
                "loaded_model_count": 0 if engine is None else 1,
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
            "graph_cache": {
                "entries": graph["entries"],
                "hits": graph["hits"],
                "misses": graph["misses"],
                "replay_hit_rate": graph["replay_hit_rate"],
            },
            "device": _selected_device_payload(config),
            "queue": {
                "depth": generation_batcher.queue_depth(),
                "max_depth": generation_batcher.max_queue_size(),
                "worker_active": generation_batcher.active(),
                "active_requests": generation_batcher.active_requests(),
                "max_active_requests": generation_batcher.max_active_requests(),
                "scheduler_fairness": _scheduler_fairness_capability(),
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
                ),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

    @app.get("/v1/models")
    async def list_models(_auth: None = Depends(require_auth)) -> dict[str, Any]:
        engine = getattr(app.state, "hipengine_llm", None)
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
                        "backend": config.backend,
                        "quant": config.quant,
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
                        },
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
        tokenizer_caps = _tokenizer_capability_flags(engine)
        configured_context = configured_max_context_tokens()
        cached_context = getattr(app.state, "hipengine_effective_max_context_tokens", None)
        effective_context = configured_context if configured_context is not None else cached_context
        return {
            "object": "hipengine.capabilities",
            "model": {
                "id": config.model_id,
                "path": config.model,
                "backend": config.backend,
                "quant": config.quant,
            },
            "context": {
                "configured_max_context_tokens": configured_context,
                "effective_max_context_tokens": None if effective_context is None else int(effective_context),
                "chat_default_max_tokens": config.chat_default_max_tokens,
                "chat_default_mode": "auto" if config.chat_default_max_tokens is None else "bounded",
            },
            "tokenizer": {
                "tokenize": tokenizer_caps["tokenize"],
                "detokenize": tokenizer_caps["detokenize"],
                "count_tokens": tokenizer_caps["count_tokens"],
                "name": None if engine is None else type(engine).__name__,
            },
            "chat_template": {
                "family": "qwen",
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
                    ],
                    "server_wall_timing": True,
                    "backend_prefill_timing": False,
                    "choice_phase": True,
                    "choice_finish_details": True,
                    "choice_token_accounting": tokenizer_caps["count_tokens"],
                    "choice_decode_state": tokenizer_caps["count_tokens"],
                },
                "choice_telemetry": _choice_telemetry_capability(),
                "structured_outputs": _structured_outputs_capability(),
                "grammars": _grammar_capability(),
                "finish_details": True,
                "token_diagnostics": {
                    "tokenize": tokenizer_caps["tokenize"],
                    "detokenize": tokenizer_caps["detokenize"],
                    "count_tokens": tokenizer_caps["count_tokens"],
                    "fit_context": tokenizer_caps["count_tokens"],
                    "session_aware_chat": tokenizer_caps["count_tokens"],
                },
                "tools": _tools_capability(tokenizer_backed=tokenizer_caps["tokenize"]),
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
                    "streaming": "buffered",
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
                    "seed",
                    "n",
                    "stop",
                ],
                "native_gpu": {
                    "enabled": _env_flag("HIPENGINE_QWEN35_NATIVE_SAMPLER"),
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
                },
                "speculative_mtp": {
                    "serving_route": False,
                    "sampling_compatible": False,
                    "compatibility_guard": "supports_speculative_mtp_sampling",
                    "allowed_execution_modes": ["greedy_fast"],
                    "incompatible_fields": list(SPECULATIVE_MTP_INCOMPATIBLE_FIELDS),
                    "incompatible_conditions": dict(SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS),
                    "processed_target_verification": False,
                },
            },
            "cache": {
                "prefix_cache": prefix_cache_mode,
                "kv_storage": config.kv_storage,
                "kv_scale_dtype": config.kv_scale_dtype,
                "kv_scale_granularity": config.kv_scale_granularity,
            },
            "sessions": {
                "resident_context": True,
                "commit_policy": _session_commit_policy_capability(),
                "continuations": _session_continuation_capability(),
                "metadata": _session_metadata_capability(config.max_chat_sessions),
            },
            "admission": _admission_capability(config),
            "routing": {
                "loaded_model_count": 0 if engine is None else 1,
                "multiple_models": False,
            },
            "errors": _error_taxonomy_manifest(),
            "unsupported_fields": _known_unsupported_fields(),
        }

    @app.post("/v1/hipengine/tokenize")
    async def tokenize(request: TokenizeRequest, _auth: None = Depends(require_auth)) -> dict[str, Any]:
        engine = get_llm()
        token_ids = await run_in_threadpool(_tokenize_text, engine, request.text)
        return {
            "object": "hipengine.tokens",
            "text": request.text,
            "token_ids": list(token_ids),
            "token_count": len(token_ids),
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
        token_count = await run_in_threadpool(_count_tokens_strict, engine, text)
        response = {
            "object": "hipengine.count_tokens",
            "input_type": input_type,
            "text": text,
            "token_count": token_count,
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
        rendered = await diagnostic_render_for_request(request, engine)
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

    @app.post("/v1/completions", response_model=None)
    async def completions(
        request: CompletionRequest,
        raw_request: Request,
        _auth: None = Depends(require_auth),
    ) -> dict[str, Any] | StreamingResponse:
        _validate_model(config, request.model)
        _validate_generation_request(config, request)
        _validate_continuation_resume_request(request)
        continuation = await pop_continuation(request, endpoint="completion")
        _apply_continuation_defaults(request, continuation)
        prompts = continuation.resume_prompts() if continuation is not None else _normalize_prompts(request.prompt)
        n = _request_n(request)
        expanded_prompts = _expand_prompts_for_n(prompts, n)
        control = _request_control(config, request, raw_request)
        if (
            request.stream
            and len(expanded_prompts) == 1
            and not request.echo
            and not _request_logprobs_enabled(request)
            and not _response_format_result_validation(request)
        ):
            return StreamingResponse(
                stream_completion_one(expanded_prompts[0], request, control),
                media_type="text/event-stream",
            )
        batch = await generate_with_request_control(expanded_prompts, request, control)
        response_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        choices = []
        final_texts: list[str] = []
        cache_action = _session_cache_action(request)
        for index, (prompt, output, detail) in enumerate(zip(expanded_prompts, batch.outputs, batch.details, strict=True)):
            previous_text = "" if continuation is None else continuation.generated_texts[index]
            output = f"{previous_text}{output}"
            generated_text, finish_reason = _apply_stop(output, request.stop)
            server_stop = generated_text != output
            finish_reason = _finish_reason_for_output(detail, finish_reason, server_stop=server_stop)
            text = prompt + generated_text if request.echo else generated_text
            response_format_failure = _response_format_failure_reason(request, generated_text, finish_reason)
            if response_format_failure is not None:
                finish_reason = "stop"
                text = ""
            final_texts.append(text)
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
                    reason_override=response_format_failure or ("stop" if server_stop else None),
                    cache_action=cache_action,
                ),
            }
            _mark_continuation_unavailable(choice["finish_details"])
            if _continuation_can_create(request, finish_reason=finish_reason, finish_details=choice["finish_details"]):
                base_prompt = prompt if continuation is None else continuation.prompts[index]
                record = await store_continuation(
                    endpoint="completion",
                    prompts=(base_prompt,),
                    generated_texts=(generated_text,),
                    response_format=request.response_format,
                )
                _attach_continuation_metadata(choice, continuation_id=record.id)
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
        }
        await _maybe_write_agentic_result_replay_artifact(
            config,
            raw_request,
            response,
            engine=getattr(app.state, "hipengine_llm", None),
        )
        if request.stream:
            include_hipengine = _stream_include_hipengine(request)
            stream_started_at = _StreamTimingTracker.start() if include_hipengine else time.perf_counter()
            return StreamingResponse(
                _completion_stream(
                    response_id,
                    created,
                    config.model_id,
                    final_texts,
                    choices,
                    usage=batch.usage if _stream_include_usage(request) else None,
                    include_hipengine=include_hipengine,
                    stream_started_at=stream_started_at,
                ),
                media_type="text/event-stream",
            )
        return response

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: ChatCompletionRequest,
        raw_request: Request,
        _auth: None = Depends(require_auth),
    ) -> dict[str, Any] | StreamingResponse:
        _validate_model(config, request.model)
        _validate_generation_request(config, request)
        _validate_continuation_resume_request(request)
        admitted_session_id = await reserve_chat_session_if_needed(request)
        try:
            continuation = await pop_continuation(request, endpoint="chat")
            _apply_continuation_defaults(request, continuation)
            control = _request_control(config, request, raw_request)

            async def prepare_prompt() -> str:
                async with session_lock:
                    engine = get_llm()
                    await ensure_resident_context(engine, preparation_sampling(request), phase="preparation")
                    if continuation is not None:
                        return continuation.resume_prompts()[0]
                    if not request.messages:
                        raise OpenAIHTTPError(
                            400,
                            "messages must not be empty",
                            code="invalid_request",
                            param="messages",
                        )
                    session_messages = await chat_session_prefix_messages(request)
                    render_request = request
                    if session_messages:
                        render_request = _chat_request_with_messages(
                            request,
                            (*session_messages, *request.messages),
                        )
                    return chat_prompt_for_request(render_request, engine)

            prompt = await _await_with_request_control(prepare_prompt(), control)
            if request.stream:
                streamer = (
                    stream_chat_completion_many
                    if _request_n(request) > 1
                    or request.logprobs
                    or _response_format_result_validation(request)
                    else stream_chat_completion
                )
                return StreamingResponse(
                    streamer(prompt, request, control, raw_request),
                    media_type="text/event-stream",
                )
            n = _request_n(request)
            prompts = tuple(prompt for _ in range(n))
            batch = await generate_with_request_control(prompts, request, control)
            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            choices = []
            requested_cache_action = _session_cache_action(request)
            for index, (output, detail) in enumerate(zip(batch.outputs, batch.details, strict=True)):
                previous_text = "" if continuation is None else continuation.generated_texts[index]
                output = f"{previous_text}{output}"
                text, finish_reason = _apply_stop(output, request.stop)
                server_stop = text != output
                parsed = _parse_chat_tool_calls(text)
                tool_validation = _validate_chat_tool_result(request, parsed, text)
                parsed = tool_validation.parsed
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
                response_format_failure = _response_format_failure_reason(
                    request,
                    _chat_response_format_text(message),
                    finish_reason,
                )
                if response_format_failure is not None:
                    finish_reason = "stop"
                    message = {"role": "assistant", "content": ""}
                if tool_validation.failed:
                    finish_reason_override = tool_validation.failure_reason
                elif response_format_failure is not None:
                    finish_reason_override = response_format_failure
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
                effective_cache_action = _effective_session_cache_action(
                    requested_cache_action,
                    choice["finish_details"],
                )
                if effective_cache_action != requested_cache_action:
                    choice["finish_details"]["cache_action"] = effective_cache_action
                if request.logprobs:
                    choice["logprobs"] = _chat_logprobs(detail, text)
                _mark_continuation_unavailable(choice["finish_details"])
                if _continuation_can_create(request, finish_reason=finish_reason, finish_details=choice["finish_details"]):
                    base_prompt = prompt if continuation is None else continuation.prompts[index]
                    record = await store_continuation(
                        endpoint="chat",
                        prompts=(base_prompt,),
                        generated_texts=(text,),
                        response_format=request.response_format,
                    )
                    _attach_continuation_metadata(choice, continuation_id=record.id)
                _attach_choice_telemetry(choice, detail)
                if n > 1:
                    choice["request_id"] = _choice_request_id(response_id, 0, index)
                choices.append(choice)
                await commit_chat_session(
                    request,
                    request_messages=request.messages,
                    raw_output=output,
                    visible_message=message,
                    cache_action=effective_cache_action,
                )
            response = {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": config.model_id,
                "choices": choices,
                "usage": batch.usage,
            }
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
        request: ChatCompletionRequest,
        control: _RequestControl,
        raw_request: Request,
    ) -> AsyncIterator[str]:
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        include_hipengine = _stream_include_hipengine(request)
        stream_started_at = _StreamTimingTracker.start() if include_hipengine else time.perf_counter()
        try:
            n = _request_n(request)
            batch = await generate_with_request_control(tuple(prompt for _ in range(n)), request, control)
            for index, output in enumerate(batch.outputs):
                text, finish_reason = _apply_stop(output, request.stop)
                server_stop = text != output
                parsed = _parse_chat_tool_calls(text)
                tool_validation = _validate_chat_tool_result(request, parsed, text)
                parsed = tool_validation.parsed
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
                response_format_failure = _response_format_failure_reason(
                    request,
                    _chat_response_format_text_from_parsed(parsed),
                    finish_reason,
                )
                if response_format_failure is not None:
                    finish_reason = "stop"
                    parsed = _ParsedChatOutput(text="", tool_calls=())
                if tool_validation.failed:
                    finish_reason_override = tool_validation.failure_reason
                elif response_format_failure is not None:
                    finish_reason_override = response_format_failure
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
                logprobs = _chat_logprobs(batch.details[index], text) if request.logprobs else None
                yield _chat_stream_role(
                    response_id,
                    created,
                    config.model_id,
                    index=index,
                    include_hipengine=include_hipengine,
                    stream_started_at=stream_started_at,
                )
                for event in _chat_stream_parsed(
                    response_id,
                    created,
                    config.model_id,
                    parsed,
                    finish_reason,
                    index=index,
                    logprobs=logprobs,
                    finish_details=finish_details,
                    include_hipengine=include_hipengine,
                    stream_started_at=stream_started_at,
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
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
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
            )
        yield "data: [DONE]\n\n"

    async def stream_chat_completion(
        prompt: str,
        request: ChatCompletionRequest,
        control: _RequestControl,
        raw_request: Request,
    ) -> AsyncIterator[str]:
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        include_hipengine = _stream_include_hipengine(request)
        stream_started_at = _StreamTimingTracker.start() if include_hipengine else time.perf_counter()
        try:
            _validate_generation_request(config, request)
        except OpenAIHTTPError as exc:
            _record_openai_error(app.state.hipengine_server_metrics, exc)
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
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
            )
            yield "data: [DONE]\n\n"
            return
        full_text: list[str] = []
        splitter = _ReasoningSplitter()
        buffer_tool_output = bool(request.tools)

        try:
            async def prepare_stream() -> tuple[Any, SamplingParams]:
                async with session_lock:
                    engine = get_llm()
                    await ensure_resident_context(engine, preparation_sampling(request), phase="preparation")
                    sampling = sampling_params(
                        request,
                        (prompt,),
                        engine,
                        deadline_at=control.deadline_at,
                        cancellation_token=control.cancellation_token,
                    )
                    _validate_context_budget(effective_max_context_tokens(engine), engine, (prompt,), sampling)
                    return engine, sampling

            engine, sampling = await _await_with_request_control(prepare_stream(), control)
            token_accounting = _StreamTokenAccounting.for_engine(engine) if include_hipengine else None
            yield _chat_stream_role(
                response_id,
                created,
                config.model_id,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
            )
            async for token in _iterate_with_request_control(generation_batcher.stream((prompt,), sampling), control):
                text = str(token)
                if not text:
                    continue
                full_text.append(text)
                if buffer_tool_output:
                    continue
                for field, chunk in splitter.feed(text):
                    phase = "think" if field == "reasoning_content" else "answer"
                    token_payload = (
                        token_accounting.observe(phase, chunk) if token_accounting is not None else None
                    )
                    yield _chat_stream_delta(
                        response_id,
                        created,
                        config.model_id,
                        field,
                        chunk,
                        tokens=token_payload,
                        include_hipengine=include_hipengine,
                        stream_started_at=stream_started_at,
                    )
            if not buffer_tool_output:
                for field, chunk in splitter.finish():
                    phase = "think" if field == "reasoning_content" else "answer"
                    token_payload = (
                        token_accounting.observe(phase, chunk) if token_accounting is not None else None
                    )
                    yield _chat_stream_delta(
                        response_id,
                        created,
                        config.model_id,
                        field,
                        chunk,
                        tokens=token_payload,
                        include_hipengine=include_hipengine,
                        stream_started_at=stream_started_at,
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
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
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
            yield _chat_stream_error(
                response_id,
                created,
                config.model_id,
                message,
                status_code=500,
                code="generation_failed",
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
            )
            yield "data: [DONE]\n\n"
            return

        raw_text = "".join(full_text)
        text, finish_reason = _apply_stop(raw_text, request.stop)
        if text != raw_text:
            # Stop strings can split across yielded chunks; current streaming keeps
            # transport simple and reports the stop after generation completes.
            finish_reason = "stop"
        usage = _usage(engine, (prompt,), [text])
        app.state.hipengine_server_metrics.record_success(usage)
        cache_action = _session_cache_action(request)
        final_tokens = (
            _stream_usage_token_payload(usage, token_accounting)
            if include_hipengine and token_accounting is not None
            else None
        )
        if buffer_tool_output:
            parsed = _parse_chat_tool_calls(text)
            tool_validation = _validate_chat_tool_result(request, parsed, text)
            parsed = tool_validation.parsed
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
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
            ):
                yield event
        else:
            yield _chat_stream_done(
                response_id,
                created,
                config.model_id,
                finish_reason,
                finish_details=_finish_details_payload(
                    None,
                    finish_reason,
                    cache_action=cache_action,
                ),
                tokens=final_tokens,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
            )
        if _stream_include_usage(request):
            yield _chat_stream_usage(
                response_id,
                created,
                config.model_id,
                usage,
                include_hipengine=include_hipengine,
                stream_started_at=stream_started_at,
            )
        yield "data: [DONE]\n\n"

    return app


def _metrics_mode(raw: str | None) -> str:
    value = "off" if raw is None or raw == "" else str(raw).strip().lower()
    if value not in {"off", "prometheus"}:
        raise ValueError("metrics must be one of: off, prometheus")
    return value


def _render_prometheus_metrics(
    metrics: _ServerMetrics,
    *,
    engine: Any | None,
    generation_batcher: Any | None = None,
    chat_sessions: Mapping[str, Any] | None = None,
    pending_chat_sessions: set[str] | None = None,
    max_chat_sessions: int | None = None,
) -> str:
    pool = _pool_metric_values(engine)
    graph = _graph_bucket_metric_values(engine)
    queue = _generation_queue_metric_values(generation_batcher)
    values = {
        "hipengine_requests_total": metrics.request_total,
        "hipengine_request_completed_total": metrics.request_completed_total,
        "hipengine_request_failed_total": metrics.request_failed_total,
        "hipengine_request_rejected_total": metrics.request_rejected_total,
        "hipengine_request_cancelled_total": metrics.request_cancelled_total,
        "hipengine_prompt_tokens_total": metrics.prompt_tokens_total,
        "hipengine_completion_tokens_total": metrics.completion_tokens_total,
        "hipengine_kv_pool_current_bytes": pool["current_bytes"],
        "hipengine_kv_pool_high_water_observed_bytes": pool["high_water_observed_bytes"],
        "hipengine_kv_pool_grow_events_total": pool["grow_events"],
        "hipengine_kv_pool_grow_failures_total": pool["grow_failures"],
        "hipengine_kv_pool_shrink_events_total": pool["shrink_events"],
        "hipengine_kv_pool_free_pages": pool["free_pages"],
        "hipengine_kv_pool_refcounted_pages": pool["refcounted_pages"],
        "hipengine_graph_bucket_entries": graph["entries"],
        "hipengine_graph_bucket_hits_total": graph["hits"],
        "hipengine_graph_bucket_misses_total": graph["misses"],
        "hipengine_graph_bucket_replay_hit_rate": graph["replay_hit_rate"],
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
    }
    help_text = {
        "hipengine_requests_total": "Total generation requests observed by the server.",
        "hipengine_request_completed_total": "Generation requests that completed successfully.",
        "hipengine_request_failed_total": "Generation requests that failed after reaching generation validation.",
        "hipengine_request_rejected_total": "Generation requests rejected by admission/backpressure.",
        "hipengine_request_cancelled_total": "Generation requests cancelled after client disconnect or backend cancellation.",
        "hipengine_prompt_tokens_total": "Prompt tokens counted for successful requests.",
        "hipengine_completion_tokens_total": "Completion tokens counted for successful requests.",
        "hipengine_kv_pool_current_bytes": "Current dynamic KV pool bytes, or 0 when unavailable.",
        "hipengine_kv_pool_high_water_observed_bytes": "Peak observed dynamic KV pool bytes, or 0 when unavailable.",
        "hipengine_kv_pool_grow_events_total": "Dynamic KV pool grow events, or 0 when unavailable.",
        "hipengine_kv_pool_grow_failures_total": "Dynamic KV pool grow failures, or 0 when unavailable.",
        "hipengine_kv_pool_shrink_events_total": "Dynamic KV pool shrink events, or 0 when unavailable.",
        "hipengine_kv_pool_free_pages": "Current dynamic KV pool free pages, or 0 when unavailable.",
        "hipengine_kv_pool_refcounted_pages": "Current dynamic KV pool refcounted pages, or 0 when unavailable.",
        "hipengine_graph_bucket_entries": "Current graph bucket cache entries, or 0 when unavailable.",
        "hipengine_graph_bucket_hits_total": "Graph bucket cache hits, or 0 when unavailable.",
        "hipengine_graph_bucket_misses_total": "Graph bucket cache misses, or 0 when unavailable.",
        "hipengine_graph_bucket_replay_hit_rate": "Graph bucket replay hit rate, or 0 when unavailable.",
        "hipengine_generation_queue_depth": "Current generation-batcher queue depth.",
        "hipengine_generation_queue_max_depth": "Configured generation-batcher queue cap, or 0 when unset.",
        "hipengine_generation_worker_active": "Whether the generation-batcher worker is active, as 0 or 1.",
        "hipengine_generation_requests_active": "Current number of HTTP requests in the active backend batch.",
        "hipengine_generation_requests_max_active": "Configured active backend request cap, or 0 when unset.",
        "hipengine_chat_sessions_active": "Current app-local chat transcript session count.",
        "hipengine_chat_sessions_pending": "Current app-local chat session creations in flight.",
        "hipengine_chat_sessions_max_active": "Configured app-local chat session cap, or 0 when unset.",
    }
    counter_names = {
        "hipengine_requests_total",
        "hipengine_request_completed_total",
        "hipengine_request_failed_total",
        "hipengine_request_rejected_total",
        "hipengine_request_cancelled_total",
        "hipengine_prompt_tokens_total",
        "hipengine_completion_tokens_total",
        "hipengine_kv_pool_grow_events_total",
        "hipengine_kv_pool_grow_failures_total",
        "hipengine_kv_pool_shrink_events_total",
        "hipengine_graph_bucket_hits_total",
        "hipengine_graph_bucket_misses_total",
    }
    lines: list[str] = []
    for name, value in values.items():
        lines.append(f"# HELP {name} {help_text[name]}")
        lines.append(f"# TYPE {name} {'counter' if name in counter_names else 'gauge'}")
        lines.append(f"{name} {_format_metric_value(value)}")
    scheduler = _scheduler_fairness_capability()
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


def _pool_metric_values(engine: Any | None) -> dict[str, float]:
    values = {
        "current_bytes": 0.0,
        "high_water_observed_bytes": 0.0,
        "grow_events": 0.0,
        "grow_failures": 0.0,
        "shrink_events": 0.0,
        "free_pages": 0.0,
        "refcounted_pages": 0.0,
    }
    stats = _first_stats_object(engine, ("kv_pool", "kv_cache_pool", "pool", "kv_pool_stats"))
    if stats is None:
        return values
    data = _stats_to_mapping(stats)
    for key in values:
        values[key] = _non_negative_metric_value(data.get(key))
    return values


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


def _graph_bucket_metric_values(engine: Any | None) -> dict[str, Any]:
    values: dict[str, Any] = {
        "entries": 0.0,
        "hits": 0.0,
        "misses": 0.0,
        "replay_hit_rate": 0.0,
        "miss_reasons": {},
        "kernel_time_histogram_ns": {},
    }
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
        "grow_events",
        "grow_failures",
        "shrink_events",
        "free_pages",
        "refcounted_pages",
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
        if thinking_type in {"disabled", "disable", "off", "none"}:
            enabled = False
        elif thinking_type in {"enabled", "enable", "on"}:
            enabled = True
        enabled = _maybe_bool(request.thinking.get("enabled"), enabled)
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
        enabled = _maybe_bool(request.reasoning.get("enabled"), enabled)
        reasoning_type = str(request.reasoning.get("type", "")).strip().lower()
        if reasoning_type in {"disabled", "disable", "off", "none"}:
            enabled = False
        elif reasoning_type in {"enabled", "enable", "on"}:
            enabled = True
        effort = _maybe_effort(request.reasoning.get("effort"), effort)

    if _effort_disables_thinking(effort):
        enabled = False
    if enabled is not False and not _effort_disables_thinking(effort):
        if generation_budget is _THINKING_BUDGET_UNSET:
            generation_budget = _thinking_generation_budget(
                request,
                chat_default_max_tokens=chat_default_max_tokens,
            )
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
        max_think_tokens=max_think_tokens,
        min_answer_tokens=min_answer_tokens,
        hard_think_cap=hard_think_cap,
        soft_close_window=soft_close_window,
        hard_close_message=hard_close_message,
        hard_close_sequence=hard_close_sequence,
    )
    _validate_thinking_control(control)
    return control


def _render_chat_prompt_for_request(
    request: ChatCompletionRequest,
    *,
    chat_default_max_tokens: int | None,
    engine: Any | None = None,
    max_context_tokens: int | None = None,
) -> tuple[str, _ThinkingControl]:
    thinking = _thinking_control_from_request(
        request,
        chat_default_max_tokens=chat_default_max_tokens,
    )
    prompt = render_chat_prompt(
        request.messages,
        tools=request.tools,
        tool_choice=request.tool_choice,
        thinking=thinking,
        response_format=request.response_format,
    )
    if engine is None:
        return prompt, thinking

    for _ in range(4):
        generation_budget = _thinking_generation_budget_for_prompt(
            request,
            prompt,
            engine,
            max_context_tokens,
            chat_default_max_tokens=chat_default_max_tokens,
        )
        adjusted = _thinking_control_from_request(
            request,
            chat_default_max_tokens=chat_default_max_tokens,
            generation_budget=generation_budget,
        )
        adjusted_prompt = render_chat_prompt(
            request.messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            thinking=adjusted,
            response_format=request.response_format,
        )
        if adjusted == thinking and adjusted_prompt == prompt:
            return prompt, thinking
        thinking = adjusted
        prompt = adjusted_prompt
    return prompt, thinking


def _thinking_generation_budget_for_prompt(
    request: ChatCompletionRequest,
    prompt: str,
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
    if request.max_tokens is not None:
        return max(0, int(request.max_tokens))
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
            _render_tools_prompt(tools, tool_choice),
        )
        if item
    ]
    if control_prompts:
        control_block = "\n\n".join(control_prompts)
        rendered.append(f"<|im_start|>system\n{control_block}<|im_end|>")
    for index, message in enumerate(messages):
        if isinstance(message, Mapping):
            role_value = message.get("role", "")
            content_value = message.get("content", "")
            tool_calls = message.get("tool_calls")
        else:
            role_value = message.role
            content_value = message.content
            tool_calls = message.tool_calls
        role = str(role_value).strip()
        if not role:
            raise OpenAIHTTPError(
                400,
                "message role must be non-empty",
                param=f"messages[{index}].role",
            )
        content = _message_content_text(content_value, index)
        if role == "developer":
            role = "system"
        if role == "tool":
            rendered.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>")
            continue
        if role == "assistant" and tool_calls:
            tool_call_text = "\n".join(_render_tool_call_for_prompt(item) for item in tool_calls)
            content = "\n".join(part for part in (content, tool_call_text) if part)
        rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    rendered.append(_assistant_prefix_for_thinking(thinking))
    return "\n".join(rendered)


def _validate_model(config: ServerConfig, requested: str | None) -> None:
    if requested is not None and requested != config.model_id:
        raise OpenAIHTTPError(
            404,
            f"model {requested!r} is not served by this hipEngine instance",
            code="model_not_found",
            param="model",
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


def _selected_device_payload(config: ServerConfig) -> dict[str, Any]:
    return {
        "backend": config.backend,
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
    }


def _resident_session_for_engine(engine: Any) -> Any | None:
    if hasattr(engine, "kv_capacity_estimate"):
        return engine
    generator = getattr(engine, "_text_generator", None)
    if generator is not None:
        session = getattr(generator, "_session", None)
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


def _request_max_tokens(
    request: CompletionRequest | ChatCompletionRequest,
    prompts: Sequence[str],
    engine: Any,
    max_context_tokens: int | None,
    *,
    chat_default_max_tokens: int | None = 4096,
) -> int:
    if request.max_tokens is not None:
        return max(0, int(request.max_tokens))
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
    prompts: Sequence[str],
    engine: Any,
    max_context_tokens: int | None,
) -> int | None:
    if max_context_tokens is None:
        return None
    return min(
        int(max_context_tokens) - _count_tokens_for_admission(engine, str(prompt)) - 1
        for prompt in prompts
    )


def _chat_default_max_tokens_label(config: ServerConfig) -> str:
    return "auto" if config.chat_default_max_tokens is None else str(int(config.chat_default_max_tokens))


def _request_logprobs_enabled(request: CompletionRequest | ChatCompletionRequest) -> bool:
    if isinstance(request, CompletionRequest):
        return request.logprobs is not None
    return bool(request.logprobs)


def _request_top_logprobs(request: CompletionRequest | ChatCompletionRequest) -> int:
    if isinstance(request, CompletionRequest):
        return 0 if request.logprobs is None else int(request.logprobs)
    return 0 if request.top_logprobs is None else int(request.top_logprobs)


async def _generate_detailed(
    engine: Any,
    prompts: tuple[str, ...],
    sampling: SamplingParams,
) -> list[Any]:
    detailed = getattr(engine, "generate_detailed", None)
    if callable(detailed):
        return list(await run_in_threadpool(detailed, prompts, sampling))
    return [_coerce_generation_output(item) for item in await run_in_threadpool(engine.generate, prompts, sampling)]


def _coerce_generation_output(value: Any) -> GenerationOutput:
    if isinstance(value, GenerationOutput):
        return value
    token_logprobs = getattr(value, "token_logprobs", None)
    finish_details = getattr(value, "finish_details", None)
    telemetry = getattr(value, "telemetry", None)
    if token_logprobs is not None or finish_details is not None or telemetry is not None:
        return GenerationOutput(
            text=str(getattr(value, "text", value)),
            token_logprobs=tuple(token_logprobs or ()),
            finish_details=finish_details,
            telemetry=telemetry,
        )
    return GenerationOutput(text=str(value))


def _validate_logprob_details(details: Sequence[GenerationOutput], outputs: Sequence[str]) -> None:
    for output, text in zip(details, outputs, strict=True):
        if text and not output.token_logprobs:
            raise OpenAIHTTPError(
                500,
                "generator did not return token logprobs for a logprobs request",
                error_type="server_error",
                code="missing_logprobs",
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


async def _await_with_request_control(awaitable, control: _RequestControl | None = None):
    control = control or _RequestControl()
    if control.deadline_at is None and control.disconnected is None:
        return await awaitable
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
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
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


def _validate_generation_request(config: ServerConfig, request: CompletionRequest | ChatCompletionRequest) -> None:
    _request_n(request)
    extra_keys = _request_extra_keys(request)
    if extra_keys:
        param = sorted(extra_keys)[0]
        raise OpenAIHTTPError(
            400,
            f"unsupported request parameter {param!r}",
            code="unsupported_parameter",
            param=param,
        )
    _validate_session_request(request)
    unsupported_param = _unsupported_agentic_request_param(request)
    if unsupported_param is not None:
        raise OpenAIHTTPError(
            400,
            f"{unsupported_param} is not supported by this server",
            code="unsupported_parameter",
            param=unsupported_param,
        )
    _validate_response_format_request(request)
    _validate_tool_schema_requests(request)
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
    prompts: Sequence[str],
    sampling: SamplingParams,
) -> None:
    if max_context_tokens is None:
        return
    max_context = max(1, int(max_context_tokens))
    max_tokens = max(0, int(sampling.max_tokens))
    for index, prompt in enumerate(prompts):
        prompt_tokens = _count_tokens_for_admission(engine, str(prompt))
        required = prompt_tokens + max_tokens + 1
        if required > max_context:
            fit_context = _context_fit_payload(
                prompt_tokens=prompt_tokens,
                max_context_tokens=max_context,
                max_tokens=max_tokens,
            )
            raise OpenAIHTTPError(
                400,
                f"request requires {required} context tokens (prompt {prompt_tokens} + "
                f"max_tokens {max_tokens} + 1), exceeding this server's "
                f"preallocated max_context_tokens={max_context}",
                code="context_length_exceeded",
                param=f"prompts[{index}].max_tokens" if len(prompts) > 1 else "max_tokens",
                extra={"fit_context": fit_context},
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
    return {
        "prompt_tokens": prompt_count,
        "max_context_tokens": max_context,
        "effective_max_tokens": effective_max_tokens,
        "required_context_tokens": required_context,
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


def _thinking_budget_sampling_kwargs(
    config: ServerConfig,
    request: ChatCompletionRequest,
    *,
    engine: Any,
    max_context_tokens: int | None,
) -> dict[str, Any]:
    _prompt, thinking = _render_chat_prompt_for_request(
        request,
        chat_default_max_tokens=config.chat_default_max_tokens,
        engine=engine,
        max_context_tokens=max_context_tokens,
    )
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


def _required_tool_sampling_forced_token_ids(
    request: ChatCompletionRequest,
    engine: Any,
) -> tuple[int, ...]:
    if not request.tools:
        return ()
    mode, _name = _tool_choice_mode(request.tool_choice)
    if mode not in {"required", "function"}:
        return ()
    try:
        token_ids = _tokenize_text(engine, _TOOL_CALL_START_MARKER)
    except OpenAIHTTPError:
        return ()
    return tuple(int(token_id) for token_id in token_ids)


def _tool_call_sequence_completion_token_sequences(
    request: ChatCompletionRequest,
    engine: Any,
    forced_tool_token_ids: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    sequences: list[tuple[int, ...]] = []
    prefix_sequence = _specific_tool_name_prefix_token_sequence(request, engine, forced_tool_token_ids)
    if prefix_sequence:
        sequences.append(prefix_sequence)
    sequences.extend(_tool_call_close_repair_token_sequences(request, engine))
    return tuple(sequences)


def _specific_tool_name_prefix_token_sequence(
    request: ChatCompletionRequest,
    engine: Any,
    forced_tool_token_ids: Sequence[int],
) -> tuple[int, ...]:
    start_ids = tuple(int(token_id) for token_id in forced_tool_token_ids)
    if not start_ids:
        return ()
    name = _specific_tool_name_prefix_target(request)
    if name is None:
        return ()
    prefix_text = _tool_call_name_prefix_text(name)
    try:
        prefix_ids = _tokenize_text(engine, prefix_text)
    except OpenAIHTTPError:
        return ()
    prefix = tuple(int(token_id) for token_id in prefix_ids)
    if len(prefix) <= len(start_ids) or prefix[: len(start_ids)] != start_ids:
        return ()
    return prefix


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


def _tool_call_close_repair_token_sequences(
    request: ChatCompletionRequest,
    engine: Any,
) -> tuple[tuple[int, ...], ...]:
    if not request.tools:
        return ()
    mode, _name = _tool_choice_mode(request.tool_choice)
    if mode not in {"required", "function"}:
        return ()
    try:
        token_ids = _tokenize_text(engine, _TOOL_CALL_END_MARKER)
    except OpenAIHTTPError:
        return ()
    if not token_ids:
        return ()
    return (tuple(int(token_id) for token_id in token_ids),)


def _normalize_prompts(prompt: str | list[str] | None) -> tuple[str, ...]:
    if prompt is None:
        raise OpenAIHTTPError(400, "prompt is required", code="invalid_request", param="prompt")
    if isinstance(prompt, str):
        return (prompt,)
    if not prompt:
        raise OpenAIHTTPError(400, "prompt must not be empty", param="prompt")
    return tuple(str(item) for item in prompt)


def _request_n(request: CompletionRequest | ChatCompletionRequest) -> int:
    n = 1 if request.n is None else int(request.n)
    if n < 1:
        raise OpenAIHTTPError(400, "n must be at least 1", code="invalid_request", param="n")
    return n


def _unsupported_agentic_request_param(request: CompletionRequest | ChatCompletionRequest) -> str | None:
    session = getattr(request, "session", None)
    if session is not None:
        if isinstance(session, Mapping):
            if "id" in session:
                if not isinstance(request, ChatCompletionRequest):
                    return "session.id"
                if request.stream:
                    return "stream"
                if _request_n(request) != 1:
                    return "n"
                if getattr(request, "continuation_id", None) is not None:
                    return "continuation_id"
                if set(session.keys()) - {"id", "commit"}:
                    return "session"
                if _session_cache_action(request) is not None:
                    return None
                return "session.commit"
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
    if request.temperature not in (None, 0, 0.0):
        return "temperature"
    if request.logit_bias:
        return "logit_bias"
    if request.suppress_token_ids:
        return "suppress_token_ids"
    if int(request.min_tokens or 0) != 0:
        return "min_tokens"
    if float(request.repetition_penalty or 1.0) != 1.0:
        return "repetition_penalty"
    if float(request.presence_penalty or 0.0) != 0.0:
        return "presence_penalty"
    if float(request.frequency_penalty or 0.0) != 0.0:
        return "frequency_penalty"
    if isinstance(request, ChatCompletionRequest):
        if request.tools:
            return "tools"
        if request.tool_choice is not None:
            return "tool_choice"
        if request.parallel_tool_calls is not None:
            return "parallel_tool_calls"
        if _thinking_budget_sampling_kwargs_present(request):
            return "thinking"
    return None


def _continuation_can_create(
    request: CompletionRequest | ChatCompletionRequest,
    *,
    finish_reason: str,
    finish_details: Mapping[str, Any],
) -> bool:
    if not _is_length_finish(finish_reason, finish_details):
        return False
    if _session_id(request) is not None:
        return False
    if request.stream or _request_n(request) < 1 or _request_logprobs_enabled(request):
        return False
    if isinstance(request, CompletionRequest) and request.echo:
        return False
    if isinstance(request, ChatCompletionRequest):
        phase = str(finish_details.get("phase") or "")
        if phase and phase not in {"answer", "structured"}:
            return False
        if request.tools or request.tool_choice is not None or request.parallel_tool_calls is not None:
            return False
        if _thinking_budget_sampling_kwargs_present(request):
            return False
    return _continuation_sampling_is_deterministic(request)


def _thinking_budget_sampling_kwargs_present(request: ChatCompletionRequest) -> bool:
    return any(
        value is not None
        for value in (
            request.max_think_tokens,
            request.min_answer_tokens,
            request.hard_think_cap,
            request.soft_close_window,
            request.hard_close_message,
            request.hard_close_sequence,
            request.thinking_token_budget,
            request.chat_template_kwargs,
            request.thinking,
            request.reasoning,
        )
    )


def _continuation_sampling_is_deterministic(request: CompletionRequest | ChatCompletionRequest) -> bool:
    if request.temperature not in (None, 0, 0.0):
        return False
    if request.logit_bias:
        return False
    if request.suppress_token_ids:
        return False
    if int(request.min_tokens or 0) != 0:
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
        finish_details.setdefault("continuation_eligible", False)


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


def _validate_session_request(request: CompletionRequest | ChatCompletionRequest) -> None:
    session = getattr(request, "session", None)
    if session is None or not isinstance(session, Mapping):
        return
    if "id" in session:
        _session_id(request)
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
        if mode in _SESSION_COMMIT_MODES and not (set(session.keys()) - {"id", "commit"}):
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
    return {str(key): value for key, value in payload.items() if value is not None}


def _assistant_visible_session_message(message: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": "assistant"}
    if "content" in message:
        payload["content"] = message.get("content")
    else:
        payload["content"] = ""
    tool_calls = message.get("tool_calls")
    if tool_calls:
        payload["tool_calls"] = list(tool_calls)
    return payload


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


def _validate_tool_schema_requests(request: CompletionRequest | ChatCompletionRequest) -> None:
    if not isinstance(request, ChatCompletionRequest) or not request.tools:
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


def _expand_prompts_for_n(prompts: Sequence[str], n: int) -> tuple[str, ...]:
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
    if reason in {"length", "max_length", "max_tokens", "token_budget_exhausted", "budget_exhausted"}:
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
    split = _split_reasoning(parsed.text)
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
    return _split_reasoning(parsed.text).content


def _is_length_finish_payload(payload: Mapping[str, Any]) -> bool:
    reason = str(payload.get("reason", "")).strip().lower()
    return reason in {
        "length",
        "max_length",
        "max_tokens",
        "token_budget_exhausted",
        "budget_exhausted",
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


def _sampling_key(sampling: SamplingParams) -> tuple[Any, ...]:
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
        None if sampling.cancellation_token is None else id(sampling.cancellation_token),
        bool(sampling.logprobs),
        int(sampling.top_logprobs),
    )


def _completion_logprobs(detail: GenerationOutput, text: str, *, echo_text: str = "") -> dict[str, Any]:
    tokens = list(_trim_token_logprobs(detail.token_logprobs, text))
    response_tokens: list[str] = []
    token_logprobs: list[float | None] = []
    top_logprobs: list[dict[str, float] | None] = []
    offsets: list[int] = []
    cursor = 0
    if echo_text:
        response_tokens.append(echo_text)
        token_logprobs.append(None)
        top_logprobs.append(None)
        offsets.append(0)
        cursor = len(echo_text)
    for token in tokens:
        response_tokens.append(token.token_text)
        token_logprobs.append(token.logprob)
        top_logprobs.append(_completion_top_logprobs(token))
        offsets.append(cursor)
        cursor += len(token.token_text)
    return {
        "tokens": response_tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": offsets,
    }


def _completion_top_logprobs(token: TokenLogprob) -> dict[str, float] | None:
    if not token.top_logprobs:
        return None
    return {text: float(logprob) for _token_id, text, logprob in token.top_logprobs}


def _chat_logprobs(detail: GenerationOutput, text: str) -> dict[str, Any]:
    tokens = _trim_token_logprobs(detail.token_logprobs, text)
    return {
        "content": [
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
        ],
        "refusal": None,
    }


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


def _usage(engine: Any, prompts: Sequence[str], outputs: Sequence[str]) -> dict[str, int]:
    counter = getattr(engine, "count_tokens", None)
    if callable(counter):
        prompt_tokens = sum(_safe_count(counter, text) for text in prompts)
        completion_tokens = sum(_safe_count(counter, text) for text in outputs)
    else:
        # Compatibility placeholder until public tokenizer accounting lands.
        prompt_tokens = 0
        completion_tokens = 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _safe_count(counter: Any, text: str) -> int:
    try:
        return max(0, int(counter(text)))
    except Exception:
        return 0


_REASONING_OPEN_TAG = "<think>"
_REASONING_CLOSE_TAG = "</think>"


class _ReasoningSplitter:
    """Incrementally split Qwen/DeepSeek-style thinking tags from answer text."""

    def __init__(self) -> None:
        self._buffer = ""
        self._in_reasoning = False

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        self._buffer += text
        return self._drain(final=False)

    def finish(self) -> list[tuple[str, str]]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        while self._buffer:
            tag = _REASONING_CLOSE_TAG if self._in_reasoning else _REASONING_OPEN_TAG
            index = self._buffer.find(tag)
            if index >= 0:
                self._append(outputs, self._buffer[:index])
                self._buffer = self._buffer[index + len(tag) :]
                self._in_reasoning = not self._in_reasoning
                continue
            if final:
                self._append(outputs, self._buffer)
                self._buffer = ""
                break
            keep = _tag_suffix_len(self._buffer, tag)
            emit_len = len(self._buffer) - keep
            if emit_len > 0:
                self._append(outputs, self._buffer[:emit_len])
                self._buffer = self._buffer[emit_len:]
            break
        return outputs

    def _append(self, outputs: list[tuple[str, str]], text: str) -> None:
        if text:
            field = "reasoning_content" if self._in_reasoning else "content"
            outputs.append((field, text))


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


def _split_reasoning(text: str) -> _ReasoningSplit:
    splitter = _ReasoningSplitter()
    parts = splitter.feed(text) + splitter.finish()
    content = "".join(part for field, part in parts if field == "content")
    reasoning = "".join(part for field, part in parts if field == "reasoning_content")
    return _ReasoningSplit(content=content, reasoning_content=reasoning)


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


def _parse_chat_tool_calls(text: str) -> _ParsedChatOutput:
    calls: list[_ParsedToolCall] = []
    text_parts: list[str] = []
    last_end = 0
    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        text_parts.append(text[last_end : match.start()])
        parsed = _parsed_tool_call_from_json(match.group(1).strip(), raw_text=match.group(0))
        if parsed is None:
            text_parts.append(match.group(0))
        else:
            calls.append(parsed)
        last_end = match.end()
    text_parts.append(text[last_end:])
    if not calls:
        stripped = text.strip()
        parsed = _parsed_tool_call_from_json(stripped, raw_text=stripped)
        if parsed is not None:
            return _ParsedChatOutput(text="", tool_calls=(parsed,))
    return _ParsedChatOutput(text="".join(text_parts).strip(), tool_calls=tuple(calls))


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


def _chat_message_from_parsed(parsed: _ParsedChatOutput) -> tuple[dict[str, Any], str]:
    split = _split_reasoning(parsed.text)
    message: dict[str, Any] = {"role": "assistant", "content": split.content}
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
    strict = _strict_tool_validation_enabled(request)
    if not strict:
        return _ToolValidationResult(parsed)
    mode, required_name = _tool_choice_mode(request.tool_choice)
    malformed_blocks = _malformed_tool_call_blocks(raw_text)
    if mode == "none":
        if parsed.tool_calls or malformed_blocks:
            return _tool_validation_failure("invalid_tool_call")
        return _ToolValidationResult(parsed)
    if malformed_blocks:
        return _tool_validation_failure("invalid_tool_call")
    if not parsed.tool_calls:
        if mode in {"required", "function"}:
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
        schema = _tool_parameters_schema(tool)
        if schema is not None:
            schema_error = _validate_json_schema_value(_tool_call_arguments_value(call), schema, path=f"{call.name}.arguments")
            if schema_error is not None:
                return _tool_validation_failure("schema_violation")
    if mode == "function" and required_name is not None and not any(call.name == required_name for call in parsed.tool_calls):
        return _tool_validation_failure("tool_required_not_satisfied")
    return _ToolValidationResult(parsed)


def _tool_validation_failure(reason: str) -> _ToolValidationResult:
    return _ToolValidationResult(_ParsedChatOutput(text="", tool_calls=()), str(reason))


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
    malformed: list[str] = []
    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        if _parsed_tool_call_from_json(match.group(1).strip()) is None:
            malformed.append(match.group(0))
    if _has_unclosed_tool_call(text):
        open_index = text.lower().rfind("<tool_call>")
        malformed.append(text[open_index:] if open_index >= 0 else str(text))
    return tuple(malformed)


def _validate_json_schema_subset(schema: Mapping[str, Any], *, path: str) -> tuple[str, str] | None:
    for raw_key in schema:
        key = str(raw_key)
        if key not in _JSON_SCHEMA_SUPPORTED_KEYS:
            param = f"{path}.{key}"
            return (param, f"{param} is not supported by hipEngine JSON schema subset")

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

    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            return (f"{path}.properties", f"{path}.properties must be an object")
        for raw_key, subschema in properties.items():
            property_path = f"{path}.properties.{raw_key}"
            if not isinstance(subschema, Mapping):
                return (property_path, f"{property_path} must be an object")
            error = _validate_json_schema_subset(subschema, path=property_path)
            if error is not None:
                return error

    required = schema.get("required")
    if required is not None:
        if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
            return (f"{path}.required", f"{path}.required must be an array of strings")
        if any(not isinstance(item, str) for item in required):
            return (f"{path}.required", f"{path}.required must contain only strings")

    if "additionalProperties" in schema and not isinstance(schema.get("additionalProperties"), bool):
        return (
            f"{path}.additionalProperties",
            f"{path}.additionalProperties must be a boolean in hipEngine JSON schema subset",
        )

    items = schema.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            return (f"{path}.items", f"{path}.items must be an object")
        error = _validate_json_schema_subset(items, path=f"{path}.items")
        if error is not None:
            return error

    for key in ("minItems", "maxItems", "minLength", "maxLength"):
        if key in schema and _schema_nonnegative_int(schema.get(key)) is None:
            return (f"{path}.{key}", f"{path}.{key} must be a non-negative integer")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in schema and _schema_finite_number(schema.get(key)) is None:
            return (f"{path}.{key}", f"{path}.{key} must be a finite number")
    return None


def _validate_json_schema_value(value: Any, schema: Mapping[str, Any], *, path: str) -> str | None:
    enum = schema.get("enum")
    if isinstance(enum, Sequence) and not isinstance(enum, (str, bytes)) and value not in enum:
        return f"{path} is not one of the allowed enum values"
    if "const" in schema and value != schema.get("const"):
        return f"{path} does not match const"
    expected = schema.get("type")
    if expected is not None and not _json_schema_type_matches(value, expected):
        return f"{path} does not match schema type {expected!r}"
    schema_type = _primary_json_schema_type(expected, value)
    if schema_type == "object":
        if not isinstance(value, Mapping):
            return f"{path} must be an object"
        required = schema.get("required", ())
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
            for key in required:
                if isinstance(key, str) and key not in value:
                    return f"{path}.{key} is required"
        properties = schema.get("properties")
        property_map = properties if isinstance(properties, Mapping) else {}
        for key, subschema in property_map.items():
            if key in value and isinstance(key, str) and isinstance(subschema, Mapping):
                error = _validate_json_schema_value(value[key], subschema, path=f"{path}.{key}")
                if error is not None:
                    return error
        if schema.get("additionalProperties") is False:
            allowed = {str(key) for key in property_map}
            extra = sorted(str(key) for key in value.keys() if str(key) not in allowed)
            if extra:
                return f"{path}.{extra[0]} is not allowed"
    elif schema_type == "array":
        if not isinstance(value, list):
            return f"{path} must be an array"
        min_items = _schema_nonnegative_int(schema.get("minItems"))
        if min_items is not None and len(value) < min_items:
            return f"{path} must have at least {min_items} items"
        max_items = _schema_nonnegative_int(schema.get("maxItems"))
        if max_items is not None and len(value) > max_items:
            return f"{path} must have at most {max_items} items"
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                error = _validate_json_schema_value(item, items, path=f"{path}[{index}]")
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


@dataclass(slots=True)
class _StreamTokenAccounting:
    counter: Any
    streamed_tokens: int = 0
    reasoning_tokens: int = 0
    answer_tokens: int = 0
    tool_call_tokens: int = 0

    @classmethod
    def for_engine(cls, engine: Any) -> "_StreamTokenAccounting | None":
        counter = getattr(engine, "count_tokens", None)
        return cls(counter) if callable(counter) else None

    def observe(self, phase: str, text: str) -> dict[str, int]:
        delta_tokens = _safe_count(self.counter, text)
        self.streamed_tokens += delta_tokens
        phase_name = str(phase)
        if phase_name == "think":
            self.reasoning_tokens += delta_tokens
        elif phase_name == "tool_call":
            self.tool_call_tokens += delta_tokens
        else:
            self.answer_tokens += delta_tokens
        return self.snapshot(delta_tokens=delta_tokens)

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
    if usage is not None:
        payload["usage"] = dict(usage)
    return payload


def _choice_hipengine_payload(
    phase: str,
    *,
    finish_details: Mapping[str, Any] | None = None,
    tokens: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"phase": str(phase)}
    if finish_details is not None:
        payload["finish_details"] = dict(finish_details)
    if tokens is not None:
        token_payload = {str(key): max(0, int(value)) for key, value in tokens.items()}
        payload["tokens"] = token_payload
        payload["decode_state"] = DecodeState.from_stream_tokens(
            phase=phase,
            tokens=token_payload,
        ).to_json_dict()
    return payload


def _attach_choice_telemetry(choice: dict[str, Any], detail: GenerationOutput | None) -> None:
    telemetry = None if detail is None else detail.telemetry
    if telemetry is None:
        return
    payload = telemetry.to_json_dict()
    finish_details = choice.get("finish_details")
    if isinstance(finish_details, Mapping):
        payload["finish_details"] = dict(finish_details)
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
) -> dict[str, Any]:
    if include_hipengine:
        payload["hipengine"] = _stream_hipengine_payload(
            event,
            stream_started_at=stream_started_at,
            usage=usage,
            tokens=tokens,
            token_event=token_event,
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
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
) -> str:
    choice = {
        "text": text,
        "index": int(index),
        "logprobs": None if logprobs is None else dict(logprobs),
        "finish_reason": None,
    }
    if include_hipengine:
        choice["hipengine"] = _choice_hipengine_payload("answer", tokens=tokens)
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
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
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
        choice["hipengine"] = _choice_hipengine_payload(
            "done",
            finish_details=finish_payload,
            tokens=tokens,
        )
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
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
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
        )
    )


def _completion_stream(
    response_id: str,
    created: int,
    model: str,
    texts: Sequence[str],
    choices: Sequence[dict[str, Any]],
    *,
    usage: Mapping[str, int] | None = None,
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
) -> Iterator[str]:
    choices_by_index = {int(choice["index"]): choice for choice in choices}
    for index, text in enumerate(texts):
        choice = choices_by_index.get(index, {})
        yield _completion_stream_delta(
            response_id,
            created,
            model,
            text,
            index=index,
            logprobs=choice.get("logprobs"),
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
        )
    final_tokens = (
        _stream_usage_token_payload(usage, None)
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
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
        )
    if usage is not None:
        yield _completion_stream_usage(
            response_id,
            created,
            model,
            usage,
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
        )
    yield "data: [DONE]\n\n"


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
    tokens: Mapping[str, int] | None = None,
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
) -> str:
    choice: dict[str, Any] = {"index": int(index), "delta": {field: text}, "finish_reason": None}
    if logprobs is not None:
        choice["logprobs"] = dict(logprobs)
    if include_hipengine:
        phase = "think" if field == "reasoning_content" else "answer"
        choice["hipengine"] = _choice_hipengine_payload(phase, tokens=tokens)
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
    tokens: Mapping[str, int] | None = None,
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
) -> str:
    choice = {
        "index": int(index),
        "delta": {
            "tool_calls": [
                {
                    "index": int(tool_index),
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
            ]
        },
        "finish_reason": None,
    }
    if include_hipengine:
        choice["hipengine"] = _choice_hipengine_payload("tool_call", tokens=tokens)
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
        )
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
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
) -> Iterator[str]:
    split = _split_reasoning(parsed.text)
    if split.reasoning_content:
        yield _chat_stream_delta(
            response_id,
            created,
            model,
            "reasoning_content",
            split.reasoning_content,
            index=index,
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
        )
    if split.content:
        yield _chat_stream_delta(
            response_id,
            created,
            model,
            "content",
            split.content,
            index=index,
            logprobs=logprobs,
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
        )
    for tool_index, call in enumerate(parsed.tool_calls):
        yield _chat_stream_tool_call(
            response_id,
            created,
            model,
            call,
            index=index,
            tool_index=tool_index,
            include_hipengine=include_hipengine,
            stream_started_at=stream_started_at,
        )
    done_reason = "tool_calls" if parsed.tool_calls else finish_reason
    yield _chat_stream_done(
        response_id,
        created,
        model,
        done_reason,
        index=index,
        finish_details=finish_details,
        tokens=done_tokens,
        include_hipengine=include_hipengine,
        stream_started_at=stream_started_at,
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
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
) -> str:
    finish_payload = _finish_details_payload(None, finish_reason) if finish_details is None else dict(finish_details)
    choice = {
        "index": int(index),
        "delta": {},
        "finish_reason": finish_reason,
        "finish_details": finish_payload,
    }
    if include_hipengine:
        choice["hipengine"] = _choice_hipengine_payload(
            "done",
            finish_details=finish_payload,
            tokens=tokens,
        )
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
    include_hipengine: bool = False,
    stream_started_at: _StreamTimingSource = None,
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
