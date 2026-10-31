"""Pure SSE frame projection and stream accounting.

This owner turns already-derived stream values into OpenAI-compatible SSE
frames.  It does not parse text or tool calls, choose finish reasons, build
error payloads, read requests, or touch the engine/batcher.  The application
module keeps live sequencing, cancellation, backpressure, and policy.

The timing tracker is an explicitly passed mutable clock sink.  It is still an
owner-local object with no engine or application state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
import time
from typing import Any

from hipengine.generation import DecodeState, GenerationStreamChunk
from hipengine.server.chat_output import _ParsedToolCall


TOOL_CALL_ARGUMENT_STREAM_CHARS = 128


def sse_frame(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


@dataclass(slots=True)
class StreamTokenAccounting:
    """Count emitted tokens through an already-safe counter."""

    counter: Callable[[str], int]
    streamed_tokens: int = 0
    reasoning_tokens: int = 0
    answer_tokens: int = 0
    tool_call_tokens: int = 0
    structured_tokens: int = 0

    def observe(
        self,
        phase: str,
        text: str,
        *,
        cumulative_tokens: int | None = None,
    ) -> dict[str, int]:
        if cumulative_tokens is None or int(cumulative_tokens) < self.streamed_tokens:
            delta_tokens = max(0, int(self.counter(text)))
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
class StreamTimingTracker:
    started_at: float
    first_token_at: float | None = None
    first_chunk_tokens: int | None = None
    last_token_at: float | None = None
    last_chunk_tokens: int = 0
    observed_tokens: int = 0
    buffered: bool = False

    @classmethod
    def start(cls) -> "StreamTimingTracker":
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
                self.first_chunk_tokens = token_total if token_total > 0 else None
            if token_total > self.last_chunk_tokens:
                self.last_token_at = now
                self.last_chunk_tokens = token_total
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
        if str(event) in {"done", "usage"} and not self.buffered and self.last_token_at is not None:
            # The interval ends at the last token-bearing event, not at the
            # final SSE flush. Tokens in the first chunk precede this interval.
            decode_elapsed_s = max(0.0, self.last_token_at - self.first_token_at)
            payload["decode_elapsed_ms"] = round(decode_elapsed_s * 1000.0, 3)
            if self.first_chunk_tokens is not None:
                decode_tokens = self.last_chunk_tokens - self.first_chunk_tokens
                if decode_tokens > 0 and decode_elapsed_s > 0:
                    payload["decode_tokens_per_second"] = round(decode_tokens / decode_elapsed_s, 3)
        return payload


def _observe_stream_source(
    tracker: "StreamTimingSource", chunk: GenerationStreamChunk | None,
) -> None:
    """A completed first backend chunk is buffered, even if split into channels."""
    if (
        not isinstance(tracker, StreamTimingTracker)
        or tracker.first_token_at is not None
        or chunk is None
    ):
        return
    if chunk.finish_details is not None or (
        chunk.telemetry is not None and chunk.telemetry.decode_state.phase == "done"
    ):
        tracker.buffered = True


StreamTimingSource = float | StreamTimingTracker | None


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


def stream_usage_token_payload(
    usage: Mapping[str, int],
    accounting: StreamTokenAccounting | None,
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
    stream_started_at: StreamTimingSource = None,
    usage: Mapping[str, int] | None = None,
    tokens: Mapping[str, int] | None = None,
    token_event: bool = False,
    backend_timing: Mapping[str, float] | None = None,
    routing: Mapping[str, Any] | None = None,
    kv_pool: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"metadata_version": 1, "event": str(event)}
    if stream_started_at is not None:
        if isinstance(stream_started_at, StreamTimingTracker):
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


def stream_chunk_backend_timing(
    stream_chunk: GenerationStreamChunk | None,
) -> Mapping[str, float] | None:
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


def _attach_stream_hipengine(
    payload: dict[str, Any],
    *,
    include_hipengine: bool,
    event: str,
    stream_started_at: StreamTimingSource,
    usage: Mapping[str, int] | None = None,
    tokens: Mapping[str, int] | None = None,
    token_event: bool = False,
    backend_timing: Mapping[str, float] | None = None,
    routing: Mapping[str, Any] | None = None,
    kv_pool: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    # A tracker is also created when only the --info summary needs it, so it has
    # to be fed even when the client did not request stream metadata; the
    # payload itself stays opt-in.
    if include_hipengine or isinstance(stream_started_at, StreamTimingTracker):
        hipengine_payload = _stream_hipengine_payload(
            event,
            stream_started_at=stream_started_at,
            usage=usage,
            tokens=tokens,
            token_event=token_event,
            backend_timing=backend_timing,
            routing=routing,
            kv_pool=kv_pool,
        )
        if include_hipengine:
            payload["hipengine"] = hipengine_payload
    return payload


def completion_delta(
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
    stream_started_at: StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
    phase: str = "answer",
) -> str:
    _observe_stream_source(stream_started_at, stream_chunk)
    choice = {
        "text": text,
        "index": int(index),
        "logprobs": None if logprobs is None else dict(logprobs),
        "finish_reason": None,
    }
    if include_hipengine:
        choice["hipengine"] = _choice_hipengine_payload(
            phase,
            tokens=tokens,
            stream_chunk=stream_chunk,
        )
    return sse_frame(
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
            backend_timing=stream_chunk_backend_timing(stream_chunk),
            routing=routing,
        )
    )


def completion_done(
    response_id: str,
    created: int,
    model: str,
    finish_reason: str,
    *,
    finish_details: Mapping[str, Any],
    index: int = 0,
    tokens: Mapping[str, int] | None = None,
    stream_chunk: GenerationStreamChunk | None = None,
    include_hipengine: bool = False,
    extra_hipengine: Mapping[str, Any] | None = None,
    stream_started_at: StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
    kv_pool: Mapping[str, float] | None = None,
    phase: str = "done",
) -> str:
    finish_payload = dict(finish_details)
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
    return sse_frame(
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
            backend_timing=stream_chunk_backend_timing(stream_chunk),
            routing=routing,
            kv_pool=kv_pool,
        )
    )


def completion_usage(
    response_id: str,
    created: int,
    model: str,
    usage: Mapping[str, int],
    *,
    include_hipengine: bool = False,
    stream_started_at: StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
    kv_pool: Mapping[str, float] | None = None,
) -> str:
    return sse_frame(
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


def completion_error(
    response_id: str,
    created: int,
    model: str,
    *,
    error: Mapping[str, Any],
    finish_details: Mapping[str, Any] | None = None,
    include_hipengine: bool = False,
    stream_started_at: StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
) -> str:
    choice: dict[str, Any] = {
        "text": "",
        "index": 0,
        "logprobs": None,
        "finish_reason": "error",
    }
    if finish_details is not None:
        choice["finish_details"] = dict(finish_details)
    if include_hipengine:
        choice["hipengine"] = _choice_hipengine_payload(
            "done",
            finish_details=choice.get("finish_details"),
        )
    return sse_frame(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [choice],
                "error": dict(error),
            },
            include_hipengine=include_hipengine,
            event="error",
            stream_started_at=stream_started_at,
            routing=routing,
        )
    )


def chat_role(
    response_id: str,
    created: int,
    model: str,
    *,
    index: int = 0,
    include_hipengine: bool = False,
    stream_started_at: StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
) -> str:
    return sse_frame(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": int(index),
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            },
            include_hipengine=include_hipengine,
            event="role",
            stream_started_at=stream_started_at,
            routing=routing,
        )
    )


def chat_delta(
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
    stream_started_at: StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
    phase: str | None = None,
) -> str:
    _observe_stream_source(stream_started_at, stream_chunk)
    choice: dict[str, Any] = {"index": int(index), "delta": {field: text}, "finish_reason": None}
    if logprobs is not None:
        choice["logprobs"] = dict(logprobs)
    if include_hipengine:
        choice_phase = (
            phase
            if phase is not None
            else "think"
            if field == "reasoning_content"
            else "answer"
        )
        choice["hipengine"] = _choice_hipengine_payload(
            choice_phase,
            tokens=tokens,
            stream_chunk=stream_chunk,
        )
        if reasoning_logprobs is not None:
            choice["hipengine"]["reasoning_logprobs"] = dict(reasoning_logprobs)
    return sse_frame(
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
            backend_timing=stream_chunk_backend_timing(stream_chunk),
            routing=routing,
        )
    )


def chat_tool_call(
    response_id: str,
    created: int,
    model: str,
    call: _ParsedToolCall,
    *,
    index: int = 0,
    tool_index: int = 0,
    argument_chunk: str | None = None,
    include_name: bool = True,
    logprobs: Mapping[str, Any] | None = None,
    tokens: Mapping[str, int] | None = None,
    stream_chunk: GenerationStreamChunk | None = None,
    include_hipengine: bool = False,
    stream_started_at: StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
) -> str:
    _observe_stream_source(stream_started_at, stream_chunk)
    function: dict[str, Any] = {
        "arguments": call.arguments if argument_chunk is None else str(argument_chunk)
    }
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
    if logprobs is not None:
        choice["logprobs"] = dict(logprobs)
    if include_hipengine:
        choice["hipengine"] = _choice_hipengine_payload(
            "tool_call",
            tokens=tokens,
            stream_chunk=stream_chunk,
        )
    return sse_frame(
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
            backend_timing=stream_chunk_backend_timing(stream_chunk),
            routing=routing,
        )
    )


def tool_call_argument_chunks(arguments: str) -> tuple[str, ...]:
    text = str(arguments)
    if len(text) <= TOOL_CALL_ARGUMENT_STREAM_CHARS:
        return (text,)
    return tuple(
        text[start : start + TOOL_CALL_ARGUMENT_STREAM_CHARS]
        for start in range(0, len(text), TOOL_CALL_ARGUMENT_STREAM_CHARS)
    )


def chat_done(
    response_id: str,
    created: int,
    model: str,
    finish_reason: str,
    *,
    finish_details: Mapping[str, Any],
    index: int = 0,
    tokens: Mapping[str, int] | None = None,
    stream_chunk: GenerationStreamChunk | None = None,
    include_hipengine: bool = False,
    extra_hipengine: Mapping[str, Any] | None = None,
    stream_started_at: StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
    kv_pool: Mapping[str, float] | None = None,
    phase: str = "done",
) -> str:
    finish_payload = dict(finish_details)
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
    return sse_frame(
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
            backend_timing=stream_chunk_backend_timing(stream_chunk),
            routing=routing,
            kv_pool=kv_pool,
        )
    )


def chat_usage(
    response_id: str,
    created: int,
    model: str,
    usage: Mapping[str, int],
    *,
    include_hipengine: bool = False,
    stream_started_at: StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
    kv_pool: Mapping[str, float] | None = None,
) -> str:
    return sse_frame(
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


def chat_error(
    response_id: str,
    created: int,
    model: str,
    *,
    error: Mapping[str, Any],
    finish_details: Mapping[str, Any] | None = None,
    include_hipengine: bool = False,
    stream_started_at: StreamTimingSource = None,
    routing: Mapping[str, Any] | None = None,
) -> str:
    choice: dict[str, Any] = {
        "index": 0,
        "delta": {"content": ""},
        "finish_reason": "error",
    }
    if finish_details is not None:
        choice["finish_details"] = dict(finish_details)
    if include_hipengine:
        choice["hipengine"] = _choice_hipengine_payload(
            "done",
            finish_details=choice.get("finish_details"),
        )
    return sse_frame(
        _attach_stream_hipengine(
            {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [choice],
                "error": dict(error),
            },
            include_hipengine=include_hipengine,
            event="error",
            stream_started_at=stream_started_at,
            routing=routing,
        )
    )


def stream_chunk_with_phase(
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
