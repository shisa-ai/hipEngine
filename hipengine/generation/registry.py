"""Registry for torch-free text generation entry points."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class GenerationKey:
    """Concrete generation implementation key."""

    model: str
    backend: str
    quant: str
    mode: str = "greedy_one_token"


@dataclass(frozen=True)
class GenerationRequest:
    """Normalized public generation request."""

    prompts: tuple[str, ...]
    max_tokens: int
    temperature: float
    top_p: float
    ignore_eos: bool
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Any = ()
    suppress_token_ids: tuple[int, ...] = ()
    min_tokens: int = 0
    eos_token_id: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    stop_token_sequences: tuple[tuple[int, ...], ...] = ()
    forced_tokens_pending: tuple[int, ...] = ()
    forced_token_reason: str | None = None
    post_thinking_forced_tokens_pending: tuple[int, ...] = ()
    post_thinking_forced_token_reason: str | None = None
    force_sequence_completion_token_sequences: tuple[tuple[int, ...], ...] = ()
    force_sequence_completion_reason: str | None = None
    json_object_close_forcing: bool = False
    thinking_close_token_ids: tuple[int, ...] = ()
    thinking_hard_token_cap: int | None = None
    thinking_soft_close_window: int = 0
    kv_storage: str = "auto"
    kv_scale_dtype: str = "fp16"
    kv_scale_granularity: str = "per_token_head"
    seed: int | None = None
    row_seeds: tuple[int, ...] = ()
    deadline_at: float | None = None
    cancellation_token: Any | None = field(default=None, compare=False, repr=False)
    logprobs: bool = False
    top_logprobs: int = 0

    def __post_init__(self) -> None:
        from hipengine.generation.sampling import normalize_logit_bias_pairs, normalize_stop_token_sequences, validate_sampling_params

        object.__setattr__(self, "prompts", tuple(str(prompt) for prompt in self.prompts))
        object.__setattr__(self, "max_tokens", int(self.max_tokens))
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "top_p", float(self.top_p))
        object.__setattr__(self, "ignore_eos", bool(self.ignore_eos))
        object.__setattr__(self, "top_k", int(self.top_k))
        object.__setattr__(self, "min_p", float(self.min_p))
        object.__setattr__(self, "repetition_penalty", float(self.repetition_penalty))
        object.__setattr__(self, "presence_penalty", float(self.presence_penalty))
        object.__setattr__(self, "frequency_penalty", float(self.frequency_penalty))
        object.__setattr__(self, "logit_bias", normalize_logit_bias_pairs(self.logit_bias))
        object.__setattr__(self, "suppress_token_ids", tuple(int(token) for token in self.suppress_token_ids))
        object.__setattr__(self, "min_tokens", int(self.min_tokens))
        object.__setattr__(self, "eos_token_id", None if self.eos_token_id is None else int(self.eos_token_id))
        object.__setattr__(self, "stop_token_ids", tuple(int(token) for token in self.stop_token_ids))
        object.__setattr__(self, "stop_token_sequences", normalize_stop_token_sequences(self.stop_token_sequences))
        object.__setattr__(self, "forced_tokens_pending", _pending_token_tuple(self.forced_tokens_pending))
        object.__setattr__(self, "forced_token_reason", None if self.forced_token_reason is None else str(self.forced_token_reason))
        object.__setattr__(
            self,
            "post_thinking_forced_tokens_pending",
            _pending_token_tuple(self.post_thinking_forced_tokens_pending),
        )
        object.__setattr__(
            self,
            "post_thinking_forced_token_reason",
            None if self.post_thinking_forced_token_reason is None else str(self.post_thinking_forced_token_reason),
        )
        object.__setattr__(
            self,
            "force_sequence_completion_token_sequences",
            normalize_stop_token_sequences(self.force_sequence_completion_token_sequences),
        )
        object.__setattr__(
            self,
            "force_sequence_completion_reason",
            None if self.force_sequence_completion_reason is None else str(self.force_sequence_completion_reason),
        )
        object.__setattr__(self, "json_object_close_forcing", bool(self.json_object_close_forcing))
        object.__setattr__(
            self,
            "thinking_close_token_ids",
            tuple(int(token) for token in self.thinking_close_token_ids),
        )
        object.__setattr__(
            self,
            "thinking_hard_token_cap",
            None if self.thinking_hard_token_cap is None else int(self.thinking_hard_token_cap),
        )
        object.__setattr__(self, "thinking_soft_close_window", int(self.thinking_soft_close_window))
        object.__setattr__(self, "kv_storage", str(self.kv_storage))
        object.__setattr__(self, "kv_scale_dtype", str(self.kv_scale_dtype))
        object.__setattr__(self, "kv_scale_granularity", str(self.kv_scale_granularity))
        object.__setattr__(self, "seed", None if self.seed is None else int(self.seed))
        object.__setattr__(self, "row_seeds", tuple(int(seed) for seed in self.row_seeds))
        object.__setattr__(self, "deadline_at", None if self.deadline_at is None else float(self.deadline_at))
        object.__setattr__(self, "cancellation_token", self.cancellation_token)
        object.__setattr__(self, "logprobs", bool(self.logprobs))
        object.__setattr__(self, "top_logprobs", int(self.top_logprobs))
        validate_sampling_params(self)


@dataclass(frozen=True)
class TokenLogprob:
    """Host-visible logprob metadata for one generated token."""

    token_id: int
    token_text: str
    logprob: float | None = None
    top_logprobs: tuple[tuple[int, str, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_id", int(self.token_id))
        object.__setattr__(self, "token_text", str(self.token_text))
        object.__setattr__(
            self,
            "top_logprobs",
            tuple((int(token_id), str(text), float(logprob)) for token_id, text, logprob in self.top_logprobs),
        )


class DecodePhase(str, Enum):
    """Canonical generation phase labels for agent-facing telemetry."""

    PREFILL = "prefill"
    THINK = "think"
    CLOSING_THINK = "closing_think"
    ANSWER = "answer"
    TOOL_CALL = "tool_call"
    STRUCTURED = "structured"
    DONE = "done"


def _decode_phase_value(value: Any) -> str:
    return value.value if isinstance(value, DecodePhase) else str(value)


def _nonnegative_int(value: Any, default: int = 0) -> int:
    return max(0, int(default if value is None else value))


def _optional_nonnegative_int(value: Any) -> int | None:
    return None if value is None else _nonnegative_int(value)


def _pending_token_tuple(value: Any) -> tuple[int, ...]:
    return () if value is None else tuple(int(token) for token in value)


def _token_sequence_tuple(value: Any) -> tuple[tuple[int, ...], ...]:
    if value is None:
        return ()
    normalized: list[tuple[int, ...]] = []
    for raw_sequence in value:
        sequence = tuple(int(token) for token in raw_sequence)
        if sequence:
            normalized.append(sequence)
    return tuple(normalized)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class DecodeState:
    """Per-row decode phase and token-accounting snapshot."""

    request_id: str | None = None
    row_index: int = 0
    step_index: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    phase: DecodePhase | str = DecodePhase.DONE
    reasoning_tokens: int = 0
    answer_tokens: int = 0
    tool_call_tokens: int = 0
    structured_tokens: int = 0
    stop_suffix_state: Any | None = None
    forced_tokens_pending: tuple[int, ...] = ()
    forced_token_id: int | None = None
    forced_token_reason: str | None = None
    forced_tokens_remaining: int | None = None
    post_thinking_forced_tokens_pending: tuple[int, ...] = ()
    post_thinking_forced_token_reason: str | None = None
    force_sequence_completion_token_sequences: tuple[tuple[int, ...], ...] = ()
    force_sequence_completion_reason: str | None = None
    active_processors: tuple[str, ...] = ()
    sampler_fast_path_blockers: tuple[str, ...] = ()
    sampler_fallback_reason: str | None = None
    budget_pressure: str | None = None
    sampler_mode: str | None = None
    full_vocab_logits_d2h: bool | None = None
    logits_d2h_bytes: int | None = None
    execution_path: str | None = None
    native_compact_prefill: bool | None = None
    native_caware_decode: bool | None = None
    serial_decode_fallback: bool | None = None
    native_sampler_rows: bool | None = None
    continuation_eligible: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", None if self.request_id is None else str(self.request_id))
        object.__setattr__(self, "row_index", _nonnegative_int(self.row_index))
        object.__setattr__(self, "step_index", _nonnegative_int(self.step_index))
        object.__setattr__(self, "prompt_tokens", _nonnegative_int(self.prompt_tokens))
        object.__setattr__(self, "generated_tokens", _nonnegative_int(self.generated_tokens))
        object.__setattr__(self, "phase", _decode_phase_value(self.phase))
        object.__setattr__(self, "reasoning_tokens", _nonnegative_int(self.reasoning_tokens))
        object.__setattr__(self, "answer_tokens", _nonnegative_int(self.answer_tokens))
        object.__setattr__(self, "tool_call_tokens", _nonnegative_int(self.tool_call_tokens))
        object.__setattr__(self, "structured_tokens", _nonnegative_int(self.structured_tokens))
        object.__setattr__(self, "forced_tokens_pending", _pending_token_tuple(self.forced_tokens_pending))
        object.__setattr__(self, "forced_token_id", _optional_nonnegative_int(self.forced_token_id))
        object.__setattr__(
            self,
            "forced_token_reason",
            None if self.forced_token_reason is None else str(self.forced_token_reason),
        )
        object.__setattr__(self, "forced_tokens_remaining", _optional_nonnegative_int(self.forced_tokens_remaining))
        object.__setattr__(
            self,
            "post_thinking_forced_tokens_pending",
            _pending_token_tuple(self.post_thinking_forced_tokens_pending),
        )
        object.__setattr__(
            self,
            "post_thinking_forced_token_reason",
            None if self.post_thinking_forced_token_reason is None else str(self.post_thinking_forced_token_reason),
        )
        object.__setattr__(
            self,
            "force_sequence_completion_token_sequences",
            _token_sequence_tuple(self.force_sequence_completion_token_sequences),
        )
        object.__setattr__(
            self,
            "force_sequence_completion_reason",
            None if self.force_sequence_completion_reason is None else str(self.force_sequence_completion_reason),
        )
        object.__setattr__(self, "active_processors", _string_tuple(self.active_processors))
        object.__setattr__(self, "sampler_fast_path_blockers", _string_tuple(self.sampler_fast_path_blockers))
        object.__setattr__(
            self,
            "sampler_fallback_reason",
            None if self.sampler_fallback_reason is None else str(self.sampler_fallback_reason),
        )
        object.__setattr__(self, "budget_pressure", None if self.budget_pressure is None else str(self.budget_pressure))
        object.__setattr__(self, "sampler_mode", None if self.sampler_mode is None else str(self.sampler_mode))
        object.__setattr__(
            self,
            "full_vocab_logits_d2h",
            None if self.full_vocab_logits_d2h is None else bool(self.full_vocab_logits_d2h),
        )
        object.__setattr__(self, "logits_d2h_bytes", _optional_nonnegative_int(self.logits_d2h_bytes))
        object.__setattr__(self, "execution_path", None if self.execution_path is None else str(self.execution_path))
        object.__setattr__(
            self,
            "native_compact_prefill",
            None if self.native_compact_prefill is None else bool(self.native_compact_prefill),
        )
        object.__setattr__(
            self,
            "native_caware_decode",
            None if self.native_caware_decode is None else bool(self.native_caware_decode),
        )
        object.__setattr__(
            self,
            "serial_decode_fallback",
            None if self.serial_decode_fallback is None else bool(self.serial_decode_fallback),
        )
        object.__setattr__(
            self,
            "native_sampler_rows",
            None if self.native_sampler_rows is None else bool(self.native_sampler_rows),
        )
        object.__setattr__(self, "continuation_eligible", bool(self.continuation_eligible))

    @classmethod
    def from_value(cls, value: Any) -> "DecodeState":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                request_id=value.get("request_id"),
                row_index=value.get("row_index", 0),
                step_index=value.get("step_index", 0),
                prompt_tokens=value.get("prompt_tokens", 0),
                generated_tokens=value.get("generated_tokens", 0),
                phase=value.get("phase", DecodePhase.DONE.value),
                reasoning_tokens=value.get("reasoning_tokens", 0),
                answer_tokens=value.get("answer_tokens", 0),
                tool_call_tokens=value.get("tool_call_tokens", 0),
                structured_tokens=value.get("structured_tokens", 0),
                stop_suffix_state=value.get("stop_suffix_state"),
                forced_tokens_pending=value.get("forced_tokens_pending", ()),
                forced_token_id=value.get("forced_token_id"),
                forced_token_reason=value.get("forced_token_reason"),
                forced_tokens_remaining=value.get("forced_tokens_remaining"),
                post_thinking_forced_tokens_pending=value.get("post_thinking_forced_tokens_pending", ()),
                post_thinking_forced_token_reason=value.get("post_thinking_forced_token_reason"),
                force_sequence_completion_token_sequences=value.get("force_sequence_completion_token_sequences", ()),
                force_sequence_completion_reason=value.get("force_sequence_completion_reason"),
                active_processors=value.get("active_processors", ()),
                sampler_fast_path_blockers=value.get("sampler_fast_path_blockers", ()),
                sampler_fallback_reason=value.get("sampler_fallback_reason"),
                budget_pressure=value.get("budget_pressure"),
                sampler_mode=value.get("sampler_mode"),
                full_vocab_logits_d2h=value.get("full_vocab_logits_d2h"),
                logits_d2h_bytes=value.get("logits_d2h_bytes"),
                execution_path=value.get("execution_path"),
                native_compact_prefill=value.get("native_compact_prefill"),
                native_caware_decode=value.get("native_caware_decode"),
                serial_decode_fallback=value.get("serial_decode_fallback"),
                native_sampler_rows=value.get("native_sampler_rows"),
                continuation_eligible=bool(value.get("continuation_eligible", False)),
            )
        return cls(
            request_id=getattr(value, "request_id", None),
            row_index=getattr(value, "row_index", 0),
            step_index=getattr(value, "step_index", 0),
            prompt_tokens=getattr(value, "prompt_tokens", 0),
            generated_tokens=getattr(value, "generated_tokens", 0),
            phase=getattr(value, "phase", DecodePhase.DONE.value),
            reasoning_tokens=getattr(value, "reasoning_tokens", 0),
            answer_tokens=getattr(value, "answer_tokens", 0),
            tool_call_tokens=getattr(value, "tool_call_tokens", 0),
            structured_tokens=getattr(value, "structured_tokens", 0),
            stop_suffix_state=getattr(value, "stop_suffix_state", None),
            forced_tokens_pending=getattr(value, "forced_tokens_pending", ()),
            forced_token_id=getattr(value, "forced_token_id", None),
            forced_token_reason=getattr(value, "forced_token_reason", None),
            forced_tokens_remaining=getattr(value, "forced_tokens_remaining", None),
            post_thinking_forced_tokens_pending=getattr(value, "post_thinking_forced_tokens_pending", ()),
            post_thinking_forced_token_reason=getattr(value, "post_thinking_forced_token_reason", None),
            force_sequence_completion_token_sequences=getattr(value, "force_sequence_completion_token_sequences", ()),
            force_sequence_completion_reason=getattr(value, "force_sequence_completion_reason", None),
            active_processors=getattr(value, "active_processors", ()),
            sampler_fast_path_blockers=getattr(value, "sampler_fast_path_blockers", ()),
            sampler_fallback_reason=getattr(value, "sampler_fallback_reason", None),
            budget_pressure=getattr(value, "budget_pressure", None),
            sampler_mode=getattr(value, "sampler_mode", None),
            full_vocab_logits_d2h=getattr(value, "full_vocab_logits_d2h", None),
            logits_d2h_bytes=getattr(value, "logits_d2h_bytes", None),
            execution_path=getattr(value, "execution_path", None),
            native_compact_prefill=getattr(value, "native_compact_prefill", None),
            native_caware_decode=getattr(value, "native_caware_decode", None),
            serial_decode_fallback=getattr(value, "serial_decode_fallback", None),
            native_sampler_rows=getattr(value, "native_sampler_rows", None),
            continuation_eligible=bool(getattr(value, "continuation_eligible", False)),
        )

    @classmethod
    def from_stream_tokens(
        cls,
        *,
        phase: DecodePhase | str,
        tokens: Mapping[str, int],
        row_index: int = 0,
    ) -> "DecodeState":
        streamed_tokens = _nonnegative_int(tokens.get("streamed_tokens", tokens.get("generated_tokens", 0)))
        generated_tokens = _nonnegative_int(tokens.get("completion_tokens", streamed_tokens))
        return cls(
            row_index=row_index,
            step_index=streamed_tokens,
            prompt_tokens=_nonnegative_int(tokens.get("prompt_tokens", 0)),
            generated_tokens=generated_tokens,
            phase=phase,
            reasoning_tokens=_nonnegative_int(tokens.get("reasoning_tokens", 0)),
            answer_tokens=_nonnegative_int(tokens.get("answer_tokens", 0)),
            tool_call_tokens=_nonnegative_int(tokens.get("tool_call_tokens", 0)),
            structured_tokens=_nonnegative_int(tokens.get("structured_tokens", 0)),
            continuation_eligible=bool(tokens.get("continuation_eligible", False)),
        )

    def to_json_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "row_index": self.row_index,
            "step_index": self.step_index,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "phase": _decode_phase_value(self.phase),
            "continuation_eligible": self.continuation_eligible,
        }
        if self.request_id is not None:
            payload["request_id"] = self.request_id
        if self.reasoning_tokens:
            payload["reasoning_tokens"] = self.reasoning_tokens
        if self.answer_tokens:
            payload["answer_tokens"] = self.answer_tokens
        if self.tool_call_tokens:
            payload["tool_call_tokens"] = self.tool_call_tokens
        if self.structured_tokens:
            payload["structured_tokens"] = self.structured_tokens
        if self.stop_suffix_state is not None:
            payload["stop_suffix_state"] = self.stop_suffix_state
        if self.forced_tokens_pending:
            payload["forced_tokens_pending"] = list(self.forced_tokens_pending)
        if self.forced_token_id is not None:
            payload["forced_token_id"] = self.forced_token_id
        if self.forced_token_reason is not None:
            payload["forced_token_reason"] = self.forced_token_reason
        if self.forced_tokens_remaining is not None:
            payload["forced_tokens_remaining"] = self.forced_tokens_remaining
        if self.post_thinking_forced_tokens_pending:
            payload["post_thinking_forced_tokens_pending"] = list(self.post_thinking_forced_tokens_pending)
        if self.post_thinking_forced_token_reason is not None:
            payload["post_thinking_forced_token_reason"] = self.post_thinking_forced_token_reason
        if self.force_sequence_completion_token_sequences:
            payload["force_sequence_completion_token_sequences"] = [
                list(sequence) for sequence in self.force_sequence_completion_token_sequences
            ]
        if self.force_sequence_completion_reason is not None:
            payload["force_sequence_completion_reason"] = self.force_sequence_completion_reason
        if self.active_processors:
            payload["active_processors"] = list(self.active_processors)
        if self.sampler_fast_path_blockers:
            payload["sampler_fast_path_blockers"] = list(self.sampler_fast_path_blockers)
        if self.sampler_fallback_reason is not None:
            payload["sampler_fallback_reason"] = self.sampler_fallback_reason
        if self.budget_pressure is not None:
            payload["budget_pressure"] = self.budget_pressure
        if self.sampler_mode is not None:
            payload["sampler_mode"] = self.sampler_mode
        if self.full_vocab_logits_d2h is not None:
            payload["full_vocab_logits_d2h"] = self.full_vocab_logits_d2h
        if self.logits_d2h_bytes is not None:
            payload["logits_d2h_bytes"] = self.logits_d2h_bytes
        if self.execution_path is not None:
            payload["execution_path"] = self.execution_path
        if self.native_compact_prefill is not None:
            payload["native_compact_prefill"] = self.native_compact_prefill
        if self.native_caware_decode is not None:
            payload["native_caware_decode"] = self.native_caware_decode
        if self.serial_decode_fallback is not None:
            payload["serial_decode_fallback"] = self.serial_decode_fallback
        if self.native_sampler_rows is not None:
            payload["native_sampler_rows"] = self.native_sampler_rows
        return payload


@dataclass(frozen=True)
class GenerationTelemetry:
    """Agent-facing generation telemetry snapshot."""

    decode_state: DecodeState
    event: str | None = None
    timing: Mapping[str, float] | None = None
    usage: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decode_state", DecodeState.from_value(self.decode_state))
        object.__setattr__(self, "event", None if self.event is None else str(self.event))
        timing = None if self.timing is None else {str(key): float(value) for key, value in self.timing.items()}
        usage = None if self.usage is None else {str(key): max(0, int(value)) for key, value in self.usage.items()}
        object.__setattr__(self, "timing", timing)
        object.__setattr__(self, "usage", usage)

    @classmethod
    def from_value(cls, value: Any) -> "GenerationTelemetry":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                decode_state=value.get("decode_state", value),
                event=value.get("event"),
                timing=value.get("timing"),
                usage=value.get("usage"),
            )
        return cls(
            decode_state=getattr(value, "decode_state", value),
            event=getattr(value, "event", None),
            timing=getattr(value, "timing", None),
            usage=getattr(value, "usage", None),
        )

    @classmethod
    def from_decode_counts(
        cls,
        *,
        prompt_tokens: int,
        generated_tokens: int,
        row_index: int = 0,
        request_id: str | None = None,
        phase: DecodePhase | str = DecodePhase.DONE,
        sampler_mode: str | None = None,
        reasoning_tokens: int = 0,
        answer_tokens: int = 0,
        tool_call_tokens: int = 0,
        structured_tokens: int = 0,
        stop_suffix_state: Any | None = None,
        forced_tokens_pending: tuple[int, ...] = (),
        forced_token_id: int | None = None,
        forced_token_reason: str | None = None,
        forced_tokens_remaining: int | None = None,
        post_thinking_forced_tokens_pending: tuple[int, ...] = (),
        post_thinking_forced_token_reason: str | None = None,
        force_sequence_completion_token_sequences: tuple[tuple[int, ...], ...] = (),
        force_sequence_completion_reason: str | None = None,
        active_processors: tuple[str, ...] = (),
        sampler_fast_path_blockers: tuple[str, ...] = (),
        sampler_fallback_reason: str | None = None,
        budget_pressure: str | None = None,
        full_vocab_logits_d2h: bool | None = None,
        logits_d2h_bytes: int | None = None,
        execution_path: str | None = None,
        native_compact_prefill: bool | None = None,
        native_caware_decode: bool | None = None,
        serial_decode_fallback: bool | None = None,
        native_sampler_rows: bool | None = None,
        continuation_eligible: bool = False,
        event: str | None = None,
        timing: Mapping[str, float] | None = None,
        usage: Mapping[str, int] | None = None,
    ) -> "GenerationTelemetry":
        return cls(
            decode_state=DecodeState(
                request_id=request_id,
                row_index=row_index,
                step_index=generated_tokens,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                phase=phase,
                reasoning_tokens=reasoning_tokens,
                answer_tokens=answer_tokens,
                tool_call_tokens=tool_call_tokens,
                structured_tokens=structured_tokens,
                stop_suffix_state=stop_suffix_state,
                forced_tokens_pending=forced_tokens_pending,
                forced_token_id=forced_token_id,
                forced_token_reason=forced_token_reason,
                forced_tokens_remaining=forced_tokens_remaining,
                post_thinking_forced_tokens_pending=post_thinking_forced_tokens_pending,
                post_thinking_forced_token_reason=post_thinking_forced_token_reason,
                force_sequence_completion_token_sequences=force_sequence_completion_token_sequences,
                force_sequence_completion_reason=force_sequence_completion_reason,
                active_processors=active_processors,
                sampler_fast_path_blockers=sampler_fast_path_blockers,
                sampler_fallback_reason=sampler_fallback_reason,
                budget_pressure=budget_pressure,
                sampler_mode=sampler_mode,
                full_vocab_logits_d2h=full_vocab_logits_d2h,
                logits_d2h_bytes=logits_d2h_bytes,
                execution_path=execution_path,
                native_compact_prefill=native_compact_prefill,
                native_caware_decode=native_caware_decode,
                serial_decode_fallback=serial_decode_fallback,
                native_sampler_rows=native_sampler_rows,
                continuation_eligible=continuation_eligible,
            ),
            event=event,
            timing=timing,
            usage=usage,
        )

    def to_json_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"decode_state": self.decode_state.to_json_dict()}
        if self.event is not None:
            payload["event"] = self.event
        if self.timing is not None:
            payload["timing"] = dict(self.timing)
        if self.usage is not None:
            payload["usage"] = dict(self.usage)
        return payload


@dataclass(frozen=True)
class FinishDetails:
    """Structured reason and accounting metadata for a completed generation."""

    reason: str
    eos_token_id: int | None = None
    stop_sequence: tuple[int, ...] = ()
    length_limit: int | None = None
    deadline_exceeded: bool = False
    cancelled: bool = False
    forced_close: bool = False
    synthetic_tokens: int = 0
    reasoning_tokens: int = 0
    answer_tokens: int = 0
    tool_call_tokens: int = 0
    structured_tokens: int = 0
    budget_pressure: str | None = None
    cache_action: str | None = None
    sampler_mode: str | None = None
    phase: str | None = None
    continuation_eligible: bool | None = None

    def __post_init__(self) -> None:
        reason = "stop" if self.reason is None or str(self.reason).strip() == "" else str(self.reason)
        stop_sequence = () if self.stop_sequence is None else self.stop_sequence
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "eos_token_id", None if self.eos_token_id is None else int(self.eos_token_id))
        object.__setattr__(self, "stop_sequence", tuple(int(token) for token in stop_sequence))
        object.__setattr__(self, "length_limit", None if self.length_limit is None else int(self.length_limit))
        object.__setattr__(self, "deadline_exceeded", bool(self.deadline_exceeded))
        object.__setattr__(self, "cancelled", bool(self.cancelled))
        object.__setattr__(self, "forced_close", bool(self.forced_close))
        object.__setattr__(self, "synthetic_tokens", int(self.synthetic_tokens))
        object.__setattr__(self, "reasoning_tokens", int(self.reasoning_tokens))
        object.__setattr__(self, "answer_tokens", int(self.answer_tokens))
        object.__setattr__(self, "tool_call_tokens", int(self.tool_call_tokens))
        object.__setattr__(self, "structured_tokens", int(self.structured_tokens))
        object.__setattr__(self, "budget_pressure", None if self.budget_pressure is None else str(self.budget_pressure))
        object.__setattr__(self, "cache_action", None if self.cache_action is None else str(self.cache_action))
        object.__setattr__(self, "sampler_mode", None if self.sampler_mode is None else str(self.sampler_mode))
        object.__setattr__(self, "phase", None if self.phase is None else str(self.phase))
        object.__setattr__(
            self,
            "continuation_eligible",
            None if self.continuation_eligible is None else bool(self.continuation_eligible),
        )

    @classmethod
    def from_value(cls, value: Any) -> "FinishDetails":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                reason=str(value.get("reason", "stop")),
                eos_token_id=value.get("eos_token_id"),
                stop_sequence=tuple(value.get("stop_sequence", ())),
                length_limit=value.get("length_limit"),
                deadline_exceeded=bool(value.get("deadline_exceeded", False)),
                cancelled=bool(value.get("cancelled", False)),
                forced_close=bool(value.get("forced_close", False)),
                synthetic_tokens=int(value.get("synthetic_tokens", 0)),
                reasoning_tokens=int(value.get("reasoning_tokens", 0)),
                answer_tokens=int(value.get("answer_tokens", 0)),
                tool_call_tokens=int(value.get("tool_call_tokens", 0)),
                structured_tokens=int(value.get("structured_tokens", 0)),
                budget_pressure=value.get("budget_pressure"),
                cache_action=value.get("cache_action"),
                sampler_mode=value.get("sampler_mode"),
                phase=value.get("phase"),
                continuation_eligible=(
                    value.get("continuation_eligible")
                    if "continuation_eligible" in value
                    else None
                ),
            )
        return cls(
            reason=str(getattr(value, "reason", "stop")),
            eos_token_id=getattr(value, "eos_token_id", None),
            stop_sequence=tuple(getattr(value, "stop_sequence", ())),
            length_limit=getattr(value, "length_limit", None),
            deadline_exceeded=bool(getattr(value, "deadline_exceeded", False)),
            cancelled=bool(getattr(value, "cancelled", False)),
            forced_close=bool(getattr(value, "forced_close", False)),
            synthetic_tokens=int(getattr(value, "synthetic_tokens", 0)),
            reasoning_tokens=int(getattr(value, "reasoning_tokens", 0)),
            answer_tokens=int(getattr(value, "answer_tokens", 0)),
            tool_call_tokens=int(getattr(value, "tool_call_tokens", 0)),
            structured_tokens=int(getattr(value, "structured_tokens", 0)),
            budget_pressure=getattr(value, "budget_pressure", None),
            cache_action=getattr(value, "cache_action", None),
            sampler_mode=getattr(value, "sampler_mode", None),
            phase=getattr(value, "phase", None),
            continuation_eligible=getattr(value, "continuation_eligible", None),
        )

    def to_json_dict(self, *, reason: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"reason": self.reason if reason is None else str(reason)}
        if self.eos_token_id is not None:
            payload["eos_token_id"] = self.eos_token_id
        if self.stop_sequence:
            payload["stop_sequence"] = list(self.stop_sequence)
        if self.length_limit is not None:
            payload["length_limit"] = self.length_limit
        if self.deadline_exceeded:
            payload["deadline_exceeded"] = True
        if self.cancelled:
            payload["cancelled"] = True
        if self.forced_close:
            payload["forced_close"] = True
        if self.synthetic_tokens:
            payload["synthetic_tokens"] = self.synthetic_tokens
        if self.reasoning_tokens:
            payload["reasoning_tokens"] = self.reasoning_tokens
        if self.answer_tokens:
            payload["answer_tokens"] = self.answer_tokens
        if self.tool_call_tokens:
            payload["tool_call_tokens"] = self.tool_call_tokens
        if self.structured_tokens:
            payload["structured_tokens"] = self.structured_tokens
        if self.budget_pressure is not None:
            payload["budget_pressure"] = self.budget_pressure
        if self.cache_action is not None:
            payload["cache_action"] = self.cache_action
        if self.sampler_mode is not None:
            payload["sampler_mode"] = self.sampler_mode
        if self.phase is not None:
            payload["phase"] = self.phase
        if self.continuation_eligible is not None:
            payload["continuation_eligible"] = self.continuation_eligible
        return payload


@dataclass(frozen=True)
class GenerationOutput:
    """Generated text plus optional per-token sampler and finish metadata."""

    text: str
    token_logprobs: tuple[TokenLogprob, ...] = ()
    finish_details: FinishDetails | None = None
    telemetry: GenerationTelemetry | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", str(self.text))
        object.__setattr__(self, "token_logprobs", tuple(self.token_logprobs))
        if self.finish_details is not None:
            object.__setattr__(self, "finish_details", FinishDetails.from_value(self.finish_details))
        if self.telemetry is not None:
            object.__setattr__(self, "telemetry", GenerationTelemetry.from_value(self.telemetry))

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class GenerationStreamChunk:
    """Incremental generated text plus optional live backend metadata."""

    text: str
    token_logprobs: tuple[TokenLogprob, ...] = ()
    finish_details: FinishDetails | None = None
    telemetry: GenerationTelemetry | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", str(self.text))
        object.__setattr__(self, "token_logprobs", tuple(self.token_logprobs))
        if self.finish_details is not None:
            object.__setattr__(self, "finish_details", FinishDetails.from_value(self.finish_details))
        if self.telemetry is not None:
            object.__setattr__(self, "telemetry", GenerationTelemetry.from_value(self.telemetry))

    @classmethod
    def from_value(cls, value: Any) -> "GenerationStreamChunk":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                text=str(value.get("text", "")),
                token_logprobs=tuple(value.get("token_logprobs", ()) or ()),
                finish_details=value.get("finish_details"),
                telemetry=value.get("telemetry"),
            )
        return cls(text=str(value))

    def __str__(self) -> str:
        return self.text


class TextGenerator(Protocol):
    """Protocol implemented by backend/model-specific text generators."""

    def generate(self, request: GenerationRequest) -> list[str]:
        """Generate text for each prompt in ``request``."""


GeneratorFactory = Callable[..., TextGenerator]


class DuplicateGeneratorError(ValueError):
    pass


class MissingGeneratorError(LookupError):
    pass


_FACTORIES: dict[GenerationKey, GeneratorFactory] = {}


def register_text_generator(
    *,
    model: str,
    backend: str,
    quant: str,
    factory: GeneratorFactory,
    mode: str = "greedy_one_token",
    replace: bool = False,
) -> GeneratorFactory:
    """Register a model/backend/quant text generation factory."""

    key = GenerationKey(model=model, backend=backend, quant=quant, mode=mode)
    if key in _FACTORIES and not replace:
        raise DuplicateGeneratorError(f"generation implementation already registered for {key}")
    _FACTORIES[key] = factory
    return factory


def resolve_text_generator(
    *,
    model: str,
    backend: str,
    quant: str,
    mode: str = "greedy_one_token",
) -> GeneratorFactory:
    """Resolve a generation factory for the exact model/backend/quant key."""

    key = GenerationKey(model=model, backend=backend, quant=quant, mode=mode)
    try:
        return _FACTORIES[key]
    except KeyError as exc:
        known = ", ".join(
            f"({item.model}, {item.backend}, {item.quant}, {item.mode})"
            for item in sorted(_FACTORIES, key=lambda k: (k.model, k.backend, k.quant, k.mode))
        )
        raise MissingGeneratorError(
            f"no generation implementation for ({model}, {backend}, {quant}, {mode}); "
            f"known: {known or '<none>'}"
        ) from exc


def registered_text_generators() -> tuple[GenerationKey, ...]:
    return tuple(sorted(_FACTORIES, key=lambda k: (k.model, k.backend, k.quant, k.mode)))


def clear_generation_registry_for_tests() -> None:
    _FACTORIES.clear()


# Type-only helper signature for factories.  Keeping it here documents the kwargs LLM passes
# without forcing runtime dependencies on loading/model classes in this registry module.
def make_text_generator(
    *,
    model_path: str | Path,
    weight_index: Any,
    model_plugin: Any,
) -> TextGenerator:  # pragma: no cover - documentation helper only
    raise NotImplementedError
