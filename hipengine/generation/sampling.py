"""Torch-free sampling utilities for generation paths.

The helpers in this module operate on CPU/NumPy logits for the functional host
sampler path and define the request planning contract shared with future native
GPU samplers.  They intentionally avoid torch.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

import numpy as np

from hipengine.generation.constraints import (
    ForcedTokenQueue,
    JsonObjectConstraintState,
    ThinkingBudgetState,
    TokenSequenceDFAState,
    normalize_token_sequences,
)

_LOGIT_BIAS_EMPTY: tuple[tuple[int, float], ...] = ()
_UINT64_MASK = (1 << 64) - 1
_SEED_MASK = (1 << 63) - 1
_MAX_NATIVE_GPU_TOP_K = 64
NATIVE_GPU_SAMPLER_UNSUPPORTED_CAPABILITIES: tuple[str, ...] = (
    "true_batched_c_gt_1",
    "gguf",
    "top_logprobs_exceed_native_limit",
    "top_logprobs_exceed_top_k",
    "forced_tokens_pending",
    "post_thinking_forced_tokens_pending",
    "force_sequence_completion_token_sequences",
    "json_object_close_forcing",
    "thinking_budget",
)
SPECULATIVE_MTP_INCOMPATIBLE_FIELDS: tuple[str, ...] = (
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
)
SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS: dict[str, str] = {
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
}


class SamplingMode(str, Enum):
    """Token-selection execution modes."""

    GREEDY_FAST = "greedy_fast"
    PROCESSED_ARGMAX = "processed_argmax"
    HOST_LOGITS_SAMPLE = "host_logits_sample"
    GPU_SAMPLE = "gpu_sample"


@dataclass(frozen=True, slots=True)
class SamplerPlan:
    """Pure request-level sampler decision."""

    mode: SamplingMode
    active_processors: tuple[str, ...] = ()
    native_gpu_available: bool = False
    fast_path_blockers: tuple[str, ...] = ()
    fallback_reason: str | None = None

    @property
    def uses_host_logits(self) -> bool:
        return self.mode in {SamplingMode.PROCESSED_ARGMAX, SamplingMode.HOST_LOGITS_SAMPLE}


@dataclass(slots=True)
class RowSamplingState:
    """Mutable per-row sampling state used by host sampling."""

    prompt_tokens: Sequence[int] = ()
    seed: int = 0
    request_id: int = 0
    row_index: int = 0
    generated_tokens: Sequence[int] = ()
    step_index: int = 0
    stop_token_sequences: Sequence[Sequence[int]] = ()
    forced_tokens_pending: Sequence[int] | ForcedTokenQueue = ()
    forced_token_reason: str | None = None
    post_thinking_forced_tokens_pending: Sequence[int] | ForcedTokenQueue = ()
    post_thinking_forced_token_reason: str | None = None
    force_sequence_completion_token_sequences: Sequence[Sequence[int]] = ()
    force_sequence_completion_reason: str | None = None
    json_object_close_forcing: bool = False
    thinking_budget: ThinkingBudgetState | None = None
    _rng: np.random.Generator = field(init=False, repr=False)
    _stop_sequence_state: TokenSequenceDFAState = field(init=False, repr=False)
    _force_sequence_state: TokenSequenceDFAState = field(init=False, repr=False)
    _json_object_constraint: JsonObjectConstraintState | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.prompt_tokens = tuple(int(token) for token in self.prompt_tokens)
        self.generated_tokens = [int(token) for token in self.generated_tokens]
        self.seed = int(self.seed) & _SEED_MASK
        self.request_id = int(self.request_id)
        self.row_index = int(self.row_index)
        self.step_index = int(self.step_index)
        try:
            self.stop_token_sequences = normalize_token_sequences(self.stop_token_sequences)
        except ValueError as exc:
            raise ValueError("stop_token_sequences must contain non-negative token ids") from exc
        try:
            self.force_sequence_completion_token_sequences = normalize_token_sequences(
                self.force_sequence_completion_token_sequences
            )
        except ValueError as exc:
            raise ValueError(
                "force_sequence_completion_token_sequences must contain non-negative token ids"
            ) from exc
        self.force_sequence_completion_reason = (
            None if self.force_sequence_completion_reason is None else str(self.force_sequence_completion_reason)
        )
        self.json_object_close_forcing = bool(self.json_object_close_forcing)
        if isinstance(self.forced_tokens_pending, ForcedTokenQueue):
            forced = self.forced_tokens_pending
        else:
            forced = ForcedTokenQueue(
                self.forced_tokens_pending,
                reason=self.forced_token_reason,
            )
        if isinstance(self.post_thinking_forced_tokens_pending, ForcedTokenQueue):
            post_thinking_forced = self.post_thinking_forced_tokens_pending
        else:
            post_thinking_forced = ForcedTokenQueue(
                self.post_thinking_forced_tokens_pending,
                reason=self.post_thinking_forced_token_reason,
            )
        if self.thinking_budget is not None:
            budget_forced = self.thinking_budget.forced_tokens
            if forced is not budget_forced and forced.pending_tokens:
                budget_forced.extend(forced.pending_tokens, reason=forced.reason)
            forced = budget_forced
        self.forced_tokens_pending = forced
        self.forced_token_reason = forced.reason
        self.post_thinking_forced_tokens_pending = post_thinking_forced
        self.post_thinking_forced_token_reason = post_thinking_forced.reason
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        self._force_sequence_state = TokenSequenceDFAState.from_sequences(self.force_sequence_completion_token_sequences)
        self._stop_sequence_state = TokenSequenceDFAState.from_sequences(self.stop_token_sequences)
        if self.generated_tokens:
            self._stop_sequence_state = self._stop_sequence_state.observe_many(self.generated_tokens)
            self._force_sequence_state = self._force_sequence_state.observe_many(self.generated_tokens)
            if self._force_sequence_state.matched:
                self._force_sequence_state = TokenSequenceDFAState.from_sequences(
                    self.force_sequence_completion_token_sequences
                )
        self._json_object_constraint = JsonObjectConstraintState() if self.json_object_close_forcing else None
        self._rng = np.random.Generator(np.random.PCG64(self.seed))
        if self.step_index:
            # Keep reconstructed state deterministic when a caller restores a
            # step index without serializing the bit-generator internals.
            self._rng.random(self.step_index)

    @property
    def history_tokens(self) -> tuple[int, ...]:
        return (*self.prompt_tokens, *tuple(self.generated_tokens))

    def history_counts(self) -> Counter[int]:
        return Counter(int(token) for token in self.history_tokens)

    def random_unit(self) -> float:
        return float(self._rng.random())

    def observe(self, token_id: int) -> None:
        token = int(token_id)
        self.generated_tokens.append(token)
        self.step_index += 1
        if self.thinking_budget is not None:
            self.thinking_budget.observe(token)
        self._stop_sequence_state = self._stop_sequence_state.observe(token)
        self._queue_force_sequence_completion_if_partial(token)

    @property
    def forced_tokens(self) -> tuple[int, ...]:
        return self.forced_tokens_pending.pending_tokens

    @property
    def has_forced_tokens(self) -> bool:
        return bool(self.forced_tokens_pending)

    @property
    def stop_suffix_state(self) -> dict[str, Any] | None:
        payload = self._stop_sequence_state.to_json_dict()
        return payload or None

    def queue_forced_tokens(self, token_ids: Iterable[int], *, reason: str | None = None) -> None:
        self.forced_tokens_pending.extend(token_ids, reason=reason)
        if reason is not None:
            self.forced_token_reason = str(reason)

    def peek_forced_token(self) -> int | None:
        return self.forced_tokens_pending.peek()

    def pop_forced_token(self) -> int | None:
        token_id = self.forced_tokens_pending.pop()
        if not self.forced_tokens_pending:
            self.forced_token_reason = None
        return token_id

    def prepare_for_selection(self) -> None:
        if self.thinking_budget is None:
            return
        self.thinking_budget.ensure_hard_close()
        self.forced_tokens_pending = self.thinking_budget.forced_tokens
        self._queue_post_thinking_forced_tokens_if_ready()
        self.forced_token_reason = self.forced_tokens_pending.reason

    def _queue_post_thinking_forced_tokens_if_ready(self) -> None:
        if self.thinking_budget is None or self.thinking_budget.phase != "answer":
            return
        pending = self.post_thinking_forced_tokens_pending
        if not pending or self.forced_tokens_pending:
            return
        self.forced_tokens_pending.extend(pending.pending_tokens, reason=pending.reason)
        self.post_thinking_forced_tokens_pending = ForcedTokenQueue()
        self.post_thinking_forced_token_reason = None

    def _queue_force_sequence_completion_if_partial(self, token_id: int) -> None:
        if not self.force_sequence_completion_token_sequences:
            return
        self._force_sequence_state = self._force_sequence_state.observe(int(token_id))
        if self._force_sequence_state.matched:
            self._force_sequence_state = TokenSequenceDFAState.from_sequences(
                self.force_sequence_completion_token_sequences
            )
            return
        suffix = self._force_sequence_state.suffix
        if not suffix or self.forced_tokens_pending:
            return
        for sequence in self.force_sequence_completion_token_sequences:
            if sequence[: len(suffix)] != suffix:
                continue
            remaining = sequence[len(suffix) :]
            if remaining:
                self.queue_forced_tokens(
                    remaining,
                    reason=self.force_sequence_completion_reason or "sequence_completion",
                )
            return

    def observe_text_for_json_object_close(
        self,
        text: str,
        *,
        remaining_tokens: int,
        encode_text: Callable[[str], Iterable[int]],
    ) -> None:
        """Queue a structural JSON close suffix when the decode budget requires it."""

        constraint = self._json_object_constraint
        if constraint is None:
            return
        constraint.observe_text(str(text))
        if constraint.invalid or constraint.complete or self.forced_tokens_pending:
            return
        suffix = constraint.forced_close_suffix
        if not suffix:
            return
        try:
            token_ids = tuple(int(token) for token in encode_text(suffix))
        except Exception:
            return
        remaining = int(remaining_tokens)
        if token_ids and remaining == len(token_ids):
            self.queue_forced_tokens(token_ids, reason="json_object_close_forcing")


@dataclass(frozen=True, slots=True)
class SampleResult:
    """Result of selecting one token from one logits row."""

    token_id: int
    logit: float
    logprob: float | None
    mode: SamplingMode
    candidate_count: int
    top_logprobs: tuple[tuple[int, float], ...] = ()
    forced: bool = False
    forced_reason: str | None = None
    forced_tokens_remaining: int = 0
    active_processors: tuple[str, ...] = ()
    fast_path_blockers: tuple[str, ...] = ()


LogitBiasInput = Mapping[int | str, float] | Iterable[tuple[int | str, float]] | None
StopTokenSequencesInput = Iterable[Iterable[int]] | None


def normalize_logit_bias_pairs(logit_bias: LogitBiasInput = None) -> tuple[tuple[int, float], ...]:
    """Return a sorted token-id keyed logit-bias tuple.

    JSON object keys arrive from OpenAI-style requests as strings, while the
    library API may pass integer keys.  Token-string aliases are deliberately not
    accepted here; tokenizer-level lowering can add that later.
    """

    if logit_bias is None:
        return _LOGIT_BIAS_EMPTY
    if isinstance(logit_bias, Mapping):
        iterable = logit_bias.items()
    else:
        iterable = logit_bias
    values: dict[int, float] = {}
    for raw_token, raw_bias in iterable:
        if isinstance(raw_token, bool):
            raise ValueError("logit_bias token ids must be integers, not booleans")
        try:
            token_id = int(raw_token)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"logit_bias token id {raw_token!r} is not an integer") from exc
        if token_id < 0:
            raise ValueError("logit_bias token ids must be non-negative")
        bias = float(raw_bias)
        if not math.isfinite(bias):
            raise ValueError("logit_bias values must be finite")
        values[token_id] = bias
    return tuple(sorted(values.items()))


def normalize_stop_token_sequences(
    stop_token_sequences: StopTokenSequencesInput = None,
) -> tuple[tuple[int, ...], ...]:
    """Normalize non-empty token-id stop sequences."""

    if stop_token_sequences is None:
        return ()
    normalized: list[tuple[int, ...]] = []
    for raw_sequence in stop_token_sequences:
        sequence = tuple(int(token) for token in raw_sequence)
        if not sequence:
            continue
        if any(token < 0 for token in sequence):
            raise ValueError("stop_token_sequences must contain non-negative token ids")
        if sequence not in normalized:
            normalized.append(sequence)
    return tuple(normalized)


def validate_sampling_params(params: Any) -> None:
    """Validate the canonical sampler fields on ``params``.

    The function accepts either public ``SamplingParams`` or normalized
    ``GenerationRequest`` instances to avoid a dependency cycle.
    """

    temperature = float(getattr(params, "temperature", 0.0))
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    top_p = float(getattr(params, "top_p", 1.0))
    if not math.isfinite(top_p) or top_p < 0.0 or top_p > 1.0:
        raise ValueError("top_p must be finite and between 0 and 1")
    top_k = int(getattr(params, "top_k", 0))
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    min_p = float(getattr(params, "min_p", 0.0))
    if not math.isfinite(min_p) or min_p < 0.0 or min_p > 1.0:
        raise ValueError("min_p must be finite and between 0 and 1")
    repetition_penalty = float(getattr(params, "repetition_penalty", 1.0))
    if not math.isfinite(repetition_penalty) or repetition_penalty <= 0.0:
        raise ValueError("repetition_penalty must be finite and positive")
    for name in ("presence_penalty", "frequency_penalty"):
        value = float(getattr(params, name, 0.0))
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    for token_id in _suppress_token_ids(params):
        if int(token_id) < 0:
            raise ValueError("suppress_token_ids must be non-negative")
    for token_id in _forced_tokens_pending(params):
        if int(token_id) < 0:
            raise ValueError("forced_tokens_pending must be non-negative")
    post_thinking_forced_tokens = _post_thinking_forced_tokens_pending(params)
    for token_id in post_thinking_forced_tokens:
        if int(token_id) < 0:
            raise ValueError("post_thinking_forced_tokens_pending must be non-negative")
    min_tokens = int(getattr(params, "min_tokens", 0))
    if min_tokens < 0:
        raise ValueError("min_tokens must be non-negative")
    eos_token_id = getattr(params, "eos_token_id", None)
    if eos_token_id is not None and int(eos_token_id) < 0:
        raise ValueError("eos_token_id must be non-negative")
    if min_tokens > 0 and eos_token_id is None:
        raise ValueError("min_tokens requires eos_token_id")
    seed = getattr(params, "seed", None)
    if seed is not None and int(seed) < 0:
        raise ValueError("seed must be non-negative")
    for seed_value in getattr(params, "row_seeds", ()):
        if int(seed_value) < 0:
            raise ValueError("row_seeds must be non-negative")
    deadline_at = getattr(params, "deadline_at", None)
    if deadline_at is not None and not math.isfinite(float(deadline_at)):
        raise ValueError("deadline_at must be finite when set")
    top_logprobs = int(getattr(params, "top_logprobs", 0))
    if top_logprobs < 0:
        raise ValueError("top_logprobs must be non-negative")
    for token_id in _stop_token_ids(params):
        if int(token_id) < 0:
            raise ValueError("stop_token_ids must be non-negative")
    normalize_stop_token_sequences(getattr(params, "stop_token_sequences", None))
    force_sequence_completion_token_sequences = _force_sequence_completion_token_sequences(params)
    if any(token < 0 for sequence in force_sequence_completion_token_sequences for token in sequence):
        raise ValueError("force_sequence_completion_token_sequences must contain non-negative token ids")
    normalize_logit_bias_pairs(getattr(params, "logit_bias", None))
    close_token_ids = _thinking_close_token_ids(params)
    if any(token_id < 0 for token_id in close_token_ids):
        raise ValueError("thinking_close_token_ids must be non-negative")
    hard_token_cap = getattr(params, "thinking_hard_token_cap", None)
    if hard_token_cap is not None and int(hard_token_cap) < 0:
        raise ValueError("thinking_hard_token_cap must be non-negative")
    if hard_token_cap is not None and not close_token_ids:
        raise ValueError("thinking_hard_token_cap requires thinking_close_token_ids")
    if post_thinking_forced_tokens and (hard_token_cap is None or not close_token_ids):
        raise ValueError("post_thinking_forced_tokens_pending requires a thinking budget")
    soft_close_window = int(getattr(params, "thinking_soft_close_window", 0))
    if soft_close_window < 0:
        raise ValueError("thinking_soft_close_window must be non-negative")


def active_processor_names(params: Any) -> tuple[str, ...]:
    """Return active logit processors for planning/observability."""

    names: list[str] = []
    if normalize_logit_bias_pairs(getattr(params, "logit_bias", None)):
        names.append("logit_bias")
    if float(getattr(params, "repetition_penalty", 1.0)) != 1.0:
        names.append("repetition_penalty")
    if float(getattr(params, "presence_penalty", 0.0)) != 0.0:
        names.append("presence_penalty")
    if float(getattr(params, "frequency_penalty", 0.0)) != 0.0:
        names.append("frequency_penalty")
    if _suppress_token_ids(params):
        names.append("suppress_token_ids")
    if int(getattr(params, "min_tokens", 0)) > 0:
        names.append("min_tokens")
    if _stop_token_ids(params):
        names.append("stop_token_ids")
    if normalize_stop_token_sequences(getattr(params, "stop_token_sequences", None)):
        names.append("stop_token_sequences")
    if thinking_budget_active(params):
        names.append("thinking_budget")
    if _forced_tokens_pending(params):
        names.append("forced_tokens_pending")
    if _post_thinking_forced_tokens_pending(params):
        names.append("post_thinking_forced_tokens_pending")
    if _force_sequence_completion_token_sequences(params):
        names.append("force_sequence_completion_token_sequences")
    if bool(getattr(params, "json_object_close_forcing", False)):
        names.append("json_object_close_forcing")
    return tuple(names)


def sampler_fast_path_blockers(params: Any) -> tuple[str, ...]:
    """Return request fields that make the graph argmax fast path inexact."""

    validate_sampling_params(params)
    blockers: list[str] = []
    if float(getattr(params, "temperature", 0.0)) > 0.0:
        blockers.append("temperature")
    blockers.extend(active_processor_names(params))
    if bool(getattr(params, "logprobs", False)):
        blockers.append("logprobs")
    if int(getattr(params, "top_logprobs", 0)) > 0:
        blockers.append("top_logprobs")
    return tuple(dict.fromkeys(blockers))


def _stop_token_ids(params: Any) -> tuple[int, ...]:
    raw_ids = getattr(params, "stop_token_ids", None)
    if raw_ids is None:
        raw_ids = getattr(params, "stop_tokens", ())
    return tuple(int(token) for token in raw_ids)


def _suppress_token_ids(params: Any) -> tuple[int, ...]:
    raw_ids = getattr(params, "suppress_token_ids", None)
    if raw_ids is None:
        raw_ids = getattr(params, "suppress_tokens", ())
    return tuple(int(token) for token in raw_ids)


def _forced_tokens_pending(params: Any) -> tuple[int, ...]:
    queue = getattr(params, "forced_tokens_pending", ())
    if isinstance(queue, ForcedTokenQueue):
        return queue.pending_tokens
    if queue is None:
        return ()
    return tuple(int(token) for token in queue)


def _post_thinking_forced_tokens_pending(params: Any) -> tuple[int, ...]:
    queue = getattr(params, "post_thinking_forced_tokens_pending", ())
    if isinstance(queue, ForcedTokenQueue):
        return queue.pending_tokens
    if queue is None:
        return ()
    return tuple(int(token) for token in queue)


def _force_sequence_completion_token_sequences(params: Any) -> tuple[tuple[int, ...], ...]:
    try:
        return normalize_token_sequences(getattr(params, "force_sequence_completion_token_sequences", None))
    except ValueError as exc:
        raise ValueError("force_sequence_completion_token_sequences must contain non-negative token ids") from exc


def _thinking_close_token_ids(params: Any) -> tuple[int, ...]:
    token_ids = getattr(params, "thinking_close_token_ids", ())
    if token_ids is None:
        return ()
    return tuple(int(token) for token in token_ids)


def thinking_budget_active(params: Any) -> bool:
    """Return whether request params activate token-level thinking control."""

    return bool(_thinking_close_token_ids(params)) and getattr(params, "thinking_hard_token_cap", None) is not None


def thinking_budget_state_from_params(params: Any) -> ThinkingBudgetState | None:
    """Build a fresh mutable thinking-budget state for one decode row."""

    if not thinking_budget_active(params):
        return None
    return ThinkingBudgetState(
        close_sequence=_thinking_close_token_ids(params),
        hard_token_cap=getattr(params, "thinking_hard_token_cap", None),
        soft_close_window=int(getattr(params, "thinking_soft_close_window", 0)),
    )


def clone_thinking_budget_state(state: ThinkingBudgetState | None) -> ThinkingBudgetState | None:
    """Clone mutable thinking-budget state without sharing forced-token queues."""

    if state is None:
        return None
    cloned = ThinkingBudgetState(
        close_sequence=state.close_sequence,
        hard_token_cap=state.hard_token_cap,
        soft_close_window=state.soft_close_window,
        phase=state.phase,
        reasoning_tokens=state.reasoning_tokens,
        answer_tokens=state.answer_tokens,
        forced_tokens=ForcedTokenQueue(state.forced_tokens.pending_tokens, reason=state.forced_tokens.reason),
    )
    cloned.close_state = state.close_state
    return cloned


def supports_native_gpu_sampling(params: Any) -> bool:
    """Return whether current standalone GPU sampler kernels cover ``params``.

    The native route is intentionally narrower than the host sampler: selected
    logprobs are available, full-vocab top-logprobs are available for public
    ``top_k=0`` requests, and bounded top-k top-logprobs are available when
    ``top_logprobs <= top_k <= 64``.
    """

    validate_sampling_params(params)
    if float(getattr(params, "temperature", 0.0)) <= 0.0:
        return False
    if _forced_tokens_pending(params):
        return False
    if _post_thinking_forced_tokens_pending(params):
        return False
    if _force_sequence_completion_token_sequences(params):
        return False
    if bool(getattr(params, "json_object_close_forcing", False)):
        return False
    if thinking_budget_active(params):
        return False
    top_k = int(getattr(params, "top_k", 0))
    if top_k > _MAX_NATIVE_GPU_TOP_K:
        return False
    top_logprobs = int(getattr(params, "top_logprobs", 0))
    if top_logprobs > _MAX_NATIVE_GPU_TOP_K:
        return False
    if top_logprobs > 0 and top_k > 0 and top_logprobs > top_k:
        return False
    return True


def plan_sampler(
    params: Any,
    *,
    native_gpu_available: bool = False,
    native_gpu_requested: bool = False,
    native_only: bool = False,
) -> SamplerPlan:
    """Choose the token-selection mode for a request."""

    validate_sampling_params(params)
    processors = active_processor_names(params)
    fast_path_blockers = sampler_fast_path_blockers(params)
    temperature = float(getattr(params, "temperature", 0.0))
    needs_logits = bool(getattr(params, "logprobs", False)) or int(getattr(params, "top_logprobs", 0)) > 0
    if temperature <= 0.0:
        if processors or needs_logits:
            return SamplerPlan(
                SamplingMode.PROCESSED_ARGMAX,
                processors,
                native_gpu_available,
                fast_path_blockers,
                "processed_logits_required",
            )
        return SamplerPlan(SamplingMode.GREEDY_FAST, processors, native_gpu_available, fast_path_blockers)
    native_ready = native_gpu_available and supports_native_gpu_sampling(params)
    if native_ready:
        return SamplerPlan(SamplingMode.GPU_SAMPLE, processors, native_gpu_available, fast_path_blockers)
    if native_only:
        raise NotImplementedError("native GPU sampling is not available for this request")
    fallback_reason = (
        "native_gpu_unsupported_request"
        if native_gpu_available or native_gpu_requested
        else "host_sampling_required"
    )
    return SamplerPlan(
        SamplingMode.HOST_LOGITS_SAMPLE,
        processors,
        native_gpu_available,
        fast_path_blockers,
        fallback_reason,
    )


def speculative_mtp_sampling_blockers(params: Any) -> tuple[str, ...]:
    """Return request fields that make raw-argmax MTP verification inexact.

    Current MTP proposer/verifier paths produce raw target top-1 decisions.  They
    are exact for normal serving only when the autoregressive request would use
    the same greedy fast path, with no processed logits or sampler metadata.
    """

    blockers = list(sampler_fast_path_blockers(params))
    if getattr(params, "eos_token_id", None) is not None:
        blockers.append("eos_token_id")
    if bool(getattr(params, "ignore_eos", False)):
        blockers.append("ignore_eos")
    return tuple(dict.fromkeys(blockers))


def supports_speculative_mtp_sampling(params: Any) -> bool:
    """Return whether a request may use today's raw-argmax MTP route."""

    return not speculative_mtp_sampling_blockers(params)


def derive_row_seed(
    base_seed: int | None,
    row_index: int,
    *,
    request_id: int = 0,
) -> int:
    """Derive a deterministic non-negative row seed."""

    base = 0 if base_seed is None else int(base_seed)
    if base < 0:
        raise ValueError("base seed must be non-negative")
    row = int(row_index)
    if row < 0:
        raise ValueError("row_index must be non-negative")
    req = int(request_id)
    value = (
        base
        + 0x9E3779B97F4A7C15
        + row * 0xBF58476D1CE4E5B9
        + req * 0x94D049BB133111EB
    ) & _UINT64_MASK
    value ^= value >> 30
    value = (value * 0x94D049BB133111EB) & _UINT64_MASK
    value ^= value >> 31
    return int(value & _SEED_MASK)


def row_seed_for_index(params: Any, row_index: int, *, request_id: int = 0) -> int:
    row_seeds = tuple(int(seed) for seed in getattr(params, "row_seeds", ()))
    if row_index < len(row_seeds):
        seed = row_seeds[row_index]
        if seed < 0:
            raise ValueError("row_seeds must be non-negative")
        return seed
    return derive_row_seed(getattr(params, "seed", None), row_index, request_id=request_id)


def select_token(
    logits: np.ndarray | Sequence[float],
    params: Any,
    state: RowSamplingState | None = None,
) -> SampleResult:
    """Select one token from a single logits row using the documented order."""

    validate_sampling_params(params)
    row_state = (
        state
        if state is not None
        else RowSamplingState(
            seed=derive_row_seed(getattr(params, "seed", None), 0),
            forced_tokens_pending=_forced_tokens_pending(params),
            forced_token_reason=getattr(params, "forced_token_reason", None),
            stop_token_sequences=normalize_stop_token_sequences(getattr(params, "stop_token_sequences", None)),
            post_thinking_forced_tokens_pending=_post_thinking_forced_tokens_pending(params),
            post_thinking_forced_token_reason=getattr(params, "post_thinking_forced_token_reason", None),
            force_sequence_completion_token_sequences=_force_sequence_completion_token_sequences(params),
            force_sequence_completion_reason=getattr(params, "force_sequence_completion_reason", None),
            json_object_close_forcing=bool(getattr(params, "json_object_close_forcing", False)),
        )
    )
    source = np.asarray(logits, dtype=np.float32)
    if source.ndim != 1:
        raise ValueError("logits must be a one-dimensional row")
    if source.size <= 0:
        raise ValueError("logits row must not be empty")

    processed = source.astype(np.float64, copy=True)
    finite = np.isfinite(processed)
    if not np.any(finite):
        raise ValueError("logits row contains no finite values")
    processed[~finite] = -np.inf

    _apply_logit_bias(processed, normalize_logit_bias_pairs(getattr(params, "logit_bias", None)))
    _apply_history_penalties(processed, params, row_state)
    _apply_suppression_processors(processed, params, row_state)

    row_state.prepare_for_selection()
    eos_suppressed = _apply_thinking_budget_eos_suppression(processed, params, row_state)
    json_eos_suppressed = _apply_json_object_eos_suppression(processed, params, row_state)
    soft_close_biased = _apply_thinking_budget_soft_close_bias(processed, row_state)
    active_processors = active_processor_names(params)
    fast_path_blockers = sampler_fast_path_blockers(params)
    if eos_suppressed or soft_close_biased:
        active_processors = _append_unique(active_processors, "thinking_budget")
        fast_path_blockers = _append_unique(fast_path_blockers, "thinking_budget")
    if json_eos_suppressed:
        active_processors = _append_unique(active_processors, "json_object_close_forcing")
        fast_path_blockers = _append_unique(fast_path_blockers, "json_object_close_forcing")
    requested_logprobs = bool(getattr(params, "logprobs", False)) or int(getattr(params, "top_logprobs", 0)) > 0
    requested_top_logprobs = int(getattr(params, "top_logprobs", 0))
    temperature = float(getattr(params, "temperature", 0.0))
    forced_token_id = row_state.peek_forced_token()
    if forced_token_id is not None:
        token_id = int(forced_token_id)
        if token_id < 0 or token_id >= source.size:
            raise ValueError(f"forced token id {token_id} is outside vocab size {source.size}")
        forced_reason = row_state.forced_token_reason
        row_state.pop_forced_token()
        row_state.observe(token_id)
        logprob, top_logprobs = (None, ())
        if requested_logprobs and np.isfinite(processed[token_id]):
            logprob, top_logprobs = _logprob_summary(processed, token_id, requested_top_logprobs)
        result_processors = _append_unique(active_processors, "forced_tokens_pending")
        result_blockers = _append_unique(fast_path_blockers, "forced_tokens_pending")
        return SampleResult(
            token_id=token_id,
            logit=float(processed[token_id]),
            logprob=logprob,
            mode=SamplingMode.PROCESSED_ARGMAX if temperature <= 0.0 else SamplingMode.HOST_LOGITS_SAMPLE,
            candidate_count=int(np.isfinite(processed).sum()),
            top_logprobs=top_logprobs,
            forced=True,
            forced_reason=forced_reason,
            forced_tokens_remaining=len(row_state.forced_tokens_pending),
            active_processors=result_processors,
            fast_path_blockers=result_blockers,
        )
    if temperature <= 0.0:
        token_id = _argmax_lower_id(processed)
        logprob, top_logprobs = _logprob_summary(processed, token_id, requested_top_logprobs) if requested_logprobs else (None, ())
        row_state.observe(token_id)
        return SampleResult(
            token_id=token_id,
            logit=float(processed[token_id]),
            logprob=logprob,
            mode=SamplingMode.GREEDY_FAST if not active_processors and not requested_logprobs else SamplingMode.PROCESSED_ARGMAX,
            candidate_count=int(np.isfinite(processed).sum()),
            top_logprobs=top_logprobs,
            active_processors=active_processors,
            fast_path_blockers=fast_path_blockers,
        )

    scaled = processed / temperature
    candidate_ids = _top_k_candidate_ids(scaled, int(getattr(params, "top_k", 0)))
    if candidate_ids.size == 0:
        raise ValueError("sampling filters removed all finite logits")
    candidate_logits = scaled[candidate_ids]
    candidate_probs = _softmax(candidate_logits)
    retained_ids, retained_probs = _apply_probability_filters(
        candidate_ids,
        candidate_probs,
        top_p=float(getattr(params, "top_p", 1.0)),
        min_p=float(getattr(params, "min_p", 0.0)),
    )
    probs_sum = float(retained_probs.sum())
    if not math.isfinite(probs_sum) or probs_sum <= 0.0:
        raise ValueError("sampling probabilities are not normalizable")
    retained_probs = retained_probs / probs_sum
    draw = row_state.random_unit()
    cumulative = np.cumsum(retained_probs)
    choice = int(np.searchsorted(cumulative, draw, side="right"))
    if choice >= retained_ids.size:
        choice = retained_ids.size - 1
    token_id = int(retained_ids[choice])
    probability = float(retained_probs[choice])
    row_state.observe(token_id)
    top_logprobs = _top_logprob_pairs(retained_ids, retained_probs, requested_top_logprobs)
    return SampleResult(
        token_id=token_id,
        logit=float(processed[token_id]),
        logprob=float(math.log(probability)),
        mode=SamplingMode.HOST_LOGITS_SAMPLE,
        candidate_count=int(retained_ids.size),
        top_logprobs=top_logprobs,
        active_processors=active_processors,
        fast_path_blockers=fast_path_blockers,
    )


def _append_unique(values: tuple[str, ...], value: str) -> tuple[str, ...]:
    return values if value in values else (*values, value)


def _apply_logit_bias(logits: np.ndarray, pairs: tuple[tuple[int, float], ...]) -> None:
    vocab = int(logits.size)
    for token_id, bias in pairs:
        if token_id >= vocab:
            raise ValueError(f"logit_bias token id {token_id} is outside vocab size {vocab}")
        logits[token_id] += float(bias)


def _apply_history_penalties(logits: np.ndarray, params: Any, state: RowSamplingState) -> None:
    counts = state.history_counts()
    if not counts:
        return
    repetition_penalty = float(getattr(params, "repetition_penalty", 1.0))
    presence_penalty = float(getattr(params, "presence_penalty", 0.0))
    frequency_penalty = float(getattr(params, "frequency_penalty", 0.0))
    vocab = int(logits.size)
    for token_id, count in counts.items():
        if token_id < 0 or token_id >= vocab:
            continue
        if repetition_penalty != 1.0:
            if logits[token_id] < 0.0:
                logits[token_id] *= repetition_penalty
            else:
                logits[token_id] /= repetition_penalty
        if presence_penalty != 0.0:
            logits[token_id] -= presence_penalty
        if frequency_penalty != 0.0:
            logits[token_id] -= frequency_penalty * int(count)


def _apply_suppression_processors(logits: np.ndarray, params: Any, state: RowSamplingState) -> None:
    vocab = int(logits.size)
    for token_id in _suppress_token_ids(params):
        if token_id >= vocab:
            raise ValueError(f"suppress_token_ids token id {token_id} is outside vocab size {vocab}")
        logits[token_id] = -np.inf
    min_tokens = int(getattr(params, "min_tokens", 0))
    eos_token_id = getattr(params, "eos_token_id", None)
    if min_tokens > 0 and eos_token_id is not None and int(state.step_index) < min_tokens:
        token_id = int(eos_token_id)
        if token_id >= vocab:
            raise ValueError(f"eos_token_id {token_id} is outside vocab size {vocab}")
        logits[token_id] = -np.inf


def _apply_thinking_budget_eos_suppression(logits: np.ndarray, params: Any, state: RowSamplingState) -> bool:
    budget = state.thinking_budget
    eos_token_id = getattr(params, "eos_token_id", None)
    if budget is None or not budget.eos_suppression_active or budget.forced_tokens or eos_token_id is None:
        return False
    token_id = int(eos_token_id)
    if token_id < 0 or token_id >= int(logits.size):
        raise ValueError(f"eos_token_id {token_id} is outside vocab size {logits.size}")
    logits[token_id] = -np.inf
    return True


def _apply_json_object_eos_suppression(logits: np.ndarray, params: Any, state: RowSamplingState) -> bool:
    if not bool(getattr(params, "json_object_close_forcing", False)):
        return False
    constraint = state._json_object_constraint
    eos_token_id = getattr(params, "eos_token_id", None)
    if constraint is None or constraint.complete or constraint.invalid or state.forced_tokens_pending or eos_token_id is None:
        return False
    token_id = int(eos_token_id)
    if token_id < 0 or token_id >= int(logits.size):
        raise ValueError(f"eos_token_id {token_id} is outside vocab size {logits.size}")
    logits[token_id] = -np.inf
    return True


def _apply_thinking_budget_soft_close_bias(logits: np.ndarray, state: RowSamplingState) -> bool:
    budget = state.thinking_budget
    if budget is None or budget.soft_close_bias is None or budget.forced_tokens:
        return False
    if not budget.close_sequence:
        return False
    token_id = int(budget.close_sequence[0])
    if token_id < 0 or token_id >= int(logits.size):
        raise ValueError(f"thinking close token id {token_id} is outside vocab size {logits.size}")
    if np.isfinite(logits[token_id]):
        logits[token_id] += float(budget.soft_close_bias)
    return True


def _argmax_lower_id(values: np.ndarray) -> int:
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("no finite logits remain after processing")
    return int(np.argmax(values))


def _top_k_candidate_ids(values: np.ndarray, top_k: int) -> np.ndarray:
    finite_ids = np.flatnonzero(np.isfinite(values)).astype(np.int64, copy=False)
    if finite_ids.size == 0:
        return finite_ids
    order = np.lexsort((finite_ids, -values[finite_ids]))
    sorted_ids = finite_ids[order]
    if top_k > 0:
        return sorted_ids[: min(top_k, sorted_ids.size)]
    return sorted_ids


def _logprob_summary(
    logits: np.ndarray,
    token_id: int,
    top_logprobs: int,
) -> tuple[float, tuple[tuple[int, float], ...]]:
    finite_ids = np.flatnonzero(np.isfinite(logits)).astype(np.int64, copy=False)
    if finite_ids.size == 0:
        raise ValueError("no finite logits remain after processing")
    probs = _softmax(logits[finite_ids])
    token_positions = np.flatnonzero(finite_ids == int(token_id))
    if token_positions.size == 0:
        raise ValueError("selected token has no finite logit")
    selected_prob = float(probs[int(token_positions[0])])
    return float(math.log(selected_prob)), _top_logprob_pairs(finite_ids, probs, top_logprobs)


def _top_logprob_pairs(
    token_ids: np.ndarray,
    probs: np.ndarray,
    limit: int,
) -> tuple[tuple[int, float], ...]:
    if limit <= 0 or token_ids.size == 0:
        return ()
    order = np.lexsort((token_ids, -probs))
    pairs: list[tuple[int, float]] = []
    for index in order[: min(int(limit), int(token_ids.size))]:
        probability = float(probs[int(index)])
        if probability <= 0.0 or not math.isfinite(probability):
            continue
        pairs.append((int(token_ids[int(index)]), float(math.log(probability))))
    return tuple(pairs)


def _softmax(values: np.ndarray) -> np.ndarray:
    max_value = float(np.max(values))
    shifted = values - max_value
    exp = np.exp(shifted, dtype=np.float64)
    total = float(exp.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("softmax probabilities are not normalizable")
    return exp / total


def _apply_probability_filters(
    token_ids: np.ndarray,
    probs: np.ndarray,
    *,
    top_p: float,
    min_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((token_ids, -probs))
    sorted_ids = token_ids[order]
    sorted_probs = probs[order]
    if sorted_ids.size == 0:
        return sorted_ids, sorted_probs

    if top_p < 1.0:
        if top_p <= 0.0:
            keep_count = 1
        else:
            keep_count = int(np.searchsorted(np.cumsum(sorted_probs), top_p, side="left")) + 1
            keep_count = max(1, min(keep_count, sorted_ids.size))
        sorted_ids = sorted_ids[:keep_count]
        sorted_probs = sorted_probs[:keep_count]

    if min_p > 0.0:
        threshold = float(probs.max()) * min_p
        mask = sorted_probs >= threshold
        if np.any(mask):
            sorted_ids = sorted_ids[mask]
            sorted_probs = sorted_probs[mask]
        else:
            sorted_ids = sorted_ids[:1]
            sorted_probs = sorted_probs[:1]

    return sorted_ids.astype(np.int64, copy=False), sorted_probs.astype(np.float64, copy=False)


__all__ = [
    "RowSamplingState",
    "ForcedTokenQueue",
    "JsonObjectConstraintState",
    "SampleResult",
    "SamplerPlan",
    "SamplingMode",
    "NATIVE_GPU_SAMPLER_UNSUPPORTED_CAPABILITIES",
    "SPECULATIVE_MTP_INCOMPATIBLE_FIELDS",
    "SPECULATIVE_MTP_INCOMPATIBLE_CONDITIONS",
    "active_processor_names",
    "clone_thinking_budget_state",
    "derive_row_seed",
    "normalize_logit_bias_pairs",
    "normalize_stop_token_sequences",
    "plan_sampler",
    "row_seed_for_index",
    "sampler_fast_path_blockers",
    "select_token",
    "speculative_mtp_sampling_blockers",
    "thinking_budget_active",
    "thinking_budget_state_from_params",
    "supports_native_gpu_sampling",
    "supports_speculative_mtp_sampling",
    "validate_sampling_params",
]
