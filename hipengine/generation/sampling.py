"""Torch-free sampling utilities for generation paths.

The helpers in this module operate on CPU/NumPy logits for the functional host
sampler path and define the request planning contract shared with future native
GPU samplers.  They intentionally avoid torch.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

import numpy as np

_LOGIT_BIAS_EMPTY: tuple[tuple[int, float], ...] = ()
_UINT64_MASK = (1 << 64) - 1
_SEED_MASK = (1 << 63) - 1
_MAX_NATIVE_GPU_TOP_K = 64
SPECULATIVE_MTP_INCOMPATIBLE_FIELDS: tuple[str, ...] = (
    "temperature",
    "logit_bias",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "stop_token_ids",
    "stop_token_sequences",
    "logprobs",
    "top_logprobs",
)


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
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.prompt_tokens = tuple(int(token) for token in self.prompt_tokens)
        self.generated_tokens = [int(token) for token in self.generated_tokens]
        self.seed = int(self.seed) & _SEED_MASK
        self.request_id = int(self.request_id)
        self.row_index = int(self.row_index)
        self.step_index = int(self.step_index)
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
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
        self.generated_tokens.append(int(token_id))
        self.step_index += 1


@dataclass(frozen=True, slots=True)
class SampleResult:
    """Result of selecting one token from one logits row."""

    token_id: int
    logit: float
    logprob: float | None
    mode: SamplingMode
    candidate_count: int
    top_logprobs: tuple[tuple[int, float], ...] = ()


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
    seed = getattr(params, "seed", None)
    if seed is not None and int(seed) < 0:
        raise ValueError("seed must be non-negative")
    for seed_value in getattr(params, "row_seeds", ()):
        if int(seed_value) < 0:
            raise ValueError("row_seeds must be non-negative")
    top_logprobs = int(getattr(params, "top_logprobs", 0))
    if top_logprobs < 0:
        raise ValueError("top_logprobs must be non-negative")
    for token_id in getattr(params, "stop_token_ids", ()):
        if int(token_id) < 0:
            raise ValueError("stop_token_ids must be non-negative")
    normalize_stop_token_sequences(getattr(params, "stop_token_sequences", None))
    normalize_logit_bias_pairs(getattr(params, "logit_bias", None))


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
    if tuple(int(token) for token in getattr(params, "stop_token_ids", ())):
        names.append("stop_token_ids")
    if normalize_stop_token_sequences(getattr(params, "stop_token_sequences", None)):
        names.append("stop_token_sequences")
    return tuple(names)


def supports_native_gpu_sampling(params: Any) -> bool:
    """Return whether current standalone GPU sampler kernels cover ``params``.

    The native route is intentionally narrower than the host sampler: selected
    logprobs are available, but top-logprobs summaries and combined bounded
    top-k + top-p/min-p filtering are not wired yet.
    """

    validate_sampling_params(params)
    if float(getattr(params, "temperature", 0.0)) <= 0.0:
        return False
    if int(getattr(params, "top_logprobs", 0)) > 0:
        return False
    top_k = int(getattr(params, "top_k", 0))
    if top_k > _MAX_NATIVE_GPU_TOP_K:
        return False
    uses_probability_filter = float(getattr(params, "top_p", 1.0)) < 1.0 or float(getattr(params, "min_p", 0.0)) > 0.0
    if top_k > 0 and uses_probability_filter:
        return False
    return True


def plan_sampler(
    params: Any,
    *,
    native_gpu_available: bool = False,
    native_only: bool = False,
) -> SamplerPlan:
    """Choose the token-selection mode for a request."""

    validate_sampling_params(params)
    processors = active_processor_names(params)
    temperature = float(getattr(params, "temperature", 0.0))
    needs_logits = bool(getattr(params, "logprobs", False)) or int(getattr(params, "top_logprobs", 0)) > 0
    if temperature <= 0.0:
        if processors or needs_logits:
            return SamplerPlan(SamplingMode.PROCESSED_ARGMAX, processors, native_gpu_available)
        return SamplerPlan(SamplingMode.GREEDY_FAST, processors, native_gpu_available)
    native_ready = native_gpu_available and supports_native_gpu_sampling(params)
    if native_ready:
        return SamplerPlan(SamplingMode.GPU_SAMPLE, processors, native_gpu_available)
    if native_only:
        raise NotImplementedError("native GPU sampling is not available for this request")
    return SamplerPlan(SamplingMode.HOST_LOGITS_SAMPLE, processors, native_gpu_available)


def speculative_mtp_sampling_blockers(params: Any) -> tuple[str, ...]:
    """Return request fields that make raw-argmax MTP verification inexact.

    Current MTP proposer/verifier paths produce raw target top-1 decisions.  They
    are exact for normal serving only when the autoregressive request would use
    the same greedy fast path, with no processed logits or sampler metadata.
    """

    plan = plan_sampler(params, native_gpu_available=False)
    if plan.mode is SamplingMode.GREEDY_FAST:
        return ()
    blockers: list[str] = []
    if float(getattr(params, "temperature", 0.0)) > 0.0:
        blockers.append("temperature")
    blockers.extend(plan.active_processors)
    if bool(getattr(params, "logprobs", False)):
        blockers.append("logprobs")
    if int(getattr(params, "top_logprobs", 0)) > 0:
        blockers.append("top_logprobs")
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
    row_state = state if state is not None else RowSamplingState(seed=derive_row_seed(getattr(params, "seed", None), 0))
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

    requested_logprobs = bool(getattr(params, "logprobs", False)) or int(getattr(params, "top_logprobs", 0)) > 0
    requested_top_logprobs = int(getattr(params, "top_logprobs", 0))
    temperature = float(getattr(params, "temperature", 0.0))
    if temperature <= 0.0:
        token_id = _argmax_lower_id(processed)
        logprob, top_logprobs = _logprob_summary(processed, token_id, requested_top_logprobs) if requested_logprobs else (None, ())
        row_state.observe(token_id)
        return SampleResult(
            token_id=token_id,
            logit=float(processed[token_id]),
            logprob=logprob,
            mode=SamplingMode.GREEDY_FAST if not active_processor_names(params) and not requested_logprobs else SamplingMode.PROCESSED_ARGMAX,
            candidate_count=int(np.isfinite(processed).sum()),
            top_logprobs=top_logprobs,
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
    )


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
    "SampleResult",
    "SamplerPlan",
    "SamplingMode",
    "SPECULATIVE_MTP_INCOMPATIBLE_FIELDS",
    "active_processor_names",
    "derive_row_seed",
    "normalize_logit_bias_pairs",
    "normalize_stop_token_sequences",
    "plan_sampler",
    "row_seed_for_index",
    "select_token",
    "speculative_mtp_sampling_blockers",
    "supports_native_gpu_sampling",
    "supports_speculative_mtp_sampling",
    "validate_sampling_params",
]
