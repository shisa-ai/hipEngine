"""Registry for torch-free text generation entry points."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
    stop_token_ids: tuple[int, ...] = ()
    stop_token_sequences: tuple[tuple[int, ...], ...] = ()
    kv_storage: str = "auto"
    kv_scale_dtype: str = "fp16"
    kv_scale_granularity: str = "per_token_head"
    seed: int | None = None
    row_seeds: tuple[int, ...] = ()
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
        object.__setattr__(self, "stop_token_ids", tuple(int(token) for token in self.stop_token_ids))
        object.__setattr__(self, "stop_token_sequences", normalize_stop_token_sequences(self.stop_token_sequences))
        object.__setattr__(self, "kv_storage", str(self.kv_storage))
        object.__setattr__(self, "kv_scale_dtype", str(self.kv_scale_dtype))
        object.__setattr__(self, "kv_scale_granularity", str(self.kv_scale_granularity))
        object.__setattr__(self, "seed", None if self.seed is None else int(self.seed))
        object.__setattr__(self, "row_seeds", tuple(int(seed) for seed in self.row_seeds))
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", str(self.text))
        object.__setattr__(self, "token_logprobs", tuple(self.token_logprobs))
        if self.finish_details is not None:
            object.__setattr__(self, "finish_details", FinishDetails.from_value(self.finish_details))

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
