"""Registry for torch-free text generation entry points."""

from __future__ import annotations

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
    kv_storage: str = "auto"
    kv_scale_dtype: str = "fp16"
    kv_scale_granularity: str = "per_token_head"


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
