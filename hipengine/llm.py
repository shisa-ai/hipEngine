"""Top-level user API scaffolding.

The public API stays torch-free. Model-specific generation implementations are resolved
through a registry at call time so backend/quant choices do not become engine branches.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SamplingParams:
    """Minimal sampling parameter container for the public API surface."""

    max_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    ignore_eos: bool = False


class LLM:
    """Minimal public LLM API.

    Phase-0 generation currently resolves to narrow bring-up implementations registered by
    model/backend/quant. Unsupported keys fail explicitly instead of adding engine-level
    backend or quant conditionals.
    """

    def __init__(self, model: str, *, backend: str = "hip_gfx1100", quant: str = "fp16"):
        self.model = model
        self.backend = backend
        self.quant = quant
        self._weight_index: Any | None = None
        self._model_plugin: Any | None = None

    def generate(
        self,
        prompts: str | Iterable[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[str]:
        prompt_tuple = _normalize_prompts(prompts)
        if not prompt_tuple:
            return []
        params = sampling_params or SamplingParams()

        from hipengine.generation import (
            GenerationRequest,
            register_builtin_generators,
            resolve_text_generator,
        )

        register_builtin_generators()
        weight_index, model_plugin = self._load_model_metadata()
        factory = resolve_text_generator(
            model=model_plugin.name,
            backend=self.backend,
            quant=self.quant,
        )
        generator = factory(
            model_path=self.model,
            weight_index=weight_index,
            model_plugin=model_plugin,
        )
        return generator.generate(
            GenerationRequest(
                prompts=prompt_tuple,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                ignore_eos=params.ignore_eos,
            )
        )

    def _load_model_metadata(self) -> tuple[Any, Any]:
        if self._weight_index is not None and self._model_plugin is not None:
            return self._weight_index, self._model_plugin

        from hipengine.loading import load_weight_index
        from hipengine.models import resolve_model

        index = load_weight_index(self.model)
        plugin = resolve_model(_primary_architecture(index.config))
        self._weight_index = index
        self._model_plugin = plugin
        return index, plugin


def _normalize_prompts(prompts: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(prompts, str):
        return (prompts,)
    return tuple(str(prompt) for prompt in prompts)


def _primary_architecture(config: dict[str, Any]) -> str:
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    architectures = config.get("architectures") or text.get("architectures") or ()
    if architectures:
        return str(architectures[0])
    model_type = str(text.get("model_type", config.get("model_type", "")))
    raise ValueError(f"checkpoint config for model_type {model_type!r} does not declare an architecture")
