"""Qwen3.5 GGUF generation bring-up path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hipengine.generation.registry import GenerationRequest, register_text_generator
from hipengine.loading.gguf import GGUFModelInfo
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFOneLayerProbe

_PROBE_PROMPT_TOKENS = {
    # Acceptance fixture prompt. Full GGUF tokenizer support lands with the real
    # generator; this probe exists only to prove public API routing reaches native
    # GGUF resident kernels instead of safetensors/PARO paths.
    "The answer is": 760,
}


@dataclass
class Qwen35GGUFBringupGenerator:
    """Public API GGUF route until the full lm-head sampler lands."""

    model_path: str | Path
    weight_index: GGUFModelInfo
    model_plugin: Any

    def generate(self, request: GenerationRequest) -> list[str]:
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if request.temperature != 0.0 or request.top_p != 1.0:
            raise NotImplementedError("Qwen3.5 GGUF generator currently supports greedy sampling only")
        if request.max_tokens == 0:
            return ["" for _ in request.prompts]
        for prompt in request.prompts:
            token_id = _probe_token_id(prompt)
            with Qwen35GGUFOneLayerProbe(self.model_path, layer_id=0) as probe:
                result = probe.sample_next_token(token_id)
        raise NotImplementedError(
            "Qwen3.5 GGUF public path reached native GGUF lm-head sampling "
            f"(probe token_id={result.token_id}, logit={result.logit:.6g}); "
            "tokenizer detokenization and full layer chain are not wired yet"
        )


def _probe_token_id(prompt: str) -> int:
    try:
        return _PROBE_PROMPT_TOKENS[prompt]
    except KeyError as exc:
        raise NotImplementedError(
            "Qwen3.5 GGUF tokenizer is not wired yet; only the acceptance probe prompt is known"
        ) from exc


def make_qwen35_gguf_bringup_generator(
    *,
    model_path: str | Path,
    weight_index: GGUFModelInfo,
    model_plugin: Any,
) -> Qwen35GGUFBringupGenerator:
    return Qwen35GGUFBringupGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
    )


register_text_generator(
    model="qwen3_5_gguf",
    backend="hip_gfx1100",
    quant="gguf_q4_k_m",
    factory=make_qwen35_gguf_bringup_generator,
)


__all__ = [
    "Qwen35GGUFBringupGenerator",
    "make_qwen35_gguf_bringup_generator",
]
