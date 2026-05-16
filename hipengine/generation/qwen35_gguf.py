"""Qwen3.5 GGUF generation bring-up path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hipengine.generation.registry import GenerationRequest, register_text_generator
from hipengine.loading.gguf import GGUFModelInfo
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFOneLayerProbe
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer


@dataclass
class Qwen35GGUFBringupGenerator:
    """Public API GGUF route until the full lm-head sampler lands."""

    model_path: str | Path
    weight_index: GGUFModelInfo
    model_plugin: Any
    tokenizer: Qwen35GGUFTokenizer = field(init=False)

    def __post_init__(self) -> None:
        self.tokenizer = Qwen35GGUFTokenizer.from_gguf_info(self.weight_index)

    def generate(self, request: GenerationRequest) -> list[str]:
        if request.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if request.temperature != 0.0 or request.top_p != 1.0:
            raise NotImplementedError("Qwen3.5 GGUF generator currently supports greedy sampling only")
        if request.max_tokens == 0:
            return ["" for _ in request.prompts]
        for prompt in request.prompts:
            prompt_ids = self.tokenizer.encode(prompt)
            if not prompt_ids:
                raise ValueError("GGUF prompt tokenization produced no token IDs")
            with Qwen35GGUFOneLayerProbe(self.model_path, layer_id=0) as probe:
                result = probe.sample_next_token(prompt_ids[-1])
        decoded = self.tokenizer.decode([result.token_id])
        raise NotImplementedError(
            "Qwen3.5 GGUF public path reached native GGUF lm-head sampling "
            f"(probe token_id={result.token_id}, text={decoded!r}, logit={result.logit:.6g}); "
            "full layer chain is not wired yet"
        )


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
