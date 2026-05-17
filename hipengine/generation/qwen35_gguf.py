"""Qwen3.5 GGUF generation path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hipengine.generation.registry import GenerationRequest, register_text_generator
from hipengine.loading.gguf import GGUFModelInfo
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer


@dataclass
class Qwen35GGUFBringupGenerator:
    """Public API GGUF greedy generator over a persistent resident session."""

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
        outputs: list[str] = []
        with Qwen35GGUFResidentSession(self.model_path) as session:
            for prompt in request.prompts:
                prompt_ids = self.tokenizer.encode(prompt)
                if not prompt_ids:
                    raise ValueError("GGUF prompt tokenization produced no token IDs")
                generated_ids: list[int] = []
                result = session.prefill(prompt_ids)
                for index in range(request.max_tokens):
                    generated_ids.append(result.token_id)
                    if not request.ignore_eos and result.token_id == self.tokenizer.eos_token_id:
                        break
                    if index + 1 >= request.max_tokens:
                        break
                    result = session.step(result.token_id)
                outputs.append(self.tokenizer.decode(generated_ids))
        return outputs


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
