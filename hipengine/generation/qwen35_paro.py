"""Qwen3.5/PARO text generation bring-up path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hipengine.generation.registry import GenerationRequest, register_text_generator
from hipengine.loading import WeightIndex
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner


@dataclass
class Qwen35ParoOneTokenGenerator:
    """Greedy one-token Qwen3.5/PARO generator used for E2E smoke validation.

    This intentionally stays narrow: it validates public API sequencing over the
    landed layer chain, but it is not a full prefill/decode engine and does not
    claim performance.
    """

    model_path: str | Path
    weight_index: WeightIndex
    model_plugin: Any
    lm_head_chunk: int = 4096
    _runner: Qwen35ParoNextTokenRunner | None = field(default=None, init=False, repr=False)

    def generate(self, request: GenerationRequest) -> list[str]:
        if request.max_tokens != 1:
            raise NotImplementedError("Qwen3.5/PARO smoke generator currently requires max_tokens=1")
        if request.temperature != 0.0 or request.top_p != 1.0:
            raise NotImplementedError("Qwen3.5/PARO smoke generator currently supports greedy sampling only")
        runner = self._get_runner()
        return [
            runner.run_next_token(
                prompt=prompt,
                lm_head_chunk=self.lm_head_chunk,
                resident_layers=True,
                lm_head="gpu_fp16_argmax",
            ).next_token_text
            for prompt in request.prompts
        ]

    def _get_runner(self) -> Qwen35ParoNextTokenRunner:
        if self._runner is None:
            self._runner = Qwen35ParoNextTokenRunner(self.model_path, index=self.weight_index)
        return self._runner


def make_qwen35_paro_one_token_generator(
    *,
    model_path: str | Path,
    weight_index: WeightIndex,
    model_plugin: Any,
) -> Qwen35ParoOneTokenGenerator:
    return Qwen35ParoOneTokenGenerator(
        model_path=model_path,
        weight_index=weight_index,
        model_plugin=model_plugin,
    )


register_text_generator(
    model="qwen3_5_moe_paro",
    backend="hip_gfx1100",
    quant="w4_paro",
    factory=make_qwen35_paro_one_token_generator,
)
