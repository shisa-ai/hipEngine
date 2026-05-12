"""Top-level user API scaffolding.

The real loader / scheduler / engine loop lands later in Phase 0. This module exists now so
``from hipengine import LLM, SamplingParams`` is stable without importing torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SamplingParams:
    """Minimal sampling parameter container for the public API surface."""

    max_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    ignore_eos: bool = False


class LLM:
    """Placeholder public LLM API.

    Full model loading and generation are intentionally not implemented in the scaffold.
    Keeping the class here lets downstream code import the public API while Phase 0 fills in
    the runtime behind it.
    """

    def __init__(self, model: str, *, backend: str = "hip_gfx1100", quant: str = "fp16"):
        self.model = model
        self.backend = backend
        self.quant = quant

    def generate(
        self,
        prompts: str | Iterable[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[str]:
        _ = (prompts, sampling_params)
        raise NotImplementedError("LLM.generate() will land after the Phase-0 engine scaffold")
