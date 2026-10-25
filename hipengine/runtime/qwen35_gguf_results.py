"""Host result shared by Qwen GGUF generation and diagnostic probes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Qwen35GGUFNextTokenProbeResult:
    token_id: int
    logit: float
    logits: np.ndarray
