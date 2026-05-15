"""Torch-free runtime scaffolding."""

from hipengine.runtime.prefill import PrefillConfig
from hipengine.runtime.qwen35_paro import (
    Qwen35ParoAttentionScratch,
    Qwen35ParoDecodeState,
    Qwen35ParoGroupedMoeScratch,
    Qwen35ParoLinearAttentionScratch,
    Qwen35ParoMoeScratch,
)
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoAutoregressiveStepResult,
    Qwen35ParoDecodeGraph,
    Qwen35ParoLayerRecord,
    Qwen35ParoNextTokenResult,
    Qwen35ParoNextTokenRunner,
    Qwen35ParoResidentSession,
)
from hipengine.runtime.workspace import RuntimeWorkspace, WorkspaceAllocation, tensor_nbytes

__all__ = [
    "PrefillConfig",
    "Qwen35ParoAttentionScratch",
    "Qwen35ParoDecodeState",
    "Qwen35ParoGroupedMoeScratch",
    "Qwen35ParoLinearAttentionScratch",
    "Qwen35ParoMoeScratch",
    "Qwen35ParoAutoregressiveStepResult",
    "Qwen35ParoDecodeGraph",
    "Qwen35ParoLayerRecord",
    "Qwen35ParoNextTokenResult",
    "Qwen35ParoNextTokenRunner",
    "Qwen35ParoResidentSession",
    "RuntimeWorkspace",
    "WorkspaceAllocation",
    "tensor_nbytes",
]
