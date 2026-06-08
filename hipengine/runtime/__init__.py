"""Torch-free runtime scaffolding."""

from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_BF16,
    GGUF_OUTPUT_BF16,
    GGUF_OUTPUT_F32,
    GGUFLinearDispatch,
    launch_gguf_linear,
    resolve_gguf_linear_dispatch,
)
from hipengine.runtime.prefill import PrefillConfig
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFFullStackRunner,
    Qwen35GGUFOneLayerProbe,
    Qwen35GGUFResidentSession,
)
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
    "GGUF_ACTIVATION_BF16",
    "GGUF_OUTPUT_BF16",
    "GGUF_OUTPUT_F32",
    "GGUFLinearDispatch",
    "PrefillConfig",
    "Qwen35GGUFFullStackRunner",
    "Qwen35GGUFOneLayerProbe",
    "Qwen35GGUFResidentSession",
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
    "launch_gguf_linear",
    "resolve_gguf_linear_dispatch",
    "WorkspaceAllocation",
    "tensor_nbytes",
]
