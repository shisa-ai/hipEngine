"""Model plugins and registry."""

from hipengine.models.base import ModelPlugin
from hipengine.models.qwen35 import (
    QWEN35_GGUF,
    QWEN35_MOE_GGUF,
    QWEN35_PARO_MOE,
    Qwen35GGUFModel,
    Qwen35MoeGGUFModel,
    Qwen35ParoMoeModel,
)
from hipengine.models.stepfun import (
    STEPFUN_STEP37_GGUF,
    StepFunStep37GGUFModel,
    StepFunUnsupportedCapabilityError,
)
from hipengine.models.registry import (
    DuplicateModelError,
    MissingModelError,
    register_model,
    registered_models,
    resolve_model,
)
from hipengine.models.toy import TOY_ONE_LAYER, ToyOneLayerModel

__all__ = [
    "DuplicateModelError",
    "MissingModelError",
    "ModelPlugin",
    "QWEN35_GGUF",
    "QWEN35_MOE_GGUF",
    "QWEN35_PARO_MOE",
    "Qwen35GGUFModel",
    "Qwen35MoeGGUFModel",
    "Qwen35ParoMoeModel",
    "STEPFUN_STEP37_GGUF",
    "TOY_ONE_LAYER",
    "StepFunStep37GGUFModel",
    "StepFunUnsupportedCapabilityError",
    "ToyOneLayerModel",
    "register_model",
    "registered_models",
    "resolve_model",
]
