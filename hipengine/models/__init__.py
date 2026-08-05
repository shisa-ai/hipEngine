"""Model plugins and registry."""

from hipengine.models.base import ModelPlugin
from hipengine.models.laguna import LAGUNA_GGUF, LagunaGGUFModel
from hipengine.models.maple import (
    MAPLE,
    MAPLE_LAYER_PATTERN,
    MapleModel,
    MapleModelSpec,
    parse_maple_model_spec,
)
from hipengine.models.moonshine import (
    MOONSHINE,
    MoonshineForConditionalGenerationModel,
    MoonshineModelSpec,
)
from hipengine.models.qwen35 import (
    QWEN35_GGUF,
    QWEN35_MOE_GGUF,
    QWEN35_PARO_MOE,
    Qwen35GGUFModel,
    Qwen35MoeGGUFModel,
    Qwen35ParoMoeModel,
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
    "LAGUNA_GGUF",
    "MAPLE",
    "MAPLE_LAYER_PATTERN",
    "MOONSHINE",
    "QWEN35_GGUF",
    "QWEN35_MOE_GGUF",
    "QWEN35_PARO_MOE",
    "TOY_ONE_LAYER",
    "DuplicateModelError",
    "LagunaGGUFModel",
    "MapleModel",
    "MapleModelSpec",
    "MissingModelError",
    "ModelPlugin",
    "MoonshineForConditionalGenerationModel",
    "MoonshineModelSpec",
    "Qwen35GGUFModel",
    "Qwen35MoeGGUFModel",
    "Qwen35ParoMoeModel",
    "ToyOneLayerModel",
    "parse_maple_model_spec",
    "register_model",
    "registered_models",
    "resolve_model",
]
