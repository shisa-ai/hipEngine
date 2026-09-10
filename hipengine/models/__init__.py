"""Model plugins and registry."""

from hipengine.models.base import ModelPlugin
from hipengine.models.kv_capabilities import (
    KVCapabilityEvidence,
    KVCapabilityKey,
    KVCapabilityResolution,
    ModelArtifactIdentity,
    model_artifact_identity,
    resolve_kv_capability,
)
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
from hipengine.models.qwen35_dms import (
    Qwen35DMSDecisionCapability,
    register_qwen35_dms_decision_capability,
    resolve_qwen35_dms_decision_capability,
)
from hipengine.models.qwen4_exp import QWEN4_EXP_GGUF, Qwen4ExpGGUFModel
from hipengine.models.registry import (
    DuplicateModelError,
    MissingModelError,
    register_model,
    registered_models,
    resolve_model,
)
from hipengine.models.toy import TOY_ONE_LAYER, ToyOneLayerModel
from hipengine.models.evie import (
    EVIE,
    EVIE_ARCHITECTURE,
    EVIE_DEFAULT_HEAD,
    EvieModel,
    EvieModelSpec,
    expected_evie_weight_shapes,
    parse_evie_model_spec,
    validate_evie_weight_index,
)
from hipengine.models.timesfm import (
    TIMESFM,
    TIMESFM_ARCHITECTURE,
    TimesFM25Model,
    TimesFMModelSpec,
    expected_timesfm_weight_shapes,
    parse_timesfm_model_spec,
    validate_timesfm_weight_index,
)
from hipengine.models.timesfm3 import (
    PINNED_TIMESFM3_MODEL_ID,
    TIMESFM3,
    TIMESFM3_ARCHITECTURE,
    TimesFM3Model,
    TimesFM3ModelSpec,
    expected_timesfm3_weight_shapes,
    parse_timesfm3_model_spec,
    validate_timesfm3_weight_index,
)

__all__ = [
    "LAGUNA_GGUF",
    "MAPLE",
    "MAPLE_LAYER_PATTERN",
    "MOONSHINE",
    "QWEN35_GGUF",
    "QWEN35_MOE_GGUF",
    "QWEN35_PARO_MOE",
    "QWEN4_EXP_GGUF",
    "TOY_ONE_LAYER",
    "EVIE",
    "EVIE_ARCHITECTURE",
    "EVIE_DEFAULT_HEAD",
    "EvieModel",
    "EvieModelSpec",
    "expected_evie_weight_shapes",
    "parse_evie_model_spec",
    "validate_evie_weight_index",
    "TIMESFM",
    "TIMESFM3",
    "TIMESFM3_ARCHITECTURE",
    "TIMESFM_ARCHITECTURE",
    "DuplicateModelError",
    "KVCapabilityEvidence",
    "KVCapabilityKey",
    "KVCapabilityResolution",
    "LagunaGGUFModel",
    "MapleModel",
    "MapleModelSpec",
    "MissingModelError",
    "ModelArtifactIdentity",
    "ModelPlugin",
    "MoonshineForConditionalGenerationModel",
    "MoonshineModelSpec",
    "Qwen35DMSDecisionCapability",
    "Qwen35GGUFModel",
    "Qwen35MoeGGUFModel",
    "Qwen35ParoMoeModel",
    "Qwen4ExpGGUFModel",
    "ToyOneLayerModel",
    "PINNED_TIMESFM3_MODEL_ID",
    "TimesFM25Model",
    "TimesFM3Model",
    "TimesFM3ModelSpec",
    "TimesFMModelSpec",
    "model_artifact_identity",
    "parse_maple_model_spec",
    "expected_timesfm3_weight_shapes",
    "parse_timesfm3_model_spec",
    "parse_timesfm_model_spec",
    "register_model",
    "register_qwen35_dms_decision_capability",
    "registered_models",
    "resolve_kv_capability",
    "resolve_model",
    "resolve_qwen35_dms_decision_capability",
    "validate_timesfm3_weight_index",
    "validate_timesfm_weight_index",
]
