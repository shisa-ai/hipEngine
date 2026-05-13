"""Torch-free model loading helpers."""

from hipengine.loading.materialize import (
    DeviceTensorAllocation,
    DeviceWeightMap,
    dtype_from_safetensors,
    load_tensor_info_to_device,
    load_tensor_to_device,
    load_tensors_to_device,
)
from hipengine.loading.qwen35_paro import (
    Qwen35ParoConfig,
    Qwen35ParoLayoutValidation,
    normalize_qwen35_weight_name,
    qwen35_paro_config_from_hf,
    required_moe_c1_tensor_names,
    validate_qwen35_paro_moe_c1_layout,
)
from hipengine.loading.safetensors import (
    MissingConfigError,
    MissingTensorError,
    MissingWeightsError,
    TensorInfo,
    WeightIndex,
    discover_safetensor_shards,
    load_weight_index,
    read_config,
)

__all__ = [
    "MissingConfigError",
    "MissingTensorError",
    "DeviceTensorAllocation",
    "DeviceWeightMap",
    "MissingWeightsError",
    "Qwen35ParoConfig",
    "Qwen35ParoLayoutValidation",
    "TensorInfo",
    "WeightIndex",
    "discover_safetensor_shards",
    "dtype_from_safetensors",
    "normalize_qwen35_weight_name",
    "qwen35_paro_config_from_hf",
    "load_tensor_info_to_device",
    "load_tensor_to_device",
    "load_tensors_to_device",
    "load_weight_index",
    "read_config",
    "required_moe_c1_tensor_names",
    "validate_qwen35_paro_moe_c1_layout",
]
