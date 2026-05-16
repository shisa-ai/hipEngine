"""Quant plugins and registry."""

from hipengine.quant.base import QuantPlugin
from hipengine.quant.bf16 import BF16, BF16Quant
from hipengine.quant.fp16 import FP16, FP16Quant
from hipengine.quant.gguf import (
    GGMLQuantizationType,
    GGUFQuantLayout,
    GGUFValueType,
    GGUF_QUANT_LAYOUTS,
    QK_K,
    bf16_to_float32,
    dequantization_supported,
    dequantize_gguf_data,
    ggml_type,
    ggml_type_name,
    llama_file_type_name,
    nbytes_for_shape,
    quant_layout,
    quant_shape_from_byte_shape,
    quant_shape_to_byte_shape,
)
from hipengine.quant.w4_paro import W4_PARO, W4ParoQuant
from hipengine.quant.registry import (
    DuplicateQuantError,
    MissingQuantError,
    register_quant,
    registered_quants,
    resolve_quant,
)

__all__ = [
    "BF16",
    "BF16Quant",
    "DuplicateQuantError",
    "FP16",
    "FP16Quant",
    "GGMLQuantizationType",
    "GGUFQuantLayout",
    "GGUFValueType",
    "GGUF_QUANT_LAYOUTS",
    "MissingQuantError",
    "QuantPlugin",
    "QK_K",
    "W4ParoQuant",
    "W4_PARO",
    "bf16_to_float32",
    "dequantization_supported",
    "dequantize_gguf_data",
    "ggml_type",
    "ggml_type_name",
    "llama_file_type_name",
    "nbytes_for_shape",
    "quant_layout",
    "quant_shape_from_byte_shape",
    "quant_shape_to_byte_shape",
    "register_quant",
    "registered_quants",
    "resolve_quant",
]
