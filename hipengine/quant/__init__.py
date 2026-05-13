"""Quant plugins and registry."""

from hipengine.quant.base import QuantPlugin
from hipengine.quant.bf16 import BF16, BF16Quant
from hipengine.quant.fp16 import FP16, FP16Quant
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
    "MissingQuantError",
    "QuantPlugin",
    "W4ParoQuant",
    "W4_PARO",
    "register_quant",
    "registered_quants",
    "resolve_quant",
]
