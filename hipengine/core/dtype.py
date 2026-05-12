"""Torch-free dtype identifiers used by loaders, tensors, and kernel wrappers."""

from __future__ import annotations

from enum import Enum


class DType(str, Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    INT8 = "int8"
    INT4_PARO = "int4_paro"

    @classmethod
    def parse(cls, value: str | "DType") -> "DType":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            valid = ", ".join(dtype.value for dtype in cls)
            raise ValueError(f"unknown dtype {value!r}; expected one of: {valid}") from exc
