"""Torch-free dtype identifiers used by loaders, tensors, and kernel wrappers."""

from __future__ import annotations

from enum import Enum


class DType(str, Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    INT64 = "int64"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    INT8_PER_TOKEN_HEAD = "int8_per_token_head"
    BOOL = "bool"
    INT4_PARO = "int4_paro"

    @property
    def itemsize(self) -> int:
        return dtype_itemsize(self)

    @classmethod
    def parse(cls, value: str | "DType") -> "DType":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            valid = ", ".join(dtype.value for dtype in cls)
            raise ValueError(f"unknown dtype {value!r}; expected one of: {valid}") from exc


_DTYPE_ITEMSIZE = {
    DType.FP16: 2,
    DType.BF16: 2,
    DType.FP32: 4,
    DType.INT64: 8,
    DType.INT32: 4,
    DType.INT16: 2,
    DType.INT8: 1,
    DType.INT8_PER_TOKEN_HEAD: 1,
    DType.BOOL: 1,
}


def dtype_itemsize(dtype: str | DType) -> int:
    parsed = DType.parse(dtype)
    try:
        return _DTYPE_ITEMSIZE[parsed]
    except KeyError as exc:
        raise ValueError(f"dtype {parsed.value!r} does not have a fixed element byte size") from exc
