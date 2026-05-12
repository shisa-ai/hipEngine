"""Torch-free tensor handle scaffolding.

The runtime tensor object is a lightweight view around a device pointer plus shape/stride
metadata. Real allocation and dlpack interop land later; this file intentionally avoids torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from hipengine.core.device import Device
from hipengine.core.dtype import DType


@dataclass(frozen=True)
class Tensor:
    ptr: int
    shape: tuple[int, ...]
    dtype: DType
    device: Device
    strides: tuple[int, ...] | None = None

    @classmethod
    def from_handle(
        cls,
        ptr: int,
        shape: Sequence[int],
        dtype: str | DType,
        device: Device,
        strides: Sequence[int] | None = None,
    ) -> "Tensor":
        if ptr < 0:
            raise ValueError("tensor pointer must be non-negative")
        shape_tuple = tuple(int(dim) for dim in shape)
        if any(dim < 0 for dim in shape_tuple):
            raise ValueError("tensor shape dimensions must be non-negative")
        strides_tuple = None if strides is None else tuple(int(stride) for stride in strides)
        return cls(
            ptr=ptr,
            shape=shape_tuple,
            dtype=DType.parse(dtype),
            device=device,
            strides=strides_tuple,
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        out = 1
        for dim in self.shape:
            out *= dim
        return out

    def to_dlpack(self) -> object:
        raise NotImplementedError("dlpack export lands with the Phase-0 tensor runtime")
