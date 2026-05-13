"""Torch-free named runtime workspace allocator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from hipengine.core.device import Device
from hipengine.core.dtype import DType, dtype_itemsize
from hipengine.core.hip import HipRuntime
from hipengine.core.memory import DeviceBuffer, free, malloc
from hipengine.core.tensor import Tensor


@dataclass(frozen=True)
class WorkspaceAllocation:
    name: str
    buffer: DeviceBuffer
    tensor: Tensor

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        free(self.buffer, runtime=runtime)


class RuntimeWorkspace:
    """Own named scratch tensors for one engine/model runtime.

    ``reserve_tensor`` reuses an existing allocation when shape/dtype/device are
    identical. If a name is reused for a different tensor spec, the old buffer is
    freed before allocating the replacement.
    """

    def __init__(self, *, device: Device | None = None, runtime: HipRuntime | None = None) -> None:
        self.device = device or Device("hip", 0)
        self.runtime = runtime
        self._allocations: dict[str, WorkspaceAllocation] = {}

    def reserve_tensor(self, name: str, shape: Sequence[int], dtype: str | DType) -> Tensor:
        if not name:
            raise ValueError("workspace tensor name must be non-empty")
        shape_tuple = tuple(int(dim) for dim in shape)
        if any(dim < 0 for dim in shape_tuple):
            raise ValueError("workspace tensor shape dimensions must be non-negative")
        parsed_dtype = DType.parse(dtype)
        nbytes = tensor_nbytes(shape_tuple, parsed_dtype)
        current = self._allocations.get(name)
        if current is not None:
            if (
                current.tensor.shape == shape_tuple
                and current.tensor.dtype == parsed_dtype
                and current.tensor.device == self.device
                and current.buffer.nbytes == nbytes
            ):
                return current.tensor
            current.free(runtime=self.runtime)
        buffer = malloc(nbytes, runtime=self.runtime)
        tensor = Tensor.from_handle(buffer.ptr, shape_tuple, parsed_dtype, self.device)
        self._allocations[name] = WorkspaceAllocation(name=name, buffer=buffer, tensor=tensor)
        return tensor

    def allocation(self, name: str) -> WorkspaceAllocation:
        return self._allocations[name]

    def free(self) -> None:
        for allocation in reversed(tuple(self._allocations.values())):
            allocation.free(runtime=self.runtime)
        self._allocations.clear()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._allocations)


def tensor_nbytes(shape: Sequence[int], dtype: str | DType) -> int:
    count = 1
    for dim in shape:
        dim_int = int(dim)
        if dim_int < 0:
            raise ValueError("tensor shape dimensions must be non-negative")
        count *= dim_int
    return count * dtype_itemsize(dtype)
