"""Torch-free safetensors-to-device materialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from safetensors import safe_open

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime
from hipengine.core.memory import DeviceBuffer, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.loading.safetensors import TensorInfo, WeightIndex

_SAFETENSORS_DTYPE_TO_DTYPE = {
    "BOOL": DType.BOOL,
    "I8": DType.INT8,
    "I32": DType.INT32,
    "I64": DType.INT64,
    "F16": DType.FP16,
    "BF16": DType.BF16,
    "F32": DType.FP32,
}


@dataclass(frozen=True)
class DeviceTensorAllocation:
    """Owned device allocation plus the tensor view used by kernel wrappers."""

    name: str
    source: TensorInfo
    buffer: DeviceBuffer
    tensor: Tensor

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        free(self.buffer, runtime=runtime)


@dataclass(frozen=True)
class DeviceWeightMap:
    """Collection of materialized device weights.

    The map owns its buffers. Call ``free()`` when the engine/model object is
    destroyed or when a load attempt is abandoned.
    """

    tensors: Mapping[str, DeviceTensorAllocation]

    def __getitem__(self, name: str) -> Tensor:
        return self.tensors[name].tensor

    def allocation(self, name: str) -> DeviceTensorAllocation:
        return self.tensors[name]

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        for allocation in reversed(tuple(self.tensors.values())):
            allocation.free(runtime=runtime)


def dtype_from_safetensors(dtype: str) -> DType:
    try:
        return _SAFETENSORS_DTYPE_TO_DTYPE[dtype]
    except KeyError as exc:
        valid = ", ".join(sorted(_SAFETENSORS_DTYPE_TO_DTYPE))
        raise ValueError(f"unsupported safetensors dtype {dtype!r}; expected one of: {valid}") from exc


def load_tensor_to_device(
    index: WeightIndex,
    name: str,
    *,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
) -> DeviceTensorAllocation:
    """Load one safetensors tensor into HIP/CUDA device memory.

    This is intentionally byte-preserving: quantized packed weights, BF16
    buffers, and scale tensors are copied as contiguous storage without torch or
    dtype conversion. The returned ``Tensor`` is a raw handle used by kernel
    wrappers; the ``DeviceTensorAllocation`` owns the underlying buffer.
    """

    info = index.require((name,))[0]
    return load_tensor_info_to_device(info, device=device, runtime=runtime)


def load_tensor_info_to_device(
    info: TensorInfo,
    *,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
) -> DeviceTensorAllocation:
    dtype = dtype_from_safetensors(info.dtype)
    expected_nbytes = info.nbytes
    if expected_nbytes is None:
        raise ValueError(f"cannot materialize tensor {info.name!r} with unknown dtype {info.dtype!r}")
    array = _read_numpy_tensor(info)
    if tuple(int(dim) for dim in array.shape) != info.shape:
        raise ValueError(f"tensor {info.name!r} shape changed while loading: expected {info.shape}, got {array.shape}")
    if not _is_contiguous(array):
        import numpy as np

        array = np.ascontiguousarray(array)
    if int(array.nbytes) != expected_nbytes:
        raise ValueError(
            f"tensor {info.name!r} byte size mismatch: expected {expected_nbytes}, got {int(array.nbytes)}"
        )
    buffer = malloc(expected_nbytes, runtime=runtime)
    try:
        copy_host_to_device(buffer, host_array_ptr(array), expected_nbytes, runtime=runtime)
    except Exception:
        free(buffer, runtime=runtime)
        raise
    tensor = Tensor.from_handle(buffer.ptr, info.shape, dtype, device or Device("hip", 0))
    return DeviceTensorAllocation(name=info.name, source=info, buffer=buffer, tensor=tensor)


def load_tensors_to_device(
    index: WeightIndex,
    names: Iterable[str],
    *,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
) -> DeviceWeightMap:
    allocations: dict[str, DeviceTensorAllocation] = {}
    try:
        for name in names:
            allocations[name] = load_tensor_to_device(index, name, device=device, runtime=runtime)
    except Exception:
        DeviceWeightMap(allocations).free(runtime=runtime)
        raise
    return DeviceWeightMap(allocations)


def _read_numpy_tensor(info: TensorInfo):
    with safe_open(str(info.shard_path), framework="numpy") as handle:
        return handle.get_tensor(info.name)


def _is_contiguous(array: object) -> bool:
    flags = getattr(array, "flags", None)
    return bool(getattr(flags, "c_contiguous", False))
