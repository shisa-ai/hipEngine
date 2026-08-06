"""Torch-free safetensors-to-device materialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from safetensors import safe_open

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime
from hipengine.core.memory import DeviceBuffer, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.loading.safetensors import TensorInfo, WeightIndex, read_tensor_storage_bytes

_SAFETENSORS_DTYPE_TO_DTYPE = {
    "BOOL": DType.BOOL,
    "I8": DType.INT8,
    "I16": DType.INT16,
    "I32": DType.INT32,
    "I64": DType.INT64,
    "F16": DType.FP16,
    "BF16": DType.BF16,
    "F32": DType.FP32,
}
_NUMPY_DTYPE_TO_SAFETENSORS = {
    "bool": "BOOL",
    "int8": "I8",
    "int16": "I16",
    "int32": "I32",
    "int64": "I64",
    "float16": "F16",
    "float32": "F32",
}
_DTYPE_TO_SAFETENSORS = {dtype: safetensors for safetensors, dtype in _SAFETENSORS_DTYPE_TO_DTYPE.items()}


@dataclass(frozen=True)
class DeviceTensorAllocation:
    """Device allocation plus the tensor view used by kernel wrappers.

    Most entries own their ``buffer``.  qweight-neutral Marlin-K materialization
    also stores zero-copy Tensor aliases over an owning allocation; those entries
    set ``owns_buffer=False`` so ``DeviceWeightMap.free()`` cannot double-free.
    """

    name: str
    source: TensorInfo
    buffer: DeviceBuffer
    tensor: Tensor
    owns_buffer: bool = True

    def free(self, *, runtime: HipRuntime | None = None) -> None:
        if self.owns_buffer:
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
    if info.dtype == "BF16":
        import numpy as np

        array = np.frombuffer(read_tensor_storage_bytes(info), dtype=np.uint16).copy().reshape(info.shape)
        return load_host_array_to_device_as_dtype(
            info.name,
            array,
            DType.BF16,
            source_dtype="BF16",
            device=device,
            runtime=runtime,
        )
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


def load_host_array_to_device(
    name: str,
    array: object,
    *,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
) -> DeviceTensorAllocation:
    """Materialize an already prepared contiguous host array to device memory."""

    if not name:
        raise ValueError("tensor name must be non-empty")
    if not _is_contiguous(array):
        import numpy as np

        array = np.ascontiguousarray(array)
    dtype_name = _numpy_dtype_name(array)
    safetensors_dtype = _NUMPY_DTYPE_TO_SAFETENSORS.get(dtype_name)
    if safetensors_dtype is None:
        valid = ", ".join(sorted(_NUMPY_DTYPE_TO_SAFETENSORS))
        raise ValueError(f"unsupported host array dtype {dtype_name!r}; expected one of: {valid}")
    return load_host_array_to_device_as_dtype(
        name,
        array,
        dtype_from_safetensors(safetensors_dtype),
        source_dtype=safetensors_dtype,
        device=device,
        runtime=runtime,
    )


def alias_device_allocation(
    name: str,
    owner: DeviceTensorAllocation,
    shape: tuple[int, ...],
    dtype: str | DType,
    *,
    byte_offset: int = 0,
    device: Device | None = None,
) -> DeviceTensorAllocation:
    """Create a non-owning tensor alias over an existing device allocation."""

    if not name:
        raise ValueError("tensor name must be non-empty")
    if byte_offset < 0:
        raise ValueError("byte_offset must be non-negative")
    parsed = DType.parse(dtype)
    shape = tuple(int(dim) for dim in shape)
    expected_nbytes = parsed.itemsize
    for dim in shape:
        if dim < 0:
            raise ValueError("alias dimensions must be non-negative")
        expected_nbytes *= dim
    if byte_offset + expected_nbytes > owner.buffer.nbytes:
        raise ValueError(
            f"alias {name!r} byte range exceeds owner {owner.name!r}: "
            f"offset={byte_offset} nbytes={expected_nbytes} owner_nbytes={owner.buffer.nbytes}"
        )
    safetensors_dtype = _DTYPE_TO_SAFETENSORS.get(parsed)
    if safetensors_dtype is None:
        raise ValueError(f"dtype {parsed.value!r} cannot be represented as safetensors metadata")
    source = TensorInfo(name=name, shard_path=index_virtual_path(name), dtype=safetensors_dtype, shape=shape)
    tensor = Tensor.from_handle(owner.buffer.ptr + byte_offset, shape, parsed, device or owner.tensor.device)
    return DeviceTensorAllocation(name=name, source=source, buffer=owner.buffer, tensor=tensor, owns_buffer=False)


def load_host_array_to_device_as_dtype(
    name: str,
    array: object,
    dtype: str | DType,
    *,
    source_dtype: str | None = None,
    device: Device | None = None,
    runtime: HipRuntime | None = None,
) -> DeviceTensorAllocation:
    """Materialize a host array while assigning an explicit runtime dtype.

    This is used for BF16 runtime buffers prepared as raw ``uint16`` bit arrays:
    NumPy has no portable builtin BF16 dtype, but the kernel ABI still needs the
    tensor handle to advertise ``DType.BF16``.
    """

    if not name:
        raise ValueError("tensor name must be non-empty")
    if not _is_contiguous(array):
        import numpy as np

        array = np.ascontiguousarray(array)
    parsed = DType.parse(dtype)
    shape = tuple(int(dim) for dim in getattr(array, "shape"))
    nbytes = int(getattr(array, "nbytes"))
    expected_nbytes = parsed.itemsize
    for dim in shape:
        expected_nbytes *= dim
    if nbytes != expected_nbytes:
        raise ValueError(
            f"host array byte size {nbytes} does not match dtype {parsed.value!r} and shape {shape}: "
            f"expected {expected_nbytes}"
        )
    buffer = malloc(nbytes, runtime=runtime)
    try:
        copy_host_to_device(buffer, host_array_ptr(array), nbytes, runtime=runtime)
    except Exception:
        free(buffer, runtime=runtime)
        raise
    safetensors_dtype = source_dtype or _DTYPE_TO_SAFETENSORS.get(parsed)
    if safetensors_dtype is None:
        raise ValueError(f"dtype {parsed.value!r} cannot be represented as safetensors metadata")
    source = TensorInfo(name=name, shard_path=index_virtual_path(name), dtype=safetensors_dtype, shape=shape)
    tensor = Tensor.from_handle(buffer.ptr, shape, parsed, device or Device("hip", 0))
    return DeviceTensorAllocation(name=name, source=source, buffer=buffer, tensor=tensor)


def float_array_to_bf16_bits(array: object):
    """Convert a float-like host array to rounded BF16 bits as ``np.uint16``."""

    import numpy as np

    f32 = np.asarray(array, dtype=np.float32)
    bits = f32.view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return np.ascontiguousarray((rounded >> np.uint32(16)).astype(np.uint16))


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


def index_virtual_path(name: str) -> Path:
    return Path(f"<prepared:{name}>")


def _read_numpy_tensor(info: TensorInfo):
    with safe_open(str(info.shard_path), framework="numpy") as handle:
        return handle.get_tensor(info.name)


def _numpy_dtype_name(array: object) -> str:
    dtype = getattr(array, "dtype", None)
    name = getattr(dtype, "name", None)
    if name is None:
        raise TypeError("host array does not expose dtype.name")
    return str(name)


def _is_contiguous(array: object) -> bool:
    flags = getattr(array, "flags", None)
    return bool(getattr(flags, "c_contiguous", False))
