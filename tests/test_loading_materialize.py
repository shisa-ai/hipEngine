from __future__ import annotations

import ctypes
import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.loading import (
    dtype_from_safetensors,
    float_array_to_bf16_bits,
    load_host_array_to_device,
    load_host_array_to_device_as_dtype,
    load_tensor_to_device,
    load_tensors_to_device,
    load_weight_index,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.next_ptr = 0x1000
        self.buffers: dict[int, bytearray] = {}
        self.freed: list[int] = []
        self.copies: list[tuple[int, int, HipMemcpyKind]] = []

    def malloc(self, nbytes: int) -> int:
        ptr = self.next_ptr
        self.next_ptr += max(nbytes, 1) + 0x100
        self.buffers[ptr] = bytearray(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(ptr)
        self.buffers.pop(ptr, None)

    def memcpy(self, dst: int, src: int, count: int, kind: HipMemcpyKind) -> None:
        assert kind == HipMemcpyKind.HOST_TO_DEVICE
        self.buffers[dst][:count] = ctypes.string_at(src, count)
        self.copies.append((dst, count, kind))


def _write_model(tmp_path, tensors: dict[str, np.ndarray]) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "toy"}), encoding="utf-8")
    save_file(tensors, tmp_path / "model.safetensors")


def test_dtype_from_safetensors_maps_supported_runtime_dtypes() -> None:
    assert dtype_from_safetensors("BF16") is DType.BF16
    assert dtype_from_safetensors("F16") is DType.FP16
    assert dtype_from_safetensors("F32") is DType.FP32
    assert dtype_from_safetensors("I16") is DType.INT16
    assert dtype_from_safetensors("I32") is DType.INT32
    assert dtype_from_safetensors("I64") is DType.INT64
    with pytest.raises(ValueError, match="unsupported safetensors dtype"):
        dtype_from_safetensors("U32")


def test_load_tensor_to_device_copies_exact_contiguous_bytes(tmp_path) -> None:
    weight = np.arange(12, dtype=np.float32).reshape(3, 4)
    _write_model(tmp_path, {"model.weight": weight})
    index = load_weight_index(tmp_path)
    runtime = FakeRuntime()

    allocation = load_tensor_to_device(index, "model.weight", device=Device("hip", 1), runtime=runtime)

    assert allocation.name == "model.weight"
    assert allocation.source.shape == (3, 4)
    assert allocation.buffer.nbytes == weight.nbytes
    assert allocation.tensor.ptr == allocation.buffer.ptr
    assert allocation.tensor.shape == (3, 4)
    assert allocation.tensor.dtype is DType.FP32
    assert allocation.tensor.device == Device("hip", 1)
    assert bytes(runtime.buffers[allocation.buffer.ptr]) == weight.tobytes()
    assert runtime.copies == [(allocation.buffer.ptr, weight.nbytes, HipMemcpyKind.HOST_TO_DEVICE)]

    allocation.free(runtime=runtime)
    assert runtime.freed == [allocation.buffer.ptr]


def test_load_tensors_to_device_frees_partial_allocations_on_failure(tmp_path) -> None:
    _write_model(
        tmp_path,
        {
            "a": np.asarray([1, 2, 3], dtype=np.int32),
            "b": np.asarray([4, 5], dtype=np.int64),
        },
    )
    index = load_weight_index(tmp_path)
    runtime = FakeRuntime()

    with pytest.raises(KeyError, match="missing required tensors"):
        load_tensors_to_device(index, ["a", "missing", "b"], runtime=runtime)

    assert runtime.freed == [0x1000]
    assert runtime.buffers == {}


def test_load_host_array_to_device_copies_prepared_array() -> None:
    array = np.arange(6, dtype=np.int16).reshape(2, 3)
    runtime = FakeRuntime()

    allocation = load_host_array_to_device("prepared", array, runtime=runtime)

    assert allocation.name == "prepared"
    assert allocation.source.name == "prepared"
    assert allocation.source.dtype == "I16"
    assert allocation.tensor.dtype is DType.INT16
    assert allocation.tensor.shape == (2, 3)
    assert bytes(runtime.buffers[allocation.buffer.ptr]) == array.tobytes()


def test_load_host_array_to_device_as_dtype_supports_bf16_bits() -> None:
    values = np.asarray([1.0, -2.5, 0.5], dtype=np.float32)
    bits = float_array_to_bf16_bits(values)
    runtime = FakeRuntime()

    allocation = load_host_array_to_device_as_dtype("bf16_prepared", bits, DType.BF16, runtime=runtime)

    assert bits.dtype == np.uint16
    assert bits.tolist() == [0x3F80, 0xC020, 0x3F00]
    assert allocation.source.dtype == "BF16"
    assert allocation.tensor.dtype is DType.BF16
    assert allocation.tensor.shape == (3,)
    assert bytes(runtime.buffers[allocation.buffer.ptr]) == bits.tobytes()


def test_load_host_array_to_device_as_dtype_rejects_size_mismatch() -> None:
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="does not match dtype"):
        load_host_array_to_device_as_dtype("bad", np.arange(3, dtype=np.uint16), DType.FP32, runtime=runtime)


def test_load_tensors_to_device_returns_tensor_map(tmp_path) -> None:
    a = np.asarray([1, 2, 3], dtype=np.int32)
    b = np.asarray([4, 5], dtype=np.int64)
    _write_model(tmp_path, {"a": a, "b": b})
    index = load_weight_index(tmp_path)
    runtime = FakeRuntime()

    weights = load_tensors_to_device(index, ["a", "b"], runtime=runtime)

    assert weights["a"].dtype is DType.INT32
    assert weights["b"].dtype is DType.INT64
    assert bytes(runtime.buffers[weights.allocation("a").buffer.ptr]) == a.tobytes()
    assert bytes(runtime.buffers[weights.allocation("b").buffer.ptr]) == b.tobytes()
    weights.free(runtime=runtime)
    assert runtime.freed == [weights.allocation("b").buffer.ptr, weights.allocation("a").buffer.ptr]
