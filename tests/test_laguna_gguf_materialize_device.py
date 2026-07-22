from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.loading.laguna_gguf_materialize import (
    _materialize_spec,
    _spec_for_tensor,
    materialize_laguna_gguf_weights,
)
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_pack8, repack_gguf_q4_k_tile16
from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16
from tests._laguna_synthetic import laguna_tensors, make_laguna_info, tensor_info


class FakeRuntime:
    def __init__(self, *, fail_malloc_at: int | None = None) -> None:
        self.next_ptr = 0x1000
        self.buffers: dict[int, bytearray] = {}
        self.freed: list[int] = []
        self.copies: list[tuple[int, int, HipMemcpyKind]] = []
        self.malloc_calls = 0
        self.fail_malloc_at = fail_malloc_at

    def malloc(self, nbytes: int) -> int:
        self.malloc_calls += 1
        if self.fail_malloc_at == self.malloc_calls:
            raise MemoryError("synthetic allocation failure")
        ptr = self.next_ptr
        self.next_ptr += max(int(nbytes), 1) + 0x100
        self.buffers[ptr] = bytearray(int(nbytes))
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(int(ptr))
        self.buffers.pop(int(ptr), None)

    def memcpy(self, dst: int, src: int, count: int, kind: HipMemcpyKind) -> None:
        assert kind == HipMemcpyKind.HOST_TO_DEVICE
        self.buffers[int(dst)][: int(count)] = ctypes.string_at(src, count)
        self.copies.append((int(dst), int(count), kind))


class FakeReader:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.info = make_laguna_info(tensors=laguna_tensors())
        self.arrays = arrays

    def tensor_data(self, name: str) -> np.ndarray:
        return self.arrays[name]


def _device_bytes(weight, runtime: FakeRuntime, name: str) -> bytes:
    return bytes(runtime.buffers[weight.allocation(name).buffer.ptr])


def test_laguna_materialize_spec_copies_dense_and_raw_source_bytes() -> None:
    cases = (
        (
            "layers.0.attn_q",
            tensor_info("f16", (2, 4), GGMLQuantizationType.F16),
            np.arange(8, dtype=np.float16).reshape(2, 4),
            "raw",
            DType.FP16,
        ),
        (
            "layers.0.attn_norm",
            tensor_info("f32", (4,), GGMLQuantizationType.F32),
            np.arange(4, dtype=np.float32),
            "raw",
            DType.FP32,
        ),
        (
            "root.token_embedding",
            tensor_info("q4_raw", (2, 256), GGMLQuantizationType.Q4_K),
            np.arange(2 * 144, dtype=np.uint8).reshape(2, 144),
            "raw",
            DType.INT8,
        ),
    )
    for slot_path, tensor, raw, allocation_name, dtype in cases:
        runtime = FakeRuntime()
        weight = _materialize_spec(
            _spec_for_tensor(slot_path, tensor),
            _ArrayReader(tensor.name, raw),
            device=None,
            runtime=runtime,
            backend="hip_gfx1151",
        )
        assert weight.allocation(allocation_name).tensor.dtype is dtype
        assert _device_bytes(weight, runtime, allocation_name) == raw.tobytes()
        weight.free(runtime=runtime)
        assert runtime.buffers == {}


def test_laguna_materialize_spec_matches_pack8_and_t16_repack_payloads() -> None:
    rng = np.random.default_rng(17)
    cases = (
        (
            "layers.0.ffn_gate",
            tensor_info("q4_pack8", (16, 256), GGMLQuantizationType.Q4_K),
            np.zeros((16, 144), dtype=np.uint8),
            repack_gguf_q4_k_pack8,
            ("qweight", "scales", "mins"),
        ),
        (
            "layers.1.ffn_gate_exps",
            tensor_info("q4_t16", (2, 16, 256), GGMLQuantizationType.Q4_K),
            rng.integers(0, 256, size=(2, 16, 144), dtype=np.uint8),
            repack_gguf_q4_k_tile16,
            ("tiles",),
        ),
        (
            "layers.1.ffn_down_exps",
            tensor_info("q6_t16", (2, 16, 256), GGMLQuantizationType.Q6_K),
            rng.integers(0, 256, size=(2, 16, 210), dtype=np.uint8),
            repack_gguf_q6_k_tile16,
            ("tiles",),
        ),
    )
    for slot_path, tensor, raw, repack, allocation_names in cases:
        runtime = FakeRuntime()
        packed = repack(raw)
        weight = _materialize_spec(
            _spec_for_tensor(slot_path, tensor),
            _ArrayReader(tensor.name, raw),
            device=None,
            runtime=runtime,
            backend="hip_gfx1151",
        )
        assert weight.spec.allocation_names == allocation_names
        for name in allocation_names:
            expected = getattr(packed, name)
            assert _device_bytes(weight, runtime, name) == expected.tobytes()
        weight.free(runtime=runtime)
        assert runtime.buffers == {}


def test_laguna_materialize_profile_separates_repack_allocation_and_upload() -> None:
    tensor = tensor_info("q4_pack8", (16, 256), GGMLQuantizationType.Q4_K)
    raw = np.zeros((16, 144), dtype=np.uint8)
    runtime = FakeRuntime()
    profiles = []

    weight = _materialize_spec(
        _spec_for_tensor("layers.0.ffn_gate", tensor),
        _ArrayReader(tensor.name, raw),
        device=None,
        runtime=runtime,
        backend="hip_gfx1151",
        profile=profiles.append,
    )
    try:
        assert len(profiles) == 1
        profile = profiles[0]
        assert profile.slot_path == "layers.0.ffn_gate"
        assert profile.tensor_name == "q4_pack8"
        assert profile.layout == "q4_k_pack8"
        assert profile.source_nbytes == raw.nbytes
        assert profile.resident_nbytes == weight.resident_nbytes
        assert profile.allocation_count == 3
        assert profile.upload_count == 3
        assert profile.allocated_nbytes == weight.resident_nbytes
        assert profile.uploaded_nbytes == weight.resident_nbytes
        assert profile.source_map_seconds >= 0.0
        assert profile.repack_seconds >= 0.0
        assert profile.allocation_seconds >= 0.0
        assert profile.upload_seconds >= 0.0
        assert profile.total_seconds >= (
            profile.source_map_seconds
            + profile.repack_seconds
            + profile.allocation_seconds
            + profile.upload_seconds
        )
        assert profile.minor_faults >= 0
        assert profile.major_faults >= 0
        assert profile.read_bytes is None or profile.read_bytes >= 0
        assert profile.rss_bytes > 0
        assert profile.max_rss_bytes >= profile.rss_bytes
    finally:
        weight.free(runtime=runtime)


def test_laguna_materialize_spec_frees_when_profile_callback_fails() -> None:
    tensor = tensor_info("f16", (2, 4), GGMLQuantizationType.F16)
    raw = np.arange(8, dtype=np.float16).reshape(2, 4)
    runtime = FakeRuntime()

    def reject_profile(profile) -> None:
        del profile
        raise RuntimeError("synthetic profile sink failure")

    with pytest.raises(RuntimeError, match="synthetic profile sink"):
        _materialize_spec(
            _spec_for_tensor("layers.0.attn_q", tensor),
            _ArrayReader(tensor.name, raw),
            device=None,
            runtime=runtime,
            backend="hip_gfx1151",
            profile=reject_profile,
        )

    assert runtime.freed == [0x1000]
    assert runtime.buffers == {}


def test_laguna_materialize_spec_frees_partial_pack8_allocations_on_failure() -> None:
    tensor = tensor_info("q4_pack8", (16, 256), GGMLQuantizationType.Q4_K)
    raw = np.zeros((16, 144), dtype=np.uint8)
    runtime = FakeRuntime(fail_malloc_at=2)

    with pytest.raises(MemoryError, match="synthetic allocation failure"):
        _materialize_spec(
            _spec_for_tensor("layers.0.ffn_gate", tensor),
            _ArrayReader(tensor.name, raw),
            device=None,
            runtime=runtime,
            backend="hip_gfx1151",
        )

    assert runtime.freed == [0x1000]
    assert runtime.buffers == {}


def test_laguna_public_materializer_selects_slots_and_owns_teardown() -> None:
    output_norm = np.arange(3_072, dtype=np.float32)
    reader = FakeReader({"output_norm.weight": output_norm})
    runtime = FakeRuntime()

    resident = materialize_laguna_gguf_weights(
        reader,
        selected_slots=("root.output_norm",),
        context_length=4_096,
        available_bytes=120 * 2**30,
        runtime=runtime,
        backend="hip_gfx1151",
    )

    assert tuple(resident.root_weights) == ("output_norm",)
    assert resident.root("output_norm").allocation().tensor.dtype is DType.FP32
    assert resident.resident_nbytes == output_norm.nbytes
    assert len(resident.layers) == 48
    assert all(not layer.weights for layer in resident.layers)
    resident.free(runtime=runtime)
    assert runtime.buffers == {}


def test_laguna_public_materializer_rejects_unknown_selection_before_allocation() -> None:
    reader = FakeReader({})
    runtime = FakeRuntime()

    with pytest.raises(ValueError, match="unknown selected Laguna slots"):
        materialize_laguna_gguf_weights(
            reader,
            selected_slots=("layers.99.missing",),
            context_length=4_096,
            available_bytes=120 * 2**30,
            runtime=runtime,
        )

    assert runtime.malloc_calls == 0
    assert runtime.buffers == {}


class _ArrayReader:
    def __init__(self, name: str, array: np.ndarray) -> None:
        self.name = name
        self.array = array

    def tensor_data(self, name: str) -> np.ndarray:
        assert name == self.name
        return self.array
