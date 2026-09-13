from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.loading.laguna_gguf_materialize import (
    _materialize_q4_t16_dual_pair,
    _materialize_spec,
    _spec_for_tensor,
    build_laguna_repacked_cache,
    materialize_laguna_gguf_weights,
    open_laguna_repacked_cache,
)
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_q4_k import (
    interleave_gguf_q4_k_tile16_dual,
    repack_gguf_q4_k_pack8,
    repack_gguf_q4_k_tile16,
)
from hipengine.quant.gguf_t16 import (
    repack_gguf_q6_k_tile16,
    repack_gguf_q6_k_tile16_qmicro,
    repack_gguf_q6_k_tile16_qmicro_planar,
)
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
            "layers.0.attn_v",
            tensor_info("attn_v", (2, 4), GGMLQuantizationType.F16),
            np.arange(8, dtype=np.float16).reshape(2, 4),
            "raw",
            DType.FP16,
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
        assert weight.source_abs_max == (
            float(np.max(np.abs(raw))) if slot_path.endswith(".attn_norm") else None
        )
        assert weight.source_row_l2_max == (
            float(np.max(np.linalg.norm(raw.astype(np.float64), axis=1)))
            if slot_path.endswith(".attn_v")
            else None
        )
        weight.free(runtime=runtime)
        assert runtime.buffers == {}

    tensor = tensor_info(
        "q6_t16_rollback",
        (2, 16, 256),
        GGMLQuantizationType.Q6_K,
    )
    rng = np.random.default_rng(19)
    raw = rng.integers(0, 256, size=(2, 16, 210), dtype=np.uint8)
    expected_legacy = repack_gguf_q6_k_tile16(raw).tiles
    expected_interleaved = repack_gguf_q6_k_tile16_qmicro(raw).tiles
    expected_planar = repack_gguf_q6_k_tile16_qmicro_planar(raw).tiles
    runtime = FakeRuntime()
    weight = _materialize_spec(
        _spec_for_tensor("layers.1.ffn_down_exps", tensor),
        _ArrayReader(tensor.name, raw),
        device=None,
        runtime=runtime,
        backend="hip_gfx1151",
        q6_qmicro=True,
        q6_qmicro_planar=True,
    )
    assert _device_bytes(weight, runtime, "tiles") == expected_planar.tobytes()
    weight.free(runtime=runtime)
    assert runtime.buffers == {}

    runtime = FakeRuntime()
    weight = _materialize_spec(
        _spec_for_tensor("layers.1.ffn_down_exps", tensor),
        _ArrayReader(tensor.name, raw),
        device=None,
        runtime=runtime,
        backend="hip_gfx1151",
        q6_qmicro=True,
        q6_qmicro_planar=False,
    )
    assert (
        _device_bytes(weight, runtime, "tiles")
        == expected_interleaved.tobytes()
    )
    weight.free(runtime=runtime)
    assert runtime.buffers == {}

    for backend, q6_qmicro in (
        ("hip_gfx1151", False),
        ("hip_gfx1100", None),
    ):
        runtime = FakeRuntime()
        weight = _materialize_spec(
            _spec_for_tensor("layers.1.ffn_down_exps", tensor),
            _ArrayReader(tensor.name, raw),
            device=None,
            runtime=runtime,
            backend=backend,
            q6_qmicro=q6_qmicro,
        )
        assert _device_bytes(weight, runtime, "tiles") == expected_legacy.tobytes()
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
            repack_gguf_q6_k_tile16_qmicro_planar,
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


def test_laguna_materialize_q4_expert_pair_replaces_two_owned_tiles() -> None:
    rng = np.random.default_rng(20260731)
    gate_raw = rng.integers(0, 256, size=(2, 16, 144), dtype=np.uint8)
    up_raw = rng.integers(0, 256, size=(2, 16, 144), dtype=np.uint8)
    gate_info = tensor_info(
        "q4_t16_gate_pair",
        (2, 16, 256),
        GGMLQuantizationType.Q4_K,
    )
    up_info = tensor_info(
        "q4_t16_up_pair",
        (2, 16, 256),
        GGMLQuantizationType.Q4_K,
    )
    reader = FakeReader(
        {
            gate_info.name: gate_raw,
            up_info.name: up_raw,
        }
    )
    runtime = FakeRuntime()
    profiles = []

    gate_weight, up_weight = _materialize_q4_t16_dual_pair(
        _spec_for_tensor("layers.1.ffn_gate_exps", gate_info),
        _spec_for_tensor("layers.1.ffn_up_exps", up_info),
        reader,
        device=None,
        runtime=runtime,
        backend="hip_gfx1151",
        profile=profiles.append,
    )
    expected = interleave_gguf_q4_k_tile16_dual(
        repack_gguf_q4_k_tile16(gate_raw).tiles,
        repack_gguf_q4_k_tile16(up_raw).tiles,
    )
    try:
        assert tuple(gate_weight.allocations) == ("tiles_dual",)
        assert tuple(up_weight.allocations) == ()
        assert _device_bytes(
            gate_weight,
            runtime,
            "tiles_dual",
        ) == expected.tobytes()
        assert (
            gate_weight.resident_nbytes + up_weight.resident_nbytes
            == gate_weight.spec.resident_nbytes
            + up_weight.spec.resident_nbytes
        )
        assert [profile.slot_path for profile in profiles] == [
            "layers.1.ffn_gate_exps",
            "layers.1.ffn_up_exps",
        ]
        assert [profile.resident_nbytes for profile in profiles] == [
            expected.nbytes,
            0,
        ]
    finally:
        gate_weight.free(runtime=runtime)
        up_weight.free(runtime=runtime)
    assert runtime.buffers == {}


@pytest.mark.parametrize(
    "slot_path",
    ("layers.0.ffn_gate", "layers.1.ffn_down_shexp"),
)
def test_laguna_materialize_q4_decode_t16_sidecar_is_additive(
    slot_path: str,
) -> None:
    """Decode tiles attach beside pack8 without changing its cache contract."""

    tensor = tensor_info(
        "q4_pack8_sidecar",
        (16, 256),
        GGMLQuantizationType.Q4_K,
    )
    rng = np.random.default_rng(20260730)
    raw = rng.integers(0, 256, size=(16, 144), dtype=np.uint8)
    expected_t16 = repack_gguf_q4_k_tile16(raw[None, ...]).tiles
    runtime = FakeRuntime()
    weight = _materialize_spec(
        _spec_for_tensor(slot_path, tensor),
        _ArrayReader(tensor.name, raw),
        device=None,
        runtime=runtime,
        backend="hip_gfx1151",
        q4_decode_t16_sidecar=True,
    )
    try:
        assert weight.spec.allocation_names == ("qweight", "scales", "mins")
        assert tuple(weight.allocations) == (
            "qweight",
            "scales",
            "mins",
            "decode_tiles",
        )
        assert (
            _device_bytes(weight, runtime, "decode_tiles")
            == expected_t16.tobytes()
        )
        assert weight.resident_nbytes == (
            weight.spec.resident_nbytes + expected_t16.nbytes
        )
    finally:
        weight.free(runtime=runtime)
    assert runtime.buffers == {}


def test_laguna_materialize_pairs_q4_decode_sidecars_for_gfx1151() -> None:
    """A selected gate/up pair receives an exact paired rollback sidecar."""

    gate = np.arange(1_024 * 1_728, dtype=np.uint8).reshape(1_024, 1_728)
    up = np.flip(gate, axis=0).copy()
    reader = FakeReader(
        {
            "blk.1.ffn_gate_shexp.weight": gate,
            "blk.1.ffn_up_shexp.weight": up,
        }
    )
    runtime = FakeRuntime()
    resident = materialize_laguna_gguf_weights(
        reader,
        selected_slots=(
            "layers.1.ffn_gate_shexp",
            "layers.1.ffn_up_shexp",
        ),
        context_length=4_096,
        available_bytes=120 * 2**30,
        runtime=runtime,
        backend="hip_gfx1151",
        q4_decode_t16_sidecar=True,
    )
    try:
        gate_weight = resident.layer(1).weight("ffn_gate_shexp")
        up_weight = resident.layer(1).weight("ffn_up_shexp")
        expected = interleave_gguf_q4_k_tile16_dual(
            repack_gguf_q4_k_tile16(gate[None, ...]).tiles,
            repack_gguf_q4_k_tile16(up[None, ...]).tiles,
        )
        assert tuple(gate_weight.allocations)[-1:] == (
            "decode_tiles_dual",
        )
        assert "decode_tiles" not in gate_weight.allocations
        assert "decode_tiles" not in up_weight.allocations
        assert (
            _device_bytes(gate_weight, runtime, "decode_tiles_dual")
            == expected.tobytes()
        )
    finally:
        resident.free(runtime=runtime)
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
        assert profile.source_kind == "gguf"
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


def test_laguna_repacked_cache_reloads_exact_replacement_payloads(tmp_path) -> None:
    output_norm = np.arange(3_072, dtype=np.float32)
    q4_pack = np.zeros((1_024, 1_728), dtype=np.uint8)
    reader = FakeReader(
        {
            "output_norm.weight": output_norm,
            "blk.1.ffn_gate_shexp.weight": q4_pack,
        }
    )
    selected = ("root.output_norm", "layers.1.ffn_gate_shexp")
    cache_path = tmp_path / "laguna-repacked-v1"

    manifest = build_laguna_repacked_cache(
        reader,
        cache_path,
        selected_slots=selected,
        source_sha256="synthetic-sha256",
    )
    assert manifest["schema"] == 1
    assert manifest["source"]["sha256"] == "synthetic-sha256"
    assert set(manifest["entries"]) == {"layers.1.ffn_gate_shexp"}

    # Only the direct F32 source remains available. Cacheable GGUF payloads must
    # reload from their versioned replacement artifacts rather than source data.
    reader.arrays = {"output_norm.weight": output_norm}
    with pytest.raises(ValueError, match="SHA-256"):
        open_laguna_repacked_cache(cache_path, reader, source_sha256="wrong-sha256")
    open_laguna_repacked_cache(
        cache_path,
        reader,
        source_sha256="synthetic-sha256",
    )
    runtime = FakeRuntime()
    profiles = []
    resident = materialize_laguna_gguf_weights(
        reader,
        selected_slots=selected,
        context_length=4_096,
        available_bytes=120 * 2**30,
        runtime=runtime,
        backend="hip_gfx1151",
        repacked_cache=cache_path,
        repacked_cache_source_sha256="synthetic-sha256",
        q4_decode_t16_sidecar=False,
        profile=profiles.append,
    )
    try:
        assert _device_bytes(resident.root("output_norm"), runtime, "raw") == output_norm.tobytes()
        expected_pack = repack_gguf_q4_k_pack8(q4_pack)
        cached_pack = resident.layer(1).weight("ffn_gate_shexp")
        assert [profile.source_kind for profile in profiles] == ["gguf", "repacked_cache"]
        assert profiles[1].repack_seconds == 0.0
        for name in ("qweight", "scales", "mins"):
            assert (
                _device_bytes(cached_pack, runtime, name) == getattr(expected_pack, name).tobytes()
            )
    finally:
        resident.free(runtime=runtime)
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
