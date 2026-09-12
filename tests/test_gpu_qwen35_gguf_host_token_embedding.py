from __future__ import annotations

import ctypes
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.policy import QWEN35_DENSE_H5120_GEOMETRY
from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.materialize import DeviceTensorAllocation, float_array_to_bf16_bits
from hipengine.loading.qwen35_gguf_materialize import (
    Qwen35GGUFDeviceWeight,
    Qwen35GGUFMaterializationPlan,
    Qwen35GGUFResidentWeights,
    Qwen35GGUFWeightSpec,
    materialize_qwen35_gguf_weights,
)
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFFullStackRunner,
    Qwen35GGUFResidentSession,
    _gguf_mapped_host_token_embedding_storage,
    _q8_0_embedding_rows_to_bf16,
    _resolve_gguf_private_c1_small_weight_arena,
    _resolve_gguf_private_c1_weight_arena_max_allocation_bytes,
    _resolve_gguf_token_embedding_placement,
)
from tests._gguf_synthetic_weights import make_q4_k_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _q8_row(seed: int, *, blocks: int) -> np.ndarray:
    row = np.empty((blocks, 34), dtype=np.uint8)
    for block in range(blocks):
        scale = np.float16(0.125 * (1 + ((seed + block) % 4)))
        row[block, :2] = np.frombuffer(scale.tobytes(), dtype=np.uint8)
        q = ((np.arange(32, dtype=np.int16) + seed * 7 + block * 3) % 127 - 63).astype(np.int8)
        row[block, 2:] = q.view(np.uint8)
    return row.reshape(blocks * 34)


def test_q8_0_host_token_embedding_matches_reference_bf16() -> None:
    hidden_size = 64
    blocks = hidden_size // 32
    raw = np.stack([_q8_row(seed, blocks=blocks) for seed in range(5)], axis=0)
    token_ids = np.asarray([3, 1, 3, 4], dtype=np.int64)

    actual = _q8_0_embedding_rows_to_bf16(raw, token_ids, hidden_size=hidden_size, cache={})
    reference = float_array_to_bf16_bits(dequantize_gguf_data(raw[token_ids], GGMLQuantizationType.Q8_0))

    assert actual.dtype == np.uint16
    assert actual.shape == (4, hidden_size)
    np.testing.assert_array_equal(actual, reference)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q4_k_mapped_host_embedding_is_bit_exact_to_device_owner() -> None:
    from hipengine.core.hip import HIP_HOST_REGISTER_MAPPED, get_hip_runtime

    runtime = get_hip_runtime()
    hidden_size = 256
    vocab_size = 32
    raw = make_q4_k_weight(vocab_size, hidden_size)
    token_ids = np.asarray([0, 3, 31], dtype=np.int64)
    output = np.empty((len(token_ids), hidden_size), dtype=np.uint16)
    reference = np.empty_like(output)
    spec = Qwen35GGUFWeightSpec(
        slot_path="root.token_embedding",
        source=SimpleNamespace(name="token_embd.weight"),
        quant_key="gguf_q4_k",
        layout="raw_gguf",
        allocation_names=("raw",),
    )
    host_ptr = int(raw.ctypes.data)
    runtime.host_register(host_ptr, raw.nbytes, flags=HIP_HOST_REGISTER_MAPPED)
    buffers: list[DeviceBuffer] = []
    try:
        mapped_ptr = runtime.host_get_device_pointer(host_ptr)
        mapped_allocation = DeviceTensorAllocation(
            name="mapped.raw",
            source=spec.source,
            buffer=DeviceBuffer(mapped_ptr, raw.nbytes),
            tensor=Tensor.from_handle(
                mapped_ptr,
                (raw.nbytes,),
                DType.INT8,
                Device("hip", 0),
            ),
            owns_buffer=False,
        )
        mapped_weight = Qwen35GGUFDeviceWeight(
            spec=spec,
            allocations=MappingProxyType({"raw": mapped_allocation}),
            backend="hip_gfx1100",
        )
        raw_device = malloc(raw.nbytes, runtime=runtime)
        token_device = malloc(token_ids.nbytes, runtime=runtime)
        output_device = malloc(output.nbytes, runtime=runtime)
        reference_device = malloc(reference.nbytes, runtime=runtime)
        buffers.extend(
            (raw_device, token_device, output_device, reference_device)
        )
        copy_host_to_device(raw_device, host_array_ptr(raw), runtime=runtime)
        copy_host_to_device(
            token_device,
            host_array_ptr(token_ids),
            runtime=runtime,
        )
        device_allocation = DeviceTensorAllocation(
            name="device.raw",
            source=spec.source,
            buffer=raw_device,
            tensor=Tensor.from_handle(
                raw_device.ptr,
                (raw.nbytes,),
                DType.INT8,
                Device("hip", 0),
            ),
        )
        device_weight = Qwen35GGUFDeviceWeight(
            spec=spec,
            allocations=MappingProxyType({"raw": device_allocation}),
            backend="hip_gfx1100",
        )
        for weight, out_buffer in (
            (mapped_weight, output_device),
            (device_weight, reference_device),
        ):
            launch_gguf_embedding(
                weight,
                token_device.ptr,
                out_buffer.ptr,
                len(token_ids),
                hidden_size,
                vocab_size,
                runtime=runtime,
            )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(output),
            output_device,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(reference),
            reference_device,
            runtime=runtime,
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        runtime.host_unregister(host_ptr)

    np.testing.assert_array_equal(output, reference)


def test_q8_0_host_token_embedding_uses_supplied_cache() -> None:
    hidden_size = 64
    blocks = hidden_size // 32
    raw = np.stack([_q8_row(seed, blocks=blocks) for seed in range(3)], axis=0)
    cache: dict[int, np.ndarray] = {}

    first = _q8_0_embedding_rows_to_bf16(raw, np.asarray([2, 2], dtype=np.int64), hidden_size=hidden_size, cache=cache)
    cached_row = cache[2]
    second = _q8_0_embedding_rows_to_bf16(raw, np.asarray([2], dtype=np.int64), hidden_size=hidden_size, cache=cache)

    assert cache[2] is cached_row
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[0])


def _token_spec(*, quant_key: str = "gguf_q8_0") -> Qwen35GGUFWeightSpec:
    return Qwen35GGUFWeightSpec(
        slot_path="root.token_embedding",
        source=SimpleNamespace(name="token_embd.weight"),
        quant_key=quant_key,
        layout="raw_gguf",
        allocation_names=("raw",),
    )


def _resident_with_token(weight: Qwen35GGUFDeviceWeight) -> Qwen35GGUFResidentWeights:
    return Qwen35GGUFResidentWeights(
        config=SimpleNamespace(),
        root_weights=MappingProxyType({"token_embedding": weight}),
        layers=(),
        backend="hip_gfx1151",
    )


def test_host_embedding_policy_auto_routes_mapped_and_cpu_types_only_for_private_c1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.runtime.qwen35_gguf_runner as gguf_runner

    monkeypatch.delenv("HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING", raising=False)

    def capability(backend, name, default=None):
        if backend != "hip_gfx1151":
            return default
        if name == "GGUF_HOST_TOKEN_EMBEDDING_C1":
            return True
        if name == "GGUF_HOST_TOKEN_EMBEDDING_C1_GGML_TYPES":
            return ("Q8_0",)
        if name == "GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1":
            return True
        if name == "GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1_GGML_TYPES":
            return ("Q4_K",)
        return default

    monkeypatch.setattr(gguf_runner, "backend_package_capability", capability)

    assert _resolve_gguf_token_embedding_placement(
        backend="hip_gfx1151",
        max_batch_size=1,
        has_shared_runner=False,
        token_embedding_type_name="Q8_0",
    ) == ("host", "gfx1151_private_c1_auto")
    assert _resolve_gguf_token_embedding_placement(
        backend="hip_gfx1151",
        max_batch_size=1,
        has_shared_runner=False,
        token_embedding_type_name="Q4_K",
    ) == ("host", "mapped_host_private_c1_auto")
    assert _resolve_gguf_token_embedding_placement(
        backend="hip_gfx1151",
        max_batch_size=1,
        has_shared_runner=False,
        token_embedding_type_name="Q6_K",
    ) == ("device", "host_type_device_fallback")
    assert _resolve_gguf_token_embedding_placement(
        backend="hip_gfx1151",
        max_batch_size=2,
        has_shared_runner=False,
    ) == ("device", "multi_row_device_fallback")
    assert _resolve_gguf_token_embedding_placement(
        backend="hip_gfx1151",
        max_batch_size=1,
        has_shared_runner=True,
    ) == ("device", "shared_runner_device_fallback")
    assert _resolve_gguf_token_embedding_placement(
        backend="hip_gfx1100",
        max_batch_size=1,
        has_shared_runner=False,
    ) == ("device", "backend_device_fallback")


def test_mapped_host_embedding_storage_copies_backend_qualified_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.runtime.qwen35_gguf_runner as gguf_runner

    raw = np.arange(64, dtype=np.uint8).reshape(4, 16)
    monkeypatch.setattr(
        gguf_runner,
        "backend_package_capability",
        lambda backend, name, default=None: (
            True
            if backend == "hip_gfx1151"
            and name == "GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1_COPY"
            else default
        ),
    )

    storage, kind = _gguf_mapped_host_token_embedding_storage(
        backend="hip_gfx1151",
        raw=raw,
    )

    assert kind == "hip_registered_host_copy"
    assert storage.flags.c_contiguous
    assert not np.shares_memory(storage, raw)
    np.testing.assert_array_equal(storage, raw)
    raw[0, 0] ^= np.uint8(0xFF)
    assert storage[0, 0] != raw[0, 0]


def test_mapped_host_embedding_policy_admits_private_gfx1100_c1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.runtime.qwen35_gguf_runner as gguf_runner

    monkeypatch.delenv("HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING", raising=False)

    def capability(backend, name, default=None):
        if backend != "hip_gfx1100":
            return default
        if name == "GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1":
            return True
        if name == "GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1_GGML_TYPES":
            return ("Q4_K",)
        return default

    monkeypatch.setattr(gguf_runner, "backend_package_capability", capability)

    assert _resolve_gguf_token_embedding_placement(
        backend="hip_gfx1100",
        max_batch_size=1,
        has_shared_runner=False,
        token_embedding_type_name="Q4_K",
    ) == ("host", "mapped_host_private_c1_auto")
    assert _resolve_gguf_token_embedding_placement(
        backend="hip_gfx1100",
        max_batch_size=1,
        has_shared_runner=False,
        token_embedding_type_name="Q6_K",
    ) == ("device", "mapped_host_type_device_fallback")
    assert _resolve_gguf_token_embedding_placement(
        backend="hip_gfx1100",
        max_batch_size=2,
        has_shared_runner=False,
    ) == ("device", "multi_row_device_fallback")
    assert _resolve_gguf_token_embedding_placement(
        backend="hip_gfx1100",
        max_batch_size=1,
        has_shared_runner=True,
    ) == ("device", "shared_runner_device_fallback")


def test_private_c1_small_weight_arena_defaults_on_with_capability_and_private_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.runtime.qwen35_gguf_runner as gguf_runner

    monkeypatch.delenv("HIPENGINE_GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA", raising=False)
    monkeypatch.setattr(
        gguf_runner,
        "backend_package_capability",
        lambda backend, name, default=None: backend == "hip_gfx1151"
        and name == "GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA",
    )

    assert _resolve_gguf_private_c1_small_weight_arena(
        backend="hip_gfx1151",
        max_batch_size=1,
        has_shared_runner=False,
    ) == (True, "private_c1_selective")
    assert _resolve_gguf_private_c1_small_weight_arena(
        backend="hip_gfx1151",
        max_batch_size=1,
        has_shared_runner=False,
        requested=False,
    ) == (False, "disabled")
    assert _resolve_gguf_private_c1_small_weight_arena(
        backend="hip_gfx1151",
        max_batch_size=2,
        has_shared_runner=False,
        requested=True,
    ) == (False, "multi_row_fallback")
    assert _resolve_gguf_private_c1_small_weight_arena(
        backend="hip_gfx1151",
        max_batch_size=1,
        has_shared_runner=True,
        requested=True,
    ) == (False, "shared_runner_fallback")
    assert _resolve_gguf_private_c1_small_weight_arena(
        backend="hip_gfx1100",
        max_batch_size=1,
        has_shared_runner=False,
        requested=True,
    ) == (False, "backend_capability_fallback")


def test_private_c1_small_weight_arena_admits_geometry_scoped_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.runtime.qwen35_gguf_runner as gguf_runner

    policy = {
        (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {
            "enabled": True,
            "max_allocation_bytes": 80 * 1024 * 1024,
        },
    }
    monkeypatch.setattr(
        gguf_runner,
        "backend_package_capability",
        lambda backend, name, default=None: policy
        if name == "GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA_POLICIES"
        else False,
    )

    common = {
        "backend": "hip_gfx1100",
        "geometry": QWEN35_DENSE_H5120_GEOMETRY,
        "file_type_name": "MOSTLY_Q4_K_M",
    }
    assert _resolve_gguf_private_c1_small_weight_arena(
        **common,
        max_batch_size=1,
        has_shared_runner=False,
    ) == (True, "private_c1_selective")
    other = replace(QWEN35_DENSE_H5120_GEOMETRY, head_count=23)
    assert _resolve_gguf_private_c1_small_weight_arena(
        backend="hip_gfx1100",
        geometry=other,
        file_type_name="MOSTLY_Q4_K_M",
        max_batch_size=1,
        has_shared_runner=False,
    ) == (False, "backend_capability_fallback")
    assert _resolve_gguf_private_c1_weight_arena_max_allocation_bytes(
        **common,
    ) == 80 * 1024 * 1024
    assert _resolve_gguf_private_c1_weight_arena_max_allocation_bytes(
        backend="hip_gfx1100",
        geometry=other,
        file_type_name="MOSTLY_Q4_K_M",
    ) == 16 * 1024 * 1024


def test_materializer_defers_token_embedding_without_allocating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.loading.qwen35_gguf_materialize as materialize

    token_spec = _token_spec()
    output_spec = Qwen35GGUFWeightSpec(
        slot_path="root.output_norm",
        source=SimpleNamespace(name="output_norm.weight"),
        quant_key="bf16",
        layout="dense_bf16",
        allocation_names=("raw",),
    )
    plan = Qwen35GGUFMaterializationPlan(
        config=SimpleNamespace(),
        root_specs=MappingProxyType(
            {"token_embedding": token_spec, "output_norm": output_spec}
        ),
        layer_specs=(),
    )
    class FakeReader:
        def __init__(self, path=None):
            self.path = path
            self.info = object()

    allocated: list[str] = []

    monkeypatch.setattr(materialize, "GGUFReader", FakeReader)
    monkeypatch.setattr(
        materialize,
        "build_qwen35_gguf_tensor_map",
        lambda info: SimpleNamespace(info=info, layers=()),
    )
    monkeypatch.setattr(
        materialize,
        "plan_qwen35_gguf_materialization",
        lambda model_map, **_kwargs: plan,
    )

    def fake_materialize(spec, *args, **kwargs):
        del args, kwargs
        allocated.append(spec.slot_path)
        return Qwen35GGUFDeviceWeight(
            spec=spec,
            allocations=MappingProxyType({"raw": object()}),
            backend="hip_gfx1151",
        )

    monkeypatch.setattr(materialize, "_materialize_or_alias", fake_materialize)

    resident = materialize_qwen35_gguf_weights(
        "/tmp/fake.gguf",
        deferred_device_slots=("root.token_embedding",),
        runtime=object(),
        backend="hip_gfx1151",
    )

    assert allocated == ["root.output_norm"]
    assert resident.root("token_embedding").allocations == {}
    assert resident.root("token_embedding").spec is token_spec


def test_selective_weight_arena_owner_denial_falls_back_to_dedicated_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.loading.qwen35_gguf_materialize as materialize

    source = GGUFTensorInfo(
        name="tiny.weight",
        shape=(32,),
        ggml_shape=(32,),
        ggml_type=int(GGMLQuantizationType.F32),
        ggml_type_name="F32",
        n_elements=32,
        nbytes=128,
        offset=0,
        data_offset=0,
        byte_shape=(32,),
    )
    spec = Qwen35GGUFWeightSpec(
        slot_path="root.output_norm",
        source=source,
        quant_key="f32",
        layout="dense_f32",
        allocation_names=("raw",),
    )
    plan = Qwen35GGUFMaterializationPlan(
        config=SimpleNamespace(),
        root_specs=MappingProxyType({"output_norm": spec}),
        layer_specs=(),
    )

    class FakeReader:
        def __init__(self, path=None):
            self.path = path
            self.info = object()

    allocated: list[str] = []
    monkeypatch.setattr(materialize, "GGUFReader", FakeReader)
    monkeypatch.setattr(
        materialize,
        "build_qwen35_gguf_tensor_map",
        lambda info: SimpleNamespace(info=info, layers=()),
    )
    monkeypatch.setattr(
        materialize,
        "plan_qwen35_gguf_materialization",
        lambda model_map, **_kwargs: plan,
    )
    monkeypatch.setattr(
        materialize.DeviceMemoryArena,
        "create",
        lambda *args, **kwargs: (_ for _ in ()).throw(MemoryError("owner denied")),
    )

    def fake_materialize(spec, *args, **kwargs):
        del args
        assert kwargs["allocator"] is None
        allocated.append(spec.slot_path)
        return Qwen35GGUFDeviceWeight(
            spec=spec,
            allocations=MappingProxyType({"raw": object()}),
            backend="hip_gfx1151",
        )

    monkeypatch.setattr(materialize, "_materialize_or_alias", fake_materialize)

    resident = materialize_qwen35_gguf_weights(
        "/tmp/fake.gguf",
        runtime=object(),
        backend="hip_gfx1151",
        use_selective_weight_arena=True,
    )

    assert allocated == ["root.output_norm"]
    assert resident.allocation_mode == "dedicated_selective_arena_denied"
    assert resident.allocation_arena is None
    assert resident.allocation_arena_reason == "owner denied"


def test_full_stack_host_placement_defers_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.runtime.qwen35_gguf_runner as gguf_runner

    raw = np.arange(68, dtype=np.uint8).reshape(1, 68)
    reader = SimpleNamespace(tensor_data=lambda name: raw)
    token_weight = Qwen35GGUFDeviceWeight(
        spec=_token_spec(),
        allocations=MappingProxyType({}),
        backend="hip_gfx1151",
    )
    resident = _resident_with_token(token_weight)
    materialize_kwargs: list[dict[str, object]] = []

    monkeypatch.setattr(gguf_runner, "load_backend_kernel_package", lambda backend: None)
    monkeypatch.setattr(gguf_runner, "resolve", lambda **kwargs: object())
    monkeypatch.setattr(gguf_runner, "GGUFReader", lambda path: reader)

    def fake_materialize(path, **kwargs):
        del path
        materialize_kwargs.append(kwargs)
        return resident

    monkeypatch.setattr(gguf_runner, "materialize_qwen35_gguf_weights", fake_materialize)

    runner = Qwen35GGUFFullStackRunner(
        "/tmp/fake.gguf",
        runtime=object(),
        backend="hip_gfx1151",
        token_embedding_placement="host",
        use_selective_weight_arena=True,
        selective_weight_max_allocation_bytes=64 * 1024 * 1024,
    )

    assert materialize_kwargs[0]["deferred_device_slots"] == ("root.token_embedding",)
    assert materialize_kwargs[0]["use_selective_weight_arena"] is True
    assert materialize_kwargs[0]["selective_weight_max_allocation_bytes"] == 64 * 1024 * 1024
    assert runner.host_token_embedding_raw is raw
    assert runner.token_embedding_placement == "host"


def test_full_stack_maps_q4_host_embedding_and_unregisters_on_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.runtime.qwen35_gguf_runner as gguf_runner

    raw = np.arange(288, dtype=np.uint8).reshape(1, 288)
    reader = SimpleNamespace(tensor_data=lambda name: raw)
    token_weight = Qwen35GGUFDeviceWeight(
        spec=_token_spec(quant_key="gguf_q4_k"),
        allocations=MappingProxyType({}),
        backend="hip_gfx1100",
    )
    resident = Qwen35GGUFResidentWeights(
        config=SimpleNamespace(),
        root_weights=MappingProxyType({"token_embedding": token_weight}),
        layers=(),
        backend="hip_gfx1100",
    )

    class FakeRuntime:
        def __init__(self) -> None:
            self.registered: list[tuple[int, int, int]] = []
            self.unregistered: list[int] = []

        def host_register(self, ptr, nbytes, *, flags=0):
            self.registered.append((int(ptr), int(nbytes), int(flags)))

        def host_get_device_pointer(self, ptr):
            return int(ptr) + 0x1000

        def host_unregister(self, ptr):
            self.unregistered.append(int(ptr))

    runtime = FakeRuntime()
    monkeypatch.setattr(gguf_runner, "load_backend_kernel_package", lambda backend: None)
    monkeypatch.setattr(gguf_runner, "resolve", lambda **kwargs: object())
    monkeypatch.setattr(gguf_runner, "GGUFReader", lambda path: reader)
    monkeypatch.setattr(
        gguf_runner,
        "materialize_qwen35_gguf_weights",
        lambda path, **kwargs: resident,
    )

    def mapped_capability(backend, name, default=None):
        del backend
        if name == "GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1":
            return True
        if name == "GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1_GGML_TYPES":
            return ("Q4_K",)
        return default

    monkeypatch.setattr(
        gguf_runner,
        "backend_package_capability",
        mapped_capability,
    )

    runner = Qwen35GGUFFullStackRunner(
        "/tmp/fake.gguf",
        runtime=runtime,
        backend="hip_gfx1100",
        token_embedding_placement="host",
    )
    mapped = runner.host_token_embedding_mapped_weight

    assert mapped is not None
    assert resident.root("token_embedding").allocations == {}
    assert mapped.spec is token_weight.spec
    assert mapped.allocation("raw").owns_buffer is False
    assert mapped.allocation("raw").buffer.ptr == raw.ctypes.data + 0x1000
    assert runtime.registered == [(raw.ctypes.data, raw.nbytes, 2)]

    runner.close()

    assert runtime.unregistered == [raw.ctypes.data]
    assert runner.host_token_embedding_mapped_weight is None
    assert runner.host_token_embedding_raw is None


def test_full_stack_host_reader_failure_releases_materialized_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.runtime.qwen35_gguf_runner as gguf_runner

    freed: list[str] = []

    class Allocation:
        def free(self, *, runtime=None):
            del runtime
            freed.append("output_norm")

    token_weight = Qwen35GGUFDeviceWeight(
        spec=_token_spec(),
        allocations=MappingProxyType({}),
        backend="hip_gfx1151",
    )
    output_weight = Qwen35GGUFDeviceWeight(
        spec=Qwen35GGUFWeightSpec(
            slot_path="root.output_norm",
            source=SimpleNamespace(name="output_norm.weight"),
            quant_key="bf16",
            layout="dense_bf16",
            allocation_names=("raw",),
        ),
        allocations=MappingProxyType({"raw": Allocation()}),
        backend="hip_gfx1151",
    )
    resident = Qwen35GGUFResidentWeights(
        config=SimpleNamespace(),
        root_weights=MappingProxyType(
            {"token_embedding": token_weight, "output_norm": output_weight}
        ),
        layers=(),
        backend="hip_gfx1151",
    )
    monkeypatch.setattr(gguf_runner, "load_backend_kernel_package", lambda backend: None)
    monkeypatch.setattr(gguf_runner, "resolve", lambda **kwargs: object())
    monkeypatch.setattr(
        gguf_runner,
        "materialize_qwen35_gguf_weights",
        lambda path, **kwargs: resident,
    )
    monkeypatch.setattr(
        gguf_runner,
        "GGUFReader",
        lambda path: (_ for _ in ()).throw(OSError("reader denied")),
    )

    with pytest.raises(OSError, match="reader denied"):
        Qwen35GGUFFullStackRunner(
            "/tmp/fake.gguf",
            runtime=object(),
            backend="hip_gfx1151",
            token_embedding_placement="host",
        )

    assert freed == ["output_norm"]


def _host_session(runner: object) -> Qwen35GGUFResidentSession:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = runner
    session.runtime = object()
    session.host_token_embedding_enabled = True
    session.host_token_embedding_reason = "gfx1151_private_c1_auto"
    session._host_token_embedding_reader = object()
    session._host_token_embedding_raw = np.empty((1, 34), dtype=np.uint8)
    session._host_token_embedding_cache = {0: np.empty((32,), dtype=np.uint16)}
    return session


def test_device_pointer_uses_mapped_host_owner_without_vram_rehydration() -> None:
    mapped_weight = Qwen35GGUFDeviceWeight(
        spec=_token_spec(quant_key="gguf_q4_k"),
        allocations=MappingProxyType({"raw": object()}),
        backend="hip_gfx1100",
    )
    runner = SimpleNamespace(
        weights=_resident_with_token(
            Qwen35GGUFDeviceWeight(
                spec=_token_spec(quant_key="gguf_q4_k"),
                allocations=MappingProxyType({}),
                backend="hip_gfx1100",
            )
        ),
        host_token_embedding_mapped_weight=mapped_weight,
        ensure_device_token_embedding=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("mapped owner must not be rehydrated")
        ),
    )
    session = _host_session(runner)

    actual = session._device_token_embedding_weight(reason="decode_graph")
    again = session._device_token_embedding_weight(reason="packed_ar")

    assert actual is mapped_weight
    assert again is mapped_weight
    assert session.host_token_embedding_enabled is True
    assert session.host_token_embedding_reason == "gfx1151_private_c1_auto"


def test_device_pointer_fallback_rehydrates_once_and_disables_host_copy() -> None:
    device_weight = Qwen35GGUFDeviceWeight(
        spec=_token_spec(),
        allocations=MappingProxyType({"raw": object()}),
        backend="hip_gfx1151",
    )
    calls: list[object] = []
    runner = SimpleNamespace(
        weights=_resident_with_token(
            Qwen35GGUFDeviceWeight(
                spec=_token_spec(),
                allocations=MappingProxyType({}),
                backend="hip_gfx1151",
            )
        )
    )

    def ensure_device_token_embedding(**kwargs):
        calls.append(kwargs)
        runner.weights = _resident_with_token(device_weight)
        return device_weight

    runner.ensure_device_token_embedding = ensure_device_token_embedding
    session = _host_session(runner)

    actual = session._device_token_embedding_weight(reason="decode_graph")
    again = session._device_token_embedding_weight(reason="packed_ar")

    assert actual is device_weight
    assert again is device_weight
    assert len(calls) == 1
    assert session.host_token_embedding_enabled is False
    assert session.host_token_embedding_reason == "device_fallback:decode_graph"
    assert session._host_token_embedding_cache == {}


def test_device_pointer_allocation_denial_keeps_host_route_usable() -> None:
    def deny(**kwargs):
        del kwargs
        raise MemoryError("denied")

    runner = SimpleNamespace(
        weights=_resident_with_token(
            Qwen35GGUFDeviceWeight(
                spec=_token_spec(),
                allocations=MappingProxyType({}),
                backend="hip_gfx1151",
            )
        ),
        ensure_device_token_embedding=deny,
    )
    session = _host_session(runner)

    with pytest.raises(MemoryError, match="denied"):
        session._device_token_embedding_weight(reason="native_mtp_graph")

    assert session.host_token_embedding_enabled is True
    assert session.host_token_embedding_reason == "gfx1151_private_c1_auto"
    assert session._host_token_embedding_raw is not None
    assert 0 in session._host_token_embedding_cache
