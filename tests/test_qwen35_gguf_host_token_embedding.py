from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.loading.qwen35_gguf_materialize import (
    Qwen35GGUFDeviceWeight,
    Qwen35GGUFMaterializationPlan,
    Qwen35GGUFResidentWeights,
    Qwen35GGUFWeightSpec,
    materialize_qwen35_gguf_weights,
)
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFFullStackRunner,
    Qwen35GGUFResidentSession,
    _q8_0_embedding_rows_to_bf16,
    _resolve_gguf_private_c1_small_weight_arena,
    _resolve_gguf_token_embedding_placement,
)


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


def _token_spec() -> Qwen35GGUFWeightSpec:
    return Qwen35GGUFWeightSpec(
        slot_path="root.token_embedding",
        source=SimpleNamespace(name="token_embd.weight"),
        quant_key="gguf_q8_0",
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


def test_host_embedding_policy_auto_admits_only_private_c1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.runtime.qwen35_gguf_runner as gguf_runner

    monkeypatch.delenv("HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING", raising=False)
    monkeypatch.setattr(
        gguf_runner,
        "backend_package_capability",
        lambda backend, name, default=None: backend == "hip_gfx1151"
        and name == "GGUF_HOST_TOKEN_EMBEDDING_C1",
    )

    assert _resolve_gguf_token_embedding_placement(
        backend="hip_gfx1151",
        max_batch_size=1,
        has_shared_runner=False,
    ) == ("host", "gfx1151_private_c1_auto")
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
    monkeypatch.setattr(materialize, "plan_qwen35_gguf_materialization", lambda model_map, decode_repack=None: plan)

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
        lambda model_map, decode_repack=None: plan,
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
    )

    assert materialize_kwargs[0]["deferred_device_slots"] == ("root.token_embedding",)
    assert materialize_kwargs[0]["use_selective_weight_arena"] is True
    assert runner.host_token_embedding_raw is raw
    assert runner.token_embedding_placement == "host"


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
