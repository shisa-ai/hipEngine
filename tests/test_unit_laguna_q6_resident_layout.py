"""Host contracts for Laguna Q6 resident-layout comparisons."""

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.loading import laguna_gguf_materialize as materialize
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_t16 import (
    repack_gguf_q6_k_tile16,
    repack_gguf_q6_k_tile16_qmicro,
    repack_gguf_q6_k_tile16_qmicro_planar,
)
from hipengine.runtime import laguna_gguf_runner as runner
from tests._laguna_synthetic import tensor_info
from tests.test_unit_laguna_gguf_materialize_device import FakeRuntime, _ArrayReader, _device_bytes


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("slot", ["layers.1.ffn_down_exps", "root.lm_head"])
@pytest.mark.parametrize("backend,qmicro,planar,layout", [
    ("hip_gfx1151", None, None, "planar"),
    ("hip_gfx1151", False, None, "legacy"),
    ("hip_gfx1151", True, False, "interleaved"),
    ("hip_gfx1100", None, None, "legacy"),
    ("hip_gfx1100", True, True, "planar"),
])
def test_q6_source_and_cached_payloads_preserve_layout_bytes_and_ownership(tmp_path, cached, slot, backend, qmicro, planar, layout):
    shape = (2, 16, 256) if slot.endswith("ffn_down_exps") else (32, 256)
    tensor = tensor_info("q6", shape, GGMLQuantizationType.Q6_K)
    raw = np.random.default_rng(322).integers(0, 256, size=(*shape[:-1], 210), dtype=np.uint8)
    packed_raw = raw if raw.ndim == 3 else raw[None, ...]
    expected = {
        "legacy": repack_gguf_q6_k_tile16(packed_raw).tiles,
        "interleaved": repack_gguf_q6_k_tile16_qmicro(packed_raw).tiles,
        "planar": repack_gguf_q6_k_tile16_qmicro_planar(packed_raw).tiles,
    }
    spec = materialize._spec_for_tensor(slot, tensor)
    cache = None
    if cached:
        payload = expected["legacy"]
        np.save(tmp_path / "tiles.npy", payload)
        cache = materialize.LagunaGGUFRepackedCache(
            path=tmp_path,
            plan_fingerprint="synthetic-legacy-cache",
            manifest={"entries": {slot: {
                "layout": spec.layout,
                "allocations": {"tiles": {
                    "file": "tiles.npy", "shape": list(payload.shape),
                    "numpy_dtype": payload.dtype.str,
                    "runtime_dtype": "int8", "source_dtype": "I8",
                }},
            }}},
        )
        before = cache.payloads(spec)["tiles"].array.tobytes()
        reader = SimpleNamespace(tensor_data=lambda name: pytest.fail("cache hit read GGUF source"))
    else:
        reader = _ArrayReader(tensor.name, raw)
    runtime = FakeRuntime()
    profiles = []
    weight = materialize._materialize_spec(
        spec, reader, device=None, runtime=runtime, backend=backend,
        q6_qmicro=qmicro, q6_qmicro_planar=planar, repacked_cache=cache,
        profile=profiles.append,
    )
    try:
        effective = layout if slot.endswith("ffn_down_exps") else "legacy"
        assert _device_bytes(weight, runtime, "tiles") == expected[effective].tobytes()
        assert weight.spec is spec and weight.spec.layout == spec.layout
        assert weight.allocation("tiles").tensor.dtype is DType.INT8
        assert weight.resident_nbytes == spec.resident_nbytes == expected[effective].nbytes
        assert runtime.malloc_calls == len(runtime.copies) == 1
        record, = profiles
        assert record.source_kind == ("repacked_cache" if cached else "gguf")
        assert record.layout == spec.layout
        assert record.resident_nbytes == record.uploaded_nbytes == spec.resident_nbytes
        if cache is not None:
            assert cache.plan_fingerprint == "synthetic-legacy-cache"
            assert cache.payloads(spec)["tiles"].array.tobytes() == before
    finally:
        weight.free(runtime=runtime)
    assert runtime.buffers == {}


@pytest.mark.parametrize("failure", ["allocation", "profile"])
def test_q6_cached_conversion_failure_leaves_no_device_allocations(failure):
    tensor = tensor_info("q6", (2, 16, 256), GGMLQuantizationType.Q6_K)
    raw = np.zeros((2, 16, 210), dtype=np.uint8)
    spec = materialize._spec_for_tensor("layers.1.ffn_down_exps", tensor)
    payloads = materialize._resident_payloads(spec, raw)
    cache = SimpleNamespace(has=lambda selected: selected is spec, payloads=lambda selected: payloads)
    runtime = FakeRuntime(fail_malloc_at=1 if failure == "allocation" else None)

    def fail_profile(record):
        raise RuntimeError("synthetic profile failure")

    with pytest.raises((MemoryError, RuntimeError), match="synthetic"):
        materialize._materialize_spec(
            spec, SimpleNamespace(), device=None, runtime=runtime,
            backend="hip_gfx1151", repacked_cache=cache,
            profile=fail_profile if failure == "profile" else None,
        )
    assert runtime.buffers == {}
    assert len(runtime.freed) == (1 if failure == "profile" else 0)


@pytest.mark.parametrize("backend,resident,override,planar_override,expected", [
    ("hip_gfx1151", None, None, None, (True, True)),
    ("hip_gfx1100", None, None, None, (False, False)),
    ("hip_gfx1151", (False, False), None, None, (False, False)),
    ("hip_gfx1151", (True, False), None, None, (True, False)),
    ("hip_gfx1100", (True, True), None, None, (True, True)),
    ("hip_gfx1151", (True, True), False, False, (False, False)),
])
def test_session_layout_selection_adopts_resident_before_backend_default(monkeypatch, backend, resident, override, planar_override, expected):
    def stop_before_libraries(*args, **kwargs):
        raise RuntimeError("layout selection reached kernel planning")

    monkeypatch.setattr(runner, "resolve_laguna_eager_kernel_plan", stop_before_libraries)
    config = SimpleNamespace(context_length=4096)
    weights = None if resident is None else SimpleNamespace(
        config=config, backend=backend, q6_qmicro=resident[0], q6_qmicro_planar=resident[1],
    )
    monkeypatch.setattr(runner, "GGUFReader", lambda path: SimpleNamespace(info=object()))
    monkeypatch.setattr(runner, "laguna_gguf_config_from_metadata", lambda info: config)
    session = object.__new__(runner.LagunaGGUFResidentSession)
    with pytest.raises(RuntimeError, match="layout selection reached kernel planning"):
        session.__init__(
            "/synthetic/laguna.gguf", backend=backend, runtime=object(),
            resident_weights=weights, q6_qmicro=override, q6_qmicro_planar=planar_override,
            q6_wmma_prefetch_activation=False, q6_precomputed_activation_sums=False,
        )
    assert (session.q6_qmicro, session.q6_qmicro_planar) == expected
    assert session.weights is None
    assert session._closed


@pytest.mark.parametrize("resident,session_layout,message", [
    ((True, True), (False, False), "Q6 qmicro layout"),
    ((True, True), (True, False), "Q6 planar layout"),
])
def test_session_rejects_resident_consumer_layout_mismatch_before_tensor_access(resident, session_layout, message):
    session = object.__new__(runner.LagunaGGUFResidentSession)
    session.backend = "hip_gfx1151"
    session.weights = SimpleNamespace(
        config=object(), backend=session.backend,
        q6_qmicro=resident[0], q6_qmicro_planar=resident[1],
    )
    session.q6_qmicro, session.q6_qmicro_planar = session_layout
    with pytest.raises(ValueError, match=message):
        session._validate_resident_weights()
