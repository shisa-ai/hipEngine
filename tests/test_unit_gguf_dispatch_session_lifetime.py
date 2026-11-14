"""Transient dispatch owners cannot share a recycled integer identity."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear as linear


def _weight(quant, layout):
    return SimpleNamespace(
        spec=SimpleNamespace(layout=layout, quant_key=quant, slot_path="layers.0.ffn_up"),
        backend="hip_gfx1100",
        allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=4096)),
    )


def _stub_launch(monkeypatch, parent):
    monkeypatch.setattr(linear, "id", lambda value: 123, raising=False)
    monkeypatch.setattr(linear, "_DISPATCH_RESOLVE_CACHE", {})
    monkeypatch.setattr(linear, "generation", lambda: 17)
    monkeypatch.setattr(linear, "_ensure_linear_kernel_registered", lambda key: None)
    monkeypatch.setattr(linear, "resolve", lambda **kwargs: SimpleNamespace(key=KernelKey(**kwargs)))
    monkeypatch.setattr(linear, "resolve_gguf_linear_dispatch", lambda *args, **kwargs: parent)
    monkeypatch.setattr(linear, "_raw_k_prefill_rowbatch_dispatch", lambda dispatch, **kwargs: dispatch)
    monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", "0")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "0")
    calls = []
    for abi in ("raw", "t16", "t16_q5_raw_mmq", "raw_k_f32_ordered", "t16_q6_f16_rocblas"):
        monkeypatch.setitem(linear._LAUNCH_ABI, abi,
                            lambda dispatch, *args, **kwargs: calls.append(dispatch.key.variant))
    return calls


def test_q5_raw_mmq_cached_launch_observes_new_source_layout(monkeypatch):
    parent = linear.GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q5_k_t16_v1", "prefill_bf16_bf16_out"), "t16")
    calls = _stub_launch(monkeypatch, parent)
    monkeypatch.setattr(linear, "_q5_raw_mmq_target_eligible", lambda *a, **k: True)
    monkeypatch.setattr(linear, "is_registered", lambda key: True)
    for source_layout in (False, True, False):
        session = linear._Q5RawMMQTargetSession(8192, 1 << 24, None, None, source_layout)
        token = linear._q5_raw_mmq_target_session.set(session)
        try:
            for _ in range(2):
                linear.launch_gguf_linear(_weight("gguf_q5_k_t16_v1", "gguf_q5_k_t16"),
                                          100, 200, 32, 256, 16, libraries={})
        finally:
            linear._q5_raw_mmq_target_session.reset(token)
    assert calls == [
        "mmq32_q8_1_d4s4_f32_bf16_bf16_out",
        "mmq32_q8_1_d4s4_f32_bf16_bf16_out",
        "mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out",
        "mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out",
        "mmq32_q8_1_d4s4_f32_bf16_bf16_out",
        "mmq32_q8_1_d4s4_f32_bf16_bf16_out",
    ]


def test_f32_ordered_cached_launch_observes_new_row_bounds(monkeypatch):
    parent = linear.GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q5_k", "prefill_bf16_bf16_out"), "raw")
    calls = _stub_launch(monkeypatch, parent)
    monkeypatch.setattr(linear, "is_registered", lambda key: True)
    original = linear.backend_package_capability
    def capability(backend, name, default=None):
        if name == "GGUF_F32_ORDERED_PREFILL_QUANTS":
            return {"gguf_q5_k"}
        if name == "GGUF_F32_ORDERED_PREFILL_POLICIES":
            return {"gguf_q5_k": {("bf16", 256, 16): "test_geometry"}}
        return original(backend, name, default)
    monkeypatch.setattr(linear, "backend_package_capability", capability)
    for minimum in (512, 1024, 512):
        session = linear.F32OrderedPrefillSession(
            max_rows=2048, min_rows=minimum, weight_f32_ptr=8192,
            weight_f32_nbytes=1 << 24, library=None)
        with linear.f32_ordered_prefill_session(session):
            for _ in range(2):
                linear.launch_gguf_linear(_weight("gguf_q5_k", "raw_gguf"),
                                          100, 200, 512, 256, 16, libraries={})
    assert calls == ["f32_ordered_test_geometry_bf16_bf16_out"] * 2 + [
        "prefill_bf16_bf16_out"] * 2 + ["f32_ordered_test_geometry_bf16_bf16_out"] * 2


def test_t16_cached_launch_observes_new_row_bounds(monkeypatch):
    quant = "gguf_q6_k_t16_qmicro_planar_v1"
    parent = linear.GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", quant, "t16_wmma_prefill_bf16_bf16_out"), "t16")
    calls = _stub_launch(monkeypatch, parent)
    monkeypatch.setattr(linear, "is_registered", lambda key: True)
    base = linear.T16F16RocblasPrefillSession(
        min_rows=512, max_rows=4096, x_f16_ptr=0x30000000,
        x_f16_nbytes=4096 * 17408 * 2, weight_f16_ptr=0x40000000,
        weight_f16_nbytes=2048 * 17408 * 2,
        tile_out_features_by_shape={(512, 5120, 10240): 2048},
        dequant_library=None, cast_library=None)
    for minimum in (512, 1024, 512):
        session = replace(base, min_rows=minimum)
        with linear.t16_f16_rocblas_prefill_session(session):
            for _ in range(2):
                linear.launch_gguf_linear(_weight(quant, "gguf_q6_k_t16_qmicro_planar"),
                                          100, 200, 512, 5120, 10240,
                                          use_wmma_prefill=True, libraries={})
    assert calls == ["f16_rocblas_t16_qmicro_planar_bf16_bf16_out"] * 2 + [
        "t16_wmma_prefill_bf16_bf16_out"] * 2 + ["f16_rocblas_t16_qmicro_planar_bf16_bf16_out"] * 2


@pytest.mark.parametrize("kind", ["q5", "f32", "t16", "q8"])
def test_cached_identity_does_not_retain_session_and_replace_is_fresh(kind):
    import gc
    import weakref
    if kind == "q5":
        session = linear._Q5RawMMQTargetSession(8192, 1 << 24, None, None, False)
    elif kind == "f32":
        session = linear.F32OrderedPrefillSession(2048, 8192, 1 << 24, None)
    elif kind == "t16":
        session = linear.T16F16RocblasPrefillSession(
            512, 4096, 0x30000000, 4096 * 17408 * 2,
            0x40000000, 2048 * 17408 * 2,
            {(512, 5120, 10240): 2048}, None, None)
    else:
        from tests.test_unit_gguf_q8_mmq_prefill_ordering import _session
        from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import Q8MMQPrefillPolicy
        session = _session(Q8MMQPrefillPolicy({}, 4096, 0.0, 4096))
    cloned = replace(session)
    assert cloned.cache_token is not session.cache_token
    assert cloned == session
    reference = weakref.ref(session)
    cache = {session.cache_token: "dispatch"}
    del session
    gc.collect()
    assert reference() is None
    assert len(cache) == 1

