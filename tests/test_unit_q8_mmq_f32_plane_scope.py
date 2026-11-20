"""Q8 MMQ reduced-plane selection is explicitly F32-output-only."""

from __future__ import annotations

from dataclasses import replace

import pytest

from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import Q8MMQPrefillPolicy
from hipengine.kernels.registry import KernelKey
from hipengine.runtime import gguf_linear as linear
from tests.test_unit_gguf_q8_mmq_prefill_ordering import _session


@pytest.mark.parametrize("f32_planes", [2, 3])
@pytest.mark.parametrize("variant", ["prefill_bf16_bf16_out", "prefill_f32_f32_out", "gemv_f32_f32_out"])
def test_f32_plane_field_leaves_bf16_chain_on_three_planes(f32_planes, variant):
    policy = Q8MMQPrefillPolicy(
        min_rows={(2816, 4096): 64}, max_rows=4096,
        risk_threshold=0.0, max_out_features=4096, f32_planes=f32_planes,
    )
    parent = linear.GGUFLinearDispatch(KernelKey("hip_gfx1100", "linear", "gguf_q8_0", variant), "raw")
    token = linear._q8_mmq_prefill_session.set(_session(policy))
    try:
        selected = linear._q8_mmq_prefill_dispatch(parent, rows=512, in_features=2816, out_features=4096)
    finally:
        linear._q8_mmq_prefill_session.reset(token)
    if variant == "prefill_bf16_bf16_out":
        assert selected.key.variant == "mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out"
        assert selected.abi == "raw_mmq_d4x3"
    else:
        assert selected.key.variant == (
            "mmq128_raw_vec4_q8_1_d4x2_guarded_f32_f32_out" if f32_planes == 2
            else "mmq128_prefill_q8_1_d4x3_guarded_f32_f32_out")
        assert selected.abi == f"raw_mmq_d4x{f32_planes}_f32"
    assert replace(policy, f32_planes=3).f32_planes == 3


@pytest.mark.parametrize("plane", ["2", "3", "invalid"])
def test_retired_environment_is_ignored_and_explicit_f32_field_wins(monkeypatch, plane):
    from hipengine.runtime.qwen4_exp_runner import _qwen4_exp_q8_mmq_policy

    policy = Q8MMQPrefillPolicy({(2816, 4096): 64}, 4096, 0.0, 4096)
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES", plane)
    monkeypatch.setattr(linear, "register_gguf_q8_0_mmq_prefill_kernels", lambda: None)
    monkeypatch.setattr(linear, "resolve", lambda **_kwargs: policy)
    assert linear.resolve_q8_mmq_prefill_policy("test") is policy
    assert _qwen4_exp_q8_mmq_policy(policy) is policy
    assert _qwen4_exp_q8_mmq_policy(policy, f32_planes=2).f32_planes == 2
    assert policy.f32_planes == 3


def test_cached_launch_does_not_reuse_a_dead_session_identity(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("HIPENGINE_GGUF_WMMA_PREFILL", "0")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE", "0")
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_STAGED_F32_LAYERS", "")
    monkeypatch.setattr(linear, "_DISPATCH_RESOLVE_CACHE", {})
    # Model allocator ID reuse deterministically, without relying on GC timing.
    monkeypatch.setattr(linear, "id", lambda value: 123, raising=False)
    monkeypatch.setattr(linear, "generation", lambda: 17)
    monkeypatch.setattr(linear, "_ensure_linear_kernel_registered", lambda key: None)
    monkeypatch.setattr(linear, "resolve", lambda **kwargs: object())
    calls = []
    for count in (2, 3):
        monkeypatch.setitem(linear._LAUNCH_ABI, f"raw_mmq_d4x{count}_f32",
                            lambda *args, _count=count: calls.append(_count))
    weight = SimpleNamespace(
        spec=SimpleNamespace(layout="raw_gguf", quant_key="gguf_q8_0",
                             slot_path="layers.0.attn_qkv.weight"),
        backend="hip_gfx1100",
        allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=4096)),
    )
    for planes in (2, 3, 2):
        policy = Q8MMQPrefillPolicy({(2816, 4096): 64}, 4096, 0.0, 4096,
                                   f32_planes=planes)
        token = linear._q8_mmq_prefill_session.set(_session(policy))
        try:
            for _ in range(2):
                linear.launch_gguf_linear(
                    weight, 100, 200, 512, 2816, 4096,
                    activation_dtype=linear.GGUF_ACTIVATION_F32,
                    output_dtype=linear.GGUF_OUTPUT_F32, libraries={})
        finally:
            linear._q8_mmq_prefill_session.reset(token)
    assert calls == [2, 2, 3, 3, 2, 2]


@pytest.mark.parametrize("planes", [0, 1, 4])
def test_unsupported_f32_plane_counts_fail_loudly(planes):
    with pytest.raises(ValueError, match="f32_planes"):
        Q8MMQPrefillPolicy({}, 4096, 0.0, 4096, f32_planes=planes)
