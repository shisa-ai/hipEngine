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


@pytest.mark.parametrize("plane", ["2", "3"])
def test_legacy_environment_sets_only_the_named_f32_field(monkeypatch, plane):
    from hipengine.runtime.qwen4_exp_runner import _qwen4_exp_q8_mmq_policy

    policy = Q8MMQPrefillPolicy({(2816, 4096): 64}, 4096, 0.0, 4096)
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES", plane)
    monkeypatch.setattr(linear, "register_gguf_q8_0_mmq_prefill_kernels", lambda: None)
    monkeypatch.setattr(linear, "resolve", lambda **_kwargs: policy)
    assert linear.resolve_q8_mmq_prefill_policy("test").f32_planes == int(plane)
    assert _qwen4_exp_q8_mmq_policy(policy).f32_planes == int(plane)
    assert policy.f32_planes == 3


@pytest.mark.parametrize("planes", [0, 1, 4])
def test_unsupported_f32_plane_counts_fail_loudly(planes):
    with pytest.raises(ValueError, match="f32_planes"):
        Q8MMQPrefillPolicy({}, 4096, 0.0, 4096, f32_planes=planes)
