import os
from types import SimpleNamespace

from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation.qwen4_exp_profiles import (
    QWEN4_EXP_BACKEND, QWEN4_EXP_MODEL, register_qwen4_exp_gfx1151_profiles,
    PRODUCTION_ARITHMETIC_RECOVERY_FLAGS,
    PRODUCTION_Q8_QSA_RESTORED_FLAGS,
)
from scripts.qwen4exp_layer2_profile_gate import CONSERVATIVE_ARITHMETIC_FLAGS
from hipengine.runtime import qwen4_exp_runner
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels


def resolve(quant):
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    return resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant=quant, profile=ExecutionProfile.PRODUCTION)


def test_ud_profile_recovers_arithmetic_but_keeps_exact_owners(monkeypatch):
    # The conservative gate arm also closes the tiled GDN recurrence, which
    # the production plan runs.
    assert PRODUCTION_ARITHMETIC_RECOVERY_FLAGS == tuple(
        flag for flag in CONSERVATIVE_ARITHMETIC_FLAGS
        if flag != "GDN_COLWARPS_PREFILL")
    monkeypatch.setattr(os, "environ", os.environ.copy())
    profile = resolve("gguf_ud_q4_k_xl")
    profile.binder(SimpleNamespace(), profile)
    # GR_WIDE is closed by the recovery pass and reopened by the F16 route
    # composition this quant's production plan carries.
    expected = {**PRODUCTION_Q8_QSA_RESTORED_FLAGS, "GR_WIDE": "1"}
    for flag in PRODUCTION_ARITHMETIC_RECOVERY_FLAGS:
        assert os.environ["HIPENGINE_QWEN4_EXP_" + flag] == expected.get(flag, "0")
    for flag in ("Q4_IU8_EXACT", "Q51_IU8_EXACT", "GDN_REGISTER_PREFILL",
                 "GDN_WAVE_NORM", "FORKB_GROUPED_DOWN"):
        assert os.environ["HIPENGINE_QWEN4_EXP_" + flag] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q4_DP4A64"] == "0"
    variants = {row["selected_variant"] for row in profile.manifest["selections"]}
    assert "qwen4exp_sigmoid_wave_norm_prefill" in variants
    assert not any("mmq" in variant or "dp4a" in variant or
                   "gdn_tiled" in variant or "gdn_columnwarps" in variant
                   for variant in variants)


def test_failed_q4km_profile_binding_uses_strict(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    profile = resolve("gguf_q4_k_m")
    profile.binder(SimpleNamespace(), profile)
    assert profile.fell_back_to_strict
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL"] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN"] == "0"


def test_production_ud_owner_and_quarantined_q4km_binding(monkeypatch):
    """UD keeps its row-split owner; failed Q4_K_M uses strict serial recurrence."""
    from hipengine.kernels.registry import resolve as resolve_kernel

    monkeypatch.setattr(os, "environ", os.environ.copy())
    profile = resolve("gguf_ud_q4_k_xl")
    profile.binder(SimpleNamespace(), profile)

    # The UD profile keeps the colwarps gate open and names the row-split
    # recurrence, the composition's replacement for the multi-column body.
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT"] == (
        "qwen4exp_gdn_rowsplit_prefill")
    assert resolve_kernel(
        backend="hip_gfx1151", layer="gdn_recurrence_norm_gate",
        quant="f32_state", variant="qwen4exp_gdn_rowsplit_prefill",
    ) is not None

    # The recovery profile's manifest still keeps its exact owner.
    recovery_gdn = [
        row["selected_variant"]
        for row in profile.manifest["selections"]
        if row["layer"] == "gdn_recurrence_norm_gate"
    ]
    assert "qwen4exp_gdn_tiled16_multi_prefill" not in recovery_gdn, recovery_gdn

    # The failed Q4_K_M composition is quarantined, so its tiled owner cannot run.
    other = resolve("gguf_q4_k_m")
    other.binder(SimpleNamespace(), other)
    assert other.fell_back_to_strict
    assert os.environ["HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL"] == "0"
    assert not any("tiled16" in row["selected_variant"]
                   for row in other.manifest["selections"])
    # UD declares only its reachable serial scopes, not the disabled tile16
    # scope. The checked owner covers 2-4096 rows; wide remains above4096.
    recovery_scopes = {row["scope"]: row["selected_variant"]
                       for row in profile.manifest["selections"]
                       if row["layer"] == "gdn_recurrence_norm_gate"}
    assert recovery_scopes == {
        "prefill_rows_2_4096_serial_gdn_hk16_hv48_d128": "qwen4exp_sigmoid_wave_norm_narrow_prefill",
        "prefill_rows_gt4096_serial_gdn_hk16_hv48_d128": "qwen4exp_sigmoid_wave_norm_prefill",
    }

    # The superseded route stays registered as the strict fallback, so rollback is
    # a one-line revert rather than a rebuild.
    assert resolve_kernel(
        backend="hip_gfx1151", layer="gdn_recurrence_norm_gate",
        quant="f32_state", variant="qwen4exp_gdn_tiled16_dpp_prefill",
    ) is not None
