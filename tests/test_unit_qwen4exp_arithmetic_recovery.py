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
    assert PRODUCTION_ARITHMETIC_RECOVERY_FLAGS == CONSERVATIVE_ARITHMETIC_FLAGS
    monkeypatch.setattr(os, "environ", os.environ.copy())
    profile = resolve("gguf_ud_q4_k_xl")
    profile.binder(SimpleNamespace(), profile)
    for flag in CONSERVATIVE_ARITHMETIC_FLAGS:
        assert os.environ["HIPENGINE_QWEN4_EXP_" + flag] == (
            PRODUCTION_Q8_QSA_RESTORED_FLAGS.get(flag, "0"))
    for flag in ("Q4_IU8_EXACT", "Q51_IU8_EXACT", "GDN_REGISTER_PREFILL",
                 "GDN_WAVE_NORM", "FORKB_GROUPED_DOWN"):
        assert os.environ["HIPENGINE_QWEN4_EXP_" + flag] == "1"
    variants = {row["selected_variant"] for row in profile.manifest["selections"]}
    assert "qwen4exp_sigmoid_wave_norm_prefill" in variants
    assert not any("mmq" in variant or "dp4a" in variant or
                   "gdn_tiled" in variant or "gdn_columnwarps" in variant
                   for variant in variants)


def test_other_quant_profile_binding_is_unchanged(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    profile = resolve("gguf_q4_k_m")
    profile.binder(SimpleNamespace(), profile)
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN"] == "1"
