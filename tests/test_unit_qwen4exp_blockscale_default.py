import os
from types import SimpleNamespace

from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.runtime import qwen4_exp_runner


def test_default_restores_guarded_q8_and_qualified_qsa_paths(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    profile = resolve_runtime_profile(
        model="qwen4_exp_gguf", backend="hip_gfx1151",
        quant="gguf_ud_q4_k_xl", profile=ExecutionProfile.PRODUCTION)
    profile.binder(SimpleNamespace(), profile)
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_DOWN_VARIANT"] == (
        "selected_grouped_blockscale_guarded_prefill_bf16_bf16_out")
    assert os.environ["HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL"] == "page256"
    assert os.environ["HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR"] == "quad"
    for flag in ("Q8_MMQ_PREFILL", "GDN_COLWARPS_PREFILL", "Q4_DP4A64",
                 "QSA_FLASH_PREFILL", "Q8_IU8_WMM", "GR_IU8", "GR_IU8_DOWN"):
        assert os.environ["HIPENGINE_QWEN4_EXP_" + flag] == "0"
    assert os.environ["HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE"] == "1"
    assert os.environ["HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE_V2"] == "1"
    selected = {(r["layer"], r["scope"]): r["selected_variant"]
                for r in profile.manifest["selections"]}
    for scope in ("grouped_prefill_rows_lt512_q8_0_expert_down",
                  "grouped_prefill_rows_ge512_q8_0_expert_down"):
        assert selected[("linear", scope)] == (
            "selected_grouped_blockscale_guarded_prefill_bf16_bf16_out")
    assert selected[("linear", "prefill_rows_ge512_token_major_mapped_q8_down")] == (
        "selected_grouped_row4_register_gemv_bf16_bf16_out")
