from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys

from hipengine.generation.qwen4_exp_profiles import _bind
from scripts.qwen4exp_layer2_profile_gate import CANDIDATES


def test_residual_weight_is_explicit_and_default_off(monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    _bind(SimpleNamespace(), SimpleNamespace(
        manifest_sha256="test", manifest={"quant": "gguf_ud_q4_k_xl"},
    ), production=True)
    assert os.environ["HIPENGINE_QWEN4_EXP_Q8_DOWN_RESIDUAL_WEIGHT"] == "0"
    candidate = CANDIDATES["q8_residual_weight"]
    assert candidate.classification == "T1"
    assert candidate.environment == {"HIPENGINE_QWEN4_EXP_Q8_DOWN_RESIDUAL_WEIGHT": "1"}
    assert candidate.fallback_key[-1] == "selected_gemv_bf16_bf16_out"


def test_residual_variant_and_parent_register_separately():
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import resolve
    register_gfx1151_kernels(replace=True)
    candidate = CANDIDATES["q8_residual_weight"]
    key = candidate.candidate_key
    fn = resolve(backend=key[0], layer=key[1], quant=key[2], variant=key[3])
    assert "residual" in fn.__name__
    parent = resolve(backend=key[0], layer=key[1], quant=key[2],
                     variant="selected_grouped_wmma_prefill_bf16_bf16_out")
    assert parent is not fn


def test_residual_variant_registers_in_a_fresh_process():
    result = subprocess.run([
        sys.executable, "-c",
        "from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels;"
        "from hipengine.kernels.registry import resolve;"
        "register_gfx1151_kernels();"
        "resolve(backend='hip_gfx1151',layer='linear',quant='gguf_q8_0',"
        "variant='selected_grouped_wmma_residual_prefill_bf16_bf16_out')",
    ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
