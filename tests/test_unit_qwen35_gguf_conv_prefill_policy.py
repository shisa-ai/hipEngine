from __future__ import annotations

import pytest

from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen35_linear_attn_conv_prefill_f32,
    qwen35_linear_attn_conv_prefill_f32_tile32x128,
)
from hipengine.runtime import qwen35_gguf_runner as runner_module


def test_lcp1_auto_policy_is_scoped_to_gfx1151() -> None:
    assert runner_module._gguf_linear_attn_conv_prefill_mode("hip_gfx1100") == "baseline"
    assert runner_module._gguf_linear_attn_conv_prefill_mode("hip_gfx1151") == "tile32x128"


def test_lcp1_explicit_candidate_and_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE", "tile32x128")
    assert runner_module._gguf_linear_attn_conv_prefill_mode("hip_gfx1151") == "tile32x128"
    monkeypatch.setenv("HIPENGINE_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE", "baseline")
    assert runner_module._gguf_linear_attn_conv_prefill_mode("hip_gfx1151") == "baseline"


def test_lcp1_rejects_unknown_explicit_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE", "mystery")
    with pytest.raises(ValueError, match="unsupported GGUF linear-attention convolution prefill mode"):
        runner_module._gguf_linear_attn_conv_prefill_mode("hip_gfx1151")


def test_lcp1_runner_resolves_registered_candidate_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = object.__new__(runner_module.Qwen35GGUFFullStackRunner)
    runner.backend = "hip_gfx1151"

    monkeypatch.setenv("HIPENGINE_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE", "tile32x128")
    assert runner._linear_attn_conv_prefill_kernel() is qwen35_linear_attn_conv_prefill_f32_tile32x128
    runner.__dict__.pop("_gguf_linear_attn_conv_prefill_kernel_cache", None)

    monkeypatch.setenv("HIPENGINE_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE", "baseline")
    assert runner._linear_attn_conv_prefill_kernel() is qwen35_linear_attn_conv_prefill_f32
