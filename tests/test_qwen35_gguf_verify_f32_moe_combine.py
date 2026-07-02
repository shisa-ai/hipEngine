from __future__ import annotations

import pytest

from hipengine.runtime import qwen35_gguf_runner as qgr


def test_verify_f32_moe_combine_flag_is_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE", raising=False)

    assert qgr._gguf_verify_f32_moe_combine_enabled() is False
    assert (
        qgr._gguf_f32_moe_combine_out_fn()
        is qgr.weighted_sum_shared_gate_combine_residual_out_f32_f32w
    )
    assert (
        qgr._gguf_f32_moe_combine_batch_out_fn()
        is qgr.weighted_sum_shared_gate_combine_residual_batch_out_f32_f32w
    )


def test_verify_f32_moe_combine_flag_selects_accum_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE", "1")

    assert qgr._gguf_verify_f32_moe_combine_enabled() is True
    assert (
        qgr._gguf_f32_moe_combine_out_fn()
        is qgr.weighted_sum_shared_gate_combine_residual_out_f32_accum_f32w
    )
    assert (
        qgr._gguf_f32_moe_combine_batch_out_fn()
        is qgr.weighted_sum_shared_gate_combine_residual_batch_out_f32_accum_f32w
    )
