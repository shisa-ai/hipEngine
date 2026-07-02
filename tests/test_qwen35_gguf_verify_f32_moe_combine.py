from __future__ import annotations

from types import SimpleNamespace

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


def test_verify_f32_selected_down_requires_explicit_diagnostic_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    weight = SimpleNamespace(spec=SimpleNamespace(quant_key="gguf_q6_k_x8_v1"))
    scratch = SimpleNamespace(moe_down_out_f32=SimpleNamespace(ptr=1234))

    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_SELECTED_DOWN", raising=False)
    assert qgr._gguf_use_f32_selected_down(weight, scratch, True) is False

    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_SELECTED_DOWN", "1")
    assert qgr._gguf_use_f32_selected_down(weight, scratch, True) is False

    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE", "1")
    assert qgr._gguf_use_f32_selected_down(weight, scratch, True) is True
    assert qgr._gguf_use_f32_selected_down(weight, scratch, False) is False
    assert (
        qgr._gguf_use_f32_selected_down(
            SimpleNamespace(spec=SimpleNamespace(quant_key="gguf_q6_k")),
            scratch,
            True,
        )
        is False
    )


def test_verify_f32_shared_down_requires_selected_down_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = SimpleNamespace(moe_shared_out_f32=SimpleNamespace(ptr=5678))

    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_SELECTED_DOWN", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_SHARED_DOWN", raising=False)
    assert qgr._gguf_use_f32_shared_down(scratch, True, True) is False

    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_SHARED_DOWN", "1")
    assert qgr._gguf_use_f32_shared_down(scratch, True, True) is False

    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_MOE_COMBINE", "1")
    assert qgr._gguf_use_f32_shared_down(scratch, True, True) is False

    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_SELECTED_DOWN", "1")
    assert qgr._gguf_use_f32_shared_down(scratch, True, True) is True
    assert qgr._gguf_use_f32_shared_down(scratch, False, True) is False
    assert qgr._gguf_use_f32_shared_down(scratch, True, False) is False
    assert qgr._gguf_use_f32_shared_down(SimpleNamespace(), True, True) is False
