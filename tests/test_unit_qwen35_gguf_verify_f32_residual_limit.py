from __future__ import annotations

import pytest

from hipengine.runtime import qwen35_gguf_runner as qgr


def test_verify_f32_residual_layer_limit_defaults_to_all_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_RESIDUAL_LAYER_LIMIT", raising=False)

    assert qgr._gguf_verify_f32_residual_layer_limit(40) == 40


def test_verify_f32_residual_layer_limit_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_RESIDUAL_LAYER_LIMIT", "4")
    assert qgr._gguf_verify_f32_residual_layer_limit(40) == 4

    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_RESIDUAL_LAYER_LIMIT", "41")
    with pytest.raises(ValueError, match="must be within"):
        qgr._gguf_verify_f32_residual_layer_limit(40)
