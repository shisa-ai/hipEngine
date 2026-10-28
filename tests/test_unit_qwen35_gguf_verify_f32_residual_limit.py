from __future__ import annotations

import pytest

from hipengine.runtime import qwen35_gguf_runner as qgr


@pytest.mark.parametrize("raw,expected", [(None, 40), ("", 40), (" ", 40), ("0", 0), ("1", 1), ("7", 7), ("39", 39), ("40", 40)])
def test_verify_f32_residual_prefix_limit_accepts_integer_boundaries(monkeypatch, raw, expected):
    name = "HIPENGINE_GGUF_VERIFY_F32_RESIDUAL_LAYER_LIMIT"
    if raw is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, raw)
    assert qgr._gguf_verify_f32_residual_layer_limit(40) == expected


@pytest.mark.parametrize("raw", ["-1", "41", "1,4", "1-4", "invalid"])
def test_verify_f32_residual_prefix_limit_rejects_out_of_range_and_list_syntax(monkeypatch, raw):
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_RESIDUAL_LAYER_LIMIT", raw)
    with pytest.raises(ValueError):
        qgr._gguf_verify_f32_residual_layer_limit(40)


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
