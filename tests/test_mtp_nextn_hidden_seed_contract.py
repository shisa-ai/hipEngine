from __future__ import annotations

import numpy as np

from hipengine.kernels.hip_gfx1100.speculative import mtp_nextn
from hipengine.quant.gguf import GGMLQuantizationType


def test_nextn_layer_can_return_llamacpp_h_nextn_seed(monkeypatch) -> None:
    """B2 needs llama.cpp's post-shared-head-norm h_nextn as depth+1 seed."""

    projected = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    attended = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    post_ffn_hidden = np.array([[7.0, 8.0, 9.0]], dtype=np.float32)
    h_nextn = np.array([[0.7, 0.8, 0.9]], dtype=np.float32)
    logits = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    monkeypatch.setattr(
        mtp_nextn,
        "qwen35_gguf_mtp_eh_proj_f32",
        lambda *args, **kwargs: projected,
    )
    monkeypatch.setattr(
        mtp_nextn,
        "qwen35_gguf_mtp_attention_sublayer_f32",
        lambda hidden, *args, **kwargs: attended,
    )
    monkeypatch.setattr(
        mtp_nextn,
        "qwen35_gguf_mtp_ffn_sublayer_f32",
        lambda hidden, *args, **kwargs: post_ffn_hidden,
    )

    def fake_shared_head(hidden, *args, **kwargs):
        np.testing.assert_array_equal(hidden, post_ffn_hidden)
        assert kwargs["return_normed_hidden"] is True
        return logits, h_nextn

    monkeypatch.setattr(mtp_nextn, "qwen35_gguf_mtp_shared_head_logits_f32", fake_shared_head)

    result_logits, result_hidden = mtp_nextn.qwen35_gguf_mtp_nextn_layer_logits_f32(
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        np.ones(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        np.ones(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F32,
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
        GGMLQuantizationType.F32,
        np.ones(3, dtype=np.float32),
        np.zeros((4, 3), dtype=np.float32),
        num_heads=1,
        num_kv_heads=1,
        experts_used=1,
        return_hidden_seed=True,
    )

    np.testing.assert_array_equal(result_logits, logits)
    np.testing.assert_array_equal(result_hidden, h_nextn)


def test_nextn_layer_default_returns_logits_only(monkeypatch) -> None:
    logits = np.array([[0.1, 0.2]], dtype=np.float32)
    monkeypatch.setattr(mtp_nextn, "qwen35_gguf_mtp_eh_proj_f32", lambda *a, **k: np.zeros((1, 2), dtype=np.float32))
    monkeypatch.setattr(mtp_nextn, "qwen35_gguf_mtp_attention_sublayer_f32", lambda *a, **k: np.zeros((1, 2), dtype=np.float32))
    monkeypatch.setattr(mtp_nextn, "qwen35_gguf_mtp_ffn_sublayer_f32", lambda *a, **k: np.zeros((1, 2), dtype=np.float32))
    def fake_shared_head(*args, **kwargs):
        assert kwargs.get("return_normed_hidden") is False
        return logits

    monkeypatch.setattr(mtp_nextn, "qwen35_gguf_mtp_shared_head_logits_f32", fake_shared_head)

    result = mtp_nextn.qwen35_gguf_mtp_nextn_layer_logits_f32(
        np.zeros((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32), np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32),
        np.ones(2, dtype=np.float32), np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32), np.ones(2, dtype=np.float32),
        np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32), np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32),
        GGMLQuantizationType.F32, GGMLQuantizationType.F32, GGMLQuantizationType.F32,
        np.zeros((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32),
        np.zeros((2, 1), dtype=np.float32), GGMLQuantizationType.F32, np.ones(2, dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32), num_heads=1, num_kv_heads=1, experts_used=1,
    )

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, logits)
