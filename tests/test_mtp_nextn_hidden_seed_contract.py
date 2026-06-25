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


def test_nextn_layer_forwards_draft_vocab_cap_to_shared_head(monkeypatch) -> None:
    logits = np.array([[0.1, 0.2]], dtype=np.float32)
    monkeypatch.setattr(mtp_nextn, "qwen35_gguf_mtp_eh_proj_f32", lambda *a, **k: np.zeros((1, 2), dtype=np.float32))
    monkeypatch.setattr(mtp_nextn, "qwen35_gguf_mtp_attention_sublayer_f32", lambda *a, **k: np.zeros((1, 2), dtype=np.float32))
    monkeypatch.setattr(mtp_nextn, "qwen35_gguf_mtp_ffn_sublayer_f32", lambda *a, **k: np.zeros((1, 2), dtype=np.float32))

    def fake_shared_head(*args, **kwargs):
        assert kwargs["draft_vocab_cap"] == 32768
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
        draft_vocab_cap=32768,
    )

    np.testing.assert_array_equal(result, logits)


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


def test_nextn_layer_forwards_kvlivespans_kwargs_to_attention(monkeypatch) -> None:
    """The composite NextN wrapper must not drop paged-MTP context inputs."""

    logits = np.array([[0.1, 0.2]], dtype=np.float32)
    key_cache = np.zeros((1, 2, 1, 2), dtype=np.float32)
    value_cache = np.zeros((1, 2, 1, 2), dtype=np.float32)
    kv_base_offsets = np.array([[0]], dtype=np.int64)
    kv_live_counts = np.array([1], dtype=np.int64)
    kv_token_positions = np.array([5], dtype=np.int64)
    kv_evict_mask = np.array([[False]], dtype=bool)

    monkeypatch.setattr(mtp_nextn, "qwen35_gguf_mtp_eh_proj_f32", lambda *a, **k: np.zeros((1, 2), dtype=np.float32))

    def fake_attention(hidden, *args, **kwargs):
        assert kwargs["key_cache"] is key_cache
        assert kwargs["value_cache"] is value_cache
        assert kwargs["kv_base_offsets"] is kv_base_offsets
        assert kwargs["kv_live_counts"] is kv_live_counts
        assert kwargs["kv_token_positions"] is kv_token_positions
        assert kwargs["kv_evict_mask"] is kv_evict_mask
        assert kwargs["block_size"] == 2
        return np.zeros((1, 2), dtype=np.float32)

    monkeypatch.setattr(mtp_nextn, "qwen35_gguf_mtp_attention_sublayer_f32", fake_attention)
    monkeypatch.setattr(mtp_nextn, "qwen35_gguf_mtp_ffn_sublayer_f32", lambda *a, **k: np.zeros((1, 2), dtype=np.float32))
    monkeypatch.setattr(mtp_nextn, "qwen35_gguf_mtp_shared_head_logits_f32", lambda *a, **k: logits)

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
        np.zeros((2, 2), dtype=np.float32),
        num_heads=1,
        num_kv_heads=1,
        experts_used=1,
        key_cache=key_cache,
        value_cache=value_cache,
        kv_base_offsets=kv_base_offsets,
        kv_live_counts=kv_live_counts,
        kv_token_positions=kv_token_positions,
        kv_evict_mask=kv_evict_mask,
        block_size=2,
    )

    np.testing.assert_array_equal(result, logits)
