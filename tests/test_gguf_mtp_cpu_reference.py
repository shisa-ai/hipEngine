from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import (
    qwen35_gguf_mtp_attention_sublayer,
    qwen35_gguf_mtp_boundary_logits,
    qwen35_gguf_mtp_eh_proj,
    qwen35_gguf_mtp_shared_head_logits,
)
from hipengine.kernels.registry import resolve


def _rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    variance = np.mean(x * x, axis=-1, keepdims=True)
    return (x * np.reciprocal(np.sqrt(variance + eps))) * weight


def test_qwen35_gguf_mtp_eh_proj_normalizes_and_concatenates_embedding_then_hidden() -> None:
    hidden = np.asarray([[3.0, 4.0]], dtype=np.float32)
    embedding = np.asarray([[1.0, 2.0]], dtype=np.float32)
    hnorm = np.asarray([10.0, 20.0], dtype=np.float32)
    enorm = np.asarray([30.0, 40.0], dtype=np.float32)
    # Select [e_norm[0], h_norm[1]] to pin llama.cpp concat order:
    # concat = [e_norm, h_norm], not [h_norm, e_norm].
    weight = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    out = qwen35_gguf_mtp_eh_proj(hidden, embedding, weight, hnorm, enorm)

    h_norm = _rmsnorm(hidden, hnorm)
    e_norm = _rmsnorm(embedding, enorm)
    expected = np.asarray([[e_norm[0, 0], h_norm[0, 1]]], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1.0e-6, atol=1.0e-6)


def test_qwen35_gguf_mtp_eh_proj_supports_multiple_rows() -> None:
    hidden = np.asarray([[1.0, 2.0], [5.0, 6.0]], dtype=np.float32)
    embedding = np.asarray([[3.0, 4.0], [7.0, 8.0]], dtype=np.float32)
    hnorm = np.ones((2,), dtype=np.float32)
    enorm = np.asarray([2.0, 3.0], dtype=np.float32)
    weight = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    out = qwen35_gguf_mtp_eh_proj(hidden, embedding, weight, hnorm, enorm)

    fused = np.concatenate([_rmsnorm(embedding, enorm), _rmsnorm(hidden, hnorm)], axis=-1)
    expected = np.matmul(fused, weight.T).astype(np.float32)
    np.testing.assert_allclose(out, expected, rtol=1.0e-6, atol=1.0e-6)


def test_qwen35_gguf_mtp_eh_proj_validates_shapes() -> None:
    hidden = np.asarray([[1.0, 2.0]], dtype=np.float32)
    embedding = np.asarray([[3.0, 4.0]], dtype=np.float32)
    weight = np.zeros((2, 4), dtype=np.float32)
    norm = np.ones((2,), dtype=np.float32)

    with pytest.raises(ValueError, match="hidden_seed must have shape"):
        qwen35_gguf_mtp_eh_proj(hidden[0], embedding, weight, norm, norm)
    with pytest.raises(ValueError, match="token_embedding must match"):
        qwen35_gguf_mtp_eh_proj(hidden, np.zeros((2, 2), dtype=np.float32), weight, norm, norm)
    with pytest.raises(ValueError, match="eh_proj_weight must have shape"):
        qwen35_gguf_mtp_eh_proj(hidden, embedding, np.zeros((2, 3), dtype=np.float32), norm, norm)
    with pytest.raises(ValueError, match="hnorm_weight must have shape"):
        qwen35_gguf_mtp_eh_proj(hidden, embedding, weight, np.ones((3,), dtype=np.float32), norm)
    with pytest.raises(ValueError, match="enorm_weight must have shape"):
        qwen35_gguf_mtp_eh_proj(hidden, embedding, weight, norm, np.ones((3,), dtype=np.float32))


def test_qwen35_gguf_mtp_shared_head_logits_applies_norm_then_head() -> None:
    hidden = np.asarray([[3.0, 4.0]], dtype=np.float32)
    norm = np.asarray([10.0, 20.0], dtype=np.float32)
    head = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    logits = qwen35_gguf_mtp_shared_head_logits(hidden, norm, head)

    expected_norm = _rmsnorm(hidden, norm)
    expected = np.matmul(expected_norm, head.T).astype(np.float32)
    np.testing.assert_allclose(logits, expected, rtol=1.0e-6, atol=1.0e-6)


def test_qwen35_gguf_mtp_shared_head_logits_validates_shapes() -> None:
    hidden = np.asarray([[1.0, 2.0]], dtype=np.float32)
    norm = np.ones((2,), dtype=np.float32)
    head = np.ones((3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="nextn_hidden must have shape"):
        qwen35_gguf_mtp_shared_head_logits(hidden[0], norm, head)
    with pytest.raises(ValueError, match="shared_head_norm_weight must have shape"):
        qwen35_gguf_mtp_shared_head_logits(hidden, np.ones((3,), dtype=np.float32), head)
    with pytest.raises(ValueError, match="shared_head_weight must have shape"):
        qwen35_gguf_mtp_shared_head_logits(hidden, norm, np.ones((3, 3), dtype=np.float32))


def test_qwen35_gguf_mtp_boundary_logits_composes_pinned_boundary_stages() -> None:
    hidden = np.asarray([[3.0, 4.0]], dtype=np.float32)
    embedding = np.asarray([[1.0, 2.0]], dtype=np.float32)
    hnorm = np.asarray([10.0, 20.0], dtype=np.float32)
    enorm = np.asarray([30.0, 40.0], dtype=np.float32)
    eh_proj = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    shared_norm = np.asarray([2.0, 3.0], dtype=np.float32)
    shared_head = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=np.float32,
    )

    logits = qwen35_gguf_mtp_boundary_logits(
        hidden,
        embedding,
        eh_proj,
        hnorm,
        enorm,
        shared_norm,
        shared_head,
    )

    projected = qwen35_gguf_mtp_eh_proj(hidden, embedding, eh_proj, hnorm, enorm)
    expected = qwen35_gguf_mtp_shared_head_logits(projected, shared_norm, shared_head)
    np.testing.assert_allclose(logits, expected, rtol=1.0e-6, atol=1.0e-6)


def test_qwen35_gguf_mtp_attention_sublayer_uses_interleaved_q_gate_layout() -> None:
    hidden = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    attn_norm = np.ones((4,), dtype=np.float32)
    q_norm = np.ones((2,), dtype=np.float32)
    k_norm = np.ones((2,), dtype=np.float32)
    wq = np.zeros((8, 4), dtype=np.float32)
    # llama.cpp views wq output as [Q_head0, gate_head0, Q_head1, gate_head1].
    # With one visible token the Q rows do not affect softmax, while the gate rows
    # directly scale the attention value and therefore pin the interleaved layout.
    wq[2, 0] = 1.0
    wq[3, 1] = -1.0
    wq[6, 2] = 1.0
    wq[7, 3] = -1.0
    wk = np.zeros((2, 4), dtype=np.float32)
    wv = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    wo = np.eye(4, dtype=np.float32)

    out = qwen35_gguf_mtp_attention_sublayer(
        hidden,
        attn_norm,
        wq,
        wk,
        wv,
        wo,
        q_norm,
        k_norm,
        num_heads=2,
        num_kv_heads=1,
    )

    normed = _rmsnorm(hidden, attn_norm)
    q_full = np.matmul(normed, wq.T).reshape(1, 2, 2, 2)
    gate = q_full[:, :, 1, :]
    value = np.matmul(normed, wv.T).reshape(1, 1, 2)
    attn = np.repeat(value, repeats=2, axis=1)
    gated = (attn * (1.0 / (1.0 + np.exp(-gate)))).reshape(1, 4)
    expected = hidden + gated
    np.testing.assert_allclose(out, expected, rtol=1.0e-6, atol=1.0e-6)


def test_qwen35_gguf_mtp_attention_sublayer_validates_shapes() -> None:
    hidden = np.ones((1, 4), dtype=np.float32)
    attn_norm = np.ones((4,), dtype=np.float32)
    head_norm = np.ones((2,), dtype=np.float32)
    wq = np.zeros((8, 4), dtype=np.float32)
    wk = np.zeros((2, 4), dtype=np.float32)
    wv = np.zeros((2, 4), dtype=np.float32)
    wo = np.eye(4, dtype=np.float32)

    with pytest.raises(ValueError, match="hidden size must be divisible"):
        qwen35_gguf_mtp_attention_sublayer(
            hidden,
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            head_norm,
            head_norm,
            num_heads=3,
            num_kv_heads=1,
        )
    with pytest.raises(ValueError, match="wq_weight must have shape"):
        qwen35_gguf_mtp_attention_sublayer(
            hidden,
            attn_norm,
            np.zeros((4, 4), dtype=np.float32),
            wk,
            wv,
            wo,
            head_norm,
            head_norm,
            num_heads=2,
            num_kv_heads=1,
        )
    with pytest.raises(ValueError, match="rope_cos and rope_sin"):
        qwen35_gguf_mtp_attention_sublayer(
            hidden,
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            head_norm,
            head_norm,
            num_heads=2,
            num_kv_heads=1,
            rope_cos=np.ones((1, 1, 2), dtype=np.float32),
        )


def test_qwen35_gguf_mtp_cpu_helpers_are_registered() -> None:
    eh_proj = resolve(
        backend="cpu_reference",
        layer="mtp_nextn_eh_proj",
        quant="gguf_f32",
        variant="qwen35",
    )
    shared_head = resolve(
        backend="cpu_reference",
        layer="mtp_nextn_shared_head",
        quant="gguf_f32",
        variant="qwen35",
    )
    boundary_logits = resolve(
        backend="cpu_reference",
        layer="mtp_nextn_boundary_logits",
        quant="gguf_f32",
        variant="qwen35",
    )
    attention = resolve(
        backend="cpu_reference",
        layer="mtp_nextn_attention",
        quant="gguf_f32",
        variant="qwen35_dense",
    )

    assert eh_proj is qwen35_gguf_mtp_eh_proj
    assert shared_head is qwen35_gguf_mtp_shared_head_logits
    assert boundary_logits is qwen35_gguf_mtp_boundary_logits
    assert attention is qwen35_gguf_mtp_attention_sublayer
