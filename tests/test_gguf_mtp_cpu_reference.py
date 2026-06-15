from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import (
    qwen35_gguf_mtp_attention_sublayer,
    qwen35_gguf_mtp_boundary_logits,
    qwen35_gguf_mtp_eh_proj,
    qwen35_gguf_mtp_ffn_sublayer,
    qwen35_gguf_mtp_moe_routing,
    qwen35_gguf_mtp_nextn_layer_logits,
    qwen35_gguf_mtp_shared_head_logits,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType


NEXTN_FIXTURE = Path("benchmarks/fixtures/qwen35_gguf_mtp_nextn_cpu_reference_fixture.json")


def _rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    variance = np.mean(x * x, axis=-1, keepdims=True)
    return (x * np.reciprocal(np.sqrt(variance + eps))) * weight


def _f32(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _nextn_logits_from_fixture(inputs: dict[str, object], **kwargs: object) -> np.ndarray:
    return qwen35_gguf_mtp_nextn_layer_logits(
        _f32(inputs["hidden_seed"]),
        _f32(inputs["token_embedding"]),
        _f32(inputs["eh_proj_weight"]),
        _f32(inputs["hnorm_weight"]),
        _f32(inputs["enorm_weight"]),
        _f32(inputs["attn_norm_weight"]),
        _f32(inputs["wq_weight"]),
        _f32(inputs["wk_weight"]),
        _f32(inputs["wv_weight"]),
        _f32(inputs["wo_weight"]),
        _f32(inputs["q_norm_weight"]),
        _f32(inputs["k_norm_weight"]),
        _f32(inputs["attn_post_norm_weight"]),
        _f32(inputs["router_weight"]),
        _f32(inputs["gate_qweight"]),
        _f32(inputs["up_qweight"]),
        _f32(inputs["down_qweight"]),
        GGMLQuantizationType[str(inputs["gate_qtype"])],
        GGMLQuantizationType[str(inputs["up_qtype"])],
        GGMLQuantizationType[str(inputs["down_qtype"])],
        _f32(inputs["shared_gate_logit_weight"]),
        _f32(inputs["shared_gate_qweight"]),
        _f32(inputs["shared_up_qweight"]),
        _f32(inputs["shared_down_qweight"]),
        GGMLQuantizationType[str(inputs["shared_qtype"])],
        _f32(inputs["shared_head_norm_weight"]),
        _f32(inputs["shared_head_weight"]),
        **kwargs,
    )


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


def _simple_mtp_attention_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    hidden = np.asarray([[1.0, 0.0]], dtype=np.float32)
    attn_norm = np.ones((2,), dtype=np.float32)
    q_norm = np.ones((2,), dtype=np.float32)
    k_norm = np.ones((2,), dtype=np.float32)
    wq = np.zeros((4, 2), dtype=np.float32)
    wq[0, 0] = 1.0
    wq[1, 1] = 1.0
    wk = np.zeros((2, 2), dtype=np.float32)
    wv = np.eye(2, dtype=np.float32)
    wo = np.eye(2, dtype=np.float32)
    return hidden, attn_norm, q_norm, k_norm, wq, wk, wv, wo


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


def test_qwen35_gguf_mtp_attention_sublayer_supports_gguf_head_width_greater_than_hidden_per_head() -> None:
    hidden = np.asarray([[1.0, 2.0]], dtype=np.float32)
    attn_norm = np.ones((2,), dtype=np.float32)
    q_norm = np.ones((3,), dtype=np.float32)
    k_norm = np.ones((3,), dtype=np.float32)
    wq = np.zeros((12, 2), dtype=np.float32)
    # Two heads, qk/value dim 3: [Q0(3), gate0(3), Q1(3), gate1(3)].
    # This pins the real GGUF pattern where heads * key_length can exceed hidden.
    wq[3, 0] = 1.0
    wq[4, 1] = -1.0
    wq[5, :] = 0.5
    wq[9, 0] = -0.25
    wq[10, 1] = 0.75
    wq[11, :] = -0.5
    wk = np.zeros((3, 2), dtype=np.float32)
    wv = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=np.float32,
    )
    wo = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, -0.5, 1.0],
        ],
        dtype=np.float32,
    )

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
    q_full = np.matmul(normed, wq.T).reshape(1, 2, 2, 3)
    gate = q_full[:, :, 1, :]
    value = np.matmul(normed, wv.T).reshape(1, 1, 3)
    attn = np.repeat(value, repeats=2, axis=1)
    gated = (attn * (1.0 / (1.0 + np.exp(-gate)))).reshape(1, 6)
    expected = hidden + np.matmul(gated, wo.T)
    np.testing.assert_allclose(out, expected.astype(np.float32), rtol=1.0e-6, atol=1.0e-6)


def test_qwen35_gguf_mtp_attention_sublayer_accepts_kvlivespans_paged_cache() -> None:
    hidden, attn_norm, q_norm, k_norm, wq, wk, wv, wo = _simple_mtp_attention_fixture()
    dense_key = np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
    dense_value = np.asarray([[[2.0, 0.0]], [[0.0, 4.0]]], dtype=np.float32)
    paged_key = dense_key.reshape(1, 2, 1, 2)
    paged_value = dense_value.reshape(1, 2, 1, 2)

    dense = qwen35_gguf_mtp_attention_sublayer(
        hidden,
        attn_norm,
        wq,
        wk,
        wv,
        wo,
        q_norm,
        k_norm,
        num_heads=1,
        num_kv_heads=1,
        key_cache=dense_key,
        value_cache=dense_value,
        positions=np.asarray([1], dtype=np.int64),
        context_counts=np.asarray([2], dtype=np.int64),
    )
    paged = qwen35_gguf_mtp_attention_sublayer(
        hidden,
        attn_norm,
        wq,
        wk,
        wv,
        wo,
        q_norm,
        k_norm,
        num_heads=1,
        num_kv_heads=1,
        key_cache=paged_key,
        value_cache=paged_value,
        kv_base_offsets=np.asarray([[0]], dtype=np.int32),
        kv_live_counts=np.asarray([2], dtype=np.int64),
        kv_token_positions=np.asarray([1], dtype=np.int64),
        block_size=2,
    )

    np.testing.assert_allclose(paged, dense, rtol=1.0e-6, atol=1.0e-6)


def test_qwen35_gguf_mtp_attention_sublayer_kvlivespans_evict_mask_skips_slots() -> None:
    hidden, attn_norm, q_norm, k_norm, wq, wk, wv, wo = _simple_mtp_attention_fixture()
    dense_key = np.asarray([[[0.0, 1.0]]], dtype=np.float32)
    dense_value = np.asarray([[[0.0, 4.0]]], dtype=np.float32)
    paged_key = np.asarray([[[[1.0, 0.0]], [[0.0, 1.0]]]], dtype=np.float32)
    paged_value = np.asarray([[[[2.0, 0.0]], [[0.0, 4.0]]]], dtype=np.float32)

    expected = qwen35_gguf_mtp_attention_sublayer(
        hidden,
        attn_norm,
        wq,
        wk,
        wv,
        wo,
        q_norm,
        k_norm,
        num_heads=1,
        num_kv_heads=1,
        key_cache=dense_key,
        value_cache=dense_value,
        positions=np.asarray([0], dtype=np.int64),
        context_counts=np.asarray([1], dtype=np.int64),
    )
    masked = qwen35_gguf_mtp_attention_sublayer(
        hidden,
        attn_norm,
        wq,
        wk,
        wv,
        wo,
        q_norm,
        k_norm,
        num_heads=1,
        num_kv_heads=1,
        key_cache=paged_key,
        value_cache=paged_value,
        kv_base_offsets=np.asarray([[0]], dtype=np.int32),
        kv_live_counts=np.asarray([2], dtype=np.int64),
        kv_token_positions=np.asarray([1], dtype=np.int64),
        kv_evict_mask=np.asarray([[True, False]], dtype=np.bool_),
        block_size=2,
    )

    np.testing.assert_allclose(masked, expected, rtol=1.0e-6, atol=1.0e-6)


def test_qwen35_gguf_mtp_attention_sublayer_validates_shapes() -> None:
    hidden = np.ones((1, 4), dtype=np.float32)
    attn_norm = np.ones((4,), dtype=np.float32)
    head_norm = np.ones((2,), dtype=np.float32)
    wq = np.zeros((8, 4), dtype=np.float32)
    wk = np.zeros((2, 4), dtype=np.float32)
    wv = np.zeros((2, 4), dtype=np.float32)
    wo = np.eye(4, dtype=np.float32)

    with pytest.raises(ValueError, match="num_heads must be divisible"):
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
            num_kv_heads=3,
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
    with pytest.raises(ValueError, match="wo_weight must have shape"):
        qwen35_gguf_mtp_attention_sublayer(
            hidden,
            attn_norm,
            wq,
            wk,
            wv,
            np.zeros((4, 2), dtype=np.float32),
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


def test_qwen35_gguf_mtp_moe_routing_selects_softmax_topk_and_scales() -> None:
    hidden = np.asarray(
        [
            [2.0, 1.0],
            [0.0, 3.0],
        ],
        dtype=np.float32,
    )
    router = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 2.0],
        ],
        dtype=np.float32,
    )

    experts, weights = qwen35_gguf_mtp_moe_routing(
        hidden,
        router,
        experts_used=2,
        expert_weights_scale=1.25,
    )

    logits = np.matmul(hidden, router.T).astype(np.float32)
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = (probs / np.sum(probs, axis=-1, keepdims=True)).astype(np.float32)
    expected_experts = np.argsort(-probs, axis=-1, kind="stable")[:, :2]
    expected_weights = np.take_along_axis(probs, expected_experts, axis=-1)
    expected_weights = expected_weights / np.sum(expected_weights, axis=-1, keepdims=True)
    expected_weights = (expected_weights * np.float32(1.25)).astype(np.float32)

    np.testing.assert_array_equal(experts, expected_experts)
    np.testing.assert_allclose(weights, expected_weights, rtol=1.0e-6, atol=1.0e-6)


def test_qwen35_gguf_mtp_moe_routing_validates_shapes() -> None:
    hidden = np.ones((2, 3), dtype=np.float32)
    router = np.ones((4, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="hidden must have shape"):
        qwen35_gguf_mtp_moe_routing(hidden[0], router, experts_used=2)
    with pytest.raises(ValueError, match="router_weight must have shape"):
        qwen35_gguf_mtp_moe_routing(hidden, np.ones((4, 4), dtype=np.float32), experts_used=2)
    with pytest.raises(ValueError, match="experts_used must be positive"):
        qwen35_gguf_mtp_moe_routing(hidden, router, experts_used=0)
    with pytest.raises(ValueError, match="experts_used must be <= number of experts"):
        qwen35_gguf_mtp_moe_routing(hidden, router, experts_used=5)


def test_qwen35_gguf_mtp_ffn_sublayer_applies_norm_routing_shared_expert_and_residual() -> None:
    hidden = np.asarray([[0.25, -0.5]], dtype=np.float32)
    norm = np.asarray([1.5, 0.5], dtype=np.float32)
    router = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    gate_q = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.0], [0.0, 0.5]],
        ],
        dtype=np.float32,
    )
    up_q = np.asarray(
        [
            [[0.25, 0.0], [0.0, 0.25]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    down_q = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ],
        dtype=np.float32,
    )
    shared_gate_logit = np.asarray([0.75, -0.25], dtype=np.float32)
    shared_gate_q = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    shared_up_q = np.asarray([[0.5, 0.0], [0.0, 0.5]], dtype=np.float32)
    shared_down_q = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    out = qwen35_gguf_mtp_ffn_sublayer(
        hidden,
        norm,
        router,
        gate_q,
        up_q,
        down_q,
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F32,
        shared_gate_logit,
        shared_gate_q,
        shared_up_q,
        shared_down_q,
        GGMLQuantizationType.F32,
        experts_used=1,
        expert_weights_scale=1.0,
    )

    normed = _rmsnorm(hidden, norm)
    logits = np.matmul(normed, router.T).astype(np.float32)
    expert = int(np.argmax(logits[0]))
    gate = np.matmul(normed, gate_q[expert].T)
    up = np.matmul(normed, up_q[expert].T)
    selected = np.matmul((gate / (1.0 + np.exp(-gate))) * up, down_q[expert].T)
    shared_gate = np.matmul(normed, shared_gate_q.T)
    shared_up = np.matmul(normed, shared_up_q.T)
    shared_out = np.matmul((shared_gate / (1.0 + np.exp(-shared_gate))) * shared_up, shared_down_q.T)
    shared_scale = 1.0 / (1.0 + np.exp(-np.matmul(normed, shared_gate_logit)))
    expected = hidden + selected + shared_scale[:, None] * shared_out
    np.testing.assert_allclose(out, expected.astype(np.float32), rtol=1.0e-6, atol=1.0e-6)


def test_qwen35_gguf_mtp_ffn_sublayer_validates_norm_shape() -> None:
    hidden = np.ones((1, 2), dtype=np.float32)
    weight = np.ones((1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="attn_post_norm_weight must have shape"):
        qwen35_gguf_mtp_ffn_sublayer(
            hidden,
            np.ones((3,), dtype=np.float32),
            np.ones((1, 2), dtype=np.float32),
            weight,
            weight,
            weight,
            GGMLQuantizationType.F32,
            GGMLQuantizationType.F32,
            GGMLQuantizationType.F32,
            np.ones((2,), dtype=np.float32),
            np.eye(2, dtype=np.float32),
            np.eye(2, dtype=np.float32),
            np.eye(2, dtype=np.float32),
            GGMLQuantizationType.F32,
            experts_used=1,
        )


def test_qwen35_gguf_mtp_nextn_layer_logits_composes_pinned_sublayers() -> None:
    hidden_seed = np.asarray([[0.2, -0.4]], dtype=np.float32)
    token_embedding = np.asarray([[0.3, 0.1]], dtype=np.float32)
    hnorm = np.asarray([1.0, 1.5], dtype=np.float32)
    enorm = np.asarray([0.5, 2.0], dtype=np.float32)
    # Keep the hidden width fixed, as llama.cpp's eh_proj maps [e_norm, h_norm]
    # back to the model hidden dimension.
    eh_proj = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    attn_norm = np.ones((2,), dtype=np.float32)
    q_norm = np.ones((2,), dtype=np.float32)
    k_norm = np.ones((2,), dtype=np.float32)
    wq = np.zeros((4, 2), dtype=np.float32)
    # One head: [Q_head, gate_head]. With a single visible token, gate rows pin
    # that the composed helper calls the attention sublayer in the llama.cpp order.
    wq[2, 0] = 1.0
    wq[3, 1] = -1.0
    wk = np.zeros((2, 2), dtype=np.float32)
    wv = np.eye(2, dtype=np.float32)
    wo = np.eye(2, dtype=np.float32)

    attn_post_norm = np.ones((2,), dtype=np.float32)
    router = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    gate_q = np.stack([np.eye(2, dtype=np.float32), 0.5 * np.eye(2, dtype=np.float32)]).astype(np.float32)
    up_q = np.stack([0.25 * np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32)]).astype(np.float32)
    down_q = np.stack([np.eye(2, dtype=np.float32), 2.0 * np.eye(2, dtype=np.float32)]).astype(np.float32)
    shared_gate_logit = np.asarray([0.75, -0.25], dtype=np.float32)
    shared_gate_q = np.eye(2, dtype=np.float32)
    shared_up_q = 0.5 * np.eye(2, dtype=np.float32)
    shared_down_q = np.eye(2, dtype=np.float32)
    shared_norm = np.asarray([1.0, 1.25], dtype=np.float32)
    shared_head = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, -1.0],
        ],
        dtype=np.float32,
    )

    logits = qwen35_gguf_mtp_nextn_layer_logits(
        hidden_seed,
        token_embedding,
        eh_proj,
        hnorm,
        enorm,
        attn_norm,
        wq,
        wk,
        wv,
        wo,
        q_norm,
        k_norm,
        attn_post_norm,
        router,
        gate_q,
        up_q,
        down_q,
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F32,
        shared_gate_logit,
        shared_gate_q,
        shared_up_q,
        shared_down_q,
        GGMLQuantizationType.F32,
        shared_norm,
        shared_head,
        num_heads=1,
        num_kv_heads=1,
        experts_used=1,
    )

    projected = qwen35_gguf_mtp_eh_proj(hidden_seed, token_embedding, eh_proj, hnorm, enorm)
    attended = qwen35_gguf_mtp_attention_sublayer(
        projected,
        attn_norm,
        wq,
        wk,
        wv,
        wo,
        q_norm,
        k_norm,
        num_heads=1,
        num_kv_heads=1,
    )
    ffn_out = qwen35_gguf_mtp_ffn_sublayer(
        attended,
        attn_post_norm,
        router,
        gate_q,
        up_q,
        down_q,
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F32,
        shared_gate_logit,
        shared_gate_q,
        shared_up_q,
        shared_down_q,
        GGMLQuantizationType.F32,
        experts_used=1,
    )
    expected = qwen35_gguf_mtp_shared_head_logits(ffn_out, shared_norm, shared_head)
    np.testing.assert_allclose(logits, expected, rtol=1.0e-6, atol=1.0e-6)


def test_qwen35_gguf_mtp_nextn_layer_logits_accepts_kvlivespans_paged_cache() -> None:
    fixture = json.loads(NEXTN_FIXTURE.read_text())
    inputs = fixture["inputs"]
    kwargs = dict(fixture["kwargs"])
    dense_key = np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
    dense_value = np.asarray([[[2.0, 0.0]], [[0.0, 4.0]]], dtype=np.float32)

    dense = _nextn_logits_from_fixture(
        inputs,
        **kwargs,
        key_cache=dense_key,
        value_cache=dense_value,
        positions=np.asarray([1], dtype=np.int64),
        context_counts=np.asarray([2], dtype=np.int64),
    )
    paged = _nextn_logits_from_fixture(
        inputs,
        **kwargs,
        key_cache=dense_key.reshape(1, 2, 1, 2),
        value_cache=dense_value.reshape(1, 2, 1, 2),
        kv_base_offsets=np.asarray([[0]], dtype=np.int32),
        kv_live_counts=np.asarray([2], dtype=np.int64),
        kv_token_positions=np.asarray([1], dtype=np.int64),
        block_size=2,
    )

    assert dense.dtype == np.float32
    assert paged.dtype == np.float32
    assert np.isfinite(paged).all()
    np.testing.assert_allclose(paged, dense, rtol=1.0e-6, atol=1.0e-6)


def test_qwen35_gguf_mtp_nextn_fixture_produces_finite_logits_and_topk() -> None:
    fixture = json.loads(NEXTN_FIXTURE.read_text())
    assert fixture["cpu_reference_kernel"] == [
        "cpu_reference",
        "mtp_nextn_layer",
        "gguf_moe",
        "qwen35_dense_logits",
    ]
    kernel = resolve(
        backend="cpu_reference",
        layer="mtp_nextn_layer",
        quant="gguf_moe",
        variant="qwen35_dense_logits",
    )
    inputs = fixture["inputs"]
    logits = kernel(
        _f32(inputs["hidden_seed"]),
        _f32(inputs["token_embedding"]),
        _f32(inputs["eh_proj_weight"]),
        _f32(inputs["hnorm_weight"]),
        _f32(inputs["enorm_weight"]),
        _f32(inputs["attn_norm_weight"]),
        _f32(inputs["wq_weight"]),
        _f32(inputs["wk_weight"]),
        _f32(inputs["wv_weight"]),
        _f32(inputs["wo_weight"]),
        _f32(inputs["q_norm_weight"]),
        _f32(inputs["k_norm_weight"]),
        _f32(inputs["attn_post_norm_weight"]),
        _f32(inputs["router_weight"]),
        _f32(inputs["gate_qweight"]),
        _f32(inputs["up_qweight"]),
        _f32(inputs["down_qweight"]),
        GGMLQuantizationType[inputs["gate_qtype"]],
        GGMLQuantizationType[inputs["up_qtype"]],
        GGMLQuantizationType[inputs["down_qtype"]],
        _f32(inputs["shared_gate_logit_weight"]),
        _f32(inputs["shared_gate_qweight"]),
        _f32(inputs["shared_up_qweight"]),
        _f32(inputs["shared_down_qweight"]),
        GGMLQuantizationType[inputs["shared_qtype"]],
        _f32(inputs["shared_head_norm_weight"]),
        _f32(inputs["shared_head_weight"]),
        **fixture["kwargs"],
    )

    expected_logits = _f32(fixture["expected"]["logits"])
    assert logits.dtype == np.float32
    assert np.isfinite(logits).all()
    np.testing.assert_allclose(logits, expected_logits, rtol=1.0e-6, atol=1.0e-6)
    top_k = int(fixture["top_k"])
    top_k_ids = np.argsort(-logits[0], kind="stable")[:top_k]
    np.testing.assert_array_equal(
        top_k_ids,
        np.asarray(fixture["expected"]["top_k_token_ids"]),
    )
    np.testing.assert_allclose(
        logits[0, top_k_ids],
        _f32(fixture["expected"]["top_k_logits"]),
        rtol=1.0e-6,
        atol=1.0e-6,
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
    moe_routing = resolve(
        backend="cpu_reference",
        layer="mtp_nextn_moe_routing",
        quant="gguf_f32",
        variant="qwen35_softmax_topk",
    )
    ffn = resolve(
        backend="cpu_reference",
        layer="mtp_nextn_ffn",
        quant="gguf_moe",
        variant="qwen35_shared",
    )
    nextn_layer = resolve(
        backend="cpu_reference",
        layer="mtp_nextn_layer",
        quant="gguf_moe",
        variant="qwen35_dense_logits",
    )

    assert eh_proj is qwen35_gguf_mtp_eh_proj
    assert shared_head is qwen35_gguf_mtp_shared_head_logits
    assert boundary_logits is qwen35_gguf_mtp_boundary_logits
    assert attention is qwen35_gguf_mtp_attention_sublayer
    assert moe_routing is qwen35_gguf_mtp_moe_routing
    assert ffn is qwen35_gguf_mtp_ffn_sublayer
    assert nextn_layer is qwen35_gguf_mtp_nextn_layer_logits
