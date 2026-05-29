from __future__ import annotations

import numpy as np

from hipengine.kernels.cpu_reference.ops import step_dense_mlp, step_moe_mlp, step_moe_router
from hipengine.models.stepfun import STEPFUN_STEP37_GGUF


def _silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.float32(1.0) + np.exp(-x).astype(np.float32))


def _manual_dense_mlp(
    x: np.ndarray,
    gate_weight: np.ndarray,
    up_weight: np.ndarray,
    down_weight: np.ndarray,
    *,
    limit: float | None = None,
) -> np.ndarray:
    gate = _silu(x @ gate_weight.T)
    up = x @ up_weight.T
    if limit is not None:
        gate = np.minimum(gate, np.float32(limit))
        up = np.clip(up, np.float32(-limit), np.float32(limit))
    return (gate * up) @ down_weight.T


def _manual_router(
    x: np.ndarray,
    router_weight: np.ndarray,
    router_bias: np.ndarray | None,
    *,
    top_k: int,
    routing_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    logits = x.astype(np.float32) @ router_weight.astype(np.float32).T
    probs = np.float32(1.0) / (np.float32(1.0) + np.exp(-logits).astype(np.float32))
    ranking = probs if router_bias is None else probs + router_bias.astype(np.float32)[None, :]
    indices = np.argsort(-ranking, axis=1)[:, :top_k]
    gathered = np.take_along_axis(probs, indices, axis=1)
    weights = gathered / (np.sum(gathered, axis=1, keepdims=True) + np.float32(1e-20))
    return (weights * np.float32(routing_scale)).astype(np.float32), indices.astype(np.int64)


def test_step_dense_mlp_uses_ffn_tensor_names_and_swiglu_limit() -> None:
    templates = set(STEPFUN_STEP37_GGUF.weight_name_templates)
    assert "blk.{layer}.ffn_gate.weight" in templates
    assert "blk.{layer}.ffn_up.weight" in templates
    assert "blk.{layer}.ffn_down.weight" in templates

    x = np.asarray([[[2.0, -4.0, 1.5, 3.0], [-1.0, 0.5, 5.0, -2.5]]], dtype=np.float32)
    gate_weight = np.asarray(
        [[1.0, -1.5, 0.25, 0.5], [-2.0, 0.75, 1.0, -0.5], [0.5, 0.25, -1.25, 2.0]],
        dtype=np.float32,
    )
    up_weight = np.asarray(
        [[-1.0, 0.5, 2.0, 0.25], [1.5, -2.0, 0.5, 1.0], [0.25, 1.0, -0.75, -1.5]],
        dtype=np.float32,
    )
    down_weight = np.asarray(
        [[1.0, -0.5, 0.25], [-0.75, 1.25, 0.5], [0.5, 0.5, -1.0], [1.5, -0.25, 0.75]],
        dtype=np.float32,
    )

    out = step_dense_mlp(x, gate_weight, up_weight, down_weight, swiglu_limit=2.5)
    expected = _manual_dense_mlp(
        x.reshape(-1, x.shape[-1]),
        gate_weight,
        up_weight,
        down_weight,
        limit=2.5,
    ).reshape(x.shape)
    no_limit = step_dense_mlp(x, gate_weight, up_weight, down_weight)

    assert out.shape == x.shape
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
    assert not np.allclose(out, no_limit)


def test_step_moe_router_uses_bias_for_topk_and_unbiased_weights() -> None:
    rng = np.random.default_rng(9001)
    x = rng.normal(size=(2, 4)).astype(np.float32)
    router_weight = rng.normal(size=(10, 4)).astype(np.float32)
    router_bias = np.zeros((10,), dtype=np.float32)
    router_bias[0] = 10.0
    router_bias[7] = 1.0

    weights, indices, logits = step_moe_router(
        x,
        router_weight,
        router_bias=router_bias,
        top_k=3,
        routing_scale=3.0,
    )
    expected_weights, expected_indices = _manual_router(
        x,
        router_weight,
        router_bias,
        top_k=3,
        routing_scale=3.0,
    )
    unbiased_probs = np.float32(1.0) / (np.float32(1.0) + np.exp(-logits).astype(np.float32))
    gathered_probs = np.take_along_axis(unbiased_probs, indices, axis=1)

    assert indices.shape == (2, 3)
    assert np.all(indices[:, 0] == 0)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        weights,
        gathered_probs / gathered_probs.sum(axis=1, keepdims=True) * np.float32(3.0),
        rtol=1e-6,
        atol=1e-6,
    )


def test_step_moe_mlp_handles_288_expert_top8_shared_path_and_limits() -> None:
    rng = np.random.default_rng(20260529)
    hidden_size = 4
    intermediate = 5
    num_experts = 288
    top_k = 8
    x = np.asarray([[[10.0, -8.0, 6.0, -4.0]]], dtype=np.float32)
    rows = x.reshape(-1, hidden_size)
    router_weight = rng.normal(scale=0.25, size=(num_experts, hidden_size)).astype(np.float32)
    router_bias = np.zeros((num_experts,), dtype=np.float32)
    router_bias[[17, 42, 199]] = np.asarray([4.0, 3.0, 2.0], dtype=np.float32)
    expert_gate = rng.normal(scale=0.4, size=(num_experts, intermediate, hidden_size)).astype(np.float32)
    expert_up = rng.normal(scale=0.4, size=(num_experts, intermediate, hidden_size)).astype(np.float32)
    expert_down = rng.normal(scale=0.4, size=(num_experts, hidden_size, intermediate)).astype(np.float32)
    shared_gate = rng.normal(scale=0.7, size=(intermediate, hidden_size)).astype(np.float32)
    shared_up = rng.normal(scale=0.7, size=(intermediate, hidden_size)).astype(np.float32)
    shared_down = rng.normal(scale=0.7, size=(hidden_size, intermediate)).astype(np.float32)

    out, routing_weights, selected = step_moe_mlp(
        x,
        router_weight,
        expert_gate,
        expert_up,
        expert_down,
        router_bias=router_bias,
        shared_gate_weight=shared_gate,
        shared_up_weight=shared_up,
        shared_down_weight=shared_down,
        top_k=top_k,
        routing_scale=3.0,
        swiglu_limit=7.0,
        shared_swiglu_limit=16.0,
        return_router=True,
    )
    expected_weights, expected_indices = _manual_router(
        rows,
        router_weight,
        router_bias,
        top_k=top_k,
        routing_scale=3.0,
    )
    expected = np.zeros_like(rows, dtype=np.float32)
    for slot in range(top_k):
        expert = int(expected_indices[0, slot])
        expected[0] += (
            _manual_dense_mlp(
                rows,
                expert_gate[expert],
                expert_up[expert],
                expert_down[expert],
                limit=7.0,
            )[0]
            * expected_weights[0, slot]
        )
    expected += _manual_dense_mlp(rows, shared_gate, shared_up, shared_down, limit=16.0)
    no_limit = step_moe_mlp(
        x,
        router_weight,
        expert_gate,
        expert_up,
        expert_down,
        router_bias=router_bias,
        shared_gate_weight=shared_gate,
        shared_up_weight=shared_up,
        shared_down_weight=shared_down,
        top_k=top_k,
        routing_scale=3.0,
    )

    assert out.shape == (1, 1, hidden_size)
    assert routing_weights.shape == (1, 1, top_k)
    assert selected.shape == (1, 1, top_k)
    np.testing.assert_array_equal(selected.reshape(1, top_k), expected_indices)
    np.testing.assert_allclose(routing_weights.reshape(1, top_k), expected_weights, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out.reshape(1, hidden_size), expected, rtol=1e-5, atol=1e-5)
    assert not np.allclose(out, no_limit)
