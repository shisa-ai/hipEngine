from __future__ import annotations

from dataclasses import replace

import numpy as np

from hipengine.kernels.cpu_reference.qwen4_exp import (
    Qwen4ExpGRWeights,
    Qwen4ExpMoEWeights,
    Qwen4ExpQSAWeights,
    Qwen4ExpReducedLayerWeights,
    qwen4_exp_reduced_qsa_layer,
)


def _weights(*, seed: int, zero: bool) -> Qwen4ExpReducedLayerWeights:
    rng = np.random.default_rng(seed)
    branches, hidden, low_rank = 2, 4, 2
    heads, kv_heads, head_dim = 2, 1, 2
    index_heads, index_dim = 2, 2
    experts, ffn = 3, 3

    def array(shape: tuple[int, ...]) -> np.ndarray:
        if zero:
            return np.zeros(shape, dtype=np.float32)
        return (rng.normal(size=shape) * 0.15).astype(np.float32)

    def gr() -> Qwen4ExpGRWeights:
        return Qwen4ExpGRWeights(
            norm=np.ones((branches, hidden), dtype=np.float32),
            down=array((low_rank, branches * hidden)),
            up=array((branches * hidden, low_rank)),
            inject=array((branches, branches * hidden)),
        )

    return Qwen4ExpReducedLayerWeights(
        attention_gr=gr(),
        qsa=Qwen4ExpQSAWeights(
            q=array((heads * 2 * head_dim, hidden)),
            k=array((kv_heads * head_dim, hidden)),
            v=array((kv_heads * head_dim, hidden)),
            output=array((hidden, heads * head_dim)),
            q_norm=np.ones(head_dim, dtype=np.float32),
            k_norm=np.ones(head_dim, dtype=np.float32),
            index_q=array((index_heads * index_dim, hidden)),
            index_k=array((index_dim, hidden)),
            index_q_norm=np.ones(index_dim, dtype=np.float32),
            index_k_norm=np.ones(index_dim, dtype=np.float32),
            query_heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            index_heads=index_heads,
            index_dim=index_dim,
        ),
        ffn_gr=gr(),
        moe=Qwen4ExpMoEWeights(
            router=array((experts, hidden)),
            expert_gate=array((experts, ffn, hidden)),
            expert_up=array((experts, ffn, hidden)),
            expert_down=array((experts, hidden, ffn)),
            shared_gate=array((ffn, hidden)),
            shared_up=array((ffn, hidden)),
            shared_down=array((hidden, ffn)),
            shared_gate_weight=array((hidden,)),
            experts_used=2,
        ),
    )


def test_reduced_qwen4_exp_qsa_layer_zero_blocks_preserve_four_branch_state() -> None:
    residual = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
            [[2.0, -1.0, 0.5, 3.0], [-2.0, 1.0, 4.0, 0.5]],
            [[1.5, 2.5, -0.5, 1.0], [0.5, -1.5, 2.0, 3.5]],
        ],
        dtype=np.float32,
    )
    before = residual.copy()

    result = qwen4_exp_reduced_qsa_layer(
        residual,
        _weights(seed=1, zero=True),
        positions=np.arange(3),
        compression_ratio=2,
        block_budget=1,
        rotary_dim=2,
        theta=100.0,
        eps=1e-6,
    )

    np.testing.assert_array_equal(residual, before)
    np.testing.assert_allclose(result.residual, before, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.attention_output, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.moe.output, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(result.selection.selected_positions[0], [0])
    np.testing.assert_array_equal(result.selection.selected_positions[1], [0, 1])
    np.testing.assert_array_equal(result.selection.selected_positions[2], [0, 1, 2])


def test_reduced_qwen4_exp_qsa_layer_is_pure_deterministic_and_keeps_branches_distinct() -> None:
    rng = np.random.default_rng(9)
    residual = rng.normal(size=(5, 2, 4)).astype(np.float32)
    weights = _weights(seed=11, zero=False)

    first = qwen4_exp_reduced_qsa_layer(
        residual,
        weights,
        positions=np.arange(5),
        compression_ratio=2,
        block_budget=1,
        rotary_dim=2,
        theta=100.0,
    )
    second = qwen4_exp_reduced_qsa_layer(
        residual.copy(),
        weights,
        positions=np.arange(5),
        compression_ratio=2,
        block_budget=1,
        rotary_dim=2,
        theta=100.0,
    )

    np.testing.assert_array_equal(first.residual, second.residual)
    assert np.isfinite(first.residual).all()
    assert not np.array_equal(first.residual[:, 0], first.residual[:, 1])
    assert first.moe.selected_experts.shape == (5, 2)
    np.testing.assert_allclose(np.sum(first.moe.routing_weights, axis=1), 1.0, atol=1e-6)
    assert first.selection.selected_block_starts[-1].shape == (1,)
    assert first.selection.selected_positions[-1][-1] == 4
    assert first.selection.selected_positions[-1].shape == (3,)


def test_reduced_qwen4_exp_qsa_layer_changes_only_ffn_when_moe_changes() -> None:
    rng = np.random.default_rng(21)
    residual = rng.normal(size=(3, 2, 4)).astype(np.float32)
    weights = _weights(seed=22, zero=False)
    zero_moe = replace(
        weights,
        moe=_weights(seed=22, zero=True).moe,
    )

    full = qwen4_exp_reduced_qsa_layer(
        residual,
        weights,
        positions=np.arange(3),
        compression_ratio=2,
        block_budget=1,
        rotary_dim=2,
        theta=100.0,
    )
    no_moe = qwen4_exp_reduced_qsa_layer(
        residual,
        zero_moe,
        positions=np.arange(3),
        compression_ratio=2,
        block_budget=1,
        rotary_dim=2,
        theta=100.0,
    )

    np.testing.assert_array_equal(full.attention_output, no_moe.attention_output)
    assert not np.array_equal(full.residual, no_moe.residual)
