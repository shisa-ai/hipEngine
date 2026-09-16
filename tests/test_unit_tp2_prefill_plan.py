"""CPU ownership/shape/state contract for rank-local bulk TP2 prefill.

No device, no model file, no GPU: these tests pin the planning geometry and the
independent reference math that the GPU execution packets must satisfy. They are
RED until ``hipengine.distributed.tp2_prefill`` exists.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from hipengine.distributed.tp2_prefill import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    PrefillPlanError,
    batched_sharded_mlp_reference,
    chunk_ranges,
    plan_batched_prefill,
    silu,
)


@dataclass(frozen=True)
class Config:
    hidden_size: int = 5120
    feed_forward_length: int = 17408
    head_count: int = 24
    head_count_kv: int = 4
    key_length: int = 256
    value_length: int = 256
    ssm_inner_size: int = 6144
    ssm_group_count: int = 16
    ssm_state_size: int = 128
    ssm_conv_kernel: int = 4
    ssm_time_step_rank: int = 48
    layer_types: tuple[str, ...] = tuple(
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(64)
    )


def test_plan_geometry_matches_the_supplied_model():
    plan = plan_batched_prefill(Config(), num_ranks=2, chunk_rows=16)
    assert plan.num_ranks == 2
    assert plan.chunk_rows == 16
    assert plan.hidden == 5120
    assert plan.exchange_payload_floats == 16 * 5120
    assert len(plan.linear_attention_layers) == 48
    assert len(plan.full_attention_layers) == 16
    linear = plan.layers[0]
    assert linear.layer_type == LINEAR_ATTENTION
    assert linear.linear_qkv_width == 10240
    assert linear.conv_state == (4, 10240)
    assert linear.recurrent_state == (48, 128, 128)
    assert linear.gate_shape == (8704, 5120)
    assert linear.up_shape == (8704, 5120)
    assert linear.down_shape == (5120, 8704)
    full = plan.layers[3]
    assert full.layer_type == FULL_ATTENTION
    assert full.kv_heads == 4
    assert full.key_length == 256
    assert full.value_length == 256
    assert full.conv_state is None and full.recurrent_state is None
    plan.validate()


def test_plan_rejects_a_non_block_aligned_degree():
    # N=8 makes per_rank_ffn 2176, which is not a 256-element quant block.
    with pytest.raises(PrefillPlanError):
        plan_batched_prefill(Config(), num_ranks=8, chunk_rows=16)


def test_plan_rejects_a_non_positive_chunk():
    with pytest.raises(PrefillPlanError):
        plan_batched_prefill(Config(), num_ranks=2, chunk_rows=0)
    with pytest.raises(PrefillPlanError):
        plan_batched_prefill(Config(), num_ranks=0, chunk_rows=16)


def test_chunk_ranges_cover_rows_without_gaps_or_overlap():
    assert chunk_ranges(64, 16) == ((0, 16), (16, 32), (32, 48), (48, 64))
    assert chunk_ranges(65, 16) == ((0, 16), (16, 32), (32, 48), (48, 64), (64, 65))
    assert chunk_ranges(8, 16) == ((0, 8),)
    assert chunk_ranges(0, 16) == ()
    for total, chunk in ((64, 16), (65, 16), (100, 7), (1, 1)):
        ranges = chunk_ranges(total, chunk)
        assert ranges[0][0] == 0
        assert ranges[-1][1] == total
        for (a_start, a_stop), (b_start, b_stop) in zip(ranges, ranges[1:]):
            assert a_stop == b_start
            assert a_stop > a_start and b_stop > b_start


def test_batched_sharded_mlp_reference_matches_the_unsplit_matmul():
    rng = np.random.default_rng(0)
    hidden, ffn, rows, ranks = 4, 8, 3, 2
    gate = rng.standard_normal((ffn, hidden), dtype=np.float32)
    up = rng.standard_normal((ffn, hidden), dtype=np.float32)
    down = rng.standard_normal((hidden, ffn), dtype=np.float32)
    x = rng.standard_normal((rows, hidden), dtype=np.float32)
    out, partials = batched_sharded_mlp_reference(gate, up, down, x, num_ranks=ranks)
    reference = (silu(x @ gate.T) * (x @ up.T)) @ down.T
    np.testing.assert_allclose(out, reference, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.sum(partials, axis=0), out, rtol=1e-6, atol=1e-6)
    assert len(partials) == ranks


def test_batched_sharded_mlp_is_row_independent():
    rng = np.random.default_rng(1)
    hidden, ffn, rows, ranks = 4, 8, 3, 2
    gate = rng.standard_normal((ffn, hidden), dtype=np.float32)
    up = rng.standard_normal((ffn, hidden), dtype=np.float32)
    down = rng.standard_normal((hidden, ffn), dtype=np.float32)
    x = rng.standard_normal((rows, hidden), dtype=np.float32)
    base, _ = batched_sharded_mlp_reference(gate, up, down, x, num_ranks=ranks)
    perturbed = x.copy()
    perturbed[1] += 10.0
    changed, _ = batched_sharded_mlp_reference(gate, up, down, perturbed, num_ranks=ranks)
    np.testing.assert_array_equal(base[0], changed[0])
    np.testing.assert_array_equal(base[2], changed[2])
    assert not np.array_equal(base[1], changed[1])


def test_plan_layer_ids_and_types_are_complete_and_ordered():
    plan = plan_batched_prefill(Config(), num_ranks=2, chunk_rows=32)
    assert [layer.layer_id for layer in plan.layers] == list(range(64))
    assert [layer.layer_type for layer in plan.layers] == list(Config().layer_types)
    assert plan.chunk_ranges(64) == chunk_ranges(64, 32)
