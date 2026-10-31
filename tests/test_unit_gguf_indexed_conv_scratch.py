"""Packed indexed Conv must not overwrite its live BF16 projection input."""
from __future__ import annotations

import pytest

from hipengine.runtime.qwen35_gguf_prefill_plan import (
    _GGUF_PREFILL_SCRATCH_COMPACT_DISABLED_FIELDS,
    build_prefill_scratch_plan,
)
from tests.test_unit_qwen35_gguf_prefill_scratch_liveness import _fake_runner


def _overlap(left, right):
    owner_a, offset_a, size_a = left
    owner_b, offset_b, size_b = right
    return (owner_a == owner_b and offset_a < offset_b + size_b
            and offset_b < offset_a + size_a)


@pytest.mark.parametrize("grouping", ["single", "owner_slots"])
@pytest.mark.parametrize("rows", [2, 8, 128, 768])
def test_indexed_conv_input_and_output_have_disjoint_storage(grouping, rows):
    runner = _fake_runner("hip_gfx1151")
    plan = build_prefill_scratch_plan(
        runner.weights.config,
        hidden_size=runner.hidden_size,
        ffn_size=runner.ffn_size,
        q_width=runner.q_width,
        kv_width=runner.kv_width,
        linear_qkv_width=runner.linear_qkv_width,
        ssm_value_dim=runner.ssm_value_dim,
        rows=rows,
        capacity=rows,
        segments=2,
        allocate_kv_cache=False,
        normalized_qk_heads=0,
        f16_policy=None,
        disabled_fields=_GGUF_PREFILL_SCRATCH_COMPACT_DISABLED_FIELDS,
        arena_grouping=grouping,
    )
    # The indexed Conv reads BF16 QKV and writes FP32 output in the same
    # launch: even matching element indices use different byte offsets.
    assert plan.field_sizes["conv_out"] == 2 * plan.field_sizes["linear_qkv"]
    assert not _overlap(plan.views["linear_qkv"], plan.views["conv_out"])
    # The prefill FP32 input must likewise remain live throughout Conv.
    assert not _overlap(plan.views["linear_qkv_f32"], plan.views["conv_out"])
