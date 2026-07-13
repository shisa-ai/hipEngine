from __future__ import annotations

import numpy as np

from scripts.qwen35_paro_kv_format_ablation import FormatSpec
from scripts.qwen35_paro_kv_policy_ablation import (
    PolicySpec,
    _apply_policy_arrays,
    _parse_index_list,
    _policy_memory_bytes,
    _select_policy_recommendation,
)


def test_policy_array_emulation_preserves_selected_layer_head_sink_and_recent_rows() -> None:
    key = np.asarray(
        [
            [[100.0, 1.0, 0.5, -0.5], [2.0, 1.0, 0.5, -0.5]],
            [[100.0, 1.0, 0.5, -0.5], [2.0, 1.0, 0.5, -0.5]],
            [[100.0, 1.0, 0.5, -0.5], [2.0, 1.0, 0.5, -0.5]],
            [[100.0, 1.0, 0.5, -0.5], [2.0, 1.0, 0.5, -0.5]],
        ],
        dtype=np.float32,
    )
    value = key / np.float32(2.0)
    format_spec = FormatSpec("per_head", k_group_size=4, v_group_size=4)
    policy = PolicySpec(
        "mixed",
        format_spec=format_spec,
        bf16_layer_indices=(1,),
        bf16_head_indices=(1,),
        sink_tokens=1,
        recent_tokens=1,
    )

    layer0_k, layer0_v = _apply_policy_arrays(
        key,
        value,
        policy,
        layer_index=0,
        active_tokens=4,
        scale_dtype="fp16",
    )
    layer1_k, layer1_v = _apply_policy_arrays(
        key,
        value,
        policy,
        layer_index=1,
        active_tokens=4,
        scale_dtype="fp16",
    )

    np.testing.assert_array_equal(layer0_k[0], key[0])
    np.testing.assert_array_equal(layer0_k[-1], key[-1])
    np.testing.assert_array_equal(layer0_k[:, 1], key[:, 1])
    assert not np.array_equal(layer0_k[1:3, 0], key[1:3, 0])
    np.testing.assert_array_equal(layer0_v[0], value[0])
    np.testing.assert_array_equal(layer1_k, key)
    np.testing.assert_array_equal(layer1_v, value)


def test_policy_memory_counts_layer_head_replacement_and_residual_side_cache() -> None:
    baseline = FormatSpec("per_head", k_group_size=256, v_group_size=256)
    policy = PolicySpec(
        "mixed",
        format_spec=baseline,
        bf16_layer_indices=(0,),
        bf16_head_indices=(1,),
        sink_tokens=64,
        recent_tokens=128,
    )

    memory = _policy_memory_bytes(
        policy,
        tokens=1024,
        full_layers=4,
        num_kv_heads=2,
        head_dim=256,
        scale_dtype="fp16",
    )

    # One layer is wholly BF16; in the other three, one head is BF16 and the
    # other keeps baseline INT8 plus a 192-row BF16 residual side cache.
    assert memory["bf16_full_layers"] == 1
    assert memory["bf16_heads_per_quantized_layer"] == 1
    assert memory["residual_rows"] == 192
    assert memory["residual_bytes"] == 192 * 3 * 1 * 256 * 2 * 2
    assert memory["total_bytes"] > memory["base_format_bytes"]


def test_parse_index_list_supports_sorted_unique_indices_and_empty() -> None:
    assert _parse_index_list("2,0,2,1") == (0, 1, 2)
    assert _parse_index_list("none") == ()


def test_policy_recommendation_prioritizes_gate_pass_then_top1_and_budget() -> None:
    rows = [
        {"name": "low_kl_bad_top1", "extra_bytes_over_baseline": 10, "quality_gate_passed": False, "logit_gate": {"mean_kl": 0.04, "top1_agreement": 0.7}},
        {"name": "top1_clean", "extra_bytes_over_baseline": 20, "quality_gate_passed": False, "logit_gate": {"mean_kl": 0.08, "top1_agreement": 1.0}},
        {"name": "passes", "extra_bytes_over_baseline": 30, "quality_gate_passed": True, "logit_gate": {"mean_kl": 0.03, "top1_agreement": 1.0}},
        {"name": "over_budget", "extra_bytes_over_baseline": 1000, "quality_gate_passed": True, "logit_gate": {"mean_kl": 0.0, "top1_agreement": 1.0}},
    ]

    recommendation = _select_policy_recommendation(rows, extra_budget_bytes=100)

    assert recommendation["name"] == "passes"
    assert recommendation["quality_gate_passed"] is True
    assert recommendation["fit_candidates"] == ["low_kl_bad_top1", "top1_clean", "passes"]
