from __future__ import annotations

import numpy as np

from scripts.qwen35_paro_kv_format_ablation import (
    FormatSpec,
    _distribution_summary,
    _format_memory_bytes,
    _parse_candidates,
    _quantize_dequantize,
    _reconstruction_summary,
    _select_recommendation,
)


def test_quantize_dequantize_groupwise_reduces_outlier_collateral_error() -> None:
    values = np.asarray([[[100.0, 1.0, -1.0, 0.5, 0.25, -0.5, 0.75, -0.25]]], dtype=np.float32)

    per_head = _quantize_dequantize(values, group_size=8, clip_ratio=1.0, scale_dtype="fp16")
    group4 = _quantize_dequantize(values, group_size=4, clip_ratio=1.0, scale_dtype="fp16")

    assert per_head.shape == values.shape
    assert group4.shape == values.shape
    assert np.mean(np.square(group4[..., 4:] - values[..., 4:])) < np.mean(
        np.square(per_head[..., 4:] - values[..., 4:])
    )


def test_quantize_dequantize_uses_float_scale_for_codes_and_rounded_scale_for_restore() -> None:
    values = np.asarray([[[1.0, -0.25, 0.125, 0.0]]], dtype=np.float32)

    restored = _quantize_dequantize(values, group_size=4, clip_ratio=1.0, scale_dtype="fp16")
    float_scale = np.max(np.abs(values), axis=-1, keepdims=True) / np.float32(127.0)
    codes = np.clip(np.rint(values / float_scale), -127.0, 127.0)
    expected = codes * float_scale.astype(np.float16).astype(np.float32)

    np.testing.assert_array_equal(restored, expected)


def test_distribution_and_reconstruction_summaries_are_finite() -> None:
    values = np.asarray([[[0.0, 1.0, -2.0, 4.0], [0.25, -0.5, 1.0, -1.0]]], dtype=np.float32)
    restored = _quantize_dequantize(values, group_size=2, clip_ratio=0.99, scale_dtype="fp16")

    distribution = _distribution_summary(values)
    reconstruction = _reconstruction_summary(values, restored)

    assert distribution["elements"] == values.size
    assert distribution["abs_percentiles"]["100"] == 4.0
    assert distribution["row_absmax_percentiles"]["100"] == 4.0
    assert np.isfinite(reconstruction["rmse"])
    assert np.isfinite(reconstruction["normalized_rmse"])
    assert reconstruction["max_abs_error"] > 0.0


def test_format_memory_estimate_accounts_for_group_scales_and_mixed_value() -> None:
    baseline = FormatSpec("baseline", k_mode="int8", v_mode="int8", k_group_size=256, v_group_size=256)
    group16 = FormatSpec("group16", k_mode="int8", v_mode="int8", k_group_size=16, v_group_size=16)
    key_only = FormatSpec("key_only", k_mode="int8", v_mode="bf16", k_group_size=256, v_group_size=256)

    baseline_bytes = _format_memory_bytes(
        baseline,
        tokens=1024,
        full_layers=10,
        num_kv_heads=2,
        head_dim=256,
        scale_dtype="fp16",
    )
    group16_bytes = _format_memory_bytes(
        group16,
        tokens=1024,
        full_layers=10,
        num_kv_heads=2,
        head_dim=256,
        scale_dtype="fp16",
    )
    key_only_bytes = _format_memory_bytes(
        key_only,
        tokens=1024,
        full_layers=10,
        num_kv_heads=2,
        head_dim=256,
        scale_dtype="fp16",
    )

    assert baseline_bytes["payload_bytes"] == 2 * 1024 * 10 * 2 * 256
    assert baseline_bytes["scale_bytes"] == 2 * 1024 * 10 * 2 * 2
    assert group16_bytes["scale_bytes"] == 16 * baseline_bytes["scale_bytes"]
    assert key_only_bytes["payload_bytes"] > baseline_bytes["payload_bytes"]
    assert key_only_bytes["scale_bytes"] == baseline_bytes["scale_bytes"] // 2


def test_parse_candidates_and_recommendation_respect_budget() -> None:
    candidates = _parse_candidates("baseline_max,group32,key_int8_value_bf16", head_dim=256)
    assert [candidate.name for candidate in candidates] == [
        "baseline_max",
        "group32",
        "key_int8_value_bf16",
    ]

    rows = [
        {"name": "baseline_max", "extra_bytes_over_baseline": 0, "logit_gate": {"mean_kl": 0.5, "top1_agreement": 0.5}},
        {"name": "group32", "extra_bytes_over_baseline": 100, "logit_gate": {"mean_kl": 0.1, "top1_agreement": 0.8}},
        {"name": "key_int8_value_bf16", "extra_bytes_over_baseline": 1000, "logit_gate": {"mean_kl": 0.01, "top1_agreement": 1.0}},
    ]
    recommendation = _select_recommendation(rows, extra_budget_bytes=200)

    assert recommendation["name"] == "group32"
    assert recommendation["fit_candidates"] == ["baseline_max", "group32"]
