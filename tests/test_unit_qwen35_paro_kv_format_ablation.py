from __future__ import annotations

import numpy as np

from scripts import qwen35_paro_kv_format_ablation as ablation
from scripts.qwen35_paro_kv_format_ablation import (
    FormatSpec,
    _aggregate_reconstruction,
    _distribution_summary,
    _format_memory_bytes,
    _format_quantizes_layer,
    _fp8_e4m3fn_quantize_dequantize,
    _normalized_hadamard,
    _parse_candidates,
    _quantize_dequantize,
    _reconstruction_summary,
    _roundtrip_pair,
    _select_recommendation,
    _variance_normalize,
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


def test_fp8_e4m3fn_roundtrip_handles_ties_subnormals_and_saturation() -> None:
    values = np.asarray(
        [0.0, -0.0, 2**-9, 1.0, 1.0625, 1.1875, 448.0, 500.0, -500.0],
        dtype=np.float32,
    )

    restored = _fp8_e4m3fn_quantize_dequantize(values)

    expected = np.asarray(
        [0.0, -0.0, 2**-9, 1.0, 1.0, 1.25, 448.0, 448.0, -448.0],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(restored, expected)
    assert np.signbit(restored[1])


def test_normalized_hadamard_is_an_involution() -> None:
    values = np.arange(32, dtype=np.float32).reshape(2, 2, 8) / 7.0

    rotated = _normalized_hadamard(values, group_size=8)
    restored = _normalized_hadamard(rotated, group_size=8)

    np.testing.assert_allclose(restored, values, rtol=1e-6, atol=1e-6)
    assert not np.array_equal(rotated, values)


def test_kivi_roundtrip_keeps_incomplete_chunk_unquantized() -> None:
    key = np.linspace(-3.0, 5.0, 40, dtype=np.float32).reshape(5, 1, 8)
    value = np.square(key, dtype=np.float32) - 1.25
    spec = FormatSpec(
        "kivi_test",
        strategy="kivi",
        v_group_size=4,
        chunk_size=4,
        residual_tokens=3,
    )

    key_out, value_out = _roundtrip_pair(key, value, spec, scale_dtype="fp16")

    assert np.any(key_out[:4] != key[:4])
    assert np.any(value_out[:4] != value[:4])
    np.testing.assert_array_equal(key_out[4:], key[4:])
    np.testing.assert_array_equal(value_out[4:], value[4:])


def test_variance_normalize_reconstructs_and_reduces_axis_imbalance() -> None:
    values = np.asarray(
        [
            [1.0, 2.0, 4.0, 8.0],
            [0.1, 0.4, 1.6, 6.4],
            [3.0, 3.5, 4.0, 4.5],
            [-8.0, -2.0, -0.5, -0.125],
        ],
        dtype=np.float32,
    )

    balanced, column_scale, row_scale = _variance_normalize(values, iterations=8)
    reconstructed = balanced * column_scale * row_scale

    np.testing.assert_allclose(reconstructed, values, rtol=2e-6, atol=2e-6)
    initial_spread = np.std(values, axis=0).max() / np.std(values, axis=0).min()
    balanced_spread = np.std(balanced, axis=0).max() / np.std(balanced, axis=0).min()
    assert balanced_spread < initial_spread


def test_kvarn_roundtrip_preserves_permanent_sink_tokens() -> None:
    rng = np.random.default_rng(9)
    key = rng.normal(size=(6, 1, 8)).astype(np.float32)
    value = rng.normal(size=(6, 1, 8)).astype(np.float32)
    spec = FormatSpec(
        "kvarn_sink_test",
        strategy="kvarn",
        chunk_size=2,
        residual_tokens=1,
        sink_tokens=2,
        hadamard_group_size=8,
    )

    key_out, value_out = _roundtrip_pair(key, value, spec, scale_dtype="fp16")

    np.testing.assert_array_equal(key_out[:2], key[:2])
    np.testing.assert_array_equal(value_out[:2], value[:2])
    assert np.any(key_out[2:] != key[2:])
    assert np.any(value_out[2:] != value[2:])


def test_kvarn_roundtrip_is_finite_deterministic_and_zero_safe() -> None:
    rng = np.random.default_rng(7)
    key = rng.normal(size=(8, 2, 8)).astype(np.float32)
    value = np.zeros_like(key)
    spec = FormatSpec(
        "kvarn_test",
        strategy="kvarn",
        chunk_size=4,
        residual_tokens=3,
        hadamard_group_size=8,
    )

    first = _roundtrip_pair(key, value, spec, scale_dtype="fp16")
    second = _roundtrip_pair(key, value, spec, scale_dtype="fp16")

    assert all(np.isfinite(item).all() for item in first)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], value)


def test_run_loaded_session_resets_and_reuses_resident_weights(monkeypatch) -> None:
    class FakeResult:
        def __init__(self, token_id: int) -> None:
            self.token_id = token_id

    class FakeSession:
        full_caches: dict[int, object] = {}

        def __init__(self) -> None:
            self.reset_count = 0
            self.positions: list[int] = []

        def reset(self) -> None:
            self.reset_count += 1

        def _resolve_prefill_config_for_length(self, prompt_length: int) -> None:
            assert prompt_length == 3

        def prefill_native(self, prompt_tokens, *, sample: bool):
            assert prompt_tokens == [1, 2, 3]
            assert sample
            return FakeResult(7)

        def step(self, token_id: int, *, position: int, sample: bool):
            assert sample
            self.positions.append(position)
            return FakeResult(token_id + 100)

    session = FakeSession()
    monkeypatch.setattr(ablation, "_read_logits", lambda _session: np.asarray([0.0, 1.0], dtype=np.float32))

    result, captured, full_layers = ablation._run_loaded_session(
        session,
        prompt_tokens=[1, 2, 3],
        prompt_length=3,
        decode_steps=2,
        forced_input_ids=[7, 8],
        scale_dtype="fp16",
    )

    assert session.reset_count == 1
    assert session.positions == [3, 4]
    assert result["generated_token_ids"] == [107, 108]
    assert result["elapsed_seconds"] >= 0.0
    assert captured is None
    assert full_layers == 0


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


def test_tail_four_mixed_formats_hit_the_mild_256k_memory_target() -> None:
    catalog = ablation._candidate_catalog(256)
    fp8 = catalog["tail4_fp8_e4m3"]
    per_head = catalog["tail4_int8_per_head"]
    group32 = catalog["tail4_group32"]
    kwargs = {
        "tokens": 262144,
        "full_layers": 10,
        "num_kv_heads": 2,
        "head_dim": 256,
        "scale_dtype": "fp16",
    }

    assert [_format_quantizes_layer(fp8, index, 10) for index in range(10)] == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    assert _format_memory_bytes(fp8, **kwargs)["total_bytes"] == 4 * 1024**3
    assert _format_memory_bytes(per_head, **kwargs)["total_bytes"] == int(4.0078125 * 1024**3)
    assert _format_memory_bytes(group32, **kwargs)["total_bytes"] == int(4.0625 * 1024**3)


def test_tail_four_reconstruction_preserves_the_first_six_layers() -> None:
    keys = [np.full((2, 1, 8), index + 0.125, dtype=np.float32) for index in range(10)]
    values = [np.full((2, 1, 8), index + 0.25, dtype=np.float32) for index in range(10)]
    spec = FormatSpec(
        "tail4",
        k_group_size=8,
        v_group_size=8,
        quantized_tail_layers=4,
    )

    summary = _aggregate_reconstruction(keys, values, spec, scale_dtype="fp16")

    assert summary["preserved_bf16_layers"] == 6
    assert summary["quantized_layers"] == 4
    assert summary["key"]["normalized_rmse"] > 0.0
    assert summary["value"]["normalized_rmse"] > 0.0


def test_parse_candidates_and_recommendation_respect_budget() -> None:
    candidates = _parse_candidates(
        "baseline_max,group32,hadamard_group32,kivi_int8,kvarn_int8,key_int8_value_bf16,"
        "key_fp8_e4m3_value_bf16,tail4_fp8_e4m3,tail4_int8_per_head,tail4_group32",
        head_dim=256,
    )
    assert [candidate.name for candidate in candidates] == [
        "baseline_max",
        "group32",
        "hadamard_group32",
        "kivi_int8",
        "kvarn_int8",
        "key_int8_value_bf16",
        "key_fp8_e4m3_value_bf16",
        "tail4_fp8_e4m3",
        "tail4_int8_per_head",
        "tail4_group32",
    ]

    rows = [
        {"name": "baseline_max", "extra_bytes_over_baseline": 0, "logit_gate": {"mean_kl": 0.5, "top1_agreement": 0.5}},
        {"name": "group32", "extra_bytes_over_baseline": 100, "logit_gate": {"mean_kl": 0.1, "top1_agreement": 0.8}},
        {"name": "key_int8_value_bf16", "extra_bytes_over_baseline": 1000, "logit_gate": {"mean_kl": 0.01, "top1_agreement": 1.0}},
    ]
    recommendation = _select_recommendation(rows, extra_budget_bytes=200)

    assert recommendation["name"] == "group32"
    assert recommendation["fit_candidates"] == ["baseline_max", "group32"]
