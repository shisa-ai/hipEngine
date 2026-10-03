"""CPU contracts for identical-input attention numerical diagnostics."""
import numpy as np
from scripts.gemma4_attention_input_diagnostic import (
    bf16_to_f64, f64_to_bf16, output_comparison, oracle_samples,
)


def test_bf16_conversion_rounds_ties_to_even():
    values = np.array([1.0, -1.0, 2.0, 1.00390625, 1.01171875])
    bits = f64_to_bf16(values)
    assert bits.tolist() == [0x3f80, 0xbf80, 0x4000, 0x3f80, 0x3f82]
    assert bf16_to_f64(bits[:3]).tolist() == [1.0, -1.0, 2.0]


def test_output_comparison_keeps_bitwise_and_arithmetic_contracts_separate():
    b = f64_to_bf16([1.0, 0.0, -2.0])
    c = f64_to_bf16([1.0078125, 0.0, -2.0])
    summary = output_comparison(b, c)
    assert summary['differing_elements'] == 1
    assert summary['max_abs_diff'] == 0.0078125
    assert summary['bitwise_equal'] is False
    assert output_comparison(b, b)['bitwise_equal'] is True


def test_float64_oracle_respects_keep_mask_and_grouped_query_mapping():
    query = f64_to_bf16(np.zeros((2, 4, 2)))
    key = f64_to_bf16(np.zeros((3, 2, 2)))
    value = f64_to_bf16(np.array([[[1, 3], [10, 30]],
                                [[3, 5], [30, 50]],
                                [[5, 7], [50, 70]]]))
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    expected = f64_to_bf16(np.array([[[3, 5], [3, 5], [30, 50], [30, 50]],
                                  [[3, 5], [3, 5], [30, 50], [30, 50]]]))
    samples = oracle_samples(query, key, value, mask, expected, expected,
                             scale=1.0, rows=(0, 1), heads=(0, 1, 2, 3))
    assert len(samples) == 8
    assert all(s['strict_bf16_reference_mismatches'] == 0 for s in samples)
    assert all(s['wmma_bf16_reference_mismatches'] == 0 for s in samples)


def test_float64_conversion_avoids_float32_double_rounding_at_bf16_tie():
    boundary = 1.00390625
    values = np.array([boundary - 2e-9, boundary + 2e-9])
    assert f64_to_bf16(values).tolist() == [0x3f80, 0x3f81]


def test_float64_oracle_applies_query_scale_before_softmax():
    q = f64_to_bf16(np.array([[[1.0]]]))
    k = f64_to_bf16(np.array([[[0.0]], [[2.0]]]))
    v = f64_to_bf16(np.array([[[0.0]], [[1.0]]]))
    out = f64_to_bf16(np.array([[[1 / (1 + np.exp(-1.0))]]]))
    samples = oracle_samples(q, k, v, np.ones((1, 2), np.uint8), out, out,
                             scale=0.5, rows=(0,), heads=(0,))
    assert samples[0]['strict_bf16_reference_mismatches'] == 0


def test_output_comparison_rejects_nonfinite_data():
    import pytest
    with pytest.raises(ValueError, match='nonfinite'):
        output_comparison(np.array([0x7f80], np.uint16), np.array([0x7f80], np.uint16))
