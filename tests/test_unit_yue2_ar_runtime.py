"""AR runtime numerics that can be checked without a GPU.

The RoPE table is the one piece of the AR runtime whose arithmetic is decided on
the host; the released reference builds it from an fp32 angle, which is the
sensitivity limit for any independent implementation. These tests pin the
exponent layout (the bug the fixtures caught) and the bf16 conversion rules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.runtime.yue2_ar import (
    SPAN_ATTENTION_MAX_CONTEXT,
    bf16_bits_to_f32,
    bf16_bits_to_fp16_bits,
    f32_to_bf16_bits,
    rope_tables,
)

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/yue2"


def _reference_table():
    data = np.load(FIXTURES / "operators/rope_cos_sin.npz")
    return data["positions"], data["cos"], data["sin"]


def test_rope_exponents_follow_the_reference_layout():
    """Column k must carry exponent k/64, not k/128 (the fixture-caught bug)."""
    positions, cos, sin = _reference_table()
    implied = np.arctan2(sin[1].astype(np.float64), cos[1].astype(np.float64))
    half = 64
    expected = 1.0 / (1000000.0 ** (np.arange(0, half, dtype=np.float64) / half))
    np.testing.assert_allclose(implied, expected, rtol=2e-5)
    # A wrong denominator shifts every column by a factor of two in the exponent.
    wrong = 1.0 / (1000000.0 ** (np.arange(0, half, dtype=np.float64) / 128))
    assert np.abs(implied - wrong).max() > 0.2


@pytest.mark.parametrize("mode", ["reference", "f64"])
def test_rope_table_matches_the_reference_within_fp32_angle_sensitivity(mode):
    positions, cos, sin = _reference_table()
    table_cos, table_sin = rope_tables(int(positions[-1]) + 1, 128, 1000000.0, mode=mode)
    assert table_cos.shape == table_sin.shape == (int(positions[-1]) + 1, 128)
    # The duplicated halves must be identical (HF rotate-half layout).
    np.testing.assert_array_equal(table_cos[:, :64], table_cos[:, 64:])
    np.testing.assert_array_equal(table_sin[:, :64], table_sin[:, 64:])
    got_cos = table_cos[positions][:, :64]
    got_sin = table_sin[positions][:, :64]
    # The reference evaluates the angle in fp32; at position 2047 the top
    # frequency's own fp32 ulp is ~1e-4 rad, which is the floor for any
    # independent table.
    assert np.abs(got_cos - cos).max() <= 3e-4
    assert np.abs(got_sin - sin).max() <= 3e-4
    identical = ((got_cos == cos) & (got_sin == sin)).mean()
    assert identical > 0.2


def test_rope_table_reference_mode_is_the_closer_of_the_two():
    positions, cos, sin = _reference_table()
    reference_cos, _ = rope_tables(int(positions[-1]) + 1, 128, 1000000.0, mode="reference")
    f64_cos, _ = rope_tables(int(positions[-1]) + 1, 128, 1000000.0, mode="f64")
    reference_error = np.abs(reference_cos[positions][:, :64] - cos).max()
    f64_error = np.abs(f64_cos[positions][:, :64] - cos).max()
    assert reference_error <= f64_error


def test_rope_table_rejects_unknown_mode():
    with pytest.raises(ValueError):
        rope_tables(4, 128, 1000000.0, mode="float16")


def test_rope_table_uses_the_requested_theta():
    narrow, _ = rope_tables(8, 128, 10000.0)
    wide, _ = rope_tables(8, 128, 1000000.0)
    assert not np.array_equal(narrow, wide)
    np.testing.assert_allclose(narrow[0], 1.0)


def test_f32_to_bf16_bits_rounds_to_nearest_even():
    rng = np.random.default_rng(7)
    values = np.concatenate(
        [
            rng.standard_normal(4096).astype(np.float32) * 4.0,
            np.array([0.0, -0.0, 1.0, -1.0, 1e-8, -1e-8, 1e38, -1e38], dtype=np.float32),
        ]
    )
    got = f32_to_bf16_bits(values)
    # Reference: widen to 64 bits, add the round bit plus the even-adjustment.
    wide = values.view(np.uint32).astype(np.uint64)
    expected = (
        (wide + np.uint64(0x7FFF) + ((wide >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16)
    ).astype(np.uint16)
    np.testing.assert_array_equal(got, expected)
    # Round-trip error must stay inside one bf16 ulp of the value, and the
    # rounded value must be the nearer of its two bf16 neighbours.
    restored = bf16_bits_to_f32(got)
    error = np.abs(restored.astype(np.float64) - values.astype(np.float64))
    assert error.max() <= (np.abs(values).astype(np.float64) * 2.0**-8).max()
    for index in range(values.size):
        bits = int(got[index])
        candidates = [bits]
        if bits > 0:
            candidates.append(bits - 1)
        if bits < 0xFFFF:
            candidates.append(bits + 1)
        distances = [
            abs(float(bf16_bits_to_f32(np.uint16(candidate))) - float(values[index]))
            for candidate in candidates
        ]
        assert distances[0] <= min(distances) + 1e-12 * max(1.0, abs(float(values[index])))


def test_bf16_bits_to_fp16_bits_is_value_exact_in_range():
    rng = np.random.default_rng(11)
    values = (rng.standard_normal(2048).astype(np.float32) * 8.0).astype(np.float32)
    bits = f32_to_bf16_bits(values)
    fp16 = bf16_bits_to_fp16_bits(bits)
    assert fp16.dtype == np.uint16
    np.testing.assert_allclose(
        fp16.view(np.float16).astype(np.float32), bf16_bits_to_f32(bits), rtol=0, atol=0
    )


def test_span_attention_context_limit_is_documented():
    # The in-tree span attention kernel keeps the whole score row in shared
    # memory; the runtime must refuse a larger context rather than corrupt it.
    assert SPAN_ATTENTION_MAX_CONTEXT == 16000


def test_runtime_rejects_invalid_arguments_without_a_gpu():
    from hipengine.runtime.yue2_ar import Yue2ArRuntime

    class _Config:
        pass

    with pytest.raises(ValueError):
        Yue2ArRuntime(_Config(), max_context=SPAN_ATTENTION_MAX_CONTEXT + 1)
    with pytest.raises(ValueError):
        Yue2ArRuntime(_Config(), max_context=0)
    with pytest.raises(ValueError):
        Yue2ArRuntime(_Config(), max_context=16, branches=3)
