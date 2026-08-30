"""Tests for the prefill fixed-vs-marginal fit used by the first-token probe.

The fitted fixed term is what the small-row prefill work quotes (~250 ms per prefill on
W7900 / Qwen3.8-27B Q4_K_M), so the regime selection and the refusal cases are pinned here
rather than eyeballed from a log line.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gguf_first_token_probe.py"

spec = importlib.util.spec_from_file_location("gguf_first_token_probe", SCRIPT)
assert spec and spec.loader
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def rows(pair_list):
    return [{"prompt_tokens": t, "wall_seconds": w} for t, w in pair_list]


def test_recovers_a_known_fixed_and_slope():
    # 0.25 s fixed plus 0.80 ms/token, exactly.
    data = rows([(t, 0.25 + 0.0008 * t) for t in (16, 32, 64, 96, 128)])
    fixed_ms, per_token_ms, r2, used = probe.fixed_marginal_fit(data)
    assert fixed_ms == pytest.approx(250.0, abs=1e-6)
    assert per_token_ms == pytest.approx(0.80, abs=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-9)
    assert used == 5


def test_long_prompts_are_excluded_because_they_bend_the_slope():
    # Fixed 0.25 s at 0.8 ms/token up to 128 tokens, then the compute regime doubles the
    # slope. Fitting everything would report a slope near neither regime.
    data = rows([(t, 0.25 + 0.0008 * t) for t in (16, 32, 64, 128)]
                + [(512, 0.25 + 0.0008 * 128 + 0.0016 * 384)])
    small = probe.fixed_marginal_fit(data)
    everything = probe.fixed_marginal_fit(data, regime_factor=99.0)
    assert small[1] == pytest.approx(0.80, rel=1e-6)
    assert small[3] == 4
    assert everything[1] > small[1] * 1.2          # the bent fit is measurably different


def test_fewer_than_three_rows_is_not_fit():
    assert probe.fixed_marginal_fit(rows([(16, 0.26), (32, 0.27)])) is None
    assert probe.fixed_marginal_fit([]) is None


def test_repeated_token_count_is_not_fit():
    # Three rows but one distinct x: slope is undefined, and a returned number here would
    # be a fabrication.
    assert probe.fixed_marginal_fit(rows([(16, 0.3), (16, 0.31), (16, 0.29)])) is None


def test_missing_fields_are_dropped_not_zeroed():
    data = [{"prompt_tokens": 16, "wall_seconds": 0.26}, {"prompt_tokens": None,
            "wall_seconds": 0.27}, {"prompt_tokens": 64, "wall_seconds": 0.30},
            {"prompt_tokens": 96, "wall_seconds": 0.32}]
    fitted = probe.fixed_marginal_fit(data)
    assert fitted is not None and fitted[3] == 3


def test_measured_w7900_scan_shape_still_reports_the_fixed_term():
    # Real walls from the 2026-08-30 shipping-route scan, to keep the helper honest about
    # the magnitude it exists to report.
    data = rows([(9, 0.3041), (9, 0.2644), (13, 0.2648), (19, 0.2807), (33, 0.2913),
                 (69, 0.3206), (141, 0.3600), (285, 0.7124), (665, 1.1828)])
    fixed_ms, per_token_ms, _, used = probe.fixed_marginal_fit(data)
    assert 200.0 < fixed_ms < 320.0
    assert 0.3 < per_token_ms < 1.5
    assert used < len(data)                          # the two compute-regime rows excluded


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
