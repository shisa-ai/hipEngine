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


def _fake_request(table: dict[str, float], fail_prompt: str | None = None, fail_text: str = ""):
    """Stand-in for the HTTP call: prompt text -> wall, optionally raising on one prompt."""

    def request(prompt: str) -> dict:
        if prompt == fail_prompt:
            raise RuntimeError(fail_text)
        return {"wall_seconds": table[prompt], "prompt_tokens": int(prompt[1:])}

    return request


def test_length_scan_stops_at_the_context_ceiling(capsys) -> None:
    # A 400 context_length_exceeded is the sweep running out of context, not a broken probe:
    # it used to traceback after nine good measurements and lose the whole scan.
    table = {"p1": 0.30, "p2": 0.26, "p4": 0.27}
    rows = probe.run_length_scan(
        _fake_request(
            table,
            fail_prompt="p8",
            fail_text='HTTP 400: {"error":{"code":"context_length_exceeded"}}',
        ),
        lambda target: f"p{target}",
        "1,2,4,8,16",
        reps=1,
        max_sequence_length=1024,
    )
    assert [row["prompt_tokens"] for row in rows] == [1, 2, 4]
    out = capsys.readouterr().out
    assert "skipped (exceeds --max-sequence-length 1024)" in out
    assert "length 16" not in out  # stops rather than pressing on past the ceiling


def test_length_scan_reraises_any_other_http_failure() -> None:
    with pytest.raises(RuntimeError, match="HTTP 500"):
        probe.run_length_scan(
            _fake_request({}, fail_prompt="p1", fail_text="HTTP 500: boom"),
            lambda target: f"p{target}",
            "1",
            reps=1,
            max_sequence_length=1024,
        )


def test_length_scan_carries_the_median_and_shows_the_spread(capsys) -> None:
    rows = probe.run_length_scan(
        _fake_request({"p1": 0.300, "p2": 0.260}),
        lambda target: f"p{target}",
        "1",
        reps=3,
        max_sequence_length=1024,
    )
    assert len(rows) == 1
    assert rows[0]["samples"] == [0.300, 0.300, 0.300]
    out = capsys.readouterr().out
    assert "median_of=3" in out and "wall=0.3000s" in out


def test_fixed_marginal_fit_still_consumes_the_scan_rows() -> None:
    # Walls on an exact line: 296 ms fixed + 4 ms/token. All three stay inside the 1.5x
    # small-row regime, so the fit must consume them (a wider last wall is refused on purpose,
    # which the refusal tests already pin).
    rows = probe.run_length_scan(
        _fake_request({"p1": 0.3000, "p2": 0.3040, "p4": 0.3120}),
        lambda target: f"p{target}",
        "1,2,4",
        reps=1,
        max_sequence_length=1024,
    )
    fit = probe.fixed_marginal_fit(rows)
    assert fit is not None
    fixed_ms, per_token_ms, r2, used = fit
    assert used == 3 and r2 > 0.999
    assert abs(fixed_ms - 296.0) < 1.0
    assert abs(per_token_ms - 4.0) < 0.1
