"""Teacher-forced evaluator gate: math, thresholds, schema, and chain wiring.

The evaluator freezes the incumbent path's full-vocabulary logits on the
frozen prompt chain, then gates changed-arithmetic candidates against it with
the production KL/top-1 limits from ``docs/EXECUTION-PROFILES.md``. Only the
math, schema, and chain construction are tested here on the CPU tier; the GPU
capture path is exercised by the freeze run itself, which captures the chain
twice and requires a bit-identical (KL == 0, top-1 100%) self-gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.gemma4_teacher_forced_gate import (
    MARGIN_BANDS,
    THRESHOLDS,
    capture_chain,
    force_slices,
    evaluate,
    margin_report,
    row_kl_divergence,
    row_top2_margin,
)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class _FakeRunner:
    """Records the token each forward consumes and returns canned logits."""

    def __init__(self, rows: list[np.ndarray]) -> None:
        self._rows = list(rows)
        self.calls: list[list[int]] = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def forward(self, tokens):  # noqa: ANN001 - mirrors the real signature
        self.calls.append(list(tokens))
        if not self._rows:
            raise AssertionError("fake runner exhausted")
        return self._rows.pop(0)


class TestRowKL:
    def test_identical_rows_have_zero_divergence(self) -> None:
        row = np.array([0.1, -2.0, 3.5, 0.0], dtype=np.float32)
        assert row_kl_divergence(row, row) == pytest.approx(0.0, abs=1e-12)

    def test_known_two_point_divergence_matches_closed_form(self) -> None:
        # KL(p || q) for p = softmax([0, a]), q = softmax([0, b]):
        base = np.array([0.0, 1.0], dtype=np.float32)
        cand = np.array([0.0, 2.0], dtype=np.float32)
        p = _softmax(base.astype(np.float64))
        q = _softmax(cand.astype(np.float64))
        expected = float(np.sum(p * (np.log(p) - np.log(q))))
        assert row_kl_divergence(base, cand) == pytest.approx(expected, rel=1e-9)

    def test_divergence_is_not_symmetric_and_reports_base_to_candidate(self) -> None:
        base = np.array([0.0, 1.0], dtype=np.float32)
        cand = np.array([0.0, 2.0], dtype=np.float32)
        forward = row_kl_divergence(base, cand)
        reverse = row_kl_divergence(cand, base)
        assert forward != pytest.approx(reverse)
        assert forward > 0 and reverse > 0


class TestEvaluate:
    def _pair(self, n_rows: int = 8, vocab: int = 16):
        rng = np.random.default_rng(7)
        base = rng.normal(size=(n_rows, vocab)).astype(np.float32)
        return base, base.copy()

    def test_empty_logits_are_rejected(self):
        with pytest.raises(ValueError, match="nonempty"):
            evaluate(np.empty((0, 4)), np.empty((0, 4)))

    def test_screen_requires_zero_flips_even_with_99_percent_agreement(self):
        base = np.tile([0.0, 0.00001], (100, 1))
        candidate = base.copy()
        candidate[0] = candidate[0, ::-1]
        verdict = evaluate(base, candidate)
        assert verdict["top1_rate"] == 0.99
        assert not verdict["passed"]
        assert "screen_top1_flips" in verdict["failed"]

    def test_identical_capture_passes_every_threshold(self) -> None:
        base, same = self._pair()
        verdict = evaluate(base, same)
        assert verdict["passed"] is True
        assert verdict["kl_mean"] == pytest.approx(0.0, abs=1e-12)
        assert verdict["top1_rate"] == pytest.approx(1.0)
        assert verdict["top1_flips"] == 0
        for key in ("kl_mean", "kl_p95", "kl_p99", "kl_max"):
            assert verdict[key] <= THRESHOLDS[key]

    def test_argmax_flip_fails_the_top1_bar(self) -> None:
        base, cand = self._pair()
        # Move row 0's argmax decisively to another index.
        flipped = base.copy()
        flipped[0] = np.roll(flipped[0], 1)
        flipped[0, 0] = flipped[0, 0] + 50.0
        verdict = evaluate(base, flipped)
        assert verdict["top1_flips"] == 1
        assert verdict["top1_rate"] == pytest.approx(7 / 8)
        assert verdict["passed"] is False
        assert "top1_rate" in verdict["failed"]

    def test_large_mean_kl_fails_the_mean_bar(self) -> None:
        rng = np.random.default_rng(11)
        base = rng.normal(size=(64, 32)).astype(np.float32)
        cand = base + rng.normal(scale=2.0, size=base.shape).astype(np.float32)
        verdict = evaluate(base, cand)
        assert verdict["kl_mean"] > THRESHOLDS["kl_mean"]
        assert verdict["passed"] is False
        assert "kl_mean" in verdict["failed"]

    def test_shape_or_vocabulary_mismatch_is_rejected(self) -> None:
        base, _ = self._pair(n_rows=4, vocab=16)
        short = base[:3]
        narrow = base[:, :8]
        with pytest.raises(ValueError, match="row"):
            evaluate(base, short)
        with pytest.raises(ValueError, match="vocab"):
            evaluate(base, narrow)

    def test_non_finite_logits_are_rejected(self) -> None:
        base, cand = self._pair()
        cand = cand.copy()
        cand[2, 5] = np.nan
        with pytest.raises(ValueError, match="finite"):
            evaluate(base, cand)


class TestMarginReport:
    """The margin breakdown that makes a kl_max breach readable.

    The frozen chain is eight sentences cycled, so its rows are near one-hot and
    the top-1 bar cannot see a divergence that only matters where the decision is
    close. These tests pin the measure and the banding, not any threshold.
    """

    def _pair(self) -> tuple[np.ndarray, np.ndarray]:
        base = np.zeros((4, 8), dtype=np.float32)
        base[0, 3], base[0, 5] = 2.0, 1.9       # two close leaders
        base[1, 2], base[1, 4] = 9.0, 0.0       # dominant leader
        base[2, 1], base[2, 6] = 1.0, 0.5       # moderate
        return base, base.copy()

    def test_margin_is_the_top_two_probability_gap(self) -> None:
        base, _ = self._pair()
        margin, gap = row_top2_margin(base)
        assert margin.shape == (4,)
        # A flat row has two equal leaders, so the gap is exactly zero.
        assert margin[3] == pytest.approx(0.0, abs=1e-12)
        # A dominant leader is near one.
        assert margin[1] > 0.99
        # Row 0's logits differ by 0.1 over six zeros: p1 - p2 is a few percent.
        assert margin[0] == pytest.approx(0.0351, abs=2e-3)
        assert gap[0] == pytest.approx(0.1, abs=1e-6)
        assert gap[1] == pytest.approx(9.0, abs=1e-6)

    def test_bands_partition_the_rows_in_ascending_margin_order(self) -> None:
        base, _ = self._pair()
        verdict = evaluate(base, base)
        bands = verdict["margin_report"]["bands"]
        assert [b["band"] for b in bands] == [name for name, _ in MARGIN_BANDS]
        assert sum(b["rows"] for b in bands) == base.shape[0]
        by_name = {b["band"]: b for b in bands}
        assert by_name["margin_lt_0.01"]["rows"] == 1        # the flat row
        assert by_name["margin_0.01_to_0.05"]["rows"] == 1   # the close-leader row
        assert by_name["margin_0.05_to_0.20"]["rows"] == 1  # moderate gap
        assert by_name["margin_ge_0.20"]["rows"] == 1      # the dominant row

    def test_a_flip_at_a_close_margin_is_separated_from_a_decisive_one(self) -> None:
        base, _ = self._pair()
        cand = base.copy()
        cand[1] = np.roll(cand[1], 1)        # flips the decisive row
        cand[1, 0] = cand[1, 0] + 50.0
        # A tie is the most fragile decision there is: an epsilon moves it.
        cand[3, 4] = 1e-6
        verdict = evaluate(base, cand)
        assert verdict["top1_flips"] == 2
        assert verdict["top1_flips_close_margin"] == 1
        by_name = {b["band"]: b for b in verdict["margin_report"]["bands"]}
        assert by_name["margin_ge_0.20"]["flips"] == 1
        assert by_name["margin_lt_0.01"]["flips"] == 1
        assert by_name["margin_lt_0.01"]["kl_max"] < 1e-9

    def test_report_is_present_even_when_nothing_diverges(self) -> None:
        base, same = self._pair()
        report = margin_report(base, same, np.zeros(4), [])
        assert report["close_margin_flips"] == 0
        assert report["close_margin_rows"] == 2
        assert report["margin_median"] > 0.0

    def test_an_empty_band_reports_no_kl_rather_than_nan(self) -> None:
        base = np.zeros((2, 4), dtype=np.float32)
        base[:, 0] = 20.0                    # every row decisive
        verdict = evaluate(base, base)
        by_name = {b["band"]: b for b in verdict["margin_report"]["bands"]}
        assert by_name["margin_lt_0.01"]["rows"] == 0
        assert by_name["margin_lt_0.01"]["kl_mean"] is None
        assert by_name["margin_lt_0.01"]["kl_max"] is None


class TestCaptureChain:
    def test_rows_are_snapshots_of_reused_runner_storage(self):
        class ReusingRunner:
            def reset(self):
                self.row = np.zeros(2, dtype=np.float32)

            def forward(self, tokens):
                self.row[:] = tokens[0]
                return self.row

        got = capture_chain(ReusingRunner(), [1, 2, 3, 4])
        np.testing.assert_array_equal(got[:, 0], [1, 2, 3])

    def test_chain_teacher_forces_the_prompt_ids_in_order(self) -> None:
        prompt = [9, 8, 7, 6]
        rows = [np.zeros(4, dtype=np.float32) for _ in range(len(prompt) - 1)]
        runner = _FakeRunner(rows)
        got = capture_chain(runner, prompt)
        assert runner.reset_calls == 1
        # Row t is the distribution after consuming prompt[t], predicting t+1.
        assert runner.calls == [[9], [8], [7]]
        assert got.shape == (len(prompt) - 1, 4)

    def test_chain_rejects_a_prompt_too_short_to_evaluate(self) -> None:
        runner = _FakeRunner([])
        with pytest.raises(ValueError, match="prompt"):
            capture_chain(runner, [1])

    def test_chain_stores_float32_rows(self) -> None:
        rows = [np.array([0.5, -0.5], dtype=np.float32) for _ in range(3)]
        runner = _FakeRunner(rows)
        got = capture_chain(runner, [1, 2, 3, 4])
        assert got.dtype == np.float32
        assert np.allclose(got[0], [0.5, -0.5])

    def test_prefill_primes_the_cache_in_one_forward_and_scores_the_rest(self) -> None:
        """The chain must reach the key counts the split engages at.

        Without a prefill every row runs at key counts 1..len(prompt)-1, which
        at a 1024-token prompt stops one key below the decode split's entry
        threshold - so the gate silently compared the single-kernel path with
        itself and reported kl_max of exactly 0.0. Priming the cache in one
        forward is what makes the scored rows sit above the threshold.
        """

        prompt = [9, 8, 7, 6, 5, 4]
        rows = [np.zeros(3, dtype=np.float32) for _ in range(3)]
        runner = _FakeRunner(rows)
        got = capture_chain(runner, prompt, prefill=3)
        assert runner.reset_calls == 1
        # One priming forward, then one forward per scored position: 3 and 4.
        assert runner.calls == [[9, 8, 7], [6], [5]]
        assert got.shape == (len(prompt) - 1 - 3, 3)

    def test_prefill_must_leave_at_least_one_scored_row(self) -> None:
        runner = _FakeRunner([])
        for bad in (-1, 3):
            with pytest.raises(ValueError, match="prefill"):
                capture_chain(runner, [1, 2, 3, 4], prefill=bad)

    def test_force_slices_matches_the_shipped_signature(self) -> None:
        """The patch must accept what the callers pass.

        The launchers call ``decode_slices(keys, head_dim)``. A one-argument
        patch satisfies a test that calls it with one argument and then fails
        every real launch with a TypeError - which is exactly what happened, and
        what this test now pins.
        """

        from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention

        original = gemma4_attention.decode_slices
        try:
            force_slices(2)
            # Two positional arguments, as the real callers use.
            assert gemma4_attention.decode_slices(8192, 256) == 2
            assert gemma4_attention.decode_slices(512, 256) == 2
            force_slices(1)
            assert gemma4_attention.decode_slices(8192, 512) == 1
        finally:
            gemma4_attention.decode_slices = original

    def test_force_slices_pins_the_policy_and_one_selects_the_strict_path(self) -> None:
        from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention

        original = gemma4_attention.decode_slices
        try:
            force_slices(1)
            assert gemma4_attention.decode_slices(8192, 256) == 1
            force_slices(4)
            assert gemma4_attention.decode_slices(8192, 256) == 4
            force_slices(None)
            assert gemma4_attention.decode_slices is original
        finally:
            gemma4_attention.decode_slices = original