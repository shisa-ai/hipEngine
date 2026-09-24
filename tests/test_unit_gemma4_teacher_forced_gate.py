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
    THRESHOLDS,
    capture_chain,
    evaluate,
    row_kl_divergence,
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


class TestCaptureChain:
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