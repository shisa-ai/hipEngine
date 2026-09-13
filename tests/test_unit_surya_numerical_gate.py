"""Unit tests for the Surya numerical gate's pure math and driver.

No model, GPU, or torch: the arms are scripted logit generators, so the KL,
top-1, aggregation, and verdict logic are exercised exactly and cheaply.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.surya_numerical_gate import (
    ENVELOPE,
    ArmComparison,
    KlReport,
    PreparedPrompt,
    ProductionEnvelope,
    evaluate_envelope,
    force_chain,
    generate_chain,
    kl_divergence,
    log_softmax,
    merge_reports,
    review_rows,
    run_comparison,
    summarize,
    verdict,
)


# ---------------------------------------------------------------------------
# log-softmax and KL
# ---------------------------------------------------------------------------


def test_log_softmax_rows_are_normalized() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]])

    probs = np.exp(log_softmax(logits))

    assert np.allclose(probs.sum(axis=-1), 1.0)
    assert np.allclose(
        probs,
        np.array([[9.00305732e-02, 2.44728471e-01, 6.65240956e-01],
                  [4.50940412e-05, 6.69254912e-03, 9.93262357e-01]]),
        atol=1e-8,
    )


def test_log_softmax_is_stable_for_large_logits() -> None:
    logits = np.array([[1000.0, 1001.0, 1002.0]])

    probs = np.exp(log_softmax(logits))

    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=-1), 1.0)


def test_kl_is_zero_for_identical_distributions() -> None:
    logits = np.array([[0.5, -1.0, 2.0, 0.0]])

    kl = kl_divergence(logits, logits.copy())

    assert kl.shape == (1,)
    assert kl[0] == pytest.approx(0.0, abs=1e-15)


def test_kl_is_positive_and_asymmetric_for_different_distributions() -> None:
    # teacher is peaked on one token; candidate spreads over two.
    teacher = np.array([[2.0, 0.0, 0.0, 0.0]])
    candidate = np.array([[1.0, 1.0, 0.0, 0.0]])

    forward = kl_divergence(teacher, candidate)[0]
    backward = kl_divergence(candidate, teacher)[0]

    assert forward == pytest.approx(0.28063537313516185, rel=1e-12)
    assert backward == pytest.approx(0.33434408583496317, rel=1e-12)
    assert forward != pytest.approx(backward, rel=1e-6)


def test_kl_is_invariant_to_a_constant_shift() -> None:
    teacher = np.array([[0.1, 0.2, 0.3]])
    candidate = np.array([[0.4, -0.2, 0.9]])

    base = kl_divergence(teacher, candidate)[0]
    shifted = kl_divergence(teacher + 12.0, candidate - 7.0)[0]

    assert shifted == pytest.approx(base, rel=1e-12)


def test_kl_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        kl_divergence(np.zeros((2, 4)), np.zeros((2, 5)))


def test_kl_rejects_non_matrix_input() -> None:
    with pytest.raises(ValueError, match="rows, vocab"):
        kl_divergence(np.zeros(4), np.zeros(4))


def test_kl_matches_a_hand_computed_binary_case() -> None:
    # p = (0.5, 0.5); q = (0.9, 0.1) -> KL = 0.5*ln(0.5/0.9) + 0.5*ln(0.5/0.1)
    teacher = np.log(np.array([[0.5, 0.5]]))
    candidate = np.log(np.array([[0.9, 0.1]]))

    kl = kl_divergence(teacher, candidate)[0]

    expected = 0.5 * np.log(0.5 / 0.9) + 0.5 * np.log(0.5 / 0.1)
    assert kl == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def test_summarize_reports_mean_tails_and_top1() -> None:
    kl = np.array([0.0, 0.001, 0.002, 0.003, 0.004])
    teacher = np.array([0, 1, 2, 3, 4])
    candidate = np.array([0, 1, 2, 3, 9])  # one flip

    report = summarize(kl, teacher, candidate, scope="page")

    assert report.scope == "page"
    assert report.n_rows == 5
    assert report.mean_kl == pytest.approx(0.002)
    assert report.max_kl == pytest.approx(0.004)
    assert report.p95_kl == pytest.approx(float(np.percentile(kl, 95)))
    assert report.top1_flips == 1
    assert report.top1_agreement == pytest.approx(0.8)
    assert report.finite


def test_summarize_counts_rows_over_the_review_bar() -> None:
    kl = np.array([0.0, 0.01, 0.02, 0.03, 0.5])

    report = summarize(kl, np.zeros(5), np.zeros(5), scope="s")

    assert report.rows_over_review == 2  # strictly greater than 2e-2


def test_summarize_folds_non_finite_kl_to_infinity() -> None:
    kl = np.array([0.0, np.nan, np.inf])

    report = summarize(kl, np.zeros(3), np.zeros(3), scope="s")

    assert not report.finite
    assert np.isinf(report.max_kl)


def test_summarize_rejects_mismatched_top1_length() -> None:
    with pytest.raises(ValueError, match="one entry per row"):
        summarize(np.zeros(3), np.zeros(2), np.zeros(3), scope="s")


def test_summarize_handles_zero_rows() -> None:
    report = summarize(np.zeros(0), np.zeros(0), np.zeros(0), scope="empty")

    assert report.n_rows == 0
    assert report.top1_agreement == 1.0
    assert report.max_kl == 0.0


# ---------------------------------------------------------------------------
# envelope verdict
# ---------------------------------------------------------------------------


def _report(**overrides) -> KlReport:
    base = dict(
        scope="s", n_rows=100, mean_kl=1e-5, p95_kl=1e-4, p99_kl=1e-3,
        max_kl=1e-2, top1_agreement=1.0, top1_flips=0, rows_over_review=0,
        finite=True,
    )
    base.update(overrides)
    return KlReport(**base)


def test_envelope_passes_a_clean_report() -> None:
    passed, failures = evaluate_envelope(_report())

    assert passed, failures


@pytest.mark.parametrize(
    "override,needle",
    [
        ({"mean_kl": 2e-3}, "mean_kl"),
        ({"p95_kl": 6e-3}, "p95_kl"),
        ({"p99_kl": 3e-2}, "p99_kl"),
        ({"max_kl": 6e-2}, "max_kl"),
        ({"top1_agreement": 0.98}, "top1"),
        ({"rows_over_review": 1}, "review KL"),
        ({"finite": False}, "non-finite"),
    ],
)
def test_envelope_names_the_failing_bound(override: dict, needle: str) -> None:
    passed, failures = evaluate_envelope(_report(**override))

    assert not passed
    assert any(needle in f for f in failures)


def test_envelope_bars_are_the_documented_values() -> None:
    assert ENVELOPE.mean_kl == 1e-3
    assert ENVELOPE.p95_kl == 5e-3
    assert ENVELOPE.p99_kl == 2e-2
    assert ENVELOPE.max_kl == 5e-2
    assert ENVELOPE.top1 == 0.99
    assert ENVELOPE.per_scope_top1 == 0.97
    assert ENVELOPE.review_kl == 2e-2


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def test_merge_weights_the_mean_by_row_count() -> None:
    a = _report(scope="a", n_rows=100, mean_kl=1e-4, max_kl=1e-3)
    b = _report(scope="b", n_rows=300, mean_kl=5e-4, max_kl=9e-3)

    merged = merge_reports([a, b])

    assert merged.n_rows == 400
    assert merged.mean_kl == pytest.approx((1e-4 * 100 + 5e-4 * 300) / 400)
    assert merged.max_kl == pytest.approx(9e-3)


def test_merge_sums_flips_and_takes_the_worst_tail() -> None:
    a = _report(scope="a", n_rows=10, top1_flips=1, p99_kl=3e-3)
    b = _report(scope="b", n_rows=10, top1_flips=2, p99_kl=7e-3)

    merged = merge_reports([a, b])

    assert merged.top1_flips == 3
    assert merged.top1_agreement == pytest.approx(1 - 3 / 20)
    assert merged.p99_kl == pytest.approx(7e-3)


def test_merge_of_nothing_is_empty() -> None:
    merged = merge_reports([])

    assert merged.n_rows == 0
    assert merged.scope == "global"


# ---------------------------------------------------------------------------
# review-row diagnostics
# ---------------------------------------------------------------------------


def test_review_rows_is_empty_when_nothing_exceeds_the_bar() -> None:
    kl = np.array([0.0, 1e-3])
    logits = np.zeros((2, 4))

    assert review_rows(kl, logits, logits, scope="s") == []


def test_review_rows_reports_overlap_and_margin() -> None:
    teacher = np.array([[10.0, 1.0, 0.0, -1.0]])
    # Candidate keeps the argmax but moves mass onto the runner-up.
    candidate = np.array([[2.0, 1.9, 0.0, -1.0]])
    kl = kl_divergence(teacher, candidate)
    assert kl[0] > 2e-2

    rows = review_rows(kl, teacher, candidate, scope="page")

    assert len(rows) == 1
    row = rows[0]
    assert row.scope == "page"
    assert row.index == 0
    assert row.teacher_top1 == 0
    assert row.candidate_top1 == 0
    assert not row.flipped
    assert row.top5_overlap == 4
    assert row.teacher_logit_margin == pytest.approx(9.0)
    assert row.teacher_prob_margin > 0.0
    assert row.kl == pytest.approx(float(kl[0]))


def test_review_rows_marks_a_flip() -> None:
    teacher = np.array([[3.0, 2.0, 0.0, 0.0]])
    candidate = np.array([[0.0, 4.0, 0.0, 0.0]])
    kl = kl_divergence(teacher, candidate)

    rows = review_rows(kl, teacher, candidate, scope="s")

    assert rows and rows[0].flipped
    assert rows[0].teacher_top1 == 0
    assert rows[0].candidate_top1 == 1


def test_review_rows_orders_worst_first_and_honors_the_limit() -> None:
    teacher = np.zeros((3, 4))
    candidate = np.array([[5.0, 0.0, 0.0, 0.0],
                          [3.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0]])
    kl = kl_divergence(teacher, candidate)

    rows = review_rows(kl, teacher, candidate, scope="s", limit=1)

    assert len(rows) == 1
    assert rows[0].index == 0


def test_review_rows_tolerates_non_finite_kl() -> None:
    teacher = np.zeros((2, 4))
    candidate = np.zeros((2, 4))
    kl = np.array([np.nan, 0.5])

    rows = review_rows(kl, teacher, candidate, scope="s")

    assert [r.index for r in rows] == [0, 1]


# ---------------------------------------------------------------------------
# scripted arms
# ---------------------------------------------------------------------------


class ScriptedArm:
    """Logits depend only on ``(seed, step)``, so chains are reproducible."""

    def __init__(self, name: str, *, seed: int = 0, vocab: int = 8, scale: float = 2.0):
        self.name = name
        self.seed = seed
        self.vocab = vocab
        self.scale = scale
        self._step = 0

    def prepare(self, prompt: str, page_path) -> PreparedPrompt:
        return PreparedPrompt(payload=(prompt, page_path), first_position=10)

    def _logits(self, step: int) -> np.ndarray:
        rng = np.random.default_rng((self.seed, step))
        return rng.normal(size=self.vocab) * self.scale

    def start(self, prepared: PreparedPrompt) -> np.ndarray:
        self._step = 0
        return self._logits(0)

    def step(self, token_id: int, position: int) -> np.ndarray:
        self._step += 1
        return self._logits(self._step)


def test_generate_chain_is_deterministic_and_returns_one_row_per_token() -> None:
    arm = ScriptedArm("a", seed=1)

    chain, rows = generate_chain(
        arm, arm.prepare("p", "page.png"), max_tokens=5, eos_token_id=-1
    )

    assert len(chain) == 5
    assert rows.shape == (5, 8)
    assert chain == [int(np.argmax(rows[i])) for i in range(5)]


def test_generate_chain_stops_at_eos_without_emitting_it() -> None:
    arm = ScriptedArm("a", seed=1)
    prepared = arm.prepare("p", "page.png")
    full_chain, _ = generate_chain(arm, prepared, max_tokens=3, eos_token_id=-1)
    eos = full_chain[1]

    chain, rows = generate_chain(arm, prepared, max_tokens=5, eos_token_id=eos)

    assert chain == full_chain[:1]
    assert rows.shape == (1, 8)


def test_force_chain_returns_one_row_per_chain_token() -> None:
    arm = ScriptedArm("a", seed=2)

    rows = force_chain(arm, arm.prepare("p", "page.png"), [1, 2, 3, 4])

    assert rows.shape == (4, 8)


def test_an_identical_arm_has_zero_kl_and_perfect_top1() -> None:
    teacher = ScriptedArm("teacher", seed=7)
    twin = ScriptedArm("twin", seed=7)
    case = _case()

    comparisons = run_comparison(
        teacher=teacher, arms=[twin], cases=[case], eos_token_id=-1
    )

    report = comparisons[0].global_report
    assert report.max_kl == pytest.approx(0.0, abs=1e-15)
    assert report.top1_agreement == 1.0
    assert report.n_rows > 0


def test_a_different_arm_reports_positive_kl_and_flips() -> None:
    teacher = ScriptedArm("teacher", seed=7)
    other = ScriptedArm("other", seed=99)
    case = _case()

    comparisons = run_comparison(
        teacher=teacher, arms=[other], cases=[case], eos_token_id=-1
    )

    report = comparisons[0].global_report
    assert report.mean_kl > 0.0
    assert report.max_kl > 0.0
    # Different seeds flip many argmaxes; the point is that they are counted.
    assert report.top1_flips > 0


def _case():
    from scripts.surya_numerical_gate import Case

    return Case(name="page", page_path="page.png", prompt="p", max_tokens=6)


def test_run_comparison_marks_the_teacher_as_zero_drift() -> None:
    teacher = ScriptedArm("teacher", seed=3)

    comparisons = run_comparison(
        teacher=teacher, arms=[teacher], cases=[_case()], eos_token_id=-1
    )

    assert comparisons[0].arm == "teacher"
    assert comparisons[0].teacher == "teacher"
    assert comparisons[0].global_report.max_kl == 0.0


def test_run_comparison_covers_every_scope() -> None:
    teacher = ScriptedArm("teacher", seed=3)
    other = ScriptedArm("other", seed=4)
    cases = [_case(), _case()]

    comparisons = run_comparison(
        teacher=teacher, arms=[other], cases=cases, eos_token_id=-1
    )

    assert [r.scope for r in comparisons[0].scopes] == ["page", "page"]


def test_run_comparison_attaches_review_rows_for_high_kl() -> None:
    teacher = ScriptedArm("teacher", seed=3)
    other = ScriptedArm("other", seed=99)

    comparisons = run_comparison(
        teacher=teacher, arms=[other], cases=[_case()], eos_token_id=-1
    )

    # The scripted arms disagree hard, so the worst rows must be diagnosable.
    assert comparisons[0].global_report.rows_over_review > 0
    assert comparisons[0].review_rows
    assert all(r.top5_overlap >= 0 for r in comparisons[0].review_rows)
    assert comparisons[0].as_dict()["review_rows"]


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------


def test_verdict_passes_a_perfect_comparison() -> None:
    comparison = ArmComparison(
        arm="a", teacher="t", scopes=[_report(scope="p1"), _report(scope="p2")]
    )

    passed, failures = verdict(comparison)

    assert passed, failures


def test_verdict_applies_the_per_scope_top1_bar() -> None:
    comparison = ArmComparison(
        arm="a", teacher="t",
        scopes=[_report(scope="p1", top1_agreement=0.98)],
    )

    passed, failures = verdict(comparison)

    assert not passed
    assert any("p1: top1" in f for f in failures)


def test_verdict_can_take_a_tighter_envelope() -> None:
    comparison = ArmComparison(arm="a", teacher="t", scopes=[_report()])

    passed, failures = verdict(
        comparison, ProductionEnvelope(mean_kl=1e-9)
    )

    assert not passed
    assert any("mean_kl" in f for f in failures)
