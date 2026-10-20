import numpy as np

from scripts.qwen4exp_long_context_gate import (
    CONTROL_MEAN_KL_CAP,
    _summary,
    decide,
    gold_answer_tokens,
    score_retrieval,
)


def _stats(mean, p95, top1, kl_max=None):
    return {"kl_mean": mean, "kl_p95": p95, "kl_max": kl_max if kl_max is not None else p95 * 2,
            "top1_matches": top1}


CONTROLS = [_stats(1.5e-3, 8e-3, 98), _stats(2.5e-3, 1.2e-2, 99, 4e-2),
            _stats(3.5e-3, 1.6e-2, 99)]


def test_candidate_inside_round_off_floor_passes():
    result = decide(_stats(2.1e-3, 1.0e-2, 98, 3e-2), CONTROLS, 16, 16, deterministic=True)
    assert result["status"] == "passed"
    # Median for mean/p95, largest max, lowest top-1.
    assert result["floor"] == {"kl_mean": 2.5e-3, "kl_p95": 1.2e-2, "kl_max": 4e-2,
                               "top1_matches": 98}


def test_each_binding_axis_fails_on_its_own():
    base = dict(candidate=_stats(2e-3, 1e-2, 98, 3e-2), sc=16, cc=16, det=True)
    cases = {
        "mean_within_floor": dict(base, candidate=_stats(2.6e-3, 1e-2, 98, 3e-2)),
        "p95_within_floor": dict(base, candidate=_stats(2e-3, 1.3e-2, 98, 3e-2)),
        "max_within_floor": dict(base, candidate=_stats(2e-3, 1e-2, 98, 5e-2)),
        "top1_within_floor": dict(base, candidate=_stats(2e-3, 1e-2, 97, 3e-2)),
        "retrieval_not_below_strict": dict(base, cc=15),
        "candidate_repeats_identical": dict(base, det=False),
    }
    for check, case in cases.items():
        result = decide(case["candidate"], CONTROLS, case["cc"], case["sc"],
                        deterministic=case["det"])
        assert result["status"] == "failed", check
        assert [k for k, v in result["checks"].items() if not v] == [check]


def test_ineffective_or_oversized_controls_invalidate_the_measurement():
    dead = [_stats(0.0, 0.0, 100, 0.0), CONTROLS[0]]
    assert decide(_stats(0, 0, 100, 0), dead, 16, 16, deterministic=True)["status"] == "invalid"
    loud = [_stats(CONTROL_MEAN_KL_CAP * 2, 0.1, 90, 0.2), CONTROLS[0]]
    assert decide(_stats(1e-3, 1e-2, 99, 2e-2), loud, 16, 16,
                  deterministic=True)["status"] == "invalid"


def test_non_finite_control_invalidates_without_missing_metric_crash():
    teacher = {"prompt": ({"token_id": 0, "logits": np.array([1.0, 0.0])},)}
    broken = {"prompt": ({"token_id": 0, "logits": np.array([np.nan, np.nan])},)}
    summary = _summary(teacher, broken, {"prompt": "code"}, "control")
    result = decide(_stats(0, 0, 100, 0), [*CONTROLS, summary], 16, 16,
                    deterministic=True)
    assert result["status"] == "invalid"
    assert not result["measurement_valid"]
    assert not result["checks"]["controls_finite"]
    assert result["checks"]["candidate_finite"]
    assert result["floor"] is None


def test_non_finite_candidate_cannot_pass_the_floor():
    for value in (np.nan, np.inf):
        result = decide(_stats(value, 0, 100, 0), CONTROLS, 16, 16,
                        deterministic=True)
        assert result["status"] == "invalid"
        assert not result["checks"]["candidate_finite"]
        assert result["checks"]["controls_finite"]


class _DigitTokenizer:
    def encode(self, text):
        return [ord(ch) for ch in text]


def test_retrieval_scores_each_answer_by_its_number_tokens():
    gold, spans = gold_answer_tokens(_DigitTokenizer(), [42, 7])
    assert "".join(chr(t) for t in gold) == "1: 42\n2: 7"
    assert ["".join(chr(gold[i]) for i in span) for span in spans] == ["42", "7"]
    vocab = 128

    def row(token):
        logits = np.zeros(vocab, dtype=np.float32)
        logits[token] = 10.0
        return {"logits": logits}

    perfect = [row(t) for t in gold]
    assert score_retrieval(perfect, gold, spans)["correct"] == [True, True]
    # A wrong format token does not cost an answer; a wrong digit does.
    wrong_format = list(perfect)
    wrong_format[1] = row(ord("."))
    assert score_retrieval(wrong_format, gold, spans)["correct_count"] == 2
    wrong_digit = list(perfect)
    wrong_digit[spans[0][1]] = row(ord("3"))
    assert score_retrieval(wrong_digit, gold, spans)["correct"] == [False, True]
