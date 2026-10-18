"""CPU controls for Gemma's same-prefix attribution (no HIP imports)."""
import copy

import numpy as np
import pytest

from scripts.gemma4_mtp_prefix_attribution import committed_rows, paired_summary, teacher_schedule


def trace():
    eye = np.eye(10, dtype=np.float32)
    return [
        {"start": 0, "tokens": [9], "logits": eye[1]},
        {"start": 1, "tokens": [1, 2, 7], "keep": 3, "logits": eye[[2, 3, 8]]},
        {"start": 3, "tokens": [3], "keep": 4, "logits": eye[4]},
    ]


def test_rows_predict_the_next_published_token_not_the_consumed_token():
    rows = committed_rows([9], [1, 2, 3, 4], trace())
    assert rows.argmax(1).tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize("mutation", [
    lambda t: t[1].update(start=2),
    lambda t: t[1].update(keep=5),
    lambda t: t[1].update(tokens=[1, 5, 7]),
    lambda t: t[2].update(logits=np.eye(10)[5]),
])
def test_cursor_wrong_prefix_or_wrong_publication_is_not_numerical_drift(mutation):
    captured = copy.deepcopy(trace())
    mutation(captured)
    with pytest.raises(ValueError):
        committed_rows([9], [1, 2, 3, 4], captured)


def test_teacher_replay_replaces_tokens_but_preserves_verify_and_rewind_shapes():
    class Runner:
        position = 0

        def reset(self):
            self.position = 0

        def forward(self, tokens, **kwargs):
            self.position += len(tokens)
            rows = np.array([[token, token + 1] for token in tokens])
            return rows if kwargs.get("logits_rows", 1) > 1 else rows[-1]

        def rewind(self, position):
            self.position = position

    captured = trace()
    for block in captured:
        block["kwargs"] = {"logits_rows": len(block["tokens"])}
    runner = Runner()
    rows = teacher_schedule(runner, [9], [4, 5, 6, 7], captured)
    assert rows.tolist() == [[9, 10], [4, 5], [5, 6], [6, 7]]
    assert runner.position == 4
    with pytest.raises(ValueError, match="cannot fill"):
        teacher_schedule(runner, [9], [4], captured)


def test_near_tie_reports_numeric_change_without_claiming_ownership_failure():
    baseline = np.array([[1, 1.001, -2], [2, -1, 0]])
    candidate = baseline.copy()
    candidate[0, 0] += .002
    report = paired_summary(baseline, candidate)
    assert report["top1_flips"] == [0]
    assert 0 < report["kl_max"] < 1e-5
    assert "passed" not in report
    assert "ownership_checks_passed" not in report


def test_sparse_moe_damage_statistics_do_not_turn_a_diagnostic_into_a_gate():
    baseline = np.tile([8.0, 0.0], (100, 1))
    candidate = baseline.copy()
    candidate[0] = [0.0, 8.0]
    report = paired_summary(baseline, candidate)
    assert report["kl_median"] == 0
    assert report["kl_p90"] == 0
    assert report["kl_above_0_05_rate"] == .01
    assert report["top1_agreement"] == .99
    assert report["kl_max"] > .05
    assert "passed" not in report


@pytest.mark.parametrize("other", [np.ones((3, 2)), np.array([[float("nan"), 1]])])
def test_misaligned_or_nonfinite_logits_cannot_be_scored(other):
    with pytest.raises(ValueError):
        paired_summary(np.ones((1, 2)), other)
