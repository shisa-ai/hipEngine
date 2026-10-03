"""CPU contracts for the Gemma multi-category numerical packet."""
import numpy as np
import pytest

from scripts.gemma4_production_quality import paired_summary, check_task_answer, greedy_output, padded_chat


def test_padding_expands_text_corpus_without_cycling(monkeypatch):
    from scripts import gemma4_campaign_bench as bench
    counts = []
    def corpus(count=96, *, seed):
        counts.append(count)
        return tuple(f"record-{i} " for i in range(count))
    monkeypatch.setattr(bench, "probe_corpus", corpus)
    class Generator:
        def render_chat_prompt(self, messages, **kwargs):
            return messages[0]["content"]
        def tokenize(self, text):
            assert isinstance(text, str)
            return list(text.encode())
    ids = padded_chat(Generator(), "task", 8191, 123)
    assert len(ids) == 8191
    assert counts[-1] > 96


def test_task_generation_stops_at_tokenizer_eog():
    class Runner:
        position = 0
        calls = 0
        def reset(self):
            self.position = 0
        def forward(self, ids):
            self.calls += 1
            self.position += len(ids)
            return np.array([0., 2., 0.]) if self.calls == 1 else np.array([0., 0., 2.])
    runner = Runner()
    assert greedy_output(runner, [8, 9], 32, stop_ids={2}) == [1]
    assert runner.calls == 2


def test_paired_scope_uses_97_percent_while_global_uses_99():
    strict = np.zeros((64, 3))
    candidate = strict.copy()
    candidate[0, 1] = 0.001
    assert paired_summary(strict, candidate, scope=True)["passed"]
    assert not paired_summary(strict, candidate)["passed"]


def test_summary_rejects_shape_mismatch_and_nonfinite():
    with pytest.raises(ValueError):
        paired_summary(np.zeros((3, 4)), np.zeros((2, 4)))
    with pytest.raises(ValueError):
        paired_summary(np.zeros((3, 4)), np.full((3, 4), np.nan))


def test_summary_distinguishes_review_boundary_from_ceiling():
    strict = np.array([[0.0, 0.0]])
    report = paired_summary(strict, np.array([[0.5, 0.0]]))
    assert report["requires_review"]
    assert 0.02 < report["kl_max"] < 0.05
    assert report["top1_rate"] == 1.0


def test_small_scope_one_flip_cannot_hide_in_global_pass():
    strict = np.zeros((64, 3))
    candidate = strict.copy()
    candidate[0, 1] = 0.001
    report = paired_summary(strict, candidate)
    assert report["top1_rate"] == 63 / 64
    assert not report["passed"]


@pytest.mark.parametrize("output,expected,valid", [
    ('{"answer": "maple-481"}', "maple-481", True),
    ('```json\n{"answer": "maple-481"}\n```', "maple-481", True),
    ('{"answer": "wrong"}', "maple-481", False),
    ('{"answer": "maple-481", "other": 2}', "maple-481", False),
    ('some prose maple-481', "maple-481", False),
])
def test_task_checks_actual_json_and_exact_retrieval(output, expected, valid):
    assert check_task_answer(output, expected) is valid
