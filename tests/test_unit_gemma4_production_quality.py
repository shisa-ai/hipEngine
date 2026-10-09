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


# --- frozen 2026-10-07 stability classification ---------------------------

def test_row_stability_marks_resolvable_and_unresolvable_rows():
    from scripts.gemma4_production_quality import row_stability

    # A peaked, well-separated row: stable.
    peaked = np.zeros(64, dtype=np.float32)
    peaked[3] = 5.0
    peaked[9] = 3.0
    assert row_stability(peaked)
    # A near-tie at the top: two valid bf16 roundings can swap these.
    tied = np.zeros(64, dtype=np.float32)
    tied[3] = 5.0
    tied[9] = 4.999
    assert not row_stability(tied)
    # Near-uniform: KL is unbounded under any perturbation.
    flat = np.zeros(256, dtype=np.float32)
    assert not row_stability(flat)


def test_stable_paired_summary_excludes_unstable_rows_and_reports_them():
    from scripts.gemma4_production_quality import stable_paired_summary

    rng = np.random.default_rng(7)
    vocab = 128
    rows = 24
    baseline = (rng.normal(size=(rows, vocab)).astype(np.float32) * 0.5)
    # A strong single peak on every row keeps the top-2 gap wide so only the
    # two rows below are unstable.
    baseline[np.arange(rows), rng.integers(0, vocab, rows)] += 6.0
    # One row a near-tie at the top, one row near-uniform.
    baseline[5, :2] = 5.0
    baseline[5, 2:] = 0.0
    baseline[11, :] = 0.0
    # A candidate identical on stable rows, distributionally different on the
    # unstable ones: a third peak on the tied row, a lone peak on the flat one.
    candidate = baseline.copy()
    candidate[5, 2] += 8.0
    candidate[11, 3] += 12.0
    verdict, diagnostics = stable_paired_summary(baseline, candidate)
    assert diagnostics["rows_total"] == rows
    assert diagnostics["rows_stable"] == rows - 2
    assert diagnostics["rows_unstable"] == 2
    # The stable rows are bitwise equal, so every stable bar passes with zero KL.
    assert verdict["kl_mean"] == 0.0 and verdict["kl_max"] == 0.0
    assert verdict["passed"]
    # The unstable rows are reported, not hidden.
    assert diagnostics["unstable_kl_max"] > 0.0
