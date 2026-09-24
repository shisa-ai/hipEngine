"""Accounting tests for the Gemma 4 campaign benchmark harness (G0).

The regression these cover: the campaign's headline metric is only trustworthy
if the harness counts phases the way the contract defines them — 128 outputs
mean one prefill (which yields the first token) plus exactly 127 decode
forwards, the first-token latency includes the prefill that produced token one,
and a zero-length decode never divides by zero. The timing values here come
from a deterministic fake clock and a fake runner, so these tests prove the
arithmetic, not the GPU.
"""

from __future__ import annotations

import math

import pytest

from scripts.gemma4_campaign_bench import (
    exact_prompt_ids,
    memory_row,
    run_instrumented,
    summarize,
)


class _FakeRunner:
    """Stands in for Gemma4Runner: records calls, returns scripted tokens."""

    def __init__(self, token_ids=()):
        self._tokens = list(token_ids)
        self._cursor = 0
        self.forward_calls: list[list[int]] = []
        self.next_token_calls = 0
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def forward(self, token_ids, **kwargs):
        self.forward_calls.append([int(t) for t in token_ids])
        return object()  # opaque logits handle for the fake sampler

    def next_token(self, logits) -> int:
        self.next_token_calls += 1
        token = self._tokens[self._cursor % len(self._tokens)]
        self._cursor += 1
        return int(token)


class _FakeClock:
    """Returns 0.0, 0.1, 0.2, ... so every timing difference is exact."""

    def __init__(self):
        self.calls = 0

    def __call__(self) -> float:
        value = self.calls * 0.1
        self.calls += 1
        return value


def _record(**overrides):
    record = {
        "prompt_tokens": 1024,
        "generated_tokens": 128,
        "decode_forwards": 127,
        "prefill_s": 1.0,
        "first_token_s": 1.5,
        "first_sample_s": 0.5,
        "decode_s": 10.0,
        "wall_s": 11.5,
        "syncs": 128,
        "finish_reason": "length",
        "eos_ignored": True,
        "generated_token_ids": [],
    }
    record.update(overrides)
    return record


def test_run_instrumented_counts_phases_exactly():
    runner = _FakeRunner(token_ids=(7, 8))
    clock = _FakeClock()
    syncs: list[str] = []

    record = run_instrumented(
        runner,
        prompt_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        max_tokens=5,
        clock=clock,
        sync=lambda: syncs.append("s"),
    )

    # One prefill carrying the whole prompt, then max_tokens - 1 decode
    # forwards of exactly one token each: the 128-output denominator.
    assert runner.forward_calls[0] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert all(len(call) == 1 for call in runner.forward_calls[1:])
    assert len(runner.forward_calls) == 1 + (5 - 1)
    assert runner.next_token_calls == 5  # first token from prefill + 4 decode
    assert record["decode_forwards"] == 4
    assert record["prompt_tokens"] == 8
    assert record["generated_tokens"] == 5

    # Clock call order: t0, t1 (prefill), tf (first token), then a start/end
    # pair per decode step, then the wall stop. Fake delta is 0.1 each.
    assert record["prefill_s"] == pytest.approx(0.1)
    assert record["first_token_s"] == pytest.approx(0.2)
    assert record["first_sample_s"] == pytest.approx(0.1)
    assert record["decode_s"] == pytest.approx(4 * 0.1)
    assert record["wall_s"] == pytest.approx((clock.calls - 1) * 0.1)
    # 2 around the prefill + 1 per decode step.
    assert len(syncs) == 2 + 4
    assert record["syncs"] == 6

    # Fixed-length campaign rows ignore EOS by construction and say so.
    assert record["finish_reason"] == "length"
    assert record["eos_ignored"] is True
    assert record["generated_token_ids"] == [7, 8, 7, 8, 7]


def test_run_instrumented_single_output_has_no_decode_phase():
    runner = _FakeRunner(token_ids=(3,))
    clock = _FakeClock()

    record = run_instrumented(
        runner, prompt_ids=[1, 2], max_tokens=1, clock=clock, sync=lambda: None
    )

    assert record["decode_forwards"] == 0
    assert record["decode_s"] == 0.0
    assert record["generated_tokens"] == 1
    # Only the prefill and its single sampler survived; no decode pair ran.
    assert clock.calls == 4  # t0, t1, tf, wall
    # A zero-length decode must not divide by zero downstream.
    stats = summarize([record])
    assert stats["decode_tps"] is None


def test_run_instrumented_rejects_empty_prompt_and_nonpositive_output():
    runner = _FakeRunner(token_ids=(1,))
    with pytest.raises(ValueError, match="prompt"):
        run_instrumented(runner, prompt_ids=[], max_tokens=4)
    with pytest.raises(ValueError, match="max_tokens"):
        run_instrumented(runner, prompt_ids=[1], max_tokens=0)


def test_summarize_median_p95_and_phase_accounting():
    # decode_tps = 127 / decode_s → 10, 12, 30 tok/s.
    records = [
        _record(decode_s=127 / 10.0, prefill_s=3.0, first_token_s=3.6, wall_s=15.7),
        _record(decode_s=127 / 12.0, prefill_s=1.0, first_token_s=1.4, wall_s=11.6),
        _record(decode_s=127 / 30.0, prefill_s=2.0, first_token_s=2.5, wall_s=6.6),
    ]

    stats = summarize(records)

    assert stats["samples"] == 3
    assert stats["decode_tps"] == pytest.approx(12.0)
    assert stats["decode_tps_min"] == pytest.approx(10.0)
    assert stats["decode_tps_max"] == pytest.approx(30.0)
    # Nearest-rank p95 over 3 samples is the max; stated explicitly so the
    # statistic is reproducible rather than library-dependent.
    assert stats["decode_tps_p95"] == pytest.approx(30.0)
    assert stats["decode_tps_stdev"] == pytest.approx(
        _stdev([10.0, 12.0, 30.0]), rel=1e-9
    )
    assert stats["prefill_s"] == pytest.approx(2.0)  # median
    assert stats["prefill_tps"] == pytest.approx(1024 / 2.0)
    assert stats["first_token_s"] == pytest.approx(2.5)
    assert stats["wall_s"] == pytest.approx(11.6)
    assert stats["prompt_tokens"] == 1024
    assert stats["generated_tokens"] == 128
    assert stats["decode_forwards_per_sample"] == 127


def test_summarize_rejects_mixed_shapes():
    with pytest.raises(ValueError, match="shape"):
        summarize([_record(), _record(prompt_tokens=512)])


def _stdev(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def test_exact_prompt_ids_hits_the_target_exactly():
    tokenize = lambda text: [len(word) for word in text.split()]  # noqa: E731
    corpus = ("alpha beta gamma", "delta epsilon")

    ids = exact_prompt_ids(tokenize, 4, corpus=corpus)

    assert ids == [5, 4, 5, 5]  # first sentence, then first word of the next
    assert len(ids) == 4
    assert exact_prompt_ids(tokenize, 4, corpus=corpus) == ids  # deterministic
    assert exact_prompt_ids(tokenize, 2, corpus=corpus) == [5, 4]
    assert exact_prompt_ids(tokenize, 7, corpus=corpus) == [5, 4, 5, 5, 7] + [
        5,
        4,
    ]


def test_memory_row_reports_used_and_peak_from_labeled_samples():
    row = memory_row(
        [
            ("before_load", 25_600_000_000, 25_753_026_560),
            ("after_load", 10_400_000_000, 25_753_026_560),
            ("after_runs", 10_100_000_000, 25_753_026_560),
        ]
    )
    assert row["total_bytes"] == 25_753_026_560
    assert row["used_bytes"]["before_load"] == 25_753_026_560 - 25_600_000_000
    assert row["used_bytes"]["after_load"] == 25_753_026_560 - 10_400_000_000
    assert row["peak_used_bytes"] == 25_753_026_560 - 10_100_000_000
    assert row["labels"] == ["before_load", "after_load", "after_runs"]


def test_memory_row_rejects_empty_bad_shape_and_inconsistent_totals():
    with pytest.raises(ValueError, match="at least one"):
        memory_row([])
    with pytest.raises(ValueError, match="label"):
        memory_row([("only", 100, 200, "extra")])
    with pytest.raises(ValueError, match="total"):
        memory_row([("a", 100, 200), ("b", 90, 210)])


def test_exact_prompt_ids_rejects_nonpositive_target():
    with pytest.raises(ValueError, match="target"):
        exact_prompt_ids(lambda text: [1], 0, corpus=("x y",))