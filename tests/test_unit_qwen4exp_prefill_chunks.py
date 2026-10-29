"""Bounded prompt work keeps the same chunk arithmetic and absolute cursors."""
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner
from hipengine.runtime.qwen4_exp_mtp import Qwen4ExpGGUFMTPDraftRunner


class _TargetHarness(SimpleNamespace):
    def __getattr__(self, name):
        return getattr(Qwen4ExpGGUFResidentModelRunner, name).__get__(self)


def _target(monkeypatch):
    module = "hipengine.runtime.qwen4_exp_runner"
    runner = _TargetHarness()
    runner.q8_wave_scale_prefill = False
    runner.max_sequence_length = 8
    runner.prefill_chunk_size = 2
    runner.position = 0
    runner.history = []
    runner.chunks = []
    runner.config = SimpleNamespace(
        residual_branch_count=1, hidden_size=2, residual_low_rank=1, vocab_size=4,
    )
    runner.head_scratch = object()
    runner.runtime = SimpleNamespace(memcpy=lambda *args: None)
    runner.last_target_hidden = SimpleNamespace(ptr=1, nbytes=8)
    runner.logits_buffer = SimpleNamespace(ptr=2)
    weight = SimpleNamespace(allocation=lambda stage: SimpleNamespace(tensor=SimpleNamespace(ptr=3)))
    runner.resident = SimpleNamespace(weight=lambda name: weight)
    runner._q8_mmq_prefill_context = nullcontext
    runner._wide_f16_activation_context = nullcontext

    def reset():
        runner.position = 0
        runner.history.clear()

    def chunk(tokens, **kwargs):
        runner.chunks.append((runner.position, tuple(tokens)))
        rows = []
        for token in tokens:
            runner.history.append(int(token))
            runner.position += 1
            rows.append([sum(runner.history), runner.position])
        return 1, np.asarray(rows, dtype=np.float32) if kwargs["capture_hidden_seeds"] else None

    runner.reset = reset
    runner._prefill_chunk = chunk
    runner._finish_lm_head = lambda **kwargs: SimpleNamespace(
        token_id=sum(runner.history), hidden_seeds=kwargs["hidden_seeds"],
    )
    monkeypatch.setattr(module + ".run_qwen4_exp_gr_read", lambda *a, **kw: SimpleNamespace(mixed=SimpleNamespace(ptr=1)))
    monkeypatch.setattr(module + ".launch_gguf_linear", lambda *a, **kw: None)
    return runner


def test_prefill_append_preserves_chunk_history_and_hidden_rows(monkeypatch):
    runner = _target(monkeypatch)
    baseline = runner.prefill((1, 2, 3, 4, 5), capture_hidden_seeds=True)
    expected_chunks = list(runner.chunks)
    runner.reset()
    runner.chunks.clear()
    pieces = [runner.prefill_append(tokens, capture_hidden_seeds=True)
              for tokens in ((1, 2), (3, 4), (5,))]
    assert runner.chunks == expected_chunks
    assert runner.position == 5
    assert pieces[-1].token_id == baseline.token_id == 15
    np.testing.assert_array_equal(np.concatenate([piece.hidden_seeds for piece in pieces]), baseline.hidden_seeds)


def test_prefill_append_checks_remaining_capacity_before_mutation(monkeypatch):
    runner = _target(monkeypatch)
    runner.prefill((1, 2, 3, 4, 5, 6, 7))
    with pytest.raises(ValueError, match="capacity"):
        runner.prefill_append((8, 9))
    assert runner.position == 7 and runner.history == list(range(1, 8))
    # Ordinary prefill still begins a new request.
    runner.prefill((4, 5))
    assert runner.position == 2 and runner.history == [4, 5]


class _DraftHarness(Qwen4ExpGGUFMTPDraftRunner):
    @property
    def rows(self):
        return self.staged[:self.position]


def _draft():
    draft = object.__new__(_DraftHarness)
    draft.closed = False
    draft.max_sequence_length = 8
    draft.config = SimpleNamespace(residual_width=2)
    draft.position = 0
    draft.staged = []
    draft.cache_tiles = []
    draft.index = SimpleNamespace(count=0)
    draft.index.restore_count = lambda count: setattr(draft.index, "count", count)
    draft.attention = SimpleNamespace(set_position=lambda pos: None)

    def store(position, token, hidden):
        row = (int(token), tuple(float(value) for value in hidden))
        if position == len(draft.staged):
            draft.staged.append(row)
        else:
            draft.staged[position] = row

    def reset():
        draft.position = draft.index.count = 0
        draft.staged.clear()
        draft.cache_tiles.clear()

    def forward(token, hidden, **kwargs):
        store(draft.position, token, hidden)
        draft.position += 1
        draft.index.count = draft.position

    def append(tokens, hidden):
        draft.cache_tiles.append(tuple(tokens))
        for offset, (token, seed) in enumerate(zip(tokens, hidden, strict=True)):
            store(draft.position + offset, token, seed)
        draft.index.count += len(tokens)

    draft._prefill_workspace = SimpleNamespace(rows=2, append=append)
    draft.reset = reset
    draft.forward = forward
    return draft


def test_draft_priming_chunks_preserve_shifted_hidden_boundary():
    tokens = (1, 2, 3, 4, 5)
    hidden = np.arange(10, dtype=np.float32).reshape(5, 2)
    draft = _draft()
    draft.prime_prompt(tokens, hidden)
    expected = list(draft.rows)
    draft.reset()
    draft.prime_prompt_chunk(tokens[:2], hidden[:2], previous_hidden=None)
    draft.prime_prompt_chunk(tokens[2:4], hidden[2:4], previous_hidden=hidden[1])
    draft.prime_prompt_chunk(tokens[4:], hidden[4:], previous_hidden=hidden[3])
    assert draft.position == len(tokens)
    assert draft.rows == expected


def test_draft_chunk_requires_a_real_cross_chunk_seed():
    draft = _draft()
    hidden = np.zeros((2, 2), dtype=np.float32)
    draft.prime_prompt_chunk((1, 2), hidden, previous_hidden=None)
    with pytest.raises(ValueError, match="previous_hidden"):
        draft.prime_prompt_chunk((3, 4), hidden, previous_hidden=None)
    assert draft.position == 2


def test_draft_chunk_checks_cancellation_before_each_token():
    from hipengine.generation.deadline import GenerationCancelled

    draft = _draft()
    checks = []

    def check():
        checks.append(draft.position)
        if draft.position == 1:
            raise GenerationCancelled()

    with pytest.raises(GenerationCancelled):
        draft.prime_prompt_chunk(
            (1, 2, 3), np.zeros((3, 2), dtype=np.float32),
            previous_hidden=None, check_cancelled=check,
        )
    assert checks == [0, 1]
    assert draft.position == 1 and draft.rows == [(1, (0.0, 0.0))]
    assert draft.index.count == 1
    assert len(draft.staged) == 2  # The second raw row was never committed.
    # Reclaim can reset this partial request without touching a neighbor.
    draft.reset()
    draft.prime_prompt_chunk((9,), np.ones((1, 2), dtype=np.float32), previous_hidden=None)
    assert draft.position == 1 and draft.rows == [(9, (0.0, 0.0))]
