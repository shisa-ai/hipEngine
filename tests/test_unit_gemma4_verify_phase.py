"""CPU contracts for explicit speculative verification phase selection."""
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime import gemma4 as module


@pytest.mark.parametrize("rows", [1, 3, 8, 11])
@pytest.mark.parametrize("verification", [False, True])
def test_forward_propagates_phase_without_mutating_prompt_policy(monkeypatch, rows, verification):
    runner = module.Gemma4Runner.__new__(module.Gemma4Runner)
    runner._closed = False
    runner._position = 0
    runner.capacity = 32
    runner.max_block = 8
    runner.max_logits_rows = 1
    runner.weights = SimpleNamespace(config=SimpleNamespace(vocab_size=64))
    runner.prefill_attention_variants = ("gemma4_wmma_flash",)
    calls = []
    sessions = []

    @contextmanager
    def session(enabled):
        sessions.append(enabled)
        yield

    def forward_block(tokens, **kwargs):
        calls.append((len(tokens), kwargs["verification"]))
        return np.zeros(64, dtype=np.float32)

    monkeypatch.setattr(module, "_gemma4_block_wmma_session", session)
    monkeypatch.setattr(runner, "_forward_block", forward_block)
    runner.forward(list(range(rows)), verification=verification)
    widths = [min(8, rows - i) for i in range(0, rows, 8)]
    assert calls == [(width, verification) for width in widths]
    assert sessions == [not verification and width == 8 for width in widths]
    assert runner.prefill_attention_variants == ("gemma4_wmma_flash",)
