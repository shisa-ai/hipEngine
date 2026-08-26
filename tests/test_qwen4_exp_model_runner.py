from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner
from tests.test_qwen4_exp_gguf_config import _info
from hipengine.loading.qwen4_exp_gguf import qwen4_exp_gguf_config_from_metadata


def test_qwen4_exp_model_runner_rejects_context_above_dense_equivalence_before_allocating() -> None:
    config = qwen4_exp_gguf_config_from_metadata(_info())
    resident = SimpleNamespace(plan=SimpleNamespace(config=config))

    with pytest.raises(ValueError, match="1..2051"):
        Qwen4ExpGGUFResidentModelRunner(resident, max_sequence_length=2052)


def test_qwen4_exp_model_runner_generation_does_not_consume_after_last_output() -> None:
    runner = object.__new__(Qwen4ExpGGUFResidentModelRunner)
    runner.closed = False
    calls: list[int] = []
    runner.prefill = lambda tokens: SimpleNamespace(token_id=7, logits=None)

    def step(token):
        calls.append(int(token))
        return SimpleNamespace(token_id=int(token) + 1, logits=None)

    runner.step = step
    assert runner.generate([1], max_new_tokens=3) == (7, 8, 9)
    assert calls == [7, 8]


def test_qwen4_exp_model_runner_public_methods_fail_closed_on_empty_work() -> None:
    runner = object.__new__(Qwen4ExpGGUFResidentModelRunner)
    runner.closed = False
    runner.reset = lambda: None
    runner.step = lambda token: SimpleNamespace(token_id=int(token), logits=None)

    with pytest.raises(ValueError, match="at least one"):
        runner.prefill([])
    with pytest.raises(ValueError, match="positive"):
        runner.generate([1], max_new_tokens=0)
