from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.kernels.cpu_reference.qwen4_exp import PLEHashState
from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner
from tests.test_qwen4_exp_gguf_config import _info
from hipengine.loading.qwen4_exp_gguf import qwen4_exp_gguf_config_from_metadata


def test_qwen4_exp_model_runner_rejects_context_above_native_limit_before_allocating() -> None:
    config = qwen4_exp_gguf_config_from_metadata(_info())
    resident = SimpleNamespace(plan=SimpleNamespace(config=config))

    with pytest.raises(ValueError, match="1..262144"):
        Qwen4ExpGGUFResidentModelRunner(resident, max_sequence_length=262145)


def test_qwen4_exp_model_runner_snapshot_restores_transactional_cursors_and_state() -> None:
    events: list[object] = []

    class DecodeState:
        def snapshot(self):
            events.append("snapshot")
            return "device-state"

        def restore(self, value):
            events.append(("restore", value))

    class Attention:
        def set_position(self, value):
            events.append(("attention", int(value)))

    class Index:
        def restore_count(self, value):
            events.append(("index", int(value)))

    runner = object.__new__(Qwen4ExpGGUFResidentModelRunner)
    runner.closed = False
    runner.max_sequence_length = 8
    runner.state = DecodeState()
    runner.position = 3
    runner._ple_hash_states = {0: PLEHashState((4, 5), 3)}
    runner.attention_states = (Attention(), Attention())
    runner.index_states = (Index(), Index())

    snapshot = runner.snapshot()
    runner.position = 6
    runner._ple_hash_states = {}
    runner.restore(snapshot)

    assert runner.position == 3
    assert runner._ple_hash_states == {0: PLEHashState((4, 5), 3)}
    assert events == [
        "snapshot",
        ("restore", "device-state"),
        ("attention", 2),
        ("attention", 2),
        ("index", 3),
        ("index", 3),
    ]


def test_qwen4_exp_model_runner_generation_does_not_consume_after_last_output() -> None:
    runner = object.__new__(Qwen4ExpGGUFResidentModelRunner)
    runner.closed = False
    calls: list[int] = []
    runner.prefill = lambda tokens, **_kwargs: SimpleNamespace(token_id=7, logits=None)

    def step(token, **_kwargs):
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
