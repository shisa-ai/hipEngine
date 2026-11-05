"""Bounded reset is frozen by the runner, not selected again at reset time."""
from types import SimpleNamespace

import pytest

from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner


@pytest.mark.parametrize("enabled", [False, True])
def test_reset_uses_owner_choice_despite_opposing_process_input(monkeypatch, enabled):
    events = []
    attention = SimpleNamespace(
        key_cache=SimpleNamespace(ptr=1, nbytes=64),
        value_cache=SimpleNamespace(ptr=2, nbytes=64),
        clear_written_cache=lambda: events.append("bounded"),
        set_position=lambda position: events.append(("position", position)),
    )
    owner = SimpleNamespace(
        bounded_kv_reset=enabled,
        _require_open=lambda: None,
        state=SimpleNamespace(zero=lambda: events.append("state")),
        runtime=SimpleNamespace(memset=lambda *args: events.append(("full", args))),
        attention_states=[attention], index_states=[],
        _ple_hash_states={1: object()}, position=3,
    )
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_BOUNDED_KV_RESET", "0" if enabled else "1")
    Qwen4ExpGGUFResidentModelRunner.reset(owner)
    assert ("bounded" in events) is enabled
    assert sum(isinstance(event, tuple) and event[0] == "full" for event in events) == (0 if enabled else 2)
    assert owner.position == 0 and owner._ple_hash_states == {}


def test_reset_policy_is_constructor_input():
    import inspect

    parameters = inspect.signature(Qwen4ExpGGUFResidentModelRunner).parameters
    assert parameters["bounded_kv_reset"].default is None
