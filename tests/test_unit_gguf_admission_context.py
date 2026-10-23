"""Reject impossible context lengths before publishing resident request rows."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hipengine.generation.engine_loop import EngineLoopConfig
from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner
from tests.test_unit_gguf_prefix_cache_runner import _FakePrefixOwner, _request


@pytest.fixture
def runner():
    owner = _FakePrefixOwner()
    runner = Qwen35GGUFResidentModelRunner(owner, capacity=1)
    runner.configure_engine_loop(EngineLoopConfig(max_active_requests=1, prefix_cache="off"))
    yield runner
    runner.close()


@pytest.mark.parametrize("prompt_length", [17, 1020, 1021, 1022, 1400])
def test_registration_checks_actual_kv_capacity_before_publication(runner, prompt_length):
    prompt = (42,) * prompt_length
    request = _request(prompt, max_tokens=4)
    if prompt_length + 3 > 1024:
        batch_id = runner._next_batch_id
        with pytest.raises(ValueError, match="KV positions but resident capacity is 1024"):
            runner.register_batch((1,), request, prompt_rows=(prompt,))
        assert runner._rows == {}
        assert runner._next_batch_id == batch_id
        assert len(runner._available) == 1
        # An unrelated valid request still registers and reserves after refusal.
        short = (42,) * 17
        runner.register_batch((2,), _request(short, max_tokens=4), prompt_rows=(short,))
        runner.reserve_admission(SimpleNamespace(request_id=2))
        assert runner._rows[2].lease is not None
    else:
        runner.register_batch((1,), request, prompt_rows=(prompt,))
        runner.reserve_admission(SimpleNamespace(request_id=1))
        assert runner._rows[1].lease is not None


def test_batch_validation_is_atomic_and_uses_busy_session_capacity(runner):
    short = (42,) * 17
    runner.register_batch((1,), _request(short, max_tokens=4), prompt_rows=(short,))
    runner.reserve_admission(SimpleNamespace(request_id=1))
    assert not runner._available
    neighbor = runner._rows[1]
    long = (42,) * 1022
    request = replace(_request(short, max_tokens=4), prompts=(short, long))
    with pytest.raises(ValueError, match="resident capacity is 1024"):
        runner.register_batch((2, 3), request, prompt_rows=(short, long))
    assert runner._rows == {1: neighbor}
    assert neighbor.lease is not None


def test_zero_output_does_not_reserve_prompt_kv(runner):
    prompt = (42,) * 1400
    runner.register_batch((1,), _request(prompt, max_tokens=0), prompt_rows=(prompt,))
    runner.reserve_admission(SimpleNamespace(request_id=1))
    assert runner._rows[1].lease is None


@pytest.mark.parametrize("row_limit", [0, 2])
def test_capacity_validation_matches_effective_dms_row_limit(runner, monkeypatch, row_limit):
    monkeypatch.setattr(runner, "scheduler_max_new_tokens_for_prompt", lambda *args: row_limit)
    prompt = (42,) * 1023
    request = _request(prompt, max_tokens=4)
    if row_limit == 0:
        with pytest.raises(ValueError, match="resident capacity is 1024"):
            runner.register_batch((1,), request, prompt_rows=(prompt,))
    else:
        runner.register_batch((1,), request, prompt_rows=(prompt,))
        assert runner._rows[1].request.max_tokens == 2
        runner.reserve_admission(SimpleNamespace(request_id=1))
        assert runner._rows[1].lease is not None
