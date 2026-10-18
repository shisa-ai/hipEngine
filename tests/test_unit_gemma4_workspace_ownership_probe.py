"""CPU protocol checks for public workspace ownership and device-greedy observation."""
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import gemma4_workspace_ownership_probe as probe


def _drive(monkeypatch, tmp_path, *, expected=(2, 3)):
    from scripts import gemma4_campaign_bench as bench
    from hipengine.core import hip, memory
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as attention

    class Owner:
        def __init__(self):
            self._closed = False
            self._owned = []
            self._current = {}

        def close(self):
            self._closed = True

        def use(self):
            assert not self._closed
            if not self._owned:
                buffer = SimpleNamespace(nbytes=128)
                self._owned.append(buffer)
                self._current[0] = buffer

    closed = []
    runners = []

    def resolve(model, capacity):
        assert str(model) == "/synthetic.gguf" and capacity == 8
        owner = Owner()
        runner = SimpleNamespace(_scratches=[SimpleNamespace(attention=owner) for _ in range(2)])

        def forward_argmax(ids):
            for scratch in runner._scratches:
                scratch.attention.use()
            runner._last_logits = np.array([0.0, 1.0], dtype=np.float32)
            return 2

        runner.forward_argmax = forward_argmax
        runner.forward = lambda ids: pytest.fail("public greedy must not call forward")
        runners.append(runner)

        def generate(ids, params):
            assert ids == [1, 2] and params.max_tokens == 2 and params.ignore_eos
            runner.forward_argmax(ids)
            runner.forward_argmax([2])
            return [SimpleNamespace(generated_token_ids=(2, 3))]

        def close():
            for scratch in runner._scratches:
                scratch.attention.close()
            closed.append(runner)

        return SimpleNamespace(generate_detailed=generate, close=close), runner, {}

    monkeypatch.setattr(attention, "Gemma4AttentionScratch", Owner)
    monkeypatch.setattr(bench, "_resolve_generator", resolve)
    monkeypatch.setattr(probe, "_read_runner_logits", lambda runner: runner._last_logits)
    monkeypatch.setattr(hip, "get_hip_runtime", lambda: SimpleNamespace(device_synchronize=lambda: None))
    monkeypatch.setattr(memory, "memory_stats", lambda: {})
    monkeypatch.setattr(memory, "reset_memory_stats", lambda: None)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "prompt_token_ids": [1, 2], "provenance": {"artifact": {"path": "/synthetic.gguf"}},
        "loading": {"runner_capacity": 8}, "workload": {"output_tokens": 2},
        "samples": [{"generated_token_ids": list(expected)}],
    }))
    output = tmp_path / "output.json"
    monkeypatch.setattr(sys, "argv", ["scripts/gemma4_workspace_ownership_probe.py", "--baseline-json", str(baseline), "--out", str(output)])
    return closed, runners, output


def test_public_workspace_probe_compares_full_logits_and_owners(monkeypatch, tmp_path):
    closed, runners, output = _drive(monkeypatch, tmp_path)
    assert probe.main() == 0
    packet = json.loads(output.read_text())
    assert [row["owners"] for row in packet["rows"]] == [2, 1]
    assert [row["owned_bytes"] for row in packet["rows"]] == [256, 128]
    assert all(packet["correctness"].values())
    assert len(packet["rows"][0]["full_logits_hashes"]) == 2
    assert closed == runners
    assert all(s.attention._closed for runner in runners for s in runner._scratches)


def test_public_workspace_probe_closes_on_output_mismatch(monkeypatch, tmp_path):
    closed, runners, output = _drive(monkeypatch, tmp_path, expected=(2, 99))
    with pytest.raises(AssertionError):
        probe.main()
    assert closed == runners
    assert not output.exists()


class Runner:
    def __init__(self):
        self.calls = []

    def forward_argmax(self, token_ids, *, apply_softcap=True):
        self.calls.append((token_ids, apply_softcap))
        return 7


def test_observer_captures_logits_without_changing_greedy_call(monkeypatch):
    runner = Runner()
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    monkeypatch.setattr(probe, '_read_runner_logits', lambda value: logits, raising=False)
    with probe.observe_argmax_logits(runner) as (hashes, finite):
        assert runner.forward_argmax([12, 34], apply_softcap=False) == 7
        assert runner.forward_argmax([7]) == 7
    assert hashes == [hashlib.sha256(logits.tobytes()).hexdigest()] * 2
    assert finite == [True, True]
    assert runner.calls == [([12, 34], False), ([7], True)]
    assert 'forward_argmax' not in runner.__dict__


def test_observer_restores_instance_override_and_records_nonfinite(monkeypatch):
    runner = Runner()
    original = lambda ids: 3
    runner.forward_argmax = original
    monkeypatch.setattr(probe, '_read_runner_logits',
                        lambda value: np.array([np.nan], dtype=np.float32), raising=False)
    with probe.observe_argmax_logits(runner) as (hashes, finite):
        assert runner.forward_argmax([3]) == 3
    assert len(hashes) == 1 and finite == [False]
    assert runner.forward_argmax is original


def test_failed_greedy_call_is_not_captured(monkeypatch):
    runner = Runner()
    def fail(ids):
        raise RuntimeError('greedy failed')
    runner.forward_argmax = fail
    monkeypatch.setattr(probe, '_read_runner_logits',
                        lambda value: pytest.fail('must not capture a failed call'), raising=False)
    with pytest.raises(RuntimeError, match='greedy failed'):
        with probe.observe_argmax_logits(runner) as (hashes, finite):
            runner.forward_argmax([3])
    assert hashes == finite == []
    assert runner.forward_argmax is fail
