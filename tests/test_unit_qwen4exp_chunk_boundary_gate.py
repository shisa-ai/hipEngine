from types import SimpleNamespace

import numpy as np
import pytest

from scripts import qwen4exp_chunk_boundary_gate as gate


def test_boundary_matrix_preserves_source_and_covers_both_transitions():
    fixture = {"cases": [
        {"id": f"{category}-p4096", "prompt_token_ids": list(range(4096))}
        for category in ("code", "general_ja")]}
    cases = gate.boundary_cases(fixture)
    assert len(cases) == 14
    assert {case["prompt_tokens"] for case in cases} == set(gate.LENGTHS)
    for case in cases:
        assert len(case["prompt_token_ids"]) == case["prompt_tokens"]
    assert cases[-1]["prompt_token_ids"][-2:] == [4095, 4095]
    assert len(fixture["cases"][0]["prompt_token_ids"]) == 4096
    fixture["cases"][0]["prompt_token_ids"].pop()
    with pytest.raises(ValueError, match="exactly4096"):
        gate.boundary_cases(fixture)


def test_trajectory_reconstructs_prefix_and_forces_reference_tokens(monkeypatch):
    calls = []

    class Runner:
        def _prefill_chunk(self, values):
            calls.append(("chunk", list(values)))

        def prefill(self, prompt):
            self._prefill_chunk(prompt[:2])
            self._prefill_chunk(prompt[2:])
            return SimpleNamespace(logits=np.array([1., 2.]), token_id=1)

        def step(self, token):
            calls.append(("step", token))
            return SimpleNamespace(logits=np.array([3., 4.]), token_id=1)

    monkeypatch.setattr(gate, "full_state", lambda runner: {"payload": "hash"})
    logits, tokens, payload = gate.trajectory(
        Runner(), [4, 5, 6], steps=2, chunk=2, teacher=[8, 9, 10])
    assert calls == [("chunk", [4, 5]), ("chunk", [6]), ("step", 8), ("step", 9)]
    assert logits.shape == (3, 2)
    assert tokens == [1, 1, 1]
    assert payload["chunks"] == [2, 1]


def test_full_state_hashes_live_index_rows_in_logical_order(monkeypatch):
    import ctypes
    from hipengine.core import memory

    def buffer(values):
        array = np.array(values, dtype=np.float32)
        return SimpleNamespace(host=array, nbytes=array.nbytes)

    raw = buffer([[1, 2], [3, 4], [5, 6], [7, 8]])
    pooled = buffer([[9, 10], [11, 12]])
    index = SimpleNamespace(
        raw_keys=raw, pooled_keys=pooled, capacity=4, index_dim=2,
        physical_positions_host=np.array([2, 0, 3, 1]), count=2, pooled_count=1)
    runtime = SimpleNamespace(device_synchronize=lambda: None)
    runner = SimpleNamespace(runtime=runtime, attention_states=[], index_states=[index])
    monkeypatch.setattr(memory, "copy_device_to_host",
                        lambda dest, buf, **kwargs: ctypes.memmove(dest, buf.host.ctypes.data, buf.nbytes))
    monkeypatch.setattr(gate, "_state_summary", lambda runner: {"finite": True})
    first = gate.full_state(runner)
    assert first["live_index_bytes"] == 24
    raw.host[1, 0] = np.nan
    pooled.host[1, 0] = np.nan
    assert gate.full_state(runner) == first
    raw.host[2, 0] = np.nan
    changed = gate.full_state(runner)
    assert not changed["live_index_finite"]
    assert changed["live_index_sha256"] != first["live_index_sha256"]
    kv = SimpleNamespace(host=np.array([0x3F80, 0x7F80], dtype=np.uint16), nbytes=4)
    runner.attention_states = [SimpleNamespace(key_cache=kv, value_cache=kv)]
    assert not gate.full_state(runner)["full_kv_finite"]
    kv.host[1] = 0
    assert gate.full_state(runner)["full_kv_finite"]


def test_control_gate_does_not_require_arithmetic_hash_identity():
    before = dict(recurrent={"state_sha256": "a", "finite": True, "position": 8},
                  full_kv_bytes=16, live_index_bytes=32)
    after = {**before, "recurrent": {**before["recurrent"], "state_sha256": "b"}}
    assert gate.control_state(before) == gate.control_state(after)
    after["recurrent"]["position"] = 9
    assert gate.control_state(before) != gate.control_state(after)
