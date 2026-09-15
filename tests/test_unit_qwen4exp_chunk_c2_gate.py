from types import SimpleNamespace

import pytest

from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
from scripts import qwen4exp_chunk_c2_gate as gate
from tests.test_unit_qwen4_exp_resident_serving import _Tokenizer


class Runner:
    max_sequence_length = 4352
    prefill_chunk_size = 2048
    runtime = None

    def __init__(self):
        self.closed = False
        self.position = 0
        self.history = ()
        self.calls = []

    def reset(self):
        self.position = 0
        self.history = ()

    def prefill(self, tokens, **kwargs):
        assert kwargs == {"capture_logits": False, "capture_target_hidden": False}
        self.history = tuple(tokens)
        self.position = len(tokens)
        self.next_token = sum(tokens) % 50
        self.calls.append("prefill")
        return SimpleNamespace(token_id=self.next_token)

    def step(self, token, **kwargs):
        assert kwargs == {"capture_logits": False, "capture_target_hidden": False,
                          "token_id_resident": True}
        self.history += (token,)
        self.position += 1
        self.next_token = token + 1
        self.calls.append("step")
        return SimpleNamespace(token_id=self.next_token)

    def close(self):
        self.closed = True


def pool(monkeypatch):
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused.gguf", weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(), tokenizer=_Tokenizer(), runner=Runner())
    generator._resident = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner",
        lambda *args, **kwargs: Runner())
    return generator.create_resident_model_runner(capacity=2)


PROMPTS = {"a": [1] * 2052, "b": [2] * 4097, "c": [3] * 2049}


def capture(runner):
    return {"next_token": runner.next_token, "state": (runner.position, runner.history)}


def test_c2_sequence_uses_real_pool_and_keeps_compact_calls(monkeypatch):
    native = pool(monkeypatch)
    try:
        result = gate.exercise(native, PROMPTS, capture_fn=capture)
        assert len(result["repeats"]) == 3
        assert all(row["checkpoints"] == 28 for row in result["repeats"])
        assert {row["owners"]["a"] for row in result["repeats"]} == {0, 1}
        assert not native.active_request_ids
        assert sum(r.calls.count("prefill") for r in native._all_runners) == 12
    finally:
        native.close()


def test_c2_sequence_rejects_peer_dependent_state(monkeypatch):
    native = pool(monkeypatch)

    def contaminated(runner):
        result = capture(runner)
        if len(native.active_request_ids) == 2:
            result["state"] = "corrupted"
        return result

    try:
        with pytest.raises(ValueError, match="differs"):
            gate.exercise(native, PROMPTS, capture_fn=contaminated)
    finally:
        native.close()


def test_deferred_inspection_preserves_interleaving_without_checkpoint_reads(monkeypatch):
    native = pool(monkeypatch)
    inspections = []

    def counted(runner):
        inspections.append(len(native.active_request_ids))
        return capture(runner)

    try:
        result = gate.exercise(native, PROMPTS, capture_fn=counted, checkpoint_each=False)
        assert [row["checkpoints"] for row in result["repeats"]] == [2, 2, 2]
        assert len(inspections) == 27 + 6
        assert set(inspections) == {1}
    finally:
        native.close()


def test_deferred_inspection_rejects_bad_final_state(monkeypatch):
    native = pool(monkeypatch)
    calls = 0

    def bad(runner):
        nonlocal calls
        calls += 1
        result = capture(runner)
        if calls > 27:
            result["state"] = "corrupted"
        return result

    try:
        with pytest.raises(ValueError, match="deferred c2 state"):
            gate.exercise(native, PROMPTS, capture_fn=bad, checkpoint_each=False)
    finally:
        native.close()


def test_disjoint_ranges_allows_adjacency_but_not_partial_overlap():
    assert gate.disjoint([(10, 20)], [(20, 30)])
    assert not gate.disjoint([(10, 20)], [(19, 30)])
    assert not gate.disjoint([(10, 20)], [(12, 15)])


@pytest.mark.parametrize("chunk,expected_calls", [(2048, 28), (4096, 16)])
def test_trace_gate_rejects_prefill_for_cancelled_partial_request(chunk, expected_calls):
    expected = {1: 2052, 2: 4097, 3: 2049}
    for repeat in range(3):
        base = 100 + repeat * 10
        expected.update({base: 2052, base + 1: 4097, base + 3: 2049})
    traces = []
    for rid, length in expected.items():
        count, tail = divmod(length, chunk)
        traces.extend({"request_id": rid, "rows": size}
                      for size in [chunk] * count + ([tail] if tail else []))
    assert gate.validate_traces(traces, PROMPTS, chunk=chunk)["chunk_calls"] == expected_calls
    with pytest.raises(ValueError, match="request set"):
        gate.validate_traces(traces + [{"request_id": 102, "rows": 1024}], PROMPTS, chunk=chunk)
    traces[0]["rows"] = 1024
    with pytest.raises(ValueError, match="coverage"):
        gate.validate_traces(traces, PROMPTS, chunk=chunk)
