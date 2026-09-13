"""The publication sweep must use public graph admission and discard warmup."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import qwen38_gfx1151_readme_sweep as sweep


def test_workload_discards_warmup_and_closes_graphs(monkeypatch):
    calls, closed = [], []
    monkeypatch.setattr(sweep, "_default_decode_graph_request", lambda session, steps: True)

    def run_once(**kwargs):
        calls.append(kwargs)
        if kwargs["graph_holder"] is not None:
            kwargs["graph_holder"]["graph"] = SimpleNamespace(close=lambda: closed.append(1))
        return {"measured": kwargs["measured"]}

    monkeypatch.setattr(sweep, "_run_existing_session_once", run_once)
    args = SimpleNamespace(model="model.gguf", decode_tokens=128,
                           warmups=1, repetitions=3)
    session = SimpleNamespace(runtime=object())
    measured = sweep.run_workload(session, args, 512)
    assert len(calls) == 4
    assert len(measured) == 3
    assert all(row["measured"] for row in measured)
    assert len(closed) == 3
    assert [call["graph_replay_decode"] for call in calls] == [False, True, True, True]


@pytest.mark.parametrize("graph_ids,passed", [([2, 3], True), ([2, 4], False)])
def test_graph_gate_checks_all_generated_ids(monkeypatch, graph_ids, passed):
    monkeypatch.setattr(sweep, "_run_logits_trajectory", lambda *a, **k: [
        {"token_id": token, "logits": np.array([1.0, 2.0])} for token in [1, 2, 3]])
    monkeypatch.setattr(sweep, "_state_summary", lambda *a: {"finite": True, "state_sha256": "same"})
    graph = SimpleNamespace(
        replay=lambda steps: None,
        read_generated_token_ids=lambda count: graph_ids,
        read_sample=lambda: SimpleNamespace(token_id=3, logits=np.array([1.0, 2.0])),
        close=lambda: None)
    session = SimpleNamespace(
        reset=lambda: None, position=2,
        prefill=lambda *a, **k: SimpleNamespace(token_id=1),
        capture_decode_graph=lambda **kwargs: graph)
    result = sweep.verify_graph_prompt(session, [10, 11], steps=2)
    assert result["passed"] is passed
