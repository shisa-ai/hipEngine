"""The publication sweep must use public graph admission and discard warmup."""

from types import SimpleNamespace

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
