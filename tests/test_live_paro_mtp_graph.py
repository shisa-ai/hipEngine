"""PARO graph plans match the faster eager default and survive replay/reuse."""
from contextlib import closing
import os
from pathlib import Path

import pytest

from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
from hipengine.generation.deadline import GenerationCancellationToken
from hipengine.generation.qwen35_paro import Qwen35ParoResidentModelRunner
from hipengine.generation.qwen35_paro_mtp2 import Qwen35ParoMTP2Adapter
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession
from test_live_paro_mtp_rollback import _hip_available, _snapshot

pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is unavailable")


def _routes(value):
    if isinstance(value, dict):
        if "routes" in value and "recent_completed" in value["routes"]:
            return value["routes"]
        for child in value.values():
            found = _routes(child)
            if found is not None:
                return found
    return None


@pytest.fixture(scope="module")
def llm():
    model = Path(os.environ.get("HIPENGINE_TEST_PARO_MTP_MODEL", "/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16"))
    if not model.is_dir():
        pytest.skip("packed PARO model with BF16 MTP weights is unavailable")
    before = memory_stats()
    with closing(LLM(str(model), backend="auto", quant="w4_paro", max_active_requests=1,
                     max_sequence_length=4352, prefix_cache="off")) as owner:
        yield owner
    after = memory_stats()
    assert after["active_allocations"] == before["active_allocations"]
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]


@pytest.mark.parametrize("length", [37, 255, 257, 511, 513, 1025, 4096])
def test_graph_plan_matches_default_live_state_outputs_and_repeated_request(llm, monkeypatch, length):
    original_verify = Qwen35ParoResidentSession.verify_chain_bulk_and_commit
    original_cycle = Qwen35ParoMTP2Adapter.execute_target_frontier
    states = []
    graph_results = []
    cleanups = []
    eager = True

    def verify(session, *args, **kwargs):
        assert kwargs["graph_mode"] == ("off" if eager else "auto")
        result = original_verify(session, *args, **kwargs)
        graph_results.append(dict(result.graph))
        cleanups.append(dict(session.verify_graph_cleanup()))
        return result

    def cycle(adapter, plan, *args, **kwargs):
        result = original_cycle(adapter, plan, *args, **kwargs)
        state = adapter._states[plan.request_ids[0]]
        snapshot = _snapshot(adapter.owner._session, state.proposer)
        # These are per-launch input buffers, not the committed logical cursor.
        # Rollback checks separately require their pre/post failure equality.
        for name in ("position_buf", "context_buf", "token_id_buf"):
            snapshot.pop(name)
        states.append(snapshot)
        return result

    monkeypatch.setattr(Qwen35ParoResidentSession, "verify_chain_bulk_and_commit", verify)
    monkeypatch.setattr(Qwen35ParoMTP2Adapter, "execute_target_frontier", cycle)
    params = SamplingParams(max_tokens=32, temperature=0, ignore_eos=True,
                            speculative_mtp=True, kv_storage="bf16")
    runs = []
    for arm in ("default", "graph", "graph_repeat"):
        eager = arm == "default"
        if not eager:
            monkeypatch.setattr(Qwen35ParoResidentModelRunner, "speculative_graph_available",
                                lambda self, work: True)
        states.clear()
        graph_results.clear()
        cleanups.clear()
        output = llm.generate_detailed([[9707] * length], params)[0]
        routes = _routes(llm.live_loop_snapshot())
        assert routes is not None
        route = routes["recent_completed"][-1]
        assert route["specdec2_mtp2_used"] and route["specdec2_mtp2_cycles"] > 0
        assert not any(key.startswith("specdec2_mtp2_recovered") for key in routes["fallback_reasons"])
        assert len(output.generated_token_ids) == 32
        if not eager:
            assert any(value.get("status") == "replayed" for value in graph_results), graph_results
            assert "graph" in route["specdec2_mtp2_execution_routes"], route
            # A replayed cycle must not leave a graph the runtime refused to
            # release, and the retained graph is what makes the next cycle cheap.
            assert cleanups, "graph route reported without an observed cache"
            assert not any(item["pending_close"] for item in cleanups), cleanups
            assert not any(item["pending_partial_handles"] for item in cleanups), cleanups
            assert max(item["reusable"] for item in cleanups) >= 1, cleanups
        else:
            assert set(route["specdec2_mtp2_execution_routes"]) == {"eager"}
        runs.append((tuple(output.generated_token_ids), list(states), route["specdec2_mtp2_accepted_counts"]))
    assert runs[0] == runs[1] == runs[2]


@pytest.mark.parametrize("length", [37, 1025])
@pytest.mark.parametrize("action", ["failure", "cancel"])
def test_replayed_cycle_restores_state_and_next_request(llm, monkeypatch, length, action):
    from dataclasses import replace

    params = SamplingParams(max_tokens=16, temperature=0, ignore_eos=True,
                            speculative_mtp=True, kv_storage="bf16")
    baseline = llm.generate_detailed([[9707] * length], params)[0]
    monkeypatch.setattr(Qwen35ParoResidentModelRunner, "speculative_graph_available",
                        lambda self, work: True)
    token = GenerationCancellationToken()
    capture_original = Qwen35ParoMTP2Adapter._checkpoint_target
    restore_original = Qwen35ParoMTP2Adapter._restore_target_and_provider
    verify_original = Qwen35ParoResidentSession.verify_chain_bulk_and_commit
    pending = {}
    injected = []
    restored = []

    def capture(adapter, state, session):
        capture_original(adapter, state, session)
        pending["state"] = _snapshot(session, state.proposer)
        row = adapter.owner._row(state.request_id)
        pending["prefix"] = tuple(step.token_id for step in row.generated_steps)

    def restore(adapter, state, session):
        active = state.journal is not None
        restore_original(adapter, state, session)
        if active and injected and not restored:
            assert _snapshot(session, state.proposer) == pending["state"]
            assert not state.proposer.result_valid
            restored.append(True)

    def verify(session, *args, **kwargs):
        result = verify_original(session, *args, **kwargs)
        if result.graph.get("status") == "replayed" and not injected:
            injected.append(tuple(pending["prefix"]))
            if action == "failure":
                raise RuntimeError("injected after PARO graph replay")
            token.cancel()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Qwen35ParoMTP2Adapter, "_checkpoint_target", capture)
        patch.setattr(Qwen35ParoMTP2Adapter, "_restore_target_and_provider", restore)
        patch.setattr(Qwen35ParoResidentSession, "verify_chain_bulk_and_commit", verify)
        output = llm.generate_detailed([[9707] * length], replace(params, cancellation_token=token))[0]
    assert len(injected) == 1 and restored == [True]
    assert injected[0], "cancellation/failure must follow at least one committed cycle"
    if action == "cancel":
        assert output.finish_details.cancelled
        assert tuple(output.generated_token_ids) == injected[0]
    else:
        assert len(output.generated_token_ids) == 16
    recovered = llm.generate_detailed([[9707] * length], params)[0]
    assert recovered.generated_token_ids == baseline.generated_token_ids
    assert _routes(llm.live_loop_snapshot())["recent_completed"][-1]["specdec2_mtp2_used"]
