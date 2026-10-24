"""Public PARO depth plans, output tails, reuse and interrupted-cycle ownership."""
from contextlib import closing
from dataclasses import replace
import os
from pathlib import Path

import pytest

from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
from hipengine.generation.deadline import GenerationCancellationToken
from hipengine.generation.qwen35_paro import Qwen35ParoResidentModelRunner
from hipengine.generation.qwen35_paro_mtp2 import Qwen35ParoMTP2Adapter
from test_live_paro_mtp_rollback import _hip_available, _snapshot
from test_live_paro_mtp_graph import _routes

pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is unavailable")


@pytest.fixture(scope="module")
def llm():
    model = Path(os.environ.get("HIPENGINE_TEST_PARO_MTP_MODEL", "/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16"))
    if not model.is_dir():
        pytest.skip("packed PARO model with BF16 MTP weights is unavailable")
    before = memory_stats()
    with closing(LLM(str(model), backend="auto", quant="w4_paro", max_active_requests=1,
                     max_sequence_length=4352, prefix_cache="off",
                     speculative_candidate_budget=3)) as owner:
        yield owner
    after = memory_stats()
    assert after["active_allocations"] == before["active_allocations"]
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]


def _params(count):
    return SamplingParams(max_tokens=count, temperature=0, ignore_eos=True,
                          speculative_mtp=True, kv_storage="bf16")


def _completed(llm):
    routes = _routes(llm.live_loop_snapshot())
    assert routes is not None
    assert not any(k.startswith("specdec2_mtp2_recovered") for k in routes["fallback_reasons"])
    return routes["recent_completed"][-1]


@pytest.mark.parametrize("length", [37, 255, 257, 511, 513, 1025, 4096])
def test_depth_plans_repeat_state_and_survive_wider_then_narrower_reuse(llm, monkeypatch, length):
    desired = 1
    monkeypatch.setattr(Qwen35ParoResidentModelRunner, "speculative_desired_candidate_count",
                        lambda self, request: desired)
    original = Qwen35ParoMTP2Adapter.execute_target_frontier
    states = []

    def cycle(adapter, plan, *args, **kwargs):
        result = original(adapter, plan, *args, **kwargs)
        state = adapter._states[plan.request_ids[0]]
        snapshot = _snapshot(adapter.owner._session, state.proposer)
        for name in ("position_buf", "context_buf", "token_id_buf"):
            snapshot.pop(name)
        states.append(snapshot)
        return result

    monkeypatch.setattr(Qwen35ParoMTP2Adapter, "execute_target_frontier", cycle)
    reference = {}
    for desired in (1, 2, 3, 2, 1, 3):
        states.clear()
        output = llm.generate_detailed([[9707] * length], _params(17))[0]
        route = _completed(llm)
        assert len(output.generated_token_ids) == 17
        assert route["specdec2_mtp2_used"]
        counts = route["specdec2_mtp2_candidate_counts"]
        assert max(counts) == desired and all(1 <= n <= desired for n in counts)
        assert set(route["specdec2_mtp2_execution_routes"]) == {"eager"}
        value = (tuple(output.generated_token_ids), list(states), counts,
                 route["specdec2_mtp2_accepted_counts"])
        if desired in reference:
            assert value == reference[desired]
        else:
            reference[desired] = value


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("cap", [1, 2, 3, 4, 5, 7, 9])
def test_output_cap_tails_are_exact(llm, monkeypatch, depth, cap):
    monkeypatch.setattr(Qwen35ParoResidentModelRunner, "speculative_desired_candidate_count",
                        lambda self, request: depth if request.max_tokens > 1 else 0)
    results = []
    for _ in range(2):
        output = llm.generate_detailed([[9707] * 37], _params(cap))[0]
        assert len(output.generated_token_ids) == cap
        route = _completed(llm)
        counts = route["specdec2_mtp2_candidate_counts"]
        assert all(1 <= n <= depth for n in counts)
        results.append((output.generated_token_ids, counts, route["specdec2_mtp2_accepted_counts"]))
    assert results[0] == results[1]


@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("length", [37, 1025])
@pytest.mark.parametrize("action", ["failure", "cancel"])
def test_deep_cycle_failure_and_cancel_restore_state_and_next_request(llm, monkeypatch, depth, length, action):
    monkeypatch.setattr(Qwen35ParoResidentModelRunner, "speculative_desired_candidate_count",
                        lambda self, request: depth)
    params = _params(17)
    baseline = llm.generate_detailed([[9707] * length], params)[0]
    token = GenerationCancellationToken()
    capture_original = Qwen35ParoMTP2Adapter._checkpoint_target
    propose_original = Qwen35ParoMTP2Adapter.propose_batch
    restore_original = Qwen35ParoMTP2Adapter._restore_target_and_provider
    verify_original = Qwen35ParoMTP2Adapter._verify_target
    pending, injected, restored, mismatches = {}, [], [], []
    calls = 0

    def propose(adapter, plan, *args, **kwargs):
        state = adapter._states[plan.request_ids[0]]
        before = _snapshot(adapter.owner._session, state.proposer)
        pending["provider"] = {key: before[key] for key in (
            "draft_cursor", "draft_position", "final_hidden_buf",
            "key_cache_buf", "value_cache_buf",
        )}
        return propose_original(adapter, plan, *args, **kwargs)

    def capture(adapter, state, session):
        capture_original(adapter, state, session)
        pending["state"] = _snapshot(session, state.proposer)
        # Provider rollback starts before drafting; the target checkpoint is
        # taken afterwards, including its freshly copied spare slot.
        pending["state"].update(pending["provider"])
        pending["prefix"] = tuple(step.token_id for step in adapter.owner._row(state.request_id).generated_steps)

    def restore(adapter, state, session):
        active = state.journal is not None
        restore_original(adapter, state, session)
        if active and injected and not restored:
            after = _snapshot(session, state.proposer)
            mismatches.extend((key, value, after[key]) for key, value in pending["state"].items()
                              if after[key] != value)
            assert not state.proposer.result_valid
            restored.append(True)

    def verify(*args, **kwargs):
        nonlocal calls
        result = verify_original(*args, **kwargs)
        calls += 1
        if calls == 2:
            injected.append(pending["prefix"])
            if action == "failure":
                raise RuntimeError("injected after deep PARO verification")
            token.cancel()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Qwen35ParoMTP2Adapter, "propose_batch", propose)
        patch.setattr(Qwen35ParoMTP2Adapter, "_checkpoint_target", capture)
        patch.setattr(Qwen35ParoMTP2Adapter, "_restore_target_and_provider", restore)
        patch.setattr(Qwen35ParoMTP2Adapter, "_verify_target", staticmethod(verify))
        output = llm.generate_detailed([[9707] * length], replace(params, cancellation_token=token))[0]
    assert len(injected) == 1 and injected[0] and restored == [True]
    assert mismatches == []
    if action == "cancel":
        assert output.finish_details.cancelled
        assert tuple(output.generated_token_ids) == injected[0]
    else:
        assert len(output.generated_token_ids) == 17
    recovered = llm.generate_detailed([[9707] * length], params)[0]
    assert recovered.generated_token_ids == baseline.generated_token_ids
    assert _routes(llm.live_loop_snapshot())["recent_completed"][-1]["specdec2_mtp2_used"]
