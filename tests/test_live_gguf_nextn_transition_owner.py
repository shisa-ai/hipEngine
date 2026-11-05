"""Public dense NextN shared-transition engagement and device commit lifecycle."""
import ctypes
from pathlib import Path

import pytest

MODEL = Path('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')


def _hip_available():
    try:
        ctypes.CDLL('libamdhip64.so')
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.exists()), reason='requires ROCm and dense NextN GGUF')


def test_public_dense_nextn_transition_commit_repeat_and_cancel(monkeypatch):
    from hipengine import LLM, SamplingParams
    from hipengine.core.memory import memory_stats
    from hipengine.core.hip import get_hip_runtime
    from hipengine.generation.deadline import GenerationCancellationToken
    from hipengine.runtime.qwen35_gguf_nextn import Qwen35GGUFNextNExecutor
    from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
    from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner

    plans = []
    frontier_original = Qwen35GGUFMTP2Adapter.execute_target_frontier

    def frontier(self, plan, *args, **kwargs):
        plans.append(plan.execution_route)
        return frontier_original(self, plan, *args, **kwargs)

    monkeypatch.setattr(Qwen35GGUFMTP2Adapter, 'execute_target_frontier', frontier)
    accepts = []
    accept_original = Qwen35GGUFMTP2Adapter._enqueue_target_batch_accept

    def accept(self, *args, **kwargs):
        result = accept_original(self, *args, **kwargs)
        accepts.append(True)
        return result

    monkeypatch.setattr(Qwen35GGUFMTP2Adapter, '_enqueue_target_batch_accept', accept)
    calls = []
    original = Qwen35GGUFNextNExecutor._enqueue_batch_hidden_transition
    cancellation = None

    def transition(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        calls.append(dict(kwargs))
        if cancellation is not None:
            cancellation.cancel()
        return result

    monkeypatch.setattr(Qwen35GGUFNextNExecutor, '_enqueue_batch_hidden_transition', transition)
    before = memory_stats()
    llm = LLM(str(MODEL), backend='hip_gfx1151', max_active_requests=2,
              max_sequence_length=256, prefix_cache='off')
    try:
        params = SamplingParams(max_tokens=12, temperature=0, ignore_eos=True, speculative_mtp=True)
        outputs = []
        for graph in (False, True, True):
            monkeypatch.setattr(Qwen35GGUFResidentModelRunner, 'speculative_graph_available',
                                lambda self, work: graph)
            plans.clear()
            calls.clear()
            accepts.clear()
            output = llm.generate_detailed([[9707] * 17, [9707] * 23], params)[0]
            assert accepts, 'public batch did not enqueue device accept'
            assert plans and set(plans) == ({'graph'} if graph else {'eager'})
            assert any(options.get('synchronize') is False for options in calls)
            assert calls, 'public NextN request did not execute the shared transition'
            assert len(output.generated_token_ids) == 12
            outputs.append(tuple(output.generated_token_ids))
        assert outputs[0] == outputs[1] == outputs[2]
        cancellation = GenerationCancellationToken()
        calls.clear()
        cancelled = llm.generate_detailed([[9707] * 17, [9707] * 23], SamplingParams(
            max_tokens=12, temperature=0, ignore_eos=True, speculative_mtp=True,
            cancellation_token=cancellation))[0]
        assert calls and cancellation.cancel_requested
        assert len(cancelled.generated_token_ids) < 12
        cancellation = None
        recovered = llm.generate_detailed([[9707] * 17, [9707] * 23], params)[0]
        assert tuple(recovered.generated_token_ids) == outputs[0]
    finally:
        llm.close()
    get_hip_runtime().device_synchronize()
    after = memory_stats()
    for key in ('active_allocations', 'current_allocated_bytes'):
        assert after[key] == before[key]
