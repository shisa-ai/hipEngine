"""Compatibility worker reuse is bounded by one request, including failure."""
from types import SimpleNamespace

import pytest

from hipengine.generation import qwen35_gguf


@pytest.mark.parametrize("fail", [False, True, "cancel", "deadline"])
def test_request_worker_pool_reuses_threads_and_drains_before_return(monkeypatch, fail):
    events = []
    class Pool:
        def __init__(self, *, max_workers):
            events.append(("create", max_workers))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            events.append("shutdown")
    monkeypatch.setattr(qwen35_gguf.concurrent.futures, "ThreadPoolExecutor", Pool)
    owner = object.__new__(qwen35_gguf.Qwen35GGUFBringupGenerator)
    slots = [SimpleNamespace(done=False), SimpleNamespace(done=False)]
    seen = []
    def cycle(slots, *args, **kwargs):
        seen.append(slots[0].worker_pool)
        assert all(slot.worker_pool is seen[0] for slot in slots)
        if len(seen) == 1:
            slots[0].done = True
        else:
            if fail == "cancel":
                from hipengine.generation.deadline import GenerationCancelled
                raise GenerationCancelled("cancelled")
            if fail == "deadline":
                raise TimeoutError("deadline expired")
            if fail:
                raise RuntimeError("worker failed")
            slots[1].done = True
    owner._run_mtp_serving_slots_cycle = cycle
    if fail:
        with pytest.raises(Exception, match="worker failed|cancelled|deadline expired"):
            owner._run_mtp_serving_slots(slots, None, None, base_env={})
    else:
        owner._run_mtp_serving_slots(slots, None, None, base_env={})
    assert len(seen) == 2 and seen[0] is seen[1]
    assert events == [("create", 2), "shutdown"]
    assert all(slot.worker_pool is None for slot in slots)
