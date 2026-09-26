"""Public GGUF serving ownership checks with a small local model.

These short requests check allocation, cancellation and refill, not completion
at the configured context depth or throughput. Missing weights or HIP skip.
"""
from pathlib import Path
import time

import pytest

from scripts.gguf_packed_lease_probe import _pool_summary, _resident_runner


@pytest.mark.parametrize("capacity,context", [(1, 511), (4, 1536), (4, 5379)])
def test_public_generation_capacity_cancel_refill(hip_test_target_arch, capacity, context):
    from hipengine import LLM, SamplingParams
    from hipengine.core.memory import memory_stats
    from hipengine.kernels.backends import HIP_TARGET_ARCH_BACKEND

    model = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
    if not model.is_file():
        pytest.skip(f"local small GGUF fixture unavailable: {model}")
    baseline = memory_stats()
    llm = LLM(
        str(model), backend=HIP_TARGET_ARCH_BACKEND[hip_test_target_arch],
        execution_profile="production", max_active_requests=capacity,
        max_sequence_length=context, kv_storage="bf16",
        speculative_mtp_serving="off", prefix_cache="off",
    )
    try:
        sampling = SamplingParams(max_tokens=4, temperature=0)
        outputs = llm.generate(["Say hello."] * capacity, sampling)
        assert len(outputs) == capacity
        runner = _resident_runner(llm)
        estimate = runner.generator._resident_capacity_estimate(
            runner._shared_runner, max_batch_size=capacity,
            defer_kv_allocation=True, requested_context_tokens=context,
        )
        initial = _pool_summary(runner)
        assert estimate.workspace_lease_pages == initial["workspace_lease_pages"]
        assert initial["route_counts"], "generation must exercise a resident route"
        health = llm.engine_service_health()
        assert health is not None and health["status"] == "ok", health

        stream = llm.stream_detailed(
            "Count from one to one hundred.",
            SamplingParams(max_tokens=64, temperature=0),
        )
        try:
            first = next(stream)
            assert first.finish_details is None, "stream finished before cancellation"
            assert runner._rows, "stream must still own a resident row before cancellation"
        finally:
            stream.close()
        # Cancellation crosses the engine-service queue. Wait only for its
        # acknowledgement, not an arbitrary sleep before examining ownership.
        deadline = time.monotonic() + 10
        while runner._rows and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not runner._rows, "cancelled request still owns a resident row"
        assert _pool_summary(runner)["refcounted_pages"] == initial["refcounted_pages"]

        refilled = llm.generate(["Say hello."] * capacity, sampling)
        assert refilled == outputs
        after = _pool_summary(runner)
        assert after["workspace_lease_pages"] == initial["workspace_lease_pages"]
        assert after["refcounted_pages"] == initial["refcounted_pages"]
        assert llm.engine_service_health()["status"] == "ok"
        print({"capacity": capacity, "context": context, "before": initial, "after": after})
    finally:
        llm.close()
    final = memory_stats()
    assert final["active_allocations"] == baseline["active_allocations"]
    assert final["current_allocated_bytes"] == baseline["current_allocated_bytes"]
