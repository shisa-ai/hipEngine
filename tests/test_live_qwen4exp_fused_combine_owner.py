"""Public explicit fused-combine comparison, engagement, repeats and cleanup."""
from contextlib import ExitStack
import numpy as np
import pytest
from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason="requires ROCm and Qwen4Exp GGUF")


def test_public_fused_combine_owner(monkeypatch):
    from hipengine import LLM, SamplingParams
    from hipengine.runtime import qwen4_exp_runner as runtime
    from hipengine.core.memory import memory_stats
    from hipengine.core.hip import get_hip_runtime

    before = memory_stats()
    captures, calls = [], []
    _install_logit_capture(monkeypatch, captures)
    original = runtime.weighted_lanes_sum_shared_gate_combine_batch_out_bf16_f32w
    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(runtime, "weighted_lanes_sum_shared_gate_combine_batch_out_bf16_f32w", counted)
    llm = LLM(str(MODEL), backend="hip_gfx1151", execution_profile="production",
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache="off")
    try:
        owner = _unwrap(llm._get_text_generator())
        serving = owner.create_resident_model_runner(capacity=1)
        serving.prepare(max_sequence_length=CONTEXT)
        runners = list(serving._all_runners)
        if all(owner.runner is not r for r in runners):
            runners.append(owner.runner)
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
        for length in (1, 17, 73):
            results = []
            for enabled in (False, True, True):
                monkeypatch.setenv("HIPENGINE_QWEN4_EXP_FUSED_COMBINE", "0" if enabled else "1")
                with ExitStack() as stack:
                    for target in [owner, *runners]:
                        stack.enter_context(runtime.fused_combine_enabled(target, enabled))
                        stack.enter_context(runtime.routed_f16_enabled(target, False))
                    calls.clear()
                    captures.clear()
                    output = llm.generate([list(range(100, 100 + length))], params)
                    assert bool(calls) is (enabled and length > 1)
                    results.append((output, [x.copy() for x in captures]))
            for output, rows in results[1:]:
                assert output == results[0][0]
                assert rows and len(rows) == len(results[0][1])
                for actual, expected in zip(rows, results[0][1], strict=True):
                    assert np.isfinite(actual).all()
                    np.testing.assert_array_equal(actual, expected)
    finally:
        llm.close()
    get_hip_runtime().device_synchronize()
    after = memory_stats()
    for key in ("active_allocations", "current_allocated_bytes"):
        assert after[key] == before[key]
