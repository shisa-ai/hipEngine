"""Public requests preserve constructor-bound KV reset despite ambient changes."""
import numpy as np
import pytest

from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason="requires ROCm and Flash-Next weights",
)


def test_public_reset_policy_is_frozen_and_opposing_environment_is_inert(monkeypatch):
    from hipengine import LLM, SamplingParams
    from hipengine.core.memory import memory_stats
    from hipengine.core.hip import get_hip_runtime
    from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner

    before = memory_stats()
    captures, resets = [], []
    _install_logit_capture(monkeypatch, captures)
    original = Qwen4ExpGGUFResidentModelRunner.reset

    def reset(owner):
        resets.append(owner.bounded_kv_reset)
        return original(owner)

    monkeypatch.setattr(Qwen4ExpGGUFResidentModelRunner, "reset", reset)
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_BOUNDED_KV_RESET", "0")
    llm = LLM(str(MODEL), backend="hip_gfx1151", execution_profile="production",
              max_sequence_length=CONTEXT, prefix_cache="off")
    try:
        runner = _unwrap(llm._get_text_generator()).runner
        assert runner.bounded_kv_reset is False
        results = []
        for value in ("1", "0", "1"):
            monkeypatch.setenv("HIPENGINE_QWEN4_EXP_BOUNDED_KV_RESET", value)
            captures.clear()
            resets.clear()
            output = llm.generate_detailed([[1] * 73], SamplingParams(
                max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False,
            ))
            assert resets and not any(resets)
            assert captures
            results.append((tuple(output[0].generated_token_ids), [x.copy() for x in captures]))
        for output, rows in results[1:]:
            assert output == results[0][0]
            assert len(rows) == len(results[0][1])
            for actual, expected in zip(rows, results[0][1], strict=True):
                np.testing.assert_array_equal(actual, expected)
    finally:
        llm.close()
    get_hip_runtime().device_synchronize()
    after = memory_stats()
    for key in ("active_allocations", "current_allocated_bytes"):
        assert after[key] == before[key]
