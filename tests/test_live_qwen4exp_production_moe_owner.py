"""Grouped-MoE owner choice ignores retired ambient input on public generation."""
from contextlib import ExitStack
import os

import numpy as np
import pytest

from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('length', [1, 17, 73])
def test_public_grouped_moe_owner(monkeypatch, length):
    from hipengine import LLM, SamplingParams
    from hipengine.runtime import qwen4_exp_runner as runtime
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats

    hip = get_hip_runtime()
    hip.device_synchronize()
    before = memory_stats()
    captures, selected, compact = [], [], []
    _install_logit_capture(monkeypatch, captures)
    original_selector = runtime._qwen4_exp_production_moe_prefill_enabled
    original_compact = runtime.gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out

    def selector(weight, *, rows, enabled=None):
        result = original_selector(weight, rows=rows, enabled=enabled)
        selected.append((rows, enabled, result))
        return result

    def counted(*args, **kwargs):
        compact.append(True)
        return original_compact(*args, **kwargs)

    monkeypatch.setattr(runtime, '_qwen4_exp_production_moe_prefill_enabled', selector)
    monkeypatch.setattr(runtime, 'gguf_q4_k_selected_dual_wmma_prefill_compact_bf16_bf16_out', counted)
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    try:
        owner = _unwrap(llm._get_text_generator())
        serving = owner.create_resident_model_runner(capacity=1)
        serving.prepare(max_sequence_length=CONTEXT)
        runners = list(serving._all_runners)
        if all(owner.runner is not r for r in runners):
            runners.append(owner.runner)
        targets = (owner, *runners)
        prior = [(r, r.production_moe_prefill_enabled) for r in targets]
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
        for enabled in (False, True):
            ambient = '0' if enabled else '1'
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL', ambient)
            with ExitStack() as scopes:
                for target in targets:
                    scopes.enter_context(runtime.production_moe_prefill_enabled(target, enabled))
                    scopes.enter_context(runtime.routed_f16_enabled(target, False))
                    scopes.enter_context(runtime.q4_exact_grouped_gate(target, False))
                    scopes.enter_context(runtime.q4_iu8_prefill_enabled(target, False))
                    # Q5_K exact selection also prepares the shared grouped path;
                    # clear it to isolate the production grouped-MoE selector.
                    scopes.enter_context(runtime.q5k_iu8_exact_enabled(target, False))
                outputs, logits = [], []
                for _ in range(2):
                    selected.clear()
                    compact.clear()
                    start = len(captures)
                    outputs.append(llm.generate([list(range(100, 100 + length))], params))
                    assert selected
                    assert all(choice is enabled for _, choice, _ in selected)
                    assert any(result for _, _, result in selected) is (enabled and length >= 16)
                    assert bool(compact) is (enabled and length >= 16)
                    logits.append([x.copy() for x in captures[start:]])
                assert outputs[0] == outputs[1]
                assert logits[0] and len(logits[0]) == len(logits[1])
                for a, b in zip(*logits):
                    assert np.isfinite(a).all()
                    np.testing.assert_array_equal(a, b)
                assert os.environ['HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL'] == ambient
            for target, value in prior:
                assert target.production_moe_prefill_enabled is value
    finally:
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    assert after['active_allocations'] == before['active_allocations']
    assert after['current_allocated_bytes'] == before['current_allocated_bytes']
