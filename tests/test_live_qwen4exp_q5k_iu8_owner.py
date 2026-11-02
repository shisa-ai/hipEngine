"""Public Q5_K exact IU8 owner selection, repeats and teardown."""
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
@pytest.mark.parametrize('interface', ['owner', 'gap', 'ar', 'campaign', 'state', 'mapped_preemptor'])
def test_public_q5k_iu8_owner(monkeypatch, length, interface):
    from hipengine import LLM, SamplingParams
    from hipengine.runtime import qwen4_exp_runner as runtime
    from hipengine.kernels.registry import KernelKey, register, resolve
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    hip = get_hip_runtime()
    hip.device_synchronize()
    before = memory_stats()
    captures, calls = [], []
    _install_logit_capture(monkeypatch, captures)
    key = KernelKey('hip_gfx1151', 'moe_linear', 'gguf_q5_k',
                    'selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out')
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    original = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    register(key, counted, replace=True)
    try:
        owner = _unwrap(llm._get_text_generator())
        serving = owner.create_resident_model_runner(capacity=1)
        serving.prepare(max_sequence_length=CONTEXT)
        runners = list(serving._all_runners)
        if all(owner.runner is not r for r in runners):
            runners.append(owner.runner)
        prior = [(r, r.q5k_iu8_exact_enabled) for r in runners]
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
        for enabled in (False, True):
            ambient = '0' if enabled else '1'
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT', ambient)
            with ExitStack() as scopes:
                for target in [owner, *runners]:
                    if interface == 'owner':
                        scopes.enter_context(runtime.q5k_iu8_exact_enabled(target, enabled))
                    elif interface == 'mapped_preemptor':
                        scopes.enter_context(runtime.q5k_iu8_exact_enabled(target, True))
                        if not enabled:
                            if target is owner:
                                scopes.enter_context(runtime.q5k_iu8_exact_enabled(target, False))
                            else:
                                from scripts.qwen4exp_halo_box_campaign_ab import q8_mapped_down_scope
                                scopes.enter_context(q8_mapped_down_scope(runner=target))
                        assert runtime.resolve_q5k_iu8_exact(target) is enabled
                    elif interface in {'campaign', 'state'}:
                        if interface == 'campaign':
                            from scripts.qwen4exp_halo_box_campaign_ab import route_scope
                            scope = route_scope('q5k-iu8-exact', 'after' if enabled else 'before', target)
                        else:
                            from scripts.qwen4exp_row4_state_gate import arm_control
                            scope = arm_control('q5k-iu8-exact', '1' if enabled else '0', target)
                        scopes.enter_context(scope)
                        assert runtime.resolve_q5k_iu8_exact(target) is enabled
                    else:
                        if interface == 'ar':
                            from scripts.qwen4exp_candidate_ar import bind_q5k_iu8_override
                        else:
                            from scripts.qwen4exp_profile_gap import bind_q5k_iu8_override
                        report = bind_q5k_iu8_override(scopes, target, {
                            'HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT': '1' if enabled else '0'})
                        assert report['q5k_iu8_exact_enabled'] is enabled
                for runner in runners:
                    # The routed-F16 composite precedes this independent grouped
                    # fallback; isolate the route under test on each runner.
                    scopes.enter_context(runtime.routed_f16_enabled(runner, False))
                outputs, logits = [], []
                for _ in range(2):
                    calls.clear()
                    start = len(captures)
                    outputs.append(llm.generate([list(range(100, 100 + length))], params))
                    assert bool(calls) is (enabled and length >= 2)
                    logits.append([x.copy() for x in captures[start:]])
                assert outputs[0] == outputs[1]
                assert logits[0] and len(logits[0]) == len(logits[1])
                for a, b in zip(*logits):
                    assert np.isfinite(a).all()
                    np.testing.assert_array_equal(a, b)
                assert os.environ['HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT'] == ambient
            for runner, value in prior:
                assert runner.q5k_iu8_exact_enabled is value
    finally:
        register(key, original, replace=True)
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    assert after['active_allocations'] == before['active_allocations']
    assert after['current_allocated_bytes'] == before['current_allocated_bytes']
