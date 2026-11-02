"""Public Q5_1 exact IU8 owner selection, repeats and teardown."""
from contextlib import ExitStack
import os
import numpy as np
import pytest
from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('interface', ['direct', 'profile_gap', 'candidate_ar', 'campaign', 'state', 'pair2', 'pf13', 'profile'])
@pytest.mark.parametrize('length', [1, 17, 73])
def test_public_q51_iu8_owner(monkeypatch, length, interface):
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
    key = KernelKey('hip_gfx1151', 'moe_linear', 'gguf_q5_1',
                    'selected_wmma_iu8_risk_prefill_bf16_bf16_out')
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
        prior = [(r, r.q51_iu8_exact_enabled) for r in runners]
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
        assert owner.q51_iu8_exact_enabled is True
        assert all(r.q51_iu8_exact_enabled is True for r in runners)
        for enabled in ((True,) if interface == 'profile' else (False, True)):
            ambient = '0' if enabled else '1'
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT', ambient)
            with ExitStack() as scopes:
                def bind(target):
                    if interface == 'profile':
                        return  # Exercise the profile's published default.
                    if interface in {'profile_gap', 'candidate_ar'}:
                        if interface == 'candidate_ar':
                            from scripts.qwen4exp_candidate_ar import bind_q51_iu8_override
                        else:
                            from scripts.qwen4exp_profile_gap import bind_q51_iu8_override
                        report = bind_q51_iu8_override(scopes, target, {
                            'HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT': '1' if enabled else '0'})
                        assert report['q51_iu8_exact_enabled'] is enabled
                    elif interface in {'campaign', 'state'}:
                        if interface == 'campaign':
                            from scripts.qwen4exp_halo_box_campaign_ab import route_scope
                            scope = route_scope('q51-iu8-exact', 'after' if enabled else 'before', target)
                        else:
                            from scripts.qwen4exp_row4_state_gate import arm_control
                            scope = arm_control('q51-iu8-exact', '1' if enabled else '0', target)
                        scopes.enter_context(scope)
                    else:
                        scopes.enter_context(runtime.q51_iu8_exact_enabled(target, enabled))
                bind(owner)
                for runner in runners:
                    bind(runner)
                    # The production routed-F16 composite precedes this
                    # independent grouped fallback; isolate the route under test.
                    scopes.enter_context(runtime.routed_f16_enabled(runner, False))
                    if interface in {'pair2', 'pf13'}:
                        from scripts import qwen4exp_halo_box_campaign_ab as campaign
                        scope = (campaign.q51_pair2_scope(runner=runner) if interface == 'pair2'
                                 else campaign.pf13_scope('after', runner))
                        scopes.enter_context(scope)
                        assert runner.q51_iu8_exact_enabled is False
                outputs, logits = [], []
                for _ in range(2):
                    calls.clear()
                    start = len(captures)
                    outputs.append(llm.generate([list(range(100, 100 + length))], params))
                    assert bool(calls) is (enabled and length >= 2 and interface not in {'pair2', 'pf13'})
                    logits.append([x.copy() for x in captures[start:]])
                assert outputs[0] == outputs[1]
                assert logits[0] and len(logits[0]) == len(logits[1])
                for a, b in zip(*logits):
                    assert np.isfinite(a).all()
                    np.testing.assert_array_equal(a, b)
                assert os.environ['HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT'] == ambient
            for runner, value in prior:
                assert runner.q51_iu8_exact_enabled is value
    finally:
        register(key, original, replace=True)
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    assert after['active_allocations'] == before['active_allocations']
    assert after['current_allocated_bytes'] == before['current_allocated_bytes']
