"""Public Q4 exact IU8 owner selection, repeats and teardown."""
from contextlib import ExitStack
from unittest.mock import patch
import os
import numpy as np
import pytest
from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('interface', ['direct', 'profile_gap', 'ar', 'campaign', 'state', 'pair2', 'out4', 'ablation', 'profile', 'risk', 'risk_gap', 'risk_ar', 'risk_candidate'])
@pytest.mark.parametrize('length', [1, 17, 73])
def test_public_q4_iu8_owner(monkeypatch, length, interface):
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
    key = KernelKey('hip_gfx1151', 'moe_linear', 'gguf_q4_k',
                    'selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out')
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    original = resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
    def counted(*args, **kwargs):
        calls.append(float(args[10]))
        return original(*args, **kwargs)
    register(key, counted, replace=True)
    try:
        owner = _unwrap(llm._get_text_generator())
        serving = owner.create_resident_model_runner(capacity=1)
        serving.prepare(max_sequence_length=CONTEXT)
        runners = list(serving._all_runners)
        if all(owner.runner is not r for r in runners):
            runners.append(owner.runner)
        prior = [(r, r.q4_iu8_exact_enabled) for r in runners]
        prior_risk = [(r, r.q4_iu8_risk_mult) for r in runners]
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
        for enabled in (False, True):
            if interface == 'profile':
                assert all(r.q4_iu8_exact_enabled is True for r in runners)
            ambient = '0' if enabled else '1'
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT', ambient)
            with ExitStack() as scopes:
                for target in (owner, *runners):
                    if interface in {'risk', 'risk_gap', 'risk_ar', 'risk_candidate'}:
                        multiplier = 1.0 if enabled else 4.0
                        monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT', '8')
                        if interface == 'risk_candidate':
                            from dataclasses import replace
                            from scripts.qwen4exp_layer2_profile_gate import (
                                CANDIDATES, bind_candidate_q4_prefill,
                            )
                            spec = replace(CANDIDATES['production_baseline'],
                                           q4_iu8_risk_mult=multiplier)
                            bind_candidate_q4_prefill(scopes, target, spec)
                        elif interface == 'risk':
                            scopes.enter_context(runtime.q4_iu8_risk_mult(target, multiplier))
                        else:
                            if interface == 'risk_ar':
                                from scripts.qwen4exp_candidate_ar import bind_q4_iu8_risk_override
                            else:
                                from scripts.qwen4exp_profile_gap import bind_q4_iu8_risk_override
                            report = bind_q4_iu8_risk_override(scopes, target, {
                                'HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT': str(multiplier)})
                            assert report['q4_iu8_risk_mult'] == multiplier
                        scopes.enter_context(runtime.q4_iu8_exact_enabled(target, True))
                    elif interface in {'profile_gap', 'ar'}:
                        if interface == 'ar':
                            from scripts.qwen4exp_candidate_ar import bind_q4_iu8_override
                        else:
                            from scripts.qwen4exp_profile_gap import bind_q4_iu8_override
                        report = bind_q4_iu8_override(scopes, target, {
                            'HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT': '1' if enabled else '0'})
                        assert report['q4_iu8_exact_enabled'] is enabled
                    elif interface == 'profile':
                        from hipengine.generation.qwen4_exp_profiles import _publish_q4_iu8
                        scopes.enter_context(runtime.q4_iu8_exact_enabled(target, enabled))
                        _publish_q4_iu8(target, enabled)
                    elif interface in {'campaign', 'state'}:
                        if interface == 'campaign':
                            from scripts.qwen4exp_halo_box_campaign_ab import route_scope
                            scope = route_scope('q4-iu8-exact', 'after' if enabled else 'before', target)
                        else:
                            from scripts.qwen4exp_row4_state_gate import arm_control
                            scope = arm_control('q4-iu8-exact', '1' if enabled else '0', target)
                        scopes.enter_context(scope)
                    else:
                        scopes.enter_context(runtime.q4_iu8_exact_enabled(target, enabled))
                for runner in runners:
                    if interface in {'out4', 'ablation'}:
                        # The CLI module sets build defaults at import time.
                        with patch.dict(os.environ):
                            from scripts.qwen4exp_r2e_route_ablation import RUNNER_ROUTE_OFF_WITH_ENV
                        route = 'q4_out4' if interface == 'out4' else 'q4_iu8'
                        bindings, envs, off = RUNNER_ROUTE_OFF_WITH_ENV[route]
                        env_scope = scopes.enter_context(monkeypatch.context())
                        for key_name in envs:
                            env_scope.setenv(key_name, off)
                        for field, value in bindings:
                            scopes.enter_context(getattr(runtime, field)(runner, value))
                        assert runner.q4_iu8_exact_enabled is False
                    if interface == 'pair2':
                        from scripts.qwen4exp_halo_box_campaign_ab import q4_pair_scope
                        scopes.enter_context(q4_pair_scope(runner, True))
                        assert runner.q4_iu8_exact_enabled is False
                    # Isolate the grouped Q4 fallback behind the routed F16 owner.
                    scopes.enter_context(runtime.routed_f16_enabled(runner, False))
                outputs, logits = [], []
                for _ in range(2):
                    calls.clear()
                    start = len(captures)
                    outputs.append(llm.generate([list(range(100, 100 + length))], params))
                    assert bool(calls) is ((enabled or interface in {'risk', 'risk_gap', 'risk_ar', 'risk_candidate'}) and length >= 2 and interface not in {'pair2', 'out4', 'ablation'})
                    if interface in {'risk', 'risk_gap', 'risk_ar', 'risk_candidate'}:
                        assert all(value == multiplier for value in calls)
                        assert os.environ['HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT'] == '8'
                    logits.append([x.copy() for x in captures[start:]])
                assert outputs[0] == outputs[1]
                assert logits[0] and len(logits[0]) == len(logits[1])
                for a, b in zip(*logits):
                    assert np.isfinite(a).all()
                    np.testing.assert_array_equal(a, b)
                assert os.environ['HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT'] == ambient
            for runner, value in prior:
                assert runner.q4_iu8_exact_enabled is value
            for runner, value in prior_risk:
                assert runner.q4_iu8_risk_mult == value
    finally:
        register(key, original, replace=True)
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    assert after['active_allocations'] == before['active_allocations']
    assert after['current_allocated_bytes'] == before['current_allocated_bytes']
