"""Public plain-Q4 prefill owner selection, repeatability and teardown."""
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


@pytest.mark.parametrize('interface', ['direct', 'profile_gap', 'ablation', 'decode_sync', 'candidate', 'candidate_scope', 'journey', 'declared', 'profile', 'layers', 'layers_gap', 'layers_candidate', 'layers_journey', 'layers_declared', 'layers_profile'])
@pytest.mark.parametrize('length', [1, 17, 73])
def test_public_plain_q4_prefill_owner(monkeypatch, length, interface):
    from hipengine import LLM, SamplingParams
    from hipengine.runtime import qwen4_exp_runner as runtime
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    hip = get_hip_runtime()
    hip.device_synchronize()
    before = memory_stats()
    captures, calls = [], []
    _install_logit_capture(monkeypatch, captures)
    original = runtime.gguf_q4_k_selected_dual_wmma_iu8_prefill_bf16_bf16_out
    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(runtime, 'gguf_q4_k_selected_dual_wmma_iu8_prefill_bf16_bf16_out', counted)
    if interface == 'layers_profile':
        monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS', '999')
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    try:
        owner = _unwrap(llm._get_text_generator())
        serving = owner.create_resident_model_runner(capacity=1)
        serving.prepare(max_sequence_length=CONTEXT)
        runners = list(serving._all_runners)
        if all(owner.runner is not r for r in runners):
            runners.append(owner.runner)
        prior = [(r, r.q4_iu8_prefill_enabled) for r in runners]
        prior_layers = [(r, r.q4_iu8_layers) for r in runners]
        if interface == 'layers_profile':
            from hipengine.generation.qwen4_exp_profiles import PRODUCTION_Q4_IU8_PREFILL_LAYERS
            suffix = ','.join(map(str, PRODUCTION_Q4_IU8_PREFILL_LAYERS))
            assert all(value == suffix for _, value in prior_layers)
            assert os.environ['HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS'] == '999'
        monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS', 'all')
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
        for enabled in (False, True):
            if interface == 'profile':
                assert all(r.q4_iu8_prefill_enabled is False for r in runners)
            ambient = '0' if enabled else '1'
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL', ambient)
            with ExitStack() as scopes:
                for target in (owner, *runners):
                    if interface == 'layers_profile':
                        from hipengine.generation.qwen4_exp_profiles import _publish_q4_layers
                        monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS', '999')
                        scopes.enter_context(runtime.q4_iu8_prefill_enabled(target, True))
                        scopes.enter_context(runtime.q4_iu8_layers(target, '999'))
                        _publish_q4_layers(target, production=enabled)
                        assert target.q4_iu8_layers == (suffix if enabled else '')
                    elif interface in {'layers', 'layers_gap', 'layers_candidate', 'layers_journey', 'layers_declared'}:
                        monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS',
                                           '999' if enabled else 'all')
                        scopes.enter_context(runtime.q4_iu8_prefill_enabled(target, True))
                        layers = 'all' if enabled else '999'
                        if interface == 'layers_declared':
                            from scripts.qwen4exp_layer2_profile_gate import CANDIDATES, candidate_q4_prefill_scope
                            scopes.enter_context(runtime.q4_iu8_layers(target, '999'))
                            if enabled:
                                scopes.enter_context(candidate_q4_prefill_scope(
                                    target, CANDIDATES['repair_free_gate_up']))
                            assert target.q4_iu8_layers == layers
                        elif interface == 'layers_journey':
                            from scripts import qwen4exp_journey_localize as journey
                            flags = journey.q4_layers_snapshot({}, published=layers)
                            choices = journey.migrate_q4_layers_arms(
                                {'public': flags}, production_value='ambient')
                            monkeypatch.setattr(journey, '_Q4_LAYERS_ARM_VALUES', choices)
                            assert scopes.enter_context(journey.q4_layers_arm_scope(target, 'public')) == layers
                        elif interface == 'layers_candidate':
                            from dataclasses import replace
                            from scripts.qwen4exp_layer2_profile_gate import CANDIDATES, candidate_q4_prefill_scope
                            spec = replace(CANDIDATES['production_baseline'], q4_iu8_layers=layers)
                            scopes.enter_context(candidate_q4_prefill_scope(target, spec))
                            assert target.q4_iu8_layers == layers
                        elif interface == 'layers_gap':
                            from scripts.qwen4exp_profile_gap import bind_q4_iu8_layers_override
                            report = bind_q4_iu8_layers_override(scopes, target, {
                                'HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS': layers})
                            assert report['q4_iu8_layers'] == layers
                        else:
                            assert scopes.enter_context(runtime.q4_iu8_layers(target, layers)) == layers
                    elif interface == 'profile':
                        from hipengine.generation.qwen4_exp_profiles import _publish_q4_prefill
                        scopes.enter_context(runtime.q4_iu8_prefill_enabled(target, enabled))
                        _publish_q4_prefill(target, production=enabled, quant='gguf_q4_k_m')
                        assert target.q4_iu8_prefill_enabled is enabled
                    elif interface == 'journey':
                        from scripts import qwen4exp_journey_localize as journey
                        flags = journey.q4_prefill_snapshot({}, published=enabled)
                        choices = journey.migrate_q4_prefill_arms(
                            {'public': flags}, production_value=not enabled)
                        monkeypatch.setattr(journey, '_Q4_PREFILL_ARM_VALUES', choices)
                        assert scopes.enter_context(journey.q4_prefill_arm_scope(target, 'public')) is enabled
                    elif interface in {'candidate', 'candidate_scope', 'declared'}:
                        from dataclasses import replace
                        from scripts.qwen4exp_layer2_profile_gate import (
                            CANDIDATES, bind_candidate_q4_prefill, candidate_q4_prefill_scope,
                        )
                        spec = (CANDIDATES['production_moe_restore' if enabled
                                           else 'production_conservative']
                                if interface == 'declared' else
                                replace(CANDIDATES['production_baseline'],
                                        q4_iu8_prefill_enabled=enabled))
                        if interface == 'candidate_scope':
                            assert scopes.enter_context(candidate_q4_prefill_scope(target, spec)) is enabled
                        else:
                            assert bind_candidate_q4_prefill(scopes, target, spec) is enabled
                    elif interface == 'decode_sync':
                        from scripts.qwen4exp_decode_sync_ab import bind_production_q4_prefill
                        monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS', '999')
                        assert bind_production_q4_prefill(scopes, target) is True
                        assert target.q4_iu8_layers == '35-47'
                    elif interface == 'ablation':
                        # Preserve process build defaults set by the CLI import.
                        with patch.dict(os.environ):
                            from scripts.qwen4exp_r2e_route_ablation import RUNNER_ROUTE_OFF_WITH_ENV
                        bindings, envs, off = RUNNER_ROUTE_OFF_WITH_ENV['q4_iu8']
                        assert envs == ()
                        for field, value in bindings:
                            scopes.enter_context(getattr(runtime, field)(target, value))
                    elif interface == 'profile_gap':
                        from scripts.qwen4exp_profile_gap import bind_q4_iu8_prefill_override
                        report = bind_q4_iu8_prefill_override(scopes, target, {
                            'HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL': '1' if enabled else '0'})
                        assert report['q4_iu8_prefill_enabled'] is enabled
                    else:
                        scopes.enter_context(runtime.q4_iu8_prefill_enabled(target, enabled))
                for runner in runners:
                    scopes.enter_context(runtime.routed_f16_enabled(runner, False))
                    scopes.enter_context(runtime.q4_exact_grouped_gate(runner, False))
                    scopes.enter_context(runtime.production_moe_prefill_enabled(runner, False))
                outputs, logits = [], []
                for _ in range(2):
                    calls.clear()
                    start = len(captures)
                    outputs.append(llm.generate([list(range(100, 100 + length))], params))
                    assert bool(calls) is (length >= 2 and interface != 'ablation'
                                           and (enabled or interface in {'decode_sync', 'layers_profile'}))
                    logits.append([x.copy() for x in captures[start:]])
                assert outputs[0] == outputs[1]
                assert logits[0] and len(logits[0]) == len(logits[1])
                for a, b in zip(*logits):
                    assert np.isfinite(a).all()
                    np.testing.assert_array_equal(a, b)
                assert os.environ['HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL'] == ambient
            for runner, value in prior:
                assert runner.q4_iu8_prefill_enabled is value
            for runner, value in prior_layers:
                assert runner.q4_iu8_layers == value
    finally:
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    assert after['active_allocations'] == before['active_allocations']
    assert after['current_allocated_bytes'] == before['current_allocated_bytes']
