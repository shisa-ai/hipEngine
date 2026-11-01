"""Public selected Q8 down owner beats the opposite ambient request."""
from contextlib import ExitStack
import os
import numpy as np
import pytest
from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('interface', ['direct', 'profile_gap', 'candidate_ar', 'logits_probe', 'layer2', 'journey', 'layer_probe', 'campaign', 'sorted_preemptor', 'mapped_preemptor', 'pf13_preemptor'])
@pytest.mark.parametrize('length', [17, 73, 273])
def test_public_selected_q8_down_owner(monkeypatch, length, interface):
    from hipengine import LLM, SamplingParams
    from hipengine.runtime import qwen4_exp_runner as runtime
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    hip = get_hip_runtime()
    hip.device_synchronize()
    before = memory_stats()
    captures, launches = [], []
    _install_logit_capture(monkeypatch, captures)
    original = runtime._run_qwen4_exp_q8_wmma_down
    def launch(*args, **kwargs):
        launches.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(runtime, '_run_qwen4_exp_q8_wmma_down', launch)
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    try:
        owner = _unwrap(llm._get_text_generator())
        serving = owner.create_resident_model_runner(capacity=1)
        serving.prepare(max_sequence_length=CONTEXT)
        runners = list(serving._all_runners)
        if all(owner.runner is not r for r in runners):
            runners.append(owner.runner)
        assert owner.q8_selected_down_enabled is None
        assert all(r.q8_selected_down_enabled is None for r in runners)
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
        for enabled in (False, True):
            ambient = '0' if enabled else '1'
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN', ambient)
            with ExitStack() as scopes:
                if interface.endswith('_preemptor'):
                    from scripts import qwen4exp_halo_box_campaign_ab as campaign
                    scopes.enter_context(runtime.q8_selected_down_enabled(owner, True))
                    def bind_preemptor(target):
                        scopes.enter_context(runtime.q8_selected_down_enabled(target, True))
                        if not enabled:
                            if interface == 'sorted_preemptor':
                                scope = campaign.q8_down_scope(runner=target)
                            elif interface == 'mapped_preemptor':
                                scope = campaign.q8_mapped_down_scope(runner=target)
                            else:
                                scope = campaign.pf13_scope('before', target)
                            scopes.enter_context(scope)
                    bind_preemptor(owner.runner)
                elif interface == 'campaign':
                    from scripts.qwen4exp_halo_box_campaign_ab import route_scope
                    scopes.enter_context(route_scope(
                        'q8-wmma-down', 'after' if enabled else 'before', owner))
                elif interface == 'layer_probe':
                    from scripts.qwen4exp_q8_localize import selected_down_layer_kwargs
                    original_moe = runtime.run_qwen4_exp_moe
                    def layer_call(mixed_ptr, weights, **kwargs):
                        layer = weights['expert_down'].spec.slot_path.split('.')[1]
                        return original_moe(mixed_ptr, weights, **selected_down_layer_kwargs(
                            kwargs, active_layer=layer, disabled_layer=None if enabled else layer))
                    monkeypatch.setattr(runtime, 'run_qwen4_exp_moe', layer_call)
                    scopes.callback(setattr, runtime, 'run_qwen4_exp_moe', original_moe)
                    scopes.enter_context(runtime.q8_selected_down_enabled(owner, True))
                elif interface == 'journey':
                    from scripts import qwen4exp_journey_localize as journey
                    monkeypatch.setattr(journey, '_SELECTED_DOWN_ARM_VALUES', {'public': enabled})
                    monkeypatch.setattr(journey, '_SELECTED_DOWN_STRICT_VALUE', enabled)
                    assert scopes.enter_context(journey.selected_down_arm_scope(owner, 'public')) is enabled
                    scopes.enter_context(journey.selected_down_split_scope(
                        owner, 'strict_prefill_production_decode', 'prefill'))
                elif interface == 'layer2':
                    from dataclasses import replace
                    from scripts.qwen4exp_layer2_profile_gate import CANDIDATES, bind_candidate_selected_q8_down
                    spec = replace(CANDIDATES['production_baseline'], q8_selected_down_enabled=enabled)
                    assert bind_candidate_selected_q8_down(scopes, owner, spec) is enabled
                elif interface == 'logits_probe':
                    from scripts.qwen4exp_q8_0_wmma_down_logits_probe import selected_down_probe_scope
                    report = scopes.enter_context(selected_down_probe_scope(owner, enabled))
                    assert report['enabled'] is enabled
                elif interface in {'profile_gap', 'candidate_ar'}:
                    if interface == 'candidate_ar':
                        from scripts.qwen4exp_candidate_ar import bind_selected_q8_down_override
                    else:
                        from scripts.qwen4exp_profile_gap import bind_selected_q8_down_override
                    report = bind_selected_q8_down_override(scopes, owner, {
                        'HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN': '1' if enabled else '0'})
                    assert report == {'q8_selected_down_override': enabled,
                                      'q8_selected_down_enabled': enabled}
                else:
                    scopes.enter_context(runtime.q8_selected_down_enabled(owner, enabled))
                for runner in runners:
                    # The production routed-F16 composite preempts this
                    # independent grouped fallback at larger prompt shapes.
                    scopes.enter_context(runtime.routed_f16_enabled(runner, False))
                    if runner is not owner.runner:
                        if interface.endswith('_preemptor'):
                            bind_preemptor(runner)
                        elif interface == 'campaign':
                            scopes.enter_context(route_scope(
                                'q8-wmma-down', 'after' if enabled else 'before', runner))
                        elif interface == 'logits_probe':
                            scopes.enter_context(selected_down_probe_scope(runner, enabled))
                        else:
                            scopes.enter_context(runtime.q8_selected_down_enabled(
                                runner, True if interface == 'layer_probe' else enabled))
                repeats = []
                for _ in range(2):
                    captures.clear()
                    launches.clear()
                    output = llm.generate_detailed([[1] * length], params)
                    assert bool(launches) is enabled
                    assert os.environ['HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN'] == ambient
                    assert captures and all(np.isfinite(row).all() for row in captures)
                    repeats.append((tuple(output[0].generated_token_ids), [r.copy() for r in captures]))
                assert repeats[0][0] == repeats[1][0]
                for a, b in zip(repeats[0][1], repeats[1][1], strict=True):
                    np.testing.assert_array_equal(a, b)
            assert owner.q8_selected_down_enabled is None
            assert all(r.q8_selected_down_enabled is None for r in runners)
    finally:
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
