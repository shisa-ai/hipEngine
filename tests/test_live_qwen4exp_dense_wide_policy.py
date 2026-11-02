"""Public prefill lends one runner-owned wide policy to dispatch and staging."""
import numpy as np
import pytest

from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('interface,length', [
    *((interface, 273) for interface in ('direct', 'layer_gate', 'dense_ab',
       'last_row_probe', 'profile_gap', 'layer2', 'restore_ab')),
    *(('candidate_ar', length) for length in (73, 273, 517)),
])
def test_public_explicit_wide_policy_restores_after_requests(monkeypatch, interface, length):
    from hipengine import LLM, SamplingParams
    from hipengine.runtime import gguf_linear as gl
    from hipengine.runtime import qwen4_exp_runner as runner_module
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats

    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    captures = []
    _install_logit_capture(monkeypatch, captures)
    policies, route_choices = [], []
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_SHORT_ROWS', '0')
    real_route = gl._record_route_engagement

    def record_route(selector, **kwargs):
        if selector == 'dense_wide':
            route_choices.append(kwargs)
        return real_route(selector, **kwargs)

    monkeypatch.setattr(gl, '_record_route_engagement', record_route)
    real = gl._record_gguf_linear_dispatch

    def dispatch(**kwargs):
        if int(kwargs.get('rows', 0)) > 1:
            policies.append((gl._q8_dense_wide_enabled(), gl._q8_dense_wide_layers(),
                             kwargs['variant']))
        return real(**kwargs)

    monkeypatch.setattr(gl, '_record_gguf_linear_dispatch', dispatch)
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    try:
        generator = _unwrap(llm._get_text_generator())
        serving = generator.create_resident_model_runner(capacity=1)
        serving.prepare(max_sequence_length=CONTEXT)
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True,
                                speculative_mtp=False)
        bound_policy = generator.q8_dense_wide_policy
        for enabled in (False, True):
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE', '0' if enabled else '1')
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS', 'broken')
            policy = gl.Q8DenseWidePolicy(enabled, frozenset(range(16, 48)))
            if interface == 'candidate_ar':
                from contextlib import ExitStack
                from scripts.qwen4exp_candidate_ar import bind_dense_wide_override
                control = ExitStack()
                report = bind_dense_wide_override(control, generator, {
                    'HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE': '1' if enabled else '0',
                    'HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS': ','.join(map(str, range(16, 48)))})
                assert report['q8_dense_wide_policy'] == policy.to_report()
            elif interface == 'direct':
                control = gl.q8_dense_wide_policy(generator, policy)
            elif interface == 'layer_gate':
                from scripts import execution_profile_q8_wmma_prefill_layers_gate as gate
                control = gate._apply_route_env(
                    gate.SELECTORS['dense_wide'], candidate=enabled,
                    layers=','.join(map(str, range(16, 48))), generator=generator)
            elif interface in {'layer2', 'restore_ab'}:
                from contextlib import contextmanager, ExitStack
                from dataclasses import replace
                from scripts.qwen4exp_layer2_profile_gate import CANDIDATES, bind_candidate_dense_wide

                @contextmanager
                def layer2_control():
                    prior = gl.Q8DenseWidePolicy(enabled, frozenset({7}))
                    spec = replace(CANDIDATES['production_baseline'],
                                   q8_dense_wide_layers=tuple(range(16, 48)))
                    with gl.q8_dense_wide_policy(generator, prior):
                        with ExitStack() as stack:
                            if interface == 'restore_ab':
                                from scripts.qwen4exp_prefill_restore_ab import arm_dense_wide_scope
                                with monkeypatch.context() as patch:
                                    patch.setitem(CANDIDATES, 'test_public_wide', spec)
                                    stack.enter_context(arm_dense_wide_scope(generator, 'test_public_wide'))
                            else:
                                report = bind_candidate_dense_wide(stack, generator, spec)
                                assert report == policy.to_report()
                            yield
                        assert generator.q8_dense_wide_policy is prior
                control = layer2_control()
            elif interface == 'profile_gap':
                from scripts.qwen4exp_profile_gap import _wide_owner_override
                control = _wide_owner_override(generator, enabled, ','.join(map(str, range(16, 48))))
            elif interface == 'last_row_probe':
                from scripts.qwen4exp_prefill_last_row_probe import route_environment
                control = route_environment(candidate=enabled, runner=generator)
            else:
                from scripts.qwen4exp_dense_route_ab import ARMS, _apply_arm
                arm_name = 'wide-16-47' if enabled else 'exact'
                control = _apply_arm(next(arm for arm in ARMS if arm.name == arm_name), generator)
            with (control,
                  runner_module.q8_mmq_prefill_enabled(generator, False),
                  gl.q4k_rowtile_session(False)):
                policy = gl.resolve_q8_dense_wide_policy(generator.runner)
                from hipengine.generation.qwen4_exp_profiles import selected_prefill_routes
                selected = selected_prefill_routes(generator)
                assert selected['q8_dense_wide_policy'] == policy.to_report()
                assert selected['q8_wmma_layers'] == gl.resolve_q8_wmma_layers(generator).to_report()
                from scripts.qwen4exp_dp8_staged_scope_prefill_ab import bound_owner_record
                assert bound_owner_record(generator)['bound_dense_wide_policy'] == policy.to_report()
                results = []
                for _ in range(2):
                    policies.clear()
                    route_choices.clear()
                    captures.clear()
                    output = llm.generate_detailed([[1] * length], params)
                    assert policies
                    assert all(item[:2] == (enabled, policy.layers) for item in policies), set(policies)
                    wide = [item for item in policies if 'dense_wide' in item[2]]
                    assert bool(wide) is (enabled and length > 256), set(policies)
                    if enabled and length <= 256:
                        prefill_choices = [row for row in route_choices if row['rows'] == length]
                        assert prefill_choices
                        assert all(not row['engaged'] and row['reason'] == 'ineligible shape'
                                   for row in prefill_choices), prefill_choices
                    assert captures and all(np.isfinite(row).all() for row in captures)
                    results.append((tuple(output[0].generated_token_ids),
                                    [row.copy() for row in captures]))
                assert results[0][0] == results[1][0]
                assert len(results[0][1]) == len(results[1][1])
                for a, b in zip(results[0][1], results[1][1], strict=True):
                    np.testing.assert_array_equal(a, b)
            assert generator.q8_dense_wide_policy is bound_policy
    finally:
        llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for name in ('current_allocated_bytes', 'active_allocations'):
        assert after[name] == before[name]
