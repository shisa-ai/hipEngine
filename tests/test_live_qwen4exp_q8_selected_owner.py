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


@pytest.mark.parametrize('interface', ['direct', 'profile_gap'])
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
                if interface == 'profile_gap':
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
                        scopes.enter_context(runtime.q8_selected_down_enabled(runner, enabled))
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
