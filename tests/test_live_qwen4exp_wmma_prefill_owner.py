"""Public generic WMMA owner selection with opposite ambient input."""
from contextlib import ExitStack
import os
import numpy as np
import pytest
from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('interface', ['direct', 'profile_gap', 'candidate_ar', 'campaign', 'journey',
                                       'profile_strict', 'profile_production'])
@pytest.mark.parametrize('length', [17, 73, 273])
def test_public_generic_wmma_owner(monkeypatch, length, interface):
    from hipengine import LLM, SamplingParams
    from hipengine.runtime import qwen4_exp_runner as runtime
    from hipengine.runtime import gguf_linear as linear
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    hip = get_hip_runtime()
    hip.device_synchronize()
    before = memory_stats()
    captures, launches, selections = [], [], []
    _install_logit_capture(monkeypatch, captures)
    original_select = linear._wmma_prefill_dispatch
    def select(dispatch, **kwargs):
        result = original_select(dispatch, **kwargs)
        selections.append((kwargs['use_wmma'], dispatch.key.variant, result.abi))
        return result
    monkeypatch.setattr(linear, '_wmma_prefill_dispatch', select)
    original = linear._LAUNCH_ABI['wmma_raw']
    def launch(*args, **kwargs):
        launches.append(True)
        return original(*args, **kwargs)
    monkeypatch.setitem(linear._LAUNCH_ABI, 'wmma_raw', launch)
    monkeypatch.setenv('HIPENGINE_GGUF_WMMA_PREFILL', '1')
    profile = 'strict' if interface == 'profile_strict' else 'production'
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile=profile,
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    try:
        owner = _unwrap(llm._get_text_generator())
        assert owner.gguf_wmma_prefill_enabled is owner.runner.gguf_wmma_prefill_enabled is False
        assert os.environ['HIPENGINE_GGUF_WMMA_PREFILL'] == '1'
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
        for enabled in (False, True):
            ambient = '1' if interface in {'campaign', 'journey', 'profile_strict', 'profile_production'} else ('0' if enabled else '1')
            monkeypatch.setenv('HIPENGINE_GGUF_WMMA_PREFILL', ambient)
            # Isolate generic WMMA from the production wide route, which replaces
            # its supported raw Q8 leaves above 256 rows.
            with (
                ExitStack() as stack,
                linear.q8_dense_wide_policy(owner, linear.Q8DenseWidePolicy(False, None)),
            ):
                if interface in {'profile_strict', 'profile_production'}:
                    expected_wmma = False
                elif interface == 'journey':
                    from scripts.qwen4exp_journey_localize import generic_wmma_baseline_scope
                    stack.enter_context(generic_wmma_baseline_scope(owner))
                    expected_wmma = False
                    assert owner.gguf_wmma_prefill_enabled is owner.runner.gguf_wmma_prefill_enabled is False
                elif interface == 'campaign':
                    from scripts.qwen4exp_halo_box_campaign_ab import route_scope
                    stack.enter_context(route_scope(
                        'q8-wave-scale', 'after' if enabled else 'before', owner.runner))
                    expected_wmma = False
                    assert owner.runner.gguf_wmma_prefill_enabled is False
                    assert owner.runner.q8_wave_scale_prefill is enabled
                elif interface == 'profile_gap':
                    from scripts.qwen4exp_profile_gap import bind_generic_wmma_override
                    report = bind_generic_wmma_override(stack, owner, {
                        'HIPENGINE_GGUF_WMMA_PREFILL': '1' if enabled else '0'})
                    assert report['gguf_wmma_prefill_enabled'] is enabled
                elif interface == 'candidate_ar':
                    from scripts.qwen4exp_candidate_ar import bind_generic_wmma_override
                    report = bind_generic_wmma_override(stack, owner, enabled)
                    assert report['gguf_wmma_prefill_enabled'] is enabled
                else:
                    stack.enter_context(runtime.gguf_wmma_prefill(owner, enabled))
                if interface in {'direct', 'profile_gap', 'candidate_ar'}:
                    expected_wmma = enabled
                repeats = []
                for _ in range(2):
                    selections.clear()
                    launches.clear()
                    captures.clear()
                    output = llm.generate_detailed([[1] * length], params)
                    assert bool(launches) is expected_wmma, sorted(set(selections))
                    assert os.environ['HIPENGINE_GGUF_WMMA_PREFILL'] == ambient
                    assert captures and all(np.isfinite(row).all() for row in captures)
                    repeats.append((tuple(output[0].generated_token_ids), [r.copy() for r in captures]))
                assert repeats[0][0] == repeats[1][0]
                assert len(repeats[0][1]) == len(repeats[1][1])
                for a, b in zip(repeats[0][1], repeats[1][1], strict=True):
                    np.testing.assert_array_equal(a, b)
            assert owner.gguf_wmma_prefill_enabled is owner.runner.gguf_wmma_prefill_enabled is False
    finally:
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
