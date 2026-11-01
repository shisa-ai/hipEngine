"""Public GR IU8 pair ignores opposite ambient inputs and restores owners."""
from contextlib import ExitStack
import os
import numpy as np
import pytest
from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('interface', ['direct', 'profile_gap', 'candidate_ar', 'layer2'])
@pytest.mark.parametrize('length', [17, 255, 256, 257, 389])
def test_public_gr_iu8_owner(monkeypatch, length, interface):
    from hipengine import LLM, SamplingParams
    from hipengine.runtime import qwen4_exp_runner as runtime
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    hip = get_hip_runtime()
    hip.device_synchronize()
    before = memory_stats()
    captures, calls = [], []
    _install_logit_capture(monkeypatch, captures)
    original = runtime.gguf_q8_0_iu8_wmma_prefill_f32_f32
    def launch(*args, **kwargs):
        calls.append((args[3], args[4], args[5]))
        return original(*args, **kwargs)
    monkeypatch.setattr(runtime, 'gguf_q8_0_iu8_wmma_prefill_f32_f32', launch)
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    try:
        owner = _unwrap(llm._get_text_generator())
        serving = owner.create_resident_model_runner(capacity=1)
        serving.prepare(max_sequence_length=CONTEXT)
        runners = list(serving._all_runners)
        if all(owner.runner is not r for r in runners):
            runners.append(owner.runner)
        prior = [(r, r.gr_iu8_policy) for r in runners]
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
        for up, down in [(False, False), (True, False), (False, True)]:
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GR_IU8', '0' if up else '1')
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GR_IU8_DOWN', '0' if down else '1')
            policy = runtime.GRIU8Policy(up, down)
            with ExitStack() as scopes:
                def bind(target):
                    if interface == 'layer2':
                        from dataclasses import replace
                        from scripts.qwen4exp_layer2_profile_gate import CANDIDATES, bind_candidate_gr_iu8
                        spec = replace(CANDIDATES['production_baseline'], gr_iu8_up=up, gr_iu8_down=down)
                        assert bind_candidate_gr_iu8(scopes, target, spec) == {'up': up, 'down': down}
                    elif interface in {'profile_gap', 'candidate_ar'}:
                        if interface == 'profile_gap':
                            from scripts.qwen4exp_profile_gap import bind_gr_iu8_override
                        else:
                            from scripts.qwen4exp_candidate_ar import bind_gr_iu8_override
                        report = bind_gr_iu8_override(scopes, target, {
                            'HIPENGINE_QWEN4_EXP_GR_IU8': '1' if up else '0',
                            'HIPENGINE_QWEN4_EXP_GR_IU8_DOWN': '1' if down else '0'})
                        assert report['gr_iu8_policy'] == {'up': up, 'down': down}
                    else:
                        scopes.enter_context(runtime.gr_iu8(target, policy))
                bind(owner)
                for runner in runners:
                    bind(runner)
                repeats = []
                for _ in range(2):
                    calls.clear()
                    captures.clear()
                    output = llm.generate_detailed([[1] * length], params)
                    assert bool(calls) is ((up or down) and length > 256)
                    if calls:
                        assert all(k < n if up else k > n for _, k, n in calls)
                    assert os.environ['HIPENGINE_QWEN4_EXP_GR_IU8'] == ('0' if up else '1')
                    assert os.environ['HIPENGINE_QWEN4_EXP_GR_IU8_DOWN'] == ('0' if down else '1')
                    assert captures and all(np.isfinite(row).all() for row in captures)
                    repeats.append((tuple(output[0].generated_token_ids), [r.copy() for r in captures]))
                assert repeats[0][0] == repeats[1][0]
                for a, b in zip(repeats[0][1], repeats[1][1], strict=True):
                    np.testing.assert_array_equal(a, b)
            for runner, value in prior:
                assert runner.gr_iu8_policy is value
        assert owner.gr_iu8_policy is None
    finally:
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
