"""Public QSA flash owner policy overrides opposite ambient enablement."""
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
def test_public_flash_owner(monkeypatch, length, interface):
    from hipengine import LLM, SamplingParams
    from hipengine.runtime import qwen4_exp_runner as runtime
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    hip = get_hip_runtime()
    hip.device_synchronize()
    before = memory_stats()
    captures, launches = [], []
    _install_logit_capture(monkeypatch, captures)
    original = runtime.qwen4_exp_qsa_flash_prefill
    def launch(*args, **kwargs):
        launches.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(runtime, 'qwen4_exp_qsa_flash_prefill', launch)
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    try:
        owner = _unwrap(llm._get_text_generator())
        serving = owner.create_resident_model_runner(capacity=1)
        serving.prepare(max_sequence_length=CONTEXT)
        runners = list(serving._all_runners)
        if all(owner.runner is not r for r in runners):
            runners.append(owner.runner)
        assert owner.qsa_flash_prefill_policy is None
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
        for enabled in (False, True):
            ambient = '0' if enabled else '1'
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL', ambient)
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS', '' if enabled else 'all')
            policy = runtime.QSAFlashPrefillPolicy(enabled, '35-47')
            with ExitStack() as scopes:
                if interface == 'profile_gap':
                    from scripts.qwen4exp_profile_gap import bind_flash_prefill_override
                    report = bind_flash_prefill_override(scopes, owner, {
                        'HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL': '1' if enabled else '0',
                        'HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS': policy.layers})
                    assert report['qsa_flash_prefill_policy'] == {'enabled': enabled, 'layers': policy.layers}
                else:
                    scopes.enter_context(runtime.qsa_flash_prefill(owner, policy))
                for runner in runners:
                    scopes.enter_context(runtime.qsa_wmma_attention(runner, False))
                    if runner is not owner.runner:
                        scopes.enter_context(runtime.qsa_flash_prefill(runner, policy))
                repeats = []
                for _ in range(2):
                    captures.clear()
                    launches.clear()
                    output = llm.generate_detailed([[1] * length], params)
                    assert bool(launches) is enabled
                    assert os.environ['HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL'] == ambient
                    assert os.environ['HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS'] == ('' if enabled else 'all')
                    assert captures and all(np.isfinite(row).all() for row in captures)
                    repeats.append((tuple(output[0].generated_token_ids), [r.copy() for r in captures]))
                assert repeats[0][0] == repeats[1][0]
                for a, b in zip(repeats[0][1], repeats[1][1], strict=True):
                    np.testing.assert_array_equal(a, b)
            assert owner.qsa_flash_prefill_policy is None
            assert all(r.qsa_flash_prefill_policy is None for r in runners)
    finally:
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
