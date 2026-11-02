"""Public generation carries explicit F32 plane policy into its MMQ session."""
import os
import numpy as np
import pytest
from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('length', [73, 273, 517])
def test_public_mmq_planes_owner(monkeypatch, length):
    from hipengine import LLM, SamplingParams
    from hipengine.runtime import qwen4_exp_runner as runtime
    from hipengine.runtime import gguf_linear as linear
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    hip = get_hip_runtime()
    hip.device_synchronize()
    before = memory_stats()
    captures, policies, launches = [], [], []
    _install_logit_capture(monkeypatch, captures)
    for count in (2, 3):
        abi = f'raw_mmq_d4x{count}_f32'
        launch = linear._LAUNCH_ABI[abi]

        def observed_launch(*args, _launch=launch, _count=count, **kwargs):
            launches.append(_count)
            return _launch(*args, **kwargs)

        monkeypatch.setitem(linear._LAUNCH_ABI, abi, observed_launch)
    original = runtime._qwen4_exp_q8_mmq_policy

    def observe(policy, **kwargs):
        result = original(policy, **kwargs)
        if result is not None:
            policies.append((kwargs.get('f32_planes'), result.f32_planes))
        return result

    monkeypatch.setattr(runtime, '_qwen4_exp_q8_mmq_policy', observe)
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES', '3')
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    try:
        owner = _unwrap(llm._get_text_generator())
        assert owner.q8_mmq_f32_planes is owner.runner.q8_mmq_f32_planes is None
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
        with runtime.q8_mmq_prefill_enabled(owner, True), runtime.q8_mmq_prepack_enabled(owner, False):
            owner.runner.configure_mmq_prefill_resources()
            for planes in (2, 3):
                monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES', str(5 - planes))
                with runtime.q8_mmq_f32_planes(owner, planes):
                    repeats = []
                    for _ in range(2):
                        policies.clear()
                        launches.clear()
                        captures.clear()
                        result = llm.generate_detailed([[1] * length], params)
                        assert policies and all(pair == (planes, planes) for pair in policies)
                        assert launches and set(launches) == {planes}
                        assert os.environ['HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES'] == str(5 - planes)
                        assert captures and all(np.isfinite(row).all() for row in captures)
                        repeats.append((tuple(result[0].generated_token_ids), [row.copy() for row in captures]))
                    assert repeats[0][0] == repeats[1][0]
                    for left, right in zip(repeats[0][1], repeats[1][1], strict=True):
                        np.testing.assert_array_equal(left, right)
                assert owner.q8_mmq_f32_planes is owner.runner.q8_mmq_f32_planes is None
    finally:
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
