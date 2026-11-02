"""Public prefill carries explicit raw geometry through actual dispatch."""
from contextlib import ExitStack
import os
import numpy as np
import pytest
from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('length', [17, 73, 273])
def test_public_raw_prefill_owner(monkeypatch, length):
    from hipengine import LLM, SamplingParams
    from hipengine.runtime import qwen4_exp_runner as runtime
    from hipengine.runtime import gguf_linear as linear
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    hip = get_hip_runtime()
    hip.device_synchronize()
    before = memory_stats()
    captures, launches = [], []
    _install_logit_capture(monkeypatch, captures)
    original = linear._record_gguf_linear_dispatch
    def observe(**kwargs):
        if int(kwargs.get('rows', 0)) > 1:
            launches.append((linear.raw_k_prefill_rowbatch(), linear.raw_k_prefill_variant(),
                             kwargs['variant']))
        return original(**kwargs)
    monkeypatch.setattr(linear, '_record_gguf_linear_dispatch', observe)
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    try:
        owner = _unwrap(llm._get_text_generator())
        assert owner.raw_prefill_policy is owner.runner.raw_prefill_policy is None
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
        with ExitStack() as stack:
            stack.enter_context(linear.q8_dense_wide_policy(owner, linear.Q8DenseWidePolicy(False, None)))
            stack.enter_context(runtime.q8_mmq_prefill_enabled(owner, False))
            stack.enter_context(runtime.gguf_wmma_prefill(owner, False))
            stack.enter_context(linear.q8_wmma_layer_scope(owner, linear.Q8WMMALayerScope.empty()))
            for policy in (runtime.RawPrefillPolicy(8, 'rowbatch'), runtime.RawPrefillPolicy(32, 'coltile8')):
                monkeypatch.setenv('HIPENGINE_QWEN4_EXP_RAW_ROWBATCH', 'invalid')
                monkeypatch.setenv('HIPENGINE_QWEN4_EXP_RAW_VARIANT', 'invalid')
                with runtime.raw_prefill_policy(owner, policy):
                    repeats = []
                    for _ in range(2):
                        launches.clear()
                        captures.clear()
                        output = llm.generate_detailed([[1] * length], params)
                        selected = ('coltile8_wave_scale' if policy.variant == 'coltile8'
                                    and owner.runner.q8_wave_scale_prefill else policy.variant)
                        assert launches and all(row[:2] == (policy.rowbatch, selected) for row in launches)
                        assert any(('rowbatch8' in row[2] if policy.variant == 'rowbatch'
                                    else 'coltile8' in row[2]) for row in launches), set(launches)
                        assert os.environ['HIPENGINE_QWEN4_EXP_RAW_ROWBATCH'] == 'invalid'
                        assert os.environ['HIPENGINE_QWEN4_EXP_RAW_VARIANT'] == 'invalid'
                        assert captures and all(np.isfinite(row).all() for row in captures)
                        repeats.append((tuple(output[0].generated_token_ids), [row.copy() for row in captures]))
                    assert repeats[0][0] == repeats[1][0]
                    for a, b in zip(repeats[0][1], repeats[1][1], strict=True):
                        np.testing.assert_array_equal(a, b)
                assert owner.raw_prefill_policy is owner.runner.raw_prefill_policy is None
    finally:
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
