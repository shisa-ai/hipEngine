"""Public MMQ attention-gate selection overrides opposite ambient input."""
from contextlib import ExitStack
import os
import numpy as np
import pytest
from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('interface', ['direct', 'layer2', 'candidate_ar', 'profile_gap'])
@pytest.mark.parametrize('length', [73, 273, 517])
def test_public_attention_gate_owner(monkeypatch, length, interface):
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
    original = linear._LAUNCH_ABI['raw_mmq_d4x3_f32']
    def launch(fn, weight, x, out, rows, hidden, outputs, kwargs):
        launches.append((rows, hidden, outputs))
        return original(fn, weight, x, out, rows, hidden, outputs, kwargs)
    monkeypatch.setitem(linear._LAUNCH_ABI, 'raw_mmq_d4x3_f32', launch)
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    try:
        owner = _unwrap(llm._get_text_generator())
        assert owner.q8_mmq_attn_gate_enabled is owner.runner.q8_mmq_attn_gate_enabled is None
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
        with runtime.q8_mmq_prefill_enabled(owner, True), runtime.q8_mmq_prepack_enabled(owner, False):
            owner.runner.configure_mmq_prefill_resources()
            for enabled in (False, True):
                monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE', '0' if enabled else '1')
                with ExitStack() as stack:
                    if interface == 'profile_gap':
                        from scripts.qwen4exp_profile_gap import bind_attention_override
                        record = bind_attention_override(stack, owner, {
                            'HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE': '1' if enabled else '0'}, None)
                        assert record['q8_mmq_attn_gate_enabled'] is enabled
                    elif interface == 'candidate_ar':
                        from scripts.qwen4exp_candidate_ar import attention_override_scope
                        record = stack.enter_context(attention_override_scope(owner, enabled))
                        assert record['q8_mmq_attn_gate_enabled'] is enabled
                        assert record['q8_mmq_prefill_enabled'] is True
                    elif interface == 'layer2':
                        from dataclasses import replace
                        from scripts.qwen4exp_layer2_profile_gate import (
                            CANDIDATES, bind_candidate_q8_mmq_attn_gate,
                        )
                        spec = replace(CANDIDATES['q8_mmq_attn_gate'],
                                       q8_mmq_attn_gate_enabled=enabled)
                        assert bind_candidate_q8_mmq_attn_gate(stack, owner, spec) is enabled
                    else:
                        stack.enter_context(runtime.q8_mmq_attn_gate_enabled(owner, enabled))
                    repeats = []
                    for _ in range(2):
                        launches.clear()
                        captures.clear()
                        output = llm.generate_detailed([[1] * length], params)
                        assert launches, 'no actual MMQ launch observed'
                        assert any((h, o) == (2560, 6144) for _, h, o in launches) is enabled, launches
                        assert os.environ['HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE'] == ('0' if enabled else '1')
                        assert captures and all(np.isfinite(row).all() for row in captures)
                        repeats.append((tuple(output[0].generated_token_ids), [r.copy() for r in captures]))
                    assert repeats[0][0] == repeats[1][0]
                    assert len(repeats[0][1]) == len(repeats[1][1])
                    for a, b in zip(repeats[0][1], repeats[1][1], strict=True):
                        np.testing.assert_array_equal(a, b)
                assert owner.q8_mmq_attn_gate_enabled is owner.runner.q8_mmq_attn_gate_enabled is None
    finally:
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
