"""Public generation obeys owner prepacking in both MMQ resource consumers."""
import os
import numpy as np
import pytest
from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('interface', ['direct', 'vector_prerequisite', 'prepack_campaign', 'prepack_state_gate'])
@pytest.mark.parametrize('length', [73, 273, 517])
def test_public_mmq_prepack_owner_ignores_ambient_and_restores(monkeypatch, length, interface):
    from hipengine import LLM, SamplingParams
    from hipengine.runtime import qwen4_exp_runner as runtime
    from hipengine.runtime import gguf_linear as linear
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    hip = get_hip_runtime()
    hip.device_synchronize()
    before = memory_stats()
    captures, leaves = [], []
    _install_logit_capture(monkeypatch, captures)
    resolve = linear.resolve
    raw_launch = linear._launch_raw_mmq_d4x3_f32
    def launch_raw(fn, weight, x, out, rows, hidden, outputs, kwargs):
        def parent(*args, **kw):
            leaves.append('raw_mmq_parent')
            return fn(*args, **kw)
        return raw_launch(parent, weight, x, out, rows, hidden, outputs, kwargs)
    monkeypatch.setattr(linear, '_launch_raw_mmq_d4x3_f32', launch_raw)
    def resolved(**kw):
        fn = resolve(**kw)
        if kw.get('layer') == 'linear' and 'mmq' in kw.get('variant', ''):
            def launch(*args, **kwargs):
                leaves.append(kw['variant'])
                return fn(*args, **kwargs)
            return launch
        return fn
    monkeypatch.setattr(linear, 'resolve', resolved)
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    try:
        owner = _unwrap(llm._get_text_generator())
        assert owner.q8_mmq_prepack_enabled is None
        with (runtime.q8_mmq_prefill_enabled(owner, True),
              runtime.q8_mmq_prepack_enabled(owner, True)):
            owner.runner.configure_mmq_prefill_resources()
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
        for enabled in (False, True):
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK', '0' if enabled else '1')
            if interface == 'vector_prerequisite':
                from scripts.qwen4exp_halo_box_campaign_ab import mmq_vector_prerequisite
                scope = mmq_vector_prerequisite(
                    owner.runner, 'q8-mmq-vec4' if enabled else 'q8-mmq-raw-vector')
            elif interface == 'prepack_campaign':
                from scripts.qwen4exp_halo_box_campaign_ab import route_scope
                scope = route_scope('q8-mmq-prepack', 'after' if enabled else 'before', owner.runner)
            elif interface == 'prepack_state_gate':
                from scripts.qwen4exp_row4_state_gate import arm_control
                scope = arm_control('q8-mmq-prepack', '1' if enabled else '0', owner.runner)
            else:
                scope = runtime.q8_mmq_prepack_enabled(owner, enabled)
            with runtime.q8_mmq_prefill_enabled(owner, True), scope:
                owner.runner.configure_mmq_prefill_resources()
                sidecars = owner.runner._q8_mmq_weight_sidecars
                assert sidecars is not None
                repeats = []
                for _ in range(2):
                    captures.clear()
                    leaves.clear()
                    output = llm.generate_detailed([[1] * length], params)
                    assert leaves, 'no actual MMQ launch observed'
                    assert any('prepacked' in name for name in leaves) is enabled, set(leaves)
                    assert owner.runner.q8_mmq_prepack_enabled is enabled
                    assert os.environ['HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK'] == ('0' if enabled else '1')
                    assert owner.runner._q8_mmq_weight_sidecars is sidecars
                    assert captures and all(np.isfinite(row).all() for row in captures)
                    repeats.append((tuple(output[0].generated_token_ids), [r.copy() for r in captures]))
                assert repeats[0][0] == repeats[1][0]
                assert len(repeats[0][1]) == len(repeats[1][1])
                for a, b in zip(repeats[0][1], repeats[1][1], strict=True):
                    np.testing.assert_array_equal(a, b)
            assert owner.q8_mmq_prepack_enabled is owner.runner.q8_mmq_prepack_enabled is None
    finally:
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
