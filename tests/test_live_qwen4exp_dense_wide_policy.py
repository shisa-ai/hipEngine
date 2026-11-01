"""Public prefill lends one runner-owned wide policy to dispatch and staging."""
import numpy as np
import pytest

from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


def test_public_explicit_wide_policy_restores_after_requests(monkeypatch):
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
    policies = []
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
        for enabled in (False, True):
            policy = gl.Q8DenseWidePolicy(enabled, frozenset(range(16, 48)))
            with (gl.q8_dense_wide_policy(generator, policy),
                  runner_module.q8_mmq_prefill_enabled(generator, False),
                  gl.q4k_rowtile_session(False)):
                results = []
                for _ in range(2):
                    policies.clear()
                    captures.clear()
                    output = llm.generate_detailed([[1] * 273], params)
                    assert policies
                    assert all(item[:2] == (enabled, policy.layers) for item in policies), set(policies)
                    wide = [item for item in policies if 'dense_wide' in item[2]]
                    assert bool(wide) is enabled, set(policies)
                    assert captures and all(np.isfinite(row).all() for row in captures)
                    results.append((tuple(output[0].generated_token_ids),
                                    [row.copy() for row in captures]))
                assert results[0][0] == results[1][0]
                assert len(results[0][1]) == len(results[1][1])
                for a, b in zip(results[0][1], results[1][1], strict=True):
                    np.testing.assert_array_equal(a, b)
            assert getattr(generator, 'q8_dense_wide_policy', None) is None
    finally:
        llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for name in ('current_allocated_bytes', 'active_allocations'):
        assert after[name] == before[name]
