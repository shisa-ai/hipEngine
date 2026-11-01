"""Default public generation uses the profile's explicit wide policy."""
import numpy as np
import pytest
from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('length', [73, 273, 517])
def test_default_public_wide_policy_and_cleanup(monkeypatch, length):
    from hipengine import LLM, SamplingParams
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.runtime import gguf_linear as gl
    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    rows, captures = [], []
    original = gl._record_gguf_linear_dispatch

    def record(**kwargs):
        rows.append(dict(kwargs))
        return original(**kwargs)

    monkeypatch.setattr(gl, '_record_gguf_linear_dispatch', record)
    _install_logit_capture(monkeypatch, captures)
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    try:
        gen = _unwrap(llm._get_text_generator())
        expected = gl.Q8DenseWidePolicy(True, frozenset(range(48)))
        assert gen.q8_dense_wide_policy == expected
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True,
                                speculative_mtp=False)
        repeated = []
        for _ in range(2):
            rows.clear()
            captures.clear()
            output = llm.generate_detailed([[1] * length], params)
            assert gen.runner.q8_dense_wide_policy == expected
            wide = [row for row in rows if row['rows'] == length and 'dense_wide' in row['variant']]
            assert bool(wide) is (length > 256), {(r['rows'], r['variant']) for r in rows}
            assert captures and all(np.isfinite(row).all() for row in captures)
            repeated.append((tuple(output[0].generated_token_ids), [row.copy() for row in captures]))
        assert repeated[0][0] == repeated[1][0]
        assert len(repeated[0][1]) == len(repeated[1][1])
        for a, b in zip(repeated[0][1], repeated[1][1], strict=True):
            np.testing.assert_array_equal(a, b)
    finally:
        llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for name in ('current_allocated_bytes', 'active_allocations'):
        assert after[name] == before[name]
