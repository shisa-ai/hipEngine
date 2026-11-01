"""Public generation forwards explicit GDN layer scopes to the recurrence leaf."""
import numpy as np
import pytest
from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('length', [73, 273, 517])
def test_public_gdn_layer_scope_overrides_ambient_and_restores(monkeypatch, length):
    from hipengine import LLM, SamplingParams
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.runtime import qwen4_exp_runner as module
    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    captures, tiled = [], []
    _install_logit_capture(monkeypatch, captures)
    original = module.resolve

    def observe(**kwargs):
        fn = original(**kwargs)
        if kwargs.get('layer') != 'gdn_recurrence_norm_gate':
            return fn
        def launch(*args, **kw):
            if ('tiled16' in kwargs['variant']
                    or kwargs['variant'] == 'qwen4exp_gdn_rowsplit_prefill'):
                tiled.append(kwargs['variant'])
            return fn(*args, **kw)
        for attr in ('supports_any_rows', 'supports_state_rows', 'publishes_f16_image'):
            if hasattr(fn, attr):
                setattr(launch, attr, getattr(fn, attr))
        return launch

    monkeypatch.setattr(module, 'resolve', observe)
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    try:
        owner = _unwrap(llm._get_text_generator())
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True,
                                speculative_mtp=False)
        for scope in ('999', 'all'):
            # Oppose the explicit owner after binding, when the legacy
            # diagnostic selector would otherwise be consulted.
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS',
                               'all' if scope == '999' else '999')
            repeated = []
            with module.gdn_colwarps_layers(owner, scope):
                for _ in range(2):
                    captures.clear()
                    tiled.clear()
                    result = llm.generate_detailed([[1] * length], params)
                    assert bool(tiled) is (scope == 'all'), tiled
                    assert owner.runner.gdn_colwarps_layers == scope
                    assert captures and all(np.isfinite(row).all() for row in captures)
                    repeated.append((tuple(result[0].generated_token_ids),
                                     [row.copy() for row in captures]))
                assert repeated[0][0] == repeated[1][0]
                assert len(repeated[0][1]) == len(repeated[1][1])
                for a, b in zip(repeated[0][1], repeated[1][1], strict=True):
                    np.testing.assert_array_equal(a, b)
            assert owner.gdn_colwarps_layers is None
            assert owner.runner.gdn_colwarps_layers is None
    finally:
        llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
