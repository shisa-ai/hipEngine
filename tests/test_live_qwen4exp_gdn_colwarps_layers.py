"""Public generation forwards explicit GDN layer scopes to the recurrence leaf."""
import numpy as np
import pytest
from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('interface', ['direct', 'profile_gap', 'layer2', 'restore', 'decode_sync', 'profile_default'])
@pytest.mark.parametrize('length', [73, 273, 517])
def test_public_gdn_layer_scope_overrides_ambient_and_restores(monkeypatch, length, interface):
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
    if interface == 'profile_default':
        monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GDN_COLWARPS_DECODE_LAYERS', '999')
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    try:
        owner = _unwrap(llm._get_text_generator())
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True,
                                speculative_mtp=False)
        bound_layers = owner.gdn_colwarps_layers
        if interface == 'profile_default':
            import os
            assert os.environ['HIPENGINE_QWEN4_EXP_GDN_COLWARPS_DECODE_LAYERS'] == '999'
        scopes = ((bound_layers,) if interface == 'profile_default' else
                  ('27-47',) if interface == 'decode_sync' else ('999', 'all'))
        for scope in scopes:
            # Oppose the explicit owner after binding, when the legacy
            # diagnostic selector would otherwise be consulted.
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS',
                               'all' if scope == '999' else '999')
            repeated = []
            if interface == 'profile_default':
                from contextlib import nullcontext
                from types import SimpleNamespace
                from hipengine.execution_profiles import resolve_runtime_profile
                peer = resolve_runtime_profile(
                    model='qwen4_exp_gguf', backend='hip_gfx1151',
                    quant='gguf_ud_q4_k_xl', profile='strict')
                peer_owner = SimpleNamespace(runner=SimpleNamespace())
                peer.binder(peer_owner, peer)
                assert peer_owner.gdn_colwarps_layers == ''
                assert owner.gdn_colwarps_layers == bound_layers == ','.join(map(str, range(48)))
                binding = nullcontext()
            elif interface == 'decode_sync':
                from contextlib import ExitStack
                from scripts.qwen4exp_decode_sync_ab import bind_production_gdn
                binding = ExitStack()
                assert bind_production_gdn(binding, owner) == scope
            elif interface == 'layer2':
                from contextlib import ExitStack
                from dataclasses import replace
                from scripts.qwen4exp_layer2_profile_gate import CANDIDATES, bind_candidate_gdn_layers
                binding = ExitStack()
                bind_candidate_gdn_layers(binding, owner, replace(
                    CANDIDATES['gdn_isolated'], gdn_colwarps_layers=scope))
            elif interface == 'restore':
                from dataclasses import replace
                from scripts import qwen4exp_prefill_restore_ab as restore
                monkeypatch.setitem(restore.CANDIDATES, 'scope_probe', replace(
                    restore.CANDIDATES['gdn_isolated'], gdn_colwarps_layers=scope))
                binding = restore.arm_gdn_layers_scope(owner, 'scope_probe')
            elif interface == 'profile_gap':
                from scripts.qwen4exp_profile_gap import _gdn_layers_owner_override
                binding = _gdn_layers_owner_override(owner, scope)
            else:
                binding = module.gdn_colwarps_layers(owner, scope)
            with binding:
                for _ in range(2):
                    captures.clear()
                    tiled.clear()
                    result = llm.generate_detailed([[1] * length], params)
                    assert bool(tiled) is (scope != '999'), tiled
                    assert owner.runner.gdn_colwarps_layers == scope
                    assert captures and all(np.isfinite(row).all() for row in captures)
                    repeated.append((tuple(result[0].generated_token_ids),
                                     [row.copy() for row in captures]))
                assert repeated[0][0] == repeated[1][0]
                assert len(repeated[0][1]) == len(repeated[1][1])
                for a, b in zip(repeated[0][1], repeated[1][1], strict=True):
                    np.testing.assert_array_equal(a, b)
            assert owner.gdn_colwarps_layers == bound_layers
            assert owner.runner.gdn_colwarps_layers == bound_layers
    finally:
        llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
