"""Public WMMA layer scopes stay explicit after ambient fallback retirement."""
from contextlib import ExitStack
import os

import numpy as np
import pytest
from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('length', [2, 73, 273])
def test_public_wmma_layer_scope(monkeypatch, length):
    from hipengine import LLM, SamplingParams
    from hipengine.runtime import gguf_linear as gl
    from hipengine.runtime import qwen4_exp_runner as qr
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats

    hip = get_hip_runtime()
    hip.device_synchronize()
    before = memory_stats()
    captures, launches = [], []
    _install_logit_capture(monkeypatch, captures)
    original = gl._record_gguf_linear_dispatch

    def record(**kwargs):
        if kwargs['rows'] > 1 and kwargs['quant'] == 'gguf_q8_0':
            launches.append((gl._q8_wmma_layers_scope(), kwargs['abi']))
        return original(**kwargs)

    monkeypatch.setattr(gl, '_record_gguf_linear_dispatch', record)
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    try:
        owner = _unwrap(llm._get_text_generator())
        serving = owner.create_resident_model_runner(capacity=1)
        serving.prepare(max_sequence_length=CONTEXT)
        runners = list(serving._all_runners)
        if all(owner.runner is not r for r in runners):
            runners.append(owner.runner)
        targets = (owner, *runners)
        prior = [(target, target.q8_wmma_layer_scope) for target in targets]
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
        for raw in ('', '0', ','.join(map(str, range(48)))):
            scope = gl.Q8WMMALayerScope.parse(raw)
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS', '999')
            with ExitStack() as scopes:
                for target in targets:
                    scopes.enter_context(gl.q8_wmma_layer_scope(target, scope))
                    scopes.enter_context(qr.q8_mmq_prefill_enabled(target, False))
                    scopes.enter_context(qr.gguf_wmma_prefill(target, False))
                    scopes.enter_context(gl.q8_dense_wide_policy(target, gl.Q8DenseWidePolicy(False, None)))
                repeats = []
                for _ in range(2):
                    launches.clear()
                    captures.clear()
                    output = llm.generate_detailed([[1] * length], params)
                    assert launches
                    assert all(layers == scope.layers for layers, _ in launches)
                    assert any(abi == 'wmma_raw' for _, abi in launches) is (not scope.is_empty)
                    assert captures and all(np.isfinite(row).all() for row in captures)
                    repeats.append((tuple(output[0].generated_token_ids), [x.copy() for x in captures]))
                assert repeats[0][0] == repeats[1][0]
                assert len(repeats[0][1]) == len(repeats[1][1])
                for a, b in zip(repeats[0][1], repeats[1][1], strict=True):
                    np.testing.assert_array_equal(a, b)
                assert os.environ['HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS'] == '999'
            for target, value in prior:
                assert target.q8_wmma_layer_scope is value
    finally:
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
