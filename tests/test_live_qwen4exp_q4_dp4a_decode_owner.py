"""Public Q4 DP4A decode choices ignore ambient keys and restore owners."""
from contextlib import ExitStack
import os

import numpy as np
import pytest

from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('interface', ['direct', 'profile'])
@pytest.mark.parametrize('length', [1, 17, 73])
def test_public_q4_dp4a_decode_owner(monkeypatch, length, interface):
    from hipengine import LLM, SamplingParams
    from hipengine.runtime import qwen4_exp_runner as runtime
    from hipengine.generation.qwen4_exp_profiles import PRODUCTION_Q4_DP4A_DECODE_LAYERS
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.runtime.moe_graph import MoeGraphCache

    hip = get_hip_runtime()
    hip.device_synchronize()
    before = memory_stats()
    captures, calls, graph_runs = [], [], []
    _install_logit_capture(monkeypatch, captures)
    original_graph_run = MoeGraphCache.run

    def graph_run(cache, *args, **kwargs):
        result = original_graph_run(cache, *args, **kwargs)
        graph_runs.append(result)
        return result

    monkeypatch.setattr(MoeGraphCache, 'run', graph_run)
    original = runtime.resolve

    def observe(**kwargs):
        fn = original(**kwargs)
        if (kwargs.get('layer') != 'linear'
                or not kwargs.get('variant', '').startswith('selected_dual_q8_1_dp4a_silu_')):
            return fn
        def launch(*args, **kw):
            calls.append(kwargs['variant'])
            return fn(*args, **kw)
        return launch

    monkeypatch.setattr(runtime, 'resolve', observe)
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q4_DP4A64', '1')
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS', 'invalid-ambient')
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    try:
        owner = _unwrap(llm._get_text_generator())
        serving = owner.create_resident_model_runner(capacity=1)
        serving.prepare(max_sequence_length=CONTEXT)
        runners = list(serving._all_runners)
        if all(owner.runner is not runner for runner in runners):
            runners.append(owner.runner)
        assert serving._ar_batch is not None
        batch_runner = serving._ar_batch.runner
        assert batch_runner is not None
        runners.append(batch_runner)
        targets = [owner, *runners]
        prior = [(target, target.q4_dp4a_decode_enabled, target.q4_dp4a_decode_layers)
                 for target in targets]
        assert all(enabled is False for _, enabled, _ in prior)
        params = SamplingParams(max_tokens=3, temperature=0, ignore_eos=True,
                                speculative_mtp=False)
        for enabled in ([False] if interface == 'profile' else [False, True]):
            ambient = '0' if enabled else '1'
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q4_DP4A64', ambient)
            with ExitStack() as scopes:
                if interface == 'direct':
                    for target in targets:
                        scopes.enter_context(runtime.q4_dp4a_decode(
                            target, enabled, PRODUCTION_Q4_DP4A_DECODE_LAYERS))
                repeats = []
                calls.clear()
                for _ in range(2):
                    captures.clear()
                    graph_runs.clear()
                    prior_call_count = len(calls)
                    output = llm.generate_detailed([[1] * length], params)
                    assert bool(calls) is enabled
                    if enabled and len(calls) == prior_call_count:
                        assert 'replay' in graph_runs
                    assert os.environ['HIPENGINE_QWEN4_EXP_Q4_DP4A64'] == ambient
                    assert os.environ['HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS'] == 'invalid-ambient'
                    assert captures and all(np.isfinite(row).all() for row in captures)
                    repeats.append((tuple(output[0].generated_token_ids), [row.copy() for row in captures]))
                assert repeats[0][0] == repeats[1][0]
                for left, right in zip(repeats[0][1], repeats[1][1], strict=True):
                    np.testing.assert_array_equal(left, right)
            for target, prior_enabled, prior_layers in prior:
                assert target.q4_dp4a_decode_enabled is prior_enabled
                assert target.q4_dp4a_decode_layers == prior_layers
    finally:
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
