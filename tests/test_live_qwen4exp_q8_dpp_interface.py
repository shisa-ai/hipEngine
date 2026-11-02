"""Public Q8 DPP family comparison preserves exact parents and cached choice."""
import os

import numpy as np
import pytest

from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()),
    reason='requires ROCm and the Flash-Next UD-Q4_K_XL artifact',
)

PREFIX = 'coltile8_rowbatch4_wave_scale_pipelined'
PARENT = PREFIX + '_f32_f32_out'
DPP = PREFIX + '_dpp_f32_f32_out'
CHECKED = PREFIX + '_dpp_checked_full_rows_f32_f32_out'


@pytest.mark.parametrize('interface', ['campaign', 'owner'])
@pytest.mark.parametrize('length', [73, 512, 513])
def test_public_dpp_family_route_cache_parity_and_cleanup(monkeypatch, length, interface):
    from hipengine import LLM, SamplingParams
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.runtime import gguf_linear as linear
    from scripts.qwen4exp_halo_box_campaign_ab import route_scope, Q8_WAVE_ENV

    key = linear._Q8_DPP_REDUCTION_ENV
    before_dpp = os.environ.get(key)
    before_env = {name: os.environ.get(name) for name in Q8_WAVE_ENV}
    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    captures = []
    finals = []
    real_record = linear._record_gguf_linear_dispatch

    def record(**kwargs):
        finals.append(dict(kwargs))
        return real_record(**kwargs)

    monkeypatch.setattr(linear, '_record_gguf_linear_dispatch', record)
    _install_logit_capture(monkeypatch, captures)
    try:
        generator = _unwrap(llm._get_text_generator())
        serving = generator.create_resident_model_runner(capacity=1)
        serving.prepare(max_sequence_length=CONTEXT)
        runner = generator.runner
        prior_wide = runner.q8_dense_wide_policy
        from hipengine.runtime.gguf_linear import Q8DenseWidePolicy
        # An explicit enabled preemptor must be closed by the campaign scope,
        # even after profile transport no longer reads the process key.
        runner.q8_dense_wide_policy = Q8DenseWidePolicy(True, frozenset({16, 47}))
        bound_wide = runner.q8_dense_wide_policy
        bound_env = {name: os.environ.get(name) for name in Q8_WAVE_ENV}

        def arm(value):
            finals.clear()
            captures.clear()
            # Use the maintained campaign scope to expose the pipelined parent;
            # this comparison changes only its DPP family decision.
            from contextlib import nullcontext
            from hipengine.runtime.qwen4_exp_runner import q8_dpp_reduction_enabled
            # Both interfaces must ignore opposing ambient input.
            os.environ[key] = '1' if value == '0' else '0'
            with route_scope('q8-dpp-reduction',
                             ('after' if value == '0' else 'before') if interface == 'owner'
                             else ('before' if value == '0' else 'after'), runner):
                assert linear.resolve_q8_dense_wide_policy(runner) == Q8DenseWidePolicy(False, None)
                choice = (q8_dpp_reduction_enabled(runner, value != '0')
                          if interface == 'owner' else nullcontext())
                with choice:
                    output = llm.generate_detailed(
                        [[1] * length], SamplingParams(max_tokens=3, temperature=0,
                                                      ignore_eos=True, speculative_mtp=False))
            assert runner.q8_dense_wide_policy is bound_wide
            assert {name: os.environ.get(name) for name in Q8_WAVE_ENV} == bound_env
            assert len(output[0].generated_token_ids) == 3
            assert captures and all(np.isfinite(row).all() for row in captures)
            prefill = [row for row in finals if row['quant'] == 'gguf_q8_0'
                       and row['rows'] == length and row['variant'].startswith(PREFIX)]
            assert prefill, ('public prefill did not reach the Q8 comparison family',
                             length, value,
                             sorted({(r['quant'], r['variant'], r['rows'])
                                     for r in finals}))
            expected = []
            for row in prefill:
                checked = linear.q8_0_coltile_dpp_checked_full_rows_qualified(
                    row['rows'], row['in_features'], row['out_features'])
                expected.append(PARENT if value == '0' else CHECKED if checked else DPP)
            assert [row['variant'] for row in prefill] == expected
            if length == 512 and value != '0':
                assert CHECKED in expected, 'complete rows never engaged checked sibling'
            if length == 513 and value != '0':
                assert CHECKED not in expected and DPP in expected
            return (prefill, [row.copy() for row in captures],
                    tuple(output[0].generated_token_ids))

        arm('0')  # warm graphs before comparing full final-dispatch sequences
        narrow = arm('0')
        enabled = arm('1')
        default = arm(None)
        restored = arm('0')
        geometry = lambda result: [(r['rows'], r['in_features'], r['out_features'])
                                   for r in result[0]]
        for result in (enabled, default, restored):
            assert geometry(narrow) == geometry(result)
            assert narrow[2] == result[2]
            assert len(narrow[1]) == len(result[1])
            for expected, actual in zip(narrow[1], result[1], strict=True):
                np.testing.assert_array_equal(expected, actual)
    finally:
        if 'runner' in locals():
            runner.q8_dense_wide_policy = prior_wide
        llm.close()
        if before_dpp is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = before_dpp
    # The profile owns its process pins beyond generator close. This
    # diagnostic restores the post-binding baseline, not the pre-profile state.
    assert {name: os.environ.get(name) for name in Q8_WAVE_ENV} == bound_env
    assert os.environ.get(key) == before_dpp
    runtime.device_synchronize()
    after = memory_stats()
    for name in ('current_allocated_bytes', 'active_allocations'):
        assert after[name] == before[name]
