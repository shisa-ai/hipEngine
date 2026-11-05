"""Public state parity for exact raw-prefill and expert-grid policy choices."""
from contextlib import ExitStack
import os
from unittest.mock import patch

import numpy as np
import pytest

from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('length', [17, 73, 273])
def test_public_transport_policy_state_parity(monkeypatch, length):
    from hipengine import LLM, SamplingParams
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.runtime import qwen4_exp_runner as runtime
    with patch.dict(os.environ):
        from scripts.qwen4exp_layer2_profile_gate import _state_summary

    hip = get_hip_runtime()
    hip.device_synchronize()
    before = memory_stats()
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    logits, states = [], []
    _install_logit_capture(monkeypatch, logits)
    real_steps = runtime.Qwen4ExpGGUFResidentModelRunner._prefill_chunk_steps

    def record_steps(runner, *args, **kwargs):
        result = yield from real_steps(runner, *args, **kwargs)
        states.append(_state_summary(runner))
        return result

    monkeypatch.setattr(runtime.Qwen4ExpGGUFResidentModelRunner,
                        '_prefill_chunk_steps', record_steps)
    try:
        owner = _unwrap(llm._get_text_generator())
        baseline = None
        with ExitStack() as scope:
            # Expose the exact expert ladders rather than production preemptors.
            scope.enter_context(runtime.routed_f16_enabled(owner.runner, False))
            scope.enter_context(runtime.production_moe_prefill_enabled(owner.runner, False))
            scope.enter_context(runtime.q4_iu8_exact_enabled(owner, False))
            scope.enter_context(runtime.q51_iu8_exact_enabled(owner, False))
            scope.enter_context(runtime.q4_pair_prefill(owner.runner, False))
            scope.enter_context(runtime.q4_bundle_prefill(owner.runner, False))
            scope.enter_context(runtime.q51_pair_prefill(owner.runner, False))
            scope.enter_context(runtime.q51_down_m1(owner.runner, False))
            for enabled in (False, True, False):
                with ExitStack() as arm:
                    arm.enter_context(runtime.raw_prefill_policy(
                        owner, runtime.RawPrefillPolicy(32, 'coltile8')))
                    arm.enter_context(runtime.expert_grid_policy(
                        owner, runtime.ExpertGridPolicy(enabled, enabled)))
                    arm.enter_context(runtime.q8_dpp_reduction_enabled(owner, enabled))
                    logits.clear()
                    states.clear()
                    result = llm.generate_detailed([[1] * length], SamplingParams(
                        max_tokens=3, temperature=0, ignore_eos=True, speculative_mtp=False))
                    assert states and all(row['finite'] for row in states)
                    assert logits and all(np.isfinite(row).all() for row in logits)
                    current = (tuple(result[0].generated_token_ids),
                               list(states), [row.copy() for row in logits])
                    if baseline is None:
                        baseline = current
                    else:
                        assert current[:2] == baseline[:2]
                        assert len(current[2]) == len(baseline[2])
                        for expected, actual in zip(baseline[2], current[2], strict=True):
                            np.testing.assert_array_equal(expected, actual)
    finally:
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    for name in ('active_allocations', 'current_allocated_bytes'):
        assert after[name] == before[name]
