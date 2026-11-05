"""Public cancellation restores raw-prefill and expert-grid diagnostic owners."""
from contextlib import ExitStack

import pytest

from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('length', [17, 73])
def test_prefill_cancellation_restores_transport_owners_and_continues(monkeypatch, length):
    from hipengine import LLM, SamplingParams
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.generation.deadline import GenerationCancellationToken, GenerationCancelled
    from hipengine.runtime import qwen4_exp_runner as runtime
    from hipengine.runtime import gguf_linear as linear

    hip = get_hip_runtime()
    hip.device_synchronize()
    before = memory_stats()
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    token = GenerationCancellationToken()
    observed = []
    real_steps = runtime.Qwen4ExpGGUFResidentModelRunner._prefill_chunk_steps

    def cancelled_steps(runner, *args, **kwargs):
        result = yield from real_steps(runner, *args, **kwargs)
        observed.append((runner.raw_prefill_policy, runner.expert_grid_policy,
                         runner.q8_dpp_reduction_enabled, linear._q8_dpp_reduction_raw()))
        token.cancel()
        token.raise_if_cancelled()
        return result

    try:
        owner = _unwrap(llm._get_text_generator())
        prior = (owner.raw_prefill_policy, owner.expert_grid_policy,
                 owner.q8_dpp_reduction_enabled)
        raw = runtime.RawPrefillPolicy(32, 'coltile8_wave_scale')
        grid = runtime.ExpertGridPolicy(False, True)
        with ExitStack() as scopes:
            scopes.enter_context(runtime.raw_prefill_policy(owner, raw))
            scopes.enter_context(runtime.expert_grid_policy(owner, grid))
            scopes.enter_context(runtime.q8_dpp_reduction_enabled(owner, False))
            with monkeypatch.context() as patch:
                patch.setattr(runtime.Qwen4ExpGGUFResidentModelRunner,
                              '_prefill_chunk_steps', cancelled_steps)
                try:
                    output = llm.generate_detailed([[1] * length], SamplingParams(
                        max_tokens=3, temperature=0, ignore_eos=True,
                        speculative_mtp=False, cancellation_token=token))
                except GenerationCancelled:
                    pass
                else:
                    assert output[0].finish_details.cancelled
            assert observed
            assert all(row == (raw, grid, False, '0') for row in observed)
        assert (owner.raw_prefill_policy, owner.expert_grid_policy,
                owner.q8_dpp_reduction_enabled) == prior
        assert linear._q8_dpp_reduction_raw() == '1'
        params = SamplingParams(max_tokens=3, temperature=0, ignore_eos=True,
                                speculative_mtp=False)
        first = llm.generate_detailed([[1] * length], params)
        second = llm.generate_detailed([[1] * length], params)
        assert len(first[0].generated_token_ids) == 3
        assert first[0].generated_token_ids == second[0].generated_token_ids
    finally:
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    for name in ('active_allocations', 'current_allocated_bytes'):
        assert after[name] == before[name]
