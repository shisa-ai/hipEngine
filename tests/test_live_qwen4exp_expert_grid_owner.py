"""Public exact expert-grid owner selection, repeated logits and teardown."""
from contextlib import ExitStack
import os
import numpy as np
import pytest
from tests.test_live_qwen4exp_staged_f32_interface import (
    MODEL, CONTEXT, _hip_available, _unwrap, _install_logit_capture,
)

pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()), reason='requires ROCm and Qwen4Exp GGUF')


@pytest.mark.parametrize('interface', ['direct', 'profile_gap', 'candidate_ar', 'campaign_pair', 'campaign_pf13', 'ablation_q4', 'ablation_q51'])
@pytest.mark.parametrize('length', [17, 73, 273])
def test_public_expert_grid_owner(monkeypatch, length, interface):
    from hipengine import LLM, SamplingParams
    from hipengine.runtime import qwen4_exp_runner as runtime
    from hipengine.kernels.registry import register, resolve, KernelKey
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    hip = get_hip_runtime()
    hip.device_synchronize()
    before = memory_stats()
    captures, calls, originals = [], [], []
    _install_logit_capture(monkeypatch, captures)
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    variants = [
        ('gguf_q4_k', 'selected_dual_grouped_rowbatch8_bf16_bf16_out'),
        ('gguf_q5_1', 'selected_grouped_prefill_compact_rowbatch8_bf16_bf16_out'),
        ('gguf_q4_k', 'selected_dual_grouped_pair2_bf16_bf16_out'),
        ('gguf_q4_k', 'selected_dual_grouped_rowbatch8_out4_expertgrid64_bundle_bf16_bf16_out'),
        ('gguf_q5_1', 'selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out'),
        ('gguf_q4_k', 'selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out'),
        ('gguf_q4_k', 'selected_dual_grouped_rowbatch8_out4_bf16_bf16_out'),
        ('gguf_q5_1', 'selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out'),
        ('gguf_q5_1', 'selected_grouped_prefill_compact_rowbatch8_out8_bf16_bf16_out'),
    ]
    try:
        owner = _unwrap(llm._get_text_generator())
        for quant, variant in variants:
            key = KernelKey('hip_gfx1151', 'moe_linear', quant, variant)
            fn = resolve(backend=key.backend, layer=key.layer, quant=quant, variant=variant)
            originals.append((key, fn))
            def counted(*args, _fn=fn, _quant=quant, _variant=variant, **kwargs):
                calls.append((_quant, _variant))
                return _fn(*args, **kwargs)
            register(key, counted, replace=True)
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, speculative_mtp=False)
        with ExitStack() as stack:
            stack.enter_context(runtime.routed_f16_enabled(owner.runner, False))
            stack.enter_context(runtime.production_moe_prefill_enabled(owner.runner, False))
            stack.enter_context(runtime.q4_iu8_exact_enabled(owner, False))
            stack.enter_context(runtime.q51_iu8_exact_enabled(owner, False))
            stack.enter_context(runtime.q4_pair_prefill(owner.runner, False))
            stack.enter_context(runtime.q4_bundle_prefill(owner.runner, False))
            stack.enter_context(runtime.q51_pair_prefill(owner.runner, False))
            stack.enter_context(runtime.q51_down_m1(owner.runner, False))
            for enabled in (False, True):
                ambient = '0' if enabled else '64'
                monkeypatch.setenv('HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID', ambient)
                with ExitStack() as policy_stack:
                    if interface == 'direct':
                        policy_stack.enter_context(runtime.expert_grid_policy(
                            owner, runtime.ExpertGridPolicy(enabled, enabled)))
                    elif interface == 'campaign_pair':
                        from scripts.qwen4exp_halo_box_campaign_ab import q4_pair_scope
                        policy_stack.enter_context(q4_pair_scope(owner.runner, enabled))
                        assert owner.runner.expert_grid_policy == runtime.ExpertGridPolicy(True, True)
                    elif interface == 'campaign_pf13':
                        from scripts.qwen4exp_halo_box_campaign_ab import pf13_scope
                        policy_stack.enter_context(runtime.expert_grid_policy(
                            owner.runner, runtime.ExpertGridPolicy(False, True)))
                        policy_stack.enter_context(pf13_scope('after' if enabled else 'before', owner.runner))
                    elif interface.startswith('ablation_'):
                        from unittest.mock import patch
                        with patch.dict(os.environ):
                            from scripts import qwen4exp_r2e_route_ablation as ablation
                        route = 'q4_out4' if interface == 'ablation_q4' else 'q51_out8'
                        for field, value in ablation.RUNNER_ROUTE_OFF_WITH_ENV[route][0]:
                            if field != 'expert_grid_policy':
                                policy_stack.enter_context(getattr(runtime, field)(owner.runner, value))
                        assert ablation.bind_expert_grid_ablation(policy_stack, owner.runner, route) == {
                            'q4': False, 'q5': False}
                        policy_stack.enter_context(getattr(runtime, route)(owner.runner, enabled))
                    else:
                        import importlib
                        script = importlib.import_module('scripts.qwen4exp_' + interface)
                        report = script.bind_expert_grid_override(policy_stack, owner, {
                            'HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID': '64' if enabled else '0'})
                        assert report['expert_grid_policy'] == runtime.ExpertGridPolicy(enabled, enabled).to_report()
                    repeats = []
                    for _ in range(2):
                        calls.clear()
                        captures.clear()
                        output = llm.generate_detailed([[1] * length], params)
                        assert {q for q, _ in calls} == {'gguf_q4_k', 'gguf_q5_1'}, set(calls)
                        if interface == 'campaign_pair':
                            assert all('expertgrid64' in v or 'pair2' in v for _, v in calls)
                            assert any('pair2' in v for _, v in calls) is (enabled and length >= 64)
                        elif interface == 'campaign_pf13':
                            assert all(('expertgrid64' in v) is (q == 'gguf_q5_1') for q, v in calls)
                            assert any('expertgrid64_m1_' in v for _, v in calls) is enabled
                        elif interface.startswith('ablation_'):
                            assert all('expertgrid64' not in v for _, v in calls)
                            quant = 'gguf_q4_k' if interface == 'ablation_q4' else 'gguf_q5_1'
                            leaf = 'out4_' if quant == 'gguf_q4_k' else 'out8_'
                            assert all((leaf in v) is enabled for q, v in calls if q == quant)
                        else:
                            assert all(('expertgrid64' in v) is enabled for _, v in calls), set(calls)
                        assert os.environ['HIPENGINE_QWEN4_EXP_EXACT_EXPERT_GRID'] == ambient
                        assert captures and all(np.isfinite(row).all() for row in captures)
                        repeats.append((tuple(output[0].generated_token_ids), [r.copy() for r in captures]))
                    assert repeats[0][0] == repeats[1][0]
                    for a, b in zip(repeats[0][1], repeats[1][1], strict=True):
                        np.testing.assert_array_equal(a, b)
                assert owner.expert_grid_policy is owner.runner.expert_grid_policy is None
    finally:
        for key, fn in originals:
            register(key, fn, replace=True)
        llm.close()
    hip.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
