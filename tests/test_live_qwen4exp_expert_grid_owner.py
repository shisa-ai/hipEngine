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


@pytest.mark.parametrize('length', [17, 73, 273])
def test_public_expert_grid_owner(monkeypatch, length):
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
                with runtime.expert_grid_policy(owner, runtime.ExpertGridPolicy(enabled, enabled)):
                    repeats = []
                    for _ in range(2):
                        calls.clear()
                        captures.clear()
                        output = llm.generate_detailed([[1] * length], params)
                        assert {q for q, _ in calls} == {'gguf_q4_k', 'gguf_q5_1'}, set(calls)
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
