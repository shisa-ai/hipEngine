"""Public GDN variant selection is owner-bound, including diagnostic scopes."""
from __future__ import annotations

import ctypes
from contextlib import nullcontext
from pathlib import Path

import pytest

MODEL = Path('/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL')


def _hip_available():
    try:
        ctypes.CDLL('libamdhip64.so')
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()),
    reason='requires ROCm and the Qwen3.8-Flash-Next UD-Q4_K_XL artifact',
)


@pytest.mark.parametrize('profile,length,override,expected', [
    ('production', 73, 'published', 'qwen4exp_gdn_rowsplit_prefill'),
    ('production', 257, 'qwen4exp_gdn_tiled16_multi_prefill',
     'qwen4exp_gdn_tiled16_multi_prefill'),
    ('strict', 389, 'qwen4exp_gdn_tiled16_dpp_prefill',
     'qwen4exp_gdn_tiled16_dpp_prefill'),
    ('production', 73, None, 'default_tiled16'),
])
def test_public_gdn_variant_owner(monkeypatch, profile, length, override, expected):
    from hipengine import LLM, SamplingParams
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.runtime import qwen4_exp_runner as module

    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT', 'not_a_kernel')
    calls = []
    original = module.resolve

    def instrument(fn, variant):
        def launch(*args, **kwargs):
            calls.append(variant)
            return fn(*args, **kwargs)
        for attr in ('supports_any_rows', 'supports_state_rows', 'publishes_f16_image'):
            if hasattr(fn, attr):
                setattr(launch, attr, getattr(fn, attr))
        return launch

    def observe(**kwargs):
        fn = original(**kwargs)
        if kwargs.get('layer') == 'gdn_recurrence_norm_gate':
            return instrument(fn, kwargs['variant'])
        return fn

    monkeypatch.setattr(module, 'resolve', observe)
    monkeypatch.setattr(module, 'qwen4_exp_gdn_prefill_tiled16_f32', instrument(
        module.qwen4_exp_gdn_prefill_tiled16_f32, 'default_tiled16'))
    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile=profile,
              max_sequence_length=4096, max_active_requests=1, prefix_cache='off')
    try:
        generator = llm._get_text_generator()
        while getattr(generator, 'inner', None) is not None:
            generator = generator.inner
        published = ('qwen4exp_gdn_rowsplit_prefill'
                     if profile == 'production' else None)
        assert generator.gdn_tile16_variant == published
        assert generator.runner.gdn_tile16_variant == published
        scope = (nullcontext() if override == 'published'
                 else module.gdn_tile16_variant(generator, override))
        with scope:
            # Diagnostic selection does not change the existing capability gate.
            # Open it explicitly only for these diagnostic override cases.
            if override != 'published':
                monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL', '1')
            outputs = llm.generate_detailed(
                [[1] * length], SamplingParams(max_tokens=3, temperature=0,
                                              ignore_eos=True, speculative_mtp=False))
            assert len(outputs) == 1
            assert len(outputs[0].generated_token_ids) == 3
            assert expected in calls, calls
            assert 'not_a_kernel' not in calls
        assert generator.gdn_tile16_variant == published
        assert generator.runner.gdn_tile16_variant == published
    finally:
        llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
