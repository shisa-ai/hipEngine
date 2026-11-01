"""Public columnwarp admission follows its owner and restores diagnostics."""
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
    ('production', 23, None, 'qwen4exp_gdn_rowsplit_prefill'),
    ('production', 73, False, 'qwen4exp_sigmoid_wave_norm_narrow_prefill'),
    ('production', 257, True, 'qwen4exp_gdn_rowsplit_prefill'),
    ('strict', 389, None, 'direct_serial'),
    ('strict', 73, True, 'direct_tiled16'),
])
def test_public_columnwarp_admission_owner(monkeypatch, profile, length, override, expected):
    from hipengine import LLM, SamplingParams
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.runtime import qwen4_exp_runner as module

    # A stale request must neither open strict admission nor close production.
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL',
                       '0' if profile == 'production' else '1')
    calls = []
    original_resolve = module.resolve

    def instrument(fn, variant):
        def launch(*args, **kwargs):
            calls.append(variant)
            return fn(*args, **kwargs)
        for attr in ('supports_any_rows', 'supports_state_rows', 'publishes_f16_image'):
            if hasattr(fn, attr):
                setattr(launch, attr, getattr(fn, attr))
        return launch

    def observe(**kwargs):
        fn = original_resolve(**kwargs)
        return (instrument(fn, kwargs['variant'])
                if kwargs.get('layer') == 'gdn_recurrence_norm_gate' else fn)

    monkeypatch.setattr(module, 'resolve', observe)
    monkeypatch.setattr(module, 'qwen4_exp_gdn_prefill_f32', instrument(
        module.qwen4_exp_gdn_prefill_f32, 'direct_serial'))
    monkeypatch.setattr(module, 'qwen4_exp_gdn_prefill_tiled16_f32', instrument(
        module.qwen4_exp_gdn_prefill_tiled16_f32, 'direct_tiled16'))
    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile=profile,
              max_sequence_length=4096, max_active_requests=1, prefix_cache='off')
    try:
        generator = llm._get_text_generator()
        while getattr(generator, 'inner', None) is not None:
            generator = generator.inner
        published = profile == 'production'
        assert generator.gdn_colwarps_prefill is published
        assert generator.runner.gdn_colwarps_prefill is published
        scope = (nullcontext() if override is None
                 else module.gdn_colwarps_prefill(generator, override))
        with scope:
            effective = published if override is None else override
            assert module.resolve_gdn_colwarps_prefill(generator.runner) is effective
            outputs = llm.generate_detailed(
                [[1] * length], SamplingParams(max_tokens=3, temperature=0,
                                              ignore_eos=True, speculative_mtp=False))
            assert len(outputs) == 1
            assert len(outputs[0].generated_token_ids) == 3
            assert expected in calls, calls
            if not effective:
                assert not any('tiled16' in name or name == 'qwen4exp_gdn_rowsplit_prefill'
                               for name in calls), calls
        assert generator.gdn_colwarps_prefill is published
        assert generator.runner.gdn_colwarps_prefill is published
    finally:
        llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
