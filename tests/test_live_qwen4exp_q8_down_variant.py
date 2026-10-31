"""Public Q8 expert-down selection follows its profile owner, not stale env."""
from __future__ import annotations

import ctypes
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


@pytest.mark.parametrize('profile,length,engaged', [
    ('strict', 23, False), ('production', 23, True), ('production', 73, False),
])
def test_public_profile_owns_q8_down_fallback(monkeypatch, profile, length, engaged):
    from hipengine import LLM, SamplingParams
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.runtime import qwen4_exp_runner as module

    # The process key is retired, including when it names an invalid owner.
    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_Q8_DOWN_VARIANT', 'not_a_kernel')
    original = module.resolve
    calls = []

    def observe(**kwargs):
        owner = original(**kwargs)
        if (kwargs.get('quant') != 'gguf_q8_0'
                or not kwargs.get('variant', '').startswith('selected_grouped_')):
            return owner

        def launch(*args, **launch_kwargs):
            calls.append((kwargs['variant'], bool(launch_kwargs.get('risk_count_ptr'))))
            return owner(*args, **launch_kwargs)

        return launch

    monkeypatch.setattr(module, 'resolve', observe)
    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile=profile,
              max_sequence_length=4096, max_active_requests=1, prefix_cache='off')
    try:
        outputs = llm.generate_detailed(
            [[1] * length], SamplingParams(max_tokens=3, temperature=0,
                                          ignore_eos=True, speculative_mtp=False))
        assert len(outputs) == 1
        assert len(outputs[0].generated_token_ids) == 3
        generator = llm._get_text_generator()
        while getattr(generator, 'inner', None) is not None:
            generator = generator.inner
        expected = (module.Q8_DOWN_GUARDED_VARIANT if profile == 'production'
                    else module.Q8_DOWN_WMMA_VARIANT)
        assert generator.q8_down_variant == expected
        assert generator.runner.q8_down_variant == expected
        assert bool(calls) is engaged, calls
        assert all(variant == expected and risk for variant, risk in calls)
    finally:
        llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
