"""Public scalar and batch QSA routes keep independent execution owners."""
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


@pytest.mark.parametrize('profile,length,override,v2,expected', [
    ('production', 2039, None, None, 'dense'),
    ('production', 2063, None, None, 'ordered_v2'),
    ('strict', 2077, None, None, 'unordered'),
    ('production', 2111, False, None, 'unordered'),
    ('strict', 2063, True, False, 'ordered_v1'),
])
def test_generator_scalar_ordered_qsa_owner(
        monkeypatch, profile, length, override, v2, expected):
    from hipengine import LLM
    from hipengine.generation import GenerationRequest
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.runtime import qwen4_exp_runner as module

    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE',
                       '0' if profile == 'production' else '1')
    calls = []

    def observe(fn, name):
        def launch(*args, **kwargs):
            calls.append(name)
            return fn(*args, **kwargs)
        return launch

    for attr, name in (
        ('qwen35_paged_full_attn_decode_context_bf16_spans', 'dense'),
        ('qwen4_exp_qsa_sparse_attention_paged_bf16_f32', 'unordered'),
        ('qwen4_exp_qsa_sparse_attention_paged_bf16_ordered_f32', 'ordered_v1'),
    ):
        monkeypatch.setattr(module, attr, observe(getattr(module, attr), name))
    real_resolve = module.resolve

    def resolve(**kwargs):
        fn = real_resolve(**kwargs)
        return (observe(fn, 'ordered_v2')
                if kwargs.get('variant') == 'strict_ordered_three_pass_v2_spans'
                else fn)

    monkeypatch.setattr(module, 'resolve', resolve)
    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile=profile,
              max_sequence_length=4608, max_active_requests=1, prefix_cache='off')
    try:
        generator = llm._get_text_generator()
        while getattr(generator, 'inner', None) is not None:
            generator = generator.inner
        runner = generator.runner
        published = profile == 'production'
        assert generator.qsa_ordered_decode is published
        assert runner.qsa_ordered_decode is published
        published_v2 = runner.qsa_ordered_decode_v2
        scope = (nullcontext() if override is None
                 else module.qsa_ordered_decode(runner, override))
        leaf_scope = (nullcontext() if v2 is None
                      else module.qsa_ordered_decode_v2(runner, v2))
        with scope, leaf_scope:
            # The model generator exposes scalar generation. LLM serving uses
            # the independent rows owner even at width one; do not reroute it.
            result = generator.generate_detailed(GenerationRequest(
                prompts=(tuple([1] * length),), max_tokens=3, temperature=0,
                top_p=1.0, ignore_eos=True))
            assert len(result) == 1
            assert len(result[0].generated_token_ids) == 3
            assert expected in calls, calls
            sparse = set(calls) - {'dense'}
            assert sparse == (set() if expected == 'dense' else {expected}), calls
        assert runner.qsa_ordered_decode is published
        assert runner.qsa_ordered_decode_v2 is published_v2
    finally:
        llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]


@pytest.mark.parametrize('profile,width,enabled', [
    ('production', 1, False),
    ('strict', 1, True),
    ('production', 2, False),
    ('production', 2, True),
])
def test_public_batch_qsa_does_not_read_scalar_ordered_owner(
        monkeypatch, profile, width, enabled):
    import numpy as np

    from hipengine import LLM, SamplingParams
    from hipengine.core.hip import get_hip_runtime, HipMemcpyKind
    from hipengine.core.memory import memory_stats
    from hipengine.runtime import qwen4_exp_batch_qsa as batch
    from hipengine.runtime import qwen4_exp_runner as module
    from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner

    monkeypatch.setenv('HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE',
                       '1' if not enabled else '0')
    calls, logits = [], []
    real_select = batch.batch_qsa_attention_implementation
    real_step = Qwen4ExpGGUFBatchRunner.step

    def select(**kwargs):
        fn, variant = real_select(**kwargs)

        def launch(*args, **launch_kwargs):
            calls.append(variant)
            return fn(*args, **launch_kwargs)
        return launch, variant

    def step(self, *args, **kwargs):
        result = real_step(self, *args, **kwargs)
        # Greedy serving skips readback, not computation. Observe the existing
        # device logits without changing the user's sampler path.
        captured = result.logits
        if captured is None:
            self.runtime.device_synchronize()
            captured = np.empty((len(result.slot_ids), self.config.vocab_size),
                                dtype=np.float32)
            self.runtime.memcpy(captured.ctypes.data, self.logits.ptr,
                                captured.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
        logits.append(np.array(captured, copy=True))
        return result

    monkeypatch.setattr(batch, 'batch_qsa_attention_implementation', select)
    monkeypatch.setattr(Qwen4ExpGGUFBatchRunner, 'step', step)
    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile=profile,
              max_sequence_length=4608, max_active_requests=width, prefix_cache='off')
    try:
        generator = llm._get_text_generator()
        while getattr(generator, 'inner', None) is not None:
            generator = generator.inner
        with module.qsa_ordered_decode(generator.runner, enabled):
            result = llm.generate_detailed(
                [[1] * length for length in (2063, 2077)[:width]],
                SamplingParams(max_tokens=3, temperature=0,
                               ignore_eos=True, speculative_mtp=False))
            assert len(result) == width
            assert all(len(row.generated_token_ids) == 3 for row in result)
        assert generator.runner.qsa_ordered_decode is (profile == 'production')
        assert calls
        assert set(calls) == {'strict_h256_head_pair_rows_spans'}
        assert logits
        assert all(np.isfinite(row).all() for row in logits)
    finally:
        llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
