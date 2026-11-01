"""Public GR DPP selection preserves its scope and restores owner state."""
from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path

import pytest

MODEL = Path('/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL')
WAVE = 'coltile2_branch4_rowbatch4_wave_scale_f32_exact'
DPP = 'coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact'


def _hip_available():
    try:
        ctypes.CDLL('libamdhip64.so')
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()),
    reason='requires ROCm and the Flash-Next UD-Q4_K_XL artifact',
)


@pytest.mark.parametrize('profile,width,length', [
    ('production', 1, 255),
    ('production', 1, 256),
    ('production', 1, 257),
    ('strict', 1, 389),
    ('production', 2, 389),
])
def test_public_gr_dpp_scope_route_logits_and_cleanup(
        monkeypatch, profile, width, length):
    import numpy as np

    from hipengine import LLM, SamplingParams
    from hipengine.core.hip import get_hip_runtime, HipMemcpyKind
    from hipengine.core.memory import memory_stats
    from hipengine.runtime import qwen4_exp_runner as module
    from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner

    real_resolve = module.resolve
    real_step = Qwen4ExpGGUFBatchRunner.step
    real_prefill = module.Qwen4ExpGGUFResidentModelRunner._prefill_chunk_steps
    calls, captures, prefill_owners = [], [], []

    def prefill(self, *args, **kwargs):
        prefill_owners.append((self.gr_wide, self.gr_wave_scale,
                               self.gr_dpp_reduction))
        return real_prefill(self, *args, **kwargs)

    def resolve(**kwargs):
        fn = real_resolve(**kwargs)
        if kwargs.get('layer') != 'linear+gr_gated_mean':
            return fn

        def launch(*args, **launch_kwargs):
            calls.append(kwargs['variant'])
            return fn(*args, **launch_kwargs)
        return launch

    def step(self, *args, **kwargs):
        result = real_step(self, *args, **kwargs)
        logits = result.logits
        if logits is None:
            self.runtime.device_synchronize()
            logits = np.empty((len(result.slot_ids), self.config.vocab_size),
                              dtype=np.float32)
            self.runtime.memcpy(logits.ctypes.data, self.logits.ptr,
                                logits.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
        captures.append(np.array(logits, copy=True))
        return result

    monkeypatch.setattr(module, 'resolve', resolve)
    monkeypatch.setattr(Qwen4ExpGGUFBatchRunner, 'step', step)
    monkeypatch.setattr(module.Qwen4ExpGGUFResidentModelRunner,
                        '_prefill_chunk_steps', prefill)
    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile=profile,
              max_sequence_length=4608, max_active_requests=width,
              prefix_cache='off')
    try:
        generator = llm._get_text_generator()
        while getattr(generator, 'inner', None) is not None:
            generator = generator.inner
        owner = generator.runner
        published = profile == 'production'
        assert owner.gr_dpp_reduction is published
        assert generator.gr_dpp_reduction is published
        # A runner-local diagnostic does not retune its serving siblings.
        # Materialize the pool first, then bind every measured staging owner.
        serving = generator.create_resident_model_runner(capacity=width)
        serving.prepare(max_sequence_length=4608)
        owners = serving._all_runners
        assert len(owners) == width
        results = []
        for enabled in (False, True):
            calls.clear()
            captures.clear()
            prefill_owners.clear()
            # Contradict the diagnostic with the retired process key. The
            # runner-bound scope, not ambient transport, must select the leaf.
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION',
                               '0' if enabled else '1')
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GR_IU8', '0')
            monkeypatch.setenv('HIPENGINE_QWEN4_EXP_GR_IU8_DOWN', '0')
            with ExitStack() as scopes:
                for measured in owners:
                    scopes.enter_context(module.gr_wide(measured, False))
                    scopes.enter_context(module.gr_wave_scale(measured, True))
                    scopes.enter_context(module.gr_dpp_reduction(measured, enabled))
                output = llm.generate_detailed(
                    [[1] * length for _ in range(width)],
                    SamplingParams(max_tokens=3, temperature=0,
                                   ignore_eos=True, speculative_mtp=False))
            assert all(measured.gr_dpp_reduction is published for measured in owners)
            assert len(prefill_owners) == width
            assert set(prefill_owners) == {(False, True, enabled)}
            assert len(output) == width
            assert all(len(row.generated_token_ids) == 3 for row in output)
            assert captures and all(row.size for row in captures)
            assert all(np.isfinite(row).all() for row in captures)
            if length > 256:
                assert calls and set(calls) == {DPP if enabled else WAVE}, calls
            else:
                assert not calls, calls
            results.append((tuple(tuple(row.generated_token_ids) for row in output),
                            [row.copy() for row in captures]))
        assert results[0][0] == results[1][0]
        assert len(results[0][1]) == len(results[1][1])
        for reference, candidate in zip(results[0][1], results[1][1], strict=True):
            np.testing.assert_array_equal(reference, candidate)
    finally:
        llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
