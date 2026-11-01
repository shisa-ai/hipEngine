"""Implicit launch libraries observe build state; explicit handles stay pinned."""
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


def test_public_library_warm_reload_and_explicit_handle(monkeypatch):
    from hipengine import LLM, SamplingParams
    from hipengine.core import build
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.kernels.hip_gfx1100.quant import gguf_k_gemv as kernels

    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    # A real, already-cached shared object. No compiler or kernel edits.
    pinned = kernels.build_gguf_k_gemv(require_cached=True)
    loaded = []
    launches = []
    explicit = [False]
    real_build = kernels.build_gguf_k_gemv
    real_launch = kernels._launch

    def builder(**kwargs):
        result = real_build(**kwargs)
        loaded.append(result)
        return result

    def launch(*args, **kwargs):
        start = len(loaded)
        if explicit[0]:
            kwargs['library'] = pinned
        result = real_launch(*args, **kwargs)
        used = loaded[start:]
        if explicit[0]:
            assert not used, 'explicit handle unexpectedly consulted the builder'
        else:
            assert len(used) == 1, 'implicit wrapper did not resolve its own library'
        launches.append((args[1], pinned if explicit[0] else used[0]))
        return result

    monkeypatch.setattr(kernels, 'build_gguf_k_gemv', builder)
    monkeypatch.setattr(kernels, '_launch', launch)
    captures = []
    _install_logit_capture(monkeypatch, captures)
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1, prefix_cache='off')
    try:
        generator = _unwrap(llm._get_text_generator())
        serving = generator.create_resident_model_runner(capacity=1)
        serving.prepare(max_sequence_length=CONTEXT)

        def arm(use_explicit=False):
            explicit[0] = use_explicit
            launches.clear()
            captures.clear()
            output = llm.generate_detailed(
                [[1]], SamplingParams(max_tokens=3, temperature=0, ignore_eos=True,
                                      speculative_mtp=False))
            assert launches and captures
            assert all(np.isfinite(row).all() for row in captures)
            assert len(output[0].generated_token_ids) == 3
            return (list(launches), [row.copy() for row in captures],
                    tuple(output[0].generated_token_ids))

        arm()  # first-use graph construction is not a steady launch sequence
        warm = arm()
        assert all(handle is pinned for _, handle in warm[0])
        build.invalidate_build_resolution()
        fresh = arm()
        fresh_handle = fresh[0][0][1]
        assert fresh_handle is not pinned
        assert all(handle is fresh_handle for _, handle in fresh[0])
        fixed = arm(True)
        assert all(handle is pinned for _, handle in fixed[0])
        # Observe changed raw build environment without selecting an
        # incompatible target. Normalization resolves the same architecture.
        prior_arch = os.environ.get('HIPENGINE_HIP_ARCH')
        assert prior_arch == 'gfx1151'
        monkeypatch.setenv('HIPENGINE_HIP_ARCH', ' gfx1151 ')
        alias = arm()
        assert all(handle is fresh_handle for _, handle in alias[0])
        monkeypatch.setenv('HIPENGINE_HIP_ARCH', prior_arch)
        repeat = arm()
        for result in (fresh, fixed, alias, repeat):
            assert [symbol for symbol, _ in result[0]] == [symbol for symbol, _ in warm[0]]
            assert result[2] == warm[2]
            assert len(result[1]) == len(warm[1])
            for expected, actual in zip(warm[1], result[1], strict=True):
                np.testing.assert_array_equal(expected, actual)
    finally:
        llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for name in ('current_allocated_bytes', 'active_allocations'):
        assert after[name] == before[name]
