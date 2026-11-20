"""Public ordinary-prefill convolution selection and teardown."""
from __future__ import annotations

from pathlib import Path

import pytest

_MODEL = Path('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')


@pytest.mark.parametrize('mode', ['baseline', 'tile32x128'])
def test_public_ordinary_prefill_selects_conv_owner(monkeypatch, mode, hip_test_target_arch):
    if hip_test_target_arch not in {'gfx1100', 'gfx1151'}:
        pytest.skip('requires gfx1100 or gfx1151')
    if not _MODEL.exists():
        pytest.skip('requires dense GGUF checkpoint')
    from hipengine import LLM, SamplingParams
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.kernels.backends import HIP_TARGET_ARCH_BACKEND
    from hipengine.runtime import qwen35_gguf_runner as runner

    backend = HIP_TARGET_ARCH_BACKEND[hip_test_target_arch]
    monkeypatch.setenv('HIPENGINE_GGUF_LINEAR_ATTN_CONV_PREFILL_MODE', mode)
    selected = []
    original = runner.Qwen35GGUFFullStackRunner._linear_attn_conv_prefill_kernel

    def select(self):
        kernel = original(self)
        selected.append(kernel)
        return kernel

    monkeypatch.setattr(runner.Qwen35GGUFFullStackRunner,
                        '_linear_attn_conv_prefill_kernel', select)
    before = memory_stats()
    llm = LLM(str(_MODEL), backend=backend, max_active_requests=1,
              max_sequence_length=256, prefix_cache='off')
    try:
        params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True,
                                speculative_mtp=False)
        outputs = [llm.generate_detailed([[9707] * 73], params)[0]
                   for _ in range(2)]
        assert len(outputs[0].generated_token_ids) == 2
        assert outputs[0].generated_token_ids == outputs[1].generated_token_ids
        expected = runner.resolve(
            backend=backend, layer='linear_attn_conv_prefill',
            quant='gguf_qwen35',
            variant={'baseline': 'f32_baseline', 'tile32x128': 'f32_tile32x128'}[mode])
        assert selected and all(kernel is expected for kernel in selected)
    finally:
        llm.close()
    get_hip_runtime().device_synchronize()
    after = memory_stats()
    for key in ('active_allocations', 'current_allocated_bytes'):
        assert after[key] == before[key]
