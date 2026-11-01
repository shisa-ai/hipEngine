"""Public paired launches use variant-aware lookup without changing defaults."""
import ctypes
from pathlib import Path

import pytest


def _hip_available():
    try:
        ctypes.CDLL('libamdhip64.so')
        return True
    except OSError:
        return False


MODEL = Path('/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf')
pytestmark = pytest.mark.skipif(
    not _hip_available() or not MODEL.exists(), reason='requires ROCm and local GGUF')


def test_public_pair_libraries_use_variant_owner(monkeypatch):
    from hipengine import LLM, SamplingParams
    from hipengine.runtime import gguf_linear as gl
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats

    real = gl._variant_scoped_library
    calls = []

    def lookup(libraries, key, **kwargs):
        result = real(libraries, key, **kwargs)
        if key.layer in {'linear_pair', 'linear_pair_silu'}:
            calls.append((key, result))
        return result

    monkeypatch.setattr(gl, '_variant_scoped_library', lookup)
    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    llm = LLM(str(MODEL), backend='hip_gfx1151', max_sequence_length=256,
              max_active_requests=1, prefix_cache='off')
    try:
        params = SamplingParams(max_tokens=3, temperature=0, ignore_eos=True,
                                kv_storage='bf16', speculative_mtp=False)
        first = llm.generate_detailed([[9707] * 37], params)
        assert len(first[0].generated_token_ids) == 3
        assert calls, 'public generation did not reach a paired variant lookup'
        calls.clear()
        second = llm.generate_detailed([[9707] * 37], params)
        assert calls, 'repeat bypassed paired variant lookup'
        assert first[0].generated_token_ids == second[0].generated_token_ids
    finally:
        llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for name in ('current_allocated_bytes', 'active_allocations'):
        assert after[name] == before[name]
