"""Public PARO shared-prefill fusion survives retired environment settings."""
from contextlib import closing
import ctypes
import os
from pathlib import Path

import pytest

from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
import hipengine.runtime.qwen35_paro as paro


def _hip_available():
    try:
        runtime = ctypes.CDLL("libamdhip64.so")
        count = ctypes.c_int()
        return runtime.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value > 0
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is unavailable")


def test_public_shared_prefill_uses_pair_rotate_with_retired_opt_out(monkeypatch):
    model = Path(os.environ.get(
        "HIPENGINE_TEST_PARO_MTP_MODEL",
        "/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16",
    ))
    if not model.is_dir():
        pytest.skip("packed PARO model is unavailable")
    before = memory_stats()
    original = paro.silu_mul_pair_rotate_out_fp16
    calls = []

    def traced(*args, **kwargs):
        calls.append(int(args[6]))
        return original(*args, **kwargs)

    monkeypatch.setattr(paro, "silu_mul_pair_rotate_out_fp16", traced)
    params = SamplingParams(max_tokens=2, temperature=0, ignore_eos=True, kv_storage="bf16")
    with closing(LLM(str(model), backend="auto", quant="w4_paro", max_active_requests=1,
                     max_sequence_length=128, prefix_cache="off")) as llm:
        for length in (15, 17, 37):
            outputs = []
            for setting in (None, "0"):
                if setting is None:
                    monkeypatch.delenv("HIPENGINE_SHARED_PREFILL_SILU_ROTATE_FUSED", raising=False)
                else:
                    monkeypatch.setenv("HIPENGINE_SHARED_PREFILL_SILU_ROTATE_FUSED", setting)
                calls.clear()
                output = llm.generate([[9707] * length], params)
                assert len(output) == 1
                assert calls and length in calls, (length, setting, calls)
                outputs.append(output)
            assert outputs[0] == outputs[1]
    after = memory_stats()
    assert after["active_allocations"] == before["active_allocations"]
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
