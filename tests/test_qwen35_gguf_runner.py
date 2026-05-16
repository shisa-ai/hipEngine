from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner, Qwen35GGUFOneLayerProbe

MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")


def test_qwen35_gguf_one_layer_probe_runs_finite_deterministic_hidden() -> None:
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    with Qwen35GGUFOneLayerProbe(MODEL, layer_id=0) as probe:
        first = probe.run_token(760)
        second = probe.run_token(760)
        sample1 = probe.sample_next_token(760)
        sample2 = probe.sample_next_token(760)

    assert first.shape == (1, 1024)
    assert first.dtype == np.uint16
    assert np.array_equal(first, second)
    f32 = bf16_to_float32(first)
    assert np.all(np.isfinite(f32))
    assert int(np.count_nonzero(f32)) > 0
    assert sample1.logits.shape == (1, 248320)
    assert np.all(np.isfinite(sample1.logits))
    assert sample1.token_id == sample2.token_id
    assert sample1.logit == sample2.logit


def test_qwen35_gguf_full_stack_runs_finite_deterministic_hidden() -> None:
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    with Qwen35GGUFFullStackRunner(MODEL) as runner:
        first = runner.run_prompt_hidden([760, 4087, 369])
        second = runner.run_prompt_hidden([760, 4087, 369])

    assert first.shape == (1, 1024)
    assert first.dtype == np.uint16
    assert np.array_equal(first, second)
    f32 = bf16_to_float32(first)
    assert np.all(np.isfinite(f32))
    assert int(np.count_nonzero(f32)) > 0


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True
