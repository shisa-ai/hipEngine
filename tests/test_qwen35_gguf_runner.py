from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFFullStackRunner,
    Qwen35GGUFOneLayerProbe,
    Qwen35GGUFResidentSession,
)

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


def test_qwen35_gguf_resident_decode_graph_matches_eager_logits() -> None:
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    prompt_ids = [760, 4087, 369]
    with Qwen35GGUFResidentSession(MODEL) as eager:
        eager_first = eager.prefill(prompt_ids)
        eager_second = eager.step(eager_first.token_id)
    with Qwen35GGUFResidentSession(MODEL) as graph_session:
        graph_first = graph_session.prefill(prompt_ids)
        with graph_session.capture_decode_graph(position=len(prompt_ids), max_replay_steps=1, record_steps=1) as graph:
            graph.replay(1)
            graph_ids = [graph_first.token_id, *graph.read_generated_token_ids(1)]
            graph_second = graph.read_sample()

    assert [eager_first.token_id, eager_second.token_id] == [220, 16]
    assert graph_ids == [220, 16]
    assert graph_second.token_id == eager_second.token_id
    assert graph_second.logits.shape == eager_second.logits.shape == (1, 248320)
    assert np.all(np.isfinite(graph_second.logits))
    assert float(np.max(np.abs(graph_second.logits - eager_second.logits))) == 0.0


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True
