from __future__ import annotations

import ctypes
from pathlib import Path
from types import SimpleNamespace

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


def test_qwen35_gguf_resident_session_can_allocate_benchmark_length_cache() -> None:
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    with Qwen35GGUFResidentSession(MODEL, max_sequence_length=512 + 128 + 1) as session:
        assert session.scratch is not None
        assert session.scratch.max_positions >= 512 + 128 + 1
        assert session.scratch.block_table_tensor.numel >= 3


def test_qwen35moe_prefill_default_selects_fast_bulk_with_native_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(
        weights=SimpleNamespace(
            config=SimpleNamespace(
                is_moe=True,
                ssm_conv_kernel=4,
            )
        )
    )
    calls: list[tuple[list[int], str, bool]] = []

    def fake_bulk_prefill(
        self: Qwen35GGUFResidentSession,
        token_ids: list[int] | tuple[int, ...],
        *,
        bulk_attention_mode: str,
        return_logits: bool,
    ) -> SimpleNamespace:
        _ = self
        calls.append((list(token_ids), bulk_attention_mode, return_logits))
        return SimpleNamespace(token_id=42, logit=1.0, logits=None)

    monkeypatch.setattr(Qwen35GGUFResidentSession, "_run_bulk_prefill_and_sample", fake_bulk_prefill)

    default = session.prefill([760, 4087, 369, 220], return_logits=False)
    native = session.prefill([760, 4087, 369, 220], bulk_attention_mode="native", return_logits=True)

    assert default.token_id == native.token_id == 42
    assert calls == [
        ([760, 4087, 369, 220], "bulk", False),
        ([760, 4087, 369, 220], "native", True),
    ]


def test_qwen35_gguf_bulk_prefill_matches_serial_for_conv_length_prompt() -> None:
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    prompt_ids = [760, 4087, 369, 220]
    with Qwen35GGUFResidentSession(MODEL, max_sequence_length=16) as serial:
        serial_first = serial.prefill(prompt_ids, use_bulk=False)
    with Qwen35GGUFResidentSession(MODEL, max_sequence_length=16) as bulk:
        bulk_first = bulk.prefill(prompt_ids, use_bulk=True)

    assert bulk_first.token_id == serial_first.token_id
    assert bulk_first.logits.shape == serial_first.logits.shape == (1, 248320)
    assert np.all(np.isfinite(bulk_first.logits))
    assert _kl_divergence(serial_first.logits.reshape(-1), bulk_first.logits.reshape(-1)) <= 0.05
    assert float(np.max(np.abs(bulk_first.logits - serial_first.logits))) <= 0.2


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


def _kl_divergence(reference_logits: np.ndarray, candidate_logits: np.ndarray) -> float:
    ref = reference_logits.astype(np.float64, copy=False)
    cand = candidate_logits.astype(np.float64, copy=False)
    ref_exp = np.exp(ref - float(np.max(ref)))
    cand_exp = np.exp(cand - float(np.max(cand)))
    ref_prob = ref_exp / float(np.sum(ref_exp))
    cand_prob = cand_exp / float(np.sum(cand_exp))
    return float(np.sum(ref_prob * (np.log(ref_prob + 1.0e-30) - np.log(cand_prob + 1.0e-30))))


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True
