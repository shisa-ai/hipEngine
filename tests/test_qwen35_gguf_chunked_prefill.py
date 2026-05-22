from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFResidentSession,
    _chunk_ranges,
    _gguf_aotriton_prefill_mode,
)

MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"local GGUF fixture not found: {MODEL}")


def test_gguf_chunk_ranges_merge_tiny_tail() -> None:
    assert _chunk_ranges(4097, 4096, min_chunk_size=4) == ((0, 4097),)
    assert _chunk_ranges(8193, 4096, min_chunk_size=4) == ((0, 4096), (4096, 8193))


def test_gguf_aotriton_prefill_mode_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_AOTRITON_PREFILL", raising=False)
    assert _gguf_aotriton_prefill_mode(0, 4096, 4096) == "v3"
    assert _gguf_aotriton_prefill_mode(4096, 4096, 8192) == "v3"

    monkeypatch.setenv("HIPENGINE_GGUF_AOTRITON_PREFILL", "auto")
    assert _gguf_aotriton_prefill_mode(0, 4096, 4096) == "v2"
    assert _gguf_aotriton_prefill_mode(4096, 4096, 8192) == "v3"

    monkeypatch.setenv("HIPENGINE_GGUF_AOTRITON_PREFILL", "v2")
    assert _gguf_aotriton_prefill_mode(0, 4096, 4096) == "v2"
    with pytest.raises(ValueError, match="only valid for full-context prefill"):
        _gguf_aotriton_prefill_mode(4096, 4096, 8192)


def test_qwen35_gguf_chunked_prefill_matches_unchunked() -> None:
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    # 8 tokens prompt to test chunking into 2 chunks of size 4
    prompt_ids = [760, 4087, 369, 220, 760, 4087, 369, 220]

    with Qwen35GGUFResidentSession(MODEL, max_sequence_length=16, prefill_chunk_size=999999) as unchunked:
        unchunked_res = unchunked.prefill(prompt_ids, use_bulk=True)

    with Qwen35GGUFResidentSession(MODEL, max_sequence_length=16, prefill_chunk_size=4) as chunked:
        chunked_res = chunked.prefill(prompt_ids, use_bulk=True)

    assert chunked_res.token_id == unchunked_res.token_id
    assert chunked_res.logits.shape == unchunked_res.logits.shape == (1, 248320)
    assert np.all(np.isfinite(chunked_res.logits))
    assert _kl_divergence(unchunked_res.logits.reshape(-1), chunked_res.logits.reshape(-1)) <= 0.1


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
