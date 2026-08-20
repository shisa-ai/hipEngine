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
_MODEL_REQUIRED = pytest.mark.skipif(
    not MODEL.exists(),
    reason=f"local GGUF fixture not found: {MODEL}",
)


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


@_MODEL_REQUIRED
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


@_MODEL_REQUIRED
def test_qwen35_gguf_packed_prefill_returns_per_slot_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", "1")
    prompts = ([760, 4087, 369, 220], [760, 4087, 369, 221])

    with Qwen35GGUFResidentSession(MODEL, max_sequence_length=16) as owner:
        assert owner.runner is not None
        with Qwen35GGUFResidentSession(
            MODEL,
            shared_runner=owner.runner,
            max_sequence_length=16,
        ) as peer:
            packed = owner.prefill_batch_native(
                prompts,
                sessions=(owner, peer),
                return_logits=True,
            )
            packed_logits = [result.logits.copy() for result in packed if result is not None]
            packed_tokens = [int(result.token_id) for result in packed if result is not None]
            plan = dict(owner.last_packed_prefill_plan)

            owner.reset()
            peer.reset()
            scalar = [
                owner.prefill(prompts[0], return_logits=True),
                peer.prefill(prompts[1], return_logits=True),
            ]

    assert len(packed_logits) == len(packed_tokens) == len(scalar) == 2
    assert all(logits.shape == (1, 248320) for logits in packed_logits)
    assert all(np.all(np.isfinite(logits)) for logits in packed_logits)
    assert packed_tokens == [int(result.token_id) for result in scalar]
    assert all(
        _kl_divergence(result.logits.reshape(-1), logits.reshape(-1)) <= 0.05
        for result, logits in zip(scalar, packed_logits, strict=True)
    )
    assert plan["host_logits_d2h"] is True
    assert plan["host_logits_d2h_bytes"] == 2 * 248320 * np.dtype(np.float32).itemsize


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


_PRODUCTION_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf")


@pytest.mark.skipif(
    not _PRODUCTION_MODEL.exists(),
    reason=f"local GGUF fixture not found: {_PRODUCTION_MODEL}",
)
@pytest.mark.parametrize(
    ("state_env", "expected_fp16_state"),
    ((None, True), ("0", False), ("1", True)),
)
def test_qwen35_gguf_packed_ar_prefill_decode_runs_without_verify_capture(
    monkeypatch: pytest.MonkeyPatch,
    state_env: str | None,
    expected_fp16_state: bool,
) -> None:
    """Packed AR prefill+decode works in the production route (no verify-capture).

    Regression for the removed fail-closed guards that raised
    ``NotImplementedError`` when ``HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN``
    was unset.  The packed AR path is self-contained (segmented compact-peer
    per-slot state; c1-exact per-slot decode), so the production route must be
    supported.  Asserts the packed prefill+decode token streams match scalar
    prefill+decode on the same session pair.
    """
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", raising=False)
    if state_env is None:
        monkeypatch.delenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", state_env)
    from hipengine.runtime.qwen35_gguf_runner import (
        Qwen35GGUFResidentSession,
        _gguf_verify_capture_prefill_gdn_enabled,
    )
    assert not _gguf_verify_capture_prefill_gdn_enabled()

    prompt_a = [760, 4087, 369, 220, 760, 4087, 369, 220]
    prompt_b = [760, 4087, 369, 221, 760, 4087, 369, 221]

    with Qwen35GGUFResidentSession(
        _PRODUCTION_MODEL,
        backend="hip_gfx1151",
        max_sequence_length=64,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as owner:
        assert owner.runner is not None
        assert owner.runner.fp16_recurrent_state is expected_fp16_state
        with Qwen35GGUFResidentSession(
            _PRODUCTION_MODEL,
            shared_runner=owner.runner,
            backend="hip_gfx1151",
            max_sequence_length=64,
            use_wmma_prefill=True,
            use_gemv_decode=True,
        ) as peer:
            packed = owner.prefill_batch_native(
                (prompt_a, prompt_b),
                sessions=(owner, peer),
                return_logits=True,
            )
            packed_tokens = [int(res.token_id) for res in packed if res is not None]
            packed_logits = [
                np.ascontiguousarray(res.logits, dtype=np.float32)
                for res in packed
                if res is not None
            ]
            dec = owner.step_batch_native(
                tuple(packed_tokens),
                sessions=(owner, peer),
                return_logits=True,
            )
            dec_tokens = [int(res.token_id) for res in dec]

            owner.reset()
            peer.reset()
            scalar_a = owner.prefill(prompt_a, return_logits=True)
            scalar_b = peer.prefill(prompt_b, return_logits=True)
            scalar_tokens = [int(scalar_a.token_id), int(scalar_b.token_id)]
            scalar_logits = [
                np.ascontiguousarray(scalar_a.logits, dtype=np.float32),
                np.ascontiguousarray(scalar_b.logits, dtype=np.float32),
            ]
            dec_a = owner.step(scalar_tokens[0], return_logits=True)
            dec_b = peer.step(scalar_tokens[1], return_logits=True)
            scalar_dec = [int(dec_a.token_id), int(dec_b.token_id)]

    assert packed_tokens == scalar_tokens
    assert all(
        _kl_divergence(p.reshape(-1), s.reshape(-1)) <= 0.05
        for p, s in zip(packed_logits, scalar_logits, strict=True)
    )
    assert dec_tokens == scalar_dec
