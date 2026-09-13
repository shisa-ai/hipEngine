"""Real-model all-history correctness gate for UD-Q3_K_M bulk prefill."""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from scripts.qwen35_gguf_bulk_parity import _sample_serial_and_bulk, _scan_layer_drift

_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf")
_TOKENS = (9707, 11, 220, 264, 73, 13, 107561, 13883) * 8
_MIN_FREE_VRAM_BYTES = 18 * 1024**3


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.fixture
def require_q3_model_vram() -> None:
    """Skip when another in-process GPU fixture leaves too little model headroom."""

    from hipengine.core.hip import get_hip_runtime

    free_bytes, _ = get_hip_runtime().mem_get_info()
    if free_bytes < _MIN_FREE_VRAM_BYTES:
        pytest.skip(
            f"UD-Q3_K_M parity needs 18 GiB free VRAM; only {free_bytes / 1024**3:.2f} GiB available"
        )


@pytest.mark.skipif(not _MODEL.exists(), reason=f"local GGUF fixture not found: {_MODEL}")
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_ud_q3_k_m_fully_bulk_prefill_passes_serial_correctness_gate_mixed64(
    require_q3_model_vram: None,
) -> None:
    """Merged F32-router semantics must stay within the project correctness gate."""

    sample = _sample_serial_and_bulk(
        _MODEL,
        list(_TOKENS),
        compiler_version=None,
        require_cached_build=False,
        prefill_quant="gguf_ud_q3_k_m",
        attn_aotriton_min_tokens=0,
    )

    for route in ("default", "native_attention_bulk_ffn", "fast_bulk_attention"):
        comparison = sample[f"{route}_comparison"]
        assert comparison["top1_match"]
        assert comparison["kl_serial_to_bulk"] <= 0.05
        assert comparison["finite"]

    native = sample["native_attention_bulk_ffn_comparison"]
    assert native["kl_serial_to_bulk"] == 0.0
    assert native["max_abs_logit"] == 0.0


@pytest.mark.skipif(not _MODEL.exists(), reason=f"local GGUF fixture not found: {_MODEL}")
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_ud_q3_k_m_fully_bulk_prefill_matches_first_attention_boundary(
    require_q3_model_vram: None,
) -> None:
    """Pin all-row parity through the first full-attention boundary."""

    scan = _scan_layer_drift(
        _MODEL,
        [9419, 11, 271, 40],
        (0, 3, 4),
        prefill_quant="gguf_ud_q3_k_m",
        attn_aotriton_min_tokens=0,
    )

    for route in ("aotriton_full_attention", "native_full_attention"):
        assert scan[route]["first_drift_limit"] is None
        for entry in scan[route]["entries"]:
            assert entry["bit_equal"]
            assert entry["nonzero_count"] == 0
            assert entry["mismatched_rows"] == []
            assert entry["row_mismatch_counts"] == [0, 0, 0, 0]
