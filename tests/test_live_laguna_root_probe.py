from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from scripts.laguna_root_probe import run_laguna_root_probe

MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not MODEL.exists(), reason=f"local Laguna GGUF not found: {MODEL}")
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_real_root_probe_matches_raw_cpu_reference() -> None:
    result = run_laguna_root_probe(MODEL, backend="hip_gfx1151", token_id=100257)

    assert result["pass"] is True
    assert result["embedding_max_abs"] == 0.0
    assert result["output_norm_max_abs"] == 0.0
    assert result["finite_logits"] is True
    assert result["kl_divergence"] <= 0.05
    assert result["top1_agreement"] >= 0.9
    assert result["cpu_top1"] == result["gpu_top1"] == 81364
