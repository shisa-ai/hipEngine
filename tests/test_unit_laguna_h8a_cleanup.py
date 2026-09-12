"""WPF-H8A post-production candidate-seam cleanup contract."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PRODUCTION_ARTIFACT = _ROOT / (
    "benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-"
    "resident-q5-global-f32-cache-production.json"
)
_PRODUCTION_ARTIFACT_SHA256 = (
    "7f3560515c8a8470e4127eb4885db7c693467ef78839c335dbcbb81f7f960748"
)
_SOURCE_CAPABILITY = "LAGUNA_Q5_F32_RESIDENT_GLOBAL_CACHE"
_REMOVED_SUPPORTED_CAPABILITY = (
    "LAGUNA_Q5_F32_RESIDENT_GLOBAL_CACHE_SUPPORTED"
)


def test_h8a_cleanup_collapses_positive_candidate_seam_after_publication() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.runtime import laguna_gguf_runner as runner

    artifact_bytes = _PRODUCTION_ARTIFACT.read_bytes()
    assert hashlib.sha256(artifact_bytes).hexdigest() == _PRODUCTION_ARTIFACT_SHA256
    artifact = json.loads(artifact_bytes)
    assert artifact["status"] == "retained_production_default"
    assert artifact["decision"]["source_promoted"] is True
    assert artifact["decision"]["transient_h7g_rollback_retained"] is True
    assert artifact["clean_production_wall"]["median_tok_s"] > 437.189274
    assert artifact["clean_production_profile"]["application_dispatches_each"] == 2_262

    assert getattr(hip_gfx1100, _SOURCE_CAPABILITY) is True
    assert not hasattr(hip_gfx1100, _REMOVED_SUPPORTED_CAPABILITY)
    assert not hasattr(hip_gfx1151, _SOURCE_CAPABILITY)
    assert not hasattr(hip_gfx1151, _REMOVED_SUPPORTED_CAPABILITY)
    assert _REMOVED_SUPPORTED_CAPABILITY not in hip_gfx1100.__all__

    resolve = runner.resolve_laguna_q5_f32_resident_global_cache
    assert resolve("hip_gfx1100", None) is True
    assert resolve("hip_gfx1100", False) is False
    assert resolve("hip_gfx1151", None) is False
    with pytest.raises(ValueError, match="positive selector was removed"):
        resolve("hip_gfx1100", True)
    with pytest.raises(ValueError, match="positive selector was removed"):
        resolve("hip_gfx1151", True)

    resolver_source = inspect.getsource(
        runner.resolve_laguna_q5_f32_resident_global_cache
    )
    allocation_source = inspect.getsource(
        runner.LagunaQ5F32ResidentGlobalCache.allocate
    )
    session_source = inspect.getsource(runner.LagunaGGUFResidentSession.__init__)
    assert _REMOVED_SUPPORTED_CAPABILITY not in resolver_source
    assert _REMOVED_SUPPORTED_CAPABILITY not in allocation_source
    assert _SOURCE_CAPABILITY in resolver_source
    assert _SOURCE_CAPABILITY in allocation_source
    assert "resident_cache_request = (" not in session_source
    assert "resident_q5_f32_cache is not None" in session_source
    assert "use_q5_f32_resident_global_cache is False" in session_source
