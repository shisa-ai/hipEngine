"""WPF-H8B post-production candidate-seam cleanup contract."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PRODUCTION_ARTIFACT = _ROOT / (
    "benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-"
    "scoped-activation-pack-reuse-production.json"
)
_PRODUCTION_ARTIFACT_SHA256 = (
    "d7f62709a66d255caf8e7a8a4ec2eaf9ce713916c7b78ffaecd7e68d2ff69d91"
)
_SOURCE_CAPABILITY = "LAGUNA_ACTIVATION_PACK_REUSE"
_REMOVED_SUPPORTED_CAPABILITY = "LAGUNA_ACTIVATION_PACK_REUSE_SUPPORTED"


def test_h8b_cleanup_collapses_positive_candidate_seam_after_publication() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.runtime import laguna_gguf_runner as runner

    artifact_bytes = _PRODUCTION_ARTIFACT.read_bytes()
    assert hashlib.sha256(artifact_bytes).hexdigest() == _PRODUCTION_ARTIFACT_SHA256
    artifact = json.loads(artifact_bytes)
    assert artifact["status"] == "retained_production_default"
    assert artifact["decision"]["source_promoted"] is True
    assert artifact["decision"]["disabled_rollback_retained"] is True
    assert artifact["clean_production_wall"]["median_tok_s"] > 440.353372
    assert artifact["clean_production_profile"]["activation_packs_each"] == 223
    assert artifact["clean_production_profile"]["application_dispatches_each"] == 2_155
    assert artifact["scope_contract"]["complete_recurrence_groups"] == 95
    assert artifact["scope_contract"]["removed_pack_calls"] == 107

    assert getattr(hip_gfx1100, _SOURCE_CAPABILITY) is True
    assert not hasattr(hip_gfx1151, _SOURCE_CAPABILITY)

    # Sole intentional RED: source is now the only positive route, so the
    # duplicate support capability/export and explicit positive selector go.
    assert not hasattr(hip_gfx1100, _REMOVED_SUPPORTED_CAPABILITY)
    assert not hasattr(hip_gfx1151, _REMOVED_SUPPORTED_CAPABILITY)
    assert _REMOVED_SUPPORTED_CAPABILITY not in hip_gfx1100.__all__

    resolve = runner.resolve_laguna_activation_pack_reuse
    assert resolve("hip_gfx1100", None) is True
    assert resolve("hip_gfx1100", False) is False
    assert resolve("hip_gfx1151", None) is False
    with pytest.raises(ValueError, match="positive selector was removed"):
        resolve("hip_gfx1100", True)
    with pytest.raises(ValueError, match="positive selector was removed"):
        resolve("hip_gfx1151", True)

    resolver_source = inspect.getsource(runner.resolve_laguna_activation_pack_reuse)
    session_source = inspect.getsource(runner.LagunaGGUFResidentSession.__init__)
    assert _REMOVED_SUPPORTED_CAPABILITY not in resolver_source
    assert _SOURCE_CAPABILITY in resolver_source
    assert "activation_pack_reuse_request = (" not in session_source
    assert "use_activation_pack_reuse is False" in session_source
