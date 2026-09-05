"""WPF-H8A resident global-Q5 F32 cache source-default RED contract."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_ARTIFACT = _ROOT / (
    "benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-"
    "resident-q5-global-f32-cache-runtime-candidate.json"
)
_RUNTIME_ARTIFACT_SHA256 = (
    "50ed3cd81a5a350ac39e70eaed3560d88144251e40f07198acbbb10e2ae1a325"
)
_PACKAGE = _ROOT / "hipengine/kernels/hip_gfx1100/__init__.py"
_SOURCE_CAPABILITY = "LAGUNA_Q5_F32_RESIDENT_GLOBAL_CACHE"
_REMOVED_SUPPORTED_CAPABILITY = (
    "LAGUNA_Q5_F32_RESIDENT_GLOBAL_CACHE_SUPPORTED"
)
_PLANE_BYTES = 75_497_472
_PLANE_COUNT = 24
_RESIDENT_BYTES = 1_811_939_328
_SOURCE_TOPOLOGY = {
    "setup_coltile16_producers": 24,
    "request_coltile16_producers": 0,
    "request_target_activation_packs": 24,
    "request_h7g_consumers": 24,
    "request_application_dispatches": 2_262,
    "queues": 1,
    "streams": 1,
}
_ROLLBACK_TOPOLOGY = {
    "setup_coltile16_producers": 0,
    "request_coltile16_producers": 24,
    "request_target_activation_packs": 24,
    "request_h7g_consumers": 24,
    "request_application_dispatches": 2_286,
    "queues": 1,
    "streams": 1,
}
_SOURCE_ADMISSION = {
    "complete_planes": 24,
    "complete_state_exact": True,
    "fixed_c4096_m512_samples": 5,
    "lengths": (512, 1_024, 4_096),
    "samples_per_length": 3,
    "every_source_median_must_win": True,
    "named_request_dispatches": 2_262,
    "post_commit_wall_samples": 5,
    "post_commit_profiled_requests": 5,
    "transient_h7g_rollback_required": True,
    "candidate_capability_cleanup_after_checkpoint": True,
    "no_subset_or_favorable_rerun": True,
}
_NORMALIZED_PACKAGE_SHA256 = (
    "061b70c782485038633d1f6f2564d738dc522d42a44bf339044c22d13ecb27b8"
)
_SOURCE_SHA256 = {
    "hipengine/kernels/hip_gfx1100/quant/gguf_q5_k_f32_rocblas_prefill.py": (
        "fb9b2ae1a88300ac1e754b8c3214310db65d3e2343598b7631ac185ec141f33e"
    ),
    "hipengine/kernels/hip_gfx1100/quant/gguf_q5_k_f32_rocblas_prefill.hip": (
        "1a06011ea6e7bda8e0b48fd357cbcbadaff76793a1b5c49bd217cc83d32b7110"
    ),
    "hipengine/kernels/hip_gfx1151/__init__.py": (
        "a5838ffc8fd8df367cd828f397e701f94f2268c7992d0a5e143c8d7e2b8ba3b3"
    ),
    "hipengine/runtime/gguf_linear.py": (
        "f9ebb089b31937dcaea27f8bb43bfc2936b294d541c2841465c498d6f6dbd363"
    ),
    "hipengine/runtime/laguna_gguf_runner.py": (
        "edea1fc2df3c8ca46fe3396663ac14f9000b4ee0cc967ebafb55208afad50654"
    ),
    "tests/test_laguna_h8a_resident_q5_global_cache.py": (
        "fca107c250f9f510c43c1bd324c9e0d464040fdd28046bb60aff80e76ffb8dd8"
    ),
}
_POST_MERGE_SOURCE_SHA256 = {
    # Later Qwen3.8 and execution-profile policies do not alter H8A's owner.
    "hipengine/kernels/hip_gfx1151/__init__.py": (
        "fb30ffbf954bb9a1255f2cd55f0484aade2ee54f50ca765dec078a866f9d0842"
    ),
    "hipengine/runtime/gguf_linear.py": (
        "63538f3a671d53d3bfe1fa3d6873bcb5e41e892b9195d91eb091bdac76192dad"
    ),
    "hipengine/runtime/laguna_gguf_runner.py": (
        "ae45f9e3e39fd93f971e5aa0b3394b3e5ce0a797b7cef8a9e1a20b1f2a133825"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_package_sha256() -> str:
    source = _PACKAGE.read_text()
    disabled = f"{_SOURCE_CAPABILITY} = False"
    enabled = f"{_SOURCE_CAPABILITY} = True"
    assert source.count(disabled) + source.count(enabled) == 1
    normalized = source.replace(
        disabled,
        f"{_SOURCE_CAPABILITY} = <SOURCE>",
    ).replace(
        enabled,
        f"{_SOURCE_CAPABILITY} = <SOURCE>",
    )
    return hashlib.sha256(normalized.encode()).hexdigest()


def test_h8a_source_red_pins_qualified_owner_and_promotion_contract() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.runtime import laguna_gguf_runner as runner

    artifact_bytes = _RUNTIME_ARTIFACT.read_bytes()
    assert hashlib.sha256(artifact_bytes).hexdigest() == _RUNTIME_ARTIFACT_SHA256
    artifact = json.loads(artifact_bytes)
    assert artifact["status"] == "qualified_bounded_default_off"
    assert artifact["decision"]["candidate_admitted"] is True
    assert artifact["decision"]["runtime_owner_retained"] is True
    assert artifact["decision"]["source_default_changed"] is False
    assert artifact["candidate"]["resident_allocations"] == _PLANE_COUNT
    assert artifact["candidate"]["plane_bytes_each"] == _PLANE_BYTES
    assert artifact["candidate"]["resident_bytes"] == _RESIDENT_BYTES
    assert artifact["complete_planes"]["complete_plane_matches"] == _PLANE_COUNT
    assert artifact["complete_state"]["kl_max"] == 0.0
    assert artifact["complete_state"]["top1_agreement"] == 1.0
    assert artifact["complete_state"]["hidden_boundaries_exact"] == 48
    assert artifact["complete_state"]["kv_and_spans_exact"] is True
    assert artifact["fixed_c4096_m512"]["gain_percent"] > 0
    assert artifact["fixed_c4096_m512"]["paired_candidate_wins"] == 5
    assert all(
        artifact["clean_length_gate"][length]["gain_percent"] > 0
        and artifact["clean_length_gate"][length]["paired_candidate_wins"] == 3
        for length in ("512", "1024", "4096")
    )
    trace = artifact["named_trace"]
    assert {
        "setup_coltile16_producers": trace["setup_producers"],
        "request_coltile16_producers": trace["request_producers"],
        "request_target_activation_packs": trace[
            "request_target_activation_packs"
        ],
        "request_h7g_consumers": trace["request_h7g_consumers"],
        "request_application_dispatches": trace[
            "request_application_dispatches"
        ],
        "queues": trace["queues"],
        "streams": trace["streams"],
    } == _SOURCE_TOPOLOGY
    assert _ROLLBACK_TOPOLOGY == {
        "setup_coltile16_producers": 0,
        "request_coltile16_producers": 24,
        "request_target_activation_packs": 24,
        "request_h7g_consumers": 24,
        "request_application_dispatches": 2_286,
        "queues": 1,
        "streams": 1,
    }
    assert _SOURCE_ADMISSION == {
        "complete_planes": 24,
        "complete_state_exact": True,
        "fixed_c4096_m512_samples": 5,
        "lengths": (512, 1_024, 4_096),
        "samples_per_length": 3,
        "every_source_median_must_win": True,
        "named_request_dispatches": 2_262,
        "post_commit_wall_samples": 5,
        "post_commit_profiled_requests": 5,
        "transient_h7g_rollback_required": True,
        "candidate_capability_cleanup_after_checkpoint": True,
        "no_subset_or_favorable_rerun": True,
    }
    assert getattr(hip_gfx1100, _SOURCE_CAPABILITY) is True
    assert not hasattr(hip_gfx1100, _REMOVED_SUPPORTED_CAPABILITY)
    assert _normalized_package_sha256() == _NORMALIZED_PACKAGE_SHA256
    parameters = inspect.signature(runner.LagunaGGUFResidentSession.__init__).parameters
    assert "resident_q5_f32_cache" in parameters
    assert "use_q5_f32_resident_global_cache" in parameters
    allocation_source = inspect.getsource(
        runner.LagunaQ5F32ResidentGlobalCache.allocate
    )
    assert "len(targets) != 24 or len(raw_ptrs) != 24" in allocation_source
    assert "runtime.device_synchronize()" in allocation_source
    session_source = inspect.getsource(runner.LagunaGGUFResidentSession.__init__)
    assert "self.use_q5_f32_resident_global_cache" in session_source
    assert "and self._owns_weights" in session_source
    for relative, expected in _SOURCE_SHA256.items():
        assert _sha256(_ROOT / relative) == _POST_MERGE_SOURCE_SHA256.get(
            relative, expected
        )


def test_h8a_source_default_selects_owner_and_preserves_transient_rollback() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.runtime.laguna_gguf_runner import (
        resolve_laguna_q5_f32_resident_global_cache,
    )

    live_source = getattr(hip_gfx1100, _SOURCE_CAPABILITY)

    # Source ownership is now the sole positive policy; explicit false keeps
    # transient H7G rollback and the former positive candidate seam is removed.
    assert live_source is True
    assert not hasattr(hip_gfx1100, _REMOVED_SUPPORTED_CAPABILITY)
    assert not hasattr(hip_gfx1151, _SOURCE_CAPABILITY)
    assert not hasattr(hip_gfx1151, _REMOVED_SUPPORTED_CAPABILITY)
    assert resolve_laguna_q5_f32_resident_global_cache("hip_gfx1100", None) is True
    assert resolve_laguna_q5_f32_resident_global_cache("hip_gfx1100", False) is False
    assert resolve_laguna_q5_f32_resident_global_cache("hip_gfx1151", None) is False
    with pytest.raises(ValueError, match="positive selector was removed"):
        resolve_laguna_q5_f32_resident_global_cache("hip_gfx1100", True)
    with pytest.raises(ValueError, match="positive selector was removed"):
        resolve_laguna_q5_f32_resident_global_cache("hip_gfx1151", True)
    assert _normalized_package_sha256() == _NORMALIZED_PACKAGE_SHA256
