"""WPF-H8B scoped activation-pack reuse source-default RED contract."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_ARTIFACT = _ROOT / (
    "benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-"
    "scoped-activation-pack-reuse-runtime-candidate.json"
)
_RUNTIME_ARTIFACT_SHA256 = (
    "ed3ee363ceb8b095b38fd28c0ebd1eb692d22dce61a4ce4ebd9e14be25cd73e7"
)
_PACKAGE = _ROOT / "hipengine/kernels/hip_gfx1100/__init__.py"
_SOURCE_CAPABILITY = "LAGUNA_ACTIVATION_PACK_REUSE"
_REMOVED_SUPPORTED_CAPABILITY = "LAGUNA_ACTIVATION_PACK_REUSE_SUPPORTED"
_NORMALIZED_PACKAGE_SHA256 = (
    "ddb58da4fa53896c34e7137ef0611984ea0536f78cf2d2b4e91082bd4aed18c7"
)
_RUNTIME_ARTIFACT_RUNNER_SHA256 = (
    "2f505e84319e7a3f8eecc6df69d521d8d0d66b47f6571f67a185629de86a6bbf"
)
_RUNTIME_ARTIFACT_H8B_TEST_SHA256 = (
    "710bcdc740ac691c2ac4f0fa5e9fa81f55be939b1582be78a9d62c6218ff06ff"
)
_RUNTIME_ARTIFACT_REFACTOR_SHA256 = (
    "3de213a389e980e7b84393acdae64e81eed0c93356ab10fac63605fbd2fcee71"
)
_SOURCE_SHA256 = {
    "hipengine/kernels/activation_pack.py": (
        "2b10234b49ee19417e439fa598b0b069ae4b832eebb3751357c7819891072f67"
    ),
    "hipengine/kernels/hip_gfx1100/quant/gguf_q5_k_f32_rocblas_prefill.py": (
        "fb9b2ae1a88300ac1e754b8c3214310db65d3e2343598b7631ac185ec141f33e"
    ),
    "hipengine/kernels/hip_gfx1100/quant/gguf_q5_k_f32_rocblas_prefill.hip": (
        "1a06011ea6e7bda8e0b48fd357cbcbadaff76793a1b5c49bd217cc83d32b7110"
    ),
    "hipengine/kernels/hip_gfx1151/__init__.py": (
        "a5838ffc8fd8df367cd828f397e701f94f2268c7992d0a5e143c8d7e2b8ba3b3"
    ),
    "hipengine/runtime/laguna_gguf_runner.py": (
        "edea1fc2df3c8ca46fe3396663ac14f9000b4ee0cc967ebafb55208afad50654"
    ),
    "hipengine/runtime/laguna_moe.py": (
        "0507c0ab9bcabddfda9d0390c66d46f80aaaf7c42357a58dfa24c692d43414fd"
    ),
    "tests/test_laguna_h8b_scoped_activation_pack_reuse.py": (
        "c8c7c949f9ca314aa3048e90f4d34481e486d5bc0f0cf307fb7fcf54200398e8"
    ),
}
_POST_MERGE_SOURCE_SHA256 = {
    # Later Qwen3.8 and execution-profile policies do not alter H8B's owner.
    "hipengine/kernels/hip_gfx1151/__init__.py": (
        "83383b28d43ca4a139f129274c4a9bcbeeb5c0a080284ed884624e34170179ce"
    ),
    "hipengine/runtime/laguna_gguf_runner.py": (
        "ae45f9e3e39fd93f971e5aa0b3394b3e5ce0a797b7cef8a9e1a20b1f2a133825"
    ),
    "hipengine/runtime/laguna_moe.py": (
        "b37bc2a1aaadbf94700dad9a67f90815b69d783a8a82fcc47b5496a17de83987"
    ),
}
_SOURCE_TOPOLOGY = {
    "request_activation_packs_before": 330,
    "request_activation_packs_after": 223,
    "request_dispatches_before": 2_262,
    "request_dispatches_after": 2_155,
    "queues": 1,
    "streams": 1,
    "compiler_processes": 0,
}
_ROLLBACK_TOPOLOGY = {
    "request_activation_packs": 330,
    "request_application_dispatches": 2_262,
    "queues": 1,
    "streams": 1,
}
_SOURCE_ADMISSION = {
    "complete_state_exact": True,
    "complete_recurrence_groups": 95,
    "removed_pack_calls": 107,
    "fixed_c4096_m512_samples": 5,
    "lengths": (512, 1_024, 4_096),
    "samples_per_length": 3,
    "every_source_median_must_win": True,
    "named_request_packs": 223,
    "named_request_dispatches": 2_155,
    "post_commit_wall_samples": 5,
    "post_commit_profiled_requests": 5,
    "disabled_rollback_required": True,
    "candidate_capability_cleanup_after_checkpoint": True,
    "no_subset_or_favorable_rerun": True,
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


def test_h8b_source_red_pins_qualified_owner_and_promotion_contract() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.runtime import laguna_gguf_runner as runner

    artifact_bytes = _RUNTIME_ARTIFACT.read_bytes()
    assert hashlib.sha256(artifact_bytes).hexdigest() == _RUNTIME_ARTIFACT_SHA256
    artifact = json.loads(artifact_bytes)
    assert artifact["status"] == "qualified_bounded_default_off"
    assert artifact["candidate"]["id"] == "WPF-H8B"
    assert artifact["decision"]["candidate_admitted"] is True
    assert artifact["decision"]["runtime_owner_retained"] is True
    assert artifact["decision"]["source_default_changed"] is False
    assert artifact["candidate"]["complete_recurrence_groups"] == 95
    assert artifact["candidate"]["removed_pack_calls"] == 107
    assert artifact["candidate"]["new_allocation_bytes"] == 0
    assert artifact["candidate"]["new_workspace_bytes"] == 0
    assert artifact["complete_state"]["state_exact"] is True
    assert artifact["complete_state"]["next_token_id"] == 2930
    assert artifact["complete_state"]["final_position"] == 511
    assert artifact["complete_state"]["activation_packs"] == 223
    assert artifact["complete_state"]["resident_packs"] == 24
    assert artifact["complete_state"]["transient_packs"] == 199

    trace = artifact["named_trace"]
    assert {
        "request_activation_packs_before": trace[
            "request_activation_packs_before"
        ],
        "request_activation_packs_after": trace[
            "request_activation_packs_after"
        ],
        "request_dispatches_before": trace["request_dispatches_before"],
        "request_dispatches_after": trace["request_dispatches_after"],
        "queues": trace["queues"],
        "streams": trace["streams"],
        "compiler_processes": trace["compiler_processes"],
    } == _SOURCE_TOPOLOGY
    assert trace["nonpack_kernel_names_and_counts_unchanged"] is True
    assert trace["removed_pack_counts_by_row_batch"] == {
        "4": 46,
        "5": 60,
        "12": 1,
    }

    fixed = artifact["fixed_c4096_m512"]
    assert fixed["samples_per_arm"] == 5
    assert fixed["candidate_median_tok_s"] > fixed["control_median_tok_s"]
    assert fixed["paired_candidate_wins"] == 4
    assert fixed["exact_all_samples"] is True
    assert all(
        artifact["clean_length_gate"][length]["candidate_median_tok_s"]
        > artifact["clean_length_gate"][length]["control_median_tok_s"]
        for length in ("512", "1024", "4096")
    )
    assert _ROLLBACK_TOPOLOGY == {
        "request_activation_packs": 330,
        "request_application_dispatches": 2_262,
        "queues": 1,
        "streams": 1,
    }
    assert _SOURCE_ADMISSION == {
        "complete_state_exact": True,
        "complete_recurrence_groups": 95,
        "removed_pack_calls": 107,
        "fixed_c4096_m512_samples": 5,
        "lengths": (512, 1_024, 4_096),
        "samples_per_length": 3,
        "every_source_median_must_win": True,
        "named_request_packs": 223,
        "named_request_dispatches": 2_155,
        "post_commit_wall_samples": 5,
        "post_commit_profiled_requests": 5,
        "disabled_rollback_required": True,
        "candidate_capability_cleanup_after_checkpoint": True,
        "no_subset_or_favorable_rerun": True,
    }

    assert getattr(hip_gfx1100, _SOURCE_CAPABILITY) is True
    assert not hasattr(hip_gfx1100, _REMOVED_SUPPORTED_CAPABILITY)
    assert _SOURCE_CAPABILITY in hip_gfx1100.__all__
    assert _REMOVED_SUPPORTED_CAPABILITY not in hip_gfx1100.__all__
    assert not hasattr(hip_gfx1151, _SOURCE_CAPABILITY)
    assert not hasattr(hip_gfx1151, _REMOVED_SUPPORTED_CAPABILITY)
    assert _normalized_package_sha256() == _NORMALIZED_PACKAGE_SHA256

    parameters = inspect.signature(runner.LagunaGGUFResidentSession.__init__).parameters
    assert "use_activation_pack_reuse" in parameters
    resolver_source = inspect.getsource(runner.resolve_laguna_activation_pack_reuse)
    assert _REMOVED_SUPPORTED_CAPABILITY not in resolver_source
    assert _SOURCE_CAPABILITY in resolver_source
    session_source = inspect.getsource(runner.LagunaGGUFResidentSession.__init__)
    assert "self.use_activation_pack_reuse" in session_source
    # Preserve the immutable artifact's historical documentation provenance,
    # but do not bind the live kernel/source RED to the entire mutable refactor
    # ledger. Current ownership is asserted semantically above.
    assert artifact["source_sha256"]["docs/REFACTOR.md"] == (
        _RUNTIME_ARTIFACT_REFACTOR_SHA256
    )
    for relative, expected in _SOURCE_SHA256.items():
        assert _sha256(_ROOT / relative) == _POST_MERGE_SOURCE_SHA256.get(
            relative, expected
        )
        artifact_expected = {
            "hipengine/runtime/laguna_gguf_runner.py": (
                _RUNTIME_ARTIFACT_RUNNER_SHA256
            ),
            "tests/test_laguna_h8b_scoped_activation_pack_reuse.py": (
                _RUNTIME_ARTIFACT_H8B_TEST_SHA256
            ),
        }.get(relative, expected)
        assert artifact["source_sha256"].get(relative, artifact_expected) == (
            artifact_expected
        )


def test_h8b_source_default_selects_complete_owner_and_preserves_rollback() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.runtime.laguna_gguf_runner import (
        resolve_laguna_activation_pack_reuse,
    )

    live_source = getattr(hip_gfx1100, _SOURCE_CAPABILITY)

    # Source ownership is now the sole positive policy; explicit false keeps
    # complete H8A rollback and the former positive candidate seam is removed.
    assert live_source is True
    assert not hasattr(hip_gfx1100, _REMOVED_SUPPORTED_CAPABILITY)
    assert not hasattr(hip_gfx1151, _SOURCE_CAPABILITY)
    assert not hasattr(hip_gfx1151, _REMOVED_SUPPORTED_CAPABILITY)
    assert resolve_laguna_activation_pack_reuse("hip_gfx1100", None) is True
    assert resolve_laguna_activation_pack_reuse("hip_gfx1100", False) is False
    assert resolve_laguna_activation_pack_reuse("hip_gfx1151", None) is False
    assert resolve_laguna_activation_pack_reuse("hip_gfx1151", False) is False
    with pytest.raises(ValueError, match="positive selector was removed"):
        resolve_laguna_activation_pack_reuse("hip_gfx1100", True)
    with pytest.raises(ValueError, match="positive selector was removed"):
        resolve_laguna_activation_pack_reuse("hip_gfx1151", True)
    assert _normalized_package_sha256() == _NORMALIZED_PACKAGE_SHA256
