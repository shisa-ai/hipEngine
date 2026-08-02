"""WPF-H7Y lane-major SWA cache source-default RED contract."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_ARTIFACT = _ROOT / (
    "benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-"
    "swa-lane-major-cache-runtime-candidate.json"
)
_RUNTIME_ARTIFACT_SHA256 = (
    "359a1c1fc85037a2b0b4147ffc53b9bf65464998dd5d567f2b143de72eceddbd"
)
_PACKAGE = _ROOT / "hipengine/kernels/hip_gfx1100/__init__.py"
_SOURCE_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS"
_H7Y_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H7Y_ROLE_VARIANTS"
_H6Z_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6Z_ROLE_VARIANTS"
_H6W_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6W_ROLE_VARIANTS"
_H6A_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6A_ROLE_VARIANTS"
_GLOBAL_ROLE = "global_m128_c4096_first_fill_exact"
_SWA_ROLE = "swa_qrow4_m128_c512_no_wrap_exact"
_H6N_GLOBAL = "global_context_rows_dense_initial_fixed512_cached_exact_spans"
_H6Z_GLOBAL = (
    "global_context_rows_qrow4_dense_initial_global_score_weight_replay_"
    "exact_spans"
)
_H6A_SWA = "swa_context_rows_qrow4_dense_initial_cached_exact_spans"
_H6W_SWA = (
    "swa_context_rows_qrow4_dense_initial_global_score_replay_exact_spans"
)
_H7Y_SWA = (
    "swa_context_rows_qrow4_dense_initial_lane_major_"
    "global_score_replay_exact_spans"
)
_H6A_POLICY = {_GLOBAL_ROLE: _H6N_GLOBAL, _SWA_ROLE: _H6A_SWA}
_H6W_POLICY = {_GLOBAL_ROLE: _H6N_GLOBAL, _SWA_ROLE: _H6W_SWA}
_H6Z_POLICY = {_GLOBAL_ROLE: _H6Z_GLOBAL, _SWA_ROLE: _H6W_SWA}
_H7Y_POLICY = {_GLOBAL_ROLE: _H6Z_GLOBAL, _SWA_ROLE: _H7Y_SWA}
_FUSED_WRITER = "swa_f32_rows_natural_lane_major_spans"
_NATURAL_SWA_WRITER = "swa_f32_rows_spans"
_GLOBAL_WRITER = "global_f32_rows_spans"
_MIRROR_BYTES = 75_497_472
_MIRROR_ALLOCATIONS = 72
_SOURCE_TOPOLOGY = {
    "H6N": 24,
    "H6Z": 24,
    "H6A": 72,
    "H7Y": 72,
    "fused_writer": 144,
    "natural_swa_writer": 0,
    "global_writer": 48,
}
_H6Z_ROLLBACK_TOPOLOGY = {
    "H6N": 24,
    "H6Z": 24,
    "H6A": 72,
    "H6W": 72,
    "fused_writer": 0,
    "natural_swa_writer": 144,
    "global_writer": 48,
}
_SOURCE_ADMISSION = {
    "complete_state_exact": True,
    "fixed_c4096_m512_samples": 5,
    "lengths": (512, 1_024, 4_096),
    "samples_per_length": 3,
    "every_source_median_must_win": True,
    "named_request_dispatches": 2_286,
    "post_commit_wall_samples": 5,
    "post_commit_profiled_requests": 5,
    "natural_h6z_h6w_h6a_rollback_required": True,
    "candidate_capability_cleanup_after_checkpoint": True,
    "no_subset_or_favorable_rerun": True,
}
_SOURCE_SHA256 = {
    "hipengine/kernels/hip_gfx1100/attention/laguna_kv.py": (
        "77398def2dca3430a544e390d8f8f588adcfb69b82940536b3b13f3306c773a8"
    ),
    "hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.hip": (
        "e5ac3c0825de4ccdc1f22eea25a9bc2e541bb76402bf73c348c643446a43fe83"
    ),
    "hipengine/kernels/hip_gfx1151/__init__.py": (
        "54e9b72005e919ce5f738b4ad4c8b81604138378e0a4fff65fdd172bf2800758"
    ),
    "hipengine/runtime/laguna_kv.py": (
        "97d168047053f0c57ef1e00b776ccc943f22febeee5f103f2ab7f2cf97ca337c"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_package_sha256() -> str:
    source = _PACKAGE.read_text()
    h6z = (
        "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS = dict(\n"
        "    LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6Z_ROLE_VARIANTS\n"
        ")"
    )
    h7y = (
        "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS = dict(\n"
        "    LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H7Y_ROLE_VARIANTS\n"
        ")"
    )
    assert source.count(h6z) + source.count(h7y) == 1
    normalized = source.replace(h6z, "<H7Y_SOURCE_POLICY>").replace(
        h7y, "<H7Y_SOURCE_POLICY>"
    )
    return hashlib.sha256(normalized.encode()).hexdigest()


def _launch_counts(calls: list[tuple[str, tuple, dict]]) -> dict[str, int]:
    variants = Counter(
        call[0].removeprefix("launch:")
        for call in calls
        if call[0].startswith("launch:")
    )
    return {
        "H6N": variants[_H6N_GLOBAL],
        "H6Z": variants[_H6Z_GLOBAL],
        "H6A": variants[_H6A_SWA],
        "H6W": variants[_H6W_SWA],
        "H7Y": variants[_H7Y_SWA],
        "fused_writer": variants[_FUSED_WRITER],
        "natural_swa_writer": variants[_NATURAL_SWA_WRITER],
        "global_writer": variants[_GLOBAL_WRITER],
    }


def _run_policy(
    monkeypatch: pytest.MonkeyPatch,
    policy: dict[str, str],
) -> tuple[dict[str, int], int, int, int]:
    from hipengine.kernels import hip_gfx1100
    from hipengine.runtime import laguna_kv as module
    from tests.test_laguna_h7y_runtime_policy import (
        _FakeRuntime,
        _dispatch,
        _install_fake_dispatch,
        _production_config,
    )

    monkeypatch.setattr(hip_gfx1100, _SOURCE_CAPABILITY, dict(policy))
    runtime = _FakeRuntime()
    cache = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4_096,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    calls: list[tuple[str, tuple, dict]] = []
    _install_fake_dispatch(cache, calls)
    try:
        cache.bind_prefill_score_scratch(0x40000000, 150_994_944)
        for layer_id in range(48):
            for start in (0, 128, 256, 384):
                _dispatch(cache, layer_id, start)
        return (
            _launch_counts(calls),
            cache.lane_major_mirror_nbytes,
            cache.lane_major_mirror_allocation_count,
            cache.allocation_count,
        )
    finally:
        cache.free()
        assert runtime.allocations == {}


def test_h7y_source_red_pins_qualified_owner_and_promotion_contract() -> None:
    from hipengine.kernels import hip_gfx1100

    artifact_bytes = _RUNTIME_ARTIFACT.read_bytes()
    assert hashlib.sha256(artifact_bytes).hexdigest() == _RUNTIME_ARTIFACT_SHA256
    artifact = json.loads(artifact_bytes)
    assert artifact["status"] == "qualified_bounded_default_off"
    assert artifact["decision"]["candidate_admitted"] is True
    assert artifact["decision"]["runtime_owner_retained"] is True
    assert artifact["decision"]["source_default_changed"] is False
    assert artifact["complete_state"]["kl_max"] == 0.0
    assert artifact["complete_state"]["hidden_boundaries_exact"] == 48
    assert artifact["writer_inclusive_fixed_c4096_m512"]["gain_percent"] > 0
    assert all(
        artifact["clean_length_gate"][length]["gain_percent"] > 0
        for length in ("512", "1024", "4096")
    )
    assert artifact["named_trace"]["application_dispatches"] == 2_286
    assert artifact["named_trace"]["counts"] == {
        "H6A": 72,
        "H6N": 24,
        "H6W": 0,
        "H6Z": 24,
        "H7Y": 72,
        "global_natural_writer": 48,
        "swa_natural_lane_major_writer": 144,
        "swa_natural_writer": 0,
    }
    assert artifact["candidate"]["mirror_bytes"] == _MIRROR_BYTES
    assert artifact["candidate"]["mirror_allocations"] == _MIRROR_ALLOCATIONS
    assert _SOURCE_ADMISSION == {
        "complete_state_exact": True,
        "fixed_c4096_m512_samples": 5,
        "lengths": (512, 1_024, 4_096),
        "samples_per_length": 3,
        "every_source_median_must_win": True,
        "named_request_dispatches": 2_286,
        "post_commit_wall_samples": 5,
        "post_commit_profiled_requests": 5,
        "natural_h6z_h6w_h6a_rollback_required": True,
        "candidate_capability_cleanup_after_checkpoint": True,
        "no_subset_or_favorable_rerun": True,
    }
    assert getattr(hip_gfx1100, _H7Y_CAPABILITY) == _H7Y_POLICY
    assert getattr(hip_gfx1100, _H6Z_CAPABILITY) == _H6Z_POLICY
    assert getattr(hip_gfx1100, _H6W_CAPABILITY) == _H6W_POLICY
    assert getattr(hip_gfx1100, _H6A_CAPABILITY) == _H6A_POLICY
    assert getattr(hip_gfx1100, _SOURCE_CAPABILITY) in (_H6Z_POLICY, _H7Y_POLICY)
    assert _normalized_package_sha256() == (
        "53bae0a5cb936bf487330ac08dc1a4e8d4de4e68566e752a242449c2d240f7fb"
    )
    for relative, expected in _SOURCE_SHA256.items():
        assert _sha256(_ROOT / relative) == expected


def test_h7y_source_default_selects_h7y_and_preserves_natural_rollbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151

    live_source = getattr(hip_gfx1100, _SOURCE_CAPABILITY)

    # Intentional RED after the qualified artifact, immutable implementation,
    # named capability, and package-normalization controls: source still H6Z.
    assert live_source == _H7Y_POLICY
    assert getattr(hip_gfx1100, _H7Y_CAPABILITY) == _H7Y_POLICY
    assert getattr(hip_gfx1100, _H6Z_CAPABILITY) == _H6Z_POLICY
    assert getattr(hip_gfx1100, _H6W_CAPABILITY) == _H6W_POLICY
    assert getattr(hip_gfx1100, _H6A_CAPABILITY) == _H6A_POLICY
    assert not hasattr(hip_gfx1151, _H7Y_CAPABILITY)

    selected, mirror_bytes, mirror_allocations, selected_allocations = _run_policy(
        monkeypatch,
        _H7Y_POLICY,
    )
    assert {key: selected[key] for key in _SOURCE_TOPOLOGY} == _SOURCE_TOPOLOGY
    assert selected["H6W"] == 0
    assert mirror_bytes == _MIRROR_BYTES
    assert mirror_allocations == _MIRROR_ALLOCATIONS

    rollback, rollback_bytes, rollback_allocations, natural_allocations = _run_policy(
        monkeypatch,
        _H6Z_POLICY,
    )
    assert {
        key: rollback[key] for key in _H6Z_ROLLBACK_TOPOLOGY
    } == _H6Z_ROLLBACK_TOPOLOGY
    assert rollback["H7Y"] == 0
    assert rollback_bytes == 0
    assert rollback_allocations == 0
    assert selected_allocations - natural_allocations == _MIRROR_ALLOCATIONS

    complete, complete_bytes, complete_allocations, _ = _run_policy(
        monkeypatch,
        _H6A_POLICY,
    )
    assert complete["H6N"] == 48
    assert complete["H6A"] == 144
    assert complete["H6Z"] == complete["H6W"] == complete["H7Y"] == 0
    assert complete["natural_swa_writer"] == 144
    assert complete["fused_writer"] == 0
    assert complete_bytes == 0
    assert complete_allocations == 0
