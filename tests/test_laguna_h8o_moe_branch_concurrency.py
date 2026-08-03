"""RED contracts for WPF-H8O exact gfx1100 MoE branch concurrency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hipengine.kernels import hip_gfx1100, hip_gfx1151
from hipengine.kernels.backends import backend_package_capability

_ROOT = Path(__file__).resolve().parents[1]
_TARGET = (
    _ROOT
    / "benchmarks/results/"
    "2026-08-03-gfx1100-laguna-q2-xl-post-h8n-"
    "moe-shared-after-router-low-priority-target.json"
)
_TARGET_SHA256 = "975a67d6bdef02b4fddf7b15265e5a08029a3d4a722ec91ff80cfe016bbddd23"
_CANDIDATE_CAPABILITY = "LAGUNA_MOE_BRANCH_CONCURRENCY_H8O_CANDIDATE"
_FIXED_SCHEDULE = {
    "supported": True,
    "enabled_by_default": False,
    "gpu_max_hw_queues": 2,
    "shared_after_router": True,
    "shared_low_priority": True,
    "shared_priority": 1,
    "caller_priority": "default",
    "event_flags": 0x2,
}
_SOURCE_SHA256 = {
    "hipengine/runtime/laguna_gguf_runner.py": (
        "edea1fc2df3c8ca46fe3396663ac14f9000b4ee0cc967ebafb55208afad50654"
    ),
    "hipengine/runtime/laguna_moe.py": (
        "0507c0ab9bcabddfda9d0390c66d46f80aaaf7c42357a58dfa24c692d43414fd"
    ),
    "hipengine/kernels/hip_gfx1151/__init__.py": (
        "a5838ffc8fd8df367cd828f397e701f94f2268c7992d0a5e143c8d7e2b8ba3b3"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_h8o_target_serial_window_and_admission_are_frozen() -> None:
    assert _sha256(_TARGET) == _TARGET_SHA256
    artifact = json.loads(_TARGET.read_text(encoding="utf-8"))
    assert artifact["status"] == "target_selected_no_candidate_no_speed_result"
    assert artifact["performance_claim"] is False
    assert artifact["operation_contract"]["id"] == "WPF-H8O"
    assert artifact["decision"] == {
        "candidate_implemented": False,
        "next_action": (
            "Commit this target-only packet, then freeze RED before explicitly "
            "executing the existing two-stream path on gfx1100."
        ),
        "performance_measured": False,
        "production_changed": False,
        "target_selected": True,
    }

    shared = artifact["source_unchanged_audit"]["serial_shared_branch"]
    assert shared["sparse_layers"] == 47
    assert shared["regular_segments"] == 46
    assert shared["special_segments"] == 1
    assert shared["dispatches"] == 421
    assert shared["kernel_sum_ms"] == 49.799391
    assert shared["serial_segment_span_ms"] == 51.119578
    assert shared["serial_inter_kernel_idle_ms"] == 1.320187
    assert sum(row["dispatches"] for row in shared["composition"].values()) == 421

    ceiling = artifact["source_unchanged_audit"][
        "zero_contention_ceiling_not_speed_result"
    ]
    assert ceiling["not_performance_claim"] is True
    assert ceiling["estimated_wall_tok_s"] == 461.19440599246406
    assert ceiling["gain_percent"] == 4.604699884823327

    fixed = artifact["admission_contract"]["fixed_c4096_m512"]
    assert fixed["warmups_per_arm"] == 1
    assert fixed["counter_rotated_repetitions"] == 7
    assert fixed["queue_matched_GPU_MAX_HW_QUEUES"] == 2
    assert fixed["required_candidate_pair_wins"] == 5
    assert fixed["required_candidate_median_wall_improvement"] is True
    trace = artifact["admission_contract"]["named_trace"]
    assert trace["candidate_application_dispatches"] == 2_155
    assert trace["candidate_shared_secondary_dispatches"] == 421
    assert trace["candidate_caller_dispatches"] == 1_734
    assert trace["candidate_queues"] == trace["candidate_streams"] == 2
    assert trace["same_kernel_multiset_as_control"] is True
    assert trace["positive_cross_queue_overlap_required"] is True
    assert artifact["admission_contract"]["clean_length_transfer_if_fixed_passes"][
        "lengths"
    ] == [512, 1024, 4096]


def test_h8o_gfx1100_candidate_capability_is_explicit_and_default_off() -> None:
    candidate = getattr(hip_gfx1100, _CANDIDATE_CAPABILITY)
    assert candidate == _FIXED_SCHEDULE
    assert candidate["supported"] is True
    assert candidate["enabled_by_default"] is False
    assert not backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_MOE_BRANCH_CONCURRENCY",
        False,
    )
    assert not backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_MOE_SHARED_AFTER_ROUTER",
        False,
    )
    assert not backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_MOE_SHARED_LOW_PRIORITY",
        False,
    )
    assert not hasattr(hip_gfx1151, _CANDIDATE_CAPABILITY)


def test_h8o_existing_event_protocol_and_byte_oracle_lineage_are_frozen() -> None:
    for relative, expected in _SOURCE_SHA256.items():
        assert _sha256(_ROOT / relative) == expected

    runner = (_ROOT / "hipengine/runtime/laguna_gguf_runner.py").read_text(
        encoding="utf-8"
    )
    moe = (_ROOT / "hipengine/runtime/laguna_moe.py").read_text(encoding="utf-8")
    gpu_oracle = (_ROOT / "tests/test_laguna_moe_gpu.py").read_text(encoding="utf-8")
    assert "priority=priority_range[0]" in runner
    concurrent_body = moe[
        moe.index("    def launch_concurrent_shared() -> None:") :
        moe.index("    if shared_concurrent and not shared_after_router:")
    ]
    ordered_needles = (
        "active_runtime.event_record(shared_input_ready_event, stream)",
        "active_runtime.stream_wait_event(\n            shared_stream,",
        "_launch_laguna_shared_rows(",
        "active_runtime.event_record(\n            shared_output_ready_event,",
    )
    offsets = [concurrent_body.index(needle) for needle in ordered_needles]
    assert offsets == sorted(offsets)
    assert (
        "if shared_concurrent and shared_after_router:\n"
        "        launch_concurrent_shared()"
    ) in moe
    assert (
        "active_runtime.stream_wait_event(\n"
        "            stream,\n"
        "            shared_output_ready_event,"
    ) in moe
    assert "np.testing.assert_array_equal(\n            _f32_to_bf16_u16(" in gpu_oracle
    assert "concurrent_actual" in gpu_oracle
    assert "after_router_actual" in gpu_oracle
