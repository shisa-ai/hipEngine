from __future__ import annotations

import hashlib
import json
from pathlib import Path


ARTIFACT = Path(
    "benchmarks/results/2026-07-22-w7900-agentic-a5-pressure-soak-closure.json"
)
CACHE_LIFECYCLE = Path(
    "benchmarks/results/2026-07-21-w7900-agentic-a2-prefix-lifecycle-closure.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_agentic_a5_pressure_soak_closes_the_frozen_packet() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["kind"] == "gfx1100_agentic_a5_pressure_soak"
    assert payload["status"] == "passed_pressure_soak_no_performance_claim"
    assert payload["passed"] is True
    assert payload["measurement_valid"] is True
    assert payload["correctness_claim"] is True
    assert payload["timing_claim"] is True
    assert payload["performance_claim"] is False
    assert payload["source"] == {
        "clean": True,
        "commit": "414d6d9e0fc8a1333bbece4db851271f031936bf",
        "dirty": False,
        "pushed": True,
    }

    provenance = payload["hipengine_artifact_provenance"]
    assert provenance["kind"] == "hipengine_artifact_provenance"
    assert provenance["schema_version"] == 1
    assert provenance["hipengine_commit"] == payload["source"]["commit"]
    assert provenance["configured_backend"] == "hip_gfx1100"
    assert provenance["resolved_backend"] == "hip_gfx1100"
    assert provenance["target_arch"] == "gfx1100"
    assert provenance["device_name"] == "AMD Radeon Pro W7900"
    assert provenance["dirty"] is False
    assert provenance["staged_dirty"] is False
    assert provenance["unstaged_dirty"] is False
    assert provenance["untracked_dirty"] is False
    assert provenance["environment"]["HIP_VISIBLE_DEVICES"] == "0"
    assert provenance["environment"]["ROCR_VISIBLE_DEVICES"] == "0"

    protocol = payload["protocol"]
    assert protocol["configuration"]["prefix_cache"] == "off"
    assert protocol["configuration"]["selected_generation_batch_window_ms"] == 0.0
    assert protocol["configuration"]["selected_fair_prefill_burst_chunks"] == 1
    assert protocol["soak_seconds"] == 80.0
    assert protocol["soak_rate_per_second"] == 0.5
    assert protocol["tuning_skipped"] is True
    assert protocol["complete_workload_set"] == [
        "static_c1",
        "static_c8",
        "ragged_burst",
        "continuous_fixed",
        "continuous_poisson",
        "cancellation_disconnect",
        "overload",
        "idle_recovery",
        "soak",
    ]

    aggregate = payload["aggregate"]
    assert aggregate == {
        "completed": 108,
        "completed_generated_tokens": 2480,
        "disconnected": 1,
        "disconnected_generated_tokens": 2,
        "exact_generated_tokens": 2482,
        "minimum_completed_required": 100,
        "outcome_accounting_passed": True,
        "rejected": 12,
        "requests": 122,
        "timed_out": 1,
    }
    assert set(payload["workloads"]) == set(protocol["complete_workload_set"])
    for workload in payload["workloads"].values():
        assert workload["passed"] is True
        assert workload["correctness"]["passed"] is True
        assert workload["metrics_accounting_passed"] is True
        assert all(workload["slo_checks"].values())

    cancel = payload["workloads"]["cancellation_disconnect"]
    assert cancel["outcomes"] == {
        "completed": 6,
        "disconnected": 1,
        "timeout": 1,
    }
    assert cancel["latency_seconds"]["cancellation_ack"]["count"] == 1
    assert cancel["latency_seconds"]["cancellation_ack"]["max"] < 0.05
    assert len(cancel["survivor_generated_ids"]) == 6
    for row in cancel["control_rows"]:
        assert row["exact"] is True
        assert row["prompt_exact"] is True
        assert row["http_protocol_exact"] is True
        assert row["generated_ids"] == row["expected_ids"]
    disconnected = next(
        row for row in cancel["control_rows"] if row["action"] == "disconnect"
    )
    assert disconnected["outcome"] == "disconnected"
    assert disconnected["disconnect_triggered"] is True
    assert disconnected["generated_ids"] == [9709, 9709]
    timed_out = next(row for row in cancel["control_rows"] if row["action"] == "timeout")
    assert timed_out["outcome"] == "timeout"
    assert timed_out["error_code"] == "deadline_exceeded"
    assert timed_out["generated_ids"] == []
    slow = cancel["slow_consumer"]
    assert slow["label"] == "cancel-0001"
    assert slow["read_delay_seconds"] == 0.05
    assert slow["outcome"] == "completed"
    assert slow["exact"] is True
    assert slow["generated_ids"] == slow["expected_ids"]

    overload = payload["workloads"]["overload"]
    assert overload["outcomes"] == {"completed": 20, "rejected": 12}
    assert overload["overload"] == {
        "accepted": 20,
        "engine_busy_rejected": 12,
        "passed": True,
        "rejected": 12,
        "required": True,
    }
    assert len(overload["completed_request_ids"]) == 20
    assert len(overload["rejected_labels"]) == 12

    pressure = payload["resource_pressure"]
    assert pressure["queue_and_stream"] == {
        "configured_generation_active_cap": 8,
        "configured_generation_queue_cap": 16,
        "configured_stream_queue_max_chunks": 16,
        "observed_generation_queue_depth_max": 16,
        "observed_resident_active_rows_max": 4,
        "observed_resident_pending_max": 3,
        "observed_stream_queue_depth_max": 1,
        "overload_completed": 20,
        "overload_rejected": 12,
    }
    assert pressure["kv_pool"]["initial_pages"] == 3
    assert pressure["kv_pool"]["high_water_pages"] == 12
    assert pressure["kv_pool"]["grow_events"] == 15
    assert pressure["kv_pool"]["shrink_events"] == 15
    assert pressure["kv_pool"]["grow_failures"] == 0
    assert pressure["kv_pool"]["final_refcounted_pages"] == 0
    assert pressure["kv_pool"]["final_pinned_pages"] == 0
    assert pressure["graph"]["captures"] == pressure["graph"]["invalidations"] == 28
    assert pressure["graph"]["final_entries"] == 0
    assert pressure["workspace"]["release_events"] == 42
    assert pressure["workspace"]["released_bytes"] == 7_245_205_456
    assert pressure["workspace"]["final_current_bytes"] == 0
    assert pressure["memory"]["tracked_delta_bytes"] < 0

    cache = payload["cache_pressure_evidence"]
    assert cache["current_packet_mode"] == "off"
    assert cache["current_packet_final_resident_bytes"] == 0
    assert cache["separate_exact_lifecycle_artifact"] == str(CACHE_LIFECYCLE)
    assert cache["separate_exact_lifecycle_sha256"] == _sha256(CACHE_LIFECYCLE)
    assert cache["p2048_completed_resident_bytes_before_eviction"] == 108_789_760
    assert cache["p8192_completed_resident_bytes_before_eviction"] == 234_618_880
    assert cache["explicit_eviction_final_refcounted_pages"] == 0

    exclusivity = payload["gpu_exclusivity"]
    assert exclusivity["passed"] is True
    assert exclusivity["monitor_samples"] == 41
    assert exclusivity["other_kfd_processes"] == []
    assert exclusivity["peer_card_use_percent_max"] == 0
    assert exclusivity["target_process_pid"] == 1_551_464

    assert all(payload["gates"].values())
    ownership = payload["final_ownership"]
    assert ownership["active_requests"] == 0
    assert ownership["pending_requests"] == 0
    assert ownership["model_active_requests"] == 0
    assert ownership["model_active_request_ids"] == []
    assert ownership["generation_queue_depth"] == 0
    assert ownership["generation_active_requests"] == 0
    assert ownership["generation_batcher_active"] is False
    assert ownership["kv_refcounted_pages"] == 0
    assert ownership["kv_pinned_pages"] == 0
    assert ownership["graph_entries"] == 0
    assert ownership["workspace_current_bytes"] == 0
