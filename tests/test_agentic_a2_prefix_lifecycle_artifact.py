from __future__ import annotations

import hashlib
import json
from pathlib import Path


ARTIFACT = Path(
    "benchmarks/results/2026-07-21-w7900-agentic-a2-prefix-lifecycle-closure.json"
)
C1_ARTIFACT = Path(
    "benchmarks/results/2026-07-21-w7900-agentic-a2-c1-prefix-rejected.json"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_agentic_a2_prefix_lifecycle_closes_every_fail_closed_boundary() -> None:
    payload = _load(ARTIFACT)

    assert payload["kind"] == "gfx1100_agentic_a2_prefix_lifecycle_closure"
    assert payload["status"] == "passed_correctness_route_rejected_performance"
    assert payload["passed"] is True
    assert payload["correctness_claim"] is True
    assert payload["performance_claim"] is False
    assert payload["timing_claim"] is False

    gates = payload["real_gpu_active_and_completed_gates"]
    assert len(gates) == 4
    assert {(row["prefix_tokens"], row["source_lifecycle"]) for row in gates} == {
        (2048, "active"),
        (2048, "completed"),
        (8192, "active"),
        (8192, "completed"),
    }
    for row in gates:
        source_path = Path(row["artifact"])
        source = _load(source_path)
        assert row["sha256"] == _sha256(source_path)
        assert source["passed"] is True
        assert source["target_arch"] == "gfx1100"
        assert source["workload"]["prefix_tokens"] == row["prefix_tokens"]
        assert source["workload"]["source_lifecycle"] == row["source_lifecycle"]
        assert source["production_route"]["sampler_route_exact"] is True
        assert source["production_route"]["metadata_exact"] is True
        assert source["prefill_oracle"]["output_exact"] is True
        assert source["prefill_oracle"]["initial_state_exact"] is True
        assert source["prefill_oracle"]["source_immutable"] is True
        assert source["teacher_forced"]["trajectory_exact"] is True
        assert source["teacher_forced"]["final_state_exact"] is True
        assert source["teacher_forced"]["candidate_response_token_ids"] == row[
            "candidate_response_token_ids"
        ]
        assert row["response_ids_exact"] is True
        assert row["initial_state_exact"] is True
        assert row["final_state_exact"] is True
        assert row["kl_max"] == 0.0
        assert row["top1_agreement"] == 1.0
        assert row["lifecycle"]["exact"] is True
        assert row["lifecycle"]["final_refcounted_pages"] == 0
        assert row["cow_fork_events"] == 0
        assert row["pinned_pages"] == 0
        assert row["pool_current_bytes"] == row["pool_high_water_bytes"]
        assert row["cache_resident_bytes_before_eviction"] <= row[
            "cache_resident_limit_bytes"
        ]
        assert row["saved_live_bytes"] == row["reused_pages"] * 5_242_880
        assert row["reused_tokens"] == row["prefix_tokens"]
        if row["source_lifecycle"] == "active":
            assert row["lifecycle"]["source_refcount_before_release"] == 2
            assert row["lifecycle"]["shared_refcount_after_continuation_release"] == 0
            assert row["cache_resident_bytes_before_eviction"] == 0
        else:
            assert row["lifecycle"]["source_refcount_before_release"] == 1
            assert row["lifecycle"]["shared_refcount_after_continuation_release"] == 1
            assert row["lifecycle"]["snapshot_evicted"] is True
            assert row["cache_resident_bytes_before_eviction"] > 0

    economics = payload["real_agentic_cache_economics"]
    assert economics["artifact"] == str(C1_ARTIFACT)
    assert economics["sha256"] == _sha256(C1_ARTIFACT)
    assert economics["all_target_gpu0_exclusive"] is True
    assert economics["all_final_ownership_bounded"] is True
    assert economics["performance_route_accepted"] is False
    expected_economics = {
        "small_repo": (124_518_400, 11, 234_618_880, 0.0),
        "growing_history": (129_761_280, 12, 234_618_880, 42_240.0),
        "medium_repo": (255_590_400, 36, 486_277_120, 28_160.0),
    }
    for family, (resident_bytes, pages, limit_bytes, bytes_per_token) in (
        expected_economics.items()
    ):
        row = economics["families"][family]
        assert row["max_final_cache_resident_bytes"] == resident_bytes
        assert row["max_final_cache_resident_pages"] == pages
        assert row["max_allowed_cache_bytes"] == limit_bytes
        assert resident_bytes <= limit_bytes
        assert row["max_cache_bytes_per_reused_token"] == bytes_per_token
        assert row["all_non_cache_final_owners_zero"] is True

    lifecycle = payload["lifecycle_matrix"]
    assert all(section["covered"] is True for section in lifecycle.values())
    assert lifecycle["hit_miss_source_boundaries"]["sources"] == [
        "active_current",
        "completed_snapshot",
    ]
    assert set(lifecycle["hit_miss_source_boundaries"]["fallbacks"]) == {
        "miss",
        "full_prompt_boundary_requires_suffix",
        "sampling_unsupported",
    }
    assert lifecycle["bounded_lru_residency_and_eviction"][
        "explicit_eviction_final_refcounted_pages"
    ] == 0
    assert lifecycle["cow_refcount_pin"]["host_cow_fork_and_pin_unpin_gate"] is True
    assert lifecycle["cancellation_and_admission_rollback"][
        "prefix_runner_rollback_releases_destination_and_preserves_source"
    ] is True
    assert lifecycle["disconnect_and_slow_consumer"][
        "slow_stream_queue_bounded_without_blocking_neighbor"
    ] is True
    assert lifecycle["deadline"]["timeout_returns_deadline_and_server_reuses"] is True
    assert lifecycle["fork_and_rollback"]["resident_state_reuse_supported"] is False

    validation = payload["mechanical_validation"]
    assert validation["command"] == "uv run pytest -q " + " ".join(
        validation["test_nodes"]
    )
    assert validation["passed"] == 50
    assert validation["failed"] == 0

    assert payload["acceptance"] == {
        "exact_survivor_ids_and_state_kv": True,
        "bounded_declared_cache_bytes": True,
        "zero_non_cache_final_owners": True,
        "active_and_completed_sources_exact": True,
        "eviction_and_refcount_drain_exact": True,
        "cancellation_disconnect_slow_deadline_fail_closed": True,
        "fork_rollback_explicitly_rejected_for_resident_state": True,
        "performance_promotion_allowed": False,
    }
