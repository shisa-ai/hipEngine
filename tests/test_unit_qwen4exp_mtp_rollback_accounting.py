from __future__ import annotations

from scripts.qwen4exp_mtp_rollback_accounting import run_accounting_screen


def test_qwen4exp_mtp_accounting_covers_rejection_depths_and_full_accept() -> None:
    artifact = run_accounting_screen(budget=4)

    assert artifact["passed"] is True
    assert artifact["candidate_admitted"] is False
    assert artifact["production_defaults_changed"] is False
    rows = artifact["screen"]["rows"]
    assert [row["rejection_depth"] for row in rows] == [0, 1, 2, 3, None]
    assert [row["target_forwards"] for row in rows] == [1, 2, 3, 4, 4]
    assert all(row["draft_forwards"] == 4 for row in rows)
    assert all(row["restored_bytes"] == 0 for row in rows)
    assert all(row["restored_allocations"] == 0 for row in rows)
    assert all(row["per_cycle_allocations"] == 0 for row in rows)
    assert all(row["shared_weight_allocations"] == 1 for row in rows)
    assert all(row["snapshot_calls"] == 0 for row in rows)
    assert all(row["restore_calls"] == 0 for row in rows)
    assert all(row["target_position"] == row["draft_position"] for row in rows)
    assert artifact["invariants"]["snapshot_restore_replay_forwards"] is False
    assert artifact["invariants"]["shared_weight_double_count"] is False
