from __future__ import annotations

import pytest

from hipengine.dispatch.batch import WorkItem, WorkKind
from hipengine.dispatch.d2_resolver import (
    CostTable,
    PhysicalWidthCost,
    ceiling_partition,
    cost_table_from_artifact,
    d2_partition,
    plan_d2_groups,
)

# Measured post-promotion Qwen3.8-27B Q4_K_M W7900 inter-token model-step
# median (ms) per direct physical width (Q5/Q6 rowtile-through-8 packet). These
# are the fixture standing in for a loaded benchmark artifact; the resolver
# itself takes a CostTable and never hard-codes constants.
POST_PROMOTION_STEP_MS = {
    1: 33.1701,
    2: 37.5209,
    3: 40.0602,
    4: 43.2973,
    5: 48.0149,
    6: 52.7025,
    7: 57.8864,
    8: 63.5257,
}


def _cost_table() -> CostTable:
    return CostTable(
        tuple(
            PhysicalWidthCost(width, ms, "post-promotion-fixture")
            for width, ms in POST_PROMOTION_STEP_MS.items()
        )
    )


def test_cost_table_validates_records() -> None:
    with pytest.raises(ValueError, match="c1 route"):
        CostTable((PhysicalWidthCost(2, 37.5, "x"),))
    with pytest.raises(ValueError, match="unique and strictly increasing"):
        CostTable(
            (
                PhysicalWidthCost(1, 33.0, "x"),
                PhysicalWidthCost(2, 37.0, "x"),
                PhysicalWidthCost(2, 38.0, "x"),
            )
        )
    with pytest.raises(ValueError, match="positive"):
        PhysicalWidthCost(1, 0.0, "x")
    with pytest.raises(ValueError, match="empty"):
        CostTable(())
    with pytest.raises(KeyError):
        _cost_table().cost_ms(9)


def test_ceiling_partition_is_largest_width_chunking() -> None:
    widths = (1, 2, 3, 4, 5, 6, 7, 8)
    assert ceiling_partition(9, widths) == (8, 1)
    assert ceiling_partition(13, widths) == (8, 5)
    assert ceiling_partition(16, widths) == (8, 8)
    assert ceiling_partition(5, widths) == (5,)


def test_d2_partition_recovers_expected_choices() -> None:
    ct = _cost_table()
    expected = {
        9: (5, 4),
        10: (6, 4),
        11: (6, 5),
        12: (6, 6),
        13: (7, 6),
        14: (7, 7),
        15: (8, 7),
        16: (8, 8),
    }
    for rows, want in expected.items():
        got = d2_partition(rows, ct)
        assert tuple(sorted(got, reverse=True)) == want, f"c{rows}: {got}"

    # Single-group / within-width ranges stay exact.
    for rows in range(1, 9):
        assert d2_partition(rows, ct) == (rows,)


def test_d2_partition_beats_ceiling_for_c9_and_c10() -> None:
    ct = _cost_table()
    widths = ct.widths
    for rows in (9, 10, 13, 14):
        d2 = d2_partition(rows, ct)
        ceil = ceiling_partition(rows, widths)
        d2_cost = sum(ct.cost_ms(w) for w in d2)
        ceil_cost = sum(ct.cost_ms(w) for w in ceil)
        assert d2_cost <= ceil_cost, f"c{rows}: d2 {d2} vs ceiling {ceil}"
    # c9 specifically: 5+4 is cheaper than ceiling 8+1.
    assert sum(ct.cost_ms(w) for w in (5, 4)) < sum(ct.cost_ms(w) for w in (8, 1))


def test_d2_partition_breaks_ties_toward_fewer_groups() -> None:
    # Two equal-cost singleton splits never outrank one direct width.
    flat = CostTable(
        (
            PhysicalWidthCost(1, 10.0, "x"),
            PhysicalWidthCost(2, 20.0, "x"),
            PhysicalWidthCost(4, 40.0, "x"),
        )
    )
    assert d2_partition(4, flat) == (4,)


def test_d2_partition_rejects_non_positive_rows() -> None:
    with pytest.raises(ValueError, match="positive"):
        d2_partition(0, _cost_table())
    with pytest.raises(ValueError, match="positive"):
        ceiling_partition(0, (1, 2))


def test_plan_d2_groups_builds_dense_composition() -> None:
    work = WorkItem(
        kind=WorkKind.DECODE,
        request_ids=(100, 101, 102, 103, 104, 105, 106, 107, 108),
        row_to_request=(100, 101, 102, 103, 104, 105, 106, 107, 108),
        slot_ids=(0, 1, 2, 3, 4, 5, 6, 7, 8),
    )
    groups = plan_d2_groups(work, _cost_table())  # c9 -> 5 + 4
    assert [g.physical_rows for g in groups] == [5, 4]
    assert [tuple(g.request_ids) for g in groups] == [
        (100, 101, 102, 103, 104),
        (105, 106, 107, 108),
    ]
    flattened = tuple(
        request_id
        for group in groups
        for request_id in group.request_ids
    )
    assert flattened == work.request_ids
    for group in groups:
        assert group.dense_execution_rows
        assert group.active_mask == tuple(True for _ in range(group.physical_rows))


def test_cost_table_from_artifact_loads_measured_medians(tmp_path) -> None:
    import json

    # Canonical benchmark artifact shape: summaries.<label>.latency.
    # inter_token_model_step_seconds.median in seconds.
    artifact = {
        "summaries": {
            label: {
                "latency": {
                    "inter_token_model_step_seconds": {
                        "median": seconds,
                        "samples": [seconds],
                    }
                }
            }
            for label, seconds in [
                ("c1", 0.0331),
                ("c2", 0.0375),
                ("c3", 0.0400),
                ("c4", 0.0433),
                ("c5", 0.0480),
                ("c6", 0.0527),
                ("c7", 0.0578),
                ("c8", 0.0635),
            ]
        }
    }
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    label_by_width = {w: f"c{w}" for w in range(1, 9)}
    ct = cost_table_from_artifact(path, label_by_width=label_by_width)
    assert ct.widths == (1, 2, 3, 4, 5, 6, 7, 8)
    assert abs(ct.cost_ms(4) - 43.3) < 0.1
    # The loaded table recovers the expected D2 choices.
    assert d2_partition(13, ct) == (7, 6)
    assert d2_partition(16, ct) == (8, 8)

    with pytest.raises(KeyError):
        cost_table_from_artifact(path, label_by_width={1: "c1", 9: "c9"})


def test_cost_table_from_artifact_rejects_missing_median(tmp_path) -> None:
    import json

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"summaries": {"c1": {}}}), encoding="utf-8")
    with pytest.raises(KeyError, match="no model-step median"):
        cost_table_from_artifact(path, label_by_width={1: "c1"})


def test_plan_d2_groups_preserves_slot_identity_across_holes() -> None:
    # Non-contiguous but active slots are grouped in slot order by D2 widths.
    work = WorkItem(
        kind=WorkKind.DECODE,
        request_ids=(0, 1, 2, 3, 4),
        row_to_request=(0, 1, 2, 3, 4),
        slot_ids=(0, 1, 2, 4, 6),
        active_mask=(True, True, True, False, True, False, True),
    )
    groups = plan_d2_groups(work, _cost_table())  # c5 -> single group of 5
    assert len(groups) == 1
    assert groups[0].physical_rows == 5
    assert groups[0].global_slot_indices == (0, 1, 2, 4, 6)
    assert groups[0].request_ids == (0, 1, 2, 3, 4)
