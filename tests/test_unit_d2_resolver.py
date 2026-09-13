from __future__ import annotations

import json
from pathlib import Path

import pytest

from hipengine.dispatch.batch import WorkItem, WorkKind
from hipengine.dispatch.d2_resolver import (
    D2_COST_ARTIFACT_KIND,
    CostTable,
    CostTableExpectation,
    PhysicalWidthCost,
    PrimitiveCostRecord,
    ceiling_partition,
    cost_table_from_artifact,
    d2_partition,
    plan_d2_groups,
)

POST_PROMOTION_STEP_MS = {
    1: 33.17009820602834,
    2: 37.52094367519021,
    3: 40.06023961119354,
    4: 43.29730453900993,
    5: 48.01489459350705,
    6: 52.70247510634363,
    7: 57.886386988684535,
    8: 63.525706296786666,
}
_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "2512f262273074db82860f1f3d6c15b4d9054b29b3c4babb0e2c770d6474c850"


def _record(width: int, cost_ms: float, *, workspace_bytes: int = 0) -> PhysicalWidthCost:
    return PhysicalWidthCost(
        active_rows=width,
        physical_width=width,
        mask_class="dense_all_active",
        model_step_ms=cost_ms,
        workspace_bytes=workspace_bytes,
        route_manifest_sha256=_SHA_A,
        correctness_sha256=_SHA_B,
        source="post-promotion-fixture",
    )


def _cost_table() -> CostTable:
    return CostTable(tuple(_record(width, ms) for width, ms in POST_PROMOTION_STEP_MS.items()))


def _expectation() -> CostTableExpectation:
    return CostTableExpectation(
        backend="hip_gfx1100",
        target_arch="gfx1100",
        host_name="epyc",
        device_name="AMD Radeon Pro W7900",
        model_fingerprint=_SHA_C,
        quant="gguf_q4_k_m",
        kv_dtype="bf16",
        execution_profile="strict",
        graph_mode="captured_replay",
        physical_widths=tuple(range(1, 9)),
    )


def _artifact() -> dict[str, object]:
    expectation = _expectation()
    return {
        "schema": 1,
        "kind": D2_COST_ARTIFACT_KIND,
        "status": "accepted",
        "passed": True,
        "measurement_valid": True,
        "performance_claim": False,
        "identity": expectation.to_json_dict(),
        "source_measurement": {
            "status": "measurement_complete",
            "passed": True,
            "complete_packet": True,
            "cross_configuration_correctness": {
                "passed": True,
                "all_direct_c1_c8_exact": True,
                "all_measured_runs_repeatable": True,
            },
            "sha256": _SHA_A,
            "summary_artifact": "benchmarks/results/source.json",
            "summary_artifact_sha256": _SHA_B,
            "provenance": {
                "dirty": False,
                "resolved_backend": "hip_gfx1100",
                "target_arch": "gfx1100",
                "host_name": "epyc",
                "device_name": "AMD Radeon Pro W7900",
                "model_fingerprint": {"value": _SHA_C},
                "quant": "gguf_q4_k_m",
                "kv_dtype": "bf16",
                "build_profile": "gfx1100_gguf_packed_graph_direct_c1_c8_controls",
            },
        },
        "correctness": {
            "quality_artifact": "benchmarks/results/quality.json",
            "quality_artifact_sha256": _SHA_A,
            "lifecycle_artifact": "benchmarks/results/lifecycle.json",
            "lifecycle_artifact_sha256": _SHA_B,
        },
        "primitive_records": [],
        "physical_group_records": [
            {
                "active_rows": width,
                "physical_width": width,
                "mask_class": "dense_all_active",
                "model_step_ms": cost,
                "workspace_bytes": 0,
                "workspace_scope": "preallocated_shared_union",
                "route_manifest_sha256": _SHA_A,
                "correctness_sha256": _SHA_B,
                "sample_count": 16,
            }
            for width, cost in POST_PROMOTION_STEP_MS.items()
        ],
    }


def test_cost_table_validates_records_and_bounds() -> None:
    with pytest.raises(ValueError, match="c1 route"):
        CostTable((_record(2, 37.5),))
    with pytest.raises(ValueError, match="unique and strictly increasing"):
        CostTable((_record(1, 33.0), _record(2, 37.0), _record(2, 38.0)))
    for invalid in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            _record(1, invalid)
    with pytest.raises(ValueError, match="default_max_width"):
        CostTable((_record(1, 10.0), _record(9, 20.0)), default_max_width=8)
    with pytest.raises(ValueError, match="same source"):
        CostTable(
            (
                _record(1, 10.0),
                PhysicalWidthCost(
                    active_rows=2,
                    physical_width=2,
                    mask_class="dense_all_active",
                    model_step_ms=20.0,
                    workspace_bytes=0,
                    route_manifest_sha256=_SHA_A,
                    correctness_sha256=_SHA_B,
                    source="other",
                ),
            )
        )


def test_primitive_cost_schema_validates_complete_axes() -> None:
    record = PrimitiveCostRecord(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q5_k_t16_v1",
        variant="t16_gemv_rowtile_col4_bf16_bf16_out",
        operation="linear",
        role="ssm_out",
        k=6144,
        n=5120,
        active_rows=5,
        physical_width=5,
        mask_class="dense_all_active",
        graph_mode="captured_replay",
        latency_ms=0.1,
        workspace_bytes=0,
        strict_fallback="t16_gemv_bf16_bf16_out",
        correctness_sha256=_SHA_A,
    )
    assert record.active_rows == record.physical_width == 5


def test_ceiling_partition_matches_masked_ceiling_semantics() -> None:
    assert ceiling_partition(3, (1, 2, 4, 8)) == (4,)
    assert ceiling_partition(5, (1, 2, 4, 8)) == (8,)
    assert ceiling_partition(7, (1, 2, 4, 8)) == (8,)
    assert ceiling_partition(13, (1, 2, 4, 8)) == (8, 8)
    assert ceiling_partition(13, tuple(range(1, 9))) == (8, 5)
    with pytest.raises(ValueError, match="strictly increasing"):
        ceiling_partition(7, (1, 8, 4))


def test_d2_partition_recovers_expected_choices_and_constraints() -> None:
    table = _cost_table()
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
        assert d2_partition(rows, table) == want
    for rows in range(1, 9):
        assert d2_partition(rows, table) == (rows,)

    constrained = CostTable(
        tuple(_record(width, ms, workspace_bytes=width * 100) for width, ms in POST_PROMOTION_STEP_MS.items())
    )
    assert 8 not in d2_partition(8, constrained, max_workspace_bytes=700)
    assert 8 not in d2_partition(8, table, max_group_model_step_ms=60.0)


def test_d2_partition_ties_use_fewer_then_canonical_groups() -> None:
    flat = CostTable((_record(1, 10.0), _record(2, 20.0), _record(3, 30.0), _record(4, 40.0)))
    assert d2_partition(4, flat) == (4,)
    # Equal cost/count for 3+1 and 2+2; canonical descending tuple wins.
    equal_count = CostTable((_record(1, 10.0), _record(2, 20.0), _record(3, 30.0)))
    assert d2_partition(4, equal_count) == (3, 1)


def test_plan_d2_groups_preserves_dense_and_sparse_slot_identity() -> None:
    work = WorkItem(
        kind=WorkKind.DECODE,
        request_ids=tuple(range(100, 109)),
        row_to_request=tuple(range(100, 109)),
        slot_ids=tuple(range(9)),
    )
    groups = plan_d2_groups(work, _cost_table())
    assert tuple(group.physical_rows for group in groups) == (5, 4)
    assert tuple(request for group in groups for request in group.request_ids) == work.request_ids

    sparse = WorkItem(
        kind=WorkKind.DECODE,
        request_ids=(0, 1, 2, 3, 4),
        row_to_request=(0, 1, 2, 3, 4),
        slot_ids=(0, 1, 2, 4, 6),
        active_mask=(True, True, True, False, True, False, True),
    )
    sparse_group = plan_d2_groups(sparse, _cost_table())[0]
    assert sparse_group.physical_rows == 5
    assert sparse_group.global_slot_indices == (0, 1, 2, 4, 6)
    assert sparse_group.request_ids == sparse.request_ids


def test_cost_table_from_artifact_requires_clean_matching_identity(tmp_path: Path) -> None:
    path = tmp_path / "cost-map.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    table = cost_table_from_artifact(path, expected=_expectation())
    assert table.widths == tuple(range(1, 9))
    assert d2_partition(13, table) == (7, 6)
    assert table.identity == _expectation()

    payload = _artifact()
    payload["source_measurement"]["provenance"]["dirty"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="clean source measurement"):
        cost_table_from_artifact(path, expected=_expectation())


def test_cost_table_from_artifact_rejects_failed_wrong_or_malformed_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "cost-map.json"
    for mutate, match in (
        (lambda p: p.update(status="failed"), "accepted passed measurement"),
        (lambda p: p["identity"].update(backend="cuda_sm86"), "identity mismatch"),
        (lambda p: p["physical_group_records"][0].update(model_step_ms=float("nan")), "finite and positive"),
        (lambda p: p["physical_group_records"].pop(), "physical widths"),
    ):
        payload = _artifact()
        mutate(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            cost_table_from_artifact(path, expected=_expectation())


def test_retained_d2_cost_map_recovers_current_choices() -> None:
    path = Path("benchmarks/results/2026-08-20-concurrency2-qwen38-d2-cost-map.json")
    table = cost_table_from_artifact(path, expected=_expectation())
    assert d2_partition(9, table) == (5, 4)
    assert d2_partition(13, table) == (7, 6)
    assert d2_partition(16, table) == (8, 8)
