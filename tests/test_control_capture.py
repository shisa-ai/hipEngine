"""RED/GREEN tests for the standardized control-capture model.

Covers ``hipengine/benchmark/control_capture.py`` (PN1):

- schema round-trip through the existing ``load_control_capture`` and
  ``load_control_fixture`` loaders;
- per-field correctness of the derived control record for a c1 schedule;
- deterministic ownership hashes that change when ownership changes;
- the execution-profile gate's ``compare_control_records`` detects injected
  ownership/lifecycle corruption (swapped row, wrong position, wrong KV owner,
  wrong scatter owner, stale graph bucket, missing record); and
- an independent frozen fixture agrees with a live-derived capture for an
  identical c1 schedule.
"""

from __future__ import annotations

import json

import pytest

from hipengine.benchmark.control_capture import (
    RowControlPrimitives,
    active_mask_hash,
    build_control_capture,
    build_control_fixture,
    derive_control_record,
    derive_control_records,
    mask_manifest_hash,
    route_decision_hash,
    route_scatter_owner_hash,
    transaction_id,
)
from hipengine.benchmark.execution_profiles import (
    compare_control_records,
    load_control_capture,
    load_control_fixture,
)

SCENARIO = "qwen36_zbook_c1_smoke"
RUN_ID = "run-0001"
ROUTE_TOP_K = 8
GRAPH_BUCKET = "eager"


def _primitive(
    *,
    step: int,
    request_id: str = "prompt-0",
    token: int | None = None,
    position: int | None = None,
    context: int | None = None,
    **overrides,
) -> RowControlPrimitives:
    row_token = int(100 + step) if token is None else token
    row_position = int(512 + step - 1) if position is None else position
    base = {
        "scenario_id": SCENARIO,
        "scenario_step": step,
        "request_id": request_id,
        "input_token_id": row_token,
        "position": row_position,
        "context_length": context if context is not None else row_position + 1,
        "route_top_k": ROUTE_TOP_K,
        "graph_bucket": GRAPH_BUCKET,
        "rng_seed": 42,
    }
    base.update(overrides)
    return RowControlPrimitives(**base)


def _c1_primitives(steps: int = 4) -> tuple[RowControlPrimitives, ...]:
    return tuple(_primitive(step=step) for step in range(steps))


def test_control_capture_round_trip(tmp_path) -> None:
    records = derive_control_records(_c1_primitives())
    payload = build_control_capture(scenario_id=SCENARIO, run_id=RUN_ID, records=records)
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    run_id, loaded = load_control_capture(path)
    assert run_id == RUN_ID
    assert [record.to_dict() for record in loaded] == [record.to_dict() for record in records]


def test_control_fixture_round_trip(tmp_path) -> None:
    records = derive_control_records(_c1_primitives())
    payload = build_control_fixture(scenario_id=SCENARIO, records=records)
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_control_fixture(path)
    assert [record.to_dict() for record in loaded] == [record.to_dict() for record in records]


def test_derived_record_field_correctness() -> None:
    record = derive_control_record(_primitive(step=3, token=777, position=514))
    assert record.scenario_id == SCENARIO
    assert record.scenario_step == 3
    assert record.request_id == "prompt-0"
    assert record.input_token_id == 777
    assert record.position == 514
    assert record.context_length == 515
    assert record.physical_slot == 0
    assert record.execution_row == 0
    assert record.physical_width == 1
    assert record.active is True
    assert record.work_class == "decode"
    assert record.transaction_phase == "commit"
    assert record.accepted_token_count == 1
    assert record.route_owner_request_id == "prompt-0"
    assert record.route_top_k == ROUTE_TOP_K
    assert record.state_owner_request_id == "prompt-0"
    assert record.state_update_ordinal == 3
    assert record.rng_owner_request_id == "prompt-0"
    assert record.rng_seed == 42
    assert record.rng_counter == 3
    assert record.graph_bucket == GRAPH_BUCKET
    # KV layout is contiguous c1: live count = context, appended at the tail.
    assert record.kv_base_offset == 0
    assert record.kv_live_count == 515
    assert record.kv_token_position == 514
    assert record.kv_append_ordinal == 514
    assert record.kv_evict is False
    assert record.kv_values_finite is True
    assert record.state_values_finite is True
    assert record.route_values_finite is True
    assert record.publication_ordinal == 3
    assert record.transaction_id == "prompt-0:3"


def test_ownership_hashes_are_deterministic_and_sensitive() -> None:
    first = derive_control_record(_primitive(step=1))
    second = derive_control_record(_primitive(step=1))
    assert first.route_decision_hash == second.route_decision_hash
    assert first.route_scatter_owner_hash == second.route_scatter_owner_hash
    assert first.active_mask_hash == second.active_mask_hash
    assert first.mask_manifest_hash == mask_manifest_hash(active_mask_hash_value=first.active_mask_hash)
    # Changing the owner changes the scatter and decision ownership hashes.
    other = derive_control_record(_primitive(step=1, request_id="prompt-other"))
    assert other.route_scatter_owner_hash != first.route_scatter_owner_hash
    assert other.route_decision_hash != first.route_decision_hash
    assert other.state_owner_request_id != first.state_owner_request_id
    # Changing the schedule changes the decision hash.
    later = derive_control_record(_primitive(step=2))
    assert later.route_decision_hash != first.route_decision_hash


def test_transaction_and_mask_hashes_are_stable() -> None:
    assert transaction_id(request_id="prompt-0", scenario_step=5) == "prompt-0:5"
    mask_a = active_mask_hash(active_rows=[(0, 1)])
    mask_b = active_mask_hash(active_rows=[(0, 1)])
    assert mask_a == mask_b
    assert active_mask_hash(active_rows=[(0, 4)]) != mask_a
    assert route_scatter_owner_hash(rows=[(0, "prompt-0")]) == route_scatter_owner_hash(
        rows=[(0, "prompt-0")]
    )
    assert route_decision_hash(
        request_id="prompt-0", scenario_step=3, route_top_k=8, graph_bucket="eager"
    ) == route_decision_hash(
        request_id="prompt-0", scenario_step=3, route_top_k=8, graph_bucket="eager"
    )


def test_compare_passes_for_identical_fixture_and_capture() -> None:
    expected = derive_control_records(_c1_primitives())
    actual = derive_control_records(_c1_primitives())
    comparison = compare_control_records(expected, actual)
    assert comparison["passed"] is True
    assert comparison["mismatches"] == []


def test_compare_detects_swapped_row_owner() -> None:
    expected = derive_control_records(_c1_primitives())
    # Same (step, request_id) key but the row is decoded at the wrong physical
    # slot/row: a swapped-row ownership corruption.
    corrupted = [
        record
        if record.scenario_step != 2
        else derive_control_record(
            _primitive(step=2, physical_slot=1, physical_width=2, execution_row=1)
        )
        for record in expected
    ]
    comparison = compare_control_records(expected, corrupted)
    assert comparison["passed"] is False
    field_names = {item["field"] for item in comparison["mismatches"]}
    assert {"physical_slot", "execution_row", "physical_width",
            "active_mask_hash", "mask_manifest_hash",
            "route_scatter_owner_hash"} <= field_names


def test_compare_detects_wrong_request_owner_key() -> None:
    expected = derive_control_records(_c1_primitives())
    # A different owner shifts the record key and is detected as a missing
    # (unmatched) record.
    corrupted = [
        record
        if record.scenario_step != 2
        else derive_control_record(_primitive(step=2, request_id="prompt-wrong"))
        for record in expected
    ]
    comparison = compare_control_records(expected, corrupted)
    assert comparison["passed"] is False
    assert any(item["field"] == "__record__" for item in comparison["mismatches"])


def test_compare_detects_wrong_position() -> None:
    expected = derive_control_records(_c1_primitives())
    corrupted = [
        record if record.scenario_step != 2 else derive_control_record(
            _primitive(step=2, position=999, context=1000)
        )
        for record in expected
    ]
    comparison = compare_control_records(expected, corrupted)
    assert comparison["passed"] is False
    assert {"position", "context_length", "kv_live_count"} <= {
        item["field"] for item in comparison["mismatches"]
    }


def test_compare_detects_wrong_kv_owner() -> None:
    expected = derive_control_records(_c1_primitives())
    corrupted = [
        record if record.scenario_step != 2 else derive_control_record(
            _primitive(step=2, kv_base_offset=4096)
        )
        for record in expected
    ]
    comparison = compare_control_records(expected, corrupted)
    assert comparison["passed"] is False
    assert "kv_base_offset" in {item["field"] for item in comparison["mismatches"]}


def test_compare_detects_wrong_scatter_owner() -> None:
    expected = derive_control_records(_c1_primitives())
    corrupted = [
        record if record.scenario_step != 2 else derive_control_record(
            _primitive(step=2, request_id="prompt-0", graph_bucket="wrong-bucket")
        )
        for record in expected
    ]
    comparison = compare_control_records(expected, corrupted)
    assert comparison["passed"] is False
    assert "graph_bucket" in {item["field"] for item in comparison["mismatches"]}


def test_compare_detects_stale_graph_bucket() -> None:
    expected = derive_control_records(_c1_primitives())
    corrupted = list(expected)
    stale = corrupted[2]
    # Rebuild with a stale graph bucket via the same primitives except bucket.
    corrupted[2] = derive_control_record(
        _primitive(step=2, graph_bucket="stale-graph-bucket")
    )
    comparison = compare_control_records(expected, corrupted)
    assert comparison["passed"] is False
    assert {"graph_bucket", "route_decision_hash"} <= {
        item["field"] for item in comparison["mismatches"]
    }


def test_compare_detects_lifecycle_leak_missing_record() -> None:
    expected = derive_control_records(_c1_primitives())
    actual = list(expected)[:2]
    comparison = compare_control_records(expected, actual)
    assert comparison["passed"] is False
    assert any(item["field"] == "__record__" for item in comparison["mismatches"])


def test_derive_rejects_invalid_primitives() -> None:
    with pytest.raises(ValueError, match="context_length"):
        derive_control_record(_primitive(step=1, context=3, position=5))
    with pytest.raises(ValueError, match="physical_width"):
        derive_control_record(_primitive(step=1, physical_width=2, execution_row=2))
    with pytest.raises(ValueError, match="route_top_k"):
        derive_control_record(_primitive(step=1, route_top_k=0))


def test_fixture_and_capture_agree_across_c1_schedule() -> None:
    # Independent derivation from the same frozen schedule agrees.
    fixtures = build_control_fixture(
        scenario_id=SCENARIO, records=derive_control_records(_c1_primitives(7))
    )
    capture = build_control_capture(
        scenario_id=SCENARIO, run_id=RUN_ID, records=derive_control_records(_c1_primitives(7))
    )
    assert fixtures["controls"] == capture["controls"]
