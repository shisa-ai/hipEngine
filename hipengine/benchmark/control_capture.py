"""Standardized execution-profile control-capture model and serializers.

PN1 "standardized actual-control capture" contract: every teacher row of a
strict/production run emits one :class:`ControlRecord` covering request/slot/row
ownership, token/position/context, masks, KVLiveSpans/transaction fields,
state/KV owner hashes, routing/scatter ownership, lifecycle counters, and the
resolved profile manifest identity.

This module provides the deterministic control-state model (a pure function from
observable scheduling primitives to a full :class:`ControlRecord`), the
independent expected-control fixture builder, and the capture/fixture
serializers. For the c1 single-request teacher-forced schedule every field is a
deterministic function of the scheduling primitives (one request, slot 0,
contiguous KV, one active row, no eviction/compaction), so a frozen golden
fixture and a live capture agree when ownership is preserved and diverge when a
candidate corrupts ownership (wrong slot, wrong position, wrong KV owner,
wrong scatter owner, stale graph bucket, lifecycle leak).

The route hashes are canonical ownership hashes over the observable routing
decision and expert-scatter ownership (not raw expert logits). They are binding
for strict captures and diagnostic for production candidates, matching the
execution-profile gate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from hipengine.benchmark.execution_profiles import (
    EXECUTION_PROFILE_CONTROL_CAPTURE_KIND,
    EXECUTION_PROFILE_CONTROL_CAPTURE_SCHEMA_VERSION,
    EXECUTION_PROFILE_CONTROL_FIXTURE_KIND,
    EXECUTION_PROFILE_CONTROL_FIXTURE_SCHEMA_VERSION,
    ControlRecord,
)

_WORK_CLASS_DECODE = "decode"
_TRANSACTION_PHASE_COMMIT = "commit"
_MASK_MANIFEST_KIND = "hipengine_active_mask_manifest_v1"


def _canonical_json(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def active_mask_hash(*, active_rows: Sequence[tuple[int, int]]) -> str:
    """Canonical hash of the active-mask rows as ``(execution_row, physical_width)``."""

    return _sha256_hex(_canonical_json([list(row) for row in active_rows]))


def mask_manifest_hash(*, active_mask_hash_value: str) -> str:
    """Canonical hash of the active-mask manifest for the decode step."""

    return _sha256_hex(
        _canonical_json(
            {
                "kind": _MASK_MANIFEST_KIND,
                "active_mask_hash": active_mask_hash_value,
            }
        )
    )


def transaction_id(*, request_id: str, scenario_step: int) -> str:
    return f"{request_id}:{int(scenario_step)}"


def route_decision_hash(
    *,
    request_id: str,
    scenario_step: int,
    route_top_k: int,
    graph_bucket: str,
) -> str:
    """Canonical ownership hash of the routing decision for this row."""

    return _sha256_hex(
        _canonical_json(
            {
                "request_id": request_id,
                "scenario_step": int(scenario_step),
                "route_top_k": int(route_top_k),
                "graph_bucket": graph_bucket,
            }
        )
    )


def route_scatter_owner_hash(
    *,
    rows: Sequence[tuple[int, str]],
) -> str:
    """Canonical hash of the expert-scatter owner map ``(execution_row, owner)``."""

    return _sha256_hex(_canonical_json([list(row) for row in rows]))


@dataclass(frozen=True, slots=True)
class RowControlPrimitives:
    """Observable scheduling primitives for one teacher row.

    The producer reads these from the live run; the fixture builder derives them
    from the frozen schedule spec. Both must agree for a correct candidate.
    """

    scenario_id: str
    scenario_step: int
    request_id: str
    input_token_id: int
    position: int
    context_length: int
    route_top_k: int
    graph_bucket: str
    rng_seed: int
    physical_slot: int = 0
    execution_row: int = 0
    physical_width: int = 1
    active: bool = True
    kv_base_offset: int = 0
    kv_evict: bool = False
    kv_values_finite: bool = True
    state_values_finite: bool = True
    route_values_finite: bool = True
    # Independent schedule fingerprint the fixture carries for provenance.
    schedule_fingerprint: str | None = field(default=None, repr=False)


def derive_control_record(primitives: RowControlPrimitives) -> ControlRecord:
    """Derive the full control record for one row from scheduling primitives."""

    if primitives.position < 0 or primitives.context_length <= 0:
        raise ValueError("position and context_length must be valid for a decode row")
    if primitives.context_length <= primitives.position:
        raise ValueError("context_length must exceed the row position")
    if primitives.physical_width <= 0 or primitives.execution_row >= primitives.physical_width:
        raise ValueError("physical_width must be positive and execution_row in range")
    if primitives.route_top_k <= 0:
        raise ValueError("route_top_k must be positive")
    kv_live_count = primitives.context_length
    kv_token_position = primitives.context_length - 1
    kv_append_ordinal = primitives.context_length - 1
    mask_hash = active_mask_hash(
        active_rows=[(primitives.execution_row, primitives.physical_width)]
    )
    scatter_owner_hash = route_scatter_owner_hash(
        rows=[(primitives.execution_row, primitives.request_id)]
    )
    decision_hash = route_decision_hash(
        request_id=primitives.request_id,
        scenario_step=primitives.scenario_step,
        route_top_k=primitives.route_top_k,
        graph_bucket=primitives.graph_bucket,
    )
    return ControlRecord(
        scenario_id=primitives.scenario_id,
        scenario_step=primitives.scenario_step,
        work_class=_WORK_CLASS_DECODE,
        request_id=primitives.request_id,
        physical_slot=primitives.physical_slot,
        execution_row=primitives.execution_row,
        physical_width=primitives.physical_width,
        input_token_id=primitives.input_token_id,
        position=primitives.position,
        context_length=primitives.context_length,
        active=primitives.active,
        active_mask_hash=mask_hash,
        mask_manifest_hash=mask_manifest_hash(active_mask_hash_value=mask_hash),
        publication_ordinal=primitives.scenario_step,
        transaction_id=transaction_id(
            request_id=primitives.request_id,
            scenario_step=primitives.scenario_step,
        ),
        transaction_phase=_TRANSACTION_PHASE_COMMIT,
        accepted_token_count=1,
        route_decision_hash=decision_hash,
        route_scatter_owner_hash=scatter_owner_hash,
        route_owner_request_id=primitives.request_id,
        route_top_k=primitives.route_top_k,
        kv_base_offset=primitives.kv_base_offset,
        kv_live_count=kv_live_count,
        kv_token_position=kv_token_position,
        kv_evict=primitives.kv_evict,
        kv_values_finite=primitives.kv_values_finite,
        kv_append_ordinal=kv_append_ordinal,
        state_owner_request_id=primitives.request_id,
        state_update_ordinal=primitives.scenario_step,
        state_values_finite=primitives.state_values_finite,
        rng_owner_request_id=primitives.request_id,
        rng_seed=primitives.rng_seed,
        rng_counter=primitives.scenario_step,
        route_values_finite=primitives.route_values_finite,
        graph_bucket=primitives.graph_bucket,
    )


def derive_control_records(
    primitives: Sequence[RowControlPrimitives],
) -> tuple[ControlRecord, ...]:
    return tuple(derive_control_record(item) for item in primitives)


def build_control_capture(
    *,
    scenario_id: str,
    run_id: str,
    records: Sequence[ControlRecord],
) -> dict[str, Any]:
    """Assemble a validated control-capture payload for a live run."""

    payload = {
        "kind": EXECUTION_PROFILE_CONTROL_CAPTURE_KIND,
        "schema_version": EXECUTION_PROFILE_CONTROL_CAPTURE_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "run_id": run_id,
        "controls": [record.to_dict() for record in records],
    }
    if not records or any(record.scenario_id != scenario_id for record in records):
        raise ValueError("control capture needs records matching its scenario_id")
    return payload


def build_control_fixture(
    *,
    scenario_id: str,
    records: Sequence[ControlRecord],
) -> dict[str, Any]:
    """Assemble a validated expected-control fixture payload."""

    payload = {
        "kind": EXECUTION_PROFILE_CONTROL_FIXTURE_KIND,
        "schema_version": EXECUTION_PROFILE_CONTROL_FIXTURE_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "controls": [record.to_dict() for record in records],
    }
    if not records or any(record.scenario_id != scenario_id for record in records):
        raise ValueError("control fixture needs records matching its scenario_id")
    return payload


__all__ = [
    "RowControlPrimitives",
    "active_mask_hash",
    "build_control_capture",
    "build_control_fixture",
    "derive_control_record",
    "derive_control_records",
    "mask_manifest_hash",
    "route_decision_hash",
    "route_scatter_owner_hash",
    "transaction_id",
]
