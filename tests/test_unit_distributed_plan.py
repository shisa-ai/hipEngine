"""CPU tests for the immutable distributed plan identity."""

from __future__ import annotations

import pytest

from hipengine.core.device import Device
from hipengine.distributed.plan import (
    DistributedPlan,
    PlanError,
    RankSpec,
    device_ids,
    dtype_bytes,
    payload_bytes,
    plan_from_mapping,
)


def test_resolve_orders_ranks_and_roles() -> None:
    plan = DistributedPlan.resolve([0, 1], hidden_size=5120, topology_fingerprint="pcie:0d+10")
    assert plan.world_size == 2
    assert [spec.device for spec in plan.ranks] == [Device("hip", 0), Device("hip", 1)]
    assert plan.control_rank == 0
    assert plan.draft_rank is None
    assert not plan.is_single_rank


def test_resolve_with_explicit_roles() -> None:
    plan = DistributedPlan.resolve([1, 0], hidden_size=4096, control_rank=1, draft_rank=0)
    assert plan.control_rank == 1
    assert plan.draft_rank == 0
    assert plan.rank_spec(0).role == "draft"
    assert plan.rank_spec(1).role == "control"


def test_resolve_rejects_impossible_plans() -> None:
    with pytest.raises(PlanError):
        DistributedPlan.resolve([], hidden_size=4096)
    with pytest.raises(PlanError):
        DistributedPlan.resolve([0, 0], hidden_size=4096)
    with pytest.raises(PlanError):
        DistributedPlan.resolve([0, 1], hidden_size=0)
    with pytest.raises(PlanError):
        DistributedPlan.resolve([0, 1], hidden_size=4096, comm_dtype="int8")
    with pytest.raises(PlanError):
        DistributedPlan.resolve([0, 1], hidden_size=4096, algorithm="magic")
    with pytest.raises(PlanError):
        DistributedPlan.resolve([0, 1], hidden_size=4096, draft_rank=0, control_rank=0)
    with pytest.raises(PlanError):
        RankSpec(rank=0, device=Device("cpu"))


def test_plan_hash_is_deterministic_and_sensitive() -> None:
    base = DistributedPlan.resolve([0, 1], hidden_size=5120, topology_fingerprint="topo-a")
    same = DistributedPlan.resolve([0, 1], hidden_size=5120, topology_fingerprint="topo-a")
    assert base.plan_hash() == same.plan_hash()

    reordered = DistributedPlan.resolve([1, 0], hidden_size=5120, topology_fingerprint="topo-a")
    other_dtype = DistributedPlan.resolve([0, 1], hidden_size=5120, comm_dtype="bf16", topology_fingerprint="topo-a")
    other_topo = DistributedPlan.resolve([0, 1], hidden_size=5120, topology_fingerprint="topo-b")
    other_hidden = DistributedPlan.resolve([0, 1], hidden_size=4096, topology_fingerprint="topo-a")
    assert len({base.plan_hash(), reordered.plan_hash(), other_dtype.plan_hash(), other_topo.plan_hash(), other_hidden.plan_hash()}) == 5


def test_plan_round_trips_through_mapping() -> None:
    plan = DistributedPlan.resolve([1, 0], hidden_size=2048, comm_dtype="fp16", topology_fingerprint="topo")
    rebuilt = plan_from_mapping(plan.to_dict())
    assert rebuilt == plan
    assert rebuilt.plan_hash() == plan.plan_hash()


def test_plan_from_mapping_rejects_malformed_payload() -> None:
    with pytest.raises(PlanError):
        plan_from_mapping({})
    with pytest.raises(PlanError):
        plan_from_mapping({"ranks": [{"rank": 0}], "hidden_size": 16})
    with pytest.raises(PlanError):
        plan_from_mapping("not-a-mapping")


def test_single_rank_plan_is_not_distributed() -> None:
    plan = DistributedPlan.resolve([1], hidden_size=5120)
    assert plan.is_single_rank
    assert plan.world_size == 1
    assert plan.device(0) == Device("hip", 1)


def test_payload_and_dtype_sizes() -> None:
    assert dtype_bytes("fp32") == 4
    assert dtype_bytes("fp16") == 2
    assert payload_bytes(rows=1, hidden_size=5120, dtype="fp32") == 20480
    assert payload_bytes(rows=5, hidden_size=5120, dtype="bf16") == 51200
    assert payload_bytes(rows=0, hidden_size=5120, dtype="fp32") == 0
    with pytest.raises(PlanError):
        payload_bytes(rows=-1, hidden_size=5120, dtype="fp32")
    with pytest.raises(PlanError):
        dtype_bytes("int8")


def test_device_ids_normalizes_ints_and_strings() -> None:
    assert device_ids([0, "hip:1"]) == (Device("hip", 0), Device("hip", 1))
