from collections import Counter
from types import SimpleNamespace

import pytest

from scripts.qwen4exp_layer2_profile_gate import CANDIDATES
from scripts.qwen4exp_prefill_restore_ab import (
    DECODER_OWNERS,
    FULL_DONOR_REFERENCE,
    OWNERSHIP,
    PREFILL_ONLY_OWNER,
    PREFILL_OWNERS,
    admitted_owner_bytes,
    donor_owner_bytes,
    owner_mode,
    prefill_owner_bytes,
    prepare_prefill_owner,
    prepare_reference_owner,
    require_reserve,
    round_order,
    validate_shared_graphs,
)
from tests.test_unit_qwen4exp_scratch_plan import config


def test_shared_graphs_allow_only_reviewed_uncaptured_switches():
    validate_shared_graphs([
        CANDIDATES["q8_blockscale_guarded_quad"].environment,
        CANDIDATES["q8_blockscale_guarded_ordered"].environment,
    ])
    with pytest.raises(ValueError, match="captured decode"):
        validate_shared_graphs([{"HIPENGINE_QWEN4_EXP_LAYER_GRAPHS": "0"}])


def test_three_arm_order_balances_positions_and_neighbor_directions():
    arms = ("before", "quad", "ordered")
    permutations = []
    for case_index in (0, 1):
        orders = [round_order(case_index, arms, repetition) for repetition in range(3)]
        for position in range(3):
            assert Counter(order[position] for order in orders) == Counter(arms)
        permutations.extend(tuple(order) for order in orders)
    assert len(set(permutations)) == 6


def test_two_arm_order_preserves_counterbalance():
    arms = ("before", "after")
    assert [round_order(0, arms, i) for i in range(3)] == [
        ["before", "after"], ["after", "before"], ["before", "after"]]


def test_owner_mode_allocates_a_decoder_only_for_the_executed_reference():
    assert owner_mode(reference=False) == PREFILL_ONLY_OWNER
    assert owner_mode(reference=True) == FULL_DONOR_REFERENCE
    assert "No donor decoder or donor KV cache is allocated" in OWNERSHIP[PREFILL_ONLY_OWNER]
    assert "independent full donor runner is executed" in OWNERSHIP[FULL_DONOR_REFERENCE]


def test_timing_comparison_admits_the_prefill_owner_set_only():
    from hipengine.loading.qwen4_exp_scratch import qwen4_exp_scratch_breakdown

    cfg = config()
    scratch = qwen4_exp_scratch_breakdown(cfg, context_tokens=256, prefill_chunk_size=16)
    admission_plan = {"scratch_bytes": 900, "kv_bytes": 300, "index_bytes": 200,
                      "runtime_state_bytes": 100, "reserve_bytes": 4 << 30}

    timing = admitted_owner_bytes(
        cfg, reference=False, context_tokens=256, chunk_size=16,
        admission_plan=admission_plan)
    reference = admitted_owner_bytes(
        cfg, reference=True, context_tokens=256, chunk_size=16,
        admission_plan=admission_plan)

    # Timing admits only the prefill owners; nothing decoder-shaped is admitted.
    assert not set(PREFILL_OWNERS) & set(DECODER_OWNERS)
    assert timing == prefill_owner_bytes(cfg, context_tokens=256, chunk_size=16)
    assert timing == sum(scratch[name] for name in PREFILL_OWNERS)
    # The executed reference still admits its full decoder/KV owner.
    assert reference == donor_owner_bytes(admission_plan) == 1500
    assert reference == sum(admission_plan[name] for name in DECODER_OWNERS)


def test_require_reserve_refuses_before_anything_is_allocated():
    assert require_reserve(100, 40, 60, mode=PREFILL_ONLY_OWNER) is None
    with pytest.raises(MemoryError, match="prefill_only_owner would consume the declared reserve"):
        require_reserve(99, 40, 60, mode=PREFILL_ONLY_OWNER)
    with pytest.raises(MemoryError, match="full_donor_reference"):
        require_reserve(0, 0, 1, mode=FULL_DONOR_REFERENCE)


@pytest.fixture
def simulated():
    from scripts.qwen4exp_allocation_census import simulated_runner

    with simulated_runner(config(), context=256, chunk=32) as (runner, runtime):
        yield runner, runtime


def test_prefill_owner_leaves_decode_ownership_and_exact_budget_intact(simulated):
    from hipengine.core.memory import memory_stats
    from scripts.qwen4exp_chunk_memory_probe import prepare_lazy_group_risk

    runner, runtime = simulated
    prepare_lazy_group_risk(runner)
    state, attention, index = runner.state, runner.attention_states, runner.index_states
    live_before = dict(runtime.live)

    owner, guard = prepare_prefill_owner(
        runner, chunk_size=16, context_tokens=256, reserve_bytes=1 << 30)

    bound = prefill_owner_bytes(runner.config, context_tokens=256, chunk_size=16)
    assert guard["owner_mode"] == PREFILL_ONLY_OWNER
    assert guard["admitted_owner_bytes"] == bound
    assert guard["observed_owner_bytes"] == bound
    assert guard["reserve_bytes"] == 1 << 30
    # The extra owner is prefill scratch plus prefill inputs, nothing else.
    assert sum(runtime.live.values()) - sum(live_before.values()) == bound
    assert memory_stats()["current_allocated_bytes"] == sum(runtime.live.values())
    assert guard["free_before_owner"] - guard["free_after_setup"] == bound
    assert guard["free_after_setup"] > guard["reserve_bytes"]

    # No decoder/KV owner was added or replaced.
    assert runner.state is state
    assert runner.attention_states is attention
    assert runner.index_states is index
    assert runner._prefill_workspaces[-1] is owner
    assert owner.prefill_chunk_size == 16
    assert runner.prefill_chunk_size == 32
    assert runner.gdn_prefill_scratch is runner._prefill_workspaces[0].gdn_prefill_scratch


def test_prefill_owner_refusal_allocates_nothing(simulated):
    runner, runtime = simulated
    workspaces = list(runner._prefill_workspaces)
    live = dict(runtime.live)

    with pytest.raises(MemoryError, match="prefill_only_owner would consume"):
        prepare_prefill_owner(
            runner, chunk_size=16, context_tokens=256, reserve_bytes=1 << 61)

    assert runner._prefill_workspaces == workspaces
    assert runtime.live == live


def test_prefill_owner_is_closed_and_deregistered_when_accounting_disagrees(
        simulated, monkeypatch):
    from hipengine.core import memory

    runner, runtime = simulated
    workspaces = list(runner._prefill_workspaces)
    live = dict(runtime.live)
    real = memory.memory_stats
    calls = []

    def inflated():
        calls.append(1)
        value = dict(real())
        if len(calls) > 1:
            value["current_allocated_bytes"] += 1
        return value

    with monkeypatch.context() as patch:
        patch.setattr(memory, "memory_stats", inflated)
        with pytest.raises(ValueError, match="allocated .* bytes, admitted"):
            prepare_prefill_owner(
                runner, chunk_size=16, context_tokens=256, reserve_bytes=1 << 30)

    assert len(calls) == 2
    assert runner._prefill_workspaces == workspaces
    assert runtime.live == live


class _FakeDonor:
    """Stand-in for the executed reference runner; it allocates nothing."""

    def __init__(self, block_host):
        self.attention_states = (SimpleNamespace(block_host=block_host),)
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def reference(monkeypatch):
    """Patch the donor constructor and its lazy queues, not the reference logic."""
    from hipengine.runtime import qwen4_exp_runner
    from scripts import qwen4exp_chunk_memory_probe

    monkeypatch.setattr(qwen4exp_chunk_memory_probe, "prepare_lazy_group_risk", lambda owner: [])
    block_host = [None]
    constructed = []

    def factory(resident, **kwargs):
        donor = _FakeDonor(block_host[0])
        constructed.append((resident, kwargs, donor))
        return donor

    monkeypatch.setattr(qwen4_exp_runner, "Qwen4ExpGGUFResidentModelRunner", factory)
    return constructed, block_host


def test_reference_owner_keeps_its_independent_decoder_allocation(simulated, reference):
    runner, runtime = simulated
    constructed, block_host = reference
    block_host[0] = runner.attention_states[0].block_host.copy()
    plan = {"scratch_bytes": 900, "kv_bytes": 300, "index_bytes": 200,
            "runtime_state_bytes": 100, "reserve_bytes": 4 << 30}
    live = dict(runtime.live)

    donor, guard = prepare_reference_owner(
        runner, chunk_size=16, allocation_plan=plan, reserve_bytes=1 << 30)

    assert len(constructed) == 1
    resident, kwargs, built = constructed[0]
    assert built is donor and resident is runner.resident
    assert kwargs == {"max_sequence_length": 256, "prefill_chunk_size": 16,
                      "backend": "cpu_census", "runtime": runtime}
    assert guard["owner_mode"] == FULL_DONOR_REFERENCE
    assert guard["admitted_owner_bytes"] == donor_owner_bytes(plan) == 1500
    assert guard["observed_owner_bytes"] is None
    assert donor.closed is False
    # The reference owner is the only thing this path allocates here.
    assert runtime.live == live


def test_reference_owner_refuses_before_constructing_a_donor(simulated, reference):
    runner, _ = simulated
    constructed, _ = reference

    with pytest.raises(MemoryError, match="full_donor_reference would consume"):
        prepare_reference_owner(
            runner, chunk_size=16, allocation_plan={key: 1 for key in DECODER_OWNERS},
            reserve_bytes=1 << 61)

    assert constructed == []


def test_reference_owner_closes_the_donor_when_the_kv_layout_differs(simulated, reference):
    runner, _ = simulated
    constructed, block_host = reference
    block_host[0] = runner.attention_states[0].block_host + 1

    with pytest.raises(ValueError, match="donor block indices differ"):
        prepare_reference_owner(
            runner, chunk_size=16,
            allocation_plan={key: 1 for key in DECODER_OWNERS}, reserve_bytes=1 << 30)

    assert len(constructed) == 1
    assert constructed[0][2].closed is True
