from __future__ import annotations

from dataclasses import replace

import pytest

from hipengine.dispatch import WorkKind
from hipengine.generation.batch_scheduler import GeneratedToken
from hipengine.generation.engine_loop import ResidentEngineLoop
from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
from hipengine.speculative import (
    AcceptResult,
    ProviderAttachment,
    ProviderCatchupMode,
    SpecCycleResult,
    SpecCycleTelemetry,
    SpecCycleTransaction,
    SpecK0Class,
    SpecPlanReason,
    SpecTransactionMode,
    SpeculativeCapability,
)


class _CycleRunner:
    def __init__(self) -> None:
        self.prefills = []
        self.decodes = []
        self.cycle_plans = []
        self._transaction_id = 0
        self._decode_counts: dict[int, int] = {}

    def prefill_batch(self, work, *, commit):
        assert commit
        self.prefills.append(work)

    def decode_batch(self, work, *, commit):
        assert commit
        self.decodes.append(work)
        output = []
        for request_id in work.request_ids:
            count = self._decode_counts.get(request_id, 0)
            self._decode_counts[request_id] = count + 1
            output.append(GeneratedToken(request_id, 9000 + request_id * 10 + count))
        return tuple(output)

    def compact_batch(self, moves):
        return None

    def reclaim(self, completed):
        return None

    def speculative_capability(self, request_semantics):
        return SpeculativeCapability(
            capability_key="fake:mtp2:strict",
            target_key="fake_target",
            provider_key="fake_nextn",
            method_key="mtp2",
            policy_fingerprint="fake-policy:v1",
            execution_profile="strict",
            kv_backend_key="fake_kv",
            attachment=ProviderAttachment.TARGET_ATTACHED,
            catchup_mode=ProviderCatchupMode.TARGET_OUTPUT,
            supported_modes=("verify_chain",),
            supported_sampling_modes=("greedy",),
            max_requests=4,
            max_candidates_per_request=3,
            max_frontier_rows=16,
            proposal_widths=(1, 2, 4),
            target_row_buckets=(1, 2, 4, 8, 16),
            target_transaction_mode=SpecTransactionMode.PACKED_SCRATCH,
            provider_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
            graph_supported=False,
            eager_supported=True,
            strict_fallback_key="fake_target_ar",
            max_context_tokens=128,
        )

    def speculative_claims_fit(self, plan):
        return True

    def execute_speculative_cycle(self, plan, *, commit):
        assert commit
        self.cycle_plans.append(plan)
        self._transaction_id += 1
        provider_ids = plan.speculative_request_ids
        transaction = SpecCycleTransaction(
            operation_id=plan.operation_id,
            transaction_id=self._transaction_id,
            cycle_id=plan.cycle_id,
            request_ids=plan.request_ids,
            reserved_claims=ResourceClaimSet.from_mapping(
                plan.operation_id,
                {"fake.target": plan.logical_frontier_rows},
                lifetime=ClaimLifetime.TRANSACTION,
            ),
            pre_target_cursors=(0,) * len(plan.request_ids),
            pre_rng_counters=(0,) * len(plan.request_ids),
            target_transaction_mode=plan.target_transaction_mode,
            target_owner=f"{plan.operation_id}:target",
            target_checkpoint_ids=tuple(
                f"target:{request_id}" for request_id in plan.request_ids
            ),
            pre_provider_cursors=(0,) * len(provider_ids),
            provider_transaction_mode=plan.provider_transaction_mode,
            provider_owner=(
                None if not provider_ids else f"{plan.operation_id}:provider"
            ),
            provider_request_ids=provider_ids,
            provider_checkpoint_ids=tuple(
                f"provider:{request_id}" for request_id in provider_ids
            ),
            target_open=True,
            provider_open=bool(provider_ids),
            target_committed=True,
            provider_committed=bool(provider_ids),
        )
        accepted_tokens = tuple(
            tuple(request_id * 100 + depth for depth in range(1, count + 1))
            for request_id, count in zip(
                plan.request_ids, plan.candidate_counts, strict=True
            )
        )
        corrections = tuple(8000 + request_id for request_id in plan.request_ids)
        outputs = tuple(
            (*tokens, correction)
            for tokens, correction in zip(accepted_tokens, corrections, strict=True)
        )
        accept = AcceptResult(
            request_ids=plan.request_ids,
            accepted_counts=plan.candidate_counts,
            accepted_tokens=accepted_tokens,
            transaction_id=transaction.transaction_id,
            correction_or_bonus_tokens=corrections,
            target_cursor_deltas=tuple(map(len, outputs)),
            provider_cursor_deltas=plan.candidate_counts,
            finish_reasons=(None,) * len(plan.request_ids),
        )
        telemetry = SpecCycleTelemetry(
            operation_id=plan.operation_id,
            request_ids=plan.request_ids,
            candidate_counts=plan.candidate_counts,
            plan_reasons=plan.reasons,
            k0_classes=plan.k0_classes,
            proposal_widths=plan.proposal_widths,
            target_row_decomposition=plan.target_row_decomposition,
            execution_route=plan.execution_route,
        )
        return SpecCycleResult.committed(transaction, accept, telemetry=telemetry)


class _OpaquePreferredRunner(_CycleRunner):
    def speculative_frontier_available(self, plan):
        return False

    def speculative_component_claims(self, plan):
        raise AssertionError("frontier claims must not run")

    def reserve_speculative_claims(self, claims):
        raise AssertionError("frontier reserve must not run")

    def release_speculative_claims(self, reservation):
        raise AssertionError("frontier release must not run")

    def prepare_speculative_requests(self, *args, **kwargs):
        raise AssertionError("frontier prepare must not run")

    def propose_speculative_batch(self, *args, **kwargs):
        raise AssertionError("frontier proposal must not run")

    def execute_target_frontier(self, *args, **kwargs):
        raise AssertionError("frontier target must not run")


class _RecoveringOpaqueRunner(_OpaquePreferredRunner):
    def __init__(self) -> None:
        super().__init__()
        self.recovered = []
        self.disabled = False

    def speculative_capability(self, request_semantics):
        if self.disabled:
            return None
        return super().speculative_capability(request_semantics)

    def execute_speculative_cycle(self, plan, *, commit):
        raise RuntimeError("opaque precommit failure")

    def recover_speculative_cycle_failure(self, plan, error):
        self.recovered.append((plan, error))
        self.disabled = True
        return True


class _NoSpecRunner(_CycleRunner):
    def speculative_capability(self, request_semantics):
        return None


class _SwitchableQualificationRunner(_CycleRunner):
    def __init__(self) -> None:
        super().__init__()
        self.enabled = True
        self.k0_prepares = []

    def speculative_capability(self, request_semantics):
        if not self.enabled:
            return None
        return super().speculative_capability(request_semantics)

    def prepare_speculative_k0(self, plan, request_semantics, *, stream=None):
        self.k0_prepares.append(
            (
                plan,
                tuple(item.request_id for item in request_semantics),
                stream,
            )
        )


class _C1OnlyRunner(_CycleRunner):
    def __init__(self) -> None:
        super().__init__()
        self.k0_prepares = []

    def speculative_capability(self, request_semantics):
        if len(tuple(request_semantics)) != 1:
            return None
        return super().speculative_capability(request_semantics)

    def prepare_speculative_k0(self, plan, request_semantics, *, stream=None):
        self.k0_prepares.append(
            (
                plan,
                tuple(item.request_id for item in request_semantics),
                stream,
            )
        )


def test_runner_can_select_bounded_opaque_cycle_before_frontier_mutation() -> None:
    runner = _OpaquePreferredRunner()
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    request_id = loop.submit_speculative(
        [10],
        max_new_tokens=3,
        desired_candidate_count=2,
    )

    events = loop.poll(max_ticks=2)

    assert len(runner.cycle_plans) == 1
    assert [event.token_id for event in events if event.kind == "token"] == [
        1,
        2,
        8000,
    ]
    assert loop.completed[request_id].generated_tokens == (1, 2, 8000)


def test_opaque_cycle_failure_uses_shared_recovery_boundary() -> None:
    runner = _RecoveringOpaqueRunner()
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    request_id = loop.submit_speculative(
        [10],
        max_new_tokens=3,
        desired_candidate_count=2,
    )

    events = loop.poll(max_ticks=4)

    assert len(runner.recovered) == 1
    assert str(runner.recovered[0][1]) == "opaque precommit failure"
    assert [event.token_id for event in events if event.kind == "token"] == [
        9000,
        9001,
        9002,
    ]
    assert loop.completed[request_id].generated_tokens == (9000, 9001, 9002)


class _DisjointMixedRunner(_CycleRunner):
    def speculative_capability(self, request_semantics):
        semantics = tuple(request_semantics)
        return None if len(semantics) > 1 else super().speculative_capability(semantics)


class _PartitionedRunner(_CycleRunner):
    def speculative_partition_max_requests(self, work):
        del work
        return 2


class _WideCycleRunner(_CycleRunner):
    def speculative_capability(self, request_semantics):
        capability = super().speculative_capability(request_semantics)
        return replace(
            capability,
            capability_key="fake:mtp2:c8:strict",
            max_requests=8,
            max_frontier_rows=32,
            proposal_widths=tuple(range(1, 9)),
            target_row_buckets=tuple(range(1, 33)),
        )

    def speculative_partition_max_requests(self, work):
        del work
        return 8


class _WideRecoveringRunner(_WideCycleRunner):
    def __init__(self) -> None:
        super().__init__()
        self.recovered = []
        self.disabled = False

    def speculative_capability(self, request_semantics):
        if self.disabled:
            return None
        return super().speculative_capability(request_semantics)

    def execute_speculative_cycle(self, plan, *, commit):
        del commit
        raise RuntimeError("wide precommit failure")

    def recover_speculative_cycle_failure(self, plan, error):
        self.recovered.append((plan, error))
        self.disabled = True
        return True


class _WholeBatchARPolicyRunner(_CycleRunner):
    def speculative_partition_max_requests(self, work):
        del work
        return 0


def test_one_speculative_cycle_is_one_engine_tick_with_multi_token_events() -> None:
    runner = _CycleRunner()
    loop = ResidentEngineLoop(runner, capacity=2, prefill_chunk_size=8)

    request_id = loop.submit_speculative(
        [10],
        max_new_tokens=3,
        desired_candidate_count=2,
    )
    assert request_id == 0
    assert runner.cycle_plans == []

    events = loop.poll(max_ticks=2)

    assert len(runner.cycle_plans) == 1
    assert runner.decodes == []
    assert [event.work_kind for event in events if event.kind == "work"] == [
        WorkKind.PREFILL,
        WorkKind.VERIFY_CHAIN,
    ]
    token_events = [event for event in events if event.kind == "token"]
    assert [event.token_id for event in token_events] == [1, 2, 8000]
    assert loop.completed[request_id].generated_tokens == (1, 2, 8000)
    assert loop.completed[request_id].finish_reason == "length"


def test_late_ar_arrival_joins_future_mixed_cycle_without_second_loop() -> None:
    runner = _CycleRunner()
    loop = ResidentEngineLoop(
        runner,
        capacity=2,
        prefill_chunk_size=8,
        prefill_decode_policy="protect_ttft",
    )
    spec_id = loop.submit_speculative(
        [10],
        max_new_tokens=6,
        desired_candidate_count=1,
    )
    loop.poll(max_ticks=2)
    assert len(runner.cycle_plans) == 1
    assert loop.completed.get(spec_id) is None

    ar_id = loop.submit([20], max_new_tokens=1)
    assert runner.cycle_plans[-1].request_ids == (spec_id,)
    loop.poll(max_ticks=1)  # admit and prefill the late AR row
    assert len(runner.cycle_plans) == 1
    events = loop.poll(max_ticks=1)

    assert len(runner.cycle_plans) == 2
    mixed = runner.cycle_plans[-1]
    assert mixed.request_ids == (spec_id, ar_id)
    assert mixed.candidate_counts == (1, 0)
    assert mixed.reasons == (
        SpecPlanReason.SPECULATIVE_QUALIFIED,
        SpecPlanReason.POLICY_SELECTED_AR,
    )
    assert mixed.k0_classes == (
        SpecK0Class.NOT_K0,
        SpecK0Class.PURE,
    )
    assert [event.request_id for event in events if event.kind == "completed"] == [
        ar_id
    ]
    assert loop.completed[ar_id].generated_tokens == (8001,)
    assert loop.active_count == 1


def test_realized_route_switches_mtp_k0_mtp_only_at_cycle_boundaries() -> None:
    runner = _SwitchableQualificationRunner()
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    request_id = loop.submit_speculative(
        [10],
        max_new_tokens=6,
        desired_candidate_count=1,
    )

    loop.poll(max_ticks=2)
    runner.enabled = False
    loop.poll(max_ticks=1)
    runner.enabled = True
    loop.poll(max_ticks=1)

    assert [plan.request_ids for plan in runner.cycle_plans] == [
        (request_id,),
        (request_id,),
    ]
    assert len(runner.k0_prepares) == 1
    assert runner.k0_prepares[0][1] == (request_id,)
    assert len(runner.decodes) == 1
    assert runner.decodes[0].request_ids == (request_id,)
    assert loop.completed.get(request_id) is None
    assert loop.active_count == 1


def test_realized_c2_without_capability_prepares_k0_before_one_ar_batch() -> None:
    runner = _C1OnlyRunner()
    loop = ResidentEngineLoop(
        runner,
        capacity=2,
        prefill_chunk_size=8,
        prefill_decode_policy="protect_ttft",
    )
    first = loop.submit_speculative(
        [10],
        max_new_tokens=2,
        desired_candidate_count=1,
    )
    second = loop.submit_speculative(
        [20],
        max_new_tokens=2,
        desired_candidate_count=1,
    )

    loop.poll(max_ticks=3)

    assert runner.cycle_plans == []
    assert len(runner.k0_prepares) == 1
    assert runner.k0_prepares[0][1] == (first, second)
    assert len(runner.decodes) == 1
    assert runner.decodes[0].request_ids == (first, second)
    assert loop.completed.get(first) is None
    assert loop.completed.get(second) is None
    assert loop.active_count == 2


def test_realized_speculative_width_switches_c1_c2_c1_at_cycle_boundaries() -> None:
    runner = _CycleRunner()
    loop = ResidentEngineLoop(
        runner,
        capacity=2,
        prefill_chunk_size=8,
        prefill_decode_policy="protect_ttft",
    )
    survivor = loop.submit_speculative(
        [10],
        max_new_tokens=8,
        desired_candidate_count=1,
    )
    loop.poll(max_ticks=2)
    late = loop.submit_speculative(
        [20],
        max_new_tokens=2,
        desired_candidate_count=1,
    )

    loop.poll(max_ticks=1)  # admit and prefill the late request
    loop.poll(max_ticks=2)  # one C2 cycle, then the survivor returns to C1

    assert [plan.request_ids for plan in runner.cycle_plans[:3]] == [
        (survivor,),
        (survivor, late),
        (survivor,),
    ]
    assert [plan.candidate_counts for plan in runner.cycle_plans[:3]] == [
        (1,),
        (1, 1),
        (1,),
    ]
    assert loop.completed[late].generated_tokens == (101, 8001)
    assert loop.completed.get(survivor) is None
    assert loop.active_count == 1


def test_ar_neighbor_uses_disjoint_decode_without_erasing_speculative_intent() -> None:
    runner = _DisjointMixedRunner()
    loop = ResidentEngineLoop(
        runner,
        capacity=2,
        prefill_chunk_size=8,
        prefill_decode_policy="protect_ttft",
    )
    spec_id = loop.submit_speculative(
        [10],
        max_new_tokens=6,
        desired_candidate_count=1,
    )
    loop.poll(max_ticks=2)
    ar_id = loop.submit([20], max_new_tokens=1)
    loop.poll(max_ticks=1)

    events = loop.poll(max_ticks=1)

    assert runner.cycle_plans[-1].request_ids == (spec_id,)
    assert runner.decodes[-1].request_ids == (ar_id,)
    assert loop.completed[ar_id].generated_tokens == (9000 + ar_id * 10,)
    recent = loop.recent_speculative_plans[-2:]
    assert recent[0].request_ids == (spec_id, ar_id)
    assert recent[0].candidate_counts == (0, 0)
    assert recent[0].k0_classes == (
        SpecK0Class.TRANSITIONAL,
        SpecK0Class.PURE,
    )
    assert recent[1].request_ids == (spec_id,)
    assert recent[1].candidate_counts == (1,)
    assert [event.request_id for event in events if event.kind == "completed"] == [
        ar_id
    ]
    assert loop.active_count == 1

@pytest.mark.parametrize("width", (5, 8))
def test_wide_physical_due_work_owns_every_request_in_one_cycle(width: int) -> None:
    runner = _WideCycleRunner()
    loop = ResidentEngineLoop(
        runner,
        capacity=width,
        prefill_chunk_size=8,
        prefill_decode_policy="protect_ttft",
    )
    request_ids = tuple(
        loop.submit_speculative(
            [10 + index],
            max_new_tokens=2,
            desired_candidate_count=1,
        )
        for index in range(width)
    )

    events = loop.poll(max_ticks=width + 2)

    assert [plan.request_ids for plan in runner.cycle_plans] == [request_ids]
    assert runner.cycle_plans[0].proposal_widths == (width,)
    assert runner.decodes == []
    assert [event.request_id for event in events if event.kind == "completed"] == list(
        request_ids
    )
    assert loop.active_count == 0


def test_wide_physical_failure_recovers_all_requests_without_partial_ownership() -> None:
    runner = _WideRecoveringRunner()
    loop = ResidentEngineLoop(
        runner,
        capacity=8,
        prefill_chunk_size=8,
        prefill_decode_policy="protect_ttft",
    )
    request_ids = tuple(
        loop.submit_speculative(
            [10 + index],
            max_new_tokens=2,
            desired_candidate_count=3,
        )
        for index in range(8)
    )

    events = loop.poll(max_ticks=10)

    assert len(runner.recovered) == 1
    assert runner.recovered[0][0].request_ids == request_ids
    assert str(runner.recovered[0][1]) == "wide precommit failure"
    assert [work.request_ids for work in runner.decodes] == [request_ids, request_ids]
    assert [event.request_id for event in events if event.kind == "completed"] == list(
        request_ids
    )
    assert loop.active_count == 0


def test_wide_due_work_partitions_each_speculative_row_once_per_tick() -> None:
    runner = _PartitionedRunner()
    loop = ResidentEngineLoop(
        runner,
        capacity=5,
        prefill_chunk_size=8,
        prefill_decode_policy="protect_ttft",
    )
    request_ids = tuple(
        loop.submit_speculative(
            [10 + index],
            max_new_tokens=2,
            desired_candidate_count=1,
        )
        for index in range(5)
    )

    events = loop.poll(max_ticks=7)

    assert [plan.request_ids for plan in runner.cycle_plans] == [
        request_ids[:2],
        request_ids[2:4],
        request_ids[4:],
    ]
    assert [plan.proposal_widths for plan in runner.cycle_plans] == [(2,), (2,), (1,)]
    assert runner.decodes == []
    assert [event.request_id for event in events if event.kind == "completed"] == list(
        request_ids
    )
    assert all(loop.completed[request_id].generated_tokens for request_id in request_ids)


def test_zero_partition_bound_routes_wide_due_work_to_one_ar_batch() -> None:
    runner = _WholeBatchARPolicyRunner()
    loop = ResidentEngineLoop(
        runner,
        capacity=5,
        prefill_chunk_size=8,
        prefill_decode_policy="protect_ttft",
    )
    request_ids = tuple(
        loop.submit_speculative(
            [10 + index],
            max_new_tokens=1,
            desired_candidate_count=1,
        )
        for index in range(5)
    )

    events = loop.poll(max_ticks=7)

    assert runner.cycle_plans == []
    assert [work.request_ids for work in runner.decodes] == [request_ids]
    assert loop.last_speculative_plan is not None
    assert loop.last_speculative_plan.is_ar_only
    assert [event.request_id for event in events if event.kind == "completed"] == list(
        request_ids
    )


def test_wide_mixed_due_work_partitions_spec_rows_and_decodes_ar_once() -> None:
    runner = _PartitionedRunner()
    loop = ResidentEngineLoop(
        runner,
        capacity=6,
        prefill_chunk_size=8,
        prefill_decode_policy="protect_ttft",
    )
    spec_ids = tuple(
        loop.submit_speculative(
            [10 + index],
            max_new_tokens=2,
            desired_candidate_count=1,
        )
        for index in range(5)
    )
    ar_id = loop.submit([99], max_new_tokens=1)

    events = loop.poll(max_ticks=8)

    assert [plan.request_ids for plan in runner.cycle_plans] == [
        spec_ids[:2],
        spec_ids[2:4],
        spec_ids[4:],
    ]
    assert runner.decodes[-1].request_ids == (ar_id,)
    assert [event.request_id for event in events if event.kind == "completed"] == [
        *spec_ids,
        ar_id,
    ]
    assert loop.completed[ar_id].generated_tokens == (9000 + ar_id * 10,)


def test_missing_capability_uses_normal_decode_k0_path() -> None:
    runner = _NoSpecRunner()
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    request_id = loop.submit_speculative(
        [10],
        max_new_tokens=1,
        desired_candidate_count=2,
    )

    events = loop.poll(max_ticks=2)

    assert runner.cycle_plans == []
    assert len(runner.decodes) == 1
    # The child retained speculative intent, but this cycle had no provider
    # capability and therefore used transitional K0 rather than permanent AR.
    plan = loop.last_speculative_plan
    assert plan is not None
    assert plan.k0_classes == (SpecK0Class.TRANSITIONAL,)
    assert [event.work_kind for event in events if event.kind == "work"] == [
        WorkKind.PREFILL,
        WorkKind.DECODE,
    ]
    assert loop.completed[request_id].generated_tokens == (9000,)


def test_cancelled_speculative_request_never_opens_a_cycle() -> None:
    runner = _CycleRunner()
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    request_id = loop.submit_speculative(
        [10],
        max_new_tokens=4,
        desired_candidate_count=2,
    )

    assert loop.cancel(request_id)
    assert runner.cycle_plans == []
    assert loop.completed[request_id].finish_reason == "cancel"
    assert loop.poll(max_ticks=1) == ()


def test_two_speculative_requests_stagger_retire_and_refill_with_k3() -> None:
    runner = _CycleRunner()
    loop = ResidentEngineLoop(
        runner,
        capacity=2,
        prefill_chunk_size=8,
        prefill_decode_policy="protect_ttft",
    )
    first = loop.submit_speculative(
        [10],
        max_new_tokens=2,
        desired_candidate_count=3,
    )
    survivor = loop.submit_speculative(
        [20],
        max_new_tokens=8,
        desired_candidate_count=3,
    )

    loop.poll(max_ticks=3)

    assert runner.cycle_plans[0].request_ids == (first, survivor)
    assert runner.cycle_plans[0].candidate_counts == (1, 3)
    assert loop.completed[first].generated_tokens == (1, 8000)
    assert loop.active_count == 1

    refill = loop.submit_speculative(
        [30],
        max_new_tokens=3,
        desired_candidate_count=2,
    )
    loop.poll(max_ticks=2)

    assert runner.cycle_plans[1].request_ids == (refill, survivor)
    assert runner.cycle_plans[1].candidate_counts == (2, 3)
    assert loop.completed[refill].generated_tokens == (201, 202, 8002)
    assert len(loop.completed[survivor].generated_tokens) == 8
    assert loop.active_count == 0
