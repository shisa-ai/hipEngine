from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns

import pytest

from hipengine.dispatch import WorkItem, WorkKind
from hipengine.generation import (
    EngineLoopConfig,
    GeneratedToken,
    ResidentEngineLoop,
    TokenBudgetSLO,
)
from hipengine.dispatch.execution_planner import (
    ExecutionCompatibilityKey,
    ExecutionPlan,
    plan_execution_groups,
)
from hipengine.kvcache import (
    DenseKVAdmissionManager,
    DenseKVResidentRunnerAdapter,
    create_dense_bf16_backend,
)


def _work(width: int, *, slots: tuple[int, ...] | None = None) -> WorkItem:
    request_ids = tuple(range(width))
    slot_ids = tuple(range(width)) if slots is None else slots
    capacity = max(slot_ids, default=-1) + 1
    slot_set = set(slot_ids)
    return WorkItem(
        kind=WorkKind.DECODE,
        request_ids=request_ids,
        row_to_request=request_ids,
        slot_ids=slot_ids,
        active_mask=tuple(slot in slot_set for slot in range(capacity)),
    )


def _key(
    *,
    context: str = "ctx:0",
    masked: bool = True,
    compact: bool = False,
) -> ExecutionCompatibilityKey:
    return ExecutionCompatibilityKey(
        backend_key="hip_gfx1100",
        layout_key="dense-bf16",
        kernel_bundle_key="dense-global",
        work_class="decode",
        context_bucket=context,
        workspace_key="workspace:default",
        physical_widths=(1, 2, 4, 8),
        supports_masked_rows=masked,
        supports_dense_compaction=compact,
    )


@pytest.mark.parametrize("logical_c", range(1, 33))
def test_execution_planner_lowers_every_logical_width_through_c32(logical_c: int) -> None:
    plan = plan_execution_groups(_work(logical_c), key_resolver=lambda request_id: _key())

    assert isinstance(plan, ExecutionPlan)
    assert all(group.work.declared_logical_c == logical_c for group in plan.groups)
    assert tuple(
        request_id
        for group in plan.groups
        for physical in group.physical_groups
        for request_id in physical.request_ids
    ) == tuple(range(logical_c))
    assert all(
        physical.physical_rows in (1, 2, 4, 8)
        for group in plan.groups
        for physical in group.physical_groups
    )
    assert all(group.execution_path == "registered_masked_or_exact" for group in plan.groups)
    assert plan.logical_rows == logical_c
    assert plan.planner_duration_ns >= 0


def test_execution_planner_never_mixes_exact_compatibility_keys() -> None:
    work = _work(8)
    plan = plan_execution_groups(
        work,
        key_resolver=lambda request_id: _key(context=f"ctx:{request_id % 2}"),
    )

    assert len(plan.groups) == 2
    assert plan.groups[0].compatibility_key.context_bucket == "ctx:0"
    assert plan.groups[0].work.request_ids == (0, 2, 4, 6)
    assert plan.groups[1].compatibility_key.context_bucket == "ctx:1"
    assert plan.groups[1].work.request_ids == (1, 3, 5, 7)
    assert all(
        len({request_id % 2 for request_id in group.work.request_ids}) == 1
        for group in plan.groups
    )


def test_sparse_group_uses_honest_masked_compact_or_serial_labels() -> None:
    sparse = _work(2, slots=(0, 7))
    masked = plan_execution_groups(
        sparse,
        key_resolver=lambda request_id: _key(masked=True),
    )
    assert masked.groups[0].execution_path == "registered_masked_or_exact"
    assert masked.groups[0].physical_groups[0].physical_rows == 8

    compact = plan_execution_groups(
        sparse,
        key_resolver=lambda request_id: _key(masked=False, compact=True),
    )
    assert compact.groups[0].execution_path == "registered_dense_compaction"
    assert compact.groups[0].physical_groups[0].physical_rows == 2
    assert compact.groups[0].physical_groups[0].dense_execution_rows is True

    serial = plan_execution_groups(
        sparse,
        key_resolver=lambda request_id: _key(masked=False, compact=False),
    )
    assert serial.groups[0].execution_path == "serial_c1_fallback"
    assert [group.physical_rows for group in serial.groups[0].physical_groups] == [1, 1]


def test_token_budget_slo_derives_bounded_round_work() -> None:
    policy = TokenBudgetSLO.from_targets(
        ttft_target_ms=80.0,
        itl_target_ms=20.0,
        measured_prefill_tokens_per_ms=16.0,
        measured_decode_rows_per_ms=2.0,
        max_prefill_chunk_tokens=256,
        resident_capacity=32,
    )
    assert policy.prefill_token_budget == 256
    assert policy.decode_row_budget == 32
    assert policy.ttft_target_ms == 80.0
    assert policy.itl_target_ms == 20.0


@dataclass
class _BudgetRunner:
    capacity: int
    supports_prefill_decode_same_round = True
    supports_multiple_prefill_quanta_per_round = True

    def __post_init__(self) -> None:
        self.prefill_order: list[int] = []
        self.decode_widths: list[int] = []

    def prefill_batch(self, work, *, commit: bool):
        assert commit is True
        self.prefill_order.extend(work.request_ids)

    def decode_batch(self, work, *, commit: bool):
        assert commit is True
        self.decode_widths.append(len(work.request_ids))
        return tuple(
            GeneratedToken(request_id, 100 + request_id)
            for request_id in work.request_ids
        )

    def compact_batch(self, moves):
        del moves

    def reclaim(self, completed):
        del completed


def test_token_budget_round_runs_multiple_fair_prefills_then_all_due_decode() -> None:
    runner = _BudgetRunner(capacity=4)
    loop = ResidentEngineLoop(
        runner,
        config=EngineLoopConfig(
            prefill_decode_policy="token_budget",
            max_active_requests=4,
            max_prefill_chunk_tokens=2,
            round_prefill_token_budget=8,
            round_decode_row_budget=4,
        ),
    )
    request_ids = tuple(
        loop.submit((request_id * 10 + 1, request_id * 10 + 2, request_id * 10 + 3, request_id * 10 + 4), max_new_tokens=1)
        for request_id in range(4)
    )

    first = loop.tick()
    assert [event.kind for event in first].count("work") == 4
    assert runner.prefill_order == list(request_ids)
    assert runner.decode_widths == []

    second = loop.tick()
    assert [event.kind for event in second].count("work") == 5
    assert runner.prefill_order == [*request_ids, *request_ids]
    assert runner.decode_widths == [4]
    assert loop.active_count == 0
    assert set(loop.completed) == set(request_ids)
    policy = loop.observability_snapshot()["scheduler_policy"]
    assert policy["rounds"] == 2
    assert policy["round_prefill_tokens"] == 16
    assert policy["round_decode_rows"] == 4


def test_token_budget_legacy_runner_defers_decode_until_next_barrier() -> None:
    runner = _BudgetRunner(capacity=1)
    runner.supports_prefill_decode_same_round = False
    loop = ResidentEngineLoop(
        runner,
        config=EngineLoopConfig(
            prefill_decode_policy="token_budget",
            max_active_requests=1,
            max_prefill_chunk_tokens=128,
            round_prefill_token_budget=128,
            round_decode_row_budget=1,
        ),
    )
    request_id = loop.submit(tuple(range(128)), max_new_tokens=1)
    first = loop.tick()
    assert [event.work_kind for event in first if event.kind == "work"] == [
        WorkKind.PREFILL
    ]
    assert runner.decode_widths == []
    second = loop.tick()
    assert [event.work_kind for event in second if event.kind == "work"] == [
        WorkKind.DECODE
    ]
    assert runner.decode_widths == [1]
    assert request_id in loop.completed


def test_token_budget_legacy_runner_limits_prefill_to_one_transition_per_barrier() -> None:
    runner = _BudgetRunner(capacity=4)
    runner.supports_prefill_decode_same_round = False
    runner.supports_multiple_prefill_quanta_per_round = False
    loop = ResidentEngineLoop(
        runner,
        config=EngineLoopConfig(
            prefill_decode_policy="token_budget",
            max_active_requests=4,
            max_prefill_chunk_tokens=128,
            round_prefill_token_budget=512,
            round_decode_row_budget=4,
        ),
    )
    request_ids = tuple(loop.submit((request_id,), max_new_tokens=1) for request_id in range(4))
    for expected_prefills in range(1, 5):
        loop.tick()
        assert len(runner.prefill_order) == expected_prefills
        assert runner.decode_widths == []
    loop.tick()
    assert runner.decode_widths == [4]
    assert set(loop.completed) == set(request_ids)


def test_token_budget_c32_sparse_retirement_cancel_and_refill_has_no_width_cliff() -> None:
    runner = _BudgetRunner(capacity=32)
    loop = ResidentEngineLoop(
        runner,
        config=EngineLoopConfig(
            prefill_decode_policy="token_budget",
            max_active_requests=32,
            max_prefill_chunk_tokens=1,
            round_prefill_token_budget=32,
            round_decode_row_budget=32,
        ),
    )
    first_wave = tuple(loop.submit((request_id,), max_new_tokens=2) for request_id in range(32))
    loop.tick()
    assert runner.decode_widths == [32]
    for request_id in first_wave[::2]:
        loop.cancel(request_id)
    refill = tuple(loop.submit((1000 + index,), max_new_tokens=1) for index in range(16))
    loop.tick()
    assert loop.active_count == 0
    assert set(loop.completed).issuperset(refill)
    assert runner.decode_widths[-1] == 32
    snapshot = loop.observability_snapshot()
    assert snapshot["physical_bucket"]["capacity"] == 32
    assert snapshot["requests"]["reclaimed_total"] == 48


def test_dense_runner_executes_logical_c17_as_registered_c8_c8_c1_groups() -> None:
    backend = create_dense_bf16_backend(
        page_capacity=40,
        block_size=2,
        backend_fingerprint="dense-bf16:token-budget-c17",
    )
    admission = DenseKVAdmissionManager(backend)

    class PhysicalRunner:
        capacity = 17
        kv_supports_masked_rows = True
        kv_supports_dense_compaction = False
        kv_prefill_supports_dense_compaction = True
        kv_workspace_key = "workspace:c17"

        def __init__(self) -> None:
            self.kv_kernel_bundle_key = backend.spec.kernel_bundle_key
            self.kv_storage_layout_keys = (backend.storage_view().layout_key,)
            self.decode_widths = []

        def kv_execution_context_bucket(self, request_id, work_kind):
            del request_id, work_kind
            return "context:short"

        def prefill_batch_with_kv(self, work, *, kv_batch_view, commit):
            raise AssertionError("registered physical prefill should be used")

        def decode_batch_with_kv(self, work, *, kv_batch_view, commit):
            raise AssertionError("registered physical decode should be used")

        def prefill_physical_group_with_kv(
            self,
            work,
            *,
            physical_group,
            kv_batch_view,
            commit,
        ):
            assert commit is True
            assert set(physical_group.request_ids).issubset(work.request_ids)
            assert kv_batch_view.storage_view is backend.storage_view()

        def decode_physical_group_with_kv(
            self,
            work,
            *,
            physical_group,
            kv_batch_view,
            commit,
        ):
            assert commit is True
            assert set(physical_group.request_ids).issubset(work.request_ids)
            assert kv_batch_view.storage_view is backend.storage_view()
            self.decode_widths.append(physical_group.physical_rows)
            return tuple(
                GeneratedToken(request_id, 200 + request_id)
                for request_id in physical_group.request_ids
            )

        def compact_batch(self, moves):
            del moves

        def reclaim(self, completed):
            del completed

    physical_runner = PhysicalRunner()
    runner = DenseKVResidentRunnerAdapter(physical_runner, admission)
    loop = ResidentEngineLoop(
        runner,
        config=EngineLoopConfig(
            prefill_decode_policy="token_budget",
            max_active_requests=17,
            max_prefill_chunk_tokens=1,
            round_prefill_token_budget=17,
            round_decode_row_budget=17,
        ),
    )
    request_ids = tuple(loop.submit((request_id,), max_new_tokens=1) for request_id in range(17))
    loop.tick()

    assert physical_runner.decode_widths == [8, 8, 1]
    assert set(loop.completed) == set(request_ids)
    planner = loop.observability_snapshot()["resources"]["execution_planner"]
    assert planner["physical_width_counts"][8] == 2
    assert planner["physical_width_counts"][1] == 18
    assert planner["execution_path_counts"] == {
        "registered_dense_compaction": 17,
        "registered_masked_or_exact": 1,
    }
    assert planner["planner_duration_ns"] > 0


def test_execution_planner_host_cost_is_reported_for_c1_c8_c32() -> None:
    timings = {}
    for width in (1, 8, 32):
        start = perf_counter_ns()
        plan = plan_execution_groups(_work(width), key_resolver=lambda request_id: _key())
        wall = perf_counter_ns() - start
        timings[width] = plan.planner_duration_ns
        assert 0 <= plan.planner_duration_ns <= wall
        assert plan.logical_rows == width
    assert set(timings) == {1, 8, 32}
