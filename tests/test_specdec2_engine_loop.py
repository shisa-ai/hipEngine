from __future__ import annotations

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
            max_candidates_per_request=2,
            max_frontier_rows=12,
            proposal_widths=(1, 2, 4),
            target_row_buckets=(1, 2, 4, 8, 12),
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
            proposal_widths=plan.proposal_widths,
            target_row_decomposition=plan.target_row_decomposition,
            execution_route=plan.execution_route,
        )
        return SpecCycleResult.committed(transaction, accept, telemetry=telemetry)


class _NoSpecRunner(_CycleRunner):
    def speculative_capability(self, request_semantics):
        return None


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
    assert [event.request_id for event in events if event.kind == "completed"] == [
        ar_id
    ]
    assert loop.completed[ar_id].generated_tokens == (8001,)
    assert loop.active_count == 1


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
