from __future__ import annotations

from dataclasses import replace

import pytest

from hipengine.kvcache import ClaimLifetime, ResourceClaimSet
from hipengine.speculative import CandidateGraph, TargetFrontier
from tests.test_specdec2_engine_loop import _CycleRunner
from hipengine.generation.engine_loop import ResidentEngineLoop


class _StagedCycleRunner(_CycleRunner):
    def __init__(self, *, fail_target_once: bool = False) -> None:
        super().__init__()
        self.fail_target_once = bool(fail_target_once)
        self.failed = False
        self.stage_order: list[str] = []
        self.active_claims: ResourceClaimSet | None = None
        self.last_complete_claims: ResourceClaimSet | None = None
        self.last_frontier: TargetFrontier | None = None
        self._semantics = ()

    def speculative_capability(self, request_semantics):
        if self.failed:
            return None
        return super().speculative_capability(request_semantics)

    def speculative_component_claims(self, plan):
        self.stage_order.append("claims")
        return {
            "target": ResourceClaimSet.from_mapping(
                f"{plan.operation_id}:target",
                {"fake.target": plan.logical_frontier_rows},
                lifetime=ClaimLifetime.TRANSACTION,
            ),
            "provider": ResourceClaimSet.from_mapping(
                f"{plan.operation_id}:provider",
                {"fake.provider": sum(plan.candidate_counts)},
                lifetime=ClaimLifetime.TRANSACTION,
            ),
            "transient": ResourceClaimSet.from_mapping(
                f"{plan.operation_id}:transient",
                {"fake.results": len(plan.request_ids)},
                lifetime=ClaimLifetime.WORK_ITEM,
            ),
        }

    def reserve_speculative_claims(self, claims):
        self.stage_order.append("reserve")
        assert self.active_claims is None
        self.active_claims = claims
        self.last_complete_claims = claims
        return claims.claim_id

    def release_speculative_claims(self, reservation):
        self.stage_order.append("release")
        assert self.active_claims is not None
        assert reservation == self.active_claims.claim_id
        self.active_claims = None

    def prepare_speculative_requests(self, plan, request_semantics, *, stream=None):
        self.stage_order.append("prepare")
        self._semantics = tuple(request_semantics)

    def propose_speculative_batch(self, plan, request_semantics, *, stream=None):
        self.stage_order.append("propose")
        tokens: list[int] = []
        owners: list[int] = []
        parents: list[int] = []
        depths: list[int] = []
        offsets = [0]
        offset = 0
        for request_id, count in zip(
            plan.request_ids, plan.candidate_counts, strict=True
        ):
            for depth in range(1, count + 1):
                tokens.append(request_id * 100 + depth)
                owners.append(request_id)
                parents.append(-1 if depth == 1 else offset + depth - 2)
                depths.append(depth)
            offset += count
            offsets.append(offset)
        return CandidateGraph(
            provider_key=str(plan.provider_key),
            method_key="mtp2",
            policy_fingerprint="fake-policy:v1",
            cycle_id=plan.cycle_id,
            transaction_id=plan.cycle_id,
            request_ids=plan.request_ids,
            resident_slots=plan.resident_slots,
            root_positions=tuple(row.context_tokens - 1 for row in request_semantics),
            row_offsets=tuple(offsets),
            row_to_request=tuple(owners),
            parent_candidate_rows=tuple(parents),
            draft_depths=tuple(depths),
            active_mask=(True,) * len(tokens),
            candidate_tokens=tuple(tokens),
        )

    def speculative_kv_live_spans_owner(self, plan):
        return "fake-live-spans"

    def execute_target_frontier(self, plan, frontier, complete_claims, *, commit):
        self.stage_order.append("target")
        assert commit
        assert self.active_claims is complete_claims
        self.last_frontier = frontier
        if self.fail_target_once:
            self.fail_target_once = False
            self.failed = True
            raise RuntimeError("injected target frontier failure")
        result = super().execute_speculative_cycle(plan, commit=commit)
        transaction = replace(
            result.transaction,
            reserved_claims=complete_claims,
        )
        return replace(result, transaction=transaction)

    def rollback_speculative_cycle(self, plan, candidate_graph, error):
        self.stage_order.append("rollback")
        assert isinstance(error, RuntimeError)


def test_engine_drives_claims_proposal_and_frontier_before_target_commit() -> None:
    runner = _StagedCycleRunner()
    loop = ResidentEngineLoop(runner, capacity=2, prefill_chunk_size=8)
    request_id = loop.submit_speculative(
        [10],
        max_new_tokens=3,
        desired_candidate_count=2,
    )

    loop.poll(max_ticks=2)

    assert runner.stage_order == [
        "claims",
        "reserve",
        "prepare",
        "propose",
        "target",
        "release",
    ]
    assert runner.active_claims is None
    assert runner.last_complete_claims is not None
    assert runner.last_complete_claims.units_by_pool() == {
        "fake.provider": 2,
        "fake.results": 1,
        "fake.target": 3,
    }
    frontier = runner.last_frontier
    assert frontier is not None
    assert frontier.request_ids == (request_id,)
    assert frontier.root_tokens == (10,)
    assert frontier.root_positions == (0,)
    assert frontier.logical_rows == 3
    assert frontier.target_batch is not None
    assert frontier.target_batch.tokens == (10, 1, 2)
    assert loop.completed[request_id].generated_tokens == (1, 2, 8000)


def test_target_failure_rolls_back_releases_and_next_tick_runs_healthy_ar() -> None:
    runner = _StagedCycleRunner(fail_target_once=True)
    loop = ResidentEngineLoop(runner, capacity=1, prefill_chunk_size=8)
    request_id = loop.submit_speculative(
        [10],
        max_new_tokens=2,
        desired_candidate_count=1,
    )
    loop.poll(max_ticks=1)  # admission and prefill

    with pytest.raises(RuntimeError, match="injected target frontier failure"):
        loop.poll(max_ticks=1)

    assert runner.stage_order[-2:] == ["rollback", "release"]
    assert runner.active_claims is None
    assert loop.active_count == 1
    assert loop.scheduler.active_batch.requests[request_id].generated_tokens == ()

    events = loop.poll(max_ticks=2)

    assert runner.decodes[-1].request_ids == (request_id,)
    assert loop.completed[request_id].generated_tokens == (9000, 9001)
    assert [event.request_id for event in events if event.kind == "completed"] == [
        request_id
    ]
