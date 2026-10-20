"""Peer-induced speculative ineligibility in the resident engine loop.

A due group resolves exactly one composed provider/target capability. An adapter
that requires every row of the group to be ready -- ``Qwen4ExpServingMixed.
_mixed_rows_ready`` requires every row imported into the shared batch -- answers
``None`` for a group that mixes a ready explicit-MTP row with an ordinary AR
peer. Planning that group K0 as a whole took the ready row's speculation down
with the peer for as long as the peer stayed resident, because the automatic
candidate counts of the ineligible peer are still non-zero and the desired-count
subset retry in ``_run_decode`` therefore never fires.

The engine now partitions such a group by individual capability: the capable
rows run one speculative cycle, the rows that cannot be served on their own
decode autoregressively in the same tick, and each partition publishes its own
plan so a refused row keeps its own reason. A group whose rows are *all*
individually capable stays on the shipped route -- one batched autoregressive
decode -- because that refusal is a group resource miss, not peer-induced
ineligibility (``HIPENGINE_SPEC_MTP_SPLIT_REFUSED_GROUPS`` re-opens the
measurement arm; see ``docs/REFACTOR.md``).
"""

from __future__ import annotations

from collections import Counter

import pytest

from hipengine.dispatch import WorkKind
from hipengine.generation.batch_scheduler import GeneratedToken, PerRowSamplingParams
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


class _PeerEligibilityRunner:
    """One provider capability per composed group, resolved from row readiness.

    This is the mixed-adapter shape: the provider serves a group only when every
    row of that group is individually ready, and it refuses the group outright
    otherwise. A row that is not ready is refused on its own too, so the group
    mixes a capable row with an incapable peer.
    """

    def __init__(
        self,
        *,
        max_group_rows: int = 4,
        supported_sampling: tuple[str, ...] = ("greedy", "processed"),
        max_context_tokens: int | None = None,
    ) -> None:
        self.max_group_rows = int(max_group_rows)
        self.supported_sampling = tuple(supported_sampling)
        self.max_context_tokens = max_context_tokens
        self.eligible: set[int] = set()
        # Rows the provider serves only inside a shared target step: each is
        # refused on its own and the group is refused when a non-joint row is
        # mixed in (the mixed-MTP row shape).
        self.joint_rows: set[int] = set()
        self.prefills = []
        self.decodes = []
        self.cycle_plans = []
        self.plans = []
        self.k0_prepares = []
        self.capability_calls = []
        self.ar_commits = []
        self._transaction_id = 0
        self._decode_counts: dict[int, int] = {}

    # ---------------------------------------------------------------- runner ABI

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

    def _row_eligible(self, item) -> bool:
        if int(item.request_id) not in self.eligible:
            return False
        if str(item.sampling_mode) not in self.supported_sampling:
            return False
        if (
            self.max_context_tokens is not None
            and int(item.context_tokens) >= self.max_context_tokens
        ):
            return False
        return True

    def speculative_capability(self, request_semantics):
        semantics = tuple(request_semantics)
        request_ids = tuple(int(item.request_id) for item in semantics)
        self.capability_calls.append(request_ids)
        if len(semantics) > self.max_group_rows:
            return None
        if any(request_id in self.joint_rows for request_id in request_ids):
            if len(request_ids) < 2 or any(
                request_id not in self.joint_rows for request_id in request_ids
            ):
                return None
            return self._capability()
        if not all(self._row_eligible(item) for item in semantics):
            return None
        return self._capability()

    def _capability(self) -> SpeculativeCapability:
        return SpeculativeCapability(
            capability_key="fake:mixed-mtp",
            target_key="fake_target",
            provider_key="fake_nextn",
            method_key="mtp2",
            policy_fingerprint="fake-policy:v1",
            execution_profile="strict",
            kv_backend_key="fake_kv",
            attachment=ProviderAttachment.TARGET_ATTACHED,
            catchup_mode=ProviderCatchupMode.TARGET_OUTPUT,
            supported_modes=("verify_chain",),
            supported_sampling_modes=self.supported_sampling,
            max_requests=max(1, self.max_group_rows),
            max_candidates_per_request=3,
            max_frontier_rows=32,
            proposal_widths=(1,),
            target_row_buckets=tuple(range(1, 33)),
            target_transaction_mode=SpecTransactionMode.PACKED_SCRATCH,
            provider_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
            graph_supported=False,
            eager_supported=True,
            strict_fallback_key="fake_target_ar",
            max_context_tokens=self.max_context_tokens,
        )

    def speculative_claims_fit(self, plan):
        return True

    def note_speculative_plan(self, plan):
        self.plans.append(plan)

    def prepare_speculative_k0(self, plan, request_semantics, *, stream=None):
        self.k0_prepares.append(
            (tuple(plan.request_ids), tuple(int(item.request_id) for item in request_semantics))
        )

    def note_speculative_ar_commit(self, request_id, reason=None):
        self.ar_commits.append((int(request_id), reason))

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


def _loop(runner: _PeerEligibilityRunner, *, capacity: int = 2) -> ResidentEngineLoop:
    return ResidentEngineLoop(
        runner,
        capacity=capacity,
        prefill_chunk_size=8,
        prefill_decode_policy="protect_ttft",
    )


def _decode_phase_work(events) -> list[tuple[WorkKind, tuple[int, ...]]]:
    return [
        (event.work_kind, tuple(event.request_ids))
        for event in events
        if event.kind == "work"
        and event.work_kind in {WorkKind.VERIFY_CHAIN, WorkKind.DECODE}
    ]


@pytest.mark.parametrize("ineligibility", ("not_ready", "sampling", "context"))
def test_capable_row_speculates_beside_an_incapable_peer(ineligibility: str) -> None:
    """A capable row keeps its cycle when its peer cannot be served at all.

    The three ineligibility kinds are the shapes the serving route reaches:
    a peer the shared batch has not imported, a peer whose sampling mode the
    provider does not serve, and a peer past the provider's context window. In
    every case the composed group capability is ``None`` and both rows still
    carry non-zero automatic candidate counts, so the whole group would
    otherwise decode autoregressively.
    """

    runner = _PeerEligibilityRunner(
        supported_sampling=("greedy",) if ineligibility == "sampling" else ("greedy", "processed"),
        max_context_tokens=4 if ineligibility == "context" else None,
    )
    loop = _loop(runner)
    capable = loop.submit_speculative(
        [10],
        max_new_tokens=2,
        desired_candidate_count=1,
    )
    peer = loop.submit_speculative(
        [20, 21, 22, 23, 24, 25] if ineligibility == "context" else [20],
        max_new_tokens=1,
        desired_candidate_count=1,
        sampling=PerRowSamplingParams(temperature=0.7)
        if ineligibility == "sampling"
        else None,
    )
    runner.eligible = {capable}

    events = loop.poll(max_ticks=8)

    # The capable row runs one real cycle on its own.
    assert [plan.request_ids for plan in runner.cycle_plans] == [(capable,)]
    assert runner.cycle_plans[0].candidate_counts == (1,)
    assert runner.cycle_plans[0].reasons == (SpecPlanReason.SPECULATIVE_QUALIFIED,)
    # The refused peer decodes autoregressively in the same tick, and the two
    # physical steps report exactly the rows each one owned.
    assert [work.request_ids for work in runner.decodes] == [(peer,)]
    assert _decode_phase_work(events) == [
        (WorkKind.VERIFY_CHAIN, (capable,)),
        (WorkKind.DECODE, (peer,)),
    ]
    # Every due row ran exactly once and emitted once.
    assert [event.request_id for event in events if event.kind == "completed"] == [
        capable,
        peer,
    ]
    assert loop.completed[capable].generated_tokens == (1, 8000)
    assert loop.completed[peer].generated_tokens == (9000 + peer * 10,)
    # The refused peer keeps its own reason on its own plan, not the capable
    # row's speculative-qualified decision.
    refused_plan = loop.last_speculative_plan
    assert refused_plan.request_ids == (peer,)
    assert refused_plan.reasons == (SpecPlanReason.NO_PROVIDER,)
    assert runner.k0_prepares == [((peer,), (peer,))]
    # The split runs once: the whole group is resolved once, and no resolution
    # repeats more than the single per-partition retry.
    calls = Counter(runner.capability_calls)
    assert calls[(capable, peer)] == 1
    assert max(calls.values()) <= 2
    assert sum(calls.values()) <= 2 * 2 + 2


def test_each_refused_row_keeps_its_own_plan_reason() -> None:
    """Two refused rows of one partition report their own reasons, not the peer's."""

    runner = _PeerEligibilityRunner(max_group_rows=4)
    loop = _loop(runner, capacity=3)
    capable = loop.submit_speculative([10], max_new_tokens=2, desired_candidate_count=1)
    refused_spec = loop.submit_speculative(
        [20],
        max_new_tokens=2,
        desired_candidate_count=1,
    )
    plain = loop.submit([30], max_new_tokens=2)
    runner.eligible = {capable}

    events = loop.poll(max_ticks=8)

    assert [plan.request_ids for plan in runner.cycle_plans] == [(capable,)]
    # One autoregressive step covers both refused rows, once each.
    assert runner.decodes[0].request_ids == (refused_spec, plain)
    assert refused_spec not in {
        request_id
        for plan in runner.cycle_plans
        for request_id in plan.request_ids
    }
    # The refused partition plans itself: one row carried speculative intent and
    # lost its provider, the other never carried any. The desired-count retry in
    # ``_run_decode`` re-plans the intent-carrying row alone on a later tick, so
    # the partition plan is identified by its request ids.
    refused_plans = [
        plan for plan in runner.plans if plan.request_ids == (refused_spec, plain)
    ]
    assert refused_plans
    assert refused_plans[0].candidate_counts == (0, 0)
    assert refused_plans[0].reasons == (
        SpecPlanReason.NO_PROVIDER,
        SpecPlanReason.POLICY_SELECTED_AR,
    )
    assert runner.k0_prepares[0] == (
        (refused_spec, plain),
        (refused_spec, plain),
    )
    # Each refused row is attributed its own reason where it emitted.
    first_reason: dict[int, object] = {}
    for request_id, reason in runner.ar_commits:
        first_reason.setdefault(request_id, reason)
    assert first_reason == {
        refused_spec: SpecPlanReason.NO_PROVIDER,
        plain: SpecPlanReason.POLICY_SELECTED_AR,
    }
    assert sorted(
        event.request_id for event in events if event.kind == "completed"
    ) == sorted((capable, refused_spec, plain))


def test_all_individually_capable_group_stays_one_autoregressive_batch() -> None:
    """A group resource miss must not serialize into one cycle per capable row.

    Both rows resolve on their own, and the provider refuses the pair because it
    serves width one. That is the arm the serving ladder measured slower per
    token, so the shipped route stays: one batched autoregressive decode, with
    the group plan reporting both rows and each row keeping its own reason.
    """

    runner = _PeerEligibilityRunner(max_group_rows=1)
    loop = _loop(runner)
    first = loop.submit_speculative([10], max_new_tokens=1, desired_candidate_count=1)
    second = loop.submit_speculative([20], max_new_tokens=1, desired_candidate_count=1)
    runner.eligible = {first, second}

    events = loop.poll(max_ticks=6)

    assert runner.cycle_plans == []
    assert [work.request_ids for work in runner.decodes] == [(first, second)]
    assert [plan.request_ids for plan in runner.plans] == [(first, second)]
    assert runner.plans[0].reasons == (
        SpecPlanReason.NO_PROVIDER,
        SpecPlanReason.NO_PROVIDER,
    )
    assert [event.request_id for event in events if event.kind == "completed"] == [
        first,
        second,
    ]


def test_peer_drain_and_refill_restores_one_shared_cycle() -> None:
    """Draining the refused peer lets a refill share the survivor's cycle again."""

    runner = _PeerEligibilityRunner()
    loop = _loop(runner)
    survivor = loop.submit_speculative([10], max_new_tokens=8, desired_candidate_count=1)
    peer = loop.submit_speculative([20], max_new_tokens=1, desired_candidate_count=1)
    runner.eligible = {survivor}

    loop.poll(max_ticks=3)

    assert [plan.request_ids for plan in runner.cycle_plans] == [(survivor,)]
    assert [work.request_ids for work in runner.decodes] == [(peer,)]
    assert loop.completed[peer].generated_tokens == (9000 + peer * 10,)
    assert loop.completed.get(survivor) is None

    refill = loop.submit_speculative([30], max_new_tokens=2, desired_candidate_count=1)
    runner.eligible = {survivor, refill}
    loop.poll(max_ticks=1)  # admit and prefill the refill row
    loop.poll(max_ticks=1)

    plans = [plan.request_ids for plan in runner.cycle_plans]
    assert plans[0] == (survivor,)
    assert plans[-1] == (survivor, refill)
    assert runner.cycle_plans[-1].candidate_counts == (1, 1)
    assert loop.completed[refill].generated_tokens
    assert loop.active_count == 1


def test_a_partition_that_resolves_as_a_group_keeps_its_own_cycle() -> None:
    """The refused partition may still resolve on its own; its output is kept.

    A mixed-MTP row is refused alone (its root is emitted before any shared
    target step) and the group is refused when a non-shared row is mixed in, but
    the shared rows resolve together. The partition therefore runs a real cycle,
    and dropping its events would lose their tokens.
    """

    runner = _PeerEligibilityRunner()
    loop = _loop(runner, capacity=3)
    explicit = loop.submit_speculative([10], max_new_tokens=2, desired_candidate_count=1)
    shared_a = loop.submit_speculative([20], max_new_tokens=2, desired_candidate_count=1)
    shared_b = loop.submit_speculative([30], max_new_tokens=2, desired_candidate_count=1)
    runner.eligible = {explicit}
    runner.joint_rows = {shared_a, shared_b}

    events = loop.poll(max_ticks=8)

    assert [plan.request_ids for plan in runner.cycle_plans] == [
        (explicit,),
        (shared_a, shared_b),
    ]
    assert runner.cycle_plans[1].candidate_counts == (1, 1)
    assert runner.decodes == []
    # Every row emits exactly once: no partition is skipped and no row is
    # served by both partitions.
    assert [event.request_id for event in events if event.kind == "completed"] == [
        explicit,
        shared_a,
        shared_b,
    ]
    assert loop.completed[explicit].generated_tokens == (1, 8000)
    assert loop.completed[shared_a].generated_tokens == (101, 8001)
    assert loop.completed[shared_b].generated_tokens == (201, 8002)
    assert loop.active_count == 0


def test_measurement_flag_keeps_the_rejected_singleton_arm(monkeypatch) -> None:
    """The split flag still reproduces one singleton cycle per capable row.

    The flag exists to re-measure the arm the serving ladder rejected, so it must
    keep charge of a refused group instead of silently re-pointing at the
    partition route: two individually capable rows beside an incapable peer run
    two one-row cycles, not one two-row cycle.
    """

    monkeypatch.setenv("HIPENGINE_SPEC_MTP_SPLIT_REFUSED_GROUPS", "1")
    runner = _PeerEligibilityRunner()
    loop = _loop(runner, capacity=3)
    first = loop.submit_speculative([10], max_new_tokens=2, desired_candidate_count=1)
    second = loop.submit_speculative([20], max_new_tokens=2, desired_candidate_count=1)
    peer = loop.submit_speculative([30], max_new_tokens=1, desired_candidate_count=1)
    runner.eligible = {first, second}

    events = loop.poll(max_ticks=8)

    assert [plan.request_ids for plan in runner.cycle_plans] == [(first,), (second,)]
    assert [work.request_ids for work in runner.decodes] == [(peer,)]
    assert [event.request_id for event in events if event.kind == "completed"] == [
        first,
        second,
        peer,
    ]
    assert loop.completed[peer].generated_tokens == (9000 + peer * 10,)


def test_partitions_refused_together_fall_back_to_one_autoregressive_batch() -> None:
    """Capable rows a narrow provider still refuses share the batch AR step.

    Two individually capable rows beside an incapable peer resolve into a
    partition the width-one provider refuses, so neither partition opens a
    cycle. The due item then runs the one autoregressive batch the whole-group
    route would have run, instead of two narrower batches.
    """

    runner = _PeerEligibilityRunner(max_group_rows=1)
    loop = _loop(runner, capacity=3)
    first = loop.submit_speculative([10], max_new_tokens=1, desired_candidate_count=1)
    second = loop.submit_speculative([20], max_new_tokens=1, desired_candidate_count=1)
    peer = loop.submit_speculative([30], max_new_tokens=1, desired_candidate_count=1)
    runner.eligible = {first, second}

    loop.poll(max_ticks=8)

    assert runner.cycle_plans == []
    assert [work.request_ids for work in runner.decodes] == [(first, second, peer)]
    # Each row keeps the reason its own partition plan gave it; the merged batch
    # is the one physical step the work event reports.
    assert [plan.request_ids for plan in runner.plans] == [(first, second), (peer,)]
    assert runner.plans[0].reasons == (
        SpecPlanReason.NO_PROVIDER,
        SpecPlanReason.NO_PROVIDER,
    )
    assert runner.plans[1].reasons == (SpecPlanReason.NO_PROVIDER,)
    assert runner.k0_prepares == [
        ((first, second), (first, second)),
        ((peer,), (peer,)),
    ]
    assert loop.active_count == 0


def test_a_single_refused_row_never_partitions_a_lone_due_row() -> None:
    """A one-row due item has no peer to blame, so it keeps its K0 route."""
    """A one-row due item has no peer to blame, so it keeps its K0 route."""

    runner = _PeerEligibilityRunner()
    loop = _loop(runner, capacity=1)
    request_id = loop.submit_speculative(
        [10],
        max_new_tokens=1,
        desired_candidate_count=1,
    )
    runner.eligible = set()

    loop.poll(max_ticks=4)

    assert runner.cycle_plans == []
    assert [work.request_ids for work in runner.decodes] == [(request_id,)]
    assert [plan.request_ids for plan in runner.plans] == [(request_id,)]
    assert runner.plans[0].reasons == (SpecPlanReason.NO_PROVIDER,)
