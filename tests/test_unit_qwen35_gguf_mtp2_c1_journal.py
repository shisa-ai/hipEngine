"""Singleton GGUF MTP2 cycles own provider and target undo in one journal.

The C1 route captures the provider backup during ``propose_batch`` and the
prepared target transaction during ``execute_target_frontier``. One
``SpeculativeStateJournal`` then owns both: it restores them in reverse capture
order (target before provider), commits them together once the target published
its commit, and releases each device backup exactly once. These tests pin that
ownership, the exact restore/commit order, the release counts, and the two
properties the migration exists to guarantee -- a retry never repeats a restore
that already succeeded, and a post-commit failure never rolls back published
state.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.generation.qwen35_gguf_mtp2 as mtp2_module
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.generation.qwen35_gguf_mtp2 import (
    Qwen35GGUFMTP2Adapter,
    _MTP2RequestState,
)
from hipengine.kvcache.backend import ResourceClaimSet
from hipengine.speculative import (
    CandidateGraph,
    DraftBatch,
    SpecPlanReason,
    SpecRequestPlan,
    SpecK0Class,
    SpecTransactionMode,
    SpeculativeRequestSemantics,
    TargetAcceptSummary,
    TargetFrontier,
)
from hipengine.speculative.chain import ChainDraftRequest, compile_chain_draft
from hipengine.speculative.interfaces import TargetVerifyBatch
from hipengine.speculative.transaction import SpecCycleStage

_RID = 7
_ROOT_POSITION = 5
_ROOT_TOKEN = 90


class _Recorder:
    """Ordered event log for one cycle's capture/restore/commit/release work."""

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def add(self, name: str, detail: object = None) -> None:
        self.events.append((name, detail))

    def names(self) -> list[str]:
        return [name for name, _detail in self.events]

    def count(self, name: str) -> int:
        return sum(1 for recorded, _detail in self.events if recorded == name)


class _Executor:
    """NextN provider executor double: the three checkpoint calls, in order."""

    def __init__(self, log: _Recorder, *, restore_error: Exception | None = None) -> None:
        self.log = log
        self.restore_error = restore_error
        self.restore_attempts = 0
        self.hidden_size = 8

    def capture_request_checkpoint(self, request_id: int) -> str:
        self.log.add("capture:provider", int(request_id))
        return f"checkpoint-{int(request_id)}"

    def restore_request_checkpoint(self, checkpoint: str) -> None:
        self.restore_attempts += 1
        self.log.add("restore:provider", checkpoint)
        if self.restore_error is not None:
            raise self.restore_error

    def release_request_checkpoint(self, checkpoint: str) -> None:
        self.log.add("release:provider", checkpoint)


class _Target:
    def __init__(self, log: _Recorder) -> None:
        self.log = log
        self.position = _ROOT_POSITION
        # Host/device copies are not part of the checkpoint ownership this
        # module pins, so they stay out of the ordered event log.
        self.runtime = SimpleNamespace(
            memcpy=lambda *args: None,
            device_synchronize=lambda: None,
        )
        self.last_target_hidden = Tensor.from_handle(
            0x1100, (1, 8), DType.BF16, Device("hip", 0)
        )


class _Verifier:
    """``Qwen35GGUFTransactionalVerifier`` double with one open transaction."""

    def __init__(
        self,
        log: _Recorder,
        target: _Target,
        summary: TargetAcceptSummary,
        *,
        prepare_error: Exception | None = None,
    ) -> None:
        self.log = log
        self.target = target
        self.summary = summary
        self.prepare_error = prepare_error
        self.rollback_calls = 0
        self.commit_calls = 0
        self.finish_calls = 0
        self.journal = SimpleNamespace(
            hidden_rows_tensor=lambda rows: self.log.add("hidden_rows", rows)
        )

    def graph_bucket(self, key, batch):
        return ("bucket", key)

    def device_proposal_ready(self, budget, *, remaining_decode):
        return False

    def prepare(self, batch, **kwargs):
        self.log.add("prepare:target", int(batch.rows))
        if self.prepare_error is not None:
            raise self.prepare_error
        return SimpleNamespace(
            batch=batch,
            summary=self.summary,
            target_logits=SimpleNamespace(nbytes=0),
            native_device_accept_commit=False,
            native_graph_submitted=False,
            device_state_commit_buffers=None,
        )

    def adopt_accept_summary(self, prepared, summary):
        prepared.summary = summary
        return prepared

    def adopt_terminal_summary(self, prepared, summary):
        prepared.summary = summary
        return prepared

    def commit(self, prepared, plan):
        self.commit_calls += 1
        self.log.add("commit:target", plan.commit_positions[0])
        self.target.position = int(plan.commit_positions[0]) + 1

    def finish(self, prepared):
        self.finish_calls += 1
        self.log.add("finish:target")

    def rollback(self, prepared):
        self.rollback_calls += 1
        self.log.add("rollback:target", int(prepared.batch.rows))

    def close(self):
        self.log.add("close:verifier")


def _owner(target: _Target) -> SimpleNamespace:
    return SimpleNamespace(
        generator=SimpleNamespace(
            backend="hip_gfx1100",
            execution_profile="production",
            tokenizer=SimpleNamespace(eos_token_id=None),
        ),
        capacity=1,
        _shared_runner=None,
        _flush_row_owner=lambda owned: None,
        _release_mtp_draft_runner=lambda *args: None,
        _row=lambda rid: SimpleNamespace(
            request_id=int(rid),
            native_greedy=True,
            first_token_emitted=True,
            lease=SimpleNamespace(session=target),
            slot=SimpleNamespace(),
        ),
    )


def _batch() -> TargetVerifyBatch:
    draft = compile_chain_draft(
        [
            ChainDraftRequest(
                request_id=_RID,
                root_position=_ROOT_POSITION,
                candidate_tokens=(101, 102, 103),
            )
        ],
        candidate_budget=3,
    )
    return TargetVerifyBatch.from_draft(
        draft,
        root_tokens=(_ROOT_TOKEN,),
        root_positions=(_ROOT_POSITION,),
    )


def _summary(batch: TargetVerifyBatch, *, accepted: int = 2) -> TargetAcceptSummary:
    accepted_tokens = tuple(int(token) for token in batch.tokens[1 : 1 + accepted])
    root = int(batch.root_rows[0])
    return TargetAcceptSummary(
        request_ids=(_RID,),
        accepted_counts=(accepted,),
        accepted_tokens=(accepted_tokens,),
        commit_rows=(root + accepted,),
        commit_tokens=(accepted_tokens[-1] if accepted_tokens else _ROOT_TOKEN,),
        commit_positions=(_ROOT_POSITION + accepted,),
        full_accept=(False,),
        next_tokens=(104,),
        candidate_counts=(3,),
    )


def _plan() -> SpecRequestPlan:
    return SpecRequestPlan(
        operation_id="specdec2-cycle:journal",
        cycle_id=1,
        request_ids=(_RID,),
        resident_slots=(0,),
        candidate_counts=(3,),
        reasons=(SpecPlanReason.SPECULATIVE_QUALIFIED,),
        k0_classes=(SpecK0Class.NOT_K0,),
        mode="verify_chain",
        capability_key="gguf_mtp2_c1:hip_gfx1100:gguf_q4_k_m:native:3",
        provider_key="qwen_nextn_dense",
        target_transaction_mode=SpecTransactionMode.RESERVED_APPEND,
        provider_transaction_mode=SpecTransactionMode.RESERVED_APPEND,
        proposal_widths=(1,),
        target_row_decomposition=(4,),
        context_bucket_size=64,
        execution_route="graph",
    )


def _frontier(
    plan: SpecRequestPlan,
    *,
    token_ids: object | None = None,
    with_batch: bool = True,
) -> TargetFrontier:
    graph = CandidateGraph(
        provider_key="qwen_nextn_dense",
        method_key="mtp2",
        policy_fingerprint="journal-test:v1",
        cycle_id=plan.cycle_id,
        transaction_id=1,
        request_ids=plan.request_ids,
        resident_slots=plan.resident_slots,
        root_positions=(_ROOT_POSITION,),
        row_offsets=(0, 3),
        row_to_request=(_RID, _RID, _RID),
        parent_candidate_rows=(-1, 0, 1),
        draft_depths=(1, 2, 3),
        active_mask=(True, True, True),
        candidate_tokens=(101, 102, 103),
        token_ids=token_ids,
    )
    return TargetFrontier(
        operation_id=plan.operation_id,
        cycle_id=plan.cycle_id,
        request_ids=plan.request_ids,
        resident_slots=plan.resident_slots,
        root_tokens=(_ROOT_TOKEN,),
        root_positions=(_ROOT_POSITION,),
        physical_row_decomposition=(4,),
        transaction_mode=plan.target_transaction_mode,
        kv_storage_view_key="bf16",
        kv_live_spans_owner="journal-live-spans",
        execution_route="eager",
        candidate_graph=graph,
        target_batch=_batch() if with_batch else None,
        provider_transaction_id=1,
    )


def _row(target: _Target) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=_RID,
        request=SimpleNamespace(
            eos_token_id=None,
            ignore_eos=False,
            max_tokens=64,
            min_tokens=0,
            stop_token_ids=(),
            stop_token_sequences=(),
        ),
        native_greedy=True,
        native_sampler=False,
        lease=SimpleNamespace(session=target),
        slot=SimpleNamespace(
            generated_ids=[_ROOT_TOKEN],
            prev_token=_ROOT_TOKEN,
            seq_position=_ROOT_POSITION,
            native_decode_steps=0,
            done=False,
        ),
        samples=[],
        mtp2_cycles=0,
        mtp2_candidate_counts=[],
        mtp2_accepted_counts=[],
        mtp2_proposal_ms=0.0,
        mtp2_target_ms=0.0,
        mtp2_provider_update_ms=0.0,
        mtp2_target_pass_ms=[],
        mtp2_target_pass_start_ns=[],
        mtp2_target_pass_end_ns=[],
        mtp2_cycle_profile_start_ns=[],
        mtp2_cycle_profile_end_ns=[],
        mtp2_provider_update_pass_ms=[],
        mtp2_execution_routes=[],
        mtp2_device_accept_calls=0,
    )


def _draft() -> DraftBatch:
    return DraftBatch(
        request_ids=(_RID,),
        candidate_tokens=(101, 102, 103),
        parent_positions=(_ROOT_POSITION, _ROOT_POSITION + 1, _ROOT_POSITION + 2),
        draft_depths=(1, 2, 3),
        row_to_request=(_RID, _RID, _RID),
        tree_parents=(-1, 0, 1),
        active_mask=(True, True, True),
    )


class _Harness:
    """One adapter with a live C1 proposal already journaled."""

    def __init__(
        self,
        monkeypatch,
        *,
        prepare_error: Exception | None = None,
        restore_error: Exception | None = None,
    ) -> None:
        self.log = _Recorder()
        self.target = _Target(self.log)
        self.row = _row(self.target)
        self.batch = _batch()
        self.summary = _summary(self.batch)
        self.executor = _Executor(self.log, restore_error=restore_error)
        self.verifier = _Verifier(
            self.log, self.target, self.summary, prepare_error=prepare_error
        )
        monkeypatch.setattr(
            mtp2_module, "free", lambda buffer, runtime=None: self.log.add("free")
        )
        self.adapter = Qwen35GGUFMTP2Adapter(
            _owner(self.target),
            enabled=True,
            target_verify_mode="native",
            candidate_budget=3,
        )
        self.adapter.owner._row = lambda rid: self.row
        self.adapter._intents = {_RID: 3}
        self.adapter._static_eligibility_by_request = {}
        self.adapter._prompt_hidden_rows = {}
        self.adapter._states = {}
        self.adapter._disabled_requests = set()
        self.adapter._active_claims = None
        self.adapter._cycle_hidden_tensors = lambda runtime, hidden_size: (
            Tensor.from_handle(0x2000, (1, 8), DType.BF16, Device("hip", 0)),
            Tensor.from_handle(0x3000, (1, 8), DType.BF16, Device("hip", 0)),
        )
        self.provider = SimpleNamespace(
            executor=self.executor,
            release_request=lambda rid: self.log.add("release_request", int(rid)),
            propose=self._propose,
        )
        self.state = _MTP2RequestState(
            request_id=_RID,
            provider=self.provider,
            provider_pool_key=None,
            provider_group_key=(_RID,),
            verifier=self.verifier,
            root_hidden_buffer=SimpleNamespace(ptr=1),
        )
        self.adapter._states[_RID] = self.state
        self.plan = _plan()
        self.claims = ResourceClaimSet(claim_id=self.plan.operation_id)
        self.adapter._active_claims = self.claims

    def _propose(self, context, **kwargs):
        self.log.add("propose:provider", kwargs.get("candidate_budget"))
        return _draft()

    def propose(self) -> object:
        return self.adapter.propose_batch(
            self.plan,
            (SpeculativeRequestSemantics(_RID, "greedy", "verify_chain", 6, 8),),
        )

    def execute(
        self, *, cancelled_after: int | None = None, frontier: TargetFrontier | None = None
    ) -> object:
        """Run one target frontier, cancelling after ``cancelled_after`` probes.

        ``None`` never cancels; ``0`` cancels before the target is prepared, and
        ``n`` cancels at the probe that follows the n-th phase.
        """

        probes = iter([()] * (cancelled_after or 0) + [(_RID,)])

        def probe():
            if cancelled_after is None:
                return ()
            return next(probes, (_RID,))

        return self.adapter.execute_target_frontier(
            self.plan,
            _frontier(self.plan) if frontier is None else frontier,
            self.claims,
            commit=True,
            cancelled_request_ids=probe,
        )


def test_proposal_capture_failure_leaves_no_journal_and_releases_nothing(
    monkeypatch,
) -> None:
    harness = _Harness(monkeypatch)

    def fail(request_id):
        harness.log.add("capture:provider", int(request_id))
        raise RuntimeError("checkpoint capture failed")

    harness.executor.capture_request_checkpoint = fail

    with pytest.raises(RuntimeError, match="checkpoint capture failed"):
        harness.propose()

    # A capture that never completed owns nothing: the journal closes empty and
    # no device backup exists to restore or release.
    assert harness.log.names() == ["capture:provider"]
    assert harness.state.state_journal is None
    assert harness.state.proposal_checkpoint is None


def test_proposal_failure_after_capture_rolls_the_backup_back_once(monkeypatch) -> None:
    harness = _Harness(monkeypatch)

    def fail(context, **kwargs):
        harness.log.add("propose:provider", "failed")
        raise RuntimeError("proposal failed")

    harness.provider.propose = fail

    with pytest.raises(RuntimeError, match="proposal failed"):
        harness.propose()

    assert harness.log.names() == [
        "capture:provider",
        "propose:provider",
        "restore:provider",
        "release:provider",
    ]
    assert harness.state.state_journal is None
    assert harness.state.proposal_checkpoint is None


def test_cancel_before_target_prepare_restores_only_the_provider(monkeypatch) -> None:
    harness = _Harness(monkeypatch)
    harness.propose()
    harness.log.events.clear()

    result = harness.execute(cancelled_after=0)

    assert result.stage is SpecCycleStage.CANCELLED
    assert result.cancelled_request_ids == (_RID,)
    assert result.transaction.rolled_back is True
    assert result.transaction.target_open is False
    assert result.transaction.provider_open is False
    # The target transaction was never opened, so only the provider owner is
    # restored, and its backup is released exactly once.
    assert harness.log.names() == [
        "restore:provider",
        "release:provider",
        "close:verifier",
        "free",
        "release_request",
    ]
    assert harness.verifier.rollback_calls == 0
    assert harness.state.state_journal is None
    assert harness.state.proposal_checkpoint is None


def test_cancel_after_verification_restores_target_then_provider(monkeypatch) -> None:
    harness = _Harness(monkeypatch)
    harness.propose()
    harness.log.events.clear()

    result = harness.execute(cancelled_after=1)

    assert result.stage is SpecCycleStage.CANCELLED
    assert harness.log.names() == [
        "prepare:target",
        "rollback:target",
        "restore:provider",
        "release:provider",
        "close:verifier",
        "free",
        "release_request",
    ]
    assert harness.verifier.commit_calls == 0
    assert harness.verifier.finish_calls == 0
    assert harness.row.slot.generated_ids == [_ROOT_TOKEN]
    assert harness.state.state_journal is None


def test_cancel_after_provider_repair_restores_both_and_publishes_nothing(
    monkeypatch,
) -> None:
    harness = _Harness(monkeypatch)
    harness.propose()
    harness.adapter._repair_provider_state = (
        lambda state, accepted_count, candidate_count: harness.log.add(
            "repair:provider", int(accepted_count)
        )
    )
    harness.state.proposal_device = None
    harness.log.events.clear()

    result = harness.execute(cancelled_after=2)

    assert result.stage is SpecCycleStage.CANCELLED
    assert harness.log.names() == [
        "prepare:target",
        "repair:provider",
        "rollback:target",
        "restore:provider",
        "release:provider",
        "close:verifier",
        "free",
        "release_request",
    ]
    # The provider repair already ran, so the target and provider restores both
    # have to undo it; nothing may be published from a cancelled cycle.
    assert harness.verifier.commit_calls == 0
    assert harness.row.slot.generated_ids == [_ROOT_TOKEN]
    assert harness.row.mtp2_candidate_counts == []


def test_target_prepare_failure_restores_the_provider_only(monkeypatch) -> None:
    harness = _Harness(monkeypatch, prepare_error=RuntimeError("prepare failed"))
    harness.propose()
    harness.log.events.clear()

    with pytest.raises(RuntimeError, match="prepare failed"):
        harness.execute()

    assert harness.log.names() == ["prepare:target", "restore:provider", "release:provider"]
    assert harness.verifier.rollback_calls == 0
    assert harness.state.state_journal is None
    assert harness.state.proposal_checkpoint is None


def test_containment_does_not_restore_an_already_rolled_back_cycle(monkeypatch) -> None:
    harness = _Harness(monkeypatch, prepare_error=RuntimeError("prepare failed"))
    harness.propose()
    with pytest.raises(RuntimeError, match="prepare failed"):
        harness.execute()
    harness.log.events.clear()

    harness.adapter.rollback_cycle(harness.plan, None, RuntimeError("prepare failed"))

    # The journal already restored and released; the containment hook must not
    # repeat either call on the same backup.
    assert harness.log.names() == ["close:verifier", "free", "release_request"]
    assert harness.executor.restore_attempts == 1
    assert harness.log.count("release:provider") == 0


def test_restore_failure_keeps_the_journal_and_backup_reachable_for_retry(
    monkeypatch,
) -> None:
    harness = _Harness(monkeypatch, restore_error=RuntimeError("restore failed"))
    harness.propose()
    harness.log.events.clear()

    with pytest.raises(ExceptionGroup, match="rollback failed"):
        harness.execute()

    # The target restore succeeded and must not run again. The provider backup
    # stays allocated and the journal stays on the request, so the retry below
    # can reach it.
    assert harness.log.names() == ["prepare:target", "rollback:target", "restore:provider"]
    assert harness.verifier.rollback_calls == 1
    assert harness.state.state_journal is not None
    assert harness.state.proposal_checkpoint == f"checkpoint-{_RID}"
    assert harness.log.count("release:provider") == 0

    harness.executor.restore_error = None
    harness.log.events.clear()
    harness.adapter._rollback_state_journal(harness.state)

    assert harness.log.names() == ["restore:provider", "release:provider"]
    assert harness.verifier.rollback_calls == 1
    assert harness.state.state_journal is None
    assert harness.state.proposal_checkpoint is None


def test_commit_releases_the_backup_once_after_the_target_published(monkeypatch) -> None:
    harness = _Harness(monkeypatch)
    harness.propose()
    harness.adapter._repair_provider_state = (
        lambda state, accepted_count, candidate_count: harness.log.add(
            "repair:provider", int(accepted_count)
        )
    )
    harness.state.proposal_device = None
    harness.log.events.clear()

    result = harness.execute()

    assert result.stage is SpecCycleStage.COMMITTED
    assert result.transaction.committed is True
    assert harness.log.names() == [
        "prepare:target",
        "repair:provider",
        "commit:target",
        "finish:target",
        "release:provider",
    ]
    assert harness.log.count("release:provider") == 1
    assert harness.log.count("restore:provider") == 0
    assert harness.verifier.rollback_calls == 0
    assert harness.state.state_journal is None
    assert harness.state.proposal_checkpoint is None
    assert harness.row.slot.generated_ids == [_ROOT_TOKEN, 101, 102, 104]


def test_post_commit_failure_cannot_roll_back_published_state(monkeypatch) -> None:
    harness = _Harness(monkeypatch)
    harness.propose()
    harness.adapter._repair_provider_state = (
        lambda state, accepted_count, candidate_count: None
    )
    harness.state.proposal_device = None
    # A committed cycle that produces no visible token fails after the target
    # published its commit; the undo must already be gone.
    harness.verifier.summary = TargetAcceptSummary(
        request_ids=(_RID,),
        accepted_counts=(0,),
        accepted_tokens=((),),
        commit_rows=(int(harness.batch.root_rows[0]),),
        commit_tokens=(_ROOT_TOKEN,),
        commit_positions=(_ROOT_POSITION,),
        full_accept=(False,),
        next_tokens=(None,),
        candidate_counts=(3,),
    )
    harness.log.events.clear()

    with pytest.raises(RuntimeError, match="no visible token"):
        harness.execute()

    assert harness.log.names() == [
        "prepare:target",
        "commit:target",
        "finish:target",
        "release:provider",
    ]
    assert harness.verifier.rollback_calls == 0
    assert harness.log.count("restore:provider") == 0
    assert harness.state.state_journal is None
    assert harness.state.proposal_checkpoint is None


def test_release_failure_after_commit_keeps_the_backup_reachable(monkeypatch) -> None:
    harness = _Harness(monkeypatch)
    harness.propose()
    harness.adapter._repair_provider_state = (
        lambda state, accepted_count, candidate_count: None
    )
    harness.state.proposal_device = None
    releases = []

    def fail_release(checkpoint):
        releases.append(checkpoint)
        harness.log.add("release:provider", checkpoint)
        if len(releases) == 1:
            raise RuntimeError("release failed")

    harness.executor.release_request_checkpoint = fail_release
    harness.log.events.clear()

    with pytest.raises(RuntimeError, match="release failed"):
        harness.execute()

    # The commit is published and must not be rolled back, and the backup is
    # still on the request so teardown can free it.
    assert harness.log.names() == [
        "prepare:target",
        "commit:target",
        "finish:target",
        "release:provider",
    ]
    assert harness.verifier.rollback_calls == 0
    assert harness.state.state_journal is None
    # Retired, not discarded: the restore is forbidden from now on, and the
    # backup stays reachable so containment can retry its release.
    assert harness.state.proposal_checkpoint == f"checkpoint-{_RID}"
    assert harness.state.proposal_checkpoint_restorable is False
    assert releases == [f"checkpoint-{_RID}"]


def test_teardown_rolls_back_a_cycle_whose_undo_was_never_applied(monkeypatch) -> None:
    harness = _Harness(monkeypatch)
    harness.propose()
    harness.log.events.clear()

    harness.adapter._drop_request(_RID, disable=True)

    assert harness.log.names() == [
        "restore:provider",
        "release:provider",
        "close:verifier",
        "free",
        "release_request",
    ]
    assert harness.state.state_journal is None
    assert harness.state.proposal_checkpoint is None


def test_a_second_cycle_refuses_a_journal_the_first_never_closed(monkeypatch) -> None:
    harness = _Harness(monkeypatch)
    harness.propose()
    harness.log.events.clear()

    with pytest.raises(RuntimeError, match="still owns a state journal"):
        harness.propose()

    # The refusal happens before any capture, so it cannot strand a backup.
    assert harness.log.count("capture:provider") == 0
    assert harness.log.count("release:provider") == 0
    assert harness.state.state_journal is not None


def test_packed_routes_keep_their_explicit_restore_sites(monkeypatch) -> None:
    harness = _Harness(monkeypatch)
    # A packed route has no adapter-level target undo, so it opens no journal.
    assert harness.adapter._state_journal_shape((harness.state,)) is True
    harness.state.verifier = None
    assert harness.adapter._state_journal_shape((harness.state,)) is False
    assert harness.adapter._state_journal_shape((harness.state, harness.state)) is False

    harness.state.proposal_checkpoint = "packed-checkpoint"
    harness.state.proposal_checkpoint_restorable = True
    harness.state.state_journal = None
    harness.log.events.clear()
    harness.adapter._restore_provider_checkpoint(harness.state)

    assert harness.log.names() == ["restore:provider", "release:provider"]
    assert harness.state.proposal_checkpoint is None

    # A backup is only restorable once a proposal published it as such. The
    # default direction is deliberate: containment must never restore a backup
    # it was not told is live.
    harness.state.proposal_checkpoint = "unpublished-checkpoint"
    harness.log.events.clear()
    harness.adapter._restore_provider_checkpoint(harness.state)

    assert harness.log.names() == ["release:provider"]
    assert harness.state.proposal_checkpoint is None




@pytest.mark.parametrize(
    ("proposal_device", "token_ids", "error", "match"),
    [
        # No host batch and no device chain to shape one from.
        (None, None, NotImplementedError, "host or device chain"),
        # The graph kept a device proposal that is no longer the staged one.
        (
            SimpleNamespace(result_ptr=0x9999),
            SimpleNamespace(ptr=0x1111, strides=(2,), shape=(3,)),
            RuntimeError,
            "lost its device proposal",
        ),
        # The device chain is intact but the provider cannot shape a batch.
        (
            SimpleNamespace(result_ptr=0x1111),
            SimpleNamespace(ptr=0x1111, strides=(2,), shape=(3,)),
            RuntimeError,
            "omitted its device placeholder",
        ),
    ],
)
def test_unresolvable_target_batch_still_rolls_the_journal_back(
    monkeypatch, proposal_device, token_ids, error, match
) -> None:
    """Every refusal after the journal opened must close it, not strand it."""

    harness = _Harness(monkeypatch)
    harness.propose()
    harness.state.proposal_device = proposal_device
    harness.log.events.clear()

    with pytest.raises(error, match=match):
        harness.execute(frontier=_frontier(harness.plan, token_ids=token_ids, with_batch=False))

    assert harness.log.names() == ["restore:provider", "release:provider"]
    assert harness.state.state_journal is None
    assert harness.state.proposal_checkpoint is None
    assert harness.verifier.rollback_calls == 0


def test_containment_after_commit_release_failure_never_restores(monkeypatch) -> None:
    """A committed cycle whose release failed must only retry the release."""

    harness = _Harness(monkeypatch)
    harness.propose()
    harness.adapter._repair_provider_state = (
        lambda state, accepted_count, candidate_count: None
    )
    harness.state.proposal_device = None
    attempts = []

    def fail_release(checkpoint):
        attempts.append(checkpoint)
        harness.log.add("release:provider", checkpoint)
        if len(attempts) == 1:
            raise RuntimeError("release failed")

    harness.executor.release_request_checkpoint = fail_release
    harness.log.events.clear()

    with pytest.raises(RuntimeError, match="release failed"):
        harness.execute()

    harness.log.events.clear()
    harness.adapter.rollback_cycle(harness.plan, None, RuntimeError("release failed"))

    # The target committed, so its undo is gone. Containment may retry the
    # outstanding release but must never restore the retired backup.
    assert harness.log.names() == [
        "release:provider",
        "close:verifier",
        "free",
        "release_request",
    ]
    assert harness.log.count("restore:provider") == 0
    assert harness.verifier.rollback_calls == 0
    assert harness.executor.restore_attempts == 0
    assert attempts == [f"checkpoint-{_RID}", f"checkpoint-{_RID}"]


def test_containment_after_rollback_release_failure_never_repeats_the_restore(
    monkeypatch,
) -> None:
    """A rolled-back cycle whose release failed must only retry the release."""

    harness = _Harness(monkeypatch)
    harness.propose()
    attempts = []

    def fail_release(checkpoint):
        attempts.append(checkpoint)
        harness.log.add("release:provider", checkpoint)
        if len(attempts) == 1:
            raise RuntimeError("release failed")

    harness.executor.release_request_checkpoint = fail_release
    harness.log.events.clear()

    with pytest.raises(RuntimeError, match="release failed"):
        harness.execute(cancelled_after=1)

    # The journal restored the target and the provider, then the release failed.
    assert harness.log.names() == [
        "prepare:target",
        "rollback:target",
        "restore:provider",
        "release:provider",
    ]
    assert harness.executor.restore_attempts == 1

    harness.log.events.clear()
    harness.adapter.rollback_cycle(harness.plan, None, RuntimeError("release failed"))

    # Restore already happened and must not repeat; only the release retries.
    assert harness.log.names() == [
        "release:provider",
        "close:verifier",
        "free",
        "release_request",
    ]
    assert harness.executor.restore_attempts == 1
    assert harness.verifier.rollback_calls == 1
    assert attempts == [f"checkpoint-{_RID}", f"checkpoint-{_RID}"]


def test_drop_request_retains_owners_until_the_restore_succeeds(monkeypatch) -> None:
    """A teardown that cannot restore must not free the owners it still needs."""

    harness = _Harness(monkeypatch, restore_error=RuntimeError("restore failed"))
    harness.propose()
    with pytest.raises(ExceptionGroup, match="rollback failed"):
        harness.execute()
    harness.log.events.clear()

    with pytest.raises(ExceptionGroup, match="rollback failed"):
        harness.adapter.release_request(_RID)

    # Nothing was freed or closed: the journal, the backup, the target and the
    # provider are all still reachable for a retry.
    assert harness.log.names() == ["restore:provider"]
    assert harness.state.state_journal is not None
    assert harness.state.proposal_checkpoint == f"checkpoint-{_RID}"
    assert harness.adapter._states[_RID] is harness.state
    assert harness.log.count("free") == 0
    assert harness.log.count("close:verifier") == 0
    assert harness.log.count("release_request") == 0

    harness.executor.restore_error = None
    harness.log.events.clear()
    harness.adapter.release_request(_RID)

    assert harness.log.names() == [
        "restore:provider",
        "release:provider",
        "close:verifier",
        "free",
        "release_request",
    ]
    assert harness.state.state_journal is None
    assert harness.state.proposal_checkpoint is None
    assert _RID not in harness.adapter._states


def test_drop_request_retains_the_backup_when_its_release_fails(monkeypatch) -> None:
    """A failed backup release is retryable, so its pointer must survive."""

    harness = _Harness(monkeypatch)
    harness.propose()
    attempts = []

    def fail_release(checkpoint):
        attempts.append(checkpoint)
        harness.log.add("release:provider", checkpoint)
        if len(attempts) == 1:
            raise RuntimeError("release failed")

    harness.executor.release_request_checkpoint = fail_release
    harness.log.events.clear()

    with pytest.raises(RuntimeError, match="release failed"):
        harness.adapter.release_request(_RID)

    assert harness.log.names() == [
        "restore:provider",
        "release:provider",
    ]
    assert harness.state.proposal_checkpoint == f"checkpoint-{_RID}"
    assert harness.adapter._states[_RID] is harness.state
    assert _RID in harness.adapter._disabled_requests

    harness.log.events.clear()
    harness.adapter.release_request(_RID)

    assert harness.log.names() == [
        "release:provider",
        "close:verifier",
        "free",
        "release_request",
    ]
    assert _RID not in harness.adapter._states


def _second_request(
    harness: "_Harness", request_id: int
) -> tuple[_MTP2RequestState, SimpleNamespace]:
    """Register one more request on the harness so teardown can be batched."""

    target = _Target(harness.log)
    row = _row(target)
    rows = {_RID: harness.row, request_id: row}
    harness.adapter.owner._row = lambda rid: rows[int(rid)]
    state = _MTP2RequestState(
        request_id=request_id,
        provider=SimpleNamespace(
            executor=harness.executor,
            release_request=lambda rid: harness.log.add("release_request", int(rid)),
        ),
        provider_pool_key=None,
        provider_group_key=(request_id,),
        verifier=_Verifier(harness.log, target, harness.summary),
        root_hidden_buffer=SimpleNamespace(ptr=2),
    )
    harness.adapter._states[request_id] = state
    return state, row


def test_close_preserves_an_interrupt_and_still_drops_other_requests(monkeypatch) -> None:
    """A BaseException must stay a BaseException, and must not stop the sweep."""

    harness = _Harness(monkeypatch)
    harness.propose()
    _second_request(harness, _RID + 1)

    def interrupted_close():
        harness.log.add("close:verifier", _RID)
        raise KeyboardInterrupt("interrupted")

    harness.verifier.close = interrupted_close
    harness.log.events.clear()

    with pytest.raises(BaseExceptionGroup) as caught:
        harness.adapter.close()

    group = caught.value
    # An ExceptionGroup cannot nest a KeyboardInterrupt, so the sweep has to
    # report through a BaseExceptionGroup instead of losing the interrupt.
    assert isinstance(group, BaseExceptionGroup)
    assert not isinstance(group, ExceptionGroup)
    assert any(isinstance(error, KeyboardInterrupt) for error in group.exceptions)
    # The other request was still disposed rather than stranded behind it.
    assert ("release_request", _RID + 1) in harness.log.events
    assert _RID + 1 not in harness.adapter._states
    # The interrupted request keeps its owners so a later close can retry.
    assert _RID in harness.adapter._states
    assert harness.adapter._states[_RID].state_journal is None
    assert harness.adapter._states[_RID].released_phases == frozenset()


def test_drop_request_retries_after_a_verifier_close_failure(monkeypatch) -> None:
    """A failed close retains the frees it never reached; a retry runs only those."""

    harness = _Harness(monkeypatch)
    harness.propose()
    closes = []

    def fail_close():
        closes.append(1)
        harness.log.add("close:verifier")
        if len(closes) == 1:
            raise RuntimeError("close failed")

    harness.verifier.close = fail_close
    harness.log.events.clear()

    with pytest.raises(RuntimeError, match="close failed"):
        harness.adapter._drop_request(_RID, disable=True)

    assert harness.log.names() == ["restore:provider", "release:provider", "close:verifier"]
    assert _RID in harness.adapter._states
    assert harness.log.count("free") == 0
    assert harness.log.count("release_request") == 0

    harness.log.events.clear()
    harness.adapter._drop_request(_RID, disable=True)

    # The journal and the backup are already released, so only the remaining
    # phases run and no free happens twice.
    assert harness.log.names() == ["close:verifier", "free", "release_request"]
    assert harness.log.count("restore:provider") == 0
    assert harness.log.count("release:provider") == 0
    assert harness.log.count("free") == 1
    assert _RID not in harness.adapter._states


def test_drop_request_retries_after_a_root_buffer_free_failure(monkeypatch) -> None:
    """A failed free retains the provider release and never repeats the close."""

    harness = _Harness(monkeypatch)
    harness.propose()
    frees = []

    def fail_free(buffer, runtime=None):
        frees.append(int(buffer.ptr))
        harness.log.add("free")
        if len(frees) == 1:
            raise RuntimeError("free failed")

    monkeypatch.setattr(mtp2_module, "free", fail_free)
    harness.log.events.clear()

    with pytest.raises(RuntimeError, match="free failed"):
        harness.adapter._drop_request(_RID, disable=True)

    assert harness.log.names() == [
        "restore:provider",
        "release:provider",
        "close:verifier",
        "free",
    ]
    assert _RID in harness.adapter._states
    assert harness.log.count("release_request") == 0

    harness.log.events.clear()
    harness.adapter._drop_request(_RID, disable=True)

    assert harness.log.names() == ["free", "release_request"]
    assert frees == [1, 1]
    assert harness.log.count("close:verifier") == 0
    assert _RID not in harness.adapter._states


def test_drop_request_retains_the_group_until_the_provider_release_finishes(
    monkeypatch,
) -> None:
    """The provider release and its group outlive a failed draft-runner release."""

    harness = _Harness(monkeypatch)
    harness.propose()
    released_runners = []

    def fail_runner(pool_key, provider):
        released_runners.append(pool_key)
        if len(released_runners) == 1:
            raise RuntimeError("runner release failed")

    harness.adapter.generator._release_mtp_draft_runner = fail_runner
    group = SimpleNamespace(
        key=(_RID,),
        provider=harness.provider,
        provider_pool_key=None,
        request_ids={_RID},
    )
    harness.adapter._provider_groups[(_RID,)] = group
    harness.log.events.clear()

    with pytest.raises(RuntimeError, match="runner release failed"):
        harness.adapter._drop_request(_RID, disable=True)

    assert _RID in harness.adapter._states
    # The group stays registered so the runner release can be retried.
    assert harness.adapter._provider_groups[(_RID,)] is group

    harness.log.events.clear()
    harness.adapter._drop_request(_RID, disable=True)

    assert released_runners == [None, None]
    assert (_RID,) not in harness.adapter._provider_groups
    assert harness.log.count("close:verifier") == 0
    assert harness.log.count("free") == 0
    assert harness.log.count("release_request") == 0
    assert _RID not in harness.adapter._states
