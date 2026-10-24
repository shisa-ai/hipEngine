"""One Qwen4 MTP step's rollback, owned by the shared state journal.

``Qwen4ExpServingMTP._advance_mtp`` captures four owners around one iterator
step: the target's device transaction, the draft cursor, the row's sampler state
and its grammar. These cases pin the journal's containment contract on that
boundary -- every published capture is rolled back even when one restore fails,
the iterator is always closed, and the undo is dropped only after the target
commit is published.

The doubles are host objects: no device state, no runner, no model.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.generation.qwen4_exp_serving_mtp import (
    Qwen4ExpServingMTP,
    _RequestRetired,
)
from hipengine.speculative.frontier import (
    SpecK0Class,
    SpecPlanReason,
    SpecRequestPlan,
    SpecTransactionMode,
)


class _Checkpoint:
    """A transaction handle whose identity the cycle record carries."""

    def __init__(self, name: str, position: int = 5) -> None:
        self.name = name
        self.position = int(position)


class _Target:
    """Device target double; every transaction transition is one event."""

    def __init__(
        self,
        events: list,
        *,
        position: int = 5,
        fail_capture: bool = False,
        fail_restore: bool = False,
        fail_commit: bool = False,
    ) -> None:
        self.position = int(position)
        self.events = events
        self.fail_capture = fail_capture
        self.fail_restore = fail_restore
        self.fail_commit = fail_commit

    def begin_device_transaction(self, *, reuse_snapshot: bool = False):
        self.events.append(("target_capture", reuse_snapshot))
        if self.fail_capture:
            raise RuntimeError("target capture failed")
        return _Checkpoint("target", self.position)

    def rollback_device_transaction(self, checkpoint) -> None:
        self.events.append(("target_restore", checkpoint.name))
        if self.fail_restore:
            raise RuntimeError("target restore failed")

    def commit_device_transaction(self, checkpoint) -> None:
        self.events.append(("target_commit", checkpoint.name))
        if self.fail_commit:
            raise RuntimeError("target commit failed")


class _Draft:
    """Draft cursor double; the snapshot is a host position, not an allocation."""

    def __init__(
        self,
        events: list,
        *,
        position: int = 5,
        fail_capture: bool = False,
    ) -> None:
        self.position = int(position)
        self.events = events
        self.fail_capture = fail_capture

    def snapshot(self):
        self.events.append(("draft_capture",))
        if self.fail_capture:
            raise RuntimeError("draft capture failed")
        return _Checkpoint("draft", self.position)

    def restore(self, snapshot) -> None:
        self.events.append(("draft_restore", snapshot.name))


class _HostState:
    """Sampler/grammar double: ``clone`` returns an independent replacement."""

    def __init__(self, events: list, name: str, *, rng_draw_count: int = 0) -> None:
        self.events = events
        self.name = name
        self.rng_draw_count = int(rng_draw_count)
        self.clones: list = []

    def clone(self):
        self.events.append((f"{self.name}_capture",))
        clone = SimpleNamespace(name=self.name, rng_draw_count=self.rng_draw_count)
        self.clones.append(clone)
        return clone


class _Runner(Qwen4ExpServingMTP):
    """The mixin under test with only the request-lifetime probe replaced."""

    def __init__(self, *, retire_on_call: int | None = None) -> None:
        self.retire_on_call = retire_on_call
        self.live_checks = 0

    def _check_request_live(self, row) -> None:
        self.live_checks += 1
        if self.retire_on_call is not None and self.live_checks == self.retire_on_call:
            raise _RequestRetired()


def _plan() -> SpecRequestPlan:
    return SpecRequestPlan(
        operation_id="qwen4exp-journal:1",
        cycle_id=4,
        request_ids=(11,),
        resident_slots=(0,),
        candidate_counts=(2,),
        reasons=(SpecPlanReason.SPECULATIVE_QUALIFIED,),
        k0_classes=(SpecK0Class.NOT_K0,),
        mode="verify_chain",
        capability_key="qwen4exp:mtp",
        provider_key="qwen4exp.mtp",
        target_transaction_mode=SpecTransactionMode.REVERSIBLE_JOURNAL,
        provider_transaction_mode=SpecTransactionMode.RESERVED_APPEND,
        proposal_widths=(1,),
        target_row_decomposition=(3,),
        context_bucket_size=256,
        execution_route="eager",
    )


def _progress(
    *, finished: bool = False, token_ids: tuple[int, ...] = (7, 8)
) -> SimpleNamespace:
    return SimpleNamespace(
        finished=finished,
        token_ids=token_ids,
        token_logprobs=[f"logprob:{token}" for token in token_ids],
        cycle=SimpleNamespace(candidates=(7, 8), accepted=1),
    )


class _Steps:
    """Iterator double that records ``close`` however far it was advanced.

    A real generator only runs its ``finally`` once it has started, so an
    unstarted one would hide a missed close. This double reports the call.
    """

    def __init__(self, events: list, progress) -> None:
        self.events = events
        self.progress = progress
        self.closed = False

    def __next__(self):
        self.events.append(("iterator_next",))
        return self.progress

    def send(self, budget):
        self.events.append(("iterator_send", budget))
        return self.progress

    def close(self) -> None:
        self.events.append(("iterator_closed",))
        self.closed = True


def _row(events: list, target, draft):
    return SimpleNamespace(
        request_id=11,
        row_index=0,
        request=SimpleNamespace(max_tokens=16),
        runner=target,
        draft=draft,
        sampling_state=_HostState(events, "sampling", rng_draw_count=3),
        grammar=_HostState(events, "grammar"),
        mtp_steps=_Steps(events, _progress()),
        generated_ids=[],
        token_logprobs=[],
        mtp_output=None,
    )


def _setup(*, target_restore_fails: bool = False, target_commit_fails: bool = False):
    events: list = []
    target = _Target(
        events,
        fail_restore=target_restore_fails,
        fail_commit=target_commit_fails,
    )
    draft = _Draft(events)
    row = _row(events, target, draft)
    return events, target, draft, row


def test_successful_step_commits_the_target_then_drops_the_journal() -> None:
    events, _target, _draft, row = _setup()
    runner = _Runner()

    progress, transaction = runner._advance_mtp(row, 2, plan=_plan())

    assert events == [
        ("grammar_capture",),
        ("sampling_capture",),
        ("draft_capture",),
        ("target_capture", True),
        ("iterator_next",),
        ("target_commit", "target"),
    ]
    assert progress.token_ids == (7, 8)
    assert transaction.committed and transaction.rolled_back is False
    assert transaction.pre_target_cursors == (5,)
    assert transaction.pre_provider_cursors == (5,)
    assert transaction.pre_rng_counters == (3,)
    assert transaction.target_checkpoint_ids[0].startswith("device:")
    assert transaction.provider_checkpoint_ids[0].startswith("cursor:")
    assert row.generated_ids == [7, 8]
    assert row.token_logprobs == ["logprob:7", "logprob:8"]


def test_successful_step_keeps_the_iterator_open_for_the_next_cycle() -> None:
    events, _target, _draft, row = _setup()
    runner = _Runner()

    _progress_value, transaction = runner._advance_mtp(row, 2, plan=None)

    assert transaction is None
    assert row.mtp_steps.closed is False
    assert ("iterator_closed",) not in events
    row.mtp_steps.close()
    assert events[-1] == ("iterator_closed",)


def test_target_restore_failure_still_restores_every_owner_and_closes_the_iterator() -> None:
    events, target, _draft, row = _setup(target_restore_fails=True)
    sampling, grammar = row.sampling_state, row.grammar
    runner = _Runner()
    # An empty step violates the cycle's output bounds, so the step fails after
    # every owner has been captured.
    row.mtp_steps = _Steps(events, _progress(token_ids=()))

    with pytest.raises(ExceptionGroup, match="rollback") as raised:
        runner._advance_mtp(row, 2, plan=_plan())

    # The failed target restore is reported with its checkpoint name, and the
    # other three owners were still restored afterwards.
    assert str(raised.value.exceptions[0]) == "target restore failed"
    assert raised.value.exceptions[0].__notes__ == [
        "speculative checkpoint: target"
    ]
    assert ("target_restore", "target") in events
    assert ("draft_restore", "draft") in events
    assert row.sampling_state is sampling.clones[0]
    assert row.grammar is grammar.clones[0]
    assert events[-1] == ("iterator_closed",)
    assert isinstance(raised.value.__context__, RuntimeError)
    assert "Flash-Next MTP step violated output/cursor bounds" in str(
        raised.value.__context__
    )


@pytest.mark.parametrize("failing_owner", ["draft_capture", "target_capture"])
def test_capture_failure_rolls_back_the_snapshots_already_published(
    failing_owner: str,
) -> None:
    events, target, draft, row = _setup()
    sampling, grammar = row.sampling_state, row.grammar
    if failing_owner == "draft_capture":
        draft.fail_capture = True
    else:
        target.fail_capture = True
    runner = _Runner()

    with pytest.raises(RuntimeError, match=f"{failing_owner.split('_')[0]} capture failed"):
        runner._advance_mtp(row, 2, plan=_plan())

    assert events[-1] == ("iterator_closed",)
    assert row.grammar is grammar.clones[0]
    assert row.sampling_state is sampling.clones[0]
    if failing_owner == "draft_capture":
        # The target lease was never taken, so nothing device-side is rolled back.
        assert ("target_capture", True) not in events
        assert not [event for event in events if event[0] == "target_restore"]
    else:
        assert ("draft_restore", "draft") in events


def test_commit_failure_keeps_the_undo_and_rolls_back_every_owner() -> None:
    events, _target, _draft, row = _setup(target_commit_fails=True)
    sampling, grammar = row.sampling_state, row.grammar
    runner = _Runner()

    # Every restore succeeds, so the commit failure itself is what surfaces.
    with pytest.raises(RuntimeError, match="target commit failed"):
        runner._advance_mtp(row, 2, plan=_plan())

    assert ("target_commit", "target") in events
    assert ("target_restore", "target") in events
    assert ("draft_restore", "draft") in events
    assert row.sampling_state is sampling.clones[0]
    assert row.grammar is grammar.clones[0]
    assert events[-1] == ("iterator_closed",)
    assert row.generated_ids == []


def test_cancellation_publication_is_unchanged_and_restores_every_owner() -> None:
    events, _target, _draft, row = _setup()
    sampling, grammar = row.sampling_state, row.grammar
    runner = _Runner(retire_on_call=2)

    progress, transaction = runner._advance_mtp(row, 2, plan=_plan())

    assert progress is None
    assert transaction.rolled_back and transaction.committed is False
    assert events == [
        ("grammar_capture",),
        ("sampling_capture",),
        ("draft_capture",),
        ("target_capture", True),
        ("iterator_next",),
        ("target_restore", "target"),
        ("draft_restore", "draft"),
        ("iterator_closed",),
    ]
    assert row.sampling_state is sampling.clones[0]
    assert row.grammar is grammar.clones[0]
    assert row.generated_ids == []


def test_cancellation_never_publishes_a_result_when_a_restore_fails() -> None:
    events, _target, _draft, row = _setup(target_restore_fails=True)
    runner = _Runner(retire_on_call=2)

    with pytest.raises(ExceptionGroup, match="rollback"):
        runner._advance_mtp(row, 2, plan=_plan())

    assert events[-1] == ("iterator_closed",)
