"""PARO MTP candidate depths: planning, chained proposals, and exact repair.

Every case here is CPU-only. The planner cases run the real shared policy
against the adapter's own capability, and the cycle cases run the real adapter
over a proposer double that records the provider calls it receives.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.generation.qwen35_paro_mtp2 as paro_mtp2_module
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.generation.qwen35_paro import Qwen35ParoOneTokenGenerator
from hipengine.generation.qwen35_paro_mtp2 import (
    PARO_MTP2_MAX_CANDIDATE_DEPTH,
    Qwen35ParoMTP2Adapter,
    _ParoMTP2RequestState,
    _proposer_capacity_bucket,
    _proposer_token_requirement,
)
from hipengine.kvcache import ResourceClaimSet
from hipengine.speculative import (
    SpeculativeRequestSemantics,
    plan_speculative_requests,
)
from hipengine.speculative.frontier import SpecTransactionMode, TargetFrontier

REQUEST_ID = 7


class _ChainProposer:
    """MTP proposer double that drafts from host tokens and keeps snapshots."""

    def __init__(self, *, token: int, drafted: tuple[int, ...] = (), position: int = 127):
        self.position = int(position)
        self.cache_len = self.position + 1
        self.current = SimpleNamespace(token=int(token))
        self.final_hidden_valid = True
        self.result_valid = True
        self.closed = False
        self.calls: list[tuple] = []
        self._drafted = list(drafted)
        self._snapshots: dict[int, tuple] = {}

    # -- snapshot/restore --------------------------------------------------

    def save_state(self, slot: int, *, stream: int = 0) -> tuple:
        snapshot = ("snapshot", int(slot), self.position, self.cache_len)
        self._snapshots[int(slot)] = snapshot
        self.calls.append(("save", int(slot), self.position))
        return snapshot

    def restore_state(self, snapshot: tuple, *, stream: int = 0) -> None:
        self.calls.append(("restore", snapshot[1]))
        self.position = int(snapshot[2])
        self.cache_len = int(snapshot[3])
        self.result_valid = False

    # -- advances ----------------------------------------------------------

    def advance_with_previous_hidden(self, **kwargs) -> SimpleNamespace:
        self.calls.append(
            (
                "previous",
                int(kwargs["input_token"]),
                int(kwargs["position"]),
                bool(kwargs["need_result"]),
                # The repair advance leaves ``read_token_id`` at its default
                # because it consumes no result; drafting always sets it.
                bool(kwargs.get("read_token_id", True)),
            )
        )
        self.position = int(kwargs["position"])
        self.cache_len = self.position + 1
        self.final_hidden_valid = True
        if kwargs["need_result"]:
            self.result_valid = True
            self.current = SimpleNamespace(token=self._drafted.pop(0))
        return self.current

    def advance_with_target_hidden(self, **kwargs) -> SimpleNamespace:
        self.calls.append(
            (
                "target",
                int(kwargs["input_token"]),
                int(kwargs["position"]),
                int(kwargs["target_hidden_ptr"]),
                bool(kwargs["read_token_id"]),
            )
        )
        self.position = int(kwargs["position"])
        self.cache_len = self.position + 1
        self.final_hidden_valid = True
        self.result_valid = True
        # A device-only read leaves the host sentinel in place.
        self.current = SimpleNamespace(
            token=int(kwargs["input_token"]) if kwargs["read_token_id"] else -1
        )
        return self.current

    def device_candidate_token_ids(self) -> Tensor:
        self.calls.append(("device_candidates",))
        return Tensor.from_handle(0x7000, (1,), DType.INT32, Device("hip", 0))

    def reset(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _VerifyResult:
    def __init__(self, *, accepted_tokens: tuple[int, ...], bonus: int) -> None:
        self.accepted_tokens = tuple(int(token) for token in accepted_tokens)
        self.accepted_count = len(self.accepted_tokens)
        self.next_token = int(bonus)
        self.target_top1 = (*self.accepted_tokens, int(bonus))
        self.commit_row = self.accepted_count
        self.selected_target_hidden_ptr = 0x5000
        self.graph = {"status": "direct", "replayed": False}


class _Session:
    def __init__(self, *, max_sequence_length: int = 4096, verify=None) -> None:
        self.max_sequence_length = int(max_sequence_length)
        self.max_batch_size = PARO_MTP2_MAX_CANDIDATE_DEPTH + 1
        self.position_arr = [0]
        self.slot_positions: dict[int, int] = {}
        self.copies: list[tuple[int, int, int]] = []
        self.scratch: tuple[int, str, int] | None = None
        self.verify_calls: list[tuple] = []
        self.tokenizer = None
        self.runtime = SimpleNamespace(device_synchronize=lambda: None)
        self._verify = verify

    def copy_slot_state(self, source, destination, *, kv_rows) -> None:
        self.copies.append((int(source), int(destination), int(kv_rows)))

    def _record_slot_position_host(self, position, *, slot) -> None:
        self.slot_positions[int(slot)] = int(position)

    def prepare_specdec2_verify_scratch(
        self, *, rows, chain_attn_mode, max_context_tokens
    ) -> None:
        self.scratch = (int(rows), str(chain_attn_mode), int(max_context_tokens))

    def verify_chain_bulk_and_commit(self, batch, **kwargs):
        self.verify_calls.append((batch, kwargs))
        return self._verify


def _semantics(*, context_tokens: int = 128, remaining_decode: int = 32):
    return SpeculativeRequestSemantics(
        REQUEST_ID,
        "greedy",
        "verify_chain",
        int(context_tokens),
        int(remaining_decode),
    )


def _row(*, max_tokens: int = 8, prompt_ids: tuple[int, ...] = (10, 11)) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=REQUEST_ID,
        model_slot=0,
        native_greedy=True,
        first_token_emitted=True,
        prompt_ids=tuple(prompt_ids),
        request=SimpleNamespace(max_tokens=int(max_tokens)),
        mtp2_cycles=0,
        mtp2_candidate_counts=[],
        mtp2_accepted_counts=[],
        mtp2_execution_routes=[],
        mtp2_candidate_device_handoffs=0,
        mtp2_candidate_d2h_after_target=0,
        mtp2_candidate_budget=0,
        mtp2_prompt_prime_ms=0.0,
        mtp2_provider_open_ms=0.0,
        mtp2_prompt_hidden_buffer=None,
    )


def _owner(*, session=None, row=None, steps=None, profile: str = "production"):
    session = _Session() if session is None else session
    row = _row() if row is None else row
    steps = [] if steps is None else steps
    return SimpleNamespace(
        generator=SimpleNamespace(
            backend="hip_gfx1100",
            execution_profile=profile,
            model_path="/tmp/paro-model",
        ),
        _row=lambda request_id: row,
        _session=session,
        _record_step=lambda selected, step: steps.append((selected, step)),
        _release_mtp2_prompt_capture=lambda selected: None,
    )


def _adapter(*, depth: int = 1, owner=None, proposer=None):
    owner = _owner() if owner is None else owner
    adapter = Qwen35ParoMTP2Adapter(owner, candidate_budget=depth)
    adapter.register_request(REQUEST_ID, depth)
    if proposer is not None:
        adapter._states[REQUEST_ID] = _ParoMTP2RequestState(REQUEST_ID, proposer)
    return adapter


def _plan(adapter, *, desired: int, context_tokens: int = 128, remaining_decode: int = 32):
    """Plan through the real shared policy against the adapter's capability."""

    semantics = (
        _semantics(
            context_tokens=context_tokens,
            remaining_decode=remaining_decode,
        ),
    )
    capability = adapter.capability(semantics)
    assert capability is not None
    return plan_speculative_requests(
        capability,
        semantics,
        resident_slots=(0,),
        desired_candidate_counts=(desired,),
        operation_id="op",
        cycle_id=1,
        context_bucket_size=256,
        graph_available=False,
    )


def _complete_claims(count: int) -> ResourceClaimSet:
    """Return the cycle's merged reservation, keyed by the plan's operation."""

    return ResourceClaimSet.from_mapping(
        "op", {"paro_mtp2.target_rows": int(count) + 1}
    )


def _frontier(adapter, graph):
    return TargetFrontier.from_candidate_graph(
        operation_id="op",
        candidate_graph=graph,
        root_tokens=(100,),
        physical_row_decomposition=(graph.candidate_rows + 1,),
        transaction_mode=SpecTransactionMode.PACKED_SCRATCH,
        kv_storage_view_key="paged_bf16_kv_live_spans",
        kv_live_spans_owner=adapter.kv_live_spans_owner(
            SimpleNamespace(operation_id="op")
        ),
        execution_route="eager",
    )


# -- capability ------------------------------------------------------------


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_capability_declares_the_registered_candidate_depth(depth) -> None:
    adapter = _adapter(depth=depth)
    adapter._states[REQUEST_ID] = _ParoMTP2RequestState(
        REQUEST_ID, _ChainProposer(token=200)
    )

    capability = adapter.capability((_semantics(),))

    assert capability is not None
    assert capability.max_requests == 1
    assert capability.max_candidates_per_request == depth
    assert capability.max_frontier_rows == depth + 1
    assert capability.target_row_buckets == tuple(range(2, depth + 2))
    assert capability.proposal_widths == (1,)
    # The graph route stays declared; the runner, not the adapter, selects eager.
    assert capability.graph_supported is True
    assert capability.eager_supported is True
    assert f"paro_mtp2_c{depth}" in capability.capability_key
    assert f"paro-mtp2:c{depth}:k1" in capability.policy_fingerprint


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_capability_context_bound_is_the_target_capacity(depth) -> None:
    adapter = _adapter(depth=depth)
    adapter._states[REQUEST_ID] = _ParoMTP2RequestState(
        REQUEST_ID, _ChainProposer(token=200)
    )

    capability = adapter.capability((_semantics(),))

    assert capability is not None
    # The shared planner reserves the root itself with its own ``- 1``, so the
    # capability publishes the target's capacity and nothing depth-dependent.
    assert capability.max_context_tokens == 4096


def test_a_shorter_session_lowers_the_published_capacity() -> None:
    adapter = _adapter(depth=3, owner=_owner(session=_Session(max_sequence_length=1024)))
    adapter._states[REQUEST_ID] = _ParoMTP2RequestState(
        REQUEST_ID, _ChainProposer(token=200)
    )

    capability = adapter.capability((_semantics(),))

    assert capability is not None
    assert capability.max_context_tokens == 1024


def test_capability_is_absent_without_a_registered_depth() -> None:
    adapter = Qwen35ParoMTP2Adapter(_owner(), candidate_budget=3)
    adapter._states[REQUEST_ID] = _ParoMTP2RequestState(
        REQUEST_ID, _ChainProposer(token=200)
    )

    assert adapter.capability((_semantics(),)) is None


def test_request_depth_clamps_to_the_adapter_budget() -> None:
    adapter = Qwen35ParoMTP2Adapter(_owner(), candidate_budget=2)
    adapter.register_request(REQUEST_ID, 3)

    assert adapter.candidate_budget == 2
    assert adapter.request_depth(REQUEST_ID) == 2

    shallow = Qwen35ParoMTP2Adapter(_owner(), candidate_budget=1)
    shallow.register_request(REQUEST_ID, 3)
    assert shallow.request_depth(REQUEST_ID) == 1


def test_adapter_budget_is_capped_by_the_structural_depth() -> None:
    adapter = Qwen35ParoMTP2Adapter(_owner(), candidate_budget=99)

    assert adapter.candidate_budget == PARO_MTP2_MAX_CANDIDATE_DEPTH == 3


# -- planning through the shared policy ------------------------------------


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_shared_policy_plans_the_registered_depth(depth) -> None:
    adapter = _adapter(depth=depth)
    adapter._states[REQUEST_ID] = _ParoMTP2RequestState(
        REQUEST_ID, _ChainProposer(token=200)
    )

    plan = _plan(adapter, desired=depth)

    assert plan.candidate_counts == (depth,)
    assert plan.proposal_widths == (1,)
    assert plan.target_row_decomposition == (depth + 1,)
    assert plan.execution_route == "eager"
    assert plan.capability_key is not None
    assert adapter.claims_fit(plan) is True


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_shared_policy_clamps_desired_depth_to_the_capability(depth) -> None:
    adapter = _adapter(depth=depth)
    adapter._states[REQUEST_ID] = _ParoMTP2RequestState(
        REQUEST_ID, _ChainProposer(token=200)
    )

    plan = _plan(adapter, desired=PARO_MTP2_MAX_CANDIDATE_DEPTH)

    assert plan.candidate_counts == (depth,)
    assert plan.requested_candidate_counts == (PARO_MTP2_MAX_CANDIDATE_DEPTH,)


@pytest.mark.parametrize(
    ("depth", "remaining_decode", "expected"),
    (
        (3, 32, 3),
        (3, 4, 3),
        (3, 3, 2),
        (3, 2, 1),
        (3, 1, 0),
        (2, 3, 2),
        (1, 2, 1),
    ),
)
def test_shared_policy_keeps_the_output_tail(depth, remaining_decode, expected) -> None:
    """A cycle may publish ``accepted + 1`` tokens and never exceed the tail."""

    adapter = _adapter(depth=depth)
    adapter._states[REQUEST_ID] = _ParoMTP2RequestState(
        REQUEST_ID, _ChainProposer(token=200)
    )

    plan = _plan(adapter, desired=depth, remaining_decode=remaining_decode)

    assert plan.candidate_counts == (expected,)
    if expected:
        # Full acceptance publishes one bonus token past the candidates.
        assert expected + 1 <= remaining_decode


@pytest.mark.parametrize(
    ("context_tokens", "expected"),
    (
        # 4096 positions: the planner reserves one for the root and lets the
        # chain consume the rest, so the selection steps down 3 -> 2 -> 1 -> 0
        # exactly as the room shrinks.
        (128, 3),
        (4090, 3),
        (4092, 3),
        (4093, 2),
        (4094, 1),
        (4095, 0),
    ),
)
def test_a_depth_three_plan_steps_down_as_context_space_shrinks(
    context_tokens, expected
) -> None:
    adapter = _adapter(depth=3)
    adapter._states[REQUEST_ID] = _ParoMTP2RequestState(
        REQUEST_ID, _ChainProposer(token=200)
    )

    plan = _plan(
        adapter,
        desired=3,
        context_tokens=context_tokens,
        remaining_decode=64,
    )

    assert plan.candidate_counts == (expected,)
    if expected:
        # The root sits at ``context - 1`` and candidate ``k`` at
        # ``context - 1 + k``, so this is the highest position the cycle writes.
        highest = context_tokens - 1 + expected
        assert highest < 4096
        assert context_tokens + expected <= 4096
    else:
        assert plan.execution_route == "ar"


# -- proposal chains -------------------------------------------------------


def test_chained_proposal_drafts_host_tokens_and_keeps_repair_snapshots() -> None:
    proposer = _ChainProposer(token=201, drafted=(202, 203))
    adapter = _adapter(depth=3, proposer=proposer)
    plan = _plan(adapter, desired=3)
    row = adapter.owner._row(REQUEST_ID)

    graph = adapter.propose_batch(plan, (_semantics(),))

    assert graph.candidate_tokens == (201, 202, 203)
    assert graph.token_ids is None
    assert graph.provider_metadata == (("candidate_handoff", "host_i32"),)
    assert graph.row_offsets == (0, 3)
    assert graph.row_to_request == (REQUEST_ID,) * 3
    assert graph.parent_candidate_rows == (-1, 0, 1)
    assert graph.draft_depths == (1, 2, 3)
    assert graph.root_positions == (127,)
    assert graph.policy_fingerprint == "paro-mtp2:c3:k1"
    advances = [call for call in proposer.calls if call[0] == "previous"]
    assert advances == [
        ("previous", 201, 128, True, True),
        ("previous", 202, 129, True, True),
    ]
    assert [call for call in proposer.calls if call[0] == "save"] == [
        # Slot 0 is the cycle checkpoint; slots 1..count-1 are repair points.
        ("save", 0, 127),
        ("save", 1, 127),
        ("save", 2, 128),
    ]
    # The proposal checkpoint uses slot 0 and must not collide with repair slots.
    assert adapter._states[REQUEST_ID].checkpoint == ("snapshot", 0, 127, 128)
    assert [snapshot[1] for snapshot in adapter._states[REQUEST_ID].draft_snapshots] == [1, 2]
    assert row.mtp2_candidate_device_handoffs == 0


def test_shallow_plan_at_a_deep_budget_uses_host_tokens_and_no_snapshots() -> None:
    proposer = _ChainProposer(token=201, drafted=(202,))
    adapter = _adapter(depth=3, proposer=proposer)
    plan = _plan(adapter, desired=1)

    graph = adapter.propose_batch(plan, (_semantics(),))

    assert plan.candidate_counts == (1,)
    assert graph.candidate_tokens == (201,)
    assert graph.token_ids is None
    assert graph.row_offsets == (0, 1)
    assert graph.provider_metadata == (("candidate_handoff", "host_i32"),)
    assert [call for call in proposer.calls if call[0] == "previous"] == []
    assert [call for call in proposer.calls if call[0] == "save"] == [
        ("save", 0, 127)
    ]
    assert adapter._states[REQUEST_ID].draft_snapshots == ()


def test_depth_one_proposal_keeps_the_device_candidate_handoff() -> None:
    proposer = _ChainProposer(token=201)
    adapter = _adapter(depth=1, proposer=proposer)
    plan = _plan(adapter, desired=1)
    row = adapter.owner._row(REQUEST_ID)

    graph = adapter.propose_batch(plan, (_semantics(),))

    assert graph.candidate_tokens == (0,)
    assert graph.token_ids is not None and graph.token_ids.ptr == 0x7000
    assert graph.provider_metadata == (("candidate_handoff", "device_i32"),)
    assert graph.policy_fingerprint == "paro-mtp2:c1:k1"
    assert row.mtp2_candidate_device_handoffs == 1
    assert [call for call in proposer.calls if call[0] == "previous"] == []
    assert adapter._states[REQUEST_ID].draft_snapshots == ()


def test_chained_proposal_refuses_a_device_only_candidate_token() -> None:
    proposer = _ChainProposer(token=-1, drafted=(202, 203))
    adapter = _adapter(depth=3, proposer=proposer)
    plan = _plan(adapter, desired=3)

    with pytest.raises(RuntimeError, match="host candidate token"):
        adapter.propose_batch(plan, (_semantics(),))


def test_proposal_refuses_a_depth_beyond_the_registered_budget() -> None:
    proposer = _ChainProposer(token=201)
    adapter = _adapter(depth=2, proposer=proposer)
    plan = SimpleNamespace(
        request_ids=(REQUEST_ID,),
        resident_slots=(0,),
        provider_key="qwen_paro_mtp_bf16",
        cycle_id=1,
        candidate_counts=(3,),
    )

    with pytest.raises(RuntimeError, match="candidate budget"):
        adapter.propose_batch(plan, (_semantics(),))


# -- acceptance repair -----------------------------------------------------


@pytest.mark.parametrize(
    ("depth", "count", "accepted"),
    (
        (1, 1, 0),
        (1, 1, 1),
        (2, 2, 0),
        (2, 2, 1),
        (2, 2, 2),
        (3, 2, 0),
        (3, 2, 1),
        (3, 2, 2),
        (3, 3, 0),
        (3, 3, 1),
        (3, 3, 2),
        (3, 3, 3),
    ),
)
def test_cycle_repairs_the_provider_to_the_accepted_prefix(depth, count, accepted) -> None:
    drafted = tuple(201 + step for step in range(count))
    proposer = _ChainProposer(token=drafted[0], drafted=drafted[1:])
    verify = _VerifyResult(accepted_tokens=drafted[:accepted], bonus=900)
    adapter = _adapter(depth=depth, proposer=proposer)
    adapter.owner._session._verify = verify
    plan = _plan(adapter, desired=count)
    assert plan.candidate_counts == (count,)

    graph = adapter.propose_batch(plan, (_semantics(),))
    proposer.calls.clear()
    adapter.reserve_claims(_complete_claims(count))
    adapter.prepare_requests(plan, (_semantics(),))

    result = adapter.execute_target_frontier(
        plan,
        _frontier(adapter, graph),
        complete_claims=_complete_claims(count),
        commit=True,
        cancelled_request_ids=lambda: (),
    )

    advances = [call for call in proposer.calls if call[0] in {"previous", "restore"}]
    if accepted >= count:
        assert advances == [("previous", drafted[-1], 127 + count, False, True)]
    elif accepted == count - 1:
        assert advances == []
    else:
        assert advances == [("restore", accepted + 1)]
    # The bonus row is always taken from the repaired prefix.
    target = [call for call in proposer.calls if call[0] == "target"]
    assert target == [("target", 900, 127 + accepted + 1, 0x5000, depth > 1)]
    assert proposer.position == 127 + accepted + 1
    assert adapter._states[REQUEST_ID].draft_snapshots == ()

    assert result.accept_result is not None
    assert result.accept_result.accepted_counts == (accepted,)
    assert result.accept_result.correction_or_bonus_tokens == (900,)
    assert result.accept_result.provider_cursor_deltas == (accepted,)
    assert result.accept_result.target_cursor_deltas == (accepted + 1,)


def test_cycle_publishes_the_accepted_prefix_then_the_bonus_token() -> None:
    steps: list = []
    owner = _owner(steps=steps)
    proposer = _ChainProposer(token=201, drafted=(202, 203))
    verify = _VerifyResult(accepted_tokens=(201, 202), bonus=900)
    owner._session._verify = verify
    adapter = _adapter(depth=3, owner=owner, proposer=proposer)
    plan = _plan(adapter, desired=3)

    graph = adapter.propose_batch(plan, (_semantics(),))
    adapter.reserve_claims(_complete_claims(3))
    adapter.prepare_requests(plan, (_semantics(),))

    adapter.execute_target_frontier(
        plan,
        _frontier(adapter, graph),
        complete_claims=_complete_claims(3),
        commit=True,
        cancelled_request_ids=lambda: (),
    )

    row = owner._row(REQUEST_ID)
    assert [step.token_id for _selected, step in steps] == [201, 202, 900]
    assert row.mtp2_cycles == 1
    assert row.mtp2_candidate_counts == [3]
    assert row.mtp2_accepted_counts == [2]
    assert row.mtp2_execution_routes == ["eager"]


def test_depth_one_cycle_keeps_the_one_candidate_shape() -> None:
    steps: list = []
    owner = _owner(steps=steps)
    proposer = _ChainProposer(token=201)
    verify = _VerifyResult(accepted_tokens=(201,), bonus=900)
    owner._session._verify = verify
    adapter = _adapter(depth=1, owner=owner, proposer=proposer)
    plan = _plan(adapter, desired=1)

    graph = adapter.propose_batch(plan, (_semantics(),))
    adapter.reserve_claims(_complete_claims(1))
    adapter.prepare_requests(plan, (_semantics(),))

    adapter.execute_target_frontier(
        plan,
        _frontier(adapter, graph),
        complete_claims=_complete_claims(1),
        commit=True,
        cancelled_request_ids=lambda: (),
    )

    row = owner._row(REQUEST_ID)
    assert [step.token_id for _selected, step in steps] == [201, 900]
    assert row.mtp2_candidate_counts == [1]
    assert row.mtp2_candidate_device_handoffs == 1
    # One-candidate cycles still hand the target the device scalar.
    assert owner._session.verify_calls[-1][1]["candidate_token_ids_i32"] is not None
    # And they keep the eager route the runner selects.
    assert owner._session.verify_calls[-1][1]["graph_mode"] == "off"


# -- restoration -----------------------------------------------------------


def test_proposal_failure_leaves_a_rollbackable_provider_journal() -> None:
    proposer = _ChainProposer(token=201, drafted=(202, 203))
    adapter = _adapter(depth=3, proposer=proposer)
    plan = _plan(adapter, desired=3)

    def fail(**kwargs):
        raise RuntimeError("injected draft failure")

    proposer.advance_with_previous_hidden = fail
    with pytest.raises(RuntimeError, match="injected draft failure"):
        adapter.propose_batch(plan, (_semantics(),))

    state = adapter._states[REQUEST_ID]
    assert state.journal is not None
    assert state.checkpoint is not None

    adapter.rollback_cycle(plan, None, RuntimeError("injected draft failure"))

    assert state.journal is None
    assert state.checkpoint is None
    assert state.draft_snapshots == ()
    # Only the provider was captured before the failure, so only the provider
    # is restored; the target is untouched and owns no checkpoint yet.
    assert ("restore", 0) in proposer.calls
    assert adapter.owner._session.copies == []


def test_cancel_after_verify_restores_both_owners_and_publishes_nothing() -> None:
    steps: list = []
    owner = _owner(steps=steps)
    proposer = _ChainProposer(token=201, drafted=(202, 203))
    verify = _VerifyResult(accepted_tokens=(201,), bonus=900)
    owner._session._verify = verify
    adapter = _adapter(depth=3, owner=owner, proposer=proposer)
    plan = _plan(adapter, desired=3)

    graph = adapter.propose_batch(plan, (_semantics(),))
    adapter.reserve_claims(_complete_claims(3))
    adapter.prepare_requests(plan, (_semantics(),))

    result = adapter.execute_target_frontier(
        plan,
        _frontier(adapter, graph),
        complete_claims=_complete_claims(3),
        commit=True,
        cancelled_request_ids=lambda: (REQUEST_ID,),
    )

    assert result.cancelled_request_ids == (REQUEST_ID,)
    assert steps == []
    assert (1, 0, 0) in owner._session.copies
    assert adapter._states[REQUEST_ID].journal is None
    assert adapter._states[REQUEST_ID].draft_snapshots == ()
    assert owner._row(REQUEST_ID).mtp2_cycles == 0


# -- sizing ----------------------------------------------------------------


def test_proposer_capacity_covers_the_whole_candidate_chain() -> None:
    assert _proposer_token_requirement(2, 8, 1) == 2 + 16 + 1 + 8
    assert _proposer_token_requirement(2, 8, 3) == 2 + 16 + 3 + 8
    # The chain rows are what crosses the bucket, not a second copy of context.
    assert _proposer_capacity_bucket(_proposer_token_requirement(110, 68, 1)) == 256
    assert _proposer_capacity_bucket(_proposer_token_requirement(110, 68, 3)) == 512


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_prefill_reserves_one_verify_row_per_candidate(monkeypatch, depth) -> None:
    calls: list = []

    class FakeProposer:
        closed = False

        def __init__(self, _model, **kwargs) -> None:
            self.max_positions = kwargs["max_positions"]
            self.max_mtp_tokens = kwargs["max_mtp_tokens"]
            self.scoring_head = kwargs["scoring_head"]

        def reset(self) -> None:
            return None

    monkeypatch.setattr(paro_mtp2_module, "NativeMtpChainProposer", FakeProposer)
    session = _Session()
    session.lm_head_weight = SimpleNamespace(tensor=SimpleNamespace(ptr=0x1000))
    session.lm_head_scale = SimpleNamespace(tensor=SimpleNamespace(ptr=0x2000))
    session.vocab_size = 248320
    session.lm_head_threads = 256
    session.runtime = object()
    session.compiler_version = None
    session.prepare_specdec2_verify_scratch = (
        lambda *, rows, chain_attn_mode, max_context_tokens: calls.append(
            (int(rows), str(chain_attn_mode), int(max_context_tokens))
        )
    )
    row = _row(max_tokens=8, prompt_ids=(10, 11))
    owner = _owner(session=session, row=row)
    adapter = _adapter(depth=depth, owner=owner)

    adapter.begin_prompt(REQUEST_ID)
    state = adapter._states[REQUEST_ID]
    state.prompt_rows_consumed = len(row.prompt_ids)
    adapter.observe_prefill_result(REQUEST_ID)

    assert calls == [(depth + 1, "decode_batched", 2 + 8 + depth)]
    assert state.proposer.max_mtp_tokens == _proposer_capacity_bucket(
        _proposer_token_requirement(2, 8, depth)
    )


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_prompt_root_reads_the_host_token_only_for_chained_depths(depth) -> None:
    proposer = _ChainProposer(token=201)
    adapter = _adapter(depth=depth, proposer=proposer)
    row = adapter.owner._row(REQUEST_ID)
    # Every earlier prompt row is already primed; this is the root row.
    adapter._states[REQUEST_ID].prompt_rows_consumed = len(row.prompt_ids) - 1

    adapter.consume_prompt_row(
        REQUEST_ID,
        prompt_index=len(row.prompt_ids) - 1,
        target_hidden_ptr=0x3000,
        seed_token=99,
    )

    assert [call for call in proposer.calls if call[0] == "target"] == [
        ("target", 99, len(row.prompt_ids), 0x3000, depth > 1)
    ]


@pytest.mark.parametrize(("declared", "expected"), ((None, 2), (1, 1), (2, 2), (3, 3)))
def test_generator_depth_prefers_two_and_preserves_explicit_requests(declared, expected) -> None:
    generator = object.__new__(Qwen35ParoOneTokenGenerator)
    generator.backend = "hip_gfx1100"
    generator.model_plugin = SimpleNamespace(
        speculative_mtp_serving_implementations=(
            SimpleNamespace(
                kv_storage="bf16",
                backends=(("hip_gfx1100", "gfx1100"),),
                max_candidate_count=3,
            ),
        )
    )
    if declared is not None:
        generator.speculative_candidate_budget = declared

    assert generator.mtp2_candidate_depth() == expected


def test_generator_default_depth_is_the_adapter_structural_max() -> None:
    generator = object.__new__(Qwen35ParoOneTokenGenerator)
    generator.backend = "auto"
    generator.model_plugin = SimpleNamespace(
        speculative_mtp_serving_implementations=(),
    )

    assert generator.speculative_candidate_budget_default() == 2
