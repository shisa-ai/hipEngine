"""CPU contract coverage for the PARO MTP2 adapter at candidate depths 1-3.

The adapter plans one to three candidates per cycle. A one-candidate cycle hands
the target the provider's own packed top-1 scalar; a deeper cycle walks the
provider's device candidate rows, publishing each drafted candidate's int64
chain token so no draft id round-trips through the host. Either way the proposal
keeps one provider prefix per drafted candidate, so after verification the
accepted prefix is restored exactly before the target's bonus row is taken.

These tests drive the real ``Qwen35ParoMTP2Adapter`` cycle -- proposal, claims,
preparation, target execution, output accounting -- against a proposer double
that models the provider's device candidate rows, chain token, and prefixes.

Device copies, verifier arithmetic, and the resident target run on the live
routes. What is pinned here is the host-side protocol: which candidate row is
published and consumed, which prefix is saved and restored, which advance runs,
which rows the cycle declares, and what the request reports after a commit, a
rollback, or a cancellation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.generation.qwen35_paro_mtp2 as paro_mtp2_module
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.generation.qwen35_paro_mtp2 import (
    PARO_MTP2_MAX_CANDIDATE_DEPTH,
    Qwen35ParoMTP2Adapter,
    _ParoMTP2RequestState,
    _proposer_capacity_bucket,
    _proposer_token_requirement,
)
from hipengine.kvcache import ResourceClaimSet
from hipengine.speculative import SpeculativeRequestSemantics, plan_speculative_requests
from hipengine.speculative.frontier import TargetFrontier

_REQUEST_ID = 7
_ROOT_POSITION = 127
_SELECTED_HIDDEN = 0x5000
_DEVICE_CANDIDATES = 0x7000


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _ChainProposer:
    """Provider state machine: device candidate rows, chain token, prefixes.

    ``rows[0]`` is the candidate the prompt root's (or the previous cycle's
    reseed) scoring pass left behind. Each ``advance_chain_step`` consumes the
    published chain token and scores the next row; ``publish_chain_token``
    widens the *saved* candidate for the current row, never scoring scratch that
    a later prefix overwrites. A prefix snapshot owns the rows, the published
    token, and the cursors together, exactly as ``NativeMtpStateSnapshot`` does.
    """

    def __init__(
        self,
        *,
        first_candidate: int = 201,
        position: int = _ROOT_POSITION,
        max_candidate_rows: int = 3,
        fail_chain_at: int | None = None,
        fail_bonus: bool = False,
    ) -> None:
        # Every scoring pass produces a fresh id from one monotonic counter, so
        # a pass over a restored prefix is distinguishable from the pass that
        # produced the rejected candidate.
        self._first = int(first_candidate)
        self._scored = 0
        self.rows: list[int | None] = [self._first, *([None] * 8)]
        self._candidate_row = 0
        self._chain_token: int | None = None
        self._chain_token_current = False
        self.current = SimpleNamespace(token=-1)
        self.position = int(position)
        self.cache_len = int(position) + 1
        self.max_positions = 4096
        self.max_mtp_tokens = 256
        self.max_candidate_rows = int(max_candidate_rows)
        self.scoring_head: SimpleNamespace | None = None
        self.closed = False
        self.calls: list[tuple] = []
        self._snapshots: dict[int, SimpleNamespace] = {}
        self._fail_chain_at = fail_chain_at
        self._fail_bonus = bool(fail_bonus)

    # -- device candidate chain -------------------------------------------

    def _score(self, row: int) -> int:
        self._scored += 1
        token = self._first + self._scored
        self.rows[int(row)] = token
        self._candidate_row = int(row)
        self._chain_token_current = False
        self.current = SimpleNamespace(token=-1)
        return token

    def ensure_chain_token(self) -> None:
        self.calls.append(("ensure",))
        if not self._chain_token_current:
            self.publish_chain_token()

    def publish_chain_token(self) -> None:
        if self.rows[self._candidate_row] is None:
            raise RuntimeError("no device candidate is ready to publish")
        self._chain_token = int(self.rows[self._candidate_row])
        self._chain_token_current = True
        self.calls.append(("publish", self._candidate_row, self._chain_token))

    def advance_chain_step(self, *, candidate_row: int) -> SimpleNamespace:
        if not self._chain_token_current:
            raise RuntimeError("chained advance requires a published chain token")
        if self._fail_chain_at == int(candidate_row):
            raise RuntimeError("MTP chain step failed")
        consumed = self._chain_token
        self.position += 1
        self.cache_len += 1
        self._score(int(candidate_row))
        self.calls.append(("step", int(candidate_row), consumed, self.position))
        return self.current

    def device_candidate_token_ids(self, rows: int = 1) -> Tensor:
        self.calls.append(("device_candidates", int(rows)))
        return Tensor.from_handle(
            _DEVICE_CANDIDATES, (int(rows),), DType.INT32, Device("hip", 0)
        )

    # -- prefixes ---------------------------------------------------------

    def save_state(self, slot: int, *, stream: int = 0) -> SimpleNamespace:
        snapshot = SimpleNamespace(
            slot=int(slot),
            position=self.position,
            cache_len=self.cache_len,
            rows=list(self.rows),
            candidate_row=self._candidate_row,
            chain_token=self._chain_token,
            chain_token_current=self._chain_token_current,
        )
        self._snapshots[int(slot)] = snapshot
        self.calls.append(("save", int(slot), self.position))
        return snapshot

    def restore_state(self, snapshot: SimpleNamespace, *, stream: int = 0) -> None:
        self.position = int(snapshot.position)
        self.cache_len = int(snapshot.cache_len)
        self.rows = list(snapshot.rows)
        self._candidate_row = int(snapshot.candidate_row)
        self._chain_token = snapshot.chain_token
        self._chain_token_current = bool(snapshot.chain_token_current)
        self.current = SimpleNamespace(token=-1)
        self.calls.append(("restore", int(snapshot.slot), self.position))

    # -- post-verify ------------------------------------------------------

    def advance_with_previous_hidden(self, **kwargs) -> SimpleNamespace:
        self.position = int(kwargs["position"])
        self.cache_len = self.position + 1
        if kwargs["need_result"]:
            self._score(0)
        else:
            # A repair advance produces no candidate and leaves no hidden carry.
            self._chain_token_current = False
            self.current = SimpleNamespace(token=-1)
        self.calls.append(
            (
                "previous",
                int(kwargs["input_token"]),
                self.position,
                bool(kwargs["need_result"]),
                bool(kwargs.get("read_token_id", True)),
            )
        )
        return self.current

    def advance_with_target_hidden(self, **kwargs) -> SimpleNamespace:
        if self._fail_bonus:
            raise RuntimeError("MTP bonus reseed failed")
        self.position = int(kwargs["position"])
        self.cache_len = self.position + 1
        self._score(0)
        self.calls.append(
            (
                "target",
                int(kwargs["input_token"]),
                self.position,
                int(kwargs["target_hidden_ptr"]),
                bool(kwargs.get("read_token_id", True)),
            )
        )
        return self.current

    # -- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        self.calls.append(("reset",))

    def close(self) -> None:
        self.closed = True
        self.calls.append(("close",))

    def kinds(self) -> list[str]:
        return [call[0] for call in self.calls]


class _Verify:
    """One verifier result for a cycle of ``count`` candidate rows."""

    def __init__(
        self,
        *,
        accepted_tokens: tuple[int, ...],
        bonus: int,
        count: int,
    ) -> None:
        self.accepted_tokens = tuple(int(token) for token in accepted_tokens)
        self.accepted_count = len(self.accepted_tokens)
        self.next_token = int(bonus)
        self.target_top1 = (*self.accepted_tokens, int(bonus))
        self.commit_row = self.accepted_count
        self.commit_token = int(bonus)
        self.commit_position = _ROOT_POSITION + self.accepted_count
        self.rows = int(count) + 1
        self.selected_target_hidden_ptr = _SELECTED_HIDDEN
        self.graph = {"status": "direct", "replayed": False}


class _Session:
    def __init__(self, *, verify: _Verify | None = None) -> None:
        self.max_sequence_length = 4096
        self.position_arr = [_ROOT_POSITION]
        self.tokenizer = object()
        self.runtime = SimpleNamespace(device_synchronize=lambda: None)
        self.lm_head_weight = SimpleNamespace(tensor=SimpleNamespace(ptr=0x1000))
        self.lm_head_scale = SimpleNamespace(tensor=SimpleNamespace(ptr=0x2000))
        self.vocab_size = 248320
        self.lm_head_threads = 256
        self.compiler_version = None
        self.copies: list[tuple[int, int, int]] = []
        self.slot_positions: list[tuple[int, int]] = []
        self.verify_calls: list[tuple] = []
        self.scratch: list[tuple[int, str, int]] = []
        self.verify = verify

    def copy_slot_state(self, source: int, destination: int, *, kv_rows: int) -> None:
        self.copies.append((int(source), int(destination), int(kv_rows)))

    def _record_slot_position_host(self, position: int, *, slot: int) -> None:
        self.slot_positions.append((int(position), int(slot)))

    def verify_chain_bulk_and_commit(self, batch, **kwargs):
        self.verify_calls.append((batch, kwargs))
        assert self.verify is not None
        return self.verify

    def prepare_specdec2_verify_scratch(
        self, *, rows, chain_attn_mode, max_context_tokens
    ) -> None:
        self.scratch.append((int(rows), str(chain_attn_mode), int(max_context_tokens)))


class _Owner:
    def __init__(
        self,
        *,
        session: _Session | None = None,
        row: SimpleNamespace | None = None,
        profile: str = "production",
    ) -> None:
        self.generator = SimpleNamespace(
            backend="hip_gfx1100",
            execution_profile=profile,
            model_path="/tmp/paro-model",
        )
        # One request row for the whole adapter: the adapter caches counters on
        # the object its owner hands it, so a fresh row per lookup would lose
        # them.
        self.row = _row() if row is None else row
        self._row = lambda request_id: self.row
        self._session = session if session is not None else _Session()
        self.recorded_tokens: list[int] = []
        self.released_captures: list[object] = []

    def _record_step(self, selected, step) -> None:
        self.recorded_tokens.append(int(step.token_id))
        selected.generated_steps.append(step)

    def _release_mtp2_prompt_capture(self, selected) -> None:
        self.released_captures.append(selected)


def _row(*, prompt_ids: tuple[int, ...] = (10, 11), max_tokens: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=_REQUEST_ID,
        model_slot=0,
        native_greedy=True,
        first_token_emitted=True,
        prompt_ids=tuple(prompt_ids),
        request=SimpleNamespace(max_tokens=int(max_tokens)),
        generated_steps=[],
        mtp2_candidate_budget=0,
        mtp2_requested_budget=0,
        mtp2_cycles=0,
        mtp2_candidate_counts=[],
        mtp2_accepted_counts=[],
        mtp2_execution_routes=[],
        mtp2_mtp_output_tokens=0,
        mtp2_ar_step_output_tokens=0,
        mtp2_ar_step_reasons={},
        mtp2_candidate_device_handoffs=0,
        mtp2_candidate_d2h_after_target=0,
        mtp2_proposal_ms=0.0,
        mtp2_target_ms=0.0,
        mtp2_provider_update_ms=0.0,
        mtp2_prompt_prime_ms=0.0,
        mtp2_provider_open_ms=0.0,
        mtp2_bulk_prompt_chunks=0,
        mtp2_bulk_prompt_rows=0,
        mtp2_prompt_hidden_buffer=None,
    )


def _adapter(
    *,
    depth: int,
    owner: _Owner | None = None,
    proposer: _ChainProposer | None = None,
) -> Qwen35ParoMTP2Adapter:
    owner = _Owner() if owner is None else owner
    adapter = Qwen35ParoMTP2Adapter(owner, candidate_budget=depth)
    adapter.register_request(_REQUEST_ID, depth)
    if proposer is not None:
        adapter._states[_REQUEST_ID] = _ParoMTP2RequestState(_REQUEST_ID, proposer)
    return adapter


def _semantics(*, context_tokens: int = 128, remaining_decode: int = 32):
    return SpeculativeRequestSemantics(
        _REQUEST_ID,
        "greedy",
        "verify_chain",
        int(context_tokens),
        int(remaining_decode),
    )


def _plan(adapter: Qwen35ParoMTP2Adapter, *, desired: int):
    """Plan through the real shared policy against the adapter's capability."""

    semantics = (_semantics(),)
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


def _claims(count: int) -> ResourceClaimSet:
    return ResourceClaimSet.from_mapping(
        "op", {"paro_mtp2.target_rows": int(count) + 1}
    )


def _frontier(adapter: Qwen35ParoMTP2Adapter, plan, graph) -> TargetFrontier:
    return TargetFrontier.from_candidate_graph(
        operation_id=plan.operation_id,
        candidate_graph=graph,
        root_tokens=(100,),
        physical_row_decomposition=plan.target_row_decomposition,
        transaction_mode=plan.target_transaction_mode,
        kv_storage_view_key="paged_bf16_kv_live_spans",
        kv_live_spans_owner=adapter.kv_live_spans_owner(plan),
        execution_route=plan.execution_route,
    )


def _run_cycle(
    adapter: Qwen35ParoMTP2Adapter,
    *,
    desired: int,
    verify: _Verify,
    cancelled=lambda: (),
):
    """Propose and execute one committed-or-cancelled cycle for ``desired`` rows."""

    session = adapter.owner._session
    session.verify = verify
    plan = _plan(adapter, desired=desired)
    assert plan.candidate_counts == (desired,)
    claims = _claims(desired)
    adapter.reserve_claims(claims)
    try:
        adapter.prepare_requests(plan, (_semantics(),))
        graph = adapter.propose_batch(plan, (_semantics(),))
        # Cycle assertions start at the verifier: the proposal's own provider
        # calls are covered by the proposal cases above.
        adapter._states[_REQUEST_ID].proposer.calls.clear()
        result = adapter.execute_target_frontier(
            plan,
            _frontier(adapter, plan, graph),
            claims,
            commit=True,
            cancelled_request_ids=cancelled,
        )
    finally:
        adapter.release_claims(claims)
    return plan, graph, result


def _drafted_tokens(count: int) -> tuple[int, ...]:
    """Candidate ids the provider's scoring passes produce, in order."""

    return tuple(201 + step for step in range(count))


# ---------------------------------------------------------------------------
# Declared depth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_register_request_keeps_every_executable_depth(depth) -> None:
    adapter = Qwen35ParoMTP2Adapter(_Owner(), candidate_budget=depth)

    adapter.register_request(_REQUEST_ID, depth)

    assert adapter.request_depth(_REQUEST_ID) == depth


@pytest.mark.parametrize(("budget", "requested", "expected"), (
    (1, 3, 1),
    (2, 3, 2),
    (3, 1, 1),
    (99, 3, 3),
))
def test_register_request_clamps_to_the_adapter_budget(budget, requested, expected) -> None:
    adapter = Qwen35ParoMTP2Adapter(_Owner(), candidate_budget=budget)

    adapter.register_request(_REQUEST_ID, requested)

    assert adapter.candidate_budget == min(budget, PARO_MTP2_MAX_CANDIDATE_DEPTH)
    assert adapter.request_depth(_REQUEST_ID) == expected


def test_register_request_refuses_a_non_positive_budget() -> None:
    adapter = Qwen35ParoMTP2Adapter(_Owner(), candidate_budget=3)

    with pytest.raises(ValueError, match="candidate budget must be positive"):
        adapter.register_request(_REQUEST_ID, 0)

    assert adapter.request_depth(_REQUEST_ID) == 0


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_capability_declares_the_requested_depth(depth) -> None:
    adapter = _adapter(depth=depth, proposer=_ChainProposer())

    capability = adapter.capability((_semantics(),))

    assert capability is not None
    assert capability.max_requests == 1
    assert capability.max_candidates_per_request == depth
    assert capability.max_frontier_rows == depth + 1
    assert capability.proposal_widths == (1,)
    assert capability.target_row_buckets == tuple(range(2, depth + 2))
    assert f"paro_mtp2_c{depth}" in capability.capability_key
    assert f"paro-mtp2:c{depth}:k1" in capability.policy_fingerprint
    # The target arithmetic contract must not move with depth.
    assert "fast_d64_candidate" in capability.capability_key
    assert capability.supports_shape(
        request_count=1, candidate_counts=(depth,), mode="verify_chain"
    )
    assert not capability.supports_shape(
        request_count=1, candidate_counts=(depth + 1,), mode="verify_chain"
    )

    adapter.owner.generator.execution_profile = "strict"
    strict = adapter.capability((_semantics(),))
    assert strict is not None
    assert strict.max_candidates_per_request == depth
    assert "strict_exact" in strict.capability_key
    assert f"paro-mtp2:c{depth}:k1" in strict.policy_fingerprint


def test_capability_declares_one_depth_per_request_not_the_ceiling() -> None:
    adapter = _adapter(depth=3, proposer=_ChainProposer())
    adapter.register_request(9, 3)
    adapter._states[9] = _ParoMTP2RequestState(9, _ChainProposer())

    ceiling = adapter.capability((_semantics(),))
    assert ceiling is not None
    assert ceiling.max_candidates_per_request == 3

    adapter.register_request(_REQUEST_ID, 1)
    per_request = adapter.capability((_semantics(),))
    assert per_request is not None
    assert per_request.max_candidates_per_request == 1
    assert per_request.max_frontier_rows == 2
    # Two requests on one adapter advertise different depths.
    assert per_request.capability_key != ceiling.capability_key


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_claims_and_preparation_follow_the_selected_count(depth) -> None:
    adapter = _adapter(depth=depth, proposer=_ChainProposer())
    plan = _plan(adapter, desired=depth)

    assert adapter.claims_fit(plan) is True
    claims = adapter.component_claims(plan)
    assert claims["provider"].units_by_pool() == {"paro_mtp2.provider_rows": 1}
    assert claims["target"].units_by_pool() == {"paro_mtp2.target_rows": depth + 1}
    assert claims["result"].units_by_pool() == {"paro_mtp2.result_rows": 1}
    assert all(claim.request_id == _REQUEST_ID for claim in claims.values())

    adapter.reserve_claims(_claims(depth))
    adapter.prepare_requests(plan, (_semantics(),))
    # An open claim blocks a second fit for the same request.
    assert adapter.claims_fit(plan) is False


def test_preparation_refuses_a_plan_deeper_than_the_request_declared() -> None:
    adapter = _adapter(depth=1, proposer=_ChainProposer())
    deep = SimpleNamespace(
        request_ids=(_REQUEST_ID,),
        speculative_request_ids=(_REQUEST_ID,),
        candidate_counts=(2,),
    )

    assert adapter.claims_fit(deep) is False
    adapter.reserve_claims(_claims(2))
    with pytest.raises(RuntimeError, match="not prepared/claim-fit"):
        adapter.prepare_requests(deep, (_semantics(),))


# ---------------------------------------------------------------------------
# Proposal: device scalar at one candidate, device chain above
# ---------------------------------------------------------------------------


def test_one_candidate_proposal_hands_the_target_the_device_scalar() -> None:
    proposer = _ChainProposer()
    adapter = _adapter(depth=1, proposer=proposer)
    row = adapter.owner._row(_REQUEST_ID)

    graph = adapter.propose_batch(_plan(adapter, desired=1), (_semantics(),))

    # No chain machinery runs for a single candidate: one checkpoint and one
    # device scalar, so the draft id never reaches the host.
    assert proposer.calls == [("save", 0, _ROOT_POSITION), ("device_candidates", 1)]
    assert graph.candidate_tokens == (0,)
    assert graph.token_ids is not None
    assert graph.token_ids.ptr == _DEVICE_CANDIDATES
    assert graph.token_ids.shape == (1,)
    assert graph.provider_metadata == (("candidate_handoff", "device_i32"),)
    assert graph.policy_fingerprint == "paro-mtp2:c1:k1"
    assert adapter._states[_REQUEST_ID].draft_snapshots == ()
    assert row.mtp2_candidate_device_handoffs == 1


@pytest.mark.parametrize("depth", (2, 3))
def test_chained_proposal_walks_the_device_candidate_rows(depth) -> None:
    drafted = _drafted_tokens(depth + 1)
    proposer = _ChainProposer()
    adapter = _adapter(depth=depth, proposer=proposer)
    row = adapter.owner._row(_REQUEST_ID)

    graph = adapter.propose_batch(_plan(adapter, desired=depth), (_semantics(),))

    assert graph.candidate_counts == (depth,)
    # Every candidate is a placeholder: the verifier reads the device rows.
    assert graph.candidate_tokens == (0,) * depth
    assert graph.token_ids is not None
    assert graph.token_ids.ptr == _DEVICE_CANDIDATES
    assert graph.token_ids.shape == (depth,)
    assert graph.provider_metadata == (("candidate_handoff", "device_i32"),)
    assert graph.row_offsets == (0, depth)
    assert graph.row_to_request == (_REQUEST_ID,) * depth
    assert graph.parent_candidate_rows == tuple(
        -1 if step == 0 else step - 1 for step in range(depth)
    )
    assert graph.draft_depths == tuple(range(1, depth + 1))
    assert graph.active_mask == (True,) * depth
    assert graph.root_positions == (_ROOT_POSITION,)
    assert graph.policy_fingerprint == f"paro-mtp2:c{depth}:k1"
    assert row.mtp2_candidate_device_handoffs == 1

    # Candidate 0 is already on device from the priming/reseed pass, so the
    # proposal publishes it and consumes it from the device chain. Each later
    # candidate comes from the previous step's own candidate row.
    expected_kinds = ["save", "ensure", "publish", "save"]
    for step in range(1, depth):
        expected_kinds += ["step", "publish"]
        if step < depth - 1:
            expected_kinds.append("save")
    expected_kinds.append("device_candidates")
    assert proposer.kinds() == expected_kinds
    assert [
        (call[1], call[2]) for call in proposer.calls if call[0] == "step"
    ] == [(step, drafted[step - 1]) for step in range(1, depth)]
    assert [call[1] for call in proposer.calls if call[0] == "publish"] == list(
        range(depth)
    )
    assert proposer.rows[:depth] == list(drafted[:depth])
    assert proposer.position == _ROOT_POSITION + depth - 1
    assert proposer.cache_len == _ROOT_POSITION + depth

    # Slot 0 is the cycle checkpoint; slots 1..count-1 are the repair prefixes,
    # each taken at the position the accepted prefix it repairs ends on.
    assert [call for call in proposer.calls if call[0] == "save"] == [
        ("save", slot, _ROOT_POSITION + max(0, slot - 1)) for slot in range(depth)
    ]
    state = adapter._states[_REQUEST_ID]
    assert state.checkpoint is proposer._snapshots[0]
    assert [snapshot.slot for snapshot in state.draft_snapshots] == list(range(1, depth))
    assert [snapshot.position for snapshot in state.draft_snapshots] == [
        _ROOT_POSITION + offset for offset in range(depth - 1)
    ]


def test_a_one_candidate_plan_at_a_deep_budget_keeps_the_scalar_handoff() -> None:
    """The selected count, not the declared depth, chooses the proposal shape."""

    proposer = _ChainProposer()
    adapter = _adapter(depth=3, proposer=proposer)

    plan = _plan(adapter, desired=1)
    graph = adapter.propose_batch(plan, (_semantics(),))

    assert plan.candidate_counts == (1,)
    assert graph.candidate_tokens == (0,)
    assert graph.token_ids is not None and graph.token_ids.shape == (1,)
    assert proposer.kinds() == ["save", "device_candidates"]
    assert adapter._states[_REQUEST_ID].draft_snapshots == ()


def test_proposal_refuses_a_cycle_deeper_than_the_registered_budget() -> None:
    proposer = _ChainProposer()
    adapter = _adapter(depth=2, proposer=proposer)
    deep = SimpleNamespace(
        request_ids=(_REQUEST_ID,),
        resident_slots=(0,),
        provider_key="qwen_paro_mtp_bf16",
        cycle_id=1,
        candidate_counts=(3,),
    )

    with pytest.raises(RuntimeError, match="candidate budget"):
        adapter.propose_batch(deep, (_semantics(),))


# ---------------------------------------------------------------------------
# Partial acceptance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", (1, 2, 3))
@pytest.mark.parametrize("accepted", (0, 1, 2, 3))
def test_each_accepted_prefix_repairs_the_provider_exactly(depth, accepted) -> None:
    if accepted > depth:
        pytest.skip("a cycle cannot accept more drafts than it proposed")
    drafted = _drafted_tokens(depth + 1)
    proposer = _ChainProposer()
    adapter = _adapter(depth=depth, proposer=proposer)
    row = adapter.owner._row(_REQUEST_ID)
    pre_position = proposer.position
    pre_cursor = proposer.cache_len
    verify = _Verify(accepted_tokens=drafted[:accepted], bonus=900, count=depth)

    plan, graph, result = _run_cycle(adapter, desired=depth, verify=verify)

    # The provider ends exactly where consuming the accepted drafts plus the
    # target's bonus row leaves it, whatever the accepted count is.
    assert proposer.position == pre_position + accepted + 1
    assert proposer.cache_len == pre_cursor + accepted + 1

    repairs = [call for call in proposer.calls if call[0] in {"previous", "restore"}]
    if accepted >= depth:
        # Full acceptance materializes the last draft from the live prefix.
        assert repairs == [
            ("previous", drafted[depth - 1], pre_position + depth, False, True)
        ]
    elif accepted == depth - 1:
        # The live draft state already sits on the accepted prefix.
        assert repairs == []
    else:
        # A shorter prefix restores the snapshot taken after that many drafts,
        # which is the prefix the next cycle publishes from.
        assert repairs == [("restore", accepted + 1, pre_position + accepted)]

    # The bonus row always comes from the repaired prefix on the target's
    # selected hidden, and it leaves its candidate on device.
    assert [call for call in proposer.calls if call[0] == "target"] == [
        ("target", 900, pre_position + accepted + 1, _SELECTED_HIDDEN, False)
    ]
    assert adapter._states[_REQUEST_ID].draft_snapshots == ()
    assert adapter._states[_REQUEST_ID].checkpoint is None
    assert adapter._states[_REQUEST_ID].journal is None

    # Output is the accepted drafts plus exactly one target token.
    assert adapter.owner.recorded_tokens == [*drafted[:accepted], 900]
    assert [step.token_id for step in row.generated_steps] == [*drafted[:accepted], 900]
    assert result.stage.value == "committed"
    accept = result.accept_result
    assert accept is not None
    assert accept.accepted_counts == (accepted,)
    assert accept.accepted_tokens == (drafted[:accepted],)
    assert accept.correction_or_bonus_tokens == (900,)
    assert result.provider_cursor_deltas == (accepted,)
    assert result.target_cursor_deltas == (accepted + 1,)
    assert result.committed_output_lengths == (accepted + 1,)
    assert result.committed_output_ids == ((*drafted[:accepted], 900),)
    assert result.telemetry is not None
    assert result.telemetry.candidate_counts == (depth,)

    # The request reports the committed cycle and its visible output split.
    assert row.mtp2_cycles == 1
    assert row.mtp2_candidate_counts == [depth]
    assert row.mtp2_accepted_counts == [accepted]
    assert row.mtp2_mtp_output_tokens == accepted + 1
    assert row.mtp2_execution_routes == ["eager"]
    assert row.mtp2_target_ms >= 0.0
    assert row.mtp2_provider_update_ms >= 0.0
    assert graph.candidate_counts == (depth,)
    assert plan.candidate_counts == (depth,)


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_full_accept_bonus_uses_the_verifier_next_token(depth) -> None:
    drafted = _drafted_tokens(depth + 1)
    adapter = _adapter(depth=depth, proposer=_ChainProposer())
    verify = _Verify(accepted_tokens=drafted[:depth], bonus=777, count=depth)

    _run_cycle(adapter, desired=depth, verify=verify)

    reseeds = [call for call in adapter._states[_REQUEST_ID].proposer.calls if call[0] == "target"]
    assert reseeds[0][1] == 777
    assert adapter.owner.recorded_tokens[-1] == 777


def test_a_plan_below_the_declared_depth_runs_only_the_selected_rows() -> None:
    """The plan's selected count -- not the request's declared depth -- runs.

    A short remaining-decode budget caps the plan below the declared depth, and
    every depth-shaped thing follows the selection: drafts, repair prefixes,
    verify rows, and the recorded cycle depth.
    """

    drafted = _drafted_tokens(4)
    proposer = _ChainProposer()
    adapter = _adapter(depth=3, proposer=proposer)
    verify = _Verify(accepted_tokens=drafted[:2], bonus=900, count=2)

    plan, graph, result = _run_cycle(adapter, desired=2, verify=verify)

    assert plan.candidate_counts == (2,)
    assert graph.candidate_counts == (2,)
    assert graph.draft_depths == (1, 2)
    assert graph.token_ids is not None and graph.token_ids.shape == (2,)
    # One repair prefix for the two-candidate cycle, not one per declared row.
    assert [snapshot.slot for snapshot in _snapshots_saved(proposer)] == [0, 1]
    batch, kwargs = adapter.owner._session.verify_calls[0]
    assert batch.rows == 3
    assert kwargs["capture_hidden_concat"].shape == (3, 0)
    assert result.telemetry is not None
    assert result.telemetry.candidate_counts == (2,)
    assert adapter.owner._row(_REQUEST_ID).mtp2_candidate_counts == [2]


def _snapshots_saved(proposer: _ChainProposer) -> list[SimpleNamespace]:
    return [proposer._snapshots[slot] for slot in sorted(proposer._snapshots)]


# ---------------------------------------------------------------------------
# Verify shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_verify_rows_follow_the_cycle(depth) -> None:
    drafted = _drafted_tokens(depth + 1)
    adapter = _adapter(depth=depth, proposer=_ChainProposer())

    _run_cycle(
        adapter,
        desired=depth,
        verify=_Verify(accepted_tokens=drafted[:depth], bonus=900, count=depth),
    )

    ((batch, kwargs),) = adapter.owner._session.verify_calls
    assert batch.rows == depth + 1
    assert kwargs["capture_hidden_concat"].shape == (depth + 1, 0)
    assert kwargs["capture_hidden_concat"].dtype == DType.BF16
    assert kwargs["chain_attn_mode"] == "decode_batched"
    assert kwargs["graph_mode"] == "off"
    assert kwargs["synchronize_after_commit"] is False
    # The verifier always reads the provider's device candidate rows.
    assert kwargs["candidate_token_ids_i32"].ptr == _DEVICE_CANDIDATES
    assert kwargs["candidate_token_ids_i32"].shape == (depth,)
    # The target is checkpointed into the spare slot before verify runs, and the
    # KV prefix is never copied: only the mutable recurrent state moves.
    assert adapter.owner._session.copies[0] == (0, 1, 0)
    assert adapter.owner._session.slot_positions[0] == (_ROOT_POSITION, 1)


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_prefill_reserves_verifier_scratch_for_the_cycle_depth(depth) -> None:
    row = _row(prompt_ids=(10, 11, 12), max_tokens=8)
    owner = _Owner(row=row)
    adapter = _adapter(depth=depth, owner=owner, proposer=_ChainProposer())
    adapter._states[_REQUEST_ID].prompt_rows_consumed = len(row.prompt_ids)

    adapter.observe_prefill_result(_REQUEST_ID)

    # rows = root + candidates, and the reserved context covers the highest
    # position this request's cycles can touch.
    assert owner._session.scratch == [
        (depth + 1, "decode_batched", len(row.prompt_ids) + 8 + depth)
    ]
    assert owner.released_captures == [row]

    owner.generator.execution_profile = "strict"
    adapter.observe_prefill_result(_REQUEST_ID)
    assert owner._session.scratch[-1] == (
        depth + 1,
        "c1_loop",
        len(row.prompt_ids) + 8 + depth,
    )


# ---------------------------------------------------------------------------
# Rollback, cancellation, and failure
# ---------------------------------------------------------------------------


def test_chain_step_failure_leaves_a_rollbackable_journal() -> None:
    proposer = _ChainProposer(fail_chain_at=2)
    adapter = _adapter(depth=3, proposer=proposer)
    plan = _plan(adapter, desired=3)

    with pytest.raises(RuntimeError, match="MTP chain step failed"):
        adapter.propose_batch(plan, (_semantics(),))

    state = adapter._states[_REQUEST_ID]
    assert state.journal is not None
    assert state.checkpoint is not None
    # The failed proposal published no repair prefix.
    assert state.draft_snapshots == ()

    adapter.rollback_cycle(plan, None, RuntimeError("chain"))

    assert ("restore", 0, _ROOT_POSITION) in proposer.calls
    assert proposer.position == _ROOT_POSITION
    assert proposer.cache_len == _ROOT_POSITION + 1
    assert state.journal is None
    assert state.checkpoint is None
    # Only the provider was captured before the failure, so only the provider is
    # restored; the target is untouched and owns no checkpoint yet.
    assert adapter.owner._session.copies == []
    assert adapter.owner.recorded_tokens == []


def test_rollback_leaves_the_pre_proposal_candidate_for_the_next_cycle() -> None:
    """A rolled-back cycle must not leave a later draft as the next candidate.

    Proposal overwrites the device candidate rows, so the following cycle can
    only chain correctly if the rollback restored the pre-proposal candidate
    identity along with the hidden row. Without that restore the second proposal
    would consume the last draft of the cancelled cycle instead of the primed
    candidate.
    """

    drafted = _drafted_tokens(4)
    proposer = _ChainProposer()
    adapter = _adapter(depth=3, proposer=proposer)
    plan = _plan(adapter, desired=3)

    adapter.propose_batch(plan, (_semantics(),))
    assert proposer.rows[:3] == list(drafted[:3])
    adapter.rollback_cycle(plan, None, RuntimeError("cancelled"))
    proposer.calls.clear()

    adapter.propose_batch(plan, (_semantics(),))

    assert proposer.kinds() == [
        "save",
        "ensure",
        "publish",
        "save",
        "step",
        "publish",
        "save",
        "step",
        "publish",
        "device_candidates",
    ]
    # The restored prefix owns its published candidate, so nothing is
    # re-published from the cancelled cycle's later rows.
    assert [call[1] for call in proposer.calls if call[0] == "publish"] == [0, 1, 2]
    consumed = [call[2] for call in proposer.calls if call[0] == "step"]
    assert consumed[0] == drafted[0]
    assert drafted[2] not in consumed
    assert proposer.position == _ROOT_POSITION + 2
    assert proposer.cache_len == _ROOT_POSITION + 3


def test_following_cycle_chains_from_the_reseeded_candidate() -> None:
    drafted = _drafted_tokens(6)
    proposer = _ChainProposer()
    adapter = _adapter(depth=3, proposer=proposer)

    # Accept one draft: the restored prefix sits on candidate 0, then the bonus
    # reseed scores a fresh candidate into row 0.
    _run_cycle(
        adapter,
        desired=3,
        verify=_Verify(accepted_tokens=drafted[:1], bonus=900, count=3),
    )
    reseeded = proposer.rows[0]
    assert reseeded == drafted[3]
    proposer.calls.clear()

    adapter.propose_batch(_plan(adapter, desired=3), (_semantics(),))

    # The reseed left no valid chain token, so this cycle publishes row 0 first,
    # and every step consumes the candidate the previous prefix published.
    assert proposer.kinds()[:3] == ["save", "ensure", "publish"]
    consumed = [call[2] for call in proposer.calls if call[0] == "step"]
    assert consumed == [reseeded, reseeded + 1]


def test_cancellation_before_verify_restores_both_owners_and_publishes_nothing() -> None:
    drafted = _drafted_tokens(4)
    proposer = _ChainProposer()
    adapter = _adapter(depth=3, proposer=proposer)
    row = adapter.owner._row(_REQUEST_ID)

    _, _, result = _run_cycle(
        adapter,
        desired=3,
        verify=_Verify(accepted_tokens=(), bonus=900, count=3),
        cancelled=lambda: (_REQUEST_ID,),
    )

    assert result.stage.value == "cancelled"
    assert result.transaction.rolled_back is True
    assert result.cancelled_request_ids == (_REQUEST_ID,)
    assert adapter.owner._session.verify_calls == []
    assert ("restore", 0, _ROOT_POSITION) in proposer.calls
    assert proposer.position == _ROOT_POSITION
    # The spare target slot is copied back and the cycle publishes nothing.
    assert adapter.owner._session.copies == [(0, 1, 0), (1, 0, 0)]
    assert adapter.owner.recorded_tokens == []
    assert row.mtp2_cycles == 0
    assert row.mtp2_candidate_counts == []
    assert row.mtp2_accepted_counts == []
    assert adapter._states[_REQUEST_ID].journal is None
    assert adapter._states[_REQUEST_ID].draft_snapshots == ()


def test_cancellation_after_the_repair_restores_both_owners_and_publishes_nothing() -> None:
    drafted = _drafted_tokens(4)
    proposer = _ChainProposer()
    adapter = _adapter(depth=3, proposer=proposer)
    row = adapter.owner._row(_REQUEST_ID)
    probes = iter(((), (_REQUEST_ID,)))

    _, _, result = _run_cycle(
        adapter,
        desired=3,
        verify=_Verify(accepted_tokens=drafted[:2], bonus=900, count=3),
        cancelled=lambda: next(probes, (_REQUEST_ID,)),
    )

    assert result.stage.value == "cancelled"
    assert len(adapter.owner._session.verify_calls) == 1
    # A cycle cancelled after the repair still never publishes its output.
    assert adapter.owner.recorded_tokens == []
    assert row.generated_steps == []
    assert row.mtp2_cycles == 0
    assert row.mtp2_candidate_counts == []
    assert ("restore", 0, _ROOT_POSITION) in proposer.calls
    assert proposer.position == _ROOT_POSITION
    assert adapter.owner._session.copies == [(0, 1, 0), (1, 0, 0)]


def test_provider_update_failure_restores_both_owners() -> None:
    drafted = _drafted_tokens(4)
    proposer = _ChainProposer(fail_bonus=True)
    adapter = _adapter(depth=3, proposer=proposer)
    row = adapter.owner._row(_REQUEST_ID)

    with pytest.raises(RuntimeError, match="MTP bonus reseed failed"):
        _run_cycle(
            adapter,
            desired=3,
            verify=_Verify(accepted_tokens=drafted[:1], bonus=900, count=3),
        )

    # Both owners return to their pre-proposal state and nothing is reported.
    assert ("restore", 0, _ROOT_POSITION) in proposer.calls
    assert proposer.position == _ROOT_POSITION
    assert adapter.owner._session.copies == [(0, 1, 0), (1, 0, 0)]
    assert adapter.owner.recorded_tokens == []
    assert row.mtp2_cycles == 0
    assert adapter._states[_REQUEST_ID].draft_snapshots == ()
    assert adapter._states[_REQUEST_ID].journal is None


# ---------------------------------------------------------------------------
# Provider capacity and pooling
# ---------------------------------------------------------------------------


def test_proposer_capacity_covers_the_chain_not_a_second_context_copy() -> None:
    assert _proposer_token_requirement(2, 8, 1) == 2 + 16 + 1 + 8
    assert _proposer_token_requirement(2, 8, 3) == 2 + 16 + 3 + 8
    # The chain rows are what crosses a bucket, not a second copy of the prompt.
    assert _proposer_capacity_bucket(_proposer_token_requirement(110, 68, 1)) == 256
    assert _proposer_capacity_bucket(_proposer_token_requirement(110, 68, 3)) == 512


def test_pool_reuses_a_proposer_that_covers_the_chain(monkeypatch) -> None:
    session = _Session()
    required = _proposer_token_requirement(2, 8, 3)
    pooled = _ChainProposer()
    pooled.max_mtp_tokens = _proposer_capacity_bucket(required)
    pooled.scoring_head = SimpleNamespace(owner=session)
    owner = _Owner(session=session)
    adapter = Qwen35ParoMTP2Adapter(owner, candidate_budget=3)
    adapter._proposer_pool.append(pooled)
    adapter.register_request(_REQUEST_ID, 3)
    built: list[dict] = []

    def _build(*_args, **kwargs):
        built.append(kwargs)
        return _ChainProposer()

    monkeypatch.setattr(paro_mtp2_module, "NativeMtpChainProposer", _build)
    adapter.begin_prompt(_REQUEST_ID)

    assert built == []
    assert pooled.closed is False
    assert adapter._proposer_reuses == 1
    assert adapter._proposer_builds == 0
    assert adapter._states[_REQUEST_ID].proposer is pooled


def test_pool_rebuilds_a_proposer_that_cannot_hold_the_chain(monkeypatch) -> None:
    session = _Session()
    required = _proposer_token_requirement(2, 8, 3)
    shallow = _ChainProposer()
    shallow.max_mtp_tokens = required - 1
    shallow.scoring_head = SimpleNamespace(owner=session)
    owner = _Owner(session=session)
    adapter = Qwen35ParoMTP2Adapter(owner, candidate_budget=3)
    adapter._proposer_pool.append(shallow)
    adapter.register_request(_REQUEST_ID, 3)
    built: list[dict] = []

    def _build(*_args, **kwargs):
        built.append(kwargs)
        replacement = _ChainProposer(
            max_candidate_rows=int(kwargs["max_candidate_rows"])
        )
        replacement.max_mtp_tokens = kwargs["max_mtp_tokens"]
        replacement.scoring_head = kwargs["scoring_head"]
        return replacement

    monkeypatch.setattr(paro_mtp2_module, "NativeMtpChainProposer", _build)
    adapter.begin_prompt(_REQUEST_ID)

    # The shallow proposer is closed rather than silently reused, and the
    # replacement is sized from the whole chain.
    assert shallow.closed is True
    assert len(built) == 1
    assert built[0]["max_mtp_tokens"] == _proposer_capacity_bucket(required)
    assert adapter._proposer_builds == 1
    assert adapter._proposer_reuses == 0
    assert adapter._states[_REQUEST_ID].proposer.max_mtp_tokens == _proposer_capacity_bucket(
        required
    )


def test_pool_does_not_reuse_a_shallow_candidate_row_proposer(monkeypatch) -> None:
    """A proposer that cannot hold the chain's candidate rows is not reused.

    Capacity in tokens is not enough: draft ``i`` is scored straight into device
    row ``i``, so a proposer built for a one-candidate request would overwrite
    its own row on a depth-three chain.
    """

    session = _Session()
    required = _proposer_token_requirement(2, 8, 3)
    shallow = _ChainProposer(max_candidate_rows=1)
    shallow.max_mtp_tokens = _proposer_capacity_bucket(required)
    shallow.scoring_head = SimpleNamespace(owner=session)
    owner = _Owner(session=session)
    adapter = Qwen35ParoMTP2Adapter(owner, candidate_budget=3)
    adapter._proposer_pool.append(shallow)
    adapter.register_request(_REQUEST_ID, 3)
    built: list[dict] = []

    def _build(*_args, **kwargs):
        built.append(kwargs)
        replacement = _ChainProposer(
            max_candidate_rows=int(kwargs["max_candidate_rows"])
        )
        replacement.max_mtp_tokens = kwargs["max_mtp_tokens"]
        replacement.scoring_head = kwargs["scoring_head"]
        return replacement

    monkeypatch.setattr(paro_mtp2_module, "NativeMtpChainProposer", _build)
    adapter.begin_prompt(_REQUEST_ID)

    assert shallow.closed is True
    assert len(built) == 1
    assert built[0]["max_candidate_rows"] == 3
    assert built[0]["max_mtp_tokens"] == _proposer_capacity_bucket(required)
    assert built[0]["max_positions"] == session.max_sequence_length
    assert adapter._proposer_builds == 1
    assert adapter._states[_REQUEST_ID].proposer.max_candidate_rows == 3


def test_observability_reports_the_request_lifecycle() -> None:
    adapter = _adapter(depth=2, proposer=_ChainProposer())

    snapshot = adapter.observability_snapshot()

    assert snapshot["enabled"] is True
    assert snapshot["intents"] == 1
    assert snapshot["states"] == 1
    assert snapshot["active_claim"] is False
    assert snapshot["disabled_requests"] == []
    assert snapshot["bulk_prompt_chunks"] == 0
    assert snapshot["bulk_prompt_rows"] == 0

    adapter.close()

    assert adapter.observability_snapshot()["states"] == 0
    assert adapter.observability_snapshot()["intents"] == 0
