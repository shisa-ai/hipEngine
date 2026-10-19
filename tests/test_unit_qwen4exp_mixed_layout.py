"""Immutable host layout for one shared target pass: AR rows plus chains.

These tests pin the M1 layout prerequisite only. They cover ragged chains, zero
to many candidates per request, autoregressive peers, sparse physical slots,
arbitrary physical row permutations, every accepted depth, rejection and bonus
destinations, duplicate ownership/slot/row metadata, and the rule that a
validation failure mutates nothing. No kernel, runner or owner is exercised.
"""

from __future__ import annotations

import dataclasses

import pytest

from hipengine.runtime.qwen4_exp_mixed_layout import (
    Qwen4ExpMixedLayout,
    Qwen4ExpMixedRequest,
    Qwen4ExpMixedRow,
    compile_mixed_layout,
    mixed_request_from_proposal,
)
from hipengine.speculative.interfaces import TargetCommitPlan, TargetVerifyBatch


def _request(
    request_id: int,
    *,
    slot: int | None = None,
    owner: int | None = None,
    start: int = 2,
    root: int = 10,
    candidates=(),
) -> Qwen4ExpMixedRequest:
    return Qwen4ExpMixedRequest(
        request_id=request_id,
        slot=request_id if slot is None else slot,
        owner_token=request_id + 100 if owner is None else owner,
        start_position=start,
        root_token=root,
        candidates=tuple(candidates),
    )


def _mixed_layout(*, physical_row_order=None) -> Qwen4ExpMixedLayout:
    """Request 1 speculates two tokens, request 2 is an AR peer, 3 one token."""

    return compile_mixed_layout(
        [
            _request(1, slot=5, start=2, root=10, candidates=(11, 12)),
            _request(2, slot=0, start=40, root=20),
            _request(3, slot=7, start=9, root=30, candidates=(31,)),
        ],
        physical_row_order=physical_row_order,
    )


# -- physical width versus total rows ---------------------------------------


def test_layout_separates_physical_request_width_from_total_rows() -> None:
    layout = _mixed_layout()

    assert layout.physical_request_width == 3
    assert layout.total_rows == 6
    assert layout.max_chain_rows == 3
    assert layout.request_ids == (1, 2, 3)
    assert layout.autoregressive_request_ids == (2,)
    assert layout.verify_batch.rows == layout.total_rows == 6


@pytest.mark.parametrize("depth", [0, 1, 2, 4])
def test_zero_one_and_multiple_candidates(depth: int) -> None:
    candidates = tuple(100 + index for index in range(depth))
    layout = compile_mixed_layout([_request(7, start=5, root=9, candidates=candidates)])

    assert layout.physical_request_width == 1
    assert layout.total_rows == depth + 1
    rows = layout.rows_for_request(7)
    assert [row.depth for row in rows] == list(range(depth + 1))
    assert [row.position for row in rows] == [5 + index for index in range(depth + 1)]
    assert [row.token for row in rows] == [9, *candidates]
    assert layout.verify_batch.candidate_counts == (depth,)
    assert layout.verify_batch.draft_depth == depth


def test_ragged_chains_keep_a_contiguous_semantic_chain_order() -> None:
    layout = compile_mixed_layout(
        [
            _request(1, start=2, root=10, candidates=(11, 12, 13)),
            _request(2, start=8, root=20),
            _request(3, start=20, root=30, candidates=(31,)),
        ]
    )

    assert [row.semantic_row for row in layout.rows] == [0, 1, 2, 3, 4, 5, 6]
    assert [row.request_id for row in layout.rows] == [1, 1, 1, 1, 2, 3, 3]
    assert [row.depth for row in layout.rows] == [0, 1, 2, 3, 0, 0, 1]
    assert [row.position for row in layout.rows] == [2, 3, 4, 5, 8, 20, 21]
    assert layout.verify_batch.candidate_counts == (3, 0, 1)
    assert layout.max_chain_rows == 4


def test_autoregressive_peers_are_single_root_rows() -> None:
    layout = compile_mixed_layout(
        [
            Qwen4ExpMixedRequest.autoregressive(
                request_id=1, slot=1, owner_token=101, start_position=2, root_token=10
            ),
            _request(2, start=6, root=20, candidates=(21,)),
        ]
    )
    peer = layout.request(1)
    rows = layout.rows_for_request(1)

    assert peer.is_autoregressive is True
    assert peer.depth == 0
    assert len(rows) == 1
    assert rows[0].depth == 0
    assert rows[0].token == 10
    assert rows[0].position == 2
    assert rows[0].context == 3
    # The generic verifier sees the peer as a root row only.
    assert layout.verify_batch.root_rows[0] == 0
    assert layout.verify_batch.candidate_rows == (2,)
    assert layout.verify_batch.draft_depths[0] == 0
    assert layout.verify_batch.row_to_request == (1, 2, 2)
    assert layout.verify_batch.candidate_counts == (0, 1)


def test_sparse_physical_slots_are_preserved_verbatim() -> None:
    layout = compile_mixed_layout(
        [
            _request(1, slot=9, owner=901, start=2, root=10),
            _request(2, slot=0, owner=902, start=4, root=20, candidates=(21,)),
            _request(3, slot=33, owner=903, start=6, root=30),
        ]
    )

    assert [request.slot for request in layout.requests] == [9, 0, 33]
    assert [row.slot for row in layout.rows] == [9, 0, 0, 33]
    assert [row.owner_token for row in layout.rows] == [901, 902, 902, 903]


# -- physical row permutations ---------------------------------------------


def test_physical_row_permutation_maps_back_to_request_and_depth() -> None:
    # Physical rows: 2/root, 1/root, 3/root, 1/depth1, 3/depth1, 1/depth2.
    layout = _mixed_layout(physical_row_order=(3, 0, 4, 1, 5, 2))

    assert layout.total_rows == 6
    assert [
        (row.request_id, row.depth) for row in layout.rows
    ] == [(2, 0), (1, 0), (3, 0), (1, 1), (3, 1), (1, 2)]
    assert [row.semantic_row for row in layout.rows] == [3, 0, 4, 1, 5, 2]
    for physical, row in enumerate(layout.rows):
        assert row.physical_row == physical
        assert layout.row_at(physical) is row
        assert layout.physical_row(row.request_id, row.depth) == physical
    assert layout.physical_row(1, 2) == 5
    assert layout.semantic_row(1, 2) == 2
    assert layout.verify_batch.positions == (40, 2, 9, 3, 10, 4)
    assert layout.verify_batch.tokens == (20, 10, 30, 11, 31, 12)
    assert layout.verify_batch.parent_rows == (-1, -1, -1, 1, 2, 3)


def test_permutation_that_places_a_candidate_before_its_parent_is_refused() -> None:
    with pytest.raises(ValueError, match="parent"):
        _mixed_layout(physical_row_order=(1, 0, 2, 3, 4, 5))


def test_row_permutation_must_be_a_permutation() -> None:
    with pytest.raises(ValueError, match="physical_row_order"):
        _mixed_layout(physical_row_order=(0, 1, 2))
    with pytest.raises(ValueError, match="physical_row_order"):
        _mixed_layout(physical_row_order=(0, 0, 1, 2, 3, 4))
    with pytest.raises(ValueError, match="physical_row_order"):
        _mixed_layout(physical_row_order=(0, 1, 2, 3, 4, 6))


# -- commit destinations ----------------------------------------------------


@pytest.mark.parametrize(
    ("accepted", "row", "token", "position"),
    [(0, 0, 10, 2), (1, 1, 11, 3), (2, 2, 12, 4)],
)
def test_each_accepted_depth_selects_the_matching_row(
    accepted: int, row: int, token: int, position: int
) -> None:
    layout = compile_mixed_layout(
        [_request(1, start=2, root=10, candidates=(11, 12))]
    )

    plan = layout.commit_destinations((accepted,), transaction_id=4)

    assert isinstance(plan, TargetCommitPlan)
    assert plan.transaction_id == 4
    assert plan.request_ids == (1,)
    assert plan.accepted_counts == (accepted,)
    assert plan.commit_rows == (row,)
    assert plan.commit_tokens == (token,)
    assert plan.commit_positions == (position,)
    assert plan.candidate_counts == (2,)
    assert plan.draft_depth == 2
    assert plan.mode == "verify_chain"


def test_autoregressive_peer_advances_exactly_once() -> None:
    layout = compile_mixed_layout(
        [
            _request(1, start=2, root=10, candidates=(11, 12)),
            _request(2, slot=4, start=30, root=20),
        ]
    )

    plan = layout.commit_destinations((1, 0))

    # The AR peer commits its own root row, one token, one position.
    assert plan.commit_rows[1] == layout.physical_row(2, 0)
    assert plan.commit_tokens[1] == 20
    assert plan.commit_positions[1] == 30
    assert plan.candidate_counts[1] == 0
    with pytest.raises(ValueError, match="autoregressive"):
        layout.commit_destinations((1, 1))


def test_rejection_and_bonus_destinations() -> None:
    layout = compile_mixed_layout(
        [_request(1, start=2, root=10, candidates=(11, 12))]
    )

    rejected = layout.commit_destinations((0,), next_tokens=(99,))
    assert rejected.commit_rows == (0,)
    assert rejected.commit_tokens == (10,)
    assert rejected.next_tokens == (99,)

    bonus = layout.commit_destinations((2,), next_tokens=(13,))
    assert bonus.commit_rows == (2,)
    assert bonus.commit_tokens == (12,)
    assert bonus.commit_positions == (4,)
    assert bonus.next_tokens == (13,)


def test_commit_destinations_validate_counts_before_selecting_rows() -> None:
    layout = _mixed_layout()

    with pytest.raises(ValueError, match="accepted_counts"):
        layout.commit_destinations((2,))
    with pytest.raises(ValueError, match="accepted_counts"):
        layout.commit_destinations((3, 0, 1))
    with pytest.raises(ValueError, match="accepted_counts"):
        layout.commit_destinations((2, 0, -1))
    with pytest.raises(TypeError, match="accepted_counts"):
        layout.commit_destinations((2.5, 0, 1))
    with pytest.raises(ValueError, match="next_tokens"):
        layout.commit_destinations((2, 0, 1), next_tokens=(1, 2))


def test_commit_destinations_follow_the_physical_permutation() -> None:
    layout = _mixed_layout(physical_row_order=(3, 0, 4, 1, 5, 2))

    plan = layout.commit_destinations((2, 0, 1))

    assert plan.commit_rows == (5, 0, 4)
    assert plan.commit_tokens == (12, 20, 31)
    assert plan.commit_positions == (4, 40, 10)


# -- duplicate and invalid metadata ----------------------------------------


def test_duplicate_request_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="request id"):
        compile_mixed_layout(
            [_request(1, slot=1), _request(1, slot=2, owner=200)]
        )


def test_duplicate_slots_are_refused() -> None:
    with pytest.raises(ValueError, match="slot"):
        compile_mixed_layout(
            [_request(1, slot=3), _request(2, slot=3, owner=200)]
        )


def test_duplicate_owner_tokens_are_refused() -> None:
    with pytest.raises(ValueError, match="owner"):
        compile_mixed_layout(
            [_request(1, slot=1, owner=77), _request(2, slot=2, owner=77)]
        )


def test_an_empty_request_set_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one request"):
        compile_mixed_layout([])


def test_an_unowned_owner_token_is_refused() -> None:
    # The batch owner starts every slot at 0 and admit() returns 1..N, so 0 is
    # the unowned sentinel and never a live admission.
    with pytest.raises(ValueError, match="unowned"):
        Qwen4ExpMixedRequest(
            request_id=1, slot=0, owner_token=0, start_position=2, root_token=10
        )
    with pytest.raises(ValueError, match="unowned"):
        Qwen4ExpMixedRequest.autoregressive(
            request_id=1, slot=0, owner_token=0, start_position=2, root_token=10
        )
    with pytest.raises(ValueError, match="unowned"):
        Qwen4ExpMixedRow(
            physical_row=0,
            semantic_row=0,
            request_id=1,
            slot=0,
            owner_token=0,
            depth=0,
            position=2,
            token=10,
        )


def test_the_first_admitted_owner_token_is_accepted() -> None:
    layout = compile_mixed_layout(
        [_request(1, slot=0, owner=1, start=2, root=10, candidates=(11,))]
    )

    assert layout.request(1).owner_token == 1
    assert [row.owner_token for row in layout.rows] == [1, 1]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"start_position": -1}, "start_position"),
        ({"root_token": -2}, "root_token"),
        ({"candidates": (-1,)}, "candidate"),
        ({"slot": -1}, "slot"),
        ({"owner_token": -1}, "owner_token"),
        ({"request_id": -1}, "request_id"),
        ({"slot": True}, "slot"),
        ({"candidates": 7}, "candidate"),
    ],
)
def test_invalid_request_metadata_is_refused(overrides, match) -> None:
    kwargs = {
        "request_id": 1,
        "slot": 1,
        "owner_token": 101,
        "start_position": 2,
        "root_token": 10,
        "candidates": (),
    }
    kwargs.update(overrides)

    with pytest.raises((TypeError, ValueError), match=match):
        Qwen4ExpMixedRequest(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"position": 99}, "position"),
        ({"token": 99}, "token"),
        ({"depth": 3}, "depth"),
        ({"slot": 99}, "slot"),
        ({"owner_token": 99}, "owner"),
        ({"physical_row": 9}, "physical_row"),
    ],
)
def test_invalid_row_metadata_is_refused(overrides, match) -> None:
    layout = _mixed_layout()
    broken = dataclasses.replace(layout.rows[1], **overrides)

    with pytest.raises(ValueError, match=match):
        Qwen4ExpMixedLayout(
            requests=layout.requests,
            rows=(layout.rows[0], broken, *layout.rows[2:]),
        )


def test_rows_must_cover_every_request_and_depth_exactly_once() -> None:
    layout = _mixed_layout()

    with pytest.raises(ValueError, match="row"):
        Qwen4ExpMixedLayout(requests=layout.requests, rows=layout.rows[:-1])

    with pytest.raises(ValueError, match="semantic"):
        Qwen4ExpMixedLayout(
            requests=layout.requests,
            rows=(
                dataclasses.replace(layout.rows[0], semantic_row=1),
                dataclasses.replace(layout.rows[1], semantic_row=0),
                *layout.rows[2:],
            ),
        )


def test_a_request_cannot_gain_a_row_it_did_not_prepare() -> None:
    layout = compile_mixed_layout([_request(1, start=2, root=10)])
    peer = dataclasses.replace(layout.requests[0], candidates=(11,))

    with pytest.raises(ValueError, match="row"):
        Qwen4ExpMixedLayout(requests=(peer,), rows=layout.rows)


def test_a_row_cannot_name_a_request_outside_the_pass() -> None:
    layout = compile_mixed_layout([_request(1, start=2, root=10)])
    stray = dataclasses.replace(layout.rows[0], request_id=2)

    with pytest.raises(ValueError, match="request 2"):
        Qwen4ExpMixedLayout(requests=layout.requests, rows=(stray,))


# -- immutability -----------------------------------------------------------


def test_a_validation_failure_mutates_nothing() -> None:
    requests = [
        _request(1, slot=5, start=2, root=10, candidates=(11, 12)),
        _request(2, slot=0, start=40, root=20),
    ]
    layout = compile_mixed_layout(requests)
    rows_before = layout.rows
    batch_before = layout.verify_batch
    requests_before = tuple(requests)

    with pytest.raises(ValueError):
        compile_mixed_layout(requests, physical_row_order=(0, 0, 1))
    with pytest.raises(ValueError):
        layout.commit_destinations((3, 0))

    assert tuple(requests) == requests_before
    assert layout.rows is rows_before
    assert layout.verify_batch is batch_before
    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.requests = ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.rows[0].token = 0


# -- generic verifier compatibility ----------------------------------------


def test_require_verifier_compatible_returns_the_generic_row_contract() -> None:
    layout = _mixed_layout()
    batch = layout.require_verifier_compatible()

    assert isinstance(batch, TargetVerifyBatch)
    assert batch is layout.verify_batch
    assert batch.request_ids == (1, 2, 3)
    assert batch.root_rows == (0, 3, 4)
    assert batch.candidate_rows == (1, 2, 5)
    assert batch.draft_depths == (0, 1, 2, 0, 0, 1)
    assert batch.active_mask == (True,) * 6
    assert batch.candidate_counts == (2, 0, 1)
    assert batch.draft_depth == 2
    assert batch.mode == "verify_chain"
    # A mixed pass still has candidate rows, so the generic work projection
    # exists. An all-AR layout cannot project one (see the layout docstring).
    work = batch.to_work_item()
    assert work.row_to_request == (1, 1, 3)


def test_all_autoregressive_layout_has_no_generic_work_projection() -> None:
    layout = compile_mixed_layout([_request(1), _request(2, slot=3, owner=200)])

    assert layout.autoregressive_request_ids == (1, 2)
    assert layout.verify_batch.candidate_rows == ()
    assert layout.verify_batch.candidate_counts == (0, 0)
    with pytest.raises(ValueError, match="row_to_request"):
        layout.verify_batch.to_work_item()


def test_require_verifier_compatible_refuses_a_tampered_batch() -> None:
    layout = _mixed_layout()
    other = compile_mixed_layout(
        [_request(1, start=2, root=10, candidates=(11, 12))]
    )
    object.__setattr__(layout, "verify_batch", other.verify_batch)

    with pytest.raises(ValueError, match="verifier"):
        layout.require_verifier_compatible()


# -- prepared-proposal adapter --------------------------------------------


def test_a_prepared_proposal_becomes_a_request_group() -> None:
    from hipengine.generation.qwen4_exp_mtp_stages import Qwen4ExpMTPProposal

    proposal = Qwen4ExpMTPProposal(
        start_position=6, root_token=40, candidates=(41, 42, 43)
    )

    group = mixed_request_from_proposal(
        proposal, request_id=12, slot=3, owner_token=778
    )

    assert isinstance(group, Qwen4ExpMixedRequest)
    assert group.request_id == 12
    assert group.slot == 3
    assert group.owner_token == 778
    assert group.start_position == 6
    assert group.root_token == 40
    assert group.candidates == (41, 42, 43)
    assert group.depth == 3
    assert group.rows == 4


def test_a_proposal_adapter_refuses_a_missing_chain() -> None:
    with pytest.raises(TypeError, match="proposal"):
        mixed_request_from_proposal(
            object(), request_id=1, slot=0, owner_token=1
        )
