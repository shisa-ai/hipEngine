from __future__ import annotations

import pytest

from hipengine.speculative import (
    DFLASH_CHAIN_CANDIDATE_BUDGETS,
    DFlashChainCompiler,
    DFlashDraftRequest,
    TargetAcceptSummary,
    TargetVerifyBatch,
    compile_dflash_chain,
)


def _single_request_target() -> TargetVerifyBatch:
    request = DFlashDraftRequest.from_root_prefixed(
        request_id=7,
        root_position=8,
        token_ids=(30, 31, 32),
        expected_root_token=30,
    )
    draft = compile_dflash_chain([request], candidate_budget=2)

    assert draft.request_ids == (7,)
    assert draft.candidate_tokens == (31, 32)
    assert draft.parent_positions == (8, 9)
    assert draft.draft_depths == (1, 2)
    assert draft.row_to_request == (7, 7)
    assert draft.active_mask == (True, True)
    assert draft.mode == "verify_chain"

    target = TargetVerifyBatch.from_draft(draft, root_tokens=(30,), root_positions=(8,))
    assert target.tokens == (30, 31, 32)
    assert target.root_rows == (0,)
    assert target.candidate_rows == (1, 2)
    assert target.parent_rows == (-1, 0, 1)
    assert target.positions == (8, 9, 10)
    return target


def test_dflash_chain_compiler_reject_at_root_acceptance() -> None:
    target = _single_request_target()

    result = target.accept_from_top1((99, 31, 32), transaction_id=11)

    assert result.transaction_id == 11
    assert result.accepted_counts == (0,)
    assert result.accepted_tokens == ((),)
    assert result.selected_candidate_rows == (0,)
    assert result.next_tokens == (99,)
    summary = TargetAcceptSummary.from_accept_result(target, result)
    assert summary.commit_rows == (0,)
    assert summary.commit_tokens == (30,)
    assert summary.commit_positions == (8,)
    assert summary.full_accept == (False,)


def test_dflash_chain_compiler_partial_acceptance() -> None:
    target = _single_request_target()

    result = target.accept_from_top1((31, 99, 44))

    assert result.accepted_counts == (1,)
    assert result.accepted_tokens == ((31,),)
    assert result.selected_candidate_rows == (1,)
    assert result.next_tokens == (99,)
    summary = TargetAcceptSummary.from_accept_result(target, result)
    assert summary.commit_rows == (1,)
    assert summary.commit_tokens == (31,)
    assert summary.commit_positions == (9,)
    assert summary.full_accept == (False,)


def test_dflash_chain_compiler_full_acceptance() -> None:
    target = _single_request_target()

    result = target.accept_from_top1((31, 32, 44))

    assert result.accepted_counts == (2,)
    assert result.accepted_tokens == ((31, 32),)
    assert result.selected_candidate_rows == (2,)
    assert result.next_tokens == (44,)
    summary = TargetAcceptSummary.from_accept_result(target, result)
    assert summary.commit_rows == (2,)
    assert summary.commit_tokens == (32,)
    assert summary.commit_positions == (10,)
    assert summary.full_accept == (True,)


def test_dflash_chain_compiler_multi_request_row_mapping_has_no_root_rows() -> None:
    draft = compile_dflash_chain(
        [
            DFlashDraftRequest(request_id=10, root_position=5, candidate_tokens=(101, 102), active_count=2),
            DFlashDraftRequest.from_root_prefixed(
                request_id=20,
                root_position=12,
                token_ids=(2000, 201, 202, 203),
                expected_root_token=2000,
            ),
        ],
        candidate_budget=4,
        pad_token_id=248070,
    )

    assert draft.request_ids == (10, 20)
    assert draft.draft_rows == 8
    assert draft.candidate_tokens == (101, 102, 248070, 248070, 201, 202, 203, 248070)
    assert draft.parent_positions == (5, 6, 7, 8, 12, 13, 14, 15)
    assert draft.draft_depths == (1, 2, 3, 4, 1, 2, 3, 4)
    assert draft.row_to_request == (10, 10, 10, 10, 20, 20, 20, 20)
    assert draft.active_mask == (True, True, False, False, True, True, True, False)
    assert draft.mode == "verify_chain"

    target = TargetVerifyBatch.from_draft(draft, root_tokens=(1000, 2000), root_positions=(5, 12))

    assert target.rows == 10
    assert target.root_rows == (0, 1)
    assert target.candidate_rows == (2, 3, 4, 5, 6, 7, 8, 9)
    assert target.tokens == (1000, 2000, 101, 102, 248070, 248070, 201, 202, 203, 248070)
    assert target.positions == (5, 12, 6, 7, 8, 9, 13, 14, 15, 16)
    assert target.row_to_request == (10, 20, 10, 10, 10, 10, 20, 20, 20, 20)
    assert target.parent_rows == (-1, -1, 0, 2, 3, 4, 1, 6, 7, 8)
    assert target.draft_depths == (0, 0, 1, 2, 3, 4, 1, 2, 3, 4)
    assert target.active_mask == (True, True, True, True, False, False, True, True, True, False)
    assert target.candidate_counts == (4, 4)
    assert target.tree_shape == (0, 1, 2, 3, 0, 5, 6, 7)


def test_dflash_chain_compiler_validates_budget_and_root_prefixed_output() -> None:
    assert DFLASH_CHAIN_CANDIDATE_BUDGETS == (2, 4, 8)
    with pytest.raises(ValueError, match="candidate_budget"):
        DFlashChainCompiler(candidate_budget=3)
    with pytest.raises(ValueError, match="expected root"):
        DFlashDraftRequest.from_root_prefixed(
            request_id=1,
            root_position=0,
            token_ids=(10, 11),
            expected_root_token=9,
        )
    with pytest.raises(ValueError, match="active_count"):
        compile_dflash_chain(
            [DFlashDraftRequest(request_id=1, root_position=0, candidate_tokens=(11,), active_count=2)],
            candidate_budget=2,
        )
