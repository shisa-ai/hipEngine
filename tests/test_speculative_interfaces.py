from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.dispatch import ActiveBatch, RequestState
from hipengine.kvcache import FixedPagedKVPolicy
from hipengine.speculative import AcceptResult, DraftBatch, TargetAcceptSummary, TargetCommitPlan, TargetVerifyBatch
from scripts.qwen35_dflash_ddtree_blocker import build_payload


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def test_draft_batch_and_accept_result_validate_row_metadata() -> None:
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        active_mask=(True, True, True),
    )
    assert draft.draft_rows == 3
    assert draft.kind == "verify_chain"

    result = AcceptResult(request_ids=(1, 2), accepted_counts=(2, 1), accepted_tokens=((10, 11), (20,)))
    assert result.accepted_tokens == ((10, 11), (20,))

    with pytest.raises(ValueError, match="align"):
        DraftBatch(
            request_ids=(1,),
            candidate_tokens=(10,),
            parent_positions=(),
            draft_depths=(1,),
            row_to_request=(1,),
        )
    with pytest.raises(ValueError, match="lengths"):
        AcceptResult(request_ids=(1,), accepted_counts=(2,), accepted_tokens=((10,),))


def test_target_verify_batch_materializes_root_and_candidate_rows() -> None:
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
        active_mask=(True, True, False),
    )

    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100, 200), root_positions=(5, 3))

    assert target.rows == 5
    assert target.candidate_count == 3
    assert target.request_ids == (1, 2)
    assert target.tokens == (100, 200, 10, 11, 20)
    assert target.positions == (5, 3, 6, 7, 4)
    assert target.row_to_request == (1, 2, 1, 1, 2)
    assert target.root_rows == (0, 1)
    assert target.candidate_rows == (2, 3, 4)
    assert target.parent_rows == (-1, -1, 0, 2, 1)
    assert target.draft_depths == (0, 0, 1, 2, 1)
    assert target.active_mask == (True, True, True, True, False)
    assert target.candidate_counts == (2, 1)
    assert target.draft_depth == 2
    assert target.tree_shape == (0, 1, 0)
    assert target.mode == "verify_tree"

    chain = TargetVerifyBatch.from_draft(
        DraftBatch(
            request_ids=(7,),
            candidate_tokens=(31, 32, 33),
            parent_positions=(8, 9, 10),
            draft_depths=(1, 2, 3),
            row_to_request=(7, 7, 7),
        ),
        root_tokens=(30,),
        root_positions=(8,),
    )
    assert chain.parent_rows == (-1, 0, 1, 2)
    assert chain.positions == (8, 9, 10, 11)
    assert chain.tree_shape == (0, 1, 2)


def test_target_verify_batch_builds_graph_shape_key_from_active_batch() -> None:
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100, 200), root_positions=(5, 3))
    active = ActiveBatch(2)
    active.admit(RequestState(request_id=1, prompt_tokens=(1, 2, 3, 4, 5), max_new_tokens=4, next_prompt_index=5))
    active.admit(RequestState(request_id=2, prompt_tokens=(6, 7, 8), max_new_tokens=4, next_prompt_index=3))

    key = target.shape_key(active, context_bucket_size=4, top_k=8, experts_per_token=8, replay_steps=2)

    assert key.mode.value == "verify_tree"
    assert key.active_c == 2
    assert key.context_bucket == 8
    assert key.active_mask == (True, True)
    assert key.top_k == 8
    assert key.experts_per_token == 8
    assert key.replay_steps == 2
    assert key.draft_depth == 2
    assert key.tree_shape == (0, 1, 0)


def test_target_verify_batch_projects_candidate_rows_to_work_item() -> None:
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100, 200), root_positions=(5, 3))

    work = target.to_work_item()

    assert work.kind.value == "verify_tree"
    assert work.request_ids == (1, 2)
    assert work.row_to_request == (1, 1, 2)
    assert work.token_rows == ((10,), (11,), (20,))
    assert work.draft_depth == 2
    assert work.tree_parents == (0, 1, 0)


def test_target_verify_batch_selects_commit_rows_from_accept_counts() -> None:
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100, 200), root_positions=(5, 3))

    assert target.candidate_counts == (2, 1)
    selected = target.select_commit_rows((2, 1))
    assert selected.request_ids == (1, 2)
    assert selected.accepted_counts == (2, 1)
    assert selected.selected_rows == (3, 4)
    assert selected.selected_tokens == (11, 20)
    assert selected.selected_positions == (7, 4)
    assert selected.mode == "verify_tree"

    zero = target.select_commit_rows((0, 0))
    assert zero.selected_rows == (0, 1)
    assert zero.selected_tokens == (100, 200)
    assert zero.selected_positions == (5, 3)


def test_target_accept_summary_validates_paths_and_commit_rows() -> None:
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100, 200), root_positions=(5, 3))

    summary = TargetAcceptSummary.from_accept_result(
        target,
        AcceptResult(request_ids=(1, 2), accepted_counts=(2, 1), accepted_tokens=((10, 11), (20,))),
    )

    assert summary.request_ids == (1, 2)
    assert summary.accepted_counts == (2, 1)
    assert summary.accepted_tokens == ((10, 11), (20,))
    assert summary.commit_rows == (3, 4)
    assert summary.commit_tokens == (11, 20)
    assert summary.commit_positions == (7, 4)
    assert summary.full_accept == (True, True)
    assert summary.mode == "verify_tree"

    zero = TargetAcceptSummary.from_accept_result(
        target,
        AcceptResult(request_ids=(1, 2), accepted_counts=(0, 0), accepted_tokens=((), ())),
    )
    assert zero.commit_rows == (0, 1)
    assert zero.commit_tokens == (100, 200)
    assert zero.commit_positions == (5, 3)
    assert zero.full_accept == (False, False)

    with pytest.raises(ValueError, match="selected target verify paths"):
        TargetAcceptSummary.from_accept_result(
            target,
            AcceptResult(request_ids=(1, 2), accepted_counts=(2, 1), accepted_tokens=((10, 12), (20,))),
        )


def test_target_commit_plan_binds_accept_summary_to_kv_transaction() -> None:
    policy = FixedPagedKVPolicy()
    for request_id, ptr in [(1, 0x1000), (2, 0x2000)]:
        policy.register(
            request_id,
            block_table=_tensor(ptr, (4,), "int32"),
            live_counts=_tensor(ptr + 0x100, (1,), "int64"),
            max_live_count=4,
        )
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100, 200), root_positions=(5, 3))
    summary = TargetAcceptSummary.from_accept_result(
        target,
        AcceptResult(request_ids=(1, 2), accepted_counts=(2, 1), accepted_tokens=((10, 11), (20,))),
    )
    txn = policy.begin_transaction((1, 2), target)

    plan = TargetCommitPlan.from_summary(summary, txn)

    assert plan.transaction_id == txn.transaction_id
    assert plan.request_ids == (1, 2)
    assert plan.accepted_counts == (2, 1)
    assert plan.kv_accept_counts == (2, 1)
    assert plan.commit_rows == (3, 4)
    assert plan.commit_tokens == (11, 20)
    assert plan.commit_positions == (7, 4)
    assert plan.candidate_counts == (2, 1)
    assert plan.mode == "verify_tree"
    committed = policy.commit(txn, plan.kv_accept_counts)
    assert committed.committed
    assert committed.accepted_counts == plan.accepted_counts

    with pytest.raises(ValueError, match="candidate_counts"):
        TargetCommitPlan(
            transaction_id=7,
            request_ids=(1,),
            accepted_counts=(2,),
            commit_rows=(3,),
            commit_tokens=(11,),
            commit_positions=(7,),
            candidate_counts=(1,),
        )


def test_target_verify_batch_requires_selected_rows_for_ambiguous_tree_depth() -> None:
    target = TargetVerifyBatch.from_draft(
        DraftBatch(
            request_ids=(1,),
            candidate_tokens=(10, 11),
            parent_positions=(5, 5),
            draft_depths=(1, 1),
            row_to_request=(1, 1),
            mode="verify_tree",
            tree_parents=(-1, -1),
        ),
        root_tokens=(100,),
        root_positions=(5,),
    )

    with pytest.raises(ValueError, match="ambiguous"):
        target.select_commit_rows((1,))
    selected = target.select_commit_rows((1,), selected_candidate_rows=(2,))
    assert selected.selected_rows == (2,)
    assert selected.selected_tokens == (11,)
    summary = TargetAcceptSummary.from_accept_result(
        target,
        AcceptResult(request_ids=(1,), accepted_counts=(1,), accepted_tokens=((11,),)),
        selected_candidate_rows=(2,),
    )
    assert summary.commit_rows == (2,)
    assert summary.accepted_tokens == ((11,),)
    with pytest.raises(ValueError, match="candidate row"):
        target.select_commit_rows((1,), selected_candidate_rows=(0,))


def test_target_verify_batch_validates_native_row_layout() -> None:
    with pytest.raises(ValueError, match="root tokens/positions"):
        TargetVerifyBatch.from_draft(
            DraftBatch(
                request_ids=(1, 2),
                candidate_tokens=(10,),
                parent_positions=(5,),
                draft_depths=(1,),
                row_to_request=(1,),
            ),
            root_tokens=(100,),
            root_positions=(5,),
        )
    with pytest.raises(ValueError, match="earlier candidate"):
        TargetVerifyBatch.from_draft(
            DraftBatch(
                request_ids=(1,),
                candidate_tokens=(10, 11),
                parent_positions=(5, 6),
                draft_depths=(1, 2),
                row_to_request=(1, 1),
                tree_parents=(1, 0),
            ),
            root_tokens=(100,),
            root_positions=(5,),
        )
    with pytest.raises(ValueError, match="root rows"):
        TargetVerifyBatch(
            request_ids=(1,),
            tokens=(100, 10),
            positions=(5, 6),
            row_to_request=(1, 1),
            parent_rows=(0, 0),
            root_rows=(0,),
            candidate_rows=(1,),
            draft_depths=(0, 1),
            active_mask=(True, True),
        )
    with pytest.raises(ValueError, match="same request"):
        TargetVerifyBatch(
            request_ids=(1, 2),
            tokens=(100, 200, 10),
            positions=(5, 3, 6),
            row_to_request=(1, 2, 1),
            parent_rows=(-1, -1, 1),
            root_rows=(0, 1),
            candidate_rows=(2,),
            draft_depths=(0, 0, 1),
            active_mask=(True, True, True),
        )


def test_speculative_draft_batch_drives_kv_transaction_commit_and_rollback() -> None:
    policy = FixedPagedKVPolicy()
    for request_id, ptr in [(1, 0x1000), (2, 0x2000)]:
        policy.register(
            request_id,
            block_table=_tensor(ptr, (4,), "int32"),
            live_counts=_tensor(ptr + 0x100, (1,), "int64"),
            max_live_count=4,
        )
    draft = DraftBatch(
        request_ids=(1, 2),
        candidate_tokens=(10, 11, 20),
        parent_positions=(5, 6, 3),
        draft_depths=(1, 2, 1),
        row_to_request=(1, 1, 2),
        mode="verify_tree",
        tree_parents=(-1, 0, -1),
    )

    txn = policy.begin_transaction([SimpleNamespace(request_id=1), SimpleNamespace(request_id=2)], draft)
    assert txn.request_ids == (1, 2)
    assert txn.draft_rows == 3
    assert txn.candidate_counts == (2, 1)
    assert txn.role == "verify_tree"

    accepted = AcceptResult(request_ids=(1, 2), accepted_counts=(2, 1), accepted_tokens=((10, 11), (20,)))
    committed = policy.commit(txn, accepted.accepted_counts)
    assert committed.committed
    assert committed.accepted_counts == (2, 1)

    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100, 200), root_positions=(5, 3))
    txn_target = policy.begin_transaction([SimpleNamespace(request_id=1), SimpleNamespace(request_id=2)], target)
    assert txn_target.request_ids == (1, 2)
    assert txn_target.draft_rows == 3
    assert txn_target.candidate_counts == (2, 1)
    assert txn_target.role == "verify_tree"
    with pytest.raises(ValueError, match="candidate_counts"):
        policy.commit(txn_target, [3, 0])

    txn2 = policy.begin_transaction([1], DraftBatch(
        request_ids=(1,),
        candidate_tokens=(12,),
        parent_positions=(7,),
        draft_depths=(1,),
        row_to_request=(1,),
    ))
    rolled = policy.rollback(txn2)
    assert rolled.rolled_back


def test_qwen35_dflash_blocker_payload_records_missing_native_verifier(tmp_path) -> None:
    batch_artifact = tmp_path / "batch.json"
    batch_artifact.write_text(
        json.dumps(
            {
                "status": "blocked",
                "performance_claim": False,
                "workload": {"concurrency": 8},
                "execution": {
                    "batch_execution": {
                        "path": "scheduler_serial_slot_bridge",
                        "row_execution": "serial_c1_layer_path",
                        "throughput_claim_eligible": False,
                        "native_prefill_plan": {
                            "linear_prefix_layers": 3,
                            "full_layer_limit_native": False,
                            "first_unsupported_layer": 3,
                            "first_unsupported_type": "full_attention",
                        },
                    }
                },
            }
        )
    )
    prefill_artifact = tmp_path / "prefill.json"
    prefill_artifact.write_text(json.dumps({"native_prefill_plan": {"linear_prefix_layers": 3}}))

    payload = build_payload(batch_artifact=batch_artifact, prefill_artifact=prefill_artifact, argv=[])

    assert payload["status"] == "blocked"
    assert not payload["performance_claim"]
    assert not payload["implementation_status"]["native_target_verify_ready"]
    assert payload["implementation_status"]["interfaces_present"]["target_verify_batch"] == "TargetVerifyBatch"
    assert payload["implementation_status"]["interfaces_present"]["target_accept_summary"] == "TargetAcceptSummary"
    assert payload["implementation_status"]["interfaces_present"]["target_commit_plan"] == "TargetCommitPlan"
    assert payload["implementation_status"]["kv_transaction_target_verify"]["target_verify_rows"] == 5
    assert payload["implementation_status"]["kv_transaction_target_verify"]["candidate_counts"] == [2, 1]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["commit_selection_rows"] == [3, 4]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["accept_summary"]["commit_rows"] == [3, 4]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["accept_summary"]["accepted_tokens"] == [[10, 11], [20]]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["commit_plan"]["accepted_counts"] == [2, 1]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["commit_plan"]["candidate_counts"] == [2, 1]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["shape_key"]["tree_shape"] == [0, 1, 0]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["work_item"]["tree_parents"] == [0, 1, 0]
    assert payload["implementation_status"]["kv_transaction_target_verify"]["transaction_draft_rows"] == 3
    assert payload["implementation_status"]["kv_transaction_target_verify"]["root_rows_excluded_from_journal"]
    assert payload["implementation_status"]["resident_api"]["step_batch_serial"]
    assert not payload["implementation_status"]["resident_api"]["speculative_verify_batch"]
    assert payload["evidence"]["batch_execution"]["path"] == "scheduler_serial_slot_bridge"
    assert any("TargetVerifyBatch" in blocker for blocker in payload["blockers"])
    assert any("throughput_claim_eligible=false" in blocker for blocker in payload["blockers"])
