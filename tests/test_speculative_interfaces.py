from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kvcache import FixedPagedKVPolicy
from hipengine.speculative import AcceptResult, DraftBatch
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
    assert txn.role == "verify_tree"

    accepted = AcceptResult(request_ids=(1, 2), accepted_counts=(2, 1), accepted_tokens=((10, 11), (20,)))
    committed = policy.commit(txn, accepted.accepted_counts)
    assert committed.committed
    assert committed.accepted_counts == (2, 1)

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
    assert payload["implementation_status"]["resident_api"]["step_batch_serial"]
    assert not payload["implementation_status"]["resident_api"]["speculative_verify_batch"]
    assert payload["evidence"]["batch_execution"]["path"] == "scheduler_serial_slot_bridge"
    assert any("TargetVerifyBatch" in blocker for blocker in payload["blockers"])
    assert any("throughput_claim_eligible=false" in blocker for blocker in payload["blockers"])
