#!/usr/bin/env python3
"""Emit the current Qwen3.5/PARO DFlash/DDTree implementation blocker.

The speculative interfaces and KV transaction scaffolding exist, but the native
DFlash/DDTree fast path requires a batched target verifier with selectable
per-row state.  This helper records the exact evidence tying that dependency to
current Qwen3.5/PARO resident metadata and c>N diagnostics.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.dispatch import ActiveBatch, RequestState, WorkKind
from hipengine.generation import ResidentBatchScheduler
from hipengine.kvcache import FixedPagedKVPolicy, KVTransaction
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession
from hipengine.speculative import (
    AcceptResult,
    DraftBatch,
    DraftModel,
    TargetAcceptSummary,
    TargetCommitPlan,
    TargetStateCommitBuffers,
    TargetVerifyBatch,
    TargetVerifyBuffers,
    Verifier,
)

DEFAULT_BATCH_ARTIFACT = Path("benchmarks/results/2026-05-15-hipengine-qwen35-c8-scheduler-serial-bench-blocked.json")
DEFAULT_PREFILL_ARTIFACT = Path("benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-attn-boundary-blocked.json")
DEFAULT_OUT = Path("benchmarks/results/2026-05-15-hipengine-qwen35-dflash-ddtree-blocked.json")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _command(argv: Sequence[str] | None) -> str:
    parts = ["python3", "scripts/qwen35_dflash_ddtree_blocker.py"]
    parts.extend(sys.argv[1:] if argv is None else list(argv))
    return " ".join(shlex.quote(part) for part in parts)


def _speculative_request_ids_unique_checked() -> bool:
    try:
        DraftBatch(
            request_ids=(1, 1),
            candidate_tokens=(10,),
            parent_positions=(0,),
            draft_depths=(1,),
            row_to_request=(1,),
        )
    except ValueError as exc:
        return "unique" in str(exc)
    return False


def _kv_transaction_request_ids_unique_checked() -> bool:
    try:
        KVTransaction(transaction_id=0, request_ids=(1, 1), draft_rows=1, role="verify_chain")
    except ValueError as exc:
        return "unique" in str(exc)
    return False


def _kv_transaction_terminal_state_checked() -> bool:
    try:
        KVTransaction(transaction_id=0, request_ids=(1,), draft_rows=1, role="verify_chain", committed=True)
    except ValueError as exc:
        committed_requires_counts = "requires accepted_counts" in str(exc)
    else:
        committed_requires_counts = False
    try:
        KVTransaction(transaction_id=0, request_ids=(1,), draft_rows=1, role="verify_chain", accepted_counts=(1,))
    except ValueError as exc:
        counts_require_commit = "require committed" in str(exc)
    else:
        counts_require_commit = False
    return committed_requires_counts and counts_require_commit


def _kv_transaction_role_checked() -> bool:
    try:
        KVTransaction(transaction_id=0, request_ids=(1,), draft_rows=1, role="decode")
    except ValueError as exc:
        return "role" in str(exc)
    return False


def _sample_tree_accept_summary(*, transaction_id: int | None = None) -> TargetAcceptSummary:
    draft = DraftBatch(
        request_ids=(1,),
        candidate_tokens=(10,),
        parent_positions=(5,),
        draft_depths=(1,),
        row_to_request=(1,),
        mode="verify_tree",
        tree_parents=(-1,),
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100,), root_positions=(5,))
    return TargetAcceptSummary.from_accept_result(
        target,
        AcceptResult(request_ids=(1,), accepted_counts=(1,), accepted_tokens=((10,),), transaction_id=transaction_id),
    )


def _target_commit_plan_transaction_role_checked() -> bool:
    summary = _sample_tree_accept_summary()
    mismatch = SimpleNamespace(
        transaction_id=0,
        request_ids=(1,),
        candidate_counts=(1,),
        committed=False,
        rolled_back=False,
        role="verify_chain",
    )
    invalid = SimpleNamespace(
        transaction_id=1,
        request_ids=(1,),
        candidate_counts=(1,),
        committed=False,
        rolled_back=False,
        role="decode",
    )
    try:
        TargetCommitPlan.from_summary(summary, mismatch)
    except ValueError as exc:
        mismatch_rejected = "role must match" in str(exc)
    else:
        mismatch_rejected = False
    try:
        TargetCommitPlan.from_summary(summary, invalid)
    except ValueError as exc:
        invalid_rejected = "verify_chain or verify_tree" in str(exc)
    else:
        invalid_rejected = False
    return mismatch_rejected and invalid_rejected


def _target_commit_plan_candidate_budget_checked() -> bool:
    summary = _sample_tree_accept_summary()
    mismatch = SimpleNamespace(
        transaction_id=2,
        request_ids=(1,),
        candidate_counts=(2,),
        committed=False,
        rolled_back=False,
        role="verify_tree",
    )
    try:
        TargetCommitPlan.from_summary(summary, mismatch)
    except ValueError as exc:
        return "candidate_counts must match" in str(exc)
    return False


def _target_accept_summary_transaction_id_checked() -> bool:
    try:
        _sample_tree_accept_summary(transaction_id=-1)
    except ValueError as exc:
        return "transaction_id" in str(exc)
    return False


def _target_accept_oracle_checked() -> bool:
    draft = DraftBatch(
        request_ids=(1,),
        candidate_tokens=(10, 11),
        parent_positions=(5, 6),
        draft_depths=(1, 2),
        row_to_request=(1, 1),
        mode="verify_tree",
        tree_parents=(-1, 0),
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100,), root_positions=(5,))
    result = target.accept_from_top1((10, 11, 12), transaction_id=7, remaining_decode=(3,))
    if result.accepted_counts != (2,) or result.accepted_tokens != ((10, 11),):
        return False
    if result.selected_candidate_rows != (2,) or result.next_tokens != (12,) or result.transaction_id != 7:
        return False
    try:
        target.accept_from_top1((10, 11))
    except ValueError as exc:
        return "target_top1" in str(exc)
    return False


def _accept_result_selected_rows_checked() -> bool:
    try:
        AcceptResult(request_ids=(1,), accepted_counts=(1,), accepted_tokens=((10,),), selected_candidate_rows=(-1,))
    except ValueError as exc:
        negative_rejected = "selected_candidate_rows" in str(exc)
    else:
        negative_rejected = False
    draft = DraftBatch(
        request_ids=(1,),
        candidate_tokens=(10, 11),
        parent_positions=(5, 5),
        draft_depths=(1, 1),
        row_to_request=(1, 1),
        mode="verify_tree",
        tree_parents=(-1, -1),
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(100,), root_positions=(5,))
    try:
        TargetAcceptSummary.from_accept_result(
            target,
            AcceptResult(request_ids=(1,), accepted_counts=(1,), accepted_tokens=((11,),), selected_candidate_rows=(2,)),
        )
    except ValueError:
        resolved_ambiguous_row = False
    else:
        resolved_ambiguous_row = True
    return negative_rejected and resolved_ambiguous_row


def _accept_result_next_tokens_checked() -> bool:
    try:
        AcceptResult(request_ids=(1,), accepted_counts=(1,), accepted_tokens=((10,),), next_tokens=(-1,))
    except ValueError as exc:
        negative_rejected = "next_tokens" in str(exc)
    else:
        negative_rejected = False
    summary = _sample_tree_accept_summary()
    try:
        replace(summary, next_tokens=(-1,))
    except ValueError as exc:
        summary_rejected = "next_tokens" in str(exc)
    else:
        summary_rejected = False
    return negative_rejected and summary_rejected


def _target_accept_summary_topology_checked() -> bool:
    summary = _sample_tree_accept_summary()
    try:
        replace(summary, draft_depth=-1)
    except ValueError as exc:
        depth_rejected = "draft_depth" in str(exc)
    else:
        depth_rejected = False
    try:
        replace(summary, tree_shape=())
    except ValueError as exc:
        tree_rejected = "tree_shape" in str(exc)
    else:
        tree_rejected = False
    return depth_rejected and tree_rejected


def _sample_chain_target_batch() -> TargetVerifyBatch:
    draft = DraftBatch(
        request_ids=(1,),
        candidate_tokens=(10,),
        parent_positions=(5,),
        draft_depths=(1,),
        row_to_request=(1,),
    )
    return TargetVerifyBatch.from_draft(draft, root_tokens=(100,), root_positions=(5,))


def _sample_target_verify_buffers(
    target: TargetVerifyBatch,
    *,
    transaction_id: int | None = None,
    candidate_counts: Sequence[int] | None = None,
    draft_depth: int | None = None,
    tree_shape: Sequence[int] | None = None,
    next_tokens: Tensor | None = None,
) -> TargetVerifyBuffers:
    device = Device("hip", 0)
    return TargetVerifyBuffers.for_batch(
        target,
        token_ids=Tensor.from_handle(0x2A00, (target.rows,), "int32", device),
        positions=Tensor.from_handle(0x2B00, (target.rows,), "int32", device),
        parent_rows=Tensor.from_handle(0x2C00, (target.rows,), "int32", device),
        draft_depths=Tensor.from_handle(0x2D00, (target.rows,), "int32", device),
        row_to_request=Tensor.from_handle(0x2E00, (target.rows,), "int32", device),
        active_mask=Tensor.from_handle(0x2F00, (target.rows,), "bool", device),
        target_top1=Tensor.from_handle(0x3000, (target.rows,), "int32", device),
        accepted_counts=Tensor.from_handle(0x3100, (len(target.request_ids),), "int32", device),
        commit_rows=Tensor.from_handle(0x3200, (len(target.request_ids),), "int32", device),
        commit_tokens=Tensor.from_handle(0x3300, (len(target.request_ids),), "int32", device),
        commit_positions=Tensor.from_handle(0x3400, (len(target.request_ids),), "int32", device),
        next_tokens=next_tokens,
        transaction_id=transaction_id,
        candidate_counts=candidate_counts,
        draft_depth=draft_depth,
        tree_shape=tree_shape,
    )


def _target_verify_buffers_transaction_id_checked() -> bool:
    target = _sample_chain_target_batch()
    try:
        _sample_target_verify_buffers(target, transaction_id=-1)
    except ValueError as exc:
        return "transaction_id" in str(exc)
    return False


def _target_verify_buffers_candidate_counts_checked() -> bool:
    target = _sample_chain_target_batch()
    try:
        _sample_target_verify_buffers(target, candidate_counts=(2,))
    except ValueError as exc:
        return "candidate_counts" in str(exc)
    return False


def _target_verify_buffers_topology_checked() -> bool:
    target = _sample_chain_target_batch()
    try:
        _sample_target_verify_buffers(target, draft_depth=-1)
    except ValueError as exc:
        depth_rejected = "draft_depth" in str(exc)
    else:
        depth_rejected = False
    try:
        _sample_target_verify_buffers(target, tree_shape=())
    except ValueError as exc:
        tree_rejected = "tree_shape" in str(exc)
    else:
        tree_rejected = False
    return depth_rejected and tree_rejected


def _target_verify_buffers_next_tokens_checked() -> bool:
    target = _sample_chain_target_batch()
    device = Device("hip", 0)
    try:
        _sample_target_verify_buffers(target, next_tokens=Tensor.from_handle(0x3500, (2,), "int32", device))
    except ValueError as exc:
        shape_rejected = "summary tensors" in str(exc)
    else:
        shape_rejected = False
    try:
        _sample_target_verify_buffers(target, next_tokens=Tensor.from_handle(0x3600, (len(target.request_ids),), "fp16", device))
    except ValueError as exc:
        dtype_rejected = "integer buffers" in str(exc)
    else:
        dtype_rejected = False
    return shape_rejected and dtype_rejected


def _scheduler_speculative_next_tokens_checked() -> bool:
    scheduler = ResidentBatchScheduler(capacity=1)
    request_id = scheduler.submit([10], max_new_tokens=2)
    scheduler.admit_pending()
    scheduler.next_prefill_work(chunk_size=8)
    draft = DraftBatch(
        request_ids=(request_id,),
        candidate_tokens=(101,),
        parent_positions=(0,),
        draft_depths=(1,),
        row_to_request=(request_id,),
    )
    work = scheduler.next_speculative_verify_work(draft, root_tokens=(10,), root_positions=(0,))
    summary = TargetAcceptSummary.from_accept_result(
        work.target_batch,
        AcceptResult(request_ids=(request_id,), accepted_counts=(1,), accepted_tokens=((101,),), next_tokens=(102,)),
    )
    completed = scheduler.record_speculative_accept(summary)
    emitted = scheduler.completed[request_id].generated_tokens == (101, 102) and tuple(item.request_id for item in completed) == (request_id,)

    over_budget_scheduler = ResidentBatchScheduler(capacity=1)
    over_budget_request_id = over_budget_scheduler.submit([10], max_new_tokens=1)
    over_budget_scheduler.admit_pending()
    over_budget_scheduler.next_prefill_work(chunk_size=8)
    over_budget_work = over_budget_scheduler.next_speculative_verify_work(draft, root_tokens=(10,), root_positions=(0,))
    over_budget_summary = TargetAcceptSummary.from_accept_result(
        over_budget_work.target_batch,
        AcceptResult(request_ids=(over_budget_request_id,), accepted_counts=(1,), accepted_tokens=((101,),), next_tokens=(102,)),
    )
    try:
        over_budget_scheduler.record_speculative_accept(over_budget_summary)
    except ValueError as exc:
        budget_rejected = "remaining decode" in str(exc)
    else:
        budget_rejected = False
    return emitted and budget_rejected


def _interface_status() -> dict[str, Any]:
    batch = ActiveBatch(2)
    batch.admit(RequestState.from_tokens(0, [1], max_new_tokens=1))
    shape_key = batch.shape_key(mode=WorkKind.VERIFY_TREE, context_bucket_size=256, draft_depth=2, tree_shape=(1, 2))
    return {
        "draft_batch": DraftBatch.__name__,
        "accept_result": AcceptResult.__name__,
        "draft_model_protocol": DraftModel.__name__,
        "target_verify_batch": TargetVerifyBatch.__name__,
        "target_accept_summary": TargetAcceptSummary.__name__,
        "target_commit_plan": TargetCommitPlan.__name__,
        "target_state_commit_buffers": TargetStateCommitBuffers.__name__,
        "target_verify_buffers": TargetVerifyBuffers.__name__,
        "verifier_protocol": Verifier.__name__,
        "kv_policy": FixedPagedKVPolicy.__name__,
        "kv_transaction": KVTransaction.__name__,
        "speculative_request_ids_unique_checked": _speculative_request_ids_unique_checked(),
        "kv_transaction_request_ids_unique_checked": _kv_transaction_request_ids_unique_checked(),
        "kv_transaction_terminal_state_checked": _kv_transaction_terminal_state_checked(),
        "kv_transaction_role_checked": _kv_transaction_role_checked(),
        "target_commit_plan_transaction_role_checked": _target_commit_plan_transaction_role_checked(),
        "target_commit_plan_candidate_budget_checked": _target_commit_plan_candidate_budget_checked(),
        "target_accept_summary_transaction_id_checked": _target_accept_summary_transaction_id_checked(),
        "target_accept_oracle_checked": _target_accept_oracle_checked(),
        "accept_result_selected_rows_checked": _accept_result_selected_rows_checked(),
        "accept_result_next_tokens_checked": _accept_result_next_tokens_checked(),
        "target_accept_summary_topology_checked": _target_accept_summary_topology_checked(),
        "target_verify_buffers_transaction_id_checked": _target_verify_buffers_transaction_id_checked(),
        "target_verify_buffers_candidate_counts_checked": _target_verify_buffers_candidate_counts_checked(),
        "target_verify_buffers_topology_checked": _target_verify_buffers_topology_checked(),
        "target_verify_buffers_next_tokens_checked": _target_verify_buffers_next_tokens_checked(),
        "scheduler_speculative_verify_work": hasattr(ResidentBatchScheduler, "next_speculative_verify_work"),
        "scheduler_speculative_accept": hasattr(ResidentBatchScheduler, "record_speculative_accept"),
        "scheduler_speculative_next_tokens_checked": _scheduler_speculative_next_tokens_checked(),
        "scheduler_speculative_shape_key": hasattr(ResidentBatchScheduler, "speculative_verify_shape_key"),
        "scheduler_speculative_graph_cache": hasattr(ResidentBatchScheduler, "get_or_create_speculative_verify_graph"),
        "scheduler_speculative_kv_transaction": hasattr(ResidentBatchScheduler, "begin_speculative_verify_transaction"),
        "scheduler_speculative_verify_plan": hasattr(ResidentBatchScheduler, "plan_speculative_verify"),
        "scheduler_speculative_buffer_plan": hasattr(ResidentBatchScheduler, "bind_speculative_verify_buffers"),
        "scheduler_speculative_commit_plan": hasattr(ResidentBatchScheduler, "plan_speculative_commit"),
        "scheduler_speculative_commit_from_top1": hasattr(ResidentBatchScheduler, "plan_speculative_commit_from_top1"),
        "scheduler_speculative_state_commit_plan": hasattr(ResidentBatchScheduler, "bind_speculative_commit_buffers"),
        "scheduler_speculative_kv_commit": hasattr(ResidentBatchScheduler, "commit_speculative_kv_transaction"),
        "scheduler_speculative_kv_rollback": hasattr(ResidentBatchScheduler, "rollback_speculative_kv_transaction"),
        "scheduler_speculative_accept_finalize": hasattr(ResidentBatchScheduler, "finalize_speculative_accept"),
        "verify_graph_shape_key": {
            "mode": shape_key.mode.value,
            "active_c": shape_key.active_c,
            "draft_depth": shape_key.draft_depth,
            "tree_shape": list(shape_key.tree_shape),
        },
    }


def _resident_target_verify_transaction_id_checked() -> bool:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.closed = False
    session.max_batch_size = 2
    session.max_sequence_length = 4
    session.vocab_size = 100
    session.device = Device("hip", 0)
    draft = DraftBatch(
        request_ids=(1,),
        candidate_tokens=(10,),
        parent_positions=(0,),
        draft_depths=(1,),
        row_to_request=(1,),
    )
    target = session.target_verify_batch(draft, root_tokens=(9,), root_positions=(0,))
    try:
        session.verify_speculative_batch(
            target,
            token_ids=Tensor.from_handle(0x0A00, (target.rows,), "int32", session.device),
            positions=Tensor.from_handle(0x0B00, (target.rows,), "int32", session.device),
            parent_rows=Tensor.from_handle(0x0C00, (target.rows,), "int32", session.device),
            draft_depths=Tensor.from_handle(0x0D00, (target.rows,), "int32", session.device),
            row_to_request=Tensor.from_handle(0x0E00, (target.rows,), "int32", session.device),
            active_mask=Tensor.from_handle(0x0F00, (target.rows,), "bool", session.device),
            target_top1=Tensor.from_handle(0x1000, (target.rows,), "int32", session.device),
            accepted_counts=Tensor.from_handle(0x1100, (len(target.request_ids),), "int32", session.device),
            commit_rows=Tensor.from_handle(0x1200, (len(target.request_ids),), "int32", session.device),
            commit_tokens=Tensor.from_handle(0x1300, (len(target.request_ids),), "int32", session.device),
            commit_positions=Tensor.from_handle(0x1400, (len(target.request_ids),), "int32", session.device),
            transaction_id=-1,
        )
    except ValueError as exc:
        return "transaction_id" in str(exc)
    return False


def _resident_target_verify_device_checked() -> bool:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.closed = False
    session.max_batch_size = 2
    session.max_sequence_length = 4
    session.vocab_size = 100
    session.device = Device("hip", 0)
    draft = DraftBatch(
        request_ids=(1,),
        candidate_tokens=(10,),
        parent_positions=(0,),
        draft_depths=(1,),
        row_to_request=(1,),
    )
    target = session.target_verify_batch(draft, root_tokens=(9,), root_positions=(0,))
    other = Device("hip", 1)
    try:
        session.verify_speculative_batch(
            target,
            token_ids=Tensor.from_handle(0x1000, (target.rows,), "int32", other),
            positions=Tensor.from_handle(0x1100, (target.rows,), "int32", other),
            parent_rows=Tensor.from_handle(0x1200, (target.rows,), "int32", other),
            draft_depths=Tensor.from_handle(0x1300, (target.rows,), "int32", other),
            row_to_request=Tensor.from_handle(0x1400, (target.rows,), "int32", other),
            active_mask=Tensor.from_handle(0x1500, (target.rows,), "bool", other),
            target_top1=Tensor.from_handle(0x1600, (target.rows,), "int32", other),
            accepted_counts=Tensor.from_handle(0x1700, (len(target.request_ids),), "int32", other),
            commit_rows=Tensor.from_handle(0x1800, (len(target.request_ids),), "int32", other),
            commit_tokens=Tensor.from_handle(0x1900, (len(target.request_ids),), "int32", other),
            commit_positions=Tensor.from_handle(0x1A00, (len(target.request_ids),), "int32", other),
        )
    except ValueError as exc:
        return "resident device" in str(exc)
    return False


def _resident_state_commit_transaction_id_checked() -> bool:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.closed = False
    session.device = Device("hip", 0)
    plan = TargetCommitPlan(
        transaction_id=0,
        request_ids=(1,),
        accepted_counts=(1,),
        commit_rows=(1,),
        commit_tokens=(10,),
        commit_positions=(6,),
        candidate_counts=(1,),
        mode="verify_chain",
    )
    buffers = TargetStateCommitBuffers.for_plan(
        plan,
        accepted_counts=Tensor.from_handle(0x1B00, (1,), "int32", session.device),
        commit_rows=Tensor.from_handle(0x1C00, (1,), "int32", session.device),
        commit_positions=Tensor.from_handle(0x1D00, (1,), "int32", session.device),
        parent_rows=Tensor.from_handle(0x1D80, (2,), "int32", session.device),
        kv_rows_src=Tensor.from_handle(0x1E00, (2, 8, 128), "bf16", session.device),
        kv_rows_dst=Tensor.from_handle(0x1F00, (1, 8, 128), "bf16", session.device),
    )
    try:
        session.commit_verified_state(plan, replace(buffers, transaction_id=1), execute_copies=False)
    except ValueError as exc:
        return "transaction_id" in str(exc)
    return False


def _resident_state_commit_row_coverage_checked() -> bool:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    session.closed = False
    session.device = Device("hip", 0)
    plan = TargetCommitPlan(
        transaction_id=0,
        request_ids=(1, 2),
        accepted_counts=(2, 1),
        commit_rows=(3, 4),
        commit_tokens=(11, 20),
        commit_positions=(7, 4),
        candidate_counts=(2, 1),
        mode="verify_tree",
    )
    buffers = TargetStateCommitBuffers.for_plan(
        plan,
        accepted_counts=Tensor.from_handle(0x2000, (2,), "int32", session.device),
        commit_rows=Tensor.from_handle(0x2100, (2,), "int32", session.device),
        commit_positions=Tensor.from_handle(0x2200, (2,), "int32", session.device),
        parent_rows=Tensor.from_handle(0x2280, (5,), "int32", session.device),
        kv_rows_src=Tensor.from_handle(0x2300, (5, 8, 128), "bf16", session.device),
        kv_rows_dst=Tensor.from_handle(0x2400, (2, 8, 128), "bf16", session.device),
    )
    try:
        session.commit_verified_state(plan, buffers, execute_copies=False)
    except ValueError as exc:
        return "accepted token rows" in str(exc)
    return False


def _resident_api_status() -> dict[str, Any]:
    session = Qwen35ParoResidentSession.__new__(Qwen35ParoResidentSession)
    speculative = session.speculative_execution_metadata().to_json_dict()
    return {
        "step_batch_serial": hasattr(Qwen35ParoResidentSession, "step_batch_serial"),
        "batch_execution_metadata": hasattr(Qwen35ParoResidentSession, "batch_execution_metadata"),
        "target_verify_buffers_transaction_id_checked": _resident_target_verify_transaction_id_checked(),
        "target_verify_buffers_resident_device_checked": _resident_target_verify_device_checked(),
        "commit_verified_state_transaction_id_checked": _resident_state_commit_transaction_id_checked(),
        "commit_verified_state_row_coverage_checked": _resident_state_commit_row_coverage_checked(),
        **speculative,
    }


def _kv_transaction_status() -> dict[str, Any]:
    device = Device("hip", 0)
    policy = FixedPagedKVPolicy()
    for request_id, ptr in ((1, 0x1000), (2, 0x2000)):
        policy.register(
            request_id,
            block_table=Tensor.from_handle(ptr, (4,), "int32", device),
            live_counts=Tensor.from_handle(ptr + 0x100, (1,), "int64", device),
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
    txn = policy.begin_transaction((1, 2), target)
    target_top1 = (10, 20, 11, 12, 21)
    accept = target.accept_from_top1(target_top1, transaction_id=txn.transaction_id)
    summary = TargetAcceptSummary.from_accept_result(target, accept)
    plan = TargetCommitPlan.from_summary(summary, txn)
    buffers = TargetVerifyBuffers.for_batch(
        target,
        token_ids=Tensor.from_handle(0x3000, (target.rows,), "int32", device),
        positions=Tensor.from_handle(0x3100, (target.rows,), "int32", device),
        parent_rows=Tensor.from_handle(0x3200, (target.rows,), "int32", device),
        draft_depths=Tensor.from_handle(0x3300, (target.rows,), "int32", device),
        row_to_request=Tensor.from_handle(0x3400, (target.rows,), "int32", device),
        active_mask=Tensor.from_handle(0x3500, (target.rows,), "bool", device),
        target_top1=Tensor.from_handle(0x3600, (target.rows,), "int32", device),
        accepted_counts=Tensor.from_handle(0x3700, (len(target.request_ids),), "int32", device),
        commit_rows=Tensor.from_handle(0x3800, (len(target.request_ids),), "int32", device),
        commit_tokens=Tensor.from_handle(0x3900, (len(target.request_ids),), "int32", device),
        commit_positions=Tensor.from_handle(0x3A00, (len(target.request_ids),), "int32", device),
        next_tokens=Tensor.from_handle(0x3B00, (len(target.request_ids),), "int32", device),
    )
    state_buffers = TargetStateCommitBuffers.for_plan(
        plan,
        accepted_counts=Tensor.from_handle(0x3C00, (len(target.request_ids),), "int32", device),
        commit_rows=Tensor.from_handle(0x3D00, (len(target.request_ids),), "int32", device),
        commit_positions=Tensor.from_handle(0x3E00, (len(target.request_ids),), "int32", device),
        parent_rows=Tensor.from_handle(0x3E80, (target.rows,), "int32", device),
        linear_state_src=Tensor.from_handle(0x3F00, (target.rows, 40, 128), "bf16", device),
        linear_state_dst=Tensor.from_handle(0x4000, (len(target.request_ids), 40, 128), "bf16", device),
        kv_rows_src=Tensor.from_handle(0x4100, (target.rows, 8, 128), "bf16", device),
        kv_rows_dst=Tensor.from_handle(0x4200, (sum(summary.accepted_counts), 8, 128), "bf16", device),
    )
    selection = target.select_commit_rows(summary.accepted_counts)
    active = ActiveBatch(2)
    active.admit(RequestState(request_id=1, prompt_tokens=(1, 2, 3, 4, 5), max_new_tokens=4, next_prompt_index=5))
    active.admit(RequestState(request_id=2, prompt_tokens=(6, 7, 8), max_new_tokens=4, next_prompt_index=3))
    key = target.shape_key(active, context_bucket_size=4, top_k=8, experts_per_token=8, replay_steps=1)
    work = target.to_work_item()
    scheduler = ResidentBatchScheduler(capacity=2, context_bucket_size=4)
    scheduler.submit([1, 2, 3, 4, 5], max_new_tokens=4, request_id=1)
    scheduler.submit([6, 7, 8], max_new_tokens=4, request_id=2)
    scheduler.admit_pending()
    scheduler.next_prefill_work(chunk_size=8)
    scheduler.next_prefill_work(chunk_size=8)
    scheduler_work = scheduler.next_speculative_verify_work(
        draft,
        root_tokens=(100, 200),
        root_positions=(5, 3),
    )
    scheduler_plan = scheduler.plan_speculative_verify(
        policy,
        scheduler_work,
        lambda bucket: {"mode": bucket.mode.value, "draft_depth": bucket.draft_depth},
        top_k=8,
        experts_per_token=8,
        replay_steps=1,
    )
    scheduler_rollback_plan = scheduler.plan_speculative_verify(
        policy,
        scheduler_work,
        lambda bucket: {"mode": bucket.mode.value, "draft_depth": bucket.draft_depth},
        top_k=8,
        experts_per_token=8,
        replay_steps=1,
    )
    scheduler_rolled_txn = scheduler.rollback_speculative_kv_transaction(policy, scheduler_rollback_plan)
    scheduler_buffers = TargetVerifyBuffers.for_batch(
        scheduler_work.target_batch,
        token_ids=buffers.token_ids,
        positions=buffers.positions,
        parent_rows=buffers.parent_rows,
        draft_depths=buffers.draft_depths,
        row_to_request=buffers.row_to_request,
        active_mask=buffers.active_mask,
        target_top1=buffers.target_top1,
        accepted_counts=buffers.accepted_counts,
        commit_rows=buffers.commit_rows,
        commit_tokens=buffers.commit_tokens,
        commit_positions=buffers.commit_positions,
        next_tokens=buffers.next_tokens,
        transaction_id=scheduler_plan.transaction.transaction_id,
    )
    scheduler_buffer_plan = scheduler.bind_speculative_verify_buffers(scheduler_plan, scheduler_buffers)
    scheduler_commit_plan = scheduler.plan_speculative_commit_from_top1(scheduler_buffer_plan, target_top1)
    scheduler_state_buffers = TargetStateCommitBuffers.for_plan(
        scheduler_commit_plan.commit_plan,
        accepted_counts=state_buffers.accepted_counts,
        commit_rows=state_buffers.commit_rows,
        commit_positions=state_buffers.commit_positions,
        parent_rows=state_buffers.parent_rows,
        linear_state_src=state_buffers.linear_state_src,
        linear_state_dst=state_buffers.linear_state_dst,
        kv_rows_src=state_buffers.kv_rows_src,
        kv_rows_dst=state_buffers.kv_rows_dst,
    )
    scheduler_state_plan = scheduler.bind_speculative_commit_buffers(scheduler_commit_plan, scheduler_state_buffers)
    scheduler_committed_txn = scheduler.commit_speculative_kv_transaction(policy, scheduler_state_plan)
    scheduler_completed = scheduler.finalize_speculative_accept(scheduler_committed_txn, scheduler_state_plan)
    return {
        "target_verify_rows": target.rows,
        "candidate_rows": target.candidate_count,
        "candidate_counts": list(txn.candidate_counts) if txn.candidate_counts is not None else None,
        "transaction_draft_rows": txn.draft_rows,
        "role": txn.role,
        "root_rows_excluded_from_journal": txn.draft_rows == target.candidate_count,
        "commit_selection_rows": list(selection.selected_rows),
        "commit_selection_positions": list(selection.selected_positions),
        "target_top1": list(target_top1),
        "accept_result": {
            "transaction_id": accept.transaction_id,
            "selected_candidate_rows": [] if accept.selected_candidate_rows is None else list(accept.selected_candidate_rows),
            "next_tokens": [] if accept.next_tokens is None else list(accept.next_tokens),
        },
        "accept_summary": {
            "transaction_id": summary.transaction_id,
            "accepted_counts": list(summary.accepted_counts),
            "accepted_tokens": [list(row) for row in summary.accepted_tokens],
            "next_tokens": [] if summary.next_tokens is None else list(summary.next_tokens),
            "candidate_counts": None if summary.candidate_counts is None else list(summary.candidate_counts),
            "draft_depth": summary.draft_depth,
            "tree_shape": [] if summary.tree_shape is None else list(summary.tree_shape),
            "commit_rows": list(summary.commit_rows),
            "commit_tokens": list(summary.commit_tokens),
            "commit_positions": list(summary.commit_positions),
            "full_accept": list(summary.full_accept),
        },
        "commit_plan": {
            "transaction_id": plan.transaction_id,
            "accepted_counts": list(plan.accepted_counts),
            "commit_rows": list(plan.commit_rows),
            "commit_positions": list(plan.commit_positions),
            "next_tokens": [] if plan.next_tokens is None else list(plan.next_tokens),
            "candidate_counts": None if plan.candidate_counts is None else list(plan.candidate_counts),
            "draft_depth": plan.draft_depth,
            "tree_shape": [] if plan.tree_shape is None else list(plan.tree_shape),
            "mode": plan.mode,
        },
        "device_buffers": {
            "transaction_id": buffers.transaction_id,
            "rows": buffers.rows,
            "candidate_rows": buffers.candidate_rows,
            "candidate_counts": None if buffers.candidate_counts is None else list(buffers.candidate_counts),
            "draft_depth": buffers.draft_depth,
            "tree_shape": [] if buffers.tree_shape is None else list(buffers.tree_shape),
            "summary_rows": buffers.request_count,
            "device": str(buffers.device),
            "token_ids_dtype": buffers.token_ids.dtype.value,
            "active_mask_dtype": buffers.active_mask.dtype.value,
            "target_top1_shape": list(buffers.target_top1.shape),
            "accepted_counts_shape": list(buffers.accepted_counts.shape),
            "next_tokens_shape": [] if buffers.next_tokens is None else list(buffers.next_tokens.shape),
            "next_tokens_dtype": None if buffers.next_tokens is None else buffers.next_tokens.dtype.value,
        },
        "state_commit_buffers": {
            "transaction_id": state_buffers.transaction_id,
            "request_rows": state_buffers.request_count,
            "device": str(state_buffers.device),
            "has_linear_state": state_buffers.has_linear_state,
            "linear_state_tail_shape": list(state_buffers.linear_state_src.shape[1:]) if state_buffers.linear_state_src else [],
            "has_kv_rows": state_buffers.has_kv_rows,
            "kv_dst_rows": state_buffers.kv_rows_dst.shape[0] if state_buffers.kv_rows_dst else 0,
        },
        "shape_key": {
            "mode": key.mode.value,
            "active_c": key.active_c,
            "context_bucket": key.context_bucket,
            "active_mask": list(key.active_mask),
            "top_k": key.top_k,
            "experts_per_token": key.experts_per_token,
            "replay_steps": key.replay_steps,
            "draft_depth": key.draft_depth,
            "tree_shape": list(key.tree_shape),
        },
        "work_item": {
            "kind": work.kind.value,
            "request_ids": list(work.request_ids),
            "row_to_request": list(work.row_to_request),
            "token_rows": [list(row) for row in work.token_rows],
            "draft_depth": work.draft_depth,
            "tree_parents": list(work.tree_parents),
        },
        "scheduler_verify_plan": {
            "transaction_id": scheduler_plan.transaction.transaction_id,
            "request_ids": list(scheduler_plan.transaction.request_ids),
            "transaction_draft_rows": scheduler_plan.transaction.draft_rows,
            "candidate_counts": list(scheduler_plan.transaction.candidate_counts or ()),
            "shape_key_matches_target": scheduler_plan.shape_key == key,
            "graph_cache_entries": scheduler.graph_buckets.stats.entries,
            "graph_mode": scheduler_plan.graph["mode"],
            "graph_draft_depth": scheduler_plan.graph["draft_depth"],
        },
        "scheduler_buffer_plan": {
            "request_ids": list(scheduler_buffer_plan.buffers.request_ids),
            "rows": scheduler_buffer_plan.buffers.rows,
            "candidate_rows": scheduler_buffer_plan.buffers.candidate_rows,
            "candidate_counts": None if scheduler_buffer_plan.buffers.candidate_counts is None else list(scheduler_buffer_plan.buffers.candidate_counts),
            "target_candidate_counts": list(scheduler_buffer_plan.plan.target_batch.candidate_counts),
            "candidate_counts_match": scheduler_buffer_plan.buffers.candidate_counts == scheduler_buffer_plan.plan.target_batch.candidate_counts,
            "draft_depth": scheduler_buffer_plan.buffers.draft_depth,
            "target_draft_depth": scheduler_buffer_plan.plan.target_batch.draft_depth,
            "draft_depth_matches": scheduler_buffer_plan.buffers.draft_depth == scheduler_buffer_plan.plan.target_batch.draft_depth,
            "tree_shape": [] if scheduler_buffer_plan.buffers.tree_shape is None else list(scheduler_buffer_plan.buffers.tree_shape),
            "target_tree_shape": list(scheduler_buffer_plan.plan.target_batch.tree_shape),
            "tree_shape_matches": scheduler_buffer_plan.buffers.tree_shape == scheduler_buffer_plan.plan.target_batch.tree_shape,
            "mode": scheduler_buffer_plan.buffers.mode,
            "target_batch_rows": scheduler_buffer_plan.plan.target_batch.rows,
            "buffer_transaction_id": scheduler_buffer_plan.buffers.transaction_id,
            "transaction_id": scheduler_buffer_plan.plan.transaction.transaction_id,
            "transaction_id_matches": scheduler_buffer_plan.buffers.transaction_id == scheduler_buffer_plan.plan.transaction.transaction_id,
            "next_tokens_shape": [] if scheduler_buffer_plan.buffers.next_tokens is None else list(scheduler_buffer_plan.buffers.next_tokens.shape),
        },
        "scheduler_commit_plan": {
            "transaction_id": scheduler_commit_plan.commit_plan.transaction_id,
            "request_ids": list(scheduler_commit_plan.commit_plan.request_ids),
            "summary_transaction_id": scheduler_commit_plan.summary.transaction_id,
            "from_top1": True,
            "accepted_counts": list(scheduler_commit_plan.commit_plan.accepted_counts),
            "commit_rows": list(scheduler_commit_plan.commit_plan.commit_rows),
            "commit_positions": list(scheduler_commit_plan.commit_plan.commit_positions),
            "next_tokens": [] if scheduler_commit_plan.commit_plan.next_tokens is None else list(scheduler_commit_plan.commit_plan.next_tokens),
            "candidate_counts": list(scheduler_commit_plan.commit_plan.candidate_counts or ()),
            "mode": scheduler_commit_plan.commit_plan.mode,
        },
        "scheduler_state_commit_plan": {
            "request_ids": list(scheduler_state_plan.buffers.request_ids),
            "request_rows": scheduler_state_plan.buffers.request_count,
            "mode": scheduler_state_plan.buffers.mode,
            "buffer_transaction_id": scheduler_state_plan.buffers.transaction_id,
            "device": str(scheduler_state_plan.buffers.device),
            "verify_device": str(scheduler_state_plan.commit_plan.verify_plan.buffers.device),
            "device_matches_verify": scheduler_state_plan.buffers.device == scheduler_state_plan.commit_plan.verify_plan.buffers.device,
            "target_rows": scheduler_state_plan.commit_plan.verify_plan.plan.target_batch.rows,
            "accepted_rows": sum(scheduler_state_plan.commit_plan.commit_plan.accepted_counts),
            "has_linear_state": scheduler_state_plan.buffers.has_linear_state,
            "linear_src_rows": scheduler_state_plan.buffers.linear_state_src.shape[0] if scheduler_state_plan.buffers.linear_state_src else 0,
            "linear_src_covers_target": bool(
                scheduler_state_plan.buffers.linear_state_src
                and scheduler_state_plan.buffers.linear_state_src.shape[0] >= scheduler_state_plan.commit_plan.verify_plan.plan.target_batch.rows
            ),
            "has_kv_rows": scheduler_state_plan.buffers.has_kv_rows,
            "kv_src_rows": scheduler_state_plan.buffers.kv_rows_src.shape[0] if scheduler_state_plan.buffers.kv_rows_src else 0,
            "kv_src_covers_target": bool(
                scheduler_state_plan.buffers.kv_rows_src
                and scheduler_state_plan.buffers.kv_rows_src.shape[0] >= scheduler_state_plan.commit_plan.verify_plan.plan.target_batch.rows
            ),
            "kv_dst_rows": scheduler_state_plan.buffers.kv_rows_dst.shape[0] if scheduler_state_plan.buffers.kv_rows_dst else 0,
            "kv_dst_covers_accepts": bool(
                scheduler_state_plan.buffers.kv_rows_dst
                and scheduler_state_plan.buffers.kv_rows_dst.shape[0] >= sum(scheduler_state_plan.commit_plan.commit_plan.accepted_counts)
            ),
            "transaction_id": scheduler_state_plan.commit_plan.commit_plan.transaction_id,
            "transaction_id_matches": scheduler_state_plan.buffers.transaction_id == scheduler_state_plan.commit_plan.commit_plan.transaction_id,
        },
        "scheduler_kv_commit": {
            "transaction_id": scheduler_committed_txn.transaction_id,
            "request_ids": list(scheduler_committed_txn.request_ids),
            "accepted_counts": list(scheduler_committed_txn.accepted_counts or ()),
            "committed": scheduler_committed_txn.committed,
            "rolled_back": scheduler_committed_txn.rolled_back,
        },
        "scheduler_kv_rollback": {
            "transaction_id": scheduler_rolled_txn.transaction_id,
            "request_ids": list(scheduler_rolled_txn.request_ids),
            "committed": scheduler_rolled_txn.committed,
            "rolled_back": scheduler_rolled_txn.rolled_back,
        },
        "scheduler_accept_finalize": {
            "completed_request_ids": [item.request_id for item in scheduler_completed],
            "active_generated_counts": {
                str(request_id): len(request.generated_tokens)
                for request_id, request in scheduler.active_batch.requests.items()
            },
            "completed_generated_counts": {
                str(request_id): len(done.generated_tokens)
                for request_id, done in scheduler.completed.items()
            },
        },
    }


def build_payload(*, batch_artifact: Path, prefill_artifact: Path, argv: Sequence[str] | None = None) -> dict[str, Any]:
    batch = _load_json(batch_artifact)
    prefill = _load_json(prefill_artifact)
    batch_execution = batch.get("execution", {}).get("batch_execution", {})
    native_prefill_plan = batch_execution.get("native_prefill_plan") or prefill.get("native_prefill_plan") or prefill.get("plan") or {}
    resident_api = _resident_api_status()
    blockers = [
        *(f"resident speculative status: {blocker}" for blocker in resident_api.get("blockers", ())),
        "step_batch_serial is still the only c>N target path and executes rows through the c=1 layer path",
        "exact selectable per-row target state from a native root+candidate target forward is not exposed",
        "GPU accept-summary and state/KV commit kernels exist but are not yet wired into an integrated native verifier loop",
        "native compact/full-attention prefill remains incomplete, so speculative verify rows cannot share the final c>N target forward required by DFlash/DDTree",
    ]
    if batch_execution.get("throughput_claim_eligible") is False:
        blockers.append("latest c=8 scheduler artifact reports batch_execution.throughput_claim_eligible=false")
    if native_prefill_plan and not native_prefill_plan.get("full_layer_limit_native", False):
        blockers.append(
            "native prefill plan stops at linear prefix layer "
            f"{native_prefill_plan.get('linear_prefix_layers')} with first unsupported layer "
            f"{native_prefill_plan.get('first_unsupported_layer')} ({native_prefill_plan.get('first_unsupported_type')})"
        )
    return {
        "schema": 1,
        "status": "blocked",
        "summary": "Qwen3.5/PARO DFlash/DDTree native speculative decoding is blocked on native target verification and selectable state commit",
        "model": "Qwen3.5-35B-A3B-PARO",
        "quant": "w4_paro",
        "backend": "hip_gfx1100",
        "command": _command(argv),
        "performance_claim": False,
        "specdec_enabled": False,
        "implementation_status": {
            "interfaces_present": _interface_status(),
            "kv_transaction_target_verify": _kv_transaction_status(),
            "resident_api": resident_api,
            "native_target_verify_ready": bool(resident_api["native_target_verify_ready"]),
        },
        "evidence": {
            "batch_artifact": str(batch_artifact),
            "batch_status": batch.get("status"),
            "batch_performance_claim": batch.get("performance_claim"),
            "batch_workload": batch.get("workload"),
            "batch_execution": batch_execution,
            "prefill_artifact": str(prefill_artifact),
            "native_prefill_plan": native_prefill_plan,
            "docs": ["docs/DFLASH.md", "docs/MTP.md", "docs/BENCHMARK.md"],
        },
        "blockers": blockers,
        "required_next_actions": [
            "Complete Task #15 native compact/c-aware target path first: batched verify rows must not execute through step_batch_serial.",
            "Wire Qwen35ParoResidentSession.verify_speculative_batch metadata into an actual native root+candidate target forward over TargetVerifyBuffers.",
            "Wire Qwen35ParoResidentSession.commit_verified_state into the integrated native verifier loop with real verifier scratch buffers.",
            "Connect GPU top1/accept-summary buffers and deterministic equality gates before any speculative throughput artifact can be accepted.",
        ],
        "decision": {
            "accepted": False,
            "reason": "Speculative interfaces are present, but native DFlash/DDTree cannot be implemented as a throughput path until the target verifier/commit dependencies above land.",
        },
        "notes": [
            "This is a blocker artifact, not a benchmark result.",
            "It completes the current Task #44 evidence requirement without promoting speculative decoding performance claims.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-artifact", type=Path, default=DEFAULT_BATCH_ARTIFACT)
    parser.add_argument("--prefill-artifact", type=Path, default=DEFAULT_PREFILL_ARTIFACT)
    parser.add_argument("--json", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    payload = build_payload(batch_artifact=args.batch_artifact, prefill_artifact=args.prefill_artifact, argv=argv)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
