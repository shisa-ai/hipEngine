#!/usr/bin/env python3
"""Smoke-test DFlash verified-state/KV commit copies.

The script exercises the compact device path: target_top1 is already compact,
`dflash_accept_chain_i32` writes per-request summaries, and
`dflash_commit_chain_i32` copies only the accepted verifier rows into canonical
state/KV/output buffers.  It never replays accepted prefixes through the target
model, and rejected suffix rows are checked for non-leakage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.speculative import (
    build_dflash_accept,
    build_dflash_commit,
    dflash_accept_chain_i32,
    dflash_commit_chain_i32,
)
from hipengine.speculative import DFlashDraftRequest, TargetAcceptSummary, TargetCommitPlan, TargetStateCommitBuffers, TargetVerifyBatch, compile_dflash_chain


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiler-version-file",
        type=Path,
        default=None,
        help="Read precomputed hipcc --version text before loading cached HIP libraries.",
    )
    parser.add_argument(
        "--require-cached-build",
        action="store_true",
        help="Fail instead of invoking hipcc if the expected HIP cache artifact is absent.",
    )
    args = parser.parse_args()
    compiler_version = _read_text(args.compiler_version_file) if args.compiler_version_file else None

    runtime = get_hip_runtime()
    accept_lib = build_dflash_accept(load=True, compiler_version=compiler_version, require_cached=args.require_cached_build)
    commit_lib = build_dflash_commit(load=True, compiler_version=compiler_version, require_cached=args.require_cached_build)

    target = _single_request_target()
    _run_commit_case(runtime, accept_lib, commit_lib, "reject", target, (99, 31, 32))
    _run_commit_case(runtime, accept_lib, commit_lib, "partial", target, (31, 99, 32))
    _run_commit_case(runtime, accept_lib, commit_lib, "full", target, (31, 32, 44))
    _run_commit_case(runtime, accept_lib, commit_lib, "budgeted_no_bonus", target, (31, 32, 44), remaining_decode=(2,))
    _run_commit_case(runtime, accept_lib, commit_lib, "multi_prefix", _multi_request_target(), (41, 51, 42, 77, 99, 88))
    print("dflash_commit_chain_smoke passed (copy/select only; no accepted-prefix target re-forward)")
    return 0


def _run_commit_case(
    runtime,
    accept_lib,
    commit_lib,
    name: str,
    target: TargetVerifyBatch,
    target_top1: Sequence[int],
    *,
    remaining_decode: Sequence[int] | None = None,
) -> None:
    rows = target.rows
    request_count = len(target.request_ids)
    output_src_stride = rows
    output_dst_stride = rows + 1
    linear_width = 7
    kv_width = 5
    tap_count = 2
    hidden_width = 6
    device = Device("hip", 0)

    result = target.accept_from_top1(target_top1, transaction_id=11, remaining_decode=remaining_decode)
    summary = TargetAcceptSummary.from_accept_result(target, result)
    plan = TargetCommitPlan(
        transaction_id=11,
        request_ids=summary.request_ids,
        accepted_counts=summary.accepted_counts,
        commit_rows=summary.commit_rows,
        commit_tokens=summary.commit_tokens,
        commit_positions=summary.commit_positions,
        next_tokens=summary.next_tokens,
        candidate_counts=summary.candidate_counts,
        draft_depth=summary.draft_depth,
        tree_shape=summary.tree_shape,
        mode=summary.mode,
    )

    token_ids = np.asarray(target.tokens, dtype=np.int32)
    positions = np.asarray(target.positions, dtype=np.int32)
    parent_rows = np.asarray(target.parent_rows, dtype=np.int32)
    draft_depths = np.asarray(target.draft_depths, dtype=np.int32)
    active_mask = np.asarray(target.active_mask, dtype=np.uint8)
    top1 = np.asarray(target_top1, dtype=np.int32)
    remaining = None if remaining_decode is None else np.asarray(remaining_decode, dtype=np.int32)

    linear_src = _pattern_u16((rows, linear_width), base=100)
    kv_src = _pattern_u16((rows, kv_width), base=1000)
    hidden_src = _pattern_u16((tap_count, rows, hidden_width), base=2000)
    linear_dst = np.empty((request_count, linear_width), dtype=np.uint16)
    kv_dst = np.empty((max(1, target.candidate_count), kv_width), dtype=np.uint16)
    hidden_dst = np.empty((tap_count, request_count, hidden_width), dtype=np.uint16)
    output_ids_dst = np.empty((request_count, output_dst_stride), dtype=np.int32)
    output_lengths_dst = np.empty((request_count,), dtype=np.int32)
    last_positions_dst = np.empty((request_count,), dtype=np.int32)
    context_lengths_dst = np.empty((request_count,), dtype=np.int32)

    accepted_counts = np.empty((request_count,), dtype=np.int32)
    commit_rows = np.empty((request_count,), dtype=np.int32)
    commit_tokens = np.empty((request_count,), dtype=np.int32)
    commit_positions = np.empty((request_count,), dtype=np.int32)
    next_tokens = np.empty((request_count,), dtype=np.int32)
    full_accept = np.empty((request_count,), dtype=np.uint8)
    committed_output_ids = np.empty((request_count, output_src_stride), dtype=np.int32)
    committed_output_lengths = np.empty((request_count,), dtype=np.int32)

    buffers = []
    try:
        token_ids_dev = _dev(runtime, buffers, token_ids)
        positions_dev = _dev(runtime, buffers, positions)
        parent_rows_dev = _dev(runtime, buffers, parent_rows)
        draft_depths_dev = _dev(runtime, buffers, draft_depths)
        active_mask_dev = _dev(runtime, buffers, active_mask)
        top1_dev = _dev(runtime, buffers, top1)
        remaining_dev = None if remaining is None else _dev(runtime, buffers, remaining)
        accepted_counts_dev = _empty_dev(runtime, buffers, accepted_counts)
        commit_rows_dev = _empty_dev(runtime, buffers, commit_rows)
        commit_tokens_dev = _empty_dev(runtime, buffers, commit_tokens)
        commit_positions_dev = _empty_dev(runtime, buffers, commit_positions)
        next_tokens_dev = _empty_dev(runtime, buffers, next_tokens)
        full_accept_dev = _empty_dev(runtime, buffers, full_accept)
        committed_output_ids_dev = _empty_dev(runtime, buffers, committed_output_ids)
        committed_output_lengths_dev = _empty_dev(runtime, buffers, committed_output_lengths)

        dflash_accept_chain_i32(
            token_ids_dev.ptr,
            positions_dev.ptr,
            parent_rows_dev.ptr,
            draft_depths_dev.ptr,
            active_mask_dev.ptr,
            top1_dev.ptr,
            None if remaining_dev is None else remaining_dev.ptr,
            accepted_counts_dev.ptr,
            commit_rows_dev.ptr,
            commit_tokens_dev.ptr,
            commit_positions_dev.ptr,
            next_tokens_dev.ptr,
            full_accept_dev.ptr,
            committed_output_ids_dev.ptr,
            committed_output_lengths_dev.ptr,
            rows,
            request_count,
            output_src_stride,
            library=accept_lib,
            runtime=runtime,
        )

        linear_src_dev = _dev(runtime, buffers, linear_src)
        kv_src_dev = _dev(runtime, buffers, kv_src)
        hidden_src_dev = _dev(runtime, buffers, hidden_src)
        linear_dst_dev = _empty_dev(runtime, buffers, linear_dst)
        kv_dst_dev = _empty_dev(runtime, buffers, kv_dst)
        hidden_dst_dev = _empty_dev(runtime, buffers, hidden_dst)
        output_ids_dst_dev = _empty_dev(runtime, buffers, output_ids_dst)
        output_lengths_dst_dev = _empty_dev(runtime, buffers, output_lengths_dst)
        last_positions_dst_dev = _empty_dev(runtime, buffers, last_positions_dst)
        context_lengths_dst_dev = _empty_dev(runtime, buffers, context_lengths_dst)

        commit_buffers = TargetStateCommitBuffers.for_plan(
            plan,
            accepted_counts=_tensor(accepted_counts_dev, accepted_counts.shape, DType.INT32, device),
            commit_rows=_tensor(commit_rows_dev, commit_rows.shape, DType.INT32, device),
            commit_positions=_tensor(commit_positions_dev, commit_positions.shape, DType.INT32, device),
            parent_rows=_tensor(parent_rows_dev, parent_rows.shape, DType.INT32, device),
            linear_state_src=_tensor(linear_src_dev, linear_src.shape, DType.BF16, device),
            linear_state_dst=_tensor(linear_dst_dev, linear_dst.shape, DType.BF16, device),
            kv_rows_src=_tensor(kv_src_dev, kv_src.shape, DType.BF16, device),
            kv_rows_dst=_tensor(kv_dst_dev, kv_dst.shape, DType.BF16, device),
            hidden_taps_src=_tensor(hidden_src_dev, hidden_src.shape, DType.BF16, device),
            hidden_taps_dst=_tensor(hidden_dst_dev, hidden_dst.shape, DType.BF16, device),
            next_tokens_src=_tensor(next_tokens_dev, next_tokens.shape, DType.INT32, device),
            committed_output_ids_src=_tensor(committed_output_ids_dev, committed_output_ids.shape, DType.INT32, device),
            committed_output_lengths_src=_tensor(committed_output_lengths_dev, committed_output_lengths.shape, DType.INT32, device),
            output_ids_dst=_tensor(output_ids_dst_dev, output_ids_dst.shape, DType.INT32, device),
            output_lengths_dst=_tensor(output_lengths_dst_dev, output_lengths_dst.shape, DType.INT32, device),
            last_positions_dst=_tensor(last_positions_dst_dev, last_positions_dst.shape, DType.INT32, device),
            context_lengths_dst=_tensor(context_lengths_dst_dev, context_lengths_dst.shape, DType.INT32, device),
        )
        dflash_commit_chain_i32(
            commit_buffers,
            target_rows=rows,
            accepted_rows=sum(summary.accepted_counts),
            library=commit_lib,
            runtime=runtime,
        )
        runtime.device_synchronize()

        for host, dev in (
            (accepted_counts, accepted_counts_dev),
            (commit_rows, commit_rows_dev),
            (commit_tokens, commit_tokens_dev),
            (commit_positions, commit_positions_dev),
            (next_tokens, next_tokens_dev),
            (committed_output_ids, committed_output_ids_dev),
            (committed_output_lengths, committed_output_lengths_dev),
            (linear_dst, linear_dst_dev),
            (kv_dst, kv_dst_dev),
            (hidden_dst, hidden_dst_dev),
            (output_ids_dst, output_ids_dst_dev),
            (output_lengths_dst, output_lengths_dst_dev),
            (last_positions_dst, last_positions_dst_dev),
            (context_lengths_dst, context_lengths_dst_dev),
        ):
            copy_device_to_host(host_array_ptr(host), dev, runtime=runtime)
    finally:
        _free_all(runtime, buffers)

    np.testing.assert_array_equal(accepted_counts, np.asarray(summary.accepted_counts, dtype=np.int32))
    np.testing.assert_array_equal(commit_rows, np.asarray(summary.commit_rows, dtype=np.int32))
    np.testing.assert_array_equal(commit_tokens, np.asarray(summary.commit_tokens, dtype=np.int32))
    np.testing.assert_array_equal(commit_positions, np.asarray(summary.commit_positions, dtype=np.int32))
    expected_next = np.asarray([-1 if token is None else token for token in (summary.next_tokens or ())], dtype=np.int32)
    np.testing.assert_array_equal(next_tokens, expected_next)

    expected_linear = np.stack([linear_src[row] for row in summary.commit_rows])
    np.testing.assert_array_equal(linear_dst, expected_linear)
    expected_hidden = np.stack([hidden_src[:, row, :] for row in summary.commit_rows], axis=1)
    np.testing.assert_array_equal(hidden_dst, expected_hidden)
    expected_kv = np.zeros_like(kv_dst)
    cursor = 0
    for row, count in zip(summary.commit_rows, summary.accepted_counts, strict=True):
        for path_row in _path_rows(target, row, count):
            expected_kv[cursor] = kv_src[path_row]
            cursor += 1
    np.testing.assert_array_equal(kv_dst, expected_kv)

    expected_output_ids = np.full_like(output_ids_dst, -1)
    expected_output_lengths = np.zeros_like(output_lengths_dst)
    for request_index, accepted_tokens in enumerate(summary.accepted_tokens):
        ids = [target.tokens[target.root_rows[request_index]], *accepted_tokens]
        next_token = None if summary.next_tokens is None else summary.next_tokens[request_index]
        if next_token is not None:
            ids.append(next_token)
        expected_output_lengths[request_index] = min(len(ids), output_dst_stride)
        expected_output_ids[request_index, : expected_output_lengths[request_index]] = ids[:output_dst_stride]
    np.testing.assert_array_equal(output_ids_dst, expected_output_ids)
    np.testing.assert_array_equal(output_lengths_dst, expected_output_lengths)
    np.testing.assert_array_equal(last_positions_dst, np.asarray(summary.commit_positions, dtype=np.int32))
    np.testing.assert_array_equal(context_lengths_dst, np.asarray(summary.commit_positions, dtype=np.int32) + 1)

    accepted_path_rows = {
        path_row
        for row, count in zip(summary.commit_rows, summary.accepted_counts, strict=True)
        for path_row in _path_rows(target, row, count)
    }
    rejected_rows = set(target.candidate_rows) - accepted_path_rows
    for rejected in rejected_rows:
        assert not np.any(np.all(linear_dst == linear_src[rejected], axis=1)), f"{name}: rejected linear row leaked"
        assert not np.any(np.all(kv_dst == kv_src[rejected], axis=1)), f"{name}: rejected KV row leaked"
    print(
        f"{name}: accepted={list(summary.accepted_counts)} commit_rows={list(summary.commit_rows)} "
        f"kv_rows={cursor} output_lengths={output_lengths_dst.tolist()}"
    )


def _single_request_target() -> TargetVerifyBatch:
    draft = compile_dflash_chain(
        [DFlashDraftRequest(request_id=7, root_position=8, candidate_tokens=(31, 32))],
        candidate_budget=2,
    )
    return TargetVerifyBatch.from_draft(draft, root_tokens=(30,), root_positions=(8,))


def _multi_request_target() -> TargetVerifyBatch:
    draft = compile_dflash_chain(
        [
            DFlashDraftRequest(request_id=10, root_position=5, candidate_tokens=(41, 42)),
            DFlashDraftRequest(request_id=20, root_position=12, candidate_tokens=(51, 52)),
        ],
        candidate_budget=2,
    )
    return TargetVerifyBatch.from_draft(draft, root_tokens=(40, 50), root_positions=(5, 12))


def _path_rows(target: TargetVerifyBatch, selected_row: int, accepted_count: int) -> tuple[int, ...]:
    if accepted_count == 0:
        return ()
    rows = []
    row = int(selected_row)
    roots = set(target.root_rows)
    while row not in roots:
        rows.append(row)
        row = target.parent_rows[row]
    rows.reverse()
    assert len(rows) == accepted_count
    return tuple(rows)


def _pattern_u16(shape: tuple[int, ...], *, base: int) -> np.ndarray:
    return (np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) + np.uint16(base)).astype(np.uint16)


def _tensor(buffer, shape: tuple[int, ...], dtype: DType, device: Device) -> Tensor:
    return Tensor.from_handle(buffer.ptr, shape, dtype, device)


def _dev(runtime, buffers: list, array: np.ndarray):
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(contiguous.nbytes, runtime=runtime)
    buffers.append(buffer)
    copy_host_to_device(buffer, host_array_ptr(contiguous), runtime=runtime)
    return buffer


def _empty_dev(runtime, buffers: list, array: np.ndarray):
    contiguous = np.ascontiguousarray(array)
    buffer = malloc(contiguous.nbytes, runtime=runtime)
    buffers.append(buffer)
    return buffer


def _free_all(runtime, buffers: list) -> None:
    for buffer in reversed(buffers):
        free(buffer, runtime=runtime)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
