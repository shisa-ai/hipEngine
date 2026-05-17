#!/usr/bin/env python3
"""Smoke-test DFlash target top1 + chain accept kernels.

The fast-path section intentionally keeps full logits on device: row-wise argmax
fills ``target_top1`` and the accept kernel consumes that buffer directly.  The
script only reads compact top1/debug summaries and per-request accept outputs.
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

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.linear import (
    argmax_f32_rows_i32,
    build_lm_head,
    lm_head_argmax_stage1_blocks,
    lm_head_fp16_argmax_bf16_rows_i32,
)
from hipengine.kernels.hip_gfx1100.speculative import build_dflash_accept, dflash_accept_chain_i32
from hipengine.speculative import DFlashDraftRequest, TargetAcceptSummary, TargetVerifyBatch, compile_dflash_chain


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
    parser.add_argument(
        "--debug-top1-readback",
        action="store_true",
        help="Read back target_top1 ids after argmax for debugging. The accept fast path does not require this.",
    )
    args = parser.parse_args()
    compiler_version = _read_text(args.compiler_version_file) if args.compiler_version_file else None

    runtime = get_hip_runtime()
    lm_head_lib = build_lm_head(load=True, compiler_version=compiler_version, require_cached=args.require_cached_build)
    accept_lib = build_dflash_accept(load=True, compiler_version=compiler_version, require_cached=args.require_cached_build)

    _run_lm_head_rows_top1_case(runtime, lm_head_lib)
    _run_single_request_accept_patterns(runtime, lm_head_lib, accept_lib, debug_top1=args.debug_top1_readback)
    _run_multi_request_real_layout_case(runtime, lm_head_lib, accept_lib, debug_top1=args.debug_top1_readback)
    _run_budgeted_accept_case(runtime, lm_head_lib, accept_lib, debug_top1=args.debug_top1_readback)
    print("dflash_accept_chain_smoke passed")
    return 0


def _run_lm_head_rows_top1_case(runtime, library) -> None:
    rows = 4
    hidden_size = 16
    vocab_size = 19
    threads = 128
    hidden_f32 = _pattern((rows, hidden_size), scale=1.0 / 17.0, offset=-0.4, mul=5, add=3)
    hidden_bits = _float32_to_bf16_bits(hidden_f32)
    hidden_ref = _bf16_bits_to_float32(hidden_bits)
    weight_fp16 = _pattern((vocab_size, hidden_size), scale=1.0 / 13.0, offset=-0.3, mul=7, add=1).astype(np.float16)
    expected_logits = hidden_ref.astype(np.float32) @ weight_fp16.astype(np.float32).T
    expected_ids = np.argmax(expected_logits, axis=1).astype(np.int32)
    expected_values = expected_logits[np.arange(rows), expected_ids].astype(np.float32)

    stage1_blocks = lm_head_argmax_stage1_blocks(vocab_size, threads=threads)
    logits = np.empty((rows, vocab_size), dtype=np.float32)
    block_values = np.empty((rows, stage1_blocks), dtype=np.float32)
    block_indices = np.empty((rows, stage1_blocks), dtype=np.int32)
    out_ids = np.empty((rows,), dtype=np.int32)
    out_values = np.empty((rows,), dtype=np.float32)

    buffers = []
    try:
        hidden_dev = _dev(runtime, buffers, hidden_bits)
        weight_dev = _dev(runtime, buffers, weight_fp16)
        logits_dev = _empty_dev(runtime, buffers, logits)
        block_values_dev = _empty_dev(runtime, buffers, block_values)
        block_indices_dev = _empty_dev(runtime, buffers, block_indices)
        out_ids_dev = _empty_dev(runtime, buffers, out_ids)
        out_values_dev = _empty_dev(runtime, buffers, out_values)
        lm_head_fp16_argmax_bf16_rows_i32(
            hidden_dev.ptr,
            weight_dev.ptr,
            logits_dev.ptr,
            block_values_dev.ptr,
            block_indices_dev.ptr,
            out_ids_dev.ptr,
            out_values_dev.ptr,
            rows,
            hidden_size,
            vocab_size,
            threads=threads,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out_ids), out_ids_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(out_values), out_values_dev, runtime=runtime)
    finally:
        _free_all(runtime, buffers)

    np.testing.assert_array_equal(out_ids, expected_ids)
    np.testing.assert_allclose(out_values, expected_values, rtol=1.0e-3, atol=1.0e-3)
    print(f"lm_head_rows_top1 rows={rows} vocab={vocab_size} ids={out_ids.tolist()}")


def _run_single_request_accept_patterns(runtime, lm_head_lib, accept_lib, *, debug_top1: bool) -> None:
    target = _single_request_target()
    for name, top1 in {
        "reject": (99, 31, 32),
        "partial": (31, 99, 44),
        "full": (31, 32, 44),
    }.items():
        observed = _run_accept_from_top1(
            runtime,
            lm_head_lib,
            accept_lib,
            target,
            top1,
            remaining_decode=None,
            debug_top1=debug_top1,
        )
        assert observed["accepted_counts"].tolist() == [target.accept_from_top1(top1).accepted_counts[0]]
        print(
            f"single_{name} accepted={observed['accepted_counts'].tolist()} "
            f"commit_rows={observed['commit_rows'].tolist()} next={observed['next_tokens'].tolist()}"
        )


def _run_multi_request_real_layout_case(runtime, lm_head_lib, accept_lib, *, debug_top1: bool) -> None:
    draft = compile_dflash_chain(
        [
            DFlashDraftRequest(request_id=10, root_position=5, candidate_tokens=(41, 42, 43, 44)),
            DFlashDraftRequest(request_id=20, root_position=12, candidate_tokens=(51, 52, 53), active_count=3),
        ],
        candidate_budget=4,
        pad_token_id=0,
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(40, 50), root_positions=(5, 12))
    top1 = (41, 51, 42, 43, 44, 77, 52, 99, 88, 55)
    observed = _run_accept_from_top1(
        runtime,
        lm_head_lib,
        accept_lib,
        target,
        top1,
        remaining_decode=None,
        debug_top1=debug_top1,
    )
    print(
        "multi_real_layout "
        f"accepted={observed['accepted_counts'].tolist()} full={observed['full_accept'].tolist()} "
        f"lengths={observed['committed_output_lengths'].tolist()}"
    )


def _run_budgeted_accept_case(runtime, lm_head_lib, accept_lib, *, debug_top1: bool) -> None:
    draft = compile_dflash_chain(
        [
            DFlashDraftRequest(request_id=10, root_position=5, candidate_tokens=(41, 42, 43, 44)),
            DFlashDraftRequest(request_id=20, root_position=12, candidate_tokens=(51, 52, 53), active_count=3),
        ],
        candidate_budget=4,
        pad_token_id=0,
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(40, 50), root_positions=(5, 12))
    top1 = (41, 51, 42, 43, 44, 77, 52, 53, 88, 55)
    observed = _run_accept_from_top1(
        runtime,
        lm_head_lib,
        accept_lib,
        target,
        top1,
        remaining_decode=(2, 1),
        debug_top1=debug_top1,
    )
    assert observed["next_tokens"].tolist() == [-1, -1]
    print(
        "budgeted_real_layout "
        f"accepted={observed['accepted_counts'].tolist()} next={observed['next_tokens'].tolist()}"
    )


def _run_accept_from_top1(
    runtime,
    lm_head_lib,
    accept_lib,
    target: TargetVerifyBatch,
    target_top1: Sequence[int],
    *,
    remaining_decode: Sequence[int] | None,
    debug_top1: bool,
) -> dict[str, np.ndarray]:
    rows = target.rows
    request_count = len(target.request_ids)
    vocab_size = max(max(target_top1), max(target.tokens), 16) + 8
    threads = 128
    stage1_blocks = lm_head_argmax_stage1_blocks(vocab_size, threads=threads)
    logits = _logits_for_top1(target_top1, vocab_size)

    token_ids = np.asarray(target.tokens, dtype=np.int32)
    positions = np.asarray(target.positions, dtype=np.int32)
    parent_rows = np.asarray(target.parent_rows, dtype=np.int32)
    draft_depths = np.asarray(target.draft_depths, dtype=np.int32)
    active_mask = np.asarray(target.active_mask, dtype=np.uint8)
    remaining = None if remaining_decode is None else np.asarray(remaining_decode, dtype=np.int32)

    block_values = np.empty((rows, stage1_blocks), dtype=np.float32)
    block_indices = np.empty((rows, stage1_blocks), dtype=np.int32)
    target_top1_out = np.empty((rows,), dtype=np.int32)
    accepted_counts = np.empty((request_count,), dtype=np.int32)
    commit_rows = np.empty((request_count,), dtype=np.int32)
    commit_tokens = np.empty((request_count,), dtype=np.int32)
    commit_positions = np.empty((request_count,), dtype=np.int32)
    next_tokens = np.empty((request_count,), dtype=np.int32)
    full_accept = np.empty((request_count,), dtype=np.uint8)
    output_stride = rows
    committed_output_ids = np.empty((request_count, output_stride), dtype=np.int32)
    committed_output_lengths = np.empty((request_count,), dtype=np.int32)

    buffers = []
    try:
        logits_dev = _dev(runtime, buffers, logits)
        block_values_dev = _empty_dev(runtime, buffers, block_values)
        block_indices_dev = _empty_dev(runtime, buffers, block_indices)
        target_top1_dev = _empty_dev(runtime, buffers, target_top1_out)
        argmax_f32_rows_i32(
            logits_dev.ptr,
            block_values_dev.ptr,
            block_indices_dev.ptr,
            target_top1_dev.ptr,
            None,
            rows,
            vocab_size,
            threads=threads,
            library=lm_head_lib,
            runtime=runtime,
        )
        token_ids_dev = _dev(runtime, buffers, token_ids)
        positions_dev = _dev(runtime, buffers, positions)
        parent_rows_dev = _dev(runtime, buffers, parent_rows)
        draft_depths_dev = _dev(runtime, buffers, draft_depths)
        active_mask_dev = _dev(runtime, buffers, active_mask)
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
            target_top1_dev.ptr,
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
            output_stride,
            library=accept_lib,
            runtime=runtime,
        )
        runtime.device_synchronize()
        if debug_top1:
            copy_device_to_host(host_array_ptr(target_top1_out), target_top1_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(accepted_counts), accepted_counts_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(commit_rows), commit_rows_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(commit_tokens), commit_tokens_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(commit_positions), commit_positions_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(next_tokens), next_tokens_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(full_accept), full_accept_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(committed_output_ids), committed_output_ids_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(committed_output_lengths), committed_output_lengths_dev, runtime=runtime)
    finally:
        _free_all(runtime, buffers)

    expected_top1 = tuple(int(x) for x in target_top1)
    if debug_top1:
        np.testing.assert_array_equal(target_top1_out, np.asarray(expected_top1, dtype=np.int32))
    result = target.accept_from_top1(expected_top1, remaining_decode=remaining_decode)
    summary = TargetAcceptSummary.from_accept_result(target, result)
    expected_next = np.asarray([-1 if token is None else int(token) for token in (summary.next_tokens or ())], dtype=np.int32)

    np.testing.assert_array_equal(accepted_counts, np.asarray(summary.accepted_counts, dtype=np.int32))
    np.testing.assert_array_equal(commit_rows, np.asarray(summary.commit_rows, dtype=np.int32))
    np.testing.assert_array_equal(commit_tokens, np.asarray(summary.commit_tokens, dtype=np.int32))
    np.testing.assert_array_equal(commit_positions, np.asarray(summary.commit_positions, dtype=np.int32))
    np.testing.assert_array_equal(next_tokens, expected_next)
    np.testing.assert_array_equal(full_accept.astype(np.bool_), np.asarray(summary.full_accept, dtype=np.bool_))
    np.testing.assert_array_equal(committed_output_lengths, accepted_counts + 1)
    for request_index, accepted in enumerate(summary.accepted_tokens):
        expected_ids = [target.tokens[target.root_rows[request_index]], *accepted]
        length = len(expected_ids)
        np.testing.assert_array_equal(committed_output_ids[request_index, :length], np.asarray(expected_ids, dtype=np.int32))
        np.testing.assert_array_equal(committed_output_ids[request_index, length:], np.full(output_stride - length, -1, dtype=np.int32))

    return {
        "accepted_counts": accepted_counts,
        "commit_rows": commit_rows,
        "commit_tokens": commit_tokens,
        "commit_positions": commit_positions,
        "next_tokens": next_tokens,
        "full_accept": full_accept.astype(np.bool_),
        "committed_output_ids": committed_output_ids,
        "committed_output_lengths": committed_output_lengths,
    }


def _single_request_target() -> TargetVerifyBatch:
    draft = compile_dflash_chain(
        [DFlashDraftRequest(request_id=7, root_position=8, candidate_tokens=(31, 32))],
        candidate_budget=2,
    )
    return TargetVerifyBatch.from_draft(draft, root_tokens=(30,), root_positions=(8,))


def _logits_for_top1(top1: Sequence[int], vocab_size: int) -> np.ndarray:
    rows = len(top1)
    base = -np.arange(vocab_size, dtype=np.float32) * np.float32(1.0e-3)
    logits = np.vstack([base - np.float32(row) * np.float32(1.0e-4) for row in range(rows)]).astype(np.float32)
    for row, token in enumerate(top1):
        logits[row, int(token)] = np.float32(10.0 + row)
    return np.ascontiguousarray(logits)


def _pattern(
    shape: tuple[int, ...],
    *,
    scale: float,
    offset: float,
    mul: int = 7,
    add: int = 3,
) -> np.ndarray:
    indices = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    values = ((indices * mul + add) % 31).astype(np.float32) * np.float32(scale) + np.float32(offset)
    return values.astype(np.float32)


def _float32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    bits = arr.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    arr = np.asarray(bits, dtype=np.uint16)
    widened = arr.astype(np.uint32) << np.uint32(16)
    return widened.view(np.float32)


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
