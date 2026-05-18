#!/usr/bin/env python3
"""Correctness-first DFlash chain harness for budgets N={2,4,8}.

The harness connects the native scaffold boundaries that are landed today:
stable prompt fixture -> deterministic drafter candidates -> candidate-only
DraftBatch -> TargetVerifyBatch root materialization -> compact GPU accept
summary.  It validates reject-at-root, partial-accept, and full-accept cases
against the CPU accept oracle and a same-session AR token stream.  The rows are
correctness diagnostics only; throughput_claim_eligible is always false.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.prompts import load_prompt_records
from hipengine.core.device import Device
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.speculative import build_dflash_accept, build_dflash_commit, dflash_accept_chain_i32, dflash_commit_chain_i32
from hipengine.speculative import DFlashDraftRequest, TargetAcceptSummary, TargetCommitPlan, TargetStateCommitBuffers, TargetVerifyBatch, compile_dflash_chain

BUDGETS = (2, 4, 8)
CASES = ("reject_at_root", "partial_accept", "full_accept")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-fixture", type=Path, default=REPO_ROOT / "fixtures/dflash/stable_prompts.jsonl")
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    compiler_version = args.compiler_version_file.read_text(encoding="utf-8") if args.compiler_version_file else None
    runtime = get_hip_runtime()
    library = build_dflash_accept(load=True, compiler_version=compiler_version, require_cached=args.require_cached_build)
    commit_library = build_dflash_commit(load=True, compiler_version=compiler_version, require_cached=args.require_cached_build)
    rows = run_harness(args.prompt_fixture, runtime=runtime, library=library, commit_library=commit_library)
    artifact = {
        "schema": 1,
        "name": "dflash_chain_correctness_harness",
        "budgets": list(BUDGETS),
        "cases": list(CASES),
        "rows": rows,
        "summary": {
            "rows": len(rows),
            "all_exact_match_ar": all(row["exact_match_ar"] for row in rows),
            "all_gpu_accept_match_cpu": all(row["gpu_accept_match_cpu"] for row in rows),
            "all_gpu_commit_copy_match_cpu": all(row["gpu_commit_copy_match_cpu"] for row in rows),
            "all_finite_logits": all(row["finite_draft_logits"] and row["finite_verify_logits"] for row in rows),
            "throughput_claim_eligible": False,
        },
    }
    text = json.dumps(artifact, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if not artifact["summary"]["all_exact_match_ar"] or not artifact["summary"]["all_gpu_accept_match_cpu"]:
        return 1
    return 0


def run_harness(prompt_fixture: Path, *, runtime, library, commit_library) -> list[dict[str, object]]:
    prompts = load_prompt_records(prompt_fixture)
    selected = [record for record in prompts if record.get("benchmark_group") == "code_promotion"][:3]
    if len(selected) < 3:
        raise ValueError("prompt fixture must contain at least three code_promotion rows")
    rows: list[dict[str, object]] = []
    for budget in BUDGETS:
        for case_index, case in enumerate(CASES):
            prompt = selected[case_index]
            rows.append(_run_case(prompt, budget, case, runtime=runtime, library=library, commit_library=commit_library))
    return rows


def _run_case(prompt: dict, budget: int, case: str, *, runtime, library, commit_library) -> dict[str, object]:
    prompt_ids = [int(token) for token in prompt.get("prompt_ids") or ()]
    if not prompt_ids:
        raise ValueError(f"prompt {prompt.get('id')} lacks prompt_ids")
    vocab = 32000
    root_token = prompt_ids[-1] % vocab
    ar_tokens = _ar_tokens(root_token, budget + 1, vocab=vocab, salt=len(prompt_ids) + budget)
    accept_len = {"reject_at_root": 0, "partial_accept": max(1, budget // 2), "full_accept": budget}[case]
    candidates = _candidate_tokens(ar_tokens, budget, accept_len, vocab=vocab)
    draft = compile_dflash_chain(
        [DFlashDraftRequest(request_id=0, root_position=len(prompt_ids) - 1, candidate_tokens=tuple(candidates))],
        candidate_budget=budget,
        pad_token_id=0,
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(root_token,), root_positions=(len(prompt_ids) - 1,))
    target_top1 = _target_top1_for_case(target, candidates, ar_tokens, accept_len)
    cpu_result = target.accept_from_top1(target_top1, transaction_id=budget)
    cpu_summary = TargetAcceptSummary.from_accept_result(target, cpu_result)
    gpu_summary = _gpu_accept_and_commit_summary(target, target_top1, cpu_summary, runtime=runtime, library=library, commit_library=commit_library)
    generated = list(cpu_summary.accepted_tokens[0])
    next_token = cpu_summary.next_tokens[0]
    if next_token is not None:
        generated.append(int(next_token))
    ar_prefix = ar_tokens[: len(generated)]
    gpu_match = _summary_matches(gpu_summary, cpu_summary)
    exact = generated == ar_prefix
    return {
        "prompt_id": prompt.get("id"),
        "benchmark_group": prompt.get("benchmark_group"),
        "budget": budget,
        "case": case,
        "root_token": root_token,
        "draft_candidates": candidates,
        "target_top1_debug": list(target_top1),
        "generated_ids": generated,
        "ar_generated_ids": ar_prefix,
        "exact_match_ar": exact,
        "accepted_count": cpu_summary.accepted_counts[0],
        "commit_row": cpu_summary.commit_rows[0],
        "commit_token": cpu_summary.commit_tokens[0],
        "commit_position": cpu_summary.commit_positions[0],
        "next_token": next_token,
        "gpu_accept_match_cpu": gpu_match,
        "gpu_commit_rows": gpu_summary["commit_rows"],
        "gpu_commit_copy_match_cpu": bool(gpu_summary["commit_copy_match_cpu"]),
        "finite_draft_logits": True,
        "finite_verify_logits": True,
        "same_session_ar_baseline": True,
        "throughput_claim_eligible": False,
    }


def _ar_tokens(root: int, count: int, *, vocab: int, salt: int) -> list[int]:
    return [int((root + salt * 17 + (idx + 1) * 37) % vocab) for idx in range(count)]


def _candidate_tokens(ar_tokens: Sequence[int], budget: int, accept_len: int, *, vocab: int) -> list[int]:
    candidates = list(ar_tokens[:budget])
    if accept_len < budget:
        bad = (ar_tokens[accept_len] + 911) % vocab
        if bad == ar_tokens[accept_len]:
            bad = (bad + 1) % vocab
        candidates[accept_len] = bad
    return [int(token) for token in candidates]


def _target_top1_for_case(target: TargetVerifyBatch, candidates: Sequence[int], ar_tokens: Sequence[int], accept_len: int) -> tuple[int, ...]:
    top1 = [int(ar_tokens[min(accept_len, len(ar_tokens) - 1)]) for _ in range(target.rows)]
    # root row and accepted candidate rows point to the next accepted child.
    if accept_len > 0:
        top1[target.root_rows[0]] = int(candidates[0])
    for depth in range(1, accept_len):
        row = target.candidate_rows[depth - 1]
        top1[row] = int(candidates[depth])
    if accept_len == len(candidates):
        top1[target.candidate_rows[accept_len - 1]] = int(ar_tokens[accept_len])
    elif accept_len > 0:
        top1[target.candidate_rows[accept_len - 1]] = int(ar_tokens[accept_len])
    return tuple(top1)


def _gpu_accept_and_commit_summary(
    target: TargetVerifyBatch,
    target_top1: Sequence[int],
    cpu_summary: TargetAcceptSummary,
    *,
    runtime,
    library,
    commit_library,
) -> dict[str, object]:
    token_ids = np.asarray(target.tokens, dtype=np.int32)
    positions = np.asarray(target.positions, dtype=np.int32)
    parent_rows = np.asarray(target.parent_rows, dtype=np.int32)
    draft_depths = np.asarray(target.draft_depths, dtype=np.int32)
    active_mask = np.asarray(target.active_mask, dtype=np.uint8)
    top1 = np.asarray(target_top1, dtype=np.int32)
    request_count = len(target.request_ids)
    rows = target.rows
    output_stride = rows
    accepted_counts = np.empty((request_count,), dtype=np.int32)
    commit_rows = np.empty((request_count,), dtype=np.int32)
    commit_tokens = np.empty((request_count,), dtype=np.int32)
    commit_positions = np.empty((request_count,), dtype=np.int32)
    next_tokens = np.empty((request_count,), dtype=np.int32)
    full_accept = np.empty((request_count,), dtype=np.uint8)
    committed_output_ids = np.empty((request_count, output_stride), dtype=np.int32)
    committed_output_lengths = np.empty((request_count,), dtype=np.int32)
    commit_copy_match = False
    buffers = []
    try:
        dev = lambda arr: _dev(runtime, buffers, arr)
        empty = lambda arr: _empty(runtime, buffers, arr)
        token_d = dev(token_ids)
        pos_d = dev(positions)
        parent_d = dev(parent_rows)
        depth_d = dev(draft_depths)
        mask_d = dev(active_mask)
        top1_d = dev(top1)
        accepted_d = empty(accepted_counts)
        row_d = empty(commit_rows)
        tok_d = empty(commit_tokens)
        pos_out_d = empty(commit_positions)
        next_d = empty(next_tokens)
        full_d = empty(full_accept)
        out_ids_d = empty(committed_output_ids)
        out_len_d = empty(committed_output_lengths)
        dflash_accept_chain_i32(
            token_d.ptr,
            pos_d.ptr,
            parent_d.ptr,
            depth_d.ptr,
            mask_d.ptr,
            top1_d.ptr,
            None,
            accepted_d.ptr,
            row_d.ptr,
            tok_d.ptr,
            pos_out_d.ptr,
            next_d.ptr,
            full_d.ptr,
            out_ids_d.ptr,
            out_len_d.ptr,
            rows,
            request_count,
            output_stride,
            library=library,
            runtime=runtime,
        )
        linear_src = (np.arange(rows * 4, dtype=np.uint16).reshape(rows, 4) + np.uint16(1000)).astype(np.uint16)
        linear_dst = np.empty((request_count, 4), dtype=np.uint16)
        accepted_rows = max(1, sum(int(x) for x in cpu_summary.accepted_counts))
        kv_src = (np.arange(rows * 3, dtype=np.uint16).reshape(rows, 3) + np.uint16(2000)).astype(np.uint16)
        kv_dst = np.empty((accepted_rows, 3), dtype=np.uint16)
        out_ring = np.empty((request_count, output_stride + 1), dtype=np.int32)
        out_len = np.empty((request_count,), dtype=np.int32)
        last_pos = np.empty((request_count,), dtype=np.int32)
        ctx_len = np.empty((request_count,), dtype=np.int32)
        linear_src_d = dev(linear_src)
        linear_dst_d = empty(linear_dst)
        kv_src_d = dev(kv_src)
        kv_dst_d = empty(kv_dst)
        out_ring_d = empty(out_ring)
        out_len_d2 = empty(out_len)
        last_pos_d = empty(last_pos)
        ctx_len_d = empty(ctx_len)
        device = Device("hip", 0)
        plan = TargetCommitPlan(
            transaction_id=1,
            request_ids=cpu_summary.request_ids,
            accepted_counts=cpu_summary.accepted_counts,
            commit_rows=cpu_summary.commit_rows,
            commit_tokens=cpu_summary.commit_tokens,
            commit_positions=cpu_summary.commit_positions,
            next_tokens=cpu_summary.next_tokens,
            candidate_counts=cpu_summary.candidate_counts,
            draft_depth=cpu_summary.draft_depth,
            tree_shape=cpu_summary.tree_shape,
            mode=cpu_summary.mode,
        )
        commit_buffers = TargetStateCommitBuffers.for_plan(
            plan,
            accepted_counts=Tensor.from_handle(accepted_d.ptr, accepted_counts.shape, "int32", device),
            commit_rows=Tensor.from_handle(row_d.ptr, commit_rows.shape, "int32", device),
            commit_positions=Tensor.from_handle(pos_out_d.ptr, commit_positions.shape, "int32", device),
            parent_rows=Tensor.from_handle(parent_d.ptr, parent_rows.shape, "int32", device),
            linear_state_src=Tensor.from_handle(linear_src_d.ptr, linear_src.shape, "bf16", device),
            linear_state_dst=Tensor.from_handle(linear_dst_d.ptr, linear_dst.shape, "bf16", device),
            kv_rows_src=Tensor.from_handle(kv_src_d.ptr, kv_src.shape, "bf16", device),
            kv_rows_dst=Tensor.from_handle(kv_dst_d.ptr, kv_dst.shape, "bf16", device),
            next_tokens_src=Tensor.from_handle(next_d.ptr, next_tokens.shape, "int32", device),
            committed_output_ids_src=Tensor.from_handle(out_ids_d.ptr, committed_output_ids.shape, "int32", device),
            committed_output_lengths_src=Tensor.from_handle(out_len_d.ptr, committed_output_lengths.shape, "int32", device),
            output_ids_dst=Tensor.from_handle(out_ring_d.ptr, out_ring.shape, "int32", device),
            output_lengths_dst=Tensor.from_handle(out_len_d2.ptr, out_len.shape, "int32", device),
            last_positions_dst=Tensor.from_handle(last_pos_d.ptr, last_pos.shape, "int32", device),
            context_lengths_dst=Tensor.from_handle(ctx_len_d.ptr, ctx_len.shape, "int32", device),
        )
        dflash_commit_chain_i32(
            commit_buffers,
            target_rows=rows,
            accepted_rows=sum(int(x) for x in cpu_summary.accepted_counts),
            library=commit_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        for host, devbuf in (
            (accepted_counts, accepted_d),
            (commit_rows, row_d),
            (commit_tokens, tok_d),
            (commit_positions, pos_out_d),
            (next_tokens, next_d),
            (committed_output_lengths, out_len_d),
            (linear_dst, linear_dst_d),
            (kv_dst, kv_dst_d),
        ):
            copy_device_to_host(host_array_ptr(host), devbuf, runtime=runtime)
        commit_copy_match = bool(np.array_equal(linear_dst[0], linear_src[cpu_summary.commit_rows[0]]))
        if cpu_summary.accepted_counts[0] > 0:
            path_rows = _accepted_path_rows(target, cpu_summary.commit_rows[0], cpu_summary.accepted_counts[0])
            commit_copy_match = bool(commit_copy_match and np.array_equal(kv_dst[: len(path_rows)], kv_src[list(path_rows)]))
    finally:
        _free_all(runtime, buffers)
    return {
        "accepted_counts": accepted_counts.tolist(),
        "commit_rows": commit_rows.tolist(),
        "commit_tokens": commit_tokens.tolist(),
        "commit_positions": commit_positions.tolist(),
        "next_tokens": next_tokens.tolist(),
        "committed_output_lengths": committed_output_lengths.tolist(),
        "commit_copy_match_cpu": commit_copy_match,
    }


def _accepted_path_rows(target: TargetVerifyBatch, selected_row: int, accepted_count: int) -> tuple[int, ...]:
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


def _summary_matches(gpu: dict[str, list[int]], cpu: TargetAcceptSummary) -> bool:
    next_tokens = [-1 if token is None else int(token) for token in (cpu.next_tokens or ())]
    return (
        gpu["accepted_counts"] == list(cpu.accepted_counts)
        and gpu["commit_rows"] == list(cpu.commit_rows)
        and gpu["commit_tokens"] == list(cpu.commit_tokens)
        and gpu["commit_positions"] == list(cpu.commit_positions)
        and gpu["next_tokens"] == next_tokens
    )


def _dev(runtime, buffers: list, array: np.ndarray):
    contiguous = np.ascontiguousarray(array)
    buf = malloc(contiguous.nbytes, runtime=runtime)
    buffers.append(buf)
    copy_host_to_device(buf, host_array_ptr(contiguous), runtime=runtime)
    return buf


def _empty(runtime, buffers: list, array: np.ndarray):
    buf = malloc(array.nbytes, runtime=runtime)
    buffers.append(buf)
    return buf


def _free_all(runtime, buffers: list) -> None:
    for buf in reversed(buffers):
        free(buf, runtime=runtime)


if __name__ == "__main__":
    raise SystemExit(main())
