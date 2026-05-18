#!/usr/bin/env python3
"""Validate fixed-shape DFlash verify graph buckets for chain N={2,4,8}.

The smoke captures the fixed-address accept+commit launch sequence into a HIP
graph for each supported chain bucket, replays it, and compares graph outputs to
direct mode exactly. The validation artifact records captured buckets for
supported chain depths and a rare-shape direct fallback that preserves semantics.
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

from hipengine.core.device import Device
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.speculative import build_dflash_accept, build_dflash_commit, dflash_accept_chain_i32, dflash_commit_chain_i32
from hipengine.speculative import (
    DFlashDraftRequest,
    DFlashVerifyGraphAddresses,
    DFlashVerifyGraphBucketKey,
    DFlashVerifyGraphValidation,
    TargetAcceptSummary,
    TargetCommitPlan,
    TargetStateCommitBuffers,
    TargetVerifyBatch,
    compile_dflash_chain,
    fingerprint_int_arrays,
)

BUDGETS = (2, 4, 8)
CASES = ("reject_at_root", "partial_accept", "full_accept")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    compiler_version = args.compiler_version_file.read_text(encoding="utf-8") if args.compiler_version_file else None
    runtime = get_hip_runtime()
    accept_library = build_dflash_accept(load=True, compiler_version=compiler_version, require_cached=args.require_cached_build)
    commit_library = build_dflash_commit(load=True, compiler_version=compiler_version, require_cached=args.require_cached_build)
    validations = []
    for budget in BUDGETS:
        validations.append(_validate_bucket(budget, runtime=runtime, accept_library=accept_library, commit_library=commit_library))
    validations.append(_validate_rare_shape_fallback(runtime=runtime, accept_library=accept_library, commit_library=commit_library))
    artifact = {
        "schema": 1,
        "name": "dflash_verify_graph_capture_buckets",
        "status": "diagnostic",
        "performance_claim": False,
        "backend": "hip_gfx1151",
        "hardware": {"gpu": "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S", "arch": "gfx1151"},
        "commands": {"graph_validation": " ".join(sys.argv)},
        "validations": [validation.as_artifact_row() for validation in validations],
        "summary": {
            "captured_buckets": sum(1 for validation in validations if validation.status == "captured"),
            "direct_fallbacks": sum(1 for validation in validations if validation.status == "direct_fallback"),
            "all_captured_match_direct": all(
                validation.direct_match and validation.graph_validation_passed
                for validation in validations
                if validation.status == "captured"
            ),
            "fallback_semantics_preserved": all(
                validation.direct_match for validation in validations if validation.status == "direct_fallback"
            ),
        },
    }
    text = json.dumps(artifact, indent=2, sort_keys=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    ok = artifact["summary"]["all_captured_match_direct"] and artifact["summary"]["fallback_semantics_preserved"]
    return 0 if ok else 1


def _validate_bucket(budget: int, *, runtime, accept_library, commit_library) -> DFlashVerifyGraphValidation:
    target, top1, cpu_summary = _build_case(budget, CASES[budget % len(CASES)])
    bucket = DFlashVerifyGraphBucketKey.from_batch(
        target,
        backend="hip_gfx1151",
        context_bucket=4096,
        page_bucket=128,
        top_k=1,
        experts_per_token=8,
        replay_steps=2,
    )
    buffers = _FixedVerifyBuffers.allocate(target, top1, runtime=runtime)
    graph = 0
    graph_exec = 0
    stream = 0
    try:
        direct = buffers.run(target, cpu_summary, accept_library=accept_library, commit_library=commit_library, runtime=runtime)
        stream = runtime.stream_create()
        runtime.stream_begin_capture(stream)
        buffers.launch(target, cpu_summary, accept_library=accept_library, commit_library=commit_library, runtime=runtime, stream=stream)
        graph = runtime.stream_end_capture(stream)
        graph_exec = runtime.graph_instantiate(graph)
        runtime.graph_launch(graph_exec, stream)
        runtime.stream_synchronize(stream)
        replay = buffers.read_outputs(runtime=runtime)
    finally:
        addresses = buffers.addresses
        if graph_exec:
            runtime.graph_exec_destroy(graph_exec)
        if graph:
            runtime.graph_destroy(graph)
        if stream:
            runtime.stream_destroy(stream)
        buffers.free(runtime=runtime)
    direct_fp = fingerprint_int_arrays(direct["fingerprint_arrays"])
    replay_fp = fingerprint_int_arrays(replay["fingerprint_arrays"])
    direct_payload = {key: value for key, value in direct.items() if key != "fingerprint_arrays"}
    replay_payload = {key: value for key, value in replay.items() if key != "fingerprint_arrays"}
    return DFlashVerifyGraphValidation(
        bucket_key=bucket,
        status="captured",
        replay_steps=2,
        fixed_addresses=addresses,
        direct_match=direct_payload == replay_payload,
        graph_validation_passed=direct_fp == replay_fp and direct_payload == replay_payload,
        direct_output_fingerprint=direct_fp,
        graph_output_fingerprint=replay_fp,
    )


def _validate_rare_shape_fallback(*, runtime, accept_library, commit_library) -> DFlashVerifyGraphValidation:
    target, top1, cpu_summary = _build_case(2, "partial_accept")
    bucket = DFlashVerifyGraphBucketKey.from_batch(
        target,
        backend="hip_gfx1151",
        context_bucket=123,
        page_bucket=17,
        top_k=1,
        experts_per_token=8,
        replay_steps=1,
    )
    buffers = _FixedVerifyBuffers.allocate(target, top1, runtime=runtime)
    try:
        direct = buffers.run(target, cpu_summary, accept_library=accept_library, commit_library=commit_library, runtime=runtime)
    finally:
        buffers.free(runtime=runtime)
    return DFlashVerifyGraphValidation(
        bucket_key=bucket,
        status="direct_fallback",
        replay_steps=1,
        direct_match=direct["accepted_counts"] == list(cpu_summary.accepted_counts),
        graph_validation_passed=None,
        fallback_reason=bucket.fallback_reason or "rare shape forced direct fallback",
        direct_output_fingerprint=fingerprint_int_arrays(direct["fingerprint_arrays"]),
        graph_output_fingerprint=None,
    )


def _build_case(budget: int, case: str) -> tuple[TargetVerifyBatch, tuple[int, ...], TargetAcceptSummary]:
    root_token = 100 + budget
    root_position = 31
    ar_tokens = [(root_token + (idx + 1) * 13) % 32000 for idx in range(budget + 1)]
    accept_len = {"reject_at_root": 0, "partial_accept": max(1, budget // 2), "full_accept": budget}[case]
    candidates = list(ar_tokens[:budget])
    if accept_len < budget:
        candidates[accept_len] = (candidates[accept_len] + 777) % 32000
    draft = compile_dflash_chain(
        [DFlashDraftRequest(request_id=0, root_position=root_position, candidate_tokens=tuple(candidates))],
        candidate_budget=budget,
        pad_token_id=0,
    )
    target = TargetVerifyBatch.from_draft(draft, root_tokens=(root_token,), root_positions=(root_position,))
    top1 = [ar_tokens[min(accept_len, len(ar_tokens) - 1)] for _ in range(target.rows)]
    if accept_len > 0:
        top1[target.root_rows[0]] = candidates[0]
        for depth in range(1, accept_len):
            top1[target.candidate_rows[depth - 1]] = candidates[depth]
        top1[target.candidate_rows[accept_len - 1]] = ar_tokens[accept_len]
    result = target.accept_from_top1(tuple(int(token) for token in top1), transaction_id=budget)
    return target, tuple(int(token) for token in top1), TargetAcceptSummary.from_accept_result(target, result)


class _FixedVerifyBuffers:
    def __init__(self, buffers: list, arrays: dict[str, np.ndarray], device_buffers: dict[str, object]) -> None:
        self._buffers = buffers
        self.arrays = arrays
        self.device_buffers = device_buffers

    @classmethod
    def allocate(cls, target: TargetVerifyBatch, top1: Sequence[int], *, runtime) -> "_FixedVerifyBuffers":
        arrays = {
            "token_ids": np.asarray(target.tokens, dtype=np.int32),
            "positions": np.asarray(target.positions, dtype=np.int32),
            "parent_rows": np.asarray(target.parent_rows, dtype=np.int32),
            "draft_depths": np.asarray(target.draft_depths, dtype=np.int32),
            "active_mask": np.asarray(target.active_mask, dtype=np.uint8),
            "top1": np.asarray(top1, dtype=np.int32),
            "accepted_counts": np.empty((len(target.request_ids),), dtype=np.int32),
            "commit_rows": np.empty((len(target.request_ids),), dtype=np.int32),
            "commit_tokens": np.empty((len(target.request_ids),), dtype=np.int32),
            "commit_positions": np.empty((len(target.request_ids),), dtype=np.int32),
            "next_tokens": np.empty((len(target.request_ids),), dtype=np.int32),
            "full_accept": np.empty((len(target.request_ids),), dtype=np.uint8),
            "committed_output_ids": np.empty((len(target.request_ids), target.rows), dtype=np.int32),
            "committed_output_lengths": np.empty((len(target.request_ids),), dtype=np.int32),
            "linear_src": (np.arange(target.rows * 4, dtype=np.uint16).reshape(target.rows, 4) + np.uint16(11)),
            "linear_dst": np.empty((len(target.request_ids), 4), dtype=np.uint16),
            "kv_src": (np.arange(target.rows * 3, dtype=np.uint16).reshape(target.rows, 3) + np.uint16(101)),
            "kv_dst": np.empty((max(1, target.candidate_count), 3), dtype=np.uint16),
            "output_ids": np.empty((len(target.request_ids), target.rows + 1), dtype=np.int32),
            "output_lengths": np.empty((len(target.request_ids),), dtype=np.int32),
            "last_positions": np.empty((len(target.request_ids),), dtype=np.int32),
            "context_lengths": np.empty((len(target.request_ids),), dtype=np.int32),
        }
        buffers: list = []
        device_buffers = {}
        for name, array in arrays.items():
            if name in {"accepted_counts", "commit_rows", "commit_tokens", "commit_positions", "next_tokens", "full_accept", "committed_output_ids", "committed_output_lengths", "linear_dst", "kv_dst", "output_ids", "output_lengths", "last_positions", "context_lengths"}:
                device_buffers[name] = _empty(runtime, buffers, array)
            else:
                device_buffers[name] = _dev(runtime, buffers, array)
        return cls(buffers, arrays, device_buffers)

    @property
    def addresses(self) -> DFlashVerifyGraphAddresses:
        return DFlashVerifyGraphAddresses.from_mapping({name: buf.ptr for name, buf in self.device_buffers.items()})

    def run(self, target: TargetVerifyBatch, summary: TargetAcceptSummary, *, accept_library, commit_library, runtime) -> dict[str, object]:
        self.launch(target, summary, accept_library=accept_library, commit_library=commit_library, runtime=runtime, stream=0)
        runtime.device_synchronize()
        return self.read_outputs(runtime=runtime)

    def launch(self, target: TargetVerifyBatch, summary: TargetAcceptSummary, *, accept_library, commit_library, runtime, stream: int) -> None:
        db = self.device_buffers
        rows = target.rows
        request_count = len(target.request_ids)
        dflash_accept_chain_i32(
            db["token_ids"].ptr,
            db["positions"].ptr,
            db["parent_rows"].ptr,
            db["draft_depths"].ptr,
            db["active_mask"].ptr,
            db["top1"].ptr,
            None,
            db["accepted_counts"].ptr,
            db["commit_rows"].ptr,
            db["commit_tokens"].ptr,
            db["commit_positions"].ptr,
            db["next_tokens"].ptr,
            db["full_accept"].ptr,
            db["committed_output_ids"].ptr,
            db["committed_output_lengths"].ptr,
            rows,
            request_count,
            rows,
            stream=stream,
            library=accept_library,
            runtime=runtime,
        )
        device = Device("hip", 0)
        plan = TargetCommitPlan(
            transaction_id=1,
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
        commit_buffers = TargetStateCommitBuffers.for_plan(
            plan,
            accepted_counts=Tensor.from_handle(db["accepted_counts"].ptr, self.arrays["accepted_counts"].shape, "int32", device),
            commit_rows=Tensor.from_handle(db["commit_rows"].ptr, self.arrays["commit_rows"].shape, "int32", device),
            commit_positions=Tensor.from_handle(db["commit_positions"].ptr, self.arrays["commit_positions"].shape, "int32", device),
            parent_rows=Tensor.from_handle(db["parent_rows"].ptr, self.arrays["parent_rows"].shape, "int32", device),
            linear_state_src=Tensor.from_handle(db["linear_src"].ptr, self.arrays["linear_src"].shape, "bf16", device),
            linear_state_dst=Tensor.from_handle(db["linear_dst"].ptr, self.arrays["linear_dst"].shape, "bf16", device),
            kv_rows_src=Tensor.from_handle(db["kv_src"].ptr, self.arrays["kv_src"].shape, "bf16", device),
            kv_rows_dst=Tensor.from_handle(db["kv_dst"].ptr, self.arrays["kv_dst"].shape, "bf16", device),
            next_tokens_src=Tensor.from_handle(db["next_tokens"].ptr, self.arrays["next_tokens"].shape, "int32", device),
            committed_output_ids_src=Tensor.from_handle(db["committed_output_ids"].ptr, self.arrays["committed_output_ids"].shape, "int32", device),
            committed_output_lengths_src=Tensor.from_handle(db["committed_output_lengths"].ptr, self.arrays["committed_output_lengths"].shape, "int32", device),
            output_ids_dst=Tensor.from_handle(db["output_ids"].ptr, self.arrays["output_ids"].shape, "int32", device),
            output_lengths_dst=Tensor.from_handle(db["output_lengths"].ptr, self.arrays["output_lengths"].shape, "int32", device),
            last_positions_dst=Tensor.from_handle(db["last_positions"].ptr, self.arrays["last_positions"].shape, "int32", device),
            context_lengths_dst=Tensor.from_handle(db["context_lengths"].ptr, self.arrays["context_lengths"].shape, "int32", device),
        )
        dflash_commit_chain_i32(
            commit_buffers,
            target_rows=rows,
            accepted_rows=sum(summary.accepted_counts),
            stream=stream,
            library=commit_library,
            runtime=runtime,
        )

    def read_outputs(self, *, runtime) -> dict[str, object]:
        db = self.device_buffers
        out = {}
        for name in ("accepted_counts", "commit_rows", "commit_tokens", "commit_positions", "next_tokens", "committed_output_lengths", "linear_dst", "kv_dst", "output_lengths", "last_positions", "context_lengths"):
            copy_device_to_host(host_array_ptr(self.arrays[name]), db[name], runtime=runtime)
            out[name] = self.arrays[name].copy()
        fingerprint_arrays = [out[name].copy() for name in sorted(out)]
        payload = _jsonable(out)
        payload["fingerprint_arrays"] = fingerprint_arrays
        return payload

    def free(self, *, runtime) -> None:
        for buf in reversed(self._buffers):
            free(buf, runtime=runtime)


def _jsonable(mapping: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in mapping.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
            out[key] = [array.copy() for array in value]
        else:
            out[key] = value
    return out


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


if __name__ == "__main__":
    raise SystemExit(main())
