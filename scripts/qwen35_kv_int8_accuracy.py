#!/usr/bin/env python3
"""Layer-level dense paged INT8-KV accuracy tool.

This is a synthetic correctness harness for K1.  It builds deterministic paged
K/V rows, checks BF16 and ``int8_per_token_head`` decode against CPU-reference
oracles, and can optionally run the available HIP BF16 path.  The HIP INT8 path
is wired as an explicit requirement check so later kernel tasks can plug in the
wrappers without changing the artifact format.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.correctness import LogitCorrectness, evaluate_logits
from hipengine.kvcache import resolve_kv_policy
from hipengine.kernels.cpu_reference import (
    paged_attn_decode_int8_per_token_head,
    write_paged_kv_int8_per_token_head,
)

DeviceMode = Literal["cpu", "hip"]


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    context_len: int
    block_size: int
    block_table: np.ndarray
    positions: np.ndarray
    query: np.ndarray
    key_rows: np.ndarray
    value_rows: np.ndarray
    scale: float
    logit_projection: np.ndarray

    @property
    def num_q_heads(self) -> int:
        return int(self.query.shape[0])

    @property
    def num_kv_heads(self) -> int:
        return int(self.key_rows.shape[1])

    @property
    def head_dim(self) -> int:
        return int(self.key_rows.shape[2])

    @property
    def blocks(self) -> int:
        return int(np.max(self.block_table)) + 1

    @property
    def crosses_page_boundary(self) -> bool:
        return self.context_len > self.block_size


@dataclass(frozen=True)
class PathCheck:
    name: str
    candidate: str
    passed: bool
    max_abs_attn: float | None = None
    max_rel_attn: float | None = None
    pseudo_logit_gate: LogitCorrectness | None = None
    blocked_reason: str | None = None
    mismatch_reason: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "candidate": self.candidate,
            "passed": self.passed,
            "max_abs_attn": self.max_abs_attn,
            "max_rel_attn": self.max_rel_attn,
            "blocked_reason": self.blocked_reason,
            "mismatch_reason": self.mismatch_reason,
        }
        if self.pseudo_logit_gate is not None:
            payload["pseudo_logit_gate"] = {
                "kl_mean": self.pseudo_logit_gate.kl_mean,
                "kl_max": self.pseudo_logit_gate.kl_max,
                "top1_agreement": self.pseudo_logit_gate.top1_agreement,
                "passed": self.pseudo_logit_gate.passed,
            }
        return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    contexts = _parse_contexts(args.contexts)
    scale_dtype = _parse_scale_dtype(args.scale_dtype)
    cases = [
        _make_case(
            context_len=context,
            block_size=args.block_size,
            num_q_heads=args.num_q_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            seed=args.seed + idx * 1009,
            vocab_size=args.pseudo_vocab_size,
        )
        for idx, context in enumerate(contexts)
    ]
    if args.device == "hip":
        _validate_hip_shape_args(args)
    case_payloads: list[dict[str, Any]] = []
    blocked_reasons: list[str] = []
    correctness_failures: list[str] = []
    for case in cases:
        payload, blockers, failures = _run_case(
            case,
            device=args.device,
            scale_dtype=scale_dtype,
            max_abs_threshold=float(args.max_abs_threshold),
            kl_threshold=float(args.kl_threshold),
            top1_threshold=float(args.top1_threshold),
            compiler_version=_read_compiler_version(args.compiler_version_file),
            require_cached_build=bool(args.require_cached_build),
            require_int8_hip=bool(args.require_int8_hip),
            allow_missing_int8_hip=bool(args.allow_missing_int8_hip),
        )
        case_payloads.append(payload)
        blocked_reasons.extend(blockers)
        correctness_failures.extend(failures)
    passed = not blocked_reasons and not correctness_failures
    status = "accepted" if passed else ("blocked" if blocked_reasons else "rejected_correctness")
    return {
        "schema": 1,
        "status": status,
        "passed": passed,
        "mode": "qwen35_kv_int8_layer_accuracy",
        "device": args.device,
        "command": _command(args),
        "performance_claim": False,
        "thresholds": {
            "max_abs_attn": float(args.max_abs_threshold),
            "kl_mean_max": float(args.kl_threshold),
            "top1_agreement_min": float(args.top1_threshold),
        },
        "kv_policy": resolve_kv_policy(
            "int8_per_token_head",
            block_size=int(args.block_size),
            scale_dtype=args.scale_dtype,
            scale_granularity="per_token_head",
        ).to_json_dict(),
        "shape": {
            "contexts": contexts,
            "block_size": int(args.block_size),
            "num_q_heads": int(args.num_q_heads),
            "num_kv_heads": int(args.num_kv_heads),
            "head_dim": int(args.head_dim),
            "scale_dtype": args.scale_dtype,
            "pseudo_vocab_size": int(args.pseudo_vocab_size),
        },
        "cases": case_payloads,
        "blocked_reasons": blocked_reasons,
        "correctness_failures": correctness_failures,
        "notes": [
            "BF16 and int8_per_token_head are compared against separate CPU-reference oracles.",
            "pseudo_logit_gate projects layer outputs through a deterministic synthetic matrix so KL/top-1 are available without model weights.",
            "device=hip runs the BF16 HIP path and registered INT8 HIP writer/decode wrappers; use --require-int8-hip for K1 promotion gates.",
        ],
    }


def _run_case(
    case: SyntheticCase,
    *,
    device: DeviceMode,
    scale_dtype: np.dtype,
    max_abs_threshold: float,
    kl_threshold: float,
    top1_threshold: float,
    compiler_version: str | None,
    require_cached_build: bool,
    require_int8_hip: bool,
    allow_missing_int8_hip: bool,
) -> tuple[dict[str, Any], list[str], list[str]]:
    bf16_cache = _bf16_cache_from_rows(case)
    bf16_oracle = _paged_attn_decode_bf16_oracle(case, bf16_cache[0], bf16_cache[1])
    int8_cache = write_paged_kv_int8_per_token_head(
        case.key_rows,
        case.value_rows,
        case.positions,
        case.block_table,
        block_size=case.block_size,
        cache_blocks=case.blocks,
        scale_dtype=scale_dtype,
    )
    int8_oracle = paged_attn_decode_int8_per_token_head(
        case.query,
        int8_cache[0],
        int8_cache[1],
        int8_cache[2],
        int8_cache[3],
        np.asarray([case.context_len], dtype=np.int64),
        block_table=case.block_table,
        block_size=case.block_size,
        scale=case.scale,
        output_dtype=np.float32,
    )

    if device == "cpu":
        bf16_check = _compare_path(
            "bf16",
            "cpu_reference",
            bf16_oracle,
            bf16_oracle,
            case.logit_projection,
            max_abs_threshold=max_abs_threshold,
            kl_threshold=kl_threshold,
            top1_threshold=top1_threshold,
        )
        int8_check = _compare_path(
            "int8_per_token_head",
            "cpu_reference",
            int8_oracle,
            int8_oracle,
            case.logit_projection,
            max_abs_threshold=max_abs_threshold,
            kl_threshold=kl_threshold,
            top1_threshold=top1_threshold,
        )
    else:
        bf16_candidate = _run_bf16_hip(case, bf16_cache, compiler_version=compiler_version, require_cached_build=require_cached_build)
        bf16_check = _compare_path(
            "bf16",
            "hip_gfx1100",
            bf16_oracle,
            bf16_candidate,
            case.logit_projection,
            max_abs_threshold=max_abs_threshold,
            kl_threshold=kl_threshold,
            top1_threshold=top1_threshold,
        )
        int8_check = _run_int8_hip_or_blocked(
            case,
            int8_cache,
            int8_oracle,
            max_abs_threshold=max_abs_threshold,
            kl_threshold=kl_threshold,
            top1_threshold=top1_threshold,
            compiler_version=compiler_version,
            require_cached_build=require_cached_build,
            require_int8_hip=require_int8_hip,
            allow_missing_int8_hip=allow_missing_int8_hip,
        )

    quality = _compare_quantized_to_bf16(case, bf16_oracle, int8_oracle, kl_threshold=kl_threshold, top1_threshold=top1_threshold)
    paths = {"bf16": bf16_check.to_json(), "int8_per_token_head": int8_check.to_json()}
    blockers = [f"{case.name}:{path['blocked_reason']}" for path in paths.values() if path.get("blocked_reason")]
    if allow_missing_int8_hip and not require_int8_hip:
        blockers = [reason for reason in blockers if "INT8 HIP wrappers are not registered" not in reason]
    failures = [f"{case.name}:{name}:{path['mismatch_reason']}" for name, path in paths.items() if path.get("mismatch_reason")]
    return (
        {
            "name": case.name,
            "context_len": int(case.context_len),
            "crosses_page_boundary": bool(case.crosses_page_boundary),
            "block_table": case.block_table.astype(int).tolist(),
            "paths": paths,
            "bf16_vs_int8_quantization": quality,
        },
        blockers,
        failures,
    )


def _compare_path(
    name: str,
    candidate_name: str,
    expected: np.ndarray,
    candidate: np.ndarray,
    projection: np.ndarray,
    *,
    max_abs_threshold: float,
    kl_threshold: float,
    top1_threshold: float,
) -> PathCheck:
    if expected.shape != candidate.shape:
        return PathCheck(
            name=name,
            candidate=candidate_name,
            passed=False,
            mismatch_reason=f"shape mismatch: expected {expected.shape}, got {candidate.shape}",
        )
    diff = np.abs(candidate.astype(np.float32) - expected.astype(np.float32))
    max_abs = float(np.max(diff)) if diff.size else 0.0
    denom = np.maximum(np.abs(expected.astype(np.float32)), 1.0e-12)
    max_rel = float(np.max(diff / denom)) if diff.size else 0.0
    logits_expected = _pseudo_logits(expected, projection)
    logits_candidate = _pseudo_logits(candidate, projection)
    gate = evaluate_logits(logits_expected, logits_candidate, kl_threshold=kl_threshold, top1_threshold=top1_threshold)
    passed = bool(max_abs <= max_abs_threshold and gate.passed)
    mismatch = None if passed else (
        f"max_abs {max_abs:.6g} > {max_abs_threshold:.6g} or "
        f"KL/top1 failed (kl_mean={gate.kl_mean:.6g}, top1={gate.top1_agreement:.6g})"
    )
    return PathCheck(
        name=name,
        candidate=candidate_name,
        passed=passed,
        max_abs_attn=max_abs,
        max_rel_attn=max_rel,
        pseudo_logit_gate=gate,
        mismatch_reason=mismatch,
    )


def _compare_quantized_to_bf16(
    case: SyntheticCase,
    bf16: np.ndarray,
    int8: np.ndarray,
    *,
    kl_threshold: float,
    top1_threshold: float,
) -> dict[str, Any]:
    diff = np.abs(int8.astype(np.float32) - bf16.astype(np.float32))
    gate = evaluate_logits(
        _pseudo_logits(bf16, case.logit_projection),
        _pseudo_logits(int8, case.logit_projection),
        kl_threshold=kl_threshold,
        top1_threshold=top1_threshold,
    )
    return {
        "max_abs_attn": float(np.max(diff)) if diff.size else 0.0,
        "max_rel_attn": float(np.max(diff / np.maximum(np.abs(bf16.astype(np.float32)), 1.0e-12))) if diff.size else 0.0,
        "pseudo_logit_gate": {
            "kl_mean": gate.kl_mean,
            "kl_max": gate.kl_max,
            "top1_agreement": gate.top1_agreement,
            "passed": gate.passed,
        },
    }


def _run_bf16_hip(
    case: SyntheticCase,
    bf16_cache: tuple[np.ndarray, np.ndarray],
    *,
    compiler_version: str | None,
    require_cached_build: bool,
) -> np.ndarray:
    from hipengine.core.device import Device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.core.tensor import Tensor
    from hipengine.kernels.hip_gfx1100.attention import (
        build_qwen35_paged_attn_decode,
        build_qwen35_paged_kv_write,
        qwen35_paged_full_attn_decode_context_bf16_spans,
        qwen35_write_paged_kv_mixed_value_bf16_spans,
    )
    from hipengine.kvcache import KVLiveSpans

    runtime = get_hip_runtime()
    buffers = []

    def dev(array: np.ndarray):
        host = np.ascontiguousarray(array)
        buf = malloc(host.nbytes, runtime=runtime)
        buffers.append(buf)
        copy_host_to_device(buf, host_array_ptr(host), runtime=runtime)
        return buf

    def out_dev(array: np.ndarray):
        buf = malloc(array.nbytes, runtime=runtime)
        buffers.append(buf)
        return buf

    device = Device("hip", 0)
    block_table = np.ascontiguousarray(case.block_table.astype(np.int32))
    positions = np.ascontiguousarray(case.positions.astype(np.int64))
    live_counts = np.asarray([case.context_len], dtype=np.int64)
    key_rows = np.ascontiguousarray(case.key_rows.astype(np.float32))
    value_rows_bf16 = np.ascontiguousarray(_float32_to_bf16_bits(case.value_rows))
    key_cache = np.zeros_like(bf16_cache[0])
    value_cache = np.zeros_like(bf16_cache[1])
    query = np.ascontiguousarray(case.query.astype(np.float32))
    out = np.zeros_like(query, dtype=np.float32)
    try:
        block_table_b = dev(block_table)
        positions_b = dev(positions)
        live_counts_b = dev(live_counts)
        key_rows_b = dev(key_rows)
        value_rows_b = dev(value_rows_bf16)
        key_cache_b = dev(key_cache)
        value_cache_b = dev(value_cache)
        query_b = dev(query)
        out_b = out_dev(out)
        kv_lib = build_qwen35_paged_kv_write(
            load=True,
            compiler_version=compiler_version,
            require_cached=require_cached_build,
        )
        attn_lib = build_qwen35_paged_attn_decode(
            load=True,
            compiler_version=compiler_version,
            require_cached=require_cached_build,
        )
        row_key_bytes = case.num_kv_heads * case.head_dim * np.dtype(np.float32).itemsize
        row_value_bytes = case.num_kv_heads * case.head_dim * np.dtype(np.uint16).itemsize
        pos_bytes = np.dtype(np.int64).itemsize
        for row in range(case.context_len):
            spans = KVLiveSpans.paged_uniform(
                block_table=Tensor.from_handle(block_table_b.ptr, block_table.shape, "int32", device),
                live_counts=Tensor.from_handle(positions_b.ptr + row * pos_bytes, (1,), "int64", device),
                max_live_count=case.context_len - 1,
                storage_dtype="bf16",
            )
            qwen35_write_paged_kv_mixed_value_bf16_spans(
                key_rows_b.ptr + row * row_key_bytes,
                value_rows_b.ptr + row * row_value_bytes,
                key_cache_b.ptr,
                value_cache_b.ptr,
                spans,
                case.block_size,
                case.num_kv_heads,
                case.head_dim,
                library=kv_lib,
                runtime=runtime,
            )
        decode_spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(block_table_b.ptr, block_table.shape, "int32", device),
            live_counts=Tensor.from_handle(live_counts_b.ptr, live_counts.shape, "int64", device),
            max_live_count=case.context_len,
            storage_dtype="bf16",
        )
        qwen35_paged_full_attn_decode_context_bf16_spans(
            query_b.ptr,
            key_cache_b.ptr,
            value_cache_b.ptr,
            out_b.ptr,
            decode_spans,
            case.context_len,
            case.block_size,
            case.num_q_heads,
            case.num_kv_heads,
            case.head_dim,
            case.scale,
            library=attn_lib,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_b, runtime=runtime)
        return out
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def _run_int8_hip(
    case: SyntheticCase,
    int8_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    compiler_version: str | None,
    require_cached_build: bool,
) -> np.ndarray:
    from hipengine.core.device import Device
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.core.tensor import Tensor
    from hipengine.dispatch import resolve_paged_attn_decode, resolve_paged_kv_write
    from hipengine.kernels.hip_gfx1100.attention import build_qwen35_paged_attn_decode, build_qwen35_paged_kv_write
    from hipengine.kvcache import KVLiveSpans, KVScaleMetadata
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    buffers = []

    def dev(array: np.ndarray):
        host = np.ascontiguousarray(array)
        buf = malloc(host.nbytes, runtime=runtime)
        buffers.append(buf)
        copy_host_to_device(buf, host_array_ptr(host), runtime=runtime)
        return buf

    def out_dev(array: np.ndarray):
        buf = malloc(array.nbytes, runtime=runtime)
        buffers.append(buf)
        return buf

    device = Device("hip", 0)
    block_table = np.ascontiguousarray(case.block_table.astype(np.int32))
    positions = np.ascontiguousarray(case.positions.astype(np.int64))
    live_counts = np.asarray([case.context_len], dtype=np.int64)
    key_rows = np.ascontiguousarray(case.key_rows.astype(np.float32))
    value_rows = np.ascontiguousarray(case.value_rows.astype(np.float32))
    key_cache = np.zeros_like(int8_cache[0])
    value_cache = np.zeros_like(int8_cache[1])
    k_scale = np.zeros_like(int8_cache[2])
    v_scale = np.zeros_like(int8_cache[3])
    query = np.ascontiguousarray(case.query.astype(np.float32))
    out = np.zeros_like(query, dtype=np.float32)
    num_splits = max(1, (case.context_len + case.block_size - 1) // case.block_size)
    chunk_size = case.block_size
    partial_out = np.zeros((case.num_q_heads, num_splits, case.head_dim), dtype=np.float32)
    partial_m = np.zeros((case.num_q_heads, num_splits), dtype=np.float32)
    partial_l = np.zeros((case.num_q_heads, num_splits), dtype=np.float32)
    scale_dtype = "fp16" if k_scale.dtype == np.float16 else "fp32"
    try:
        block_table_b = dev(block_table)
        positions_b = dev(positions)
        live_counts_b = dev(live_counts)
        key_rows_b = dev(key_rows)
        value_rows_b = dev(value_rows)
        key_cache_b = dev(key_cache)
        value_cache_b = dev(value_cache)
        k_scale_b = dev(k_scale)
        v_scale_b = dev(v_scale)
        query_b = dev(query)
        out_b = out_dev(out)
        partial_out_b = dev(partial_out)
        partial_m_b = dev(partial_m)
        partial_l_b = dev(partial_l)
        kv_lib = build_qwen35_paged_kv_write(load=True, compiler_version=compiler_version, require_cached=require_cached_build)
        attn_lib = build_qwen35_paged_attn_decode(load=True, compiler_version=compiler_version, require_cached=require_cached_build)
        scale_metadata = KVScaleMetadata(
            k_scale=Tensor.from_handle(k_scale_b.ptr, k_scale.shape, scale_dtype, device),
            v_scale=Tensor.from_handle(v_scale_b.ptr, v_scale.shape, scale_dtype, device),
            scale_dtype=scale_dtype,
        )
        dispatch_spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(block_table_b.ptr, block_table.shape, "int32", device),
            live_counts=Tensor.from_handle(live_counts_b.ptr, live_counts.shape, "int64", device),
            max_live_count=case.context_len,
            storage_dtype="int8_per_token_head",
            scale_metadata=scale_metadata,
        )
        write_fn = resolve_paged_kv_write(
            backend="hip_gfx1100",
            spans=dispatch_spans,
            kind="decode",
            source_dtype="fp32",
        )
        decode_fn = resolve_paged_attn_decode(
            backend="hip_gfx1100",
            spans=dispatch_spans,
            kind="gqa_splitk",
        )
        row_bytes = case.num_kv_heads * case.head_dim * np.dtype(np.float32).itemsize
        pos_bytes = np.dtype(np.int64).itemsize
        for row in range(case.context_len):
            spans = KVLiveSpans.paged_uniform(
                block_table=Tensor.from_handle(block_table_b.ptr, block_table.shape, "int32", device),
                live_counts=Tensor.from_handle(positions_b.ptr + row * pos_bytes, (1,), "int64", device),
                max_live_count=case.context_len - 1,
                storage_dtype="int8_per_token_head",
                scale_metadata=scale_metadata,
            )
            write_fn(
                key_rows_b.ptr + row * row_bytes,
                value_rows_b.ptr + row * row_bytes,
                key_cache_b.ptr,
                value_cache_b.ptr,
                k_scale_b.ptr,
                v_scale_b.ptr,
                spans,
                case.block_size,
                case.num_kv_heads,
                case.head_dim,
                library=kv_lib,
                runtime=runtime,
            )
        decode_spans = KVLiveSpans.paged_uniform(
            block_table=Tensor.from_handle(block_table_b.ptr, block_table.shape, "int32", device),
            live_counts=Tensor.from_handle(live_counts_b.ptr, live_counts.shape, "int64", device),
            max_live_count=case.context_len,
            storage_dtype="int8_per_token_head",
            scale_metadata=scale_metadata,
        )
        decode_fn(
            query_b.ptr,
            key_cache_b.ptr,
            value_cache_b.ptr,
            k_scale_b.ptr,
            v_scale_b.ptr,
            out_b.ptr,
            partial_out_b.ptr,
            partial_m_b.ptr,
            partial_l_b.ptr,
            decode_spans,
            chunk_size,
            num_splits,
            case.block_size,
            case.num_q_heads,
            case.num_kv_heads,
            case.head_dim,
            case.scale,
            library=attn_lib,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_b, runtime=runtime)
        return out
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def _run_int8_hip_or_blocked(
    case: SyntheticCase,
    int8_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    int8_oracle: np.ndarray,
    *,
    max_abs_threshold: float,
    kl_threshold: float,
    top1_threshold: float,
    compiler_version: str | None,
    require_cached_build: bool,
    require_int8_hip: bool,
    allow_missing_int8_hip: bool,
) -> PathCheck:
    try:
        from hipengine.kernels.registry import KernelKey, registered_keys
    except Exception as exc:  # pragma: no cover - defensive import guard
        return PathCheck("int8_per_token_head", "hip_gfx1100", False, blocked_reason=f"cannot inspect registry: {exc}")
    expected_keys = {
        KernelKey("hip_gfx1100", "paged_kv_write", "int8_per_token_head", "per_token_head_spans"),
        KernelKey("hip_gfx1100", "paged_kv_write", "int8_per_token_head", "per_token_head_prompt_spans"),
        KernelKey("hip_gfx1100", "paged_kv_write", "int8_per_token_head", "per_token_head_batch_spans"),
        KernelKey("hip_gfx1100", "paged_attn_decode", "int8_per_token_head", "gqa_splitk_spans"),
        KernelKey("hip_gfx1100", "paged_attn_decode", "int8_per_token_head", "gqa_splitk_gate_bf16_spans"),
        KernelKey("hip_gfx1100", "paged_attn_decode", "int8_per_token_head", "gqa_splitk_gate_fp16_spans"),
        KernelKey("hip_gfx1100", "paged_attn_decode", "int8_per_token_head", "per_token_head_gqa_splitk_spans"),
        KernelKey("hip_gfx1100", "paged_attn_decode", "int8_per_token_head", "per_token_head_gqa_splitk_gate_bf16_spans"),
        KernelKey("hip_gfx1100", "paged_attn_decode", "int8_per_token_head", "per_token_head_gqa_splitk_gate_fp16_spans"),
    }
    missing = sorted(key.display() for key in expected_keys.difference(set(registered_keys())))
    if missing:
        reason = "INT8 HIP wrappers are not registered yet; missing exact keys: " + "; ".join(missing)
        if require_int8_hip or not allow_missing_int8_hip:
            return PathCheck("int8_per_token_head", "hip_gfx1100", False, blocked_reason=reason)
        return PathCheck("int8_per_token_head", "hip_gfx1100", True, blocked_reason=reason)
    try:
        candidate = _run_int8_hip(
            case,
            int8_cache,
            compiler_version=compiler_version,
            require_cached_build=require_cached_build,
        )
    except Exception as exc:  # pragma: no cover - requires future INT8 HIP wrappers/GPU
        return PathCheck("int8_per_token_head", "hip_gfx1100", False, blocked_reason=f"failed to execute INT8 HIP wrappers: {exc}")
    return _compare_path(
        "int8_per_token_head",
        "hip_gfx1100",
        int8_oracle,
        candidate,
        case.logit_projection,
        max_abs_threshold=max_abs_threshold,
        kl_threshold=kl_threshold,
        top1_threshold=top1_threshold,
    )


def _make_case(
    *,
    context_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seed: int,
    vocab_size: int,
) -> SyntheticCase:
    if context_len <= 0:
        raise ValueError("contexts must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("head counts must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    rng = np.random.default_rng(seed)
    blocks = (context_len + block_size - 1) // block_size
    block_table = np.arange(blocks, dtype=np.int32)
    if blocks > 1:
        block_table = np.roll(block_table[::-1], 1).astype(np.int32)
    positions = np.arange(context_len, dtype=np.int64)
    query = rng.normal(0.0, 0.25, size=(num_q_heads, head_dim)).astype(np.float32)
    key_rows = rng.normal(0.0, 0.25, size=(context_len, num_kv_heads, head_dim)).astype(np.float32)
    value_rows = rng.normal(0.0, 0.25, size=(context_len, num_kv_heads, head_dim)).astype(np.float32)
    if context_len >= 2:
        # A deterministic all-zero edge row exercises INT8 zero-scale handling.
        key_rows[1, 0] = 0.0
        value_rows[1, 0] = 0.0
    projection = rng.normal(0.0, 0.05, size=(num_q_heads * head_dim, vocab_size)).astype(np.float32)
    return SyntheticCase(
        name=f"ctx{context_len}",
        context_len=int(context_len),
        block_size=int(block_size),
        block_table=block_table,
        positions=positions,
        query=query,
        key_rows=key_rows,
        value_rows=value_rows,
        scale=float(head_dim ** -0.5),
        logit_projection=projection,
    )


def _bf16_cache_from_rows(case: SyntheticCase) -> tuple[np.ndarray, np.ndarray]:
    key_cache = np.zeros((case.blocks, case.block_size, case.num_kv_heads, case.head_dim), dtype=np.uint16)
    value_cache = np.zeros_like(key_cache)
    key_bits = _float32_to_bf16_bits(case.key_rows)
    value_bits = _float32_to_bf16_bits(case.value_rows)
    for row, position in enumerate(case.positions):
        logical_block = int(position) // case.block_size
        block_offset = int(position) % case.block_size
        physical_block = int(case.block_table[logical_block])
        key_cache[physical_block, block_offset] = key_bits[row]
        value_cache[physical_block, block_offset] = value_bits[row]
    return key_cache, value_cache


def _paged_attn_decode_bf16_oracle(case: SyntheticCase, key_cache: np.ndarray, value_cache: np.ndarray) -> np.ndarray:
    key = _bf16_bits_to_float32(key_cache)
    value = _bf16_bits_to_float32(value_cache)
    out = np.empty_like(case.query, dtype=np.float32)
    kv_group = case.num_q_heads // case.num_kv_heads
    for q_head in range(case.num_q_heads):
        kv_head = q_head // kv_group
        keys = np.stack([
            _paged_row(key, position, kv_head, block_table=case.block_table, block_size=case.block_size)
            for position in range(case.context_len)
        ], axis=0)
        values = np.stack([
            _paged_row(value, position, kv_head, block_table=case.block_table, block_size=case.block_size)
            for position in range(case.context_len)
        ], axis=0)
        weights = _softmax(np.matmul(keys, case.query[q_head]) * case.scale)
        out[q_head] = np.matmul(weights, values)
    return out


def _paged_row(cache: np.ndarray, position: int, kv_head: int, *, block_table: np.ndarray, block_size: int) -> np.ndarray:
    logical_block = int(position) // block_size
    block_offset = int(position) % block_size
    physical_block = int(block_table[logical_block])
    return cache[physical_block, block_offset, kv_head]


def _pseudo_logits(out: np.ndarray, projection: np.ndarray) -> np.ndarray:
    return out.astype(np.float32).reshape(1, -1) @ projection.astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x.astype(np.float64) - float(np.max(x))
    exp = np.exp(shifted)
    return (exp / np.sum(exp)).astype(np.float32)


def _float32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    bits = arr.view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    arr = np.asarray(bits, dtype=np.uint16)
    return (arr.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _parse_contexts(value: str) -> list[int]:
    contexts = [int(item) for item in value.split(",") if item.strip()]
    if not contexts or any(item <= 0 for item in contexts):
        raise ValueError("--contexts must contain positive integers")
    return contexts


def _parse_scale_dtype(value: str) -> np.dtype:
    if value == "fp16":
        return np.dtype(np.float16)
    if value == "fp32":
        return np.dtype(np.float32)
    raise ValueError("--scale-dtype must be fp16 or fp32")


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"compiler version file {path} is empty")
    return text


def _validate_hip_shape_args(args: argparse.Namespace) -> None:
    if int(args.block_size) != 256:
        raise ValueError("device=hip uses current Qwen paged kernels and requires --block-size 256")
    if int(args.head_dim) > 256:
        raise ValueError("device=hip BF16 paged decode requires --head-dim <= 256")


def _command(args: argparse.Namespace) -> str:
    parts = [
        "python3 scripts/qwen35_kv_int8_accuracy.py",
        f"--device {args.device}",
        f"--contexts {args.contexts}",
        f"--block-size {args.block_size}",
        f"--num-q-heads {args.num_q_heads}",
        f"--num-kv-heads {args.num_kv_heads}",
        f"--head-dim {args.head_dim}",
        f"--scale-dtype {args.scale_dtype}",
        f"--seed {args.seed}",
    ]
    if args.compiler_version_file is not None:
        parts.append(f"--compiler-version-file {args.compiler_version_file}")
    if args.require_cached_build:
        parts.append("--require-cached-build")
    if args.require_int8_hip:
        parts.append("--require-int8-hip")
    if args.allow_missing_int8_hip:
        parts.append("--allow-missing-int8-hip")
    if args.json is not None:
        parts.append(f"--json {args.json}")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "hip"), default="cpu")
    parser.add_argument("--contexts", default="64,520", help="Comma-separated context lengths; default covers short and page-boundary long cases.")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-q-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--scale-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--pseudo-vocab-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-abs-threshold", type=float, default=5.0e-3)
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--top1-threshold", type=float, default=0.90)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--require-int8-hip", action="store_true", help="For device=hip, fail if INT8 HIP wrappers are not available and executed.")
    parser.add_argument("--allow-missing-int8-hip", action="store_true", help="For early K1 tooling runs, record missing INT8 HIP wrappers without failing the whole artifact.")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.pseudo_vocab_size <= 0:
        raise ValueError("--pseudo-vocab-size must be positive")
    payload = run(args)
    if args.json is not None:
        payload["artifact_path"] = str(args.json)
        payload["source_artifact_path"] = str(args.json)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    if payload["status"] == "blocked":
        return 2
    if not payload["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
