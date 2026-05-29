#!/usr/bin/env python3
"""Qwen3.5/PARO native c>N hidden-state bisection diagnostic.

This diagnostic compares compact native c=2 decode hidden tensors against
independent c=1 resident sessions at configurable layer limits.  It is a
correctness-only tool: it emits JSON with token and hidden mismatches, and it
never marks a throughput result accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr
from hipengine.core.tensor import Tensor
from hipengine.generation import ResidentBatchScheduler
from hipengine.kvcache import FixedPagedKVPolicy
from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner, Qwen35ParoResidentSession
from scripts.qwen35_batch_retained_bench import DEFAULT_FIXTURE, DEFAULT_MODEL, _compiler_version, _load_prompt_slices


DECODE_FULL_ATTENTION_TRACE_STAGES = (
    "input",
    "attn_input_pre_qkv",
    "attn_input_after_rotate",
    "attn_input_after_project",
    "q_proj_key_after_project",
    "value_after_project",
    "query_raw_after_split",
    "key_raw_after_cast",
    "gate_after_split",
    "query_after_prepare",
    "key_after_prepare",
    "attn_input_after_prepare",
    "attn_input",
    "gate",
    "query",
    "attn_context",
    "gated_attn",
    "o_proj",
    "residual",
    "mlp_input",
    "output",
)
DECODE_FULL_CONTEXT_ORACLE_ATOL = 3.0e-5


@dataclass(frozen=True)
class HiddenRun:
    seed_tokens: list[int]
    generated_tokens: list[list[int]]
    hidden_bits_by_step: list[np.ndarray]
    prefill_hidden_bits: np.ndarray | None = None
    prefill_execution: dict[str, Any] | None = None
    prefill_linear_states: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    prefill_linear_inputs: dict[int, list[np.ndarray]] = field(default_factory=dict)
    prefill_full_kv_prefix_hashes: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    decode_linear_inputs_by_step: list[dict[int, np.ndarray]] = field(default_factory=list)
    decode_full_attention_by_step: list[dict[int, dict[str, np.ndarray]]] = field(default_factory=list)
    decode_full_context_oracles_by_step: list[dict[int, dict[str, np.ndarray]]] = field(default_factory=list)
    decode_full_kv_samples_by_step: list[dict[int, dict[str, np.ndarray | tuple[str, ...]]]] = field(default_factory=list)
    decode_linear_states_by_step: list[dict[int, dict[str, np.ndarray]]] = field(default_factory=list)
    decode_execution_by_step: list[dict[str, Any] | None] = field(default_factory=list)


def _command(argv: Sequence[str] | None) -> str:
    parts = ["python3", "scripts/qwen35_batch_hidden_bisect.py"]
    parts.extend(sys.argv[1:] if argv is None else list(argv))
    return " ".join(shlex.quote(part) for part in parts)


def _parse_layer_limits(value: str | None, *, max_layers: int) -> list[int]:
    if max_layers <= 0:
        raise ValueError("max_layers must be positive")
    if value is None or not value.strip() or value.strip().lower() == "all":
        return list(range(1, max_layers + 1))
    limits: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError("layer limit ranges must be ascending")
            limits.extend(range(start, end + 1))
        else:
            limits.append(int(part))
    if not limits:
        raise ValueError("at least one layer limit is required")
    deduped = sorted(set(limits))
    if deduped[0] <= 0 or deduped[-1] > max_layers:
        raise ValueError(f"layer limits must be within [1, {max_layers}]")
    return deduped


def _parse_focus_hidden_flat_indices(values: Sequence[str] | None) -> list[int]:
    if not values:
        return []
    indices: list[int] = []
    seen: set[int] = set()
    for value in values:
        for raw_part in str(value).split(","):
            part = raw_part.strip()
            if not part:
                continue
            index = int(part)
            if index < 0:
                raise ValueError("focus hidden flat indices must be non-negative")
            if index in seen:
                continue
            indices.append(index)
            seen.add(index)
    return indices


_MAX_HIDDEN_DIFF_EXAMPLES = 8


def _fp16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return np.asarray(bits, dtype=np.uint16).view(np.float16).astype(np.float32)


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint32) << np.uint32(16)).view(np.float32)


def _hidden_diff_example_at_flat_index(
    batch_bits: np.ndarray,
    c1_bits: np.ndarray,
    batch_f32: np.ndarray,
    c1_f32: np.ndarray,
    signed_diff: np.ndarray,
    diff: np.ndarray,
    *,
    flat_index: int,
) -> dict[str, Any]:
    flat_batch_bits = np.asarray(batch_bits, dtype=np.uint16).reshape(-1)
    flat_c1_bits = np.asarray(c1_bits, dtype=np.uint16).reshape(-1)
    return {
        "flat_index": int(flat_index),
        "index": [int(index) for index in np.unravel_index(int(flat_index), diff.shape)],
        "abs_diff": float(diff.reshape(-1)[int(flat_index)]),
        "signed_diff": float(signed_diff.reshape(-1)[int(flat_index)]),
        "batch_value": float(batch_f32.reshape(-1)[int(flat_index)]),
        "c1_value": float(c1_f32.reshape(-1)[int(flat_index)]),
        "batch_bits": int(flat_batch_bits[int(flat_index)]),
        "c1_bits": int(flat_c1_bits[int(flat_index)]),
    }


def _top_abs_diff_examples(
    batch_bits: np.ndarray,
    c1_bits: np.ndarray,
    batch_f32: np.ndarray,
    c1_f32: np.ndarray,
    signed_diff: np.ndarray,
    diff: np.ndarray,
    *,
    limit: int = _MAX_HIDDEN_DIFF_EXAMPLES,
) -> list[dict[str, Any]]:
    if diff.size == 0 or limit <= 0:
        return []
    flat_diff = diff.reshape(-1)
    nonzero_indices = [int(index) for index in np.flatnonzero(flat_diff > 0.0)]
    selected = sorted(nonzero_indices, key=lambda index: (-float(flat_diff[index]), index))[:limit]
    return [
        _hidden_diff_example_at_flat_index(
            batch_bits,
            c1_bits,
            batch_f32,
            c1_f32,
            signed_diff,
            diff,
            flat_index=flat_index,
        )
        for flat_index in selected
    ]


def _selected_abs_diff_examples(
    batch_bits: np.ndarray,
    c1_bits: np.ndarray,
    batch_f32: np.ndarray,
    c1_f32: np.ndarray,
    signed_diff: np.ndarray,
    diff: np.ndarray,
    *,
    flat_indices: Sequence[int],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw_index in flat_indices:
        flat_index = int(raw_index)
        if flat_index in seen:
            continue
        if flat_index < 0 or flat_index >= diff.size:
            raise ValueError(f"selected hidden flat index {flat_index} is outside [0, {diff.size})")
        seen.add(flat_index)
        examples.append(
            _hidden_diff_example_at_flat_index(
                batch_bits,
                c1_bits,
                batch_f32,
                c1_f32,
                signed_diff,
                diff,
                flat_index=flat_index,
            )
        )
    return examples


def hidden_comparison(
    batch_bits: np.ndarray,
    c1_bits: np.ndarray,
    *,
    atol: float,
    selected_flat_indices: Sequence[int] = (),
) -> dict[str, Any]:
    if batch_bits.shape != c1_bits.shape:
        raise ValueError(f"hidden shapes differ: batch={batch_bits.shape!r} c1={c1_bits.shape!r}")
    batch_f32 = _fp16_bits_to_f32(batch_bits)
    c1_f32 = _fp16_bits_to_f32(c1_bits)
    signed_diff = batch_f32 - c1_f32
    diff = np.abs(signed_diff)
    bit_mismatch = int(np.count_nonzero(np.asarray(batch_bits, dtype=np.uint16) != np.asarray(c1_bits, dtype=np.uint16)))
    max_abs = float(diff.max(initial=0.0))
    if diff.size:
        max_abs_flat_index = int(np.argmax(diff))
        max_abs_index = [int(index) for index in np.unravel_index(max_abs_flat_index, diff.shape)]
        batch_value = float(batch_f32.flat[max_abs_flat_index])
        c1_value = float(c1_f32.flat[max_abs_flat_index])
        max_signed_diff = float(signed_diff.flat[max_abs_flat_index])
    else:
        max_abs_flat_index = None
        max_abs_index = []
        batch_value = 0.0
        c1_value = 0.0
        max_signed_diff = 0.0
    result = {
        "shape": list(batch_bits.shape),
        "max_abs": max_abs,
        "max_abs_flat_index": max_abs_flat_index,
        "max_abs_index": max_abs_index,
        "batch_value_at_max_abs": batch_value,
        "c1_value_at_max_abs": c1_value,
        "signed_diff_at_max_abs": max_signed_diff,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "elements_over_atol": int(np.count_nonzero(diff > float(atol))),
        "bit_mismatch": bit_mismatch,
        "top_abs_diffs": _top_abs_diff_examples(batch_bits, c1_bits, batch_f32, c1_f32, signed_diff, diff),
        "passed": bool(max_abs <= float(atol)),
    }
    if selected_flat_indices:
        result["selected_abs_diffs"] = _selected_abs_diff_examples(
            batch_bits,
            c1_bits,
            batch_f32,
            c1_f32,
            signed_diff,
            diff,
            flat_indices=selected_flat_indices,
        )
    return result


def _trace_array_to_f32(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint16:
        return _fp16_bits_to_f32(array)
    return np.asarray(array, dtype=np.float32)


def _inferred_rmsnorm_oracle_comparison(
    *,
    batch_residual: np.ndarray,
    batch_mlp_input: np.ndarray,
    c1_residual: np.ndarray,
    c1_mlp_input: np.ndarray,
    atol: float,
    eps: float = 1.0e-6,
) -> dict[str, Any]:
    batch_residual = np.asarray(batch_residual, dtype=np.float32)
    batch_mlp_input = np.asarray(batch_mlp_input, dtype=np.float32)
    c1_residual = np.asarray(c1_residual, dtype=np.float32)
    c1_mlp_input = np.asarray(c1_mlp_input, dtype=np.float32)
    if batch_residual.shape != batch_mlp_input.shape or c1_residual.shape != c1_mlp_input.shape:
        raise ValueError(
            "RMSNorm oracle residual/mlp_input shapes differ: "
            f"batch_residual={batch_residual.shape} batch_mlp_input={batch_mlp_input.shape} "
            f"c1_residual={c1_residual.shape} c1_mlp_input={c1_mlp_input.shape}"
        )
    if batch_residual.shape != c1_residual.shape:
        raise ValueError(
            f"RMSNorm oracle c>N/c1 shapes differ: batch={batch_residual.shape} c1={c1_residual.shape}"
        )
    batch_rms = float(np.sqrt(float(np.mean(batch_residual * batch_residual)) + float(eps)))
    c1_rms = float(np.sqrt(float(np.mean(c1_residual * c1_residual)) + float(eps)))
    valid = np.abs(c1_residual) > 1.0e-7
    expected = batch_mlp_input.copy()
    if np.any(valid):
        inferred_weight = np.zeros_like(c1_residual, dtype=np.float32)
        inferred_weight[valid] = c1_mlp_input[valid] * np.float32(c1_rms) / c1_residual[valid]
        expected[valid] = batch_residual[valid] * inferred_weight[valid] / np.float32(batch_rms)
    # The traced mlp_input is FP16, so round the inferred expectation to the
    # same dtype before applying the hidden-state tolerance.
    expected = expected.astype(np.float16).astype(np.float32)
    comparison = numeric_comparison(batch_mlp_input, expected, atol=atol)
    comparison["batch_residual_rms"] = batch_rms
    comparison["c1_residual_rms"] = c1_rms
    comparison["inferred_weight_valid_elements"] = int(np.count_nonzero(valid))
    comparison["ignored_c1_zero_residual_elements"] = int(valid.size - np.count_nonzero(valid))
    return comparison


def _numeric_abs_diff_at_flat_index(batch: np.ndarray, c1: np.ndarray, flat_index: int) -> dict[str, Any]:
    if batch.shape != c1.shape:
        raise ValueError(f"numeric shapes differ: batch={batch.shape!r} c1={c1.shape!r}")
    batch_f32 = np.asarray(batch, dtype=np.float32)
    c1_f32 = np.asarray(c1, dtype=np.float32)
    if flat_index < 0 or flat_index >= int(batch_f32.size):
        raise IndexError(f"flat index {flat_index} out of bounds for shape {batch_f32.shape!r}")
    signed_diff = batch_f32 - c1_f32
    diff = np.abs(signed_diff)
    return {
        "flat_index": int(flat_index),
        "index": [int(index) for index in np.unravel_index(int(flat_index), diff.shape)],
        "abs_diff": float(diff.reshape(-1)[int(flat_index)]),
        "signed_diff": float(signed_diff.reshape(-1)[int(flat_index)]),
        "batch_value": float(batch_f32.reshape(-1)[int(flat_index)]),
        "c1_value": float(c1_f32.reshape(-1)[int(flat_index)]),
    }


def _numeric_top_abs_diff_examples(
    batch: np.ndarray,
    c1: np.ndarray,
    signed_diff: np.ndarray,
    diff: np.ndarray,
    *,
    limit: int = _MAX_HIDDEN_DIFF_EXAMPLES,
) -> list[dict[str, Any]]:
    if diff.size == 0 or limit <= 0:
        return []
    flat_diff = diff.reshape(-1)
    nonzero_indices = [int(index) for index in np.flatnonzero(flat_diff > 0.0)]
    selected = sorted(nonzero_indices, key=lambda index: (-float(flat_diff[index]), index))[:limit]
    batch_flat = batch.reshape(-1)
    c1_flat = c1.reshape(-1)
    signed_flat = signed_diff.reshape(-1)
    examples: list[dict[str, Any]] = []
    for flat_index in selected:
        examples.append(
            {
                "flat_index": int(flat_index),
                "index": [int(index) for index in np.unravel_index(flat_index, diff.shape)],
                "abs_diff": float(flat_diff[flat_index]),
                "signed_diff": float(signed_flat[flat_index]),
                "batch_value": float(batch_flat[flat_index]),
                "c1_value": float(c1_flat[flat_index]),
            }
        )
    return examples


def bf16_bits_comparison(batch_bits: np.ndarray, c1_bits: np.ndarray, *, atol: float) -> dict[str, Any]:
    comparison = numeric_comparison(_bf16_bits_to_f32(batch_bits), _bf16_bits_to_f32(c1_bits), atol=atol)
    comparison["bit_mismatch"] = int(np.count_nonzero(np.asarray(batch_bits, dtype=np.uint16) != np.asarray(c1_bits, dtype=np.uint16)))
    return comparison


def numeric_comparison(batch: np.ndarray, c1: np.ndarray, *, atol: float) -> dict[str, Any]:
    if batch.shape != c1.shape:
        raise ValueError(f"numeric shapes differ: batch={batch.shape!r} c1={c1.shape!r}")
    batch_f32 = np.asarray(batch, dtype=np.float32)
    c1_f32 = np.asarray(c1, dtype=np.float32)
    signed_diff = batch_f32 - c1_f32
    diff = np.abs(signed_diff)
    max_abs = float(diff.max(initial=0.0))
    if diff.size:
        max_abs_flat_index = int(np.argmax(diff))
        max_abs_index = [int(index) for index in np.unravel_index(max_abs_flat_index, diff.shape)]
        batch_value = float(batch_f32.flat[max_abs_flat_index])
        c1_value = float(c1_f32.flat[max_abs_flat_index])
        max_signed_diff = float(signed_diff.flat[max_abs_flat_index])
    else:
        max_abs_flat_index = None
        max_abs_index = []
        batch_value = 0.0
        c1_value = 0.0
        max_signed_diff = 0.0
    return {
        "shape": list(batch.shape),
        "max_abs": max_abs,
        "max_abs_flat_index": max_abs_flat_index,
        "max_abs_index": max_abs_index,
        "batch_value_at_max_abs": batch_value,
        "c1_value_at_max_abs": c1_value,
        "signed_diff_at_max_abs": max_signed_diff,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "elements_over_atol": int(np.count_nonzero(diff > float(atol))),
        "top_abs_diffs": _numeric_top_abs_diff_examples(batch_f32, c1_f32, signed_diff, diff),
        "passed": bool(max_abs <= float(atol)),
    }


def _numeric_row_summaries(batch: np.ndarray, c1: np.ndarray, *, atol: float) -> list[dict[str, Any]]:
    if batch.shape != c1.shape:
        raise ValueError(f"numeric shapes differ: batch={batch.shape!r} c1={c1.shape!r}")
    if batch.ndim == 0:
        return []
    rows: list[dict[str, Any]] = []
    for row in range(int(batch.shape[0])):
        comparison = numeric_comparison(batch[row], c1[row], atol=atol)
        rows.append(
            {
                "row": int(row),
                "passed": bool(comparison["passed"]),
                "max_abs": float(comparison["max_abs"]),
                "max_abs_flat_index": comparison["max_abs_flat_index"],
                "max_abs_index": comparison["max_abs_index"],
                "batch_value_at_max_abs": float(comparison["batch_value_at_max_abs"]),
                "c1_value_at_max_abs": float(comparison["c1_value_at_max_abs"]),
                "signed_diff_at_max_abs": float(comparison["signed_diff_at_max_abs"]),
                "elements_over_atol": int(comparison["elements_over_atol"]),
                "top_abs_diffs": comparison["top_abs_diffs"][:3],
            }
        )
    return rows


def _hidden_mismatch_record(
    summary: dict[str, Any],
    step: dict[str, Any],
    row: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "layer_limit": int(summary["layer_limit"]),
        "decode_step": int(step["decode_step"]),
        "generated_index": int(step["generated_index"]),
        "row": int(row["row"]),
        "max_abs": float(comparison.get("max_abs", 0.0)),
        "max_abs_flat_index": comparison.get("max_abs_flat_index"),
        "max_abs_index": comparison.get("max_abs_index", []),
        "batch_value_at_max_abs": float(comparison.get("batch_value_at_max_abs", 0.0)),
        "c1_value_at_max_abs": float(comparison.get("c1_value_at_max_abs", 0.0)),
        "signed_diff_at_max_abs": float(comparison.get("signed_diff_at_max_abs", 0.0)),
        "elements_over_atol": int(comparison.get("elements_over_atol", 0)),
        "bit_mismatch": int(comparison.get("bit_mismatch", 0)),
        "passed_under_atol": bool(comparison.get("passed", False)),
        "top_abs_diffs": comparison.get("top_abs_diffs", []),
    }
    if "hidden_atol" in summary:
        result["hidden_atol"] = float(summary["hidden_atol"])
    if "last_layer_index" in summary:
        result["last_layer_index"] = int(summary["last_layer_index"])
    if "last_layer_type" in summary:
        result["last_layer_type"] = str(summary["last_layer_type"])
    decode_execution = step.get("batch_decode_execution")
    if isinstance(decode_execution, dict):
        result["batch_decode_execution"] = decode_execution
    return result


def _first_hidden_mismatch(layer_summaries: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for summary in layer_summaries:
        for step in summary.get("steps", []):
            for row in step.get("rows", []):
                comparison = row.get("hidden_comparison", {})
                if not comparison.get("passed", False):
                    return _hidden_mismatch_record(summary, step, row, comparison)
    return None


def _first_hidden_bit_drift(layer_summaries: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for summary in layer_summaries:
        for step in summary.get("steps", []):
            for row in step.get("rows", []):
                comparison = row.get("hidden_comparison", {})
                if int(comparison.get("bit_mismatch", 0)) > 0:
                    return _hidden_mismatch_record(summary, step, row, comparison)
    return None


def _first_token_mismatch(layer_summaries: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for summary in layer_summaries:
        for row in summary.get("token_mismatches", []):
            return {"layer_limit": int(summary["layer_limit"]), **row}
    return None


def _hidden_failure_rows(summary: dict[str, Any]) -> list[int]:
    rows: list[int] = []
    seen: set[int] = set()
    for step in summary.get("steps", []):
        for row in step.get("rows", []):
            comparison = row.get("hidden_comparison", {})
            if comparison.get("passed", False):
                continue
            row_index = int(row["row"])
            if row_index not in seen:
                rows.append(row_index)
                seen.add(row_index)
    return rows


def _hidden_bit_drift_rows(summary: dict[str, Any]) -> list[int]:
    rows: list[int] = []
    seen: set[int] = set()
    for step in summary.get("steps", []):
        for row in step.get("rows", []):
            comparison = row.get("hidden_comparison", {})
            if int(comparison.get("bit_mismatch", 0)) <= 0:
                continue
            row_index = int(row["row"])
            if row_index not in seen:
                rows.append(row_index)
                seen.add(row_index)
    return rows


def _token_failure_rows(summary: dict[str, Any]) -> list[int]:
    rows: list[int] = []
    seen: set[int] = set()
    for mismatch in summary.get("token_mismatches", []):
        row_index = int(mismatch["row"])
        if row_index not in seen:
            rows.append(row_index)
            seen.add(row_index)
    return rows


def _unique_rows_from_layer_summaries(layer_summaries: Sequence[dict[str, Any]], key: str) -> list[int]:
    rows: list[int] = []
    seen: set[int] = set()
    for summary in layer_summaries:
        for raw_row in summary.get(key, []):
            row_index = int(raw_row)
            if row_index in seen:
                continue
            rows.append(row_index)
            seen.add(row_index)
    return rows


def _row_failure_summary(layer_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    hidden_failure_rows = _unique_rows_from_layer_summaries(layer_summaries, "hidden_failure_rows")
    strict_hidden_bit_drift_rows = _unique_rows_from_layer_summaries(layer_summaries, "strict_hidden_bit_drift_rows")
    token_failure_rows = _unique_rows_from_layer_summaries(layer_summaries, "token_failure_rows")
    return {
        "hidden_failure_rows": hidden_failure_rows,
        "hidden_failure_row_count": len(hidden_failure_rows),
        "strict_hidden_bit_drift_rows": strict_hidden_bit_drift_rows,
        "strict_hidden_bit_drift_row_count": len(strict_hidden_bit_drift_rows),
        "token_failure_rows": token_failure_rows,
        "token_failure_row_count": len(token_failure_rows),
        "layer_limits": [
            {
                "layer_limit": int(summary["layer_limit"]),
                "failure_modes": list(summary.get("failure_modes", [])),
                "hidden_failure_rows": [int(row) for row in summary.get("hidden_failure_rows", [])],
                "hidden_failure_row_count": int(summary.get("hidden_failure_row_count", 0)),
                "strict_hidden_bit_drift_rows": [int(row) for row in summary.get("strict_hidden_bit_drift_rows", [])],
                "strict_hidden_bit_drift_row_count": int(summary.get("strict_hidden_bit_drift_row_count", 0)),
                "token_failure_rows": [int(row) for row in summary.get("token_failure_rows", [])],
                "token_failure_row_count": int(summary.get("token_failure_row_count", 0)),
            }
            for summary in layer_summaries
        ],
    }


def _layer_execution_at_step(summary: dict[str, Any], *, decode_step: int, layer_index: int) -> dict[str, Any] | None:
    for step in summary.get("steps", []):
        if int(step.get("decode_step", -1)) != int(decode_step):
            continue
        decode_execution = step.get("batch_decode_execution")
        if not isinstance(decode_execution, dict):
            return None
        for layer_execution in decode_execution.get("layer_executions", []):
            if not isinstance(layer_execution, dict):
                continue
            if int(layer_execution.get("layer_index", -1)) == int(layer_index):
                return layer_execution
        return None
    return None


def _layer_execution_for_index(summary: dict[str, Any], layer_index: int) -> dict[str, Any] | None:
    for step in summary.get("steps", []):
        decode_execution = step.get("batch_decode_execution")
        if not isinstance(decode_execution, dict):
            continue
        for layer_execution in decode_execution.get("layer_executions", []):
            if not isinstance(layer_execution, dict):
                continue
            if int(layer_execution.get("layer_index", -1)) == int(layer_index):
                return layer_execution
    return None


def _top_abs_diff_in_comparison(comparison: dict[str, Any], *, flat_index: int) -> dict[str, Any] | None:
    for diff in comparison.get("top_abs_diffs", []):
        if isinstance(diff, dict) and int(diff.get("flat_index", -1)) == int(flat_index):
            return diff
    return None


def _top_abs_diff_for_flat_index(summary: dict[str, Any], *, row_index: int, flat_index: int) -> dict[str, Any] | None:
    for step in summary.get("steps", []):
        for row in step.get("rows", []):
            if int(row.get("row", -1)) != int(row_index):
                continue
            comparison = row.get("hidden_comparison", {})
            top_diff = _top_abs_diff_in_comparison(comparison, flat_index=flat_index)
            if top_diff is not None:
                return top_diff
    return None


def _row_focus_for_flat_index(summary: dict[str, Any], *, flat_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in summary.get("steps", []):
        for row in step.get("rows", []):
            comparison = row.get("hidden_comparison", {})
            row_index = int(row["row"])
            top_diff = _top_abs_diff_in_comparison(comparison, flat_index=flat_index)
            rows.append(
                {
                    "decode_step": step.get("decode_step"),
                    "generated_index": step.get("generated_index"),
                    "row": row_index,
                    "passed": bool(comparison.get("passed", False)),
                    "max_abs": comparison.get("max_abs"),
                    "max_abs_flat_index": comparison.get("max_abs_flat_index"),
                    "elements_over_atol": comparison.get("elements_over_atol"),
                    "same_flat_index_in_top_abs_diffs": top_diff is not None,
                    "same_flat_index_top_diff": top_diff,
                }
            )
    return rows


def _transition_trace_summaries(summary: dict[str, Any]) -> dict[str, Any]:
    traces: dict[str, Any] = {}
    for key in (
        "decode_full_attention",
        "decode_full_context_oracle",
        "decode_full_kv_samples",
        "decode_linear_inputs",
        "decode_linear_states",
    ):
        trace = summary.get(key)
        if not isinstance(trace, dict):
            continue
        compact: dict[str, Any] = {"passed": bool(trace.get("passed", True))}
        if "input_passed" in trace:
            compact["input_passed"] = bool(trace["input_passed"])
        if "output_passed" in trace:
            compact["output_passed"] = bool(trace["output_passed"])
        if isinstance(trace.get("stage_passed"), dict):
            compact["stage_passed"] = trace["stage_passed"]
        if "state_atol" in trace:
            compact["state_atol"] = float(trace["state_atol"])
        if "state_focus_atol" in trace:
            compact["state_focus_atol"] = float(trace["state_focus_atol"])
        if "passed_under_focus_atol" in trace:
            compact["passed_under_focus_atol"] = bool(trace["passed_under_focus_atol"])
        if isinstance(trace.get("first_mismatch"), dict):
            compact["first_mismatch"] = trace["first_mismatch"]
        if isinstance(trace.get("first_mismatch_over_focus_atol"), dict):
            compact["first_mismatch_over_focus_atol"] = trace["first_mismatch_over_focus_atol"]
        if isinstance(trace.get("worst_diff"), dict):
            compact["worst_diff"] = trace["worst_diff"]
        traces[key] = compact
    return traces


def _compact_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "passed": bool(comparison.get("passed", False)),
        "max_abs": float(comparison.get("max_abs", 0.0)),
        "max_abs_index": comparison.get("max_abs_index", []),
        "elements_over_atol": int(comparison.get("elements_over_atol", 0)),
    }
    if "max_abs_flat_index" in comparison:
        compact["max_abs_flat_index"] = comparison.get("max_abs_flat_index")
    if "bit_mismatch" in comparison:
        compact["bit_mismatch"] = int(comparison.get("bit_mismatch", 0))
    if "top_abs_diffs" in comparison:
        compact["top_abs_diffs"] = comparison.get("top_abs_diffs", [])[:3]
    for key in ("batch_residual_rms", "c1_residual_rms"):
        if key in comparison:
            compact[key] = float(comparison[key])
    for key in ("inferred_weight_valid_elements", "ignored_c1_zero_residual_elements"):
        if key in comparison:
            compact[key] = int(comparison[key])
    return compact


def _linear_state_mismatch_record(
    step: dict[str, Any],
    layer: dict[str, Any],
    state_name: str,
    row_summary: dict[str, Any],
    *,
    focus_atol: float | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "decode_step": int(step.get("decode_step", 0)),
        "generated_index": int(step.get("generated_index", int(step.get("decode_step", 0)) + 1)),
        "layer_index": int(layer.get("layer_index", -1)),
        "state": state_name,
        "row": int(row_summary.get("row", -1)),
        "max_abs": float(row_summary.get("max_abs", 0.0)),
        "max_abs_flat_index": row_summary.get("max_abs_flat_index"),
        "max_abs_index": row_summary.get("max_abs_index", []),
        "elements_over_atol": int(row_summary.get("elements_over_atol", 0)),
    }
    if focus_atol is not None:
        record["state_focus_atol"] = float(focus_atol)
        record["passed_under_focus_atol"] = bool(float(row_summary.get("max_abs", 0.0)) <= float(focus_atol))
    return record


def _linear_state_mismatch_from_trace(
    trace: dict[str, Any],
    *,
    first_key: str = "first_mismatch",
    focus_atol: float | None = None,
) -> dict[str, Any] | None:
    first = trace.get(first_key)
    if isinstance(first, dict):
        result = dict(first)
        if focus_atol is not None and "state_focus_atol" not in result:
            result["state_focus_atol"] = float(focus_atol)
            result["passed_under_focus_atol"] = bool(float(result.get("max_abs", 0.0)) <= float(focus_atol))
        return result
    for step in trace.get("steps", []):
        for layer in step.get("layers", []):
            states = layer.get("states", {})
            if not isinstance(states, dict):
                continue
            for state_name in ("conv", "recurrent"):
                state_summary = states.get(state_name)
                if not isinstance(state_summary, dict):
                    continue
                for row_summary in state_summary.get("row_summaries", []):
                    if focus_atol is None:
                        failed = not bool(row_summary.get("passed", False))
                    else:
                        failed = float(row_summary.get("max_abs", 0.0)) > float(focus_atol)
                    if not failed:
                        continue
                    return _linear_state_mismatch_record(
                        step,
                        layer,
                        state_name,
                        row_summary,
                        focus_atol=focus_atol,
                    )
    return None


def _hidden_row_comparison_at(summary: dict[str, Any], *, decode_step: int, row_index: int) -> dict[str, Any] | None:
    for step in summary.get("steps", []):
        if int(step.get("decode_step", -1)) != int(decode_step):
            continue
        for row in step.get("rows", []):
            if int(row.get("row", -1)) == int(row_index) and isinstance(row.get("hidden_comparison"), dict):
                return _compact_comparison(row["hidden_comparison"])
    return None


def _decode_linear_input_comparison_at(
    summary: dict[str, Any],
    *,
    decode_step: int,
    layer_index: int,
    row_index: int,
) -> dict[str, Any] | None:
    trace = summary.get("decode_linear_inputs")
    if not isinstance(trace, dict):
        return None
    for step in trace.get("steps", []):
        if int(step.get("decode_step", -1)) != int(decode_step):
            continue
        for layer in step.get("layers", []):
            if int(layer.get("layer_index", -1)) != int(layer_index):
                continue
            for row in layer.get("rows", []):
                if int(row.get("row", -1)) == int(row_index) and isinstance(row.get("hidden_comparison"), dict):
                    return _compact_comparison(row["hidden_comparison"])
    return None


def _decode_full_attention_layer_focus_at(
    summary: dict[str, Any],
    *,
    decode_step: int,
    layer_index: int,
    row_index: int,
) -> dict[str, Any] | None:
    trace = summary.get("decode_full_attention")
    if not isinstance(trace, dict):
        return None
    for step in trace.get("steps", []):
        if int(step.get("decode_step", -1)) != int(decode_step):
            continue
        for layer in step.get("layers", []):
            if int(layer.get("layer_index", -1)) != int(layer_index):
                continue
            focus: dict[str, Any] = {
                "decode_step": decode_step,
                "generated_index": int(step.get("generated_index", decode_step + 1)),
                "layer_index": layer_index,
                "row": row_index,
                "stage_passed": {},
                "stages": {},
            }
            layer_execution = _layer_execution_at_step(summary, decode_step=decode_step, layer_index=layer_index)
            if layer_execution is not None:
                focus["batch_decode_layer_execution"] = layer_execution
            first_bad_stage: dict[str, Any] | None = None
            for stage in DECODE_FULL_ATTENTION_TRACE_STAGES:
                stage_summary = layer.get("stages", {}).get(stage)
                if not isinstance(stage_summary, dict):
                    continue
                for row in stage_summary.get("rows", []):
                    if int(row.get("row", -1)) != row_index or not isinstance(row.get("hidden_comparison"), dict):
                        continue
                    compact = _compact_comparison(row["hidden_comparison"])
                    compact["comparison_kind"] = str(row.get("comparison_kind", "unknown"))
                    focus["stages"][stage] = compact
                    focus["stage_passed"][stage] = bool(compact["passed"])
                    if first_bad_stage is None and not bool(compact["passed"]):
                        first_bad_stage = {
                            "stage": stage,
                            "max_abs": float(compact["max_abs"]),
                            "max_abs_index": compact.get("max_abs_index", []),
                            "elements_over_atol": int(compact.get("elements_over_atol", 0)),
                        }
                    break
            stage_deltas = layer.get("stage_deltas", {})
            if isinstance(stage_deltas, dict):
                focus["stage_delta_passed"] = {}
                focus["stage_deltas"] = {}
                first_bad_delta: dict[str, Any] | None = None
                for delta_name, delta_summary in stage_deltas.items():
                    if not isinstance(delta_summary, dict):
                        continue
                    for row in delta_summary.get("rows", []):
                        if int(row.get("row", -1)) != row_index or not isinstance(row.get("delta_comparison"), dict):
                            continue
                        compact = _compact_comparison(row["delta_comparison"])
                        compact["comparison_kind"] = str(row.get("comparison_kind", "unknown"))
                        focus["stage_deltas"][str(delta_name)] = compact
                        focus["stage_delta_passed"][str(delta_name)] = bool(compact["passed"])
                        if first_bad_delta is None and not bool(compact["passed"]):
                            first_bad_delta = {
                                "stage_delta": str(delta_name),
                                "max_abs": float(compact["max_abs"]),
                                "max_abs_index": compact.get("max_abs_index", []),
                                "elements_over_atol": int(compact.get("elements_over_atol", 0)),
                            }
                        break
                if not focus["stage_deltas"]:
                    focus.pop("stage_deltas")
                    focus.pop("stage_delta_passed")
                elif first_bad_delta is not None:
                    focus["first_over_atol_stage_delta"] = first_bad_delta
            stage_oracles = layer.get("stage_oracles", {})
            if isinstance(stage_oracles, dict):
                focus["stage_oracle_passed"] = {}
                focus["stage_oracles"] = {}
                first_bad_oracle: dict[str, Any] | None = None
                for oracle_name, oracle_summary in stage_oracles.items():
                    if not isinstance(oracle_summary, dict):
                        continue
                    for row in oracle_summary.get("rows", []):
                        if int(row.get("row", -1)) != row_index or not isinstance(row.get("oracle_comparison"), dict):
                            continue
                        compact = _compact_comparison(row["oracle_comparison"])
                        compact["comparison_kind"] = str(row.get("comparison_kind", "unknown"))
                        focus["stage_oracles"][str(oracle_name)] = compact
                        focus["stage_oracle_passed"][str(oracle_name)] = bool(compact["passed"])
                        if first_bad_oracle is None and not bool(compact["passed"]):
                            first_bad_oracle = {
                                "stage_oracle": str(oracle_name),
                                "max_abs": float(compact["max_abs"]),
                                "max_abs_index": compact.get("max_abs_index", []),
                                "elements_over_atol": int(compact.get("elements_over_atol", 0)),
                            }
                        break
                if not focus["stage_oracles"]:
                    focus.pop("stage_oracles")
                    focus.pop("stage_oracle_passed")
                elif first_bad_oracle is not None:
                    focus["first_over_atol_stage_oracle"] = first_bad_oracle
            if first_bad_stage is not None:
                focus["first_over_atol_stage"] = first_bad_stage
            return focus if focus["stages"] else None
    return None


def _first_linear_state_mismatch_focus(
    summary: dict[str, Any],
    *,
    first_key: str = "first_mismatch",
    focus_atol_key: str | None = None,
) -> dict[str, Any] | None:
    trace = summary.get("decode_linear_states")
    if not isinstance(trace, dict):
        return None
    focus_atol = None
    if focus_atol_key is not None:
        if focus_atol_key not in trace:
            return None
        focus_atol = float(trace[focus_atol_key])
    first = _linear_state_mismatch_from_trace(trace, first_key=first_key, focus_atol=focus_atol)
    if first is None:
        return None
    decode_step = int(first.get("decode_step", -1))
    layer_index = int(first.get("layer_index", -1))
    row_index = int(first.get("row", -1))
    focus: dict[str, Any] = {**first}
    hidden_row = _hidden_row_comparison_at(summary, decode_step=decode_step, row_index=row_index)
    if hidden_row is not None:
        focus["hidden_row_at_state_mismatch"] = hidden_row
    linear_input = _decode_linear_input_comparison_at(
        summary,
        decode_step=decode_step,
        layer_index=layer_index,
        row_index=row_index,
    )
    if linear_input is not None:
        focus["decode_linear_input_at_state_mismatch"] = linear_input
    return focus


def _linear_state_focus_history(summary: dict[str, Any], focus: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(focus, dict):
        return []
    trace = summary.get("decode_linear_states")
    if not isinstance(trace, dict):
        return []
    layer_index = int(focus.get("layer_index", -1))
    state_name = str(focus.get("state", ""))
    row_index = int(focus.get("row", -1))
    if layer_index < 0 or not state_name or row_index < 0:
        return []
    focus_atol = focus.get("state_focus_atol")
    focus_flat_index = focus.get("max_abs_flat_index")
    focus_max_abs_index = focus.get("max_abs_index", [])

    def _attach_context(entry: dict[str, Any]) -> dict[str, Any]:
        decode_step = int(entry.get("decode_step", -1))
        layer_execution = _layer_execution_at_step(summary, decode_step=decode_step, layer_index=layer_index)
        if layer_execution is not None:
            entry["batch_decode_layer_execution"] = layer_execution
        hidden_row = _hidden_row_comparison_at(summary, decode_step=decode_step, row_index=row_index)
        if hidden_row is not None:
            entry["hidden_row"] = hidden_row
        linear_input = _decode_linear_input_comparison_at(
            summary,
            decode_step=decode_step,
            layer_index=layer_index,
            row_index=row_index,
        )
        if linear_input is not None:
            entry["decode_linear_input"] = linear_input
        producer_full_attention = _decode_full_attention_layer_focus_at(
            summary,
            decode_step=decode_step,
            layer_index=layer_index - 1,
            row_index=row_index,
        )
        if producer_full_attention is not None:
            entry["decode_linear_input_producer_full_attention"] = producer_full_attention
        return entry

    exact_history = trace.get("first_mismatch_over_focus_atol_history")
    if isinstance(exact_history, list):
        history = []
        for item in exact_history:
            if not isinstance(item, dict):
                continue
            if int(item.get("layer_index", -1)) != layer_index:
                continue
            if str(item.get("state", "")) != state_name:
                continue
            if int(item.get("row", -1)) != row_index:
                continue
            history.append(_attach_context(dict(item)))
        if history:
            return history

    history: list[dict[str, Any]] = []
    for step in trace.get("steps", []):
        decode_step = int(step.get("decode_step", -1))
        for layer in step.get("layers", []):
            if int(layer.get("layer_index", -1)) != layer_index:
                continue
            states = layer.get("states", {})
            if not isinstance(states, dict):
                continue
            state_summary = states.get(state_name)
            if not isinstance(state_summary, dict):
                continue
            for row_summary in state_summary.get("row_summaries", []):
                if int(row_summary.get("row", -1)) != row_index:
                    continue
                top_abs_diffs = row_summary.get("top_abs_diffs", [])
                same_focus_top_diff = None
                for top_diff in top_abs_diffs:
                    if not isinstance(top_diff, dict):
                        continue
                    if isinstance(focus_flat_index, int) and int(top_diff.get("flat_index", -1)) == int(focus_flat_index):
                        same_focus_top_diff = top_diff
                        break
                    if focus_max_abs_index and top_diff.get("index") == focus_max_abs_index:
                        same_focus_top_diff = top_diff
                        break
                entry: dict[str, Any] = {
                    "decode_step": decode_step,
                    "generated_index": int(step.get("generated_index", decode_step + 1)),
                    "layer_index": layer_index,
                    "state": state_name,
                    "row": row_index,
                    "passed": bool(row_summary.get("passed", False)),
                    "max_abs": float(row_summary.get("max_abs", 0.0)),
                    "max_abs_flat_index": row_summary.get("max_abs_flat_index"),
                    "max_abs_index": row_summary.get("max_abs_index", []),
                    "elements_over_atol": int(row_summary.get("elements_over_atol", 0)),
                    "focus_max_abs_index": focus_max_abs_index,
                    "same_focus_index_in_top_abs_diffs": same_focus_top_diff is not None,
                    "same_focus_index_top_diff": same_focus_top_diff,
                    "top_abs_diffs": top_abs_diffs,
                }
                if isinstance(focus_flat_index, int):
                    entry["focus_flat_index"] = int(focus_flat_index)
                if focus_atol is not None:
                    entry["state_focus_atol"] = float(focus_atol)
                    entry["passed_under_focus_atol"] = bool(float(row_summary.get("max_abs", 0.0)) <= float(focus_atol))
                history.append(_attach_context(entry))
                break
            break
    return history


def _linear_state_focus_for_hidden_mismatch(
    summary: dict[str, Any],
    first_hidden_mismatch: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(first_hidden_mismatch, dict):
        return []
    decode_step = int(first_hidden_mismatch.get("decode_step", -1))
    row_index = int(first_hidden_mismatch.get("row", -1))
    trace = summary.get("decode_linear_states")
    if not isinstance(trace, dict):
        return []
    focus: list[dict[str, Any]] = []
    for step in trace.get("steps", []):
        if int(step.get("decode_step", -1)) != decode_step:
            continue
        for layer in step.get("layers", []):
            layer_index = int(layer.get("layer_index", -1))
            states = layer.get("states", {})
            if not isinstance(states, dict):
                continue
            for state_name in ("conv", "recurrent"):
                state_summary = states.get(state_name)
                if not isinstance(state_summary, dict):
                    continue
                for row_summary in state_summary.get("row_summaries", []):
                    if int(row_summary.get("row", -1)) != row_index:
                        continue
                    focus.append(
                        {
                            "decode_step": decode_step,
                            "generated_index": int(step.get("generated_index", decode_step + 1)),
                            "layer_index": layer_index,
                            "state": state_name,
                            "row": row_index,
                            "passed": bool(row_summary.get("passed", False)),
                            "max_abs": float(row_summary.get("max_abs", 0.0)),
                            "max_abs_index": row_summary.get("max_abs_index", []),
                            "batch_value_at_max_abs": float(row_summary.get("batch_value_at_max_abs", 0.0)),
                            "c1_value_at_max_abs": float(row_summary.get("c1_value_at_max_abs", 0.0)),
                            "signed_diff_at_max_abs": float(row_summary.get("signed_diff_at_max_abs", 0.0)),
                            "elements_over_atol": int(row_summary.get("elements_over_atol", 0)),
                            "top_abs_diffs": row_summary.get("top_abs_diffs", []),
                        }
                    )
        break
    return focus


def _transition_hidden_focus(
    summary: dict[str, Any],
    previous_green: dict[str, Any] | None,
    first_hidden_mismatch: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(first_hidden_mismatch, dict):
        return None
    flat_index = first_hidden_mismatch.get("max_abs_flat_index")
    if not isinstance(flat_index, int) or isinstance(flat_index, bool):
        return None
    row_index = int(first_hidden_mismatch["row"])
    focus: dict[str, Any] = {
        "row": row_index,
        "flat_index": int(flat_index),
        "index": first_hidden_mismatch.get("max_abs_index", []),
        "failing_layer_limit": int(summary["layer_limit"]),
        "failing_top_diff": _top_abs_diff_for_flat_index(summary, row_index=row_index, flat_index=int(flat_index)),
        "failing_rows_for_flat_index": _row_focus_for_flat_index(summary, flat_index=int(flat_index)),
    }
    if previous_green is not None:
        previous_diff = _top_abs_diff_for_flat_index(previous_green, row_index=row_index, flat_index=int(flat_index))
        focus["previous_green_layer_limit"] = int(previous_green["layer_limit"])
        focus["previous_green_same_flat_index_in_top_abs_diffs"] = previous_diff is not None
        focus["previous_green_same_flat_index_top_diff"] = previous_diff
        focus["previous_green_rows_for_flat_index"] = _row_focus_for_flat_index(previous_green, flat_index=int(flat_index))
    return focus


def _first_failing_layer_transition(layer_summaries: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    previous_green: dict[str, Any] | None = None
    for summary in layer_summaries:
        hidden_passed = bool(summary.get("hidden_passed", False))
        token_passed = bool(summary.get("token_passed", False))
        if hidden_passed and token_passed:
            previous_green = summary
            continue
        layer_limit = int(summary["layer_limit"])
        hidden_rows = _hidden_failure_rows(summary)
        strict_hidden_rows = _hidden_bit_drift_rows(summary)
        token_rows = _token_failure_rows(summary)
        failure_modes: list[str] = []
        if not hidden_passed:
            failure_modes.append("hidden")
        if not token_passed:
            failure_modes.append("token")
        failing_last_layer_index = int(summary.get("last_layer_index", layer_limit - 1))
        failing_layer_execution = _layer_execution_for_index(summary, failing_last_layer_index)
        first_hidden_mismatch = _first_hidden_mismatch([summary])
        first_hidden_bit_drift = _first_hidden_bit_drift([summary])
        transition: dict[str, Any] = {
            "failing_layer_limit": layer_limit,
            "failing_last_layer_index": failing_last_layer_index,
            "failing_layer_execution": failing_layer_execution,
            "failure_modes": failure_modes,
            "hidden_atol": float(summary.get("hidden_atol", 0.0)),
            "hidden_passed": hidden_passed,
            "token_passed": token_passed,
            "hidden_failure_rows": hidden_rows,
            "hidden_failure_row_count": len(hidden_rows),
            "strict_hidden_bit_drift_rows": strict_hidden_rows,
            "strict_hidden_bit_drift_row_count": len(strict_hidden_rows),
            "token_failure_rows": token_rows,
            "token_failure_row_count": len(token_rows),
            "first_hidden_mismatch": first_hidden_mismatch,
            "first_tolerance_hidden_mismatch": first_hidden_mismatch,
            "first_strict_hidden_bit_drift": first_hidden_bit_drift,
            "hidden_mismatch_kind": (
                "over_atol"
                if first_hidden_mismatch is not None
                else ("bit_drift_only" if first_hidden_bit_drift is not None else None)
            ),
            "first_token_mismatch": _first_token_mismatch([summary]),
        }
        if "last_layer_type" in summary:
            transition["failing_last_layer_type"] = str(summary["last_layer_type"])
        failing_trace_summaries = _transition_trace_summaries(summary)
        if failing_trace_summaries:
            transition["failing_trace_summaries"] = failing_trace_summaries
        if previous_green is not None:
            previous_limit = int(previous_green["layer_limit"])
            transition.update(
                {
                    "previous_green_layer_limit": previous_limit,
                    "previous_green_last_layer_index": int(previous_green.get("last_layer_index", previous_limit - 1)),
                    "previous_green_hidden_passed": bool(previous_green.get("hidden_passed", False)),
                    "previous_green_token_passed": bool(previous_green.get("token_passed", False)),
                    "adjacent_layer_limits": bool(layer_limit - previous_limit == 1),
                }
            )
            previous_layer_execution = _layer_execution_for_index(previous_green, int(previous_green.get("last_layer_index", previous_limit - 1)))
            transition["previous_green_layer_execution"] = previous_layer_execution
            if "last_layer_type" in previous_green:
                transition["previous_green_last_layer_type"] = str(previous_green["last_layer_type"])
            previous_green_trace_summaries = _transition_trace_summaries(previous_green)
            if previous_green_trace_summaries:
                transition["previous_green_trace_summaries"] = previous_green_trace_summaries
        transition["first_hidden_mismatch_focus"] = _transition_hidden_focus(summary, previous_green, first_hidden_mismatch)
        transition["first_linear_state_mismatch_focus"] = _first_linear_state_mismatch_focus(summary)
        first_focus_state = _first_linear_state_mismatch_focus(
            summary,
            first_key="first_mismatch_over_focus_atol",
            focus_atol_key="state_focus_atol",
        )
        if first_focus_state is not None:
            transition["first_linear_state_mismatch_over_focus_atol"] = first_focus_state
            transition["first_linear_state_mismatch_over_focus_atol_history"] = _linear_state_focus_history(
                summary,
                first_focus_state,
            )
        transition["first_hidden_mismatch_linear_state_focus"] = _linear_state_focus_for_hidden_mismatch(
            summary,
            first_hidden_mismatch,
        )
        return transition
    return None


def _copy_hidden_bits(session: Qwen35ParoResidentSession, hidden, *, rows: int) -> np.ndarray:
    bits = np.empty((rows, session.config.hidden_size), dtype=np.uint16)
    copy_device_to_host(
        host_array_ptr(bits),
        DeviceBuffer(hidden.ptr, bits.nbytes),
        runtime=session.runtime,
    )
    return bits


def _prefill_batch(
    session: Qwen35ParoResidentSession,
    prompts: list[list[int]],
    *,
    decode_tokens: int,
) -> list[int]:
    scheduler = ResidentBatchScheduler(capacity=len(prompts))
    request_ids = [scheduler.submit(prompt, max_new_tokens=decode_tokens) for prompt in prompts]
    admitted = scheduler.admit_pending()
    if tuple(request_ids) != tuple(admitted):
        raise RuntimeError(f"unexpected admitted request ids {admitted!r}")
    slabs = scheduler.next_compact_prefill_slabs(chunk_size=max(len(prompt) for prompt in prompts), block_size=session.block_size)
    if len(slabs) != 1:
        raise RuntimeError(f"expected one compact prefill slab, got {len(slabs)}")
    results = session.prefill_native_packed(slabs[0], sample=True)
    seed_tokens: list[int] = []
    for result in results:
        if result is None:
            raise RuntimeError("batch prefill did not produce a seed token")
        seed_tokens.append(int(result.token_id))
    return seed_tokens


def _batch_prefill_hidden_tensor(session: Qwen35ParoResidentSession, *, rows: int) -> Tensor:
    return Tensor.from_handle(session.batch_hidden.ptr, (rows, session.config.hidden_size), DType.FP16, session.device)


def _copy_tensor_f32(session: Qwen35ParoResidentSession, tensor: Tensor) -> np.ndarray:
    if tensor.dtype != DType.FP32:
        raise ValueError(f"expected FP32 tensor, got {tensor.dtype}")
    array = np.empty(tuple(int(dim) for dim in tensor.shape), dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(array),
        DeviceBuffer(tensor.ptr, array.nbytes),
        runtime=session.runtime,
    )
    return array


def _copy_prefill_linear_states(session: Qwen35ParoResidentSession, *, rows: int) -> dict[int, dict[str, np.ndarray]]:
    states: dict[int, dict[str, np.ndarray]] = {}
    layer_types = tuple(str(layer_type) for layer_type in getattr(session.config, "layer_types", ()))
    for layer_id, layer_type in enumerate(layer_types[: len(session.states)]):
        if layer_type != "linear_attention":
            continue
        conv_rows: list[np.ndarray] = []
        recurrent_rows: list[np.ndarray] = []
        for slot in range(rows):
            conv_state, recurrent_state = session._slot_linear_state(layer_id, slot)
            conv_rows.append(_copy_tensor_f32(session, conv_state))
            recurrent_rows.append(_copy_tensor_f32(session, recurrent_state))
        states[int(layer_id)] = {
            "conv": np.stack(conv_rows, axis=0),
            "recurrent": np.stack(recurrent_rows, axis=0),
        }
    return states


def _prefill_linear_input_rows_from_trace(
    trace: Sequence[dict[str, Any]] | None,
    *,
    prompt_lengths: Sequence[int],
) -> dict[int, list[np.ndarray]]:
    rows_by_layer: dict[int, list[np.ndarray]] = {}
    if not trace:
        return rows_by_layer
    total_tokens = sum(int(length) for length in prompt_lengths)
    for entry in trace:
        layer_id = int(entry["layer_index"])
        bits = np.asarray(entry["bits"], dtype=np.uint16)
        if bits.ndim != 2:
            raise ValueError(f"prefill linear input trace for layer {layer_id} must be rank-2")
        if int(bits.shape[0]) < total_tokens:
            raise ValueError(
                f"prefill linear input trace for layer {layer_id} has {bits.shape[0]} rows, expected at least {total_tokens}"
            )
        offset = 0
        rows: list[np.ndarray] = []
        for length in prompt_lengths:
            end = offset + int(length)
            rows.append(bits[offset:end].copy())
            offset = end
        rows_by_layer[layer_id] = rows
    return rows_by_layer


def _merge_prefill_linear_input_rows(
    target: dict[int, list[np.ndarray]],
    captured: dict[int, list[np.ndarray]],
) -> None:
    for layer_id, rows in captured.items():
        if len(rows) != 1:
            raise ValueError("c=1 prefill input traces must contain exactly one row")
        target.setdefault(int(layer_id), []).append(rows[0].copy())


def _decode_linear_input_layers_from_trace(trace: Sequence[dict[str, Any]] | None) -> dict[int, np.ndarray]:
    layers: dict[int, np.ndarray] = {}
    if not trace:
        return layers
    for entry in trace:
        layer_id = int(entry["layer_index"])
        bits = np.asarray(entry["bits"], dtype=np.uint16)
        if bits.ndim != 2:
            raise ValueError(f"decode linear input trace for layer {layer_id} must be rank-2")
        layers[layer_id] = bits.copy()
    return layers


def _decode_full_attention_layers_from_trace(
    trace: Sequence[dict[str, Any]] | None,
) -> dict[int, dict[str, np.ndarray]]:
    grouped: dict[int, dict[str, list[np.ndarray]]] = {}
    if not trace:
        return {}
    for entry in trace:
        layer_id = int(entry["layer_index"])
        stage = str(entry.get("stage", ""))
        if stage not in DECODE_FULL_ATTENTION_TRACE_STAGES:
            raise ValueError(f"decode full-attention trace for layer {layer_id} has unrecognized stage {stage!r}")
        if "bits" in entry:
            values = np.asarray(entry["bits"], dtype=np.uint16)
        elif "values" in entry:
            values = np.asarray(entry["values"], dtype=np.float32)
        else:
            raise ValueError(f"decode full-attention trace for layer {layer_id}, stage {stage} has no payload")
        if values.ndim != 2:
            raise ValueError(f"decode full-attention trace for layer {layer_id}, stage {stage} must be rank-2")
        grouped.setdefault(layer_id, {}).setdefault(stage, []).append(values.copy())
    return {
        int(layer_id): {stage: np.concatenate(rows, axis=0) for stage, rows in stages.items()}
        for layer_id, stages in grouped.items()
    }


def _merge_decode_full_attention_rows(
    target: dict[int, dict[str, list[np.ndarray]]],
    captured: dict[int, dict[str, np.ndarray]],
) -> None:
    for layer_id, stages in captured.items():
        target_layer = target.setdefault(int(layer_id), {})
        for stage, bits in stages.items():
            if int(bits.shape[0]) != 1:
                raise ValueError("c=1 decode full-attention traces must contain exactly one row per stage")
            target_layer.setdefault(str(stage), []).append(bits.copy())


def _stack_decode_full_attention_rows(
    rows_by_layer: dict[int, dict[str, list[np.ndarray]]]
) -> dict[int, dict[str, np.ndarray]]:
    return {
        int(layer_id): {stage: np.concatenate(rows, axis=0) for stage, rows in stages.items()}
        for layer_id, stages in rows_by_layer.items()
    }


def _bf16_token_crc32(bits: np.ndarray) -> np.ndarray:
    bits_array = np.asarray(bits, dtype=np.uint16)
    token_bits = np.ascontiguousarray(bits_array.reshape(int(bits_array.shape[0]), -1))
    return np.asarray([zlib.crc32(row.tobytes()) for row in token_bits], dtype=np.uint64)


def _pad_row_arrays(arrays: Sequence[np.ndarray], *, dtype: np.dtype | type) -> np.ndarray:
    if not arrays:
        return np.empty((0, 0), dtype=dtype)
    row_count = len(arrays)
    max_width = max(int(np.asarray(array).shape[1]) for array in arrays)
    padded = np.zeros((row_count, max_width), dtype=dtype)
    for row, array in enumerate(arrays):
        row_array = np.asarray(array, dtype=dtype)
        if row_array.ndim != 2 or int(row_array.shape[0]) != 1:
            raise ValueError("padded row arrays must have shape [1, width]")
        padded[row, : int(row_array.shape[1])] = row_array[0]
    return padded


def _numpy_full_attention_context_row(
    query: np.ndarray,
    key_bits: np.ndarray,
    value_bits: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    query_f32 = np.asarray(query, dtype=np.float32)
    key = _bf16_bits_to_f32(key_bits)
    value = _bf16_bits_to_f32(value_bits)
    if query_f32.ndim != 2 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("full-attention context oracle expects query [heads,dim] and KV [tokens,kv_heads,dim]")
    if key.shape != value.shape:
        raise ValueError("full-attention context oracle key/value shapes must match")
    num_q_heads, head_dim = (int(dim) for dim in query_f32.shape)
    context_len, num_kv_heads, kv_head_dim = (int(dim) for dim in key.shape)
    if context_len <= 0:
        raise ValueError("full-attention context oracle requires at least one live token")
    if kv_head_dim != head_dim or num_q_heads % num_kv_heads != 0:
        raise ValueError("full-attention context oracle has incompatible query/KV shapes")
    kv_group = num_q_heads // num_kv_heads
    out = np.empty((num_q_heads, head_dim), dtype=np.float32)
    for q_head in range(num_q_heads):
        kv_head = q_head // kv_group
        scores = np.empty((context_len,), dtype=np.float32)
        for token in range(context_len):
            scores[token] = float(np.sum(query_f32[q_head] * key[token, kv_head], dtype=np.float32) * scale)
        probs = np.exp(scores - np.max(scores)).astype(np.float32)
        probs /= np.sum(probs, dtype=np.float32)
        out[q_head] = np.sum(probs[:, None] * value[:, kv_head, :], axis=0, dtype=np.float32)
    return out


def _copy_decode_full_kv_prefix_bits(
    session: Qwen35ParoResidentSession,
    *,
    layer_id: int,
    slot: int,
    live_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if live_count <= 0:
        raise ValueError("live_count must be positive")
    key_cache, value_cache = session._slot_full_cache(layer_id, int(slot))
    if key_cache.dtype != DType.BF16 or value_cache.dtype != DType.BF16:
        raise ValueError("full-attention context oracle currently expects BF16 retained KV")
    if len(key_cache.shape) != 4 or len(value_cache.shape) != 4:
        raise ValueError(f"full-attention KV cache for layer {layer_id} must be rank-4")
    blocks, block_size, num_kv_heads, head_dim = (int(dim) for dim in key_cache.shape)
    if int(live_count) > blocks * block_size:
        raise ValueError("live_count exceeds retained KV cache capacity")
    shape = (int(live_count), num_kv_heads, head_dim)
    nbytes = int(np.prod(shape)) * DType.BF16.itemsize
    key_bits = np.empty(shape, dtype=np.uint16)
    value_bits = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(
        host_array_ptr(key_bits),
        DeviceBuffer(key_cache.ptr, nbytes),
        runtime=session.runtime,
    )
    copy_device_to_host(
        host_array_ptr(value_bits),
        DeviceBuffer(value_cache.ptr, nbytes),
        runtime=session.runtime,
    )
    return key_bits, value_bits


def _copy_full_kv_prefix_hashes(
    session: Qwen35ParoResidentSession,
    *,
    rows: int,
    prompt_lengths: Sequence[int],
    slots: Sequence[int],
) -> dict[int, dict[str, np.ndarray]]:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if len(prompt_lengths) != rows or len(slots) != rows:
        raise ValueError("prompt_lengths and slots must match rows")
    context_lens = np.asarray([int(length) for length in prompt_lengths], dtype=np.int64)
    if np.any(context_lens <= 0):
        raise ValueError("prompt lengths must be positive")
    max_context_len = int(np.max(context_lens))
    hashes: dict[int, dict[str, np.ndarray]] = {}
    layer_types = tuple(str(layer_type) for layer_type in getattr(session.config, "layer_types", ()))
    for layer_id, layer_type in enumerate(layer_types[: len(session.states)]):
        if layer_type != "full_attention":
            continue
        key_prefix_hashes = np.zeros((rows, max_context_len), dtype=np.uint64)
        value_prefix_hashes = np.zeros((rows, max_context_len), dtype=np.uint64)
        for row, (slot, live_count) in enumerate(zip(slots, context_lens, strict=True)):
            live = int(live_count)
            key_bits, value_bits = _copy_decode_full_kv_prefix_bits(
                session,
                layer_id=int(layer_id),
                slot=int(slot),
                live_count=live,
            )
            key_prefix_hashes[row, :live] = _bf16_token_crc32(key_bits)
            value_prefix_hashes[row, :live] = _bf16_token_crc32(value_bits)
        hashes[int(layer_id)] = {
            "context_lens": context_lens.copy(),
            "key_prefix_hashes": key_prefix_hashes,
            "value_prefix_hashes": value_prefix_hashes,
        }
    return hashes


def _merge_full_kv_prefix_hash_rows(
    target: dict[int, dict[str, list[np.ndarray]]],
    captured: dict[int, dict[str, np.ndarray]],
) -> None:
    for layer_id, payload in captured.items():
        target_layer = target.setdefault(int(layer_id), {"context_lens": [], "key_prefix_hashes": [], "value_prefix_hashes": []})
        for key in ("context_lens", "key_prefix_hashes", "value_prefix_hashes"):
            array = np.asarray(payload[key])
            if int(array.shape[0]) != 1:
                raise ValueError("c=1 prefill full-KV prefix hashes must contain exactly one row")
            target_layer.setdefault(key, []).append(array.copy())


def _stack_full_kv_prefix_hash_rows(
    rows_by_layer: dict[int, dict[str, list[np.ndarray]]]
) -> dict[int, dict[str, np.ndarray]]:
    stacked: dict[int, dict[str, np.ndarray]] = {}
    for layer_id, payload in rows_by_layer.items():
        stacked[int(layer_id)] = {
            "context_lens": np.concatenate(payload.get("context_lens", []), axis=0),
            "key_prefix_hashes": _pad_row_arrays(payload.get("key_prefix_hashes", []), dtype=np.uint64),
            "value_prefix_hashes": _pad_row_arrays(payload.get("value_prefix_hashes", []), dtype=np.uint64),
        }
    return stacked


def _decode_full_context_oracles_from_trace(
    session: Qwen35ParoResidentSession,
    traced_layers: dict[int, dict[str, np.ndarray]],
    *,
    rows: int,
    positions: Sequence[int],
    slots: Sequence[int],
) -> dict[int, dict[str, np.ndarray]]:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if len(positions) != rows or len(slots) != rows:
        raise ValueError("positions and slots must match rows")
    if not traced_layers:
        return {}
    num_q_heads = int(session.config.num_attention_heads)
    head_dim = int(session.config.head_dim)
    scale = np.float32(head_dim ** -0.5)
    context_lens = np.asarray([int(position) + 1 for position in positions], dtype=np.int64)
    oracles: dict[int, dict[str, np.ndarray]] = {}
    for layer_id, stages in traced_layers.items():
        query_source = "query_after_prepare" if "query_after_prepare" in stages else "query"
        if query_source not in stages or "attn_context" not in stages:
            continue
        query_flat = np.asarray(stages[query_source], dtype=np.float32)
        expected_width = num_q_heads * head_dim
        if query_flat.shape != (rows, expected_width):
            raise ValueError(
                f"decode full-attention {query_source} trace for layer {layer_id} has shape {query_flat.shape}, "
                f"expected {(rows, expected_width)}"
            )
        query = query_flat.reshape(rows, num_q_heads, head_dim)
        context = np.empty((rows, num_q_heads, head_dim), dtype=np.float32)
        max_context_len = int(np.max(context_lens))
        key_prefix_hashes = np.zeros((rows, max_context_len), dtype=np.uint64)
        value_prefix_hashes = np.zeros((rows, max_context_len), dtype=np.uint64)
        for row, (slot, live_count) in enumerate(zip(slots, context_lens, strict=True)):
            key_bits, value_bits = _copy_decode_full_kv_prefix_bits(
                session,
                layer_id=int(layer_id),
                slot=int(slot),
                live_count=int(live_count),
            )
            live = int(live_count)
            key_prefix_hashes[row, :live] = _bf16_token_crc32(key_bits)
            value_prefix_hashes[row, :live] = _bf16_token_crc32(value_bits)
            context[row] = _numpy_full_attention_context_row(
                query[row],
                key_bits,
                value_bits,
                scale=float(scale),
            )
        oracles[int(layer_id)] = {
            "context": context.reshape(rows, expected_width),
            "context_lens": context_lens.copy(),
            "key_prefix_hashes": key_prefix_hashes,
            "value_prefix_hashes": value_prefix_hashes,
            "query_source": query_source,
        }
    return oracles


def _merge_decode_full_context_oracle_rows(
    target: dict[int, dict[str, list[np.ndarray]]],
    captured: dict[int, dict[str, np.ndarray]],
) -> None:
    for layer_id, payload in captured.items():
        target_layer = target.setdefault(int(layer_id), {"context": [], "context_lens": [], "query_source": []})
        for key in ("context", "context_lens", "key_prefix_hashes", "value_prefix_hashes"):
            if key not in payload:
                continue
            array = np.asarray(payload[key])
            if int(array.shape[0]) != 1:
                raise ValueError("c=1 decode full-context oracle traces must contain exactly one row")
            target_layer.setdefault(key, []).append(array.copy())
        target_layer.setdefault("query_source", []).append(str(payload.get("query_source", "query")))


def _stack_decode_full_context_oracle_rows(
    rows_by_layer: dict[int, dict[str, list[np.ndarray]]]
) -> dict[int, dict[str, np.ndarray]]:
    stacked: dict[int, dict[str, np.ndarray]] = {}
    for layer_id, payload in rows_by_layer.items():
        query_sources = [str(source) for source in payload.get("query_source", [])]
        query_source = query_sources[0] if query_sources else "query"
        if any(source != query_source for source in query_sources):
            raise ValueError(f"decode full-context oracle query source differs across c=1 rows for layer {layer_id}")
        layer_payload: dict[str, np.ndarray | str] = {
            "context": np.concatenate(payload.get("context", []), axis=0),
            "context_lens": np.concatenate(payload.get("context_lens", []), axis=0),
            "query_source": query_source,
        }
        if payload.get("key_prefix_hashes"):
            layer_payload["key_prefix_hashes"] = _pad_row_arrays(payload.get("key_prefix_hashes", []), dtype=np.uint64)
        if payload.get("value_prefix_hashes"):
            layer_payload["value_prefix_hashes"] = _pad_row_arrays(payload.get("value_prefix_hashes", []), dtype=np.uint64)
        stacked[int(layer_id)] = layer_payload  # type: ignore[assignment]
    return stacked


DECODE_FULL_KV_SAMPLE_LABELS = ("first", "page0_last", "page1_first", "previous", "current")


def _decode_full_kv_sample_positions(position: int, *, block_size: int = 256) -> tuple[int, int, int, int, int]:
    pos = int(position)
    if pos < 0:
        raise ValueError("position must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return (0, min(pos, block_size - 1), min(pos, block_size), max(0, pos - 1), pos)


def _copy_decode_full_kv_samples(
    session: Qwen35ParoResidentSession,
    *,
    rows: int,
    positions: Sequence[int],
    slots: Sequence[int],
) -> dict[int, dict[str, np.ndarray | tuple[str, ...]]]:
    samples: dict[int, dict[str, np.ndarray | tuple[str, ...]]] = {}
    layer_types = tuple(str(layer_type) for layer_type in getattr(session.config, "layer_types", ()))
    if len(positions) != rows or len(slots) != rows:
        raise ValueError("positions and slots must match rows")
    for layer_id, layer_type in enumerate(layer_types[: len(session.states)]):
        if layer_type != "full_attention":
            continue
        key_rows: list[np.ndarray] = []
        value_rows: list[np.ndarray] = []
        sample_position_rows: list[np.ndarray] = []
        for slot, position in zip(slots, positions, strict=True):
            key_cache, value_cache = session._slot_full_cache(layer_id, int(slot))
            if key_cache.dtype != DType.BF16 or value_cache.dtype != DType.BF16:
                continue
            if len(key_cache.shape) != 4 or len(value_cache.shape) != 4:
                raise ValueError(f"full-attention KV cache for layer {layer_id} must be rank-4")
            _blocks, block_size, num_kv_heads, head_dim = (int(dim) for dim in key_cache.shape)
            sample_positions = np.asarray(_decode_full_kv_sample_positions(int(position), block_size=block_size), dtype=np.int64)
            sample_position_rows.append(sample_positions)
            key_samples = np.empty((len(sample_positions), num_kv_heads, head_dim), dtype=np.uint16)
            value_samples = np.empty_like(key_samples)
            token_width_bytes = num_kv_heads * head_dim * DType.BF16.itemsize
            for sample_idx, token_position in enumerate(sample_positions):
                token = int(token_position)
                block = token // block_size
                block_offset = token - block * block_size
                token_offset_bytes = (block * block_size + block_offset) * token_width_bytes
                copy_device_to_host(
                    host_array_ptr(key_samples[sample_idx]),
                    DeviceBuffer(key_cache.ptr + token_offset_bytes, token_width_bytes),
                    runtime=session.runtime,
                )
                copy_device_to_host(
                    host_array_ptr(value_samples[sample_idx]),
                    DeviceBuffer(value_cache.ptr + token_offset_bytes, token_width_bytes),
                    runtime=session.runtime,
                )
            key_rows.append(key_samples)
            value_rows.append(value_samples)
        if key_rows:
            samples[int(layer_id)] = {
                "sample_labels": DECODE_FULL_KV_SAMPLE_LABELS,
                "sample_positions": np.stack(sample_position_rows, axis=0),
                "key_bits": np.stack(key_rows, axis=0),
                "value_bits": np.stack(value_rows, axis=0),
            }
    return samples


def _merge_decode_full_kv_sample_rows(
    target: dict[int, dict[str, list[np.ndarray] | tuple[str, ...]]],
    captured: dict[int, dict[str, np.ndarray | tuple[str, ...]]],
) -> None:
    for layer_id, sample in captured.items():
        target_layer = target.setdefault(int(layer_id), {"sample_labels": DECODE_FULL_KV_SAMPLE_LABELS})
        for key in ("sample_positions", "key_bits", "value_bits"):
            array = np.asarray(sample[key])
            if int(array.shape[0]) != 1:
                raise ValueError("c=1 decode full-KV samples must contain exactly one row")
            target_layer.setdefault(key, []).append(array.copy())  # type: ignore[union-attr]


def _stack_decode_full_kv_sample_rows(
    rows_by_layer: dict[int, dict[str, list[np.ndarray] | tuple[str, ...]]]
) -> dict[int, dict[str, np.ndarray | tuple[str, ...]]]:
    stacked: dict[int, dict[str, np.ndarray | tuple[str, ...]]] = {}
    for layer_id, sample in rows_by_layer.items():
        stacked[int(layer_id)] = {
            "sample_labels": tuple(sample.get("sample_labels", DECODE_FULL_KV_SAMPLE_LABELS)),
            "sample_positions": np.concatenate(sample.get("sample_positions", []), axis=0),
            "key_bits": np.concatenate(sample.get("key_bits", []), axis=0),
            "value_bits": np.concatenate(sample.get("value_bits", []), axis=0),
        }
    return stacked


def _merge_decode_linear_input_row(
    target: dict[int, list[np.ndarray]],
    captured: dict[int, np.ndarray],
) -> None:
    for layer_id, bits in captured.items():
        if int(bits.shape[0]) != 1:
            raise ValueError("c=1 decode input traces must contain exactly one row")
        target.setdefault(int(layer_id), []).append(bits.copy())


def _stack_decode_linear_input_rows(rows_by_layer: dict[int, list[np.ndarray]]) -> dict[int, np.ndarray]:
    return {int(layer_id): np.concatenate(rows, axis=0) for layer_id, rows in rows_by_layer.items()}


def _merge_prefill_linear_state_row(
    target: dict[int, dict[str, list[np.ndarray]]],
    captured: dict[int, dict[str, np.ndarray]],
) -> None:
    for layer_id, layer_states in captured.items():
        target_layer = target.setdefault(layer_id, {"conv": [], "recurrent": []})
        target_layer["conv"].append(layer_states["conv"][0].copy())
        target_layer["recurrent"].append(layer_states["recurrent"][0].copy())


def _stack_prefill_linear_state_rows(
    rows_by_layer: dict[int, dict[str, list[np.ndarray]]]
) -> dict[int, dict[str, np.ndarray]]:
    return {
        int(layer_id): {
            "conv": np.stack(layer_states["conv"], axis=0),
            "recurrent": np.stack(layer_states["recurrent"], axis=0),
        }
        for layer_id, layer_states in rows_by_layer.items()
    }


def _run_batch_hidden(
    runner: Qwen35ParoNextTokenRunner,
    prompts: list[list[int]],
    *,
    layer_limit: int,
    decode_tokens: int,
    max_sequence_length: int,
    compiler_version: str | None,
    require_cached_build: bool,
) -> HiddenRun:
    rows = len(prompts)
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=layer_limit,
        max_batch_size=rows,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
    ) as session:
        session._prefill_linear_input_trace = []
        seed_tokens = _prefill_batch(session, prompts, decode_tokens=decode_tokens)
        session.runtime.device_synchronize()
        prefill_hidden_bits = _copy_hidden_bits(session, _batch_prefill_hidden_tensor(session, rows=rows), rows=rows)
        prefill_linear_states = _copy_prefill_linear_states(session, rows=rows)
        prompt_lengths = [len(prompt) for prompt in prompts]
        prefill_linear_inputs = _prefill_linear_input_rows_from_trace(
            getattr(session, "_prefill_linear_input_trace", None),
            prompt_lengths=prompt_lengths,
        )
        prefill_full_kv_prefix_hashes = _copy_full_kv_prefix_hashes(
            session,
            rows=rows,
            prompt_lengths=prompt_lengths,
            slots=tuple(range(rows)),
        )
        prefill_execution = getattr(session, "last_prefill_execution", None)
        prefill_execution_copy = json.loads(json.dumps(prefill_execution)) if isinstance(prefill_execution, dict) else None
        next_tokens = list(seed_tokens)
        generated_tokens = [[] for _ in prompts]
        hidden_bits_by_step: list[np.ndarray] = []
        decode_linear_inputs_by_step: list[dict[int, np.ndarray]] = []
        decode_full_attention_by_step: list[dict[int, dict[str, np.ndarray]]] = []
        decode_full_context_oracles_by_step: list[dict[int, dict[str, np.ndarray]]] = []
        decode_full_kv_samples_by_step: list[dict[int, dict[str, np.ndarray | tuple[str, ...]]]] = []
        decode_linear_states_by_step: list[dict[int, dict[str, np.ndarray]]] = []
        decode_execution_by_step: list[dict[str, Any] | None] = []
        for step in range(decode_tokens):
            positions = tuple(len(prompt) + step for prompt in prompts)
            session._decode_linear_input_trace = []
            session._decode_full_attention_trace = []
            session._set_batch_token_embeddings(next_tokens, stream=0)
            session._set_batch_positions(positions, stream=0)
            hidden = session._run_layers_batch_decode(
                rows=rows,
                positions=positions,
                slots=tuple(range(rows)),
                stream=0,
            )
            decode_execution = getattr(session, "last_batch_decode_execution", None)
            decode_execution_by_step.append(
                json.loads(json.dumps(decode_execution)) if isinstance(decode_execution, dict) else None
            )
            session.runtime.device_synchronize()
            hidden_bits_by_step.append(_copy_hidden_bits(session, hidden, rows=rows))
            decode_linear_inputs_by_step.append(
                _decode_linear_input_layers_from_trace(getattr(session, "_decode_linear_input_trace", None))
            )
            decode_full_attention_layers = _decode_full_attention_layers_from_trace(
                getattr(session, "_decode_full_attention_trace", None)
            )
            decode_full_attention_by_step.append(decode_full_attention_layers)
            decode_full_context_oracles_by_step.append(
                _decode_full_context_oracles_from_trace(
                    session,
                    decode_full_attention_layers,
                    rows=rows,
                    positions=positions,
                    slots=tuple(range(rows)),
                )
            )
            decode_full_kv_samples_by_step.append(
                _copy_decode_full_kv_samples(
                    session,
                    rows=rows,
                    positions=positions,
                    slots=tuple(range(rows)),
                )
            )
            decode_linear_states_by_step.append(_copy_prefill_linear_states(session, rows=rows))
            results = session._sample_batch_from_hidden(hidden, rows=rows)
            next_tokens = []
            for row, result in enumerate(results):
                token_id = int(result.token_id)
                generated_tokens[row].append(token_id)
                next_tokens.append(token_id)
        return HiddenRun(
            seed_tokens=seed_tokens,
            generated_tokens=generated_tokens,
            hidden_bits_by_step=hidden_bits_by_step,
            prefill_hidden_bits=prefill_hidden_bits,
            prefill_execution=prefill_execution_copy,
            prefill_linear_states=prefill_linear_states,
            prefill_linear_inputs=prefill_linear_inputs,
            prefill_full_kv_prefix_hashes=prefill_full_kv_prefix_hashes,
            decode_linear_inputs_by_step=decode_linear_inputs_by_step,
            decode_full_attention_by_step=decode_full_attention_by_step,
            decode_full_context_oracles_by_step=decode_full_context_oracles_by_step,
            decode_full_kv_samples_by_step=decode_full_kv_samples_by_step,
            decode_linear_states_by_step=decode_linear_states_by_step,
            decode_execution_by_step=decode_execution_by_step,
        )


def _run_c1_hidden(
    runner: Qwen35ParoNextTokenRunner,
    prompts: list[list[int]],
    *,
    layer_limit: int,
    decode_tokens: int,
    max_sequence_length: int,
    compiler_version: str | None,
    require_cached_build: bool,
) -> HiddenRun:
    rows = len(prompts)
    seed_tokens: list[int] = []
    generated_tokens: list[list[int]] = []
    prefill_hidden_bits = np.empty((rows, runner.config.hidden_size), dtype=np.uint16)
    prefill_linear_state_rows: dict[int, dict[str, list[np.ndarray]]] = {}
    prefill_linear_input_rows: dict[int, list[np.ndarray]] = {}
    prefill_full_kv_prefix_hash_rows: dict[int, dict[str, list[np.ndarray]]] = {}
    decode_linear_input_rows_by_step: list[dict[int, list[np.ndarray]]] = [{} for _ in range(decode_tokens)]
    decode_full_attention_rows_by_step: list[dict[int, dict[str, list[np.ndarray]]]] = [{} for _ in range(decode_tokens)]
    decode_full_context_oracle_rows_by_step: list[dict[int, dict[str, list[np.ndarray]]]] = [
        {} for _ in range(decode_tokens)
    ]
    decode_full_kv_sample_rows_by_step: list[dict[int, dict[str, list[np.ndarray] | tuple[str, ...]]]] = [
        {} for _ in range(decode_tokens)
    ]
    decode_linear_state_rows_by_step: list[dict[int, dict[str, list[np.ndarray]]]] = [{} for _ in range(decode_tokens)]
    hidden_by_step = [np.empty((rows, runner.config.hidden_size), dtype=np.uint16) for _ in range(decode_tokens)]
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=layer_limit,
        max_batch_size=1,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
    ) as session:
        for row, prompt in enumerate(prompts):
            session._prefill_linear_input_trace = []
            result = session.prefill_native(prompt, sample=True)
            if result is None:
                raise RuntimeError("c=1 prefill did not produce a seed token")
            next_token = int(result.token_id)
            seed_tokens.append(next_token)
            session.runtime.device_synchronize()
            prefill_hidden_bits[row : row + 1] = _copy_hidden_bits(session, session.hidden, rows=1)
            _merge_prefill_linear_state_row(
                prefill_linear_state_rows,
                _copy_prefill_linear_states(session, rows=1),
            )
            _merge_prefill_linear_input_rows(
                prefill_linear_input_rows,
                _prefill_linear_input_rows_from_trace(
                    getattr(session, "_prefill_linear_input_trace", None),
                    prompt_lengths=[len(prompt)],
                ),
            )
            _merge_full_kv_prefix_hash_rows(
                prefill_full_kv_prefix_hash_rows,
                _copy_full_kv_prefix_hashes(session, rows=1, prompt_lengths=[len(prompt)], slots=(0,)),
            )
            row_generated: list[int] = []
            for step in range(decode_tokens):
                position = len(prompt) + step
                session._decode_linear_input_trace = []
                session._decode_full_attention_trace = []
                session._set_token_embedding(next_token, stream=0)
                session._set_position(position, stream=0)
                hidden = session._run_layers(position=position, stream=0)
                session.runtime.device_synchronize()
                hidden_by_step[step][row : row + 1] = _copy_hidden_bits(session, hidden, rows=1)
                _merge_decode_linear_input_row(
                    decode_linear_input_rows_by_step[step],
                    _decode_linear_input_layers_from_trace(getattr(session, "_decode_linear_input_trace", None)),
                )
                decode_full_attention_layers = _decode_full_attention_layers_from_trace(
                    getattr(session, "_decode_full_attention_trace", None)
                )
                _merge_decode_full_attention_rows(
                    decode_full_attention_rows_by_step[step],
                    decode_full_attention_layers,
                )
                _merge_decode_full_context_oracle_rows(
                    decode_full_context_oracle_rows_by_step[step],
                    _decode_full_context_oracles_from_trace(
                        session,
                        decode_full_attention_layers,
                        rows=1,
                        positions=(position,),
                        slots=(0,),
                    ),
                )
                _merge_decode_full_kv_sample_rows(
                    decode_full_kv_sample_rows_by_step[step],
                    _copy_decode_full_kv_samples(session, rows=1, positions=(position,), slots=(0,)),
                )
                _merge_prefill_linear_state_row(
                    decode_linear_state_rows_by_step[step],
                    _copy_prefill_linear_states(session, rows=1),
                )
                step_result = session._sample_from_hidden(hidden)
                next_token = int(step_result.token_id)
                row_generated.append(next_token)
            generated_tokens.append(row_generated)
            session.reset()
    return HiddenRun(
        seed_tokens=seed_tokens,
        generated_tokens=generated_tokens,
        hidden_bits_by_step=hidden_by_step,
        prefill_hidden_bits=prefill_hidden_bits,
        prefill_linear_states=_stack_prefill_linear_state_rows(prefill_linear_state_rows),
        prefill_linear_inputs=prefill_linear_input_rows,
        prefill_full_kv_prefix_hashes=_stack_full_kv_prefix_hash_rows(prefill_full_kv_prefix_hash_rows),
        decode_linear_inputs_by_step=[
            _stack_decode_linear_input_rows(rows_by_layer) for rows_by_layer in decode_linear_input_rows_by_step
        ],
        decode_full_attention_by_step=[
            _stack_decode_full_attention_rows(rows_by_layer) for rows_by_layer in decode_full_attention_rows_by_step
        ],
        decode_full_context_oracles_by_step=[
            _stack_decode_full_context_oracle_rows(rows_by_layer)
            for rows_by_layer in decode_full_context_oracle_rows_by_step
        ],
        decode_full_kv_samples_by_step=[
            _stack_decode_full_kv_sample_rows(rows_by_layer) for rows_by_layer in decode_full_kv_sample_rows_by_step
        ],
        decode_linear_states_by_step=[
            _stack_prefill_linear_state_rows(rows_by_layer) for rows_by_layer in decode_linear_state_rows_by_step
        ],
    )


def _failure_modes(*, hidden_passed: bool, token_passed: bool) -> list[str]:
    modes: list[str] = []
    if not hidden_passed:
        modes.append("hidden")
    if not token_passed:
        modes.append("token")
    return modes


def _token_mismatches(batch: HiddenRun, c1: HiddenRun) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for row, (batch_seed, c1_seed) in enumerate(zip(batch.seed_tokens, c1.seed_tokens, strict=True)):
        batch_sequence = [int(batch_seed), *[int(token) for token in batch.generated_tokens[row]]]
        c1_sequence = [int(c1_seed), *[int(token) for token in c1.generated_tokens[row]]]
        if batch_sequence != c1_sequence:
            first_index = next(
                (idx for idx, (left, right) in enumerate(zip(batch_sequence, c1_sequence, strict=False)) if left != right),
                min(len(batch_sequence), len(c1_sequence)),
            )
            mismatches.append(
                {
                    "row": row,
                    "first_index": int(first_index),
                    "batch": batch_sequence,
                    "c1": c1_sequence,
                }
            )
    return mismatches


def _layer_limit_metadata(layer_limit: int, layer_types: Sequence[str] | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"last_layer_index": int(layer_limit) - 1}
    if layer_types is not None and 0 <= metadata["last_layer_index"] < len(layer_types):
        metadata["last_layer_type"] = str(layer_types[metadata["last_layer_index"]])
    return metadata


def _prefill_summary(
    batch: HiddenRun,
    c1: HiddenRun,
    *,
    atol: float,
    focus_hidden_flat_indices: Sequence[int] = (),
) -> dict[str, Any] | None:
    if batch.prefill_hidden_bits is None or c1.prefill_hidden_bits is None:
        return None
    rows: list[dict[str, Any]] = []
    for row in range(batch.prefill_hidden_bits.shape[0]):
        rows.append(
            {
                "row": row,
                "hidden_comparison": hidden_comparison(
                    batch.prefill_hidden_bits[row : row + 1],
                    c1.prefill_hidden_bits[row : row + 1],
                    atol=atol,
                    selected_flat_indices=focus_hidden_flat_indices,
                ),
            }
        )
    summary: dict[str, Any] = {
        "stage": "prefill_final_hidden",
        "hidden_passed": all(row["hidden_comparison"]["passed"] for row in rows),
        "rows": rows,
    }
    if batch.prefill_execution is not None:
        summary["batch_prefill_execution"] = batch.prefill_execution
    return summary


def _prefill_full_kv_prefix_failure_record(
    layer: dict[str, Any],
    row_summary: dict[str, Any],
    kind: str,
    *,
    layer_limit: int | None = None,
) -> dict[str, Any]:
    comparison = row_summary.get(f"{kind}_prefix_hash_comparison", {})
    first = comparison.get("first_mismatch") if isinstance(comparison, dict) else None
    record: dict[str, Any] = {
        "layer_index": int(layer.get("layer_index", -1)),
        "row": int(row_summary.get("row", -1)),
        "kind": str(kind),
        "context_len": int(row_summary.get("context_len", 0)),
        "context_len_match": bool(row_summary.get("context_len_match", False)),
        "mismatch_count": int(comparison.get("mismatch_count", 0)) if isinstance(comparison, dict) else 0,
        "first_mismatch_position": None if not isinstance(first, dict) else int(first.get("position", -1)),
        "batch_hash": None if not isinstance(first, dict) else int(first.get("batch_hash", 0)),
        "c1_hash": None if not isinstance(first, dict) else int(first.get("c1_hash", 0)),
    }
    if layer_limit is not None:
        record = {"layer_limit": int(layer_limit), **record}
    return record


def _prefill_full_kv_prefix_failure_summary(layers: Sequence[dict[str, Any]]) -> dict[str, Any]:
    kind_summaries: dict[str, Any] = {}
    first_failure: dict[str, Any] | None = None
    for kind in ("key", "value"):
        rows: list[int] = []
        seen_rows: set[int] = set()
        kind_first_failure: dict[str, Any] | None = None
        for layer in layers:
            for row_summary in layer.get("rows", []):
                comparison = row_summary.get(f"{kind}_prefix_hash_comparison")
                if not isinstance(comparison, dict) or bool(comparison.get("passed", True)):
                    continue
                record = _prefill_full_kv_prefix_failure_record(layer, row_summary, kind)
                if kind_first_failure is None:
                    kind_first_failure = record
                if first_failure is None:
                    first_failure = record
                row_index = int(row_summary.get("row", -1))
                if row_index >= 0 and row_index not in seen_rows:
                    rows.append(row_index)
                    seen_rows.add(row_index)
        kind_summaries[kind] = {
            "passed": not rows,
            "failure_rows": rows,
            "failure_row_count": len(rows),
            "first_failure": kind_first_failure,
        }
    failed_kinds = [kind for kind in ("key", "value") if kind_summaries[kind]["failure_rows"]]
    return {
        "failed_kinds": failed_kinds,
        "failed_kind_count": len(failed_kinds),
        "first_failure": first_failure,
        "kinds": kind_summaries,
    }


def _prefill_full_kv_prefix_summary(batch: HiddenRun, c1: HiddenRun) -> dict[str, Any] | None:
    if not batch.prefill_full_kv_prefix_hashes or not c1.prefill_full_kv_prefix_hashes:
        return None
    layers: list[dict[str, Any]] = []
    for layer_id in sorted(set(batch.prefill_full_kv_prefix_hashes) & set(c1.prefill_full_kv_prefix_hashes)):
        batch_payload = batch.prefill_full_kv_prefix_hashes[layer_id]
        c1_payload = c1.prefill_full_kv_prefix_hashes[layer_id]
        batch_lens = np.asarray(batch_payload["context_lens"], dtype=np.int64)
        c1_lens = np.asarray(c1_payload["context_lens"], dtype=np.int64)
        batch_key_hashes = np.asarray(batch_payload["key_prefix_hashes"], dtype=np.uint64)
        c1_key_hashes = np.asarray(c1_payload["key_prefix_hashes"], dtype=np.uint64)
        batch_value_hashes = np.asarray(batch_payload["value_prefix_hashes"], dtype=np.uint64)
        c1_value_hashes = np.asarray(c1_payload["value_prefix_hashes"], dtype=np.uint64)
        if batch_lens.shape != c1_lens.shape:
            raise ValueError(f"prefill full-KV context-lens shape differs for layer {layer_id}")
        for name, left, right in (
            ("key", batch_key_hashes, c1_key_hashes),
            ("value", batch_value_hashes, c1_value_hashes),
        ):
            if left.ndim != 2 or right.ndim != 2 or int(left.shape[0]) != int(batch_lens.shape[0]) or int(right.shape[0]) != int(batch_lens.shape[0]):
                raise ValueError(f"prefill full-KV {name} prefix hashes have incompatible shapes")
        row_summaries: list[dict[str, Any]] = []
        for row in range(int(batch_lens.shape[0])):
            key_comparison = _kv_prefix_hash_comparison(
                batch_key_hashes[row],
                c1_key_hashes[row],
                context_len=int(batch_lens[row]),
            )
            value_comparison = _kv_prefix_hash_comparison(
                batch_value_hashes[row],
                c1_value_hashes[row],
                context_len=int(batch_lens[row]),
            )
            row_summaries.append(
                {
                    "row": int(row),
                    "context_len": int(batch_lens[row]),
                    "context_len_match": bool(batch_lens[row] == c1_lens[row]),
                    "passed": bool(batch_lens[row] == c1_lens[row])
                    and bool(key_comparison["passed"])
                    and bool(value_comparison["passed"]),
                    "key_prefix_hash_comparison": key_comparison,
                    "value_prefix_hash_comparison": value_comparison,
                }
            )
        layers.append({"layer_index": int(layer_id), "passed": all(row["passed"] for row in row_summaries), "rows": row_summaries})
    failure_summary = _prefill_full_kv_prefix_failure_summary(layers)
    return {
        "stage": "prefill_full_kv_prefix_hashes",
        "passed": all(layer["passed"] for layer in layers),
        "failure_summary": failure_summary,
        "layers": layers,
    }


def _prefill_full_kv_prefix_rollup(layer_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    kind_summaries: dict[str, Any] = {}
    first_failure: dict[str, Any] | None = None
    for kind in ("key", "value"):
        rows: list[int] = []
        seen_rows: set[int] = set()
        kind_first_failure: dict[str, Any] | None = None
        for summary in layer_summaries:
            layer_limit = int(summary.get("layer_limit", 0))
            prefill = summary.get("prefill_full_kv_prefix_hashes")
            if not isinstance(prefill, dict):
                continue
            kind_summary = prefill.get("failure_summary", {}).get("kinds", {}).get(kind)
            if not isinstance(kind_summary, dict):
                continue
            failure = kind_summary.get("first_failure")
            if isinstance(failure, dict):
                failure_with_limit = {"layer_limit": layer_limit, **failure}
                if kind_first_failure is None:
                    kind_first_failure = failure_with_limit
                if first_failure is None:
                    first_failure = failure_with_limit
            for raw_row in kind_summary.get("failure_rows", []):
                row_index = int(raw_row)
                if row_index in seen_rows:
                    continue
                rows.append(row_index)
                seen_rows.add(row_index)
        kind_summaries[kind] = {
            "passed": not rows,
            "failure_rows": rows,
            "failure_row_count": len(rows),
            "first_failure": kind_first_failure,
        }
    failed_kinds = [kind for kind in ("key", "value") if kind_summaries[kind]["failure_rows"]]
    return {
        "failed_kinds": failed_kinds,
        "failed_kind_count": len(failed_kinds),
        "first_failure": first_failure,
        "kinds": kind_summaries,
        "layer_limits": [
            {
                "layer_limit": int(summary.get("layer_limit", 0)),
                "passed": bool(summary.get("prefill_full_kv_prefix_hashes", {}).get("passed", True))
                if isinstance(summary.get("prefill_full_kv_prefix_hashes"), dict)
                else True,
            }
            for summary in layer_summaries
            if isinstance(summary.get("prefill_full_kv_prefix_hashes"), dict)
        ],
    }


def _linear_state_layers_summary(
    batch_states: dict[int, dict[str, np.ndarray]],
    c1_states: dict[int, dict[str, np.ndarray]],
    *,
    atol: float,
) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    for layer_id in sorted(set(batch_states) & set(c1_states)):
        layer_batch = batch_states[layer_id]
        layer_c1 = c1_states[layer_id]
        state_summaries: dict[str, Any] = {}
        for state_name in ("conv", "recurrent"):
            if state_name not in layer_batch or state_name not in layer_c1:
                continue
            state_summary = numeric_comparison(layer_batch[state_name], layer_c1[state_name], atol=atol)
            state_summary["row_summaries"] = _numeric_row_summaries(layer_batch[state_name], layer_c1[state_name], atol=atol)
            state_summaries[state_name] = state_summary
        layers.append(
            {
                "layer_index": int(layer_id),
                "passed": all(summary["passed"] for summary in state_summaries.values()),
                "states": state_summaries,
            }
        )
    return layers


def _prefill_linear_state_summary(
    batch: HiddenRun,
    c1: HiddenRun,
    *,
    atol: float,
) -> dict[str, Any] | None:
    if not batch.prefill_linear_states or not c1.prefill_linear_states:
        return None
    layers = _linear_state_layers_summary(batch.prefill_linear_states, c1.prefill_linear_states, atol=atol)
    return {
        "stage": "prefill_linear_states",
        "state_atol": float(atol),
        "passed": all(layer["passed"] for layer in layers),
        "layers": layers,
    }


def _decode_linear_input_summary(
    batch: HiddenRun,
    c1: HiddenRun,
    *,
    atol: float,
    focus_hidden_flat_indices: Sequence[int] = (),
) -> dict[str, Any] | None:
    if not batch.decode_linear_inputs_by_step or not c1.decode_linear_inputs_by_step:
        return None
    steps: list[dict[str, Any]] = []
    for step, (batch_layers, c1_layers) in enumerate(
        zip(batch.decode_linear_inputs_by_step, c1.decode_linear_inputs_by_step, strict=True)
    ):
        layers: list[dict[str, Any]] = []
        for layer_id in sorted(set(batch_layers) & set(c1_layers)):
            layer_batch = batch_layers[layer_id]
            layer_c1 = c1_layers[layer_id]
            if layer_batch.shape != layer_c1.shape:
                raise ValueError(
                    f"decode linear input trace shape differs for step {step}, layer {layer_id}: "
                    f"batch={layer_batch.shape} c1={layer_c1.shape}"
                )
            row_summaries: list[dict[str, Any]] = []
            for row in range(int(layer_batch.shape[0])):
                comparison = hidden_comparison(
                    layer_batch[row : row + 1],
                    layer_c1[row : row + 1],
                    atol=atol,
                    selected_flat_indices=focus_hidden_flat_indices,
                )
                row_summaries.append(
                    {
                        "row": int(row),
                        "hidden_comparison": comparison,
                        "passed": bool(comparison["passed"]),
                    }
                )
            layers.append(
                {
                    "layer_index": int(layer_id),
                    "passed": all(row["passed"] for row in row_summaries),
                    "rows": row_summaries,
                }
            )
        steps.append(
            {
                "decode_step": int(step),
                "generated_index": int(step + 1),
                "passed": all(layer["passed"] for layer in layers),
                "layers": layers,
            }
        )
    first_mismatch: dict[str, Any] | None = None
    for step_summary in steps:
        for layer in step_summary["layers"]:
            for row in layer["rows"]:
                if row["passed"]:
                    continue
                comparison = row["hidden_comparison"]
                first_mismatch = {
                    "decode_step": int(step_summary["decode_step"]),
                    "generated_index": int(step_summary["generated_index"]),
                    "layer_index": int(layer["layer_index"]),
                    "row": int(row["row"]),
                    "max_abs": float(comparison["max_abs"]),
                    "max_abs_flat_index": int(comparison["max_abs_flat_index"]),
                    "max_abs_index": comparison["max_abs_index"],
                    "elements_over_atol": int(comparison["elements_over_atol"]),
                }
                break
            if first_mismatch is not None:
                break
        if first_mismatch is not None:
            break
    worst_diff: dict[str, Any] | None = None
    for step_summary in steps:
        for layer in step_summary["layers"]:
            for row in layer["rows"]:
                comparison = row["hidden_comparison"]
                if worst_diff is not None and float(comparison["max_abs"]) <= float(worst_diff["max_abs"]):
                    continue
                max_abs_flat_index = comparison.get("max_abs_flat_index")
                worst_diff = {
                    "decode_step": int(step_summary["decode_step"]),
                    "generated_index": int(step_summary["generated_index"]),
                    "layer_index": int(layer["layer_index"]),
                    "row": int(row["row"]),
                    "passed": bool(row["passed"]),
                    "max_abs": float(comparison["max_abs"]),
                    "max_abs_flat_index": None if max_abs_flat_index is None else int(max_abs_flat_index),
                    "max_abs_index": comparison["max_abs_index"],
                    "elements_over_atol": int(comparison["elements_over_atol"]),
                }
    result = {
        "stage": "decode_linear_inputs",
        "hidden_atol": float(atol),
        "passed": all(step["passed"] for step in steps),
        "steps": steps,
    }
    if first_mismatch is not None:
        result["first_mismatch"] = first_mismatch
    if worst_diff is not None:
        result["worst_diff"] = worst_diff
    return result


def _decode_full_attention_failure_record(
    step_summary: dict[str, Any],
    layer: dict[str, Any],
    stage: str,
    row_summary: dict[str, Any],
    *,
    layer_limit: int | None = None,
) -> dict[str, Any]:
    comparison = row_summary.get("hidden_comparison", {})
    max_abs_flat_index = comparison.get("max_abs_flat_index")
    record = {
        "decode_step": int(step_summary.get("decode_step", 0)),
        "generated_index": int(step_summary.get("generated_index", int(step_summary.get("decode_step", 0)) + 1)),
        "layer_index": int(layer.get("layer_index", -1)),
        "stage": stage,
        "row": int(row_summary.get("row", -1)),
        "comparison_kind": str(row_summary.get("comparison_kind", "unknown")),
        "max_abs": float(comparison.get("max_abs", 0.0)),
        "max_abs_flat_index": None if max_abs_flat_index is None else int(max_abs_flat_index),
        "max_abs_index": comparison.get("max_abs_index", []),
        "elements_over_atol": int(comparison.get("elements_over_atol", 0)),
    }
    if layer_limit is not None:
        return {"layer_limit": int(layer_limit), **record}
    return record


def _decode_full_attention_stage_failure_summary(steps: Sequence[dict[str, Any]]) -> dict[str, Any]:
    stage_summaries: dict[str, Any] = {}
    for stage in DECODE_FULL_ATTENTION_TRACE_STAGES:
        rows: list[int] = []
        seen_rows: set[int] = set()
        first_failure: dict[str, Any] | None = None
        for step_summary in steps:
            for layer in step_summary.get("layers", []):
                stage_summary = layer.get("stages", {}).get(stage)
                if not isinstance(stage_summary, dict):
                    continue
                for row_summary in stage_summary.get("rows", []):
                    if bool(row_summary.get("passed", False)):
                        continue
                    row_index = int(row_summary.get("row", -1))
                    if first_failure is None:
                        first_failure = _decode_full_attention_failure_record(step_summary, layer, stage, row_summary)
                    if row_index < 0 or row_index in seen_rows:
                        continue
                    rows.append(row_index)
                    seen_rows.add(row_index)
        stage_summaries[stage] = {
            "passed": not rows,
            "failure_rows": rows,
            "failure_row_count": len(rows),
            "first_failure": first_failure,
        }
    first_failure: dict[str, Any] | None = None
    for step_summary in steps:
        for layer in step_summary.get("layers", []):
            for stage in DECODE_FULL_ATTENTION_TRACE_STAGES:
                stage_summary = layer.get("stages", {}).get(stage)
                if not isinstance(stage_summary, dict):
                    continue
                for row_summary in stage_summary.get("rows", []):
                    if bool(row_summary.get("passed", False)):
                        continue
                    first_failure = _decode_full_attention_failure_record(step_summary, layer, stage, row_summary)
                    break
                if first_failure is not None:
                    break
            if first_failure is not None:
                break
        if first_failure is not None:
            break
    failed_stages = [stage for stage in DECODE_FULL_ATTENTION_TRACE_STAGES if stage_summaries[stage]["failure_rows"]]
    return {
        "failed_stages": failed_stages,
        "failed_stage_count": len(failed_stages),
        "first_failure": first_failure,
        "stages": stage_summaries,
    }


def _decode_full_attention_stage_rollup(layer_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    stage_summaries: dict[str, Any] = {}
    for stage in DECODE_FULL_ATTENTION_TRACE_STAGES:
        rows: list[int] = []
        seen_rows: set[int] = set()
        comparison_kinds: list[str] = []
        seen_comparison_kinds: set[str] = set()
        first_failure: dict[str, Any] | None = None
        for summary in layer_summaries:
            layer_limit = int(summary.get("layer_limit", 0))
            trace = summary.get("decode_full_attention")
            if not isinstance(trace, dict):
                continue
            for step_summary in trace.get("steps", []):
                for layer in step_summary.get("layers", []):
                    stage_summary = layer.get("stages", {}).get(stage)
                    if not isinstance(stage_summary, dict):
                        continue
                    for row_summary in stage_summary.get("rows", []):
                        if bool(row_summary.get("passed", False)):
                            continue
                        record = _decode_full_attention_failure_record(
                            step_summary,
                            layer,
                            stage,
                            row_summary,
                            layer_limit=layer_limit,
                        )
                        if first_failure is None:
                            first_failure = record
                        row_index = int(row_summary.get("row", -1))
                        if row_index >= 0 and row_index not in seen_rows:
                            rows.append(row_index)
                            seen_rows.add(row_index)
                        comparison_kind = str(row_summary.get("comparison_kind", "unknown"))
                        if comparison_kind not in seen_comparison_kinds:
                            comparison_kinds.append(comparison_kind)
                            seen_comparison_kinds.add(comparison_kind)
        stage_summaries[stage] = {
            "passed": not rows,
            "failure_rows": rows,
            "failure_row_count": len(rows),
            "failure_comparison_kinds": comparison_kinds,
            "first_failure": first_failure,
        }

    first_failure: dict[str, Any] | None = None
    for summary in layer_summaries:
        layer_limit = int(summary.get("layer_limit", 0))
        trace = summary.get("decode_full_attention")
        if not isinstance(trace, dict):
            continue
        for step_summary in trace.get("steps", []):
            for layer in step_summary.get("layers", []):
                for stage in DECODE_FULL_ATTENTION_TRACE_STAGES:
                    stage_summary = layer.get("stages", {}).get(stage)
                    if not isinstance(stage_summary, dict):
                        continue
                    for row_summary in stage_summary.get("rows", []):
                        if bool(row_summary.get("passed", False)):
                            continue
                        first_failure = _decode_full_attention_failure_record(
                            step_summary,
                            layer,
                            stage,
                            row_summary,
                            layer_limit=layer_limit,
                        )
                        break
                    if first_failure is not None:
                        break
                if first_failure is not None:
                    break
            if first_failure is not None:
                break
        if first_failure is not None:
            break

    failed_stages = [stage for stage in DECODE_FULL_ATTENTION_TRACE_STAGES if stage_summaries[stage]["failure_rows"]]
    layer_limits: list[dict[str, Any]] = []
    for summary in layer_summaries:
        trace = summary.get("decode_full_attention")
        stage_failure_summary = trace.get("stage_failure_summary", {}) if isinstance(trace, dict) else {}
        failed_for_limit = stage_failure_summary.get("failed_stages", []) if isinstance(stage_failure_summary, dict) else []
        layer_limits.append(
            {
                "layer_limit": int(summary.get("layer_limit", 0)),
                "passed": True if not isinstance(trace, dict) else bool(trace.get("passed", True)),
                "failed_stages": [str(stage) for stage in failed_for_limit],
            }
        )
    return {
        "failed_stages": failed_stages,
        "failed_stage_count": len(failed_stages),
        "input_passed": bool(stage_summaries["input"]["passed"]),
        "output_passed": bool(stage_summaries["output"]["passed"]),
        "first_failure": first_failure,
        "stages": stage_summaries,
        "layer_limits": layer_limits,
    }


def _decode_full_attention_summary(
    batch: HiddenRun,
    c1: HiddenRun,
    *,
    atol: float,
    focus_hidden_flat_indices: Sequence[int] = (),
) -> dict[str, Any] | None:
    if not batch.decode_full_attention_by_step or not c1.decode_full_attention_by_step:
        return None
    if not any(step_layers for step_layers in batch.decode_full_attention_by_step) or not any(
        step_layers for step_layers in c1.decode_full_attention_by_step
    ):
        return None
    steps: list[dict[str, Any]] = []
    for step, (batch_layers, c1_layers) in enumerate(
        zip(batch.decode_full_attention_by_step, c1.decode_full_attention_by_step, strict=True)
    ):
        layers: list[dict[str, Any]] = []
        for layer_id in sorted(set(batch_layers) & set(c1_layers)):
            layer_batch = batch_layers[layer_id]
            layer_c1 = c1_layers[layer_id]
            stage_summaries: dict[str, Any] = {}
            for stage in DECODE_FULL_ATTENTION_TRACE_STAGES:
                if stage not in layer_batch or stage not in layer_c1:
                    continue
                stage_batch = layer_batch[stage]
                stage_c1 = layer_c1[stage]
                if stage_batch.shape != stage_c1.shape:
                    raise ValueError(
                        f"decode full-attention {stage} trace shape differs for step {step}, layer {layer_id}: "
                        f"batch={stage_batch.shape} c1={stage_c1.shape}"
                    )
                row_summaries: list[dict[str, Any]] = []
                for row in range(int(stage_batch.shape[0])):
                    if stage_batch.dtype == np.uint16 and stage_c1.dtype == np.uint16:
                        batch_row = stage_batch[row : row + 1]
                        c1_row = stage_c1[row : row + 1]
                        stage_focus_indices = [
                            int(flat_index)
                            for flat_index in focus_hidden_flat_indices
                            if 0 <= int(flat_index) < int(batch_row.size)
                        ]
                        comparison = hidden_comparison(
                            batch_row,
                            c1_row,
                            atol=atol,
                            selected_flat_indices=stage_focus_indices,
                        )
                        comparison_kind = "fp16_bits"
                    else:
                        comparison = numeric_comparison(
                            stage_batch[row : row + 1],
                            stage_c1[row : row + 1],
                            atol=atol,
                        )
                        comparison_kind = "fp32"
                    row_summaries.append(
                        {
                            "row": int(row),
                            "comparison_kind": comparison_kind,
                            "hidden_comparison": comparison,
                            "passed": bool(comparison["passed"]),
                        }
                    )
                stage_summaries[stage] = {
                    "passed": all(row["passed"] for row in row_summaries),
                    "rows": row_summaries,
                }
            stage_delta_summaries: dict[str, Any] = {}
            if all(stage in layer_batch and stage in layer_c1 for stage in ("o_proj", "output")):
                batch_output = _trace_array_to_f32(layer_batch["output"])
                batch_o_proj = _trace_array_to_f32(layer_batch["o_proj"])
                c1_output = _trace_array_to_f32(layer_c1["output"])
                c1_o_proj = _trace_array_to_f32(layer_c1["o_proj"])
                if batch_output.shape != batch_o_proj.shape or c1_output.shape != c1_o_proj.shape:
                    raise ValueError(
                        f"decode full-attention output/o_proj trace shape differs for step {step}, layer {layer_id}: "
                        f"batch_output={batch_output.shape} batch_o_proj={batch_o_proj.shape} "
                        f"c1_output={c1_output.shape} c1_o_proj={c1_o_proj.shape}"
                    )
                batch_delta = batch_output - batch_o_proj
                c1_delta = c1_output - c1_o_proj
                row_delta_summaries: list[dict[str, Any]] = []
                for row in range(int(batch_delta.shape[0])):
                    comparison = numeric_comparison(batch_delta[row : row + 1], c1_delta[row : row + 1], atol=atol)
                    row_delta_summaries.append(
                        {
                            "row": int(row),
                            "comparison_kind": "fp32_delta",
                            "delta_comparison": comparison,
                            "passed": bool(comparison["passed"]),
                        }
                    )
                stage_delta_summaries["output_minus_o_proj"] = {
                    "passed": all(row["passed"] for row in row_delta_summaries),
                    "rows": row_delta_summaries,
                }
            stage_oracle_summaries: dict[str, Any] = {}
            if all(stage in layer_batch and stage in layer_c1 for stage in ("residual", "mlp_input")):
                batch_residual = _trace_array_to_f32(layer_batch["residual"])
                batch_mlp_input = _trace_array_to_f32(layer_batch["mlp_input"])
                c1_residual = _trace_array_to_f32(layer_c1["residual"])
                c1_mlp_input = _trace_array_to_f32(layer_c1["mlp_input"])
                if batch_residual.shape != batch_mlp_input.shape or c1_residual.shape != c1_mlp_input.shape:
                    raise ValueError(
                        f"decode full-attention residual/mlp_input trace shape differs for step {step}, layer {layer_id}: "
                        f"batch_residual={batch_residual.shape} batch_mlp_input={batch_mlp_input.shape} "
                        f"c1_residual={c1_residual.shape} c1_mlp_input={c1_mlp_input.shape}"
                    )
                row_oracle_summaries: list[dict[str, Any]] = []
                for row in range(int(batch_residual.shape[0])):
                    comparison = _inferred_rmsnorm_oracle_comparison(
                        batch_residual=batch_residual[row : row + 1],
                        batch_mlp_input=batch_mlp_input[row : row + 1],
                        c1_residual=c1_residual[row : row + 1],
                        c1_mlp_input=c1_mlp_input[row : row + 1],
                        atol=atol,
                    )
                    row_oracle_summaries.append(
                        {
                            "row": int(row),
                            "comparison_kind": "fp32_inferred_rmsnorm_from_c1",
                            "oracle_comparison": comparison,
                            "passed": bool(comparison["passed"]),
                        }
                    )
                stage_oracle_summaries["mlp_input_from_residual_inferred_weight"] = {
                    "passed": all(row["passed"] for row in row_oracle_summaries),
                    "rows": row_oracle_summaries,
                }
            layer_summary = {
                "layer_index": int(layer_id),
                "passed": all(stage_summary["passed"] for stage_summary in stage_summaries.values()),
                "stages": stage_summaries,
            }
            if stage_delta_summaries:
                layer_summary["stage_deltas"] = stage_delta_summaries
            if stage_oracle_summaries:
                layer_summary["stage_oracles"] = stage_oracle_summaries
            layers.append(layer_summary)
        steps.append(
            {
                "decode_step": int(step),
                "generated_index": int(step + 1),
                "passed": all(layer["passed"] for layer in layers),
                "layers": layers,
            }
        )
    stage_passed = {
        stage: all(
            layer["stages"].get(stage, {"passed": True})["passed"]
            for step_summary in steps
            for layer in step_summary["layers"]
        )
        for stage in DECODE_FULL_ATTENTION_TRACE_STAGES
    }
    first_mismatch: dict[str, Any] | None = None
    for step_summary in steps:
        for layer in step_summary["layers"]:
            for stage in DECODE_FULL_ATTENTION_TRACE_STAGES:
                stage_summary = layer["stages"].get(stage)
                if stage_summary is None:
                    continue
                for row in stage_summary["rows"]:
                    if row["passed"]:
                        continue
                    comparison = row["hidden_comparison"]
                    first_mismatch = {
                        "decode_step": int(step_summary["decode_step"]),
                        "generated_index": int(step_summary["generated_index"]),
                        "layer_index": int(layer["layer_index"]),
                        "stage": stage,
                        "row": int(row["row"]),
                        "max_abs": float(comparison["max_abs"]),
                        "max_abs_flat_index": int(comparison["max_abs_flat_index"]),
                        "max_abs_index": comparison["max_abs_index"],
                        "elements_over_atol": int(comparison["elements_over_atol"]),
                    }
                    break
                if first_mismatch is not None:
                    break
            if first_mismatch is not None:
                break
        if first_mismatch is not None:
            break
    worst_diff: dict[str, Any] | None = None
    for step_summary in steps:
        for layer in step_summary["layers"]:
            for stage in DECODE_FULL_ATTENTION_TRACE_STAGES:
                stage_summary = layer["stages"].get(stage)
                if stage_summary is None:
                    continue
                for row in stage_summary["rows"]:
                    comparison = row["hidden_comparison"]
                    if worst_diff is not None and float(comparison["max_abs"]) <= float(worst_diff["max_abs"]):
                        continue
                    max_abs_flat_index = comparison.get("max_abs_flat_index")
                    worst_diff = {
                        "decode_step": int(step_summary["decode_step"]),
                        "generated_index": int(step_summary["generated_index"]),
                        "layer_index": int(layer["layer_index"]),
                        "stage": stage,
                        "row": int(row["row"]),
                        "passed": bool(row["passed"]),
                        "max_abs": float(comparison["max_abs"]),
                        "max_abs_flat_index": None if max_abs_flat_index is None else int(max_abs_flat_index),
                        "max_abs_index": comparison["max_abs_index"],
                        "elements_over_atol": int(comparison["elements_over_atol"]),
                    }
    result = {
        "stage": "decode_full_attention",
        "hidden_atol": float(atol),
        "input_passed": bool(stage_passed["input"]),
        "output_passed": bool(stage_passed["output"]),
        "stage_passed": {stage: bool(passed) for stage, passed in stage_passed.items()},
        "stage_failure_summary": _decode_full_attention_stage_failure_summary(steps),
        "passed": all(step["passed"] for step in steps),
        "steps": steps,
    }
    if first_mismatch is not None:
        result["first_mismatch"] = first_mismatch
    if worst_diff is not None:
        result["worst_diff"] = worst_diff
    return result


DECODE_FULL_CONTEXT_ORACLE_COMPARISONS = (
    "context_len_match",
    "batch_context_vs_numpy",
    "c1_context_vs_numpy",
    "batch_numpy_vs_c1_numpy",
)


def _decode_full_context_oracle_rollup(layer_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    comparison_summaries: dict[str, Any] = {}
    first_failure: dict[str, Any] | None = None
    for comparison_name in DECODE_FULL_CONTEXT_ORACLE_COMPARISONS:
        rows: list[int] = []
        seen_rows: set[int] = set()
        comparison_first_failure: dict[str, Any] | None = None
        for summary in layer_summaries:
            layer_limit = int(summary.get("layer_limit", 0))
            oracle = summary.get("decode_full_context_oracle")
            if not isinstance(oracle, dict):
                continue
            failure_summary = oracle.get("comparison_failure_summary")
            if not isinstance(failure_summary, dict):
                continue
            comparison_summary = failure_summary.get("comparisons", {}).get(comparison_name)
            if not isinstance(comparison_summary, dict):
                continue
            failure = comparison_summary.get("first_failure")
            if isinstance(failure, dict):
                failure_with_limit = {"layer_limit": layer_limit, **failure}
                if comparison_first_failure is None:
                    comparison_first_failure = failure_with_limit
                if first_failure is None:
                    first_failure = failure_with_limit
            for raw_row in comparison_summary.get("failure_rows", []):
                row_index = int(raw_row)
                if row_index in seen_rows:
                    continue
                rows.append(row_index)
                seen_rows.add(row_index)
        comparison_summaries[comparison_name] = {
            "passed": not rows,
            "failure_rows": rows,
            "failure_row_count": len(rows),
            "first_failure": comparison_first_failure,
        }
    failed_comparisons = [
        comparison_name
        for comparison_name in DECODE_FULL_CONTEXT_ORACLE_COMPARISONS
        if comparison_summaries[comparison_name]["failure_rows"]
    ]
    return {
        "failed_comparisons": failed_comparisons,
        "failed_comparison_count": len(failed_comparisons),
        "first_failure": first_failure,
        "comparisons": comparison_summaries,
        "layer_limits": [
            {
                "layer_limit": int(summary["layer_limit"]),
                "passed": bool(summary.get("decode_full_context_oracle_passed", True)),
                "failed_comparisons": list(
                    summary.get("decode_full_context_oracle", {})
                    .get("comparison_failure_summary", {})
                    .get("failed_comparisons", [])
                ),
            }
            for summary in layer_summaries
            if isinstance(summary.get("decode_full_context_oracle"), dict)
        ],
    }


def _kv_prefix_hash_comparison(batch_hashes: np.ndarray, c1_hashes: np.ndarray, *, context_len: int) -> dict[str, Any]:
    live = int(context_len)
    batch_row = np.asarray(batch_hashes, dtype=np.uint64).reshape(-1)
    c1_row = np.asarray(c1_hashes, dtype=np.uint64).reshape(-1)
    if live < 0 or live > int(batch_row.shape[0]) or live > int(c1_row.shape[0]):
        raise ValueError("KV prefix hash comparison context_len exceeds hash row width")
    mismatches = np.flatnonzero(batch_row[:live] != c1_row[:live])
    first_mismatch: dict[str, Any] | None = None
    if mismatches.size:
        pos = int(mismatches[0])
        first_mismatch = {
            "position": pos,
            "batch_hash": int(batch_row[pos]),
            "c1_hash": int(c1_row[pos]),
        }
    return {
        "passed": bool(mismatches.size == 0),
        "context_len": live,
        "mismatch_count": int(mismatches.size),
        "first_mismatch": first_mismatch,
    }


def _decode_full_context_kv_prefix_failure_record(
    step_summary: dict[str, Any],
    layer: dict[str, Any],
    row_summary: dict[str, Any],
    kind: str,
    *,
    layer_limit: int | None = None,
) -> dict[str, Any]:
    comparison = row_summary.get(f"{kind}_prefix_hash_comparison", {})
    first = comparison.get("first_mismatch") if isinstance(comparison, dict) else None
    record: dict[str, Any] = {
        "decode_step": int(step_summary.get("decode_step", 0)),
        "generated_index": int(step_summary.get("generated_index", int(step_summary.get("decode_step", 0)) + 1)),
        "layer_index": int(layer.get("layer_index", -1)),
        "row": int(row_summary.get("row", -1)),
        "kind": str(kind),
        "context_len": int(row_summary.get("context_len", 0)),
        "mismatch_count": int(comparison.get("mismatch_count", 0)) if isinstance(comparison, dict) else 0,
        "first_mismatch_position": None if not isinstance(first, dict) else int(first.get("position", -1)),
        "batch_hash": None if not isinstance(first, dict) else int(first.get("batch_hash", 0)),
        "c1_hash": None if not isinstance(first, dict) else int(first.get("c1_hash", 0)),
    }
    if layer_limit is not None:
        record = {"layer_limit": int(layer_limit), **record}
    return record


def _decode_full_context_kv_prefix_failure_summary(steps: Sequence[dict[str, Any]]) -> dict[str, Any]:
    kind_summaries: dict[str, Any] = {}
    first_failure: dict[str, Any] | None = None
    for kind in ("key", "value"):
        rows: list[int] = []
        seen_rows: set[int] = set()
        kind_first_failure: dict[str, Any] | None = None
        for step_summary in steps:
            for layer in step_summary.get("layers", []):
                for row_summary in layer.get("rows", []):
                    comparison = row_summary.get(f"{kind}_prefix_hash_comparison")
                    if not isinstance(comparison, dict) or bool(comparison.get("passed", True)):
                        continue
                    record = _decode_full_context_kv_prefix_failure_record(step_summary, layer, row_summary, kind)
                    if kind_first_failure is None:
                        kind_first_failure = record
                    if first_failure is None:
                        first_failure = record
                    row_index = int(row_summary.get("row", -1))
                    if row_index >= 0 and row_index not in seen_rows:
                        rows.append(row_index)
                        seen_rows.add(row_index)
        kind_summaries[kind] = {
            "passed": not rows,
            "failure_rows": rows,
            "failure_row_count": len(rows),
            "first_failure": kind_first_failure,
        }
    failed_kinds = [kind for kind in ("key", "value") if kind_summaries[kind]["failure_rows"]]
    return {
        "failed_kinds": failed_kinds,
        "failed_kind_count": len(failed_kinds),
        "first_failure": first_failure,
        "kinds": kind_summaries,
    }


def _decode_full_context_kv_prefix_rollup(layer_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    kind_summaries: dict[str, Any] = {}
    first_failure: dict[str, Any] | None = None
    for kind in ("key", "value"):
        rows: list[int] = []
        seen_rows: set[int] = set()
        kind_first_failure: dict[str, Any] | None = None
        for summary in layer_summaries:
            layer_limit = int(summary.get("layer_limit", 0))
            oracle = summary.get("decode_full_context_oracle")
            if not isinstance(oracle, dict):
                continue
            prefix_summary = oracle.get("kv_prefix_failure_summary")
            if not isinstance(prefix_summary, dict):
                continue
            kind_summary = prefix_summary.get("kinds", {}).get(kind)
            if not isinstance(kind_summary, dict):
                continue
            failure = kind_summary.get("first_failure")
            if isinstance(failure, dict):
                failure_with_limit = {"layer_limit": layer_limit, **failure}
                if kind_first_failure is None:
                    kind_first_failure = failure_with_limit
                if first_failure is None:
                    first_failure = failure_with_limit
            for raw_row in kind_summary.get("failure_rows", []):
                row_index = int(raw_row)
                if row_index in seen_rows:
                    continue
                rows.append(row_index)
                seen_rows.add(row_index)
        kind_summaries[kind] = {
            "passed": not rows,
            "failure_rows": rows,
            "failure_row_count": len(rows),
            "first_failure": kind_first_failure,
        }
    failed_kinds = [kind for kind in ("key", "value") if kind_summaries[kind]["failure_rows"]]
    return {
        "failed_kinds": failed_kinds,
        "failed_kind_count": len(failed_kinds),
        "first_failure": first_failure,
        "kinds": kind_summaries,
        "layer_limits": [
            {
                "layer_limit": int(summary.get("layer_limit", 0)),
                "passed": bool(
                    summary.get("decode_full_context_oracle", {})
                    .get("kv_prefix_failure_summary", {})
                    .get("failed_kind_count", 0)
                    == 0
                )
                if isinstance(summary.get("decode_full_context_oracle"), dict)
                else True,
            }
            for summary in layer_summaries
            if isinstance(summary.get("decode_full_context_oracle"), dict)
        ],
    }


def _decode_full_context_oracle_failure_summary(steps: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def _failure_record(
        step_summary: dict[str, Any],
        layer: dict[str, Any],
        row_summary: dict[str, Any],
        comparison_name: str,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "decode_step": int(step_summary.get("decode_step", 0)),
            "generated_index": int(step_summary.get("generated_index", int(step_summary.get("decode_step", 0)) + 1)),
            "layer_index": int(layer.get("layer_index", -1)),
            "row": int(row_summary.get("row", -1)),
            "comparison": comparison_name,
            "context_len": int(row_summary.get("context_len", 0)),
            "context_len_match": bool(row_summary.get("context_len_match", False)),
        }
        comparison = row_summary.get(comparison_name)
        if isinstance(comparison, dict):
            max_abs_flat_index = comparison.get("max_abs_flat_index")
            record.update(
                {
                    "max_abs": float(comparison.get("max_abs", 0.0)),
                    "max_abs_flat_index": None if max_abs_flat_index is None else int(max_abs_flat_index),
                    "max_abs_index": comparison.get("max_abs_index", []),
                    "elements_over_atol": int(comparison.get("elements_over_atol", 0)),
                }
            )
        return record

    comparison_summaries: dict[str, Any] = {}
    first_failure: dict[str, Any] | None = None
    for comparison_name in DECODE_FULL_CONTEXT_ORACLE_COMPARISONS:
        rows: list[int] = []
        seen_rows: set[int] = set()
        comparison_first_failure: dict[str, Any] | None = None
        for step_summary in steps:
            for layer in step_summary.get("layers", []):
                for row_summary in layer.get("rows", []):
                    if comparison_name == "context_len_match":
                        failed = not bool(row_summary.get("context_len_match", False))
                    else:
                        comparison = row_summary.get(comparison_name)
                        failed = isinstance(comparison, dict) and not bool(comparison.get("passed", False))
                    if not failed:
                        continue
                    record = _failure_record(step_summary, layer, row_summary, comparison_name)
                    if comparison_first_failure is None:
                        comparison_first_failure = record
                    if first_failure is None:
                        first_failure = record
                    row_index = int(row_summary.get("row", -1))
                    if row_index < 0 or row_index in seen_rows:
                        continue
                    rows.append(row_index)
                    seen_rows.add(row_index)
        comparison_summaries[comparison_name] = {
            "passed": not rows,
            "failure_rows": rows,
            "failure_row_count": len(rows),
            "first_failure": comparison_first_failure,
        }
    failed_comparisons = [
        comparison_name
        for comparison_name in DECODE_FULL_CONTEXT_ORACLE_COMPARISONS
        if comparison_summaries[comparison_name]["failure_rows"]
    ]
    return {
        "failed_comparisons": failed_comparisons,
        "failed_comparison_count": len(failed_comparisons),
        "first_failure": first_failure,
        "comparisons": comparison_summaries,
    }


def _decode_full_context_oracle_summary(
    batch: HiddenRun,
    c1: HiddenRun,
    *,
    atol: float = DECODE_FULL_CONTEXT_ORACLE_ATOL,
) -> dict[str, Any] | None:
    if not batch.decode_full_context_oracles_by_step or not c1.decode_full_context_oracles_by_step:
        return None
    if not any(step_layers for step_layers in batch.decode_full_context_oracles_by_step) or not any(
        step_layers for step_layers in c1.decode_full_context_oracles_by_step
    ):
        return None
    oracle_atol = max(float(atol), float(DECODE_FULL_CONTEXT_ORACLE_ATOL))
    steps: list[dict[str, Any]] = []
    first_mismatch: dict[str, Any] | None = None
    worst_diff: dict[str, Any] | None = None
    for step, (batch_oracle_layers, c1_oracle_layers) in enumerate(
        zip(batch.decode_full_context_oracles_by_step, c1.decode_full_context_oracles_by_step, strict=True)
    ):
        batch_trace_layers = batch.decode_full_attention_by_step[step] if step < len(batch.decode_full_attention_by_step) else {}
        c1_trace_layers = c1.decode_full_attention_by_step[step] if step < len(c1.decode_full_attention_by_step) else {}
        layers: list[dict[str, Any]] = []
        for layer_id in sorted(set(batch_oracle_layers) & set(c1_oracle_layers)):
            if layer_id not in batch_trace_layers or layer_id not in c1_trace_layers:
                continue
            if "attn_context" not in batch_trace_layers[layer_id] or "attn_context" not in c1_trace_layers[layer_id]:
                continue
            batch_payload = batch_oracle_layers[layer_id]
            c1_payload = c1_oracle_layers[layer_id]
            batch_context = np.asarray(batch_trace_layers[layer_id]["attn_context"], dtype=np.float32)
            c1_context = np.asarray(c1_trace_layers[layer_id]["attn_context"], dtype=np.float32)
            batch_oracle = np.asarray(batch_payload["context"], dtype=np.float32)
            c1_oracle = np.asarray(c1_payload["context"], dtype=np.float32)
            batch_lens = np.asarray(batch_payload["context_lens"], dtype=np.int64)
            c1_lens = np.asarray(c1_payload["context_lens"], dtype=np.int64)
            batch_query_source = str(batch_payload.get("query_source", "query"))
            c1_query_source = str(c1_payload.get("query_source", "query"))
            batch_key_hashes = (
                np.asarray(batch_payload["key_prefix_hashes"], dtype=np.uint64)
                if "key_prefix_hashes" in batch_payload and "key_prefix_hashes" in c1_payload
                else None
            )
            c1_key_hashes = (
                np.asarray(c1_payload["key_prefix_hashes"], dtype=np.uint64)
                if batch_key_hashes is not None
                else None
            )
            batch_value_hashes = (
                np.asarray(batch_payload["value_prefix_hashes"], dtype=np.uint64)
                if "value_prefix_hashes" in batch_payload and "value_prefix_hashes" in c1_payload
                else None
            )
            c1_value_hashes = (
                np.asarray(c1_payload["value_prefix_hashes"], dtype=np.uint64)
                if batch_value_hashes is not None
                else None
            )
            if batch_context.shape != batch_oracle.shape or c1_context.shape != c1_oracle.shape:
                raise ValueError(
                    f"decode full-context oracle shape differs for step {step}, layer {layer_id}: "
                    f"batch_context={batch_context.shape} batch_oracle={batch_oracle.shape} "
                    f"c1_context={c1_context.shape} c1_oracle={c1_oracle.shape}"
                )
            if batch_oracle.shape != c1_oracle.shape or batch_lens.shape != c1_lens.shape:
                raise ValueError(
                    f"decode full-context oracle c>N/c1 shape differs for step {step}, layer {layer_id}: "
                    f"batch={batch_oracle.shape}/{batch_lens.shape} c1={c1_oracle.shape}/{c1_lens.shape}"
                )
            for name, left, right in (
                ("key", batch_key_hashes, c1_key_hashes),
                ("value", batch_value_hashes, c1_value_hashes),
            ):
                if left is None or right is None:
                    continue
                if left.ndim != 2 or right.ndim != 2 or int(left.shape[0]) != int(batch_oracle.shape[0]) or int(right.shape[0]) != int(batch_oracle.shape[0]):
                    raise ValueError(f"decode full-context {name} prefix hashes have incompatible shapes")
            row_summaries: list[dict[str, Any]] = []
            for row in range(int(batch_oracle.shape[0])):
                comparisons = {
                    "batch_context_vs_numpy": numeric_comparison(
                        batch_context[row : row + 1],
                        batch_oracle[row : row + 1],
                        atol=oracle_atol,
                    ),
                    "c1_context_vs_numpy": numeric_comparison(
                        c1_context[row : row + 1],
                        c1_oracle[row : row + 1],
                        atol=oracle_atol,
                    ),
                    "batch_numpy_vs_c1_numpy": numeric_comparison(
                        batch_oracle[row : row + 1],
                        c1_oracle[row : row + 1],
                        atol=oracle_atol,
                    ),
                }
                prefix_comparisons: dict[str, Any] = {}
                if batch_key_hashes is not None and c1_key_hashes is not None:
                    prefix_comparisons["key_prefix_hash_comparison"] = _kv_prefix_hash_comparison(
                        batch_key_hashes[row],
                        c1_key_hashes[row],
                        context_len=int(batch_lens[row]),
                    )
                if batch_value_hashes is not None and c1_value_hashes is not None:
                    prefix_comparisons["value_prefix_hash_comparison"] = _kv_prefix_hash_comparison(
                        batch_value_hashes[row],
                        c1_value_hashes[row],
                        context_len=int(batch_lens[row]),
                    )
                row_summary = {
                    "row": int(row),
                    "context_len": int(batch_lens[row]),
                    "context_len_match": bool(batch_lens[row] == c1_lens[row]),
                    "query_source": {"batch": batch_query_source, "c1": c1_query_source},
                    "kv_prefix_passed": all(bool(comparison["passed"]) for comparison in prefix_comparisons.values()),
                    "passed": bool(batch_lens[row] == c1_lens[row])
                    and all(bool(comparison["passed"]) for comparison in comparisons.values()),
                    **comparisons,
                    **prefix_comparisons,
                }
                row_summaries.append(row_summary)
                for comparison_name, comparison in comparisons.items():
                    mismatch = {
                        "decode_step": int(step),
                        "generated_index": int(step + 1),
                        "layer_index": int(layer_id),
                        "row": int(row),
                        "comparison": comparison_name,
                        "max_abs": float(comparison["max_abs"]),
                        "max_abs_flat_index": int(comparison["max_abs_flat_index"]),
                        "max_abs_index": comparison["max_abs_index"],
                        "elements_over_atol": int(comparison["elements_over_atol"]),
                    }
                    if first_mismatch is None and (not row_summary["context_len_match"] or not bool(comparison["passed"])):
                        first_mismatch = mismatch
                    if worst_diff is None or float(comparison["max_abs"]) > float(worst_diff["max_abs"]):
                        worst_diff = {**mismatch, "passed": bool(comparison["passed"])}
            layers.append(
                {
                    "layer_index": int(layer_id),
                    "query_source": {"batch": batch_query_source, "c1": c1_query_source},
                    "passed": all(row["passed"] for row in row_summaries),
                    "rows": row_summaries,
                }
            )
        steps.append(
            {
                "decode_step": int(step),
                "generated_index": int(step + 1),
                "passed": all(layer["passed"] for layer in layers),
                "layers": layers,
            }
        )
    kv_prefix_failure_summary = _decode_full_context_kv_prefix_failure_summary(steps)
    result = {
        "stage": "decode_full_context_oracle",
        "oracle": "numpy_softmax_bf16_kv",
        "oracle_atol": float(oracle_atol),
        "passed": all(step_summary["passed"] for step_summary in steps),
        "kv_prefix_passed": bool(kv_prefix_failure_summary["failed_kind_count"] == 0),
        "comparison_failure_summary": _decode_full_context_oracle_failure_summary(steps),
        "kv_prefix_failure_summary": kv_prefix_failure_summary,
        "steps": steps,
    }
    if first_mismatch is not None:
        result["first_mismatch"] = first_mismatch
    if worst_diff is not None:
        result["worst_diff"] = worst_diff
    return result


def _decode_full_kv_sample_comparison_record(
    step_summary: dict[str, Any],
    layer: dict[str, Any],
    row: dict[str, Any],
    kind: str,
    sample: dict[str, Any],
    *,
    layer_limit: int | None = None,
) -> dict[str, Any]:
    comparison = sample.get("comparison", {})
    max_abs_flat_index = comparison.get("max_abs_flat_index") if isinstance(comparison, dict) else None
    record: dict[str, Any] = {
        "decode_step": int(step_summary.get("decode_step", 0)),
        "generated_index": int(step_summary.get("generated_index", int(step_summary.get("decode_step", 0)) + 1)),
        "layer_index": int(layer.get("layer_index", -1)),
        "row": int(row.get("row", -1)),
        "kind": str(kind),
        "sample_index": int(sample.get("sample_index", -1)),
        "sample_label": str(sample.get("sample_label", "")),
        "sample_position": int(sample.get("sample_position", -1)),
        "sample_position_match": bool(sample.get("sample_position_match", False)),
        "max_abs": float(comparison.get("max_abs", 0.0)) if isinstance(comparison, dict) else 0.0,
        "max_abs_flat_index": None if max_abs_flat_index is None else int(max_abs_flat_index),
        "max_abs_index": comparison.get("max_abs_index", []) if isinstance(comparison, dict) else [],
        "elements_over_atol": int(comparison.get("elements_over_atol", 0)) if isinstance(comparison, dict) else 0,
        "bit_mismatch": int(comparison.get("bit_mismatch", 0)) if isinstance(comparison, dict) else 0,
    }
    if layer_limit is not None:
        record = {"layer_limit": int(layer_limit), **record}
    return record


def _decode_full_kv_sample_rollup(layer_summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    kind_summaries: dict[str, Any] = {}
    first_failure: dict[str, Any] | None = None
    for kind in ("key", "value"):
        rows: list[int] = []
        seen_rows: set[int] = set()
        sample_labels: list[str] = []
        seen_sample_labels: set[str] = set()
        kind_first_failure: dict[str, Any] | None = None
        for summary in layer_summaries:
            layer_limit = int(summary.get("layer_limit", 0))
            trace = summary.get("decode_full_kv_samples")
            if not isinstance(trace, dict):
                continue
            for step_summary in trace.get("steps", []):
                for layer in step_summary.get("layers", []):
                    for row_summary in layer.get("rows", []):
                        for sample in row_summary.get(f"{kind}_sample_comparisons", []):
                            comparison = sample.get("comparison") if isinstance(sample, dict) else None
                            if not isinstance(comparison, dict) or bool(comparison.get("passed", False)):
                                continue
                            record = _decode_full_kv_sample_comparison_record(
                                step_summary,
                                layer,
                                row_summary,
                                kind,
                                sample,
                                layer_limit=layer_limit,
                            )
                            if kind_first_failure is None:
                                kind_first_failure = record
                            if first_failure is None:
                                first_failure = record
                            row_index = int(row_summary.get("row", -1))
                            if row_index >= 0 and row_index not in seen_rows:
                                rows.append(row_index)
                                seen_rows.add(row_index)
                            sample_label = str(sample.get("sample_label", ""))
                            if sample_label and sample_label not in seen_sample_labels:
                                sample_labels.append(sample_label)
                                seen_sample_labels.add(sample_label)
        kind_summaries[kind] = {
            "passed": not rows,
            "failure_rows": rows,
            "failure_row_count": len(rows),
            "failed_sample_labels": sample_labels,
            "first_failure": kind_first_failure,
        }
    failed_kinds = [kind for kind in ("key", "value") if kind_summaries[kind]["failure_rows"]]
    return {
        "failed_kinds": failed_kinds,
        "failed_kind_count": len(failed_kinds),
        "first_failure": first_failure,
        "kinds": kind_summaries,
        "layer_limits": [
            {
                "layer_limit": int(summary.get("layer_limit", 0)),
                "passed": bool(summary.get("decode_full_kv_sample_passed", True)),
                "first_mismatch": summary.get("decode_full_kv_samples", {}).get("first_mismatch")
                if isinstance(summary.get("decode_full_kv_samples"), dict)
                else None,
            }
            for summary in layer_summaries
            if isinstance(summary.get("decode_full_kv_samples"), dict)
        ],
    }


def _decode_full_kv_sample_summary(
    batch: HiddenRun,
    c1: HiddenRun,
    *,
    atol: float,
) -> dict[str, Any] | None:
    if not batch.decode_full_kv_samples_by_step or not c1.decode_full_kv_samples_by_step:
        return None
    if not any(step_layers for step_layers in batch.decode_full_kv_samples_by_step) or not any(
        step_layers for step_layers in c1.decode_full_kv_samples_by_step
    ):
        return None
    steps: list[dict[str, Any]] = []
    for step, (batch_layers, c1_layers) in enumerate(
        zip(batch.decode_full_kv_samples_by_step, c1.decode_full_kv_samples_by_step, strict=True)
    ):
        layers: list[dict[str, Any]] = []
        for layer_id in sorted(set(batch_layers) & set(c1_layers)):
            batch_sample = batch_layers[layer_id]
            c1_sample = c1_layers[layer_id]
            labels = tuple(str(label) for label in batch_sample.get("sample_labels", DECODE_FULL_KV_SAMPLE_LABELS))
            batch_positions = np.asarray(batch_sample["sample_positions"], dtype=np.int64)
            c1_positions = np.asarray(c1_sample["sample_positions"], dtype=np.int64)
            if batch_positions.shape != c1_positions.shape:
                raise ValueError(
                    f"decode full-KV sample positions differ for step {step}, layer {layer_id}: "
                    f"batch={batch_positions.shape} c1={c1_positions.shape}"
                )
            row_summaries: list[dict[str, Any]] = []
            for row in range(int(batch_positions.shape[0])):
                row_summary: dict[str, Any] = {
                    "row": int(row),
                    "sample_labels": list(labels),
                    "sample_positions": [int(value) for value in batch_positions[row].tolist()],
                    "sample_positions_match": bool(np.array_equal(batch_positions[row], c1_positions[row])),
                }
                row_passed = bool(row_summary["sample_positions_match"])
                for kind, key in (("key", "key_bits"), ("value", "value_bits")):
                    batch_bits = np.asarray(batch_sample[key], dtype=np.uint16)
                    c1_bits = np.asarray(c1_sample[key], dtype=np.uint16)
                    if batch_bits.shape != c1_bits.shape:
                        raise ValueError(
                            f"decode full-KV {kind} sample shape differs for step {step}, layer {layer_id}: "
                            f"batch={batch_bits.shape} c1={c1_bits.shape}"
                        )
                    comparison = bf16_bits_comparison(batch_bits[row : row + 1], c1_bits[row : row + 1], atol=atol)
                    sample_comparisons: list[dict[str, Any]] = []
                    for sample_index, label in enumerate(labels):
                        sample_comparison = bf16_bits_comparison(
                            batch_bits[row : row + 1, sample_index : sample_index + 1],
                            c1_bits[row : row + 1, sample_index : sample_index + 1],
                            atol=atol,
                        )
                        sample_comparisons.append(
                            {
                                "sample_index": int(sample_index),
                                "sample_label": str(label),
                                "sample_position": int(batch_positions[row, sample_index]),
                                "sample_position_match": bool(
                                    batch_positions[row, sample_index] == c1_positions[row, sample_index]
                                ),
                                "comparison": sample_comparison,
                            }
                        )
                    row_summary[f"{kind}_comparison"] = comparison
                    row_summary[f"{kind}_sample_comparisons"] = sample_comparisons
                    row_passed = row_passed and bool(comparison["passed"])
                row_summary["passed"] = row_passed
                row_summaries.append(row_summary)
            layers.append(
                {
                    "layer_index": int(layer_id),
                    "passed": all(row["passed"] for row in row_summaries),
                    "rows": row_summaries,
                }
            )
        steps.append(
            {
                "decode_step": int(step),
                "generated_index": int(step + 1),
                "passed": all(layer["passed"] for layer in layers),
                "layers": layers,
            }
        )
    first_mismatch: dict[str, Any] | None = None
    worst_diff: dict[str, Any] | None = None
    for step_summary in steps:
        for layer in step_summary["layers"]:
            for row in layer["rows"]:
                for kind in ("key", "value"):
                    comparison = row[f"{kind}_comparison"]
                    failed_samples = [
                        sample
                        for sample in row.get(f"{kind}_sample_comparisons", [])
                        if not bool(sample.get("comparison", {}).get("passed", False))
                    ]
                    first_failed_sample = failed_samples[0] if failed_samples else None
                    max_sample: dict[str, Any] | None = None
                    max_abs_index = comparison.get("max_abs_index", [])
                    if isinstance(max_abs_index, list) and len(max_abs_index) >= 2:
                        sample_index = int(max_abs_index[1])
                        samples = row.get(f"{kind}_sample_comparisons", [])
                        if 0 <= sample_index < len(samples):
                            max_sample = samples[sample_index]
                    if worst_diff is None or float(comparison["max_abs"]) > float(worst_diff["max_abs"]):
                        worst_diff = {
                            "decode_step": int(step_summary["decode_step"]),
                            "generated_index": int(step_summary["generated_index"]),
                            "layer_index": int(layer["layer_index"]),
                            "row": int(row["row"]),
                            "kind": kind,
                            "passed": bool(comparison["passed"]),
                            "max_abs": float(comparison["max_abs"]),
                            "max_abs_flat_index": comparison.get("max_abs_flat_index"),
                            "max_abs_index": comparison.get("max_abs_index", []),
                            "elements_over_atol": int(comparison["elements_over_atol"]),
                            "bit_mismatch": int(comparison.get("bit_mismatch", 0)),
                        }
                        if isinstance(max_sample, dict):
                            worst_diff.update(
                                {
                                    "sample_index": int(max_sample.get("sample_index", -1)),
                                    "sample_label": str(max_sample.get("sample_label", "")),
                                    "sample_position": int(max_sample.get("sample_position", -1)),
                                }
                            )
                    if first_mismatch is None and not bool(comparison["passed"]):
                        first_mismatch = {
                            "decode_step": int(step_summary["decode_step"]),
                            "generated_index": int(step_summary["generated_index"]),
                            "layer_index": int(layer["layer_index"]),
                            "row": int(row["row"]),
                            "kind": kind,
                            "sample_labels": row["sample_labels"],
                            "sample_positions": row["sample_positions"],
                            "sample_positions_match": bool(row["sample_positions_match"]),
                            "max_abs": float(comparison["max_abs"]),
                            "max_abs_flat_index": comparison.get("max_abs_flat_index"),
                            "max_abs_index": comparison.get("max_abs_index", []),
                            "elements_over_atol": int(comparison["elements_over_atol"]),
                            "bit_mismatch": int(comparison.get("bit_mismatch", 0)),
                        }
                        if isinstance(first_failed_sample, dict):
                            first_mismatch["first_failed_sample"] = _decode_full_kv_sample_comparison_record(
                                step_summary,
                                layer,
                                row,
                                kind,
                                first_failed_sample,
                            )
    result = {
        "stage": "decode_full_kv_samples",
        "bf16_atol": float(atol),
        "passed": all(step["passed"] for step in steps),
        "steps": steps,
    }
    if first_mismatch is not None:
        result["first_mismatch"] = first_mismatch
    if worst_diff is not None:
        result["worst_diff"] = worst_diff
    return result


def _decode_linear_state_focus_history(
    batch: HiddenRun,
    c1: HiddenRun,
    *,
    focus: dict[str, Any],
    atol: float,
    focus_atol: float | None,
) -> list[dict[str, Any]]:
    layer_index = int(focus.get("layer_index", -1))
    state_name = str(focus.get("state", ""))
    row_index = int(focus.get("row", -1))
    focus_flat_index = focus.get("max_abs_flat_index")
    focus_max_abs_index = focus.get("max_abs_index", [])
    history: list[dict[str, Any]] = []
    for step, (batch_states, c1_states) in enumerate(
        zip(batch.decode_linear_states_by_step, c1.decode_linear_states_by_step, strict=True)
    ):
        if layer_index not in batch_states or layer_index not in c1_states:
            continue
        layer_batch = batch_states[layer_index]
        layer_c1 = c1_states[layer_index]
        if state_name not in layer_batch or state_name not in layer_c1:
            continue
        batch_state = layer_batch[state_name]
        c1_state = layer_c1[state_name]
        if row_index < 0 or row_index >= int(batch_state.shape[0]):
            continue
        row_batch = batch_state[row_index]
        row_c1 = c1_state[row_index]
        previous_batch_states = batch.prefill_linear_states if step == 0 else batch.decode_linear_states_by_step[step - 1]
        previous_c1_states = c1.prefill_linear_states if step == 0 else c1.decode_linear_states_by_step[step - 1]
        previous_row_batch = None
        previous_row_c1 = None
        if layer_index in previous_batch_states and layer_index in previous_c1_states:
            previous_layer_batch = previous_batch_states[layer_index]
            previous_layer_c1 = previous_c1_states[layer_index]
            if state_name in previous_layer_batch and state_name in previous_layer_c1:
                previous_state_batch = previous_layer_batch[state_name]
                previous_state_c1 = previous_layer_c1[state_name]
                if row_index < int(previous_state_batch.shape[0]) and row_index < int(previous_state_c1.shape[0]):
                    previous_row_batch = previous_state_batch[row_index]
                    previous_row_c1 = previous_state_c1[row_index]
        comparison = numeric_comparison(row_batch, row_c1, atol=atol)
        selected_flat_index: int | None = int(focus_flat_index) if isinstance(focus_flat_index, int) else None
        if selected_flat_index is None and focus_max_abs_index:
            selected_flat_index = int(np.ravel_multi_index(tuple(int(index) for index in focus_max_abs_index), row_batch.shape))
        entry: dict[str, Any] = {
            "decode_step": int(step),
            "generated_index": int(step + 1),
            "layer_index": layer_index,
            "state": state_name,
            "row": row_index,
            "passed": bool(comparison["passed"]),
            "max_abs": float(comparison["max_abs"]),
            "max_abs_flat_index": comparison["max_abs_flat_index"],
            "max_abs_index": comparison["max_abs_index"],
            "elements_over_atol": int(comparison["elements_over_atol"]),
            "top_abs_diffs": comparison["top_abs_diffs"][:3],
            "focus_max_abs_index": focus_max_abs_index,
        }
        if selected_flat_index is not None:
            selected = _numeric_abs_diff_at_flat_index(row_batch, row_c1, selected_flat_index)
            entry["focus_flat_index"] = selected_flat_index
            entry["same_focus_index_diff"] = selected
            if focus_atol is not None:
                entry["same_focus_index_passed_under_focus_atol"] = bool(float(selected["abs_diff"]) <= float(focus_atol))
        if previous_row_batch is not None and previous_row_c1 is not None:
            previous_comparison = numeric_comparison(previous_row_batch, previous_row_c1, atol=atol)
            batch_delta = np.asarray(row_batch, dtype=np.float32) - np.asarray(previous_row_batch, dtype=np.float32)
            c1_delta = np.asarray(row_c1, dtype=np.float32) - np.asarray(previous_row_c1, dtype=np.float32)
            delta_comparison = numeric_comparison(batch_delta, c1_delta, atol=atol)
            entry["previous_state_comparison"] = _compact_comparison(previous_comparison)
            entry["state_update_delta_comparison"] = _compact_comparison(delta_comparison)
            if selected_flat_index is not None:
                entry["same_focus_index_previous_diff"] = _numeric_abs_diff_at_flat_index(
                    previous_row_batch,
                    previous_row_c1,
                    selected_flat_index,
                )
                entry["same_focus_index_delta_diff"] = _numeric_abs_diff_at_flat_index(
                    batch_delta,
                    c1_delta,
                    selected_flat_index,
                )
        if focus_atol is not None:
            entry["state_focus_atol"] = float(focus_atol)
            entry["passed_under_focus_atol"] = bool(float(comparison["max_abs"]) <= float(focus_atol))
        history.append(entry)
    return history


def _decode_linear_state_summary(
    batch: HiddenRun,
    c1: HiddenRun,
    *,
    atol: float,
    focus_atol: float | None = None,
) -> dict[str, Any] | None:
    if not batch.decode_linear_states_by_step or not c1.decode_linear_states_by_step:
        return None
    steps: list[dict[str, Any]] = []
    for step, (batch_states, c1_states) in enumerate(
        zip(batch.decode_linear_states_by_step, c1.decode_linear_states_by_step, strict=True)
    ):
        layers = _linear_state_layers_summary(batch_states, c1_states, atol=atol)
        steps.append(
            {
                "decode_step": int(step),
                "generated_index": int(step + 1),
                "passed": all(layer["passed"] for layer in layers),
                "layers": layers,
            }
        )
    first_mismatch: dict[str, Any] | None = None
    for step_summary in steps:
        for layer in step_summary["layers"]:
            for state_name in ("conv", "recurrent"):
                state_summary = layer["states"].get(state_name)
                if state_summary is None:
                    continue
                for row in state_summary.get("row_summaries", []):
                    if row["passed"]:
                        continue
                    first_mismatch = {
                        "decode_step": int(step_summary["decode_step"]),
                        "generated_index": int(step_summary["generated_index"]),
                        "layer_index": int(layer["layer_index"]),
                        "state": state_name,
                        "row": int(row["row"]),
                        "max_abs": float(row["max_abs"]),
                        "max_abs_flat_index": row.get("max_abs_flat_index"),
                        "max_abs_index": row["max_abs_index"],
                        "elements_over_atol": int(row["elements_over_atol"]),
                    }
                    break
                if first_mismatch is not None:
                    break
            if first_mismatch is not None:
                break
        if first_mismatch is not None:
            break
    first_focus_mismatch: dict[str, Any] | None = None
    if focus_atol is not None:
        for step_summary in steps:
            for layer in step_summary["layers"]:
                for state_name in ("conv", "recurrent"):
                    state_summary = layer["states"].get(state_name)
                    if state_summary is None:
                        continue
                    for row in state_summary.get("row_summaries", []):
                        if float(row["max_abs"]) <= float(focus_atol):
                            continue
                        first_focus_mismatch = {
                            "decode_step": int(step_summary["decode_step"]),
                            "generated_index": int(step_summary["generated_index"]),
                            "layer_index": int(layer["layer_index"]),
                            "state": state_name,
                            "row": int(row["row"]),
                            "max_abs": float(row["max_abs"]),
                            "max_abs_flat_index": row.get("max_abs_flat_index"),
                            "max_abs_index": row["max_abs_index"],
                            "elements_over_atol": int(row["elements_over_atol"]),
                            "state_focus_atol": float(focus_atol),
                            "passed_under_focus_atol": False,
                        }
                        break
                    if first_focus_mismatch is not None:
                        break
                if first_focus_mismatch is not None:
                    break
            if first_focus_mismatch is not None:
                break
    worst_diff: dict[str, Any] | None = None
    for step_summary in steps:
        for layer in step_summary["layers"]:
            for state_name in ("conv", "recurrent"):
                state_summary = layer["states"].get(state_name)
                if state_summary is None:
                    continue
                for row in state_summary.get("row_summaries", []):
                    if worst_diff is not None and float(row["max_abs"]) <= float(worst_diff["max_abs"]):
                        continue
                    worst_diff = {
                        "decode_step": int(step_summary["decode_step"]),
                        "generated_index": int(step_summary["generated_index"]),
                        "layer_index": int(layer["layer_index"]),
                        "state": state_name,
                        "row": int(row["row"]),
                        "passed": bool(row["passed"]),
                        "max_abs": float(row["max_abs"]),
                        "max_abs_flat_index": row.get("max_abs_flat_index"),
                        "max_abs_index": row["max_abs_index"],
                        "elements_over_atol": int(row["elements_over_atol"]),
                    }
    result = {
        "stage": "decode_linear_states",
        "state_atol": float(atol),
        "passed": all(step["passed"] for step in steps),
        "steps": steps,
    }
    if focus_atol is not None:
        result["state_focus_atol"] = float(focus_atol)
        result["passed_under_focus_atol"] = first_focus_mismatch is None
    if first_mismatch is not None:
        result["first_mismatch"] = first_mismatch
    if first_focus_mismatch is not None:
        result["first_mismatch_over_focus_atol"] = first_focus_mismatch
        result["first_mismatch_over_focus_atol_history"] = _decode_linear_state_focus_history(
            batch,
            c1,
            focus=first_focus_mismatch,
            atol=atol,
            focus_atol=focus_atol,
        )
    if worst_diff is not None:
        result["worst_diff"] = worst_diff
    return result


def _prefill_linear_input_summary(
    batch: HiddenRun,
    c1: HiddenRun,
    *,
    atol: float,
    focus_hidden_flat_indices: Sequence[int] = (),
) -> dict[str, Any] | None:
    if not batch.prefill_linear_inputs or not c1.prefill_linear_inputs:
        return None
    layers: list[dict[str, Any]] = []
    for layer_id in sorted(set(batch.prefill_linear_inputs) & set(c1.prefill_linear_inputs)):
        batch_rows = batch.prefill_linear_inputs[layer_id]
        c1_rows = c1.prefill_linear_inputs[layer_id]
        if len(batch_rows) != len(c1_rows):
            raise ValueError(
                f"prefill linear input trace row count differs for layer {layer_id}: batch={len(batch_rows)} c1={len(c1_rows)}"
            )
        row_summaries: list[dict[str, Any]] = []
        for row, (batch_bits, c1_bits) in enumerate(zip(batch_rows, c1_rows, strict=True)):
            full_comparison = hidden_comparison(batch_bits, c1_bits, atol=atol)
            last_token_comparison = hidden_comparison(
                batch_bits[-1:],
                c1_bits[-1:],
                atol=atol,
                selected_flat_indices=focus_hidden_flat_indices,
            )
            row_summaries.append(
                {
                    "row": int(row),
                    "tokens": int(batch_bits.shape[0]),
                    "hidden_comparison": full_comparison,
                    "last_token_hidden_comparison": last_token_comparison,
                    "passed": bool(full_comparison["passed"]),
                }
            )
        layers.append(
            {
                "layer_index": int(layer_id),
                "passed": all(row["passed"] for row in row_summaries),
                "rows": row_summaries,
            }
        )
    return {
        "stage": "prefill_linear_inputs",
        "hidden_atol": float(atol),
        "passed": all(layer["passed"] for layer in layers),
        "layers": layers,
    }


def _summarize_layer_limit(
    batch: HiddenRun,
    c1: HiddenRun,
    *,
    layer_limit: int,
    atol: float,
    state_atol: float = 1.0e-6,
    state_focus_atol: float | None = None,
    layer_types: Sequence[str] | None = None,
    focus_hidden_flat_indices: Sequence[int] = (),
) -> dict[str, Any]:
    prefill = _prefill_summary(batch, c1, atol=atol, focus_hidden_flat_indices=focus_hidden_flat_indices)
    prefill_linear_states = _prefill_linear_state_summary(batch, c1, atol=state_atol)
    prefill_linear_inputs = _prefill_linear_input_summary(
        batch,
        c1,
        atol=atol,
        focus_hidden_flat_indices=focus_hidden_flat_indices,
    )
    prefill_full_kv_prefix_hashes = _prefill_full_kv_prefix_summary(batch, c1)
    decode_linear_inputs = _decode_linear_input_summary(
        batch,
        c1,
        atol=atol,
        focus_hidden_flat_indices=focus_hidden_flat_indices,
    )
    decode_full_attention = _decode_full_attention_summary(
        batch,
        c1,
        atol=atol,
        focus_hidden_flat_indices=focus_hidden_flat_indices,
    )
    decode_full_context_oracle = _decode_full_context_oracle_summary(batch, c1)
    decode_full_kv_samples = _decode_full_kv_sample_summary(batch, c1, atol=0.0)
    decode_linear_states = _decode_linear_state_summary(batch, c1, atol=state_atol, focus_atol=state_focus_atol)
    steps: list[dict[str, Any]] = []
    for step, (batch_bits, c1_bits) in enumerate(zip(batch.hidden_bits_by_step, c1.hidden_bits_by_step, strict=True)):
        rows: list[dict[str, Any]] = []
        for row in range(batch_bits.shape[0]):
            rows.append(
                {
                    "row": row,
                    "hidden_comparison": hidden_comparison(
                        batch_bits[row : row + 1],
                        c1_bits[row : row + 1],
                        atol=atol,
                        selected_flat_indices=focus_hidden_flat_indices,
                    ),
                }
            )
        step_summary: dict[str, Any] = {"decode_step": step, "generated_index": step + 1, "rows": rows}
        if step < len(batch.decode_execution_by_step) and batch.decode_execution_by_step[step] is not None:
            step_summary["batch_decode_execution"] = batch.decode_execution_by_step[step]
        steps.append(step_summary)
    token_mismatches = _token_mismatches(batch, c1)
    hidden_passed = all(row["hidden_comparison"]["passed"] for step in steps for row in step["rows"])
    token_passed = not token_mismatches
    summary = {
        "layer_limit": int(layer_limit),
        "hidden_atol": float(atol),
        **_layer_limit_metadata(layer_limit, layer_types),
        "prefill_hidden_passed": True if prefill is None else bool(prefill["hidden_passed"]),
        "prefill_linear_input_passed": True if prefill_linear_inputs is None else bool(prefill_linear_inputs["passed"]),
        "prefill_linear_state_passed": True if prefill_linear_states is None else bool(prefill_linear_states["passed"]),
        "prefill_full_kv_prefix_passed": (
            True if prefill_full_kv_prefix_hashes is None else bool(prefill_full_kv_prefix_hashes["passed"])
        ),
        "decode_linear_input_passed": True if decode_linear_inputs is None else bool(decode_linear_inputs["passed"]),
        "decode_full_attention_input_passed": (
            True if decode_full_attention is None else bool(decode_full_attention["input_passed"])
        ),
        "decode_full_attention_output_passed": (
            True if decode_full_attention is None else bool(decode_full_attention["output_passed"])
        ),
        "decode_full_context_oracle_passed": (
            True if decode_full_context_oracle is None else bool(decode_full_context_oracle["passed"])
        ),
        "decode_full_kv_sample_passed": True if decode_full_kv_samples is None else bool(decode_full_kv_samples["passed"]),
        "decode_linear_state_passed": True if decode_linear_states is None else bool(decode_linear_states["passed"]),
        "hidden_passed": hidden_passed,
        "token_passed": token_passed,
        "failure_modes": _failure_modes(hidden_passed=hidden_passed, token_passed=token_passed),
        "seed_tokens": {"batch": batch.seed_tokens, "c1": c1.seed_tokens},
        "generated_tokens": {"batch": batch.generated_tokens, "c1": c1.generated_tokens},
        "token_mismatches": token_mismatches,
        "steps": steps,
    }
    hidden_failure_rows = _hidden_failure_rows(summary)
    strict_hidden_bit_drift_rows = _hidden_bit_drift_rows(summary)
    token_failure_rows = _token_failure_rows(summary)
    summary.update(
        {
            "hidden_failure_rows": hidden_failure_rows,
            "hidden_failure_row_count": len(hidden_failure_rows),
            "strict_hidden_bit_drift_rows": strict_hidden_bit_drift_rows,
            "strict_hidden_bit_drift_row_count": len(strict_hidden_bit_drift_rows),
            "token_failure_rows": token_failure_rows,
            "token_failure_row_count": len(token_failure_rows),
        }
    )
    if prefill is not None:
        summary["prefill"] = prefill
    if prefill_linear_states is not None:
        summary["prefill_linear_states"] = prefill_linear_states
    if prefill_linear_inputs is not None:
        summary["prefill_linear_inputs"] = prefill_linear_inputs
    if prefill_full_kv_prefix_hashes is not None:
        summary["prefill_full_kv_prefix_hashes"] = prefill_full_kv_prefix_hashes
    if decode_linear_inputs is not None:
        summary["decode_linear_inputs"] = decode_linear_inputs
    if decode_full_attention is not None:
        summary["decode_full_attention"] = decode_full_attention
    if decode_full_context_oracle is not None:
        summary["decode_full_context_oracle"] = decode_full_context_oracle
    if decode_full_kv_samples is not None:
        summary["decode_full_kv_samples"] = decode_full_kv_samples
    if decode_linear_states is not None:
        summary["decode_linear_states"] = decode_linear_states
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--layer-limits", default=None, help="Comma/range list such as '1,4,8' or '1-8'; default all")
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--hidden-atol", type=float, default=1.0e-3)
    parser.add_argument("--state-atol", type=float, default=1.0e-6, help="Absolute tolerance for linear-state comparisons.")
    parser.add_argument(
        "--state-focus-atol",
        type=float,
        default=None,
        help="Optional secondary absolute tolerance for reporting the first decode linear-state mismatch above a diagnostic focus threshold.",
    )
    parser.add_argument(
        "--focus-hidden-flat-index",
        action="append",
        default=None,
        help="Optional hidden flat index to record for every row/layer comparison; may be repeated or comma-separated.",
    )
    parser.add_argument(
        "--batch-decode-moe-path",
        choices=("grouped_compact", "selected_c1"),
        default="grouped_compact",
        help="Diagnostic MoE path for native c>N batch decode; selected_c1 forces the non-retained selected-c1 probe.",
    )
    parser.add_argument(
        "--batch-decode-linear-path",
        choices=("batch_segments", "per_row"),
        default="batch_segments",
        help="Diagnostic linear-attention decode path for native c>N batch decode; per_row forces the non-retained row loop.",
    )
    parser.add_argument(
        "--batch-decode-full-attn-path",
        choices=("native_batch", "per_row"),
        default="native_batch",
        help="Diagnostic full-attention decode path for native c>N batch decode; per_row forces the existing non-retained row loop.",
    )
    parser.add_argument(
        "--batch-decode-attn-input-path",
        choices=("batch", "per_row"),
        default="batch",
        help="Diagnostic full-attention input RMSNorm path for c>N batch decode; per_row forces token-1 row kernels and blocks retained claims.",
    )
    parser.add_argument(
        "--batch-decode-post-attn-path",
        choices=("batch", "per_row"),
        default="batch",
        help="Diagnostic post-attention add/RMSNorm path for c>N batch decode; per_row forces token-1 row kernels and blocks retained claims.",
    )
    parser.add_argument(
        "--batch-prefill-linear-path",
        choices=("packed_segments", "per_segment"),
        default="packed_segments",
        help="Diagnostic linear-attention packed-prefill path; per_segment forces per-request c=1-style linear prefill.",
    )
    parser.add_argument(
        "--batch-prefill-full-attn-path",
        choices=("packed_varlen", "per_segment"),
        default="packed_varlen",
        help="Diagnostic full-attention packed-prefill path; per_segment forces per-request c=1-style full-attention prefill.",
    )
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Emit planned layer limits and commands without touching HIP")
    return parser


def run(args: argparse.Namespace, argv: Sequence[str] | None = None) -> dict[str, Any]:
    layer_limits = _parse_layer_limits(args.layer_limits, max_layers=args.max_layers)
    focus_hidden_flat_indices = _parse_focus_hidden_flat_indices(args.focus_hidden_flat_index)
    prompt_lengths: list[int] = []
    if args.dry_run:
        prompts = []
    else:
        prompts = _load_prompt_slices(Path(args.fixture), prompt_length=args.prompt_length, batch_size=args.batch_size)
        prompt_lengths = [len(prompt) for prompt in prompts]
        if args.max_sequence_length < max(prompt_lengths) + args.decode_tokens + 1:
            raise ValueError("max_sequence_length must cover prompt_length + decode_tokens + 1")
    payload: dict[str, Any] = {
        "schema": 1,
        "status": "planned" if args.dry_run else "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "qwen35_paro_native_hidden_bisect",
        "command": _command(argv),
        "performance_claim": False,
        "workload": {
            "model": str(args.model),
            "fixture": str(args.fixture),
            "prompt_length": int(args.prompt_length),
            "prompt_lengths": prompt_lengths,
            "batch_size": int(args.batch_size),
            "decode_tokens": int(args.decode_tokens),
            "max_layers": int(args.max_layers),
            "layer_limits": layer_limits,
            "max_sequence_length": int(args.max_sequence_length),
            "kv_storage_dtype": "bf16",
            "native_compact_prefill": True,
            "focus_hidden_flat_indices": focus_hidden_flat_indices,
            "prefill_linear_state_atol": float(args.state_atol),
            "linear_state_atol": float(args.state_atol),
            "batch_prefill_linear_path": str(args.batch_prefill_linear_path),
            "batch_prefill_full_attention_path": str(args.batch_prefill_full_attn_path),
            "batch_decode_moe_path": str(args.batch_decode_moe_path),
            "batch_decode_linear_path": str(args.batch_decode_linear_path),
            "batch_decode_full_attention_path": str(args.batch_decode_full_attn_path),
            "batch_decode_attention_input_path": str(args.batch_decode_attn_input_path),
            "batch_decode_post_attention_path": str(args.batch_decode_post_attn_path),
            "native_caware_decode": bool(
                args.prompt_length + args.decode_tokens < 1024
                and args.batch_decode_linear_path == "batch_segments"
                and args.batch_decode_full_attn_path == "native_batch"
                and args.batch_decode_attn_input_path == "batch"
                and args.batch_decode_post_attn_path == "batch"
            ),
            "full_attention_decode_path": (
                "per_row_context_fallback"
                if args.batch_decode_full_attn_path == "per_row" and args.prompt_length + args.decode_tokens < 1024
                else "batch_context" if args.prompt_length + args.decode_tokens < 1024 else "per_row_splitk_fallback"
            ),
        },
        "correctness": {
            "oracle": "hidden tensors and generated-token IDs vs independent c=1 resident sessions",
            "hidden_atol": float(args.hidden_atol),
            "prefill_linear_state_atol": float(args.state_atol),
            "linear_state_atol": float(args.state_atol),
            "passed": False,
        },
        "layer_summaries": [],
        "blockers": [],
    }
    if args.state_focus_atol is not None:
        payload["workload"]["linear_state_focus_atol"] = float(args.state_focus_atol)
        payload["correctness"]["linear_state_focus_atol"] = float(args.state_focus_atol)
    if args.dry_run:
        payload["commands"] = [
            _command([*sys.argv[1:], "--layer-limits", str(limit)] if argv is None else [*argv, "--layer-limits", str(limit)])
            for limit in layer_limits
        ]
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=2) + "\n")
        return payload

    os.environ.setdefault("HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE", "1")
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_MOE"] = (
        "1" if args.batch_decode_moe_path == "selected_c1" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_LINEAR"] = (
        "1" if args.batch_decode_linear_path == "per_row" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_FULL_ATTN_NATIVE"] = (
        "0" if args.batch_decode_full_attn_path == "per_row" else "1"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_INPUT"] = (
        "1" if args.batch_decode_attn_input_path == "per_row" else "0"
    )
    os.environ["HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_POST_ATTN"] = (
        "1" if args.batch_decode_post_attn_path == "per_row" else "0"
    )
    os.environ["HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_LINEAR"] = (
        "1" if args.batch_prefill_linear_path == "per_segment" else "0"
    )
    os.environ["HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_FULL_ATTN"] = (
        "1" if args.batch_prefill_full_attn_path == "per_segment" else "0"
    )
    runner = Qwen35ParoNextTokenRunner(args.model)
    layer_types = tuple(str(layer_type) for layer_type in getattr(runner.config, "layer_types", ()))
    compiler_version = _compiler_version(args.compiler_version_file)
    layer_summaries: list[dict[str, Any]] = []
    for layer_limit in layer_limits:
        batch = _run_batch_hidden(
            runner,
            prompts,
            layer_limit=layer_limit,
            decode_tokens=args.decode_tokens,
            max_sequence_length=args.max_sequence_length,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
        c1 = _run_c1_hidden(
            runner,
            prompts,
            layer_limit=layer_limit,
            decode_tokens=args.decode_tokens,
            max_sequence_length=args.max_sequence_length,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
        )
        layer_summaries.append(
            _summarize_layer_limit(
                batch,
                c1,
                layer_limit=layer_limit,
                atol=args.hidden_atol,
                state_atol=args.state_atol,
                state_focus_atol=args.state_focus_atol,
                layer_types=layer_types,
                focus_hidden_flat_indices=focus_hidden_flat_indices,
            )
        )

    hidden_mismatch = _first_hidden_mismatch(layer_summaries)
    hidden_bit_drift = _first_hidden_bit_drift(layer_summaries)
    token_mismatch = _first_token_mismatch(layer_summaries)
    hidden_passed = hidden_mismatch is None
    token_passed = token_mismatch is None
    passed = hidden_passed and token_passed
    payload["status"] = "eq_ok" if passed else "mismatch_found"
    payload["correctness"].update(
        {
            "passed": passed,
            "hidden_passed": hidden_passed,
            "token_passed": token_passed,
            "failure_modes": _failure_modes(hidden_passed=hidden_passed, token_passed=token_passed),
            "row_failure_summary": _row_failure_summary(layer_summaries),
            "decode_full_attention_stage_failure_summary": _decode_full_attention_stage_rollup(layer_summaries),
            "prefill_full_kv_prefix_failure_summary": _prefill_full_kv_prefix_rollup(layer_summaries),
            "decode_full_context_oracle_failure_summary": _decode_full_context_oracle_rollup(layer_summaries),
            "decode_full_context_kv_prefix_failure_summary": _decode_full_context_kv_prefix_rollup(layer_summaries),
            "decode_full_kv_sample_failure_summary": _decode_full_kv_sample_rollup(layer_summaries),
            "first_hidden_mismatch": hidden_mismatch,
            "first_tolerance_hidden_mismatch": hidden_mismatch,
            "first_hidden_bit_drift": hidden_bit_drift,
            "first_token_mismatch": token_mismatch,
            "first_failing_layer_transition": _first_failing_layer_transition(layer_summaries),
        }
    )
    payload["layer_summaries"] = layer_summaries
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run(args, argv)
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] in {"eq_ok", "mismatch_found", "planned"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
