#!/usr/bin/env python3
"""Calibrate activation-only producer-row repair for Laguna Q4 gate prefill."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.core.memory import host_array_ptr
from hipengine.loading.gguf import GGUFReader
from hipengine.runtime import laguna_gguf_runner
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession
from hipengine.runtime.laguna_moe import (
    _launch_selected_gate_up_mmq32_d4x3,
)
from hipengine.tokenization.gguf import LagunaGGUFTokenizer
from scripts.laguna_grouped_down_category_bench import _extend_prompt_streams
from scripts.laguna_target_ar_bench import (
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_SHA256,
    DEFAULT_PROMPTS,
    _compiler_version,
    _load_prompts,
    _progress,
)


_ROWS = 512
_ATTENTION_ROWS = 128
_PRODUCTION_GATE_UP = "mmq128x32_d8_f32_wavecols_direct_doublebuf"
_PRODUCTION_DOWN = "mmq64x64_d4_f32_q6_wavecols_direct_q4"
_PRODUCTION_F16 = "hipblaslt_range_direct"
_PRODUCTION_DENSE = "wmma_pack8"
_PRODUCTION_GLOBAL = "global_context_rows_qrow4_m128_online_spans"
_PRODUCTION_SWA = "swa_context_rows_qrow4_m128_online_spans"
_DEFAULT_THRESHOLDS = (
    1.125,
    1.25,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
    12.0,
    16.0,
    24.0,
    32.0,
    48.0,
    64.0,
)
_FEATURE_THRESHOLDS = {
    "half_scale_ratio_max": _DEFAULT_THRESHOLDS,
    "half_scale_ratio_p95": _DEFAULT_THRESHOLDS,
    "half_scale_ratio_mean_log2": (
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.75,
        1.0,
    ),
    "half_scale_ratio_fraction_gt2": (
        0.0,
        1.0 / 96.0,
        2.0 / 96.0,
        4.0 / 96.0,
        8.0 / 96.0,
        16.0 / 96.0,
        32.0 / 96.0,
    ),
    "activation_abs_max": (
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
        12.0,
        16.0,
        24.0,
        32.0,
        64.0,
    ),
    "d4_vs_d8_delta_relative_l2": (
        0.004,
        0.00425,
        0.0045,
        0.00475,
        0.005,
        0.00525,
        0.0055,
        0.006,
        0.007,
        0.008,
    ),
    "d4_vs_d8_delta_max_abs": (
        0.0025,
        0.005,
        0.0075,
        0.01,
        0.0125,
        0.015,
        0.02,
        0.03,
        0.05,
        0.1,
        0.2,
    ),
}


def _bf16_bits_to_f32(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint16)
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _activation_risk_features(hidden: np.ndarray) -> dict[str, np.ndarray]:
    """Return source-row features available inside the Q8_1 producer."""

    values = np.asarray(hidden, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % 32:
        raise ValueError("hidden must be rank-2 with a K dimension divisible by 32")
    rows = values.shape[0]
    halves = np.abs(values).reshape(rows, -1, 2, 16).max(axis=3)
    high = halves.max(axis=2)
    low = halves.min(axis=2)
    ratio = np.divide(
        high,
        low,
        out=np.full_like(high, np.inf),
        where=low > 0.0,
    )
    both_zero = (high == 0.0) & (low == 0.0)
    ratio[both_zero] = 1.0
    finite_ratio = np.where(np.isfinite(ratio), ratio, np.finfo(np.float32).max)

    block32 = values.reshape(rows, -1, 32)
    block16 = values.reshape(rows, -1, 16)
    scale32 = np.abs(block32).max(axis=2, keepdims=True) / np.float32(127.0)
    scale16 = np.abs(block16).max(axis=2, keepdims=True) / np.float32(127.0)
    q32 = np.divide(
        block32,
        scale32,
        out=np.zeros_like(block32),
        where=scale32 > 0.0,
    )
    q16 = np.divide(
        block16,
        scale16,
        out=np.zeros_like(block16),
        where=scale16 > 0.0,
    )
    reconstructed32 = (
        np.clip(np.rint(q32), -127.0, 127.0) * scale32
    ).reshape(values.shape)
    reconstructed16 = (
        np.clip(np.rint(q16), -127.0, 127.0) * scale16
    ).reshape(values.shape)
    delta = reconstructed32 - reconstructed16
    norm_sq = np.square(values, dtype=np.float32).sum(axis=1, dtype=np.float64)
    delta_sq = np.square(delta, dtype=np.float32).sum(axis=1, dtype=np.float64)

    return {
        "half_scale_ratio_max": finite_ratio.max(axis=1),
        "half_scale_ratio_p95": np.quantile(
            finite_ratio, 0.95, axis=1
        ).astype(np.float32),
        "half_scale_ratio_mean_log2": np.log2(finite_ratio).mean(
            axis=1, dtype=np.float64
        ),
        "half_scale_ratio_fraction_gt2": (finite_ratio > 2.0).mean(
            axis=1, dtype=np.float64
        ),
        "activation_abs_max": np.abs(values).max(axis=1),
        "activation_rms": np.sqrt(norm_sq / values.shape[1]),
        "d4_vs_d8_delta_relative_l2": np.sqrt(
            np.divide(
                delta_sq,
                norm_sq,
                out=np.zeros_like(delta_sq),
                where=norm_sq > 0.0,
            )
        ),
        "d4_vs_d8_delta_max_abs": np.abs(delta).max(axis=1),
    }


def _aggregate_role_error(
    gate_d4: np.ndarray,
    gate_d8: np.ndarray,
    up_d8: np.ndarray,
    lane_to_row: np.ndarray,
    route_weights: np.ndarray,
    *,
    rows: int,
) -> dict[str, np.ndarray]:
    """Aggregate compact-route projection error back to producer rows."""

    d4 = np.asarray(gate_d4, dtype=np.float32)
    d8 = np.asarray(gate_d8, dtype=np.float32)
    up = np.asarray(up_d8, dtype=np.float32)
    mapping = np.asarray(lane_to_row, dtype=np.int64)
    weights = np.asarray(route_weights, dtype=np.float32)
    if d4.shape != d8.shape or d4.shape != up.shape or d4.ndim != 2:
        raise ValueError("gate/up arrays must be matching rank-2 arrays")
    if mapping.shape != (d4.shape[0],) or weights.shape != mapping.shape:
        raise ValueError("compact metadata must have one entry per route row")
    if np.any(mapping < 0) or np.any(mapping >= int(rows)):
        raise ValueError("lane_to_row contains an out-of-range producer row")

    gate_delta = d4 - d8
    gate_error_sq = np.square(gate_delta, dtype=np.float32).sum(
        axis=1, dtype=np.float64
    )
    d4_silu = d4 / (np.float32(1.0) + np.exp(-np.clip(d4, -80.0, 80.0)))
    d8_silu = d8 / (np.float32(1.0) + np.exp(-np.clip(d8, -80.0, 80.0)))
    intermediate_delta = (d4_silu - d8_silu) * up
    intermediate_error_sq = np.square(
        intermediate_delta, dtype=np.float32
    ).sum(axis=1, dtype=np.float64)
    weighted_error_sq = intermediate_error_sq * np.square(
        weights, dtype=np.float32
    )

    result = {
        "gate_error_sq": np.zeros(rows, dtype=np.float64),
        "intermediate_error_sq": np.zeros(rows, dtype=np.float64),
        "route_weighted_intermediate_error_sq": np.zeros(rows, dtype=np.float64),
        "intermediate_error_max_abs": np.zeros(rows, dtype=np.float32),
    }
    np.add.at(result["gate_error_sq"], mapping, gate_error_sq)
    np.add.at(
        result["intermediate_error_sq"],
        mapping,
        intermediate_error_sq,
    )
    np.add.at(
        result["route_weighted_intermediate_error_sq"],
        mapping,
        weighted_error_sq,
    )
    np.maximum.at(
        result["intermediate_error_max_abs"],
        mapping,
        np.abs(intermediate_delta).max(axis=1),
    )
    return result


def _threshold_sweep(
    risk: np.ndarray,
    error_mass: np.ndarray,
    *,
    thresholds: Sequence[float],
) -> list[dict[str, float]]:
    risk_values = np.asarray(risk, dtype=np.float64)
    errors = np.asarray(error_mass, dtype=np.float64)
    if risk_values.ndim != 1 or errors.shape != risk_values.shape:
        raise ValueError("risk and error_mass must be matching vectors")
    if np.any(errors < 0.0):
        raise ValueError("error_mass must be nonnegative")
    total_error = float(errors.sum())
    severe_cutoff = float(np.quantile(errors, 0.99))
    severe = errors >= severe_cutoff
    severe_count = int(severe.sum())
    rows = max(1, risk_values.size)
    result: list[dict[str, float]] = []
    for threshold in thresholds:
        repaired = risk_values >= float(threshold)
        repaired_error = float(errors[repaired].sum())
        result.append(
            {
                "threshold": float(threshold),
                "repair_fraction": float(repaired.sum() / rows),
                "error_mass_coverage": (
                    repaired_error / total_error if total_error > 0.0 else 1.0
                ),
                "severe_row_coverage": (
                    float(np.count_nonzero(repaired & severe) / severe_count)
                    if severe_count
                    else 1.0
                ),
            }
        )
    return result


def _copy_device_array(
    runtime,
    pointer: int,
    shape: tuple[int, ...],
    dtype,
) -> np.ndarray:
    result = np.empty(shape, dtype=dtype)
    runtime.memcpy(
        host_array_ptr(result),
        int(pointer),
        result.nbytes,
        HipMemcpyKind.DEVICE_TO_HOST,
    )
    return result


class _RoleRiskCollector:
    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self.prompt_id = ""
        self.rows: list[dict[str, np.ndarray | int | str]] = []
        self._original = laguna_gguf_runner.run_laguna_moe_rows

    def install(self) -> None:
        laguna_gguf_runner.run_laguna_moe_rows = self

    def restore(self) -> None:
        laguna_gguf_runner.run_laguna_moe_rows = self._original

    def __call__(self, hidden_ptr, layer, scratch, **kwargs):
        result = self._original(hidden_ptr, layer, scratch, **kwargs)
        rows = int(kwargs["rows"])
        if rows != _ROWS:
            return result
        plan = scratch.plan
        lanes = rows * plan.top_k
        hidden_bits = _copy_device_array(
            self.runtime,
            hidden_ptr,
            (rows, plan.hidden_size),
            np.uint16,
        )
        common = {
            "x_rows": rows,
            "lanes": lanes,
            "stream": int(kwargs.get("stream", 0)),
            "runtime": kwargs.get("runtime"),
            "libraries": kwargs.get("libraries"),
            "residual_passes": 1,
            "f32_wide": True,
            "split16": True,
            "rowvec": True,
            "wave_cols": True,
            "direct_wave_decode": True,
            "double_buffer_activation": True,
            "group_compact_mode": str(
                kwargs.get("group_compact_mode", "serial")
            ),
            "defer_silu_pack": True,
        }
        launched = _launch_selected_gate_up_mmq32_d4x3(
            hidden_ptr,
            layer,
            scratch,
            role_gate_split16=False,
            **common,
        )
        if not launched:
            raise RuntimeError("D4-gate/D8-up calibration launch was rejected")
        gate_d4_bits = _copy_device_array(
            self.runtime,
            scratch.expert_gate.ptr,
            (lanes, plan.expert_ffn_size),
            np.uint16,
        )
        up_d8_bits = _copy_device_array(
            self.runtime,
            scratch.expert_up.ptr,
            (lanes, plan.expert_ffn_size),
            np.uint16,
        )
        lane_to_row = _copy_device_array(
            self.runtime,
            scratch.grouped_lane_to_row.ptr,
            (lanes,),
            np.int64,
        )
        route_weights = _copy_device_array(
            self.runtime,
            scratch.grouped_sorted_weights.ptr,
            (lanes,),
            np.float32,
        )
        launched = _launch_selected_gate_up_mmq32_d4x3(
            hidden_ptr,
            layer,
            scratch,
            role_gate_split16=True,
            **common,
        )
        if not launched:
            raise RuntimeError("D8-gate/D4-up calibration launch was rejected")
        gate_d8_bits = _copy_device_array(
            self.runtime,
            scratch.expert_gate.ptr,
            (lanes, plan.expert_ffn_size),
            np.uint16,
        )
        hidden = _bf16_bits_to_f32(hidden_bits)
        error = _aggregate_role_error(
            _bf16_bits_to_f32(gate_d4_bits),
            _bf16_bits_to_f32(gate_d8_bits),
            _bf16_bits_to_f32(up_d8_bits),
            lane_to_row,
            route_weights,
            rows=rows,
        )
        self.rows.append(
            {
                "prompt_id": self.prompt_id,
                "layer_id": int(layer.layer_id),
                **_activation_risk_features(hidden),
                **error,
            }
        )
        print(
            f"capture {self.prompt_id} layer={int(layer.layer_id)} "
            f"rows={rows}",
            file=sys.stderr,
            flush=True,
        )
        return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-id", action="append")
    parser.add_argument("--prompt-count", type=int, default=1)
    parser.add_argument("--repacked-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _select_prompts(
    prompts: Sequence[Mapping[str, Any]],
    *,
    prompt_ids: Sequence[str] | None,
    prompt_count: int,
) -> list[dict[str, Any]]:
    if prompt_count <= 0:
        raise ValueError("--prompt-count must be positive")
    rows = [dict(prompt) for prompt in prompts]
    if prompt_ids:
        requested = tuple(dict.fromkeys(str(value) for value in prompt_ids))
        by_id = {str(prompt["id"]): prompt for prompt in rows}
        missing = [prompt_id for prompt_id in requested if prompt_id not in by_id]
        if missing:
            raise ValueError(f"unknown prompt IDs: {missing}")
        return [by_id[prompt_id] for prompt_id in requested]
    return rows[:prompt_count]


def _summarize(
    captured: Sequence[Mapping[str, Any]],
    *,
    prompt_ids: Sequence[str],
    elapsed_seconds: float,
) -> dict[str, Any]:
    if not captured:
        raise ValueError("calibration captured no M512 sparse-MoE rows")
    vector_keys = tuple(
        key
        for key in captured[0]
        if key not in {"prompt_id", "layer_id"}
    )
    combined = {
        key: np.concatenate(
            [np.asarray(row[key], dtype=np.float64) for row in captured]
        )
        for key in vector_keys
    }
    error_key = "route_weighted_intermediate_error_sq"
    features = (
        "half_scale_ratio_max",
        "half_scale_ratio_p95",
        "half_scale_ratio_mean_log2",
        "half_scale_ratio_fraction_gt2",
        "activation_abs_max",
        "d4_vs_d8_delta_relative_l2",
        "d4_vs_d8_delta_max_abs",
    )
    sweeps = {
        feature: _threshold_sweep(
            combined[feature],
            combined[error_key],
            thresholds=_FEATURE_THRESHOLDS[feature],
        )
        for feature in features
    }
    best_bounded = {}
    for feature, rows in sweeps.items():
        eligible = [row for row in rows if row["repair_fraction"] <= 0.251]
        best_bounded[feature] = (
            max(eligible, key=lambda row: row["error_mass_coverage"])
            if eligible
            else None
        )
    return {
        "schema_version": 1,
        "kind": "hipengine_laguna_q4_role_risk_calibration",
        "status": "diagnostic",
        "model": str(DEFAULT_MODEL),
        "quant": "gguf_q4_k_m",
        "hardware": "AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151)",
        "workload": {
            "prompt_ids": list(prompt_ids),
            "matrix_rows": _ROWS,
            "attention_rows": _ATTENTION_ROWS,
            "captured_layers": len(captured),
            "producer_rows": int(combined[error_key].size),
            "production_state_preserved": True,
        },
        "policy_constraints": {
            "risk_inputs": "activation_only",
            "scope": "global_layer_and_prompt_independent",
            "repair_unit": "producer_row_all_top10_routes",
            "maximum_screen_repair_fraction": 0.25,
        },
        "elapsed_seconds": float(elapsed_seconds),
        "feature_summary": {
            key: {
                "min": float(np.min(combined[key])),
                "median": float(np.median(combined[key])),
                "p95": float(np.quantile(combined[key], 0.95)),
                "p99": float(np.quantile(combined[key], 0.99)),
                "max": float(np.max(combined[key])),
            }
            for key in features
        },
        "error_summary": {
            key: {
                "sum": float(np.sum(combined[key])),
                "median": float(np.median(combined[key])),
                "p95": float(np.quantile(combined[key], 0.95)),
                "p99": float(np.quantile(combined[key], 0.99)),
                "max": float(np.max(combined[key])),
            }
            for key in (
                "gate_error_sq",
                "intermediate_error_sq",
                error_key,
                "intermediate_error_max_abs",
            )
        },
        "threshold_sweeps": sweeps,
        "best_at_or_below_25_percent_repair": best_bounded,
    }


def main() -> int:
    args = _parse_args()
    reader = GGUFReader(args.model)
    tokenizer = LagunaGGUFTokenizer.from_gguf_info(reader.info)
    prompts = _extend_prompt_streams(
        _load_prompts(args.prompts, tokenizer),
        _ROWS,
    )
    selected = _select_prompts(
        prompts,
        prompt_ids=args.prompt_id,
        prompt_count=args.prompt_count,
    )
    runtime = get_hip_runtime()
    previous_f16 = os.environ.get("HIPENGINE_LAGUNA_F16_PREFILL")
    os.environ["HIPENGINE_LAGUNA_F16_PREFILL"] = "wmma_comp_swa"
    owner: LagunaGGUFResidentSession | None = None
    collector = _RoleRiskCollector(runtime)
    started = time.perf_counter()
    try:
        owner = LagunaGGUFResidentSession(
            args.model,
            context_length=_ROWS,
            backend="hip_gfx1151",
            runtime=runtime,
            compiler_version=_compiler_version(args.compiler_version_file),
            require_cached_build=args.require_cached_build,
            progress=_progress,
            repacked_cache=args.repacked_cache,
            model_sha256=args.model_sha256,
            prefill_chunk_size=_ROWS,
            prefill_attention_chunk_size=_ATTENTION_ROWS,
            global_prefill_variant=_PRODUCTION_GLOBAL,
            swa_prefill_variant=_PRODUCTION_SWA,
        )
        collector.install()
        for prompt in selected:
            assert owner.weights is not None
            child = LagunaGGUFResidentSession(
                resident_weights=owner.weights,
                context_length=_ROWS,
                backend="hip_gfx1151",
                runtime=runtime,
                compiler_version=_compiler_version(
                    args.compiler_version_file
                ),
                require_cached_build=True,
                prefill_chunk_size=_ROWS,
                prefill_attention_chunk_size=_ATTENTION_ROWS,
                global_prefill_variant=_PRODUCTION_GLOBAL,
                swa_prefill_variant=_PRODUCTION_SWA,
            )
            try:
                child.set_selected_gate_up_mode(_PRODUCTION_GATE_UP)
                child.set_selected_down_mode(_PRODUCTION_DOWN)
                child.set_f16_prefill_mode(_PRODUCTION_F16)
                child.set_dense_q4_prefill_mode(_PRODUCTION_DENSE)
                child.set_prefill_attention_hipblaslt(True)
                collector.prompt_id = str(prompt["id"])
                child.prefill(prompt["token_ids"], use_bulk=True)
                runtime.device_synchronize()
            finally:
                child.close()
    finally:
        collector.restore()
        if owner is not None:
            owner.close()
        if previous_f16 is None:
            os.environ.pop("HIPENGINE_LAGUNA_F16_PREFILL", None)
        else:
            os.environ["HIPENGINE_LAGUNA_F16_PREFILL"] = previous_f16
    result = _summarize(
        collector.rows,
        prompt_ids=[str(prompt["id"]) for prompt in selected],
        elapsed_seconds=time.perf_counter() - started,
    )
    result["command"] = [str(Path(sys.executable).resolve()), *sys.argv]
    result["model"] = str(args.model)
    result["model_sha256"] = str(args.model_sha256)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
