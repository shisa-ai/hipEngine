#!/usr/bin/env python3
# ruff: noqa: E402
"""Screen Qwen3.5/PARO KV quantization formats on real retained caches.

The harness captures BF16 full-attention K/V distributions, measures host-side
quantize/dequantize reconstruction, and emulates candidate formats by replacing
a BF16 resident cache with reconstructed BF16 values before teacher-forced
decode.  This permits model-logit ranking before a candidate gets production
HIP kernels.  An optional real ``int8_per_token_head`` run anchors how closely
the emulated baseline predicts the current runtime.

This is a bounded diagnostic, not a performance benchmark.  The current-token
K/V row is consumed as BF16 during its own emulated attention step and is
round-tripped immediately afterward, so candidates must still pass the native
runtime quality gate after implementation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    host_array_ptr,
)
from hipengine.kvcache import resolve_kv_policy
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.runtime import PrefillConfig
from hipengine.runtime.qwen35_paro_runner import (
    Qwen35ParoNextTokenRunner,
    Qwen35ParoResidentSession,
)
from scripts.qwen35_paro_bench import _prompt_tokens
from scripts.qwen35_paro_int8_kv_quality_sweep import _compare_logits, _read_logits

DEFAULT_MODEL = Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")
DEFAULT_CANDIDATES = (
    "baseline_max,calibrated_clip,clip_99,group64,group32,group16,"
    "key_group16,value_group16,key_int8_value_bf16,key_bf16_value_int8"
)
FAST_ACCURACY_CANDIDATES = "baseline_max,hadamard_group32,kivi_int8,kvarn_int8"
FAST_SCREEN_PROMPT = (
    "KV cache fidelity probe. Retrieve the prior fact, solve 17 * 23, explain a Python iterator, "
    "respond briefly in English and 日本語, and preserve the JSON tool argument {\"limit\": 8}. "
)


@dataclass(frozen=True)
class FormatSpec:
    name: str
    k_mode: str = "int8"
    v_mode: str = "int8"
    k_group_size: int = 256
    v_group_size: int = 256
    k_clip_ratio: float = 1.0
    v_clip_ratio: float = 1.0
    strategy: str = "groupwise"
    chunk_size: int = 0
    residual_tokens: int = 0
    sink_tokens: int = 0
    hadamard_group_size: int = 0

    def __post_init__(self) -> None:
        if self.k_mode not in {"int8", "bf16"} or self.v_mode not in {"int8", "bf16"}:
            raise ValueError("K/V modes must be int8 or bf16")
        if self.strategy not in {"groupwise", "hadamard_groupwise", "kivi", "kvarn"}:
            raise ValueError(f"unsupported format strategy {self.strategy!r}")
        if self.k_group_size <= 0 or self.v_group_size <= 0:
            raise ValueError("group sizes must be positive")
        if not (0.0 < self.k_clip_ratio <= 1.0) or not (0.0 < self.v_clip_ratio <= 1.0):
            raise ValueError("clip ratios must be in (0, 1]")
        if self.strategy in {"kivi", "kvarn"} and self.chunk_size <= 0:
            raise ValueError(f"{self.strategy} requires a positive chunk_size")
        if self.residual_tokens < 0 or self.sink_tokens < 0:
            raise ValueError("residual_tokens and sink_tokens must be non-negative")
        if self.strategy in {"hadamard_groupwise", "kvarn"} and self.hadamard_group_size <= 0:
            raise ValueError(f"{self.strategy} requires a positive hadamard_group_size")

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "k_mode": self.k_mode,
            "v_mode": self.v_mode,
            "k_group_size": self.k_group_size if self.k_mode == "int8" else None,
            "v_group_size": self.v_group_size if self.v_mode == "int8" else None,
            "k_clip_ratio": self.k_clip_ratio if self.k_mode == "int8" else None,
            "v_clip_ratio": self.v_clip_ratio if self.v_mode == "int8" else None,
            "chunk_size": self.chunk_size or None,
            "residual_tokens": self.residual_tokens or None,
            "sink_tokens": self.sink_tokens or None,
            "hadamard_group_size": self.hadamard_group_size or None,
        }


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def _quantize_dequantize(
    values: np.ndarray,
    *,
    group_size: int,
    clip_ratio: float,
    scale_dtype: str,
) -> np.ndarray:
    """Emulate writer semantics: float scale for codes, stored scale for restore."""

    source = np.asarray(values, dtype=np.float32)
    if source.ndim < 1 or source.shape[-1] % int(group_size):
        raise ValueError("group_size must divide the last dimension")
    if not (0.0 < float(clip_ratio) <= 1.0):
        raise ValueError("clip_ratio must be in (0, 1]")
    scale_type = np.float16 if scale_dtype == "fp16" else np.float32
    if scale_dtype not in {"fp16", "fp32"}:
        raise ValueError("scale_dtype must be fp16 or fp32")
    grouped = source.reshape(*source.shape[:-1], source.shape[-1] // int(group_size), int(group_size))
    absmax = np.max(np.abs(grouped), axis=-1, keepdims=True)
    float_scale = absmax * np.float32(clip_ratio / 127.0)
    safe_scale = np.where(float_scale > 0.0, float_scale, np.float32(1.0))
    codes = np.clip(np.rint(grouped / safe_scale), -127.0, 127.0)
    codes = np.where(float_scale > 0.0, codes, np.float32(0.0))
    stored_scale = float_scale.astype(scale_type).astype(np.float32)
    return np.ascontiguousarray((codes * stored_scale).reshape(source.shape), dtype=np.float32)


def _normalized_hadamard(values: np.ndarray, *, group_size: int) -> np.ndarray:
    """Apply a normalized Walsh-Hadamard transform to last-axis groups."""

    source = np.asarray(values, dtype=np.float32)
    size = int(group_size)
    if size <= 0 or size & (size - 1):
        raise ValueError("Hadamard group_size must be a positive power of two")
    if source.ndim < 1 or source.shape[-1] % size:
        raise ValueError("Hadamard group_size must divide the last dimension")
    transformed = source.reshape(*source.shape[:-1], source.shape[-1] // size, size).copy()
    width = 1
    while width < size:
        for start in range(0, size, 2 * width):
            left = transformed[..., start : start + width].copy()
            right = transformed[..., start + width : start + 2 * width].copy()
            transformed[..., start : start + width] = left + right
            transformed[..., start + width : start + 2 * width] = left - right
        width *= 2
    transformed *= np.float32(1.0 / math.sqrt(size))
    return np.ascontiguousarray(transformed.reshape(source.shape), dtype=np.float32)


def _asymmetric_quantize_dequantize(
    values: np.ndarray,
    *,
    group_size: int,
    scale_dtype: str,
) -> np.ndarray:
    """Per-row/group affine UINT8 emulation with rounded scale and minimum."""

    source = np.asarray(values, dtype=np.float32)
    if source.ndim < 1 or source.shape[-1] % int(group_size):
        raise ValueError("group_size must divide the last dimension")
    if scale_dtype not in {"fp16", "fp32"}:
        raise ValueError("scale_dtype must be fp16 or fp32")
    scale_type = np.float16 if scale_dtype == "fp16" else np.float32
    grouped = source.reshape(*source.shape[:-1], source.shape[-1] // int(group_size), int(group_size))
    minimum = np.min(grouped, axis=-1, keepdims=True)
    maximum = np.max(grouped, axis=-1, keepdims=True)
    float_scale = (maximum - minimum) / np.float32(255.0)
    safe_scale = np.where(float_scale > 0.0, float_scale, np.float32(1.0))
    codes = np.clip(np.rint((grouped - minimum) / safe_scale), 0.0, 255.0)
    codes = np.where(float_scale > 0.0, codes, np.float32(0.0))
    stored_scale = float_scale.astype(scale_type).astype(np.float32)
    stored_minimum = minimum.astype(scale_type).astype(np.float32)
    return np.ascontiguousarray((codes * stored_scale + stored_minimum).reshape(source.shape), dtype=np.float32)


def _channel_chunk_quantize_dequantize(values: np.ndarray, *, scale_dtype: str) -> np.ndarray:
    """Affine INT8 over the token axis independently for every head/channel."""

    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 3:
        raise ValueError("channel-chunk quantization expects [tokens, heads, channels]")
    if scale_dtype not in {"fp16", "fp32"}:
        raise ValueError("scale_dtype must be fp16 or fp32")
    scale_type = np.float16 if scale_dtype == "fp16" else np.float32
    minimum = np.min(source, axis=0, keepdims=True)
    maximum = np.max(source, axis=0, keepdims=True)
    float_scale = (maximum - minimum) / np.float32(255.0)
    safe_scale = np.where(float_scale > 0.0, float_scale, np.float32(1.0))
    codes = np.clip(np.rint((source - minimum) / safe_scale), 0.0, 255.0)
    codes = np.where(float_scale > 0.0, codes, np.float32(0.0))
    stored_scale = float_scale.astype(scale_type).astype(np.float32)
    stored_minimum = minimum.astype(scale_type).astype(np.float32)
    return np.ascontiguousarray(codes * stored_scale + stored_minimum, dtype=np.float32)


def _chunk_slices(rows: int, chunk_size: int) -> list[slice]:
    full_rows = int(rows) // int(chunk_size) * int(chunk_size)
    return [slice(start, start + int(chunk_size)) for start in range(0, full_rows, int(chunk_size))]


def _kivi_roundtrip_pair(
    key: np.ndarray,
    value: np.ndarray,
    spec: FormatSpec,
    *,
    scale_dtype: str,
) -> tuple[np.ndarray, np.ndarray]:
    key_out = np.asarray(key, dtype=np.float32).copy()
    value_out = np.asarray(value, dtype=np.float32).copy()
    for chunk in _chunk_slices(key_out.shape[0], spec.chunk_size):
        key_out[chunk] = _channel_chunk_quantize_dequantize(key_out[chunk], scale_dtype=scale_dtype)
        value_out[chunk] = _asymmetric_quantize_dequantize(
            value_out[chunk],
            group_size=spec.v_group_size,
            scale_dtype=scale_dtype,
        )
    return key_out, value_out


def _variance_imbalance(tile: np.ndarray) -> float:
    column_std = np.std(tile, axis=0, ddof=1)
    row_std = np.std(tile, axis=1, ddof=1)
    return float(
        np.max(column_std) / max(float(np.min(column_std)), 1e-8)
        + np.max(row_std) / max(float(np.min(row_std)), 1e-8)
    )


def _variance_normalize(
    tile: np.ndarray,
    *,
    iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best-so-far log-domain row/column variance normalization.

    This NumPy screen follows the algorithm and constants in
    huawei-csl/KVarN@7586257f ``kvarn/sinkhorn.py`` without introducing a
    Torch/runtime dependency.
    """

    source = np.asarray(tile, dtype=np.float32)
    if source.ndim != 2 or min(source.shape) < 2:
        raise ValueError("variance normalization expects a 2D tile with both axes >= 2")
    if int(iterations) <= 0:
        raise ValueError("variance normalization iterations must be positive")
    log_column_scale = np.zeros((1, source.shape[1]), dtype=np.float32)
    log_row_scale = np.zeros((source.shape[0], 1), dtype=np.float32)
    current = source.copy()
    best_imbalance = _variance_imbalance(current)
    best_column_scale = np.ones_like(log_column_scale)
    best_row_scale = np.ones_like(log_row_scale)
    for _ in range(int(iterations)):
        column_std = np.clip(np.std(current, axis=0, ddof=1, keepdims=True), 1e-3, 1e3)
        log_column_scale = np.clip(log_column_scale + np.log(column_std), -0.3, 10.0)
        current = source / np.exp(log_column_scale) / np.exp(log_row_scale)
        row_std = np.clip(np.std(current, axis=1, ddof=1, keepdims=True), 1e-3, 1e3)
        log_row_scale = np.clip(log_row_scale + np.log(row_std), -0.3, 10.0)
        current = source / np.exp(log_column_scale) / np.exp(log_row_scale)
        imbalance = _variance_imbalance(current)
        if imbalance <= best_imbalance:
            best_imbalance = imbalance
            best_column_scale = np.exp(log_column_scale).astype(np.float32)
            best_row_scale = np.exp(log_row_scale).astype(np.float32)
    balanced = source / best_column_scale / best_row_scale
    return (
        np.ascontiguousarray(balanced, dtype=np.float32),
        best_column_scale,
        best_row_scale,
    )


def _kvarn_affine_reconstruct(
    balanced: np.ndarray,
    column_scale: np.ndarray,
    row_scale: np.ndarray,
    *,
    scale_dtype: str,
) -> np.ndarray:
    if scale_dtype not in {"fp16", "fp32"}:
        raise ValueError("scale_dtype must be fp16 or fp32")
    scale_type = np.float16 if scale_dtype == "fp16" else np.float32
    minimum = np.min(balanced, axis=1, keepdims=True)
    maximum = np.max(balanced, axis=1, keepdims=True)
    float_scale = (maximum - minimum) / np.float32(255.0)
    safe_scale = np.where(float_scale > 0.0, float_scale, np.float32(1.0))
    codes = np.clip(np.rint((balanced - minimum) / safe_scale), 0.0, 255.0)
    codes = np.where(float_scale > 0.0, codes, np.float32(0.0))
    absorbed_scale = (row_scale * float_scale).astype(scale_type).astype(np.float32)
    absorbed_minimum = (row_scale * minimum).astype(scale_type).astype(np.float32)
    stored_column_scale = column_scale.astype(scale_type).astype(np.float32)
    return np.ascontiguousarray(
        (codes * absorbed_scale + absorbed_minimum) * stored_column_scale,
        dtype=np.float32,
    )


def _kvarn_chunk_roundtrip(
    values: np.ndarray,
    *,
    component: str,
    hadamard_group_size: int,
    scale_dtype: str,
    sinkhorn_iterations: int = 8,
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 3:
        raise ValueError("KVarN-inspired quantization expects [tokens, heads, channels]")
    if component not in {"key", "value"}:
        raise ValueError("KVarN component must be key or value")
    rotated = _normalized_hadamard(source, group_size=hadamard_group_size)
    reconstructed = np.empty_like(rotated)
    for head in range(rotated.shape[1]):
        token_channel = rotated[:, head, :]
        tile = token_channel.T if component == "key" else token_channel
        balanced, column_scale, row_scale = _variance_normalize(
            tile,
            iterations=sinkhorn_iterations,
        )
        restored = _kvarn_affine_reconstruct(
            balanced,
            column_scale,
            row_scale,
            scale_dtype=scale_dtype,
        )
        reconstructed[:, head, :] = restored.T if component == "key" else restored
    return _normalized_hadamard(reconstructed, group_size=hadamard_group_size)


def _kvarn_roundtrip_pair(
    key: np.ndarray,
    value: np.ndarray,
    spec: FormatSpec,
    *,
    scale_dtype: str,
) -> tuple[np.ndarray, np.ndarray]:
    key_out = np.asarray(key, dtype=np.float32).copy()
    value_out = np.asarray(value, dtype=np.float32).copy()
    quantized_rows = max(0, key_out.shape[0] - int(spec.sink_tokens))
    for relative_chunk in _chunk_slices(quantized_rows, spec.chunk_size):
        chunk = slice(
            int(spec.sink_tokens) + int(relative_chunk.start or 0),
            int(spec.sink_tokens) + int(relative_chunk.stop or 0),
        )
        key_out[chunk] = _kvarn_chunk_roundtrip(
            key_out[chunk],
            component="key",
            hadamard_group_size=spec.hadamard_group_size,
            scale_dtype=scale_dtype,
        )
        value_out[chunk] = _kvarn_chunk_roundtrip(
            value_out[chunk],
            component="value",
            hadamard_group_size=spec.hadamard_group_size,
            scale_dtype=scale_dtype,
        )
    return key_out, value_out


def _percentiles(values: np.ndarray) -> dict[str, float]:
    labels = ("50", "90", "99", "99.9", "100")
    points = (50.0, 90.0, 99.0, 99.9, 100.0)
    if values.size == 0:
        return {label: 0.0 for label in labels}
    result = np.percentile(np.asarray(values, dtype=np.float32), points)
    return {label: float(value) for label, value in zip(labels, result, strict=True)}


def _distribution_summary(values: np.ndarray) -> dict[str, Any]:
    source = np.asarray(values, dtype=np.float32)
    row_absmax = np.max(np.abs(source), axis=-1) if source.size else np.empty((0,), dtype=np.float32)
    return {
        "elements": int(source.size),
        "mean": float(np.mean(source)) if source.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(source, dtype=np.float64)))) if source.size else 0.0,
        "abs_percentiles": _percentiles(np.abs(source).reshape(-1)),
        "row_absmax_percentiles": _percentiles(row_absmax.reshape(-1)),
    }


def _reconstruction_summary(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    expected = np.asarray(reference, dtype=np.float32)
    actual = np.asarray(candidate, dtype=np.float32)
    if expected.shape != actual.shape:
        raise ValueError("reconstruction arrays must have equal shapes")
    error = actual.astype(np.float64) - expected.astype(np.float64)
    mse = float(np.mean(np.square(error))) if error.size else 0.0
    ref_mse = float(np.mean(np.square(expected.astype(np.float64)))) if expected.size else 0.0
    return {
        "elements": int(expected.size),
        "rmse": float(math.sqrt(mse)),
        "normalized_rmse": float(math.sqrt(mse / ref_mse)) if ref_mse > 0.0 else 0.0,
        "mean_abs_error": float(np.mean(np.abs(error))) if error.size else 0.0,
        "max_abs_error": float(np.max(np.abs(error))) if error.size else 0.0,
    }


def _candidate_catalog(head_dim: int) -> dict[str, FormatSpec]:
    h = int(head_dim)
    return {
        "baseline_max": FormatSpec("baseline_max", k_group_size=h, v_group_size=h),
        "calibrated_clip": FormatSpec("calibrated_clip", k_group_size=h, v_group_size=h),
        "clip_99": FormatSpec("clip_99", k_group_size=h, v_group_size=h, k_clip_ratio=0.99, v_clip_ratio=0.99),
        "clip_98": FormatSpec("clip_98", k_group_size=h, v_group_size=h, k_clip_ratio=0.98, v_clip_ratio=0.98),
        "group64": FormatSpec("group64", k_group_size=64, v_group_size=64),
        "group32": FormatSpec("group32", k_group_size=32, v_group_size=32),
        "group16": FormatSpec("group16", k_group_size=16, v_group_size=16),
        "hadamard_group32": FormatSpec(
            "hadamard_group32",
            k_group_size=32,
            v_group_size=32,
            strategy="hadamard_groupwise",
            hadamard_group_size=32,
        ),
        "kivi_int8": FormatSpec(
            "kivi_int8",
            v_group_size=32,
            strategy="kivi",
            chunk_size=128,
            residual_tokens=127,
        ),
        "kvarn_int8": FormatSpec(
            "kvarn_int8",
            strategy="kvarn",
            chunk_size=128,
            residual_tokens=127,
            sink_tokens=128,
            hadamard_group_size=h,
        ),
        "key_group16": FormatSpec("key_group16", k_group_size=16, v_group_size=h),
        "value_group16": FormatSpec("value_group16", k_group_size=h, v_group_size=16),
        "key_group32_value_group16": FormatSpec(
            "key_group32_value_group16", k_group_size=32, v_group_size=16
        ),
        "key_group16_value_group32": FormatSpec(
            "key_group16_value_group32", k_group_size=16, v_group_size=32
        ),
        "key_int8_value_bf16": FormatSpec(
            "key_int8_value_bf16", k_mode="int8", v_mode="bf16", k_group_size=h, v_group_size=h
        ),
        "key_group32_value_bf16": FormatSpec(
            "key_group32_value_bf16", k_mode="int8", v_mode="bf16", k_group_size=32, v_group_size=h
        ),
        "key_group16_value_bf16": FormatSpec(
            "key_group16_value_bf16", k_mode="int8", v_mode="bf16", k_group_size=16, v_group_size=h
        ),
        "key_hadamard_group32_value_bf16": FormatSpec(
            "key_hadamard_group32_value_bf16",
            k_mode="int8",
            v_mode="bf16",
            k_group_size=32,
            v_group_size=h,
            strategy="hadamard_groupwise",
            hadamard_group_size=32,
        ),
        "key_bf16_value_int8": FormatSpec(
            "key_bf16_value_int8", k_mode="bf16", v_mode="int8", k_group_size=h, v_group_size=h
        ),
        "key_bf16_value_group32": FormatSpec(
            "key_bf16_value_group32", k_mode="bf16", v_mode="int8", k_group_size=h, v_group_size=32
        ),
    }


def _parse_candidates(text: str, *, head_dim: int) -> list[FormatSpec]:
    catalog = _candidate_catalog(head_dim)
    names = [item.strip() for item in text.split(",") if item.strip()]
    if not names:
        raise ValueError("expected at least one candidate")
    unknown = [name for name in names if name not in catalog]
    if unknown:
        raise ValueError(f"unknown candidates: {', '.join(unknown)}")
    candidates = [catalog[name] for name in names]
    for candidate in candidates:
        for mode, group in ((candidate.k_mode, candidate.k_group_size), (candidate.v_mode, candidate.v_group_size)):
            if mode == "int8" and int(head_dim) % int(group):
                raise ValueError(f"candidate {candidate.name} group {group} does not divide head_dim {head_dim}")
        if candidate.hadamard_group_size and int(head_dim) % int(candidate.hadamard_group_size):
            raise ValueError(
                f"candidate {candidate.name} Hadamard group {candidate.hadamard_group_size} "
                f"does not divide head_dim {head_dim}"
            )
    return candidates


def _component_memory_bytes(
    *, mode: str, tokens: int, full_layers: int, num_kv_heads: int, head_dim: int, group_size: int, scale_bytes: int
) -> tuple[int, int]:
    elements = int(tokens) * int(full_layers) * int(num_kv_heads) * int(head_dim)
    if mode == "bf16":
        return 2 * elements, 0
    groups = int(head_dim) // int(group_size)
    scales = int(tokens) * int(full_layers) * int(num_kv_heads) * groups
    return elements, scales * int(scale_bytes)


def _format_memory_bytes(
    spec: FormatSpec,
    *,
    tokens: int,
    full_layers: int,
    num_kv_heads: int,
    head_dim: int,
    scale_dtype: str,
) -> dict[str, int]:
    scale_bytes = 2 if scale_dtype == "fp16" else 4
    elements = int(tokens) * int(full_layers) * int(num_kv_heads) * int(head_dim)
    residual_bytes = 0
    if spec.strategy in {"groupwise", "hadamard_groupwise"}:
        kp, ks = _component_memory_bytes(
            mode=spec.k_mode,
            tokens=tokens,
            full_layers=full_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            group_size=spec.k_group_size,
            scale_bytes=scale_bytes,
        )
        vp, vs = _component_memory_bytes(
            mode=spec.v_mode,
            tokens=tokens,
            full_layers=full_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            group_size=spec.v_group_size,
            scale_bytes=scale_bytes,
        )
        payload_bytes = kp + vp
        metadata_bytes = ks + vs
    elif spec.strategy == "kivi":
        chunks = math.ceil(int(tokens) / int(spec.chunk_size))
        payload_bytes = 2 * elements
        key_metadata = chunks * int(full_layers) * int(num_kv_heads) * int(head_dim) * 2 * scale_bytes
        value_groups = int(head_dim) // int(spec.v_group_size)
        value_metadata = int(tokens) * int(full_layers) * int(num_kv_heads) * value_groups * 2 * scale_bytes
        metadata_bytes = key_metadata + value_metadata
        residual_bytes = (
            2 * int(spec.residual_tokens) * int(full_layers) * int(num_kv_heads) * int(head_dim) * 2
        )
    elif spec.strategy == "kvarn":
        quantized_tokens = max(0, int(tokens) - int(spec.sink_tokens))
        chunks = math.ceil(quantized_tokens / int(spec.chunk_size))
        payload_bytes = 2 * elements
        # Per tile: K stores 2*D + T metadata values; V stores D + 2*T.
        metadata_values_per_tile = 3 * (int(head_dim) + int(spec.chunk_size))
        metadata_bytes = (
            chunks * int(full_layers) * int(num_kv_heads) * metadata_values_per_tile * scale_bytes
        )
        residual_bytes = (
            2
            * (int(spec.sink_tokens) + int(spec.residual_tokens))
            * int(full_layers)
            * int(num_kv_heads)
            * int(head_dim)
            * 2
        )
    else:  # pragma: no cover - guarded by FormatSpec
        raise ValueError(f"unsupported strategy {spec.strategy!r}")
    return {
        "payload_bytes": int(payload_bytes),
        "scale_bytes": int(metadata_bytes),
        "residual_bytes": int(residual_bytes),
        "total_bytes": int(payload_bytes + metadata_bytes + residual_bytes),
    }


def _roundtrip_component(values: np.ndarray, *, mode: str, group_size: int, clip_ratio: float, scale_dtype: str) -> np.ndarray:
    if mode == "bf16":
        return np.asarray(values, dtype=np.float32).copy()
    return _quantize_dequantize(values, group_size=group_size, clip_ratio=clip_ratio, scale_dtype=scale_dtype)


def _roundtrip_pair(key: np.ndarray, value: np.ndarray, spec: FormatSpec, *, scale_dtype: str) -> tuple[np.ndarray, np.ndarray]:
    if spec.strategy == "kivi":
        return _kivi_roundtrip_pair(key, value, spec, scale_dtype=scale_dtype)
    if spec.strategy == "kvarn":
        return _kvarn_roundtrip_pair(key, value, spec, scale_dtype=scale_dtype)
    if spec.strategy == "hadamard_groupwise":
        def roundtrip_hadamard_component(
            values: np.ndarray,
            *,
            mode: str,
            group_size: int,
            clip_ratio: float,
        ) -> np.ndarray:
            if mode == "bf16":
                return np.asarray(values, dtype=np.float32).copy()
            rotated = _normalized_hadamard(values, group_size=spec.hadamard_group_size)
            restored = _roundtrip_component(
                rotated,
                mode=mode,
                group_size=group_size,
                clip_ratio=clip_ratio,
                scale_dtype=scale_dtype,
            )
            return _normalized_hadamard(restored, group_size=spec.hadamard_group_size)

        return (
            roundtrip_hadamard_component(
                key,
                mode=spec.k_mode,
                group_size=spec.k_group_size,
                clip_ratio=spec.k_clip_ratio,
            ),
            roundtrip_hadamard_component(
                value,
                mode=spec.v_mode,
                group_size=spec.v_group_size,
                clip_ratio=spec.v_clip_ratio,
            ),
        )
    return (
        _roundtrip_component(key, mode=spec.k_mode, group_size=spec.k_group_size, clip_ratio=spec.k_clip_ratio, scale_dtype=scale_dtype),
        _roundtrip_component(value, mode=spec.v_mode, group_size=spec.v_group_size, clip_ratio=spec.v_clip_ratio, scale_dtype=scale_dtype),
    )


def _calibrate_clip(values: Sequence[np.ndarray], *, group_size: int, scale_dtype: str) -> dict[str, Any]:
    ratios = (1.0, 0.999, 0.995, 0.99, 0.98, 0.95)
    rows = []
    for ratio in ratios:
        squared_error = 0.0
        squared_reference = 0.0
        elements = 0
        for value in values:
            restored = _quantize_dequantize(value, group_size=group_size, clip_ratio=ratio, scale_dtype=scale_dtype)
            diff = restored.astype(np.float64) - value.astype(np.float64)
            squared_error += float(np.sum(np.square(diff)))
            squared_reference += float(np.sum(np.square(value.astype(np.float64))))
            elements += int(value.size)
        nrmse = math.sqrt(squared_error / squared_reference) if squared_reference > 0.0 else 0.0
        rows.append({"clip_ratio": ratio, "normalized_rmse": nrmse, "elements": elements})
    best = min(rows, key=lambda row: (float(row["normalized_rmse"]), -float(row["clip_ratio"])))
    return {"selected_clip_ratio": float(best["clip_ratio"]), "grid": rows}


def _capture_cache_sample(session: Qwen35ParoResidentSession, *, tokens: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rows = min(int(tokens), int(session.max_sequence_length))
    row_shape = (int(session.config.num_key_value_heads), int(session.config.head_dim))
    shape = (rows, *row_shape)
    nbytes = int(np.prod(shape)) * np.dtype(np.uint16).itemsize
    keys: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for layer_id in sorted(session.full_caches):
        _key_tensor, _value_tensor, key_buf, value_buf = session.full_caches[layer_id]
        key_bits = np.empty(shape, dtype=np.uint16)
        value_bits = np.empty(shape, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(key_bits), DeviceBuffer(key_buf.ptr, nbytes), nbytes, runtime=session.runtime)
        copy_device_to_host(host_array_ptr(value_bits), DeviceBuffer(value_buf.ptr, nbytes), nbytes, runtime=session.runtime)
        keys.append(_bf16_bits_to_float32(key_bits))
        values.append(_bf16_bits_to_float32(value_bits))
    return keys, values


def _roundtrip_session_cache(
    session: Qwen35ParoResidentSession,
    spec: FormatSpec,
    *,
    start: int,
    rows: int,
    scale_dtype: str,
) -> None:
    if rows <= 0:
        return
    width = int(session.config.num_key_value_heads) * int(session.config.head_dim)
    row_bytes = width * np.dtype(np.uint16).itemsize
    shape = (int(rows), int(session.config.num_key_value_heads), int(session.config.head_dim))
    offset_bytes = int(start) * row_bytes
    nbytes = int(rows) * row_bytes
    for layer_id in sorted(session.full_caches):
        key_tensor, value_tensor, key_buf, value_buf = session.full_caches[layer_id]
        if key_tensor.dtype.value != "bf16" or value_tensor.dtype.value != "bf16":
            raise ValueError("format emulation requires a BF16 resident cache")
        key_bits = np.empty(shape, dtype=np.uint16)
        value_bits = np.empty(shape, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(key_bits), DeviceBuffer(key_buf.ptr + offset_bytes, nbytes), nbytes, runtime=session.runtime)
        copy_device_to_host(host_array_ptr(value_bits), DeviceBuffer(value_buf.ptr + offset_bytes, nbytes), nbytes, runtime=session.runtime)
        key, value = _roundtrip_pair(
            _bf16_bits_to_float32(key_bits),
            _bf16_bits_to_float32(value_bits),
            spec,
            scale_dtype=scale_dtype,
        )
        key_out = float_array_to_bf16_bits(key)
        value_out = float_array_to_bf16_bits(value)
        copy_host_to_device(DeviceBuffer(key_buf.ptr + offset_bytes, nbytes), host_array_ptr(key_out), nbytes, runtime=session.runtime)
        copy_host_to_device(DeviceBuffer(value_buf.ptr + offset_bytes, nbytes), host_array_ptr(value_out), nbytes, runtime=session.runtime)


def _run_loaded_session(
    session: Qwen35ParoResidentSession,
    *,
    prompt_tokens: Sequence[int],
    prompt_length: int,
    decode_steps: int,
    forced_input_ids: Sequence[int] | None,
    scale_dtype: str,
    emulated_spec: FormatSpec | None = None,
    sample_tokens: int = 0,
) -> tuple[dict[str, Any], tuple[list[np.ndarray], list[np.ndarray]] | None, int]:
    """Run one matched case while retaining the session's resident weights."""

    started = time.perf_counter()
    resolved_prompt_tokens = [int(item) for item in prompt_tokens]
    if len(resolved_prompt_tokens) != int(prompt_length):
        raise ValueError("prompt_tokens length must equal prompt_length")
    if forced_input_ids is not None and len(forced_input_ids) != int(decode_steps):
        raise ValueError("forced_input_ids must contain exactly decode_steps tokens")
    session.reset()
    session._resolve_prefill_config_for_length(int(prompt_length))
    seed = session.prefill_native(resolved_prompt_tokens, sample=True)
    if seed is None:
        raise RuntimeError("native prefill did not produce a seed")
    logits = [_read_logits(session)]
    captured = _capture_cache_sample(session, tokens=min(prompt_length, sample_tokens)) if sample_tokens > 0 else None
    if emulated_spec is not None:
        _roundtrip_session_cache(session, emulated_spec, start=0, rows=prompt_length, scale_dtype=scale_dtype)
    generated_ids: list[int] = []
    current = seed
    for offset in range(decode_steps):
        input_id = int(current.token_id) if forced_input_ids is None else int(forced_input_ids[offset])
        current = session.step(input_id, position=prompt_length + offset, sample=True)
        if current is None:
            raise RuntimeError(f"decode did not produce token {offset}")
        generated_ids.append(int(current.token_id))
        logits.append(_read_logits(session))
        if emulated_spec is not None:
            _roundtrip_session_cache(
                session,
                emulated_spec,
                start=prompt_length + offset,
                rows=1,
                scale_dtype=scale_dtype,
            )
    result = {
        "seed_token_id": int(seed.token_id),
        "generated_token_ids": generated_ids,
        "logits": logits,
        "finite_logits": bool(all(np.isfinite(item).all() for item in logits)),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return result, captured, len(session.full_caches)


def _run_session(
    *,
    runner: Qwen35ParoNextTokenRunner,
    model: Path,
    prompt_length: int,
    decode_steps: int,
    token_id: int,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
    prefill_config: PrefillConfig,
    storage: str,
    scale_dtype: str,
    forced_input_ids: Sequence[int] | None,
    prompt_tokens: Sequence[int] | None = None,
    emulated_spec: FormatSpec | None = None,
    sample_tokens: int = 0,
) -> tuple[dict[str, Any], tuple[list[np.ndarray], list[np.ndarray]] | None, int]:
    policy = resolve_kv_policy(storage, block_size=256, scale_dtype=scale_dtype)
    resolved_prompt_tokens = (
        _prompt_tokens(model, "Hello", token_id, prompt_length)
        if prompt_tokens is None
        else [int(item) for item in prompt_tokens]
    )
    max_sequence_length = int(prompt_length) + int(decode_steps) + 2
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=max_layers,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        prefill_config=prefill_config,
        kv_policy=policy.create_policy(),
        kv_scale_dtype=policy.scale_dtype,
        kv_scale_granularity=policy.scale_granularity,
    ) as session:
        result = _run_loaded_session(
            session,
            prompt_tokens=resolved_prompt_tokens,
            prompt_length=prompt_length,
            decode_steps=decode_steps,
            forced_input_ids=forced_input_ids,
            scale_dtype=scale_dtype,
            emulated_spec=emulated_spec,
            sample_tokens=sample_tokens,
        )
    gc.collect()
    return result


def _compact_run(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key != "logits"}


def _aggregate_reconstruction(
    keys: Sequence[np.ndarray], values: Sequence[np.ndarray], spec: FormatSpec, *, scale_dtype: str
) -> dict[str, Any]:
    restored = [
        _roundtrip_pair(key, value, spec, scale_dtype=scale_dtype)
        for key, value in zip(keys, values, strict=True)
    ]
    key_ref = np.concatenate([item.reshape(-1, item.shape[-1]) for item in keys], axis=0)
    value_ref = np.concatenate([item.reshape(-1, item.shape[-1]) for item in values], axis=0)
    key_out = np.concatenate([item[0].reshape(-1, item[0].shape[-1]) for item in restored], axis=0)
    value_out = np.concatenate([item[1].reshape(-1, item[1].shape[-1]) for item in restored], axis=0)
    return {
        "key": _reconstruction_summary(key_ref, key_out),
        "value": _reconstruction_summary(value_ref, value_out),
    }


def _select_recommendation(rows: Sequence[dict[str, Any]], *, extra_budget_bytes: int) -> dict[str, Any]:
    fit = [row for row in rows if int(row["extra_bytes_over_baseline"]) <= int(extra_budget_bytes)]
    if not fit:
        return {"name": None, "fit_candidates": [], "reason": "no candidate fits extra-byte budget"}
    best = min(
        fit,
        key=lambda row: (
            float(row["logit_gate"]["mean_kl"]),
            -float(row["logit_gate"]["top1_agreement"]),
            int(row["extra_bytes_over_baseline"]),
        ),
    )
    return {
        "name": str(best["name"]),
        "fit_candidates": [str(row["name"]) for row in fit],
        "reason": "lowest matched-context mean KL, then highest top-1, among formats fitting the extra-byte budget",
        "mean_kl": float(best["logit_gate"]["mean_kl"]),
        "top1_agreement": float(best["logit_gate"]["top1_agreement"]),
        "extra_bytes_over_baseline": int(best["extra_bytes_over_baseline"]),
    }


def _git_provenance() -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, capture_output=True, check=True).stdout.strip())
    return {"hipengine_commit": commit, "dirty": dirty}


def _read_compiler_version(path: Path | None) -> str | None:
    return None if path is None else path.read_text(encoding="utf-8")


def _run_bf16_candidate_screen(
    *,
    runner: Qwen35ParoNextTokenRunner,
    args: argparse.Namespace,
    compiler_version: str | None,
    prefill_config: PrefillConfig,
    prompt_tokens: Sequence[int],
    candidates: Sequence[FormatSpec],
) -> dict[str, Any]:
    """Load BF16 model state once, then reset only request state per candidate."""

    policy = resolve_kv_policy("bf16", block_size=256, scale_dtype=args.scale_dtype)
    max_sequence_length = int(args.prompt_length) + int(args.decode_steps) + 2
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
        prefill_config=prefill_config,
        kv_policy=policy.create_policy(),
        kv_scale_dtype=policy.scale_dtype,
        kv_scale_granularity=policy.scale_granularity,
    ) as session:
        reference, captured, full_layers = _run_loaded_session(
            session,
            prompt_tokens=prompt_tokens,
            prompt_length=args.prompt_length,
            decode_steps=args.decode_steps,
            forced_input_ids=None,
            scale_dtype=args.scale_dtype,
            sample_tokens=args.sample_tokens,
        )
        if captured is None or not captured[0]:
            raise RuntimeError("reference cache capture was empty")
        keys, values = captured
        calibration = {
            "key": _calibrate_clip(keys, group_size=int(runner.config.head_dim), scale_dtype=args.scale_dtype),
            "value": _calibrate_clip(values, group_size=int(runner.config.head_dim), scale_dtype=args.scale_dtype),
        }
        calibrated = FormatSpec(
            "calibrated_clip",
            k_group_size=int(runner.config.head_dim),
            v_group_size=int(runner.config.head_dim),
            k_clip_ratio=float(calibration["key"]["selected_clip_ratio"]),
            v_clip_ratio=float(calibration["value"]["selected_clip_ratio"]),
        )
        resolved_candidates = [calibrated if item.name == "calibrated_clip" else item for item in candidates]
        forced_ids = (
            []
            if int(args.decode_steps) == 0
            else [
                int(reference["seed_token_id"]),
                *[int(item) for item in reference["generated_token_ids"][: args.decode_steps - 1]],
            ]
        )
        baseline_memory = _format_memory_bytes(
            _candidate_catalog(int(runner.config.head_dim))["baseline_max"],
            tokens=args.target_context_tokens,
            full_layers=full_layers,
            num_kv_heads=int(runner.config.num_key_value_heads),
            head_dim=int(runner.config.head_dim),
            scale_dtype=args.scale_dtype,
        )
        rows: list[dict[str, Any]] = []
        for spec in resolved_candidates:
            candidate_run, _, _ = _run_loaded_session(
                session,
                prompt_tokens=prompt_tokens,
                prompt_length=args.prompt_length,
                decode_steps=args.decode_steps,
                forced_input_ids=forced_ids,
                scale_dtype=args.scale_dtype,
                emulated_spec=spec,
            )
            memory = _format_memory_bytes(
                spec,
                tokens=args.target_context_tokens,
                full_layers=full_layers,
                num_kv_heads=int(runner.config.num_key_value_heads),
                head_dim=int(runner.config.head_dim),
                scale_dtype=args.scale_dtype,
            )
            gate = _compare_logits(reference["logits"], candidate_run["logits"])
            rows.append(
                {
                    **spec.to_json(),
                    "emulation_semantics": (
                        "BF16 cache replaced by format reconstruction before decode; "
                        "current row round-tripped after its own attention"
                    ),
                    "logit_gate": gate,
                    "quality_gate_passed": bool(
                        gate["mean_kl"] <= args.kl_threshold
                        and gate["top1_agreement"] >= args.top1_threshold
                    ),
                    "reconstruction": _aggregate_reconstruction(
                        keys,
                        values,
                        spec,
                        scale_dtype=args.scale_dtype,
                    ),
                    "target_context_memory": memory,
                    "extra_bytes_over_baseline": int(memory["total_bytes"] - baseline_memory["total_bytes"]),
                    "candidate": _compact_run(candidate_run),
                }
            )
    gc.collect()
    return {
        "reference": reference,
        "keys": keys,
        "values": values,
        "full_layers": full_layers,
        "calibration": calibration,
        "baseline_memory": baseline_memory,
        "rows": rows,
        "forced_ids": forced_ids,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    compiler_version = _read_compiler_version(args.compiler_version_file)
    prefill_config = PrefillConfig(attn_aotriton_min_tokens=int(args.attn_aotriton_min_tokens))
    runner = Qwen35ParoNextTokenRunner(
        args.model,
        shared_expert_format=None if args.shared_expert_format == "auto" else args.shared_expert_format,
        backend=args.backend,
    )
    head_dim = int(runner.config.head_dim)
    candidates = _parse_candidates(args.candidates, head_dim=head_dim)
    prompt_text = FAST_SCREEN_PROMPT if args.prompt_profile == "mixed_v1" else "Hello"
    prompt_token_id = None if args.prompt_profile == "mixed_v1" else args.token_id
    prompt_tokens = _prompt_tokens(args.model, prompt_text, prompt_token_id, args.prompt_length)
    prompt_digest = hashlib.sha256(np.asarray(prompt_tokens, dtype=np.int32).tobytes()).hexdigest()
    screen = _run_bf16_candidate_screen(
        runner=runner,
        args=args,
        compiler_version=compiler_version,
        prefill_config=prefill_config,
        prompt_tokens=prompt_tokens,
        candidates=candidates,
    )
    reference = screen["reference"]
    keys = screen["keys"]
    values = screen["values"]
    full_layers = int(screen["full_layers"])
    calibration = screen["calibration"]
    baseline_memory = screen["baseline_memory"]
    rows = screen["rows"]
    forced_ids = screen["forced_ids"]

    runtime_int8 = None
    if args.run_runtime_int8_baseline:
        runtime_run, _, _ = _run_session(
            runner=runner,
            model=args.model,
            prompt_length=args.prompt_length,
            decode_steps=args.decode_steps,
            token_id=args.token_id,
            max_layers=args.max_layers,
            compiler_version=compiler_version,
            require_cached_build=args.require_cached_build,
            prefill_config=prefill_config,
            storage="int8_per_token_head",
            scale_dtype=args.scale_dtype,
            forced_input_ids=forced_ids,
            prompt_tokens=prompt_tokens,
        )
        runtime_int8 = {
            "logit_gate": _compare_logits(reference["logits"], runtime_run["logits"]),
            "run": _compact_run(runtime_run),
        }
    extra_budget_bytes = int(float(args.extra_budget_gib) * 1024**3)
    recommendation = _select_recommendation(rows, extra_budget_bytes=extra_budget_bytes)
    elapsed_seconds = float(time.perf_counter() - started)
    return {
        "schema": 1,
        "status": "diagnostic_complete",
        "performance_claim": False,
        "mode": "qwen35_paro_kv_format_ablation",
        "provenance": _git_provenance(),
        "model": str(args.model),
        "backend": runner.backend,
        "target_arch": runner.target_arch,
        "workload": {
            "prompt_length": int(args.prompt_length),
            "decode_steps": int(args.decode_steps),
            "prompt_profile": args.prompt_profile,
            "token_id": None if args.prompt_profile == "mixed_v1" else int(args.token_id),
            "prompt_token_sha256": prompt_digest,
            "distinct_prompt_tokens": int(len(set(prompt_tokens))),
        },
        "screen_budget": {
            "profile": "fast_accuracy_v1" if args.fast_accuracy_screen else "custom",
            "wall_time_budget_seconds": float(args.wall_time_budget_seconds),
            "elapsed_seconds": elapsed_seconds,
            "met": bool(elapsed_seconds <= float(args.wall_time_budget_seconds)),
            "candidate_count": len(rows),
            "model_setup": "one BF16 resident-weight session reset and reused across reference and emulated candidates",
        },
        "shape": {
            "max_layers": int(args.max_layers),
            "full_attention_layers": int(full_layers),
            "num_kv_heads": int(runner.config.num_key_value_heads),
            "head_dim": head_dim,
            "scale_dtype": args.scale_dtype,
            "sample_tokens": min(int(args.sample_tokens), int(args.prompt_length)),
        },
        "quality_thresholds": {"kl_mean_max": float(args.kl_threshold), "top1_agreement_min": float(args.top1_threshold)},
        "target_memory": {
            "context_tokens": int(args.target_context_tokens),
            "extra_budget_bytes": extra_budget_bytes,
            "baseline": baseline_memory,
        },
        "reference": _compact_run(reference),
        "captured_distribution": {
            "key": _distribution_summary(np.concatenate([item.reshape(-1, head_dim) for item in keys], axis=0)),
            "value": _distribution_summary(np.concatenate([item.reshape(-1, head_dim) for item in values], axis=0)),
        },
        "calibrated_clipping": calibration,
        "runtime_int8_baseline": runtime_int8,
        "candidates": rows,
        "recommendation": recommendation,
        "elapsed_seconds": elapsed_seconds,
        "notes": [
            "Diagnostic only: emulation ranks formats without production candidate kernels.",
            "The emulated current-token K/V row is BF16 during its own attention and quantized immediately afterward.",
            "A candidate must pass native matched-context, long-context utility, memory, and performance gates before retention.",
            "KIVI and KVarN rows are NumPy emulations informed by external research, not paper-faithful production implementations.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--fast-accuracy-screen", action="store_true")
    parser.add_argument("--prompt-profile", choices=("repeated_token", "mixed_v1"), default="repeated_token")
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--sample-tokens", type=int, default=512)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--scale-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--target-context-tokens", type=int, default=262144)
    parser.add_argument("--extra-budget-gib", type=float, default=1.0)
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--top1-threshold", type=float, default=0.90)
    parser.add_argument("--run-runtime-int8-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wall-time-budget-seconds", type=float, default=600.0)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--attn-aotriton-min-tokens", type=int, default=512)
    parser.add_argument("--backend", choices=("auto", "hip_gfx1100", "hip_gfx1151"), default="hip_gfx1100")
    parser.add_argument("--shared-expert-format", choices=("auto", "legacy_fp16", "packed_paro_w4"), default="packed_paro_w4")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.fast_accuracy_screen:
        args.prompt_length = 512
        args.decode_steps = 8
        args.sample_tokens = 512
        args.candidates = FAST_ACCURACY_CANDIDATES
        args.prompt_profile = "mixed_v1"
        args.run_runtime_int8_baseline = False
        args.wall_time_budget_seconds = 600.0
    for name in ("prompt_length", "sample_tokens", "target_context_tokens"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.decode_steps < 0 or args.extra_budget_gib < 0.0 or args.wall_time_budget_seconds <= 0.0:
        raise ValueError("decode steps and extra budget must be non-negative; wall-time budget must be positive")
    payload = run(args)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
