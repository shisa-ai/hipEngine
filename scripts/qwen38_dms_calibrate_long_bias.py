#!/usr/bin/env python3
"""Calibrate per-layer/head DMS bias on disjoint long-context score streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import (
    copy_device_to_host,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kvcache.dms import load_dms_retrofit_config
from hipengine.kvcache.dms_device import DMSExternalLinearDeviceProjector
from hipengine.kvcache.dms_sidecar import (
    ExternalDMSLinearSidecar,
    load_external_dms_sidecar,
)
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFFullStackRunner,
    Qwen35GGUFResidentSession,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    )
    return {"commit": commit, "working_tree_clean": not dirty}


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return np.ascontiguousarray(rounded >> 16, dtype=np.uint16)


def _bf16_float(values: np.ndarray) -> np.ndarray:
    return (_bf16_bits(values).astype(np.uint32) << np.uint32(16)).view(np.float32)


def _write_bf16_safetensors(
    path: Path,
    *,
    bias: np.ndarray,
    weight: np.ndarray,
) -> None:
    tensors = (
        ("bias", _bf16_bits(bias)),
        ("weight", _bf16_bits(weight)),
    )
    offset = 0
    header: dict[str, Any] = {}
    payloads: list[bytes] = []
    for name, bits in tensors:
        raw = bits.astype("<u2", copy=False).tobytes(order="C")
        header[name] = {
            "dtype": "BF16",
            "shape": list(bits.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        payloads.append(raw)
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    padded = encoded + b" " * ((8 - len(encoded) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(padded)))
        handle.write(padded)
        for payload in payloads:
            handle.write(payload)


class _LongScoreCollector:
    """Capture all external-linear logits on device without retaining hidden/Q/K."""

    requires_teacher_logits = False

    def __init__(self, source: ExternalDMSLinearSidecar, *, tokens: int, backend: str) -> None:
        self.source = source
        self.physical_layer_ids = source.config.physical_layer_ids
        self.hidden_size = int(source.config.hidden_size)
        self.num_q_heads = int(source.config.num_q_heads)
        self.num_kv_heads = int(source.config.num_kv_heads)
        self.head_dim = int(source.config.head_dim)
        self.input_stage = str(source.config.input_stage)
        self.tokens = int(tokens)
        self._runtime = get_hip_runtime()
        self._projector = DMSExternalLinearDeviceProjector(source, backend=backend)
        cells = source.config.num_layers * self.tokens * self.num_kv_heads
        self._scores = malloc(cells * 4, runtime=self._runtime)
        self._decisions = malloc(cells, runtime=self._runtime)
        self._next = np.zeros(source.config.num_layers, dtype=np.int32)
        self._closed = False

    def capture_device_chunk(
        self,
        *,
        physical_layer_id: int,
        compact_layer_index: int,
        start: int,
        rows: int,
        hidden_ptr: int,
        stream: int,
    ) -> None:
        layer = int(compact_layer_index)
        if self.source.compact_layer_index(int(physical_layer_id)) != layer:
            raise ValueError("long-score collector physical layer map mismatch")
        if int(start) != int(self._next[layer]):
            raise ValueError("long-score chunks must be contiguous")
        if int(start) + int(rows) > self.tokens:
            raise ValueError("long-score capture exceeds sequence")
        cell_offset = (layer * self.tokens + int(start)) * self.num_kv_heads
        self._projector.project(
            hidden_ptr=int(hidden_ptr),
            compact_layer_index=layer,
            tokens=int(rows),
            logits_ptr=self._scores.ptr + cell_offset * 4,
            evict_ptr=self._decisions.ptr + cell_offset,
            stream=int(stream),
        )
        self._next[layer] = int(start) + int(rows)

    def capture_teacher_logits(self, logits: np.ndarray) -> None:
        del logits

    def finalize(self) -> np.ndarray:
        if np.any(self._next != self.tokens):
            raise RuntimeError("long-score collector lacks complete layer coverage")
        self._runtime.device_synchronize()
        values = np.empty(
            (self.source.config.num_layers, self.tokens, self.num_kv_heads),
            dtype=np.float32,
        )
        copy_device_to_host(
            host_array_ptr(values), self._scores, values.nbytes, runtime=self._runtime
        )
        return values

    def close(self) -> None:
        if self._closed:
            return
        free(self._decisions, runtime=self._runtime)
        free(self._scores, runtime=self._runtime)
        self._projector.close()
        self._closed = True


def _load_sequences(path: Path, *, split: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload["sequences"] if str(row["split"]) == str(split)]
    if not rows:
        raise ValueError(f"long data manifest has no {split!r} rows")
    for row in rows:
        if "token_ids" not in row:
            raise ValueError("long bias calibration requires pretokenized sequences")
    return rows


def _live_summary(
    scores: np.ndarray,
    thresholds: np.ndarray,
    *,
    window: int,
) -> dict[str, Any]:
    layers, tokens, heads = scores.shape
    eligible = max(0, tokens - int(window))
    evictions = np.zeros((layers, heads), dtype=np.int64)
    if eligible:
        evictions = np.count_nonzero(
            scores[:, :eligible, :] > thresholds[:, None, :], axis=1
        )
    live = tokens - evictions
    return {
        "tokens": tokens,
        "logical_rows": int(layers * heads * tokens),
        "live_rows": int(live.sum()),
        "live_compression_ratio": float(layers * heads * tokens / live.sum()),
        "mean_live_count": float(live.mean()),
        "min_live_count": int(live.min()),
        "max_live_count": int(live.max()),
        "per_layer_head_live_counts": live.tolist(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--target-cr", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = args.model.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    data_path = args.data_manifest.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    target_cr = int(args.target_cr)
    if target_cr < 2:
        raise ValueError("target-cr must be at least two")
    config = load_dms_retrofit_config(model, metadata_path=metadata_path)
    source = load_external_dms_sidecar(config)
    rows = _load_sequences(data_path, split=str(args.split))
    token_counts = {len(row["token_ids"]) for row in rows}
    if len(token_counts) != 1:
        raise ValueError("long bias calibration requires uniform sequence lengths")
    tokens = token_counts.pop()
    if tokens <= source.config.window_size:
        raise ValueError("calibration context must materially exceed the DMS window")

    started = time.perf_counter()
    runner = Qwen35GGUFFullStackRunner(model, backend=str(args.backend))
    captures: list[dict[str, Any]] = []
    score_arrays: list[np.ndarray] = []
    try:
        for row in rows:
            collector = _LongScoreCollector(
                source, tokens=tokens, backend=str(args.backend)
            )
            sequence_started = time.perf_counter()
            try:
                with Qwen35GGUFResidentSession(
                    model,
                    backend=str(args.backend),
                    shared_runner=runner,
                    max_sequence_length=tokens,
                    use_wmma_prefill=True,
                    use_gemv_decode=True,
                ) as session:
                    session.prefill(
                        [int(token) for token in row["token_ids"]],
                        use_bulk=True,
                        bulk_attention_mode="bulk",
                        return_logits=False,
                        dms_capture=collector,
                    )
                    scores = collector.finalize()
            finally:
                collector.close()
            score_arrays.append(scores)
            captures.append(
                {
                    "sequence_id": str(row["sequence_id"]),
                    "category": str(row["category"]),
                    "tokens": tokens,
                    "seconds": time.perf_counter() - sequence_started,
                    "score_min": float(scores.min()),
                    "score_max": float(scores.max()),
                    "score_mean": float(scores.mean()),
                }
            )
    finally:
        runner.close()

    eligible = tokens - source.config.window_size
    stacked = np.concatenate(
        [scores[:, :eligible, :] for scores in score_arrays], axis=1
    )
    evict_fraction = 1.0 - 1.0 / float(target_cr)
    threshold_quantile = 1.0 - evict_fraction
    thresholds = np.quantile(stacked, threshold_quantile, axis=1).astype(np.float32)
    old_offset = float(source.config.alpha_offset)
    alpha_scale = float(source.config.alpha_scale)
    comparison_threshold = old_offset / alpha_scale
    new_bias_unrounded = (
        source.bias + np.float32(comparison_threshold) - thresholds
    )
    new_bias = _bf16_float(new_bias_unrounded)
    effective_thresholds = (
        source.bias + np.float32(comparison_threshold) - new_bias
    ).astype(np.float32)

    output.mkdir(parents=True, exist_ok=True)
    sidecar_path = output / "qwen38-27b-q4km-dms-sidecar.safetensors"
    _write_bf16_safetensors(sidecar_path, bias=new_bias, weight=source.weight)
    sidecar_sha = _sha256(sidecar_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    parent_qualification = metadata.get("qualification")
    metadata["sidecar"]["sha256"] = sidecar_sha
    metadata["calibration"] = {
        "method": "long_context_per_layer_head_bias_quantile_v1",
        "source_sidecar_sha256": source.config.sidecar.sha256,
        "source_metadata_sha256": _sha256(metadata_path),
        "data_manifest": str(data_path),
        "data_manifest_sha256": _sha256(data_path),
        "split": str(args.split),
        "sequence_ids": [str(row["sequence_id"]) for row in rows],
        "context_tokens": tokens,
        "window_size": int(source.config.window_size),
        "target_compression_ratio": target_cr,
        "evict_fraction": evict_fraction,
        "alpha_scale": alpha_scale,
        "alpha_offset": old_offset,
        "raw_score_comparison_threshold": comparison_threshold,
        "per_layer_head_original_score_threshold": thresholds.tolist(),
        "per_layer_head_effective_bf16_score_threshold": effective_thresholds.tolist(),
        "per_layer_head_bias_delta": (new_bias - source.bias).tolist(),
    }
    metadata["evidence_source"] = str(output / "calibration_summary.json")
    metadata["qualification"] = {
        "status": "long_bias_calibrated_candidate_quality_open",
        "parent_qualification": parent_qualification,
        "limitations": [
            "threshold/bias calibration only; weights retain 768-token training",
            "long heldout quality not yet run",
            "public serving/lifecycle/performance gates open",
        ],
    }
    output_metadata = output / "dms_metadata.json"
    output_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Fail closed if our minimal BF16 writer or metadata mutation is not accepted
    # by the normal torch-free loader.
    checked_config = load_dms_retrofit_config(model, metadata_path=output_metadata)
    checked_source = load_external_dms_sidecar(checked_config)
    np.testing.assert_array_equal(checked_source.weight, source.weight)
    np.testing.assert_array_equal(checked_source.bias, new_bias)

    sequence_summaries = [
        {
            "sequence_id": str(row["sequence_id"]),
            "category": str(row["category"]),
            **_live_summary(
                scores,
                effective_thresholds,
                window=source.config.window_size,
            ),
        }
        for row, scores in zip(rows, score_arrays, strict=True)
    ]
    summary = {
        "schema_version": 1,
        "kind": "hipengine_qwen38_dms_long_bias_calibration",
        "status": "candidate_quality_open",
        "performance_claim": False,
        "host": socket.gethostname(),
        "backend": str(args.backend),
        "model": {"path": str(model), "sha256": _sha256(model)},
        "source_metadata": {
            "path": str(metadata_path),
            "sha256": _sha256(metadata_path),
        },
        "source_sidecar_sha256": source.config.sidecar.sha256,
        "data_manifest": {"path": str(data_path), "sha256": _sha256(data_path)},
        "protocol": {
            "split": str(args.split),
            "target_cr": target_cr,
            "window_size": int(source.config.window_size),
            "method": "per-layer/head eligible-score median folded into BF16 bias; heldout untouched",
        },
        "captures": captures,
        "calibrated_sequences": sequence_summaries,
        "sidecar": {
            "path": str(sidecar_path),
            "sha256": sidecar_sha,
            "bytes": sidecar_path.stat().st_size,
            "changed_parameters": int(new_bias.size),
            "unchanged_weight_parameters": int(source.weight.size),
        },
        "metadata": {
            "path": str(output_metadata),
            "sha256": _sha256(output_metadata),
        },
        "duration_seconds": time.perf_counter() - started,
        "memory_after_close": memory_stats(),
        "provenance": _git(),
    }
    summary_path = output / "calibration_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Metadata evidence hash is intentionally not self-referential; package
    # validation binds the sidecar/data/model, while the summary binds metadata.
    return summary


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "captures": result["captures"],
                "calibrated_sequences": result["calibrated_sequences"],
                "sidecar": result["sidecar"],
                "metadata": result["metadata"],
                "duration_seconds": result["duration_seconds"],
                "memory_after_close": result["memory_after_close"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
