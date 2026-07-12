#!/usr/bin/env python3
"""Compare GGUF final output_norm/lm_head logits against a CPU dequant oracle."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.memory import (  # noqa: E402
    DeviceBuffer,
    copy_device_to_host,
    host_array_ptr,
)
from hipengine.loading.gguf import GGUFReader  # noqa: E402
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map  # noqa: E402
from hipengine.quant.gguf import bf16_to_float32, dequantize_gguf_data  # noqa: E402
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession  # noqa: E402
from scripts.gguf_linear_boundary_capture import _parse_tokens, _resolve_position  # noqa: E402
from scripts.gguf_linear_layer_capture import DEFAULT_PROMPT_TOKENS  # noqa: E402

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
DEFAULT_OUTPUT = Path("benchmarks/results/mtp-gguf-iter279-output-head-compare.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--tokens", default=",".join(str(token) for token in DEFAULT_PROMPT_TOKENS)
    )
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--iteration", type=int, default=279)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compiler-version")
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-abs-tolerance", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tokens = _parse_tokens(args.tokens)
    position = _resolve_position(args.position, len(tokens))
    if args.dry_run:
        artifact = _plan_artifact(
            model=args.model,
            tokens=tokens,
            position=position,
            status="dry_run",
            iteration=args.iteration,
            chunk_rows=args.chunk_rows,
            top_k=args.top_k,
        )
    elif not _hip_available():
        artifact = _plan_artifact(
            model=args.model,
            tokens=tokens,
            position=position,
            status="skipped_no_hip_runtime",
            iteration=args.iteration,
            chunk_rows=args.chunk_rows,
            top_k=args.top_k,
        )
    else:
        artifact = build_output_head_artifact(
            model=args.model,
            tokens=tokens,
            position=position,
            compiler_version=args.compiler_version,
            require_cached_build=bool(args.require_cached_build),
            max_sequence_length=args.max_sequence_length,
            chunk_rows=args.chunk_rows,
            top_k=args.top_k,
            max_abs_tolerance=args.max_abs_tolerance,
            iteration=args.iteration,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "position": artifact["position"],
                "token_id": artifact["token_id"],
                "top1_match": artifact.get("comparison", {}).get("top1_match"),
                "max_abs_diff": artifact.get("comparison", {}).get("diff", {}).get(
                    "max_abs_diff"
                ),
                "within_tolerance": artifact.get("within_tolerance"),
            },
            indent=2,
        )
    )


def build_output_head_artifact(
    *,
    model: Path,
    tokens: tuple[int, ...],
    position: int,
    compiler_version: str | None,
    require_cached_build: bool,
    max_sequence_length: int | None,
    chunk_rows: int,
    top_k: int,
    max_abs_tolerance: float,
    iteration: int = 279,
) -> dict[str, Any]:
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    reader = GGUFReader(model)
    model_map = build_qwen35_gguf_tensor_map(reader.info)
    lm_head_name = model_map.root("lm_head").name
    input_tokens = tokens[: position + 1]
    max_seq = int(max_sequence_length or max(len(input_tokens) + 8, 32))
    with Qwen35GGUFResidentSession(
        model,
        compiler_version=compiler_version,
        require_cached_build=bool(require_cached_build),
        max_sequence_length=max_seq,
    ) as session:
        hidden_bits, hidden_seed_fp32, device_logits = _run_resident_prompt_head(
            session,
            input_tokens,
        )
    hidden_f32 = bf16_to_float32(hidden_bits.reshape(-1)).astype(np.float32)
    cpu_logits = stream_lm_head_logits(
        reader,
        lm_head_name,
        hidden_f32,
        chunk_rows=int(chunk_rows),
    )
    comparison = compare_output_head_logits(device_logits, cpu_logits, top_k=int(top_k))
    within = bool(
        comparison["top1_match"]
        and comparison["diff"]["max_abs_diff"] <= float(max_abs_tolerance)
    )
    artifact = _plan_artifact(
        model=model,
        tokens=tokens,
        position=position,
        status="compared",
        iteration=iteration,
        chunk_rows=chunk_rows,
        top_k=top_k,
    )
    artifact.update(
        {
            "input_tokens": list(input_tokens),
            "lm_head_tensor": lm_head_name,
            "lm_head_shape": list(reader.tensor_info(lm_head_name).shape),
            "lm_head_ggml_type": reader.tensor_info(lm_head_name).ggml_type_name,
            "hidden_bf16_summary": _array_summary(hidden_f32),
            "hidden_seed_fp32_vs_bf16": _hidden_seed_diff(hidden_seed_fp32, hidden_f32),
            "comparison": comparison,
            "max_abs_tolerance": float(max_abs_tolerance),
            "within_tolerance": within,
            "conclusion": _conclusion(comparison, within, max_abs_tolerance),
        }
    )
    return artifact


def _run_resident_prompt_head(
    session: Qwen35GGUFResidentSession,
    input_tokens: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not input_tokens:
        raise ValueError("input_tokens must be non-empty")
    if session.runner is None:
        raise RuntimeError("GGUF resident session is closed")
    runtime = session.runtime
    session.reset()
    hidden_ptr: int | None = None
    final_index = len(input_tokens) - 1
    for pos, token_id in enumerate(input_tokens):
        hidden_ptr = session._run_token_to_final_hidden(  # noqa: SLF001
            int(token_id),
            position=pos,
            capture_hidden_seed_fp32=pos == final_index,
        )
    if hidden_ptr is None:
        raise RuntimeError("resident prompt did not produce a final hidden pointer")
    result = session._sample_from_hidden(hidden_ptr, return_logits=True)  # noqa: SLF001
    hidden_bits = np.empty((1, session.runner.hidden_size), dtype=np.uint16)
    copy_device_to_host(
        host_array_ptr(hidden_bits),
        DeviceBuffer(int(hidden_ptr), hidden_bits.nbytes),
        runtime=runtime,
    )
    hidden_seed = np.empty((session.runner.hidden_size,), dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(hidden_seed),
        DeviceBuffer(session.fp32_hidden_seed_ptr(), hidden_seed.nbytes),
        runtime=runtime,
    )
    return hidden_bits, hidden_seed, result.logits.reshape(-1).astype(np.float32)


def stream_lm_head_logits(
    reader: GGUFReader,
    tensor_name: str,
    hidden: np.ndarray,
    *,
    chunk_rows: int = 1024,
) -> np.ndarray:
    info = reader.tensor_info(tensor_name)
    raw = reader.tensor_data(tensor_name)
    vocab_size, hidden_size = _matrix_shape(info.shape, tensor_name)
    hidden = np.asarray(hidden, dtype=np.float32).reshape(-1)
    if hidden.size != hidden_size:
        raise ValueError(
            f"hidden size {hidden.size} does not match {tensor_name} width {hidden_size}"
        )
    chunks = _reader_weight_chunks(reader, tensor_name, chunk_rows=chunk_rows)
    return stream_lm_head_logits_from_chunks(chunks, hidden, vocab_size=vocab_size)


def stream_lm_head_logits_from_chunks(
    chunks: Iterable[tuple[int, np.ndarray]],
    hidden: np.ndarray,
    *,
    vocab_size: int,
) -> np.ndarray:
    hidden = np.asarray(hidden, dtype=np.float32).reshape(-1)
    logits = np.empty((int(vocab_size),), dtype=np.float32)
    filled = np.zeros((int(vocab_size),), dtype=np.bool_)
    for start, weight_chunk in chunks:
        start = int(start)
        weight = np.asarray(weight_chunk, dtype=np.float32)
        if weight.ndim != 2:
            raise ValueError("weight chunks must be rank-2 matrices")
        if weight.shape[1] != hidden.size:
            raise ValueError("weight chunk width must match hidden size")
        end = start + int(weight.shape[0])
        if start < 0 or end > vocab_size or end <= start:
            raise ValueError("weight chunk range is outside vocab_size")
        logits[start:end] = weight @ hidden
        filled[start:end] = True
    if not bool(np.all(filled)):
        raise ValueError("weight chunks did not cover every vocab row exactly once")
    return logits


def compare_output_head_logits(
    device_logits: np.ndarray,
    cpu_logits: np.ndarray,
    *,
    top_k: int = 8,
) -> dict[str, Any]:
    device = np.asarray(device_logits, dtype=np.float32).reshape(-1)
    cpu = np.asarray(cpu_logits, dtype=np.float32).reshape(-1)
    if device.shape != cpu.shape:
        raise ValueError("device_logits and cpu_logits must have the same shape")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    diff = device - cpu
    device_top = _topk(device, top_k)
    cpu_top = _topk(cpu, top_k)
    selected = sorted(
        {idx for idx, _value in device_top} | {idx for idx, _value in cpu_top}
    )
    return {
        "vocab_size": int(device.size),
        "top_k": int(min(top_k, device.size)),
        "device_top": _top_payload(device_top),
        "cpu_top": _top_payload(cpu_top),
        "device_top1_token_id": int(device_top[0][0]),
        "cpu_top1_token_id": int(cpu_top[0][0]),
        "top1_match": bool(device_top[0][0] == cpu_top[0][0]),
        "topk_overlap_count": int(
            len({idx for idx, _ in device_top} & {idx for idx, _ in cpu_top})
        ),
        "selected_logits": [
            {
                "token_id": int(idx),
                "device_logit": float(device[idx]),
                "cpu_logit": float(cpu[idx]),
                "diff": float(diff[idx]),
                "abs_diff": float(abs(diff[idx])),
            }
            for idx in selected
        ],
        "diff": _diff_metrics(device, cpu),
    }


def _reader_weight_chunks(
    reader: GGUFReader,
    tensor_name: str,
    *,
    chunk_rows: int,
) -> Iterable[tuple[int, np.ndarray]]:
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    info = reader.tensor_info(tensor_name)
    raw = reader.tensor_data(tensor_name)
    rows, _hidden = _matrix_shape(info.shape, tensor_name)
    for start in range(0, rows, int(chunk_rows)):
        end = min(start + int(chunk_rows), rows)
        yield start, dequantize_gguf_data(raw[start:end], info.ggml_type).astype(
            np.float32
        )


def _topk(values: np.ndarray, k: int) -> list[tuple[int, float]]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise ValueError("values must be non-empty")
    k = int(min(k, flat.size))
    indices = np.argpartition(-flat, k - 1)[:k]
    order = np.lexsort((indices, -flat[indices]))
    return [(int(idx), float(flat[idx])) for idx in indices[order]]


def _top_payload(items: list[tuple[int, float]]) -> list[dict[str, float | int]]:
    return [{"token_id": int(idx), "logit": float(value)} for idx, value in items]


def _diff_metrics(device: np.ndarray, cpu: np.ndarray) -> dict[str, float | int]:
    diff = np.asarray(device, dtype=np.float32) - np.asarray(cpu, dtype=np.float32)
    abs_diff = np.abs(diff)
    return {
        "count": int(diff.size),
        "max_abs_diff": float(np.max(abs_diff)) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(abs_diff, dtype=np.float32)) if diff.size else 0.0,
        "rms_abs_diff": float(np.sqrt(np.mean(diff * diff, dtype=np.float32)))
        if diff.size
        else 0.0,
        "device_rms": float(np.sqrt(np.mean(device * device, dtype=np.float32)))
        if diff.size
        else 0.0,
        "cpu_rms": float(np.sqrt(np.mean(cpu * cpu, dtype=np.float32))) if diff.size else 0.0,
    }


def _hidden_seed_diff(
    hidden_seed_fp32: np.ndarray, hidden_bf16_f32: np.ndarray
) -> dict[str, Any]:
    seed = np.asarray(hidden_seed_fp32, dtype=np.float32).reshape(-1)
    bf16 = np.asarray(hidden_bf16_f32, dtype=np.float32).reshape(-1)
    if seed.shape != bf16.shape:
        raise ValueError("hidden_seed_fp32 and hidden_bf16_f32 must have the same shape")
    return _diff_metrics(seed, bf16)


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float32).reshape(-1)
    return {
        "shape": list(np.asarray(array).shape),
        "dtype": str(np.asarray(array).dtype),
        "finite": bool(np.all(np.isfinite(values))),
        "min": float(np.min(values)) if values.size else 0.0,
        "max": float(np.max(values)) if values.size else 0.0,
        "mean": float(np.mean(values, dtype=np.float32)) if values.size else 0.0,
        "rms": float(np.sqrt(np.mean(values * values, dtype=np.float32)))
        if values.size
        else 0.0,
    }


def _matrix_shape(shape: tuple[int, ...], name: str) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"expected rank-2 lm_head tensor for {name}, got shape {shape}")
    return int(shape[0]), int(shape[1])


def _plan_artifact(
    *,
    model: Path,
    tokens: tuple[int, ...],
    position: int,
    status: str,
    iteration: int,
    chunk_rows: int,
    top_k: int,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": "mtp_gguf_output_head_cpu_compare",
        "date": "2026-06-20",
        "loop": "mtp-gguf/run-20260615-103738",
        "iteration": int(iteration),
        "status": str(status),
        "model": str(model),
        "position": int(position),
        "token_id": int(tokens[position]),
        "prompt_tokens": list(tokens),
        "chunk_rows": int(chunk_rows),
        "top_k": int(top_k),
        "api": "Qwen35GGUFResidentSession._run_token_to_final_hidden+_sample_from_hidden",
        "note": (
            "Runs the prompt through the resident full stack to get the exact BF16 "
            "post-output_norm hidden row consumed by the device lm_head and the device logits, "
            "then streams dequantized lm_head rows on CPU to compare logits/top-1 without "
            "materializing the full head matrix."
        ),
    }


def _conclusion(
    comparison: dict[str, Any], within: bool, max_abs_tolerance: float
) -> str:
    diff = comparison["diff"]
    if within:
        return (
            "Final output_norm/lm_head boundary matches the streaming CPU dequant oracle: "
            f"top1 token {comparison['device_top1_token_id']} agrees and "
            f"max_abs_diff={diff['max_abs_diff']:.6g} <= {max_abs_tolerance:.6g}. "
            "AR parity should continue narrowing earlier hidden-state/KV propagation."
        )
    return (
        "Final output_norm/lm_head boundary diverges from the streaming CPU dequant oracle: "
        f"device_top1={comparison['device_top1_token_id']} "
        f"cpu_top1={comparison['cpu_top1_token_id']} "
        f"max_abs_diff={diff['max_abs_diff']:.6g}. Inspect output_norm/lm_head quant path."
    )


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


if __name__ == "__main__":
    main()
