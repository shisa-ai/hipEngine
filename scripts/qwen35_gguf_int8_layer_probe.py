#!/usr/bin/env python3
# ruff: noqa: E402
"""Layer-local GGUF INT8 KV drift probe.

This diagnostic loads one short mirrored ``int8_per_token_head`` GGUF session,
so every full-attention layer has both the retained INT8 cache and a BF16 mirror.
It then walks one decode token along the BF16 trajectory and, at each
full-attention layer, runs attention twice with the same hidden input:

* BF16 mirror cache path (reference)
* INT8-only retained cache path (candidate, mirror hidden via a scratch view)

The output pinpoints where the current INT8 KV cache first diverges from BF16
without requiring two resident models or a long-context allocation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.kvcache import FixedPagedKVPolicy
import hipengine.runtime.qwen35_gguf_runner as gguf_runtime
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

DEFAULT_MODEL = Path("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")


def _parse_count(text: str) -> int:
    value = text.strip().lower()
    if value.endswith("k"):
        return int(float(value[:-1]) * 1024)
    return int(value)


def _read_compiler_version(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text().strip()


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    arr = np.asarray(bits, dtype=np.uint16)
    return (arr.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    bits = arr.view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _read_array(runtime, ptr: int, shape: tuple[int, ...], dtype: np.dtype | type) -> np.ndarray:
    out = np.empty(shape, dtype=np.dtype(dtype))
    runtime.memcpy(int(out.ctypes.data), int(ptr), int(out.nbytes), HipMemcpyKind.DEVICE_TO_HOST)
    return out


def _read_bf16(runtime, ptr: int, numel: int) -> np.ndarray:
    out = _read_array(runtime, ptr, (int(numel),), np.uint16)
    return _bf16_bits_to_f32(out).astype(np.float32, copy=False)


def _compare(ref: np.ndarray, cand: np.ndarray) -> dict[str, Any]:
    diff = np.asarray(cand, dtype=np.float32) - np.asarray(ref, dtype=np.float32)
    abs_diff = np.abs(diff)
    ref_abs = np.maximum(np.abs(ref), np.float32(1.0e-8))
    return {
        "max_abs": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "rms_abs": float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))) if diff.size else 0.0,
        "max_rel": float(np.max(abs_diff / ref_abs)) if abs_diff.size else 0.0,
        "top_abs_indices": [int(i) for i in np.argsort(abs_diff.reshape(-1))[-5:][::-1].tolist()],
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x))).astype(np.float32)


def _attention_cpu(
    query: np.ndarray,
    key_rows: np.ndarray,
    value_rows: np.ndarray,
    gate: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    num_q_heads, head_dim = query.shape
    num_kv_heads = key_rows.shape[1]
    group = num_q_heads // num_kv_heads
    out = np.empty((num_q_heads, head_dim), dtype=np.float32)
    for q_head in range(num_q_heads):
        kv_head = q_head // group
        scores = (key_rows[:, kv_head, :].astype(np.float32) @ query[q_head].astype(np.float32)) * np.float32(scale)
        scores = scores - np.max(scores)
        weights = np.exp(scores).astype(np.float32)
        weights = weights / np.sum(weights, dtype=np.float32)
        out[q_head] = weights.astype(np.float32) @ value_rows[:, kv_head, :].astype(np.float32)
    gated = out * _sigmoid(gate)
    return _bf16_bits_to_f32(_f32_to_bf16_bits(gated)).astype(np.float32, copy=False)


def _cpu_cache_decomposition(
    session: Qwen35GGUFResidentSession,
    layer_id: int,
    active_context: int,
) -> dict[str, Any]:
    if session.scratch is None or session.runner is None or session.runner.weights is None:
        raise RuntimeError("GGUF resident session is not initialized")
    runtime = session.runtime
    cfg = session.runner.weights.config
    blocks = int(session.scratch.max_positions // session.scratch.block_size)
    cache_shape = (blocks, int(session.scratch.block_size), int(cfg.head_count_kv), int(cfg.key_length))
    scale_shape = (blocks, int(session.scratch.block_size), int(cfg.head_count_kv))
    key_cache, value_cache = session.scratch.full_cache(layer_id)
    mirror = session.scratch.full_bf16_mirror_cache(layer_id)
    metadata = session.scratch.full_scale_metadata(layer_id)
    if mirror is None or metadata is None:
        raise RuntimeError("CPU decomposition requires INT8 cache plus BF16 mirror")
    mirror_key_cache, mirror_value_cache = mirror
    bf16_key_rows = _bf16_bits_to_f32(_read_array(runtime, mirror_key_cache.ptr, cache_shape, np.uint16)).reshape(
        -1,
        int(cfg.head_count_kv),
        int(cfg.key_length),
    )[:active_context]
    bf16_value_rows = _bf16_bits_to_f32(_read_array(runtime, mirror_value_cache.ptr, cache_shape, np.uint16)).reshape(
        -1,
        int(cfg.head_count_kv),
        int(cfg.key_length),
    )[:active_context]
    int8_key_rows = _read_array(runtime, key_cache.ptr, cache_shape, np.int8).reshape(
        -1,
        int(cfg.head_count_kv),
        int(cfg.key_length),
    )[:active_context]
    int8_value_rows = _read_array(runtime, value_cache.ptr, cache_shape, np.int8).reshape(
        -1,
        int(cfg.head_count_kv),
        int(cfg.key_length),
    )[:active_context]
    scale_dtype = np.float16 if metadata.scale_dtype == DType.FP16 else np.float32
    key_scales = _read_array(runtime, metadata.k_scale.ptr, scale_shape, scale_dtype).astype(np.float32).reshape(
        -1,
        int(cfg.head_count_kv),
    )[:active_context]
    value_scales = _read_array(runtime, metadata.v_scale.ptr, scale_shape, scale_dtype).astype(np.float32).reshape(
        -1,
        int(cfg.head_count_kv),
    )[:active_context]
    deq_key_rows = int8_key_rows.astype(np.float32) * key_scales[:, :, None]
    deq_value_rows = int8_value_rows.astype(np.float32) * value_scales[:, :, None]
    query = _read_array(
        runtime,
        session.scratch.full_query.ptr,
        (int(cfg.head_count), int(cfg.key_length)),
        np.float32,
    )
    gate = _read_bf16(runtime, session.scratch.full_gate.ptr, int(cfg.head_count) * int(cfg.key_length)).reshape(
        int(cfg.head_count),
        int(cfg.key_length),
    )
    attn_scale = float(int(cfg.key_length) ** -0.5)
    bf16_out = _attention_cpu(query, bf16_key_rows, bf16_value_rows, gate, scale=attn_scale)
    key_only = _attention_cpu(query, deq_key_rows, bf16_value_rows, gate, scale=attn_scale)
    value_only = _attention_cpu(query, bf16_key_rows, deq_value_rows, gate, scale=attn_scale)
    both = _attention_cpu(query, deq_key_rows, deq_value_rows, gate, scale=attn_scale)
    key_error = deq_key_rows - bf16_key_rows
    value_error = deq_value_rows - bf16_value_rows
    return {
        "layer_id": int(layer_id),
        "active_context": int(active_context),
        "scale_dtype": str(metadata.scale_dtype.value),
        "quantization_error": {
            "key": _compare(np.zeros_like(key_error), key_error),
            "value": _compare(np.zeros_like(value_error), value_error),
            "key_scale_min": float(np.min(key_scales)),
            "key_scale_max": float(np.max(key_scales)),
            "value_scale_min": float(np.min(value_scales)),
            "value_scale_max": float(np.max(value_scales)),
        },
        "attention_gated_diff": {
            "key_only": _compare(bf16_out.reshape(-1), key_only.reshape(-1)),
            "value_only": _compare(bf16_out.reshape(-1), value_only.reshape(-1)),
            "key_and_value": _compare(bf16_out.reshape(-1), both.reshape(-1)),
        },
    }


def _logit_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    metrics = evaluate_logits(reference.reshape(1, -1), candidate.reshape(1, -1))
    return {
        "kl_mean": float(metrics.kl_mean),
        "kl_max": float(metrics.kl_max),
        "top1_agreement": float(metrics.top1_agreement),
        "reference_top1": int(np.argmax(reference)),
        "candidate_top1": int(np.argmax(candidate)),
        "max_abs": float(np.max(np.abs(candidate - reference))),
        "mean_abs": float(np.mean(np.abs(candidate - reference))),
    }


def _backup_state_buffers(runtime, session: Qwen35GGUFResidentSession) -> list[tuple[object, int]]:
    if session.scratch is None:
        return []
    backups: list[tuple[object, int]] = []
    for buffer in (*session.scratch.layer_conv_states, *session.scratch.layer_recurrent_states):
        if buffer is None:
            continue
        backup_ptr = runtime.malloc(buffer.nbytes)
        runtime.memcpy(backup_ptr, buffer.ptr, buffer.nbytes, HipMemcpyKind.DEVICE_TO_DEVICE)
        backups.append((buffer, backup_ptr))
    return backups


def _restore_state_buffers(runtime, backups: list[tuple[object, int]]) -> None:
    for buffer, backup_ptr in backups:
        runtime.memcpy(buffer.ptr, backup_ptr, buffer.nbytes, HipMemcpyKind.DEVICE_TO_DEVICE)


def _free_state_backups(runtime, backups: list[tuple[object, int]]) -> None:
    for _buffer, backup_ptr in backups:
        runtime.free(backup_ptr)


def _command(args: argparse.Namespace) -> str:
    parts = [
        "python3 scripts/qwen35_gguf_int8_layer_probe.py",
        f"--model {args.model}",
        f"--prompt-length {args.prompt_length_raw}",
        f"--decode-steps {args.decode_steps}",
        f"--token-id {args.token_id}",
        f"--max-sequence-length {args.max_sequence_length}",
        f"--scale-dtype {args.scale_dtype}",
        f"--cpu-cache-layer {args.cpu_cache_layer}",
    ]
    if args.compiler_version_file is not None:
        parts.append(f"--compiler-version-file {args.compiler_version_file}")
    if args.require_cached_build:
        parts.append("--require-cached-build")
    if args.json is not None:
        parts.append(f"--json {args.json}")
    return " ".join(parts)


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompt_length = _parse_count(args.prompt_length_raw)
    max_sequence_length = int(args.max_sequence_length or max(8192, prompt_length + int(args.decode_steps) + 2))
    prompt_tokens = [int(args.token_id)] * prompt_length
    compiler_version = _read_compiler_version(args.compiler_version_file)
    policy = FixedPagedKVPolicy(block_size=256, storage_dtype=DType.INT8_PER_TOKEN_HEAD)
    rows: list[dict[str, Any]] = []
    cpu_cache_decomposition: dict[str, Any] | None = None
    with Qwen35GGUFResidentSession(
        args.model,
        compiler_version=compiler_version,
        require_cached_build=bool(args.require_cached_build),
        max_sequence_length=max_sequence_length,
        use_wmma_prefill=True,
        use_gemv_decode=True,
        kv_policy=policy,
        kv_scale_dtype=DType.parse(args.scale_dtype),
        kv_scale_granularity="per_token_head",
    ) as session:
        if session.scratch is None or session.runner is None or session.runner.weights is None:
            raise RuntimeError("GGUF resident session is not initialized")
        mirror_count = sum(1 for key in session.scratch.full_bf16_mirror_key_caches if key is not None)
        if mirror_count == 0:
            raise RuntimeError("layer probe requires a short INT8 session with BF16 mirror caches")
        first = session.prefill(prompt_tokens, use_bulk=True, bulk_attention_mode="bulk", return_logits=True)
        decode_token = int(first.token_id)
        position = int(prompt_length)
        session._set_full_attention_position_device(position)
        session._set_token_id_device(decode_token)
        runtime = session.runtime
        hidden_size = int(session.runner.hidden_size)
        q_width = int(session.runner.q_width)
        src = session._hidden_a
        dst = session._hidden_b
        if src is None or dst is None:
            raise RuntimeError("session hidden buffers are closed")
        no_mirror_scratch = replace(
            session.scratch,
            full_bf16_mirror_key_caches=tuple(None for _ in session.scratch.full_bf16_mirror_key_caches),
            full_bf16_mirror_value_caches=tuple(None for _ in session.scratch.full_bf16_mirror_value_caches),
        )
        state_backups = _backup_state_buffers(runtime, session)

        def decode_variant(int8_full_attention_indices: set[int]) -> tuple[int, np.ndarray]:
            _restore_state_buffers(runtime, state_backups)
            session._set_full_attention_position_device(position)
            session._set_token_id_device(decode_token)
            variant_src = session._hidden_a
            variant_dst = session._hidden_b
            if variant_src is None or variant_dst is None:
                raise RuntimeError("session hidden buffers are closed")
            full_index = 0
            for variant_layer_id, variant_layer_type in enumerate(session.runner.weights.config.layer_types):
                if variant_layer_type == gguf_runtime.LINEAR_ATTENTION:
                    session.runner._run_linear_attention_layer(
                        variant_layer_id,
                        variant_src.ptr,
                        variant_dst.ptr,
                        session.scratch,
                    )
                elif variant_layer_type == gguf_runtime.FULL_ATTENTION:
                    active_scratch = no_mirror_scratch if full_index in int8_full_attention_indices else session.scratch
                    session.runner._run_full_attention_layer(
                        variant_layer_id,
                        variant_src.ptr,
                        variant_dst.ptr,
                        active_scratch,
                        position=position,
                    )
                    full_index += 1
                else:
                    raise ValueError(f"unsupported layer type {variant_layer_type!r}")
                variant_src, variant_dst = variant_dst, variant_src
            gguf_runtime.gguf_rmsnorm_bf16_f32_weight(
                variant_src.ptr,
                session.runner.weights.root("output_norm").allocation().tensor.ptr,
                session.scratch.norm.ptr,
                rows=1,
                hidden_size=hidden_size,
                eps=session.runner.weights.config.rms_norm_eps,
                runtime=runtime,
            )
            result = session._sample_from_hidden(session.scratch.norm.ptr, return_logits=True)
            return int(result.token_id), np.asarray(result.logits, dtype=np.float32).reshape(-1).copy()

        full_attention_layer_ids = [
            int(layer_id)
            for layer_id, layer_type in enumerate(session.runner.weights.config.layer_types)
            if layer_type == gguf_runtime.FULL_ATTENTION
        ]
        try:
            for layer_id, layer_type in enumerate(session.runner.weights.config.layer_types):
                if layer_type == gguf_runtime.LINEAR_ATTENTION:
                    session.runner._run_linear_attention_layer(layer_id, src.ptr, dst.ptr, session.scratch)
                    src, dst = dst, src
                    continue
                if layer_type != gguf_runtime.FULL_ATTENTION:
                    raise ValueError(f"unsupported layer type {layer_type!r}")

                session.runner._run_full_attention_attn_only(
                    layer_id,
                    src.ptr,
                    session.scratch.attn_out.ptr,
                    session.scratch,
                    position=position,
                )
                bf16_gated = _read_bf16(runtime, session.scratch.full_gated.ptr, q_width)
                bf16_attn_out = _read_bf16(runtime, session.scratch.attn_out.ptr, hidden_size)

                session.runner._run_full_attention_attn_only(
                    layer_id,
                    src.ptr,
                    session.scratch.attn_out.ptr,
                    no_mirror_scratch,
                    position=position,
                )
                int8_gated = _read_bf16(runtime, session.scratch.full_gated.ptr, q_width)
                int8_attn_out = _read_bf16(runtime, session.scratch.attn_out.ptr, hidden_size)

                full_attention_index = len(rows)
                rows.append(
                    {
                        "layer_id": int(layer_id),
                        "full_attention_index": int(full_attention_index),
                        "gated_diff": _compare(bf16_gated, int8_gated),
                        "attn_out_diff": _compare(bf16_attn_out, int8_attn_out),
                    }
                )
                if int(args.cpu_cache_layer) in {int(layer_id), int(full_attention_index)}:
                    cpu_cache_decomposition = _cpu_cache_decomposition(
                        session,
                        int(layer_id),
                        int(position) + 1,
                    )

                # Continue the walk on the BF16 mirror trajectory so later comparisons
                # use the same hidden inputs as the BF16 reference path.
                session.runner._run_full_attention_layer(
                    layer_id,
                    src.ptr,
                    dst.ptr,
                    session.scratch,
                    position=position,
                )
                src, dst = dst, src
            gguf_runtime.gguf_rmsnorm_bf16_f32_weight(
                src.ptr,
                session.runner.weights.root("output_norm").allocation().tensor.ptr,
                session.scratch.norm.ptr,
                rows=1,
                hidden_size=hidden_size,
                eps=session.runner.weights.config.rms_norm_eps,
                runtime=runtime,
            )
            final = session._sample_from_hidden(session.scratch.norm.ptr, return_logits=False)
            bf16_token, bf16_logits = decode_variant(set())
            ablations: list[dict[str, Any]] = [
                {
                    "label": "all_bf16_reference",
                    "int8_full_attention_indices": [],
                    "layer_ids": [],
                    "token_id": int(bf16_token),
                    "logit_metrics_vs_bf16": _logit_metrics(bf16_logits, bf16_logits),
                }
            ]
            for full_index, layer_id in enumerate(full_attention_layer_ids):
                token, logits = decode_variant({full_index})
                ablations.append(
                    {
                        "label": f"single_full_layer_{full_index}",
                        "int8_full_attention_indices": [int(full_index)],
                        "layer_ids": [int(layer_id)],
                        "token_id": int(token),
                        "logit_metrics_vs_bf16": _logit_metrics(bf16_logits, logits),
                    }
                )
            for start in range(1, len(full_attention_layer_ids)):
                suffix_indices = set(range(start, len(full_attention_layer_ids)))
                token, logits = decode_variant(suffix_indices)
                ablations.append(
                    {
                        "label": f"suffix_from_full_layer_{start}",
                        "int8_full_attention_indices": list(range(start, len(full_attention_layer_ids))),
                        "layer_ids": full_attention_layer_ids[start:],
                        "token_id": int(token),
                        "logit_metrics_vs_bf16": _logit_metrics(bf16_logits, logits),
                    }
                )
            all_token, all_logits = decode_variant(set(range(len(full_attention_layer_ids))))
            ablations.append(
                {
                    "label": "all_full_attention_layers",
                    "int8_full_attention_indices": list(range(len(full_attention_layer_ids))),
                    "layer_ids": full_attention_layer_ids,
                    "token_id": int(all_token),
                    "logit_metrics_vs_bf16": _logit_metrics(bf16_logits, all_logits),
                }
            )
        finally:
            _free_state_backups(runtime, state_backups)
    return {
        "schema": 1,
        "mode": "qwen35_gguf_int8_layer_probe",
        "command": _command(args),
        "model": str(args.model),
        "prompt_length": int(prompt_length),
        "decode_token": int(decode_token),
        "max_sequence_length": int(max_sequence_length),
        "mirror_layer_count": int(mirror_count),
        "bf16_trajectory_final_token_after_probe": int(final.token_id),
        "rows": rows,
        "ablations": ablations,
        "cpu_cache_decomposition": cpu_cache_decomposition,
        "summary": {
            "max_gated_diff_layer": max(rows, key=lambda row: row["gated_diff"]["max_abs"]),
            "max_attn_out_diff_layer": max(rows, key=lambda row: row["attn_out_diff"]["max_abs"]),
            "all_int8_logit_metrics": ablations[-1]["logit_metrics_vs_bf16"],
            "worst_single_layer_logit_metrics": max(
                (row for row in ablations if row["label"].startswith("single_full_layer_")),
                key=lambda row: row["logit_metrics_vs_bf16"]["kl_mean"],
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-length", dest="prompt_length_raw", default="4K")
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--token-id", type=int, default=9707)
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    parser.add_argument("--scale-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--cpu-cache-layer",
        type=int,
        default=-1,
        help="Optional full-attention index or layer id to decompose with a CPU key-only/value-only check.",
    )
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
