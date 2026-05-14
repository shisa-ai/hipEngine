#!/usr/bin/env python3
"""Minimal torch-free Qwen3.5/PARO one-token next-token smoke.

This is a correctness/bring-up harness, not a performance path.  It processes a
single decode position through HIPENGINE layer chains, then does the final
lm-head argmax on CPU with NumPy so we can validate real-model sequencing before
porting the optimized lm-head path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from safetensors import safe_open

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.norm import paro_rmsnorm_out_bf16
from hipengine.kvcache import KVLiveSpans
from hipengine.loading import (
    float_array_to_bf16_bits,
    load_weight_index,
    materialize_qwen35_paro_full_attention_moe_c1_runtime_layer,
    materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer,
    normalize_qwen35_weight_name,
    qwen35_paro_config_from_hf,
)
from hipengine.loading.materialize import load_host_array_to_device_as_dtype
from hipengine.runtime import Qwen35ParoDecodeState, RuntimeWorkspace

DEFAULT_MODEL = "/models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--token-id", type=int, default=None, help="Bypass tokenizer and decode this single token id")
    parser.add_argument("--max-layers", type=int, default=0, help="Debug limit; 0 means all layers")
    parser.add_argument("--lm-head-chunk", type=int, default=4096)
    args = parser.parse_args()

    model = Path(args.model)
    index = load_weight_index(model)
    config = qwen35_paro_config_from_hf(index.config)
    normalized = _normalized_infos(index)
    token_id, prompt_ids = _select_token(model, args.prompt, args.token_id)

    runtime = get_hip_runtime()
    device = Device("hip", 0)
    buffers: list[DeviceBuffer] = []
    states_to_free: list[Any] = []

    def dev(array: np.ndarray) -> DeviceBuffer:
        buf = malloc(array.nbytes, runtime=runtime)
        buffers.append(buf)
        copy_host_to_device(buf, host_array_ptr(array), runtime=runtime)
        return buf

    hidden_bits = float_array_to_bf16_bits(_read_tensor(normalized, "language_model.embed_tokens.weight")[token_id : token_id + 1])
    if hidden_bits.shape != (1, config.hidden_size):
        raise ValueError(f"unexpected embedding row shape {hidden_bits.shape}, expected (1, {config.hidden_size})")
    hidden_a = dev(hidden_bits)
    hidden_b = malloc(hidden_bits.nbytes, runtime=runtime)
    buffers.append(hidden_b)
    hidden = Tensor.from_handle(hidden_a.ptr, hidden_bits.shape, DType.BF16, device)
    next_hidden = Tensor.from_handle(hidden_b.ptr, hidden_bits.shape, DType.BF16, device)

    # One-token decode smoke: all full-attention layers can reuse the same temporary KV page,
    # and all linear layers can reuse zeroed recurrent/conv state inputs.
    block_size = 256
    block_table_arr = np.asarray([0], dtype=np.int32)
    position_arr = np.asarray([0], dtype=np.int64)
    context_arr = np.asarray([1], dtype=np.int64)
    block_table_buf = dev(block_table_arr)
    position_buf = dev(position_arr)
    context_buf = dev(context_arr)
    block_table = Tensor.from_handle(block_table_buf.ptr, block_table_arr.shape, DType.INT32, device)
    position = Tensor.from_handle(position_buf.ptr, position_arr.shape, DType.INT64, device)
    context = Tensor.from_handle(context_buf.ptr, context_arr.shape, DType.INT64, device)
    append_spans = KVLiveSpans.paged_uniform(block_table=block_table, live_counts=position, max_live_count=0, storage_dtype=DType.BF16)
    decode_spans = KVLiveSpans.paged_uniform(block_table=block_table, live_counts=context, max_live_count=1, storage_dtype=DType.BF16)
    cos_arr, sin_arr = _rope_tables(max_positions=1, rotary_dim=config.rotary_dim or config.head_dim, base=config.rope_theta)
    cos_buf = dev(cos_arr)
    sin_buf = dev(sin_arr)
    cos = Tensor.from_handle(cos_buf.ptr, cos_arr.shape, DType.FP32, device)
    sin = Tensor.from_handle(sin_buf.ptr, sin_arr.shape, DType.FP32, device)

    key_cache_arr = np.zeros((1, block_size, config.num_key_value_heads, config.head_dim), dtype=np.uint16)
    value_cache_arr = np.zeros_like(key_cache_arr)
    key_cache_buf = dev(key_cache_arr)
    value_cache_buf = dev(value_cache_arr)
    key_cache = Tensor.from_handle(key_cache_buf.ptr, key_cache_arr.shape, DType.BF16, device)
    value_cache = Tensor.from_handle(value_cache_buf.ptr, value_cache_arr.shape, DType.BF16, device)

    qkv_width = 2 * config.linear_num_key_heads * config.linear_key_head_dim + config.linear_num_value_heads * config.linear_value_head_dim
    conv_zero = np.zeros((qkv_width, config.linear_conv_kernel_dim), dtype=np.float32)
    recurrent_zero = np.zeros(
        (config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim),
        dtype=np.float32,
    )
    conv_buf = dev(conv_zero)
    recurrent_buf = dev(recurrent_zero)
    conv_state = Tensor.from_handle(conv_buf.ptr, conv_zero.shape, DType.FP32, device)
    recurrent_state = Tensor.from_handle(recurrent_buf.ptr, recurrent_zero.shape, DType.FP32, device)

    layer_limit = config.num_hidden_layers if args.max_layers <= 0 else min(args.max_layers, config.num_hidden_layers)
    layer_records: list[dict[str, Any]] = []
    try:
        for layer_id in range(layer_limit):
            layer_type = config.layer_types[layer_id]
            if layer_type == "linear_attention":
                _copy_zero(runtime, conv_buf, conv_zero)
                _copy_zero(runtime, recurrent_buf, recurrent_zero)
                weights = materialize_qwen35_paro_linear_attention_moe_c1_runtime_layer(index, layer_id=layer_id, runtime=runtime)
                state = Qwen35ParoDecodeState(layer_weights=weights, workspace=RuntimeWorkspace(runtime=runtime), runtime=runtime)
                out = state.run_linear_attention_moe_c1_layer_bf16(
                    hidden,
                    conv_state=conv_state,
                    recurrent_state=recurrent_state,
                )
            elif layer_type == "full_attention":
                _copy_zero(runtime, key_cache_buf, key_cache_arr)
                _copy_zero(runtime, value_cache_buf, value_cache_arr)
                weights = materialize_qwen35_paro_full_attention_moe_c1_runtime_layer(index, layer_id=layer_id, runtime=runtime)
                state = Qwen35ParoDecodeState(layer_weights=weights, workspace=RuntimeWorkspace(runtime=runtime), runtime=runtime)
                out = state.run_full_attention_moe_c1_layer_bf16(
                    hidden,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    append_spans=append_spans,
                    decode_spans=decode_spans,
                    cos_table=cos,
                    sin_table=sin,
                    position=position,
                    max_positions=1,
                )
            else:
                raise ValueError(f"unsupported layer type {layer_type!r} at layer {layer_id}")
            runtime.memcpy(next_hidden.ptr, out.ptr, hidden_bits.nbytes, HipMemcpyKind.DEVICE_TO_DEVICE)
            state.free()
            hidden, next_hidden = next_hidden, hidden
            layer_records.append({"layer": layer_id, "type": layer_type})

        norm_bits = float_array_to_bf16_bits(_read_tensor(normalized, "language_model.norm.weight"))
        norm_weight = load_host_array_to_device_as_dtype("model.norm.weight", norm_bits, DType.BF16, runtime=runtime)
        states_to_free.append(norm_weight)
        norm_out_buf = malloc(hidden_bits.nbytes, runtime=runtime)
        buffers.append(norm_out_buf)
        norm_out = Tensor.from_handle(norm_out_buf.ptr, hidden_bits.shape, DType.BF16, device)
        paro_rmsnorm_out_bf16(hidden.ptr, norm_weight.tensor.ptr, norm_out.ptr, 1, config.hidden_size, config.rms_norm_eps, runtime=runtime)
        runtime.device_synchronize()
        final_bits = np.empty(hidden_bits.shape, dtype=np.uint16)
        copy_device_to_host(host_array_ptr(final_bits), DeviceBuffer(norm_out.ptr, final_bits.nbytes), runtime=runtime)
        final_hidden = _bf16_bits_to_float32(final_bits.reshape(-1))
        next_id, next_logit = _lm_head_argmax(normalized, final_hidden, chunk_size=args.lm_head_chunk)
        decoded = _decode_token(model, next_id)
        print(
            json.dumps(
                {
                    "model": str(model),
                    "prompt": args.prompt,
                    "prompt_ids": prompt_ids,
                    "input_token_id": token_id,
                    "layers_run": layer_records,
                    "next_token_id": next_id,
                    "next_token_text": decoded,
                    "next_token_logit": next_logit,
                    "lm_head": "cpu_numpy_argmax",
                },
                ensure_ascii=False,
            )
        )
    finally:
        for allocation in reversed(states_to_free):
            allocation.free(runtime=runtime)
        for buf in reversed(buffers):
            free(buf, runtime=runtime)
    return 0


def _normalized_infos(index) -> dict[str, Any]:
    out = {}
    for name, info in index.tensors.items():
        out[normalize_qwen35_weight_name(name)] = info
    return out


def _read_tensor(normalized: dict[str, Any], name: str) -> np.ndarray:
    key = normalize_qwen35_weight_name(name)
    info = normalized[key]
    with safe_open(str(info.shard_path), framework="numpy") as handle:
        return np.ascontiguousarray(handle.get_tensor(info.name))


def _select_token(model: Path, prompt: str, token_id: int | None) -> tuple[int, list[int]]:
    if token_id is not None:
        return int(token_id), [int(token_id)]
    try:
        from tokenizers import Tokenizer
    except Exception as exc:  # pragma: no cover - optional runtime dependency guard
        raise RuntimeError("tokenizers is required unless --token-id is supplied") from exc
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    ids = tokenizer.encode(prompt).ids
    if not ids:
        raise ValueError("prompt produced no tokens")
    return int(ids[-1]), [int(x) for x in ids]


def _decode_token(model: Path, token_id: int) -> str:
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
        return tokenizer.decode([int(token_id)])
    except Exception:
        return ""


def _copy_zero(runtime, buffer: DeviceBuffer, zeros: np.ndarray) -> None:
    copy_host_to_device(buffer, host_array_ptr(zeros), runtime=runtime)


def _rope_tables(*, max_positions: int, rotary_dim: int, base: float) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    dims = np.arange(rotary_dim // 2, dtype=np.float32)[None, :]
    inv_freq = np.power(np.float32(base), -2.0 * dims / np.float32(rotary_dim))
    freqs = positions * inv_freq
    cos_half = np.cos(freqs).astype(np.float32, copy=False)
    sin_half = np.sin(freqs).astype(np.float32, copy=False)
    cos = np.concatenate([cos_half, cos_half], axis=1).astype(np.float32, copy=False)
    sin = np.concatenate([sin_half, sin_half], axis=1).astype(np.float32, copy=False)
    return np.ascontiguousarray(cos), np.ascontiguousarray(sin)


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _lm_head_argmax(normalized: dict[str, Any], hidden: np.ndarray, *, chunk_size: int) -> tuple[int, float]:
    info = normalized["lm_head.weight"]
    best_id = -1
    best_logit = -float("inf")
    with safe_open(str(info.shard_path), framework="numpy") as handle:
        weight = handle.get_tensor(info.name)
        rows = int(weight.shape[0])
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            logits = weight[start:end].astype(np.float32) @ hidden.astype(np.float32)
            local = int(np.argmax(logits))
            value = float(logits[local])
            if value > best_logit:
                best_logit = value
                best_id = start + local
    return best_id, best_logit


if __name__ == "__main__":
    raise SystemExit(main())
