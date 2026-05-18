#!/usr/bin/env python3
"""Smoke-test native DFlash drafter root/query primitives on HIP.

This covers the pieces currently landed before full decoder-block wiring:
root/mask id+position+embedding prep (BF16 copy and FP16->BF16 conversion) and
non-causal grouped-query attention over pre-projected q/k/v tensors. It is a
fixture for kernel correctness only; full DFlash top1/topk parity still requires
wiring q/k/v projections, rotary, MLP, final norm, and lm-head.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.speculative import (
    build_dflash_drafter,
    dflash_dense_bf16_to_f32,
    dflash_gqa_attention_f32_bf16,
    dflash_prepare_noise_inputs_bf16_i32,
    dflash_prepare_noise_inputs_f16_to_bf16_i32,
    dflash_rmsnorm_bf16,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    args = parser.parse_args()
    compiler_version = args.compiler_version_file.read_text(encoding="utf-8") if args.compiler_version_file else None
    runtime = get_hip_runtime()
    library = build_dflash_drafter(
        load=True,
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    _smoke_prepare_noise(runtime, library)
    _smoke_rmsnorm(runtime, library)
    _smoke_dense_projection(runtime, library)
    _smoke_gqa_attention(runtime, library)
    print("dflash_drafter_root_query_smoke passed")
    return 0


def _smoke_prepare_noise(runtime, library) -> None:
    roots = np.array([3, 5], dtype=np.int32)
    positions = np.array([7, 11], dtype=np.int32)
    vocab = 9
    hidden = 6
    block = 4
    embed_bf16 = np.arange(vocab * hidden, dtype=np.uint16).reshape(vocab, hidden) + np.uint16(100)
    embed_f16 = (np.arange(vocab * hidden, dtype=np.float16).reshape(vocab, hidden) / np.float16(8.0)).astype(np.float16)
    cases = (
        ("bf16", dflash_prepare_noise_inputs_bf16_i32, embed_bf16, embed_bf16),
        ("f16_to_bf16", dflash_prepare_noise_inputs_f16_to_bf16_i32, embed_f16, _f32_to_bf16_bits(embed_f16)),
    )
    for name, fn, embedding, expected_table in cases:
        ids = np.empty((2, block), dtype=np.int32)
        pos_out = np.empty((2, block), dtype=np.int32)
        emb = np.empty((2, block, hidden), dtype=np.uint16)
        buffers = []
        try:
            roots_dev = _dev(runtime, buffers, roots)
            pos_dev = _dev(runtime, buffers, positions)
            embed_dev = _dev(runtime, buffers, embedding)
            ids_dev = _empty(runtime, buffers, ids)
            pos_out_dev = _empty(runtime, buffers, pos_out)
            emb_dev = _empty(runtime, buffers, emb)
            fn(
                roots_dev.ptr,
                pos_dev.ptr,
                embed_dev.ptr,
                ids_dev.ptr,
                pos_out_dev.ptr,
                emb_dev.ptr,
                2,
                block,
                hidden,
                vocab,
                mask_token_id=8,
                threads=64,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(host_array_ptr(ids), ids_dev, runtime=runtime)
            copy_device_to_host(host_array_ptr(pos_out), pos_out_dev, runtime=runtime)
            copy_device_to_host(host_array_ptr(emb), emb_dev, runtime=runtime)
        finally:
            _free_all(runtime, buffers)
        assert ids.tolist() == [[3, 8, 8, 8], [5, 8, 8, 8]]
        assert pos_out.tolist() == [[7, 8, 9, 10], [11, 12, 13, 14]]
        np.testing.assert_array_equal(emb[0, 0], expected_table[3])
        np.testing.assert_array_equal(emb[0, 1], expected_table[8])
        np.testing.assert_array_equal(emb[1, 0], expected_table[5])
        print(f"prepare_noise_{name}: ids={ids.tolist()} positions={pos_out.tolist()}")


def _smoke_rmsnorm(runtime, library) -> None:
    rng = np.random.default_rng(2)
    hidden = _f32_to_bf16_bits(rng.normal(size=(3, 8)).astype(np.float32) * 0.5)
    weight = _f32_to_bf16_bits(0.75 + rng.random(size=(8,)).astype(np.float32) * 0.5)
    out = np.empty_like(hidden)
    hidden_f = _bf16_bits_to_f32(hidden)
    weight_f = _bf16_bits_to_f32(weight)
    rms = np.sqrt(np.mean(hidden_f * hidden_f, axis=1, keepdims=True) + 1.0e-6).astype(np.float32)
    expected = _f32_to_bf16_bits((hidden_f / rms) * weight_f)
    buffers = []
    try:
        hidden_dev = _dev(runtime, buffers, hidden)
        weight_dev = _dev(runtime, buffers, weight)
        out_dev = _empty(runtime, buffers, out)
        dflash_rmsnorm_bf16(
            hidden_dev.ptr,
            weight_dev.ptr,
            out_dev.ptr,
            rows=3,
            hidden_size=8,
            eps=1.0e-6,
            threads=64,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
    finally:
        _free_all(runtime, buffers)
    np.testing.assert_array_equal(out, expected)
    print(f"rmsnorm_bf16: sample={out.reshape(-1)[:4].tolist()}")


def _smoke_dense_projection(runtime, library) -> None:
    rng = np.random.default_rng(3)
    hidden_f32 = rng.normal(size=(3, 8)).astype(np.float32) * 0.5
    weight_f32 = rng.normal(size=(5, 8)).astype(np.float32) * 0.25
    hidden = _f32_to_bf16_bits(hidden_f32)
    weight = _f32_to_bf16_bits(weight_f32)
    out = np.empty((3, 5), dtype=np.float32)
    expected = _bf16_bits_to_f32(hidden).astype(np.float32) @ _bf16_bits_to_f32(weight).astype(np.float32).T
    buffers = []
    try:
        hidden_dev = _dev(runtime, buffers, hidden)
        weight_dev = _dev(runtime, buffers, weight)
        out_dev = _empty(runtime, buffers, out)
        dflash_dense_bf16_to_f32(
            hidden_dev.ptr,
            weight_dev.ptr,
            out_dev.ptr,
            rows=3,
            in_features=8,
            out_features=5,
            threads=64,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
    finally:
        _free_all(runtime, buffers)
    max_abs = float(np.max(np.abs(out - expected)))
    assert max_abs <= 1.0e-5, max_abs
    print(f"dense_bf16_to_f32: max_abs={max_abs:.3e} sample={out.reshape(-1)[:4].tolist()}")


def _smoke_gqa_attention(runtime, library) -> None:
    rng = np.random.default_rng(1)
    query = (rng.normal(size=(1, 2, 4, 8)).astype(np.float32) * 0.25).astype(np.float32)
    key = (rng.normal(size=(1, 3, 2, 8)).astype(np.float32) * 0.25).astype(np.float32)
    value = _f32_to_bf16_bits(rng.normal(size=(1, 3, 2, 8)).astype(np.float32) * 0.5)
    out = np.empty((1, 2, 4, 8), dtype=np.uint16)
    expected = _attention_oracle(query, key, value, scale=8**-0.5)
    buffers = []
    try:
        q_dev = _dev(runtime, buffers, query)
        k_dev = _dev(runtime, buffers, key)
        v_dev = _dev(runtime, buffers, value)
        out_dev = _empty(runtime, buffers, out)
        dflash_gqa_attention_f32_bf16(
            q_dev.ptr,
            k_dev.ptr,
            v_dev.ptr,
            out_dev.ptr,
            1,
            2,
            3,
            4,
            2,
            8,
            threads=64,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
    finally:
        _free_all(runtime, buffers)
    np.testing.assert_array_equal(out, expected)
    diff = np.max(np.abs(_bf16_bits_to_f32(out) - _bf16_bits_to_f32(expected)))
    print(f"gqa_attention: max_abs={float(diff)} sample={out.reshape(-1)[:8].tolist()}")


def _attention_oracle(query: np.ndarray, key: np.ndarray, value_bf16: np.ndarray, *, scale: float) -> np.ndarray:
    batch, query_len, q_heads, head_dim = query.shape
    _, kv_len, kv_heads, _ = key.shape
    group = q_heads // kv_heads
    value = _bf16_bits_to_f32(value_bf16)
    out = np.zeros_like(query, dtype=np.float32)
    for b in range(batch):
        for q in range(query_len):
            for head in range(q_heads):
                kv_head = head // group
                scores = np.array(
                    [np.dot(query[b, q, head], key[b, k, kv_head]) * scale for k in range(kv_len)],
                    dtype=np.float32,
                )
                probs = np.exp(scores - np.max(scores))
                probs /= np.sum(probs)
                for k in range(kv_len):
                    out[b, q, head] += probs[k] * value[b, k, kv_head]
    return _f32_to_bf16_bits(out)


def _f32_to_bf16_bits(array: np.ndarray) -> np.ndarray:
    f32 = np.asarray(array, dtype=np.float32)
    u32 = f32.view(np.uint32)
    rounded = u32 + np.uint32(0x7FFF) + ((u32 >> 16) & 1).astype(np.uint32)
    return (rounded >> 16).astype(np.uint16)


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


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
