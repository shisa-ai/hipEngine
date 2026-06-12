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
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.kernels.hip_gfx1100.linear import build_lm_head, topk_f32_rows_i32
from hipengine.kernels.hip_gfx1100.speculative import (
    build_dflash_drafter,
    dflash_add_bf16,
    dflash_concat_rows_bf16,
    dflash_concat_rows_f32,
    dflash_dense_bf16_to_bf16,
    dflash_dense_bf16_to_f32,
    dflash_gqa_attention_f32_bf16,
    dflash_head_rmsnorm_rotary_f32,
    dflash_head_rmsnorm_rotary_indexed_key_f32,
    dflash_prepare_noise_inputs_bf16_i32,
    dflash_prepare_noise_inputs_f16_to_bf16_i32,
    dflash_qkv_proj_bf16_mixed,
    dflash_qkv_proj_bf16_mixed_indexed_v,
    dflash_rmsnorm_bf16,
    dflash_silu_mul_bf16,
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
    lm_head_library = build_lm_head(
        load=True,
        compiler_version=compiler_version,
        require_cached=args.require_cached_build,
    )
    _smoke_prepare_noise(runtime, library)
    _smoke_rmsnorm(runtime, library)
    _smoke_add_and_concat(runtime, library)
    _smoke_dense_projection(runtime, library)
    _smoke_silu_and_dense_bf16(runtime, library)
    _smoke_indexed_kv_write(runtime, library)
    _smoke_head_rotary(runtime, library)
    _smoke_gqa_attention(runtime, library)
    _smoke_tiny_decoder_topk(runtime, library, lm_head_library)
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


def _smoke_add_and_concat(runtime, library) -> None:
    a = _f32_to_bf16_bits(np.array([[1.0, -2.0, 3.5], [0.25, 0.5, -0.75]], dtype=np.float32))
    b = _f32_to_bf16_bits(np.array([[0.5, 1.0, -1.5], [0.75, -0.25, 0.25]], dtype=np.float32))
    out = np.empty_like(a)
    expected = _f32_to_bf16_bits(_bf16_bits_to_f32(a) + _bf16_bits_to_f32(b))
    ctx_f32 = np.arange(1 * 2 * 3, dtype=np.float32).reshape(1, 2, 3)
    qry_f32 = (np.arange(1 * 2 * 3, dtype=np.float32).reshape(1, 2, 3) + 100.0).astype(np.float32)
    cat_f32 = np.empty((1, 4, 3), dtype=np.float32)
    ctx_bf16 = _f32_to_bf16_bits(ctx_f32)
    qry_bf16 = _f32_to_bf16_bits(qry_f32)
    cat_bf16 = np.empty((1, 4, 3), dtype=np.uint16)
    buffers = []
    try:
        a_dev = _dev(runtime, buffers, a)
        b_dev = _dev(runtime, buffers, b)
        out_dev = _empty(runtime, buffers, out)
        ctx_f32_dev = _dev(runtime, buffers, ctx_f32)
        qry_f32_dev = _dev(runtime, buffers, qry_f32)
        cat_f32_dev = _empty(runtime, buffers, cat_f32)
        ctx_bf16_dev = _dev(runtime, buffers, ctx_bf16)
        qry_bf16_dev = _dev(runtime, buffers, qry_bf16)
        cat_bf16_dev = _empty(runtime, buffers, cat_bf16)
        dflash_add_bf16(a_dev.ptr, b_dev.ptr, out_dev.ptr, a.size, threads=64, library=library, runtime=runtime)
        dflash_concat_rows_f32(ctx_f32_dev.ptr, qry_f32_dev.ptr, cat_f32_dev.ptr, 1, 2, 2, 3, threads=64, library=library, runtime=runtime)
        dflash_concat_rows_bf16(ctx_bf16_dev.ptr, qry_bf16_dev.ptr, cat_bf16_dev.ptr, 1, 2, 2, 3, threads=64, library=library, runtime=runtime)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(out), out_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(cat_f32), cat_f32_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(cat_bf16), cat_bf16_dev, runtime=runtime)
    finally:
        _free_all(runtime, buffers)
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(cat_f32, np.concatenate([ctx_f32, qry_f32], axis=1))
    np.testing.assert_array_equal(cat_bf16, np.concatenate([ctx_bf16, qry_bf16], axis=1))
    print("add_concat: bf16_add and f32/bf16 row concat passed")


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


def _smoke_silu_and_dense_bf16(runtime, library) -> None:
    rng = np.random.default_rng(5)
    hidden_f32 = rng.normal(size=(3, 8)).astype(np.float32) * 0.35
    weight_f32 = rng.normal(size=(5, 8)).astype(np.float32) * 0.2
    hidden = _f32_to_bf16_bits(hidden_f32)
    weight = _f32_to_bf16_bits(weight_f32)
    dense_out = np.empty((3, 5), dtype=np.uint16)
    expected_dense = _f32_to_bf16_bits(_bf16_bits_to_f32(hidden).astype(np.float32) @ _bf16_bits_to_f32(weight).astype(np.float32).T)
    gate = _f32_to_bf16_bits(rng.normal(size=(3, 5)).astype(np.float32) * 0.5)
    up = _f32_to_bf16_bits(rng.normal(size=(3, 5)).astype(np.float32) * 0.5)
    silu_out = np.empty((3, 5), dtype=np.uint16)
    gate_f = _bf16_bits_to_f32(gate)
    expected_silu = _f32_to_bf16_bits((gate_f / (1.0 + np.exp(-gate_f))) * _bf16_bits_to_f32(up))
    buffers = []
    try:
        hidden_dev = _dev(runtime, buffers, hidden)
        weight_dev = _dev(runtime, buffers, weight)
        dense_dev = _empty(runtime, buffers, dense_out)
        gate_dev = _dev(runtime, buffers, gate)
        up_dev = _dev(runtime, buffers, up)
        silu_dev = _empty(runtime, buffers, silu_out)
        dflash_dense_bf16_to_bf16(
            hidden_dev.ptr,
            weight_dev.ptr,
            dense_dev.ptr,
            rows=3,
            in_features=8,
            out_features=5,
            threads=64,
            library=library,
            runtime=runtime,
        )
        dflash_silu_mul_bf16(gate_dev.ptr, up_dev.ptr, silu_dev.ptr, gate.size, threads=64, library=library, runtime=runtime)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(dense_out), dense_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(silu_out), silu_dev, runtime=runtime)
    finally:
        _free_all(runtime, buffers)
    np.testing.assert_array_equal(dense_out, expected_dense)
    np.testing.assert_array_equal(silu_out, expected_silu)
    print("dense_bf16_to_bf16+silu_mul: BF16 oracle checks passed")


def _smoke_indexed_kv_write(runtime, library) -> None:
    rng = np.random.default_rng(7)
    rows = 2
    hidden = 8
    q_features = 6
    kv_features = 4
    cache_rows = 5
    slot = np.array([1], dtype=np.int32)
    hidden_bf16 = _f32_to_bf16_bits(rng.normal(size=(rows, hidden)).astype(np.float32) * 0.35)
    q_weight = _f32_to_bf16_bits(rng.normal(size=(q_features, hidden)).astype(np.float32) * 0.2)
    k_weight = _f32_to_bf16_bits(rng.normal(size=(kv_features, hidden)).astype(np.float32) * 0.2)
    v_weight = _f32_to_bf16_bits(rng.normal(size=(kv_features, hidden)).astype(np.float32) * 0.2)
    q_direct = np.empty((rows, q_features), dtype=np.float32)
    k_direct = np.empty((rows, kv_features), dtype=np.float32)
    v_direct = np.empty((rows, kv_features), dtype=np.uint16)
    q_indexed = np.empty_like(q_direct)
    k_indexed = np.empty_like(k_direct)
    sentinel = np.full((cache_rows, kv_features), np.uint16(0xA55A), dtype=np.uint16)
    v_cache = sentinel.copy()
    buffers = []
    try:
        hidden_d = _dev(runtime, buffers, hidden_bf16)
        qw_d = _dev(runtime, buffers, q_weight)
        kw_d = _dev(runtime, buffers, k_weight)
        vw_d = _dev(runtime, buffers, v_weight)
        q_direct_d = _empty(runtime, buffers, q_direct)
        k_direct_d = _empty(runtime, buffers, k_direct)
        v_direct_d = _empty(runtime, buffers, v_direct)
        q_indexed_d = _empty(runtime, buffers, q_indexed)
        k_indexed_d = _empty(runtime, buffers, k_indexed)
        v_cache_d = _dev(runtime, buffers, v_cache)
        slot_d = _dev(runtime, buffers, slot)
        dflash_qkv_proj_bf16_mixed(
            hidden_d.ptr,
            qw_d.ptr,
            kw_d.ptr,
            vw_d.ptr,
            q_direct_d.ptr,
            k_direct_d.ptr,
            v_direct_d.ptr,
            rows,
            hidden,
            q_features,
            kv_features,
            threads=64,
            library=library,
            runtime=runtime,
        )
        dflash_qkv_proj_bf16_mixed_indexed_v(
            hidden_d.ptr,
            qw_d.ptr,
            kw_d.ptr,
            vw_d.ptr,
            q_indexed_d.ptr,
            k_indexed_d.ptr,
            v_cache_d.ptr,
            slot_d.ptr,
            cache_rows,
            rows,
            hidden,
            q_features,
            kv_features,
            threads=64,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(q_direct), q_direct_d, runtime=runtime)
        copy_device_to_host(host_array_ptr(k_direct), k_direct_d, runtime=runtime)
        copy_device_to_host(host_array_ptr(v_direct), v_direct_d, runtime=runtime)
        copy_device_to_host(host_array_ptr(q_indexed), q_indexed_d, runtime=runtime)
        copy_device_to_host(host_array_ptr(k_indexed), k_indexed_d, runtime=runtime)
        copy_device_to_host(host_array_ptr(v_cache), v_cache_d, runtime=runtime)
    finally:
        _free_all(runtime, buffers)
    np.testing.assert_array_equal(q_indexed, q_direct)
    np.testing.assert_array_equal(k_indexed, k_direct)
    np.testing.assert_array_equal(v_cache[slot[0] : slot[0] + rows], v_direct)
    np.testing.assert_array_equal(v_cache[: slot[0]], sentinel[: slot[0]])
    np.testing.assert_array_equal(v_cache[slot[0] + rows :], sentinel[slot[0] + rows :])

    query = rng.normal(size=(1, rows, 2, 4)).astype(np.float32) * 0.25
    key = rng.normal(size=(1, rows, 1, 4)).astype(np.float32) * 0.25
    q_weight_h = _f32_to_bf16_bits(0.75 + rng.random(size=(4,)).astype(np.float32) * 0.5)
    k_weight_h = _f32_to_bf16_bits(0.70 + rng.random(size=(4,)).astype(np.float32) * 0.5)
    max_positions = 8
    rotary_dim = 4
    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    dims = np.arange(rotary_dim // 2, dtype=np.float32)[None, :]
    inv_freq = np.power(np.float32(10000.0), -2.0 * dims / np.float32(rotary_dim))
    angles = positions * inv_freq
    cos_table = np.concatenate([np.cos(angles), np.cos(angles)], axis=1).astype(np.float32)
    sin_table = np.concatenate([np.sin(angles), np.sin(angles)], axis=1).astype(np.float32)
    q_positions = np.array([[2, 3]], dtype=np.int32)
    k_positions = np.array([[2, 3]], dtype=np.int32)
    q_rot_direct = np.empty_like(query)
    k_rot_direct = np.empty_like(key)
    q_rot_indexed = np.empty_like(query)
    k_cache = np.full((cache_rows, 1, 4), np.float32(-123.0), dtype=np.float32)
    k_sentinel = k_cache.copy()
    expected_q = _head_rotary_oracle(query, q_weight_h, cos_table, sin_table, q_positions)
    expected_k = _head_rotary_oracle(key, k_weight_h, cos_table, sin_table, k_positions)
    buffers = []
    try:
        q_d = _dev(runtime, buffers, query)
        k_d = _dev(runtime, buffers, key)
        qw_d = _dev(runtime, buffers, q_weight_h)
        kw_d = _dev(runtime, buffers, k_weight_h)
        cos_d = _dev(runtime, buffers, cos_table)
        sin_d = _dev(runtime, buffers, sin_table)
        qp_d = _dev(runtime, buffers, q_positions)
        kp_d = _dev(runtime, buffers, k_positions)
        qo_direct_d = _empty(runtime, buffers, q_rot_direct)
        ko_direct_d = _empty(runtime, buffers, k_rot_direct)
        qo_indexed_d = _empty(runtime, buffers, q_rot_indexed)
        k_cache_d = _dev(runtime, buffers, k_cache)
        slot_d = _dev(runtime, buffers, slot)
        dflash_head_rmsnorm_rotary_f32(
            q_d.ptr,
            k_d.ptr,
            qw_d.ptr,
            kw_d.ptr,
            cos_d.ptr,
            sin_d.ptr,
            qp_d.ptr,
            kp_d.ptr,
            qo_direct_d.ptr,
            ko_direct_d.ptr,
            1,
            rows,
            rows,
            2,
            1,
            4,
            rotary_dim,
            max_positions,
            eps=1.0e-6,
            threads=64,
            library=library,
            runtime=runtime,
        )
        dflash_head_rmsnorm_rotary_indexed_key_f32(
            q_d.ptr,
            k_d.ptr,
            qw_d.ptr,
            kw_d.ptr,
            cos_d.ptr,
            sin_d.ptr,
            qp_d.ptr,
            kp_d.ptr,
            qo_indexed_d.ptr,
            k_cache_d.ptr,
            slot_d.ptr,
            cache_rows,
            1,
            rows,
            rows,
            2,
            1,
            4,
            rotary_dim,
            max_positions,
            eps=1.0e-6,
            threads=64,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(q_rot_direct), qo_direct_d, runtime=runtime)
        copy_device_to_host(host_array_ptr(k_rot_direct), ko_direct_d, runtime=runtime)
        copy_device_to_host(host_array_ptr(q_rot_indexed), qo_indexed_d, runtime=runtime)
        copy_device_to_host(host_array_ptr(k_cache), k_cache_d, runtime=runtime)
    finally:
        _free_all(runtime, buffers)
    max_abs = max(float(np.max(np.abs(q_rot_direct - expected_q))), float(np.max(np.abs(k_rot_direct - expected_k))))
    assert max_abs <= 1.0e-6, max_abs
    np.testing.assert_array_equal(q_rot_indexed, q_rot_direct)
    np.testing.assert_array_equal(k_cache[slot[0] : slot[0] + rows], k_rot_direct.reshape(rows, 1, 4))
    np.testing.assert_array_equal(k_cache[: slot[0]], k_sentinel[: slot[0]])
    np.testing.assert_array_equal(k_cache[slot[0] + rows :], k_sentinel[slot[0] + rows :])
    print("indexed_kv_write: QKV V-cache and rotary K-cache slots match direct outputs")


def _smoke_head_rotary(runtime, library) -> None:
    rng = np.random.default_rng(4)
    query = rng.normal(size=(1, 2, 4, 8)).astype(np.float32) * 0.25
    key = rng.normal(size=(1, 3, 2, 8)).astype(np.float32) * 0.25
    q_weight = _f32_to_bf16_bits(0.75 + rng.random(size=(8,)).astype(np.float32) * 0.5)
    k_weight = _f32_to_bf16_bits(0.70 + rng.random(size=(8,)).astype(np.float32) * 0.5)
    max_positions = 16
    rotary_dim = 8
    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    dims = np.arange(rotary_dim // 2, dtype=np.float32)[None, :]
    inv_freq = np.power(np.float32(10000.0), -2.0 * dims / np.float32(rotary_dim))
    angles = positions * inv_freq
    cos_half = np.cos(angles).astype(np.float32)
    sin_half = np.sin(angles).astype(np.float32)
    cos_table = np.concatenate([cos_half, cos_half], axis=1).astype(np.float32)
    sin_table = np.concatenate([sin_half, sin_half], axis=1).astype(np.float32)
    query_positions = np.array([[5, 6]], dtype=np.int32)
    key_positions = np.array([[3, 4, 5]], dtype=np.int32)
    query_out = np.empty_like(query)
    key_out = np.empty_like(key)
    expected_q = _head_rotary_oracle(query, q_weight, cos_table, sin_table, query_positions)
    expected_k = _head_rotary_oracle(key, k_weight, cos_table, sin_table, key_positions)
    buffers = []
    try:
        q_dev = _dev(runtime, buffers, query)
        k_dev = _dev(runtime, buffers, key)
        qw_dev = _dev(runtime, buffers, q_weight)
        kw_dev = _dev(runtime, buffers, k_weight)
        cos_dev = _dev(runtime, buffers, cos_table)
        sin_dev = _dev(runtime, buffers, sin_table)
        qp_dev = _dev(runtime, buffers, query_positions)
        kp_dev = _dev(runtime, buffers, key_positions)
        qo_dev = _empty(runtime, buffers, query_out)
        ko_dev = _empty(runtime, buffers, key_out)
        dflash_head_rmsnorm_rotary_f32(
            q_dev.ptr,
            k_dev.ptr,
            qw_dev.ptr,
            kw_dev.ptr,
            cos_dev.ptr,
            sin_dev.ptr,
            qp_dev.ptr,
            kp_dev.ptr,
            qo_dev.ptr,
            ko_dev.ptr,
            1,
            2,
            3,
            4,
            2,
            8,
            rotary_dim,
            max_positions,
            eps=1.0e-6,
            threads=64,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(query_out), qo_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(key_out), ko_dev, runtime=runtime)
    finally:
        _free_all(runtime, buffers)
    max_abs = max(float(np.max(np.abs(query_out - expected_q))), float(np.max(np.abs(key_out - expected_k))))
    assert max_abs <= 1.0e-6, max_abs
    print(f"head_rmsnorm_rotary: max_abs={max_abs:.3e} sample={query_out.reshape(-1)[:4].tolist()}")


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


def _head_rotary_oracle(raw: np.ndarray, weight_bf16: np.ndarray, cos_table: np.ndarray, sin_table: np.ndarray, positions: np.ndarray) -> np.ndarray:
    weight = _bf16_bits_to_f32(weight_bf16)
    out = np.empty_like(raw, dtype=np.float32)
    rotary_dim = cos_table.shape[1]
    half = rotary_dim // 2
    for b in range(raw.shape[0]):
        for row in range(raw.shape[1]):
            cos = cos_table[int(positions[b, row])]
            sin = sin_table[int(positions[b, row])]
            for head in range(raw.shape[2]):
                src = raw[b, row, head]
                inv = np.float32(1.0 / np.sqrt(np.mean(src * src) + 1.0e-6))
                normed = src * inv * weight
                dst = normed.copy()
                for dim in range(rotary_dim):
                    pair = dim + half if dim < half else dim - half
                    rotated = -normed[pair] if dim < half else normed[pair]
                    dst[dim] = normed[dim] * cos[dim] + rotated * sin[dim]
                out[b, row, head] = dst
    return out


def _smoke_tiny_decoder_topk(runtime, library, lm_head_library) -> None:
    rng = np.random.default_rng(6)
    batch = 1
    context_len = 2
    query_len = 3
    hidden = 8
    q_heads = 2
    kv_heads = 1
    head_dim = 4
    intermediate = 10
    vocab = 11
    top_k = 3
    noise = _f32_to_bf16_bits(rng.normal(size=(query_len, hidden)).astype(np.float32) * 0.35)
    target_hidden = _f32_to_bf16_bits(rng.normal(size=(context_len, hidden)).astype(np.float32) * 0.30)
    weights = {
        "in_norm": _f32_to_bf16_bits(0.8 + rng.random(hidden).astype(np.float32) * 0.4),
        "post_norm": _f32_to_bf16_bits(0.8 + rng.random(hidden).astype(np.float32) * 0.4),
        "final_norm": _f32_to_bf16_bits(0.8 + rng.random(hidden).astype(np.float32) * 0.4),
        "q_norm": _f32_to_bf16_bits(0.8 + rng.random(head_dim).astype(np.float32) * 0.4),
        "k_norm": _f32_to_bf16_bits(0.8 + rng.random(head_dim).astype(np.float32) * 0.4),
        "q": _f32_to_bf16_bits(rng.normal(size=(q_heads * head_dim, hidden)).astype(np.float32) * 0.22),
        "k": _f32_to_bf16_bits(rng.normal(size=(kv_heads * head_dim, hidden)).astype(np.float32) * 0.22),
        "v": _f32_to_bf16_bits(rng.normal(size=(kv_heads * head_dim, hidden)).astype(np.float32) * 0.22),
        "o": _f32_to_bf16_bits(rng.normal(size=(hidden, q_heads * head_dim)).astype(np.float32) * 0.18),
        "gate": _f32_to_bf16_bits(rng.normal(size=(intermediate, hidden)).astype(np.float32) * 0.18),
        "up": _f32_to_bf16_bits(rng.normal(size=(intermediate, hidden)).astype(np.float32) * 0.18),
        "down": _f32_to_bf16_bits(rng.normal(size=(hidden, intermediate)).astype(np.float32) * 0.16),
        "lm": _f32_to_bf16_bits(rng.normal(size=(vocab, hidden)).astype(np.float32) * 0.20),
    }
    max_positions = 16
    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    dims = np.arange(head_dim // 2, dtype=np.float32)[None, :]
    inv_freq = np.power(np.float32(10000.0), -2.0 * dims / np.float32(head_dim))
    angles = positions * inv_freq
    cos_half = np.cos(angles).astype(np.float32)
    sin_half = np.sin(angles).astype(np.float32)
    cos_table = np.concatenate([cos_half, cos_half], axis=1).astype(np.float32)
    sin_table = np.concatenate([sin_half, sin_half], axis=1).astype(np.float32)
    query_positions = np.arange(context_len, context_len + query_len, dtype=np.int32).reshape(1, query_len)
    key_positions = np.arange(context_len + query_len, dtype=np.int32).reshape(1, context_len + query_len)

    expected_hidden, expected_logits = _tiny_decoder_oracle(
        noise,
        target_hidden,
        weights,
        cos_table,
        sin_table,
        query_positions,
        key_positions,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    expected_topk = _topk_indices(expected_logits[1:], top_k)
    parent_fixture = json.loads((REPO_ROOT / "fixtures/dflash/drafter_root_query_parent_fixture.json").read_text(encoding="utf-8"))
    parent_topk = np.asarray(parent_fixture["expected_topk_ids"], dtype=np.int32)
    parent_logits = np.asarray(parent_fixture["expected_logits"], dtype=np.float32)

    buffers = []
    try:
        dev = lambda arr: _dev(runtime, buffers, arr)
        empty = lambda arr: _empty(runtime, buffers, arr)
        noise_d = dev(noise)
        target_d = dev(target_hidden)
        wd = {name: dev(value) for name, value in weights.items()}
        cos_d = dev(cos_table)
        sin_d = dev(sin_table)
        qpos_d = dev(query_positions)
        kpos_d = dev(key_positions)
        norm = np.empty_like(noise)
        q_raw = np.empty((query_len, q_heads * head_dim), dtype=np.float32)
        k_ctx = np.empty((context_len, kv_heads * head_dim), dtype=np.float32)
        k_q = np.empty((query_len, kv_heads * head_dim), dtype=np.float32)
        k_all = np.empty((batch, context_len + query_len, kv_heads * head_dim), dtype=np.float32)
        v_ctx = np.empty((context_len, kv_heads * head_dim), dtype=np.uint16)
        v_q = np.empty((query_len, kv_heads * head_dim), dtype=np.uint16)
        v_all = np.empty((batch, context_len + query_len, kv_heads * head_dim), dtype=np.uint16)
        q_rot = np.empty((batch, query_len, q_heads, head_dim), dtype=np.float32)
        k_rot = np.empty((batch, context_len + query_len, kv_heads, head_dim), dtype=np.float32)
        attn = np.empty((batch, query_len, q_heads, head_dim), dtype=np.uint16)
        attn_proj = np.empty_like(noise)
        hidden_attn = np.empty_like(noise)
        post = np.empty_like(noise)
        gate = np.empty((query_len, intermediate), dtype=np.uint16)
        up = np.empty((query_len, intermediate), dtype=np.uint16)
        act = np.empty((query_len, intermediate), dtype=np.uint16)
        mlp = np.empty_like(noise)
        hidden_out = np.empty_like(noise)
        final_norm = np.empty_like(noise)
        logits = np.empty((query_len - 1, vocab), dtype=np.float32)
        topk_ids = np.empty((query_len - 1, top_k), dtype=np.int32)
        topk_values = np.empty((query_len - 1, top_k), dtype=np.float32)
        norm_d = empty(norm); q_raw_d = empty(q_raw); k_ctx_d = empty(k_ctx); k_q_d = empty(k_q)
        k_all_d = empty(k_all); v_ctx_d = empty(v_ctx); v_q_d = empty(v_q); v_all_d = empty(v_all)
        q_rot_d = empty(q_rot); k_rot_d = empty(k_rot); attn_d = empty(attn); attn_proj_d = empty(attn_proj)
        hidden_attn_d = empty(hidden_attn); post_d = empty(post); gate_d = empty(gate); up_d = empty(up); act_d = empty(act)
        mlp_d = empty(mlp); hidden_out_d = empty(hidden_out); final_norm_d = empty(final_norm)
        logits_d = empty(logits); topk_ids_d = empty(topk_ids); topk_values_d = empty(topk_values)

        dflash_rmsnorm_bf16(noise_d.ptr, wd["in_norm"].ptr, norm_d.ptr, query_len, hidden, threads=64, library=library, runtime=runtime)
        dflash_dense_bf16_to_f32(norm_d.ptr, wd["q"].ptr, q_raw_d.ptr, query_len, hidden, q_heads * head_dim, threads=64, library=library, runtime=runtime)
        dflash_dense_bf16_to_f32(target_d.ptr, wd["k"].ptr, k_ctx_d.ptr, context_len, hidden, kv_heads * head_dim, threads=64, library=library, runtime=runtime)
        dflash_dense_bf16_to_f32(norm_d.ptr, wd["k"].ptr, k_q_d.ptr, query_len, hidden, kv_heads * head_dim, threads=64, library=library, runtime=runtime)
        dflash_dense_bf16_to_bf16(target_d.ptr, wd["v"].ptr, v_ctx_d.ptr, context_len, hidden, kv_heads * head_dim, threads=64, library=library, runtime=runtime)
        dflash_dense_bf16_to_bf16(norm_d.ptr, wd["v"].ptr, v_q_d.ptr, query_len, hidden, kv_heads * head_dim, threads=64, library=library, runtime=runtime)
        dflash_concat_rows_f32(k_ctx_d.ptr, k_q_d.ptr, k_all_d.ptr, batch, context_len, query_len, kv_heads * head_dim, threads=64, library=library, runtime=runtime)
        dflash_concat_rows_bf16(v_ctx_d.ptr, v_q_d.ptr, v_all_d.ptr, batch, context_len, query_len, kv_heads * head_dim, threads=64, library=library, runtime=runtime)
        dflash_head_rmsnorm_rotary_f32(q_raw_d.ptr, k_all_d.ptr, wd["q_norm"].ptr, wd["k_norm"].ptr, cos_d.ptr, sin_d.ptr, qpos_d.ptr, kpos_d.ptr, q_rot_d.ptr, k_rot_d.ptr, batch, query_len, context_len + query_len, q_heads, kv_heads, head_dim, head_dim, max_positions, threads=64, library=library, runtime=runtime)
        dflash_gqa_attention_f32_bf16(q_rot_d.ptr, k_rot_d.ptr, v_all_d.ptr, attn_d.ptr, batch, query_len, context_len + query_len, q_heads, kv_heads, head_dim, threads=64, library=library, runtime=runtime)
        dflash_dense_bf16_to_bf16(attn_d.ptr, wd["o"].ptr, attn_proj_d.ptr, query_len, hidden, hidden, threads=64, library=library, runtime=runtime)
        dflash_add_bf16(noise_d.ptr, attn_proj_d.ptr, hidden_attn_d.ptr, noise.size, threads=64, library=library, runtime=runtime)
        dflash_rmsnorm_bf16(hidden_attn_d.ptr, wd["post_norm"].ptr, post_d.ptr, query_len, hidden, threads=64, library=library, runtime=runtime)
        dflash_dense_bf16_to_bf16(post_d.ptr, wd["gate"].ptr, gate_d.ptr, query_len, hidden, intermediate, threads=64, library=library, runtime=runtime)
        dflash_dense_bf16_to_bf16(post_d.ptr, wd["up"].ptr, up_d.ptr, query_len, hidden, intermediate, threads=64, library=library, runtime=runtime)
        dflash_silu_mul_bf16(gate_d.ptr, up_d.ptr, act_d.ptr, act.size, threads=64, library=library, runtime=runtime)
        dflash_dense_bf16_to_bf16(act_d.ptr, wd["down"].ptr, mlp_d.ptr, query_len, intermediate, hidden, threads=64, library=library, runtime=runtime)
        dflash_add_bf16(hidden_attn_d.ptr, mlp_d.ptr, hidden_out_d.ptr, noise.size, threads=64, library=library, runtime=runtime)
        dflash_rmsnorm_bf16(hidden_out_d.ptr, wd["final_norm"].ptr, final_norm_d.ptr, query_len, hidden, threads=64, library=library, runtime=runtime)
        dflash_dense_bf16_to_f32(final_norm_d.ptr + hidden * 2, wd["lm"].ptr, logits_d.ptr, query_len - 1, hidden, vocab, threads=64, library=library, runtime=runtime)
        topk_f32_rows_i32(logits_d.ptr, topk_values_d.ptr, topk_ids_d.ptr, query_len - 1, vocab, top_k, threads=128, library=lm_head_library, runtime=runtime)
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(final_norm), final_norm_d, runtime=runtime)
        copy_device_to_host(host_array_ptr(logits), logits_d, runtime=runtime)
        copy_device_to_host(host_array_ptr(topk_ids), topk_ids_d, runtime=runtime)
    finally:
        _free_all(runtime, buffers)

    final_abs = float(np.max(np.abs(_bf16_bits_to_f32(final_norm) - _bf16_bits_to_f32(expected_hidden))))
    logits_abs = float(np.max(np.abs(logits - expected_logits[1:])))
    parent_abs = float(np.max(np.abs(logits - parent_logits)))
    assert final_abs <= 2.0e-2, final_abs
    assert logits_abs <= 5.0e-2, logits_abs
    assert parent_abs <= float(parent_fixture["tolerance"]["native_vs_parent_logits_max_abs"]), parent_abs
    np.testing.assert_array_equal(topk_ids, expected_topk)
    np.testing.assert_array_equal(topk_ids, parent_topk)
    print(
        f"tiny_decoder_topk: final_abs={final_abs:.3e} logits_abs={logits_abs:.3e} "
        f"parent_abs={parent_abs:.3e} topk={topk_ids.tolist()}"
    )


def _tiny_decoder_oracle(
    noise: np.ndarray,
    target_hidden: np.ndarray,
    weights: dict[str, np.ndarray],
    cos_table: np.ndarray,
    sin_table: np.ndarray,
    query_positions: np.ndarray,
    key_positions: np.ndarray,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = noise
    norm = _rmsnorm_oracle(hidden, weights["in_norm"])
    q_raw = _dense_bf16_to_f32_oracle(norm, weights["q"]).reshape(1, hidden.shape[0], q_heads, head_dim)
    k_ctx = _dense_bf16_to_f32_oracle(target_hidden, weights["k"])
    k_q = _dense_bf16_to_f32_oracle(norm, weights["k"])
    k_all = np.concatenate([k_ctx, k_q], axis=0).reshape(1, target_hidden.shape[0] + hidden.shape[0], kv_heads, head_dim)
    v_ctx = _dense_bf16_to_bf16_oracle(target_hidden, weights["v"])
    v_q = _dense_bf16_to_bf16_oracle(norm, weights["v"])
    v_all = np.concatenate([v_ctx, v_q], axis=0).reshape(1, target_hidden.shape[0] + hidden.shape[0], kv_heads, head_dim)
    q_rot = _head_rotary_oracle(q_raw, weights["q_norm"], cos_table, sin_table, query_positions)
    k_rot = _head_rotary_oracle(k_all, weights["k_norm"], cos_table, sin_table, key_positions)
    attn = _attention_oracle(q_rot, k_rot, v_all, scale=head_dim**-0.5).reshape(hidden.shape)
    attn_proj = _dense_bf16_to_bf16_oracle(attn, weights["o"])
    hidden = _add_bf16_oracle(hidden, attn_proj)
    post = _rmsnorm_oracle(hidden, weights["post_norm"])
    gate = _dense_bf16_to_bf16_oracle(post, weights["gate"])
    up = _dense_bf16_to_bf16_oracle(post, weights["up"])
    act = _silu_mul_oracle(gate, up)
    mlp = _dense_bf16_to_bf16_oracle(act, weights["down"])
    hidden = _add_bf16_oracle(hidden, mlp)
    final = _rmsnorm_oracle(hidden, weights["final_norm"])
    logits = _dense_bf16_to_f32_oracle(final, weights["lm"])
    return final, logits


def _dense_bf16_to_f32_oracle(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return _bf16_bits_to_f32(x).astype(np.float32) @ _bf16_bits_to_f32(weight).astype(np.float32).T


def _dense_bf16_to_bf16_oracle(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return _f32_to_bf16_bits(_dense_bf16_to_f32_oracle(x, weight))


def _rmsnorm_oracle(x: np.ndarray, weight: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    xf = _bf16_bits_to_f32(x)
    wf = _bf16_bits_to_f32(weight)
    rms = np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps).astype(np.float32)
    return _f32_to_bf16_bits((xf / rms) * wf)


def _add_bf16_oracle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _f32_to_bf16_bits(_bf16_bits_to_f32(a) + _bf16_bits_to_f32(b))


def _silu_mul_oracle(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    gf = _bf16_bits_to_f32(gate)
    return _f32_to_bf16_bits((gf / (1.0 + np.exp(-gf))) * _bf16_bits_to_f32(up))


def _topk_indices(logits: np.ndarray, top_k: int) -> np.ndarray:
    out = np.empty((logits.shape[0], top_k), dtype=np.int32)
    for row, values in enumerate(logits):
        order = sorted(range(values.shape[0]), key=lambda idx: (-float(values[idx]), idx))[:top_k]
        out[row] = np.asarray(order, dtype=np.int32)
    return out


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
