#!/usr/bin/env python3
"""Smoke-test append-only DFlash draft context K/V materialization on HIP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.device import Device
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.speculative import build_dflash_drafter
from hipengine.speculative import (
    DFlashDraftKVCacheOwner,
    DFlashDraftKVCacheSpec,
    DFlashDraftKVLayerWeights,
    DFlashDraftKVMaterializerScratch,
    full_context_kv_reference,
    materialize_dflash_draft_kv_append_from_projected,
    plan_dflash_draft_kv_append,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    args = parser.parse_args()
    compiler_version = args.compiler_version_file.read_text(encoding="utf-8") if args.compiler_version_file else None
    runtime = get_hip_runtime()
    library = build_dflash_drafter(load=True, compiler_version=compiler_version, require_cached=args.require_cached_build)
    _run_smoke(runtime, library)
    print("dflash_context_kv_materializer_smoke passed")
    return 0


def _run_smoke(runtime, library) -> None:
    rng = np.random.default_rng(7)
    device = Device("hip", 0)
    layers = 2
    capacity = 6
    hidden = 8
    rows = 4
    kv_heads = 2
    head_dim = 4
    kv_features = kv_heads * head_dim
    max_positions = 16
    projected = _f32_to_bf16_bits(rng.normal(size=(rows, hidden)).astype(np.float32) * 0.35)
    positions_all = np.array([3, 4, 5, 6], dtype=np.int32)
    k_proj = _f32_to_bf16_bits(rng.normal(size=(layers, kv_features, hidden)).astype(np.float32) * 0.20)
    v_proj = _f32_to_bf16_bits(rng.normal(size=(layers, kv_features, hidden)).astype(np.float32) * 0.18)
    k_norm = _f32_to_bf16_bits(0.8 + rng.random(size=(layers, head_dim)).astype(np.float32) * 0.4)
    cos_table, sin_table = _rotary_tables(max_positions=max_positions, rotary_dim=head_dim)
    full_keys = np.empty((layers, rows, kv_heads, head_dim), dtype=np.float32)
    full_values = np.empty((layers, rows, kv_heads, head_dim), dtype=np.uint16)
    for layer in range(layers):
        raw_k = _dense_bf16_to_f32(projected, k_proj[layer]).reshape(rows, kv_heads, head_dim)
        full_keys[layer] = _key_rotary_oracle(raw_k, k_norm[layer], cos_table, sin_table, positions_all)
        full_values[layer] = _dense_bf16_to_bf16(projected, v_proj[layer]).reshape(rows, kv_heads, head_dim)
    expected_keys, expected_values = full_context_kv_reference(full_keys, full_values, capacity_tokens=capacity)

    key_cache = np.full((layers, capacity, kv_heads, head_dim), -999.0, dtype=np.float32)
    value_cache = np.full((layers, capacity, kv_heads, head_dim), 65535, dtype=np.uint16)
    position_cache = np.full((capacity,), -1, dtype=np.int32)
    live_count = np.zeros((1,), dtype=np.int32)
    buffers = []
    try:
        key_cache_dev = _dev(runtime, buffers, key_cache)
        value_cache_dev = _dev(runtime, buffers, value_cache)
        position_cache_dev = _dev(runtime, buffers, position_cache)
        live_count_dev = _dev(runtime, buffers, live_count)
        owner = DFlashDraftKVCacheOwner(
            spec=DFlashDraftKVCacheSpec(
                backend="hip_gfx1151",
                bucket="smoke",
                device=device,
                layer_count=layers,
                capacity_tokens=capacity,
                num_kv_heads=kv_heads,
                head_dim=head_dim,
            ),
            keys=Tensor.from_handle(key_cache_dev.ptr, key_cache.shape, "fp32", device),
            values=Tensor.from_handle(value_cache_dev.ptr, value_cache.shape, "bf16", device),
            positions=Tensor.from_handle(position_cache_dev.ptr, position_cache.shape, "int32", device),
            live_count=Tensor.from_handle(live_count_dev.ptr, live_count.shape, "int32", device),
        )
        layer_weights = []
        for layer in range(layers):
            k_dev = _dev(runtime, buffers, k_proj[layer])
            v_dev = _dev(runtime, buffers, v_proj[layer])
            n_dev = _dev(runtime, buffers, k_norm[layer])
            layer_weights.append(
                DFlashDraftKVLayerWeights(
                    k_proj=Tensor.from_handle(k_dev.ptr, k_proj[layer].shape, "bf16", device),
                    v_proj=Tensor.from_handle(v_dev.ptr, v_proj[layer].shape, "bf16", device),
                    k_norm=Tensor.from_handle(n_dev.ptr, k_norm[layer].shape, "bf16", device),
                )
            )
        scratch_projected = _empty(runtime, buffers, np.empty((2, hidden), dtype=np.uint16))
        scratch_key = _empty(runtime, buffers, np.empty((2, kv_features), dtype=np.float32))
        scratch = DFlashDraftKVMaterializerScratch(
            projected_hidden=Tensor.from_handle(scratch_projected.ptr, (2, hidden), "bf16", device),
            key_raw=Tensor.from_handle(scratch_key.ptr, (2, kv_features), "fp32", device),
        )
        cos_dev = _dev(runtime, buffers, cos_table)
        sin_dev = _dev(runtime, buffers, sin_table)
        cos_tensor = Tensor.from_handle(cos_dev.ptr, cos_table.shape, "fp32", device)
        sin_tensor = Tensor.from_handle(sin_dev.ptr, sin_table.shape, "fp32", device)
        for start, count in ((0, 2), (2, 2)):
            proj_chunk = np.ascontiguousarray(projected[start : start + count])
            pos_chunk = np.ascontiguousarray(positions_all[start : start + count])
            proj_dev = _dev(runtime, buffers, proj_chunk)
            pos_dev = _dev(runtime, buffers, pos_chunk)
            result = materialize_dflash_draft_kv_append_from_projected(
                owner=owner,
                plan=plan_dflash_draft_kv_append(live_count=start, new_positions=tuple(int(x) for x in pos_chunk), capacity_tokens=capacity),
                projected_hidden=Tensor.from_handle(proj_dev.ptr, proj_chunk.shape, "bf16", device),
                positions=Tensor.from_handle(pos_dev.ptr, pos_chunk.shape, "int32", device),
                layer_weights=layer_weights,
                scratch=scratch,
                cos_table=cos_tensor,
                sin_table=sin_tensor,
                library=library,
                runtime=runtime,
                threads=64,
            )
            assert result.live_count == start + count
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(key_cache), key_cache_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(value_cache), value_cache_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(position_cache), position_cache_dev, runtime=runtime)
        copy_device_to_host(host_array_ptr(live_count), live_count_dev, runtime=runtime)
    finally:
        _free_all(runtime, buffers)
    np.testing.assert_allclose(key_cache[:, :rows], expected_keys[:, :rows], rtol=0, atol=2.0e-6)
    np.testing.assert_array_equal(value_cache[:, :rows], expected_values[:, :rows])
    np.testing.assert_array_equal(position_cache[:rows], positions_all)
    np.testing.assert_array_equal(position_cache[rows:], np.full((capacity - rows,), -1, dtype=np.int32))
    np.testing.assert_array_equal(value_cache[:, rows:], np.full_like(value_cache[:, rows:], 65535))
    assert int(live_count[0]) == rows
    print(
        "append_materializer: live_count=%d key_abs=%.3e value_match=%s bytes=%d"
        % (int(live_count[0]), float(np.max(np.abs(key_cache[:, :rows] - expected_keys[:, :rows]))), True, layers * capacity * kv_features * 6)
    )


def _dense_bf16_to_f32(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return _bf16_bits_to_f32(x).astype(np.float32) @ _bf16_bits_to_f32(weight).astype(np.float32).T


def _dense_bf16_to_bf16(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return _f32_to_bf16_bits(_dense_bf16_to_f32(x, weight))


def _key_rotary_oracle(raw: np.ndarray, weight_bf16: np.ndarray, cos: np.ndarray, sin: np.ndarray, positions: np.ndarray) -> np.ndarray:
    weight = _bf16_bits_to_f32(weight_bf16)
    out = np.empty_like(raw, dtype=np.float32)
    rotary_dim = cos.shape[1]
    half = rotary_dim // 2
    for row in range(raw.shape[0]):
        for head in range(raw.shape[1]):
            src = raw[row, head]
            inv = np.float32(1.0 / np.sqrt(np.mean(src * src) + 1.0e-6))
            normed = src * inv * weight
            dst = normed.copy()
            c = cos[int(positions[row])]
            s = sin[int(positions[row])]
            for dim in range(rotary_dim):
                pair = dim + half if dim < half else dim - half
                rotated = -normed[pair] if dim < half else normed[pair]
                dst[dim] = normed[dim] * c[dim] + rotated * s[dim]
            out[row, head] = dst
    return out


def _rotary_tables(*, max_positions: int, rotary_dim: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(max_positions, dtype=np.float32)[:, None]
    dims = np.arange(rotary_dim // 2, dtype=np.float32)[None, :]
    inv_freq = np.power(np.float32(10000.0), -2.0 * dims / np.float32(rotary_dim))
    angles = positions * inv_freq
    cos_half = np.cos(angles).astype(np.float32)
    sin_half = np.sin(angles).astype(np.float32)
    return np.concatenate([cos_half, cos_half], axis=1).astype(np.float32), np.concatenate([sin_half, sin_half], axis=1).astype(np.float32)


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
