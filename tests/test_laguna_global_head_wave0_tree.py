"""Exact global-only current-P4 head/KV wave-0-tree primitive contract."""

from __future__ import annotations

import ctypes
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.core.tensor import Tensor
from hipengine.kernels.cpu_reference import (
    LagunaRopeConfig,
    laguna_apply_rope,
    laguna_head_rmsnorm,
    laguna_rope_tables,
)
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.materialize import float_array_to_bf16_bits

_SOURCE = Path("hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.hip")
_RUNTIME = Path("hipengine/runtime/laguna_gguf_runner.py")
_LAYER = "head_rmsnorm+partial_rotary+kv_write"
_QUANT = "laguna_f32_weight"
_VARIANT = "global_wave0_tree_f32_bf16_spans"
_RETAINED_VARIANT = "global_f32_bf16_spans"
_HELPER = "laguna_head_rmsnorm_partial_rotary_wave0_tree_f32_weight_row"
_KERNEL = "laguna_global_head_rmsnorm_partial_rotary_write_kv_wave0_tree_f32_bf16_kernel"
_SYMBOL = "hipengine_laguna_global_head_rmsnorm_rope_write_kv_wave0_tree_f32_bf16_spans"
_WRAPPER = "laguna_global_head_rmsnorm_rope_write_kv_wave0_tree_f32_spans"

_WAVE0_TREE = """  partial[tid] = sum;
  __syncthreads();
  float total;
  if (tid < 32) {
    const float p0 = partial[tid] + partial[tid + 128];
    const float p64 = partial[tid + 64] + partial[tid + 192];
    const float left = p0 + p64;
    const float p32 = partial[tid + 32] + partial[tid + 160];
    const float p96 = partial[tid + 96] + partial[tid + 224];
    const float right = p32 + p96;
    total = left + right;
#pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
      total += __shfl_down(total, stride, 32);
    }
    if (tid == 0) {
      partial[0] = total;
    }
  }
  __syncthreads();
  total = partial[0];
"""


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _require_cached_build() -> bool:
    return os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _fake_global_spans(capacity: int = 512) -> KVLiveSpans:
    return KVLiveSpans.paged_dense(
        block_table=_tensor(0x1000, ((capacity + 255) // 256,), "int32"),
        live_counts=_tensor(0x2000, (1,), "int64"),
        token_positions=_tensor(0x3000, (capacity,), "int64"),
        evict_mask=_tensor(0x4000, (capacity,), "bool"),
        row_positions=_tensor(0x5000, (1,), "int64"),
        capacity=capacity,
        block_size=256,
        storage_dtype="bf16",
    )


def _fake_ring_spans(capacity: int = 512) -> KVLiveSpans:
    return KVLiveSpans.sliding_ring(
        base_offsets=_tensor(0x1000, (capacity,), "int32"),
        live_counts=_tensor(0x2000, (1,), "int64"),
        token_positions=_tensor(0x3000, (capacity,), "int64"),
        evict_mask=_tensor(0x4000, (capacity,), "bool"),
        row_positions=_tensor(0x5000, (1,), "int64"),
        capacity=capacity,
        storage_dtype="bf16",
    )


def _function_block(source: str, marker: str) -> str:
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function: {marker}")


def test_global_head_wave0_tree_source_preserves_frozen_tree_and_global_abi() -> None:
    source = _SOURCE.read_text(encoding="utf-8")
    helper = _function_block(
        source,
        f"__device__ __forceinline__ void {_HELPER}(",
    )
    candidate = _function_block(source, f"__global__ __launch_bounds__(256) void {_KERNEL}(")
    assert helper.count(_WAVE0_TREE) == 1
    assert helper.count("sum += src_value * src_value;") == 1
    assert helper.count("constexpr int64_t threads = 256;") == 1
    assert helper.count("dim += threads") == 2
    assert "blockDim.x" not in helper
    assert helper.count("const float inv_rms = rsqrtf(total / 128.0f + eps);") == 1
    assert helper.count("laguna_float_to_bf16_bits(out_value)") == 1
    assert helper.count("laguna_float_to_bf16_bits(value_row[dim])") == 1
    assert candidate.count(_HELPER) == 2
    assert candidate.count("laguna_global_physical_slot(base_offsets, position, block_size)") == 1
    for field in (
        "base_offsets",
        "live_counts",
        "token_positions",
        "evict_mask",
        "row_positions",
    ):
        assert field in candidate
    assert helper.count("__syncthreads();") == 2
    assert helper.count("__shfl_down(total, stride, 32)") == 1
    assert source.count(f'extern "C" int {_SYMBOL}(') == 1


def test_global_head_wave0_tree_wrapper_validates_before_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.kernels.hip_gfx1100.attention.laguna_kv as module

    def fail_build(**_kwargs):
        raise AssertionError("build reached")

    monkeypatch.setattr(module, "build_laguna_kv_attention", fail_build)
    launch = getattr(module, _WRAPPER)
    pointers = [0x1000 + index * 0x1000 for index in range(11)]
    for index in range(len(pointers)):
        invalid = list(pointers)
        invalid[index] = 0
        with pytest.raises(ValueError, match="non-zero"):
            launch(
                *invalid,
                _fake_global_spans(),
                1.0e-6,
                48,
                8,
                128,
                64,
                512,
            )
    with pytest.raises(ValueError, match="uniform"):
        launch(
            *pointers,
            _fake_ring_spans(),
            1.0e-6,
            48,
            8,
            128,
            64,
            512,
        )


def test_global_head_wave0_tree_package_registry_scope_fallback_and_default_off_owner() -> None:
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.hip_gfx1100 import attention
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        laguna_global_head_rmsnorm_rope_write_kv_f32_spans,
        register_laguna_kv_attention_kernels,
    )
    from hipengine.kernels.registry import KernelKey, is_registered, resolve

    candidate = getattr(attention, _WRAPPER)
    assert _WRAPPER in attention.__all__
    register_laguna_kv_attention_kernels()
    candidate_key = KernelKey("hip_gfx1100", _LAYER, _QUANT, _VARIANT)
    retained_key = KernelKey("hip_gfx1100", _LAYER, _QUANT, _RETAINED_VARIANT)
    assert resolve(
        backend=candidate_key.backend,
        layer=candidate_key.layer,
        quant=candidate_key.quant,
        variant=candidate_key.variant,
    ) is candidate
    assert resolve(
        backend=retained_key.backend,
        layer=retained_key.layer,
        quant=retained_key.quant,
        variant=retained_key.variant,
    ) is laguna_global_head_rmsnorm_rope_write_kv_f32_spans

    load_backend_kernel_package("hip_gfx1151")
    for backend in ("hip_gfx1151", "cuda_sm86", "cpu_reference"):
        assert not is_registered(KernelKey(backend, _LAYER, _QUANT, _VARIANT))

    runtime_source = _RUNTIME.read_text(encoding="utf-8")
    assert _VARIANT in runtime_source
    assert _SYMBOL not in runtime_source
    import hipengine.kernels.hip_gfx1100 as backend

    assert backend.LAGUNA_GLOBAL_HEAD_WAVE0_TREE is False
    assert "LAGUNA_GLOBAL_HEAD_WAVE0_TREE = False" in inspect.getsource(backend)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_global_head_wave0_tree_matches_retained_unfused_and_cpu_edges() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        build_laguna_kv_attention,
        laguna_global_head_rmsnorm_rope_write_kv_f32_spans,
        laguna_global_head_rmsnorm_rope_write_kv_wave0_tree_f32_spans,
        laguna_global_write_kv_f32_spans,
    )
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import build_gguf_ops
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache
    from hipengine.runtime.laguna_rope import (
        launch_laguna_head_rmsnorm_rope,
        materialize_laguna_rope_tables,
    )

    runtime = get_hip_runtime()
    kv_library = build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    rope_library = build_gguf_ops(load=True, require_cached=_require_cached_build())
    rope = LagunaRopeConfig(
        rope_type="yarn",
        rotary_dim=64,
        freq_base=500000.0,
        scaling_factor=32.0,
        original_context_length=8192,
        yarn_attn_factor=1.0,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
    )
    positions = (0, 255, 256, 4095)
    context_length = 4096
    config = SimpleNamespace(
        block_count=2,
        layer_types=("full_attention", "sliding_attention"),
        head_counts=(48, 72),
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )
    unfused_cache = allocate_laguna_kv_cache(
        config,
        context_length=context_length,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    retained_cache = allocate_laguna_kv_cache(
        config,
        context_length=context_length,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    candidate_cache = allocate_laguna_kv_cache(
        config,
        context_length=context_length,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    states = (
        unfused_cache.layer(0),
        retained_cache.layer(0),
        candidate_cache.layer(0),
    )
    tables = materialize_laguna_rope_tables(max(positions) + 1, rope, runtime=runtime)
    rng = np.random.default_rng(0x1100)
    query = rng.normal(0.0, 0.2, size=(48, 128)).astype(np.float32)
    key = rng.normal(0.0, 0.2, size=(8, 128)).astype(np.float32)
    value = rng.normal(0.0, 0.2, size=(8, 128)).astype(np.float32)
    query.reshape(-1)[:6] = np.asarray(
        [0.0, -0.0, 2.0**-60, -(2.0**-60), 2.0**16, -(2.0**16)],
        dtype=np.float32,
    )
    key.reshape(-1)[:6] = np.asarray(
        [0.0, -0.0, 2.0**-40, -(2.0**-40), 2.0**12, -(2.0**12)],
        dtype=np.float32,
    )
    value.reshape(-1)[:8] = np.asarray(
        [0.0, -0.0, 2.0**-130, -(2.0**-130), 1.00390625, -1.00390625, 2.0**8, -(2.0**8)],
        dtype=np.float32,
    )
    q_weight = rng.normal(1.0, 0.05, size=128).astype(np.float32)
    k_weight = rng.normal(1.0, 0.05, size=128).astype(np.float32)
    allocations = []
    try:
        dquery = _upload(query, runtime, allocations)
        dkey = _upload(key, runtime, allocations)
        dvalue = _upload(value, runtime, allocations)
        dqw = _upload(q_weight, runtime, allocations)
        dkw = _upload(k_weight, runtime, allocations)
        unfused_q = _alloc(query.shape, np.float32, runtime, allocations)
        unfused_k = _alloc(key.shape, np.float32, runtime, allocations)
        retained_q = _alloc(query.shape, np.float32, runtime, allocations)
        retained_k = _alloc(key.shape, np.float32, runtime, allocations)
        candidate_q = _alloc(query.shape, np.float32, runtime, allocations)
        candidate_k = _alloc(key.shape, np.float32, runtime, allocations)
        offsets = np.roll(np.arange(states[0].spans.base_offsets.numel, dtype=np.int32), 3)
        for state in states:
            _copy_array(state.spans.base_offsets.ptr, offsets, runtime)

        for position in positions:
            row = np.asarray([position], dtype=np.int64)
            for state in states:
                _copy_array(state.spans.row_positions.ptr, row, runtime)
            launch_laguna_head_rmsnorm_rope(
                dquery.ptr,
                dkey.ptr,
                dqw.ptr,
                dkw.ptr,
                states[0].spans.row_positions.ptr,
                unfused_q.ptr,
                unfused_k.ptr,
                1.0e-6,
                1,
                48,
                8,
                128,
                tables,
                backend="hip_gfx1100",
                library=rope_library,
                runtime=runtime,
            )
            laguna_global_write_kv_f32_spans(
                unfused_k.ptr,
                dvalue.ptr,
                states[0].key_cache.ptr,
                states[0].value_cache.ptr,
                states[0].spans,
                8,
                128,
                library=kv_library,
                runtime=runtime,
            )
            laguna_global_head_rmsnorm_rope_write_kv_f32_spans(
                dquery.ptr,
                dkey.ptr,
                dvalue.ptr,
                dqw.ptr,
                dkw.ptr,
                tables.cos.tensor.ptr,
                tables.sin.tensor.ptr,
                retained_q.ptr,
                retained_k.ptr,
                states[1].key_cache.ptr,
                states[1].value_cache.ptr,
                states[1].spans,
                1.0e-6,
                48,
                8,
                128,
                64,
                max(positions) + 1,
                library=kv_library,
                runtime=runtime,
            )
            laguna_global_head_rmsnorm_rope_write_kv_wave0_tree_f32_spans(
                dquery.ptr,
                dkey.ptr,
                dvalue.ptr,
                dqw.ptr,
                dkw.ptr,
                tables.cos.tensor.ptr,
                tables.sin.tensor.ptr,
                candidate_q.ptr,
                candidate_k.ptr,
                states[2].key_cache.ptr,
                states[2].value_cache.ptr,
                states[2].spans,
                1.0e-6,
                48,
                8,
                128,
                64,
                max(positions) + 1,
                library=kv_library,
                runtime=runtime,
            )
            runtime.device_synchronize()

            outputs = (
                _download(unfused_q, query.shape, np.float32, runtime),
                _download(retained_q, query.shape, np.float32, runtime),
                _download(candidate_q, query.shape, np.float32, runtime),
                _download(unfused_k, key.shape, np.float32, runtime),
                _download(retained_k, key.shape, np.float32, runtime),
                _download(candidate_k, key.shape, np.float32, runtime),
            )
            for control, actual in (
                (outputs[0], outputs[1]),
                (outputs[0], outputs[2]),
                (outputs[3], outputs[4]),
                (outputs[3], outputs[5]),
            ):
                np.testing.assert_array_equal(actual.view(np.uint32), control.view(np.uint32))

            cos, sin = laguna_rope_tables(np.asarray([position], dtype=np.int64), rope)
            expected_q = laguna_apply_rope(
                laguna_head_rmsnorm(query, q_weight),
                cos[0],
                sin[0],
                rotary_dim=64,
            )
            expected_k = laguna_apply_rope(
                laguna_head_rmsnorm(key, k_weight),
                cos[0],
                sin[0],
                rotary_dim=64,
            )
            for actual, expected in ((outputs[2], expected_q), (outputs[5], expected_k)):
                kl, top1 = _quality(actual, expected)
                assert kl <= 0.05
                assert top1 >= 0.90

            logical_block, block_offset = divmod(position, 256)
            physical_slot = int(offsets[logical_block]) * 256 + block_offset
            row_nbytes = 8 * 128 * DType.BF16.itemsize
            expected_value_bits = float_array_to_bf16_bits(value)
            for state in states:
                key_bits = _download_offset(
                    state.key_cache.ptr + physical_slot * row_nbytes,
                    (8, 128),
                    np.uint16,
                    runtime,
                )
                value_bits = _download_offset(
                    state.value_cache.ptr + physical_slot * row_nbytes,
                    (8, 128),
                    np.uint16,
                    runtime,
                )
                np.testing.assert_array_equal(key_bits, float_array_to_bf16_bits(outputs[5]))
                np.testing.assert_array_equal(value_bits, expected_value_bits)
            for name, dtype, count in (
                ("live_counts", np.int64, 1),
                ("token_positions", np.int64, context_length),
                ("evict_mask", np.bool_, context_length),
            ):
                rows = [
                    _download_offset(getattr(state.spans, name).ptr, (count,), dtype, runtime)
                    for state in states
                ]
                np.testing.assert_array_equal(rows[1], rows[0])
                np.testing.assert_array_equal(rows[2], rows[0])
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        tables.free(runtime=runtime)
        candidate_cache.free()
        retained_cache.free()
        unfused_cache.free()


def _quality(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    actual64 = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected64 = np.asarray(expected, dtype=np.float64).reshape(-1)
    actual_prob = np.exp(actual64 - float(np.max(actual64)))
    expected_prob = np.exp(expected64 - float(np.max(expected64)))
    actual_prob /= np.sum(actual_prob)
    expected_prob /= np.sum(expected_prob)
    kl = float(
        np.sum(
            expected_prob
            * np.log(np.maximum(expected_prob, 1e-300) / np.maximum(actual_prob, 1e-300))
        )
    )
    top1 = float(int(np.argmax(actual64)) == int(np.argmax(expected64)))
    return kl, top1


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _alloc(shape, dtype, runtime, allocations):
    device = malloc(int(np.prod(shape)) * np.dtype(dtype).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _copy_array(device_ptr: int, array: np.ndarray, runtime) -> None:
    host = np.ascontiguousarray(array)
    runtime.memcpy(
        int(device_ptr),
        host_array_ptr(host),
        host.nbytes,
        HipMemcpyKind.HOST_TO_DEVICE,
    )


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host


def _download_offset(device_ptr: int, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    runtime.memcpy(
        host_array_ptr(host),
        int(device_ptr),
        host.nbytes,
        HipMemcpyKind.DEVICE_TO_HOST,
    )
    return host
