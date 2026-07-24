from __future__ import annotations

import ctypes
import os
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
from hipengine.kernels.cpu_reference import LagunaRopeConfig
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION


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
    blocks = (capacity + 255) // 256
    return KVLiveSpans.paged_dense(
        block_table=_tensor(0x1000, (blocks,), "int32"),
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


def test_laguna_d14_head_kv_registry_plan_and_fail_closed_validation() -> None:
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        laguna_global_head_rmsnorm_rope_write_kv_f32_spans,
        laguna_swa_head_rmsnorm_rope_write_kv_f32_spans,
        plan_laguna_kv_attention_build,
        register_laguna_kv_attention_kernels,
    )
    from hipengine.kernels.registry import resolve

    artifact = plan_laguna_kv_attention_build(
        cache_root="/tmp/hipengine-laguna-d14-plan-test",
        compiler_version="hipcc laguna d14 plan test",
    )
    assert artifact.family == "laguna_kv_attention"
    assert any(str(path).endswith("laguna_kv_attention.hip") for path in artifact.sources)

    register_laguna_kv_attention_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="head_rmsnorm+partial_rotary+kv_write",
            quant="laguna_f32_weight",
            variant="global_f32_bf16_spans",
        )
        is laguna_global_head_rmsnorm_rope_write_kv_f32_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="head_rmsnorm+partial_rotary+kv_write",
            quant="laguna_f32_weight",
            variant="swa_f32_bf16_spans",
        )
        is laguna_swa_head_rmsnorm_rope_write_kv_f32_spans
    )
    load_backend_kernel_package("hip_gfx1151")
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="head_rmsnorm+partial_rotary+kv_write",
            quant="laguna_f32_weight",
            variant="global_f32_bf16_spans",
            missing="none",
        )
        is None
    )

    common = (0,) * 11
    with pytest.raises(ValueError, match="uniform"):
        laguna_global_head_rmsnorm_rope_write_kv_f32_spans(
            *common,
            _fake_ring_spans(),
            1.0e-6,
            48,
            8,
            128,
            64,
            512,
        )
    with pytest.raises(ValueError, match="sliding_ring"):
        laguna_swa_head_rmsnorm_rope_write_kv_f32_spans(
            *common,
            _fake_global_spans(),
            1.0e-6,
            72,
            8,
            128,
            128,
            512,
        )
    with pytest.raises(ValueError, match="divisible"):
        laguna_swa_head_rmsnorm_rope_write_kv_f32_spans(
            *common,
            _fake_ring_spans(),
            1.0e-6,
            70,
            8,
            128,
            128,
            512,
        )


@pytest.mark.parametrize(
    ("attention_type", "q_heads", "rope", "positions"),
    [
        (
            FULL_ATTENTION,
            48,
            LagunaRopeConfig(
                rope_type="yarn",
                rotary_dim=64,
                freq_base=500000.0,
                scaling_factor=32.0,
                original_context_length=8192,
                yarn_attn_factor=1.0,
                yarn_beta_fast=32.0,
                yarn_beta_slow=1.0,
            ),
            (0, 255, 256, 4095),
        ),
        (
            SLIDING_ATTENTION,
            72,
            LagunaRopeConfig(rope_type="default", rotary_dim=128, freq_base=10000.0),
            (0, 1, 511, 512, 513, 1024),
        ),
    ],
)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_d14_fused_head_kv_is_bit_exact_to_registered_chain(
    attention_type: str,
    q_heads: int,
    rope: LagunaRopeConfig,
    positions: tuple[int, ...],
) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        build_laguna_kv_attention,
        laguna_global_head_rmsnorm_rope_write_kv_f32_spans,
        laguna_global_write_kv_f32_spans,
        laguna_swa_head_rmsnorm_rope_write_kv_f32_spans,
        laguna_swa_write_kv_f32_spans,
    )
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import build_gguf_ops
    from hipengine.loading.materialize import float_array_to_bf16_bits
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
    context_length = 4096
    config = SimpleNamespace(
        block_count=2,
        layer_types=(FULL_ATTENTION, SLIDING_ATTENTION),
        head_counts=(48, 72),
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )
    control = allocate_laguna_kv_cache(
        config,
        context_length=context_length,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    candidate = allocate_laguna_kv_cache(
        config,
        context_length=context_length,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    layer_id = 0 if attention_type == FULL_ATTENTION else 1
    control_state = control.layer(layer_id)
    candidate_state = candidate.layer(layer_id)
    max_positions = max(positions) + 1
    tables = materialize_laguna_rope_tables(max_positions, rope, runtime=runtime)
    rng = np.random.default_rng(1414 + q_heads)
    query = rng.normal(0.0, 0.2, size=(q_heads, 128)).astype(np.float32)
    key = rng.normal(0.0, 0.2, size=(8, 128)).astype(np.float32)
    value = rng.normal(0.0, 0.2, size=(8, 128)).astype(np.float32)
    query.reshape(-1)[:4] = np.asarray([0.0, -0.0, 2.0**-60, 2.0**20], dtype=np.float32)
    key.reshape(-1)[:4] = np.asarray([0.0, -0.0, 2.0**-40, -(2.0**16)], dtype=np.float32)
    value.reshape(-1)[:6] = np.asarray(
        [0.0, -0.0, np.inf, -np.inf, np.nan, 1.00390625],
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
        control_q = _alloc(query.shape, np.float32, runtime, allocations)
        control_k = _alloc(key.shape, np.float32, runtime, allocations)
        candidate_q = _alloc(query.shape, np.float32, runtime, allocations)
        candidate_k = _alloc(key.shape, np.float32, runtime, allocations)

        if attention_type == FULL_ATTENTION:
            global_offsets = np.roll(
                np.arange(control_state.spans.base_offsets.numel, dtype=np.int32),
                3,
            )
            _copy_array(control_state.spans.base_offsets.ptr, global_offsets, runtime)
            _copy_array(candidate_state.spans.base_offsets.ptr, global_offsets, runtime)
            control_write = laguna_global_write_kv_f32_spans
            fused = laguna_global_head_rmsnorm_rope_write_kv_f32_spans
        else:
            ring_offsets = np.arange(511, -1, -1, dtype=np.int32)
            _copy_array(control_state.spans.base_offsets.ptr, ring_offsets, runtime)
            _copy_array(candidate_state.spans.base_offsets.ptr, ring_offsets, runtime)
            control_write = laguna_swa_write_kv_f32_spans
            fused = laguna_swa_head_rmsnorm_rope_write_kv_f32_spans

        for position in positions:
            position_row = np.asarray([position], dtype=np.int64)
            _copy_array(control_state.spans.row_positions.ptr, position_row, runtime)
            _copy_array(candidate_state.spans.row_positions.ptr, position_row, runtime)
            launch_laguna_head_rmsnorm_rope(
                dquery.ptr,
                dkey.ptr,
                dqw.ptr,
                dkw.ptr,
                control_state.spans.row_positions.ptr,
                control_q.ptr,
                control_k.ptr,
                1.0e-6,
                1,
                q_heads,
                8,
                128,
                tables,
                backend="hip_gfx1100",
                library=rope_library,
                runtime=runtime,
            )
            control_write(
                control_k.ptr,
                dvalue.ptr,
                control_state.key_cache.ptr,
                control_state.value_cache.ptr,
                control_state.spans,
                8,
                128,
                library=kv_library,
                runtime=runtime,
            )
            fused(
                dquery.ptr,
                dkey.ptr,
                dvalue.ptr,
                dqw.ptr,
                dkw.ptr,
                tables.cos.tensor.ptr,
                tables.sin.tensor.ptr,
                candidate_q.ptr,
                candidate_k.ptr,
                candidate_state.key_cache.ptr,
                candidate_state.value_cache.ptr,
                candidate_state.spans,
                1.0e-6,
                q_heads,
                8,
                128,
                rope.rotary_dim,
                max_positions,
                library=kv_library,
                runtime=runtime,
            )
            runtime.device_synchronize()

            control_q_host = _download(control_q, query.shape, np.float32, runtime)
            control_k_host = _download(control_k, key.shape, np.float32, runtime)
            candidate_q_host = _download(candidate_q, query.shape, np.float32, runtime)
            candidate_k_host = _download(candidate_k, key.shape, np.float32, runtime)
            np.testing.assert_array_equal(candidate_q_host.view(np.uint32), control_q_host.view(np.uint32))
            np.testing.assert_array_equal(candidate_k_host.view(np.uint32), control_k_host.view(np.uint32))

            if attention_type == FULL_ATTENTION:
                logical_block, block_offset = divmod(position, 256)
                physical_slot = int(global_offsets[logical_block]) * 256 + block_offset
                metadata_slot = position
            else:
                metadata_slot = position % 512
                physical_slot = int(ring_offsets[metadata_slot])
            row_nbytes = 8 * 128 * DType.BF16.itemsize
            control_key_bits = _download_offset(
                control_state.key_cache.ptr + physical_slot * row_nbytes,
                (8, 128),
                np.uint16,
                runtime,
            )
            candidate_key_bits = _download_offset(
                candidate_state.key_cache.ptr + physical_slot * row_nbytes,
                (8, 128),
                np.uint16,
                runtime,
            )
            control_value_bits = _download_offset(
                control_state.value_cache.ptr + physical_slot * row_nbytes,
                (8, 128),
                np.uint16,
                runtime,
            )
            candidate_value_bits = _download_offset(
                candidate_state.value_cache.ptr + physical_slot * row_nbytes,
                (8, 128),
                np.uint16,
                runtime,
            )
            np.testing.assert_array_equal(candidate_key_bits, control_key_bits)
            np.testing.assert_array_equal(candidate_value_bits, control_value_bits)
            np.testing.assert_array_equal(control_key_bits, float_array_to_bf16_bits(control_k_host))
            np.testing.assert_array_equal(control_value_bits, float_array_to_bf16_bits(value))

            for name, dtype, count in (
                ("live_counts", np.int64, 1),
                ("token_positions", np.int64, control_state.capacity),
                ("evict_mask", np.bool_, control_state.capacity),
            ):
                control_tensor = getattr(control_state.spans, name)
                candidate_tensor = getattr(candidate_state.spans, name)
                control_metadata = _download_offset(
                    control_tensor.ptr,
                    (count,),
                    dtype,
                    runtime,
                )
                candidate_metadata = _download_offset(
                    candidate_tensor.ptr,
                    (count,),
                    dtype,
                    runtime,
                )
                np.testing.assert_array_equal(candidate_metadata, control_metadata)
            token_positions = _download_offset(
                candidate_state.spans.token_positions.ptr,
                (candidate_state.capacity,),
                np.int64,
                runtime,
            )
            evict_mask = _download_offset(
                candidate_state.spans.evict_mask.ptr,
                (candidate_state.capacity,),
                np.bool_,
                runtime,
            )
            assert token_positions[metadata_slot] == position
            assert not evict_mask[metadata_slot]
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        tables.free(runtime=runtime)
        candidate.free()
        control.free()


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
