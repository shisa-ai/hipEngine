from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import (
    LagunaRopeConfig,
    laguna_apply_rope,
    laguna_head_rmsnorm,
    laguna_rope_tables,
    laguna_softplus_head_gate,
)
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32


def _require_cached_build() -> bool:
    return os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_laguna_rope_and_softplus_registry_resolve_on_gfx1151() -> None:
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.hip_gfx1100.fused.laguna_attention import (
        laguna_softplus_head_gate_f32_out,
        register_laguna_attention_kernels,
    )
    from hipengine.kernels.registry import resolve
    from hipengine.runtime.laguna_rope import register_laguna_rope_kernels

    register_laguna_attention_kernels()
    register_laguna_rope_kernels()
    load_backend_kernel_package("hip_gfx1151")
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="attention_gate",
            quant="f32",
            variant="softplus_broadcast_f32_out",
        )
        is laguna_softplus_head_gate_f32_out
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="head_rmsnorm+partial_rotary",
            quant="laguna_f32_weight",
            variant="positions_f32",
        )
        is not None
    )


@pytest.mark.parametrize(
    ("q_heads", "rope"),
    [
        (
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
        ),
        (72, LagunaRopeConfig(rope_type="default", rotary_dim=128, freq_base=10000.0)),
    ],
)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_head_rmsnorm_rope_matches_cpu_at_production_heads(q_heads, rope) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import build_gguf_ops
    from hipengine.runtime.laguna_rope import (
        launch_laguna_head_rmsnorm_rope,
        materialize_laguna_rope_tables,
    )

    runtime = get_hip_runtime()
    library = build_gguf_ops(load=True, require_cached=_require_cached_build())
    rng = np.random.default_rng(282 + q_heads)
    tokens, kv_heads, head_dim = 2, 8, 128
    positions = np.asarray([511, 8193 if rope.rope_type == "yarn" else 513], dtype=np.int64)
    query = rng.normal(0.0, 0.2, size=(tokens, q_heads, head_dim)).astype(np.float32)
    key = rng.normal(0.0, 0.2, size=(tokens, kv_heads, head_dim)).astype(np.float32)
    q_weight = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
    k_weight = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
    cos, sin = laguna_rope_tables(positions, rope)
    expected_q = laguna_apply_rope(
        laguna_head_rmsnorm(query, q_weight, eps=1e-6),
        cos[:, None, :],
        sin[:, None, :],
        rotary_dim=rope.rotary_dim,
    )
    expected_k = laguna_apply_rope(
        laguna_head_rmsnorm(key, k_weight, eps=1e-6),
        cos[:, None, :],
        sin[:, None, :],
        rotary_dim=rope.rotary_dim,
    )

    allocations = []
    tables = materialize_laguna_rope_tables(
        int(np.max(positions)) + 1,
        rope,
        runtime=runtime,
    )
    try:
        dq = _upload(query, runtime, allocations)
        dk = _upload(key, runtime, allocations)
        dqw = _upload(q_weight, runtime, allocations)
        dkw = _upload(k_weight, runtime, allocations)
        dpos = _upload(positions, runtime, allocations)
        dqo = _alloc(query.shape, np.float32, runtime, allocations)
        dko = _alloc(key.shape, np.float32, runtime, allocations)
        launch_laguna_head_rmsnorm_rope(
            dq.ptr,
            dk.ptr,
            dqw.ptr,
            dkw.ptr,
            dpos.ptr,
            dqo.ptr,
            dko.ptr,
            1e-6,
            tokens,
            q_heads,
            kv_heads,
            head_dim,
            tables,
            backend="hip_gfx1151",
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual_q = _download(dqo, query.shape, np.float32, runtime)
        actual_k = _download(dko, key.shape, np.float32, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        tables.free(runtime=runtime)

    np.testing.assert_allclose(actual_q, expected_q, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_k, expected_k, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_head_rmsnorm_rope_packed_query_tiles_match_generic() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import build_gguf_ops
    from hipengine.runtime.laguna_rope import (
        launch_laguna_head_rmsnorm_rope,
        materialize_laguna_rope_tables,
    )

    runtime = get_hip_runtime()
    library = build_gguf_ops(load=True, require_cached=_require_cached_build())
    rng = np.random.default_rng(290)
    tokens, q_heads, kv_heads, head_dim = 256, 72, 8, 128
    rope = LagunaRopeConfig(
        rope_type="default",
        rotary_dim=128,
        freq_base=10000.0,
    )
    positions = np.arange(tokens, dtype=np.int64)
    query = rng.normal(
        0.0,
        0.2,
        size=(tokens, q_heads, head_dim),
    ).astype(np.float32)
    key = rng.normal(
        0.0,
        0.2,
        size=(tokens, kv_heads, head_dim),
    ).astype(np.float32)
    q_weight = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
    k_weight = rng.normal(1.0, 0.05, size=head_dim).astype(np.float32)
    allocations = []
    tables = materialize_laguna_rope_tables(
        tokens,
        rope,
        runtime=runtime,
    )
    try:
        dq = _upload(query, runtime, allocations)
        dk = _upload(key, runtime, allocations)
        dqw = _upload(q_weight, runtime, allocations)
        dkw = _upload(k_weight, runtime, allocations)
        dpos = _upload(positions, runtime, allocations)
        dqo_generic = _alloc(
            query.shape,
            np.float32,
            runtime,
            allocations,
        )
        dko_generic = _alloc(key.shape, np.float32, runtime, allocations)
        dqo_packed = _alloc(query.shape, np.float32, runtime, allocations)
        dko_packed = _alloc(key.shape, np.float32, runtime, allocations)
        common = (
            dq.ptr,
            dk.ptr,
            dqw.ptr,
            dkw.ptr,
            dpos.ptr,
        )
        launch_laguna_head_rmsnorm_rope(
            *common,
            dqo_generic.ptr,
            dko_generic.ptr,
            1e-6,
            tokens,
            q_heads,
            kv_heads,
            head_dim,
            tables,
            backend="hip_gfx1151",
            library=library,
            runtime=runtime,
        )
        launch_laguna_head_rmsnorm_rope(
            *common,
            dqo_packed.ptr,
            dko_packed.ptr,
            1e-6,
            tokens,
            q_heads,
            kv_heads,
            head_dim,
            tables,
            packed_query_begin=128,
            packed_query_end=256,
            backend="hip_gfx1151",
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        generic_q = _download(
            dqo_generic,
            query.shape,
            np.float32,
            runtime,
        )
        packed_q = _download(
            dqo_packed,
            query.shape,
            np.float32,
            runtime,
        )
        generic_k = _download(
            dko_generic,
            key.shape,
            np.float32,
            runtime,
        )
        packed_k = _download(
            dko_packed,
            key.shape,
            np.float32,
            runtime,
        )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        tables.free(runtime=runtime)

    unpacked_q = packed_q.copy()
    unpacked_q[128:256] = (
        packed_q[128:256]
        .reshape(q_heads, 128, head_dim)
        .transpose(1, 0, 2)
    )
    np.testing.assert_array_equal(unpacked_q, generic_q)
    np.testing.assert_array_equal(packed_k, generic_k)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_softplus_head_gate_broadcast_matches_cpu() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.fused.laguna_attention import (
        build_laguna_attention,
        laguna_softplus_head_gate_f32_bf16_out,
        laguna_softplus_head_gate_f32_bf16_packed_tiles_out,
        laguna_softplus_head_gate_f32_out,
    )

    runtime = get_hip_runtime()
    library = build_laguna_attention(load=True, require_cached=_require_cached_build())
    rng = np.random.default_rng(284)
    rows, heads, head_dim = 2, 72, 128
    context = rng.normal(0.0, 0.2, size=(rows, heads, head_dim)).astype(np.float32)
    gate = rng.normal(0.0, 4.0, size=(rows, heads)).astype(np.float32)
    gate[0, :4] = np.asarray([-100.0, -20.0, 20.0, 100.0], dtype=np.float32)
    expected = laguna_softplus_head_gate(context, gate)

    allocations = []
    try:
        dcontext = _upload(context, runtime, allocations)
        dgate = _upload(gate, runtime, allocations)
        dout = _alloc(context.shape, np.float32, runtime, allocations)
        dout_bf16 = _alloc(context.shape, np.uint16, runtime, allocations)
        laguna_softplus_head_gate_f32_out(
            dcontext.ptr,
            dgate.ptr,
            dout.ptr,
            rows,
            heads,
            head_dim,
            library=library,
            runtime=runtime,
        )
        laguna_softplus_head_gate_f32_bf16_out(
            dcontext.ptr,
            dgate.ptr,
            dout_bf16.ptr,
            rows,
            heads,
            head_dim,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = _download(dout, context.shape, np.float32, runtime)
        actual_bf16 = bf16_to_float32(_download(dout_bf16, context.shape, np.uint16, runtime))
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(
        actual_bf16,
        bf16_to_float32(float_array_to_bf16_bits(actual)),
    )

    rows = 384
    context = rng.normal(
        0.0,
        0.2,
        size=(rows, heads, head_dim),
    ).astype(np.float32)
    gate = rng.normal(0.0, 4.0, size=(rows, heads)).astype(np.float32)
    packed = context.copy()
    for begin in (128, 256):
        packed[begin : begin + 128] = (
            context[begin : begin + 128]
            .transpose(1, 0, 2)
            .reshape(128, heads, head_dim)
        )
    allocations = []
    try:
        dcontext = _upload(packed, runtime, allocations)
        dcontext_generic = _upload(context, runtime, allocations)
        dgate = _upload(gate, runtime, allocations)
        dout_bf16 = _alloc(context.shape, np.uint16, runtime, allocations)
        dout_generic_bf16 = _alloc(
            context.shape,
            np.uint16,
            runtime,
            allocations,
        )
        laguna_softplus_head_gate_f32_bf16_out(
            dcontext_generic.ptr,
            dgate.ptr,
            dout_generic_bf16.ptr,
            rows,
            heads,
            head_dim,
            library=library,
            runtime=runtime,
        )
        laguna_softplus_head_gate_f32_bf16_packed_tiles_out(
            dcontext.ptr,
            dgate.ptr,
            dout_bf16.ptr,
            rows,
            heads,
            head_dim,
            128,
            384,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual_bf16 = bf16_to_float32(
            _download(dout_bf16, context.shape, np.uint16, runtime)
        )
        generic_bf16 = bf16_to_float32(
            _download(
                dout_generic_bf16,
                context.shape,
                np.uint16,
                runtime,
            )
        )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
    np.testing.assert_array_equal(actual_bf16, generic_bf16)


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


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
