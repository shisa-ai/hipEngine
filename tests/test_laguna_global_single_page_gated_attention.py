"""Exact gated single-page Laguna global-attention primitive contract."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans
from tests.test_laguna_global_single_page_attention import (
    _function_block,
    _ordered_attention_reference,
    _quality,
)

_SOURCE = Path("hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.hip")
_RUNTIME = Path("hipengine/runtime/laguna_kv.py")
_LAYER = "laguna_attention_decode+attention_gate"
_VARIANT = "global_single_page_softplus_bf16_spans"
_KERNEL = "laguna_global_attention_decode_single_page_softplus_gate_bf16_kernel"
_SYMBOL = "hipengine_laguna_global_attention_decode_single_page_softplus_gate_bf16_spans"


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
    pages = (capacity + 255) // 256
    return KVLiveSpans.paged_dense(
        block_table=_tensor(0x1000, (pages,), "int32"),
        live_counts=_tensor(0x2000, (1,), "int64"),
        token_positions=_tensor(0x3000, (capacity,), "int64"),
        evict_mask=_tensor(0x4000, (capacity,), "bool"),
        row_positions=_tensor(0x5000, (1,), "int64"),
        capacity=capacity,
        block_size=256,
        storage_dtype="bf16",
    )


def test_gated_single_page_source_is_exact_one_page_plus_gate_epilogue() -> None:
    source = _SOURCE.read_text(encoding="utf-8")
    one_page = _function_block(
        source,
        "__global__ void laguna_global_attention_decode_single_page_bf16_kernel(",
    )
    candidate = _function_block(source, f"__global__ void {_KERNEL}(")
    expected = one_page.replace(
        "laguna_global_attention_decode_single_page_bf16_kernel",
        _KERNEL,
        1,
    ).replace(
        "    float* out,\n    const int32_t* base_offsets,",
        "    float* out,\n    const float* gate,\n    uint16_t* gated_out,\n"
        "    const int32_t* base_offsets,",
        1,
    ).replace(
        "    out[q_offset + dim] = acc;",
        "    const float context_value = acc;\n"
        "    out[q_offset + dim] = context_value;\n"
        "    gated_out[q_offset + dim] = laguna_float_to_bf16_bits(\n"
        "        context_value * laguna_softplus_f32(gate[q_head]));",
        1,
    )
    assert candidate == expected
    assert candidate.count(
        "static_cast<int64_t>(base_offsets[0]) * block_size + token"
    ) == 2
    assert "context_len > block_size" in candidate
    assert candidate.count("__syncthreads()") == one_page.count("__syncthreads()") == 5
    assert candidate.count("laguna_softplus_f32(gate[q_head])") == 1
    assert source.count(f'extern "C" int {_SYMBOL}(') == 1


def test_gated_single_page_wrapper_validates_before_build(monkeypatch) -> None:
    import hipengine.kernels.hip_gfx1100.attention.laguna_kv as module

    def fail_build(**_kwargs):
        raise AssertionError("build reached")

    monkeypatch.setattr(module, "build_laguna_kv_attention", fail_build)
    launch = module.laguna_global_attention_decode_single_page_softplus_gate_bf16_spans
    spans = _fake_global_spans()
    valid = (0x6000, 0x7000, 0x8000, 0x9000, 0xA000, 0xB000)

    for index in range(6):
        pointers = list(valid)
        pointers[index] = 0
        with pytest.raises(ValueError, match="non-zero"):
            launch(*pointers, spans, 512, 48, 8, 128, 128**-0.5)
    with pytest.raises(ValueError, match="max_context_len"):
        launch(*valid, spans, 256, 48, 8, 128, 128**-0.5)
    with pytest.raises(ValueError, match="num_q_heads"):
        launch(*valid, spans, 512, 0, 8, 128, 128**-0.5)
    with pytest.raises(ValueError, match="num_q_heads"):
        launch(*valid, spans, 512, 72, 8, 128, 128**-0.5)
    with pytest.raises(ValueError, match="num_kv_heads"):
        launch(*valid, spans, 512, 48, 4, 128, 128**-0.5)
    with pytest.raises(ValueError, match="head_dim"):
        launch(*valid, spans, 512, 48, 8, 64, 64**-0.5)


def test_gated_single_page_package_registry_backend_scope_and_no_runtime_owner() -> None:
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.hip_gfx1100 import attention
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        laguna_global_attention_decode_single_page_softplus_gate_bf16_spans,
        register_laguna_kv_attention_kernels,
    )
    from hipengine.kernels.registry import KernelKey, is_registered, resolve

    assert (
        attention.laguna_global_attention_decode_single_page_softplus_gate_bf16_spans
        is laguna_global_attention_decode_single_page_softplus_gate_bf16_spans
    )
    assert (
        "laguna_global_attention_decode_single_page_softplus_gate_bf16_spans"
        in attention.__all__
    )
    register_laguna_kv_attention_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer=_LAYER,
            quant="bf16",
            variant=_VARIANT,
        )
        is laguna_global_attention_decode_single_page_softplus_gate_bf16_spans
    )

    load_backend_kernel_package("hip_gfx1151")
    for backend in ("hip_gfx1151", "cuda_sm86", "cpu_reference"):
        assert not is_registered(KernelKey(backend, _LAYER, "bf16", _VARIANT))

    runtime_source = _RUNTIME.read_text(encoding="utf-8")
    assert _VARIANT not in runtime_source
    assert _SYMBOL not in runtime_source


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_gated_single_page_matches_unfused_chain_and_cpu() -> None:
    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        build_laguna_kv_attention,
        laguna_global_attention_decode_single_page_bf16_spans,
        laguna_global_attention_decode_single_page_softplus_gate_bf16_spans,
    )
    from hipengine.kernels.hip_gfx1100.fused.laguna_attention import (
        build_laguna_attention,
        laguna_softplus_head_gate_f32_bf16_out,
    )
    from hipengine.loading.materialize import float_array_to_bf16_bits
    from hipengine.quant.gguf import bf16_to_float32

    runtime = get_hip_runtime()
    attention_library = build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    gate_library = build_laguna_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    capacity = 512
    block_size = 256
    num_q_heads = 48
    num_kv_heads = 8
    head_dim = 128
    storage_slots = 3 * block_size
    rng = np.random.default_rng(20270727)

    query = rng.normal(0.0, 0.12, size=(num_q_heads, head_dim)).astype(np.float32)
    edge_bits = np.array(
        [0x0000, 0x8000, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x3F00, 0xBF00],
        dtype=np.uint16,
    )
    query[:, : edge_bits.size] = bf16_to_float32(edge_bits)[None, :]
    gate = rng.normal(0.0, 2.0, size=(num_q_heads,)).astype(np.float32)
    gate[:12] = np.array(
        [
            0.0,
            -0.0,
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            np.nextafter(np.float32(0.0), np.float32(-1.0)),
            1.0,
            -1.0,
            20.0,
            21.0,
            -20.0,
            np.inf,
            -np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    logical_keys = float_array_to_bf16_bits(
        rng.normal(0.0, 0.12, size=(capacity, num_kv_heads, head_dim)).astype(np.float32)
    )
    logical_values = float_array_to_bf16_bits(
        rng.normal(0.0, 0.12, size=(capacity, num_kv_heads, head_dim)).astype(np.float32)
    )
    logical_keys[0, 0, : edge_bits.size] = edge_bits
    logical_values[0, 0, : edge_bits.size] = edge_bits[::-1]

    base_offsets = np.array([2, 0], dtype=np.int32)
    physical_keys = np.full(
        (storage_slots, num_kv_heads, head_dim),
        np.uint16(0x3F40),
        dtype=np.uint16,
    )
    physical_values = np.full_like(physical_keys, np.uint16(0xBF40))
    for page, physical_page in enumerate(base_offsets):
        logical_slice = slice(page * block_size, (page + 1) * block_size)
        physical_slice = slice(
            int(physical_page) * block_size,
            (int(physical_page) + 1) * block_size,
        )
        physical_keys[physical_slice] = logical_keys[logical_slice]
        physical_values[physical_slice] = logical_values[logical_slice]

    live_counts = np.array([0], dtype=np.int64)
    token_positions = np.arange(capacity, dtype=np.int64)
    evict_mask = np.zeros(capacity, dtype=np.bool_)
    row_positions = np.array([0], dtype=np.int64)
    allocations = []
    try:
        query_device = malloc(query.nbytes, runtime=runtime)
        gate_device = malloc(gate.nbytes, runtime=runtime)
        key_device = malloc(physical_keys.nbytes, runtime=runtime)
        value_device = malloc(physical_values.nbytes, runtime=runtime)
        baseline_context_device = malloc(query.nbytes, runtime=runtime)
        candidate_context_device = malloc(query.nbytes, runtime=runtime)
        baseline_gated_device = malloc(query.size * 2, runtime=runtime)
        candidate_gated_device = malloc(query.size * np.dtype(np.uint16).itemsize, runtime=runtime)
        offsets_device = malloc(base_offsets.nbytes, runtime=runtime)
        live_device = malloc(live_counts.nbytes, runtime=runtime)
        positions_device = malloc(token_positions.nbytes, runtime=runtime)
        evict_device = malloc(evict_mask.nbytes, runtime=runtime)
        row_device = malloc(row_positions.nbytes, runtime=runtime)
        allocations.extend(
            (
                query_device,
                gate_device,
                key_device,
                value_device,
                baseline_context_device,
                candidate_context_device,
                baseline_gated_device,
                candidate_gated_device,
                offsets_device,
                live_device,
                positions_device,
                evict_device,
                row_device,
            )
        )
        for buffer, array in (
            (query_device, query),
            (gate_device, gate),
            (key_device, physical_keys),
            (value_device, physical_values),
            (offsets_device, base_offsets),
            (positions_device, token_positions),
            (evict_device, evict_mask),
            (row_device, row_positions),
        ):
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes, runtime=runtime)

        spans = KVLiveSpans.paged_dense(
            block_table=_tensor(offsets_device.ptr, base_offsets.shape, "int32"),
            live_counts=_tensor(live_device.ptr, live_counts.shape, "int64"),
            token_positions=_tensor(
                positions_device.ptr,
                token_positions.shape,
                "int64",
            ),
            evict_mask=_tensor(evict_device.ptr, evict_mask.shape, "bool"),
            row_positions=_tensor(row_device.ptr, row_positions.shape, "int64"),
            capacity=capacity,
            block_size=block_size,
            storage_dtype="bf16",
        )

        for live_count in (1, 70, 126, 256):
            live_counts[0] = live_count
            row_positions[0] = max(0, live_count - 5)
            evict_mask.fill(False)
            if live_count > 2:
                evict_mask[1] = True
            if live_count > 17:
                evict_mask[17] = True
            runtime.memcpy(
                live_device.ptr,
                host_array_ptr(live_counts),
                live_counts.nbytes,
                HipMemcpyKind.HOST_TO_DEVICE,
            )
            runtime.memcpy(
                evict_device.ptr,
                host_array_ptr(evict_mask),
                evict_mask.nbytes,
                HipMemcpyKind.HOST_TO_DEVICE,
            )
            runtime.memcpy(
                row_device.ptr,
                host_array_ptr(row_positions),
                row_positions.nbytes,
                HipMemcpyKind.HOST_TO_DEVICE,
            )

            laguna_global_attention_decode_single_page_bf16_spans(
                query_device.ptr,
                key_device.ptr,
                value_device.ptr,
                baseline_context_device.ptr,
                spans,
                capacity,
                num_q_heads,
                num_kv_heads,
                head_dim,
                head_dim**-0.5,
                library=attention_library,
                runtime=runtime,
            )
            laguna_softplus_head_gate_f32_bf16_out(
                baseline_context_device.ptr,
                gate_device.ptr,
                baseline_gated_device.ptr,
                1,
                num_q_heads,
                head_dim,
                library=gate_library,
                runtime=runtime,
            )
            laguna_global_attention_decode_single_page_softplus_gate_bf16_spans(
                query_device.ptr,
                key_device.ptr,
                value_device.ptr,
                candidate_context_device.ptr,
                gate_device.ptr,
                candidate_gated_device.ptr,
                spans,
                capacity,
                num_q_heads,
                num_kv_heads,
                head_dim,
                head_dim**-0.5,
                library=attention_library,
                runtime=runtime,
            )
            runtime.device_synchronize()

            baseline_context = np.empty_like(query)
            candidate_context = np.empty_like(query)
            baseline_gated = np.empty(query.shape, dtype=np.uint16)
            candidate_gated = np.empty_like(baseline_gated)
            for host, device in (
                (baseline_context, baseline_context_device),
                (candidate_context, candidate_context_device),
                (baseline_gated, baseline_gated_device),
                (candidate_gated, candidate_gated_device),
            ):
                copy_device_to_host(host_array_ptr(host), device, runtime=runtime)

            np.testing.assert_array_equal(candidate_context, baseline_context)
            np.testing.assert_array_equal(candidate_gated, baseline_gated)
            expected = _ordered_attention_reference(
                query,
                logical_keys[:live_count],
                logical_values[:live_count],
                token_positions=token_positions[:live_count],
                evict_mask=evict_mask[:live_count],
                query_position=int(row_positions[0]),
                num_kv_heads=num_kv_heads,
            )
            kl, agreement = _quality(candidate_context, expected)
            assert np.isfinite(candidate_context).all()
            assert kl <= 0.05
            assert agreement >= 0.90

        context_sentinel = np.full_like(query, np.float32(-123.25))
        gated_sentinel = np.full(query.shape, np.uint16(0xA5A5), dtype=np.uint16)
        copy_host_to_device(
            candidate_context_device,
            host_array_ptr(context_sentinel),
            context_sentinel.nbytes,
            runtime=runtime,
        )
        copy_host_to_device(
            candidate_gated_device,
            host_array_ptr(gated_sentinel),
            gated_sentinel.nbytes,
            runtime=runtime,
        )
        live_counts[0] = 257
        runtime.memcpy(
            live_device.ptr,
            host_array_ptr(live_counts),
            live_counts.nbytes,
            HipMemcpyKind.HOST_TO_DEVICE,
        )
        laguna_global_attention_decode_single_page_softplus_gate_bf16_spans(
            query_device.ptr,
            key_device.ptr,
            value_device.ptr,
            candidate_context_device.ptr,
            gate_device.ptr,
            candidate_gated_device.ptr,
            spans,
            capacity,
            num_q_heads,
            num_kv_heads,
            head_dim,
            head_dim**-0.5,
            library=attention_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        untouched_context = np.empty_like(context_sentinel)
        untouched_gated = np.empty_like(gated_sentinel)
        copy_device_to_host(
            host_array_ptr(untouched_context),
            candidate_context_device,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(untouched_gated),
            candidate_gated_device,
            runtime=runtime,
        )
        np.testing.assert_array_equal(untouched_context, context_sentinel)
        np.testing.assert_array_equal(untouched_gated, gated_sentinel)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
