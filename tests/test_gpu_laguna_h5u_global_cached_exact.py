from __future__ import annotations

import ctypes
import os
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.laguna_gguf import FULL_ATTENTION


_VARIANT = "global_context_rows_cached_exact_spans"
_SYMBOL = "hipengine_laguna_global_attention_prefill_cached_exact_bf16_spans"
_REMOVED_H5R_GLOBAL_VARIANT = "global_context_rows_qrow4_cached_exact_spans"
_STARTS = (0, 128, 256, 384)
_PACKAGE_H5R_POLICY = {
    "swa_qrow4_m128_c512_no_wrap_exact": (
        "swa_context_rows_qrow4_cached_exact_spans"
    )
}


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


def _global_spans(capacity: int = 4096) -> KVLiveSpans:
    block_size = 256
    block_count = (capacity + block_size - 1) // block_size
    return KVLiveSpans.paged_dense(
        block_table=_tensor(0x11000, (block_count,), "int32"),
        live_counts=_tensor(0x12000, (1,), "int64"),
        token_positions=_tensor(0x13000, (capacity,), "int64"),
        evict_mask=_tensor(0x14000, (capacity,), "bool"),
        row_positions=_tensor(0x15000, (1,), "int64"),
        capacity=capacity,
        block_size=block_size,
        storage_dtype="bf16",
        span_role="prefill",
    )


def test_h5u_global_cached_exact_registry_preflight_and_policy_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.hip_gfx1100.attention import laguna_kv as module
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        laguna_global_attention_prefill_cached_exact_bf16_spans,
        register_laguna_kv_attention_kernels,
    )
    from hipengine.kernels.registry import KernelKey, is_registered, resolve

    register_laguna_kv_attention_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant=_VARIANT,
        )
        is laguna_global_attention_prefill_cached_exact_bf16_spans
    )
    assert not is_registered(
        KernelKey(
            "hip_gfx1100",
            "laguna_attention_prefill",
            "bf16",
            _REMOVED_H5R_GLOBAL_VARIANT,
        )
    )
    assert not hasattr(
        module,
        "laguna_global_attention_prefill_qrow4_cached_exact_bf16_spans",
    )
    assert hip_gfx1100.LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS == _PACKAGE_H5R_POLICY
    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(
        KernelKey("hip_gfx1151", "laguna_attention_prefill", "bf16", _VARIANT)
    )

    calls: list[tuple[object, ...]] = []

    class FakeFn:
        argtypes = None
        restype = None

        def __call__(self, *args: object) -> int:
            calls.append(args)
            return 0

    library = SimpleNamespace(**{_SYMBOL: FakeFn()})
    common = (0x6000, 0x7000, 0x8000, 0x9000, 0xA000, 0xB000)

    def launch(*, use_library: bool = True, **overrides: object) -> None:
        arguments = {
            "spans": _global_spans(),
            "rows": 128,
            "max_context_len": 4096,
            "num_q_heads": 48,
            "num_kv_heads": 8,
            "head_dim": 128,
            "start_position": 0,
        } | overrides
        laguna_global_attention_prefill_cached_exact_bf16_spans(
            *common,
            arguments.pop("spans"),
            arguments.pop("rows"),
            arguments.pop("max_context_len"),
            arguments.pop("num_q_heads"),
            arguments.pop("num_kv_heads"),
            arguments.pop("head_dim"),
            128**-0.5,
            **arguments,
            library=library if use_library else None,
            runtime=SimpleNamespace(),
        )

    for start in _STARTS:
        launch(start_position=start)
    assert len(calls) == len(_STARTS)
    for invalid in (
        {"rows": 127},
        {"spans": _global_spans(2048), "max_context_len": 2048},
        {"max_context_len": 2048},
        {"num_q_heads": 72},
        {"num_kv_heads": 4},
        {"head_dim": 64},
        {"start_position": None},
        {"start_position": 64},
        {"start_position": 512},
    ):
        with pytest.raises(ValueError, match="cached-exact global"):
            launch(**invalid)
    assert len(calls) == len(_STARTS)

    def fail_build(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid H5U preflight loaded HIP")

    monkeypatch.setattr(module, "build_laguna_kv_attention", fail_build)
    with pytest.raises(ValueError, match="cached-exact global"):
        launch(rows=64, use_library=False)
    assert len(calls) == len(_STARTS)


def _copy_metadata(tensor: Tensor, runtime) -> np.ndarray:
    from hipengine.core.hip import HipMemcpyKind
    from hipengine.core.memory import host_array_ptr

    dtype = {
        "int32": np.int32,
        "int64": np.int64,
        "bool": np.uint8,
    }[tensor.dtype.value]
    host = np.empty(tensor.numel, dtype=dtype)
    runtime.memcpy(
        host_array_ptr(host),
        tensor.ptr,
        host.nbytes,
        HipMemcpyKind.DEVICE_TO_HOST,
    )
    return host


@pytest.mark.parametrize("start_position", _STARTS)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h5u_global_cached_exact_matches_production_cpu_and_complete_spans(
    start_position: int,
) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
        memory_stats,
    )
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        build_laguna_kv_attention,
        laguna_global_attention_prefill_bf16_spans,
        laguna_global_attention_prefill_cached_exact_bf16_spans,
    )
    from hipengine.loading.materialize import float_array_to_bf16_bits
    from hipengine.quant.gguf import bf16_to_float32
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    rows = 128
    query_heads = 48
    kv_heads = 8
    head_dim = 128
    capacity = 4096
    config = SimpleNamespace(
        block_count=1,
        layer_types=(FULL_ATTENTION,),
        head_counts=(query_heads,),
        head_count_kv=kv_heads,
        key_length=head_dim,
        value_length=head_dim,
        sliding_window=512,
    )
    before = memory_stats()
    control = allocate_laguna_kv_cache(
        config,
        context_length=capacity,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    candidate = allocate_laguna_kv_cache(
        config,
        context_length=capacity,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    rng = np.random.default_rng(0x5A30 + start_position)
    total_rows = start_position + rows
    keys = rng.normal(0.0, 0.12, size=(total_rows, kv_heads, head_dim)).astype(
        np.float32
    )
    values = rng.normal(0.0, 0.12, size=keys.shape).astype(np.float32)
    queries = rng.normal(
        0.0,
        0.12,
        size=(rows, query_heads, head_dim),
    ).astype(np.float32)
    control_out = np.empty_like(queries)
    candidate_out = np.empty_like(queries)
    allocations = []
    try:
        key_rows = malloc(keys.nbytes, runtime=runtime)
        value_rows = malloc(values.nbytes, runtime=runtime)
        query_rows = malloc(queries.nbytes, runtime=runtime)
        control_device = malloc(control_out.nbytes, runtime=runtime)
        candidate_device = malloc(candidate_out.nbytes, runtime=runtime)
        allocations.extend(
            (key_rows, value_rows, query_rows, control_device, candidate_device)
        )
        for device, host in (
            (key_rows, keys),
            (value_rows, values),
            (query_rows, queries),
        ):
            copy_host_to_device(
                device,
                host_array_ptr(host),
                host.nbytes,
                runtime=runtime,
            )

        if start_position:
            seed_positions = tuple(range(start_position))
            for cache in (control, candidate):
                cache.prepare_rows(seed_positions)
                cache.append_rows(
                    0,
                    key_rows.ptr,
                    value_rows.ptr,
                    start_position,
                    library=library,
                )
                cache.commit_rows()
            if start_position == 384:
                control.evict_position(0, 17)
                candidate.evict_position(0, 17)

        positions = tuple(range(start_position, start_position + rows))
        control.prepare_rows(positions)
        candidate.prepare_rows(positions)
        row_nbytes = kv_heads * head_dim * np.dtype(np.float32).itemsize
        current_key_ptr = key_rows.ptr + start_position * row_nbytes
        current_value_ptr = value_rows.ptr + start_position * row_nbytes

        control_state = control.layer(0)
        laguna_global_attention_prefill_bf16_spans(
            query_rows.ptr,
            current_key_ptr,
            current_value_ptr,
            control_state.key_cache.ptr,
            control_state.value_cache.ptr,
            control_device.ptr,
            control_state.spans,
            rows,
            capacity,
            query_heads,
            kv_heads,
            head_dim,
            head_dim**-0.5,
            library=library,
            runtime=runtime,
        )

        candidate.append_rows(
            0,
            current_key_ptr,
            current_value_ptr,
            rows,
            library=library,
        )
        candidate_state = candidate.layer(0)
        laguna_global_attention_prefill_cached_exact_bf16_spans(
            query_rows.ptr,
            current_key_ptr,
            current_value_ptr,
            candidate_state.key_cache.ptr,
            candidate_state.value_cache.ptr,
            candidate_device.ptr,
            candidate_state.spans,
            rows,
            capacity,
            query_heads,
            kv_heads,
            head_dim,
            head_dim**-0.5,
            start_position=start_position,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        for host, device in (
            (control_out, control_device),
            (candidate_out, candidate_device),
        ):
            copy_device_to_host(
                host_array_ptr(host),
                device,
                host.nbytes,
                runtime=runtime,
            )
        np.testing.assert_array_equal(candidate_out, control_out)

        keys_bf16 = bf16_to_float32(float_array_to_bf16_bits(keys))
        values_bf16 = bf16_to_float32(float_array_to_bf16_bits(values))
        evicted = {17} if start_position == 384 else set()
        kv_for_q = np.arange(query_heads, dtype=np.int64) // (
            query_heads // kv_heads
        )
        for row in (0, 63, 127):
            visible_positions = np.array(
                [
                    position
                    for position in range(start_position + row + 1)
                    if position not in evicted
                ],
                dtype=np.int64,
            )
            expanded_keys = keys_bf16[visible_positions][:, kv_for_q, :]
            expanded_values = values_bf16[visible_positions][:, kv_for_q, :]
            scores = np.einsum(
                "hd,thd->ht",
                queries[row],
                expanded_keys,
                dtype=np.float32,
            )
            scores *= np.float32(head_dim**-0.5)
            scores -= np.max(scores, axis=1, keepdims=True)
            weights = np.exp(scores, dtype=np.float32)
            weights /= np.sum(weights, axis=1, keepdims=True, dtype=np.float32)
            expected = np.einsum(
                "ht,thd->hd",
                weights,
                expanded_values,
                dtype=np.float32,
            )
            np.testing.assert_allclose(
                candidate_out[row],
                expected,
                rtol=3e-4,
                atol=3e-4,
            )

        control.append_rows(
            0,
            current_key_ptr,
            current_value_ptr,
            rows,
            library=library,
        )
        runtime.device_synchronize()
        for field in ("base_offsets", "live_counts", "token_positions", "evict_mask"):
            control_tensor = getattr(control_state.spans, field)
            candidate_tensor = getattr(candidate_state.spans, field)
            assert control_tensor is not None and candidate_tensor is not None
            np.testing.assert_array_equal(
                _copy_metadata(candidate_tensor, runtime),
                _copy_metadata(control_tensor, runtime),
            )
        control.discard_rows()
        candidate.discard_rows()
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        control.free()
        candidate.free()
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
