"""WPF-H6A exact dense-initial cached-only attention contract."""

from __future__ import annotations

import ctypes
import importlib
import os
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION

_STARTS = (0, 128, 256, 384)
_SWA_VARIANT = "swa_context_rows_qrow4_dense_initial_cached_exact_spans"
_GLOBAL_VARIANT = "global_context_rows_dense_initial_cached_exact_spans"
_SWA_FUNCTION = (
    "laguna_swa_attention_prefill_qrow4_dense_initial_cached_exact_bf16_spans"
)
_GLOBAL_FUNCTION = (
    "laguna_global_attention_prefill_dense_initial_cached_exact_bf16_spans"
)
_SWA_SYMBOL = (
    "hipengine_laguna_swa_attention_prefill_qrow4_"
    "dense_initial_cached_exact_bf16_spans"
)
_GLOBAL_SYMBOL = (
    "hipengine_laguna_global_attention_prefill_"
    "dense_initial_cached_exact_bf16_spans"
)
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


def _module():
    return importlib.import_module(
        "hipengine.kernels.hip_gfx1100.attention.laguna_kv"
    )


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _swa_spans(capacity: int = 512) -> KVLiveSpans:
    return KVLiveSpans.sliding_ring(
        base_offsets=_tensor(0x11000, (capacity,), "int32"),
        live_counts=_tensor(0x12000, (1,), "int64"),
        token_positions=_tensor(0x13000, (capacity,), "int64"),
        evict_mask=_tensor(0x14000, (capacity,), "bool"),
        row_positions=_tensor(0x15000, (1,), "int64"),
        capacity=capacity,
        storage_dtype="bf16",
    )


def _global_spans(capacity: int = 4096) -> KVLiveSpans:
    block_size = 256
    blocks = (capacity + block_size - 1) // block_size
    return KVLiveSpans.paged_dense(
        block_table=_tensor(0x21000, (blocks,), "int32"),
        live_counts=_tensor(0x22000, (1,), "int64"),
        token_positions=_tensor(0x23000, (capacity,), "int64"),
        evict_mask=_tensor(0x24000, (capacity,), "bool"),
        row_positions=_tensor(0x25000, (1,), "int64"),
        capacity=capacity,
        block_size=block_size,
        storage_dtype="bf16",
        span_role="prefill",
    )


def test_h6a_registry_preflight_backend_and_package_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.registry import KernelKey, is_registered, resolve

    module = _module()
    swa_candidate = getattr(module, _SWA_FUNCTION)
    global_candidate = getattr(module, _GLOBAL_FUNCTION)
    module.register_laguna_kv_attention_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant=_SWA_VARIANT,
        )
        is swa_candidate
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant=_GLOBAL_VARIANT,
        )
        is global_candidate
    )
    assert hip_gfx1100.LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS == _PACKAGE_H5R_POLICY
    load_backend_kernel_package("hip_gfx1151")
    for variant in (_SWA_VARIANT, _GLOBAL_VARIANT):
        assert not is_registered(
            KernelKey("hip_gfx1151", "laguna_attention_prefill", "bf16", variant)
        )

    calls: dict[str, list[tuple[object, ...]]] = {"swa": [], "global": []}

    class FakeFn:
        argtypes = None
        restype = None

        def __init__(self, role: str) -> None:
            self.role = role

        def __call__(self, *args: object) -> int:
            calls[self.role].append(args)
            return 0

    library = SimpleNamespace(
        **{
            _SWA_SYMBOL: FakeFn("swa"),
            _GLOBAL_SYMBOL: FakeFn("global"),
        }
    )
    common = (0x6000, 0x7000, 0x8000, 0x9000, 0xA000, 0xB000)

    def launch_swa(**overrides: object) -> None:
        arguments = {
            "spans": _swa_spans(),
            "rows": 128,
            "num_q_heads": 72,
            "num_kv_heads": 8,
            "head_dim": 128,
            "sliding_window": 512,
            "start_position": 0,
        }
        arguments.update(overrides)
        swa_candidate(
            *common,
            arguments.pop("spans"),
            arguments.pop("rows"),
            arguments.pop("num_q_heads"),
            arguments.pop("num_kv_heads"),
            arguments.pop("head_dim"),
            128**-0.5,
            **arguments,
            library=library,
            runtime=SimpleNamespace(),
        )

    def launch_global(**overrides: object) -> None:
        arguments = {
            "spans": _global_spans(),
            "rows": 128,
            "max_context_len": 4096,
            "num_q_heads": 48,
            "num_kv_heads": 8,
            "head_dim": 128,
            "start_position": 0,
        }
        arguments.update(overrides)
        global_candidate(
            *common,
            arguments.pop("spans"),
            arguments.pop("rows"),
            arguments.pop("max_context_len"),
            arguments.pop("num_q_heads"),
            arguments.pop("num_kv_heads"),
            arguments.pop("head_dim"),
            128**-0.5,
            **arguments,
            library=library,
            runtime=SimpleNamespace(),
        )

    for start in _STARTS:
        launch_swa(start_position=start)
        launch_global(start_position=start)
    assert len(calls["swa"]) == len(_STARTS)
    assert len(calls["global"]) == len(_STARTS)

    def fail_build(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid H6A preflight loaded HIP")

    monkeypatch.setattr(module, "build_laguna_kv_attention", fail_build)
    swa_invalid = (
        {"rows": 127},
        {"spans": _swa_spans(384)},
        {"num_q_heads": 48},
        {"num_kv_heads": 4},
        {"head_dim": 64},
        {"sliding_window": 256},
        {"start_position": None},
        {"start_position": 512},
    )
    for invalid in swa_invalid:
        with pytest.raises(ValueError, match="dense-initial cached-exact SWA"):
            launch_swa(**invalid)
    global_invalid = (
        {"rows": 127},
        {"spans": _global_spans(2048), "max_context_len": 2048},
        {"max_context_len": 2048},
        {"num_q_heads": 72},
        {"num_kv_heads": 4},
        {"head_dim": 64},
        {"start_position": None},
        {"start_position": 64},
        {"start_position": 512},
    )
    for invalid in global_invalid:
        with pytest.raises(ValueError, match="dense-initial cached-exact global"):
            launch_global(**invalid)
    assert len(calls["swa"]) == len(_STARTS)
    assert len(calls["global"]) == len(_STARTS)


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


def _span_snapshot(spans: KVLiveSpans, runtime) -> tuple[np.ndarray, ...]:
    return tuple(
        _copy_metadata(tensor, runtime)
        for tensor in (
            spans.base_offsets,
            spans.live_counts,
            spans.token_positions,
            spans.evict_mask,
            spans.row_positions,
        )
    )


def _cpu_rows(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    start_position: int,
) -> dict[int, np.ndarray]:
    from hipengine.loading.materialize import float_array_to_bf16_bits
    from hipengine.quant.gguf import bf16_to_float32

    keys_bf16 = bf16_to_float32(float_array_to_bf16_bits(keys))
    values_bf16 = bf16_to_float32(float_array_to_bf16_bits(values))
    kv_for_q = np.arange(query_heads, dtype=np.int64) // (query_heads // kv_heads)
    expected: dict[int, np.ndarray] = {}
    for row in (0, 63, 127):
        visible = start_position + row + 1
        expanded_keys = keys_bf16[:visible, kv_for_q, :]
        expanded_values = values_bf16[:visible, kv_for_q, :]
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
        expected[row] = np.einsum(
            "ht,thd->hd",
            weights,
            expanded_values,
            dtype=np.float32,
        )
    return expected


@pytest.mark.parametrize("attention_type", (SLIDING_ATTENTION, FULL_ATTENTION))
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h6a_dense_initial_cached_exact_matches_controls_cpu_and_spans(
    attention_type: str,
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
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    module = _module()
    candidate = getattr(
        module,
        _SWA_FUNCTION if attention_type == SLIDING_ATTENTION else _GLOBAL_FUNCTION,
    )
    control = (
        module.laguna_swa_attention_prefill_qrow4_cached_exact_bf16_spans
        if attention_type == SLIDING_ATTENTION
        else module.laguna_global_attention_prefill_cached_exact_bf16_spans
    )
    runtime = get_hip_runtime()
    library = module.build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    rows = 128
    query_heads = 72 if attention_type == SLIDING_ATTENTION else 48
    kv_heads = 8
    head_dim = 128
    capacity = 512 if attention_type == SLIDING_ATTENTION else 4096
    config = SimpleNamespace(
        block_count=1,
        layer_types=(attention_type,),
        head_counts=(query_heads,),
        head_count_kv=kv_heads,
        key_length=head_dim,
        value_length=head_dim,
        sliding_window=512,
    )
    before = memory_stats()
    for start_position in _STARTS:
        cache = allocate_laguna_kv_cache(
            config,
            context_length=capacity,
            backend="hip_gfx1151",
            runtime=runtime,
        )
        rng = np.random.default_rng(
            0x6A00 + start_position + (0 if attention_type == SLIDING_ATTENTION else 1)
        )
        total_rows = start_position + rows
        keys = rng.normal(
            0.0,
            0.12,
            size=(total_rows, kv_heads, head_dim),
        ).astype(np.float32)
        values = rng.normal(0.0, 0.12, size=keys.shape).astype(np.float32)
        queries = rng.normal(
            0.0,
            0.12,
            size=(rows, query_heads, head_dim),
        ).astype(np.float32)
        control_host = np.empty_like(queries)
        candidate_host = np.empty_like(queries)
        allocations = []
        try:
            key_rows = malloc(keys.nbytes, runtime=runtime)
            value_rows = malloc(values.nbytes, runtime=runtime)
            query_rows = malloc(queries.nbytes, runtime=runtime)
            control_out = malloc(control_host.nbytes, runtime=runtime)
            candidate_out = malloc(candidate_host.nbytes, runtime=runtime)
            allocations.extend(
                (key_rows, value_rows, query_rows, control_out, candidate_out)
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
                cache.prepare_rows(tuple(range(start_position)))
                cache.append_rows(
                    0,
                    key_rows.ptr,
                    value_rows.ptr,
                    start_position,
                    library=library,
                )
                cache.commit_rows()
            cache.prepare_rows(tuple(range(start_position, total_rows)))
            row_nbytes = kv_heads * head_dim * np.dtype(np.float32).itemsize
            current_key_ptr = key_rows.ptr + start_position * row_nbytes
            current_value_ptr = value_rows.ptr + start_position * row_nbytes
            cache.append_rows(
                0,
                current_key_ptr,
                current_value_ptr,
                rows,
                library=library,
            )
            state = cache.layer(0)
            before_spans = _span_snapshot(state.spans, runtime)
            common = (
                query_rows.ptr,
                current_key_ptr,
                current_value_ptr,
                state.key_cache.ptr,
                state.value_cache.ptr,
            )
            suffix = (
                state.spans,
                rows,
                query_heads,
                kv_heads,
                head_dim,
                head_dim**-0.5,
            )
            if attention_type == SLIDING_ATTENTION:
                control(
                    *common,
                    control_out.ptr,
                    *suffix,
                    sliding_window=512,
                    start_position=start_position,
                    library=library,
                    runtime=runtime,
                )
                candidate(
                    *common,
                    candidate_out.ptr,
                    *suffix,
                    sliding_window=512,
                    start_position=start_position,
                    library=library,
                    runtime=runtime,
                )
            else:
                global_suffix = (
                    state.spans,
                    rows,
                    capacity,
                    query_heads,
                    kv_heads,
                    head_dim,
                    head_dim**-0.5,
                )
                control(
                    *common,
                    control_out.ptr,
                    *global_suffix,
                    start_position=start_position,
                    library=library,
                    runtime=runtime,
                )
                candidate(
                    *common,
                    candidate_out.ptr,
                    *global_suffix,
                    start_position=start_position,
                    library=library,
                    runtime=runtime,
                )
            runtime.device_synchronize()
            for host, device in (
                (control_host, control_out),
                (candidate_host, candidate_out),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_array_equal(candidate_host, control_host)
            after_spans = _span_snapshot(state.spans, runtime)
            for actual, expected in zip(after_spans, before_spans, strict=True):
                np.testing.assert_array_equal(actual, expected)
            for row, expected in _cpu_rows(
                queries,
                keys,
                values,
                query_heads=query_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                start_position=start_position,
            ).items():
                np.testing.assert_allclose(
                    candidate_host[row],
                    expected,
                    rtol=3e-4,
                    atol=3e-4,
                )
            cache.discard_rows()
        finally:
            for allocation in reversed(allocations):
                free(allocation, runtime=runtime)
            cache.free()
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
