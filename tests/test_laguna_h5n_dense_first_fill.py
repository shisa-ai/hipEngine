from __future__ import annotations

import ctypes
import os
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.laguna_gguf import SLIDING_ATTENTION


_VARIANT = "swa_context_rows_qrow4_dense_first_fill_exact_spans"
_SYMBOL = "hipengine_laguna_swa_attention_prefill_qrow4_dense_first_fill_exact_bf16_spans"


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


def _ring_spans(capacity: int = 512) -> KVLiveSpans:
    return KVLiveSpans.sliding_ring(
        base_offsets=_tensor(0x1000, (capacity,), "int32"),
        live_counts=_tensor(0x2000, (1,), "int64"),
        token_positions=_tensor(0x3000, (capacity,), "int64"),
        evict_mask=_tensor(0x4000, (capacity,), "bool"),
        row_positions=_tensor(0x5000, (1,), "int64"),
        capacity=capacity,
        storage_dtype="bf16",
    )


def test_h5n_dense_first_fill_registry_and_preflight() -> None:
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        laguna_swa_attention_prefill_qrow4_dense_first_fill_exact_bf16_spans,
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
        is laguna_swa_attention_prefill_qrow4_dense_first_fill_exact_bf16_spans
    )
    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(
        KernelKey("hip_gfx1151", "laguna_attention_prefill", "bf16", _VARIANT)
    )

    calls: list[tuple[object, ...]] = []

    class FakeFn:
        argtypes = None
        restype = None

        def __call__(self, *args):
            calls.append(args)
            return 0

    library = SimpleNamespace(**{_SYMBOL: FakeFn()})
    common = {
        "rows": 128,
        "num_q_heads": 72,
        "num_kv_heads": 8,
        "head_dim": 128,
        "sliding_window": 512,
        "start_position": 256,
    }

    def launch(**overrides: object) -> None:
        arguments = common | overrides
        laguna_swa_attention_prefill_qrow4_dense_first_fill_exact_bf16_spans(
            0x6000,
            0x7000,
            0x8000,
            0x9000,
            0xA000,
            0xB000,
            arguments.pop("spans", _ring_spans()),
            int(arguments.pop("rows")),
            int(arguments.pop("num_q_heads")),
            int(arguments.pop("num_kv_heads")),
            int(arguments.pop("head_dim")),
            128**-0.5,
            **arguments,
            library=library,
            runtime=SimpleNamespace(),
        )

    launch()
    launch(start_position=384)
    assert len(calls) == 2
    for invalid in (
        {"rows": 127},
        {"num_q_heads": 48},
        {"num_kv_heads": 4},
        {"head_dim": 64},
        {"sliding_window": 256},
        {"start_position": None},
        {"start_position": 128},
        {"start_position": 512},
        {"spans": _ring_spans(384)},
    ):
        with pytest.raises(ValueError, match="dense-first-fill"):
            launch(**invalid)
    assert len(calls) == 2


def test_h5n_runtime_role_requires_dense_first_fill(monkeypatch) -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.runtime import laguna_kv as module

    role = "qrow4_m128_c256_exact"
    candidate = _VARIANT
    h5m = "swa_context_rows_qrow4_sourcequal_exact_spans"
    retained = "swa_context_rows_qrow4_m128_c256_exact_spans"
    assert hip_gfx1100.LAGUNA_SWA_PREFILL_ROLE_VARIANTS == {role: h5m}
    monkeypatch.setattr(
        hip_gfx1100,
        "LAGUNA_SWA_PREFILL_ROLE_VARIANTS",
        {role: candidate},
    )
    assert module._resolve_laguna_swa_prefill_role_variants("hip_gfx1100") == {
        role: candidate
    }

    state = SimpleNamespace(
        attention_type=SLIDING_ATTENTION,
        attention_prefill_variant=retained,
        capacity=512,
        q_heads=72,
        key_cache=SimpleNamespace(ptr=0x9000),
        value_cache=SimpleNamespace(ptr=0xA000),
        spans=_ring_spans(),
    )

    class FakeCache:
        sliding_window = 512
        swa_prefill_role_variants = {role: candidate}
        runtime = SimpleNamespace()
        dense = True
        _pending_positions: tuple[int, ...] = ()

        def layer(self, layer_id: int):
            assert layer_id == 0
            return state

        def _bulk_slice_spans(self, spans, **kwargs):
            return spans

        def can_dense_initial_prefill(self, layer_id: int, rows: int, **kwargs):
            return self.dense and layer_id == 0 and rows == 128

        def _resolve(self, layer: str, variant: str):
            assert layer == "laguna_attention_prefill"
            variants.append(variant)
            return lambda *args, **kwargs: None

    cache = FakeCache()
    variants: list[str] = []

    def dispatch(start: int, rows: int = 128, *, dense: bool = True) -> None:
        cache.dense = dense
        cache._pending_positions = tuple(range(start, start + rows))
        module.LagunaKVCache.attend_prefill(
            cache,
            0,
            0x6000,
            0x7000,
            0x8000,
            0xB000,
            rows,
        )

    dispatch(256)
    dispatch(384)
    dispatch(512)
    dispatch(256, dense=False)
    dispatch(256, rows=127)
    cache.swa_prefill_role_variants = {role: h5m}
    dispatch(512, dense=False)
    assert variants == [candidate, candidate, retained, retained, retained, h5m]


@pytest.mark.parametrize("start_position", [256, 384])
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h5n_dense_first_fill_matches_h5m_wave32_and_cpu(
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
        laguna_swa_attention_prefill_qrow4_dense_first_fill_exact_bf16_spans,
        laguna_swa_attention_prefill_qrow4_sourcequal_exact_bf16_spans,
        laguna_swa_attention_prefill_wave32_exact_bf16_spans,
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
    query_heads = 72
    kv_heads = 8
    head_dim = 128
    config = SimpleNamespace(
        block_count=1,
        layer_types=(SLIDING_ATTENTION,),
        head_counts=(query_heads,),
        head_count_kv=kv_heads,
        key_length=head_dim,
        value_length=head_dim,
        sliding_window=512,
    )
    before = memory_stats()
    cache = allocate_laguna_kv_cache(
        config,
        context_length=512,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    rng = np.random.default_rng(0x5A10 + start_position)
    keys = rng.normal(
        0.0, 0.12, size=(start_position + rows, kv_heads, head_dim)
    ).astype(np.float32)
    values = rng.normal(0.0, 0.12, size=keys.shape).astype(np.float32)
    queries = rng.normal(0.0, 0.12, size=(rows, query_heads, head_dim)).astype(
        np.float32
    )
    control = np.empty_like(queries)
    candidate = np.empty_like(queries)
    wave32 = np.empty_like(queries)
    allocations = []
    try:
        key_rows = malloc(keys.nbytes, runtime=runtime)
        value_rows = malloc(values.nbytes, runtime=runtime)
        query_rows = malloc(queries.nbytes, runtime=runtime)
        control_out = malloc(control.nbytes, runtime=runtime)
        candidate_out = malloc(candidate.nbytes, runtime=runtime)
        wave32_out = malloc(wave32.nbytes, runtime=runtime)
        allocations.extend(
            (
                key_rows,
                value_rows,
                query_rows,
                control_out,
                candidate_out,
                wave32_out,
            )
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

        cache.prepare_rows(tuple(range(start_position)))
        cache.append_rows(
            0,
            key_rows.ptr,
            value_rows.ptr,
            start_position,
            library=library,
        )
        cache.commit_rows()
        cache.prepare_rows(tuple(range(start_position, start_position + rows)))
        state = cache.layer(0)
        row_nbytes = kv_heads * head_dim * np.dtype(np.float32).itemsize
        current_key_ptr = key_rows.ptr + start_position * row_nbytes
        current_value_ptr = value_rows.ptr + start_position * row_nbytes
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
        laguna_swa_attention_prefill_qrow4_sourcequal_exact_bf16_spans(
            *common,
            control_out.ptr,
            *suffix,
            sliding_window=512,
            start_position=start_position,
            library=library,
            runtime=runtime,
        )
        laguna_swa_attention_prefill_qrow4_dense_first_fill_exact_bf16_spans(
            *common,
            candidate_out.ptr,
            *suffix,
            sliding_window=512,
            start_position=start_position,
            library=library,
            runtime=runtime,
        )
        laguna_swa_attention_prefill_wave32_exact_bf16_spans(
            *common,
            wave32_out.ptr,
            *suffix,
            sliding_window=512,
            start_position=start_position,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        for host, device in (
            (control, control_out),
            (candidate, candidate_out),
            (wave32, wave32_out),
        ):
            copy_device_to_host(
                host_array_ptr(host),
                device,
                host.nbytes,
                runtime=runtime,
            )
        cache.discard_rows()

        np.testing.assert_array_equal(candidate, control)
        np.testing.assert_array_equal(candidate, wave32)
        keys_bf16 = bf16_to_float32(float_array_to_bf16_bits(keys))
        values_bf16 = bf16_to_float32(float_array_to_bf16_bits(values))
        kv_for_q = np.arange(query_heads, dtype=np.int64) // (query_heads // kv_heads)
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
            expected = np.einsum(
                "ht,thd->hd",
                weights,
                expanded_values,
                dtype=np.float32,
            )
            np.testing.assert_allclose(
                candidate[row], expected, rtol=3e-4, atol=3e-4
            )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        cache.free()
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
