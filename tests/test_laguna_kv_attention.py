from __future__ import annotations

import ctypes
import os
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.tensor import Tensor
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


def test_laguna_global_online_prefill_admits_model_scale_capacity() -> None:
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        laguna_dense_initial_cache_bf16_to_f32_spans,
        laguna_dense_initial_causal_softmax_wave_rows_f32_spans,
        laguna_global_attention_prefill_qrow4_cached_meta_online_bf16_spans,
        laguna_global_attention_prefill_qrow4_dense_initial_online_bf16_spans,
    )

    capacity = 131_072
    spans = KVLiveSpans.paged_dense(
        block_table=_tensor(0x1000, (capacity // 256,), "int32"),
        live_counts=_tensor(0x2000, (1,), "int64"),
        token_positions=_tensor(0x3000, (capacity,), "int64"),
        evict_mask=_tensor(0x4000, (capacity,), "bool"),
        row_positions=_tensor(0x5000, (1,), "int64"),
        capacity=capacity,
        block_size=256,
        storage_dtype="bf16",
    )
    calls: list[tuple[object, ...]] = []

    class FakeFn:
        argtypes = None
        restype = None

        def __call__(self, *args):
            calls.append(args)
            return 0

    library = SimpleNamespace(
        hipengine_laguna_dense_initial_cache_bf16_to_f32_spans=FakeFn(),
        hipengine_laguna_dense_initial_causal_softmax_wave_rows_f32_spans=FakeFn(),
        hipengine_laguna_global_attention_prefill_qrow4_cached_meta_online_bf16_spans=FakeFn(),
        hipengine_laguna_global_attention_prefill_qrow4_dense_initial_online_bf16_spans=FakeFn(),
    )
    common = (
        0x6000,
        0x7000,
        0x8000,
        0x9000,
        0xA000,
        0xB000,
        spans,
        128,
        capacity,
        48,
        8,
        128,
        128**-0.5,
    )
    laguna_global_attention_prefill_qrow4_cached_meta_online_bf16_spans(
        *common,
        library=library,
        runtime=SimpleNamespace(),
    )
    laguna_global_attention_prefill_qrow4_dense_initial_online_bf16_spans(
        *common,
        start_position=0,
        library=library,
        runtime=SimpleNamespace(),
    )
    laguna_dense_initial_cache_bf16_to_f32_spans(
        0x9000,
        0xA000,
        0xB000,
        0xC000,
        spans,
        capacity,
        8,
        128,
        library=library,
        runtime=SimpleNamespace(),
    )
    laguna_dense_initial_causal_softmax_wave_rows_f32_spans(
        0xD000,
        spans,
        128,
        capacity,
        48,
        capacity - 128,
        128**-0.5,
        library=library,
        runtime=SimpleNamespace(),
    )
    assert len(calls) == 4


def test_laguna_long_hipblaslt_shape_and_algorithm_bands() -> None:
    from hipengine.runtime.laguna_attention_hipblaslt import (
        LagunaAttentionHipblasLt,
        _preferred_algorithm_index,
    )

    route = object.__new__(LagunaAttentionHipblasLt)
    route.max_context = 131072
    route.max_q_heads = 48
    route.query_rows = 128
    assert route._supports_shape(
        rows=128,
        start_position=130944,
        num_q_heads=48,
        num_kv_heads=8,
        head_dim=128,
    )
    assert not route._supports_shape(
        rows=128,
        start_position=130944,
        num_q_heads=72,
        num_kv_heads=8,
        head_dim=128,
    )
    expected = {
        640: (30, 0),
        2048: (20, 19),
        4096: (20, 25),
        16384: (28, 1),
        65536: (28, 3),
        131072: (28, 3),
    }
    for context, (qk_index, pv_index) in expected.items():
        assert _preferred_algorithm_index(
            query_rows=128,
            query_heads=48,
            context=context,
            operation="qk",
            packed_queries=True,
        ) == qk_index
        assert _preferred_algorithm_index(
            query_rows=128,
            query_heads=48,
            context=context,
            operation="pv",
            packed_queries=True,
        ) == pv_index
    assert _preferred_algorithm_index(
        query_rows=2_048,
        query_heads=48,
        context=2_048,
        operation="qk",
        packed_queries=True,
    ) == 15
    assert _preferred_algorithm_index(
        query_rows=2_048,
        query_heads=48,
        context=4_096,
        operation="pv",
        packed_queries=True,
    ) == 2


def test_laguna_swa_hipblaslt_shape_contract() -> None:
    from hipengine.runtime.laguna_attention_hipblaslt import (
        LagunaSwaAttentionHipblasLt,
    )

    assert LagunaSwaAttentionHipblasLt.supports(
        rows=128,
        start_position=512,
        num_q_heads=72,
        num_kv_heads=8,
        head_dim=128,
        sliding_window=512,
    )
    assert not LagunaSwaAttentionHipblasLt.supports(
        rows=127,
        start_position=512,
        num_q_heads=72,
        num_kv_heads=8,
        head_dim=128,
        sliding_window=512,
    )
    assert not LagunaSwaAttentionHipblasLt.supports(
        rows=128,
        start_position=384,
        num_q_heads=72,
        num_kv_heads=8,
        head_dim=128,
        sliding_window=512,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_swa_hipblaslt_matches_online_route_after_ring_wrap() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        build_laguna_kv_attention,
    )
    from hipengine.runtime.laguna_attention_hipblaslt import (
        LagunaSwaAttentionHipblasLt,
    )
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    config = SimpleNamespace(
        block_count=1,
        layer_types=(SLIDING_ATTENTION,),
        head_counts=(72,),
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )
    cache = allocate_laguna_kv_cache(
        config,
        context_length=640,
        backend="hip_gfx1151",
        runtime=runtime,
        swa_prefill_variant="swa_context_rows_qrow2_online_spans",
    )
    route = LagunaSwaAttentionHipblasLt(runtime=runtime)
    rng = np.random.default_rng(0x1C2)
    seed_rows = 512
    rows = 128
    keys = rng.normal(0.0, 0.12, size=(seed_rows + rows, 8, 128)).astype(
        np.float32
    )
    values = rng.normal(0.0, 0.12, size=keys.shape).astype(np.float32)
    queries = rng.normal(0.0, 0.12, size=(rows, 72, 128)).astype(np.float32)
    baseline = np.empty_like(queries)
    candidate = np.empty_like(queries)
    allocations = []
    try:
        key_rows = malloc(keys.nbytes, runtime=runtime)
        value_rows = malloc(values.nbytes, runtime=runtime)
        query_rows = malloc(queries.nbytes, runtime=runtime)
        baseline_out = malloc(baseline.nbytes, runtime=runtime)
        candidate_out = malloc(candidate.nbytes, runtime=runtime)
        allocations.extend(
            (key_rows, value_rows, query_rows, baseline_out, candidate_out)
        )
        for buffer, array in (
            (key_rows, keys),
            (value_rows, values),
            (query_rows, queries),
        ):
            copy_host_to_device(
                buffer,
                host_array_ptr(array),
                array.nbytes,
                runtime=runtime,
            )
        cache.prepare_rows(tuple(range(seed_rows)))
        cache.append_rows(
            0,
            key_rows.ptr,
            value_rows.ptr,
            seed_rows,
            library=library,
        )
        cache.commit_rows()
        cache.prepare_rows(tuple(range(seed_rows, seed_rows + rows)))
        row_nbytes = 8 * 128 * np.dtype(np.float32).itemsize
        current_key_ptr = key_rows.ptr + seed_rows * row_nbytes
        current_value_ptr = value_rows.ptr + seed_rows * row_nbytes
        cache.attend_prefill(
            0,
            query_rows.ptr,
            current_key_ptr,
            current_value_ptr,
            baseline_out.ptr,
            rows,
            library=library,
        )
        state = cache.layer(0)
        route.launch(
            query_rows.ptr,
            current_key_ptr,
            current_value_ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
            candidate_out.ptr,
            state.spans,
            rows=rows,
            start_position=seed_rows,
            num_q_heads=72,
            num_kv_heads=8,
            head_dim=128,
            sliding_window=512,
            scale=128**-0.5,
            kv_library=library,
        )
        runtime.device_synchronize()
        for host, device in (
            (baseline, baseline_out),
            (candidate, candidate_out),
        ):
            copy_device_to_host(
                host_array_ptr(host),
                device,
                host.nbytes,
                runtime=runtime,
            )
        cache.discard_rows()
    finally:
        route.close()
        cache.free()
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
    np.testing.assert_allclose(candidate, baseline, rtol=2e-3, atol=2e-4)


def test_kv_live_spans_sliding_ring_requires_complete_absolute_metadata() -> None:
    spans = _ring_spans()

    assert spans.spans_mode == "sliding_ring"
    assert spans.max_live_count == 512
    assert spans.token_positions is not None
    assert spans.token_positions.dtype is DType.INT64
    assert spans.evict_mask is not None
    assert spans.row_positions is not None

    with pytest.raises(ValueError, match="token_positions"):
        KVLiveSpans(
            base_offsets=_tensor(1, (512,), "int32"),
            live_counts=_tensor(2, (1,), "int64"),
            max_live_count=512,
            token_positions=None,
            evict_mask=_tensor(4, (512,), "bool"),
            row_positions=_tensor(5, (1,), "int64"),
            storage_dtype=DType.BF16,
            spans_mode="sliding_ring",
        )
    with pytest.raises(ValueError, match="capacity"):
        KVLiveSpans.sliding_ring(
            base_offsets=_tensor(1, (511,), "int32"),
            live_counts=_tensor(2, (1,), "int64"),
            token_positions=_tensor(3, (512,), "int64"),
            evict_mask=_tensor(4, (512,), "bool"),
            row_positions=_tensor(5, (1,), "int64"),
            capacity=512,
            storage_dtype="bf16",
        )


def test_laguna_swa_build_plan_registry_and_validation(tmp_path) -> None:
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        laguna_global_attention_decode_bf16_spans,
        laguna_global_attention_prefill_bf16_spans,
        laguna_global_attention_prefill_qrow2_online_bf16_spans,
        laguna_global_attention_prefill_qrow4_cached_meta_online_bf16_spans,
        laguna_global_attention_prefill_qrow4_cached_online_bf16_spans,
        laguna_global_attention_prefill_qrow4_dense_initial_online_bf16_spans,
        laguna_global_attention_prefill_qrow4_m128_online_bf16_spans,
        laguna_global_attention_prefill_qrow4_online_bf16_spans,
        laguna_global_attention_prefill_qrow6_cached_meta_online_bf16_spans,
        laguna_global_attention_prefill_qrow6_dense_initial_online_bf16_spans,
        laguna_global_write_kv_rows_f32_spans,
        laguna_swa_attention_decode_bf16_spans,
        laguna_swa_attention_decode_token4_exact_bf16_spans,
        laguna_swa_attention_prefill_bf16_spans,
        laguna_swa_attention_prefill_qrow2_m128_c128_exact_bf16_spans,
        laguna_swa_attention_prefill_qrow2_exact_bf16_spans,
        laguna_swa_attention_prefill_qrow2_online_bf16_spans,
        laguna_swa_attention_prefill_qrow4_cached_meta_online_bf16_spans,
        laguna_swa_attention_prefill_qrow4_dense_initial_online_bf16_spans,
        laguna_swa_attention_prefill_qrow4_cached_online_bf16_spans,
        laguna_swa_attention_prefill_qrow4_m128_online_bf16_spans,
        laguna_swa_attention_prefill_qrow4_online_bf16_spans,
        laguna_swa_attention_prefill_qrow4_sourcequal_online_bf16_spans,
        laguna_swa_attention_prefill_wave32_exact_bf16_spans,
        laguna_swa_write_kv_f32_spans,
        laguna_swa_write_kv_rows_f32_spans,
        plan_laguna_kv_attention_build,
        register_laguna_kv_attention_kernels,
    )
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.registry import resolve

    artifact = plan_laguna_kv_attention_build(
        cache_root=tmp_path / "cache",
        compiler_version="hipcc laguna swa test version",
    )
    assert artifact.family == "laguna_kv_attention"
    assert artifact.output_path.name == "laguna_kv_attention.so"
    assert any(str(path).endswith("laguna_kv_attention.hip") for path in artifact.sources)
    assert not artifact.cache_dir.exists()

    register_laguna_kv_attention_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_decode",
            quant="bf16",
            variant="global_context_spans",
        )
        is laguna_global_attention_decode_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_decode",
            quant="bf16",
            variant="swa_context_spans",
        )
        is laguna_swa_attention_decode_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_decode",
            quant="bf16",
            variant="swa_context_token4_exact_spans",
        )
        is laguna_swa_attention_decode_token4_exact_bf16_spans
    )
    load_backend_kernel_package("hip_gfx1151")
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_decode",
            quant="bf16",
            variant="swa_context_spans",
        )
        is laguna_swa_attention_decode_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="global_context_rows_spans",
        )
        is laguna_global_attention_prefill_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="global_context_rows_qrow2_online_spans",
        )
        is laguna_global_attention_prefill_qrow2_online_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="global_context_rows_qrow4_online_spans",
        )
        is laguna_global_attention_prefill_qrow4_online_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="global_context_rows_qrow4_cached_meta_online_spans",
        )
        is laguna_global_attention_prefill_qrow4_cached_meta_online_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="global_context_rows_qrow4_dense_initial_online_spans",
        )
        is laguna_global_attention_prefill_qrow4_dense_initial_online_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="global_context_rows_qrow6_cached_meta_online_spans",
        )
        is laguna_global_attention_prefill_qrow6_cached_meta_online_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="global_context_rows_qrow6_dense_initial_online_spans",
        )
        is laguna_global_attention_prefill_qrow6_dense_initial_online_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="global_context_rows_qrow4_cached_online_spans",
        )
        is laguna_global_attention_prefill_qrow4_cached_online_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="global_context_rows_qrow4_m128_online_spans",
        )
        is laguna_global_attention_prefill_qrow4_m128_online_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="swa_context_rows_spans",
        )
        is laguna_swa_attention_prefill_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="swa_context_rows_wave32_exact_spans",
        )
        is laguna_swa_attention_prefill_wave32_exact_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="swa_context_rows_qrow2_exact_spans",
        )
        is laguna_swa_attention_prefill_qrow2_exact_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="swa_context_rows_qrow2_online_spans",
        )
        is laguna_swa_attention_prefill_qrow2_online_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="swa_context_rows_qrow4_online_spans",
        )
        is laguna_swa_attention_prefill_qrow4_online_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="swa_context_rows_qrow4_sourcequal_online_spans",
        )
        is laguna_swa_attention_prefill_qrow4_sourcequal_online_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="swa_context_rows_qrow4_cached_meta_online_spans",
        )
        is laguna_swa_attention_prefill_qrow4_cached_meta_online_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="swa_context_rows_qrow4_dense_initial_online_spans",
        )
        is laguna_swa_attention_prefill_qrow4_dense_initial_online_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="swa_context_rows_qrow4_cached_online_spans",
        )
        is laguna_swa_attention_prefill_qrow4_cached_online_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="swa_context_rows_qrow4_m128_online_spans",
        )
        is laguna_swa_attention_prefill_qrow4_m128_online_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant="swa_context_rows_qrow2_m128_c128_exact_spans",
        )
        is laguna_swa_attention_prefill_qrow2_m128_c128_exact_bf16_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_kv_write",
            quant="bf16",
            variant="global_f32_rows_spans",
        )
        is laguna_global_write_kv_rows_f32_spans
    )
    assert (
        resolve(
            backend="hip_gfx1151",
            layer="laguna_kv_write",
            quant="bf16",
            variant="swa_f32_rows_spans",
        )
        is laguna_swa_write_kv_rows_f32_spans
    )

    with pytest.raises(ValueError, match="num_kv_heads"):
        laguna_swa_write_kv_f32_spans(0, 0, 0, 0, _ring_spans(), 0, 128)
    with pytest.raises(ValueError, match="divisible"):
        laguna_swa_attention_decode_bf16_spans(
            0,
            0,
            0,
            0,
            _ring_spans(),
            70,
            8,
            128,
            128**-0.5,
        )


def test_laguna_swa_qrow2_auto_requires_m128_and_128_prior_tokens(
    monkeypatch,
) -> None:
    import hipengine.kernels.hip_gfx1100.attention.laguna_kv as module

    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "laguna_swa_attention_prefill_wave32_exact_bf16_spans",
        lambda *args, **kwargs: calls.append("wave32"),
    )
    monkeypatch.setattr(
        module,
        "laguna_swa_attention_prefill_qrow2_exact_bf16_spans",
        lambda *args, **kwargs: calls.append("qrow2"),
    )
    common = (1, 2, 3, 4, 5, 6, _ring_spans())
    module.laguna_swa_attention_prefill_qrow2_m128_c128_exact_bf16_spans(
        *common,
        127,
        72,
        8,
        128,
        128**-0.5,
        start_position=128,
    )
    module.laguna_swa_attention_prefill_qrow2_m128_c128_exact_bf16_spans(
        *common,
        128,
        72,
        8,
        128,
        128**-0.5,
        start_position=127,
    )
    module.laguna_swa_attention_prefill_qrow2_m128_c128_exact_bf16_spans(
        *common,
        128,
        72,
        8,
        128,
        128**-0.5,
        start_position=128,
    )
    assert calls == ["wave32", "wave32", "qrow2"]


class _FakeRuntime:
    def __init__(self, *, fail_malloc_at: int | None = None) -> None:
        self.next_ptr = 0x10000000
        self.allocations: dict[int, int] = {}
        self.freed: list[int] = []
        self.copies: list[tuple[int, int, HipMemcpyKind]] = []
        self.memsets: list[tuple[int, int, int]] = []
        self.fail_malloc_at = fail_malloc_at
        self.malloc_calls = 0

    def malloc(self, nbytes: int) -> int:
        self.malloc_calls += 1
        if self.fail_malloc_at == self.malloc_calls:
            raise MemoryError("synthetic Laguna KV allocation failure")
        ptr = self.next_ptr
        self.next_ptr += 0x1000
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.freed.append(int(ptr))
        self.allocations.pop(int(ptr), None)

    def memcpy(self, dst: int, src: int, count: int, kind: HipMemcpyKind) -> None:
        assert int(dst) in self.allocations or any(
            base < int(dst) < base + nbytes for base, nbytes in self.allocations.items()
        )
        assert kind == HipMemcpyKind.HOST_TO_DEVICE
        self.copies.append((int(dst), int(count), kind))

    def memset(self, dst: int, value: int, nbytes: int) -> None:
        assert int(dst) in self.allocations or any(
            base < int(dst) < base + size for base, size in self.allocations.items()
        )
        self.memsets.append((int(dst), int(value), int(nbytes)))


def _production_config() -> SimpleNamespace:
    layer_types = tuple(
        FULL_ATTENTION if layer_id % 4 == 0 else SLIDING_ATTENTION for layer_id in range(48)
    )
    head_counts = tuple(
        48 if attention_type == FULL_ATTENTION else 72 for attention_type in layer_types
    )
    return SimpleNamespace(
        block_count=48,
        layer_types=layer_types,
        head_counts=head_counts,
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )


def test_laguna_kv_owner_allocates_12_global_36_bounded_rings_and_tears_down() -> None:
    from hipengine.runtime.laguna_kv import (
        allocate_laguna_kv_cache,
        resolve_laguna_global_prefill_variant,
        resolve_laguna_swa_decode_variant,
        resolve_laguna_swa_prefill_variant,
    )

    runtime = _FakeRuntime()
    cache = allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    bytes_per_token = 8 * 128 * 2 * 2
    expected_payload = bytes_per_token * (12 * 4096 + 36 * 512)

    assert len(cache.layers) == 48
    assert sum(layer.attention_type == FULL_ATTENTION for layer in cache.layers) == 12
    assert sum(layer.attention_type == SLIDING_ATTENTION for layer in cache.layers) == 36
    assert cache.payload_nbytes == expected_payload == 264 * 1024 * 1024
    assert all(
        layer.capacity == (4096 if layer.attention_type == FULL_ATTENTION else 512)
        for layer in cache.layers
    )
    assert all(
        layer.spans.spans_mode
        == ("uniform" if layer.attention_type == FULL_ATTENTION else "sliding_ring")
        for layer in cache.layers
    )
    assert all(layer.spans.token_positions is not None for layer in cache.layers)
    assert all(layer.spans.evict_mask is not None for layer in cache.layers)
    assert all(
        layer.attention_prefill_variant
        == "global_context_rows_qrow4_m128_online_spans"
        for layer in cache.layers
        if layer.attention_type == FULL_ATTENTION
    )
    assert all(
        layer.attention_variant == "swa_context_spans"
        for layer in cache.layers
        if layer.attention_type == SLIDING_ATTENTION
    )
    assert all(
        layer.attention_prefill_variant == "swa_context_rows_qrow4_m128_online_spans"
        for layer in cache.layers
        if layer.attention_type == SLIDING_ATTENTION
    )
    assert cache.allocation_count == 243
    assert (
        resolve_laguna_global_prefill_variant("hip_gfx1151")
        == "global_context_rows_qrow4_m128_online_spans"
    )
    assert (
        resolve_laguna_global_prefill_variant(
            "hip_gfx1151",
            "global_context_rows_spans",
        )
        == "global_context_rows_spans"
    )
    assert (
        resolve_laguna_swa_prefill_variant("hip_gfx1151")
        == "swa_context_rows_qrow4_m128_online_spans"
    )
    assert (
        resolve_laguna_swa_prefill_variant(
            "hip_gfx1151",
            "swa_context_rows_qrow2_m128_c128_exact_spans",
        )
        == "swa_context_rows_qrow2_m128_c128_exact_spans"
    )
    assert (
        resolve_laguna_swa_decode_variant("hip_gfx1100")
        == "swa_context_token4_exact_spans"
    )
    assert resolve_laguna_swa_decode_variant("hip_gfx1151") == "swa_context_spans"
    assert (
        resolve_laguna_swa_decode_variant("hip_gfx1100", "swa_context_spans")
        == "swa_context_spans"
    )
    assert resolve_laguna_swa_prefill_variant("hip_gfx1100") == "swa_context_rows_spans"
    assert (
        resolve_laguna_swa_prefill_variant("hip_gfx1151", "swa_context_rows_spans")
        == "swa_context_rows_spans"
    )
    assert (
        resolve_laguna_swa_prefill_variant(
            "hip_gfx1151",
            "swa_context_rows_qrow2_online_spans",
        )
        == "swa_context_rows_qrow2_online_spans"
    )

    cache.prepare_position(0)
    cache.prepare_position(1)
    with pytest.raises(ValueError, match="token-serial"):
        cache.prepare_position(3)
    assert cache.position == 1

    cache.prepare_rows((2, 3, 4))
    assert cache.pending_positions == (2, 3, 4)
    with pytest.raises(RuntimeError, match="bulk positions are pending"):
        cache.prepare_position(2)
    cache.discard_rows()
    assert cache.position == 1
    assert cache.pending_positions == ()
    cache.prepare_rows((2, 3, 4))
    cache.commit_rows()
    assert cache.position == 4
    assert cache.pending_positions == ()
    with pytest.raises(ValueError, match="consecutive"):
        cache.prepare_rows((5, 7))
    wide_positions = tuple(range(5, 5 + 2_048))
    cache.prepare_rows(wide_positions)
    assert cache.pending_positions == wide_positions
    cache.discard_rows()

    cache.reset()
    assert cache.position == -1
    assert cache.pending_positions == ()
    assert len(runtime.memsets) == 48 * 3
    assert {value for _, value, _ in runtime.memsets} == {0, 1, 0xFF}

    allocated_count = len(runtime.allocations)
    assert allocated_count > 96
    cache.free()
    assert len(runtime.freed) == allocated_count
    assert runtime.allocations == {}
    cache.free()


def test_laguna_kv_bulk_slice_uses_resident_row_position_view() -> None:
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = _FakeRuntime()
    cache = allocate_laguna_kv_cache(
        _production_config(),
        context_length=16_384,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    calls: list[tuple[str, tuple, dict]] = []

    def resolve(layer: str, variant: str):
        del variant

        def record(*args, **kwargs):
            calls.append((layer, args, kwargs))

        return record

    cache._resolve = resolve
    positions_ptr = 0x71000000
    try:
        cache.prepare_rows(tuple(range(256)))
        assert cache.can_preappend_prefill(0, 128, row_offset=0)
        assert cache.can_preappend_prefill(1, 128, row_offset=128)
        assert not cache.can_preappend_prefill(1, 64, row_offset=0)
        cache.attend_prefill(
            1,
            0x1000,
            0x2000,
            0x3000,
            0x4000,
            128,
            row_offset=128,
            row_positions_ptr=positions_ptr + 128 * DType.INT64.itemsize,
        )
        cache.append_rows(
            1,
            0x2000,
            0x3000,
            128,
            row_offset=128,
            row_positions_ptr=positions_ptr + 128 * DType.INT64.itemsize,
        )

        attention_spans = calls[0][1][6]
        append_spans = calls[1][1][4]
        assert attention_spans.row_positions.ptr == positions_ptr + 128 * 8
        assert append_spans.row_positions.ptr == positions_ptr + 128 * 8
        assert attention_spans.live_counts.ptr == cache.layer(1).spans.live_counts.ptr
        assert append_spans.live_counts.ptr == cache.layer(1).append_spans.live_counts.ptr
        assert calls[0][2]["start_position"] == 128
        with pytest.raises(ValueError, match="slice"):
            cache.attend_prefill(
                1,
                0x1000,
                0x2000,
                0x3000,
                0x4000,
                129,
                row_offset=128,
                row_positions_ptr=positions_ptr + 128 * 8,
            )
        cache.discard_rows()
        cache.position = 511
        cache.prepare_rows(tuple(range(512, 640)))
        assert cache.can_preappend_prefill(0, 128)
        assert not cache.can_preappend_prefill(1, 128)
        cache.discard_rows()
        wide_positions = tuple(range(512, 512 + 2_048))
        cache.prepare_rows(wide_positions)
        assert cache.can_preappend_prefill(0, 2_048)
        cache.append_rows(0, 0x2000, 0x3000, 2_048)
        with pytest.raises(ValueError, match="SWA ring"):
            cache.append_rows(1, 0x2000, 0x3000, 2_048)
        cache.append_rows(
            1,
            0x2000,
            0x3000,
            128,
            row_offset=1_024,
            row_positions_ptr=positions_ptr + 1_024 * DType.INT64.itemsize,
        )
        wide_spans = calls[-1][1][4]
        assert wide_spans.row_positions.ptr == positions_ptr + 1_024 * 8
        cache.discard_rows()
    finally:
        cache.free()


def test_laguna_dense_initial_policy_qualifies_complete_unmodified_tiles() -> None:
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = _FakeRuntime()
    cache = allocate_laguna_kv_cache(
        _production_config(),
        context_length=1024,
        backend="hip_gfx1151",
        runtime=runtime,
        prefill_cached_meta=True,
        prefill_global_qrow6=True,
        prefill_dense_initial=True,
    )
    variants: list[str] = []
    starts: list[int] = []

    def resolve(layer: str, variant: str):
        assert layer == "laguna_attention_prefill"
        variants.append(variant)

        def record(*args, **kwargs):
            starts.append(int(kwargs["start_position"]))

        return record

    cache._resolve = resolve
    try:
        cache.prepare_rows(tuple(range(256)))
        for layer_id, row_offset in ((0, 0), (0, 128), (1, 0), (1, 128)):
            cache.attend_prefill_cached(
                layer_id,
                0x1000,
                0x2000,
                0x3000,
                0x4000,
                128,
                row_offset=row_offset,
                row_positions_ptr=0x5000 + row_offset * DType.INT64.itemsize,
            )
        assert variants == [
            "global_context_rows_qrow4_dense_initial_online_spans",
            "global_context_rows_qrow6_dense_initial_online_spans",
            "swa_context_rows_qrow4_dense_initial_online_spans",
            "swa_context_rows_qrow4_dense_initial_online_spans",
        ]
        assert starts == [0, 128, 0, 128]
    finally:
        cache.free()


def test_laguna_dense_initial_policy_falls_back_after_explicit_eviction() -> None:
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = _FakeRuntime()
    cache = allocate_laguna_kv_cache(
        _production_config(),
        context_length=512,
        backend="hip_gfx1151",
        runtime=runtime,
        prefill_cached_meta=True,
        prefill_global_qrow6=True,
        prefill_dense_initial=True,
    )
    variants: list[str] = []

    def resolve(layer: str, variant: str):
        assert layer == "laguna_attention_prefill"
        variants.append(variant)
        return lambda *args, **kwargs: None

    cache._resolve = resolve
    try:
        cache.position = 127
        cache.evict_position(0, 0)
        cache.prepare_rows(tuple(range(128, 256)))
        cache.attend_prefill_cached(
            0,
            0x1000,
            0x2000,
            0x3000,
            0x4000,
            128,
        )
        assert variants == [
            "global_context_rows_qrow6_cached_meta_online_spans",
        ]
    finally:
        cache.free()


def test_laguna_cached_metadata_policy_has_explicit_full_rollback() -> None:
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = _FakeRuntime()
    cache = allocate_laguna_kv_cache(
        _production_config(),
        context_length=512,
        backend="hip_gfx1151",
        runtime=runtime,
        prefill_cached_meta=False,
        prefill_global_qrow6=False,
        prefill_dense_initial=False,
    )
    variants: list[str] = []

    def resolve(layer: str, variant: str):
        assert layer == "laguna_attention_prefill"
        variants.append(variant)
        return lambda *args, **kwargs: None

    cache._resolve = resolve
    try:
        cache.prepare_rows(tuple(range(128)))
        for layer_id in (0, 1):
            cache.attend_prefill_cached(
                layer_id,
                0x1000,
                0x2000,
                0x3000,
                0x4000,
                128,
            )
        assert variants == [
            "global_context_rows_qrow4_cached_online_spans",
            "swa_context_rows_qrow4_cached_online_spans",
        ]
    finally:
        cache.free()


def test_laguna_kv_owner_cleans_partial_allocation_failure() -> None:
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    with pytest.raises(ValueError, match="global prefill variant"):
        allocate_laguna_kv_cache(
            _production_config(),
            context_length=4096,
            backend="hip_gfx1151",
            runtime=_FakeRuntime(),
            global_prefill_variant="missing",
        )

    with pytest.raises(ValueError, match="SWA prefill variant"):
        allocate_laguna_kv_cache(
            _production_config(),
            context_length=4096,
            backend="hip_gfx1151",
            runtime=_FakeRuntime(),
            swa_prefill_variant="missing",
        )

    runtime = _FakeRuntime(fail_malloc_at=12)
    with pytest.raises(MemoryError, match="synthetic Laguna KV"):
        allocate_laguna_kv_cache(
            _production_config(),
            context_length=4096,
            backend="hip_gfx1151",
            runtime=runtime,
        )
    assert runtime.allocations == {}


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_production_kv_owner_live_allocation_and_teardown() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    before = memory_stats()
    cache = allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    try:
        during = memory_stats()
        assert cache.payload_nbytes == 264 * 1024 * 1024
        assert (
            during["current_allocated_bytes"] - before["current_allocated_bytes"]
            == cache.resident_nbytes
        )
        assert during["active_allocations"] - before["active_allocations"] == cache.allocation_count
    finally:
        cache.free()
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_global_and_swa_token_serial_attention_match_cpu_across_wraps() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        build_laguna_kv_attention,
        laguna_global_attention_decode_split_exact_bf16_spans,
        laguna_swa_attention_decode_split_exact_bf16_spans,
        laguna_swa_attention_decode_split_tile16_exact_bf16_spans,
        laguna_swa_attention_decode_token4_exact_bf16_spans,
    )
    from hipengine.loading.materialize import float_array_to_bf16_bits
    from hipengine.quant.gguf import bf16_to_float32
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    kv_library = build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    config = SimpleNamespace(
        block_count=2,
        layer_types=(FULL_ATTENTION, SLIDING_ATTENTION),
        head_counts=(48, 72),
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )
    cache = allocate_laguna_kv_cache(
        config,
        context_length=1026,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    global_offsets = np.roll(
        np.arange(cache.layer(0).spans.base_offsets.numel, dtype=np.int32),
        1,
    )
    swa_offsets = np.arange(511, -1, -1, dtype=np.int32)
    runtime.memcpy(
        cache.layer(0).spans.base_offsets.ptr,
        host_array_ptr(global_offsets),
        global_offsets.nbytes,
        HipMemcpyKind.HOST_TO_DEVICE,
    )
    runtime.memcpy(
        cache.layer(1).spans.base_offsets.ptr,
        host_array_ptr(swa_offsets),
        swa_offsets.nbytes,
        HipMemcpyKind.HOST_TO_DEVICE,
    )
    rng = np.random.default_rng(290)
    total_tokens = 1026
    keys = rng.normal(0.0, 0.12, size=(total_tokens, 8, 128)).astype(np.float32)
    values = rng.normal(0.0, 0.12, size=(total_tokens, 8, 128)).astype(np.float32)
    keys_bf16 = bf16_to_float32(float_array_to_bf16_bits(keys))
    values_bf16 = bf16_to_float32(float_array_to_bf16_bits(values))
    selected = {510, 511, 512, 513, 1024, 1025}
    allocations = []
    evicted_position = 700
    try:
        key_device = malloc(keys[0].nbytes, runtime=runtime)
        value_device = malloc(values[0].nbytes, runtime=runtime)
        query_device = malloc(72 * 128 * 4, runtime=runtime)
        output_device = malloc(72 * 128 * 4, runtime=runtime)
        token4_output_device = malloc(72 * 128 * 4, runtime=runtime)
        split_output_device = malloc(72 * 128 * 4, runtime=runtime)
        tile16_output_device = malloc(72 * 128 * 4, runtime=runtime)
        split_scratch_elements = max(48 * 1026, 72 * 512)
        score_scratch_device = malloc(split_scratch_elements * 4, runtime=runtime)
        physical_scratch_device = malloc(split_scratch_elements * 4, runtime=runtime)
        allocations.extend(
            (
                key_device,
                value_device,
                query_device,
                output_device,
                token4_output_device,
                split_output_device,
                tile16_output_device,
                score_scratch_device,
                physical_scratch_device,
            )
        )

        for position in range(total_tokens):
            key_row = np.ascontiguousarray(keys[position])
            value_row = np.ascontiguousarray(values[position])
            copy_host_to_device(
                key_device,
                host_array_ptr(key_row),
                runtime=runtime,
            )
            copy_host_to_device(
                value_device,
                host_array_ptr(value_row),
                runtime=runtime,
            )
            cache.prepare_position(position)
            cache.append(
                0,
                key_device.ptr,
                value_device.ptr,
                library=kv_library,
            )
            cache.append(
                1,
                key_device.ptr,
                value_device.ptr,
                library=kv_library,
            )

            if position not in selected:
                continue
            query = rng.normal(0.0, 0.12, size=(72, 128)).astype(np.float32)
            copy_host_to_device(
                query_device,
                host_array_ptr(query),
                runtime=runtime,
            )
            if position == 1025:
                cache.evict_swa_position(1, evicted_position)
            cache.attend(
                1,
                query_device.ptr,
                output_device.ptr,
                library=kv_library,
            )
            swa_state = cache.layer(1)
            laguna_swa_attention_decode_token4_exact_bf16_spans(
                query_device.ptr,
                swa_state.key_cache.ptr,
                swa_state.value_cache.ptr,
                token4_output_device.ptr,
                swa_state.spans,
                swa_state.q_heads,
                8,
                128,
                128**-0.5,
                sliding_window=512,
                library=kv_library,
                runtime=runtime,
            )
            laguna_swa_attention_decode_split_exact_bf16_spans(
                query_device.ptr,
                swa_state.key_cache.ptr,
                swa_state.value_cache.ptr,
                split_output_device.ptr,
                score_scratch_device.ptr,
                physical_scratch_device.ptr,
                swa_state.spans,
                min(position + 1, 512),
                swa_state.q_heads,
                8,
                128,
                128**-0.5,
                sliding_window=512,
                library=kv_library,
                runtime=runtime,
            )
            laguna_swa_attention_decode_split_tile16_exact_bf16_spans(
                query_device.ptr,
                swa_state.key_cache.ptr,
                swa_state.value_cache.ptr,
                tile16_output_device.ptr,
                score_scratch_device.ptr,
                physical_scratch_device.ptr,
                swa_state.spans,
                min(position + 1, 512),
                swa_state.q_heads,
                8,
                128,
                128**-0.5,
                sliding_window=512,
                library=kv_library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            actual = np.empty((72, 128), dtype=np.float32)
            token4_actual = np.empty_like(actual)
            split_actual = np.empty_like(actual)
            tile16_actual = np.empty_like(actual)
            copy_device_to_host(
                host_array_ptr(actual),
                output_device,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(token4_actual),
                token4_output_device,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(split_actual),
                split_output_device,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(tile16_actual),
                tile16_output_device,
                runtime=runtime,
            )
            np.testing.assert_array_equal(token4_actual, actual)
            np.testing.assert_array_equal(split_actual, token4_actual)
            np.testing.assert_array_equal(tile16_actual, split_actual)
            visible = np.arange(max(0, position - 511), position + 1)
            if position == 1025:
                visible = visible[visible != evicted_position]
            expected = _attention_reference(
                query,
                keys_bf16[visible],
                values_bf16[visible],
                num_kv_heads=8,
            )
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)

            if position == 513:
                global_evicted_position = 100
                cache.evict_position(0, global_evicted_position)
                global_query = np.ascontiguousarray(query[:48])
                copy_host_to_device(
                    query_device,
                    host_array_ptr(global_query),
                    nbytes=global_query.nbytes,
                    runtime=runtime,
                )
                cache.attend(
                    0,
                    query_device.ptr,
                    output_device.ptr,
                    library=kv_library,
                )
                global_state = cache.layer(0)
                laguna_global_attention_decode_split_exact_bf16_spans(
                    query_device.ptr,
                    global_state.key_cache.ptr,
                    global_state.value_cache.ptr,
                    split_output_device.ptr,
                    score_scratch_device.ptr,
                    physical_scratch_device.ptr,
                    global_state.spans,
                    position + 1,
                    global_state.capacity,
                    global_state.q_heads,
                    8,
                    128,
                    128**-0.5,
                    library=kv_library,
                    runtime=runtime,
                )
                runtime.device_synchronize()
                global_actual = np.empty((48, 128), dtype=np.float32)
                global_split_actual = np.empty_like(global_actual)
                copy_device_to_host(
                    host_array_ptr(global_actual),
                    output_device,
                    nbytes=global_actual.nbytes,
                    runtime=runtime,
                )
                copy_device_to_host(
                    host_array_ptr(global_split_actual),
                    split_output_device,
                    nbytes=global_split_actual.nbytes,
                    runtime=runtime,
                )
                np.testing.assert_array_equal(global_split_actual, global_actual)
                global_visible = np.arange(position + 1)
                global_visible = global_visible[global_visible != global_evicted_position]
                global_expected = _attention_reference(
                    global_query,
                    keys_bf16[global_visible],
                    values_bf16[global_visible],
                    num_kv_heads=8,
                )
                np.testing.assert_allclose(
                    global_actual,
                    global_expected,
                    rtol=3e-4,
                    atol=3e-4,
                )

        token_positions = np.empty(512, dtype=np.int64)
        mask = np.empty(512, dtype=np.bool_)
        swa_spans = cache.layer(1).spans
        runtime.memcpy(
            host_array_ptr(token_positions),
            swa_spans.token_positions.ptr,
            token_positions.nbytes,
            HipMemcpyKind.DEVICE_TO_HOST,
        )
        runtime.memcpy(
            host_array_ptr(mask),
            swa_spans.evict_mask.ptr,
            mask.nbytes,
            HipMemcpyKind.DEVICE_TO_HOST,
        )
        assert token_positions[0] == 1024
        assert token_positions[1] == 1025
        assert token_positions[evicted_position % 512] == evicted_position
        assert mask[evicted_position % 512]

        physical_slot = int(swa_offsets[1025 % 512])
        physical_key = np.empty((8, 128), dtype=np.uint16)
        runtime.memcpy(
            host_array_ptr(physical_key),
            cache.layer(1).key_cache.ptr + physical_slot * physical_key.nbytes,
            physical_key.nbytes,
            HipMemcpyKind.DEVICE_TO_HOST,
        )
        np.testing.assert_array_equal(
            physical_key,
            float_array_to_bf16_bits(keys[1025]),
        )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        cache.free()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_preappend_cached_qrow4_matches_current_source_qrow4() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        build_laguna_kv_attention,
        laguna_global_attention_prefill_qrow4_cached_meta_online_bf16_spans,
        laguna_global_attention_prefill_qrow4_cached_online_bf16_spans,
        laguna_global_attention_prefill_qrow4_dense_initial_online_bf16_spans,
        laguna_global_attention_prefill_qrow4_online_bf16_spans,
        laguna_global_attention_prefill_qrow6_cached_meta_online_bf16_spans,
        laguna_global_attention_prefill_qrow6_dense_initial_online_bf16_spans,
        laguna_swa_attention_prefill_qrow4_cached_meta_online_bf16_spans,
        laguna_swa_attention_prefill_qrow4_dense_initial_online_bf16_spans,
        laguna_swa_attention_prefill_qrow4_cached_online_bf16_spans,
        laguna_swa_attention_prefill_qrow4_sourcequal_online_bf16_spans,
    )
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    config = SimpleNamespace(
        block_count=2,
        layer_types=(FULL_ATTENTION, SLIDING_ATTENTION),
        head_counts=(48, 72),
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )
    baseline = allocate_laguna_kv_cache(
        config,
        context_length=512,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    cached = allocate_laguna_kv_cache(
        config,
        context_length=512,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    rows = 128
    rng = np.random.default_rng(20260726)
    keys = rng.normal(0.0, 0.12, size=(rows, 8, 128)).astype(np.float32)
    values = rng.normal(0.0, 0.12, size=(rows, 8, 128)).astype(np.float32)
    query_global = rng.normal(0.0, 0.12, size=(rows, 48, 128)).astype(np.float32)
    query_swa = rng.normal(0.0, 0.12, size=(rows, 72, 128)).astype(np.float32)
    allocations = []
    try:
        key_rows = malloc(keys.nbytes, runtime=runtime)
        value_rows = malloc(values.nbytes, runtime=runtime)
        global_query_rows = malloc(query_global.nbytes, runtime=runtime)
        swa_query_rows = malloc(query_swa.nbytes, runtime=runtime)
        global_baseline_out = malloc(query_global.nbytes, runtime=runtime)
        global_cached_out = malloc(query_global.nbytes, runtime=runtime)
        global_cached_meta_out = malloc(query_global.nbytes, runtime=runtime)
        global_qrow6_cached_meta_out = malloc(query_global.nbytes, runtime=runtime)
        global_dense_initial_out = malloc(query_global.nbytes, runtime=runtime)
        global_qrow6_dense_initial_out = malloc(
            query_global.nbytes, runtime=runtime
        )
        swa_baseline_out = malloc(query_swa.nbytes, runtime=runtime)
        swa_cached_out = malloc(query_swa.nbytes, runtime=runtime)
        swa_cached_meta_out = malloc(query_swa.nbytes, runtime=runtime)
        swa_dense_initial_out = malloc(query_swa.nbytes, runtime=runtime)
        allocations.extend(
            (
                key_rows,
                value_rows,
                global_query_rows,
                swa_query_rows,
                global_baseline_out,
                global_cached_out,
                global_cached_meta_out,
                global_qrow6_cached_meta_out,
                global_dense_initial_out,
                global_qrow6_dense_initial_out,
                swa_baseline_out,
                swa_cached_out,
                swa_cached_meta_out,
                swa_dense_initial_out,
            )
        )
        for buffer, array in (
            (key_rows, keys),
            (value_rows, values),
            (global_query_rows, query_global),
            (swa_query_rows, query_swa),
        ):
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes, runtime=runtime)

        positions = tuple(range(rows))
        baseline.prepare_rows(positions)
        global_layer = baseline.layer(0)
        laguna_global_attention_prefill_qrow4_online_bf16_spans(
            global_query_rows.ptr,
            key_rows.ptr,
            value_rows.ptr,
            global_layer.key_cache.ptr,
            global_layer.value_cache.ptr,
            global_baseline_out.ptr,
            global_layer.spans,
            rows,
            global_layer.capacity,
            global_layer.q_heads,
            config.head_count_kv,
            config.key_length,
            config.key_length**-0.5,
            library=library,
            runtime=runtime,
        )
        swa_layer = baseline.layer(1)
        laguna_swa_attention_prefill_qrow4_sourcequal_online_bf16_spans(
            swa_query_rows.ptr,
            key_rows.ptr,
            value_rows.ptr,
            swa_layer.key_cache.ptr,
            swa_layer.value_cache.ptr,
            swa_baseline_out.ptr,
            swa_layer.spans,
            rows,
            swa_layer.q_heads,
            config.head_count_kv,
            config.key_length,
            config.key_length**-0.5,
            sliding_window=config.sliding_window,
            start_position=0,
            library=library,
            runtime=runtime,
        )

        cached.prepare_rows(positions)
        for layer_id in range(2):
            cached.append_rows(
                layer_id,
                key_rows.ptr,
                value_rows.ptr,
                rows,
                library=library,
            )
        global_layer = cached.layer(0)
        laguna_global_attention_prefill_qrow4_cached_online_bf16_spans(
            global_query_rows.ptr,
            key_rows.ptr,
            value_rows.ptr,
            global_layer.key_cache.ptr,
            global_layer.value_cache.ptr,
            global_cached_out.ptr,
            global_layer.spans,
            rows,
            global_layer.capacity,
            global_layer.q_heads,
            config.head_count_kv,
            config.key_length,
            config.key_length**-0.5,
            library=library,
            runtime=runtime,
        )
        swa_layer = cached.layer(1)
        laguna_swa_attention_prefill_qrow4_cached_online_bf16_spans(
            swa_query_rows.ptr,
            key_rows.ptr,
            value_rows.ptr,
            swa_layer.key_cache.ptr,
            swa_layer.value_cache.ptr,
            swa_cached_out.ptr,
            swa_layer.spans,
            rows,
            swa_layer.q_heads,
            config.head_count_kv,
            config.key_length,
            config.key_length**-0.5,
            sliding_window=config.sliding_window,
            start_position=0,
            library=library,
            runtime=runtime,
        )
        laguna_global_attention_prefill_qrow4_cached_meta_online_bf16_spans(
            global_query_rows.ptr,
            key_rows.ptr,
            value_rows.ptr,
            global_layer.key_cache.ptr,
            global_layer.value_cache.ptr,
            global_cached_meta_out.ptr,
            global_layer.spans,
            rows,
            global_layer.capacity,
            global_layer.q_heads,
            config.head_count_kv,
            config.key_length,
            config.key_length**-0.5,
            library=library,
            runtime=runtime,
        )
        laguna_swa_attention_prefill_qrow4_cached_meta_online_bf16_spans(
            swa_query_rows.ptr,
            key_rows.ptr,
            value_rows.ptr,
            swa_layer.key_cache.ptr,
            swa_layer.value_cache.ptr,
            swa_cached_meta_out.ptr,
            swa_layer.spans,
            rows,
            swa_layer.q_heads,
            config.head_count_kv,
            config.key_length,
            config.key_length**-0.5,
            sliding_window=config.sliding_window,
            start_position=0,
            library=library,
            runtime=runtime,
        )
        laguna_swa_attention_prefill_qrow4_dense_initial_online_bf16_spans(
            swa_query_rows.ptr,
            key_rows.ptr,
            value_rows.ptr,
            swa_layer.key_cache.ptr,
            swa_layer.value_cache.ptr,
            swa_dense_initial_out.ptr,
            swa_layer.spans,
            rows,
            swa_layer.q_heads,
            config.head_count_kv,
            config.key_length,
            config.key_length**-0.5,
            sliding_window=config.sliding_window,
            start_position=0,
            library=library,
            runtime=runtime,
        )
        laguna_global_attention_prefill_qrow6_cached_meta_online_bf16_spans(
            global_query_rows.ptr,
            key_rows.ptr,
            value_rows.ptr,
            global_layer.key_cache.ptr,
            global_layer.value_cache.ptr,
            global_qrow6_cached_meta_out.ptr,
            global_layer.spans,
            rows,
            global_layer.capacity,
            global_layer.q_heads,
            config.head_count_kv,
            config.key_length,
            config.key_length**-0.5,
            library=library,
            runtime=runtime,
        )
        laguna_global_attention_prefill_qrow4_dense_initial_online_bf16_spans(
            global_query_rows.ptr,
            key_rows.ptr,
            value_rows.ptr,
            global_layer.key_cache.ptr,
            global_layer.value_cache.ptr,
            global_dense_initial_out.ptr,
            global_layer.spans,
            rows,
            global_layer.capacity,
            global_layer.q_heads,
            config.head_count_kv,
            config.key_length,
            config.key_length**-0.5,
            start_position=0,
            library=library,
            runtime=runtime,
        )
        laguna_global_attention_prefill_qrow6_dense_initial_online_bf16_spans(
            global_query_rows.ptr,
            key_rows.ptr,
            value_rows.ptr,
            global_layer.key_cache.ptr,
            global_layer.value_cache.ptr,
            global_qrow6_dense_initial_out.ptr,
            global_layer.spans,
            rows,
            global_layer.capacity,
            global_layer.q_heads,
            config.head_count_kv,
            config.key_length,
            config.key_length**-0.5,
            start_position=0,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()

        actual_global = np.empty_like(query_global)
        actual_global_meta = np.empty_like(query_global)
        actual_global_qrow6_meta = np.empty_like(query_global)
        actual_global_dense_initial = np.empty_like(query_global)
        actual_global_qrow6_dense_initial = np.empty_like(query_global)
        expected_global = np.empty_like(query_global)
        actual_swa = np.empty_like(query_swa)
        actual_swa_meta = np.empty_like(query_swa)
        actual_swa_dense_initial = np.empty_like(query_swa)
        expected_swa = np.empty_like(query_swa)
        for output, buffer in (
            (expected_global, global_baseline_out),
            (actual_global, global_cached_out),
            (actual_global_meta, global_cached_meta_out),
            (actual_global_qrow6_meta, global_qrow6_cached_meta_out),
            (actual_global_dense_initial, global_dense_initial_out),
            (
                actual_global_qrow6_dense_initial,
                global_qrow6_dense_initial_out,
            ),
            (expected_swa, swa_baseline_out),
            (actual_swa, swa_cached_out),
            (actual_swa_meta, swa_cached_meta_out),
            (actual_swa_dense_initial, swa_dense_initial_out),
        ):
            copy_device_to_host(
                host_array_ptr(output),
                buffer,
                output.nbytes,
                runtime=runtime,
            )
        np.testing.assert_array_equal(actual_global, expected_global)
        np.testing.assert_array_equal(actual_global_meta, expected_global)
        np.testing.assert_array_equal(actual_global_qrow6_meta, expected_global)
        np.testing.assert_array_equal(actual_global_dense_initial, expected_global)
        np.testing.assert_array_equal(
            actual_global_qrow6_dense_initial,
            expected_global,
        )
        np.testing.assert_array_equal(actual_swa, expected_swa)
        np.testing.assert_array_equal(actual_swa_meta, expected_swa)
        np.testing.assert_array_equal(actual_swa_dense_initial, expected_swa)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        cached.free()
        baseline.free()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_bulk_global_and_swa_prefill_match_serial_across_ring_wrap() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        build_laguna_kv_attention,
        laguna_global_attention_prefill_qrow2_online_bf16_spans,
        laguna_global_attention_prefill_qrow4_online_bf16_spans,
        laguna_swa_attention_prefill_qrow2_online_bf16_spans,
        laguna_swa_attention_prefill_qrow4_online_bf16_spans,
        laguna_swa_attention_prefill_qrow4_sourcequal_online_bf16_spans,
    )
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    config = SimpleNamespace(
        block_count=2,
        layer_types=(FULL_ATTENTION, SLIDING_ATTENTION),
        head_counts=(48, 72),
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )
    serial = allocate_laguna_kv_cache(
        config,
        context_length=520,
        backend="hip_gfx1151",
        runtime=runtime,
        global_prefill_variant="global_context_rows_spans",
    )
    bulk = allocate_laguna_kv_cache(
        config,
        context_length=520,
        backend="hip_gfx1151",
        runtime=runtime,
        global_prefill_variant="global_context_rows_spans",
        swa_prefill_variant="swa_context_rows_spans",
    )
    wave32 = allocate_laguna_kv_cache(
        config,
        context_length=520,
        backend="hip_gfx1151",
        runtime=runtime,
        global_prefill_variant="global_context_rows_spans",
        swa_prefill_variant="swa_context_rows_wave32_exact_spans",
    )
    qrow2 = allocate_laguna_kv_cache(
        config,
        context_length=520,
        backend="hip_gfx1151",
        runtime=runtime,
        global_prefill_variant="global_context_rows_spans",
        swa_prefill_variant="swa_context_rows_qrow2_exact_spans",
    )
    rng = np.random.default_rng(1207)
    seed_rows = 508
    rows = 8
    keys = rng.normal(0.0, 0.12, size=(seed_rows + rows, 8, 128)).astype(np.float32)
    values = rng.normal(0.0, 0.12, size=(seed_rows + rows, 8, 128)).astype(np.float32)
    query_global = rng.normal(0.0, 0.12, size=(rows, 48, 128)).astype(np.float32)
    query_swa = rng.normal(0.0, 0.12, size=(rows, 72, 128)).astype(np.float32)
    allocations = []
    try:
        key_rows = malloc(keys.nbytes, runtime=runtime)
        value_rows = malloc(values.nbytes, runtime=runtime)
        global_query_rows = malloc(query_global.nbytes, runtime=runtime)
        swa_query_rows = malloc(query_swa.nbytes, runtime=runtime)
        global_bulk_out = malloc(query_global.nbytes, runtime=runtime)
        global_online_out = malloc(query_global.nbytes, runtime=runtime)
        global_online_odd_out = malloc(query_global.nbytes, runtime=runtime)
        global_qrow4_online_out = malloc(query_global.nbytes, runtime=runtime)
        global_qrow4_online_odd_out = malloc(query_global.nbytes, runtime=runtime)
        swa_bulk_out = malloc(query_swa.nbytes, runtime=runtime)
        swa_wave32_out = malloc(query_swa.nbytes, runtime=runtime)
        swa_qrow2_out = malloc(query_swa.nbytes, runtime=runtime)
        swa_qrow2_online_out = malloc(query_swa.nbytes, runtime=runtime)
        swa_qrow4_online_out = malloc(query_swa.nbytes, runtime=runtime)
        swa_qrow4_online_odd_out = malloc(query_swa.nbytes, runtime=runtime)
        swa_qrow4_sourcequal_online_out = malloc(query_swa.nbytes, runtime=runtime)
        swa_qrow4_sourcequal_online_odd_out = malloc(query_swa.nbytes, runtime=runtime)
        global_serial_out = malloc(query_global[0].nbytes, runtime=runtime)
        swa_serial_out = malloc(query_swa[0].nbytes, runtime=runtime)
        allocations.extend(
            (
                key_rows,
                value_rows,
                global_query_rows,
                swa_query_rows,
                global_bulk_out,
                global_online_out,
                global_online_odd_out,
                global_qrow4_online_out,
                global_qrow4_online_odd_out,
                swa_bulk_out,
                swa_wave32_out,
                swa_qrow2_out,
                swa_qrow2_online_out,
                swa_qrow4_online_out,
                swa_qrow4_online_odd_out,
                swa_qrow4_sourcequal_online_out,
                swa_qrow4_sourcequal_online_odd_out,
                global_serial_out,
                swa_serial_out,
            )
        )
        for buffer, array in (
            (key_rows, keys),
            (value_rows, values),
            (global_query_rows, query_global),
            (swa_query_rows, query_swa),
        ):
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes, runtime=runtime)

        # Populate both owners through the bulk write path, then compare one
        # wrap-crossing 508..515 chunk against the established token-serial path.
        seed_positions = tuple(range(seed_rows))
        for cache in (serial, bulk, wave32, qrow2):
            cache.prepare_rows(seed_positions)
            for layer_id in range(2):
                cache.append_rows(
                    layer_id,
                    key_rows.ptr,
                    value_rows.ptr,
                    seed_rows,
                    library=library,
                )
            cache.commit_rows()
            cache.evict_position(0, 200)
            cache.evict_swa_position(1, 200)

        row_bytes = 8 * 128 * np.dtype(np.float32).itemsize
        positions = tuple(range(seed_rows, seed_rows + rows))
        bulk.prepare_rows(positions)
        bulk.attend_prefill(
            0,
            global_query_rows.ptr,
            key_rows.ptr + seed_rows * row_bytes,
            value_rows.ptr + seed_rows * row_bytes,
            global_bulk_out.ptr,
            rows,
            library=library,
        )
        global_layer = bulk.layer(0)
        for online_rows, online_out in (
            (rows, global_online_out),
            (rows - 1, global_online_odd_out),
        ):
            laguna_global_attention_prefill_qrow2_online_bf16_spans(
                global_query_rows.ptr,
                key_rows.ptr + seed_rows * row_bytes,
                value_rows.ptr + seed_rows * row_bytes,
                global_layer.key_cache.ptr,
                global_layer.value_cache.ptr,
                online_out.ptr,
                global_layer.spans,
                online_rows,
                global_layer.capacity,
                global_layer.q_heads,
                config.head_count_kv,
                config.key_length,
                config.key_length**-0.5,
                library=library,
                runtime=runtime,
            )
        for online_rows, online_out in (
            (rows, global_qrow4_online_out),
            (rows - 1, global_qrow4_online_odd_out),
        ):
            laguna_global_attention_prefill_qrow4_online_bf16_spans(
                global_query_rows.ptr,
                key_rows.ptr + seed_rows * row_bytes,
                value_rows.ptr + seed_rows * row_bytes,
                global_layer.key_cache.ptr,
                global_layer.value_cache.ptr,
                online_out.ptr,
                global_layer.spans,
                online_rows,
                global_layer.capacity,
                global_layer.q_heads,
                config.head_count_kv,
                config.key_length,
                config.key_length**-0.5,
                library=library,
                runtime=runtime,
            )
        bulk.append_rows(
            0,
            key_rows.ptr + seed_rows * row_bytes,
            value_rows.ptr + seed_rows * row_bytes,
            rows,
            library=library,
        )
        bulk.attend_prefill(
            1,
            swa_query_rows.ptr,
            key_rows.ptr + seed_rows * row_bytes,
            value_rows.ptr + seed_rows * row_bytes,
            swa_bulk_out.ptr,
            rows,
            library=library,
        )
        bulk.append_rows(
            1,
            key_rows.ptr + seed_rows * row_bytes,
            value_rows.ptr + seed_rows * row_bytes,
            rows,
            library=library,
        )
        bulk.commit_rows()

        wave32.prepare_rows(positions)
        wave32.attend_prefill(
            1,
            swa_query_rows.ptr,
            key_rows.ptr + seed_rows * row_bytes,
            value_rows.ptr + seed_rows * row_bytes,
            swa_wave32_out.ptr,
            rows,
            library=library,
        )
        wave32.append_rows(
            1,
            key_rows.ptr + seed_rows * row_bytes,
            value_rows.ptr + seed_rows * row_bytes,
            rows,
            library=library,
        )
        wave32.commit_rows()

        qrow2.prepare_rows(positions)
        qrow2.attend_prefill(
            1,
            swa_query_rows.ptr,
            key_rows.ptr + seed_rows * row_bytes,
            value_rows.ptr + seed_rows * row_bytes,
            swa_qrow2_out.ptr,
            rows,
            library=library,
        )
        swa_layer = qrow2.layer(1)
        laguna_swa_attention_prefill_qrow2_online_bf16_spans(
            swa_query_rows.ptr,
            key_rows.ptr + seed_rows * row_bytes,
            value_rows.ptr + seed_rows * row_bytes,
            swa_layer.key_cache.ptr,
            swa_layer.value_cache.ptr,
            swa_qrow2_online_out.ptr,
            swa_layer.spans,
            rows,
            swa_layer.q_heads,
            config.head_count_kv,
            config.key_length,
            config.key_length**-0.5,
            sliding_window=config.sliding_window,
            start_position=seed_rows,
            library=library,
            runtime=runtime,
        )
        for online_rows, online_out in (
            (rows, swa_qrow4_online_out),
            (rows - 1, swa_qrow4_online_odd_out),
        ):
            laguna_swa_attention_prefill_qrow4_online_bf16_spans(
                swa_query_rows.ptr,
                key_rows.ptr + seed_rows * row_bytes,
                value_rows.ptr + seed_rows * row_bytes,
                swa_layer.key_cache.ptr,
                swa_layer.value_cache.ptr,
                online_out.ptr,
                swa_layer.spans,
                online_rows,
                swa_layer.q_heads,
                config.head_count_kv,
                config.key_length,
                config.key_length**-0.5,
                sliding_window=config.sliding_window,
                start_position=seed_rows,
                library=library,
                runtime=runtime,
            )
        for online_rows, online_out in (
            (rows, swa_qrow4_sourcequal_online_out),
            (rows - 1, swa_qrow4_sourcequal_online_odd_out),
        ):
            laguna_swa_attention_prefill_qrow4_sourcequal_online_bf16_spans(
                swa_query_rows.ptr,
                key_rows.ptr + seed_rows * row_bytes,
                value_rows.ptr + seed_rows * row_bytes,
                swa_layer.key_cache.ptr,
                swa_layer.value_cache.ptr,
                online_out.ptr,
                swa_layer.spans,
                online_rows,
                swa_layer.q_heads,
                config.head_count_kv,
                config.key_length,
                config.key_length**-0.5,
                sliding_window=config.sliding_window,
                start_position=seed_rows,
                library=library,
                runtime=runtime,
            )
        qrow2.append_rows(
            1,
            key_rows.ptr + seed_rows * row_bytes,
            value_rows.ptr + seed_rows * row_bytes,
            rows,
            library=library,
        )
        qrow2.commit_rows()

        expected_global = np.empty_like(query_global)
        expected_swa = np.empty_like(query_swa)
        for row, position in enumerate(positions):
            serial.prepare_position(position)
            for layer_id, (query, output) in enumerate(
                (
                    (query_global[row], global_serial_out),
                    (query_swa[row], swa_serial_out),
                )
            ):
                serial.append(
                    layer_id,
                    key_rows.ptr + position * row_bytes,
                    value_rows.ptr + position * row_bytes,
                    library=library,
                )
                serial.attend(
                    layer_id,
                    (global_query_rows if layer_id == 0 else swa_query_rows).ptr
                    + row * query.nbytes,
                    output.ptr,
                    library=library,
                )
                runtime.device_synchronize()
                destination = expected_global[row] if layer_id == 0 else expected_swa[row]
                copy_device_to_host(
                    host_array_ptr(destination),
                    output,
                    destination.nbytes,
                    runtime=runtime,
                )

        actual_global = np.empty_like(query_global)
        actual_global_online = np.empty_like(query_global)
        actual_global_online_odd = np.empty_like(query_global[:-1])
        actual_global_qrow4_online = np.empty_like(query_global)
        actual_global_qrow4_online_odd = np.empty_like(query_global[:-1])
        actual_swa = np.empty_like(query_swa)
        actual_swa_wave32 = np.empty_like(query_swa)
        actual_swa_qrow2 = np.empty_like(query_swa)
        actual_swa_qrow2_online = np.empty_like(query_swa)
        actual_swa_qrow4_online = np.empty_like(query_swa)
        actual_swa_qrow4_online_odd = np.empty_like(query_swa[:-1])
        actual_swa_qrow4_sourcequal_online = np.empty_like(query_swa)
        actual_swa_qrow4_sourcequal_online_odd = np.empty_like(query_swa[:-1])
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(actual_global),
            global_bulk_out,
            actual_global.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(actual_global_online),
            global_online_out,
            actual_global_online.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(actual_global_online_odd),
            global_online_odd_out,
            actual_global_online_odd.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(actual_global_qrow4_online),
            global_qrow4_online_out,
            actual_global_qrow4_online.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(actual_global_qrow4_online_odd),
            global_qrow4_online_odd_out,
            actual_global_qrow4_online_odd.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(actual_swa),
            swa_bulk_out,
            actual_swa.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(actual_swa_wave32),
            swa_wave32_out,
            actual_swa_wave32.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(actual_swa_qrow2),
            swa_qrow2_out,
            actual_swa_qrow2.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(actual_swa_qrow2_online),
            swa_qrow2_online_out,
            actual_swa_qrow2_online.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(actual_swa_qrow4_online),
            swa_qrow4_online_out,
            actual_swa_qrow4_online.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(actual_swa_qrow4_online_odd),
            swa_qrow4_online_odd_out,
            actual_swa_qrow4_online_odd.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(actual_swa_qrow4_sourcequal_online),
            swa_qrow4_sourcequal_online_out,
            actual_swa_qrow4_sourcequal_online.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(actual_swa_qrow4_sourcequal_online_odd),
            swa_qrow4_sourcequal_online_odd_out,
            actual_swa_qrow4_sourcequal_online_odd.nbytes,
            runtime=runtime,
        )
        np.testing.assert_array_equal(actual_global, expected_global)
        np.testing.assert_allclose(actual_global_online, actual_global, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(
            actual_global_online_odd,
            actual_global[:-1],
            rtol=2e-5,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            actual_global_qrow4_online,
            actual_global,
            rtol=2e-5,
            atol=2e-6,
        )
        np.testing.assert_array_equal(
            actual_global_qrow4_online,
            actual_global_online,
        )
        np.testing.assert_allclose(
            actual_global_qrow4_online_odd,
            actual_global[:-1],
            rtol=2e-5,
            atol=2e-6,
        )
        np.testing.assert_array_equal(actual_swa, expected_swa)
        np.testing.assert_array_equal(actual_swa_wave32, actual_swa)
        np.testing.assert_array_equal(actual_swa_qrow2, actual_swa_wave32)
        np.testing.assert_allclose(
            actual_swa_qrow2_online,
            actual_swa_wave32,
            rtol=2e-5,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            actual_swa_qrow4_online,
            actual_swa_wave32,
            rtol=2e-5,
            atol=2e-6,
        )
        np.testing.assert_array_equal(
            actual_swa_qrow4_online,
            actual_swa_qrow2_online,
        )
        np.testing.assert_array_equal(
            actual_swa_qrow4_sourcequal_online,
            actual_swa_qrow4_online,
        )
        np.testing.assert_allclose(
            actual_swa_qrow4_online_odd,
            actual_swa_wave32[:-1],
            rtol=2e-5,
            atol=2e-6,
        )
        np.testing.assert_array_equal(
            actual_swa_qrow4_sourcequal_online_odd,
            actual_swa_qrow4_online_odd,
        )
        assert bulk.position == serial.position == wave32.position == qrow2.position == 515

        for layer_id in range(2):
            bulk_state = bulk.layer(layer_id)
            serial_state = serial.layer(layer_id)
            capacity = bulk_state.capacity
            for field, dtype in (("token_positions", np.int64), ("evict_mask", np.bool_)):
                bulk_values = np.empty(capacity, dtype=dtype)
                serial_values = np.empty(capacity, dtype=dtype)
                bulk_tensor = getattr(bulk_state.spans, field)
                serial_tensor = getattr(serial_state.spans, field)
                runtime.memcpy(
                    host_array_ptr(bulk_values),
                    bulk_tensor.ptr,
                    bulk_values.nbytes,
                    HipMemcpyKind.DEVICE_TO_HOST,
                )
                runtime.memcpy(
                    host_array_ptr(serial_values),
                    serial_tensor.ptr,
                    serial_values.nbytes,
                    HipMemcpyKind.DEVICE_TO_HOST,
                )
                np.testing.assert_array_equal(bulk_values, serial_values)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        qrow2.free()
        wave32.free()
        bulk.free()
        serial.free()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_swa_resident_attention_slices_match_chunks_across_ring_wrap() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        build_laguna_kv_attention,
    )
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    config = SimpleNamespace(
        block_count=1,
        layer_types=(SLIDING_ATTENTION,),
        head_counts=(72,),
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )
    baseline = allocate_laguna_kv_cache(
        config,
        context_length=1_024,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    sliced = allocate_laguna_kv_cache(
        config,
        context_length=1_024,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    rng = np.random.default_rng(1513)
    seed_rows = 384
    rows = 640
    keys = rng.normal(0.0, 0.12, size=(seed_rows + rows, 8, 128)).astype(np.float32)
    values = rng.normal(0.0, 0.12, size=(seed_rows + rows, 8, 128)).astype(np.float32)
    queries = rng.normal(0.0, 0.12, size=(rows, 72, 128)).astype(np.float32)
    positions = np.arange(seed_rows, seed_rows + rows, dtype=np.int64)
    allocations = []
    try:
        key_rows = malloc(keys.nbytes, runtime=runtime)
        value_rows = malloc(values.nbytes, runtime=runtime)
        query_rows = malloc(queries.nbytes, runtime=runtime)
        baseline_out = malloc(queries.nbytes, runtime=runtime)
        sliced_out = malloc(queries.nbytes, runtime=runtime)
        position_rows = malloc(positions.nbytes, runtime=runtime)
        allocations.extend(
            (key_rows, value_rows, query_rows, baseline_out, sliced_out, position_rows)
        )
        for buffer, array in (
            (key_rows, keys),
            (value_rows, values),
            (query_rows, queries),
            (position_rows, positions),
        ):
            copy_host_to_device(buffer, host_array_ptr(array), array.nbytes, runtime=runtime)

        row_bytes = 8 * 128 * np.dtype(np.float32).itemsize
        query_row_bytes = 72 * 128 * np.dtype(np.float32).itemsize
        seed_positions = tuple(range(seed_rows))
        for cache in (baseline, sliced):
            cache.prepare_rows(seed_positions)
            cache.append_rows(0, key_rows.ptr, value_rows.ptr, seed_rows, library=library)
            cache.commit_rows()

        attention_slices = tuple(
            (offset, min(128, rows - offset))
            for offset in range(0, rows, 128)
        )
        for offset, count in attention_slices:
            chunk_positions = tuple(int(value) for value in positions[offset : offset + count])
            baseline.prepare_rows(chunk_positions)
            baseline.attend_prefill(
                0,
                query_rows.ptr + offset * query_row_bytes,
                key_rows.ptr + (seed_rows + offset) * row_bytes,
                value_rows.ptr + (seed_rows + offset) * row_bytes,
                baseline_out.ptr + offset * query_row_bytes,
                count,
                library=library,
            )
            baseline.append_rows(
                0,
                key_rows.ptr + (seed_rows + offset) * row_bytes,
                value_rows.ptr + (seed_rows + offset) * row_bytes,
                count,
                library=library,
            )
            baseline.commit_rows()

        sliced.prepare_rows(tuple(int(value) for value in positions))
        for offset, count in attention_slices:
            slice_position_ptr = position_rows.ptr + offset * DType.INT64.itemsize
            sliced.attend_prefill(
                0,
                query_rows.ptr + offset * query_row_bytes,
                key_rows.ptr + (seed_rows + offset) * row_bytes,
                value_rows.ptr + (seed_rows + offset) * row_bytes,
                sliced_out.ptr + offset * query_row_bytes,
                count,
                row_offset=offset,
                row_positions_ptr=slice_position_ptr,
                library=library,
            )
            sliced.append_rows(
                0,
                key_rows.ptr + (seed_rows + offset) * row_bytes,
                value_rows.ptr + (seed_rows + offset) * row_bytes,
                count,
                row_offset=offset,
                row_positions_ptr=slice_position_ptr,
                library=library,
            )
        sliced.commit_rows()
        runtime.device_synchronize()

        baseline_context = np.empty_like(queries)
        sliced_context = np.empty_like(queries)
        copy_device_to_host(
            host_array_ptr(baseline_context),
            baseline_out,
            baseline_context.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(sliced_context),
            sliced_out,
            sliced_context.nbytes,
            runtime=runtime,
        )
        np.testing.assert_array_equal(sliced_context, baseline_context)
        assert baseline.position == sliced.position == 1_023

        baseline_state = baseline.layer(0)
        sliced_state = sliced.layer(0)
        for baseline_buffer, sliced_buffer, dtype in (
            (baseline_state.key_cache, sliced_state.key_cache, np.uint16),
            (baseline_state.value_cache, sliced_state.value_cache, np.uint16),
        ):
            baseline_values = np.empty(baseline_buffer.nbytes // np.dtype(dtype).itemsize, dtype=dtype)
            sliced_values = np.empty_like(baseline_values)
            copy_device_to_host(
                host_array_ptr(baseline_values),
                baseline_buffer,
                baseline_values.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(sliced_values),
                sliced_buffer,
                sliced_values.nbytes,
                runtime=runtime,
            )
            np.testing.assert_array_equal(sliced_values, baseline_values)
        for field, dtype in (
            ("live_counts", np.int64),
            ("token_positions", np.int64),
            ("evict_mask", np.bool_),
        ):
            baseline_tensor = getattr(baseline_state.spans, field)
            sliced_tensor = getattr(sliced_state.spans, field)
            baseline_values = np.empty(baseline_tensor.numel, dtype=dtype)
            sliced_values = np.empty_like(baseline_values)
            runtime.memcpy(
                host_array_ptr(baseline_values),
                baseline_tensor.ptr,
                baseline_values.nbytes,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
            runtime.memcpy(
                host_array_ptr(sliced_values),
                sliced_tensor.ptr,
                sliced_values.nbytes,
                HipMemcpyKind.DEVICE_TO_HOST,
            )
            np.testing.assert_array_equal(sliced_values, baseline_values)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        sliced.free()
        baseline.free()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_dense_initial_blas_helpers_match_cpu() -> None:
    """Cache widening and causal softmax must honor the complete span ABI."""

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        build_laguna_kv_attention,
        laguna_dense_initial_cache_bf16_to_f32_spans,
        laguna_dense_initial_causal_softmax_f32_spans,
        laguna_dense_initial_causal_softmax_wave_rows_f32_spans,
        laguna_swa_attention_prefill_qrow4_dense_initial_online_bf16_spans,
    )
    from hipengine.loading.materialize import float_array_to_bf16_bits
    from hipengine.quant.gguf import bf16_to_float32
    from hipengine.runtime.laguna_attention_hipblaslt import (
        LagunaAttentionHipblasLt,
    )
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    context = 256
    rows = 128
    query_heads = 72
    kv_heads = 8
    head_dim = 128
    start_position = context - rows
    config = SimpleNamespace(
        block_count=1,
        layer_types=(SLIDING_ATTENTION,),
        head_counts=(query_heads,),
        head_count_kv=kv_heads,
        key_length=head_dim,
        value_length=head_dim,
        sliding_window=512,
    )
    cache = allocate_laguna_kv_cache(
        config,
        context_length=512,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    state = cache.layer(0)
    rng = np.random.default_rng(20260727)
    key_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(context, kv_heads, head_dim)).astype(
            np.float32
        )
    )
    value_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(context, kv_heads, head_dim)).astype(
            np.float32
        )
    )
    scores = rng.normal(
        0.0,
        0.4,
        size=(query_heads, rows, context),
    ).astype(np.float32)
    key_f32 = np.empty(key_bits.shape, dtype=np.float32)
    value_f32 = np.empty(value_bits.shape, dtype=np.float32)
    actual_scores = np.empty_like(scores)
    wave_scores = np.empty_like(scores)
    expected_scores = np.zeros_like(scores)
    query = rng.normal(
        0.0,
        0.2,
        size=(rows, query_heads, head_dim),
    ).astype(np.float32)
    current_attention = np.empty_like(query)
    blas_attention = np.empty_like(query)
    packed_attention = np.empty_like(query)
    wave_attention = np.empty_like(query)
    blocked_attention = np.empty_like(query)
    scale = 0.5
    for head in range(query_heads):
        for row in range(rows):
            visible = start_position + row + 1
            logits = scores[head, row, :visible] * scale
            logits -= np.max(logits)
            weights = np.exp(logits)
            expected_scores[head, row, :visible] = weights / np.sum(weights)

    route = LagunaAttentionHipblasLt(runtime=runtime)
    packed_route = LagunaAttentionHipblasLt(
        runtime=runtime,
        packed_queries=True,
    )
    wave_route = LagunaAttentionHipblasLt(
        runtime=runtime,
        packed_queries=True,
        wave_rows_softmax=True,
    )
    blocked_route = LagunaAttentionHipblasLt(
        runtime=runtime,
        packed_queries=True,
        wave_rows_softmax=True,
        block_context=128,
    )
    allocations = []
    try:
        key_out = malloc(key_f32.nbytes, runtime=runtime)
        value_out = malloc(value_f32.nbytes, runtime=runtime)
        score_device = malloc(scores.nbytes, runtime=runtime)
        wave_score_device = malloc(scores.nbytes, runtime=runtime)
        query_device = malloc(query.nbytes, runtime=runtime)
        current_out = malloc(current_attention.nbytes, runtime=runtime)
        blas_out = malloc(blas_attention.nbytes, runtime=runtime)
        packed_out = malloc(packed_attention.nbytes, runtime=runtime)
        wave_out = malloc(wave_attention.nbytes, runtime=runtime)
        blocked_out = malloc(blocked_attention.nbytes, runtime=runtime)
        allocations.extend(
            (
                key_out,
                value_out,
                score_device,
                wave_score_device,
                query_device,
                current_out,
                blas_out,
                packed_out,
                wave_out,
                blocked_out,
            )
        )
        copy_host_to_device(
            state.key_cache,
            host_array_ptr(key_bits),
            key_bits.nbytes,
            runtime=runtime,
        )
        copy_host_to_device(
            state.value_cache,
            host_array_ptr(value_bits),
            value_bits.nbytes,
            runtime=runtime,
        )
        copy_host_to_device(
            score_device,
            host_array_ptr(scores),
            scores.nbytes,
            runtime=runtime,
        )
        copy_host_to_device(
            wave_score_device,
            host_array_ptr(scores),
            scores.nbytes,
            runtime=runtime,
        )
        copy_host_to_device(
            query_device,
            host_array_ptr(query),
            query.nbytes,
            runtime=runtime,
        )
        live_count = np.asarray([context], dtype=np.int64)
        token_positions = np.full(512, -1, dtype=np.int64)
        token_positions[:context] = np.arange(context, dtype=np.int64)
        evict_mask = np.ones(512, dtype=np.bool_)
        evict_mask[:context] = False
        row_position = np.asarray([start_position], dtype=np.int64)
        for tensor, host in (
            (state.spans.live_counts, live_count),
            (state.spans.token_positions, token_positions),
            (state.spans.evict_mask, evict_mask),
            (state.spans.row_positions, row_position),
        ):
            runtime.memcpy(
                tensor.ptr,
                host_array_ptr(host),
                host.nbytes,
                HipMemcpyKind.HOST_TO_DEVICE,
            )
        laguna_dense_initial_cache_bf16_to_f32_spans(
            state.key_cache.ptr,
            state.value_cache.ptr,
            key_out.ptr,
            value_out.ptr,
            state.spans,
            context,
            kv_heads,
            head_dim,
            library=library,
            runtime=runtime,
        )
        laguna_dense_initial_causal_softmax_f32_spans(
            score_device.ptr,
            state.spans,
            rows,
            context,
            query_heads,
            start_position,
            scale,
            library=library,
            runtime=runtime,
        )
        laguna_dense_initial_causal_softmax_wave_rows_f32_spans(
            wave_score_device.ptr,
            state.spans,
            rows,
            context,
            query_heads,
            start_position,
            scale,
            library=library,
            runtime=runtime,
        )
        laguna_swa_attention_prefill_qrow4_dense_initial_online_bf16_spans(
            query_device.ptr,
            key_out.ptr,
            value_out.ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
            current_out.ptr,
            state.spans,
            rows,
            query_heads,
            kv_heads,
            head_dim,
            head_dim**-0.5,
            sliding_window=512,
            start_position=start_position,
            library=library,
            runtime=runtime,
        )
        route.launch(
            query_device.ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
            blas_out.ptr,
            state.spans,
            rows=rows,
            start_position=start_position,
            num_q_heads=query_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            scale=head_dim**-0.5,
            kv_library=library,
        )
        packed_route.launch(
            query_device.ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
            packed_out.ptr,
            state.spans,
            rows=rows,
            start_position=start_position,
            num_q_heads=query_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            scale=head_dim**-0.5,
            kv_library=library,
        )
        wave_route.launch(
            query_device.ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
            wave_out.ptr,
            state.spans,
            rows=rows,
            start_position=start_position,
            num_q_heads=query_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            scale=head_dim**-0.5,
            kv_library=library,
        )
        blocked_route.launch(
            query_device.ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
            blocked_out.ptr,
            state.spans,
            rows=rows,
            start_position=start_position,
            num_q_heads=query_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            scale=head_dim**-0.5,
            kv_library=library,
        )
        runtime.device_synchronize()
        for host, device in (
            (key_f32, key_out),
            (value_f32, value_out),
            (actual_scores, score_device),
            (wave_scores, wave_score_device),
            (current_attention, current_out),
            (blas_attention, blas_out),
            (packed_attention, packed_out),
            (wave_attention, wave_out),
            (blocked_attention, blocked_out),
        ):
            copy_device_to_host(
                host_array_ptr(host),
                device,
                host.nbytes,
                runtime=runtime,
            )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        packed_route.close()
        wave_route.close()
        route.close()
        blocked_route.close()
        cache.free()

    np.testing.assert_array_equal(key_f32, bf16_to_float32(key_bits))
    np.testing.assert_array_equal(value_f32, bf16_to_float32(value_bits))
    np.testing.assert_allclose(actual_scores, expected_scores, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(wave_scores, expected_scores, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        blas_attention,
        current_attention,
        rtol=2e-3,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        packed_attention,
        current_attention,
        rtol=2e-3,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        wave_attention,
        current_attention,
        rtol=2e-3,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        blocked_attention,
        current_attention,
        rtol=2e-3,
        atol=2e-4,
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_dense_initial_contiguous_global_cache_block_matches_spans() -> None:
    """The dense contiguous global-cache lane must equal the full span ABI."""

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        build_laguna_kv_attention,
        laguna_dense_initial_cache_block_bf16_to_f32_spans,
        laguna_dense_initial_contiguous_cache_block_bf16_to_f32_spans,
    )
    from hipengine.loading.materialize import float_array_to_bf16_bits
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    context = 512
    logical_start = 256
    count = 256
    kv_heads = 8
    head_dim = 128
    config = SimpleNamespace(
        block_count=1,
        layer_types=(FULL_ATTENTION,),
        head_counts=(48,),
        head_count_kv=kv_heads,
        key_length=head_dim,
        value_length=head_dim,
        sliding_window=512,
    )
    cache = allocate_laguna_kv_cache(
        config,
        context_length=context,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    state = cache.layer(0)
    rng = np.random.default_rng(0x1C4)
    key_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.2, size=(context, kv_heads, head_dim)).astype(
            np.float32
        )
    )
    value_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.2, size=key_bits.shape).astype(np.float32)
    )
    output_shape = (count, kv_heads, head_dim)
    allocations = []
    try:
        output_nbytes = np.empty(output_shape, dtype=np.float32).nbytes
        generic_key = malloc(output_nbytes, runtime=runtime)
        generic_value = malloc(generic_key.nbytes, runtime=runtime)
        contiguous_key = malloc(generic_key.nbytes, runtime=runtime)
        contiguous_value = malloc(generic_key.nbytes, runtime=runtime)
        allocations.extend(
            (generic_key, generic_value, contiguous_key, contiguous_value)
        )
        copy_host_to_device(
            state.key_cache,
            host_array_ptr(key_bits),
            key_bits.nbytes,
            runtime=runtime,
        )
        copy_host_to_device(
            state.value_cache,
            host_array_ptr(value_bits),
            value_bits.nbytes,
            runtime=runtime,
        )
        live_count = np.asarray([context], dtype=np.int64)
        token_positions = np.arange(context, dtype=np.int64)
        evict_mask = np.zeros(context, dtype=np.bool_)
        for tensor, host in (
            (state.spans.live_counts, live_count),
            (state.spans.token_positions, token_positions),
            (state.spans.evict_mask, evict_mask),
        ):
            runtime.memcpy(
                tensor.ptr,
                host_array_ptr(host),
                host.nbytes,
                HipMemcpyKind.HOST_TO_DEVICE,
            )
        common = (
            state.key_cache.ptr,
            state.value_cache.ptr,
            state.spans,
            logical_start,
            count,
            context,
            kv_heads,
            head_dim,
        )
        laguna_dense_initial_cache_block_bf16_to_f32_spans(
            common[0],
            common[1],
            generic_key.ptr,
            generic_value.ptr,
            *common[2:],
            library=library,
            runtime=runtime,
        )
        laguna_dense_initial_contiguous_cache_block_bf16_to_f32_spans(
            common[0],
            common[1],
            contiguous_key.ptr,
            contiguous_value.ptr,
            *common[2:],
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        for generic, contiguous in (
            (generic_key, contiguous_key),
            (generic_value, contiguous_value),
        ):
            expected = np.empty(output_shape, dtype=np.float32)
            actual = np.empty_like(expected)
            copy_device_to_host(
                host_array_ptr(expected),
                generic,
                expected.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(actual),
                contiguous,
                actual.nbytes,
                runtime=runtime,
            )
            np.testing.assert_array_equal(actual, expected)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        cache.free()


def _attention_reference(
    query: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    num_kv_heads: int,
) -> np.ndarray:
    num_q_heads, head_dim = query.shape
    group = num_q_heads // num_kv_heads
    kv_for_q = np.arange(num_q_heads, dtype=np.int64) // group
    expanded_keys = keys[:, kv_for_q, :]
    expanded_values = values[:, kv_for_q, :]
    scores = np.einsum("hd,thd->ht", query, expanded_keys, dtype=np.float32)
    scores *= np.float32(head_dim**-0.5)
    scores -= np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores, dtype=np.float32)
    weights /= np.sum(weights, axis=1, keepdims=True, dtype=np.float32)
    return np.einsum("ht,thd->hd", weights, expanded_values, dtype=np.float32)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_swa_gqa3_vstage64_matches_cpu_after_wrap_and_eviction() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        build_laguna_kv_attention,
        laguna_swa_attention_decode_fused_exact_gated_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        laguna_swa_attention_decode_fused_exact_gated_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        laguna_swa_attention_decode_fused_exact_gated_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        laguna_swa_attention_decode_fused_exact_gated_mixed32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        laguna_swa_attention_decode_fused_exact_gated_gqa3_local384_fixed512_bf16_spans,
        laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_fixed512_bf16_spans,
        laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans,
        laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_fixed512_bf16_spans,
        laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_fixed512_bf16_spans,
    )
    from hipengine.loading.materialize import float_array_to_bf16_bits
    from hipengine.quant.gguf import bf16_to_float32
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    config = SimpleNamespace(
        block_count=1,
        layer_types=(SLIDING_ATTENTION,),
        head_counts=(72,),
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )
    cache = allocate_laguna_kv_cache(
        config,
        context_length=520,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    rng = np.random.default_rng(0xB164)
    keys = rng.normal(0.0, 0.12, size=(520, 8, 128)).astype(np.float32)
    values = rng.normal(0.0, 0.12, size=(520, 8, 128)).astype(np.float32)
    query = rng.normal(0.0, 0.12, size=(72, 128)).astype(np.float32)
    gate = rng.normal(0.0, 0.4, size=72).astype(np.float32)
    keys_bf16 = bf16_to_float32(float_array_to_bf16_bits(keys))
    values_bf16 = bf16_to_float32(float_array_to_bf16_bits(values))
    allocations = []
    try:
        key_device = malloc(keys.nbytes, runtime=runtime)
        value_device = malloc(values.nbytes, runtime=runtime)
        query_device = malloc(query.nbytes, runtime=runtime)
        gate_device = malloc(gate.nbytes, runtime=runtime)
        control_out = malloc(query.nbytes, runtime=runtime)
        candidate_out = malloc(query.nbytes, runtime=runtime)
        control_gated = malloc(query.size * 2, runtime=runtime)
        candidate_gated = malloc(query.size * 2, runtime=runtime)
        score_scratch = malloc(72 * 512 * 4, runtime=runtime)
        physical_scratch = malloc(72 * 512 * 4, runtime=runtime)
        allocations.extend(
            (
                key_device,
                value_device,
                query_device,
                gate_device,
                control_out,
                candidate_out,
                control_gated,
                candidate_gated,
                score_scratch,
                physical_scratch,
            )
        )
        for device, host in (
            (key_device, keys),
            (value_device, values),
            (query_device, query),
            (gate_device, gate),
        ):
            copy_host_to_device(
                device,
                host_array_ptr(host),
                host.nbytes,
                runtime=runtime,
            )
        cache.prepare_rows(tuple(range(512)))
        cache.append_rows(
            0,
            key_device.ptr,
            value_device.ptr,
            512,
            library=library,
        )
        cache.commit_rows()
        row_nbytes = 8 * 128 * np.dtype(np.float32).itemsize
        state = cache.layer(0)
        common = (
            query_device.ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
        )
        tail = (
            score_scratch.ptr,
            physical_scratch.ptr,
            state.spans,
            512,
            72,
            8,
            128,
            128**-0.5,
        )
        for position in range(512, 520):
            cache.prepare_position(position)
            cache.append(
                0,
                key_device.ptr + position * row_nbytes,
                value_device.ptr + position * row_nbytes,
                library=library,
            )
            if position == 512:
                cache.evict_swa_position(0, 200)
            laguna_swa_attention_decode_fused_exact_gated_gqa3_local384_fixed512_bf16_spans(
                *common,
                control_out.ptr,
                gate_device.ptr,
                control_gated.ptr,
                *tail,
                sliding_window=512,
                library=library,
                runtime=runtime,
            )
            laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_fixed512_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                sliding_window=512,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            control = np.empty_like(query)
            candidate = np.empty_like(query)
            control_gate_bits = np.empty(query.shape, dtype=np.uint16)
            candidate_gate_bits = np.empty_like(control_gate_bits)
            for host, device in (
                (control, control_out),
                (candidate, candidate_out),
                (control_gate_bits, control_gated),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            visible = np.arange(max(0, position - 511), position + 1)
            visible = visible[visible != 200]
            expected = _attention_reference(
                query,
                keys_bf16[visible],
                values_bf16[visible],
                num_kv_heads=8,
            )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
            laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_fixed512_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                sliding_window=512,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            for host, device in (
                (candidate, candidate_out),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
            laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                sliding_window=512,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            for host, device in (
                (candidate, candidate_out),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
            laguna_swa_attention_decode_fused_exact_gated_mixed32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                sliding_window=512,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            for host, device in (
                (candidate, candidate_out),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                sliding_window=512,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            for host, device in (
                (candidate, candidate_out),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                sliding_window=512,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            for host, device in (
                (candidate, candidate_out),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                sliding_window=512,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            for host, device in (
                (candidate, candidate_out),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                sliding_window=512,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            for host, device in (
                (candidate, candidate_out),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                sliding_window=512,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            for host, device in (
                (candidate, candidate_out),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
            laguna_swa_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                sliding_window=512,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            for host, device in (
                (candidate, candidate_out),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
            laguna_swa_attention_decode_fused_exact_gated_gqa3_vstage64_vec16_fixed512_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                sliding_window=512,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            for host, device in (
                (candidate, candidate_out),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        cache.free()


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_global_gqa2_vstage64_matches_cpu_with_eviction() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        build_laguna_kv_attention,
        laguna_global_attention_decode_fused_exact_gated_gqa1_fixedshape_bf16_spans,
        laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_fixedshape_bf16_spans,
        laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_fixedshape_bf16_spans,
        laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
        laguna_global_attention_decode_fused_exact_gated_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
        laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_fixedshape_bf16_spans,
        laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
        laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans,
    )
    from hipengine.loading.materialize import float_array_to_bf16_bits
    from hipengine.quant.gguf import bf16_to_float32
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    library = build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )
    config = SimpleNamespace(
        block_count=1,
        layer_types=(FULL_ATTENTION,),
        head_counts=(48,),
        head_count_kv=8,
        key_length=128,
        value_length=128,
        sliding_window=512,
    )
    cache = allocate_laguna_kv_cache(
        config,
        context_length=4096,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    rng = np.random.default_rng(0xB264)
    keys = rng.normal(0.0, 0.12, size=(640, 8, 128)).astype(np.float32)
    values = rng.normal(0.0, 0.12, size=(640, 8, 128)).astype(np.float32)
    query = rng.normal(0.0, 0.12, size=(48, 128)).astype(np.float32)
    gate = rng.normal(0.0, 0.4, size=48).astype(np.float32)
    keys_bf16 = bf16_to_float32(float_array_to_bf16_bits(keys))
    values_bf16 = bf16_to_float32(float_array_to_bf16_bits(values))
    allocations = []
    try:
        key_device = malloc(keys.nbytes, runtime=runtime)
        value_device = malloc(values.nbytes, runtime=runtime)
        query_device = malloc(query.nbytes, runtime=runtime)
        gate_device = malloc(gate.nbytes, runtime=runtime)
        control_out = malloc(query.nbytes, runtime=runtime)
        candidate_out = malloc(query.nbytes, runtime=runtime)
        control_gated = malloc(query.size * 2, runtime=runtime)
        candidate_gated = malloc(query.size * 2, runtime=runtime)
        score_scratch = malloc(48 * 4096 * 4, runtime=runtime)
        physical_scratch = malloc(48 * 4096 * 4, runtime=runtime)
        allocations.extend(
            (
                key_device,
                value_device,
                query_device,
                gate_device,
                control_out,
                candidate_out,
                control_gated,
                candidate_gated,
                score_scratch,
                physical_scratch,
            )
        )
        for device, host in (
            (key_device, keys),
            (value_device, values),
            (query_device, query),
            (gate_device, gate),
        ):
            copy_host_to_device(
                device,
                host_array_ptr(host),
                host.nbytes,
                runtime=runtime,
            )
        cache.prepare_rows(tuple(range(640)))
        cache.append_rows(
            0,
            key_device.ptr,
            value_device.ptr,
            640,
            library=library,
        )
        cache.commit_rows()
        cache.evict_position(0, 200)
        cache.prepare_position(640)
        state = cache.layer(0)
        common = (
            query_device.ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
        )
        for scan_slots in (513, 576, 639):
            tail = (
                score_scratch.ptr,
                physical_scratch.ptr,
                state.spans,
                scan_slots,
                4096,
                48,
                8,
                128,
                128**-0.5,
            )
            laguna_global_attention_decode_fused_exact_gated_gqa1_fixedshape_bf16_spans(
                *common,
                control_out.ptr,
                gate_device.ptr,
                control_gated.ptr,
                *tail,
                library=library,
                runtime=runtime,
            )
            laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_fixedshape_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            control = np.empty_like(query)
            candidate = np.empty_like(query)
            control_gate_bits = np.empty(query.shape, dtype=np.uint16)
            candidate_gate_bits = np.empty_like(control_gate_bits)
            for host, device in (
                (control, control_out),
                (candidate, candidate_out),
                (control_gate_bits, control_gated),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            visible = np.arange(scan_slots)
            visible = visible[visible != 200]
            expected = _attention_reference(
                query,
                keys_bf16[visible],
                values_bf16[visible],
                num_kv_heads=8,
            )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
            laguna_global_attention_decode_fused_exact_gated_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            for host, device in (
                (candidate, candidate_out),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
            laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            for host, device in (
                (candidate, candidate_out),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
            laguna_global_attention_decode_fused_exact_gated_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            for host, device in (
                (candidate, candidate_out),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
            laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_fixedshape_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            for host, device in (
                (candidate, candidate_out),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
            laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_direct_assume_exp_fixedshape_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            for host, device in (
                (candidate, candidate_out),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
            laguna_global_attention_decode_fused_exact_gated_gqa2_vstage64_vec16_fixedshape_bf16_spans(
                *common,
                candidate_out.ptr,
                gate_device.ptr,
                candidate_gated.ptr,
                *tail,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            for host, device in (
                (candidate, candidate_out),
                (candidate_gate_bits, candidate_gated),
            ):
                copy_device_to_host(
                    host_array_ptr(host),
                    device,
                    host.nbytes,
                    runtime=runtime,
                )
            np.testing.assert_allclose(candidate, expected, rtol=3e-4, atol=3e-4)
            assert np.array_equal(candidate, control)
            assert np.array_equal(candidate_gate_bits, control_gate_bits)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        cache.free()


def test_laguna_kv_owner_defaults_bounded_split_workspace_and_retains_rollback() -> None:
    from hipengine.runtime.laguna_kv import (
        allocate_laguna_kv_cache,
        resolve_laguna_split_thresholds,
        resolve_laguna_swa_split_tile16_threshold,
    )

    assert resolve_laguna_split_thresholds(
        "hip_gfx1100",
        context_length=4096,
        sliding_window=512,
    ) == (127, 65)
    assert resolve_laguna_split_thresholds(
        "hip_gfx1151",
        context_length=4096,
        sliding_window=512,
    ) == (127, 65)
    assert resolve_laguna_swa_split_tile16_threshold(
        "hip_gfx1100", sliding_window=512
    ) == 257
    assert resolve_laguna_swa_split_tile16_threshold(
        "hip_gfx1100", sliding_window=512, use_swa_split_tile16=False
    ) is None
    assert resolve_laguna_swa_split_tile16_threshold(
        "hip_gfx1151", sliding_window=512
    ) == 257

    runtime = _FakeRuntime()
    cache = allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    split_elements = max(48 * 4096, 72 * 512)
    try:
        assert cache.global_split_min_live == 127
        assert cache.swa_split_min_live == 65
        assert cache.swa_split_tile16_min_live == 257
        assert cache.split_gate_fusion
        assert cache.swa_split_wave_local
        assert not cache.swa_split_gqa3_scores
        assert cache.allocation_count == 245
        assert cache.resident_nbytes == sum(runtime.allocations.values())
        assert sorted(runtime.allocations.values()).count(split_elements * 4) == 2
    finally:
        cache.free()
    assert runtime.allocations == {}

    gfx1151_runtime = _FakeRuntime()
    gfx1151_cache = allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1151",
        runtime=gfx1151_runtime,
    )
    try:
        assert gfx1151_cache.global_split_min_live == 127
        assert gfx1151_cache.swa_split_min_live == 65
        assert gfx1151_cache.swa_split_tile16_min_live == 257
        assert gfx1151_cache.split_gate_fusion
        assert gfx1151_cache.global_split_fixedshape_reduce
        assert gfx1151_cache.global_fused_fixedshape
        assert gfx1151_cache.global_gqa2_vstage64_fixedshape
        assert gfx1151_cache.global_gqa2_vstage64_vec16_fixedshape
        assert gfx1151_cache.global_gqa2_vstage64_vec16_direct_fixedshape
        assert (
            gfx1151_cache.global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape
        )
        assert (
            gfx1151_cache.global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape
        )
        assert (
            gfx1151_cache.global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape
        )
        assert (
            gfx1151_cache.global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape
            is True
        )
        assert gfx1151_cache.swa_split_wave_local
        assert gfx1151_cache.swa_split_gqa3_scores
        assert gfx1151_cache.swa_split_fixed512_reduce
        assert gfx1151_cache.swa_fused_fixed512
        assert gfx1151_cache.swa_gqa3_local384_fixed512
        assert gfx1151_cache.swa_gqa3_vstage64_fixed512
        assert gfx1151_cache.swa_gqa3_vstage64_vec16_fixed512
        assert gfx1151_cache.swa_gqa3_vstage64_vec16_direct_fixed512
        assert gfx1151_cache.swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512
        assert (
            gfx1151_cache.swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512
        )
        assert (
            gfx1151_cache.swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512
        )
        assert (
            gfx1151_cache.swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512
        )
        assert (
            gfx1151_cache.swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512
        )
        assert (
            gfx1151_cache.swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512
        )
        assert (
            gfx1151_cache.swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512
            is True
        )
        assert (
            gfx1151_cache.swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512
            is False
        )
        gfx1151_cache.swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512 = (
            False
        )
        assert gfx1151_cache.allocation_count == 245
        resolved_variants = []

        def resolve_probe(layer, variant):
            resolved_variants.append((layer, variant))
            return lambda *args, **kwargs: None

        gfx1151_cache._resolve = resolve_probe
        gfx1151_cache.position = 256
        gfx1151_cache.global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape = (
            True
        )
        gfx1151_cache.attend(0, 1, 2, gate_ptr=3, gated_out_ptr=4)
        gfx1151_cache.global_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixedshape = (
            False
        )
        gfx1151_cache.attend(0, 1, 2, gate_ptr=3, gated_out_ptr=4)
        gfx1151_cache.global_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixedshape = (
            False
        )
        gfx1151_cache.attend(0, 1, 2, gate_ptr=3, gated_out_ptr=4)
        gfx1151_cache.global_gqa2_exp32_vstage64_vec16_direct_assume_exp_fixedshape = (
            False
        )
        gfx1151_cache.attend(0, 1, 2, gate_ptr=3, gated_out_ptr=4)
        gfx1151_cache.global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape = (
            False
        )
        gfx1151_cache.attend(0, 1, 2, gate_ptr=3, gated_out_ptr=4)
        gfx1151_cache.global_gqa2_vstage64_vec16_direct_assume_exp_fixedshape = (
            True
        )
        gfx1151_cache.position = 4000
        gfx1151_cache.attend(0, 1, 2, gate_ptr=3, gated_out_ptr=4)
        gfx1151_cache.position = 64
        gfx1151_cache.attend(1, 1, 2, gate_ptr=3, gated_out_ptr=4)
        gfx1151_cache.position = 511
        gfx1151_cache.swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512 = False
        gfx1151_cache.attend(1, 1, 2, gate_ptr=3, gated_out_ptr=4)
        gfx1151_cache.swa_gqa3_vstage64_vec16_direct_assume_exp_fixed512 = True
        gfx1151_cache.swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512 = (
            False
        )
        gfx1151_cache.attend(1, 1, 2, gate_ptr=3, gated_out_ptr=4)
        gfx1151_cache.swa_mixed32_vstage64_vec16_direct_assume_exp_fixed512 = (
            True
        )
        gfx1151_cache.swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512 = (
            False
        )
        gfx1151_cache.attend(1, 1, 2, gate_ptr=3, gated_out_ptr=4)
        gfx1151_cache.swa_mixed32_exp4_vstage64_vec16_direct_assume_exp_fixed512 = (
            True
        )
        gfx1151_cache.swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512 = (
            False
        )
        gfx1151_cache.attend(1, 1, 2, gate_ptr=3, gated_out_ptr=4)
        gfx1151_cache.swa_mixed32_exp8_vstage64_vec16_direct_assume_exp_fixed512 = (
            True
        )
        gfx1151_cache.swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512 = (
            False
        )
        gfx1151_cache.attend(1, 1, 2, gate_ptr=3, gated_out_ptr=4)
        gfx1151_cache.swa_mixed32_exp16_vstage64_vec16_direct_assume_exp_fixed512 = (
            True
        )
        gfx1151_cache.swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512 = (
            False
        )
        gfx1151_cache.attend(1, 1, 2, gate_ptr=3, gated_out_ptr=4)
        gfx1151_cache.swa_mixed32_exp32_vstage64_vec16_direct_assume_exp_fixed512 = (
            True
        )
        gfx1151_cache.swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512 = (
            True
        )
        gfx1151_cache.swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512 = (
            True
        )
        gfx1151_cache.attend(1, 1, 2, gate_ptr=3, gated_out_ptr=4)
        gfx1151_cache.swa_mixed32_exp32_producer_max_gate_vstage64_vec16_direct_assume_exp_fixed512 = (
            False
        )
        gfx1151_cache.attend(1, 1, 2, gate_ptr=3, gated_out_ptr=4)
        gfx1151_cache.swa_mixed32_exp32_producer_max_vstage64_vec16_direct_assume_exp_fixed512 = (
            False
        )
        gfx1151_cache.attend(1, 1, 2, gate_ptr=3, gated_out_ptr=4)
        assert resolved_variants == [
            (
                "laguna_attention_decode",
                (
                    "global_context_fused_exact_gated_mixed32_exp32_"
                    "producer_max_vstage64_vec16_direct_assume_exp_"
                    "fixedshape_spans"
                ),
            ),
            (
                "laguna_attention_decode",
                (
                    "global_context_fused_exact_gated_mixed32_exp32_vstage64_"
                    "vec16_direct_assume_exp_fixedshape_spans"
                ),
            ),
            (
                "laguna_attention_decode",
                (
                    "global_context_fused_exact_gated_gqa2_exp32_vstage64_"
                    "vec16_direct_assume_exp_fixedshape_spans"
                ),
            ),
            (
                "laguna_attention_decode",
                (
                    "global_context_fused_exact_gated_gqa2_vstage64_"
                    "vec16_direct_assume_exp_fixedshape_spans"
                ),
            ),
            (
                "laguna_attention_decode",
                (
                    "global_context_fused_exact_gated_"
                    "gqa2_vstage64_vec16_direct_fixedshape_spans"
                ),
            ),
            (
                "laguna_attention_decode",
                "global_context_fused_exact_gated_gqa1_fixedshape_spans",
            ),
            (
                "laguna_attention_decode",
                "swa_context_split_exact_gated_gqa3_scores_spans",
            ),
            (
                "laguna_attention_decode",
                (
                    "swa_context_fused_exact_gated_"
                    "gqa3_vstage64_vec16_direct_fixed512_spans"
                ),
            ),
            (
                "laguna_attention_decode",
                (
                    "swa_context_fused_exact_gated_gqa3_vstage64_"
                    "vec16_direct_assume_exp_fixed512_spans"
                ),
            ),
            (
                "laguna_attention_decode",
                (
                    "swa_context_fused_exact_gated_mixed32_vstage64_"
                    "vec16_direct_assume_exp_fixed512_spans"
                ),
            ),
            (
                "laguna_attention_decode",
                (
                    "swa_context_fused_exact_gated_mixed32_exp4_vstage64_"
                    "vec16_direct_assume_exp_fixed512_spans"
                ),
            ),
            (
                "laguna_attention_decode",
                (
                    "swa_context_fused_exact_gated_mixed32_exp8_vstage64_"
                    "vec16_direct_assume_exp_fixed512_spans"
                ),
            ),
            (
                "laguna_attention_decode",
                (
                    "swa_context_fused_exact_gated_mixed32_exp16_vstage64_"
                    "vec16_direct_assume_exp_fixed512_spans"
                ),
            ),
            (
                "laguna_attention_decode",
                (
                    "swa_context_fused_exact_gated_"
                    "mixed32_exp32_producer_max_gate_vstage64_vec16_direct_"
                    "assume_exp_fixed512_spans"
                ),
            ),
            (
                "laguna_attention_decode",
                (
                    "swa_context_fused_exact_gated_"
                    "mixed32_exp32_producer_max_vstage64_vec16_direct_"
                    "assume_exp_fixed512_spans"
                ),
            ),
            (
                "laguna_attention_decode",
                (
                    "swa_context_fused_exact_gated_mixed32_exp32_vstage64_"
                    "vec16_direct_assume_exp_fixed512_spans"
                ),
            ),
        ]
    finally:
        gfx1151_cache.free()
    assert gfx1151_runtime.allocations == {}

    shared_reducer_runtime = _FakeRuntime()
    shared_reducer = allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=shared_reducer_runtime,
        use_swa_split_wave_local=False,
    )
    try:
        assert not shared_reducer.swa_split_wave_local
        assert not shared_reducer.swa_split_gqa3_scores
        assert shared_reducer.split_gate_fusion
        assert shared_reducer.allocation_count == 245
    finally:
        shared_reducer.free()
    assert shared_reducer_runtime.allocations == {}

    tile16_runtime = _FakeRuntime()
    tile16 = allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=tile16_runtime,
        swa_split_tile16_min_live=257,
    )
    try:
        assert tile16.swa_split_tile16_min_live == 257
        assert tile16.split_gate_fusion
        assert tile16.allocation_count == 245
        assert sorted(tile16_runtime.allocations.values()).count(split_elements * 4) == 2
    finally:
        tile16.free()
    assert tile16_runtime.allocations == {}

    tile16_rollback_runtime = _FakeRuntime()
    tile16_rollback = allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=tile16_rollback_runtime,
        use_swa_split_tile16=False,
    )
    try:
        assert tile16_rollback.global_split_min_live == 127
        assert tile16_rollback.swa_split_min_live == 65
        assert tile16_rollback.swa_split_tile16_min_live is None
        assert tile16_rollback.split_gate_fusion
        assert tile16_rollback.allocation_count == 245
    finally:
        tile16_rollback.free()
    assert tile16_rollback_runtime.allocations == {}

    rollback_runtime = _FakeRuntime()
    rollback = allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=rollback_runtime,
        use_split_attention=False,
    )
    try:
        assert rollback.global_split_min_live is None
        assert rollback.swa_split_min_live is None
        assert rollback.swa_split_tile16_min_live is None
        assert not rollback.split_gate_fusion
        assert not rollback.swa_split_wave_local
        assert not rollback.swa_split_gqa3_scores
        assert rollback.allocation_count == 243
        assert sorted(rollback_runtime.allocations.values()).count(split_elements * 4) == 0
    finally:
        rollback.free()
    assert rollback_runtime.allocations == {}

    with pytest.raises(ValueError, match="global_split_min_live"):
        allocate_laguna_kv_cache(
            _production_config(),
            context_length=4096,
            backend="hip_gfx1100",
            runtime=_FakeRuntime(),
            global_split_min_live=4097,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        allocate_laguna_kv_cache(
            _production_config(),
            context_length=4096,
            backend="hip_gfx1100",
            runtime=_FakeRuntime(),
            use_split_attention=False,
            global_split_min_live=127,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        allocate_laguna_kv_cache(
            _production_config(),
            context_length=4096,
            backend="hip_gfx1100",
            runtime=_FakeRuntime(),
            use_split_attention=False,
            use_swa_split_wave_local=True,
        )
    with pytest.raises(ValueError, match="requires exact split attention"):
        allocate_laguna_kv_cache(
            _production_config(),
            context_length=4096,
            backend="hip_gfx1100",
            runtime=_FakeRuntime(),
            use_split_gate_fusion=False,
            use_swa_split_wave_local=True,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        allocate_laguna_kv_cache(
            _production_config(),
            context_length=4096,
            backend="hip_gfx1100",
            runtime=_FakeRuntime(),
            swa_split_tile16_min_live=257,
            use_swa_split_tile16=False,
        )
    with pytest.raises(ValueError, match="swa_split_tile16_min_live"):
        allocate_laguna_kv_cache(
            _production_config(),
            context_length=4096,
            backend="hip_gfx1100",
            runtime=_FakeRuntime(),
            swa_split_tile16_min_live=513,
        )
