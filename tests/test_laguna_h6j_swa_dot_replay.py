"""WPF-H6J exact dense-initial SWA qrow4 unscaled-dot replay contract."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.laguna_gguf import SLIDING_ATTENTION

_STARTS = (0, 128, 256, 384)
_VARIANT = "swa_context_rows_qrow4_dense_initial_dot_replay_exact_spans"
_FUNCTION = (
    "laguna_swa_attention_prefill_qrow4_"
    "dense_initial_dot_replay_exact_bf16_spans"
)
_SYMBOL = (
    "hipengine_laguna_swa_attention_prefill_qrow4_"
    "dense_initial_dot_replay_exact_bf16_spans"
)
_KERNEL = (
    "laguna_swa_attention_prefill_qrow4_"
    "dense_initial_dot_replay_exact_bf16_kernel"
)
_H6A_VARIANT = "swa_context_rows_qrow4_dense_initial_cached_exact_spans"
_H6A_FUNCTION = (
    "laguna_swa_attention_prefill_qrow4_"
    "dense_initial_cached_exact_bf16_spans"
)
_H6A_KERNEL = "laguna_attention_prefill_qrow4_cached_exact_bf16_kernel"
_H6A_DENSE_BRANCH_SHA256 = (
    "17d57f47e42e618e3b2cf4531136613b65e8fd8e48a600aebb268785d984e6cf"
)
_PRODUCTION_POLICY = {
    "global_m128_c4096_first_fill_exact": (
        "global_context_rows_dense_initial_cached_exact_spans"
    ),
    "swa_qrow4_m128_c512_no_wrap_exact": _H6A_VARIANT,
}
_H5R_POLICY = {
    "swa_qrow4_m128_c512_no_wrap_exact": (
        "swa_context_rows_qrow4_cached_exact_spans"
    )
}
_LDS_BYTES = 4 * 512 * np.dtype(np.float32).itemsize


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


def _candidate():
    return getattr(_module(), _FUNCTION)


def _function_body(source: str, declaration: str) -> str:
    start = source.index(declaration)
    body_start = source.index("{", start)
    depth = 0
    for offset in range(body_start, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[body_start + 1 : offset]
    raise AssertionError(f"unterminated function: {declaration}")


def _branch_body(source: str, declaration: str, branch: str) -> str:
    body = _function_body(source, declaration)
    start = body.index(branch)
    body_start = body.index("{", start)
    depth = 0
    for offset in range(body_start, len(body)):
        if body[offset] == "{":
            depth += 1
        elif body[offset] == "}":
            depth -= 1
            if depth == 0:
                return body[body_start + 1 : offset]
    raise AssertionError(f"unterminated branch: {branch}")


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


def test_h6j_registry_source_schedule_and_production_immutability() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.registry import KernelKey, is_registered, resolve

    module = _module()
    module.register_laguna_kv_attention_kernels()
    h6a = getattr(module, _H6A_FUNCTION)
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant=_H6A_VARIANT,
        )
        is h6a
    )
    assert (
        hip_gfx1100.LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS
        == _PRODUCTION_POLICY
    )
    assert hip_gfx1100.LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS == _H5R_POLICY
    assert _VARIANT not in _PRODUCTION_POLICY.values()
    assert _LDS_BYTES == 8_192

    source = Path(module.__file__).with_suffix(".hip").read_text()
    dense_branch = _branch_body(
        source,
        f"__global__ __launch_bounds__(32) void {_H6A_KERNEL}",
        "  if constexpr (kDenseInitialMetadata) {",
    )
    assert hashlib.sha256(dense_branch.encode()).hexdigest() == (
        _H6A_DENSE_BRANCH_SHA256
    )
    assert dense_branch.count("laguna_wave32_sum_128_exact(") == 2
    assert dense_branch.count("key_cache[") == 2
    assert dense_branch.count("value_cache[") == 1
    assert "dot * scale - dense_max_scores[row_index]" in dense_branch
    assert "dense_output_acc[row_index][part] / safe_denominator" in dense_branch

    candidate_key = KernelKey(
        "hip_gfx1100", "laguna_attention_prefill", "bf16", _VARIANT
    )
    gfx1151_key = KernelKey(
        "hip_gfx1151", "laguna_attention_prefill", "bf16", _VARIANT
    )
    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(gfx1151_key)

    candidate = _candidate()
    assert candidate.__name__ == _FUNCTION
    assert resolve(
        backend="hip_gfx1100",
        layer="laguna_attention_prefill",
        quant="bf16",
        variant=_VARIANT,
    ) is candidate
    assert is_registered(candidate_key)

    source = Path(module.__file__).with_suffix(".hip").read_text()
    assert source.count(_SYMBOL) == 1
    assert source.count(_KERNEL) == 2
    candidate_body = _function_body(
        source,
        f"__global__ __launch_bounds__(32) void {_KERNEL}",
    )
    assert "constexpr int kQueryRows = 4;" in candidate_body
    assert "constexpr int kCapacity = 512;" in candidate_body
    assert "__shared__ float replay_dots[kQueryRows][kCapacity];" in candidate_body
    assert "static_assert(sizeof(replay_dots) == 8192);" in candidate_body
    assert candidate_body.count("laguna_wave32_sum_128_exact(") == 1
    assert candidate_body.count("key_cache[") == 1
    assert candidate_body.count("value_cache[") == 1
    assert "replay_dots[row_index][logical_slot] = dot;" in candidate_body
    assert "const float dot = replay_dots[row_index][logical_slot];" in candidate_body
    assert "dot * scale - max_scores[row_index]" in candidate_body
    assert "output_acc[row_index][part] / safe_denominator" in candidate_body
    for metadata in ("base_offsets[", "token_positions[", "evict_mask["):
        assert metadata not in candidate_body


def test_h6j_strict_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    candidate = _candidate()
    calls: list[tuple[object, ...]] = []

    class FakeFn:
        argtypes = None
        restype = None

        def __call__(self, *args: object) -> int:
            calls.append(args)
            return 0

    library = SimpleNamespace(**{_SYMBOL: FakeFn()})
    common = (0x6000, 0x7000, 0x8000, 0x9000, 0xA000, 0xB000)

    def launch(**overrides: object) -> None:
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
        candidate(
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

    for start in _STARTS:
        launch(start_position=start)
    assert len(calls) == len(_STARTS)

    def fail_build(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid H6J preflight loaded HIP")

    monkeypatch.setattr(module, "build_laguna_kv_attention", fail_build)
    for invalid in (
        {"rows": 127},
        {"spans": _swa_spans(384)},
        {"num_q_heads": 48},
        {"num_kv_heads": 4},
        {"head_dim": 64},
        {"sliding_window": 256},
        {"start_position": None},
        {"start_position": 512},
    ):
        with pytest.raises(ValueError, match="dot-replay exact SWA"):
            launch(**invalid)
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
    start_position: int,
) -> dict[int, np.ndarray]:
    from hipengine.loading.materialize import float_array_to_bf16_bits
    from hipengine.quant.gguf import bf16_to_float32

    query_heads = 72
    kv_heads = 8
    head_dim = 128
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


@pytest.fixture(scope="module")
def h6a_library():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    return _module().build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )


@pytest.mark.parametrize("start_position", _STARTS)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h6j_complete_output_matches_h6a_cpu_and_immutable_spans(
    h6a_library,
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
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    module = _module()
    h6a = getattr(module, _H6A_FUNCTION)
    runtime = get_hip_runtime()
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
    rng = np.random.default_rng(0x6A10 + start_position)
    total_rows = start_position + rows
    keys = rng.normal(
        0.0, 0.12, size=(total_rows, kv_heads, head_dim)
    ).astype(np.float32)
    values = rng.normal(0.0, 0.12, size=keys.shape).astype(np.float32)
    queries = rng.normal(
        0.0, 0.12, size=(rows, query_heads, head_dim)
    ).astype(np.float32)
    h6a_host = np.empty_like(queries)
    candidate_host = np.empty_like(queries)
    allocations = []
    try:
        key_rows = malloc(keys.nbytes, runtime=runtime)
        value_rows = malloc(values.nbytes, runtime=runtime)
        query_rows = malloc(queries.nbytes, runtime=runtime)
        h6a_out = malloc(h6a_host.nbytes, runtime=runtime)
        candidate_out = malloc(candidate_host.nbytes, runtime=runtime)
        allocations.extend(
            (key_rows, value_rows, query_rows, h6a_out, candidate_out)
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
                library=h6a_library,
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
            library=h6a_library,
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
        h6a(
            *common,
            h6a_out.ptr,
            *suffix,
            sliding_window=512,
            start_position=start_position,
            library=h6a_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(h6a_host),
            h6a_out,
            h6a_host.nbytes,
            runtime=runtime,
        )
        assert np.isfinite(h6a_host).all()
        after_h6a_spans = _span_snapshot(state.spans, runtime)
        for actual, expected in zip(
            after_h6a_spans, before_spans, strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
        for row, expected in _cpu_rows(
            queries,
            keys,
            values,
            start_position=start_position,
        ).items():
            np.testing.assert_allclose(
                h6a_host[row], expected, rtol=3e-4, atol=3e-4
            )

        candidate = _candidate()
        runtime.memset(candidate_out.ptr, 0xA5, candidate_out.nbytes)
        candidate(
            *common,
            candidate_out.ptr,
            *suffix,
            sliding_window=512,
            start_position=start_position,
            library=h6a_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(candidate_host),
            candidate_out,
            candidate_host.nbytes,
            runtime=runtime,
        )
        np.testing.assert_array_equal(candidate_host, h6a_host)
        after_candidate_spans = _span_snapshot(state.spans, runtime)
        for actual, expected in zip(
            after_candidate_spans, before_spans, strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
        cache.discard_rows()
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        cache.free()
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
