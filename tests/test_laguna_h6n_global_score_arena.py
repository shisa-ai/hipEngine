"""WPF-H6N exact global dense-initial fixed-512 score-arena contract."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans

_STARTS = (0, 128, 256, 384)
_H6A_VARIANT = "global_context_rows_dense_initial_cached_exact_spans"
_H6A_FUNCTION = "laguna_global_attention_prefill_dense_initial_cached_exact_bf16_spans"
_H6N_VARIANT = "global_context_rows_dense_initial_fixed512_cached_exact_spans"
_H6N_FUNCTION = (
    "laguna_global_attention_prefill_dense_initial_fixed512_cached_exact_bf16_spans"
)
_H6N_SYMBOL = (
    "hipengine_laguna_global_attention_prefill_"
    "dense_initial_fixed512_cached_exact_bf16_spans"
)
_H6N_KERNEL = (
    "laguna_global_attention_prefill_"
    "dense_initial_fixed512_cached_exact_bf16_kernel"
)
_DENSE_ROLE = "global_m128_c4096_first_fill_exact"
_SWA_ROLE = "swa_qrow4_m128_c512_no_wrap_exact"
_SWA_H6A_VARIANT = "swa_context_rows_qrow4_dense_initial_cached_exact_spans"
_PRODUCTION_POLICY = {
    _DENSE_ROLE: _H6N_VARIANT,
    _SWA_ROLE: _SWA_H6A_VARIANT,
}
_H6A_KERNEL_SHA256 = "9c6ec1d45e375f22c9e97854f2d8c7a70dbcfaa2df9dcd710a8dba4fbd56721b"
_H6A_WRAPPER_SHA256 = "535f454badd8ccd2692d865c8aa0f8cdf80e8737644cf101ce49181f154d9abd"


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


def _source_path() -> Path:
    return Path(_module().__file__).with_name("laguna_kv_attention.hip")


def _extract_body(source: str, anchor: str) -> str:
    start = source.index(anchor)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated body for {anchor}")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


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


def test_h6n_source_registry_policy_and_static_contract() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.registry import KernelKey, is_registered, resolve
    from hipengine.runtime.laguna_gguf_runner import LagunaQ5F32OrderedScratch

    module = _module()
    source = _source_path().read_text()
    h6a_kernel = _extract_body(
        source,
        "template <bool kDenseInitialMetadata = false>\n"
        "__global__ void laguna_global_attention_prefill_cached_exact_bf16_kernel(",
    )
    h6a_wrapper = _extract_body(
        source,
        'extern "C" int '
        "hipengine_laguna_global_attention_prefill_"
        "dense_initial_cached_exact_bf16_spans(",
    )
    assert _sha256_text(h6a_kernel) == _H6A_KERNEL_SHA256
    assert _sha256_text(h6a_wrapper) == _H6A_WRAPPER_SHA256
    assert hip_gfx1100.LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS == (
        _PRODUCTION_POLICY
    )
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256
    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            "laguna_attention_prefill",
            "bf16",
            _H6N_VARIANT,
        )
    )

    candidate = getattr(module, _H6N_FUNCTION)
    module.register_laguna_kv_attention_kernels(replace=True)
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant=_H6N_VARIANT,
        )
        is candidate
    )
    assert hip_gfx1100.LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS == (
        _PRODUCTION_POLICY
    )

    candidate_kernel = _extract_body(
        source,
        f"__global__ void {_H6N_KERNEL}(",
    )
    candidate_wrapper = _extract_body(
        source,
        f'extern "C" int {_H6N_SYMBOL}(',
    )
    assert "constexpr int64_t kScoreCapacity = 512;" in candidate_kernel
    assert "float* warp_buf = scores + kScoreCapacity;" in candidate_kernel
    assert "float* q_shared = warp_buf + num_warps;" in candidate_kernel
    assert "scores + capacity" not in candidate_kernel
    assert candidate_kernel.count(
        "for (int64_t token = warp_id; token < context_len; token += num_warps)"
    ) == 2
    assert candidate_kernel.count(
        "for (int64_t token = 0; token < context_len; ++token)"
    ) == 1
    for statement in (
        "scores[token] = score;",
        "scores[token] = weight;",
        "const float weight = scores[token] * inv_denom;",
        "acc += weight * v;",
        "out[q_offset + dim] = acc;",
        "physical_slot = token;",
    ):
        assert statement in candidate_kernel
    for metadata_read in (
        "token_positions[token]",
        "evict_mask[token]",
        "laguna_global_physical_slot(base_offsets, token",
    ):
        assert metadata_read not in candidate_kernel
    assert "constexpr int64_t kScoreCapacity = 512;" in candidate_wrapper
    assert (
        "static_cast<size_t>(kScoreCapacity + warps + head_dim) * sizeof(float)"
        in candidate_wrapper
    )
    assert f"{_H6N_KERNEL}" in candidate_wrapper
    assert "capacity + warps + head_dim" not in candidate_wrapper
    assert "dim3(static_cast<unsigned int>(num_q_heads)," in candidate_wrapper
    assert "dim3(threads)" in candidate_wrapper

    wrapper_source = inspect.getsource(candidate)
    assert "parsed_start + int(rows) > 512" in wrapper_source
    assert "fixed-512 score-arena global requires" in wrapper_source
    assert "_SYMBOL_GLOBAL_PREFILL_DENSE_INITIAL_FIXED512_CACHED_EXACT" in (
        wrapper_source
    )


def test_h6n_strict_preflight_before_hip_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    candidate = getattr(module, _H6N_FUNCTION)
    calls: list[tuple[object, ...]] = []

    class FakeFn:
        argtypes = None
        restype = None

        def __call__(self, *args: object) -> int:
            calls.append(args)
            return 0

    library = SimpleNamespace(**{_H6N_SYMBOL: FakeFn()})

    def launch(*, fake_library: object | None = library, **overrides: object) -> None:
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
        candidate(
            0x6000,
            0x7000,
            0x8000,
            0x9000,
            0xA000,
            0xB000,
            arguments.pop("spans"),
            arguments.pop("rows"),
            arguments.pop("max_context_len"),
            arguments.pop("num_q_heads"),
            arguments.pop("num_kv_heads"),
            arguments.pop("head_dim"),
            128**-0.5,
            **arguments,
            library=fake_library,
            runtime=SimpleNamespace(),
        )

    for start in _STARTS:
        launch(start_position=start)
    assert len(calls) == len(_STARTS)
    for args, start in zip(calls, _STARTS, strict=True):
        assert args[11].value == 128
        assert args[12].value == 4096
        assert args[13].value == 256
        assert args[14].value == 16
        assert args[15].value == 48
        assert args[16].value == 8
        assert args[17].value == 128
        assert args[18].value == start

    def fail_build(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid H6N preflight loaded HIP")

    monkeypatch.setattr(module, "build_laguna_kv_attention", fail_build)
    invalid_cases = (
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
    for invalid in invalid_cases:
        with pytest.raises(ValueError, match="fixed-512 score-arena global"):
            launch(fake_library=None, **invalid)
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

    query_heads = 48
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


@pytest.mark.parametrize("start_position", _STARTS)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h6n_complete_output_matches_h6a_cpu_and_spans(
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
    from hipengine.loading.laguna_gguf import FULL_ATTENTION
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    module = _module()
    control = getattr(module, _H6A_FUNCTION)
    runtime = get_hip_runtime()
    library = module.build_laguna_kv_attention(
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
    cache = allocate_laguna_kv_cache(
        config,
        context_length=capacity,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    total_rows = start_position + rows
    rng = np.random.default_rng(0x6E00 + start_position)
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
    candidate_host = np.full_like(queries, np.nan)
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
            (candidate_out, candidate_host),
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
            capacity,
            query_heads,
            kv_heads,
            head_dim,
            head_dim**-0.5,
        )
        control(
            *common,
            control_out.ptr,
            *suffix,
            start_position=start_position,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(control_host),
            control_out,
            control_host.nbytes,
            runtime=runtime,
        )
        assert np.isfinite(control_host).all()
        after_control_spans = _span_snapshot(state.spans, runtime)
        for actual, expected in zip(
            after_control_spans,
            before_spans,
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        for row, expected in _cpu_rows(
            queries,
            keys,
            values,
            start_position=start_position,
        ).items():
            np.testing.assert_allclose(
                control_host[row],
                expected,
                rtol=3e-4,
                atol=3e-4,
            )

        candidate = getattr(module, _H6N_FUNCTION)
        candidate(
            *common,
            candidate_out.ptr,
            *suffix,
            start_position=start_position,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(candidate_host),
            candidate_out,
            candidate_host.nbytes,
            runtime=runtime,
        )
        assert np.isfinite(candidate_host).all()
        np.testing.assert_array_equal(candidate_host, control_host)
        after_candidate_spans = _span_snapshot(state.spans, runtime)
        for actual, expected in zip(
            after_candidate_spans,
            before_spans,
            strict=True,
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
