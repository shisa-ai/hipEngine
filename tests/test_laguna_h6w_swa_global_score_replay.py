"""WPF-H6W exact late-start SWA aligned global-score replay contract."""

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
from hipengine.loading.laguna_gguf import SLIDING_ATTENTION

_CANDIDATE_STARTS = (256, 384)
_FALLBACK_STARTS = (0, 128)
_CALL_WEIGHTS = {256: 36, 384: 36}
_VARIANT = (
    "swa_context_rows_qrow4_dense_initial_global_score_replay_exact_spans"
)
_FUNCTION = (
    "laguna_swa_attention_prefill_qrow4_"
    "dense_initial_global_score_replay_exact_bf16_spans"
)
_SYMBOL = (
    "hipengine_laguna_swa_attention_prefill_qrow4_"
    "dense_initial_global_score_replay_exact_bf16_spans"
)
_KERNEL = (
    "laguna_swa_attention_prefill_qrow4_"
    "dense_initial_global_score_replay_exact_bf16_kernel"
)
_H6A_VARIANT = "swa_context_rows_qrow4_dense_initial_cached_exact_spans"
_H6A_FUNCTION = (
    "laguna_swa_attention_prefill_qrow4_"
    "dense_initial_cached_exact_bf16_spans"
)
_H6A_SYMBOL = (
    "hipengine_laguna_swa_attention_prefill_qrow4_"
    "dense_initial_cached_exact_bf16_spans"
)
_H6A_KERNEL_DECLARATION = (
    "template <bool kGlobalLayout, bool kDenseInitialMetadata = false>\n"
    "__global__ __launch_bounds__(32) void "
    "laguna_attention_prefill_qrow4_cached_exact_bf16_kernel("
)
_H6A_WRAPPER_DECLARATION = (
    'extern "C" int '
    "hipengine_laguna_swa_attention_prefill_qrow4_"
    "dense_initial_cached_exact_bf16_spans("
)
_H6A_REDUCTION_DECLARATION = (
    "__device__ __forceinline__ float laguna_wave32_sum_128_exact("
)
_H6A_REDUCTION_SHA256 = (
    "5aa8e6ef384bb15d41a4942e65bc01b5abe20008b89f2c8ad3f55700ead456aa"
)
_H6A_KERNEL_SHA256 = (
    "c73d333779e5db7382ea9142874ca3992ea3abb393517ec01e80f23e149a2806"
)
_H6A_WRAPPER_SHA256 = (
    "cb2b530f8881f75c8ad9722b8a534619a9a8b35a7c91451bdc2b2d492908229c"
)
_DENSE_ROLE = "global_m128_c4096_first_fill_exact"
_SWA_ROLE = "swa_qrow4_m128_c512_no_wrap_exact"
_H6N_GLOBAL_VARIANT = (
    "global_context_rows_dense_initial_fixed512_cached_exact_spans"
)
_PRODUCTION_POLICY = {
    _DENSE_ROLE: _H6N_GLOBAL_VARIANT,
    _SWA_ROLE: _H6A_VARIANT,
}
_H5R_POLICY = {
    _SWA_ROLE: "swa_context_rows_qrow4_cached_exact_spans",
}
_QUERY_HEADS = 72
_KV_HEADS = 8
_HEAD_DIM = 128
_ROWS = 128
_CAPACITY = 512
_QUERY_ROW_GROUPS = _ROWS // 4
_WORKGROUPS = _QUERY_HEADS * _QUERY_ROW_GROUPS
_SCORE_RECORD_BYTES = 16
_SCORE_ALIGNMENT = 16
_SCORE_SCRATCH_BYTES = _WORKGROUPS * _CAPACITY * _SCORE_RECORD_BYTES
_EXISTING_WORKSPACE_BYTES = 161_120_256
_CONTROL_MEDIANS_MS = {
    256: {"event": 0.7998931884765625, "wall": 0.8205093909054995},
    384: {"event": 1.0785511970520019, "wall": 1.09054297208786},
}
_EXPECTED_PHYSICAL = {
    "local_size": 32,
    "grid_x": 2304,
    "grid_y": 32,
    "global_load_u16": 8,
    "ds_bpermute_b32": 32,
    "global_store_b128_score_records": 1,
    "global_load_b128_score_records": 1,
    "global_store_b32_output": 16,
    "second_qk_key_load_sites": 0,
    "second_qk_wave_reduce_sites": 0,
    "v_exp_f32_e32": 4,
    "lds_bytes": 0,
    "block_barriers": 0,
    "private_bytes": 0,
    "spill_count": 0,
    "runtime_scratch_bytes": 0,
    "metadata_vgpr_max": 80,
    "runtime_vgpr_max": 80,
    "runtime_sgpr_max": 128,
}
_CONTEXT_ITERATIONS_PER_START_LAYER = {256: 741_888, 384: 1_036_800}
_DYNAMIC_WORK_MODEL = {
    "context_iterations_per_request": 64_032_768,
    "removed_bpermute_wave_instructions": 1_280_655_360,
    "removed_second_qk_u16_load_wave_instructions": 256_131_072,
    "aligned_score_record_load_store_wave_instructions": 128_065_536,
    "removed_logical_k_bytes": 16_392_388_608,
    "added_logical_score_bytes": 2_049_048_576,
    "net_logical_bytes_removed": 14_343_340_032,
}
_TRACE_PROTOCOL = {
    "kernel": _KERNEL,
    "require_cached_build": True,
    "compiler_processes_allowed": 0,
    "positive_duration_required": True,
    "local_size": 32,
    "lds_bytes": 0,
    "runtime_scratch_bytes": 0,
    "runtime_vgpr_max": 80,
}
_PROMOTION_TOPOLOGY = {"H6N": 48, "H6A": 72, "H6W": 72}
_TIMING_PROTOCOL = {
    "warmups_per_arm": 5,
    "samples_per_arm": 15,
    "launches_per_sample": 5,
    "counter_rotated_modes": True,
    "clocks": ("hip_event", "synchronized_wall"),
    "required_winners": _CANDIDATE_STARTS,
    "weighted_calls": sum(_CALL_WEIGHTS.values()),
    "no_follow_up_tuning_on_miss": True,
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


def _candidate():
    return getattr(_module(), _FUNCTION)


def _extract_braced(source: str, declaration: str) -> str:
    start = source.index(declaration)
    body_start = source.index("{", start)
    depth = 0
    for offset in range(body_start, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[start : offset + 1]
    raise AssertionError(f"unterminated body: {declaration}")


def _body(source: str, declaration: str) -> str:
    full = _extract_braced(source, declaration)
    return full[full.index("{") + 1 : -1]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _swa_spans(capacity: int = _CAPACITY) -> KVLiveSpans:
    return KVLiveSpans.sliding_ring(
        base_offsets=_tensor(0x11000, (capacity,), "int32"),
        live_counts=_tensor(0x12000, (1,), "int64"),
        token_positions=_tensor(0x13000, (capacity,), "int64"),
        evict_mask=_tensor(0x14000, (capacity,), "bool"),
        row_positions=_tensor(0x15000, (1,), "int64"),
        capacity=capacity,
        storage_dtype="bf16",
    )


def test_h6w_frozen_leaf_and_one_shot_admission_contract() -> None:
    assert _CANDIDATE_STARTS == (256, 384)
    assert _FALLBACK_STARTS == (0, 128)
    assert _CALL_WEIGHTS == {256: 36, 384: 36}
    assert sum(_CALL_WEIGHTS.values()) == 72
    assert _WORKGROUPS == 2_304
    assert _SCORE_SCRATCH_BYTES == 18_874_368
    assert _SCORE_SCRATCH_BYTES < _EXISTING_WORKSPACE_BYTES
    assert _SCORE_SCRATCH_BYTES % _SCORE_ALIGNMENT == 0
    assert _EXPECTED_PHYSICAL == {
        "local_size": 32,
        "grid_x": 2304,
        "grid_y": 32,
        "global_load_u16": 8,
        "ds_bpermute_b32": 32,
        "global_store_b128_score_records": 1,
        "global_load_b128_score_records": 1,
        "global_store_b32_output": 16,
        "second_qk_key_load_sites": 0,
        "second_qk_wave_reduce_sites": 0,
        "v_exp_f32_e32": 4,
        "lds_bytes": 0,
        "block_barriers": 0,
        "private_bytes": 0,
        "spill_count": 0,
        "runtime_scratch_bytes": 0,
        "metadata_vgpr_max": 80,
        "runtime_vgpr_max": 80,
        "runtime_sgpr_max": 128,
    }
    context_iterations = sum(
        _CALL_WEIGHTS[start] * _CONTEXT_ITERATIONS_PER_START_LAYER[start]
        for start in _CANDIDATE_STARTS
    )
    assert context_iterations == 64_032_768
    assert _DYNAMIC_WORK_MODEL == {
        "context_iterations_per_request": context_iterations,
        "removed_bpermute_wave_instructions": context_iterations * 20,
        "removed_second_qk_u16_load_wave_instructions": context_iterations * 4,
        "aligned_score_record_load_store_wave_instructions": (
            context_iterations * 2
        ),
        "removed_logical_k_bytes": context_iterations * 256,
        "added_logical_score_bytes": context_iterations * 32,
        "net_logical_bytes_removed": context_iterations * 224,
    }
    assert _TRACE_PROTOCOL == {
        "kernel": _KERNEL,
        "require_cached_build": True,
        "compiler_processes_allowed": 0,
        "positive_duration_required": True,
        "local_size": 32,
        "lds_bytes": 0,
        "runtime_scratch_bytes": 0,
        "runtime_vgpr_max": 80,
    }
    assert _PROMOTION_TOPOLOGY == {"H6N": 48, "H6A": 72, "H6W": 72}
    event_ms = sum(
        _CALL_WEIGHTS[start] * _CONTROL_MEDIANS_MS[start]["event"]
        for start in _CANDIDATE_STARTS
    )
    wall_ms = sum(
        _CALL_WEIGHTS[start] * _CONTROL_MEDIANS_MS[start]["wall"]
        for start in _CANDIDATE_STARTS
    )
    assert event_ms == pytest.approx(67.62399787902832)
    assert wall_ms == pytest.approx(68.79788506776094)
    assert _TIMING_PROTOCOL == {
        "warmups_per_arm": 5,
        "samples_per_arm": 15,
        "launches_per_sample": 5,
        "counter_rotated_modes": True,
        "clocks": ("hip_event", "synchronized_wall"),
        "required_winners": (256, 384),
        "weighted_calls": 72,
        "no_follow_up_tuning_on_miss": True,
    }
    # Admission is conjunctive: each start and the weighted aggregate must win
    # both clocks. Any miss deletes every H6W surface and skips runtime work.
    required_results = {
        (start, clock)
        for start in (*_CANDIDATE_STARTS, "weighted_72_call")
        for clock in _TIMING_PROTOCOL["clocks"]
    }
    assert len(required_results) == 6
    assert sum(_PROMOTION_TOPOLOGY.values()) == 192


def test_h6w_registry_source_abi_and_production_immutability() -> None:
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

    source = Path(module.__file__).with_name("laguna_kv_attention.hip").read_text()
    h6a_reduction = _extract_braced(source, _H6A_REDUCTION_DECLARATION)
    h6a_kernel = _extract_braced(source, _H6A_KERNEL_DECLARATION)
    h6a_wrapper = _extract_braced(source, _H6A_WRAPPER_DECLARATION)
    assert _sha256_text(h6a_reduction) == _H6A_REDUCTION_SHA256
    assert _sha256_text(h6a_kernel) == _H6A_KERNEL_SHA256
    assert _sha256_text(h6a_wrapper) == _H6A_WRAPPER_SHA256
    assert h6a_reduction.count("laguna_ordered_add_f32") == 4
    assert "for (int offset = 16; offset > 0; offset >>= 1)" in h6a_reduction
    assert h6a_kernel.count("laguna_wave32_sum_128_exact(") == 4
    assert h6a_wrapper.count(
        "laguna_attention_prefill_qrow4_cached_exact_bf16_kernel<false, true>"
    ) == 1

    candidate_key = KernelKey(
        "hip_gfx1100", "laguna_attention_prefill", "bf16", _VARIANT
    )
    gfx1151_key = KernelKey(
        "hip_gfx1151", "laguna_attention_prefill", "bf16", _VARIANT
    )
    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(gfx1151_key)

    candidate = _candidate()
    module.register_laguna_kv_attention_kernels(replace=True)
    assert candidate.__name__ == _FUNCTION
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant=_VARIANT,
        )
        is candidate
    )
    assert is_registered(candidate_key)
    assert not is_registered(gfx1151_key)
    assert (
        hip_gfx1100.LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS
        == _PRODUCTION_POLICY
    )

    source = Path(module.__file__).with_name("laguna_kv_attention.hip").read_text()
    assert source.count(_SYMBOL) == 1
    assert source.count(_KERNEL) == 2
    candidate_body = _body(
        source,
        f"__global__ __launch_bounds__(32) void {_KERNEL}(",
    )
    assert "constexpr int kQueryRows = 4;" in candidate_body
    assert "constexpr int kCapacity = 512;" in candidate_body
    assert "float4" in candidate_body
    assert "score_records" in candidate_body
    assert "__shared__" not in candidate_body
    assert "__syncthreads" not in candidate_body
    assert candidate_body.count("laguna_wave32_sum_128_exact(") == 1
    assert candidate_body.count("key_cache[") == 1
    assert candidate_body.count("value_cache[") == 1
    assert "dot * scale" in candidate_body
    assert "safe_denominator" in candidate_body
    for metadata_read in ("base_offsets[", "token_positions[", "evict_mask["):
        assert metadata_read not in candidate_body

    wrapper_source = inspect.getsource(candidate)
    assert "score_scratch_ptr" in wrapper_source
    assert "score_scratch_nbytes" in wrapper_source
    assert "parsed_start not in {256, 384}" in wrapper_source
    assert "score_scratch_ptr % 16" in wrapper_source
    assert "18_874_368" in wrapper_source
    assert "global-score replay exact SWA requires" in wrapper_source


def test_h6w_strict_late_start_and_scratch_preflight_before_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    h6a = getattr(module, _H6A_FUNCTION)
    h6a_calls: list[tuple[object, ...]] = []

    class FakeFn:
        argtypes = None
        restype = None

        def __init__(self, calls: list[tuple[object, ...]]) -> None:
            self.calls = calls

        def __call__(self, *args: object) -> int:
            self.calls.append(args)
            return 0

    h6a_library = SimpleNamespace(**{_H6A_SYMBOL: FakeFn(h6a_calls)})
    common = (0x6000, 0x7000, 0x8000, 0x9000, 0xA000, 0xB000)
    for start in _FALLBACK_STARTS:
        h6a(
            *common,
            _swa_spans(),
            _ROWS,
            _QUERY_HEADS,
            _KV_HEADS,
            _HEAD_DIM,
            _HEAD_DIM**-0.5,
            sliding_window=_CAPACITY,
            start_position=start,
            library=h6a_library,
            runtime=SimpleNamespace(),
        )
    assert len(h6a_calls) == len(_FALLBACK_STARTS)

    candidate = _candidate()
    calls: list[tuple[object, ...]] = []
    library = SimpleNamespace(**{_SYMBOL: FakeFn(calls)})

    def launch(*, fake_library: object | None = library, **overrides: object) -> None:
        arguments = {
            "score_scratch_ptr": 0xC000,
            "score_scratch_nbytes": _SCORE_SCRATCH_BYTES,
            "spans": _swa_spans(),
            "rows": _ROWS,
            "num_q_heads": _QUERY_HEADS,
            "num_kv_heads": _KV_HEADS,
            "head_dim": _HEAD_DIM,
            "sliding_window": _CAPACITY,
            "start_position": 256,
        }
        arguments.update(overrides)
        candidate(
            *common,
            arguments.pop("score_scratch_ptr"),
            arguments.pop("spans"),
            arguments.pop("rows"),
            arguments.pop("num_q_heads"),
            arguments.pop("num_kv_heads"),
            arguments.pop("head_dim"),
            _HEAD_DIM**-0.5,
            score_scratch_nbytes=arguments.pop("score_scratch_nbytes"),
            **arguments,
            library=fake_library,
            runtime=SimpleNamespace(),
        )

    for start in _CANDIDATE_STARTS:
        launch(start_position=start)
    assert len(calls) == len(_CANDIDATE_STARTS)
    for args, start in zip(calls, _CANDIDATE_STARTS, strict=True):
        assert args[6].value == 0xC000
        assert args[12].value == _ROWS
        assert args[13].value == _CAPACITY
        assert args[14].value == _CAPACITY
        assert args[15].value == _QUERY_HEADS
        assert args[16].value == _KV_HEADS
        assert args[17].value == _HEAD_DIM
        assert args[18].value == start

    def fail_build(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid H6W preflight loaded HIP")

    monkeypatch.setattr(module, "build_laguna_kv_attention", fail_build)
    invalid_cases = (
        {"rows": 127},
        {"spans": _swa_spans(384)},
        {"num_q_heads": 48},
        {"num_kv_heads": 4},
        {"head_dim": 64},
        {"sliding_window": 256},
        {"start_position": None},
        {"start_position": 0},
        {"start_position": 128},
        {"start_position": 64},
        {"start_position": 512},
        {"score_scratch_ptr": 0},
        {"score_scratch_ptr": 0xC008},
        {"score_scratch_nbytes": _SCORE_SCRATCH_BYTES - 16},
        {"score_scratch_nbytes": _SCORE_SCRATCH_BYTES + 16},
    )
    for invalid in invalid_cases:
        with pytest.raises(ValueError, match="global-score replay exact SWA"):
            launch(fake_library=None, **invalid)
    assert len(calls) == len(_CANDIDATE_STARTS)


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

    keys_bf16 = bf16_to_float32(float_array_to_bf16_bits(keys))
    values_bf16 = bf16_to_float32(float_array_to_bf16_bits(values))
    kv_for_q = np.arange(_QUERY_HEADS, dtype=np.int64) // (
        _QUERY_HEADS // _KV_HEADS
    )
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
        scores *= np.float32(_HEAD_DIM**-0.5)
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


def _assert_scratch_record_coverage(
    scratch: np.ndarray,
    *,
    start_position: int,
) -> None:
    records = scratch.reshape(
        _QUERY_ROW_GROUPS,
        _QUERY_HEADS,
        _CAPACITY,
        _SCORE_RECORD_BYTES,
    )
    poison = np.full(_SCORE_RECORD_BYTES, 0xA5, dtype=np.uint8)
    for row_group in range(_QUERY_ROW_GROUPS):
        dense_context_len = start_position + row_group * 4 + 4
        prefix = records[row_group, :, :dense_context_len, :].reshape(
            -1, _SCORE_RECORD_BYTES
        )
        assert not np.any(np.all(prefix == poison, axis=1))
        suffix = records[row_group, :, dense_context_len:, :]
        assert np.all(suffix == 0xA5)


@pytest.fixture(scope="module")
def h6a_library():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    return _module().build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )


@pytest.mark.parametrize("start_position", _CANDIDATE_STARTS)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h6w_complete_output_scratch_cpu_spans_and_lifecycle(
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
    config = SimpleNamespace(
        block_count=1,
        layer_types=(SLIDING_ATTENTION,),
        head_counts=(_QUERY_HEADS,),
        head_count_kv=_KV_HEADS,
        key_length=_HEAD_DIM,
        value_length=_HEAD_DIM,
        sliding_window=_CAPACITY,
    )
    before = memory_stats()
    cache = allocate_laguna_kv_cache(
        config,
        context_length=_CAPACITY,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    rng = np.random.default_rng(0x6A70 + start_position)
    total_rows = start_position + _ROWS
    keys = rng.normal(
        0.0, 0.12, size=(total_rows, _KV_HEADS, _HEAD_DIM)
    ).astype(np.float32)
    values = rng.normal(0.0, 0.12, size=keys.shape).astype(np.float32)
    queries = rng.normal(
        0.0, 0.12, size=(_ROWS, _QUERY_HEADS, _HEAD_DIM)
    ).astype(np.float32)
    h6a_host = np.empty_like(queries)
    candidate_host = np.empty_like(queries)
    scratch_host = np.empty(_SCORE_SCRATCH_BYTES, dtype=np.uint8)
    allocations = []
    try:
        key_rows = malloc(keys.nbytes, runtime=runtime)
        value_rows = malloc(values.nbytes, runtime=runtime)
        query_rows = malloc(queries.nbytes, runtime=runtime)
        h6a_out = malloc(h6a_host.nbytes, runtime=runtime)
        candidate_out = malloc(candidate_host.nbytes, runtime=runtime)
        score_scratch = malloc(_SCORE_SCRATCH_BYTES, runtime=runtime)
        allocations.extend(
            (
                key_rows,
                value_rows,
                query_rows,
                h6a_out,
                candidate_out,
                score_scratch,
            )
        )
        assert score_scratch.ptr % _SCORE_ALIGNMENT == 0
        assert score_scratch.nbytes == _SCORE_SCRATCH_BYTES
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
        row_nbytes = _KV_HEADS * _HEAD_DIM * np.dtype(np.float32).itemsize
        current_key_ptr = key_rows.ptr + start_position * row_nbytes
        current_value_ptr = value_rows.ptr + start_position * row_nbytes
        cache.append_rows(
            0,
            current_key_ptr,
            current_value_ptr,
            _ROWS,
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
            _ROWS,
            _QUERY_HEADS,
            _KV_HEADS,
            _HEAD_DIM,
            _HEAD_DIM**-0.5,
        )
        h6a(
            *common,
            h6a_out.ptr,
            *suffix,
            sliding_window=_CAPACITY,
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
        for actual, expected in zip(after_h6a_spans, before_spans, strict=True):
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
        runtime.memset(score_scratch.ptr, 0xA5, score_scratch.nbytes)
        candidate(
            *common,
            candidate_out.ptr,
            score_scratch.ptr,
            *suffix,
            score_scratch_nbytes=score_scratch.nbytes,
            sliding_window=_CAPACITY,
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
        copy_device_to_host(
            host_array_ptr(scratch_host),
            score_scratch,
            scratch_host.nbytes,
            runtime=runtime,
        )
        assert np.isfinite(candidate_host).all()
        np.testing.assert_array_equal(candidate_host, h6a_host)
        _assert_scratch_record_coverage(
            scratch_host,
            start_position=start_position,
        )
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
