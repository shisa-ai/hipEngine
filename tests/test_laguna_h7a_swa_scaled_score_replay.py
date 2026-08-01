"""WPF-H7A exact late-SWA scaled-score replay RED contract."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import importlib.util
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

_HELPER_PATH = Path(__file__).with_name(
    "test_laguna_h6w_swa_global_score_replay.py"
)
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "laguna_h7a_h6w_helper", _HELPER_PATH
)
assert _HELPER_SPEC is not None and _HELPER_SPEC.loader is not None
h6w = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(h6w)

_CANDIDATE_STARTS = (256, 384)
_FALLBACK_STARTS = (0, 128)
_CALL_WEIGHTS = {256: 36, 384: 36}
_VARIANT = (
    "swa_context_rows_qrow4_dense_initial_"
    "global_scaled_score_replay_exact_spans"
)
_FUNCTION = (
    "laguna_swa_attention_prefill_qrow4_dense_initial_"
    "global_scaled_score_replay_exact_bf16_spans"
)
_SYMBOL = (
    "hipengine_laguna_swa_attention_prefill_qrow4_dense_initial_"
    "global_scaled_score_replay_exact_bf16_spans"
)
_KERNEL = (
    "laguna_swa_attention_prefill_qrow4_dense_initial_"
    "global_scaled_score_replay_exact_bf16_kernel"
)
_H6W_VARIANT = (
    "swa_context_rows_qrow4_dense_initial_global_score_replay_exact_spans"
)
_H6W_FUNCTION = (
    "laguna_swa_attention_prefill_qrow4_"
    "dense_initial_global_score_replay_exact_bf16_spans"
)
_H6W_KERNEL_DECLARATION = (
    "__global__ __launch_bounds__(32) void "
    "laguna_swa_attention_prefill_qrow4_dense_initial_"
    "global_score_replay_exact_bf16_kernel("
)
_H6W_WRAPPER_DECLARATION = (
    'extern "C" int '
    "hipengine_laguna_swa_attention_prefill_qrow4_dense_initial_"
    "global_score_replay_exact_bf16_spans("
)
_H6W_KERNEL_SHA256 = (
    "d025a3c73c499c75b2dc0444ae4fdd479cb72729eaad2b1e1aa6e38491748aee"
)
_H6W_WRAPPER_SHA256 = (
    "01b12957a7fc404486d96928918a371169648835be60b08c1b477dd153dddde9"
)
_DENSE_ROLE = "global_m128_c4096_first_fill_exact"
_SWA_ROLE = "swa_qrow4_m128_c512_no_wrap_exact"
_H6Z_VARIANT = (
    "global_context_rows_qrow4_dense_initial_"
    "global_score_weight_replay_exact_spans"
)
_SOURCE_POLICY = {
    _DENSE_ROLE: _H6Z_VARIANT,
    _SWA_ROLE: _H6W_VARIANT,
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
_VISIBLE_SCORE_REPLAYS = {
    start: sum(start + row + 1 for row in range(_ROWS))
    * _QUERY_HEADS
    * _CALL_WEIGHTS[start]
    for start in _CANDIDATE_STARTS
}
_DYNAMIC_WORK_MODEL = {
    "visible_score_replays_by_start": {256: 106_334_208, 384: 148_801_536},
    "removed_duplicate_scale_multiplies": 255_135_744,
    "logical_bytes_changed": 0,
    "workgroups_changed": 0,
}
_EXPECTED_PHYSICAL = {
    "local_size": 32,
    "grid_x": 2304,
    "grid_y": 32,
    "control_second_pass_scale_subtract_v_fma_sites": 4,
    "candidate_second_pass_scale_subtract_v_fma_sites": 0,
    "control_total_v_fma_f32": 56,
    "candidate_total_v_fma_f32_max": 52,
    "global_load_u16": 8,
    "global_load_b128_score_records": 1,
    "global_store_b128_score_records": 1,
    "ds_bpermute_b32": 32,
    "v_exp_f32_e32": 4,
    "code_bytes_max": 4_984,
    "instruction_slots_max": 871,
    "metadata_vgpr_max": 54,
    "runtime_vgpr_max": 56,
    "runtime_sgpr_max": 128,
    "lds_bytes": 0,
    "private_bytes": 0,
    "spill_count": 0,
    "runtime_scratch_bytes": 0,
}
_TRACE_PROTOCOL = {
    "kernel": _KERNEL,
    "require_cached_build": True,
    "compiler_processes_allowed": 0,
    "positive_duration_required": True,
    "local_size": 32,
    "grid_x": 2304,
    "lds_bytes": 0,
    "runtime_scratch_bytes": 0,
    "runtime_vgpr_max": 56,
}
_TIMING_PROTOCOL = {
    "warmups_per_arm": 5,
    "samples_per_arm": 15,
    "launches_per_sample": 5,
    "counter_rotated_modes": True,
    "clocks": ("hip_event", "synchronized_wall"),
    "required_winners": _CANDIDATE_STARTS,
    "weighted_calls": 72,
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
        base_offsets=_tensor(0x71000, (capacity,), "int32"),
        live_counts=_tensor(0x72000, (1,), "int64"),
        token_positions=_tensor(0x73000, (capacity,), "int64"),
        evict_mask=_tensor(0x74000, (capacity,), "bool"),
        row_positions=_tensor(0x75000, (1,), "int64"),
        capacity=capacity,
        storage_dtype="bf16",
    )


def test_h7a_frozen_target_and_h6w_immutability() -> None:
    from hipengine.kernels import hip_gfx1100

    assert _CANDIDATE_STARTS == (256, 384)
    assert _FALLBACK_STARTS == (0, 128)
    assert _CALL_WEIGHTS == {256: 36, 384: 36}
    assert _WORKGROUPS == 2_304
    assert _SCORE_SCRATCH_BYTES == 18_874_368
    assert _VISIBLE_SCORE_REPLAYS == {
        256: 106_334_208,
        384: 148_801_536,
    }
    assert _DYNAMIC_WORK_MODEL == {
        "visible_score_replays_by_start": _VISIBLE_SCORE_REPLAYS,
        "removed_duplicate_scale_multiplies": sum(
            _VISIBLE_SCORE_REPLAYS.values()
        ),
        "logical_bytes_changed": 0,
        "workgroups_changed": 0,
    }
    assert _EXPECTED_PHYSICAL["control_total_v_fma_f32"] == 56
    assert _EXPECTED_PHYSICAL["candidate_total_v_fma_f32_max"] == 52
    assert _EXPECTED_PHYSICAL["metadata_vgpr_max"] == 54
    assert _EXPECTED_PHYSICAL["runtime_vgpr_max"] == 56
    assert _EXPECTED_PHYSICAL["lds_bytes"] == 0
    assert _EXPECTED_PHYSICAL["private_bytes"] == 0
    assert _EXPECTED_PHYSICAL["spill_count"] == 0
    assert _EXPECTED_PHYSICAL["runtime_scratch_bytes"] == 0
    assert _TRACE_PROTOCOL["compiler_processes_allowed"] == 0
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
    required = {
        (start, clock)
        for start in (*_CANDIDATE_STARTS, "weighted_72_call")
        for clock in _TIMING_PROTOCOL["clocks"]
    }
    assert len(required) == 6
    assert (
        hip_gfx1100.LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS
        == _SOURCE_POLICY
    )
    assert _VARIANT not in _SOURCE_POLICY.values()

    module = _module()
    source = Path(module.__file__).with_name("laguna_kv_attention.hip").read_text()
    h6w_kernel = _extract_braced(source, _H6W_KERNEL_DECLARATION)
    h6w_wrapper = _extract_braced(source, _H6W_WRAPPER_DECLARATION)
    assert _sha256_text(h6w_kernel) == _H6W_KERNEL_SHA256
    assert _sha256_text(h6w_wrapper) == _H6W_WRAPPER_SHA256
    h6w_body = h6w_kernel[h6w_kernel.index("{") + 1 : -1]
    assert h6w_body.count("dot * scale") == 2
    assert "replay_values[row_index] = dot;" in h6w_body
    assert "expf(dot * scale - max_scores[row_index])" in h6w_body


def test_h7a_registry_source_abi_and_gfx1151_exclusion() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.registry import KernelKey, is_registered, resolve

    module = _module()
    module.register_laguna_kv_attention_kernels()
    h6w_fn = getattr(module, _H6W_FUNCTION)
    assert resolve(
        backend="hip_gfx1100",
        layer="laguna_attention_prefill",
        quant="bf16",
        variant=_H6W_VARIANT,
    ) is h6w_fn
    assert (
        hip_gfx1100.LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS
        == _SOURCE_POLICY
    )
    assert _VARIANT not in _SOURCE_POLICY.values()
    gfx1151_key = KernelKey(
        "hip_gfx1151", "laguna_attention_prefill", "bf16", _VARIANT
    )
    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(gfx1151_key)

    candidate = _candidate()
    module.register_laguna_kv_attention_kernels(replace=True)
    key = KernelKey(
        "hip_gfx1100", "laguna_attention_prefill", "bf16", _VARIANT
    )
    assert is_registered(key)
    assert not is_registered(gfx1151_key)
    assert resolve(
        backend="hip_gfx1100",
        layer="laguna_attention_prefill",
        quant="bf16",
        variant=_VARIANT,
    ) is candidate
    assert _VARIANT not in _SOURCE_POLICY.values()

    source = Path(module.__file__).with_name("laguna_kv_attention.hip").read_text()
    assert source.count(_SYMBOL) == 1
    assert source.count(_KERNEL) == 2
    candidate_body = _body(
        source,
        f"__global__ __launch_bounds__(32) void {_KERNEL}(",
    )
    assert candidate_body.count("laguna_wave32_sum_128_exact(") == 1
    assert candidate_body.count("key_cache[") == 1
    assert candidate_body.count("value_cache[") == 1
    assert candidate_body.count("laguna_ordered_mul_f32(dot, scale)") == 1
    assert "replay_values[row_index] = scaled_score;" in candidate_body
    assert "max_scores[row_index] = fmaxf(" in candidate_body
    assert "max_scores[row_index], scaled_score" in candidate_body
    assert "expf(score - max_scores[row_index])" in candidate_body
    assert "dot * scale" not in candidate_body
    assert "__shared__" not in candidate_body
    assert "__syncthreads" not in candidate_body
    for metadata_read in ("base_offsets[", "token_positions[", "evict_mask["):
        assert metadata_read not in candidate_body

    wrapper_source = inspect.getsource(candidate)
    assert "score_scratch_ptr" in wrapper_source
    assert "score_scratch_nbytes" in wrapper_source
    assert "parsed_start not in {256, 384}" in wrapper_source
    assert "score_scratch_ptr % 16" in wrapper_source
    assert "18_874_368" in wrapper_source
    assert "scaled-score replay exact SWA requires" in wrapper_source


def test_h7a_strict_preflight_after_h6w_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    h6w_fn = getattr(module, _H6W_FUNCTION)
    h6w_calls: list[tuple[object, ...]] = []

    class FakeFn:
        argtypes = None
        restype = None

        def __init__(self, calls: list[tuple[object, ...]]) -> None:
            self.calls = calls

        def __call__(self, *args: object) -> int:
            self.calls.append(args)
            return 0

    common = (0x81000, 0x82000, 0x83000, 0x84000, 0x85000, 0x86000)
    h6w_library = SimpleNamespace(**{h6w._SYMBOL: FakeFn(h6w_calls)})
    for start in _CANDIDATE_STARTS:
        h6w_fn(
            *common,
            0x87000,
            _swa_spans(),
            _ROWS,
            _QUERY_HEADS,
            _KV_HEADS,
            _HEAD_DIM,
            _HEAD_DIM**-0.5,
            score_scratch_nbytes=_SCORE_SCRATCH_BYTES,
            sliding_window=_CAPACITY,
            start_position=start,
            library=h6w_library,
            runtime=SimpleNamespace(),
        )
    assert len(h6w_calls) == 2

    candidate = _candidate()
    calls: list[tuple[object, ...]] = []
    library = SimpleNamespace(**{_SYMBOL: FakeFn(calls)})

    def launch(*, fake_library: object | None = library, **overrides: object) -> None:
        arguments = {
            "score_scratch_ptr": 0x88000,
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
    assert len(calls) == 2
    for args, start in zip(calls, _CANDIDATE_STARTS, strict=True):
        assert args[6].value == 0x88000
        assert args[12].value == _ROWS
        assert args[13].value == _CAPACITY
        assert args[14].value == _CAPACITY
        assert args[15].value == _QUERY_HEADS
        assert args[16].value == _KV_HEADS
        assert args[17].value == _HEAD_DIM
        assert args[18].value == start

    def fail_build(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid H7A preflight loaded HIP")

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
        {"score_scratch_ptr": 0x88008},
        {"score_scratch_nbytes": _SCORE_SCRATCH_BYTES - 16},
        {"score_scratch_nbytes": _SCORE_SCRATCH_BYTES + 16},
    )
    for invalid in invalid_cases:
        with pytest.raises(ValueError, match="scaled-score replay exact SWA"):
            launch(fake_library=None, **invalid)
    assert len(calls) == 2


def _expected_scaled_records(
    control_bytes: np.ndarray,
    *,
    start_position: int,
) -> np.ndarray:
    control = control_bytes.view(np.float32).reshape(
        _QUERY_ROW_GROUPS,
        _QUERY_HEADS,
        _CAPACITY,
        4,
    )
    expected = control.copy()
    scale = np.float32(_HEAD_DIM**-0.5)
    for row_group in range(_QUERY_ROW_GROUPS):
        query_base = row_group * 4
        context_len = start_position + query_base + 4
        for row_index in range(4):
            visible = start_position + query_base + row_index + 1
            expected[row_group, :, :visible, row_index] = np.multiply(
                control[row_group, :, :visible, row_index],
                scale,
                dtype=np.float32,
            )
            expected[row_group, :, visible:context_len, row_index] = 0.0
    return expected.view(np.uint8).reshape(-1)


@pytest.fixture(scope="module")
def h7a_library():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    return _module().build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )


@pytest.mark.parametrize("start_position", _CANDIDATE_STARTS)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h7a_complete_output_scaled_records_spans_and_lifecycle(
    h7a_library,
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
    h6w_fn = getattr(module, _H6W_FUNCTION)
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
    rng = np.random.default_rng(0x7A00 + start_position)
    total_rows = start_position + _ROWS
    keys = rng.normal(
        0.0, 0.12, size=(total_rows, _KV_HEADS, _HEAD_DIM)
    ).astype(np.float32)
    values = rng.normal(0.0, 0.12, size=keys.shape).astype(np.float32)
    queries = rng.normal(
        0.0, 0.12, size=(_ROWS, _QUERY_HEADS, _HEAD_DIM)
    ).astype(np.float32)
    control_host = np.empty_like(queries)
    candidate_host = np.empty_like(queries)
    control_scratch_host = np.empty(_SCORE_SCRATCH_BYTES, dtype=np.uint8)
    candidate_scratch_host = np.empty(_SCORE_SCRATCH_BYTES, dtype=np.uint8)
    allocations = []
    try:
        key_rows = malloc(keys.nbytes, runtime=runtime)
        value_rows = malloc(values.nbytes, runtime=runtime)
        query_rows = malloc(queries.nbytes, runtime=runtime)
        control_out = malloc(control_host.nbytes, runtime=runtime)
        candidate_out = malloc(candidate_host.nbytes, runtime=runtime)
        control_scratch = malloc(_SCORE_SCRATCH_BYTES, runtime=runtime)
        candidate_scratch = malloc(_SCORE_SCRATCH_BYTES, runtime=runtime)
        allocations.extend(
            (
                key_rows,
                value_rows,
                query_rows,
                control_out,
                candidate_out,
                control_scratch,
                candidate_scratch,
            )
        )
        assert control_scratch.ptr % _SCORE_ALIGNMENT == 0
        assert candidate_scratch.ptr % _SCORE_ALIGNMENT == 0
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
                library=h7a_library,
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
            library=h7a_library,
        )
        state = cache.layer(0)
        before_spans = h6w._span_snapshot(state.spans, runtime)
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
        runtime.memset(control_scratch.ptr, 0xA5, control_scratch.nbytes)
        h6w_fn(
            *common,
            control_out.ptr,
            control_scratch.ptr,
            *suffix,
            score_scratch_nbytes=control_scratch.nbytes,
            sliding_window=_CAPACITY,
            start_position=start_position,
            library=h7a_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(control_host),
            control_out,
            control_host.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(control_scratch_host),
            control_scratch,
            control_scratch_host.nbytes,
            runtime=runtime,
        )
        assert np.isfinite(control_host).all()
        after_control_spans = h6w._span_snapshot(state.spans, runtime)
        for actual, expected in zip(
            after_control_spans, before_spans, strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
        for row, expected in h6w._cpu_rows(
            queries,
            keys,
            values,
            start_position=start_position,
        ).items():
            np.testing.assert_allclose(
                control_host[row], expected, rtol=3e-4, atol=3e-4
            )

        candidate = _candidate()
        runtime.memset(candidate_out.ptr, 0xA5, candidate_out.nbytes)
        runtime.memset(candidate_scratch.ptr, 0xA5, candidate_scratch.nbytes)
        candidate(
            *common,
            candidate_out.ptr,
            candidate_scratch.ptr,
            *suffix,
            score_scratch_nbytes=candidate_scratch.nbytes,
            sliding_window=_CAPACITY,
            start_position=start_position,
            library=h7a_library,
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
            host_array_ptr(candidate_scratch_host),
            candidate_scratch,
            candidate_scratch_host.nbytes,
            runtime=runtime,
        )
        assert np.isfinite(candidate_host).all()
        np.testing.assert_array_equal(candidate_host, control_host)
        h6w._assert_scratch_record_coverage(
            candidate_scratch_host,
            start_position=start_position,
        )
        np.testing.assert_array_equal(
            candidate_scratch_host,
            _expected_scaled_records(
                control_scratch_host,
                start_position=start_position,
            ),
        )
        after_candidate_spans = h6w._span_snapshot(state.spans, runtime)
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
