"""WPF-H6Z exact late-start global qrow4 score/weight replay contract."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from hipengine.core.device import Device
from hipengine.core.tensor import Tensor
from hipengine.kvcache import KVLiveSpans
from hipengine.loading.laguna_gguf import FULL_ATTENTION

_CANDIDATE_STARTS = (256, 384)
_FALLBACK_STARTS = (0, 128)
_ALL_STARTS = (*_FALLBACK_STARTS, *_CANDIDATE_STARTS)
_CALL_WEIGHTS = {256: 12, 384: 12}
_VARIANT = (
    "global_context_rows_qrow4_dense_initial_"
    "global_score_weight_replay_exact_spans"
)
_FUNCTION = (
    "laguna_global_attention_prefill_qrow4_dense_initial_"
    "global_score_weight_replay_exact_bf16_spans"
)
_SYMBOL = (
    "hipengine_laguna_global_attention_prefill_qrow4_dense_initial_"
    "global_score_weight_replay_exact_bf16_spans"
)
_KERNEL = (
    "laguna_global_attention_prefill_qrow4_dense_initial_"
    "global_score_weight_replay_exact_bf16_kernel"
)
_H6N_VARIANT = "global_context_rows_dense_initial_fixed512_cached_exact_spans"
_H6N_FUNCTION = (
    "laguna_global_attention_prefill_"
    "dense_initial_fixed512_cached_exact_bf16_spans"
)
_H6N_SYMBOL = (
    "hipengine_laguna_global_attention_prefill_"
    "dense_initial_fixed512_cached_exact_bf16_spans"
)
_H6N_KERNEL_DECLARATION = (
    "__global__ void "
    "laguna_global_attention_prefill_"
    "dense_initial_fixed512_cached_exact_bf16_kernel("
)
_H6N_WRAPPER_DECLARATION = f'extern "C" int {_H6N_SYMBOL}('
_H6W_VARIANT = (
    "swa_context_rows_qrow4_dense_initial_global_score_replay_exact_spans"
)
_H6W_FUNCTION = (
    "laguna_swa_attention_prefill_qrow4_"
    "dense_initial_global_score_replay_exact_bf16_spans"
)
_H6W_KERNEL_DECLARATION = (
    "__global__ __launch_bounds__(32) void "
    "laguna_swa_attention_prefill_qrow4_"
    "dense_initial_global_score_replay_exact_bf16_kernel("
)
_H6W_WRAPPER_DECLARATION = (
    'extern "C" int '
    "hipengine_laguna_swa_attention_prefill_qrow4_"
    "dense_initial_global_score_replay_exact_bf16_spans("
)
_REDUCTION_DECLARATION = (
    "__device__ __forceinline__ float laguna_wave32_sum_128_exact("
)
_REDUCTION_SHA256 = (
    "5aa8e6ef384bb15d41a4942e65bc01b5abe20008b89f2c8ad3f55700ead456aa"
)
_H6N_KERNEL_SHA256 = (
    "53cba40dc70d52a7337da6d540983aaee5fd468e088b3064d28a4a3b9dbaa9e1"
)
_H6N_WRAPPER_SHA256 = (
    "b1e17114ee64d1042e2a74c862f875da80c3f01d23023d450ba76d39796512a4"
)
_H6N_PYTHON_SHA256 = (
    "95c3bf0d9d0f7131f40d8762d0bd4e90451b79dd7d0c677a9a1e1445b6bfd436"
)
_H6W_KERNEL_SHA256 = (
    "d025a3c73c499c75b2dc0444ae4fdd479cb72729eaad2b1e1aa6e38491748aee"
)
_H6W_WRAPPER_SHA256 = (
    "01b12957a7fc404486d96928918a371169648835be60b08c1b477dd153dddde9"
)
_H6W_PYTHON_SHA256 = (
    "b43dc2db7baa7b6bf2fe15653274a397ba6ba582ef40c51ec3e804fbcefac912"
)
_DENSE_ROLE = "global_m128_c4096_first_fill_exact"
_SWA_ROLE = "swa_qrow4_m128_c512_no_wrap_exact"
_H6A_SWA_VARIANT = "swa_context_rows_qrow4_dense_initial_cached_exact_spans"
_PRODUCTION_POLICY = {
    _DENSE_ROLE: _H6N_VARIANT,
    _SWA_ROLE: _H6W_VARIANT,
}
_H6A_ROLLBACK_POLICY = {
    _DENSE_ROLE: _H6N_VARIANT,
    _SWA_ROLE: _H6A_SWA_VARIANT,
}
_ROWS = 128
_QUERY_HEADS = 48
_KV_HEADS = 8
_HEAD_DIM = 128
_CAPACITY = 4096
_BLOCK_SIZE = 256
_SCORE_CAPACITY = 512
_QUERY_ROW_GROUPS = _ROWS // 4
_WORKGROUPS_PER_CALL = _QUERY_HEADS * _QUERY_ROW_GROUPS
_SCORE_RECORD_BYTES = 16
_SCORE_ALIGNMENT = 16
_SCORE_SCRATCH_BYTES = (
    _WORKGROUPS_PER_CALL * _SCORE_CAPACITY * _SCORE_RECORD_BYTES
)
_H6W_SCORE_SCRATCH_BYTES = 18_874_368
_EXISTING_WORKSPACE_BYTES = 161_120_256
_CONTROL_WEIGHTED_MS = {
    256: 9.912082000000002,
    384: 13.809434,
}
_CANDIDATE_CONTEXT_ITERATIONS_PER_CALL = {
    start: sum(
        start + row_group * 4 + 4 for row_group in range(_QUERY_ROW_GROUPS)
    )
    * _QUERY_HEADS
    for start in _CANDIDATE_STARTS
}
_CONTROL_CONTEXT_ITERATIONS_PER_CALL = {
    start: sum(start + row + 1 for row in range(_ROWS)) * _QUERY_HEADS
    for start in _CANDIDATE_STARTS
}
_DYNAMIC_WORK_MODEL = {
    "control_workgroups": 147_456,
    "candidate_workgroups": 36_864,
    "workgroups_removed": 110_592,
    "control_kv_bytes": 29_028_777_984,
    "candidate_kv_bytes": 7_285_506_048,
    "candidate_record_bytes": 910_688_256,
    "candidate_kv_plus_record_bytes": 8_196_194_304,
    "logical_bytes_removed": 20_832_583_680,
}
_EXPECTED_PHYSICAL = {
    "local_size": 32,
    "grid_x": 1536,
    "grid_y": 32,
    "global_store_b128_score_records": 1,
    "global_load_b128_scores_for_denominator": 1,
    "global_store_b128_weight_records": 1,
    "global_load_b128_weights_for_pv": 1,
    "lds_bytes": 0,
    "block_barriers": 0,
    "private_bytes": 0,
    "spill_count": 0,
    "runtime_scratch_bytes": 0,
    "metadata_vgpr_max": 96,
    "runtime_vgpr_max": 96,
}
_TRACE_PROTOCOL = {
    "kernel": _KERNEL,
    "require_cached_build": True,
    "compiler_processes_allowed": 0,
    "positive_duration_required": True,
    "local_size": 32,
    "grid_x": 1536,
    "lds_bytes": 0,
    "runtime_scratch_bytes": 0,
    "runtime_vgpr_max": 96,
}
_QUALIFICATION_TOPOLOGY = {
    "H6N": 24,
    "H6Z": 24,
    "H6A": 72,
    "H6W": 72,
}
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


def _global_spans(capacity: int = _CAPACITY) -> KVLiveSpans:
    blocks = (capacity + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    return KVLiveSpans.paged_dense(
        block_table=_tensor(0x21000, (blocks,), "int32"),
        live_counts=_tensor(0x22000, (1,), "int64"),
        token_positions=_tensor(0x23000, (capacity,), "int64"),
        evict_mask=_tensor(0x24000, (capacity,), "bool"),
        row_positions=_tensor(0x25000, (1,), "int64"),
        capacity=capacity,
        block_size=_BLOCK_SIZE,
        storage_dtype="bf16",
        span_role="prefill",
    )


def test_h6z_frozen_leaf_physical_timing_and_rejection_contract() -> None:
    assert _CANDIDATE_STARTS == (256, 384)
    assert _FALLBACK_STARTS == (0, 128)
    assert _CALL_WEIGHTS == {256: 12, 384: 12}
    assert sum(_CALL_WEIGHTS.values()) == 24
    assert _QUERY_ROW_GROUPS == 32
    assert _WORKGROUPS_PER_CALL == 1_536
    assert _SCORE_SCRATCH_BYTES == 12_582_912
    assert _SCORE_SCRATCH_BYTES * 3 == _H6W_SCORE_SCRATCH_BYTES * 2
    assert _SCORE_SCRATCH_BYTES < _H6W_SCORE_SCRATCH_BYTES
    assert _H6W_SCORE_SCRATCH_BYTES < _EXISTING_WORKSPACE_BYTES
    assert _SCORE_SCRATCH_BYTES % _SCORE_ALIGNMENT == 0
    assert _CANDIDATE_CONTEXT_ITERATIONS_PER_CALL == {
        256: 494_592,
        384: 691_200,
    }
    assert _CONTROL_CONTEXT_ITERATIONS_PER_CALL == {
        256: 1_969_152,
        384: 2_755_584,
    }
    control_iterations = sum(
        _CALL_WEIGHTS[start] * _CONTROL_CONTEXT_ITERATIONS_PER_CALL[start]
        for start in _CANDIDATE_STARTS
    )
    candidate_iterations = sum(
        _CALL_WEIGHTS[start] * _CANDIDATE_CONTEXT_ITERATIONS_PER_CALL[start]
        for start in _CANDIDATE_STARTS
    )
    control_workgroups = sum(_CALL_WEIGHTS.values()) * _QUERY_HEADS * _ROWS
    candidate_workgroups = (
        sum(_CALL_WEIGHTS.values()) * _WORKGROUPS_PER_CALL
    )
    assert _DYNAMIC_WORK_MODEL == {
        "control_workgroups": control_workgroups,
        "candidate_workgroups": candidate_workgroups,
        "workgroups_removed": control_workgroups - candidate_workgroups,
        "control_kv_bytes": control_iterations * 2 * _HEAD_DIM * 2,
        "candidate_kv_bytes": candidate_iterations * 2 * _HEAD_DIM * 2,
        "candidate_record_bytes": candidate_iterations
        * 4
        * _SCORE_RECORD_BYTES,
        "candidate_kv_plus_record_bytes": candidate_iterations
        * (2 * _HEAD_DIM * 2 + 4 * _SCORE_RECORD_BYTES),
        "logical_bytes_removed": control_iterations * 2 * _HEAD_DIM * 2
        - candidate_iterations
        * (2 * _HEAD_DIM * 2 + 4 * _SCORE_RECORD_BYTES),
    }
    assert candidate_workgroups * 4 == control_workgroups
    assert _DYNAMIC_WORK_MODEL["candidate_kv_plus_record_bytes"] / (
        _DYNAMIC_WORK_MODEL["control_kv_bytes"]
    ) == pytest.approx(0.2823471971066907)
    assert _EXPECTED_PHYSICAL == {
        "local_size": 32,
        "grid_x": 1536,
        "grid_y": 32,
        "global_store_b128_score_records": 1,
        "global_load_b128_scores_for_denominator": 1,
        "global_store_b128_weight_records": 1,
        "global_load_b128_weights_for_pv": 1,
        "lds_bytes": 0,
        "block_barriers": 0,
        "private_bytes": 0,
        "spill_count": 0,
        "runtime_scratch_bytes": 0,
        "metadata_vgpr_max": 96,
        "runtime_vgpr_max": 96,
    }
    assert _TRACE_PROTOCOL == {
        "kernel": _KERNEL,
        "require_cached_build": True,
        "compiler_processes_allowed": 0,
        "positive_duration_required": True,
        "local_size": 32,
        "grid_x": 1536,
        "lds_bytes": 0,
        "runtime_scratch_bytes": 0,
        "runtime_vgpr_max": 96,
    }
    assert _QUALIFICATION_TOPOLOGY == {
        "H6N": 24,
        "H6Z": 24,
        "H6A": 72,
        "H6W": 72,
    }
    assert sum(_QUALIFICATION_TOPOLOGY.values()) == 192
    assert sum(_CONTROL_WEIGHTED_MS.values()) == pytest.approx(23.721516)
    assert _TIMING_PROTOCOL == {
        "warmups_per_arm": 5,
        "samples_per_arm": 15,
        "launches_per_sample": 5,
        "counter_rotated_modes": True,
        "clocks": ("hip_event", "synchronized_wall"),
        "required_winners": (256, 384),
        "weighted_calls": 24,
        "no_follow_up_tuning_on_miss": True,
    }
    required_results = {
        (start, clock)
        for start in (*_CANDIDATE_STARTS, "weighted_24_call")
        for clock in _TIMING_PROTOCOL["clocks"]
    }
    assert len(required_results) == 6
    # Admission is conjunctive. Any correctness, physical, trace, per-start, or
    # weighted miss deletes every H6Z surface without tuning or a second screen.


def test_h6z_registry_abi_source_policy_and_backend_boundary() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.backends import load_backend_kernel_package
    from hipengine.kernels.registry import KernelKey, is_registered, resolve

    module = _module()
    module.register_laguna_kv_attention_kernels()
    h6n = getattr(module, _H6N_FUNCTION)
    h6w = getattr(module, _H6W_FUNCTION)
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant=_H6N_VARIANT,
        )
        is h6n
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_attention_prefill",
            quant="bf16",
            variant=_H6W_VARIANT,
        )
        is h6w
    )
    assert (
        hip_gfx1100.LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS
        == _PRODUCTION_POLICY
    )
    assert (
        hip_gfx1100.LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6W_ROLE_VARIANTS
        == _PRODUCTION_POLICY
    )
    assert (
        hip_gfx1100.LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6A_ROLE_VARIANTS
        == _H6A_ROLLBACK_POLICY
    )
    assert _VARIANT not in _PRODUCTION_POLICY.values()

    source = Path(module.__file__).with_name("laguna_kv_attention.hip").read_text()
    reduction = _extract_braced(source, _REDUCTION_DECLARATION)
    h6n_kernel = _extract_braced(source, _H6N_KERNEL_DECLARATION)
    h6n_wrapper = _extract_braced(source, _H6N_WRAPPER_DECLARATION)
    h6w_kernel = _extract_braced(source, _H6W_KERNEL_DECLARATION)
    h6w_wrapper = _extract_braced(source, _H6W_WRAPPER_DECLARATION)
    assert _sha256_text(reduction) == _REDUCTION_SHA256
    assert _sha256_text(h6n_kernel) == _H6N_KERNEL_SHA256
    assert _sha256_text(h6n_wrapper) == _H6N_WRAPPER_SHA256
    assert _sha256_text(inspect.getsource(h6n)) == _H6N_PYTHON_SHA256
    assert _sha256_text(h6w_kernel) == _H6W_KERNEL_SHA256
    assert _sha256_text(h6w_wrapper) == _H6W_WRAPPER_SHA256
    assert _sha256_text(inspect.getsource(h6w)) == _H6W_PYTHON_SHA256
    assert h6n_kernel.count("for (int offset = warpSize / 2;") == 3
    assert h6n_kernel.count("token = warp_id;") == 2
    assert h6n_kernel.count("token = 0;") == 1
    assert "float total_sum = warp_buf[0];" in h6n_kernel
    assert "for (int warp = 1; warp < num_warps; ++warp)" in h6n_kernel
    assert "scores[token] = score;" in h6n_kernel
    assert "scores[token] = weight;" in h6n_kernel
    assert "const float weight = scores[token] * inv_denom;" in h6n_kernel

    candidate_key = KernelKey(
        "hip_gfx1100", "laguna_attention_prefill", "bf16", _VARIANT
    )
    gfx1151_key = KernelKey(
        "hip_gfx1151", "laguna_attention_prefill", "bf16", _VARIANT
    )
    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(gfx1151_key)

    # Intentional RED only after H6N/H6W source and package policy are frozen.
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
    assert "constexpr int kScoreCapacity = 512;" in candidate_body
    assert "float4* score_records" in candidate_body
    assert "__shared__" not in candidate_body
    assert "__syncthreads" not in candidate_body
    assert candidate_body.count("key_cache[") == 1
    assert candidate_body.count("value_cache[") == 1
    assert candidate_body.count("expf(") == 1
    assert "logical_slot = lane;" in candidate_body
    assert "logical_slot += 8" in candidate_body
    assert "source_lane = 1; source_lane < 8; ++source_lane" in candidate_body
    assert "float4 score_record" in candidate_body
    assert "float4 weight_record" in candidate_body
    assert "safe_denominators" in candidate_body
    assert "normalized_weights" in candidate_body
    assert "laguna_ordered_mul_f32" in candidate_body
    for metadata_read in ("base_offsets[", "token_positions[", "evict_mask["):
        assert metadata_read not in candidate_body

    wrapper_source = inspect.getsource(candidate)
    assert "score_scratch_ptr" in wrapper_source
    assert "score_scratch_nbytes" in wrapper_source
    assert "parsed_start not in {256, 384}" in wrapper_source
    assert "score_scratch_ptr % 16" in wrapper_source
    assert "12_582_912" in wrapper_source
    assert "global score/weight replay exact requires" in wrapper_source


def test_h6z_strict_late_start_and_caller_plane_preflight_before_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    h6n = getattr(module, _H6N_FUNCTION)
    h6n_calls: list[tuple[object, ...]] = []

    class FakeFn:
        argtypes = None
        restype = None

        def __init__(self, calls: list[tuple[object, ...]]) -> None:
            self.calls = calls

        def __call__(self, *args: object) -> int:
            self.calls.append(args)
            return 0

    h6n_library = SimpleNamespace(**{_H6N_SYMBOL: FakeFn(h6n_calls)})
    common = (0x6000, 0x7000, 0x8000, 0x9000, 0xA000, 0xB000)
    for start in _ALL_STARTS:
        h6n(
            *common,
            _global_spans(),
            _ROWS,
            _CAPACITY,
            _QUERY_HEADS,
            _KV_HEADS,
            _HEAD_DIM,
            _HEAD_DIM**-0.5,
            start_position=start,
            library=h6n_library,
            runtime=SimpleNamespace(),
        )
    assert len(h6n_calls) == len(_ALL_STARTS)

    # Intentional RED only after retained H6N accepts all four frozen starts.
    candidate = _candidate()
    calls: list[tuple[object, ...]] = []
    library = SimpleNamespace(**{_SYMBOL: FakeFn(calls)})

    def launch(*, fake_library: object | None = library, **overrides: object) -> None:
        arguments = {
            "score_scratch_ptr": 0xC000,
            "score_scratch_nbytes": _SCORE_SCRATCH_BYTES,
            "spans": _global_spans(),
            "rows": _ROWS,
            "max_context_len": _CAPACITY,
            "num_q_heads": _QUERY_HEADS,
            "num_kv_heads": _KV_HEADS,
            "head_dim": _HEAD_DIM,
            "start_position": 256,
        }
        arguments.update(overrides)
        candidate(
            *common,
            arguments.pop("score_scratch_ptr"),
            arguments.pop("spans"),
            arguments.pop("rows"),
            arguments.pop("max_context_len"),
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
        assert args[14].value == _BLOCK_SIZE
        assert args[15].value == _CAPACITY // _BLOCK_SIZE
        assert args[16].value == _QUERY_HEADS
        assert args[17].value == _KV_HEADS
        assert args[18].value == _HEAD_DIM
        assert args[19].value == start

    def fail_build(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid H6Z preflight loaded HIP")

    monkeypatch.setattr(module, "build_laguna_kv_attention", fail_build)
    invalid_cases = (
        {"rows": 127},
        {"spans": _global_spans(2048), "max_context_len": 2048},
        {"max_context_len": 2048},
        {"num_q_heads": 72},
        {"num_kv_heads": 4},
        {"head_dim": 64},
        {"start_position": None},
        {"start_position": 0},
        {"start_position": 128},
        {"start_position": 64},
        {"start_position": 512},
        {"score_scratch_ptr": 0},
        {"score_scratch_ptr": 0xC008},
        {"score_scratch_nbytes": _SCORE_SCRATCH_BYTES - 16},
        {"score_scratch_nbytes": _SCORE_SCRATCH_BYTES + 16},
        {"score_scratch_nbytes": _H6W_SCORE_SCRATCH_BYTES},
    )
    for invalid in invalid_cases:
        with pytest.raises(
            ValueError,
            match="global score/weight replay exact requires",
        ):
            launch(fake_library=None, **invalid)
    assert len(calls) == len(_CANDIDATE_STARTS)


def _copy_metadata(tensor: Tensor, runtime: Any) -> np.ndarray:
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


def _span_snapshot(spans: KVLiveSpans, runtime: Any) -> tuple[np.ndarray, ...]:
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


def _assert_output_overwrites_poison(output: np.ndarray) -> None:
    output_bytes = output.view(np.uint8).reshape(-1, np.dtype(np.float32).itemsize)
    assert not np.any(np.all(output_bytes == 0xA5, axis=1))


def _assert_weight_record_coverage(
    scratch: np.ndarray,
    *,
    start_position: int,
) -> None:
    records = scratch.reshape(
        _QUERY_ROW_GROUPS,
        _QUERY_HEADS,
        _SCORE_CAPACITY,
        _SCORE_RECORD_BYTES,
    )
    poison = np.full(_SCORE_RECORD_BYTES, 0xA5, dtype=np.uint8)
    for row_group in range(_QUERY_ROW_GROUPS):
        context_len = start_position + row_group * 4 + 4
        prefix = records[row_group, :, :context_len, :]
        flat_prefix = prefix.reshape(-1, _SCORE_RECORD_BYTES)
        assert not np.any(np.all(flat_prefix == poison, axis=1))
        prefix_f32 = np.ascontiguousarray(prefix).view(np.float32)
        assert np.isfinite(prefix_f32).all()
        assert np.all(prefix_f32 >= 0.0)
        suffix = records[row_group, :, context_len:, :]
        assert np.all(suffix == 0xA5)


def _run_attention(
    fn: Any,
    library: Any,
    *,
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    start_position: int,
    use_score_scratch: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
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

    runtime = get_hip_runtime()
    config = SimpleNamespace(
        block_count=1,
        layer_types=(FULL_ATTENTION,),
        head_counts=(_QUERY_HEADS,),
        head_count_kv=_KV_HEADS,
        key_length=_HEAD_DIM,
        value_length=_HEAD_DIM,
        sliding_window=512,
    )
    baseline = memory_stats()
    cache = allocate_laguna_kv_cache(
        config,
        context_length=_CAPACITY,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    output_host = np.empty_like(queries)
    scratch_host = (
        np.empty(_SCORE_SCRATCH_BYTES, dtype=np.uint8)
        if use_score_scratch
        else None
    )
    allocations = []
    try:
        key_rows = malloc(keys.nbytes, runtime=runtime)
        value_rows = malloc(values.nbytes, runtime=runtime)
        query_rows = malloc(queries.nbytes, runtime=runtime)
        output = malloc(output_host.nbytes, runtime=runtime)
        allocations.extend((key_rows, value_rows, query_rows, output))
        score_scratch = None
        if use_score_scratch:
            score_scratch = malloc(_SCORE_SCRATCH_BYTES, runtime=runtime)
            allocations.append(score_scratch)
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
        runtime.memset(output.ptr, 0xA5, output.nbytes)
        if score_scratch is not None:
            runtime.memset(score_scratch.ptr, 0xA5, score_scratch.nbytes)
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
        total_rows = start_position + _ROWS
        cache.prepare_rows(tuple(range(start_position, total_rows)))
        row_nbytes = _KV_HEADS * _HEAD_DIM * np.dtype(np.float32).itemsize
        current_key_ptr = key_rows.ptr + start_position * row_nbytes
        current_value_ptr = value_rows.ptr + start_position * row_nbytes
        cache.append_rows(
            0,
            current_key_ptr,
            current_value_ptr,
            _ROWS,
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
            output.ptr,
        )
        suffix = (
            state.spans,
            _ROWS,
            _CAPACITY,
            _QUERY_HEADS,
            _KV_HEADS,
            _HEAD_DIM,
            _HEAD_DIM**-0.5,
        )
        if score_scratch is None:
            fn(
                *common,
                *suffix,
                start_position=start_position,
                library=library,
                runtime=runtime,
            )
        else:
            fn(
                *common,
                score_scratch.ptr,
                *suffix,
                score_scratch_nbytes=score_scratch.nbytes,
                start_position=start_position,
                library=library,
                runtime=runtime,
            )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(output_host),
            output,
            output_host.nbytes,
            runtime=runtime,
        )
        if score_scratch is not None and scratch_host is not None:
            copy_device_to_host(
                host_array_ptr(scratch_host),
                score_scratch,
                scratch_host.nbytes,
                runtime=runtime,
            )
        after_spans = _span_snapshot(state.spans, runtime)
        for actual, expected in zip(after_spans, before_spans, strict=True):
            np.testing.assert_array_equal(actual, expected)
        cache.discard_rows()
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        cache.free()
    after = memory_stats()
    assert after["current_allocated_bytes"] == baseline["current_allocated_bytes"]
    assert after["active_allocations"] == baseline["active_allocations"]
    return output_host, scratch_host


@pytest.fixture(scope="module")
def h6n_library():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    return _module().build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )


@pytest.mark.parametrize("start_position", _CANDIDATE_STARTS)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h6z_complete_output_weight_records_cpu_spans_and_lifecycle(
    h6n_library: Any,
    start_position: int,
) -> None:
    module = _module()
    h6n = getattr(module, _H6N_FUNCTION)
    rng = np.random.default_rng(0x6A80 + start_position)
    total_rows = start_position + _ROWS
    keys = rng.normal(
        0.0,
        0.12,
        size=(total_rows, _KV_HEADS, _HEAD_DIM),
    ).astype(np.float32)
    values = rng.normal(0.0, 0.12, size=keys.shape).astype(np.float32)
    queries = rng.normal(
        0.0,
        0.12,
        size=(_ROWS, _QUERY_HEADS, _HEAD_DIM),
    ).astype(np.float32)

    expected, control_scratch = _run_attention(
        h6n,
        h6n_library,
        queries=queries,
        keys=keys,
        values=values,
        start_position=start_position,
        use_score_scratch=False,
    )
    assert control_scratch is None
    assert np.isfinite(expected).all()
    _assert_output_overwrites_poison(expected)
    for row, cpu_expected in _cpu_rows(
        queries,
        keys,
        values,
        start_position=start_position,
    ).items():
        np.testing.assert_allclose(
            expected[row],
            cpu_expected,
            rtol=3e-4,
            atol=3e-4,
        )

    # Intentional RED only after complete H6N bytes, sampled CPU rows, all five
    # spans, poison overwrite, finiteness, and allocation lifecycle pass.
    candidate = _candidate()
    actual, scratch = _run_attention(
        candidate,
        h6n_library,
        queries=queries,
        keys=keys,
        values=values,
        start_position=start_position,
        use_score_scratch=True,
    )
    assert scratch is not None
    assert np.isfinite(actual).all()
    _assert_output_overwrites_poison(actual)
    np.testing.assert_array_equal(actual, expected)
    _assert_weight_record_coverage(scratch, start_position=start_position)
