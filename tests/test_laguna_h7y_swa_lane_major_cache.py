"""WPF-H7Y exact H6W lane-major BF16 K/V mirror-load RED."""

from __future__ import annotations

import ast
import ctypes
import hashlib
import importlib
import importlib.util
import inspect
import json
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
    "laguna_h7y_h6w_helper", _HELPER_PATH
)
assert _HELPER_SPEC is not None and _HELPER_SPEC.loader is not None
h6w = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(h6w)

_ROOT = Path(__file__).resolve().parents[1]
_TARGET_ARTIFACT = _ROOT / (
    "benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-"
    "post-h7x-swa-lane-major-cache-target.json"
)
_TARGET_ARTIFACT_SHA256 = (
    "0461a3cb560cfe1226d5eaeb74afa1de09cf010dcc46940913663a90b54eae73"
)
_CANDIDATE_STARTS = (256, 384)
_FALLBACK_STARTS = (0, 128)
_CALL_WEIGHTS = {256: 36, 384: 36}
_VARIANT = (
    "swa_context_rows_qrow4_dense_initial_lane_major_"
    "global_score_replay_exact_spans"
)
_FUNCTION = (
    "laguna_swa_attention_prefill_qrow4_dense_initial_lane_major_"
    "global_score_replay_exact_bf16_spans"
)
_SYMBOL = "hipengine_" + _FUNCTION
_KERNEL = (
    "laguna_swa_attention_prefill_qrow4_dense_initial_lane_major_"
    "global_score_replay_exact_bf16_kernel"
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
_H6W_PYTHON_SHA256 = (
    "b43dc2db7baa7b6bf2fe15653274a397ba6ba582ef40c51ec3e804fbcefac912"
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
_CONTEXT_SLOTS_PER_LAYER = {256: 741_888, 384: 1_036_800}
_MIRROR_BYTES_PER_TENSOR_PER_LAYER = _CAPACITY * _KV_HEADS * _HEAD_DIM * 2
_MIRROR_BYTES_PER_TENSOR_REQUEST = 36 * _MIRROR_BYTES_PER_TENSOR_PER_LAYER
_MIRROR_BYTES_KV_REQUEST = 2 * _MIRROR_BYTES_PER_TENSOR_REQUEST
_DYNAMIC_WORK_MODEL = {
    "slots_per_request_per_pass": 64_032_768,
    "current_dynamic_global_load_issue_slots": 512_262_144,
    "candidate_dynamic_global_load_issue_slots": 128_065_536,
    "removed_dynamic_global_load_issue_slots": 384_196_608,
    "current_dynamic_vmcnt_wait_issue_slots": 512_262_144,
    "candidate_dynamic_vmcnt_wait_issue_slots": 128_065_536,
    "removed_dynamic_vmcnt_wait_issue_slots": 384_196_608,
    "global_load_issue_reduction_percent": 75.0,
    "logical_attention_payload_bytes_unchanged": 32_784_777_216,
    "qualified_runtime_mirror_bytes_kv": 75_497_472,
    "request_dispatch_delta_standalone": 0,
    "standalone_new_runtime_allocation_bytes": 0,
}
_EXPECTED_PHYSICAL = {
    "local_size": 32,
    "wavefront_size": 32,
    "grid": (2304, 32),
    "global_load_b64": 2,
    "global_load_u16": 0,
    "vmcnt_wait_sites_max": 2,
    "global_load_b128_score_records": 1,
    "global_store_b128_score_records": 1,
    "ds_bpermute_b32": 32,
    "v_exp_f32": 4,
    "v_fma_f32": 56,
    "v_fmac_f32": 41,
    "global_store_b32_output": 16,
    "code_bytes_max": 5_200,
    "instruction_slots_max": 900,
    "metadata_vgpr_max": 56,
    "runtime_vgpr_max": 56,
    "metadata_sgpr_max": 48,
    "lds_bytes": 0,
    "private_bytes": 0,
    "spill_count": 0,
    "runtime_scratch_bytes": 0,
    "block_barriers": 0,
}
_TRACE_PROTOCOL = {
    "H7Y": 72,
    "H6A": 72,
    "H6N": 24,
    "H6Z": 24,
    "application_dispatches": 2_286,
    "queues": 1,
    "streams": 1,
    "compiler_processes_allowed": 0,
    "positive_duration_required": True,
}
_TIMING_PROTOCOL = {
    "warmups_per_arm": 5,
    "samples_per_arm": 15,
    "launches_per_sample": 5,
    "counter_rotated_modes": True,
    "clocks": ("hip_event", "synchronized_wall"),
    "required_winners": _CANDIDATE_STARTS,
    "weighted_calls": 72,
    "mirrors_prepared_outside_timing": True,
    "no_follow_up_tuning_on_miss": True,
    "no_subset_salvage": True,
}
_AFFECTED_SOURCE_PATHS = {
    "hipengine/kernels/hip_gfx1100/__init__.py",
    "hipengine/kernels/hip_gfx1100/attention/laguna_kv_attention.hip",
    "hipengine/kernels/hip_gfx1100/attention/laguna_kv.py",
    "hipengine/kernels/hip_gfx1151/__init__.py",
    "hipengine/runtime/laguna_kv.py",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


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


def _python_function(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    lines = source.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def _tensor(ptr: int, shape: tuple[int, ...], dtype: str) -> Tensor:
    return Tensor.from_handle(ptr, shape, dtype, Device("hip", 0))


def _swa_spans(capacity: int = _CAPACITY) -> KVLiveSpans:
    return KVLiveSpans.sliding_ring(
        base_offsets=_tensor(0xA1000, (capacity,), "int32"),
        live_counts=_tensor(0xA2000, (1,), "int64"),
        token_positions=_tensor(0xA3000, (capacity,), "int64"),
        evict_mask=_tensor(0xA4000, (capacity,), "bool"),
        row_positions=_tensor(0xA5000, (1,), "int64"),
        capacity=capacity,
        storage_dtype="bf16",
    )


def _lane_major_records(natural: np.ndarray) -> np.ndarray:
    if natural.shape[-1] != _HEAD_DIM:
        raise ValueError("lane-major mirror requires head_dim 128")
    return np.ascontiguousarray(
        natural.reshape(*natural.shape[:-1], 4, 32).swapaxes(-2, -1)
    )


def _natural_records(lane_major: np.ndarray) -> np.ndarray:
    if lane_major.shape[-2:] != (32, 4):
        raise ValueError("lane-major mirror requires [lane32][part4]")
    return np.ascontiguousarray(
        lane_major.swapaxes(-2, -1).reshape(*lane_major.shape[:-2], _HEAD_DIM)
    )


def test_h7y_frozen_target_artifact_work_model_and_admission_protocol() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.runtime import laguna_kv as runtime_module

    assert _sha256(_TARGET_ARTIFACT) == _TARGET_ARTIFACT_SHA256
    artifact = json.loads(_TARGET_ARTIFACT.read_text())
    assert artifact["status"] == "accepted_target_only_no_candidate_run"
    assert artifact["kind"].endswith("h6w_lane_major_bf16_kv_cache_target")
    assert artifact["decision"] == {
        "candidate_built": False,
        "candidate_executed": False,
        "candidate_implemented": False,
        "next_action": (
            "commit this target-only packet, then freeze RED before any H7Y "
            "implementation, build, or execution"
        ),
        "production_changed": False,
        "speed_result_exists": False,
        "target_selected": True,
    }
    assert artifact["production"]["retained_wall_tok_s"] == pytest.approx(
        437.1892736544479
    )
    assert artifact["production"]["kernel_sum_ms"] == pytest.approx(
        1_153.3470579999982
    )
    assert artifact["production"]["dispatches"] == 2_286
    assert artifact["current_h6w"]["calls"] == 72
    assert artifact["current_h6w"]["median_ms"] == pytest.approx(
        62.656442999999996
    )
    assert artifact["current_h6w"]["physical"]["metadata_vgpr"] == 54
    assert artifact["current_h6w"]["runtime_resources"]["vgpr"] == 56
    target = artifact["target"]
    assert target["candidate_present"] is False
    assert target["candidate_built"] is False
    assert target["candidate_executed"] is False
    assert target["speed_result_exists"] is False
    for key, expected in _DYNAMIC_WORK_MODEL.items():
        assert target[key] == pytest.approx(expected)

    assert _CANDIDATE_STARTS == (256, 384)
    assert _FALLBACK_STARTS == (0, 128)
    assert _CALL_WEIGHTS == {256: 36, 384: 36}
    assert _WORKGROUPS == 2_304
    assert _SCORE_SCRATCH_BYTES == 18_874_368
    slots = sum(
        _CALL_WEIGHTS[start] * _CONTEXT_SLOTS_PER_LAYER[start]
        for start in _CANDIDATE_STARTS
    )
    assert slots == _DYNAMIC_WORK_MODEL["slots_per_request_per_pass"]
    assert _MIRROR_BYTES_PER_TENSOR_PER_LAYER == 1_048_576
    assert _MIRROR_BYTES_PER_TENSOR_REQUEST == 37_748_736
    assert _MIRROR_BYTES_KV_REQUEST == 75_497_472
    assert _DYNAMIC_WORK_MODEL["current_dynamic_global_load_issue_slots"] == (
        slots * 8
    )
    assert _DYNAMIC_WORK_MODEL["candidate_dynamic_global_load_issue_slots"] == (
        slots * 2
    )
    assert _DYNAMIC_WORK_MODEL["removed_dynamic_global_load_issue_slots"] == (
        slots * 6
    )
    assert _DYNAMIC_WORK_MODEL["logical_attention_payload_bytes_unchanged"] == (
        slots * 2 * _HEAD_DIM * 2
    )
    assert 100.0 * _MIRROR_BYTES_KV_REQUEST / (
        _DYNAMIC_WORK_MODEL["logical_attention_payload_bytes_unchanged"]
    ) == pytest.approx(target["extra_writer_bytes_over_h7y_attention_payload_percent"])
    natural = np.arange(2 * 3 * _HEAD_DIM, dtype=np.uint16).reshape(
        2, 3, _HEAD_DIM
    )
    lane_major = _lane_major_records(natural)
    assert lane_major.shape == (2, 3, 32, 4)
    assert lane_major.flags.c_contiguous
    for lane in range(32):
        for part in range(4):
            np.testing.assert_array_equal(
                lane_major[..., lane, part],
                natural[..., lane + part * 32],
            )
    np.testing.assert_array_equal(_natural_records(lane_major), natural)
    with pytest.raises(ValueError, match="head_dim 128"):
        _lane_major_records(natural[..., :64])
    with pytest.raises(ValueError, match=r"\[lane32\]\[part4\]"):
        _natural_records(natural)
    assert _TRACE_PROTOCOL == {
        "H7Y": 72,
        "H6A": 72,
        "H6N": 24,
        "H6Z": 24,
        "application_dispatches": 2_286,
        "queues": 1,
        "streams": 1,
        "compiler_processes_allowed": 0,
        "positive_duration_required": True,
    }
    assert _TIMING_PROTOCOL == {
        "warmups_per_arm": 5,
        "samples_per_arm": 15,
        "launches_per_sample": 5,
        "counter_rotated_modes": True,
        "clocks": ("hip_event", "synchronized_wall"),
        "required_winners": (256, 384),
        "weighted_calls": 72,
        "mirrors_prepared_outside_timing": True,
        "no_follow_up_tuning_on_miss": True,
        "no_subset_salvage": True,
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
    assert runtime_module._PREFILL_GLOBAL_SCORE_REPLAY_SCRATCH_BYTES == (
        _SCORE_SCRATCH_BYTES
    )
    assert _VARIANT in repr(
        runtime_module._PREFILL_DENSE_INITIAL_ROLE_CANDIDATES
    )

    candidate_present = hasattr(_module(), _FUNCTION)
    for relative, expected in artifact["source_sha256"].items():
        if candidate_present and relative in _AFFECTED_SOURCE_PATHS:
            continue
        if relative == "tests/test_laguna_kv_attention.py":
            # The gfx1151 handoff extended this shared test after H7Y without
            # changing the frozen gfx1100 candidate sources.
            expected = "35e6aa5c48c4ae797c80b7aff3dea9397d16d77a01295e57452506aa7a2c9f02"
        assert _sha256(_ROOT / relative) == expected


def test_h7y_h6w_immutability_candidate_source_and_physical_contract() -> None:
    module = _module()
    hip_path = Path(module.__file__).with_name("laguna_kv_attention.hip")
    hip_source = hip_path.read_text()
    py_source = Path(module.__file__).read_text()
    h6w_kernel = _extract_braced(hip_source, _H6W_KERNEL_DECLARATION)
    h6w_wrapper = _extract_braced(hip_source, _H6W_WRAPPER_DECLARATION)
    assert _sha256_text(h6w_kernel) == _H6W_KERNEL_SHA256
    assert _sha256_text(h6w_wrapper) == _H6W_WRAPPER_SHA256
    assert _sha256_text(_python_function(py_source, _H6W_FUNCTION)) == (
        _H6W_PYTHON_SHA256
    )
    assert _EXPECTED_PHYSICAL == {
        "local_size": 32,
        "wavefront_size": 32,
        "grid": (2304, 32),
        "global_load_b64": 2,
        "global_load_u16": 0,
        "vmcnt_wait_sites_max": 2,
        "global_load_b128_score_records": 1,
        "global_store_b128_score_records": 1,
        "ds_bpermute_b32": 32,
        "v_exp_f32": 4,
        "v_fma_f32": 56,
        "v_fmac_f32": 41,
        "global_store_b32_output": 16,
        "code_bytes_max": 5_200,
        "instruction_slots_max": 900,
        "metadata_vgpr_max": 56,
        "runtime_vgpr_max": 56,
        "metadata_sgpr_max": 48,
        "lds_bytes": 0,
        "private_bytes": 0,
        "spill_count": 0,
        "runtime_scratch_bytes": 0,
        "block_barriers": 0,
    }

    candidate = _candidate()
    candidate_body = _body(
        hip_source,
        f"__global__ __launch_bounds__(32) void {_KERNEL}(",
    )
    assert candidate.__name__ == _FUNCTION
    assert hip_source.count(_SYMBOL) == 1
    assert hip_source.count(_KERNEL) == 2
    assert "lane_record_base" in candidate_body
    assert candidate_body.count("const uint2 packed_") == 2
    assert candidate_body.count("reinterpret_cast<const uint2*>") == 2
    assert candidate_body.count("lane * 4") == 2
    assert candidate_body.count("part * 32") == 2
    assert "replay_values[row_index] = dot;" in candidate_body
    assert "max_scores[row_index] = fmaxf(" in candidate_body
    assert "workgroup_score_records[logical_slot] = make_float4(" in (
        candidate_body
    )
    assert "expf(dot * scale - max_scores[row_index])" in candidate_body
    assert "denominators[row_index] += weight;" in candidate_body
    assert "output_acc[row_index][part] += weight * cached_values[part];" in (
        candidate_body
    )
    assert "output_acc[row_index][part] / safe_denominator" in candidate_body
    assert "__shared__" not in candidate_body
    assert "__syncthreads" not in candidate_body
    for metadata_read in ("base_offsets[", "token_positions[", "evict_mask["):
        assert metadata_read not in candidate_body


def test_h7y_registry_abi_source_owner_and_gfx1151_exclusion() -> None:
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

    candidate = _candidate()
    module.register_laguna_kv_attention_kernels(replace=True)
    key = KernelKey(
        "hip_gfx1100", "laguna_attention_prefill", "bf16", _VARIANT
    )
    gfx1151_key = KernelKey(
        "hip_gfx1151", "laguna_attention_prefill", "bf16", _VARIANT
    )
    load_backend_kernel_package("hip_gfx1151")
    assert is_registered(key)
    assert not is_registered(gfx1151_key)
    assert resolve(
        backend="hip_gfx1100",
        layer="laguna_attention_prefill",
        quant="bf16",
        variant=_VARIANT,
    ) is candidate
    assert _VARIANT not in _SOURCE_POLICY.values()
    assert (
        _ROOT / "hipengine/kernels/hip_gfx1151/__init__.py"
    ).read_text().count(f'"{_VARIANT}"') == 1

    wrapper_source = inspect.getsource(candidate)
    assert "score_scratch_ptr" in wrapper_source
    assert "score_scratch_nbytes" in wrapper_source
    assert "parsed_start not in {256, 384}" in wrapper_source
    assert "score_scratch_ptr % 16" in wrapper_source
    assert "lane_major_key_cache_ptr % 8" in wrapper_source
    assert "lane_major_value_cache_ptr % 8" in wrapper_source
    assert "lane_major_cache_nbytes" in wrapper_source
    assert "1_048_576" in wrapper_source
    assert "18_874_368" in wrapper_source
    assert "lane-major global-score replay exact SWA requires" in wrapper_source


def test_h7y_strict_preflight_after_h6w_controls(
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

    h6w_common = (
        0xB1000,
        0xB2000,
        0xB3000,
        0xB4000,
        0xB5000,
        0xB6000,
    )
    candidate_common = (
        0xB1000,
        0xB2000,
        0xB3000,
        0xB9000,
        0xBA000,
        0xB6000,
    )
    h6w_library = SimpleNamespace(**{h6w._SYMBOL: FakeFn(h6w_calls)})
    for start in _CANDIDATE_STARTS:
        h6w_fn(
            *h6w_common,
            0xB7000,
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
            "lane_major_key_cache_ptr": candidate_common[3],
            "lane_major_value_cache_ptr": candidate_common[4],
            "score_scratch_ptr": 0xB8000,
            "score_scratch_nbytes": _SCORE_SCRATCH_BYTES,
            "lane_major_cache_nbytes": _MIRROR_BYTES_PER_TENSOR_PER_LAYER,
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
            *candidate_common[:3],
            arguments.pop("lane_major_key_cache_ptr"),
            arguments.pop("lane_major_value_cache_ptr"),
            candidate_common[5],
            arguments.pop("score_scratch_ptr"),
            arguments.pop("spans"),
            arguments.pop("rows"),
            arguments.pop("num_q_heads"),
            arguments.pop("num_kv_heads"),
            arguments.pop("head_dim"),
            _HEAD_DIM**-0.5,
            score_scratch_nbytes=arguments.pop("score_scratch_nbytes"),
            lane_major_cache_nbytes=arguments.pop(
                "lane_major_cache_nbytes"
            ),
            **arguments,
            library=fake_library,
            runtime=SimpleNamespace(),
        )

    for start in _CANDIDATE_STARTS:
        launch(start_position=start)
    assert len(calls) == 2
    for args, start in zip(calls, _CANDIDATE_STARTS, strict=True):
        assert args[3].value == 0xB9000
        assert args[4].value == 0xBA000
        assert args[6].value == 0xB8000
        assert args[12].value == _ROWS
        assert args[13].value == _CAPACITY
        assert args[14].value == _CAPACITY
        assert args[15].value == _QUERY_HEADS
        assert args[16].value == _KV_HEADS
        assert args[17].value == _HEAD_DIM
        assert args[18].value == start

    def fail_build(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid H7Y preflight loaded HIP")

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
        {"lane_major_key_cache_ptr": 0},
        {"lane_major_key_cache_ptr": 0xB9002},
        {"lane_major_value_cache_ptr": 0},
        {"lane_major_value_cache_ptr": 0xBA002},
        {"score_scratch_ptr": 0},
        {"score_scratch_ptr": 0xB8008},
        {"score_scratch_nbytes": _SCORE_SCRATCH_BYTES - 16},
        {"score_scratch_nbytes": _SCORE_SCRATCH_BYTES + 16},
        {"lane_major_cache_nbytes": _MIRROR_BYTES_PER_TENSOR_PER_LAYER - 2},
        {"lane_major_cache_nbytes": _MIRROR_BYTES_PER_TENSOR_PER_LAYER + 2},
    )
    for invalid in invalid_cases:
        with pytest.raises(ValueError, match="lane-major global-score replay exact SWA"):
            launch(fake_library=None, **invalid)
    assert len(calls) == 2


@pytest.fixture(scope="module")
def h7y_library():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    return _module().build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )


@pytest.mark.parametrize("start_position", _CANDIDATE_STARTS)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h7y_complete_output_records_cpu_repeat_spans_and_lifecycle(
    h7y_library,
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
    from hipengine.loading.materialize import float_array_to_bf16_bits
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
    rng = np.random.default_rng(0x7C00 + start_position)
    total_rows = start_position + _ROWS
    keys = rng.normal(
        0.0, 0.12, size=(total_rows, _KV_HEADS, _HEAD_DIM)
    ).astype(np.float32)
    values = rng.normal(0.0, 0.12, size=keys.shape).astype(np.float32)
    queries = rng.normal(
        0.0, 0.12, size=(_ROWS, _QUERY_HEADS, _HEAD_DIM)
    ).astype(np.float32)
    key_bits = float_array_to_bf16_bits(keys)
    value_bits = float_array_to_bf16_bits(values)
    key_mirror_host = np.full(
        (_CAPACITY, _KV_HEADS, 32, 4),
        0xA5A5,
        dtype=np.uint16,
    )
    value_mirror_host = np.full_like(key_mirror_host, 0x5A5A)
    key_mirror_host[:total_rows] = _lane_major_records(key_bits)
    value_mirror_host[:total_rows] = _lane_major_records(value_bits)
    np.testing.assert_array_equal(
        _natural_records(key_mirror_host[:total_rows]), key_bits
    )
    np.testing.assert_array_equal(
        _natural_records(value_mirror_host[:total_rows]), value_bits
    )
    assert key_mirror_host.nbytes == _MIRROR_BYTES_PER_TENSOR_PER_LAYER
    assert value_mirror_host.nbytes == _MIRROR_BYTES_PER_TENSOR_PER_LAYER
    control_host = np.empty_like(queries)
    candidate_host = np.empty_like(queries)
    repeat_host = np.empty_like(queries)
    control_scratch_host = np.empty(_SCORE_SCRATCH_BYTES, dtype=np.uint8)
    candidate_scratch_host = np.empty(_SCORE_SCRATCH_BYTES, dtype=np.uint8)
    repeat_scratch_host = np.empty(_SCORE_SCRATCH_BYTES, dtype=np.uint8)
    allocations = []
    try:
        key_rows = malloc(keys.nbytes, runtime=runtime)
        value_rows = malloc(values.nbytes, runtime=runtime)
        query_rows = malloc(queries.nbytes, runtime=runtime)
        key_mirror = malloc(key_mirror_host.nbytes, runtime=runtime)
        value_mirror = malloc(value_mirror_host.nbytes, runtime=runtime)
        control_out = malloc(control_host.nbytes, runtime=runtime)
        candidate_out = malloc(candidate_host.nbytes, runtime=runtime)
        repeat_out = malloc(repeat_host.nbytes, runtime=runtime)
        control_scratch = malloc(_SCORE_SCRATCH_BYTES, runtime=runtime)
        candidate_scratch = malloc(_SCORE_SCRATCH_BYTES, runtime=runtime)
        repeat_scratch = malloc(_SCORE_SCRATCH_BYTES, runtime=runtime)
        allocations.extend(
            (
                key_rows,
                value_rows,
                query_rows,
                key_mirror,
                value_mirror,
                control_out,
                candidate_out,
                repeat_out,
                control_scratch,
                candidate_scratch,
                repeat_scratch,
            )
        )
        assert key_mirror.ptr % 8 == 0
        assert value_mirror.ptr % 8 == 0
        assert control_scratch.ptr % _SCORE_ALIGNMENT == 0
        assert candidate_scratch.ptr % _SCORE_ALIGNMENT == 0
        assert repeat_scratch.ptr % _SCORE_ALIGNMENT == 0
        for device, host in (
            (key_rows, keys),
            (value_rows, values),
            (query_rows, queries),
            (key_mirror, key_mirror_host),
            (value_mirror, value_mirror_host),
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
            library=h7y_library,
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
            library=h7y_library,
        )
        state = cache.layer(0)
        before_spans = h6w._span_snapshot(state.spans, runtime)
        control_common = (
            query_rows.ptr,
            current_key_ptr,
            current_value_ptr,
            state.key_cache.ptr,
            state.value_cache.ptr,
        )
        candidate_common = (
            query_rows.ptr,
            current_key_ptr,
            current_value_ptr,
            key_mirror.ptr,
            value_mirror.ptr,
        )
        suffix = (
            state.spans,
            _ROWS,
            _QUERY_HEADS,
            _KV_HEADS,
            _HEAD_DIM,
            _HEAD_DIM**-0.5,
        )
        runtime.memset(control_out.ptr, 0xA5, control_out.nbytes)
        runtime.memset(control_scratch.ptr, 0xA5, control_scratch.nbytes)
        h6w_fn(
            *control_common,
            control_out.ptr,
            control_scratch.ptr,
            *suffix,
            score_scratch_nbytes=control_scratch.nbytes,
            sliding_window=_CAPACITY,
            start_position=start_position,
            library=h7y_library,
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
        h6w._assert_scratch_record_coverage(
            control_scratch_host,
            start_position=start_position,
        )
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
        for out, scratch, out_host, scratch_host in (
            (
                candidate_out,
                candidate_scratch,
                candidate_host,
                candidate_scratch_host,
            ),
            (repeat_out, repeat_scratch, repeat_host, repeat_scratch_host),
        ):
            runtime.memset(out.ptr, 0xA5, out.nbytes)
            runtime.memset(scratch.ptr, 0xA5, scratch.nbytes)
            candidate(
                *candidate_common,
                out.ptr,
                scratch.ptr,
                *suffix,
                score_scratch_nbytes=scratch.nbytes,
                lane_major_cache_nbytes=key_mirror.nbytes,
                sliding_window=_CAPACITY,
                start_position=start_position,
                library=h7y_library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            copy_device_to_host(
                host_array_ptr(out_host),
                out,
                out_host.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(scratch_host),
                scratch,
                scratch_host.nbytes,
                runtime=runtime,
            )
            assert np.isfinite(out_host).all()
            np.testing.assert_array_equal(out_host, control_host)
            np.testing.assert_array_equal(scratch_host, control_scratch_host)
            h6w._assert_scratch_record_coverage(
                scratch_host,
                start_position=start_position,
            )
            after_candidate_spans = h6w._span_snapshot(state.spans, runtime)
            for actual, expected in zip(
                after_candidate_spans, before_spans, strict=True
            ):
                np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(repeat_host, candidate_host)
        np.testing.assert_array_equal(
            repeat_scratch_host,
            candidate_scratch_host,
        )
        cache.discard_rows()
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        cache.free()
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
