"""WPF-H7K exact late-SWA score-to-weight publication RED contract."""

from __future__ import annotations

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
    "laguna_h7k_h6w_helper", _HELPER_PATH
)
assert _HELPER_SPEC is not None and _HELPER_SPEC.loader is not None
h6w = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(h6w)

_ROOT = Path(__file__).resolve().parents[1]
_TARGET_ARTIFACT = _ROOT / (
    "benchmarks/results/2026-08-02-gfx1100-laguna-q2-xl-"
    "post-h7j-matched-swa-weight-publication-target.json"
)
_TARGET_ARTIFACT_SHA256 = (
    "ea6f85443668d44634ab4c5c442de1579d8d6a427f443ed9945d95b337b12733"
)
_CANDIDATE_STARTS = (256, 384)
_FALLBACK_STARTS = (0, 128)
_CALL_WEIGHTS = {256: 36, 384: 36}
_VARIANT = (
    "swa_context_rows_qrow4_dense_initial_"
    "global_score_weight_publication_exact_spans"
)
_FUNCTION = (
    "laguna_swa_attention_prefill_qrow4_dense_initial_"
    "global_score_weight_publication_exact_bf16_spans"
)
_SYMBOL = (
    "hipengine_laguna_swa_attention_prefill_qrow4_dense_initial_"
    "global_score_weight_publication_exact_bf16_spans"
)
_KERNEL = (
    "laguna_swa_attention_prefill_qrow4_dense_initial_"
    "global_score_weight_publication_exact_bf16_kernel"
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
_H6Z_KERNEL_DECLARATION = (
    "__global__ __launch_bounds__(32) void "
    "laguna_global_attention_prefill_qrow4_dense_initial_"
    "global_score_weight_replay_exact_bf16_kernel("
)
_H6Z_KERNEL_SHA256 = (
    "3a21a68d13eefb1e865d193869e7fe3b4e86f559349620b091eec14594a5f66b"
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
_EXISTING_ORDERED_WORKSPACE_BYTES = 161_120_256
_CONTEXT_ITERATIONS_PER_CALL = {256: 741_888, 384: 1_036_800}
_VISIBLE_SCORES = {256: 106_334_208, 384: 148_801_536}
_DYNAMIC_WORK_MODEL = {
    "context_iterations_per_request": 64_032_768,
    "visible_score_weight_broadcasts_removed": 255_135_744,
    "aligned_weight_record_wave_operations_added": 128_065_536,
    "aligned_weight_record_logical_bytes_added": 2_049_048_576,
    "net_exchange_wave_operations_removed": 127_070_208,
    "workgroups_changed": 0,
    "kv_logical_bytes_changed": 0,
}
_EXPECTED_PHYSICAL = {
    "local_size": 32,
    "grid": [2304, 32],
    "global_load_u16": 8,
    "global_load_b32_query": 16,
    "global_store_b32_output": 16,
    "global_load_b128_records": 2,
    "global_store_b128_records": 2,
    "ds_bpermute_b32": 28,
    "v_exp_f32_e32": 4,
    "v_fma_f32": 56,
    "code_bytes_max": 5_500,
    "instruction_slots_max": 950,
    "metadata_vgpr_max": 54,
    "runtime_vgpr_max": 56,
    "lds_private_spill_runtime_scratch": 0,
    "block_barriers": 0,
}
_TRACE_PROTOCOL = {
    "kernel": _KERNEL,
    "require_cached_build": True,
    "compiler_processes_allowed": 0,
    "positive_duration_required": True,
    "local_size": 32,
    "grid": (2304, 32),
    "runtime_vgpr_max": 56,
    "lds_bytes": 0,
    "runtime_scratch_bytes": 0,
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
    "no_subset_salvage": True,
}
_RUNTIME_SHA256 = (
    "a9d046bd793aca0a3085d4040a14992d80f2b7261ee16f2721ce3ad2ed8bc31f"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        base_offsets=_tensor(0x91000, (capacity,), "int32"),
        live_counts=_tensor(0x92000, (1,), "int64"),
        token_positions=_tensor(0x93000, (capacity,), "int64"),
        evict_mask=_tensor(0x94000, (capacity,), "bool"),
        row_positions=_tensor(0x95000, (1,), "int64"),
        capacity=capacity,
        storage_dtype="bf16",
    )


def test_h7k_frozen_target_artifact_controls_and_protocol() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.runtime import laguna_kv as runtime_module

    assert _sha256(_TARGET_ARTIFACT) == _TARGET_ARTIFACT_SHA256
    artifact = json.loads(_TARGET_ARTIFACT.read_text())
    assert artifact["status"] == (
        "accepted_matched_production_rerank_and_exact_h7k_target"
    )
    assert artifact["decision"] == {
        "candidate_implemented": False,
        "correctness_claim": "none_before_red_and_candidate_implementation",
        "h7k_target_selected": True,
        "next_action": (
            "Freeze the inseparable two-start H7K RED before implementation, "
            "then build one repository object and reject before timing on any "
            "correctness or physical miss."
        ),
        "performance_claim": "target_selection_only",
        "production_changed": False,
    }
    target = artifact["target"]
    assert target["id"] == "WPF-H7K"
    assert target["implementation_absent"] is True
    assert target["physical_gate"] == _EXPECTED_PHYSICAL
    assert target["dynamic_model"]["warning"].endswith("not physical traffic or a speed claim.")
    for key, expected in _DYNAMIC_WORK_MODEL.items():
        assert target["dynamic_model"][key] == expected
    assert target["admission"]["red_first"] is True
    assert target["admission"]["runtime_and_source_separate"] is True
    assert "no start/layer/prompt subset" in target["admission"]["no_salvage"]
    assert artifact["production"]["wall_tok_s"] == 431.310165
    assert artifact["production"]["kernel_sum_ms"] == pytest.approx(
        1_172.241239
    )
    assert artifact["production"]["h6w"]["calls"] == 72
    assert artifact["production"]["h6w"]["median_ms"] == {
        "start_256_ms": 26.467881,
        "start_384_ms": 36.163843,
        "total_ms": 62.627239,
    }

    assert _CANDIDATE_STARTS == (256, 384)
    assert _FALLBACK_STARTS == (0, 128)
    assert _CALL_WEIGHTS == {256: 36, 384: 36}
    assert _WORKGROUPS == 2_304
    assert _SCORE_SCRATCH_BYTES == 18_874_368
    assert _SCORE_SCRATCH_BYTES < _EXISTING_ORDERED_WORKSPACE_BYTES
    assert _SCORE_SCRATCH_BYTES % _SCORE_ALIGNMENT == 0
    context_iterations = sum(
        _CALL_WEIGHTS[start] * _CONTEXT_ITERATIONS_PER_CALL[start]
        for start in _CANDIDATE_STARTS
    )
    assert context_iterations == 64_032_768
    assert _DYNAMIC_WORK_MODEL == {
        "context_iterations_per_request": context_iterations,
        "visible_score_weight_broadcasts_removed": sum(
            _VISIBLE_SCORES.values()
        ),
        "aligned_weight_record_wave_operations_added": context_iterations * 2,
        "aligned_weight_record_logical_bytes_added": context_iterations * 32,
        "net_exchange_wave_operations_removed": (
            sum(_VISIBLE_SCORES.values()) - context_iterations * 2
        ),
        "workgroups_changed": 0,
        "kv_logical_bytes_changed": 0,
    }
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
    assert _sha256(Path(runtime_module.__file__)) == _RUNTIME_SHA256
    assert _VARIANT not in repr(runtime_module._PREFILL_DENSE_INITIAL_ROLE_CANDIDATES)

    module = _module()
    source = Path(module.__file__).with_name("laguna_kv_attention.hip").read_text()
    h6w_kernel = _extract_braced(source, _H6W_KERNEL_DECLARATION)
    h6w_wrapper = _extract_braced(source, _H6W_WRAPPER_DECLARATION)
    h6z_kernel = _extract_braced(source, _H6Z_KERNEL_DECLARATION)
    assert _sha256_text(h6w_kernel) == _H6W_KERNEL_SHA256
    assert _sha256_text(h6w_wrapper) == _H6W_WRAPPER_SHA256
    assert _sha256_text(h6z_kernel) == _H6Z_KERNEL_SHA256
    assert _sha256_text(inspect.getsource(getattr(module, _H6W_FUNCTION))) == (
        _H6W_PYTHON_SHA256
    )
    h6w_body = h6w_kernel[h6w_kernel.index("{") + 1 : -1]
    assert "replay_values[row_index] = dot;" in h6w_body
    assert "expf(dot * scale - max_scores[row_index])" in h6w_body
    assert "denominators[row_index] += weight;" in h6w_body
    assert "output_acc[row_index][part] += weight * cached_values[part];" in h6w_body
    assert "output_acc[row_index][part] / safe_denominator" in h6w_body


def test_h7k_registry_source_abi_and_gfx1151_exclusion() -> None:
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
    assert candidate_body.count(
        "workgroup_score_records[logical_slot] = make_float4("
    ) == 2
    assert "replay_values[row_index] = dot;" in candidate_body
    assert "dot * scale - max_scores[row_index]" in candidate_body
    assert "denominators[row_index] += weight;" in candidate_body
    assert "weights[row_index] = weight;" in candidate_body
    assert "float4 weight_record = workgroup_score_records[logical_slot];" in (
        candidate_body
    )
    assert "output_acc[row_index][part] += weight * cached_values[part];" in (
        candidate_body
    )
    assert "output_acc[row_index][part] / safe_denominator" in candidate_body
    assert "logical_slot = lane" not in candidate_body
    assert "normalized_weights" not in candidate_body
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
    assert "score-to-weight publication exact SWA requires" in wrapper_source

    gfx1151_source = (_ROOT / "hipengine/kernels/hip_gfx1151/__init__.py").read_text()
    assert gfx1151_source.count(f'"{_VARIANT}"') == 1


def test_h7k_strict_preflight_after_h6w_control(
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

    common = (0xA1000, 0xA2000, 0xA3000, 0xA4000, 0xA5000, 0xA6000)
    h6w_library = SimpleNamespace(**{h6w._SYMBOL: FakeFn(h6w_calls)})
    for start in _CANDIDATE_STARTS:
        h6w_fn(
            *common,
            0xA7000,
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
            "score_scratch_ptr": 0xA8000,
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
        assert args[6].value == 0xA8000
        assert args[12].value == _ROWS
        assert args[13].value == _CAPACITY
        assert args[14].value == _CAPACITY
        assert args[15].value == _QUERY_HEADS
        assert args[16].value == _KV_HEADS
        assert args[17].value == _HEAD_DIM
        assert args[18].value == start

    def fail_build(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid H7K preflight loaded HIP")

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
        {"score_scratch_ptr": 0xA8008},
        {"score_scratch_nbytes": _SCORE_SCRATCH_BYTES - 16},
        {"score_scratch_nbytes": _SCORE_SCRATCH_BYTES + 16},
    )
    for invalid in invalid_cases:
        with pytest.raises(
            ValueError, match="score-to-weight publication exact SWA"
        ):
            launch(fake_library=None, **invalid)
    assert len(calls) == 2


def _assert_weight_record_coverage_and_values(
    candidate_bytes: np.ndarray,
    control_bytes: np.ndarray,
    *,
    start_position: int,
) -> None:
    candidate = candidate_bytes.view(np.float32).reshape(
        _QUERY_ROW_GROUPS,
        _QUERY_HEADS,
        _CAPACITY,
        4,
    )
    control = control_bytes.view(np.float32).reshape(candidate.shape)
    candidate_records = candidate_bytes.reshape(
        _QUERY_ROW_GROUPS,
        _QUERY_HEADS,
        _CAPACITY,
        _SCORE_RECORD_BYTES,
    )
    poison = np.full(_SCORE_RECORD_BYTES, 0xA5, dtype=np.uint8)
    scale = np.float32(_HEAD_DIM**-0.5)
    for row_group in range(_QUERY_ROW_GROUPS):
        query_base = row_group * 4
        context_len = start_position + query_base + 4
        active_records = candidate_records[
            row_group, :, :context_len, :
        ].reshape(-1, _SCORE_RECORD_BYTES)
        assert not np.any(np.all(active_records == poison, axis=1))
        assert np.all(
            candidate_records[row_group, :, context_len:, :] == 0xA5
        )
        for row_index in range(4):
            visible = start_position + query_base + row_index + 1
            actual = candidate[row_group, :, :visible, row_index]
            assert np.isfinite(actual).all()
            assert np.all(actual > 0.0)
            assert np.all(
                candidate[
                    row_group,
                    :,
                    visible:context_len,
                    row_index,
                ]
                == 0.0
            )
            scores = np.multiply(
                control[row_group, :, :visible, row_index],
                scale,
                dtype=np.float32,
            )
            expected = np.exp(
                scores - np.max(scores, axis=1, keepdims=True),
                dtype=np.float32,
            )
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=3e-6,
                atol=3e-7,
            )


@pytest.fixture(scope="module")
def h7k_library():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    return _module().build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )


@pytest.mark.parametrize("start_position", _CANDIDATE_STARTS)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h7k_complete_output_weight_records_cpu_spans_and_lifecycle(
    h7k_library,
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
    rng = np.random.default_rng(0x7B00 + start_position)
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
        cache.prepare_rows(tuple(range(start_position)))
        cache.append_rows(
            0,
            key_rows.ptr,
            value_rows.ptr,
            start_position,
            library=h7k_library,
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
            library=h7k_library,
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
            library=h7k_library,
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
            library=h7k_library,
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
        _assert_weight_record_coverage_and_values(
            candidate_scratch_host,
            control_scratch_host,
            start_position=start_position,
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
