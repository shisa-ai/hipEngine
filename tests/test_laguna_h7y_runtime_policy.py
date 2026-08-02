"""WPF-H7Y bounded lane-major SWA cache runtime ownership RED."""

from __future__ import annotations

import ast
import ctypes
import hashlib
import importlib
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.hip import HipMemcpyKind
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION

_ROOT = Path(__file__).resolve().parents[1]
_STANDALONE_ARTIFACT = _ROOT / (
    "benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-"
    "swa-lane-major-cache-candidate.json"
)
_STANDALONE_ARTIFACT_SHA256 = (
    "e2c0e6685bbd5401bb0433e79ebdb931269ca8d710634e8b022d74acabbb1d67"
)
_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H7Y_ROLE_VARIANTS"
_SOURCE_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS"
_H6Z_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6Z_ROLE_VARIANTS"
_H6W_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6W_ROLE_VARIANTS"
_H6A_CAPABILITY = "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6A_ROLE_VARIANTS"
_GLOBAL_ROLE = "global_m128_c4096_first_fill_exact"
_SWA_ROLE = "swa_qrow4_m128_c512_no_wrap_exact"
_H6N_GLOBAL = "global_context_rows_dense_initial_fixed512_cached_exact_spans"
_H6Z_GLOBAL = (
    "global_context_rows_qrow4_dense_initial_global_score_weight_replay_"
    "exact_spans"
)
_H6A_SWA = "swa_context_rows_qrow4_dense_initial_cached_exact_spans"
_H6W_SWA = (
    "swa_context_rows_qrow4_dense_initial_global_score_replay_exact_spans"
)
_H7Y_SWA = (
    "swa_context_rows_qrow4_dense_initial_lane_major_"
    "global_score_replay_exact_spans"
)
_H6A_POLICY = {_GLOBAL_ROLE: _H6N_GLOBAL, _SWA_ROLE: _H6A_SWA}
_H6W_POLICY = {_GLOBAL_ROLE: _H6N_GLOBAL, _SWA_ROLE: _H6W_SWA}
_H6Z_POLICY = {_GLOBAL_ROLE: _H6Z_GLOBAL, _SWA_ROLE: _H6W_SWA}
_H7Y_POLICY = {_GLOBAL_ROLE: _H6Z_GLOBAL, _SWA_ROLE: _H7Y_SWA}
_WRITER_VARIANT = "swa_f32_rows_natural_lane_major_spans"
_WRITER_FUNCTION = "laguna_swa_write_kv_rows_natural_lane_major_f32_spans"
_WRITER_SYMBOL = "hipengine_" + _WRITER_FUNCTION.replace(
    "_f32_spans", "_f32_bf16_spans"
)
_WRITER_KERNEL = "laguna_swa_write_kv_rows_natural_lane_major_f32_bf16_kernel"
_NATURAL_WRITER_KERNEL_DECLARATION = (
    "__global__ void laguna_swa_write_kv_rows_f32_bf16_kernel("
)
_NATURAL_WRITER_WRAPPER_DECLARATION = (
    'extern "C" int hipengine_laguna_swa_write_kv_rows_f32_bf16_spans('
)
_NATURAL_WRITER_FUNCTION = "laguna_swa_write_kv_rows_f32_spans"
_NATURAL_WRITER_KERNEL_SHA256 = (
    "6ff8e5c8d7ab570b144a7991d5101664d67245732b0cf7ee3ab8bd9c71fc683f"
)
_NATURAL_WRITER_WRAPPER_SHA256 = (
    "e964bcc7b8d45ade275a2e72c7bf10fa27ae9e7776719b1e9f03b7de5d1e5e76"
)
_NATURAL_WRITER_PYTHON_SHA256 = (
    "004c437a04b8f0bde83d8de549902345a83f6c329ba0450d0860883492678f62"
)
_FALLBACK_STARTS = (0, 128)
_H7Y_STARTS = (256, 384)
_ROWS = 128
_CAPACITY = 512
_KV_HEADS = 8
_HEAD_DIM = 128
_SWA_LAYERS = 36
_MIRROR_BYTES_PER_TENSOR_PER_LAYER = (
    _CAPACITY * _KV_HEADS * _HEAD_DIM * np.dtype(np.uint16).itemsize
)
_MIRROR_BYTES_PER_REQUEST = (
    2 * _SWA_LAYERS * _MIRROR_BYTES_PER_TENSOR_PER_LAYER
)
_SCORE_SCRATCH_BYTES = 18_874_368
_Q5_WEIGHT_PLANE_BYTES = 150_994_944
_EXPECTED_RUNTIME_TOPOLOGY = {
    "H6N": 24,
    "H6Z": 24,
    "H6A": 72,
    "H7Y": 72,
    "swa_natural_lane_major_writer": 144,
    "swa_natural_writer": 0,
    "global_natural_writer": 48,
    "application_dispatches": 2_286,
    "queues": 1,
    "streams": 1,
    "compiler_processes_allowed": 0,
}
_WRITER_PHYSICAL = {
    "local_size": 256,
    "wavefront_size": 32,
    "grid": (1_024, 128),
    "input_global_load_b32": 2,
    "natural_and_mirror_global_store_d16": 4,
    "code_bytes_max": 2_200,
    "instruction_slots_max": 450,
    "metadata_vgpr_max": 24,
    "runtime_vgpr_max": 24,
    "metadata_sgpr_max": 64,
    "lds_bytes": 0,
    "private_bytes": 0,
    "sgpr_spills": 0,
    "vgpr_spills": 0,
    "runtime_scratch_bytes": 0,
    "block_barriers": 0,
}
_RUNTIME_ADMISSION = {
    "fixed_warmups": 1,
    "fixed_samples": 5,
    "lengths": (512, 1_024, 4_096),
    "paired_samples_per_length": 3,
    "complete_state_byte_exact": True,
    "writer_inclusive": True,
    "every_fixed_sample_finite": True,
    "fixed_median_must_win": True,
    "every_length_median_must_win": True,
    "natural_h6a_h6w_rollback_required": True,
    "no_subset_or_favorable_rerun": True,
}


class _FakeRuntime:
    def __init__(self) -> None:
        self.next_ptr = 0x10000000
        self.allocations: dict[int, int] = {}
        self.malloc_calls = 0

    def malloc(self, nbytes: int) -> int:
        self.malloc_calls += 1
        ptr = self.next_ptr
        self.next_ptr += max(0x1000, int(nbytes) + 0x100)
        self.allocations[ptr] = int(nbytes)
        return ptr

    def free(self, ptr: int) -> None:
        self.allocations.pop(int(ptr), None)

    def memcpy(self, dst: int, src: int, count: int, kind: HipMemcpyKind) -> None:
        del src, count
        assert kind == HipMemcpyKind.HOST_TO_DEVICE
        assert any(
            base <= int(dst) < base + max(size, 1)
            for base, size in self.allocations.items()
        )

    def memset(self, dst: int, value: int, nbytes: int) -> None:
        del value, nbytes
        assert any(
            base <= int(dst) < base + max(size, 1)
            for base, size in self.allocations.items()
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


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


def _production_config() -> SimpleNamespace:
    layer_types = tuple(
        FULL_ATTENTION if layer_id % 4 == 0 else SLIDING_ATTENTION
        for layer_id in range(48)
    )
    return SimpleNamespace(
        block_count=48,
        layer_types=layer_types,
        head_counts=tuple(
            48 if attention_type == FULL_ATTENTION else 72
            for attention_type in layer_types
        ),
        head_count_kv=_KV_HEADS,
        key_length=_HEAD_DIM,
        value_length=_HEAD_DIM,
        sliding_window=_CAPACITY,
    )


def _one_swa_layer_config() -> SimpleNamespace:
    return SimpleNamespace(
        block_count=1,
        layer_types=(SLIDING_ATTENTION,),
        head_counts=(72,),
        head_count_kv=_KV_HEADS,
        key_length=_HEAD_DIM,
        value_length=_HEAD_DIM,
        sliding_window=_CAPACITY,
    )


def _candidate_capability():
    from hipengine.kernels import hip_gfx1100

    return getattr(hip_gfx1100, _CAPABILITY)


def _candidate_writer():
    module = importlib.import_module(
        "hipengine.kernels.hip_gfx1100.attention.laguna_kv"
    )
    return getattr(module, _WRITER_FUNCTION)


def _install_fake_dispatch(cache, calls: list[tuple[str, tuple, dict]]) -> None:
    def resolve(layer: str, variant: str):
        calls.append((f"resolve:{layer}:{variant}", (), {}))

        def launch(*args: object, **kwargs: object) -> None:
            calls.append((f"launch:{variant}", args, dict(kwargs)))

        return launch

    cache._resolve = resolve


def _dispatch(cache, layer_id: int, start: int) -> None:
    cache.position = start - 1
    cache.prepare_rows(tuple(range(start, start + _ROWS)))
    assert cache.can_preappend_attention_prefill(layer_id, _ROWS)
    cache.append_rows(layer_id, 0x2000, 0x3000, _ROWS)
    cache.attend_prefill_cached(
        layer_id,
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        _ROWS,
    )
    cache.discard_rows()


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _require_cached_build() -> bool:
    return os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def test_h7y_runtime_frozen_owner_physical_topology_and_admission_contract() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
        register_laguna_kv_attention_kernels,
    )
    from hipengine.kernels.registry import KernelKey, is_registered

    register_laguna_kv_attention_kernels(replace=True)
    assert _sha256(_STANDALONE_ARTIFACT) == _STANDALONE_ARTIFACT_SHA256
    assert _FALLBACK_STARTS == (0, 128)
    assert _H7Y_STARTS == (256, 384)
    assert _MIRROR_BYTES_PER_TENSOR_PER_LAYER == 1_048_576
    assert _MIRROR_BYTES_PER_REQUEST == 75_497_472 == 72 * 1024 * 1024
    assert _SCORE_SCRATCH_BYTES == 2_304 * 512 * 16
    assert _EXPECTED_RUNTIME_TOPOLOGY == {
        "H6N": 24,
        "H6Z": 24,
        "H6A": 72,
        "H7Y": 72,
        "swa_natural_lane_major_writer": 144,
        "swa_natural_writer": 0,
        "global_natural_writer": 48,
        "application_dispatches": 2_286,
        "queues": 1,
        "streams": 1,
        "compiler_processes_allowed": 0,
    }
    assert _WRITER_PHYSICAL == {
        "local_size": 256,
        "wavefront_size": 32,
        "grid": (1_024, 128),
        "input_global_load_b32": 2,
        "natural_and_mirror_global_store_d16": 4,
        "code_bytes_max": 2_200,
        "instruction_slots_max": 450,
        "metadata_vgpr_max": 24,
        "runtime_vgpr_max": 24,
        "metadata_sgpr_max": 64,
        "lds_bytes": 0,
        "private_bytes": 0,
        "sgpr_spills": 0,
        "vgpr_spills": 0,
        "runtime_scratch_bytes": 0,
        "block_barriers": 0,
    }
    assert _RUNTIME_ADMISSION["lengths"] == (512, 1_024, 4_096)
    assert _RUNTIME_ADMISSION["writer_inclusive"] is True
    assert _RUNTIME_ADMISSION["natural_h6a_h6w_rollback_required"] is True
    assert _RUNTIME_ADMISSION["no_subset_or_favorable_rerun"] is True
    assert getattr(hip_gfx1100, _SOURCE_CAPABILITY) == _H6Z_POLICY
    assert getattr(hip_gfx1100, _H6Z_CAPABILITY) == _H6Z_POLICY
    assert getattr(hip_gfx1100, _H6W_CAPABILITY) == _H6W_POLICY
    assert getattr(hip_gfx1100, _H6A_CAPABILITY) == _H6A_POLICY
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "laguna_attention_prefill",
            "bf16",
            _H7Y_SWA,
        )
    )


def test_h7y_runtime_capability_fused_writer_and_natural_writer_immutability() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.kernels.registry import KernelKey, is_registered, resolve

    module = importlib.import_module(
        "hipengine.kernels.hip_gfx1100.attention.laguna_kv"
    )
    capability = _candidate_capability()
    assert capability == _H7Y_POLICY
    assert getattr(hip_gfx1100, _SOURCE_CAPABILITY) == _H6Z_POLICY
    assert not hasattr(hip_gfx1151, _CAPABILITY)
    assert is_registered(
        KernelKey("hip_gfx1100", "laguna_kv_write", "bf16", _WRITER_VARIANT)
    )
    assert not is_registered(
        KernelKey("hip_gfx1151", "laguna_kv_write", "bf16", _WRITER_VARIANT)
    )
    candidate = _candidate_writer()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="laguna_kv_write",
            quant="bf16",
            variant=_WRITER_VARIANT,
        )
        is candidate
    )

    hip_source = Path(module.__file__).with_name(
        "laguna_kv_attention.hip"
    ).read_text()
    py_source = Path(module.__file__).read_text()
    assert _sha256_text(
        _extract_braced(hip_source, _NATURAL_WRITER_KERNEL_DECLARATION)
    ) == _NATURAL_WRITER_KERNEL_SHA256
    assert _sha256_text(
        _extract_braced(hip_source, _NATURAL_WRITER_WRAPPER_DECLARATION)
    ) == _NATURAL_WRITER_WRAPPER_SHA256
    assert _sha256_text(
        _python_function(py_source, _NATURAL_WRITER_FUNCTION)
    ) == _NATURAL_WRITER_PYTHON_SHA256

    body = _extract_braced(
        hip_source,
        f"__global__ __launch_bounds__(256) void {_WRITER_KERNEL}(",
    )
    assert body.count("laguna_float_to_bf16_bits") == 2
    assert "natural_key_bits" in body
    assert "natural_value_bits" in body
    assert "key_cache[cache_offset] = natural_key_bits;" in body
    assert "value_cache[cache_offset] = natural_value_bits;" in body
    assert "const int64_t lane = dim & 31;" in body
    assert "const int64_t part = dim >> 5;" in body
    assert "lane * 4 + part" in body
    assert "lane_major_key_cache[mirror_offset] = natural_key_bits;" in body
    assert "lane_major_value_cache[mirror_offset] = natural_value_bits;" in body
    assert body.count("token_positions[logical_slot] = position;") == 1
    assert body.count("live_counts[0]") >= 2
    assert hip_source.count(_WRITER_SYMBOL) == 1
    assert hip_source.count(_WRITER_KERNEL) == 2


def test_h7y_runtime_allocates_exact_mirrors_routes_late_starts_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.runtime import laguna_kv as module

    capability = _candidate_capability()
    assert capability == _H7Y_POLICY

    rollback_runtime = _FakeRuntime()
    rollback = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=rollback_runtime,
    )
    try:
        assert rollback.prefill_preappend_role_variants == _H6Z_POLICY
        assert rollback.lane_major_mirror_nbytes == 0
        assert rollback.lane_major_mirror_allocation_count == 0
        assert all(
            state.lane_major_key_cache is None
            and state.lane_major_value_cache is None
            for state in rollback.layers
        )
    finally:
        rollback.free()
    assert rollback_runtime.allocations == {}

    monkeypatch.setattr(hip_gfx1100, _SOURCE_CAPABILITY, dict(capability))
    runtime = _FakeRuntime()
    cache = module.allocate_laguna_kv_cache(
        _production_config(),
        context_length=4096,
        backend="hip_gfx1100",
        runtime=runtime,
    )
    calls: list[tuple[str, tuple, dict]] = []
    _install_fake_dispatch(cache, calls)
    try:
        assert cache.prefill_preappend_role_variants == _H7Y_POLICY
        assert cache.lane_major_mirror_nbytes == _MIRROR_BYTES_PER_REQUEST
        assert cache.lane_major_mirror_allocation_count == 72
        assert cache.resident_nbytes - rollback.resident_nbytes == (
            _MIRROR_BYTES_PER_REQUEST
        )
        assert cache.allocation_count - rollback.allocation_count == 72
        global_states = [
            state
            for state in cache.layers
            if state.attention_type == FULL_ATTENTION
        ]
        swa_states = [
            state
            for state in cache.layers
            if state.attention_type == SLIDING_ATTENTION
        ]
        assert len(global_states) == 12
        assert len(swa_states) == _SWA_LAYERS
        assert all(
            state.lane_major_key_cache is None
            and state.lane_major_value_cache is None
            and state.write_rows_variant == "global_f32_rows_spans"
            for state in global_states
        )
        for state in swa_states:
            assert state.lane_major_key_cache is not None
            assert state.lane_major_value_cache is not None
            assert state.lane_major_key_cache.nbytes == (
                _MIRROR_BYTES_PER_TENSOR_PER_LAYER
            )
            assert state.lane_major_value_cache.nbytes == (
                _MIRROR_BYTES_PER_TENSOR_PER_LAYER
            )
            assert state.lane_major_key_cache.ptr % 8 == 0
            assert state.lane_major_value_cache.ptr % 8 == 0
            assert state.lane_major_key_cache.ptr != state.key_cache.ptr
            assert state.lane_major_value_cache.ptr != state.value_cache.ptr
            assert state.write_rows_variant == _WRITER_VARIANT

        cache.bind_prefill_score_scratch(
            0x40000000,
            _Q5_WEIGHT_PLANE_BYTES,
        )
        for layer_id in range(48):
            for start in (*_FALLBACK_STARTS, *_H7Y_STARTS):
                _dispatch(cache, layer_id, start)

        launches = [
            call[0].removeprefix("launch:")
            for call in calls
            if call[0].startswith("launch:")
        ]
        counts = Counter(launches)
        assert counts["global_f32_rows_spans"] == 48
        assert counts[_WRITER_VARIANT] == 144
        assert counts["swa_f32_rows_spans"] == 0
        assert counts[_H6N_GLOBAL] == 24
        assert counts[_H6Z_GLOBAL] == 24
        assert counts[_H6A_SWA] == 72
        assert counts[_H7Y_SWA] == 72
        assert sum(counts.values()) == 384

        writer_launches = [
            call for call in calls if call[0] == f"launch:{_WRITER_VARIANT}"
        ]
        for launch in writer_launches:
            args = launch[1]
            assert args[4] > 0
            assert args[5] > 0
            assert args[4] % 8 == 0
            assert args[5] % 8 == 0
            assert launch[2]["lane_major_cache_nbytes"] == (
                _MIRROR_BYTES_PER_TENSOR_PER_LAYER
            )

        h7y_launches = [
            call for call in calls if call[0] == f"launch:{_H7Y_SWA}"
        ]
        assert {call[2]["start_position"] for call in h7y_launches} == {
            256,
            384,
        }
        for launch in h7y_launches:
            args = launch[1]
            assert args[3] > 0 and args[3] % 8 == 0
            assert args[4] > 0 and args[4] % 8 == 0
            assert args[6] == 0x40000000
            assert launch[2]["lane_major_cache_nbytes"] == (
                _MIRROR_BYTES_PER_TENSOR_PER_LAYER
            )
    finally:
        cache.free()
    assert runtime.allocations == {}


@pytest.fixture(scope="module")
def h7y_runtime_library():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    module = importlib.import_module(
        "hipengine.kernels.hip_gfx1100.attention.laguna_kv"
    )
    return module.build_laguna_kv_attention(
        load=True,
        require_cached=_require_cached_build(),
    )


@pytest.mark.parametrize("start_position", (0, 128, 256, 384))
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h7y_fused_writer_natural_mirror_bits_spans_and_lifecycle(
    h7y_runtime_library,
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

    helper = importlib.import_module(
        "tests.test_laguna_h6w_swa_global_score_replay"
    )
    runtime = get_hip_runtime()
    before = memory_stats()
    baseline = allocate_laguna_kv_cache(
        _one_swa_layer_config(),
        context_length=_CAPACITY,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    candidate_cache = allocate_laguna_kv_cache(
        _one_swa_layer_config(),
        context_length=_CAPACITY,
        backend="hip_gfx1151",
        runtime=runtime,
    )
    rng = np.random.default_rng(0x7D00 + start_position)
    keys = rng.normal(
        0.0,
        0.12,
        size=(_ROWS, _KV_HEADS, _HEAD_DIM),
    ).astype(np.float32)
    values = rng.normal(0.0, 0.12, size=keys.shape).astype(np.float32)
    key_bits = float_array_to_bf16_bits(keys)
    value_bits = float_array_to_bf16_bits(values)
    poison = np.uint16(0xA5A5)
    natural_shape = (_CAPACITY, _KV_HEADS, _HEAD_DIM)
    mirror_shape = (_CAPACITY, _KV_HEADS, 32, 4)
    baseline_key_host = np.empty(natural_shape, dtype=np.uint16)
    baseline_value_host = np.empty(natural_shape, dtype=np.uint16)
    candidate_key_host = np.empty(natural_shape, dtype=np.uint16)
    candidate_value_host = np.empty(natural_shape, dtype=np.uint16)
    mirror_key_host = np.empty(mirror_shape, dtype=np.uint16)
    mirror_value_host = np.empty(mirror_shape, dtype=np.uint16)
    allocations = []
    try:
        key_rows = malloc(keys.nbytes, runtime=runtime)
        value_rows = malloc(values.nbytes, runtime=runtime)
        mirror_key = malloc(
            _MIRROR_BYTES_PER_TENSOR_PER_LAYER,
            runtime=runtime,
        )
        mirror_value = malloc(
            _MIRROR_BYTES_PER_TENSOR_PER_LAYER,
            runtime=runtime,
        )
        allocations.extend((key_rows, value_rows, mirror_key, mirror_value))
        for device, host in ((key_rows, keys), (value_rows, values)):
            copy_host_to_device(
                device,
                host_array_ptr(host),
                host.nbytes,
                runtime=runtime,
            )
        for cache in (baseline, candidate_cache):
            state = cache.layer(0)
            runtime.memset(state.key_cache.ptr, 0xA5, state.key_cache.nbytes)
            runtime.memset(state.value_cache.ptr, 0xA5, state.value_cache.nbytes)
            cache.position = start_position - 1
            cache.prepare_rows(
                tuple(range(start_position, start_position + _ROWS))
            )
        runtime.memset(mirror_key.ptr, 0xA5, mirror_key.nbytes)
        runtime.memset(mirror_value.ptr, 0xA5, mirror_value.nbytes)

        baseline.append_rows(
            0,
            key_rows.ptr,
            value_rows.ptr,
            _ROWS,
            library=h7y_runtime_library,
        )
        runtime.device_synchronize()
        baseline_spans = helper._span_snapshot(
            baseline.layer(0).spans,
            runtime,
        )

        candidate = _candidate_writer()
        candidate_state = candidate_cache.layer(0)
        candidate(
            key_rows.ptr,
            value_rows.ptr,
            candidate_state.key_cache.ptr,
            candidate_state.value_cache.ptr,
            mirror_key.ptr,
            mirror_value.ptr,
            candidate_state.append_spans,
            _ROWS,
            _KV_HEADS,
            _HEAD_DIM,
            lane_major_cache_nbytes=mirror_key.nbytes,
            library=h7y_runtime_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        candidate_spans = helper._span_snapshot(
            candidate_state.spans,
            runtime,
        )
        for actual, expected in zip(
            candidate_spans,
            baseline_spans,
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)

        for host, device in (
            (baseline_key_host, baseline.layer(0).key_cache),
            (baseline_value_host, baseline.layer(0).value_cache),
            (candidate_key_host, candidate_state.key_cache),
            (candidate_value_host, candidate_state.value_cache),
            (mirror_key_host, mirror_key),
            (mirror_value_host, mirror_value),
        ):
            copy_device_to_host(
                host_array_ptr(host),
                device,
                host.nbytes,
                runtime=runtime,
            )
        np.testing.assert_array_equal(candidate_key_host, baseline_key_host)
        np.testing.assert_array_equal(candidate_value_host, baseline_value_host)
        expected_key = np.full(natural_shape, poison, dtype=np.uint16)
        expected_value = np.full(natural_shape, poison, dtype=np.uint16)
        expected_key[start_position : start_position + _ROWS] = key_bits
        expected_value[start_position : start_position + _ROWS] = value_bits
        np.testing.assert_array_equal(candidate_key_host, expected_key)
        np.testing.assert_array_equal(candidate_value_host, expected_value)
        expected_mirror_key = np.full(mirror_shape, poison, dtype=np.uint16)
        expected_mirror_value = np.full(mirror_shape, poison, dtype=np.uint16)
        expected_mirror_key[start_position : start_position + _ROWS] = (
            key_bits.reshape(_ROWS, _KV_HEADS, 4, 32).swapaxes(-2, -1)
        )
        expected_mirror_value[start_position : start_position + _ROWS] = (
            value_bits.reshape(_ROWS, _KV_HEADS, 4, 32).swapaxes(-2, -1)
        )
        np.testing.assert_array_equal(mirror_key_host, expected_mirror_key)
        np.testing.assert_array_equal(mirror_value_host, expected_mirror_value)
        baseline.discard_rows()
        candidate_cache.discard_rows()
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        candidate_cache.free()
        baseline.free()
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
