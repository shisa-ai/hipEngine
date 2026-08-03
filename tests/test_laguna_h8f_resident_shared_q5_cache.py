"""WPF-H8F complete resident shared-Q5 F32 cache RED."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels.hip_gfx1100.quant import (
    gguf_q5_k_f32_rocblas_prefill as q5_f32,
)
from hipengine.kernels.registry import KernelKey, is_registered, register, resolve
from hipengine.runtime import gguf_linear
from hipengine.runtime import laguna_gguf_runner as runner_module
from tests.test_gguf_q5_k_f32_rocblas_prefill import _device
from tests.test_laguna_h5y_q5_activation_tile_k_row import (
    _activation_plane_shape,
    _expected_activation_plane,
    _hip_available,
    _suffix,
)
from tests.test_laguna_h7g_q5_padded_compute import (
    _bf16_bits,
    _bf16_to_f32,
    _production_q5_weight,
    _sampled_cpu_gate,
)
from tests.test_laguna_h7h_q5_full_group_compute import _h7h_names
from tests.test_laguna_h8a_resident_q5_global_cache import (
    _FakeRuntime,
    _FakeWeight,
    _FakeWeights,
)

_ROOT = Path(__file__).resolve().parents[1]
_TARGET = _ROOT / (
    "benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8e-"
    "resident-shared-q5-f32-cache-target.json"
)
_TARGET_SHA256 = "aaa9a5bc4145f6f6150df31c2d1fbfc3099ab3b315d59590324b7c077f7dde99"
_HIP_SOURCE_SHA256 = "1a06011ea6e7bda8e0b48fd357cbcbadaff76793a1b5c49bd217cc83d32b7110"

_LAYERS = tuple(range(1, 47))
_SLOTS = ("ffn_gate_shexp", "ffn_up_shexp")
_ROWS = (1, 7, 8, 9, 512)
_IN_FEATURES = 3_072
_OUT_FEATURES = 1_024
_SOURCE_SHAPE = (1_024, 3_072)
_OUTPUT_DTYPE = "bf16"
_WEIGHT_LAYOUT = "tile_k_col"
_COMPUTE_KIND = "full_group_compute"
_COL_TILE = 8
_ROW_BATCH = 4
_PLANE_BYTES = _IN_FEATURES * _OUT_FEATURES * np.dtype(np.float32).itemsize
_SHARED_PLANES = 92
_GLOBAL_PLANES = 24
_TOTAL_PLANES = 116
_SHARED_BYTES = _SHARED_PLANES * _PLANE_BYTES
_TOTAL_BYTES = 1_811_939_328 + _SHARED_BYTES

_SUPPORTED_CAPABILITY = "LAGUNA_Q5_F32_RESIDENT_SHARED_CACHE_SUPPORTED"
_SOURCE_CAPABILITY = "LAGUNA_Q5_F32_RESIDENT_SHARED_CACHE"
_RESOLVER = "resolve_laguna_q5_f32_resident_shared_cache"
_SESSION_PARAMETER = "use_q5_f32_resident_shared_cache"
_INCLUDE_SHARED_PARAMETER = "include_shared"

_EXPECTED_TOPOLOGY = {
    "setup_global_coltile16_producers": 24,
    "setup_shared_coltile8_producers": 92,
    "request_shared_coltile8_producers": 0,
    "request_shared_activation_packs": 46,
    "request_h7h_consumers": 92,
    "request_application_dispatches": 2_063,
    "queues": 1,
    "streams": 1,
    "compiler_processes_allowed": 0,
}
_NO_SALVAGE = (
    "partial-plane",
    "layer",
    "role",
    "prompt",
    "token",
    "route",
    "length",
    "threshold",
    "favorable-rerun",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_cached_build() -> bool:
    return os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _shared_producer_name() -> str:
    return (
        "gguf_q5_k_dequantize_f32_exact_tile_k_col_"
        + _suffix(_COL_TILE, _ROW_BATCH, _OUTPUT_DTYPE)
    )


def _shared_producer_variant() -> str:
    return (
        "raw_f32_exact_tile_k_col_"
        + _suffix(_COL_TILE, _ROW_BATCH, _OUTPUT_DTYPE)
    )


def _candidate_name() -> str:
    _primitive, composite = _h7h_names(
        _COL_TILE,
        _ROW_BATCH,
        _OUTPUT_DTYPE,
        _WEIGHT_LAYOUT,
    )
    return composite.replace("f32_ordered_", "f32_resident_ordered_", 1)


def _candidate_variant() -> str:
    stem = (
        f"weight_major_{_WEIGHT_LAYOUT}_activation_tile_k_row_"
        f"{_COMPUTE_KIND}_{_suffix(_COL_TILE, _ROW_BATCH, _OUTPUT_DTYPE)}"
    )
    return f"f32_resident_ordered_{stem}"


def _control_variant() -> str:
    stem = (
        f"weight_major_{_WEIGHT_LAYOUT}_activation_tile_k_row_"
        f"{_COMPUTE_KIND}_{_suffix(_COL_TILE, _ROW_BATCH, _OUTPUT_DTYPE)}"
    )
    return f"f32_ordered_{stem}"


def _candidate_key(*, backend: str = "hip_gfx1100") -> KernelKey:
    return KernelKey(backend, "linear", "gguf_q5_k", _candidate_variant())


def _control_key(*, backend: str = "hip_gfx1100") -> KernelKey:
    return KernelKey(backend, "linear", "gguf_q5_k", _control_variant())


def _target_weights(*, missing_last: bool = False) -> _FakeWeights:
    weights = _FakeWeights()
    for layer_id in _LAYERS:
        layer = weights.layer(layer_id)
        for slot_index, slot in enumerate(_SLOTS):
            if missing_last and layer_id == _LAYERS[-1] and slot == _SLOTS[-1]:
                continue
            layer.weights[slot] = _FakeWeight(
                raw_ptr=(
                    0x2800_0000
                    + layer_id * 0x20_000
                    + slot_index * 0x10_000
                ),
                source_shape=_SOURCE_SHAPE,
            )
    return weights


def _shared_raw_ptrs(weights: _FakeWeights) -> tuple[int, ...]:
    return tuple(
        weights.layer(layer_id).weight(slot).allocation("raw").tensor.ptr
        for layer_id in _LAYERS
        for slot in _SLOTS
    )


def _install_fake_producers(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[dict[str, Any]],
    *,
    fail_at: int | None = None,
) -> None:
    global_variants = {
        "raw_f32_exact_tile_k_col_"
        + _suffix(16, 5, output_dtype)
        for output_dtype in ("bf16", "f32")
    }
    variants = global_variants | {_shared_producer_variant()}

    def fake_resolve(
        *,
        backend: str,
        layer: str,
        quant: str,
        variant: str,
        missing: str = "raise",
    ):
        del missing
        assert backend == "hip_gfx1100"
        assert layer == "dequant"
        assert quant == "gguf_q5_k"
        assert variant in variants

        def producer(
            qweight_ptr: int,
            out_ptr: int,
            in_features: int,
            out_features: int,
            **kwargs: Any,
        ) -> None:
            calls.append(
                {
                    "variant": variant,
                    "qweight_ptr": int(qweight_ptr),
                    "out_ptr": int(out_ptr),
                    "in_features": int(in_features),
                    "out_features": int(out_features),
                    "kwargs": dict(kwargs),
                }
            )
            if fail_at is not None and len(calls) == fail_at:
                raise RuntimeError("injected H8F shared producer failure")

        return producer

    monkeypatch.setattr(runner_module, "resolve", fake_resolve)


def test_h8f_frozen_target_contract_and_retained_sources() -> None:
    artifact = json.loads(_TARGET.read_text())
    assert _sha256(_TARGET) == _TARGET_SHA256
    assert artifact["target"]["id"] == "WPF-H8F"
    assert artifact["status"] == "accepted_target_only_no_candidate_implementation"
    assert artifact["performance_claim"] is False
    assert artifact["target"]["layers"] == list(_LAYERS)
    assert artifact["target"]["roles"] == list(_SLOTS)
    assert artifact["target"]["tensor_count"] == _SHARED_PLANES
    assert artifact["target"]["plane_bytes"] == _PLANE_BYTES
    assert artifact["target"]["resident_bytes"] == _SHARED_BYTES
    assert artifact["target"]["existing_h8a_planes_before"] == _GLOBAL_PLANES
    assert artifact["target"]["complete_resident_map_planes_after"] == _TOTAL_PLANES
    assert artifact["target"]["request_dispatches_before"] == 2_155
    assert artifact["target"]["request_dispatches_after_model"] == 2_063
    assert artifact["target"]["activation_packs_before_and_after"] == 46
    assert artifact["target"]["h7h_consumers_before_and_after"] == 92
    assert artifact["target"]["new_device_body"] is False
    assert artifact["target"]["new_jit_object"] is False
    assert artifact["target"]["hip_source_change_allowed"] is False
    assert artifact["target"]["consumer_arithmetic_changed"] is False
    assert artifact["decision"]["candidate_implemented"] is False
    assert artifact["decision"]["speed_result_exists"] is False
    assert artifact["memory_feasibility"]["after_exact_m512_free_bytes"] == 3_009_413_120
    assert artifact["memory_feasibility"]["exact_m512_token"] == 2930
    assert artifact["production_trace"]["producer_dispatches_each"] == [92] * 5
    assert artifact["production_trace"]["median_producer_ms"] == pytest.approx(
        3.439745,
        abs=1e-12,
    )
    assert _EXPECTED_TOPOLOGY["request_application_dispatches"] == 2_063
    assert _NO_SALVAGE == (
        "partial-plane",
        "layer",
        "role",
        "prompt",
        "token",
        "route",
        "length",
        "threshold",
        "favorable-rerun",
    )
    hip_source = Path(q5_f32.__file__).with_suffix(".hip")
    assert _sha256(hip_source) == _HIP_SOURCE_SHA256
    assert artifact["source_sha256"][str(hip_source.relative_to(_ROOT))] == (
        _HIP_SOURCE_SHA256
    )


def test_h8f_registry_plane_geometry_policy_and_gfx1151_fail_closed() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151

    # Intentional RED: these bounded package/runtime and registered-composite
    # surfaces do not exist at the committed target boundary.
    assert getattr(hip_gfx1100, _SUPPORTED_CAPABILITY) is True
    assert getattr(hip_gfx1100, _SOURCE_CAPABILITY) is False
    assert not getattr(hip_gfx1151, _SUPPORTED_CAPABILITY, False)
    assert not getattr(hip_gfx1151, _SOURCE_CAPABILITY, False)
    policy_resolver = getattr(runner_module, _RESOLVER)
    assert policy_resolver("hip_gfx1100", None) is False
    assert policy_resolver("hip_gfx1100", True) is True
    assert policy_resolver("hip_gfx1100", False) is False
    assert policy_resolver("hip_gfx1151", None) is False
    with pytest.raises(ValueError, match="supported backend"):
        policy_resolver("hip_gfx1151", True)
    parameters = inspect.signature(
        runner_module.LagunaGGUFResidentSession.__init__
    ).parameters
    assert _SESSION_PARAMETER in parameters
    assert _INCLUDE_SHARED_PARAMETER in inspect.signature(
        runner_module.LagunaQ5F32ResidentGlobalCache.allocate
    ).parameters

    plane = gguf_linear.Q5F32ResidentPlane(
        raw_weight_ptr=0x1000_0000,
        weight_f32_ptr=0x2000_0000,
        weight_f32_nbytes=_PLANE_BYTES,
        in_features=_IN_FEATURES,
        out_features=_OUT_FEATURES,
        output_dtype=_OUTPUT_DTYPE,
        weight_layout=_WEIGHT_LAYOUT,
        col_tile=_COL_TILE,
        row_batch=_ROW_BATCH,
        compute_kind=_COMPUTE_KIND,
    )
    assert plane.ordered_geometry == (
        "weight_major_tile_k_col_activation_tile_k_row_"
        "full_group_compute_coltile8_rowbatch4"
    )
    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    candidate = getattr(q5_f32, _candidate_name())
    assert resolve(
        backend="hip_gfx1100",
        layer="linear",
        quant="gguf_q5_k",
        variant=_candidate_variant(),
    ) is candidate
    assert is_registered(_control_key())
    assert is_registered(_candidate_key())
    assert not is_registered(_candidate_key(backend="hip_gfx1151"))


def test_h8f_owner_publishes_exact_116_plane_map_and_reverse_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = _target_weights()
    runtime = _FakeRuntime()
    calls: list[dict[str, Any]] = []
    _install_fake_producers(monkeypatch, calls)
    before = memory_stats()
    owner = runner_module.LagunaQ5F32ResidentGlobalCache.allocate(
        weights,
        backend="hip_gfx1100",
        library=object(),
        runtime=runtime,
        stream=7,
        include_shared=True,
    )
    try:
        shared_ptrs = _shared_raw_ptrs(weights)
        assert owner.closed is False
        assert owner.shared_enabled is True
        assert owner.global_plane_count == _GLOBAL_PLANES
        assert owner.shared_plane_count == _SHARED_PLANES
        assert owner.allocation_count == _TOTAL_PLANES
        assert len(owner.resident_planes) == _TOTAL_PLANES
        assert owner.nbytes == _TOTAL_BYTES
        assert len(calls) == _TOTAL_PLANES
        assert runtime.synchronizations == 2
        assert tuple(owner.resident_planes)[-92:] == shared_ptrs
        assert [call["variant"] for call in calls[-92:]] == [
            _shared_producer_variant()
        ] * _SHARED_PLANES
        assert [call["in_features"] for call in calls[-92:]] == [
            _IN_FEATURES
        ] * _SHARED_PLANES
        assert [call["out_features"] for call in calls[-92:]] == [
            _OUT_FEATURES
        ] * _SHARED_PLANES
        for raw_ptr in shared_ptrs:
            plane = owner.resident_planes[raw_ptr]
            assert plane.weight_f32_nbytes == _PLANE_BYTES
            assert plane.in_features == _IN_FEATURES
            assert plane.out_features == _OUT_FEATURES
            assert plane.output_dtype == _OUTPUT_DTYPE
            assert plane.weight_layout == _WEIGHT_LAYOUT
            assert plane.col_tile == _COL_TILE
            assert plane.row_batch == _ROW_BATCH
            assert plane.compute_kind == _COMPUTE_KIND
            assert plane.ordered_geometry.endswith(
                "full_group_compute_coltile8_rowbatch4"
            )
        with pytest.raises(TypeError):
            owner.resident_planes[shared_ptrs[0]] = object()
    finally:
        owner.free(runtime=runtime)
    owner.free(runtime=runtime)
    assert runtime.allocations == {}
    assert runtime.free_order == list(reversed(runtime.malloc_order))
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]


def test_h8f_shared_failure_or_incomplete_class_retains_only_complete_h8a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = _target_weights()
    runtime = _FakeRuntime()
    calls: list[dict[str, Any]] = []
    _install_fake_producers(monkeypatch, calls, fail_at=_GLOBAL_PLANES + 7)
    owner = runner_module.LagunaQ5F32ResidentGlobalCache.allocate(
        weights,
        backend="hip_gfx1100",
        library=object(),
        runtime=runtime,
        include_shared=True,
    )
    try:
        assert owner.shared_enabled is False
        assert owner.global_plane_count == _GLOBAL_PLANES
        assert owner.shared_plane_count == 0
        assert owner.allocation_count == _GLOBAL_PLANES
        assert len(owner.resident_planes) == _GLOBAL_PLANES
        assert len(calls) == _GLOBAL_PLANES + 7
        assert runtime.synchronizations == 1
        assert set(runtime.allocations) == set(runtime.malloc_order[:_GLOBAL_PLANES])
        assert runtime.free_order == list(reversed(runtime.malloc_order[_GLOBAL_PLANES:]))
    finally:
        owner.free(runtime=runtime)
    assert runtime.allocations == {}

    incomplete = _target_weights(missing_last=True)
    runtime = _FakeRuntime()
    calls = []
    _install_fake_producers(monkeypatch, calls)
    owner = runner_module.LagunaQ5F32ResidentGlobalCache.allocate(
        incomplete,
        backend="hip_gfx1100",
        library=object(),
        runtime=runtime,
        include_shared=True,
    )
    try:
        assert owner.shared_enabled is False
        assert owner.allocation_count == _GLOBAL_PLANES
        assert len(owner.resident_planes) == _GLOBAL_PLANES
        assert len(calls) == _GLOBAL_PLANES
    finally:
        owner.free(runtime=runtime)
    assert runtime.allocations == {}


def test_h8f_shared_plane_selects_resident_h7h_and_unshared_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del monkeypatch
    raw_ptr = 0x3100_0000
    resident_ptr = 0x3200_0000
    activation_ptr = 0x3300_0000
    transient_ptr = 0x3400_0000
    plane = gguf_linear.Q5F32ResidentPlane(
        raw_weight_ptr=raw_ptr,
        weight_f32_ptr=resident_ptr,
        weight_f32_nbytes=_PLANE_BYTES,
        in_features=_IN_FEATURES,
        out_features=_OUT_FEATURES,
        output_dtype=_OUTPUT_DTYPE,
        weight_layout=_WEIGHT_LAYOUT,
        col_tile=_COL_TILE,
        row_batch=_ROW_BATCH,
        compute_kind=_COMPUTE_KIND,
    )
    immutable = MappingProxyType({raw_ptr: plane})
    candidate_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    fallback_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def candidate(*args: Any, **kwargs: Any) -> None:
        candidate_calls.append((args, dict(kwargs)))

    def fallback(*args: Any, **kwargs: Any) -> None:
        fallback_calls.append((args, dict(kwargs)))

    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    register(_candidate_key(), candidate, replace=True)
    register(_control_key(), fallback, replace=True)
    gguf_linear.clear_gguf_linear_dispatch_cache()
    library = object()
    session = gguf_linear.Q5F32OrderedPrefillSession(
        min_rows=512,
        max_rows=512,
        weight_f32_ptr=transient_ptr,
        weight_f32_nbytes=150_994_944,
        activation_bf16_ptr=activation_ptr,
        activation_bf16_nbytes=10_125_312,
        library=library,
        resident_weight_f32_planes=immutable,
    )
    matching = _FakeWeight(raw_ptr=raw_ptr, source_shape=_SOURCE_SHAPE)
    unshared = _FakeWeight(raw_ptr=raw_ptr + 0x1000, source_shape=_SOURCE_SHAPE)
    kwargs = {
        "rows": 512,
        "in_features": _IN_FEATURES,
        "out_features": _OUT_FEATURES,
        "output_dtype": _OUTPUT_DTYPE,
        "backend": "hip_gfx1100",
        "libraries": {
            "gguf_q5_k": library,
            f"gguf_q5_k:{_candidate_variant()}": library,
        },
    }
    with gguf_linear.q5_f32_ordered_prefill_session(session):
        gguf_linear.launch_gguf_linear(
            matching,
            0x3500_0000,
            0x3600_0000,
            **kwargs,
        )
        gguf_linear.launch_gguf_linear(
            unshared,
            0x3700_0000,
            0x3800_0000,
            **kwargs,
        )
    assert len(candidate_calls) == 1
    assert len(fallback_calls) == 1
    assert candidate_calls[0][0][:4] == (
        0x3500_0000,
        resident_ptr,
        0x3600_0000,
        activation_ptr,
    )
    assert candidate_calls[0][0][4:] == (512, _IN_FEATURES, _OUT_FEATURES)
    assert fallback_calls[0][0][1] == raw_ptr + 0x1000
    assert fallback_calls[0][0][3] == transient_ptr
    assert fallback_calls[0][0][4] == activation_ptr


@pytest.fixture(scope="module")
def h8f_library():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = (
        Path(version_file).read_text(encoding="utf-8").strip()
        if version_file
        else None
    )
    return q5_f32.build_gguf_q5_k_f32_rocblas_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=_require_cached_build(),
    )


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h8f_complete_plane_and_rows_match_retained_h7h(h8f_library: Any) -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    before = memory_stats()
    rng = np.random.default_rng(0x8F00)
    x_bf16 = _bf16_bits(
        rng.normal(0.0, 0.2, size=(max(_ROWS), _IN_FEATURES)).astype(np.float32)
    )
    qweight = _production_q5_weight(_OUT_FEATURES, _IN_FEATURES)
    max_output = np.empty((max(_ROWS), _OUT_FEATURES), dtype=np.uint16)
    activation_host = np.empty(
        _activation_plane_shape(max(_ROWS), _IN_FEATURES, _ROW_BATCH),
        dtype=np.uint16,
    )
    row_major_host = np.empty((_OUT_FEATURES, _IN_FEATURES), dtype=np.float32)
    resident_host = np.empty(
        (_OUT_FEATURES // _COL_TILE, _IN_FEATURES, _COL_TILE),
        dtype=np.float32,
    )
    buffers: list[DeviceBuffer] = []
    baseline_outputs: dict[int, np.ndarray] = {}
    try:
        x_dev = _device(x_bf16, runtime)
        qweight_dev = _device(qweight, runtime)
        row_major_dev = malloc(_PLANE_BYTES, runtime=runtime)
        resident_dev = malloc(_PLANE_BYTES, runtime=runtime)
        activation_dev = malloc(activation_host.nbytes, runtime=runtime)
        baseline_dev = malloc(max_output.nbytes, runtime=runtime)
        candidate_dev = malloc(max_output.nbytes, runtime=runtime)
        buffers.extend((x_dev, qweight_dev, row_major_dev, resident_dev, activation_dev, baseline_dev, candidate_dev))
        runtime.memset(row_major_dev.ptr, 0xA5, row_major_dev.nbytes)
        runtime.memset(resident_dev.ptr, 0x5A, resident_dev.nbytes)
        q5_f32.gguf_q5_k_dequantize_f32_exact(
            qweight_dev.ptr,
            row_major_dev.ptr,
            _IN_FEATURES,
            _OUT_FEATURES,
            library=h8f_library,
            runtime=runtime,
        )
        producer = getattr(q5_f32, _shared_producer_name())
        producer(
            qweight_dev.ptr,
            resident_dev.ptr,
            _IN_FEATURES,
            _OUT_FEATURES,
            library=h8f_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(row_major_host), row_major_dev, row_major_host.nbytes, runtime=runtime)
        copy_device_to_host(host_array_ptr(resident_host), resident_dev, resident_host.nbytes, runtime=runtime)
        expected_plane = row_major_host.reshape(
            _OUT_FEATURES // _COL_TILE,
            _COL_TILE,
            _IN_FEATURES,
        ).transpose(0, 2, 1)
        np.testing.assert_array_equal(resident_host.view(np.uint32), expected_plane.view(np.uint32))
        resident_sha256 = hashlib.sha256(resident_host.tobytes()).hexdigest()
        assert not np.all(resident_host.view(np.uint8) == np.uint8(0x5A))

        _primitive, control_name = _h7h_names(
            _COL_TILE,
            _ROW_BATCH,
            _OUTPUT_DTYPE,
            _WEIGHT_LAYOUT,
        )
        control = getattr(q5_f32, control_name)
        for rows in _ROWS:
            runtime.memset(activation_dev.ptr, 0xA5, activation_dev.nbytes)
            runtime.memset(baseline_dev.ptr, 0x5A, max_output.nbytes)
            control(
                x_dev.ptr,
                qweight_dev.ptr,
                baseline_dev.ptr,
                row_major_dev.ptr,
                activation_dev.ptr,
                rows,
                _IN_FEATURES,
                _OUT_FEATURES,
                library=h8f_library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            baseline = np.empty((rows, _OUT_FEATURES), dtype=np.uint16)
            activation = np.empty(
                _activation_plane_shape(rows, _IN_FEATURES, _ROW_BATCH),
                dtype=np.uint16,
            )
            copy_device_to_host(host_array_ptr(baseline), baseline_dev, baseline.nbytes, runtime=runtime)
            copy_device_to_host(host_array_ptr(activation), activation_dev, activation.nbytes, runtime=runtime)
            np.testing.assert_array_equal(activation, _expected_activation_plane(x_bf16[:rows], _ROW_BATCH))
            _sampled_cpu_gate(
                baseline,
                x_bf16[:rows],
                qweight,
                row_batch=_ROW_BATCH,
                output_dtype=_OUTPUT_DTYPE,
                out_features=_OUT_FEATURES,
            )
            assert np.isfinite(_bf16_to_f32(baseline)).all()
            baseline_outputs[rows] = baseline.copy()

        # Intentional RED only after complete retained producer/H7H controls.
        candidate = getattr(q5_f32, _candidate_name())
        for rows in _ROWS:
            expected = baseline_outputs[rows]
            for _repeat in range(2):
                runtime.memset(activation_dev.ptr, 0xA5, activation_dev.nbytes)
                runtime.memset(candidate_dev.ptr, 0x5A, max_output.nbytes)
                candidate(
                    x_dev.ptr,
                    resident_dev.ptr,
                    candidate_dev.ptr,
                    activation_dev.ptr,
                    rows,
                    _IN_FEATURES,
                    _OUT_FEATURES,
                    library=h8f_library,
                    runtime=runtime,
                )
                runtime.device_synchronize()
                actual = np.empty_like(expected)
                activation = np.empty(
                    _activation_plane_shape(rows, _IN_FEATURES, _ROW_BATCH),
                    dtype=np.uint16,
                )
                copy_device_to_host(host_array_ptr(actual), candidate_dev, actual.nbytes, runtime=runtime)
                copy_device_to_host(host_array_ptr(activation), activation_dev, activation.nbytes, runtime=runtime)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(activation, _expected_activation_plane(x_bf16[:rows], _ROW_BATCH))
                assert np.isfinite(_bf16_to_f32(actual)).all()
        copy_device_to_host(host_array_ptr(resident_host), resident_dev, resident_host.nbytes, runtime=runtime)
        assert hashlib.sha256(resident_host.tobytes()).hexdigest() == resident_sha256
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
