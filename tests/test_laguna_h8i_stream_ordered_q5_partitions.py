"""RED contracts for WPF-H8I exact stream-ordered Q5 partitions."""

from __future__ import annotations

import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

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
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.runtime.laguna_gguf_runner import LagunaQ5F32OrderedScratch
from tests.test_laguna_h5y_q5_activation_tile_k_row import (
    _activation_plane_shape,
    _expected_activation_plane,
    _hip_available,
    _suffix,
)
from tests.test_laguna_h7g_q5_padded_compute import (
    _H5Y_KERNEL,
    _H5Y_KERNEL_SHA256,
    _H7G_KERNEL,
    _declaration,
    _h7g_names,
    _production_q5_weight,
    _sha256 as _text_sha256,
)
from tests.test_laguna_h7h_q5_full_group_compute import (
    _H7G_KERNEL_SHA256,
    _h7h_names,
)
from tests.test_gguf_q5_k_f32_rocblas_prefill import (
    _bf16_bits,
    _bf16_to_f32,
    _device,
)

_TARGET = Path(
    "benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-"
    "post-h8h-streamed-q5-partitions-target.json"
)
_TARGET_SHA256 = "9063b0099905daa49a347ec3934e80ff6178798234a6d9ae98972a735900f556"
_ROWS = (1, 7, 8, 9, 17, 33, 512)
# Geometry, dtype, layout, exact K/N, natural-M512 call weight, retained control,
# current runtime VGPR, and immutable current+8 first-object ceiling.
_ROLES = (
    (8, 4, "bf16", "tile_k_col", 3_072, 1_024, 92, "H7H", 72, 80),
    (8, 12, "bf16", "row_major", 3_072, 12_288, 2, "H7G", 200, 208),
    (16, 5, "bf16", "tile_k_col", 6_144, 3_072, 12, "H7G", 168, 176),
    (12, 8, "bf16", "row_major", 9_216, 3_072, 35, "H7H", 200, 208),
    (16, 5, "f32", "tile_k_col", 3_072, 6_144, 12, "H7G", 168, 176),
    (8, 10, "f32", "tile_k_col", 3_072, 9_216, 35, "H7G", 168, 176),
)
_CANDIDATE_KERNEL = (
    "gguf_q5_k_f32_weight_ordered_weight_major_activation_tile_k_row_"
    "stream_ordered_k_partition_kernel"
)
_STAGE_HELPER = (
    "_launch_q5_f32_weight_ordered_weight_major_stream_ordered_k_partition"
)
_STAGE_SYMBOL = (
    "hipengine_gguf_q5_k_f32_weight_ordered_weight_major_{weight_layout}_"
    "activation_tile_k_row_stream_ordered_k_partition{partition}_{suffix}"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_names(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    weight_layout: str,
) -> tuple[str, str]:
    suffix = _suffix(col_tile, row_batch, output_dtype)
    stem = (
        f"weight_major_{weight_layout}_activation_tile_k_row_"
        f"stream_ordered_k_partitions_{suffix}"
    )
    return (
        f"gguf_q5_k_f32_weight_ordered_{stem}",
        f"gguf_q5_k_f32_ordered_{stem}",
    )


def _candidate_keys(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    weight_layout: str,
    *,
    backend: str = "hip_gfx1100",
) -> tuple[KernelKey, KernelKey]:
    suffix = _suffix(col_tile, row_batch, output_dtype)
    stem = (
        f"weight_major_{weight_layout}_activation_tile_k_row_"
        f"stream_ordered_k_partitions_{suffix}"
    )
    return (
        KernelKey(backend, "linear", "f32_weight", f"ordered_{stem}"),
        KernelKey(backend, "linear", "gguf_q5_k", f"f32_ordered_{stem}"),
    )


def _candidate_functions() -> dict[
    tuple[int, int, str, str],
    tuple[Callable[..., None], Callable[..., None]],
]:
    candidates = {}
    for col_tile, row_batch, output_dtype, weight_layout, *_ in _ROLES:
        primitive_name, composite_name = _candidate_names(
            col_tile,
            row_batch,
            output_dtype,
            weight_layout,
        )
        candidates[(col_tile, row_batch, output_dtype, weight_layout)] = (
            getattr(q5_f32, primitive_name),
            getattr(q5_f32, composite_name),
        )
    return candidates


def _control_composite(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    weight_layout: str,
    control: str,
) -> Callable[..., None]:
    names = (
        _h7h_names(col_tile, row_batch, output_dtype, weight_layout)
        if control == "H7H"
        else _h7g_names(col_tile, row_batch, output_dtype, weight_layout)
    )
    return getattr(q5_f32, names[1])


def _stage_symbol(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    weight_layout: str,
    partition: int,
) -> str:
    return _STAGE_SYMBOL.format(
        weight_layout=weight_layout,
        partition=partition,
        suffix=_suffix(col_tile, row_batch, output_dtype),
    )


def _require_cached_build() -> bool:
    return os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def test_h8i_target_source_and_admission_are_frozen() -> None:
    assert _sha256(_TARGET) == _TARGET_SHA256
    artifact = json.loads(_TARGET.read_text(encoding="utf-8"))
    assert artifact["selection"]["id"] == "WPF-H8I"
    assert artifact["status"] == (
        "selected_exact_target_only_no_candidate_or_speed_result"
    )
    assert artifact["performance_claim"] is False
    assert artifact["operation_contract"]["partition_order"] == [0, 1, 2, 3]
    assert artifact["operation_contract"]["stage_launches"] == 4
    assert artifact["operation_contract"]["raw_pointer_abi"] is True
    assert artifact["operation_contract"]["gfx1151_fail_closed"] is True
    model = artifact["operation_model"]
    assert model["consumer_calls_before"] == 188
    assert model["partition_dispatches_after"] == 752
    assert model["application_dispatches_before"] == 2_155
    assert model["application_dispatches_after"] == 2_719
    assert model["control_workgroups"] == 5_021_440
    assert model["control_compute_waves"] == 20_085_760
    assert model["candidate_compute_waves"] == 20_085_760
    assert model["compute_wave_delta"] == 0
    assert model["extra_global_bytes"] == 8_103_395_328
    assert model["max_f32_accumulation_workspace_bytes"] == 25_165_824
    assert model["not_a_speed_claim"] is True
    admission = artifact["admission"]
    assert admission["red_first"] is True
    assert admission["all_six_roles_inseparable"] is True
    assert admission["each_role_both_clock_positive"] is True
    assert admission["weighted_188_call_aggregate_both_clock_positive"] is True
    assert admission["no_subset_tuning_recompile_or_favorable_rerun"] is True
    assert "resource-rewrite" in admission["forbidden_salvage"]
    assert artifact["decision"] == {
        "candidate_implemented": False,
        "next_action": (
            "Commit this target-only packet, then freeze RED before any "
            "executable source change."
        ),
        "performance_measured": False,
        "production_changed": False,
        "target_selected": True,
    }
    artifact_roles = {
        (
            role["col_tile"],
            role["row_batch"],
            role["dtype"],
            role["weight_layout"],
            role["k"],
            role["n"],
            role["calls"],
            role["control"],
            role["runtime_vgpr"],
            role["candidate_vgpr_ceiling"],
        )
        for role in artifact["roles"]
    }
    assert artifact_roles == set(_ROLES)
    assert artifact["lineage"]["source_sha256"] == {
        "hipengine/kernels/hip_gfx1100/__init__.py": (
            "3638a8fb56d7f87b928bd4f9c8f533f3923381db4d7d7a6e1929ae283b37968d"
        ),
        "hipengine/kernels/hip_gfx1100/quant/gguf_q5_k_f32_rocblas_prefill.hip": (
            "1a06011ea6e7bda8e0b48fd357cbcbadaff76793a1b5c49bd217cc83d32b7110"
        ),
        "hipengine/kernels/hip_gfx1100/quant/gguf_q5_k_f32_rocblas_prefill.py": (
            "fb9b2ae1a88300ac1e754b8c3214310db65d3e2343598b7631ac185ec141f33e"
        ),
        "hipengine/kernels/hip_gfx1151/__init__.py": (
            "a5838ffc8fd8df367cd828f397e701f94f2268c7992d0a5e143c8d7e2b8ba3b3"
        ),
    }
    source = Path(q5_f32.__file__).with_suffix(".hip").read_text(encoding="utf-8")
    h5y = _declaration(source, f"__global__ void {_H5Y_KERNEL}(")
    h7g = _declaration(source, f"__global__ void {_H7G_KERNEL}(")
    assert _text_sha256(h5y) == _H5Y_KERNEL_SHA256
    assert _text_sha256(h7g) == _H7G_KERNEL_SHA256


def test_h8i_registry_static_resources_workspace_and_backend_exclusion() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    candidates = _candidate_functions()
    stage_launcher = getattr(q5_f32, _STAGE_HELPER)
    workspace_nbytes = getattr(
        q5_f32,
        "q5_k_f32_stream_ordered_partition_workspace_nbytes",
    )
    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    assert hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY == (
        hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_H7H_POLICY
    )
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256
    assert inspect.isfunction(stage_launcher)

    source = Path(q5_f32.__file__).with_suffix(".hip").read_text(encoding="utf-8")
    h5y = _declaration(source, f"__global__ void {_H5Y_KERNEL}(")
    h7g = _declaration(source, f"__global__ void {_H7G_KERNEL}(")
    assert _text_sha256(h5y) == _H5Y_KERNEL_SHA256
    assert _text_sha256(h7g) == _H7G_KERNEL_SHA256
    candidate = _declaration(source, f"__global__ void {_CANDIDATE_KERNEL}(")
    assert "__launch_bounds__(32" in source[
        max(0, source.index(f"__global__ void {_CANDIDATE_KERNEL}(") - 256) :
        source.index(f"__global__ void {_CANDIDATE_KERNEL}(")
    ]
    assert "__syncthreads" not in candidate
    assert "__shared__" not in candidate
    assert "__shfl_down" in candidate
    assert "PARTITION * 32 + lane" in candidate
    assert "k += 128" in candidate

    assert q5_f32._Q5_STREAM_ORDERED_K_PARTITION_ROLES == tuple(
        role[:6] for role in _ROLES
    )
    for (
        col_tile,
        row_batch,
        output_dtype,
        weight_layout,
        in_features,
        out_features,
        _,
        _,
        runtime_vgpr,
        ceiling,
    ) in _ROLES:
        assert ceiling == runtime_vgpr + 8
        primitive, composite = candidates[
            (col_tile, row_batch, output_dtype, weight_layout)
        ]
        assert primitive is not composite
        assert tuple(inspect.signature(primitive).parameters)[:7] == (
            "activation_ptr",
            "weight_f32_ptr",
            "out_ptr",
            "accumulation_ptr",
            "rows",
            "in_features",
            "out_features",
        )
        assert tuple(inspect.signature(composite).parameters)[:9] == (
            "x_ptr",
            "qweight_ptr",
            "out_ptr",
            "weight_f32_ptr",
            "activation_ptr",
            "accumulation_ptr",
            "rows",
            "in_features",
            "out_features",
        )
        for key, function in zip(
            _candidate_keys(col_tile, row_batch, output_dtype, weight_layout),
            (primitive, composite),
            strict=True,
        ):
            assert resolve(
                backend=key.backend,
                layer=key.layer,
                quant=key.quant,
                variant=key.variant,
            ) is function
            assert not is_registered(
                KernelKey("hip_gfx1151", key.layer, key.quant, key.variant)
            )
        expected_workspace = 512 * out_features * 4 if output_dtype == "bf16" else 0
        assert workspace_nbytes(512, out_features, output_dtype) == expected_workspace
        assert expected_workspace <= 25_165_824
        for partition in range(4):
            assert source.count(
                _stage_symbol(
                    col_tile,
                    row_batch,
                    output_dtype,
                    weight_layout,
                    partition,
                )
            ) == 1


def test_h8i_strict_preflight_raw_abi_and_four_stage_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _candidate_functions()
    build_attempts = 0

    def _unexpected_build(**_kwargs: object) -> None:
        nonlocal build_attempts
        build_attempts += 1
        raise AssertionError("invalid H8I role reached HIP loading")

    monkeypatch.setattr(
        q5_f32,
        "build_gguf_q5_k_f32_rocblas_prefill",
        _unexpected_build,
    )
    for (
        col_tile,
        row_batch,
        output_dtype,
        weight_layout,
        in_features,
        out_features,
        *_rest,
    ) in _ROLES:
        primitive, _ = candidates[(col_tile, row_batch, output_dtype, weight_layout)]
        valid_pointers = [0x1000, 0x2000, 0x3000, 0x4000]
        if output_dtype == "f32":
            valid_pointers[3] = valid_pointers[2]
        invalid_shapes = (
            (0, in_features, out_features, "rows must be positive"),
            (17, in_features - 256, out_features, f"exactly {in_features}"),
            (17, in_features, out_features - col_tile, f"exactly {out_features}"),
        )
        for rows, hidden, outputs, message in invalid_shapes:
            with pytest.raises(ValueError, match=message):
                primitive(*valid_pointers, rows, hidden, outputs)
        for pointer_index in range(3):
            pointers = list(valid_pointers)
            pointers[pointer_index] = 0
            with pytest.raises(ValueError, match="pointer must be non-zero"):
                primitive(*pointers, 17, in_features, out_features)
        if output_dtype == "bf16":
            pointers = list(valid_pointers)
            pointers[3] = 0
            with pytest.raises(ValueError, match="accumulation pointer must be non-zero"):
                primitive(*pointers, 17, in_features, out_features)
            pointers[3] = pointers[2]
            with pytest.raises(ValueError, match="must not alias output"):
                primitive(*pointers, 17, in_features, out_features)
        else:
            pointers = list(valid_pointers)
            pointers[3] += 4
            with pytest.raises(ValueError, match="must alias output"):
                primitive(*pointers, 17, in_features, out_features)
    assert build_attempts == 0

    calls: list[str] = []

    class FakeFn:
        argtypes = None
        restype = None

        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def __call__(self, *_args: object) -> int:
            calls.append(self.symbol)
            return 0

    class FakeLibrary:
        def __getattr__(self, symbol: str) -> FakeFn:
            return FakeFn(symbol)

    library = FakeLibrary()
    runtime = SimpleNamespace(check=lambda error: pytest.fail(f"HIP error {error}"))
    for (
        col_tile,
        row_batch,
        output_dtype,
        weight_layout,
        in_features,
        out_features,
        *_rest,
    ) in _ROLES:
        calls.clear()
        primitive, composite = candidates[
            (col_tile, row_batch, output_dtype, weight_layout)
        ]
        out_ptr = 0x3000
        accumulation_ptr = 0x4000 if output_dtype == "bf16" else out_ptr
        expected_stages = [
            _stage_symbol(
                col_tile,
                row_batch,
                output_dtype,
                weight_layout,
                partition,
            )
            for partition in range(4)
        ]
        primitive(
            0x1000,
            0x2000,
            out_ptr,
            accumulation_ptr,
            17,
            in_features,
            out_features,
            stream=0x5000,
            library=library,
            runtime=runtime,
        )
        assert calls == expected_stages
        calls.clear()
        composite(
            0x6000,
            0x7000,
            out_ptr,
            0x2000,
            0x1000,
            accumulation_ptr,
            17,
            in_features,
            out_features,
            stream=0x5000,
            library=library,
            runtime=runtime,
        )
        assert len(calls) == 6
        assert calls[-4:] == expected_stages


_LIBM = ctypes.CDLL("libm.so.6")
_FMAF = _LIBM.fmaf
_FMAF.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
_FMAF.restype = ctypes.c_float


def _weight_value(
    weight: np.ndarray,
    *,
    weight_layout: str,
    in_features: int,
    col_tile: int,
    out_col: int,
    k: int,
) -> np.float32:
    flat = weight.reshape(-1)
    if weight_layout == "tile_k_col":
        out_tile, tile_col = divmod(out_col, col_tile)
        index = out_tile * in_features * col_tile + k * col_tile + tile_col
    else:
        index = out_col * in_features + k
    return np.float32(flat[index])


def _partition_totals(
    x_bf16: np.ndarray,
    weight: np.ndarray,
    *,
    row: int,
    out_col: int,
    col_tile: int,
    in_features: int,
    weight_layout: str,
) -> tuple[np.float32, ...]:
    x = _bf16_to_f32(x_bf16[row])
    partials = []
    for partition in range(4):
        lanes = np.zeros(32, dtype=np.float32)
        for lane in range(32):
            acc = np.float32(0.0)
            for k in range(partition * 32 + lane, in_features, 128):
                acc = np.float32(
                    _FMAF(
                        ctypes.c_float(x[k]),
                        ctypes.c_float(
                            _weight_value(
                                weight,
                                weight_layout=weight_layout,
                                in_features=in_features,
                                col_tile=col_tile,
                                out_col=out_col,
                                k=k,
                            )
                        ),
                        ctypes.c_float(acc),
                    )
                )
            lanes[lane] = acc
        for offset in (16, 8, 4, 2, 1):
            before = lanes.copy()
            for lane in range(32 - offset):
                lanes[lane] = np.float32(before[lane] + before[lane + offset])
        partials.append(lanes[0])
    total = np.float32(0.0)
    totals = []
    for partial in partials:
        total = np.float32(total + partial)
        totals.append(total)
    return tuple(totals)


def _copy_f32_element(buffer: DeviceBuffer, index: int, runtime: Any) -> np.float32:
    host = np.empty(1, dtype=np.float32)
    view = DeviceBuffer(ptr=buffer.ptr + index * 4, nbytes=4)
    copy_device_to_host(host_array_ptr(host), view, runtime=runtime)
    return host[0]


def _run_exact_dtype(output_dtype: str) -> None:
    from hipengine.core.hip import get_hip_runtime

    _candidate_functions()
    stage_launcher = getattr(q5_f32, _STAGE_HELPER)
    runtime = get_hip_runtime()
    before = memory_stats()
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = (
        Path(version_file).read_text(encoding="utf-8").strip()
        if version_file
        else None
    )
    library = q5_f32.build_gguf_q5_k_f32_rocblas_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=_require_cached_build(),
    )
    for (
        col_tile,
        row_batch,
        role_dtype,
        weight_layout,
        in_features,
        out_features,
        call_weight,
        control,
        *_resources,
    ) in _ROLES:
        if role_dtype != output_dtype:
            continue
        assert call_weight > 0
        qweight = _production_q5_weight(out_features, in_features)
        weight_host = np.empty((in_features * out_features,), dtype=np.float32)
        role_buffers: list[DeviceBuffer] = []
        try:
            qweight_dev = _device(qweight, runtime)
            weight_dev = malloc(weight_host.nbytes, runtime=runtime)
            role_buffers.extend((qweight_dev, weight_dev))
            copied_weight = False
            for rows in _ROWS:
                rng = np.random.default_rng(
                    0x8_1000 + rows + 17 * in_features + out_features
                )
                x_bf16 = _bf16_bits(
                    rng.normal(0.0, 0.2, (rows, in_features)).astype(np.float32)
                )
                host_dtype = np.uint16 if output_dtype == "bf16" else np.float32
                expected = np.empty((rows, out_features), dtype=host_dtype)
                actual = np.empty_like(expected)
                plane_shape = _activation_plane_shape(rows, in_features, row_batch)
                plane = np.empty(plane_shape, dtype=np.uint16)
                expected_plane = _expected_activation_plane(x_bf16, row_batch)
                buffers: list[DeviceBuffer] = []
                try:
                    x_dev = _device(x_bf16, runtime)
                    expected_dev = malloc(expected.nbytes, runtime=runtime)
                    actual_dev = malloc(actual.nbytes, runtime=runtime)
                    activation_dev = malloc(plane.nbytes, runtime=runtime)
                    accumulation_dev = (
                        malloc(rows * out_features * 4, runtime=runtime)
                        if output_dtype == "bf16"
                        else actual_dev
                    )
                    buffers.extend((x_dev, expected_dev, actual_dev, activation_dev))
                    if accumulation_dev is not actual_dev:
                        buffers.append(accumulation_dev)
                    control_composite = _control_composite(
                        col_tile,
                        row_batch,
                        output_dtype,
                        weight_layout,
                        control,
                    )
                    control_composite(
                        x_dev.ptr,
                        qweight_dev.ptr,
                        expected_dev.ptr,
                        weight_dev.ptr,
                        activation_dev.ptr,
                        rows,
                        in_features,
                        out_features,
                        library=library,
                        runtime=runtime,
                    )
                    runtime.device_synchronize()
                    copy_device_to_host(
                        host_array_ptr(expected),
                        expected_dev,
                        expected.nbytes,
                        runtime=runtime,
                    )
                    copy_device_to_host(
                        host_array_ptr(plane),
                        activation_dev,
                        plane.nbytes,
                        runtime=runtime,
                    )
                    np.testing.assert_array_equal(plane, expected_plane)
                    if not copied_weight:
                        copy_device_to_host(
                            host_array_ptr(weight_host),
                            weight_dev,
                            weight_host.nbytes,
                            runtime=runtime,
                        )
                        copied_weight = True

                    runtime.memset(actual_dev.ptr, 0x5A, actual.nbytes)
                    if accumulation_dev is not actual_dev:
                        runtime.memset(
                            accumulation_dev.ptr,
                            0xA5,
                            accumulation_dev.nbytes,
                        )
                    sample_row = rows - 1
                    sample_col = out_features // 2 + col_tile - 1
                    totals = _partition_totals(
                        x_bf16,
                        weight_host,
                        row=sample_row,
                        out_col=sample_col,
                        col_tile=col_tile,
                        in_features=in_features,
                        weight_layout=weight_layout,
                    )
                    for partition in range(4):
                        stage_launcher(
                            partition=partition,
                            col_tile=col_tile,
                            row_batch=row_batch,
                            output_dtype=output_dtype,
                            weight_layout=weight_layout,
                            activation_ptr=activation_dev.ptr,
                            weight_f32_ptr=weight_dev.ptr,
                            out_ptr=actual_dev.ptr,
                            accumulation_ptr=accumulation_dev.ptr,
                            rows=rows,
                            in_features=in_features,
                            out_features=out_features,
                            library=library,
                            runtime=runtime,
                        )
                        runtime.device_synchronize()
                        if output_dtype == "f32" or partition < 3:
                            index = sample_row * out_features + sample_col
                            observed = _copy_f32_element(
                                accumulation_dev,
                                index,
                                runtime,
                            )
                            assert observed.view(np.uint32) == totals[partition].view(
                                np.uint32
                            )
                    copy_device_to_host(
                        host_array_ptr(actual),
                        actual_dev,
                        actual.nbytes,
                        runtime=runtime,
                    )
                    if output_dtype == "bf16":
                        np.testing.assert_array_equal(actual, expected)
                        assert np.isfinite(_bf16_to_f32(actual)).all()
                    else:
                        np.testing.assert_array_equal(
                            actual.view(np.uint32), expected.view(np.uint32)
                        )
                        assert np.isfinite(actual).all()
                    assert not np.all(actual.view(np.uint8) == 0x5A)
                    copy_device_to_host(
                        host_array_ptr(plane),
                        activation_dev,
                        plane.nbytes,
                        runtime=runtime,
                    )
                    np.testing.assert_array_equal(plane, expected_plane)
                finally:
                    for buffer in reversed(buffers):
                        free(buffer, runtime=runtime)
        finally:
            for buffer in reversed(role_buffers):
                free(buffer, runtime=runtime)
    after = memory_stats()
    assert after["active_allocations"] == before["active_allocations"]
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h8i_bf16_roles_exact_partials_outputs_poison_and_lifecycle() -> None:
    _run_exact_dtype("bf16")


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h8i_f32_roles_exact_partials_outputs_poison_and_lifecycle() -> None:
    _run_exact_dtype("f32")
