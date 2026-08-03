"""RED contracts for WPF-H8N exact Q5 twin-team weight staging."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
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
    _sampled_cpu_gate,
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

_ROOT = Path(__file__).resolve().parents[1]
_TARGET = (
    _ROOT
    / "benchmarks/results/"
    "2026-08-03-gfx1100-laguna-q2-xl-post-h8m-"
    "q5-twin-team-weight-staging-target.json"
)
_TARGET_SHA256 = "d1d0420f79445385a50b231d78926048e35ed776f561a80be1a393d5105a0a69"
_ROWS = (1, 4, 5, 7, 8, 9, 10, 12, 13, 17, 33, 512)
# Candidate geometry/dtype/layout, exact K/N, call weight, retained control,
# retained geometry, and immutable metadata/runtime VGPR plus fixed-LDS bounds.
_ROLES = (
    (8, 4, "bf16", "tile_k_col", 3_072, 1_024, 92, "H7H", 8, 4, 80, 9_216),
    (8, 8, "bf16", "row_major", 3_072, 12_288, 2, "H7G", 8, 12, 144, 10_240),
    (16, 5, "bf16", "tile_k_col", 6_144, 3_072, 12, "H7G", 16, 5, 176, 18_944),
    (8, 8, "bf16", "row_major", 9_216, 3_072, 35, "H7H", 12, 8, 144, 10_240),
    (16, 5, "f32", "tile_k_col", 3_072, 6_144, 12, "H7G", 16, 5, 176, 18_944),
    (8, 10, "f32", "tile_k_col", 3_072, 9_216, 35, "H7G", 8, 10, 176, 10_752),
)
_CANDIDATE_KERNEL = (
    "gguf_q5_k_f32_weight_ordered_weight_major_activation_tile_k_row_"
    "twin_team_weight_staging_kernel"
)
_ROLE_LIST = "_Q5_TWIN_TEAM_WEIGHT_STAGING_ROLES"
_SYMBOL = (
    "hipengine_gguf_q5_k_f32_weight_ordered_weight_major_{weight_layout}_"
    "activation_tile_k_row_twin_team_weight_staging_k{in_features}_n{out_features}_"
    "{suffix}"
)
_PHYSICAL_CONTRACT = {
    (role[2], role[4], role[5]): {
        "local_size": 256,
        "metadata_vgpr_max": role[10],
        "runtime_vgpr_max": role[10],
        "fixed_lds_bytes": role[11],
        "private_bytes": 0,
        "vgpr_spills": 0,
        "sgpr_spills": 0,
        "runtime_scratch_bytes": 0,
        "lexical_barriers": 2,
        "dynamic_barriers": role[4] // 128 + 1,
    }
    for role in _ROLES
}
_TRACE_CONTRACT = {
    (role[2], role[4], role[5]): {
        "symbol": _SYMBOL.format(
            weight_layout=role[3],
            in_features=role[4],
            out_features=role[5],
            suffix=_suffix(role[0], role[1], role[2]),
        ),
        "calls": role[6],
        "grid_x": math.ceil(math.ceil(512 / role[1]) / 2) * (role[5] // role[0]),
        "local_size": 256,
        "lds_bytes": role[11],
        "runtime_vgpr_max": role[10],
        "scratch_bytes": 0,
    }
    for role in _ROLES
}
_TIMING_CONTRACT = {
    "warmups": 5,
    "samples": 15,
    "launches_per_sample": 5,
    "order": "counter_rotated",
    "required_clocks": ("hip_event", "synchronized_wall"),
    "weighted_calls": 188,
    "require_every_role_both_clocks": True,
    "require_weighted_aggregate_both_clocks": True,
    "allow_role_subset": False,
    "allow_dtype_or_shape_subset": False,
    "allow_geometry_or_buffer_change": False,
    "allow_k_tile_change": False,
    "allow_resource_rewrite": False,
    "allow_recompile": False,
    "allow_favorable_rerun": False,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_stem(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    weight_layout: str,
    in_features: int,
    out_features: int,
) -> str:
    return (
        f"weight_major_{weight_layout}_activation_tile_k_row_"
        f"twin_team_weight_staging_k{in_features}_n{out_features}_"
        f"{_suffix(col_tile, row_batch, output_dtype)}"
    )


def _candidate_names(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    weight_layout: str,
    in_features: int,
    out_features: int,
) -> tuple[str, str]:
    stem = _candidate_stem(
        col_tile,
        row_batch,
        output_dtype,
        weight_layout,
        in_features,
        out_features,
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
    in_features: int,
    out_features: int,
    *,
    backend: str = "hip_gfx1100",
) -> tuple[KernelKey, KernelKey]:
    stem = _candidate_stem(
        col_tile,
        row_batch,
        output_dtype,
        weight_layout,
        in_features,
        out_features,
    )
    return (
        KernelKey(backend, "linear", "f32_weight", f"ordered_{stem}"),
        KernelKey(backend, "linear", "gguf_q5_k", f"f32_ordered_{stem}"),
    )


def _candidate_functions() -> dict[
    tuple[str, int, int], tuple[Callable[..., None], Callable[..., None]]
]:
    candidates = {}
    for col_tile, row_batch, dtype, layout, hidden, outputs, *_ in _ROLES:
        primitive_name, composite_name = _candidate_names(
            col_tile,
            row_batch,
            dtype,
            layout,
            hidden,
            outputs,
        )
        candidates[(dtype, hidden, outputs)] = (
            getattr(q5_f32, primitive_name),
            getattr(q5_f32, composite_name),
        )
    return candidates


def _control_names(role: tuple[Any, ...]) -> tuple[str, str]:
    _, _, dtype, layout, _, _, _, control, control_col, control_rows, *_ = role
    factory = _h7h_names if control == "H7H" else _h7g_names
    return factory(control_col, control_rows, dtype, layout)


def _symbol(role: tuple[Any, ...]) -> str:
    col_tile, row_batch, dtype, layout, hidden, outputs, *_ = role
    return _SYMBOL.format(
        weight_layout=layout,
        in_features=hidden,
        out_features=outputs,
        suffix=_suffix(col_tile, row_batch, dtype),
    )


def _require_cached_build() -> bool:
    return os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def test_h8n_target_arithmetic_resources_trace_and_admission_are_frozen() -> None:
    assert _sha256(_TARGET) == _TARGET_SHA256
    artifact = json.loads(_TARGET.read_text(encoding="utf-8"))
    assert artifact["status"] == "target_selected_no_candidate_no_speed_result"
    assert artifact["performance_claim"] is False
    assert artifact["operation_contract"]["id"] == "WPF-H8N"
    assert artifact["operation_contract"]["candidate_name_suffix"] == (
        "twin_team_weight_staging"
    )
    assert artifact["operation_contract"]["raw_pointer_abi"] is True
    assert artifact["operation_contract"]["gfx1151_fail_closed"] is True
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

    model = artifact["operation_model"]
    assert model["calls"] == 188
    assert model["current_workgroups"] == 5_021_440
    assert model["candidate_workgroups"] == 2_689_792
    assert model["current_weight_bytes"] == 807_571_292_160
    assert model["candidate_weight_bytes"] == 407_862_509_568
    assert model["useful_fmas"] == 1_433_445_335_040
    assert model["current_barrier_epochs"] == 5_021_440
    assert model["candidate_barrier_epochs"] == 90_764_032
    assert math.isclose(model["workgroup_reduction_percent"], 46.43385164414988)
    assert math.isclose(
        model["logical_f32_weight_byte_reduction_percent"],
        49.495169834839515,
    )
    assert math.isclose(model["barrier_epoch_multiplier"], 18.07529951567678)
    assert model["not_cache_traffic_or_speed_result"] is True

    artifact_roles = {}
    for row in artifact["operation_contract"]["fixed_role_geometries"]:
        geometry = row["candidate_team_geometry"]
        bounds = row["candidate_first_object_bounds"]
        key = (row["dtype"], row["shape"]["k"], row["shape"]["n"])
        artifact_roles[key] = (
            geometry["col_tile"],
            geometry["row_batch"],
            row["calls"],
            bounds["metadata_vgpr_max"],
            bounds["fixed_lds_bytes"],
            row["candidate_workgroups_per_call"],
        )
    assert artifact_roles == {
        (role[2], role[4], role[5]): (
            role[0],
            role[1],
            role[6],
            role[10],
            role[11],
            _TRACE_CONTRACT[(role[2], role[4], role[5])]["grid_x"],
        )
        for role in _ROLES
    }
    assert tuple(_PHYSICAL_CONTRACT) == tuple(_TRACE_CONTRACT)
    assert sum(facts["calls"] for facts in _TRACE_CONTRACT.values()) == 188
    assert {facts["local_size"] for facts in _TRACE_CONTRACT.values()} == {256}
    assert len({facts["symbol"] for facts in _TRACE_CONTRACT.values()}) == 6
    assert all(facts["lexical_barriers"] == 2 for facts in _PHYSICAL_CONTRACT.values())
    assert all(facts["private_bytes"] == 0 for facts in _PHYSICAL_CONTRACT.values())
    assert all(facts["vgpr_spills"] == 0 for facts in _PHYSICAL_CONTRACT.values())
    assert all(facts["sgpr_spills"] == 0 for facts in _PHYSICAL_CONTRACT.values())
    assert all(
        facts["runtime_scratch_bytes"] == 0
        for facts in _PHYSICAL_CONTRACT.values()
    )
    assert _TIMING_CONTRACT["weighted_calls"] == 188
    assert _TIMING_CONTRACT["require_every_role_both_clocks"]
    assert _TIMING_CONTRACT["require_weighted_aggregate_both_clocks"]
    assert not any(
        _TIMING_CONTRACT[key]
        for key in _TIMING_CONTRACT
        if key.startswith("allow_")
    )

    source = Path(q5_f32.__file__).with_suffix(".hip").read_text(encoding="utf-8")
    h5y = _declaration(source, f"__global__ void {_H5Y_KERNEL}(")
    h7g = _declaration(source, f"__global__ void {_H7G_KERNEL}(")
    assert _text_sha256(h5y) == _H5Y_KERNEL_SHA256
    assert _text_sha256(h7g) == _H7G_KERNEL_SHA256
    assert artifact["lineage"]["source_sha256"] == {
        "hipengine/kernels/hip_gfx1100/quant/gguf_q5_k_f32_rocblas_prefill.hip": (
            "1a06011ea6e7bda8e0b48fd357cbcbadaff76793a1b5c49bd217cc83d32b7110"
        ),
        "hipengine/kernels/hip_gfx1100/quant/gguf_q5_k_f32_rocblas_prefill.py": (
            "fb9b2ae1a88300ac1e754b8c3214310db65d3e2343598b7631ac185ec141f33e"
        ),
        "hipengine/kernels/hip_gfx1151/__init__.py": (
            "a5838ffc8fd8df367cd828f397e701f94f2268c7992d0a5e143c8d7e2b8ba3b3"
        ),
        "hipengine/runtime/qwen35_paro_runner.py": (
            "dd08b9f91419bcd4ba0e0962a8f9dfca0b4c2f068224eb7a85880562ca1128fe"
        ),
    }


def test_h8n_registry_source_physical_contract_and_backend_exclusion() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    # Intentional RED: resolve every separately shape-qualified wrapper first.
    candidates = _candidate_functions()
    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    assert hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY == (
        hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_H7H_POLICY
    )
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256
    expected_roles = tuple(role[:6] for role in _ROLES)
    assert getattr(q5_f32, _ROLE_LIST) == expected_roles

    source = Path(q5_f32.__file__).with_suffix(".hip").read_text(encoding="utf-8")
    h5y = _declaration(source, f"__global__ void {_H5Y_KERNEL}(")
    h7g = _declaration(source, f"__global__ void {_H7G_KERNEL}(")
    assert _text_sha256(h5y) == _H5Y_KERNEL_SHA256
    assert _text_sha256(h7g) == _H7G_KERNEL_SHA256
    candidate = _declaration(source, f"__global__ void {_CANDIDATE_KERNEL}(")
    anchor = source.index(f"__global__ void {_CANDIDATE_KERNEL}(")
    assert "__launch_bounds__(256, 1)" in source[max(0, anchor - 256) : anchor]
    assert candidate.count("__syncthreads();") == 2
    assert "__shared__ float weight_slabs[2][128][COL_TILE];" in candidate
    assert "__shared__ float wave_sums[2][4][ROW_BATCH][COL_TILE];" in candidate
    assert "const int team = tid >> 7;" in candidate
    assert "const int team_tid = tid & 127;" in candidate
    assert "const int row_group = pair_group * 2 + team;" in candidate
    assert "const bool team_active = row_group < row_groups;" in candidate
    assert "if (team == 0)" in candidate
    assert "k_base += 128" in candidate
    assert "slab ^= 1" in candidate
    assert "fmaf(input_value, staged_weights[col], acc[row_index][col])" in candidate
    assert "__shfl_down" in candidate
    assert "for (int wave_index = 0; wave_index < 4; ++wave_index)" in candidate

    for role in _ROLES:
        col_tile, row_batch, dtype, layout, hidden, outputs, *_ = role
        primitive, composite = candidates[(dtype, hidden, outputs)]
        assert primitive is not composite
        assert tuple(inspect.signature(primitive).parameters)[:6] == (
            "activation_ptr",
            "weight_f32_ptr",
            "out_ptr",
            "rows",
            "in_features",
            "out_features",
        )
        assert tuple(inspect.signature(composite).parameters)[:8] == (
            "x_ptr",
            "qweight_ptr",
            "out_ptr",
            "weight_f32_ptr",
            "activation_ptr",
            "rows",
            "in_features",
            "out_features",
        )
        for key, function in zip(
            _candidate_keys(
                col_tile,
                row_batch,
                dtype,
                layout,
                hidden,
                outputs,
            ),
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
        assert source.count(_symbol(role)) == 1


def test_h8n_strict_preflight_and_one_shape_qualified_launch_per_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _candidate_functions()
    build_attempts = 0

    def _unexpected_build(**_kwargs: object) -> None:
        nonlocal build_attempts
        build_attempts += 1
        raise AssertionError("invalid H8N role reached HIP loading")

    monkeypatch.setattr(
        q5_f32,
        "build_gguf_q5_k_f32_rocblas_prefill",
        _unexpected_build,
    )
    for role in _ROLES:
        col_tile, _, dtype, _, hidden, outputs, *_ = role
        primitive, composite = candidates[(dtype, hidden, outputs)]
        for function, pointers in (
            (primitive, (0x1000, 0x2000, 0x3000)),
            (composite, (0x4000, 0x5000, 0x3000, 0x2000, 0x1000)),
        ):
            invalid = (
                (0, hidden, outputs, "rows must be positive"),
                (17, hidden - 256, outputs, f"exactly {hidden}"),
                (17, hidden, outputs - col_tile, f"exactly {outputs}"),
            )
            for rows, in_features, out_features, message in invalid:
                with pytest.raises(ValueError, match=message):
                    function(*pointers, rows, in_features, out_features)
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
    for role in _ROLES:
        _, _, dtype, _, hidden, outputs, *_ = role
        primitive, _ = candidates[(dtype, hidden, outputs)]
        calls.clear()
        primitive(
            0x1000,
            0x2000,
            0x3000,
            17,
            hidden,
            outputs,
            stream=0x6000,
            library=library,
            runtime=runtime,
        )
        assert calls == [_symbol(role)]


def _weight_producer(role: tuple[Any, ...]) -> Callable[..., None]:
    col_tile, row_batch, dtype, layout, *_ = role
    if layout == "tile_k_col":
        return q5_f32._Q5_TILE_K_COL_PRODUCERS[(col_tile, row_batch, dtype)]
    return q5_f32.gguf_q5_k_dequantize_f32_exact


def _pack(
    col_tile: int,
    row_batch: int,
    dtype: str,
    layout: str,
) -> Callable[..., None]:
    return q5_f32._Q5_ACTIVATION_TILE_K_ROW_PACKS[
        (col_tile, row_batch, dtype, layout)
    ]


def _copy(buffer: DeviceBuffer, target: np.ndarray, runtime: Any) -> None:
    copy_device_to_host(
        host_array_ptr(target),
        buffer,
        target.nbytes,
        runtime=runtime,
    )


def _run_exact_dtype(output_dtype: str) -> None:
    from hipengine.core.hip import get_hip_runtime

    # Intentional RED before HIP build, weight construction, or allocation.
    candidates = _candidate_functions()
    runtime = get_hip_runtime()
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
    before = memory_stats()
    for role in _ROLES:
        (
            col_tile,
            row_batch,
            dtype,
            layout,
            hidden,
            outputs,
            call_weight,
            _,
            control_col,
            control_rows,
            *_,
        ) = role
        if dtype != output_dtype:
            continue
        assert call_weight > 0
        qweight = _production_q5_weight(outputs, hidden)
        role_buffers: list[DeviceBuffer] = []
        try:
            qweight_dev = _device(qweight, runtime)
            weight_dev = malloc(hidden * outputs * 4, runtime=runtime)
            role_buffers.extend((qweight_dev, weight_dev))
            _weight_producer(role)(
                qweight_dev.ptr,
                weight_dev.ptr,
                hidden,
                outputs,
                library=library,
                runtime=runtime,
            )
            runtime.device_synchronize()
            control_primitive = getattr(q5_f32, _control_names(role)[0])
            candidate_primitive = candidates[(dtype, hidden, outputs)][0]
            control_pack = _pack(control_col, control_rows, dtype, layout)
            candidate_pack = _pack(col_tile, row_batch, dtype, layout)

            for rows in _ROWS:
                rng = np.random.default_rng(
                    0x8_2000 + rows + 17 * hidden + outputs
                )
                x_bf16 = _bf16_bits(
                    rng.normal(0.0, 0.2, (rows, hidden)).astype(np.float32)
                )
                host_dtype = np.uint16 if dtype == "bf16" else np.float32
                expected = np.empty((rows, outputs), dtype=host_dtype)
                actual = np.empty_like(expected)
                repeated = np.empty_like(expected)
                control_plane = np.empty(
                    _activation_plane_shape(rows, hidden, control_rows),
                    dtype=np.uint16,
                )
                candidate_plane = np.empty(
                    _activation_plane_shape(rows, hidden, row_batch),
                    dtype=np.uint16,
                )
                buffers: list[DeviceBuffer] = []
                try:
                    x_dev = _device(x_bf16, runtime)
                    expected_dev = malloc(expected.nbytes, runtime=runtime)
                    actual_dev = malloc(actual.nbytes, runtime=runtime)
                    control_activation_dev = malloc(
                        control_plane.nbytes,
                        runtime=runtime,
                    )
                    candidate_activation_dev = malloc(
                        candidate_plane.nbytes,
                        runtime=runtime,
                    )
                    buffers.extend(
                        (
                            x_dev,
                            expected_dev,
                            actual_dev,
                            control_activation_dev,
                            candidate_activation_dev,
                        )
                    )
                    control_pack(
                        x_dev.ptr,
                        control_activation_dev.ptr,
                        rows,
                        hidden,
                        library=library,
                        runtime=runtime,
                    )
                    candidate_pack(
                        x_dev.ptr,
                        candidate_activation_dev.ptr,
                        rows,
                        hidden,
                        library=library,
                        runtime=runtime,
                    )
                    control_primitive(
                        control_activation_dev.ptr,
                        weight_dev.ptr,
                        expected_dev.ptr,
                        rows,
                        hidden,
                        outputs,
                        library=library,
                        runtime=runtime,
                    )
                    runtime.memset(actual_dev.ptr, 0x5A, actual.nbytes)
                    candidate_primitive(
                        candidate_activation_dev.ptr,
                        weight_dev.ptr,
                        actual_dev.ptr,
                        rows,
                        hidden,
                        outputs,
                        library=library,
                        runtime=runtime,
                    )
                    runtime.device_synchronize()
                    _copy(expected_dev, expected, runtime)
                    _copy(actual_dev, actual, runtime)
                    _copy(control_activation_dev, control_plane, runtime)
                    _copy(candidate_activation_dev, candidate_plane, runtime)
                    np.testing.assert_array_equal(
                        control_plane,
                        _expected_activation_plane(x_bf16, control_rows),
                    )
                    np.testing.assert_array_equal(
                        candidate_plane,
                        _expected_activation_plane(x_bf16, row_batch),
                    )
                    np.testing.assert_array_equal(actual, expected)
                    _sampled_cpu_gate(
                        actual,
                        x_bf16,
                        qweight,
                        row_batch=row_batch,
                        output_dtype=dtype,
                        out_features=outputs,
                    )
                    actual_f32 = (
                        _bf16_to_f32(actual)
                        if dtype == "bf16"
                        else np.asarray(actual, dtype=np.float32)
                    )
                    assert np.isfinite(actual_f32).all()
                    assert not np.all(actual.view(np.uint8) == 0x5A)

                    runtime.memset(actual_dev.ptr, 0xA5, actual.nbytes)
                    candidate_primitive(
                        candidate_activation_dev.ptr,
                        weight_dev.ptr,
                        actual_dev.ptr,
                        rows,
                        hidden,
                        outputs,
                        library=library,
                        runtime=runtime,
                    )
                    runtime.device_synchronize()
                    _copy(actual_dev, repeated, runtime)
                    np.testing.assert_array_equal(repeated, actual)
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
def test_h8n_bf16_roles_exact_controls_cpu_planes_poison_repeat_and_lifecycle() -> None:
    _run_exact_dtype("bf16")


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h8n_f32_roles_exact_controls_cpu_planes_poison_repeat_and_lifecycle() -> None:
    _run_exact_dtype("f32")
