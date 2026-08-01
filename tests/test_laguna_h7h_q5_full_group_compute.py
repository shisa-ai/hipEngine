"""RED contracts for WPF-H7H exact full-group Q5 compute."""

from __future__ import annotations

import inspect
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hipengine.core.memory import (
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
    _Q5_PRODUCTION_POLICY,
    _activation_plane_shape,
    _candidate_names as _h5y_names,
    _expected_activation_plane,
    _hip_available,
    _suffix,
)
from tests.test_laguna_h7g_q5_padded_compute import (
    _H5Y_GUARDED_COMPUTE,
    _H5Y_KERNEL,
    _H5Y_KERNEL_SHA256,
    _H7G_KERNEL,
    _H7G_UNGUARDED_COMPUTE,
    _ROLES as _H7G_ROLES,
    _bf16_bits,
    _bf16_to_f32,
    _declaration,
    _device,
    _production_q5_weight,
    _sampled_cpu_gate,
    _sha256,
)

# Both and only the H5Y roles whose natural-M512 row groups are exactly full.
# No post-timing role subset is admissible.
_ROLES = (
    (8, 4, "bf16", "tile_k_col", 3_072, 1_024, 92),
    (12, 8, "bf16", "row_major", 9_216, 3_072, 35),
)
_ROWS = (1, 7, 8, 9, 512)
_H7G_KERNEL_SHA256 = (
    "0e091497b0986f760aae723ce9b7f0e76b3ed5f3b239d72d4eb058dcb974790c"
)
_SELECTION = {
    "revision": "16fee41a9+out-of-tree-h7h-selection-probe",
    "rows": 512,
    "warmups": 5,
    "counter_rotated_repetitions": 15,
    "launches_per_sample": 5,
    "weighted_schedule_calls": 127,
    "harness_sha256": (
        "f39f5777ecb910d46e195cb57cbc250821dbd8a46e5b91470996c246168cd0d4"
    ),
    "candidate_source_sha256": (
        "9c067a9bcd55ce273ee4d69b689a5640b0971e5d0d4d05952a6ad0eb689112b5"
    ),
    "candidate_library_sha256": (
        "e03e465caf8e87d46eb6e33450351936f2e2dd8abec295e160e306edc8e84e06"
    ),
    "raw_json_sha256": (
        "922929713eef50c2919da8672bd063f44ad1f64930cb88d5832eecf6764b2ef6"
    ),
    "h5y_event_weighted_ms": 140.32659397125244,
    "h7h_event_weighted_ms": 118.79937973022462,
    "event_speedup": 1.1812064531810929,
    "h5y_wall_weighted_ms": 140.00293798744678,
    "h7h_wall_weighted_ms": 121.17274859920144,
    "wall_speedup": 1.1553995399619865,
    "all_roles_both_clock_positive": True,
    "all_candidate_bytes_exact": True,
    "first_and_only_timing_run": True,
    "subset_salvage_allowed": False,
}
_EXPECTED_PHYSICAL = {
    (8, 4, "bf16", "tile_k_col", 3_072, 1_024): {
        "control_dual_fmac": 1,
        "control_scalar_fmac": 31,
        "candidate_dual_fmac": 15,
        "candidate_scalar_fmac": 5,
        "control_instruction_slots": 635,
        "candidate_instruction_slots_max": 587,
        "control_code_bytes": 3_840,
        "candidate_code_bytes_max": 3_584,
        "control_sgpr": 23,
        "candidate_sgpr_max": 21,
        "metadata_vgpr": 72,
        "lds_bytes": 512,
    },
    (12, 8, "bf16", "row_major", 9_216, 3_072): {
        "control_dual_fmac": 1,
        "control_scalar_fmac": 95,
        "candidate_dual_fmac": 47,
        "candidate_scalar_fmac": 9,
        "control_instruction_slots": 1_674,
        "candidate_instruction_slots_max": 1_588,
        "control_code_bytes": 9_984,
        "candidate_code_bytes_max": 9_728,
        "control_sgpr": 47,
        "candidate_sgpr_max": 43,
        "metadata_vgpr": 194,
        "lds_bytes": 1_536,
    },
}


def _h7h_names(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    weight_layout: str,
) -> tuple[str, str]:
    suffix = _suffix(col_tile, row_batch, output_dtype)
    stem = (
        f"weight_major_{weight_layout}_activation_tile_k_row_"
        f"full_group_compute_{suffix}"
    )
    return (
        f"gguf_q5_k_f32_weight_ordered_{stem}",
        f"gguf_q5_k_f32_ordered_{stem}",
    )


def _h7h_keys(
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
        f"full_group_compute_{suffix}"
    )
    return (
        KernelKey(backend, "linear", "f32_weight", f"ordered_{stem}"),
        KernelKey(backend, "linear", "gguf_q5_k", f"f32_ordered_{stem}"),
    )


def _candidate(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    weight_layout: str,
):
    _, composite_name = _h7h_names(
        col_tile,
        row_batch,
        output_dtype,
        weight_layout,
    )
    return getattr(q5_f32, composite_name)


@pytest.fixture(scope="module")
def library():
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
        require_cached=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1",
    )


@pytest.fixture(scope="module")
def production_qweights() -> dict[tuple[int, int], np.ndarray]:
    return {
        (in_features, out_features): _production_q5_weight(
            out_features,
            in_features,
        )
        for _, _, _, _, in_features, out_features, _ in _ROLES
    }


def test_h7h_frozen_selection_and_physical_contract() -> None:
    assert sum(role[-1] for role in _ROLES) == 127
    assert all(512 % role[1] == 0 for role in _ROLES)
    assert _SELECTION["rows"] == 512
    assert _SELECTION["warmups"] == 5
    assert _SELECTION["counter_rotated_repetitions"] == 15
    assert _SELECTION["launches_per_sample"] == 5
    assert _SELECTION["weighted_schedule_calls"] == 127
    assert _SELECTION["all_roles_both_clock_positive"]
    assert _SELECTION["all_candidate_bytes_exact"]
    assert _SELECTION["first_and_only_timing_run"]
    assert not _SELECTION["subset_salvage_allowed"]
    assert math.isclose(
        _SELECTION["h5y_event_weighted_ms"]
        / _SELECTION["h7h_event_weighted_ms"],
        _SELECTION["event_speedup"],
    )
    assert math.isclose(
        _SELECTION["h5y_wall_weighted_ms"]
        / _SELECTION["h7h_wall_weighted_ms"],
        _SELECTION["wall_speedup"],
    )
    assert _SELECTION["event_speedup"] > 1.0
    assert _SELECTION["wall_speedup"] > 1.0
    assert set(_EXPECTED_PHYSICAL) == {role[:6] for role in _ROLES}
    for facts in _EXPECTED_PHYSICAL.values():
        assert facts["candidate_dual_fmac"] > facts["control_dual_fmac"]
        assert facts["candidate_scalar_fmac"] < facts["control_scalar_fmac"]
        assert (
            facts["candidate_instruction_slots_max"]
            < facts["control_instruction_slots"]
        )
        assert facts["candidate_code_bytes_max"] < facts["control_code_bytes"]
        assert facts["candidate_sgpr_max"] <= facts["control_sgpr"]
        assert facts["metadata_vgpr"] <= 194
        assert facts["lds_bytes"] <= 1_536


def test_h7h_source_registry_and_h5y_h7g_immutability() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    assert hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_H5Y_POLICY == (
        _Q5_PRODUCTION_POLICY
    )
    assert hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY == (
        hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_H7G_POLICY
    )
    assert {
        role
        for role, variant in hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY.items()
        if variant != _Q5_PRODUCTION_POLICY[role]
    } == {(role[2], role[4], role[5]) for role in _H7G_ROLES}
    assert all(
        hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY[
            (output_dtype, in_features, out_features)
        ]
        == _Q5_PRODUCTION_POLICY[(output_dtype, in_features, out_features)]
        for _, _, output_dtype, _, in_features, out_features, _ in _ROLES
    )
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256

    source = Path(q5_f32.__file__).with_suffix(".hip").read_text()
    h5y_kernel = _declaration(source, f"__global__ void {_H5Y_KERNEL}(")
    h7g_kernel = _declaration(source, f"__global__ void {_H7G_KERNEL}(")
    assert _sha256(h5y_kernel) == _H5Y_KERNEL_SHA256
    assert _sha256(h7g_kernel) == _H7G_KERNEL_SHA256
    assert h5y_kernel.count(_H5Y_GUARDED_COMPUTE) == 1
    assert h7g_kernel.count(_H7G_UNGUARDED_COMPUTE) == 1
    assert q5_f32._Q5_PADDED_COMPUTE_ROLES == tuple(
        role[:6] for role in _H7G_ROLES
    )

    for col_tile, row_batch, output_dtype, weight_layout, *_ in _ROLES:
        _, h5y_primitive_name, h5y_composite_name = _h5y_names(
            col_tile,
            row_batch,
            output_dtype,
            weight_layout,
        )
        assert inspect.isfunction(getattr(q5_f32, h5y_primitive_name))
        assert inspect.isfunction(getattr(q5_f32, h5y_composite_name))

        # Intentional RED only after source, fallback, and workspace controls.
        primitive_name, composite_name = _h7h_names(
            col_tile,
            row_batch,
            output_dtype,
            weight_layout,
        )
        primitive = getattr(q5_f32, primitive_name)
        composite = getattr(q5_f32, composite_name)
        assert primitive is not composite
        for key, function in zip(
            _h7h_keys(col_tile, row_batch, output_dtype, weight_layout),
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

        suffix = _suffix(col_tile, row_batch, output_dtype)
        symbol = (
            "hipengine_gguf_q5_k_f32_weight_ordered_weight_major_"
            f"{weight_layout}_activation_tile_k_row_full_group_compute_{suffix}"
        )
        assert source.count(symbol) == 1

    assert q5_f32._Q5_FULL_GROUP_COMPUTE_ROLES == tuple(
        role[:6] for role in _ROLES
    )


def test_h7h_strict_role_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_attempts = 0

    def _unexpected_build(**_kwargs: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H7H role reached HIP loading")

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
        _,
    ) in _ROLES:
        primitive_name, composite_name = _h7h_names(
            col_tile,
            row_batch,
            output_dtype,
            weight_layout,
        )
        for function, pointers in (
            (getattr(q5_f32, primitive_name), (1, 2, 3)),
            (getattr(q5_f32, composite_name), (1, 2, 3, 4, 5)),
        ):
            invalid = (
                (0, in_features, out_features, "rows must be positive"),
                (17, in_features - 256, out_features, f"exactly {in_features}"),
                (
                    17,
                    in_features,
                    out_features - col_tile,
                    f"exactly {out_features}",
                ),
            )
            for rows, hidden, outputs, message in invalid:
                with pytest.raises(ValueError, match=message):
                    function(*pointers, rows, hidden, outputs)
    assert load_attempts == 0


@pytest.mark.parametrize("rows", _ROWS)
@pytest.mark.parametrize(
    (
        "col_tile",
        "row_batch",
        "output_dtype",
        "weight_layout",
        "in_features",
        "out_features",
        "call_weight",
    ),
    _ROLES,
    ids=("bf16-k3072-n1024-r4", "bf16-k9216-n3072-r8"),
)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h7h_complete_plane_outputs_and_cpu_values_match_h5y(
    rows: int,
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    weight_layout: str,
    in_features: int,
    out_features: int,
    call_weight: int,
    library: Any,
    production_qweights: dict[tuple[int, int], np.ndarray],
) -> None:
    from hipengine.core.hip import get_hip_runtime

    assert call_weight > 0
    rng = np.random.default_rng(
        20260803 + 53 * rows + 17 * in_features + out_features
    )
    x_bf16 = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    qweight = production_qweights[(in_features, out_features)]
    expected = np.empty((rows, out_features), dtype=np.uint16)
    actual = np.empty_like(expected)
    plane_shape = _activation_plane_shape(rows, in_features, row_batch)
    plane = np.empty(plane_shape, dtype=np.uint16)
    expected_plane = _expected_activation_plane(x_bf16, row_batch)

    runtime = get_hip_runtime()
    before = memory_stats()
    buffers: list[Any] = []
    try:
        x_dev = _device(x_bf16, runtime)
        weight_dev = _device(qweight, runtime)
        weight_f32_dev = malloc(
            q5_f32.q5_k_f32_ordered_workspace_nbytes(
                in_features,
                out_features,
            ),
            runtime=runtime,
        )
        expected_dev = malloc(expected.nbytes, runtime=runtime)
        actual_dev = malloc(actual.nbytes, runtime=runtime)
        activation_dev = malloc(plane.nbytes, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                weight_dev,
                weight_f32_dev,
                expected_dev,
                actual_dev,
                activation_dev,
            )
        )

        _, _, h5y_composite_name = _h5y_names(
            col_tile,
            row_batch,
            output_dtype,
            weight_layout,
        )
        control = getattr(q5_f32, h5y_composite_name)
        control(
            x_dev.ptr,
            weight_dev.ptr,
            expected_dev.ptr,
            weight_f32_dev.ptr,
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
        _sampled_cpu_gate(
            expected,
            x_bf16,
            qweight,
            row_batch=row_batch,
            output_dtype=output_dtype,
            out_features=out_features,
        )

        # Intentional RED only after complete H5Y bytes and CPU/plane gates.
        runtime.memset(activation_dev.ptr, 0xA5, plane.nbytes)
        runtime.memset(actual_dev.ptr, 0x5A, actual.nbytes)
        candidate = _candidate(
            col_tile,
            row_batch,
            output_dtype,
            weight_layout,
        )
        candidate(
            x_dev.ptr,
            weight_dev.ptr,
            actual_dev.ptr,
            weight_f32_dev.ptr,
            activation_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(
            host_array_ptr(actual),
            actual_dev,
            actual.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(plane),
            activation_dev,
            plane.nbytes,
            runtime=runtime,
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(plane, expected_plane)
        assert np.isfinite(_bf16_to_f32(actual)).all()
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
