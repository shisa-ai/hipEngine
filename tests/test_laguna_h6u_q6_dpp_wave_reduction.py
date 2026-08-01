"""RED contracts for WPF-H6U exact Q6 DPP-add wave reduction."""

from __future__ import annotations

import hashlib
import inspect
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
from tests.test_gguf_q5_k_f32_rocblas_prefill import (
    _bf16_bits,
    _device,
    _edge_q6_weight,
)
from tests.test_laguna_h6e_q6_activation_tile_k_row import (
    _Q5_PRODUCTION_POLICY,
    _Q6_PRODUCTION_POLICY,
    _ROLES,
    _activation_plane_shape,
    _expected_activation_plane,
    _hip_available,
    _sampled_cpu_gate,
    _suffix,
)

_H6E_KERNEL = (
    "gguf_q6_k_f32_weight_ordered_weight_major_row_major_"
    "activation_tile_k_row_kernel"
)
_H6U_KERNEL = (
    "gguf_q6_k_f32_weight_ordered_weight_major_row_major_"
    "activation_tile_k_row_dpp_wave_reduction_kernel"
)
_H6U_REDUCE_HELPER = "h6u_reduce_wave_accumulators_dpp"
_H6U_PERMLANE_HELPER = "h6u_permlanex16_f32"
_H6U_MOVE_HELPER = "h6u_dpp_move_f32"
_H6U_ADD_HELPER = "h6u_dpp_add_row_shl1_f32"
_H6E_KERNEL_SHA256 = (
    "d304f6c663b1afe4f3576258db32e6907df9a5f983cfff54795cfe566b07fcd5"
)
_H6E_PRIMITIVE_WRAPPER_SHA256 = (
    "6188e5d4b253013712c1f2974d2291f1cde104d1baa6c2e45cc723221b493162"
)
_H6E_COMPOSITE_WRAPPER_SHA256 = (
    "d799437ebc6f0fc43c21c251d0e342440fce492b1722ced9965ec38701ae8196"
)
_H6U_EXPECTED_ISA = {
    (16, 4, "bf16"): {
        "permlanex16": 64,
        "v_add_f32_dpp": 256,
        "ds_bpermute_b32": 0,
        "v_mov_b32_dpp": 0,
        "code_bytes_max": 7_244,
        "instruction_slots_max": 1_245,
        "metadata_vgpr_max": 130,
        "runtime_vgpr_max": 136,
        "runtime_lds_bytes": 1_024,
    },
    (16, 5, "bf16"): {
        "permlanex16": 80,
        "v_add_f32_dpp": 320,
        "ds_bpermute_b32": 0,
        "v_mov_b32_dpp": 0,
        "code_bytes_max": 8_516,
        "instruction_slots_max": 1_440,
        "metadata_vgpr_max": 162,
        "runtime_vgpr_max": 168,
        "runtime_lds_bytes": 1_536,
    },
    (16, 5, "f32"): {
        "permlanex16": 80,
        "v_add_f32_dpp": 320,
        "ds_bpermute_b32": 0,
        "v_mov_b32_dpp": 0,
        "code_bytes_max": 8_488,
        "instruction_slots_max": 1_437,
        "metadata_vgpr_max": 162,
        "runtime_vgpr_max": 168,
        "runtime_lds_bytes": 1_536,
    },
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _declaration(source: str, anchor: str) -> str:
    start = source.index(anchor)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated declaration: {anchor}")


def _h6e_names(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
) -> tuple[str, str]:
    suffix = _suffix(col_tile, row_batch, output_dtype)
    stem = f"weight_major_row_major_activation_tile_k_row_{suffix}"
    return (
        f"gguf_q5_k_f32_weight_ordered_{stem}",
        f"gguf_q6_k_f32_ordered_{stem}",
    )


def _h6u_names(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
) -> tuple[str, str]:
    suffix = _suffix(col_tile, row_batch, output_dtype)
    stem = (
        "weight_major_row_major_activation_tile_k_row_"
        f"dpp_wave_reduction_{suffix}"
    )
    return (
        f"gguf_q5_k_f32_weight_ordered_{stem}",
        f"gguf_q6_k_f32_ordered_{stem}",
    )


def _h6u_keys(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    *,
    backend: str = "hip_gfx1100",
) -> tuple[KernelKey, KernelKey]:
    suffix = _suffix(col_tile, row_batch, output_dtype)
    stem = (
        "weight_major_row_major_activation_tile_k_row_"
        f"dpp_wave_reduction_{suffix}"
    )
    return (
        KernelKey(backend, "linear", "f32_weight", f"ordered_{stem}"),
        KernelKey(backend, "linear", "gguf_q6_k", f"f32_ordered_{stem}"),
    )


def _candidate(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
):
    _, composite_name = _h6u_names(col_tile, row_batch, output_dtype)
    return getattr(q5_f32, composite_name)


@pytest.fixture(scope="module")
def library():
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(version_file).read_text() if version_file else None
    return q5_f32.build_gguf_q5_k_f32_rocblas_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1",
    )


@pytest.fixture(scope="module")
def production_qweights() -> dict[tuple[int, int], np.ndarray]:
    return {
        (in_features, out_features): _edge_q6_weight(
            out_features,
            in_features,
        )
        for _, _, _, in_features, out_features, _ in _ROLES
    }


def test_h6u_registry_source_policy_and_h6e_immutability() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    assert q5_f32._Q6_ACTIVATION_TILE_K_ROW_ROLES == tuple(
        role[:5] for role in _ROLES
    )
    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    assert hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY == (
        _Q5_PRODUCTION_POLICY
    )
    assert hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_POLICY == (
        _Q6_PRODUCTION_POLICY
    )
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256

    source = Path(q5_f32.__file__).with_suffix(".hip").read_text()
    h6e_kernel = _declaration(source, f"__global__ void {_H6E_KERNEL}(")
    assert _sha256(h6e_kernel) == _H6E_KERNEL_SHA256
    assert h6e_kernel.count("__shfl_down") == 1
    assert h6e_kernel.count("__syncthreads();") == 1
    assert "float acc[ROW_BATCH][COL_TILE] = {};" in h6e_kernel

    for col_tile, row_batch, output_dtype, _, _, _ in _ROLES:
        h6e_primitive_name, h6e_composite_name = _h6e_names(
            col_tile,
            row_batch,
            output_dtype,
        )
        assert _sha256(inspect.getsource(getattr(q5_f32, h6e_primitive_name))) == (
            _H6E_PRIMITIVE_WRAPPER_SHA256
        )
        assert _sha256(inspect.getsource(getattr(q5_f32, h6e_composite_name))) == (
            _H6E_COMPOSITE_WRAPPER_SHA256
        )

    # Intentional RED: production, H6E source bytes, policy, allocation, and
    # gfx1151 exclusion are proven before the separately named H6U wrappers.
    for col_tile, row_batch, output_dtype, _, _, _ in _ROLES:
        primitive_name, composite_name = _h6u_names(
            col_tile,
            row_batch,
            output_dtype,
        )
        primitive = getattr(q5_f32, primitive_name)
        composite = getattr(q5_f32, composite_name)
        assert primitive is not composite
        for key, function in zip(
            _h6u_keys(col_tile, row_batch, output_dtype),
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

    assert q5_f32._Q6_DPP_WAVE_REDUCTION_ROLES == tuple(
        role[:5] for role in _ROLES
    )
    assert set(_H6U_EXPECTED_ISA) == {
        (col_tile, row_batch, output_dtype)
        for col_tile, row_batch, output_dtype, *_ in _ROLES
    }
    assert source.count(f"__global__ void {_H6U_KERNEL}(") == 1
    assert source.count(f"hipengine_{_H6U_KERNEL.removesuffix('_kernel')}") == 3

    helper = _declaration(
        source,
        f"template <int ROW_BATCH, int COL_TILE>\n"
        f"__device__ inline void {_H6U_REDUCE_HELPER}(",
    )
    expected_steps = (
        f"acc[row][col] += {_H6U_PERMLANE_HELPER}(acc[row][col]);",
        f"acc[row][col] += {_H6U_MOVE_HELPER}<0x108>(acc[row][col]);",
        f"acc[row][col] += {_H6U_MOVE_HELPER}<0x104>(acc[row][col]);",
        f"acc[row][col] += {_H6U_MOVE_HELPER}<0x102>(acc[row][col]);",
        f"acc[row][col] = {_H6U_ADD_HELPER}(acc[row][col]);",
    )
    offsets = [helper.index(step) for step in expected_steps]
    assert offsets == sorted(offsets)
    assert "__shfl_down" not in helper
    assert f"{_H6U_MOVE_HELPER}<0x101>" not in helper
    assert "__syncthreads" not in helper

    candidate_kernel = _declaration(source, f"__global__ void {_H6U_KERNEL}(")
    assert candidate_kernel.count(_H6U_REDUCE_HELPER) == 1
    assert "__shfl_down" not in candidate_kernel
    assert candidate_kernel.count("__syncthreads();") == 1
    assert "float acc[ROW_BATCH][COL_TILE] = {};" in candidate_kernel
    assert "constexpr int COL_TILE = 16;" in candidate_kernel
    assert "ROW_BATCH == 4 || ROW_BATCH == 5" in candidate_kernel

    permlane = _declaration(
        source,
        f"__device__ inline float {_H6U_PERMLANE_HELPER}(",
    )
    assert "__builtin_amdgcn_permlanex16" in permlane
    assert "0x76543210U" in permlane and "0xFEDCBA98U" in permlane
    direct_add = _declaration(
        source,
        f"__device__ inline float {_H6U_ADD_HELPER}(",
    )
    assert "v_add_f32_dpp %0, %1, %1 row_shl:1" in direct_add
    assert "row_mask:0xf bank_mask:0xf bound_ctrl:1" in direct_add


def test_h6u_strict_role_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_attempts = 0

    def _unexpected_build(**_kwargs: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H6U role reached HIP loading")

    monkeypatch.setattr(
        q5_f32,
        "build_gguf_q5_k_f32_rocblas_prefill",
        _unexpected_build,
    )
    for col_tile, row_batch, output_dtype, in_features, out_features, _ in _ROLES:
        _, h6e_composite_name = _h6e_names(
            col_tile,
            row_batch,
            output_dtype,
        )
        control = getattr(q5_f32, h6e_composite_name)
        invalid = (
            (0, in_features, out_features, "rows must be positive"),
            (17, in_features - 256, out_features, f"exactly {in_features}"),
            (17, in_features, out_features - 16, f"exactly {out_features}"),
        )
        for rows, hidden, outputs, message in invalid:
            with pytest.raises(ValueError, match=message):
                control(1, 2, 3, 4, 5, rows, hidden, outputs)
    assert load_attempts == 0

    # Intentional RED only after all retained H6E role bounds are proven.
    for col_tile, row_batch, output_dtype, in_features, out_features, _ in _ROLES:
        candidate = _candidate(col_tile, row_batch, output_dtype)
        invalid = (
            (0, in_features, out_features, "rows must be positive"),
            (17, in_features - 256, out_features, f"exactly {in_features}"),
            (17, in_features, out_features - 16, f"exactly {out_features}"),
        )
        for rows, hidden, outputs, message in invalid:
            with pytest.raises(ValueError, match=message):
                candidate(1, 2, 3, 4, 5, rows, hidden, outputs)
    assert load_attempts == 0


@pytest.mark.parametrize("rows", [17, 33, 512])
@pytest.mark.parametrize(
    (
        "col_tile",
        "row_batch",
        "output_dtype",
        "in_features",
        "out_features",
        "pack_col_tile",
    ),
    _ROLES,
    ids=("bf16-k3072-n1024", "bf16-k1024-n3072", "f32-k3072-n1024"),
)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h6u_complete_plane_outputs_and_cpu_values_match_h6e(
    rows: int,
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    in_features: int,
    out_features: int,
    pack_col_tile: int,
    library: Any,
    production_qweights: dict[tuple[int, int], np.ndarray],
) -> None:
    from hipengine.core.hip import get_hip_runtime

    rng = np.random.default_rng(
        20260801 + 43 * rows + 11 * in_features + out_features
    )
    x_bf16 = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    qweight = production_qweights[(in_features, out_features)]
    host_dtype = np.uint16 if output_dtype == "bf16" else np.float32
    expected = np.empty((rows, out_features), dtype=host_dtype)
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
            q5_f32.q6_k_f32_ordered_workspace_nbytes(
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

        _, h6e_composite_name = _h6e_names(
            col_tile,
            row_batch,
            output_dtype,
        )
        control = getattr(q5_f32, h6e_composite_name)
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
            in_features=in_features,
            out_features=out_features,
        )

        # Intentional RED only after complete H6E, activation-plane, and
        # independent sampled CPU bytes pass for this actual role and tail.
        runtime.memset(activation_dev.ptr, 0xA5, plane.nbytes)
        runtime.memset(actual_dev.ptr, 0x5A, actual.nbytes)
        candidate = _candidate(col_tile, row_batch, output_dtype)
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
        assert np.isfinite(actual.view(np.uint16 if output_dtype == "bf16" else np.float32)).all()
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
