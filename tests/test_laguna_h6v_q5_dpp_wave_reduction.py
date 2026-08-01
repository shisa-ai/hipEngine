"""RED contracts for WPF-H6V exact Q5 DPP-add wave reduction."""

from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import (
    copy_device_to_host,
    free,
    host_array_ptr,
    malloc,
    memory_stats,
)
from hipengine.kernels.cpu_reference import gguf_q5_k_gemv
from hipengine.kernels.hip_gfx1100.quant import (
    gguf_q5_k_f32_rocblas_prefill as q5_f32,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.runtime.laguna_gguf_runner import LagunaQ5F32OrderedScratch
from tests.test_gguf_q5_k_f32_rocblas_prefill import (
    _bf16_bits,
    _bf16_to_f32,
    _device,
)
from tests.test_gguf_k_gemv import _make_q5_k_block
from tests.test_laguna_h5y_q5_activation_tile_k_row import (
    _Q5_PRODUCTION_POLICY,
    _activation_plane_shape,
    _candidate_names as _h5y_names,
    _expected_activation_plane,
    _hip_available,
    _suffix,
)
from tests.test_laguna_h6e_q6_activation_tile_k_row import (
    _Q6_PRODUCTION_POLICY,
)

# Geometry, exact production K/N, and reconciled natural-M512 call weight.
_ROLES = (
    (8, 4, "bf16", "tile_k_col", 3_072, 1_024, 92),
    (8, 12, "bf16", "row_major", 3_072, 12_288, 2),
    (16, 5, "bf16", "tile_k_col", 6_144, 3_072, 12),
    (12, 8, "bf16", "row_major", 9_216, 3_072, 35),
    (16, 5, "f32", "tile_k_col", 3_072, 6_144, 12),
    (8, 10, "f32", "tile_k_col", 3_072, 9_216, 35),
)
_H5Y_KERNEL = (
    "gguf_q5_k_f32_weight_ordered_weight_major_"
    "activation_tile_k_row_kernel"
)
_H6V_KERNEL = (
    "gguf_q5_k_f32_weight_ordered_weight_major_"
    "activation_tile_k_row_dpp_wave_reduction_kernel"
)
_H6V_REDUCE_HELPER = "h6v_reduce_wave_accumulators_dpp"
_H6V_PERMLANE_HELPER = "h6v_permlanex16_f32"
_H6V_MOVE_HELPER = "h6v_dpp_move_f32"
_H6V_ADD_HELPER = "h6v_dpp_add_row_shl1_f32"
_H5Y_KERNEL_SHA256 = (
    "5a8ba4a9ec504bef2687aff93c8bd92833bb77ef2baf664b1282ca3fcf256f54"
)
_H5Y_PRIMITIVE_WRAPPER_SHA256 = (
    "8e5a2ca92c84c4414e2a7ad70dcf47653d0da3b1c1570bbe31e5a22810af327a"
)
_H5Y_COMPOSITE_WRAPPER_SHA256 = (
    "c9cb4b2ccc8f5b3cc2831eb73a25e50e2d42d16e48bf9bb52074722a96a12339"
)
_H5Y_REDUCTION = """#pragma unroll
  for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
#pragma unroll
    for (int col = 0; col < COL_TILE; ++col) {
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[row_index][col] += __shfl_down(acc[row_index][col], offset);
      }
    }
  }
"""
_ONE_SHOT_PROTOCOL = {
    "warmups": 5,
    "counter_rotated_repetitions": 15,
    "launches_per_sample": 5,
    "all_roles_must_win_event_and_wall": True,
    "weighted_schedule_calls": 188,
    "remove_all_surfaces_on_any_miss": True,
}
_H6V_EXPECTED_PHYSICAL = {
    (8, 4, "bf16", "tile_k_col", 3_072, 1_024): {
        "source_bpermute": 160,
        "permlanex16": 32,
        "v_add_f32_dpp": 128,
        "code_bytes_max": 3_660,
        "instruction_slots_max": 635,
        "runtime_vgpr_max": 72,
        "runtime_lds_bytes": 512,
        "calls": 92,
        "request_workgroups": 1_507_328,
        "dynamic_source_bpermute": 964_689_920,
    },
    (8, 12, "bf16", "row_major", 3_072, 12_288): {
        "source_bpermute": 480,
        "permlanex16": 96,
        "v_add_f32_dpp": 384,
        "code_bytes_max": 9_920,
        "instruction_slots_max": 1_718,
        "runtime_vgpr_max": 200,
        "runtime_lds_bytes": 1_536,
        "calls": 2,
        "request_workgroups": 132_096,
        "dynamic_source_bpermute": 253_624_320,
    },
    (16, 5, "bf16", "tile_k_col", 6_144, 3_072): {
        "source_bpermute": 400,
        "permlanex16": 80,
        "v_add_f32_dpp": 320,
        "code_bytes_max": 7_796,
        "instruction_slots_max": 1_316,
        "runtime_vgpr_max": 168,
        "runtime_lds_bytes": 1_536,
        "calls": 12,
        "request_workgroups": 237_312,
        "dynamic_source_bpermute": 379_699_200,
    },
    (12, 8, "bf16", "row_major", 9_216, 3_072): {
        "source_bpermute": 480,
        "permlanex16": 96,
        "v_add_f32_dpp": 384,
        "code_bytes_max": 9_796,
        "instruction_slots_max": 1_674,
        "runtime_vgpr_max": 200,
        "runtime_lds_bytes": 1_536,
        "calls": 35,
        "request_workgroups": 573_440,
        "dynamic_source_bpermute": 1_101_004_800,
    },
    (16, 5, "f32", "tile_k_col", 3_072, 6_144): {
        "source_bpermute": 400,
        "permlanex16": 80,
        "v_add_f32_dpp": 320,
        "code_bytes_max": 7_768,
        "instruction_slots_max": 1_313,
        "runtime_vgpr_max": 168,
        "runtime_lds_bytes": 1_536,
        "calls": 12,
        "request_workgroups": 474_624,
        "dynamic_source_bpermute": 759_398_400,
    },
    (8, 10, "f32", "tile_k_col", 3_072, 9_216): {
        "source_bpermute": 400,
        "permlanex16": 80,
        "v_add_f32_dpp": 320,
        "code_bytes_max": 7_960,
        "instruction_slots_max": 1_369,
        "runtime_vgpr_max": 168,
        "runtime_lds_bytes": 1_536,
        "calls": 35,
        "request_workgroups": 2_096_640,
        "dynamic_source_bpermute": 3_354_624_000,
    },
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _production_q5_weight(out_features: int, in_features: int) -> np.ndarray:
    """Build deterministic valid Q5 blocks without large-row uint8 overflow."""

    blocks_per_row = in_features // 256
    row_nbytes = blocks_per_row * 176
    basis_rows = min(out_features, 128)
    basis = np.empty((basis_rows, row_nbytes), dtype=np.uint8)
    for out_index in range(basis_rows):
        for block_index in range(blocks_per_row):
            start = block_index * 176
            basis[out_index, start : start + 176] = _make_q5_k_block(
                out_index,
                block_index,
            )
    return np.ascontiguousarray(basis[np.arange(out_features) % basis_rows])


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


def _h6v_names(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    weight_layout: str,
) -> tuple[str, str]:
    suffix = _suffix(col_tile, row_batch, output_dtype)
    stem = (
        f"weight_major_{weight_layout}_activation_tile_k_row_"
        f"dpp_wave_reduction_{suffix}"
    )
    return (
        f"gguf_q5_k_f32_weight_ordered_{stem}",
        f"gguf_q5_k_f32_ordered_{stem}",
    )


def _h6v_keys(
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
        f"dpp_wave_reduction_{suffix}"
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
    _, composite_name = _h6v_names(
        col_tile,
        row_batch,
        output_dtype,
        weight_layout,
    )
    return getattr(q5_f32, composite_name)


def _sampled_cpu_gate(
    actual: np.ndarray,
    x_bf16: np.ndarray,
    qweight: np.ndarray,
    *,
    row_batch: int,
    output_dtype: str,
    out_features: int,
) -> None:
    sample_rows = np.unique(
        np.asarray(
            [
                0,
                min(row_batch - 1, actual.shape[0] - 1),
                min(row_batch, actual.shape[0] - 1),
                actual.shape[0] - 1,
            ]
        )
    )
    sample_cols = np.unique(
        np.concatenate(
            (
                np.arange(0, 16),
                np.arange(out_features // 2, out_features // 2 + 8),
                np.arange(out_features - 16, out_features),
            )
        )
    )
    cpu = gguf_q5_k_gemv(
        _bf16_to_f32(x_bf16[sample_rows]),
        np.ascontiguousarray(qweight[sample_cols]),
    )
    sampled = actual[np.ix_(sample_rows, sample_cols)]
    sampled_f32 = (
        _bf16_to_f32(sampled)
        if output_dtype == "bf16"
        else np.asarray(sampled, dtype=np.float32)
    )
    relative = np.abs(sampled_f32 - cpu) / np.maximum(np.abs(cpu), 1.0)
    assert float(np.max(relative)) <= 0.05
    assert evaluate_logits(cpu, sampled_f32).passed


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
        (in_features, out_features): _production_q5_weight(
            out_features,
            in_features,
        )
        for _, _, _, _, in_features, out_features, _ in _ROLES
    }


def test_h6v_registry_source_policy_and_h5y_immutability() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    assert q5_f32._Q5_ACTIVATION_TILE_K_ROW_GEOMETRIES == tuple(
        role[:4] for role in _ROLES
    )
    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    assert hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY == (
        _Q5_PRODUCTION_POLICY
    )
    assert hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_POLICY == (
        _Q6_PRODUCTION_POLICY
    )
    assert all(
        "dpp_wave_reduction" not in variant
        for variant in _Q5_PRODUCTION_POLICY.values()
    )
    assert LagunaQ5F32OrderedScratch.weight_f32_planned_nbytes() == 150_994_944
    assert (
        LagunaQ5F32OrderedScratch.activation_bf16_planned_nbytes(max_rows=512)
        == 10_125_312
    )
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256

    source = Path(q5_f32.__file__).with_suffix(".hip").read_text()
    h5y_kernel = _declaration(source, f"__global__ void {_H5Y_KERNEL}(")
    assert _sha256(h5y_kernel) == _H5Y_KERNEL_SHA256
    assert h5y_kernel.count("__shfl_down") == 1
    assert h5y_kernel.count("__syncthreads();") == 1
    assert "float acc[ROW_BATCH][COL_TILE] = {};" in h5y_kernel

    for col_tile, row_batch, output_dtype, weight_layout, *_ in _ROLES:
        _, h5y_primitive_name, h5y_composite_name = _h5y_names(
            col_tile,
            row_batch,
            output_dtype,
            weight_layout,
        )
        assert _sha256(
            inspect.getsource(getattr(q5_f32, h5y_primitive_name))
        ) == _H5Y_PRIMITIVE_WRAPPER_SHA256
        assert _sha256(
            inspect.getsource(getattr(q5_f32, h5y_composite_name))
        ) == _H5Y_COMPOSITE_WRAPPER_SHA256

    # Intentional RED: live H5Y policy/source/allocation and gfx1151 exclusion
    # are frozen before any separately named H6V implementation is added.
    for col_tile, row_batch, output_dtype, weight_layout, *_ in _ROLES:
        primitive_name, composite_name = _h6v_names(
            col_tile,
            row_batch,
            output_dtype,
            weight_layout,
        )
        primitive = getattr(q5_f32, primitive_name)
        composite = getattr(q5_f32, composite_name)
        assert primitive is not composite
        for key, function in zip(
            _h6v_keys(col_tile, row_batch, output_dtype, weight_layout),
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

    assert q5_f32._Q5_DPP_WAVE_REDUCTION_ROLES == tuple(
        role[:6] for role in _ROLES
    )
    assert set(_H6V_EXPECTED_PHYSICAL) == {
        role[:6] for role in _ROLES
    }
    assert source.count(f"__global__ void {_H6V_KERNEL}(") == 1
    for col_tile, row_batch, output_dtype, weight_layout, *_ in _ROLES:
        suffix = _suffix(col_tile, row_batch, output_dtype)
        symbol = (
            "hipengine_gguf_q5_k_f32_weight_ordered_weight_major_"
            f"{weight_layout}_activation_tile_k_row_dpp_wave_reduction_"
            f"{suffix}"
        )
        assert source.count(symbol) == 1

    helper = _declaration(
        source,
        f"template <int ROW_BATCH, int COL_TILE>\n"
        f"__device__ inline void {_H6V_REDUCE_HELPER}(",
    )
    expected_steps = (
        f"acc[row][col] += {_H6V_PERMLANE_HELPER}(acc[row][col]);",
        f"acc[row][col] += {_H6V_MOVE_HELPER}<0x108>(acc[row][col]);",
        f"acc[row][col] += {_H6V_MOVE_HELPER}<0x104>(acc[row][col]);",
        f"acc[row][col] += {_H6V_MOVE_HELPER}<0x102>(acc[row][col]);",
        f"acc[row][col] = {_H6V_ADD_HELPER}(acc[row][col]);",
    )
    offsets = [helper.index(step) for step in expected_steps]
    assert offsets == sorted(offsets)
    assert "__shfl_down" not in helper
    assert f"{_H6V_MOVE_HELPER}<0x101>" not in helper
    assert "__syncthreads" not in helper

    candidate_kernel = _declaration(source, f"__global__ void {_H6V_KERNEL}(")
    assert candidate_kernel.count(_H6V_REDUCE_HELPER) == 1
    assert "__shfl_down" not in candidate_kernel
    assert candidate_kernel.count("__syncthreads();") == 1
    assert "float acc[ROW_BATCH][COL_TILE] = {};" in candidate_kernel
    assert "bool TILE_K_COL" in source[
        source.rfind("template <", 0, source.index(candidate_kernel)) :
        source.index(candidate_kernel)
    ]

    normalized_h5y = h5y_kernel.replace(_H5Y_REDUCTION, "  H6V_REDUCTION;\n")
    normalized_h6v = candidate_kernel.replace(
        f"  {_H6V_REDUCE_HELPER}<ROW_BATCH, COL_TILE>(acc);\n",
        "  H6V_REDUCTION;\n",
    ).replace(_H6V_KERNEL, _H5Y_KERNEL)
    assert normalized_h6v == normalized_h5y

    permlane = _declaration(
        source,
        f"__device__ inline float {_H6V_PERMLANE_HELPER}(",
    )
    assert "__builtin_amdgcn_permlanex16" in permlane
    assert "0x76543210U" in permlane and "0xFEDCBA98U" in permlane
    direct_add = _declaration(
        source,
        f"__device__ inline float {_H6V_ADD_HELPER}(",
    )
    assert "v_add_f32_dpp %0, %1, %1 row_shl:1" in direct_add
    assert "row_mask:0xf bank_mask:0xf bound_ctrl:1" in direct_add


def test_h6v_frozen_physical_and_one_shot_admission_contract() -> None:
    assert sum(role[-1] for role in _ROLES) == 188
    assert sum(
        facts["request_workgroups"] for facts in _H6V_EXPECTED_PHYSICAL.values()
    ) == 5_021_440
    assert sum(
        facts["dynamic_source_bpermute"]
        for facts in _H6V_EXPECTED_PHYSICAL.values()
    ) == 6_813_040_640
    for role, facts in _H6V_EXPECTED_PHYSICAL.items():
        col_tile, row_batch, *_ = role
        accumulators = col_tile * row_batch
        assert facts["source_bpermute"] == 5 * accumulators
        assert facts["permlanex16"] == accumulators
        assert facts["v_add_f32_dpp"] == 4 * accumulators
        assert facts["runtime_vgpr_max"] in {72, 168, 200}
        assert facts["runtime_lds_bytes"] in {512, 1_536}
        assert facts["dynamic_source_bpermute"] == (
            4 * facts["request_workgroups"] * facts["source_bpermute"]
        )
    # The executable screen is deliberately one-shot. Every role and the
    # weighted aggregate must win both HIP-event and synchronized-wall clocks;
    # any correctness, physical, lifecycle, role-clock, or weighted-clock miss
    # removes every H6V surface without follow-up tuning.
    assert _ONE_SHOT_PROTOCOL == {
        "warmups": 5,
        "counter_rotated_repetitions": 15,
        "launches_per_sample": 5,
        "all_roles_must_win_event_and_wall": True,
        "weighted_schedule_calls": 188,
        "remove_all_surfaces_on_any_miss": True,
    }


def test_h6v_strict_role_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_attempts = 0

    def _unexpected_build(**_kwargs: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H6V role reached HIP loading")

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
        _, h5y_primitive_name, h5y_composite_name = _h5y_names(
            col_tile,
            row_batch,
            output_dtype,
            weight_layout,
        )
        for function, pointers in (
            (getattr(q5_f32, h5y_primitive_name), (1, 2, 3)),
            (getattr(q5_f32, h5y_composite_name), (1, 2, 3, 4, 5)),
        ):
            with pytest.raises(ValueError, match="rows must be positive"):
                function(*pointers, 0, in_features, out_features)
    assert load_attempts == 0

    # Intentional RED only after retained H5Y bounds are proven.
    for (
        col_tile,
        row_batch,
        output_dtype,
        weight_layout,
        in_features,
        out_features,
        _,
    ) in _ROLES:
        primitive_name, composite_name = _h6v_names(
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
                (17, in_features, out_features - 8, f"exactly {out_features}"),
            )
            for rows, hidden, outputs, message in invalid:
                with pytest.raises(ValueError, match=message):
                    function(*pointers, rows, hidden, outputs)
    assert load_attempts == 0


@pytest.mark.parametrize("rows", [17, 33, 512])
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
    ids=(
        "bf16-k3072-n1024",
        "bf16-k3072-n12288",
        "bf16-k6144-n3072",
        "bf16-k9216-n3072",
        "f32-k3072-n6144",
        "f32-k3072-n9216",
    ),
)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h6v_complete_plane_outputs_and_cpu_values_match_h5y(
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
        20260801 + 47 * rows + 13 * in_features + out_features
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

        # Intentional RED only after complete H5Y, activation-plane, and
        # independent sampled CPU/logit gates pass for this role and tail.
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
        actual_f32 = (
            _bf16_to_f32(actual)
            if output_dtype == "bf16"
            else np.asarray(actual, dtype=np.float32)
        )
        assert np.isfinite(actual_f32).all()
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
