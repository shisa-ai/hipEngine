"""RED contracts for WPF-H7G exact Q5 padded-row compute."""

from __future__ import annotations

import hashlib
import inspect
import math
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
# Only row batches with a real M512 padded tail were selected before timing.
_ROLES = (
    (8, 12, "bf16", "row_major", 3_072, 12_288, 2),
    (16, 5, "bf16", "tile_k_col", 6_144, 3_072, 12),
    (16, 5, "f32", "tile_k_col", 3_072, 6_144, 12),
    (8, 10, "f32", "tile_k_col", 3_072, 9_216, 35),
)
_EXCLUDED_EXACT_DIVISIBILITY_ROW_BATCHES = (4, 8)
_H5Y_KERNEL = (
    "gguf_q5_k_f32_weight_ordered_weight_major_"
    "activation_tile_k_row_kernel"
)
_H7G_KERNEL = (
    "gguf_q5_k_f32_weight_ordered_weight_major_"
    "activation_tile_k_row_padded_compute_kernel"
)
_H5Y_KERNEL_SHA256 = (
    "5a8ba4a9ec504bef2687aff93c8bd92833bb77ef2baf664b1282ca3fcf256f54"
)
_H5Y_PRIMITIVE_WRAPPER_SHA256 = (
    "8e5a2ca92c84c4414e2a7ad70dcf47653d0da3b1c1570bbe31e5a22810af327a"
)
_H5Y_COMPOSITE_WRAPPER_SHA256 = (
    "c9cb4b2ccc8f5b3cc2831eb73a25e50e2d42d16e48bf9bb52074722a96a12339"
)
_H5Y_GUARDED_COMPUTE = """#pragma unroll
    for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
      const int row = row_base + row_index;
      if (row < rows) {
        const uint16_t input_bits = static_cast<uint16_t>(
            activation_words[row_index >> 1] >> ((row_index & 1) * 16));
        const float input_value = bf16_bits_to_float(input_bits);
#pragma unroll
        for (int col = 0; col < COL_TILE; ++col) {
          acc[row_index][col] =
              fmaf(input_value, weights[col], acc[row_index][col]);
        }
      }
    }
"""
_H7G_UNGUARDED_COMPUTE = """#pragma unroll
    for (int row_index = 0; row_index < ROW_BATCH; ++row_index) {
      const uint16_t input_bits = static_cast<uint16_t>(
          activation_words[row_index >> 1] >> ((row_index & 1) * 16));
      const float input_value = bf16_bits_to_float(input_bits);
#pragma unroll
      for (int col = 0; col < COL_TILE; ++col) {
        acc[row_index][col] =
            fmaf(input_value, weights[col], acc[row_index][col]);
      }
    }
"""
_SELECTION = {
    "revision": "1f6ca6a8c+out-of-tree-h7g-selection-probe",
    "rows": 512,
    "warmups": 5,
    "counter_rotated_repetitions": 15,
    "launches_per_sample": 5,
    "weighted_schedule_calls": 61,
    "harness_sha256": (
        "7fe987e3de20dc834119b4db616a41130e86bd90bb546d4b4f0100b80a54e3dd"
    ),
    "candidate_source_sha256": (
        "9c067a9bcd55ce273ee4d69b689a5640b0971e5d0d4d05952a6ad0eb689112b5"
    ),
    "candidate_library_sha256": (
        "e03e465caf8e87d46eb6e33450351936f2e2dd8abec295e160e306edc8e84e06"
    ),
    "raw_json_sha256": (
        "cd7306331a8dfd50ce67f0dc07852ff81fdeeaad7106f8ac2da94fcdc40c582f"
    ),
    "h5y_event_weighted_ms": 136.91840705871581,
    "h7g_event_weighted_ms": 128.59810218811035,
    "event_speedup": 1.0647000595579141,
    "h5y_wall_weighted_ms": 137.0091316755861,
    "h7g_wall_weighted_ms": 129.4963836669922,
    "wall_speedup": 1.058015118228424,
    "all_roles_both_clock_positive": True,
    "all_candidate_bytes_exact": True,
}
_H7G_EXPECTED_PHYSICAL = {
    (8, 12, "bf16", "row_major", 3_072, 12_288): {
        "control_dual_fmac": 1,
        "control_scalar_fmac": 95,
        "candidate_dual_fmac": 91,
        "candidate_scalar_fmac": 5,
        "candidate_instruction_slots_max": 1_577,
        "candidate_code_bytes_max": 9_728,
        "metadata_vgpr_max": 194,
    },
    (16, 5, "bf16", "tile_k_col", 6_144, 3_072): {
        "control_dual_fmac": 1,
        "control_scalar_fmac": 79,
        "candidate_dual_fmac": 66,
        "candidate_scalar_fmac": 14,
        "candidate_instruction_slots_max": 1_234,
        "candidate_code_bytes_max": 7_680,
        "metadata_vgpr_max": 162,
    },
    (16, 5, "f32", "tile_k_col", 3_072, 6_144): {
        "control_dual_fmac": 1,
        "control_scalar_fmac": 79,
        "candidate_dual_fmac": 66,
        "candidate_scalar_fmac": 14,
        "candidate_instruction_slots_max": 1_231,
        "candidate_code_bytes_max": 7_680,
        "metadata_vgpr_max": 162,
    },
    (8, 10, "f32", "tile_k_col", 3_072, 9_216): {
        "control_dual_fmac": 1,
        "control_scalar_fmac": 79,
        "candidate_dual_fmac": 73,
        "candidate_scalar_fmac": 7,
        "candidate_instruction_slots_max": 1_248,
        "candidate_code_bytes_max": 7_680,
        "metadata_vgpr_max": 162,
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


def _h7g_names(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    weight_layout: str,
) -> tuple[str, str]:
    suffix = _suffix(col_tile, row_batch, output_dtype)
    stem = (
        f"weight_major_{weight_layout}_activation_tile_k_row_"
        f"padded_compute_{suffix}"
    )
    return (
        f"gguf_q5_k_f32_weight_ordered_{stem}",
        f"gguf_q5_k_f32_ordered_{stem}",
    )


def _h7g_keys(
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
        f"padded_compute_{suffix}"
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
    _, composite_name = _h7g_names(
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


def test_h7g_frozen_selection_contract() -> None:
    assert sum(role[-1] for role in _ROLES) == 61
    assert all(512 % role[1] for role in _ROLES)
    assert all(512 % row_batch == 0 for row_batch in _EXCLUDED_EXACT_DIVISIBILITY_ROW_BATCHES)
    assert _SELECTION["rows"] == 512
    assert _SELECTION["warmups"] == 5
    assert _SELECTION["counter_rotated_repetitions"] == 15
    assert _SELECTION["launches_per_sample"] == 5
    assert _SELECTION["weighted_schedule_calls"] == 61
    assert _SELECTION["all_roles_both_clock_positive"]
    assert _SELECTION["all_candidate_bytes_exact"]
    assert math.isclose(
        _SELECTION["h5y_event_weighted_ms"]
        / _SELECTION["h7g_event_weighted_ms"],
        _SELECTION["event_speedup"],
    )
    assert math.isclose(
        _SELECTION["h5y_wall_weighted_ms"]
        / _SELECTION["h7g_wall_weighted_ms"],
        _SELECTION["wall_speedup"],
    )
    assert _SELECTION["event_speedup"] > 1.0
    assert _SELECTION["wall_speedup"] > 1.0
    assert set(_H7G_EXPECTED_PHYSICAL) == {role[:6] for role in _ROLES}
    for facts in _H7G_EXPECTED_PHYSICAL.values():
        assert facts["candidate_dual_fmac"] > facts["control_dual_fmac"]
        assert facts["candidate_scalar_fmac"] < facts["control_scalar_fmac"]
        assert facts["candidate_instruction_slots_max"] <= 1_577
        assert facts["candidate_code_bytes_max"] <= 9_728
        assert facts["metadata_vgpr_max"] <= 194


def test_h7g_registry_source_policy_and_h5y_immutability() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    assert hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY == _Q5_PRODUCTION_POLICY
    assert hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_POLICY == _Q6_PRODUCTION_POLICY
    assert all("padded_compute" not in variant for variant in _Q5_PRODUCTION_POLICY.values())
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
    assert h5y_kernel.count(_H5Y_GUARDED_COMPUTE) == 1
    assert h5y_kernel.count("if (row < rows)") == 2
    assert h5y_kernel.count("__shfl_down") == 1
    assert h5y_kernel.count("__syncthreads();") == 1

    for col_tile, row_batch, output_dtype, weight_layout, *_ in _ROLES:
        _, h5y_primitive_name, h5y_composite_name = _h5y_names(
            col_tile,
            row_batch,
            output_dtype,
            weight_layout,
        )
        assert _sha256(inspect.getsource(getattr(q5_f32, h5y_primitive_name))) == (
            _H5Y_PRIMITIVE_WRAPPER_SHA256
        )
        assert _sha256(inspect.getsource(getattr(q5_f32, h5y_composite_name))) == (
            _H5Y_COMPOSITE_WRAPPER_SHA256
        )

        # Intentional RED after every retained H5Y/source/allocation control.
        primitive_name, composite_name = _h7g_names(
            col_tile,
            row_batch,
            output_dtype,
            weight_layout,
        )
        primitive = getattr(q5_f32, primitive_name)
        composite = getattr(q5_f32, composite_name)
        assert primitive is not composite
        for key, function in zip(
            _h7g_keys(col_tile, row_batch, output_dtype, weight_layout),
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

    assert q5_f32._Q5_PADDED_COMPUTE_ROLES == tuple(role[:6] for role in _ROLES)
    assert source.count(f"__global__ void {_H7G_KERNEL}(") == 1
    candidate_kernel = _declaration(source, f"__global__ void {_H7G_KERNEL}(")
    assert candidate_kernel.count(_H7G_UNGUARDED_COMPUTE) == 1
    assert candidate_kernel.count(_H5Y_GUARDED_COMPUTE) == 0
    assert candidate_kernel.count("if (row < rows)") == 1
    assert candidate_kernel.count("__shfl_down") == 1
    assert candidate_kernel.count("__syncthreads();") == 1
    normalized_h7g = candidate_kernel.replace(
        _H7G_UNGUARDED_COMPUTE,
        _H5Y_GUARDED_COMPUTE,
    ).replace(_H7G_KERNEL, _H5Y_KERNEL)
    assert normalized_h7g == h5y_kernel

    for col_tile, row_batch, output_dtype, weight_layout, *_ in _ROLES:
        suffix = _suffix(col_tile, row_batch, output_dtype)
        symbol = (
            "hipengine_gguf_q5_k_f32_weight_ordered_weight_major_"
            f"{weight_layout}_activation_tile_k_row_padded_compute_{suffix}"
        )
        assert source.count(symbol) == 1


def test_h7g_strict_role_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_attempts = 0

    def _unexpected_build(**_kwargs: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H7G role reached HIP loading")

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

    for (
        col_tile,
        row_batch,
        output_dtype,
        weight_layout,
        in_features,
        out_features,
        _,
    ) in _ROLES:
        primitive_name, composite_name = _h7g_names(
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
                (17, in_features, out_features - col_tile, f"exactly {out_features}"),
            )
            for rows, hidden, outputs, message in invalid:
                with pytest.raises(ValueError, match=message):
                    function(*pointers, rows, hidden, outputs)
    assert load_attempts == 0


@pytest.mark.parametrize("rows", [1, 7, 8, 9, 512])
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
        "bf16-k3072-n12288-r12",
        "bf16-k6144-n3072-r5",
        "f32-k3072-n6144-r5",
        "f32-k3072-n9216-r10",
    ),
)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h7g_complete_plane_outputs_and_cpu_values_match_h5y(
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
        20260802 + 53 * rows + 17 * in_features + out_features
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

        # Intentional RED only after retained complete bytes and CPU gates.
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
