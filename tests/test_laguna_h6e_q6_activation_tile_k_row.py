"""RED contracts for WPF-H6E exact Q6 activation-tile-K-row transfer."""

from __future__ import annotations

import ctypes
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
from hipengine.kernels.hip_gfx1100.quant import (
    gguf_q5_k_f32_rocblas_prefill as q5_f32,
)
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.runtime.laguna_gguf_runner import LagunaQ5F32OrderedScratch
from tests.test_gguf_q5_k_f32_rocblas_prefill import (
    _bf16_bits,
    _bf16_to_f32,
    _device,
    _edge_q6_weight,
    _exact_q6_f32_cpu,
)

# col tile, row batch, output dtype, exact K, exact N, retained H5Y pack col tile
_ROLES = (
    (16, 5, "bf16", 3_072, 1_024, 16),
    (16, 4, "bf16", 1_024, 3_072, 8),
    (16, 5, "f32", 3_072, 1_024, 16),
)
_Q5_PRODUCTION_POLICY = {
    ("bf16", 3072, 1024): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile8_rowbatch4"
    ),
    ("bf16", 3072, 12288): (
        "weight_major_row_major_activation_tile_k_row_coltile8_rowbatch12"
    ),
    ("bf16", 6144, 3072): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile16_rowbatch5"
    ),
    ("bf16", 9216, 3072): (
        "weight_major_row_major_activation_tile_k_row_coltile12_rowbatch8"
    ),
    ("f32", 3072, 48): "coltile12_rowbatch4",
    ("f32", 3072, 72): "coltile8_rowbatch4",
    ("f32", 3072, 6144): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile16_rowbatch5"
    ),
    ("f32", 3072, 9216): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile8_rowbatch10"
    ),
}
_Q6_PRODUCTION_POLICY = {
    ("bf16", 3072, 1024): (
        "weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch5"
    ),
    ("bf16", 1024, 3072): (
        "weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch4"
    ),
    ("f32", 3072, 72): "coltile8_rowbatch4",
    ("f32", 3072, 1024): (
        "weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch5"
    ),
}


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _suffix(col_tile: int, row_batch: int, output_dtype: str) -> str:
    return (
        f"coltile{col_tile}_rowbatch{row_batch}_"
        f"bf16_{output_dtype}_out"
    )


def _candidate_names(
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


def _candidate_keys(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    *,
    backend: str = "hip_gfx1100",
) -> tuple[KernelKey, KernelKey]:
    suffix = _suffix(col_tile, row_batch, output_dtype)
    stem = f"weight_major_row_major_activation_tile_k_row_{suffix}"
    return (
        KernelKey(backend, "linear", "f32_weight", f"ordered_{stem}"),
        KernelKey(backend, "linear", "gguf_q6_k", f"f32_ordered_{stem}"),
    )


def _retained_pack(
    pack_col_tile: int,
    row_batch: int,
    output_dtype: str,
):
    return getattr(
        q5_f32,
        "gguf_bf16_activation_pack_tile_k_row_"
        f"{_suffix(pack_col_tile, row_batch, output_dtype)}",
    )


def _activation_plane_shape(
    rows: int,
    in_features: int,
    row_batch: int,
) -> tuple[int, int, int]:
    return (
        math.ceil(rows / row_batch),
        in_features,
        math.ceil(row_batch / 8) * 8,
    )


def _expected_activation_plane(
    x_bf16: np.ndarray,
    row_batch: int,
) -> np.ndarray:
    rows, in_features = x_bf16.shape
    row_groups, _, padded_rows = _activation_plane_shape(
        rows,
        in_features,
        row_batch,
    )
    expected = np.zeros(
        (row_groups, in_features, padded_rows),
        dtype=np.uint16,
    )
    for row_group in range(row_groups):
        row_base = row_group * row_batch
        live_rows = min(row_batch, rows - row_base)
        expected[row_group, :, :live_rows] = x_bf16[
            row_base : row_base + live_rows
        ].T
    return expected


def _sampled_cpu_gate(
    actual: np.ndarray,
    x_bf16: np.ndarray,
    qweight: np.ndarray,
    *,
    row_batch: int,
    output_dtype: str,
    in_features: int,
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
    cpu_weight = _exact_q6_f32_cpu(qweight[sample_cols], in_features)
    cpu = np.asarray(
        _bf16_to_f32(x_bf16[sample_rows]) @ cpu_weight.T,
        dtype=np.float32,
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
        (in_features, out_features): _edge_q6_weight(
            out_features,
            in_features,
        )
        for _, _, _, in_features, out_features, _ in _ROLES
    }


def test_h6e_registry_workspace_scope_and_production_immutability() -> None:
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

    assert LagunaQ5F32OrderedScratch.weight_f32_planned_nbytes() == 150_994_944
    assert (
        LagunaQ5F32OrderedScratch.activation_bf16_planned_nbytes(max_rows=512)
        == 10_125_312
    )
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == 161_120_256

    for (
        col_tile,
        row_batch,
        output_dtype,
        in_features,
        out_features,
        pack_col_tile,
    ) in _ROLES:
        plane_nbytes = q5_f32.q5_k_f32_activation_tile_k_row_nbytes(
            512,
            in_features,
            row_batch,
        )
        assert plane_nbytes in {2_097_152, 5_062_656}
        assert plane_nbytes <= 10_125_312
        retained_pack = _retained_pack(
            pack_col_tile,
            row_batch,
            output_dtype,
        )
        retained_pack_key = KernelKey(
            "hip_gfx1100",
            "activation_pack",
            "bf16",
            "tile_k_row_"
            f"{_suffix(pack_col_tile, row_batch, output_dtype)}",
        )
        assert resolve(
            backend=retained_pack_key.backend,
            layer=retained_pack_key.layer,
            quant=retained_pack_key.quant,
            variant=retained_pack_key.variant,
        ) is retained_pack

        primitive_name, composite_name = _candidate_names(
            col_tile,
            row_batch,
            output_dtype,
        )
        primitive = getattr(q5_f32, primitive_name)
        composite = getattr(q5_f32, composite_name)
        control = getattr(
            q5_f32,
            "gguf_q6_k_f32_ordered_weight_major_"
            f"{_suffix(col_tile, row_batch, output_dtype)}",
        )
        assert composite is not control
        for key, function in zip(
            _candidate_keys(col_tile, row_batch, output_dtype),
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


def test_h6e_strict_role_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_attempts = 0

    def _unexpected_build(**_kwargs: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H6E role reached HIP loading")

    monkeypatch.setattr(
        q5_f32,
        "build_gguf_q5_k_f32_rocblas_prefill",
        _unexpected_build,
    )
    for col_tile, row_batch, output_dtype, in_features, out_features, _ in _ROLES:
        primitive_name, composite_name = _candidate_names(
            col_tile,
            row_batch,
            output_dtype,
        )
        primitive = getattr(q5_f32, primitive_name)
        composite = getattr(q5_f32, composite_name)
        for function, pointers in (
            (primitive, (1, 2, 3)),
            (composite, (1, 2, 3, 4, 5)),
        ):
            with pytest.raises(ValueError, match="rows must be positive"):
                function(*pointers, 0, in_features, out_features)
            with pytest.raises(ValueError, match=f"exactly {in_features}"):
                function(*pointers, 17, in_features - 256, out_features)
            with pytest.raises(ValueError, match=f"exactly {out_features}"):
                function(*pointers, 17, in_features, out_features - 16)
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
def test_h6e_complete_plane_outputs_and_cpu_values_match_h5w(
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
        20260731 + 41 * rows + 7 * in_features + out_features
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

        control = getattr(
            q5_f32,
            "gguf_q6_k_f32_ordered_weight_major_"
            f"{_suffix(col_tile, row_batch, output_dtype)}",
        )
        control(
            x_dev.ptr,
            weight_dev.ptr,
            expected_dev.ptr,
            weight_f32_dev.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        retained_pack = _retained_pack(
            pack_col_tile,
            row_batch,
            output_dtype,
        )
        retained_pack(
            x_dev.ptr,
            activation_dev.ptr,
            rows,
            in_features,
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

        runtime.memset(activation_dev.ptr, 0xA5, plane.nbytes)
        runtime.memset(actual_dev.ptr, 0x5A, actual.nbytes)
        _, composite_name = _candidate_names(
            col_tile,
            row_batch,
            output_dtype,
        )
        candidate = getattr(q5_f32, composite_name)
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
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
