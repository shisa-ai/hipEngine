"""RED contracts for WPF-H6G exact Q5 one-step K-record prefetch."""

from __future__ import annotations

import ctypes
import hashlib
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

# col tile, row batch, output dtype, weight layout, exact K, exact N
_ROLES = (
    (12, 8, "bf16", "row_major", 9_216, 3_072),
    (8, 10, "f32", "tile_k_col", 3_072, 9_216),
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
_SOURCE = Path(q5_f32.__file__).with_suffix(".hip")
_H5Y_KERNEL = (
    "gguf_q5_k_f32_weight_ordered_weight_major_"
    "activation_tile_k_row_kernel"
)
_H5Y_KERNEL_SHA256 = (
    "5a8ba4a9ec504bef2687aff93c8bd92833bb77ef2baf664b1282ca3fcf256f54"
)
_H6G_KERNEL = (
    "gguf_q5_k_f32_weight_ordered_weight_major_"
    "activation_tile_k_row_k_record_prefetch_kernel"
)
_LOAD_HELPER = "load_q5_k_f32_activation_tile_k_row_prefetch_record"
_QK_K = 256
_Q5_BLOCK_BYTES = 176


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


def _retained_names(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    weight_layout: str,
) -> tuple[str, str, str]:
    suffix = _suffix(col_tile, row_batch, output_dtype)
    stem = f"weight_major_{weight_layout}_activation_tile_k_row_{suffix}"
    return (
        f"gguf_bf16_activation_pack_tile_k_row_{suffix}",
        f"gguf_q5_k_f32_weight_ordered_{stem}",
        f"gguf_q5_k_f32_ordered_{stem}",
    )


def _candidate_names(
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    weight_layout: str,
) -> tuple[str, str]:
    suffix = _suffix(col_tile, row_batch, output_dtype)
    stem = (
        f"weight_major_{weight_layout}_activation_tile_k_row_"
        f"k_record_prefetch_{suffix}"
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
        f"k_record_prefetch_{suffix}"
    )
    return (
        KernelKey(backend, "linear", "f32_weight", f"ordered_{stem}"),
        KernelKey(backend, "linear", "gguf_q5_k", f"f32_ordered_{stem}"),
    )


def _function_source(source: str, declaration: str) -> str:
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
    raise AssertionError(f"unterminated function: {declaration}")


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
    out_features = actual.shape[1]
    sample_cols = np.unique(
        np.concatenate(
            (
                np.arange(0, 16),
                np.arange(out_features // 2, out_features // 2 + 16),
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


def _production_q5_weight(
    out_features: int,
    in_features: int,
) -> np.ndarray:
    blocks_per_row = in_features // _QK_K
    rng = np.random.default_rng(20260731 + in_features + out_features)
    raw = rng.integers(
        0,
        256,
        size=(out_features, blocks_per_row * _Q5_BLOCK_BYTES),
        dtype=np.uint8,
    )
    blocks = raw.reshape(out_features, blocks_per_row, _Q5_BLOCK_BYTES)
    d = np.asarray([np.float16(0.015625)]).view(np.uint8)
    dmin = np.asarray([np.float16(0.0078125)]).view(np.uint8)
    blocks[..., 0:2] = d
    blocks[..., 2:4] = dmin
    return raw


@pytest.fixture(scope="module")
def production_qweights() -> dict[tuple[int, int], np.ndarray]:
    return {
        (in_features, out_features): _production_q5_weight(
            out_features,
            in_features,
        )
        for _, _, _, _, in_features, out_features in _ROLES
    }


def test_h6g_registry_schedule_workspace_and_production_immutability() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    assert hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY == (
        _Q5_PRODUCTION_POLICY
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

    source = _SOURCE.read_text()
    h5y_source = _function_source(source, f"__global__ void {_H5Y_KERNEL}")
    assert hashlib.sha256(h5y_source.encode()).hexdigest() == (
        _H5Y_KERNEL_SHA256
    )

    assert q5_f32._Q5_ACTIVATION_TILE_K_ROW_PREFETCH_ROLES == _ROLES
    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)

    for col_tile, row_batch, output_dtype, weight_layout, _, _ in _ROLES:
        pack_name, control_primitive_name, control_composite_name = (
            _retained_names(
                col_tile,
                row_batch,
                output_dtype,
                weight_layout,
            )
        )
        primitive_name, composite_name = _candidate_names(
            col_tile,
            row_batch,
            output_dtype,
            weight_layout,
        )
        pack = getattr(q5_f32, pack_name)
        control_primitive = getattr(q5_f32, control_primitive_name)
        control_composite = getattr(q5_f32, control_composite_name)
        primitive = getattr(q5_f32, primitive_name)
        composite = getattr(q5_f32, composite_name)
        assert primitive is not control_primitive
        assert composite is not control_composite

        pack_key = KernelKey(
            "hip_gfx1100",
            "activation_pack",
            "bf16",
            f"tile_k_row_{_suffix(col_tile, row_batch, output_dtype)}",
        )
        assert resolve(
            backend=pack_key.backend,
            layer=pack_key.layer,
            quant=pack_key.quant,
            variant=pack_key.variant,
        ) is pack
        for key, function in zip(
            _candidate_keys(
                col_tile,
                row_batch,
                output_dtype,
                weight_layout,
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
        assert source.count(f"hipengine_{primitive_name}") == 1

    candidate_source = _function_source(
        source,
        f"__global__ void {_H6G_KERNEL}",
    )
    helper_offsets = []
    search_from = 0
    while True:
        offset = candidate_source.find(_LOAD_HELPER, search_from)
        if offset < 0:
            break
        helper_offsets.append(offset)
        search_from = offset + len(_LOAD_HELPER)
    assert len(helper_offsets) == 2
    assert "const int next_k" in candidate_source
    assert helper_offsets[1] < candidate_source.index("fmaf(")
    assert "for (int offset = 16; offset > 0; offset >>= 1)" in candidate_source
    assert "wave_sums[4][ROW_BATCH][COL_TILE]" in candidate_source
    assert "for (int wave_index = 0; wave_index < 4; ++wave_index)" in (
        candidate_source
    )
    assert "store_output<out_t>(" in candidate_source


def test_h6g_strict_role_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_attempts = 0

    def _unexpected_build(**_kwargs: object) -> None:
        nonlocal load_attempts
        load_attempts += 1
        raise AssertionError("invalid H6G role reached HIP loading")

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
    ) in _ROLES:
        primitive_name, composite_name = _candidate_names(
            col_tile,
            row_batch,
            output_dtype,
            weight_layout,
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
                function(*pointers, 17, in_features, out_features - col_tile)
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
    ),
    _ROLES,
    ids=("bf16-k9216-n3072", "f32-k3072-n9216"),
)
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h6g_complete_outputs_planes_and_cpu_values_match_h5y(
    rows: int,
    col_tile: int,
    row_batch: int,
    output_dtype: str,
    weight_layout: str,
    in_features: int,
    out_features: int,
    library: Any,
    production_qweights: dict[tuple[int, int], np.ndarray],
) -> None:
    from hipengine.core.hip import get_hip_runtime

    rng = np.random.default_rng(
        20260731 + 43 * rows + 7 * in_features + out_features
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
    weight_plane = np.empty(in_features * out_features, dtype=np.float32)

    runtime = get_hip_runtime()
    before = memory_stats()
    buffers: list[Any] = []
    try:
        x_dev = _device(x_bf16, runtime)
        weight_dev = _device(qweight, runtime)
        expected_weight_dev = malloc(weight_plane.nbytes, runtime=runtime)
        actual_weight_dev = malloc(weight_plane.nbytes, runtime=runtime)
        expected_dev = malloc(expected.nbytes, runtime=runtime)
        actual_dev = malloc(actual.nbytes, runtime=runtime)
        expected_activation_dev = malloc(plane.nbytes, runtime=runtime)
        actual_activation_dev = malloc(plane.nbytes, runtime=runtime)
        buffers.extend(
            (
                x_dev,
                weight_dev,
                expected_weight_dev,
                actual_weight_dev,
                expected_dev,
                actual_dev,
                expected_activation_dev,
                actual_activation_dev,
            )
        )

        _, _, control_name = _retained_names(
            col_tile,
            row_batch,
            output_dtype,
            weight_layout,
        )
        control = getattr(q5_f32, control_name)
        control(
            x_dev.ptr,
            weight_dev.ptr,
            expected_dev.ptr,
            expected_weight_dev.ptr,
            expected_activation_dev.ptr,
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
            expected_activation_dev,
            plane.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(weight_plane),
            expected_weight_dev,
            weight_plane.nbytes,
            runtime=runtime,
        )
        np.testing.assert_array_equal(plane, expected_plane)
        expected_weight_sha256 = hashlib.sha256(weight_plane).hexdigest()
        _sampled_cpu_gate(
            expected,
            x_bf16,
            qweight,
            row_batch=row_batch,
            output_dtype=output_dtype,
        )

        runtime.memset(actual_weight_dev.ptr, 0xA5, weight_plane.nbytes)
        runtime.memset(actual_activation_dev.ptr, 0x5A, plane.nbytes)
        runtime.memset(actual_dev.ptr, 0x3C, actual.nbytes)
        _, composite_name = _candidate_names(
            col_tile,
            row_batch,
            output_dtype,
            weight_layout,
        )
        candidate = getattr(q5_f32, composite_name)
        candidate(
            x_dev.ptr,
            weight_dev.ptr,
            actual_dev.ptr,
            actual_weight_dev.ptr,
            actual_activation_dev.ptr,
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
            actual_activation_dev,
            plane.nbytes,
            runtime=runtime,
        )
        copy_device_to_host(
            host_array_ptr(weight_plane),
            actual_weight_dev,
            weight_plane.nbytes,
            runtime=runtime,
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(plane, expected_plane)
        assert hashlib.sha256(weight_plane).hexdigest() == (
            expected_weight_sha256
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
