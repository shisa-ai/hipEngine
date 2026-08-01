"""RED contracts for WPF-H5X exact tile-K-col F32 Q5 planes."""

from __future__ import annotations

import ctypes

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
from tests.test_gguf_q5_k_f32_rocblas_prefill import (
    _bf16_bits,
    _device,
    _edge_q5_weight,
)

_CANDIDATES = (
    (8, 4, "bf16"),
    (16, 5, "bf16"),
    (16, 5, "f32"),
    (8, 10, "f32"),
)
_REJECTED = (
    (8, 12, "bf16"),
    (12, 8, "bf16"),
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
) -> tuple[str, str, str]:
    suffix = _suffix(col_tile, row_batch, output_dtype)
    return (
        f"gguf_q5_k_dequantize_f32_exact_tile_k_col_{suffix}",
        f"gguf_q5_k_f32_weight_ordered_weight_major_tile_k_col_{suffix}",
        f"gguf_q5_k_f32_ordered_weight_major_tile_k_col_{suffix}",
    )


def test_h5x_registry_scope_and_production_immutability() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    assert q5_f32._Q5_TILE_K_COL_GEOMETRIES == _CANDIDATES
    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    assert hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY == (
        _Q5_PRODUCTION_POLICY
    )
    assert hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_POLICY == (
        _Q6_PRODUCTION_POLICY
    )

    for col_tile, row_batch, output_dtype in _CANDIDATES:
        suffix = _suffix(col_tile, row_batch, output_dtype)
        retained_primitive = getattr(
            q5_f32,
            f"gguf_q5_k_f32_weight_ordered_weight_major_{suffix}",
        )
        retained_composite = getattr(
            q5_f32,
            f"gguf_q5_k_f32_ordered_weight_major_{suffix}",
        )
        producer_name, primitive_name, composite_name = _candidate_names(
            col_tile,
            row_batch,
            output_dtype,
        )
        producer = getattr(q5_f32, producer_name)
        primitive = getattr(q5_f32, primitive_name)
        composite = getattr(q5_f32, composite_name)
        assert primitive is not retained_primitive
        assert composite is not retained_composite

        retained_keys = (
            (
                KernelKey(
                    "hip_gfx1100",
                    "linear",
                    "f32_weight",
                    f"ordered_weight_major_{suffix}",
                ),
                retained_primitive,
            ),
            (
                KernelKey(
                    "hip_gfx1100",
                    "linear",
                    "gguf_q5_k",
                    f"f32_ordered_weight_major_{suffix}",
                ),
                retained_composite,
            ),
        )
        candidate_keys = (
            (
                KernelKey(
                    "hip_gfx1100",
                    "dequant",
                    "gguf_q5_k",
                    f"raw_f32_exact_tile_k_col_{suffix}",
                ),
                producer,
            ),
            (
                KernelKey(
                    "hip_gfx1100",
                    "linear",
                    "f32_weight",
                    f"ordered_weight_major_tile_k_col_{suffix}",
                ),
                primitive,
            ),
            (
                KernelKey(
                    "hip_gfx1100",
                    "linear",
                    "gguf_q5_k",
                    f"f32_ordered_weight_major_tile_k_col_{suffix}",
                ),
                composite,
            ),
        )
        for key, function in (*retained_keys, *candidate_keys):
            assert resolve(
                backend=key.backend,
                layer=key.layer,
                quant=key.quant,
                variant=key.variant,
            ) is function
        for key, _ in candidate_keys:
            assert not is_registered(
                KernelKey("hip_gfx1151", key.layer, key.quant, key.variant)
            )

    for col_tile, row_batch, output_dtype in _REJECTED:
        suffix = _suffix(col_tile, row_batch, output_dtype)
        for name in _candidate_names(col_tile, row_batch, output_dtype):
            assert not hasattr(q5_f32, name)
        for layer, quant, variant in (
            (
                "dequant",
                "gguf_q5_k",
                f"raw_f32_exact_tile_k_col_{suffix}",
            ),
            (
                "linear",
                "f32_weight",
                f"ordered_weight_major_tile_k_col_{suffix}",
            ),
            (
                "linear",
                "gguf_q5_k",
                f"f32_ordered_weight_major_tile_k_col_{suffix}",
            ),
        ):
            assert not is_registered(
                KernelKey("hip_gfx1100", layer, quant, variant)
            )
            assert not is_registered(
                KernelKey("hip_gfx1151", layer, quant, variant)
            )


def test_h5x_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, composite_name = _candidate_names(8, 4, "bf16")
    candidate = getattr(q5_f32, composite_name)

    def _unexpected_build(**_kwargs):
        raise AssertionError("invalid H5X shape reached HIP loading")

    monkeypatch.setattr(
        q5_f32,
        "build_gguf_q5_k_f32_rocblas_prefill",
        _unexpected_build,
    )
    with pytest.raises(ValueError, match="rows must be positive"):
        candidate(1, 2, 3, 4, 0, 512, 48)
    with pytest.raises(ValueError, match="multiple of 256"):
        candidate(1, 2, 3, 4, 17, 384, 48)
    with pytest.raises(ValueError, match="divisible by 8"):
        candidate(1, 2, 3, 4, 17, 512, 50)


@pytest.mark.parametrize("rows", [17, 33])
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h5x_plane_permutation_and_outputs_match_h5l_bytes(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime

    in_features, out_features = 512, 48
    rng = np.random.default_rng(20260730 + 31 * rows)
    x_bf16 = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    qweight = _edge_q5_weight(out_features, in_features)
    runtime = get_hip_runtime()
    library = q5_f32.build_gguf_q5_k_f32_rocblas_prefill(load=True)
    plane_nbytes = q5_f32.q5_k_f32_ordered_workspace_nbytes(
        in_features,
        out_features,
    )
    before = memory_stats()
    buffers = []
    try:
        x_dev = _device(x_bf16, runtime)
        weight_dev = _device(qweight, runtime)
        row_major_dev = malloc(plane_nbytes, runtime=runtime)
        tile_k_col_dev = malloc(plane_nbytes, runtime=runtime)
        buffers.extend((x_dev, weight_dev, row_major_dev, tile_k_col_dev))
        row_major = np.empty((out_features, in_features), dtype=np.float32)
        tile_k_col = np.empty(row_major.size, dtype=np.float32)

        for col_tile, row_batch, output_dtype in _CANDIDATES:
            host_dtype = np.uint16 if output_dtype == "bf16" else np.float32
            expected = np.empty((rows, out_features), dtype=host_dtype)
            actual = np.empty_like(expected)
            expected_dev = malloc(expected.nbytes, runtime=runtime)
            actual_dev = malloc(actual.nbytes, runtime=runtime)
            buffers.extend((expected_dev, actual_dev))
            suffix = _suffix(col_tile, row_batch, output_dtype)
            control = getattr(
                q5_f32,
                f"gguf_q5_k_f32_ordered_weight_major_{suffix}",
            )
            producer_name, _, composite_name = _candidate_names(
                col_tile,
                row_batch,
                output_dtype,
            )
            producer = getattr(q5_f32, producer_name)
            candidate = getattr(q5_f32, composite_name)

            control(
                x_dev.ptr,
                weight_dev.ptr,
                expected_dev.ptr,
                row_major_dev.ptr,
                rows,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
            candidate(
                x_dev.ptr,
                weight_dev.ptr,
                actual_dev.ptr,
                tile_k_col_dev.ptr,
                rows,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
            producer(
                weight_dev.ptr,
                tile_k_col_dev.ptr,
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
                host_array_ptr(actual),
                actual_dev,
                actual.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(row_major),
                row_major_dev,
                row_major.nbytes,
                runtime=runtime,
            )
            copy_device_to_host(
                host_array_ptr(tile_k_col),
                tile_k_col_dev,
                tile_k_col.nbytes,
                runtime=runtime,
            )
            np.testing.assert_array_equal(actual, expected)
            expected_tile_k_col = (
                row_major.reshape(
                    out_features // col_tile,
                    col_tile,
                    in_features,
                )
                .transpose(0, 2, 1)
                .copy()
                .reshape(-1)
            )
            np.testing.assert_array_equal(
                tile_k_col.view(np.uint32),
                expected_tile_k_col.view(np.uint32),
            )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
