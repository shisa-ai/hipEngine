"""RED contracts for WPF-H5W exact Q6 weight-major composites."""

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
    _edge_q6_weight,
)

_CANDIDATES = (
    (16, 5, "bf16"),
    (16, 4, "bf16"),
    (16, 5, "f32"),
)
_Q5_PRODUCTION_POLICY = {
    ("bf16", 3072, 1024): "weight_major_coltile8_rowbatch4",
    ("bf16", 3072, 12288): "weight_major_coltile8_rowbatch12",
    ("bf16", 6144, 3072): "weight_major_coltile16_rowbatch5",
    ("bf16", 9216, 3072): "weight_major_coltile12_rowbatch8",
    ("f32", 3072, 48): "coltile12_rowbatch4",
    ("f32", 3072, 72): "coltile8_rowbatch4",
    ("f32", 3072, 6144): "weight_major_coltile16_rowbatch5",
    ("f32", 3072, 9216): "weight_major_coltile8_rowbatch10",
}
_Q6_PRODUCTION_POLICY = {
    ("bf16", 3072, 1024): "coltile16_rowbatch5",
    ("bf16", 1024, 3072): "coltile16_rowbatch4",
    ("f32", 3072, 72): "coltile8_rowbatch4",
    ("f32", 3072, 1024): "coltile16_rowbatch5",
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


def test_h5w_registry_scope_and_production_immutability() -> None:
    from hipengine.kernels import hip_gfx1100
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    assert q5_f32._Q6_ORDERED_WEIGHT_MAJOR_GEOMETRIES == _CANDIDATES
    q5_f32.register_gguf_q5_k_f32_rocblas_prefill_kernels(replace=True)
    register_gfx1151_kernels(replace=True)
    assert hip_gfx1100.GGUF_Q5_F32_ORDERED_PREFILL_POLICY == _Q5_PRODUCTION_POLICY
    assert hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_POLICY == _Q6_PRODUCTION_POLICY

    for col_tile, row_batch, output_dtype in _CANDIDATES:
        suffix = _suffix(col_tile, row_batch, output_dtype)
        primitive = getattr(
            q5_f32,
            f"gguf_q5_k_f32_weight_ordered_weight_major_{suffix}",
        )
        q5_composite = getattr(
            q5_f32,
            f"gguf_q5_k_f32_ordered_weight_major_{suffix}",
        )
        q6_composite = getattr(
            q5_f32,
            f"gguf_q6_k_f32_ordered_weight_major_{suffix}",
        )
        primitive_key = KernelKey(
            "hip_gfx1100",
            "linear",
            "f32_weight",
            f"ordered_weight_major_{suffix}",
        )
        q5_key = KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q5_k",
            f"f32_ordered_weight_major_{suffix}",
        )
        q6_key = KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k",
            f"f32_ordered_weight_major_{suffix}",
        )
        for key, function in (
            (primitive_key, primitive),
            (q5_key, q5_composite),
            (q6_key, q6_composite),
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


def test_h5w_preflight_rejects_before_loading_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = getattr(
        q5_f32,
        "gguf_q6_k_f32_ordered_weight_major_"
        "coltile16_rowbatch5_bf16_bf16_out",
    )

    def _unexpected_build(**_kwargs):
        raise AssertionError("invalid H5W shape reached HIP loading")

    monkeypatch.setattr(
        q5_f32,
        "build_gguf_q5_k_f32_rocblas_prefill",
        _unexpected_build,
    )
    with pytest.raises(ValueError, match="rows must be positive"):
        candidate(1, 2, 3, 4, 0, 512, 48)
    with pytest.raises(ValueError, match="multiple of 256"):
        candidate(1, 2, 3, 4, 17, 384, 48)
    with pytest.raises(ValueError, match="divisible by 16"):
        candidate(1, 2, 3, 4, 17, 512, 50)


@pytest.mark.parametrize("rows", [17, 33])
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_h5w_composites_match_h5i_bytes(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime

    in_features, out_features = 512, 48
    rng = np.random.default_rng(20260730 + 29 * rows)
    x_bf16 = _bf16_bits(
        rng.normal(0.0, 0.2, size=(rows, in_features)).astype(np.float32)
    )
    qweight = _edge_q6_weight(out_features, in_features)
    runtime = get_hip_runtime()
    library = q5_f32.build_gguf_q5_k_f32_rocblas_prefill(load=True)
    before = memory_stats()
    buffers = []
    try:
        x_dev = _device(x_bf16, runtime)
        weight_dev = _device(qweight, runtime)
        weight_f32_dev = malloc(
            q5_f32.q6_k_f32_ordered_workspace_nbytes(
                in_features, out_features
            ),
            runtime=runtime,
        )
        buffers.extend((x_dev, weight_dev, weight_f32_dev))
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
                f"gguf_q6_k_f32_ordered_{suffix}",
            )
            candidate = getattr(
                q5_f32,
                f"gguf_q6_k_f32_ordered_weight_major_{suffix}",
            )
            for function, output in (
                (control, expected_dev),
                (candidate, actual_dev),
            ):
                function(
                    x_dev.ptr,
                    weight_dev.ptr,
                    output.ptr,
                    weight_f32_dev.ptr,
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
                host_array_ptr(actual),
                actual_dev,
                actual.nbytes,
                runtime=runtime,
            )
            np.testing.assert_array_equal(actual, expected)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
    after = memory_stats()
    assert after["current_allocated_bytes"] == before["current_allocated_bytes"]
    assert after["active_allocations"] == before["active_allocations"]
