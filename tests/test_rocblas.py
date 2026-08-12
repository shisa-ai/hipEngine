from __future__ import annotations

import pytest

from hipengine.core.rocblas import (
    ROCBLAS_DATATYPE_F16_R,
    ROCBLAS_DATATYPE_F32_R,
    Rocblas,
)


class _Call:
    def __init__(self) -> None:
        self.args = None

    def __call__(self, *args):
        self.args = args
        return 0


class _Library:
    def __init__(self) -> None:
        self.rocblas_set_stream = _Call()
        self.rocblas_set_workspace = _Call()
        self.rocblas_gemm_ex = _Call()
        self.rocblas_sgemm = _Call()


def _value(arg) -> int:
    if hasattr(arg, "value"):
        return 0 if arg.value is None else int(arg.value)
    return int(arg)


def test_rocblas_caller_workspace_validates_and_forwards_contract() -> None:
    library = _Library()
    blas = Rocblas(library=library, handle=17)

    blas.set_workspace(0, 0)
    assert library.rocblas_set_workspace.args is not None
    assert tuple(_value(arg) for arg in library.rocblas_set_workspace.args) == (
        17,
        0,
        0,
    )

    with pytest.raises(ValueError, match="pointer/size"):
        blas.set_workspace(0, 4096)
    with pytest.raises(ValueError, match="pointer/size"):
        blas.set_workspace(-1, 0)


def test_rocblas_fp16_compute_f16_uses_f16_output_and_compute_descriptors() -> None:
    library = _Library()
    blas = Rocblas(library=library, handle=17)

    blas.gemm_ex_rowmajor_nt_fp16_compute_f16(
        101,
        202,
        303,
        rows=512,
        in_features=3072,
        out_features=1024,
        stream=404,
    )

    assert library.rocblas_set_stream.args is not None
    assert tuple(_value(arg) for arg in library.rocblas_set_stream.args) == (17, 404)
    args = library.rocblas_gemm_ex.args
    assert args is not None
    assert _value(args[8]) == ROCBLAS_DATATYPE_F16_R
    assert _value(args[11]) == ROCBLAS_DATATYPE_F16_R
    assert _value(args[15]) == ROCBLAS_DATATYPE_F16_R
    assert _value(args[18]) == ROCBLAS_DATATYPE_F16_R
    assert _value(args[20]) == ROCBLAS_DATATYPE_F16_R
    assert _value(args[16]) == 1024
    assert _value(args[19]) == 1024


def test_rocblas_fp16_f32_out_uses_fp32_c_and_d_descriptors() -> None:
    library = _Library()
    blas = Rocblas(library=library, handle=17)

    blas.gemm_ex_rowmajor_nt_fp16_f32_out(
        101,
        202,
        303,
        rows=32,
        in_features=3072,
        out_features=9216,
        stream=404,
    )

    assert library.rocblas_set_stream.args is not None
    assert tuple(_value(arg) for arg in library.rocblas_set_stream.args) == (17, 404)
    args = library.rocblas_gemm_ex.args
    assert args is not None
    assert _value(args[8]) == ROCBLAS_DATATYPE_F16_R
    assert _value(args[11]) == ROCBLAS_DATATYPE_F16_R
    assert _value(args[15]) == ROCBLAS_DATATYPE_F32_R
    assert _value(args[18]) == ROCBLAS_DATATYPE_F32_R
    assert _value(args[20]) == ROCBLAS_DATATYPE_F32_R
    assert _value(args[16]) == 9216
    assert _value(args[19]) == 9216


def test_rocblas_fp16_f32_out_rejects_invalid_shapes_before_launch() -> None:
    library = _Library()
    blas = Rocblas(library=library, handle=17)

    with pytest.raises(ValueError, match="rows must be positive"):
        blas.gemm_ex_rowmajor_nt_fp16_f32_out(
            101,
            202,
            303,
            rows=0,
            in_features=3072,
            out_features=9216,
        )

    assert library.rocblas_gemm_ex.args is None


def test_rocblas_sgemm_rowmajor_nt_uses_f32_shape_and_leading_dimensions() -> None:
    library = _Library()
    blas = Rocblas(library=library, handle=17)

    blas.sgemm_rowmajor_nt(
        101,
        202,
        303,
        rows=512,
        in_features=3072,
        out_features=12288,
        stream=404,
    )

    assert library.rocblas_set_stream.args is not None
    assert tuple(_value(arg) for arg in library.rocblas_set_stream.args) == (17, 404)
    args = library.rocblas_sgemm.args
    assert args is not None
    assert tuple(_value(args[index]) for index in (3, 4, 5)) == (12288, 512, 3072)
    assert _value(args[8]) == 3072
    assert _value(args[10]) == 3072
    assert _value(args[13]) == 12288


def test_rocblas_sgemm_rowmajor_nt_rejects_invalid_shapes_before_launch() -> None:
    library = _Library()
    blas = Rocblas(library=library, handle=17)

    with pytest.raises(ValueError, match="out_features must be positive"):
        blas.sgemm_rowmajor_nt(
            101,
            202,
            303,
            rows=17,
            in_features=256,
            out_features=0,
        )

    assert library.rocblas_sgemm.args is None
