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
        self.rocblas_gemm_ex = _Call()


def _value(arg) -> int:
    return int(arg.value) if hasattr(arg, "value") else int(arg)


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
