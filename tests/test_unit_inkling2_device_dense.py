"""CPU tests for the Inkling 2 resident dense projection's BLAS entry choice (no GPU)."""

from types import SimpleNamespace

from hipengine.loading.inkling2_device_dense import Inkling2DeviceDense


def _dense(calls):
    blas = SimpleNamespace(sgemv_rowmajor_nt=lambda *args, **kwargs: calls.append("gemv"),
                           sgemm_rowmajor_nt=lambda *args, **kwargs: calls.append("gemm"))
    return Inkling2DeviceDense(None, runtime=None, blas=blas)


def test_one_resident_row_goes_through_the_matrix_vector_entry():
    # Resident plans always describe their rows as a 2-D block. rocBLAS's GEMM entry answers a
    # one-row block with a 64x64-tile Tensile kernel: 143 us against 12 us for GEMV on gfx1100.
    calls = []
    matrix = (SimpleNamespace(ptr=0x3000), (2304, 1152), False)
    _dense(calls)._launch_projection(0x1000, 0x2000, 1, 2, matrix, None)
    assert calls == ["gemv"]


def test_vector_and_multi_row_blocks_keep_their_entries():
    calls = []
    matrix = (SimpleNamespace(ptr=0x3000), (2304, 1152), False)
    dense = _dense(calls)
    dense._launch_projection(0x1000, 0x2000, 1, 1, matrix, None)
    dense._launch_projection(0x1000, 0x2000, 64, 2, matrix, None)
    dense._launch_projection(0x1000, 0x2000, 2, 2, matrix, None)
    assert calls == ["gemv", "gemm", "gemm"]


def test_native_bf16_rows_never_reach_rocblas():
    calls = []
    native = []
    dense = _dense(calls)
    dense._launch_projection(0x1000, 0x2000, 1, 2, (SimpleNamespace(ptr=0x3000), (2304, 1152), True),
                             lambda *args, **kwargs: native.append(args))
    assert calls == [] and native == [(0x1000, 0x3000, 0x2000, 1, 1152, 2304)]
