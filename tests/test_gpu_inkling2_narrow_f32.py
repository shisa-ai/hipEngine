"""The F32 projections of few outputs in an Inkling prompt block through the narrow FP32 kernel.

`fast` and `production` take the kernel from a few rows, and a row at a time below that; `strict`
keeps rocBLAS. The kernel keeps FP32 products and sums and changes only the order of the sum over
the inputs, so the oracle is the FP64 product. The last test needs the GGUF of
`HIPENGINE_INKLING2_MINI_GGUF` and compares the kernel with rocBLAS on the weights of the model.
"""

from __future__ import annotations

import ctypes
import os
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from hipengine import LLM
from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.core.rocblas import Rocblas
from hipengine.kernels.backends import resolve_backend
from hipengine.kernels.hip_gfx1100.linear.narrow_f32_gemm import (
    NARROW_F32_VARIANT,
    narrow_f32_gemm,
    serves,
)
from hipengine.loading.inkling2_gguf import Inkling2GGUFWeights
from hipengine.loading.inkling2_gguf_device import Q, _NARROW_F32_MIN_ROWS, Inkling2GGUFDeviceProjections

from tests.test_unit_inkling2_gguf import _file

Copy = HipMemcpyKind
# The outputs of the Inkling projections and their inputs; 128 inputs is the smallest step served.
SHAPES = [(2304, 24), (2304, 32), (4608, 64), (2304, 128), (128, 24)]


def _backend():
    try:
        ctypes.CDLL("libamdhip64.so")
        ctypes.CDLL("librocblas.so")
    except OSError as exc:
        pytest.skip(f"HIP or rocBLAS unavailable: {exc}")
    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    return backend


@pytest.mark.parametrize("rows", [2, 9, 33, 300])
@pytest.mark.parametrize("n,m", SHAPES)
def test_narrow_f32_gemm_matches_the_fp64_product(rows, n, m):
    """Every output is the FP64 product to FP32 rounding, and the rows past the input stay untouched."""
    _backend()
    rt = get_hip_runtime()
    rng = np.random.default_rng(rows + n + m)
    x = rng.normal(0, 0.7, (rows, n)).astype(np.float32)
    weight = (rng.normal(0, 0.05, (m, n)) ** 3).astype(np.float32)
    weight[0, :4] = (0, 1, -1, 0.5)
    actual = np.full((rows + 3, m), np.nan, np.float32)
    x_ptr, w_ptr, out_ptr = (rt.malloc(a.nbytes) for a in (x, weight, actual))
    try:
        for ptr, a in ((x_ptr, x), (w_ptr, weight), (out_ptr, actual)):
            rt.memcpy(ptr, a.ctypes.data, a.nbytes, Copy.HOST_TO_DEVICE)
        assert narrow_f32_gemm(x_ptr, w_ptr, out_ptr, rows, n, m)
        rt.memcpy(actual.ctypes.data, out_ptr, actual.nbytes, Copy.DEVICE_TO_HOST)
    finally:
        for ptr in (x_ptr, w_ptr, out_ptr):
            rt.free(ptr)
    expected = x.astype(np.float64) @ weight.astype(np.float64).T
    bound = np.abs(x.astype(np.float64)) @ np.abs(weight.astype(np.float64)).T
    # A sum of n products in FP32 is off by a few times 2^-24 of the sum of the magnitudes.
    assert (np.abs(actual[:rows] - expected) <= 8e-7 * bound + 1e-30).all()
    assert np.isnan(actual[rows:]).all()


def test_narrow_f32_gemm_leaves_other_shapes_to_the_caller():
    """One row, another count of outputs, or an input count the step cannot divide: no launch, False."""
    _backend()
    rt = get_hip_runtime()
    page = np.zeros(4096, np.float32)
    ptr = rt.malloc(page.nbytes)
    try:
        rt.memcpy(ptr, page.ctypes.data, page.nbytes, Copy.HOST_TO_DEVICE)
        for rows, n, m in ((1, 4608, 64), (300, 4608, 48), (300, 4608, 256), (300, 100, 64), (300, 2000, 24), (0, 4608, 64)):
            assert not serves(rows, n, m)
            assert not narrow_f32_gemm(ptr, ptr, ptr, rows, n, m)
        rt.memcpy(page.ctypes.data, ptr, page.nbytes, Copy.DEVICE_TO_HOST)
    finally:
        rt.free(ptr)
    assert not page.any()


def _route(owner, rows, outputs, inputs=2304):
    """The launch an F32 projection of this shape takes, with every launch replaced by a mock."""
    # The narrow launch answers as the real one does about the shape.
    launches = {"narrow": Mock(side_effect=lambda src, w, dst, rows, n, m: serves(rows, n, m)),
                "gemv": Mock(), "gemm": Mock()}
    saved = owner._narrow_f32, owner._blas
    # An owner without the kernel keeps going without it.
    owner._narrow_f32 = launches["narrow"] if owner._narrow_f32 is not None else None
    owner._blas = Mock(sgemv_rowmajor_nt=launches["gemv"], sgemm_rowmajor_nt=launches["gemm"])
    try:
        owner._launch_projection(0, 0, rows, 2, (Mock(ptr=1), (outputs, inputs), None), None)
    finally:
        owner._narrow_f32, owner._blas = saved
    # A narrow launch that declines the shape is not a route: what ran last is.
    taken = [name for name in ("gemm", "gemv") if launches[name].called]
    if not taken and launches["narrow"].called:
        assert serves(*launches["narrow"].call_args.args[3:]), "a declined shape must fall through"
        taken = ["narrow"]
    assert len(taken) == 1, taken
    return taken[0]


def test_prompt_rows_take_the_narrow_kernel_where_it_is_faster():
    backend = _backend()
    options = dict(runtime=get_hip_runtime(), blas=Rocblas.load(), backend=backend)
    with (Inkling2GGUFDeviceProjections(Inkling2GGUFWeights(Mock(info=_file(False))), **options) as plain,
          Inkling2GGUFDeviceProjections(Inkling2GGUFWeights(Mock(info=_file(False))), narrow_f32=NARROW_F32_VARIANT,
                                        **options) as narrow):
        assert plain.narrow_f32 is None and plain._narrow_f32 is None
        assert narrow._narrow_f32 is not None
        assert (_NARROW_F32_MIN_ROWS[24], _NARROW_F32_MIN_ROWS[32],
                _NARROW_F32_MIN_ROWS[64], _NARROW_F32_MIN_ROWS[128]) == (2, 2, 9, 9)
        for outputs in (24, 32, 64, 128):
            start = _NARROW_F32_MIN_ROWS[outputs]
            assert _route(narrow, start, outputs) == "narrow"
            assert _route(narrow, 1024, outputs) == "narrow"
            # Below the kernel's rows, one row or a row at a time.
            assert _route(narrow, start - 1, outputs) == "gemv"
            assert _route(plain, 1024, outputs) == "gemm"
            assert _route(plain, 1, outputs) == "gemv"
            assert _route(plain, 9, outputs) == "gemm"
        # An input count the kernel cannot step, or another count of outputs, stays on rocBLAS.
        assert _route(narrow, 1024, 64, inputs=2000) == "gemm"
        assert _route(narrow, 1024, 256) == "gemm"
        # Below the kernel's rows a projection of 64 outputs goes a row at a time: two calls for two rows.
        saved = narrow._narrow_f32, narrow._blas
        narrow._narrow_f32, narrow._blas = Mock(return_value=True), Mock()
        try:
            narrow._launch_projection(0, 0, 3, 2, (Mock(ptr=1), (64, 2304), None), None)
            assert narrow._blas.sgemv_rowmajor_nt.call_count == 3
            narrow._blas.sgemv_rowmajor_nt.assert_any_call(0, 1, 0, in_features=2304, out_features=64)
            narrow._blas.sgemv_rowmajor_nt.reset_mock()
            narrow._launch_projection(0, 0, 3, 2, (Mock(ptr=1), (256, 2304), None), None)
            narrow._blas.sgemv_rowmajor_nt.assert_not_called()
            assert narrow._blas.sgemm_rowmajor_nt.call_count == 1
        finally:
            narrow._narrow_f32, narrow._blas = saved
    with pytest.raises(ValueError, match="unknown Inkling narrow FP32 kernel"):
        Inkling2GGUFDeviceProjections(Inkling2GGUFWeights(Mock(info=_file(False))), narrow_f32="no_such_kernel",
                                      **options)


def _skip_unless_runnable() -> str:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("ROCm runtime is not available")
    model = os.environ.get("HIPENGINE_INKLING2_MINI_GGUF", "")
    if not model or not Path(model).is_file():
        pytest.skip("HIPENGINE_INKLING2_MINI_GGUF must point at the Inkling 2 GGUF model")
    return model


@pytest.mark.parametrize("profile", ["fast", "production", "strict"])
def test_the_model_weights_project_as_rocblas_does(profile):
    """Through the projection owner: the kernel under `fast` and `production`, rocBLAS under `strict`."""
    model = _skip_unless_runnable()
    with closing(LLM(model=model, backend="hip_gfx1100", execution_profile=profile, max_sequence_length=512,
                     prefix_cache="off", speculative_mtp_serving="off")) as llm:
        projection = llm._get_text_generator().model.projection
        assert projection.narrow_f32 == (None if profile == "strict" else NARROW_F32_VARIANT)
        layer = llm._get_text_generator().model.mapping.layers[0]
        served = [t for t in layer.tensors.values() if Q(t.ggml_type) is Q.F32 and len(t.shape) == 2
                  and t.shape[0] in (24, 32, 64, 128) and t.shape[1] % 128 == 0]
        assert served, "the model has F32 projections of few outputs"
        rng = np.random.default_rng(5)
        for tensor in served:
            hidden = rng.normal(0, 0.1, (575, tensor.shape[1])).astype(np.float32)
            through = projection.project(tensor.name, hidden)
            launch = projection._narrow_f32
            projection._narrow_f32 = None
            try:
                blas = projection.project(tensor.name, hidden)
            finally:
                projection._narrow_f32 = launch
            if launch is None:
                np.testing.assert_array_equal(through, blas)
                continue
            # The two differ in the order of 2,304 or 4,608 FP32 sums: a few 2^-24 of their magnitude.
            np.testing.assert_allclose(through, blas, rtol=0, atol=1e-4 * np.sqrt(np.mean(blas.astype(np.float64) ** 2)))
