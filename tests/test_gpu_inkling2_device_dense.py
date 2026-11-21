import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.loading.inkling2_device_dense import Inkling2DeviceDense


def test_device_dense_vector_and_block_match_cpu(monkeypatch):
    try:
        ctypes.CDLL("libamdhip64.so")
        ctypes.CDLL("librocblas.so")
    except OSError as exc:
        pytest.skip(f"HIP or rocBLAS unavailable: {exc}")
    from hipengine.core.hip import HipError, get_hip_runtime
    from hipengine.core.rocblas import Rocblas
    from hipengine.kernels.backends import resolve_backend
    from hipengine.kernels.hip_gfx1100.linear.dense_gemv import dense_gemv_f32_bf16w_f32_out

    if resolve_backend("auto", execution_device=True) != "hip_gfx1100":
        pytest.skip("test requires gfx1100")

    rng = np.random.default_rng(15)
    matrix = rng.standard_normal((19, 32), dtype=np.float32)
    rows = rng.standard_normal((4, 32), dtype=np.float32)
    reads = []

    def dense(name, selector):
        reads.append(name)
        return matrix[selector] if name in ("head", "other") else matrix[:7][selector]

    blas = Rocblas.load()
    try:
        bits = (matrix.view(np.uint32) >> 16).astype(np.uint16)
        widened = (bits.astype(np.uint32) << 16).view(np.float32)
        weights = SimpleNamespace(dense=dense, native_bf16=lambda name, selector: bits[selector] if name == "head" else None)
        with Inkling2DeviceDense(weights, runtime=get_hip_runtime(), blas=blas) as owner:
            np.testing.assert_allclose(owner.project("head", rows[0], limit=12), matrix[:12] @ rows[0], atol=1e-5)
            np.testing.assert_allclose(owner.project("head", rows, limit=12), rows @ matrix[:12].T, atol=1e-5)
            assert reads == ["head"]
            for x in (rows[0], rows):
                native = owner.project("head", x, limit=12, native_bf16=dense_gemv_f32_bf16w_f32_out)
                np.testing.assert_allclose(native, x @ widened[:12].T, atol=1e-5)
                # Native reads must not expand the original bits when the strict FP32 copy already exists.
                assert reads == ["head"]
            np.testing.assert_array_equal(owner.project("other", rows, native_bf16=dense_gemv_f32_bf16w_f32_out),
                                          owner.project("other", rows))
            for x in (rows[0], rows):
                names = ("head", "small", "other", "smaller")
                expected = tuple(owner.project(name, x) for name in names)
                actual = owner.project_many(names, x)
                for paired, single in zip(actual, expected):
                    np.testing.assert_array_equal(paired.view(np.uint32), single.view(np.uint32))
                launch = dense_gemv_f32_bf16w_f32_out
                native = owner.project_many(names, x, native_bf16=launch, pinned=True)
                assert owner._transfer.host_input is not None and not owner._transfer.pin_unavailable
                for name, paired in zip(names, native, strict=True):
                    single = owner.project(name, x, native_bf16=launch)
                    pinned = owner.project(name, x, native_bf16=launch, pinned=True)
                    np.testing.assert_array_equal(paired.view(np.uint32), single.view(np.uint32))
                    np.testing.assert_array_equal(pinned.view(np.uint32), single.view(np.uint32))
            launch = dense_gemv_f32_bf16w_f32_out
            groups = ((("head", "small"), rows[::-1]), (("other",), rows[0]), (("smaller",), rows[:2]))
            for native in (None, launch):
                bundled = owner.project_groups(groups, native_bf16=native, pinned=True)
                for (names, x), actual in zip(groups, bundled, strict=True):
                    expected = owner.project_many(names, x, native_bf16=native, pinned=True)
                    for a, b in zip(actual, expected, strict=True):
                        np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
        assert owner._transfer.host_input is owner._transfer.host_output is None
        with pytest.raises(RuntimeError, match="closed"):
            owner.project("head", rows)

        def unavailable(*args, **kwargs):
            raise HipError(801, "host pinning unavailable")

        runtime = get_hip_runtime()
        monkeypatch.setattr(type(runtime), "host_register", unavailable)
        with Inkling2DeviceDense(weights, runtime=runtime, blas=blas) as owner:
            fallback = owner.project("head", rows, native_bf16=launch, pinned=True)
            assert owner._transfer.pin_unavailable and owner._transfer.host_input is None
            np.testing.assert_array_equal(fallback, owner.project("head", rows, native_bf16=launch))
    finally:
        blas.close()
