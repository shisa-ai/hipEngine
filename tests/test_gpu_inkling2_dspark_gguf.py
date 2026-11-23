import ctypes
from types import SimpleNamespace

import numpy as np
import pytest


def test_inkling_dspark_keeps_q8_projections_packed_and_releases_them():
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.core.rocblas import Rocblas
    from hipengine.kernels.backends import resolve_backend
    from hipengine.loading.inkling2_dspark_gguf_device import Inkling2DSparkDeviceProjections
    from hipengine.quant.gguf import GGMLQuantizationType as Q, dequantize_gguf_data

    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    rng = np.random.default_rng(41)
    q8 = rng.integers(0, 256, (48, 34), dtype=np.uint8)
    q8[:, :2] = np.array([0.0078125], dtype="<f2").view(np.uint8)
    dense = rng.normal(size=(12, 32)).astype(np.float32)
    weights = SimpleNamespace(names={
        "ctx": SimpleNamespace(ggml_type=Q.Q8_0, shape=(48, 32)),
        "recurrent": SimpleNamespace(ggml_type=Q.F32, shape=dense.shape),
    }, raw=lambda name: q8 if name == "ctx" else pytest.fail(name),
       dense=lambda name, rows: dense[rows] if name == "recurrent" else pytest.fail("expanded a packed matrix"))
    unpacked = dequantize_gguf_data(q8, Q.Q8_0)
    start = memory_stats()["current_allocated_bytes"]
    blas = Rocblas.load()
    try:
        with Inkling2DSparkDeviceProjections(weights, runtime=get_hip_runtime(), blas=blas, backend=backend) as owner:
            for shape in ((32,), (4, 32)):
                x = rng.normal(size=shape).astype(np.float32)
                np.testing.assert_allclose(owner.project("ctx", x), x @ unpacked.T, rtol=2e-5, atol=3e-6)
                np.testing.assert_allclose(owner.project("recurrent", x), x @ dense.T, rtol=2e-5, atol=3e-6)
            ptr, shape, packed = owner._matrix("ctx", None)
            assert shape == (48, 32) and packed == Q.Q8_0 and ptr.nbytes == q8.nbytes
            assert owner._matrix("ctx", None)[0] is ptr
    finally:
        blas.close()
    assert memory_stats()["current_allocated_bytes"] == start
