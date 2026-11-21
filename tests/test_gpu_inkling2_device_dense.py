import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.loading.inkling2_device_dense import Inkling2DeviceDense


def test_device_dense_vector_and_block_match_cpu():
    try:
        ctypes.CDLL("libamdhip64.so")
        ctypes.CDLL("librocblas.so")
    except OSError as exc:
        pytest.skip(f"HIP or rocBLAS unavailable: {exc}")
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.rocblas import Rocblas
    from hipengine.kernels.backends import resolve_backend

    if resolve_backend("auto", execution_device=True) != "hip_gfx1100":
        pytest.skip("test requires gfx1100")

    rng = np.random.default_rng(15)
    matrix = rng.standard_normal((19, 32), dtype=np.float32)
    rows = rng.standard_normal((4, 32), dtype=np.float32)
    reads = []

    def dense(name, selector):
        reads.append(name)
        return matrix[selector]

    blas = Rocblas.load()
    try:
        with Inkling2DeviceDense(SimpleNamespace(dense=dense), runtime=get_hip_runtime(), blas=blas) as owner:
            np.testing.assert_allclose(owner.project("head", rows[0], limit=12), matrix[:12] @ rows[0], atol=1e-5)
            np.testing.assert_allclose(owner.project("head", rows, limit=12), rows @ matrix[:12].T, atol=1e-5)
            assert reads == ["head"]
        with pytest.raises(RuntimeError, match="closed"):
            owner.project("head", rows)
    finally:
        blas.close()
