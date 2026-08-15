"""D08-X2-K5 dense BF16 WMMA bulk prefill correctness fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _hip_available() -> bool:
    import ctypes

    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


requires_rocm = pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP runtime unavailable")

from hipengine.core.hip import get_hip_runtime  # noqa: E402
from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import (  # noqa: E402
    build_dense_gemv,
    dense_prefill_gemm_out_bf16,
    dense_prefill_wmma_out_bf16,
)
from hipengine.loading.materialize import float_array_to_bf16_bits  # noqa: E402
from hipengine.quant.gguf import bf16_to_float32  # noqa: E402

_COMPILER = Path("/tmp/d08-c0/hipcc-version.txt")


def test_dense_prefill_wmma_rejects_unaligned_k_before_launch() -> None:
    """The BK32 kernel cannot safely consume a partial final K tile."""

    with pytest.raises(ValueError, match="in_features % 32"):
        dense_prefill_wmma_out_bf16(
            1,
            2,
            3,
            16,
            1000,
            128,
            library=object(),
            runtime=object(),
        )


@pytest.mark.parametrize(
    ("n", "k", "rows"),
    [
        (128, 256, 16),
        (128, 512, 64),
        (256, 512, 20),   # non-multiple rows guard
        (384, 768, 33),   # multiple column blocks plus odd rows
    ],
)
@requires_rocm
def test_dense_prefill_wmma_matches_reference(n: int, k: int, rows: int) -> None:
    rng = np.random.default_rng(20260815)
    weights = rng.normal(0, 0.08, size=(n, k)).astype(np.float32)
    x = rng.normal(0, 0.35, size=(rows, k)).astype(np.float32)
    expected = x @ weights.T

    runtime = get_hip_runtime()
    compiler = _COMPILER.read_text() if _COMPILER.exists() else None
    library = build_dense_gemv(load=True, compiler_version=compiler)
    weights_bf16 = np.ascontiguousarray(float_array_to_bf16_bits(weights))
    x_bf16 = np.ascontiguousarray(float_array_to_bf16_bits(x))
    host_ref = np.empty((rows, n), dtype=np.uint16)
    host_got = np.empty((rows, n), dtype=np.uint16)
    buffers = []

    def upload(array):
        buf = malloc(array.nbytes, runtime=runtime)
        buffers.append(buf)
        copy_host_to_device(buf, host_array_ptr(array), runtime=runtime)
        return buf

    try:
        weights_dev = upload(weights_bf16)
        x_dev = upload(x_bf16)
        ref_out = malloc(host_ref.nbytes, runtime=runtime)
        got_out = malloc(host_got.nbytes, runtime=runtime)
        buffers.extend((ref_out, got_out))
        dense_prefill_gemm_out_bf16(
            x_dev.ptr, weights_dev.ptr, ref_out.ptr, rows, k, n,
            library=library, runtime=runtime,
        )
        dense_prefill_wmma_out_bf16(
            x_dev.ptr, weights_dev.ptr, got_out.ptr, rows, k, n,
            library=library, runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_ref), ref_out, runtime=runtime)
        copy_device_to_host(host_array_ptr(host_got), got_out, runtime=runtime)
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)

    ref = bf16_to_float32(host_ref)
    got = bf16_to_float32(host_got)
    assert np.isfinite(got).all()
    # f16 WMMA operands round both inputs; scale-aware tolerance matches the
    # accepted quant WMMA prefill class.
    scale = max(float(np.abs(expected).max()), 1e-6)
    assert np.abs(got.astype(np.float64) - ref.astype(np.float64)).max() <= 0.035 * scale
    assert float(np.mean(np.argmax(ref, 1) == np.argmax(got, 1))) >= 0.99
