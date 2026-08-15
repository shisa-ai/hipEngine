"""D08-X2-K1 pack8 wmma64 large-tile prefill correctness fixtures."""

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
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill import (  # noqa: E402
    build_gguf_q4_k_prefill,
    gguf_q4_k_pack8_wmma64_prefill_bf16_bf16_out,
)
from hipengine.loading.materialize import float_array_to_bf16_bits  # noqa: E402
from hipengine.quant.gguf import bf16_to_float32  # noqa: E402

_COMPILER = Path("/tmp/d08-c0/hipcc-version.txt")


def _pack8_shift(i: int) -> int:
    """Mirror of the device helper pack8_shift_for_lane for lane i in 0..7."""

    pos = (4 + (i >> 1)) if (i & 1) else (i >> 1)
    return pos * 4


_SHIFTS = np.array([_pack8_shift(i) for i in range(8)], dtype=np.uint32)


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
def test_pack8_wmma64_matches_cpu_reference(n: int, k: int, rows: int) -> None:
    rng = np.random.default_rng(20260815)
    q = rng.integers(0, 16, size=(n, k), dtype=np.uint32)
    scale_plane = rng.uniform(0.001, 0.02, size=(k // 32, n)).astype(np.float32)
    min_plane = rng.uniform(0.0, 0.004, size=(k // 32, n)).astype(np.float32)
    groups = np.repeat(np.arange(k // 32, dtype=np.int64), 32)
    weights = (scale_plane[groups, :] * q.T - min_plane[groups, :]).astype(np.float32)

    x_f32 = rng.normal(0.0, 0.35, size=(rows, k)).astype(np.float32)
    expected = x_f32 @ weights

    qw = np.zeros((n // 8, k), dtype=np.uint32)
    for c in range(n):
        qw[c >> 3, :] |= q[c] << int(_SHIFTS[c & 7])

    runtime = get_hip_runtime()
    compiler = _COMPILER.read_text() if _COMPILER.exists() else None
    lib = build_gguf_q4_k_prefill(load=True, compiler_version=compiler)
    x_bf16 = np.ascontiguousarray(float_array_to_bf16_bits(x_f32))
    qw_dev_holder = np.ascontiguousarray(qw.astype(np.int32))
    sc_dev_holder = np.ascontiguousarray(scale_plane.reshape(-1))
    mn_dev_holder = np.ascontiguousarray(min_plane.reshape(-1))
    host_out = np.empty((rows, n), dtype=np.uint16)
    buffers = []

    def upload(arr):
        buf = malloc(arr.nbytes, runtime=runtime)
        buffers.append(buf)
        copy_host_to_device(buf, host_array_ptr(arr), runtime=runtime)
        return buf

    try:
        x_dev = upload(x_bf16)
        qweight = upload(qw_dev_holder)
        scales = upload(sc_dev_holder)
        mins = upload(mn_dev_holder)
        out = malloc(host_out.nbytes, runtime=runtime)
        buffers.append(out)
        gguf_q4_k_pack8_wmma64_prefill_bf16_bf16_out(
            x_dev.ptr,
            qweight.ptr,
            scales.ptr,
            mins.ptr,
            out.ptr,
            rows,
            k,
            n,
            library=lib,
            runtime=runtime,
        )
        runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(host_out), out, runtime=runtime)
    finally:
        for buf in buffers:
            free(buf, runtime=runtime)

    got = bf16_to_float32(host_out)
    assert np.isfinite(got).all()
    delta = np.max(np.abs(got.astype(np.float64) - expected.astype(np.float64)))
    scale = max(float(np.abs(expected).max()), 1e-6)
    # f16 WMMA operands plus BF16 output rounding: same tolerance class as the
    # accepted small-tile pack8 WMMA leaf.
    assert delta <= 0.0078125 * scale, delta
    top1_expected = np.argmax(expected, axis=1)
    top1_got = np.argmax(got, axis=1)
    assert float(np.mean(top1_expected == top1_got)) >= 0.9
