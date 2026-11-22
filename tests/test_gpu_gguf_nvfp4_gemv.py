import ctypes

import numpy as np
import pytest

from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
from hipengine.kernels.backends import resolve_backend
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data


@pytest.mark.parametrize("batch,columns,nout,outer", [(1, 64, 3, 1.0), (1, 2304, 17, 0.013), (5, 1152, 19, -0.021)])
def test_nvfp4_projects_raw_gguf_rows(batch, columns, nout, outer):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP runtime unavailable: {exc}")
    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    from hipengine.kernels.hip_gfx1100.quant import gguf_nvfp4_gemv  # noqa: F401

    rng = np.random.default_rng(77821)
    blocks = rng.integers(0, 256, (nout, columns // 64, 36), dtype=np.uint8)
    # Cover unusual scale encodings as well as in-distribution finite positives.
    blocks[..., :4] &= 0x7F
    blocks[0, 0, :4] = [0x00, 0x01, 0x7F, 0xFF]
    packed = blocks.reshape(nout, -1)
    x = rng.normal(0, 0.002, (batch, columns)).astype(np.float32)
    actual = np.empty((batch, nout), np.float32)
    launch = resolve(backend=backend, layer="linear", quant="gguf_nvfp4", variant="gemv_f32_f32_out")
    runtime = get_hip_runtime()
    pointers = []
    try:
        for arr in (x, packed, actual):
            pointers.append(runtime.malloc(arr.nbytes))
            runtime.memcpy(pointers[-1], arr.ctypes.data, arr.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        launch(*pointers, batch, columns, nout, outer_scale=outer)
        runtime.memcpy(actual.ctypes.data, pointers[-1], actual.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
    finally:
        for ptr in reversed(pointers):
            runtime.free(ptr)

    dense = dequantize_gguf_data(packed, GGMLQuantizationType.NVFP4).astype(np.float64)
    expected = (x.astype(np.float64) @ dense.T) * np.float32(outer)
    abs_products = (np.abs(x.astype(np.float64)) @ np.abs(dense).T) * abs(np.float32(outer))
    # Two FMAs per packed block in a lane, then five wave additions and the outer scale.
    operations = 2 * (columns // 64) + 5 + 1
    eps = np.finfo(np.float32).eps / 2
    gamma = operations * eps / (1 - operations * eps)
    assert np.all(np.abs(actual - expected) <= gamma * abs_products)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))

    def log_probs(logits):
        shifted = logits - logits.max(axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    ref_logp = log_probs(expected)
    gpu_logp = log_probs(actual.astype(np.float64))
    assert np.max((np.exp(ref_logp) * (ref_logp - gpu_logp)).sum(axis=-1)) <= 0.05
