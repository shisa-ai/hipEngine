"""B3 foundation: validate the PARO oracle primitives (AWQ pack8 GEMV, rotate1)
against the deployed GPU kernels.

These cpu_reference primitives are the building blocks for the PARO selected-FFN
oracle that the B3 fused megakernel will be gated against (KL<=0.05). They
reproduce the AWQ W4 pack8 dequant (``(q-z)*scale``) and the PARO single-output
incoherence rotation (per-group Givens rounds) exactly as the HIP kernels do;
this test pins them to ~1 bf16 ULP vs the GPU.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.kernels.cpu_reference import awq_pack8_gemv_transposed, paro_rotate1


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


def _f32_to_bf16_u16(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, np.float32); u = a.view(np.uint32).copy(); lsb = (u >> 16) & 1
    return (((u + 0x7FFF + lsb) >> 16).astype(np.uint16)).reshape(a.shape)


def _bf16_u16_to_f32(a: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(a, np.uint16); return (u.astype(np.uint32) << 16).view(np.float32).reshape(u.shape).copy()


def test_paro_rotate1_numpy_is_self_consistent():
    """Pure-numpy sanity: rotate1 is row-independent and orthogonal-norm-preserving
    when scales==1 and pairs form a matching."""

    hidden, gs, krot = 256, 128, 2
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((4, hidden)) * 0.1).astype(np.float32)
    half = gs // 2
    pairs = np.zeros((krot, hidden), np.int64)
    for r in range(krot):
        for g in range(hidden // gs):
            for lane in range(half):
                pairs[r, g * gs + 2 * lane] = 2 * lane
                pairs[r, g * gs + 2 * lane + 1] = 2 * lane + 1
    theta = rng.uniform(-1, 1, (krot, hidden // 2)).astype(np.float32)
    scales = np.ones(hidden, np.float32)
    out = paro_rotate1(x, pairs, theta, scales, gs, krot)
    # Givens rotations are orthogonal -> per-group L2 norm preserved.
    for g in range(hidden // gs):
        sl = slice(g * gs, (g + 1) * gs)
        np.testing.assert_allclose(np.linalg.norm(out[:, sl], axis=1), np.linalg.norm(x[:, sl], axis=1), rtol=1e-5)
    # Row-independent (the T1 row-invariance property at the primitive level).
    np.testing.assert_array_equal(paro_rotate1(x[:1], pairs, theta, scales, gs, krot)[0], out[0])


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_paro_rotate1_matches_gpu():
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.rotary.paro_rotate import build_paro_rotate, paro_rotate1_bf16

    build_paro_rotate(load=True)
    rng = np.random.default_rng(1); gs, hidden, krot, tokens = 128, 256, 2, 4
    half = gs // 2
    x = _f32_to_bf16_u16((rng.standard_normal((tokens, hidden)) * 0.1).astype(np.float32))
    pairs = np.zeros((krot, hidden), np.int16)
    for r in range(krot):
        for g in range(hidden // gs):
            for lane in range(half):
                pairs[r, g * gs + 2 * lane] = 2 * lane
                pairs[r, g * gs + 2 * lane + 1] = 2 * lane + 1
    theta = _f32_to_bf16_u16(rng.uniform(-1, 1, (krot, hidden // 2)).astype(np.float32))
    scales = _f32_to_bf16_u16(rng.uniform(0.5, 1.5, hidden).astype(np.float32))
    bufs = []
    def up(a):
        a = np.ascontiguousarray(a); b = malloc(a.nbytes); copy_host_to_device(b, host_array_ptr(a), a.nbytes); bufs.append(b); return b
    try:
        xb = up(x); ob = malloc(x.nbytes); bufs.append(ob)
        paro_rotate1_bf16(xb.ptr, ob.ptr, up(pairs).ptr, up(theta).ptr, up(scales).ptr, tokens, hidden, gs, krot)
        gpu = np.empty((tokens, hidden), np.uint16); copy_device_to_host(host_array_ptr(gpu), ob, gpu.nbytes)
    finally:
        for b in bufs:
            free(b)
    npr = paro_rotate1(_bf16_u16_to_f32(x), pairs, _bf16_u16_to_f32(theta), _bf16_u16_to_f32(scales), gs, krot)
    max_rel = float(np.max(np.abs(_bf16_u16_to_f32(gpu) - npr) / np.maximum(np.abs(npr), 1e-3)))
    assert max_rel < 1e-2, f"rotate1 numpy vs GPU max_rel={max_rel}"


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_awq_pack8_gemv_transposed_matches_gpu():
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import build_paro_awq_gemv, gemv_awq_pack8_transposed_bf16

    build_paro_awq_gemv(load=True)
    rng = np.random.default_rng(2); in_f, out_f, gs, tokens = 256, 64, 128, 4
    out_packed = out_f // 8; groups = in_f // gs
    x = _f32_to_bf16_u16((rng.standard_normal((tokens, in_f)) * 0.05).astype(np.float32))
    qweight = rng.integers(0, 2**32, size=(out_packed, in_f), dtype=np.uint64).astype(np.uint32).view(np.int32)
    qzeros = rng.integers(0, 2**32, size=(groups, out_packed), dtype=np.uint64).astype(np.uint32).view(np.int32)
    scales = _f32_to_bf16_u16(rng.uniform(0.001, 0.04, (groups, out_f)).astype(np.float32))
    bufs = []
    def up(a):
        a = np.ascontiguousarray(a); b = malloc(a.nbytes); copy_host_to_device(b, host_array_ptr(a), a.nbytes); bufs.append(b); return b
    try:
        xb = up(x); qwb = up(qweight); qzb = up(qzeros); sb = up(scales)
        ob = malloc(tokens * out_f * 2); bufs.append(ob)
        gemv_awq_pack8_transposed_bf16(xb.ptr, qwb.ptr, qzb.ptr, sb.ptr, ob.ptr, tokens, in_f, out_packed, gs)
        gpu = np.empty((tokens, out_f), np.uint16); copy_device_to_host(host_array_ptr(gpu), ob, gpu.nbytes)
    finally:
        for b in bufs:
            free(b)
    npy = awq_pack8_gemv_transposed(_bf16_u16_to_f32(x), qweight, qzeros, _bf16_u16_to_f32(scales), in_f, out_f, gs)
    max_rel = float(np.max(np.abs(_bf16_u16_to_f32(gpu) - npy) / np.maximum(np.abs(npy), 1e-2)))
    assert max_rel < 1e-2, f"awq gemv numpy vs GPU max_rel={max_rel}"
