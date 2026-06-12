"""GPU correctness tests for the R3.4 WMMA-tiled DFlash drafter dense kernels.

These tests require a working ROCm/HIP gfx1100 device and are skipped when the
runtime cannot load.  They compare the WMMA kernel results against the naive
kernel and a CPU-reference matmul on the actual drafter dense shapes.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest


def _has_gfx1100() -> bool:
    try:  # noqa: SIM105 - explicit fallback
        from hipengine.core.hip import get_hip_runtime  # type: ignore
    except Exception:
        return False
    try:
        get_hip_runtime()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gfx1100(), reason="gfx1100 HIP runtime not available")


def _to_bf16_bits(x_f32: np.ndarray) -> np.ndarray:
    bits = x_f32.view(np.uint32)
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16).astype(np.uint16)


def _from_bf16_bits(x_u16: np.ndarray) -> np.ndarray:
    return (x_u16.astype(np.uint32) << 16).view(np.float32)


def _cpu_reference(x_bf16: np.ndarray, w_bf16: np.ndarray) -> np.ndarray:
    """``out[m, n] = sum_k bf16(x[m,k]) * bf16(weight[n,k])`` in FP32."""
    x_f32 = _from_bf16_bits(x_bf16).astype(np.float32)
    w_f32 = _from_bf16_bits(w_bf16).astype(np.float32)
    return (x_f32 @ w_f32.T).astype(np.float32)


@pytest.fixture(scope="module")
def _drafter_lib():
    from hipengine.kernels.backends import hip_target_arch_environment
    from hipengine.kernels.hip_gfx1100.speculative import build_dflash_drafter

    compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = pathlib.Path(compiler_file).read_text(encoding="utf-8") if compiler_file else None
    with hip_target_arch_environment("gfx1100"):
        return build_dflash_drafter(load=True, compiler_version=compiler_version)


@pytest.fixture(scope="module")
def _runtime():
    from hipengine.core.hip import get_hip_runtime
    return get_hip_runtime()


def _upload(runtime, bufs, array):
    from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc
    arr = np.ascontiguousarray(array)
    buf = malloc(arr.nbytes, runtime=runtime)
    bufs.append(buf)
    copy_host_to_device(buf, host_array_ptr(arr), runtime=runtime)
    return buf


def _download(runtime, buf, shape, dtype):
    from hipengine.core.memory import copy_device_to_host, host_array_ptr
    arr = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(arr), buf, runtime=runtime)
    return arr


_SHAPES = (
    # Drafter dense ops at the actual z-lab Qwen3.6 shape mix.
    (16, 2048, 2048),    # Q / O
    (16, 2048, 512),     # K / V
    (16, 2048, 6144),    # gate / up
    (16, 6144, 2048),    # down
    # Tail/edge shapes for bounds-check coverage.
    (8, 32, 16),
    (24, 48, 17),
    (16, 2048, 248320),  # vocab
)


def _free_all(runtime, bufs):
    from hipengine.core.memory import free
    for buf in reversed(bufs):
        free(buf, runtime=runtime)


@pytest.mark.parametrize("rows,in_features,out_features", _SHAPES)
def test_dflash_dense_bf16_to_f32_wmma_matches_cpu_reference(_drafter_lib, _runtime, rows, in_features, out_features):
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
        dflash_dense_bf16_to_f32,
        dflash_dense_bf16_to_f32_wmma,
    )

    if (in_features % 16) != 0:
        pytest.skip("WMMA path requires in_features % 16 == 0")
    rng = np.random.default_rng(0xD41A5)
    x_f32 = (rng.standard_normal((rows, in_features), dtype=np.float32) * 0.05)
    w_f32 = (rng.standard_normal((out_features, in_features), dtype=np.float32) * 0.02)
    x_bf = _to_bf16_bits(x_f32)
    w_bf = _to_bf16_bits(w_f32)
    bufs = []
    try:
        x_dev = _upload(_runtime, bufs, x_bf)
        w_dev = _upload(_runtime, bufs, w_bf)
        out_naive = _upload(_runtime, bufs, np.zeros((rows, out_features), np.float32))
        out_wmma = _upload(_runtime, bufs, np.zeros((rows, out_features), np.float32))
        dflash_dense_bf16_to_f32(x_dev.ptr, w_dev.ptr, out_naive.ptr, rows, in_features, out_features, library=_drafter_lib, runtime=_runtime)
        dflash_dense_bf16_to_f32_wmma(x_dev.ptr, w_dev.ptr, out_wmma.ptr, rows, in_features, out_features, library=_drafter_lib, runtime=_runtime)
        _runtime.device_synchronize()
        naive = _download(_runtime, out_naive, (rows, out_features), np.float32)
        wmma = _download(_runtime, out_wmma, (rows, out_features), np.float32)
        cpu = _cpu_reference(x_bf, w_bf)
        # WMMA and naive both accumulate FP32 and read BF16 inputs; the only
        # difference is reduction order. Tolerate a tiny relative gap from FP32
        # reorder vs the CPU path.
        scale = max(float(np.max(np.abs(cpu))), 1.0)
        np.testing.assert_allclose(wmma, naive, atol=1e-4 * scale, rtol=1e-4)
        np.testing.assert_allclose(wmma, cpu, atol=1e-4 * scale, rtol=1e-4)
    finally:
        _free_all(_runtime, bufs)


@pytest.mark.parametrize("rows,in_features,out_features", _SHAPES)
def test_dflash_dense_bf16_to_bf16_wmma_matches_naive(_drafter_lib, _runtime, rows, in_features, out_features):
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
        dflash_dense_bf16_to_bf16,
        dflash_dense_bf16_to_bf16_wmma,
    )

    if (in_features % 16) != 0:
        pytest.skip("WMMA path requires in_features % 16 == 0")
    rng = np.random.default_rng(0xD41A6)
    x_f32 = (rng.standard_normal((rows, in_features), dtype=np.float32) * 0.05)
    w_f32 = (rng.standard_normal((out_features, in_features), dtype=np.float32) * 0.02)
    x_bf = _to_bf16_bits(x_f32)
    w_bf = _to_bf16_bits(w_f32)
    bufs = []
    try:
        x_dev = _upload(_runtime, bufs, x_bf)
        w_dev = _upload(_runtime, bufs, w_bf)
        out_naive = _upload(_runtime, bufs, np.zeros((rows, out_features), np.uint16))
        out_wmma = _upload(_runtime, bufs, np.zeros((rows, out_features), np.uint16))
        dflash_dense_bf16_to_bf16(x_dev.ptr, w_dev.ptr, out_naive.ptr, rows, in_features, out_features, library=_drafter_lib, runtime=_runtime)
        dflash_dense_bf16_to_bf16_wmma(x_dev.ptr, w_dev.ptr, out_wmma.ptr, rows, in_features, out_features, library=_drafter_lib, runtime=_runtime)
        _runtime.device_synchronize()
        naive = _from_bf16_bits(_download(_runtime, out_naive, (rows, out_features), np.uint16))
        wmma = _from_bf16_bits(_download(_runtime, out_wmma, (rows, out_features), np.uint16))
        cpu = _cpu_reference(x_bf, w_bf)
        # BF16 storage round-off dominates; allow up to half a BF16 ULP near zero.
        scale = max(float(np.max(np.abs(cpu))), 1.0)
        np.testing.assert_allclose(wmma, naive, atol=2e-3 * scale, rtol=2e-3)
        np.testing.assert_allclose(wmma, cpu, atol=2e-3 * scale, rtol=2e-3)
    finally:
        _free_all(_runtime, bufs)


def test_dflash_dense_wmma_rejects_unsupported_k(_drafter_lib, _runtime):
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import dflash_dense_bf16_to_bf16_wmma

    with pytest.raises(ValueError, match="multiple of 16"):
        dflash_dense_bf16_to_bf16_wmma(0, 0, 0, 16, 17, 16, library=_drafter_lib, runtime=_runtime)


def test_dflash_dense_dispatch_env(monkeypatch, _drafter_lib, _runtime):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
        _drafter_dense_use_wmma,
        dflash_dense_bf16_to_bf16,
    )

    monkeypatch.setenv("HIPENGINE_DFLASH_DRAFTER_DENSE", "naive")
    assert _drafter_dense_use_wmma() is False
    monkeypatch.setenv("HIPENGINE_DFLASH_DRAFTER_DENSE", "wmma")
    assert _drafter_dense_use_wmma() is True
    monkeypatch.setenv("HIPENGINE_DFLASH_DRAFTER_DENSE", "bogus")
    with pytest.raises(ValueError):
        dflash_dense_bf16_to_bf16(0, 0, 0, 16, 32, 16, library=_drafter_lib, runtime=get_hip_runtime())


def test_dflash_dense_bf16_to_bf16_expert_routes_matches_scalar(_drafter_lib, _runtime):
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
        dflash_dense_bf16_to_bf16_expert,
        dflash_dense_bf16_to_bf16_expert_routes,
    )

    rng = np.random.default_rng(0xD41A7)
    routes = 4
    rows = 1
    in_features = 32
    out_features = 17
    num_experts = 6
    expert_ids = np.array([3, 1, 5, 0], dtype=np.int32)
    x_shared = _to_bf16_bits(rng.standard_normal((rows, in_features), dtype=np.float32) * 0.05)
    x_routes = _to_bf16_bits(rng.standard_normal((routes, rows, in_features), dtype=np.float32) * 0.05)
    weights = _to_bf16_bits(rng.standard_normal((num_experts, out_features, in_features), dtype=np.float32) * 0.02)
    scalar_shared = np.zeros((routes, rows, out_features), dtype=np.uint16)
    batch_shared = np.zeros_like(scalar_shared)
    scalar_routed = np.zeros_like(scalar_shared)
    batch_routed = np.zeros_like(scalar_shared)
    bytes_per_out = rows * out_features * np.dtype(np.uint16).itemsize
    bytes_per_x = rows * in_features * np.dtype(np.uint16).itemsize
    bufs = []
    try:
        x_shared_dev = _upload(_runtime, bufs, x_shared)
        x_routes_dev = _upload(_runtime, bufs, x_routes)
        weights_dev = _upload(_runtime, bufs, weights)
        expert_ids_dev = _upload(_runtime, bufs, expert_ids)
        scalar_shared_dev = _upload(_runtime, bufs, scalar_shared)
        batch_shared_dev = _upload(_runtime, bufs, batch_shared)
        scalar_routed_dev = _upload(_runtime, bufs, scalar_routed)
        batch_routed_dev = _upload(_runtime, bufs, batch_routed)
        expert_stride = out_features * in_features
        for route in range(routes):
            dflash_dense_bf16_to_bf16_expert(
                x_shared_dev.ptr,
                weights_dev.ptr,
                expert_ids_dev.ptr,
                scalar_shared_dev.ptr + route * bytes_per_out,
                route,
                expert_stride,
                rows,
                in_features,
                out_features,
                threads=64,
                library=_drafter_lib,
                runtime=_runtime,
            )
            dflash_dense_bf16_to_bf16_expert(
                x_routes_dev.ptr + route * bytes_per_x,
                weights_dev.ptr,
                expert_ids_dev.ptr,
                scalar_routed_dev.ptr + route * bytes_per_out,
                route,
                expert_stride,
                rows,
                in_features,
                out_features,
                threads=64,
                library=_drafter_lib,
                runtime=_runtime,
            )
        dflash_dense_bf16_to_bf16_expert_routes(
            x_shared_dev.ptr,
            weights_dev.ptr,
            expert_ids_dev.ptr,
            batch_shared_dev.ptr,
            routes,
            0,
            expert_stride,
            rows,
            in_features,
            out_features,
            threads=64,
            library=_drafter_lib,
            runtime=_runtime,
        )
        dflash_dense_bf16_to_bf16_expert_routes(
            x_routes_dev.ptr,
            weights_dev.ptr,
            expert_ids_dev.ptr,
            batch_routed_dev.ptr,
            routes,
            rows * in_features,
            expert_stride,
            rows,
            in_features,
            out_features,
            threads=64,
            library=_drafter_lib,
            runtime=_runtime,
        )
        _runtime.device_synchronize()
        scalar_shared = _download(_runtime, scalar_shared_dev, scalar_shared.shape, np.uint16)
        batch_shared = _download(_runtime, batch_shared_dev, batch_shared.shape, np.uint16)
        scalar_routed = _download(_runtime, scalar_routed_dev, scalar_routed.shape, np.uint16)
        batch_routed = _download(_runtime, batch_routed_dev, batch_routed.shape, np.uint16)
    finally:
        _free_all(_runtime, bufs)

    assert np.array_equal(batch_shared, scalar_shared)
    assert np.array_equal(batch_routed, scalar_routed)


def test_mtp_shared_gate_up_dual_wmma_matches_two_dflash_dense_calls(_drafter_lib, _runtime):
    from hipengine.kernels.hip_gfx1100.linear.dense_gemv import build_dense_gemv, dense_dual_gemv_out_bf16_wmma
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import dflash_dense_bf16_to_bf16

    rng = np.random.default_rng(0xD41A9)
    rows = 1
    in_features = 2048
    out_a = 4096
    out_b = 4096
    x = _to_bf16_bits(rng.standard_normal((rows, in_features), dtype=np.float32) * 0.05)
    weight_a = _to_bf16_bits(rng.standard_normal((out_a, in_features), dtype=np.float32) * 0.02)
    weight_b = _to_bf16_bits(rng.standard_normal((out_b, in_features), dtype=np.float32) * 0.02)
    out_a_ref = np.zeros((rows, out_a), dtype=np.uint16)
    out_b_ref = np.zeros((rows, out_b), dtype=np.uint16)
    out_dual = np.zeros((rows, out_a + out_b), dtype=np.uint16)
    bufs = []
    try:
        x_dev = _upload(_runtime, bufs, x)
        weight_a_dev = _upload(_runtime, bufs, weight_a)
        weight_b_dev = _upload(_runtime, bufs, weight_b)
        out_a_dev = _upload(_runtime, bufs, out_a_ref)
        out_b_dev = _upload(_runtime, bufs, out_b_ref)
        out_dual_dev = _upload(_runtime, bufs, out_dual)
        dense_lib = build_dense_gemv(load=True)
        dflash_dense_bf16_to_bf16(x_dev.ptr, weight_a_dev.ptr, out_a_dev.ptr, rows, in_features, out_a, library=_drafter_lib, runtime=_runtime)
        dflash_dense_bf16_to_bf16(x_dev.ptr, weight_b_dev.ptr, out_b_dev.ptr, rows, in_features, out_b, library=_drafter_lib, runtime=_runtime)
        dense_dual_gemv_out_bf16_wmma(
            x_dev.ptr,
            weight_a_dev.ptr,
            weight_b_dev.ptr,
            out_dual_dev.ptr,
            rows,
            in_features,
            out_a,
            out_b,
            library=dense_lib,
            runtime=_runtime,
        )
        _runtime.device_synchronize()
        out_a_ref = _download(_runtime, out_a_dev, (rows, out_a), np.uint16)
        out_b_ref = _download(_runtime, out_b_dev, (rows, out_b), np.uint16)
        out_dual = _download(_runtime, out_dual_dev, (rows, out_a + out_b), np.uint16)
    finally:
        _free_all(_runtime, bufs)

    assert np.array_equal(out_dual, np.concatenate([out_a_ref, out_b_ref], axis=1))


def test_dflash_silu_mul_gate_up_routes_matches_scalar(_drafter_lib, _runtime):
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
        dflash_silu_mul_bf16,
        dflash_silu_mul_gate_up_routes_bf16,
    )

    rng = np.random.default_rng(0xD41A8)
    routes = 4
    features = 19
    gate_up = _to_bf16_bits(rng.standard_normal((routes, 2 * features), dtype=np.float32) * 0.5)
    scalar = np.zeros((routes, features), dtype=np.uint16)
    batched = np.zeros_like(scalar)
    bytes_per_gate_up = 2 * features * np.dtype(np.uint16).itemsize
    bytes_per_out = features * np.dtype(np.uint16).itemsize
    bufs = []
    try:
        gate_up_dev = _upload(_runtime, bufs, gate_up)
        scalar_dev = _upload(_runtime, bufs, scalar)
        batched_dev = _upload(_runtime, bufs, batched)
        for route in range(routes):
            base = gate_up_dev.ptr + route * bytes_per_gate_up
            dflash_silu_mul_bf16(
                base,
                base + features * np.dtype(np.uint16).itemsize,
                scalar_dev.ptr + route * bytes_per_out,
                features,
                threads=64,
                library=_drafter_lib,
                runtime=_runtime,
            )
        dflash_silu_mul_gate_up_routes_bf16(
            gate_up_dev.ptr,
            batched_dev.ptr,
            routes,
            features,
            threads=64,
            library=_drafter_lib,
            runtime=_runtime,
        )
        _runtime.device_synchronize()
        scalar = _download(_runtime, scalar_dev, scalar.shape, np.uint16)
        batched = _download(_runtime, batched_dev, batched.shape, np.uint16)
    finally:
        _free_all(_runtime, bufs)

    assert np.array_equal(batched, scalar)
