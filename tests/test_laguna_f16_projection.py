from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.registry import resolve
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def test_laguna_f16_projection_registry_resolves_all_mixed_variants() -> None:
    from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
        laguna_f16w_gemv_bf16_f32_out,
        laguna_f16w_triple_gemv_bf16_f32_out,
    )

    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="fp16_weight",
            variant="bf16_f32_out",
        )
        is laguna_f16w_gemv_bf16_f32_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear_triple",
            quant="fp16_weight",
            variant="bf16_f32_out",
        )
        is laguna_f16w_triple_gemv_bf16_f32_out
    )


def test_laguna_f16_projection_runtime_uses_resident_weight_abi() -> None:
    from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
        register_laguna_f16_projection_kernels,
    )
    from hipengine.runtime.f16_weight_linear import launch_f16_weight_linear_triple

    register_laguna_f16_projection_kernels()
    key = ("hip_gfx1100", "linear_triple", "fp16_weight", "bf16_f32_out")
    original = resolve(backend=key[0], layer=key[1], quant=key[2], variant=key[3])
    calls = []

    def fake_kernel(*args, **kwargs):
        calls.append((args, kwargs))

    from hipengine.kernels.registry import KernelKey, register

    registry_key = KernelKey(*key)
    register(registry_key, fake_kernel, replace=True)
    weights = tuple(
        SimpleNamespace(
            backend="hip_gfx1100",
            spec=SimpleNamespace(layout="dense_f16", quant_key="fp16"),
            allocation=lambda name, ptr=ptr: SimpleNamespace(tensor=SimpleNamespace(ptr=ptr)),
        )
        for ptr in (11, 12, 13)
    )
    try:
        launch_f16_weight_linear_triple(
            *weights,
            x_ptr=10,
            out_a_ptr=20,
            out_b_ptr=21,
            out_c_ptr=22,
            rows=2,
            in_features=3072,
            out_a_features=6144,
            out_b_features=1024,
            out_c_features=1024,
            threads=128,
            stream=7,
            runtime="sentinel",
        )
    finally:
        register(registry_key, original, replace=True)

    assert calls == [
        (
            (10, 11, 12, 13, 20, 21, 22, 2, 3072, 6144, 1024, 1024),
            {"threads": 128, "stream": 7, "runtime": "sentinel"},
        )
    ]


@pytest.mark.parametrize("q_heads", [48, 72])
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_f16_projection_single_dual_triple_match_cpu(q_heads: int) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
        build_laguna_f16_projection,
        laguna_f16w_dual_gemv_bf16_f32_out,
        laguna_f16w_gemv_bf16_bf16_out,
        laguna_f16w_gemv_bf16_f32_out,
        laguna_f16w_triple_gemv_bf16_f32_out,
    )

    rng = np.random.default_rng(281)
    rows, in_features = 1, 256
    out_a, out_b, out_c = q_heads * 128, 8 * 128, 8 * 128
    x_f32 = rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32)
    x_bits = float_array_to_bf16_bits(x_f32)
    x_round = bf16_to_float32(x_bits)
    wa = rng.normal(0.0, 0.05, size=(out_a, in_features)).astype(np.float16)
    wb = rng.normal(0.0, 0.05, size=(out_b, in_features)).astype(np.float16)
    wc = rng.normal(0.0, 0.05, size=(out_c, in_features)).astype(np.float16)
    expected_a = x_round @ wa.astype(np.float32).T
    expected_b = x_round @ wb.astype(np.float32).T
    expected_c = x_round @ wc.astype(np.float32).T

    runtime = get_hip_runtime()
    library = build_laguna_f16_projection(load=True)
    allocations = []
    try:
        dx = _upload(x_bits, runtime, allocations)
        dwa = _upload(wa, runtime, allocations)
        dwb = _upload(wb, runtime, allocations)
        dwc = _upload(wc, runtime, allocations)
        da = _alloc((rows, out_a), np.float32, runtime, allocations)
        db = _alloc((rows, out_b), np.float32, runtime, allocations)
        dc = _alloc((rows, out_c), np.float32, runtime, allocations)
        da_bf16 = _alloc((rows, out_a), np.uint16, runtime, allocations)

        laguna_f16w_gemv_bf16_f32_out(
            dx.ptr, dwa.ptr, da.ptr, rows, in_features, out_a, library=library, runtime=runtime
        )
        laguna_f16w_gemv_bf16_bf16_out(
            dx.ptr,
            dwa.ptr,
            da_bf16.ptr,
            rows,
            in_features,
            out_a,
            library=library,
            runtime=runtime,
        )
        laguna_f16w_dual_gemv_bf16_f32_out(
            dx.ptr,
            dwa.ptr,
            dwb.ptr,
            da.ptr,
            db.ptr,
            rows,
            in_features,
            out_a,
            out_b,
            library=library,
            runtime=runtime,
        )
        laguna_f16w_triple_gemv_bf16_f32_out(
            dx.ptr,
            dwa.ptr,
            dwb.ptr,
            dwc.ptr,
            da.ptr,
            db.ptr,
            dc.ptr,
            rows,
            in_features,
            out_a,
            out_b,
            out_c,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual_a = _download(da, (rows, out_a), np.float32, runtime)
        actual_b = _download(db, (rows, out_b), np.float32, runtime)
        actual_c = _download(dc, (rows, out_c), np.float32, runtime)
        actual_a_bf16 = bf16_to_float32(_download(da_bf16, (rows, out_a), np.uint16, runtime))
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual_a, expected_a, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_b, expected_b, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_c, expected_c, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(
        actual_a_bf16, bf16_to_float32(float_array_to_bf16_bits(expected_a))
    )


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _alloc(shape, dtype, runtime, allocations):
    device = malloc(int(np.prod(shape)) * np.dtype(dtype).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _download(device, shape, dtype, runtime):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host
