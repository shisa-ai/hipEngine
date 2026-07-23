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
        laguna_f16w_tiled_bf16_f32_out,
        laguna_f16w_triple_gemv_bf16_f32_out,
        laguna_f16w_triple_tiled_bf16_f32_out,
        register_laguna_f16_projection_kernels,
    )

    register_laguna_f16_projection_kernels()
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
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="fp16_weight",
            variant="tiled_bf16_f32_out",
        )
        is laguna_f16w_tiled_bf16_f32_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear_triple",
            quant="fp16_weight",
            variant="tiled_bf16_f32_out",
        )
        is laguna_f16w_triple_tiled_bf16_f32_out
    )


def test_laguna_f16_projection_registry_resolves_matrix_variants() -> None:
    from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
        laguna_f16w_triple_wmma_bf16_f32_out,
        laguna_f16w_wmma_bf16_bf16_out,
        laguna_f16w_wmma_bf16_f32_out,
        register_laguna_f16_projection_kernels,
    )

    register_laguna_f16_projection_kernels()
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="fp16_weight",
            variant="wmma_bf16_f32_out",
        )
        is laguna_f16w_wmma_bf16_f32_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear",
            quant="fp16_weight",
            variant="wmma_bf16_bf16_out",
        )
        is laguna_f16w_wmma_bf16_bf16_out
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="linear_triple",
            quant="fp16_weight",
            variant="wmma_bf16_f32_out",
        )
        is laguna_f16w_triple_wmma_bf16_f32_out
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


def test_laguna_f16_projection_runtime_forced_bulk_keeps_c1_gemv(monkeypatch) -> None:
    from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
        register_laguna_f16_projection_kernels,
    )
    from hipengine.kernels.registry import KernelKey, register
    from hipengine.runtime.f16_weight_linear import launch_f16_weight_linear

    register_laguna_f16_projection_kernels()
    gemv_key = KernelKey("hip_gfx1100", "linear", "fp16_weight", "bf16_f32_out")
    bulk_key = KernelKey(
        "hip_gfx1100", "linear", "fp16_weight", "tiled_bf16_f32_out"
    )
    originals = {
        key: resolve(backend=key.backend, layer=key.layer, quant=key.quant, variant=key.variant)
        for key in (gemv_key, bulk_key)
    }
    calls = []

    def fake(name):
        def kernel(*args, **kwargs):
            calls.append((name, args, kwargs))

        return kernel

    register(gemv_key, fake("gemv"), replace=True)
    register(bulk_key, fake("tiled"), replace=True)
    weight = SimpleNamespace(
        backend="hip_gfx1100",
        spec=SimpleNamespace(layout="dense_f16", quant_key="fp16"),
        allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=11)),
    )
    monkeypatch.setenv("HIPENGINE_LAGUNA_F16_PREFILL", "tiled")
    try:
        for rows in (1, 2):
            launch_f16_weight_linear(
                weight,
                x_ptr=10,
                out_ptr=20,
                rows=rows,
                in_features=3072,
                out_features=6144,
                backend="hip_gfx1100",
                runtime="sentinel",
            )
    finally:
        for key, fn in originals.items():
            register(key, fn, replace=True)

    assert [name for name, _, _ in calls] == ["gemv", "tiled"]


def test_laguna_f16_projection_runtime_forced_wmma_uses_measured_m16_threshold(
    monkeypatch,
) -> None:
    from hipengine.runtime.f16_weight_linear import _prefill_strategy

    monkeypatch.setenv("HIPENGINE_LAGUNA_F16_PREFILL", "wmma")
    assert _prefill_strategy(
        rows=1, activation_dtype="bf16", backend="hip_gfx1151"
    ) is None
    assert _prefill_strategy(
        rows=15, activation_dtype="bf16", backend="hip_gfx1151"
    ) == "tiled"
    assert _prefill_strategy(
        rows=16, activation_dtype="bf16", backend="hip_gfx1151"
    ) == "wmma"
    assert _prefill_strategy(
        rows=512, activation_dtype="bf16", backend="hip_gfx1151"
    ) == "wmma"


def test_laguna_f16_projection_runtime_auto_uses_measured_gfx1151_threshold(
    monkeypatch,
) -> None:
    from hipengine.runtime.f16_weight_linear import _prefill_strategy

    monkeypatch.delenv("HIPENGINE_LAGUNA_F16_PREFILL", raising=False)
    assert _prefill_strategy(
        rows=1, activation_dtype="bf16", backend="hip_gfx1151"
    ) is None
    assert _prefill_strategy(
        rows=2, activation_dtype="bf16", backend="hip_gfx1151"
    ) == "tiled"
    assert _prefill_strategy(
        rows=128, activation_dtype="bf16", backend="hip_gfx1151"
    ) == "tiled"
    assert _prefill_strategy(
        rows=128, activation_dtype="bf16", backend="hip_gfx1100"
    ) is None


@pytest.mark.parametrize("q_heads", [48, 72])
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_f16_projection_single_dual_triple_match_cpu(q_heads: int) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
        build_laguna_f16_projection,
        laguna_f16w_dual_gemv_bf16_f32_out,
        laguna_f16w_gemv_bf16_bf16_out,
        laguna_f16w_gemv_bf16_f32_out,
        laguna_f16w_gemv_f32_bf16_out,
        laguna_f16w_gemv_f32_f32_out,
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
        dx_f32 = _upload(x_f32, runtime, allocations)
        dwa = _upload(wa, runtime, allocations)
        dwb = _upload(wb, runtime, allocations)
        dwc = _upload(wc, runtime, allocations)
        da = _alloc((rows, out_a), np.float32, runtime, allocations)
        db = _alloc((rows, out_b), np.float32, runtime, allocations)
        dc = _alloc((rows, out_c), np.float32, runtime, allocations)
        da_bf16 = _alloc((rows, out_a), np.uint16, runtime, allocations)
        da_f32_input = _alloc((rows, out_a), np.float32, runtime, allocations)
        da_f32_input_bf16 = _alloc((rows, out_a), np.uint16, runtime, allocations)

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
        laguna_f16w_gemv_f32_f32_out(
            dx_f32.ptr,
            dwa.ptr,
            da_f32_input.ptr,
            rows,
            in_features,
            out_a,
            library=library,
            runtime=runtime,
        )
        laguna_f16w_gemv_f32_bf16_out(
            dx_f32.ptr,
            dwa.ptr,
            da_f32_input_bf16.ptr,
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
        actual_f32_input = _download(da_f32_input, (rows, out_a), np.float32, runtime)
        actual_f32_input_bf16 = bf16_to_float32(
            _download(da_f32_input_bf16, (rows, out_a), np.uint16, runtime)
        )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual_a, expected_a, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_b, expected_b, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_c, expected_c, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(
        actual_a_bf16, bf16_to_float32(float_array_to_bf16_bits(expected_a))
    )
    expected_f32_input = x_f32 @ wa.astype(np.float32).T
    np.testing.assert_allclose(actual_f32_input, expected_f32_input, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(
        actual_f32_input_bf16,
        bf16_to_float32(float_array_to_bf16_bits(actual_f32_input)),
    )


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 17])
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_f16_projection_tiled_is_bit_exact_to_gemv(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
        build_laguna_f16_projection,
        build_laguna_f16_projection_prefill,
        laguna_f16w_gemv_bf16_bf16_out,
        laguna_f16w_gemv_bf16_f32_out,
        laguna_f16w_tiled_bf16_bf16_out,
        laguna_f16w_tiled_bf16_f32_out,
    )

    rng = np.random.default_rng(701 + rows)
    in_features, out_features = 512, 35
    x_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32)
    )
    weight = rng.normal(0.0, 0.05, size=(out_features, in_features)).astype(np.float16)
    runtime = get_hip_runtime()
    decode = build_laguna_f16_projection(load=True)
    prefill = build_laguna_f16_projection_prefill(load=True)
    allocations = []
    try:
        dx = _upload(x_bits, runtime, allocations)
        dw = _upload(weight, runtime, allocations)
        gemv_f32 = _alloc((rows, out_features), np.float32, runtime, allocations)
        tiled_f32 = _alloc((rows, out_features), np.float32, runtime, allocations)
        gemv_bf16 = _alloc((rows, out_features), np.uint16, runtime, allocations)
        tiled_bf16 = _alloc((rows, out_features), np.uint16, runtime, allocations)
        laguna_f16w_gemv_bf16_f32_out(
            dx.ptr, dw.ptr, gemv_f32.ptr, rows, in_features, out_features,
            library=decode, runtime=runtime,
        )
        laguna_f16w_tiled_bf16_f32_out(
            dx.ptr, dw.ptr, tiled_f32.ptr, rows, in_features, out_features,
            library=prefill, runtime=runtime,
        )
        laguna_f16w_gemv_bf16_bf16_out(
            dx.ptr, dw.ptr, gemv_bf16.ptr, rows, in_features, out_features,
            library=decode, runtime=runtime,
        )
        laguna_f16w_tiled_bf16_bf16_out(
            dx.ptr, dw.ptr, tiled_bf16.ptr, rows, in_features, out_features,
            library=prefill, runtime=runtime,
        )
        runtime.device_synchronize()
        actual_gemv_f32 = _download(gemv_f32, (rows, out_features), np.float32, runtime)
        actual_tiled_f32 = _download(tiled_f32, (rows, out_features), np.float32, runtime)
        actual_gemv_bf16 = _download(gemv_bf16, (rows, out_features), np.uint16, runtime)
        actual_tiled_bf16 = _download(tiled_bf16, (rows, out_features), np.uint16, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_array_equal(actual_tiled_f32, actual_gemv_f32)
    np.testing.assert_array_equal(actual_tiled_bf16, actual_gemv_bf16)


@pytest.mark.parametrize("rows", [16, 17, 32, 64, 128, 256, 512])
@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_f16_projection_wmma_passes_cpu_quality_gate(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
        build_laguna_f16_projection_prefill,
        laguna_f16w_wmma_bf16_bf16_out,
        laguna_f16w_wmma_bf16_f32_out,
    )

    rng = np.random.default_rng(0xF160 + rows)
    in_features, out_features = 64, 35
    x_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32)
    )
    x_round = bf16_to_float32(x_bits)
    weight = rng.normal(
        0.0, 0.05, size=(out_features, in_features)
    ).astype(np.float16)
    expected = x_round @ weight.astype(np.float32).T
    runtime = get_hip_runtime()
    library = build_laguna_f16_projection_prefill(load=True)
    allocations = []
    try:
        dx = _upload(x_bits, runtime, allocations)
        dw = _upload(weight, runtime, allocations)
        out_f32 = _alloc((rows, out_features), np.float32, runtime, allocations)
        out_bf16 = _alloc((rows, out_features), np.uint16, runtime, allocations)
        laguna_f16w_wmma_bf16_f32_out(
            dx.ptr,
            dw.ptr,
            out_f32.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        laguna_f16w_wmma_bf16_bf16_out(
            dx.ptr,
            dw.ptr,
            out_bf16.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual_f32 = _download(out_f32, (rows, out_features), np.float32, runtime)
        actual_bf16 = _download(out_bf16, (rows, out_features), np.uint16, runtime)
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    np.testing.assert_allclose(actual_f32, expected, rtol=3e-4, atol=3e-4)
    np.testing.assert_array_equal(
        actual_bf16, float_array_to_bf16_bits(actual_f32)
    )
    shifted_expected = expected - np.max(expected, axis=1, keepdims=True)
    shifted_actual = actual_f32 - np.max(actual_f32, axis=1, keepdims=True)
    p = np.exp(shifted_expected)
    q = np.exp(shifted_actual)
    p /= np.sum(p, axis=1, keepdims=True)
    q /= np.sum(q, axis=1, keepdims=True)
    kl = np.sum(p * (np.log(p) - np.log(q)), axis=1)
    top1 = np.mean(np.argmax(expected, axis=1) == np.argmax(actual_f32, axis=1))
    assert float(np.max(kl)) <= 0.05
    assert float(top1) >= 0.9


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_laguna_f16_projection_triple_wmma_matches_three_cpu_matrices() -> None:
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
        build_laguna_f16_projection_prefill,
        laguna_f16w_triple_wmma_bf16_f32_out,
    )

    rng = np.random.default_rng(0x3F16)
    rows, in_features = 16, 64
    widths = (19, 11, 7)
    x_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32)
    )
    x_round = bf16_to_float32(x_bits)
    weights = tuple(
        rng.normal(0.0, 0.05, size=(width, in_features)).astype(np.float16)
        for width in widths
    )
    runtime = get_hip_runtime()
    library = build_laguna_f16_projection_prefill(load=True)
    allocations = []
    try:
        dx = _upload(x_bits, runtime, allocations)
        device_weights = tuple(
            _upload(weight, runtime, allocations) for weight in weights
        )
        outputs = tuple(
            _alloc((rows, width), np.float32, runtime, allocations) for width in widths
        )
        laguna_f16w_triple_wmma_bf16_f32_out(
            dx.ptr,
            *(weight.ptr for weight in device_weights),
            *(out.ptr for out in outputs),
            rows,
            in_features,
            *widths,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual = tuple(
            _download(out, (rows, width), np.float32, runtime)
            for out, width in zip(outputs, widths, strict=True)
        )
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)

    for got, weight in zip(actual, weights, strict=True):
        expected = x_round @ weight.astype(np.float32).T
        np.testing.assert_allclose(got, expected, rtol=3e-4, atol=3e-4)


def test_laguna_f16_projection_wmma_rejects_non_tile_k() -> None:
    from hipengine.kernels.hip_gfx1100.linear.laguna_f16_projection import (
        laguna_f16w_wmma_bf16_f32_out,
    )

    with pytest.raises(ValueError, match="multiple of 16"):
        laguna_f16w_wmma_bf16_f32_out(1, 2, 3, 16, 63, 32)


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
