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
from hipengine.kernels.hip_gfx1100.linear import dense_gemv_bf16_f32w_bf16_out
from hipengine.kernels.hip_gfx1100.linear import dense_gemv as dense_gemv_module
from hipengine.kernels.registry import KernelKey, is_registered, register, resolve, unregister
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_DENSE_F32
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.gguf_linear import launch_gguf_linear_pair

_PAIR_KEY = KernelKey(
    "hip_gfx1100",
    "linear_pair",
    "f32",
    "bf16_hidden_bf16_out",
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _pair_wrapper():
    wrapper = getattr(dense_gemv_module, "dense_pair_gemv_bf16_f32w_bf16_out", None)
    assert callable(wrapper), "dense F32 alpha/beta pair wrapper is not implemented"
    return wrapper


def _fake_weight(ptr: int, *, backend: str = "hip_gfx1100"):
    allocation = SimpleNamespace(tensor=SimpleNamespace(ptr=int(ptr)))
    return SimpleNamespace(
        backend=backend,
        spec=SimpleNamespace(layout=LAYOUT_DENSE_F32, quant_key="f32"),
        allocation=lambda _name="raw": allocation,
    )


def test_dense_f32_pair_registers_only_on_screened_gfx1100_backend() -> None:
    dense_gemv_module.register_dense_gemv_kernels(replace=True)
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)

    assert resolve(
        backend=_PAIR_KEY.backend,
        layer=_PAIR_KEY.layer,
        quant=_PAIR_KEY.quant,
        variant=_PAIR_KEY.variant,
    ) is _pair_wrapper()
    assert not is_registered(
        KernelKey(
            "hip_gfx1151",
            _PAIR_KEY.layer,
            _PAIR_KEY.quant,
            _PAIR_KEY.variant,
        )
    )


def test_dense_f32_pair_wrapper_selects_flat_rows1_to3_and_tile2_row4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    def signed(_library, symbol, _argtypes, _restype):
        def launch(*args):
            calls.append((symbol, args))
            return 0

        return launch

    monkeypatch.setattr(dense_gemv_module, "signed_kernel_fn", signed)
    wrapper = _pair_wrapper()
    runtime = SimpleNamespace(check=lambda err: pytest.fail(f"unexpected HIP error {err}"))
    library = object()

    for rows in (1, 2, 3, 4):
        wrapper(
            10,
            20,
            30,
            40,
            50,
            rows,
            5120,
            48,
            library=library,
            runtime=runtime,
        )

    assert [symbol for symbol, _args in calls] == [
        "hipengine_dense_pair_gemv_bf16_f32w_bf16_out",
        "hipengine_dense_pair_gemv_bf16_f32w_bf16_out",
        "hipengine_dense_pair_gemv_bf16_f32w_bf16_out",
        "hipengine_dense_pair_gemv_bf16_f32w_bf16_out_rowtile2",
    ]
    assert [int(args[5]) for _symbol, args in calls] == [1, 2, 3, 4]


def test_dense_f32_pair_dispatch_owns_equal_small_rows_and_fails_closed() -> None:
    wrapper = _pair_wrapper()
    original = resolve(
        backend=_PAIR_KEY.backend,
        layer=_PAIR_KEY.layer,
        quant=_PAIR_KEY.quant,
        variant=_PAIR_KEY.variant,
        missing="none",
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def pair(*args, **kwargs):
        calls.append((args, kwargs))

    register(_PAIR_KEY, pair, replace=True)
    try:
        for rows in (1, 2, 3, 4):
            assert launch_gguf_linear_pair(
                _fake_weight(20),
                _fake_weight(30),
                x_ptr=10,
                out_a_ptr=40,
                out_b_ptr=50,
                rows=rows,
                in_features=5120,
                out_features=48,
                backend="hip_gfx1100",
                stream=7,
                runtime="runtime-sentinel",
            )
        assert not launch_gguf_linear_pair(
            _fake_weight(20),
            _fake_weight(30),
            10,
            40,
            50,
            5,
            5120,
            48,
            backend="hip_gfx1100",
        )
        assert not launch_gguf_linear_pair(
            _fake_weight(20),
            _fake_weight(30),
            10,
            40,
            50,
            4,
            5120,
            48,
            out_features_b=96,
            backend="hip_gfx1100",
        )
        assert not launch_gguf_linear_pair(
            _fake_weight(20, backend="hip_gfx1151"),
            _fake_weight(30, backend="hip_gfx1151"),
            10,
            40,
            50,
            4,
            5120,
            48,
            backend="hip_gfx1151",
        )
        unregister(_PAIR_KEY)
        assert not launch_gguf_linear_pair(
            _fake_weight(20),
            _fake_weight(30),
            10,
            40,
            50,
            4,
            5120,
            48,
            backend="hip_gfx1100",
        )
    finally:
        register(_PAIR_KEY, original or wrapper, replace=True)

    assert len(calls) == 4
    assert [int(args[5]) for args, _kwargs in calls] == [1, 2, 3, 4]
    assert all(args[:5] == (10, 20, 30, 40, 50) for args, _kwargs in calls)
    assert all(kwargs["stream"] == 7 for _args, kwargs in calls)


def test_dense_f32_pair_wrapper_rejects_unscreened_shapes_before_loading() -> None:
    wrapper = _pair_wrapper()
    with pytest.raises(ValueError, match="rows must be between 1 and 4"):
        wrapper(0, 0, 0, 0, 0, 0, 5120, 48)
    with pytest.raises(ValueError, match="rows must be between 1 and 4"):
        wrapper(0, 0, 0, 0, 0, 5, 5120, 48)
    with pytest.raises(ValueError, match="in_features must be positive"):
        wrapper(0, 0, 0, 0, 0, 1, 0, 48)
    with pytest.raises(ValueError, match="out_features must be positive"):
        wrapper(0, 0, 0, 0, 0, 1, 5120, 0)
    with pytest.raises(ValueError, match="threads must equal 256"):
        wrapper(0, 0, 0, 0, 0, 1, 5120, 48, threads=128)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", (1, 2, 3, 4))
def test_dense_f32_pair_is_scalar_bit_exact_and_passes_cpu_kl_top1(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    library = dense_gemv_module.build_dense_gemv(load=True)
    pair = _pair_wrapper()
    in_features = 5120
    out_features = 48
    rng = np.random.default_rng(0xF32A + rows)
    x_bits = float_array_to_bf16_bits(
        rng.normal(0.0, 0.25, size=(rows, in_features)).astype(np.float32)
    )
    weight_a = rng.normal(0.0, 0.05, size=(out_features, in_features)).astype(np.float32)
    weight_b = rng.normal(0.0, 0.05, size=(out_features, in_features)).astype(np.float32)
    shape = (rows, out_features)
    scalar_a = np.empty(shape, dtype=np.uint16)
    scalar_b = np.empty(shape, dtype=np.uint16)
    pair_a = np.empty(shape, dtype=np.uint16)
    pair_b = np.empty(shape, dtype=np.uint16)
    buffers = []
    try:
        dx = malloc(x_bits.nbytes, runtime=runtime)
        dwa = malloc(weight_a.nbytes, runtime=runtime)
        dwb = malloc(weight_b.nbytes, runtime=runtime)
        dsa = malloc(scalar_a.nbytes, runtime=runtime)
        dsb = malloc(scalar_b.nbytes, runtime=runtime)
        dpa = malloc(pair_a.nbytes, runtime=runtime)
        dpb = malloc(pair_b.nbytes, runtime=runtime)
        buffers.extend((dx, dwa, dwb, dsa, dsb, dpa, dpb))
        copy_host_to_device(dx, host_array_ptr(x_bits), runtime=runtime)
        copy_host_to_device(dwa, host_array_ptr(weight_a), runtime=runtime)
        copy_host_to_device(dwb, host_array_ptr(weight_b), runtime=runtime)
        dense_gemv_bf16_f32w_bf16_out(
            dx.ptr,
            dwa.ptr,
            dsa.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        dense_gemv_bf16_f32w_bf16_out(
            dx.ptr,
            dwb.ptr,
            dsb.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        pair(
            dx.ptr,
            dwa.ptr,
            dwb.ptr,
            dpa.ptr,
            dpb.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        for host, device in (
            (scalar_a, dsa),
            (scalar_b, dsb),
            (pair_a, dpa),
            (pair_b, dpb),
        ):
            copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(pair_a, scalar_a)
    np.testing.assert_array_equal(pair_b, scalar_b)
    expected = np.concatenate(
        (
            bf16_to_float32(x_bits) @ weight_a.T,
            bf16_to_float32(x_bits) @ weight_b.T,
        ),
        axis=1,
    )
    actual = np.concatenate((bf16_to_float32(pair_a), bf16_to_float32(pair_b)), axis=1)
    p = np.exp(expected - np.max(expected, axis=-1, keepdims=True), dtype=np.float64)
    p /= np.sum(p, axis=-1, keepdims=True)
    q = np.exp(actual - np.max(actual, axis=-1, keepdims=True), dtype=np.float64)
    q /= np.sum(q, axis=-1, keepdims=True)
    kl = np.sum(p * (np.log(p + 1.0e-30) - np.log(q + 1.0e-30)), axis=-1)
    assert float(np.max(kl)) <= 0.05
    assert float(np.mean(np.argmax(expected, axis=-1) == np.argmax(actual, axis=-1))) >= 0.90
