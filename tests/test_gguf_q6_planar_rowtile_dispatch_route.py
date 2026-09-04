"""Route contract: planar-Q6 rows 2-8 linear launches select the rowtile leaf.

The proposal/verify stack launches the full-vocab Q6 planar lm-head at rows 2-8
through ``launch_gguf_linear``. The dispatch table's base variant is the
per-row decode leaf, which re-reads the whole head weight once per row
(grid_y = rows). The registered planar rowtile leaf
(``t16_gemv_rowtile_bf16_f32_out`` / ``..._bf16_bf16_out``) reads each weight
tile once for all rows and must stay bit-identical to the per-row decode
kernel. These tests pin the dispatch routing (rows 2-8 -> rowtile, rows 1 and
rows > 8 -> decode) and the bit-exact parity contract.
"""
from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_BF16,
    GGUF_OUTPUT_BF16,
    GGUF_OUTPUT_F32,
    launch_gguf_linear,
    resolve_gguf_linear_dispatch,
)
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
)

QK_K = 256


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


def _planar_weight(*, backend: str | None = None):
    allocations = {
        "raw": SimpleNamespace(tensor=SimpleNamespace(ptr=10)),
        "tiles": SimpleNamespace(tensor=SimpleNamespace(ptr=14)),
    }

    class Weight:
        def __init__(self) -> None:
            self.spec = SimpleNamespace(
                layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
                quant_key="gguf_q6_k_t16_qmicro_planar_v1",
            )
            self.allocations = allocations
            if backend is not None:
                self.backend = backend

        def allocation(self, name: str = "raw"):
            return allocations[name]

    return Weight()


@pytest.mark.parametrize("rows", (2, 3, 4, 5, 6, 7, 8))
def test_planar_q6_launch_routes_rows_2_8_to_rowtile_f32(rows: int) -> None:
    """launch_gguf_linear(rows∈[2,8]) resolves the registered rowtile leaf."""

    from hipengine.kernels.registry import KernelKey, register, resolve

    weight = _planar_weight()
    rowtile_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "t16_gemv_rowtile_bf16_f32_out",
    )
    original = resolve(
        backend=rowtile_key.backend,
        layer=rowtile_key.layer,
        quant=rowtile_key.quant,
        variant=rowtile_key.variant,
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))

    register(rowtile_key, capture, replace=True)
    try:
        launch_gguf_linear(
            weight,
            x_ptr=0x1000,
            out_ptr=0x2000,
            rows=rows,
            in_features=5120,
            out_features=131072,
            output_dtype=GGUF_OUTPUT_F32,
            backend="hip_gfx1100",
            stream=7,
            runtime="runtime-sentinel",
        )
    finally:
        register(rowtile_key, original, replace=True)
        import hipengine.runtime.gguf_linear as gguf_linear_module

        gguf_linear_module.clear_gguf_linear_dispatch_cache()
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:2] == (0x1000, 14)
    assert args[3:6] == (rows, 5120, 131072)
    assert kwargs.get("stream") == 7


@pytest.mark.parametrize("rows", (2, 3, 4, 5, 6, 7, 8))
def test_planar_q6_launch_routes_rows_2_8_to_rowtile_bf16(rows: int) -> None:
    from hipengine.kernels.registry import KernelKey, register, resolve

    weight = _planar_weight()
    rowtile_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "t16_gemv_rowtile_bf16_bf16_out",
    )
    original = resolve(
        backend=rowtile_key.backend,
        layer=rowtile_key.layer,
        quant=rowtile_key.quant,
        variant=rowtile_key.variant,
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))

    register(rowtile_key, capture, replace=True)
    try:
        launch_gguf_linear(
            weight,
            x_ptr=0x1000,
            out_ptr=0x2000,
            rows=rows,
            in_features=5120,
            out_features=131072,
            output_dtype=GGUF_OUTPUT_BF16,
            backend="hip_gfx1100",
            stream=7,
            runtime="runtime-sentinel",
        )
    finally:
        register(rowtile_key, original, replace=True)
        import hipengine.runtime.gguf_linear as gguf_linear_module

        gguf_linear_module.clear_gguf_linear_dispatch_cache()
    assert len(calls) == 1
    args, _kwargs = calls[0]
    assert args[3:6] == (rows, 5120, 131072)


@pytest.mark.parametrize("rows", (1, 9, 16, 64))
def test_planar_q6_launch_keeps_decode_outside_rowtile_window(rows: int) -> None:
    from hipengine.kernels.registry import KernelKey, register, resolve

    weight = _planar_weight()
    rowtile_keys = (
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_rowtile_bf16_f32_out",
        ),
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_rowtile_bf16_bf16_out",
        ),
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in rowtile_keys
    }
    decode_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "t16_gemv_decode_bf16_f32_out",
    )
    decode_original = resolve(
        backend=decode_key.backend,
        layer=decode_key.layer,
        quant=decode_key.quant,
        variant=decode_key.variant,
    )
    rowtile_calls: list[object] = []
    decode_calls: list[object] = []

    def capture_rowtile(*args, **kwargs):
        rowtile_calls.append((args, kwargs))

    def capture_decode(*args, **kwargs):
        decode_calls.append((args, kwargs))

    register(rowtile_keys[0], capture_rowtile, replace=True)
    register(rowtile_keys[1], capture_rowtile, replace=True)
    register(decode_key, capture_decode, replace=True)
    try:
        launch_gguf_linear(
            weight,
            x_ptr=0x1000,
            out_ptr=0x2000,
            rows=rows,
            in_features=5120,
            out_features=131072,
            output_dtype=GGUF_OUTPUT_F32,
            backend="hip_gfx1100",
            stream=7,
            runtime="runtime-sentinel",
        )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)
        register(decode_key, decode_original, replace=True)
        import hipengine.runtime.gguf_linear as gguf_linear_module

        gguf_linear_module.clear_gguf_linear_dispatch_cache()
    assert not rowtile_calls
    assert len(decode_calls) == 1


def test_planar_q6_explicit_wmma_still_wins_over_rowtile() -> None:
    """use_wmma_prefill=True keeps WMMA precedence at rows 2-8."""

    from hipengine.kernels.registry import KernelKey, register, resolve

    weight = _planar_weight()
    wmma_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "t16_wmma_prefill_bf16_bf16_out",
    )
    rowtile_key = KernelKey(
        "hip_gfx1100",
        "linear",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "t16_gemv_rowtile_bf16_bf16_out",
    )
    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (wmma_key, rowtile_key)
    }
    calls: list[str] = []

    def make_capture(label: str):
        def fake(*args, **kwargs):
            calls.append(label)

        return fake

    register(wmma_key, make_capture("wmma"), replace=True)
    register(rowtile_key, make_capture("rowtile"), replace=True)
    try:
        launch_gguf_linear(
            weight,
            x_ptr=0x1000,
            out_ptr=0x2000,
            rows=4,
            in_features=5120,
            out_features=131072,
            output_dtype=GGUF_OUTPUT_BF16,
            backend="hip_gfx1100",
            use_wmma_prefill=True,
            stream=7,
            runtime="runtime-sentinel",
        )
    finally:
        for key, original in originals.items():
            register(key, original, replace=True)
        import hipengine.runtime.gguf_linear as gguf_linear_module

        gguf_linear_module.clear_gguf_linear_dispatch_cache()
    assert calls == ["wmma"]


def test_planar_q6_rowtile_variants_are_registered() -> None:
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        register_gguf_q6_k_t16_gemv_kernels,
    )

    register_gguf_q6_k_t16_gemv_kernels()
    for variant in (
        "t16_gemv_rowtile_bf16_f32_out",
        "t16_gemv_rowtile_bf16_bf16_out",
    ):
        assert is_registered(
            KernelKey(
                "hip_gfx1100",
                "linear",
                "gguf_q6_k_t16_qmicro_planar_v1",
                variant,
            )
        ), variant


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", (2, 4, 5, 7, 8))
def test_planar_q6_rowtile_launch_bitexact_vs_per_row_decode(rows: int) -> None:
    from hipengine.core.hip import get_hip_runtime, HipMemcpyKind
    from hipengine.core.memory import copy_device_to_host, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        build_gguf_q6_k_t16_gemv,
        gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_out,
        gguf_q6_k_t16_qmicro_planar_gemv_rowtile_bf16_f32_out,
        register_gguf_q6_k_t16_gemv_kernels,
    )
    from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16_qmicro_planar
    from tests._gguf_synthetic_weights import make_q6_k_weight

    register_gguf_q6_k_t16_gemv_kernels()
    library = build_gguf_q6_k_t16_gemv(load=True)
    runtime = get_hip_runtime()

    in_features = 512
    out_features = 96
    qweight = make_q6_k_weight(out_features, in_features)[None, ...]
    planar = np.ascontiguousarray(
        repack_gguf_q6_k_tile16_qmicro_planar(qweight).tiles[0]
    )

    rng = np.random.default_rng(rows)
    x_bits = (
        (rng.standard_normal(rows * in_features).astype(np.float32) * 0.2)
        .astype(np.float16)
        .view(np.uint16)
    )

    buffers = []

    def upload(value: np.ndarray):
        value = np.ascontiguousarray(value)
        buffer = malloc(value.nbytes, runtime=runtime)
        buffers.append(buffer)
        runtime.memcpy(
            buffer.ptr,
            host_array_ptr(value),
            value.nbytes,
            HipMemcpyKind.HOST_TO_DEVICE,
        )
        return buffer

    try:
        x_buf = upload(x_bits.view(np.uint8))
        tiles_buf = upload(planar.view(np.uint8))
        ref_buf = malloc(rows * out_features * 4, runtime=runtime)
        got_buf = malloc(rows * out_features * 4, runtime=runtime)
        buffers.extend((ref_buf, got_buf))

        for r in range(rows):
            gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_out(
                x_buf.ptr + r * in_features * 2,
                tiles_buf.ptr,
                ref_buf.ptr + r * out_features * 4,
                1,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
        gguf_q6_k_t16_qmicro_planar_gemv_rowtile_bf16_f32_out(
            x_buf.ptr,
            tiles_buf.ptr,
            got_buf.ptr,
            rows,
            in_features,
            out_features,
            library=library,
            runtime=runtime,
        )
        runtime.device_synchronize()

        ref = np.empty((rows * out_features,), dtype=np.float32)
        got = np.empty((rows * out_features,), dtype=np.float32)
        copy_device_to_host(host_array_ptr(ref), ref_buf, ref.nbytes, runtime=runtime)
        copy_device_to_host(host_array_ptr(got), got_buf, got.nbytes, runtime=runtime)
        np.testing.assert_array_equal(got, ref)
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_planar_q6_launch_gguf_linear_routes_rows8_through_rowtile() -> None:
    """Full-route check: launch_gguf_linear(rows=8) hits the rowtile leaf."""

    from hipengine.core.hip import get_hip_runtime, HipMemcpyKind
    from hipengine.core.memory import copy_device_to_host, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
        build_gguf_q6_k_t16_gemv,
        gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_out,
        register_gguf_q6_k_t16_gemv_kernels,
    )
    from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16_qmicro_planar
    from tests._gguf_synthetic_weights import make_q6_k_weight

    register_gguf_q6_k_t16_gemv_kernels()
    library = build_gguf_q6_k_t16_gemv(load=True)
    runtime = get_hip_runtime()

    in_features = 512
    out_features = 96
    rows = 8
    qweight = make_q6_k_weight(out_features, in_features)[None, ...]
    planar = np.ascontiguousarray(
        repack_gguf_q6_k_tile16_qmicro_planar(qweight).tiles[0]
    )

    rng = np.random.default_rng(1234)
    x_bits = (
        (rng.standard_normal(rows * in_features).astype(np.float32) * 0.2)
        .astype(np.float16)
        .view(np.uint16)
    )

    buffers = []

    def upload(value: np.ndarray):
        value = np.ascontiguousarray(value)
        buffer = malloc(value.nbytes, runtime=runtime)
        buffers.append(buffer)
        runtime.memcpy(
            buffer.ptr,
            host_array_ptr(value),
            value.nbytes,
            HipMemcpyKind.HOST_TO_DEVICE,
        )
        return buffer

    class RouteWeight:
        def __init__(self, tiles_ptr: int) -> None:
            self.spec = SimpleNamespace(
                layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
                quant_key="gguf_q6_k_t16_qmicro_planar_v1",
            )
            self.backend = "hip_gfx1100"
            self._tiles = SimpleNamespace(tensor=SimpleNamespace(ptr=tiles_ptr))

        def allocation(self, name: str = "raw"):
            if name == "tiles":
                return self._tiles
            raise KeyError(name)

    try:
        x_buf = upload(x_bits.view(np.uint8))
        tiles_buf = upload(planar.view(np.uint8))
        ref_buf = malloc(rows * out_features * 4, runtime=runtime)
        got_buf = malloc(rows * out_features * 4, runtime=runtime)
        buffers.extend((ref_buf, got_buf))

        for r in range(rows):
            gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_f32_out(
                x_buf.ptr + r * in_features * 2,
                tiles_buf.ptr,
                ref_buf.ptr + r * out_features * 4,
                1,
                in_features,
                out_features,
                library=library,
                runtime=runtime,
            )
        launch_gguf_linear(
            RouteWeight(tiles_buf.ptr),
            x_buf.ptr,
            got_buf.ptr,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            output_dtype=GGUF_OUTPUT_F32,
            libraries={"gguf_q6_k_t16_qmicro_planar_v1": library},
            runtime=runtime,
        )
        runtime.device_synchronize()

        ref = np.empty((rows * out_features,), dtype=np.float32)
        got = np.empty((rows * out_features,), dtype=np.float32)
        copy_device_to_host(host_array_ptr(ref), ref_buf, ref.nbytes, runtime=runtime)
        copy_device_to_host(host_array_ptr(got), got_buf, got.nbytes, runtime=runtime)
        np.testing.assert_array_equal(got, ref)
    finally:
        for buffer in buffers:
            free(buffer, runtime=runtime)
