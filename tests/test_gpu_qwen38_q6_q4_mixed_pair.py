from __future__ import annotations

import ctypes
import importlib
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
from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
    build_gguf_q6_k_t16_gemv,
    gguf_q6_k_t16_gemv_decode_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q4_k_t16_dense_single_local32_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q6_K_T16,
)
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16
from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16
from hipengine.runtime.gguf_linear import (
    clear_gguf_linear_dispatch_cache,
    launch_gguf_linear_pair,
    native_batch_decode_session,
)
from tests._gguf_synthetic_weights import make_q4_k_weight, make_q6_k_weight

_QUANT = "gguf_q6_k_t16_v1+gguf_q4_k_t16_v1"
_VARIANT = "mixed_grid_bf16_bf16_out"
_GFX1100_KEY = KernelKey("hip_gfx1100", "linear_pair", _QUANT, _VARIANT)
_GFX1151_KEY = KernelKey("hip_gfx1151", "linear_pair", _QUANT, _VARIANT)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _pair_module():
    try:
        return importlib.import_module(
            "hipengine.kernels.hip_gfx1100.fused.gguf_q6_q4_pair"
        )
    except ModuleNotFoundError:
        pytest.fail("Qwen3.8 Q6/Q4 mixed-grid pair module is not implemented")


def _pair_wrapper():
    fn = getattr(
        _pair_module(),
        "gguf_q6_q4_t16_mixed_grid_pair_bf16_bf16_out",
        None,
    )
    assert callable(fn), "Qwen3.8 Q6/Q4 mixed-grid wrapper is not implemented"
    return fn


def _fake_weight(
    ptr: int,
    *,
    backend: str,
    layout: str,
    quant_key: str,
):
    allocation = SimpleNamespace(tensor=SimpleNamespace(ptr=int(ptr)))
    return SimpleNamespace(
        backend=backend,
        spec=SimpleNamespace(layout=layout, quant_key=quant_key),
        allocation=lambda _name="tiles": allocation,
    )


def _f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    bits = f32.view(np.uint32).copy()
    bits += 0x7FFF + ((bits >> 16) & 1)
    return (bits >> 16).astype(np.uint16).reshape(f32.shape)


def _bf16_to_f32(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype=np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32).reshape(bits.shape).copy()


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - np.max(values)
    probs = np.exp(shifted)
    return probs / np.sum(probs)


def _upload(runtime, buffers, value: np.ndarray):
    array = np.ascontiguousarray(value)
    buffer = malloc(array.nbytes, runtime=runtime)
    buffers.append(buffer)
    copy_host_to_device(buffer, host_array_ptr(array), array.nbytes, runtime=runtime)
    return buffer


def _read(runtime, buffer, shape) -> np.ndarray:
    value = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(value), buffer, value.nbytes, runtime=runtime)
    return value


def test_q6_q4_mixed_grid_wrapper_is_exposed() -> None:
    assert callable(_pair_wrapper())


def test_q6_q4_mixed_grid_registers_and_routes_only_qualified_gfx1151_shape() -> None:
    module = _pair_module()
    module.register_gguf_q6_q4_pair_kernels(replace=True)
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    assert resolve(
        backend=_GFX1100_KEY.backend,
        layer=_GFX1100_KEY.layer,
        quant=_GFX1100_KEY.quant,
        variant=_GFX1100_KEY.variant,
    ) is _pair_wrapper()
    assert resolve(
        backend=_GFX1151_KEY.backend,
        layer=_GFX1151_KEY.layer,
        quant=_GFX1151_KEY.quant,
        variant=_GFX1151_KEY.variant,
    ) is _pair_wrapper()
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_Q6_Q4_T16_MIXED_GRID_DECODE_SHAPES",
        (),
    ) == frozenset({(1, 5_120, 10_240, 6_144)})
    assert backend_package_capability(
        "hip_gfx1100",
        "GGUF_Q6_Q4_T16_MIXED_GRID_DECODE_SHAPES",
        (),
    ) == ()

    original = resolve(
        backend=_GFX1151_KEY.backend,
        layer=_GFX1151_KEY.layer,
        quant=_GFX1151_KEY.quant,
        variant=_GFX1151_KEY.variant,
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def pair(*args, **kwargs):
        calls.append((args, kwargs))

    register(_GFX1151_KEY, pair, replace=True)
    clear_gguf_linear_dispatch_cache()
    try:
        common = dict(
            weight_a=_fake_weight(
                20,
                backend="hip_gfx1151",
                layout=LAYOUT_GGUF_Q6_K_T16,
                quant_key="gguf_q6_k",
            ),
            weight_b=_fake_weight(
                30,
                backend="hip_gfx1151",
                layout=LAYOUT_GGUF_Q4_K_T16,
                quant_key="gguf_q4_k",
            ),
            x_ptr=10,
            out_a_ptr=40,
            out_b_ptr=50,
            in_features=5_120,
            out_features=10_240,
            out_features_b=6_144,
            backend="hip_gfx1151",
            stream=7,
            runtime="runtime-sentinel",
        )
        assert launch_gguf_linear_pair(rows=1, **common)
        with native_batch_decode_session():
            assert not launch_gguf_linear_pair(rows=1, **common)
        assert not launch_gguf_linear_pair(rows=2, **common)
        bad_width = {
            key: value for key, value in common.items() if key != "out_features_b"
        }
        assert not launch_gguf_linear_pair(
            rows=1,
            out_features_b=6_160,
            **bad_width,
        )
    finally:
        register(_GFX1151_KEY, original, replace=True)
        clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            (10, 20, 30, 40, 50, 1, 5_120, 10_240, 6_144),
            {"stream": 7, "runtime": "runtime-sentinel"},
        )
    ]


def test_q6_q4_mixed_grid_wrapper_rejects_unscreened_geometry() -> None:
    pair = _pair_wrapper()
    with pytest.raises(ValueError, match="rows == 1"):
        pair(0, 0, 0, 0, 0, 2, 512, 256, 256)
    with pytest.raises(ValueError, match="multiple of 256"):
        pair(0, 0, 0, 0, 0, 1, 384, 256, 256)
    with pytest.raises(ValueError, match="Q6 output.*multiple of 16"):
        pair(0, 0, 0, 0, 0, 1, 512, 248, 256)
    with pytest.raises(ValueError, match="Q4 output.*multiple of 32"):
        pair(0, 0, 0, 0, 0, 1, 512, 256, 240)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_q6_q4_mixed_grid_is_bit_exact_and_passes_cpu_kl_top1() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    pair_library = _pair_module().build_gguf_q6_q4_pair(load=True)
    q6_library = build_gguf_q6_k_t16_gemv(load=True)
    q4_library = build_gguf_t16_selected_gemv(load=True)
    pair = _pair_wrapper()
    rows = 1
    in_features = 512
    q6_features = 256
    q4_features = 256
    rng = np.random.default_rng(0x6_4_38)
    x_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.25, size=(rows, in_features)).astype(np.float32)
    )
    q6_raw = make_q6_k_weight(q6_features, in_features)
    q4_raw = make_q4_k_weight(q4_features, in_features)
    q6_tiles = repack_gguf_q6_k_tile16(q6_raw[None, ...]).tiles
    q4_tiles = repack_gguf_q4_k_tile16(q4_raw[None, ...]).tiles
    buffers = []
    try:
        x = _upload(runtime, buffers, x_bits)
        q6 = _upload(runtime, buffers, q6_tiles)
        q4 = _upload(runtime, buffers, q4_tiles)
        control_q6 = malloc(rows * q6_features * 2, runtime=runtime)
        control_q4 = malloc(rows * q4_features * 2, runtime=runtime)
        candidate_q6 = malloc(rows * q6_features * 2, runtime=runtime)
        candidate_q4 = malloc(rows * q4_features * 2, runtime=runtime)
        buffers.extend((control_q6, control_q4, candidate_q6, candidate_q4))

        gguf_q6_k_t16_gemv_decode_bf16_bf16_out(
            x.ptr,
            q6.ptr,
            control_q6.ptr,
            rows,
            in_features,
            q6_features,
            library=q6_library,
            runtime=runtime,
        )
        gguf_q4_k_t16_dense_single_local32_bf16_bf16_out(
            x.ptr,
            q4.ptr,
            control_q4.ptr,
            rows,
            in_features,
            q4_features,
            library=q4_library,
            runtime=runtime,
        )
        pair(
            x.ptr,
            q6.ptr,
            q4.ptr,
            candidate_q6.ptr,
            candidate_q4.ptr,
            rows,
            in_features,
            q6_features,
            q4_features,
            library=pair_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        expected_q6 = _read(runtime, control_q6, (rows, q6_features))
        expected_q4 = _read(runtime, control_q4, (rows, q4_features))
        actual_q6 = _read(runtime, candidate_q6, (rows, q6_features))
        actual_q4 = _read(runtime, candidate_q4, (rows, q4_features))
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(actual_q6, expected_q6)
    np.testing.assert_array_equal(actual_q4, expected_q4)

    x_f32 = _bf16_to_f32(x_bits)
    cpu_logits = np.concatenate(
        (
            gguf_quant_gemv(x_f32, q6_raw, GGMLQuantizationType.Q6_K),
            gguf_quant_gemv(x_f32, q4_raw, GGMLQuantizationType.Q4_K),
        ),
        axis=1,
    )[0]
    gpu_logits = np.concatenate(
        (_bf16_to_f32(actual_q6), _bf16_to_f32(actual_q4)), axis=1
    )[0]
    cpu_probs = _softmax(cpu_logits)
    gpu_probs = _softmax(gpu_logits)
    kl = float(np.sum(cpu_probs * np.log(cpu_probs / gpu_probs)))
    assert kl <= 0.05
    assert int(np.argmax(cpu_logits)) == int(np.argmax(gpu_logits))
