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
from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.fused import gguf_q6_q4_pair as pair_module
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
    build_gguf_q6_k_t16_gemv,
    gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    build_gguf_t16_selected_gemv,
    gguf_q4_k_t16_dense_single_col4_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, register, resolve
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
)
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_tile16
from hipengine.quant.gguf_t16 import repack_gguf_q6_k_tile16_qmicro_planar
from hipengine.runtime.gguf_linear import (
    clear_gguf_linear_dispatch_cache,
    launch_gguf_linear_pair,
    native_batch_decode_session,
)
from tests._gguf_synthetic_weights import make_q4_k_weight, make_q6_k_weight

_Q4_Q4_QUANT = "gguf_q4_k_t16_v1"
_Q4_Q4_VARIANT = "narrow_col4_pair_bf16_bf16_out"
_Q4_Q6_QUANT = "gguf_q4_k_t16_v1+gguf_q6_k_t16_qmicro_planar_v1"
_Q4_Q6_VARIANT = "narrow_col4_planar_pair_bf16_bf16_out"
_Q4_Q4_GFX1151_KEY = KernelKey(
    "hip_gfx1151", "linear_pair", _Q4_Q4_QUANT, _Q4_Q4_VARIANT
)
_Q4_Q6_GFX1151_KEY = KernelKey(
    "hip_gfx1151", "linear_pair", _Q4_Q6_QUANT, _Q4_Q6_VARIANT
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _wrapper(name: str):
    fn = getattr(pair_module, name, None)
    assert callable(fn), f"Qwen3.8 narrow K/V wrapper {name} is not implemented"
    return fn


def _q4_q4_wrapper():
    return _wrapper("gguf_q4_q4_t16_narrow_col4_pair_bf16_bf16_out")


def _q4_q6_wrapper():
    return _wrapper("gguf_q4_q6_t16_narrow_col4_planar_pair_bf16_bf16_out")


def _fake_weight(ptr: int, *, backend: str, layout: str, quant_key: str):
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


def test_narrow_kv_pair_wrappers_are_exposed() -> None:
    assert callable(_q4_q4_wrapper())
    assert callable(_q4_q6_wrapper())


def test_narrow_kv_pairs_route_only_qualified_gfx1151_shapes() -> None:
    pair_module.register_gguf_q6_q4_pair_kernels(replace=True)
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels

    register_gfx1151_kernels(replace=True)
    assert backend_package_capability(
        "hip_gfx1151",
        "GGUF_NARROW_KV_PAIR_DECODE_SHAPES",
        (),
    ) == frozenset({(1, 5_120, 1_024, 1_024)})
    assert backend_package_capability(
        "hip_gfx1100",
        "GGUF_NARROW_KV_PAIR_DECODE_SHAPES",
        (),
    ) == ()

    originals = {
        key: resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        for key in (_Q4_Q4_GFX1151_KEY, _Q4_Q6_GFX1151_KEY)
    }
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def pair_q4(*args, **kwargs):
        calls.append(("q4_q4", args, kwargs))

    def pair_q6(*args, **kwargs):
        calls.append(("q4_q6", args, kwargs))

    register(_Q4_Q4_GFX1151_KEY, pair_q4, replace=True)
    register(_Q4_Q6_GFX1151_KEY, pair_q6, replace=True)
    clear_gguf_linear_dispatch_cache()
    q4_a = _fake_weight(
        20,
        backend="hip_gfx1151",
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k",
    )
    q4_b = _fake_weight(
        30,
        backend="hip_gfx1151",
        layout=LAYOUT_GGUF_Q4_K_T16,
        quant_key="gguf_q4_k",
    )
    q6_b = _fake_weight(
        31,
        backend="hip_gfx1151",
        layout=LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        quant_key="gguf_q6_k_t16_qmicro_planar_v1",
    )
    common = dict(
        x_ptr=10,
        out_a_ptr=40,
        out_b_ptr=50,
        rows=1,
        in_features=5_120,
        out_features=1_024,
        backend="hip_gfx1151",
        stream=7,
        runtime="runtime-sentinel",
    )
    try:
        assert launch_gguf_linear_pair(q4_a, q4_b, **common)
        assert launch_gguf_linear_pair(q4_a, q6_b, **common)
        with native_batch_decode_session():
            assert not launch_gguf_linear_pair(q4_a, q4_b, **common)
            assert not launch_gguf_linear_pair(q4_a, q6_b, **common)
        assert not launch_gguf_linear_pair(
            q4_a,
            q4_b,
            **{**common, "out_features": 1_040},
        )
    finally:
        for key, fn in originals.items():
            register(key, fn, replace=True)
        clear_gguf_linear_dispatch_cache()

    expected_args = (10, 20, 30, 40, 50, 1, 5_120, 1_024, 1_024)
    expected_q6_args = (10, 20, 31, 40, 50, 1, 5_120, 1_024, 1_024)
    expected_kwargs = {"stream": 7, "runtime": "runtime-sentinel"}
    assert calls == [
        ("q4_q4", expected_args, expected_kwargs),
        ("q4_q6", expected_q6_args, expected_kwargs),
    ]


def test_narrow_kv_pair_wrappers_reject_unscreened_geometry() -> None:
    for pair in (_q4_q4_wrapper(), _q4_q6_wrapper()):
        with pytest.raises(ValueError, match="rows == 1"):
            pair(0, 0, 0, 0, 0, 2, 512, 256, 256)
        with pytest.raises(ValueError, match="multiple of 256"):
            pair(0, 0, 0, 0, 0, 1, 384, 256, 256)
        with pytest.raises(ValueError, match="multiple of 16"):
            pair(0, 0, 0, 0, 0, 1, 512, 248, 256)


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
def test_narrow_kv_pairs_are_bit_exact_and_pass_cpu_kl_top1() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    pair_library = pair_module.build_gguf_q6_q4_pair(load=True)
    q4_library = build_gguf_t16_selected_gemv(load=True)
    q6_library = build_gguf_q6_k_t16_gemv(load=True)
    rows = 1
    in_features = 512
    out_features = 256
    rng = np.random.default_rng(0x4_6_4_38)
    x_bits = _f32_to_bf16_bits(
        rng.normal(0.0, 0.25, size=(rows, in_features)).astype(np.float32)
    )
    raw_q4_a = make_q4_k_weight(out_features, in_features)
    raw_q4_b = make_q4_k_weight(out_features, in_features)
    raw_q6_b = make_q6_k_weight(out_features, in_features)
    tiles_q4_a = repack_gguf_q4_k_tile16(raw_q4_a[None, ...]).tiles
    tiles_q4_b = repack_gguf_q4_k_tile16(raw_q4_b[None, ...]).tiles
    tiles_q6_b = repack_gguf_q6_k_tile16_qmicro_planar(raw_q6_b[None, ...]).tiles
    buffers = []
    try:
        x = _upload(runtime, buffers, x_bits)
        q4_a = _upload(runtime, buffers, tiles_q4_a)
        q4_b = _upload(runtime, buffers, tiles_q4_b)
        q6_b = _upload(runtime, buffers, tiles_q6_b)
        control_a = malloc(rows * out_features * 2, runtime=runtime)
        control_q4_b = malloc(rows * out_features * 2, runtime=runtime)
        control_q6_b = malloc(rows * out_features * 2, runtime=runtime)
        candidate_a = malloc(rows * out_features * 2, runtime=runtime)
        candidate_b = malloc(rows * out_features * 2, runtime=runtime)
        buffers.extend(
            (control_a, control_q4_b, control_q6_b, candidate_a, candidate_b)
        )

        gguf_q4_k_t16_dense_single_col4_bf16_bf16_out(
            x.ptr,
            q4_a.ptr,
            control_a.ptr,
            rows,
            in_features,
            out_features,
            library=q4_library,
            runtime=runtime,
        )
        gguf_q4_k_t16_dense_single_col4_bf16_bf16_out(
            x.ptr,
            q4_b.ptr,
            control_q4_b.ptr,
            rows,
            in_features,
            out_features,
            library=q4_library,
            runtime=runtime,
        )
        gguf_q6_k_t16_qmicro_planar_gemv_decode_bf16_bf16_out(
            x.ptr,
            q6_b.ptr,
            control_q6_b.ptr,
            rows,
            in_features,
            out_features,
            library=q6_library,
            runtime=runtime,
        )
        _q4_q4_wrapper()(
            x.ptr,
            q4_a.ptr,
            q4_b.ptr,
            candidate_a.ptr,
            candidate_b.ptr,
            rows,
            in_features,
            out_features,
            out_features,
            library=pair_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        expected_a = _read(runtime, control_a, (rows, out_features))
        expected_q4_b = _read(runtime, control_q4_b, (rows, out_features))
        actual_a = _read(runtime, candidate_a, (rows, out_features))
        actual_q4_b = _read(runtime, candidate_b, (rows, out_features))
        _q4_q6_wrapper()(
            x.ptr,
            q4_a.ptr,
            q6_b.ptr,
            candidate_a.ptr,
            candidate_b.ptr,
            rows,
            in_features,
            out_features,
            out_features,
            library=pair_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        actual_q6_a = _read(runtime, candidate_a, (rows, out_features))
        actual_q6_b = _read(runtime, candidate_b, (rows, out_features))
        expected_q6_b = _read(runtime, control_q6_b, (rows, out_features))
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)

    np.testing.assert_array_equal(actual_a, expected_a)
    np.testing.assert_array_equal(actual_q4_b, expected_q4_b)
    np.testing.assert_array_equal(actual_q6_a, expected_a)
    np.testing.assert_array_equal(actual_q6_b, expected_q6_b)

    x_f32 = _bf16_to_f32(x_bits)
    for raw_b, gpu_b in (
        (raw_q4_b, actual_q4_b),
        (raw_q6_b, actual_q6_b),
    ):
        cpu_logits = np.concatenate(
            (
                gguf_quant_gemv(x_f32, raw_q4_a, GGMLQuantizationType.Q4_K),
                gguf_quant_gemv(
                    x_f32,
                    raw_b,
                    GGMLQuantizationType.Q4_K
                    if raw_b is raw_q4_b
                    else GGMLQuantizationType.Q6_K,
                ),
            ),
            axis=1,
        )[0]
        gpu_logits = np.concatenate(
            (_bf16_to_f32(actual_a), _bf16_to_f32(gpu_b)), axis=1
        )[0]
        cpu_probs = _softmax(cpu_logits)
        gpu_probs = _softmax(gpu_logits)
        kl = float(np.sum(cpu_probs * np.log(cpu_probs / gpu_probs)))
        assert kl <= 0.05
        assert int(np.argmax(cpu_logits)) == int(np.argmax(gpu_logits))
