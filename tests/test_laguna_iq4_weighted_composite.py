"""Laguna top-10/K1024 certification for the exact IQ4 weighted composite.

The device body and four-axis key predate Laguna.  This module certifies the
Laguna E256/top-10/K1024/N3072 boundary against the registered selected-single
plus weighted-sum fallback without duplicating the primitive.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.benchmark.correctness import evaluate_logits
from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.hip_gfx1100 import quant
from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
    build_paro_combine,
    weighted_sum_out_bf16_f32w,
)
from hipengine.kernels.hip_gfx1100.quant import gguf_iq_gemv
from hipengine.kernels.registry import KernelKey, resolve
from hipengine.quant.gguf import GGMLQuantizationType, bf16_to_float32, dequantize_gguf_data

_VARIANT = "selected_weighted_down_gemv_decode_bf16_bf16_out"
_WRAPPER = "gguf_iq4_xs_weighted_selected_down_bf16_bf16_out"
_KEY = KernelKey("hip_gfx1100", "moe_linear", "gguf_iq4_xs", _VARIANT)
_CONTROL = gguf_iq_gemv.gguf_iq4_xs_selected_gemv_bf16_bf16_out
_SOURCE = Path(gguf_iq_gemv.__file__).with_suffix(".hip")
_QK_K = 256
_BLOCK_BYTES = 136
_TOP_K = 10
_IN_FEATURES = 1024
_THREADS = 256


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def hip_context():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    from hipengine.core.hip import get_hip_runtime

    return (
        get_hip_runtime(),
        gguf_iq_gemv.build_gguf_iq_gemv(load=True),
        build_paro_combine(load=True),
    )


def _candidate():
    return getattr(gguf_iq_gemv, _WRAPPER, None)


def _f32_to_bf16(values: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32).copy()
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    rounded[np.isnan(values)] = np.uint16(0x7FC0)
    return rounded.reshape(values.shape)


def _edge_x(kind: str) -> np.ndarray:
    if kind == "random":
        rng = np.random.default_rng(2026072703)
        return _f32_to_bf16(rng.normal(0.0, 0.2, size=(_TOP_K, _IN_FEATURES)).astype(np.float32))
    edge_bits = np.asarray(
        (
            0x0000,
            0x8000,
            0x0001,
            0x8001,
            0x007F,
            0x807F,
            0x0080,
            0x8080,
            0x3F7F,
            0x3F80,
            0x3F81,
            0xBF7F,
            0xBF80,
            0xBF81,
            0x4700,
            0xC700,
            0x7F7F,
            0xFF7F,
        ),
        dtype=np.uint16,
    )
    if kind == "edges":
        return np.resize(edge_bits, (_TOP_K, _IN_FEATURES)).copy()
    if kind == "signed_zero":
        result = np.zeros((_TOP_K, _IN_FEATURES), dtype=np.uint16)
        result[:, 1::2] = np.uint16(0x8000)
        return result
    raise AssertionError(kind)


def _make_iq4_weight(num_experts: int, out_features: int) -> np.ndarray:
    blocks = _IN_FEATURES // _QK_K
    out = np.empty((num_experts, out_features, blocks * _BLOCK_BYTES), dtype=np.uint8)
    view = out.reshape(num_experts, out_features, blocks, _BLOCK_BYTES)
    rng = np.random.default_rng(0x1A40 + out_features)
    for expert in range(num_experts):
        for row in range(out_features):
            for block in range(blocks):
                scale = np.float16(0.0009765625 * (1 + expert + row + block))
                view[expert, row, block, :2] = np.asarray([scale], dtype=np.float16).view(np.uint8)
                # Cover signed six-bit scale nibbles and every packed Q4 byte
                # without depending on the candidate's device decoder.
                view[expert, row, block, 2:8] = rng.integers(0, 256, size=6, dtype=np.uint8)
                view[expert, row, block, 8:] = rng.integers(0, 256, size=128, dtype=np.uint8)
    return out


def _upload(buffers: list, array: np.ndarray, *, runtime):
    array = np.ascontiguousarray(array)
    buffer = malloc(array.nbytes, runtime=runtime)
    buffers.append(buffer)
    copy_host_to_device(buffer, host_array_ptr(array), array.nbytes, runtime=runtime)
    return buffer


def _download(buffer, shape: tuple[int, ...], *, runtime) -> np.ndarray:
    out = np.empty(shape, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(out), buffer, out.nbytes, runtime=runtime)
    return out


def _run_pair(
    x_bits: np.ndarray,
    selected: np.ndarray,
    routing: np.ndarray,
    qweight: np.ndarray,
    *,
    runtime,
    iq_library,
    combine_library,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = _candidate()
    assert callable(candidate), "IQ4 weighted wrapper must remain registered"
    x_bits = np.ascontiguousarray(x_bits, dtype=np.uint16)
    selected = np.ascontiguousarray(selected, dtype=np.int64)
    routing = np.ascontiguousarray(routing, dtype=np.float32)
    qweight = np.ascontiguousarray(qweight, dtype=np.uint8)
    out_features = qweight.shape[1]
    buffers: list = []
    try:
        x_d = _upload(buffers, x_bits, runtime=runtime)
        selected_d = _upload(buffers, selected, runtime=runtime)
        routing_d = _upload(buffers, routing, runtime=runtime)
        weight_d = _upload(buffers, qweight, runtime=runtime)
        route_d = malloc(_TOP_K * out_features * 2, runtime=runtime)
        control_d = malloc(out_features * 2, runtime=runtime)
        candidate_d = malloc(out_features * 2, runtime=runtime)
        buffers.extend((route_d, control_d, candidate_d))
        _CONTROL(
            x_d.ptr,
            selected_d.ptr,
            weight_d.ptr,
            route_d.ptr,
            x_rows=_TOP_K,
            rows=_TOP_K,
            num_experts=qweight.shape[0],
            in_features=_IN_FEATURES,
            out_features=out_features,
            threads=_THREADS,
            library=iq_library,
            runtime=runtime,
        )
        weighted_sum_out_bf16_f32w(
            route_d.ptr,
            routing_d.ptr,
            control_d.ptr,
            _TOP_K,
            out_features,
            library=combine_library,
            runtime=runtime,
        )
        candidate(
            x_d.ptr,
            selected_d.ptr,
            routing_d.ptr,
            weight_d.ptr,
            candidate_d.ptr,
            tokens=1,
            top_k=_TOP_K,
            num_experts=qweight.shape[0],
            in_features=_IN_FEATURES,
            out_features=out_features,
            library=iq_library,
            runtime=runtime,
        )
        runtime.device_synchronize()
        return (
            _download(control_d, (out_features,), runtime=runtime),
            _download(candidate_d, (out_features,), runtime=runtime),
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def _cpu_reference(
    x_bits: np.ndarray,
    selected: np.ndarray,
    routing: np.ndarray,
    qweight: np.ndarray,
) -> np.ndarray:
    x = bf16_to_float32(x_bits).astype(np.float32)
    weights = dequantize_gguf_data(qweight, GGMLQuantizationType.IQ4_XS)
    routes = np.zeros((_TOP_K, qweight.shape[1]), dtype=np.float32)
    for slot, expert_value in enumerate(selected):
        expert = int(expert_value)
        if 0 <= expert < qweight.shape[0]:
            routes[slot] = bf16_to_float32(
                _f32_to_bf16((weights[expert] @ x[slot]).astype(np.float32))
            )
    out = np.zeros((qweight.shape[1],), dtype=np.float32)
    for slot in range(_TOP_K):
        out = np.add(
            np.multiply(routes[slot], routing[slot], dtype=np.float32),
            out,
            dtype=np.float32,
        )
    return out


def test_iq4_composite_registry_export_default_geometry_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    assert callable(candidate), "IQ4 weighted wrapper must remain registered"
    assert getattr(quant, _WRAPPER, None) is candidate
    gguf_iq_gemv.register_gguf_iq_gemv_kernels(replace=True)
    assert (
        resolve(
            backend=_KEY.backend,
            layer=_KEY.layer,
            quant=_KEY.quant,
            variant=_KEY.variant,
        )
        is candidate
    )
    assert (
        resolve(
            backend="hip_gfx1100",
            layer="moe_linear",
            quant="gguf_iq4_xs",
            variant="selected_gemv_decode_bf16_bf16_out",
        )
        is _CONTROL
    )
    assert (
        gguf_iq_gemv.iq_weighted_down_default_threads(top_k=_TOP_K, in_features=_IN_FEATURES)
        == _THREADS
    )

    build_calls: list[bool] = []

    def fail_build(*, load: bool = True, **_kwargs):
        build_calls.append(load)
        raise AssertionError("validation must reject before library load")

    monkeypatch.setattr(gguf_iq_gemv, "build_gguf_iq_gemv", fail_build)
    pointers = (1,) * 5
    common = dict(tokens=1, top_k=10, num_experts=256, in_features=1024, out_features=3072)
    for field, value, message in (
        ("tokens", 0, "tokens must be positive"),
        ("top_k", 0, "top_k must be positive"),
        ("num_experts", 0, "num_experts must be positive"),
        ("in_features", 1023, "in_features must be positive and divisible by 256"),
        ("out_features", 0, "out_features must be positive"),
        ("threads", 32, "threads must be one of 64, 128, 256"),
    ):
        kwargs = common | {field: value}
        with pytest.raises(ValueError, match=message):
            candidate(*pointers, **kwargs)
    assert build_calls == []


def test_iq4_composite_key_is_excluded_from_unvalidated_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.kernels.hip_gfx1151 as backend

    for unvalidated_backend in ("hip_gfx1151", "cuda_sm86", "cpu_reference"):
        assert (
            resolve(
                backend=unvalidated_backend,
                layer=_KEY.layer,
                quant=_KEY.quant,
                variant=_KEY.variant,
                missing="none",
            )
            is None
        )

    registered: list[KernelKey] = []
    monkeypatch.setattr(backend, "import_module", lambda _name: None)
    monkeypatch.setattr(backend, "registered_keys", lambda: (_KEY,))
    monkeypatch.setattr(backend, "is_registered", lambda _key: False)
    monkeypatch.setattr(backend, "resolve", lambda **_kwargs: object())
    monkeypatch.setattr(
        backend,
        "register",
        lambda key, _kernel, *, replace=False: registered.append(key),
    )
    backend.register_gfx1151_kernels()
    assert _KEY not in registered


def test_iq4_composite_source_preserves_selected_fallback_and_exact_topology() -> None:
    source = _SOURCE.read_text()
    selected = source.split("__global__ void gguf_iq4_xs_selected_gemv_kernel(", 1)[1].split(
        "\n}\n", 1
    )[0]
    assert "iq4_xs_subblock_dot(block, subblock, x_subblock)" in selected
    assert "const float total = reduce_block_sum(acc)" in selected
    assert "float_to_bf16_bits(total)" in selected

    candidate = source.split("__global__ void gguf_iq4_xs_weighted_selected_down_kernel(", 1)[
        1
    ].split("\n}\n", 1)[0]
    assert "__launch_bounds__(256, 2)" in source
    assert "for (int slot = 0; slot < top_k; ++slot)" in candidate
    assert "iq4_xs_subblock_dot(block, subblock, x_subblock)" in candidate
    assert "const float route_total = reduce_block_sum(route_acc)" in candidate
    assert "bf16_bits_to_float(float_to_bf16_bits(route_total))" in candidate
    assert "weighted_total += routing_weights[route_base + slot] * route_bf16" in candidate
    assert candidate.count("__syncthreads()") == 1
    assert "float_to_bf16_bits(weighted_total)" in candidate
    assert "hipengine_gguf_iq4_xs_weighted_selected_down_bf16_bf16_out" in source


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("kind", "out_features"),
    (("edges", 3), ("signed_zero", 17), ("random", 19)),
)
def test_iq4_composite_is_bit_exact_to_selected_plus_registered_reducer(
    hip_context,
    kind: str,
    out_features: int,
) -> None:
    runtime, iq_library, combine_library = hip_context
    x_bits = _edge_x(kind)
    selected = np.asarray([2, 0, 1, -1, 2, 1, 0, 2, 1, 0], dtype=np.int64)
    routing = np.asarray(
        (0.0, -0.0, 1.0, -1.0, 0.61, 0.49, 0.37, 0.13, 0.0078125, -0.00390625),
        dtype=np.float32,
    )
    qweight = _make_iq4_weight(3, out_features)
    control, candidate = _run_pair(
        x_bits,
        selected,
        routing,
        qweight,
        runtime=runtime,
        iq_library=iq_library,
        combine_library=combine_library,
    )
    np.testing.assert_array_equal(candidate, control)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_iq4_composite_passes_independent_cpu_gate(hip_context) -> None:
    runtime, iq_library, combine_library = hip_context
    x_bits = _edge_x("random")
    selected = np.asarray([2, 0, 1, 2, 1, 0, 2, 1, 0, 2], dtype=np.int64)
    routing = np.asarray(
        (0.61, 0.49, 0.37, 0.29, 0.23, 0.19, 0.13, 0.09, 0.07, 0.03),
        dtype=np.float32,
    )
    qweight = _make_iq4_weight(3, 23)
    control, candidate = _run_pair(
        x_bits,
        selected,
        routing,
        qweight,
        runtime=runtime,
        iq_library=iq_library,
        combine_library=combine_library,
    )
    np.testing.assert_array_equal(candidate, control)
    reference = _cpu_reference(x_bits, selected, routing, qweight)
    quality = evaluate_logits(reference, bf16_to_float32(candidate))
    assert quality.kl_max <= 0.05
    assert quality.top1_agreement >= 0.90
