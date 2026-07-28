"""Exact RED/GREEN gate for Laguna's IQ3 wave10 sign-bit fusion."""

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
from hipengine.kernels.hip_gfx1100.quant import gguf_iq_gemv
from hipengine.kernels.registry import KernelKey, resolve
from hipengine.quant.gguf import GGMLQuantizationType, bf16_to_float32, dequantize_gguf_data

_VARIANT = (
    "selected_weighted_down_gemv_decode_k1024_wave10_signbit_bf16_bf16_out"
)
_WRAPPER = (
    "gguf_iq3_xxs_weighted_selected_down_k1024_wave10_signbit_bf16_bf16_out"
)
_KEY = KernelKey("hip_gfx1100", "moe_linear", "gguf_iq3_xxs", _VARIANT)
_CONTROL = (
    gguf_iq_gemv.gguf_iq3_xxs_weighted_selected_down_k1024_wave10_bf16_bf16_out
)
_UNFUSED = gguf_iq_gemv.gguf_iq3_xxs_selected_gemv_k1024_wave4_bf16_bf16_out
_SOURCE = Path(gguf_iq_gemv.__file__).with_suffix(".hip")
_QK_K = 256
_BLOCK_BYTES = 98
_TOP_K = 10
_IN_FEATURES = 1024
_THREADS = 320


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

    return get_hip_runtime(), gguf_iq_gemv.build_gguf_iq_gemv(load=True)


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
        rng = np.random.default_rng(2026072702)
        return _f32_to_bf16(
            rng.normal(0.0, 0.2, size=(_TOP_K, _IN_FEATURES)).astype(np.float32)
        )
    if kind == "edges":
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
        return np.resize(edge_bits, (_TOP_K, _IN_FEATURES)).copy()
    if kind == "signed_zero":
        result = np.zeros((_TOP_K, _IN_FEATURES), dtype=np.uint16)
        result[:, 1::2] = np.uint16(0x8000)
        return result
    raise AssertionError(kind)


def _make_iq3_weight(num_experts: int, out_features: int, *, exhaustive: bool) -> np.ndarray:
    blocks = _IN_FEATURES // _QK_K
    out = np.zeros(
        (num_experts, out_features, blocks * _BLOCK_BYTES), dtype=np.uint8
    )
    rng = np.random.default_rng(0x1A30 + out_features)
    view = out.reshape(num_experts, out_features, blocks, _BLOCK_BYTES)
    for expert in range(num_experts):
        for row in range(out_features):
            for block in range(blocks):
                scale = np.float16(0.0009765625 * (1 + expert + row + block))
                view[expert, row, block, :2] = np.asarray(
                    [scale], dtype=np.float16
                ).view(np.uint8)
                if exhaustive:
                    for offset in range(64):
                        base = block * 64 + offset
                        grid_index = (
                            base
                            if row % 3 == 0
                            else (255 - base if row % 3 == 1 else 73 * base + 19)
                        )
                        view[expert, row, block, 2 + offset] = np.uint8(
                            (grid_index + 37 * expert) & 255
                        )
                    for group32 in range(8):
                        aux = np.uint32((expert + row + block) & 15) << np.uint32(28)
                        for local8 in range(4):
                            base = block * 32 + group32 * 4 + local8
                            selector = base if row % 2 == 0 else 127 - base
                            selector = (selector + 29 * expert) & 127
                            aux |= np.uint32(selector) << np.uint32(7 * local8)
                        view[expert, row, block, 66 + 4 * group32 : 70 + 4 * group32] = np.asarray(
                            [aux], dtype="<u4"
                        ).view(np.uint8)
                else:
                    view[expert, row, block, 2:] = rng.integers(
                        0, 256, size=_BLOCK_BYTES - 2, dtype=np.uint8
                    )
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
) -> tuple[np.ndarray, np.ndarray]:
    candidate = _candidate()
    assert callable(candidate), "IQ3 wave10 sign-bit wrapper must be admitted"
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
        control_d = malloc(out_features * 2, runtime=runtime)
        candidate_d = malloc(out_features * 2, runtime=runtime)
        buffers.extend((control_d, candidate_d))
        for launch, output in ((_CONTROL, control_d), (candidate, candidate_d)):
            launch(
                x_d.ptr,
                selected_d.ptr,
                routing_d.ptr,
                weight_d.ptr,
                output.ptr,
                tokens=1,
                top_k=_TOP_K,
                num_experts=qweight.shape[0],
                in_features=_IN_FEATURES,
                out_features=out_features,
                threads=_THREADS,
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
    weights = dequantize_gguf_data(qweight, GGMLQuantizationType.IQ3_XXS)
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


def test_wave10_signbit_registry_package_export_and_preload_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    assert callable(candidate), "IQ3 wave10 sign-bit wrapper must be admitted"
    assert getattr(quant, _WRAPPER, None) is candidate
    gguf_iq_gemv.register_gguf_iq_gemv_kernels(replace=True)
    assert resolve(
        backend=_KEY.backend,
        layer=_KEY.layer,
        quant=_KEY.quant,
        variant=_KEY.variant,
    ) is candidate
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant="selected_weighted_down_gemv_decode_k1024_wave10_bf16_bf16_out",
    ) is _CONTROL
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant="selected_gemv_decode_k1024_wave4_bf16_bf16_out",
    ) is _UNFUSED

    build_calls: list[bool] = []

    def fail_build(*, load: bool = True, **_kwargs):
        build_calls.append(load)
        raise AssertionError("validation must reject before library load")

    monkeypatch.setattr(gguf_iq_gemv, "build_gguf_iq_gemv", fail_build)
    pointers = (1,) * 5
    common = dict(top_k=10, num_experts=256, in_features=1024, out_features=3072)
    with pytest.raises(ValueError, match="tokens must be exactly 1"):
        candidate(*pointers, tokens=2, **common)
    with pytest.raises(ValueError, match="top_k must be exactly 10"):
        candidate(*pointers, tokens=1, **(common | {"top_k": 8}))
    with pytest.raises(ValueError, match="num_experts must be positive"):
        candidate(*pointers, tokens=1, **(common | {"num_experts": 0}))
    with pytest.raises(ValueError, match="in_features must be exactly 1024"):
        candidate(*pointers, tokens=1, **(common | {"in_features": 512}))
    with pytest.raises(ValueError, match="out_features must be positive"):
        candidate(*pointers, tokens=1, **(common | {"out_features": 0}))
    with pytest.raises(ValueError, match="threads must be exactly 320"):
        candidate(*pointers, tokens=1, threads=32, **common)
    for index, name in enumerate(
        ("x_ptr", "selected_ptr", "routing_weights_ptr", "qweight_ptr", "out_ptr")
    ):
        nulls = list(pointers)
        nulls[index] = 0
        with pytest.raises(ValueError, match=rf"{name} must be non-zero"):
            candidate(*nulls, tokens=1, **common)
    assert build_calls == []


def test_wave10_signbit_key_is_excluded_from_unvalidated_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hipengine.kernels.hip_gfx1151 as backend

    for unvalidated_backend in ("hip_gfx1151", "cuda_sm86", "cpu_reference"):
        assert resolve(
            backend=unvalidated_backend,
            layer=_KEY.layer,
            quant=_KEY.quant,
            variant=_KEY.variant,
            missing="none",
        ) is None

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
    assert registered == []


def test_wave10_signbit_source_preserves_control_and_exact_topology() -> None:
    source = _SOURCE.read_text()
    control = source.split(
        "__global__ void gguf_iq3_xxs_weighted_selected_down_k1024_wave10_kernel(",
        1,
    )[1].split("\n}\n", 1)[0]
    candidate = source.split(
        "__global__ void "
        "gguf_iq3_xxs_weighted_selected_down_k1024_wave10_signbit_kernel(",
        1,
    )[1].split("\n}\n", 1)[0]
    for body in (control, candidate):
        assert "const int route = threadIdx.x >> 5" in body
        assert "const int lane = threadIdx.x & 31" in body
        assert "float partials[BLOCKS]" in body
        assert "__shared__ uint16_t route_values[TOP_K]" in body
        assert "route_values[route] = float_to_bf16_bits(route_total)" in body
        assert body.count("__syncthreads()") == 1
        assert "float total = 0.0f" in body
        assert "total = fmaf(" in body
        assert "out[out_col] = float_to_bf16_bits(total)" in body
    assert "partials[block_idx] = iq3_xxs_group_dot(block, lane, x_values);" in control
    assert "iq3_xxs_group_dot_signbit(block, lane, x_values);" not in control
    assert (
        "partials[block_idx] = "
        "iq3_xxs_group_dot_signbit(block, lane, x_values);" in candidate
    )
    assert "hipengine_gguf_iq3_xxs_weighted_selected_down_k1024_wave10_" in source
    assert "wave10_signbit_bf16_bf16_out" in source


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("kind", "out_features", "exhaustive"),
    (("edges", 3, True), ("signed_zero", 17, False), ("random", 19, False)),
)
def test_wave10_signbit_is_bit_exact_to_retained_wave10(
    hip_context,
    kind: str,
    out_features: int,
    exhaustive: bool,
) -> None:
    runtime, iq_library = hip_context
    x_bits = _edge_x(kind)
    selected = np.asarray([2, 0, 1, -1, 2, 1, 0, 2, 1, 0], dtype=np.int64)
    routing = np.asarray(
        (0.0, -0.0, 1.0, -1.0, 0.61, 0.49, 0.37, 0.13, 0.0078125, -0.00390625),
        dtype=np.float32,
    )
    qweight = _make_iq3_weight(3, out_features, exhaustive=exhaustive)
    control, candidate = _run_pair(
        x_bits,
        selected,
        routing,
        qweight,
        runtime=runtime,
        iq_library=iq_library,
    )
    np.testing.assert_array_equal(candidate, control)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_wave10_signbit_passes_independent_iq3_cpu_gate(hip_context) -> None:
    runtime, iq_library = hip_context
    x_bits = _edge_x("random")
    selected = np.asarray([2, 0, 1, 2, 1, 0, 2, 1, 0, 2], dtype=np.int64)
    routing = np.asarray(
        (0.61, 0.49, 0.37, 0.29, 0.23, 0.19, 0.13, 0.09, 0.07, 0.03),
        dtype=np.float32,
    )
    qweight = _make_iq3_weight(3, 23, exhaustive=False)
    control, candidate = _run_pair(
        x_bits,
        selected,
        routing,
        qweight,
        runtime=runtime,
        iq_library=iq_library,
    )
    np.testing.assert_array_equal(candidate, control)
    reference = _cpu_reference(x_bits, selected, routing, qweight)
    quality = evaluate_logits(reference, bf16_to_float32(candidate))
    assert quality.kl_max <= 0.05
    assert quality.top1_agreement >= 0.90
