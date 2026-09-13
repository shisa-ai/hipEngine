"""Exact RED/GREEN gate for Laguna's fixed-local64 IQ2 grid64 reduction."""

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
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import GGMLQuantizationType, bf16_to_float32, dequantize_gguf_data

_VARIANT = "selected_dual_silu_gemv_decode_tile2_grid64_local64_reduce_bf16_bf16_out"
_WRAPPER = "gguf_iq2_xs_selected_dual_silu_gemv_tile2_grid64_local64_reduce_bf16_bf16_out"
_KEY = KernelKey("hip_gfx1100", "moe_linear", "gguf_iq2_xs", _VARIANT)
_CONTROL = gguf_iq_gemv.gguf_iq2_xs_selected_dual_silu_gemv_tile2_grid64_bf16_bf16_out
_SOURCE = Path(gguf_iq_gemv.__file__).with_suffix(".hip")
_QK_K = 256
_BLOCK_BYTES = 74
_THREADS = 64


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


def _make_weight(
    num_experts: int,
    out_features: int,
    in_features: int,
    *,
    seed: int,
) -> np.ndarray:
    assert in_features % _QK_K == 0
    blocks = in_features // _QK_K
    rng = np.random.default_rng(seed)
    out = np.empty(
        (num_experts, out_features, blocks * _BLOCK_BYTES), dtype=np.uint8
    )
    view = out.reshape(num_experts, out_features, blocks, _BLOCK_BYTES)
    scale_values = np.asarray(
        (0.0, -0.0009765625, 0.0009765625, -0.00390625, 0.0078125),
        dtype=np.float16,
    )
    scales = np.resize(scale_values, view[..., :2].shape[:-1])
    view[..., :2] = scales[..., None].view(np.uint8).reshape(view[..., :2].shape)
    view[..., 2:66] = rng.integers(0, 256, size=view[..., 2:66].shape, dtype=np.uint8)
    view[..., 66:74] = rng.integers(0, 256, size=view[..., 66:74].shape, dtype=np.uint8)
    return out


def _make_x(in_features: int, kind: str) -> np.ndarray:
    rng = np.random.default_rng(2026072701 + in_features + sum(kind.encode()))
    if kind == "random":
        return float_array_to_bf16_bits(
            rng.normal(0.0, 0.35, size=in_features).astype(np.float32)
        )
    if kind == "bf16_edges":
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
        return np.resize(edge_bits, in_features).copy()
    if kind == "signed_zero":
        result = np.zeros(in_features, dtype=np.uint16)
        result[1::2] = np.uint16(0x8000)
        return result
    raise AssertionError(kind)


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


def _launch_pair(
    x_bits: np.ndarray,
    selected: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    *,
    runtime,
    library,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = _candidate()
    assert callable(candidate), "fixed-local64 IQ2 reduction wrapper must be admitted"
    selected = np.ascontiguousarray(selected, dtype=np.int64)
    out_shape = (selected.size, gate.shape[1])
    buffers: list = []
    try:
        x_d = _upload(buffers, x_bits, runtime=runtime)
        selected_d = _upload(buffers, selected, runtime=runtime)
        gate_d = _upload(buffers, gate, runtime=runtime)
        up_d = _upload(buffers, up, runtime=runtime)
        control_d = malloc(int(np.prod(out_shape)) * 2, runtime=runtime)
        candidate_d = malloc(int(np.prod(out_shape)) * 2, runtime=runtime)
        buffers.extend((control_d, candidate_d))
        common = dict(
            x_rows=1,
            rows=selected.size,
            num_experts=gate.shape[0],
            in_features=x_bits.size,
            out_features=gate.shape[1],
            threads=_THREADS,
            library=library,
            runtime=runtime,
        )
        _CONTROL(
            x_d.ptr,
            selected_d.ptr,
            gate_d.ptr,
            up_d.ptr,
            control_d.ptr,
            **common,
        )
        candidate(
            x_d.ptr,
            selected_d.ptr,
            gate_d.ptr,
            up_d.ptr,
            candidate_d.ptr,
            **common,
        )
        runtime.device_synchronize()
        return (
            _download(control_d, out_shape, runtime=runtime),
            _download(candidate_d, out_shape, runtime=runtime),
        )
    finally:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)


def _cpu_reference(
    x_bits: np.ndarray,
    selected: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    x = bf16_to_float32(x_bits).astype(np.float32)
    gate_f32 = dequantize_gguf_data(gate, GGMLQuantizationType.IQ2_XS)
    up_f32 = dequantize_gguf_data(up, GGMLQuantizationType.IQ2_XS)
    result = np.zeros((selected.size, gate.shape[1]), dtype=np.float32)
    for row, expert_value in enumerate(selected):
        expert = int(expert_value)
        if not 0 <= expert < gate.shape[0]:
            continue
        gate_bits = float_array_to_bf16_bits(
            (gate_f32[expert] @ x).astype(np.float32)
        )
        up_bits = float_array_to_bf16_bits((up_f32[expert] @ x).astype(np.float32))
        gate_value = bf16_to_float32(gate_bits)
        up_value = bf16_to_float32(up_bits)
        result[row] = (
            gate_value
            * (np.float32(1.0) / (np.float32(1.0) + np.exp(-gate_value)))
            * up_value
        ).astype(np.float32)
    return result


def test_local64_reduction_registry_package_export_and_preload_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate()
    assert callable(candidate), "fixed-local64 IQ2 reduction wrapper must be admitted"
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
        quant="gguf_iq2_xs",
        variant="selected_dual_silu_gemv_decode_tile2_grid64_bf16_bf16_out",
    ) is _CONTROL

    build_calls: list[bool] = []

    def fail_build(*, load: bool = True, **_kwargs):
        build_calls.append(load)
        raise AssertionError("validation must reject before library load")

    monkeypatch.setattr(gguf_iq_gemv, "build_gguf_iq_gemv", fail_build)
    pointers = (1,) * 5
    common = dict(rows=10, num_experts=256, in_features=3072, out_features=1024)
    with pytest.raises(ValueError, match="x_rows must be exactly 1"):
        candidate(*pointers, x_rows=2, **common)
    with pytest.raises(ValueError, match="rows must be positive"):
        candidate(*pointers, x_rows=1, **(common | {"rows": 0}))
    with pytest.raises(ValueError, match="num_experts must be positive"):
        candidate(*pointers, x_rows=1, **(common | {"num_experts": 0}))
    with pytest.raises(ValueError, match="divisible by 256"):
        candidate(*pointers, x_rows=1, **(common | {"in_features": 3071}))
    with pytest.raises(ValueError, match="out_features must be positive"):
        candidate(*pointers, x_rows=1, **(common | {"out_features": 0}))
    with pytest.raises(ValueError, match="threads must be exactly 64"):
        candidate(*pointers, x_rows=1, threads=128, **common)
    pointer_names = (
        "x_ptr",
        "selected_ptr",
        "gate_weight_ptr",
        "up_weight_ptr",
        "out_ptr",
    )
    for index, pointer_name in enumerate(pointer_names):
        null_pointers = list(pointers)
        null_pointers[index] = 0
        with pytest.raises(ValueError, match=rf"{pointer_name} must be non-zero"):
            candidate(*null_pointers, x_rows=1, **common)
    assert build_calls == []


def test_local64_reduction_key_is_excluded_from_unvalidated_backends(
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
    assert _KEY not in registered


def test_local64_reduction_source_preserves_generic_reducer_and_exact_tree() -> None:
    source = _SOURCE.read_text()
    generic = source.split("__device__ inline void reduce_block_quad(", 1)[1].split(
        "\n}\n", 1
    )[0]
    assert "__shfl_down" in generic
    assert "const int waves = (blockDim.x + 31) >> 5" in generic
    assert "reduce_block_quad_local64_dpp" in source
    candidate = source.split(
        "__device__ inline void reduce_block_quad_local64_dpp(", 1
    )[1].split("\n}\n", 1)[0]
    assert candidate.count("permlanex16_f32") == 4
    for control in ("0x108", "0x104", "0x102", "0x101"):
        assert f"IQ2_DPP_REDUCE_STEP({control})" in candidate
    assert "wave_sums_a[2]" in candidate
    assert "const int waves" not in candidate
    assert source.count("reduce_block_quad_local64_dpp(") == 2


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    ("in_features", "out_features", "rows", "kind"),
    (
        (256, 17, 1, "bf16_edges"),
        (1024, 18, 10, "signed_zero"),
        (3072, 23, 10, "random"),
    ),
)
def test_local64_reduction_is_bf16_bit_exact_to_retained_grid64(
    hip_context,
    in_features: int,
    out_features: int,
    rows: int,
    kind: str,
) -> None:
    runtime, library = hip_context
    x_bits = _make_x(in_features, kind)
    selected = np.resize(np.asarray([2, 0, 1, -1, 2], dtype=np.int64), rows)
    gate = _make_weight(3, out_features, in_features, seed=0x6420 + in_features)
    up = _make_weight(3, out_features, in_features, seed=0x6430 + in_features)
    control, candidate = _launch_pair(
        x_bits,
        selected,
        gate,
        up,
        runtime=runtime,
        library=library,
    )
    np.testing.assert_array_equal(candidate, control)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_local64_reduction_passes_independent_iq2_cpu_gate(hip_context) -> None:
    runtime, library = hip_context
    in_features = 1024
    out_features = 19
    x_bits = _make_x(in_features, "random")
    selected = np.asarray([2, 0, 1, -1, 2, 1, 0, 2, 1, 0], dtype=np.int64)
    gate = _make_weight(3, out_features, in_features, seed=0x6441)
    up = _make_weight(3, out_features, in_features, seed=0x6442)
    control, candidate = _launch_pair(
        x_bits,
        selected,
        gate,
        up,
        runtime=runtime,
        library=library,
    )
    np.testing.assert_array_equal(candidate, control)
    reference = _cpu_reference(x_bits, selected, gate, up)
    candidate_f32 = bf16_to_float32(candidate)
    quality = evaluate_logits(reference, candidate_f32)
    assert quality.kl_max <= 0.05
    assert quality.top1_agreement >= 0.90
