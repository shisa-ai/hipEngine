"""Decode correctness for raw selected IQ3_XXS/IQ4_XS MoE kernels.

The target Qwen3.6 Q3 model stores routed gate/up experts as IQ3_XXS (except
layer 39, which uses IQ4_XS) and routed down experts as IQ4_XS (except the
three existing Q6_K down layers).  These tests cover the production selected
ABI directly: BF16 ``x[x_rows, in_features]``, device ``selected[rows]``, raw
rank-3 expert weights, and BF16 row-major output.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kernels.cpu_reference import gguf_quant_gemv
from hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_gemv import (
    build_gguf_iq_selected_gemv,
    gguf_iq3_xxs_selected_fused_gate_up_silu_bf16_bf16_out,
    gguf_iq4_xs_selected_fused_gate_up_silu_bf16_bf16_out,
    gguf_iq4_xs_selected_gemv_bf16_bf16_out,
    plan_gguf_iq_selected_gemv_build,
    register_gguf_iq_selected_gemv_kernels,
)
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import GGMLQuantizationType
from tests._gguf_synthetic_weights import make_iq3_xxs_weight, make_iq4_xs_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


@pytest.fixture(scope="module")
def iq_selected_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    return build_gguf_iq_selected_gemv(load=True)


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    u32 = f32.view(np.uint32).copy()
    nan_mask = np.isnan(f32)
    lsb = (u32 >> 16) & 1
    rounded = ((u32 + 0x7FFF + lsb) >> 16).astype(np.uint16)
    rounded[nan_mask] = 0x7FC0
    return rounded.reshape(f32.shape)


def _bf16_u16_to_f32(arr: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(u16.shape).copy()


def _stack_experts(builder, out_features: int, in_features: int, num_experts: int, shift: int) -> np.ndarray:
    base = builder(out_features, in_features)
    return np.stack(
        [np.roll(base, shift=shift + expert, axis=0) for expert in range(num_experts)],
        axis=0,
    )


def _x_row(row: int, x_rows: int, rows: int) -> int:
    return 0 if x_rows == 1 else row // (rows // x_rows)


def _expected_gate_up_silu(
    x_ref: np.ndarray,
    selected: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    qtype: GGMLQuantizationType,
) -> np.ndarray:
    rows = selected.size
    out = np.empty((rows, gate.shape[1]), dtype=np.float32)
    for row, expert in enumerate(selected.tolist()):
        xr = _x_row(row, x_ref.shape[0], rows)
        gate_row = gguf_quant_gemv(x_ref[xr : xr + 1], gate[expert], qtype)[0]
        up_row = gguf_quant_gemv(x_ref[xr : xr + 1], up[expert], qtype)[0]
        # Match the existing BF16 pair -> SiLU chain: fused decode must retain
        # the materialized gate/up rounding boundary before applying SiLU.
        gate_row = _bf16_u16_to_f32(_f32_to_bf16_u16(gate_row))
        up_row = _bf16_u16_to_f32(_f32_to_bf16_u16(up_row))
        out[row] = (gate_row / (np.float32(1.0) + np.exp(-gate_row))) * up_row
    return out


def _expected_selected(
    x_ref: np.ndarray,
    selected: np.ndarray,
    weight: np.ndarray,
    qtype: GGMLQuantizationType,
) -> np.ndarray:
    rows = selected.size
    out = np.empty((rows, weight.shape[1]), dtype=np.float32)
    for row, expert in enumerate(selected.tolist()):
        xr = _x_row(row, x_ref.shape[0], rows)
        out[row] = gguf_quant_gemv(x_ref[xr : xr + 1], weight[expert], qtype)[0]
    return out


def _copy_to_device(array: np.ndarray):
    buf = malloc(array.nbytes)
    copy_host_to_device(buf, host_array_ptr(array), array.nbytes)
    return buf


def _run_gate_up(
    fn,
    x: np.ndarray,
    selected: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    library,
) -> np.ndarray:
    rows = selected.size
    out = np.zeros((rows, gate.shape[1]), dtype=np.uint16)
    buffers = [
        _copy_to_device(x),
        _copy_to_device(selected),
        _copy_to_device(gate),
        _copy_to_device(up),
        malloc(out.nbytes),
    ]
    try:
        fn(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            buffers[3].ptr,
            buffers[4].ptr,
            x.shape[0],
            rows,
            gate.shape[0],
            x.shape[1],
            gate.shape[1],
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), buffers[4], out.nbytes)
        return out
    finally:
        for buf in buffers:
            free(buf)


def _run_selected(
    fn,
    x: np.ndarray,
    selected: np.ndarray,
    weight: np.ndarray,
    library,
) -> np.ndarray:
    rows = selected.size
    out = np.zeros((rows, weight.shape[1]), dtype=np.uint16)
    buffers = [
        _copy_to_device(x),
        _copy_to_device(selected),
        _copy_to_device(weight),
        malloc(out.nbytes),
    ]
    try:
        fn(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            buffers[3].ptr,
            x.shape[0],
            rows,
            weight.shape[0],
            x.shape[1],
            weight.shape[1],
            library=library,
        )
        copy_device_to_host(host_array_ptr(out), buffers[3], out.nbytes)
        return out
    finally:
        for buf in buffers:
            free(buf)


# ---------------------------------------------------------------------------
# No-GPU surface and dispatch contract.
# ---------------------------------------------------------------------------


def test_iq_selected_registry_keys_resolve() -> None:
    register_gguf_iq_selected_gemv_kernels()
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq3_xxs",
        variant="selected_fused_gate_up_silu_bf16_bf16_out",
    ) is gguf_iq3_xxs_selected_fused_gate_up_silu_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq4_xs",
        variant="selected_fused_gate_up_silu_bf16_bf16_out",
    ) is gguf_iq4_xs_selected_fused_gate_up_silu_bf16_bf16_out
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq4_xs",
        variant="selected_gemv_bf16_bf16_out",
    ) is gguf_iq4_xs_selected_gemv_bf16_bf16_out


def test_iq_selected_build_plan_is_dry_run_safe() -> None:
    plan = plan_gguf_iq_selected_gemv_build()
    assert plan.output_path.name == "gguf_iq_selected_gemv.so"
    assert "-mcumode" in plan.command
    assert any(flag.startswith("-DHIPENGINE_IQ_TABLES_REV=") for flag in plan.flags)


def test_iq_selected_wrappers_validate_shape_contract() -> None:
    with pytest.raises(ValueError, match="x_rows must be positive"):
        gguf_iq3_xxs_selected_fused_gate_up_silu_bf16_bf16_out(
            0, 0, 0, 0, 0, 0, 8, 256, 2048, 512
        )
    with pytest.raises(ValueError, match="rows must be divisible by x_rows"):
        gguf_iq4_xs_selected_fused_gate_up_silu_bf16_bf16_out(
            0, 0, 0, 0, 0, 2, 3, 256, 2048, 512
        )
    with pytest.raises(ValueError, match="in_features must be divisible by GGUF IQ block size 256"):
        gguf_iq4_xs_selected_gemv_bf16_bf16_out(
            0, 0, 0, 0, 1, 1, 256, 511, 2048
        )


# ---------------------------------------------------------------------------
# Device correctness against the canonical host dequant oracle.
# ---------------------------------------------------------------------------


_GATE_UP_CASES = [
    pytest.param(
        make_iq3_xxs_weight,
        GGMLQuantizationType.IQ3_XXS,
        gguf_iq3_xxs_selected_fused_gate_up_silu_bf16_bf16_out,
        id="IQ3_XXS",
    ),
    pytest.param(
        make_iq4_xs_weight,
        GGMLQuantizationType.IQ4_XS,
        gguf_iq4_xs_selected_fused_gate_up_silu_bf16_bf16_out,
        id="IQ4_XS",
    ),
]


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("builder,qtype,fn", _GATE_UP_CASES)
@pytest.mark.parametrize("in_features,out_features", [(256, 8), (2048, 512)])
def test_iq_selected_fused_gate_up_silu_matches_cpu_oracle(
    builder,
    qtype: GGMLQuantizationType,
    fn,
    in_features: int,
    out_features: int,
    iq_selected_library,
) -> None:
    selected = np.asarray([2, 0, 1, 2, 1, 0, 2, 1], dtype=np.int64)
    num_experts = 3
    gate = _stack_experts(builder, out_features, in_features, num_experts, shift=1)
    up = _stack_experts(builder, out_features, in_features, num_experts, shift=5)
    rng = np.random.default_rng(701 + int(qtype) * 17 + in_features)
    x_f32 = rng.normal(0.0, 0.2, size=(1, in_features)).astype(np.float32)
    x = _f32_to_bf16_u16(x_f32)
    x_ref = _bf16_u16_to_f32(x)

    actual = _bf16_u16_to_f32(_run_gate_up(fn, x, selected, gate, up, iq_selected_library))
    expected = _expected_gate_up_silu(x_ref, selected, gate, up, qtype)
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))

    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected_bf16, atol=5.0e-2, rtol=2.0e-2)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("builder,qtype,fn", _GATE_UP_CASES)
def test_iq_selected_fused_gate_up_silu_multirow_prefill_matches_cpu_oracle(
    builder,
    qtype: GGMLQuantizationType,
    fn,
    iq_selected_library,
) -> None:
    x_rows = 3
    selected = np.asarray(
        [4, 1, 3, 0, 2, 4, 1, 0] * x_rows,
        dtype=np.int64,
    )
    gate = _stack_experts(builder, 11, 512, 5, shift=2)
    up = _stack_experts(builder, 11, 512, 5, shift=7)
    rng = np.random.default_rng(2309 + int(qtype))
    x_f32 = rng.normal(0.0, 0.2, size=(x_rows, 512)).astype(np.float32)
    x = _f32_to_bf16_u16(x_f32)
    x_ref = _bf16_u16_to_f32(x)

    actual = _bf16_u16_to_f32(_run_gate_up(fn, x, selected, gate, up, iq_selected_library))
    expected = _expected_gate_up_silu(x_ref, selected, gate, up, qtype)
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))

    np.testing.assert_allclose(actual, expected_bf16, atol=5.0e-2, rtol=2.0e-2)


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize("in_features,out_features", [(256, 8), (512, 2048)])
def test_iq4_xs_selected_down_matches_cpu_oracle(
    in_features: int,
    out_features: int,
    iq_selected_library,
) -> None:
    selected = np.asarray([2, 0, 1, 2, 1, 0, 2, 1], dtype=np.int64)
    weight = _stack_experts(make_iq4_xs_weight, out_features, in_features, 3, shift=3)
    rng = np.random.default_rng(1901 + in_features + out_features)
    x_f32 = rng.normal(0.0, 0.15, size=(selected.size, in_features)).astype(np.float32)
    x = _f32_to_bf16_u16(x_f32)
    x_ref = _bf16_u16_to_f32(x)

    actual = _bf16_u16_to_f32(
        _run_selected(
            gguf_iq4_xs_selected_gemv_bf16_bf16_out,
            x,
            selected,
            weight,
            iq_selected_library,
        )
    )
    expected = _expected_selected(x_ref, selected, weight, GGMLQuantizationType.IQ4_XS)
    expected_bf16 = _bf16_u16_to_f32(_f32_to_bf16_u16(expected))

    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected_bf16, atol=2.5e-2, rtol=2.0e-2)
