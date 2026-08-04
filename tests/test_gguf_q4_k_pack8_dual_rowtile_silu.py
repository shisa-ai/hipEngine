"""Exact fused gate/up row reuse plus SiLU for resident Q4_K pack8 weights."""

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
from hipengine.kernels.cpu_reference import gguf_q4_k_pack8_gemv
from hipengine.kernels.hip_gfx1100.fused.paro_silu import (
    silu_mul_separate_out_bf16,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    build_gguf_q4_k_gemv,
    gguf_q4_k_pack8_dual_rowtile_silu_bf16_bf16_out,
    gguf_q4_k_pack8_rowtile_bf16_bf16_out,
)
from hipengine.kernels.registry import KernelKey, resolve
from hipengine.quant.gguf_q4_k import repack_gguf_q4_k_pack8
from tests.test_gguf_q4_k_gemv import make_q4_k_weight


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    words = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).copy()
    words += 0x7FFF + ((words >> 16) & 1)
    return (words >> 16).astype(np.uint16)


def _bf16_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    probs = np.exp(shifted, dtype=np.float64)
    return probs / np.sum(probs, axis=-1, keepdims=True)


def test_pack8_dual_rowtile_silu_registry_contract() -> None:
    key = KernelKey(
        "hip_gfx1100",
        "linear_pair_silu",
        "gguf_q4_k",
        "pack8_dual_rowtile_bf16_bf16_out",
    )
    assert resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    ) is gguf_q4_k_pack8_dual_rowtile_silu_bf16_bf16_out


def test_pack8_dual_rowtile_silu_wrapper_rejects_unsupported_launches() -> None:
    for rows in (1, 5):
        with pytest.raises(ValueError, match="rows must be 2, 3, or 4"):
            gguf_q4_k_pack8_dual_rowtile_silu_bf16_bf16_out(
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                rows,
                256,
                16,
            )
    with pytest.raises(ValueError, match="threads must be 0 or 64"):
        gguf_q4_k_pack8_dual_rowtile_silu_bf16_bf16_out(
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            4,
            256,
            16,
            threads=32,
        )


def _run_fused_chain(
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gate = repack_gguf_q4_k_pack8(make_q4_k_weight(out_features, in_features))
    up = repack_gguf_q4_k_pack8(make_q4_k_weight(out_features, in_features))
    # Make the two projections distinct without changing the packed layout.
    up.qweight[:] ^= np.int32(0x13579BDF)
    rng = np.random.default_rng(0xD28 + rows * 17 + in_features + out_features)
    x_bits = _bf16_bits(
        rng.normal(0.0, 0.15, size=(rows, in_features)).astype(np.float32)
    )
    control = np.empty((rows, out_features), dtype=np.uint16)
    candidate = np.empty_like(control)
    arrays = (
        x_bits,
        gate.qweight,
        gate.scales,
        gate.mins,
        up.qweight,
        up.scales,
        up.mins,
    )
    inputs = [malloc(array.nbytes) for array in arrays]
    output_nbytes = control.nbytes
    gate_d = malloc(output_nbytes)
    up_d = malloc(output_nbytes)
    control_d = malloc(output_nbytes)
    candidate_d = malloc(output_nbytes)
    library = build_gguf_q4_k_gemv(load=True)
    try:
        for array, allocation in zip(arrays, inputs, strict=True):
            copy_host_to_device(allocation, host_array_ptr(array), array.nbytes)
        x_d, gate_q_d, gate_s_d, gate_m_d, up_q_d, up_s_d, up_m_d = inputs
        gguf_q4_k_pack8_rowtile_bf16_bf16_out(
            x_d.ptr,
            gate_q_d.ptr,
            gate_s_d.ptr,
            gate_m_d.ptr,
            gate_d.ptr,
            rows,
            in_features,
            out_features,
            library=library,
        )
        gguf_q4_k_pack8_rowtile_bf16_bf16_out(
            x_d.ptr,
            up_q_d.ptr,
            up_s_d.ptr,
            up_m_d.ptr,
            up_d.ptr,
            rows,
            in_features,
            out_features,
            library=library,
        )
        silu_mul_separate_out_bf16(
            gate_d.ptr,
            up_d.ptr,
            control_d.ptr,
            rows=rows,
            features=out_features,
        )
        gguf_q4_k_pack8_dual_rowtile_silu_bf16_bf16_out(
            x_d.ptr,
            gate_q_d.ptr,
            gate_s_d.ptr,
            gate_m_d.ptr,
            up_q_d.ptr,
            up_s_d.ptr,
            up_m_d.ptr,
            candidate_d.ptr,
            rows,
            in_features,
            out_features,
            library=library,
        )
        copy_device_to_host(host_array_ptr(control), control_d, control.nbytes)
        copy_device_to_host(host_array_ptr(candidate), candidate_d, candidate.nbytes)
    finally:
        for allocation in (candidate_d, control_d, up_d, gate_d, *inputs):
            free(allocation)

    gate_cpu = gguf_q4_k_pack8_gemv(
        _bf16_f32(x_bits), gate.qweight, gate.scales, gate.mins
    )
    up_cpu = gguf_q4_k_pack8_gemv(
        _bf16_f32(x_bits), up.qweight, up.scales, up.mins
    )
    gate_cpu = _bf16_f32(_bf16_bits(gate_cpu))
    up_cpu = _bf16_f32(_bf16_bits(up_cpu))
    with np.errstate(over="ignore"):
        cpu = gate_cpu * (1.0 / (1.0 + np.exp(-gate_cpu))) * up_cpu
    return control, candidate, cpu


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("rows", (2, 3, 4))
@pytest.mark.parametrize("in_features,out_features", ((256, 16), (512, 64)))
def test_pack8_dual_rowtile_silu_is_bit_exact_and_passes_cpu_gate(
    rows: int,
    in_features: int,
    out_features: int,
) -> None:
    control, candidate, cpu = _run_fused_chain(
        rows=rows,
        in_features=in_features,
        out_features=out_features,
    )
    np.testing.assert_array_equal(candidate, control)

    gpu = _bf16_f32(candidate)
    np.testing.assert_allclose(gpu, cpu, rtol=0.04, atol=0.04)
    p = _softmax(cpu)
    q = _softmax(gpu)
    kl = np.sum(p * (np.log(p + 1.0e-30) - np.log(q + 1.0e-30)), axis=-1)
    top1 = np.mean(np.argmax(cpu, axis=-1) == np.argmax(gpu, axis=-1))
    assert float(np.max(kl)) <= 0.05
    assert float(top1) >= 0.90
