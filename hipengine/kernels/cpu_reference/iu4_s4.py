"""Independent NumPy oracle for the experimental U4 x S4 gate/up boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hipengine.quant.iu4_s4 import bf16_bits_to_f32, f32_to_bf16_bits


@dataclass(frozen=True)
class U4HadamardRows:
    quantized: np.ndarray
    packed_words: np.ndarray
    scales: np.ndarray
    zero_points: np.ndarray
    transformed: np.ndarray


def _hadamard_signs(cols: int, seed: int) -> np.ndarray:
    value = np.arange(cols, dtype=np.uint32) + np.uint32(seed)
    value ^= value >> np.uint32(16)
    value *= np.uint32(0x7FEB352D)
    value ^= value >> np.uint32(15)
    value *= np.uint32(0x846CA68B)
    value ^= value >> np.uint32(16)
    return np.where(value & np.uint32(1), np.float32(-1.0), np.float32(1.0))


def block_hadamard_f32(values: object, *, seed: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] <= 0 or source.shape[1] % 1024:
        raise ValueError("Hadamard source must be rank-2 with K divisible by 1024")
    rows, cols = source.shape
    blocks = cols // 1024
    transformed = np.ascontiguousarray(
        source * _hadamard_signs(cols, seed)[None, :], dtype=np.float32
    ).reshape(rows, blocks, 1024)
    for stride in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        groups = transformed.reshape(rows, blocks, 1024 // (2 * stride), 2 * stride)
        a = groups[..., :stride].copy()
        b = groups[..., stride:].copy()
        groups[..., :stride] = a + b
        groups[..., stride:] = a - b
    transformed *= np.float32(1.0 / 32.0)
    return np.ascontiguousarray(transformed.reshape(rows, cols))


def _quantize_hadamard_rows(values: np.ndarray, *, seed: int) -> U4HadamardRows:
    transformed = block_hadamard_f32(values, seed=seed)
    lo = transformed.min(axis=1)
    hi = transformed.max(axis=1)
    scales = ((hi - lo) * np.float32(1.0 / 15.0)).astype(np.float32)
    scales = np.where(scales > 0.0, scales, np.float32(1.0)).astype(np.float32)
    zeros = np.rint(-lo / scales).clip(0, 15).astype(np.int32)
    quantized = np.rint(transformed / scales[:, None]).astype(np.int32)
    quantized = (quantized + zeros[:, None]).clip(0, 15).astype(np.uint8)
    words = quantized.reshape(quantized.shape[0], -1, 8).astype(np.uint32)
    shifts = (np.arange(8, dtype=np.uint32) * np.uint32(4))[None, None, :]
    packed = np.bitwise_or.reduce(words << shifts, axis=2).astype(np.uint32)
    return U4HadamardRows(
        quantized=np.ascontiguousarray(quantized),
        packed_words=np.ascontiguousarray(packed),
        scales=np.ascontiguousarray(scales),
        zero_points=np.ascontiguousarray(zeros),
        transformed=transformed,
    )


def quantize_u4_hadamard_bf16(values: object, *, seed: int) -> U4HadamardRows:
    bits = np.asarray(values)
    if bits.dtype != np.uint16 or bits.ndim != 2:
        raise ValueError("Hadamard U4 source must be rank-2 BF16 bits")
    return _quantize_hadamard_rows(bf16_bits_to_f32(bits), seed=seed)


def quantize_u4_swiglu_hadamard_bf16(
    gate_up_values: object,
    *,
    width: int,
    seed: int,
) -> U4HadamardRows:
    bits = np.asarray(gate_up_values)
    if (
        bits.dtype != np.uint16
        or bits.ndim != 2
        or width <= 0
        or bits.shape[1] != 2 * width
    ):
        raise ValueError("SwiGLU Hadamard source must be BF16 [rows, 2*width]")
    values = bf16_bits_to_f32(bits)
    gate = values[:, :width]
    up = values[:, width:]
    swiglu = (gate / (np.float32(1.0) + np.exp(-gate))) * up
    return _quantize_hadamard_rows(swiglu.astype(np.float32), seed=seed)


def iu4_s4_corrected_i32(
    q_activations: object,
    activation_zero_points: object,
    q_weights: object,
    weight_sums: object,
) -> np.ndarray:
    activations = np.asarray(q_activations, dtype=np.int32)
    zeros = np.asarray(activation_zero_points, dtype=np.int32)
    weights = np.asarray(q_weights, dtype=np.int32)
    sums = np.asarray(weight_sums, dtype=np.int32)
    if activations.ndim != 2 or weights.ndim != 2:
        raise ValueError("U4 activations and S4 weights must be rank-2")
    if activations.shape[1] != weights.shape[1]:
        raise ValueError("U4/S4 K dimensions differ")
    if zeros.shape != (activations.shape[0],) or sums.shape != (weights.shape[0],):
        raise ValueError("correction metadata shapes differ")
    return np.ascontiguousarray(
        activations @ weights.T - zeros[:, None] * sums[None, :],
        dtype=np.int32,
    )


def iu4_s4_gate_up_silu_bf16(
    q_activations: object,
    activation_scales: object,
    activation_zero_points: object,
    q_gate: object,
    gate_scales: object,
    gate_sums: object,
    q_up: object,
    up_scales: object,
    up_sums: object,
) -> np.ndarray:
    activation_scales = np.asarray(activation_scales, dtype=np.float32)
    gate_scales = np.asarray(gate_scales, dtype=np.float32)
    up_scales = np.asarray(up_scales, dtype=np.float32)
    gate_i32 = iu4_s4_corrected_i32(
        q_activations, activation_zero_points, q_gate, gate_sums
    )
    up_i32 = iu4_s4_corrected_i32(
        q_activations, activation_zero_points, q_up, up_sums
    )
    if activation_scales.shape != (gate_i32.shape[0],):
        raise ValueError("activation scales must have one value per row")
    if gate_scales.shape != (gate_i32.shape[1],) or up_scales.shape != (up_i32.shape[1],):
        raise ValueError("weight scales must have one value per output")
    gate_bits = f32_to_bf16_bits(
        gate_i32.astype(np.float32)
        * activation_scales[:, None]
        * gate_scales[None, :]
    )
    up_bits = f32_to_bf16_bits(
        up_i32.astype(np.float32)
        * activation_scales[:, None]
        * up_scales[None, :]
    )
    gate = bf16_bits_to_f32(gate_bits)
    up = bf16_bits_to_f32(up_bits)
    return f32_to_bf16_bits((gate / (np.float32(1.0) + np.exp(-gate))) * up)


__all__ = [
    "U4HadamardRows",
    "block_hadamard_f32",
    "iu4_s4_corrected_i32",
    "iu4_s4_gate_up_silu_bf16",
    "quantize_u4_hadamard_bf16",
    "quantize_u4_swiglu_hadamard_bf16",
]
