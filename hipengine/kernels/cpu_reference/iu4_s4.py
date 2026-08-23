"""Independent NumPy oracle for the experimental U4 x S4 gate/up boundary."""

from __future__ import annotations

import numpy as np

from hipengine.quant.iu4_s4 import bf16_bits_to_f32, f32_to_bf16_bits


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


__all__ = ["iu4_s4_corrected_i32", "iu4_s4_gate_up_silu_bf16"]
