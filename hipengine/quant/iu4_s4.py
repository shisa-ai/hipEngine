"""Host oracles for the experimental GFX11 U4 x S4 sidecar format.

The format is deliberately simple: K-contiguous low-nibble-first packed values,
one FP32 scale and one I32 sum per output channel, and asymmetric U4
activations with one FP32 scale/I32 zero point per physical row.  It is a T3
research representation, not an implicit replacement for GGUF Q4_K.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hipengine.quant.registry import register_quant


@dataclass(frozen=True)
class IU4S4SidecarQuant:
    """Experimental T3 signed-I4 companion plus dynamic asymmetric-U4 rows."""

    name: str = "iu4_s4_sidecar_v1"
    weight_storage: str = "s4_wmma_n16_k32pair_per_output_scale_sum"
    activation_preprocess: str = "dynamic_u4_asymmetric_per_row"
    compute_dtype: str = "iu4_dot8_or_wmma_i32_accum_bf16_out"
    scale_granularity: str = "one_weight_scale_per_output_one_activation_scale_per_row"
    calibration_artifact: str = "unqualified_t3_research"
    kernel_family: str = "gfx1151_iu4_s4_sidecar"


IU4_S4_SIDECAR = register_quant(IU4S4SidecarQuant())


def f32_to_bf16_bits(values: object) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return np.ascontiguousarray((rounded >> np.uint32(16)).astype(np.uint16))


def bf16_bits_to_f32(values: object) -> np.ndarray:
    bits = np.asarray(values, dtype=np.uint16)
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32).copy()


def _validate_quantized_matrix(values: object, *, low: int, high: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] <= 0 or array.shape[1] % 2:
        raise ValueError(f"{name} must be rank 2 with a positive even K dimension")
    as_i32 = array.astype(np.int32)
    if np.any(as_i32 < low) or np.any(as_i32 > high):
        raise ValueError(f"{name} values must be in [{low}, {high}]")
    return as_i32


def _pack_nibbles(values: np.ndarray) -> np.ndarray:
    unsigned = (values.astype(np.int32) & 0xF).astype(np.uint8)
    return np.ascontiguousarray(unsigned[:, 0::2] | (unsigned[:, 1::2] << np.uint8(4)))


def unpack_u4(packed: object) -> np.ndarray:
    values = np.asarray(packed, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] <= 0:
        raise ValueError("packed U4 values must be a non-empty rank-2 array")
    result = np.empty((values.shape[0], values.shape[1] * 2), dtype=np.uint8)
    result[:, 0::2] = values & np.uint8(0xF)
    result[:, 1::2] = values >> np.uint8(4)
    return result


def unpack_s4(packed: object) -> np.ndarray:
    unsigned = unpack_u4(packed).astype(np.int16)
    return np.where(unsigned < 8, unsigned, unsigned - 16).astype(np.int8)


def pack_s4_wmma_tiles(sidecar: "S4Sidecar") -> np.ndarray:
    """Reorder row-major S4 bytes into coalesced [N16,K32,N,16] tiles."""

    if sidecar.out_features % 16 or sidecar.in_features % 32:
        raise ValueError("S4 WMMA tiles require N divisible by 16 and K by 32")
    return np.ascontiguousarray(
        sidecar.packed.reshape(
            sidecar.out_features // 16,
            16,
            sidecar.in_features // 32,
            16,
        ).transpose(0, 2, 1, 3)
    )


def unpack_u4_wmma_tiles(packed: object, *, rows: int, hidden: int) -> np.ndarray:
    """Return logical U4 values from padded [M16,K32,M,16] activation tiles."""

    if rows <= 0 or hidden <= 0 or hidden % 32:
        raise ValueError("U4 WMMA tiles require positive rows and K divisible by 32")
    padded_rows = ((rows + 15) // 16) * 16
    tiles = np.asarray(packed, dtype=np.uint8)
    expected = (padded_rows // 16, hidden // 32, 16, 16)
    if tiles.shape != expected:
        raise ValueError(f"U4 WMMA tile shape must be {expected}, got {tiles.shape}")
    row_packed = np.ascontiguousarray(
        tiles.transpose(0, 2, 1, 3).reshape(padded_rows, hidden // 2)
    )
    return unpack_u4(row_packed[:rows])


@dataclass(frozen=True)
class S4Sidecar:
    packed: np.ndarray
    scales: np.ndarray
    sums: np.ndarray

    def __post_init__(self) -> None:
        packed = np.ascontiguousarray(self.packed, dtype=np.uint8)
        scales = np.ascontiguousarray(self.scales, dtype=np.float32)
        sums = np.ascontiguousarray(self.sums, dtype=np.int32)
        if packed.ndim != 2 or packed.shape[1] <= 0:
            raise ValueError("S4 packed weights must be a non-empty rank-2 array")
        if scales.shape != (packed.shape[0],) or sums.shape != (packed.shape[0],):
            raise ValueError("S4 scales and sums must have one value per output row")
        if not np.isfinite(scales).all() or np.any(scales <= 0.0):
            raise ValueError("S4 scales must be finite and positive")
        object.__setattr__(self, "packed", packed)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "sums", sums)

    @classmethod
    def from_quantized(cls, values: object, *, scales: object) -> "S4Sidecar":
        q = _validate_quantized_matrix(values, low=-8, high=7, name="S4 weights")
        scale_array = np.ascontiguousarray(scales, dtype=np.float32)
        if scale_array.shape != (q.shape[0],):
            raise ValueError("S4 scales must have one value per output row")
        return cls(
            packed=_pack_nibbles(q),
            scales=scale_array,
            sums=np.ascontiguousarray(q.sum(axis=1, dtype=np.int32)),
        )

    @property
    def out_features(self) -> int:
        return int(self.packed.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.packed.shape[1] * 2)

    @property
    def nbytes(self) -> int:
        return int(self.packed.nbytes + self.scales.nbytes + self.sums.nbytes)


@dataclass(frozen=True)
class U4Rows:
    packed: np.ndarray
    scales: np.ndarray
    zero_points: np.ndarray

    def __post_init__(self) -> None:
        packed = np.ascontiguousarray(self.packed, dtype=np.uint8)
        scales = np.ascontiguousarray(self.scales, dtype=np.float32)
        zero_points = np.ascontiguousarray(self.zero_points, dtype=np.int32)
        if packed.ndim != 2 or packed.shape[1] <= 0:
            raise ValueError("U4 packed rows must be a non-empty rank-2 array")
        if scales.shape != (packed.shape[0],) or zero_points.shape != (packed.shape[0],):
            raise ValueError("U4 scales and zero points must have one value per row")
        if not np.isfinite(scales).all() or np.any(scales <= 0.0):
            raise ValueError("U4 scales must be finite and positive")
        if np.any(zero_points < 0) or np.any(zero_points > 15):
            raise ValueError("U4 zero points must be in [0, 15]")
        object.__setattr__(self, "packed", packed)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "zero_points", zero_points)

    @classmethod
    def from_quantized(
        cls,
        values: object,
        *,
        scales: object,
        zero_points: object,
    ) -> "U4Rows":
        q = _validate_quantized_matrix(values, low=0, high=15, name="U4 activations")
        return cls(
            packed=_pack_nibbles(q),
            scales=np.ascontiguousarray(scales, dtype=np.float32),
            zero_points=np.ascontiguousarray(zero_points, dtype=np.int32),
        )

    @property
    def rows(self) -> int:
        return int(self.packed.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.packed.shape[1] * 2)


def quantize_s4_per_output(weights: object) -> S4Sidecar:
    values = np.asarray(weights, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] <= 0 or values.shape[1] % 2:
        raise ValueError("weights must be rank 2 with a positive even K dimension")
    if not np.isfinite(values).all():
        raise ValueError("weights must be finite")
    negative_scale = np.maximum(-values.min(axis=1), 0.0) / np.float32(8.0)
    positive_scale = np.maximum(values.max(axis=1), 0.0) / np.float32(7.0)
    scales = np.maximum(negative_scale, positive_scale).astype(np.float32)
    scales = np.where(scales > 0.0, scales, np.float32(1.0)).astype(np.float32)
    quantized = np.rint(values / scales[:, None]).clip(-8, 7).astype(np.int8)
    return S4Sidecar.from_quantized(quantized, scales=scales)


def quantize_u4_per_row(bf16_values: object) -> U4Rows:
    bits = np.asarray(bf16_values)
    if bits.dtype != np.uint16 or bits.ndim != 2 or bits.shape[1] <= 0 or bits.shape[1] % 2:
        raise ValueError("U4 source must be rank-2 BF16 bits with a positive even K")
    values = bf16_bits_to_f32(bits)
    if not np.isfinite(values).all():
        raise ValueError("U4 source must be finite")
    lo = np.minimum(values.min(axis=1), np.float32(0.0))
    hi = np.maximum(values.max(axis=1), np.float32(0.0))
    scales = ((hi - lo) / np.float32(15.0)).astype(np.float32)
    scales = np.where(scales > 0.0, scales, np.float32(1.0)).astype(np.float32)
    zero_points = np.rint(-lo / scales).clip(0, 15).astype(np.int32)
    quantized = np.rint(values / scales[:, None]).astype(np.int32)
    quantized += zero_points[:, None]
    quantized = quantized.clip(0, 15).astype(np.uint8)
    return U4Rows.from_quantized(
        quantized,
        scales=scales,
        zero_points=zero_points,
    )


def iu4_s4_i32_reference(activations: U4Rows, weights: S4Sidecar) -> np.ndarray:
    if activations.in_features != weights.in_features:
        raise ValueError("U4 activation and S4 weight K dimensions differ")
    q_a = unpack_u4(activations.packed).astype(np.int32)
    q_w = unpack_s4(weights.packed).astype(np.int32)
    dots = q_a @ q_w.T
    return np.ascontiguousarray(
        dots - activations.zero_points[:, None] * weights.sums[None, :],
        dtype=np.int32,
    )


def iu4_s4_linear_reference(activations: U4Rows, weights: S4Sidecar) -> np.ndarray:
    corrected = iu4_s4_i32_reference(activations, weights).astype(np.float32)
    return corrected * activations.scales[:, None] * weights.scales[None, :]


def iu4_s4_gate_up_silu_reference(
    bf16_values: object,
    gate: S4Sidecar,
    up: S4Sidecar,
    *,
    return_projections: bool = False,
):
    if gate.packed.shape != up.packed.shape:
        raise ValueError("gate and up S4 sidecars must have identical shapes")
    activations = quantize_u4_per_row(bf16_values)
    gate_bits = f32_to_bf16_bits(iu4_s4_linear_reference(activations, gate))
    up_bits = f32_to_bf16_bits(iu4_s4_linear_reference(activations, up))
    gate_f32 = bf16_bits_to_f32(gate_bits)
    up_f32 = bf16_bits_to_f32(up_bits)
    output = f32_to_bf16_bits((gate_f32 / (np.float32(1.0) + np.exp(-gate_f32))) * up_f32)
    if return_projections:
        return output, gate_bits, up_bits
    return output


__all__ = [
    "IU4S4SidecarQuant",
    "IU4_S4_SIDECAR",
    "S4Sidecar",
    "U4Rows",
    "bf16_bits_to_f32",
    "f32_to_bf16_bits",
    "iu4_s4_gate_up_silu_reference",
    "iu4_s4_i32_reference",
    "iu4_s4_linear_reference",
    "pack_s4_wmma_tiles",
    "quantize_s4_per_output",
    "quantize_u4_per_row",
    "unpack_s4",
    "unpack_u4",
    "unpack_u4_wmma_tiles",
]
