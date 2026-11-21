"""Reference decoders for the original Inkling 2 Mini expert storage.

These decode weights only. They do not simulate activation quantization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hipengine.quant.registry import register_quant

INKLING2_MIXED_QUANT = "inkling2_mixed"


@dataclass(frozen=True)
class Inkling2MixedQuant:
    name: str = INKLING2_MIXED_QUANT
    weight_storage: str = "nvfp4_mxfp8_bf16"
    activation_preprocess: str = "none"
    compute_dtype: str = "fp32"
    scale_granularity: str = "block_and_expert"
    calibration_artifact: str = "none"
    kernel_family: str = "inkling2_mixed"


INKLING2_MIXED = register_quant(Inkling2MixedQuant())


def _e4m3_table() -> np.ndarray:
    codes = np.arange(256, dtype=np.uint8)
    exponent = ((codes >> 3) & 15).astype(np.int32)
    fraction = (codes & 7).astype(np.float32) / 8
    values = np.ldexp(np.where(exponent == 0, fraction, 1 + fraction), np.maximum(exponent, 1) - 7)
    values[(codes & 127) == 127] = np.nan
    return np.where(codes & 128, -values, values)


_E4M3 = _e4m3_table()
_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
_CODES = np.arange(256, dtype=np.uint8)
_E2M1_PAIRS = np.stack((_E2M1[_CODES & 15], _E2M1[_CODES >> 4]), axis=-1)
_UE8M0 = np.empty(256, dtype=np.float32)
_UE8M0[:-1] = np.ldexp(np.ones(255, dtype=np.float32), np.arange(255) - 127)
_UE8M0[-1] = np.nan


def _check_layout(weights: np.ndarray, scales: np.ndarray, block_bytes: int) -> None:
    if weights.dtype != np.uint8 or scales.dtype != np.uint8:
        raise ValueError("expert weight and scale storage must be raw uint8 arrays")
    if weights.ndim != 2 or scales.ndim != 2 or weights.shape != (scales.shape[0], scales.shape[1] * block_bytes):
        raise ValueError(f"incompatible expert layout: {weights.shape=} {scales.shape=} {block_bytes=}")


def decode_nvfp4_expert(
    weights: np.ndarray, scales: np.ndarray, *, amax: float, denominator: float
) -> np.ndarray:
    """Decode an expert's packed E2M1 matrix with E4M3 scales per 16 columns."""
    _check_layout(weights, scales, 8)
    if not (np.isfinite(amax) and amax >= 0 and np.isfinite(denominator) and denominator > 0):
        raise ValueError("invalid NVFP4 expert outer scale")
    values = _E2M1_PAIRS[weights].reshape(*scales.shape, 16)
    values *= _E4M3[scales][..., None]
    values *= np.float32(amax) / np.float32(denominator)
    return values.reshape(weights.shape[0], weights.shape[1] * 2)


def decode_mxfp8_expert(weights: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Decode an E4M3 matrix with one expanded UE8M0 scale per 32 columns."""
    _check_layout(weights, scales, 32)
    values = _E4M3[weights].reshape(*scales.shape, 32)
    values *= _UE8M0[scales][..., None]
    return values.reshape(weights.shape)
