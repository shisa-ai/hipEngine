"""X8 replacement-layout helpers for GGUF selected-down dp4a kernels.

The X8 layout is a byte-neutral replacement for selected MoE down experts. It
keeps each GGUF Q5_K/Q6_K block byte-exact, but stores eight adjacent output
rows for one K block contiguously:

``tiles[expert, out_pack8, k_block, 8 * block_bytes]``.

That matches the raw selected-pack8 q8_1+sudot4 diagnostic kernel shape without
requiring raw expert bytes to stay resident next to a replacement layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hipengine.quant.gguf import QK_K
from hipengine.quant.gguf_t16 import GGUF_Q5_K_BLOCK_BYTES, GGUF_Q6_K_BLOCK_BYTES
from hipengine.quant.registry import register_quant

GGUF_X8_COLS = 8
GGUF_Q4_K_BLOCK_BYTES = 144
GGUF_Q4_K_X8_BLOCK_BYTES = GGUF_X8_COLS * GGUF_Q4_K_BLOCK_BYTES
GGUF_Q5_K_X8_BLOCK_BYTES = GGUF_X8_COLS * GGUF_Q5_K_BLOCK_BYTES
GGUF_Q6_K_X8_BLOCK_BYTES = GGUF_X8_COLS * GGUF_Q6_K_BLOCK_BYTES


@dataclass(frozen=True)
class GGUFQ4KX8Quant:
    """X8 replacement-layout plugin key for GGUF block_q4_K experts."""

    name: str = "gguf_q4_k_x8_v1"
    weight_storage: str = "gguf_block_q4_k_x8_v1"
    activation_preprocess: str = "q8_1"
    compute_dtype: str = "fp32_accum"
    scale_granularity: str = "block256_subblock32_scale_min"
    calibration_artifact: str = "gguf"
    kernel_family: str = "gguf_x8_gemv"


@dataclass(frozen=True)
class GGUFQ5KX8Quant:
    """X8 replacement-layout plugin key for GGUF block_q5_K experts."""

    name: str = "gguf_q5_k_x8_v1"
    weight_storage: str = "gguf_block_q5_k_x8_v1"
    activation_preprocess: str = "q8_1"
    compute_dtype: str = "fp32_accum"
    scale_granularity: str = "block256_subblock32_scale_min"
    calibration_artifact: str = "gguf"
    kernel_family: str = "gguf_x8_gemv"


@dataclass(frozen=True)
class GGUFQ6KX8Quant:
    """X8 replacement-layout plugin key for GGUF block_q6_K experts."""

    name: str = "gguf_q6_k_x8_v1"
    weight_storage: str = "gguf_block_q6_k_x8_v1"
    activation_preprocess: str = "q8_1"
    compute_dtype: str = "fp32_accum"
    scale_granularity: str = "block256_subblock16_scale"
    calibration_artifact: str = "gguf"
    kernel_family: str = "gguf_x8_gemv"


@dataclass(frozen=True)
class GGUFQ4KX8:
    """Byte-exact X8 replacement layout for Q4_K selected experts."""

    tiles: np.ndarray
    experts: int
    out_features: int
    in_features: int

    @property
    def out_packed(self) -> int:
        return self.out_features // GGUF_X8_COLS

    @property
    def blocks_per_row(self) -> int:
        return self.in_features // QK_K


@dataclass(frozen=True)
class GGUFQ5KX8:
    """Byte-exact X8 replacement layout for Q5_K selected experts."""

    tiles: np.ndarray
    experts: int
    out_features: int
    in_features: int

    @property
    def out_packed(self) -> int:
        return self.out_features // GGUF_X8_COLS

    @property
    def blocks_per_row(self) -> int:
        return self.in_features // QK_K


@dataclass(frozen=True)
class GGUFQ6KX8:
    """Byte-exact X8 replacement layout for Q6_K selected experts."""

    tiles: np.ndarray
    experts: int
    out_features: int
    in_features: int

    @property
    def out_packed(self) -> int:
        return self.out_features // GGUF_X8_COLS

    @property
    def blocks_per_row(self) -> int:
        return self.in_features // QK_K


GGUF_Q4_K_X8_V1 = register_quant(GGUFQ4KX8Quant())
GGUF_Q5_K_X8_V1 = register_quant(GGUFQ5KX8Quant())
GGUF_Q6_K_X8_V1 = register_quant(GGUFQ6KX8Quant())


def _as_expert_raw(raw_qweight: Any, *, block_bytes: int, quant_name: str) -> tuple[np.ndarray, int, int, int, int]:
    raw = np.ascontiguousarray(raw_qweight, dtype=np.uint8)
    if raw.ndim != 3:
        raise ValueError(f"raw_qweight must have GGUF {quant_name} expert byte shape [experts, out_features, bytes_per_row]")
    experts, out_features, bytes_per_row = (int(raw.shape[0]), int(raw.shape[1]), int(raw.shape[2]))
    if experts <= 0:
        raise ValueError("experts must be positive")
    if out_features <= 0 or out_features % GGUF_X8_COLS != 0:
        raise ValueError("out_features must be positive and divisible by 8")
    if bytes_per_row <= 0 or bytes_per_row % block_bytes != 0:
        raise ValueError(f"bytes_per_row must be a positive multiple of {block_bytes}")
    return raw, experts, out_features, bytes_per_row, bytes_per_row // block_bytes


def _repack_x8(raw_qweight: Any, *, block_bytes: int, quant_name: str) -> tuple[np.ndarray, int, int, int]:
    raw, experts, out_features, _bytes_per_row, blocks_per_row = _as_expert_raw(
        raw_qweight,
        block_bytes=block_bytes,
        quant_name=quant_name,
    )
    out_packed = out_features // GGUF_X8_COLS
    blocks = raw.reshape(experts, out_features, blocks_per_row, block_bytes)
    tiles = np.empty((experts, out_packed, blocks_per_row, GGUF_X8_COLS * block_bytes), dtype=np.uint8)
    for out_pack in range(out_packed):
        cols = blocks[:, out_pack * GGUF_X8_COLS : (out_pack + 1) * GGUF_X8_COLS]
        tiles[:, out_pack] = cols.transpose(0, 2, 1, 3).reshape(
            experts,
            blocks_per_row,
            GGUF_X8_COLS * block_bytes,
        )
    return tiles, experts, out_features, blocks_per_row * QK_K


def _unpack_x8(tiles: np.ndarray, *, block_bytes: int, out_features: int | None) -> np.ndarray:
    arr = np.asarray(tiles, dtype=np.uint8)
    if arr.ndim != 4 or arr.shape[-1] != GGUF_X8_COLS * block_bytes:
        raise ValueError(
            f"tiles must have shape [experts, out_pack8, blocks_per_row, {GGUF_X8_COLS * block_bytes}]"
        )
    experts, out_packed, blocks_per_row, _ = (int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3]))
    inferred_out = out_packed * GGUF_X8_COLS
    if out_features is not None and int(out_features) != inferred_out:
        raise ValueError(f"out_features mismatch: expected {out_features}, tile layout implies {inferred_out}")
    blocks = np.empty((experts, inferred_out, blocks_per_row, block_bytes), dtype=np.uint8)
    for out_pack in range(out_packed):
        src = arr[:, out_pack].reshape(experts, blocks_per_row, GGUF_X8_COLS, block_bytes)
        blocks[:, out_pack * GGUF_X8_COLS : (out_pack + 1) * GGUF_X8_COLS] = src.transpose(0, 2, 1, 3)
    return blocks.reshape(experts, inferred_out, blocks_per_row * block_bytes)


def repack_gguf_q5_k_x8(raw_qweight: Any) -> GGUFQ5KX8:
    """Repack rank-3 raw GGUF Q5_K expert weights into byte-exact X8 tiles."""

    tiles, experts, out_features, in_features = _repack_x8(
        raw_qweight,
        block_bytes=GGUF_Q5_K_BLOCK_BYTES,
        quant_name="Q5_K",
    )
    return GGUFQ5KX8(tiles=tiles, experts=experts, out_features=out_features, in_features=in_features)


def repack_gguf_q4_k_x8(raw_qweight: Any) -> GGUFQ4KX8:
    """Repack rank-3 raw GGUF Q4_K expert weights into byte-exact X8 tiles."""

    tiles, experts, out_features, in_features = _repack_x8(
        raw_qweight,
        block_bytes=GGUF_Q4_K_BLOCK_BYTES,
        quant_name="Q4_K",
    )
    return GGUFQ4KX8(tiles=tiles, experts=experts, out_features=out_features, in_features=in_features)


def unpack_gguf_q4_k_x8(packed: GGUFQ4KX8 | np.ndarray, *, out_features: int | None = None) -> np.ndarray:
    """Reconstruct raw GGUF Q4_K expert bytes from X8 tiles."""

    if isinstance(packed, GGUFQ4KX8):
        tiles = packed.tiles
        expected_out = packed.out_features
    else:
        tiles = packed
        expected_out = out_features
    return _unpack_x8(tiles, block_bytes=GGUF_Q4_K_BLOCK_BYTES, out_features=expected_out)


def unpack_gguf_q5_k_x8(packed: GGUFQ5KX8 | np.ndarray, *, out_features: int | None = None) -> np.ndarray:
    """Reconstruct raw GGUF Q5_K expert bytes from X8 tiles."""

    if isinstance(packed, GGUFQ5KX8):
        tiles = packed.tiles
        expected_out = packed.out_features
    else:
        tiles = packed
        expected_out = out_features
    return _unpack_x8(tiles, block_bytes=GGUF_Q5_K_BLOCK_BYTES, out_features=expected_out)


def repack_gguf_q6_k_x8(raw_qweight: Any) -> GGUFQ6KX8:
    """Repack rank-3 raw GGUF Q6_K expert weights into byte-exact X8 tiles."""

    tiles, experts, out_features, in_features = _repack_x8(
        raw_qweight,
        block_bytes=GGUF_Q6_K_BLOCK_BYTES,
        quant_name="Q6_K",
    )
    return GGUFQ6KX8(tiles=tiles, experts=experts, out_features=out_features, in_features=in_features)


def repack_gguf_q6_k_x8_dscale_f32(raw_qweight: Any) -> np.ndarray:
    """Build X8-aligned FP32 ``d * scale`` sidecar tiles for raw Q6_K weights.

    The returned layout is ``[experts, out_pack8, k_block, 8, 16]`` and matches
    ``repack_gguf_q6_k_x8(...).tiles`` by ``(expert, out_pack8, k_block, row)``.
    """

    raw, experts, out_features, _bytes_per_row, blocks_per_row = _as_expert_raw(
        raw_qweight,
        block_bytes=GGUF_Q6_K_BLOCK_BYTES,
        quant_name="Q6_K",
    )
    out_packed = out_features // GGUF_X8_COLS
    blocks = raw.reshape(experts, out_features, blocks_per_row, GGUF_Q6_K_BLOCK_BYTES)
    d = np.ascontiguousarray(blocks[..., 208:210]).view(np.float16).reshape(
        experts,
        out_features,
        blocks_per_row,
    )
    scales = np.ascontiguousarray(blocks[..., 192:208]).view(np.int8).astype(np.float32)
    dscale_rows = d.astype(np.float32)[..., None] * scales
    dscale = np.empty((experts, out_packed, blocks_per_row, GGUF_X8_COLS, 16), dtype=np.float32)
    for out_pack in range(out_packed):
        rows = dscale_rows[:, out_pack * GGUF_X8_COLS : (out_pack + 1) * GGUF_X8_COLS]
        dscale[:, out_pack] = rows.transpose(0, 2, 1, 3)
    return np.ascontiguousarray(dscale)


def unpack_gguf_q6_k_x8(packed: GGUFQ6KX8 | np.ndarray, *, out_features: int | None = None) -> np.ndarray:
    """Reconstruct raw GGUF Q6_K expert bytes from X8 tiles."""

    if isinstance(packed, GGUFQ6KX8):
        tiles = packed.tiles
        expected_out = packed.out_features
    else:
        tiles = packed
        expected_out = out_features
    return _unpack_x8(tiles, block_bytes=GGUF_Q6_K_BLOCK_BYTES, out_features=expected_out)


__all__ = [
    "GGUF_Q4_K_X8_BLOCK_BYTES",
    "GGUF_Q4_K_X8_V1",
    "GGUF_Q5_K_X8_BLOCK_BYTES",
    "GGUF_Q5_K_X8_V1",
    "GGUF_Q6_K_X8_BLOCK_BYTES",
    "GGUF_Q6_K_X8_V1",
    "GGUF_X8_COLS",
    "GGUFQ4KX8",
    "GGUFQ4KX8Quant",
    "GGUFQ5KX8",
    "GGUFQ5KX8Quant",
    "GGUFQ6KX8",
    "GGUFQ6KX8Quant",
    "repack_gguf_q4_k_x8",
    "repack_gguf_q5_k_x8",
    "repack_gguf_q6_k_x8",
    "repack_gguf_q6_k_x8_dscale_f32",
    "unpack_gguf_q4_k_x8",
    "unpack_gguf_q5_k_x8",
    "unpack_gguf_q6_k_x8",
]
