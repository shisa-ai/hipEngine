"""T16 replacement-layout helpers for GGUF K-family decode kernels.

These helpers are deliberately CPU-side and bit-lossless.  They define the
resident byte layouts that the P9.H3 HIP rows=1 decode kernels will consume, and
provide inverse transforms for tests/oracles.  Runtime benchmark paths must use
these layouts as replacements for covered raw GGUF tensors, not as always-on
sidecars next to raw expert weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hipengine.quant.gguf import QK_K, unpack_q4_k_scale_min
from hipengine.quant.registry import register_quant

GGUF_T16_COLS = 16

GGUF_Q5_K_BLOCK_BYTES = 176
GGUF_Q5_K_SUBBLOCKS = 8
GGUF_Q5_K_SUBBLOCK = 32
GGUF_Q5_K_T16_D_OFFSET = 0
GGUF_Q5_K_T16_DMIN_OFFSET = GGUF_Q5_K_T16_D_OFFSET + GGUF_T16_COLS * 2
GGUF_Q5_K_T16_SCALE_OFFSET = GGUF_Q5_K_T16_DMIN_OFFSET + GGUF_T16_COLS * 2
GGUF_Q5_K_T16_MIN_OFFSET = GGUF_Q5_K_T16_SCALE_OFFSET + GGUF_Q5_K_SUBBLOCKS * GGUF_T16_COLS
GGUF_Q5_K_T16_QL_OFFSET = GGUF_Q5_K_T16_MIN_OFFSET + GGUF_Q5_K_SUBBLOCKS * GGUF_T16_COLS
GGUF_Q5_K_T16_QH_OFFSET = GGUF_Q5_K_T16_QL_OFFSET + GGUF_Q5_K_SUBBLOCKS * GGUF_Q5_K_SUBBLOCK * (GGUF_T16_COLS // 2)
GGUF_Q5_K_T16_BLOCK_BYTES = GGUF_Q5_K_T16_QH_OFFSET + GGUF_Q5_K_SUBBLOCKS * GGUF_Q5_K_SUBBLOCK * (GGUF_T16_COLS // 8)

GGUF_Q6_K_BLOCK_BYTES = 210
GGUF_Q6_K_GROUPS = 16
GGUF_Q6_K_T16_D_OFFSET = 0
GGUF_Q6_K_T16_SCALE_OFFSET = GGUF_Q6_K_T16_D_OFFSET + GGUF_T16_COLS * 2
GGUF_Q6_K_T16_QL_OFFSET = GGUF_Q6_K_T16_SCALE_OFFSET + GGUF_Q6_K_GROUPS * GGUF_T16_COLS
GGUF_Q6_K_T16_QH_OFFSET = GGUF_Q6_K_T16_QL_OFFSET + QK_K * (GGUF_T16_COLS // 2)
GGUF_Q6_K_T16_BLOCK_BYTES = GGUF_Q6_K_T16_QH_OFFSET + QK_K * (GGUF_T16_COLS // 4)

GGUF_Q8_0_BLOCK_BYTES = 34
GGUF_Q8_0_QK = 32
GGUF_Q8_0_T16_D_OFFSET = 0
GGUF_Q8_0_T16_Q_OFFSET = GGUF_Q8_0_T16_D_OFFSET + GGUF_T16_COLS * 2
GGUF_Q8_0_T16_BLOCK_BYTES = GGUF_Q8_0_T16_Q_OFFSET + GGUF_Q8_0_QK * GGUF_T16_COLS


@dataclass(frozen=True)
class GGUFQ5KT16Quant:
    """T16 replacement-layout plugin key for GGUF block_q5_K weights."""

    name: str = "gguf_q5_k_t16_v1"
    weight_storage: str = "gguf_block_q5_k_t16_v1"
    activation_preprocess: str = "none"
    compute_dtype: str = "fp32_accum"
    scale_granularity: str = "block256_subblock32_scale_min"
    calibration_artifact: str = "gguf"
    kernel_family: str = "gguf_t16_gemv"


@dataclass(frozen=True)
class GGUFQ6KT16Quant:
    """T16 replacement-layout plugin key for GGUF block_q6_K weights."""

    name: str = "gguf_q6_k_t16_v1"
    weight_storage: str = "gguf_block_q6_k_t16_v1"
    activation_preprocess: str = "none"
    compute_dtype: str = "fp32_accum"
    scale_granularity: str = "block256_subblock16_scale"
    calibration_artifact: str = "gguf"
    kernel_family: str = "gguf_t16_gemv"


@dataclass(frozen=True)
class GGUFQ80T16Quant:
    """T16 replacement-layout plugin key for GGUF block_q8_0 weights."""

    name: str = "gguf_q8_0_t16_v1"
    weight_storage: str = "gguf_block_q8_0_t16_v1"
    activation_preprocess: str = "none"
    compute_dtype: str = "fp32_accum"
    scale_granularity: str = "block32_scale"
    calibration_artifact: str = "gguf"
    kernel_family: str = "gguf_t16_gemv"


@dataclass(frozen=True)
class GGUFQ5KTile16:
    """Tile-major Q5_K selected-expert replacement layout.

    ``tiles`` has shape ``[experts, out_tiles16, blocks_per_row, 2880]``.
    """

    tiles: np.ndarray
    experts: int
    out_features: int
    in_features: int

    @property
    def out_tiles(self) -> int:
        return self.out_features // GGUF_T16_COLS

    @property
    def blocks_per_row(self) -> int:
        return self.in_features // QK_K


@dataclass(frozen=True)
class GGUFQ6KTile16:
    """Tile-major Q6_K selected-expert replacement layout.

    ``tiles`` has shape ``[experts, out_tiles16, blocks_per_row, 3360]``.
    """

    tiles: np.ndarray
    experts: int
    out_features: int
    in_features: int

    @property
    def out_tiles(self) -> int:
        return self.out_features // GGUF_T16_COLS

    @property
    def blocks_per_row(self) -> int:
        return self.in_features // QK_K


@dataclass(frozen=True)
class GGUFQ80Tile16:
    """Tile-major Q8_0 dense/shared replacement layout.

    ``tiles`` has shape ``[out_tiles16, blocks_per_row, 544]``.
    """

    tiles: np.ndarray
    out_features: int
    in_features: int

    @property
    def out_tiles(self) -> int:
        return self.out_features // GGUF_T16_COLS

    @property
    def blocks_per_row(self) -> int:
        return self.in_features // GGUF_Q8_0_QK


GGUF_Q5_K_T16_V1 = register_quant(GGUFQ5KT16Quant())
GGUF_Q6_K_T16_V1 = register_quant(GGUFQ6KT16Quant())
GGUF_Q8_0_T16_V1 = register_quant(GGUFQ80T16Quant())


def _pack_q4_k_scale_min(scales: np.ndarray, mins: np.ndarray) -> np.ndarray:
    """Inverse of ``unpack_q4_k_scale_min`` for uint8 scale/min arrays."""

    sc = np.asarray(scales, dtype=np.uint8)
    mn = np.asarray(mins, dtype=np.uint8)
    if sc.shape != mn.shape or sc.shape[-1] != GGUF_Q5_K_SUBBLOCKS:
        raise ValueError("scales/mins must have matching shape ending in 8")
    packed = np.empty((*sc.shape[:-1], 12), dtype=np.uint8)
    packed[..., 0:4] = (sc[..., 0:4] & np.uint8(0x3F)) | ((sc[..., 4:8] & np.uint8(0x30)) << np.uint8(2))
    packed[..., 4:8] = (mn[..., 0:4] & np.uint8(0x3F)) | ((mn[..., 4:8] & np.uint8(0x30)) << np.uint8(2))
    packed[..., 8:12] = (sc[..., 4:8] & np.uint8(0x0F)) | ((mn[..., 4:8] & np.uint8(0x0F)) << np.uint8(4))
    return packed


def _as_expert_raw(raw_qweight: Any, *, block_bytes: int, quant_name: str) -> tuple[np.ndarray, int, int, int, int]:
    raw = np.ascontiguousarray(raw_qweight, dtype=np.uint8)
    if raw.ndim != 3:
        raise ValueError(f"raw_qweight must have GGUF {quant_name} expert byte shape [experts, out_features, bytes_per_row]")
    experts, out_features, bytes_per_row = (int(raw.shape[0]), int(raw.shape[1]), int(raw.shape[2]))
    if experts <= 0:
        raise ValueError("experts must be positive")
    if out_features <= 0 or out_features % GGUF_T16_COLS != 0:
        raise ValueError("out_features must be positive and divisible by 16")
    if bytes_per_row <= 0 or bytes_per_row % block_bytes != 0:
        raise ValueError(f"bytes_per_row must be a positive multiple of {block_bytes}")
    return raw, experts, out_features, bytes_per_row, bytes_per_row // block_bytes


def _as_dense_raw(raw_qweight: Any, *, block_bytes: int, quant_name: str) -> tuple[np.ndarray, int, int, int]:
    raw = np.ascontiguousarray(raw_qweight, dtype=np.uint8)
    if raw.ndim != 2:
        raise ValueError(f"raw_qweight must have GGUF {quant_name} dense byte shape [out_features, bytes_per_row]")
    out_features, bytes_per_row = (int(raw.shape[0]), int(raw.shape[1]))
    if out_features <= 0 or out_features % GGUF_T16_COLS != 0:
        raise ValueError("out_features must be positive and divisible by 16")
    if bytes_per_row <= 0 or bytes_per_row % block_bytes != 0:
        raise ValueError(f"bytes_per_row must be a positive multiple of {block_bytes}")
    return raw, out_features, bytes_per_row, bytes_per_row // block_bytes


def repack_gguf_q5_k_tile16(raw_qweight: Any) -> GGUFQ5KTile16:
    """Repack rank-3 raw GGUF Q5_K expert weights into bit-lossless Q5T16 tiles."""

    raw, experts, out_features, _bytes_per_row, blocks_per_row = _as_expert_raw(
        raw_qweight,
        block_bytes=GGUF_Q5_K_BLOCK_BYTES,
        quant_name="Q5_K",
    )
    out_tiles = out_features // GGUF_T16_COLS
    blocks = raw.reshape(experts, out_features, blocks_per_row, GGUF_Q5_K_BLOCK_BYTES)
    tiles = np.empty((experts, out_tiles, blocks_per_row, GGUF_Q5_K_T16_BLOCK_BYTES), dtype=np.uint8)
    col_bits8 = np.arange(8, dtype=np.uint16).reshape(1, 1, 1, 1, 1, 8)

    for out_tile in range(out_tiles):
        cols = blocks[:, out_tile * GGUF_T16_COLS : (out_tile + 1) * GGUF_T16_COLS]
        dst = tiles[:, out_tile]
        dst[..., GGUF_Q5_K_T16_D_OFFSET:GGUF_Q5_K_T16_DMIN_OFFSET] = (
            cols[..., 0:2].transpose(0, 2, 1, 3).reshape(experts, blocks_per_row, GGUF_T16_COLS * 2)
        )
        dst[..., GGUF_Q5_K_T16_DMIN_OFFSET:GGUF_Q5_K_T16_SCALE_OFFSET] = (
            cols[..., 2:4].transpose(0, 2, 1, 3).reshape(experts, blocks_per_row, GGUF_T16_COLS * 2)
        )
        sc, mn = unpack_q4_k_scale_min(cols[..., 4:16].reshape(-1, 12))
        sc = sc.reshape(experts, GGUF_T16_COLS, blocks_per_row, GGUF_Q5_K_SUBBLOCKS)
        mn = mn.reshape(experts, GGUF_T16_COLS, blocks_per_row, GGUF_Q5_K_SUBBLOCKS)
        dst[..., GGUF_Q5_K_T16_SCALE_OFFSET:GGUF_Q5_K_T16_MIN_OFFSET] = (
            sc.transpose(0, 2, 3, 1).reshape(experts, blocks_per_row, GGUF_Q5_K_SUBBLOCKS * GGUF_T16_COLS)
        )
        dst[..., GGUF_Q5_K_T16_MIN_OFFSET:GGUF_Q5_K_T16_QL_OFFSET] = (
            mn.transpose(0, 2, 3, 1).reshape(experts, blocks_per_row, GGUF_Q5_K_SUBBLOCKS * GGUF_T16_COLS)
        )

        ql_pairs = cols[..., 48:176].reshape(experts, GGUF_T16_COLS, blocks_per_row, 4, GGUF_Q5_K_SUBBLOCK)
        ql = np.empty((experts, GGUF_T16_COLS, blocks_per_row, GGUF_Q5_K_SUBBLOCKS, GGUF_Q5_K_SUBBLOCK), dtype=np.uint8)
        for sb in range(GGUF_Q5_K_SUBBLOCKS):
            packed = ql_pairs[..., sb >> 1, :]
            ql[..., sb, :] = ((packed >> np.uint8(4)) if (sb & 1) else packed) & np.uint8(0x0F)
        ql_tile = ql.transpose(0, 2, 3, 4, 1)
        ql_packed_cols = (ql_tile[..., 0::2] & np.uint8(0x0F)) | ((ql_tile[..., 1::2] & np.uint8(0x0F)) << np.uint8(4))
        dst[..., GGUF_Q5_K_T16_QL_OFFSET:GGUF_Q5_K_T16_QH_OFFSET] = ql_packed_cols.reshape(
            experts,
            blocks_per_row,
            GGUF_Q5_K_SUBBLOCKS * GGUF_Q5_K_SUBBLOCK * (GGUF_T16_COLS // 2),
        )

        qh_raw = cols[..., 16:48]
        qh = ((qh_raw[:, :, :, None, :] >> np.arange(8, dtype=np.uint8).reshape(1, 1, 1, 8, 1)) & np.uint8(0x01))
        qh_tile = qh.transpose(0, 2, 3, 4, 1).astype(np.uint16, copy=False)
        qh_packed_cols = ((qh_tile.reshape(experts, blocks_per_row, GGUF_Q5_K_SUBBLOCKS, GGUF_Q5_K_SUBBLOCK, 2, 8) & np.uint16(1)) << col_bits8).sum(axis=-1).astype(np.uint8)
        dst[..., GGUF_Q5_K_T16_QH_OFFSET:] = qh_packed_cols.reshape(
            experts,
            blocks_per_row,
            GGUF_Q5_K_SUBBLOCKS * GGUF_Q5_K_SUBBLOCK * (GGUF_T16_COLS // 8),
        )

    return GGUFQ5KTile16(tiles=tiles, experts=experts, out_features=out_features, in_features=blocks_per_row * QK_K)


def unpack_gguf_q5_k_tile16(packed: GGUFQ5KTile16 | np.ndarray, *, out_features: int | None = None) -> np.ndarray:
    """Reconstruct raw GGUF Q5_K expert bytes from Q5T16 tiles."""

    if isinstance(packed, GGUFQ5KTile16):
        tiles = np.asarray(packed.tiles, dtype=np.uint8)
        expected_out = packed.out_features
    else:
        tiles = np.asarray(packed, dtype=np.uint8)
        expected_out = out_features
    if tiles.ndim != 4 or tiles.shape[-1] != GGUF_Q5_K_T16_BLOCK_BYTES:
        raise ValueError("tiles must have shape [experts, out_tiles16, blocks_per_row, 2880]")
    experts, out_tiles, blocks_per_row, _ = (int(tiles.shape[0]), int(tiles.shape[1]), int(tiles.shape[2]), int(tiles.shape[3]))
    inferred_out = out_tiles * GGUF_T16_COLS
    if expected_out is not None and int(expected_out) != inferred_out:
        raise ValueError(f"out_features mismatch: expected {expected_out}, tile layout implies {inferred_out}")

    blocks = np.empty((experts, inferred_out, blocks_per_row, GGUF_Q5_K_BLOCK_BYTES), dtype=np.uint8)
    for out_tile in range(out_tiles):
        src = tiles[:, out_tile]
        cols = blocks[:, out_tile * GGUF_T16_COLS : (out_tile + 1) * GGUF_T16_COLS]
        cols[..., 0:2] = src[..., GGUF_Q5_K_T16_D_OFFSET:GGUF_Q5_K_T16_DMIN_OFFSET].reshape(
            experts, blocks_per_row, GGUF_T16_COLS, 2
        ).transpose(0, 2, 1, 3)
        cols[..., 2:4] = src[..., GGUF_Q5_K_T16_DMIN_OFFSET:GGUF_Q5_K_T16_SCALE_OFFSET].reshape(
            experts, blocks_per_row, GGUF_T16_COLS, 2
        ).transpose(0, 2, 1, 3)
        sc = src[..., GGUF_Q5_K_T16_SCALE_OFFSET:GGUF_Q5_K_T16_MIN_OFFSET].reshape(
            experts, blocks_per_row, GGUF_Q5_K_SUBBLOCKS, GGUF_T16_COLS
        ).transpose(0, 3, 1, 2)
        mn = src[..., GGUF_Q5_K_T16_MIN_OFFSET:GGUF_Q5_K_T16_QL_OFFSET].reshape(
            experts, blocks_per_row, GGUF_Q5_K_SUBBLOCKS, GGUF_T16_COLS
        ).transpose(0, 3, 1, 2)
        cols[..., 4:16] = _pack_q4_k_scale_min(sc, mn)

        ql_packed_cols = src[..., GGUF_Q5_K_T16_QL_OFFSET:GGUF_Q5_K_T16_QH_OFFSET].reshape(
            experts,
            blocks_per_row,
            GGUF_Q5_K_SUBBLOCKS,
            GGUF_Q5_K_SUBBLOCK,
            GGUF_T16_COLS // 2,
        )
        ql = np.empty((experts, blocks_per_row, GGUF_Q5_K_SUBBLOCKS, GGUF_Q5_K_SUBBLOCK, GGUF_T16_COLS), dtype=np.uint8)
        ql[..., 0::2] = ql_packed_cols & np.uint8(0x0F)
        ql[..., 1::2] = ql_packed_cols >> np.uint8(4)
        ql_by_col = ql.transpose(0, 4, 1, 2, 3)
        ql_pairs = np.empty((experts, GGUF_T16_COLS, blocks_per_row, 4, GGUF_Q5_K_SUBBLOCK), dtype=np.uint8)
        for pair in range(4):
            ql_pairs[..., pair, :] = (ql_by_col[..., 2 * pair, :] & np.uint8(0x0F)) | (
                (ql_by_col[..., 2 * pair + 1, :] & np.uint8(0x0F)) << np.uint8(4)
            )
        cols[..., 48:176] = ql_pairs.reshape(experts, GGUF_T16_COLS, blocks_per_row, 128)

        qh_packed_cols = src[..., GGUF_Q5_K_T16_QH_OFFSET:].reshape(
            experts,
            blocks_per_row,
            GGUF_Q5_K_SUBBLOCKS,
            GGUF_Q5_K_SUBBLOCK,
            GGUF_T16_COLS // 8,
        )
        qh_bits = (
            qh_packed_cols[..., None] >> np.arange(8, dtype=np.uint8).reshape(1, 1, 1, 1, 1, 8)
        ) & np.uint8(0x01)
        qh_by_col = qh_bits.reshape(experts, blocks_per_row, GGUF_Q5_K_SUBBLOCKS, GGUF_Q5_K_SUBBLOCK, GGUF_T16_COLS).transpose(0, 4, 1, 2, 3)
        qh_raw = np.zeros((experts, GGUF_T16_COLS, blocks_per_row, GGUF_Q5_K_SUBBLOCK), dtype=np.uint8)
        for sb in range(GGUF_Q5_K_SUBBLOCKS):
            qh_raw |= (qh_by_col[..., sb, :] & np.uint8(0x01)) << np.uint8(sb)
        cols[..., 16:48] = qh_raw

    return blocks.reshape(experts, inferred_out, blocks_per_row * GGUF_Q5_K_BLOCK_BYTES)


def repack_gguf_q6_k_tile16(raw_qweight: Any) -> GGUFQ6KTile16:
    """Repack rank-3 raw GGUF Q6_K expert weights into bit-lossless Q6T16 tiles."""

    raw, experts, out_features, _bytes_per_row, blocks_per_row = _as_expert_raw(
        raw_qweight,
        block_bytes=GGUF_Q6_K_BLOCK_BYTES,
        quant_name="Q6_K",
    )
    out_tiles = out_features // GGUF_T16_COLS
    blocks = raw.reshape(experts, out_features, blocks_per_row, GGUF_Q6_K_BLOCK_BYTES)
    tiles = np.empty((experts, out_tiles, blocks_per_row, GGUF_Q6_K_T16_BLOCK_BYTES), dtype=np.uint8)
    col_bits4 = (2 * np.arange(4, dtype=np.uint16)).reshape(1, 1, 1, 1, 4)

    for out_tile in range(out_tiles):
        cols = blocks[:, out_tile * GGUF_T16_COLS : (out_tile + 1) * GGUF_T16_COLS]
        dst = tiles[:, out_tile]
        dst[..., GGUF_Q6_K_T16_D_OFFSET:GGUF_Q6_K_T16_SCALE_OFFSET] = (
            cols[..., 208:210].transpose(0, 2, 1, 3).reshape(experts, blocks_per_row, GGUF_T16_COLS * 2)
        )
        dst[..., GGUF_Q6_K_T16_SCALE_OFFSET:GGUF_Q6_K_T16_QL_OFFSET] = (
            cols[..., 192:208].transpose(0, 2, 3, 1).reshape(experts, blocks_per_row, GGUF_Q6_K_GROUPS * GGUF_T16_COLS)
        )

        ql_raw = cols[..., 0:128].reshape(experts, GGUF_T16_COLS, blocks_per_row, 2, 1, 64)
        ql = ((ql_raw >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 1, 1, 2, 1)) & np.uint8(0x0F)).reshape(
            experts, GGUF_T16_COLS, blocks_per_row, QK_K
        )
        ql_tile = ql.transpose(0, 2, 3, 1)
        ql_packed_cols = (ql_tile[..., 0::2] & np.uint8(0x0F)) | ((ql_tile[..., 1::2] & np.uint8(0x0F)) << np.uint8(4))
        dst[..., GGUF_Q6_K_T16_QL_OFFSET:GGUF_Q6_K_T16_QH_OFFSET] = ql_packed_cols.reshape(
            experts,
            blocks_per_row,
            QK_K * (GGUF_T16_COLS // 2),
        )

        qh_raw = cols[..., 128:192].reshape(experts, GGUF_T16_COLS, blocks_per_row, 2, 1, 32)
        qh = ((qh_raw >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 1, 1, 4, 1)) & np.uint8(0x03)).reshape(
            experts, GGUF_T16_COLS, blocks_per_row, QK_K
        )
        qh_tile = qh.transpose(0, 2, 3, 1).astype(np.uint16, copy=False)
        qh_packed_cols = ((qh_tile.reshape(experts, blocks_per_row, QK_K, 4, 4) & np.uint16(0x03)) << col_bits4).sum(axis=-1).astype(np.uint8)
        dst[..., GGUF_Q6_K_T16_QH_OFFSET:] = qh_packed_cols.reshape(experts, blocks_per_row, QK_K * (GGUF_T16_COLS // 4))

    return GGUFQ6KTile16(tiles=tiles, experts=experts, out_features=out_features, in_features=blocks_per_row * QK_K)


def unpack_gguf_q6_k_tile16(packed: GGUFQ6KTile16 | np.ndarray, *, out_features: int | None = None) -> np.ndarray:
    """Reconstruct raw GGUF Q6_K expert bytes from Q6T16 tiles."""

    if isinstance(packed, GGUFQ6KTile16):
        tiles = np.asarray(packed.tiles, dtype=np.uint8)
        expected_out = packed.out_features
    else:
        tiles = np.asarray(packed, dtype=np.uint8)
        expected_out = out_features
    if tiles.ndim != 4 or tiles.shape[-1] != GGUF_Q6_K_T16_BLOCK_BYTES:
        raise ValueError("tiles must have shape [experts, out_tiles16, blocks_per_row, 3360]")
    experts, out_tiles, blocks_per_row, _ = (int(tiles.shape[0]), int(tiles.shape[1]), int(tiles.shape[2]), int(tiles.shape[3]))
    inferred_out = out_tiles * GGUF_T16_COLS
    if expected_out is not None and int(expected_out) != inferred_out:
        raise ValueError(f"out_features mismatch: expected {expected_out}, tile layout implies {inferred_out}")

    blocks = np.empty((experts, inferred_out, blocks_per_row, GGUF_Q6_K_BLOCK_BYTES), dtype=np.uint8)
    for out_tile in range(out_tiles):
        src = tiles[:, out_tile]
        cols = blocks[:, out_tile * GGUF_T16_COLS : (out_tile + 1) * GGUF_T16_COLS]
        cols[..., 208:210] = src[..., GGUF_Q6_K_T16_D_OFFSET:GGUF_Q6_K_T16_SCALE_OFFSET].reshape(
            experts, blocks_per_row, GGUF_T16_COLS, 2
        ).transpose(0, 2, 1, 3)
        cols[..., 192:208] = src[..., GGUF_Q6_K_T16_SCALE_OFFSET:GGUF_Q6_K_T16_QL_OFFSET].reshape(
            experts, blocks_per_row, GGUF_Q6_K_GROUPS, GGUF_T16_COLS
        ).transpose(0, 3, 1, 2)

        ql_packed_cols = src[..., GGUF_Q6_K_T16_QL_OFFSET:GGUF_Q6_K_T16_QH_OFFSET].reshape(
            experts,
            blocks_per_row,
            QK_K,
            GGUF_T16_COLS // 2,
        )
        ql = np.empty((experts, blocks_per_row, QK_K, GGUF_T16_COLS), dtype=np.uint8)
        ql[..., 0::2] = ql_packed_cols & np.uint8(0x0F)
        ql[..., 1::2] = ql_packed_cols >> np.uint8(4)
        ql_by_col = ql.transpose(0, 3, 1, 2).reshape(experts, GGUF_T16_COLS, blocks_per_row, 2, 2, 64)
        ql_raw = (ql_by_col[..., 0, :] & np.uint8(0x0F)) | ((ql_by_col[..., 1, :] & np.uint8(0x0F)) << np.uint8(4))
        cols[..., 0:128] = ql_raw.reshape(experts, GGUF_T16_COLS, blocks_per_row, 128)

        qh_packed_cols = src[..., GGUF_Q6_K_T16_QH_OFFSET:].reshape(
            experts,
            blocks_per_row,
            QK_K,
            GGUF_T16_COLS // 4,
        )
        qh = (
            qh_packed_cols[..., None] >> (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 1, 1, 4)
        ) & np.uint8(0x03)
        qh_by_col = qh.reshape(experts, blocks_per_row, QK_K, GGUF_T16_COLS).transpose(0, 3, 1, 2).reshape(
            experts, GGUF_T16_COLS, blocks_per_row, 2, 4, 32
        )
        qh_raw = np.zeros((experts, GGUF_T16_COLS, blocks_per_row, 2, 32), dtype=np.uint8)
        for part in range(4):
            qh_raw |= (qh_by_col[..., part, :] & np.uint8(0x03)) << np.uint8(2 * part)
        cols[..., 128:192] = qh_raw.reshape(experts, GGUF_T16_COLS, blocks_per_row, 64)

    return blocks.reshape(experts, inferred_out, blocks_per_row * GGUF_Q6_K_BLOCK_BYTES)


def repack_gguf_q8_0_tile16(raw_qweight: Any) -> GGUFQ80Tile16:
    """Repack rank-2 raw GGUF Q8_0 weights into bit-lossless Q8T16 tiles."""

    raw, out_features, _bytes_per_row, blocks_per_row = _as_dense_raw(
        raw_qweight,
        block_bytes=GGUF_Q8_0_BLOCK_BYTES,
        quant_name="Q8_0",
    )
    out_tiles = out_features // GGUF_T16_COLS
    blocks = raw.reshape(out_features, blocks_per_row, GGUF_Q8_0_BLOCK_BYTES)
    tiles = np.empty((out_tiles, blocks_per_row, GGUF_Q8_0_T16_BLOCK_BYTES), dtype=np.uint8)

    for out_tile in range(out_tiles):
        cols = blocks[out_tile * GGUF_T16_COLS : (out_tile + 1) * GGUF_T16_COLS]
        dst = tiles[out_tile]
        dst[..., GGUF_Q8_0_T16_D_OFFSET:GGUF_Q8_0_T16_Q_OFFSET] = (
            cols[..., 0:2].transpose(1, 0, 2).reshape(blocks_per_row, GGUF_T16_COLS * 2)
        )
        dst[..., GGUF_Q8_0_T16_Q_OFFSET:] = (
            cols[..., 2:34].transpose(1, 2, 0).reshape(blocks_per_row, GGUF_Q8_0_QK * GGUF_T16_COLS)
        )

    return GGUFQ80Tile16(tiles=tiles, out_features=out_features, in_features=blocks_per_row * GGUF_Q8_0_QK)


def unpack_gguf_q8_0_tile16(packed: GGUFQ80Tile16 | np.ndarray, *, out_features: int | None = None) -> np.ndarray:
    """Reconstruct raw GGUF Q8_0 dense bytes from Q8T16 tiles."""

    if isinstance(packed, GGUFQ80Tile16):
        tiles = np.asarray(packed.tiles, dtype=np.uint8)
        expected_out = packed.out_features
    else:
        tiles = np.asarray(packed, dtype=np.uint8)
        expected_out = out_features
    if tiles.ndim != 3 or tiles.shape[-1] != GGUF_Q8_0_T16_BLOCK_BYTES:
        raise ValueError("tiles must have shape [out_tiles16, blocks_per_row, 544]")
    out_tiles, blocks_per_row, _ = (int(tiles.shape[0]), int(tiles.shape[1]), int(tiles.shape[2]))
    inferred_out = out_tiles * GGUF_T16_COLS
    if expected_out is not None and int(expected_out) != inferred_out:
        raise ValueError(f"out_features mismatch: expected {expected_out}, tile layout implies {inferred_out}")

    blocks = np.empty((inferred_out, blocks_per_row, GGUF_Q8_0_BLOCK_BYTES), dtype=np.uint8)
    for out_tile in range(out_tiles):
        src = tiles[out_tile]
        cols = blocks[out_tile * GGUF_T16_COLS : (out_tile + 1) * GGUF_T16_COLS]
        cols[..., 0:2] = src[..., GGUF_Q8_0_T16_D_OFFSET:GGUF_Q8_0_T16_Q_OFFSET].reshape(
            blocks_per_row, GGUF_T16_COLS, 2
        ).transpose(1, 0, 2)
        cols[..., 2:34] = src[..., GGUF_Q8_0_T16_Q_OFFSET:].reshape(
            blocks_per_row, GGUF_Q8_0_QK, GGUF_T16_COLS
        ).transpose(2, 0, 1)

    return blocks.reshape(inferred_out, blocks_per_row * GGUF_Q8_0_BLOCK_BYTES)


__all__ = [
    "GGUF_Q5_K_BLOCK_BYTES",
    "GGUF_Q5_K_T16_BLOCK_BYTES",
    "GGUF_Q5_K_T16_V1",
    "GGUF_Q6_K_BLOCK_BYTES",
    "GGUF_Q6_K_T16_BLOCK_BYTES",
    "GGUF_Q6_K_T16_V1",
    "GGUF_Q8_0_BLOCK_BYTES",
    "GGUF_Q8_0_T16_BLOCK_BYTES",
    "GGUF_Q8_0_T16_V1",
    "GGUF_T16_COLS",
    "GGUFQ5KTile16",
    "GGUFQ5KT16Quant",
    "GGUFQ6KTile16",
    "GGUFQ6KT16Quant",
    "GGUFQ80Tile16",
    "GGUFQ80T16Quant",
    "repack_gguf_q5_k_tile16",
    "repack_gguf_q6_k_tile16",
    "repack_gguf_q8_0_tile16",
    "unpack_gguf_q5_k_tile16",
    "unpack_gguf_q6_k_tile16",
    "unpack_gguf_q8_0_tile16",
]
