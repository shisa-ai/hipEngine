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

# Byte-neutral Q5 qmicro keeps T16's d/dmin and quant planes while encoding
# each four-column group of 6-bit scale/min coefficients in one 24-bit record.
GGUF_Q5_K_QMICRO_T16_D_OFFSET = 0
GGUF_Q5_K_QMICRO_T16_DMIN_OFFSET = GGUF_Q5_K_QMICRO_T16_D_OFFSET + GGUF_T16_COLS * 2
GGUF_Q5_K_QMICRO_T16_META_OFFSET = GGUF_Q5_K_QMICRO_T16_DMIN_OFFSET + GGUF_T16_COLS * 2
GGUF_Q5_K_QMICRO_T16_QL_OFFSET = (
    GGUF_Q5_K_QMICRO_T16_META_OFFSET + 2 * GGUF_Q5_K_SUBBLOCKS * (GGUF_T16_COLS // 4) * 3
)
GGUF_Q5_K_QMICRO_T16_QH_OFFSET = (
    GGUF_Q5_K_QMICRO_T16_QL_OFFSET
    + GGUF_Q5_K_SUBBLOCKS * GGUF_Q5_K_SUBBLOCK * (GGUF_T16_COLS // 2)
)
GGUF_Q5_K_QMICRO_T16_BLOCK_BYTES = (
    GGUF_Q5_K_QMICRO_T16_QH_OFFSET
    + GGUF_Q5_K_SUBBLOCKS * GGUF_Q5_K_SUBBLOCK * (GGUF_T16_COLS // 8)
)

# Planar Q5 qmicro: the qmicro ql/qh planes are reordered into the same
# 12-byte dp4a record shape as the Q6 planar qmicro (per subblock x column
# quartet x 4-quant pack: four col0/col1 low-nibble bytes, four col2/col3
# bytes, then four bytes of packed per-column high bits). d/dmin planes and
# the 24-bit scale/min metadata records are carried over unchanged.
GGUF_Q5_K_QMICRO_PLANAR_T16_RECORD_BYTES = 12  # matches GGUF_Q6_K_T16_QMICRO_RECORD_BYTES
GGUF_Q5_K_QMICRO_PLANAR_T16_BLOCK_BYTES = (
    GGUF_Q5_K_QMICRO_T16_QL_OFFSET
    + GGUF_Q5_K_SUBBLOCKS * (GGUF_T16_COLS // 4) * (GGUF_Q5_K_SUBBLOCK // 4)
    * GGUF_Q5_K_QMICRO_PLANAR_T16_RECORD_BYTES
)

GGUF_Q6_K_BLOCK_BYTES = 210
GGUF_Q6_K_GROUPS = 16
GGUF_Q6_K_T16_D_OFFSET = 0
GGUF_Q6_K_T16_SCALE_OFFSET = GGUF_Q6_K_T16_D_OFFSET + GGUF_T16_COLS * 2
GGUF_Q6_K_T16_QL_OFFSET = GGUF_Q6_K_T16_SCALE_OFFSET + GGUF_Q6_K_GROUPS * GGUF_T16_COLS
GGUF_Q6_K_T16_QH_OFFSET = GGUF_Q6_K_T16_QL_OFFSET + QK_K * (GGUF_T16_COLS // 2)
GGUF_Q6_K_T16_BLOCK_BYTES = GGUF_Q6_K_T16_QH_OFFSET + QK_K * (GGUF_T16_COLS // 4)
GGUF_Q6_K_T16_QMICRO_OFFSET = GGUF_Q6_K_T16_QL_OFFSET
GGUF_Q6_K_T16_QMICRO_RECORD_BYTES = 12

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
class GGUFQ5KQMicroT16Quant:
    """Byte-neutral qmicro T16 plugin key for GGUF block_q5_K weights."""

    name: str = "gguf_q5_k_qmicro_t16_v1"
    weight_storage: str = "gguf_block_q5_k_qmicro_t16_v1"
    activation_preprocess: str = "none"
    compute_dtype: str = "fp32_accum"
    scale_granularity: str = "block256_subblock32_scale_min_packed24x4"
    calibration_artifact: str = "gguf"
    kernel_family: str = "gguf_t16_qmicro_gemv"


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
class GGUFQ6KT16QMicroPlanarQuant:
    """Planar-qmicro replacement-layout key for dense Q6T16 weights."""

    name: str = "gguf_q6_k_t16_qmicro_planar_v1"
    weight_storage: str = "gguf_block_q6_k_t16_qmicro_planar_v1"
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
class GGUFQ5KQMicroTile16:
    """Byte-neutral Q5_K T16 layout with packed 24-bit metadata records."""

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
GGUF_Q5_K_QMICRO_T16_V1 = register_quant(GGUFQ5KQMicroT16Quant())
GGUF_Q6_K_T16_V1 = register_quant(GGUFQ6KT16Quant())
GGUF_Q6_K_T16_QMICRO_PLANAR_V1 = register_quant(
    GGUFQ6KT16QMicroPlanarQuant()
)
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


def convert_gguf_q5_k_tile16_to_qmicro(
    packed: GGUFQ5KTile16 | np.ndarray,
) -> GGUFQ5KQMicroTile16:
    """Compact expanded Q5T16 scale/min planes into exact 24-bit records."""

    tiles = np.asarray(packed.tiles if isinstance(packed, GGUFQ5KTile16) else packed, dtype=np.uint8)
    if tiles.ndim != 4 or tiles.shape[-1] != GGUF_Q5_K_T16_BLOCK_BYTES:
        raise ValueError("tiles must have shape [experts, out_tiles16, blocks_per_row, 2880]")
    experts, out_tiles, blocks_per_row, _ = map(int, tiles.shape)
    compact = np.empty(
        (experts, out_tiles, blocks_per_row, GGUF_Q5_K_QMICRO_T16_BLOCK_BYTES),
        dtype=np.uint8,
    )
    compact[..., :GGUF_Q5_K_QMICRO_T16_META_OFFSET] = tiles[..., :GGUF_Q5_K_T16_SCALE_OFFSET]
    scales = tiles[..., GGUF_Q5_K_T16_SCALE_OFFSET:GGUF_Q5_K_T16_MIN_OFFSET].reshape(
        experts, out_tiles, blocks_per_row, GGUF_Q5_K_SUBBLOCKS, GGUF_T16_COLS
    )
    mins = tiles[..., GGUF_Q5_K_T16_MIN_OFFSET:GGUF_Q5_K_T16_QL_OFFSET].reshape(
        experts, out_tiles, blocks_per_row, GGUF_Q5_K_SUBBLOCKS, GGUF_T16_COLS
    )
    coeffs = np.stack((scales, mins), axis=-3).reshape(
        experts,
        out_tiles,
        blocks_per_row,
        2,
        GGUF_Q5_K_SUBBLOCKS,
        GGUF_T16_COLS // 4,
        4,
    ).astype(np.uint32)
    words = (
        coeffs[..., 0]
        | (coeffs[..., 1] << np.uint32(6))
        | (coeffs[..., 2] << np.uint32(12))
        | (coeffs[..., 3] << np.uint32(18))
    )
    records = np.stack(
        (
            words.astype(np.uint8),
            (words >> np.uint32(8)).astype(np.uint8),
            (words >> np.uint32(16)).astype(np.uint8),
        ),
        axis=-1,
    )
    compact[..., GGUF_Q5_K_QMICRO_T16_META_OFFSET:GGUF_Q5_K_QMICRO_T16_QL_OFFSET] = records.reshape(
        experts, out_tiles, blocks_per_row, -1
    )
    compact[..., GGUF_Q5_K_QMICRO_T16_QL_OFFSET:GGUF_Q5_K_QMICRO_T16_QH_OFFSET] = tiles[
        ..., GGUF_Q5_K_T16_QL_OFFSET:GGUF_Q5_K_T16_QH_OFFSET
    ]
    compact[..., GGUF_Q5_K_QMICRO_T16_QH_OFFSET:] = tiles[..., GGUF_Q5_K_T16_QH_OFFSET:]
    return GGUFQ5KQMicroTile16(
        tiles=compact,
        experts=experts,
        out_features=out_tiles * GGUF_T16_COLS,
        in_features=blocks_per_row * QK_K,
    )


def repack_gguf_q5_k_qmicro_tile16(raw_qweight: Any) -> GGUFQ5KQMicroTile16:
    """Repack raw Q5_K experts into byte-neutral qmicro T16 tiles."""

    return convert_gguf_q5_k_tile16_to_qmicro(repack_gguf_q5_k_tile16(raw_qweight))


def unpack_gguf_q5_k_qmicro_tile16(
    packed: GGUFQ5KQMicroTile16 | np.ndarray,
    *,
    out_features: int | None = None,
) -> np.ndarray:
    """Reconstruct raw GGUF Q5_K bytes from qmicro T16 tiles."""

    if isinstance(packed, GGUFQ5KQMicroTile16):
        tiles = np.asarray(packed.tiles, dtype=np.uint8)
        expected_out = packed.out_features
    else:
        tiles = np.asarray(packed, dtype=np.uint8)
        expected_out = out_features
    if tiles.ndim != 4 or tiles.shape[-1] != GGUF_Q5_K_QMICRO_T16_BLOCK_BYTES:
        raise ValueError("tiles must have shape [experts, out_tiles16, blocks_per_row, 2816]")
    experts, out_tiles, blocks_per_row, _ = map(int, tiles.shape)
    inferred_out = out_tiles * GGUF_T16_COLS
    if expected_out is not None and int(expected_out) != inferred_out:
        raise ValueError(f"out_features mismatch: expected {expected_out}, tile layout implies {inferred_out}")

    expanded = np.empty(
        (experts, out_tiles, blocks_per_row, GGUF_Q5_K_T16_BLOCK_BYTES),
        dtype=np.uint8,
    )
    expanded[..., :GGUF_Q5_K_T16_SCALE_OFFSET] = tiles[..., :GGUF_Q5_K_QMICRO_T16_META_OFFSET]
    records = tiles[
        ..., GGUF_Q5_K_QMICRO_T16_META_OFFSET:GGUF_Q5_K_QMICRO_T16_QL_OFFSET
    ].reshape(
        experts,
        out_tiles,
        blocks_per_row,
        2,
        GGUF_Q5_K_SUBBLOCKS,
        GGUF_T16_COLS // 4,
        3,
    )
    words = (
        records[..., 0].astype(np.uint32)
        | (records[..., 1].astype(np.uint32) << np.uint32(8))
        | (records[..., 2].astype(np.uint32) << np.uint32(16))
    )
    coeffs = np.stack(
        tuple(
            ((words >> np.uint32(6 * lane)) & np.uint32(0x3F)).astype(np.uint8)
            for lane in range(4)
        ),
        axis=-1,
    ).reshape(experts, out_tiles, blocks_per_row, 2, GGUF_Q5_K_SUBBLOCKS, GGUF_T16_COLS)
    expanded[..., GGUF_Q5_K_T16_SCALE_OFFSET:GGUF_Q5_K_T16_MIN_OFFSET] = coeffs[..., 0, :, :].reshape(
        experts, out_tiles, blocks_per_row, -1
    )
    expanded[..., GGUF_Q5_K_T16_MIN_OFFSET:GGUF_Q5_K_T16_QL_OFFSET] = coeffs[..., 1, :, :].reshape(
        experts, out_tiles, blocks_per_row, -1
    )
    expanded[..., GGUF_Q5_K_T16_QL_OFFSET:GGUF_Q5_K_T16_QH_OFFSET] = tiles[
        ..., GGUF_Q5_K_QMICRO_T16_QL_OFFSET:GGUF_Q5_K_QMICRO_T16_QH_OFFSET
    ]
    expanded[..., GGUF_Q5_K_T16_QH_OFFSET:] = tiles[..., GGUF_Q5_K_QMICRO_T16_QH_OFFSET:]
    return unpack_gguf_q5_k_tile16(expanded, out_features=inferred_out)


def convert_gguf_q5_k_qmicro_tile16_to_planar(
    packed: GGUFQ5KQMicroTile16 | np.ndarray,
) -> GGUFQ5KQMicroTile16:
    """Reorder Q5 qmicro ql/qh planes into 12-byte planar dp4a records.

    Each record covers one (subblock, column-quartet, 4-quant pack) and
    stores four col0/col1 low-nibble bytes, four col2/col3 bytes, then four
    bytes of packed per-column high bits - the same shape the Q6 planar
    qmicro conversion produces, so the grouped dp4a kernel structure is
    shared. Adjacent packs are adjacent records for coalesced u32 loads.
    """

    tiles = np.asarray(
        packed.tiles if isinstance(packed, GGUFQ5KQMicroTile16) else packed,
        dtype=np.uint8,
    )
    if tiles.ndim != 4 or tiles.shape[-1] != GGUF_Q5_K_QMICRO_T16_BLOCK_BYTES:
        raise ValueError(
            "tiles must have shape [experts, out_tiles16, blocks_per_row, 2816]"
        )
    experts, out_tiles, blocks_per_row, _ = map(int, tiles.shape)
    planar = np.empty(
        (experts, out_tiles, blocks_per_row, GGUF_Q5_K_QMICRO_PLANAR_T16_BLOCK_BYTES),
        dtype=np.uint8,
    )
    planar[..., : GGUF_Q5_K_QMICRO_T16_QL_OFFSET] = tiles[
        ..., : GGUF_Q5_K_QMICRO_T16_QL_OFFSET
    ]

    ql = tiles[..., GGUF_Q5_K_QMICRO_T16_QL_OFFSET : GGUF_Q5_K_QMICRO_T16_QH_OFFSET].reshape(
        experts, out_tiles, blocks_per_row,
        GGUF_Q5_K_SUBBLOCKS, GGUF_Q5_K_SUBBLOCK, GGUF_T16_COLS // 2,
    )
    qh = tiles[..., GGUF_Q5_K_QMICRO_T16_QH_OFFSET :].reshape(
        experts, out_tiles, blocks_per_row,
        GGUF_Q5_K_SUBBLOCKS, GGUF_Q5_K_SUBBLOCK, GGUF_T16_COLS // 8,
    )
    shifts = np.array([0, 4], dtype=np.uint8).reshape(1, 1, 1, 1, 1, 2, 1)
    low = np.swapaxes((ql[..., None, :] >> shifts) & np.uint8(0x0F), -1, -2).reshape(
        experts, out_tiles, blocks_per_row,
        GGUF_Q5_K_SUBBLOCKS, GGUF_Q5_K_SUBBLOCK, GGUF_T16_COLS,
    )
    bit_shifts = np.arange(8, dtype=np.uint8).reshape(1, 1, 1, 1, 1, 8, 1)
    high = np.swapaxes((qh[..., None, :] >> bit_shifts) & np.uint8(0x01), -1, -2).reshape(
        experts, out_tiles, blocks_per_row,
        GGUF_Q5_K_SUBBLOCKS, GGUF_Q5_K_SUBBLOCK, GGUF_T16_COLS,
    )

    subblocks = GGUF_Q5_K_SUBBLOCKS
    lane32 = GGUF_Q5_K_SUBBLOCK
    records = planar[..., GGUF_Q5_K_QMICRO_T16_QL_OFFSET:].reshape(
        experts, out_tiles, blocks_per_row,
        subblocks, GGUF_T16_COLS // 4, lane32 // 4,
        GGUF_Q5_K_QMICRO_PLANAR_T16_RECORD_BYTES,
    )
    # record[..., sb, quartet, pack]: low01(4B) low23(4B) high(4B) where
    # byte k holds that k's values (low nibbles) / high bits for the 4 cols.
    cols = np.arange(GGUF_T16_COLS).reshape(4, 4)  # [quartet, col4]
    for quartet in range(GGUF_T16_COLS // 4):
        c = cols[quartet]
        low01 = low[..., c[0]] | (low[..., c[1]] << np.uint8(4))  # [.., sb, lane32]
        low23 = low[..., c[2]] | (low[..., c[3]] << np.uint8(4))
        high4 = (
            high[..., c[0]]
            | (high[..., c[1]] << np.uint8(1))
            | (high[..., c[2]] << np.uint8(2))
            | (high[..., c[3]] << np.uint8(3))
        )  # [.., sb, lane32] - 4 bits, one per col
        # lane32 = 4*pack + k
        low01_r = low01.reshape(
            experts, out_tiles, blocks_per_row, subblocks, lane32 // 4, 4
        )
        low23_r = low23.reshape(
            experts, out_tiles, blocks_per_row, subblocks, lane32 // 4, 4
        )
        high_r = high4.reshape(
            experts, out_tiles, blocks_per_row, subblocks, lane32 // 4, 4
        )
        records[:, :, :, :, quartet, :, 0:4] = low01_r
        records[:, :, :, :, quartet, :, 4:8] = low23_r
        records[:, :, :, :, quartet, :, 8:12] = high_r
    return GGUFQ5KQMicroTile16(
        tiles=planar,
        experts=experts,
        out_features=out_tiles * GGUF_T16_COLS,
        in_features=blocks_per_row * QK_K,
    )


def unpack_gguf_q5_k_qmicro_planar_tile16(
    packed: GGUFQ5KQMicroTile16 | np.ndarray,
    *,
    out_features: int | None = None,
) -> np.ndarray:
    """Reconstruct raw GGUF Q5_K bytes from planar qmicro tiles."""

    if isinstance(packed, GGUFQ5KQMicroTile16):
        planar = np.asarray(packed.tiles, dtype=np.uint8)
        expected_out = packed.out_features
    else:
        planar = np.asarray(packed, dtype=np.uint8)
        expected_out = out_features
    if (
        planar.ndim != 4
        or planar.shape[-1] != GGUF_Q5_K_QMICRO_PLANAR_T16_BLOCK_BYTES
    ):
        raise ValueError(
            "tiles must have shape [experts, out_tiles16, blocks_per_row, 3328]"
        )
    experts, out_tiles, blocks_per_row, _ = map(int, planar.shape)
    qmicro = np.empty(
        (experts, out_tiles, blocks_per_row, GGUF_Q5_K_QMICRO_T16_BLOCK_BYTES),
        dtype=np.uint8,
    )
    qmicro[..., : GGUF_Q5_K_QMICRO_T16_QL_OFFSET] = planar[
        ..., : GGUF_Q5_K_QMICRO_T16_QL_OFFSET
    ]
    records = planar[..., GGUF_Q5_K_QMICRO_T16_QL_OFFSET:].reshape(
        experts, out_tiles, blocks_per_row,
        GGUF_Q5_K_SUBBLOCKS, GGUF_T16_COLS // 4, GGUF_Q5_K_SUBBLOCK // 4,
        GGUF_Q5_K_QMICRO_PLANAR_T16_RECORD_BYTES,
    )
    low01 = records[..., 0:4].reshape(
        experts, out_tiles, blocks_per_row,
        GGUF_Q5_K_SUBBLOCKS, GGUF_T16_COLS // 4, GGUF_Q5_K_SUBBLOCK,
    )  # [..., sb, quartet, lane32]
    low23 = records[..., 4:8].reshape(
        experts, out_tiles, blocks_per_row,
        GGUF_Q5_K_SUBBLOCKS, GGUF_T16_COLS // 4, GGUF_Q5_K_SUBBLOCK,
    )
    high4 = records[..., 8:12].reshape(
        experts, out_tiles, blocks_per_row,
        GGUF_Q5_K_SUBBLOCKS, GGUF_T16_COLS // 4, GGUF_Q5_K_SUBBLOCK,
    )
    lane32 = GGUF_Q5_K_SUBBLOCK
    ql_out = qmicro[..., GGUF_Q5_K_QMICRO_T16_QL_OFFSET : GGUF_Q5_K_QMICRO_T16_QH_OFFSET].reshape(
        experts, out_tiles, blocks_per_row,
        GGUF_Q5_K_SUBBLOCKS, lane32, GGUF_T16_COLS // 2,
    )
    qh_out = qmicro[..., GGUF_Q5_K_QMICRO_T16_QH_OFFSET :].reshape(
        experts, out_tiles, blocks_per_row,
        GGUF_Q5_K_SUBBLOCKS, lane32, GGUF_T16_COLS // 8,
    )
    # ql byte (sb, lane32, colpair): colpair even -> low01 of quartet colpair//2;
    # colpair odd -> low23 of quartet colpair//2.
    ql_out[..., :, :, 0::2] = np.moveaxis(low01, -2, -1)  # quartet axis -> colpair slot
    ql_out[..., :, :, 1::2] = np.moveaxis(low23, -2, -1)
    # qh byte (sb, lane32, colbyte): cols 8cb..8cb+7 = quartet 2cb in bits 0-3
    # and quartet 2cb+1 in bits 4-7.
    qh_out[..., :, :, 0] = high4[..., :, 0, :] | (high4[..., :, 1, :] << np.uint8(4))
    qh_out[..., :, :, 1] = high4[..., :, 2, :] | (high4[..., :, 3, :] << np.uint8(4))
    return unpack_gguf_q5_k_qmicro_tile16(
        GGUFQ5KQMicroTile16(
            tiles=qmicro,
            experts=experts,
            out_features=out_tiles * GGUF_T16_COLS,
            in_features=blocks_per_row * QK_K,
        ),
        out_features=out_tiles * GGUF_T16_COLS,
    )


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


def repack_gguf_q6_k_tile16_qmicro(raw_qweight: Any) -> GGUFQ6KTile16:
    """Repack Q6T16 with each K4/column-quartet payload in one 12-byte record.

    Metadata is identical to ``repack_gguf_q6_k_tile16``.  The 3,072-byte
    quant payload is reordered as ``[K32][col4][K4][QL8,QH4]`` so a selected
    prefill work item can replace twelve scalar byte gathers with three
    aligned dword loads.  The layout remains byte-neutral with raw Q6_K.
    """

    return convert_gguf_q6_k_tile16_to_qmicro(
        repack_gguf_q6_k_tile16(raw_qweight)
    )


def convert_gguf_q6_k_tile16_to_qmicro(
    packed: GGUFQ6KTile16 | np.ndarray,
) -> GGUFQ6KTile16:
    """Convert existing Q6T16 tiles to byte-neutral qmicro records."""

    if isinstance(packed, GGUFQ6KTile16):
        legacy_tiles = np.asarray(packed.tiles, dtype=np.uint8)
        expected_out = packed.out_features
    else:
        legacy_tiles = np.asarray(packed, dtype=np.uint8)
        expected_out = None
    if (
        legacy_tiles.ndim != 4
        or legacy_tiles.shape[-1] != GGUF_Q6_K_T16_BLOCK_BYTES
    ):
        raise ValueError(
            "tiles must have shape "
            "[experts, out_tiles16, blocks_per_row, 3360]"
        )
    experts, out_tiles, blocks_per_row, _ = legacy_tiles.shape
    out_features = int(out_tiles) * GGUF_T16_COLS
    if expected_out is not None and int(expected_out) != out_features:
        raise ValueError(
            f"out_features mismatch: expected {expected_out}, "
            f"tile layout implies {out_features}"
        )
    tiles = np.empty_like(legacy_tiles)
    tiles[..., :GGUF_Q6_K_T16_QMICRO_OFFSET] = legacy_tiles[
        ..., :GGUF_Q6_K_T16_QMICRO_OFFSET
    ]
    ql = legacy_tiles[
        ..., GGUF_Q6_K_T16_QL_OFFSET:GGUF_Q6_K_T16_QH_OFFSET
    ].reshape(
        experts,
        out_tiles,
        blocks_per_row,
        8,
        8,
        4,
        4,
        2,
    )
    qh = legacy_tiles[..., GGUF_Q6_K_T16_QH_OFFSET:].reshape(
        experts,
        out_tiles,
        blocks_per_row,
        8,
        8,
        4,
        4,
    )
    records = tiles[..., GGUF_Q6_K_T16_QMICRO_OFFSET:].reshape(
        experts,
        out_tiles,
        blocks_per_row,
        8,
        4,
        8,
        GGUF_Q5_K_QMICRO_PLANAR_T16_RECORD_BYTES,
    )
    records[..., :8] = ql.transpose(0, 1, 2, 3, 6, 4, 5, 7).reshape(
        experts,
        out_tiles,
        blocks_per_row,
        8,
        4,
        8,
        8,
    )
    records[..., 8:] = qh.transpose(0, 1, 2, 3, 6, 4, 5)
    return GGUFQ6KTile16(
        tiles=tiles,
        experts=int(experts),
        out_features=out_features,
        in_features=int(blocks_per_row) * QK_K,
    )


def unpack_gguf_q6_k_tile16_qmicro(
    packed: GGUFQ6KTile16 | np.ndarray,
    *,
    out_features: int | None = None,
) -> np.ndarray:
    """Reconstruct raw GGUF Q6_K bytes from the Q6T16 qmicro layout."""

    if isinstance(packed, GGUFQ6KTile16):
        tiles = np.asarray(packed.tiles, dtype=np.uint8)
        expected_out = packed.out_features
    else:
        tiles = np.asarray(packed, dtype=np.uint8)
        expected_out = out_features
    if tiles.ndim != 4 or tiles.shape[-1] != GGUF_Q6_K_T16_BLOCK_BYTES:
        raise ValueError(
            "tiles must have shape [experts, out_tiles16, blocks_per_row, 3360]"
        )
    experts, out_tiles, blocks_per_row, _ = tiles.shape
    inferred_out = int(out_tiles) * GGUF_T16_COLS
    if expected_out is not None and int(expected_out) != inferred_out:
        raise ValueError(
            f"out_features mismatch: expected {expected_out}, "
            f"tile layout implies {inferred_out}"
        )

    legacy = np.empty_like(tiles)
    legacy[..., :GGUF_Q6_K_T16_QMICRO_OFFSET] = tiles[
        ..., :GGUF_Q6_K_T16_QMICRO_OFFSET
    ]
    records = tiles[..., GGUF_Q6_K_T16_QMICRO_OFFSET:].reshape(
        experts,
        out_tiles,
        blocks_per_row,
        8,
        4,
        8,
        GGUF_Q5_K_QMICRO_PLANAR_T16_RECORD_BYTES,
    )
    legacy[..., GGUF_Q6_K_T16_QL_OFFSET:GGUF_Q6_K_T16_QH_OFFSET] = (
        records[..., :8]
        .reshape(experts, out_tiles, blocks_per_row, 8, 4, 8, 4, 2)
        .transpose(0, 1, 2, 3, 5, 6, 4, 7)
        .reshape(
            experts,
            out_tiles,
            blocks_per_row,
            QK_K * (GGUF_T16_COLS // 2),
        )
    )
    legacy[..., GGUF_Q6_K_T16_QH_OFFSET:] = records[..., 8:].transpose(
        0, 1, 2, 3, 5, 6, 4
    ).reshape(
        experts,
        out_tiles,
        blocks_per_row,
        QK_K * (GGUF_T16_COLS // 4),
    )
    return unpack_gguf_q6_k_tile16(legacy, out_features=inferred_out)


def repack_gguf_q6_k_tile16_qmicro_planar(
    raw_qweight: Any,
) -> GGUFQ6KTile16:
    """Repack Q6T16 into byte-neutral planar qmicro records.

    Each 12-byte record stores four QL01 bytes, then four QL23 bytes, then
    four QH bytes.  This preserves the qmicro record size while making all
    three prefill inputs directly loadable as aligned dwords.
    """

    return convert_gguf_q6_k_tile16_to_qmicro_planar(
        repack_gguf_q6_k_tile16(raw_qweight)
    )


def convert_gguf_q6_k_tile16_to_qmicro_planar(
    packed: GGUFQ6KTile16 | np.ndarray,
) -> GGUFQ6KTile16:
    """Convert existing Q6T16 tiles to byte-neutral planar qmicro records."""

    interleaved = convert_gguf_q6_k_tile16_to_qmicro(packed)
    tiles = interleaved.tiles.copy()
    records = tiles[..., GGUF_Q6_K_T16_QMICRO_OFFSET:].reshape(
        *tiles.shape[:-1],
        8,
        4,
        8,
        GGUF_Q5_K_QMICRO_PLANAR_T16_RECORD_BYTES,
    )
    ql = records[..., :8].copy()
    records[..., :4] = ql[..., 0::2]
    records[..., 4:8] = ql[..., 1::2]
    return GGUFQ6KTile16(
        tiles=tiles,
        experts=interleaved.experts,
        out_features=interleaved.out_features,
        in_features=interleaved.in_features,
    )


def unpack_gguf_q6_k_tile16_qmicro_planar(
    packed: GGUFQ6KTile16 | np.ndarray,
    *,
    out_features: int | None = None,
) -> np.ndarray:
    """Reconstruct raw GGUF Q6_K bytes from planar qmicro records."""

    if isinstance(packed, GGUFQ6KTile16):
        planar_tiles = np.asarray(packed.tiles, dtype=np.uint8)
        expected_out = packed.out_features
    else:
        planar_tiles = np.asarray(packed, dtype=np.uint8)
        expected_out = out_features
    if (
        planar_tiles.ndim != 4
        or planar_tiles.shape[-1] != GGUF_Q6_K_T16_BLOCK_BYTES
    ):
        raise ValueError(
            "tiles must have shape [experts, out_tiles16, blocks_per_row, 3360]"
        )
    tiles = planar_tiles.copy()
    records = tiles[..., GGUF_Q6_K_T16_QMICRO_OFFSET:].reshape(
        *tiles.shape[:-1],
        8,
        4,
        8,
        GGUF_Q5_K_QMICRO_PLANAR_T16_RECORD_BYTES,
    )
    ql01 = records[..., :4].copy()
    ql23 = records[..., 4:8].copy()
    records[..., :8:2] = ql01
    records[..., 1:8:2] = ql23
    return unpack_gguf_q6_k_tile16_qmicro(
        tiles,
        out_features=expected_out,
    )


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
    "GGUF_Q5_K_QMICRO_T16_BLOCK_BYTES",
    "GGUF_Q5_K_QMICRO_T16_DMIN_OFFSET",
    "GGUF_Q5_K_QMICRO_T16_D_OFFSET",
    "GGUF_Q5_K_QMICRO_T16_META_OFFSET",
    "GGUF_Q5_K_QMICRO_T16_QH_OFFSET",
    "GGUF_Q5_K_QMICRO_T16_QL_OFFSET",
    "GGUF_Q5_K_QMICRO_T16_V1",
    "GGUF_Q5_K_T16_BLOCK_BYTES",
    "GGUF_Q5_K_T16_V1",
    "GGUF_Q6_K_BLOCK_BYTES",
    "GGUF_Q6_K_T16_BLOCK_BYTES",
    "GGUF_Q6_K_T16_QMICRO_OFFSET",
    "GGUF_Q6_K_T16_QMICRO_PLANAR_V1",
    "GGUF_Q6_K_T16_QMICRO_RECORD_BYTES",
    "GGUF_Q6_K_T16_V1",
    "GGUF_Q8_0_BLOCK_BYTES",
    "GGUF_Q8_0_T16_BLOCK_BYTES",
    "GGUF_Q8_0_T16_V1",
    "GGUF_T16_COLS",
    "GGUFQ5KQMicroT16Quant",
    "GGUFQ5KQMicroTile16",
    "GGUFQ5KTile16",
    "GGUFQ5KT16Quant",
    "GGUFQ6KTile16",
    "GGUFQ6KT16QMicroPlanarQuant",
    "GGUFQ6KT16Quant",
    "GGUFQ80Tile16",
    "GGUFQ80T16Quant",
    "convert_gguf_q5_k_tile16_to_qmicro",
    "convert_gguf_q6_k_tile16_to_qmicro",
    "convert_gguf_q6_k_tile16_to_qmicro_planar",
    "repack_gguf_q5_k_qmicro_tile16",
    "repack_gguf_q5_k_tile16",
    "repack_gguf_q6_k_tile16",
    "repack_gguf_q6_k_tile16_qmicro",
    "repack_gguf_q6_k_tile16_qmicro_planar",
    "repack_gguf_q8_0_tile16",
    "unpack_gguf_q5_k_qmicro_tile16",
    "unpack_gguf_q5_k_tile16",
    "unpack_gguf_q6_k_tile16",
    "unpack_gguf_q6_k_tile16_qmicro",
    "unpack_gguf_q6_k_tile16_qmicro_planar",
    "unpack_gguf_q8_0_tile16",
]
