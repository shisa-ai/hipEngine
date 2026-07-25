"""GGUF Q4_K quantization plugin metadata and repack helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hipengine.quant.gguf import QK_K, unpack_q4_k_scale_min
from hipengine.quant.registry import register_quant

GGUF_Q4_K_BLOCK_BYTES = 144
GGUF_Q4_K_SUBBLOCK = 32
GGUF_Q4_K_SUBBLOCKS = 8
GGUF_Q4_K_PACK = 8
GGUF_Q8_1_BLOCK = 32
GGUF_Q8_1_MMQ_BLOCK = 4 * GGUF_Q8_1_BLOCK
GGUF_Q8_1_MMQ_DS4_BYTES = 8 * np.dtype(np.uint16).itemsize + GGUF_Q8_1_MMQ_BLOCK

# P9.C12/P9.C13 tile-major Q4_K replacement-layout prototype.
# One tile stores 16 adjacent output columns for one 256-wide K block.
GGUF_Q4_K_TILE16_COLS = 16
GGUF_Q4_K_TILE16_D_OFFSET = 0
GGUF_Q4_K_TILE16_DMIN_OFFSET = GGUF_Q4_K_TILE16_D_OFFSET + GGUF_Q4_K_TILE16_COLS * 2
GGUF_Q4_K_TILE16_SCALE_OFFSET = GGUF_Q4_K_TILE16_DMIN_OFFSET + GGUF_Q4_K_TILE16_COLS * 2
GGUF_Q4_K_TILE16_MIN_OFFSET = GGUF_Q4_K_TILE16_SCALE_OFFSET + GGUF_Q4_K_SUBBLOCKS * GGUF_Q4_K_TILE16_COLS
GGUF_Q4_K_TILE16_Q_OFFSET = GGUF_Q4_K_TILE16_MIN_OFFSET + GGUF_Q4_K_SUBBLOCKS * GGUF_Q4_K_TILE16_COLS
GGUF_Q4_K_TILE16_BLOCK_BYTES = GGUF_Q4_K_TILE16_Q_OFFSET + GGUF_Q4_K_SUBBLOCKS * GGUF_Q4_K_SUBBLOCK * (GGUF_Q4_K_TILE16_COLS // 2)


@dataclass(frozen=True)
class GGUFQ4KQuant:
    """GGUF block_q4_K weight-only quantization contract.

    The GGUF tensor layout is block-256 with eight 32-value subblocks.  Each
    block carries fp16 ``d``/``dmin`` plus packed 6-bit scale/min metadata; the
    HIP kernels preserve that math instead of translating to PARO/AWQ zeros.
    """

    name: str = "gguf_q4_k"
    weight_storage: str = "gguf_block_q4_k"
    activation_preprocess: str = "none"
    compute_dtype: str = "fp32_accum"
    scale_granularity: str = "block256_subblock32_scale_min"
    calibration_artifact: str = "gguf"
    kernel_family: str = "gguf_q4_k_gemv"


@dataclass(frozen=True)
class GGUFQ4KT16Quant:
    """T16 replacement-layout plugin key for GGUF block_q4_K weights."""

    name: str = "gguf_q4_k_t16_v1"
    weight_storage: str = "gguf_block_q4_k_t16_v1"
    activation_preprocess: str = "none"
    compute_dtype: str = "fp32_accum"
    scale_granularity: str = "block256_subblock32_scale_min"
    calibration_artifact: str = "gguf"
    kernel_family: str = "gguf_t16_gemv"


@dataclass(frozen=True)
class GGUFQ4KPack8:
    """Lossless pack8 layout for GGUF Q4_K GEMV.

    ``qweight`` has shape ``[out_features / 8, in_features]`` and packs one
    4-bit Q4_K value for each of eight adjacent output channels into an int32.
    ``scales`` and ``mins`` have shape ``[in_features / 32, out_features]`` and
    store the precomputed FP32 terms for ``q * scale - min``.
    """

    qweight: np.ndarray
    scales: np.ndarray
    mins: np.ndarray
    in_features: int
    out_features: int

    @property
    def out_packed(self) -> int:
        return self.out_features // GGUF_Q4_K_PACK


@dataclass(frozen=True)
class GGUFQ4KTile16:
    """P9.C13 tile-major Q4_K prototype layout.

    ``tiles`` has shape ``[experts, out_tiles16, blocks_per_row, 2368]``.
    Each tile stores 16 adjacent output columns for one 256-wide K block:

    - fp16 ``d`` bits for each output column.
    - fp16 ``dmin`` bits for each output column.
    - predecoded uint8 scale/min for each of the 8 subblocks and 16 columns.
    - q4 nibbles arranged by ``[subblock, k_lane32, col_pair]`` so a WMMA
      kernel can load a B-fragment column vector with contiguous/coalesced
      accesses instead of 16 strided raw GGUF rows.

    This is a replacement-layout prototype, not a sidecar contract for default
    runtime use. A replay prototype may allocate it next to raw weights; a
    retained runtime path should materialize it instead of raw Q4 gate/up.
    """

    tiles: np.ndarray
    experts: int
    out_features: int
    in_features: int

    @property
    def out_tiles(self) -> int:
        return self.out_features // GGUF_Q4_K_TILE16_COLS

    @property
    def blocks_per_row(self) -> int:
        return self.in_features // QK_K


@dataclass(frozen=True)
class GGUFQ4KMMQTile16Preview:
    """Host-side preview of the llama.cpp-style Q4_K MMQ tile operands.

    This is a diagnostic/oracle layout for the Q8_1-MMQ port, not a resident
    runtime weight format.  ``q4`` stores unpacked Q4_K nibbles for 16 adjacent
    output columns; ``scales`` and ``mins`` store the precomputed FP32
    ``d*scale`` and ``dmin*min`` terms used after the integer dot.
    """

    q4: np.ndarray
    scales: np.ndarray
    mins: np.ndarray
    in_features: int
    out_features: int

    @property
    def out_tiles(self) -> int:
        return self.out_features // GGUF_Q4_K_TILE16_COLS

    @property
    def blocks_per_row(self) -> int:
        return self.in_features // QK_K


GGUF_Q4_K = register_quant(GGUFQ4KQuant())
GGUF_Q4_K_T16_V1 = register_quant(GGUFQ4KT16Quant())


def awq_pack8_shift_for_lane(lane: int) -> int:
    if lane < 0 or lane >= GGUF_Q4_K_PACK:
        raise ValueError("lane must be in [0, 7]")
    packed_pos = (4 + (lane >> 1)) if (lane & 1) else (lane >> 1)
    return packed_pos * 4



def _pack_q4_k_scale_min(scales: np.ndarray, mins: np.ndarray) -> np.ndarray:
    """Inverse of ``unpack_q4_k_scale_min`` for uint8 scale/min arrays."""

    sc = np.asarray(scales, dtype=np.uint8)
    mn = np.asarray(mins, dtype=np.uint8)
    if sc.shape != mn.shape or sc.shape[-1] != GGUF_Q4_K_SUBBLOCKS:
        raise ValueError("scales/mins must have matching shape ending in 8")
    packed = np.empty((*sc.shape[:-1], 12), dtype=np.uint8)
    packed[..., 0:4] = (sc[..., 0:4] & np.uint8(0x3F)) | ((sc[..., 4:8] & np.uint8(0x30)) << np.uint8(2))
    packed[..., 4:8] = (mn[..., 0:4] & np.uint8(0x3F)) | ((mn[..., 4:8] & np.uint8(0x30)) << np.uint8(2))
    packed[..., 8:12] = (sc[..., 4:8] & np.uint8(0x0F)) | ((mn[..., 4:8] & np.uint8(0x0F)) << np.uint8(4))
    return packed


def repack_gguf_q4_k_tile16(raw_qweight: Any) -> GGUFQ4KTile16:
    """Repack rank-3 raw GGUF Q4_K expert weights into Q4T16 tiles.

    ``raw_qweight`` must have GGUF byte shape ``[experts, out_features,
    bytes_per_row]``. The repack is bit-lossless: ``unpack_gguf_q4_k_tile16``
    reconstructs the original raw bytes exactly.
    """

    raw = np.ascontiguousarray(raw_qweight, dtype=np.uint8)
    if raw.ndim != 3:
        raise ValueError("raw_qweight must have GGUF expert byte shape [experts, out_features, bytes_per_row]")
    experts, out_features, bytes_per_row = (int(raw.shape[0]), int(raw.shape[1]), int(raw.shape[2]))
    if experts <= 0:
        raise ValueError("experts must be positive")
    if out_features <= 0 or out_features % GGUF_Q4_K_TILE16_COLS != 0:
        raise ValueError("out_features must be positive and divisible by 16")
    if bytes_per_row <= 0 or bytes_per_row % GGUF_Q4_K_BLOCK_BYTES != 0:
        raise ValueError("bytes_per_row must be a positive multiple of 144")

    blocks_per_row = bytes_per_row // GGUF_Q4_K_BLOCK_BYTES
    out_tiles = out_features // GGUF_Q4_K_TILE16_COLS
    blocks = raw.reshape(experts, out_features, blocks_per_row, GGUF_Q4_K_BLOCK_BYTES)
    tiles = np.empty(
        (experts, out_tiles, blocks_per_row, GGUF_Q4_K_TILE16_BLOCK_BYTES),
        dtype=np.uint8,
    )

    for out_tile in range(out_tiles):
        cols = blocks[:, out_tile * GGUF_Q4_K_TILE16_COLS : (out_tile + 1) * GGUF_Q4_K_TILE16_COLS]
        # cols: [E, 16, B, 144] -> tile slabs [E, B, ...]
        dst = tiles[:, out_tile]
        dst[..., GGUF_Q4_K_TILE16_D_OFFSET:GGUF_Q4_K_TILE16_DMIN_OFFSET] = (
            cols[..., 0:2].transpose(0, 2, 1, 3).reshape(experts, blocks_per_row, GGUF_Q4_K_TILE16_COLS * 2)
        )
        dst[..., GGUF_Q4_K_TILE16_DMIN_OFFSET:GGUF_Q4_K_TILE16_SCALE_OFFSET] = (
            cols[..., 2:4].transpose(0, 2, 1, 3).reshape(experts, blocks_per_row, GGUF_Q4_K_TILE16_COLS * 2)
        )

        sc, mn = unpack_q4_k_scale_min(cols[..., 4:16].reshape(-1, 12))
        sc = sc.reshape(experts, GGUF_Q4_K_TILE16_COLS, blocks_per_row, GGUF_Q4_K_SUBBLOCKS)
        mn = mn.reshape(experts, GGUF_Q4_K_TILE16_COLS, blocks_per_row, GGUF_Q4_K_SUBBLOCKS)
        dst[..., GGUF_Q4_K_TILE16_SCALE_OFFSET:GGUF_Q4_K_TILE16_MIN_OFFSET] = (
            sc.transpose(0, 2, 3, 1).reshape(experts, blocks_per_row, GGUF_Q4_K_SUBBLOCKS * GGUF_Q4_K_TILE16_COLS)
        )
        dst[..., GGUF_Q4_K_TILE16_MIN_OFFSET:GGUF_Q4_K_TILE16_Q_OFFSET] = (
            mn.transpose(0, 2, 3, 1).reshape(experts, blocks_per_row, GGUF_Q4_K_SUBBLOCKS * GGUF_Q4_K_TILE16_COLS)
        )

        qs_pairs = cols[..., 16:144].reshape(experts, GGUF_Q4_K_TILE16_COLS, blocks_per_row, 4, GGUF_Q4_K_SUBBLOCK)
        q = np.empty(
            (experts, GGUF_Q4_K_TILE16_COLS, blocks_per_row, GGUF_Q4_K_SUBBLOCKS, GGUF_Q4_K_SUBBLOCK),
            dtype=np.uint8,
        )
        for sb in range(GGUF_Q4_K_SUBBLOCKS):
            packed = qs_pairs[..., sb >> 1, :]
            q[..., sb, :] = ((packed >> np.uint8(4)) if (sb & 1) else packed) & np.uint8(0x0F)
        q_tile = q.transpose(0, 2, 3, 4, 1)  # [E, B, sb, lane32, col]
        q_packed_cols = (q_tile[..., 0::2] & np.uint8(0x0F)) | ((q_tile[..., 1::2] & np.uint8(0x0F)) << np.uint8(4))
        dst[..., GGUF_Q4_K_TILE16_Q_OFFSET:] = q_packed_cols.reshape(
            experts,
            blocks_per_row,
            GGUF_Q4_K_SUBBLOCKS * GGUF_Q4_K_SUBBLOCK * (GGUF_Q4_K_TILE16_COLS // 2),
        )

    return GGUFQ4KTile16(
        tiles=tiles,
        experts=experts,
        out_features=out_features,
        in_features=blocks_per_row * QK_K,
    )


def unpack_gguf_q4_k_tile16(packed: GGUFQ4KTile16 | np.ndarray, *, out_features: int | None = None) -> np.ndarray:
    """Reconstruct raw GGUF Q4_K expert bytes from a Q4T16 tile layout."""

    if isinstance(packed, GGUFQ4KTile16):
        tiles = np.asarray(packed.tiles, dtype=np.uint8)
        expected_out = packed.out_features
    else:
        tiles = np.asarray(packed, dtype=np.uint8)
        expected_out = out_features
    if tiles.ndim != 4 or tiles.shape[-1] != GGUF_Q4_K_TILE16_BLOCK_BYTES:
        raise ValueError("tiles must have shape [experts, out_tiles16, blocks_per_row, 2368]")
    experts, out_tiles, blocks_per_row, _ = (int(tiles.shape[0]), int(tiles.shape[1]), int(tiles.shape[2]), int(tiles.shape[3]))
    inferred_out = out_tiles * GGUF_Q4_K_TILE16_COLS
    if expected_out is not None and int(expected_out) != inferred_out:
        raise ValueError(f"out_features mismatch: expected {expected_out}, tile layout implies {inferred_out}")

    blocks = np.empty(
        (experts, inferred_out, blocks_per_row, GGUF_Q4_K_BLOCK_BYTES),
        dtype=np.uint8,
    )
    for out_tile in range(out_tiles):
        src = tiles[:, out_tile]
        cols = blocks[:, out_tile * GGUF_Q4_K_TILE16_COLS : (out_tile + 1) * GGUF_Q4_K_TILE16_COLS]
        cols[..., 0:2] = src[..., GGUF_Q4_K_TILE16_D_OFFSET:GGUF_Q4_K_TILE16_DMIN_OFFSET].reshape(
            experts, blocks_per_row, GGUF_Q4_K_TILE16_COLS, 2
        ).transpose(0, 2, 1, 3)
        cols[..., 2:4] = src[..., GGUF_Q4_K_TILE16_DMIN_OFFSET:GGUF_Q4_K_TILE16_SCALE_OFFSET].reshape(
            experts, blocks_per_row, GGUF_Q4_K_TILE16_COLS, 2
        ).transpose(0, 2, 1, 3)
        sc = src[..., GGUF_Q4_K_TILE16_SCALE_OFFSET:GGUF_Q4_K_TILE16_MIN_OFFSET].reshape(
            experts, blocks_per_row, GGUF_Q4_K_SUBBLOCKS, GGUF_Q4_K_TILE16_COLS
        ).transpose(0, 3, 1, 2)
        mn = src[..., GGUF_Q4_K_TILE16_MIN_OFFSET:GGUF_Q4_K_TILE16_Q_OFFSET].reshape(
            experts, blocks_per_row, GGUF_Q4_K_SUBBLOCKS, GGUF_Q4_K_TILE16_COLS
        ).transpose(0, 3, 1, 2)
        cols[..., 4:16] = _pack_q4_k_scale_min(sc, mn)

        q_packed_cols = src[..., GGUF_Q4_K_TILE16_Q_OFFSET:].reshape(
            experts,
            blocks_per_row,
            GGUF_Q4_K_SUBBLOCKS,
            GGUF_Q4_K_SUBBLOCK,
            GGUF_Q4_K_TILE16_COLS // 2,
        )
        q = np.empty(
            (experts, blocks_per_row, GGUF_Q4_K_SUBBLOCKS, GGUF_Q4_K_SUBBLOCK, GGUF_Q4_K_TILE16_COLS),
            dtype=np.uint8,
        )
        q[..., 0::2] = q_packed_cols & np.uint8(0x0F)
        q[..., 1::2] = q_packed_cols >> np.uint8(4)
        q_by_col = q.transpose(0, 4, 1, 2, 3)  # [E, col, B, sb, lane32]
        qs_pairs = np.empty((experts, GGUF_Q4_K_TILE16_COLS, blocks_per_row, 4, GGUF_Q4_K_SUBBLOCK), dtype=np.uint8)
        for pair in range(4):
            qs_pairs[..., pair, :] = (q_by_col[..., 2 * pair, :] & np.uint8(0x0F)) | (
                (q_by_col[..., 2 * pair + 1, :] & np.uint8(0x0F)) << np.uint8(4)
            )
        cols[..., 16:144] = qs_pairs.reshape(experts, GGUF_Q4_K_TILE16_COLS, blocks_per_row, 128)

    return blocks.reshape(experts, inferred_out, blocks_per_row * GGUF_Q4_K_BLOCK_BYTES)


def _bf16_u16_to_f32(arr: object) -> np.ndarray:
    u16 = np.ascontiguousarray(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(u16.shape).copy()


def pack_q8_1_mmq_ds4_from_bf16(x_bf16: object) -> np.ndarray:
    """Pack BF16 activations into llama.cpp-style DS4 ``block_q8_1_mmq`` bytes.

    The output has byte shape ``[rows, hidden / 128, 144]``. Each 144-byte block
    stores four FP16 ``(d, sum)`` pairs followed by 128 signed int8 quants. The
    sum is the original BF16 activation sum for each 32-wide Q8_1 subblock.
    """

    x = _bf16_u16_to_f32(x_bf16).astype(np.float32, copy=False)
    if x.ndim != 2:
        raise ValueError("x_bf16 must have shape [rows, hidden]")
    rows, hidden = int(x.shape[0]), int(x.shape[1])
    if rows <= 0:
        raise ValueError("rows must be positive")
    if hidden <= 0 or hidden % GGUF_Q8_1_MMQ_BLOCK:
        raise ValueError("hidden dimension must be a positive multiple of 128 for DS4 Q8_1 MMQ")

    blocks = x.reshape(rows, hidden // GGUF_Q8_1_MMQ_BLOCK, 4, GGUF_Q8_1_BLOCK)
    max_abs = np.max(np.abs(blocks), axis=-1)
    d = (max_abs / 127.0).astype(np.float32)
    safe_d = np.where(d > 0.0, d, 1.0).astype(np.float32)
    qs = np.rint(blocks / safe_d[..., None]).clip(-127, 127).astype(np.int8)
    qs = np.where(d[..., None] > 0.0, qs, np.zeros_like(qs)).astype(np.int8, copy=False)
    sums = blocks.sum(axis=-1, dtype=np.float32).astype(np.float32)

    ds4 = np.empty((*d.shape, 2), dtype=np.uint16)
    ds4[..., 0] = d.astype(np.float16).view(np.uint16)
    ds4[..., 1] = sums.astype(np.float16).view(np.uint16)
    out = np.empty((rows, hidden // GGUF_Q8_1_MMQ_BLOCK, GGUF_Q8_1_MMQ_DS4_BYTES), dtype=np.uint8)
    out[..., :16] = ds4.reshape(rows, hidden // GGUF_Q8_1_MMQ_BLOCK, 8).view(np.uint8)
    out[..., 16:] = qs.reshape(rows, hidden // GGUF_Q8_1_MMQ_BLOCK, GGUF_Q8_1_MMQ_BLOCK).view(np.uint8)
    return np.ascontiguousarray(out)


def pack_gguf_q4_k_mmq_tile16_preview(raw_qweight: Any) -> GGUFQ4KMMQTile16Preview:
    """Build a host oracle preview of 16-column Q4_K MMQ tile operands.

    ``raw_qweight`` must have GGUF byte shape ``[out_features, bytes_per_row]``
    for one expert. The returned arrays are shaped like the future 16-column
    WMMA tile contract: ``q4`` is ``[out_tiles, 16, blocks, 8, 32]`` and
    ``scales``/``mins`` are ``[out_tiles, 16, blocks, 8]``.
    """

    raw = np.ascontiguousarray(raw_qweight, dtype=np.uint8)
    if raw.ndim != 2:
        raise ValueError("raw_qweight must have GGUF byte shape [out_features, bytes_per_row]")
    out_features, bytes_per_row = (int(raw.shape[0]), int(raw.shape[1]))
    if out_features <= 0 or out_features % GGUF_Q4_K_TILE16_COLS != 0:
        raise ValueError("out_features must be positive and divisible by 16")
    if bytes_per_row <= 0 or bytes_per_row % GGUF_Q4_K_BLOCK_BYTES != 0:
        raise ValueError("bytes_per_row must be a positive multiple of 144")

    blocks_per_row = bytes_per_row // GGUF_Q4_K_BLOCK_BYTES
    blocks = raw.reshape(out_features, blocks_per_row, GGUF_Q4_K_BLOCK_BYTES)
    d = blocks[..., 0:2].copy().view(np.float16).astype(np.float32).reshape(out_features, blocks_per_row)
    dmin = blocks[..., 2:4].copy().view(np.float16).astype(np.float32).reshape(out_features, blocks_per_row)
    sc, mn = unpack_q4_k_scale_min(blocks[..., 4:16].reshape(-1, 12))
    sc = sc.reshape(out_features, blocks_per_row, GGUF_Q4_K_SUBBLOCKS)
    mn = mn.reshape(out_features, blocks_per_row, GGUF_Q4_K_SUBBLOCKS)
    scales = d[..., None] * sc.astype(np.float32)
    mins = dmin[..., None] * mn.astype(np.float32)

    qs_pairs = blocks[..., 16:144].reshape(out_features, blocks_per_row, 4, GGUF_Q4_K_SUBBLOCK)
    q4_by_col = np.empty((out_features, blocks_per_row, GGUF_Q4_K_SUBBLOCKS, GGUF_Q4_K_SUBBLOCK), dtype=np.uint8)
    for sb in range(GGUF_Q4_K_SUBBLOCKS):
        packed = qs_pairs[..., sb >> 1, :]
        q4_by_col[..., sb, :] = ((packed >> np.uint8(4)) if (sb & 1) else packed) & np.uint8(0x0F)

    out_tiles = out_features // GGUF_Q4_K_TILE16_COLS
    return GGUFQ4KMMQTile16Preview(
        q4=q4_by_col.reshape(out_tiles, GGUF_Q4_K_TILE16_COLS, blocks_per_row, GGUF_Q4_K_SUBBLOCKS, GGUF_Q4_K_SUBBLOCK),
        scales=scales.reshape(out_tiles, GGUF_Q4_K_TILE16_COLS, blocks_per_row, GGUF_Q4_K_SUBBLOCKS).astype(np.float32),
        mins=mins.reshape(out_tiles, GGUF_Q4_K_TILE16_COLS, blocks_per_row, GGUF_Q4_K_SUBBLOCKS).astype(np.float32),
        in_features=blocks_per_row * QK_K,
        out_features=out_features,
    )


def _fp16_u16_to_f32(bits: np.ndarray) -> np.ndarray:
    u16 = np.ascontiguousarray(bits, dtype=np.uint16)
    return u16.view(np.float16).astype(np.float32).reshape(u16.shape)


def gguf_q4_k_mmq_tile16_preview_matmul(x_q8_ds4: object, preview: GGUFQ4KMMQTile16Preview) -> np.ndarray:
    """Evaluate the DS4 Q8_1 x Q4_K MMQ preview in FP32 on CPU.

    This intentionally mirrors the scalar formula that the tiled WMMA port must
    preserve: ``sum((q4 * xq) * q4_scale * q8_d - q4_min * q8_sum)`` over each
    32-wide subblock.
    """

    x = np.ascontiguousarray(x_q8_ds4, dtype=np.uint8)
    if x.ndim != 3 or x.shape[-1] != GGUF_Q8_1_MMQ_DS4_BYTES:
        raise ValueError("x_q8_ds4 must have byte shape [rows, q8_blocks, 144]")
    rows = int(x.shape[0])
    expected_q8_blocks = preview.blocks_per_row * 2
    if int(x.shape[1]) != expected_q8_blocks:
        raise ValueError(f"x_q8_ds4 block count mismatch: expected {expected_q8_blocks}")

    out = np.zeros((rows, preview.out_features), dtype=np.float32)
    for out_tile in range(preview.out_tiles):
        for col in range(GGUF_Q4_K_TILE16_COLS):
            acc = np.zeros(rows, dtype=np.float32)
            for blk in range(preview.blocks_per_row):
                for sb in range(GGUF_Q4_K_SUBBLOCKS):
                    xb = x[:, blk * 2 + (sb >> 2)]
                    ds4 = xb[:, :16].copy().view(np.uint16).reshape(rows, 4, 2)
                    xd = _fp16_u16_to_f32(ds4[:, sb & 3, 0])
                    xsum = _fp16_u16_to_f32(ds4[:, sb & 3, 1])
                    xq_start = 16 + (sb & 3) * GGUF_Q8_1_BLOCK
                    xq = xb[:, xq_start : xq_start + GGUF_Q8_1_BLOCK].view(np.int8).astype(np.int32)
                    q4 = preview.q4[out_tile, col, blk, sb].astype(np.int32)
                    dot = xq @ q4
                    acc += (
                        preview.scales[out_tile, col, blk, sb] * xd * dot.astype(np.float32)
                        - preview.mins[out_tile, col, blk, sb] * xsum
                    ).astype(np.float32)
            out[:, out_tile * GGUF_Q4_K_TILE16_COLS + col] = acc
    return out


def repack_gguf_q4_k_pack8(raw_qweight: Any) -> GGUFQ4KPack8:
    """Repack raw GGUF ``block_q4_K`` bytes into a pack8 GEMV layout.

    The repack is lossless with respect to GGUF Q4_K math.  It moves the 4-bit
    quants for eight adjacent output channels into one int32 and precomputes
    per-32-value FP32 scale/min terms, avoiding repeated raw metadata decode in
    the device kernel.
    """

    raw = np.asarray(raw_qweight, dtype=np.uint8)
    if raw.ndim != 2:
        raise ValueError("raw_qweight must have GGUF byte shape [out_features, bytes_per_row]")
    out_features, bytes_per_row = (int(raw.shape[0]), int(raw.shape[1]))
    if out_features <= 0 or out_features % GGUF_Q4_K_PACK != 0:
        raise ValueError("out_features must be positive and divisible by 8")
    if bytes_per_row <= 0 or bytes_per_row % GGUF_Q4_K_BLOCK_BYTES != 0:
        raise ValueError("bytes_per_row must be a positive multiple of 144")

    blocks_per_row = bytes_per_row // GGUF_Q4_K_BLOCK_BYTES
    in_features = blocks_per_row * QK_K
    out_packed = out_features // GGUF_Q4_K_PACK
    groups32 = blocks_per_row * GGUF_Q4_K_SUBBLOCKS
    blocks = raw.reshape(out_features, blocks_per_row, GGUF_Q4_K_BLOCK_BYTES)
    qweight_u32 = np.zeros((out_packed, in_features), dtype=np.uint32)
    scales = np.empty((groups32, out_features), dtype=np.float32)
    mins = np.empty((groups32, out_features), dtype=np.float32)

    for out_col in range(out_features):
        lane = out_col & (GGUF_Q4_K_PACK - 1)
        out_pack = out_col >> 3
        shift = awq_pack8_shift_for_lane(lane)
        raw_blocks = blocks[out_col]
        d = raw_blocks[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
        dmin = raw_blocks[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(-1)
        sc, minv = unpack_q4_k_scale_min(raw_blocks[:, 4:16])
        group_slice = slice(0, groups32)
        scales[group_slice, out_col] = (d[:, None] * sc.astype(np.float32)).reshape(-1)
        mins[group_slice, out_col] = (dmin[:, None] * minv.astype(np.float32)).reshape(-1)

        qs = raw_blocks[:, 16:144].reshape(blocks_per_row, 4, 1, GGUF_Q4_K_SUBBLOCK)
        q_groups = (qs >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)) & np.uint8(0x0F)
        q_values = q_groups.reshape(blocks_per_row, QK_K).astype(np.uint32).reshape(-1)
        qweight_u32[out_pack] |= q_values << np.uint32(shift)

    return GGUFQ4KPack8(
        qweight=qweight_u32.view(np.int32),
        scales=scales,
        mins=mins,
        in_features=in_features,
        out_features=out_features,
    )


__all__ = [
    "GGUF_Q4_K",
    "GGUF_Q4_K_BLOCK_BYTES",
    "GGUF_Q4_K_PACK",
    "GGUF_Q4_K_TILE16_BLOCK_BYTES",
    "GGUF_Q4_K_TILE16_COLS",
    "GGUF_Q4_K_SUBBLOCK",
    "GGUF_Q4_K_SUBBLOCKS",
    "GGUF_Q4_K_T16_V1",
    "GGUF_Q8_1_BLOCK",
    "GGUF_Q8_1_MMQ_BLOCK",
    "GGUF_Q8_1_MMQ_DS4_BYTES",
    "GGUFQ4KMMQTile16Preview",
    "GGUFQ4KPack8",
    "GGUFQ4KTile16",
    "GGUFQ4KQuant",
    "GGUFQ4KT16Quant",
    "awq_pack8_shift_for_lane",
    "gguf_q4_k_mmq_tile16_preview_matmul",
    "pack_gguf_q4_k_mmq_tile16_preview",
    "pack_q8_1_mmq_ds4_from_bf16",
    "repack_gguf_q4_k_pack8",
    "repack_gguf_q4_k_tile16",
    "unpack_gguf_q4_k_tile16",
]
