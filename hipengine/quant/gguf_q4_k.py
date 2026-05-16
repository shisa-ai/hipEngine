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


GGUF_Q4_K = register_quant(GGUFQ4KQuant())


def awq_pack8_shift_for_lane(lane: int) -> int:
    if lane < 0 or lane >= GGUF_Q4_K_PACK:
        raise ValueError("lane must be in [0, 7]")
    packed_pos = (4 + (lane >> 1)) if (lane & 1) else (lane >> 1)
    return packed_pos * 4


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
    "GGUF_Q4_K_SUBBLOCK",
    "GGUF_Q4_K_SUBBLOCKS",
    "GGUFQ4KPack8",
    "GGUFQ4KQuant",
    "awq_pack8_shift_for_lane",
    "repack_gguf_q4_k_pack8",
]
