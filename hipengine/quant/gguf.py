"""GGUF/GGML quant layout metadata and CPU dequant helpers.

This module is intentionally torch-free.  It records the on-disk GGML tensor
layouts used inside GGUF files and provides small NumPy CPU dequantizers for
loader/fallback validation.  Native HIP execution should still register its own
quant plugins/kernels instead of special-casing these formats in dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import prod
from typing import Callable, Sequence

import numpy as np

QK_K = 256


class GGMLQuantizationType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    TQ1_0 = 34
    TQ2_0 = 35
    MXFP4 = 39
    NVFP4 = 40
    Q1_0 = 41


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class LlamaFileType(IntEnum):
    ALL_F32 = 0
    MOSTLY_F16 = 1
    MOSTLY_Q4_0 = 2
    MOSTLY_Q4_1 = 3
    MOSTLY_Q8_0 = 7
    MOSTLY_Q5_0 = 8
    MOSTLY_Q5_1 = 9
    MOSTLY_Q2_K = 10
    MOSTLY_Q3_K_S = 11
    MOSTLY_Q3_K_M = 12
    MOSTLY_Q3_K_L = 13
    MOSTLY_Q4_K_S = 14
    MOSTLY_Q4_K_M = 15
    MOSTLY_Q5_K_S = 16
    MOSTLY_Q5_K_M = 17
    MOSTLY_Q6_K = 18
    MOSTLY_IQ2_XXS = 19
    MOSTLY_IQ2_XS = 20
    MOSTLY_Q2_K_S = 21
    MOSTLY_IQ3_XS = 22
    MOSTLY_IQ3_XXS = 23
    MOSTLY_IQ1_S = 24
    MOSTLY_IQ4_NL = 25
    MOSTLY_IQ3_S = 26
    MOSTLY_IQ3_M = 27
    MOSTLY_IQ2_S = 28
    MOSTLY_IQ2_M = 29
    MOSTLY_IQ4_XS = 30
    MOSTLY_IQ1_M = 31
    MOSTLY_BF16 = 32
    MOSTLY_TQ1_0 = 36
    MOSTLY_TQ2_0 = 37
    MOSTLY_MXFP4_MOE = 38
    MOSTLY_NVFP4 = 39
    MOSTLY_Q1_0 = 40
    GUESSED = 1024


@dataclass(frozen=True)
class GGUFQuantLayout:
    """GGML tensor storage metadata for one GGUF tensor type."""

    type_id: int
    name: str
    block_size: int
    type_size: int
    storage_dtype: str
    dequant_supported: bool = False
    native_status: str = "unsupported"

    @property
    def is_block_quantized(self) -> bool:
        return self.block_size != 1 or self.storage_dtype == "uint8_blocks"


def _layout(
    qtype: GGMLQuantizationType,
    block_size: int,
    type_size: int,
    storage_dtype: str,
    *,
    dequant_supported: bool = False,
    native_status: str = "unsupported",
) -> GGUFQuantLayout:
    return GGUFQuantLayout(
        type_id=int(qtype),
        name=qtype.name,
        block_size=block_size,
        type_size=type_size,
        storage_dtype=storage_dtype,
        dequant_supported=dequant_supported,
        native_status=native_status,
    )


GGUF_QUANT_LAYOUTS: dict[GGMLQuantizationType, GGUFQuantLayout] = {
    GGMLQuantizationType.F32: _layout(
        GGMLQuantizationType.F32, 1, 4, "float32", dequant_supported=True
    ),
    GGMLQuantizationType.F16: _layout(
        GGMLQuantizationType.F16, 1, 2, "float16", dequant_supported=True
    ),
    GGMLQuantizationType.Q4_0: _layout(
        GGMLQuantizationType.Q4_0, 32, 2 + 16, "uint8_blocks", dequant_supported=True
    ),
    GGMLQuantizationType.Q4_1: _layout(
        GGMLQuantizationType.Q4_1, 32, 2 + 2 + 16, "uint8_blocks", dequant_supported=True
    ),
    GGMLQuantizationType.Q5_0: _layout(
        GGMLQuantizationType.Q5_0, 32, 2 + 4 + 16, "uint8_blocks", dequant_supported=True
    ),
    GGMLQuantizationType.Q5_1: _layout(
        GGMLQuantizationType.Q5_1, 32, 2 + 2 + 4 + 16, "uint8_blocks", dequant_supported=True
    ),
    GGMLQuantizationType.Q8_0: _layout(
        GGMLQuantizationType.Q8_0, 32, 2 + 32, "uint8_blocks", dequant_supported=True
    ),
    GGMLQuantizationType.Q8_1: _layout(
        GGMLQuantizationType.Q8_1, 32, 4 + 4 + 32, "uint8_blocks"
    ),
    GGMLQuantizationType.Q2_K: _layout(
        GGMLQuantizationType.Q2_K,
        256,
        2 + 2 + QK_K // 16 + QK_K // 4,
        "uint8_blocks",
    ),
    GGMLQuantizationType.Q3_K: _layout(
        GGMLQuantizationType.Q3_K,
        256,
        2 + QK_K // 4 + QK_K // 8 + 12,
        "uint8_blocks",
    ),
    GGMLQuantizationType.Q4_K: _layout(
        GGMLQuantizationType.Q4_K,
        256,
        2 + 2 + QK_K // 2 + 12,
        "uint8_blocks",
        dequant_supported=True,
    ),
    GGMLQuantizationType.Q5_K: _layout(
        GGMLQuantizationType.Q5_K,
        256,
        2 + 2 + QK_K // 2 + QK_K // 8 + 12,
        "uint8_blocks",
        dequant_supported=True,
    ),
    GGMLQuantizationType.Q6_K: _layout(
        GGMLQuantizationType.Q6_K,
        256,
        2 + QK_K // 2 + QK_K // 4 + QK_K // 16,
        "uint8_blocks",
        dequant_supported=True,
    ),
    GGMLQuantizationType.Q8_K: _layout(
        GGMLQuantizationType.Q8_K, 256, 4 + QK_K + QK_K // 8, "uint8_blocks"
    ),
    GGMLQuantizationType.IQ2_XXS: _layout(
        GGMLQuantizationType.IQ2_XXS, 256, 2 + QK_K // 4, "uint8_blocks"
    ),
    GGMLQuantizationType.IQ2_XS: _layout(
        GGMLQuantizationType.IQ2_XS,
        256,
        2 + QK_K // 4 + QK_K // 32,
        "uint8_blocks",
    ),
    GGMLQuantizationType.IQ3_XXS: _layout(
        GGMLQuantizationType.IQ3_XXS,
        256,
        2 + QK_K // 4 + QK_K // 8,
        "uint8_blocks",
    ),
    GGMLQuantizationType.IQ1_S: _layout(
        GGMLQuantizationType.IQ1_S,
        256,
        2 + QK_K // 8 + QK_K // 16,
        "uint8_blocks",
    ),
    GGMLQuantizationType.IQ4_NL: _layout(
        GGMLQuantizationType.IQ4_NL, 32, 2 + 16, "uint8_blocks", dequant_supported=True
    ),
    GGMLQuantizationType.IQ3_S: _layout(
        GGMLQuantizationType.IQ3_S,
        256,
        2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4,
        "uint8_blocks",
    ),
    GGMLQuantizationType.IQ2_S: _layout(
        GGMLQuantizationType.IQ2_S,
        256,
        2 + QK_K // 4 + QK_K // 16,
        "uint8_blocks",
    ),
    GGMLQuantizationType.IQ4_XS: _layout(
        GGMLQuantizationType.IQ4_XS,
        256,
        2 + 2 + QK_K // 2 + QK_K // 64,
        "uint8_blocks",
        dequant_supported=True,
    ),
    GGMLQuantizationType.I8: _layout(
        GGMLQuantizationType.I8, 1, 1, "int8", dequant_supported=True
    ),
    GGMLQuantizationType.I16: _layout(
        GGMLQuantizationType.I16, 1, 2, "int16", dequant_supported=True
    ),
    GGMLQuantizationType.I32: _layout(
        GGMLQuantizationType.I32, 1, 4, "int32", dequant_supported=True
    ),
    GGMLQuantizationType.I64: _layout(
        GGMLQuantizationType.I64, 1, 8, "int64", dequant_supported=True
    ),
    GGMLQuantizationType.F64: _layout(
        GGMLQuantizationType.F64, 1, 8, "float64", dequant_supported=True
    ),
    GGMLQuantizationType.IQ1_M: _layout(
        GGMLQuantizationType.IQ1_M,
        256,
        QK_K // 8 + QK_K // 16 + QK_K // 32,
        "uint8_blocks",
    ),
    GGMLQuantizationType.BF16: _layout(
        GGMLQuantizationType.BF16, 1, 2, "bf16", dequant_supported=True
    ),
    GGMLQuantizationType.TQ1_0: _layout(
        GGMLQuantizationType.TQ1_0, 256, 2 + 4 * 13, "uint8_blocks"
    ),
    GGMLQuantizationType.TQ2_0: _layout(
        GGMLQuantizationType.TQ2_0, 256, 2 + 64, "uint8_blocks"
    ),
    GGMLQuantizationType.MXFP4: _layout(
        GGMLQuantizationType.MXFP4, 32, 1 + 16, "uint8_blocks", dequant_supported=True
    ),
    GGMLQuantizationType.NVFP4: _layout(
        GGMLQuantizationType.NVFP4, 64, 4 + 32, "uint8_blocks"
    ),
    GGMLQuantizationType.Q1_0: _layout(
        GGMLQuantizationType.Q1_0, 128, 2 + 16, "uint8_blocks"
    ),
}

_NUMPY_STORAGE_DTYPES = {
    "float32": np.float32,
    "float16": np.float16,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "bf16": np.uint16,
    "uint8_blocks": np.uint8,
}

_IQ4_NL_KVALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
_MXFP4_KVALUES = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def ggml_type(type_id: int | GGMLQuantizationType) -> GGMLQuantizationType:
    try:
        if isinstance(type_id, GGMLQuantizationType):
            return type_id
        return GGMLQuantizationType(int(type_id))
    except ValueError as exc:
        raise KeyError(f"unknown GGML quantization type id {int(type_id)!r}") from exc


def ggml_type_name(type_id: int | GGMLQuantizationType) -> str:
    try:
        return ggml_type(type_id).name
    except KeyError:
        return f"UNKNOWN_{int(type_id)}"


def llama_file_type_name(file_type: object) -> str | None:
    if file_type is None:
        return None
    try:
        return LlamaFileType(int(file_type)).name
    except (TypeError, ValueError):
        return None


def quant_layout(type_id: int | GGMLQuantizationType) -> GGUFQuantLayout:
    qtype = ggml_type(type_id)
    try:
        return GGUF_QUANT_LAYOUTS[qtype]
    except KeyError as exc:
        raise KeyError(f"missing GGUF layout for GGML type {qtype.name}") from exc


def numpy_storage_dtype(type_id: int | GGMLQuantizationType) -> type[np.generic]:
    layout = quant_layout(type_id)
    return _NUMPY_STORAGE_DTYPES[layout.storage_dtype]


def nbytes_for_shape(shape: Sequence[int], type_id: int | GGMLQuantizationType) -> int:
    layout = quant_layout(type_id)
    elements = int(prod(int(dim) for dim in shape))
    if elements % layout.block_size != 0:
        raise ValueError(
            f"shape {tuple(shape)} has {elements} elements, not a multiple of "
            f"{layout.name} block size {layout.block_size}"
        )
    return elements // layout.block_size * layout.type_size


def quant_shape_to_byte_shape(
    shape: Sequence[int], type_id: int | GGMLQuantizationType
) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in shape)
    layout = quant_layout(type_id)
    if layout.storage_dtype != "uint8_blocks":
        return shape
    if not shape:
        raise ValueError(f"block-quantized {layout.name} tensor must have at least one dimension")
    if shape[-1] % layout.block_size != 0:
        raise ValueError(
            f"quantized tensor row size {shape[-1]} is not a multiple of "
            f"{layout.name} block size {layout.block_size}"
        )
    return (*shape[:-1], shape[-1] // layout.block_size * layout.type_size)


def quant_shape_from_byte_shape(
    shape: Sequence[int], type_id: int | GGMLQuantizationType
) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in shape)
    layout = quant_layout(type_id)
    if layout.storage_dtype != "uint8_blocks":
        return shape
    if not shape:
        raise ValueError(
            f"block-quantized {layout.name} byte tensor must have at least one dimension"
        )
    if shape[-1] % layout.type_size != 0:
        raise ValueError(
            f"quantized tensor bytes per row {shape[-1]} is not a multiple of "
            f"{layout.name} type size {layout.type_size}"
        )
    return (*shape[:-1], shape[-1] // layout.type_size * layout.block_size)


def dequantization_supported(type_id: int | GGMLQuantizationType) -> bool:
    return quant_layout(type_id).dequant_supported


def bf16_to_float32(array: object) -> np.ndarray:
    bits = np.asarray(array, dtype=np.uint16)
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def dequantize_gguf_data(data: object, type_id: int | GGMLQuantizationType) -> np.ndarray:
    """Dequantize a GGUF tensor or row slice to float32.

    For block quantized tensors ``data`` must use the GGUF byte shape, i.e. the
    final dimension is bytes per row rather than logical values per row.  This
    is exactly the shape returned by :meth:`hipengine.loading.gguf.GGUFReader.tensor_data`.
    """

    qtype = ggml_type(type_id)
    if qtype == GGMLQuantizationType.F32:
        return np.asarray(data, dtype=np.float32)
    if qtype == GGMLQuantizationType.F16:
        return np.asarray(data, dtype=np.float16).astype(np.float32)
    if qtype == GGMLQuantizationType.F64:
        return np.asarray(data, dtype=np.float64).astype(np.float32)
    if qtype == GGMLQuantizationType.BF16:
        return bf16_to_float32(data)
    if qtype in {
        GGMLQuantizationType.I8,
        GGMLQuantizationType.I16,
        GGMLQuantizationType.I32,
        GGMLQuantizationType.I64,
    }:
        return np.asarray(data).astype(np.float32)

    fn = _DEQUANT_BLOCKS.get(qtype)
    if fn is None:
        raise NotImplementedError(
            f"dequantization for GGUF tensor type {qtype.name} is not implemented"
        )
    return _dequantize_block_rows(np.asarray(data).view(np.uint8), qtype, fn)


def _dequantize_block_rows(
    rows: np.ndarray,
    qtype: GGMLQuantizationType,
    dequantize_blocks: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    layout = quant_layout(qtype)
    rows = rows.view(np.uint8)
    byte_shape = rows.shape
    if rows.size % layout.type_size != 0:
        raise ValueError(
            f"{qtype.name} byte size {rows.size} is not a multiple of block size "
            f"{layout.type_size}"
        )
    blocks = rows.reshape((rows.size // layout.type_size, layout.type_size))
    out = dequantize_blocks(blocks)
    if out.dtype != np.float32:
        out = out.astype(np.float32)
    return out.reshape(quant_shape_from_byte_shape(byte_shape, qtype))


def _split(blocks: np.ndarray, indices: list[int] | tuple[int, ...]) -> list[np.ndarray]:
    return list(np.hsplit(blocks, indices))


def _dequant_q4_0_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d, qs = _split(blocks, [2])
    d = d.view(np.float16).astype(np.float32)
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.int8) - np.int8(8)
    return d * qs.astype(np.float32)


def _dequant_q4_1_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d, rest = _split(blocks, [2])
    m, qs = _split(rest, [2])
    d = d.view(np.float16).astype(np.float32)
    m = m.view(np.float16).astype(np.float32)
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.float32)
    return d * qs + m


def _dequant_q5_0_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d, rest = _split(blocks, [2])
    qh, qs = _split(rest, [4])
    d = d.view(np.float16).astype(np.float32)
    qh = qh.view(np.uint32).reshape((n_blocks, 1))
    qh = qh >> np.arange(32, dtype=np.uint32).reshape((1, 32))
    ql = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qh = (qh & np.uint32(0x01)).astype(np.uint8)
    ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1))
    qs = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(16)
    return d * qs.astype(np.float32)


def _dequant_q5_1_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d, rest = _split(blocks, [2])
    m, rest = _split(rest, [2])
    qh, qs = _split(rest, [4])
    d = d.view(np.float16).astype(np.float32)
    m = m.view(np.float16).astype(np.float32)
    qh = qh.view(np.uint32).reshape((n_blocks, 1))
    qh = qh >> np.arange(32, dtype=np.uint32).reshape((1, 32))
    ql = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qh = (qh & np.uint32(0x01)).astype(np.uint8)
    ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1))
    qs = (ql | (qh << np.uint8(4))).astype(np.float32)
    return d * qs + m


def _dequant_q8_0_blocks(blocks: np.ndarray) -> np.ndarray:
    d, x = np.split(blocks, [2], axis=1)
    d = d.view(np.float16).astype(np.float32)
    x = x.view(np.int8).astype(np.float32)
    return x * d


def unpack_q4_k_scale_min(scales: object) -> tuple[np.ndarray, np.ndarray]:
    """Unpack GGUF Q4_K 12-byte scale/min fields to uint8 arrays.

    Returns ``(scales, mins)`` with shape ``[blocks, 8]``.  The eight columns
    correspond to the 32-value subblocks inside each 256-value GGUF Q4_K block.
    """

    scales = np.asarray(scales, dtype=np.uint8)
    n_blocks = scales.shape[0]
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = np.split(scales, 3, axis=-2)
    sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    minv = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc.reshape((n_blocks, 8)), minv.reshape((n_blocks, 8))


def _q4_k_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return unpack_q4_k_scale_min(scales)


def _dequant_q4_k_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d, rest = _split(blocks, [2])
    dmin, rest = _split(rest, [2])
    scales, qs = _split(rest, [12])
    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)
    sc, m = _q4_k_scale_min(scales)
    d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
    dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 32)).astype(np.float32)
    return (d * qs - dm).reshape((n_blocks, QK_K))


def _dequant_q5_k_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d, rest = _split(blocks, [2])
    dmin, rest = _split(rest, [2])
    scales, rest = _split(rest, [12])
    qh, qs = _split(rest, [QK_K // 8])
    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)
    sc, m = _q4_k_scale_min(scales)
    d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
    dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.arange(8, dtype=np.uint8).reshape((1, 1, 8, 1))
    ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
    qh = (qh & np.uint8(0x01)).reshape((n_blocks, -1, 32))
    q = (ql | (qh << np.uint8(4))).astype(np.float32)
    return (d * q - dm).reshape((n_blocks, QK_K))


def _dequant_q6_k_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    ql, rest = _split(blocks, [QK_K // 2])
    qh, rest = _split(rest, [QK_K // 4])
    scales, d = _split(rest, [QK_K // 16])
    scales = scales.view(np.int8).astype(np.float32)
    d = d.view(np.float16).astype(np.float32)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array(
        [0, 2, 4, 6], dtype=np.uint8
    ).reshape((1, 1, 4, 1))
    qh = (qh & np.uint8(0x03)).reshape((n_blocks, -1, 32))
    q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(32)
    q = q.reshape((n_blocks, QK_K // 16, -1)).astype(np.float32)
    return (d * q).reshape((n_blocks, QK_K))


def _dequant_iq4_nl_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d, qs = _split(blocks, [2])
    d = d.view(np.float16).astype(np.float32)
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 1))
    kvalues = np.array(_IQ4_NL_KVALUES, dtype=np.int8).reshape(1, 1, 16)
    qs = np.take_along_axis(kvalues, qs, axis=-1).astype(np.float32).reshape((n_blocks, -1))
    return d * qs


def _dequant_iq4_xs_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d, rest = _split(blocks, [2])
    scales_h, rest = _split(rest, [2])
    scales_l, qs = _split(rest, [QK_K // 64])
    d = d.view(np.float16).astype(np.float32)
    scales_h = scales_h.view(np.uint16)
    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> np.array(
        [0, 4], dtype=np.uint8
    ).reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, 1, -1)) >> np.array(
        [2 * i for i in range(QK_K // 32)], dtype=np.uint16
    ).reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1)) & np.uint8(0x0F)
    scales_h = scales_h.reshape((n_blocks, -1)).astype(np.uint8) & np.uint8(0x03)
    scales = (scales_l | (scales_h << np.uint8(4))).astype(np.int8) - np.int8(32)
    dl = (d * scales.astype(np.float32)).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = qs.reshape((n_blocks, -1, 32, 1)) & np.uint8(0x0F)
    kvalues = np.array(_IQ4_NL_KVALUES, dtype=np.int8).reshape((1, 1, 1, -1))
    qs = np.take_along_axis(kvalues, qs, axis=-1).astype(np.float32).reshape((n_blocks, -1, 32))
    return (dl * qs).reshape((n_blocks, -1))


def _mxfp4_e8m0_to_fp32_half(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint32)
    bits = np.where(x < 2, np.uint32(0x00200000) << x, (x - np.uint32(1)) << np.uint32(23))
    return bits.view(np.float32)


def _dequant_mxfp4_blocks(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    e, qs = _split(blocks, [1])
    d = _mxfp4_e8m0_to_fp32_half(e)
    qs = qs.reshape((n_blocks, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 2, 1))
    qs = (qs & np.uint8(0x0F)).view(np.int8)
    kvalues = np.array(_MXFP4_KVALUES, dtype=np.int8).reshape(1, 1, 16)
    qs = np.take_along_axis(kvalues, qs, axis=-1).reshape((n_blocks, 32))
    return d * qs.astype(np.float32)


_DEQUANT_BLOCKS: dict[GGMLQuantizationType, Callable[[np.ndarray], np.ndarray]] = {
    GGMLQuantizationType.Q4_0: _dequant_q4_0_blocks,
    GGMLQuantizationType.Q4_1: _dequant_q4_1_blocks,
    GGMLQuantizationType.Q5_0: _dequant_q5_0_blocks,
    GGMLQuantizationType.Q5_1: _dequant_q5_1_blocks,
    GGMLQuantizationType.Q8_0: _dequant_q8_0_blocks,
    GGMLQuantizationType.Q4_K: _dequant_q4_k_blocks,
    GGMLQuantizationType.Q5_K: _dequant_q5_k_blocks,
    GGMLQuantizationType.Q6_K: _dequant_q6_k_blocks,
    GGMLQuantizationType.IQ4_NL: _dequant_iq4_nl_blocks,
    GGMLQuantizationType.IQ4_XS: _dequant_iq4_xs_blocks,
    GGMLQuantizationType.MXFP4: _dequant_mxfp4_blocks,
}


__all__ = [
    "GGMLQuantizationType",
    "GGUFQuantLayout",
    "GGUFValueType",
    "GGUF_QUANT_LAYOUTS",
    "LlamaFileType",
    "QK_K",
    "bf16_to_float32",
    "dequantization_supported",
    "dequantize_gguf_data",
    "ggml_type",
    "ggml_type_name",
    "llama_file_type_name",
    "nbytes_for_shape",
    "numpy_storage_dtype",
    "quant_layout",
    "quant_shape_from_byte_shape",
    "quant_shape_to_byte_shape",
    "unpack_q4_k_scale_min",
]
