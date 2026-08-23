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
    # ciru-ai/ROCmFPX@e1da26bb custom GGUF tensor IDs used by the immutable
    # Qwen3.8 Kairic Edge release. Keep the numeric IDs aligned with its GGUF.
    Q4_0_ROCMFP4 = 100
    Q6_0_ROCMFPX = 102


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
    MOSTLY_Q4_0_ROCMFP4 = 100
    MOSTLY_Q4_0_ROCMFP4_LEAN = 101
    MOSTLY_Q4_0_ROCMFP4_COHERENT = 102
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
        dequant_supported=True,
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
        dequant_supported=True,
    ),
    GGMLQuantizationType.IQ3_XXS: _layout(
        GGMLQuantizationType.IQ3_XXS,
        256,
        2 + QK_K // 4 + QK_K // 8,
        "uint8_blocks",
        dequant_supported=True,
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
    GGMLQuantizationType.Q4_0_ROCMFP4: _layout(
        GGMLQuantizationType.Q4_0_ROCMFP4,
        32,
        16 + 2,
        "uint8_blocks",
        dequant_supported=True,
        native_status="lossless_dense_bf16_authority_fallback",
    ),
    GGMLQuantizationType.Q6_0_ROCMFPX: _layout(
        GGMLQuantizationType.Q6_0_ROCMFPX,
        32,
        24 + 2,
        "uint8_blocks",
        dequant_supported=True,
        native_status="lossless_dense_f32_authority_fallback",
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
_ROCMFP4_KVALUES = (0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10)

# IQ2_XS codebook from llama.cpp@1ebf790cd gguf-py/gguf/quants.py. Each
# little-endian u16 packs eight 2-bit selectors mapped through (8, 25, 43).
_IQ2_XS_GRID_PACKED = np.frombuffer(
    bytes.fromhex(
        "00000200050008000a0011001400160019002000220025002800410044004600"
        "49005000520055005800610064008000820085008800910094009900a0000101"
        "04010601090110011201150118011a0121012401400142014501480151015401"
        "6001680181018401900100020202050208021102140220024102440250025502"
        "80028a0201040404060409041004120415041804210424044004420445044804"
        "5104540456046004810484049004000502050505080511051405200541054405"
        "500561058005010604061006260640064206840600080208050808080a081108"
        "14082008250841084408500858088008a008aa08010904091009400981098909"
        "000a200a280a960aa00a01100410061009101010121015101810211024104010"
        "4210451048105110541060106a10811084109010001102110511081111111411"
        "2011411144115011801194119611011204120612101240126012001402140514"
        "0814111414142014411444144914501464148014011504151015401500161416"
        "49160118041810181218401854188618001905196619511aa91a002002200520"
        "08200a201120142020204120442050208020a020012104211021402148216521"
        "002222228022a82201240424102429244024002541255225992501261a26a626"
        "002808280a28202855288828a22868299029082a202a822a882a8a2a01400440"
        "0640094010401240154018402140244040404240454048404a40514054406040"
        "6540814084409040004102410541084111411441204141414441504180418541"
        "a241014204421042124229424042004402440544084411441444194420444144"
        "4444504480449444014504451045244540459a4500460a464446504601480448"
        "1048404845485448624800491149444950496949044a00500250055008501150"
        "145020502850415044505050805001510451105115514051425100524452aa52"
        "0154045410542154405460548154a154005508558055885521566856a1560058"
        "14584158505899581a5940594259855a0160046010604060546062608660a960"
        "006124624a62926200641664106540654565a46501686a682569066a546a626a"
        "00800280058008801180148020802a8041804480508080808280a880aa800181"
        "0481068110814081518159810082208280828282a082a8820184048410841284"
        "158440846084898400854485a58518866a860088088825885a8880888288a888"
        "0689228a808a888a968aa88a0190049010904090569084900091229164915692"
        "89920094059444945094589429959095929541965198a6984999159a609a00a0"
        "02a008a00aa020a02aa0a0a051a159a1a6a100a202a208a22aa280a2a0a240a4"
        "95a465a698a60aa820a822a828a8a0a8a8a804a984a986a928aa2aaa91aaaaaa"
    ),
    dtype="<u2",
).copy()
_IQ2_XS_GRID_MAGNITUDES = np.array([8, 25, 43, 0], dtype=np.uint8)

# IQ3_XXS codebook from llama.cpp@1ebf790cd ggml-common.h (GGML_TABLE iq3xxs_grid).
# Each u32 packs four unsigned grid magnitudes (little-endian byte order).
_IQ3_XXS_GRID = np.array(
    [
        0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
        0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
        0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
        0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
        0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
        0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
        0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
        0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
        0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
        0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
        0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
        0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
        0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
        0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
        0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
        0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
        0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
        0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
        0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
        0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
        0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
        0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
        0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
        0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
        0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
        0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
        0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
        0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
        0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
        0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
        0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
        0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
    ],
    dtype=np.uint32,
)

# ksigns_iq2xs[i] = i | (parity(i) << 7): computed, matching llama.cpp.
_KSIGNS_IQ2XS = np.array(
    [i | ((bin(i).count("1") & 1) << 7) for i in range(128)], dtype=np.uint8
)
_IQ3_XXS_GRID_BYTES = _IQ3_XXS_GRID.view(np.uint8).reshape(256, 4)


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


def _dequant_iq2_xs_blocks(blocks: np.ndarray) -> np.ndarray:
    """Dequantize IQ2_XS blocks (74 bytes per 256 values).

    Mirrors llama.cpp ``dequantize_row_iq2_xs``. Each 32-value group has four
    grid/sign words and two 4-bit scales; one fp16 super-scale covers the full
    256-value block.
    """

    n_blocks = blocks.shape[0]
    d, rest = _split(blocks, [2])
    qs_raw, scales = _split(rest, [2 * QK_K // 8])
    d = d.view(np.float16).astype(np.float32).reshape(n_blocks, 1, 1)
    qs = np.ascontiguousarray(qs_raw).view("<u2").reshape(
        n_blocks, QK_K // 32, 4
    )

    scale_nibbles = (
        scales[:, :, None]
        >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2)
    ) & np.uint8(0x0F)
    db = d * (np.float32(0.5) + scale_nibbles.astype(np.float32)) * np.float32(0.25)
    db = np.repeat(db[:, :, :, None], 2, axis=2)

    grid_idx = qs & np.uint16(511)
    packed = _IQ2_XS_GRID_PACKED[grid_idx]
    selectors = (
        packed[:, :, :, None]
        >> (2 * np.arange(8, dtype=np.uint16)).reshape(1, 1, 1, 8)
    ) & np.uint16(3)
    grid = _IQ2_XS_GRID_MAGNITUDES[selectors].astype(np.float32)

    sign_bytes = _KSIGNS_IQ2XS[qs >> np.uint16(9)]
    sign_bits = (
        sign_bytes[:, :, :, None]
        >> np.arange(8, dtype=np.uint8).reshape(1, 1, 1, 8)
    ) & np.uint8(1)
    signs = np.float32(1.0) - np.float32(2.0) * sign_bits.astype(np.float32)
    return (db * grid * signs).reshape(n_blocks, QK_K)


def _dequant_iq3_xxs_blocks(blocks: np.ndarray) -> np.ndarray:
    """Dequantize IQ3_XXS blocks (98 bytes per 256 values).

    Mirrors llama.cpp ``dequantize_row_iq3_xxs``: per 32-value group, an aux
    u32 supplies four 7-bit sign selectors (via ``ksigns_iq2xs``) and a 4-bit
    sub-scale; each 8-value sub-group reads two codebook grids and applies
    the sign bits. All float ops follow the C reference order bit-exactly.
    """

    n_blocks = blocks.shape[0]
    d, qs = _split(blocks, [2])
    d = d.view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    grid_idx = qs[:, : QK_K // 4].reshape(n_blocks, QK_K // 32, 4, 2)
    aux = np.ascontiguousarray(qs[:, QK_K // 4 :]).view(np.uint32).reshape(n_blocks, QK_K // 32)
    db = (d * (0.5 + (aux >> np.uint32(28)).astype(np.float32)) * 0.5).reshape(n_blocks, QK_K // 32, 1, 1)
    sel = (aux[:, :, None] >> (7 * np.arange(4, dtype=np.uint32))) & np.uint32(127)
    sign_bytes = _KSIGNS_IQ2XS[sel]
    bits = ((sign_bytes[:, :, :, None] >> np.arange(8, dtype=np.uint8)) & np.uint8(1)).astype(np.float32)
    sgn = 1.0 - 2.0 * bits
    grid_vals = _IQ3_XXS_GRID_BYTES[grid_idx].reshape(n_blocks, QK_K // 32, 4, 8).astype(np.float32)
    return (db * grid_vals * sgn).reshape(n_blocks, QK_K)


def _dequant_q3_k_blocks(blocks: np.ndarray) -> np.ndarray:
    """Dequantize Q3_K blocks (110 bytes per 256 values).

    Field order follows ``block_q3_K``: 32-byte hmask, 64 bytes of packed
    2-bit quants, 12 bytes of packed 6-bit scales, then the fp16 super scale.
    Mirrors llama.cpp ``dequantize_row_q3_K`` bit-exactly.
    """

    n_blocks = blocks.shape[0]
    hm, rest = _split(blocks, [QK_K // 8])
    q, rest = _split(rest, [QK_K // 4])
    scales_raw, d = _split(rest, [12])
    d = d.view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    s = np.ascontiguousarray(scales_raw).view(np.uint32).reshape(n_blocks, 3)
    kmask1 = np.uint32(0x03030303)
    kmask2 = np.uint32(0x0F0F0F0F)
    tmp = s[:, 2]
    aux0 = (s[:, 0] & kmask2) | ((tmp & kmask1) << np.uint32(4))
    aux1 = (s[:, 1] & kmask2) | (((tmp >> np.uint32(2)) & kmask1) << np.uint32(4))
    aux2 = ((s[:, 0] >> np.uint32(4)) & kmask2) | (((tmp >> np.uint32(4)) & kmask1) << np.uint32(4))
    aux3 = ((s[:, 1] >> np.uint32(4)) & kmask2) | (((tmp >> np.uint32(6)) & kmask1) << np.uint32(4))
    scales = (
        np.stack([aux0, aux1, aux2, aux3], axis=1)
        .astype(np.uint32)
        .view(np.int8)
        .reshape(n_blocks, 16)
        .astype(np.float32)
    )
    dl = d * (scales - np.float32(32.0))

    out = np.empty((n_blocks, QK_K), dtype=np.float32)
    for h in range(2):
        q_half = q[:, h * 32 : (h + 1) * 32]
        for j in range(4):
            shift = np.uint8(2 * j)
            mask_bit = np.uint8(1 << (h * 4 + j))
            for seg in range(2):
                vals = ((q_half[:, seg * 16 : (seg + 1) * 16] >> shift) & np.uint8(0x03)).astype(np.float32)
                hbit = (hm[:, seg * 16 : (seg + 1) * 16] & mask_bit) != 0
                qv = vals - np.where(hbit, np.float32(0.0), np.float32(4.0))
                col0 = h * 128 + j * 32 + seg * 16
                out[:, col0 : col0 + 16] = dl[:, h * 8 + j * 2 + seg][:, None] * qv
    return out


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


def _rocmfpx_ue4m3_to_fp32(scales: np.ndarray) -> np.ndarray:
    """Decode finite unsigned E4M3 scale bytes from ROCmFPX.

    IDs above ``0x7e`` are invalid in the source format and decode to zero,
    matching ciru-ai/ROCmFPX@e1da26bb ``rocmfpx_ue4m3_to_fp32``.
    """

    encoded = np.asarray(scales, dtype=np.uint8)
    exponent = encoded >> np.uint8(3)
    mantissa = encoded & np.uint8(7)
    subnormal = mantissa.astype(np.float32) * np.float32(2.0**-10)
    normal = np.ldexp(
        (np.uint8(8) + mantissa).astype(np.float32),
        exponent.astype(np.int16) - np.int16(11),
    ).astype(np.float32)
    decoded = np.where(exponent == 0, subnormal, normal).astype(np.float32)
    return np.where(encoded <= np.uint8(0x7E), decoded, np.float32(0.0))


def _dequant_rocmfp4_blocks(blocks: np.ndarray) -> np.ndarray:
    packed, scale_bytes = _split(blocks, [16])
    low_codes = packed & np.uint8(0x0F)
    high_codes = packed >> np.uint8(4)
    codebook = np.asarray(_ROCMFP4_KVALUES, dtype=np.int8)
    low = codebook[low_codes].astype(np.float32)
    high = codebook[high_codes].astype(np.float32)
    scales = _rocmfpx_ue4m3_to_fp32(scale_bytes)
    return np.concatenate(
        [low * scales[:, 0:1], high * scales[:, 1:2]], axis=1
    ).astype(np.float32)


def _dequant_rocmfpx_fp6_blocks(blocks: np.ndarray) -> np.ndarray:
    packed, scale_bytes = _split(blocks, [24])
    groups = packed.reshape(blocks.shape[0], 8, 3)
    codes = np.empty((blocks.shape[0], 8, 4), dtype=np.uint8)
    codes[:, :, 0] = groups[:, :, 0] & np.uint8(0x3F)
    codes[:, :, 1] = (
        ((groups[:, :, 0] >> np.uint8(6)) & np.uint8(0x03))
        | ((groups[:, :, 1] & np.uint8(0x0F)) << np.uint8(2))
    )
    codes[:, :, 2] = (
        ((groups[:, :, 1] >> np.uint8(4)) & np.uint8(0x0F))
        | ((groups[:, :, 2] & np.uint8(0x03)) << np.uint8(4))
    )
    codes[:, :, 3] = (groups[:, :, 2] >> np.uint8(2)) & np.uint8(0x3F)
    codes = codes.reshape(blocks.shape[0], 32)

    magnitudes = (codes & np.uint8(31)).astype(np.int16)
    negative = (codes & np.uint8(32)) != 0
    signed = np.where(
        negative,
        -np.where(magnitudes == 0, np.int16(32), magnitudes),
        magnitudes,
    ).astype(np.float32)
    scales = _rocmfpx_ue4m3_to_fp32(scale_bytes)
    signed[:, :16] *= scales[:, 0:1]
    signed[:, 16:] *= scales[:, 1:2]
    return signed


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
    GGMLQuantizationType.Q3_K: _dequant_q3_k_blocks,
    GGMLQuantizationType.Q4_K: _dequant_q4_k_blocks,
    GGMLQuantizationType.Q5_K: _dequant_q5_k_blocks,
    GGMLQuantizationType.Q6_K: _dequant_q6_k_blocks,
    GGMLQuantizationType.IQ2_XS: _dequant_iq2_xs_blocks,
    GGMLQuantizationType.IQ3_XXS: _dequant_iq3_xxs_blocks,
    GGMLQuantizationType.IQ4_NL: _dequant_iq4_nl_blocks,
    GGMLQuantizationType.IQ4_XS: _dequant_iq4_xs_blocks,
    GGMLQuantizationType.MXFP4: _dequant_mxfp4_blocks,
    GGMLQuantizationType.Q4_0_ROCMFP4: _dequant_rocmfp4_blocks,
    GGMLQuantizationType.Q6_0_ROCMFPX: _dequant_rocmfpx_fp6_blocks,
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
