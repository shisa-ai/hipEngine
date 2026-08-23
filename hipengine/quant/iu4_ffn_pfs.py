"""Immutable PromptForge IU4 FFN sidecar product reader.

The PFSIU4F file is a six-entry-per-layer container: combined gate/up S4,
per-output scales/sums, then down S4 with its scales/sums. Device kernels use a
separate paired-K32 tile view; conversion is byte-lossless and does not alter
signed nibbles or metadata.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from hipengine.quant.registry import register_quant

_HEADER = struct.Struct("<8sIIIIQQQQQ")
_ENTRY = struct.Struct("<HHBBHIIQQ32s")
_MAGIC: Final = b"PFSIU4F\0"
_VERSION: Final = 1

_GATE_WEIGHT = 10
_GATE_SCALE = 11
_GATE_SUM = 12
_DOWN_WEIGHT = 13
_DOWN_SCALE = 14
_DOWN_SUM = 15
_DTYPE_F32 = 2
_DTYPE_I32 = 3
_DTYPE_S4 = 4

KAIRIC_QWEN38_FFN_REPOSITORY: Final = "jcbtc/Qwen3.8-27B-IU4-Kairic-Edge"
KAIRIC_QWEN38_FFN_REVISION: Final = "42de3b69d6ed039745e60b6bab0b6fe70061bfcc"
KAIRIC_QWEN38_FFN_FILENAME: Final = "Qwen3.8-27B-Kairic-IU4-FFN.pfs"
KAIRIC_QWEN38_FFN_SHA256: Final = "adcbb90a7b429a30a2a39043366d68320d72e8b4816a0f498e882b2f80a2ba2b"
KAIRIC_QWEN38_FFN_BYTES: Final = 8_576_856_064


class PFSFormatError(ValueError):
    """Raised when an IU4 FFN sidecar fails its immutable format contract."""


@dataclass(frozen=True)
class KairicIU4FFNQuant:
    """Explicit T3 product identity for the published Kairic FFN sidecar."""

    name: str = "iu4_s4_kairic_ffn_v1"
    weight_storage: str = "pfsiu4f_s4_n64_k8word_per_output_scale_sum"
    activation_preprocess: str = "dynamic_u4_block_hadamard1024_per_row"
    compute_dtype: str = "iu4_wmma_i32_accum_bf16_boundary"
    scale_granularity: str = "one_long_k_segment_per_output_and_activation_row"
    calibration_artifact: str = "kairic_edge_qwen38_27b_v1"
    kernel_family: str = "gfx1151_iu4_s4_sidecar"


IU4_S4_KAIRIC_FFN_V1 = register_quant(KairicIU4FFNQuant())


@dataclass(frozen=True)
class IU4FFNSpec:
    layers: int = 64
    hidden: int = 5120
    intermediate: int = 17408

    def __post_init__(self) -> None:
        if self.layers <= 0:
            raise ValueError("IU4 FFN layers must be positive")
        if self.hidden <= 0 or self.hidden % 64:
            raise ValueError("IU4 FFN hidden must be a positive multiple of 64")
        if self.intermediate <= 0 or self.intermediate % 64:
            raise ValueError("IU4 FFN intermediate must be a positive multiple of 64")

    @property
    def entry_count(self) -> int:
        return self.layers * 6


KAIRIC_QWEN38_FFN_SPEC = IU4FFNSpec()


@dataclass(frozen=True)
class _EntryView:
    layer: int
    kind: int
    dtype: int
    rank: int
    rows: int
    cols: int
    offset: int
    length: int


@dataclass(frozen=True)
class IU4FFNLayer:
    gate_weight: np.ndarray
    gate_scales: np.ndarray
    gate_sums: np.ndarray
    down_weight: np.ndarray
    down_scales: np.ndarray
    down_sums: np.ndarray


class IU4FFNSidecar:
    def __init__(
        self,
        path: str | Path,
        *,
        spec: IU4FFNSpec,
        expected_sha256: str | None,
    ) -> None:
        self.path = Path(path).resolve()
        self.spec = spec
        if not self.path.is_file():
            raise PFSFormatError(f"IU4 FFN sidecar does not exist: {self.path}")
        if expected_sha256 is not None:
            actual = _sha256_file(self.path)
            if actual != expected_sha256.lower():
                raise PFSFormatError(
                    f"IU4 FFN sidecar SHA-256 mismatch: expected {expected_sha256}, got {actual}"
                )
        self.sha256 = expected_sha256.lower() if expected_sha256 is not None else None
        self._mapping = np.memmap(self.path, mode="r", dtype=np.uint8)
        try:
            self._entries = _validate_container(self._mapping, spec=spec)
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "IU4FFNSidecar":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            mmap_object = getattr(mapping, "_mmap", None)
            if mmap_object is not None:
                mmap_object.close()
            self._mapping = None

    def layer(self, layer: int) -> IU4FFNLayer:
        if layer < 0 or layer >= self.spec.layers:
            raise IndexError(f"IU4 FFN layer must be in [0, {self.spec.layers - 1}]")
        base = layer * 6
        gate_weight, gate_scale, gate_sum, down_weight, down_scale, down_sum = (
            self._entries[base + index] for index in range(6)
        )
        return IU4FFNLayer(
            gate_weight=self._weight_view(gate_weight),
            gate_scales=self._vector_view(gate_scale, np.float32),
            gate_sums=self._vector_view(gate_sum, np.int32),
            down_weight=self._weight_view(down_weight),
            down_scales=self._vector_view(down_scale, np.float32),
            down_sums=self._vector_view(down_sum, np.int32),
        )

    def _weight_view(self, entry: _EntryView) -> np.ndarray:
        return np.ndarray(
            shape=(entry.rows // 64, entry.cols // 8, 64),
            dtype=np.uint32,
            buffer=self._mapping,
            offset=entry.offset,
        )

    def _vector_view(self, entry: _EntryView, dtype: np.dtype) -> np.ndarray:
        return np.ndarray(
            shape=(entry.rows,),
            dtype=dtype,
            buffer=self._mapping,
            offset=entry.offset,
        )


def open_iu4_ffn_pfs(
    path: str | Path,
    *,
    spec: IU4FFNSpec = KAIRIC_QWEN38_FFN_SPEC,
    expected_sha256: str | None = None,
) -> IU4FFNSidecar:
    return IU4FFNSidecar(path, spec=spec, expected_sha256=expected_sha256)


def open_kairic_qwen38_ffn(
    path: str | Path,
    *,
    verify_sha256: bool = True,
) -> IU4FFNSidecar:
    candidate = Path(path)
    if candidate.stat().st_size != KAIRIC_QWEN38_FFN_BYTES:
        raise PFSFormatError(
            f"Kairic Qwen3.8 FFN byte size mismatch: expected {KAIRIC_QWEN38_FFN_BYTES}, "
            f"got {candidate.stat().st_size}"
        )
    return open_iu4_ffn_pfs(
        candidate,
        spec=KAIRIC_QWEN38_FFN_SPEC,
        expected_sha256=KAIRIC_QWEN38_FFN_SHA256 if verify_sha256 else None,
    )


def pfs_s4_to_n16_k32_tiles(weight_words: object) -> np.ndarray:
    """Losslessly transpose PFS ``[N64,K8word,N]`` into paired-K32 tiles."""

    source = np.asarray(weight_words, dtype=np.uint32)
    if source.ndim != 3 or source.shape[2] != 64:
        raise ValueError("PFS S4 weights must have shape [N/64, K/8, 64]")
    n64, words_per_row, _ = source.shape
    if n64 <= 0 or words_per_row <= 0 or words_per_row % 4:
        raise ValueError("PFS S4 N and K dimensions must be positive and K divisible by 32")
    k_pairs = words_per_row // 4
    words = np.ascontiguousarray(
        source.reshape(n64, k_pairs, 4, 4, 16)
        .transpose(0, 3, 1, 4, 2)
        .reshape(n64 * 4, k_pairs, 16, 4)
    )
    return words.view(np.uint8).reshape(n64 * 4, k_pairs, 16, 16)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_container(mapping: np.ndarray, *, spec: IU4FFNSpec) -> tuple[_EntryView, ...]:
    if mapping.nbytes < _HEADER.size:
        raise PFSFormatError("IU4 FFN sidecar is smaller than its header")
    (
        magic,
        version,
        header_bytes,
        entry_bytes,
        entry_count,
        table_offset,
        table_bytes,
        data_offset,
        file_bytes,
        reserved,
    ) = _HEADER.unpack_from(mapping, 0)
    actual_bytes = int(mapping.nbytes)
    expected_table_bytes = spec.entry_count * _ENTRY.size
    if magic != _MAGIC or version != _VERSION:
        raise PFSFormatError("IU4 FFN sidecar magic/version mismatch")
    if header_bytes != _HEADER.size or entry_bytes != _ENTRY.size:
        raise PFSFormatError("IU4 FFN sidecar header/entry size mismatch")
    if entry_count != spec.entry_count or table_offset != _HEADER.size:
        raise PFSFormatError("IU4 FFN sidecar entry-count/table-offset mismatch")
    if table_bytes != expected_table_bytes:
        raise PFSFormatError("IU4 FFN sidecar table byte count mismatch")
    if data_offset < table_offset + table_bytes or data_offset % 4096:
        raise PFSFormatError("IU4 FFN sidecar data offset is invalid")
    if file_bytes != actual_bytes or reserved != 0:
        raise PFSFormatError("IU4 FFN sidecar file byte count/reserved field mismatch")

    entries: list[_EntryView] = []
    next_offset = int(data_offset)
    for layer in range(spec.layers):
        expected = _expected_layer_entries(spec, layer)
        for local_index, expected_fields in enumerate(expected):
            table_index = layer * 6 + local_index
            raw = _ENTRY.unpack_from(mapping, int(table_offset) + table_index * _ENTRY.size)
            entry = _EntryView(
                layer=int(raw[0]),
                kind=int(raw[1]),
                dtype=int(raw[2]),
                rank=int(raw[3]),
                rows=int(raw[5]),
                cols=int(raw[6]),
                offset=int(raw[7]),
                length=int(raw[8]),
            )
            if any(raw[9]) or int(raw[4]) != 0:
                raise PFSFormatError("IU4 FFN sidecar entry reserved fields must be zero")
            if (
                entry.layer,
                entry.kind,
                entry.dtype,
                entry.rank,
                entry.rows,
                entry.cols,
                entry.length,
            ) != expected_fields:
                raise PFSFormatError(
                    f"IU4 FFN sidecar entry contract mismatch at layer {layer}, index {local_index}"
                )
            if entry.offset != next_offset:
                raise PFSFormatError(
                    f"IU4 FFN sidecar entries must be contiguous: expected {next_offset}, "
                    f"got {entry.offset}"
                )
            if entry.offset > actual_bytes or entry.length > actual_bytes - entry.offset:
                raise PFSFormatError("IU4 FFN sidecar entry exceeds file bounds")
            entries.append(entry)
            next_offset += entry.length
    if next_offset != actual_bytes:
        raise PFSFormatError(
            f"IU4 FFN sidecar trailing/missing bytes: entries end at {next_offset}, file at {actual_bytes}"
        )
    return tuple(entries)


def _expected_layer_entries(spec: IU4FFNSpec, layer: int) -> tuple[tuple[int, ...], ...]:
    gate_rows = 2 * spec.intermediate
    down_rows = spec.hidden
    return (
        (layer, _GATE_WEIGHT, _DTYPE_S4, 2, gate_rows, spec.hidden, gate_rows * spec.hidden // 2),
        (layer, _GATE_SCALE, _DTYPE_F32, 1, gate_rows, 1, gate_rows * 4),
        (layer, _GATE_SUM, _DTYPE_I32, 1, gate_rows, 1, gate_rows * 4),
        (layer, _DOWN_WEIGHT, _DTYPE_S4, 2, down_rows, spec.intermediate, down_rows * spec.intermediate // 2),
        (layer, _DOWN_SCALE, _DTYPE_F32, 1, down_rows, 1, down_rows * 4),
        (layer, _DOWN_SUM, _DTYPE_I32, 1, down_rows, 1, down_rows * 4),
    )


__all__ = [
    "IU4FFNLayer",
    "IU4FFNSidecar",
    "IU4FFNSpec",
    "IU4_S4_KAIRIC_FFN_V1",
    "KAIRIC_QWEN38_FFN_BYTES",
    "KAIRIC_QWEN38_FFN_FILENAME",
    "KAIRIC_QWEN38_FFN_REPOSITORY",
    "KAIRIC_QWEN38_FFN_REVISION",
    "KAIRIC_QWEN38_FFN_SHA256",
    "KAIRIC_QWEN38_FFN_SPEC",
    "KairicIU4FFNQuant",
    "PFSFormatError",
    "open_iu4_ffn_pfs",
    "open_kairic_qwen38_ffn",
    "pfs_s4_to_n16_k32_tiles",
]
