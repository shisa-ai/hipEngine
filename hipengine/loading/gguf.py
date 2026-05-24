"""Torch-free GGUF scanner and lazy tensor reader."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any, Iterable, Mapping

from hipengine.loading.hf_cache import resolve_model_path
from hipengine.quant.gguf import (
    GGMLQuantizationType,
    GGUFValueType,
    dequantize_gguf_data,
    ggml_type,
    ggml_type_name,
    llama_file_type_name,
    nbytes_for_shape,
    numpy_storage_dtype,
    quant_shape_to_byte_shape,
)

GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32
GGUF_SUPPORTED_VERSIONS = (2, 3)


class GGUFFormatError(ValueError):
    pass


class MissingGGUFTensorError(KeyError):
    pass


@dataclass(frozen=True)
class GGUFTensorInfo:
    """Metadata for one tensor in a GGUF file.

    ``shape`` is the NumPy/row-major logical shape.  ``ggml_shape`` preserves
    the dimension order stored in the GGUF tensor-info table.
    """

    name: str
    shape: tuple[int, ...]
    ggml_shape: tuple[int, ...]
    ggml_type: int
    ggml_type_name: str
    n_elements: int
    nbytes: int
    offset: int
    data_offset: int
    byte_shape: tuple[int, ...]


@dataclass(frozen=True)
class GGUFModelInfo:
    """Resolved GGUF header, metadata, and tensor table."""

    path: Path
    version: int
    alignment: int
    metadata: Mapping[str, Any]
    tensors: tuple[GGUFTensorInfo, ...]
    tensor_data_offset: int

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    @property
    def total_tensor_nbytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.tensors)

    @property
    def architecture(self) -> str | None:
        value = self.metadata.get("general.architecture")
        return str(value) if value is not None else None

    @property
    def file_type(self) -> int | None:
        value = self.metadata.get("general.file_type")
        return int(value) if value is not None else None

    @property
    def file_type_name(self) -> str | None:
        return llama_file_type_name(self.file_type)

    def tensor(self, name: str) -> GGUFTensorInfo:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise MissingGGUFTensorError(f"missing GGUF tensor: {name}")

    def require(self, names: Iterable[str]) -> tuple[GGUFTensorInfo, ...]:
        found: list[GGUFTensorInfo] = []
        missing: list[str] = []
        by_name = {tensor.name: tensor for tensor in self.tensors}
        for name in names:
            tensor = by_name.get(name)
            if tensor is None:
                missing.append(name)
            else:
                found.append(tensor)
        if missing:
            preview = ", ".join(missing[:8])
            more = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
            raise MissingGGUFTensorError(f"missing required GGUF tensors: {preview}{more}")
        return tuple(found)

    def names_with_prefix(self, prefix: str) -> tuple[str, ...]:
        return tuple(tensor.name for tensor in self.tensors if tensor.name.startswith(prefix))


class GGUFReader:
    """Header scanner plus lazy NumPy memmap access to GGUF tensor payloads."""

    def __init__(self, path: str | Path):
        self.info = scan_gguf(path)

    @property
    def path(self) -> Path:
        return self.info.path

    def tensor_info(self, name: str) -> GGUFTensorInfo:
        return self.info.tensor(name)

    def tensor_data(self, name: str):
        """Return a read-only NumPy memmap for a tensor's raw GGUF storage.

        Dense F32/F16/I* tensors are shaped as their logical tensor shape.
        BF16 tensors use ``uint16`` storage with the logical shape.  Block
        quantized tensors use the GGUF byte shape, where the final dimension is
        bytes per row.
        """

        import numpy as np

        tensor = self.tensor_info(name)
        dtype = numpy_storage_dtype(tensor.ggml_type)
        return np.memmap(
            self.info.path,
            mode="r",
            dtype=dtype,
            offset=tensor.data_offset,
            shape=tensor.byte_shape,
        )

    def dequantize_tensor(self, name: str):
        tensor = self.tensor_info(name)
        return dequantize_gguf_data(self.tensor_data(name), tensor.ggml_type)


def discover_gguf_files(model_path: str | Path) -> tuple[Path, ...]:
    path = resolve_model_path(model_path)
    if path.is_file() and path.suffix.lower() == ".gguf":
        return (path.resolve(),)
    if not path.is_dir():
        raise FileNotFoundError(f"GGUF path is not a .gguf file or directory: {path}")
    files = tuple(sorted(p.resolve() for p in path.glob("*.gguf")))
    if not files:
        raise FileNotFoundError(f"no .gguf files found under {path}")
    return files


def scan_gguf(path: str | Path) -> GGUFModelInfo:
    """Parse a GGUF v2/v3 header and tensor-info table without reading weights."""

    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as fh:
        if _read_exact(fh, 4) != GGUF_MAGIC:
            raise GGUFFormatError(f"GGUF magic invalid: {resolved}")
        version = _read_scalar(fh, GGUFValueType.UINT32)
        if version not in GGUF_SUPPORTED_VERSIONS:
            supported = ", ".join(str(v) for v in GGUF_SUPPORTED_VERSIONS)
            raise GGUFFormatError(
                f"unsupported GGUF version {version}; expected one of: {supported}"
            )
        tensor_count = _read_scalar(fh, GGUFValueType.UINT64)
        metadata_count = _read_scalar(fh, GGUFValueType.UINT64)

        metadata: dict[str, Any] = {}
        for _ in range(int(metadata_count)):
            key = _read_string(fh)
            value_type = GGUFValueType(_read_scalar(fh, GGUFValueType.UINT32))
            if key in metadata:
                raise GGUFFormatError(f"duplicate GGUF metadata key {key!r}")
            metadata[key] = _read_value(fh, value_type)

        raw_tensors: list[tuple[str, tuple[int, ...], int, int]] = []
        seen_names: set[str] = set()
        for _ in range(int(tensor_count)):
            name = _read_string(fh)
            if name in seen_names:
                raise GGUFFormatError(f"duplicate GGUF tensor name {name!r}")
            seen_names.add(name)
            n_dims = _read_scalar(fh, GGUFValueType.UINT32)
            ggml_shape = tuple(
                int(_read_scalar(fh, GGUFValueType.UINT64)) for _ in range(int(n_dims))
            )
            ggml_type_id = int(_read_scalar(fh, GGUFValueType.UINT32))
            offset = int(_read_scalar(fh, GGUFValueType.UINT64))
            raw_tensors.append((name, ggml_shape, ggml_type_id, offset))

        alignment = int(metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
        if alignment <= 0 or alignment & (alignment - 1):
            raise GGUFFormatError(
                f"invalid GGUF alignment {alignment}; expected non-zero power of two"
            )
        tensor_data_offset = _align_up(fh.tell(), alignment)

    file_size = resolved.stat().st_size
    tensors: list[GGUFTensorInfo] = []
    for name, ggml_shape, ggml_type_id, offset in raw_tensors:
        qtype = ggml_type(ggml_type_id)
        shape = tuple(reversed(ggml_shape))
        n_elements = int(prod(shape))
        nbytes = nbytes_for_shape(shape, qtype)
        data_offset = tensor_data_offset + offset
        byte_shape = quant_shape_to_byte_shape(shape, qtype)
        if data_offset < tensor_data_offset or data_offset + nbytes > file_size:
            raise GGUFFormatError(
                f"tensor {name!r} byte range [{data_offset}, {data_offset + nbytes}) "
                f"falls outside file size {file_size}"
            )
        tensors.append(
            GGUFTensorInfo(
                name=name,
                shape=shape,
                ggml_shape=ggml_shape,
                ggml_type=int(qtype),
                ggml_type_name=ggml_type_name(qtype),
                n_elements=n_elements,
                nbytes=nbytes,
                offset=offset,
                data_offset=data_offset,
                byte_shape=byte_shape,
            )
        )

    return GGUFModelInfo(
        path=resolved,
        version=int(version),
        alignment=alignment,
        metadata=metadata,
        tensors=tuple(tensors),
        tensor_data_offset=tensor_data_offset,
    )


load_gguf_index = scan_gguf


def _align_up(value: int, alignment: int) -> int:
    remainder = value % alignment
    return value if remainder == 0 else value + alignment - remainder


def _read_exact(fh, nbytes: int) -> bytes:
    data = fh.read(nbytes)
    if len(data) != nbytes:
        raise EOFError(f"unexpected EOF while reading {nbytes} bytes")
    return data


def _read_string(fh) -> str:
    length = _read_scalar(fh, GGUFValueType.UINT64)
    return _read_exact(fh, int(length)).decode("utf-8")


def _read_value(fh, value_type: GGUFValueType) -> Any:
    if value_type == GGUFValueType.STRING:
        return _read_string(fh)
    if value_type == GGUFValueType.ARRAY:
        item_type = GGUFValueType(_read_scalar(fh, GGUFValueType.UINT32))
        length = int(_read_scalar(fh, GGUFValueType.UINT64))
        return _read_array(fh, item_type, length)
    return _read_scalar(fh, value_type)


def _read_array(fh, item_type: GGUFValueType, length: int) -> list[Any]:
    if length == 0:
        return []
    if item_type == GGUFValueType.STRING:
        return [_read_string(fh) for _ in range(length)]
    if item_type == GGUFValueType.ARRAY:
        return [_read_value(fh, item_type) for _ in range(length)]

    fmt = _SCALAR_STRUCT_FORMATS[item_type]
    data = _read_exact(fh, struct.calcsize(fmt) * length)
    values = struct.unpack(f"<{length}{fmt}", data)
    return list(values)


def _read_scalar(fh, value_type: GGUFValueType) -> Any:
    fmt = _SCALAR_STRUCT_FORMATS[value_type]
    return struct.unpack(f"<{fmt}", _read_exact(fh, struct.calcsize(fmt)))[0]


_SCALAR_STRUCT_FORMATS = {
    GGUFValueType.UINT8: "B",
    GGUFValueType.INT8: "b",
    GGUFValueType.UINT16: "H",
    GGUFValueType.INT16: "h",
    GGUFValueType.UINT32: "I",
    GGUFValueType.INT32: "i",
    GGUFValueType.FLOAT32: "f",
    GGUFValueType.BOOL: "?",
    GGUFValueType.UINT64: "Q",
    GGUFValueType.INT64: "q",
    GGUFValueType.FLOAT64: "d",
}


__all__ = [
    "GGUF_DEFAULT_ALIGNMENT",
    "GGUF_MAGIC",
    "GGUF_SUPPORTED_VERSIONS",
    "GGUFFormatError",
    "GGUFModelInfo",
    "GGUFReader",
    "GGUFTensorInfo",
    "MissingGGUFTensorError",
    "discover_gguf_files",
    "load_gguf_index",
    "scan_gguf",
]
