"""Defensive AMDGPU MessagePack metadata parsing for exact kernarg layouts."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Final

from hipengine.core.pm4.elf import elf_sections_of_type
from hipengine.core.pm4.errors import Pm4InspectionError

_SHT_NOTE: Final[int] = 7
_NT_AMDGPU_METADATA: Final[int] = 32
_MAX_MSGPACK_DEPTH: Final[int] = 32
_MAX_MSGPACK_ENTRIES: Final[int] = 100_000
_MAX_STRING_BYTES: Final[int] = 1 << 20
_MAX_KERNARG_BYTES: Final[int] = 1 << 20


@dataclass(frozen=True, slots=True)
class KernargField:
    offset: int
    size: int
    value_kind: str


@dataclass(frozen=True, slots=True)
class AmdgpuKernelMetadata:
    name: str
    symbol: str
    kernarg_size: int
    kernarg_align: int
    group_segment_size: int
    private_segment_size: int
    dynamic_stack: bool
    wavefront_size: int
    args: tuple[KernargField, ...]


class _MsgpackReader:
    def __init__(self, data: bytes):
        self.data = data
        self.cursor = 0
        self.entries = 0

    def _take(self, size: int) -> bytes:
        end = self.cursor + size
        if size < 0 or end < self.cursor or end > len(self.data):
            raise Pm4InspectionError("truncated MessagePack metadata")
        result = self.data[self.cursor:end]
        self.cursor = end
        return result

    def _count(self, count: int) -> None:
        if count < 0 or count > _MAX_MSGPACK_ENTRIES - self.entries:
            raise Pm4InspectionError("MessagePack metadata entry limit exceeded")
        self.entries += count

    def read(self, depth: int = 0) -> Any:
        if depth > _MAX_MSGPACK_DEPTH:
            raise Pm4InspectionError("MessagePack metadata nesting limit exceeded")
        tag = self._take(1)[0]
        if tag <= 0x7F:
            return tag
        if tag >= 0xE0:
            return tag - 256
        if 0x80 <= tag <= 0x8F:
            return self._map(tag & 0x0F, depth)
        if 0x90 <= tag <= 0x9F:
            return self._array(tag & 0x0F, depth)
        if 0xA0 <= tag <= 0xBF:
            return self._text(tag & 0x1F)

        if tag == 0xC0:
            return None
        if tag == 0xC2:
            return False
        if tag == 0xC3:
            return True
        if tag == 0xC4:
            return self._binary(struct.unpack(">B", self._take(1))[0])
        if tag == 0xC5:
            return self._binary(struct.unpack(">H", self._take(2))[0])
        if tag == 0xC6:
            return self._binary(struct.unpack(">I", self._take(4))[0])
        if tag == 0xCA:
            value = struct.unpack(">f", self._take(4))[0]
            return value if math.isfinite(value) else None
        if tag == 0xCB:
            value = struct.unpack(">d", self._take(8))[0]
            return value if math.isfinite(value) else None
        if tag == 0xCC:
            return struct.unpack(">B", self._take(1))[0]
        if tag == 0xCD:
            return struct.unpack(">H", self._take(2))[0]
        if tag == 0xCE:
            return struct.unpack(">I", self._take(4))[0]
        if tag == 0xCF:
            return struct.unpack(">Q", self._take(8))[0]
        if tag == 0xD0:
            return struct.unpack(">b", self._take(1))[0]
        if tag == 0xD1:
            return struct.unpack(">h", self._take(2))[0]
        if tag == 0xD2:
            return struct.unpack(">i", self._take(4))[0]
        if tag == 0xD3:
            return struct.unpack(">q", self._take(8))[0]
        if tag == 0xD9:
            return self._text(struct.unpack(">B", self._take(1))[0])
        if tag == 0xDA:
            return self._text(struct.unpack(">H", self._take(2))[0])
        if tag == 0xDB:
            return self._text(struct.unpack(">I", self._take(4))[0])
        if tag == 0xDC:
            return self._array(struct.unpack(">H", self._take(2))[0], depth)
        if tag == 0xDD:
            return self._array(struct.unpack(">I", self._take(4))[0], depth)
        if tag == 0xDE:
            return self._map(struct.unpack(">H", self._take(2))[0], depth)
        if tag == 0xDF:
            return self._map(struct.unpack(">I", self._take(4))[0], depth)
        raise Pm4InspectionError(f"unsupported MessagePack metadata tag 0x{tag:02x}")

    def _binary(self, size: int) -> bytes:
        if size > _MAX_STRING_BYTES:
            raise Pm4InspectionError("MessagePack binary value is too large")
        return self._take(size)

    def _text(self, size: int) -> str:
        if size > _MAX_STRING_BYTES:
            raise Pm4InspectionError("MessagePack string is too large")
        try:
            return self._take(size).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise Pm4InspectionError("MessagePack string is not UTF-8") from exc

    def _array(self, count: int, depth: int) -> list[Any]:
        self._count(count)
        return [self.read(depth + 1) for _ in range(count)]

    def _map(self, count: int, depth: int) -> dict[str, Any]:
        self._count(count)
        result: dict[str, Any] = {}
        for _ in range(count):
            key = self.read(depth + 1)
            if isinstance(key, bytes):
                try:
                    key = key.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise Pm4InspectionError("MessagePack map key is not UTF-8") from exc
            if not isinstance(key, str):
                raise Pm4InspectionError("MessagePack metadata map key is not a string")
            if key in result:
                raise Pm4InspectionError(f"duplicate MessagePack metadata key {key!r}")
            result[key] = self.read(depth + 1)
        return result


def _parse_msgpack(data: bytes) -> Any:
    reader = _MsgpackReader(data)
    try:
        result = reader.read()
    except Pm4InspectionError as exc:
        if "MessagePack" in str(exc):
            raise
        raise Pm4InspectionError(f"invalid MessagePack metadata: {exc}") from exc
    if reader.cursor != len(data):
        raise Pm4InspectionError("MessagePack metadata has trailing bytes")
    return result


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def _metadata_notes(hsaco: bytes) -> tuple[bytes, ...]:
    notes: list[bytes] = []
    for section in elf_sections_of_type(hsaco, _SHT_NOTE):
        cursor = 0
        while cursor < len(section):
            if len(section) - cursor < 12:
                if not any(section[cursor:]):
                    break
                raise Pm4InspectionError("truncated ELF note header")
            name_size, description_size, note_type = struct.unpack_from("<III", section, cursor)
            cursor += 12
            name_end = cursor + name_size
            if name_end > len(section):
                raise Pm4InspectionError("ELF note owner range exceeds section")
            owner = section[cursor:name_end].rstrip(b"\0")
            cursor = _align(name_end, 4)
            description_end = cursor + description_size
            if description_end > len(section):
                raise Pm4InspectionError("ELF note description range exceeds section")
            description = section[cursor:description_end]
            cursor = _align(description_end, 4)
            if cursor > len(section):
                raise Pm4InspectionError("ELF note alignment exceeds section")
            if note_type == _NT_AMDGPU_METADATA and owner == b"AMDGPU":
                notes.append(bytes(description))
    if not notes:
        raise Pm4InspectionError("HSACO has no AMDGPU metadata note")
    return tuple(notes)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Pm4InspectionError(f"{label} is not a metadata map")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise Pm4InspectionError(f"{label} is not a metadata array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Pm4InspectionError(f"{label} is not a non-empty string")
    return value


def _integer(value: Any, label: str, *, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Pm4InspectionError(f"{label} is not a non-negative integer")
    return value


def _boolean(value: Any, label: str, *, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise Pm4InspectionError(f"{label} is not a boolean")
    return value


def _parse_kernel(value: Any) -> AmdgpuKernelMetadata:
    kernel = _mapping(value, "AMDGPU kernel")
    name = _string(kernel.get(".name"), "kernel .name")
    symbol = _string(kernel.get(".symbol"), "kernel .symbol")
    kernarg_size = _integer(kernel.get(".kernarg_segment_size"), "kernarg segment size")
    if kernarg_size > _MAX_KERNARG_BYTES:
        raise Pm4InspectionError("kernarg segment is too large")
    kernarg_align = _integer(
        kernel.get(".kernarg_segment_align"), "kernarg segment alignment", default=1
    )
    if kernarg_align == 0 or kernarg_align & (kernarg_align - 1):
        raise Pm4InspectionError("kernarg segment alignment is not a positive power of two")
    group_size = _integer(
        kernel.get(".group_segment_fixed_size"), "group segment size", default=0
    )
    private_size = _integer(
        kernel.get(".private_segment_fixed_size"), "private segment size", default=0
    )
    dynamic_stack = _boolean(kernel.get(".uses_dynamic_stack"), "dynamic-stack flag")
    wavefront_size = _integer(kernel.get(".wavefront_size"), "wavefront size", default=0)

    args_value = kernel.get(".args", [])
    args = _sequence(args_value, "kernel .args")
    fields: list[KernargField] = []
    previous_end = 0
    for index, value in enumerate(args):
        arg = _mapping(value, f"kernel argument {index}")
        offset = _integer(arg.get(".offset"), f"kernel argument {index} offset")
        size = _integer(arg.get(".size"), f"kernel argument {index} size")
        if size == 0:
            raise Pm4InspectionError(f"kernel argument {index} has zero size")
        value_kind = _string(
            arg.get(".value_kind", "by_value"), f"kernel argument {index} value kind"
        )
        end = offset + size
        if end < offset or end > kernarg_size:
            raise Pm4InspectionError(f"kernel argument {index} exceeds kernarg segment")
        if offset < previous_end:
            raise Pm4InspectionError(f"kernel argument {index} overlaps the preceding field")
        previous_end = end
        fields.append(KernargField(offset, size, value_kind))

    return AmdgpuKernelMetadata(
        name=name,
        symbol=symbol,
        kernarg_size=kernarg_size,
        kernarg_align=kernarg_align,
        group_segment_size=group_size,
        private_segment_size=private_size,
        dynamic_stack=dynamic_stack,
        wavefront_size=wavefront_size,
        args=tuple(fields),
    )


def parse_amdgpu_kernels(hsaco: bytes) -> dict[str, AmdgpuKernelMetadata]:
    """Parse exact `.symbol` keyed kernel metadata from an AMDGPU HSACO."""

    kernels: dict[str, AmdgpuKernelMetadata] = {}
    names: set[str] = set()
    for note in _metadata_notes(hsaco):
        try:
            root = _mapping(_parse_msgpack(note), "AMDGPU metadata root")
        except Pm4InspectionError as exc:
            if "MessagePack" in str(exc):
                raise
            raise Pm4InspectionError(f"invalid AMDGPU MessagePack metadata: {exc}") from exc
        entries = _sequence(root.get("amdhsa.kernels"), "amdhsa.kernels")
        for entry in entries:
            kernel = _parse_kernel(entry)
            if kernel.symbol in kernels or kernel.name in names:
                raise Pm4InspectionError(f"duplicate AMDGPU kernel metadata for {kernel.symbol!r}")
            kernels[kernel.symbol] = kernel
            names.add(kernel.name)
    if not kernels:
        raise Pm4InspectionError("AMDGPU metadata contains no kernels")
    return kernels


def resolve_kernel_metadata(
    kernels: dict[str, AmdgpuKernelMetadata], requested_name: str
) -> AmdgpuKernelMetadata:
    """Resolve one HIP kernel name without accepting ambiguous aliases."""

    candidates = [
        kernel
        for kernel in kernels.values()
        if requested_name in {kernel.name, kernel.symbol, kernel.symbol.removesuffix(".kd")}
    ]
    if not candidates:
        raise Pm4InspectionError(f"AMDGPU metadata has no kernel matching {requested_name!r}")
    if len(candidates) != 1:
        raise Pm4InspectionError(f"AMDGPU metadata kernel {requested_name!r} is ambiguous")
    return candidates[0]
