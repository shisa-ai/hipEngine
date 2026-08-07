"""Bounded ELF64 and clang offload-bundle parsing for PM4 graph inspection.

The parser intentionally supports only the little-endian ELF64/classic clang
formats emitted by hipEngine's current HIP toolchain. Unsupported or ambiguous
inputs fail closed instead of invoking external utilities or guessing.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Final

from hipengine.core.pm4.errors import Pm4InspectionError

_ELF_MAGIC: Final[bytes] = b"\x7fELF"
_CLANG_BUNDLE_MAGIC: Final[bytes] = b"__CLANG_OFFLOAD_BUNDLE__"
_MAX_SECTIONS: Final[int] = 65535
_MAX_BUNDLE_ENTRIES: Final[int] = 1024
_MAX_BUNDLE_ID_BYTES: Final[int] = 4096
_SHT_NOBITS: Final[int] = 8


@dataclass(frozen=True, slots=True)
class ElfSection:
    """One validated ELF64 section."""

    name: str
    section_type: int
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class SelectedCodeObject:
    """One exact architecture-matched AMDGPU image from a clang bundle."""

    target_id: str
    image: bytes
    sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pm4InspectionError(message)


def _range(data: bytes, offset: int, size: int, label: str) -> memoryview:
    _require(offset >= 0 and size >= 0, f"{label} range is negative")
    end = offset + size
    _require(end >= offset and end <= len(data), f"{label} range exceeds input")
    return memoryview(data)[offset:end]


def _u16(data: bytes, offset: int, label: str) -> int:
    return struct.unpack("<H", _range(data, offset, 2, label))[0]


def _u32(data: bytes, offset: int, label: str) -> int:
    return struct.unpack("<I", _range(data, offset, 4, label))[0]


def _u64(data: bytes, offset: int, label: str) -> int:
    return struct.unpack("<Q", _range(data, offset, 8, label))[0]


def _cstring(table: bytes, offset: int, label: str) -> str:
    _require(offset < len(table), f"{label} string offset exceeds table")
    end = table.find(b"\0", offset)
    _require(end >= 0, f"{label} string is not NUL terminated")
    try:
        return table[offset:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Pm4InspectionError(f"{label} string is not UTF-8") from exc


def parse_elf_sections(image: bytes) -> tuple[ElfSection, ...]:
    """Return validated sections from a little-endian ELF64 image."""

    _require(isinstance(image, bytes), "ELF image must be immutable bytes")
    _require(len(image) >= 64, "truncated ELF64 header")
    _require(image[:4] == _ELF_MAGIC, "input is not an ELF image")
    _require(image[4] == 2, "ELF image is not 64-bit")
    _require(image[5] == 1, "ELF image is not little-endian")
    _require(image[6] == 1, "unsupported ELF identification version")

    section_offset = _u64(image, 40, "ELF section table")
    section_entry_size = _u16(image, 58, "ELF section entry size")
    section_count = _u16(image, 60, "ELF section count")
    string_index = _u16(image, 62, "ELF section string index")
    _require(section_entry_size >= 64, "ELF64 section entries are too small")
    _require(0 < section_count <= _MAX_SECTIONS, "unsupported extended/empty ELF section table")
    _require(string_index < section_count, "ELF section string-table index is invalid")
    _range(
        image,
        section_offset,
        section_entry_size * section_count,
        "ELF section table",
    )

    def section_word(index: int, field_offset: int, width: int, label: str) -> int:
        base = section_offset + index * section_entry_size + field_offset
        if width == 4:
            return _u32(image, base, label)
        return _u64(image, base, label)

    strings_offset = section_word(string_index, 24, 8, "section strings offset")
    strings_size = section_word(string_index, 32, 8, "section strings size")
    strings = bytes(_range(image, strings_offset, strings_size, "section string table"))

    sections: list[ElfSection] = []
    for index in range(1, section_count):
        name_offset = section_word(index, 0, 4, "section name")
        section_type = section_word(index, 4, 4, "section type")
        offset = section_word(index, 24, 8, "section offset")
        size = section_word(index, 32, 8, "section size")
        name = _cstring(strings, name_offset, f"section {index} name")
        if section_type != _SHT_NOBITS:
            _range(image, offset, size, f"section {name!r}")
        sections.append(ElfSection(name, section_type, offset, size))
    return tuple(sections)


def extract_elf_section(image: bytes, name: str) -> bytes:
    """Extract exactly one named data-bearing section from ``image``."""

    _require(bool(name), "ELF section name must be non-empty")
    matches = [section for section in parse_elf_sections(image) if section.name == name]
    _require(bool(matches), f"ELF section {name!r} is missing")
    _require(len(matches) == 1, f"ELF section {name!r} is ambiguous")
    section = matches[0]
    _require(section.section_type != _SHT_NOBITS, f"ELF section {name!r} has no file bytes")
    return bytes(_range(image, section.offset, section.size, f"ELF section {name!r}"))


def elf_sections_of_type(image: bytes, section_type: int) -> tuple[bytes, ...]:
    """Return copied bytes for every section with ``section_type``."""

    result = []
    for section in parse_elf_sections(image):
        if section.section_type == section_type:
            _require(section.section_type != _SHT_NOBITS, "requested ELF section has no bytes")
            result.append(bytes(_range(image, section.offset, section.size, section.name)))
    return tuple(result)


def _gfx_arch(target_id: str) -> str | None:
    if "amdgcn-amd-amdhsa" not in target_id:
        return None
    target = target_id.rsplit("--", 1)[-1]
    arch = target.split(":", 1)[0]
    return arch if arch.startswith("gfx") else None


def select_amdgpu_code_object(bundle: bytes, gfx_arch: str) -> SelectedCodeObject:
    """Select exactly one classic clang bundle entry for ``gfx_arch``."""

    _require(gfx_arch.startswith("gfx") and not any(char.isspace() for char in gfx_arch),
             f"invalid gfx architecture {gfx_arch!r}")
    _require(bundle.startswith(_CLANG_BUNDLE_MAGIC), "input is not a classic clang offload bundle")
    cursor = len(_CLANG_BUNDLE_MAGIC)
    _require(cursor + 8 <= len(bundle), "truncated clang bundle count")
    entry_count = _u64(bundle, cursor, "clang bundle count")
    cursor += 8
    _require(0 < entry_count <= _MAX_BUNDLE_ENTRIES, "clang bundle entry count is invalid")

    entries: list[tuple[int, int, str]] = []
    for index in range(entry_count):
        _require(cursor + 24 <= len(bundle), f"truncated clang bundle descriptor {index}")
        offset = _u64(bundle, cursor, f"bundle entry {index} offset")
        size = _u64(bundle, cursor + 8, f"bundle entry {index} size")
        identifier_size = _u64(bundle, cursor + 16, f"bundle entry {index} ID size")
        cursor += 24
        _require(identifier_size <= _MAX_BUNDLE_ID_BYTES, "clang bundle ID is too large")
        identifier_bytes = bytes(
            _range(bundle, cursor, identifier_size, f"bundle entry {index} ID")
        )
        cursor += identifier_size
        try:
            identifier = identifier_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise Pm4InspectionError("clang bundle ID is not UTF-8") from exc
        _range(bundle, offset, size, f"bundle entry {index} payload")
        entries.append((offset, size, identifier))

    toc_end = cursor
    occupied: list[tuple[int, int]] = []
    for offset, size, identifier in entries:
        if size:
            _require(offset >= toc_end, f"bundle entry {identifier!r} payload overlaps its table")
            end = offset + size
            for other_start, other_end in occupied:
                _require(end <= other_start or offset >= other_end,
                         "clang bundle payload ranges overlap")
            occupied.append((offset, end))

    matches = [entry for entry in entries if _gfx_arch(entry[2]) == gfx_arch]
    _require(bool(matches), f"no AMDGPU code object matches {gfx_arch}")
    _require(len(matches) == 1, f"multiple AMDGPU code objects match {gfx_arch}")
    offset, size, identifier = matches[0]
    image = bytes(_range(bundle, offset, size, f"selected {gfx_arch} code object"))
    _require(image.startswith(_ELF_MAGIC), f"selected {gfx_arch} code object is not ELF")
    # Parse now so malformed selected code is rejected at the identity boundary.
    parse_elf_sections(image)
    return SelectedCodeObject(identifier, image, hashlib.sha256(image).hexdigest())
