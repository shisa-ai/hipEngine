"""Pure gfx1100 PM4 and vendor-AQL packet encoding.

The production native core carries the same constants because loaded kernel
addresses are available only after HSA executable relocation. These pure
functions are the deterministic packet oracle and are also useful for manifest
diagnostics.

Register and vendor-packet shapes are adapted from the Apache-2.0 Redline
reference at 33683f3d4f302a6c56bcc7a4c33ab8be3262dd2e and
ROCm/rocm-systems@c0430a50286200ab0562f4733445cdee6e48d416's public
``aqlprofile`` PM4-IB packet definition.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Final, Sequence

from hipengine.core.pm4.errors import Pm4InspectionError

PACKET3_SET_SH_REG: Final[int] = 0x76
PACKET3_DISPATCH_DIRECT: Final[int] = 0x15
PACKET3_EVENT_WRITE: Final[int] = 0x46
PACKET3_ACQUIRE_MEM: Final[int] = 0x58
PACKET3_INDIRECT_BUFFER: Final[int] = 0x3F

COMPUTE_NUM_THREAD_X: Final[int] = 0x207
COMPUTE_PGM_LO: Final[int] = 0x20C
COMPUTE_PGM_RSRC1: Final[int] = 0x212
COMPUTE_RESOURCE_LIMITS: Final[int] = 0x215
COMPUTE_TMPRING_SIZE: Final[int] = 0x218
COMPUTE_PGM_RSRC3: Final[int] = 0x228
COMPUTE_USER_DATA_0: Final[int] = 0x240

ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER: Final[int] = 1 << 0
ENABLE_SGPR_DISPATCH_PTR: Final[int] = 1 << 1
ENABLE_SGPR_QUEUE_PTR: Final[int] = 1 << 2
ENABLE_SGPR_KERNARG_SEGMENT_PTR: Final[int] = 1 << 3
ENABLE_SGPR_DISPATCH_ID: Final[int] = 1 << 4
ENABLE_SGPR_FLAT_SCRATCH_INIT: Final[int] = 1 << 5
ENABLE_SGPR_PRIVATE_SEGMENT_SIZE: Final[int] = 1 << 6
ENABLE_WAVEFRONT_SIZE32: Final[int] = 1 << 10
_SUPPORTED_PROPERTIES: Final[int] = (
    ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER | ENABLE_SGPR_KERNARG_SEGMENT_PTR | ENABLE_WAVEFRONT_SIZE32
)
_LDS_SIZE_MASK: Final[int] = 0x00FF8000
_LDS_SIZE_SHIFT: Final[int] = 15
_LDS_GRANULE: Final[int] = 512


@dataclass(frozen=True, slots=True)
class Gfx1100KernelImage:
    code_entry: int
    compute_pgm_rsrc1: int
    compute_pgm_rsrc2: int
    compute_pgm_rsrc3: int
    group_segment_size: int
    private_segment_size: int
    dynamic_callstack: bool
    wave32: bool
    kernel_code_properties: int


@dataclass(frozen=True, slots=True)
class DispatchGeometry:
    grid_workitems: tuple[int, int, int]
    block: tuple[int, int, int]

    def __post_init__(self) -> None:
        if len(self.grid_workitems) != 3 or any(value <= 0 for value in self.grid_workitems):
            raise Pm4InspectionError("dispatch grid dimensions must be positive")
        if len(self.block) != 3 or any(value <= 0 or value > 0xFFFF for value in self.block):
            raise Pm4InspectionError("dispatch block dimensions must be in 1..65535")
        if any(value > 0xFFFFFFFF for value in self.grid_workitems):
            raise Pm4InspectionError("dispatch grid dimensions exceed uint32")


def packet3(opcode: int, body_dwords: int, *, compute: bool) -> int:
    if not 0 <= opcode <= 0xFF or not 1 <= body_dwords <= 0x4000:
        raise Pm4InspectionError("invalid PACKET3 opcode or body length")
    return (3 << 30) | ((body_dwords - 1) << 16) | (opcode << 8) | ((1 << 1) if compute else 0)


def acquire_system() -> tuple[int, ...]:
    return (
        packet3(PACKET3_ACQUIRE_MEM, 7, compute=False),
        0,
        0xFFFFFFFF,
        0xFF,
        0,
        0,
        4,
        (1 << 16)
        | (1 << 15)
        | (1 << 14)
        | (1 << 9)
        | (1 << 8)
        | (1 << 7)
        | (1 << 6)
        | (1 << 5)
        | (1 << 4)
        | 1,
    )


def wait_compute_idle() -> tuple[int, ...]:
    return (packet3(PACKET3_EVENT_WRITE, 1, compute=False), 0x407)


def dependency_global() -> tuple[int, ...]:
    return (
        *wait_compute_idle(),
        packet3(PACKET3_ACQUIRE_MEM, 7, compute=False),
        0,
        0xFFFFFFFF,
        0x00FFFFFF,
        0,
        0,
        10,
        0x0C380,
    )


def dependency_local_cache() -> tuple[int, ...]:
    return (
        *wait_compute_idle(),
        packet3(PACKET3_ACQUIRE_MEM, 7, compute=False),
        0,
        0xFFFFFFFF,
        0x00FFFFFF,
        0,
        0,
        10,
        0x00380,
    )


def _validate_image(image: Gfx1100KernelImage, kernarg_address: int) -> list[int]:
    if image.private_segment_size != 0 or image.dynamic_callstack:
        raise Pm4InspectionError(
            "gfx1100 retained PM4 does not support scratch/private segments or dynamic stacks"
        )
    if image.code_entry <= 0 or image.code_entry & 0xFF:
        raise Pm4InspectionError("gfx1100 kernel code entry must be nonzero and 256-byte aligned")
    unsupported = image.kernel_code_properties & ~_SUPPORTED_PROPERTIES
    if unsupported:
        raise Pm4InspectionError(f"unsupported gfx1100 implicit SGPR properties 0x{unsupported:x}")
    property_wave32 = bool(image.kernel_code_properties & ENABLE_WAVEFRONT_SIZE32)
    if not image.wave32 or not property_wave32:
        raise Pm4InspectionError("initial gfx1100 retained PM4 path requires wave32")

    user_sgprs: list[int] = []
    if image.kernel_code_properties & ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER:
        user_sgprs.extend((0, 0, 0, 0))
    if image.kernel_code_properties & ENABLE_SGPR_KERNARG_SEGMENT_PTR:
        if kernarg_address <= 0:
            raise Pm4InspectionError("kernel requires a non-null kernarg address")
        user_sgprs.extend((kernarg_address & 0xFFFFFFFF, (kernarg_address >> 32) & 0xFFFFFFFF))
    if len(user_sgprs) > 16:
        raise Pm4InspectionError("gfx1100 dispatch requires more than 16 user SGPR dwords")
    return user_sgprs


def _set_sh_regs(
    words: list[int],
    first: int,
    values: Sequence[int],
    state: dict[int, int] | None,
) -> None:
    if not values:
        raise Pm4InspectionError("SET_SH_REG requires at least one value")
    if state is None:
        words.extend((packet3(PACKET3_SET_SH_REG, len(values) + 1, compute=True), first, *values))
        return

    run_first: int | None = None
    run_values: list[int] = []

    def flush() -> None:
        nonlocal run_first, run_values
        if run_first is not None:
            words.extend(
                (
                    packet3(PACKET3_SET_SH_REG, len(run_values) + 1, compute=True),
                    run_first,
                    *run_values,
                )
            )
        run_first = None
        run_values = []

    for offset, value in enumerate(values):
        register = first + offset
        value &= 0xFFFFFFFF
        if state.get(register) == value:
            flush()
            continue
        state[register] = value
        if run_first is None:
            run_first = register
        run_values.append(value)
    flush()


def _dispatch(
    words: list[int],
    image: Gfx1100KernelImage,
    geometry: DispatchGeometry,
    dynamic_group_bytes: int,
    kernarg_address: int,
    state: dict[int, int] | None,
) -> None:
    user_sgprs = _validate_image(image, kernarg_address)
    if dynamic_group_bytes < 0:
        raise Pm4InspectionError("dynamic group-segment size must be non-negative")
    total_group = image.group_segment_size + dynamic_group_bytes
    if total_group > 0xFFFFFFFF:
        raise Pm4InspectionError("group-segment size overflows uint32")
    lds_blocks = (total_group + _LDS_GRANULE - 1) // _LDS_GRANULE
    if lds_blocks > (_LDS_SIZE_MASK >> _LDS_SIZE_SHIFT):
        raise Pm4InspectionError("group-segment size cannot be encoded")
    rsrc2 = (image.compute_pgm_rsrc2 & ~_LDS_SIZE_MASK) | (lds_blocks << _LDS_SIZE_SHIFT)

    workgroups: list[int] = []
    for grid, block in zip(geometry.grid_workitems, geometry.block, strict=True):
        if grid % block:
            raise Pm4InspectionError("gfx1100 direct dispatch requires integral workgroups")
        workgroups.append(grid // block)

    _set_sh_regs(
        words,
        COMPUTE_PGM_LO,
        ((image.code_entry >> 8) & 0xFFFFFFFF, (image.code_entry >> 40) & 0xFFFFFFFF),
        state,
    )
    _set_sh_regs(words, COMPUTE_PGM_RSRC1, (image.compute_pgm_rsrc1, rsrc2), state)
    _set_sh_regs(words, COMPUTE_PGM_RSRC3, (image.compute_pgm_rsrc3,), state)
    _set_sh_regs(words, COMPUTE_TMPRING_SIZE, (0,), state)
    _set_sh_regs(words, COMPUTE_NUM_THREAD_X, geometry.block, state)
    _set_sh_regs(words, COMPUTE_RESOURCE_LIMITS, (0,), state)
    if user_sgprs:
        _set_sh_regs(words, COMPUTE_USER_DATA_0, user_sgprs, state)

    initiator = (1 << 0) | (1 << 2) | (1 << 3) | (1 << 15)
    words.extend((packet3(PACKET3_DISPATCH_DIRECT, 4, compute=True), *workgroups, initiator))


def encode_gfx1100_graph(
    dispatches: Sequence[tuple[Gfx1100KernelImage, DispatchGeometry, int, int]],
    *,
    acquire: bool = True,
    conservative_dependencies: bool = True,
    local_cache_dependencies: bool = False,
    stateful: bool = False,
) -> tuple[int, ...]:
    """Encode one serialized gfx1100 kernel tape and mandatory final flush."""

    if not dispatches:
        raise Pm4InspectionError("cannot encode an empty gfx1100 graph")
    words: list[int] = list(acquire_system()) if acquire else []
    state: dict[int, int] | None = {} if stateful else None
    for index, (image, geometry, dynamic_group, kernarg_address) in enumerate(dispatches):
        if index and conservative_dependencies:
            words.extend(
                dependency_local_cache() if local_cache_dependencies else dependency_global()
            )
        _dispatch(words, image, geometry, dynamic_group, kernarg_address, state)
    words.extend(wait_compute_idle())
    return tuple(words)


def vendor_pm4_ib_packet(*, address: int, dwords: int, completion_signal: int) -> tuple[bytes, int]:
    """Build AMD's 64-byte vendor-AQL PM4 indirect-buffer packet."""

    if address <= 0 or address & 3:
        raise Pm4InspectionError("PM4 indirect-buffer address must be nonzero and 4-byte aligned")
    if not 1 <= dwords <= 0x000FFFFF:
        raise Pm4InspectionError("PM4 indirect-buffer dword count is outside 1..0xfffff")
    if completion_signal <= 0:
        raise Pm4InspectionError("PM4 completion signal must be nonzero")
    aql_header = 1 << 8  # vendor type zero, barrier bit set
    publication = aql_header | (1 << 16)
    packet = bytearray(64)
    struct.pack_into("<HH", packet, 0, aql_header, 1)
    struct.pack_into("<I", packet, 4, packet3(PACKET3_INDIRECT_BUFFER, 3, compute=False))
    struct.pack_into("<II", packet, 8, address & 0xFFFFFFFC, (address >> 32) & 0xFFFFFFFF)
    struct.pack_into("<I", packet, 16, dwords | (1 << 23) | (3 << 28))
    struct.pack_into("<I", packet, 20, 10)
    struct.pack_into("<Q", packet, 56, completion_signal)
    return bytes(packet), publication
