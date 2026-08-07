"""Exact explicit and hidden AMDGPU kernarg packing."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Sequence

from hipengine.core.pm4.errors import Pm4InspectionError
from hipengine.core.pm4.metadata import AmdgpuKernelMetadata


@dataclass(frozen=True, slots=True)
class LaunchContext:
    grid_blocks: tuple[int, int, int]
    block: tuple[int, int, int]
    dynamic_shared_bytes: int

    def __post_init__(self) -> None:
        if len(self.grid_blocks) != 3 or any(value <= 0 for value in self.grid_blocks):
            raise Pm4InspectionError("launch grid-block dimensions must be positive")
        if len(self.block) != 3 or any(value <= 0 for value in self.block):
            raise Pm4InspectionError("launch block dimensions must be positive")
        if self.dynamic_shared_bytes < 0:
            raise Pm4InspectionError("dynamic shared-memory size must be non-negative")


_ZERO_HIDDEN_KINDS = frozenset(
    {
        "hidden_remainder_x",
        "hidden_remainder_y",
        "hidden_remainder_z",
        "hidden_global_offset_x",
        "hidden_global_offset_y",
        "hidden_global_offset_z",
        # Dedicated-queue service pointers are admitted as zero only when they
        # appear by these exact ABI names. New kinds must be reviewed, not
        # silently folded into this set.
        "hidden_printf_buffer",
        "hidden_default_queue",
        "hidden_completion_action",
        "hidden_multigrid_sync_arg",
        "hidden_hostcall_buffer",
    }
)


def _grid_dimensions(context: LaunchContext) -> int:
    if context.grid_blocks[2] > 1:
        return 3
    if context.grid_blocks[1] > 1:
        return 2
    return 1


def _hidden_value(kind: str, context: LaunchContext) -> int:
    values = {
        "hidden_block_count_x": context.grid_blocks[0],
        "hidden_block_count_y": context.grid_blocks[1],
        "hidden_block_count_z": context.grid_blocks[2],
        "hidden_group_size_x": context.block[0],
        "hidden_group_size_y": context.block[1],
        "hidden_group_size_z": context.block[2],
        "hidden_dynamic_lds_size": context.dynamic_shared_bytes,
        "hidden_grid_dims": _grid_dimensions(context),
    }
    if kind in values:
        return values[kind]
    if kind in _ZERO_HIDDEN_KINDS:
        return 0
    raise Pm4InspectionError(f"unsupported hidden kernarg value kind {kind!r}")


def pack_kernargs(
    metadata: AmdgpuKernelMetadata,
    explicit_values: Sequence[bytes],
    context: LaunchContext,
) -> bytes:
    """Pack explicit values and allowlisted hidden fields into one segment."""

    explicit_fields = [field for field in metadata.args if not field.value_kind.startswith("hidden_")]
    if len(explicit_values) != len(explicit_fields):
        raise Pm4InspectionError(
            "explicit argument count does not match AMDGPU metadata "
            f"({len(explicit_values)} != {len(explicit_fields)})"
        )

    packed = bytearray(metadata.kernarg_size)
    explicit_index = 0
    for field in metadata.args:
        end = field.offset + field.size
        if end > len(packed):
            raise Pm4InspectionError("kernarg field exceeds the declared segment")
        if field.value_kind.startswith("hidden_"):
            value = _hidden_value(field.value_kind, context)
            try:
                encoded = value.to_bytes(field.size, "little", signed=False)
            except OverflowError as exc:
                raise Pm4InspectionError(
                    f"hidden kernarg {field.value_kind!r} does not fit {field.size} bytes"
                ) from exc
        else:
            encoded = bytes(explicit_values[explicit_index])
            explicit_index += 1
            if len(encoded) != field.size:
                raise Pm4InspectionError(
                    f"explicit argument {explicit_index - 1} size {len(encoded)} "
                    f"does not match metadata size {field.size}"
                )
        packed[field.offset:end] = encoded
    return bytes(packed)


def _pointer_value(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    raw = getattr(value, "value", None)
    return int(raw or 0)


def _copy_extra_buffer(extra: object, segment_size: int) -> bytes | None:
    if not bool(extra):
        return None
    buffer_address = 0
    size_address = 0
    terminated = False
    for pair in range(16):
        key = _pointer_value(extra[pair * 2])
        if key == 1:
            buffer_address = _pointer_value(extra[pair * 2 + 1])
        elif key == 2:
            size_address = _pointer_value(extra[pair * 2 + 1])
        elif key == 3:
            terminated = True
            break
        else:
            raise Pm4InspectionError(f"unsupported HIP launch-extra key {key}")
    if not terminated:
        raise Pm4InspectionError("HIP launch-extra list is not terminated")
    if not buffer_address or not size_address:
        raise Pm4InspectionError("HIP launch-extra list lacks buffer pointer or size")
    size = int(ctypes.c_size_t.from_address(size_address).value)
    if size != segment_size:
        raise Pm4InspectionError(
            f"HIP packed kernarg size {size} does not match metadata size {segment_size}"
        )
    return ctypes.string_at(buffer_address, size)


def pack_kernel_node_params(
    metadata: AmdgpuKernelMetadata,
    kernel_params: object,
    extra: object,
    context: LaunchContext,
) -> bytes:
    """Copy a live HIP graph node's argument values into an owned segment."""

    packed_extra = _copy_extra_buffer(extra, metadata.kernarg_size)
    if packed_extra is not None:
        return packed_extra

    explicit_fields = [field for field in metadata.args if not field.value_kind.startswith("hidden_")]
    if explicit_fields and not bool(kernel_params):
        raise Pm4InspectionError("HIP kernel node has no explicit argument pointer array")
    explicit_values: list[bytes] = []
    for index, field in enumerate(explicit_fields):
        address = _pointer_value(kernel_params[index])
        if not address:
            raise Pm4InspectionError(f"HIP kernel argument {index} has a null value pointer")
        explicit_values.append(ctypes.string_at(address, field.size))
    return pack_kernargs(metadata, explicit_values, context)
