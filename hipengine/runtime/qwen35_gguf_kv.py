"""Physical KV storage for the qwen35/qwen35moe GGUF architecture family.

Owns resolved layouts, page backing, byte accounting and allocation/release.
Execution, session policy, admission and growth coordination belong to callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime
from hipengine.core.memory import DeviceBuffer, free, malloc
from hipengine.core.tensor import Tensor
from hipengine.core.virtual_memory import VirtualMemoryBuffer
from hipengine.kvcache import KVScaleMetadata
from hipengine.loading.qwen35_gguf import LINEAR_ATTENTION


@dataclass(frozen=True)
class Qwen35GGUFKVChunkLayout:
    """Complete policy identity for one scheduler-owned GGUF KV chunk.

    ``layer_storage_dtypes`` is aligned with model layers and uses ``None`` for
    linear-attention layers.  Keeping the resolved per-layer layout beside the
    backing lets resident sessions reject payload/scale mismatches before any
    device metadata is changed.
    """

    storage_dtype: DType
    storage_layout: str
    scale_dtype: DType
    scale_granularity: str
    int8_kv_value_bf16: bool
    layer_storage_dtypes: tuple[DType | None, ...]
    bf16_mirror_layer_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        storage = DType.parse(self.storage_dtype)
        scale = DType.parse(self.scale_dtype)
        layers = tuple(
            None if layer_storage is None else DType.parse(layer_storage)
            for layer_storage in self.layer_storage_dtypes
        )
        mirrors = tuple(sorted({int(index) for index in self.bf16_mirror_layer_indices}))
        object.__setattr__(self, "storage_dtype", storage)
        object.__setattr__(self, "scale_dtype", scale)
        object.__setattr__(self, "layer_storage_dtypes", layers)
        object.__setattr__(self, "bf16_mirror_layer_indices", mirrors)
        if storage not in {DType.BF16, DType.INT8_PER_TOKEN_HEAD}:
            raise ValueError("GGUF dynamic KV storage must be bf16 or int8_per_token_head")
        if scale not in {DType.FP16, DType.FP32}:
            raise ValueError("GGUF dynamic KV scales must use fp16 or fp32")
        if self.storage_layout not in {"uniform", "tail4_hadamard_group32"}:
            raise ValueError("unsupported GGUF dynamic KV storage layout")
        if self.scale_granularity not in {"per_token_head", "block16", "hadamard_group32"}:
            raise ValueError("unsupported GGUF dynamic KV scale granularity")
        if not layers or not any(layer_storage is not None for layer_storage in layers):
            raise ValueError("GGUF dynamic KV layout requires a full-attention layer")
        if any(
            layer_storage not in {None, DType.BF16, DType.INT8_PER_TOKEN_HEAD}
            for layer_storage in layers
        ):
            raise ValueError("unsupported per-layer GGUF dynamic KV storage dtype")
        int8_layers = tuple(
            index
            for index, layer_storage in enumerate(layers)
            if layer_storage == DType.INT8_PER_TOKEN_HEAD
        )
        if storage == DType.BF16 and int8_layers:
            raise ValueError("BF16 GGUF dynamic KV layout cannot contain INT8 layers")
        if storage == DType.INT8_PER_TOKEN_HEAD and not int8_layers:
            raise ValueError("INT8 GGUF dynamic KV layout must contain an INT8 layer")
        if self.storage_layout == "tail4_hadamard_group32" and (
            storage != DType.INT8_PER_TOKEN_HEAD
            or self.scale_granularity != "hadamard_group32"
            or len(int8_layers) != 4
        ):
            raise ValueError("tail4_hadamard_group32 requires exactly four INT8 layers")
        if self.int8_kv_value_bf16 and (
            storage != DType.INT8_PER_TOKEN_HEAD
            or self.scale_granularity != "per_token_head"
        ):
            raise ValueError("BF16-value INT8 GGUF dynamic KV requires per-token-head scales")
        if any(index < 0 or index >= len(layers) for index in mirrors):
            raise ValueError("GGUF dynamic KV mirror layer index is outside the model")
        if any(layers[index] != DType.INT8_PER_TOKEN_HEAD for index in mirrors):
            raise ValueError("GGUF dynamic KV mirrors are valid only for INT8 layers")
        if mirrors and self.scale_granularity == "hadamard_group32":
            raise ValueError("Hadamard-group32 GGUF dynamic KV does not use BF16 mirrors")


def _gguf_kv_attention_source(layout: Qwen35GGUFKVChunkLayout) -> str:
    int8_layers = {
        layer_id
        for layer_id, storage in enumerate(layout.layer_storage_dtypes)
        if storage == DType.INT8_PER_TOKEN_HEAD
    }
    if not int8_layers:
        return "bf16"
    if int8_layers - set(layout.bf16_mirror_layer_indices):
        return "int8_direct"
    return "bf16_mirror"


def _validate_bound_blocks_against_capacity(
    block_ids: tuple[int, ...],
    *,
    start_block_id: int,
    page_capacity: int,
) -> None:
    """Validate request-local page ids against a whole pool's page range.

    The per-chunk check in ``Qwen35GGUFKVChunkBacking.validate_bound_blocks``
    is right for a pool that hands out one contiguous chunk. A pool that grows
    by appending chunks keeps page ids stable and reaches every page through
    its pointer tables, so its allocations must be validated against the pool
    capacity instead.
    """

    blocks = tuple(int(block_id) for block_id in block_ids)
    if not blocks:
        raise ValueError("bound GGUF KV allocation must contain pages")
    if len(set(blocks)) != len(blocks):
        raise ValueError("bound GGUF KV allocation pages must be unique")
    local_pages = tuple(block_id - int(start_block_id) for block_id in blocks)
    if any(local_page < 0 or local_page >= int(page_capacity) for local_page in local_pages):
        raise ValueError("GGUF KV allocation is outside its pool page range")


class VmmPlaneBuffer:
    """``DeviceBuffer``-compatible view over one plane of a stable VMM arena.

    The view reports the logical payload (``pages * plane_page_bytes``) rather
    than the granularity-aligned committed bytes, so the chunk invariant
    ``nbytes % pages == 0`` holds and page strides stay exact while the arena's
    physical commitment trails the reservation. ``close`` releases the owning
    :class:`VirtualMemoryBuffer`, which unmaps every committed segment and frees
    the address reservation exactly once. A caller that needs the physical
    accounting reads :attr:`VirtualMemoryBuffer.committed_bytes` through
    :attr:`owner`.
    """

    __slots__ = ("_owner", "_offset", "_nbytes")

    def __init__(self, owner: VirtualMemoryBuffer, *, offset: int, nbytes: int) -> None:
        self._owner = owner
        self._offset = int(offset)
        self._nbytes = int(nbytes)

    @property
    def ptr(self) -> int:
        return int(self._owner.ptr) + self._offset

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def owner(self) -> VirtualMemoryBuffer:
        return self._owner

    def close(self) -> None:
        self._owner.close()


def _free_gguf_kv_plane_buffer(buffer: object, *, runtime: HipRuntime) -> None:
    """Free one KV plane buffer, dispatching VMM views to their arena close."""

    if isinstance(buffer, VmmPlaneBuffer):
        buffer.close()
        return
    free(buffer, runtime=runtime)  # type: ignore[arg-type]


def _gguf_kv_backing_arena_identity(backing: object) -> tuple[int, ...] | None:
    """Stable device-arena identity for a GGUF KV chunk backing.

    VMM growth rebuilds the backing object while reusing the same per-plane
    :class:`VirtualMemoryBuffer` owners, so two backing objects over one arena
    address the same physical pages. A plain ``hipMalloc`` chunk has no stable
    identity beyond the object itself, so this returns ``None`` and the caller
    keeps its strict object-identity check.
    """

    buffers = getattr(backing, "buffers", None)
    if buffers is None:
        return None
    owners: list[int] = []
    for buffer in buffers:
        owner = getattr(buffer, "owner", None)
        if owner is None:
            return None
        owners.append(int(owner.ptr))
    return tuple(owners) if owners else None


def _gguf_kv_backings_share_arena(first: object, second: object) -> bool:
    """True when two chunk backings address the same stable device arena.

    Object identity is still the fast path; the arena identity only widens the
    check to a rebuilt VMM backing over the same owners. It does not relax any
    other safety check.
    """

    if first is second:
        return True
    identity = _gguf_kv_backing_arena_identity(first)
    return identity is not None and identity == _gguf_kv_backing_arena_identity(second)


@dataclass(frozen=True)
class Qwen35GGUFKVChunkBacking:
    """Payload and scale backing for one contiguous chunk of logical KV pages."""

    layout: Qwen35GGUFKVChunkLayout
    start_block_id: int
    pages: int
    full_key_caches: tuple[DeviceBuffer | None, ...]
    full_value_caches: tuple[DeviceBuffer | None, ...]
    full_bf16_mirror_key_caches: tuple[DeviceBuffer | None, ...]
    full_bf16_mirror_value_caches: tuple[DeviceBuffer | None, ...]
    full_k_scale_caches: tuple[DeviceBuffer | None, ...]
    full_v_scale_caches: tuple[DeviceBuffer | None, ...]
    full_kv_scale_metadata: tuple[KVScaleMetadata | None, ...]
    buffers: tuple[DeviceBuffer, ...]

    def __post_init__(self) -> None:
        if int(self.start_block_id) < 0:
            raise ValueError("GGUF KV chunk start_block_id must be non-negative")
        if int(self.pages) <= 0:
            raise ValueError("GGUF KV chunk pages must be positive")
        layer_count = len(self.layout.layer_storage_dtypes)
        aligned = (
            self.full_key_caches,
            self.full_value_caches,
            self.full_bf16_mirror_key_caches,
            self.full_bf16_mirror_value_caches,
            self.full_k_scale_caches,
            self.full_v_scale_caches,
            self.full_kv_scale_metadata,
        )
        if any(len(values) != layer_count for values in aligned):
            raise ValueError("GGUF KV chunk layer tuples must align with the layout")
        mirror_layers = frozenset(self.layout.bf16_mirror_layer_indices)
        for layer_id, layer_storage in enumerate(self.layout.layer_storage_dtypes):
            key_cache = self.full_key_caches[layer_id]
            value_cache = self.full_value_caches[layer_id]
            mirror_key = self.full_bf16_mirror_key_caches[layer_id]
            mirror_value = self.full_bf16_mirror_value_caches[layer_id]
            k_scale = self.full_k_scale_caches[layer_id]
            v_scale = self.full_v_scale_caches[layer_id]
            metadata = self.full_kv_scale_metadata[layer_id]
            if layer_storage is None:
                if any(
                    value is not None
                    for value in (key_cache, value_cache, mirror_key, mirror_value, k_scale, v_scale, metadata)
                ):
                    raise ValueError("linear-attention GGUF KV chunk entries must be empty")
                continue
            if key_cache is None or value_cache is None:
                raise ValueError("full-attention GGUF KV chunk payload is incomplete")
            if (mirror_key is None) != (mirror_value is None):
                raise ValueError("GGUF KV chunk BF16 mirror payload must be paired")
            if (layer_id in mirror_layers) != (mirror_key is not None):
                raise ValueError("GGUF KV chunk BF16 mirror payload does not match its layout")
            if layer_storage == DType.INT8_PER_TOKEN_HEAD:
                if k_scale is None or v_scale is None or metadata is None:
                    raise ValueError("INT8 GGUF KV chunk requires paired scale payload and metadata")
                if int(metadata.k_scale.ptr) != int(k_scale.ptr) or int(metadata.v_scale.ptr) != int(v_scale.ptr):
                    raise ValueError("GGUF KV chunk scale metadata does not reference its backing")
                if metadata.scale_dtype != self.layout.scale_dtype:
                    raise ValueError("GGUF KV chunk scale dtype does not match its layout")
                if metadata.granularity != self.layout.scale_granularity:
                    raise ValueError("GGUF KV chunk scale granularity does not match its layout")
                if int(metadata.k_scale.shape[0]) != int(self.pages):
                    raise ValueError("GGUF KV chunk scale metadata does not cover every page")
            elif any(value is not None for value in (k_scale, v_scale, metadata)):
                raise ValueError("BF16 GGUF KV chunk layer cannot carry INT8 scales")
        if any(int(buffer.nbytes) <= 0 or int(buffer.nbytes) % int(self.pages) for buffer in self.buffers):
            raise ValueError("GGUF KV chunk buffers must contain whole positive pages")

    def release_full_attention_layer(self, layer_id: int, *, runtime: HipRuntime) -> None:
        """Free one full-attention layer's payload planes early.

        Layerwise DMS prefill (2026-09-07 memory review, target 1): after a
        compact layer has been packed from this dense backing, its source
        planes are dead. The layer's key/value (and mirror/scale) buffers are
        freed, their tuple entries are replaced with None and the buffers are
        pruned so a later chunk free never double-frees them. Only valid on
        backings whose layers are whole-buffer owned (every GGUF chunk backing
        is); linear-attention layers and already-released layers are rejected.
        """

        layer_id = int(layer_id)
        if not 0 <= layer_id < len(self.layout.layer_storage_dtypes):
            raise ValueError("GGUF KV chunk layer index is out of range")
        if self.layout.layer_storage_dtypes[layer_id] is None:
            raise ValueError("cannot release a linear-attention GGUF KV chunk layer")
        if self.full_key_caches[layer_id] is None:
            raise ValueError("GGUF KV chunk layer is already released")
        released = [
            self.full_key_caches[layer_id],
            self.full_value_caches[layer_id],
            self.full_bf16_mirror_key_caches[layer_id],
            self.full_bf16_mirror_value_caches[layer_id],
            self.full_k_scale_caches[layer_id],
            self.full_v_scale_caches[layer_id],
        ]
        released = [buffer for buffer in released if buffer is not None]
        for buffer in released:
            _free_gguf_kv_plane_buffer(buffer, runtime=runtime)
        released_ids = {id(buffer) for buffer in released}
        object.__setattr__(
            self,
            "full_key_caches",
            tuple(
                None if index == layer_id else buffer
                for index, buffer in enumerate(self.full_key_caches)
            ),
        )
        object.__setattr__(
            self,
            "full_value_caches",
            tuple(
                None if index == layer_id else buffer
                for index, buffer in enumerate(self.full_value_caches)
            ),
        )
        object.__setattr__(
            self,
            "full_bf16_mirror_key_caches",
            tuple(
                None if index == layer_id else buffer
                for index, buffer in enumerate(self.full_bf16_mirror_key_caches)
            ),
        )
        object.__setattr__(
            self,
            "full_bf16_mirror_value_caches",
            tuple(
                None if index == layer_id else buffer
                for index, buffer in enumerate(self.full_bf16_mirror_value_caches)
            ),
        )
        object.__setattr__(
            self,
            "full_k_scale_caches",
            tuple(
                None if index == layer_id else buffer
                for index, buffer in enumerate(self.full_k_scale_caches)
            ),
        )
        object.__setattr__(
            self,
            "full_v_scale_caches",
            tuple(
                None if index == layer_id else buffer
                for index, buffer in enumerate(self.full_v_scale_caches)
            ),
        )
        object.__setattr__(
            self,
            "full_kv_scale_metadata",
            tuple(
                None if index == layer_id else metadata
                for index, metadata in enumerate(self.full_kv_scale_metadata)
            ),
        )
        object.__setattr__(
            self,
            "buffers",
            tuple(
                buffer
                for buffer in self.buffers
                if id(buffer) not in released_ids
            ),
        )

    @property
    def total_nbytes(self) -> int:
        return sum(int(buffer.nbytes) for buffer in self.buffers)

    @property
    def page_bytes(self) -> int:
        return self.total_nbytes // int(self.pages)

    def page_pointer(self, local_page: int) -> int:
        page = int(local_page)
        if page < 0 or page >= self.pages:
            raise IndexError("local KV page is outside the chunk")
        first = next((buffer for buffer in self.full_key_caches if buffer is not None), None)
        if first is None:
            raise RuntimeError("GGUF KV chunk has no full-attention layer backing")
        return int(first.ptr) + page * (int(first.nbytes) // int(self.pages))

    def validate_bound_blocks(self, block_ids: tuple[int, ...]) -> None:
        """Validate request-local page ids without slicing the shared backing."""

        blocks = tuple(int(block_id) for block_id in block_ids)
        if not blocks:
            raise ValueError("bound GGUF KV allocation must contain pages")
        if len(set(blocks)) != len(blocks):
            raise ValueError("bound GGUF KV allocation pages must be unique")
        local_pages = tuple(block_id - int(self.start_block_id) for block_id in blocks)
        if any(local_page < 0 or local_page >= self.pages for local_page in local_pages):
            raise ValueError("GGUF KV allocation is outside its backing chunk")


def _qwen35_gguf_kv_scale_shape(
    cfg: object,
    *,
    pages: int,
    granularity: str,
) -> tuple[int, ...]:
    base = (int(pages), 256, int(cfg.head_count_kv))
    if granularity == "per_token_head":
        return base
    group_size = 16 if granularity == "block16" else 32
    if int(cfg.key_length) % group_size:
        raise ValueError(f"GGUF {granularity} KV requires key_length divisible by {group_size}")
    groups = int(cfg.key_length) // group_size
    if granularity == "block16" and groups != 16:
        raise ValueError("GGUF INT8 KV block16 scales require head_dim/key_length 256")
    return (*base, groups)


def _qwen35_gguf_kv_page_bytes(cfg: object, layout: Qwen35GGUFKVChunkLayout) -> int:
    payload_elements = 256 * int(cfg.head_count_kv) * int(cfg.key_length)
    scale_elements = int(np.prod(_qwen35_gguf_kv_scale_shape(cfg, pages=1, granularity=layout.scale_granularity)))
    mirror_layers = frozenset(layout.bf16_mirror_layer_indices)
    page_bytes = 0
    for layer_id, layer_storage in enumerate(layout.layer_storage_dtypes):
        if layer_storage is None:
            continue
        key_dtype = DType.INT8 if layer_storage == DType.INT8_PER_TOKEN_HEAD else DType.BF16
        value_dtype = (
            DType.BF16
            if layer_storage == DType.INT8_PER_TOKEN_HEAD and layout.int8_kv_value_bf16
            else key_dtype
        )
        page_bytes += payload_elements * (key_dtype.itemsize + value_dtype.itemsize)
        if layer_id in mirror_layers:
            page_bytes += 2 * payload_elements * DType.BF16.itemsize
        if layer_storage == DType.INT8_PER_TOKEN_HEAD:
            page_bytes += 2 * scale_elements * layout.scale_dtype.itemsize
    return int(page_bytes)


def _qwen35_gguf_kv_plane_specs(
    cfg: object,
    layout: Qwen35GGUFKVChunkLayout,
) -> tuple[tuple[str, int], ...]:
    """Return ``(role, page_nbytes)`` for every device plane of the layout.

    This is the single source of the plane role names and per-page byte sizes;
    both the VMM allocator and the pointer-table builder derive their planes from
    it, so a plane can never be reserved under one name and published under
    another.
    """

    payload_elements = 256 * int(cfg.head_count_kv) * int(cfg.key_length)
    scale_elements = int(
        np.prod(
            _qwen35_gguf_kv_scale_shape(
                cfg, pages=1, granularity=layout.scale_granularity
            )
        )
    )
    mirror_layers = frozenset(layout.bf16_mirror_layer_indices)
    specs: list[tuple[str, int]] = []
    for layer_id, layer_storage in enumerate(layout.layer_storage_dtypes):
        if layer_storage is None:
            continue
        key_dtype = DType.INT8 if layer_storage == DType.INT8_PER_TOKEN_HEAD else DType.BF16
        value_dtype = (
            DType.BF16
            if layer_storage == DType.INT8_PER_TOKEN_HEAD and layout.int8_kv_value_bf16
            else key_dtype
        )
        specs.append(
            (f"layer{layer_id}.key_payload", payload_elements * key_dtype.itemsize)
        )
        specs.append(
            (f"layer{layer_id}.value_payload", payload_elements * value_dtype.itemsize)
        )
        if layer_id in mirror_layers:
            mirror_nbytes = payload_elements * DType.BF16.itemsize
            specs.append((f"layer{layer_id}.bf16_mirror_key", mirror_nbytes))
            specs.append((f"layer{layer_id}.bf16_mirror_value", mirror_nbytes))
        if layer_storage == DType.INT8_PER_TOKEN_HEAD:
            scale_nbytes = scale_elements * layout.scale_dtype.itemsize
            specs.append((f"layer{layer_id}.key_scale", scale_nbytes))
            specs.append((f"layer{layer_id}.value_scale", scale_nbytes))
    return tuple(specs)


def _qwen35_gguf_vmm_physical_bytes(
    pages: int,
    *,
    plane_page_nbytes: Mapping[str, int],
    granularity: int,
) -> int:
    """Granularity-aligned physical bytes a VMM arena commits for ``pages``.

    Every plane rounds its own committed region up to the device allocation
    granularity, so the physical commitment is the sum of the padded plane
    payloads, not ``pages * page_bytes``. Used to price a prospective initial or
    growth commitment before any physical memory is mapped.
    """

    count = max(0, int(pages))
    gran = max(1, int(granularity))
    return sum(
        ((count * int(page_nbytes) + gran - 1) // gran) * gran
        for page_nbytes in plane_page_nbytes.values()
    )


def _qwen35_gguf_kv_backing_planes(
    backing: Qwen35GGUFKVChunkBacking,
):
    """Yield ``(role, buffer)`` for every non-empty plane of a chunk backing."""

    for layer_id in range(len(backing.layout.layer_storage_dtypes)):
        yield f"layer{layer_id}.key_payload", backing.full_key_caches[layer_id]
        yield f"layer{layer_id}.value_payload", backing.full_value_caches[layer_id]
        yield (
            f"layer{layer_id}.bf16_mirror_key",
            backing.full_bf16_mirror_key_caches[layer_id],
        )
        yield (
            f"layer{layer_id}.bf16_mirror_value",
            backing.full_bf16_mirror_value_caches[layer_id],
        )
        yield f"layer{layer_id}.key_scale", backing.full_k_scale_caches[layer_id]
        yield f"layer{layer_id}.value_scale", backing.full_v_scale_caches[layer_id]


def _build_qwen35_gguf_kv_backing_from_plane_views(
    *,
    cfg: object,
    layout: Qwen35GGUFKVChunkLayout,
    pages: int,
    start_block_id: int,
    plane_views: Mapping[str, object],
) -> Qwen35GGUFKVChunkBacking:
    """Assemble a chunk backing from caller-supplied per-plane buffers.

    Scale metadata is rebuilt for ``pages`` so its tensor bounds are honest for
    the committed capacity rather than for a larger reservation.
    """

    page_count = int(pages)
    device = Device("hip", 0)
    scale_shape = _qwen35_gguf_kv_scale_shape(
        cfg, pages=page_count, granularity=layout.scale_granularity
    )
    mirror_layers = frozenset(layout.bf16_mirror_layer_indices)
    key_caches: list[object | None] = []
    value_caches: list[object | None] = []
    mirror_key_caches: list[object | None] = []
    mirror_value_caches: list[object | None] = []
    k_scale_caches: list[object | None] = []
    v_scale_caches: list[object | None] = []
    scale_metadata: list[KVScaleMetadata | None] = []
    buffers: list[object] = []
    for layer_id, layer_storage in enumerate(layout.layer_storage_dtypes):
        if layer_storage is None:
            key_caches.append(None)
            value_caches.append(None)
            mirror_key_caches.append(None)
            mirror_value_caches.append(None)
            k_scale_caches.append(None)
            v_scale_caches.append(None)
            scale_metadata.append(None)
            continue
        key_view = plane_views[f"layer{layer_id}.key_payload"]
        value_view = plane_views[f"layer{layer_id}.value_payload"]
        key_caches.append(key_view)
        value_caches.append(value_view)
        buffers.append(key_view)
        buffers.append(value_view)
        if layer_id in mirror_layers:
            mirror_key = plane_views[f"layer{layer_id}.bf16_mirror_key"]
            mirror_value = plane_views[f"layer{layer_id}.bf16_mirror_value"]
            mirror_key_caches.append(mirror_key)
            mirror_value_caches.append(mirror_value)
            buffers.append(mirror_key)
            buffers.append(mirror_value)
        else:
            mirror_key_caches.append(None)
            mirror_value_caches.append(None)
        if layer_storage == DType.INT8_PER_TOKEN_HEAD:
            k_scale = plane_views[f"layer{layer_id}.key_scale"]
            v_scale = plane_views[f"layer{layer_id}.value_scale"]
            k_scale_caches.append(k_scale)
            v_scale_caches.append(v_scale)
            buffers.append(k_scale)
            buffers.append(v_scale)
            scale_metadata.append(
                KVScaleMetadata(
                    k_scale=Tensor.from_handle(
                        k_scale.ptr, scale_shape, layout.scale_dtype, device
                    ),
                    v_scale=Tensor.from_handle(
                        v_scale.ptr, scale_shape, layout.scale_dtype, device
                    ),
                    scale_dtype=layout.scale_dtype,
                    granularity=layout.scale_granularity,
                )
            )
        else:
            k_scale_caches.append(None)
            v_scale_caches.append(None)
            scale_metadata.append(None)
    return Qwen35GGUFKVChunkBacking(
        layout=layout,
        start_block_id=int(start_block_id),
        pages=page_count,
        full_key_caches=tuple(key_caches),
        full_value_caches=tuple(value_caches),
        full_bf16_mirror_key_caches=tuple(mirror_key_caches),
        full_bf16_mirror_value_caches=tuple(mirror_value_caches),
        full_k_scale_caches=tuple(k_scale_caches),
        full_v_scale_caches=tuple(v_scale_caches),
        full_kv_scale_metadata=tuple(scale_metadata),
        buffers=tuple(buffers),
    )


def _allocate_qwen35_gguf_kv_chunk_vmm(
    cfg: object,
    *,
    runtime: HipRuntime,
    layout: Qwen35GGUFKVChunkLayout,
    initial_pages: int,
    max_pages: int,
    start_block_id: int = 0,
) -> tuple[
    Qwen35GGUFKVChunkBacking,
    dict[str, VirtualMemoryBuffer],
    dict[str, int],
]:
    """Reserve one stable VMM arena per plane and commit only ``initial_pages``.

    Returns the initial backing, the per-role arena owner, and each plane's
    per-page byte size. Every plane's virtual address range is reserved for
    ``max_pages`` so later growth maps physical memory into the same base and no
    existing page pointer moves. Only the initial committed granules are
    physical device memory; the reservation is not.
    """

    if len(layout.layer_storage_dtypes) != len(cfg.layer_types):
        raise ValueError("GGUF KV chunk layout does not match the model layer count")
    initial = int(initial_pages)
    maximum = int(max_pages)
    if initial <= 0 or maximum < initial:
        raise ValueError("VMM KV arena requires 0 < initial_pages <= max_pages")
    specs = _qwen35_gguf_kv_plane_specs(cfg, layout)
    plane_owners: dict[str, VirtualMemoryBuffer] = {}
    plane_page_nbytes: dict[str, int] = {}
    plane_views: dict[str, VmmPlaneBuffer] = {}
    owners: list[VirtualMemoryBuffer] = []
    try:
        for role, page_nbytes in specs:
            owner = VirtualMemoryBuffer.reserve(
                int(page_nbytes) * maximum,
                runtime=runtime,
            )
            owners.append(owner)
            owner.commit_to(int(page_nbytes) * initial)
            plane_owners[role] = owner
            plane_page_nbytes[role] = int(page_nbytes)
            plane_views[role] = VmmPlaneBuffer(
                owner, offset=0, nbytes=int(page_nbytes) * initial
            )
        backing = _build_qwen35_gguf_kv_backing_from_plane_views(
            cfg=cfg,
            layout=layout,
            pages=initial,
            start_block_id=start_block_id,
            plane_views=plane_views,
        )
        expected_page_bytes = _qwen35_gguf_kv_page_bytes(cfg, layout)
        if backing.page_bytes != expected_page_bytes:
            raise RuntimeError("GGUF VMM KV arena byte accounting drift")
    except BaseException:
        for owner in reversed(owners):
            try:
                owner.close()
            except Exception:  # pragma: no cover - defensive
                pass
        raise
    return backing, plane_owners, plane_page_nbytes


def _allocate_qwen35_gguf_kv_chunk(
    cfg: object,
    *,
    runtime: HipRuntime,
    start_block_id: int,
    pages: int,
    layout: Qwen35GGUFKVChunkLayout,
) -> Qwen35GGUFKVChunkBacking:
    if len(layout.layer_storage_dtypes) != len(cfg.layer_types):
        raise ValueError("GGUF KV chunk layout does not match the model layer count")
    payload_elements = int(pages) * 256 * int(cfg.head_count_kv) * int(cfg.key_length)
    scale_shape = _qwen35_gguf_kv_scale_shape(
        cfg,
        pages=int(pages),
        granularity=layout.scale_granularity,
    )
    scale_nbytes = int(np.prod(scale_shape)) * layout.scale_dtype.itemsize
    mirror_layers = frozenset(layout.bf16_mirror_layer_indices)
    device = Device("hip", 0)
    key_caches: list[DeviceBuffer | None] = []
    value_caches: list[DeviceBuffer | None] = []
    mirror_key_caches: list[DeviceBuffer | None] = []
    mirror_value_caches: list[DeviceBuffer | None] = []
    k_scale_caches: list[DeviceBuffer | None] = []
    v_scale_caches: list[DeviceBuffer | None] = []
    scale_metadata: list[KVScaleMetadata | None] = []
    buffers: list[DeviceBuffer] = []
    try:
        for layer_id, (layer_type, layer_storage) in enumerate(
            zip(cfg.layer_types, layout.layer_storage_dtypes, strict=True)
        ):
            if layer_type == LINEAR_ATTENTION:
                if layer_storage is not None:
                    raise ValueError("GGUF KV chunk layout assigns storage to a linear-attention layer")
                key_caches.append(None)
                value_caches.append(None)
                mirror_key_caches.append(None)
                mirror_value_caches.append(None)
                k_scale_caches.append(None)
                v_scale_caches.append(None)
                scale_metadata.append(None)
                continue
            if layer_storage is None:
                raise ValueError("GGUF KV chunk layout omits a full-attention layer")
            key_dtype = DType.INT8 if layer_storage == DType.INT8_PER_TOKEN_HEAD else DType.BF16
            value_dtype = (
                DType.BF16
                if layer_storage == DType.INT8_PER_TOKEN_HEAD and layout.int8_kv_value_bf16
                else key_dtype
            )
            key_cache = malloc(payload_elements * key_dtype.itemsize, runtime=runtime)
            buffers.append(key_cache)
            value_cache = malloc(payload_elements * value_dtype.itemsize, runtime=runtime)
            buffers.append(value_cache)
            key_caches.append(key_cache)
            value_caches.append(value_cache)
            if layer_id in mirror_layers:
                mirror_nbytes = payload_elements * DType.BF16.itemsize
                mirror_key = malloc(mirror_nbytes, runtime=runtime)
                buffers.append(mirror_key)
                mirror_value = malloc(mirror_nbytes, runtime=runtime)
                buffers.append(mirror_value)
                mirror_key_caches.append(mirror_key)
                mirror_value_caches.append(mirror_value)
            else:
                mirror_key_caches.append(None)
                mirror_value_caches.append(None)
            if layer_storage == DType.INT8_PER_TOKEN_HEAD:
                k_scale = malloc(scale_nbytes, runtime=runtime)
                buffers.append(k_scale)
                v_scale = malloc(scale_nbytes, runtime=runtime)
                buffers.append(v_scale)
                k_scale_caches.append(k_scale)
                v_scale_caches.append(v_scale)
                scale_metadata.append(
                    KVScaleMetadata(
                        k_scale=Tensor.from_handle(k_scale.ptr, scale_shape, layout.scale_dtype, device),
                        v_scale=Tensor.from_handle(v_scale.ptr, scale_shape, layout.scale_dtype, device),
                        scale_dtype=layout.scale_dtype,
                        granularity=layout.scale_granularity,
                    )
                )
            else:
                k_scale_caches.append(None)
                v_scale_caches.append(None)
                scale_metadata.append(None)
    except Exception:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        raise
    try:
        backing = Qwen35GGUFKVChunkBacking(
            layout=layout,
            start_block_id=int(start_block_id),
            pages=int(pages),
            full_key_caches=tuple(key_caches),
            full_value_caches=tuple(value_caches),
            full_bf16_mirror_key_caches=tuple(mirror_key_caches),
            full_bf16_mirror_value_caches=tuple(mirror_value_caches),
            full_k_scale_caches=tuple(k_scale_caches),
            full_v_scale_caches=tuple(v_scale_caches),
            full_kv_scale_metadata=tuple(scale_metadata),
            buffers=tuple(buffers),
        )
        expected_page_bytes = _qwen35_gguf_kv_page_bytes(cfg, layout)
        if backing.page_bytes != expected_page_bytes:
            raise RuntimeError("GGUF KV chunk byte accounting drift")
    except Exception:
        for buffer in reversed(buffers):
            free(buffer, runtime=runtime)
        raise
    return backing


def _free_qwen35_gguf_kv_chunk(
    backing: Qwen35GGUFKVChunkBacking,
    *,
    runtime: HipRuntime,
) -> None:
    for buffer in reversed(backing.buffers):
        _free_gguf_kv_plane_buffer(buffer, runtime=runtime)
