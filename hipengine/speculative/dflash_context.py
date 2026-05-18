"""DFlash draft context K/V cache ownership and append planning.

The DFlash drafter sees target-hidden context rows through a fixed projection
(`fc + hidden_norm`) and, for every drafter layer, through layer-local K/V
projections.  This module owns the torch-free cache ABI for those per-layer
context K/V rows and a deterministic NumPy reference used to prove append-only
materialization is equivalent to rebuilding the full context prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from hipengine.loading.dflash import DFlashDrafterDeviceWeights

from hipengine.core.device import Device
from hipengine.core.dtype import DType, dtype_itemsize
from hipengine.core.tensor import Tensor


class TensorWorkspace(Protocol):
    device: Device

    def reserve_tensor(self, name: str, shape: Sequence[int], dtype: str | DType) -> Tensor:
        ...


@dataclass(frozen=True, slots=True)
class DFlashDraftKVCacheSpec:
    """Fixed draft-context K/V cache bucket for one drafter graph shape."""

    backend: str
    bucket: str
    device: Device
    layer_count: int
    capacity_tokens: int
    num_kv_heads: int
    head_dim: int
    key_dtype: DType | str = DType.FP32
    value_dtype: DType | str = DType.BF16
    metadata_dtype: DType | str = DType.INT32

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if not self.bucket:
            raise ValueError("bucket must be non-empty")
        for name, value in (
            ("layer_count", self.layer_count),
            ("capacity_tokens", self.capacity_tokens),
            ("num_kv_heads", self.num_kv_heads),
            ("head_dim", self.head_dim),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        object.__setattr__(self, "key_dtype", DType.parse(self.key_dtype))
        object.__setattr__(self, "value_dtype", DType.parse(self.value_dtype))
        metadata_dtype = DType.parse(self.metadata_dtype)
        if metadata_dtype not in {DType.INT32, DType.INT64}:
            raise ValueError("metadata_dtype must be int32 or int64")
        object.__setattr__(self, "metadata_dtype", metadata_dtype)

    @property
    def kv_shape(self) -> tuple[int, int, int, int]:
        return (self.layer_count, self.capacity_tokens, self.num_kv_heads, self.head_dim)

    @property
    def position_shape(self) -> tuple[int]:
        return (self.capacity_tokens,)

    @property
    def live_shape(self) -> tuple[int]:
        return (1,)

    @property
    def key_bytes(self) -> int:
        return _nbytes(self.kv_shape, self.key_dtype)

    @property
    def value_bytes(self) -> int:
        return _nbytes(self.kv_shape, self.value_dtype)

    @property
    def metadata_bytes(self) -> int:
        return _nbytes(self.position_shape, self.metadata_dtype) + _nbytes(self.live_shape, self.metadata_dtype)

    @property
    def total_bytes(self) -> int:
        return self.key_bytes + self.value_bytes + self.metadata_bytes


@dataclass(frozen=True, slots=True)
class DFlashDraftKVCacheOwner:
    """Stable Tensor handles for one DFlash draft context K/V cache."""

    spec: DFlashDraftKVCacheSpec
    keys: Tensor
    values: Tensor
    positions: Tensor
    live_count: Tensor

    def __post_init__(self) -> None:
        self._validate("keys", self.keys, self.spec.kv_shape, self.spec.key_dtype)
        self._validate("values", self.values, self.spec.kv_shape, self.spec.value_dtype)
        self._validate("positions", self.positions, self.spec.position_shape, self.spec.metadata_dtype)
        self._validate("live_count", self.live_count, self.spec.live_shape, self.spec.metadata_dtype)

    @classmethod
    def allocate(
        cls,
        spec: DFlashDraftKVCacheSpec,
        *,
        workspace: TensorWorkspace,
        prefix: str | None = None,
    ) -> "DFlashDraftKVCacheOwner":
        if workspace.device != spec.device:
            raise ValueError("workspace device must match DFlash draft KV cache spec")
        base = prefix or f"dflash_draft_kv/{spec.backend}/{spec.device}/{spec.bucket}"

        def reserve(name: str, shape: Sequence[int], dtype: DType | str) -> Tensor:
            return workspace.reserve_tensor(f"{base}/{name}", shape, dtype)

        return cls(
            spec=spec,
            keys=reserve("keys", spec.kv_shape, spec.key_dtype),
            values=reserve("values", spec.kv_shape, spec.value_dtype),
            positions=reserve("positions", spec.position_shape, spec.metadata_dtype),
            live_count=reserve("live_count", spec.live_shape, spec.metadata_dtype),
        )

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "backend": self.spec.backend,
            "bucket": self.spec.bucket,
            "device": str(self.spec.device),
            "layer_count": self.spec.layer_count,
            "capacity_tokens": self.spec.capacity_tokens,
            "num_kv_heads": self.spec.num_kv_heads,
            "head_dim": self.spec.head_dim,
            "key_dtype": self.spec.key_dtype.value,
            "value_dtype": self.spec.value_dtype.value,
            "key_bytes": self.spec.key_bytes,
            "value_bytes": self.spec.value_bytes,
            "metadata_bytes": self.spec.metadata_bytes,
            "total_bytes": self.spec.total_bytes,
            "phases": ("full_context_rebuild", "append_materialize", "query_only_drafter"),
        }

    def _validate(self, name: str, tensor: Tensor, shape: tuple[int, ...], dtype: DType) -> None:
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tensor.shape}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype.value}, got {tensor.dtype.value}")
        if tensor.device != self.spec.device:
            raise ValueError(f"{name} must live on {self.spec.device}, got {tensor.device}")


@dataclass(frozen=True, slots=True)
class DFlashDraftKVMaterializerScratch:
    """Scratch tensors used by append-only draft K/V materialization."""

    projected_hidden: Tensor
    key_raw: Tensor

    def validate(self, *, max_rows: int, hidden_size: int, kv_features: int, device: Device) -> None:
        if self.projected_hidden.ndim != 2 or self.projected_hidden.shape[0] < max_rows or self.projected_hidden.shape[1] != hidden_size:
            raise ValueError("projected_hidden scratch must have shape (>=append_count, hidden_size)")
        if self.projected_hidden.dtype != DType.BF16 or self.projected_hidden.device != device:
            raise ValueError("projected_hidden scratch must be BF16 on the draft KV device")
        if self.key_raw.ndim != 2 or self.key_raw.shape[0] < max_rows or self.key_raw.shape[1] != kv_features:
            raise ValueError("key_raw scratch must have shape (>=append_count, num_kv_heads * head_dim)")
        if self.key_raw.dtype != DType.FP32 or self.key_raw.device != device:
            raise ValueError("key_raw scratch must be FP32 on the draft KV device")


@dataclass(frozen=True, slots=True)
class DFlashDraftKVLayerWeights:
    """Layer-local DFlash K/V projection weights."""

    k_proj: Tensor
    v_proj: Tensor
    k_norm: Tensor

    def validate(self, *, hidden_size: int, kv_features: int, head_dim: int, device: Device) -> None:
        if self.k_proj.shape != (kv_features, hidden_size):
            raise ValueError("k_proj weight shape must be (num_kv_heads * head_dim, hidden_size)")
        if self.v_proj.shape != (kv_features, hidden_size):
            raise ValueError("v_proj weight shape must be (num_kv_heads * head_dim, hidden_size)")
        if self.k_norm.shape != (head_dim,):
            raise ValueError("k_norm weight shape must be (head_dim,)")
        for name, tensor in (("k_proj", self.k_proj), ("v_proj", self.v_proj), ("k_norm", self.k_norm)):
            if tensor.dtype != DType.BF16:
                raise ValueError(f"{name} must use BF16 storage")
            if tensor.device != device:
                raise ValueError(f"{name} must live on {device}")


@dataclass(frozen=True, slots=True)
class DFlashDraftKVMaterializeResult:
    append_start: int
    append_count: int
    live_count: int
    draft_kv_bytes: int
    key_bytes: int
    value_bytes: int
    capacity_tokens: int
    phases: tuple[str, ...]

    def as_metadata(self) -> dict[str, object]:
        return {
            "append_start": self.append_start,
            "append_count": self.append_count,
            "live_count": self.live_count,
            "draft_kv_bytes": self.draft_kv_bytes,
            "key_bytes": self.key_bytes,
            "value_bytes": self.value_bytes,
            "capacity_tokens": self.capacity_tokens,
            "phases": self.phases,
        }


@dataclass(frozen=True, slots=True)
class DFlashDraftKVAppendPlan:
    """Append-only materialization plan for newly committed target-hidden rows."""

    start: int
    count: int
    positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.count < 0:
            raise ValueError("count must be non-negative")
        positions = tuple(int(pos) for pos in self.positions)
        if len(positions) != self.count:
            raise ValueError("positions length must match count")
        object.__setattr__(self, "positions", positions)

    @property
    def end(self) -> int:
        return self.start + self.count

    def validate_capacity(self, capacity_tokens: int) -> None:
        if self.end > capacity_tokens:
            raise ValueError(f"append end {self.end} exceeds draft KV capacity {capacity_tokens}")


def plan_dflash_draft_kv_append(*, live_count: int, new_positions: Sequence[int], capacity_tokens: int) -> DFlashDraftKVAppendPlan:
    positions = tuple(int(pos) for pos in new_positions)
    plan = DFlashDraftKVAppendPlan(start=int(live_count), count=len(positions), positions=positions)
    plan.validate_capacity(int(capacity_tokens))
    return plan


def dflash_layer_kv_weights(weights: DFlashDrafterDeviceWeights, layer: int) -> DFlashDraftKVLayerWeights:
    prefix = f"layers.{int(layer)}.self_attn"
    return DFlashDraftKVLayerWeights(
        k_proj=weights.tensor(f"{prefix}.k_proj.weight"),
        v_proj=weights.tensor(f"{prefix}.v_proj.weight"),
        k_norm=weights.tensor(f"{prefix}.k_norm.weight"),
    )


def materialize_dflash_draft_kv_append_from_projected(
    *,
    owner: DFlashDraftKVCacheOwner,
    plan: DFlashDraftKVAppendPlan,
    projected_hidden: Tensor,
    positions: Tensor,
    layer_weights: Sequence[DFlashDraftKVLayerWeights],
    scratch: DFlashDraftKVMaterializerScratch,
    cos_table: Tensor,
    sin_table: Tensor,
    stream: int = 0,
    library=None,
    runtime=None,
    threads: int = 128,
) -> DFlashDraftKVMaterializeResult:
    """Append newly projected DFlash context rows into fixed draft K/V buffers."""

    plan.validate_capacity(owner.spec.capacity_tokens)
    _validate_materializer_inputs(owner, plan, projected_hidden, positions, layer_weights, scratch, cos_table, sin_table)
    if plan.count == 0:
        return _materialize_result(owner, plan)
    from hipengine.kernels.hip_gfx1100.speculative.dflash_drafter import (
        dflash_dense_bf16_to_bf16,
        dflash_dense_bf16_to_f32,
        dflash_key_rmsnorm_rotary_f32,
        dflash_update_kv_metadata_i32,
    )

    kv_features = owner.spec.num_kv_heads * owner.spec.head_dim
    key_item = dtype_itemsize(owner.spec.key_dtype)
    value_item = dtype_itemsize(owner.spec.value_dtype)
    for layer, weights in enumerate(layer_weights):
        key_dst_ptr = owner.keys.ptr + ((layer * owner.spec.capacity_tokens + plan.start) * kv_features * key_item)
        value_dst_ptr = owner.values.ptr + ((layer * owner.spec.capacity_tokens + plan.start) * kv_features * value_item)
        dflash_dense_bf16_to_f32(
            projected_hidden.ptr,
            weights.k_proj.ptr,
            scratch.key_raw.ptr,
            plan.count,
            projected_hidden.shape[1],
            kv_features,
            threads=threads,
            stream=stream,
            library=library,
            runtime=runtime,
        )
        dflash_key_rmsnorm_rotary_f32(
            scratch.key_raw.ptr,
            weights.k_norm.ptr,
            cos_table.ptr,
            sin_table.ptr,
            positions.ptr,
            key_dst_ptr,
            plan.count,
            owner.spec.num_kv_heads,
            owner.spec.head_dim,
            cos_table.shape[1],
            cos_table.shape[0],
            threads=threads,
            stream=stream,
            library=library,
            runtime=runtime,
        )
        dflash_dense_bf16_to_bf16(
            projected_hidden.ptr,
            weights.v_proj.ptr,
            value_dst_ptr,
            plan.count,
            projected_hidden.shape[1],
            kv_features,
            threads=threads,
            stream=stream,
            library=library,
            runtime=runtime,
        )
    dflash_update_kv_metadata_i32(
        positions.ptr,
        owner.positions.ptr,
        owner.live_count.ptr,
        start=plan.start,
        count=plan.count,
        end=plan.end,
        threads=256,
        stream=stream,
        library=library,
        runtime=runtime,
    )
    return _materialize_result(owner, plan)


def materialize_dflash_draft_kv_append(
    *,
    owner: DFlashDraftKVCacheOwner,
    plan: DFlashDraftKVAppendPlan,
    target_hidden_concat: Tensor,
    positions: Tensor,
    weights: DFlashDrafterDeviceWeights,
    scratch: DFlashDraftKVMaterializerScratch,
    cos_table: Tensor,
    sin_table: Tensor,
    stream: int = 0,
    library=None,
    runtime=None,
    threads: int = 128,
) -> DFlashDraftKVMaterializeResult:
    """Project newly committed target-hidden rows, then append draft K/V rows."""

    from hipengine.speculative.dflash_drafter import project_dflash_target_hidden_bf16

    if target_hidden_concat.shape[0] < plan.count:
        raise ValueError("target_hidden_concat rows must cover append count")
    if scratch.projected_hidden.shape[0] < plan.count:
        raise ValueError("projected_hidden scratch rows must cover append count")
    projected_view = Tensor.from_handle(
        scratch.projected_hidden.ptr,
        (plan.count, weights.config.hidden_size),
        scratch.projected_hidden.dtype,
        scratch.projected_hidden.device,
    )
    target_view = Tensor.from_handle(
        target_hidden_concat.ptr,
        (plan.count, weights.config.target_hidden_concat_size),
        target_hidden_concat.dtype,
        target_hidden_concat.device,
    )
    project_dflash_target_hidden_bf16(
        target_view,
        projected_view,
        projected_view,
        weights,
        stream=stream,
        libraries={"norm": library} if library is not None else None,
        threads=threads,
    )
    layer_weights = [dflash_layer_kv_weights(weights, layer) for layer in range(owner.spec.layer_count)]
    return materialize_dflash_draft_kv_append_from_projected(
        owner=owner,
        plan=plan,
        projected_hidden=projected_view,
        positions=positions,
        layer_weights=layer_weights,
        scratch=scratch,
        cos_table=cos_table,
        sin_table=sin_table,
        stream=stream,
        library=library,
        runtime=runtime,
        threads=threads,
    )


def _validate_materializer_inputs(
    owner: DFlashDraftKVCacheOwner,
    plan: DFlashDraftKVAppendPlan,
    projected_hidden: Tensor,
    positions: Tensor,
    layer_weights: Sequence[DFlashDraftKVLayerWeights],
    scratch: DFlashDraftKVMaterializerScratch,
    cos_table: Tensor,
    sin_table: Tensor,
) -> None:
    if projected_hidden.ndim != 2 or projected_hidden.shape[0] < plan.count:
        raise ValueError("projected_hidden must have at least append_count rows")
    hidden_size = projected_hidden.shape[1]
    kv_features = owner.spec.num_kv_heads * owner.spec.head_dim
    if projected_hidden.dtype != DType.BF16 or projected_hidden.device != owner.spec.device:
        raise ValueError("projected_hidden must be BF16 on the draft KV device")
    scratch.validate(max_rows=plan.count, hidden_size=hidden_size, kv_features=kv_features, device=owner.spec.device)
    if len(layer_weights) != owner.spec.layer_count:
        raise ValueError("layer_weights length must match draft KV layer_count")
    for weights in layer_weights:
        weights.validate(hidden_size=hidden_size, kv_features=kv_features, head_dim=owner.spec.head_dim, device=owner.spec.device)
    if positions.shape != (plan.count,) or positions.dtype != owner.spec.metadata_dtype or positions.device != owner.spec.device:
        raise ValueError("positions tensor must have append_count rows on the draft KV device")
    if cos_table.ndim != 2 or sin_table.shape != cos_table.shape:
        raise ValueError("cos/sin tables must be matching rank-2 tensors")
    if cos_table.dtype != DType.FP32 or sin_table.dtype != DType.FP32:
        raise ValueError("cos/sin tables must use FP32 storage")
    if cos_table.shape[1] <= 0 or cos_table.shape[1] > owner.spec.head_dim or cos_table.shape[1] % 2:
        raise ValueError("rotary dimension must be even and no larger than head_dim")
    if cos_table.device != owner.spec.device or sin_table.device != owner.spec.device:
        raise ValueError("cos/sin tables must live on the draft KV device")


def _materialize_result(owner: DFlashDraftKVCacheOwner, plan: DFlashDraftKVAppendPlan) -> DFlashDraftKVMaterializeResult:
    return DFlashDraftKVMaterializeResult(
        append_start=plan.start,
        append_count=plan.count,
        live_count=plan.end,
        draft_kv_bytes=owner.spec.total_bytes,
        key_bytes=owner.spec.key_bytes,
        value_bytes=owner.spec.value_bytes,
        capacity_tokens=owner.spec.capacity_tokens,
        phases=("full_context_rebuild", "append_materialize", "query_only_drafter"),
    )


def append_materialized_kv_reference(existing_keys, existing_values, new_keys, new_values, *, start: int):
    """Append ``new_*`` rows into copies of existing NumPy K/V arrays.

    Arrays are shaped ``[layers, capacity, kv_heads, head_dim]`` for existing
    cache and ``[layers, new_rows, kv_heads, head_dim]`` for materialized rows.
    The function is intentionally NumPy-only and used by tests to prove that the
    append path matches a full-context rebuild prefix while rejected/suffix rows
    remain untouched.
    """

    import numpy as np

    keys = np.array(existing_keys, copy=True)
    values = np.array(existing_values, copy=True)
    new_keys_arr = np.asarray(new_keys)
    new_values_arr = np.asarray(new_values)
    if keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("existing keys/values must share shape [layers, capacity, kv_heads, head_dim]")
    if new_keys_arr.ndim != 4 or new_values_arr.shape != new_keys_arr.shape:
        raise ValueError("new keys/values must share shape [layers, new_rows, kv_heads, head_dim]")
    if new_keys_arr.shape[0] != keys.shape[0] or new_keys_arr.shape[2:] != keys.shape[2:]:
        raise ValueError("new K/V rows must match existing layer/head dimensions")
    end = int(start) + int(new_keys_arr.shape[1])
    if start < 0 or end > keys.shape[1]:
        raise ValueError("append range exceeds existing K/V capacity")
    keys[:, start:end, :, :] = new_keys_arr
    values[:, start:end, :, :] = new_values_arr
    return keys, values


def full_context_kv_reference(materialized_keys, materialized_values, *, capacity_tokens: int):
    """Build a full-context cache prefix from materialized per-layer rows."""

    import numpy as np

    key_rows = np.asarray(materialized_keys)
    value_rows = np.asarray(materialized_values)
    if key_rows.ndim != 4 or value_rows.shape != key_rows.shape:
        raise ValueError("materialized K/V must share shape [layers, rows, kv_heads, head_dim]")
    if key_rows.shape[1] > int(capacity_tokens):
        raise ValueError("materialized rows exceed draft KV capacity")
    keys = np.zeros((key_rows.shape[0], int(capacity_tokens), key_rows.shape[2], key_rows.shape[3]), dtype=key_rows.dtype)
    values = np.zeros_like(keys, dtype=value_rows.dtype)
    keys[:, : key_rows.shape[1], :, :] = key_rows
    values[:, : value_rows.shape[1], :, :] = value_rows
    return keys, values


def _nbytes(shape: tuple[int, ...], dtype: DType) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * dtype_itemsize(dtype)


__all__ = [
    "DFlashDraftKVAppendPlan",
    "DFlashDraftKVCacheOwner",
    "DFlashDraftKVCacheSpec",
    "DFlashDraftKVLayerWeights",
    "DFlashDraftKVMaterializeResult",
    "DFlashDraftKVMaterializerScratch",
    "append_materialized_kv_reference",
    "dflash_layer_kv_weights",
    "full_context_kv_reference",
    "materialize_dflash_draft_kv_append",
    "materialize_dflash_draft_kv_append_from_projected",
    "plan_dflash_draft_kv_append",
]
