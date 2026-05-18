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
    "append_materialized_kv_reference",
    "full_context_kv_reference",
    "plan_dflash_draft_kv_append",
]
