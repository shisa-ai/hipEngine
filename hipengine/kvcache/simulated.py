"""Deterministic fake KV backends for Generation-2 host conformance tests."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.kvcache.backend import (
    ClaimLifetime,
    KVBackendSpec,
    KVBatchView,
    KVLease,
    KVPlaneView,
    KVPoolPlan,
    KVPoolSpec,
    KVStorageView,
    ResourceClaimSet,
    ResourceDelta,
)
from hipengine.kvcache.spans import KVLiveSpans, KVScaleMetadata

FAKE_KV_BACKEND_KINDS = (
    "dense_bf16",
    "dense_int8",
    "mixed_bf16_packed",
    "dms_variable",
)

_RECENT_WINDOW = 4
_DMS_HEAD_WINDOWS = (2, 3, 5, 7)
_CPU = Device("cpu", 0)


@dataclass(frozen=True, slots=True)
class SimulatedKVOperation:
    operation_id: str
    lease: KVLease
    current_tokens: int
    next_tokens: int

    def __post_init__(self) -> None:
        if self.current_tokens < 0 or self.next_tokens < 0:
            raise ValueError("simulated KV token counts must be non-negative")
        if self.next_tokens < self.current_tokens:
            raise ValueError("simulated commit cannot reduce logical token count")


class SimulatedKVBackend:
    """One protocol implementation parameterized only by a resolved fake spec.

    All four fake compositions expose identical logical token behavior.  Their
    pool sets, ownership vectors, liveness shape, and storage planes differ.
    """

    def __init__(self, kind: str, *, capacity_tokens: int) -> None:
        if kind not in FAKE_KV_BACKEND_KINDS:
            raise ValueError(f"unknown fake KV backend {kind!r}")
        capacity = int(capacity_tokens)
        if capacity <= 0:
            raise ValueError("capacity_tokens must be positive")
        self.kind = kind
        self.capacity_tokens = capacity
        self.spec = _backend_spec(kind)
        self._plan = _pool_plan(self.spec, capacity)
        self._storage_view = _storage_view(self.spec, self._plan)

    def plan_pools(self, load_plan: Any) -> KVPoolPlan:
        del load_plan
        return self._plan

    def estimate(self, request: Any, prefix: Any, stage: Any) -> ResourceClaimSet:
        del prefix
        request_id = int(getattr(request, "request_id"))
        stage_map: Mapping[str, Any] = stage if isinstance(stage, Mapping) else {}
        stage_kind = str(stage_map.get("kind", "admission"))
        if stage_kind == "work_item":
            current = int(stage_map.get("current_tokens", 0))
            following = int(stage_map.get("next_tokens", current))
            workspace: dict[str, int] = {}
            if self.kind == "mixed_bf16_packed" and following > _RECENT_WINDOW:
                workspace["kv.demotion_workspace"] = 1
            elif self.kind == "dms_variable" and following > min(_DMS_HEAD_WINDOWS):
                workspace["kv.compaction_workspace"] = 1
            return ResourceClaimSet.from_mapping(
                f"work:{request_id}:{current}:{following}",
                workspace,
                request_id=request_id,
                lifetime=ClaimLifetime.WORK_ITEM,
            )
        tokens = int(stage_map.get("tokens", len(getattr(request, "prompt_tokens", ()))))
        return self._claims_for_tokens(request_id, tokens, claim_id=f"{stage_kind}:{request_id}:{tokens}")

    def reserve(self, claims: ResourceClaimSet) -> KVLease:
        if claims.request_id is None:
            raise ValueError("fake backend reservations require a request_id")
        return KVLease(
            lease_id=f"lease:{claims.request_id}",
            request_id=claims.request_id,
            backend_fingerprint=self.spec.fingerprint,
            generation=self._plan.generation,
            claims=claims.with_claim_id(f"lease:{claims.request_id}:ownership"),
            private_handles=(f"private:{claims.request_id}",),
            writable_tail_handle=f"tail:{claims.request_id}",
            metadata_handles=(f"metadata:{claims.request_id}",),
        )

    def prepare(self, work_item: Any) -> KVBatchView:
        request_ids = tuple(int(request_id) for request_id in getattr(work_item, "request_ids"))
        if not request_ids:
            raise ValueError("fake backend prepare requires request ids")
        context_lengths = tuple(
            int(length)
            for length in getattr(work_item, "context_lengths", (0,) * len(request_ids))
        )
        if len(context_lengths) != len(request_ids):
            raise ValueError("context lengths must align with request ids")
        rows = len(request_ids)
        pointer_base = 0x70000000 + (FAKE_KV_BACKEND_KINDS.index(self.kind) + 1) * 0x01000000
        request_tensor = Tensor.from_handle(pointer_base + 0x1000, (rows,), DType.INT64, _CPU)
        live_count_ptr = pointer_base + 0x2000
        base_offset_ptr = pointer_base + 0x3000
        row_position_ptr = pointer_base + 0x4000
        max_live = max(context_lengths, default=0)
        scale_metadata = None
        storage_dtype = DType.BF16
        if self.kind == "dense_int8":
            storage_dtype = DType.INT8_PER_TOKEN_HEAD
            scale_metadata = KVScaleMetadata(
                k_scale=Tensor.from_handle(pointer_base + 0x5000, (rows, 1), DType.FP16, _CPU),
                v_scale=Tensor.from_handle(pointer_base + 0x6000, (rows, 1), DType.FP16, _CPU),
            )
        if self.kind == "dms_variable":
            live_spans = KVLiveSpans(
                base_offsets=Tensor.from_handle(
                    base_offset_ptr,
                    (rows, 1, len(_DMS_HEAD_WINDOWS)),
                    DType.INT32,
                    _CPU,
                ),
                live_counts=Tensor.from_handle(
                    live_count_ptr,
                    (rows, 1, len(_DMS_HEAD_WINDOWS)),
                    DType.INT32,
                    _CPU,
                ),
                max_live_count=min(max_live, max(_DMS_HEAD_WINDOWS)),
                token_positions=Tensor.from_handle(
                    pointer_base + 0x7000,
                    (rows, max(1, min(max_live, max(_DMS_HEAD_WINDOWS)))),
                    DType.INT32,
                    _CPU,
                ),
                evict_mask=None,
                storage_dtype=DType.BF16,
                spans_mode="per_head_variable",
                request_ids=request_tensor,
                row_positions=Tensor.from_handle(row_position_ptr, (rows,), DType.INT32, _CPU),
                span_role="decode",
            )
        else:
            live_spans = KVLiveSpans.paged_uniform(
                block_table=Tensor.from_handle(base_offset_ptr, (rows, 1), DType.INT32, _CPU),
                live_counts=Tensor.from_handle(live_count_ptr, (rows,), DType.INT32, _CPU),
                max_live_count=max_live,
                storage_dtype=storage_dtype,
                request_ids=request_tensor,
                row_positions=Tensor.from_handle(row_position_ptr, (rows,), DType.INT32, _CPU),
                span_role="decode",
                scale_metadata=scale_metadata,
            )
        return KVBatchView(
            live_spans=live_spans,
            storage_view=self._storage_view,
            kernel_bundle_key=self.spec.kernel_bundle_key,
            execution_compatibility_key=(*self.spec.compatibility_key, "decode"),
        )

    def begin_transaction(self, rows: Sequence[Any], draft: Any) -> SimulatedKVOperation:
        del draft
        if len(rows) != 1:
            raise ValueError("fake transaction helper expects one lease row")
        lease = getattr(rows[0], "lease", rows[0])
        if not isinstance(lease, KVLease):
            raise TypeError("fake transaction row must provide a KVLease")
        return SimulatedKVOperation(
            operation_id=f"transaction:{lease.request_id}",
            lease=lease,
            current_tokens=0,
            next_tokens=0,
        )

    def commit(self, operation: Any, result: Any) -> ResourceDelta:
        del result
        try:
            lease = operation.lease
            current_tokens = int(operation.current_tokens)
            next_tokens = int(operation.next_tokens)
            source_operation_id = str(operation.operation_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError("fake backend commit requires a lease/token operation") from exc
        if not isinstance(lease, KVLease):
            raise TypeError("fake backend commit operation must carry a KVLease")
        if self.kind == "mixed_bf16_packed" and next_tokens > _RECENT_WINDOW:
            operation_kind = "demote"
        elif self.kind == "dms_variable" and next_tokens > min(_DMS_HEAD_WINDOWS):
            operation_kind = "compact"
        else:
            operation_kind = "grow"
        operation_id = f"{operation_kind}:{source_operation_id}"
        before = self._claims_for_tokens(
            lease.request_id,
            current_tokens,
            claim_id=f"before:{operation_id}",
        )
        after = self._claims_for_tokens(
            lease.request_id,
            next_tokens,
            claim_id=f"after:{operation_id}",
        )
        return ResourceDelta.between(
            operation_id=operation_id,
            lease_id=lease.lease_id,
            request_id=lease.request_id,
            before=before,
            after=after,
        )

    def rollback(self, operation: Any) -> ResourceDelta:
        try:
            lease = operation.lease
            source_operation_id = str(operation.operation_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError("fake backend rollback requires a lease operation") from exc
        if not isinstance(lease, KVLease):
            raise TypeError("fake backend rollback operation must carry a KVLease")
        return ResourceDelta(
            operation_id=f"rollback:{source_operation_id}",
            lease_id=lease.lease_id,
            request_id=lease.request_id,
        )

    def reclaim(self, lease: KVLease) -> ResourceDelta:
        empty = ResourceClaimSet(
            claim_id=f"reclaimed:{lease.lease_id}",
            request_id=lease.request_id,
        )
        return ResourceDelta.between(
            operation_id=f"reclaim:{lease.lease_id}",
            lease_id=lease.lease_id,
            request_id=lease.request_id,
            before=lease.claims,
            after=empty,
        )

    def prefix_lookup(self, tokens: Sequence[int]) -> Any:
        return SimpleNamespace(hit=False, matched_tokens=(), remaining_tokens=tuple(int(token) for token in tokens))

    def maintenance(self, budget: Any) -> list[Any]:
        del budget
        return []

    def _claims_for_tokens(self, request_id: int, tokens: int, *, claim_id: str) -> ResourceClaimSet:
        count = int(tokens)
        if count < 0:
            raise ValueError("logical token count must be non-negative")
        units: dict[str, int]
        if self.kind == "dense_bf16":
            units = {
                "kv.k_payload": count,
                "kv.v_payload": count,
                "kv.row_metadata": 1,
            }
        elif self.kind == "dense_int8":
            units = {
                "kv.k_int8": count,
                "kv.v_int8": count,
                "kv.k_scale": ceil(count / 2),
                "kv.v_scale": ceil(count / 2),
                "kv.row_metadata": 1,
            }
        elif self.kind == "mixed_bf16_packed":
            recent = min(count, _RECENT_WINDOW)
            history = max(0, count - _RECENT_WINDOW)
            units = {
                "kv.k_recent_bf16": recent,
                "kv.v_recent_bf16": recent,
                "kv.k_history_packed": ceil(history / 2),
                "kv.v_history_packed": ceil(history / 2),
                "kv.history_scale": ceil(history / 4),
                "kv.row_metadata": 1,
            }
        else:
            live_cells = sum(min(count, window) for window in _DMS_HEAD_WINDOWS)
            units = {
                "kv.k_live_cells": live_cells,
                "kv.v_live_cells": live_cells,
                "kv.span_metadata": len(_DMS_HEAD_WINDOWS) + min(count, max(_DMS_HEAD_WINDOWS)),
                "kv.row_metadata": 1,
            }
        return ResourceClaimSet.from_mapping(
            claim_id,
            units,
            request_id=request_id,
            lifetime=ClaimLifetime.LEASE,
        )


def create_fake_kv_backend(kind: str, *, capacity_tokens: int = 65_536) -> SimulatedKVBackend:
    return SimulatedKVBackend(kind, capacity_tokens=capacity_tokens)


def _backend_spec(kind: str) -> KVBackendSpec:
    settings = {
        "dense_bf16": ("paged_dense", "bf16", "immutable_pages", "fake_dense_bf16"),
        "dense_int8": ("paged_dense", "int8_per_token_head", "immutable_pages", "fake_dense_int8"),
        "mixed_bf16_packed": ("paged_dense", "bf16_packed_history", "snapshot_overlay", "fake_mixed"),
        "dms_variable": ("dms_compact", "bf16", "unsupported", "fake_dms_bf16"),
    }
    topology, codec, prefix_mode, kernels = settings[kind]
    return KVBackendSpec(
        topology_key=topology,
        hot_codec_key=codec,
        tier_key="device_only",
        layout_fingerprint=f"fake-layout:{kind}:v1",
        artifact_fingerprint=f"fake-artifact:{kind}:v1",
        prefix_mode=prefix_mode,
        transaction_mode="journal",
        kernel_bundle_key=kernels,
        physical_widths=(1, 2, 4, 8),
    )


def _pool_plan(spec: KVBackendSpec, capacity: int) -> KVPoolPlan:
    lease_only = (ClaimLifetime.LEASE, ClaimLifetime.CACHE)
    workspace_only = (ClaimLifetime.WORK_ITEM, ClaimLifetime.TRANSACTION)
    if spec.hot_codec_key == "int8_per_token_head":
        pools = (
            KVPoolSpec("kv.k_int8", capacity, unit="cells", plane_role="k_payload", lifetimes=lease_only),
            KVPoolSpec("kv.v_int8", capacity, unit="cells", plane_role="v_payload", lifetimes=lease_only),
            KVPoolSpec("kv.k_scale", capacity, unit="scales", plane_role="k_scale", lifetimes=lease_only),
            KVPoolSpec("kv.v_scale", capacity, unit="scales", plane_role="v_scale", lifetimes=lease_only),
            KVPoolSpec("kv.row_metadata", capacity, unit="rows", plane_role="row_metadata", lifetimes=lease_only),
        )
    elif spec.hot_codec_key == "bf16_packed_history":
        pools = (
            KVPoolSpec("kv.k_recent_bf16", capacity, unit="cells", plane_role="k_recent", lifetimes=lease_only),
            KVPoolSpec("kv.v_recent_bf16", capacity, unit="cells", plane_role="v_recent", lifetimes=lease_only),
            KVPoolSpec("kv.k_history_packed", capacity, unit="packed_cells", plane_role="k_history", lifetimes=lease_only),
            KVPoolSpec("kv.v_history_packed", capacity, unit="packed_cells", plane_role="v_history", lifetimes=lease_only),
            KVPoolSpec("kv.history_scale", capacity, unit="scales", plane_role="history_scale", lifetimes=lease_only),
            KVPoolSpec("kv.row_metadata", capacity, unit="rows", plane_role="row_metadata", lifetimes=lease_only),
            KVPoolSpec("kv.demotion_workspace", max(1, capacity // 8), unit="rows", plane_role="demotion_workspace", lifetimes=workspace_only),
        )
    elif spec.topology_key == "dms_compact":
        cell_capacity = capacity * len(_DMS_HEAD_WINDOWS)
        pools = (
            KVPoolSpec("kv.k_live_cells", cell_capacity, unit="cells", plane_role="k_payload", lifetimes=lease_only),
            KVPoolSpec("kv.v_live_cells", cell_capacity, unit="cells", plane_role="v_payload", lifetimes=lease_only),
            KVPoolSpec("kv.span_metadata", cell_capacity, unit="entries", plane_role="live_spans", lifetimes=lease_only),
            KVPoolSpec("kv.row_metadata", capacity, unit="rows", plane_role="row_metadata", lifetimes=lease_only),
            KVPoolSpec("kv.compaction_workspace", max(1, capacity // 8), unit="rows", plane_role="compaction_workspace", lifetimes=workspace_only),
        )
    else:
        pools = (
            KVPoolSpec("kv.k_payload", capacity, unit="cells", plane_role="k_payload", lifetimes=lease_only),
            KVPoolSpec("kv.v_payload", capacity, unit="cells", plane_role="v_payload", lifetimes=lease_only),
            KVPoolSpec("kv.row_metadata", capacity, unit="rows", plane_role="row_metadata", lifetimes=lease_only),
        )
    return KVPoolPlan(backend_fingerprint=spec.fingerprint, generation=1, pools=pools)


def _storage_view(spec: KVBackendSpec, plan: KVPoolPlan) -> KVStorageView:
    non_plane_roles = {"row_metadata", "demotion_workspace", "compaction_workspace"}
    dtype_by_role = {
        "k_payload": "int8" if spec.hot_codec_key == "int8_per_token_head" else "bf16",
        "v_payload": "int8" if spec.hot_codec_key == "int8_per_token_head" else "bf16",
        "k_scale": "fp16",
        "v_scale": "fp16",
        "k_recent": "bf16",
        "v_recent": "bf16",
        "k_history": "int8",
        "v_history": "int8",
        "history_scale": "fp16",
        "live_spans": "int32",
    }
    planes = []
    pointer_base = 0x50000000 + (FAKE_KV_BACKEND_KINDS.index(
        next(kind for kind in FAKE_KV_BACKEND_KINDS if f":{kind}:" in spec.layout_fingerprint)
    ) + 1) * 0x01000000
    for index, pool in enumerate(plan.pools):
        if pool.plane_role in non_plane_roles:
            continue
        planes.append(
            KVPlaneView(
                role=pool.plane_role,
                dtype=dtype_by_role[pool.plane_role],
                ptr=pointer_base + index * 0x00100000,
                shape=(pool.capacity,),
                strides=(1,),
            )
        )
    return KVStorageView(
        layout_key=f"{spec.topology_key}+{spec.hot_codec_key}",
        generation=plan.generation,
        planes=tuple(planes),
        metadata_descriptor_ptr=pointer_base + 0x00F00000,
        metadata_descriptor_bytes=256,
        artifact_fingerprint=spec.artifact_fingerprint,
    )


__all__ = [
    "FAKE_KV_BACKEND_KINDS",
    "SimulatedKVBackend",
    "SimulatedKVOperation",
    "create_fake_kv_backend",
]
