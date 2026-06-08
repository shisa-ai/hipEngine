"""KV policy protocols and fixed-page bookkeeping scaffolding."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.kvcache.spans import KVLiveSpans, KVScaleMetadata

KV_STORAGE_AUTO = "auto"
KV_STORAGE_CHOICES = (KV_STORAGE_AUTO, DType.BF16.value, DType.INT8_PER_TOKEN_HEAD.value)
KV_SCALE_DTYPE_CHOICES = (DType.FP16.value, DType.FP32.value)
KV_SCALE_GRANULARITY_CHOICES = ("per_token_head",)


def _validate_unique_request_ids(request_ids: Sequence[int]) -> None:
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request_ids must be unique")


@dataclass(frozen=True, slots=True)
class ResolvedKVPolicy:
    """Resolved user-facing KV storage controls for a fixed-page runtime."""

    requested_storage: str
    storage_dtype: DType
    block_size: int = 256
    scale_dtype: DType = DType.FP16
    scale_granularity: str = "per_token_head"
    selection_reason: str = "default_bf16"
    int8_explicit: bool = False
    int8_admission_gated: bool = False
    spans_mode: str = "uniform"
    policy_class: str = "FixedPagedKVPolicy"

    def __post_init__(self) -> None:
        storage = DType.parse(self.storage_dtype)
        object.__setattr__(self, "storage_dtype", storage)
        if storage not in {DType.BF16, DType.INT8_PER_TOKEN_HEAD}:
            raise ValueError("KV storage must resolve to bf16 or int8_per_token_head")
        scale = DType.parse(self.scale_dtype)
        object.__setattr__(self, "scale_dtype", scale)
        if scale not in {DType.FP16, DType.FP32}:
            raise ValueError("INT8 KV scale dtype must be fp16 or fp32")
        if self.scale_granularity not in KV_SCALE_GRANULARITY_CHOICES:
            raise ValueError("INT8 KV scale granularity must be per_token_head")
        if self.block_size <= 0:
            raise ValueError("KV block_size must be positive")
        if self.spans_mode != "uniform":
            raise ValueError("only uniform KV spans are supported by the fixed-page policy")

    @property
    def uses_int8(self) -> bool:
        return self.storage_dtype == DType.INT8_PER_TOKEN_HEAD

    def create_policy(self) -> "FixedPagedKVPolicy":
        return FixedPagedKVPolicy(block_size=self.block_size, storage_dtype=self.storage_dtype)

    def to_json_dict(self) -> dict[str, Any]:
        scale_format = {
            "present": bool(self.uses_int8),
            "scale_dtype": self.scale_dtype.value if self.uses_int8 else None,
            "granularity": self.scale_granularity if self.uses_int8 else None,
            "k_scale": "per_token_head" if self.uses_int8 else None,
            "v_scale": "per_token_head" if self.uses_int8 else None,
        }
        return {
            "policy_class": self.policy_class,
            "spans_mode": self.spans_mode,
            "block_size": int(self.block_size),
            "requested_storage": self.requested_storage,
            "resolved_storage_dtype": self.storage_dtype.value,
            "storage_dtype": self.storage_dtype.value,
            "requested_scale_dtype": self.scale_dtype.value,
            "requested_scale_granularity": self.scale_granularity,
            "scale_metadata_format": scale_format,
            "int8_explicit": bool(self.int8_explicit),
            "int8_admission_gated": bool(self.int8_admission_gated),
            "selection_reason": self.selection_reason,
        }


def resolve_kv_policy(
    requested_storage: str | DType = KV_STORAGE_AUTO,
    *,
    block_size: int = 256,
    scale_dtype: str | DType = DType.FP16,
    scale_granularity: str = "per_token_head",
    admission_gated_int8: bool = False,
) -> ResolvedKVPolicy:
    """Resolve user/test KV controls into concrete fixed-page policy metadata.

    ``auto`` deliberately resolves to BF16 unless the caller marks the request as
    requiring INT8 for admission.  Explicit ``int8_per_token_head`` records that
    the user/test requested INT8 rather than an admission fallback.
    """

    requested = requested_storage.value if isinstance(requested_storage, DType) else str(requested_storage)
    if requested == KV_STORAGE_AUTO:
        if admission_gated_int8:
            storage = DType.INT8_PER_TOKEN_HEAD
            reason = "admission_gated_int8"
            int8_admission = True
        else:
            storage = DType.BF16
            reason = "default_bf16"
            int8_admission = False
        int8_explicit = False
    else:
        storage = DType.parse(requested)
        if storage not in {DType.BF16, DType.INT8_PER_TOKEN_HEAD}:
            raise ValueError("KV storage must be bf16, int8_per_token_head, or auto")
        reason = "explicit_int8" if storage == DType.INT8_PER_TOKEN_HEAD else "explicit_bf16"
        int8_explicit = storage == DType.INT8_PER_TOKEN_HEAD
        int8_admission = False
    return ResolvedKVPolicy(
        requested_storage=requested,
        storage_dtype=storage,
        block_size=int(block_size),
        scale_dtype=DType.parse(scale_dtype),
        scale_granularity=scale_granularity,
        selection_reason=reason,
        int8_explicit=int8_explicit,
        int8_admission_gated=int8_admission,
    )


@dataclass(frozen=True, slots=True)
class KVReservation:
    """Host-side reservation metadata for one request's KV arena."""

    request_id: int
    block_table: Tensor
    live_counts: Tensor
    max_live_count: int
    capacity_tokens: int
    storage_dtype: DType
    scale_metadata: KVScaleMetadata | None = None

    def __post_init__(self) -> None:
        if self.request_id < 0:
            raise ValueError("request_id must be non-negative")
        if self.block_table.dtype != DType.INT32:
            raise ValueError("block_table must be int32")
        if self.live_counts.dtype not in {DType.INT32, DType.INT64}:
            raise ValueError("live_counts must be int32 or int64")
        if self.block_table.device != self.live_counts.device:
            raise ValueError("block_table and live_counts must be on the same device")
        if self.max_live_count < 0:
            raise ValueError("max_live_count must be non-negative")
        if self.capacity_tokens <= 0:
            raise ValueError("capacity_tokens must be positive")
        if self.max_live_count > self.capacity_tokens:
            raise ValueError("max_live_count cannot exceed capacity_tokens")
        storage = DType.parse(self.storage_dtype)
        object.__setattr__(self, "storage_dtype", storage)
        if storage == DType.INT8_PER_TOKEN_HEAD and self.scale_metadata is None:
            raise ValueError("int8_per_token_head reservations require scale metadata")
        if storage != DType.INT8_PER_TOKEN_HEAD and self.scale_metadata is not None:
            raise ValueError("scale metadata is only valid for int8_per_token_head reservations")
        if self.scale_metadata is not None and self.scale_metadata.device != self.block_table.device:
            raise ValueError("scale metadata must be on the same device as block_table")


@dataclass(frozen=True, slots=True)
class KVTransaction:
    """Speculative KV write transaction metadata."""

    transaction_id: int
    request_ids: tuple[int, ...]
    draft_rows: int
    role: str
    candidate_counts: tuple[int, ...] | None = None
    committed: bool = False
    rolled_back: bool = False
    accepted_counts: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.transaction_id < 0:
            raise ValueError("transaction_id must be non-negative")
        if not self.request_ids:
            raise ValueError("transaction must include at least one request")
        _validate_unique_request_ids(self.request_ids)
        if self.draft_rows <= 0:
            raise ValueError("draft_rows must be positive")
        if self.role not in {"verify_chain", "verify_tree"}:
            raise ValueError("role must be verify_chain or verify_tree")
        if self.committed and self.rolled_back:
            raise ValueError("transaction cannot be both committed and rolled back")
        if self.candidate_counts is not None:
            if len(self.candidate_counts) != len(self.request_ids):
                raise ValueError("candidate_counts must match request_ids")
            if any(count < 0 for count in self.candidate_counts):
                raise ValueError("candidate_counts must be non-negative")
            if sum(self.candidate_counts) > self.draft_rows:
                raise ValueError("candidate_counts cannot exceed draft_rows")
        if self.committed and self.accepted_counts is None:
            raise ValueError("committed transaction requires accepted_counts")
        if self.accepted_counts is not None:
            if not self.committed:
                raise ValueError("accepted_counts require committed transaction")
            if len(self.accepted_counts) != len(self.request_ids):
                raise ValueError("accepted_counts must match request_ids")
            if any(count < 0 for count in self.accepted_counts):
                raise ValueError("accepted_counts must be non-negative")
            if self.candidate_counts is not None:
                for accepted, available in zip(self.accepted_counts, self.candidate_counts, strict=True):
                    if accepted > available:
                        raise ValueError("accepted_counts cannot exceed candidate_counts")
            elif sum(self.accepted_counts) > self.draft_rows:
                raise ValueError("accepted_counts cannot exceed draft_rows")


@runtime_checkable
class KVPolicy(Protocol):
    """Scheduler-facing KV policy protocol."""

    spans_mode: str
    storage_dtype: DType

    def admission_cap(self, seq: Any | None = None) -> int: ...

    def batch_spans(self, batch: Sequence[Any], *, role: str = "decode", **metadata: Any) -> KVLiveSpans: ...

    def begin_transaction(self, seqs: Sequence[Any], draft: Any) -> KVTransaction: ...

    def commit(self, txn: KVTransaction, accepted_counts: Sequence[int]) -> KVTransaction: ...

    def rollback(self, txn: KVTransaction) -> KVTransaction: ...

    def reclaim(self, seq: Any) -> KVReservation: ...


class FixedPagedKVPolicy:
    """Host bookkeeping for uniform fixed-page KV spans.

    Device metadata tensors are still owned by the runtime/scheduler.  For c=1,
    ``batch_spans`` can reuse the reservation tensors directly.  For c>1 or
    speculative verification rows, callers pass prepacked device metadata tensors
    so this policy never fabricates device memory.
    """

    spans_mode = "uniform"

    def __init__(
        self,
        *,
        block_size: int = 256,
        storage_dtype: str | DType = DType.BF16,
        total_capacity_tokens: int | None = None,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if total_capacity_tokens is not None and int(total_capacity_tokens) <= 0:
            raise ValueError("total_capacity_tokens must be positive")
        self.block_size = int(block_size)
        self.storage_dtype = DType.parse(storage_dtype)
        self.total_capacity_tokens = None if total_capacity_tokens is None else int(total_capacity_tokens)
        self._reservations: dict[int, KVReservation] = {}
        self._transactions: dict[int, KVTransaction] = {}
        self._block_pointer_by_id: dict[int, int] = {}
        self._live_block_owner_by_id: dict[int, int] = {}
        self._reservation_block_ids: dict[int, tuple[int, ...]] = {}
        self._next_transaction_id = 0

    @property
    def reservations(self) -> dict[int, KVReservation]:
        return dict(self._reservations)

    @property
    def transactions(self) -> dict[int, KVTransaction]:
        return dict(self._transactions)

    def register(
        self,
        request_id: int,
        *,
        block_table: Tensor,
        live_counts: Tensor,
        max_live_count: int,
        capacity_tokens: int | None = None,
        scale_metadata: KVScaleMetadata | None = None,
        block_pointer_map: Mapping[int, int] | None = None,
    ) -> KVReservation:
        rid = int(request_id)
        if rid in self._reservations:
            raise ValueError(f"request_id {rid} already has a KV reservation")
        capacity = int(capacity_tokens) if capacity_tokens is not None else block_table.numel * self.block_size
        if (
            self.total_capacity_tokens is not None
            and self._reserved_capacity_tokens() + capacity > self.total_capacity_tokens
        ):
            raise ValueError("KV reservation exceeds current policy admission capacity")
        reservation = KVReservation(
            request_id=rid,
            block_table=block_table,
            live_counts=live_counts,
            max_live_count=int(max_live_count),
            capacity_tokens=capacity,
            storage_dtype=self.storage_dtype,
            scale_metadata=scale_metadata,
        )
        self._reserve_block_pointers(rid, block_pointer_map)
        self._reservations[rid] = reservation
        return reservation

    def admission_cap(self, seq: Any | None = None) -> int:
        if self.total_capacity_tokens is not None:
            return max(0, self.total_capacity_tokens - self._reserved_capacity_tokens())
        if seq is None:
            return sum(
                reservation.capacity_tokens - reservation.max_live_count
                for reservation in self._reservations.values()
            )
        reservation = self._reservation_for(seq)
        return reservation.capacity_tokens - reservation.max_live_count

    def batch_spans(
        self,
        batch: Sequence[Any],
        *,
        role: str = "decode",
        block_table: Tensor | None = None,
        live_counts: Tensor | None = None,
        request_ids: Tensor | None = None,
        row_positions: Tensor | None = None,
        max_live_count: int | None = None,
        scale_metadata: KVScaleMetadata | None = None,
    ) -> KVLiveSpans:
        request_ids_tuple = tuple(_request_id(item) for item in batch)
        if not request_ids_tuple:
            raise ValueError("batch_spans requires at least one request")
        reservations = tuple(self._reservations[rid] for rid in request_ids_tuple)
        if len(reservations) == 1 and block_table is None and live_counts is None:
            reservation = reservations[0]
            block_table = reservation.block_table
            live_counts = reservation.live_counts
            max_live_count = reservation.max_live_count if max_live_count is None else max_live_count
            scale_metadata = reservation.scale_metadata if scale_metadata is None else scale_metadata
        elif block_table is None or live_counts is None:
            raise ValueError("c>1 batch_spans requires packed block_table and live_counts tensors")
        if max_live_count is None:
            max_live_count = max(reservation.max_live_count for reservation in reservations)
        return KVLiveSpans.paged_uniform(
            block_table=block_table,
            live_counts=live_counts,
            max_live_count=int(max_live_count),
            storage_dtype=self.storage_dtype,
            request_ids=request_ids,
            row_positions=row_positions,
            span_role=role,
            scale_metadata=scale_metadata,
        )

    def begin_transaction(self, seqs: Sequence[Any], draft: Any) -> KVTransaction:
        request_ids = tuple(_request_id(seq) for seq in seqs)
        if not request_ids:
            raise ValueError("begin_transaction requires at least one request")
        _validate_unique_request_ids(request_ids)
        for rid in request_ids:
            if rid not in self._reservations:
                raise KeyError(rid)
        candidate_rows = getattr(draft, "candidate_rows", None)
        row_to_request = getattr(draft, "row_to_request", None)
        candidate_row_to_request = None
        if row_to_request is not None:
            row_requests = tuple(int(item) for item in row_to_request)
            if candidate_rows is not None:
                candidate_row_to_request = tuple(row_requests[int(row)] for row in candidate_rows)
            else:
                candidate_row_to_request = row_requests
            unknown = set(candidate_row_to_request) - set(request_ids)
            if unknown:
                raise ValueError(f"draft rows reference unknown request ids {sorted(unknown)!r}")
        if candidate_rows is not None:
            draft_rows = len(candidate_rows)
        else:
            draft_rows = len(candidate_row_to_request) if candidate_row_to_request is not None else len(request_ids)
        candidate_counts = (
            tuple(sum(1 for row_request in candidate_row_to_request if row_request == request_id) for request_id in request_ids)
            if candidate_row_to_request is not None
            else None
        )
        role = str(getattr(draft, "kind", getattr(draft, "mode", "verify_chain")))
        if role.startswith("WorkKind."):
            role = role.rsplit(".", 1)[-1].lower()
        if role not in {"verify_chain", "verify_tree"}:
            role = "verify_chain"
        txn = KVTransaction(
            transaction_id=self._next_transaction_id,
            request_ids=request_ids,
            draft_rows=draft_rows,
            role=role,
            candidate_counts=candidate_counts,
        )
        self._next_transaction_id += 1
        self._transactions[txn.transaction_id] = txn
        return txn

    def commit(self, txn: KVTransaction, accepted_counts: Sequence[int]) -> KVTransaction:
        current = self._current_transaction(txn)
        if current.rolled_back:
            raise ValueError("cannot commit a rolled-back KV transaction")
        if current.committed:
            raise ValueError("cannot commit an already committed KV transaction")
        counts = tuple(int(item) for item in accepted_counts)
        if len(counts) != len(current.request_ids):
            raise ValueError("accepted_counts must match transaction request_ids")
        if any(item < 0 for item in counts):
            raise ValueError("accepted_counts must be non-negative")
        if current.candidate_counts is not None:
            for accepted, available in zip(counts, current.candidate_counts, strict=True):
                if accepted > available:
                    raise ValueError("accepted_counts cannot exceed candidate_counts")
        updated = replace(current, committed=True, accepted_counts=counts)
        self._transactions[updated.transaction_id] = updated
        return updated

    def rollback(self, txn: KVTransaction) -> KVTransaction:
        current = self._current_transaction(txn)
        if current.committed:
            raise ValueError("cannot rollback a committed KV transaction")
        if current.rolled_back:
            raise ValueError("cannot rollback an already rolled-back KV transaction")
        updated = replace(current, rolled_back=True)
        self._transactions[updated.transaction_id] = updated
        return updated

    def reclaim(self, seq: Any) -> KVReservation:
        rid = _request_id(seq)
        reservation = self._reservations.pop(rid)
        for block_id in self._reservation_block_ids.pop(rid, ()):
            self._live_block_owner_by_id.pop(block_id, None)
        return reservation

    def _reservation_for(self, seq: Any) -> KVReservation:
        rid = _request_id(seq)
        try:
            return self._reservations[rid]
        except KeyError as exc:
            raise KeyError(f"no KV reservation for request_id {rid}") from exc

    def _reserved_capacity_tokens(self) -> int:
        return sum(reservation.capacity_tokens for reservation in self._reservations.values())

    def _reserve_block_pointers(self, request_id: int, block_pointer_map: Mapping[int, int] | None) -> None:
        if block_pointer_map is None:
            return
        normalized: dict[int, int] = {}
        for raw_block_id, raw_pointer in block_pointer_map.items():
            block_id = int(raw_block_id)
            pointer = int(raw_pointer)
            if block_id < 0:
                raise ValueError("block ids must be non-negative")
            if pointer < 0:
                raise ValueError("block backing pointers must be non-negative")
            normalized[block_id] = pointer
        for block_id, pointer in normalized.items():
            live_owner = self._live_block_owner_by_id.get(block_id)
            if live_owner is not None:
                raise ValueError(f"block id {block_id} is already live for request_id {live_owner}")
            known_pointer = self._block_pointer_by_id.get(block_id)
            if known_pointer is not None and known_pointer != pointer:
                raise ValueError(
                    f"block id {block_id} backing pointer changed from {known_pointer} to {pointer}"
                )
        for block_id, pointer in normalized.items():
            self._block_pointer_by_id.setdefault(block_id, pointer)
            self._live_block_owner_by_id[block_id] = request_id
        self._reservation_block_ids[request_id] = tuple(sorted(normalized))

    def _current_transaction(self, txn: KVTransaction) -> KVTransaction:
        try:
            return self._transactions[txn.transaction_id]
        except KeyError as exc:
            raise KeyError(f"unknown KV transaction {txn.transaction_id}") from exc


def _request_id(seq: Any) -> int:
    if isinstance(seq, int):
        rid = seq
    else:
        rid = getattr(seq, "request_id")
    rid = int(rid)
    if rid < 0:
        raise ValueError("request_id must be non-negative")
    return rid


__all__ = ["FixedPagedKVPolicy", "KVPolicy", "KVReservation", "KVScaleMetadata", "KVTransaction"]
