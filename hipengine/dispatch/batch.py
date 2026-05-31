"""Batch-shaped scheduler metadata for continuous decoding.

This module is deliberately host-only and torch-free.  It gives the rest of the
runtime stable request identities, physical slot maps, row maps, and graph shape
keys before the Qwen3.5/PARO kernels are fully c>1-capable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import ceil
from typing import Iterable, Mapping, Sequence


class WorkKind(str, Enum):
    """Scheduler work class for one batch step."""

    PREFILL = "prefill"
    DECODE = "decode"
    VERIFY_CHAIN = "verify_chain"
    VERIFY_TREE = "verify_tree"


@dataclass(frozen=True, slots=True)
class RequestState:
    """Stable per-request state owned by the scheduler.

    ``request_id`` never changes.  Physical batch slots may change after
    compaction; kernels should receive row/slot metadata rather than infer
    request identity from row number.
    """

    request_id: int
    prompt_tokens: tuple[int, ...]
    max_new_tokens: int
    next_prompt_index: int = 0
    generated_tokens: tuple[int, ...] = ()
    finished: bool = False

    def __post_init__(self) -> None:
        if self.request_id < 0:
            raise ValueError("request_id must be non-negative")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if self.next_prompt_index < 0 or self.next_prompt_index > len(self.prompt_tokens):
            raise ValueError("next_prompt_index must be within prompt_tokens")
        if any(int(token) < 0 for token in self.prompt_tokens):
            raise ValueError("prompt token ids must be non-negative")
        if any(int(token) < 0 for token in self.generated_tokens):
            raise ValueError("generated token ids must be non-negative")

    @classmethod
    def from_tokens(cls, request_id: int, prompt_tokens: Iterable[int], *, max_new_tokens: int) -> "RequestState":
        return cls(request_id=int(request_id), prompt_tokens=tuple(int(token) for token in prompt_tokens), max_new_tokens=int(max_new_tokens))

    @property
    def context_len(self) -> int:
        """Committed token count visible to attention/KV for this request."""

        return self.next_prompt_index + len(self.generated_tokens)

    @property
    def remaining_prefill(self) -> int:
        return len(self.prompt_tokens) - self.next_prompt_index

    @property
    def remaining_decode(self) -> int:
        return max(0, self.max_new_tokens - len(self.generated_tokens))

    def take_prefill(self, limit: int) -> tuple["RequestState", tuple[int, ...]]:
        """Return an updated request plus the next prompt chunk."""

        if limit <= 0:
            raise ValueError("prefill chunk limit must be positive")
        end = min(len(self.prompt_tokens), self.next_prompt_index + int(limit))
        chunk = self.prompt_tokens[self.next_prompt_index : end]
        return replace(self, next_prompt_index=end), chunk

    def append_generated(self, token_id: int, *, finished: bool = False) -> "RequestState":
        token = int(token_id)
        if token < 0:
            raise ValueError("generated token id must be non-negative")
        generated = (*self.generated_tokens, token)
        done = bool(finished) or len(generated) >= self.max_new_tokens
        return replace(self, generated_tokens=generated, finished=done)

    def mark_finished(self) -> "RequestState":
        return replace(self, finished=True)


@dataclass(frozen=True, slots=True)
class BatchSlot:
    """One physical batch slot."""

    slot: int
    request_id: int
    active: bool = True

    def __post_init__(self) -> None:
        if self.slot < 0:
            raise ValueError("slot must be non-negative")
        if self.request_id < 0:
            raise ValueError("request_id must be non-negative")


@dataclass(frozen=True, slots=True)
class SlotMove:
    """Result of a compaction/reorder operation."""

    request_id: int
    old_slot: int
    new_slot: int


@dataclass(frozen=True, slots=True)
class WorkItem:
    """A scheduler work item expressed in stable request ids and row metadata."""

    kind: WorkKind
    request_ids: tuple[int, ...]
    row_to_request: tuple[int, ...]
    token_rows: tuple[tuple[int, ...], ...] = ()
    draft_depth: int = 0
    tree_parents: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.request_ids:
            raise ValueError("work item must include at least one request")
        if not self.row_to_request:
            raise ValueError("work item must include row_to_request metadata")
        known = set(self.request_ids)
        if any(request_id not in known for request_id in self.row_to_request):
            raise ValueError("row_to_request contains a request id not in request_ids")
        if self.draft_depth < 0:
            raise ValueError("draft_depth must be non-negative")
        if self.kind in {WorkKind.VERIFY_CHAIN, WorkKind.VERIFY_TREE} and self.draft_depth <= 0:
            raise ValueError("verify work requires positive draft_depth")


@dataclass(frozen=True, slots=True)
class BatchShapeKey:
    """Shape key for graph capture/replay caches.

    This is intentionally richer than batch size: SpecDec verification and
    continuous batching need separate buckets for mode, context/page bucket,
    masks, KV storage dtype, layer plan, draft/tree shape, experts, and replay
    length.
    """

    mode: WorkKind
    active_c: int
    context_bucket: int
    active_mask: tuple[bool, ...]
    kv_storage_dtype: str = "bf16"
    layer_plan: str = "all"
    top_k: int = 0
    experts_per_token: int = 0
    replay_steps: int = 1
    draft_depth: int = 0
    tree_shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.active_c < 0:
            raise ValueError("active_c must be non-negative")
        if self.context_bucket < 0:
            raise ValueError("context_bucket must be non-negative")
        if self.active_c != sum(1 for active in self.active_mask if active):
            raise ValueError("active_c must match active_mask")
        if not isinstance(self.kv_storage_dtype, str) or not self.kv_storage_dtype.strip():
            raise ValueError("kv_storage_dtype must be a non-empty string")
        if not isinstance(self.layer_plan, str) or not self.layer_plan.strip():
            raise ValueError("layer_plan must be a non-empty string")
        if self.top_k < 0 or self.experts_per_token < 0:
            raise ValueError("top_k and experts_per_token must be non-negative")
        if self.replay_steps <= 0:
            raise ValueError("replay_steps must be positive")
        if self.draft_depth < 0:
            raise ValueError("draft_depth must be non-negative")
        if any(item < 0 for item in self.tree_shape):
            raise ValueError("tree_shape entries must be non-negative")


class ActiveBatch:
    """Mutable active-request table with stable ids and compactable slots."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._slots: list[BatchSlot | None] = [None] * self.capacity
        self._requests: dict[int, RequestState] = {}

    @property
    def requests(self) -> Mapping[int, RequestState]:
        return self._requests

    @property
    def slots(self) -> tuple[BatchSlot | None, ...]:
        return tuple(self._slots)

    @property
    def request_to_slot(self) -> dict[int, int]:
        return {slot.request_id: index for index, slot in enumerate(self._slots) if slot is not None and slot.active}

    @property
    def active_request_ids(self) -> tuple[int, ...]:
        return tuple(slot.request_id for slot in self._slots if slot is not None and slot.active)

    @property
    def active_count(self) -> int:
        return len(self.active_request_ids)

    @property
    def active_mask(self) -> tuple[bool, ...]:
        return tuple(slot is not None and slot.active for slot in self._slots)

    @property
    def slot_to_request(self) -> tuple[int | None, ...]:
        return tuple(slot.request_id if slot is not None and slot.active else None for slot in self._slots)

    def admit(self, request: RequestState) -> int:
        """Admit a request into the first free physical slot and return it."""

        if request.request_id in self._requests:
            raise ValueError(f"request_id {request.request_id} is already admitted")
        try:
            slot_index = next(index for index, slot in enumerate(self._slots) if slot is None or not slot.active)
        except StopIteration as exc:
            raise RuntimeError("active batch is full") from exc
        self._requests[request.request_id] = request
        self._slots[slot_index] = BatchSlot(slot=slot_index, request_id=request.request_id)
        return slot_index

    def update_request(self, request: RequestState) -> None:
        if request.request_id not in self._requests:
            raise KeyError(request.request_id)
        self._requests[request.request_id] = request

    def finish(self, request_id: int) -> None:
        """Mark a request inactive but leave its physical slot reusable."""

        slot_index = self.slot_for(request_id)
        self._requests[request_id] = self._requests[request_id].mark_finished()
        self._slots[slot_index] = None

    def reclaim(self, request_id: int) -> RequestState:
        """Remove a finished/inactive request from scheduler ownership."""

        request = self._requests.pop(request_id)
        for index, slot in enumerate(self._slots):
            if slot is not None and slot.request_id == request_id:
                self._slots[index] = None
        return request

    def slot_for(self, request_id: int) -> int:
        for index, slot in enumerate(self._slots):
            if slot is not None and slot.active and slot.request_id == request_id:
                return index
        raise KeyError(request_id)

    def compact(self, order: Sequence[int] | None = None) -> tuple[SlotMove, ...]:
        """Pack active slots to the front, optionally in a request-id order."""

        active_ids = list(self.active_request_ids)
        if order is not None:
            requested = tuple(int(request_id) for request_id in order)
            if set(requested) != set(active_ids):
                raise ValueError("compaction order must contain exactly the active request ids")
            active_ids = list(requested)
        old_slots = self.request_to_slot
        self._slots = [None] * self.capacity
        moves: list[SlotMove] = []
        for new_slot, request_id in enumerate(active_ids):
            self._slots[new_slot] = BatchSlot(slot=new_slot, request_id=request_id)
            moves.append(SlotMove(request_id=request_id, old_slot=old_slots[request_id], new_slot=new_slot))
        return tuple(moves)

    def row_map(self, *, rows_per_request: int = 1, request_ids: Sequence[int] | None = None) -> tuple[int, ...]:
        """Return physical slot ids repeated for routed/candidate rows."""

        if rows_per_request <= 0:
            raise ValueError("rows_per_request must be positive")
        ids = tuple(self.active_request_ids if request_ids is None else tuple(int(item) for item in request_ids))
        return tuple(slot for request_id in ids for slot in (self.slot_for(request_id),) * rows_per_request)

    def request_row_map(self, *, rows_per_request: int = 1, request_ids: Sequence[int] | None = None) -> tuple[int, ...]:
        """Return stable request ids repeated for routed/candidate rows."""

        if rows_per_request <= 0:
            raise ValueError("rows_per_request must be positive")
        ids = tuple(self.active_request_ids if request_ids is None else tuple(int(item) for item in request_ids))
        for request_id in ids:
            self.slot_for(request_id)
        return tuple(request_id for request_id in ids for _ in range(rows_per_request))

    def shape_key(
        self,
        *,
        mode: WorkKind | str,
        context_bucket_size: int,
        top_k: int = 0,
        experts_per_token: int = 0,
        replay_steps: int = 1,
        kv_storage_dtype: str = "bf16",
        layer_plan: str = "all",
        draft_depth: int = 0,
        tree_shape: Sequence[int] = (),
    ) -> BatchShapeKey:
        if context_bucket_size <= 0:
            raise ValueError("context_bucket_size must be positive")
        max_context = max((self._requests[request_id].context_len for request_id in self.active_request_ids), default=0)
        context_bucket = 0 if max_context == 0 else int(ceil(max_context / context_bucket_size) * context_bucket_size)
        return BatchShapeKey(
            mode=WorkKind(mode),
            active_c=self.active_count,
            context_bucket=context_bucket,
            active_mask=self.active_mask,
            kv_storage_dtype=str(kv_storage_dtype),
            layer_plan=str(layer_plan),
            top_k=int(top_k),
            experts_per_token=int(experts_per_token),
            replay_steps=int(replay_steps),
            draft_depth=int(draft_depth),
            tree_shape=tuple(int(item) for item in tree_shape),
        )


__all__ = [
    "ActiveBatch",
    "BatchShapeKey",
    "BatchSlot",
    "RequestState",
    "SlotMove",
    "WorkItem",
    "WorkKind",
]
