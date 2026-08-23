"""Provider-neutral nonuniform speculative tree metadata compiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from hipengine.speculative.interfaces import DraftBatch


@dataclass(frozen=True, slots=True)
class TreeDraftRequest:
    request_id: int
    root_position: int
    candidate_tokens: tuple[int, ...]
    parent_candidate_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if min(int(self.request_id), int(self.root_position)) < 0:
            raise ValueError("request_id/root_position must be non-negative")
        tokens = tuple(int(token) for token in self.candidate_tokens)
        parents = tuple(int(parent) for parent in self.parent_candidate_ids)
        if not tokens or len(tokens) != len(parents):
            raise ValueError("tree candidate tokens/parents must be non-empty and align")
        if any(token < 0 for token in tokens):
            raise ValueError("candidate token ids must be non-negative")
        for index, parent in enumerate(parents):
            if parent < -1 or parent >= index:
                raise ValueError("tree parent must be -1 or an earlier candidate")
        object.__setattr__(self, "request_id", int(self.request_id))
        object.__setattr__(self, "root_position", int(self.root_position))
        object.__setattr__(self, "candidate_tokens", tokens)
        object.__setattr__(self, "parent_candidate_ids", parents)


def compile_tree_draft(
    requests: Sequence[TreeDraftRequest],
    *,
    max_verifier_rows: int,
    resident_slots: Mapping[int, int],
    cycle_id: int,
) -> DraftBatch:
    """Flatten nonuniform request-local trees without losing parent ownership."""

    rows = tuple(requests)
    if not rows or len({request.request_id for request in rows}) != len(rows):
        raise ValueError("tree requests must be non-empty with unique request ids")
    total = sum(len(request.candidate_tokens) for request in rows)
    if total > int(max_verifier_rows):
        raise ValueError("tree rows exceed max_verifier_rows")
    if set(resident_slots) != {request.request_id for request in rows}:
        raise ValueError("resident_slots must exactly cover tree request ids")
    slots = tuple(int(resident_slots[request.request_id]) for request in rows)
    if any(slot < 0 for slot in slots) or len(set(slots)) != len(slots):
        raise ValueError("tree resident slots must be unique and non-negative")

    candidate_tokens: list[int] = []
    parent_positions: list[int] = []
    draft_depths: list[int] = []
    row_to_request: list[int] = []
    tree_parents: list[int] = []
    flattened_slots: list[int] = []
    candidate_ids: list[int] = []
    offset = 0
    for request, slot in zip(rows, slots, strict=True):
        local_depths: list[int] = []
        for candidate_id, (token, parent) in enumerate(
            zip(request.candidate_tokens, request.parent_candidate_ids, strict=True)
        ):
            depth = 1 if parent < 0 else local_depths[parent] + 1
            local_depths.append(depth)
            candidate_tokens.append(token)
            parent_positions.append(request.root_position + depth - 1)
            draft_depths.append(depth)
            row_to_request.append(request.request_id)
            tree_parents.append(-1 if parent < 0 else offset + parent)
            flattened_slots.append(slot)
            candidate_ids.append(candidate_id)
        offset += len(request.candidate_tokens)
    return DraftBatch(
        request_ids=tuple(request.request_id for request in rows),
        candidate_tokens=tuple(candidate_tokens),
        parent_positions=tuple(parent_positions),
        draft_depths=tuple(draft_depths),
        row_to_request=tuple(row_to_request),
        tree_parents=tuple(tree_parents),
        active_mask=(True,) * total,
        mode="verify_tree",
        cycle_id=int(cycle_id),
        resident_slots=tuple(flattened_slots),
        candidate_ids=tuple(candidate_ids),
        provider_metadata=(("tree_request_count", len(rows)), ("tree_rows", total)),
    )
