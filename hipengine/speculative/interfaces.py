"""Speculative decoding interfaces shared by MTP/EAGLE3/DFlash-style plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class DraftBatch:
    """Flattened candidate rows proposed by a draft provider."""

    request_ids: tuple[int, ...]
    candidate_tokens: tuple[int, ...]
    parent_positions: tuple[int, ...]
    draft_depths: tuple[int, ...]
    row_to_request: tuple[int, ...]
    tree_parents: tuple[int, ...] = ()
    active_mask: tuple[bool, ...] = ()
    mode: str = "verify_chain"

    def __post_init__(self) -> None:
        rows = len(self.candidate_tokens)
        if rows == 0:
            raise ValueError("DraftBatch must contain at least one candidate row")
        if not self.request_ids:
            raise ValueError("DraftBatch must contain at least one request")
        if len(self.parent_positions) != rows or len(self.draft_depths) != rows or len(self.row_to_request) != rows:
            raise ValueError("candidate_tokens, parent_positions, draft_depths, and row_to_request must align")
        if self.tree_parents and len(self.tree_parents) != rows:
            raise ValueError("tree_parents must be empty or one entry per row")
        if self.active_mask and len(self.active_mask) != rows:
            raise ValueError("active_mask must be empty or one entry per row")
        known = set(self.request_ids)
        if any(request_id not in known for request_id in self.row_to_request):
            raise ValueError("row_to_request contains request id not present in request_ids")
        if any(token < 0 for token in self.candidate_tokens):
            raise ValueError("candidate token ids must be non-negative")
        if any(pos < 0 for pos in self.parent_positions):
            raise ValueError("parent positions must be non-negative")
        if any(depth <= 0 for depth in self.draft_depths):
            raise ValueError("draft depths must be positive")
        if self.mode not in {"verify_chain", "verify_tree"}:
            raise ValueError("mode must be verify_chain or verify_tree")

    @property
    def draft_rows(self) -> int:
        return len(self.candidate_tokens)

    @property
    def kind(self) -> str:
        return self.mode


@dataclass(frozen=True, slots=True)
class AcceptResult:
    """Verifier accept/reject result per live request."""

    request_ids: tuple[int, ...]
    accepted_counts: tuple[int, ...]
    accepted_tokens: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.request_ids:
            raise ValueError("AcceptResult must contain at least one request")
        if len(self.accepted_counts) != len(self.request_ids) or len(self.accepted_tokens) != len(self.request_ids):
            raise ValueError("accepted counts/tokens must align with request_ids")
        if any(count < 0 for count in self.accepted_counts):
            raise ValueError("accepted_counts must be non-negative")
        for count, tokens in zip(self.accepted_counts, self.accepted_tokens, strict=True):
            if count != len(tokens):
                raise ValueError("accepted_counts must match accepted_tokens lengths")
            if any(token < 0 for token in tokens):
                raise ValueError("accepted token ids must be non-negative")


@runtime_checkable
class DraftModel(Protocol):
    """Protocol for MTP/EAGLE3/DFlash/Medusa/Lookahead draft providers."""

    def propose(self, request_ids: Sequence[int], *, max_draft_tokens: int) -> DraftBatch: ...


@runtime_checkable
class Verifier(Protocol):
    """Protocol for target-model verification over a DraftBatch."""

    def verify(self, draft: DraftBatch) -> AcceptResult: ...


__all__ = ["AcceptResult", "DraftBatch", "DraftModel", "Verifier"]
