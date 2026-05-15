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
class TargetVerifyBatch:
    """Root + draft rows for one native target-verification forward.

    ``DraftBatch`` carries candidate rows only.  A native verifier needs the
    already-committed root row for every request so candidate parent indices can
    be resolved without host-side depth loops.  ``TargetVerifyBatch`` is pure
    metadata: kernels/runtimes still own the token, position, mask, and state
    buffers, but this object fixes the row layout used by graph buckets and
    transaction bookkeeping.
    """

    request_ids: tuple[int, ...]
    tokens: tuple[int, ...]
    positions: tuple[int, ...]
    row_to_request: tuple[int, ...]
    parent_rows: tuple[int, ...]
    root_rows: tuple[int, ...]
    candidate_rows: tuple[int, ...]
    draft_depths: tuple[int, ...]
    active_mask: tuple[bool, ...]
    mode: str = "verify_chain"

    def __post_init__(self) -> None:
        rows = len(self.tokens)
        if rows == 0:
            raise ValueError("TargetVerifyBatch must contain at least one row")
        if not self.request_ids:
            raise ValueError("TargetVerifyBatch must contain at least one request")
        aligned = (len(self.positions), len(self.row_to_request), len(self.parent_rows), len(self.draft_depths), len(self.active_mask))
        if any(length != rows for length in aligned):
            raise ValueError("tokens, positions, row_to_request, parent_rows, draft_depths, and active_mask must align")
        if len(self.root_rows) != len(self.request_ids):
            raise ValueError("root_rows must contain one root row per request")
        root_set = set(self.root_rows)
        candidate_set = set(self.candidate_rows)
        if root_set & candidate_set:
            raise ValueError("root_rows and candidate_rows must be disjoint")
        if len(root_set) != len(self.root_rows) or len(candidate_set) != len(self.candidate_rows):
            raise ValueError("row index sets must not contain duplicates")
        if root_set | candidate_set != set(range(rows)):
            raise ValueError("root_rows and candidate_rows must cover every row")
        known = set(self.request_ids)
        if any(request_id not in known for request_id in self.row_to_request):
            raise ValueError("row_to_request contains request id not present in request_ids")
        if any(token < 0 for token in self.tokens):
            raise ValueError("target verify token ids must be non-negative")
        if any(pos < 0 for pos in self.positions):
            raise ValueError("target verify positions must be non-negative")
        if any(depth < 0 for depth in self.draft_depths):
            raise ValueError("target verify depths must be non-negative")
        for row, parent in enumerate(self.parent_rows):
            if row in root_set:
                if parent != -1:
                    raise ValueError("root rows must have parent row -1")
            elif parent < 0 or parent >= row:
                raise ValueError("candidate parent rows must reference an earlier row")
        if self.mode not in {"verify_chain", "verify_tree"}:
            raise ValueError("mode must be verify_chain or verify_tree")

    @classmethod
    def from_draft(
        cls,
        draft: DraftBatch,
        *,
        root_tokens: Sequence[int],
        root_positions: Sequence[int],
    ) -> "TargetVerifyBatch":
        roots = tuple(int(token) for token in root_tokens)
        root_pos = tuple(int(position) for position in root_positions)
        if len(roots) != len(draft.request_ids) or len(root_pos) != len(draft.request_ids):
            raise ValueError("root tokens/positions must align with draft request_ids")
        root_rows = tuple(range(len(draft.request_ids)))
        candidate_base = len(root_rows)
        candidate_rows = tuple(range(candidate_base, candidate_base + draft.draft_rows))
        root_row_by_request = dict(zip(draft.request_ids, root_rows, strict=True))
        parent_rows: list[int] = [-1] * len(root_rows)
        tree_parents = draft.tree_parents or tuple(-1 if depth == 1 else index - 1 for index, depth in enumerate(draft.draft_depths))
        for index, parent in enumerate(tree_parents):
            request_id = draft.row_to_request[index]
            if parent < 0:
                parent_rows.append(root_row_by_request[request_id])
            else:
                if parent >= index:
                    raise ValueError("tree parent must reference an earlier candidate row")
                parent_rows.append(candidate_base + int(parent))
        active_candidates = draft.active_mask or (True,) * draft.draft_rows
        return cls(
            request_ids=draft.request_ids,
            tokens=(*roots, *draft.candidate_tokens),
            positions=(*root_pos, *(int(position) + 1 for position in draft.parent_positions)),
            row_to_request=(*draft.request_ids, *draft.row_to_request),
            parent_rows=tuple(parent_rows),
            root_rows=root_rows,
            candidate_rows=candidate_rows,
            draft_depths=(*tuple(0 for _ in root_rows), *draft.draft_depths),
            active_mask=(*tuple(True for _ in root_rows), *active_candidates),
            mode=draft.mode,
        )

    @property
    def rows(self) -> int:
        return len(self.tokens)

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_rows)


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
    """Protocol for target-model verification over a root+candidate batch."""

    def verify(self, batch: TargetVerifyBatch) -> AcceptResult: ...


__all__ = ["AcceptResult", "DraftBatch", "DraftModel", "TargetVerifyBatch", "Verifier"]
