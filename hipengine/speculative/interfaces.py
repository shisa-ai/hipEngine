"""Speculative decoding interfaces shared by MTP/EAGLE3/DFlash-style plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from hipengine.dispatch.batch import WorkItem, WorkKind


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
class TargetCommitSelection:
    """Per-request target rows selected for state/KV commit."""

    request_ids: tuple[int, ...]
    accepted_counts: tuple[int, ...]
    selected_rows: tuple[int, ...]
    selected_tokens: tuple[int, ...]
    selected_positions: tuple[int, ...]
    mode: str = "verify_chain"

    def __post_init__(self) -> None:
        if not self.request_ids:
            raise ValueError("TargetCommitSelection must contain at least one request")
        lengths = (len(self.accepted_counts), len(self.selected_rows), len(self.selected_tokens), len(self.selected_positions))
        if any(length != len(self.request_ids) for length in lengths):
            raise ValueError("commit selection fields must align with request_ids")
        if any(count < 0 for count in self.accepted_counts):
            raise ValueError("accepted_counts must be non-negative")
        if any(row < 0 for row in self.selected_rows):
            raise ValueError("selected_rows must be non-negative")
        if any(token < 0 for token in self.selected_tokens):
            raise ValueError("selected_tokens must be non-negative")
        if any(position < 0 for position in self.selected_positions):
            raise ValueError("selected_positions must be non-negative")
        if self.mode not in {"verify_chain", "verify_tree"}:
            raise ValueError("mode must be verify_chain or verify_tree")


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
            elif self.row_to_request[parent] != self.row_to_request[row]:
                raise ValueError("candidate parent rows must belong to the same request")
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

    @property
    def candidate_counts(self) -> tuple[int, ...]:
        return tuple(
            sum(1 for row in self.candidate_rows if self.row_to_request[row] == request_id)
            for request_id in self.request_ids
        )

    @property
    def draft_depth(self) -> int:
        return max((self.draft_depths[row] for row in self.candidate_rows), default=0)

    @property
    def tree_shape(self) -> tuple[int, ...]:
        candidate_index_by_row = {row: index for index, row in enumerate(self.candidate_rows)}
        root_row_by_request = dict(zip(self.request_ids, self.root_rows, strict=True))
        shape: list[int] = []
        for row in self.candidate_rows:
            parent = self.parent_rows[row]
            if parent == root_row_by_request[self.row_to_request[row]]:
                shape.append(0)
            else:
                shape.append(candidate_index_by_row[parent] + 1)
        return tuple(shape)

    def shape_key(
        self,
        active_batch,
        *,
        context_bucket_size: int,
        top_k: int = 0,
        experts_per_token: int = 0,
        replay_steps: int = 1,
    ):
        return active_batch.shape_key(
            mode=self.mode,
            context_bucket_size=context_bucket_size,
            top_k=top_k,
            experts_per_token=experts_per_token,
            replay_steps=replay_steps,
            draft_depth=self.draft_depth,
            tree_shape=self.tree_shape,
        )

    def to_work_item(self) -> WorkItem:
        """Project candidate rows into scheduler work metadata.

        Root rows are committed context and stay out of row-to-request/token rows;
        graph/verify code can recover root topology from ``tree_shape`` and the
        accompanying ``TargetVerifyBatch`` metadata.
        """

        return WorkItem(
            kind=WorkKind(self.mode),
            request_ids=self.request_ids,
            row_to_request=tuple(self.row_to_request[row] for row in self.candidate_rows),
            token_rows=tuple((self.tokens[row],) for row in self.candidate_rows),
            draft_depth=self.draft_depth,
            tree_parents=self.tree_shape,
        )

    def select_commit_rows(
        self,
        accepted_counts: Sequence[int],
        *,
        selected_candidate_rows: Sequence[int | None] | None = None,
    ) -> TargetCommitSelection:
        counts = tuple(int(count) for count in accepted_counts)
        if len(counts) != len(self.request_ids):
            raise ValueError("accepted_counts must align with request_ids")
        if any(count < 0 for count in counts):
            raise ValueError("accepted_counts must be non-negative")
        selected_hint = None if selected_candidate_rows is None else tuple(selected_candidate_rows)
        if selected_hint is not None and len(selected_hint) != len(self.request_ids):
            raise ValueError("selected_candidate_rows must align with request_ids")
        candidate_set = set(self.candidate_rows)
        selected_rows: list[int] = []
        for index, (request_id, accepted_count) in enumerate(zip(self.request_ids, counts, strict=True)):
            if accepted_count == 0:
                hint = None if selected_hint is None else selected_hint[index]
                root_row = self.root_rows[index]
                if hint is not None and int(hint) != root_row:
                    raise ValueError("zero accepted candidates must select the request root row")
                selected_rows.append(root_row)
                continue
            hint = None if selected_hint is None else selected_hint[index]
            if hint is None:
                matches = [
                    row
                    for row in self.candidate_rows
                    if self.row_to_request[row] == request_id
                    and self.draft_depths[row] == accepted_count
                    and self.active_mask[row]
                ]
                if len(matches) != 1:
                    raise ValueError("accepted_count is ambiguous for request; pass selected_candidate_rows")
                row = matches[0]
            else:
                row = int(hint)
                if row not in candidate_set:
                    raise ValueError("selected candidate row must be a candidate row")
                if self.row_to_request[row] != request_id:
                    raise ValueError("selected candidate row belongs to a different request")
                if self.draft_depths[row] != accepted_count:
                    raise ValueError("selected candidate row depth must match accepted_count")
                if not self.active_mask[row]:
                    raise ValueError("selected candidate row is inactive")
            selected_rows.append(row)
        return TargetCommitSelection(
            request_ids=self.request_ids,
            accepted_counts=counts,
            selected_rows=tuple(selected_rows),
            selected_tokens=tuple(self.tokens[row] for row in selected_rows),
            selected_positions=tuple(self.positions[row] for row in selected_rows),
            mode=self.mode,
        )


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


@dataclass(frozen=True, slots=True)
class TargetAcceptSummary:
    """Accepted target paths plus the per-request row selected for commit."""

    request_ids: tuple[int, ...]
    accepted_counts: tuple[int, ...]
    accepted_tokens: tuple[tuple[int, ...], ...]
    commit_rows: tuple[int, ...]
    commit_tokens: tuple[int, ...]
    commit_positions: tuple[int, ...]
    full_accept: tuple[bool, ...]
    mode: str = "verify_chain"

    def __post_init__(self) -> None:
        if not self.request_ids:
            raise ValueError("TargetAcceptSummary must contain at least one request")
        lengths = (
            len(self.accepted_counts),
            len(self.accepted_tokens),
            len(self.commit_rows),
            len(self.commit_tokens),
            len(self.commit_positions),
            len(self.full_accept),
        )
        if any(length != len(self.request_ids) for length in lengths):
            raise ValueError("accept summary fields must align with request_ids")
        if any(count < 0 for count in self.accepted_counts):
            raise ValueError("accepted_counts must be non-negative")
        if any(row < 0 for row in self.commit_rows):
            raise ValueError("commit_rows must be non-negative")
        if any(token < 0 for token in self.commit_tokens):
            raise ValueError("commit_tokens must be non-negative")
        if any(position < 0 for position in self.commit_positions):
            raise ValueError("commit_positions must be non-negative")
        for count, tokens in zip(self.accepted_counts, self.accepted_tokens, strict=True):
            if count != len(tokens):
                raise ValueError("accepted_counts must match accepted_tokens lengths")
            if any(token < 0 for token in tokens):
                raise ValueError("accepted token ids must be non-negative")
        if self.mode not in {"verify_chain", "verify_tree"}:
            raise ValueError("mode must be verify_chain or verify_tree")

    @classmethod
    def from_accept_result(
        cls,
        target: TargetVerifyBatch,
        result: AcceptResult,
        *,
        selected_candidate_rows: Sequence[int | None] | None = None,
    ) -> "TargetAcceptSummary":
        if tuple(result.request_ids) != target.request_ids:
            raise ValueError("accept result request_ids must match target batch request_ids")
        selection = target.select_commit_rows(
            result.accepted_counts,
            selected_candidate_rows=selected_candidate_rows,
        )
        accepted_tokens = tuple(tuple(int(token) for token in tokens) for tokens in result.accepted_tokens)
        expected_tokens = tuple(
            _target_path_tokens(target, row, count)
            for row, count in zip(selection.selected_rows, selection.accepted_counts, strict=True)
        )
        if accepted_tokens != expected_tokens:
            raise ValueError("accepted_tokens must match selected target verify paths")
        max_depths = tuple(
            max(
                (
                    target.draft_depths[row]
                    for row in target.candidate_rows
                    if target.row_to_request[row] == request_id and target.active_mask[row]
                ),
                default=0,
            )
            for request_id in target.request_ids
        )
        return cls(
            request_ids=target.request_ids,
            accepted_counts=selection.accepted_counts,
            accepted_tokens=accepted_tokens,
            commit_rows=selection.selected_rows,
            commit_tokens=selection.selected_tokens,
            commit_positions=selection.selected_positions,
            full_accept=tuple(
                accepted == max_depth
                for accepted, max_depth in zip(selection.accepted_counts, max_depths, strict=True)
            ),
            mode=target.mode,
        )


def _target_path_tokens(target: TargetVerifyBatch, selected_row: int, accepted_count: int) -> tuple[int, ...]:
    if accepted_count == 0:
        if selected_row not in target.root_rows:
            raise ValueError("zero accepted candidates must select a root row")
        return ()
    root_rows = set(target.root_rows)
    row = int(selected_row)
    path_rows: list[int] = []
    while row not in root_rows:
        if row < 0 or row >= target.rows:
            raise ValueError("selected target row is outside the target batch")
        path_rows.append(row)
        row = target.parent_rows[row]
    path_rows.reverse()
    if len(path_rows) != accepted_count:
        raise ValueError("selected target path length must match accepted_count")
    if any(not target.active_mask[row] for row in path_rows):
        raise ValueError("selected target path contains inactive rows")
    return tuple(target.tokens[row] for row in path_rows)


@runtime_checkable
class DraftModel(Protocol):
    """Protocol for MTP/EAGLE3/DFlash/Medusa/Lookahead draft providers."""

    def propose(self, request_ids: Sequence[int], *, max_draft_tokens: int) -> DraftBatch: ...


@runtime_checkable
class Verifier(Protocol):
    """Protocol for target-model verification over a root+candidate batch."""

    def verify(self, batch: TargetVerifyBatch) -> AcceptResult: ...


__all__ = [
    "AcceptResult",
    "DraftBatch",
    "DraftModel",
    "TargetAcceptSummary",
    "TargetCommitSelection",
    "TargetVerifyBatch",
    "Verifier",
]
