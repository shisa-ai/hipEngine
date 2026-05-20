"""Speculative decoding interfaces shared by MTP/EAGLE3/DFlash-style plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.dispatch.batch import WorkItem, WorkKind


def _validate_unique_request_ids(request_ids: Sequence[int]) -> None:
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request_ids must be unique")


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
        _validate_unique_request_ids(self.request_ids)
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
        _validate_unique_request_ids(self.request_ids)
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
        _validate_unique_request_ids(self.request_ids)
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

    def accept_from_top1(
        self,
        target_top1: Sequence[int],
        *,
        transaction_id: int | None = None,
        remaining_decode: Sequence[int] | None = None,
    ) -> "AcceptResult":
        """CPU oracle for device accept-summary kernels.

        ``target_top1[row]`` is the target model's selected next token for the
        prefix ending at ``row``.  The oracle follows matching draft edges from
        each request root, returns the accepted draft prefix, records the final
        row selected for commit, and emits the target correction/bonus token
        when the caller's remaining decode budget has room for it.
        """

        top1 = tuple(int(token) for token in target_top1)
        if len(top1) != self.rows:
            raise ValueError("target_top1 must align with target verify rows")
        if any(token < 0 for token in top1):
            raise ValueError("target_top1 tokens must be non-negative")
        budgets = None if remaining_decode is None else tuple(int(count) for count in remaining_decode)
        if budgets is not None:
            if len(budgets) != len(self.request_ids):
                raise ValueError("remaining_decode must align with request_ids")
            if any(count < 0 for count in budgets):
                raise ValueError("remaining_decode must be non-negative")

        child_rows: dict[int, list[int]] = {row: [] for row in range(self.rows)}
        for row in self.candidate_rows:
            if self.active_mask[row]:
                child_rows[self.parent_rows[row]].append(row)

        accepted_counts: list[int] = []
        accepted_tokens: list[tuple[int, ...]] = []
        selected_rows: list[int] = []
        next_tokens: list[int | None] = []
        for index, (request_id, root_row) in enumerate(zip(self.request_ids, self.root_rows, strict=True)):
            budget = None if budgets is None else budgets[index]
            row = root_row
            request_tokens: list[int] = []
            while budget is None or len(request_tokens) < budget:
                matches = [
                    child
                    for child in child_rows[row]
                    if self.row_to_request[child] == request_id and self.tokens[child] == top1[row]
                ]
                if not matches:
                    break
                if len(matches) > 1:
                    raise ValueError("target_top1 matches multiple candidate rows")
                row = matches[0]
                request_tokens.append(self.tokens[row])
            selected_rows.append(row)
            accepted_counts.append(len(request_tokens))
            accepted_tokens.append(tuple(request_tokens))
            if budget is not None and len(request_tokens) >= budget:
                next_tokens.append(None)
            else:
                next_tokens.append(top1[row])

        return AcceptResult(
            request_ids=self.request_ids,
            accepted_counts=tuple(accepted_counts),
            accepted_tokens=tuple(accepted_tokens),
            transaction_id=transaction_id,
            selected_candidate_rows=tuple(selected_rows),
            next_tokens=tuple(next_tokens),
        )


@dataclass(frozen=True, slots=True)
class TargetVerifyBuffers:
    """Device-buffer ABI descriptor for one target verification replay."""

    request_ids: tuple[int, ...]
    rows: int
    candidate_rows: int
    token_ids: Tensor
    positions: Tensor
    parent_rows: Tensor
    draft_depths: Tensor
    row_to_request: Tensor
    active_mask: Tensor
    target_top1: Tensor
    accepted_counts: Tensor
    commit_rows: Tensor
    commit_tokens: Tensor
    commit_positions: Tensor
    next_tokens: Tensor | None = None
    full_accept: Tensor | None = None
    committed_output_ids: Tensor | None = None
    committed_output_lengths: Tensor | None = None
    mode: str = "verify_chain"
    transaction_id: int | None = None
    candidate_counts: tuple[int, ...] | None = None
    draft_depth: int | None = None
    tree_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.request_ids:
            raise ValueError("TargetVerifyBuffers must contain at least one request")
        _validate_unique_request_ids(self.request_ids)
        if self.transaction_id is not None and self.transaction_id < 0:
            raise ValueError("transaction_id must be non-negative")
        if self.draft_depth is not None and self.draft_depth < 0:
            raise ValueError("draft_depth must be non-negative")
        if self.rows <= 0:
            raise ValueError("rows must be positive")
        if self.candidate_rows <= 0 or self.candidate_rows > self.rows:
            raise ValueError("candidate_rows must be positive and no larger than rows")
        if self.candidate_counts is not None:
            if len(self.candidate_counts) != len(self.request_ids):
                raise ValueError("candidate_counts must align with request_ids")
            if any(count < 0 for count in self.candidate_counts):
                raise ValueError("candidate_counts must be non-negative")
            if sum(self.candidate_counts) != self.candidate_rows:
                raise ValueError("candidate_counts must sum to candidate_rows")
        if self.tree_shape is not None:
            if len(self.tree_shape) != self.candidate_rows:
                raise ValueError("tree_shape must align with candidate_rows")
            if any(parent < 0 for parent in self.tree_shape):
                raise ValueError("tree_shape entries must be non-negative")
        row_tensors = (
            self.token_ids,
            self.positions,
            self.parent_rows,
            self.draft_depths,
            self.row_to_request,
            self.active_mask,
            self.target_top1,
        )
        summary_tensors = (self.accepted_counts, self.commit_rows, self.commit_tokens, self.commit_positions)
        optional_summary_tensors = tuple(
            tensor for tensor in (self.next_tokens, self.committed_output_lengths) if tensor is not None
        )
        optional_device_tensors = tuple(
            tensor for tensor in (self.full_accept, self.committed_output_ids) if tensor is not None
        )
        for tensor in row_tensors:
            if tensor.shape != (self.rows,):
                raise ValueError("row tensors must have shape (rows,)")
        for tensor in (*summary_tensors, *optional_summary_tensors):
            if tensor.shape != (len(self.request_ids),):
                raise ValueError("summary tensors must have shape (request_count,)")
        if self.full_accept is not None and self.full_accept.shape != (len(self.request_ids),):
            raise ValueError("full_accept tensor must have shape (request_count,)")
        if self.committed_output_ids is not None:
            if self.committed_output_ids.ndim != 2 or self.committed_output_ids.shape[0] != len(self.request_ids):
                raise ValueError("committed_output_ids tensor must have shape (request_count, output_stride)")
            if self.committed_output_ids.shape[1] <= 0:
                raise ValueError("committed_output_ids output_stride must be positive")
        for tensor in (*row_tensors, *summary_tensors, *optional_summary_tensors, *optional_device_tensors):
            if tensor.device != self.token_ids.device:
                raise ValueError("target verify buffers must live on one device")
        for tensor in (self.token_ids, self.positions, self.parent_rows, self.draft_depths, self.row_to_request, self.target_top1, *summary_tensors, *optional_summary_tensors):
            if tensor.dtype not in {DType.INT32, DType.INT64}:
                raise ValueError("target verify integer buffers must be int32 or int64")
        if self.committed_output_ids is not None and self.committed_output_ids.dtype not in {DType.INT32, DType.INT64}:
            raise ValueError("committed_output_ids buffer must be int32 or int64")
        if self.active_mask.dtype != DType.BOOL:
            raise ValueError("active_mask buffer must be bool")
        if self.full_accept is not None and self.full_accept.dtype != DType.BOOL:
            raise ValueError("full_accept buffer must be bool")
        if self.mode not in {"verify_chain", "verify_tree"}:
            raise ValueError("mode must be verify_chain or verify_tree")

    @classmethod
    def for_batch(
        cls,
        batch: TargetVerifyBatch,
        *,
        token_ids: Tensor,
        positions: Tensor,
        parent_rows: Tensor,
        draft_depths: Tensor,
        row_to_request: Tensor,
        active_mask: Tensor,
        target_top1: Tensor,
        accepted_counts: Tensor,
        commit_rows: Tensor,
        commit_tokens: Tensor,
        commit_positions: Tensor,
        next_tokens: Tensor | None = None,
        full_accept: Tensor | None = None,
        committed_output_ids: Tensor | None = None,
        committed_output_lengths: Tensor | None = None,
        transaction_id: int | None = None,
        candidate_counts: Sequence[int] | None = None,
        draft_depth: int | None = None,
        tree_shape: Sequence[int] | None = None,
    ) -> "TargetVerifyBuffers":
        return cls(
            request_ids=batch.request_ids,
            rows=batch.rows,
            candidate_rows=batch.candidate_count,
            token_ids=token_ids,
            positions=positions,
            parent_rows=parent_rows,
            draft_depths=draft_depths,
            row_to_request=row_to_request,
            active_mask=active_mask,
            target_top1=target_top1,
            accepted_counts=accepted_counts,
            commit_rows=commit_rows,
            commit_tokens=commit_tokens,
            commit_positions=commit_positions,
            next_tokens=next_tokens,
            full_accept=full_accept,
            committed_output_ids=committed_output_ids,
            committed_output_lengths=committed_output_lengths,
            mode=batch.mode,
            transaction_id=transaction_id,
            candidate_counts=batch.candidate_counts if candidate_counts is None else tuple(int(count) for count in candidate_counts),
            draft_depth=batch.draft_depth if draft_depth is None else int(draft_depth),
            tree_shape=batch.tree_shape if tree_shape is None else tuple(int(parent) for parent in tree_shape),
        )

    @property
    def request_count(self) -> int:
        return len(self.request_ids)

    @property
    def device(self):
        return self.token_ids.device


@dataclass(frozen=True, slots=True)
class AcceptResult:
    """Verifier accept/reject result per live request."""

    request_ids: tuple[int, ...]
    accepted_counts: tuple[int, ...]
    accepted_tokens: tuple[tuple[int, ...], ...]
    transaction_id: int | None = None
    selected_candidate_rows: tuple[int | None, ...] | None = None
    next_tokens: tuple[int | None, ...] | None = None

    def __post_init__(self) -> None:
        if not self.request_ids:
            raise ValueError("AcceptResult must contain at least one request")
        _validate_unique_request_ids(self.request_ids)
        if self.transaction_id is not None and self.transaction_id < 0:
            raise ValueError("transaction_id must be non-negative")
        if self.selected_candidate_rows is not None:
            if len(self.selected_candidate_rows) != len(self.request_ids):
                raise ValueError("selected_candidate_rows must align with request_ids")
            if any(row is not None and row < 0 for row in self.selected_candidate_rows):
                raise ValueError("selected_candidate_rows must be non-negative")
        if self.next_tokens is not None:
            if len(self.next_tokens) != len(self.request_ids):
                raise ValueError("next_tokens must align with request_ids")
            if any(token is not None and token < 0 for token in self.next_tokens):
                raise ValueError("next_tokens must be non-negative")
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
    next_tokens: tuple[int | None, ...] | None = None
    candidate_counts: tuple[int, ...] | None = None
    transaction_id: int | None = None
    draft_depth: int | None = None
    tree_shape: tuple[int, ...] | None = None
    mode: str = "verify_chain"

    def __post_init__(self) -> None:
        if not self.request_ids:
            raise ValueError("TargetAcceptSummary must contain at least one request")
        _validate_unique_request_ids(self.request_ids)
        if self.transaction_id is not None and self.transaction_id < 0:
            raise ValueError("transaction_id must be non-negative")
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
        if self.next_tokens is not None:
            if len(self.next_tokens) != len(self.request_ids):
                raise ValueError("next_tokens must align with request_ids")
            if any(token is not None and token < 0 for token in self.next_tokens):
                raise ValueError("next_tokens must be non-negative")
        for count, tokens in zip(self.accepted_counts, self.accepted_tokens, strict=True):
            if count != len(tokens):
                raise ValueError("accepted_counts must match accepted_tokens lengths")
            if any(token < 0 for token in tokens):
                raise ValueError("accepted token ids must be non-negative")
        if self.candidate_counts is not None:
            if len(self.candidate_counts) != len(self.request_ids):
                raise ValueError("candidate_counts must align with request_ids")
            if any(count < 0 for count in self.candidate_counts):
                raise ValueError("candidate_counts must be non-negative")
            for accepted, available in zip(self.accepted_counts, self.candidate_counts, strict=True):
                if accepted > available:
                    raise ValueError("accepted_counts cannot exceed candidate_counts")
        if self.draft_depth is not None and self.draft_depth < 0:
            raise ValueError("draft_depth must be non-negative")
        if self.tree_shape is not None:
            if self.candidate_counts is not None and len(self.tree_shape) != sum(self.candidate_counts):
                raise ValueError("tree_shape must align with candidate_counts")
            if any(parent < 0 for parent in self.tree_shape):
                raise ValueError("tree_shape entries must be non-negative")
        if self.mode not in {"verify_chain", "verify_tree"}:
            raise ValueError("mode must be verify_chain or verify_tree")

    @classmethod
    def from_gpu_payload(
        cls,
        target: TargetVerifyBatch,
        payload: dict[str, tuple[int, ...] | tuple[bool, ...]],
    ) -> "TargetAcceptSummary":
        """Build a summary from the GPU accept-summary kernel output.

        Reconstructs ``accepted_tokens`` by walking parent rows from the
        commit row back to the root, so the result is semantically identical
        to the CPU oracle path.
        """
        accepted_counts = payload["accepted_counts"]
        commit_rows = payload["commit_rows"]
        commit_tokens = payload["commit_tokens"]
        commit_positions = payload["commit_positions"]
        next_tokens_raw = payload.get("next_tokens", ())
        next_tokens: tuple[int | None, ...] | None = tuple(
            None if token < 0 else int(token) for token in next_tokens_raw
        ) if next_tokens_raw else None
        full_accept = payload["full_accept"]

        def _reconstruct(row: int) -> tuple[int, ...]:
            tokens: list[int] = []
            current = int(row)
            while 0 <= current < len(target.parent_rows):
                parent = target.parent_rows[current]
                if parent < 0:
                    break
                tokens.append(int(target.tokens[current]))
                current = int(parent)
            tokens.reverse()
            return tuple(tokens)

        accepted_tokens = tuple(_reconstruct(int(cr)) for cr in commit_rows)

        return cls(
            request_ids=target.request_ids,
            accepted_counts=accepted_counts,
            accepted_tokens=accepted_tokens,
            commit_rows=commit_rows,
            commit_tokens=commit_tokens,
            commit_positions=commit_positions,
            full_accept=full_accept,
            next_tokens=next_tokens,
            candidate_counts=target.candidate_counts,
            draft_depth=target.draft_depth,
            tree_shape=target.tree_shape,
            mode=target.mode,
        )

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
        result_selected_rows = None if result.selected_candidate_rows is None else tuple(result.selected_candidate_rows)
        if selected_candidate_rows is None:
            selected_rows = result_selected_rows
        else:
            selected_rows = tuple(selected_candidate_rows)
            if result_selected_rows is not None and result_selected_rows != selected_rows:
                raise ValueError("selected_candidate_rows must match accept result")
        selection = target.select_commit_rows(
            result.accepted_counts,
            selected_candidate_rows=selected_rows,
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
            next_tokens=None if result.next_tokens is None else tuple(result.next_tokens),
            candidate_counts=target.candidate_counts,
            transaction_id=result.transaction_id,
            draft_depth=target.draft_depth,
            tree_shape=target.tree_shape,
            mode=target.mode,
        )


@dataclass(frozen=True, slots=True)
class TargetCommitPlan:
    """Validated host commit contract for a target verification transaction."""

    transaction_id: int
    request_ids: tuple[int, ...]
    accepted_counts: tuple[int, ...]
    commit_rows: tuple[int, ...]
    commit_tokens: tuple[int, ...]
    commit_positions: tuple[int, ...]
    next_tokens: tuple[int | None, ...] | None = None
    candidate_counts: tuple[int, ...] | None = None
    draft_depth: int | None = None
    tree_shape: tuple[int, ...] | None = None
    mode: str = "verify_chain"

    def __post_init__(self) -> None:
        if self.transaction_id < 0:
            raise ValueError("transaction_id must be non-negative")
        if not self.request_ids:
            raise ValueError("TargetCommitPlan must contain at least one request")
        _validate_unique_request_ids(self.request_ids)
        lengths = (len(self.accepted_counts), len(self.commit_rows), len(self.commit_tokens), len(self.commit_positions))
        if any(length != len(self.request_ids) for length in lengths):
            raise ValueError("commit plan fields must align with request_ids")
        if any(count < 0 for count in self.accepted_counts):
            raise ValueError("accepted_counts must be non-negative")
        if any(row < 0 for row in self.commit_rows):
            raise ValueError("commit_rows must be non-negative")
        if any(token < 0 for token in self.commit_tokens):
            raise ValueError("commit_tokens must be non-negative")
        if any(position < 0 for position in self.commit_positions):
            raise ValueError("commit_positions must be non-negative")
        if self.next_tokens is not None:
            if len(self.next_tokens) != len(self.request_ids):
                raise ValueError("next_tokens must align with request_ids")
            if any(token is not None and token < 0 for token in self.next_tokens):
                raise ValueError("next_tokens must be non-negative")
        if self.candidate_counts is not None:
            if len(self.candidate_counts) != len(self.request_ids):
                raise ValueError("candidate_counts must align with request_ids")
            if any(count < 0 for count in self.candidate_counts):
                raise ValueError("candidate_counts must be non-negative")
            for accepted, available in zip(self.accepted_counts, self.candidate_counts, strict=True):
                if accepted > available:
                    raise ValueError("accepted_counts cannot exceed candidate_counts")
        if self.draft_depth is not None and self.draft_depth < 0:
            raise ValueError("draft_depth must be non-negative")
        if self.tree_shape is not None:
            if self.candidate_counts is not None and len(self.tree_shape) != sum(self.candidate_counts):
                raise ValueError("tree_shape must align with candidate_counts")
            if any(parent < 0 for parent in self.tree_shape):
                raise ValueError("tree_shape entries must be non-negative")
        if self.mode not in {"verify_chain", "verify_tree"}:
            raise ValueError("mode must be verify_chain or verify_tree")

    @classmethod
    def from_summary(cls, summary: TargetAcceptSummary, transaction) -> "TargetCommitPlan":
        request_ids = tuple(int(request_id) for request_id in getattr(transaction, "request_ids"))
        if request_ids != summary.request_ids:
            raise ValueError("transaction request_ids must match target accept summary")
        if bool(getattr(transaction, "committed", False)):
            raise ValueError("cannot build a commit plan for an already committed transaction")
        if bool(getattr(transaction, "rolled_back", False)):
            raise ValueError("cannot build a commit plan for a rolled-back transaction")
        transaction_id = int(getattr(transaction, "transaction_id"))
        if summary.transaction_id is not None and summary.transaction_id != transaction_id:
            raise ValueError("target accept summary transaction_id must match transaction")
        candidate_counts = getattr(transaction, "candidate_counts", None)
        counts = None if candidate_counts is None else tuple(int(count) for count in candidate_counts)
        summary_candidate_counts = getattr(summary, "candidate_counts", None)
        if summary_candidate_counts is not None:
            expected_counts = tuple(int(count) for count in summary_candidate_counts)
            if counts != expected_counts:
                raise ValueError("transaction candidate_counts must match target accept summary")
        role = str(getattr(transaction, "role", summary.mode))
        if role.startswith("WorkKind."):
            role = role.rsplit(".", 1)[-1].lower()
        if role not in {"verify_chain", "verify_tree"}:
            raise ValueError("transaction role must be verify_chain or verify_tree")
        if role != summary.mode:
            raise ValueError("transaction role must match target accept summary mode")
        return cls(
            transaction_id=transaction_id,
            request_ids=summary.request_ids,
            accepted_counts=summary.accepted_counts,
            commit_rows=summary.commit_rows,
            commit_tokens=summary.commit_tokens,
            commit_positions=summary.commit_positions,
            next_tokens=summary.next_tokens,
            candidate_counts=counts,
            draft_depth=summary.draft_depth,
            tree_shape=summary.tree_shape,
            mode=summary.mode,
        )

    @property
    def kv_accept_counts(self) -> tuple[int, ...]:
        return self.accepted_counts


@dataclass(frozen=True, slots=True)
class TargetStateCommitBuffers:
    """Device-buffer ABI descriptor for committing verified target state rows."""

    request_ids: tuple[int, ...]
    transaction_id: int
    accepted_counts: Tensor
    commit_rows: Tensor
    commit_positions: Tensor
    parent_rows: Tensor | None = None
    linear_state_src: Tensor | None = None
    linear_state_dst: Tensor | None = None
    kv_rows_src: Tensor | None = None
    kv_rows_dst: Tensor | None = None
    hidden_taps_src: Tensor | None = None
    hidden_taps_dst: Tensor | None = None
    next_tokens_src: Tensor | None = None
    committed_output_ids_src: Tensor | None = None
    committed_output_lengths_src: Tensor | None = None
    output_ids_dst: Tensor | None = None
    output_lengths_dst: Tensor | None = None
    last_positions_dst: Tensor | None = None
    context_lengths_dst: Tensor | None = None
    mode: str = "verify_chain"

    def __post_init__(self) -> None:
        if not self.request_ids:
            raise ValueError("TargetStateCommitBuffers must contain at least one request")
        _validate_unique_request_ids(self.request_ids)
        if self.transaction_id < 0:
            raise ValueError("transaction_id must be non-negative")
        count = len(self.request_ids)
        summary_tensors = (self.accepted_counts, self.commit_rows, self.commit_positions)
        for tensor in summary_tensors:
            if tensor.shape != (count,):
                raise ValueError("state commit summary tensors must have shape (request_count,)")
            if tensor.dtype not in {DType.INT32, DType.INT64}:
                raise ValueError("state commit summary tensors must be int32 or int64")
            if tensor.device != self.commit_rows.device:
                raise ValueError("state commit buffers must live on one device")
        if self.parent_rows is not None:
            if self.parent_rows.ndim != 1 or self.parent_rows.shape[0] <= 0:
                raise ValueError("parent_rows must be a non-empty row tensor")
            if self.parent_rows.dtype not in {DType.INT32, DType.INT64}:
                raise ValueError("parent_rows must be int32 or int64")
            if self.parent_rows.device != self.device:
                raise ValueError("parent_rows must live on the state commit device")
        self._validate_optional_pair("linear_state", self.linear_state_src, self.linear_state_dst, count)
        self._validate_optional_pair("kv_rows", self.kv_rows_src, self.kv_rows_dst, count, min_dst_rows=0)
        if self.kv_rows_src is not None and self.parent_rows is None:
            raise ValueError("parent_rows are required when committing KV rows")
        self._validate_hidden_taps_pair(count)
        self._validate_output_ring(count)
        for name, tensor in (
            ("last_positions_dst", self.last_positions_dst),
            ("context_lengths_dst", self.context_lengths_dst),
        ):
            if tensor is None:
                continue
            if tensor.shape != (count,):
                raise ValueError(f"{name} must have shape (request_count,)")
            if tensor.dtype not in {DType.INT32, DType.INT64}:
                raise ValueError(f"{name} must be int32 or int64")
            if tensor.device != self.device:
                raise ValueError(f"{name} must live on the state commit device")
        if self.mode not in {"verify_chain", "verify_tree"}:
            raise ValueError("mode must be verify_chain or verify_tree")

    @classmethod
    def for_plan(
        cls,
        plan: TargetCommitPlan,
        *,
        accepted_counts: Tensor,
        commit_rows: Tensor,
        commit_positions: Tensor,
        parent_rows: Tensor | None = None,
        linear_state_src: Tensor | None = None,
        linear_state_dst: Tensor | None = None,
        kv_rows_src: Tensor | None = None,
        kv_rows_dst: Tensor | None = None,
        hidden_taps_src: Tensor | None = None,
        hidden_taps_dst: Tensor | None = None,
        next_tokens_src: Tensor | None = None,
        committed_output_ids_src: Tensor | None = None,
        committed_output_lengths_src: Tensor | None = None,
        output_ids_dst: Tensor | None = None,
        output_lengths_dst: Tensor | None = None,
        last_positions_dst: Tensor | None = None,
        context_lengths_dst: Tensor | None = None,
    ) -> "TargetStateCommitBuffers":
        return cls(
            request_ids=plan.request_ids,
            transaction_id=plan.transaction_id,
            accepted_counts=accepted_counts,
            commit_rows=commit_rows,
            commit_positions=commit_positions,
            parent_rows=parent_rows,
            linear_state_src=linear_state_src,
            linear_state_dst=linear_state_dst,
            kv_rows_src=kv_rows_src,
            kv_rows_dst=kv_rows_dst,
            hidden_taps_src=hidden_taps_src,
            hidden_taps_dst=hidden_taps_dst,
            next_tokens_src=next_tokens_src,
            committed_output_ids_src=committed_output_ids_src,
            committed_output_lengths_src=committed_output_lengths_src,
            output_ids_dst=output_ids_dst,
            output_lengths_dst=output_lengths_dst,
            last_positions_dst=last_positions_dst,
            context_lengths_dst=context_lengths_dst,
            mode=plan.mode,
        )

    @property
    def request_count(self) -> int:
        return len(self.request_ids)

    @property
    def device(self):
        return self.commit_rows.device

    @property
    def has_linear_state(self) -> bool:
        return self.linear_state_src is not None

    @property
    def has_kv_rows(self) -> bool:
        return self.kv_rows_src is not None

    @property
    def has_hidden_taps(self) -> bool:
        return self.hidden_taps_src is not None

    @property
    def has_output_ring(self) -> bool:
        return self.committed_output_ids_src is not None

    @property
    def has_context_metadata(self) -> bool:
        return self.last_positions_dst is not None or self.context_lengths_dst is not None

    def _validate_optional_pair(
        self,
        name: str,
        src: Tensor | None,
        dst: Tensor | None,
        request_count: int,
        *,
        min_dst_rows: int | None = None,
    ) -> None:
        if (src is None) != (dst is None):
            raise ValueError(f"{name} buffers must be provided as a src/dst pair")
        if src is None or dst is None:
            return
        if src.device != self.device or dst.device != self.device:
            raise ValueError(f"{name} buffers must live on the state commit device")
        if src.dtype != dst.dtype:
            raise ValueError(f"{name} source and destination buffers must share dtype")
        if src.ndim == 0 or dst.ndim == 0:
            raise ValueError(f"{name} buffers must be row-major tensors")
        if src.shape[1:] != dst.shape[1:]:
            raise ValueError(f"{name} source and destination row tail shape must match")
        required_dst_rows = request_count if min_dst_rows is None else int(min_dst_rows)
        if src.shape[0] <= 0 or dst.shape[0] < required_dst_rows:
            raise ValueError(f"{name} buffers must contain enough rows")

    def _validate_hidden_taps_pair(self, request_count: int) -> None:
        if (self.hidden_taps_src is None) != (self.hidden_taps_dst is None):
            raise ValueError("hidden_taps buffers must be provided as a src/dst pair")
        if self.hidden_taps_src is None or self.hidden_taps_dst is None:
            return
        src = self.hidden_taps_src
        dst = self.hidden_taps_dst
        if src.device != self.device or dst.device != self.device:
            raise ValueError("hidden_taps buffers must live on the state commit device")
        if src.dtype != dst.dtype:
            raise ValueError("hidden_taps source and destination buffers must share dtype")
        if src.ndim != 3 or dst.ndim != 3:
            raise ValueError("hidden_taps buffers must have shape (tap_count, rows, hidden_size)")
        if src.shape[0] != dst.shape[0] or src.shape[2] != dst.shape[2]:
            raise ValueError("hidden_taps source and destination tap/hidden shapes must match")
        if src.shape[1] <= 0 or dst.shape[1] < request_count:
            raise ValueError("hidden_taps buffers must contain enough rows")

    def _validate_output_ring(self, request_count: int) -> None:
        output_tensors = (
            self.committed_output_ids_src,
            self.committed_output_lengths_src,
            self.output_ids_dst,
            self.output_lengths_dst,
        )
        if any(tensor is not None for tensor in output_tensors) and not all(tensor is not None for tensor in output_tensors):
            raise ValueError("output ring buffers must include ids/lengths source and destination tensors")
        if self.committed_output_ids_src is None:
            if self.next_tokens_src is not None:
                raise ValueError("next_tokens_src requires output ring buffers")
            return
        ids_src = self.committed_output_ids_src
        lengths_src = self.committed_output_lengths_src
        ids_dst = self.output_ids_dst
        lengths_dst = self.output_lengths_dst
        assert lengths_src is not None and ids_dst is not None and lengths_dst is not None
        for name, tensor in (
            ("committed_output_ids_src", ids_src),
            ("committed_output_lengths_src", lengths_src),
            ("output_ids_dst", ids_dst),
            ("output_lengths_dst", lengths_dst),
            ("next_tokens_src", self.next_tokens_src),
        ):
            if tensor is None:
                continue
            if tensor.device != self.device:
                raise ValueError(f"{name} must live on the state commit device")
            if tensor.dtype not in {DType.INT32, DType.INT64}:
                raise ValueError(f"{name} must be int32 or int64")
        if ids_src.ndim != 2 or ids_dst.ndim != 2:
            raise ValueError("output id buffers must be 2D request-major tensors")
        if ids_src.shape[0] != request_count or ids_dst.shape[0] != request_count:
            raise ValueError("output id buffers must have request_count rows")
        if ids_src.shape[1] <= 0 or ids_dst.shape[1] <= 0:
            raise ValueError("output id buffers must have a positive stride")
        if lengths_src.shape != (request_count,) or lengths_dst.shape != (request_count,):
            raise ValueError("output length buffers must have shape (request_count,)")
        if self.next_tokens_src is not None and self.next_tokens_src.shape != (request_count,):
            raise ValueError("next_tokens_src must have shape (request_count,)")


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
    "TargetCommitPlan",
    "TargetCommitSelection",
    "TargetStateCommitBuffers",
    "TargetVerifyBatch",
    "TargetVerifyBuffers",
    "Verifier",
]
