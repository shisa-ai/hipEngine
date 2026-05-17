"""DFlash speculative-draft helpers.

This module is intentionally metadata-only: it turns drafter proposals into the
``DraftBatch`` topology consumed by the native verifier without allocating
PyTorch/HF objects or inserting target root rows.  Root rows are added only by
``TargetVerifyBatch.from_draft()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from hipengine.speculative.interfaces import DraftBatch

DFLASH_CHAIN_CANDIDATE_BUDGETS: tuple[int, ...] = (2, 4, 8)


@dataclass(frozen=True, slots=True)
class DFlashDraftRequest:
    """Candidate-only DFlash proposal for one live request.

    ``candidate_tokens`` excludes the already-committed root token.  The
    compiler uses ``root_position`` as the parent position for depth-1 rows;
    candidate positions are materialized later by ``TargetVerifyBatch`` as
    ``parent_position + 1``.
    """

    request_id: int
    root_position: int
    candidate_tokens: tuple[int, ...]
    active_count: int | None = None

    def __post_init__(self) -> None:
        if self.root_position < 0:
            raise ValueError("root_position must be non-negative")
        if any(token < 0 for token in self.candidate_tokens):
            raise ValueError("candidate token ids must be non-negative")
        if self.active_count is not None:
            if self.active_count < 0:
                raise ValueError("active_count must be non-negative")
            if self.active_count > len(self.candidate_tokens):
                raise ValueError("active_count cannot exceed candidate token count")

    @classmethod
    def from_root_prefixed(
        cls,
        *,
        request_id: int,
        root_position: int,
        token_ids: Sequence[int],
        expected_root_token: int | None = None,
        active_count: int | None = None,
    ) -> "DFlashDraftRequest":
        """Adapt legacy/root-prefixed drafter output to candidate-only form.

        Parent DFlash harnesses often returned ``[root, draft_1, ...]``.  The
        hipEngine verifier boundary carries only draft candidates, so this
        adapter strips the leading root and optionally verifies it.
        """

        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("root-prefixed DFlash output must contain a root token")
        if expected_root_token is not None and tokens[0] != int(expected_root_token):
            raise ValueError("root-prefixed DFlash output does not match expected root token")
        return cls(
            request_id=int(request_id),
            root_position=int(root_position),
            candidate_tokens=tokens[1:],
            active_count=active_count,
        )


@runtime_checkable
class DFlashDraftProvider(Protocol):
    """Provider boundary for native/Python drafter implementations.

    Implementations may use root tokens and target hidden taps internally, but
    the returned requests must contain candidate tokens only.
    """

    def propose_chain(
        self,
        *,
        request_ids: Sequence[int],
        root_tokens: Sequence[int],
        root_positions: Sequence[int],
        candidate_budget: int,
    ) -> Sequence[DFlashDraftRequest]:
        """Return candidate-only chain proposals for ``request_ids``."""
        ...


@dataclass(frozen=True, slots=True)
class DFlashChainCompiler:
    """Compile top-k=1 DFlash chain proposals into ``DraftBatch`` rows.

    ``candidate_budget`` is the number of draft candidate rows per request and
    excludes target root rows.  Supported fixed buckets are ``2``, ``4``, and
    ``8`` so downstream buffer owners and graph buckets can rely on stable
    shapes.
    """

    candidate_budget: int
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if self.candidate_budget not in DFLASH_CHAIN_CANDIDATE_BUDGETS:
            raise ValueError(f"candidate_budget must be one of {DFLASH_CHAIN_CANDIDATE_BUDGETS}")
        if self.pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative")

    def compile(self, requests: Sequence[DFlashDraftRequest]) -> DraftBatch:
        """Return a candidate-only ``DraftBatch`` in request-major chain order."""

        draft_requests = tuple(requests)
        if not draft_requests:
            raise ValueError("at least one DFlash draft request is required")

        request_ids = tuple(int(request.request_id) for request in draft_requests)
        candidate_tokens: list[int] = []
        parent_positions: list[int] = []
        draft_depths: list[int] = []
        row_to_request: list[int] = []
        active_mask: list[bool] = []

        for request in draft_requests:
            tokens = tuple(int(token) for token in request.candidate_tokens[: self.candidate_budget])
            active_count = len(tokens) if request.active_count is None else int(request.active_count)
            if active_count > self.candidate_budget:
                raise ValueError("active_count cannot exceed candidate_budget")
            if active_count > len(tokens):
                raise ValueError("active rows must have candidate tokens")

            for depth in range(1, self.candidate_budget + 1):
                token_index = depth - 1
                token = tokens[token_index] if token_index < len(tokens) else self.pad_token_id
                candidate_tokens.append(token)
                parent_positions.append(int(request.root_position) + token_index)
                draft_depths.append(depth)
                row_to_request.append(int(request.request_id))
                active_mask.append(depth <= active_count)

        return DraftBatch(
            request_ids=request_ids,
            candidate_tokens=tuple(candidate_tokens),
            parent_positions=tuple(parent_positions),
            draft_depths=tuple(draft_depths),
            row_to_request=tuple(row_to_request),
            active_mask=tuple(active_mask),
            mode="verify_chain",
        )


def compile_dflash_chain(
    requests: Sequence[DFlashDraftRequest],
    *,
    candidate_budget: int,
    pad_token_id: int = 0,
) -> DraftBatch:
    """Convenience wrapper around ``DFlashChainCompiler``."""

    return DFlashChainCompiler(candidate_budget=candidate_budget, pad_token_id=pad_token_id).compile(requests)


__all__ = [
    "DFLASH_CHAIN_CANDIDATE_BUDGETS",
    "DFlashChainCompiler",
    "DFlashDraftProvider",
    "DFlashDraftRequest",
    "compile_dflash_chain",
]
