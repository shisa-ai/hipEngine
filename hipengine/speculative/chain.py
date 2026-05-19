"""Shared chain-draft metadata compiler for speculative providers.

DFlash, MTP, EAGLE-style chain proposals, and simple lookahead providers all
share the same verifier-facing shape: candidate rows only.  The committed root
row is materialized by :class:`TargetVerifyBatch.from_draft` immediately before
target verification.  This module keeps that contract provider-neutral so MTP
does not need to fork DFlash's verifier/accept/commit path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from hipengine.speculative.interfaces import DraftBatch


@dataclass(frozen=True, slots=True)
class ChainDraftRequest:
    """Candidate-only chain proposal for one live request.

    ``candidate_tokens`` excludes the already-committed root token.  The
    compiler uses ``root_position`` as the parent position for the depth-1 row;
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
    ) -> "ChainDraftRequest":
        """Adapt legacy/root-prefixed output to candidate-only form."""

        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("root-prefixed output must contain a root token")
        if expected_root_token is not None and tokens[0] != int(expected_root_token):
            raise ValueError("root-prefixed output does not match expected root token")
        return cls(
            request_id=int(request_id),
            root_position=int(root_position),
            candidate_tokens=tokens[1:],
            active_count=active_count,
        )


@dataclass(frozen=True, slots=True)
class ChainDraftCompiler:
    """Compile chain proposals into provider-neutral ``DraftBatch`` rows."""

    candidate_budget: int
    pad_token_id: int = 0
    allowed_budgets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.candidate_budget <= 0:
            raise ValueError("candidate_budget must be positive")
        if self.allowed_budgets and self.candidate_budget not in self.allowed_budgets:
            raise ValueError(f"candidate_budget must be one of {self.allowed_budgets}")
        if self.pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative")

    def compile(self, requests: Sequence[ChainDraftRequest]) -> DraftBatch:
        """Return a candidate-only ``DraftBatch`` in request-major chain order."""

        draft_requests = tuple(requests)
        if not draft_requests:
            raise ValueError("at least one chain draft request is required")

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


def compile_chain_draft(
    requests: Sequence[ChainDraftRequest],
    *,
    candidate_budget: int,
    pad_token_id: int = 0,
    allowed_budgets: Sequence[int] = (),
) -> DraftBatch:
    """Convenience wrapper around ``ChainDraftCompiler``."""

    return ChainDraftCompiler(
        candidate_budget=candidate_budget,
        pad_token_id=pad_token_id,
        allowed_budgets=tuple(int(budget) for budget in allowed_budgets),
    ).compile(requests)


__all__ = [
    "ChainDraftCompiler",
    "ChainDraftRequest",
    "compile_chain_draft",
]
