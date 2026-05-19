"""MTP speculative-draft metadata and provider boundary.

MTP proposal quality is model-specific, but its verifier-facing output is not:
plain chain MTP emits candidate rows ``[d1, d2, ... dB]`` and the shared target
verifier materializes ``[root, d1, ... dB]``.  This module intentionally contains
only provider-neutral chain compilation and the target-attached provider
protocol; Qwen3.5/Qwen3.6 tensor loading lives in :mod:`hipengine.loading.mtp`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, runtime_checkable

from hipengine.core.tensor import Tensor
from hipengine.speculative.chain import ChainDraftCompiler, ChainDraftRequest, compile_chain_draft
from hipengine.speculative.interfaces import DraftBatch

MTP_CHAIN_CANDIDATE_BUDGETS: tuple[int, ...] = (1, 2, 3, 5)
MtpDraftRequest = ChainDraftRequest


@dataclass(frozen=True, slots=True)
class MtpProposalContext:
    """Inputs a target-attached MTP provider needs for one proposal step.

    ``target_hidden`` is the committed target hidden/final-hidden row for the
    same root token(s).  Providers may keep extra MTP KV/state internally, but
    the returned ``DraftBatch`` remains candidate-only.
    """

    request_ids: tuple[int, ...]
    root_tokens: tuple[int, ...]
    root_positions: tuple[int, ...]
    target_hidden: Tensor | None = None

    def __post_init__(self) -> None:
        if not self.request_ids:
            raise ValueError("MTP proposal context must contain at least one request")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("request_ids must be unique")
        if len(self.root_tokens) != len(self.request_ids) or len(self.root_positions) != len(self.request_ids):
            raise ValueError("root tokens/positions must align with request_ids")
        if any(token < 0 for token in self.root_tokens):
            raise ValueError("root token ids must be non-negative")
        if any(position < 0 for position in self.root_positions):
            raise ValueError("root positions must be non-negative")
        if self.target_hidden is not None and self.target_hidden.ndim != 2:
            raise ValueError("target_hidden must be rank-2 [request_count, hidden_size]")
        if self.target_hidden is not None and self.target_hidden.shape[0] != len(self.request_ids):
            raise ValueError("target_hidden row count must match request_ids")


class MtpChainCompiler(ChainDraftCompiler):
    """MTP wrapper around the shared candidate-only chain compiler."""

    def __init__(self, candidate_budget: int, pad_token_id: int = 0) -> None:
        super().__init__(
            candidate_budget=candidate_budget,
            pad_token_id=pad_token_id,
            allowed_budgets=MTP_CHAIN_CANDIDATE_BUDGETS,
        )


MtpTokenGenerator = Callable[[MtpProposalContext, int], Sequence[Sequence[int]]]


@runtime_checkable
class MtpDraftProvider(Protocol):
    """Target-attached MTP provider boundary.

    Implementations run normalized token embedding + normalized target hidden ->
    MTP block(s) -> shared lm-head/top1 internally, then return a candidate-only
    ``DraftBatch``.  Verification/accept/commit are deliberately out of scope
    and must use the shared target verifier path.
    """

    def propose(self, context: MtpProposalContext, *, candidate_budget: int) -> DraftBatch:
        """Return candidate-only MTP chain rows for ``context.request_ids``."""
        ...


class Qwen35MtpDraftProvider:
    """Small provider shell that converts generated MTP tokens to DraftBatch.

    ``token_generator`` is the model-specific execution boundary.  A native
    implementation will run the target-attached MTP tensors and shared lm-head;
    tests can inject a deterministic generator to validate verifier-facing
    metadata without creating a fake verifier path.
    """

    def __init__(self, token_generator: MtpTokenGenerator, *, pad_token_id: int = 0) -> None:
        self._token_generator = token_generator
        self._pad_token_id = int(pad_token_id)
        if self._pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative")

    def propose(self, context: MtpProposalContext, *, candidate_budget: int) -> DraftBatch:
        token_rows = tuple(tuple(int(token) for token in row) for row in self._token_generator(context, int(candidate_budget)))
        if len(token_rows) != len(context.request_ids):
            raise ValueError("token generator must return one candidate row per request")
        requests: list[MtpDraftRequest] = []
        for index, row in enumerate(token_rows):
            if any(token < 0 for token in row):
                raise ValueError("MTP token generator returned a negative token id")
            requests.append(
                MtpDraftRequest(
                    request_id=int(context.request_ids[index]),
                    root_position=int(context.root_positions[index]),
                    candidate_tokens=tuple(row[: int(candidate_budget)]),
                    active_count=min(len(row), int(candidate_budget)),
                )
            )
        return compile_mtp_chain(requests, candidate_budget=int(candidate_budget), pad_token_id=self._pad_token_id)


def compile_mtp_chain(
    requests: Sequence[MtpDraftRequest],
    *,
    candidate_budget: int,
    pad_token_id: int = 0,
) -> DraftBatch:
    """Compile MTP chain requests into the shared ``DraftBatch`` layout."""

    return compile_chain_draft(
        requests,
        candidate_budget=candidate_budget,
        pad_token_id=pad_token_id,
        allowed_budgets=MTP_CHAIN_CANDIDATE_BUDGETS,
    )


class MissingMtpWeightsError(RuntimeError):
    """Raised when a target checkpoint does not carry target-attached MTP tensors."""


__all__ = [
    "MTP_CHAIN_CANDIDATE_BUDGETS",
    "MissingMtpWeightsError",
    "MtpChainCompiler",
    "MtpDraftProvider",
    "MtpDraftRequest",
    "MtpProposalContext",
    "MtpTokenGenerator",
    "Qwen35MtpDraftProvider",
    "compile_mtp_chain",
]
