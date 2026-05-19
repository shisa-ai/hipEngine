"""DFlash speculative-draft helpers.

This module is intentionally metadata-only: it turns drafter proposals into the
``DraftBatch`` topology consumed by the native verifier without allocating
PyTorch/HF objects or inserting target root rows.  Root rows are added only by
``TargetVerifyBatch.from_draft()``.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from hipengine.speculative.chain import ChainDraftCompiler, ChainDraftRequest, compile_chain_draft
from hipengine.speculative.interfaces import DraftBatch

DFLASH_CHAIN_CANDIDATE_BUDGETS: tuple[int, ...] = (2, 4, 8)


DFlashDraftRequest = ChainDraftRequest


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


class DFlashChainCompiler(ChainDraftCompiler):
    """DFlash wrapper around the provider-neutral chain compiler."""

    def __init__(self, candidate_budget: int, pad_token_id: int = 0) -> None:
        super().__init__(
            candidate_budget=candidate_budget,
            pad_token_id=pad_token_id,
            allowed_budgets=DFLASH_CHAIN_CANDIDATE_BUDGETS,
        )


def compile_dflash_chain(
    requests: Sequence[DFlashDraftRequest],
    *,
    candidate_budget: int,
    pad_token_id: int = 0,
) -> DraftBatch:
    """Convenience wrapper around ``DFlashChainCompiler``."""

    return compile_chain_draft(
        requests,
        candidate_budget=candidate_budget,
        pad_token_id=pad_token_id,
        allowed_budgets=DFLASH_CHAIN_CANDIDATE_BUDGETS,
    )


__all__ = [
    "DFLASH_CHAIN_CANDIDATE_BUDGETS",
    "DFlashChainCompiler",
    "DFlashDraftProvider",
    "DFlashDraftRequest",
    "compile_dflash_chain",
]
