"""Speculative decoding plugin interfaces."""

from hipengine.speculative.dflash import (
    DFLASH_CHAIN_CANDIDATE_BUDGETS,
    DFlashChainCompiler,
    DFlashDraftProvider,
    DFlashDraftRequest,
    compile_dflash_chain,
)
from hipengine.speculative.interfaces import (
    AcceptResult,
    DraftBatch,
    DraftModel,
    TargetAcceptSummary,
    TargetCommitPlan,
    TargetCommitSelection,
    TargetStateCommitBuffers,
    TargetVerifyBatch,
    TargetVerifyBuffers,
    Verifier,
)

__all__ = [
    "DFLASH_CHAIN_CANDIDATE_BUDGETS",
    "DFlashChainCompiler",
    "DFlashDraftProvider",
    "DFlashDraftRequest",
    "compile_dflash_chain",
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
