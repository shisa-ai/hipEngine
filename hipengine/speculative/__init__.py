"""Speculative decoding plugin interfaces."""

from hipengine.speculative.buffers import (
    TargetVerifyBufferOwner,
    TargetVerifyBufferSpec,
    TargetVerifyScratchHandle,
    TargetVerifyScratchSpec,
)
from hipengine.speculative.dflash import (
    DFLASH_CHAIN_CANDIDATE_BUDGETS,
    DFlashChainCompiler,
    DFlashDraftProvider,
    DFlashDraftRequest,
    compile_dflash_chain,
)
from hipengine.speculative.dflash_drafter import (
    DFlashRootQueryPlan,
    DFlashRootQueryRequest,
    draft_batch_from_topk,
    project_dflash_target_hidden_bf16,
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
from hipengine.speculative.ladder import (
    TargetVerifyLadderMismatch,
    TargetVerifyLadderStageComparison,
    TargetVerifyLayerLadderResult,
    TargetVerifyStageSnapshot,
    TargetVerifyStateRows,
    compare_target_verify_ladder,
    synthetic_chain_target_verify_ladder,
    synthetic_chain_target_verify_snapshots,
)

__all__ = [
    "TargetVerifyBufferOwner",
    "TargetVerifyBufferSpec",
    "TargetVerifyScratchHandle",
    "TargetVerifyScratchSpec",
    "DFLASH_CHAIN_CANDIDATE_BUDGETS",
    "DFlashChainCompiler",
    "DFlashDraftProvider",
    "DFlashDraftRequest",
    "DFlashRootQueryPlan",
    "DFlashRootQueryRequest",
    "compile_dflash_chain",
    "draft_batch_from_topk",
    "project_dflash_target_hidden_bf16",
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
    "TargetVerifyLadderMismatch",
    "TargetVerifyLadderStageComparison",
    "TargetVerifyLayerLadderResult",
    "TargetVerifyStageSnapshot",
    "TargetVerifyStateRows",
    "compare_target_verify_ladder",
    "synthetic_chain_target_verify_ladder",
    "synthetic_chain_target_verify_snapshots",
]
