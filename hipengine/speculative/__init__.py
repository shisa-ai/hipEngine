"""Speculative decoding plugin interfaces."""

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
