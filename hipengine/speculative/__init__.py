"""Speculative decoding plugin interfaces."""

from hipengine.speculative.interfaces import (
    AcceptResult,
    DraftBatch,
    DraftModel,
    TargetAcceptSummary,
    TargetCommitPlan,
    TargetCommitSelection,
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
    "TargetVerifyBatch",
    "TargetVerifyBuffers",
    "Verifier",
]
