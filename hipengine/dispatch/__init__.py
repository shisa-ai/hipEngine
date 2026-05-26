"""Dispatch, batching, and fusion planning."""

from hipengine.dispatch.batch import ActiveBatch, BatchShapeKey, BatchSlot, RequestState, SlotMove, WorkItem, WorkKind
from hipengine.dispatch.fusion import BoundKernel, FusionPlanner, KernelPlanStep, resolve_plan
from hipengine.dispatch.kv import (
    KVKernelSelection,
    PagedAttnDecodeKind,
    PagedKVWriteKind,
    bind_paged_attn_decode,
    bind_paged_kv_write,
    plan_paged_attn_decode,
    plan_paged_kv_write,
    resolve_paged_attn_decode,
    resolve_paged_kv_write,
)
from hipengine.dispatch.projection import (
    ProjectionDispatchCandidate,
    ProjectionDispatchDecision,
    ProjectionDispatchEvidence,
    ProjectionKernelSelection,
    plan_projection_dispatch,
)
from hipengine.dispatch.sampling import BatchSamplerDispatchDecision, BatchSamplerMode, plan_batch_sampler_dispatch

__all__ = [
    "ActiveBatch",
    "BatchSamplerDispatchDecision",
    "BatchSamplerMode",
    "BatchShapeKey",
    "BatchSlot",
    "BoundKernel",
    "FusionPlanner",
    "KVKernelSelection",
    "KernelPlanStep",
    "PagedAttnDecodeKind",
    "PagedKVWriteKind",
    "ProjectionDispatchCandidate",
    "ProjectionDispatchDecision",
    "ProjectionDispatchEvidence",
    "ProjectionKernelSelection",
    "RequestState",
    "SlotMove",
    "WorkItem",
    "WorkKind",
    "bind_paged_attn_decode",
    "bind_paged_kv_write",
    "plan_paged_attn_decode",
    "plan_paged_kv_write",
    "plan_projection_dispatch",
    "plan_batch_sampler_dispatch",
    "resolve_paged_attn_decode",
    "resolve_paged_kv_write",
    "resolve_plan",
]
