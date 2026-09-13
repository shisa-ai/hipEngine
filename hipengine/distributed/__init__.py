"""Torch-free single-host tensor-parallel infrastructure.

This package is the host-side plan/ownership/communication layer for the TP=N
architecture in ``docs/QWEN38-27B-GFX1100-TP2.md``. It does not enable public
tensor-parallel serving: model construction, ``LLM.generate``, and the server
still advertise world size 1 until the Packet 6 integration gates pass.

Nothing here imports ``torch``. ``librccl.so`` is loaded lazily by the RCCL
transport, and the CPU mock transport keeps plan/group-protocol tests runnable
without ROCm.
"""

from hipengine.distributed.context import DistributedContext, RankRuntime
from hipengine.distributed.plan import (
    RANK_ROLES,
    SCHEMA_VERSION,
    SUPPORTED_ALGORITHMS,
    SUPPORTED_COMM_DTYPES,
    DistributedPlan,
    PlanError,
    RankSpec,
    device_ids,
    dtype_bytes,
    payload_bytes,
    plan_from_mapping,
)
from hipengine.distributed.transport import (
    CollectiveKind,
    CollectiveRequest,
    CollectiveTransport,
    CommunicatorAbortedError,
    EnqueueRecorder,
    TimeoutError_,
    TransportError,
    TransportStateError,
    TransportUnavailableError,
)

__all__ = [
    "CollectiveKind",
    "CollectiveRequest",
    "CollectiveTransport",
    "CommunicatorAbortedError",
    "DistributedContext",
    "DistributedPlan",
    "EnqueueRecorder",
    "PlanError",
    "RANK_ROLES",
    "RankRuntime",
    "RankSpec",
    "SCHEMA_VERSION",
    "SUPPORTED_ALGORITHMS",
    "SUPPORTED_COMM_DTYPES",
    "TimeoutError_",
    "TransportError",
    "TransportStateError",
    "TransportUnavailableError",
    "device_ids",
    "dtype_bytes",
    "payload_bytes",
    "plan_from_mapping",
]
