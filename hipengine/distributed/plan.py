"""Immutable single-host tensor-parallel plan identity.

The plan is resolved once, before weight allocation, and is the only place that
knows the ordered rank-to-device mapping, the communication dtype/algorithm, and
the topology fingerprint. It deliberately does not know model-specific shard
geometry: model/quant plugins resolve per-layer partitions against the plan and
reject degrees they cannot serve.

Design rules (docs/QWEN38-27B-GFX1100-TP2.md):
  * ordered unique devices, never a pair-specific ``1 - rank`` mapping;
  * deterministic serialization and hash, so two processes cannot disagree;
  * N=1 resolves without a communicator and preserves existing TP1 behavior;
  * unsupported degrees fail before any allocation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from hipengine.core.device import Device

SCHEMA_VERSION = 1

#: Communication dtypes the transport layer can enqueue. Names are the plan's
#: stable identifiers, not backend enums.
SUPPORTED_COMM_DTYPES: tuple[str, ...] = ("fp32", "fp16", "bf16")

#: Collective implementations this build can resolve.
SUPPORTED_ALGORITHMS: tuple[str, ...] = ("rccl", "peer", "mock")

#: Rank roles. One control owner decides sampling/commit; the draft owner runs
#: the small sequential MTP block in the initial design.
RANK_ROLES: tuple[str, ...] = ("worker", "control", "draft")


class PlanError(ValueError):
    """Raised when a distributed plan is internally inconsistent or unsupported."""


@dataclass(frozen=True)
class RankSpec:
    """One participating rank and the physical device it owns."""

    rank: int
    device: Device
    role: str = "worker"

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise PlanError("rank must be non-negative")
        if self.device.kind != "hip":
            raise PlanError(f"rank {self.rank} requires a hip device, got {self.device.kind!r}")
        if self.role not in RANK_ROLES:
            raise PlanError(f"rank {self.rank} has unknown role {self.role!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "device": str(self.device), "role": self.role}


@dataclass(frozen=True)
class DistributedPlan:
    """Resolved immutable plan identity for one TP group."""

    ranks: tuple[RankSpec, ...]
    hidden_size: int
    comm_dtype: str = "fp32"
    algorithm: str = "rccl"
    topology_fingerprint: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.ranks:
            raise PlanError("a distributed plan requires at least one rank")
        expected = list(range(len(self.ranks)))
        if [spec.rank for spec in self.ranks] != expected:
            raise PlanError("ranks must be contiguous and ordered from 0")
        devices = [spec.device for spec in self.ranks]
        if len(set(devices)) != len(devices):
            raise PlanError("each rank must own a unique device")
        if int(self.hidden_size) <= 0:
            raise PlanError("hidden_size must be positive")
        if self.comm_dtype not in SUPPORTED_COMM_DTYPES:
            raise PlanError(
                f"unsupported communication dtype {self.comm_dtype!r}; "
                f"expected one of: {', '.join(SUPPORTED_COMM_DTYPES)}"
            )
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise PlanError(
                f"unsupported collective algorithm {self.algorithm!r}; "
                f"expected one of: {', '.join(SUPPORTED_ALGORITHMS)}"
            )
        if int(self.schema_version) != SCHEMA_VERSION:
            raise PlanError(f"unsupported plan schema version {self.schema_version}")

    # -- identity -----------------------------------------------------------

    @property
    def world_size(self) -> int:
        return len(self.ranks)

    @property
    def is_single_rank(self) -> bool:
        return len(self.ranks) == 1

    @property
    def control_rank(self) -> int:
        for spec in self.ranks:
            if spec.role == "control":
                return spec.rank
        return 0

    @property
    def draft_rank(self) -> int | None:
        for spec in self.ranks:
            if spec.role == "draft":
                return spec.rank
        return None

    def rank_spec(self, rank: int) -> RankSpec:
        try:
            return self.ranks[int(rank)]
        except (IndexError, ValueError) as error:
            raise PlanError(f"rank {rank!r} is outside this plan's 0..{self.world_size - 1}") from error

    def device(self, rank: int) -> Device:
        return self.rank_spec(rank).device

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "world_size": self.world_size,
            "hidden_size": int(self.hidden_size),
            "comm_dtype": self.comm_dtype,
            "algorithm": self.algorithm,
            "topology_fingerprint": self.topology_fingerprint,
            "ranks": [spec.to_dict() for spec in self.ranks],
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def plan_hash(self) -> str:
        """Stable identity hash; includes topology, degree, roles and dtype."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def with_roles(self, *, control_rank: int = 0, draft_rank: int | None = None) -> "DistributedPlan":
        """Return a copy with explicit control/draft owners (single control owner)."""

        if not 0 <= int(control_rank) < self.world_size:
            raise PlanError(f"control rank {control_rank} is outside this plan")
        if draft_rank is not None and not 0 <= int(draft_rank) < self.world_size:
            raise PlanError(f"draft rank {draft_rank} is outside this plan")
        if draft_rank is not None and int(draft_rank) == int(control_rank):
            raise PlanError("draft rank must differ from the control rank")
        ranks = tuple(
            replace(spec, role="control" if spec.rank == control_rank else ("draft" if spec.rank == draft_rank else "worker"))
            for spec in self.ranks
        )
        return replace(self, ranks=ranks)

    # -- construction -------------------------------------------------------

    @classmethod
    def resolve(
        cls,
        devices: Sequence[Device | str | int],
        *,
        hidden_size: int,
        comm_dtype: str = "fp32",
        algorithm: str = "rccl",
        topology_fingerprint: str = "",
        control_rank: int = 0,
        draft_rank: int | None = None,
    ) -> "DistributedPlan":
        """Resolve an ordered device list into a validated plan.

        ``devices`` is the rank order: ``devices[0]`` is rank 0. Plain integers
        are interpreted as HIP indices so callers can pass ``[0, 1]`` directly.
        """

        if not devices:
            raise PlanError("device list must contain at least one device")
        specs: list[RankSpec] = []
        for rank, raw in enumerate(devices):
            device = raw if isinstance(raw, Device) else Device("hip", int(raw))
            specs.append(RankSpec(rank=rank, device=device))
        plan = cls(
            ranks=tuple(specs),
            hidden_size=int(hidden_size),
            comm_dtype=str(comm_dtype),
            algorithm=str(algorithm),
            topology_fingerprint=str(topology_fingerprint),
        )
        return plan.with_roles(control_rank=control_rank, draft_rank=draft_rank)


def payload_bytes(*, rows: int, hidden_size: int, dtype: str) -> int:
    """Return the byte payload of one full-hidden row block."""

    if int(rows) < 0:
        raise PlanError("rows must be non-negative")
    return int(rows) * int(hidden_size) * dtype_bytes(dtype)


def dtype_bytes(dtype: str) -> int:
    try:
        return {"fp32": 4, "fp16": 2, "bf16": 2}[str(dtype)]
    except KeyError as error:
        raise PlanError(f"unsupported communication dtype {dtype!r}") from error


def plan_from_mapping(payload: Mapping[str, Any]) -> DistributedPlan:
    """Rebuild a plan from :meth:`DistributedPlan.to_dict` output (round-trip)."""

    if not isinstance(payload, Mapping):
        raise PlanError("plan payload must be a mapping")
    try:
        ranks = tuple(
            RankSpec(rank=int(entry["rank"]), device=Device.parse(str(entry["device"])), role=str(entry.get("role", "worker")))
            for entry in payload["ranks"]
        )
    except (KeyError, TypeError) as error:
        raise PlanError(f"malformed plan payload: {error}") from error
    return DistributedPlan(
        ranks=ranks,
        hidden_size=int(payload["hidden_size"]),
        comm_dtype=str(payload.get("comm_dtype", "fp32")),
        algorithm=str(payload.get("algorithm", "rccl")),
        topology_fingerprint=str(payload.get("topology_fingerprint", "")),
        schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
    )


def device_ids(devices: Iterable[Device | str | int]) -> tuple[Device, ...]:
    """Normalize a device iterable without resolving a plan (used by screens)."""

    normalized: list[Device] = []
    for raw in devices:
        if isinstance(raw, Device):
            normalized.append(raw)
        elif isinstance(raw, str):
            normalized.append(Device.parse(raw))
        else:
            normalized.append(Device("hip", int(raw)))
    return tuple(normalized)
