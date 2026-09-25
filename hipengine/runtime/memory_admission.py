"""One admission budget over every resident consumer.

The KV-pool construction priced exactly one consumer -- the pool itself -- against
free device memory minus a hardcoded reserve. Every other resident consumer
(draft/provider KV, verifier scratch, captured graphs, retained prefix ownership)
was not priced at admission at all, so an overload was discovered later as a HIP
out-of-memory fault from somewhere in the middle of a request.

This module prices the consumers together and refuses *before* allocating, naming
the consumer whose edge was crossed. Nothing has been allocated when a refusal is
raised, so the engine is still healthy and the request can be refused with a
capacity status rather than being reported as an internal fault.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

#: Headroom kept free of the admission arithmetic. The historical KV-pool check
#: used this same figure; it now covers every priced consumer instead of the pool
#: alone.
DEFAULT_ADMISSION_RESERVE_BYTES = 3 * 1024**3

#: API error code carried by a refusal. Distinct from ``engine_unavailable``: the
#: engine is healthy and the refusal is about capacity.
ADMISSION_REFUSAL_CODE = "insufficient_memory"


@dataclass(frozen=True, slots=True)
class MemoryConsumer:
    """One resident allocation that an admission decision must price."""

    name: str
    bytes: int

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("memory consumer needs a name")
        if int(self.bytes) < 0:
            raise ValueError(f"memory consumer {self.name!r} has negative bytes")


@dataclass(frozen=True, slots=True)
class MemoryAdmissionDecision:
    """The priced outcome of one admission decision."""

    admitted: bool
    reason: str
    refused_consumer: str | None
    required_bytes: int
    available_bytes: int
    consumers: tuple[MemoryConsumer, ...]

    @property
    def headroom_bytes(self) -> int:
        """Bytes left after every priced consumer; negative when refused."""

        return int(self.available_bytes) - int(self.required_bytes)

    def as_dict(self) -> dict[str, object]:
        return {
            "admitted": self.admitted,
            "reason": self.reason,
            "refused_consumer": self.refused_consumer,
            "required_bytes": int(self.required_bytes),
            "available_bytes": int(self.available_bytes),
            "headroom_bytes": self.headroom_bytes,
            "consumers": {item.name: int(item.bytes) for item in self.consumers},
        }


class MemoryAdmissionRefused(MemoryError):
    """A named overload refusal raised before any allocation.

    Subclasses ``MemoryError`` so existing handlers keep catching it, but carries
    the structure an API needs: the consumer that did not fit, the priced totals,
    and the error code to report.
    """

    code = ADMISSION_REFUSAL_CODE

    def __init__(self, decision: MemoryAdmissionDecision) -> None:
        self.decision = decision
        self.refused_consumer = decision.refused_consumer
        self.required_bytes = int(decision.required_bytes)
        self.available_bytes = int(decision.available_bytes)
        super().__init__(decision.reason)


def price_memory_admission(
    *,
    free_bytes: int,
    consumers: Iterable[MemoryConsumer] | Mapping[str, int],
    reserve_bytes: int = DEFAULT_ADMISSION_RESERVE_BYTES,
) -> MemoryAdmissionDecision:
    """Price every consumer in declaration order against free memory minus reserve.

    The first consumer that does not fit the remaining headroom is the named
    refusal reason, so the message points at the allocation that has to shrink
    rather than at a total. Ordering is the caller's: it is the order the
    consumers are allocated in, which is also the order an unpriced overload
    would have been discovered in.
    """

    if isinstance(consumers, Mapping):
        priced = tuple(MemoryConsumer(name=str(k), bytes=int(v)) for k, v in consumers.items())
    else:
        priced = tuple(consumers)
    reserve = int(reserve_bytes)
    if reserve < 0:
        raise ValueError("admission reserve must not be negative")
    available = int(free_bytes) - reserve
    required = 0
    refused: str | None = None
    for item in priced:
        required += int(item.bytes)
        if refused is None and required > available:
            refused = item.name
    if refused is None:
        return MemoryAdmissionDecision(
            admitted=True,
            reason="fits_admission_budget",
            refused_consumer=None,
            required_bytes=required,
            available_bytes=available,
            consumers=priced,
        )
    return MemoryAdmissionDecision(
        admitted=False,
        reason=(
            f"{refused} does not fit the admission budget: "
            f"{required} bytes priced across {len(priced)} consumers, "
            f"{available} available after a {reserve}-byte reserve"
        ),
        refused_consumer=refused,
        required_bytes=required,
        available_bytes=available,
        consumers=priced,
    )


def require_memory_admission(**kwargs: object) -> MemoryAdmissionDecision:
    """Price an admission decision and raise the named refusal when it fails."""

    decision = price_memory_admission(**kwargs)  # type: ignore[arg-type]
    if not decision.admitted:
        raise MemoryAdmissionRefused(decision)
    return decision
