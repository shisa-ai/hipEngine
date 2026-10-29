"""Typed observation and policy seam for request-owned MTP diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


@dataclass(frozen=True, slots=True)
class MtpBudgetCycleResult:
    """Completed-cycle observation published after commit and draft repair."""

    cycle: int
    budget: int
    accepted_count: int
    visible_tokens: int
    cycle_wall_ms: float
    full_accept: bool

    def __post_init__(self) -> None:
        if int(self.cycle) <= 0:
            raise ValueError("cycle must be positive")
        if int(self.budget) <= 0:
            raise ValueError("budget must be positive")
        if int(self.accepted_count) < 0 or int(self.accepted_count) > int(self.budget):
            raise ValueError("accepted_count must be within the selected budget")
        if int(self.visible_tokens) <= 0:
            raise ValueError("visible_tokens must be positive")
        if not math.isfinite(float(self.cycle_wall_ms)) or float(self.cycle_wall_ms) < 0.0:
            raise ValueError("cycle_wall_ms must be finite and non-negative")
        if bool(self.full_accept) != (int(self.accepted_count) == int(self.budget)):
            raise ValueError("full_accept must match accepted_count == budget")


class MtpBudgetPolicy(Protocol):
    """Duck-typed request-owned policy consumed by the GGUF MTP loop."""

    def start_request(self, *, request_id: int, max_budget: int) -> None: ...

    def choose_budget(
        self,
        *,
        cycle: int,
        max_budget: int,
        remaining_decode: int,
    ) -> int: ...

    def record_cycle(self, result: MtpBudgetCycleResult) -> None: ...

    def summary(self) -> dict[str, object]: ...


__all__ = [
    "MtpBudgetCycleResult",
    "MtpBudgetPolicy",
]
