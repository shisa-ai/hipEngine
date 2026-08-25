"""Request-owned fixed and adaptive MTP candidate-budget policies.

Policies see only cycle-local economics and aggregate conditional acceptance.
They do not receive prompt text, token IDs, categories, or request content.
"""

from __future__ import annotations

from collections import Counter
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


class _RequestOwnedPolicy:
    def __init__(self) -> None:
        self._request_id: int | None = None
        self._max_budget: int | None = None

    def start_request(self, *, request_id: int, max_budget: int) -> None:
        budget = int(max_budget)
        if budget <= 0:
            raise ValueError("max_budget must be positive")
        if self._request_id is not None:
            raise RuntimeError(
                f"budget policy already owns request {self._request_id}; "
                f"cannot start request {int(request_id)}"
            )
        self._request_id = int(request_id)
        self._max_budget = budget

    def _check_choice(self, *, budget: int, max_budget: int) -> int:
        if self._request_id is None or self._max_budget is None:
            raise RuntimeError("budget policy request has not started")
        selected = int(budget)
        cycle_max = int(max_budget)
        if selected <= 0:
            raise ValueError("selected budget must be positive")
        if selected > cycle_max:
            raise ValueError(
                f"selected budget {selected} exceeds cycle maximum {cycle_max}"
            )
        if selected > self._max_budget:
            raise ValueError("selected budget exceeds request maximum")
        return selected


class MtpBudgetSequencePolicy(_RequestOwnedPolicy):
    """Deterministic diagnostic schedule used by transition-contract gates."""

    def __init__(self, budgets: tuple[int, ...]) -> None:
        super().__init__()
        values = tuple(int(value) for value in budgets)
        if not values or any(value <= 0 for value in values):
            raise ValueError("budget sequence must contain positive budgets")
        self.budgets = values
        self._decisions: list[int] = []
        self._results: list[MtpBudgetCycleResult] = []

    def choose_budget(
        self,
        *,
        cycle: int,
        max_budget: int,
        remaining_decode: int,
    ) -> int:
        _ = remaining_decode
        index = (int(cycle) - 1) % len(self.budgets)
        selected = self._check_choice(
            budget=min(self.budgets[index], int(max_budget)),
            max_budget=max_budget,
        )
        self._decisions.append(selected)
        return selected

    def record_cycle(self, result: MtpBudgetCycleResult) -> None:
        self._results.append(result)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "mtp_budget_sequence",
            "request_id": self._request_id,
            "request_max_budget": self._max_budget,
            "schedule": list(self.budgets),
            "decisions": list(self._decisions),
            "cycle_count": len(self._results),
            "results": [
                {
                    "cycle": row.cycle,
                    "budget": row.budget,
                    "accepted_count": row.accepted_count,
                    "visible_tokens": row.visible_tokens,
                    "cycle_wall_ms": row.cycle_wall_ms,
                    "full_accept": row.full_accept,
                }
                for row in self._results
            ],
        }


@dataclass(frozen=True, slots=True)
class MtpAdaptiveBudgetConfig:
    """EMA and hysteresis controls for B1/B2/B3 online selection."""

    budgets: tuple[int, ...] = (1, 2, 3)
    ema_alpha: float = 0.25
    switch_margin: float = 0.02
    exploration_samples_per_budget: int = 1

    def __post_init__(self) -> None:
        values = tuple(int(value) for value in self.budgets)
        if not values or values != tuple(sorted(set(values))) or any(value <= 0 for value in values):
            raise ValueError("budgets must be unique positive values in ascending order")
        if not 0.0 < float(self.ema_alpha) <= 1.0:
            raise ValueError("ema_alpha must be within (0, 1]")
        if not math.isfinite(float(self.switch_margin)) or float(self.switch_margin) < 0.0:
            raise ValueError("switch_margin must be finite and non-negative")
        if int(self.exploration_samples_per_budget) <= 0:
            raise ValueError("exploration_samples_per_budget must be positive")
        object.__setattr__(self, "budgets", values)


class MtpAdaptiveBudgetPolicy(_RequestOwnedPolicy):
    """Content-agnostic EMA scorer over independent fixed-budget buckets."""

    def __init__(self, config: MtpAdaptiveBudgetConfig | None = None) -> None:
        super().__init__()
        self.config = config or MtpAdaptiveBudgetConfig()
        self._conditional_acceptance: list[float | None] = [
            None for _ in range(max(self.config.budgets))
        ]
        self._conditional_samples = [0 for _ in self._conditional_acceptance]
        self._wall_ms: dict[int, float] = {}
        self._wall_samples: Counter[int] = Counter()
        self._decision_counts: Counter[int] = Counter()
        self._results: list[MtpBudgetCycleResult] = []
        self._current_budget: int | None = None
        self._last_scores: dict[int, float] = {}

    def start_request(self, *, request_id: int, max_budget: int) -> None:
        super().start_request(request_id=request_id, max_budget=max_budget)
        if not any(budget <= int(max_budget) for budget in self.config.budgets):
            raise ValueError("adaptive policy has no budget within request maximum")

    def choose_budget(
        self,
        *,
        cycle: int,
        max_budget: int,
        remaining_decode: int,
    ) -> int:
        _ = cycle, remaining_decode
        available = tuple(
            budget
            for budget in self.config.budgets
            if budget <= int(max_budget)
            and (self._max_budget is None or budget <= self._max_budget)
        )
        if not available:
            raise ValueError("adaptive policy has no qualified cycle budget")
        for budget in available:
            if self._wall_samples[budget] < self.config.exploration_samples_per_budget:
                return self._publish_choice(budget, max_budget=max_budget)

        scores = {budget: self._score(budget) for budget in available}
        self._last_scores = dict(scores)
        best = max(available, key=lambda budget: (scores[budget], budget))
        current = self._current_budget
        if current in scores:
            current_score = scores[current]
            if scores[best] <= current_score * (1.0 + self.config.switch_margin):
                best = int(current)
        return self._publish_choice(best, max_budget=max_budget)

    def _publish_choice(self, budget: int, *, max_budget: int) -> int:
        selected = self._check_choice(budget=budget, max_budget=max_budget)
        self._current_budget = selected
        self._decision_counts[selected] += 1
        return selected

    def record_cycle(self, result: MtpBudgetCycleResult) -> None:
        budget = int(result.budget)
        if budget not in self.config.budgets:
            raise ValueError("cycle result budget is outside adaptive policy")
        alpha = float(self.config.ema_alpha)
        prior_wall = self._wall_ms.get(budget)
        self._wall_ms[budget] = (
            float(result.cycle_wall_ms)
            if prior_wall is None
            else (1.0 - alpha) * prior_wall + alpha * float(result.cycle_wall_ms)
        )
        self._wall_samples[budget] += 1
        accepted = int(result.accepted_count)
        for depth in range(1, budget + 1):
            if accepted < depth - 1:
                break
            outcome = 1.0 if accepted >= depth else 0.0
            index = depth - 1
            prior = self._conditional_acceptance[index]
            self._conditional_acceptance[index] = (
                outcome if prior is None else (1.0 - alpha) * prior + alpha * outcome
            )
            self._conditional_samples[index] += 1
        self._results.append(result)

    def _score(self, budget: int) -> float:
        wall = self._wall_ms.get(int(budget))
        if wall is None or wall <= 0.0:
            return 0.0
        visible = 1.0
        path_probability = 1.0
        for depth in range(1, int(budget) + 1):
            estimate = self._conditional_acceptance[depth - 1]
            path_probability *= 0.0 if estimate is None else float(estimate)
            visible += path_probability
        return visible / wall

    def summary(self) -> dict[str, object]:
        scores = {
            budget: self._score(budget)
            for budget in self.config.budgets
            if self._wall_samples[budget] > 0
        }
        return {
            "kind": "mtp_adaptive_budget_v1",
            "request_id": self._request_id,
            "request_max_budget": self._max_budget,
            "config": {
                "budgets": list(self.config.budgets),
                "ema_alpha": self.config.ema_alpha,
                "switch_margin": self.config.switch_margin,
                "exploration_samples_per_budget": self.config.exploration_samples_per_budget,
            },
            "cycle_count": len(self._results),
            "decision_counts": {
                f"B{budget}": self._decision_counts[budget]
                for budget in self.config.budgets
                if self._decision_counts[budget]
            },
            "conditional_acceptance": list(self._conditional_acceptance),
            "conditional_samples": list(self._conditional_samples),
            "wall_ms_ema": {f"B{budget}": value for budget, value in sorted(self._wall_ms.items())},
            "wall_samples": {
                f"B{budget}": self._wall_samples[budget]
                for budget in self.config.budgets
                if self._wall_samples[budget]
            },
            "scores": {f"B{budget}": value for budget, value in sorted(scores.items())},
            "current_budget": None if self._current_budget is None else f"B{self._current_budget}",
        }


__all__ = [
    "MtpAdaptiveBudgetConfig",
    "MtpAdaptiveBudgetPolicy",
    "MtpBudgetCycleResult",
    "MtpBudgetPolicy",
    "MtpBudgetSequencePolicy",
]
