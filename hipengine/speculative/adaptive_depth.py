"""Acceptance-triggered online candidate-depth adaptation for block DFlash.

The context-budget sweep (2026-11-08) showed the optimum depth is
acceptance-driven per prompt, not context-driven: the natural suite peaks at
budget 4 while high-acceptance prompts peak higher, and the curve is flat near
each peak. A static default cannot capture the per-prompt optimum; this
controller probes adjacent depths only when the rolling acceptance signal
says the current depth is likely suboptimal:

- sustained high acceptance -> probe one depth up (more proposals may commit);
- sustained low acceptance  -> probe one depth down (fewer wasted verify rows);
- mid-band acceptance       -> stay (probing would only cost cycles).

A probe runs the neighbor depth for a few cycles, then keeps it only if its
smoothed tokens/ms beats the incumbent by a margin; otherwise it reverts and
a cooldown prevents flapping.

Output safety: block DFlash verify is target-corrected greedy, so the depth
changes how many tokens commit per cycle, never which tokens. Timing-driven
budget decisions therefore cannot change generated IDs (same-build seeded
repeats stay identical); only cycle counts and throughput move. The
controller is host-only and per-request, mirroring ``adaptive_budget.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AdaptiveDepthConfig:
    """Hysteresis for :class:`AdaptiveDepthController`."""

    min_budget: int = 2
    max_budget: int = 7
    decision_interval: int = 8
    probe_cycles: int = 4
    probe_cooldown: int = 16
    high_acceptance: float = 0.8
    low_acceptance: float = 0.2
    rate_margin: float = 0.03
    ewma_alpha: float = 0.25
    min_samples: int = 3

    def __post_init__(self) -> None:
        if self.min_budget < 1:
            raise ValueError("min_budget must be positive")
        if self.max_budget < self.min_budget:
            raise ValueError("max_budget must be >= min_budget")
        if self.decision_interval <= 0 or self.probe_cycles <= 0 or self.probe_cooldown < 0:
            raise ValueError("decision intervals must be positive")
        if not 0.0 < self.low_acceptance < self.high_acceptance <= 1.0:
            raise ValueError("acceptance thresholds must satisfy 0 < low < high <= 1")
        if self.rate_margin < 0.0 or not 0.0 < self.ewma_alpha <= 1.0 or self.min_samples <= 0:
            raise ValueError("invalid rate_margin/ewma_alpha/min_samples")


@dataclass
class _DepthStats:
    samples: int = 0
    tokens_ewma: float = 0.0
    ms_ewma: float = 0.0
    acceptance_ewma: float = 0.0

    def rate(self) -> float | None:
        if self.samples == 0 or self.ms_ewma <= 0.0:
            return None
        return self.tokens_ewma / self.ms_ewma


class AdaptiveDepthController:
    """Per-request depth policy; ``enabled=False`` is observational static."""

    def __init__(
        self,
        *,
        enabled: bool,
        initial_budget: int,
        max_budget_cap: int | None = None,
        config: AdaptiveDepthConfig | None = None,
    ) -> None:
        self.config = config or AdaptiveDepthConfig()
        cap = self.config.max_budget
        if max_budget_cap is not None:
            cap = min(cap, int(max_budget_cap))
        self.min_budget = min(self.config.min_budget, max(1, cap))
        self.max_budget = max(self.min_budget, cap)
        self.enabled = bool(enabled)
        self.initial_budget = max(self.min_budget, min(int(initial_budget), self.max_budget))
        self.current_budget = self.initial_budget
        self._stats: dict[int, _DepthStats] = {}
        self._cycle = 0
        self._probe_budget: int | None = None
        self._probe_remaining = 0
        self._probe_return_budget: int | None = None
        self._cooldowns: dict[int, int] = {}
        self._decisions: list[dict[str, Any]] = []

    def begin_cycle(self) -> int:
        """Return the candidate depth for the cycle about to run."""

        self._cycle += 1
        if not self.enabled:
            return self.initial_budget
        for budget in list(self._cooldowns):
            self._cooldowns[budget] -= 1
            if self._cooldowns[budget] <= 0:
                del self._cooldowns[budget]
        if self._probe_remaining > 0:
            self._probe_remaining -= 1
            if self._probe_remaining == 0:
                self._finish_probe()
            return self._probe_budget if self._probe_budget is not None else self.current_budget
        if self._cycle % self.config.decision_interval == 0:
            self._maybe_start_probe()
        return self.current_budget if self._probe_budget is None else self._probe_budget

    def record_cycle(
        self,
        *,
        budget: int,
        visible_tokens: int,
        cycle_wall_ms: float,
        accepted_tokens: int,
        proposed_tokens: int,
    ) -> None:
        """Update the smoothed stats for the depth that actually ran."""

        if visible_tokens <= 0 or cycle_wall_ms < 0 or proposed_tokens <= 0:
            raise ValueError("visible_tokens/proposed_tokens must be positive, cycle_wall_ms non-negative")
        stats = self._stats.setdefault(int(budget), _DepthStats())
        alpha = self.config.ewma_alpha
        acceptance = min(1.0, max(0.0, accepted_tokens / proposed_tokens))
        if stats.samples == 0:
            stats.tokens_ewma = float(visible_tokens)
            stats.ms_ewma = float(cycle_wall_ms)
            stats.acceptance_ewma = acceptance
        else:
            stats.tokens_ewma += alpha * (visible_tokens - stats.tokens_ewma)
            stats.ms_ewma += alpha * (cycle_wall_ms - stats.ms_ewma)
            stats.acceptance_ewma += alpha * (acceptance - stats.acceptance_ewma)
        stats.samples += 1

    def _maybe_start_probe(self) -> None:
        stats = self._stats.get(self.current_budget)
        if stats is None or stats.samples < self.config.min_samples:
            return
        direction = 0
        if stats.acceptance_ewma >= self.config.high_acceptance and self.current_budget < self.max_budget:
            direction = 1
        elif stats.acceptance_ewma <= self.config.low_acceptance and self.current_budget > self.min_budget:
            direction = -1
        if direction == 0:
            return
        neighbor = self.current_budget + direction
        if neighbor in self._cooldowns:
            return
        self._probe_budget = neighbor
        self._probe_return_budget = self.current_budget
        self._probe_remaining = self.config.probe_cycles
        self._decisions.append({
            "cycle": self._cycle,
            "action": "probe",
            "from": self.current_budget,
            "to": neighbor,
            "acceptance_ewma": stats.acceptance_ewma,
        })

    def _finish_probe(self) -> None:
        probed = self._probe_budget
        incumbent = self._probe_return_budget
        self._probe_budget = None
        self._probe_return_budget = None
        if probed is None or incumbent is None:
            return
        probe_stats = self._stats.get(probed)
        base_stats = self._stats.get(incumbent)
        probe_rate = probe_stats.rate() if probe_stats else None
        base_rate = base_stats.rate() if base_stats else None
        keep = (
            probe_rate is not None
            and base_rate is not None
            and probe_stats.samples >= self.config.min_samples
            and probe_rate > base_rate * (1.0 + self.config.rate_margin)
        )
        if keep:
            self._decisions.append({
                "cycle": self._cycle, "action": "keep", "from": incumbent, "to": probed,
                "probe_rate": probe_rate, "base_rate": base_rate,
            })
            self.current_budget = probed
        else:
            self._decisions.append({
                "cycle": self._cycle, "action": "revert", "from": probed, "to": incumbent,
                "probe_rate": probe_rate, "base_rate": base_rate,
            })
            self.current_budget = incumbent
            self._cooldowns[probed] = self.config.probe_cooldown

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "initial_budget": self.initial_budget,
            "final_budget": self.current_budget,
            "cycles": self._cycle,
            "bounds": [self.min_budget, self.max_budget],
            "decisions": list(self._decisions),
            "stats": {
                str(budget): {
                    "samples": s.samples,
                    "tokens_per_ms": s.rate(),
                    "acceptance_ewma": s.acceptance_ewma,
                }
                for budget, s in sorted(self._stats.items())
            },
        }