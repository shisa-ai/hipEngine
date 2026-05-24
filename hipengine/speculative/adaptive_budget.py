"""Adaptive budget controller for speculative decoding.

The controller is deliberately host-only.  It decides whether the next decode
cycle should run speculative DFlash or fall back to a plain AR step based on a
bounded startup probe and a rolling profit signal measured in milliseconds:

    profit_ms = visible_tokens * ar_ms_per_token - cycle_wall_ms

Positive profit means the speculative cycle emitted more visible tokens than an
AR decode would have emitted in the same wall time.  Negative profit means the
cycle should have stayed on AR.  Adaptive mode starts in ``AR_PROBE`` by
default, so a losing prompt pays for one speculative cycle before locking to AR.
The state machine adds cooldown/probe hysteresis so borderline prompts do not
flap every cycle.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Deque, Literal

AdaptiveState = Literal["DFLASH", "AR_LOCKED", "AR_PROBE"]


@dataclass(frozen=True)
class AdaptiveBudgetDecision:
    """Decision returned at the start of a decode cycle."""

    cycle: int
    state: AdaptiveState
    use_dflash: bool
    reason: str


@dataclass(frozen=True)
class AdaptiveBudgetConfig:
    """Tunable hysteresis for :class:`AdaptiveBudgetController`."""

    profit_window_size: int = 8
    demote_after_cycles: int = 1
    demote_mean_profit_ms: float = -5.0
    initial_cooldown_cycles: int = 32
    retry_cooldown_cycles: int = 32
    probe_cycles: int = 1
    promote_mean_profit_ms: float = 0.5
    min_remaining_tokens_for_dflash: int = 0
    initial_probe_cycles: int = 1
    probe_min_amortization_tokens: int = 64

    def __post_init__(self) -> None:
        if self.profit_window_size <= 0:
            raise ValueError("profit_window_size must be positive")
        if self.demote_after_cycles <= 0:
            raise ValueError("demote_after_cycles must be positive")
        if self.demote_after_cycles > self.profit_window_size:
            raise ValueError("demote_after_cycles must fit inside profit_window_size")
        if self.initial_cooldown_cycles <= 0:
            raise ValueError("initial_cooldown_cycles must be positive")
        if self.retry_cooldown_cycles <= 0:
            raise ValueError("retry_cooldown_cycles must be positive")
        if self.probe_cycles <= 0:
            raise ValueError("probe_cycles must be positive")
        if self.min_remaining_tokens_for_dflash < 0:
            raise ValueError("min_remaining_tokens_for_dflash must be non-negative")
        if self.initial_probe_cycles < 0:
            raise ValueError("initial_probe_cycles must be non-negative")
        if self.probe_min_amortization_tokens < 0:
            raise ValueError("probe_min_amortization_tokens must be non-negative")


class AdaptiveBudgetController:
    """Hysteretic DFlash-vs-AR controller.

    ``enabled=False`` makes the controller observational only: decisions always
    choose DFlash, but cycle metrics are still logged in the summary.  This is
    useful for default-off validation and artifact compatibility.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        ar_decode_tok_s_estimate: float | None,
        config: AdaptiveBudgetConfig | None = None,
        max_cycle_log: int | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.config = config or AdaptiveBudgetConfig()
        if ar_decode_tok_s_estimate is None or ar_decode_tok_s_estimate <= 0:
            if self.enabled:
                raise ValueError("enabled adaptive budget requires a positive AR tok/s estimate")
            self.ar_decode_tok_s_estimate = None
            self.ar_ms_per_token = None
        else:
            self.ar_decode_tok_s_estimate = float(ar_decode_tok_s_estimate)
            self.ar_ms_per_token = 1000.0 / float(ar_decode_tok_s_estimate)
        self.max_cycle_log = max_cycle_log
        self.state: AdaptiveState = "AR_PROBE" if self.enabled and self.config.initial_probe_cycles > 0 else "DFLASH"
        self.cooldown_remaining = 0
        self.probe_remaining = self.config.initial_probe_cycles if self.state == "AR_PROBE" else 0
        self._profit_window: Deque[float] = deque(maxlen=self.config.profit_window_size)
        self._probe_profit: list[float] = []
        self._cycle_log: list[dict[str, Any]] = []
        self._transitions: list[dict[str, Any]] = []
        self._mode_counts: Counter[str] = Counter()
        self._decision_counts: Counter[str] = Counter()

    def begin_cycle(self, *, cycle: int, context_tokens: int, remaining_tokens: int, active_budget: int) -> AdaptiveBudgetDecision:
        """Return the mode decision for the next cycle.

        ``context_tokens`` is accepted for logging and future policies.  The
        current R3.1 controller combines a remaining-token horizon guard with a
        rolling profit window.
        """

        _ = context_tokens
        if not self.enabled:
            decision = AdaptiveBudgetDecision(cycle=int(cycle), state=self.state, use_dflash=True, reason="disabled")
        elif (
            active_budget > 0
            and self.config.min_remaining_tokens_for_dflash > 0
            and remaining_tokens < self.config.min_remaining_tokens_for_dflash
        ):
            decision = AdaptiveBudgetDecision(cycle=int(cycle), state=self.state, use_dflash=False, reason="remaining_tokens_guard")
        elif (
            active_budget > 0
            and self.state == "AR_PROBE"
            and self.config.probe_min_amortization_tokens > 0
            and remaining_tokens
            < self.config.min_remaining_tokens_for_dflash + self.config.probe_min_amortization_tokens
        ):
            decision = AdaptiveBudgetDecision(cycle=int(cycle), state=self.state, use_dflash=False, reason="probe_amortization_guard")
        elif self.state == "AR_LOCKED":
            decision = AdaptiveBudgetDecision(cycle=int(cycle), state=self.state, use_dflash=False, reason="cooldown")
        elif self.state == "AR_PROBE":
            decision = AdaptiveBudgetDecision(cycle=int(cycle), state=self.state, use_dflash=True, reason="probe")
        else:
            decision = AdaptiveBudgetDecision(cycle=int(cycle), state=self.state, use_dflash=True, reason="profit_window")
        self._decision_counts["dflash" if decision.use_dflash else "ar"] += 1
        return decision

    def record_dflash_cycle(
        self,
        decision: AdaptiveBudgetDecision,
        *,
        visible_tokens: int,
        cycle_wall_ms: float,
        accepted_tokens: int,
        active_budget: int,
        draft_ms: float | None = None,
        verify_ms: float | None = None,
        commit_ms: float | None = None,
        context_tokens: int | None = None,
    ) -> None:
        """Record a speculative cycle and update the state machine."""

        if visible_tokens <= 0:
            raise ValueError("visible_tokens must be positive")
        if cycle_wall_ms < 0:
            raise ValueError("cycle_wall_ms must be non-negative")
        profit_ms = self._profit_ms(visible_tokens=visible_tokens, cycle_wall_ms=cycle_wall_ms)
        state_before = self.state
        self._mode_counts["dflash"] += 1
        if profit_ms is not None:
            if self.enabled and decision.state == "AR_PROBE":
                self._probe_profit.append(float(profit_ms))
                self.probe_remaining = max(0, self.probe_remaining - 1)
                if self.probe_remaining == 0:
                    probe_mean = _mean(self._probe_profit)
                    if probe_mean > self.config.promote_mean_profit_ms:
                        self._profit_window.extend(self._probe_profit)
                        self._transition(
                            cycle=decision.cycle,
                            new_state="DFLASH",
                            reason=f"probe_mean_profit_ms={probe_mean:.3f} > {self.config.promote_mean_profit_ms:.3f}",
                        )
                    else:
                        self.cooldown_remaining = self.config.retry_cooldown_cycles
                        self._transition(
                            cycle=decision.cycle,
                            new_state="AR_LOCKED",
                            reason=f"probe_mean_profit_ms={probe_mean:.3f} <= {self.config.promote_mean_profit_ms:.3f}",
                        )
                    self._probe_profit.clear()
            else:
                self._profit_window.append(float(profit_ms))
                if self.enabled and len(self._profit_window) >= self.config.demote_after_cycles:
                    tail = list(self._profit_window)[-self.config.demote_after_cycles :]
                    tail_mean = _mean(tail)
                    if tail_mean < self.config.demote_mean_profit_ms:
                        self.cooldown_remaining = self.config.initial_cooldown_cycles
                        self._transition(
                            cycle=decision.cycle,
                            new_state="AR_LOCKED",
                            reason=f"mean_profit_last_{self.config.demote_after_cycles}={tail_mean:.3f} < {self.config.demote_mean_profit_ms:.3f}",
                        )
                        self._profit_window.clear()
        self._append_cycle_log(
            {
                "cycle": int(decision.cycle),
                "decision_state": decision.state,
                "state_before": state_before,
                "state_after": self.state,
                "mode_used": "DFLASH",
                "decision_reason": decision.reason,
                "context_tokens": context_tokens,
                "active_budget": int(active_budget),
                "accepted_tokens": int(accepted_tokens),
                "visible_tokens": int(visible_tokens),
                "cycle_wall_ms": float(cycle_wall_ms),
                "draft_ms": _optional_float(draft_ms),
                "verify_ms": _optional_float(verify_ms),
                "commit_ms": _optional_float(commit_ms),
                "profit_ms": _optional_float(profit_ms),
                "cooldown_remaining": int(self.cooldown_remaining),
                "probe_remaining": int(self.probe_remaining),
            }
        )

    def record_ar_cycle(
        self,
        decision: AdaptiveBudgetDecision | None,
        *,
        cycle: int,
        cycle_wall_ms: float,
        context_tokens: int | None = None,
        forced_reason: str | None = None,
        update_state: bool = True,
    ) -> None:
        """Record a plain AR cycle.

        ``update_state=False`` is used for terminal/forced AR cycles, such as
        no remaining speculative budget or a remaining-token horizon guard;
        those should not consume cooldown/probe accounting.
        """

        if cycle_wall_ms < 0:
            raise ValueError("cycle_wall_ms must be non-negative")
        state_before = self.state
        reason = forced_reason or (decision.reason if decision is not None else "forced_ar")
        self._mode_counts["ar"] += 1
        if self.enabled and update_state and self.state == "AR_LOCKED":
            self.cooldown_remaining = max(0, self.cooldown_remaining - 1)
            if self.cooldown_remaining == 0:
                self.probe_remaining = self.config.probe_cycles
                self._probe_profit.clear()
                self._transition(cycle=int(cycle), new_state="AR_PROBE", reason="cooldown_expired")
        self._append_cycle_log(
            {
                "cycle": int(cycle),
                "decision_state": decision.state if decision is not None else state_before,
                "state_before": state_before,
                "state_after": self.state,
                "mode_used": "AR",
                "decision_reason": reason,
                "context_tokens": context_tokens,
                "active_budget": 0,
                "accepted_tokens": 0,
                "visible_tokens": 1,
                "cycle_wall_ms": float(cycle_wall_ms),
                "draft_ms": 0.0,
                "verify_ms": float(cycle_wall_ms),
                "commit_ms": 0.0,
                "profit_ms": None,
                "cooldown_remaining": int(self.cooldown_remaining),
                "probe_remaining": int(self.probe_remaining),
            }
        )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serializable artifact summary."""

        profits = [float(row["profit_ms"]) for row in self._cycle_log if row.get("profit_ms") is not None]
        return {
            "mode": "on" if self.enabled else "off",
            "enabled": self.enabled,
            "state": self.state,
            "ar_decode_tok_s_estimate": _optional_float(self.ar_decode_tok_s_estimate),
            "ar_ms_per_token": _optional_float(self.ar_ms_per_token),
            "config": {
                "profit_window_size": self.config.profit_window_size,
                "demote_after_cycles": self.config.demote_after_cycles,
                "demote_mean_profit_ms": self.config.demote_mean_profit_ms,
                "initial_cooldown_cycles": self.config.initial_cooldown_cycles,
                "retry_cooldown_cycles": self.config.retry_cooldown_cycles,
                "probe_cycles": self.config.probe_cycles,
                "promote_mean_profit_ms": self.config.promote_mean_profit_ms,
                "min_remaining_tokens_for_dflash": self.config.min_remaining_tokens_for_dflash,
                "initial_probe_cycles": self.config.initial_probe_cycles,
                "probe_min_amortization_tokens": self.config.probe_min_amortization_tokens,
            },
            "mode_counts": dict(sorted(self._mode_counts.items())),
            "decision_counts": dict(sorted(self._decision_counts.items())),
            "transitions": list(self._transitions),
            "cycle_log": list(self._cycle_log),
            "cycle_log_truncated": self.max_cycle_log is not None and len(self._cycle_log) >= self.max_cycle_log,
            "profit_ms_mean": _optional_float(_mean(profits) if profits else None),
            "profit_ms_min": _optional_float(min(profits) if profits else None),
            "profit_ms_max": _optional_float(max(profits) if profits else None),
        }

    def _profit_ms(self, *, visible_tokens: int, cycle_wall_ms: float) -> float | None:
        if self.ar_ms_per_token is None:
            return None
        return float(visible_tokens) * float(self.ar_ms_per_token) - float(cycle_wall_ms)

    def _transition(self, *, cycle: int, new_state: AdaptiveState, reason: str) -> None:
        old_state = self.state
        if old_state == new_state:
            return
        self.state = new_state
        self._transitions.append(
            {
                "cycle": int(cycle),
                "from": old_state,
                "to": new_state,
                "reason": reason,
                "cooldown_remaining": int(self.cooldown_remaining),
                "probe_remaining": int(self.probe_remaining),
            }
        )

    def _append_cycle_log(self, row: dict[str, Any]) -> None:
        if self.max_cycle_log is None or len(self._cycle_log) < self.max_cycle_log:
            self._cycle_log.append(row)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)
