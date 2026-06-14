"""Helpers for enriching structured generation finish metadata."""

from __future__ import annotations

from typing import Any

from hipengine.generation.registry import FinishDetails


_THINKING_PHASE_TO_FINISH_PHASE = {
    "think": "reasoning",
    "closing_think": "closing_think",
    "answer": "answer",
    "done": "done",
}
_LENGTH_REASONS = {"length", "max_length", "max_tokens", "token_budget_exhausted", "budget_exhausted"}


def finish_details_with_sampling_state(
    details: FinishDetails,
    state: Any | None,
) -> FinishDetails:
    """Return ``details`` enriched with per-row thinking-budget metadata."""

    budget = None if state is None else getattr(state, "thinking_budget", None)
    if budget is None:
        return details
    forced_tokens = getattr(budget, "forced_tokens", None)
    forced_reason = None if forced_tokens is None else getattr(forced_tokens, "reason", None)
    forced_close = bool(details.forced_close or forced_reason == "thinking_hard_close")
    reasoning_tokens = max(int(details.reasoning_tokens), int(getattr(budget, "reasoning_tokens", 0)))
    answer_tokens = max(int(details.answer_tokens), int(getattr(budget, "answer_tokens", 0)))
    budget_pressure = details.budget_pressure or ("hard_close" if forced_close else getattr(budget, "budget_pressure", None))
    reason = details.reason
    if str(reason).strip().lower() in _LENGTH_REASONS and forced_close and answer_tokens == 0:
        reason = "thinking_budget_exhausted"
    phase = details.phase
    if phase is None and (forced_close or reasoning_tokens or answer_tokens):
        phase = _THINKING_PHASE_TO_FINISH_PHASE.get(str(getattr(budget, "phase", "")))
    return FinishDetails(
        reason=reason,
        eos_token_id=details.eos_token_id,
        stop_sequence=details.stop_sequence,
        length_limit=details.length_limit,
        deadline_exceeded=details.deadline_exceeded,
        cancelled=details.cancelled,
        forced_close=forced_close,
        synthetic_tokens=details.synthetic_tokens,
        reasoning_tokens=reasoning_tokens,
        answer_tokens=answer_tokens,
        tool_call_tokens=details.tool_call_tokens,
        structured_tokens=details.structured_tokens,
        budget_pressure=budget_pressure,
        cache_action=details.cache_action,
        sampler_mode=details.sampler_mode,
        phase=phase,
        continuation_eligible=details.continuation_eligible,
    )


__all__ = ["finish_details_with_sampling_state"]
