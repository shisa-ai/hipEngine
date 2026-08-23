"""Committed speculative-output, stop-tail, and stochastic RNG accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class SpeculativeOutputTail:
    token_ids: tuple[int, ...]
    finish_reason: str | None
    matched_stop_sequence: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class SpeculativeCommitEvent:
    request_id: int
    transaction_id: int
    token_ids: tuple[int, ...]
    accepted_count: int
    correction_or_bonus_token: int | None
    rng_counter_before: int
    rng_counter_after: int
    finish_reason: str | None
    committed: bool = True

    def __post_init__(self) -> None:
        if min(
            int(self.request_id), int(self.transaction_id), int(self.accepted_count),
            int(self.rng_counter_before), int(self.rng_counter_after),
        ) < 0:
            raise ValueError("speculative commit counters must be non-negative")
        if not self.committed:
            raise ValueError("stream events may contain only committed speculative tokens")
        if any(token < 0 for token in self.token_ids):
            raise ValueError("committed token ids must be non-negative")
        if self.correction_or_bonus_token is not None and self.correction_or_bonus_token < 0:
            raise ValueError("correction_or_bonus_token must be non-negative")
        if self.accepted_count > len(self.token_ids):
            raise ValueError("accepted_count cannot exceed committed token count")
        if self.rng_counter_after < self.rng_counter_before:
            raise ValueError("RNG counter cannot move backwards")
        if self.finish_reason is not None and (
            not self.finish_reason or self.finish_reason != self.finish_reason.strip()
        ):
            raise ValueError("finish_reason must be None or non-empty trimmed text")


@dataclass(frozen=True, slots=True)
class StochasticAcceptanceAccounting:
    request_id: int
    accepted_count: int
    accepted_tokens: tuple[int, ...]
    correction_or_bonus_token: int | None
    rng_counter_before: int
    rng_counter_after: int
    acceptance_ratios: tuple[float, ...]
    uniforms_consumed: tuple[float, ...]


def trim_speculative_output(
    token_ids: Sequence[int],
    *,
    max_tokens: int,
    min_tokens: int,
    eos_token_id: int | None,
    stop_token_ids: Sequence[int],
    stop_token_sequences: Sequence[Sequence[int]],
    ignore_eos: bool,
) -> SpeculativeOutputTail:
    """Trim one committed cycle/output to the first binding terminal boundary."""

    tokens = tuple(int(token) for token in token_ids)
    if any(token < 0 for token in tokens):
        raise ValueError("token_ids must be non-negative")
    maximum = int(max_tokens)
    minimum = int(min_tokens)
    if maximum < 0 or minimum < 0 or minimum > maximum:
        raise ValueError("min/max token bounds are invalid")
    stops = {int(token) for token in stop_token_ids}
    sequences = tuple(tuple(int(token) for token in sequence) for sequence in stop_token_sequences)
    if any(not sequence for sequence in sequences):
        raise ValueError("stop sequences must be non-empty")
    visible: list[int] = []
    reason: str | None = None
    matched: tuple[int, ...] = ()
    for token in tokens[:maximum]:
        visible.append(token)
        if len(visible) < minimum:
            continue
        if not ignore_eos and eos_token_id is not None and token == int(eos_token_id):
            reason = "eos"
            break
        if token in stops:
            reason = "stop"
            matched = (token,)
            break
        for sequence in sequences:
            if len(visible) >= len(sequence) and tuple(visible[-len(sequence):]) == sequence:
                reason = "stop"
                matched = sequence
                break
        if reason is not None:
            break
    if reason is None and len(visible) >= maximum and maximum > 0:
        reason = "length"
    return SpeculativeOutputTail(tuple(visible), reason, matched)


def stochastic_acceptance_accounting(
    *,
    request_id: int,
    candidate_tokens: Sequence[int],
    draft_probabilities: Sequence[float],
    target_probabilities: Sequence[float],
    uniforms: Sequence[float],
    rng_counter_before: int,
    correction_or_bonus_token: int | None = None,
) -> StochasticAcceptanceAccounting:
    """Reference acceptance/RNG accounting without owning sampler policy.

    Each candidate consumes one uniform until first rejection. Acceptance uses
    ``u <= min(1, p_target / p_draft)`` with the usual p_draft==0 fail-close.
    Correction sampling itself is provider/sampler-owned; this record only binds
    its token and the uniforms consumed by acceptance.
    """

    candidates = tuple(int(token) for token in candidate_tokens)
    draft = tuple(float(value) for value in draft_probabilities)
    target = tuple(float(value) for value in target_probabilities)
    randoms = tuple(float(value) for value in uniforms)
    if not (len(candidates) == len(draft) == len(target) == len(randoms)):
        raise ValueError("stochastic acceptance vectors must align")
    if any(token < 0 for token in candidates):
        raise ValueError("candidate tokens must be non-negative")
    if any(value < 0.0 or value > 1.0 for value in (*draft, *target, *randoms)):
        raise ValueError("probabilities/uniforms must be in [0, 1]")
    before = int(rng_counter_before)
    if before < 0:
        raise ValueError("rng_counter_before must be non-negative")
    accepted: list[int] = []
    ratios: list[float] = []
    consumed: list[float] = []
    for token, p_draft, p_target, uniform in zip(
        candidates, draft, target, randoms, strict=True
    ):
        ratio = 0.0 if p_draft <= 0.0 else min(1.0, p_target / p_draft)
        ratios.append(ratio)
        consumed.append(uniform)
        if uniform > ratio:
            break
        accepted.append(token)
    correction = None if correction_or_bonus_token is None else int(correction_or_bonus_token)
    if correction is not None and correction < 0:
        raise ValueError("correction_or_bonus_token must be non-negative")
    return StochasticAcceptanceAccounting(
        request_id=int(request_id),
        accepted_count=len(accepted),
        accepted_tokens=tuple(accepted),
        correction_or_bonus_token=correction,
        rng_counter_before=before,
        rng_counter_after=before + len(consumed),
        acceptance_ratios=tuple(ratios),
        uniforms_consumed=tuple(consumed),
    )
