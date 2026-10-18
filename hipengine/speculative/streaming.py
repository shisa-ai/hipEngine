"""Committed speculative-output, stop-tail, and stochastic RNG accounting."""

from __future__ import annotations

from dataclasses import dataclass, replace
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


@dataclass(frozen=True, slots=True)
class ChainFinish:
    """How a verified chain reaches the autoregressive finish rule.

    ``reason`` is the value the autoregressive route publishes: ``"eos"`` when
    the terminal token is the row's EOS, ``"stop"`` when it is a stop token id
    or the last token of a stop sequence.  The other two fields carry the detail
    the route publishes with it, so a caller reports the same thing the
    autoregressive route would rather than a bare reason string.
    """

    reason: str
    stop_token_id: int | None = None
    stop_sequence: tuple[int, ...] = ()


def chain_finish_index(
    visible: Sequence[int],
    *,
    eos_token_id: int | None,
    published_tokens: Sequence[int] = (),
    generated_tokens: int = 0,
    stop_token_ids: Sequence[int] = (),
    stop_token_sequences: Sequence[Sequence[int]] = (),
    min_tokens: int = 0,
    ignore_eos: bool = False,
) -> tuple[int, ChainFinish] | None:
    """Return the first index of ``visible`` that finishes the request, and why.

    This is the autoregressive finish rule applied to a whole verified chain, so
    the cycle commit can select a terminal prefix instead of publishing past a
    stop.  ``visible`` is the chain's accepted tokens followed by its bonus
    token, in publication order.  ``published_tokens`` is what the request has
    already emitted, which is what makes a stop sequence spanning the boundary
    between published output and the chain detectable exactly as the
    autoregressive route detects it.

    ``min_tokens`` is the EOS-suppression floor and suppresses only the EOS rule,
    matching the autoregressive processor: it does not delay a stop token id or a
    stop sequence.  Returns ``None`` when the chain runs to its end without
    finishing, which is the case that publishes the whole chain.
    """

    eos = None if ignore_eos or eos_token_id is None else int(eos_token_id)
    stops = tuple(int(stop) for stop in stop_token_ids)
    sequences = tuple(tuple(int(token) for token in row) for row in stop_token_sequences)
    prefix = tuple(int(token) for token in published_tokens)
    # Imported here because `hipengine.generation.constraints` reaches this module
    # through the generation package, so a module-scope import is circular.
    from hipengine.generation.constraints import token_sequence_state_for_tokens
    for index, token in enumerate(visible):
        token = int(token)
        if eos is not None and token == eos and generated_tokens + index + 1 >= min_tokens:
            return index, ChainFinish(reason="eos", stop_token_id=eos)
        if token in stops:
            return index, ChainFinish(reason="stop", stop_token_id=token)
        if sequences:
            matched = token_sequence_state_for_tokens(
                (*prefix, *(int(item) for item in visible[: index + 1])), sequences
            ).matched_sequence
            if matched:
                return index, ChainFinish(reason="stop", stop_sequence=tuple(matched))
    return None


def limit_chain_accept_finish(
    batch,
    summary,
    *,
    eos_token_id: int | None,
    published_tokens: Sequence[int] = (),
    generated_tokens: int = 0,
    stop_token_ids: Sequence[int] = (),
    stop_token_sequences: Sequence[Sequence[int]] = (),
    min_tokens: int = 0,
    ignore_eos: bool = False,
):
    """Keep the first finishing token as the final prediction, not consumed state.

    Returns ``(summary, finish)``.  ``finish`` is a :class:`ChainFinish` when the
    chain stops and ``None`` when it runs to its end, so a caller tests it for
    truth and reads ``finish.reason`` to publish the right finish.  The returned
    summary commits the prefix before the terminal token, keeps the terminal
    token as the next token, and leaves no model state beyond that prefix.
    """

    if batch.mode != "verify_chain" or len(summary.request_ids) != 1:
        raise ValueError("finish summary limiting requires one verified chain")
    accepted = tuple(summary.accepted_tokens[0])
    next_token = None if summary.next_tokens is None else summary.next_tokens[0]
    visible = (*accepted, *(() if next_token is None else (next_token,)))
    found = chain_finish_index(
        visible,
        eos_token_id=eos_token_id,
        published_tokens=published_tokens,
        generated_tokens=generated_tokens,
        stop_token_ids=stop_token_ids,
        stop_token_sequences=stop_token_sequences,
        min_tokens=min_tokens,
        ignore_eos=ignore_eos,
    )
    if found is None:
        return summary, None
    index, finish = found
    if index == len(visible) - 1 and next_token is not None:
        return summary, finish
    root = int(batch.root_rows[0])
    return replace(
        summary,
        accepted_counts=(index,),
        accepted_tokens=(accepted[:index],),
        commit_rows=(root + index,),
        commit_tokens=((int(batch.tokens[root]) if index == 0 else accepted[index - 1]),),
        commit_positions=(int(batch.positions[root]) + index,),
        next_tokens=(int(visible[index]),),
        full_accept=(False,),
    ), finish


def limit_chain_accept_eos(
    batch,
    summary,
    *,
    eos_token_id: int | None,
    generated_tokens: int = 0,
    min_tokens: int = 0,
    ignore_eos: bool = False,
):
    """EOS-only entry point, kept for callers that cannot stop on anything else.

    Delegates to :func:`limit_chain_accept_finish` and reports the EOS case as a
    boolean, which is what a caller with no stop metadata needs.
    """

    if eos_token_id is None or ignore_eos:
        return summary, False
    limited, finish = limit_chain_accept_finish(
        batch,
        summary,
        eos_token_id=eos_token_id,
        generated_tokens=generated_tokens,
        min_tokens=min_tokens,
        ignore_eos=ignore_eos,
    )
    return limited, finish is not None


def greedy_chain_eos_limit(
    candidate_tokens: Sequence[int],
    target_top1: Sequence[int],
    *,
    remaining_decode: int,
    eos_token_id: int | None,
) -> int:
    """Bound visible output before acceptance so EOS remains the next token.

    The target consumes the root and accepted prefix, not the final next token.
    Returning the EOS output length as the accept budget therefore excludes EOS
    and any later speculative inputs from selected target/provider state.
    Only predictions reachable before the first rejected edge may limit output.
    """
    if remaining_decode < 0 or len(target_top1) != len(candidate_tokens) + 1:
        raise ValueError("invalid greedy chain termination coordinates")
    if eos_token_id is None or remaining_decode == 0:
        return remaining_decode
    for index, token in enumerate(target_top1):
        if index >= remaining_decode:
            break
        if int(token) == int(eos_token_id):
            return index + 1
        if index == len(candidate_tokens) or int(token) != int(candidate_tokens[index]):
            break
    return remaining_decode


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
