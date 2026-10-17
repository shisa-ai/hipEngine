"""Host sampling for Gemma 4: the request's sampler law, and exact speculative accept.

Gemma 4's vocabulary is 262,144 entries, and the shared host sampler
(``hipengine.generation.sampling``) orders the whole row before it filters it,
which costs more than a decode step at that width. This module computes the same
law without a full-row sort. A top-k request partitions the row for its k
largest entries and hands only those to the shared top-k, softmax, top-p and
min-p helpers, so its law is the shared sampler's bit for bit. A request without
top-k needs the softmax over every entry, which is exponentiated in single
precision and normalized in double: each probability is the double-precision
softmax's to a relative error of about 1e-7 per nat below the row maximum, far
inside what the BF16 logits themselves resolve, and an entry some 87 nats below
the maximum underflows to a weight under 1e-37 or to zero. Top-p and min-p then run through the shared helper on the entries
large enough to matter.

A :class:`Gemma4Sampler` carries one request row's sampler fields, its token
history for the history penalties, and its random stream. The speculative route
uses the same object for the target rows and for the drafter's rows, observing
drafted tokens as it walks them and un-observing the ones it gives back.

It also owns the request's decode controls, because both routes have to serve
them and two implementations would drift: a ``min_tokens`` floor, a thinking
budget (hard close, soft close, and end-of-generation suppression while
reasoning), a multi-token stop matcher, and the per-token logprob report. The
state those controls read is a ``ThinkingBudgetState``, a
``TokenSequenceDFAState`` and the emitted-token count; ``unobserve`` restores all
three from a per-observe snapshot, so a draft that is given back leaves no
reasoning token counted and no stop match behind.

:func:`speculative_accept` is the exact coupling of Leviathan et al., "Fast
Inference from Transformers via Speculative Decoding" (2022): a token ``x`` drawn
from a draft law ``q`` is accepted with probability ``min(1, p(x) / q(x))`` and a
rejection emits a draw from ``normalize(max(0, p - q))``, so the emitted token is
distributed as the target law ``p`` whatever ``q`` is. A greedy request is the
degenerate case: both laws are point masses, a draft is accepted exactly when it
is the target's argmax, and a rejection emits that argmax.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from hipengine.generation.constraints import ThinkingBudgetState, TokenSequenceDFAState
from hipengine.generation.sampling import (
    _apply_probability_filters,
    _logprob_summary,
    _softmax,
    _top_k_candidate_ids,
    _top_logprob_pairs,
    clone_thinking_budget_state,
    logprobs_requested,
    normalize_logit_bias_pairs,
    row_seed_for_index,
    thinking_budget_state_from_params,
    validate_sampling_params,
)

#: How many of the head's most likely tokens a draft is drawn from when the
#: request's own top-k is unset or wider. A draft law may be any distribution:
#: the accept rule makes the committed token follow the target's law whatever
#: the draft was drawn from, so a narrow one costs no exactness and avoids a
#: softmax over the whole vocabulary for every drafted token.
DRAFT_TOP_K = 64

# A distribution over more entries than this samples through block sums rather
# than one cumulative sum over every entry.
_BLOCK_SAMPLE_MIN = 4096
_BLOCK = 512

# Keeps the drafter's stream apart from the request's own, so the accept walk
# consumes the same request draws whichever drafter proposed the tokens.
_DRAFT_STREAM = 0x6D7470


@dataclass(frozen=True)
class Gemma4Distribution:
    """One sampler law: retained token ids, ascending, and their probabilities."""

    token_ids: np.ndarray
    probabilities: np.ndarray

    @classmethod
    def point_mass(cls, token_id: int) -> "Gemma4Distribution":
        return cls(
            np.asarray([int(token_id)], dtype=np.int64),
            np.asarray([1.0], dtype=np.float64),
        )

    def probability(self, token_id: int) -> float:
        """The retained probability of ``token_id``, 0.0 when it was filtered out."""

        index = int(np.searchsorted(self.token_ids, int(token_id)))
        if index >= self.token_ids.size or int(self.token_ids[index]) != int(token_id):
            return 0.0
        return float(self.probabilities[index])

    def sample(self, draw: float) -> int:
        """Draw one token with one uniform ``draw`` in ``[0, 1)``."""

        unit = float(draw)
        if not 0.0 <= unit < 1.0:
            raise ValueError("sampling draws must be in [0, 1)")
        size = int(self.token_ids.size)
        if size == 1:
            return int(self.token_ids[0])
        probabilities = self.probabilities
        start = 0
        if size > _BLOCK_SAMPLE_MIN:
            # Find the block the draw lands in from the block totals, then walk
            # only that block. The same inverse CDF, in two steps.
            whole = size - size % _BLOCK
            totals = np.cumsum(probabilities[:whole].reshape(-1, _BLOCK).sum(axis=1))
            block = int(np.searchsorted(totals, unit, side="right"))
            start = min(block * _BLOCK, whole)
            if block:
                unit -= float(totals[min(block, totals.size) - 1])
            probabilities = probabilities[start:start + _BLOCK if start < whole else size]
        choice = int(np.searchsorted(np.cumsum(probabilities), unit, side="right"))
        return int(self.token_ids[min(start + choice, size - 1)])


_token_id_ranges: dict[int, np.ndarray] = {}


def _all_token_ids(size: int) -> np.ndarray:
    cached = _token_id_ranges.get(size)
    if cached is None:
        cached = np.arange(size, dtype=np.int64)
        cached.setflags(write=False)
        _token_id_ranges[size] = cached
    return cached


def residual_distribution(
    target: Gemma4Distribution, draft: Gemma4Distribution
) -> Gemma4Distribution | None:
    """``normalize(max(0, p - q))``, or ``None`` when ``q`` covers ``p`` everywhere."""

    size = int(max(target.token_ids[-1], draft.token_ids[-1])) + 1
    mass = np.zeros(size, dtype=np.float64)
    mass[target.token_ids] = target.probabilities
    mass[draft.token_ids] -= draft.probabilities
    token_ids = np.flatnonzero(mass > 0.0)
    if token_ids.size == 0:
        return None
    kept = mass[token_ids]
    return Gemma4Distribution(token_ids, kept / float(kept.sum()))


def speculative_accept(
    drafts: Sequence[int],
    draft_distributions: Sequence[Gemma4Distribution],
    target_distribution: Callable[[int], Gemma4Distribution],
    draw: Callable[[], float],
) -> tuple[int, int]:
    """Walk one drafted chain; return ``(accepted, next_token)``.

    ``draft_distributions[i]`` is the law ``drafts[i]`` was drawn from.
    ``target_distribution(i)`` is the target's law for the same decision, asked
    for only once ``drafts[:i]`` have been accepted, so the caller computes a
    row's law only when the walk reaches it and can condition it on exactly the
    accepted prefix. ``next_token`` is the residual draw at the first rejection,
    or a draw from ``target_distribution(len(drafts))`` when every draft was
    accepted. ``draw`` is called once per acceptance test and once more for the
    token that ends the walk.
    """

    if len(draft_distributions) != len(drafts):
        raise ValueError("every draft needs the distribution it was drawn from")
    for index, token in enumerate(drafts):
        target = target_distribution(index)
        draft = draft_distributions[index]
        proposed = draft.probability(int(token))
        if proposed <= 0.0:
            raise ValueError(
                f"drafted token {int(token)} has zero probability under its own draft law"
            )
        if float(draw()) < min(1.0, target.probability(int(token)) / proposed):
            continue
        residual = residual_distribution(target, draft)
        if residual is None:
            # Unreachable with exact arithmetic: a rejection needs p(x) < q(x),
            # which leaves mass elsewhere. Rounding can close that gap; the
            # target law is then the draft law and a draw from it is exact.
            residual = target
        return index, residual.sample(float(draw()))
    return len(drafts), target_distribution(len(drafts)).sample(float(draw()))


@dataclass(frozen=True)
class Gemma4Selection:
    """One selected row: the token, the law it came from, and its metadata."""

    token_id: int
    law: Gemma4Distribution
    #: True when a control (a thinking budget's close sequence) chose the token
    #: rather than the law, in which case it was emitted without a draw.
    forced: bool = False
    logprob: float | None = None
    top_logprobs: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class Gemma4RowLaw:
    """One verified row's law plus the override that governs its outgoing edge."""

    law: Gemma4Distribution
    #: The forced token this row must emit, or ``None`` when it draws freely.
    forced_token_id: int | None = None
    #: The row's logprob report when the token it commits is already determined.
    #: A greedy law is a point mass, so the accept walk can only commit that
    #: token; reporting it at row time is what lets a speculative cycle report
    #: the same value the autoregressive route would, without re-deriving the
    #: row from a state the walk has already left.
    report: tuple[float | None, tuple[tuple[int, float], ...]] | None = None


@dataclass(frozen=True)
class _ControlSnapshot:
    """The control state one ``observe`` is about to change."""

    budget: ThinkingBudgetState | None
    stop_state: TokenSequenceDFAState
    generated_count: int


class Gemma4Sampler:
    """One request row's sampler fields, token history and random stream.

    The sampler owns the request's decode controls as well as its law. A
    ``min_tokens`` floor, a thinking budget (hard close, soft close, and EOS
    suppression while reasoning) and a multi-token stop matcher all run here, on
    the same per-row state, so the autoregressive route and the speculative one
    cannot implement them differently: both ask this object for a row.
    """

    def __init__(
        self,
        request: Any,
        prompt_ids: Sequence[int],
        *,
        row_index: int = 0,
        eos_token_ids: Sequence[int] | None = None,
    ) -> None:
        validate_sampling_params(request)
        self._params = request
        self.temperature = float(getattr(request, "temperature", 0.0))
        self.top_k = int(getattr(request, "top_k", 0))
        self.top_p = float(getattr(request, "top_p", 1.0))
        self.min_p = float(getattr(request, "min_p", 0.0))
        self.repetition_penalty = float(getattr(request, "repetition_penalty", 1.0))
        self.presence_penalty = float(getattr(request, "presence_penalty", 0.0))
        self.frequency_penalty = float(getattr(request, "frequency_penalty", 0.0))
        self.logit_bias = normalize_logit_bias_pairs(getattr(request, "logit_bias", None))
        self.suppress_token_ids = tuple(
            int(token) for token in (getattr(request, "suppress_token_ids", None) or ())
        )
        self.history_dependent = (
            self.repetition_penalty != 1.0
            or self.presence_penalty != 0.0
            or self.frequency_penalty != 0.0
        )
        self.min_tokens = int(getattr(request, "min_tokens", 0) or 0)
        self.wants_logprobs = logprobs_requested(request)
        self.requested_top_logprobs = int(getattr(request, "top_logprobs", 0) or 0)
        # The end-of-generation ids this request stops on. Gemma 4 stops on a
        # set (``<eos>`` plus the turn/tool-response ids) rather than one id, so
        # ``min_tokens`` and a thinking budget have to hold the whole set back;
        # suppressing only ``request.eos_token_id`` would let the model end the
        # request early through another member of its own stop set.
        self._eos_token_ids: tuple[int, ...] = tuple(
            dict.fromkeys(
                int(token)
                for token in (
                    eos_token_ids
                    if eos_token_ids is not None
                    else (() if getattr(request, "eos_token_id", None) is None else (request.eos_token_id,))
                )
            )
        )
        self._budget = thinking_budget_state_from_params(request)
        self._stop_state = TokenSequenceDFAState.from_sequences(
            tuple(getattr(request, "stop_token_sequences", ()) or ())
        )
        self._generated_count = 0
        # Only a control that reads the emitted history needs an undo record. A
        # plain request's history is one counter, which ``unobserve`` restores
        # exactly without a snapshot.
        self._undo_tracking = self._budget is not None or bool(self._stop_state.sequences)
        self._undo: list[_ControlSnapshot] = []
        self._counts: Counter[int] = Counter(int(token) for token in prompt_ids)
        seed = row_seed_for_index(request, row_index)
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._draft_rng = np.random.Generator(np.random.PCG64([seed, _DRAFT_STREAM]))

    @property
    def greedy(self) -> bool:
        """True when every law this sampler produces is a point mass."""

        return self.temperature <= 0.0 or self.top_k == 1

    @property
    def thinking_budget(self) -> ThinkingBudgetState | None:
        """The request's thinking-budget state, or ``None`` when it has none.

        Exposed under the name ``finish_details_with_sampling_state`` reads, so a
        generator can enrich its finish metadata with this sampler directly.
        """

        return self._budget

    @property
    def stop_state(self) -> TokenSequenceDFAState:
        """The multi-token stop matcher advanced by this row's emitted tokens."""

        return self._stop_state

    @property
    def stop_sequence_matched(self) -> tuple[int, ...]:
        """The stop sequence the emitted history ends with, or ``()``."""

        return self._stop_state.matched_sequence

    @property
    def control_active(self) -> bool:
        """True when a decode control can change the selected token."""

        return self._budget is not None or self.min_tokens > 0

    @property
    def reports_full_support(self) -> bool:
        """Whether this row reports the full-support softmax for its token.

        The shared host sampler branches on temperature alone: a greedy row
        reports the model's own probability for the token, and a sampling row
        reports the retained support's probability for the token it drew. A
        ``top_k=1`` request with a temperature is a sampling row whose support
        happens to hold one token, so its report is ``0.0`` rather than the
        model's probability. Branching the same way is what makes this route's
        metadata equal the shared route's rather than merely similar.
        """

        return self.temperature <= 0.0

    @property
    def processes_logits(self) -> bool:
        """True when the token is not simply the raw row's argmax."""

        return (
            not self.greedy
            or self.history_dependent
            or bool(self.logit_bias)
            or bool(self.suppress_token_ids)
            or self.control_active
            or self.wants_logprobs
        )

    def random_unit(self) -> float:
        return float(self._rng.random())

    def draft_unit(self) -> float:
        return float(self._draft_rng.random())

    def observe(self, token_id: int) -> None:
        token = int(token_id)
        if self._undo_tracking:
            self._undo.append(self._control_snapshot())
        self._counts[token] += 1
        self._generated_count += 1
        self._stop_state = self._stop_state.observe(token)
        if self._budget is not None:
            self._budget.observe(token)

    def unobserve(self, token_id: int) -> None:
        """Undo one :meth:`observe`.

        The speculative route conditions each draft on the drafts before it and
        then gives them back, so this has to restore the control state exactly:
        a thinking budget that counted a draft twice would close the reasoning
        phase early, and a stop matcher that kept one would report a match the
        request never emitted.
        """

        token = int(token_id)
        remaining = self._counts[token] - 1
        if remaining < 0:
            raise ValueError(f"token {token} was not observed")
        if remaining:
            self._counts[token] = remaining
        else:
            del self._counts[token]
        self._generated_count -= 1
        if self._undo_tracking:
            snapshot = self._undo.pop()
            self._budget = snapshot.budget
            self._stop_state = snapshot.stop_state
            self._generated_count = snapshot.generated_count

    def _control_snapshot(self) -> _ControlSnapshot:
        return _ControlSnapshot(
            budget=clone_thinking_budget_state(self._budget),
            stop_state=self._stop_state,
            generated_count=self._generated_count,
        )

    def checkpoint_controls(self) -> _ControlSnapshot:
        """Save controls before selection can queue or consume forced tokens."""
        return self._control_snapshot()

    def restore_controls(self, snapshot: _ControlSnapshot) -> None:
        """Restore controls after reversing speculative history observations."""
        self._budget = clone_thinking_budget_state(snapshot.budget)
        self._stop_state = snapshot.stop_state
        self._generated_count = snapshot.generated_count

    def discard_undo_history(self) -> None:
        """Published tokens no longer need speculative rollback snapshots."""
        self._undo.clear()

    def prepare_selection(self) -> None:
        """Apply the step's control transitions before a row is read.

        This is where a reached thinking cap turns into the queued close
        sequence, the same transition ``RowSamplingState.prepare_for_selection``
        makes for the shared host sampler. Idempotent: it never pops a token, so
        a caller may prepare a row it goes on to reject.
        """

        if self._budget is not None:
            self._budget.ensure_hard_close()

    def pending_forced_token(self) -> int | None:
        return None if self._budget is None else self._budget.forced_tokens.peek()

    def pop_forced_token(self) -> int | None:
        return None if self._budget is None else self._budget.forced_tokens.pop()

    def _suppressed_eos_ids(self) -> tuple[int, ...]:
        """The end-of-generation ids this row may not emit, if any."""

        held_back = self.min_tokens > 0 and self._generated_count < self.min_tokens
        budget = self._budget
        if budget is not None and budget.eos_suppression_active and not budget.forced_tokens:
            held_back = True
        return self._eos_token_ids if held_back else ()

    def _soft_close_bias(self) -> tuple[int, float] | None:
        """The soft-close bias on the first close token, or ``None``."""

        budget = self._budget
        if budget is None or budget.forced_tokens:
            return None
        bias = budget.soft_close_bias
        if bias is None or not budget.close_sequence:
            return None
        return int(budget.close_sequence[0]), float(bias)

    def _processed(self, logits: np.ndarray) -> np.ndarray:
        """Bias, history penalties, suppression and controls, in the shared order."""

        self.prepare_selection()
        row = np.asarray(logits, dtype=np.float32).reshape(-1)
        penalized = self.history_dependent and bool(self._counts)
        suppressed = self._suppressed_eos_ids()
        soft_close = self._soft_close_bias()
        if not (
            penalized
            or self.logit_bias
            or self.suppress_token_ids
            or suppressed
            or soft_close is not None
        ):
            return row
        work = row.astype(np.float64)
        vocab = int(work.size)
        for token_id, bias in self.logit_bias:
            if token_id >= vocab:
                raise ValueError(f"logit_bias token id {token_id} is outside vocab size {vocab}")
            work[token_id] += float(bias)
        if penalized:
            token_ids = np.fromiter(self._counts.keys(), dtype=np.int64, count=len(self._counts))
            counts = np.fromiter(self._counts.values(), dtype=np.float64, count=len(self._counts))
            inside = (token_ids >= 0) & (token_ids < vocab)
            token_ids, counts = token_ids[inside], counts[inside]
            values = work[token_ids]
            if self.repetition_penalty != 1.0:
                values = np.where(
                    values < 0.0,
                    values * self.repetition_penalty,
                    values / self.repetition_penalty,
                )
            if self.presence_penalty != 0.0:
                values = values - self.presence_penalty
            if self.frequency_penalty != 0.0:
                values = values - self.frequency_penalty * counts
            work[token_ids] = values
        for token_id in self.suppress_token_ids:
            if token_id >= vocab:
                raise ValueError(
                    f"suppress_token_ids token id {token_id} is outside vocab size {vocab}"
                )
            work[token_id] = -np.inf
        for token_id in suppressed:
            if token_id >= vocab:
                raise ValueError(
                    f"end-of-generation token id {token_id} is outside vocab size {vocab}"
                )
            work[token_id] = -np.inf
        if soft_close is not None:
            token_id, bias = soft_close
            if token_id >= vocab:
                raise ValueError(
                    f"thinking close token id {token_id} is outside vocab size {vocab}"
                )
            if np.isfinite(work[token_id]):
                work[token_id] += bias
        return work

    def draft_distribution(self, logits: np.ndarray) -> Gemma4Distribution:
        """The law a draft is drawn from: this request's, over at most ``DRAFT_TOP_K``."""

        top_k = self.top_k if 0 < self.top_k <= DRAFT_TOP_K else DRAFT_TOP_K
        return self.distribution(logits, top_k=top_k)

    def distribution(
        self, logits: np.ndarray, *, top_k: int | None = None
    ) -> Gemma4Distribution:
        """The law this request draws its next token from, given ``logits``.

        ``top_k`` overrides the request's own; :meth:`draft_distribution` is the
        one caller that passes it. A pending control override is a point mass: a
        thinking budget's close sequence is emitted, not sampled.
        """

        self.prepare_selection()
        forced = self.pending_forced_token()
        if forced is not None:
            return Gemma4Distribution.point_mass(forced)
        return self._law(self._processed(logits), self.top_k if top_k is None else int(top_k))

    def _law(self, processed: np.ndarray, top_k: int) -> Gemma4Distribution:
        """This request's law over an already-processed row."""

        peak_id = int(np.argmax(processed))
        peak = float(processed[peak_id])
        if not np.isfinite(peak):
            # NaN sorts as the maximum, and +inf or an all -inf row has no
            # softmax. None of them is a law to sample.
            raise ValueError("logits row has no finite maximum")
        if self.greedy:
            return Gemma4Distribution.point_mass(peak_id)
        size = int(processed.size)
        if 0 < top_k < size:
            # The k largest entries, plus every entry tied with the smallest of
            # them so the shared helper breaks the tie by token id as it would
            # on the whole row.
            top = np.argpartition(processed, size - top_k)[size - top_k:]
            candidates = np.flatnonzero(processed >= processed[top].min())
            candidates = candidates[np.isfinite(processed[candidates])]
            scaled = processed[candidates].astype(np.float64) / self.temperature
            order = _top_k_candidate_ids(scaled, top_k)
            token_ids, probabilities = _apply_probability_filters(
                candidates[order], _softmax(scaled[order]),
                top_p=self.top_p, min_p=self.min_p,
            )
        else:
            token_ids, probabilities = self._filtered_dense(processed, peak)
            if self.top_p >= 1.0 and self.min_p <= 0.0:
                # Already ascending and normalized over the whole row.
                return Gemma4Distribution(token_ids, probabilities)
        ascending = np.argsort(token_ids)
        probabilities = probabilities[ascending]
        return Gemma4Distribution(
            token_ids[ascending].astype(np.int64, copy=False),
            probabilities / float(probabilities.sum()),
        )

    def row_law(self, logits: np.ndarray) -> Gemma4RowLaw:
        """The law for one verified row, with its override and greedy report.

        The speculative walk asks this for a row it may never reach, so it must
        stay free of draws and of pops: preparing the row is idempotent, and the
        caller consumes the override only for a position it publishes.
        """

        self.prepare_selection()
        forced = self.pending_forced_token()
        if forced is not None:
            law = Gemma4Distribution.point_mass(forced)
            report = self._forced_report(logits, forced) if self.wants_logprobs else None
            return Gemma4RowLaw(law=law, forced_token_id=forced, report=report)
        law = self._law(self._processed(logits), self.top_k)
        report = None
        if self.wants_logprobs and self.reports_full_support:
            # A full-support row's law is a point mass, so the token it commits
            # is already known and its logprob can be reported now.
            report = self._greedy_report(logits, int(law.token_ids[0]))
        return Gemma4RowLaw(law=law, report=report)

    def reported_logprob(
        self, law: Gemma4Distribution, token_id: int
    ) -> tuple[float | None, tuple[tuple[int, float], ...]]:
        """The metadata the autoregressive route reports for a token drawn from ``law``.

        A sampling row reports the retained support's probability for the token
        it drew, which is the distribution the draw came from; the shared host
        sampler reports the same value, so a speculative cycle's metadata equals
        the autoregressive route's rather than merely resembling it.
        """

        probability = law.probability(int(token_id))
        logprob = None if probability <= 0.0 else float(math.log(probability))
        return logprob, _top_logprob_pairs(
            law.token_ids, law.probabilities, self.requested_top_logprobs
        )

    def _greedy_report(
        self, logits: np.ndarray, token_id: int
    ) -> tuple[float | None, tuple[tuple[int, float], ...]]:
        """The full-support logprob of a row's token, as the shared route reports it."""

        processed = np.asarray(self._processed(logits), dtype=np.float64)
        if not np.isfinite(processed[int(token_id)]):
            return None, ()
        return _logprob_summary(processed, int(token_id), self.requested_top_logprobs)

    def _forced_report(
        self, logits: np.ndarray, token_id: int
    ) -> tuple[float | None, tuple[tuple[int, float], ...]]:
        """The report for a token a control emitted without a draw.

        The override was not sampled, so its law is a point mass and a draw-based
        report would claim certainty the model never had. Report the row's own
        full-support probability for the forced token instead, which is what the
        shared host sampler reports for a forced token too.
        """

        return self._greedy_report(logits, token_id)

    def select_row(self, logits: np.ndarray) -> Gemma4Selection:
        """Select the next token, record it, and report how it was chosen."""

        self.prepare_selection()
        forced = self.pending_forced_token()
        if forced is not None:
            self.pop_forced_token()
            law = Gemma4Distribution.point_mass(forced)
            report = self._forced_report(logits, forced) if self.wants_logprobs else (None, ())
            self.observe(forced)
            return Gemma4Selection(
                token_id=forced,
                law=law,
                forced=True,
                logprob=report[0],
                top_logprobs=report[1],
            )
        law = self._law(self._processed(logits), self.top_k)
        token = law.sample(self.random_unit())
        if self.wants_logprobs:
            report = (
                self._greedy_report(logits, token)
                if self.reports_full_support
                else self.reported_logprob(law, token)
            )
        else:
            report = (None, ())
        self.observe(token)
        return Gemma4Selection(
            token_id=token,
            law=law,
            logprob=report[0],
            top_logprobs=report[1],
        )

    def _filtered_dense(
        self, processed: np.ndarray, peak: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Softmax over the whole row, then top-p and min-p, without sorting it."""

        scale = np.float32(1.0 / self.temperature)
        shifted = (processed - processed.dtype.type(peak)).astype(np.float32, copy=False)
        probabilities = np.exp(shifted * scale).astype(np.float64)
        probabilities /= float(probabilities.sum())
        if self.top_p >= 1.0 and self.min_p <= 0.0:
            # A suppressed entry keeps a zero weight here; it can neither be
            # drawn nor accepted.
            return _all_token_ids(probabilities.size), probabilities
        # The shared filter sorts what it is given, so it only needs every entry
        # at least as probable as the last one it keeps. Lower the floor until
        # the entries above it hold the nucleus.
        floor = float(probabilities.max()) * (
            self.min_p if self.top_p >= 1.0 else min(1.0, max(self.min_p, 1e-3))
        )
        while True:
            head = np.flatnonzero(probabilities >= floor)
            if (
                self.top_p >= 1.0
                or floor <= 0.0
                or float(probabilities[head].sum()) >= self.top_p
            ):
                break
            floor = floor / 32.0 if floor > 1e-300 else 0.0
        return _apply_probability_filters(
            head, probabilities[head], top_p=self.top_p, min_p=self.min_p
        )

    def select(self, logits: np.ndarray) -> int:
        """Draw the next token from ``logits`` and record it in the history."""

        return self.select_row(logits).token_id


__all__ = [
    "DRAFT_TOP_K",
    "Gemma4Distribution",
    "Gemma4RowLaw",
    "Gemma4Sampler",
    "Gemma4Selection",
    "residual_distribution",
    "speculative_accept",
]
