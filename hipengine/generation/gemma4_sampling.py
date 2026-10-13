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
from typing import Any

import numpy as np

from hipengine.generation.sampling import (
    _apply_probability_filters,
    _softmax,
    _top_k_candidate_ids,
    normalize_logit_bias_pairs,
    row_seed_for_index,
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


class Gemma4Sampler:
    """One request row's sampler fields, token history and random stream."""

    def __init__(self, request: Any, prompt_ids: Sequence[int], *, row_index: int = 0) -> None:
        validate_sampling_params(request)
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
        self._counts: Counter[int] = Counter(int(token) for token in prompt_ids)
        seed = row_seed_for_index(request, row_index)
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._draft_rng = np.random.Generator(np.random.PCG64([seed, _DRAFT_STREAM]))

    @property
    def greedy(self) -> bool:
        """True when every law this sampler produces is a point mass."""

        return self.temperature <= 0.0 or self.top_k == 1

    @property
    def processes_logits(self) -> bool:
        """True when the token is not simply the raw row's argmax."""

        return (
            not self.greedy
            or self.history_dependent
            or bool(self.logit_bias)
            or bool(self.suppress_token_ids)
        )

    def random_unit(self) -> float:
        return float(self._rng.random())

    def draft_unit(self) -> float:
        return float(self._draft_rng.random())

    def observe(self, token_id: int) -> None:
        self._counts[int(token_id)] += 1

    def unobserve(self, token_id: int) -> None:
        token = int(token_id)
        remaining = self._counts[token] - 1
        if remaining < 0:
            raise ValueError(f"token {token} was not observed")
        if remaining:
            self._counts[token] = remaining
        else:
            del self._counts[token]

    def _processed(self, logits: np.ndarray) -> np.ndarray:
        """Bias, history penalties and suppression, in the shared sampler's order."""

        row = np.asarray(logits, dtype=np.float32).reshape(-1)
        penalized = self.history_dependent and bool(self._counts)
        if not (penalized or self.logit_bias or self.suppress_token_ids):
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
        one caller that passes it.
        """

        top_k = self.top_k if top_k is None else int(top_k)

        processed = self._processed(logits)
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

        token = self.distribution(logits).sample(self.random_unit())
        self.observe(token)
        return token


__all__ = [
    "DRAFT_TOP_K",
    "Gemma4Distribution",
    "Gemma4Sampler",
    "residual_distribution",
    "speculative_accept",
]
