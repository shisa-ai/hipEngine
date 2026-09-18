"""Exact speculative sampling: ``min(1, p/q)`` acceptance with residual resampling.

The greedy MTP route accepts a draft token when it equals the target's argmax.
That coupling only reproduces the target's *argmax* decision, so a request that
samples (``temperature > 0``, top-k/top-p, penalties) cannot use it. This module
implements the sampled coupling instead:

* a draft token ``x`` drawn from a draft distribution ``q`` is accepted with
  probability ``min(1, p(x) / q(x))``;
* on rejection the emitted token is drawn from the residual
  ``normalize(max(0, p - q))``;
* when every draft in the chain is accepted, the bonus token is drawn from the
  target distribution at the last verified prefix.

With ``x ~ q`` that construction emits a token distributed exactly as ``p``
(Leviathan et al., "Fast Inference from Transformers via Speculative Decoding",
2022). ``q`` may be a point mass on the drafted token, which is what a greedy
draft chain supplies; the emitted distribution is still exactly ``p``, only the
acceptance probability ``p(x)`` is lower than a sampled draft's.

Everything here is host-side and works on *processed* distributions: the caller
applies the request's sampler pipeline (bias, penalties, suppression,
temperature, top-k, top-p, min-p) to each row's logits first, so the accepted
tokens follow the same distribution the autoregressive route would emit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from hipengine.speculative.interfaces import AcceptResult, TargetVerifyBatch

__all__ = [
    "SparseDistribution",
    "acceptance_probability",
    "residual_distribution",
    "sampled_accept_from_distributions",
]


@dataclass(frozen=True, slots=True)
class SparseDistribution:
    """One processed sampling row: retained token ids and normalized weights.

    The support is the set of ids the sampler pipeline left with non-zero
    probability after temperature, top-k, top-p, and min-p filtering, which is
    exactly what ``select_token`` samples from. Ids are sorted and unique, and
    the weights are float64 and normalized to sum to one, so an acceptance
    ratio and a residual are exact up to double rounding.
    """

    token_ids: tuple[int, ...]
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        ids = tuple(int(token_id) for token_id in self.token_ids)
        if not ids:
            raise ValueError("a sparse distribution needs at least one retained id")
        if len(set(ids)) != len(ids):
            raise ValueError("sparse distribution ids must be unique")
        if tuple(sorted(ids)) != ids:
            raise ValueError("sparse distribution ids must be sorted")
        if any(token_id < 0 for token_id in ids):
            raise ValueError("sparse distribution ids must be non-negative")
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if probabilities.ndim != 1 or probabilities.size != len(ids):
            raise ValueError("sparse distribution weights must align with ids")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("sparse distribution weights must be finite and non-negative")
        total = float(probabilities.sum())
        if total <= 0.0:
            raise ValueError("sparse distribution weights must sum to a positive value")
        object.__setattr__(self, "token_ids", ids)
        object.__setattr__(self, "probabilities", probabilities / total)

    @classmethod
    def point_mass(cls, token_id: int) -> "SparseDistribution":
        """Return the deterministic coupling a greedy draft chain supplies."""

        return cls((int(token_id),), np.asarray([1.0], dtype=np.float64))

    @classmethod
    def from_pairs(
        cls,
        token_ids: Iterable[int],
        probabilities: Iterable[float],
    ) -> "SparseDistribution":
        """Combine duplicate ids (as the union of two supports does) and sort."""

        merged: dict[int, float] = {}
        for token_id, probability in zip(token_ids, probabilities, strict=True):
            merged[int(token_id)] = merged.get(int(token_id), 0.0) + float(probability)
        ids = tuple(sorted(merged))
        return cls(ids, np.asarray([merged[token_id] for token_id in ids]))

    @property
    def support_size(self) -> int:
        return len(self.token_ids)

    def probability(self, token_id: int) -> float:
        """Return the retained probability of one id (0.0 when filtered out)."""

        index = _index_of(self.token_ids, int(token_id))
        return 0.0 if index is None else float(self.probabilities[index])

    def as_dict(self) -> dict[int, float]:
        return {
            token_id: float(probability)
            for token_id, probability in zip(
                self.token_ids, self.probabilities, strict=True
            )
        }


def _index_of(token_ids: tuple[int, ...], token_id: int) -> int | None:
    """Binary-search a sorted id tuple; the supports here are small."""

    low = 0
    high = len(token_ids) - 1
    while low <= high:
        middle = (low + high) // 2
        value = token_ids[middle]
        if value == token_id:
            return middle
        if value < token_id:
            low = middle + 1
        else:
            high = middle - 1
    return None


def acceptance_probability(
    target: SparseDistribution,
    draft: SparseDistribution,
    token_id: int,
) -> float:
    """Return ``min(1, p(x)/q(x))`` for the drafted token ``x``.

    ``q(x)`` must be positive: a token the draft distribution cannot emit is not
    a legal draft, and accepting it would silently couple two different
    distributions. ``p(x) = 0`` (the target filtered the token out) always
    rejects, which is what the residual then covers.
    """

    draft_probability = draft.probability(int(token_id))
    if draft_probability <= 0.0:
        raise ValueError(
            f"drafted token {int(token_id)} has zero draft probability; "
            "the draft token must be inside its own distribution's support"
        )
    target_probability = target.probability(int(token_id))
    if target_probability <= 0.0:
        return 0.0
    return min(1.0, target_probability / draft_probability)


def residual_distribution(
    target: SparseDistribution,
    draft: SparseDistribution,
) -> SparseDistribution:
    """Return ``normalize(max(0, p - q))`` over the union support.

    The residual is the mass the draft distribution failed to cover, which is
    what keeps the rejection branch's output exactly ``p`` distributed.
    """

    merged: dict[int, float] = {
        token_id: float(probability)
        for token_id, probability in zip(
            target.token_ids, target.probabilities, strict=True
        )
    }
    for token_id, probability in zip(
        draft.token_ids, draft.probabilities, strict=True
    ):
        merged[token_id] = merged.get(token_id, 0.0) - float(probability)
    ids = tuple(sorted(token_id for token_id, value in merged.items() if value > 0.0))
    if not ids:
        # p == q on the union support, so there is no residual mass and no
        # rejection can occur. Guard the caller rather than emitting a token
        # from an undefined distribution.
        raise ValueError("target and draft distributions have no residual mass")
    return SparseDistribution(ids, np.asarray([merged[token_id] for token_id in ids]))


def _sample(distribution: SparseDistribution, draw: float) -> int:
    """Draw one token from a sparse distribution with one uniform ``draw``."""

    unit = float(draw)
    if not 0.0 <= unit < 1.0:
        raise ValueError("sampling draws must be in [0, 1)")
    cumulative = np.cumsum(distribution.probabilities)
    choice = int(np.searchsorted(cumulative, unit, side="right"))
    if choice >= len(distribution.token_ids):
        choice = len(distribution.token_ids) - 1
    return int(distribution.token_ids[choice])


def sampled_accept_from_distributions(
    batch: TargetVerifyBatch,
    target_distributions: Sequence[SparseDistribution],
    draft_distributions: Sequence[SparseDistribution],
    *,
    draws: Callable[[], float],
    transaction_id: int | None = None,
    remaining_decode: Sequence[int] | None = None,
) -> AcceptResult:
    """Walk one drafted chain and accept it with ``min(1, p/q)``.

    ``target_distributions[row]`` is the target's processed distribution for the
    token *after* the prefix ending at ``row``; ``draft_distributions[row]`` is
    the draft's distribution for the same decision, which is the distribution the
    token on ``row``'s child edge was drawn from. Both are indexed by the parent
    row, exactly like ``TargetVerifyBatch.accept_from_top1`` indexes ``top1``.

    ``draws`` supplies independent uniforms from the request's own sampling
    stream. It is called once per acceptance test and once per residual or bonus
    sample, in that order, so a caller can replay a decision from a recorded
    stream.
    """

    if len(target_distributions) != batch.rows:
        raise ValueError("target_distributions must align with target verify rows")
    if len(draft_distributions) != batch.rows:
        raise ValueError("draft_distributions must align with target verify rows")
    budgets = (
        None
        if remaining_decode is None
        else tuple(int(count) for count in remaining_decode)
    )
    if budgets is not None:
        if len(budgets) != len(batch.request_ids):
            raise ValueError("remaining_decode must align with request_ids")
        if any(count < 0 for count in budgets):
            raise ValueError("remaining_decode must be non-negative")

    child_rows: dict[int, list[int]] = {row: [] for row in range(batch.rows)}
    for row in batch.candidate_rows:
        if batch.active_mask[row]:
            child_rows[batch.parent_rows[row]].append(row)
    for parent, children in child_rows.items():
        if len(children) > 1:
            raise ValueError(
                "sampled acceptance walks a single drafted chain; "
                f"row {parent} has {len(children)} active children"
            )

    accepted_counts: list[int] = []
    accepted_tokens: list[tuple[int, ...]] = []
    selected_rows: list[int] = []
    next_tokens: list[int | None] = []
    for index, (request_id, root_row) in enumerate(
        zip(batch.request_ids, batch.root_rows, strict=True)
    ):
        budget = None if budgets is None else budgets[index]
        row = int(root_row)
        request_tokens: list[int] = []
        accepted_budget = (
            None if budget is None else max(0, int(budget) - 1)
        )
        emitted: int | None = None
        while accepted_budget is None or len(request_tokens) < accepted_budget:
            children = child_rows[row]
            if not children:
                break
            child = children[0]
            if batch.row_to_request[child] != request_id:
                raise ValueError("sampled acceptance walked a foreign request row")
            token_id = int(batch.tokens[child])
            accepted_probability = acceptance_probability(
                target_distributions[row],
                draft_distributions[row],
                token_id,
            )
            # ``u < alpha`` keeps a zero-probability draft rejecting for every
            # draw (including an exact 0.0) and a certain draft accepting for
            # every draw in [0, 1).
            if float(draws()) < accepted_probability:
                request_tokens.append(token_id)
                row = child
                continue
            emitted = _sample(
                residual_distribution(
                    target_distributions[row],
                    draft_distributions[row],
                ),
                float(draws()),
            )
            break
        if emitted is None:
            if budget is not None and int(budget) <= 0:
                next_tokens.append(None)
            else:
                # Every draft was accepted (or the chain ended): the bonus token
                # comes from the target distribution at the last verified prefix.
                next_tokens.append(
                    _sample(target_distributions[row], float(draws()))
                )
        else:
            next_tokens.append(emitted)
        accepted_counts.append(len(request_tokens))
        accepted_tokens.append(tuple(request_tokens))
        selected_rows.append(row)

    return AcceptResult(
        request_ids=batch.request_ids,
        accepted_counts=tuple(accepted_counts),
        accepted_tokens=tuple(accepted_tokens),
        transaction_id=transaction_id,
        selected_candidate_rows=tuple(selected_rows),
        next_tokens=tuple(next_tokens),
    )
