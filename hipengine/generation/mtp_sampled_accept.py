"""Sampled MTP acceptance: verified target logits -> a sampled accept summary.

The MTP verifier produces one logits row per verified prefix (root plus every
drafted token). For a request whose sampler is not greedy, the accept decision
must be taken by sampling, not by comparing raw argmaxes: this module turns
those rows into the per-row target distributions the request's own sampler would
use, couples them against the drafted chain, and returns the same
``TargetAcceptSummary`` the argmax kernel returns.

Two properties matter and both are tested:

* every row is processed against *its own* prefix history, so history-dependent
  processors (repetition/presence/frequency penalties, min-token EOS
  suppression, thinking-budget state) see exactly the tokens the autoregressive
  route would have seen at that position;
* the acceptance walk draws from the request's own sampler stream, and only the
  committed tokens are observed into the row's live state afterwards, so the
  request's stream stays aligned with the autoregressive route.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from hipengine.generation.sampling import RowSamplingState, processed_distribution, reported_logprob
from hipengine.speculative.interfaces import TargetAcceptSummary, TargetVerifyBatch
from hipengine.speculative.sampling import (
    SparseDistribution,
    sampled_accept_from_distributions,
)

__all__ = [
    "chain_edge_rows",
    "chain_token_logprobs",
    "row_prefix_states",
    "sampled_accept_summary",
]


def chain_edge_rows(
    batch: TargetVerifyBatch,
    accepted_counts: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Return, per request, the row whose logits predicted each published token.

    Row ``r`` holds the distribution for the token *after* the prefix ending at
    ``r``, so a token on a child edge is predicted by its parent row and the
    correction or bonus token is predicted by the last accepted row -- the row
    the accept walk stopped on, which is the request's commit row. The returned
    tuple therefore starts at the root row and has one entry per published token
    including that final one, so it aligns with
    ``accepted_tokens + (next_token,)``. Getting this mapping off by one row is
    the failure mode this function exists to prevent, so it walks the same
    parent/child structure ``sampled_accept_from_distributions`` walks rather
    than assuming a chain shape.
    """

    children: dict[int, list[int]] = {row: [] for row in range(batch.rows)}
    for row in batch.candidate_rows:
        if batch.active_mask[row]:
            children[int(batch.parent_rows[row])].append(int(row))
    rows: list[tuple[int, ...]] = []
    for index, root_row in enumerate(batch.root_rows):
        accepted = int(accepted_counts[index])
        path = [int(root_row)]
        row = int(root_row)
        for _ in range(accepted):
            candidates = children[row]
            if not candidates:
                raise ValueError(
                    "accepted chain is longer than the verified rows: row "
                    f"{row} has no active child"
                )
            if len(candidates) > 1:
                raise ValueError(
                    "logprob metadata walks a single drafted chain; "
                    f"row {row} has {len(candidates)} active children"
                )
            row = candidates[0]
            path.append(row)
        rows.append(tuple(path))
    return tuple(rows)


def chain_token_logprobs(
    batch: TargetVerifyBatch,
    target_logits: np.ndarray,
    states: Mapping[int, RowSamplingState],
    published_tokens: Sequence[Sequence[int]],
    accepted_counts: Sequence[int],
    *,
    params_for: Callable[[int], Any],
    token_text_for_id: Callable[[int], str] | None = None,
) -> tuple[tuple[tuple[float | None, tuple[tuple[int, float], ...]], ...], ...]:
    """Return each request's published-token logprob metadata from verified rows.

    ``published_tokens[i]`` is what request ``i`` actually published this cycle
    (accepted tokens followed by the correction or bonus token, already limited
    by the finish rule). It is never longer than the accepted chain plus that one
    final token, and the chain prefix it selects is the prefix whose metadata is
    reported: a cycle that stopped early reports metadata for the tokens it
    published and none for the ones it did not.
    """

    logits = np.asarray(target_logits)
    if logits.ndim != 2:
        raise ValueError("target_logits must be a two-dimensional row matrix")
    if logits.shape[0] != batch.rows:
        raise ValueError("target_logits rows must align with the verified batch")
    if len(published_tokens) != len(batch.request_ids):
        raise ValueError("published_tokens must align with request_ids")
    if len(accepted_counts) != len(batch.request_ids):
        raise ValueError("accepted_counts must align with request_ids")
    prefix_states = row_prefix_states(batch, states)
    rows = chain_edge_rows(batch, accepted_counts)
    metadata: list[tuple[tuple[float | None, tuple[tuple[int, float], ...]], ...]] = []
    for index, tokens in enumerate(published_tokens):
        request_id = int(batch.request_ids[index])
        params = params_for(request_id)
        per_token: list[tuple[float | None, tuple[tuple[int, float], ...]]] = []
        for token, row in zip(tokens, rows[index]):
            per_token.append(
                reported_logprob(
                    logits[row],
                    params,
                    prefix_states[row],
                    int(token),
                    token_text_for_id=token_text_for_id,
                )
            )
        metadata.append(tuple(per_token))
    return tuple(metadata)


def row_prefix_states(
    batch: TargetVerifyBatch,
    states: Mapping[int, RowSamplingState],
) -> tuple[RowSamplingState, ...]:
    """Return each row's sampler state with that row's drafted prefix observed.

    ``batch.tokens[root_row]`` is the last committed token, which the request's
    live state has already observed; a candidate row's state therefore extends
    its parent's with that row's own token. Parents always precede children in a
    verified batch, so one forward pass resolves every row.
    """

    known = set(batch.request_ids)
    missing = known - set(int(request_id) for request_id in states)
    if missing:
        raise ValueError(f"missing sampler state for requests {sorted(missing)}")
    resolved: list[RowSamplingState | None] = [None] * batch.rows
    roots = set(int(row) for row in batch.root_rows)
    for row in range(batch.rows):
        request_id = int(batch.row_to_request[row])
        if row in roots:
            resolved[row] = states[request_id].clone()
            continue
        parent = int(batch.parent_rows[row])
        if parent < 0 or parent >= row or resolved[parent] is None:
            raise ValueError("candidate row parent must be an earlier resolved row")
        child = resolved[parent].clone()
        child.observe(int(batch.tokens[row]))
        resolved[row] = child
    return tuple(state for state in resolved if state is not None)


def _draft_distributions(batch: TargetVerifyBatch) -> tuple[SparseDistribution, ...]:
    """Return the draft law for each row's outgoing edge.

    Today's MTP proposal chain is greedy, so the draft law on the edge to a child
    row is a point mass on that child's token. The acceptance probability is then
    ``min(1, p(x)/1) = p(x)``, which is the sampled coupling's value for a
    deterministic draft; a sampled proposal would supply its own distribution
    here instead.
    """

    children: dict[int, list[int]] = {row: [] for row in range(batch.rows)}
    for row in batch.candidate_rows:
        if batch.active_mask[row]:
            children[int(batch.parent_rows[row])].append(int(row))
    drafts: list[SparseDistribution] = []
    for row in range(batch.rows):
        rows = children[row]
        if len(rows) > 1:
            raise ValueError(
                "sampled acceptance supports a single drafted chain per request"
            )
        token_id = int(batch.tokens[row] if not rows else batch.tokens[rows[0]])
        drafts.append(SparseDistribution.point_mass(token_id))
    return tuple(drafts)


def sampled_accept_summary(
    batch: TargetVerifyBatch,
    target_logits: np.ndarray,
    states: Mapping[int, RowSamplingState],
    *,
    params_for: Callable[[int], Any],
    draws: Callable[[], float],
    token_text_for_id: Callable[[int], str] | None = None,
    transaction_id: int | None = None,
    remaining_decode: Sequence[int] | None = None,
) -> TargetAcceptSummary:
    """Accept a verified draft chain by sampling from the target's own law.

    ``target_logits`` is the verifier's row-major logits matrix. ``params_for``
    returns the request's sampling params (the same object the autoregressive
    route samples with), and ``draws`` supplies uniforms from that request's
    sampler stream.
    """

    logits = np.asarray(target_logits)
    if logits.ndim != 2:
        raise ValueError("target_logits must be a two-dimensional row matrix")
    if logits.shape[0] != batch.rows:
        raise ValueError("target_logits rows must align with the verified batch")
    prefix_states = row_prefix_states(batch, states)
    targets: list[SparseDistribution] = []
    for row in range(batch.rows):
        request_id = int(batch.row_to_request[row])
        token_ids, probabilities = processed_distribution(
            logits[row],
            params_for(request_id),
            prefix_states[row],
            token_text_for_id=token_text_for_id,
        )
        targets.append(SparseDistribution.from_pairs(token_ids, probabilities))
    result = sampled_accept_from_distributions(
        batch,
        targets,
        _draft_distributions(batch),
        draws=draws,
        transaction_id=transaction_id,
        remaining_decode=remaining_decode,
        eos_token_ids=tuple(
            None if bool(getattr(params_for(rid), "ignore_eos", False))
            else getattr(params_for(rid), "eos_token_id", None)
            for rid in batch.request_ids
        ),
    )
    return TargetAcceptSummary.from_accept_result(batch, result)
