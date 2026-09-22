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
    "observe_published_tokens",
    "published_forced_count",
    "row_forced_token_ids",
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
    observe_text: Callable[[RowSamplingState, int, int], None] | None = None,
) -> tuple[tuple[tuple[float | None, tuple[tuple[int, float], ...]], ...], ...]:
    """Return each request's published-token logprob metadata from verified rows.

    ``published_tokens[i]`` is what request ``i`` actually published this cycle
    (accepted tokens followed by the correction or bonus token, already limited
    by the finish rule). It is never longer than the accepted chain plus that one
    final token, and the chain prefix it selects is the prefix whose metadata is
    reported: a cycle that stopped early reports metadata for the tokens it
    published and none for the ones it did not.

    ``observe_text`` is the same per-row text observer the accept walk used. A
    constrained row's support is masked from the text of the tokens before it, so
    scoring that row against an unobserved state could report ``-inf`` for a
    token the cycle published.
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
    prefix_states = row_prefix_states(batch, states, observe_text=observe_text)
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
    *,
    observe_text: Callable[[RowSamplingState, int, int], None] | None = None,
) -> tuple[RowSamplingState, ...]:
    """Return each row's sampler state with that row's drafted prefix observed.

    ``batch.tokens[root_row]`` is the last committed token, which the request's
    live state has already observed; a candidate row's state therefore extends
    its parent's with that row's own token. Parents always precede children in a
    verified batch, so one forward pass resolves every row.
    """

    return tuple(
        state
        for state, _forced in _row_prefix_walk(
            batch, states, observe_text=observe_text
        )
    )


def row_forced_token_ids(
    batch: TargetVerifyBatch,
    states: Mapping[int, RowSamplingState],
    *,
    observe_text: Callable[[RowSamplingState, int, int], None] | None = None,
) -> tuple[int | None, ...]:
    """Return each row's pending forced-token override, or ``None``.

    Row ``r`` predicts the token after the prefix ending at ``r``, so the
    override governing its outgoing position is the head of the forced queue
    once that row's own token has been observed. The queue drains as the walk
    descends: the token a row emits is consumed for its children, so a chain that
    publishes forced tokens in order sees the queue's second entry at the second
    row rather than the same head at every row.

    The live request's queue is never touched. A cycle consumes the overrides it
    published only at commit time (``observe_published_tokens``), because a row
    that the accept walk never reaches published nothing and earned no pop.

    ``observe_text`` is the same walk-time text observer ``row_prefix_states``
    takes, and it must be the same one: a text-keyed constraint *queues* a
    closing suffix once its own DFA reaches the point where the remaining budget
    is exactly enough to close, so a row that skips the text also skips the
    override that governs its edge.
    """

    return tuple(
        forced
        for _state, forced in _row_prefix_walk(batch, states, observe_text=observe_text)
    )


def _row_prefix_walk(
    batch: TargetVerifyBatch,
    states: Mapping[int, RowSamplingState],
    *,
    observe_text: Callable[[RowSamplingState, int, int], None] | None = None,
) -> tuple[tuple[RowSamplingState, int | None], ...]:
    """Resolve every row's state and the forced override governing its edge.

    This is the single traversal behind ``row_prefix_states`` and
    ``row_forced_token_ids``: the row-to-row mapping is the part that is easy to
    get wrong by one, so both readers share one implementation. A row's state is
    the state ``_process_row`` would see at that point of the autoregressive
    decode -- the forced override it is about to emit is still at the head of the
    queue -- while a child's state has its parent's override already consumed.
    """

    known = set(batch.request_ids)
    missing = known - set(int(request_id) for request_id in states)
    if missing:
        raise ValueError(f"missing sampler state for requests {sorted(missing)}")
    resolved: list[RowSamplingState | None] = [None] * batch.rows
    forced_ids: list[int | None] = [None] * batch.rows
    depths: list[int] = [0] * batch.rows
    roots = set(int(row) for row in batch.root_rows)
    for row in range(batch.rows):
        request_id = int(batch.row_to_request[row])
        if row in roots:
            state = states[request_id].clone()
        else:
            parent = int(batch.parent_rows[row])
            if parent < 0 or parent >= row or resolved[parent] is None:
                raise ValueError("candidate row parent must be an earlier resolved row")
            state = resolved[parent].clone()
            token = int(batch.tokens[row])
            state.observe(token)
            depths[row] = depths[parent] + 1
            if observe_text is not None:
                # A text-keyed constraint (json_object close forcing, tool-call
                # constraints) masks a row from the text of the tokens before it.
                # Observing the parent's token here is what makes a cycle's later
                # rows see the DFA state ``_process_row`` would see at that point
                # of an autoregressive decode. ``depths[row]`` is the number of
                # tokens this cycle published before this row's own token.
                observe_text(state, token, depths[row])
            if forced_ids[parent] is not None:
                # The parent's override was the token emitted at this row's
                # position, so this row's queue has already consumed it. When the
                # draft disagrees with that override the accept walk stops at the
                # parent and this row is never published, so consuming it here
                # cannot lose a token the request still owes.
                state.pop_forced_token()
        resolved[row] = state
        forced_ids[row] = state.peek_forced_token()
    return tuple(
        (state, forced)
        for state, forced in zip(resolved, forced_ids, strict=True)
        if state is not None
    )


def published_forced_count(
    batch: TargetVerifyBatch,
    accepted_counts: Sequence[int],
    published: Sequence[int],
    forced_ids: Sequence[int | None],
) -> int:
    """Return how many of a cycle's published tokens carried a forced override.

    ``forced_ids`` comes from ``row_forced_token_ids`` and is aligned with the
    verified batch, while ``published`` is what the cycle actually emitted -- the
    accept walk's chain, shortened by the finish rule when it limited one. Only
    the positions the cycle published earned a pop from the live queue, so a stop
    inside the chain must not consume the overrides behind it.
    """

    path = chain_edge_rows(batch, accepted_counts)[0]
    return sum(1 for row in path[: len(published)] if forced_ids[row] is not None)


def observe_published_tokens(
    state: RowSamplingState,
    published: Sequence[int],
    *,
    forced_count: int,
    observe_text: Callable[[RowSamplingState, int, int], None] | None = None,
) -> None:
    """Observe a cycle's published tokens into the request's live sampler state.

    This mirrors one autoregressive decode step per published token: prepare the
    selection, consume the forced override when this token is one of the ones the
    cycle published because of it, then observe the token. ``forced_count`` is
    the number of leading published tokens that carried an override; a forced
    token can only be published at the position its row predicted, so the
    consumed override must be the token itself.
    """

    remaining = int(forced_count)
    if remaining < 0 or remaining > len(published):
        raise ValueError(
            f"forced_count {forced_count} is outside the {len(published)} published tokens"
        )
    for index, token in enumerate(published):
        emitted = int(token)
        if index < remaining:
            state.prepare_for_selection()
            forced = state.pop_forced_token()
            if forced != emitted:
                raise RuntimeError(
                    f"cycle published token {emitted} at position {index} but the "
                    f"pending forced token is {forced}"
                )
        state.observe(emitted)
        if observe_text is not None:
            # One autoregressive step per published token, in order: the live
            # request's DFA and its queued closing suffix advance exactly as they
            # would have on the autoregressive route. Rows the accept walk never
            # reached published nothing and advance nothing.
            observe_text(state, emitted, index)


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
    observe_text: Callable[[RowSamplingState, int, int], None] | None = None,
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
    prefix_states = row_prefix_states(batch, states, observe_text=observe_text)
    forced_ids = row_forced_token_ids(batch, states, observe_text=observe_text)
    targets: list[SparseDistribution | None] = []
    for row in range(batch.rows):
        request_id = int(batch.row_to_request[row])
        row_state = prefix_states[row]
        if row_state.token_text_constraint_invalid:
            # The draft token this row was advanced by violates a text-keyed
            # constraint, so no token can follow it and the row has no law. It is
            # also unreachable: the row before it is masked against the same DFA
            # and excludes that token, so the accept walk rejects there and
            # corrects. ``None`` records exactly that, and the accept walk raises
            # if it ever steps onto the row.
            targets.append(None)
            continue
        token_ids, probabilities = processed_distribution(
            logits[row],
            params_for(request_id),
            row_state,
            token_text_for_id=token_text_for_id,
            forced_token_id=forced_ids[row],
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
