"""Staged draft proposal, target rows and acceptance for Qwen4Exp MTP.

:meth:`hipengine.generation.qwen4_exp_mtp.Qwen4ExpMTPTextProvider.iter_generate`
performs draft proposal, target verification and accepted-prefix commit inside
one generator step. This module names those stages separately so a caller can
prepare a draft without running the target or consume rows supplied by another
executor:

* :func:`prepare_proposal` is draft-only. It takes no target owner, so it cannot
  step, prefill, verify or commit the target.
* :func:`verify_proposal_rows` runs the target's state-row verifier and returns
  the row results. It decides nothing and commits nothing.
* :func:`consume_proposal_rows` consumes those rows in row order and returns the
  accepted prefix. It samples only the consumed rows, so a later row that was
  scored with speculative history cannot advance RNG or processors.

The caller keeps every side effect that is not a stage decision: the device
commit of the consumed rows, the draft catch-up on a full match, the draft trim
to the target cursor, and the sampler and grammar owners.

This is the provider-stage prerequisite for shared physical MTP/AR execution.
It does not admit more than one speculative request per cycle, does not merge
candidate rows with autoregressive peers, and does not change what any stage
computes.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "Qwen4ExpMTPAcceptance",
    "Qwen4ExpMTPProposal",
    "Qwen4ExpMTPTargetRows",
    "consume_proposal_rows",
    "prepare_proposal",
    "verify_proposal_rows",
]


def _non_negative_int(value: object, name: str) -> int:
    """Parse an integral handle, rejecting a bool and a non-integral value.

    ``int(3.7)`` silently truncates to ``3`` and ``int(True)`` becomes ``1``, so
    a caller's bug would become a wrong position, depth or token id.
    """

    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not a bool")
    try:
        parsed = operator.index(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(parsed)


def _token_tuple(values: object, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name}s must be a sequence of token ids")
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name}s must be a sequence of token ids") from error
    return tuple(_non_negative_int(value, name) for value in items)


@dataclass(frozen=True, slots=True)
class Qwen4ExpMTPProposal:
    """One prepared draft chain, named without any target work.

    The target cursor is ``start_position``; the already-emitted
    ``root_token`` will be consumed at that position by verification.
    ``candidates`` are the proposed tokens at depths ``1..len(candidates)``.
    """

    start_position: int
    root_token: int
    candidates: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "start_position", _non_negative_int(self.start_position, "start_position")
        )
        object.__setattr__(
            self, "root_token", _non_negative_int(self.root_token, "root_token")
        )
        object.__setattr__(
            self, "candidates", _token_tuple(self.candidates, "candidate token id")
        )

    @property
    def rows(self) -> int:
        """Target rows this proposal verifies: the root row plus one per candidate."""

        return len(self.candidates) + 1

    @property
    def depth(self) -> int:
        return len(self.candidates)


@dataclass(frozen=True, slots=True)
class Qwen4ExpMTPTargetRows:
    """Per-row target results for one proposal, in row order.

    Row ``0`` is the root row; row ``i`` is candidate ``i``. ``logits`` is empty
    when the verifier did not capture them, and ``hidden_seeds`` holds one
    residual row per verified row, which is what the next proposal needs as its
    target hidden seed.
    """

    token_ids: tuple[int, ...]
    logits: tuple[Any, ...] = ()
    hidden_seeds: Any = None

    def __post_init__(self) -> None:
        tokens = _token_tuple(self.token_ids, "target row token id")
        if not tokens:
            raise ValueError("a target row set requires at least one row")
        object.__setattr__(self, "token_ids", tokens)
        logits = () if self.logits is None else tuple(self.logits)
        if logits and len(logits) != len(tokens):
            raise ValueError(
                f"target logits must hold one row per verified row, got "
                f"{len(logits)} for {len(tokens)} row(s)"
            )
        object.__setattr__(self, "logits", logits)
        hidden = self.hidden_seeds
        if hidden is None:
            raise ValueError("target rows require one hidden row per verified row")
        try:
            hidden_rows = len(hidden)
        except TypeError as error:
            raise TypeError("hidden_seeds must be an indexable row block") from error
        if hidden_rows != len(tokens):
            raise ValueError(
                f"target rows require one hidden row per verified row, got "
                f"{hidden_rows} for {len(tokens)} row(s)"
            )

    @property
    def rows(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class Qwen4ExpMTPAcceptance:
    """The accepted prefix of one verified proposal.

    ``tokens`` are the newly committed tokens for this step, ``accepted`` counts
    matched candidates, ``consumed_rows`` is how many target rows the caller must
    commit, and ``finish`` is the caller's own finish object for the last
    consumed row (``None`` while generation continues).
    """

    tokens: tuple[int, ...]
    accepted: int
    mismatch_token: int | None
    consumed_rows: int
    root_token: int
    root_hidden: Any
    finish: Any = None

    def __post_init__(self) -> None:
        tokens = _token_tuple(self.tokens, "accepted token id")
        if not tokens:
            raise ValueError("an acceptance requires at least one committed token")
        object.__setattr__(self, "tokens", tokens)
        accepted = _non_negative_int(self.accepted, "accepted")
        consumed = _non_negative_int(self.consumed_rows, "consumed_rows")
        if accepted > len(tokens):
            raise ValueError("accepted count cannot exceed the committed tokens")
        if consumed != len(tokens):
            raise ValueError("one target row is consumed per committed token")
        if self.mismatch_token is not None:
            object.__setattr__(
                self,
                "mismatch_token",
                _non_negative_int(self.mismatch_token, "mismatch_token"),
            )
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "consumed_rows", consumed)
        object.__setattr__(
            self, "root_token", _non_negative_int(self.root_token, "root_token")
        )

    @property
    def finished(self) -> bool:
        return self.finish is not None


def prepare_proposal(
    draft_runner: Any,
    *,
    start_position: int,
    root_token: int,
    target_hidden_seed: Any,
    draft_n_max: int,
    compact_output: bool = False,
) -> Qwen4ExpMTPProposal:
    """Prepare one draft chain. Performs no target work.

    The caller supplies the target cursor and hidden seed that produced the
    already-emitted root token, and owns the sampler, grammar and device commit.
    The draft cursor must already sit at ``start_position``.
    """

    start = _non_negative_int(start_position, "start_position")
    if int(draft_runner.position) != start:
        raise RuntimeError("Qwen4Exp target/draft cursors diverged before proposal")
    rows = draft_runner.propose_chain(
        start_token=_non_negative_int(root_token, "root_token"),
        target_hidden_seed=target_hidden_seed,
        draft_n_max=_non_negative_int(draft_n_max, "draft_n_max"),
        compact_output=bool(compact_output),
    )
    return Qwen4ExpMTPProposal(
        start_position=start,
        root_token=root_token,
        candidates=tuple(int(row.token_id) for row in rows),
    )


def verify_proposal_rows(
    target_runner: Any,
    proposal: Qwen4ExpMTPProposal,
    *,
    capture_kwargs: Mapping[str, Any] | None = None,
) -> Qwen4ExpMTPTargetRows:
    """Verify one prepared proposal on the target. Decides nothing.

    The target's state-row verifier keeps every row's state and leaves the
    commit to the caller, so acceptance can still select a prefix.
    """

    if not isinstance(proposal, Qwen4ExpMTPProposal):
        raise TypeError("verify_proposal_rows requires a prepared proposal")
    inputs = (proposal.root_token, *proposal.candidates)
    result = target_runner.verify_target_block_state_rows(
        inputs, **dict(capture_kwargs or {})
    )
    logits = result.logits
    return Qwen4ExpMTPTargetRows(
        token_ids=tuple(result.token_ids),
        logits=() if logits is None else tuple(logits),
        hidden_seeds=result.hidden_seeds,
    )


def consume_proposal_rows(
    proposal: Qwen4ExpMTPProposal,
    rows: Qwen4ExpMTPTargetRows,
    *,
    prior_tokens: Sequence[int],
    select: Callable[[Any], int],
    finish_for: Callable[[Sequence[int]], Any],
) -> Qwen4ExpMTPAcceptance:
    """Consume target rows in row order and return the accepted prefix.

    Row ``i``'s target token checks candidate ``i``; the first mismatch ends the
    cycle with the target's own token for that row, and when every candidate
    matches the last row is one more (bonus) token. ``select`` and ``finish_for``
    are the caller's own sampler and finish policy, so the accepted prefix
    advances controls exactly as the fused path did.
    """

    if not isinstance(proposal, Qwen4ExpMTPProposal):
        raise TypeError("consume_proposal_rows requires a prepared proposal")
    if not isinstance(rows, Qwen4ExpMTPTargetRows):
        raise TypeError("consume_proposal_rows requires target rows")
    if rows.rows != proposal.rows:
        raise ValueError(
            f"target rows must hold one row per proposed token, got {rows.rows} "
            f"for {proposal.rows}"
        )
    tokens: list[int] = []
    accepted = 0
    mismatch: int | None = None
    finish: Any = None
    for index, (raw_truth, candidate) in enumerate(
        zip(rows.token_ids, (*proposal.candidates, None), strict=True)
    ):
        # Sample only the consumed prefix. Later rows were scored with
        # speculative history, so they cannot advance RNG or processors once an
        # earlier candidate was rejected.
        truth = int(
            select(
                SimpleNamespace(
                    token_id=raw_truth,
                    logits=rows.logits[index] if rows.logits else None,
                )
            )
        )
        tokens.append(truth)
        if candidate is not None and truth != candidate:
            mismatch = truth
        elif candidate is not None:
            accepted += 1
        finish = finish_for((*prior_tokens, *tokens))
        if mismatch is not None or finish is not None:
            break
    return Qwen4ExpMTPAcceptance(
        tokens=tuple(tokens),
        accepted=accepted,
        mismatch_token=mismatch,
        consumed_rows=len(tokens),
        root_token=tokens[-1],
        root_hidden=rows.hidden_seeds[len(tokens) - 1],
        finish=finish,
    )
