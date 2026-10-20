"""Immutable host layout of one shared Qwen4Exp target pass.

A shared target pass serves ordinary autoregressive (AR) requests and prepared
speculative chains in one row set. This module names that row set: which
physical slot and owner token each request occupies, where its rows start, which
token sits at every depth and position, and where each request's accepted prefix
must be committed. It is pure host metadata: it holds no device buffer, calls no
runner, and mutates nothing.

Layout, not execution
---------------------

The ordinary batch step admits one row per unique slot. The mixed target
executor consumes this layout to extend each request in depth order while
sharing stateless work across all rows. This module only validates metadata;
it does not allocate state or select a serving route.

Request groups and rows
-----------------------

One :class:`Qwen4ExpMixedRequest` is one request's contribution: its id, its
physical slot, that slot's owner token (the token ``admit()`` returned), the
append position of its root row, the already-decided root token, and the
prepared candidates at depths ``1..N``. A request with no candidates is an AR
peer. Semantic row order is request-major and depth-ascending, so a request's
rows are always contiguous in semantic order.

A slot and an owner token each appear once per pass: a shared pass admits one
request per physical slot, and an owner token names one admission. A layout that
names a slot or an admission twice is refused rather than resolved. Owner token
``0`` is the batch owner's unowned sentinel (slots start at ``0`` and ``admit()``
returns ``1..N``), so a request or row carrying ``0`` is refused: it would name
the slot without owning it.

:class:`Qwen4ExpMixedLayout` stores rows in *physical* order. Each row names its
physical index, its semantic index, its request, depth, position and token, so
an arbitrary physical permutation still maps back to ``(request, depth)``.
:func:`compile_mixed_layout` builds one from request groups and an optional
permutation.

Generic verifier compatibility
-----------------------------

:attr:`Qwen4ExpMixedLayout.verify_batch` is the shared
:class:`~hipengine.speculative.interfaces.TargetVerifyBatch` row contract, built
from this layout and validated against it;
:meth:`Qwen4ExpMixedLayout.require_verifier_compatible` re-derives it and refuses
a layout whose stored batch no longer matches. Three consequences of that
contract are worth stating, because they are where AR rows differ:

* **An AR peer contributes exactly one root row.** It has ``draft_depth == 0``,
  is absent from ``candidate_rows``, and its ``candidate_counts`` entry is ``0``.
  A generic verifier that assumes every request contributes at least one
  candidate row cannot consume an AR peer; the plan-side
  ``SpecRequestPlan``/``CandidateGraph`` types already express the zero-candidate
  case, so the layout is consistent with them rather than with that assumption.
* **A parent row must precede its child physically.** ``TargetVerifyBatch``
  resolves chain parents by row index, so a permutation that places a candidate
  before the row it extends is refused here instead of failing later. A
  permutation that interleaves requests while keeping every chain ascending is
  accepted.
* **A pass with no candidate rows has no generic work projection.** With every
  request an AR peer, ``TargetVerifyBatch.to_work_item()`` has no candidate rows
  to project and raises; the layout still describes the pass, but that projection
  does not apply.

Commit destinations
-------------------

:meth:`Qwen4ExpMixedLayout.commit_destinations` turns per-request accepted counts
into a validated
:class:`~hipengine.speculative.interfaces.TargetCommitPlan`: the physical row,
token and position each request commits. A count of ``accepted`` candidates means
the pass consumed ``accepted + 1`` input rows for that request (its root row plus
those candidates), and the destination is the *last* of them: the final state
checkpoint the commit keeps (GDN, conv, PLE, attention and position state), not
the first row and not one row per consumed row. ``commit_rows``,
``commit_tokens`` and ``commit_positions`` therefore hold one entry per request,
naming depth ``accepted``. An AR peer prepares no candidates, so its only legal
count is ``0``: one consumed row, one checkpoint, exactly one token of advance. A
non-zero count for an AR peer is refused, because it would claim rows the peer
never proposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from typing import Any, Sequence

from hipengine.speculative.interfaces import TargetCommitPlan, TargetVerifyBatch

__all__ = [
    "Qwen4ExpMixedLayout",
    "Qwen4ExpMixedRequest",
    "Qwen4ExpMixedRow",
    "compile_mixed_layout",
    "mixed_request_from_proposal",
]


def _non_negative_int(value: object, name: str) -> int:
    """Parse an integral handle, rejecting a bool and a non-integral value."""

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


def _owner_token(value: object) -> int:
    """Parse a slot owner token. ``0`` is the batch owner's unowned sentinel."""

    parsed = _non_negative_int(value, "owner_token")
    if parsed == 0:
        raise ValueError(
            "owner_token 0 means the slot is unowned; pass the token admit() "
            "returned, which starts at 1"
        )
    return parsed


@dataclass(frozen=True, slots=True)
class Qwen4ExpMixedRequest:
    """One request's rows in a shared target pass.

    ``start_position`` is the append position of the root row's token in the
    target cache; the token itself is already decided (it is the request's last
    committed token), and the pass is what appends it. Depth ``d``'s row appends
    at ``start_position + d``. ``candidates`` are the prepared tokens at depths
    ``1..N`` in chain order; an empty tuple makes this an AR peer, whose single
    row is at ``start_position`` and produces its next token.

    ``owner_token`` is the token ``admit()`` returned for ``slot``. The batch
    owner starts every slot at ``0`` and returns ``1..N`` from ``admit()``, so
    ``0`` means unowned and is refused.
    """

    request_id: int
    slot: int
    owner_token: int
    start_position: int
    root_token: int
    candidates: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_id", _non_negative_int(self.request_id, "request_id")
        )
        object.__setattr__(self, "slot", _non_negative_int(self.slot, "slot"))
        object.__setattr__(
            self, "owner_token", _owner_token(self.owner_token)
        )
        object.__setattr__(
            self,
            "start_position",
            _non_negative_int(self.start_position, "start_position"),
        )
        object.__setattr__(
            self, "root_token", _non_negative_int(self.root_token, "root_token")
        )
        object.__setattr__(
            self, "candidates", _token_tuple(self.candidates, "candidate token id")
        )

    @classmethod
    def autoregressive(
        cls,
        *,
        request_id: int,
        slot: int,
        owner_token: int,
        start_position: int,
        root_token: int,
    ) -> "Qwen4ExpMixedRequest":
        """Build the one-row group an AR peer contributes."""

        return cls(
            request_id=request_id,
            slot=slot,
            owner_token=owner_token,
            start_position=start_position,
            root_token=root_token,
        )

    @property
    def depth(self) -> int:
        """Prepared chain depth: the deepest candidate depth, ``0`` for an AR peer."""

        return len(self.candidates)

    @property
    def rows(self) -> int:
        """Rows this request contributes: the root row plus one per candidate."""

        return len(self.candidates) + 1

    @property
    def is_autoregressive(self) -> bool:
        return not self.candidates

    @property
    def tokens(self) -> tuple[int, ...]:
        """Row tokens in depth order: the root token, then the candidates."""

        return (self.root_token, *self.candidates)

    @property
    def positions(self) -> tuple[int, ...]:
        """Row append positions in depth order."""

        return tuple(self.start_position + depth for depth in range(self.rows))


@dataclass(frozen=True, slots=True)
class Qwen4ExpMixedRow:
    """One physical row of a shared target pass.

    ``owner_token`` is the admission token of ``slot``; ``0`` is the batch
    owner's unowned sentinel and is refused, as on the request group.
    """

    physical_row: int
    semantic_row: int
    request_id: int
    slot: int
    owner_token: int
    depth: int
    position: int
    token: int

    def __post_init__(self) -> None:
        for name in (
            "physical_row",
            "semantic_row",
            "request_id",
            "slot",
            "depth",
            "position",
            "token",
        ):
            object.__setattr__(self, name, _non_negative_int(getattr(self, name), name))
        object.__setattr__(self, "owner_token", _owner_token(self.owner_token))

    @property
    def is_root(self) -> bool:
        """Whether this row is its request's committed root row (depth ``0``)."""

        return self.depth == 0

    @property
    def context(self) -> int:
        """Keys this row attends to: its own position plus one.

        Derived metadata only. A root row sees its slot's committed context; a
        candidate row at depth ``d`` additionally sees the earlier rows of its own
        chain, which are at the intervening positions of the same slot. No kernel
        reads this value here.
        """

        return self.position + 1


@dataclass(frozen=True, slots=True)
class Qwen4ExpMixedLayout:
    """Immutable row layout of one shared target pass.

    ``rows`` is in physical order: ``rows[i].physical_row == i``. The stored
    ``verify_batch`` is derived from ``requests`` and ``rows`` and validated
    against them, so every layout is a generic verifier row contract or it does
    not exist.
    """

    requests: tuple[Qwen4ExpMixedRequest, ...]
    rows: tuple[Qwen4ExpMixedRow, ...]
    verify_batch: TargetVerifyBatch = field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self) -> None:
        requests = tuple(self.requests)
        rows = tuple(self.rows)
        if not requests:
            raise ValueError("a mixed layout requires at least one request")
        if not rows:
            raise ValueError("a mixed layout requires at least one row")
        for request in requests:
            if not isinstance(request, Qwen4ExpMixedRequest):
                raise TypeError("requests must be Qwen4ExpMixedRequest values")
        for row in rows:
            if not isinstance(row, Qwen4ExpMixedRow):
                raise TypeError("rows must be Qwen4ExpMixedRow values")
        object.__setattr__(self, "requests", requests)
        object.__setattr__(self, "rows", rows)
        self._validate_ownership()
        self._validate_rows()
        batch = self._build_verify_batch()
        object.__setattr__(self, "verify_batch", batch)
        self.require_verifier_compatible()

    # -- validation -------------------------------------------------------

    def _validate_ownership(self) -> None:
        seen_ids: set[int] = set()
        seen_slots: set[int] = set()
        seen_owners: set[int] = set()
        for request in self.requests:
            if request.request_id in seen_ids:
                raise ValueError(
                    f"request id {request.request_id} appears twice in one pass"
                )
            if request.slot in seen_slots:
                raise ValueError(
                    f"slot {request.slot} is claimed by two requests in one pass; "
                    "a shared pass admits one request per physical slot"
                )
            if request.owner_token in seen_owners:
                raise ValueError(
                    f"owner token {request.owner_token} is claimed by two requests "
                    "in one pass; an owner token identifies one slot admission"
                )
            seen_ids.add(request.request_id)
            seen_slots.add(request.slot)
            seen_owners.add(request.owner_token)

    def _validate_rows(self) -> None:
        for index, row in enumerate(self.rows):
            if row.physical_row != index:
                raise ValueError(
                    f"row at position {index} reports physical_row "
                    f"{row.physical_row}; rows must be stored in physical order"
                )
        known = set(self.request_ids)
        for row in self.rows:
            if row.request_id not in known:
                raise ValueError(
                    f"row {row.physical_row} names request {row.request_id}, which is "
                    "not part of this pass"
                )
        for request in self.requests:
            expected = request.rows
            actual = tuple(row for row in self.rows if row.request_id == request.request_id)
            if len(actual) != expected:
                raise ValueError(
                    f"request {request.request_id} prepares {expected} row(s) but "
                    f"the layout holds {len(actual)}"
                )
            by_depth = {row.depth: row for row in actual}
            if sorted(by_depth) != list(range(expected)):
                raise ValueError(
                    f"request {request.request_id} must hold depths 0..{expected - 1} "
                    f"exactly once, got {sorted(by_depth)}"
                )
            for depth in range(expected):
                row = by_depth[depth]
                if row.position != request.start_position + depth:
                    raise ValueError(
                        f"row {row.physical_row} position {row.position} does not match "
                        f"request {request.request_id} depth {depth} position "
                        f"{request.start_position + depth}"
                    )
                if row.token != request.tokens[depth]:
                    raise ValueError(
                        f"row {row.physical_row} token {row.token} does not match "
                        f"request {request.request_id} depth {depth} token "
                        f"{request.tokens[depth]}"
                    )
                if row.slot != request.slot:
                    raise ValueError(
                        f"row {row.physical_row} slot {row.slot} does not match "
                        f"request {request.request_id} slot {request.slot}"
                    )
                if row.owner_token != request.owner_token:
                    raise ValueError(
                        f"row {row.physical_row} owner token {row.owner_token} does not "
                        f"match request {request.request_id} owner token "
                        f"{request.owner_token}"
                    )
        total = self.total_rows
        semantic = sorted(row.semantic_row for row in self.rows)
        if semantic != list(range(total)):
            raise ValueError(
                f"semantic rows must be a permutation of 0..{total - 1}, got {semantic}"
            )
        for offset, request in enumerate(self.requests):
            base = self._semantic_base(offset)
            for depth in range(request.rows):
                row = self._row_for(request.request_id, depth)
                if row.semantic_row != base + depth:
                    raise ValueError(
                        f"row {row.physical_row} semantic_row {row.semantic_row} does "
                        f"not match the contiguous chain order: request "
                        f"{request.request_id} depth {depth} is semantic row "
                        f"{base + depth}"
                    )

    def _semantic_base(self, request_index: int) -> int:
        return sum(request.rows for request in self.requests[:request_index])

    def _row_for(self, request_id: int, depth: int) -> Qwen4ExpMixedRow:
        matches = tuple(
            row
            for row in self.rows
            if row.request_id == request_id and row.depth == depth
        )
        if len(matches) != 1:
            raise KeyError(
                f"request {request_id} depth {depth} has {len(matches)} rows"
            )
        return matches[0]

    def _build_verify_batch(self) -> TargetVerifyBatch:
        parent_rows: list[int] = []
        for row in self.rows:
            if row.is_root:
                parent_rows.append(-1)
                continue
            parent = self._row_for(row.request_id, row.depth - 1)
            if parent.physical_row >= row.physical_row:
                raise ValueError(
                    f"physical row {row.physical_row} (request {row.request_id} depth "
                    f"{row.depth}) is placed before its parent row "
                    f"{parent.physical_row}; a target verifier resolves chain parents "
                    "by row index, so every parent must come first"
                )
            parent_rows.append(parent.physical_row)
        root_rows = tuple(
            self._row_for(request.request_id, 0).physical_row
            for request in self.requests
        )
        candidate_rows = tuple(
            row.physical_row for row in self.rows if not row.is_root
        )
        return TargetVerifyBatch(
            request_ids=tuple(request.request_id for request in self.requests),
            tokens=tuple(row.token for row in self.rows),
            positions=tuple(row.position for row in self.rows),
            row_to_request=tuple(row.request_id for row in self.rows),
            parent_rows=tuple(parent_rows),
            root_rows=root_rows,
            candidate_rows=candidate_rows,
            draft_depths=tuple(row.depth for row in self.rows),
            active_mask=(True,) * self.total_rows,
            mode="verify_chain",
        )

    # -- accessors --------------------------------------------------------

    @property
    def physical_request_width(self) -> int:
        """Requests in the pass. Distinct from the row count."""

        return len(self.requests)

    @property
    def total_rows(self) -> int:
        return len(self.rows)

    @property
    def max_chain_rows(self) -> int:
        """Rows the deepest prepared chain contributes (``1`` for an AR-only pass)."""

        return max(request.rows for request in self.requests)

    @property
    def request_ids(self) -> tuple[int, ...]:
        return tuple(request.request_id for request in self.requests)

    @property
    def autoregressive_request_ids(self) -> tuple[int, ...]:
        return tuple(
            request.request_id
            for request in self.requests
            if request.is_autoregressive
        )

    def request(self, request_id: int) -> Qwen4ExpMixedRequest:
        parsed = _non_negative_int(request_id, "request_id")
        for request in self.requests:
            if request.request_id == parsed:
                return request
        raise KeyError(f"request {parsed} is not part of this pass")

    def row_at(self, physical_row: int) -> Qwen4ExpMixedRow:
        """The row at a physical position, naming its request and depth."""

        parsed = _non_negative_int(physical_row, "physical_row")
        if parsed >= self.total_rows:
            raise KeyError(
                f"physical row {parsed} is outside [0, {self.total_rows})"
            )
        return self.rows[parsed]

    def rows_for_request(self, request_id: int) -> tuple[Qwen4ExpMixedRow, ...]:
        """A request's rows in depth order, whatever their physical positions."""

        request = self.request(request_id)
        return tuple(
            self._row_for(request.request_id, depth)
            for depth in range(request.rows)
        )

    def physical_row(self, request_id: int, depth: int) -> int:
        """Physical row holding ``(request, depth)``."""

        request = self.request(request_id)
        parsed = _non_negative_int(depth, "depth")
        if parsed >= request.rows:
            raise KeyError(
                f"request {request.request_id} prepares depths 0..{request.rows - 1}, "
                f"not {parsed}"
            )
        return self._row_for(request.request_id, parsed).physical_row

    def semantic_row(self, request_id: int, depth: int) -> int:
        """Semantic (request-major, depth-ascending) index of ``(request, depth)``."""

        return self.row_at(self.physical_row(request_id, depth)).semantic_row

    # -- generic verifier contract ----------------------------------------

    def require_verifier_compatible(self) -> TargetVerifyBatch:
        """Return the generic verifier row contract, or refuse the layout.

        The stored batch is re-derived and compared, so a layout whose batch was
        replaced with another pass's contract cannot be handed to a verifier.
        """

        batch = self.verify_batch
        if not isinstance(batch, TargetVerifyBatch):
            raise ValueError(
                "a mixed layout requires a TargetVerifyBatch verifier contract"
            )
        expected = self._build_verify_batch()
        if batch != expected:
            raise ValueError(
                "the stored verifier batch does not match this layout's rows: "
                f"stored {batch!r} vs derived {expected!r}"
            )
        if batch.candidate_counts != tuple(
            request.depth for request in self.requests
        ):
            raise ValueError(
                "verifier candidate counts do not match the prepared chains"
            )
        return batch

    # -- commit destinations ----------------------------------------------

    def commit_destinations(
        self,
        accepted_counts: Sequence[int],
        *,
        transaction_id: int = 0,
        next_tokens: Sequence[int | None] | None = None,
    ) -> TargetCommitPlan:
        """Per-request commit destination for an accepted-prefix count.

        ``accepted`` counts the candidate rows the pass consumed and kept, so a
        request consumed ``accepted + 1`` input rows: its root row plus those
        candidates. The destination is the last of those rows, the final state
        checkpoint the commit keeps, so ``commit_rows``/``commit_tokens``/
        ``commit_positions`` hold one entry per request naming depth
        ``accepted``. An AR peer prepares no candidates and must pass ``0``: one
        consumed row, one checkpoint, exactly one token of advance.
        ``next_tokens`` is the caller's correction or bonus token per request,
        passed through unchanged.
        """

        counts = self._parse_accepted_counts(accepted_counts)
        tokens = self._parse_next_tokens(next_tokens)
        commit_rows: list[int] = []
        commit_tokens: list[int] = []
        commit_positions: list[int] = []
        for request, accepted in zip(self.requests, counts, strict=True):
            row = self._row_for(request.request_id, accepted)
            commit_rows.append(row.physical_row)
            commit_tokens.append(row.token)
            commit_positions.append(row.position)
        return TargetCommitPlan(
            transaction_id=transaction_id,
            request_ids=self.request_ids,
            accepted_counts=counts,
            commit_rows=tuple(commit_rows),
            commit_tokens=tuple(commit_tokens),
            commit_positions=tuple(commit_positions),
            next_tokens=tokens,
            candidate_counts=tuple(request.depth for request in self.requests),
            draft_depth=max(request.depth for request in self.requests),
            mode="verify_chain",
        )

    def _parse_accepted_counts(self, accepted_counts: object) -> tuple[int, ...]:
        if isinstance(accepted_counts, (str, bytes)):
            raise TypeError("accepted_counts must be a sequence of integers")
        try:
            values = tuple(accepted_counts)  # type: ignore[arg-type]
        except TypeError as error:
            raise TypeError("accepted_counts must be a sequence of integers") from error
        if len(values) != self.physical_request_width:
            raise ValueError(
                f"accepted_counts must hold one count per request, got {len(values)} "
                f"for {self.physical_request_width} request(s)"
            )
        counts: list[int] = []
        for request, value in zip(self.requests, values, strict=True):
            count = _non_negative_int(value, "accepted_counts entry")
            if request.is_autoregressive and count != 0:
                raise ValueError(
                    f"an autoregressive request advances exactly once, so "
                    f"accepted_counts for request {request.request_id} must be 0, "
                    f"got {count}"
                )
            if count > request.depth:
                raise ValueError(
                    f"accepted_counts for request {request.request_id} is {count} but "
                    f"it prepares only {request.depth} candidate(s)"
                )
            counts.append(count)
        return tuple(counts)

    def _parse_next_tokens(
        self, next_tokens: Sequence[int | None] | None
    ) -> tuple[int | None, ...] | None:
        if next_tokens is None:
            return None
        if isinstance(next_tokens, (str, bytes)):
            raise TypeError("next_tokens must be a sequence of token ids")
        values = tuple(next_tokens)
        if len(values) != self.physical_request_width:
            raise ValueError(
                f"next_tokens must hold one entry per request, got {len(values)} for "
                f"{self.physical_request_width} request(s)"
            )
        return tuple(
            None if value is None else _non_negative_int(value, "next_tokens entry")
            for value in values
        )


def compile_mixed_layout(
    requests: Sequence[Qwen4ExpMixedRequest],
    *,
    physical_row_order: Sequence[int] | None = None,
) -> Qwen4ExpMixedLayout:
    """Compile request groups into one validated physical row layout.

    ``physical_row_order[p]`` is the semantic row placed at physical row ``p``;
    omitting it keeps semantic order. Rows are built from the request groups, so
    a physical permutation never rewrites a row's request, depth, position or
    token. Everything is validated before the layout exists, and the inputs are
    not mutated.
    """

    groups = tuple(requests)
    if not groups:
        raise ValueError("a mixed layout requires at least one request")
    total = sum(group.rows for group in groups)
    order = _physical_row_order(physical_row_order, total)
    semantic_rows: list[tuple[Qwen4ExpMixedRequest, int]] = []
    for group in groups:
        for depth in range(group.rows):
            semantic_rows.append((group, depth))
    rows = tuple(
        Qwen4ExpMixedRow(
            physical_row=physical,
            semantic_row=semantic,
            request_id=semantic_rows[semantic][0].request_id,
            slot=semantic_rows[semantic][0].slot,
            owner_token=semantic_rows[semantic][0].owner_token,
            depth=semantic_rows[semantic][1],
            position=semantic_rows[semantic][0].start_position
            + semantic_rows[semantic][1],
            token=semantic_rows[semantic][0].tokens[semantic_rows[semantic][1]],
        )
        for physical, semantic in enumerate(order)
    )
    return Qwen4ExpMixedLayout(requests=groups, rows=rows)


def _physical_row_order(
    physical_row_order: Sequence[int] | None, total: int
) -> tuple[int, ...]:
    if physical_row_order is None:
        return tuple(range(total))
    if isinstance(physical_row_order, (str, bytes)):
        raise TypeError("physical_row_order must be a sequence of row indices")
    values = tuple(
        _non_negative_int(value, "physical_row_order entry")
        for value in physical_row_order
    )
    if len(values) != total:
        raise ValueError(
            f"physical_row_order must name every row, got {len(values)} for "
            f"{total} row(s)"
        )
    if sorted(values) != list(range(total)):
        raise ValueError(
            f"physical_row_order must be a permutation of 0..{total - 1}, got "
            f"{values}"
        )
    return values


def mixed_request_from_proposal(
    proposal: Any,
    *,
    request_id: int,
    slot: int,
    owner_token: int,
) -> Qwen4ExpMixedRequest:
    """Adapt a prepared draft chain into one request group.

    ``proposal`` is a prepared
    :class:`~hipengine.generation.qwen4_exp_mtp_stages.Qwen4ExpMTPProposal`. The
    runtime layer deliberately does not import the generation layer, so this
    reads the three fields the prepared chain owns instead of the type.
    """

    for name in ("start_position", "root_token", "candidates"):
        if not hasattr(proposal, name):
            raise TypeError(
                "mixed_request_from_proposal requires a prepared proposal naming "
                f"start_position, root_token and candidates; {type(proposal).__name__} "
                f"has no {name}"
            )
    return Qwen4ExpMixedRequest(
        request_id=request_id,
        slot=slot,
        owner_token=owner_token,
        start_position=proposal.start_position,
        root_token=proposal.root_token,
        candidates=proposal.candidates,
    )
