"""Request-local prompt-lookup proposals copied from committed tokens.

This is a CPU policy mirror of ``src/models/qwen38_flash_next/prompt_lookup.hpp``
in gufo@f020bd3a037d58aa42138ca11fa895524e43cdd6. It keeps the C++ rule and
replaces the open-addressed 21-bit key table with plain Python tuples, so a key
is an exact token triple and cannot alias another triple.

The rule: the context ends in a three-token key. Among the newest 16 earlier
occurrences of that key, the one whose preceding tokens match the context
furthest back wins, and the tokens that followed it become the proposals.
Ties keep the newest occurrence because the length test is strict.

Policy choices carried over from the source:

* A request's first speculative step copies nothing. That step's position also
  bounds every later query, so a shared chat wrapper or tool header cannot count
  as a long matching answer before the answer exists.
* Greedy requests require 12 matched tokens and index everything they commit, so
  they may copy repetition inside their own answer. Sampled requests require 24
  and index only their starting context: an earlier part of a freshly generated
  answer is a poor source for new prose or code at temperature.
* The match is capped at 64 tokens, and a copy never crosses the indexed source
  boundary.
* Where the last verified copy's source continues, five matched tokens are
  enough to resume: output that edits a line of the text it is copying rejoins
  it a few tokens later. The copy must have kept at least half of its proposal,
  and the match must sit between 8 tokens before and 512 tokens after the resume
  point. A copy that kept fewer than half cannot open the shorter gate, because
  a long matching prefix can still precede a different field value or a
  different function body.

Proposals depend only on the committed prefix, so rejecting them cannot change
the target distribution: the verifier keeps its own sampling.

One instance belongs to one request. ``propose`` receives the prompt plus every
committed generated token, including the current root, and never uncommitted
drafts. It must extend what the instance has already seen; a shorter or
rewritten history raises instead of splicing two requests, so a caller starts a
new request with a new instance. There is no module-level state and no reset.

``snapshot`` and ``restore`` exist for a shared scheduler that plans a cycle and
then abandons the plan. They carry only the small decision state (request
boundary, resume point, pending copy, counters), never the token index: the
index holds derived entries for committed tokens only, so an aborted cycle
leaves it valid and the same history can be replayed. A rollback that shortens
the committed history is not supported and raises on the next ``propose``.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Sequence

#: Context key width: the three tokens a proposal is looked up by.
KEY_TOKENS = 3
#: Matched tokens a greedy request needs before copying.
MIN_MATCH = 12
#: Matched tokens a sampled request needs before copying.
SAMPLED_MIN_MATCH = 24
#: Longest backwards match considered, in context tokens.
MAX_MATCH = 64
#: Earlier occurrences examined per query, newest first.
MAX_CANDIDATES = 16
#: Matched tokens that resume an accepted copy.
RESUME_MIN_MATCH = 5
#: How far past the last copy's end a resumed copy may continue.
RESUME_WINDOW = 512
#: How far before the last copy's end a resumed copy may sit.
RESUME_BACK = 8

#: The C++ source keeps token ids in ``std::int32_t``; the same range is
#: accepted here so this mirror cannot silently take wider ids.
MAX_TOKEN_ID = 2**31


@dataclass(frozen=True, slots=True)
class PromptLookupMatch:
    """Winning earlier occurrence: ``start`` is the first proposal index."""

    start: int
    length: int


@dataclass(frozen=True, slots=True)
class PromptLookupSnapshot:
    """One cycle's decision state. Never holds the token index or the tokens."""

    request_start: int | None
    resume: int | None
    pending: tuple[int, int] | None
    match: PromptLookupMatch | None
    calls: int
    hit_cycles: int
    proposed_tokens: int
    accepted_tokens: int
    sampled: bool


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        number = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    return int(number)


def _as_tokens(committed: Sequence[int]) -> tuple[int, ...]:
    try:
        raw = tuple(committed)
    except TypeError:
        raise TypeError("committed must be a sequence of token ids") from None
    tokens: list[int] = []
    for token in raw:
        token_id = _as_int(token, "token id")
        if token_id < 0 or token_id >= MAX_TOKEN_ID:
            raise ValueError("token ids must be in [0, 2**31)")
        tokens.append(token_id)
    return tuple(tokens)


class PromptLookup:
    """One request's index over its own committed tokens.

    The request boundary is captured by the first :meth:`propose` call, so a new
    request needs a new instance rather than a reset.
    """

    def __init__(self, sampled: bool = False) -> None:
        if not isinstance(sampled, bool):
            raise TypeError("sampled must be a bool")
        self._sampled = sampled
        # Key -> end of its most recent occurrence; ``_previous[end]`` chains
        # earlier occurrences of the same key, newest first.
        self._heads: dict[tuple[int, ...], int] = {}
        self._previous: list[int] = []
        self._committed: tuple[int, ...] = ()
        self._request_start: int | None = None
        self._resume: int | None = None
        self._pending: tuple[int, int] | None = None
        self._match: PromptLookupMatch | None = None
        self._calls = 0
        self._hit_cycles = 0
        self._proposed_tokens = 0
        self._accepted_tokens = 0

    @property
    def sampled(self) -> bool:
        return self._sampled

    @property
    def request_start(self) -> int | None:
        """Committed index of the first proposal step, before it runs."""

        return self._request_start

    @property
    def indexed_tokens(self) -> int:
        """Committed tokens whose continuations may be copied."""

        return len(self._previous)

    @property
    def resume_point(self) -> int | None:
        """Committed index the last mostly accepted copy continues at."""

        return self._resume

    @property
    def last_match(self) -> PromptLookupMatch | None:
        """Winning match of the last :meth:`propose` call, if any."""

        return self._match

    def diagnostics(self) -> dict[str, int | bool]:
        """Plain snapshot of this request's copy counters.

        ``calls`` counts completed :meth:`propose` calls (a rejected argument or
        history does not count), ``hit_cycles`` counts the calls that returned a
        non-empty proposal, ``proposed_tokens`` sums those proposals, and
        ``accepted_tokens`` sums :meth:`observe`'s accepted counts.
        """

        return {
            "calls": self._calls,
            "hit_cycles": self._hit_cycles,
            "proposed_tokens": self._proposed_tokens,
            "accepted_tokens": self._accepted_tokens,
            "sampled": self._sampled,
        }

    def snapshot(self) -> PromptLookupSnapshot:
        """Capture this cycle's decision state without touching the index."""

        return PromptLookupSnapshot(
            request_start=self._request_start,
            resume=self._resume,
            pending=self._pending,
            match=self._match,
            calls=self._calls,
            hit_cycles=self._hit_cycles,
            proposed_tokens=self._proposed_tokens,
            accepted_tokens=self._accepted_tokens,
            sampled=self._sampled,
        )

    def restore(self, snapshot: PromptLookupSnapshot) -> None:
        """Roll the decision state back to ``snapshot``.

        The index keeps the entries it derived from committed tokens: an aborted
        cycle indexed exactly the history it was given, so replaying that same
        history proposes the same tokens. Restoring a first-step snapshot puts
        the request boundary back to ``None`` so the first step copies nothing
        again. A snapshot with a different sampling policy is rejected; callers
        must retain snapshots with their owning request.
        """

        if not isinstance(snapshot, PromptLookupSnapshot):
            raise TypeError("restore() needs a PromptLookupSnapshot")
        if snapshot.sampled != self._sampled:
            raise ValueError("snapshot belongs to a different request policy")
        self._request_start = snapshot.request_start
        self._resume = snapshot.resume
        self._pending = snapshot.pending
        self._match = snapshot.match
        self._calls = snapshot.calls
        self._hit_cycles = snapshot.hit_cycles
        self._proposed_tokens = snapshot.proposed_tokens
        self._accepted_tokens = snapshot.accepted_tokens

    def propose(self, committed: Sequence[int], max_tokens: int, *, require_full: bool = False) -> tuple[int, ...]:
        """Return up to ``max_tokens`` copied tokens for this committed prefix.

        The first call for a request captures the request boundary at the
        current root token, indexes the copy sources and returns nothing.
        Later calls must extend the same history. ``require_full`` declines a
        source with fewer than ``max_tokens`` remaining tokens, for schedulers
        whose reservation requires exactly that many candidates.
        """

        tokens = _as_tokens(committed)
        budget = _as_int(max_tokens, "max_tokens")
        if budget <= 0:
            raise ValueError("max_tokens must be positive")

        if self._request_start is None:
            if not tokens:
                raise ValueError(
                    "the first propose() call needs the prompt plus the current "
                    "root token"
                )
            if self._committed and tokens != self._committed:
                # Only a replayed first step reaches here with an index already
                # built: keep it exact rather than splicing a second request.
                raise ValueError(
                    "a restored first step must replay the history it indexed; a "
                    "new request needs a new PromptLookup"
                )
            # The root is the last committed token and has no continuation yet.
            self._request_start = len(tokens) - 1
            self._committed = tokens
            self._index(tokens, self._source_limit())
            self._pending = None
            self._match = None
            self._calls += 1
            return ()

        self._require_append_only(tokens)
        self._index(tokens, self._source_limit())
        self._calls += 1

        match = self._find(tokens)
        self._match = match
        if match is None:
            # Nothing was proposed, so there is no copy to resume or to observe.
            self._pending = None
            return ()
        count = min(budget, self.indexed_tokens - match.start)
        if count <= 0 or (require_full and count != budget):
            self._pending = None
            return ()
        self._pending = (match.start, count)
        self._hit_cycles += 1
        self._proposed_tokens += count
        return tokens[match.start : match.start + count]

    def observe(self, accepted: int) -> None:
        """Attribute ``accepted`` verified tokens to the pending proposal.

        The accepted count is recorded once and the pending proposal is cleared,
        so a repeated call cannot double count. ``accepted`` of zero after a
        cycle that proposed nothing is allowed; anything else without a pending
        proposal is a caller error.
        """

        kept = _as_int(accepted, "accepted")
        if kept < 0:
            raise ValueError("accepted must be non-negative")
        pending = self._pending
        if pending is None:
            if kept != 0:
                raise ValueError("observe() has no pending proposal to attribute")
            return
        start, proposed = pending
        if kept > proposed:
            raise ValueError("accepted must not exceed the proposed token count")
        self._pending = None
        self._accepted_tokens += kept
        if 2 * kept < proposed:
            self._resume = None
        else:
            self._resume = start + kept

    def _source_limit(self) -> int:
        if self._sampled:
            request_start = self._request_start
            if request_start is None:  # pragma: no cover - propose() sets it first
                return 0
            return request_start
        return len(self._committed)

    def _require_append_only(self, tokens: tuple[int, ...]) -> None:
        previous = self._committed
        if len(tokens) < len(previous):
            raise ValueError(
                "committed history shrank; a new request needs a new PromptLookup"
            )
        if previous and tokens[:len(previous)] != previous:
            raise ValueError(
                "committed history was rewritten in place; a new request needs a "
                "new PromptLookup"
            )
        self._committed = tokens

    def _index(self, tokens: tuple[int, ...], stop: int) -> None:
        """Index keys for every end index below ``stop``, append-only."""

        for end in range(len(self._previous), stop):
            previous = -1
            if end >= KEY_TOKENS:
                key = tokens[end - KEY_TOKENS : end]
                previous = self._heads.get(key, -1)
                self._heads[key] = end
            self._previous.append(previous)

    def _find(self, tokens: tuple[int, ...]) -> PromptLookupMatch | None:
        request_start = self._request_start
        if request_start is None:  # pragma: no cover - propose() sets it first
            return None
        context = len(tokens)
        # Only tokens at or after the first proposal step may be matched, so a
        # wrapper shared with the prompt cannot count as a long matching answer.
        available = context - min(context, request_start)
        max_match = min(MAX_MATCH, available)
        if max_match < KEY_TOKENS:
            return None
        minimum = SAMPLED_MIN_MATCH if self._sampled else MIN_MATCH
        indexed = len(self._previous)
        best: PromptLookupMatch | None = None
        end = self._heads.get(tokens[context - KEY_TOKENS : context])
        for _ in range(MAX_CANDIDATES):
            if end is None or end < 0:
                break
            length = KEY_TOKENS
            while (
                length < max_match
                and end > length
                and tokens[end - length - 1] == tokens[context - length - 1]
            ):
                length += 1
            resumes = (
                self._resume is not None
                and end + RESUME_BACK >= self._resume
                and end <= self._resume + RESUME_WINDOW
            )
            if (
                end < indexed
                and (best is None or length > best.length)
                and (length >= minimum or (resumes and length >= RESUME_MIN_MATCH))
            ):
                best = PromptLookupMatch(start=end, length=length)
            end = self._previous[end]
        return best


__all__ = [
    "KEY_TOKENS",
    "MAX_CANDIDATES",
    "MAX_MATCH",
    "MAX_TOKEN_ID",
    "MIN_MATCH",
    "PromptLookup",
    "PromptLookupMatch",
    "PromptLookupSnapshot",
    "RESUME_BACK",
    "RESUME_MIN_MATCH",
    "RESUME_WINDOW",
    "SAMPLED_MIN_MATCH",
]
