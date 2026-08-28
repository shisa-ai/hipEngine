"""Request-local exact n-gram proposals for speculative target verification.

The proposal rule follows llama.cpp's ``ngram-mod`` confidence shape: a long
committed-token suffix must match before a short verifier prefix is returned.
Unlike llama.cpp's process-wide modulo table, hipEngine keeps exact request-local
keys and copies a bounded contiguous prior continuation without wrapping through
the current suffix. That prevents hash-collision candidates, cyclic replay,
cross-tenant token leakage, benchmark-order dependence, and batch-composition
dependence while retaining the useful zero-model-compute proposal mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class NgramModConfig:
    """Bounded request-local continuation policy."""

    n_match: int = 24
    min_draft_tokens: int = 24
    max_probe_tokens: int = 64

    def __post_init__(self) -> None:
        n_match = int(self.n_match)
        minimum = int(self.min_draft_tokens)
        maximum = int(self.max_probe_tokens)
        if n_match <= 0:
            raise ValueError("n_match must be positive")
        if minimum <= 0:
            raise ValueError("min_draft_tokens must be positive")
        if maximum < minimum:
            raise ValueError(
                "max_probe_tokens must be at least min_draft_tokens"
            )
        object.__setattr__(self, "n_match", n_match)
        object.__setattr__(self, "min_draft_tokens", minimum)
        object.__setattr__(self, "max_probe_tokens", maximum)


@dataclass(frozen=True, slots=True)
class NgramModProposal:
    """A verifier-bounded prefix backed by a longer exact cache continuation."""

    candidate_tokens: tuple[int, ...]
    probed_tokens: int
    n_match: int
    committed_tokens: int

    def __post_init__(self) -> None:
        candidates = tuple(int(token) for token in self.candidate_tokens)
        if not candidates or any(token < 0 for token in candidates):
            raise ValueError("candidate_tokens must be non-empty and non-negative")
        probed = int(self.probed_tokens)
        n_match = int(self.n_match)
        committed = int(self.committed_tokens)
        if probed < len(candidates):
            raise ValueError("probed_tokens must cover every returned candidate")
        if n_match <= 0 or committed < n_match:
            raise ValueError("proposal requires a complete committed n-gram")
        object.__setattr__(self, "candidate_tokens", candidates)
        object.__setattr__(self, "probed_tokens", probed)
        object.__setattr__(self, "n_match", n_match)
        object.__setattr__(self, "committed_tokens", committed)


class NgramModRequestCache:
    """Exact latest-continuation index for one request's committed history."""

    def __init__(self, config: NgramModConfig | None = None) -> None:
        self.config = config or NgramModConfig()
        self._committed_tokens: tuple[int, ...] = ()
        self._next_by_ngram: dict[tuple[int, ...], tuple[int, int]] = {}
        self.proposal_calls = 0
        self.proposal_hits = 0
        self.probed_tokens = 0

    @property
    def committed_tokens(self) -> tuple[int, ...]:
        return self._committed_tokens

    @property
    def indexed_ngrams(self) -> int:
        return len(self._next_by_ngram)

    def sync_committed(self, tokens: Sequence[int]) -> None:
        """Synchronize from an append-only authoritative token history.

        A cancellation, prefix restore, or caller repair may replace/shrink the
        history.  Such a mismatch rebuilds the exact index rather than retaining
        candidates learned from rolled-back tokens.
        """

        history = tuple(int(token) for token in tokens)
        if any(token < 0 for token in history):
            raise ValueError("committed token IDs must be non-negative")
        previous = self._committed_tokens
        append_only = (
            len(history) >= len(previous)
            and history[: len(previous)] == previous
        )
        if not append_only:
            self._next_by_ngram.clear()
            start = self.config.n_match
        else:
            start = max(self.config.n_match, len(previous))
        n = self.config.n_match
        for next_index in range(start, len(history)):
            key = history[next_index - n : next_index]
            self._next_by_ngram[key] = (history[next_index], next_index)
        self._committed_tokens = history

    def propose(self, *, max_candidates: int) -> NgramModProposal | None:
        """Return a short verified prefix only after a long exact continuation.

        The speculative extension is never inserted into the committed index.
        A later call first synchronizes from scheduler-authoritative history, so
        rejected candidates and cancelled cycles cannot leak into future drafts.
        """

        budget = int(max_candidates)
        if budget <= 0:
            raise ValueError("max_candidates must be positive")
        self.proposal_calls += 1
        n = self.config.n_match
        if len(self._committed_tokens) < n:
            return None
        required = max(budget, self.config.min_draft_tokens)
        probe_limit = min(required, self.config.max_probe_tokens)
        if probe_limit < required:
            return None
        match = self._next_by_ngram.get(self._committed_tokens[-n:])
        if match is None:
            return None
        _first_token, source_index = match
        # The source continuation must end before the current trailing n-gram.
        # This is the exact, collision-free analogue of llama.cpp's confidence
        # gate without allowing a deterministic key cycle to manufacture an
        # arbitrarily long draft from one short repeated suffix.
        current_suffix_start = len(self._committed_tokens) - n
        if source_index + probe_limit > current_suffix_start:
            return None
        drafted = self._committed_tokens[
            source_index : source_index + probe_limit
        ]
        if len(drafted) < self.config.min_draft_tokens:
            return None
        self.proposal_hits += 1
        self.probed_tokens += len(drafted)
        return NgramModProposal(
            candidate_tokens=tuple(drafted[:budget]),
            probed_tokens=len(drafted),
            n_match=n,
            committed_tokens=len(self._committed_tokens),
        )

    def telemetry(self) -> dict[str, int]:
        return {
            "committed_tokens": len(self._committed_tokens),
            "indexed_ngrams": len(self._next_by_ngram),
            "proposal_calls": int(self.proposal_calls),
            "proposal_hits": int(self.proposal_hits),
            "probed_tokens": int(self.probed_tokens),
        }


class RequestLocalNgramMod:
    """Request-id keyed owner for scheduler-side n-gram proposal state."""

    def __init__(self, config: NgramModConfig | None = None) -> None:
        self.config = config or NgramModConfig()
        self._requests: dict[int, NgramModRequestCache] = {}

    @property
    def request_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._requests))

    def propose(
        self,
        request_id: int,
        committed_tokens: Sequence[int],
        *,
        max_candidates: int,
    ) -> NgramModProposal | None:
        rid = int(request_id)
        if rid < 0:
            raise ValueError("request_id must be non-negative")
        state = self._requests.get(rid)
        if state is None:
            state = NgramModRequestCache(self.config)
            self._requests[rid] = state
        state.sync_committed(committed_tokens)
        return state.propose(max_candidates=max_candidates)

    def telemetry(self, request_id: int) -> dict[str, int]:
        state = self._requests.get(int(request_id))
        return {} if state is None else state.telemetry()

    def release_request(self, request_id: int) -> None:
        self._requests.pop(int(request_id), None)

    def close(self) -> None:
        self._requests.clear()


__all__ = [
    "NgramModConfig",
    "NgramModProposal",
    "NgramModRequestCache",
    "RequestLocalNgramMod",
]
