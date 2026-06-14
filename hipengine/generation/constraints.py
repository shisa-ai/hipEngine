"""Tokenizer-agnostic decode constraint primitives."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenSequenceDFAState:
    """Incremental matcher for token-id sequences.

    The state keeps only the longest suffix that is also a prefix of at least
    one configured sequence. That is enough to detect future full matches while
    staying independent of tokenizer/model code.
    """

    sequences: tuple[tuple[int, ...], ...] = ()
    suffix: tuple[int, ...] = ()
    matched_sequence: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequences", normalize_token_sequences(self.sequences))
        object.__setattr__(self, "suffix", tuple(int(token) for token in self.suffix))
        object.__setattr__(self, "matched_sequence", tuple(int(token) for token in self.matched_sequence))

    @classmethod
    def from_sequences(cls, sequences: Iterable[Iterable[int]] | None) -> "TokenSequenceDFAState":
        return cls(sequences=normalize_token_sequences(sequences))

    @property
    def matched(self) -> bool:
        return bool(self.matched_sequence)

    def observe(self, token_id: int) -> "TokenSequenceDFAState":
        if not self.sequences or self.matched_sequence:
            return self
        history = (*self.suffix, int(token_id))
        matched = _matched_sequence(history, self.sequences)
        if matched:
            return TokenSequenceDFAState(
                sequences=self.sequences,
                suffix=matched,
                matched_sequence=matched,
            )
        return TokenSequenceDFAState(
            sequences=self.sequences,
            suffix=_longest_prefix_suffix(history, self.sequences),
        )

    def observe_many(self, token_ids: Iterable[int]) -> "TokenSequenceDFAState":
        state = self
        for token_id in token_ids:
            state = state.observe(int(token_id))
            if state.matched:
                break
        return state

    def to_json_dict(self) -> dict[str, Any]:
        if self.matched_sequence:
            return {"matched_sequence": list(self.matched_sequence)}
        if not self.suffix:
            return {}
        return {
            "partial_suffix": list(self.suffix),
            "candidate_sequences": [
                list(sequence)
                for sequence in self.sequences
                if sequence[: len(self.suffix)] == self.suffix
            ],
        }


@dataclass(slots=True)
class ForcedTokenQueue:
    """Mutable FIFO queue for tokens that must be emitted before sampling."""

    tokens: Iterable[int] = ()
    reason: str | None = None
    _pending: list[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pending = [int(token) for token in self.tokens]
        if any(token < 0 for token in pending):
            raise ValueError("forced tokens must be non-negative token ids")
        self._pending = pending
        self.reason = None if self.reason is None else str(self.reason)

    @property
    def pending_tokens(self) -> tuple[int, ...]:
        return tuple(self._pending)

    def __bool__(self) -> bool:
        return bool(self._pending)

    def __len__(self) -> int:
        return len(self._pending)

    def peek(self) -> int | None:
        return self._pending[0] if self._pending else None

    def pop(self) -> int | None:
        if not self._pending:
            return None
        return self._pending.pop(0)

    def extend(self, token_ids: Iterable[int], *, reason: str | None = None) -> None:
        tokens = tuple(int(token) for token in token_ids)
        if any(token < 0 for token in tokens):
            raise ValueError("forced tokens must be non-negative token ids")
        self._pending.extend(tokens)
        if reason is not None:
            self.reason = str(reason)

    def to_json_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"pending_tokens": list(self._pending)}
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


def normalize_token_sequences(sequences: Iterable[Iterable[int]] | None) -> tuple[tuple[int, ...], ...]:
    if sequences is None:
        return ()
    normalized: list[tuple[int, ...]] = []
    for raw_sequence in sequences:
        sequence = tuple(int(token) for token in raw_sequence)
        if not sequence:
            continue
        if any(token < 0 for token in sequence):
            raise ValueError("token sequences must contain non-negative token ids")
        if sequence not in normalized:
            normalized.append(sequence)
    return tuple(normalized)


def token_sequence_state_for_tokens(
    token_ids: Iterable[int],
    sequences: Iterable[Iterable[int]] | None,
) -> TokenSequenceDFAState:
    return TokenSequenceDFAState.from_sequences(sequences).observe_many(token_ids)


def _matched_sequence(
    token_ids: Sequence[int],
    sequences: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    matched: tuple[int, ...] = ()
    for sequence in sequences:
        if len(sequence) > len(token_ids):
            continue
        if tuple(int(token) for token in token_ids[-len(sequence) :]) == sequence and len(sequence) > len(matched):
            matched = sequence
    return matched


def _longest_prefix_suffix(
    token_ids: Sequence[int],
    sequences: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    best: tuple[int, ...] = ()
    tokens = tuple(int(token) for token in token_ids)
    for sequence in sequences:
        max_len = min(len(sequence) - 1, len(tokens))
        for length in range(1, max_len + 1):
            suffix = tokens[-length:]
            if suffix == sequence[:length] and len(suffix) > len(best):
                best = suffix
    return best


__all__ = [
    "ForcedTokenQueue",
    "TokenSequenceDFAState",
    "normalize_token_sequences",
    "token_sequence_state_for_tokens",
]
