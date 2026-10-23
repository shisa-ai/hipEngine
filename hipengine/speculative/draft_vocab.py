"""Request-local draft-vocabulary policy for MTP draft-head row selection.

This module owns a fixed row prefix, a per-request extra-row table, and the
decision whether a draft cycle may use them instead of the full vocabulary. It reads
only the request's own committed token history -- never draft proposals, never
another request's tokens, and never a process-wide token-frequency count -- so
isolated, batched, and restored requests draft the same way.

Behaviour follows Gufo's ``src/models/qwen38_flash_next/draft_vocab.hpp``
(``MtpDraftVocab``) at f020bd3a037d58aa42138ca11fa895524e43cdd6: the prefix is
clipped to the vocabulary, the table is the special tokens followed by the
request's own out-of-prefix tokens in first-use order, and a cycle needs a
represented anchor and at most one newly encountered unrepresented token among
the last ``window`` committed tokens. Nothing here claims a measured speedup.
"""

from __future__ import annotations

import operator
from typing import Iterable, Sequence

__all__ = [
    "DEFAULT_EXTRA_CAPACITY",
    "DEFAULT_PREFIX_SIZE",
    "DEFAULT_WINDOW",
    "MTPDraftVocabulary",
]

#: Draft-head rows covered by the fixed prefix, before clipping to the model's
#: vocabulary size.
DEFAULT_PREFIX_SIZE = 98_304

#: Per-request table rows available for special tokens and rare tokens.
DEFAULT_EXTRA_CAPACITY = 1024

#: Committed tokens the recent-miss rule looks back over.
DEFAULT_WINDOW = 32


def _as_int(value: object, name: str) -> int:
    try:
        return operator.index(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from exc


class MTPDraftVocabulary:
    """One request's draft-head row table and small-head eligibility.

    The policy is append-only: :meth:`catch_up` is handed the request's
    committed token history and consumes the tail it has not seen yet. It never
    reads proposals, so a rejected or repaired draft cannot move the table.
    """

    __slots__ = (
        "_vocab_size",
        "_prefix_size",
        "_extra_capacity",
        "_window",
        "_special_ids",
        "_special_extras",
        "_special_set",
        "_rare_capacity",
        "_rare_ids",
        "_known",
        "_scanned",
        "_misses",
    )

    def __init__(
        self,
        vocab_size: int,
        *,
        prefix_size: int = DEFAULT_PREFIX_SIZE,
        extra_capacity: int = DEFAULT_EXTRA_CAPACITY,
        special_ids: Iterable[int] = (),
        window: int = DEFAULT_WINDOW,
    ) -> None:
        vocab = _as_int(vocab_size, "vocab_size")
        if vocab <= 0:
            raise ValueError("vocab_size must be positive")
        prefix = _as_int(prefix_size, "prefix_size")
        if prefix < 0:
            raise ValueError("prefix_size must be non-negative")
        capacity = _as_int(extra_capacity, "extra_capacity")
        if capacity < 0:
            raise ValueError("extra_capacity must be non-negative")
        lookback = _as_int(window, "window")
        if lookback < 1:
            raise ValueError("window must be positive")

        effective_prefix = min(prefix, vocab)
        specials: list[int] = []
        seen: set[int] = set()
        for raw in special_ids:
            special = _as_int(raw, "special id")
            if special < 0 or special >= vocab:
                raise ValueError(
                    f"special id {special} is outside vocabulary size {vocab}"
                )
            if special in seen:
                continue
            seen.add(special)
            specials.append(special)

        # Specials outside the prefix occupy table rows first; a special inside
        # the prefix is already scored by the prefix and costs no extra row.
        special_extras = tuple(special for special in specials if special >= effective_prefix)
        if len(special_extras) > capacity:
            raise ValueError(
                f"{len(special_extras)} special tokens outside the prefix need "
                f"extra rows but extra_capacity is {capacity}"
            )

        self._vocab_size = vocab
        self._prefix_size = effective_prefix
        self._extra_capacity = capacity
        self._window = lookback
        self._special_ids = tuple(specials)
        self._special_extras = special_extras
        self._special_set = frozenset(specials)
        self._rare_capacity = capacity - len(special_extras)
        self._rare_ids: list[int] = []
        self._known: set[int] = set()
        self._scanned = 0
        self._misses: list[int] = []

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(vocab_size={self._vocab_size}, "
            f"prefix_size={self._prefix_size}, extra_capacity={self._extra_capacity}, "
            f"window={self._window}, scanned={self._scanned}, "
            f"extras={len(self.extra_ids)})"
        )

    @property
    def vocab_size(self) -> int:
        """Model vocabulary size the policy was built for."""
        return self._vocab_size

    @property
    def prefix_size(self) -> int:
        """Draft-head rows covered by the fixed prefix, clipped to the vocabulary."""
        return self._prefix_size

    @property
    def extra_capacity(self) -> int:
        """Per-request table rows available for special and rare tokens."""
        return self._extra_capacity

    @property
    def window(self) -> int:
        """Committed tokens the recent-miss rule looks back over."""
        return self._window

    @property
    def special_ids(self) -> tuple[int, ...]:
        """Declared special token IDs, de-duplicated, in declaration order."""
        return self._special_ids

    @property
    def extra_ids(self) -> tuple[int, ...]:
        """Table rows beyond the prefix: specials, then rare IDs in table order."""
        return self._special_extras + tuple(self._rare_ids)

    @property
    def scanned(self) -> int:
        """Committed tokens already consumed by :meth:`catch_up`."""
        return self._scanned

    def reset(self) -> None:
        """Start a new history: keep the special rows, drop everything else."""
        self._rare_ids = []
        self._known = set()
        self._scanned = 0
        self._misses = []

    def catch_up(self, committed: Sequence[int]) -> tuple[int, ...]:
        """Consume ``committed[scanned:]`` and return the newly tabled rare IDs.

        ``committed`` is the request's own committed token history, in order.
        Only the tail beyond :attr:`scanned` is read, so the call is
        append-only: a longer history extends the table, an equal one is a
        no-op. A shorter history would splice two different requests onto one
        table, so it is rejected; call :meth:`reset` to start a new history.

        Every newly scanned token is validated before any state changes, so a
        bad ID raises with the policy untouched.
        """
        length = len(committed)

        if length < self._scanned:
            raise ValueError(
                f"committed history shortened from {self._scanned} to {length} "
                "tokens; call reset() to start a new history"
            )

        pending = [
            _as_int(committed[index], f"committed token at index {index}")
            for index in range(self._scanned, length)
        ]
        for offset, token in enumerate(pending):
            if token < 0 or token >= self._vocab_size:
                raise ValueError(
                    f"committed token {token} at index {self._scanned + offset} "
                    f"is outside vocabulary size {self._vocab_size}"
                )

        added: list[int] = []
        for offset, token in enumerate(pending):
            if self._represented(token):
                continue
            # A newly encountered unrepresented token: record the miss whether
            # or not a table row is free, so a full table keeps it a miss.
            self._misses.append(self._scanned + offset)
            del self._misses[:-2]
            if len(self._rare_ids) >= self._rare_capacity:
                continue
            self._known.add(token)
            self._rare_ids.append(token)
            added.append(token)

        self._scanned = length
        return tuple(added)

    def fits(self, anchor: int) -> bool:
        """Whether a cycle anchored on ``anchor`` may use the small head.

        The anchor must be represented by the prefix, a special ID, or the
        request's table, and at most one of the last :attr:`window` committed
        tokens may have been a newly encountered unrepresented token. An
        out-of-vocabulary anchor is simply a miss.
        """
        token = _as_int(anchor, "anchor")
        if token < 0 or token >= self._vocab_size:
            return False
        return self._represented(token) and self._quiet()

    def _represented(self, token: int) -> bool:
        return (
            token < self._prefix_size
            or token in self._special_set
            or token in self._known
        )

    def _quiet(self) -> bool:
        # Only the second-most-recent miss matters: if it lies outside the last
        # `window` committed tokens, at most one miss is inside the window.
        if len(self._misses) < 2:
            return True
        return self._misses[-2] + self._window < self._scanned
