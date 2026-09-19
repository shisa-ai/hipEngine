"""Storage for draft-provider prefix checkpoints.

A prefix-cache hit cannot be primed from prompt hidden rows: the reused tokens
are never prefilled, so those rows do not exist. The design for serving such a
row from a checkpoint instead (worklog
``20260919T204437.506351Z-lhl-mtp2-prefix-checkpoint-design-276f2b``) left one
decision open: where the checkpoint lives, and what ends its life.

This module answers it with a side table rather than by attaching device buffers
to radix nodes:

* **The key is the token prefix, and the block ids are a validation, not the
  key.** A radix node id is not stable content -- the allocator reuses freed
  block ids for different tokens -- so a table keyed on ids alone could hand a
  row a checkpoint that describes a different prefix. Keying on the tokens and
  requiring the caller's current block ids to match the ones recorded at capture
  time gives both properties: a hit means "this exact prefix is still the one
  these blocks hold", and a reused id fails the check instead of matching.
* **The table owns the lifetime, not the radix tree.** Attaching a release
  callback to a radix node would put device-buffer ownership inside the cache
  that does not allocate it. Here the table holds an explicit ``release``
  callback and calls it exactly once per entry it drops, whether that is an
  overwrite, an LRU eviction, or a clear.
* **Capacity is bounded, and a dropped entry is not a correctness problem.** A
  checkpoint that has been evicted simply means the row falls back to
  autoregressive decoding, which is what every prefix-hit row does today. That
  is why an LRU cap is an acceptable answer to "the radix tree never told us it
  evicted": the failure mode of being wrong is a slower row, never a wrong one.

The type is deliberately payload-agnostic. It stores whatever object the
provider hands it and calls the callback it was given, so the provider keeps
ownership of its own buffer layout and this module never learns it.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["PrefixCheckpointStore", "PrefixCheckpointStoreStats"]


@dataclass(frozen=True, slots=True)
class PrefixCheckpointStoreStats:
    """What the store has done, for the served diagnostics block."""

    entries: int
    capacity: int
    stored: int
    hits: int
    misses: int
    block_id_mismatches: int
    evicted: int
    replaced: int
    released: int


@dataclass(slots=True)
class _Entry:
    tokens: tuple[int, ...]
    block_ids: tuple[int, ...]
    payload: Any
    # A payload is released once. The flag is what makes ``clear`` after an
    # eviction, or a double ``drop``, harmless rather than a double free.
    released: bool = field(default=False)


class PrefixCheckpointStore:
    """Bounded token-prefix-keyed store of provider prefix checkpoints.

    ``release`` is called with the stored payload when an entry leaves the
    table. It must be safe to call from ``put``/``drop``/``clear`` (the
    provider's own release is idempotent) and must not raise for an
    already-released payload.
    """

    def __init__(
        self,
        *,
        capacity: int,
        release: Callable[[Any], None],
    ) -> None:
        if capacity < 0:
            raise ValueError("prefix checkpoint store capacity must be non-negative")
        if not callable(release):
            raise ValueError("prefix checkpoint store requires a release callback")
        self._capacity = int(capacity)
        self._release = release
        self._entries: OrderedDict[tuple[int, ...], _Entry] = OrderedDict()
        self._stored = 0
        self._hits = 0
        self._misses = 0
        self._block_id_mismatches = 0
        self._evicted = 0
        self._replaced = 0
        self._released = 0

    # -- introspection ---------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, tokens: object) -> bool:
        try:
            key = _token_key(tokens)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return key in self._entries

    def stats(self) -> PrefixCheckpointStoreStats:
        return PrefixCheckpointStoreStats(
            entries=len(self._entries),
            capacity=self._capacity,
            stored=self._stored,
            hits=self._hits,
            misses=self._misses,
            block_id_mismatches=self._block_id_mismatches,
            evicted=self._evicted,
            replaced=self._replaced,
            released=self._released,
        )

    def token_keys(self) -> tuple[tuple[int, ...], ...]:
        """Return the stored prefixes, most recently used first."""

        return tuple(reversed(tuple(self._entries)))

    # -- mutation --------------------------------------------------------

    def put(
        self,
        tokens: Sequence[int],
        block_ids: Sequence[int],
        payload: Any,
    ) -> None:
        """Store ``payload`` for ``tokens``, replacing any entry for them.

        A zero-capacity store releases the payload immediately: the caller keeps
        one code path whether or not caching is enabled.
        """

        key = _token_key(tokens)
        blocks = _block_key(block_ids)
        if not key or not blocks:
            raise ValueError("a prefix checkpoint needs a non-empty prefix and block list")
        if self._capacity == 0:
            self._release_payload(payload)
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._replaced += 1
            self._release_entry(previous)
        self._entries[key] = _Entry(tokens=key, block_ids=blocks, payload=payload)
        self._stored += 1
        while len(self._entries) > self._capacity:
            _key, evicted = self._entries.popitem(last=False)
            self._evicted += 1
            self._release_entry(evicted)

    def get(
        self,
        tokens: Sequence[int],
        block_ids: Sequence[int],
    ) -> Any | None:
        """Return the checkpoint for ``tokens`` when ``block_ids`` still match.

        A token hit whose block ids differ is a *miss*: the blocks no longer
        hold that prefix, so the checkpoint describes state the row cannot use.
        The mismatching entry is dropped rather than kept, because the id it
        recorded has been reused and it can never be validated again.
        """

        try:
            key = _token_key(tokens)
            blocks = _block_key(block_ids)
        except (TypeError, ValueError):
            self._misses += 1
            return None
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.block_ids != blocks:
            self._block_id_mismatches += 1
            self._misses += 1
            del self._entries[key]
            self._release_entry(entry)
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return entry.payload

    def drop(self, tokens: Sequence[int]) -> bool:
        """Release and remove the entry for ``tokens``; True when one existed."""

        key = _token_key(tokens)
        entry = self._entries.pop(key, None)
        if entry is None:
            return False
        self._release_entry(entry)
        return True

    def drop_missing(self, keep: Iterable[Sequence[int]]) -> int:
        """Release every entry whose prefix is not in ``keep``; return the count.

        This is the sweep a caller runs after it knows which prefixes are still
        resident. It is not required for correctness -- ``get`` validates block
        ids -- but it is how a caller keeps the table from filling with
        checkpoints whose blocks the cache has already given away.
        """

        keep_keys = {_token_key(item) for item in keep}
        dropped = 0
        for key in tuple(self._entries):
            if key in keep_keys:
                continue
            entry = self._entries.pop(key)
            self._release_entry(entry)
            dropped += 1
        return dropped

    def clear(self) -> int:
        """Release every entry; return the number released."""

        count = len(self._entries)
        for entry in self._entries.values():
            self._release_entry(entry)
        self._entries.clear()
        return count

    # -- internals -------------------------------------------------------

    def _release_entry(self, entry: _Entry) -> None:
        if entry.released:
            return
        entry.released = True
        self._release_payload(entry.payload)

    def _release_payload(self, payload: Any) -> None:
        self._release(payload)
        self._released += 1


def _token_key(tokens: Sequence[int]) -> tuple[int, ...]:
    if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Sequence):
        raise TypeError("prefix tokens must be a sequence of integers")
    return tuple(int(token) for token in tokens)


def _block_key(block_ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(block_ids, (str, bytes)) or not isinstance(block_ids, Sequence):
        raise TypeError("prefix block ids must be a sequence of integers")
    return tuple(int(block) for block in block_ids)
