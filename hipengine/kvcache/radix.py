"""Token-id radix prefix cache scaffolding.

This is a host-only model of the prefix-cache contract used by C5.  It indexes
committed token prefixes at complete KV-block boundaries, so partial tail blocks
are never reused until they become complete.  The dynamic pool remains the owner
of block refcounts; this trie records which live request ids keep a prefix entry
valid and removes those refs on cancellation/reclaim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

PREFIX_CACHE_CHOICES = ("off", "radix")


def resolve_prefix_cache_mode(value: str | None) -> str:
    """Normalize a user-facing prefix-cache mode."""

    mode = "off" if value is None or value == "" else str(value).strip().lower()
    if mode not in PREFIX_CACHE_CHOICES:
        raise ValueError(f"prefix cache must be one of {PREFIX_CACHE_CHOICES!r}")
    return mode


@dataclass(frozen=True, slots=True)
class PrefixCacheMatch:
    """Best block-aligned prefix-cache match for a token row."""

    matched_tokens: tuple[int, ...]
    block_ids: tuple[int, ...]
    remaining_tokens: tuple[int, ...]

    @property
    def hit(self) -> bool:
        return bool(self.block_ids)

    @property
    def matched_token_count(self) -> int:
        return len(self.matched_tokens)

    @property
    def matched_block_count(self) -> int:
        return len(self.block_ids)


@dataclass(frozen=True, slots=True)
class PrefixCacheInsert:
    """Summary of one request insertion into the prefix cache."""

    request_id: int
    cached_tokens: int
    cached_blocks: int


@dataclass(frozen=True, slots=True)
class PrefixCacheCancel:
    """Summary of removing one request's prefix-cache ownership."""

    request_id: int
    removed_entries: int
    removed_blocks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PrefixCacheEntryState:
    """Pointer-independent state attached to one live radix prefix node."""

    matched_tokens: tuple[int, ...]
    block_ids: tuple[int, ...]
    owner_request_ids: tuple[int, ...]
    eviction_state: str

    @property
    def refcount(self) -> int:
        return len(self.owner_request_ids)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "matched_tokens": list(self.matched_tokens),
            "block_ids": list(self.block_ids),
            "owner_request_ids": list(self.owner_request_ids),
            "refcount": self.refcount,
            "eviction_state": self.eviction_state,
        }


@dataclass(frozen=True, slots=True)
class PrefixCacheStats:
    """Small stats payload for diagnostics and metrics."""

    entries: int
    live_requests: int
    hits: int
    misses: int
    partial_block_misses: int

    def to_json_dict(self) -> dict[str, int]:
        return {
            "entries": self.entries,
            "live_requests": self.live_requests,
            "hits": self.hits,
            "misses": self.misses,
            "partial_block_misses": self.partial_block_misses,
        }


@dataclass(slots=True)
class _RadixNode:
    children: dict[int, "_RadixNode"] = field(default_factory=dict)
    block_ids: tuple[int, ...] = ()
    owner_request_ids: set[int] = field(default_factory=set)
    eviction_state: str = "resident"

    @property
    def live(self) -> bool:
        return bool(self.block_ids) and bool(self.owner_request_ids)


class RadixCache:
    """Block-aligned token-id trie for prefix sharing.

    ``insert`` records prefix nodes only at full ``block_size`` token
    boundaries.  ``match`` returns the deepest live block-aligned prefix.  A
    request owns every node it inserted; ``cancel`` removes that ownership so a
    cancelled request no longer keeps prefix blocks reusable.
    """

    def __init__(self, *, block_size: int, mode: str = "radix") -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        resolved = resolve_prefix_cache_mode(mode)
        if resolved != "radix":
            raise ValueError("RadixCache requires mode='radix'")
        self.block_size = int(block_size)
        self.mode = resolved
        self._root = _RadixNode()
        self._request_nodes: dict[int, tuple[_RadixNode, ...]] = {}
        self._hits = 0
        self._misses = 0
        self._partial_block_misses = 0

    @property
    def stats(self) -> PrefixCacheStats:
        return PrefixCacheStats(
            entries=self._entry_count(self._root),
            live_requests=len(self._request_nodes),
            hits=self._hits,
            misses=self._misses,
            partial_block_misses=self._partial_block_misses,
        )

    def insert(
        self,
        request_id: int,
        tokens: Iterable[int],
        block_ids: Sequence[int],
    ) -> PrefixCacheInsert:
        """Index one request's complete-block token prefixes."""

        rid = int(request_id)
        if rid in self._request_nodes:
            raise ValueError(f"request_id {rid} is already indexed")
        token_tuple = tuple(int(token) for token in tokens)
        block_tuple = tuple(int(block_id) for block_id in block_ids)
        if any(token < 0 for token in token_tuple):
            raise ValueError("tokens must be non-negative")
        if any(block_id < 0 for block_id in block_tuple):
            raise ValueError("block ids must be non-negative")
        complete_blocks = min(len(token_tuple) // self.block_size, len(block_tuple))
        if complete_blocks <= 0:
            self._request_nodes[rid] = ()
            return PrefixCacheInsert(request_id=rid, cached_tokens=0, cached_blocks=0)
        full_token_count = complete_blocks * self.block_size
        node = self._root
        owned_nodes: list[_RadixNode] = []
        for depth, token in enumerate(token_tuple[:full_token_count], start=1):
            node = node.children.setdefault(token, _RadixNode())
            if depth % self.block_size != 0:
                continue
            block_count = depth // self.block_size
            prefix_blocks = block_tuple[:block_count]
            if node.live and node.block_ids != prefix_blocks:
                raise ValueError("live prefix cache entry has conflicting block ids")
            if not node.owner_request_ids:
                node.eviction_state = "resident"
            node.block_ids = prefix_blocks
            node.owner_request_ids.add(rid)
            owned_nodes.append(node)
        self._request_nodes[rid] = tuple(owned_nodes)
        return PrefixCacheInsert(
            request_id=rid,
            cached_tokens=full_token_count,
            cached_blocks=complete_blocks,
        )

    def match(self, tokens: Iterable[int]) -> PrefixCacheMatch:
        """Return the deepest live block-aligned cached prefix."""

        token_tuple = tuple(int(token) for token in tokens)
        if any(token < 0 for token in token_tuple):
            raise ValueError("tokens must be non-negative")
        node = self._root
        best_depth = 0
        best_blocks: tuple[int, ...] = ()
        traversed = 0
        for token in token_tuple:
            child = node.children.get(token)
            if child is None:
                break
            node = child
            traversed += 1
            if node.live:
                best_depth = traversed
                best_blocks = node.block_ids
        if best_blocks:
            self._hits += 1
        else:
            self._misses += 1
            if traversed and traversed < self.block_size:
                self._partial_block_misses += 1
        return PrefixCacheMatch(
            matched_tokens=token_tuple[:best_depth],
            block_ids=best_blocks,
            remaining_tokens=token_tuple[best_depth:],
        )

    def cancel(self, request_id: int) -> PrefixCacheCancel:
        """Remove a request's ownership from all prefix entries."""

        rid = int(request_id)
        nodes = self._request_nodes.pop(rid, ())
        removed_blocks: list[int] = []
        removed_entries = 0
        for node in nodes:
            if rid not in node.owner_request_ids:
                continue
            node.owner_request_ids.remove(rid)
            removed_entries += 1
            if not node.owner_request_ids:
                removed_blocks.extend(node.block_ids)
                node.block_ids = ()
                node.eviction_state = "empty"
        return PrefixCacheCancel(
            request_id=rid,
            removed_entries=removed_entries,
            removed_blocks=tuple(removed_blocks),
        )

    def entry_state(self, tokens: Iterable[int]) -> PrefixCacheEntryState:
        """Return pointer-independent state for an exact live prefix entry."""

        token_tuple = tuple(int(token) for token in tokens)
        node = self._node_for_tokens(token_tuple)
        if node is None or not node.live:
            raise KeyError("no live prefix cache entry for tokens")
        return _entry_state_from_node(token_tuple, node)

    def mark_entry_eviction_state(self, tokens: Iterable[int], eviction_state: str) -> PrefixCacheEntryState:
        """Update tier/eviction state on a radix node without rewriting block ids."""

        state = str(eviction_state)
        if not state:
            raise ValueError("eviction_state must be a non-empty string")
        token_tuple = tuple(int(token) for token in tokens)
        node = self._node_for_tokens(token_tuple)
        if node is None or not node.live:
            raise KeyError("no live prefix cache entry for tokens")
        node.eviction_state = state
        return _entry_state_from_node(token_tuple, node)

    def entry_states(self) -> tuple[PrefixCacheEntryState, ...]:
        """Return all live prefix entries without exposing block pointers."""

        states: list[PrefixCacheEntryState] = []
        self._collect_entry_states(self._root, (), states)
        return tuple(states)

    def _node_for_tokens(self, tokens: tuple[int, ...]) -> _RadixNode | None:
        if any(token < 0 for token in tokens):
            raise ValueError("tokens must be non-negative")
        node = self._root
        for token in tokens:
            child = node.children.get(token)
            if child is None:
                return None
            node = child
        return node

    def _collect_entry_states(
        self,
        node: _RadixNode,
        prefix: tuple[int, ...],
        states: list[PrefixCacheEntryState],
    ) -> None:
        if node.live:
            states.append(_entry_state_from_node(prefix, node))
        for token in sorted(node.children):
            self._collect_entry_states(node.children[token], (*prefix, token), states)

    def _entry_count(self, node: _RadixNode) -> int:
        count = 1 if node.live else 0
        for child in node.children.values():
            count += self._entry_count(child)
        return count


def _entry_state_from_node(tokens: tuple[int, ...], node: _RadixNode) -> PrefixCacheEntryState:
    return PrefixCacheEntryState(
        matched_tokens=tokens,
        block_ids=node.block_ids,
        owner_request_ids=tuple(sorted(node.owner_request_ids)),
        eviction_state=node.eviction_state,
    )


__all__ = [
    "PREFIX_CACHE_CHOICES",
    "PrefixCacheCancel",
    "PrefixCacheEntryState",
    "PrefixCacheInsert",
    "PrefixCacheMatch",
    "PrefixCacheStats",
    "RadixCache",
    "resolve_prefix_cache_mode",
]
