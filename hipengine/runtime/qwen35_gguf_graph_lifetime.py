"""Explicit graph and KV-pin lifetime ownership for Qwen GGUF resident sessions.

A resident session historically exposed three bookkeeping attributes that every
capture site mutated in place: ``_decode_graphs`` (replay handles),
``_device_kv_graph_handles`` (handles that pin the current KV allocation) and
``_decode_graph_submission_contexts`` (transport contexts reused across
captures). The session keeps those names as compatibility properties, but this
module is the only code that pins, unpins, closes or enumerates the handles.

Every entry point receives its resources explicitly: graph handles, the current
``DeviceKVPoolAllocation`` and the ``DeviceChunkedKVPool``. The owner never
reaches back into a session, so one instance can serve a session whose KV
binding changes (bind, rebind, detach) without stale references, and a resident
slot view can own independent bookkeeping over the shared pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable


def _is_closed(graph: object) -> bool:
    return bool(getattr(graph, "closed", False))


def close_graph_handles(
    graphs: Iterable[object],
    *,
    require_close: bool = False,
    error: str = "a live decode graph exposes no close()",
) -> int:
    """Close each distinct live handle once, preserving caller order.

    ``require_close=False`` is the device-KV invalidation policy: a handle with
    no callable ``close`` is skipped rather than failing the whole teardown.
    ``require_close=True`` is the packed-workspace policy: a live graph without
    ``close`` means its buffers cannot be reclaimed, so it raises ``error``.
    """

    seen: set[int] = set()
    closed = 0
    for graph in graphs:
        key = id(graph)
        if key in seen or _is_closed(graph):
            continue
        seen.add(key)
        close = getattr(graph, "close", None)
        if not callable(close):
            if require_close:
                raise RuntimeError(error)
            continue
        close()
        closed += 1
    return closed


@dataclass
class Qwen35GGUFGraphLifetime:
    """One session's decode graphs, KV pins and submission contexts."""

    decode_graphs: list[object] = field(default_factory=list)
    kv_pins: dict[int, object] = field(default_factory=dict)
    submission_contexts: dict[str, object] = field(default_factory=dict)

    def pin(self, graph: object, *, allocation: object | None, pool: object | None) -> bool:
        """Pin the allocation's pages for ``graph``, once per graph identity."""

        if allocation is None or pool is None:
            return False
        key = id(graph)
        if key in self.kv_pins:
            return False
        pool.pin(allocation.block_ids)
        self.kv_pins[key] = graph
        return True

    def unpin(self, graph: object, *, allocation: object | None, pool: object | None) -> bool:
        """Release one pin, or raise if the binding disappeared while pinned."""

        key = id(graph)
        if key not in self.kv_pins:
            return False
        if allocation is None or pool is None:
            raise RuntimeError("GGUF device KV graph pin outlived its allocation")
        pool.unpin(allocation.block_ids)
        self.kv_pins.pop(key, None)
        return True

    @property
    def pinned_count(self) -> int:
        return len(self.kv_pins)

    def graphs(self) -> tuple[object, ...]:
        """Every registered handle, pinned handles first, deduped by identity."""

        seen: set[int] = set()
        graphs: list[object] = []
        for graph in (*self.kv_pins.values(), *self.decode_graphs):
            key = id(graph)
            if key in seen:
                continue
            seen.add(key)
            graphs.append(graph)
        return tuple(graphs)

    def live_graphs(self) -> tuple[object, ...]:
        return tuple(graph for graph in self.graphs() if not _is_closed(graph))

    def blocks_kv_rebind(self) -> bool:
        """True while any handle still addresses the current KV allocation.

        A retained pin counts even if the handle reports itself closed: the
        pool's pin count, not the handle flag, is what keeps the pages
        unassignable until ``unpin`` runs.
        """

        return bool(self.kv_pins) or any(
            not _is_closed(graph) for graph in self.decode_graphs
        )

    def close_live_graphs(self) -> int:
        """Close every live registered handle; skip handles without ``close``."""

        return close_graph_handles(self.graphs())

    def close_decode_graphs(self) -> int:
        """Close every decode graph unconditionally, then clear the registry.

        Unlike :meth:`close_live_graphs`, this is session teardown: a missing
        ``close`` is a programming error and propagates, and a failed close
        leaves the registry intact for retry.
        """

        closed = 0
        for graph in tuple(self.decode_graphs):
            graph.close()
            closed += 1
        self.decode_graphs.clear()
        return closed

    def submission_context(
        self,
        transport: str,
        factory: Callable[[], object | None],
    ) -> object | None:
        """Return the retained context for ``transport``, creating it once."""

        context = self.submission_contexts.get(transport)
        if context is None:
            context = factory()
            if context is not None:
                self.submission_contexts[transport] = context
        return context

    def close_submission_contexts(self) -> dict[str, dict[str, object]]:
        """Close idle retained contexts and return teardown proof per transport."""

        proofs: dict[str, dict[str, object]] = {}
        for name, context in tuple(self.submission_contexts.items()):
            provenance = getattr(context, "provenance", None)
            close = getattr(context, "close", None)
            if not callable(provenance) or not callable(close):
                raise RuntimeError(
                    f"decode submission context {name!r} has an invalid lifecycle ABI"
                )
            before = provenance()
            close()
            after = provenance()
            proofs[str(name)] = {"before": before, "after": after}
        self.submission_contexts.clear()
        return proofs


__all__ = ["Qwen35GGUFGraphLifetime", "close_graph_handles"]
