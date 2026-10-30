"""Explicit graph and KV-pin lifetime ownership for Qwen GGUF resident sessions.

A resident session historically exposed three bookkeeping attributes that every
capture site mutated in place: ``_decode_graphs`` (replay handles),
``_device_kv_graph_handles`` (handles that pin the current KV allocation) and
``_decode_graph_submission_contexts`` (transport contexts reused across
captures). This module is the only code that registers, pins, unpins, closes or
enumerates those resources.

Every capture, close, slot-view and invalidation site now resolves this owner
through ``graph_lifetime_of`` and uses ``register``/``unregister`` and the
enumeration methods. The session no longer exposes mutable graph, pin or
submission-context containers.

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


def graph_lifetime_of(host: object) -> "Qwen35GGUFGraphLifetime":
    """Resolve the explicit owner attached to a session or packed owner.

    Capture, close and enumeration sites receive a session/owner; this is their
    only coupling to its graph storage. A resident session resolves through its
    ``_graph_lifetime()`` accessor. Tests may attach a real owner directly so
    they exercise the same protocol instead of a compatibility container.
    """

    owner = getattr(host, "_graph_lifetime", None)
    if owner is None:
        raise TypeError(
            "GGUF graph capture/close requires an explicit graph lifetime owner"
        )
    return owner() if callable(owner) else owner


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

    _decode_graphs: list[object] = field(default_factory=list)
    _kv_pins: dict[int, object] = field(default_factory=dict)
    _submission_contexts: dict[str, object] = field(default_factory=dict)

    def register(self, graph: object) -> None:
        """Register one decode-graph handle exactly once, in capture order."""

        if any(existing is graph for existing in self._decode_graphs):
            return
        self._decode_graphs.append(graph)

    def unregister(self, graph: object) -> bool:
        """Remove one registered decode-graph handle; idempotent."""

        for index, existing in enumerate(self._decode_graphs):
            if existing is graph:
                del self._decode_graphs[index]
                return True
        return False

    def decode_graphs(self) -> tuple[object, ...]:
        """Registered decode graphs in capture order."""

        return tuple(self._decode_graphs)

    def pinned_graphs(self) -> tuple[object, ...]:
        """Handles that currently hold a device-KV pin, in pin order."""

        return tuple(self._kv_pins.values())

    def pin(self, graph: object, *, allocation: object | None, pool: object | None) -> bool:
        """Pin the allocation's pages for ``graph``, once per graph identity."""

        if allocation is None or pool is None:
            return False
        key = id(graph)
        if key in self._kv_pins:
            return False
        pool.pin(allocation.block_ids)
        self._kv_pins[key] = graph
        return True

    def unpin(self, graph: object, *, allocation: object | None, pool: object | None) -> bool:
        """Release one pin, or raise if the binding disappeared while pinned."""

        key = id(graph)
        if key not in self._kv_pins:
            return False
        if allocation is None or pool is None:
            raise RuntimeError("GGUF device KV graph pin outlived its allocation")
        pool.unpin(allocation.block_ids)
        self._kv_pins.pop(key, None)
        return True

    @property
    def pinned_count(self) -> int:
        return len(self._kv_pins)

    def graphs(self) -> tuple[object, ...]:
        """Every registered handle, pinned handles first, deduped by identity."""

        seen: set[int] = set()
        graphs: list[object] = []
        for graph in (*self._kv_pins.values(), *self._decode_graphs):
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

        return bool(self._kv_pins) or any(
            not _is_closed(graph) for graph in self._decode_graphs
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
        for graph in tuple(self._decode_graphs):
            graph.close()
            closed += 1
        self._decode_graphs.clear()
        return closed

    def submission_context(
        self,
        transport: str,
        factory: Callable[[], object | None],
    ) -> object | None:
        """Return the retained context for ``transport``, creating it once."""

        context = self._submission_contexts.get(transport)
        if context is None:
            context = factory()
            if context is not None:
                self._submission_contexts[transport] = context
        return context

    def register_submission_context(
        self,
        transport: str,
        context: object | None,
    ) -> object | None:
        """Retain a caller-provided context for ``transport``.

        This is the explicit replacement for direct writes to the retired
        ``_decode_graph_submission_contexts`` compatibility container.
        """

        self._submission_contexts[str(transport)] = context
        return context

    def submission_context_names(self) -> tuple[str, ...]:
        return tuple(self._submission_contexts)

    def close_submission_contexts(self) -> dict[str, dict[str, object]]:
        """Close idle retained contexts and return teardown proof per transport."""

        proofs: dict[str, dict[str, object]] = {}
        for name, context in tuple(self._submission_contexts.items()):
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
        self._submission_contexts.clear()
        return proofs


__all__ = [
    "Qwen35GGUFGraphLifetime",
    "close_graph_handles",
    "graph_lifetime_of",
]
