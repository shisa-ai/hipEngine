"""Prompt-prefix checkpoints for Flash-Next (Qwen4Exp) serving.

A checkpoint restores a request-owned c1 runner (and, for MTP rows, its draft)
to the exact state it had after prefilling ``depth`` prompt tokens. Two kinds of
state make that up:

* Append-only rows. Every QSA layer's K/V rows, raw index keys and pooled index
  blocks, plus the draft's single-layer equivalents. A row below ``depth`` never
  changes once written, so checkpoints share them: rows live in refcounted
  segments, and a later checkpoint of the same computation references the
  earlier segments and copies only the rows written since.
* Private state at one position. The GDN recurrent matrices, GDN and PLE conv
  histories, the residual and the PLE n-gram history describe exactly one
  position and cannot be rolled back, so each checkpoint owns a copy.

Segments are shared by provenance, not by token content: a segment is reused
only by checkpoints taken from runners that restored it, so a restored request
continues exactly as the run that wrote those rows would have. Two independent
cold prefills of the same text never share rows.

The cache holds device memory under one byte budget and evicts the least
recently used checkpoint first. A lookup that finds nothing usable is a cache
miss with a named reason; it never refuses the request.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer, free, malloc

# Shortest prefix worth restoring or capturing: one KV block.
MIN_PREFIX_TOKENS = 256
# A capture closer than this to the previous one on the same row adds little.
MIN_CAPTURE_SPACING_TOKENS = 512
# Long prompts also keep evenly spaced checkpoints so an edit deep in the
# history restores near the edit rather than from the start.
GRID_MIN_PROMPT_TOKENS = 8192
GRID_POINTS = 3
DEFAULT_MAX_ENTRIES = 64
DEFAULT_LIMIT_CAP_BYTES = 32 << 30
_LIMIT_ENV = "HIPENGINE_GGUF_PREFIX_RETAINED_STATE_BYTES"


class PrefixCacheUnsupported(RuntimeError):
    """The runner layout cannot be checkpointed (a named capability miss)."""


@dataclass(frozen=True)
class _Plane:
    """One append-only runner buffer: ``granularity`` tokens per unit."""

    base: int
    unit_bytes: int
    granularity: int
    capacity_units: int

    def units(self, start: int, end: int) -> tuple[int, int]:
        return start // self.granularity, end // self.granularity


def _identity_blocks(attention: Any) -> None:
    blocks = np.asarray(attention.block_host)
    if blocks.ndim != 1 or not np.array_equal(blocks, np.arange(blocks.size)):
        raise PrefixCacheUnsupported(
            "prefix checkpoints need a contiguous c1 block table"
        )


def _attention_planes(attention: Any, index: Any) -> list[_Plane]:
    _identity_blocks(attention)
    positions = int(attention.max_positions)
    blocks = int(np.asarray(attention.block_host).size)
    rows = blocks * int(attention.block_size)
    row_bytes = int(attention.key_cache.nbytes) // rows
    ratio = int(index.compression_ratio)
    capacity = int(index.capacity)
    pooled_blocks = capacity // ratio
    if capacity < positions or pooled_blocks <= 0:
        raise PrefixCacheUnsupported("QSA index capacity is below the KV capacity")
    return [
        _Plane(int(attention.key_cache.ptr), row_bytes, 1, rows),
        _Plane(int(attention.value_cache.ptr), row_bytes, 1, rows),
        _Plane(int(index.raw_keys.ptr), int(index.raw_keys.nbytes) // capacity, 1, capacity),
        _Plane(
            int(index.pooled_keys.ptr),
            int(index.pooled_keys.nbytes) // pooled_blocks,
            ratio,
            pooled_blocks,
        ),
    ]


def target_planes(runner: Any) -> tuple[_Plane, ...]:
    try:
        pairs = tuple(zip(runner.attention_states, runner.index_states, strict=True))
        runner.state.owned_buffers
    except (AttributeError, ValueError) as error:
        raise PrefixCacheUnsupported(f"runner exposes no checkpointable state: {error}") from error
    if not pairs:
        raise PrefixCacheUnsupported("runner has no QSA attention state")
    planes: list[_Plane] = []
    for attention, index in pairs:
        planes.extend(_attention_planes(attention, index))
    return tuple(planes)


def draft_planes(draft: Any) -> tuple[_Plane, ...]:
    try:
        attention, index = draft.attention, draft.index
    except AttributeError as error:
        raise PrefixCacheUnsupported(f"draft exposes no checkpointable state: {error}") from error
    return tuple(_attention_planes(attention, index))


def _layout(planes: Sequence[_Plane]) -> tuple[tuple[int, int], ...]:
    return tuple((plane.unit_bytes, plane.granularity) for plane in planes)


class _Segment:
    """Rows ``[start, end)`` of every plane, in one device allocation."""

    def __init__(
        self,
        runtime: Any,
        start: int,
        end: int,
        target: Sequence[_Plane],
        draft: Sequence[_Plane] | None,
    ) -> None:
        self.runtime = runtime
        self.start = int(start)
        self.end = int(end)
        self.has_draft = draft is not None
        self._offsets: list[int] = []
        total = 0
        for plane in (*target, *(draft or ())):
            first, last = plane.units(self.start, self.end)
            self._offsets.append(total)
            total += (last - first) * plane.unit_bytes
        self.nbytes = total
        self.refs = 0
        self.buffer: DeviceBuffer | None = (
            malloc(total, runtime=runtime) if total > 0 else None
        )

    def _pairs(self, target, draft):
        planes = (*target, *(draft if self.has_draft and draft is not None else ()))
        return zip(planes, self._offsets[: len(planes)], strict=True)

    def copy_from(self, target, draft) -> None:
        if self.buffer is None:
            return
        for plane, offset in self._pairs(target, draft):
            first, last = plane.units(self.start, self.end)
            if last > first:
                self.runtime.memcpy(
                    self.buffer.ptr + offset,
                    plane.base + first * plane.unit_bytes,
                    (last - first) * plane.unit_bytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )

    def copy_to(self, target, draft) -> int:
        if self.buffer is None:
            return 0
        copied = 0
        for plane, offset in self._pairs(target, draft):
            first, last = plane.units(self.start, self.end)
            if last > first:
                size = (last - first) * plane.unit_bytes
                self.runtime.memcpy(
                    plane.base + first * plane.unit_bytes,
                    self.buffer.ptr + offset,
                    size,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )
                copied += size
        return copied

    def close(self) -> None:
        if self.buffer is not None:
            free(self.buffer, runtime=self.runtime)
            self.buffer = None


@dataclass
class PrefixChain:
    """Segments a live row's runner provably holds, covering ``[0, depth)``."""

    segments: tuple[_Segment, ...] = ()
    depth: int = 0

    @property
    def has_draft(self) -> bool:
        return all(segment.has_draft for segment in self.segments)


@dataclass(eq=False)
class PrefixSnapshot:
    """One allocation owner; eviction compares identity, not token arrays."""
    tokens: np.ndarray
    depth: int
    segments: tuple[_Segment, ...]
    recurrent: DeviceBuffer
    ple_hash_states: Mapping[int, Any]
    previous_hidden: np.ndarray | None
    kind: str
    last_used: int = 0

    @property
    def has_draft(self) -> bool:
        return self.previous_hidden is not None and all(
            segment.has_draft for segment in self.segments
        )


@dataclass(frozen=True)
class PrefixLookup:
    snapshot: PrefixSnapshot | None
    matched_tokens: int
    reason: str | None


def default_limit_bytes(runtime: Any) -> int:
    """Half of free device memory, capped at 32 GiB, unless set explicitly."""

    raw = os.environ.get(_LIMIT_ENV)
    if raw not in (None, ""):
        value = int(raw)
        if value < 0:
            raise ValueError(f"{_LIMIT_ENV} must be non-negative")
        return value
    free_bytes, _ = runtime.mem_get_info()
    return int(min(DEFAULT_LIMIT_CAP_BYTES, int(free_bytes) // 2))


def capture_targets(prompt_tokens: int, chunk: int) -> tuple[int, ...]:
    """Depths a fresh request tries to checkpoint before its prompt end.

    The last chunk boundary before the end lets an identical prompt (a retry or
    regenerate) prefill at most one chunk. Long prompts add evenly spaced grid
    points. Each target is captured at the first completed prefill boundary at
    or past it.
    """

    length = int(prompt_tokens)
    targets: set[int] = set()
    tail = length - int(chunk)
    if tail >= MIN_PREFIX_TOKENS:
        targets.add(tail)
    if length >= GRID_MIN_PROMPT_TOKENS:
        for point in range(1, GRID_POINTS + 1):
            targets.add(length * point // (GRID_POINTS + 1))
    return tuple(sorted(targets))


class Qwen4ExpPrefixCache:
    """Device-resident prefix checkpoints with one LRU byte budget."""

    def __init__(
        self,
        runtime: Any,
        *,
        limit_bytes: int | None = None,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self.runtime = runtime
        self._limit_bytes = None if limit_bytes is None else int(limit_bytes)
        self.max_entries = int(max_entries)
        self._snapshots: list[PrefixSnapshot] = []
        self._segments: set[_Segment] = set()
        self._clock = 0
        self._layout: tuple[Any, ...] | None = None
        self.counters: dict[str, int] = {
            "lookups": 0, "hits": 0, "reused_tokens": 0, "captures": 0,
            "capture_bytes": 0, "restore_bytes": 0, "evictions": 0,
        }
        self.fallback_reasons: dict[str, int] = {}
        self.capture_skips: dict[str, int] = {}
        self.closed = False

    # -- accounting -------------------------------------------------------

    @property
    def limit_bytes(self) -> int:
        if self._limit_bytes is None:
            self._limit_bytes = default_limit_bytes(self.runtime)
        return self._limit_bytes

    @property
    def resident_bytes(self) -> int:
        return sum(segment.nbytes for segment in self._segments) + sum(
            snapshot.recurrent.nbytes for snapshot in self._snapshots
        )

    @property
    def entries(self) -> int:
        return len(self._snapshots)

    def stats(self) -> dict[str, Any]:
        return {
            **self.counters,
            "snapshot_entries": len(self._snapshots),
            "snapshot_limit": self.max_entries,
            "resident_bytes": self.resident_bytes,
            "resident_limit_bytes": int(self._limit_bytes or 0),
            "segment_count": len(self._segments),
            "segment_bytes": sum(segment.nbytes for segment in self._segments),
            "fallback_reasons": dict(self.fallback_reasons),
            "capture_skips": dict(self.capture_skips),
        }

    def note_fallback(self, reason: str) -> None:
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1

    def _skip(self, reason: str) -> None:
        self.capture_skips[reason] = self.capture_skips.get(reason, 0) + 1

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    # -- lookup -----------------------------------------------------------

    def lookup(self, prompt_ids: Sequence[int], *, need_draft: bool) -> PrefixLookup:
        """Deepest checkpoint that is a strict prefix of ``prompt_ids``.

        ``matched_tokens`` is the longest common prefix with any checkpoint,
        usable or not; a caller can capture there for the next branch.
        """

        self.counters["lookups"] += 1
        prompt = np.asarray(prompt_ids, dtype=np.int64)
        length = int(prompt.size)
        best: PrefixSnapshot | None = None
        matched = 0
        saw_depth_only = False
        for snapshot in self._snapshots:
            depth = snapshot.depth
            span = min(depth, length)
            equal = prompt[:span] == snapshot.tokens[:span]
            common = span if bool(equal.all()) else int(np.argmin(equal))
            matched = max(matched, common)
            if common != depth or depth >= length:
                continue
            if need_draft and not snapshot.has_draft:
                saw_depth_only = True
                continue
            if best is None or (depth, snapshot.last_used) > (best.depth, best.last_used):
                best = snapshot
        if best is None:
            reason = "draft_state_not_cached" if saw_depth_only else "no_cached_prefix"
            return PrefixLookup(None, matched, reason)
        best.last_used = self._tick()
        return PrefixLookup(best, matched, None)

    # -- restore ----------------------------------------------------------

    def restore(self, snapshot: PrefixSnapshot, runner: Any, draft: Any | None) -> tuple[PrefixChain, int]:
        """Write ``snapshot`` into a freshly reset runner (and draft)."""

        target = target_planes(runner)
        drafts = None if draft is None else draft_planes(draft)
        self._check_layout(runner, target, drafts)
        if drafts is not None and not snapshot.has_draft:
            raise ValueError("checkpoint has no draft state")
        copied = 0
        for segment in snapshot.segments:
            copied += segment.copy_to(target, drafts)
        offset = 0
        for buffer in runner.state.owned_buffers.values():
            self.runtime.memcpy(
                buffer.ptr, snapshot.recurrent.ptr + offset, buffer.nbytes,
                HipMemcpyKind.DEVICE_TO_DEVICE,
            )
            offset += buffer.nbytes
        copied += offset
        runner.adopt_prefix_cursor(snapshot.depth, snapshot.ple_hash_states)
        if draft is not None:
            draft.adopt_prefix_position(snapshot.depth)
        chain = self._chain(snapshot.segments, snapshot.depth)
        self.counters["hits"] += 1
        self.counters["reused_tokens"] += snapshot.depth
        self.counters["restore_bytes"] += copied
        return chain, copied

    # -- capture ----------------------------------------------------------

    def capture(
        self,
        tokens: Sequence[int],
        runner: Any,
        draft: Any | None,
        *,
        chain: PrefixChain,
        previous_hidden: np.ndarray | None,
        kind: str,
    ) -> tuple[PrefixChain, PrefixSnapshot | None, str | None]:
        """Checkpoint ``runner`` at ``len(tokens)``; extend ``chain``.

        Returns the row's new chain, the snapshot (or ``None``) and the reason
        a capture was skipped. A skip never affects the request.
        """

        depth = len(tokens)
        if int(runner.position) != depth or chain.depth > depth:
            self._skip("cursor_mismatch")
            return chain, None, "cursor_mismatch"
        if draft is not None and (
            int(draft.position) != depth or previous_hidden is None or not chain.has_draft
        ):
            draft = None
        token_array = np.asarray(tokens, dtype=np.int64)
        for snapshot in self._snapshots:
            if snapshot.depth == depth and np.array_equal(snapshot.tokens, token_array) and (
                draft is None or snapshot.has_draft
            ):
                snapshot.last_used = self._tick()
                self._skip("duplicate")
                return chain, None, "duplicate"
        target = target_planes(runner)
        drafts = None if draft is None else draft_planes(draft)
        self._check_layout(runner, target, drafts)
        recurrent_bytes = sum(
            buffer.nbytes for buffer in runner.state.owned_buffers.values()
        )
        new_segment_bytes = 0
        if depth > chain.depth:
            for plane in (*target, *(drafts or ())):
                first, last = plane.units(chain.depth, depth)
                new_segment_bytes += (last - first) * plane.unit_bytes
        if not self._make_room(new_segment_bytes + recurrent_bytes):
            self._skip("budget")
            return chain, None, "budget"
        segment = None
        recurrent = None
        try:
            if depth > chain.depth:
                segment = _Segment(self.runtime, chain.depth, depth, target, drafts)
                segment.copy_from(target, drafts)
            recurrent = malloc(recurrent_bytes, runtime=self.runtime)
            offset = 0
            for buffer in runner.state.owned_buffers.values():
                self.runtime.memcpy(
                    recurrent.ptr + offset, buffer.ptr, buffer.nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                )
                offset += buffer.nbytes
        except Exception:
            if segment is not None:
                segment.close()
            if recurrent is not None:
                free(recurrent, runtime=self.runtime)
            self._skip("device_alloc_failed")
            return chain, None, "device_alloc_failed"
        segments = chain.segments + (() if segment is None else (segment,))
        if segment is not None:
            self._segments.add(segment)
        new_chain = self._chain(segments, depth)
        self.release_chain(chain)
        snapshot = PrefixSnapshot(
            tokens=token_array.copy(),
            depth=depth,
            segments=segments,
            recurrent=recurrent,
            ple_hash_states=MappingProxyType(dict(runner.ple_hash_states)),
            previous_hidden=(
                None if draft is None
                else np.array(previous_hidden, dtype=np.float32, copy=True)
            ),
            kind=str(kind),
            last_used=self._tick(),
        )
        for item in segments:
            item.refs += 1
        self._snapshots.append(snapshot)
        self.counters["captures"] += 1
        self.counters["capture_bytes"] += (0 if segment is None else segment.nbytes) + recurrent_bytes
        return new_chain, snapshot, None

    # -- ownership --------------------------------------------------------

    def _chain(self, segments: tuple[_Segment, ...], depth: int) -> PrefixChain:
        for segment in segments:
            segment.refs += 1
        return PrefixChain(segments, int(depth))

    def release_chain(self, chain: PrefixChain) -> None:
        for segment in chain.segments:
            self._unref(segment)
        chain.segments = ()
        chain.depth = 0

    def _unref(self, segment: _Segment) -> None:
        segment.refs -= 1
        if segment.refs <= 0 and segment in self._segments:
            self._segments.discard(segment)
            segment.close()

    def _evict(self, snapshot: PrefixSnapshot) -> None:
        self._snapshots.remove(snapshot)
        free(snapshot.recurrent, runtime=self.runtime)
        for segment in snapshot.segments:
            self._unref(segment)
        self.counters["evictions"] += 1

    def _make_room(self, nbytes: int) -> bool:
        # Evicting a checkpoint whose rows a live row still holds frees only its
        # private state; those rows stay until the row releases its chain.
        limit = self.limit_bytes
        held_by_snapshots: dict[_Segment, int] = {}
        for snapshot in self._snapshots:
            for segment in snapshot.segments:
                held_by_snapshots[segment] = held_by_snapshots.get(segment, 0) + 1
        pinned = sum(
            segment.nbytes for segment in self._segments
            if segment.refs > held_by_snapshots.get(segment, 0)
        )
        if pinned + nbytes > limit:
            # Evicting every checkpoint would still not make room; keep them.
            return False
        while self._snapshots and (
            self.resident_bytes + nbytes > limit or len(self._snapshots) >= self.max_entries
        ):
            self._evict(min(self._snapshots, key=lambda item: item.last_used))
        return self.resident_bytes + nbytes <= limit and len(self._snapshots) < self.max_entries

    def _check_layout(self, runner, target, drafts) -> None:
        layout = (
            _layout(target),
            tuple(buffer.nbytes for buffer in runner.state.owned_buffers.values()),
            None if drafts is None else _layout(drafts),
        )
        if self._layout is None:
            self._layout = layout[:2]
            self._draft_layout = layout[2]
            return
        if layout[:2] != self._layout:
            raise PrefixCacheUnsupported("runner layout differs from cached checkpoints")
        if layout[2] is not None:
            if getattr(self, "_draft_layout", None) is None:
                self._draft_layout = layout[2]
            elif layout[2] != self._draft_layout:
                raise PrefixCacheUnsupported("draft layout differs from cached checkpoints")

    def clear(self) -> None:
        for snapshot in tuple(self._snapshots):
            self._evict(snapshot)

    def close(self) -> None:
        if self.closed:
            return
        self.clear()
        for segment in tuple(self._segments):
            segment.close()
        self._segments.clear()
        self.closed = True


__all__ = [
    "MIN_CAPTURE_SPACING_TOKENS",
    "MIN_PREFIX_TOKENS",
    "PrefixCacheUnsupported",
    "PrefixChain",
    "PrefixLookup",
    "PrefixSnapshot",
    "Qwen4ExpPrefixCache",
    "capture_targets",
    "default_limit_bytes",
    "draft_planes",
    "target_planes",
]
