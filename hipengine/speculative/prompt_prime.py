"""Request-owned prompt-priming chunk contract, split plan, and row ledger.

Prompt priming materializes a request's target hidden rows into a draft
provider's KV before the first speculative cycle. Every route that does it
repeats the same three host-side jobs: refuse a chunk that does not fit, cut an
incoming chunk at row limits and prefix-capture boundaries, and count the rows
that actually landed. Those jobs live here so the routes cannot drift apart.

This module is host-only. It makes no device calls and imports nothing from
``hipengine``, so any runtime or generation module may import it without a
cycle. The sink keeps every device call, its own stream, and its own borrowed
row lifetime.

A ledger is two-phase on purpose. :meth:`PromptPrimeLedger.check` decides
whether a chunk is admissible and returns a refusal code; a sink calls
:meth:`PromptPrimeLedger.publish` only after its device enqueues for that chunk
succeeded. A chunk that fails mid-enqueue therefore never advances the cursor,
and the sink's published row count stays a statement about device state rather
than about validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

__all__ = [
    "PROMPT_PRIME_REFUSAL_EMPTY_CHUNK",
    "PROMPT_PRIME_REFUSAL_FINISHED",
    "PROMPT_PRIME_REFUSAL_NOT_CONTIGUOUS",
    "PROMPT_PRIME_REFUSAL_OUTSIDE_PROMPT",
    "PromptPrimeChunk",
    "PromptPrimeLedger",
    "PromptPrimePlan",
    "PromptPrimeSink",
]

#: The sink already published its completion and accepts no further rows.
PROMPT_PRIME_REFUSAL_FINISHED = "finished"
#: A sequential sink was handed a chunk that does not start at its cursor.
PROMPT_PRIME_REFUSAL_NOT_CONTIGUOUS = "not_contiguous"
#: The chunk carries no rows.
PROMPT_PRIME_REFUSAL_EMPTY_CHUNK = "empty_chunk"
#: The chunk starts before the prompt or runs past its last row.
PROMPT_PRIME_REFUSAL_OUTSIDE_PROMPT = "outside_prompt"


@runtime_checkable
class PromptPrimeSink(Protocol):
    """Request-owned consumer for completed post-output-norm target chunks.

    The shared caller fans slot-major packed target rows out to one sink per
    request. A sink owns its buffers, its stream, and the lifetime of any
    borrowed row pointer; the shared layer only decides which rows belong to
    which request and in what order.
    """

    request_id: int
    hidden_size: int
    total_rows: int

    def consume(
        self,
        *,
        request_id: int,
        chunk_start: int,
        hidden_ptr: int,
        rows: int,
        stream: int,
    ) -> None: ...

    def finish(self, *, request_id: int, total_rows: int, stream: int) -> None: ...


@dataclass(frozen=True, slots=True)
class PromptPrimeChunk:
    """One contiguous slice of an incoming chunk, in the sink's coordinates.

    ``chunk_start`` is the slice's first row measured in the same space as the
    sink's other ``chunk_start`` values. ``row_offset`` is the slice's first row
    measured from the incoming chunk's first row, which is what a sink needs to
    offset a borrowed row pointer without re-deriving the split.
    """

    chunk_start: int
    rows: int
    row_offset: int


@dataclass(frozen=True, slots=True)
class PromptPrimePlan:
    """Chunk-shape policy for one request's prompt timeline.

    ``start_position`` places the timeline in absolute sequence coordinates so
    that capture boundaries, which are absolute, can cut a chunk that is
    addressed relatively. ``max_chunk_rows`` bounds how many rows one slice may
    carry; ``0`` means unbounded.
    """

    start_position: int = 0
    max_chunk_rows: int = 0

    def __post_init__(self) -> None:
        if int(self.start_position) < 0:
            raise ValueError("prompt-prime plan start_position must be non-negative")
        if int(self.max_chunk_rows) < 0:
            raise ValueError("prompt-prime plan max_chunk_rows must be non-negative")

    def split(
        self,
        *,
        chunk_start: int,
        rows: int,
        capture_boundaries: Sequence[int] = (),
    ) -> tuple[PromptPrimeChunk, ...]:
        """Return the slices of one incoming chunk, in consumption order.

        The row limit cuts from the incoming chunk's first row. A capture
        boundary cuts only where it falls strictly inside the chunk's absolute
        span, so a boundary on either edge cannot emit a zero-row slice. When a
        limit and a boundary coincide, the cut happens once.
        """

        start = int(chunk_start)
        count = int(rows)
        if count <= 0:
            return ()
        offsets = {0}
        limit = int(self.max_chunk_rows)
        if limit:
            offsets.update(range(limit, count, limit))
        if capture_boundaries:
            absolute_start = int(self.start_position) + start
            absolute_end = absolute_start + count
            for point in capture_boundaries:
                boundary = int(point)
                if absolute_start < boundary < absolute_end:
                    offsets.add(boundary - absolute_start)
        edges = sorted(offsets)
        edges.append(count)
        return tuple(
            PromptPrimeChunk(
                chunk_start=start + left,
                rows=right - left,
                row_offset=left,
            )
            for left, right in zip(edges, edges[1:])
        )


class PromptPrimeLedger:
    """Two-phase row accounting for one request-owned prompt-priming sink.

    ``contiguous=True`` models a sequential appender: every chunk must start at
    the published cursor, which is the shifted-hidden seam the dense NextN
    route relies on. ``contiguous=False`` models a position-addressed sink that
    writes each chunk at its own offset, so gaps and out-of-order chunks are
    legal and ``published_rows`` counts covered rows instead of a cursor.
    """

    __slots__ = ("_contiguous", "_covered", "_cursor", "_finished", "_request_id", "_total_rows")

    def __init__(self, *, request_id: int, total_rows: int, contiguous: bool = True) -> None:
        if int(request_id) < 0:
            raise ValueError("prompt-prime ledger request_id must be non-negative")
        if int(total_rows) < 0:
            raise ValueError("prompt-prime ledger total_rows must be non-negative")
        self._request_id = int(request_id)
        self._total_rows = int(total_rows)
        self._contiguous = bool(contiguous)
        self._cursor = 0
        # Merged half-open row intervals, used only by a position-addressed
        # ledger. Insertions are one per prefill chunk, so merging in place is
        # cheaper than any indexed structure.
        self._covered: list[tuple[int, int]] = []
        self._finished = False

    @property
    def request_id(self) -> int:
        return self._request_id

    @property
    def total_rows(self) -> int:
        return self._total_rows

    @property
    def contiguous(self) -> bool:
        return self._contiguous

    @property
    def published_rows(self) -> int:
        """Rows published so far: the cursor, or the covered row count."""

        if self._contiguous:
            return self._cursor
        return sum(right - left for left, right in self._covered)

    @property
    def complete(self) -> bool:
        """Whether every prompt row has been published."""

        return self.published_rows == self._total_rows

    @property
    def finished(self) -> bool:
        return self._finished

    def check(self, *, chunk_start: int, rows: int) -> str | None:
        """Return a refusal code for this chunk, or ``None`` when admissible."""

        start = int(chunk_start)
        count = int(rows)
        if self._finished:
            return PROMPT_PRIME_REFUSAL_FINISHED
        if self._contiguous and start != self._cursor:
            return PROMPT_PRIME_REFUSAL_NOT_CONTIGUOUS
        if count <= 0:
            return PROMPT_PRIME_REFUSAL_EMPTY_CHUNK
        if start < 0 or start + count > self._total_rows:
            return PROMPT_PRIME_REFUSAL_OUTSIDE_PROMPT
        return None

    def publish(self, *, chunk_start: int, rows: int) -> None:
        """Record a chunk whose device work already succeeded.

        Publishing a chunk that :meth:`check` would refuse is a caller bug, so
        it raises instead of silently corrupting the accounting.
        """

        refusal = self.check(chunk_start=chunk_start, rows=rows)
        if refusal is not None:
            raise ValueError(f"prompt-prime ledger cannot publish a refused chunk: {refusal}")
        start = int(chunk_start)
        count = int(rows)
        if self._contiguous:
            self._cursor += count
            return
        self._cover(start, start + count)

    def mark_finished(self) -> None:
        """Record that the sink accepted its completion."""

        self._finished = True

    def _cover(self, start: int, end: int) -> None:
        merged: list[tuple[int, int]] = []
        for left, right in self._covered:
            if right < start or left > end:
                merged.append((left, right))
                continue
            start = min(start, left)
            end = max(end, right)
        merged.append((start, end))
        merged.sort()
        self._covered = merged
