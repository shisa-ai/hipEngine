"""Cache-owned prefix-state snapshots for GGUF hybrid resident sessions.

One exact 256-token KV boundary needs the paired per-layer GDN Conv and
recurrent state copied out of the live session so a later request can resume
from it. The pooled arena that backs those copies and the snapshot that owns
them live here, independent of the runner, so the capture transaction can be
exercised on CPU.

Boundary admission, KV-block ownership, and dirty packed-state checks belong to
the resident session that calls :meth:`Qwen35GGUFPrefixStateSnapshot.capture`;
this module only owns the device-side copy transaction and its cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind, HipRuntime
from hipengine.core.memory import DeviceBuffer, DeviceMemoryArena, free

# Snapshot views are carved from one arena with its own alignment, so capacity
# planning rounds every per-layer state to this boundary instead of summing raw
# byte counts.
_ARENA_ALIGNMENT_BYTES = 4096
_MIN_ARENA_BYTES = 4096


class _GGUFPrefixSnapshotArenaPool:
    """Recycle whole snapshot arenas so a steady-state capture is memcpy-only.

    A capture used to ``malloc`` one buffer per GDN conv and recurrent state -
    ~54 allocations on a 27-GDN-layer model - plus a device sync, and paid that
    on every eligible turn whether or not the snapshot was ever reused. The
    state geometry is identical for every snapshot of a given model, so the
    arena from an evicted snapshot fits the next capture exactly. Retaining a
    few makes the common capture zero-allocation; the cost was never the
    copying.
    """

    def __init__(self, runtime: HipRuntime, *, max_retained: int = 4) -> None:
        self.runtime = runtime
        self.max_retained = max(0, int(max_retained))
        self._free: dict[int, list[DeviceMemoryArena]] = {}
        # Arenas whose queued copies have not provably retired. They are never
        # recycled - a later acquire would overwrite live device memory - and
        # never freed until their capture stream is known to be idle.
        self._quarantine: list[tuple[DeviceMemoryArena, int]] = []
        self.acquires = 0
        self.arena_allocations = 0
        self.closed = False

    def acquire(self, nbytes: int) -> DeviceMemoryArena:
        if self.closed:
            raise RuntimeError("prefix snapshot arena pool is closed")
        capacity = max(1, int(nbytes))
        self.acquires += 1
        bucket = self._free.get(capacity)
        if bucket:
            arena = bucket.pop()
            arena.rewind()
            return arena
        self.arena_allocations += 1
        return DeviceMemoryArena.create(capacity, runtime=self.runtime)

    def release(self, arena: DeviceMemoryArena) -> None:
        """Take an arena back, or free it when the retention budget is full."""

        if any(held is arena for held, _stream in self._quarantine):
            raise RuntimeError("cannot release a quarantined prefix snapshot arena")
        if self.closed or arena.closed:
            if not arena.closed:
                arena.close()
            return
        bucket = self._free.setdefault(int(arena.capacity_bytes), [])
        if len(bucket) >= self.max_retained:
            arena.close()
            return
        arena.rewind()
        bucket.append(arena)

    def quarantine(self, arena: DeviceMemoryArena, *, stream: int = 0) -> None:
        """Retain an arena whose queued copies may not have retired yet.

        A quarantined arena is never recycled and never freed by :meth:`release`;
        only :meth:`close` may reclaim it, and only after synchronizing its
        capture stream. That is what keeps a failed retirement sync from turning
        into a device use-after-free or a silent leak.
        """

        if arena.closed:
            return
        if any(held is arena for held, _stream in self._quarantine):
            return
        self._quarantine.append((arena, int(stream)))

    def close(self) -> None:
        # Quarantined arenas may still be read by queued copies, so every
        # capture stream must retire before any of them is freed. A failed
        # synchronization leaves the quarantine, the retained arenas, and
        # ``closed`` untouched, so a later close retries the whole teardown
        # instead of leaking or recycling an unsafe arena.
        self._retire_quarantine()
        if self.closed:
            return
        self.closed = True
        for bucket in self._free.values():
            for arena in bucket:
                arena.close()
        self._free.clear()

    def _retire_quarantine(self) -> None:
        if not self._quarantine:
            return
        streams: list[int] = []
        for _arena, stream in self._quarantine:
            if stream not in streams:
                streams.append(stream)
        for stream in streams:
            _synchronize(self.runtime, stream)
        for arena, _stream in list(self._quarantine):
            if not arena.closed:
                arena.close()
        self._quarantine.clear()

    def stats(self) -> dict[str, int]:
        return {
            "acquires": int(self.acquires),
            "arena_allocations": int(self.arena_allocations),
            "retained_arenas": int(sum(len(b) for b in self._free.values())),
            "quarantined_arenas": int(len(self._quarantine)),
        }


@dataclass
class Qwen35GGUFPrefixStateSnapshot:
    """Cache-owned device snapshot for one exact GGUF hybrid-state boundary."""

    runtime: HipRuntime
    runner: object
    kv_storage_dtype: DType
    kv_storage_layout: str
    position: int
    block_ids: tuple[int, ...]
    backing: object
    layer_conv_states: tuple[DeviceBuffer | None, ...]
    layer_recurrent_states: tuple[DeviceBuffer | None, ...]
    closed: bool = False
    # When the state came from one pooled arena, the views are non-owning and
    # the arena is returned to the pool on close instead of freed per buffer.
    arena: DeviceMemoryArena | None = None
    arena_pool: "_GGUFPrefixSnapshotArenaPool | None" = None

    @classmethod
    def capture(
        cls,
        *,
        runtime: HipRuntime,
        runner: object,
        kv_storage_dtype: DType,
        kv_storage_layout: str,
        position: int,
        block_ids: tuple[int, ...],
        backing: object,
        layer_conv_states: tuple[DeviceBuffer | None, ...],
        layer_recurrent_states: tuple[DeviceBuffer | None, ...],
        arena_pool: _GGUFPrefixSnapshotArenaPool,
        stream: int = 0,
    ) -> "Qwen35GGUFPrefixStateSnapshot":
        """Copy one boundary's paired linear state into a pooled arena.

        ``runner`` is stored only as snapshot identity: this call never looks up
        a session or runner method. The caller has already admitted the boundary
        and owns the KV blocks.

        Failure is fail-closed on device memory. Copies are issued asynchronously,
        so once a copy may have been submitted the arena cannot return to the pool
        until those copies retire: the cleanup path synchronizes first and
        releases second. If that cleanup synchronization itself fails, the arena
        is quarantined in ``arena_pool`` instead of released: it is never recycled
        and never freed until a later pool ``close()`` synchronizes the capture
        stream. The original capture failure is re-raised with a note recording
        that decision.
        """

        conv_states = tuple(layer_conv_states)
        recurrent_states = tuple(layer_recurrent_states)
        if len(conv_states) != len(recurrent_states):
            raise ValueError(
                "GGUF prefix snapshot requires one Conv and one GDN state per"
                f" layer: got {len(conv_states)} Conv and"
                f" {len(recurrent_states)} GDN states"
            )
        if arena_pool is None:
            raise ValueError("GGUF prefix snapshot requires an arena pool")
        arena_capacity = 0
        for conv_state, recurrent_state in zip(conv_states, recurrent_states):
            if conv_state is None or recurrent_state is None:
                continue
            for nbytes in (int(conv_state.nbytes), int(recurrent_state.nbytes)):
                arena_capacity += (
                    (nbytes + _ARENA_ALIGNMENT_BYTES - 1) // _ARENA_ALIGNMENT_BYTES
                ) * _ARENA_ALIGNMENT_BYTES
        arena = arena_pool.acquire(max(_MIN_ARENA_BYTES, arena_capacity))
        conv_backups: list[DeviceBuffer | None] = []
        recurrent_backups: list[DeviceBuffer | None] = []
        copies_submitted = False
        try:
            for layer_id, (conv_state, recurrent_state) in enumerate(
                zip(conv_states, recurrent_states)
            ):
                if (conv_state is None) != (recurrent_state is None):
                    raise ValueError(
                        f"GGUF prefix snapshot partial linear state at layer {layer_id}"
                    )
                if conv_state is None:
                    conv_backups.append(None)
                    recurrent_backups.append(None)
                    continue
                assert recurrent_state is not None
                conv_backup = arena.allocate(int(conv_state.nbytes))
                recurrent_backup = arena.allocate(int(recurrent_state.nbytes))
                conv_backups.append(conv_backup)
                recurrent_backups.append(recurrent_backup)
                # Set before the call: a memcpy that raises may already have
                # enqueued work, and the cleanup path must not skip its sync.
                copies_submitted = True
                runtime.memcpy_async(
                    conv_backup.ptr,
                    conv_state.ptr,
                    conv_state.nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    int(stream),
                )
                runtime.memcpy_async(
                    recurrent_backup.ptr,
                    recurrent_state.ptr,
                    recurrent_state.nbytes,
                    HipMemcpyKind.DEVICE_TO_DEVICE,
                    int(stream),
                )
            _synchronize(runtime, stream)
        except BaseException as failure:
            if copies_submitted:
                try:
                    _synchronize(runtime, stream)
                except BaseException as sync_failure:
                    arena_pool.quarantine(arena, stream=int(stream))
                    failure.add_note(
                        "GGUF prefix snapshot arena quarantined: copies may already"
                        " be queued and the cleanup synchronization failed, so the"
                        " pool retains the arena and frees it only after a later"
                        " close synchronizes the capture stream"
                    )
                    raise failure from sync_failure
            arena_pool.release(arena)
            raise
        return cls(
            runtime=runtime,
            runner=runner,
            kv_storage_dtype=kv_storage_dtype,
            kv_storage_layout=str(kv_storage_layout),
            position=int(position),
            block_ids=tuple(int(block_id) for block_id in block_ids),
            backing=backing,
            layer_conv_states=tuple(conv_backups),
            layer_recurrent_states=tuple(recurrent_backups),
            arena=arena,
            arena_pool=arena_pool,
        )

    @property
    def nbytes(self) -> int:
        return sum(
            int(buffer.nbytes)
            for buffer in (*self.layer_conv_states, *self.layer_recurrent_states)
            if buffer is not None
        )

    def close(self) -> None:
        if self.closed:
            return
        if self.arena is not None:
            # Arena-backed: the per-layer buffers are views into one owning
            # allocation, so freeing them individually would be a double free.
            if self.arena_pool is not None:
                self.arena_pool.release(self.arena)
            else:
                self.arena.close()
        else:
            for buffer in reversed(
                (*self.layer_conv_states, *self.layer_recurrent_states)
            ):
                if buffer is not None:
                    free(buffer, runtime=self.runtime)
        self.closed = True


def _synchronize(runtime: HipRuntime, stream: int) -> None:
    """Retire queued copies on the capture stream, or device-wide for stream 0."""

    if int(stream):
        runtime.stream_synchronize(int(stream))
    else:
        runtime.device_synchronize()
