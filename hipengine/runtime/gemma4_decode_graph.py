"""Capture one Gemma 4 decode step as a HIP graph and replay it per token.

The decode step launches roughly 780 kernels whose arguments never change
between positions: embeddings, norms, projections, rotary, MLP/MoE and the
lm_head all read the same buffers every step. Only two things move:

* the KV append destinations (the cache slot of the live position), and
* the position-dependent *content* — token ids, rope tables, the keep-mask —
  which is already staged into pointer-stable buffers before any launch.

A capture freezes everything, so this module freezes the launch shape to a
context bucket and re-targets the two things that must move:

* **Bucket keying.** A step at position ``p`` runs with ``keys_extent`` and
  ``kv_write_offset`` frozen to the bucket end (``ceil((p+1)/width)*width``)
  and ``key_begin`` frozen to ``bucket_start - window``. Both are constants
  for every position in the bucket, so the captured graph never changes
  shape; a bucket crossing recaptures. The keep-mask staged for the step is
  built with the frozen extent but the step's *actual* positions as content,
  so columns past the live range read False and the bucket-shaped attention
  sees exactly the live context. Frozen values are always ≤ their actual
  counterparts, so the graph attends a superset of the true range and the
  mask — not a host argument — is what bounds it.

* **Memcpy node updates.** The KV append is a device-to-device copy whose
  destination is the live position, which the host knows and the capture
  cannot. Each replay re-targets the captured append nodes with
  ``hipGraphExecMemcpyNodeSetParams1D``; every other parameter is verified
  at capture (source buffer, byte count, capture-time destination) so a node
  is only ever updated by the layer and cache it belongs to.

* **Tail zeroing.** The bucket shape reads cache slots ``[live, end)`` that
  no step has written yet. The mask keeps them out of the arithmetic, but
  ``q·k`` on uninitialized memory can be a NaN, and a NaN poisons the
  reduction regardless of the mask that would have zeroed a finite value.
  (Measured: a fresh cache held 1,644 NaN-pattern elements past live, and
  the first bucket-crossing replay returned NaN logits for every element.)
  So each capture zeroes the not-yet-written tail of every layer's key and
  value cache for ``[position + 1, end)`` once, on the capture stream,
  before the first replay — after that, every slot the graph can read is
  either written by an append or exactly zero, and a zeroed masked lane is
  bit-identical to the live-shaped launch.

The capture runs on its own stream. Position-dependent content is staged on
that stream *before* ``hipStreamBeginCapture``, so it executes immediately
and is not part of the graph, and each replay stages the next step's content
on the same stream ahead of the graph launch. The logits copy stays host-side
after the graph — the sampler reads it, so that wait is inherent (D8 row) —
and the runner's position bookkeeping happens in the collect phase, exactly
as on the normal path.

v1 scope: single-step capture with host-side sampling, Global capture mode
(the layer's parallel MoE stream and its event dependencies are recorded
into the graph, so replay keeps the launched path's two-stream overlap).
Not wired into `Gemma4Runner.forward`: the default-on attempt measured
**negative** at the campaign workload — 64.43 tok/s launched vs 61.21 with
the route at 1024/128 (−5.0%, three design variants, artifacts
`benchmarks/results/2026-09-30-gemma4-x7-graph-{pre,post,post2,post3}.json`)
— because launched decode is already device-bound: its 10.5 ms/step of host
submission is hidden under device work, while the graph adds serial host
(the per-step append retarget ≈ 0.17 ms, `hipGraphLaunch` ≈ 0.27 ms) plus
~10.5 ms of capture recording per bucket crossing. The clearing route for a
future attempt is device-positioned appends — the qwen
`record_i64_scalar_indexed` pattern — which removes the retarget entirely
(phase probe: `scratch/x7_graph_phase_probe.py`).

The runner must be warm before the first capture: run at least one normal
forward (prefill) and one normal decode step first, so every kernel the graph
records is already registered and loaded. A cold route would try to load
modules while capture is active, which HIP rejects.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import numpy as np

from hipengine.core.hip import (
    HIP_GRAPH_NODE_TYPE_MEMCPY,
    get_hip_runtime,
)
from hipengine.core.memory import MemcpyKind
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_tiled import KEY_TILE
from hipengine.runtime.gemma4 import Gemma4Runner

#: Positions per capture window. Small enough that a bucket crossing costs a
#: recapture only once per ``width`` steps, large enough that the frozen
#: ``key_begin`` walks at most ``width`` keys past the true sliding window.
BUCKET_WIDTH = 64


@dataclass(frozen=True)
class _Bucket:
    """One capture window: positions ``[start, end)`` in the sequence."""

    start: int
    end: int

    @staticmethod
    def for_position(position: int, width: int = BUCKET_WIDTH) -> "_Bucket":
        """The bucket holding ``position``, cut so the route stays constant.

        Three rules, all of which keep every launch argument inside the bucket
        equal to what the live-shaped path would have used:

        1. The bucket covers the position and is ``width`` positions wide on
           the normal grid.
        2. The global layers' tiled route admits exactly at key counts that
           are multiples of ``KEY_TILE`` (128), so a bucket whose key-count
           range contains such a multiple would replay one route while the
           launched path used another at some step. A grid bucket crossing
           that point stops one position short (``end = multiple - 1``).
        3. The position exactly at ``multiple - 1`` becomes a singleton
           bucket ``[multiple - 1, multiple)``: its key count *is* the
           multiple, so graph and launched path agree on the tiled route with
           identical parameters.
        """

        live = int(position) + 1
        end = ((live + width - 1) // width) * width
        start = end - width
        boundary = (start // KEY_TILE + 1) * KEY_TILE  # smallest multiple > start
        if int(position) == boundary - 1:
            return _Bucket(start=int(position), end=int(position) + 1)
        if boundary <= end:
            end = boundary - 1
        return _Bucket(start=start, end=end)


@dataclass(frozen=True)
class _CaptureAppend:
    """One captured KV append node, with everything a re-target needs."""

    node: int
    src: int
    count: int
    cache_base: int
    kv_width: int


class Gemma4DecodeGraphSession:
    """Replay a captured decode step for one runner.

    Usage: warm the runner (one forward, one decode step), then call
    :meth:`step` per token. The session owns a HIP stream and one executable
    graph, recapturing whenever the bucket or any baked pointer changes.
    ``close()`` releases them; the runner is untouched.
    """

    def __init__(self, runner: Gemma4Runner) -> None:
        # Weak: the runner owns this session (``_decode_graph``), so a strong
        # back-reference would make the pair a cycle whose bulk is the
        # runner's weight map -- unreachable at refcount-zero until a cyclic
        # gc that allocation pressure may never trigger. See ``_runner``.
        self._runner_ref = weakref.ref(runner)
        self._stream = get_hip_runtime().stream_create(nonblocking=True)
        self._graph = 0
        self._exec = 0
        self._key: tuple[object, ...] | None = None
        self._bucket: _Bucket | None = None
        self._appends: tuple[_CaptureAppend, ...] = ()
        self._captures = 0

    # --- public API -------------------------------------------------------

    @property
    def _runner(self) -> Gemma4Runner:
        """The runner this session drives, held weakly.

        The runner owns the session, so a strong back-reference would make the
        pair a reference cycle; a cycle whose bulk is a 17 GB weight map never
        hits refcount-zero at request end and waits for a cyclic gc the
        process may never trigger. Weak keeps runner teardown prompt: when the
        runner dies, the session dies with it.
        """

        runner = self._runner_ref()
        if runner is None:
            raise RuntimeError("the decode-graph session's runner has been freed")
        return runner

    @property
    def stream(self) -> int:
        return self._stream

    @property
    def captures(self) -> int:
        """How many times this session has captured a graph."""

        return self._captures

    def step(self, token: int, *, apply_softcap: bool = True) -> np.ndarray:
        """Run one decode step for ``token`` and return its logits.

        Mirrors ``runner.forward([token])``: same staged content, same
        launches, same collect phase — the launches come from the captured
        graph instead of 780 host-side submissions.
        """

        runner = self._runner
        if runner.uses_int8_kv:
            # The direct INT8 consumer validates its live counts, row positions
            # and page table with a synchronous device-to-host read, so it
            # cannot run inside a HIP graph capture. Refusing capture up front
            # names the limitation instead of silently disabling the storage
            # the caller asked for; the requested INT8 cache is still exactly
            # what runs without capture. Nothing is staged before the refusal.
            raise RuntimeError(
                "decode-graph capture is not supported with int8_per_token_head KV "
                "storage: the direct INT8 consumer performs a checked device-to-host "
                "readback that is not graph-capturable. Run without capture, or "
                "request bf16 KV storage."
            )
        position = runner.position
        if int(token) < 0:
            raise ValueError(f"token {token} must be non-negative")
        vocab = int(runner.weights.config.vocab_size or 0)
        if vocab <= 0:
            raise ValueError("config carries no vocab_size, so logits cannot be sized")
        if not int(token) < vocab:
            raise ValueError(f"token id {token} is outside the vocabulary of {vocab}")
        if position + 1 > runner.capacity:
            raise ValueError(f"one token from position {position} exceeds capacity {runner.capacity}")

        bucket = _Bucket.for_position(position)
        tables, masks = runner._stage_block_content(
            [int(token)], stream=self._stream, keys_extent=bucket.end
        )
        key = self._capture_key(bucket, tables, masks)
        if key != self._key or self._exec == 0:
            self._capture(bucket, tables, masks, key, token=int(token))
        # Every replay, capture included: the capture records the bucket's
        # frozen slot, and only this call knows the live position.
        self._retarget_appends(int(position))
        get_hip_runtime().graph_launch(self._exec, self._stream)
        return runner._collect_block(
            [int(token)],
            apply_softcap=apply_softcap,
            needs_logits=True,
            stream=self._stream,
        )

    def close(self) -> None:
        """Destroy the executable graph, its captured graph and the stream."""

        runtime = get_hip_runtime()
        if self._exec:
            runtime.graph_exec_destroy(self._exec)
            self._exec = 0
        if self._graph:
            runtime.graph_destroy(self._graph)
            self._graph = 0
        if self._stream:
            runtime.stream_destroy(self._stream)
            self._stream = 0
        self._key = None
        self._bucket = None
        self._appends = ()

    def __enter__(self) -> "Gemma4DecodeGraphSession":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- capture ----------------------------------------------------------

    def _frozen_key_begin(self, attention, bucket: _Bucket) -> int:
        window = attention.sliding_window
        if window is None:
            return 0
        # The smallest actual ``key_begin`` inside the bucket is at the bucket
        # start; freezing one below it walks a superset, which the staged mask
        # then bounds. Never above, which would skip live keys.
        return max(0, bucket.start - int(window))

    def _capture_key(
        self,
        bucket: _Bucket,
        tables: dict,
        masks: dict,
    ) -> tuple[object, ...]:
        runner = self._runner
        staged = tuple(
            buffer.ptr
            for group in (tuple(tables.values()), tuple(masks.values()))
            for buffers in group
            for buffer in (buffers if isinstance(buffers, tuple) else (buffers,))
        )
        kv = tuple(
            (layer.key_cache, layer.value_cache) for layer in runner._kv
        )
        return (
            bucket.start,
            bucket.end,
            staged,
            kv,
            runner._hidden.ptr,
            runner._token_ids.ptr,
            runner._logits.ptr,
            runner._normalized.ptr,
        )

    def _capture(
        self,
        bucket: _Bucket,
        tables: dict,
        masks: dict,
        key: tuple[object, ...],
        *,
        token: int,
    ) -> None:
        """Record one decode step with the bucket's frozen launch shape."""

        runner = self._runner
        if runner.position == 0:
            raise RuntimeError(
                "decode-graph capture requires a warmed runner: run at least one "
                "normal forward and one decode step first so every kernel the "
                "graph records is registered and loaded"
            )
        self._retire()

        frozen_write = bucket.end - 1
        runtime = get_hip_runtime()

        def launch(b: _Bucket, tbl: dict, msk: dict) -> None:
            runner._launch_block(
                [token],
                tables=tbl,
                masks=msk,
                kv_write_offset=b.end - 1,
                key_begin_at=lambda attention: self._frozen_key_begin(attention, b),
                stream=self._stream,
                # -1 lets the layer use its parallel MoE stream; Global mode
                # below captures that stream's nodes and event dependencies
                # into this graph, so replay keeps the launched path's
                # two-stream overlap (folding it onto one stream measured
                # 1.6 ms/step slower on device).
                stream_moe=-1,
                needs_logits=True,
                return_hidden=False,
            )

        # Pre-touch the decode route's per-stream split workspace before any
        # capture: AttentionScratch is keyed per stream, and hipMalloc inside
        # an active Global capture is rejected (HIP error 900). Sizes are not
        # monotonic in the key count (measured: keys=1088 needs 69760 B where
        # keys=8192 needs 65728 B), so the max-bucket warm-up above cannot
        # cover every smaller bucket. This workspace is the decode route's only
        # lazy allocation, and touching it costs one small malloc per layer
        # instead of a launch chain per bucket crossing.
        from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
            build_gemma4_attention,
            decode_slices,
            flash_admits,
            flash_slices,
            flash_workspace_bytes,
            split_workspace_bytes,
        )

        library = build_gemma4_attention(load=True)
        for index in range(len(runner.weights.layers)):
            attention = runner.weights.config.geometry(index)
            key_begin = self._frozen_key_begin(attention, bucket)
            keys = frozen_write + 1 - key_begin
            slices = decode_slices(keys, attention.head_dim)
            if slices > 1:
                need = split_workspace_bytes(
                    1,
                    attention.num_heads,
                    attention.head_dim,
                    keys,
                    slices,
                    library=library,
                )
                # The launched route may be flash-decoding (negative slices),
                # whose partials need more than the split's weights buffer;
                # size for whichever the admission can pick, exactly as
                # _launch_prefill does.
                if flash_admits(
                    tokens=1,
                    head_dim=attention.head_dim,
                    num_heads=attention.num_heads,
                    num_kv_heads=attention.num_kv_heads,
                ):
                    need = max(
                        need,
                        flash_workspace_bytes(
                            1,
                            attention.num_heads,
                            attention.head_dim,
                            flash_slices(keys),
                            library=library,
                        ),
                    )
                runner._scratches[index].attention.buffer(
                    need, stream=self._stream, runtime=runtime
                )
        # Global capture mode: see ``launch`` above.
        runtime.stream_begin_capture(self._stream, mode=0)
        try:
            launch(bucket, tables, masks)
            graph = runtime.stream_end_capture(self._stream)
        except Exception:
            # Leave capturing state behind and drop the partial graph: the
            # stream is unusable for capture until EndCapture has run.
            try:
                leaked = runtime.stream_end_capture(self._stream)
                if leaked:
                    runtime.graph_destroy(leaked)
            except Exception:
                pass
            raise
        if graph == 0:
            raise RuntimeError("stream capture produced no graph")
        exec_handle = runtime.graph_instantiate(graph)
        appends = self._identify_appends(graph, frozen_write=frozen_write)
        self._zero_pending_tail(bucket)
        self._graph = graph
        self._exec = exec_handle
        self._key = key
        self._bucket = bucket
        self._appends = appends
        self._captures += 1

    def _identify_appends(self, graph: int, *, frozen_write: int) -> tuple[_CaptureAppend, ...]:
        """Match every captured memcpy node to the layer and cache it belongs to.

        The sources are unique per layer (the scratch's rotated-K and V
        buffers), and a node whose destination is not exactly the frozen
        append slot is a memcpy this model does not expect — a capture that
        contains one is wrong, so it fails here rather than replaying a copy
        nobody re-targets.
        """

        runner = self._runner
        config = runner.weights.config
        layers = len(runner.weights.layers)
        expected: dict[int, tuple[int, str, int, int]] = {}
        for index in range(layers):
            geometry = config.geometry(index)
            kv_width = int(geometry.num_kv_heads) * int(geometry.head_dim)
            scratch = runner._scratches[index]
            kv = runner._kv[index]
            expected[scratch.buffer("k_rot").ptr] = (index, "k", kv.key_cache, kv_width)
            expected[scratch.buffer("v").ptr] = (index, "v", kv.value_cache, kv_width)

        found: dict[tuple[int, str], _CaptureAppend] = {}
        runtime = get_hip_runtime()
        for node in runtime.graph_nodes(graph):
            if runtime.graph_node_type(node) != HIP_GRAPH_NODE_TYPE_MEMCPY:
                continue
            src, dst, count = runtime.graph_memcpy_node_params(node)
            entry = expected.get(int(src))
            if entry is None:
                raise RuntimeError(
                    f"capture contains a memcpy from {src:#x}, which is not a "
                    "layer's rotated-K or V scratch buffer"
                )
            layer, which, cache_base, kv_width = entry
            if int(count) != kv_width * 2:
                raise RuntimeError(
                    f"append for layer {layer} ({which}) copied {count} bytes, "
                    f"expected one row of {kv_width * 2}"
                )
            want = int(cache_base) + frozen_write * kv_width * 2
            if int(dst) != want:
                raise RuntimeError(
                    f"append for layer {layer} ({which}) captured dst {dst:#x}, "
                    f"expected the frozen slot {want:#x}"
                )
            slot = (layer, which)
            if slot in found:
                raise RuntimeError(f"duplicate capture of the {which} append for layer {layer}")
            found[slot] = _CaptureAppend(
                node=int(node),
                src=int(src),
                count=int(count),
                cache_base=int(cache_base),
                kv_width=kv_width,
            )
        if len(found) != 2 * layers:
            raise RuntimeError(
                f"capture holds {len(found)} KV appends, expected {2 * layers} "
                f"(one key and one value per layer over {layers} layers)"
            )
        return tuple(found[key] for key in sorted(found))

    def _zero_pending_tail(self, bucket: _Bucket) -> None:
        """Zero every cache slot in ``[position + 1, bucket.end)`` once.

        Called per capture: the graph will read up to ``bucket.end`` slots,
        and everything above the live position is either stale-but-finite
        (safe: the mask zeroes it exactly) or never written (not safe: a NaN
        pattern in uninitialized device memory poisons the reduction before
        the mask can drop it). Zeroing on the capture stream puts a defined
        value in every slot the frozen shape can address, ahead of the first
        replay; appends overwrite the early part of the range as the bucket
        fills.
        """

        runner = self._runner
        config = runner.weights.config
        first_pending = int(runner.position) + 1
        if first_pending >= bucket.end:
            return
        runtime = get_hip_runtime()
        for index in range(len(runner.weights.layers)):
            geometry = config.geometry(index)
            kv_width = int(geometry.num_kv_heads) * int(geometry.head_dim)
            kv = runner._kv[index]
            slot_bytes = kv_width * 2  # one row of BF16
            offset = first_pending * slot_bytes
            nbytes = (bucket.end - first_pending) * slot_bytes
            runtime.memset_async(int(kv.key_cache) + offset, 0, nbytes, self._stream)
            runtime.memset_async(int(kv.value_cache) + offset, 0, nbytes, self._stream)

    def _retarget_appends(self, position: int) -> None:
        """Point every captured append at the live position's cache slot."""

        runtime = get_hip_runtime()
        for append in self._appends:
            dst = append.cache_base + int(position) * append.kv_width * 2
            runtime.graph_exec_memcpy_node_set_1d(
                self._exec,
                append.node,
                dst=dst,
                src=append.src,
                count=append.count,
                kind=MemcpyKind.DEVICE_TO_DEVICE,
            )

    def _retire(self) -> None:
        """Destroy the current capture, if any, before recording a new one."""

        runtime = get_hip_runtime()
        if self._exec:
            runtime.graph_exec_destroy(self._exec)
            self._exec = 0
        if self._graph:
            runtime.graph_destroy(self._graph)
            self._graph = 0
        self._appends = ()