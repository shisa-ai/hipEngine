"""Whole-token-step hipGraph capture/replay cache for the Maple decode step.

The Maple c1 decode step is STATELESS across tokens in the graph-relevant
sense: every kernel reads fresh device buffers through fixed session-resident
pointers, and all per-token variation lives in device memory that the kernels
address by pointer at execution time.  In particular the KV cache growth is
carried by the ``KVLiveSpans`` ABI — ``live_counts``/``token_positions``/
``evict_mask``/``row_positions`` are passed to the attention and KV-write
kernels as *device pointers*, and ``_publish_span_position`` updates those
buffers eagerly (host launch) before the graph is replayed.  So a single graph
captured once against the fixed pointer set is valid across arbitrarily many
token positions and relaunches, provided the eager span update runs first.

What stays eager (host scalar args that change per token / host copies):

* ``maple_affine4_embed_bf16`` — takes ``token`` as a host int.
* ``_publish_span_position`` — takes ``position`` as a host int.
* The final ``copy_device_to_host`` reads of ``argmax_index``/``argmax_value``.

What is captured as one graph: the 24-layer loop body plus the top-level tail
(final rmsnorm, affine4 lm_head, argmax) — launched against fixed pointers.

The cache self-validates: on first capture it replays the graph once and
compares the argmax index/value to a fresh eager reference; a mismatch (or any
capture/instantiate failure) marks the cache eager-only and keeps decode
correct.  The eager fallback is always available.

The graph is owned by the runner session and must be :meth:`close` d while its
buffers are still alive.
"""

from __future__ import annotations

from typing import Callable, Hashable

import numpy as np

from hipengine.core.hip import HipMemcpyKind, HipRuntime

# hipStreamCaptureModeRelaxed — matches MoeGraphCache / the HIP graph tests.
_CAPTURE_MODE_RELAXED = 2


class MapleGraphCache:
    """Capture/replay cache for the stateless whole-token Maple decode step."""

    def __init__(self, runtime: HipRuntime, *, enabled: bool = True) -> None:
        self._runtime = runtime
        self._enabled = bool(enabled)
        self._graph_exec: int | None = None
        self._graph: int | None = None
        self._eager_only = False
        self._stats = {"capture": 0, "replay": 0, "eager": 0, "reject": 0}
        self._capture_stream = (
            runtime.stream_create(nonblocking=True) if self._enabled else 0
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def eager_only(self) -> bool:
        return self._eager_only

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def run(
        self,
        key: Hashable,
        *,
        eager: Callable[[int], None],
        argmax_index_ptr: int,
        argmax_value_ptr: int,
        mutable_inputs: tuple[tuple[int, int], ...] = (),
        stream: int,
    ) -> str:
        """Run the captured decode step for ``key``.

        ``eager(stream)`` must launch the full stateless step (24 layers + tail)
        on the given stream and is the numerically-authoritative path.  The
        caller must have already run the eager per-token span update and embed
        so ``mutable_inputs`` (e.g. ``hidden``) hold this token's input and the
        KV spans are current.  ``argmax_index_ptr``/``argmax_value_ptr`` bound
        the graph's scalar outputs, snapshotted for the self-validating parity
        check.  Returns one of ``"capture"``/``"replay"``/``"eager"``.
        """
        if not self._enabled:
            eager(stream)
            self._stats["eager"] += 1
            return "eager"
        if self._graph_exec is not None:
            self._runtime.graph_launch(self._graph_exec, stream)
            self._stats["replay"] += 1
            return "replay"
        if self._eager_only:
            eager(stream)
            self._stats["eager"] += 1
            return "eager"
        normalized_inputs = tuple(
            (int(input_ptr), int(input_nbytes))
            for input_ptr, input_nbytes in mutable_inputs
        )
        if any(
            input_ptr <= 0 or input_nbytes <= 0
            for input_ptr, input_nbytes in normalized_inputs
        ):
            raise ValueError("mutable graph inputs require positive pointers and byte counts")
        return self._capture(
            key,
            eager,
            int(argmax_index_ptr),
            int(argmax_value_ptr),
            normalized_inputs,
            stream,
        )

    def _capture(
        self,
        key: Hashable,
        eager: Callable[[int], None],
        index_ptr: int,
        value_ptr: int,
        mutable_inputs: tuple[tuple[int, int], ...],
        stream: int,
    ) -> str:
        rt = self._runtime
        cap = self._capture_stream
        # Everything runs on the single capture stream with full syncs between
        # phases.  The Maple step body writes KV as a side effect, so running it
        # on two streams (caller + capture) races and corrupts the cache; a
        # single serialized stream keeps the eager ref, capture, and replay
        # validation deterministic.  The embed that produced ``hidden`` ran on
        # the caller stream, so barrier the whole device before the capture
        # stream reads it.
        rt.device_synchronize()
        input_snapshots = tuple(
            (input_ptr, self._snapshot(input_ptr, input_nbytes))
            for input_ptr, input_nbytes in mutable_inputs
        )
        # 1) Eager reference run on the capture stream (warms dispatch/JIT so
        #    capture launches no host stream ops). A genuine failure here is a
        #    real decode bug -> let it propagate.
        eager(cap)
        self._sync(cap)
        ref_index = self._snapshot(index_ptr, 8)
        ref_value = self._snapshot(value_ptr, 4)

        graph = 0
        graph_exec = 0
        try:
            # The eager ref just overwrote ``hidden`` with the layer output;
            # restore the fresh capture input before recording the graph so it
            # captures a correct computation.
            for input_ptr, snapshot in input_snapshots:
                rt.memcpy(
                    input_ptr,
                    int(snapshot.ctypes.data),
                    int(snapshot.nbytes),
                    HipMemcpyKind.HOST_TO_DEVICE,
                )
            rt.stream_begin_capture(cap, _CAPTURE_MODE_RELAXED)
            eager(cap)
            graph = rt.stream_end_capture(cap)
            if not graph:
                return self._reject(key, eager, stream, None, None)
            graph_exec = rt.graph_instantiate(graph)
            if not graph_exec:
                return self._reject(key, eager, stream, graph, None)
        except Exception:
            self._recover_capture_stream()
            return self._reject(key, eager, stream, graph, None)

        # 2) Replay once on the capture stream and validate bit-exact against
        #    the eager reference. Restore the fresh capture input (hidden)
        #    before validation, and clear the argmax outputs so a no-op/empty
        #    graph is detected instead of vacuously matching the prior eager
        #    reference. Ordinary replays receive a freshly produced embed
        #    output and current spans.
        for input_ptr, snapshot in input_snapshots:
            rt.memcpy(
                input_ptr,
                int(snapshot.ctypes.data),
                int(snapshot.nbytes),
                HipMemcpyKind.HOST_TO_DEVICE,
            )
        zero_idx = np.zeros(8, dtype=np.uint8)
        zero_val = np.zeros(4, dtype=np.uint8)
        rt.memcpy(
            index_ptr,
            int(zero_idx.ctypes.data),
            8,
            HipMemcpyKind.HOST_TO_DEVICE,
        )
        rt.memcpy(
            value_ptr,
            int(zero_val.ctypes.data),
            4,
            HipMemcpyKind.HOST_TO_DEVICE,
        )
        rt.graph_launch(graph_exec, cap)
        self._sync(cap)
        got_index = self._snapshot(index_ptr, 8)
        got_value = self._snapshot(value_ptr, 4)
        if not (
            np.array_equal(got_index, ref_index)
            and np.array_equal(got_value, ref_value)
        ):
            return self._reject(key, eager, stream, graph, graph_exec)

        self._graph_exec = graph_exec
        self._graph = graph
        self._stats["capture"] += 1
        # Leave the caller's expected final argmax state: restore the fresh
        # input (the validation replay overwrote ``hidden`` with layer output),
        # then rerun the eager ref on the caller stream so the token consumed by
        # this step reflects a clean, correct execution (the capture-stream
        # validation runs were for checking only).
        for input_ptr, snapshot in input_snapshots:
            rt.memcpy(
                input_ptr,
                int(snapshot.ctypes.data),
                int(snapshot.nbytes),
                HipMemcpyKind.HOST_TO_DEVICE,
            )
        eager(stream)
        self._sync(stream)
        return "capture"

    def _reject(self, key: Hashable, eager, stream, graph, graph_exec) -> str:
        rt = self._runtime
        # Restore clean, correct caller state: rerun the eager body on the
        # caller stream so the token consumed by this step is correct and the
        # KV cache is left in the eager-consistent state for subsequent steps.
        try:
            eager(stream)
            self._sync(stream)
        except Exception:
            pass
        if graph_exec:
            try:
                rt.graph_exec_destroy(graph_exec)
            except Exception:
                pass
        if graph:
            try:
                rt.graph_destroy(graph)
            except Exception:
                pass
        self._eager_only = True
        self._stats["reject"] += 1
        return "eager"

    def _recover_capture_stream(self) -> None:
        rt = self._runtime
        try:
            rt.stream_end_capture(self._capture_stream)
        except Exception:
            pass
        try:
            rt.stream_destroy(self._capture_stream)
        except Exception:
            pass
        try:
            self._capture_stream = rt.stream_create(nonblocking=True)
        except Exception:
            self._capture_stream = 0
            self._enabled = False

    def _snapshot(self, out_ptr: int, nbytes: int) -> np.ndarray:
        buf = np.empty(int(nbytes), dtype=np.uint8)
        self._runtime.memcpy(
            int(buf.ctypes.data), int(out_ptr), int(nbytes), HipMemcpyKind.DEVICE_TO_HOST
        )
        return buf

    def _sync(self, stream: int) -> None:
        if stream:
            self._runtime.stream_synchronize(stream)
        else:
            self._runtime.device_synchronize()

    def close(self) -> None:
        rt = self._runtime
        if self._graph_exec:
            try:
                rt.graph_exec_destroy(self._graph_exec)
            except Exception:
                pass
        if self._graph:
            try:
                rt.graph_destroy(self._graph)
            except Exception:
                pass
        self._graph_exec = None
        self._graph = None
        self._eager_only = False
        if self._capture_stream:
            try:
                rt.stream_destroy(self._capture_stream)
            except Exception:
                pass
            self._capture_stream = 0
