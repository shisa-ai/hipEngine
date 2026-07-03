"""Per-layer MoE graph capture/replay cache (task #15).

The GGUF decode-graph machinery was retired because the GDN conv/recurrent decode
kernels corrupt device state on the 3rd+ graph relaunch (a stateful in-place
hazard).  The per-layer MoE FFN, by contrast, is STATELESS: it recomputes from
fresh ``hidden``/``attn_out`` inputs each token through fixed scratch pointers and
holds no persistent recurrent state, so it is graph-safe across arbitrarily many
relaunches (proven bit-exact in ``tests/test_hip_graph_capture_replay.py``).

``MoeGraphCache`` wraps such a stateless capturable unit with:

* **capture-on-first-use** keyed by ``(layer_id, hidden_ptr, out_ptr)`` (stable
  per layer across tokens because the decode loop ping-pongs ``hidden_a``/
  ``hidden_b`` with fixed parity, and scratch is session-resident);
* a **self-validating** bit-exact parity check — the captured graph is replayed
  once and compared against a fresh eager reference; a mismatch (or any
  capture/instantiate failure) marks the key eager-only;
* **replay** for every subsequent call;
* an **eager fallback** that keeps decode correct whenever a key is not (yet)
  graphed.

Capture happens on a dedicated non-default stream (HIP forbids capturing the
NULL stream); the instantiated graph is launched on whatever stream the caller
passes (default stream is fine for launch).  The owning session is responsible
for calling :meth:`close` to destroy the graphs and the capture stream while its
buffers are still alive.
"""

from __future__ import annotations

from typing import Callable, Hashable

import numpy as np

from hipengine.core.hip import HipMemcpyKind, HipRuntime

# hipStreamCaptureModeRelaxed — matches tests/test_hip_graph_capture_replay.py.
_CAPTURE_MODE_RELAXED = 2


class MoeGraphCache:
    """Capture/replay cache for a stateless per-layer FFN unit."""

    def __init__(self, runtime: HipRuntime, *, enabled: bool = True) -> None:
        self._runtime = runtime
        self._enabled = bool(enabled)
        self._execs: dict[Hashable, int] = {}
        self._graphs: dict[Hashable, int] = {}
        self._eager_only: set[Hashable] = set()
        self._stats = {"capture": 0, "replay": 0, "eager": 0, "reject": 0}
        self._capture_stream = (
            runtime.stream_create(nonblocking=True) if self._enabled else 0
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def mark_eager_only(self, key: Hashable) -> None:
        """Force ``key`` onto the eager fallback (never captured)."""
        self._eager_only.add(key)

    def run(
        self,
        key: Hashable,
        *,
        eager: Callable[[int], None],
        out_ptr: int,
        out_nbytes: int,
        stream: int,
    ) -> str:
        """Run the capturable unit for ``key``.

        ``eager(stream)`` must launch the full stateless unit on the given stream
        and is the numerically-authoritative path.  ``out_ptr``/``out_nbytes``
        bound the unit's primary device output, snapshotted for the self-validating
        parity check.  Returns one of ``"capture"``/``"replay"``/``"eager"``.
        """
        if not self._enabled:
            eager(stream)
            self._stats["eager"] += 1
            return "eager"
        graph_exec = self._execs.get(key)
        if graph_exec is not None:
            self._runtime.graph_launch(graph_exec, stream)
            self._stats["replay"] += 1
            return "replay"
        if key in self._eager_only:
            eager(stream)
            self._stats["eager"] += 1
            return "eager"
        return self._capture(key, eager, int(out_ptr), int(out_nbytes), stream)

    def _capture(
        self,
        key: Hashable,
        eager: Callable[[int], None],
        out_ptr: int,
        out_nbytes: int,
        stream: int,
    ) -> str:
        rt = self._runtime
        # 1) Eager reference run on the caller's stream (also warms the
        #    dispatch-resolve cache / JIT so capture launches no host stream ops).
        #    A genuine failure here is a real decode bug -> let it propagate.
        eager(stream)
        self._sync(stream)
        ref = self._snapshot(out_ptr, out_nbytes)

        graph = 0
        graph_exec = 0
        try:
            rt.stream_begin_capture(self._capture_stream, _CAPTURE_MODE_RELAXED)
            eager(self._capture_stream)
            graph = rt.stream_end_capture(self._capture_stream)
            if not graph:
                return self._reject(key, None, None)
            graph_exec = rt.graph_instantiate(graph)
            if not graph_exec:
                return self._reject(key, graph, None)
        except Exception:
            # A capture-time error can leave the stream mid-capture; drain it and
            # recreate so one bad layer never poisons the others.
            self._recover_capture_stream()
            return self._reject(key, graph, None)

        # 2) Replay once and validate bit-exact against the eager reference.
        rt.graph_launch(graph_exec, stream)
        self._sync(stream)
        got = self._snapshot(out_ptr, out_nbytes)
        if not np.array_equal(got, ref):
            return self._reject(key, graph, graph_exec)

        self._execs[key] = graph_exec
        self._graphs[key] = graph
        self._stats["capture"] += 1
        return "capture"

    def _reject(self, key: Hashable, graph, graph_exec) -> str:
        rt = self._runtime
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
        self._eager_only.add(key)
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
        for graph_exec in self._execs.values():
            try:
                rt.graph_exec_destroy(graph_exec)
            except Exception:
                pass
        for graph in self._graphs.values():
            try:
                rt.graph_destroy(graph)
            except Exception:
                pass
        self._execs.clear()
        self._graphs.clear()
        self._eager_only.clear()
        if self._capture_stream:
            try:
                rt.stream_destroy(self._capture_stream)
            except Exception:
                pass
            self._capture_stream = 0
