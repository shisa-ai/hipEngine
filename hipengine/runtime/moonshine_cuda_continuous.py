"""Torch-free continuous batching over the Moonshine CUDA static-B decoder.

The scheduler uses the re-derived uniform-t256 design from the CUDA campaign:
live requests carry independent positions, are packed into prefix rows, and
replay one graph keyed by effective batch size.  Admission installs one row's
cross K/V without disturbing peers; EOS/cancellation reclaims a row and moves
the trailing live row's self/cross state into the hole.  A bounded LRU owns the
``(active_batch, uniform_t256)`` graph set.

This route intentionally has a changed arithmetic contract at positions 0-6:
parallel t256 self-attention replaces the c=1 one-wave t32 schedule.  Callers
must use it only under the separately retained full-route quality gate.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hipengine.runtime.moonshine_cuda_batch import MoonshineCudaBatchRuntime

_TOPOLOGY = "uniform_t256_rederived"
_THREADS = 256


class ContinuousBatchBackpressureError(RuntimeError):
    """The bounded pending queue has no admission capacity."""


@dataclass(frozen=True)
class MoonshineContinuousResult:
    request_id: str
    tokens: tuple[int, ...]
    reason: str
    submitted_sequence: int
    admitted_step: int | None
    completed_step: int


@dataclass(frozen=True)
class MoonshineContinuousBatchStep:
    step: int
    admitted: tuple[str, ...]
    tokens: dict[str, int]
    completed: tuple[str, ...]
    active: tuple[str, ...]
    pending: tuple[str, ...]
    graph_key: tuple[int, str] | None


@dataclass
class _PendingRequest:
    request_id: str
    keys: tuple[np.ndarray, ...]
    values: tuple[np.ndarray, ...]
    mask: np.ndarray
    seed_token_id: int
    submitted_sequence: int


@dataclass
class _ActiveRequest:
    request_id: str
    token_id: int
    position: int
    tokens: list[int]
    submitted_sequence: int
    admitted_step: int
    finish_reason: str | None = None


@dataclass
class _ContinuousGraph:
    decoder: MoonshineCudaBatchRuntime
    active_batch: int
    graph: int
    graph_exec: int
    capture_wall_ms: float
    instantiate_wall_ms: float
    replay_count: int = 0
    closed: bool = False

    @property
    def key(self) -> tuple[int, str]:
        return (self.active_batch, _TOPOLOGY)

    def launch(self) -> None:
        if self.closed:
            raise RuntimeError("continuous batch graph is closed")
        self.decoder.runtime.graph_launch(self.graph_exec, self.decoder.stream)
        self.replay_count += 1

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.decoder.runtime.graph_exec_destroy(self.graph_exec)
        self.decoder.runtime.graph_destroy(self.graph)


class MoonshineCudaContinuousBatchRuntime:
    """FIFO admission, mixed-position decode, reclaim, and bounded graph LRU."""

    def __init__(
        self,
        decoder: MoonshineCudaBatchRuntime,
        *,
        owns_decoder: bool = False,
        max_pending: int | None = None,
        max_graphs: int = 4,
        eos_token_id: int = 2,
    ) -> None:
        if decoder.closed or decoder.spec is None:
            raise RuntimeError("Moonshine batch decoder is closed")
        if decoder.decoder_libraries is None:
            raise RuntimeError("prepare decoder kernels before continuous batching")
        if isinstance(max_graphs, bool) or not isinstance(max_graphs, int) or max_graphs <= 0:
            raise ValueError("max_graphs must be a positive integer")
        if max_pending is None:
            max_pending = 4 * decoder.max_batch
        if (
            isinstance(max_pending, bool)
            or not isinstance(max_pending, int)
            or max_pending <= 0
        ):
            raise ValueError("max_pending must be a positive integer")
        if (
            isinstance(eos_token_id, bool)
            or not isinstance(eos_token_id, int)
            or not 0 <= eos_token_id < decoder.spec.vocab_size
        ):
            raise ValueError("eos_token_id is outside the Moonshine vocabulary")
        self.decoder = decoder
        self.owns_decoder = bool(owns_decoder)
        self.max_pending = max_pending
        self.max_graphs = max_graphs
        self.eos_token_id = eos_token_id
        self._pending: deque[_PendingRequest] = deque()
        self._active: list[_ActiveRequest] = []
        self._completed: dict[str, MoonshineContinuousResult] = {}
        self._graphs: OrderedDict[tuple[int, str], _ContinuousGraph] = OrderedDict()
        self._step = 0
        self._submit_sequence = 0
        self._captures = 0
        self._evictions = 0
        self._replays = 0
        self._capture_wall_ms = 0.0
        self._instantiate_wall_ms = 0.0
        self._admissions = 0
        self._completion_count = 0
        self._compactions = 0
        self._cancellations = 0
        self._maximum_active = 0
        self.closed = False

    @property
    def idle(self) -> bool:
        return not self._pending and not self._active

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self._active)

    @property
    def pending_request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self._pending)

    def _request_exists(self, request_id: str) -> bool:
        return (
            any(request.request_id == request_id for request in self._pending)
            or any(request.request_id == request_id for request in self._active)
            or request_id in self._completed
        )

    def submit(
        self,
        request_id: str,
        keys: Sequence[np.ndarray],
        values: Sequence[np.ndarray],
        *,
        mask: np.ndarray,
        seed_token_id: int = 1,
    ) -> None:
        """Queue one cross-cache-ready request under FIFO/backpressure policy."""

        if self.closed or self.decoder.spec is None:
            raise RuntimeError("continuous batch runtime is closed")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        if self._request_exists(request_id):
            raise ValueError(f"request_id {request_id!r} already exists")
        if len(self._pending) >= self.max_pending:
            raise ContinuousBatchBackpressureError(
                f"pending queue is full ({self.max_pending} requests)"
            )
        spec = self.decoder.spec
        if len(keys) != spec.decoder_layers or len(values) != spec.decoder_layers:
            raise ValueError(
                f"cross cache needs {spec.decoder_layers} layers, "
                f"got {len(keys)} keys / {len(values)} values"
            )
        expected = (spec.decoder_kv_heads, self.decoder.encoder_frames, spec.head_dim)
        copied_keys: list[np.ndarray] = []
        copied_values: list[np.ndarray] = []
        for layer in range(spec.decoder_layers):
            key = np.asarray(keys[layer], dtype=np.float16)
            value = np.asarray(values[layer], dtype=np.float16)
            if key.shape != expected or value.shape != expected:
                raise ValueError(
                    f"layer {layer} cross-cache shape {key.shape}/{value.shape} "
                    f"!= {expected}"
                )
            copied_keys.append(np.ascontiguousarray(key).copy())
            copied_values.append(np.ascontiguousarray(value).copy())
        copied_mask = np.asarray(mask, dtype=np.int32).reshape(-1)
        if copied_mask.shape != (self.decoder.encoder_frames,):
            raise ValueError(
                f"cross-cache mask shape {copied_mask.shape} != "
                f"{(self.decoder.encoder_frames,)}"
            )
        if not bool(((copied_mask == 0) | (copied_mask == 1)).all()):
            raise ValueError("cross-cache mask must be binary")
        if (
            isinstance(seed_token_id, bool)
            or not isinstance(seed_token_id, int)
            or not 0 <= seed_token_id < spec.vocab_size
        ):
            raise ValueError("seed_token_id is outside the Moonshine vocabulary")
        self._pending.append(
            _PendingRequest(
                request_id=request_id,
                keys=tuple(copied_keys),
                values=tuple(copied_values),
                mask=np.ascontiguousarray(copied_mask).copy(),
                seed_token_id=seed_token_id,
                submitted_sequence=self._submit_sequence,
            )
        )
        self._submit_sequence += 1

    def _admit(self) -> tuple[str, ...]:
        admitted: list[str] = []
        while self._pending and len(self._active) < self.decoder.max_batch:
            request = self._pending.popleft()
            row = len(self._active)
            self.decoder.load_cross_cache_row(
                row,
                request.keys,
                request.values,
                mask=request.mask,
            )
            self._active.append(
                _ActiveRequest(
                    request_id=request.request_id,
                    token_id=request.seed_token_id,
                    position=0,
                    tokens=[],
                    submitted_sequence=request.submitted_sequence,
                    admitted_step=self._step,
                )
            )
            admitted.append(request.request_id)
            self._admissions += 1
        self._maximum_active = max(self._maximum_active, len(self._active))
        return tuple(admitted)

    def _capture_graph(self, active_batch: int) -> _ContinuousGraph:
        runtime = self.decoder.runtime
        stream = self.decoder.stream
        runtime.stream_synchronize(stream)
        graph = 0
        capture_started = time.perf_counter_ns()
        runtime.stream_begin_capture(stream)
        try:
            self.decoder._enqueue_batch_token_step(
                route_position=7,
                threads=_THREADS,
                stream=stream,
                active_batch=active_batch,
            )
            graph = runtime.stream_end_capture(stream)
        except Exception:
            try:
                leaked = runtime.stream_end_capture(stream)
                if leaked:
                    runtime.graph_destroy(leaked)
            except Exception:
                pass
            raise
        capture_wall_ms = (time.perf_counter_ns() - capture_started) * 1.0e-6
        instantiate_started = time.perf_counter_ns()
        try:
            graph_exec = runtime.graph_instantiate(graph)
        except Exception:
            runtime.graph_destroy(graph)
            raise
        instantiate_wall_ms = (time.perf_counter_ns() - instantiate_started) * 1.0e-6
        self._captures += 1
        self._capture_wall_ms += capture_wall_ms
        self._instantiate_wall_ms += instantiate_wall_ms
        return _ContinuousGraph(
            decoder=self.decoder,
            active_batch=active_batch,
            graph=graph,
            graph_exec=graph_exec,
            capture_wall_ms=capture_wall_ms,
            instantiate_wall_ms=instantiate_wall_ms,
        )

    def _graph(self, active_batch: int) -> _ContinuousGraph:
        key = (active_batch, _TOPOLOGY)
        graph = self._graphs.pop(key, None)
        if graph is not None:
            self._graphs[key] = graph
            return graph
        graph = self._capture_graph(active_batch)
        if len(self._graphs) >= self.max_graphs:
            _old_key, old = self._graphs.popitem(last=False)
            old.close()
            self._evictions += 1
        self._graphs[key] = graph
        return graph

    def _finish_active_row(self, row: int, reason: str) -> str:
        request = self._active[row]
        result = MoonshineContinuousResult(
            request_id=request.request_id,
            tokens=tuple(request.tokens),
            reason=reason,
            submitted_sequence=request.submitted_sequence,
            admitted_step=request.admitted_step,
            completed_step=self._step,
        )
        self._completed[request.request_id] = result
        self._completion_count += 1
        last = len(self._active) - 1
        if row != last:
            self.decoder.move_batch_row(last, row)
            self._active[row] = self._active[last]
            self._compactions += 1
        self._active.pop()
        return request.request_id

    def step(self) -> MoonshineContinuousBatchStep:
        """Admit FIFO work and advance every live request by one token."""

        if self.closed:
            raise RuntimeError("continuous batch runtime is closed")
        admitted = self._admit()
        if not self._active:
            result = MoonshineContinuousBatchStep(
                step=self._step,
                admitted=admitted,
                tokens={},
                completed=(),
                active=(),
                pending=self.pending_request_ids,
                graph_key=None,
            )
            self._step += 1
            return result

        active_batch = len(self._active)
        self.decoder.set_mixed_batch_decode_state(
            tokens=[request.token_id for request in self._active],
            positions=[request.position for request in self._active],
            active_batch=active_batch,
        )
        graph = self._graph(active_batch)
        graph.launch()
        self._replays += 1
        output = self.decoder.read_tokens(active_batch=active_batch)
        emitted = {
            request.request_id: int(output[row])
            for row, request in enumerate(self._active)
        }
        for row, request in enumerate(self._active):
            token_id = int(output[row])
            request.tokens.append(token_id)
            request.token_id = token_id
            request.position += 1
            if token_id == self.eos_token_id:
                request.finish_reason = "eos"
            elif request.position >= self.decoder.spec.self_cache_capacity:
                request.finish_reason = "max_positions"

        completed: list[str] = []
        row = 0
        while row < len(self._active):
            reason = self._active[row].finish_reason
            if reason is None:
                row += 1
            else:
                completed.append(self._finish_active_row(row, reason))
        result = MoonshineContinuousBatchStep(
            step=self._step,
            admitted=admitted,
            tokens=emitted,
            completed=tuple(completed),
            active=self.active_request_ids,
            pending=self.pending_request_ids,
            graph_key=graph.key,
        )
        self._step += 1
        return result

    def cancel(self, request_id: str) -> bool:
        """Cancel queued or active work without disturbing other requests."""

        if self.closed:
            raise RuntimeError("continuous batch runtime is closed")
        for index, request in enumerate(self._pending):
            if request.request_id == request_id:
                pending = list(self._pending)
                pending.pop(index)
                self._pending = deque(pending)
                self._completed[request_id] = MoonshineContinuousResult(
                    request_id=request_id,
                    tokens=(),
                    reason="cancelled_pending",
                    submitted_sequence=request.submitted_sequence,
                    admitted_step=None,
                    completed_step=self._step,
                )
                self._completion_count += 1
                self._cancellations += 1
                return True
        for row, request in enumerate(self._active):
            if request.request_id == request_id:
                self._finish_active_row(row, "cancelled_active")
                self._cancellations += 1
                return True
        return False

    def take_completed(self, request_id: str) -> MoonshineContinuousResult:
        try:
            return self._completed.pop(request_id)
        except KeyError as error:
            raise KeyError(f"request {request_id!r} is not completed") from error

    def graph_cache_contract(self) -> dict[str, object]:
        return {
            "topology": _TOPOLOGY,
            "threads": _THREADS,
            "max_graphs": self.max_graphs,
            "size": len(self._graphs),
            "keys": [list(key) for key in self._graphs],
            "captures": self._captures,
            "evictions": self._evictions,
            "replays": self._replays,
            "resident_graph_replays": sum(
                graph.replay_count for graph in self._graphs.values()
            ),
            "capture_wall_ms": self._capture_wall_ms,
            "instantiate_wall_ms": self._instantiate_wall_ms,
        }

    def scheduler_contract(self) -> dict[str, object]:
        return {
            "max_batch": self.decoder.max_batch,
            "max_pending": self.max_pending,
            "submitted": self._submit_sequence,
            "admissions": self._admissions,
            "completed": self._completion_count,
            "cancellations": self._cancellations,
            "compactions": self._compactions,
            "steps": self._step,
            "maximum_active": self._maximum_active,
            "active": list(self.active_request_ids),
            "pending": list(self.pending_request_ids),
            "completed_waiting": sorted(self._completed),
            "idle": self.idle,
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for graph in reversed(tuple(self._graphs.values())):
            graph.close()
        self._graphs.clear()
        self._pending.clear()
        self._active.clear()
        if self.owns_decoder:
            self.decoder.close()

    def __enter__(self) -> "MoonshineCudaContinuousBatchRuntime":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
