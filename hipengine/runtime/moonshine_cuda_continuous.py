"""Torch-free continuous batching over the Moonshine CUDA static-B decoder.

Both reviewed designs are explicit. ``MoonshineCudaExactContinuousBatchRuntime``
is the correctness-qualified path: live requests are packed into a t32
positions-0-6 region, copied D2D once, then packed into a t256 positions-7-193
region. ``MoonshineCudaContinuousBatchRuntime`` retains the simpler uniform-t256
experiment whose full labeled gate regresses two characters and is not
promotable. Both own FIFO admission/backpressure, independent positions,
EOS/cancellation reclaim, compaction, and a bounded effective-batch graph LRU.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
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

@dataclass(frozen=True)
class MoonshineExactContinuousBatchStep:
    step: int
    admitted: tuple[str, ...]
    tokens: dict[str, int]
    completed: tuple[str, ...]
    transferred: tuple[str, ...]
    early_active: tuple[str, ...]
    mature_active: tuple[str, ...]
    pending: tuple[str, ...]
    graph_keys: tuple[tuple[str, int, str], ...]


@dataclass
class _ExactRegion:
    name: str
    topology: str
    threads: int
    capture_position: int
    decoder: MoonshineCudaBatchRuntime
    active: list[_ActiveRequest] = field(default_factory=list)


@dataclass
class _ExactGraph:
    region: _ExactRegion
    active_batch: int
    graph: int
    graph_exec: int
    capture_wall_ms: float
    instantiate_wall_ms: float
    replay_count: int = 0
    closed: bool = False

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.region.name, self.active_batch, self.region.topology)

    def launch(self) -> None:
        if self.closed:
            raise RuntimeError("exact continuous graph is closed")
        decoder = self.region.decoder
        decoder.runtime.graph_launch(self.graph_exec, decoder.stream)
        self.replay_count += 1

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        decoder = self.region.decoder
        decoder.runtime.graph_exec_destroy(self.graph_exec)
        decoder.runtime.graph_destroy(self.graph)


class MoonshineCudaExactContinuousBatchRuntime:
    """Bit-exact two-region continuous batching without mixed topologies.

    Positions 0-6 execute in a packed t32 region and positions 7-193 in a
    packed t256 region. A request is copied D2D between the two fixed-address
    decoders exactly once at the boundary. The total number of live requests
    never exceeds ``max_batch`` even though each region owns a full-capacity
    workspace.
    """

    def __init__(
        self,
        early_decoder: MoonshineCudaBatchRuntime,
        mature_decoder: MoonshineCudaBatchRuntime,
        *,
        owns_decoders: bool = False,
        max_pending: int | None = None,
        max_graphs: int = 8,
        eos_token_id: int = 2,
    ) -> None:
        for name, decoder in (
            ("early", early_decoder),
            ("mature", mature_decoder),
        ):
            if decoder.closed or decoder.spec is None:
                raise RuntimeError(f"Moonshine {name} batch decoder is closed")
            if decoder.decoder_libraries is None:
                raise RuntimeError(
                    f"prepare {name} decoder kernels before continuous batching"
                )
        if (
            early_decoder.max_batch != mature_decoder.max_batch
            or early_decoder.encoder_frames != mature_decoder.encoder_frames
            or early_decoder.spec != mature_decoder.spec
        ):
            raise ValueError("exact continuous decoders have incompatible geometry")
        if isinstance(max_graphs, bool) or not isinstance(max_graphs, int) or max_graphs <= 0:
            raise ValueError("max_graphs must be a positive integer")
        if max_pending is None:
            max_pending = 4 * early_decoder.max_batch
        if (
            isinstance(max_pending, bool)
            or not isinstance(max_pending, int)
            or max_pending <= 0
        ):
            raise ValueError("max_pending must be a positive integer")
        if (
            isinstance(eos_token_id, bool)
            or not isinstance(eos_token_id, int)
            or not 0 <= eos_token_id < early_decoder.spec.vocab_size
        ):
            raise ValueError("eos_token_id is outside the Moonshine vocabulary")
        self.early_decoder = early_decoder
        self.mature_decoder = mature_decoder
        self.owns_decoders = bool(owns_decoders)
        self.max_batch = early_decoder.max_batch
        self.max_pending = max_pending
        self.max_graphs = max_graphs
        self.eos_token_id = eos_token_id
        self._early = _ExactRegion(
            "early", "positions_0_6_t32_exact", 32, 0, early_decoder
        )
        self._mature = _ExactRegion(
            "mature", "positions_7_193_t256_exact", 256, 7, mature_decoder
        )
        self._pending: deque[_PendingRequest] = deque()
        self._completed: dict[str, MoonshineContinuousResult] = {}
        self._graphs: OrderedDict[tuple[str, int, str], _ExactGraph] = OrderedDict()
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
        self._transfers = 0
        self._cancellations = 0
        self._maximum_active = 0
        self.closed = False

    @property
    def idle(self) -> bool:
        return not self._pending and not self._early.active and not self._mature.active

    @property
    def active_count(self) -> int:
        return len(self._early.active) + len(self._mature.active)

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(
            request.request_id
            for request in (*self._early.active, *self._mature.active)
        )

    @property
    def pending_request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self._pending)

    def _request_exists(self, request_id: str) -> bool:
        return (
            any(request.request_id == request_id for request in self._pending)
            or request_id in self.active_request_ids
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
        if self.closed or self.early_decoder.spec is None:
            raise RuntimeError("exact continuous batch runtime is closed")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        if self._request_exists(request_id):
            raise ValueError(f"request_id {request_id!r} already exists")
        if len(self._pending) >= self.max_pending:
            raise ContinuousBatchBackpressureError(
                f"pending queue is full ({self.max_pending} requests)"
            )
        spec = self.early_decoder.spec
        if len(keys) != spec.decoder_layers or len(values) != spec.decoder_layers:
            raise ValueError(
                f"cross cache needs {spec.decoder_layers} layers, "
                f"got {len(keys)} keys / {len(values)} values"
            )
        expected = (
            spec.decoder_kv_heads,
            self.early_decoder.encoder_frames,
            spec.head_dim,
        )
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
        if copied_mask.shape != (self.early_decoder.encoder_frames,):
            raise ValueError(
                f"cross-cache mask shape {copied_mask.shape} != "
                f"{(self.early_decoder.encoder_frames,)}"
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
        while self._pending and self.active_count < self.max_batch:
            request = self._pending.popleft()
            row = len(self._early.active)
            self.early_decoder.load_cross_cache_row(
                row, request.keys, request.values, mask=request.mask
            )
            self._early.active.append(
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
        self._maximum_active = max(self._maximum_active, self.active_count)
        return tuple(admitted)

    def _capture_graph(self, region: _ExactRegion, active_batch: int) -> _ExactGraph:
        decoder = region.decoder
        runtime = decoder.runtime
        runtime.stream_synchronize(decoder.stream)
        graph = 0
        capture_started = time.perf_counter_ns()
        runtime.stream_begin_capture(decoder.stream)
        try:
            decoder._enqueue_batch_token_step(
                route_position=region.capture_position,
                threads=region.threads,
                stream=decoder.stream,
                active_batch=active_batch,
            )
            graph = runtime.stream_end_capture(decoder.stream)
        except Exception:
            try:
                leaked = runtime.stream_end_capture(decoder.stream)
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
        return _ExactGraph(
            region=region,
            active_batch=active_batch,
            graph=graph,
            graph_exec=graph_exec,
            capture_wall_ms=capture_wall_ms,
            instantiate_wall_ms=instantiate_wall_ms,
        )

    def _graph(self, region: _ExactRegion, active_batch: int) -> _ExactGraph:
        key = (region.name, active_batch, region.topology)
        graph = self._graphs.pop(key, None)
        if graph is not None:
            self._graphs[key] = graph
            return graph
        graph = self._capture_graph(region, active_batch)
        if len(self._graphs) >= self.max_graphs:
            _old_key, old = self._graphs.popitem(last=False)
            old.close()
            self._evictions += 1
        self._graphs[key] = graph
        return graph

    def _complete_region_row(
        self, region: _ExactRegion, row: int, reason: str
    ) -> str:
        request = region.active[row]
        self._completed[request.request_id] = MoonshineContinuousResult(
            request_id=request.request_id,
            tokens=tuple(request.tokens),
            reason=reason,
            submitted_sequence=request.submitted_sequence,
            admitted_step=request.admitted_step,
            completed_step=self._step,
        )
        self._completion_count += 1
        last = len(region.active) - 1
        if row != last:
            region.decoder.move_batch_row(last, row)
            region.active[row] = region.active[last]
            self._compactions += 1
        region.active.pop()
        return request.request_id

    def _run_region(
        self, region: _ExactRegion
    ) -> tuple[dict[str, int], list[str], tuple[str, int, str] | None]:
        if not region.active:
            return {}, [], None
        active_batch = len(region.active)
        region.decoder.set_mixed_batch_decode_state(
            tokens=[request.token_id for request in region.active],
            positions=[request.position for request in region.active],
            active_batch=active_batch,
        )
        graph = self._graph(region, active_batch)
        graph.launch()
        self._replays += 1
        output = region.decoder.read_tokens(active_batch=active_batch)
        emitted = {
            request.request_id: int(output[row])
            for row, request in enumerate(region.active)
        }
        for row, request in enumerate(region.active):
            token_id = int(output[row])
            request.tokens.append(token_id)
            request.token_id = token_id
            request.position += 1
            if token_id == self.eos_token_id:
                request.finish_reason = "eos"
            elif request.position >= region.decoder.spec.self_cache_capacity:
                request.finish_reason = "max_positions"
        completed: list[str] = []
        row = 0
        while row < len(region.active):
            reason = region.active[row].finish_reason
            if reason is None:
                row += 1
            else:
                completed.append(self._complete_region_row(region, row, reason))
        return emitted, completed, graph.key

    def _transfer_mature(self) -> tuple[str, ...]:
        transferred: list[str] = []
        row = 0
        while row < len(self._early.active):
            request = self._early.active[row]
            if request.position < 7:
                row += 1
                continue
            destination = len(self._mature.active)
            self.early_decoder.copy_batch_row_to(
                self.mature_decoder, row, destination
            )
            last = len(self._early.active) - 1
            moving = request
            if row != last:
                self.early_decoder.move_batch_row(last, row)
                self._early.active[row] = self._early.active[last]
                self._compactions += 1
            self._early.active.pop()
            self._mature.active.append(moving)
            transferred.append(moving.request_id)
            self._transfers += 1
        return tuple(transferred)

    def step(self) -> MoonshineExactContinuousBatchStep:
        if self.closed:
            raise RuntimeError("exact continuous batch runtime is closed")
        admitted = self._admit()
        emitted: dict[str, int] = {}
        completed: list[str] = []
        graph_keys: list[tuple[str, int, str]] = []
        # Mature rows run first so a just-crossed early row cannot advance twice
        # in one scheduler step.
        for region in (self._mature, self._early):
            region_tokens, region_completed, graph_key = self._run_region(region)
            emitted.update(region_tokens)
            completed.extend(region_completed)
            if graph_key is not None:
                graph_keys.append(graph_key)
        transferred = self._transfer_mature()
        result = MoonshineExactContinuousBatchStep(
            step=self._step,
            admitted=admitted,
            tokens=emitted,
            completed=tuple(completed),
            transferred=transferred,
            early_active=tuple(request.request_id for request in self._early.active),
            mature_active=tuple(request.request_id for request in self._mature.active),
            pending=self.pending_request_ids,
            graph_keys=tuple(graph_keys),
        )
        self._step += 1
        return result

    def cancel(self, request_id: str) -> bool:
        if self.closed:
            raise RuntimeError("exact continuous batch runtime is closed")
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
        for region in (self._early, self._mature):
            for row, request in enumerate(region.active):
                if request.request_id == request_id:
                    self._complete_region_row(region, row, "cancelled_active")
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
            "topology": "exact_two_region",
            "regions": {
                "early": self._early.topology,
                "mature": self._mature.topology,
            },
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
            "max_batch": self.max_batch,
            "max_pending": self.max_pending,
            "submitted": self._submit_sequence,
            "admissions": self._admissions,
            "completed": self._completion_count,
            "cancellations": self._cancellations,
            "compactions": self._compactions,
            "region_transfers": self._transfers,
            "steps": self._step,
            "maximum_active": self._maximum_active,
            "early_active": [request.request_id for request in self._early.active],
            "mature_active": [request.request_id for request in self._mature.active],
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
        self._early.active.clear()
        self._mature.active.clear()
        if self.owns_decoders:
            # Reverse construction order is the expected ownership discipline.
            self.mature_decoder.close()
            self.early_decoder.close()

    def __enter__(self) -> "MoonshineCudaExactContinuousBatchRuntime":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
