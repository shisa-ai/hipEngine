"""Session-local one-step HIP graph replay for exact Laguna raw-greedy decode."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

from hipengine.kernels.backends import (
    backend_package_capability,
    hip_target_arch_for_backend,
)


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ptr(value: Any) -> int | None:
    pointer = getattr(value, "ptr", None)
    return None if pointer is None else int(pointer)


def _weight_role_guard(session: Any) -> tuple[tuple[Any, ...], ...]:
    assert session.weights is not None
    return tuple(
        (
            str(weight.spec.slot_path),
            str(weight.spec.source.name),
            str(weight.spec.quant_key),
            str(weight.spec.layout),
            tuple(int(value) for value in weight.spec.source.shape),
            int(weight.spec.source.ggml_type),
            int(weight.spec.source.nbytes),
            int(weight.spec.source.data_offset),
        )
        for weight in session.weights.weights
    )


def _static_buffer_ptrs(session: Any) -> tuple[int, ...]:
    assert session.weights is not None
    assert session.scratch is not None
    assert session.moe_scratch is not None
    assert session.kv_cache is not None
    assert session.full_rope is not None
    assert session.swa_rope is not None
    pointers: list[int] = []

    def add(value: Any) -> None:
        pointer = _ptr(value)
        if pointer is not None:
            pointers.append(pointer)

    for weight in session.weights.weights:
        for allocation in weight.allocations.values():
            add(allocation.tensor)
    for buffer in session.scratch.buffers:
        add(buffer)
    for buffer in session.moe_scratch.buffers:
        add(buffer)
    for buffer in session.kv_cache.buffers:
        add(buffer)
    for layer in session.kv_cache.layers:
        add(layer.key_cache)
        add(layer.value_cache)
        spans = layer.spans
        for field in (
            "base_offsets",
            "live_counts",
            "token_positions",
            "evict_mask",
            "row_positions",
        ):
            add(getattr(spans, field, None))
    for tables in (session.full_rope, session.swa_rope):
        add(tables.cos.buffer)
        add(tables.sin.buffer)
    return tuple(pointers)


def _static_replay_guard(session: Any) -> tuple[Any, ...]:
    assert session.weights is not None
    config = session.weights.config
    return (
        str(session.backend),
        str(getattr(session, "target_arch", hip_target_arch_for_backend(session.backend))).split(
            ":", 1
        )[0],
        int(session.context_length),
        int(config.hidden_size),
        int(config.vocab_size),
        tuple(str(value) for value in config.layer_types),
        tuple(int(value) for value in config.head_counts),
        int(config.sliding_window),
        str(session.selected_down_mode),
        str(session.swa_decode_variant),
        str(session.swa_prefill_variant),
        _static_buffer_ptrs(session),
    )


@dataclass(frozen=True)
class LagunaDecodeGraphKey:
    schema_version: int
    backend: str
    target_arch: str
    active_rows: int
    context_length: int
    sliding_window: int
    hidden_size: int
    vocab_size: int
    layer_types: tuple[str, ...]
    head_counts: tuple[int, ...]
    kv_storage_dtype: str
    sampler_mode: str
    selected_down_mode: str
    swa_decode_variant: str
    swa_prefill_variant: str
    role_sha256: str
    buffer_identity_sha256: str
    buffer_count: int
    key_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))


@dataclass(frozen=True)
class LagunaDecodeReplayTicket:
    schema_version: int
    lifecycle_epoch: int
    session_position: int
    kv_position: int
    next_position: int
    remaining_capacity: int
    pending_positions: tuple[int, ...]
    dense_span_policy: str
    graph_compatible: bool
    primed: bool


def build_laguna_decode_replay_ticket(
    session: Any,
    graph: "LagunaDecodeGraph | None" = None,
) -> LagunaDecodeReplayTicket:
    kv_cache = session.kv_cache
    session_position = int(session.position)
    kv_position = int(kv_cache.position)
    epoch = int(kv_cache.graph_state_epoch)
    return LagunaDecodeReplayTicket(
        schema_version=1,
        lifecycle_epoch=epoch,
        session_position=session_position,
        kv_position=kv_position,
        next_position=session_position + 1,
        remaining_capacity=int(session.context_length) - (session_position + 1),
        pending_positions=tuple(int(value) for value in kv_cache.pending_positions),
        dense_span_policy="cursor_derived_no_eviction",
        graph_compatible=bool(kv_cache.graph_compatible),
        primed=bool(
            graph is not None
            and not graph.closed
            and graph.primed_position == session_position
            and graph.primed_kv_epoch == epoch
        ),
    )


def build_laguna_decode_graph_key(
    session: Any,
    *,
    require_supported_backend: bool = True,
) -> LagunaDecodeGraphKey:
    reason = laguna_decode_graph_ineligibility(
        session,
        require_supported_backend=require_supported_backend,
    )
    if reason is not None:
        raise RuntimeError(f"Laguna decode graph is ineligible: {reason}")
    config = session.weights.config
    roles = _weight_role_guard(session)
    pointers = _static_buffer_ptrs(session)
    payload = {
        "schema_version": 1,
        "backend": str(session.backend),
        "target_arch": str(
            getattr(session, "target_arch", hip_target_arch_for_backend(session.backend))
        ).split(":", 1)[0],
        "active_rows": 1,
        "context_length": int(session.context_length),
        "sliding_window": int(config.sliding_window),
        "hidden_size": int(config.hidden_size),
        "vocab_size": int(config.vocab_size),
        "layer_types": tuple(str(value) for value in config.layer_types),
        "head_counts": tuple(int(value) for value in config.head_counts),
        "kv_storage_dtype": "bf16",
        "sampler_mode": "raw_greedy_top1_f32_i64",
        "selected_down_mode": str(session.selected_down_mode),
        "swa_decode_variant": str(session.swa_decode_variant),
        "swa_prefill_variant": str(session.swa_prefill_variant),
        "role_sha256": _sha256_json(roles),
        "buffer_identity_sha256": _sha256_json(pointers),
        "buffer_count": len(pointers),
    }
    return LagunaDecodeGraphKey(**payload, key_sha256=_sha256_json(payload))


def laguna_decode_graph_ineligibility(
    session: Any,
    *,
    captures: Any | None = None,
    stream: int = 0,
    processed_logits: bool = False,
    diagnostic_full_logits: bool = False,
    input_token_id: int | None = None,
    require_supported_backend: bool = True,
) -> str | None:
    if bool(getattr(session, "_closed", False)):
        return "session_closed"
    for name in ("weights", "scratch", "moe_scratch", "kv_cache", "full_rope", "swa_rope"):
        if getattr(session, name, None) is None:
            return f"missing_{name}"
    if captures is not None:
        return "hidden_captures"
    if int(stream):
        return "user_stream"
    if processed_logits:
        return "processed_logits"
    if diagnostic_full_logits:
        return "diagnostic_full_logits"
    if input_token_id is not None:
        prior = getattr(session, "last_result", None)
        if prior is None:
            return "missing_device_token_owner"
        if int(prior.next_token_id) != int(input_token_id):
            return "device_token_mismatch"
    backend = str(session.backend)
    if require_supported_backend and not bool(
        backend_package_capability(backend, "LAGUNA_DECODE_GRAPH_SUPPORTED", False)
    ):
        return "backend_not_certified"
    target_arch = str(
        getattr(session, "target_arch", hip_target_arch_for_backend(backend))
    ).split(":", 1)[0]
    if target_arch != "gfx1100" and require_supported_backend:
        return "backend_not_certified"
    if int(session.context_length) <= 0 or int(session.context_length) > 4096:
        return "context_capacity"
    if getattr(session, "_staged_verifier_tokens", None) is not None:
        return "staged_verifier"
    kv_cache = session.kv_cache
    if tuple(kv_cache.pending_positions):
        return "pending_kv_rows"
    if not bool(getattr(kv_cache, "graph_compatible", False)):
        return "manual_kv_mutation"
    if int(session.position) != int(kv_cache.position):
        return "cursor_mismatch"
    if int(session.position) < 0:
        return "state_not_prefilled"
    if int(session.position) + 1 >= int(session.context_length):
        return "context_exhausted"
    return None


def capture_laguna_decode_graph(session: Any) -> "LagunaDecodeGraph":
    reason = laguna_decode_graph_ineligibility(session)
    if reason is not None:
        raise RuntimeError(f"Laguna decode graph is ineligible: {reason}")
    key = build_laguna_decode_graph_key(session)
    existing = getattr(session, "_decode_graph", None)
    if existing is not None and not existing.closed:
        if existing.key == key:
            return existing
        existing.close()

    runtime = session.runtime
    graph = 0
    graph_exec = 0
    stream = 0
    capture_started = time.perf_counter()
    try:
        stream = runtime.stream_create()
        runtime.stream_begin_capture(stream)
        try:
            session._enqueue_decode_graph_step(stream=stream)
            graph = runtime.stream_end_capture(stream)
        except BaseException:
            try:
                abandoned_graph = runtime.stream_end_capture(stream)
                if abandoned_graph:
                    graph = abandoned_graph
            except BaseException:
                pass
            raise
        if not graph:
            raise RuntimeError("HIP returned a null Laguna decode graph")
        graph_exec = runtime.graph_instantiate(graph)
        if not graph_exec:
            raise RuntimeError("HIP returned a null Laguna decode graph executable")
    except BaseException:
        for handle, destroy_name in (
            (graph_exec, "graph_exec_destroy"),
            (graph, "graph_destroy"),
            (stream, "stream_destroy"),
        ):
            if not handle:
                continue
            try:
                getattr(runtime, destroy_name)(handle)
            except BaseException:
                pass
        raise
    handle = LagunaDecodeGraph(
        session=session,
        graph=graph,
        graph_exec=graph_exec,
        stream=stream,
        key=key,
        capture_seconds=time.perf_counter() - capture_started,
        static_guard=_static_replay_guard(session),
    )
    session._decode_graph = handle
    return handle


@dataclass
class LagunaDecodeGraph:
    session: Any
    graph: int
    graph_exec: int
    stream: int
    key: LagunaDecodeGraphKey | Any
    capture_seconds: float
    static_guard: tuple[Any, ...] = ()
    replay_count: int = 0
    primed_position: int | None = None
    primed_kv_epoch: int | None = None
    closed: bool = False

    def replay(self, input_token_id: int) -> Any:
        if self.closed:
            raise RuntimeError("Laguna decode graph is closed")
        reason = laguna_decode_graph_ineligibility(self.session)
        if reason is not None:
            raise RuntimeError(f"Laguna decode graph replay is ineligible: {reason}")
        if self.static_guard and _static_replay_guard(self.session) != self.static_guard:
            raise RuntimeError("Laguna decode graph static identity changed")
        token = int(input_token_id)
        config = self.session.weights.config
        if token < 0 or token >= int(config.vocab_size):
            raise ValueError("input_token_id is outside the Laguna vocabulary")
        prior = getattr(self.session, "last_result", None)
        if prior is not None and int(prior.next_token_id) != token:
            raise ValueError("graph input token does not match the device top-1 owner")
        ticket = build_laguna_decode_replay_ticket(self.session, self)
        start_position = ticket.session_position
        kv_cache = self.session.kv_cache
        assert self.session.scratch is not None
        if not ticket.primed:
            self._prime(ticket.next_position)
        try:
            self.session.runtime.graph_launch(self.graph_exec, self.stream)
            self.session.runtime.stream_synchronize(self.stream)
            from hipengine.runtime.laguna_gguf_runner import (
                LagunaEagerTokenResult,
                _read_f32,
                _read_i64,
            )

            next_id = _read_i64(self.session.scratch.argmax_id, self.session.runtime)
            next_value = _read_f32(self.session.scratch.argmax_value, self.session.runtime)
            committed_position = start_position + 1
            kv_cache.commit_graph_position(committed_position)
            self.session.position = committed_position
            result = LagunaEagerTokenResult(
                position=committed_position,
                input_token_id=token,
                next_token_id=next_id,
                next_token_logit=next_value,
                logits=self.session.scratch.logits,
                final_hidden=self.session.scratch.final_norm,
                post_layer_hidden=self.session.scratch.hidden,
            )
            self.session.last_result = result
            self.replay_count += 1
            self.primed_position = committed_position
            self.primed_kv_epoch = int(kv_cache.graph_state_epoch)
            return result
        except BaseException:
            self.session._close(suppress_errors=True)
            raise

    def _prime(self, next_position: int) -> None:
        from hipengine.kernels.hip_gfx1100.runtime import set_i64_scalar

        assert self.session.scratch is not None
        assert self.session.kv_cache is not None
        assert self.session.libraries is not None
        set_i64_scalar(
            self.session.scratch.position.ptr,
            int(next_position),
            stream=self.stream,
            library=self.session.libraries.runtime_state,
            runtime=self.session.runtime,
        )
        set_i64_scalar(
            self.session.kv_cache.row_position.ptr,
            int(next_position),
            stream=self.stream,
            library=self.session.libraries.runtime_state,
            runtime=self.session.runtime,
        )
        self.session.runtime.stream_synchronize(self.stream)
        self.primed_position = int(self.session.position)
        self.primed_kv_epoch = int(self.session.kv_cache.graph_state_epoch)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        errors: list[BaseException] = []
        for handle, destroy in (
            (self.graph_exec, self.session.runtime.graph_exec_destroy),
            (self.graph, self.session.runtime.graph_destroy),
            (self.stream, self.session.runtime.stream_destroy),
        ):
            if not handle:
                continue
            try:
                destroy(handle)
            except BaseException as exc:  # best-effort teardown after HIP failures
                errors.append(exc)
        self.graph_exec = 0
        self.graph = 0
        self.stream = 0
        if getattr(self.session, "_decode_graph", None) is self:
            self.session._decode_graph = None
        if errors:
            raise RuntimeError("one or more Laguna graph resources failed to free") from errors[0]

    def __enter__(self) -> "LagunaDecodeGraph":
        if self.closed:
            raise RuntimeError("Laguna decode graph is closed")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "LagunaDecodeGraph",
    "LagunaDecodeGraphKey",
    "LagunaDecodeReplayTicket",
    "build_laguna_decode_graph_key",
    "build_laguna_decode_replay_ticket",
    "capture_laguna_decode_graph",
    "laguna_decode_graph_ineligibility",
]
