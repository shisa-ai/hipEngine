"""State-bound whole-step GGUF decode graph capture.

The graph is session-local: every kernel argument is bound to resident weight,
scratch, recurrent-state, KV, sampler, and token buffers.  The key records that
full identity plus the starting state generation and declared transition
window.  Callers must never reuse a graph after reset, rollback, or a cursor
change outside that window.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.kvcache import KV_STORAGE_TAIL4_HADAMARD_GROUP32
from hipengine.kernels.hip_gfx1100.runtime import (
    advance_decode_position_i64,
    record_f32_row_indexed,
    record_i64_scalar_indexed,
)
from hipengine.runtime.gguf_linear import gemv_decode_session


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _decode_graph_kv_layout_admitted(session: Any) -> bool:
    storage = _enum_value(session.kv_storage_dtype)
    if storage == DType.BF16.value:
        return True
    return bool(
        storage == DType.INT8_PER_TOKEN_HEAD.value
        and str(getattr(session, "kv_storage_layout", "uniform"))
        == KV_STORAGE_TAIL4_HADAMARD_GROUP32
        and str(getattr(session, "kv_scale_granularity", "")) == "hadamard_group32"
    )


def _buffer_ptr(buffer: Any) -> int | None:
    if buffer is None:
        return None
    ptr = getattr(buffer, "ptr", None)
    return None if ptr is None else int(ptr)


def _session_buffer_ptrs(session: Any) -> tuple[int, ...]:
    scratch = session.scratch
    buffers = [
        session._hidden_a,
        session._hidden_b,
        session._token_buf,
        session._logits_buf,
        session._lm_block_values,
        session._lm_block_indices,
        session._lm_out_index,
        session._lm_out_value,
        scratch.position_buf,
        scratch.context_buf,
        scratch.hidden_seed_fp32,
        scratch.norm,
        scratch.attn_out,
        *scratch.layer_conv_states,
        *scratch.layer_recurrent_states,
        *scratch.full_key_caches,
        *scratch.full_value_caches,
        *getattr(scratch, "full_k_scale_caches", ()),
        *getattr(scratch, "full_v_scale_caches", ()),
    ]
    pointers = [ptr for buffer in buffers if (ptr := _buffer_ptr(buffer)) is not None]
    for weight in session.runner.weights.weights:
        allocations = getattr(weight, "allocations", None)
        if allocations is not None:
            for allocation in allocations.values():
                ptr = _buffer_ptr(getattr(allocation, "tensor", None))
                if ptr is not None:
                    pointers.append(ptr)
            continue
        allocation = getattr(weight, "allocation", None)
        if callable(allocation):
            ptr = _buffer_ptr(getattr(allocation(), "tensor", None))
            if ptr is not None:
                pointers.append(ptr)
    return tuple(pointers)


def _weight_roles(session: Any) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "slot_path": str(weight.spec.slot_path),
            "quant_key": str(weight.spec.quant_key),
            "shape": [int(dim) for dim in weight.spec.source.shape],
        }
        for weight in session.runner.weights.weights
    )


def _model_identity(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path).expanduser()
    try:
        resolved = path.resolve()
        stat = resolved.stat()
        return {
            "path": str(resolved),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except OSError:
        return {"path": str(path), "size_bytes": None, "mtime_ns": None}


@dataclass(frozen=True)
class Qwen35GGUFDecodeGraphKey:
    schema_version: int
    backend: str
    target_arch: str
    model_identity_sha256: str
    active_rows: int
    state_generation: int
    start_position: int
    replay_context_limit: int
    context_bucket: int
    block_size: int
    max_positions: int
    steps_per_replay: int
    max_replay_steps: int
    record_steps: int
    record_hidden_seeds: bool
    capture_hidden_seed_fp32: bool
    hidden_size: int
    vocab_size: int
    layer_types: tuple[str, ...]
    kv_storage_dtype: str
    kv_storage_layout: str
    kv_scale_dtype: str
    kv_scale_granularity: str
    use_wmma_prefill: bool
    use_gemv_decode: bool
    decode_repack: bool
    host_token_embedding: bool
    lm_head_threads: int
    lm_head_stage1_blocks: int
    weight_role_sha256: str
    buffer_identity_sha256: str
    buffer_count: int
    key_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))


def build_qwen35_gguf_decode_graph_key(
    session: Any,
    *,
    position: int,
    steps_per_replay: int,
    max_replay_steps: int,
    attention_max_context_len: int | None = None,
    record_steps: int = 0,
    record_hidden_seeds: bool = False,
    capture_hidden_seed_fp32: bool = False,
    extra_buffer_ptrs: tuple[int, ...] = (),
) -> Qwen35GGUFDecodeGraphKey:
    if session.runner is None or session.runner.weights is None or session.scratch is None:
        raise RuntimeError("GGUF resident session is closed")
    position = int(position)
    steps_per_replay = int(steps_per_replay)
    max_replay_steps = int(max_replay_steps)
    if position < 0:
        raise ValueError("position must be non-negative")
    if int(session.position) != position:
        raise ValueError(
            f"decode graph state generation {session.position} does not match position {position}"
        )
    if steps_per_replay <= 0 or max_replay_steps <= 0:
        raise ValueError("decode graph replay sizes must be positive")
    if steps_per_replay > max_replay_steps:
        raise ValueError("steps_per_replay cannot exceed max_replay_steps")
    if record_steps < 0:
        raise ValueError("record_steps must be non-negative")
    if record_steps and record_steps < int(steps_per_replay):
        raise ValueError("record_steps must cover at least one replay launch")
    block_size = int(session.scratch.block_size)
    max_positions = int(session.scratch.max_positions)
    replay_context_limit = int(
        position + max_replay_steps
        if attention_max_context_len is None
        else attention_max_context_len
    )
    if replay_context_limit < position + max_replay_steps:
        raise ValueError("attention_max_context_len must cover the replay transition window")
    context_bucket = ((replay_context_limit + block_size - 1) // block_size) * block_size
    if context_bucket > max_positions:
        raise ValueError("decode graph transition window exceeds resident cache capacity")
    roles = _weight_roles(session)
    pointers = (*_session_buffer_ptrs(session), *(int(ptr) for ptr in extra_buffer_ptrs))
    quant_keys = [str(row["quant_key"]) for row in roles]
    payload = {
        "schema_version": 1,
        "backend": str(session.backend),
        "target_arch": str(session.runner.target_arch),
        "model_identity_sha256": _sha256_json(_model_identity(session.model_path)),
        "active_rows": 1,
        "state_generation": position,
        "start_position": position,
        "replay_context_limit": replay_context_limit,
        "context_bucket": context_bucket,
        "block_size": block_size,
        "max_positions": max_positions,
        "steps_per_replay": steps_per_replay,
        "max_replay_steps": max_replay_steps,
        "record_steps": int(record_steps),
        "record_hidden_seeds": bool(record_hidden_seeds),
        "capture_hidden_seed_fp32": bool(capture_hidden_seed_fp32),
        "hidden_size": int(session.runner.hidden_size),
        "vocab_size": int(session.runner.vocab_size),
        "layer_types": tuple(str(layer) for layer in session.runner.weights.config.layer_types),
        "kv_storage_dtype": _enum_value(session.kv_storage_dtype),
        "kv_storage_layout": str(getattr(session, "kv_storage_layout", "uniform")),
        "kv_scale_dtype": _enum_value(session.kv_scale_dtype),
        "kv_scale_granularity": str(session.kv_scale_granularity),
        "use_wmma_prefill": bool(session.use_wmma_prefill),
        "use_gemv_decode": bool(session.use_gemv_decode),
        "decode_repack": any(key.endswith("_t16_v1") or key.endswith("_x8_v1") for key in quant_keys),
        "host_token_embedding": bool(session.host_token_embedding_enabled),
        "lm_head_threads": int(session._lm_head_threads),
        "lm_head_stage1_blocks": int(session._lm_head_stage1_blocks),
        "weight_role_sha256": _sha256_json(roles),
        "buffer_identity_sha256": _sha256_json(pointers),
        "buffer_count": len(pointers),
    }
    return Qwen35GGUFDecodeGraphKey(
        **payload,
        key_sha256=_sha256_json(payload),
    )


def _enqueue_decode_step(
    session: Any,
    *,
    position: int,
    stream: int,
    replay_context_limit: int,
    record_output_ptr: int | None,
    record_hidden_seed_ptr: int | None,
    record_index_ptr: int | None,
    record_capacity: int,
    capture_hidden_seed_fp32: bool,
) -> None:
    if session._lm_out_index is None or session.scratch is None:
        raise RuntimeError("GGUF resident session is closed")
    session.scratch.position_host[0] = int(position)
    session.scratch.context_host[0] = int(position) + 1
    session._set_token_embedding_from_ptr(session._lm_out_index.ptr, stream=stream)
    hidden_ptr = session._run_current_hidden_to_final_hidden(
        position=int(position),
        stream=stream,
        attention_max_context_len=int(replay_context_limit),
        capture_hidden_seed_fp32=bool(capture_hidden_seed_fp32),
    )
    if record_hidden_seed_ptr is not None:
        if record_index_ptr is None or not capture_hidden_seed_fp32:
            raise ValueError("hidden-seed recording requires an index and FP32 hidden capture")
        record_f32_row_indexed(
            session.scratch.hidden_seed_fp32.ptr,
            record_hidden_seed_ptr,
            record_index_ptr,
            session.runner.hidden_size,
            int(record_capacity),
            stream=stream,
            library=session._runtime_state_library,
            runtime=session.runtime,
        )
    session._sample_device_from_hidden(hidden_ptr, stream=stream)
    if record_output_ptr is not None:
        if record_index_ptr is None:
            raise ValueError("generated-token recording requires an index")
        record_i64_scalar_indexed(
            session._lm_out_index.ptr,
            record_output_ptr,
            record_index_ptr,
            int(record_capacity),
            stream=stream,
            library=session._runtime_state_library,
            runtime=session.runtime,
        )
    advance_decode_position_i64(
        session.scratch.position_buf.ptr,
        session.scratch.context_buf.ptr,
        stream=stream,
        library=session._runtime_state_library,
        runtime=session.runtime,
    )


def capture_qwen35_gguf_decode_graph(
    session: Any,
    *,
    position: int,
    steps_per_replay: int = 1,
    max_replay_steps: int | None = None,
    record_steps: int = 0,
    attention_max_context_len: int | None = None,
    capture_hidden_seed_fp32: bool = False,
    record_hidden_seeds: bool = False,
) -> "Qwen35GGUFDecodeGraph":
    if session.runner is None or session.scratch is None:
        raise RuntimeError("GGUF resident session is closed")
    if session.host_token_embedding_enabled:
        raise RuntimeError("GGUF decode graph requires device-resident token embedding")
    if not _decode_graph_kv_layout_admitted(session):
        raise RuntimeError(
            "GGUF decode graph requires BF16 KV or the admitted tail4_hadamard_group32 layout"
        )
    if steps_per_replay <= 0:
        raise ValueError("steps_per_replay must be positive")
    replay_span = int(max_replay_steps if max_replay_steps is not None else steps_per_replay)
    if replay_span <= 0:
        raise ValueError("max_replay_steps must be positive")
    if record_steps < 0:
        raise ValueError("record_steps must be non-negative")
    if record_hidden_seeds and record_steps <= 0:
        raise ValueError("record_hidden_seeds requires record_steps > 0")
    replay_context_limit = int(position) + replay_span
    if attention_max_context_len is not None:
        replay_context_limit = int(attention_max_context_len)
        if replay_context_limit < int(position) + replay_span:
            raise ValueError("attention_max_context_len must cover the full replay transition window")
    if replay_context_limit > int(session.scratch.max_positions):
        raise ValueError("attention_max_context_len exceeds resident cache capacity")
    runtime: HipRuntime = session.runtime or get_hip_runtime()
    generated: DeviceBuffer | None = None
    generated_hidden_seeds: DeviceBuffer | None = None
    generated_index: DeviceBuffer | None = None
    if record_steps:
        generated = malloc(int(record_steps) * DType.INT64.itemsize, runtime=runtime)
        generated_index = malloc(DType.INT64.itemsize, runtime=runtime)
        runtime.memset(generated.ptr, 0xFF, generated.nbytes)
        if record_hidden_seeds:
            generated_hidden_seeds = malloc(
                int(record_steps) * session.runner.hidden_size * DType.FP32.itemsize,
                runtime=runtime,
            )
            runtime.memset(generated_hidden_seeds.ptr, 0, generated_hidden_seeds.nbytes)
        zero = np.zeros((1,), dtype=np.int64)
        copy_host_to_device(generated_index, host_array_ptr(zero), runtime=runtime)

    graph = 0
    stream = 0
    try:
        key = build_qwen35_gguf_decode_graph_key(
            session,
            position=int(position),
            steps_per_replay=int(steps_per_replay),
            max_replay_steps=replay_span,
            attention_max_context_len=replay_context_limit,
            record_steps=int(record_steps),
            record_hidden_seeds=bool(record_hidden_seeds),
            capture_hidden_seed_fp32=bool(capture_hidden_seed_fp32 or record_hidden_seeds),
            extra_buffer_ptrs=tuple(
                buffer.ptr
                for buffer in (generated, generated_hidden_seeds, generated_index)
                if buffer is not None
            ),
        )
        stream = runtime.stream_create()
        session._set_full_attention_position_device(int(position), stream=stream)
        runtime.stream_synchronize(stream)
        runtime.stream_begin_capture(stream)
        try:
            with gemv_decode_session(session.use_gemv_decode):
                for offset in range(int(steps_per_replay)):
                    _enqueue_decode_step(
                        session,
                        position=int(position) + offset,
                        stream=stream,
                        replay_context_limit=replay_context_limit,
                        record_output_ptr=None if generated is None else generated.ptr,
                        record_hidden_seed_ptr=(
                            None if generated_hidden_seeds is None else generated_hidden_seeds.ptr
                        ),
                        record_index_ptr=None if generated_index is None else generated_index.ptr,
                        record_capacity=int(record_steps),
                        capture_hidden_seed_fp32=bool(capture_hidden_seed_fp32 or record_hidden_seeds),
                    )
            graph = runtime.stream_end_capture(stream)
        except Exception:
            try:
                runtime.stream_end_capture(stream)
            except Exception:
                pass
            raise
        graph_exec = runtime.graph_instantiate(graph)
    except Exception:
        if graph:
            try:
                runtime.graph_destroy(graph)
            except Exception:
                pass
        if stream:
            runtime.stream_destroy(stream)
        for buffer in (generated_index, generated_hidden_seeds, generated):
            if buffer is not None:
                free(buffer, runtime=runtime)
        raise
    handle = Qwen35GGUFDecodeGraph(
        session=session,
        graph=graph,
        graph_exec=graph_exec,
        stream=stream,
        position=int(position),
        steps_per_replay=int(steps_per_replay),
        max_replay_steps=replay_span,
        generated=generated,
        generated_hidden_seeds=generated_hidden_seeds,
        generated_index=generated_index,
        record_steps=int(record_steps),
        bucket_key=key,
        attention_max_context_len=replay_context_limit,
        capture_hidden_seed_fp32=bool(capture_hidden_seed_fp32 or record_hidden_seeds),
    )
    session._decode_graphs.append(handle)
    return handle


@dataclass
class Qwen35GGUFDecodeGraph:
    session: Any
    graph: int
    graph_exec: int
    stream: int
    position: int
    steps_per_replay: int
    max_replay_steps: int
    generated: DeviceBuffer | None
    generated_hidden_seeds: DeviceBuffer | None
    generated_index: DeviceBuffer | None
    record_steps: int
    bucket_key: Qwen35GGUFDecodeGraphKey
    attention_max_context_len: int
    capture_hidden_seed_fp32: bool
    replayed_steps: int = 0
    closed: bool = False

    def replay(self, steps: int) -> None:
        if self.closed:
            raise RuntimeError("GGUF decode graph is closed")
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if steps % self.steps_per_replay:
            raise ValueError("steps must be divisible by steps_per_replay")
        if self.replayed_steps + steps > self.max_replay_steps:
            raise ValueError("cumulative graph replay exceeds the declared transition window")
        if self.record_steps and self.replayed_steps + steps > self.record_steps:
            raise ValueError("cumulative graph replay exceeds record capacity")
        expected_position = self.position + self.replayed_steps
        if int(self.session.position) != expected_position:
            raise RuntimeError(
                f"decode graph state generation mismatch: expected {expected_position}, "
                f"observed {self.session.position}"
            )
        launches = steps // self.steps_per_replay
        for _ in range(launches):
            self.session.runtime.graph_launch(self.graph_exec, self.stream)
            self.session.runtime.stream_synchronize(self.stream)
            self.replayed_steps += self.steps_per_replay
            self.session._position = self.position + self.replayed_steps
            if self.session.scratch is not None:
                self.session.scratch.position_host[0] = self.session._position
                self.session.scratch.context_host[0] = self.session._position + 1
        if self.capture_hidden_seed_fp32:
            self.session._hidden_seed_fp32_populated = True

    def read_sample(self, *, return_logits: bool = True) -> Any:
        if self.closed:
            raise RuntimeError("GGUF decode graph is closed")
        return self.session._read_sample(return_logits=return_logits)

    def read_generated_token_ids(self, count: int | None = None) -> list[int]:
        if self.closed:
            raise RuntimeError("GGUF decode graph is closed")
        if self.generated is None:
            raise RuntimeError("GGUF decode graph was captured without token recording")
        rows = int(self.record_steps if count is None else count)
        if rows < 0 or rows > self.record_steps:
            raise ValueError("count outside decode graph record capacity")
        host = np.empty((rows,), dtype=np.int64)
        copy_device_to_host(
            host_array_ptr(host),
            DeviceBuffer(self.generated.ptr, rows * DType.INT64.itemsize),
            runtime=self.session.runtime,
        )
        return [int(token) for token in host.tolist()]

    def read_generated_hidden_seeds(self, start: int = 0, count: int | None = None) -> np.ndarray:
        if self.closed:
            raise RuntimeError("GGUF decode graph is closed")
        if self.generated_hidden_seeds is None:
            raise RuntimeError("GGUF decode graph was captured without hidden-seed recording")
        start = int(start)
        rows = int(self.record_steps - start if count is None else count)
        if start < 0 or rows < 0 or start + rows > self.record_steps:
            raise ValueError("hidden-seed slice outside decode graph record capacity")
        hidden_size = int(self.session.runner.hidden_size)
        host = np.empty((rows, hidden_size), dtype=np.float32)
        row_nbytes = hidden_size * DType.FP32.itemsize
        copy_device_to_host(
            host_array_ptr(host),
            DeviceBuffer(self.generated_hidden_seeds.ptr + start * row_nbytes, rows * row_nbytes),
            runtime=self.session.runtime,
        )
        return host

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        runtime = self.session.runtime or get_hip_runtime()
        runtime.graph_exec_destroy(self.graph_exec)
        runtime.graph_destroy(self.graph)
        if self.stream:
            runtime.stream_destroy(self.stream)
        for name in ("generated_index", "generated_hidden_seeds", "generated"):
            buffer = getattr(self, name)
            if buffer is not None:
                free(buffer, runtime=runtime)
                setattr(self, name, None)
        graphs = getattr(self.session, "_decode_graphs", None)
        if isinstance(graphs, list) and self in graphs:
            graphs.remove(self)

    def __enter__(self) -> "Qwen35GGUFDecodeGraph":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "Qwen35GGUFDecodeGraph",
    "Qwen35GGUFDecodeGraphKey",
    "build_qwen35_gguf_decode_graph_key",
    "capture_qwen35_gguf_decode_graph",
]
