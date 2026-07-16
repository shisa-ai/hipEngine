"""State-bound packed GGUF decode graph capture and replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

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
from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.registry import resolve
from hipengine.quant.gguf import bf16_to_float32
from hipengine.runtime.gguf_packed_manifest import build_packed_decode_execution_manifest


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


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


def _weight_roles(owner: Any) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "slot_path": str(weight.spec.slot_path),
            "quant_key": str(weight.spec.quant_key),
            "shape": [int(dim) for dim in weight.spec.source.shape],
        }
        for weight in owner.runner.weights.weights
    )


@dataclass(frozen=True)
class Qwen35GGUFPackedDecodeGraphKey:
    schema_version: int
    backend: str
    target_arch: str
    model_identity_sha256: str
    physical_rows: int
    active_rows: int
    active_mask: tuple[bool, ...]
    state_generations: tuple[int, ...]
    replay_context_limit: int
    context_bucket: int
    block_size: int
    max_positions: int
    steps_per_replay: int
    max_replay_steps: int
    record_steps: int
    record_layer_ids: tuple[int, ...]
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
    metadata_prepare_path: str
    token_feedback_path: str
    weight_role_sha256: str
    buffer_identity_sha256: str
    buffer_count: int
    key_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))


def build_qwen35_gguf_packed_decode_graph_key(
    owner: Any,
    *,
    sessions: Sequence[Any],
    active_mask: Sequence[bool],
    block_size: int,
    max_positions: int,
    steps_per_replay: int,
    max_replay_steps: int,
    record_steps: int,
    record_layer_ids: Sequence[int],
    packed_buffer_ptrs: Sequence[int],
) -> Qwen35GGUFPackedDecodeGraphKey:
    """Build a complete fixed-shape/state key for one packed graph window."""

    session_tuple = tuple(sessions)
    positions = tuple(int(session.position) for session in session_tuple)
    mask = tuple(bool(value) for value in active_mask)
    rows = len(session_tuple)
    if rows <= 0 or rows > 4:
        raise ValueError("packed decode graphs require between one and four rows")
    if len(mask) != rows:
        raise ValueError("active_mask length must equal physical rows")
    if not any(mask):
        raise ValueError("active_mask must contain at least one active row")
    if min(positions) < 0:
        raise ValueError("session positions must be non-negative")
    step_width = int(steps_per_replay)
    replay_span = int(max_replay_steps)
    if step_width <= 0 or replay_span <= 0:
        raise ValueError("decode graph replay sizes must be positive")
    if step_width > replay_span:
        raise ValueError("steps_per_replay cannot exceed max_replay_steps")
    record_capacity = int(record_steps)
    layer_ids = tuple(int(layer_id) for layer_id in record_layer_ids)
    if record_capacity < 0:
        raise ValueError("record_steps must be non-negative")
    if layer_ids and record_capacity <= 0:
        raise ValueError("record_layer_ids require record_steps")
    if len(set(layer_ids)) != len(layer_ids) or any(layer_id < 0 for layer_id in layer_ids):
        raise ValueError("record_layer_ids must be unique non-negative ids")
    block = int(block_size)
    capacity = int(max_positions)
    if block <= 0 or capacity <= 0:
        raise ValueError("block_size and max_positions must be positive")
    replay_context_limit = max(positions) + replay_span
    context_bucket = ((replay_context_limit + block - 1) // block) * block
    if context_bucket > capacity:
        raise ValueError("packed decode graph transition window exceeds resident capacity")
    pointers = tuple(int(ptr) for ptr in packed_buffer_ptrs)
    if not pointers or any(ptr <= 0 for ptr in pointers):
        raise ValueError("packed_buffer_ptrs must contain positive device pointers")
    roles = _weight_roles(owner)
    quant_keys = tuple(str(row["quant_key"]) for row in roles)
    payload = {
        "schema_version": 1,
        "backend": str(owner.backend),
        "target_arch": str(owner.runner.target_arch),
        "model_identity_sha256": _sha256_json(_model_identity(owner.model_path)),
        "physical_rows": rows,
        "active_rows": sum(mask),
        "active_mask": mask,
        "state_generations": positions,
        "replay_context_limit": replay_context_limit,
        "context_bucket": context_bucket,
        "block_size": block,
        "max_positions": capacity,
        "steps_per_replay": step_width,
        "max_replay_steps": replay_span,
        "record_steps": record_capacity,
        "record_layer_ids": layer_ids,
        "hidden_size": int(owner.runner.hidden_size),
        "vocab_size": int(owner.runner.vocab_size),
        "layer_types": tuple(str(layer) for layer in owner.runner.weights.config.layer_types),
        "kv_storage_dtype": _enum_value(owner.kv_storage_dtype),
        "kv_storage_layout": str(owner.kv_storage_layout),
        "kv_scale_dtype": _enum_value(owner.kv_scale_dtype),
        "kv_scale_granularity": str(owner.kv_scale_granularity),
        "use_wmma_prefill": bool(owner.use_wmma_prefill),
        "use_gemv_decode": bool(owner.use_gemv_decode),
        "decode_repack": any(key.endswith("_t16_v1") or key.endswith("_x8_v1") for key in quant_keys),
        "host_token_embedding": bool(owner.host_token_embedding_enabled),
        "lm_head_threads": int(owner._lm_head_threads),
        "lm_head_stage1_blocks": int(owner._lm_head_stage1_blocks),
        "metadata_prepare_path": "device_positions_persistent",
        "token_feedback_path": "device_i32_to_i64",
        "weight_role_sha256": _sha256_json(roles),
        "buffer_identity_sha256": _sha256_json(pointers),
        "buffer_count": len(pointers),
    }
    return Qwen35GGUFPackedDecodeGraphKey(
        **payload,
        key_sha256=_sha256_json(payload),
    )


def _buffer_ptr(buffer: Any) -> int | None:
    ptr = getattr(buffer, "ptr", None)
    return None if ptr is None else int(ptr)


def _packed_graph_buffer_ptrs(
    owner: Any,
    packed_state: Any,
    packed_scratch: Any,
    *extra_buffers: DeviceBuffer | None,
) -> tuple[int, ...]:
    buffers = (
        owner._prefill_token_buf,
        owner._prefill_hidden_a,
        owner._prefill_hidden_b,
        owner._verify_logits_buf,
        owner._verify_lm_block_values,
        owner._verify_lm_block_indices_i32,
        owner._verify_lm_out_indices_i32,
        owner._verify_lm_out_values,
        *packed_state.buffers,
        *packed_scratch.buffers,
        *extra_buffers,
    )
    pointers = [
        ptr for buffer in buffers if (ptr := _buffer_ptr(buffer)) is not None
    ]
    for weight in owner.runner.weights.weights:
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


def _resolve_packed_graph_kernels(owner: Any) -> tuple[Any, Any, Any]:
    backend = str(owner.runner.backend)
    load_backend_kernel_package(backend)
    specs = (
        ("decode_metadata", "packed_c4_device_positions_i64"),
        ("decode_graph_commit", "packed_c4_i32_i64"),
        ("decode_graph_record", "packed_u16_rows_indexed"),
    )
    resolved = tuple(
        resolve(
            backend=backend,
            layer=layer,
            quant="gguf_qwen35",
            variant=variant,
            missing="none",
        )
        for layer, variant in specs
    )
    if any(kernel is None for kernel in resolved):
        missing = [
            f"{layer}/gguf_qwen35/{variant}"
            for (layer, variant), kernel in zip(specs, resolved, strict=True)
            if kernel is None
        ]
        raise NotImplementedError(
            f"backend {backend!r} does not provide packed decode graph kernels: {missing!r}"
        )
    return resolved  # type: ignore[return-value]


def _singleton_layout(token_ids: Sequence[int], positions: Sequence[int], *, capacity: int):
    from hipengine.runtime.qwen35_gguf_runner import (
        _GGUFPackedVerifySlotBlock,
        _build_gguf_packed_verify_layout,
    )

    blocks = tuple(
        _GGUFPackedVerifySlotBlock(
            input_token_ids=(int(token),),
            start_position=int(position),
        )
        for token, position in zip(token_ids, positions, strict=True)
    )
    return _build_gguf_packed_verify_layout(blocks, slot_capacity=int(capacity))


def _final_transition_layout(graph: "Qwen35GGUFPackedDecodeGraph"):
    positions = tuple(
        int(start) + int(graph.replayed_steps) - 1
        for start in graph.bucket_key.state_generations
    )
    return _singleton_layout((0,) * len(positions), positions, capacity=graph.slot_capacity)


def capture_qwen35_gguf_packed_decode_graph(
    owner: Any,
    *,
    token_ids: Sequence[int],
    sessions: Sequence[Any],
    steps_per_replay: int = 1,
    max_replay_steps: int | None = None,
    record_steps: int = 0,
    record_layer_output_hidden: Sequence[int] = (),
) -> "Qwen35GGUFPackedDecodeGraph":
    """Capture one fixed-width packed GGUF decode bucket with device feedback."""

    token_tuple = tuple(int(token) for token in token_ids)
    session_tuple = tuple(sessions)
    rows = len(session_tuple)
    if rows <= 0 or rows > 4:
        raise ValueError("packed decode graphs require between one and four rows")
    if len(token_tuple) != rows:
        raise ValueError("token_ids and sessions must have the same length")
    if owner.runner is None or owner.runner.weights is None or owner.scratch is None:
        raise RuntimeError("GGUF resident session is closed")
    if owner._prefill_token_buf is None or owner._bulk_prefill_scratch is None:
        raise RuntimeError("GGUF resident packed decode buffers are closed")
    if owner.host_token_embedding_enabled:
        raise NotImplementedError("packed decode graphs require device token embedding")
    if _enum_value(owner.kv_storage_dtype) != DType.BF16.value:
        raise NotImplementedError("packed decode graphs currently support BF16 KV only")
    if owner.use_expert_sidecar:
        raise NotImplementedError("packed decode graphs do not support expert sidecars")
    if any(session.runner is not owner.runner for session in session_tuple):
        raise NotImplementedError("packed decode graphs require shared runner sessions")
    if any(session.scratch is None for session in session_tuple):
        raise RuntimeError("packed decode graph session is closed")
    if any(token < 0 or token >= int(owner.runner.vocab_size) for token in token_tuple):
        raise ValueError("token id outside resident vocabulary")
    step_width = int(steps_per_replay)
    replay_span = int(step_width if max_replay_steps is None else max_replay_steps)
    if step_width <= 0 or replay_span <= 0 or step_width > replay_span:
        raise ValueError("invalid packed decode graph replay window")
    positions = tuple(int(session.position) for session in session_tuple)
    if max(positions) + replay_span >= 1024:
        raise NotImplementedError("packed decode graphs currently require context < 1024")
    record_capacity = int(record_steps)
    if record_capacity < 0:
        raise ValueError("record_steps must be non-negative")
    if record_capacity and record_capacity < replay_span:
        raise ValueError("record_steps must cover the declared replay window")
    layer_ids = tuple(sorted(owner._normalize_layer_output_capture(record_layer_output_hidden)))
    if layer_ids and record_capacity <= 0:
        raise ValueError("layer-hidden recording requires record_steps")

    runtime: HipRuntime = owner.runtime or get_hip_runtime()
    slot_capacity = max(1024, max(positions) + replay_span + 1)
    layout = _singleton_layout(token_tuple, positions, capacity=slot_capacity)
    packed_state, packed_scratch_base = owner._ensure_packed_verify_workspace(
        slot_count=rows,
        rows=rows,
        max_sequence_length=slot_capacity,
        runtime=runtime,
    )
    graph = 0
    stream = runtime.stream_create()
    generated_tokens: DeviceBuffer | None = None
    generated_hidden: DeviceBuffer | None = None
    record_index: DeviceBuffer | None = None
    try:
        imported_slot_indices = owner._sync_packed_decode_initial_state(
            session_tuple,
            layout,
            packed_state,
            runtime=runtime,
            stream=stream,
        )
        packed_scratch = packed_scratch_base.for_packed_verify_layout(
            layout,
            runtime=runtime,
            stream=stream,
            metadata_prepare_fn=owner.runner._packed_decode_metadata_kernel(),
        )
        replay_context_limit = max(positions) + replay_span
        packed_scratch = replace(
            packed_scratch,
            append_spans=replace(
                packed_scratch.append_spans,
                max_live_count=replay_context_limit,
            ),
            prefill_spans=replace(
                packed_scratch.prefill_spans,
                max_live_count=replay_context_limit,
            ),
        )
        owner._ensure_verify_lm_head_buffers(rows, runtime=runtime)
        token_array = np.ascontiguousarray(token_tuple, dtype=np.int64)
        copy_host_to_device(
            owner._prefill_token_buf,
            host_array_ptr(token_array),
            token_array.nbytes,
            runtime=runtime,
        )
        if record_capacity:
            generated_tokens = malloc(
                record_capacity * rows * DType.INT32.itemsize,
                runtime=runtime,
            )
            record_index = malloc(DType.INT64.itemsize, runtime=runtime)
            runtime.memset(generated_tokens.ptr, 0xFF, generated_tokens.nbytes)
            zero = np.zeros((1,), dtype=np.int64)
            copy_host_to_device(
                record_index,
                host_array_ptr(zero),
                zero.nbytes,
                runtime=runtime,
            )
            if layer_ids:
                generated_hidden = malloc(
                    record_capacity
                    * len(layer_ids)
                    * rows
                    * int(owner.runner.hidden_size)
                    * DType.BF16.itemsize,
                    runtime=runtime,
                )
                runtime.memset(generated_hidden.ptr, 0, generated_hidden.nbytes)
        metadata_kernel, commit_kernel, record_hidden_kernel = _resolve_packed_graph_kernels(owner)
        linear_decode_scratch = replace(
            owner.scratch,
            layer_conv_states=packed_state.layer_conv_states,
            layer_recurrent_states=packed_state.layer_recurrent_states,
        )
        pointers = _packed_graph_buffer_ptrs(
            owner,
            packed_state,
            packed_scratch,
            generated_tokens,
            generated_hidden,
            record_index,
        )
        key = build_qwen35_gguf_packed_decode_graph_key(
            owner,
            sessions=session_tuple,
            active_mask=(True,) * rows,
            block_size=int(packed_scratch.block_size),
            max_positions=int(packed_scratch.max_positions),
            steps_per_replay=step_width,
            max_replay_steps=replay_span,
            record_steps=record_capacity,
            record_layer_ids=layer_ids,
            packed_buffer_ptrs=pointers,
        )
        runtime.stream_synchronize(stream)
        linear_paths: set[str] = set()
        full_paths: set[str] = set()
        runtime.stream_begin_capture(stream)
        try:
            for _ in range(step_width):
                metadata_kernel(
                    packed_scratch.block_table.ptr,
                    packed_scratch.positions.ptr,
                    packed_scratch.context_counts.ptr,
                    packed_scratch.cu_q.ptr,
                    packed_scratch.cu_k.ptr,
                    packed_scratch.atomic.ptr,
                    packed_scratch.gdn_cu_seqlens.ptr,
                    packed_scratch.gdn_state_indices.ptr,
                    rows,
                    int(layout.blocks_per_slot),
                    stream=stream,
                    library=owner._runtime_state_library,
                    runtime=runtime,
                )

                def record_layer(layer_id: int, hidden_ptr: int) -> None:
                    if generated_hidden is None or record_index is None:
                        return
                    try:
                        layer_slot = layer_ids.index(int(layer_id))
                    except ValueError:
                        return
                    elements = rows * int(owner.runner.hidden_size)
                    record_hidden_kernel(
                        hidden_ptr,
                        generated_hidden.ptr + layer_slot * elements * DType.BF16.itemsize,
                        record_index.ptr,
                        elements,
                        len(layer_ids) * elements,
                        record_capacity,
                        stream=stream,
                        library=owner._runtime_state_library,
                        runtime=runtime,
                    )

                step_linear, step_full = owner._enqueue_packed_decode_model_step(
                    rows=rows,
                    state_indices=tuple(range(rows)),
                    packed_scratch=packed_scratch,
                    packed_state=packed_state,
                    linear_decode_scratch=linear_decode_scratch,
                    stream=stream,
                    layer_output_callback=record_layer if layer_ids else None,
                )
                linear_paths.update(step_linear)
                full_paths.update(step_full)
                if owner._verify_lm_out_indices_i32 is None:
                    raise RuntimeError("packed sampler token buffer is closed")
                commit_kernel(
                    owner._verify_lm_out_indices_i32.ptr,
                    owner._prefill_token_buf.ptr,
                    packed_scratch.positions.ptr,
                    packed_scratch.context_counts.ptr,
                    rows,
                    recorded_token_ids_i32_ptr=(
                        None if generated_tokens is None else generated_tokens.ptr
                    ),
                    record_index_i64_ptr=None if record_index is None else record_index.ptr,
                    record_capacity=record_capacity,
                    stream=stream,
                    library=owner._runtime_state_library,
                    runtime=runtime,
                )
            graph = runtime.stream_end_capture(stream)
        except Exception:
            try:
                runtime.stream_end_capture(stream)
            except Exception:
                pass
            raise
        if len(linear_paths) > 1 or len(full_paths) > 1:
            raise RuntimeError("packed graph capture selected inconsistent layer routes")
        graph_exec = runtime.graph_instantiate(graph)
    except Exception:
        if graph:
            try:
                runtime.graph_destroy(graph)
            except Exception:
                pass
        if stream:
            runtime.stream_destroy(stream)
        for buffer in (record_index, generated_hidden, generated_tokens):
            if buffer is not None:
                free(buffer, runtime=runtime)
        raise

    linear_path = next(iter(linear_paths)) if linear_paths else "not_applicable"
    full_path = next(iter(full_paths)) if full_paths else "not_applicable"
    manifest = build_packed_decode_execution_manifest(
        rows=rows,
        layer_types=owner.runner.weights.config.layer_types,
        imported_slot_indices=(),
        import_positions=positions,
        scatter_state=False,
        blocks_per_slot=int(layout.blocks_per_slot),
        capture_layer_count=0,
        linear_attention_decode_path=linear_path,
        full_attention_decode_path=full_path,
        moe_decode_path=(
            "selected_rows_batch" if owner.runner.weights.config.is_moe else "dense_ffn_rows"
        ),
        moe_top_k=(
            int(owner.runner.weights.config.expert_used_count)
            if owner.runner.weights.config.is_moe
            else 0
        ),
        lm_head_decode_path=owner._last_packed_lm_head_decode_path,
        sampler_decode_path=owner._last_packed_sampler_decode_path,
        metadata_prepare_path="device_positions_persistent",
    )
    linear_layers = sum(
        layer_type == "linear_attention"
        for layer_type in owner.runner.weights.config.layer_types
    )
    full_layers = sum(
        layer_type == "full_attention"
        for layer_type in owner.runner.weights.config.layer_types
    )
    state_import_copies = sum(
        2 * linear_layers + (2 * full_layers if positions[index] > 0 else 0)
        for index in imported_slot_indices
    )
    manifest["mode"] = "decode_graph_replay"
    manifest["host_device_movement"] = {
        "host_to_device_metadata_copies": 0,
        "host_to_device_metadata_bytes": 0,
        "device_metadata_prepare_launches": 1,
        "device_token_feedback_launches": 1,
        "device_layer_record_launches": len(layer_ids),
        "host_to_device_input_copies": 0,
        "host_to_device_input_bytes": 0,
        "host_to_device_total_copies": 0,
        "host_to_device_total_bytes": 0,
        "device_to_device_state_import_copies": 0,
        "device_to_device_state_scatter_copies": 0,
        "diagnostic_layer_capture_device_to_host_copies": 0,
        "device_to_host_vector_copies": 0,
        "device_to_host_vector_values": 0,
        "device_to_host_vector_bytes": 0,
        "device_to_host_scalar_copies": 0,
    }
    manifest["synchronizations"] = 0
    manifest["layer_families"]["sampler"].update(
        {
            "device_result": "argmax_i32_rows_feedback_i64",
            "host_readback": "none_in_steady_replay",
        }
    )
    manifest["graph_setup_movement"] = {
        "host_to_device_input_copies": 1,
        "host_to_device_input_bytes": rows * DType.INT64.itemsize,
        "device_metadata_prepare_launches": 1,
        "device_to_device_state_import_copies": state_import_copies,
        "imported_slot_indices": list(imported_slot_indices),
    }
    manifest["graph"] = {
        "captured": True,
        "capture_count": 1,
        "replay_count": 0,
        "replayed_steps": 0,
        "replay_call_synchronizations": 0,
        "bucket_key": key.as_dict(),
        "record_layer_ids": list(layer_ids),
        "diagnostic_readback": {
            "token_vector_copies": 1 if generated_tokens is not None else 0,
            "token_values": record_capacity * rows if generated_tokens is not None else 0,
            "layer_hidden_copies": 1 if generated_hidden is not None else 0,
            "layer_hidden_values": (
                record_capacity * len(layer_ids) * rows * int(owner.runner.hidden_size)
                if generated_hidden is not None
                else 0
            ),
            "inside_replay": False,
        },
    }
    owner.last_packed_execution_manifest = manifest
    owner._packed_decode_sessions = session_tuple
    owner._packed_decode_last_layout = layout
    owner._packed_decode_state_dirty = True
    owner._packed_decode_session_ids = tuple(id(session) for session in session_tuple)
    owner._packed_decode_positions = positions
    handle = Qwen35GGUFPackedDecodeGraph(
        owner=owner,
        sessions=session_tuple,
        graph=graph,
        graph_exec=graph_exec,
        stream=stream,
        position_tuple=positions,
        steps_per_replay=step_width,
        max_replay_steps=replay_span,
        slot_capacity=slot_capacity,
        rows=rows,
        generated_tokens=generated_tokens,
        generated_hidden=generated_hidden,
        record_index=record_index,
        record_steps=record_capacity,
        record_layer_ids=layer_ids,
        bucket_key=key,
        execution_manifest=manifest,
    )
    owner._decode_graphs.append(handle)
    return handle


@dataclass
class Qwen35GGUFPackedDecodeGraph:
    owner: Any
    sessions: tuple[Any, ...]
    graph: int
    graph_exec: int
    stream: int
    position_tuple: tuple[int, ...]
    steps_per_replay: int
    max_replay_steps: int
    slot_capacity: int
    rows: int
    generated_tokens: DeviceBuffer | None
    generated_hidden: DeviceBuffer | None
    record_index: DeviceBuffer | None
    record_steps: int
    record_layer_ids: tuple[int, ...]
    bucket_key: Qwen35GGUFPackedDecodeGraphKey
    execution_manifest: dict[str, Any]
    replayed_steps: int = 0
    replay_count: int = 0
    flushed: bool = False
    closed: bool = False

    def replay(self, steps: int) -> None:
        if self.closed:
            raise RuntimeError("packed GGUF decode graph is closed")
        if self.flushed:
            raise RuntimeError("packed GGUF decode graph cannot replay after state flush")
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if steps % self.steps_per_replay:
            raise ValueError("steps must be divisible by steps_per_replay")
        if self.replayed_steps + steps > self.max_replay_steps:
            raise ValueError("cumulative graph replay exceeds the declared transition window")
        if self.record_steps and self.replayed_steps + steps > self.record_steps:
            raise ValueError("cumulative graph replay exceeds record capacity")
        expected = tuple(start + self.replayed_steps for start in self.position_tuple)
        observed = tuple(int(session.position) for session in self.sessions)
        if observed != expected:
            raise RuntimeError(
                f"packed decode graph state generation mismatch: expected {expected!r}, "
                f"observed {observed!r}"
            )
        launches = steps // self.steps_per_replay
        for _ in range(launches):
            self.owner.runtime.graph_launch(self.graph_exec, self.stream)
        if launches:
            self.owner.runtime.stream_synchronize(self.stream)
        self.replay_count += launches
        self.replayed_steps += steps
        final_positions = tuple(start + self.replayed_steps for start in self.position_tuple)
        for session, position in zip(self.sessions, final_positions, strict=True):
            session._position = int(position)
            session.scratch.position_host[0] = int(position)
            session.scratch.context_host[0] = int(position) + 1
        self.owner._packed_decode_session_ids = tuple(id(session) for session in self.sessions)
        self.owner._packed_decode_positions = final_positions
        self.owner._packed_decode_last_layout = _final_transition_layout(self)
        self.owner._packed_decode_state_dirty = True
        graph_manifest = self.execution_manifest["graph"]
        graph_manifest["replay_count"] = self.replay_count
        graph_manifest["replayed_steps"] = self.replayed_steps
        if launches:
            graph_manifest["replay_call_synchronizations"] += 1
        self.owner.last_packed_execution_manifest = self.execution_manifest

    def read_generated_token_ids(self, count: int | None = None) -> list[list[int]]:
        if self.closed:
            raise RuntimeError("packed GGUF decode graph is closed")
        if self.generated_tokens is None:
            raise RuntimeError("packed GGUF decode graph was captured without token recording")
        steps = self.replayed_steps if count is None else int(count)
        if steps < 0 or steps > self.record_steps:
            raise ValueError("count outside packed graph record capacity")
        host = np.empty((steps, self.rows), dtype=np.int32)
        copy_device_to_host(
            host_array_ptr(host),
            DeviceBuffer(self.generated_tokens.ptr, host.nbytes),
            runtime=self.owner.runtime,
        )
        return [[int(token) for token in row] for row in host.tolist()]

    def read_generated_layer_hidden(
        self,
        *,
        start: int = 0,
        count: int | None = None,
    ) -> np.ndarray:
        if self.closed:
            raise RuntimeError("packed GGUF decode graph is closed")
        if self.generated_hidden is None:
            raise RuntimeError("packed GGUF decode graph was captured without layer recording")
        start = int(start)
        steps = self.replayed_steps - start if count is None else int(count)
        if start < 0 or steps < 0 or start + steps > self.record_steps:
            raise ValueError("hidden slice outside packed graph record capacity")
        hidden_size = int(self.owner.runner.hidden_size)
        row_elements = len(self.record_layer_ids) * self.rows * hidden_size
        host = np.empty(
            (steps, len(self.record_layer_ids), self.rows, hidden_size),
            dtype=np.uint16,
        )
        copy_device_to_host(
            host_array_ptr(host),
            DeviceBuffer(
                self.generated_hidden.ptr + start * row_elements * DType.BF16.itemsize,
                host.nbytes,
            ),
            runtime=self.owner.runtime,
        )
        return np.ascontiguousarray(bf16_to_float32(host), dtype=np.float32)

    def flush_packed_state(self) -> bool:
        if self.closed:
            raise RuntimeError("packed GGUF decode graph is closed")
        if self.flushed:
            return False
        flushed = bool(self.owner.flush_packed_decode_state(stream=self.stream))
        self.flushed = flushed
        return flushed

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        runtime = self.owner.runtime or get_hip_runtime()
        runtime.graph_exec_destroy(self.graph_exec)
        runtime.graph_destroy(self.graph)
        if self.stream:
            runtime.stream_destroy(self.stream)
        for name in ("record_index", "generated_hidden", "generated_tokens"):
            buffer = getattr(self, name)
            if buffer is not None:
                free(buffer, runtime=runtime)
                setattr(self, name, None)
        graphs = getattr(self.owner, "_decode_graphs", None)
        if isinstance(graphs, list) and self in graphs:
            graphs.remove(self)
        for session in self.sessions:
            unpin = getattr(session, "_unpin_device_kv_graph", None)
            if callable(unpin):
                unpin(self)

    def __enter__(self) -> "Qwen35GGUFPackedDecodeGraph":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "Qwen35GGUFPackedDecodeGraph",
    "Qwen35GGUFPackedDecodeGraphKey",
    "build_qwen35_gguf_packed_decode_graph_key",
    "capture_qwen35_gguf_packed_decode_graph",
]
