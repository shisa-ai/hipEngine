"""CPU-only physical invocation owners for the native GGUF route.

Availability certificates do not own runtime pointers. This module binds the
actual named launch AND readback views, and the producer's compact index
metadata. It neither reads device contents nor grants numerical permission.
"""
from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Mapping
from operator import index

from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor


def physical_operand_inventory(value):
    """Enumerate named physical views recursively, not just owning allocations.

    Scratch is a dataclass: adding a Tensor, buffer, KVLiveSpans, metadata or
    nested collection automatically enters the inventory. Host array addresses,
    shapes/dtypes/strides are bound, never mutable token/position/state contents.
    Aliases retain their full field paths even when they share an allocation.
    Named scalar geometry is included; only spans' dynamic max_live_count is
    excluded (the invocation separately pins its decode bound). Host ndarrays hold
    dynamic contents; tuple/list fields on typed owners are structural metadata.
    """
    records = []
    def visit(obj, path, scalar=False):
        if obj is None or isinstance(obj, (str, int, float, bool, type)):
            if scalar and not isinstance(obj, type):
                records.append((path, "geometry", obj))
            return
        if isinstance(obj, Tensor):
            records.append((path, "tensor", obj.ptr, obj.shape, obj.dtype, obj.device, obj.strides))
        elif hasattr(obj, "ptr") and hasattr(obj, "nbytes"):
            records.append((path, "buffer", int(obj.ptr), int(obj.nbytes), getattr(obj, "dtype", None)))
        elif hasattr(obj, "__array_interface__"):
            a = obj.__array_interface__
            records.append((path, "host", a["data"], a["shape"], a["typestr"], a.get("strides")))
        elif is_dataclass(obj):
            for field in fields(obj):
                visit(getattr(obj, field.name), path + "." + field.name,
                      scalar=field.name != "max_live_count")
        elif isinstance(obj, Mapping):
            for name, item in sorted(obj.items()):
                visit(item, path + "." + str(name), scalar=name != "max_live_count")
        elif isinstance(obj, (tuple, list)):
            for index, item in enumerate(obj):
                visit(item, path + "." + str(index), scalar=scalar)
        elif hasattr(obj, "__dict__"):
            # CPU owner mocks follow the same named-view contract. Do not walk
            # sessions/runners here: only the explicit operand plan or scratch.
            for name, item in sorted(vars(obj).items()):
                if not callable(item):
                    visit(item, path + "." + name,
                          scalar=name != "max_live_count" and not isinstance(item, (list, tuple)))
    visit(value, "operands")
    return tuple(records)


def _require_buffer(buffer, dtype, elements, role):
    dtype = DType.parse(dtype)
    size = int(elements) * dtype.itemsize
    try:
        ptr = index(getattr(buffer, "ptr", 0))
        capacity = index(getattr(buffer, "nbytes", 0))
    except TypeError as exc:
        raise ValueError(f"native operand {role}: integral pointer/extent required") from exc
    if buffer is None or ptr <= 0 or ptr % dtype.itemsize or capacity < size:
        raise ValueError(f"native operand {role}: unowned/undersized/misaligned {dtype} view")
    # DeviceBuffer is untyped; its owner declares the native use. Typed mocks
    # and views must agree with that declaration, never reinterpret a dtype.
    actual_dtype = getattr(buffer, "dtype", None)
    if actual_dtype is not None and DType.parse(actual_dtype) != dtype:
        raise ValueError(f"native operand {role}: dtype does not match owner")
    return (ptr, size)


@dataclass(frozen=True)
class NativeIndexBinding:
    """Immutable receipt of the owner's successful canonical H2D publication.

    These arrays are read-only on the native route. This records the validated
    host producer and destination, NOT a claim to have inspected GPU contents.
    Arbitrary external writes require a new validated publication, not reuse.
    """
    cu_owner: object
    state_owner: object
    inventory: tuple
    cu_values: tuple[int, ...]
    state_values: tuple[int, ...]

    @classmethod
    def after_upload(cls, cu_owner, state_owner, cu_values, state_values):
        if (str(cu_values.dtype) != "int32" or str(state_values.dtype) != "int64"
                or cu_values.ndim != 1 or state_values.ndim != 1
                or not cu_values.flags.c_contiguous or not state_values.flags.c_contiguous):
            raise ValueError("native index producer dtype/geometry mismatch")
        capacity = len(state_values)
        cu, state = tuple(map(int, cu_values)), tuple(map(int, state_values))
        if cu != tuple(range(capacity + 1)) or state != tuple(range(capacity)):
            raise ValueError("native index producer must publish compact independent rows")
        _require_buffer(cu_owner, DType.INT32, capacity + 1, "cu_seqlens")
        _require_buffer(state_owner, DType.INT64, capacity, "state_indices")
        return cls(cu_owner, state_owner, physical_operand_inventory((cu_owner, state_owner)), cu, state)

    def validate(self, cu, state, rows):
        if (cu is not self.cu_owner or state is not self.state_owner
                or physical_operand_inventory((cu, state)) != self.inventory
                or self.cu_values[:rows + 1] != tuple(range(rows + 1))
                or self.state_values[:rows] != tuple(range(rows))):
            raise ValueError("native index ownership/extent/publication changed")


@dataclass(frozen=True)
class NativeRowsOperands:
    """The session-owned launch/readback plan consumed by native callers.

    Field names intentionally match the allocator owner. Native callers use this
    object, not a second hand-written pointer list. Sampler i32 views are bounded
    views into the existing larger i64-sized allocations; allocation is unchanged.
    """
    _token_buf: object
    _hidden_a: object
    _hidden_b: object
    _logits_buf: object
    _native_cu_seqlens_buf: object
    _native_state_indices_buf: object
    _lm_block_values: object
    _lm_block_indices: object
    _lm_out_index: object
    _lm_out_value: object
    _native_token_ids_host: object

    @classmethod
    def bind(cls, session, rows):
        plan = cls(**{f.name: getattr(session, f.name, None) for f in fields(cls)})
        owners = tuple(getattr(session, "_buffers", ()))
        cfg = session.runner.weights.config
        blocks = int(session._lm_head_stage1_blocks)
        if blocks <= 0:
            raise ValueError("native sampler block geometry missing")
        geometry = (
            ("_token_buf", DType.INT64, rows),
            ("_hidden_a", DType.BF16, rows * cfg.hidden_size),
            ("_hidden_b", DType.BF16, rows * cfg.hidden_size),
            ("_logits_buf", DType.FP32, rows * cfg.vocab_size),
            ("_native_cu_seqlens_buf", DType.INT32, rows + 1),
            ("_native_state_indices_buf", DType.INT64, rows),
            ("_lm_block_values", DType.FP32, rows * blocks),
            ("_lm_block_indices", DType.INT32, rows * blocks),
            ("_lm_out_index", DType.INT32, rows),
            ("_lm_out_value", DType.FP32, rows),
        )
        extents = []
        for name, dtype, elements in geometry:
            buf = getattr(plan, name)
            if not any(buf is owner for owner in owners):
                raise ValueError(f"native operand {name}: not an allocator-owned view")
            start, size = _require_buffer(buf, dtype, elements, name)
            if any(start < end and lo < start + size for lo, end in extents):
                raise ValueError(f"native operand {name}: incompatible read/write alias")
            extents.append((start, start + size))
        host = plan._native_token_ids_host
        if (getattr(host, "ndim", 0) != 1 or str(getattr(host, "dtype", "")) != "int32"
                or len(host) < rows or not host.flags.c_contiguous or not host.flags.writeable):
            raise ValueError("native sampler host readback view mismatch")
        binding = getattr(session, "_native_index_binding", None)
        if not isinstance(binding, NativeIndexBinding):
            raise ValueError("native index owner has no validated publication")
        binding.validate(plan._native_cu_seqlens_buf, plan._native_state_indices_buf, rows)
        return plan


@dataclass(frozen=True)
class NativeInvocationContext:
    """Owner-issued compact invocation; no legacy/default pointer authority."""
    session: object
    rows: int
    operands: NativeRowsOperands
    owner_identity: tuple
    scratch_inventory: tuple
    decode_bound: int

    @classmethod
    def issue(cls, session, rows, scratch):
        identity = session._authorize_native_rows(rows)
        expected = session._native_compact_scratch(
            rows, span_role=scratch.decode_spans.span_role,
            max_context_len=int(scratch.decode_spans.max_live_count))
        inventory = physical_operand_inventory(scratch)
        if inventory != physical_operand_inventory(expected):
            raise ValueError("native invocation scratch/views differ from authoritative compact owner")
        return cls(session, rows, NativeRowsOperands.bind(session, rows), identity, inventory,
                   int(scratch.decode_spans.max_live_count))

    def validate(self, session, rows, scratch):
        if (session is not self.session or rows != self.rows
                or session._authorize_native_rows(rows) != self.owner_identity
                or physical_operand_inventory(scratch) != self.scratch_inventory
                or int(scratch.decode_spans.max_live_count) != self.decode_bound):
            raise ValueError("native invocation owner/rows/physical operands changed")
        current = NativeRowsOperands.bind(session, rows)
        if any(getattr(self.operands, f.name) is not getattr(current, f.name) for f in fields(current)):
            raise ValueError("native invocation operand plan is not the current owner plan")
        expected = session._native_compact_scratch(
            rows, span_role=scratch.decode_spans.span_role,
            max_context_len=int(scratch.decode_spans.max_live_count))
        if physical_operand_inventory(expected) != self.scratch_inventory:
            raise ValueError("native invocation scratch view is not issued by its owner")

    def validate_layer(self, runner, layer_id, scratch, rows, hidden_ptr, out_ptr,
                       *, layer_type, indices_required=False, cu_seqlens_ptr=None, state_indices_ptr=None):
        if runner is not self.session.runner or runner.runtime is not self.session.runtime:
            raise ValueError("native invocation belongs to a different runner")
        self.validate(self.session, rows, scratch)
        cfg = runner.weights.config
        if (not 0 <= layer_id < len(cfg.layer_types) or runner.hidden_size != cfg.hidden_size
                or cfg.layer_types[layer_id] != layer_type):
            raise ValueError("native layer geometry/type differs from invocation owner")
        geometry = (("linear_qkv_width", 2 * cfg.ssm_group_count * cfg.ssm_state_size + cfg.ssm_inner_size),
                    ("ssm_value_dim", cfg.ssm_inner_size // cfg.ssm_time_step_rank)) if indices_required else (
                    ("q_width", cfg.head_count * cfg.key_length),
                    ("kv_width", cfg.head_count_kv * cfg.key_length))
        if any(getattr(runner, name, None) != size for name, size in geometry):
            raise ValueError("native runner operand geometry differs from certified model")
        # The actual full-stack call plan ping-pongs BF16 rows; the row-prefix
        # is a bounded view of the allocator-owned buffer, not arbitrary offsets.
        source, target = (self.operands._hidden_a, self.operands._hidden_b)
        if layer_id % 2:
            source, target = target, source
        _require_buffer(source, DType.BF16, rows * cfg.hidden_size, "hidden input")
        _require_buffer(target, DType.BF16, rows * cfg.hidden_size, "hidden output")
        try:
            hidden_ptr, out_ptr = index(hidden_ptr), index(out_ptr)
            if indices_required:
                cu_seqlens_ptr, state_indices_ptr = index(cu_seqlens_ptr), index(state_indices_ptr)
        except TypeError as exc:
            raise ValueError("native layer requires integral owned pointer arguments") from exc
        if hidden_ptr != source.ptr or out_ptr != target.ptr:
            raise ValueError("native layer input/output does not match owned BF16 row view")
        if indices_required:
            if (cu_seqlens_ptr != self.operands._native_cu_seqlens_buf.ptr
                    or state_indices_ptr != self.operands._native_state_indices_buf.ptr):
                raise ValueError("native layer indices do not match published compact views")


def require_native_layer_context(context, runner, layer_id, scratch, rows, hidden_ptr, out_ptr, *, layer_type, **indices):
    if not isinstance(context, NativeInvocationContext):
        raise ValueError("native layer requires a pre-certified owner-bound invocation context")
    context.validate_layer(runner, layer_id, scratch, rows, hidden_ptr, out_ptr,
                           layer_type=layer_type, indices_required=bool(indices), **indices)
