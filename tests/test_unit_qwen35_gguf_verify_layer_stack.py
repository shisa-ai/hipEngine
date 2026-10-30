"""Packed-verifier layer-stack ownership and execution tests.

The packed target verifier used to inline its layer walk in
``verify_target_blocks_batch``. ``qwen35_gguf_verify_layer_stack`` now owns that
walk: ordering, the BF16 hidden ping-pong, the aggregate per-layer-type stage
marks and the packed full-attention GPU mark. The session resolves every
selected operation, state and workspace first.

These tests pin four things:

* the dependency boundary - the owner imports no runner, session, scratch,
  allocation primitive or environment policy, and its records are immutable;
* the execution contract - layer order, ping-pong endpoints, aggregate stage
  names, GPU marks and the returned final hidden buffer;
* failure containment - a failing selected operation or GPU mark propagates
  unchanged, stops the walk and leaves no aggregate mark for the failed layer;
* the packed-verifier control trace - the exact operation, argument, stage and
  mutation sequence of ``verify_target_blocks_batch`` across BF16/INT8 KV,
  rows 1/3/6, MoE C2 selected/non-selected and timing on/off.

No GPU, no ROCm runtime, no environment policy mutation.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
from contextlib import nullcontext
from dataclasses import FrozenInstanceError, dataclass, fields
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

import hipengine.runtime.qwen35_gguf_runner as gguf_runner
from hipengine.core.dtype import DType
from hipengine.core.specdec2_scope import moe_physical_c2_exact_linear_session
from hipengine.runtime.qwen35_gguf_runner import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Qwen35GGUFResidentSession,
)

from tests.test_unit_qwen35_gguf_verify_timing_teardown import _FakeEventRuntime

_OWNER_MODULE = "hipengine.runtime.qwen35_gguf_verify_layer_stack"
_TIMING_ENV = "HIPENGINE_GGUF_PACKED_VERIFY_GPU_STAGE_TIMINGS"


def _owner():
    return importlib.import_module(_OWNER_MODULE)


# ---------------------------------------------------------------------------
# owner boundary
# ---------------------------------------------------------------------------
def test_owner_module_reaches_no_runner_session_or_allocation() -> None:
    owner = _owner()
    tree = ast.parse(Path(owner.__file__).read_text())

    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert imported == {
        "__future__",
        "dataclasses",
        "typing",
        "hipengine.loading.qwen35_gguf",
    }

    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    arguments = {
        argument.arg
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for argument in (*node.args.args, *node.args.kwonlyargs)
    }
    forbidden = {"session", "runner", "runtime", "malloc", "environ"}
    assert not names & forbidden
    assert not arguments & forbidden
    assert not attributes & {
        "environ",
        "malloc",
        "free",
        "memcpy_async",
        "copy_host_array_to_device",
        "copy_device_to_host",
        "get_hip_runtime",
        "getattr",
        "setattr",
    }

    # The executor's call sites spell every argument out; no thunk is invoked
    # through a bag, and no call forwards an unpacked mapping.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert not any(keyword.arg is None for keyword in node.keywords)
    assert "partial" not in names
    assert "functools" not in {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}


def test_owner_records_are_frozen_explicit_and_keyword_bounded() -> None:
    owner = _owner()
    assert [field.name for field in fields(owner.VerifyLayerStackGeometry)] == [
        "layer_types",
        "rows",
    ]
    assert [field.name for field in fields(owner.VerifyLayerStackHidden)] == [
        "first",
        "first_ptr",
        "second",
        "second_ptr",
    ]
    assert [field.name for field in fields(owner.VerifyOrdinaryLinearLayer)] == [
        "layer_id",
        "rows",
        "run_linear_rows",
        "scratch",
        "decode_scratch",
        "linear_state_rows",
        "commit_final_linear_state",
        "stream",
    ]
    assert [field.name for field in fields(owner.VerifyMoeC2LinearLayer)] == [
        "layer_id",
        "rows",
        "run_moe_c2_linear_rows",
        "scratch",
        "decode_scratch",
        "linear_state_rows",
        "state_capacity",
        "state_indices",
        "stream",
    ]
    assert [field.name for field in fields(owner.VerifyFullAttentionLayer)] == [
        "layer_id",
        "rows",
        "run_full_attention_rows",
        "scratch",
        "split_workspace",
        "stream",
    ]
    assert [field.name for field in fields(owner.VerifyLayerStackPlan)] == [
        "geometry",
        "hidden",
        "bindings",
    ]
    for record in (
        owner.VerifyOrdinaryLinearLayer,
        owner.VerifyMoeC2LinearLayer,
        owner.VerifyFullAttentionLayer,
    ):
        field_names = {field.name for field in fields(record)}
        assert "operation" not in field_names
        assert "run_rows" not in field_names

    signature = inspect.signature(owner.run_verify_layer_stack)
    assert list(signature.parameters) == [
        "plan",
        "stage_timings",
        "gpu_stage_recorder",
    ]
    assert not any(
        parameter.kind is parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    plan = _plan(owner)
    with pytest.raises(FrozenInstanceError):
        plan.geometry.rows = 4
    with pytest.raises(FrozenInstanceError):
        plan.bindings[0].layer_id = 1
    with pytest.raises(FrozenInstanceError):
        plan.hidden.first_ptr = 0


# ---------------------------------------------------------------------------
# owner execution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Ptr:
    label: str
    ptr: int


class _Marks:
    def __init__(self) -> None:
        self.names: list[str] = []

    def mark(self, name: str, *aliases: str) -> None:
        self.names.append(name)


class _Ops:
    """Records the exact positional and keyword arguments of each operation."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def linear(
        self,
        layer_id,
        hidden_ptr,
        out_ptr,
        scratch,
        *,
        rows,
        decode_scratch,
        stream=0,
        expert_sidecar=None,
        linear_state_rows=None,
        commit_final_linear_state=True,
        hidden_f32_ptr=None,
        out_f32_ptr=None,
        stage_timings=None,
        sync_stage_timings=False,
        stage_prefix="",
        gpu_stage_recorder=None,
    ) -> None:
        self.calls.append(
            {
                "op": "linear",
                "layer_id": layer_id,
                "hidden_ptr": hidden_ptr,
                "out_ptr": out_ptr,
                "scratch": scratch,
                "rows": rows,
                "decode_scratch": decode_scratch,
                "stream": stream,
                "expert_sidecar": expert_sidecar,
                "linear_state_rows": linear_state_rows,
                "commit_final_linear_state": commit_final_linear_state,
                "hidden_f32_ptr": hidden_f32_ptr,
                "out_f32_ptr": out_f32_ptr,
                "stage_timings": stage_timings,
                "sync_stage_timings": sync_stage_timings,
                "stage_prefix": stage_prefix,
                "gpu_stage_recorder": gpu_stage_recorder,
            }
        )

    def moe(
        self,
        layer_id,
        hidden_ptr,
        out_ptr,
        scratch,
        *,
        rows,
        decode_scratch,
        linear_state_rows,
        state_capacity,
        state_indices,
        stream=0,
    ) -> None:
        self.calls.append(
            {
                "op": "moe",
                "layer_id": layer_id,
                "hidden_ptr": hidden_ptr,
                "out_ptr": out_ptr,
                "scratch": scratch,
                "rows": rows,
                "decode_scratch": decode_scratch,
                "linear_state_rows": linear_state_rows,
                "state_capacity": state_capacity,
                "state_indices": state_indices,
                "stream": stream,
            }
        )

    def full(
        self,
        layer_id,
        hidden_ptr,
        out_ptr,
        scratch,
        *,
        stream=0,
        expert_sidecar=None,
        stage_timings=None,
        sync_stage_timings=False,
        stage_prefix="",
        split_workspace=None,
    ) -> None:
        self.calls.append(
            {
                "op": "full",
                "layer_id": layer_id,
                "hidden_ptr": hidden_ptr,
                "out_ptr": out_ptr,
                "scratch": scratch,
                "stream": stream,
                "expert_sidecar": expert_sidecar,
                "stage_timings": stage_timings,
                "sync_stage_timings": sync_stage_timings,
                "stage_prefix": stage_prefix,
                "split_workspace": split_workspace,
            }
        )


_SCRATCH = _Ptr("scratch", 0x300)
_DECODE_SCRATCH = _Ptr("decode_scratch", 0x310)
_STATE_ROWS = (_Ptr("conv_rows", 0x700), _Ptr("recurrent_rows", 0x800))
_STATE_INDICES = (0,)
_SPLIT_WORKSPACE = _Ptr("split_workspace", 0xC8)


def _plan(
    owner,
    *,
    rows: int = 3,
    layer_types: tuple[str, ...] = (LINEAR_ATTENTION, FULL_ATTENTION),
    linear_kind: str = "ordinary",
    ops: _Ops | None = None,
):
    ops = _Ops() if ops is None else ops
    bindings = []
    for layer_id, layer_type in enumerate(layer_types):
        if layer_type == LINEAR_ATTENTION and linear_kind == "moe":
            bindings.append(
                owner.VerifyMoeC2LinearLayer(
                    layer_id=layer_id,
                    rows=rows,
                    run_moe_c2_linear_rows=ops.moe,
                    scratch=_SCRATCH,
                    decode_scratch=_DECODE_SCRATCH,
                    linear_state_rows=_STATE_ROWS,
                    state_capacity=1,
                    state_indices=_STATE_INDICES,
                    stream=0,
                )
            )
        elif layer_type == LINEAR_ATTENTION:
            bindings.append(
                owner.VerifyOrdinaryLinearLayer(
                    layer_id=layer_id,
                    rows=rows,
                    run_linear_rows=ops.linear,
                    scratch=_SCRATCH,
                    decode_scratch=_DECODE_SCRATCH,
                    linear_state_rows=(
                        _STATE_ROWS if rows == 6 else None
                    ),
                    commit_final_linear_state=True,
                    stream=0,
                )
            )
        else:
            bindings.append(
                owner.VerifyFullAttentionLayer(
                    layer_id=layer_id,
                    rows=rows,
                    run_full_attention_rows=ops.full,
                    scratch=_SCRATCH,
                    split_workspace=_SPLIT_WORKSPACE,
                    stream=0,
                )
            )
    return owner.VerifyLayerStackPlan(
        geometry=owner.VerifyLayerStackGeometry(
            layer_types=layer_types,
            rows=rows,
        ),
        hidden=owner.VerifyLayerStackHidden(
            first=_Ptr("a", 0xA0),
            first_ptr=0xA0,
            second=_Ptr("b", 0xB0),
            second_ptr=0xB0,
        ),
        bindings=tuple(bindings),
    )


def test_owner_calls_each_named_operation_with_explicit_fixed_arguments() -> None:
    owner = _owner()
    ops = _Ops()
    plan = _plan(
        owner,
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
        linear_kind="ordinary",
        ops=ops,
    )
    marks = _Marks()
    timings: dict[str, float] = {}

    final = owner.run_verify_layer_stack(
        plan,
        stage_timings=timings,
        gpu_stage_recorder=marks,
    )

    assert ops.calls == [
        {
            "op": "linear",
            "layer_id": 0,
            "hidden_ptr": 0xA0,
            "out_ptr": 0xB0,
            "scratch": _SCRATCH,
            "rows": 3,
            "decode_scratch": _DECODE_SCRATCH,
            "stream": 0,
            "expert_sidecar": None,
            "linear_state_rows": None,
            "commit_final_linear_state": True,
            "hidden_f32_ptr": None,
            "out_f32_ptr": None,
            "stage_timings": None,
            "sync_stage_timings": False,
            "stage_prefix": owner.LINEAR_LAYER_STAGE_PREFIX,
            "gpu_stage_recorder": marks,
        },
        {
            "op": "full",
            "layer_id": 1,
            "hidden_ptr": 0xB0,
            "out_ptr": 0xA0,
            "scratch": _SCRATCH,
            "stream": 0,
            "expert_sidecar": None,
            "stage_timings": None,
            "sync_stage_timings": False,
            "stage_prefix": owner.FULL_ATTENTION_LAYER_STAGE_PREFIX,
            "split_workspace": _SPLIT_WORKSPACE,
        },
    ]
    assert final is plan.hidden.first
    assert marks.names == [owner.FULL_ATTENTION_GPU_STAGE]
    assert list(timings) == [
        owner.LINEAR_ATTENTION_STAGE,
        owner.FULL_ATTENTION_STAGE,
    ]


def test_owner_calls_the_moe_c2_operation_with_its_fixed_arguments() -> None:
    owner = _owner()
    ops = _Ops()
    plan = _plan(
        owner,
        rows=6,
        layer_types=(LINEAR_ATTENTION,),
        linear_kind="moe",
        ops=ops,
    )
    marks = _Marks()

    owner.run_verify_layer_stack(
        plan,
        stage_timings={},
        gpu_stage_recorder=marks,
    )

    assert ops.calls == [
        {
            "op": "moe",
            "layer_id": 0,
            "hidden_ptr": 0xA0,
            "out_ptr": 0xB0,
            "scratch": _SCRATCH,
            "rows": 6,
            "decode_scratch": _DECODE_SCRATCH,
            "linear_state_rows": _STATE_ROWS,
            "state_capacity": 1,
            "state_indices": _STATE_INDICES,
            "stream": 0,
        }
    ]
    assert marks.names == []


def test_owner_without_recorder_still_marks_aggregate_stages() -> None:
    owner = _owner()
    timings: dict[str, float] = {}
    plan = _plan(owner, layer_types=(FULL_ATTENTION, LINEAR_ATTENTION))

    owner.run_verify_layer_stack(plan, stage_timings=timings)

    assert list(timings) == [
        owner.FULL_ATTENTION_STAGE,
        owner.LINEAR_ATTENTION_STAGE,
    ]


def test_owner_rejects_mismatched_plans_before_any_operation() -> None:
    owner = _owner()
    ops = _Ops()

    wrong_rows = _plan(owner, rows=3, ops=ops)
    wrong_rows = owner.VerifyLayerStackPlan(
        geometry=owner.VerifyLayerStackGeometry(
            layer_types=wrong_rows.geometry.layer_types,
            rows=4,
        ),
        hidden=wrong_rows.hidden,
        bindings=wrong_rows.bindings,
    )
    with pytest.raises(ValueError, match="was resolved for 3 rows, not 4"):
        owner.run_verify_layer_stack(wrong_rows, stage_timings={})

    wrong_order = _plan(owner, ops=ops)
    first, second = wrong_order.bindings
    wrong_order = owner.VerifyLayerStackPlan(
        geometry=wrong_order.geometry,
        hidden=wrong_order.hidden,
        bindings=(
            first.__class__(**{**first.__dict__, "layer_id": 1}),
            second,
        ),
    )
    with pytest.raises(ValueError, match="names layer 1"):
        owner.run_verify_layer_stack(wrong_order, stage_timings={})

    wrong_type = _plan(owner, layer_types=("sliding_attention",), ops=ops)
    with pytest.raises(
        ValueError, match="unsupported GGUF layer type 'sliding_attention'"
    ):
        owner.run_verify_layer_stack(wrong_type, stage_timings={})

    wrong_variant = _plan(owner, layer_types=(FULL_ATTENTION,), ops=ops)
    linear_binding = _plan(owner, layer_types=(LINEAR_ATTENTION,), ops=ops).bindings[0]
    wrong_variant = owner.VerifyLayerStackPlan(
        geometry=wrong_variant.geometry,
        hidden=wrong_variant.hidden,
        bindings=(linear_binding,),
    )
    with pytest.raises(ValueError, match="resolved as linear attention"):
        owner.run_verify_layer_stack(wrong_variant, stage_timings={})

    no_state_rows = _plan(
        owner, rows=6, layer_types=(LINEAR_ATTENTION,), linear_kind="moe", ops=ops
    )
    binding = no_state_rows.bindings[0]
    no_state_rows = owner.VerifyLayerStackPlan(
        geometry=no_state_rows.geometry,
        hidden=no_state_rows.hidden,
        bindings=(
            binding.__class__(**{**binding.__dict__, "linear_state_rows": None}),
        ),
    )
    with pytest.raises(ValueError, match="requires captured state rows"):
        owner.run_verify_layer_stack(no_state_rows, stage_timings={})

    invalid_ptr = _plan(owner, layer_types=(LINEAR_ATTENTION,), ops=ops)
    invalid_ptr = owner.VerifyLayerStackPlan(
        geometry=invalid_ptr.geometry,
        hidden=owner.VerifyLayerStackHidden(
            first=invalid_ptr.hidden.first,
            first_ptr=0,
            second=invalid_ptr.hidden.second,
            second_ptr=invalid_ptr.hidden.second_ptr,
        ),
        bindings=invalid_ptr.bindings,
    )
    with pytest.raises(ValueError, match="requires valid pointers"):
        owner.run_verify_layer_stack(invalid_ptr, stage_timings={})

    assert ops.calls == []


@pytest.mark.parametrize("same_object", [False, True])
@pytest.mark.parametrize("layer_types", [(LINEAR_ATTENTION,), (LINEAR_ATTENTION, FULL_ATTENTION, LINEAR_ATTENTION)])
def test_owner_preserves_planner_selected_in_place_hidden_views(same_object, layer_types) -> None:
    owner = _owner()
    ops = _Ops()
    plan = _plan(owner, layer_types=layer_types, ops=ops)
    first = plan.hidden.first
    second = first if same_object else _Ptr("alias-view", plan.hidden.first_ptr)
    plan = owner.VerifyLayerStackPlan(
        geometry=plan.geometry,
        hidden=owner.VerifyLayerStackHidden(first, first.ptr, second, first.ptr),
        bindings=plan.bindings,
    )

    result = owner.run_verify_layer_stack(plan, stage_timings={})

    assert result is second
    assert len(ops.calls) == len(layer_types)
    assert all(call["hidden_ptr"] == call["out_ptr"] == first.ptr for call in ops.calls)


def test_owner_propagates_operation_failure_and_stops_the_walk() -> None:
    owner = _owner()
    ops = _Ops()
    plan = _plan(owner, layer_types=(LINEAR_ATTENTION, FULL_ATTENTION), ops=ops)
    first = plan.bindings[0]
    plan = owner.VerifyLayerStackPlan(
        geometry=plan.geometry,
        hidden=plan.hidden,
        bindings=(
            first.__class__(
                **{**first.__dict__, "run_linear_rows": _raise("layer boom")}
            ),
            plan.bindings[1],
        ),
    )
    marks = _Marks()
    timings: dict[str, float] = {}

    with pytest.raises(RuntimeError, match="layer boom"):
        owner.run_verify_layer_stack(
            plan,
            stage_timings=timings,
            gpu_stage_recorder=marks,
        )

    assert ops.calls == []
    assert timings == {}
    assert marks.names == []


def test_owner_propagates_gpu_mark_failure_before_the_aggregate_mark() -> None:
    owner = _owner()
    ops = _Ops()
    plan = _plan(owner, layer_types=(FULL_ATTENTION, LINEAR_ATTENTION), ops=ops)
    timings: dict[str, float] = {}

    class _FailingMarks:
        def mark(self, name: str, *aliases: str) -> None:
            raise RuntimeError("mark boom")

    with pytest.raises(RuntimeError, match="mark boom"):
        owner.run_verify_layer_stack(
            plan,
            stage_timings=timings,
            gpu_stage_recorder=_FailingMarks(),
        )

    assert [call["op"] for call in ops.calls] == ["full"]
    assert timings == {}


def test_owner_propagates_stage_sink_failure_and_stops_the_walk() -> None:
    owner = _owner()
    ops = _Ops()
    plan = _plan(owner, layer_types=(LINEAR_ATTENTION, FULL_ATTENTION), ops=ops)

    class _FailingTimings(dict):
        def __setitem__(self, name: str, value: float) -> None:
            raise RuntimeError("stage sink boom")

    with pytest.raises(RuntimeError, match="stage sink boom"):
        owner.run_verify_layer_stack(
            plan,
            stage_timings=_FailingTimings(),
        )

    assert [call["op"] for call in ops.calls] == ["linear"]


def _raise(message: str):
    def run(*args: object, **kwargs: object) -> None:
        raise RuntimeError(message)

    return run


# ---------------------------------------------------------------------------
# packed-verifier control harness
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Buffer:
    label: str
    ptr: int


@dataclass(frozen=True)
class _PackedScratch:
    label: str
    norm: _Buffer
    append_spans: object
    prefill_spans: object
    retained_decode_kernel: object | None = None


@dataclass(frozen=True)
class _LayerScratch:
    label: str
    rows: int
    cos_table: _Buffer
    sin_table: _Buffer


@dataclass(frozen=True)
class _SessionScratch:
    hidden_seed_fp32: _Buffer
    cos_table: _Buffer
    sin_table: _Buffer
    layer_conv_states: tuple = ()
    layer_recurrent_states: tuple = ()


@dataclass(frozen=True)
class _PackedState:
    slot_count: int
    layer_conv_states: tuple
    layer_recurrent_states: tuple


class _PackedScratchBase:
    def for_packed_verify_layout(self, layout, *, runtime, stream: int = 0):
        return _PackedScratch(
            label="packed",
            norm=_Buffer("norm", 0x300),
            append_spans=None,
            prefill_spans=None,
        )


_LAYER_TYPES = (LINEAR_ATTENTION, FULL_ATTENTION, LINEAR_ATTENTION, FULL_ATTENTION)


def _record_linear(trace):
    def record(
        layer_id,
        hidden_ptr,
        out_ptr,
        scratch,
        *,
        rows,
        decode_scratch,
        stream=0,
        expert_sidecar=None,
        linear_state_rows=None,
        commit_final_linear_state=True,
        hidden_f32_ptr=None,
        out_f32_ptr=None,
        stage_timings=None,
        sync_stage_timings=False,
        stage_prefix="target_block_linear_attn",
        gpu_stage_recorder=None,
    ) -> None:
        trace["ops"].append(
            {
                "op": "linear",
                "layer": layer_id,
                "hidden_in": hidden_ptr,
                "hidden_out": out_ptr,
                "scratch": scratch.label,
                "retained_decode_kernel": (
                    None
                    if scratch.retained_decode_kernel is None
                    else scratch.retained_decode_kernel.label
                ),
                "rows": rows,
                "decode_conv": decode_scratch.layer_conv_states[layer_id].label,
                "decode_recurrent": decode_scratch.layer_recurrent_states[layer_id].label,
                "state_rows": (
                    None
                    if linear_state_rows is None
                    else [buffer.label for buffer in linear_state_rows]
                ),
                "commit_final_state": bool(commit_final_linear_state),
                "stage_prefix": stage_prefix,
                "recorder": gpu_stage_recorder is not None,
            }
        )

    return record


def _record_moe(trace):
    def record(
        layer_id,
        hidden_ptr,
        out_ptr,
        scratch,
        *,
        rows,
        decode_scratch,
        linear_state_rows,
        state_capacity,
        state_indices,
        stream=0,
    ) -> None:
        trace["ops"].append(
            {
                "op": "moe_c2_linear",
                "layer": layer_id,
                "hidden_in": hidden_ptr,
                "hidden_out": out_ptr,
                "scratch": scratch.label,
                "retained_decode_kernel": (
                    None
                    if scratch.retained_decode_kernel is None
                    else scratch.retained_decode_kernel.label
                ),
                "rows": rows,
                "decode_conv": decode_scratch.layer_conv_states[layer_id].label,
                "decode_recurrent": decode_scratch.layer_recurrent_states[layer_id].label,
                "state_rows": [buffer.label for buffer in linear_state_rows],
                "state_capacity": int(state_capacity),
                "state_indices": [int(index) for index in state_indices],
            }
        )

    return record


def _record_full(trace):
    def record(
        layer_id,
        hidden_ptr,
        out_ptr,
        scratch,
        *,
        stream=0,
        expert_sidecar=None,
        hidden_f32_ptr=None,
        out_f32_ptr=None,
        stage_timings=None,
        sync_stage_timings=False,
        stage_prefix="target_block_full_attn",
        split_workspace=None,
        kv_write_only=False,
    ) -> None:
        trace["ops"].append(
            {
                "op": "full_attention",
                "layer": layer_id,
                "hidden_in": hidden_ptr,
                "hidden_out": out_ptr,
                "scratch": scratch.label,
                "rows": int(scratch.rows),
                "cos_table": scratch.cos_table.label,
                "sin_table": scratch.sin_table.label,
                "split_workspace": (
                    None if split_workspace is None else split_workspace.label
                ),
                "stage_prefix": stage_prefix,
            }
        )

    return record


def _make_packed_session(
    monkeypatch: pytest.MonkeyPatch,
    runtime: _FakeEventRuntime,
    *,
    kv_dtype: DType,
    rows: int,
    capture_linear_state_rows: bool,
    timing: bool,
    trace: dict[str, object],
) -> tuple[Qwen35GGUFResidentSession, dict[str, object]]:
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_RESIDUAL", raising=False)
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", "1")
    monkeypatch.delenv("HIPENGINE_GGUF_PACKED_VERIFY_CAPTURE_DIR", raising=False)
    if timing:
        monkeypatch.setenv(_TIMING_ENV, "1")
    else:
        monkeypatch.delenv(_TIMING_ENV, raising=False)

    weights = SimpleNamespace(
        config=SimpleNamespace(layer_types=_LAYER_TYPES, rms_norm_eps=1e-6),
        root=lambda name: SimpleNamespace(
            allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=0x400))
        ),
    )
    runner = SimpleNamespace(weights=weights, hidden_size=8, vocab_size=128)
    runner._run_linear_attention_prefill_layer_rows = _record_linear(trace)
    runner._run_moe_physical_c2_linear_rows_exact = _record_moe(trace)
    runner._run_full_attention_decode_batch_layer_rows = _record_full(trace)

    layer_states = tuple(
        _Buffer(f"conv[{index}]", 0x500 + index) for index in range(len(_LAYER_TYPES))
    )
    recurrent_states = tuple(
        _Buffer(f"recurrent[{index}]", 0x600 + index)
        for index in range(len(_LAYER_TYPES))
    )
    packed_state = _PackedState(
        slot_count=1,
        layer_conv_states=layer_states,
        layer_recurrent_states=recurrent_states,
    )
    session_scratch = _SessionScratch(
        hidden_seed_fp32=_Buffer("hidden_seed_fp32", 0x101),
        cos_table=_Buffer("cos", 0x200),
        sin_table=_Buffer("sin", 0x201),
    )

    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = runner
    session.scratch = session_scratch
    session._prefill_token_buf = _Buffer("tokens", 0x102)
    session._prefill_hidden_a = _Buffer("hidden_a", 0xA0)
    session._prefill_hidden_b = _Buffer("hidden_b", 0xB0)
    session._verify_hidden_seed_buf = _Buffer("hidden_seed", 0x105)
    session._bulk_prefill_scratch = SimpleNamespace(rows=16)
    session.kv_storage_dtype = kv_dtype
    session.use_expert_sidecar = False
    session.runtime = runtime
    session._position = 0
    session.use_gemv_decode = False
    session.target_verifier_production_q4_rowtile = False

    session._direct_resident_verify_linear_state = MethodType(
        lambda self, jobs: None, session
    )
    session._ensure_packed_verify_workspace = MethodType(
        lambda self, **kwargs: (packed_state, _PackedScratchBase()), session
    )
    session._ensure_verify_block_buffers = MethodType(
        lambda self, rows, **kwargs: None, session
    )
    session._ensure_verify_linear_state_row_buffers = MethodType(
        lambda self, rows, **kwargs: None, session
    )
    session._packed_verify_initial_linear_state = MethodType(
        lambda self, sessions, layout, state: (layout, state, True), session
    )
    session._sync_packed_verify_initial_state = MethodType(
        lambda self, *args, **kwargs: None, session
    )
    session._packed_full_attention_scratch_for_layer = MethodType(
        lambda self, scratch, state, layer_id: _LayerScratch(
            label=f"full_scratch[{layer_id}]",
            rows=rows,
            cos_table=_Buffer("unbound_cos", 0),
            sin_table=_Buffer("unbound_sin", 0),
        ),
        session,
    )
    session._verify_linear_state_row_pair = MethodType(
        lambda self, layer_id: (
            (
                _Buffer(f"conv_rows[{layer_id}]", 0x700 + layer_id),
                _Buffer(f"recurrent_rows[{layer_id}]", 0x800 + layer_id),
            )
            if capture_linear_state_rows
            else None
        ),
        session,
    )
    session._packed_ar_direct_decode_kernel_for_sessions = MethodType(
        lambda self, sessions, **kwargs: (
            _Buffer("retained_decode_kernel", 0xC0)
            if kv_dtype == DType.INT8_PER_TOKEN_HEAD
            else None
        ),
        session,
    )
    session._ensure_packed_ar_attention_workspace = MethodType(
        lambda self, **kwargs: _Buffer("split_workspace", 0xC8), session
    )
    session._device_token_embedding_weight = MethodType(
        lambda self, **kwargs: object(), session
    )
    session._sample_target_block_rows_from_hidden = MethodType(
        lambda self, *args, **kwargs: np.zeros(rows, dtype=np.int64), session
    )
    session._scatter_packed_verify_outputs = MethodType(
        lambda self, *args, **kwargs: [], session
    )
    for name in (
        "_q5_raw_mmq_target_context",
        "_q6_dp4a_grouped_target_context",
        "_q6_integer_mmq_context",
        "_iq_dense_mmq_context",
    ):
        setattr(
            session,
            name,
            MethodType(lambda self, *args, **kwargs: nullcontext(), session),
        )

    monkeypatch.setattr(
        gguf_runner,
        "_rebind_packed_verify_layout_pages",
        lambda layout, state: layout,
    )
    monkeypatch.setattr(
        gguf_runner, "_stage_gguf_packed_verify_token_ids", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(gguf_runner, "launch_gguf_embedding", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gguf_runner,
        "gguf_rmsnorm_bf16_f32_weight",
        lambda hidden_ptr, *args, **kwargs: trace.__setitem__(
            "output_norm_hidden", int(hidden_ptr)
        ),
    )
    monkeypatch.setattr(
        gguf_runner, "gguf_rmsnorm_bf16_f32_weight_out_f32", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(gguf_runner, "copy_device_to_host", lambda *args, **kwargs: None)
    monkeypatch.setattr(gguf_runner, "host_array_ptr", lambda array: 0)
    monkeypatch.setattr(
        gguf_runner, "wmma_prefill_session", lambda *args, **kwargs: nullcontext()
    )
    monkeypatch.setattr(
        gguf_runner, "gemv_decode_session", lambda *args, **kwargs: nullcontext()
    )
    monkeypatch.setattr(
        gguf_runner,
        "target_verifier_rowtile_session",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        gguf_runner,
        "target_verifier_production_q4_rowtile_session",
        lambda *args, **kwargs: nullcontext(),
    )

    base_recorder = gguf_runner._HipEventStageRecorder

    class _RecordingRecorder(base_recorder):
        def mark(self, name: str, *aliases: str) -> None:
            trace["gpu_marks"].append(str(name))
            return super().mark(name, *aliases)

    monkeypatch.setattr(gguf_runner, "_HipEventStageRecorder", _RecordingRecorder)

    job: dict[str, object] = {
        "session": session,
        "input_token_ids": tuple(range(1, rows + 1)),
    }
    if capture_linear_state_rows:
        job["capture_linear_state_rows"] = True
        job["defer_linear_state_commit"] = True
    return session, job


def _capture_trace(
    monkeypatch: pytest.MonkeyPatch,
    *,
    kv_dtype: DType,
    rows: int,
    moe_c2: bool,
    timing: bool,
) -> dict[str, object]:
    trace: dict[str, object] = {
        "ops": [],
        "gpu_marks": [],
        "output_norm_hidden": None,
    }
    capture_linear_state_rows = rows == 6
    session, job = _make_packed_session(
        monkeypatch,
        _FakeEventRuntime(),
        kv_dtype=kv_dtype,
        rows=rows,
        capture_linear_state_rows=capture_linear_state_rows,
        timing=timing,
        trace=trace,
    )
    policy = (
        moe_physical_c2_exact_linear_session(True) if moe_c2 else nullcontext()
    )
    with policy:
        results = session.verify_target_blocks_batch([job])
    trace["results"] = len(results)
    trace["position"] = int(session._position)
    trace["stage_keys"] = list(session.last_packed_verify_stage_timings_ms)
    return trace


_CASES = tuple(
    (kv_dtype, rows, moe_c2, timing)
    for kv_dtype in ("bf16", "int8")
    for rows in (1, 3, 6)
    for moe_c2 in (False, True)
    for timing in (False, True)
)

_KV_DTYPES = {
    "bf16": DType.BF16,
    "int8": DType.INT8_PER_TOKEN_HEAD,
}


def _case_name(kv_dtype: str, rows: int, moe_c2: bool, timing: bool) -> str:
    return f"{kv_dtype}-r{rows}-c2{int(moe_c2)}-t{int(timing)}"


# The committed control trace, captured from the inline packed loop at
# HEAD b1a621b75 before the layer walk moved into the owner. It must stay
# byte-identical after the move: same selected operations, same bound
# arguments, same stage keys and GPU marks, same ping-pong endpoints.
_CONTROL_TRACE: dict[str, dict[str, object]] = json.loads(
    (Path(__file__).parent / "fixtures" / "qwen35_gguf_verify_layer_stack_control.json").read_text()
)


@pytest.mark.parametrize("kv_dtype,rows,moe_c2,timing", _CASES)
def test_packed_verifier_layer_stack_matches_committed_control_trace(
    monkeypatch: pytest.MonkeyPatch,
    kv_dtype: str,
    rows: int,
    moe_c2: bool,
    timing: bool,
) -> None:
    captured = _capture_trace(
        monkeypatch,
        kv_dtype=_KV_DTYPES[kv_dtype],
        rows=rows,
        moe_c2=moe_c2,
        timing=timing,
    )
    expected = _CONTROL_TRACE[_case_name(kv_dtype, rows, moe_c2, timing)]
    assert captured == expected


# ---------------------------------------------------------------------------
# packed-verifier failure containment
# ---------------------------------------------------------------------------
def _failing_session(
    monkeypatch: pytest.MonkeyPatch,
    runtime: _FakeEventRuntime,
    *,
    fail: str,
) -> tuple[Qwen35GGUFResidentSession, dict[str, object]]:
    trace: dict[str, object] = {"ops": [], "gpu_marks": [], "output_norm_hidden": None}
    session, job = _make_packed_session(
        monkeypatch,
        runtime,
        kv_dtype=DType.BF16,
        rows=3,
        capture_linear_state_rows=False,
        timing=True,
        trace=trace,
    )

    def fail_linear(*args: object, **kwargs: object) -> None:
        raise RuntimeError("linear boom")

    def fail_full(*args: object, **kwargs: object) -> None:
        raise RuntimeError("full boom")

    def fail_mark(name: str, *aliases: str) -> None:
        raise RuntimeError("mark boom")

    if fail == "linear":
        session.runner._run_linear_attention_prefill_layer_rows = fail_linear
    elif fail == "full":
        session.runner._run_full_attention_decode_batch_layer_rows = fail_full
    elif fail == "mark":
        base_recorder = gguf_runner._HipEventStageRecorder

        class _FailingMarkRecorder(base_recorder):
            def mark(self, name: str, *aliases: str) -> None:
                if name == "packed_verify_gpu_full_attn_layers":
                    raise RuntimeError("mark boom")
                return super().mark(name, *aliases)

        monkeypatch.setattr(gguf_runner, "_HipEventStageRecorder", _FailingMarkRecorder)
    else:  # pragma: no cover - programming error
        raise AssertionError(fail)
    return session, job


@pytest.mark.parametrize("fail", ["linear", "full", "mark"])
def test_packed_verifier_failure_preserves_error_and_releases_recorder(
    monkeypatch: pytest.MonkeyPatch,
    fail: str,
) -> None:
    runtime = _FakeEventRuntime()
    session, job = _failing_session(monkeypatch, runtime, fail=fail)

    with pytest.raises(RuntimeError, match=f"{fail} boom"):
        session.verify_target_blocks_batch([job])

    assert runtime.created
    assert sorted(runtime.created) == sorted(runtime.destroyed)
    timings = session.last_packed_verify_stage_timings_ms
    if fail == "linear":
        assert "packed_verify_linear_attn_layers" not in timings
        assert "packed_verify_full_attn_layers" not in timings
    else:
        assert "packed_verify_linear_attn_layers" in timings
        assert "packed_verify_full_attn_layers" not in timings


def test_packed_verifier_moe_c2_operation_failure_preserves_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeEventRuntime()
    trace: dict[str, object] = {"ops": [], "gpu_marks": [], "output_norm_hidden": None}
    session, job = _make_packed_session(
        monkeypatch,
        runtime,
        kv_dtype=DType.BF16,
        rows=6,
        capture_linear_state_rows=True,
        timing=True,
        trace=trace,
    )
    ordinary_calls: list[int] = []

    def fail_moe(*args: object, **kwargs: object) -> None:
        raise RuntimeError("moe boom")

    def record_ordinary(layer_id, *args: object, **kwargs: object) -> None:
        ordinary_calls.append(layer_id)

    session.runner._run_moe_physical_c2_linear_rows_exact = fail_moe
    session.runner._run_linear_attention_prefill_layer_rows = record_ordinary

    with moe_physical_c2_exact_linear_session(True):
        with pytest.raises(RuntimeError, match="moe boom"):
            session.verify_target_blocks_batch([job])

    assert ordinary_calls == []
    assert runtime.created
    assert sorted(runtime.created) == sorted(runtime.destroyed)
    assert "packed_verify_linear_attn_layers" not in (
        session.last_packed_verify_stage_timings_ms
    )


def test_packed_verifier_resolves_every_layer_before_running_any(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: dict[str, object] = {"ops": [], "gpu_marks": [], "output_norm_hidden": None}
    session, job = _make_packed_session(
        monkeypatch,
        _FakeEventRuntime(),
        kv_dtype=DType.BF16,
        rows=6,
        capture_linear_state_rows=True,
        timing=False,
        trace=trace,
    )

    def fail_lookup(self, layer_id):
        raise RuntimeError(f"resolve boom {layer_id}")

    session._verify_linear_state_row_pair = MethodType(fail_lookup, session)

    with pytest.raises(RuntimeError, match="resolve boom 0"):
        session.verify_target_blocks_batch([job])

    assert trace["ops"] == []


def test_packed_verifier_rejects_unknown_layer_type_before_running_any(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: dict[str, object] = {"ops": [], "gpu_marks": [], "output_norm_hidden": None}
    session, job = _make_packed_session(
        monkeypatch,
        _FakeEventRuntime(),
        kv_dtype=DType.BF16,
        rows=3,
        capture_linear_state_rows=False,
        timing=False,
        trace=trace,
    )
    session.runner.weights.config.layer_types = (
        LINEAR_ATTENTION,
        "sliding_attention",
    )

    with pytest.raises(
        ValueError, match="unsupported GGUF layer type 'sliding_attention'"
    ):
        session.verify_target_blocks_batch([job])

    assert trace["ops"] == []


# ---------------------------------------------------------------------------
# runner boundary (RED until the inline packed loop moves into the owner)
# ---------------------------------------------------------------------------
def test_runner_packed_loop_delegates_to_the_layer_stack_owner() -> None:
    owner = _owner()
    assert gguf_runner.run_verify_layer_stack is owner.run_verify_layer_stack

    source = inspect.getsource(Qwen35GGUFResidentSession.verify_target_blocks_batch)
    assert "run_verify_layer_stack(" in source
    assert "_run_linear_attention_prefill_layer_rows" not in source
    assert "_run_moe_physical_c2_linear_rows_exact" not in source
    assert "_run_full_attention_decode_batch_layer_rows" not in source
    assert "for layer_id, layer_type in enumerate" not in source
