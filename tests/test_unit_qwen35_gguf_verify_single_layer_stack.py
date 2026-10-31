"""Single-session eager verifier layer-stack ownership and execution tests.

``Qwen35GGUFResidentSession.verify_target_block`` used to inline its
``for layer_id, layer_type in enumerate(layer_types)`` walk. The new
``qwen35_gguf_verify_single_layer_stack`` owner owns that walk: the per-layer
F32 residual-limit transition, the optional pre-timer synchronize, the
per-layer sidecar load/free lifetime, chunk-scratch selection, the four
selected operation call sites, the aggregate wall-clock stage marks and the
layer boundary/output captures.

These tests pin four things:

* the dependency boundary - the owner imports no runner, session, runtime or
  environment policy, its records are immutable, and its call sites spell every
  argument out;
* the execution contract - route dispatch, argument identity and keyword order,
  the BF16/F32 ping-pong, the F32 layer-limit transition, state commit and
  capture offsets;
* failure containment - sync, sidecar load, operation, sidecar free, lookup and
  capture failures happen at the same point and leave the same marks and
  partial mutation as the inline loop;
* the single-verifier control trace - the exact lookup/operation/stage/mutation
  sequence of ``verify_target_block`` across bulk/native routes, rows 1/3/6,
  F32 residual and its layer limit, explicit post-norm, captures, state commit,
  sidecar lifetime, prebuilt graph scratch and timing on/off.

No GPU, no ROCm runtime, no environment policy mutation outside the declared
verifier F32 layer-limit input.
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
from hipengine.runtime.qwen35_gguf_runner import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    Qwen35GGUFResidentSession,
)

from tests.test_unit_qwen35_gguf_verify_timing_teardown import _FakeEventRuntime

_OWNER_MODULE = "hipengine.runtime.qwen35_gguf_verify_single_layer_stack"


def _owner():
    return importlib.import_module(_OWNER_MODULE)


# ---------------------------------------------------------------------------
# owner boundary
# ---------------------------------------------------------------------------
def test_owner_module_reaches_no_runner_session_runtime_or_allocation() -> None:
    owner = _owner()
    tree = ast.parse(Path(owner.__file__).read_text())

    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert imported == {
        "__future__",
        "dataclasses",
        "time",
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
        "memcpy_async",
        "copy_host_array_to_device",
        "copy_device_to_host",
        "get_hip_runtime",
        "getattr",
        "setattr",
    }

    # Every operation call site spells its arguments out; no thunk is invoked
    # through a bag and no call forwards an unpacked mapping.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert not any(keyword.arg is None for keyword in node.keywords)
    assert "partial" not in names
    assert "functools" not in {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }


def test_owner_records_are_frozen_explicit_and_keyword_bounded() -> None:
    owner = _owner()
    assert [field.name for field in fields(owner.VerifySingleLayerGeometry)] == [
        "layer_types",
        "rows",
        "start",
        "end",
        "scratch_row_start",
        "row_nbytes",
        "stream",
        "f32_residual_layer_limit",
        "chain_dense_next_rms",
        "bulk_attention_mode",
        "capture_linear_state_rows",
        "defer_linear_state_commit",
        "capture_prefill_gdn",
        "verify_f32_post_norm",
        "use_f32_post_norm",
        "sync_stages",
        "sync_stage_timings",
        "decode_row_scratches",
        "attention_context_limit",
        "row_views_keep_scalar_caps",
        "capture_layer_ids",
        "capture_layer_boundary_ids",
    ]
    assert [field.name for field in fields(owner.VerifySingleLayerHidden)] == [
        "first",
        "first_ptr",
        "second",
        "second_ptr",
        "first_f32",
        "first_f32_ptr",
        "second_f32",
        "second_f32_ptr",
        "use_f32_residual",
    ]
    assert [field.name for field in fields(owner.VerifySingleLayerScratch)] == [
        "prebuilt",
        "resolve_chunk",
        "native_conv_out",
        "decode_scratch",
        "cos_table",
        "sin_table",
    ]
    assert [field.name for field in fields(owner.VerifySingleLayerSidecar)] == [
        "load",
        "free",
    ]
    assert [field.name for field in fields(owner.VerifySingleLayerCapture)] == [
        "boundary",
        "output",
    ]
    assert [field.name for field in fields(owner.VerifySingleLayerResult)] == [
        "hidden",
        "hidden_f32",
        "use_f32_residual",
        "layer_output_hidden",
        "layer_boundary_hidden",
    ]

    signature = inspect.signature(owner.run_verify_single_layer_stack)
    assert list(signature.parameters) == [
        "geometry",
        "hidden",
        "scratch",
        "sidecar",
        "capture",
        "run_native_attention_bulk_ffn_rows",
        "run_linear_attention_prefill_rows",
        "run_full_attention_decode_batch_rows",
        "run_full_attention_prefill_aotriton",
        "resolve_next_norm_weight",
        "resolve_linear_state_rows",
        "resolve_initial_state_snapshot",
        "resolve_full_attention_cache",
        "synchronize",
        "stage_timings",
    ]
    assert not any(
        parameter.kind is parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    geometry = _geometry(owner)
    with pytest.raises(FrozenInstanceError):
        geometry.rows = 4
    with pytest.raises(FrozenInstanceError):
        _hidden(owner).first_ptr = 0
    with pytest.raises(FrozenInstanceError):
        _scratch(owner).prebuilt = object()


# ---------------------------------------------------------------------------
# direct owner execution fakes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Buffer:
    label: str
    ptr: int
    nbytes: int = 0


@dataclass(frozen=True)
class _ChunkScratch:
    label: str
    start: int
    rows: int
    norm: _Buffer
    conv_out: object | None = None
    key_cache: object | None = None
    value_cache: object | None = None
    cos_table: object | None = None
    sin_table: object | None = None


class _Ops:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def _record(self, name, args, kwargs) -> None:
        self.calls.append({"op": name, "args": args, "kwargs": dict(kwargs)})

    def native(self, *args, **kwargs) -> None:
        self._record("native", args, kwargs)

    def linear(self, *args, **kwargs) -> None:
        self._record("linear", args, kwargs)

    def full_decode(self, *args, **kwargs) -> None:
        self._record("full_decode", args, kwargs)

    def full_prefill(self, *args, **kwargs) -> None:
        self._record("full_prefill", args, kwargs)


class _Resolvers:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []
        self.fail_at: str | None = None
        self.fail_layer: int | None = None
        self.initial_snapshot_users = 0

    def _check(self, kind: str, layer_id: int) -> None:
        self.events.append((kind, int(layer_id)))
        if self.fail_at == kind and (
            self.fail_layer is None or self.fail_layer == layer_id
        ):
            raise RuntimeError(f"{kind} boom {layer_id}")

    def chunk(self, start: int, rows: int, total_tokens: int) -> object:
        self.events.append(("chunk", int(start), int(rows), int(total_tokens)))
        if self.fail_at == "chunk":
            raise RuntimeError("chunk boom")
        return _ChunkScratch(
            label=f"chunk[{start}]",
            start=int(start),
            rows=int(rows),
            norm=_Buffer(f"norm[{start}]", 0x300 + int(start)),
        )

    def next_norm_weight(self, layer_id: int) -> int:
        self._check("next_norm_weight", layer_id)
        return 0x400 + int(layer_id)

    def linear_state_rows(self, layer_id: int):
        self._check("linear_state_rows", layer_id)
        return (
            _Buffer(f"conv_rows[{layer_id}]", 0x700 + int(layer_id)),
            _Buffer(f"recurrent_rows[{layer_id}]", 0x800 + int(layer_id)),
        )

    def initial_state_snapshot(self, layer_id: int):
        if self.initial_snapshot_users <= 0:
            return None
        self._check("initial_state_snapshot", layer_id)
        return (
            _Buffer(f"conv_snap[{layer_id}]", 0x900 + int(layer_id)),
            _Buffer(f"recurrent_snap[{layer_id}]", 0xA00 + int(layer_id)),
        )

    def full_cache(self, layer_id: int):
        self._check("full_cache", layer_id)
        return (
            _Buffer(f"key[{layer_id}]", 0xB00 + int(layer_id)),
            _Buffer(f"value[{layer_id}]", 0xC00 + int(layer_id)),
        )


class _Sidecar:
    def __init__(self, events: list[tuple[object, ...]], *, fail_free: bool = False):
        self.events = events
        self.fail_free = fail_free

    def free(self) -> None:
        self.events.append(("sidecar_free",))
        if self.fail_free:
            raise RuntimeError("free boom")


def _geometry(owner, **overrides):
    values = dict(
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
        rows=3,
        start=4,
        end=7,
        scratch_row_start=4,
        row_nbytes=16,
        stream=0,
        f32_residual_layer_limit=2,
        chain_dense_next_rms=False,
        bulk_attention_mode="bulk",
        capture_linear_state_rows=False,
        defer_linear_state_commit=False,
        capture_prefill_gdn=False,
        verify_f32_post_norm=None,
        use_f32_post_norm=False,
        sync_stages=False,
        sync_stage_timings=False,
        decode_row_scratches=None,
        attention_context_limit=None,
        row_views_keep_scalar_caps=False,
        capture_layer_ids=frozenset(),
        capture_layer_boundary_ids=frozenset(),
    )
    values.update(overrides)
    return owner.VerifySingleLayerGeometry(**values)


def _hidden(owner, **overrides):
    values = dict(
        first=_Buffer("a", 0xA0),
        first_ptr=0xA0,
        second=_Buffer("b", 0xB0),
        second_ptr=0xB0,
        first_f32=_Buffer("f32_a", 0x600),
        first_f32_ptr=0x600,
        second_f32=_Buffer("f32_b", 0x610),
        second_f32_ptr=0x610,
        use_f32_residual=False,
    )
    values.update(overrides)
    return owner.VerifySingleLayerHidden(**values)


def _scratch(owner, resolvers: _Resolvers | None = None, **overrides):
    resolvers = _Resolvers() if resolvers is None else resolvers
    values = dict(
        prebuilt=None,
        resolve_chunk=resolvers.chunk,
        native_conv_out=None,
        decode_scratch=_Buffer("decode", 0x310),
        cos_table=_Buffer("cos", 0x200),
        sin_table=_Buffer("sin", 0x201),
    )
    values.update(overrides)
    return owner.VerifySingleLayerScratch(**values)


def _sidecar(owner, load=None, free=None):
    return owner.VerifySingleLayerSidecar(
        load=load,
        free=free if free is not None else (lambda sidecar: sidecar.free()),
    )


class _Capture:
    def __init__(self) -> None:
        self.boundary_calls: list[dict[str, object]] = []
        self.output_calls: list[dict[str, object]] = []
        self.fail_at: str | None = None

    def boundary(self, layer_id, layer_type, **kwargs):
        self.boundary_calls.append(
            {"layer_id": int(layer_id), "layer_type": str(layer_type), **kwargs}
        )
        if self.fail_at == "boundary":
            raise RuntimeError("boundary boom")
        return {"hidden_in": np.zeros((2,), dtype=np.float32)}

    def output(self, layer_id, **kwargs):
        self.output_calls.append({"layer_id": int(layer_id), **kwargs})
        if self.fail_at == "output":
            raise RuntimeError("output boom")
        return np.zeros((2,), dtype=np.float32)


def _capture(owner, capture: _Capture | None = None):
    capture = _Capture() if capture is None else capture
    return owner.VerifySingleLayerCapture(
        boundary=capture.boundary,
        output=capture.output,
    )


def _run(
    owner,
    *,
    geometry,
    hidden,
    scratch,
    sidecar=None,
    capture=None,
    ops=None,
    resolvers=None,
    sync=None,
    stage_timings=None,
):
    ops = _Ops() if ops is None else ops
    resolvers = _Resolvers() if resolvers is None else resolvers
    sidecar = _sidecar(owner) if sidecar is None else sidecar
    capture = _capture(owner) if capture is None else capture
    sync = (lambda: None) if sync is None else sync
    result = owner.run_verify_single_layer_stack(
        geometry,
        hidden,
        scratch,
        sidecar,
        capture,
        ops.native,
        ops.linear,
        ops.full_decode,
        ops.full_prefill,
        resolve_next_norm_weight=resolvers.next_norm_weight,
        resolve_linear_state_rows=resolvers.linear_state_rows,
        resolve_initial_state_snapshot=resolvers.initial_state_snapshot,
        resolve_full_attention_cache=resolvers.full_cache,
        synchronize=sync,
        stage_timings=stage_timings,
    )
    return result, ops, resolvers, capture


# ---------------------------------------------------------------------------
# direct owner execution
# ---------------------------------------------------------------------------
def test_owner_native_route_arguments_and_order() -> None:
    owner = _owner()
    resolvers = _Resolvers()
    resolvers.initial_snapshot_users = 1
    result, ops, _, _ = _run(
        owner,
        geometry=_geometry(
            owner,
            layer_types=(LINEAR_ATTENTION, FULL_ATTENTION, LINEAR_ATTENTION),
            bulk_attention_mode="native",
            chain_dense_next_rms=True,
            capture_linear_state_rows=True,
            defer_linear_state_commit=True,
            decode_row_scratches=(_Buffer("row0", 0xD00), _Buffer("row1", 0xD01), _Buffer("row2", 0xD02)),
            attention_context_limit=32,
            row_views_keep_scalar_caps=True,
            verify_f32_post_norm=False,
        ),
        hidden=_hidden(owner),
        scratch=_scratch(
            owner,
            resolvers,
            native_conv_out=_Buffer("conv_out_f32", 0x500),
        ),
        resolvers=resolvers,
    )

    assert [call["op"] for call in ops.calls] == ["native", "native", "native"]
    first = ops.calls[0]
    assert first["args"] == (0, LINEAR_ATTENTION, 0xA0 + 4 * 16, 0xB0 + 4 * 16, _ChunkScratch(
        label="chunk[4]", start=4, rows=3, norm=_Buffer("norm[4]", 0x304),
        conv_out=_Buffer("conv_out_f32", 0x500),
    ))
    assert list(first["kwargs"]) == [
        "rows",
        "start_position",
        "stream",
        "decode_scratch",
        "linear_state_rows",
        "initial_state_snapshot",
        "commit_final_linear_state",
        "hidden_f32_ptr",
        "out_f32_ptr",
        "input_norm_ptr",
        "next_norm_weight_ptr",
        "next_norm_out_ptr",
        "decode_row_scratches",
        "attention_context_limit",
        "row_views_keep_scalar_caps",
        "verify_f32_post_norm",
    ]
    assert first["kwargs"]["rows"] == 3
    assert first["kwargs"]["start_position"] == 4
    assert first["kwargs"]["decode_scratch"] == _Buffer("decode", 0x310)
    assert first["kwargs"]["linear_state_rows"] == (
        _Buffer("conv_rows[0]", 0x700),
        _Buffer("recurrent_rows[0]", 0x800),
    )
    assert first["kwargs"]["initial_state_snapshot"] == (
        _Buffer("conv_snap[0]", 0x900),
        _Buffer("recurrent_snap[0]", 0xA00),
    )
    assert first["kwargs"]["commit_final_linear_state"] is False
    assert first["kwargs"]["input_norm_ptr"] is None
    assert first["kwargs"]["next_norm_weight_ptr"] == 0x401
    assert first["kwargs"]["next_norm_out_ptr"] == 0x304
    assert first["kwargs"]["decode_row_scratches"] == (
        _Buffer("row0", 0xD00), _Buffer("row1", 0xD01), _Buffer("row2", 0xD02),
    )
    assert first["kwargs"]["attention_context_limit"] == 32
    assert first["kwargs"]["row_views_keep_scalar_caps"] is True
    assert first["kwargs"]["verify_f32_post_norm"] is False

    # The next layer consumes the previous next-norm output, then the last
    # layer has no next-norm lookup.
    assert ops.calls[1]["kwargs"]["input_norm_ptr"] == 0x304
    assert ops.calls[1]["kwargs"]["next_norm_weight_ptr"] == 0x402
    assert ops.calls[2]["kwargs"]["input_norm_ptr"] == 0x304
    assert ops.calls[2]["kwargs"]["next_norm_weight_ptr"] is None
    assert ops.calls[2]["kwargs"]["next_norm_out_ptr"] is None

    # Ping-pong endpoints alternate and the result is the final source.
    assert [call["args"][2] for call in ops.calls] == [
        0xA0 + 64, 0xB0 + 64, 0xA0 + 64,
    ]
    assert [call["args"][3] for call in ops.calls] == [
        0xB0 + 64, 0xA0 + 64, 0xB0 + 64,
    ]
    assert result.hidden == _Buffer("b", 0xB0)
    assert resolvers.events[0] == ("chunk", 4, 3, 7)
    assert ("next_norm_weight", 1) in resolvers.events


def test_owner_linear_route_arguments_and_order() -> None:
    owner = _owner()
    resolvers = _Resolvers()
    sidecar = _Sidecar(resolvers.events)
    result, ops, _, _ = _run(
        owner,
        geometry=_geometry(
            owner,
            layer_types=(LINEAR_ATTENTION,),
            capture_linear_state_rows=True,
            capture_prefill_gdn=True,
            verify_f32_post_norm=True,
        ),
        hidden=_hidden(owner),
        scratch=_scratch(owner, resolvers),
        sidecar=_sidecar(owner, load=lambda layer_id: sidecar),
        resolvers=resolvers,
    )

    assert len(ops.calls) == 1
    call = ops.calls[0]
    assert call["args"] == (0, 0xA0 + 64, 0xB0 + 64, _ChunkScratch(
        label="chunk[4]", start=4, rows=3, norm=_Buffer("norm[4]", 0x304),
    ))
    assert list(call["kwargs"]) == [
        "rows",
        "stream",
        "decode_scratch",
        "expert_sidecar",
        "linear_state_rows",
        "prefill_gdn_capture",
        "commit_final_linear_state",
        "hidden_f32_ptr",
        "out_f32_ptr",
        "stage_timings",
        "sync_stage_timings",
        "stage_prefix",
        "verify_f32_post_norm",
    ]
    assert call["kwargs"]["expert_sidecar"] is sidecar
    assert call["kwargs"]["prefill_gdn_capture"] is True
    assert call["kwargs"]["commit_final_linear_state"] is True
    assert call["kwargs"]["stage_prefix"] == "target_block_linear_attn"
    assert call["kwargs"]["verify_f32_post_norm"] is True
    assert result.hidden == _Buffer("b", 0xB0)


def test_owner_full_decode_and_prefill_routes() -> None:
    owner = _owner()

    decode_result, decode_ops, _, _ = _run(
        owner,
        geometry=_geometry(owner, layer_types=(FULL_ATTENTION,), end=7),
        hidden=_hidden(owner),
        scratch=_scratch(owner),
    )
    assert [call["op"] for call in decode_ops.calls] == ["full_decode"]
    decode = decode_ops.calls[0]
    assert decode["args"] == (
        0,
        0xA0 + 64,
        0xB0 + 64,
        _ChunkScratch(
            label="chunk[4]",
            start=4,
            rows=3,
            norm=_Buffer("norm[4]", 0x304),
            key_cache=_Buffer("key[0]", 0xB00),
            value_cache=_Buffer("value[0]", 0xC00),
            cos_table=_Buffer("cos", 0x200),
            sin_table=_Buffer("sin", 0x201),
        ),
    )
    assert list(decode["kwargs"]) == [
        "stream",
        "expert_sidecar",
        "stage_timings",
        "sync_stage_timings",
        "stage_prefix",
        "hidden_f32_ptr",
        "out_f32_ptr",
        "verify_f32_post_norm",
    ]
    assert decode["kwargs"]["stage_prefix"] == "target_block_full_attn"
    assert decode_result.hidden == _Buffer("b", 0xB0)

    prefill_result, prefill_ops, _, _ = _run(
        owner,
        geometry=_geometry(
            owner, layer_types=(FULL_ATTENTION,), start=1020, end=1026
        ),
        hidden=_hidden(owner),
        scratch=_scratch(owner),
    )
    assert [call["op"] for call in prefill_ops.calls] == ["full_prefill"]
    prefill = prefill_ops.calls[0]
    assert list(prefill["kwargs"]) == [
        "stream",
        "expert_sidecar",
        "verify_f32_post_norm",
    ]
    assert prefill_result.hidden == _Buffer("b", 0xB0)

    # F32 residual is decode-batch-only; prefill must reject it at the old
    # point, after the scratch and cache lookups.
    with pytest.raises(RuntimeError, match="context < 1024"):
        _run(
            owner,
            geometry=_geometry(
                owner,
                layer_types=(FULL_ATTENTION,),
                start=1020,
                end=1026,
            ),
            hidden=_hidden(owner, use_f32_residual=True),
            scratch=_scratch(owner),
        )


@pytest.mark.parametrize("rows", [1, 3, 6])
def test_owner_preserves_rows_on_every_route(rows: int) -> None:
    owner = _owner()
    result, ops, _, _ = _run(
        owner,
        geometry=_geometry(owner, rows=rows, layer_types=(LINEAR_ATTENTION,)),
        hidden=_hidden(owner),
        scratch=_scratch(owner),
    )
    assert ops.calls[0]["kwargs"]["rows"] == rows
    assert result.hidden == _Buffer("b", 0xB0)


def test_owner_f32_layer_limit_transitions_before_sync_and_timer() -> None:
    owner = _owner()
    events: list[object] = []
    resolvers = _Resolvers()
    timings: dict[str, float] = {}

    def sync() -> None:
        events.append("sync")

    result, ops, _, _ = _run(
        owner,
        geometry=_geometry(
            owner,
            layer_types=(LINEAR_ATTENTION, FULL_ATTENTION, LINEAR_ATTENTION),
            f32_residual_layer_limit=2,
            sync_stages=True,
        ),
        hidden=_hidden(owner, use_f32_residual=True),
        scratch=_scratch(owner, resolvers),
        resolvers=resolvers,
        sync=sync,
        stage_timings=timings,
    )

    assert events == ["sync", "sync", "sync"]
    assert [call["kwargs"]["hidden_f32_ptr"] for call in ops.calls] == [
        0x600, 0x610, None,
    ]
    assert [call["kwargs"]["out_f32_ptr"] for call in ops.calls] == [
        0x610, 0x600, None,
    ]
    assert result.hidden_f32 is None
    assert result.use_f32_residual is False
    assert list(timings) == [
        "target_block_linear_attn_layers",
        "target_block_layer_total",
        "target_block_full_attn_layers",
    ]


def test_owner_keeps_explicit_and_resolved_post_norm_separate() -> None:
    owner = _owner()
    capture = _Capture()
    _, ops, _, _ = _run(
        owner,
        geometry=_geometry(
            owner,
            layer_types=(FULL_ATTENTION,),
            verify_f32_post_norm=False,
            use_f32_post_norm=True,
            capture_layer_boundary_ids=frozenset({0}),
        ),
        hidden=_hidden(owner),
        scratch=_scratch(owner),
        capture=_capture(owner, capture),
    )
    assert ops.calls[0]["kwargs"]["verify_f32_post_norm"] is False
    assert capture.boundary_calls[0]["verify_f32_post_norm"] is True
    assert capture.boundary_calls[0]["prefill_gdn_capture"] is False


def test_owner_capture_offsets_boundary_uses_scratch_row_start_output_uses_start() -> None:
    owner = _owner()
    capture = _Capture()
    _, ops, _, _ = _run(
        owner,
        geometry=_geometry(
            owner,
            layer_types=(FULL_ATTENTION,),
            start=8,
            end=11,
            scratch_row_start=0,
            capture_layer_ids=frozenset({0}),
            capture_layer_boundary_ids=frozenset({0}),
        ),
        hidden=_hidden(owner),
        scratch=_scratch(owner),
        capture=_capture(owner, capture),
    )
    boundary = capture.boundary_calls[0]
    assert boundary["hidden_in_ptr"] == 0xA0
    assert boundary["layer_out_ptr"] == 0xB0
    assert boundary["hidden_in_f32_ptr"] is None
    assert boundary["layer_out_f32_ptr"] is None
    assert boundary["scratch"].label == "chunk[8]"
    # The boundary capture receives the base chunk view, not the full-attention
    # layer view that only the selected operation consumes.
    assert boundary["scratch"].key_cache is None

    output = capture.output_calls[0]
    # The output capture receives the buffer; its owner binds the absolute
    # start offset (8 * 16), not the scratch row start.
    assert output["hidden"] == _Buffer("b", 0xB0)
    assert output["hidden"].ptr + 8 * 16 == 0xB0 + 128
    assert output["hidden_f32"] is None
    assert output["use_f32_residual"] is False
    assert ops.calls[0]["args"][1] == 0xA0


def test_owner_state_commit_and_snapshot_lookup_order() -> None:
    owner = _owner()
    resolvers = _Resolvers()
    resolvers.initial_snapshot_users = 1
    _, ops, _, _ = _run(
        owner,
        geometry=_geometry(
            owner,
            layer_types=(LINEAR_ATTENTION,),
            bulk_attention_mode="native",
            capture_linear_state_rows=True,
            defer_linear_state_commit=True,
        ),
        hidden=_hidden(owner),
        scratch=_scratch(owner, resolvers),
        resolvers=resolvers,
    )
    call = ops.calls[0]
    assert call["kwargs"]["linear_state_rows"] == (
        _Buffer("conv_rows[0]", 0x700),
        _Buffer("recurrent_rows[0]", 0x800),
    )
    assert call["kwargs"]["initial_state_snapshot"] == (
        _Buffer("conv_snap[0]", 0x900),
        _Buffer("recurrent_snap[0]", 0xA00),
    )
    assert call["kwargs"]["commit_final_linear_state"] is False
    # The state-row lookup precedes the snapshot lookup at the old point.
    assert resolvers.events.index(("linear_state_rows", 0)) < resolvers.events.index(
        ("initial_state_snapshot", 0)
    )

    # Zero snapshot users keeps the state rows but skips the snapshot lookup.
    resolvers = _Resolvers()
    _, ops, _, _ = _run(
        owner,
        geometry=_geometry(
            owner,
            layer_types=(LINEAR_ATTENTION,),
            bulk_attention_mode="native",
            capture_linear_state_rows=True,
            defer_linear_state_commit=True,
        ),
        hidden=_hidden(owner),
        scratch=_scratch(owner, resolvers),
        resolvers=resolvers,
    )
    assert ops.calls[0]["kwargs"]["initial_state_snapshot"] is None
    assert ("initial_state_snapshot", 0) not in resolvers.events


def test_owner_prebuilt_scratch_skips_chunk_resolution_and_replaces_native_conv_out() -> None:
    owner = _owner()
    resolvers = _Resolvers()
    prebuilt = _ChunkScratch(
        label="prebuilt", start=0, rows=3, norm=_Buffer("prebuilt_norm", 0x340)
    )
    _, ops, _, _ = _run(
        owner,
        geometry=_geometry(owner, layer_types=(LINEAR_ATTENTION,), bulk_attention_mode="native"),
        hidden=_hidden(owner),
        scratch=_scratch(
            owner,
            resolvers,
            prebuilt=prebuilt,
            native_conv_out=_Buffer("conv_out_f32", 0x500),
        ),
        resolvers=resolvers,
    )
    assert ("chunk", 4, 3, 7) not in resolvers.events
    assert ops.calls[0]["args"][4].label == "prebuilt"
    assert ops.calls[0]["args"][4].conv_out == _Buffer("conv_out_f32", 0x500)


# ---------------------------------------------------------------------------
# direct owner failure containment
# ---------------------------------------------------------------------------
def test_owner_sync_failure_happens_before_sidecar_timer_and_operation() -> None:
    owner = _owner()
    events: list[object] = []
    timings: dict[str, float] = {}

    def sync() -> None:
        events.append("sync")
        raise RuntimeError("sync boom")

    with pytest.raises(RuntimeError, match="sync boom"):
        _run(
            owner,
            geometry=_geometry(
                owner,
                layer_types=(LINEAR_ATTENTION,),
                sync_stages=True,
            ),
            hidden=_hidden(owner),
            scratch=_scratch(owner),
            sidecar=_sidecar(owner, load=lambda layer_id: events.append(("load", layer_id))),
            sync=sync,
            stage_timings=timings,
        )
    assert events == ["sync"]
    assert timings == {}


def test_owner_chunk_resolution_failure_is_inside_try_and_records_marks() -> None:
    owner = _owner()
    resolvers = _Resolvers()
    resolvers.fail_at = "chunk"
    timings: dict[str, float] = {}
    ops = _Ops()
    with pytest.raises(RuntimeError, match="chunk boom"):
        _run(
            owner,
            geometry=_geometry(owner, layer_types=(LINEAR_ATTENTION,)),
            hidden=_hidden(owner),
            scratch=_scratch(owner, resolvers),
            resolvers=resolvers,
            ops=ops,
            stage_timings=timings,
        )
    assert ops.calls == []
    assert list(timings) == [
        "target_block_linear_attn_layers",
        "target_block_layer_total",
    ]


def test_owner_sidecar_load_failure_does_not_enter_try_or_mark() -> None:
    owner = _owner()
    events: list[tuple[object, ...]] = []
    timings: dict[str, float] = {}

    def load(layer_id: int) -> object:
        events.append(("load", layer_id))
        raise RuntimeError("load boom")

    with pytest.raises(RuntimeError, match="load boom"):
        _run(
            owner,
            geometry=_geometry(owner, layer_types=(LINEAR_ATTENTION,)),
            hidden=_hidden(owner),
            scratch=_scratch(owner),
            sidecar=_sidecar(owner, load=load),
            stage_timings=timings,
        )
    assert events == [("load", 0)]
    assert timings == {}


def test_owner_operation_failure_frees_sidecar_first_then_records_marks() -> None:
    owner = _owner()
    events: list[tuple[object, ...]] = []
    timings: dict[str, float] = {}
    sidecar = _Sidecar(events)

    def failing_linear(*args, **kwargs):
        events.append(("op",))
        raise RuntimeError("op boom")

    ops = _Ops()
    ops.linear = failing_linear
    with pytest.raises(RuntimeError, match="op boom"):
        _run(
            owner,
            geometry=_geometry(owner, layer_types=(LINEAR_ATTENTION,)),
            hidden=_hidden(owner),
            scratch=_scratch(owner),
            sidecar=_sidecar(owner, load=lambda layer_id: sidecar),
            ops=ops,
            stage_timings=timings,
        )
    assert events == [("op",), ("sidecar_free",)]
    assert list(timings) == [
        "target_block_linear_attn_layers",
        "target_block_layer_total",
    ]


def test_owner_sidecar_free_failure_suppresses_the_stage_marks() -> None:
    owner = _owner()
    events: list[tuple[object, ...]] = []
    timings: dict[str, float] = {}
    sidecar = _Sidecar(events, fail_free=True)

    with pytest.raises(RuntimeError, match="free boom"):
        _run(
            owner,
            geometry=_geometry(owner, layer_types=(LINEAR_ATTENTION,)),
            hidden=_hidden(owner),
            scratch=_scratch(owner),
            sidecar=_sidecar(owner, load=lambda layer_id: sidecar),
            stage_timings=timings,
        )
    assert events == [("sidecar_free",)]
    assert timings == {}


def test_owner_boundary_capture_runs_after_marks_and_blocks_output_capture() -> None:
    owner = _owner()
    capture = _Capture()
    capture.fail_at = "boundary"
    timings: dict[str, float] = {}

    with pytest.raises(RuntimeError, match="boundary boom"):
        _run(
            owner,
            geometry=_geometry(
                owner,
                layer_types=(FULL_ATTENTION,),
                capture_layer_ids=frozenset({0}),
                capture_layer_boundary_ids=frozenset({0}),
            ),
            hidden=_hidden(owner),
            scratch=_scratch(owner),
            capture=_capture(owner, capture),
            stage_timings=timings,
        )
    assert list(timings) == [
        "target_block_full_attn_layers",
        "target_block_layer_total",
    ]
    assert capture.output_calls == []


def test_owner_output_capture_failure_happens_after_the_pingpong_swap() -> None:
    owner = _owner()
    capture = _Capture()
    capture.fail_at = "output"
    timings: dict[str, float] = {}

    with pytest.raises(RuntimeError, match="output boom"):
        _run(
            owner,
            geometry=_geometry(
                owner,
                layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
                capture_layer_ids=frozenset({1}),
            ),
            hidden=_hidden(owner),
            scratch=_scratch(owner),
            capture=_capture(owner, capture),
            stage_timings=timings,
        )
    assert len(capture.output_calls) == 1
    assert capture.output_calls[0]["layer_id"] == 1
    assert capture.output_calls[0]["hidden"] == _Buffer("a", 0xA0)
    assert list(timings) == [
        "target_block_linear_attn_layers",
        "target_block_layer_total",
        "target_block_full_attn_layers",
    ]


def test_owner_late_lookup_failure_after_prior_layer_mutation() -> None:
    owner = _owner()
    resolvers = _Resolvers()
    resolvers.fail_at = "next_norm_weight"
    resolvers.fail_layer = 2
    timings: dict[str, float] = {}
    ops = _Ops()
    with pytest.raises(RuntimeError, match="next_norm_weight boom 2"):
        _run(
            owner,
            geometry=_geometry(
                owner,
                layer_types=(LINEAR_ATTENTION, FULL_ATTENTION, LINEAR_ATTENTION),
                chain_dense_next_rms=True,
            ),
            hidden=_hidden(owner),
            scratch=_scratch(owner, resolvers),
            resolvers=resolvers,
            ops=ops,
            stage_timings=timings,
        )
    assert [call["op"] for call in ops.calls] == ["linear"]
    # The failing lookup is inside the try, so the second layer still records
    # its family and total marks; its operation never runs.
    assert list(timings) == [
        "target_block_linear_attn_layers",
        "target_block_layer_total",
        "target_block_full_attn_layers",
    ]
    assert resolvers.events[:4] == [
        ("chunk", 4, 3, 7),
        ("next_norm_weight", 1),
        ("chunk", 4, 3, 7),
        ("next_norm_weight", 2),
    ]


def test_owner_unknown_layer_type_raises_only_when_reached() -> None:
    owner = _owner()
    ops = _Ops()
    timings: dict[str, float] = {}
    with pytest.raises(ValueError, match="unsupported GGUF layer type 'sliding_attention'"):
        _run(
            owner,
            geometry=_geometry(
                owner,
                layer_types=(LINEAR_ATTENTION, "sliding_attention"),
            ),
            hidden=_hidden(owner),
            scratch=_scratch(owner),
            ops=ops,
            stage_timings=timings,
        )
    assert [call["op"] for call in ops.calls] == ["linear"]
    assert list(timings) == [
        "target_block_linear_attn_layers",
        "target_block_layer_total",
        "target_block_other_layers",
    ]


# ---------------------------------------------------------------------------
# single-verifier control harness (old 8904 inline loop vs the owner)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Case:
    name: str
    layer_types: tuple[str, ...] = (LINEAR_ATTENTION, FULL_ATTENTION)
    rows: int = 3
    start: int = 4
    max_positions: int = 2048
    hidden_capacity_rows: int = 16
    mode: str = "bulk"
    is_moe: bool = False
    use_sidecar: bool = False
    wmma: bool = False
    f32_residual: bool = False
    f32_limit: int | None = None
    post_norm_arg: bool | None = None
    session_post_norm: bool = False
    capture_linear: bool = False
    defer_linear: bool = False
    initial_snapshot_users: int = 0
    capture_output: tuple[int, ...] = ()
    capture_boundary: tuple[int, ...] = ()
    timing: bool = False
    sync: bool = False
    chain_next_rms: bool = False
    native_row_scratches: bool = False
    prebuilt: bool = False
    stream: int = 0


_CASES = (
    _Case(name="bulk-r1", rows=1, start=0, hidden_capacity_rows=8),
    _Case(
        name="bulk-r3-timing",
        rows=3,
        start=4,
        timing=True,
        sync=True,
    ),
    _Case(
        name="bulk-r6-prefill",
        rows=6,
        start=1020,
        hidden_capacity_rows=8,
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION, LINEAR_ATTENTION),
        timing=True,
    ),
    _Case(
        name="native-r3",
        rows=3,
        start=4,
        mode="native",
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION, LINEAR_ATTENTION, FULL_ATTENTION),
        timing=True,
    ),
    _Case(
        name="native-r6-chain",
        rows=6,
        start=4,
        mode="native",
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION, LINEAR_ATTENTION),
        chain_next_rms=True,
        native_row_scratches=True,
        timing=True,
    ),
    _Case(
        name="native-r3-capture-nodefer",
        rows=3,
        start=4,
        mode="native",
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
        capture_linear=True,
        defer_linear=False,
        initial_snapshot_users=1,
        timing=True,
    ),
    _Case(
        name="native-r6-state",
        rows=6,
        start=4,
        mode="native",
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
        capture_linear=True,
        defer_linear=True,
        initial_snapshot_users=1,
        timing=True,
    ),
    _Case(
        name="bulk-r6-f32-on",
        rows=6,
        start=4,
        f32_residual=True,
        f32_limit=2,
        post_norm_arg=True,
        timing=True,
    ),
    _Case(
        name="bulk-r6-f32-limit",
        rows=6,
        start=4,
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION, LINEAR_ATTENTION, FULL_ATTENTION),
        f32_residual=True,
        f32_limit=2,
        post_norm_arg=False,
        session_post_norm=True,
        capture_output=(1, 2),
        capture_boundary=(1, 2),
        timing=True,
    ),
    _Case(
        name="bulk-r3-capture-offset",
        rows=3,
        start=8,
        hidden_capacity_rows=10,
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
        capture_output=(0, 1),
        capture_boundary=(0, 1),
        timing=True,
    ),
    _Case(
        name="bulk-r6-state",
        rows=6,
        start=4,
        capture_linear=True,
        defer_linear=True,
        initial_snapshot_users=1,
        timing=True,
    ),
    _Case(
        name="moe-r3-sidecar",
        rows=3,
        start=4,
        is_moe=True,
        use_sidecar=True,
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
        timing=True,
    ),
    _Case(
        name="prebuilt-r3",
        rows=3,
        start=4,
        prebuilt=True,
        stream=7,
        layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
    ),
)


class _TraceRuntime(_FakeEventRuntime):
    def __init__(self, trace: dict[str, object]) -> None:
        super().__init__()
        self._trace = trace

    def device_synchronize(self) -> None:
        self._trace["events"].append({"kind": "sync"})


@dataclass
class _SessionScratch:
    label: str
    max_positions: int
    hidden_seed_fp32: _Buffer
    cos_table: _Buffer
    sin_table: _Buffer
    position_buf: _Buffer
    context_buf: _Buffer
    position_host: list[int]
    context_host: list[int]
    layer_conv_states: tuple
    layer_recurrent_states: tuple
    full_caches: dict
    trace: dict[str, object]

    def full_cache(self, layer_id: int):
        self.trace["events"].append({"kind": "full_cache", "layer": int(layer_id)})
        return self.full_caches[int(layer_id)]


class _BulkScratch:
    def __init__(self, rows: int, trace: dict[str, object]) -> None:
        self.rows = rows
        self._trace = trace

    def for_chunk(self, start, rows, total_tokens, *, runtime, stream=0):
        self._trace["events"].append(
            {
                "kind": "chunk",
                "start": int(start),
                "rows": int(rows),
                "total": int(total_tokens),
                "stream": int(stream),
            }
        )
        return _ChunkScratch(
            label=f"chunk[{int(start)}]",
            start=int(start),
            rows=int(rows),
            norm=_Buffer(f"norm[{int(start)}]", 0x300 + int(start)),
        )


class _Weights:
    def __init__(
        self,
        layer_types: tuple[str, ...],
        *,
        is_moe: bool,
        trace: dict[str, object],
    ) -> None:
        self.config = SimpleNamespace(
            layer_types=layer_types,
            is_moe=is_moe,
            rms_norm_eps=1e-6,
        )
        self._trace = trace

    def layer(self, layer_id: int):
        self._trace["events"].append(
            {"kind": "weight_lookup", "layer": int(layer_id)}
        )
        return SimpleNamespace(
            weight=lambda name: SimpleNamespace(
                allocation=lambda: SimpleNamespace(
                    tensor=_Buffer(
                        f"weight[{int(layer_id)}].{name}",
                        0x400 + int(layer_id) * 0x10,
                    )
                )
            )
        )

    def root(self, name: str):
        return SimpleNamespace(
            allocation=lambda: SimpleNamespace(
                tensor=_Buffer(f"root.{name}", 0x480)
            )
        )


def _labels(value):
    if value is None:
        return None
    return [getattr(item, "label", None) for item in value]


def _record_native(trace):
    def record(
        layer_id,
        layer_type,
        hidden_ptr,
        out_ptr,
        scratch,
        *,
        rows,
        decode_scratch,
        start_position=0,
        linear_state_rows=None,
        initial_state_snapshot=None,
        commit_final_linear_state=True,
        hidden_f32_ptr=None,
        out_f32_ptr=None,
        input_norm_ptr=None,
        next_norm_weight_ptr=None,
        next_norm_out_ptr=None,
        decode_row_scratches=None,
        attention_context_limit=None,
        row_views_keep_scalar_caps=False,
        stream=0,
        verify_f32_post_norm=None,
    ) -> None:
        trace["events"].append(
            {
                "kind": "op",
                "op": "native",
                "layer": int(layer_id),
                "layer_type": str(layer_type),
                "hidden_in": int(hidden_ptr),
                "hidden_out": int(out_ptr),
                "scratch": scratch.label,
                "scratch_start": int(scratch.start),
                "scratch_rows": int(scratch.rows),
                "scratch_norm": int(scratch.norm.ptr),
                "scratch_conv_out": (
                    None if scratch.conv_out is None else scratch.conv_out.label
                ),
                "rows": int(rows),
                "start_position": int(start_position),
                "decode_scratch": decode_scratch.label,
                "linear_state_rows": _labels(linear_state_rows),
                "initial_state_snapshot": _labels(initial_state_snapshot),
                "commit_final_linear_state": bool(commit_final_linear_state),
                "hidden_f32_ptr": hidden_f32_ptr,
                "out_f32_ptr": out_f32_ptr,
                "input_norm_ptr": input_norm_ptr,
                "next_norm_weight_ptr": next_norm_weight_ptr,
                "next_norm_out_ptr": next_norm_out_ptr,
                "decode_row_scratch_count": (
                    None if decode_row_scratches is None else len(decode_row_scratches)
                ),
                "attention_context_limit": attention_context_limit,
                "row_views_keep_scalar_caps": bool(row_views_keep_scalar_caps),
                "stream": int(stream),
                "verify_f32_post_norm": verify_f32_post_norm,
            }
        )

    return record


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
        prefill_gdn_capture=None,
        commit_final_linear_state=True,
        hidden_f32_ptr=None,
        out_f32_ptr=None,
        stage_timings=None,
        sync_stage_timings=False,
        stage_prefix="target_block_linear_attn",
        gpu_stage_recorder=None,
        verify_f32_post_norm=None,
    ) -> None:
        trace["events"].append(
            {
                "kind": "op",
                "op": "linear",
                "layer": int(layer_id),
                "hidden_in": int(hidden_ptr),
                "hidden_out": int(out_ptr),
                "scratch": scratch.label,
                "scratch_start": int(scratch.start),
                "scratch_rows": int(scratch.rows),
                "scratch_norm": int(scratch.norm.ptr),
                "scratch_conv_out": (
                    None if scratch.conv_out is None else scratch.conv_out.label
                ),
                "rows": int(rows),
                "decode_scratch": decode_scratch.label,
                "expert_sidecar": (
                    None if expert_sidecar is None else int(expert_sidecar.layer_id)
                ),
                "linear_state_rows": _labels(linear_state_rows),
                "prefill_gdn_capture": prefill_gdn_capture,
                "commit_final_linear_state": bool(commit_final_linear_state),
                "hidden_f32_ptr": hidden_f32_ptr,
                "out_f32_ptr": out_f32_ptr,
                "has_stage_timings": stage_timings is not None,
                "sync_stage_timings": bool(sync_stage_timings),
                "stage_prefix": stage_prefix,
                "has_gpu_stage_recorder": gpu_stage_recorder is not None,
                "stream": int(stream),
                "verify_f32_post_norm": verify_f32_post_norm,
            }
        )

    return record


def _record_full(trace, *, op: str):
    def record(
        layer_id,
        hidden_ptr,
        out_ptr,
        scratch,
        *,
        stream=0,
        expert_sidecar=None,
        stage_timings=None,
        sync_stage_timings=False,
        stage_prefix="target_block_full_attn",
        hidden_f32_ptr=None,
        out_f32_ptr=None,
        verify_f32_post_norm=None,
        **kwargs,
    ) -> None:
        trace["events"].append(
            {
                "kind": "op",
                "op": op,
                "layer": int(layer_id),
                "hidden_in": int(hidden_ptr),
                "hidden_out": int(out_ptr),
                "scratch": scratch.label,
                "scratch_start": int(scratch.start),
                "scratch_rows": int(scratch.rows),
                "scratch_norm": int(scratch.norm.ptr),
                "scratch_conv_out": (
                    None if scratch.conv_out is None else scratch.conv_out.label
                ),
                "key_cache": (
                    None if scratch.key_cache is None else scratch.key_cache.label
                ),
                "value_cache": (
                    None if scratch.value_cache is None else scratch.value_cache.label
                ),
                "cos_table": (
                    None if scratch.cos_table is None else scratch.cos_table.label
                ),
                "sin_table": (
                    None if scratch.sin_table is None else scratch.sin_table.label
                ),
                "expert_sidecar": (
                    None if expert_sidecar is None else int(expert_sidecar.layer_id)
                ),
                "has_stage_timings": stage_timings is not None,
                "sync_stage_timings": bool(sync_stage_timings),
                "stage_prefix": stage_prefix,
                "hidden_f32_ptr": hidden_f32_ptr,
                "out_f32_ptr": out_f32_ptr,
                "stream": int(stream),
                "verify_f32_post_norm": verify_f32_post_norm,
            }
        )

    return record


def _make_single_session(
    monkeypatch: pytest.MonkeyPatch,
    case: _Case,
    trace: dict[str, object],
    *,
    snapshot_users_override: object | None = None,
):
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_F32_TOKEN_EMBEDDING", raising=False)
    if case.f32_limit is None:
        monkeypatch.delenv(
            "HIPENGINE_GGUF_VERIFY_F32_RESIDUAL_LAYER_LIMIT", raising=False
        )
    else:
        monkeypatch.setenv(
            "HIPENGINE_GGUF_VERIFY_F32_RESIDUAL_LAYER_LIMIT", str(case.f32_limit)
        )

    monkeypatch.setattr(gguf_runner, "launch_gguf_embedding", lambda *a, **k: None)
    monkeypatch.setattr(gguf_runner, "bf16_to_f32", lambda *a, **k: None)
    monkeypatch.setattr(gguf_runner, "copy_host_array_to_device", lambda *a, **k: None)
    monkeypatch.setattr(
        gguf_runner, "gguf_wmma_prefill_enabled", lambda value: bool(value)
    )
    monkeypatch.setattr(
        gguf_runner, "wmma_prefill_session", lambda *a, **k: nullcontext()
    )
    monkeypatch.setattr(
        gguf_runner, "gemv_decode_session", lambda *a, **k: nullcontext()
    )
    monkeypatch.setattr(
        gguf_runner, "native_batch_decode_session", lambda *a, **k: nullcontext()
    )
    for name in (
        "gguf_rmsnorm_f32_f32_weight",
        "gguf_rmsnorm_f32_f32_weight_out_f32",
        "gguf_rmsnorm_bf16_f32_weight",
        "gguf_rmsnorm_bf16_f32_weight_out_f32",
        "set_decode_position_i64",
        "advance_decode_position_i64",
        "copy_i32_to_i64",
        "record_i64_scalar_indexed",
    ):
        monkeypatch.setattr(gguf_runner, name, lambda *a, **k: None)
    monkeypatch.setattr(gguf_runner, "copy_device_to_host", _copy_zero)
    monkeypatch.setattr(gguf_runner, "host_array_ptr", lambda array: array)
    monkeypatch.setattr(
        gguf_runner, "_gguf_verify_row_lm_head_enabled", lambda: True
    )
    monkeypatch.setattr(
        gguf_runner, "_gguf_verify_lm_head_q6_top1_dp4a_enabled", lambda: False
    )
    monkeypatch.setattr(
        gguf_runner, "_gguf_verify_f32_token_embedding_enabled", lambda: False
    )
    monkeypatch.setattr(
        gguf_runner,
        "_copy_f32_ptr_to_host",
        _record_output_capture(trace, "f32"),
    )
    monkeypatch.setattr(
        gguf_runner,
        "_copy_bf16_rows_to_host_f32",
        _record_output_capture(trace, "bf16"),
    )

    weights = _Weights(
        tuple(case.layer_types), is_moe=case.is_moe, trace=trace
    )
    runner = SimpleNamespace(
        weights=weights,
        hidden_size=8,
        vocab_size=128,
        _rounded_add_rmsnorm_fn=(
            (lambda rows: object()) if case.chain_next_rms else (lambda rows: None)
        ),
        _cast_library=lambda: object(),
    )
    runner._run_native_attention_bulk_ffn_layer_rows = _record_native(trace)
    runner._run_linear_attention_prefill_layer_rows = _record_linear(trace)
    runner._run_full_attention_decode_batch_layer_rows = _record_full(
        trace, op="full_decode"
    )
    runner._run_full_attention_prefill_layer_aotriton = _record_full(
        trace, op="full_prefill"
    )

    runtime = _TraceRuntime(trace)
    row_nbytes = 8 * DType.BF16.itemsize
    layer_count = len(case.layer_types)
    full_caches = {
        index: (
            _Buffer(f"key[{index}]", 0xB00 + index),
            _Buffer(f"value[{index}]", 0xC00 + index),
        )
        for index in range(layer_count)
    }
    scratch = _SessionScratch(
        label="decode",
        max_positions=case.max_positions,
        hidden_seed_fp32=_Buffer("hidden_seed_fp32", 0x101),
        cos_table=_Buffer("cos", 0x200),
        sin_table=_Buffer("sin", 0x201),
        position_buf=_Buffer("position", 0x103),
        context_buf=_Buffer("context", 0x104),
        position_host=[case.start],
        context_host=[case.start + 1],
        layer_conv_states=tuple(
            _Buffer(f"conv[{index}]", 0x500 + index) for index in range(layer_count)
        ),
        layer_recurrent_states=tuple(
            _Buffer(f"recurrent[{index}]", 0x600 + index)
            for index in range(layer_count)
        ),
        full_caches=full_caches,
        trace=trace,
    )
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = runner
    session.scratch = scratch
    session.runtime = runtime
    session._position = case.start
    session._prefill_token_buf = _Buffer("tokens", 0x102)
    session._prefill_hidden_a = _Buffer(
        "hidden_a", 0xA0, case.hidden_capacity_rows * row_nbytes
    )
    session._prefill_hidden_b = _Buffer(
        "hidden_b", 0xB0, case.hidden_capacity_rows * row_nbytes
    )
    session._verify_hidden_seed_buf = _Buffer("hidden_seed", 0x105)
    session._verify_hidden_f32_a = _Buffer("f32_a", 0x600)
    session._verify_hidden_f32_b = _Buffer("f32_b", 0x610)
    session._bulk_prefill_scratch = _BulkScratch(
        max(case.hidden_capacity_rows, case.rows), trace
    )
    session._verify_token_ids_i64 = _Buffer("token_ids_i64", 0x106)
    session._verify_token_counter_i64 = _Buffer("token_counter_i64", 0x107)
    session._verify_conv_out_f32 = (
        _Buffer("conv_out_f32", 0x508) if case.mode == "native" else None
    )
    session._verify_lm_out_indices_i32 = _Buffer("lm_out_i32", 0x520)
    session._lm_out_index = _Buffer("lm_index", 0x521)
    session._verify_logits_buf = None
    session._runtime_state_library = object()
    session.use_expert_sidecar = case.use_sidecar
    session.use_gemv_decode = False
    session.use_wmma_prefill = case.wmma
    session.verifier_prefill_gdn_capture = False
    session.last_verify_stage_timings_ms = {}
    session._hidden_seed_fp32_populated = False
    session._verify_hidden_seed_rows_populated = 0
    session._verify_linear_conv_state_rows = tuple(
        _Buffer(f"conv_rows[{index}]", 0x700 + index) for index in range(layer_count)
    )
    session._verify_linear_recurrent_state_rows = tuple(
        _Buffer(f"recurrent_rows[{index}]", 0x800 + index)
        for index in range(layer_count)
    )
    session._verify_linear_conv_initial_snapshots = tuple(
        _Buffer(f"conv_snap[{index}]", 0x900 + index) for index in range(layer_count)
    )
    session._verify_linear_recurrent_initial_snapshots = tuple(
        _Buffer(f"recurrent_snap[{index}]", 0xA00 + index)
        for index in range(layer_count)
    )
    session._verify_linear_initial_snapshot_users = (
        case.initial_snapshot_users
        if snapshot_users_override is None
        else snapshot_users_override
    )

    session._capture_prefill_gdn_for_call = MethodType(
        lambda self, requested: (
            bool(requested)
            if requested is not None
            else bool(self.verifier_prefill_gdn_capture)
        ),
        session,
    )
    session._ensure_bulk_prefill_workspace = MethodType(
        lambda self, **kwargs: None, session
    )
    session._ensure_verify_block_buffers = MethodType(
        lambda self, rows, **kwargs: None, session
    )
    session._ensure_verify_linear_state_row_buffers = MethodType(
        lambda self, rows, **kwargs: None, session
    )
    session._verify_f32_residual_for_call = MethodType(
        lambda self, requested: (
            bool(requested) if requested is not None else bool(case.f32_residual)
        ),
        session,
    )
    session._verify_f32_post_norm_for_call = MethodType(
        lambda self, requested: (
            bool(requested) if requested is not None else bool(case.session_post_norm)
        ),
        session,
    )
    session._validate_f32_verifier_resources = MethodType(
        lambda self, **kwargs: None, session
    )
    session._device_token_embedding_weight = MethodType(
        lambda self, **kwargs: _Buffer("embedding_weight", 0x700), session
    )
    session._seed_verify_hidden_f32_from_token_embedding = MethodType(
        lambda self, *args, **kwargs: None, session
    )
    session._eager_native_row_views = MethodType(
        lambda self, **kwargs: (None, None), session
    )
    session._iq_dense_mmq_verify_context = MethodType(
        lambda self, **kwargs: nullcontext(), session
    )
    session._load_expert_sidecar_device_layer = MethodType(
        _load_sidecar, session
    )
    session._verify_linear_state_row_pair = MethodType(
        _verify_state_rows, session
    )
    session._verify_linear_initial_snapshot_pair = MethodType(
        _verify_state_snapshot, session
    )
    session._capture_verify_layer_boundary_rows = MethodType(
        _capture_boundary_rows, session
    )
    session._sample_target_block_rows_from_hidden = MethodType(
        lambda self, *args, **kwargs: np.zeros((case.rows,), dtype=np.int64),
        session,
    )
    session._sample_device_from_hidden = MethodType(
        lambda self, *args, **kwargs: None, session
    )
    session._enqueue_target_block_rows_from_hidden = MethodType(
        lambda self, *args, **kwargs: None, session
    )
    session._normalize_layer_output_capture = MethodType(
        lambda self, ids: set() if ids is None else {int(value) for value in ids},
        session,
    )
    return session


def _copy_zero(destination, source, nbytes, *, runtime=None) -> None:
    if isinstance(destination, np.ndarray):
        destination.fill(0)


def _record_output_capture(trace, source: str):
    def capture(ptr, *args, **kwargs):
        trace["events"].append(
            {"kind": "output_capture", "source": source, "ptr": int(ptr)}
        )
        if source == "f32":
            return np.zeros(int(args[0]), dtype=np.float32)
        return np.zeros((int(args[0]), int(args[1])), dtype=np.float32)

    return capture


def _load_sidecar(self, layer_id, *, runtime):
    self.scratch.trace["events"].append(
        {"kind": "sidecar_load", "layer": int(layer_id)}
    )
    return SimpleNamespace(
        layer_id=int(layer_id),
        free=lambda *, runtime: self.scratch.trace["events"].append(
            {"kind": "sidecar_free", "layer": int(layer_id)}
        ),
    )


def _verify_state_rows(self, layer_id):
    if not self._verify_linear_conv_state_rows:
        return None
    self.scratch.trace["events"].append(
        {"kind": "state_rows", "layer": int(layer_id)}
    )
    return (
        self._verify_linear_conv_state_rows[int(layer_id)],
        self._verify_linear_recurrent_state_rows[int(layer_id)],
    )


def _verify_state_snapshot(self, layer_id):
    self.scratch.trace["events"].append(
        {"kind": "state_snapshot", "layer": int(layer_id)}
    )
    return (
        self._verify_linear_conv_initial_snapshots[int(layer_id)],
        self._verify_linear_recurrent_initial_snapshots[int(layer_id)],
    )


def _capture_boundary_rows(
    self,
    layer_id,
    layer_type,
    *,
    hidden_in_ptr,
    hidden_in_f32_ptr,
    layer_out_ptr,
    layer_out_f32_ptr,
    scratch,
    rows,
    runtime,
    prefill_gdn_capture=None,
    verify_f32_post_norm=None,
):
    self.scratch.trace["events"].append(
        {
            "kind": "boundary_capture",
            "layer": int(layer_id),
            "layer_type": str(layer_type),
            "hidden_in": int(hidden_in_ptr),
            "hidden_in_f32": hidden_in_f32_ptr,
            "layer_out": int(layer_out_ptr),
            "layer_out_f32": layer_out_f32_ptr,
            "scratch": scratch.label,
            "prefill_gdn_capture": prefill_gdn_capture,
            "verify_f32_post_norm": verify_f32_post_norm,
        }
    )
    return {"hidden_in": np.zeros((rows, 8), dtype=np.float32)}


def _call_kwargs(case: _Case) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "bulk_attention_mode": case.mode,
        "use_wmma_prefill": case.wmma,
        "stream": case.stream,
        "capture_linear_state_rows": case.capture_linear,
        "capture_prefill_gdn": None,
        "capture_layer_output_hidden": list(case.capture_output) or None,
        "capture_layer_boundary_hidden": list(case.capture_boundary) or None,
        "record_stage_timings": case.timing,
        "sync_stage_timings": case.sync,
        "defer_linear_state_commit": case.defer_linear,
        "verify_f32_residual": True if case.f32_residual else None,
        "verify_f32_post_norm": case.post_norm_arg,
    }
    if case.native_row_scratches:
        kwargs["_native_decode_row_scratches"] = tuple(
            SimpleNamespace(kv_storage_dtype=DType.BF16) for _ in range(case.rows)
        )
        kwargs["_native_attention_context_limit"] = 16
    if case.prebuilt:
        kwargs.update(
            {
                "_pre_staged_token_ids_ptr": 0x31,
                "_target_top1_i64_ptr": 0x32,
                "_enqueue_only": True,
                "_prebuilt_bulk_scratch": _ChunkScratch(
                    label="prebuilt",
                    start=0,
                    rows=case.rows,
                    norm=_Buffer("prebuilt_norm", 0x340),
                ),
                "_dynamic_cursor_advance": True,
                "_graph_hidden_seed_buf": _Buffer("graph_seed", 0x40),
                "_graph_hidden_f32_a": _Buffer("graph_f32_a", 0x41),
                "_graph_hidden_f32_b": _Buffer("graph_f32_b", 0x42),
            }
        )
    return kwargs


def _result_digest(result) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "token_ids": [int(value) for value in result.token_ids],
        "hidden_seeds_shape": list(result.hidden_seeds.shape),
        "pre_output_norm": (
            None
            if result.pre_output_norm_hidden is None
            else list(result.pre_output_norm_hidden.shape)
        ),
        "layer_output_keys": sorted(result.layer_output_hidden or {}),
        "layer_boundary_keys": sorted(result.layer_boundary_hidden or {}),
        "lm_head_logits": result.lm_head_logits_f32 is not None,
        "linear_state_rows_captured": bool(result.linear_state_rows_captured),
        "final_linear_state_committed": bool(
            result.final_linear_state_committed
        ),
    }


def _run_case(
    case: _Case,
    verify,
    monkeypatch: pytest.MonkeyPatch,
    *,
    snapshot_users_override: object | None = None,
):
    trace: dict[str, object] = {
        "events": [],
        "stage_keys": [],
        "position": None,
        "result": None,
    }
    session = _make_single_session(
        monkeypatch,
        case,
        trace,
        snapshot_users_override=snapshot_users_override,
    )
    result = verify(session, tuple(range(1, case.rows + 1)), **_call_kwargs(case))
    trace["stage_keys"] = list(session.last_verify_stage_timings_ms)
    trace["position"] = int(session._position)
    trace["result"] = _result_digest(result)
    return trace


def _run_case_with_error(
    case: _Case,
    verify,
    monkeypatch: pytest.MonkeyPatch,
    *,
    snapshot_users_override: object | None = None,
) -> tuple[dict[str, object], BaseException | None]:
    trace: dict[str, object] = {
        "events": [],
        "stage_keys": [],
        "position": None,
        "result": None,
    }
    session = _make_single_session(
        monkeypatch,
        case,
        trace,
        snapshot_users_override=snapshot_users_override,
    )
    error: BaseException | None = None
    result = None
    try:
        result = verify(
            session, tuple(range(1, case.rows + 1)), **_call_kwargs(case)
        )
    except Exception as exc:
        error = exc
    trace["stage_keys"] = list(session.last_verify_stage_timings_ms)
    trace["position"] = int(session._position)
    trace["result"] = _result_digest(result)
    return trace, error


def _capture_single_trace(
    monkeypatch: pytest.MonkeyPatch, case: _Case
) -> dict[str, object]:
    return _run_case(
        case,
        lambda session, tokens, **kwargs: session.verify_target_block(
            tokens, **kwargs
        ),
        monkeypatch,
    )


_CONTROL_TRACE: dict[str, dict[str, object]] = json.loads(
    (
        Path(__file__).parent
        / "fixtures"
        / "qwen35_gguf_verify_single_layer_stack_control.json"
    ).read_text()
)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_single_verifier_layer_stack_matches_committed_8904_control_trace(
    monkeypatch: pytest.MonkeyPatch, case: _Case
) -> None:
    captured = _capture_single_trace(monkeypatch, case)
    assert captured == _CONTROL_TRACE[case.name]


class _ExplodingCount:
    """A snapshot-user count that fails when converted after a set number."""

    def __init__(self, *, fail_after: int = 0) -> None:
        self.reads = 0
        self.fail_after = fail_after

    def __int__(self) -> int:
        self.reads += 1
        if self.reads > self.fail_after:
            raise RuntimeError("count boom")
        return 1


_NON_COUNT_CASES = (
    "bulk-r3-timing",
    "bulk-r6-prefill",
    "bulk-r6-state",
    "native-r3",
    "native-r6-chain",
    "native-r3-capture-nodefer",
)


@pytest.mark.parametrize("case_name", _NON_COUNT_CASES)
def test_single_verifier_never_converts_snapshot_count_outside_native_capture_defer(
    monkeypatch: pytest.MonkeyPatch, case_name: str
) -> None:
    case = next(case for case in _CASES if case.name == case_name)
    count = _ExplodingCount()
    captured = _run_case(
        case,
        lambda session, tokens, **kwargs: session.verify_target_block(
            tokens, **kwargs
        ),
        monkeypatch,
        snapshot_users_override=count,
    )
    assert count.reads == 0
    assert captured == _CONTROL_TRACE[case_name]


def test_single_verifier_converts_snapshot_count_late_after_prior_layer_marks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = next(case for case in _CASES if case.name == "native-r6-state")
    count = _ExplodingCount(fail_after=1)
    captured, error = _run_case_with_error(
        case,
        lambda session, tokens, **kwargs: session.verify_target_block(
            tokens, **kwargs
        ),
        monkeypatch,
        snapshot_users_override=count,
    )
    assert isinstance(error, RuntimeError)
    assert str(error) == "count boom"
    assert count.reads == 2

    control = _CONTROL_TRACE["native-r6-state"]
    snapshot_index = next(
        index
        for index, event in enumerate(control["events"])
        if event["kind"] == "state_snapshot" and event["layer"] == 1
    )
    # The first native layer completed; the second layer's conversion runs
    # after its state-row lookup and before its snapshot pair or operation.
    assert captured["events"] == control["events"][:snapshot_index]
    assert captured["stage_keys"] == control["stage_keys"][
        : len(captured["stage_keys"])
    ]
    assert captured["stage_keys"][-3:] == [
        "target_block_linear_attn_layers",
        "target_block_layer_total",
        "target_block_full_attn_layers",
    ]
    assert any(
        event["kind"] == "op" and event["layer"] == 0
        for event in captured["events"]
    )
    assert not any(
        event["kind"] == "op" and event["layer"] == 1
        for event in captured["events"]
    )


def test_owner_snapshot_conversion_failure_retains_prior_layer_sidecar_and_marks() -> None:
    owner = _owner()
    events: list[tuple[object, ...]] = []
    timings: dict[str, float] = {}
    sidecar = _Sidecar(events)
    resolvers = _Resolvers()
    count = _ExplodingCount(fail_after=1)

    def resolve_snapshot(layer_id: int):
        if int(count) > 0:
            events.append(("snapshot_pair", int(layer_id)))
            return (
                _Buffer(f"conv_snap[{layer_id}]", 0x900 + int(layer_id)),
                _Buffer(f"recurrent_snap[{layer_id}]", 0xA00 + int(layer_id)),
            )
        return None

    resolvers.initial_state_snapshot = resolve_snapshot  # type: ignore[method-assign]
    ops = _Ops()
    ops.linear = lambda *args, **kwargs: events.append(("op", int(args[0])))
    ops.native = lambda *args, **kwargs: events.append(("op", int(args[0])))

    with pytest.raises(RuntimeError, match="count boom"):
        _run(
            owner,
            geometry=_geometry(
                owner,
                layer_types=(LINEAR_ATTENTION, FULL_ATTENTION),
                bulk_attention_mode="native",
                capture_linear_state_rows=True,
                defer_linear_state_commit=True,
            ),
            hidden=_hidden(owner),
            scratch=_scratch(owner, resolvers),
            sidecar=_sidecar(
                owner,
                load=lambda layer_id: (
                    events.append(("load", int(layer_id))),
                    sidecar,
                )[1],
            ),
            ops=ops,
            resolvers=resolvers,
            stage_timings=timings,
        )

    assert events == [
        ("load", 0),
        ("snapshot_pair", 0),
        ("op", 0),
        ("sidecar_free",),
        ("load", 1),
        ("sidecar_free",),
    ]
    assert resolvers.events[-1] == ("linear_state_rows", 1)
    assert list(timings) == [
        "target_block_linear_attn_layers",
        "target_block_layer_total",
        "target_block_full_attn_layers",
    ]


# ---------------------------------------------------------------------------
# runner boundary (RED until the inline loop moves into the owner)
# ---------------------------------------------------------------------------
def test_runner_single_loop_delegates_to_the_layer_stack_owner() -> None:
    owner = _owner()
    assert (
        gguf_runner.run_verify_single_layer_stack
        is owner.run_verify_single_layer_stack
    )

    source = inspect.getsource(Qwen35GGUFResidentSession.verify_target_block)
    assert "run_verify_single_layer_stack(" in source
    assert "for layer_id, layer_type in enumerate(layer_types)" not in source
    assert "_run_native_attention_bulk_ffn_layer_rows(" not in source
    assert "_run_linear_attention_prefill_layer_rows(" not in source
    assert "_run_full_attention_decode_batch_layer_rows(" not in source
    assert "_run_full_attention_prefill_layer_aotriton(" not in source
    # The policy/context managers stay in the session for the structural test.
    assert "self._iq_dense_mmq_verify_context(rows=rows)," in source
    assert "wmma_prefill_session(block_wmma_prefill)," in source
