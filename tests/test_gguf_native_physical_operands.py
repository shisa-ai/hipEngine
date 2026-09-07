"""F1 physical operands: CPU owner/launch assertions, not device-content proof."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tests.test_gguf_execution_authorization import native_resident
from tests.test_gguf_ud_admission import _native_entry_session, _PositionOwnerSentinel
from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.runtime import qwen35_gguf_runner as runtime


def session_fixture(monkeypatch, tmp_path, layer="linear_attention"):
    if layer == "full_attention":
        from tests._qwen35_gguf_fixture import default_fixture_tensors, fixture_metadata, write_qwen35_gguf
        from tests.test_gguf_ud_admission import _materialize_fixture_on_cpu
        from hipengine.quant.gguf import GGMLQuantizationType as Q
        tensors = [t for t in default_fixture_tensors(1) if not t[0].startswith("blk.0.ssm_")
                   and t[0] not in {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight"}]
        tensors += [("output.weight", (64, 256), Q.Q8_0),
                    ("blk.0.attn_q.weight", (256, 256), Q.BF16),
                    ("blk.0.attn_k.weight", (64, 256), Q.BF16),
                    ("blk.0.attn_v.weight", (64, 256), Q.BF16),
                    ("blk.0.attn_output.weight", (256, 128), Q.BF16),
                    ("blk.0.attn_q_norm.weight", (64,), Q.F32),
                    ("blk.0.attn_k_norm.weight", (64,), Q.F32)]
        path = write_qwen35_gguf(tmp_path / "full.gguf", tensors,
                                fixture_metadata(1, extra={"qwen35.full_attention_interval": 1}))
        resident = _materialize_fixture_on_cpu(path, monkeypatch, decode_repack=False,
                                              requested_operations=("ar_decode_native_rows",))
    else:
        resident = native_resident(monkeypatch, tmp_path)
    owner = _PositionOwnerSentinel()
    session = _native_entry_session(resident, scratch_owner=owner)
    owner.kv_storage_dtype = DType.BF16
    owner.position_tensor = Tensor.from_handle(0x30000000, (8,), DType.INT64, Device("hip", 0))
    owner.cos_table = Tensor.from_handle(0x31000000, (64, 32), DType.FP32, Device("hip", 0))
    owner.sin_table = Tensor.from_handle(0x32000000, (64, 32), DType.FP32, Device("hip", 0))
    owner.full_key_caches = (NS(ptr=0x33000000, nbytes=0x10000),)
    owner.full_value_caches = (NS(ptr=0x34000000, nbytes=0x10000),)
    # Named views are deliberately separate from the allocator's owning list.
    owner.buffers = (NS(ptr=0x30000000, nbytes=0x5000000),)
    session._native_compact_scratch = lambda *args, **kwargs: owner
    session.runner.runtime = session.runtime
    session.runner.linear_qkv_width = 192
    session.runner.ssm_value_dim = 32
    session.runner.q_width = 128
    session.runner.kv_width = 64
    return session


@pytest.mark.parametrize("field", ["_lm_block_values", "_lm_block_indices", "_lm_out_index", "_lm_out_value",
                                  "_native_token_ids_host", "host_readonly", "runtime_owner", "position_tensor", "cos_table", "sin_table",
                                  "full_key_caches", "full_value_caches"])
def test_qualified_capture_binds_named_operands_before_any_replay_work(monkeypatch, tmp_path, field):
    session = session_fixture(monkeypatch, tmp_path)
    device_calls = []
    session.runtime = NS(stream_create=lambda: 1, stream_begin_capture=lambda stream: None,
                         stream_end_capture=lambda stream: 2, graph_instantiate=lambda graph: 3)
    session.runner.runtime = session.runtime
    session._enqueue_native_rows_model = lambda *a, **kw: ({}, {})
    graph = session.capture_native_rows_graph(rows=2, max_context_len=64)
    owner = session._target_scratch_owner
    owner.calls.clear()
    if field == "runtime_owner":
        session.runner.runtime = object()
    elif field == "host_readonly":
        session._native_token_ids_host.setflags(write=False)
    elif field == "_native_token_ids_host":
        session._native_token_ids_host = session._native_token_ids_host.copy()
    elif field.startswith("_lm"):
        old = getattr(session, field)
        setattr(session, field, NS(ptr=old.ptr + 0x1000, nbytes=old.nbytes))
    else:
        old = getattr(owner, field)
        if isinstance(old, tuple):
            setattr(owner, field, (NS(ptr=old[0].ptr + 128, nbytes=old[0].nbytes),))
        else:
            setattr(owner, field, replace(old, ptr=old.ptr + 128))
    def forbidden(*args, **kwargs):
        device_calls.append("device")
        pytest.fail("changed graph operand reached replay H2D/device work")
    monkeypatch.setattr(runtime, "copy_host_to_device", forbidden)
    with pytest.raises(ValueError):
        graph.step((1, 2))
    assert device_calls == []
    assert owner.calls == []


@pytest.mark.parametrize("layer", ["linear_attention", "full_attention"])
@pytest.mark.parametrize("ptr", [0, 0xDEADBEEF])
def test_qualified_direct_layer_arguments_need_owned_context(monkeypatch, tmp_path, layer, ptr):
    session = session_fixture(monkeypatch, tmp_path, layer)
    runner = session.runner
    runner.runtime = session.runtime
    device_calls = []
    def forbidden(*args, **kwargs):
        device_calls.append("device")
        pytest.fail("unowned direct operand reached a library/kernel")
    monkeypatch.setattr(runtime, "gguf_rmsnorm_bf16_f32_weight", forbidden)
    runner._cast_library = forbidden
    runner.linear_qkv_width = 192
    runner.ssm_value_dim = 32
    extra = {"cu_seqlens_ptr": ptr, "state_indices_ptr": ptr} if layer == "linear_attention" else {}
    with pytest.raises(ValueError):
        getattr(runtime.Qwen35GGUFFullStackRunner, f"_run_{layer}_decode_rows_native")(
            runner, 0, ptr, ptr, session._target_scratch_owner, rows=2, **extra)
    assert device_calls == []


@pytest.mark.parametrize("layer,operand", [(kind, name) for kind in ("linear_attention", "full_attention")
    for name in ("hidden", "out")] + [("linear_attention", "cu"), ("linear_attention", "indices")])
@pytest.mark.parametrize("bad", ["null", "unowned", "offset", "undersized", "dtype", "nonintegral"])
def test_direct_owned_context_refuses_each_wrong_argument(monkeypatch, tmp_path, layer, operand, bad):
    session = session_fixture(monkeypatch, tmp_path, layer)
    scratch = session._target_scratch_owner
    ctx = session._native_invocation_context(2, scratch)
    owners = {"hidden": session._hidden_a, "out": session._hidden_b,
              "cu": session._native_cu_seqlens_buf, "indices": session._native_state_indices_buf}
    args = {name: buf.ptr for name, buf in owners.items()}
    if bad == "null":
        args[operand] = 0
    elif bad == "unowned":
        args[operand] = 0x12340000
    elif bad == "offset":
        # Within the owner's allocation, but not the caller's intended row-prefix.
        args[operand] += 8
    elif bad == "undersized":
        owners[operand].nbytes = 1
    elif bad == "nonintegral":
        args[operand] = float(args[operand])
    else:
        owners[operand].dtype = DType.FP32
    def forbidden(*a, **kw):
        pytest.fail("invalid physical argument reached library/kernel")
    monkeypatch.setattr(runtime, "gguf_rmsnorm_bf16_f32_weight", forbidden)
    session.runner._cast_library = forbidden
    extra = {"cu_seqlens_ptr": args["cu"], "state_indices_ptr": args["indices"]} if layer == "linear_attention" else {}
    with pytest.raises(ValueError):
        getattr(runtime.Qwen35GGUFFullStackRunner, f"_run_{layer}_decode_rows_native")(
            session.runner, 0, args["hidden"], args["out"], scratch, rows=2,
            invocation_context=ctx, **extra)


@pytest.mark.parametrize("layer", ["linear_attention", "full_attention"])
def test_legal_context_reaches_actual_layer_first_consumer(monkeypatch, tmp_path, layer):
    session = session_fixture(monkeypatch, tmp_path, layer)
    scratch = session._target_scratch_owner
    ctx = session._native_invocation_context(2, scratch)
    class ReachedNorm(RuntimeError):
        pass
    def norm(*args, **kwargs):
        assert (kwargs.get("hidden_ptr", args[0] if args else None)) == session._hidden_a.ptr
        raise ReachedNorm("owned BF16 row-prefix")
    monkeypatch.setattr(runtime, "gguf_rmsnorm_bf16_f32_weight", norm)
    session.runner._run_attention_norm_rows = norm
    session.runner._cast_library = lambda: object()
    session.runner._paged_kv_write_library = lambda: object()
    session.runner._paged_attn_decode_library = lambda: object()
    extra = {"cu_seqlens_ptr": session._native_cu_seqlens_buf.ptr,
             "state_indices_ptr": session._native_state_indices_buf.ptr} if layer == "linear_attention" else {}
    with pytest.raises(ReachedNorm):
        getattr(runtime.Qwen35GGUFFullStackRunner, f"_run_{layer}_decode_rows_native")(
            session.runner, 0, session._hidden_a.ptr, session._hidden_b.ptr, scratch,
            rows=2, invocation_context=ctx, **extra)


@pytest.mark.parametrize("bad", ["missing", "permuted", "short", "wrong_dtype"])
def test_index_publication_is_required_not_inferred_from_pointer(monkeypatch, tmp_path, bad):
    from hipengine.loading.qwen35_gguf_native_operands import NativeIndexBinding
    session = session_fixture(monkeypatch, tmp_path)
    binding = session._native_index_binding
    if bad == "missing":
        session._native_index_binding = None
    elif bad == "permuted":
        session._native_index_binding = replace(binding, state_values=(1, 0, 2, 3, 4, 5, 6, 7))
    elif bad == "short":
        session._native_index_binding = replace(binding, cu_values=(0, 1))
    else:
        with pytest.raises(ValueError, match="dtype"):
            NativeIndexBinding.after_upload(session._native_cu_seqlens_buf, session._native_state_indices_buf,
                np.arange(9, dtype=np.int64), np.arange(8, dtype=np.int64))
        return
    with pytest.raises(ValueError, match="index"):
        session._native_invocation_context(2, session._target_scratch_owner)


def real_compact_scratch(session):
    """Use the real compact-span/subview constructor over CPU-owned buffers."""
    from dataclasses import make_dataclass
    from hipengine.kvcache.spans import KVLiveSpans
    owner = session._target_scratch_owner
    owner.position_buf = NS(ptr=owner.position_tensor.ptr, nbytes=64)
    owner.context_buf = NS(ptr=0x30100000, nbytes=64)
    owner.block_table = NS(ptr=0x30200000, nbytes=32)
    owner.block_table_tensor = Tensor.from_handle(owner.block_table.ptr, (8, 1), DType.INT32, Device("hip", 0))
    owner.context_tensor = Tensor.from_handle(owner.context_buf.ptr, (8,), DType.INT64, Device("hip", 0))
    owner.context_host = np.ones(8, dtype=np.int64)
    owner.position_host = np.zeros(8, dtype=np.int64)
    owner.blocks_per_slot = 1
    owner.max_positions = 64
    owner.decode_spans = KVLiveSpans.paged_uniform(block_table=owner.block_table_tensor,
        live_counts=owner.context_tensor, max_live_count=1, storage_dtype=DType.BF16,
        row_positions=owner.position_tensor, span_role="decode")
    owner.append_spans = replace(owner.decode_spans, live_counts=owner.position_tensor)
    data = dict(vars(owner))
    calls = data.pop("calls")  # instrumentation, not an allocator/launch field
    cls = make_dataclass("OwnedScratch", [(name, object) for name in data], frozen=True,
                        namespace={"set_full_attention_positions": _PositionOwnerSentinel.set_full_attention_positions,
                                   "calls": property(lambda self: calls)})
    session._target_scratch_owner = cls(**data)
    session._target_layout = NS(max_sequence_length=64)
    del session._native_compact_scratch  # use the actual production constructor
    return session._native_compact_scratch(2, span_role="decode", max_context_len=64)


def test_real_compact_prefix_views_are_legal_and_foreign_offsets_refuse(monkeypatch, tmp_path):
    session = session_fixture(monkeypatch, tmp_path)
    scratch = real_compact_scratch(session)
    ctx = session._native_invocation_context(2, scratch)
    assert scratch.position_tensor.shape == (2,)
    assert session._target_scratch_owner.position_tensor.shape == (8,)
    assert scratch.position_tensor.ptr == session._target_scratch_owner.position_buf.ptr
    ctx.validate(session, 2, scratch)
    for name in ("position_tensor", "block_table_tensor", "cos_table"):
        bad = replace(scratch, **{name: replace(getattr(scratch, name), ptr=getattr(scratch, name).ptr + 8)})
        with pytest.raises(ValueError):
            session._native_invocation_context(2, bad)
    with pytest.raises(ValueError):
        ctx.validate(session, 3, scratch)
    with pytest.raises(ValueError):
        ctx.validate(session, 2, replace(scratch, decode_spans=replace(scratch.decode_spans, max_live_count=32)))


@pytest.mark.parametrize("layer", ["linear_attention", "full_attention"])
def test_actual_enqueue_forwards_context_and_sampler_readback_plan(monkeypatch, tmp_path, layer):
    session = session_fixture(monkeypatch, tmp_path, layer)
    scratch = session._target_scratch_owner
    ctx = session._native_invocation_context(2, scratch)
    calls = []
    def layer_call(layer_id, hidden, out, supplied_scratch, *, rows, invocation_context, **kw):
        assert invocation_context is ctx
        invocation_context.validate_layer(session.runner, layer_id, supplied_scratch, rows, hidden, out,
            layer_type=layer, indices_required=layer == "linear_attention",
            cu_seqlens_ptr=kw.get("cu_seqlens_ptr"), state_indices_ptr=kw.get("state_indices_ptr"))
        calls.append("layer")
        return "owned"
    setattr(session.runner, f"_run_{layer}_decode_rows_native", layer_call)
    session._device_token_embedding_weight = lambda **kw: session.runner.weights.root("token_embedding")
    monkeypatch.setattr(runtime, "launch_gguf_embedding", lambda *a, **kw: calls.append(("embed", a[1], a[2], kw["rows"])))
    monkeypatch.setattr(runtime, "gguf_rmsnorm_bf16_f32_weight", lambda *a, **kw: None)
    monkeypatch.setattr(runtime, "launch_gguf_linear", lambda *a, **kw: calls.append(("head", a[2])))
    monkeypatch.setattr(runtime, "argmax_f32_rows_i32", lambda *a, **kw: calls.append(("sampler", a[:5])))
    session._lm_head_library = object()
    session._enqueue_native_rows_model(scratch, rows=2, stream=0, embedding_ready=False, invocation_context=ctx)
    assert calls == [("embed", session._token_buf.ptr, session._hidden_a.ptr, 2), "layer",
                     ("head", session._logits_buf.ptr),
                     ("sampler", tuple(getattr(ctx.operands, name).ptr for name in
                       ("_logits_buf", "_lm_block_values", "_lm_block_indices", "_lm_out_index", "_lm_out_value")))]


def test_graph_allows_dynamic_contents_and_reads_captured_sampler_owner(monkeypatch, tmp_path):
    session = session_fixture(monkeypatch, tmp_path)
    real_compact_scratch(session)
    session.runtime = NS(stream_create=lambda: 1, stream_begin_capture=lambda s: None,
        stream_end_capture=lambda s: 2, graph_instantiate=lambda g: 3,
        graph_launch=lambda *a: None, stream_synchronize=lambda *a: None)
    session.runner.runtime = session.runtime
    session._enqueue_native_rows_model = lambda *a, **kw: ({}, {})
    graph = session.capture_native_rows_graph(rows=2, max_context_len=64)
    ptrs = []
    monkeypatch.setattr(runtime, "copy_host_to_device", lambda buf, *a, **kw: ptrs.append(("h2d", buf.ptr)))
    def readback(host_ptr, buffer, nbytes, **kwargs):
        assert host_ptr == session._native_token_ids_host.ctypes.data
        assert nbytes == 8
        ptrs.append(("d2h", buffer.ptr))
        session._native_token_ids_host[:2] = (7, 8)
    monkeypatch.setattr(runtime, "copy_device_to_host", readback)
    owner = session._target_scratch_owner
    owner.position_host[:] = 3
    owner.context_host[:] = 4
    session._position = 3
    # Same views, different contents: neither host token values nor mutable
    # positions are pointer/certificate hashes or inferred device observations.
    result = graph.step((1, 2))
    assert result.token_ids == (7, 8)
    assert ptrs == [("h2d", graph.invocation_context.operands._token_buf.ptr),
                    ("d2h", graph.invocation_context.operands._lm_out_index.ptr)]


def test_context_cannot_substitute_a_different_operand_plan(monkeypatch, tmp_path):
    session = session_fixture(monkeypatch, tmp_path)
    owner = session._target_scratch_owner
    ctx = session._native_invocation_context(2, owner)
    foreign = NS(ptr=0x40000000, nbytes=4096)
    forged = replace(ctx, operands=replace(ctx.operands, _hidden_a=foreign))
    with pytest.raises(ValueError, match="operand plan"):
        forged.validate_layer(session.runner, 0, owner, 2, foreign.ptr, session._hidden_b.ptr,
            layer_type="linear_attention", indices_required=True,
            cu_seqlens_ptr=session._native_cu_seqlens_buf.ptr,
            state_indices_ptr=session._native_state_indices_buf.ptr)


def test_named_inventory_covers_nested_new_fields_not_only_buffer_list(monkeypatch, tmp_path):
    from hipengine.loading.qwen35_gguf_native_operands import physical_operand_inventory
    from hipengine.kvcache.spans import KVLiveSpans
    session = session_fixture(monkeypatch, tmp_path)
    scratch = real_compact_scratch(session)
    original = physical_operand_inventory(scratch)
    # A newly introduced named operand is covered without editing a whitelist.
    nested = NS(new_operands={"rotary_view": scratch.cos_table})
    first = physical_operand_inventory(nested)
    nested.new_operands["rotary_view"] = replace(scratch.cos_table, strides=(33, 1))
    assert first != physical_operand_inventory(nested)
    from dataclasses import make_dataclass
    extra = make_dataclass("FutureOwner", [("raw_pointer_tuple", tuple)], frozen=True)((1234, 5678))
    assert physical_operand_inventory(extra) != physical_operand_inventory(replace(extra, raw_pointer_tuple=(1234, 9876)))
    for field in ("base_offsets", "live_counts", "row_positions"):
        span = replace(scratch.decode_spans,
                       **{field: replace(getattr(scratch.decode_spans, field), ptr=getattr(scratch.decode_spans, field).ptr + 8)})
        assert isinstance(span, KVLiveSpans)
        assert original != physical_operand_inventory(replace(scratch, decode_spans=span))


def test_actual_native_callers_do_not_bypass_the_shared_operand_plan():
    import ast
    import inspect
    import textwrap
    from hipengine.loading.qwen35_gguf_native_operands import NativeRowsOperands
    from dataclasses import fields
    names = {f.name for f in fields(NativeRowsOperands)}
    methods = (runtime.Qwen35GGUFResidentSession._enqueue_native_rows_model,
               runtime.Qwen35GGUFResidentSession.step_rows_native, runtime.Qwen35GGUFNativeRowsGraph.step)
    for method in methods:
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "ptr" and isinstance(node.value, ast.Attribute):
                field = node.value
                if field.attr in names:
                    assert isinstance(field.value, ast.Name) and field.value.id == "operands", ast.unparse(node)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "operands":
                assert node.attr in names  # every used operand is declared on the shared owner
        if method is runtime.Qwen35GGUFResidentSession._enqueue_native_rows_model:
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr.endswith("decode_rows_native")]
            assert len(calls) == 2
            assert all("invocation_context" in {kw.arg for kw in node.keywords} for node in calls)
