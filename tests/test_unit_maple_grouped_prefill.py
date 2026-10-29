"""Grouped expert-major prefill is the production path, without an env rollback."""

import ctypes
from types import SimpleNamespace

import pytest

from hipengine.runtime import maple


def _pointer(ptr):
    return SimpleNamespace(ptr=ptr)


@pytest.mark.parametrize("rows,chunk_size", [(1, 1), (7, 3), (257, 256)])
def test_native_prefill_always_uses_grouped_metadata_and_original_lanes(monkeypatch, rows, chunk_size):
    monkeypatch.setenv("HIPENGINE_MAPLE_PREFILL_GROUPED_MOE", "0")
    names = (
        "token_ids", "hidden", "normalized", "qkv", "attention", "projection",
        "residual", "selected_ids", "routing_weights", "router_logits",
        "expert_start", "active_experts", "active_count", "sorted_lanes",
        "sorted_experts", "sorted_weights", "expert_gate", "expert_up",
        "expert_intermediate", "expert_down",
    )
    pf = SimpleNamespace(**{name: _pointer(100 + i) for i, name in enumerate(names)})
    spec = SimpleNamespace(vocab_size=10, hidden_size=4, q_size=4, kv_size=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=2,
        num_experts_per_tok=2, moe_intermediate_size=2, num_experts=3,
        rms_norm_eps=1e-6, layer_types=("full_attention",),
        uses_rope=lambda layer: False, rotary_dim=0, rope_theta=10000)
    runner = object.__new__(maple.MapleRunner)
    runner.closed, runner.position, runner.max_context = False, 0, 512
    runner.runtime = object()
    runner.checkpoint = SimpleNamespace(spec=spec)
    runner.buffers = SimpleNamespace(pf=pf,
        **{name: _pointer(300 + i) for i, name in enumerate(("normalized", "logits",
            "argmax_block_values", "argmax_block_indices", "argmax_index", "argmax_value"))},
        sliding_span_owner=SimpleNamespace(capacity=512, spans=object()),
        global_span_owner=SimpleNamespace(capacity=512, spans=object()),
        layers=(SimpleNamespace(key_cache=_pointer(400), value_cache=_pointer(401), spans=object()),))
    ternary = SimpleNamespace(weight=_pointer(500), row_alpha=_pointer(501))
    affine = SimpleNamespace(weight=_pointer(502), scales=_pointer(503), biases=_pointer(504))
    layer = SimpleNamespace(input_layernorm=_pointer(505), q_proj=ternary,
        k_proj=ternary, v_proj=ternary, q_norm=_pointer(506), k_norm=_pointer(507),
        o_proj=ternary, post_attention_layernorm=_pointer(508), router=_pointer(509),
        expert_gate_proj=ternary, expert_up_proj=ternary, expert_down_proj=ternary)
    runner.weights = SimpleNamespace(embeddings=affine, final_norm=_pointer(510),
        lm_head=affine, layers=(layer,))
    runner.libraries = SimpleNamespace(**{name: object() for name in (
        "ternary", "attention", "moe", "norm", "lm_head", "group_scatter")})
    calls = []
    functions = (
        "maple_kv_span_update_batched", "copy_host_array_to_device",
        "maple_affine4_embed_batched_bf16", "paro_rmsnorm_out_bf16",
        "maple_ternary_qkv_gemm_bf16", "maple_qknorm_rope_kv_write_batched_bf16",
        "maple_attention_prefill_ring_bf16", "maple_ternary_gemm_bf16",
        "paro_add_rmsnorm_out_bf16", "maple_router_topk_parallel_batched_bf16",
        "qwen35_moe_group_compact_active_i32_parallel",
        "maple_selected_ternary_dual_grouped_bf16", "maple_clamped_swiglu_bf16",
        "maple_selected_ternary_grouped_bf16", "maple_weighted_residual_batched_bf16",
        "maple_affine4_gemv_wave32_exact_f32", "argmax_f32",
    )
    def record(name):
        return lambda *args, **kwargs: calls.append((name, args, kwargs))
    for name in functions:
        monkeypatch.setattr(maple, name, record(name))
    for name in ("maple_selected_ternary_dual_gemv_batched_bf16",
                 "maple_selected_ternary_gemv_batched_bf16"):
        if hasattr(maple, name):
            monkeypatch.setattr(maple, name, lambda *a, **kw: pytest.fail("obsolete gather path"))
    def read_result(host_ptr, source, **kwargs):
        if source is runner.buffers.argmax_index:
            ctypes.c_int64.from_address(host_ptr).value = 7
        else:
            assert source is runner.buffers.argmax_value
            ctypes.c_float.from_address(host_ptr).value = 3.5
    monkeypatch.setattr(maple, "copy_device_to_host", read_result)

    result = runner.prefill_native([1] * rows, chunk_size=chunk_size)
    assert (result.token_id, result.position, runner.position) == (7, rows - 1, rows)
    assert result.top_logit == 3.5
    chunks = [min(chunk_size, rows - offset) for offset in range(0, rows, chunk_size)]
    metadata = [args for name, args, kw in calls if name == functions[10]]
    assert [args[-2:] for args in metadata] == [(chunk * 2, 3) for chunk in chunks]
    assert all(args[:8] == tuple(getattr(pf, name).ptr for name in (
        "selected_ids", "routing_weights", "expert_start", "active_experts",
        "active_count", "sorted_lanes", "sorted_experts", "sorted_weights")) for args in metadata)
    dual = [args for name, args, kw in calls if name == "maple_selected_ternary_dual_grouped_bf16"]
    down = [args for name, args, kw in calls if name == "maple_selected_ternary_grouped_bf16"]
    assert [args[9] for args in dual] == chunks
    assert [args[6] for args in down] == chunks
    assert all(args[5:9] == (pf.expert_start.ptr, pf.sorted_lanes.ptr,
        pf.expert_gate.ptr, pf.expert_up.ptr) for args in dual)
    assert all(args[3:6] == (pf.expert_start.ptr, pf.sorted_lanes.ptr, pf.expert_down.ptr) for args in down)
    for name, args, kw in calls:
        if name in ("qwen35_moe_group_compact_active_i32_parallel",
                    "maple_selected_ternary_dual_grouped_bf16", "maple_selected_ternary_grouped_bf16"):
            assert kw["runtime"] is runner.runtime
    ordered = [name for name, args, kw in calls if name in functions[9:15]]
    assert ordered == list(functions[9:15]) * len(chunks)


@pytest.mark.parametrize("runner_class", [maple.MapleRunner, maple.MapleBatchRunner])
def test_runner_load_always_prepares_group_metadata(monkeypatch, runner_class):
    monkeypatch.setenv("HIPENGINE_MAPLE_PREFILL_GROUPED_MOE", "0")
    monkeypatch.setattr(maple, "resolve_backend", lambda backend: "hip_gfx1151")
    monkeypatch.setattr(maple, "load_backend_kernel_package", lambda backend: None)
    for name in ("build_maple_ternary", "build_maple_attention", "build_maple_moe",
                 "build_qwen35_rmsnorm", "build_lm_head"):
        monkeypatch.setattr(maple, name, lambda **kw: object())
    class MetadataBuildReached(Exception):
        pass
    def build_metadata(**kwargs):
        assert kwargs == {"load": True}
        raise MetadataBuildReached
    monkeypatch.setattr(maple, "build_qwen35_moe_group_scatter", build_metadata)
    monkeypatch.setattr(maple, "materialize_maple_weights", lambda *a, **kw: pytest.fail("metadata must be ready before upload"))
    kwargs = ({"batch_size": 2, "per_capacity": 512}
        if runner_class is maple.MapleBatchRunner else {"max_context": 512})
    with pytest.raises(MetadataBuildReached):
        runner_class.load(SimpleNamespace(spec=SimpleNamespace(max_position_embeddings=512)),
            backend="hip_gfx1151", runtime=object(), **kwargs)
