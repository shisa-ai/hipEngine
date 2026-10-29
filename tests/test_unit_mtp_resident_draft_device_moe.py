"""Resident draft MoE-down keeps the device-owned default without host scratch."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.speculative import mtp_resident_draft as draft


_LEGACY = ("selected_out", "gate_f32", "up_f32", "inter_f32", "down_out", "scaled", "gated_shared", "tmp")


class _ReachedDeviceMoe(Exception):
    pass


def _fail(*args, **kwargs):
    raise AssertionError("legacy host readback or scratch clear must not run")


def test_draft_allocates_only_live_device_moe_scratch():
    runner = object.__new__(draft.Qwen35GGUFResidentMTPDraftRunner)
    runner.hidden_size, runner.num_heads, runner.num_kv_heads = 256, 4, 1
    runner.qk_head_dim, runner.experts_used, runner.inter_dim = 64, 8, 512
    runner.vocab, runner._draft_chain_cap = 256, 16
    runner._draft_dense_q8_dp4a_enabled = False
    allocations = []

    def allocate(size):
        allocation = SimpleNamespace(ptr=0x1000 + len(allocations) * 0x100, nbytes=size)
        allocations.append(allocation)
        return allocation

    runner._malloc = allocate
    runner._allocate_buffers()
    assert not any(hasattr(runner, name) for name in _LEGACY)
    assert runner.inter_bf16.nbytes == 8 * 512 * 2
    assert runner.down_out_bf16.nbytes == 8 * 256 * 2
    assert len(allocations) == len({id(a) for a in allocations})


@pytest.mark.parametrize("retired", [None, "0", "false", "1"])
@pytest.mark.parametrize("stream", [0, 7])
@pytest.mark.parametrize("router_parallel", [False, True])
def test_run_one_always_reaches_device_moe_without_readback(monkeypatch, retired, stream, router_parallel):
    if retired is None:
        monkeypatch.delenv("HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_MOE", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_MOE", retired)
    runner = object.__new__(draft.Qwen35GGUFResidentMTPDraftRunner)
    # A stale in-memory selector must not authorize the removed branch either.
    runner._device_moe_enabled = False
    runner.runtime = SimpleNamespace(memset=_fail, memset_async=_fail)
    runner.hidden_size, runner.num_heads, runner.num_kv_heads = 64, 2, 1
    runner.qk_head_dim, runner.experts_used, runner.inter_dim = 32, 2, 32
    runner.sync_stage_timings, runner.eps = False, 1e-6
    runner._router_row_parallel_enabled = router_parallel
    # Execution uses the captured construction choice, not a live environment read.
    monkeypatch.setenv("HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL", str(int(not router_parallel)))
    for index, name in enumerate((
        "attn_normed", "wq", "wk", "wv", "q_full", "key_cur", "value_cur", "query", "gate",
        "q_norm", "k_norm", "attn", "gated", "wo", "wo_out", "projected", "attended",
        "post_norm_weight", "post_norm", "router_weight_f32", "router_logits", "selected",
        "routing", "post_norm_bf16", "gate_exps", "up_exps", "gate_bf16", "up_bf16",
        "shared_gate", "shared_up", "shared_gate_out", "shared_up_out", "shared_inter",
        "shared_down", "shared_out", "shared_gate_vec_f32", "shared_gate_logit", "down_exps",
        "inter_bf16", "down_out_bf16", "attended_bf16", "shared_bf16", "ffn_out_bf16", "ffn_out",
    )):
        setattr(runner, name, SimpleNamespace(ptr=0x1000 + index * 0x100, nbytes=256))
    for name in ("_mtp_lib", "_k_lib", "_q4_lib", "_cast_lib", "_router_lib", "_silu_lib", "_combine_lib"):
        setattr(runner, name, object())
    for name in ("_project_current_to_attn_normed", "_try_dense_q8_dp4a_triple_f32", "_try_dense_q8_dp4a_f32", "_try_dense_q8_dp4a_dual_f32"):
        setattr(runner, name, lambda *args, **kwargs: False)
    for name in (
        "mtp_split_q_gate_f32", "mtp_rmsnorm_f32", "mtp_rope_f32", "mtp_dense_attn_f32",
        "mtp_sigmoid_gate_mul_f32", "mtp_add_f32", "mtp_linear_f32", "qwen35_router_select",
        "f32_to_bf16", "gguf_q4_k_selected_dual_gemv_bf16_bf16_out", "gguf_q8_0_gemv_f32_f32_out",
        "gguf_q8_0_dual_gemv_f32_f32_out", "mtp_silu_mul_f32",
    ):
        monkeypatch.setattr(draft, name, lambda *args, **kwargs: None)
    router_calls = []
    for name in ("mtp_linear_f32", "qwen35_router_logits_f32_f32w"):
        monkeypatch.setattr(draft, name, lambda *args, _name=name, **kwargs: router_calls.append((_name, args, kwargs)))
    monkeypatch.setattr(draft, "copy_device_to_host", _fail)
    calls = []

    def device(**kwargs):
        calls.append(kwargs)
        raise _ReachedDeviceMoe

    monkeypatch.setattr(draft, "apply_moe_down_combine", device)
    with pytest.raises(_ReachedDeviceMoe):
        runner._run_one(SimpleNamespace(ptr=0xF000), SimpleNamespace(ptr=0xF100),
                        cos_ptr=1, sin_ptr=2, pos_ptr=3, ctx_ptr=4,
                        dense_key_cache=None, dense_value_cache=None, dense_cache_len=0,
                        rotary_dim=32, stream=stream)
    assert len(calls) == 1
    assert calls[0]["selected_ptr"] == runner.selected.ptr
    assert calls[0]["routing_ptr"] == runner.routing.ptr
    assert calls[0]["ffn_out_f32_ptr"] == runner.ffn_out.ptr
    assert calls[0]["runtime"] is runner.runtime
    assert calls[0]["stream"] == stream
    symbol = "qwen35_router_logits_f32_f32w" if router_parallel else "mtp_linear_f32"
    library = runner._router_lib if router_parallel else runner._mtp_lib
    assert [name for name, args, kwargs in router_calls] == [symbol, symbol]
    assert [args for name, args, kwargs in router_calls] == [
        (runner.post_norm.ptr, runner.router_weight_f32.ptr, runner.router_logits.ptr, 1, 64, 256),
        (runner.post_norm.ptr, runner.shared_gate_vec_f32.ptr, runner.shared_gate_logit.ptr, 1, 64, 1),
    ]
    expected_kwargs = {"stream": stream, "library": library, "runtime": runner.runtime}
    if router_parallel:
        expected_kwargs["threads"] = 256
    assert [kwargs for name, args, kwargs in router_calls] == [expected_kwargs, expected_kwargs]
    assert not any(hasattr(runner, name) for name in _LEGACY)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("stream", [0, 7])
def test_selected_silu_down_comparison_keeps_device_owned_combine(monkeypatch, fused, stream):
    calls = []
    names = (
        "silu_mul_separate_out_bf16", "gguf_q5_k_selected_gemv_bf16_bf16_out",
        "gguf_q5_k_selected_silu_gemv_bf16_bf16_out", "f32_to_bf16",
        "weighted_sum_shared_gate_combine_residual_out_bf16_f32w", "bf16_to_f32",
    )
    for name in names:
        monkeypatch.setattr(draft, name, lambda *args, _name=name, **kwargs: calls.append((_name, args, kwargs)))
    monkeypatch.setattr(draft, "copy_device_to_host", _fail)
    runtime = object()
    libs = {name: object() for name in ("silu_lib", "k_lib", "combine_lib", "cast_lib")}
    markers = []
    draft.apply_moe_down_combine(
        gate_bf16_ptr=1, up_bf16_ptr=2, selected_ptr=3, routing_ptr=4,
        shared_out_ptr=5, shared_gate_logit_ptr=6, residual_ptr=7,
        down_exps_ptr=8, inter_bf16_ptr=9, down_out_bf16_ptr=10,
        attended_bf16_ptr=11, shared_bf16_ptr=12, ffn_out_bf16_ptr=13,
        ffn_out_f32_ptr=14, top_k=2, inter=256, hidden=64, num_experts=256,
        runtime=runtime, stream=stream, stage_marker=markers.append,
        selected_silu_down_fused=fused, **libs,
    )
    expected = ([
        ("gguf_q5_k_selected_silu_gemv_bf16_bf16_out", (1, 2, 3, 8, 10, 2, 2, 256, 256, 64), libs["k_lib"]),
    ] if fused else [
        ("silu_mul_separate_out_bf16", (1, 2, 9, 2, 256), libs["silu_lib"]),
        ("gguf_q5_k_selected_gemv_bf16_bf16_out", (9, 3, 8, 10, 2, 2, 256, 256, 64), libs["k_lib"]),
    ]) + [
        ("f32_to_bf16", (7, 11, 64), libs["cast_lib"]),
        ("f32_to_bf16", (5, 12, 64), libs["cast_lib"]),
        ("weighted_sum_shared_gate_combine_residual_out_bf16_f32w", (10, 4, 12, 6, 11, 13, 2, 64), libs["combine_lib"]),
        ("bf16_to_f32", (13, 14, 64), libs["cast_lib"]),
    ]
    assert calls == [(name, args, {"stream": stream, "library": library, "runtime": runtime}) for name, args, library in expected]
    assert markers == (["draft_run_moe_selected_silu_down_fused"] if fused else
                       ["draft_run_moe_selected_silu", "draft_run_moe_selected_down"]) + [
                           "draft_run_moe_combine_cast_inputs", "draft_run_moe_weighted_combine"]


def test_constructor_has_no_retired_device_moe_selector():
    assert "HIPENGINE_RESIDENT_MTP_DRAFT_DEVICE_MOE" not in inspect.getsource(draft.Qwen35GGUFResidentMTPDraftRunner.__post_init__)
