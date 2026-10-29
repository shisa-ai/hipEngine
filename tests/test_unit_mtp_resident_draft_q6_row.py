"""Explicit Q6 head comparisons keep their stage and gather contracts."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.speculative import mtp_resident_draft as draft


@pytest.mark.parametrize("shape", ["row", "pack8", "pack8_scalehoist", "pack8_llama", "pack16", "x8", "x8_dscale"])
@pytest.mark.parametrize("dp4a", [False, True])
@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("timing_sink", [False, True])
@pytest.mark.parametrize("output", ["logits", "top1", "gather"])
def test_head_shape_dispatch_and_stage2_domain(monkeypatch, shape, dp4a, sync, timing_sink, output):
    runner = object.__new__(draft.Qwen35GGUFResidentMTPDraftRunner)
    synchronizations = []
    runner.runtime = SimpleNamespace(stream_synchronize=synchronizations.append)
    runner.hidden_size, runner.num_heads, runner.num_kv_heads = 64, 2, 1
    runner.qk_head_dim, runner.experts_used, runner.inter_dim = 32, 2, 32
    vocab = 32 if shape == "pack16" else 24
    runner.vocab, runner._draft_chain_cap = vocab, 16
    runner.sync_stage_timings, runner.eps = sync, 1e-6
    runner._router_row_parallel_enabled = False
    runner._draft_dense_q8_dp4a_enabled = False
    runner._q6_top1_dp4a_enabled = dp4a
    runner._q6_top1_stage1_shape = shape
    allocations = []

    def allocate(size):
        buf = SimpleNamespace(ptr=0x1000 + len(allocations) * 0x1000, nbytes=size)
        allocations.append(buf)
        return buf

    runner._malloc = allocate
    runner._allocate_buffers()
    assert runner.q6_top1_block_values.nbytes == runner.vocab * 4
    assert runner.q6_top1_block_indices.nbytes == runner.vocab * 4
    for name in (
        "wq", "wk", "wv", "q_norm", "k_norm", "wo", "post_norm_weight",
        "router_weight_f32", "gate_exps", "up_exps", "shared_gate", "shared_up",
        "shared_down", "shared_gate_vec_f32", "down_exps", "shared_head_norm",
        "shared_head", "shared_head_x8", "shared_head_x8_dscale", "_embed_table_f32",
    ):
        setattr(runner, name, allocate(256))
    for name in (
        "_mtp_lib", "_k_lib", "_q4_lib", "_cast_lib", "_router_lib", "_silu_lib",
        "_combine_lib", "_q6_pack8_lib",
    ):
        setattr(runner, name, object())
    for name in (
        "_project_current_to_attn_normed", "_try_dense_q8_dp4a_triple_f32",
        "_try_dense_q8_dp4a_f32", "_try_dense_q8_dp4a_dual_f32",
    ):
        setattr(runner, name, lambda *args, **kwargs: False)
    for name in (
        "mtp_split_q_gate_f32", "mtp_rmsnorm_f32", "mtp_rope_f32", "mtp_dense_attn_f32",
        "mtp_sigmoid_gate_mul_f32", "mtp_add_f32", "mtp_linear_f32", "qwen35_router_select",
        "f32_to_bf16", "gguf_q4_k_selected_dual_gemv_bf16_bf16_out",
        "gguf_q8_0_gemv_f32_f32_out", "gguf_q8_0_dual_gemv_f32_f32_out",
        "mtp_silu_mul_f32", "apply_moe_down_combine",
    ):
        monkeypatch.setattr(draft, name, lambda *args, **kwargs: None)
    calls = []

    def record(name):
        return lambda *args, **kwargs: calls.append((name, args, kwargs))

    def unexpected(*args, **kwargs):
        pytest.fail("head comparison must not silently dispatch another layout or read back")

    quant = "gguf_q4_k_quantize_bf16_q8_1"
    dp4a_prefix = "gguf_q6_k_x8_dscale" if shape == "x8_dscale" else "gguf_q6_k_x8" if shape == "x8" else "gguf_q6_k_pack8"
    shape_suffix = {"row": "row_", "pack8_scalehoist": "scalehoist_",
                    "pack8_llama": "pack8_llama_", "pack16": "pack16_"}.get(shape, "")
    dp4a_stage = f"{dp4a_prefix}_gemv_decode_q8_1_dp4a_top1_{shape_suffix}stage1_f32"
    dp4a_gather = f"{dp4a_prefix}_gemv_decode_q8_1_dp4a_top1_{shape_suffix}gather_f32"
    bf16_stage = "gguf_q6_k_pack8_gemv_decode_bf16_top1_stage1_f32"
    bf16_gather = "gguf_q6_k_pack8_gemv_decode_bf16_top1_gather_f32"
    stage2 = "gguf_q6_k_pack8_top1_stage2_gather_f32"
    logits = "gguf_q6_k_pack8_gemv_decode_bf16_f32_out"
    expected_names = {quant, dp4a_stage, dp4a_gather, bf16_stage, bf16_gather, stage2, logits}
    for name in vars(draft):
        if name.startswith("gguf_q6_k_"):
            monkeypatch.setattr(draft, name, record(name) if name in expected_names else unexpected)
    monkeypatch.setattr(draft, quant, record(quant))
    monkeypatch.setattr(draft, "copy_device_to_host", unexpected)
    timings = {} if timing_sink else None
    top1 = 0xF0000 if output != "logits" else None
    embed = 0xF1000 if output == "gather" else None
    runner._run_one(
        allocate(256), allocate(256), cos_ptr=1, sin_ptr=2, pos_ptr=3, ctx_ptr=4,
        dense_key_cache=None, dense_value_cache=None, dense_cache_len=0, rotary_dim=32,
        stage_timings=timings, top1_out_ptr=top1, top1_next_embed_ptr=embed, stream=7,
    )
    staged = sync and timing_sink
    if output == "logits":
        assert [c[0] for c in calls] == [logits]
        assert calls[0][1] == (runner.head_normed_bf16.ptr, runner.shared_head.ptr,
                               runner.logits.ptr, 1, 64, vocab)
    else:
        names = [quant] if dp4a else []
        names += [dp4a_stage if dp4a else bf16_stage, stage2] if staged else [dp4a_gather if dp4a else bf16_gather]
        assert [c[0] for c in calls] == names
        source = runner.head_normed_q8_1.ptr if dp4a else runner.head_normed_bf16.ptr
        weight = runner.shared_head_x8.ptr if dp4a and shape in {"x8", "x8_dscale"} else runner.shared_head.ptr
        prefix = (source, weight)
        if dp4a and shape == "x8_dscale":
            prefix += (runner.shared_head_x8_dscale.ptr,)
        blocks = (runner.q6_top1_block_values.ptr, runner.q6_top1_block_indices.ptr)
        prefix += blocks
        outputs = (top1, runner.topk_values.ptr,
                   runner._embed_table_f32.ptr if embed is not None else None, embed)
        hidden = 64 if embed is not None else 0
        if staged:
            num_blocks = vocab if dp4a and shape == "row" else vocab // 16 if dp4a and shape == "pack16" else vocab // 8
            assert calls[-2][1] == (*prefix, 1, 64, vocab)
            assert calls[-1][1] == (*blocks, *outputs, 1, num_blocks, hidden, vocab)
        else:
            assert calls[-1][1] == (*prefix, *outputs, 1, 64, vocab, hidden)
        if dp4a:
            assert calls[0][1] == (runner.head_normed_bf16.ptr, runner.head_normed_q8_1.ptr, 1, 64)
            assert calls[0][2] == {"stream": 7, "library": runner._q4_lib, "runtime": runner.runtime}
        if staged:
            stage_name = ({"row": "dp4a_row", "pack8_scalehoist": "dp4a_scalehoist",
                           "pack8_llama": "dp4a_pack8_llama", "pack16": "dp4a_pack16",
                           "x8": "dp4a_x8", "x8_dscale": "dp4a_x8_dscale",
                           "pack8": "dp4a"}[shape] if dp4a else "bf16")
            stage2_name = "dp4a_row" if dp4a and shape == "row" else "dp4a" if dp4a else "bf16"
            assert f"draft_run_lm_head_q6_top1_{stage_name}_stage1" in timings
            assert f"draft_run_lm_head_q6_top1_{stage2_name}_stage2_gather" in timings
            assert f"draft_run_lm_head_q6_top1_{stage_name}_gather" in timings
    for name, _args, kwargs in calls:
        if name != quant:
            assert kwargs == {"stream": 7, "library": runner._q6_pack8_lib, "runtime": runner.runtime}
    assert bool(synchronizations) is staged
    if timing_sink:
        assert bool(timings) is staged
