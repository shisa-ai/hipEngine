"""The verifier's packed row number is not its resident state slot."""
from types import SimpleNamespace as NS

import pytest

from hipengine.runtime import qwen35_gguf_runner as module


@pytest.mark.parametrize("indices,capacity", [((0, 1), 2), ((0, 1), 4), ((1, 0), 4), ((3, 1), 4), ((2, 3), 4)])
def test_moe_exact_linear_reads_resident_slots_not_packed_segments(monkeypatch, indices, capacity):
    conv_calls = []
    recurrent_calls = []
    kernels = (
        "gguf_rmsnorm_bf16_f32_weight",
        "gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out",
        "gguf_q8_0_t16_gemv_decode_rowtile4_f32_bf16_out",
    )
    for name in kernels:
        monkeypatch.setattr(module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "qwen35_linear_attn_chain_conv_decode_bf16_tloop",
                        lambda *args, **kwargs: conv_calls.append(args))
    monkeypatch.setattr(module, "qwen35_gdn_prefill_recurrent_rmsnorm_gate_bf16_decode_order_segments_state_rows_no_copy_f32",
                        lambda *args, **kwargs: recurrent_calls.append(args))
    cfg = NS(ssm_conv_kernel=4, ssm_inner_size=8, ssm_group_count=1,
             ssm_time_step_rank=2, ssm_state_size=4, rms_norm_eps=1e-6)
    layer = NS(weight=lambda name: NS(allocation=lambda *args: NS(tensor=NS(ptr=9000))))
    runner = NS(runtime=object(), weights=NS(layer=lambda _: layer, config=cfg),
                hidden_size=16, linear_qkv_width=2, ssm_value_dim=4,
                _run_linear_attention_alpha_beta_rows=lambda *args, **kwargs: None,
                _run_post_attention_ffn_rows=lambda *args, **kwargs: None)
    scratch = NS(gdn_active_segments=2, gdn_segment_capacity=2)
    for i, name in enumerate(("norm", "linear_qkv", "linear_z", "linear_alpha",
                             "linear_beta", "conv_out", "recurrent_bf16",
                             "recurrent_out", "gdn_cu_seqlens", "gdn_state_indices", "attn_out")):
        setattr(scratch, name, NS(ptr=20000 + i * 1000))
    conv_stride, recurrent_stride = 32, 128
    decode = NS(layer_conv_states=(NS(ptr=1000, nbytes=capacity * conv_stride),),
                layer_recurrent_states=(NS(ptr=5000, nbytes=capacity * recurrent_stride),))
    captures = (NS(ptr=40000, nbytes=6 * conv_stride),
                NS(ptr=50000, nbytes=6 * recurrent_stride))
    module.Qwen35GGUFFullStackRunner._run_moe_physical_c2_linear_rows_exact(
        runner, 0, 100, 200, scratch, rows=6, decode_scratch=decode,
        linear_state_rows=captures, state_capacity=capacity, state_indices=indices,
    )
    assert [args[1] for args in conv_calls] == [1000 + index * conv_stride for index in indices]
    assert [args[2] for args in conv_calls] == [40000, 40000 + 3 * conv_stride]
    assert [args[4] for args in conv_calls] == [scratch.conv_out.ptr, scratch.conv_out.ptr + 3 * 2 * 4]
    # GDN already uses the device slot map, and both captures stay packed-row indexed.
    assert recurrent_calls[0][7:9] == (5000, 50000)
    assert recurrent_calls[0][12] == scratch.gdn_state_indices.ptr


@pytest.mark.parametrize("indices", [(0,), (0, 4), (-1, 0)])
def test_moe_exact_linear_refuses_an_invalid_state_map_before_launch(indices):
    runner = NS()
    scratch = NS(gdn_active_segments=2, gdn_segment_capacity=2)
    with pytest.raises(ValueError, match="state.*(indices|capacity)"):
        module.Qwen35GGUFFullStackRunner._run_moe_physical_c2_linear_rows_exact(
            runner, 0, 100, 200, scratch, rows=6,
            decode_scratch=NS(), linear_state_rows=(), state_capacity=4,
            state_indices=indices,
        )
