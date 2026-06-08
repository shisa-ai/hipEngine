from __future__ import annotations

import inspect

from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner


def test_p10_x1_decode_repack_does_not_change_linear_attention_math() -> None:
    """T16 materialization must not silently switch non-linear-attention math.

    P10.X1 found that guarding the linear-attention ``ssm_out`` path on
    ``gguf_decode_repack_enabled()`` changed the activation contract from
    BF16-input Q8_0 GEMV to F32-input Q8T16 GEMV.  That was faster-looking but
    not equivalent enough for MoE routing.  Decode repack should select weight
    layout / kernel implementation, not change the surrounding math graph.
    """

    source = inspect.getsource(Qwen35GGUFFullStackRunner._run_linear_attention_attn_only)

    assert "gguf_decode_repack_enabled" not in source
    assert "activation_dtype=GGUF_ACTIVATION_F32" not in source


def test_p10_x1_decode_repack_does_not_switch_full_attention_math() -> None:
    """The T16 flag must not choose a different full-attention decode graph.

    GGUF may now choose split-K full-attention decode by *context length*, but
    P10.X1 still forbids changing the graph merely because T16 decode-repack is
    enabled.
    """

    source = inspect.getsource(Qwen35GGUFFullStackRunner._run_full_attention_attn_only)

    assert "gguf_decode_repack_enabled" not in source
    assert "_use_gguf_full_attention_split_decode" in source
    assert "gguf_qwen35_head_rmsnorm_partial_rotary_position_key_bf16_f32_weight" not in source
