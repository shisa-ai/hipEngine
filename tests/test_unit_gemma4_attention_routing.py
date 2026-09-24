"""Decode-vs-prefill symbol selection for Gemma 4 attention wrappers (unit tier).

The public wrappers are the registry entry point for the ``prefill_attention``
family. A decode step (``tokens == 1``) must select the warp-32 decode kernel;
multi-token blocks keep the original block kernel. ``head_dim > 512`` has no
decode-kernel register layout, so it stays on the block kernel.
"""

from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import attention_symbol


def test_single_token_selects_decode_kernel():
    assert attention_symbol("bf16", tokens=1, head_dim=128) == (
        "hipengine_gemma4_attention_decode_bf16"
    )
    assert attention_symbol("f32", tokens=1, head_dim=128) == (
        "hipengine_gemma4_attention_decode_f32"
    )


def test_multi_token_selects_prefill_kernel():
    assert attention_symbol("bf16", tokens=2, head_dim=128) == (
        "hipengine_gemma4_attention_prefill_bf16"
    )
    assert attention_symbol("f32", tokens=64, head_dim=128) == (
        "hipengine_gemma4_attention_prefill_f32"
    )


def test_large_head_dim_still_selects_decode_kernel():
    # The decode kernel mirrors the block kernel's capped thread width, so no
    # head_dim needs the block kernel at tokens == 1.
    assert attention_symbol("bf16", tokens=1, head_dim=1024) == (
        "hipengine_gemma4_attention_decode_bf16"
    )
    assert attention_symbol("bf16", tokens=1, head_dim=512) == (
        "hipengine_gemma4_attention_decode_bf16"
    )


def test_small_head_dim_stays_on_decode_kernel():
    # head_dim < 32 collapses to the single-chain (shfl-only) layout.
    assert attention_symbol("bf16", tokens=1, head_dim=6) == (
        "hipengine_gemma4_attention_decode_bf16"
    )