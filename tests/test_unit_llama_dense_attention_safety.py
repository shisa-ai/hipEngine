"""Regression coverage for dense attention admission and scratch ownership."""

from types import SimpleNamespace

import pytest


def test_strict_null_mask_is_rejected_before_launch(monkeypatch):
    from hipengine.kernels.hip_gfx1100.attention import dense_prefill_attention as attention

    def launch(*args, **kwargs):
        pytest.fail("null mask reached the exact kernel")

    monkeypatch.setattr(attention, "gemma4_attention_prefill_bf16", launch)
    with pytest.raises(ValueError, match="mask"):
        attention.dense_causal_prefill_bf16(
            1, 2, 3, 0, 4, tokens=1, num_heads=2, num_kv_heads=1,
            head_dim=128, scale=0.125, strict=True, runtime=object(),
        )


def test_tiled_scratch_does_not_allocate_a_quadratic_mask(monkeypatch):
    from hipengine.runtime import llama_prefill as prefill

    sizes = []

    def allocate(size, **kwargs):
        sizes.append(size)
        return SimpleNamespace(ptr=len(sizes), size=size)

    monkeypatch.setattr(prefill, "malloc", allocate)
    spec = SimpleNamespace(hidden_size=256, q_width=256, kv_width=128,
                           intermediate_size=512, head_dim=128)
    scratch = prefill._PrefillScratch.allocate(spec, 777, runtime=object(), needs_mask=False)
    assert scratch.mask is None
    assert 777 * 777 not in sizes


def test_strict_prefill_bound_accounts_for_exact_kernel():
    from hipengine.runtime.llama_dense import max_dense_prefill_tokens

    spec = SimpleNamespace(head_dim=128, num_attention_heads=4,
                           num_key_value_heads=2, max_position_embeddings=32768)
    assert max_dense_prefill_tokens(spec, strict=True) == 15856
    assert max_dense_prefill_tokens(spec) == 32768
