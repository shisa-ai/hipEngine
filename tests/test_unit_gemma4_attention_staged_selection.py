"""Exact staged attention selection stays capability-based and observable."""
import pytest
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import select_prefill_attention


@pytest.mark.parametrize('tokens,keys', [(1,31),(7,257),(19,1025),(3,262144)])
@pytest.mark.parametrize('heads,kv_heads,dim', [(16,8,256),(16,2,512)])
def test_explicit_staged_attention_selects_declared_geometry(tokens,keys,heads,kv_heads,dim):
    selection=select_prefill_attention(requested_variant='gemma4_staged',num_heads=heads,
                                      num_kv_heads=kv_heads,head_dim=dim,tokens=tokens,keys=keys)
    assert selection.variant == 'gemma4_staged'
    assert selection.is_strict
    assert selection.launcher.__name__ == 'gemma4_attention_staged_bf16'
    assert 'capability match' in selection.reason


def test_staged_capability_miss_names_geometry_and_keeps_strict_fallback():
    selection=select_prefill_attention(requested_variant='gemma4_staged',num_heads=4,
                                      num_kv_heads=2,head_dim=128,tokens=7,keys=257)
    assert selection.variant == 'gemma4_plain'
    assert 'capability miss' in selection.reason
    assert '256' in selection.reason and '512' in selection.reason
    assert '128' in selection.reason


def test_plain_attention_request_stays_unchanged():
    selection=select_prefill_attention(requested_variant='gemma4_plain',num_heads=16,
                                      num_kv_heads=2,head_dim=512,tokens=19,keys=1025)
    assert selection.variant == 'gemma4_plain'
    assert selection.is_strict
    assert selection.reason == 'strict'


def test_default_profile_requests_registered_staged_attention():
    from hipengine.generation.gemma4_gguf_profiles import (
        GEMMA4_GGUF_BACKEND, GEMMA4_GGUF_QUANT,
        register_gemma4_gguf_profiles, resolve_gemma4_prefill_attention_variants,
    )
    from hipengine.kernels.registry import resolve
    register_gemma4_gguf_profiles()
    assert resolve_gemma4_prefill_attention_variants(environ={}) == ('gemma4_staged',)
    fn = resolve(backend=GEMMA4_GGUF_BACKEND, layer='prefill_attention',
                 quant=GEMMA4_GGUF_QUANT, variant='gemma4_staged')
    assert fn.__name__ == 'gemma4_attention_staged_bf16'


def test_staged_capacity_is_declared_without_device_or_artifact_admission():
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import gemma4_attention_serves_keys
    assert gemma4_attention_serves_keys(head_dim=512,num_heads=16,num_kv_heads=2,
                                      keys=262144,requested_variant='gemma4_staged') is None
