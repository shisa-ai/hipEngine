"""Capability-based exact attention selection for shared-batch decode."""
import pytest

import hipengine.runtime.qwen4_exp_batch_qsa as qsa


@pytest.mark.parametrize("query_heads,kv_heads", [(24, 2), (8, 2), (2, 1)])
def test_compatible_geometry_selects_registered_head_pair(query_heads, kv_heads):
    fn, variant = qsa.batch_qsa_attention_implementation(
        head_dim=256, block_size=256, query_heads=query_heads, kv_heads=kv_heads,
    )
    assert fn is qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_h256_head_pair_rows_f32
    assert variant == "strict_h256_head_pair_rows_spans"


@pytest.mark.parametrize("changed", [
    {"head_dim": 128}, {"block_size": 128}, {"query_heads": 6},
    {"kv_heads": 0}, {"query_heads": 0}, {"query_heads": 5},
])
def test_other_geometries_keep_strict_rows(changed):
    shape = dict(head_dim=256, block_size=256, query_heads=24, kv_heads=2)
    shape.update(changed)
    fn, variant = qsa.batch_qsa_attention_implementation(**shape)
    assert fn is qsa.qwen4_exp_qsa_sparse_attention_paged_bf16_rows_f32
    assert variant == "strict_rows_spans"
