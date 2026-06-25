from __future__ import annotations

import ctypes

import numpy as np
import pytest


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP runtime is not available")
def test_mtp_dense_device_kv_cache_matches_two_row_dense_attention() -> None:
    """Sequential device-cache writes should match the existing dense cache oracle.

    The RED contract for the device-resident MTP KV path is: row 0 writes K/V,
    row 1 appends K/V and attends over rows 0..1, producing the same row-1
    attention output as the existing two-token dense attention call.
    """

    from hipengine.core.memory import free, malloc
    from hipengine.kernels.hip_gfx1100.speculative.mtp_nextn import (
        clear_weight_cache,
        qwen35_gguf_mtp_attention_sublayer_f32,
    )

    clear_weight_cache()
    rng = np.random.default_rng(1234)
    tokens = 2
    hidden = 16
    heads = 2
    kv_heads = 1
    head_dim = 4

    x = rng.normal(0.0, 0.2, size=(tokens, hidden)).astype(np.float32)
    attn_norm = rng.normal(1.0, 0.05, size=(hidden,)).astype(np.float32)
    q_norm = rng.normal(1.0, 0.05, size=(head_dim,)).astype(np.float32)
    k_norm = rng.normal(1.0, 0.05, size=(head_dim,)).astype(np.float32)
    wq = rng.normal(0.0, 0.15, size=(heads * 2 * head_dim, hidden)).astype(np.float32)
    wk = rng.normal(0.0, 0.15, size=(kv_heads * head_dim, hidden)).astype(np.float32)
    wv = rng.normal(0.0, 0.15, size=(kv_heads * head_dim, hidden)).astype(np.float32)
    wo = rng.normal(0.0, 0.15, size=(hidden, heads * head_dim)).astype(np.float32)

    expected = qwen35_gguf_mtp_attention_sublayer_f32(
        x,
        attn_norm,
        wq,
        wk,
        wv,
        wo,
        q_norm,
        k_norm,
        num_heads=heads,
        num_kv_heads=kv_heads,
        positions=np.asarray([0, 1], dtype=np.int64),
        context_counts=np.asarray([1, 2], dtype=np.int64),
    )

    key_cache = malloc(tokens * kv_heads * head_dim * 4)
    value_cache = malloc(tokens * kv_heads * head_dim * 4)
    try:
        row0 = qwen35_gguf_mtp_attention_sublayer_f32(
            x[:1],
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            num_heads=heads,
            num_kv_heads=kv_heads,
            positions=np.asarray([0], dtype=np.int64),
            context_counts=np.asarray([1], dtype=np.int64),
            dense_key_cache=key_cache,
            dense_value_cache=value_cache,
            dense_cache_len=0,
        )
        row1 = qwen35_gguf_mtp_attention_sublayer_f32(
            x[1:2],
            attn_norm,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            num_heads=heads,
            num_kv_heads=kv_heads,
            positions=np.asarray([1], dtype=np.int64),
            context_counts=np.asarray([2], dtype=np.int64),
            dense_key_cache=key_cache,
            dense_value_cache=value_cache,
            dense_cache_len=1,
        )
    finally:
        free(key_cache)
        free(value_cache)
        clear_weight_cache()

    np.testing.assert_allclose(row0, expected[:1], atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(row1, expected[1:2], atol=2e-5, rtol=2e-5)
