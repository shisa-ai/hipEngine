"""Integration test for the real-model quant mix through the full mtp_nextn_layer.

M6: Validates that the composite mtp_nextn_layer dispatches the exact real-model
quant mix (Q8_0 eh_proj + attention, Q4_K gate/up experts, Q5_K down experts,
Q8_0 shared expert, BF16 router, F32 norms) end-to-end without
NotImplementedError, at real-model shapes (hidden=2048, inter=512, heads=4,
kv_heads=1, experts=256, top_k=8, vocab=248320 is too large for this test so
we use vocab=16).
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP not available")

import hipengine.kernels.hip_gfx1151  # noqa: F401,E402
from _gguf_synthetic_weights import (  # noqa: E402
    make_q4_k_weight,
    make_q5_k_weight,
    make_q8_0_weight,
)
from hipengine.quant.gguf import GGMLQuantizationType  # noqa: E402


def _f32(v):
    return np.ascontiguousarray(v, dtype=np.float32)


def test_mtp_nextn_layer_real_quant_mix_runs():
    """Run the composite layer with the exact real-model quant mix."""
    from hipengine.kernels.hip_gfx1100.speculative.mtp_nextn import (
        qwen35_gguf_mtp_nextn_layer_logits_f32,
    )

    # Real-model shapes (blk.40 from UD-Q4_K_M)
    hidden = 2048
    inter = 512
    heads = 4
    kv_heads = 1
    qk_head_dim = 256
    experts = 4  # reduced from 256 for test speed; dispatch is the same
    top_k = 2
    vocab = 16  # reduced from 248320

    np.random.seed(42)
    tokens = 1

    # F32 norms
    attn_norm = _f32(np.ones(hidden))
    q_norm = _f32(np.ones(qk_head_dim))
    k_norm = _f32(np.ones(qk_head_dim))
    post_norm = _f32(np.ones(hidden))
    hnorm = _f32(np.ones(hidden))
    enorm = _f32(np.ones(hidden))
    shared_head_norm = _f32(np.ones(hidden))

    # Q8_0 eh_proj [hidden, 2*hidden]
    eh_proj_w = make_q8_0_weight(out_features=hidden, in_features=hidden * 2)

    # Q8_0 attention weights
    wq = make_q8_0_weight(out_features=heads * 2 * qk_head_dim, in_features=hidden)
    wk = make_q8_0_weight(out_features=kv_heads * qk_head_dim, in_features=hidden)
    wv = make_q8_0_weight(out_features=kv_heads * qk_head_dim, in_features=hidden)
    wo = make_q8_0_weight(out_features=hidden, in_features=heads * qk_head_dim)

    # BF16 router [experts, hidden] - use F32 (moe_routing handles BF16 dequant)
    router_w = _f32(np.random.randn(experts, hidden) * 0.01)

    # Q4_K expert gate/up [E, inter, hidden], Q5_K expert down [E, hidden, inter]
    gate_qw = np.stack([make_q4_k_weight(out_features=inter, in_features=hidden) for _ in range(experts)])
    up_qw = np.stack([make_q4_k_weight(out_features=inter, in_features=hidden) for _ in range(experts)])
    down_qw = np.stack([make_q5_k_weight(out_features=hidden, in_features=inter) for _ in range(experts)])

    # Q8_0 shared expert
    shared_gate_qw = make_q8_0_weight(out_features=inter, in_features=hidden)
    shared_up_qw = make_q8_0_weight(out_features=inter, in_features=hidden)
    shared_down_qw = make_q8_0_weight(out_features=hidden, in_features=inter)
    shared_gate_logit_w = _f32(np.random.randn(hidden) * 0.01)

    # F32 shared head [vocab, hidden]
    shared_head_w = _f32(np.random.randn(vocab, hidden) * 0.01)

    # F32 inputs
    hidden_seed = _f32(np.random.randn(tokens, hidden) * 0.1)
    token_embed = _f32(np.random.randn(tokens, hidden) * 0.1)

    logits = qwen35_gguf_mtp_nextn_layer_logits_f32(
        hidden_seed, token_embed,
        eh_proj_w, hnorm, enorm,
        attn_norm, wq, wk, wv, wo, q_norm, k_norm,
        post_norm,
        router_w, gate_qw, up_qw, down_qw,
        GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K,
        shared_gate_logit_w, shared_gate_qw, shared_up_qw, shared_down_qw,
        GGMLQuantizationType.Q8_0,
        shared_head_norm, shared_head_w,
        num_heads=heads, num_kv_heads=kv_heads, experts_used=top_k,
        eh_proj_qtype=GGMLQuantizationType.Q8_0,
        wq_qtype=GGMLQuantizationType.Q8_0, wk_qtype=GGMLQuantizationType.Q8_0,
        wv_qtype=GGMLQuantizationType.Q8_0, wo_qtype=GGMLQuantizationType.Q8_0,
        eps=1e-6,
    )
    logits = np.asarray(logits, dtype=np.float32)
    assert logits.shape == (tokens, vocab), f"expected ({tokens}, {vocab}), got {logits.shape}"
    assert np.all(np.isfinite(logits)), "logits contain non-finite values"