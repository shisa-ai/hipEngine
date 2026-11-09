"""RED unit tests for the clean-room EAGLE-4 CPU oracle (synthetic weights)."""
from __future__ import annotations

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.eagle4 import EAGLE4NumpyDrafter, eagle4_attention
from hipengine.kernels.cpu_reference.ops import linear, rmsnorm


def _config(hidden=64, heads=4, head_dim=16, vocab=40, rank=8, candidates=12):
    return {
        "hidden_size": hidden, "num_attention_heads": heads, "num_key_value_heads": heads,
        "head_dim": head_dim, "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 1e7},
        "sliding_window": 16, "norm_output": True, "fc_norm": True, "vocab_size": vocab,
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]},
        "efficient_lm_head_config": {"type": "low_rank", "rank": rank, "num_candidates": candidates},
    }


def _weights(hidden=64, vocab=40, rank=8, inter=128):
    rng = np.random.default_rng(0)
    def r(*shape): return rng.standard_normal(shape).astype(np.float32) * 0.1
    w = {
        "fc.weight": r(hidden, 3 * hidden),
        "fc_norm.0.weight": np.ones(hidden, np.float32),
        "fc_norm.1.weight": np.ones(hidden, np.float32),
        "fc_norm.2.weight": np.ones(hidden, np.float32),
        "embed_tokens.weight": r(vocab, hidden),
        "layers.0.input_layernorm.weight": np.ones(hidden, np.float32),
        "layers.0.hidden_norm.weight": np.ones(hidden, np.float32),
        "layers.0.self_attn.q_proj.weight": r(hidden, 2 * hidden),
        "layers.0.self_attn.k_proj.weight": r(hidden, 2 * hidden),
        "layers.0.self_attn.v_proj.weight": r(hidden, 2 * hidden),
        "layers.0.self_attn.o_proj.weight": r(hidden, hidden),
        "layers.0.post_attention_layernorm.weight": np.ones(hidden, np.float32),
        "layers.0.mlp.gate_proj.weight": r(inter, hidden),
        "layers.0.mlp.up_proj.weight": r(inter, hidden),
        "layers.0.mlp.down_proj.weight": r(hidden, inter),
        "norm.weight": np.ones(hidden, np.float32),
        "efficient_lm_head_down_proj.weight": r(rank, hidden),
        "efficient_lm_head.weight": r(vocab, rank),
        "lm_head.weight": r(vocab, hidden),
    }
    return w


def test_project_target_features_chunks_and_fc():
    cfg = _config(); w = _weights()
    nd = EAGLE4NumpyDrafter(cfg, w)
    aux = np.random.default_rng(1).standard_normal((3, 3 * cfg["hidden_size"])).astype(np.float32)
    got = nd.project_target_features(aux)
    chunks = np.split(aux, 3, axis=-1)
    normed = np.concatenate([rmsnorm(c, w[f"fc_norm.{i}.weight"], eps=1e-6) for i, c in enumerate(chunks)], axis=-1)
    expected = linear(normed, w["fc.weight"])
    assert np.allclose(got, expected, atol=1e-5)
    # Already-projected feature passes through unchanged.
    feat = np.random.default_rng(2).standard_normal((3, cfg["hidden_size"])).astype(np.float32)
    assert np.array_equal(nd.project_target_features(feat), feat)


def test_efficient_head_shortlist_matches_full_head():
    cfg = _config(); w = _weights()
    nd = EAGLE4NumpyDrafter(cfg, w)
    hidden = np.random.default_rng(3).standard_normal((4, cfg["hidden_size"])).astype(np.float32)
    logits, cand = nd.efficient_head(hidden)
    # Shortlist logits equal the full head rows gathered at the candidates.
    expected = np.einsum("th,tch->tc", hidden, w["lm_head.weight"][cand])
    assert np.allclose(logits, expected, atol=1e-4)
    # Candidates are the top-C of the low-rank estimate.
    r = linear(hidden, w["efficient_lm_head_down_proj.weight"])
    scores = linear(r, w["efficient_lm_head.weight"])
    expected_cand = np.argsort(-scores, axis=-1)[:, :cfg["efficient_lm_head_config"]["num_candidates"]]
    assert np.array_equal(cand, expected_cand)


def test_forward_output_shape_and_norm():
    cfg = _config(); w = _weights()
    nd = EAGLE4NumpyDrafter(cfg, w)
    ids = np.array([1, 2, 3], dtype=np.int64)
    aux = np.random.default_rng(4).standard_normal((3, 3 * cfg["hidden_size"])).astype(np.float32)
    pos = np.arange(3, dtype=np.int64)
    out = nd.forward(ids, aux, pos)
    assert out.shape == (3, cfg["hidden_size"])
    assert np.all(np.isfinite(out))


def test_greedy_chain_deterministic():
    cfg = _config(); w = _weights()
    nd = EAGLE4NumpyDrafter(cfg, w)
    aux = np.random.default_rng(5).standard_normal((3 * cfg["hidden_size"],)).astype(np.float32)
    d1, f1 = nd.draft_greedy_chain(aux, 7, 10, 4)
    d2, f2 = nd.draft_greedy_chain(aux, 7, 10, 4)
    assert d1.shape == (4,)
    assert np.array_equal(d1, d2)
    assert np.allclose(f1, f2)