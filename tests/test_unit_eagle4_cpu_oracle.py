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


def _reference_eagle4_forward(w, input_ids, target_feature, positions, *, hidden,
                              num_heads, head_dim, rope_theta, sliding_window, eps):
    """Independent clean-room translation of the EAGLE-4 reference forward op order
    (llama_eagle4.py): fc over three RMS-normed aux chunks; concat(input_layernorm(embed),
    hidden_norm(feature)) -> MHA -> fused add+norm -> SwiGLU -> norm(residual+mlp).
    Uses complex-multiply RoPE and an explicit per-head attention loop to cross-check
    the oracle's split-half RoPE / einsum attention."""
    def rms(x, weight):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight

    def lin(x, weight):
        return x @ weight.T

    def rope(t):
        half = head_dim // 2
        inv = 1.0 / (rope_theta ** (np.arange(half, dtype=np.float64) / half))
        ang = positions.astype(np.float64)[:, None] * inv[None, :]
        cis = np.cos(ang) + 1j * np.sin(ang)  # [T, half]
        out = np.empty_like(t)
        for h in range(t.shape[1]):
            z = t[:, h, :half].astype(np.float64) + 1j * t[:, h, half:].astype(np.float64)
            zr = z * cis
            out[:, h, :half] = zr.real
            out[:, h, half:] = zr.imag
        return out

    embeds = w["embed_tokens.weight"][input_ids]
    chunks = np.split(target_feature, 3, axis=-1)
    normed_chunks = [rms(c, w[f"fc_norm.{i}.weight"]) for i, c in enumerate(chunks)]
    feat = lin(np.concatenate(normed_chunks, axis=-1), w["fc.weight"])
    residual = feat
    x = np.concatenate([rms(embeds, w["layers.0.input_layernorm.weight"]),
                        rms(feat, w["layers.0.hidden_norm.weight"])], axis=-1)
    T = x.shape[0]
    q = lin(x, w["layers.0.self_attn.q_proj.weight"]).reshape(T, num_heads, head_dim)
    k = lin(x, w["layers.0.self_attn.k_proj.weight"]).reshape(T, num_heads, head_dim)
    v = lin(x, w["layers.0.self_attn.v_proj.weight"]).reshape(T, num_heads, head_dim)
    q = rope(q); k = rope(k)
    attn = np.zeros((T, num_heads, head_dim), dtype=np.float64)
    for t in range(T):
        lo = max(0, t - sliding_window + 1) if sliding_window else 0
        for h in range(num_heads):
            scores = (k[lo:t + 1, h, :] @ q[t, h]) / np.sqrt(head_dim)
            p = np.exp(scores - scores.max()); p = p / p.sum()
            attn[t, h] = p @ v[lo:t + 1, h, :]
    attn_out = lin(attn.reshape(T, num_heads * head_dim).astype(np.float32), w["layers.0.self_attn.o_proj.weight"])
    residual = residual + attn_out
    hn = rms(residual, w["layers.0.post_attention_layernorm.weight"])
    gate = lin(hn, w["layers.0.mlp.gate_proj.weight"])
    up = lin(hn, w["layers.0.mlp.up_proj.weight"])
    mlp = lin((gate / (1.0 + np.exp(-gate))) * up, w["layers.0.mlp.down_proj.weight"])
    summed = residual + mlp
    return rms(summed, w["norm.weight"])


def test_forward_matches_independent_reference():
    cfg = _config(); w = _weights()
    nd = EAGLE4NumpyDrafter(cfg, w)
    ids = np.array([1, 2, 3, 4], dtype=np.int64)
    aux = np.random.default_rng(7).standard_normal((4, 3 * cfg["hidden_size"])).astype(np.float32)
    pos = np.arange(4, dtype=np.int64)
    got = nd.forward(ids, aux, pos)
    expected = _reference_eagle4_forward(
        w, ids, aux, pos, hidden=cfg["hidden_size"], num_heads=cfg["num_attention_heads"],
        head_dim=cfg["head_dim"], rope_theta=1e7, sliding_window=cfg["sliding_window"], eps=1e-6)
    assert np.allclose(got, expected, atol=2e-4), f"max diff {np.max(np.abs(got - expected))}"


def test_norm_output_selects_normed_vs_summed():
    cfg = _config(); w = _weights()
    ids = np.array([1, 2], dtype=np.int64)
    aux = np.random.default_rng(8).standard_normal((2, 3 * cfg["hidden_size"])).astype(np.float32)
    pos = np.arange(2, dtype=np.int64)
    cfg_true = dict(cfg, norm_output=True)
    cfg_false = dict(cfg, norm_output=False)
    out_true = EAGLE4NumpyDrafter(cfg_true, w).forward(ids, aux, pos)
    out_false = EAGLE4NumpyDrafter(cfg_false, w).forward(ids, aux, pos)
    # norm_output=True returns the normed feature; False returns the un-normed sum.
    assert not np.allclose(out_true, out_false)
    assert np.allclose(np.sqrt(np.mean(out_true * out_true, axis=-1)), 1.0, atol=1e-3)