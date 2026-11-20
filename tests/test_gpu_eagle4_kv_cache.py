"""GPU RED: EAGLE-4 persistent-KV context draft == numpy context reference.

Validates prefill_context + rows=1 cached decode + append_committed against a
numpy draft that attends over the full committed context (synthetic weights, so
this runs in CI without the real model). The stateless chain draft never sees the
prompt; the KV cache must reproduce the context-aware reference exactly.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

try:
    ctypes.CDLL("libamdhip64.so")
    _HIP = True
except OSError:
    _HIP = False

pytestmark = pytest.mark.skipif(not _HIP, reason="ROCm runtime not available")


def _config(hidden=64, heads=4, head_dim=16, vocab=40, rank=8, candidates=12):
    return {
        "hidden_size": hidden, "num_attention_heads": heads, "num_key_value_heads": heads,
        "head_dim": head_dim, "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 1e7},
        "sliding_window": 64, "norm_output": True, "fc_norm": True, "vocab_size": vocab,
        "intermediate_size": 128,
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 2, 3]},
        "efficient_lm_head_config": {"type": "low_rank", "rank": rank, "num_candidates": candidates},
    }


def _weights(hidden=64, vocab=40, rank=8, inter=128):
    rng = np.random.default_rng(0)
    def r(*shape): return rng.standard_normal(shape).astype(np.float32) * 0.1
    return {
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


def _numpy_context_draft(nd, ctx_aux, token_seq, bonus, n_draft):
    """Reference: attend over the full committed context (feature_i, token_{i+1})."""
    P = len(token_seq)
    feats = list(nd.project_target_features(ctx_aux))
    toks = list(token_seq[1:]) + [int(bonus)]
    base_pos = P - 1
    drafts = []
    for _ in range(n_draft):
        T = len(feats)
        out_n, out_s = nd.layer(
            nd.w["embed_tokens.weight"][np.array(toks, dtype=np.int64)],
            np.stack(feats, axis=0),
            np.arange(base_pos - T + 1, base_pos + 1, dtype=np.int64))
        feature = (out_n if nd.norm_output else out_s)[-1]
        logits, cand = nd.efficient_head(feature[None, :])
        nxt = int(cand[0, np.argmax(logits[0])])
        drafts.append(nxt); feats.append(feature); toks.append(nxt); base_pos += 1
    return drafts


def test_kv_cached_draft_matches_numpy_context():
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.cpu_reference.eagle4 import EAGLE4NumpyDrafter
    from hipengine.runtime.eagle4 import EAGLE4NativeDrafter

    cfg = _config(); w = _weights()
    nd = EAGLE4NumpyDrafter(cfg, w)
    rt = get_hip_runtime()
    H = cfg["hidden_size"]
    P, n_draft = 6, 5
    rng = np.random.default_rng(3)
    ctx_aux = rng.standard_normal((P, 3 * H)).astype(np.float32)
    prompt_ids = [int(t) for t in rng.integers(0, cfg["vocab_size"], size=P)]
    bonus = int(rng.integers(0, cfg["vocab_size"]))
    native = EAGLE4NativeDrafter(cfg, w, runtime=rt, max_seq=n_draft + 2, max_context_len=64)
    try:
        expected = _numpy_context_draft(nd, ctx_aux, prompt_ids, bonus, n_draft)
        native.prefill_context(ctx_aux, prompt_ids, bonus)
        got, _ = native.draft_greedy_chain_cached(n_draft)
        assert [int(t) for t in got] == expected, f"got {list(got)} expected {expected}"
    finally:
        native.close()


def test_append_committed_extends_context_matches_numpy():
    from hipengine.core.hip import get_hip_runtime
    from hipengine.kernels.cpu_reference.eagle4 import EAGLE4NumpyDrafter
    from hipengine.runtime.eagle4 import EAGLE4NativeDrafter

    cfg = _config(); w = _weights()
    nd = EAGLE4NumpyDrafter(cfg, w)
    rt = get_hip_runtime()
    H = cfg["hidden_size"]; V = cfg["vocab_size"]
    P, n_draft = 5, 4
    rng = np.random.default_rng(5)
    ctx_aux = rng.standard_normal((P, 3 * H)).astype(np.float32)
    prompt_ids = [int(t) for t in rng.integers(0, V, size=P)]
    bonus = int(rng.integers(0, V))
    native = EAGLE4NativeDrafter(cfg, w, runtime=rt, max_seq=n_draft + 2, max_context_len=64)
    try:
        native.prefill_context(ctx_aux, prompt_ids, bonus)
        # Commit m rows: target aux for the new positions + their input tokens.
        m = 3
        new_aux = rng.standard_normal((m, 3 * H)).astype(np.float32)
        next_tokens = [int(t) for t in rng.integers(0, V, size=m)]
        native.append_committed(new_aux, next_tokens)
        got, _ = native.draft_greedy_chain_cached(n_draft)
        # Reference over the extended context: prompt + committed features/tokens.
        ext_aux = np.concatenate([ctx_aux, new_aux], axis=0)
        ext_tokens = list(prompt_ids) + [int(bonus)] + [int(t) for t in next_tokens[:-1]]
        ext_bonus = int(next_tokens[-1])
        expected = _numpy_context_draft(nd, ext_aux, ext_tokens, ext_bonus, n_draft)
        assert [int(t) for t in got] == expected, f"got {list(got)} expected {expected}"
    finally:
        native.close()
