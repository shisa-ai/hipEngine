"""CPU reference oracle for the EAGLE-4 feature-autoregression drafter (clean-room).

Independent numpy implementation of the EAGLE-4 draft mechanics reviewed from the
EAGLE-4 architecture doc + reference forward (ideas only; no code vendored — the
reference is source-available). Design: docs/reference/EAGLE4-IMPLEMENTATION-DESIGN.md.

A single Llama layer drafts at the feature level. Fusion projects the three target
auxiliary hidden states (each RMS-normed) to the draft hidden size; the layer
attends over ``concat(input_layernorm(embed), hidden_norm(feature))`` (2H input),
then a fused add+norm and SwiGLU MLP; the output feature is ``norm(residual+mlp)``
and feeds the next autoregressive step. The efficient LM head retrieves candidates
with a low-rank estimate and re-scores the shortlist with the full head rows.

This is the RED oracle for the GPU kernels and the live exactness gate.
"""
from __future__ import annotations

import numpy as np

from hipengine.kernels.cpu_reference.ops import linear, rmsnorm

_F32_TINY = 1.1754943508222875e-38
NUM_AUX = 3


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _rope_rotate(x: np.ndarray, positions: np.ndarray, theta: float, head_dim: int) -> np.ndarray:
    # x: [T, n_heads, head_dim]; standard split-half RoPE.
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
    angles = positions.astype(np.float32)[:, None] * inv_freq[None, :]  # [T, half]
    cos = np.cos(angles)[:, None, :]
    sin = np.sin(angles)[:, None, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def eagle4_attention(x, positions, q_proj, k_proj, v_proj, o_proj, *,
                     num_heads, head_dim, rope_theta, sliding_window):
    """Standard Llama MHA (no q/k norm) with causal sliding window over the sequence."""
    T = x.shape[0]
    q = linear(x, q_proj).reshape(T, num_heads, head_dim)
    k = linear(x, k_proj).reshape(T, num_heads, head_dim)
    v = linear(x, v_proj).reshape(T, num_heads, head_dim)
    q = _rope_rotate(q, positions, rope_theta, head_dim)
    k = _rope_rotate(k, positions, rope_theta, head_dim)
    scale = 1.0 / np.sqrt(np.float32(head_dim))
    out = np.zeros((T, num_heads, head_dim), dtype=np.float32)
    for t in range(T):
        lo = max(0, t - sliding_window + 1) if sliding_window else 0
        kk = k[lo:t + 1]  # [w, heads, dim]
        vv = v[lo:t + 1]
        scores = np.einsum("hd,whd->hw", q[t], kk) * scale
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        out[t] = np.einsum("hw,whd->hd", probs, vv)
    return linear(out.reshape(T, num_heads * head_dim), o_proj)


class EAGLE4NumpyDrafter:
    """Numpy EAGLE-4 drafter: fusion + single layer + efficient head + AR draft."""

    def __init__(self, config: dict, weights: dict[str, np.ndarray]) -> None:
        self.config = config
        self.w = {k: bf16_to_f32(v) if v.dtype == np.uint16 else np.asarray(v, dtype=np.float32)
                  for k, v in weights.items()}
        self.hidden_size = int(config["hidden_size"])
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.eps = float(config["rms_norm_eps"])
        self.rope_theta = float(config.get("rope_parameters", {}).get("rope_theta", config.get("rope_theta", 1e7)))
        self.sliding_window = int(config.get("sliding_window") or 0)
        self.norm_output = bool(config.get("norm_output", False))
        self.fc_norm = bool(config.get("fc_norm", False))
        ec = config["eagle_config"]
        self.aux_layers = list(ec["eagle_aux_hidden_state_layer_ids"])
        hc = config["efficient_lm_head_config"]
        self.rank = int(hc["rank"])
        self.num_candidates = int(hc["num_candidates"])
        self.vocab_size = int(config["vocab_size"])

    def project_target_features(self, aux_hidden: np.ndarray) -> np.ndarray:
        """aux_hidden [T, 3H] -> chunk, RMS-norm each, fc(3H->H)."""
        if aux_hidden.shape[-1] == self.hidden_size:
            return aux_hidden
        chunks = np.split(aux_hidden, NUM_AUX, axis=-1)
        if self.fc_norm:
            chunks = [rmsnorm(c, self.w[f"fc_norm.{i}.weight"], eps=self.eps) for i, c in enumerate(chunks)]
        return linear(np.concatenate(chunks, axis=-1), self.w["fc.weight"])

    def layer(self, embeds: np.ndarray, hidden: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """One EAGLE-4 layer; returns the output feature [T, H]."""
        residual = hidden
        x = np.concatenate([
            rmsnorm(embeds, self.w["layers.0.input_layernorm.weight"], eps=self.eps),
            rmsnorm(hidden, self.w["layers.0.hidden_norm.weight"], eps=self.eps),
        ], axis=-1)
        attn = eagle4_attention(
            x, positions,
            self.w["layers.0.self_attn.q_proj.weight"],
            self.w["layers.0.self_attn.k_proj.weight"],
            self.w["layers.0.self_attn.v_proj.weight"],
            self.w["layers.0.self_attn.o_proj.weight"],
            num_heads=self.num_heads, head_dim=self.head_dim,
            rope_theta=self.rope_theta, sliding_window=self.sliding_window)
        residual = residual + attn
        hn = rmsnorm(residual, self.w["layers.0.post_attention_layernorm.weight"], eps=self.eps)
        gate = linear(hn, self.w["layers.0.mlp.gate_proj.weight"])
        up = linear(hn, self.w["layers.0.mlp.up_proj.weight"])
        mlp = linear(_silu(gate) * up, self.w["layers.0.mlp.down_proj.weight"])
        summed = residual + mlp
        return rmsnorm(summed, self.w["norm.weight"], eps=self.eps)

    def forward(self, input_ids: np.ndarray, target_feature: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """target_feature [T, 3H] (aux concat) or [T, H] (draft output); returns output feature [T, H]."""
        embeds = self.w["embed_tokens.weight"][input_ids]
        hidden = self.project_target_features(target_feature)
        out = self.layer(embeds, hidden, positions)
        return out if self.norm_output else out

    def efficient_head(self, hidden: np.ndarray, k: int | None = None):
        """Low-rank retrieve num_candidates, then full-rank shortlist scoring.

        Returns (shortlist_logits [T, C], candidate_ids [T, C]) and, if k given,
        the top-k (probs, ids) over the candidate softmax.
        """
        r = linear(hidden, self.w["efficient_lm_head_down_proj.weight"])  # [T, rank]
        scores = linear(r, self.w["efficient_lm_head.weight"])            # [T, V] low-rank estimate
        C = self.num_candidates
        cand = np.argsort(-scores, axis=-1)[:, :C]                        # [T, C]
        head_rows = self.w["lm_head.weight"][cand]                        # [T, C, H]
        logits = np.einsum("th,tch->tc", hidden, head_rows)               # full-rank shortlist
        if k is None:
            return logits, cand
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        pick = np.argsort(-probs, axis=-1)[:, :k]
        top_probs = np.take_along_axis(probs, pick, axis=-1)
        top_ids = np.take_along_axis(cand, pick, axis=-1)
        return top_probs, top_ids

    def draft_greedy_chain(self, target_aux: np.ndarray, root_token: int, positions: np.ndarray,
                           n_draft: int):
        """Autoregressive greedy chain (no tree): draft n_draft tokens from the root.

        target_aux [3H] is the target's aux concat at the root position. The draft
        output feature feeds the next step; the token is the efficient-head argmax.
        Returns (draft_ids [n_draft], features [n_draft+1, H]).
        """
        feature = self.project_target_features(target_aux[None, :])[0]  # [H]
        token = int(root_token)
        draft_ids = []
        features = [feature]
        pos = int(positions)
        for _ in range(n_draft):
            embeds = self.w["embed_tokens.weight"][np.array([token])]
            # Single-row decode: attend over the accumulated draft feature sequence.
            feat_seq = np.stack(features, axis=0)
            tok_seq = np.array([root_token] + draft_ids, dtype=np.int64)
            pos_seq = np.arange(pos - len(features) + 1, pos + 1, dtype=np.int64)
            embed_seq = self.w["embed_tokens.weight"][tok_seq]
            out_seq = self.layer(embed_seq, feat_seq, pos_seq)
            feature = out_seq[-1]
            logits, cand = self.efficient_head(feature[None, :])
            token = int(cand[0, np.argmax(logits[0])])
            draft_ids.append(token)
            features.append(feature)
            pos += 1
        return np.array(draft_ids, dtype=np.int64), np.stack(features, axis=0)