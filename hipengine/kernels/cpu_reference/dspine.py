"""CPU reference oracle for the DSpine parallel drafter (clean-room).

Independent numpy implementation of the DSpine inference mechanics described in
docs/reference/DSPINE-IMPLEMENTATION-DESIGN.md (paper arXiv:2609.36173): a
DFlash-style block-causal backbone with per-layer adjacent causal injection in a
unified output-embedding transfer space, and predecessor-conditioned decoding via
a transition cache. This is the RED oracle for the GPU kernels and the live
exactness gate; it is written from the published equations, not vendored code.

The backbone reuses the DFlash2 CPU attention/ops helpers (same Qwen3-style
geometry). The DSpine-specific pieces are ``bridge_write`` (the gated adjacent
injection) and ``refine_greedy`` (the transition-cache chain walk).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hipengine.kernels.cpu_reference.dflash2 import dflash2_attention_forward
from hipengine.kernels.cpu_reference.ops import linear, rmsnorm

_F32_TINY = 1.1754943508222875e-38


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


@dataclass
class DSpineBridgeWeights:
    codebook: np.ndarray            # [V, C]
    early_read_weights: np.ndarray  # [L-1, C, H]
    read_weight: np.ndarray         # [C, H] (last layer)
    feature_write_weights: np.ndarray  # [L, H, R]
    gate_weight: np.ndarray         # [R, C]  (G_q)
    gate_bias: np.ndarray           # [R]
    predecessor_gate_weight: np.ndarray  # [R, C] (G_p)
    prev_weight: np.ndarray         # [R, C]  (A)

    def packed(self) -> np.ndarray:
        # [3R, C] = [G_q; G_p; A]
        return np.concatenate(
            [self.gate_weight, self.predecessor_gate_weight, self.prev_weight], axis=0
        )

    def read_weight_for(self, layer: int, num_layers: int) -> np.ndarray:
        return self.read_weight if layer == num_layers - 1 else self.early_read_weights[layer]


def bridge_write(
    raw: np.ndarray,
    input_ids: np.ndarray,
    layer: int,
    num_layers: int,
    bridge: DSpineBridgeWeights,
    next_norm_weight: np.ndarray,
    eps: float,
    *,
    stash: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One adjacent-causal-injection write after ``layer``.

    ``raw`` is the full layer output [T, H] (BF16-valued f32). Returns
    ``(next_norm(out), out)``. Position 0 (anchor) takes its code from the
    codebook and receives no message; the predecessor shift is ``t-1``.
    """
    raw = np.asarray(raw, dtype=np.float32)
    T, H = raw.shape
    C = bridge.codebook.shape[1]
    R = bridge.gate_weight.shape[0]
    sqrt_c = np.float32(np.sqrt(C))

    rms = np.sqrt(np.maximum(np.mean(raw * raw, axis=-1), _F32_TINY))  # [T]
    normalized = (raw / (rms[..., None] + eps)).astype(np.float32)
    projected = normalized @ bridge.read_weight_for(layer, num_layers).T  # [T, C]
    proj_norm = np.sqrt(np.sum(projected * projected, axis=-1, keepdims=True))
    codes = sqrt_c * projected / (proj_norm + eps)  # [T, C]
    anchor_code = sqrt_c * bridge.codebook[int(input_ids[0])]
    codes = np.concatenate([anchor_code[None, :], codes[1:, :]], axis=0)

    packed = codes @ bridge.packed().T  # [T, 3R]
    if stash is not None:
        stash["raw"] = raw
        stash["rms"] = rms
        stash["packed"] = packed
    q = packed[:, :R]
    s = packed[:, R : 2 * R]
    v = packed[:, 2 * R :]
    gate = 1.0 / (1.0 + np.exp(-(q[1:, :] + s[:-1, :] + bridge.gate_bias)))  # [T-1, R]
    message = np.concatenate([np.zeros((1, R)), gate * v[:-1, :]], axis=0)  # [T, R]
    delta = message @ bridge.feature_write_weights[layer].T  # [T, H]
    not_anchor = (np.arange(T) != 0).astype(np.float32)[:, None]
    out = raw + (rms[..., None] * delta) * not_anchor

    normed = rmsnorm(out, next_norm_weight, eps=eps)
    return normed.astype(np.float32), out.astype(np.float32)


class DSpineNumpyDrafter:
    """Numpy DSpine drafter: block-causal backbone + injection + refinement."""

    def __init__(self, config: dict, weights: dict[str, np.ndarray]) -> None:
        self.config = config
        self.w = {k: bf16_to_f32(v) if v.dtype == np.uint16 else np.asarray(v, dtype=np.float32)
                  for k, v in weights.items()}
        df = config["dflash_config"]
        self.hidden_size = int(config["hidden_size"])
        self.num_layers = int(config["num_hidden_layers"])
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config["head_dim"])
        self.eps = float(config["rms_norm_eps"])
        self.rope_theta = float(config.get("rope_theta", 1e7))
        self.sliding_window = int(config.get("sliding_window") or 0)
        self.block_size = int(config["block_size"])
        self.mask_token_id = int(df["mask_token_id"])
        self.head_topk = int(df["selected_prefix_last_write_refine"]["head_topk"])
        b = df["selected_prefix_code_bridge"]
        self.code_size = int(b["code_size"])
        self.rank = int(b["rank"])
        self.bridge = DSpineBridgeWeights(
            codebook=self.w["selected_prefix_bridge.codebook"],
            early_read_weights=self.w["selected_prefix_bridge.early_read_weights"],
            read_weight=self.w["selected_prefix_bridge.read_weight"],
            feature_write_weights=self.w["selected_prefix_bridge.feature_write_weights"],
            gate_weight=self.w["selected_prefix_bridge.gate.weight"],
            gate_bias=self.w["selected_prefix_bridge.gate.bias"],
            predecessor_gate_weight=self.w["selected_prefix_bridge.predecessor_gate.weight"],
            prev_weight=self.w["selected_prefix_bridge.prev.weight"],
        )

    def project_target_hidden(self, taps: np.ndarray) -> np.ndarray:
        projected = linear(taps, self.w["fc.weight"])
        return rmsnorm(projected, self.w["hidden_norm.weight"], eps=self.eps)

    def forward(
        self,
        target_hidden_taps: np.ndarray,
        noise_embedding: np.ndarray,
        positions: np.ndarray,
        input_ids: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Run the backbone + per-layer injection over one block.

        Returns ``(final_normed_hidden [T, H], last_layer_stash)``.
        """
        ctx = self.project_target_hidden(target_hidden_taps)
        residual = np.asarray(noise_embedding, dtype=np.float32)
        x_in = rmsnorm(residual, self.w["layers.0.input_layernorm.weight"], eps=self.eps)
        stash_out: dict = {}
        for layer in range(self.num_layers):
            p = f"layers.{layer}"
            attn = dflash2_attention_forward(
                x_in, ctx, positions,
                self.w[f"{p}.self_attn.q_proj.weight"],
                self.w[f"{p}.self_attn.k_proj.weight"],
                self.w[f"{p}.self_attn.v_proj.weight"],
                self.w[f"{p}.self_attn.o_proj.weight"],
                self.w[f"{p}.self_attn.q_norm.weight"],
                self.w[f"{p}.self_attn.k_norm.weight"],
                num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim, rope_theta=self.rope_theta,
                sliding_window=self.sliding_window, is_causal=True, eps=self.eps,
            )
            h = residual + attn
            hn = rmsnorm(h, self.w[f"{p}.post_attention_layernorm.weight"], eps=self.eps)
            gate = linear(hn, self.w[f"{p}.mlp.gate_proj.weight"])
            up = linear(hn, self.w[f"{p}.mlp.up_proj.weight"])
            mlp_out = linear(_silu(gate) * up, self.w[f"{p}.mlp.down_proj.weight"])
            out = h + mlp_out
            next_norm = (self.w[f"layers.{layer+1}.input_layernorm.weight"]
                         if layer + 1 < self.num_layers else self.w["norm.weight"])
            stash = {} if layer == self.num_layers - 1 else None
            normed_2d, out_2d = bridge_write(
                out[0], input_ids[0], layer, self.num_layers, self.bridge, next_norm,
                self.eps, stash=stash,
            )
            x_in = normed_2d[None]
            residual = out_2d[None]
            if stash is not None:
                stash_out = stash
        return x_in, stash_out

    def refine_greedy(
        self,
        final_hidden: np.ndarray,
        stash: dict,
        output_head_weight: np.ndarray,
        anchor: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transition-cache greedy refinement over proposal rows 1..T-1.

        Returns ``(draft_ids [n], candidates [n, K])``. Builds the [n, K, K]
        conditional score table from the cached last-layer pre-write state and
        walks the greedy chain left-to-right.
        """
        K = self.head_topk
        logits = linear(final_hidden[1:], output_head_weight)  # [n, V]
        cand = np.argsort(-logits, axis=-1)[:, :K]
        cand = np.sort(cand, axis=-1).astype(np.int64)         # [n, K] ascending id
        return self.refine_greedy_from_cand(final_hidden, stash, output_head_weight, anchor, cand)

    def refine_greedy_from_cand(
        self,
        final_hidden: np.ndarray,
        stash: dict,
        output_head_weight: np.ndarray,
        anchor: int,
        cand: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transition-cache score table + greedy chain walk given top-K ``cand``.

        The GPU refine computes ``cand`` on device (WMMA head projection +
        top-16) and calls this for the small [n, K, K] table and chain walk.
        """
        n = final_hidden.shape[0] - 1
        K = self.head_topk
        R = self.rank
        raw = stash["raw"][1:].astype(np.float32)        # [n, H]
        rms = stash["rms"][1:].astype(np.float32)        # [n]
        q = stash["packed"][1:, :R].astype(np.float32)   # [n, R]
        norm_w = self.w["norm.weight"]
        write_w = self.bridge.feature_write_weights[-1]  # [H, R]
        gate_b = self.bridge.gate_bias
        pred_gate_w = self.bridge.predecessor_gate_weight  # [R, C]
        prev_w = self.bridge.prev_weight                   # [R, C]
        sqrt_c = np.float32(np.sqrt(self.code_size))
        head_rows = output_head_weight[cand]                   # [n, K, H]
        base = np.sum((raw[:, None, :] * norm_w) * head_rows, axis=-1)  # [n, K]

        anchor_arr = np.full((1, K), int(anchor), dtype=np.int64)
        sid = np.concatenate([anchor_arr, cand[:-1]], axis=0) if n > 1 else anchor_arr  # [n, K]
        e = sqrt_c * self.bridge.codebook[sid]                 # [n, K, C]
        tvg = e @ pred_gate_w.T                                # [n, K, R]
        tvv = e @ prev_w.T                                     # [n, K, R]
        m = (1.0 / (1.0 + np.exp(-(q[:, None, :] + tvg + gate_b)))) * tvv  # [n, K, R]
        tg_cand = (head_rows * norm_w) @ write_w               # [n, K, R]
        s = m @ tg_cand.swapaxes(1, 2)                         # [n, K, K]
        score = base[:, None, :] + rms[:, None, None] * s      # [n, K(sender), K(recv)]
        idx = np.argmax(score, axis=-1)                        # [n, K]

        # Greedy chain walk: row 0's sender is the anchor (sid[0] broadcasts the
        # anchor across the sender axis), so start at idx[0, 0]; each next row's
        # sender is the previously selected receiver candidate.
        out = np.empty(n, dtype=np.int64)
        j = int(idx[0, 0]) if n > 0 else 0
        for i in range(n):
            out[i] = cand[i, j]
            if i + 1 < n:
                j = int(idx[i + 1, j])
        return out, cand