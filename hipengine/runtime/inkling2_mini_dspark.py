"""FP32 reference for the original Inkling 2 Mini DSpark draft network."""

from __future__ import annotations

from math import sqrt

import numpy as np

from hipengine.kernels.cpu_reference.inkling2_mini import _sigmoid
from hipengine.kernels.cpu_reference.muse_glimmer_dflash import apply_rope_neox, rope_tables_neox
from hipengine.loading.inkling2_mini import Inkling2TextWeights
from hipengine.loading.safetensors import load_weight_index
from hipengine.runtime.inkling2_mini_reference import _rms, _softmax


def markov_step(state, token, hidden, context, base, joint_w, joint_b, out_w, out_b, head, cap):
    gate, candidate, latent = np.split(joint_w @ np.concatenate((state, token, hidden, context)) + joint_b, 3)
    candidate = np.tanh(candidate)
    state = candidate + _sigmoid(gate) * (state - candidate)
    correction = head @ (out_w @ np.tanh(latent) + out_b)
    return cap * np.tanh((base + correction) / cap), state


class Inkling2DSpark:
    def __init__(self, model, draft_path):
        index = load_weight_index(draft_path)
        self.cfg = index.config["mtp_config"]
        self.target = model
        self.weights = Inkling2TextWeights(index, namespace="")
        self.taps = tuple(self.cfg["target_layer_ids"])
        if (self.cfg["markov_head_type"] != "rnn_hidden" or self.cfg.get("q_bias") or self.cfg.get("o_bias")
                or self.cfg["hidden_size"] != model.config["hidden_size"]):
            raise ValueError("unsupported Inkling 2 DSpark recurrent-head or attention geometry")
        if (not self.taps or self.taps != tuple(sorted(set(self.taps))) or self.taps[0] < 0
                or self.taps[-1] != len(model.plan.visits)):
            raise ValueError("Inkling 2 DSpark taps do not match target logical boundaries")
        self.fuse = _softmax(self.weights.dense("fuse_weight"))
        if self.fuse.shape != (self.cfg["num_hidden_layers"] + 1, len(self.taps)):
            raise ValueError("Inkling 2 DSpark fuse weights do not match draft depth and target taps")

    def session(self, capacity):
        return Inkling2DSparkSession(self, capacity)


class Inkling2DSparkSession:
    def __init__(self, model, capacity):
        self.model = model
        self.position = 0
        self.context = None
        cfg = model.cfg
        self.capacity = capacity
        shape = (cfg["num_key_value_heads"], cfg["head_dim"])
        self.keys = []
        self.values = []
        for layer in range(cfg["num_hidden_layers"]):
            length = min(capacity, cfg["sliding_window_size"]) if layer in cfg["local_layer_ids"] else capacity
            self.keys.append(np.empty((length, *shape), dtype=np.float32))
            self.values.append(np.empty_like(self.keys[-1]))

    def append(self, features):
        """Project one committed token's target taps, preserving their logical order."""
        if self.position >= self.capacity:
            raise ValueError("Inkling 2 DSpark context capacity exceeded")
        m, c = self.model, self.model.cfg
        read, eps = m.weights.dense, c["rms_norm_eps"]
        mixed = m.fuse @ np.stack([features[i] for i in m.taps])
        cos, sin = rope_tables_neox(np.array([self.position]), c["head_dim"], c["rope_theta"])
        for layer, (keys, values) in enumerate(zip(self.keys, self.values)):
            prefix = f"layers.{layer}."
            context = _rms(mixed[layer], eps) * read(prefix + "dflare_norm.weight")
            a = prefix + "self_attn."
            k = (read(a + "k_proj_ctx.weight") @ context).reshape(keys.shape[1:])
            k = _rms(k, eps) * read(a + "k_norm_ctx.weight")
            slot = self.position % len(keys)
            keys[slot] = apply_rope_neox(k[None], cos, sin)[0]
            values[slot] = (read(a + "v_proj_ctx.weight") @ context).reshape(keys.shape[1:])
        self.context = _rms(mixed[-1], eps) * read("markov_head.dflare_norm.weight")
        self.position += 1

    def propose(self, anchor, count):
        """Predict up to block_size-1 tokens without mutating committed context."""
        m, c = self.model, self.model.cfg
        if self.context is None or not 0 < count < c["block_size"]:
            raise ValueError("Inkling 2 DSpark needs target context and a valid draft block size")
        read, eps = m.weights.dense, c["rms_norm_eps"]
        target_read = m.target.weights.dense
        teps = m.target.config["rms_norm_eps"]
        # The anchor is the next token selected by the target, not yet cached.
        first = _rms(target_read("embed.weight", anchor), teps) * target_read("embed_norm.weight")
        h = np.vstack((first, np.broadcast_to(read("mask_embed"), (count, first.size))))
        nh, nk, dim = (c[k] for k in ("num_attention_heads", "num_key_value_heads", "head_dim"))
        cos, sin = rope_tables_neox(np.arange(self.position, self.position + len(h)), dim, c["rope_theta"])
        groups = nh // nk
        for layer, (keys, values) in enumerate(zip(self.keys, self.values)):
            prefix = f"layers.{layer}."
            a = prefix + "self_attn."
            x = _rms(h, eps) * read(prefix + "input_layernorm.weight")
            q = (x @ read(a + "q_proj.weight").T).reshape(-1, nh, dim)
            k = (x @ read(a + "k_proj.weight").T).reshape(-1, nk, dim)
            v = (x @ read(a + "v_proj.weight").T).reshape(-1, nk, dim)
            q = apply_rope_neox(_rms(q, eps) * read(a + "q_norm.weight"), cos, sin)
            k = apply_rope_neox(_rms(k, eps) * read(a + "k_norm.weight"), cos, sin)
            # All draft rows see the entire noise block and the same target prefix window.
            start = 0 if layer not in c["local_layer_ids"] else max(0, self.position - c["sliding_window_size"] + 1)
            slots = np.arange(start, self.position) % len(keys)
            k = np.concatenate((keys[slots], k), axis=0)
            v = np.concatenate((values[slots], v), axis=0)
            q = q.reshape(len(h), nk, groups, dim)
            logits = np.einsum("qngd,tnd->qngt", q, k) / sqrt(dim)
            out = np.einsum("qngt,tnd->qngd", _softmax(logits), v).reshape(len(h), nh * dim)
            h = h + out @ read(a + "o_proj.weight").T
            x = _rms(h, eps) * read(prefix + "post_attention_layernorm.weight")
            g = x @ read(prefix + "mlp.gate_proj.weight").T
            u = x @ read(prefix + "mlp.up_proj.weight").T
            h = h + (g * _sigmoid(g) * u) @ read(prefix + "mlp.down_proj.weight").T
        h = _rms(h[1:], eps) * read("norm.weight")
        tcfg = m.target.config
        head = target_read("unembed.weight", slice(tcfg["unpadded_vocab_size"]))
        base = h @ head.T
        joint_w = read("markov_head.rnn_proj.weight")
        joint_b = read("markov_head.rnn_proj.bias")
        out_w = read("markov_head.hidden_out.weight")
        out_b = read("markov_head.hidden_out.bias")
        state = np.zeros(c["markov_rank"], dtype=np.float32)
        cap = tcfg["final_logit_softcapping"]
        ids = []
        previous = anchor
        for hidden, raw in zip(h, base):
            token = read("markov_head.markov_w1.weight", previous)
            logits, state = markov_step(state, token, hidden, self.context, raw, joint_w, joint_b, out_w, out_b, head, cap)
            previous = int(np.argmax(logits))
            ids.append(previous)
        return tuple(ids)
