"""FP32 reference for the original Inkling 2 Mini DSpark draft network."""

from __future__ import annotations

from math import sqrt

import numpy as np

from hipengine.kernels.cpu_reference.inkling2_mini import _sigmoid
from hipengine.kernels.cpu_reference.muse_glimmer_dflash import apply_rope_neox, rope_tables_neox
from hipengine.loading.inkling2_mini import Inkling2TextWeights
from hipengine.loading.safetensors import load_weight_index
from hipengine.runtime.inkling2_mini_reference import _rms, _softmax


def _markov_update(state, joint):
    gate, candidate, latent = np.split(joint, 3)
    candidate = np.tanh(candidate)
    state = candidate + _sigmoid(gate) * (state - candidate)
    return state, np.tanh(latent)


def markov_step(state, token, hidden, context, base, joint_w, joint_b, out_w, out_b, head, cap):
    state, latent = _markov_update(state, joint_w @ np.concatenate((state, token, hidden, context)) + joint_b)
    correction = head @ (out_w @ latent + out_b)
    return cap * np.tanh((base + correction) / cap), state


class Inkling2DSpark:
    @classmethod
    def from_safetensors(cls, model, draft_path):
        index = load_weight_index(draft_path)
        return cls(model, Inkling2TextWeights(index, namespace=""), index.config["mtp_config"])

    def __init__(self, model, weights, config):
        self.cfg = config
        self.target = model
        self.kv_factory = None
        self.weights = weights
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

    def project(self, name, rows):
        return rows @ self.weights.dense(name).T

    def attend(self, q, k, v):
        rows, heads, width = q.shape
        nk = k.shape[1]
        q = q.reshape(rows, nk, heads // nk, width)
        logits = np.einsum("qngd,tnd->qngt", q, k) / sqrt(width)
        return np.einsum("qngt,tnd->qngd", _softmax(logits), v).reshape(rows, heads, width)

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
        self.kv = model.kv_factory(capacity, cfg) if model.kv_factory is not None else None
        self.keys = []
        self.values = []
        if self.kv is None:
            for layer in range(cfg["num_hidden_layers"]):
                length = min(capacity, cfg["sliding_window_size"]) if layer in cfg["local_layer_ids"] else capacity
                self.keys.append(np.empty((length, *shape), dtype=np.float32))
                self.values.append(np.empty_like(self.keys[-1]))

    def close(self):
        if self.kv is not None:
            self.kv.close()

    def append(self, features):
        """Project one committed token's target taps, preserving their logical order."""
        if self.position >= self.capacity:
            raise ValueError("Inkling 2 DSpark context capacity exceeded")
        m, c = self.model, self.model.cfg
        read, eps = m.weights.dense, c["rms_norm_eps"]
        mixed = m.fuse @ np.stack([features[i] for i in m.taps])
        cos, sin = rope_tables_neox(np.array([self.position]), c["head_dim"], c["rope_theta"])
        shape = (c["num_key_value_heads"], c["head_dim"])
        for layer in range(c["num_hidden_layers"]):
            prefix = f"layers.{layer}."
            context = _rms(mixed[layer], eps) * read(prefix + "dflare_norm.weight")
            a = prefix + "self_attn."
            k = m.project(a + "k_proj_ctx.weight", context).reshape(shape)
            k = _rms(k, eps) * read(a + "k_norm_ctx.weight")
            k = apply_rope_neox(k[None], cos, sin)[0]
            v = m.project(a + "v_proj_ctx.weight", context).reshape(shape)
            if self.kv is None:
                slot = self.position % len(self.keys[layer])
                self.keys[layer][slot], self.values[layer][slot] = k, v
            else:
                self.kv.append(layer, self.position, k, v)
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
        for layer in range(c["num_hidden_layers"]):
            prefix = f"layers.{layer}."
            a = prefix + "self_attn."
            x = _rms(h, eps) * read(prefix + "input_layernorm.weight")
            q = m.project(a + "q_proj.weight", x).reshape(-1, nh, dim)
            k = m.project(a + "k_proj.weight", x).reshape(-1, nk, dim)
            v = m.project(a + "v_proj.weight", x).reshape(-1, nk, dim)
            q = apply_rope_neox(_rms(q, eps) * read(a + "q_norm.weight"), cos, sin)
            k = apply_rope_neox(_rms(k, eps) * read(a + "k_norm.weight"), cos, sin)
            # All draft rows see the entire noise block and the same target prefix window.
            start = 0 if layer not in c["local_layer_ids"] else max(0, self.position - c["sliding_window_size"] + 1)
            if self.kv is None:
                slots = np.arange(start, self.position) % len(self.keys[layer])
                k = np.concatenate((self.keys[layer][slots], k), axis=0)
                v = np.concatenate((self.values[layer][slots], v), axis=0)
                out = m.attend(q, k, v)
            else:
                out = self.kv.attend(layer, start, self.position, q, k, v)
            out = out.reshape(len(h), nh * dim)
            h = h + m.project(a + "o_proj.weight", out)
            x = _rms(h, eps) * read(prefix + "post_attention_layernorm.weight")
            g = m.project(prefix + "mlp.gate_proj.weight", x)
            u = m.project(prefix + "mlp.up_proj.weight", x)
            h = h + m.project(prefix + "mlp.down_proj.weight", g * _sigmoid(g) * u)
        h = _rms(h[1:], eps) * read("norm.weight")
        tcfg = m.target.config
        def unembed(rows):
            return m.target.project("unembed.weight", rows, limit=tcfg["unpadded_vocab_size"])

        base = unembed(h)
        joint_b = read("markov_head.rnn_proj.bias")
        out_b = read("markov_head.hidden_out.bias")
        state = np.zeros(c["markov_rank"], dtype=np.float32)
        cap = tcfg["final_logit_softcapping"]
        ids = []
        previous = anchor
        for hidden, raw in zip(h, base):
            token = read("markov_head.markov_w1.weight", previous)
            joint = m.project("markov_head.rnn_proj.weight", np.concatenate((state, token, hidden, self.context))) + joint_b
            state, latent = _markov_update(state, joint)
            correction = unembed(m.project("markov_head.hidden_out.weight", latent) + out_b)
            logits = cap * np.tanh((raw + correction) / cap)
            previous = int(np.argmax(logits))
            ids.append(previous)
        return tuple(ids)
