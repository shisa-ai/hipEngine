"""Single-sequence FP32 reference for the original Inkling 2 Mini target.

Uses decoded expert weights; does not simulate quantized activations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np

from hipengine.kernels.cpu_reference.inkling2_mini import _sigmoid, routed_feed_forward
from hipengine.loading.inkling2_mini import Inkling2TextWeights
from hipengine.models.inkling2_mini import Inkling2Visit, Inkling2VisitPlan


def _rms(x, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def _softmax(x):
    p = np.exp(x - x.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def _read_blocks(query, blocks, partial):
    values = _rms(np.stack((*blocks, partial)))
    return _softmax((query @ values.T) / sqrt(partial.size)) @ values


def _clip(x, limit):
    return x * np.minimum(1, limit / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6))


@dataclass(slots=True)
class _VisitState:
    keys: np.ndarray
    values: np.ndarray
    conv: dict[str, np.ndarray] = field(default_factory=dict)

    def convolve(self, name, x, taps, *, dynamic=False):
        n = taps.shape[-1]
        history = self.conv.setdefault(name, np.zeros(taps.shape, dtype=np.float32))
        # All taps are in current-to-oldest order at this call boundary.
        history[:, 1:] = history[:, :-1]
        history[:, 0] = x
        return (x + np.sum(history * taps, axis=-1)) / sqrt((2 if dynamic else 1) * n + 1)


class Inkling2TextReference:
    def __init__(self, config, weights: Inkling2TextWeights):
        self.config = config["text_config"]
        self.weights = weights
        self.plan = Inkling2VisitPlan.from_config(config)
        self.project_expert = None
        self.convolution_factory = None

    def project(self, name, hidden, *, limit=None):
        weight = self.weights.dense(name) if limit is None else self.weights.dense(name, slice(limit))
        return hidden @ weight.T

    def session(self, capacity: int) -> Inkling2TextSession:
        return Inkling2TextSession(self, capacity)

    def head_shape(self, visit: Inkling2Visit):
        prefix = "swa_" if visit.is_local else ""
        return tuple(self.config[prefix + k] for k in ("num_attention_heads", "num_key_value_heads", "head_dim"))


class Inkling2TextSession:
    def __init__(self, model: Inkling2TextReference, capacity: int):
        if not 0 < capacity <= model.config.get("model_max_length", capacity):
            raise ValueError(f"invalid Inkling 2 session capacity: {capacity}")
        self.model = model
        self.capacity = capacity
        self.position = 0
        self.states = tuple(self._new_state(v) for v in sorted(model.plan.visits, key=lambda v: v.state_slot))
        self.convolution = model.convolution_factory() if model.convolution_factory is not None else None

    def close(self):
        if self.convolution is not None:
            self.convolution.close()

    def _convolve(self, visit, name, x, taps, *, dynamic=False):
        if self.convolution is not None:
            return self.convolution.step(visit.state_slot, name, x, taps, dynamic=dynamic)
        return self.states[visit.state_slot].convolve(name, x, taps, dynamic=dynamic)

    def _new_state(self, visit):
        _, nkv, dim = self.model.head_shape(visit)
        cap = min(self.capacity, self.model.config["sliding_window_size"]) if visit.is_local else self.capacity
        return _VisitState(np.empty((cap, nkv, dim), np.float32), np.empty((cap, nkv, dim), np.float32))

    def step(self, token: int, *, embedding: np.ndarray | None = None, logits: bool = True,
             capture=None) -> np.ndarray | None:
        if self.position >= self.capacity:
            raise ValueError("Inkling 2 session capacity exceeded")
        c, w = self.model.config, self.model.weights
        if embedding is None:
            if not 0 <= token < c["vocab_size"]:
                raise ValueError(f"Inkling 2 token out of range: {token}")
            h = _rms(w.dense("embed.weight", token), c["rms_norm_eps"]) * w.dense("embed_norm.weight")
        else:
            h = np.asarray(embedding, dtype=np.float32)
        blocks = []
        loop_start = c["loop_config"]["core_start"]
        for boundary, visit in enumerate(self.model.plan.visits):
            layer = visit.weight_layer
            if layer == loop_start:
                if visit.restart_from_prelude:
                    # Drop the common entry block: messages read only residuals of the previous core pass.
                    previous, previous_partial = blocks[len(prelude_blocks) + 1:], h
                    h, blocks = prelude_hidden, prelude_blocks.copy()
                else:
                    prelude_hidden, prelude_blocks = h, blocks.copy()
            if capture is not None:
                capture(boundary, h)
            messages = None
            if visit.message_pass is not None:
                i = 2 * (layer - loop_start)
                q, mlp = w.dense("middle_cycle_cross_pass_queries", slice(i, i + 2))
                k, v = w.dense("middle_cycle_cross_pass_kv_queries", slice(i, i + 2))
                messages = _read_blocks(np.stack((q, k, v, mlp)), previous, previous_partial)
            h = self._layer(visit, h, blocks, messages)

        self.position += 1
        if not logits and capture is None:
            return None
        h = _rms(_read_blocks(w.dense("attn_res.output_query"), blocks, h), c["rms_norm_eps"]) * w.dense("norm.weight")
        if capture is not None:
            capture(len(self.model.plan.visits), h)
        if not logits:
            return None
        raw = self.model.project("unembed.weight", h, limit=c["unpadded_vocab_size"])
        cap = c["final_logit_softcapping"]
        return cap * np.tanh(raw / cap)

    def _layer(self, visit, h, blocks, messages):
        c, w = self.model.config, self.model.weights
        layer = visit.weight_layer
        prefix = f"layers.{layer}."
        eps = c["rms_norm_eps"]

        def get(name):
            return w.dense(prefix + name)

        def project(name, x):
            return self.model.project(prefix + name, x)

        def merge(x, name, message):
            ctl = f"controllers.{name}."
            z = project(ctl + "down_weight", np.concatenate((_rms(x, eps), _rms(message, eps))))
            return (1 - np.tanh(project(ctl + "forget_weight", z))) * x + np.tanh(project(ctl + "gain_weight", z)) * message

        def norm(x, name):
            return _rms(x, eps) * get(name + ".weight")

        def static_conv(name, x):
            taps = get(name + ".weight")[:, 0, ::-1]
            return self._convolve(visit, name, x, taps)

        a = _read_blocks(w.dense(f"attn_res.attn_queries.{layer - 1}"), blocks, h) if layer else h
        inputs = (a, a, a) if a.ndim == 1 else a
        norm_names = ("attn_norm",) * 3 if visit.is_local else ("q_input_norm", "k_input_norm", "v_input_norm")
        if messages is None and visit.is_local:
            q, k, v = (norm(a, "attn_norm"),) * 3
        else:
            q, k, v = (norm(x if messages is None else merge(x, name, messages[i]), norm_names[i])
                       for i, (x, name) in enumerate(zip(inputs, ("attn", "k", "v"))))

        a = static_conv("attn_sconv", self._attention(visit, self.states[visit.state_slot], get, project, q, k, v))
        if layer % c["n_layers_per_attn_res_block"] == 0:
            blocks.append(h)
        else:
            a = h + a
        m = _read_blocks(w.dense(f"attn_res.mlp_queries.{layer}"), blocks, a)
        if messages is not None:
            m = merge(m, "mlp", messages[3])
        m = routed_feed_forward(w, layer, norm(m, "mlp_norm"), top_k=c["num_experts_per_tok"],
                                route_scale=c["route_scale"], gate_cap=c["mlp_gate_softcap"], up_cap=c["mlp_up_softcap"],
                                project=self.model.project_expert, project_dense=self.model.project)
        return a + static_conv("mlp_sconv", m)

    def _attention(self, visit, s, get, project, qi, ki, vi):
        c = self.model.config
        nh, nkv, dim = self.model.head_shape(visit)
        ksize = c["sconv_kernel_size"]
        pos = self.position

        def weight(name):
            return get("attn." + name)

        def linear(name, x):
            return project("attn." + name, x)

        def project_kv(name, x):
            y = linear(f"w{name}_dv.weight", x)
            conv = f"{name}_sconv."
            if visit.is_local:
                taps = weight(conv + "weight")[:, 0, ::-1]
            else:
                dynamic = linear(conv + "weight_proj", x).reshape(nkv, 1, ksize)
                taps = (weight(conv + "static_w").T.reshape(nkv, dim, ksize) + dynamic).reshape(-1, ksize)
            return self._convolve(visit, name, y, taps, dynamic=not visit.is_local).reshape(nkv, dim)

        q = linear("wq_du.weight", qi).reshape(nh, dim)
        q = _clip(q * weight("q_l2_clip.scale"), c["qk_l2_clip_threshold"])
        key = _clip(project_kv("k", ki), c["qk_l2_clip_threshold"])
        value = _rms(project_kv("v", vi), c["rms_norm_eps"])
        rel = linear("wr_du.weight", qi).reshape(nh, -1)
        rel = _clip(rel * weight("rel_l2_clip.scale"), c["rel_l2_clip_threshold"])
        if not visit.is_local:
            scale = 1 + c["log_scaling_alpha"] * np.log(max(1, (pos + 1) / c["log_scaling_n_floor"]))
            q *= scale
            rel *= scale

        capacity = len(s.keys)
        slot = pos % capacity
        s.keys[slot], s.values[slot] = key, value
        distances = (slot - np.arange(min(pos + 1, capacity))) % capacity
        used = distances.size
        groups = nh // nkv
        scores = np.einsum("kgd,tkd->kgt", q.reshape(nkv, groups, dim), s.keys[:used]).reshape(nh, used) / sqrt(dim)
        extent = c["sliding_window_size"] if visit.is_local else c["rel_extent"]
        near = distances < extent
        scores[:, near] += (rel @ weight("rel_logits_proj.proj"))[:, distances[near]]
        sink = linear("attn_sink_proj.weight", qi)
        p = _softmax(np.concatenate((scores, sink[:, None]), axis=-1))[:, :used]
        y = np.einsum("kgt,tkd->kgd", p.reshape(nkv, groups, used), s.values[:used]).reshape(-1)
        return linear("wo_ud.weight", y * _sigmoid(linear("gate_attn.weight", qi)))
