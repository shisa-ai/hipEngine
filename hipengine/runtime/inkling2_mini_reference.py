"""Single-sequence FP32 reference for the original Inkling 2 Mini target.

Uses decoded expert weights; does not simulate quantized activations.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from math import sqrt

import numpy as np

from hipengine.kernels.cpu_reference.inkling2_mini import _sigmoid, routed_feed_forward
from hipengine.loading.inkling2_mini import Inkling2TextWeights
from hipengine.models.inkling2_mini import Inkling2Visit, Inkling2VisitPlan
from hipengine.runtime.inkling2_undo import Inkling2UndoJournal


def _rms(x, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def _softmax(x):
    p = np.exp(x - x.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def _read_blocks(query, normalized_blocks, partial):
    values = np.stack((*normalized_blocks, _rms(np.ascontiguousarray(partial))), axis=-2)
    if partial.ndim == 1:
        return _softmax((query @ values.T) / sqrt(partial.size)) @ values
    if query.ndim == 1:
        scores = values @ query / sqrt(partial.shape[-1])
        return np.einsum("bp,bpd->bd", _softmax(scores), values)
    scores = query @ values.swapaxes(-1, -2) / sqrt(partial.shape[-1])
    return _softmax(scores) @ values


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
        self.project_experts_group = None
        self.feed_experts_group = None
        self.convolution_factory = None
        self.attention_factory = None
        self.project_device = None

    def project(self, name, hidden, *, limit=None):
        weight = self.weights.dense(name) if limit is None else self.weights.dense(name, slice(limit))
        return hidden @ weight.T

    def project_logits(self, hidden):
        return self.project("unembed.weight", hidden, limit=self.config["unpadded_vocab_size"])

    def project_many(self, names, hidden):
        return tuple(self.project(name, hidden) for name in names)

    def project_groups(self, groups):
        return tuple(self.project_many(names, hidden) for names, hidden in groups)

    def project_chains(self, chains):
        return tuple(self.project_many(names, self.project(first, hidden)) for first, names, hidden in chains)

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
        self._undo = None
        self.attention = model.attention_factory(capacity) if model.attention_factory is not None else None
        try:
            self.states = tuple(self._new_state(v) for v in sorted(model.plan.visits, key=lambda v: v.state_slot))
            self.convolution = model.convolution_factory() if model.convolution_factory is not None else None
        except BaseException:
            if self.attention is not None:
                self.attention.close()
            raise

    def close(self):
        try:
            if self.convolution is not None:
                self.convolution.close()
        finally:
            if self.attention is not None:
                self.attention.close()

    def _convolve(self, visit, name, x, taps, *, dynamic=False, offset=0):
        position = self.position + offset
        if self.convolution is not None:
            return self.convolution.step(visit.state_slot, name, x, taps, dynamic=dynamic,
                                         position=position, undo=self._undo)
        state = self.states[visit.state_slot]
        if self._undo is not None:
            history = state.conv.get(name)
            if history is None:
                self._undo.before(position, state.conv.pop, name, None)
            else:
                self._undo.host(position, history)
        return state.convolve(name, x, taps, dynamic=dynamic)

    def _new_state(self, visit):
        if self.attention is not None:
            return _VisitState(None, None)
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
        return self._forward(h, 1, logits=logits, capture=capture)

    def step_block(self, tokens, *, capture=None):
        """Append a nonempty token block and project every row for verification."""
        return self._step_block(tokens, logits=True, capture=capture, last_only=False)

    def prefill_block(self, tokens, *, logits=True, capture=None):
        """Commit a prompt block, projecting only the last row if requested."""
        return self._step_block(tokens, logits=logits, capture=capture, last_only=True)

    def _step_block(self, tokens, *, logits, capture, last_only):
        tokens = tuple(tokens)
        if not tokens or self.position + len(tokens) > self.capacity:
            raise ValueError("Inkling 2 block exceeds session capacity or is empty")
        c, w = self.model.config, self.model.weights
        if any(not 0 <= token < c["vocab_size"] for token in tokens):
            raise ValueError("Inkling 2 block contains an out-of-range token")
        h = _rms(w.dense("embed.weight", list(tokens)), c["rms_norm_eps"]) * w.dense("embed_norm.weight")
        return self._forward(h, len(tokens), logits=logits, capture=capture, last_only=last_only)

    @contextmanager
    def provisional_block(self, tokens, *, capture=None):
        """Stage a block and commit only the prefix selected by the verifier."""
        if self._undo is not None:
            raise RuntimeError("Inkling 2 already has a provisional block")
        tokens = tuple(tokens)
        start = self.position
        retained = 0
        journal = self._undo = Inkling2UndoJournal()

        def keep(count):
            nonlocal retained
            if not isinstance(count, int) or not 0 <= count <= len(tokens):
                raise ValueError("invalid Inkling 2 accepted block prefix")
            retained = count

        try:
            scores = self.step_block(tokens, capture=capture)
            yield scores, keep
        except BaseException:
            retained = 0
            raise
        finally:
            try:
                journal.finish(start + retained)
            finally:
                self._undo = None
                self.position = start + retained

    def _forward(self, h, width, *, logits, capture, last_only=False):
        c, w = self.model.config, self.model.weights
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

        self.position += width
        if not logits and capture is None:
            return None
        if last_only and capture is None:
            h = h[-1]
            blocks = [block[-1] for block in blocks]
        h = _rms(_read_blocks(w.dense("attn_res.output_query"), blocks, h), c["rms_norm_eps"]) * w.dense("norm.weight")
        if capture is not None:
            capture(len(self.model.plan.visits), h)
            if last_only:
                h = h[-1]
        if not logits:
            return None
        raw = self.model.project_logits(h)
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

        def controller_input(x, message):
            return np.concatenate((_rms(x, eps), _rms(message, eps)), axis=-1)

        def blend(x, message, forget, gain):
            return (1 - np.tanh(forget)) * x + np.tanh(gain) * message

        def merge_controllers(inputs, names, incoming):
            chains = []
            for x, name, message in zip(inputs, names, incoming, strict=True):
                ctl = f"{prefix}controllers.{name}."
                chains.append((ctl + "down_weight", (ctl + "forget_weight", ctl + "gain_weight"),
                               controller_input(x, message)))
            controls = self.model.project_chains(chains)
            return tuple(blend(x, message, forget, gain)
                         for x, message, (forget, gain) in zip(inputs, incoming, controls, strict=True))

        def norm(x, name):
            return _rms(x, eps) * get(name + ".weight")

        def static_conv(name, x):
            taps = get(name + ".weight")[:, 0, ::-1]
            return self._conv_rows(visit, name, x, taps)

        def device_conv(name):
            taps = get(name + ".weight")[:, 0, ::-1]
            return partial(self.convolution.apply_static, visit.state_slot, name, taps,
                           position=self.position, undo=self._undo)

        a = _read_blocks(w.dense(f"attn_res.attn_queries.{layer - 1}"), blocks, h) if layer else h
        inputs = (a, a, a) if a.ndim == h.ndim else np.moveaxis(a, -2, 0)
        norm_names = ("attn_norm",) * 3 if visit.is_local else ("q_input_norm", "k_input_norm", "v_input_norm")
        if messages is None and visit.is_local:
            q, k, v = (norm(a, "attn_norm"),) * 3
        else:
            merged = inputs if messages is None else merge_controllers(
                inputs, ("attn", "k", "v"), tuple(messages[..., i, :] for i in range(3)))
            q, k, v = (norm(x, name) for x, name in zip(merged, norm_names, strict=True))

        attention_project = None
        if q.ndim == 1 and self.model.project_device is not None and self.attention is not None and self.convolution is not None:
            def attention_project(pos, q, k, v, bias, sink, gain):
                with self.model.project_device(prefix + "attn.wo_ud.weight", gain.size,
                                               transform_output=device_conv("attn_sconv")) as projection:
                    return self.attention.step(visit.state_slot, pos, q, k, v, bias, sink,
                                               undo=self._undo, gain=gain, projection=projection)

        a = self._attention(visit, self.states[visit.state_slot], get, q, k, v, attention_project=attention_project)
        if attention_project is None:
            if self.convolution is None:
                a = static_conv("attn_sconv", project("attn.wo_ud.weight", a))
            else:
                a = self.model.project(prefix + "attn.wo_ud.weight", a, transform_output=device_conv("attn_sconv"))
        if layer % c["n_layers_per_attn_res_block"] == 0:
            blocks.append(_rms(np.ascontiguousarray(h)))
        else:
            a = h + a
        m = _read_blocks(w.dense(f"attn_res.mlp_queries.{layer}"), blocks, a)
        if messages is not None:
            m, = merge_controllers((m,), ("mlp",), (messages[..., 3, :],))
        normalized = norm(m, "mlp_norm")

        feed_group = self.model.feed_experts_group
        convolve_on_device = feed_group is not None and self.convolution is not None
        if convolve_on_device:
            feed_group = partial(feed_group, transform_output=device_conv("mlp_sconv"))

        def feed(x):
            return routed_feed_forward(w, layer, x, top_k=c["num_experts_per_tok"],
                                       route_scale=c["route_scale"], gate_cap=c["mlp_gate_softcap"],
                                       up_cap=c["mlp_up_softcap"], project=self.model.project_expert,
                                       project_dense=self.model.project, project_group=self.model.project_experts_group,
                                       feed_group=feed_group)

        if normalized.ndim == 1 or feed_group is not None:
            m = feed(normalized)
        else:
            m = np.stack([feed(row) for row in normalized])
        return a + (m if convolve_on_device else static_conv("mlp_sconv", m))

    def _conv_rows(self, visit, name, x, taps, *, dynamic=False):
        if x.ndim == 1:
            return self._convolve(visit, name, x, taps, dynamic=dynamic)
        return np.stack([self._convolve(visit, name, row, taps[i] if dynamic else taps, dynamic=dynamic, offset=i)
                         for i, row in enumerate(x)])

    def _attention(self, visit, s, get, qi, ki, vi, *, attention_project=None):
        c = self.model.config
        nh, nkv, dim = self.model.head_shape(visit)
        ksize = c["sconv_kernel_size"]
        pos = self.position
        batch = qi.ndim == 2
        leading = (len(qi),) if batch else ()

        def weight(name):
            return get("attn." + name)

        attn_prefix = f"layers.{visit.weight_layer}.attn."

        def names_in_layer(names):
            return tuple(attn_prefix + name for name in names)

        def kv_taps(name, dynamic=None):
            conv = f"{name}_sconv."
            if dynamic is None:
                return weight(conv + "weight")[:, 0, ::-1]
            dynamic = dynamic.reshape(*leading, nkv, 1, ksize)
            return (weight(conv + "static_w").T.reshape(nkv, dim, ksize) + dynamic).reshape(*leading, -1, ksize)

        query_names = ("wq_du.weight", "wr_du.weight", "attn_sink_proj.weight", "gate_attn.weight")
        key_names, value_names = ("wk_dv.weight",), ("wv_dv.weight",)
        dynamic = not visit.is_local
        if visit.is_local and qi is ki and qi is vi:
            q, rel, sink, gate, k, v = self.model.project_many(names_in_layer((*query_names, *key_names, *value_names)), qi)
            kt, vt = kv_taps("k"), kv_taps("v")
        else:
            if dynamic:
                key_names += ("k_sconv.weight_proj",)
                value_names += ("v_sconv.weight_proj",)
            (q, rel, sink, gate), (k, *kd), (v, *vd) = self.model.project_groups(
                ((names_in_layer(query_names), qi), (names_in_layer(key_names), ki), (names_in_layer(value_names), vi)))
            kt, vt = kv_taps("k", *kd), kv_taps("v", *vd)
        if self.convolution is None:
            k = self._conv_rows(visit, "k", k, kt, dynamic=dynamic)
            v = self._conv_rows(visit, "v", v, vt, dynamic=dynamic)
        else:
            k, v = self.convolution.step_kv(visit.state_slot, k, v, kt, vt,
                                            dynamic=dynamic, position=pos, undo=self._undo)
        q = q.reshape(*leading, nh, dim)
        q = _clip(q * weight("q_l2_clip.scale"), c["qk_l2_clip_threshold"])
        key = _clip(k.reshape(*leading, nkv, dim), c["qk_l2_clip_threshold"])
        value = _rms(v.reshape(*leading, nkv, dim), c["rms_norm_eps"])
        rel = rel.reshape(*leading, nh, -1)
        rel = _clip(rel * weight("rel_l2_clip.scale"), c["rel_l2_clip_threshold"])
        if not visit.is_local:
            positions = pos + 1 + np.arange(len(qi)) if batch else pos + 1
            scale = 1 + c["log_scaling_alpha"] * np.log(np.maximum(1, positions / c["log_scaling_n_floor"]))
            if batch:
                scale = scale[:, None, None]
            q *= scale
            rel *= scale

        extent = c["sliding_window_size"] if visit.is_local else c["rel_extent"]
        bias = (rel @ weight("rel_logits_proj.proj"))[..., :extent]
        gain = _sigmoid(gate)
        if attention_project is not None:
            return attention_project(pos, q, key, value, bias, sink, gain)
        if batch and self.attention is not None:
            y = self.attention.step_block(visit.state_slot, pos, q, key, value, bias, sink, undo=self._undo)
            y = y.reshape(len(q), -1)
        elif batch:
            y = np.stack([self._attend(visit, s, pos + i, *args)
                          for i, args in enumerate(zip(q, key, value, bias, sink))])
        else:
            y = self._attend(visit, s, pos, q, key, value, bias, sink)
        return y * gain

    def _attend(self, visit, s, pos, q, key, value, bias, sink):
        if self.attention is not None:
            return self.attention.step(visit.state_slot, pos, q, key, value, bias, sink, undo=self._undo).reshape(-1)
        nh, nkv, dim = self.model.head_shape(visit)
        extent = bias.shape[-1]
        capacity = len(s.keys)
        slot = pos % capacity
        if self._undo is not None and pos >= capacity:
            self._undo.host(pos, s.keys[slot])
            self._undo.host(pos, s.values[slot])
        s.keys[slot], s.values[slot] = key, value
        distances = (slot - np.arange(min(pos + 1, capacity))) % capacity
        used = distances.size
        groups = nh // nkv
        scores = np.einsum("kgd,tkd->kgt", q.reshape(nkv, groups, dim), s.keys[:used]).reshape(nh, used) / sqrt(dim)
        near = distances < extent
        scores[:, near] += bias[:, distances[near]]
        p = _softmax(np.concatenate((scores, sink[:, None]), axis=-1))[:, :used]
        return np.einsum("kgt,tkd->kgd", p.reshape(nkv, groups, used), s.values[:used]).reshape(-1)
