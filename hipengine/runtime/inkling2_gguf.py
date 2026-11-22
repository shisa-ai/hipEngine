"""Bind GGUF tensor layouts to the shared Inkling text traversal."""

from functools import partial

import numpy as np

from hipengine.kernels.cpu_reference.inkling2_mini import select_expert_routes, split_expert_activation
from hipengine.runtime.inkling2_mini_reference import Inkling2TextReference


_PROJECTED = {
    "attn.wq_du.weight": "attn_q", "attn.wk_dv.weight": "attn_k", "attn.wv_dv.weight": "attn_v",
    "attn.wo_ud.weight": "attn_output", "attn.wr_du.weight": "attn_rel",
    "attn.attn_sink_proj.weight": "attn_sink_proj", "attn.gate_attn.weight": "attn_gate",
    "attn.k_sconv.weight_proj": "attn_k_conv_proj", "attn.v_sconv.weight_proj": "attn_v_conv_proj",
    "mlp.gate.weight": "ffn_gate_inp",
}
_COEFFICIENTS = {
    "attn_norm.weight": "attn_norm", "q_input_norm.weight": "attn_q_inp_norm",
    "k_input_norm.weight": "attn_k_inp_norm", "v_input_norm.weight": "attn_v_inp_norm",
    "mlp_norm.weight": "ffn_norm", "attn.q_l2_clip.scale": "attn_q_clip", "attn.rel_l2_clip.scale": "attn_rel_clip",
}


def _conv_tensor(x):
    return x[:, None, :]


def _dynamic_static_taps(x):
    return x[:, ::-1].T


def _local_query(x):
    return x.reshape(-1)


class _TextWeights:
    def __init__(self, source):
        self.source = source
        mapping = source.mapping
        self.names = {}
        self.views = {}
        self._coefficients = {}

        def bind(alias, tensor, view=None):
            self.names[alias] = tensor
            if view is not None:
                self.views[alias] = view

        for alias, name in (("embed", "token_embd"), ("embed_norm", "token_embd_norm"),
                            ("unembed", "output"), ("norm", "output_norm")):
            bind(alias + ".weight", mapping.root[name + ".weight"])
        bind("attn_res.output_query", mapping.root["output_res_query.weight"])
        for i, layer in enumerate(mapping.layers):
            prefix = f"layers.{i}."
            t = layer.tensors

            def add(alias, name, view=None):
                bind(prefix + alias, t[name + ".weight"], view)

            for alias, name in {**_PROJECTED, **_COEFFICIENTS}.items():
                if name + ".weight" in t:
                    add(alias, name)
            for stem in ("k", "v"):
                add(f"attn.{stem}_sconv." + ("weight" if layer.is_local else "static_w"), f"attn_{stem}_conv",
                    _conv_tensor if layer.is_local else _dynamic_static_taps)
            add("attn_sconv.weight", "attn_post_conv", _conv_tensor)
            add("mlp_sconv.weight", "ffn_post_conv", _conv_tensor)
            add("attn.rel_logits_proj.proj", "attn_rel_bias", np.transpose)
            if i:
                bind(f"attn_res.attn_queries.{i - 1}", t["attn_res_query.weight"], _local_query if layer.is_local else None)
            bind(f"attn_res.mlp_queries.{i}", t["ffn_res_query.weight"])
            for stem, ctl in zip(("attn", "k", "v", "mlp"), layer.controllers):
                for op in ("down", "forget", "gain"):
                    bind(prefix + f"controllers.{stem}.{op}_weight", getattr(ctl, op))

    def dense(self, name, selector=...):
        tensor = self.names[name]
        if selector is not ...:
            return self.source.decode(tensor, selector)
        result = self._coefficients.get(name)
        if result is not None:
            return result
        data = self.source.decode(tensor)
        if name in self.views:
            data = self.views[name](data)
        # Cache small coefficients, not any expanded projection bank.
        if data.nbytes <= 128 * 1024:
            self._coefficients[name] = data
        return data


class Inkling2GGUFText(Inkling2TextReference):
    def __init__(self, weights, *, projection=None):
        m = self.mapping = weights.mapping
        self.gguf = weights
        super().__init__({"text_config": {
            "loop_config": {"core_start": m.loop_start},
            "vocab_size": m.vocab_size, "unpadded_vocab_size": m.vocab_size,
            "hidden_size": m.hidden_size, "model_max_length": m.max_context,
            "sliding_window_size": m.sliding_window, "rel_extent": m.relative_extent,
            "rms_norm_eps": m.rms_eps, "qk_l2_clip_threshold": m.qk_clip, "rel_l2_clip_threshold": m.relative_clip,
            "sconv_kernel_size": m.conv_taps, "n_layers_per_attn_res_block": m.residual_block,
            "log_scaling_alpha": m.log_scale_alpha, "log_scaling_n_floor": m.log_scale_floor,
            "final_logit_softcapping": m.logit_cap,
        }}, _TextWeights(weights), plan=m.plan)
        self.projection = projection
        if projection is not None:
            self.project_device = self.project_block_device = self._project_device

    def head_shape(self, visit):
        layer = self.mapping.layers[visit.weight_layer]
        return layer.heads, layer.kv_heads, layer.head_dim

    def loop_queries(self, visit):
        return self.gguf.loop_queries(visit)

    def project(self, name, hidden, *, limit=None, transform_output=None):
        if self.projection is None:
            return super().project(name, hidden, limit=limit)
        return self.projection.project(self.weights.names[name].name, hidden, limit=limit,
                                       transform_output=transform_output, pinned=True)

    def _project_device(self, name, width, *, rows=None, transform_output=None):
        return self.projection.project_device(self.weights.names[name].name, width, rows=rows,
                                              transform_output=transform_output)

    def project_many(self, names, hidden):
        return self.project_groups(((names, hidden),))[0]

    def project_groups(self, groups):
        return self.project_chains((None, names, x) for names, x in groups)

    def project_chains(self, chains):
        if self.projection is None:
            return super().project_chains(chains)
        tensors = self.weights.names
        return self.projection.project_chains(((None if first is None else tensors[first].name,
                                                tuple(tensors[name].name for name in names), x)
                                               for first, names, x in chains), pinned=True)

    def route(self, visit, hidden):
        layer = self.mapping.layers[visit.weight_layer]
        bias = layer.tensors.get("exp_probs_b.bias")
        logits = self.project(f"layers.{visit.weight_layer}.mlp.gate.weight", hidden)
        return select_expert_routes(logits, None if bias is None else self.gguf.decode(bias),
                                    top_k=layer.top_k, route_scale=self.mapping.route_scale)

    def _cpu_selected(self, visit, projection, ids, inputs):
        return np.stack([self.gguf.project_expert(visit, projection, e, x)
                         for e, x in zip(ids, inputs, strict=True)])

    @property
    def ffn_device_output(self):
        return self.projection is not None and self.projection.supports_device_mix

    def feed_forward(self, visit, normalized, *, transform_output=None):
        ids, mix = self.route(visit, normalized)
        if self.projection is not None and self.projection.supports_device_activation:
            return self.projection.project_experts_mixed(visit, ids, mix, normalized, transform_output=transform_output)
        k = ids.shape[-1]
        project = (self._cpu_selected if self.projection is None else
                   partial(self.projection.project_selected, single_token=normalized.ndim == 1))
        run = partial(project, visit)
        inputs = np.atleast_2d(normalized)
        if self.projection is None:
            paired = run("gate_up", ids.ravel(), np.repeat(inputs, k, axis=0))
        else:
            paired = run("gate_up", ids.ravel(), inputs, routes_per_input=k)
        gate, up = np.split(paired, 2, axis=-1)
        activated = split_expert_activation(gate, up, self.mapping.gate_cap, self.mapping.up_cap)
        if self.ffn_device_output:
            return self.projection.project_down_mixed(visit, ids, mix, activated, transform_output=transform_output)
        down = run("down", ids.ravel(), activated).reshape(*ids.shape, normalized.shape[-1])
        result = np.zeros_like(normalized)
        for i in range(k):
            result += mix[..., i, None] * down[..., i, :]
        return result
