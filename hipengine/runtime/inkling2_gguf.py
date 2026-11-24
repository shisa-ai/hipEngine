"""Bind GGUF tensor layouts to the shared Inkling text traversal."""

from contextlib import ExitStack, closing
from functools import partial
from types import SimpleNamespace

import numpy as np

from hipengine.kernels.cpu_reference.inkling2_mini import select_expert_routes, split_expert_activation
from hipengine.runtime.inkling2_mini_reference import Inkling2TextReference, _rms


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
            self.project_many_transformed = self._project_many_transformed
            self.project_groups_transformed = self._project_groups_transformed
            self.resident_controllers = (self._resident_controllers if projection.supports_resident_controllers
                                         else None)
            self.resident_attention = self._resident_attention if projection.supports_resident_attention else None
            self.resident_block = self._resident_block if projection.supports_resident_block else None

    def _resident_attention(self, session, visit, inputs, messages, norm_names):
        """Attention half of one block visit on the device; only the branch output reaches the host.

        `inputs` holds the query, key and value controller inputs, one row matrix each.
        """
        return self.projection.project_attention_resident(
            self._attention_plan(session, visit, inputs, messages, norm_names))

    def _attention_plan(self, session, visit, inputs, messages, norm_names, *, sources=None, device=None):
        """Bind one block visit's attention half to matrices and device pointers.

        Without `sources` the plan carries host rows and `project_attention_resident` owns the arena.
        A caller that owns an arena passes device `sources` and a `device` cursor instead, and keeps
        the branch output resident.
        """
        c, w, tensors = self.config, self.weights, self.weights.names
        layer = f"layers.{visit.weight_layer}"
        attn, prefix = layer + ".attn.", layer + ".controllers."
        projection = self.projection
        nh, nkv, dim = self.head_shape(visit)
        conv, slot, pos = session.convolution, visit.state_slot, session.position
        dynamic, rows, ksize = not visit.is_local, len(inputs[0]), c["sconv_kernel_size"]
        rel_width = tensors[attn + "wr_du.weight"].shape[0] // nh

        def coefficient(name, host, layout=""):
            return projection.resident_vector(tensors[attn + name].name + layout, host)

        def kv_taps(name):
            return w.dense(attn + f"{name}_sconv.weight")[:, 0, ::-1]

        def static_taps(name):
            return w.dense(attn + f"{name}_sconv.static_w").T.reshape(nkv * dim, ksize)

        def names(*suffixes):
            return tuple(tensors[attn + suffix].name for suffix in suffixes)

        query = names("wq_du.weight", "wr_du.weight", "attn_sink_proj.weight", "gate_attn.weight")
        query_none = (None,) * len(query)
        # Controllers always yield three distinct rows; without them the inputs may be shared.
        if sources is not None:
            group = (0, 1, 2) if len(sources) == 3 else (0, 0, 0)
        else:
            group = (0, 1, 2) if messages is not None or any(x is not inputs[0] for x in inputs) else (0, 0, 0)
        distinct = messages is not None if sources is None else len(sources) == 3
        convolve = None
        if dynamic:
            key, value = names("wk_dv.weight", "k_sconv.weight_proj"), names("wv_dv.weight", "v_sconv.weight_proj")
            groups = ((group[0], query, query_none), (group[1], key, (None, None)), (group[2], value, (None, None)))

            def convolve(projected):
                conv.apply_dynamic(slot, "k", static_taps("k"), nkv, (projected[4], (rows, nkv * dim)),
                                   (projected[5], (rows, nkv * ksize)))
                conv.apply_dynamic(slot, "v", static_taps("v"), nkv, (projected[6], (rows, nkv * dim)),
                                   (projected[7], (rows, nkv * ksize)))

            value_index = 6
        elif not distinct:
            # Without loop controllers one normalized input feeds every attention projection.
            groups = ((0, query + names("wk_dv.weight", "wv_dv.weight"), query_none + (
                partial(conv.apply_static, slot, "k", kv_taps("k"), position=pos),
                partial(conv.apply_static, slot, "v", kv_taps("v"), position=pos))),)
            value_index = 5
        else:
            groups = ((group[0], query, query_none), (group[1], names("wk_dv.weight"),
                                                       (partial(conv.apply_static, slot, "k", kv_taps("k"), position=pos),)),
                      (group[2], names("wv_dv.weight"),
                       (partial(conv.apply_static, slot, "v", kv_taps("v"), position=pos),)))
            value_index = 5
        if sources is not None:
            # The caller already normalized or blended its rows and keeps them resident.
            controllers, norm_weights = None, ()
        elif messages is None:
            # Without loop controllers each input is normalized on the host before its projection group.
            shared = all(x is inputs[0] for x in inputs)
            sources = tuple(_rms(x, c["rms_norm_eps"]) * w.dense(layer + "." + name + ".weight")
                            for x, name in zip(inputs[:1] if shared else inputs, norm_names))
            controllers, norm_weights = None, ()
        else:
            sources = inputs
            controllers = tuple(((tensors[prefix + name + ".down_weight"].name,
                                  tensors[prefix + name + ".forget_weight"].name,
                                  tensors[prefix + name + ".gain_weight"].name), x, messages[:, i])
                                for i, (name, x) in enumerate(zip(("attn", "k", "v"), inputs, strict=True)))
            norm_weights = tuple(w.dense(layer + "." + name + ".weight") for name in norm_names)
        table = coefficient("rel_logits_proj.proj", w.dense(attn + "rel_logits_proj.proj").T, ":lag")

        def attend(q, k, v, rel, sink, gate, out):
            session.attention.step_block_resident(slot, pos, q, k, v, rel, sink, gate, out, table, rel_width, rows)

        return SimpleNamespace(
            rows=rows, hidden=inputs[0].shape[1], position=pos, eps=c["rms_norm_eps"], sources=sources,
            controllers=controllers, norm_weights=norm_weights, groups=groups, convolve=convolve, attend=attend,
            prepare=SimpleNamespace(
                nh=nh, nkv=nkv, dim=dim, rel_width=rel_width, eps=c["rms_norm_eps"],
                clip=coefficient("q_l2_clip.scale", w.dense(attn + "q_l2_clip.scale")),
                threshold=c["qk_l2_clip_threshold"],
                rel_scale=coefficient("rel_l2_clip.scale", w.dense(attn + "rel_l2_clip.scale")),
                rel_threshold=c["rel_l2_clip_threshold"],
                logscale=None if visit.is_local else (c["log_scaling_alpha"], c["log_scaling_n_floor"]),
                query=0, relative=1, sink=2, gate=3, key=4, value=value_index),
            output=(*names("wo_ud.weight"),
                    partial(conv.apply_static, slot, "attn_sconv", w.dense(layer + ".attn_sconv.weight")[:, 0, ::-1],
                            position=pos)),
            device=device)

    def _resident_controllers(self, controllers, *, norm_weights, eps):
        # Blend and normalize loop controllers on the device; strict and unnamed profiles keep the host expression.
        tensors = self.weights.names
        return tuple(self.projection.project_controllers_normed(
            (((tensors[stem + "down_weight"].name, tensors[stem + "forget_weight"].name,
               tensors[stem + "gain_weight"].name), x, message) for stem, x, message in controllers),
            norm_weights=norm_weights, eps=eps))

    def _resident_block(self, session, h, *, limit=None):
        """Advance one prefill block with the hidden rows and residual stack resident.

        Every read, normalization, residual add, the attention half and the expert chain run inside
        one arena. Only the router logits per visit and the final rows cross the host.
        """
        c, w, tensors = self.config, self.weights, self.weights.names
        projection, eps = self.projection, c["rms_norm_eps"]
        rows, hidden = h.shape
        item = np.dtype(np.float32).itemsize
        row = rows * hidden * item
        stride = c["n_layers_per_attn_res_block"]
        loop_start = c["loop_config"]["core_start"]
        capacity = session._residual_capacity
        visits = self.plan.visits if limit is None else self.plan.visits[:limit]
        row_rms = projection.residual_kernel("row_rms")
        row_norm_mul = projection.residual_kernel("row_norm_mul")
        row_add = projection.residual_kernel("row_add")
        residual_read = projection.residual_kernel("residual_read")
        blank = np.empty((rows, hidden), np.float32)

        def norm_names(visit):
            return (("attn_norm",) * 3 if visit.is_local
                    else ("q_input_norm", "k_input_norm", "v_input_norm"))

        def weight(layer, name):
            return projection.resident_vector(tensors[f"layers.{layer}.{name}.weight"].name,
                                              w.dense(f"layers.{layer}.{name}.weight"))

        def query(layer, stem):
            return np.ascontiguousarray(w.dense(f"attn_res.{stem}.{layer}"), dtype=np.float32)

        # One probe per distinct visit shape sizes the arena the attention half needs.
        attention_bytes, seen = 0, set()
        for visit in visits:
            shape = (visit.is_local, visit.message_pass is not None)
            if shape in seen:
                continue
            seen.add(shape)
            names = norm_names(visit)
            if visit.message_pass is not None:
                probe = self._attention_plan(session, visit, (blank,) * 3, np.empty((rows, 4, hidden), np.float32),
                                             names)
            else:
                probe = self._attention_plan(session, visit, (blank,) * (1 if visit.is_local else 3), None, names,
                                             sources=(0,) * (1 if visit.is_local else 3))
            attention_bytes = max(attention_bytes, sum(projection.attention_resident_extent(probe)))
        regions = (("h", row), ("branch", row), ("m", row), ("normalized", 3 * row),
                   ("read", 4 * row), ("messages", 4 * row), ("query", 4 * hidden * item),
                   ("stack", capacity * row), ("previous", capacity * row), ("snapshot", capacity * row),
                   ("saved_h", row), ("previous_h", row), ("attention", attention_bytes))
        offsets, extent = {}, 0
        for name, nbytes in regions:
            extent = (extent + 255) // 256 * 256
            offsets[name] = extent
            extent += nbytes
        with projection.resident(extent) as base:
            def at(name):
                return base + offsets[name]

            projection.upload(at("h"), h)
            count, working, saved, previous = 0, "stack", None, None
            for visit in visits:
                layer = visit.weight_layer
                if layer == loop_start:
                    if saved is None:
                        saved = count
                        projection.copy_device(at("snapshot"), at(working), count * row)
                        projection.copy_device(at("saved_h"), at("h"), row)
                    else:
                        # The abandoned pass keeps its stack; the entry block is restored for the next one.
                        previous = (working, saved + 1, count - saved - 1)
                        projection.copy_device(at("previous_h"), at("h"), row)
                        projection.copy_device(at("h"), at("saved_h"), row)
                        working = "previous" if working == "stack" else "stack"
                        projection.copy_device(at(working), at("snapshot"), saved * row)
                        count = saved
                stack, names = at(working), norm_names(visit)
                if layer:
                    queries = query(layer - 1, "attn_queries").reshape(-1, hidden)
                    projection.upload(at("query"), queries)
                    row_rms(at("h"), stack + count * row, rows, hidden, eps)
                    residual_read(at("query"), stack, at("read"), rows, count + 1, queries.shape[0], hidden)
                    inputs = [at("read") + i * row for i in range(queries.shape[0])]
                else:
                    inputs = [at("h")]
                if visit.message_pass is not None:
                    region, start, blocks = previous
                    queries = np.ascontiguousarray(self.loop_queries(visit), dtype=np.float32)
                    projection.upload(at("query"), queries)
                    row_rms(at("previous_h"), at(region) + (start + blocks) * row, rows, hidden, eps)
                    residual_read(at("query"), at(region) + start * row, at("messages"), rows, blocks + 1,
                                  queries.shape[0], hidden)
                    cursor = at("attention")
                    for i in range(3):
                        projection.copy_device(cursor + i * row, inputs[0] if len(inputs) == 1 else inputs[i], row)
                        projection.copy_device(cursor + 3 * row + i * row, at("messages") + i * row, row)
                    plan = self._attention_plan(session, visit, (blank,) * 3, np.empty((rows, 4, hidden), np.float32),
                                                names, device=SimpleNamespace(cursor=cursor, result=at("branch")))
                else:
                    sources = []
                    for i, source in enumerate(inputs):
                        row_norm_mul(source, weight(layer, names[i]), at("normalized") + i * row, rows, hidden, eps)
                        sources.append(at("normalized") + i * row)
                    plan = self._attention_plan(session, visit, (blank,) * len(inputs), None, names,
                                                sources=sources,
                                                device=SimpleNamespace(cursor=at("attention"), result=at("branch"),
                                                                       sources=sources))
                projection.project_attention_resident(plan)
                if layer % stride == 0:
                    if not layer:
                        # The entry visit reads nothing, so its stack row is written here instead.
                        row_rms(at("h"), stack, rows, hidden, eps)
                    count += 1
                    a = at("branch")
                else:
                    row_add(at("h"), at("branch"), at("h"), rows, hidden)
                    a = at("h")
                projection.upload(at("query"), query(layer, "mlp_queries").reshape(1, hidden))
                row_rms(a, stack + count * row, rows, hidden, eps)
                residual_read(at("query"), stack, at("read"), rows, count + 1, 1, hidden)
                if visit.message_pass is None:
                    row_norm_mul(at("read"), weight(layer, "mlp_norm"), at("normalized"), rows, hidden, eps)
                    normalized = at("normalized")
                else:
                    chain = tuple(tensors[f"layers.{layer}.controllers.mlp.{op}_weight"].name
                                  for op in ("down", "forget", "gain"))
                    normalized = projection.blend_controllers_resident(
                        at("attention"), (chain,), ((tensors[f"layers.{layer}.mlp_norm.weight"].name,
                                                     w.dense(f"layers.{layer}.mlp_norm.weight")),), eps,
                        [at("read")], [at("messages") + 3 * row], rows=rows, width=hidden)
                logits = projection.project_from_device(tensors[f"layers.{layer}.mlp.gate.weight"].name, rows,
                                                        normalized)
                ids, mix = self._select_routes(visit, logits)
                projection.project_experts_mixed_rows(
                    visit, ((ids, mix, blank,
                             partial(session.convolution.apply_static, visit.state_slot, "mlp_sconv",
                                     w.dense(f"layers.{layer}.mlp_sconv.weight")[:, 0, ::-1],
                                     position=session.position)),),
                    device=(normalized, at("m")))
                row_add(a, at("m"), at("h"), rows, hidden)
            result = np.empty((rows, hidden), np.float32)
            projection.download(at("h"), result)
            blocks = np.empty((count, rows, hidden), np.float32)
            projection.download(stack, blocks)
        return result, [np.ascontiguousarray(block) for block in blocks]

    def decode_rows(self, rows):
        """Advance each visit in lockstep without sharing request state or scalar math."""
        if len(rows) <= 1 or self.projection is None:
            return tuple(session.step(token) for session, token in rows)
        with ExitStack() as owned:
            cursors = [owned.enter_context(closing(session.stage_decode(token))) for session, token in rows]
            pending = [next(cursor) for cursor in cursors]
            for visit in self.plan.visits:
                while True:
                    kind = pending[0][1]
                    inputs = []
                    for actual, actual_kind, item in pending:
                        if actual != visit or actual_kind != kind:
                            raise RuntimeError("Inkling GGUF decode stages are out of step")
                        inputs.append(item)
                    if kind == "controllers":
                        outputs = self._project_decode_controllers(inputs)
                    elif kind == "attention_inputs":
                        outputs = self._project_decode_attention(inputs)
                    elif kind == "attention_output":
                        outputs = self._project_decode_attention_output(inputs)
                    elif kind == "experts":
                        outputs = self._feed_forward_decode_rows(visit, inputs)
                    else:
                        raise RuntimeError(f"Inkling GGUF unknown decode stage: {kind}")
                    pending = [cursor.send(out) for cursor, out in zip(cursors, outputs, strict=True)]
                    if kind == "experts":
                        break
            hidden = np.stack(pending)
        return tuple(self.cap_logits(row) for row in self.projection.project_decode_head(hidden))

    def _project_decode_controllers(self, inputs):
        # Keep the scalar vectors and their down/forget/gain dependencies.
        outputs = iter(self.project_chains(chain for chains in inputs for chain in chains))
        return [tuple(next(outputs) for _ in chains) for chains in inputs]

    def _project_decode_attention(self, inputs):
        tensors = self.weights.names
        return self.projection.project_shared_attention_rows([
            (tuple((tuple(tensors[name].name for name in names), x) for names, x in groups), transforms)
            for groups, transforms in inputs])

    def _project_decode_attention_output(self, inputs):
        name, _, width, _ = inputs[0]
        if any(other != name or n != width for other, _, n, _ in inputs):
            raise ValueError("Inkling GGUF attention output projection geometry differs across rows")
        return self.projection.project_resident_attention_rows(self.weights.names[name].name, width,
                                                               [(ptr, consume) for _, ptr, _, consume in inputs])

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

    def _project_many_transformed(self, names, hidden, transforms):
        tensors = self.weights.names
        return self.projection.project_many(tuple(tensors[name].name for name in names), hidden,
                                            pinned=True, transform_outputs=transforms)

    def _project_groups_transformed(self, groups, transforms):
        tensors = self.weights.names
        return self.projection.project_groups(((tuple(tensors[name].name for name in names), x) for names, x in groups),
                                              pinned=True, transform_groups=transforms)

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
        logits = self.project(f"layers.{visit.weight_layer}.mlp.gate.weight", hidden)
        return self._select_routes(visit, logits)

    def _select_routes(self, visit, logits):
        layer = self.mapping.layers[visit.weight_layer]
        bias = layer.tensors.get("exp_probs_b.bias")
        return select_expert_routes(logits, None if bias is None else self.gguf.decode(bias),
                                    top_k=layer.top_k, route_scale=self.mapping.route_scale)

    def _feed_forward_decode_rows(self, visit, inputs):
        # One exchange contains separate vector inputs, so the projection
        # owner still launches the per-row GEMV, not FP32 GEMM.
        name = f"layers.{visit.weight_layer}.mlp.gate.weight"
        scores = self.project_groups((((name,), hidden) for hidden, _ in inputs))
        jobs = [(*self._select_routes(visit, logits), hidden, transform)
                for (hidden, transform), (logits,) in zip(inputs, scores, strict=True)]
        if self.projection.supports_device_activation:
            return self.projection.project_experts_mixed_rows(visit, jobs)
        return [self._feed_forward_routes(visit, hidden, ids, mix, transform_output=transform)
                for ids, mix, hidden, transform in jobs]

    def _cpu_selected(self, visit, projection, ids, inputs):
        return np.stack([self.gguf.project_expert(visit, projection, e, x)
                         for e, x in zip(ids, inputs, strict=True)])

    @property
    def ffn_device_output(self):
        return self.projection is not None and self.projection.supports_device_mix

    def feed_forward(self, visit, normalized, *, transform_output=None):
        ids, mix = self.route(visit, normalized)
        return self._feed_forward_routes(visit, normalized, ids, mix, transform_output=transform_output)

    def _feed_forward_routes(self, visit, normalized, ids, mix, *, transform_output):
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
