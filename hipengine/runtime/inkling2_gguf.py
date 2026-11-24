"""Bind GGUF tensor layouts to the shared Inkling text traversal."""

from contextlib import ExitStack, closing
from functools import partial
from types import SimpleNamespace

import numpy as np

from hipengine.core.hip import HipMemcpyKind, set_capture_stream
from hipengine.core.memory import free, malloc
from hipengine.kernels.cpu_reference.inkling2_mini import select_expert_routes, split_expert_activation
from hipengine.kernels.hip_gfx1100.norm.inkling2_residual import launch_resident_tick
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
        self._graph_state = None
        self._graph_disabled = False
        self._queries_staged = False
        self.decode_graph = False

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
        if limit is None and h.shape[0] == 1 and getattr(self, "decode_graph", False):
            handled = self._graph_step(session, h)
            if handled is not None:
                return handled
        return self._resident_body(session, h, limit=limit, recording=None)

    # ---- whole-token graph capture (opt-in; see HIPENGINE_INKLING2_DECODE_GRAPH) ----

    def _graph_floor(self):
        return int(self.config.get("log_scaling_n_floor", 8192))

    def _graph_min_position(self, session):
        attention = getattr(session, "attention", None)
        geometry = getattr(attention, "geometry", None)
        if not geometry:
            return None
        # The captured graph bakes min(extent, live) per visit; it stays valid while every
        # visit's window is saturated, i.e. once the position passes every visit's window.
        return max(min(g[4], g[0]) for g in geometry)

    def _drop_graph_state(self):
        state = getattr(self, "_graph_state", None)
        self._graph_state = None
        if state:
            session = state.get("session")
            attention = getattr(session, "attention", None) if session is not None else None
            if attention is not None and getattr(attention, "capture_headers", None) is not None:
                attention.capture_headers = None
        else:
            return
        rt = state.get("runtime")
        if rt is not None:
            if state.get("graph"):
                try:
                    rt.graph_exec_destroy(state["graph"])
                except Exception:
                    pass
            stream = state.get("stream")
            if stream:
                try:
                    rt.stream_destroy(stream)
                except Exception:
                    pass
            for key in ("counter", "headers", "caps"):
                buf = state.get(key)
                if buf is not None:
                    try:
                        free(buf, runtime=rt)
                    except Exception:
                        pass
            pin = state.get("pin")
            if pin is not None:
                try:
                    rt.host_unregister(state["pin_ptr"])
                except Exception:
                    pass
                pin.close()

    def _graph_step(self, session, h):
        state = getattr(self, "_graph_state", None)
        if state is not None and state.get("session") is not session:
            # Decode sessions own their KV banks; a graph baked for another session is unusable.
            self._drop_graph_state()
            state = None
        if state is not None:
            if session.position + 1 <= self._graph_floor():
                return self._replay_graph(session, h, state)
            # Past the log-scale floor the clip factors differ; retire the graph.
            self._drop_graph_state()
            self._graph_disabled = True
            return None
        if getattr(self, "_graph_disabled", False):
            return None
        floor = self._graph_floor()
        minimum = self._graph_min_position(session)
        if minimum is None or session.position + 1 < minimum or session.position + 1 > floor:
            return None
        return self._capture_graph(session, h)

    def _capture_graph(self, session, h):
        """Record the whole decode step; the current step itself runs eagerly.

        The recording pass executes nothing, so every lazily uploaded weight must
        already be resident. The eager pass below does that (and produces this
        step's result); the recorded graph serves subsequent steps.
        """
        eager = self._resident_body(session, h, limit=None, recording=None)
        state = None
        try:
            state = self._record_graph(session, h)
        except Exception as exc:
            print(f"inkling2: decode graph capture failed ({exc!r}); continuing eager", flush=True)
            import traceback
            traceback.print_exc()
            self._graph_disabled = True
            self._discard_capture(state, session)
            return eager
        self._graph_state = state
        return eager

    def _discard_capture(self, state, session):
        attention = getattr(session, "attention", None)
        if attention is not None:
            attention.capture_headers = None
        if not state:
            return
        rt = state.get("runtime")
        for key in ("stream", "counter", "headers", "caps"):
            resource = state.pop(key, None)
            if resource is None or rt is None:
                continue
            try:
                if key == "stream":
                    rt.stream_destroy(resource)
                else:
                    free(resource, runtime=rt)
            except Exception:
                pass
        pin = state.pop("pin", None)
        pin_ptr = state.pop("pin_ptr", None)
        if pin is not None:
            if pin_ptr is not None and rt is not None:
                try:
                    rt.host_unregister(pin_ptr)
                except Exception:
                    pass
            try:
                pin.close()
            except Exception:
                pass

    def _record_graph(self, session, h):
        import ctypes as _ctypes
        import mmap

        rt = self.projection._runtime
        attention = getattr(session, "attention", None)
        if attention is None:
            raise RuntimeError("decode graph requires session attention state")
        state = {"session": session, "runtime": rt}
        visits = self.plan.visits
        slots = 1 + max(v.state_slot for v in visits)
        geometry = attention.geometry
        counter = malloc(4, runtime=rt)
        headers = malloc(2 * 4 * slots, runtime=rt)
        caps = malloc(4 * slots, runtime=rt)
        state.update(counter=counter, headers=headers, caps=caps)
        caps_host = np.zeros(slots, np.int32)
        for slot in range(slots):
            caps_host[slot] = geometry[slot][0]
        rt.memcpy(caps.ptr, caps_host.ctypes.data, caps_host.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        # This step already ran eagerly; the first replay serves the next position.
        start = np.array([session.position + 1], dtype=np.int32)
        rt.memcpy(counter.ptr, start.ctypes.data, start.nbytes, HipMemcpyKind.HOST_TO_DEVICE)
        page = mmap.PAGESIZE
        pin_nbytes = (h.nbytes + page - 1) // page * page
        pin = mmap.mmap(-1, pin_nbytes)
        pin_ptr = _ctypes.addressof(_ctypes.c_char.from_buffer(pin))
        _ctypes.memmove(pin_ptr, h.ctypes.data, h.nbytes)
        try:
            rt.host_register(pin_ptr, pin_nbytes)
        except BaseException:
            pin.close()
            raise
        state["pin"] = pin
        state["pin_ptr"] = pin_ptr
        state["stream"] = rt.stream_create(nonblocking=True)
        attention.capture_headers = [headers.ptr + slot * 8 for slot in range(slots)]
        try:
            self._resident_body(session, h, limit=None, recording=state)
        except BaseException:
            set_capture_stream(None)
            try:
                rt.stream_end_capture(state["stream"])
            except Exception:
                pass
            attention.capture_headers = None
            self._discard_capture(state, session)
            raise
        set_capture_stream(None)
        graph = rt.stream_end_capture(state["stream"])
        if not graph:
            attention.capture_headers = None
            self._discard_capture(state, session)
            raise RuntimeError("decode graph capture returned no graph")
        state["graph"] = rt.graph_instantiate(graph)
        rt.graph_destroy(graph)
        return state

    def _replay_graph(self, session, h, state):
        import ctypes as _ctypes

        rt = state["runtime"]
        _ctypes.memmove(state["pin_ptr"], h.ctypes.data, h.nbytes)
        rt.graph_launch(state["graph"], state["stream"])
        rt.stream_synchronize(state["stream"])
        attention = session.attention
        for visit in list(attention.positions):
            attention.positions[visit] = session.position + 1
        at = lambda name: state["base"] + state["offsets"][name]
        hidden = h.shape[1]
        result = np.empty((1, hidden), np.float32)
        self.projection.download(at("h"), result)
        count = state["count"]
        blocks = np.empty((count, 1, hidden), np.float32)
        self.projection.download(at("stack"), blocks)
        return result, [np.ascontiguousarray(block) for block in blocks]

    def _stage_query_uploads(self, session, base, offsets, hidden):
        """Upload the per-visit residual-read queries once per arena; both decode paths read the slots."""
        if self._queries_staged:
            return
        item = np.dtype(np.float32).itemsize
        span = 16 * hidden * item  # rows 0-3 attention read queries, 4-7 loop-message queries, row 8 mlp read query

        def query(layer, stem):
            return np.ascontiguousarray(self.weights.dense(f"attn_res.{stem}.{layer}"), dtype=np.float32)

        rows_by_slot = {}

        def put(slot, part, row, array):
            array = np.ascontiguousarray(array, dtype=np.float32).reshape(-1, hidden)
            if array.shape[0] > 4:
                raise RuntimeError(f"residual-read query block of {array.shape[0]} rows exceeds its slot")
            self.projection.upload(base + offsets["queries"] + slot * span + row * hidden * item, array)
            rows_by_slot[(slot, part)] = array.shape[0]

        for visit in self.plan.visits:
            slot = visit.state_slot
            layer = visit.weight_layer
            if layer:
                put(slot, "attn", 0, query(layer - 1, "attn_queries"))
            if visit.message_pass is not None:
                put(slot, "msg", 4, self.loop_queries(visit))
            put(slot, "mlp", 8, query(layer, "mlp_queries"))
        self._query_rows = rows_by_slot
        self._queries_staged = True

    def _resident_body(self, session, h, *, limit, recording):
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
        query_slots = 1 + max(v.state_slot for v in self.plan.visits)
        query_stride = 16 * hidden * item
        regions = (("h", row), ("branch", row), ("m", row), ("normalized", 3 * row),
                   ("read", 4 * row), ("messages", 4 * row),
                   ("queries", query_slots * query_stride),
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

            def qslot(visit, row):
                return at("queries") + visit.state_slot * query_stride + row * hidden * item

            if recording is not None:
                rt = recording["runtime"]
                stream = recording["stream"]
                self._stage_query_uploads(session, base, offsets, hidden)
                rt.stream_synchronize(0)  # staged slots must be visible before capture begins
                rt.stream_begin_capture(stream, 2)  # relaxed capture
                set_capture_stream(stream)
                rt.memcpy_async(at("h"), recording["pin_ptr"], h.nbytes, HipMemcpyKind.HOST_TO_DEVICE, stream)
                launch_resident_tick(recording["headers"].ptr, recording["caps"].ptr, recording["counter"].ptr,
                                     1 + max(v.state_slot for v in visits), stream=stream)
            else:
                self._stage_query_uploads(session, base, offsets, hidden)
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
                    qrows = self._query_rows[(visit.state_slot, "attn")]
                    row_rms(at("h"), stack + count * row, rows, hidden, eps)
                    residual_read(qslot(visit, 0), stack, at("read"), rows, count + 1, qrows, hidden)
                    inputs = [at("read") + i * row for i in range(qrows)]
                else:
                    inputs = [at("h")]
                if visit.message_pass is not None:
                    region, start, blocks = previous
                    qrows = self._query_rows[(visit.state_slot, "msg")]
                    row_rms(at("previous_h"), at(region) + (start + blocks) * row, rows, hidden, eps)
                    residual_read(qslot(visit, 4), at(region) + start * row, at("messages"), rows, blocks + 1,
                                  qrows, hidden)
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
                row_rms(a, stack + count * row, rows, hidden, eps)
                residual_read(qslot(visit, 8), stack, at("read"), rows, count + 1, 1, hidden)
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
                transform = partial(session.convolution.apply_static, visit.state_slot, "mlp_sconv",
                                    w.dense(f"layers.{layer}.mlp_sconv.weight")[:, 0, ::-1],
                                    position=session.position)
                if (rows == 1 and getattr(projection, "device_routes_supported", lambda v: False)(visit)):
                    logits_ptr, _ = projection.project_router_logits_ptr(
                        tensors[f"layers.{layer}.mlp.gate.weight"].name, rows, normalized)
                    route = projection.device_routes(visit, logits_ptr, rows)
                    projection.project_experts_device_routes(
                        visit, hidden_ptr=normalized, out_ptr=at("m"), route=route,
                        transform_output=transform)
                else:
                    # No device-route kernel for this bank: stage the normalized rows on the host,
                    # take the ordinary routed path, and upload the branch output back into the arena.
                    normed = np.empty((rows, hidden), np.float32)
                    projection.download(normalized, normed)
                    routed = self.feed_forward(visit, normed[0] if rows == 1 else normed,
                                               transform_output=transform)
                    projection.upload(at("m"), np.ascontiguousarray(routed, dtype=np.float32))
                row_add(a, at("m"), at("h"), rows, hidden)
            if recording is not None:
                recording["count"] = count
                recording["base"] = base
                recording["offsets"] = dict(offsets)
                return None
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
        projection = self.projection
        if (projection is not None and normalized.ndim == 1
                and getattr(projection, "device_routes_supported", lambda v: False)(visit)):
            return self._feed_forward_device_routes(visit, normalized, transform_output=transform_output)
        ids, mix = self.route(visit, normalized)
        return self._feed_forward_routes(visit, normalized, ids, mix, transform_output=transform_output)

    def _feed_forward_device_routes(self, visit, normalized, *, transform_output=None):
        projection = self.projection
        hidden = np.ascontiguousarray(normalized, dtype=np.float32)
        name = self.weights.names[f"layers.{visit.weight_layer}.mlp.gate.weight"].name
        inp = projection.upload_scratch("_device_route_hidden", hidden)
        logits_ptr, _ = projection.project_router_logits_ptr(name, 1, inp)
        route = projection.device_routes(visit, logits_ptr, 1)
        out_ptr = projection.project_experts_device_routes(
            visit, hidden_ptr=inp, route=route, transform_output=transform_output)
        result = np.empty(normalized.shape, dtype=np.float32)
        projection.download(out_ptr, result)
        return result

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
