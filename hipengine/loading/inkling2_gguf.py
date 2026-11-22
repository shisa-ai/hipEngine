"""Physical weights and per-visit state geometry for both Inkling GGUF layouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite
from types import MappingProxyType

import numpy as np

from hipengine.loading.gguf import GGUFModelInfo, GGUFReader, GGUFTensorInfo, MissingGGUFTensorError
from hipengine.models.inkling2_mini import Inkling2Visit, Inkling2VisitPlan
from hipengine.quant.gguf import GGMLQuantizationType as Q, dequantize_gguf_data

_FLOAT = (Q.F32, Q.F16, Q.BF16)
_DENSE = (*_FLOAT, Q.Q8_0)
_EXPERT = (Q.NVFP4, Q.Q8_0, Q.Q5_K)


@dataclass(frozen=True, slots=True)
class Inkling2GGUFExpertBank:
    weight: GGUFTensorInfo
    scale: GGUFTensorInfo | None


@dataclass(frozen=True, slots=True)
class Inkling2GGUFController:
    down: GGUFTensorInfo
    forget: GGUFTensorInfo
    gain: GGUFTensorInfo


@dataclass(frozen=True, slots=True)
class Inkling2GGUFLayer:
    heads: int
    kv_heads: int
    head_dim: int
    is_local: bool
    intermediate_size: int
    top_k: int
    tensors: Mapping[str, GGUFTensorInfo]
    gate_up: Inkling2GGUFExpertBank
    down: Inkling2GGUFExpertBank
    controllers: tuple[Inkling2GGUFController, ...]
    loop_queries: GGUFTensorInfo | None


@dataclass(frozen=True, slots=True)
class Inkling2GGUFMap:
    architecture: str
    plan: Inkling2VisitPlan
    layers: tuple[Inkling2GGUFLayer, ...]
    root: Mapping[str, GGUFTensorInfo]
    loop_start: int
    hidden_size: int
    max_context: int
    vocab_size: int
    sliding_window: int
    relative_extent: int
    conv_taps: int
    residual_block: int
    rms_eps: float
    qk_clip: float
    relative_clip: float
    log_scale_floor: int
    log_scale_alpha: float
    route_scale: float
    gate_cap: float
    up_cap: float
    logit_cap: float


def build_inkling2_gguf_map(info: GGUFModelInfo) -> Inkling2GGUFMap:
    """Validate a header without reading tensor payloads or expanding expert banks."""
    arch = info.architecture
    if arch not in ("inkling2", "inkling2_loop"):
        raise ValueError(f"not an Inkling 2 GGUF architecture: {arch}")
    legacy = arch == "inkling2"
    tensors = {t.name: t for t in info.tensors}
    meta = info.metadata

    def get(key, default=...):
        name = f"{arch}.{key}"
        if name in meta:
            return meta[name]
        if default is ...:
            raise ValueError(f"missing Inkling 2 GGUF metadata: {name}")
        return default

    def positive(key):
        value = get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"invalid Inkling 2 GGUF {key}: {value!r}")
        return value

    def number(key, default=..., *, allow_zero=False):
        value = float(get(key, default))
        if not isfinite(value) or value < 0 or (value == 0 and not allow_zero):
            raise ValueError(f"invalid Inkling 2 GGUF {key}: {value!r}")
        return value

    def tensor(name, shape, types=_FLOAT, *, optional=False, singleton_row=False):
        value = tensors.get(name)
        if value is None:
            if optional:
                return None
            raise MissingGGUFTensorError(f"missing Inkling 2 GGUF tensor: {name}")
        if value.shape != shape and not (singleton_row and value.shape == (1, *shape)):
            raise ValueError(f"Inkling 2 GGUF {name} has shape {value.shape}; expected {shape}")
        if value.ggml_type not in types:
            raise ValueError(f"unsupported Inkling 2 GGUF type for {name}: {value.ggml_type_name}")
        return value

    count = positive("block_count")
    block = positive("attn_res.block_size")
    if get("loop.varying", False):
        raise ValueError("variable-pass Inkling 2 GGUF is not supported")
    passes = get("num_loops", 1) if legacy else positive("loop.count")
    if type(passes) is not int or passes < 1:
        raise ValueError(f"invalid Inkling 2 GGUF loop count: {passes!r}")
    if passes > 1:
        start = positive("loop.core_start" if legacy else "loop.start")
        end = positive("loop.core_end" if legacy else "loop.end")
        rank = positive("loop.gate_rank")
    else:
        start, end, rank = 0, count, 0
    physical = count - (passes - 1) * (end - start) if legacy else count
    if not 0 <= start < end <= physical or (passes > 1 and (start % block or end % block)):
        raise ValueError(f"invalid Inkling 2 GGUF loop core [{start}, {end}) for {physical} physical layers")

    # Legacy metadata follows execution order; mainline stores physical metadata
    # and appends state slots for extra passes. Use the shared Inkling walk in both.
    plan = Inkling2VisitPlan.from_config({"text_config": {
        "num_hidden_layers": physical, "local_layer_ids": (),
        "loop_config": {"core_start": start, "core_end": end, "num_passes": passes},
    }})
    extra_visits = (passes - 1) * (end - start)

    def layer_values(key, default=..., *, boolean=False):
        value = get(key, default)
        if not isinstance(value, (tuple, list)) and not boolean:
            value = (value,) * count
        kind = bool if boolean else int
        if (not isinstance(value, (tuple, list)) or len(value) != count
                or any(type(x) is not kind or (not boolean and x <= 0) for x in value)):
            raise ValueError(f"invalid Inkling 2 GGUF per-layer metadata: {key}")
        if not legacy:
            return value
        result = (*value[:end], *value[end + extra_visits:])
        if any(value[i] != result[v.weight_layer] for i, v in enumerate(plan.visits)):
            raise ValueError(f"Inkling 2 GGUF {key} disagrees across repeated visits")
        return result

    local = layer_values("attention.sliding_window_pattern", boolean=True)
    plan = Inkling2VisitPlan(tuple(replace(v, is_local=local[v.weight_layer], state_slot=i if legacy else v.state_slot)
                                   for i, v in enumerate(plan.visits)))
    heads = layer_values("attention.head_count")
    kv_heads = layer_values("attention.head_count_kv")
    ff = layer_values("expert_feed_forward_length")
    top = layer_values("expert_used_count")
    dim = positive("attention.key_length")
    local_dim = get("attention.key_length_swa", dim)
    if (dim != positive("attention.value_length") or local_dim != get("attention.value_length_swa", dim)
            or type(local_dim) is not int or local_dim <= 0 or get("rope.dimension_count", 0) != 0):
        raise ValueError("unsupported Inkling 2 GGUF key/value or positional encoding geometry")
    hidden = positive("embedding_length")
    vocab = positive("vocab_size_unpadded")
    padded = positive("vocab_size")
    tokens = meta.get("tokenizer.ggml.tokens")
    if vocab > padded or (tokens is not None and len(tokens) != vocab):
        raise ValueError("Inkling 2 GGUF vocabulary does not match tokenizer or padded head")
    n_experts = positive("expert_count")
    window = positive("attention.sliding_window")
    extent = positive("attention.relative_extent")
    rel = positive("attention.relative_length")
    conv = positive("shortconv.l_cache")
    if conv < 2:
        raise ValueError("Inkling 2 GGUF convolution requires at least two taps")

    roots = {name + ".weight": tensor(name + ".weight", shape, types) for name, shape, types in (
        ("token_embd", (padded, hidden), _DENSE), ("output", (padded, hidden), _DENSE),
        ("token_embd_norm", (hidden,), _FLOAT), ("output_norm", (hidden,), _FLOAT),
        ("output_res_query", (hidden,), _FLOAT),
    )}
    if legacy and passes > 1:
        for name in ("loop_res_query.weight", "loop_kv_res_query.weight"):
            roots[name] = tensor(name, (2 * (end - start), hidden))

    layers = []
    for i in range(physical):
        h, kv, intermediate, top_k = heads[i], kv_heads[i], ff[i], top[i]
        if h % kv or top_k > n_experts:
            raise ValueError(f"invalid Inkling 2 GGUF attention or router geometry for layer {i}")
        d = local_dim if local[i] else dim
        qrows, kvrows = h * d, kv * d
        prefix = f"blk.{i}."
        shapes = {
            "attn_q": (qrows, hidden), "attn_k": (kvrows, hidden), "attn_v": (kvrows, hidden),
            "attn_gate": (qrows, hidden), "attn_output": (hidden, qrows), "attn_rel": (h * rel, hidden),
            "attn_sink_proj": (h, hidden),
        }
        projection = set(shapes)
        shapes.update({
            "attn_q_clip": (h, d), "attn_rel_clip": (h, rel),
            "attn_rel_bias": (window if local[i] else extent, rel),
            "attn_k_conv": (kvrows, conv), "attn_v_conv": (kvrows, conv),
            "attn_post_conv": (hidden, conv), "ffn_post_conv": (hidden, conv),
            "ffn_norm": (hidden,), "ffn_res_query": (hidden,), "ffn_gate_inp": (n_experts, hidden),
        })
        if local[i]:
            shapes["attn_norm"] = (hidden,)
        else:
            shapes.update({f"attn_{stem}_inp_norm": (hidden,) for stem in ("q", "k", "v")})
            shapes.update({f"attn_{stem}_conv_proj": (kv * conv, hidden) for stem in ("k", "v")})
            projection.update(("attn_k_conv_proj", "attn_v_conv_proj"))
        if i:
            shapes["attn_res_query"] = (hidden,) if local[i] else (3, hidden)
        weights = {name + ".weight": tensor(prefix + name + ".weight", shape,
                                            _DENSE if name in projection else _FLOAT,
                                            singleton_row=name == "attn_res_query" and local[i])
                   for name, shape in shapes.items()}
        bias = tensor(prefix + "exp_probs_b.bias", (n_experts,), optional=True)
        if bias is not None:
            weights["exp_probs_b.bias"] = bias

        banks = []
        for stem, shape in (("gate_up", (n_experts, 2 * intermediate, hidden)),
                            ("down", (n_experts, hidden, intermediate))):
            name = prefix + f"ffn_{stem}_exps"
            weight = tensor(name + ".weight", shape, _EXPERT)
            scale = tensor(name + ".scale", (n_experts,), (Q.F32,), optional=True)
            banks.append(Inkling2GGUFExpertBank(weight, scale))

        controllers = ()
        queries = None
        if passes > 1 and start <= i < end:
            stems = ("attn", "k", "v", "mlp") if legacy else ("q_ctrl", "k_ctrl", "v_ctrl", "ffn_ctrl")
            controllers = tuple(Inkling2GGUFController(
                tensor(prefix + f"loop_{stem}_down.weight", (rank, 2 * hidden), _DENSE),
                tensor(prefix + f"loop_{stem}_forget.weight", (hidden, rank), _DENSE),
                tensor(prefix + f"loop_{stem}_gain.weight", (hidden, rank), _DENSE),
            ) for stem in stems)
            if not legacy:
                queries = tensor(prefix + "loop_msg_query.weight", (4, hidden))
        layers.append(Inkling2GGUFLayer(h, kv, d, local[i], intermediate, top_k, MappingProxyType(weights),
                                       *banks, controllers, queries))

    return Inkling2GGUFMap(
        arch, plan, tuple(layers), MappingProxyType(roots), start, hidden, positive("context_length"),
        vocab, window, extent, conv, block, number("attention.layer_norm_rms_epsilon"),
        number("attention.qk_clip"), number("attention.relative_clip"), positive("attention.log_scale_floor"),
        number("attention.log_scale_alpha", allow_zero=True), number("expert_weights_scale", 1.0),
        number("activation.situ_beta"), number("activation.situ_linear_beta"),
        number("final_logit_softcapping", 0.0, allow_zero=True),
    )


class Inkling2GGUFWeights:
    """Lazy packed weight access and a row-sliced projection oracle."""

    def __init__(self, reader: GGUFReader):
        self.reader = reader
        self.mapping = build_inkling2_gguf_map(reader.info)
        self._raw: dict[str, np.ndarray] = {}

    def raw(self, tensor: GGUFTensorInfo) -> np.ndarray:
        if tensor.name not in self._raw:
            self._raw[tensor.name] = self.reader.tensor_data(tensor.name)
        return self._raw[tensor.name]

    def decode(self, tensor: GGUFTensorInfo, rows=...) -> np.ndarray:
        return dequantize_gguf_data(self.raw(tensor)[rows], tensor.ggml_type)

    def loop_queries(self, visit: Inkling2Visit) -> np.ndarray:
        if visit.message_pass is None:
            raise ValueError("Inkling 2 loop queries are only read on repeated visits")
        mapping = self.mapping
        queries = mapping.layers[visit.weight_layer].loop_queries
        if queries is not None:
            return self.decode(queries)
        start = 2 * (visit.weight_layer - mapping.loop_start)
        rows = slice(start, start + 2)
        q, ffn = self.decode(mapping.root["loop_res_query.weight"], rows)
        k, v = self.decode(mapping.root["loop_kv_res_query.weight"], rows)
        return np.stack((q, k, v, ffn))

    def project_expert(self, visit: Inkling2Visit, projection: str, expert: int,
                       hidden: np.ndarray, *, rows: slice = slice(None)) -> np.ndarray:
        layer = self.mapping.layers[visit.weight_layer]
        bank = {"gate_up": layer.gate_up, "down": layer.down}[projection]
        if not 0 <= expert < bank.weight.shape[0]:
            raise IndexError(f"Inkling 2 GGUF expert index out of range: {expert}")
        matrix = self.decode(bank.weight, (expert, rows))
        result = hidden @ matrix.T
        # GGUF's external scale belongs to the physical projection, not the
        # visit, and applies after its dot product even on repeated passes.
        if bank.scale is not None:
            result *= self.decode(bank.scale, expert)
        return result
