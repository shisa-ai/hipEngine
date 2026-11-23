"""Header validation and lazy weight access for Inkling DSpark GGUF sidecars."""

from math import isfinite
from types import MappingProxyType

from hipengine.loading.gguf import GGUFModelInfo, GGUFReader, MissingGGUFTensorError
from hipengine.loading.inkling2_gguf import Inkling2GGUFMap
from hipengine.quant.gguf import GGMLQuantizationType as Q, dequantize_gguf_data

_FLOAT = (Q.F32, Q.F16, Q.BF16)
_MATRIX = (*_FLOAT, Q.Q8_0)


def build_inkling2_dspark_gguf_map(info: GGUFModelInfo, target: Inkling2GGUFMap):
    """Resolve recurrent draft names and validate their geometry without reading weights."""
    if info.architecture != "dflash":
        raise ValueError(f"Inkling DSpark requires a dflash sidecar, got {info.architecture!r}")
    meta = info.metadata

    def get(key):
        try:
            return meta[f"dflash.{key}"]
        except KeyError as exc:
            raise ValueError(f"missing Inkling DSpark metadata: dflash.{key}") from exc

    def integer(key):
        value = get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"invalid Inkling DSpark {key}: {value!r}")
        return value

    def positive(key):
        value = get(key)
        if type(value) not in (int, float) or value <= 0 or not isfinite(value):
            raise ValueError(f"invalid Inkling DSpark {key}: {value!r}")
        return float(value)

    n_layers = integer("block_count")
    hidden, width = integer("embedding_length"), integer("attention.key_length")
    vocab, logical_vocab = integer("vocab_size"), integer("vocab_size_unpadded")
    heads, kv_heads = integer("attention.head_count"), integer("attention.head_count_kv")
    intermediate, rank, block = integer("feed_forward_length"), integer("dspark.markov_rank"), integer("block_size")
    cap = positive("final_logit_softcapping")
    if (hidden != target.hidden_size or logical_vocab != target.vocab_size or vocab < logical_vocab
            or cap != target.logit_cap):
        raise ValueError("Inkling DSpark hidden size, vocabulary or logit cap disagrees with the target")
    if (width % 2 or heads % kv_heads or width != integer("attention.value_length")
            or width != integer("rope.dimension_count") or block < 2
            or any(get(k) is not False for k in ("attention.causal", "sample_from_anchor", "has_confidence_head"))):
        raise ValueError("unsupported Inkling DSpark attention, RoPE or proposal geometry")
    taps, local = get("target_layers"), get("attention.sliding_window_pattern")
    if (not isinstance(taps, (list, tuple)) or not taps or any(type(t) is not int or t < 0 for t in taps)
            or tuple(taps) != tuple(sorted(set(taps))) or taps[-1] != len(target.plan.visits)):
        raise ValueError("Inkling DSpark taps do not match target logical boundaries")
    if not isinstance(local, (list, tuple)) or len(local) != n_layers or any(type(v) is not bool for v in local):
        raise ValueError("Inkling DSpark sliding window pattern does not match draft layers")

    config = dict(
        markov_head_type="rnn_hidden", hidden_size=hidden, block_size=block,
        num_hidden_layers=n_layers, num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=width,
        sliding_window_size=integer("attention.sliding_window"), local_layer_ids=tuple(i for i, v in enumerate(local) if v),
        markov_rank=rank, target_layer_ids=tuple(taps), rms_norm_eps=positive("attention.layer_norm_rms_epsilon"),
        rope_theta=positive("rope.freq_base"),
    )
    tensors = {t.name: t for t in info.tensors}
    names = {}

    def bind(runtime, stored, shape, kinds=_FLOAT):
        tensor = tensors.get(stored)
        if tensor is None:
            raise MissingGGUFTensorError(f"missing Inkling DSpark tensor: {stored}")
        if tensor.shape != shape or tensor.ggml_type not in kinds:
            raise ValueError(f"Inkling DSpark {stored} has {tensor.shape} {tensor.ggml_type_name}; expected {shape} with supported storage")
        names[runtime] = tensor

    bind("fuse_weight", "dspark_fuse.weight", (n_layers + 1, len(taps)))
    bind("mask_embed", "mask_embd.weight", (hidden,))
    bind("norm.weight", "output_norm.weight", (hidden,))
    bind("markov_head.dflare_norm.weight", "dspark_norm.weight", (hidden,))
    bind("markov_head.markov_w1.weight", "markov_w1.weight", (vocab, rank))
    bind("markov_head.rnn_proj.weight", "joint_proj.weight", (3 * rank, 2 * (hidden + rank)), _MATRIX)
    bind("markov_head.rnn_proj.bias", "joint_proj.bias", (3 * rank,))
    bind("markov_head.hidden_out.weight", "dspark_hidden_out.weight", (hidden, rank), _MATRIX)
    bind("markov_head.hidden_out.bias", "dspark_hidden_out.bias", (hidden,))
    for layer in range(n_layers):
        p, g = f"layers.{layer}.", f"blk.{layer}."
        for name, stored, shape in (
            ("self_attn.q_proj", "attn_q", (heads * width, hidden)),
            ("self_attn.k_proj", "attn_k", (kv_heads * width, hidden)),
            ("self_attn.v_proj", "attn_v", (kv_heads * width, hidden)),
            ("self_attn.k_proj_ctx", "attn_k_ctx", (kv_heads * width, hidden)),
            ("self_attn.v_proj_ctx", "attn_v_ctx", (kv_heads * width, hidden)),
            ("self_attn.o_proj", "attn_output", (hidden, heads * width)),
            ("mlp.gate_proj", "ffn_gate", (intermediate, hidden)),
            ("mlp.up_proj", "ffn_up", (intermediate, hidden)),
            ("mlp.down_proj", "ffn_down", (hidden, intermediate)),
        ):
            bind(p + name + ".weight", g + stored + ".weight", shape, _MATRIX)
        for name, stored, size in (
            ("input_layernorm", "attn_norm", hidden), ("post_attention_layernorm", "ffn_norm", hidden),
            ("dflare_norm", "dflare_norm", hidden), ("self_attn.q_norm", "attn_q_norm", width),
            ("self_attn.k_norm", "attn_k_norm", width), ("self_attn.k_norm_ctx", "attn_k_ctx_norm", width),
        ):
            bind(p + name + ".weight", g + stored + ".weight", (size,))
    return MappingProxyType(config), MappingProxyType(names)


class Inkling2DSparkGGUFWeights:
    """Map runtime names and decode only the requested slice of a lazy tensor."""

    def __init__(self, reader: GGUFReader, target: Inkling2GGUFMap):
        self.reader = reader
        self.config, self.names = build_inkling2_dspark_gguf_map(reader.info, target)
        self._raw = {}

    def raw(self, name):
        if name not in self._raw:
            self._raw[name] = self.reader.tensor_data(self.names[name].name)
        return self._raw[name]

    def dense(self, name, selector=...):
        return dequantize_gguf_data(self.raw(name)[selector], self.names[name].ggml_type)
