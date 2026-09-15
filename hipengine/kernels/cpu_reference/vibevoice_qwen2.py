"""CPU reference for the VibeVoice speech-token backbone (Qwen2).

Matches the HF ``Qwen2Model`` the community fork instantiates for the
backbone (fork @ 952326dd, checkpoint @ c00898d2): RMSNorm, QKV projections
with biases, rotary embedding on q/k, GQA (12 query / 2 KV heads, head dim
128), causal attention, o_proj, residual connections, SiLU-gated MLP, final
norm, and tied-embedding logits.

All arithmetic is FP32. The real weights are BF16 in the checkpoint; the
loader widens to FP32 and validates the full shape contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PREFIX = "model.language_model."
_EMBED_NAME = "model.language_model.embed_tokens.weight"


def bf16_bytes_to_f32(payload: bytes) -> np.ndarray:
    """Widen little-endian BF16 storage to FP32."""
    raw = np.frombuffer(payload, dtype=np.uint8)
    if raw.size % 2:
        raise ValueError("bf16 payload is not a whole number of elements")
    bits = raw[1::2].astype(np.uint32) << 8 | raw[0::2].astype(np.uint32)
    return (bits << 16).view(np.float32)


def _finite(name: str, value: np.ndarray) -> np.ndarray:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


@dataclass(frozen=True)
class Qwen2Geometry:
    hidden_size: int = 1536
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 8960
    vocab_size: int = 151936
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal heads * head_dim")

    @property
    def group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


def expected_weight_shapes(geom: Qwen2Geometry) -> dict[str, tuple[int, ...]]:
    """Per-layer name suffixes plus the model-level tensors."""
    layer: dict[str, tuple[int, ...]] = {
        "input_layernorm.weight": (geom.hidden_size,),
        "self_attn.q_proj.weight": (
            geom.num_attention_heads * geom.head_dim,
            geom.hidden_size,
        ),
        "self_attn.q_proj.bias": (geom.num_attention_heads * geom.head_dim,),
        "self_attn.k_proj.weight": (
            geom.num_key_value_heads * geom.head_dim,
            geom.hidden_size,
        ),
        "self_attn.k_proj.bias": (geom.num_key_value_heads * geom.head_dim,),
        "self_attn.v_proj.weight": (
            geom.num_key_value_heads * geom.head_dim,
            geom.hidden_size,
        ),
        "self_attn.v_proj.bias": (geom.num_key_value_heads * geom.head_dim,),
        "self_attn.o_proj.weight": (
            geom.hidden_size,
            geom.num_attention_heads * geom.head_dim,
        ),
        "post_attention_layernorm.weight": (geom.hidden_size,),
        "mlp.gate_proj.weight": (geom.intermediate_size, geom.hidden_size),
        "mlp.up_proj.weight": (geom.intermediate_size, geom.hidden_size),
        "mlp.down_proj.weight": (geom.hidden_size, geom.intermediate_size),
    }
    shapes: dict[str, tuple[int, ...]] = {}
    for layer_idx in range(geom.num_hidden_layers):
        for suffix, shape in layer.items():
            shapes[f"layers.{layer_idx}.{suffix}"] = shape
    shapes["norm.weight"] = (geom.hidden_size,)
    shapes["embed_tokens.weight"] = (geom.vocab_size, geom.hidden_size)
    return shapes


def load_backbone_weights(
    read_tensor,  # Callable[[str], bytes] over full checkpoint names
    geometry: Qwen2Geometry = Qwen2Geometry(),
) -> dict[str, np.ndarray]:
    """Decode every backbone tensor to FP32 and validate the shape contract."""
    weights: dict[str, np.ndarray] = {}
    for short_name, shape in expected_weight_shapes(geometry).items():
        full = PREFIX + short_name
        array = bf16_bytes_to_f32(read_tensor(full)).reshape(shape)
        if array.shape != shape:
            raise ValueError(f"{full}: shape {array.shape} != contract {shape}")
        weights[short_name] = _finite(short_name, array)
    return weights


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(np.square(x.astype(np.float32)), axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(variance + eps)) * weight


def rope_cos_sin(
    positions: np.ndarray, head_dim: int, theta: float
) -> tuple[np.ndarray, np.ndarray]:
    """HF-style cos/sin tables: [positions, head_dim], both halves equal."""
    half = head_dim // 2
    exponent = np.arange(half, dtype=np.float64) * 2.0 / head_dim
    inv_freq = 1.0 / (theta ** exponent)
    angles = positions[:, None].astype(np.float64) * inv_freq[None, :]
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    return np.concatenate([cos, cos], axis=-1), np.concatenate([sin, sin], axis=-1)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotate-half convention: split the head dim into contiguous halves.

    ``cos``/``sin`` are the full-width HF tables (halves duplicated); the
    half-width factors are read from the first half.
    """
    half = x.shape[-1] // 2
    cos_h = cos[..., :half]
    sin_h = sin[..., :half]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([x1 * cos_h - x2 * sin_h, x2 * cos_h + x1 * sin_h], axis=-1)


def split_heads(x: np.ndarray, heads: int, head_dim: int) -> np.ndarray:
    """[tokens, hidden] -> [heads, tokens, head_dim]."""
    return x.reshape(x.shape[0], heads, head_dim).transpose(1, 0, 2)


def repeat_kv(kv: np.ndarray, group_size: int) -> np.ndarray:
    """[kv_heads, tokens, head_dim] -> [q_heads, tokens, head_dim]."""
    if group_size == 1:
        return kv
    return np.repeat(kv, group_size, axis=0)


def attention_forward(
    hidden: np.ndarray,  # [tokens, hidden]
    weights: dict[str, np.ndarray],
    prefix: str,  # e.g. "layers.0.self_attn."
    cos: np.ndarray,
    sin: np.ndarray,
    kv_cache: dict[str, np.ndarray] | None,
    layer_idx: int,
    scale: float | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """GQA causal attention for one layer.

    ``kv_cache`` maps ``("k", layer_idx)`` / ``("v", layer_idx)`` to
    [kv_heads, past_len, head_dim]; the returned cache includes the new
    tokens. Returns [tokens, hidden].
    """
    # cos/sin tables span the full head dim (halves duplicated).
    head_dim = cos.shape[1]
    tokens = hidden.shape[0]
    q = linear(hidden, weights[prefix + "q_proj.weight"], weights[prefix + "q_proj.bias"])
    k = linear(hidden, weights[prefix + "k_proj.weight"], weights[prefix + "k_proj.bias"])
    v = linear(hidden, weights[prefix + "v_proj.weight"], weights[prefix + "v_proj.bias"])
    q = split_heads(q, weights[prefix + "q_proj.weight"].shape[0] // head_dim, head_dim)
    k = split_heads(k, weights[prefix + "k_proj.weight"].shape[0] // head_dim, head_dim)
    v = split_heads(v, weights[prefix + "v_proj.weight"].shape[0] // head_dim, head_dim)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    past_len = 0
    past_k = kv_cache.get(("k", layer_idx)) if kv_cache else None
    if past_k is not None:
        past_v = kv_cache[("v", layer_idx)]
        past_len = past_k.shape[1]
        k = np.concatenate([past_k, k], axis=1)
        v = np.concatenate([past_v, v], axis=1)
    # Preserve every other layer's entries: the caller threads one cache
    # through the layer loop, so returning only this layer's k/v would drop
    # the rest of the sequence state on the next iteration.
    new_cache = dict(kv_cache) if kv_cache else {}
    new_cache[("k", layer_idx)] = k
    new_cache[("v", layer_idx)] = v

    q_heads = q.shape[0]
    kv_heads = k.shape[0]
    group = q_heads // kv_heads
    kq = repeat_kv(k, group)
    vq = repeat_kv(v, group)
    scores = q @ kq.transpose(0, 2, 1)
    scores = scores / (scale if scale is not None else math.sqrt(head_dim))
    total_len = past_len + tokens
    # Query row i may attend every cached position plus new positions <= i,
    # so the diagonal shifts right by past_len (a plain tril would mask the
    # entire cached prefix for every row after the first).
    causal = np.tril(np.ones((tokens, total_len), dtype=bool), k=total_len - tokens)
    scores = np.where(causal[None, :, :], scores, -np.inf)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    context = (probs @ vq).transpose(1, 0, 2).reshape(tokens, -1)
    out = linear(context, weights[prefix + "o_proj.weight"], None)
    return out, new_cache


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def mlp_forward(
    hidden: np.ndarray, weights: dict[str, np.ndarray], prefix: str
) -> np.ndarray:
    gate = silu(linear(hidden, weights[prefix + "gate_proj.weight"], None))
    up = linear(hidden, weights[prefix + "up_proj.weight"], None)
    return linear(gate * up, weights[prefix + "down_proj.weight"], None)


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    out = x.astype(np.float32) @ weight.T
    if bias is not None:
        out = out + bias
    return out


def forward_hidden_states(
    token_ids: np.ndarray,
    weights: dict[str, np.ndarray],
    geometry: Qwen2Geometry = Qwen2Geometry(),
    kv_cache: dict[str, np.ndarray] | None = None,
    position_offset: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Embedding through final norm. Returns [tokens, hidden] and the cache."""
    tokens = token_ids.shape[0]
    if kv_cache is None:
        kv_cache = {}
    positions = np.arange(position_offset, position_offset + tokens)
    cos, sin = rope_cos_sin(positions, geometry.head_dim, geometry.rope_theta)
    hidden = weights["embed_tokens.weight"][token_ids]
    for layer_idx in range(geometry.num_hidden_layers):
        prefix = f"layers.{layer_idx}."
        normed = rms_norm(hidden, weights[prefix + "input_layernorm.weight"], geometry.rms_norm_eps)
        attn, kv_cache = attention_forward(
            normed, weights, prefix + "self_attn.", cos, sin, kv_cache, layer_idx
        )
        hidden = hidden + attn
        normed = rms_norm(
            hidden, weights[prefix + "post_attention_layernorm.weight"], geometry.rms_norm_eps
        )
        hidden = hidden + mlp_forward(normed, weights, prefix + "mlp.")
    return rms_norm(hidden, weights["norm.weight"], geometry.rms_norm_eps), kv_cache


def forward_logits(
    token_ids: np.ndarray,
    weights: dict[str, np.ndarray],
    geometry: Qwen2Geometry = Qwen2Geometry(),
    kv_cache: dict[str, np.ndarray] | None = None,
    position_offset: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Final-norm output projected through the tied embedding. [tokens, vocab]."""
    hidden, kv_cache = forward_hidden_states(
        token_ids, weights, geometry, kv_cache, position_offset
    )
    return hidden @ weights["embed_tokens.weight"].T, kv_cache
