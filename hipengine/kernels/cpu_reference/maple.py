"""Torch-free NumPy CPU reference for Maple (deepgrove/maple-preview) ternary inference.

Math oracle for the HIP kernels and the E2E runner. Implements the exact forward
semantics of the HF (`modeling_maple.py`) and MLX (`mlx_lm/models/maple.py`)
references against the official 2-bit ternary MLX checkpoint layout:

- Ternary projections: uint32 packed 2-bit codes (16 per word, LSB first, code-1),
  one bf16 ``row_alpha`` per output row. Dequant value = alpha * (code - 1).
- Embeddings / lm_head: MLX affine 4-bit, group 64 (8 codes per uint32, LSB first),
  ``w = q * scales + biases``.
- Router stays dense bf16 and computes in fp32 (softmax over all experts, top-k,
  renormalize).
- Activations are bf16 with fp32 accumulation; this oracle emulates the same
  rounding boundaries with explicit bf16 rounding helpers.

The oracle is deliberately plain NumPy (no torch, no MLX) and favors clarity over
speed; only the top-k selected experts are unpacked per token.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from hipengine.kernels.registry import KernelKey, register

ArrayLike = Any

MLP_CLAMP = 7.0
TERNARY_CODES_PER_WORD = 16
AFFINE4_CODES_PER_WORD = 8


# ---------------------------------------------------------------------------
# bf16 helpers
# ---------------------------------------------------------------------------


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """Exact bf16 (as uint16 bit pattern) -> fp32."""

    arr = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return arr.view(np.float32)


def f32_to_bf16_bits(values: ArrayLike) -> np.ndarray:
    """Round fp32 -> bf16 (round to nearest even), returned as uint16 bits."""

    f32 = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    u32 = f32.view(np.uint32)
    lsb = (u32 >> 16) & np.uint32(1)
    rounded = (u32 + np.uint32(0x7FFF) + lsb) >> 16
    return rounded.astype(np.uint16)


def bf16_round(values: ArrayLike) -> np.ndarray:
    """Round fp32 values to bf16 precision, returned as fp32."""

    return bf16_to_f32(f32_to_bf16_bits(values))


# ---------------------------------------------------------------------------
# Packed weight decoding (exact integer paths)
# ---------------------------------------------------------------------------


def unpack_ternary_codes(packed: np.ndarray) -> np.ndarray:
    """Unpack uint32 2-bit ternary codes -> int8 in {-1, 0, 1}, LSB first.

    ``packed`` has shape ``[..., words]``; the result has shape
    ``[..., words * 16]`` with element ``j`` of word ``w`` at bits ``2*j``.
    """

    words = np.asarray(packed, dtype=np.uint32)
    shifts = np.arange(0, 32, 2, dtype=np.uint32)
    codes = (words[..., None] >> shifts[None, :]) & np.uint32(0x3)
    codes = codes.reshape(*words.shape[:-1], words.shape[-1] * TERNARY_CODES_PER_WORD)
    return codes.astype(np.int8) - np.int8(1)


def dequantize_ternary(packed: np.ndarray, row_alpha: ArrayLike) -> np.ndarray:
    """Dequantize ternary projection rows to fp32 ``[..., out, in]``.

    ``row_alpha`` is one scale per output row (bf16 bits or fp32).
    """

    codes = unpack_ternary_codes(packed)
    alpha = np.asarray(row_alpha)
    if alpha.dtype == np.uint16:
        alpha = bf16_to_f32(alpha)
    alpha = alpha.astype(np.float32)
    return codes.astype(np.float32) * alpha[..., None]


def unpack_affine4_codes(packed: np.ndarray) -> np.ndarray:
    """Unpack uint32 4-bit codes -> uint8 in [0, 15], LSB first."""

    words = np.asarray(packed, dtype=np.uint32)
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    codes = (words[..., None] >> shifts[None, :]) & np.uint32(0xF)
    return codes.reshape(*words.shape[:-1], words.shape[-1] * AFFINE4_CODES_PER_WORD).astype(
        np.uint8
    )


def dequantize_affine4(
    packed: np.ndarray,
    scales: ArrayLike,
    biases: ArrayLike,
    *,
    group_size: int = 64,
) -> np.ndarray:
    """Dequantize MLX affine 4-bit rows ``[out, in]`` to fp32.

    ``packed`` is ``[out, in/8]`` uint32; ``scales``/``biases`` are
    ``[out, in/group_size]`` (bf16 bits or fp32). Value = q * s + b per group.
    """

    codes = unpack_affine4_codes(packed).astype(np.float32)
    out_features, total = codes.shape
    if total % group_size:
        raise ValueError(f"input dim {total} not a multiple of group_size {group_size}")
    codes = codes.reshape(out_features, total // group_size, group_size)
    s = np.asarray(scales)
    b = np.asarray(biases)
    if s.dtype == np.uint16:
        s = bf16_to_f32(s)
    if b.dtype == np.uint16:
        b = bf16_to_f32(b)
    return (codes * s[..., None].astype(np.float32) + b[..., None].astype(np.float32)).reshape(
        out_features, total
    )


# ---------------------------------------------------------------------------
# Primitive ops (Maple semantics)
# ---------------------------------------------------------------------------


def rmsnorm(x: ArrayLike, weight: ArrayLike, eps: float = 1e-6) -> np.ndarray:
    """Maple RMSNorm: fp32 internal, fp32 weight multiply, one bf16 rounding."""

    x32 = np.asarray(x, dtype=np.float32)
    w32 = np.asarray(weight, dtype=np.float32)
    variance = np.mean(x32 * x32, axis=-1, keepdims=True)
    return bf16_round(x32 * np.reciprocal(np.sqrt(variance + eps)) * w32)


def partial_rope(
    x: ArrayLike,
    *,
    pos: int,
    rope_theta: float,
    rope_dim: int,
) -> np.ndarray:
    """Rotate-half partial RoPE on the first ``rope_dim`` channels (pairs j, j+rope_dim/2).

    No-op channels past ``rope_dim`` are passed through. Input/output fp32
    ``[..., head_dim]`` with the rounding left to the caller.
    """

    arr = np.asarray(x, dtype=np.float32).copy()
    if rope_dim <= 0:
        return arr
    half = rope_dim // 2
    inv_freq = rope_theta ** (-np.arange(half, dtype=np.float32) / np.float32(half))
    angles = np.float32(pos) * inv_freq
    cos = np.cos(angles)
    sin = np.sin(angles)
    x1 = arr[..., :half].copy()
    x2 = arr[..., half:rope_dim].copy()
    arr[..., :half] = x1 * cos - x2 * sin
    arr[..., half:rope_dim] = x2 * cos + x1 * sin
    return arr


def qk_norm_rope(
    q: ArrayLike,
    k: ArrayLike,
    q_norm_weight: ArrayLike,
    k_norm_weight: ArrayLike,
    *,
    pos: int,
    rope_theta: float,
    rope_dim: int,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-head RMSNorm on q/k followed by partial RoPE (SWA layers; rope_dim=0 for NoPE)."""

    qn = rmsnorm(np.asarray(q, dtype=np.float32), q_norm_weight, eps)
    kn = rmsnorm(np.asarray(k, dtype=np.float32), k_norm_weight, eps)
    if rope_dim > 0:
        qn = bf16_round(partial_rope(qn, pos=pos, rope_theta=rope_theta, rope_dim=rope_dim))
        kn = bf16_round(partial_rope(kn, pos=pos, rope_theta=rope_theta, rope_dim=rope_dim))
    return qn, kn


def attention_decode(
    q: ArrayLike,
    k_cache: ArrayLike,
    v_cache: ArrayLike,
    *,
    scale: float,
    window: int = 0,
) -> np.ndarray:
    """Single-query GQA decode attention with optional sliding window.

    ``q``: ``[num_heads, head_dim]`` fp32; ``k_cache``/``v_cache``:
    ``[seq, num_kv_heads, head_dim]`` fp32 (already RoPE'd where applicable).
    The query attends to the last ``window`` keys when ``window > 0``.
    Returns fp32 ``[num_heads, head_dim]`` (rounding left to the caller).
    """

    q32 = np.asarray(q, dtype=np.float32)
    k32 = np.asarray(k_cache, dtype=np.float32)
    v32 = np.asarray(v_cache, dtype=np.float32)
    num_heads = q32.shape[0]
    num_kv = k32.shape[1]
    if num_heads % num_kv:
        raise ValueError(f"num_heads {num_heads} not a multiple of num_kv_heads {num_kv}")
    group = num_heads // num_kv
    # GQA head->kv mapping is head // group (contiguous groups share a kv head).
    return attention_decode_grouped(
        q32.reshape(num_kv, group, -1), k32, v32, scale=scale, window=window
    ).reshape(num_heads, -1)


def router_topk(
    x: ArrayLike,
    gate_weight: ArrayLike,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """fp32 router: logits = x @ W^T, softmax over all experts, top-k, renormalize."""

    logits = np.asarray(gate_weight, dtype=np.float32) @ np.asarray(x, dtype=np.float32)
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()
    top_idx = np.argpartition(probs, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(-probs[top_idx])]
    top_scores = probs[top_idx]
    top_scores = top_scores / (top_scores.sum() + 1e-20)
    return top_idx.astype(np.int64), top_scores.astype(np.float32)


def clamped_swiglu(gate: ArrayLike, up: ArrayLike) -> np.ndarray:
    """silu(min(gate, 7)) * clip(up, -7, 7) (trained-in clamp, fp32 in/out)."""

    g = np.minimum(np.asarray(gate, dtype=np.float32), np.float32(MLP_CLAMP))
    u = np.clip(np.asarray(up, dtype=np.float32), -MLP_CLAMP, MLP_CLAMP)
    return (g / (1.0 + np.exp(-g))) * u


def ternary_gemv(x: ArrayLike, packed: np.ndarray, row_alpha: ArrayLike) -> np.ndarray:
    """y = W x with W the dequantized ternary rows (fp32 accumulate, fp32 out).

    ``packed``: ``[out, in/16]`` uint32; ``row_alpha``: ``[out]``.
    """

    codes = unpack_ternary_codes(packed).astype(np.float32)
    alpha = np.asarray(row_alpha)
    if alpha.dtype == np.uint16:
        alpha = bf16_to_f32(alpha)
    return (codes @ np.asarray(x, dtype=np.float32)) * alpha.astype(np.float32)


def affine4_gemv_f32(
    x: ArrayLike,
    packed: np.ndarray,
    scales: ArrayLike,
    biases: ArrayLike,
    *,
    group_size: int = 64,
) -> np.ndarray:
    """fp32 logits y = W x for a 4-bit affine matrix (lm_head)."""

    weight = dequantize_affine4(packed, scales, biases, group_size=group_size)
    return weight @ np.asarray(x, dtype=np.float32)


def weighted_residual(
    residual: ArrayLike,
    expert_outputs: ArrayLike,
    routing_weights: ArrayLike,
) -> np.ndarray:
    """Maple selected-expert combine plus residual with both bf16 boundaries."""

    values = np.asarray(expert_outputs, dtype=np.float32)
    scores = np.asarray(routing_weights, dtype=np.float32)
    if values.ndim != 2 or scores.shape != (values.shape[0],):
        raise ValueError("expert_outputs must be [top_k, hidden] with one routing weight per row")
    combined = bf16_round(np.sum(values * scores[:, None], axis=0, dtype=np.float32))
    return bf16_round(np.asarray(residual, dtype=np.float32) + combined)


def register_maple_cpu_reference_kernels(*, replace: bool = True) -> None:
    """Register Maple primitive oracles under explicit four-axis keys."""

    kernels = {
        ("maple_ternary_gemv", "row_alpha"): ternary_gemv,
        ("maple_affine4_gemv", "group64"): affine4_gemv_f32,
        ("maple_qknorm_rope", "partial_rotate_half"): qk_norm_rope,
        ("maple_attention_decode", "gqa_spans"): attention_decode,
        ("maple_router_topk", "softmax_renorm"): router_topk,
        ("maple_clamped_swiglu", "clamp7"): clamped_swiglu,
        ("maple_weighted_residual", "bf16_boundaries"): weighted_residual,
    }
    for (layer, variant), kernel in kernels.items():
        register(
            KernelKey("cpu_reference", layer, "maple_ternary2", variant),
            kernel,
            replace=replace,
        )


# ---------------------------------------------------------------------------
# Whole-model reference forward
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MapleReferenceConfig:
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    num_experts: int = 256
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 0.5
    sliding_window: int = 512
    layer_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.layer_types:
            pattern = ("sliding_attention",) * 3 + ("full_attention",)
            object.__setattr__(
                self,
                "layer_types",
                tuple(pattern[i % 4] for i in range(self.num_hidden_layers)),
            )

    @property
    def rope_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def q_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim


class MapleReferenceForward:
    """Decode-step reference forward over packed Maple weights.

    ``weights`` maps checkpoint tensor names to NumPy arrays (uint32 packed,
    uint16 bf16 bits, or fp32). Attention q/k/v are expected either as separate
    ``...self_attn.{q,k,v}_proj.{weight,row_alpha}`` entries or as a fused
    ``...self_attn.qkv_proj.*``; the step handles both.
    """

    def __init__(self, config: MapleReferenceConfig, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.weights = weights
        self._kv_k: list[list[np.ndarray]] = [[] for _ in range(config.num_hidden_layers)]
        self._kv_v: list[list[np.ndarray]] = [[] for _ in range(config.num_hidden_layers)]

    def reset_cache(self) -> None:
        for layer_k, layer_v in zip(self._kv_k, self._kv_v):
            layer_k.clear()
            layer_v.clear()

    @property
    def position(self) -> int:
        return len(self._kv_k[0])

    def _ternary(self, name: str, x: np.ndarray) -> np.ndarray:
        return bf16_round(
            ternary_gemv(
                x,
                self.weights[f"{name}.weight"],
                self.weights[f"{name}.row_alpha"],
            )
        )

    def _bf16(self, name: str) -> np.ndarray:
        arr = np.asarray(self.weights[name])
        if arr.dtype == np.uint16:
            return bf16_to_f32(arr)
        return arr.astype(np.float32)

    def embed(self, token_id: int) -> np.ndarray:
        prefix = "model.word_embeddings"
        row = {
            key: np.asarray(self.weights[f"{prefix}.{suffix}"])[int(token_id)]
            for key, suffix in (("weight", "weight"), ("scales", "scales"), ("biases", "biases"))
        }
        value = dequantize_affine4(
            row["weight"][None, :], row["scales"][None, :], row["biases"][None, :]
        )[0]
        return bf16_round(value)

    def step(self, token_id: int) -> np.ndarray:
        """Run one decode step for ``token_id`` and return fp32 logits [vocab]."""

        cfg = self.config
        pos = self.position
        h = self.embed(token_id)
        for layer in range(cfg.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            layer_type = cfg.layer_types[layer]
            rope_dim = cfg.rope_dim if layer_type == "sliding_attention" else 0
            window = cfg.sliding_window if layer_type == "sliding_attention" else 0

            hn = rmsnorm(h, self._bf16(f"{prefix}.input_layernorm.weight"), cfg.rms_norm_eps)

            qkv_name = f"{prefix}.self_attn.qkv_proj"
            if f"{qkv_name}.weight" in self.weights:
                qkv = self._ternary(qkv_name, hn)
                q = qkv[: cfg.q_size]
                k = qkv[cfg.q_size : cfg.q_size + cfg.kv_size]
                v = qkv[cfg.q_size + cfg.kv_size :]
            else:
                q = self._ternary(f"{prefix}.self_attn.q_proj", hn)
                k = self._ternary(f"{prefix}.self_attn.k_proj", hn)
                v = self._ternary(f"{prefix}.self_attn.v_proj", hn)

            q = q.reshape(cfg.num_attention_heads, cfg.head_dim)
            k = k.reshape(cfg.num_key_value_heads, cfg.head_dim)
            v = v.reshape(cfg.num_key_value_heads, cfg.head_dim)
            q, k = qk_norm_rope(
                q,
                k,
                self._bf16(f"{prefix}.self_attn.q_norm.weight"),
                self._bf16(f"{prefix}.self_attn.k_norm.weight"),
                pos=pos,
                rope_theta=cfg.rope_theta,
                rope_dim=rope_dim,
                eps=cfg.rms_norm_eps,
            )

            self._kv_k[layer].append(k)
            self._kv_v[layer].append(v)
            k_cache = np.stack(self._kv_k[layer])
            v_cache = np.stack(self._kv_v[layer])

            # GQA: contiguous groups of q heads share one kv head.
            group = cfg.num_attention_heads // cfg.num_key_value_heads
            q_grouped = q.reshape(cfg.num_key_value_heads, group, cfg.head_dim)
            attn = attention_decode_grouped(
                q_grouped, k_cache, v_cache, scale=cfg.head_dim**-0.5, window=window
            )
            attn = bf16_round(attn.reshape(cfg.q_size))

            o_out = bf16_round(self._ternary(f"{prefix}.self_attn.o_proj", attn))
            h = bf16_round(h + o_out)

            hn = rmsnorm(h, self._bf16(f"{prefix}.post_attention_layernorm.weight"), cfg.rms_norm_eps)
            top_idx, top_scores = router_topk(
                hn, self._bf16(f"{prefix}.mlp.gate.weight"), top_k=cfg.num_experts_per_tok
            )
            expert_outputs = []
            for expert_id in top_idx:
                gate = self._expert_ternary(prefix, "gate_proj", int(expert_id), hn)
                up = self._expert_ternary(prefix, "up_proj", int(expert_id), hn)
                act = bf16_round(clamped_swiglu(gate, up))
                expert_outputs.append(
                    self._expert_ternary(prefix, "down_proj", int(expert_id), act)
                )
            h = weighted_residual(h, np.stack(expert_outputs), top_scores)

        h = rmsnorm(h, self._bf16("model.norm.weight"), cfg.rms_norm_eps)
        return affine4_gemv_f32(
            h,
            np.asarray(self.weights["lm_head.weight"]),
            self.weights["lm_head.scales"],
            self.weights["lm_head.biases"],
        )

    def _expert_ternary(self, prefix: str, proj: str, expert_id: int, x: np.ndarray) -> np.ndarray:
        name = f"{prefix}.mlp.switch_mlp.{proj}"
        packed = np.asarray(self.weights[f"{name}.weight"])[expert_id]
        alpha = np.asarray(self.weights[f"{name}.row_alpha"])[expert_id]
        return bf16_round(ternary_gemv(x, packed, alpha))


def attention_decode_grouped(
    q_grouped: ArrayLike,
    k_cache: ArrayLike,
    v_cache: ArrayLike,
    *,
    scale: float,
    window: int = 0,
) -> np.ndarray:
    """GQA decode attention with q laid out as [num_kv, group, head_dim]."""

    q32 = np.asarray(q_grouped, dtype=np.float32)
    k32 = np.asarray(k_cache, dtype=np.float32)
    v32 = np.asarray(v_cache, dtype=np.float32)
    if window > 0:
        k32 = k32[-window:]
        v32 = v32[-window:]
    scores = np.einsum("kgd,tkd->kgt", q32, k32) * np.float32(scale)
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    return np.einsum("kgt,tkd->kgd", probs, v32)


def kl_divergence(p_logits: ArrayLike, q_logits: ArrayLike) -> float:
    """KL(p || q) over softmaxed logits, for the correctness gate."""

    p = np.asarray(p_logits, dtype=np.float64)
    q = np.asarray(q_logits, dtype=np.float64)
    p = np.exp(p - p.max())
    p /= p.sum()
    q = np.exp(q - q.max())
    q /= q.sum()
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


register_maple_cpu_reference_kernels()


def top1_agreement(a_logits: Sequence[ArrayLike], b_logits: Sequence[ArrayLike]) -> float:
    """Fraction of positions whose argmax tokens agree."""

    if len(a_logits) != len(b_logits):
        raise ValueError("logit sequences must have equal length")
    agree = sum(
        int(int(np.argmax(np.asarray(a))) == int(np.argmax(np.asarray(b))))
        for a, b in zip(a_logits, b_logits)
    )
    return agree / max(len(a_logits), 1)
