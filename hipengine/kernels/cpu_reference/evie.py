"""Torch-free NumPy CPU reference for EVIE-4.5B (ColQwen3_5).

Implements the full retrieval-encoder forward in float32 NumPy, mirroring the
transformers `Qwen3_5Model` + `ColQwen3_5` reference semantics:

- Qwen3.5-VL vision tower: Conv3d patch embed (t2, p16), learned position
  embedding table resampled bilinearly (align_corners=True) to the patch grid,
  2D RoPE on (h, w) patch positions, 24 pre-norm blocks with LayerNorm(+bias),
  fused QKV(+bias) bidirectional per-image attention, tanh-GELU MLP, and the
  spatial-merge patch merger (per-patch LayerNorm, erf-GELU MLP).
- Qwen3.5 hybrid text stack: 24 gated-DeltaNet layers + 8 full-attention
  layers (interval 4), interleaved partial mRoPE ([11,11,10] pairs, rotary dim
  64), RMSNorm weights stored as deltas (out = (1 + w) * norm(x)), q/k head
  RMSNorm, sigmoid attention-output gate, SiLU MLP.
- Prefix-MRL head: Linear(2560 -> 2048), slice to the active head dim, then
  per-token L2 normalization and attention-mask zeroing.

Gated DeltaNet recurrence (per v-head, state S: (d_k, d_v), float32):

    q = l2norm_eps1e-6(q) * sqrt(1/d_k)
    k = l2norm_eps1e-6(k)
    S *= exp(g_t)
    delta = (v_t - S @ k_t) * beta_t
    S += outer(k_t, delta)
    o_t = q_t @ S

Inputs come from the processor contract (pixel_values patches, token ids,
image grid); see ``scripts/evie_oracle_torch.py`` for the reference fixture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_EPS_RMS = 1e-6
_L2NORM_EPS = 1e-6


# ---------------------------------------------------------------------------
# basic ops
# ---------------------------------------------------------------------------


def rms_norm(x: np.ndarray, w: np.ndarray | None, eps: float = _EPS_RMS) -> np.ndarray:
    """Qwen3.5 RMSNorm: (1 + w) * x/sqrt(mean(x^2)+eps), computed in float32."""
    xf = x.astype(np.float32)
    var = np.mean(xf * xf, axis=-1, keepdims=True)
    out = xf * (1.0 / np.sqrt(var + eps))
    if w is not None:
        out = out * (1.0 + w.astype(np.float32))
    return out


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    xf = x.astype(np.float32)
    mu = xf.mean(axis=-1, keepdims=True)
    var = xf.var(axis=-1, keepdims=True)
    return (xf - mu) / np.sqrt(var + eps) * w + b


def silu(x: np.ndarray) -> np.ndarray:
    xf = x.astype(np.float32)
    return xf / (1.0 + np.exp(-xf))


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    xf = x.astype(np.float32)
    return 0.5 * xf * (1.0 + np.tanh(0.7978845608028654 * (xf + 0.044715 * xf**3)))


def gelu_erf(x: np.ndarray) -> np.ndarray:
    xf = x.astype(np.float32)
    from numpy import errstate

    with errstate(over="ignore"):
        return 0.5 * xf * (1.0 + np.vectorize(math.erf, otypes=[np.float32])(xf / math.sqrt(2.0)))


def softplus(x: np.ndarray) -> np.ndarray:
    xf = x.astype(np.float32)
    return np.logaddexp(np.float32(0.0), xf)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    xf = x.astype(np.float32)
    m = xf.max(axis=axis, keepdims=True)
    e = np.exp(xf - m)
    return e / e.sum(axis=axis, keepdims=True)


def l2norm(x: np.ndarray, eps: float = _L2NORM_EPS) -> np.ndarray:
    """FLA-style l2norm: x * rsqrt(sum(x^2) + eps)."""
    xf = x.astype(np.float32)
    inv = 1.0 / np.sqrt((xf * xf).sum(axis=-1, keepdims=True) + eps)
    return xf * inv


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvieWeights:
    """Flat fp32 numpy arrays keyed by checkpoint tensor name."""

    tensors: dict[str, np.ndarray]

    @classmethod
    def load(cls, safetensors_path: str) -> "EvieWeights":
        from hipengine.loading.safetensors import (
            TensorInfo,
            load_weight_index,
            read_tensor_storage_bytes,
        )

        index = load_weight_index(safetensors_path)
        out: dict[str, np.ndarray] = {}
        for info in index.tensors.values():
            raw = read_tensor_storage_bytes(info)
            if info.dtype == "BF16":
                u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                arr = u.view(np.float32).reshape(info.shape)
            elif info.dtype == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(info.shape)
            else:
                raise ValueError(f"unsupported dtype {info.dtype} for {info.name}")
            out[info.name] = np.ascontiguousarray(arr)
        return cls(out)

    def __getitem__(self, k: str) -> np.ndarray:
        return self.tensors[k]

    def has(self, k: str) -> bool:
        return k in self.tensors


@dataclass(frozen=True)
class EvieSpec:
    """Geometry constants for EVIE-4.5B."""

    hidden_size: int = 2560
    num_layers: int = 32
    full_attention_interval: int = 4
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    mrope_section: tuple[int, int, int] = (11, 11, 10)
    intermediate_size: int = 9216
    vocab_size: int = 248320
    image_token_id: int = 248056
    vision_hidden: int = 1024
    vision_depth: int = 24
    vision_heads: int = 16
    vision_patch: int = 16
    vision_merge: int = 2
    vision_intermediate: int = 4096
    vision_out_hidden: int = 2560
    pos_embed_grid: int = 48  # num_position_embeddings**0.5
    proj_dim: int = 2048

    def is_full_attention(self, layer: int) -> bool:
        return (layer + 1) % self.full_attention_interval == 0

    @classmethod
    def from_model_spec(cls, spec) -> "EvieSpec":
        """Build a reference spec from hipengine.models.evie.EvieModelSpec."""

        return cls(
            hidden_size=spec.hidden_size,
            num_layers=spec.num_layers,
            full_attention_interval=spec.full_attention_interval,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=spec.num_key_value_heads,
            head_dim=spec.head_dim,
            rope_theta=spec.rope_theta,
            partial_rotary_factor=spec.partial_rotary_factor,
            mrope_section=spec.mrope_section,
            intermediate_size=spec.intermediate_size,
            vocab_size=spec.vocab_size,
            image_token_id=spec.image_token_id,
            vision_hidden=spec.vision_hidden_size,
            vision_depth=spec.vision_depth,
            vision_heads=spec.vision_num_heads,
            vision_patch=spec.vision_patch_size,
            vision_merge=spec.vision_spatial_merge_size,
            vision_intermediate=spec.vision_intermediate_size,
            vision_out_hidden=spec.vision_out_hidden_size,
            proj_dim=spec.proj_dim,
        )


# ---------------------------------------------------------------------------
# mRoPE (text) and 2D RoPE (vision)
# ---------------------------------------------------------------------------


def text_rope_tables(
    spec: EvieSpec, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin tables for interleaved partial mRoPE.

    positions: (3, seq) integer position ids (t, h, w axes).
    Returns (cos, sin) each (seq, rotary_dim=64) for the half-split
    (rotate_half) application: q_rot = q[:64]*cos + rotate_half(q[:64])*sin.
    """
    rotary_dim = int(spec.head_dim * spec.partial_rotary_factor)  # 64
    inv_freq = 1.0 / (
        spec.rope_theta
        ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim)
    )  # (32,)
    freqs = inv_freq[None, :, None] * positions[:, None, :].astype(
        np.float32
    )  # (3, 32, seq)
    # interleave: pair p uses axis T if p%3==0, H if p%3==1, W if p%3==2
    # (mrope_section sums to 32; the reference slicing below matches
    # apply_interleaved_mrope with sections [11, 11, 10]).
    freqs_t = freqs[0].copy()  # (32, seq) T freqs as base
    for axis, offset in ((1, 1), (2, 2)):
        length = spec.mrope_section[axis] * 3
        idx = np.arange(offset, length, 3)
        freqs_t[idx] = freqs[axis, idx]
    emb = np.concatenate([freqs_t, freqs_t], axis=0).T  # (seq, 64)
    return np.cos(emb), np.sin(emb)


def apply_rope_half(
    x: np.ndarray, cos: np.ndarray, sin: np.ndarray
) -> np.ndarray:
    """Apply rotate_half-style RoPE to the leading rotary dims of x.

    x: (..., head_dim); cos/sin: (seq, rotary_dim) broadcast over heads.
    """
    rotary_dim = cos.shape[-1]
    xf = x.astype(np.float32)
    x_rot = xf[..., :rotary_dim]
    x_pass = xf[..., rotary_dim:]
    # cos/sin: (seq, rotary_dim) aligned to x's sequence axis (axis 0 for
    # (seq, heads, dim), axis 1 for (b, s, heads, dim)).
    seq_axis = 0 if x.shape[0] == cos.shape[0] else 1
    shape = [1] * x.ndim
    shape[seq_axis] = cos.shape[0]
    shape[-1] = cos.shape[-1]
    cos_b = cos.reshape(shape)
    sin_b = sin.reshape(shape)
    x1 = x_rot[..., : rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2 :]
    rot = np.concatenate([-x2, x1], axis=-1)
    out = x_rot * cos_b + rot * sin_b
    return np.concatenate([out, x_pass], axis=-1)


def vision_rope_tables(
    head_dim: int, positions: np.ndarray, theta: float = 10000.0
) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin for 2D vision RoPE.

    positions: (seq, 2) integer (h, w) patch positions. The rotary embedding
    splits head_dim//2 pairs between h and w (Qwen3_5VisionRotaryEmbedding
    over head_dim//2), then emb = cat(rot, rot) -> full head_dim.
    """
    half_pairs = head_dim // 4  # 16 pairs per axis for head_dim 64
    inv_freq = 1.0 / (
        theta ** (np.arange(0, half_pairs, dtype=np.float32) / half_pairs)
    )
    freqs_h = positions[:, 0][:, None].astype(np.float32) * inv_freq[None]  # (seq, 16)
    freqs_w = positions[:, 1][:, None].astype(np.float32) * inv_freq[None]
    freqs = np.concatenate([freqs_h, freqs_w], axis=-1)  # (seq, 32) = head_dim//2
    emb = np.concatenate([freqs, freqs], axis=-1)  # (seq, head_dim)
    return np.cos(emb), np.sin(emb)


# ---------------------------------------------------------------------------
# position id computation
# ---------------------------------------------------------------------------


def vision_position_ids_block_major(
    grid_thw: np.ndarray, merge: int
) -> np.ndarray:
    """(h, w) patch positions in spatial-merge-block-major order.

    grid_thw: (n_images, 3). Returns (total_patches, 2) and the permuted
    patch order used by the tower (block-major), matching
    transformers.vision_utils.get_vision_position_ids.
    """
    out = []
    for (t, h, w) in grid_thw:
        hp = np.arange(h)
        wp = np.arange(w)
        hh, ww = np.meshgrid(hp, wp, indexing="ij")  # (h, w)
        # block-major: (h//m, m, w//m, m) transpose (1, 2)
        bh, bw = h // merge, w // merge
        hh = hh.reshape(bh, merge, bw, merge).transpose(0, 2, 1, 3).reshape(-1)
        ww = ww.reshape(bh, merge, bw, merge).transpose(0, 2, 1, 3).reshape(-1)
        out.append(np.stack([hh, ww], axis=-1))
    return np.concatenate(out, axis=0)  # (t*h*w, 2)


def lm_rope_positions(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    image_grid_thw: np.ndarray,
    spec: EvieSpec,
) -> np.ndarray:
    """3-axis LM position ids, replicating Qwen3_5TextModel.get_rope_index.

    input_ids: (seq,) for one sequence (no batch); attention_mask: (seq,).
    Returns (3, n_valid) positions in valid-token order.
    """
    merge = getattr(spec, "vision_merge", None) or spec.vision_spatial_merge_size
    valid = attention_mask.astype(bool)
    ids = input_ids[valid]
    is_img = ids == spec.image_token_id

    # group into contiguous (text | image) runs
    groups: list[tuple[str, int]] = []
    for i, img in enumerate(is_img):
        kind = "image" if img else "text"
        if groups and groups[-1][0] == kind:
            groups[-1] = (kind, groups[-1][1] + 1)
        else:
            groups.append((kind, 1))

    grids = [g for g in image_grid_thw]
    pos_axes = np.zeros((3, 0), dtype=np.int64)
    current = 0
    gi = 0
    for kind, length in groups:
        if kind == "text":
            seg = np.arange(length, dtype=np.int64) + current
            pos_axes = np.concatenate([pos_axes, np.stack([seg] * 3)], axis=1)
            current += length
        else:
            t, h, w = grids[gi]
            gi += 1
            mh, mw = h // merge, w // merge
            # merged-grid raster (t, mh, mw); T axis also offset (after build)
            tt, hh, ww = np.meshgrid(
                np.arange(t), np.arange(mh) + current, np.arange(mw) + current,
                indexing="ij",
            )
            seg = np.stack(
                [tt.reshape(-1), hh.reshape(-1), ww.reshape(-1)], axis=0
            )
            seg[0] += current
            pos_axes = np.concatenate([pos_axes, seg], axis=1)
            current += max(mh, mw)
    return pos_axes  # (3, n_valid)


# ---------------------------------------------------------------------------
# vision tower
# ---------------------------------------------------------------------------


def _bilinear_interp_indices(
    grid_thw: np.ndarray, side: int, merge: int
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate F.interpolate(..., mode='bilinear', align_corners=True).

    Resamples the (side, side) learned pos-embed table to each image's
    (h, w) patch grid; emits patches in spatial-merge-block order when
    merge > 1. Returns (indices (N, 4), weights (N, 4)).
    """
    all_idx: list[np.ndarray] = []
    all_w: list[np.ndarray] = []
    for (_t, h, w) in grid_thw:
        taps_by_axis = []
        wts_by_axis = []
        for dim_out in (h, w):
            dim_in = side
            if dim_out > 1:
                src = np.arange(dim_out, dtype=np.float64) * (dim_in - 1) / (dim_out - 1)
            else:
                src = np.full(dim_out, (dim_in - 1) / 2.0)
            src = np.clip(src, 0, dim_in - 1)
            lo = np.floor(src).astype(np.int64)
            hi = np.minimum(lo + 1, dim_in - 1)
            frac = src - lo
            taps_by_axis.append((lo, hi))
            wts_by_axis.append((1 - frac, frac))
        (h_lo, h_hi), (w_lo, w_hi) = taps_by_axis
        (h_wlo, h_whi), (w_wlo, w_whi) = wts_by_axis
        # 4 bilinear taps per output patch: (h_lo/w_lo, h_hi/w_lo, h_lo/w_hi, h_hi/w_hi)
        h_taps = np.stack([h_lo, h_hi, h_lo, h_hi])  # (4, H)
        w_taps = np.stack([w_lo, w_lo, w_hi, w_hi])  # (4, W)
        h_w = np.stack([h_wlo, h_whi, h_wlo, h_whi])  # (4, H)
        w_w = np.stack([w_wlo, w_wlo, w_whi, w_whi])  # (4, W)
        H, W = h, w
        hh, ww2 = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        hh = hh.reshape(-1)
        ww2 = ww2.reshape(-1)

        def block_major(v: np.ndarray) -> np.ndarray:
            bh, bw = H // merge, W // merge
            return v.reshape(bh, merge, bw, merge).transpose(0, 2, 1, 3).reshape(-1)

        hh = block_major(hh)
        ww2 = block_major(ww2)
        idx = h_taps[:, hh] * side + w_taps[:, ww2]  # (4, N) table indices
        wt_total = h_w[:, hh] * w_w[:, ww2]  # (4, N)
        all_idx.append(idx)
        all_w.append(wt_total)
    indices = np.concatenate(all_idx, axis=1)
    weights = np.concatenate(all_w, axis=1)
    return indices.T, weights.T


def vision_forward(
    w: EvieWeights,
    spec: EvieSpec,
    pixel_values: np.ndarray,
    grid_thw: np.ndarray,
) -> np.ndarray:
    """Run the vision tower; returns merged visual features (N, 2560)."""

    merge = spec.vision_merge
    # ---- patch embed: Conv3d (in=3, out=1024, k=(2,16,16), stride=same)
    n_patches = int(np.prod(grid_thw.sum(axis=0)[:1] + grid_thw[:, 1].sum() * grid_thw[:, 2].sum() * 0)) or len(pixel_values)
    x = pixel_values.astype(np.float32).reshape(-1, 3, 2, spec.vision_patch, spec.vision_patch)
    conv_w = w["visual.patch_embed.proj.weight"].reshape(
        spec.vision_hidden, 3, 2, spec.vision_patch, spec.vision_patch
    )
    conv_b = w["visual.patch_embed.proj.bias"]
    # einsum: out(N, H) = sum over c,tp
    x = np.einsum(
        "ncp,ocp->no",
        x.reshape(len(x), 3, -1),
        conv_w.reshape(spec.vision_hidden, 3, -1),
    ) + conv_b

    # ---- learned position embedding (bilinear table resample)
    table = w["visual.pos_embed.weight"]  # (2304, 1024)
    indices, weights = _bilinear_interp_indices(grid_thw, spec.pos_embed_grid, merge)
    pos = (table[indices] * weights[:, :, None]).sum(axis=1)  # (N, 1024)
    x = x + pos

    # ---- rotary tables (block-major patch positions)
    positions = vision_position_ids_block_major(grid_thw, merge)
    cos, sin = vision_rope_tables(spec.vision_hidden // spec.vision_heads, positions)

    # ---- per-image cu_seqlens over the block-major sequence
    cu = [0]
    for (_t, h, ww_) in grid_thw:
        cu.append(cu[-1] + h * ww_)
    cu = np.asarray(cu)

    n_heads = spec.vision_heads
    head_dim = spec.vision_hidden // n_heads
    scale = head_dim**-0.5

    # ---- blocks
    for i in range(spec.vision_depth):
        p = f"visual.blocks.{i}."
        # attention
        h1 = layer_norm(x, w[p + "norm1.weight"], w[p + "norm1.bias"])
        qkv = (
            h1 @ w[p + "attn.qkv.weight"].astype(np.float32).T + w[p + "attn.qkv.bias"]
        )
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(-1, n_heads, head_dim)
        k = k.reshape(-1, n_heads, head_dim)
        q = apply_rope_half(q, cos, sin)
        k = apply_rope_half(k, cos, sin)
        v = v.reshape(-1, n_heads, head_dim)
        out = np.zeros((len(x), spec.vision_hidden), dtype=np.float32)
        for s, e in zip(cu[:-1], cu[1:]):
            qs = q[s:e].transpose(1, 0, 2)  # (heads, L, d)
            ks = k[s:e].transpose(1, 0, 2)
            vs = v[s:e].transpose(1, 0, 2)
            att = softmax(qs @ ks.transpose(0, 2, 1) * scale)
            out[s:e] = (att @ vs).transpose(1, 0, 2).reshape(e - s, -1)
        out = out @ w[p + "attn.proj.weight"].astype(np.float32).T + w[p + "attn.proj.bias"]
        x = x + out
        # mlp (tanh gelu)
        h2 = layer_norm(x, w[p + "norm2.weight"], w[p + "norm2.bias"])
        fc1 = h2 @ w[p + "mlp.linear_fc1.weight"].astype(np.float32).T + w[p + "mlp.linear_fc1.bias"]
        fc2 = gelu_tanh(fc1) @ w[p + "mlp.linear_fc2.weight"].astype(np.float32).T + w[p + "mlp.linear_fc2.bias"]
        x = x + fc2

    # ---- merger: per-patch LayerNorm(1024) -> concat 2x2 -> fc1 -> erf gelu -> fc2
    m = merge
    n_merged = len(x) // (m * m)
    xg = x.reshape(n_merged, m * m, spec.vision_hidden)
    xn = layer_norm(xg, w["visual.merger.norm.weight"], w["visual.merger.norm.bias"])
    xc = xn.reshape(n_merged, m * m * spec.vision_hidden)
    fc1 = xc @ w["visual.merger.linear_fc1.weight"].astype(np.float32).T + w["visual.merger.linear_fc1.bias"]
    fc2 = gelu_erf(fc1) @ w["visual.merger.linear_fc2.weight"].astype(np.float32).T + w["visual.merger.linear_fc2.bias"]
    return fc2  # (N_merged, 2560)


# ---------------------------------------------------------------------------
# text stack
# ---------------------------------------------------------------------------


def _gdn_layer(
    w: EvieWeights,
    spec: EvieSpec,
    layer: int,
    hidden: np.ndarray,
) -> np.ndarray:
    """One gated-DeltaNet linear-attention layer body (input already normed)."""
    p = f"language_model.layers.{layer}.linear_attn."
    b, s, d = hidden.shape
    hkv, hk, hv, nv = 16, 128, 128, 32  # key/value head config
    qkv = hidden @ w[p + "in_proj_qkv.weight"].astype(np.float32).T  # (b, s, 8192)
    z = (hidden @ w[p + "in_proj_z.weight"].astype(np.float32).T).reshape(
        b, s, nv, hv
    )
    beta_in = hidden @ w[p + "in_proj_b.weight"].astype(np.float32).T  # (b, s, 32)
    a_in = hidden @ w[p + "in_proj_a.weight"].astype(np.float32).T  # (b, s, 32)

    # causal conv1d (k=4, groups=conv_dim) with SiLU
    conv_w = w[p + "conv1d.weight"][:, 0, :]  # (8192, 4)
    qkv_t = qkv.transpose(0, 2, 1)  # (b, 8192, s)
    padded = np.pad(qkv_t, ((0, 0), (0, 0), (3, 0)))
    conv = np.zeros_like(qkv_t)
    for t in range(s):
        window = padded[:, :, t : t + 4]  # (b, 8192, 4)
        conv[:, :, t] = (window * conv_w[None]).sum(axis=-1)
    conv = silu(conv)
    qkv = conv.transpose(0, 2, 1)  # (b, s, 8192)

    key_dim = hkv * hk  # 2048
    value_dim = nv * hv  # 4096
    q, k, v = np.split(qkv, [key_dim, key_dim * 2], axis=-1)
    q = q.reshape(b, s, hkv, hk).repeat(2, axis=2)  # -> 32 heads
    k = k.reshape(b, s, hkv, hk).repeat(2, axis=2)
    v = v.reshape(b, s, nv, hv)

    beta = 1.0 / (1.0 + np.exp(-beta_in.astype(np.float32)))  # sigmoid
    A = np.exp(w[p + "A_log"].astype(np.float32))
    dt_bias = w[p + "dt_bias"].astype(np.float32)
    g = -A[None, None] * softplus(a_in + dt_bias[None, None])  # (b, s, 32)

    # recurrent gated delta rule (float32 state)
    qn = l2norm(q) * (1.0 / math.sqrt(hk))
    kn = l2norm(k)
    out = np.zeros((b, s, nv, hv), dtype=np.float32)
    S = np.zeros((b, nv, hk, hv), dtype=np.float32)
    ge = np.exp(g)  # (b, s, 32)
    for t in range(s):
        S = S * ge[:, t][:, :, None, None]
        qt = qn[:, t]  # (b, 32, 128)
        kt = kn[:, t]  # (b, 32, 128)
        vt = v[:, t]  # (b, 32, 128)
        bt = beta[:, t]  # (b, 32)
        kv_mem = (S * kt[:, :, :, None]).sum(axis=2)  # (b, 32, 128) = S @ k
        delta = (vt - kv_mem) * bt[:, :, None]
        S = S + kt[:, :, :, None] * delta[:, :, None, :]
        out[:, t] = (S * qt[:, :, :, None]).sum(axis=2)

    # RMSNormGated: rms(out) * w, then * silu(z)
    out = out.reshape(b, s, nv, hv)
    var = np.mean(out * out, axis=-1, keepdims=True)
    outn = out / np.sqrt(var + _EPS_RMS) * w[p + "norm.weight"].astype(np.float32)
    outn = (outn * silu(z)).reshape(b, s, -1)
    return outn @ w[p + "out_proj.weight"].astype(np.float32).T


def _full_attention_layer(
    w: EvieWeights,
    spec: EvieSpec,
    layer: int,
    hidden: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    bidirectional: bool = True,
) -> np.ndarray:
    """One full-attention layer body (input already normed)."""
    p = f"language_model.layers.{layer}.self_attn."
    b, s, d = hidden.shape
    nq, nk, hd = spec.num_attention_heads, spec.num_key_value_heads, spec.head_dim
    qp = hidden @ w[p + "q_proj.weight"].astype(np.float32).T  # (b, s, 8192)
    q, gate = np.split(qp.reshape(b, s, nq, hd * 2), 2, axis=-1)
    gate = gate.reshape(b, s, -1)
    k = (hidden @ w[p + "k_proj.weight"].astype(np.float32).T).reshape(b, s, nk, hd)
    v = (hidden @ w[p + "v_proj.weight"].astype(np.float32).T).reshape(b, s, nk, hd)

    q = rms_norm(q, w[p + "q_norm.weight"])
    k = rms_norm(k, w[p + "k_norm.weight"])
    q = apply_rope_half(q, cos, sin)
    k = apply_rope_half(k, cos, sin)

    # GQA: repeat kv 4x -> 16 heads
    rep = nq // nk
    k = np.repeat(k, rep, axis=2)
    v = np.repeat(v, rep, axis=2)
    qh = q.transpose(0, 2, 1, 3)  # (b, 16, s, 256)
    kh = k.transpose(0, 2, 1, 3)
    vh = v.transpose(0, 2, 1, 3)
    att = softmax(qh @ kh.transpose(0, 1, 3, 2) * (hd**-0.5))
    if not bidirectional:
        causal = np.triu(np.ones((s, s), dtype=bool), 1)
        att = np.where(causal[None, None], 0.0, att)
    out = (att @ vh).transpose(0, 2, 1, 3).reshape(b, s, -1)
    out = out * (1.0 / (1.0 + np.exp(-gate.astype(np.float32))))  # sigmoid gate
    return out @ w[p + "o_proj.weight"].astype(np.float32).T


def text_forward(
    w: EvieWeights,
    spec: EvieSpec,
    input_ids: np.ndarray,
    position_ids: np.ndarray,
    visual_features: np.ndarray | None = None,
) -> np.ndarray:
    """Text stack forward; returns final hidden states (b, s, 2560).

    position_ids: (3, s). visual_features replaces image-token embed rows.
    """
    emb = w["language_model.embed_tokens.weight"]  # (V, 2560)
    x = emb[input_ids]  # (b, s, d) via broadcast indexing
    if visual_features is not None:
        for bi in range(x.shape[0]):
            mask = input_ids[bi] == spec.image_token_id
            x[bi, mask] = visual_features

    # rope tables need (3, s) -> cos/sin (s, 64) applied per head
    cos, sin = text_rope_tables(spec, position_ids)

    for layer in range(spec.num_layers):
        lp = f"language_model.layers.{layer}."
        h = rms_norm(x, w[lp + "input_layernorm.weight"])
        if spec.is_full_attention(layer):
            attn_out = _full_attention_layer(w, spec, layer, h, cos, sin)
        else:
            attn_out = _gdn_layer(w, spec, layer, h)
        x = x + attn_out
        h2 = rms_norm(x, w[lp + "post_attention_layernorm.weight"])
        gate = h2 @ w[lp + "mlp.gate_proj.weight"].astype(np.float32).T
        up = h2 @ w[lp + "mlp.up_proj.weight"].astype(np.float32).T
        down = silu(gate) * up @ w[lp + "mlp.down_proj.weight"].astype(np.float32).T
        x = x + down
    return rms_norm(x, w["language_model.norm.weight"])


def project_embeddings(
    w: EvieWeights,
    last_hidden: np.ndarray,
    attention_mask: np.ndarray,
    head_dim: int = 128,
) -> np.ndarray:
    """Prefix-MRL head: project, slice, per-token L2 norm, mask zero."""
    proj = last_hidden @ w["custom_text_proj.weight"].astype(np.float32).T + w[
        "custom_text_proj.bias"
    ].astype(np.float32)
    proj = proj[..., :head_dim]
    norm = np.sqrt((proj * proj).sum(axis=-1, keepdims=True))
    proj = proj / np.maximum(norm, 1e-12)
    return proj * attention_mask[..., None]


def maxsim_scores(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Late-interaction MaxSim: (n_q, n_d) from (n_q, Lq, D), (n_d, Ld, D)."""
    sims = np.einsum("ld,kd->lk", queries.astype(np.float32), docs.astype(np.float32))
    return sims.max(axis=-1).sum(axis=-1)
