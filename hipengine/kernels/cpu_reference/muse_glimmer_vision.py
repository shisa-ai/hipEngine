"""NumPy reference for the Muse Glimmer vision tower (mmproj).

Contract source: ``clip_graph_muse_glimmer`` (llama.cpp mtmd, build e0f142d5d)
plus the ``PROJECTOR_TYPE_MUSE_GLIMMER`` host branch in ``clip.cpp``:

* conv2d patchify (14x14, stride 14, no bias), raster patch order;
* learned 32x32 position table bilinear-resized (half-pixel, align_corners
  False, clamped) to the dynamic patch grid and added;
* 32x32-window grouping permutation; 50 pre-LN ViT layers (LayerNorm eps 1e-5
  with bias, separate Q/K/V + out projections with bias, exact-erf GELU FFN);
* 2D interleaved RoPE on Q/K: dims [0,48) take 1-indexed width positions,
  dims [48,96) take height positions, 24 frequency pairs per half, base 10000;
* sparse layers attend block-diagonally within windows; every 4th layer and
  the last are global (bidirectional, no causal mask);
* inverse permutation, 2x2 pixel-shuffle with channel-outer concatenation;
* 3-layer adapter MLP (6144->4096->4096->6656, exact-erf GELU between).

Weights are consumed as dequantized float32 arrays from the mmproj GGUF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from hipengine.loading.gguf import GGUFModelInfo

WeightSource = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class MuseGlimmerVisionConfig:
    n_layers: int
    n_embd: int
    n_ff: int
    n_heads: int
    head_dim: int
    patch_size: int
    image_size: int
    spatial_merge_size: int
    projection_dim: int
    norm_eps: float
    rope_theta: float
    sparse_factor: int
    pos_grid: int
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]
    max_image_tokens: int

    @classmethod
    def from_gguf_info(cls, info: GGUFModelInfo) -> "MuseGlimmerVisionConfig":
        md = info.metadata
        if md.get("clip.projector_type") != "muse-glimmer":
            raise ValueError(
                f"expected muse-glimmer projector, got {md.get('clip.projector_type')!r}"
            )
        p = "clip.vision."
        n_embd = int(md[p + "embedding_length"])
        n_heads = int(md[p + "attention.head_count"])
        pos_rows = int(info.tensor("v.position_embd.weight").shape[0])
        pos_grid = int(round(pos_rows ** 0.5))
        if pos_grid * pos_grid != pos_rows:
            raise ValueError("position table is not a square grid")
        return cls(
            n_layers=int(md[p + "block_count"]),
            n_embd=n_embd,
            n_ff=int(md[p + "feed_forward_length"]),
            n_heads=n_heads,
            head_dim=n_embd // n_heads,
            patch_size=int(md[p + "patch_size"]),
            image_size=int(md[p + "image_size"]),
            spatial_merge_size=int(md[p + "spatial_merge_size"]),
            projection_dim=int(md[p + "projection_dim"]),
            norm_eps=float(md[p + "attention.layer_norm_epsilon"]),
            rope_theta=10000.0,
            sparse_factor=4,
            pos_grid=pos_grid,
            image_mean=tuple(float(v) for v in md[p + "image_mean"]),
            image_std=tuple(float(v) for v in md[p + "image_std"]),
            max_image_tokens=4096,
        )


def grid_size(
    img_w: int, img_h: int, patch_hw: int, max_tokens: int
) -> tuple[int, int]:
    """Aspect-ratio-preserving target size (post-merge patch units -> pixels).

    Replicates ``muse_glimmer_grid_size`` (transformers'
    get_aspect_ratio_preserving_size): scale down only when over the token cap,
    then pick the floor/ceil candidate pair closest to the source aspect ratio,
    breaking ties toward more tokens.
    """

    i_nph = img_h / patch_hw
    i_npw = img_w / patch_hw
    ratio = i_npw / i_nph if i_nph > 0 else 1.0
    if i_nph * i_npw > max_tokens:
        i_nph = (max_tokens / ratio) ** 0.5
        i_npw = i_nph * ratio
    hs = (int(np.floor(i_nph)), int(np.ceil(i_nph)))
    ws = (int(np.floor(i_npw)), int(np.ceil(i_npw)))
    target_ar = img_h / img_w
    best = None
    best_d = 0.0
    for nph in hs:
        for npw in ws:
            if nph < 1 or npw < 1 or nph * npw > max_tokens:
                continue
            d = abs(nph / npw - target_ar)
            if best is None or d < best_d or (d == best_d and nph * npw > best[0] * best[1]):
                best = (nph, npw)
                best_d = d
    if best is None:
        best = (max(1, round(i_nph)), max(1, round(i_npw)))
    nph, npw = best
    return npw * patch_hw, nph * patch_hw


def normalize_pixels(pixels: np.ndarray, config: MuseGlimmerVisionConfig) -> np.ndarray:
    """uint8 HWC -> float32 CHW normalized by the artifact mean/std."""

    x = np.asarray(pixels, dtype=np.float32) / 255.0
    mean = np.asarray(config.image_mean, dtype=np.float32)[:, None, None]
    std = np.asarray(config.image_std, dtype=np.float32)[:, None, None]
    x = (x.transpose(2, 0, 1) - mean) / std
    return np.ascontiguousarray(x, dtype=np.float32)


def patch_columns(pixels_chw: np.ndarray, patch: int) -> np.ndarray:
    """im2col only: [n_tok, C*patch*patch] raster-order patch columns."""

    c, h, w = pixels_chw.shape
    gh, gw = h // patch, w // patch
    cols = np.empty((gh * gw, c * patch * patch), dtype=np.float32)
    for py in range(gh):
        for px in range(gw):
            block = pixels_chw[:, py * patch:(py + 1) * patch, px * patch:(px + 1) * patch]
            cols[py * gw + px] = block.reshape(-1)
    return cols


def patchify(pixels_chw: np.ndarray, kernel: np.ndarray, patch: int) -> np.ndarray:
    """conv2d stride=patch, no padding/bias; returns [n_tok, n_embd] raster order."""

    cols = patch_columns(pixels_chw, patch)
    out_c = kernel.shape[0]
    return cols @ kernel.reshape(out_c, -1).T


def bilinear_resize_pos(
    table: np.ndarray, src_grid: int, dst_w: int, dst_h: int
) -> np.ndarray:
    """Half-pixel bilinear resize of a [src_grid*src_grid, C] table to raster
    [dst_w*dst_h, C], matching ggml_interpolate BILINEAR (align_corners=False,
    clamped indices and fractions)."""

    c = table.shape[1]
    if dst_w == src_grid and dst_h == src_grid:
        return np.ascontiguousarray(table, dtype=np.float32)
    src = table.reshape(src_grid, src_grid, c)
    sf_w = src_grid / dst_w
    sf_h = src_grid / dst_h
    out = np.empty((dst_h, dst_w, c), dtype=np.float32)
    for iy in range(dst_h):
        y = (iy + 0.5) * sf_h - 0.5
        y0 = int(np.floor(y))
        y1 = y0 + 1
        dy = min(max(y - y0, 0.0), 1.0)
        y0c = min(max(y0, 0), src_grid - 1)
        y1c = min(max(y1, 0), src_grid - 1)
        for ix in range(dst_w):
            x = (ix + 0.5) * sf_w - 0.5
            x0 = int(np.floor(x))
            x1 = x0 + 1
            dx = min(max(x - x0, 0.0), 1.0)
            x0c = min(max(x0, 0), src_grid - 1)
            x1c = min(max(x1, 0), src_grid - 1)
            a = src[y0c, x0c]
            b = src[y0c, x1c]
            cc = src[y1c, x0c]
            d = src[y1c, x1c]
            out[iy, ix] = (
                a * (1 - dx) * (1 - dy) + b * dx * (1 - dy)
                + cc * (1 - dx) * dy + d * dx * dy
            )
    return out.reshape(dst_h * dst_w, c)


def window_permutations(
    grid_w: int, grid_h: int, win: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """sp_perm, inv_perm, 1-indexed rope pos_w/pos_h (permuted order), window sizes."""

    n_tok = grid_w * grid_h
    nwin_h = (grid_h + win - 1) // win
    nwin_w = (grid_w + win - 1) // win
    sp: list[int] = []
    slens: list[int] = []
    for wy in range(nwin_h):
        for wx in range(nwin_w):
            cnt = 0
            for hh in range(win):
                for ww in range(win):
                    gy = wy * win + hh
                    gx = wx * win + ww
                    if gy < grid_h and gx < grid_w:
                        sp.append(gy * grid_w + gx)
                        cnt += 1
            if cnt > 0:
                slens.append(cnt)
    sp_perm = np.asarray(sp, dtype=np.int64)
    inv_perm = np.empty(n_tok, dtype=np.int64)
    inv_perm[sp_perm] = np.arange(n_tok)
    pos_w = (sp_perm % grid_w) + 1
    pos_h = (sp_perm // grid_w) + 1
    return sp_perm, inv_perm, pos_w, pos_h, slens


def window_mask(n_tok: int, slens: list[int]) -> np.ndarray:
    """Additive block-diagonal mask in permuted order (0 keep, -inf drop)."""

    mask = np.full((n_tok, n_tok), -np.inf, dtype=np.float32)
    off = 0
    for s in slens:
        mask[off:off + s, off:off + s] = 0.0
        off += s
    return mask


def downsample_permutation(grid_w: int, grid_h: int, f: int) -> np.ndarray:
    dsp: list[int] = []
    for oy in range(grid_h // f):
        for ox in range(grid_w // f):
            for ry in range(f):
                for rx in range(f):
                    dsp.append((oy * f + ry) * grid_w + (ox * f + rx))
    return np.asarray(dsp, dtype=np.int64)


def pixel_shuffle(x: np.ndarray, grid_w: int, grid_h: int, f: int) -> np.ndarray:
    """[n_tok, C] raster -> [n_out, C*f*f] channel-outer (feature[c*f*f+s])."""

    dsp = downsample_permutation(grid_w, grid_h, f)
    gathered = x[dsp]                      # [n_out*f*f, C]
    n_out = (grid_w // f) * (grid_h // f)
    c = x.shape[1]
    grouped = gathered.reshape(n_out, f * f, c)
    return np.ascontiguousarray(grouped.transpose(0, 2, 1).reshape(n_out, c * f * f))


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps) * weight + bias).astype(np.float32)


def erf_approx(x: np.ndarray) -> np.ndarray:
    """Vectorized erf, Abramowitz & Stegun 7.1.26 (max abs err ~1.5e-7).

    Keeps the reference torch-free and scipy-free; the error bound is pinned
    against math.erf in the unit tier and sits far below the f32
    accumulation-order drift of the cross-engine oracle gate.
    """

    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
        + t * (-1.453152027 + t * 1.061405429))))
    return (sign * (1.0 - poly * np.exp(-ax * ax))).astype(np.float32)


def gelu_erf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (0.5 * x * (1.0 + erf_approx(x / np.float32(np.sqrt(2.0))))).astype(np.float32)


def rope_2d(
    x: np.ndarray, pos_w: np.ndarray, pos_h: np.ndarray, head_dim: int, theta: float
) -> np.ndarray:
    """Interleaved 2D RoPE; x is [n_tok, n_heads, head_dim].

    Dims [0, head_dim/2) rotate by width position, [head_dim/2, head_dim) by
    height position; each half carries head_dim/4 interleaved pairs with
    inv_freq = theta ** (-2i / (head_dim/2)).
    """

    x = np.asarray(x, dtype=np.float32)
    half = head_dim // 2
    pairs = half // 2
    inv_freq = theta ** (-np.arange(pairs, dtype=np.float32) * (2.0 / half))
    out = x.copy()
    for positions, lo in ((pos_w, 0), (pos_h, half)):
        angles = positions.astype(np.float32)[:, None] * inv_freq[None, :]
        cos = np.cos(angles)[:, None, :]
        sin = np.sin(angles)[:, None, :]
        x0 = x[..., lo:lo + half:2]
        x1 = x[..., lo + 1:lo + half:2]
        out[..., lo:lo + half:2] = x0 * cos - x1 * sin
        out[..., lo + 1:lo + half:2] = x0 * sin + x1 * cos
    return out


def attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None, scale: float
) -> np.ndarray:
    """Bidirectional multi-head attention; q/k/v [n_tok, n_heads, head_dim]."""

    n_tok, n_heads, d = q.shape
    qh = q.transpose(1, 0, 2)
    kh = k.transpose(1, 0, 2)
    vh = v.transpose(1, 0, 2)
    logits = np.einsum("hqd,hkd->hqk", qh, kh) * np.float32(scale)
    if mask is not None:
        logits = logits + mask[None, :, :]
    logits = logits - logits.max(axis=-1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=-1, keepdims=True)
    out = np.einsum("hqk,hkd->qhd", w, vh)
    return np.ascontiguousarray(out.reshape(n_tok, n_heads * d), dtype=np.float32)


def gguf_weight_source(reader) -> WeightSource:
    def get(name: str) -> np.ndarray:
        return np.ascontiguousarray(reader.dequantize_tensor(name), dtype=np.float32)
    return get


def forward(
    pixels_chw: np.ndarray,
    get: WeightSource,
    config: MuseGlimmerVisionConfig,
) -> np.ndarray:
    """Encode one normalized CHW image; returns projected rows [n_out, projection_dim]."""

    cfg = config
    _, h, w = pixels_chw.shape
    patch = cfg.patch_size
    grid_h, grid_w = h // patch, w // patch
    n_tok = grid_h * grid_w
    f = cfg.spatial_merge_size
    if grid_h % f or grid_w % f:
        raise ValueError("patch grid must be divisible by the merge size")

    x = patchify(pixels_chw, get("v.patch_embd.weight"), patch)
    pos = bilinear_resize_pos(
        get("v.position_embd.weight"), cfg.pos_grid, grid_w, grid_h
    )
    x = (x + pos).astype(np.float32)

    sp_perm, inv_perm, pos_w, pos_h, slens = window_permutations(
        grid_w, grid_h, cfg.pos_grid
    )
    x = x[sp_perm]
    sp_mask = window_mask(n_tok, slens)

    x = layer_norm(x, get("v.pre_ln.weight"), get("v.pre_ln.bias"), cfg.norm_eps)
    scale = cfg.head_dim ** -0.5
    for il in range(cfg.n_layers):
        p = f"v.blk.{il}."
        residual = x
        cur = layer_norm(x, get(p + "ln1.weight"), get(p + "ln1.bias"), cfg.norm_eps)
        q = (cur @ get(p + "attn_q.weight").T + get(p + "attn_q.bias")).reshape(n_tok, cfg.n_heads, cfg.head_dim)
        k = (cur @ get(p + "attn_k.weight").T + get(p + "attn_k.bias")).reshape(n_tok, cfg.n_heads, cfg.head_dim)
        v = (cur @ get(p + "attn_v.weight").T + get(p + "attn_v.bias")).reshape(n_tok, cfg.n_heads, cfg.head_dim)
        q = rope_2d(q, pos_w, pos_h, cfg.head_dim, cfg.rope_theta)
        k = rope_2d(k, pos_w, pos_h, cfg.head_dim, cfg.rope_theta)
        is_global = il == cfg.n_layers - 1 or (il + 1) % cfg.sparse_factor == 0
        attn = attention(q, k, v, None if is_global else sp_mask, scale)
        cur = attn @ get(p + "attn_out.weight").T + get(p + "attn_out.bias")
        x = (residual + cur).astype(np.float32)
        residual = x
        cur = layer_norm(x, get(p + "ln2.weight"), get(p + "ln2.bias"), cfg.norm_eps)
        cur = cur @ get(p + "ffn_up.weight").T + get(p + "ffn_up.bias")
        cur = gelu_erf(cur)
        cur = cur @ get(p + "ffn_down.weight").T + get(p + "ffn_down.bias")
        x = (residual + cur).astype(np.float32)
    x = layer_norm(x, get("v.post_ln.weight"), get("v.post_ln.bias"), cfg.norm_eps)

    x = x[inv_perm]
    x = pixel_shuffle(x, grid_w, grid_h, f)
    x = gelu_erf(x @ get("mm.0.weight").T)
    x = gelu_erf(x @ get("mm.1.weight").T)
    x = x @ get("mm.2.weight").T
    return np.ascontiguousarray(x, dtype=np.float32)


__all__ = [
    "MuseGlimmerVisionConfig",
    "attention",
    "bilinear_resize_pos",
    "downsample_permutation",
    "forward",
    "gelu_erf",
    "gguf_weight_source",
    "grid_size",
    "layer_norm",
    "normalize_pixels",
    "patch_columns",
    "patchify",
    "pixel_shuffle",
    "rope_2d",
    "window_mask",
    "window_permutations",
]