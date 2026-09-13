"""Surya OCR 2 CPU-reference text decoder (strict oracle path).

NumPy/fp32 implementation of the Surya text stack — a causal Qwen3.5
hybrid decoder (18 gated-DeltaNet + 6 full-attention layers, interleaved
partial mRoPE, tied LM head) — used as the strict reference for hipEngine
GPU work. Fixture-generator parity target: ``tests/fixtures/surya/``
(torch fp32 oracle) via ``tests/test_live_surya_text_decoder.py``.

Shared family math (rope tables, RMSNorm, activations, half-split rope
application) is imported from the EVIE CPU reference, which is
oracle-validated for the same Qwen3.5 family with the identical mRoPE
configuration. Surya-specific differences handled here:

- GDN geometry is spec-driven (16 key heads x 128, 16 value heads x 128,
  time-step rank 16, conv width 4 over the fused 3x2048 qkv projection;
  no q/k head repeat since key heads == value heads).
- Full attention is 8 query heads x 256 over 2 KV heads (GQA 4x) with a
  sigmoid output gate, evaluated CAUSALLY over a growing KV cache.
- Decode is state-threaded: per-layer GDN conv window + recurrent state
  and per-layer KV cache advance across steps (chunked prefill and
  single-token steps share the same state container).
- Tied LM head: logits = final-norm hidden @ embed_tokens.T.

The public API is torch-free; torch appears only in the fixture
generator and tests never reach the hot path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from hipengine.kernels.cpu_reference.evie import (
    _EPS_RMS,
    _L2NORM_EPS,
    apply_rope_half,
    layer_norm,
    l2norm,
    rms_norm,
    silu,
    softplus,
    softmax,
    text_rope_tables,
)


# ---------------------------------------------------------------------------
# weights + spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuryaWeights:
    """Flat fp32 numpy arrays keyed by checkpoint tensor name."""

    tensors: dict[str, np.ndarray]

    @classmethod
    def load(cls, safetensors_path: str) -> "SuryaWeights":
        from hipengine.loading.safetensors import (
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


@dataclass(frozen=True)
class SuryaSpec:
    """Geometry constants for Surya OCR 2 (mirrors SuryaModelSpec)."""

    hidden_size: int = 1024
    num_layers: int = 24
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    mrope_section: tuple[int, int, int] = (11, 11, 10)
    intermediate_size: int = 3584
    vocab_size: int = 65425
    image_token_id: int = 11
    eos_token_id: int = 2
    gdn_num_key_heads: int = 16
    gdn_key_head_dim: int = 128
    gdn_num_value_heads: int = 16
    gdn_value_head_dim: int = 128
    gdn_conv_kernel: int = 4
    gdn_time_step_rank: int = 16
    vision_hidden_size: int = 768
    vision_depth: int = 12
    vision_num_heads: int = 12
    vision_patch_size: int = 16
    vision_temporal_patch_size: int = 2
    vision_spatial_merge_size: int = 2
    vision_intermediate_size: int = 3072
    vision_out_hidden_size: int = 1024
    vision_num_position_embeddings: int = 2304

    def vision_head_dim(self) -> int:
        return self.vision_hidden_size // self.vision_num_heads

    @classmethod
    def from_model_spec(cls, spec) -> "SuryaSpec":
        return cls(
            hidden_size=spec.hidden_size,
            num_layers=spec.num_layers,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=spec.num_key_value_heads,
            head_dim=spec.head_dim,
            rope_theta=spec.rope_theta,
            partial_rotary_factor=spec.partial_rotary_factor,
            mrope_section=tuple(spec.mrope_section),  # type: ignore[arg-type]
            intermediate_size=spec.intermediate_size,
            vocab_size=spec.vocab_size,
            image_token_id=spec.image_token_id,
            eos_token_id=spec.eos_token_id,
            gdn_num_key_heads=spec.gdn_num_key_heads,
            gdn_key_head_dim=spec.gdn_key_head_dim,
            gdn_num_value_heads=spec.gdn_num_value_heads,
            gdn_value_head_dim=spec.gdn_value_head_dim,
            gdn_conv_kernel=spec.gdn_conv_kernel,
            gdn_time_step_rank=spec.gdn_time_step_rank,
            vision_hidden_size=spec.vision_hidden_size,
            vision_depth=spec.vision_depth,
            vision_num_heads=spec.vision_num_heads,
            vision_patch_size=spec.vision_patch_size,
            vision_temporal_patch_size=spec.vision_temporal_patch_size,
            vision_spatial_merge_size=spec.vision_spatial_merge_size,
            vision_intermediate_size=spec.vision_intermediate_size,
            vision_out_hidden_size=spec.vision_out_hidden_size,
            vision_num_position_embeddings=spec.vision_num_position_embeddings,
        )

    def is_full_attention(self, layer: int) -> bool:
        # pinned schedule: 3x linear_attention + 1x full_attention, repeated
        return (layer + 1) % 4 == 0


# ---------------------------------------------------------------------------
# state containers
# ---------------------------------------------------------------------------


@dataclass
class GDNLayerState:
    """Causal-conv window and recurrent state for one GDN layer."""

    # raw (pre-conv) fused-qkv inputs, last (kernel-1) columns: (b, 3*inner, k-1)
    conv_window: np.ndarray
    # recurrent state: (b, num_value_heads, key_head_dim, value_head_dim)
    recurrent: np.ndarray


@dataclass
class TextState:
    """Decode state across the whole text stack."""

    # per GDN layer index -> state
    gdn: dict[int, GDNLayerState] = field(default_factory=dict)
    # per full-attention layer index -> (k, v), each (b, kv_heads, s, head_dim)
    kv: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    # number of tokens already processed (absolute positions)
    seq_len: int = 0
    # number of image-pad tokens already consumed from visual_features
    image_pads_consumed: int = 0


# ---------------------------------------------------------------------------
# layer bodies
# ---------------------------------------------------------------------------


def _gdn_layer_prefill(
    w: SuryaWeights,
    spec: SuryaSpec,
    layer: int,
    hidden: np.ndarray,
    init: GDNLayerState | None = None,
) -> tuple[np.ndarray, GDNLayerState]:
    """GDN layer over a prefill chunk; returns (out, post-chunk state).

    ``init`` (from an earlier chunk) seeds both the causal-conv window and
    the recurrent state so chunked prefill equals single-pass prefill.
    """

    p = f"model.language_model.layers.{layer}.linear_attn."
    b, s, d = hidden.shape
    hkv, hk = spec.gdn_num_key_heads, spec.gdn_key_head_dim
    nv, hv = spec.gdn_num_value_heads, spec.gdn_value_head_dim
    ts = spec.gdn_time_step_rank
    inner = nv * hv
    repeat = nv // hkv

    qkv = hidden @ w[p + "in_proj_qkv.weight"].astype(np.float32).T  # (b, s, 3*inner)
    z = (hidden @ w[p + "in_proj_z.weight"].astype(np.float32).T).reshape(b, s, nv, hv)
    beta_in = hidden @ w[p + "in_proj_b.weight"].astype(np.float32).T  # (b, s, ts)
    a_in = hidden @ w[p + "in_proj_a.weight"].astype(np.float32).T  # (b, s, ts)

    # causal conv1d (k=4) over the fused qkv projection, then SiLU
    k = spec.gdn_conv_kernel
    conv_w = w[p + "conv1d.weight"][:, 0, :]  # (3*inner, 4)
    qkv_t = qkv.transpose(0, 2, 1)  # (b, 3*inner, s)
    if init is None:
        padded = np.pad(qkv_t, ((0, 0), (0, 0), (k - 1, 0)))
        S0 = np.zeros((b, nv, hk, hv), dtype=np.float32)
    else:
        padded = np.concatenate([init.conv_window, qkv_t], axis=-1)
        S0 = init.recurrent
    conv = np.zeros_like(qkv_t)
    for t in range(s):
        window = padded[:, :, t : t + k]
        conv[:, :, t] = (window * conv_w[None]).sum(axis=-1)
    conv = silu(conv)
    # conv state = last (k-1) raw inputs (matches the torch cache convention)
    state = GDNLayerState(
        conv_window=np.ascontiguousarray(padded[:, :, -(k - 1) :])
        if k > 1
        else np.zeros((b, 3 * inner, 0), dtype=np.float32),
        recurrent=np.zeros((b, nv, hk, hv), dtype=np.float32),
    )

    qkv = conv.transpose(0, 2, 1)  # (b, s, 3*inner)
    key_dim = hkv * hk
    q, kout, v = np.split(qkv, [key_dim, key_dim * 2], axis=-1)
    q = q.reshape(b, s, hkv, hk)
    kt = kout.reshape(b, s, hkv, hk)
    v = v.reshape(b, s, nv, hv)
    if repeat > 1:
        q = np.repeat(q, repeat, axis=2)
        kt = np.repeat(kt, repeat, axis=2)

    beta = 1.0 / (1.0 + np.exp(-beta_in.astype(np.float32)))  # sigmoid
    A = np.exp(w[p + "A_log"].astype(np.float32))
    dt_bias = w[p + "dt_bias"].astype(np.float32)
    g = -A[None, None] * softplus(a_in + dt_bias[None, None])  # (b, s, ts->nv)

    # recurrent gated delta rule (fp32 state), threading state across steps
    qn = l2norm(q) * (1.0 / math.sqrt(hk))
    kn = l2norm(kt)
    out = np.zeros((b, s, nv, hv), dtype=np.float32)
    S = S0
    ge = np.exp(g).astype(np.float32)  # (b, s, nv)
    for t in range(s):
        S = S * ge[:, t][:, :, None, None]
        qt = qn[:, t]  # (b, nv, hk)
        kv_t = kn[:, t]  # (b, nv, hk)
        vt = v[:, t]  # (b, nv, hv)
        bt = beta[:, t]  # (b, nv)
        kv_mem = (S * kv_t[:, :, :, None]).sum(axis=2)  # (b, nv, hv)
        delta = (vt - kv_mem) * bt[:, :, None]
        S = S + kv_t[:, :, :, None] * delta[:, :, None, :]
        out[:, t] = (S * qt[:, :, :, None]).sum(axis=2)
    state.recurrent = S

    # RMSNormGated: rms(out) * w, then * silu(z)
    out = out.reshape(b, s, nv, hv)
    var = np.mean(out * out, axis=-1, keepdims=True)
    outn = out / np.sqrt(var + _EPS_RMS) * w[p + "norm.weight"].astype(np.float32)
    outn = (outn * silu(z)).reshape(b, s, -1)
    return outn @ w[p + "out_proj.weight"].astype(np.float32).T, state


def _gdn_layer_decode(
    w: SuryaWeights,
    spec: SuryaSpec,
    layer: int,
    hidden: np.ndarray,
    state: GDNLayerState,
) -> np.ndarray:
    """Single-token GDN step advancing the cached conv window + state."""

    p = f"model.language_model.layers.{layer}.linear_attn."
    b = hidden.shape[0]
    hkv, hk = spec.gdn_num_key_heads, spec.gdn_key_head_dim
    nv, hv = spec.gdn_num_value_heads, spec.gdn_value_head_dim
    ts = spec.gdn_time_step_rank
    inner = nv * hv
    repeat = nv // hkv
    k = spec.gdn_conv_kernel

    qkv = hidden @ w[p + "in_proj_qkv.weight"].astype(np.float32).T  # (b, 3*inner)
    z = (hidden @ w[p + "in_proj_z.weight"].astype(np.float32).T).reshape(b, nv, hv)
    beta_in = hidden @ w[p + "in_proj_b.weight"].astype(np.float32).T  # (b, ts)
    a_in = hidden @ w[p + "in_proj_a.weight"].astype(np.float32).T  # (b, ts)

    conv_w = w[p + "conv1d.weight"][:, 0, :]  # (3*inner, 4)
    col = qkv.transpose(0, 2, 1)  # (b, 3*inner, 1)
    window = np.concatenate([state.conv_window, col], axis=-1)[:, :, -k:]
    state.conv_window = np.ascontiguousarray(window[:, :, -(k - 1) :]) if k > 1 else window[:, :, :0]
    conv = (window * conv_w[None]).sum(axis=-1)  # (b, 3*inner)
    conv = silu(conv)

    key_dim = hkv * hk
    q, kout, v = np.split(conv, [key_dim, key_dim * 2], axis=-1)
    q = q.reshape(b, hkv, hk)
    kt = kout.reshape(b, hkv, hk)
    v = v.reshape(b, nv, hv)
    if repeat > 1:
        q = np.repeat(q, repeat, axis=1)
        kt = np.repeat(kt, repeat, axis=1)

    beta = 1.0 / (1.0 + np.exp(-beta_in.astype(np.float32)))[:, 0]  # (b, ts)
    A = np.exp(w[p + "A_log"].astype(np.float32))
    dt_bias = w[p + "dt_bias"].astype(np.float32)
    g = (-A[None] * softplus(a_in + dt_bias[None]))[:, 0]  # (b, ts)
    S = state.recurrent
    S = S * np.exp(g)[:, :, None, None]
    qn = l2norm(q[:, None])[:, 0] * (1.0 / math.sqrt(hk))  # (b, nv, hk)
    kn = l2norm(kt[:, None])[:, 0]
    kv_mem = (S * kn[:, :, :, None]).sum(axis=2)
    delta = (v - kv_mem) * beta[:, :, None]
    S = S + kn[:, :, :, None] * delta[:, :, None, :]
    state.recurrent = S
    out = (S * qn[:, :, :, None]).sum(axis=2)  # (b, nv, hv)

    var = np.mean(out * out, axis=-1, keepdims=True)
    outn = out / np.sqrt(var + _EPS_RMS) * w[p + "norm.weight"].astype(np.float32)
    outn = (outn * silu(z)).reshape(b, -1)
    return outn @ w[p + "out_proj.weight"].astype(np.float32).T


def _full_attention_layer(
    w: SuryaWeights,
    spec: SuryaSpec,
    layer: int,
    hidden: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    cache: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Causal full-attention layer with a KV cache.

    ``cache`` holds past (k, v), each (b, kv_heads, past_len, head_dim);
    returns (out, new_cache) covering the full history.
    """

    p = f"model.language_model.layers.{layer}.self_attn."
    b, s, d = hidden.shape
    nq, nk, hd = spec.num_attention_heads, spec.num_key_value_heads, spec.head_dim

    qp = hidden @ w[p + "q_proj.weight"].astype(np.float32).T  # (b, s, 2*nq*hd)
    q, gate = np.split(qp.reshape(b, s, nq, hd * 2), 2, axis=-1)
    gate = gate.reshape(b, s, -1)
    k = (hidden @ w[p + "k_proj.weight"].astype(np.float32).T).reshape(b, s, nk, hd)
    v = (hidden @ w[p + "v_proj.weight"].astype(np.float32).T).reshape(b, s, nk, hd)

    q = rms_norm(q, w[p + "q_norm.weight"])
    k = rms_norm(k, w[p + "k_norm.weight"])
    q = apply_rope_half(q, cos, sin)
    k = apply_rope_half(k, cos, sin)

    qh = q.transpose(0, 2, 1, 3)  # (b, nq, s, hd)
    kh = k.transpose(0, 2, 1, 3)
    vh = v.transpose(0, 2, 1, 3)
    if cache is not None:
        pk, pv = cache
        kh = np.concatenate([pk, kh], axis=2)
        vh = np.concatenate([pv, vh], axis=2)
    new_cache = (np.ascontiguousarray(kh), np.ascontiguousarray(vh))

    rep = nq // nk
    khr = np.repeat(kh, rep, axis=1)
    vhr = np.repeat(vh, rep, axis=1)
    past = 0 if cache is None else cache[0].shape[2]
    att = qh @ khr.transpose(0, 1, 3, 2) * (hd**-0.5)
    total = past + s
    causal = np.triu(np.ones((s, total), dtype=bool), past + 1)
    att = np.where(causal[None, None], -np.inf, att)
    att = softmax(att)
    out = (att @ vhr).transpose(0, 2, 1, 3).reshape(b, s, -1)
    out = out * (1.0 / (1.0 + np.exp(-gate.astype(np.float32))))  # sigmoid gate
    return out @ w[p + "o_proj.weight"].astype(np.float32).T, new_cache


# ---------------------------------------------------------------------------
# vision tower
# ---------------------------------------------------------------------------


def _pixel_rows_to_patches(pixel_rows: np.ndarray, spec: SuryaSpec) -> np.ndarray:
    """(n, ch*t*p*p) processor rows -> (n, ch, t, p, p) patch tensors."""

    ch = 3
    return pixel_rows.reshape(
        -1,
        ch,
        spec.vision_temporal_patch_size,
        spec.vision_patch_size,
        spec.vision_patch_size,
    )


def vision_patch_embed(
    w: SuryaWeights, spec: SuryaSpec, pixel_rows: np.ndarray
) -> np.ndarray:
    """Patch projection (conv-as-matmul) + bias: (n, vision_hidden)."""

    patches = _pixel_rows_to_patches(pixel_rows, spec)
    weight = w["model.visual.patch_embed.proj.weight"]  # (vh, 3, t, p, p)
    bias = w["model.visual.patch_embed.proj.bias"]
    vh = spec.vision_hidden_size
    n = patches.shape[0]
    flat_w = weight.reshape(vh, -1)  # (vh, ch*t*p*p)
    return patches.reshape(n, -1) @ flat_w.T + bias[None, :]


def _merge_block_major_coords(
    grid_thw: list[tuple[int, int, int]], merge: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-patch (row, col) grid coords in the processor's spatial-merge
    block-major order, plus per-patch image segment ids."""

    rows, cols, seg = [], [], []
    for si, (t, h, pw) in enumerate(grid_thw):
        n = t * h * pw
        blocks_w = pw // merge
        within = np.arange(n) % (h * pw)
        in_col = within % merge
        in_row = (within // merge) % merge
        block_col = (within // (merge * merge)) % blocks_w
        block_row = within // (merge * merge * blocks_w)
        rows.append(block_row * merge + in_row)
        cols.append(block_col * merge + in_col)
        seg.extend([si] * n)
    return np.concatenate(rows), np.concatenate(cols), np.array(seg)


def vision_pos_embed(
    w: SuryaWeights,
    spec: SuryaSpec,
    grid_thw: list[tuple[int, int, int]],
    coords: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Bilinear (align_corners=True) resample of the learned 48x48 table
    onto each patch's (row, col) grid coordinate: (n, vision_hidden)."""

    side = int(round(spec.vision_num_position_embeddings**0.5))
    assert side * side == spec.vision_num_position_embeddings
    table = w["model.visual.pos_embed.weight"]  # (side*side, vh)
    rows, cols = coords
    counts = [t * h * pw for (t, h, pw) in grid_thw]
    seg = np.repeat(np.arange(len(grid_thw)), counts)[: len(rows)]
    heights = np.array([grid_thw[si][1] for si in seg], dtype=np.float64)
    widths = np.array([grid_thw[si][2] for si in seg], dtype=np.float64)
    # align_corners=True closed form: endpoints map to 0 and side-1
    src_r = rows.astype(np.float64) * (side - 1) / np.clip(heights - 1, 1, None)
    src_c = cols.astype(np.float64) * (side - 1) / np.clip(widths - 1, 1, None)

    def _taps(src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        floor = np.floor(src)
        taps = np.clip(
            np.stack([floor, floor + 1], axis=-1).astype(np.int64), 0, side - 1
        )
        dist = np.abs(src[:, None] - floor[:, None] - np.array([0.0, 1.0])[None, :])
        weights = np.clip(1.0 - dist, 0.0, None)
        return taps, weights

    r_taps, r_w = _taps(src_r)
    c_taps, c_w = _taps(src_c)
    # separable 2D: outer product of the per-axis taps/weights
    idx = r_taps[:, :, None] * side + c_taps[:, None, :]  # (n, 2, 2)
    wts = r_w[:, :, None] * c_w[:, None, :]
    n = idx.shape[0]
    flat = idx.reshape(n, 4)
    wflat = wts.reshape(n, 4)
    # Coordinates use float64 for interpolation, but the model's activations
    # are fp32. Do not promote every subsequent vision block to float64.
    return (table[flat] * wflat[:, :, None]).sum(axis=1).astype(np.float32)


def vision_rotary(
    spec: SuryaSpec, rows: np.ndarray, cols: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin (s, head_dim) for full-dim 2-axis vision rotary (theta 1e4).

    The tower rotary uses half-dim pairs: inv_freq has head_dim//4 entries
    (dim = head_dim // 2 in the reference), giving head_dim//2 rotary
    frequencies per axis; h freqs then w freqs are concatenated.
    """

    hd = spec.vision_head_dim()  # 64
    quarter = hd // 4  # 16 inv freq entries (dim = 32)
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, 2 * quarter, 2, dtype=np.float32) / (2 * quarter)))
    freqs = np.stack([rows, cols], axis=-1).astype(np.float32)[:, :, None] * inv_freq[
        None, None, :
    ]
    freqs = freqs.reshape(-1, 2 * quarter)  # (s, 32): [h freqs, w freqs]
    emb = np.concatenate([freqs, freqs], axis=-1)  # (s, hd)
    return np.cos(emb), np.sin(emb)


def _rotate_half_vision(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _gelu_tanh_np(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def _erf(x: np.ndarray) -> np.ndarray:
    # vectorized erf without scipy: Abramowitz-Stegun 7.1.26 (|eps|<1.5e-7)
    a1, a2, a3, a4, a5 = (
        0.254829592,
        -0.284496736,
        1.421413741,
        -1.453152027,
        1.061405429,
    )
    pf = 0.3275911
    sign = np.sign(x)
    z = np.abs(x)
    t = 1.0 / (1.0 + pf * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-z * z)
    return sign * y


def _gelu_erf_np(x: np.ndarray) -> np.ndarray:
    return x * 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def vision_forward(
    w: SuryaWeights,
    spec: SuryaSpec,
    pixel_rows: np.ndarray,
    grid_thw: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full vision tower; returns (patch_embed_out, tower_out, merged).

    tower_out: (n, vision_hidden) pre-merger hidden states; merged:
    (n / merge^2, vision_out_hidden) image features in merge-block order.
    """

    vh = spec.vision_hidden_size
    nh, hd = spec.vision_num_heads, spec.vision_head_dim()
    merge = spec.vision_spatial_merge_size

    x = vision_patch_embed(w, spec, pixel_rows)
    patch_out = x.copy()
    coords = _merge_block_major_coords(grid_thw, merge)[:2]
    x = x + vision_pos_embed(w, spec, grid_thw, coords)
    cos, sin = vision_rotary(spec, *coords)

    n = x.shape[0]
    for i in range(spec.vision_depth):
        p = f"model.visual.blocks.{i}."
        h = layer_norm(x, w[p + "norm1.weight"], w[p + "norm1.bias"])
        qkv = (h @ w[p + "attn.qkv.weight"].T + w[p + "attn.qkv.bias"]).reshape(
            n, 3, nh, hd
        )
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (n, nh, hd)
        # 2D rotary on q/k (rotate_half over the hd-dim head)
        cs = cos[:, None, :]
        sn = sin[:, None, :]
        q = q * cs + _rotate_half_vision(q) * sn
        k = k * cs + _rotate_half_vision(k) * sn
        # bidirectional within each image segment
        qh = q.transpose(1, 0, 2)  # (nh, n, hd)
        kh = k.transpose(1, 0, 2)
        v_ = v.transpose(1, 0, 2)
        att = softmax(qh @ kh.transpose(0, 2, 1) * (hd**-0.5))
        out = (att @ v_).transpose(1, 0, 2).reshape(n, -1)
        x = x + out @ w[p + "attn.proj.weight"].T + w[p + "attn.proj.bias"]
        h2 = layer_norm(x, w[p + "norm2.weight"], w[p + "norm2.bias"])
        fc1 = h2 @ w[p + "mlp.linear_fc1.weight"].T + w[p + "mlp.linear_fc1.bias"]
        x = (
            x
            + _gelu_tanh_np(fc1) @ w[p + "mlp.linear_fc2.weight"].T
            + w[p + "mlp.linear_fc2.bias"]
        )

    tower_out = x
    # merger: per-patch LayerNorm -> merge-block reshape (n/4, 4*vh) ->
    # fc1 -> erf GELU -> fc2
    xm = layer_norm(
        x,
        w["model.visual.merger.norm.weight"],
        w["model.visual.merger.norm.bias"],
    )
    xm = xm.reshape(-1, merge * merge * vh)
    fc1 = (
        xm @ w["model.visual.merger.linear_fc1.weight"].T
        + w["model.visual.merger.linear_fc1.bias"]
    )
    merged = (
        _gelu_erf_np(fc1) @ w["model.visual.merger.linear_fc2.weight"].T
        + w["model.visual.merger.linear_fc2.bias"]
    )
    return patch_out, tower_out, merged


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------



def _mlp(w: SuryaWeights, prefix: str, x: np.ndarray) -> np.ndarray:
    gate = x @ w[prefix + "mlp.gate_proj.weight"].astype(np.float32).T
    up = x @ w[prefix + "mlp.up_proj.weight"].astype(np.float32).T
    return (silu(gate) * up) @ w[prefix + "mlp.down_proj.weight"].astype(np.float32).T


def text_prefill(
    w: SuryaWeights,
    spec: SuryaSpec,
    input_ids: np.ndarray,
    position_ids: np.ndarray,
    visual_features: np.ndarray | None = None,
    state: TextState | None = None,
) -> tuple[np.ndarray, TextState]:
    """Prefill a chunk; returns (final-norm hidden (b, s, d), state).

    input_ids: (b, s). position_ids: (3, s) absolute mRoPE positions for
    this chunk. visual_features (b, n_image, d) replaces image-token rows.
    """

    if state is None:
        state = TextState()
    emb = w["model.language_model.embed_tokens.weight"]
    x = emb[input_ids]  # (b, s, d)
    if visual_features is not None:
        for bi in range(x.shape[0]):
            mask = input_ids[bi] == spec.image_token_id
            offset = state.image_pads_consumed
            n_pads = int(mask.sum())
            x[bi, mask] = visual_features[bi][offset : offset + n_pads]
    cos, sin = text_rope_tables(spec, position_ids)
    past = state.seq_len
    for layer in range(spec.num_layers):
        lp = f"model.language_model.layers.{layer}."
        h = rms_norm(x, w[lp + "input_layernorm.weight"])
        if spec.is_full_attention(layer):
            cache = state.kv.get(layer)
            attn_out, new_cache = _full_attention_layer(
                w, spec, layer, h, cos, sin, cache
            )
            state.kv[layer] = new_cache
        else:
            attn_out, gstate = _gdn_layer_prefill(
                w, spec, layer, h, init=state.gdn.get(layer)
            )
            state.gdn[layer] = gstate
        x = x + attn_out
        h2 = rms_norm(x, w[lp + "post_attention_layernorm.weight"])
        x = x + _mlp(w, lp, h2)
    if visual_features is not None:
        for bi in range(x.shape[0]):
            state.image_pads_consumed += int(
                (input_ids[bi] == spec.image_token_id).sum()
            )
    state.seq_len = past + x.shape[1]
    return rms_norm(x, w["model.language_model.norm.weight"]), state


def text_decode_step(
    w: SuryaWeights,
    spec: SuryaSpec,
    token_id: int,
    state: TextState,
    position: int,
) -> np.ndarray:
    """One teacher-forced decode step; returns logits (b, vocab)."""

    emb = w["model.language_model.embed_tokens.weight"]
    x = emb[np.array([[token_id]], dtype=np.int64)]  # (b, 1, d)
    cos, sin = text_rope_tables(
        spec, np.array([[position]], dtype=np.int64).repeat(3, axis=0)
    )
    for layer in range(spec.num_layers):
        lp = f"model.language_model.layers.{layer}."
        h = rms_norm(x, w[lp + "input_layernorm.weight"])
        if spec.is_full_attention(layer):
            attn_out, new_cache = _full_attention_layer(
                w, spec, layer, h, cos, sin, state.kv[layer]
            )
            state.kv[layer] = new_cache
        else:
            attn_out = _gdn_layer_decode(w, spec, layer, h, state.gdn[layer])
        x = x + attn_out
        h2 = rms_norm(x, w[lp + "post_attention_layernorm.weight"])
        x = x + _mlp(w, lp, h2)
    state.seq_len += 1
    final = rms_norm(x, w["model.language_model.norm.weight"])
    return final[:, -1] @ emb.T  # tied LM head


def greedy_generate(
    w: SuryaWeights,
    spec: SuryaSpec,
    input_ids: np.ndarray,
    position_ids: np.ndarray,
    max_new_tokens: int,
    visual_features: np.ndarray | None = None,
) -> list[int]:
    """Greedy AR continuation with EOS stopping (behavioral smoke)."""

    hidden, state = text_prefill(
        w, spec, input_ids, position_ids, visual_features=visual_features
    )
    emb = w["model.language_model.embed_tokens.weight"]
    logits = hidden[:, -1] @ emb.T
    next_token = int(np.argmax(logits[0, -1]))
    generated: list[int] = []
    pos = position_ids[:, -1].astype(np.int64).copy()
    for _ in range(max_new_tokens):
        if next_token == spec.eos_token_id:
            break
        generated.append(next_token)
        pos = pos + 1
        logits = text_decode_step(
            w, spec, next_token, state, int(pos[0])
        )
        next_token = int(np.argmax(logits[0]))
    return generated
