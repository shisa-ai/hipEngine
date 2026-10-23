"""CPU NumPy oracle for the Gemma 4 GGUF vision tower (``clip`` / ``gemma4v``).

Mirrors the reference vision tower: ``2 * (pixels - 0.5)`` and a bias-free patch
projection plus two learned per-axis position lookups; per-layer RMSNorm,
non-causal attention with **no** ``1/sqrt(head_dim)`` scaling, per-head Q/K
RMSNorm, scale-free V RMSNorm, two-axis (x then y) NeoX RoPE, and a gated
tanh-GELU MLP with a post-norm after each sub-layer; then a
``pooling_kernel_size`` average pool scaled by ``sqrt(hidden_size)``, optional
standardization, a scale-free RMSNorm, and the input projection.

GGUF stores the patch kernel as ``(hidden, channel, patch, patch)``.  Patches are
channel-last by default; a caller holding the kernel's own ``(channel, patch,
patch)`` order passes ``patch_order="chw"``.

Operations accumulate in the working floating dtype: float32 for GGUF weights,
float64 when a caller supplies float64 arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

ArrayLike = Any

_ROPE_PAIR_ALIGNMENT = 4
_CLAMP_FIELDS = ("input_min", "input_max", "output_min", "output_max")
PATCH_ORDER_CHANNEL_LAST = "hwc"
PATCH_ORDER_CHANNEL_FIRST = "chw"
_PATCH_ORDERS = (PATCH_ORDER_CHANNEL_LAST, PATCH_ORDER_CHANNEL_FIRST)


# --- Small numeric helpers -----------------------------------------------------


def _as_float(name: str, value: ArrayLike) -> np.ndarray:
    """Return ``value`` as float32, preserving an explicit float64 request."""

    array = np.asarray(value)
    if array.dtype == np.float64:
        return array
    if not np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32)
    return array.astype(np.float32, copy=False)


def _scalar(name: str, value: Any) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return number


def _non_negative_scalar(name: str, value: Any) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number, got {value!r}")
    return number


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _positions(name: str, value: ArrayLike, length: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.shape[0] != length:
        raise ValueError(
            f"{name} must hold one position per patch ({length}), got {array.shape[0]}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.equal(np.mod(array, 1), 0)):
            raise ValueError(f"{name} must hold integer patch positions")
    array = array.astype(np.int64, copy=False)
    if array.size and int(array.min()) < 0:
        raise ValueError(f"{name} must hold non-negative patch positions")
    return array


def gemma4_vision_rms_norm(
    x: ArrayLike,
    weight: ArrayLike | None = None,
    *,
    eps: float,
) -> np.ndarray:
    """``Gemma4RMSNorm``: RMS over the last axis, optional learned scale."""

    array = _as_float("x", x)
    if array.ndim < 1 or array.shape[-1] <= 0:
        raise ValueError("x must have a non-empty trailing axis")
    epsilon = _non_negative_scalar("eps", eps)
    mean_square = np.mean(array * array, axis=-1, keepdims=True) + epsilon
    normed = array * np.power(mean_square, -0.5)
    if weight is None:
        return normed
    scale = _as_float("weight", weight)
    if scale.shape != (array.shape[-1],):
        raise ValueError(
            f"RMSNorm weight must have shape ({array.shape[-1]},), got {scale.shape}"
        )
    return normed * scale


def gemma4_vision_gelu_tanh(x: ArrayLike) -> np.ndarray:
    """``gelu_pytorch_tanh`` — the reference vision tower's activation."""

    array = _as_float("x", x)
    # Python floats, not NumPy scalars: under NEP 50 a NumPy float64 scalar
    # would promote the whole activation to float64 and leave the working dtype.
    inner = math.sqrt(2.0 / math.pi) * (array + 0.044715 * array**3)
    return 0.5 * array * (1.0 + np.tanh(inner))


def gemma4_vision_rope_inv_freq(head_dim: int, theta: float) -> np.ndarray:
    """Per-axis rotary frequencies: ``head_dim // 4`` pairs, shared by both axes."""

    head_dim = _positive_int("head_dim", head_dim)
    if head_dim % _ROPE_PAIR_ALIGNMENT:
        raise ValueError(
            f"head_dim must be a multiple of {_ROPE_PAIR_ALIGNMENT} for two-axis rotary "
            f"position embeddings, got {head_dim}"
        )
    spatial_dim = head_dim // 2
    exponents = np.arange(0, spatial_dim, 2, dtype=np.float64) / float(spatial_dim)
    return 1.0 / (float(_scalar("theta", theta)) ** exponents)


def gemma4_vision_apply_rope(
    x: ArrayLike,
    pos_x: ArrayLike,
    pos_y: ArrayLike,
    *,
    theta: float,
) -> np.ndarray:
    """Rotate ``x`` ``(patches, ..., head_dim)``: x-positions in the first half."""

    array = _as_float("x", x)
    if array.ndim < 2:
        raise ValueError("x must have shape (patches, ..., head_dim)")
    head_dim = array.shape[-1]
    inv_freq = gemma4_vision_rope_inv_freq(head_dim, theta)
    half = head_dim // 2
    positions = (
        _positions("pos_x", pos_x, array.shape[0]),
        _positions("pos_y", pos_y, array.shape[0]),
    )

    def rotate(block: np.ndarray, position: np.ndarray) -> np.ndarray:
        angle = position[:, None].astype(np.float64) * inv_freq[None, :]
        shape = (array.shape[0],) + (1,) * (array.ndim - 2) + (inv_freq.size,)
        angle = angle.reshape(shape)
        cos = np.cos(angle).astype(array.dtype)
        sin = np.sin(angle).astype(array.dtype)
        first = block[..., : inv_freq.size]
        second = block[..., inv_freq.size :]
        return np.concatenate((first * cos - second * sin, first * sin + second * cos), axis=-1)

    return np.concatenate(
        (rotate(array[..., :half], positions[0]), rotate(array[..., half:], positions[1])),
        axis=-1,
    )


def gemma4_vision_linear(
    x: ArrayLike,
    weight: ArrayLike,
    *,
    clamp: Sequence[float] | None = None,
) -> np.ndarray:
    """Bias-free ``nn.Linear``; ``clamp`` is ``(in_min, in_max, out_min, out_max)``."""

    array = _as_float("x", x)
    matrix = _as_float("weight", weight)
    if matrix.ndim != 2:
        raise ValueError(f"weight must be two-dimensional, got shape {matrix.shape}")
    if array.shape[-1] != matrix.shape[1]:
        raise ValueError(
            f"weight expects {matrix.shape[1]} input features, got {array.shape[-1]}"
        )
    if clamp is not None:
        low, high, _, _ = (float(item) for item in clamp)
        array = np.clip(array, low, high)
    out = array @ matrix.T
    if clamp is not None:
        _, _, low, high = (float(item) for item in clamp)
        out = np.clip(out, low, high)
    return out


# --- Geometry and weights ------------------------------------------------------


# Weights the reference graph projects through ``Gemma4ClippableLinear``.
_CLAMP_CAPABLE_SLOTS = (
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_out",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
)


@dataclass(frozen=True)
class Gemma4VisionGeometry:
    """Dimensions the forward needs."""

    block_count: int
    hidden_size: int
    intermediate_size: int
    output_size: int
    heads: int
    head_count_kv: int
    head_dim: int
    patch_size: int
    pooling_kernel_size: int
    rms_norm_eps: float
    rope_theta: float

    def __post_init__(self) -> None:
        for name in (
            "block_count",
            "hidden_size",
            "intermediate_size",
            "output_size",
            "heads",
            "head_count_kv",
            "head_dim",
            "patch_size",
            "pooling_kernel_size",
        ):
            _positive_int(name, getattr(self, name))
        _non_negative_scalar("rms_norm_eps", self.rms_norm_eps)
        _scalar("rope_theta", self.rope_theta)
        if self.hidden_size != self.heads * self.head_dim:
            raise ValueError(
                f"hidden_size {self.hidden_size} != heads {self.heads} * head_dim {self.head_dim}"
            )
        if self.heads % self.head_count_kv:
            raise ValueError(
                f"heads {self.heads} is not divisible by head_count_kv {self.head_count_kv}"
            )
        if self.head_dim % _ROPE_PAIR_ALIGNMENT:
            raise ValueError(
                f"head_dim {self.head_dim} must be a multiple of {_ROPE_PAIR_ALIGNMENT}"
            )

    @property
    def patch_values(self) -> int:
        return 3 * self.patch_size * self.patch_size

    @property
    def pooling_block_patches(self) -> int:
        return self.pooling_kernel_size * self.pooling_kernel_size

    def layer_slot(self, layer_id: int, slot: str) -> str:
        return f"layers.{int(layer_id)}.{slot}"


# Canonical field -> accepted attribute names, in preference order.  The second
# name is the runtime's spelling (``spatial_merge_size`` / ``norm_epsilon``).
_GEOMETRY_CONFIG_FIELDS: tuple[tuple[str, ...], ...] = (
    ("block_count",),
    ("hidden_size",),
    ("intermediate_size",),
    ("output_size",),
    ("heads",),
    ("head_count_kv",),
    ("patch_size",),
    ("pooling_kernel_size", "spatial_merge_size"),
    ("rms_norm_eps", "norm_epsilon"),
    ("rope_theta",),
)


def gemma4_vision_geometry_from_config(config: Any) -> Gemma4VisionGeometry:
    """Build the geometry from any object exposing these fields (duck-typed).

    ``spatial_merge_size`` and ``norm_epsilon`` are accepted as alternates for
    ``pooling_kernel_size`` and ``rms_norm_eps``; ``head_count_kv`` defaults to
    ``heads``.
    """

    values: dict[str, Any] = {}
    for names in _GEOMETRY_CONFIG_FIELDS:
        for name in names:
            if hasattr(config, name):
                values[names[0]] = getattr(config, name)
                break
        else:
            if names[0] == "head_count_kv":
                continue
            raise ValueError(
                "Gemma 4 vision config is missing required attribute "
                f"{names[0]!r} (or {names[1:]!r})"
            )
    values.setdefault("head_count_kv", values["heads"])
    heads = int(values["heads"])
    hidden_size = int(values["hidden_size"])
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        if hidden_size % heads:
            raise ValueError(
                f"hidden_size {hidden_size} is not divisible by heads {heads}"
            )
        head_dim = hidden_size // heads
    return Gemma4VisionGeometry(head_dim=int(head_dim), **values)


@dataclass(frozen=True)
class Gemma4VisionLayerWeights:
    """One encoder block's dequantized weights."""

    ln1: np.ndarray
    ln2: np.ndarray
    attn_q: np.ndarray
    attn_k: np.ndarray
    attn_v: np.ndarray
    attn_out: np.ndarray
    attn_q_norm: np.ndarray
    attn_k_norm: np.ndarray
    attn_post_norm: np.ndarray
    ffn_post_norm: np.ndarray
    ffn_gate: np.ndarray
    ffn_up: np.ndarray
    ffn_down: np.ndarray
    out_scale: np.ndarray | None = None
    clamps: Mapping[str, tuple[float, float, float, float]] = field(default_factory=dict)

    def clamp(self, slot: str) -> tuple[float, float, float, float] | None:
        return self.clamps.get(slot) if self.clamps else None


@dataclass(frozen=True)
class Gemma4VisionWeights:
    """Dequantized weights of one vision tower."""

    patch_embd: np.ndarray
    position_embd: np.ndarray
    input_projection: np.ndarray
    layers: tuple[Gemma4VisionLayerWeights, ...]
    std_bias: np.ndarray | None = None
    std_scale: np.ndarray | None = None
    input_projection_clamp: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class Gemma4VisionForwardResult:
    """Vision tower output for one image."""

    embeddings: np.ndarray
    hidden_states: np.ndarray
    pooled: np.ndarray
    num_soft_tokens: int
    grid_size: tuple[int, int]


def gemma4_vision_weight_shapes(
    geometry: Gemma4VisionGeometry,
) -> Mapping[str, tuple[int, ...]]:
    """Expected shape of every weight slot, keyed by canonical slot name."""

    hidden = geometry.hidden_size
    shapes: dict[str, tuple[int, ...]] = {
        "patch_embd": (hidden, 3, geometry.patch_size, geometry.patch_size),
        "position_embd": (2, -1, hidden),
        "input_projection": (geometry.output_size, hidden),
        "std_bias": (hidden,),
        "std_scale": (hidden,),
    }
    for layer_id in range(geometry.block_count):
        for name, shape in _per_layer_shapes(geometry).items():
            shapes[geometry.layer_slot(layer_id, name)] = shape
    return shapes


def _per_layer_shapes(geometry: Gemma4VisionGeometry) -> dict[str, tuple[int, ...]]:
    hidden = geometry.hidden_size
    head_dim = geometry.head_dim
    query_width = geometry.heads * head_dim
    kv_width = geometry.head_count_kv * head_dim
    return {
        "ln1": (hidden,),
        "ln2": (hidden,),
        "attn_q": (query_width, hidden),
        "attn_k": (kv_width, hidden),
        "attn_v": (kv_width, hidden),
        "attn_out": (hidden, query_width),
        "attn_q_norm": (head_dim,),
        "attn_k_norm": (head_dim,),
        "attn_post_norm": (hidden,),
        "ffn_post_norm": (hidden,),
        "ffn_gate": (geometry.intermediate_size, hidden),
        "ffn_up": (geometry.intermediate_size, hidden),
        "ffn_down": (hidden, geometry.intermediate_size),
        "out_scale": (1,),
    }


def gemma4_vision_weights_from_slots(
    slots: Mapping[str, ArrayLike],
    geometry: Gemma4VisionGeometry,
) -> Gemma4VisionWeights:
    """Assemble :class:`Gemma4VisionWeights` from the loader's slot-keyed mapping."""

    expected = gemma4_vision_weight_shapes(geometry)
    arrays: dict[str, np.ndarray] = {}
    for slot, shape in expected.items():
        if slot in ("std_bias", "std_scale", "position_embd") or slot.endswith(
            ".out_scale"
        ):
            continue
        if slot not in slots:
            raise KeyError(f"Gemma 4 vision weights are missing slot {slot!r}")
        array = _as_float(slot, slots[slot])
        if tuple(array.shape) != tuple(shape):
            raise ValueError(f"slot {slot!r} expected shape {shape}, got {array.shape}")
        arrays[slot] = array

    if "position_embd" not in slots:
        raise KeyError("Gemma 4 vision weights are missing slot 'position_embd'")
    position_embd = _as_float("position_embd", slots["position_embd"])
    if (
        position_embd.ndim != 3
        or position_embd.shape[0] != 2
        or position_embd.shape[1] <= 0
        or position_embd.shape[2] != geometry.hidden_size
    ):
        raise ValueError(
            "slot 'position_embd' expected shape (2, table_size, "
            f"{geometry.hidden_size}), got {position_embd.shape}"
        )

    clamps = _clamps_from_slots(slots, geometry)
    layer_fields = tuple(
        name for name in _per_layer_shapes(geometry) if name != "out_scale"
    )
    layers = []
    for layer_id in range(geometry.block_count):
        slot = geometry.layer_slot
        layer_clamps = {
            name: clamps[slot(layer_id, name)]
            for name in _CLAMP_CAPABLE_SLOTS
            if slot(layer_id, name) in clamps
        }
        out_scale_slot = slot(layer_id, "out_scale")
        out_scale = None
        if out_scale_slot in slots:
            out_scale = _as_float(out_scale_slot, slots[out_scale_slot]).reshape(())
        layers.append(
            Gemma4VisionLayerWeights(
                **{name: arrays[slot(layer_id, name)] for name in layer_fields},
                out_scale=out_scale,
                clamps=layer_clamps,
            )
        )

    std_bias = None if "std_bias" not in slots else _as_float("std_bias", slots["std_bias"])
    std_scale = None if "std_scale" not in slots else _as_float("std_scale", slots["std_scale"])
    if (std_bias is None) != (std_scale is None):
        raise ValueError("std_bias and std_scale must be supplied together")
    for name, value in (("std_bias", std_bias), ("std_scale", std_scale)):
        if value is not None and value.shape != (geometry.hidden_size,):
            raise ValueError(
                f"{name} expected shape ({geometry.hidden_size},), got {value.shape}"
            )

    return Gemma4VisionWeights(
        patch_embd=arrays["patch_embd"],
        position_embd=position_embd,
        input_projection=arrays["input_projection"],
        layers=tuple(layers),
        std_bias=std_bias,
        std_scale=std_scale,
        input_projection_clamp=clamps.get("input_projection"),
    )


def _clamps_from_slots(
    slots: Mapping[str, ArrayLike],
    geometry: Gemma4VisionGeometry,
) -> dict[str, tuple[float, float, float, float]]:
    """Collect optional ``Gemma4ClippableLinear`` clamp scalars."""

    clamps: dict[str, tuple[float, float, float, float]] = {}
    candidates = ["input_projection"]
    for layer_id in range(geometry.block_count):
        candidates.extend(
            geometry.layer_slot(layer_id, name) for name in _CLAMP_CAPABLE_SLOTS
        )
    for slot in candidates:
        fields = [f"{slot}.{name}" for name in _CLAMP_FIELDS]
        present = [name for name in fields if name in slots]
        if not present:
            continue
        if len(present) != len(fields):
            missing = ", ".join(name for name in fields if name not in slots)
            raise ValueError(f"clamp scalars for {slot!r} are incomplete; missing {missing}")
        clamps[slot] = tuple(  # type: ignore[assignment]
            float(np.asarray(slots[name]).reshape(-1)[0]) for name in fields
        )
    return clamps


# --- Forward stages ------------------------------------------------------------


def _patch_kernel(weight: np.ndarray, patch_order: str) -> np.ndarray:
    """Flatten the GGUF patch kernel to match a patch row ordering."""

    if patch_order == PATCH_ORDER_CHANNEL_LAST:
        return weight.transpose(0, 2, 3, 1).reshape(weight.shape[0], -1)
    return weight.reshape(weight.shape[0], -1)


def gemma4_vision_patch_embed(
    patches: ArrayLike,
    weight: ArrayLike,
    position_embd: ArrayLike | None = None,
    pos_x: ArrayLike | None = None,
    pos_y: ArrayLike | None = None,
    *,
    clamp: Sequence[float] | None = None,
    patch_order: str = PATCH_ORDER_CHANNEL_LAST,
) -> np.ndarray:
    """Pixel scaling, patch projection, and the two per-axis position lookups.

    ``patch_order`` is ``"hwc"`` for ``(row, column, channel)`` rows (what the
    preprocessing module emits) or ``"chw"`` for the GGUF kernel's own order.
    """
    if patch_order not in _PATCH_ORDERS:
        raise ValueError(
            f"patch_order must be one of {_PATCH_ORDERS}, got {patch_order!r}"
        )

    array = _as_float("patches", patches)
    kernel = _as_float("weight", weight)
    if kernel.ndim != 4:
        raise ValueError(
            "patch embedding weight must have shape (hidden, channels, patch, patch), "
            f"got {kernel.shape}"
        )
    if kernel.shape[2] != kernel.shape[3]:
        raise ValueError(
            f"patch embedding kernel must be square, got {kernel.shape[2]}x{kernel.shape[3]}"
        )
    values = 3 * kernel.shape[2] * kernel.shape[3]
    if array.ndim == 4:
        if array.shape[1:] != (kernel.shape[2], kernel.shape[3], 3):
            raise ValueError(
                "patches must have shape (num_patches, patch, patch, 3), "
                f"got {array.shape}"
            )
        if patch_order != PATCH_ORDER_CHANNEL_LAST:
            raise ValueError(
                "a four-dimensional patch array is channel-last; pass "
                f"patch_order='{PATCH_ORDER_CHANNEL_LAST}' or flatten it yourself"
            )
        array = array.reshape(array.shape[0], values)
    if array.ndim != 2 or array.shape[1] != values:
        raise ValueError(
            f"patches must have shape (num_patches, {values}), got {array.shape}"
        )
    if array.size and not (0.0 <= float(array.min()) and float(array.max()) <= 1.0):
        raise ValueError("patches must hold rescaled pixel values in [0, 1]")
    kernel = _patch_kernel(kernel, patch_order)
    hidden = gemma4_vision_linear(2.0 * (array - 0.5), kernel, clamp=clamp)
    if position_embd is None:
        return hidden
    if pos_x is None or pos_y is None:
        raise ValueError("position embeddings require both pos_x and pos_y")
    table = _as_float("position_embd", position_embd)
    if table.ndim != 3 or table.shape[0] != 2 or table.shape[2] != hidden.shape[-1]:
        raise ValueError(
            "position_embd must have shape (2, table_size, hidden_size), "
            f"got {table.shape}"
        )
    x = _positions("pos_x", pos_x, hidden.shape[0])
    y = _positions("pos_y", pos_y, hidden.shape[0])
    if x.size and int(x.max()) >= table.shape[1] or y.size and int(y.max()) >= table.shape[1]:
        raise ValueError(
            f"patch positions exceed the position table size {table.shape[1]}"
        )
    return hidden + table[0, x] + table[1, y]


def gemma4_vision_attention(
    hidden: ArrayLike,
    layer: Gemma4VisionLayerWeights,
    geometry: Gemma4VisionGeometry,
    pos_x: ArrayLike,
    pos_y: ArrayLike,
) -> np.ndarray:
    """Non-causal attention: Q/K/V projection and norm, RoPE, no scaling."""

    array = _as_float("hidden", hidden)
    if array.ndim != 2 or array.shape[1] != geometry.hidden_size:
        raise ValueError(
            f"hidden must have shape (patches, {geometry.hidden_size}), got {array.shape}"
        )
    num_patches = array.shape[0]
    head_dim = geometry.head_dim
    heads = geometry.heads
    kv_heads = geometry.head_count_kv

    query = gemma4_vision_linear(array, layer.attn_q, clamp=layer.clamp("attn_q"))
    query = query.reshape(num_patches, heads, head_dim)
    query = gemma4_vision_rms_norm(query, layer.attn_q_norm, eps=geometry.rms_norm_eps)
    query = gemma4_vision_apply_rope(query, pos_x, pos_y, theta=geometry.rope_theta)

    key = gemma4_vision_linear(array, layer.attn_k, clamp=layer.clamp("attn_k"))
    key = key.reshape(num_patches, kv_heads, head_dim)
    key = gemma4_vision_rms_norm(key, layer.attn_k_norm, eps=geometry.rms_norm_eps)
    key = gemma4_vision_apply_rope(key, pos_x, pos_y, theta=geometry.rope_theta)

    value = gemma4_vision_linear(array, layer.attn_v, clamp=layer.clamp("attn_v"))
    value = value.reshape(num_patches, kv_heads, head_dim)
    # The reference V norm carries no learned scale.
    value = gemma4_vision_rms_norm(value, None, eps=geometry.rms_norm_eps)

    if kv_heads != heads:
        repeats = heads // kv_heads
        key = np.repeat(key, repeats, axis=1)
        value = np.repeat(value, repeats, axis=1)

    # kq_scale is 1.0 in the reference vision graph: no 1/sqrt(head_dim) factor.
    scores = np.einsum("nhd,mhd->hnm", query, key)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities = probabilities / np.sum(probabilities, axis=-1, keepdims=True)
    context = np.einsum("hnm,mhd->nhd", probabilities, value)
    context = context.reshape(num_patches, heads * head_dim)
    return gemma4_vision_linear(context, layer.attn_out, clamp=layer.clamp("attn_out"))


def gemma4_vision_mlp(
    hidden: ArrayLike,
    layer: Gemma4VisionLayerWeights,
) -> np.ndarray:
    """Gated tanh-GELU MLP: ``down(gelu(gate(x)) * up(x))``."""

    array = _as_float("hidden", hidden)
    gate = gemma4_vision_gelu_tanh(
        gemma4_vision_linear(array, layer.ffn_gate, clamp=layer.clamp("ffn_gate"))
    )
    up = gemma4_vision_linear(array, layer.ffn_up, clamp=layer.clamp("ffn_up"))
    return gemma4_vision_linear(gate * up, layer.ffn_down, clamp=layer.clamp("ffn_down"))


def gemma4_vision_encoder_layer(
    hidden: ArrayLike,
    layer: Gemma4VisionLayerWeights,
    geometry: Gemma4VisionGeometry,
    pos_x: ArrayLike,
    pos_y: ArrayLike,
) -> np.ndarray:
    """One pre-norm encoder block with post-norms and residual adds."""

    array = _as_float("hidden", hidden)
    residual = array
    normed = gemma4_vision_rms_norm(array, layer.ln1, eps=geometry.rms_norm_eps)
    normed = gemma4_vision_attention(normed, layer, geometry, pos_x, pos_y)
    normed = gemma4_vision_rms_norm(
        normed, layer.attn_post_norm, eps=geometry.rms_norm_eps
    )
    array = residual + normed

    residual = array
    normed = gemma4_vision_rms_norm(array, layer.ln2, eps=geometry.rms_norm_eps)
    normed = gemma4_vision_mlp(normed, layer)
    normed = gemma4_vision_rms_norm(
        normed, layer.ffn_post_norm, eps=geometry.rms_norm_eps
    )
    array = residual + normed
    if layer.out_scale is not None:
        array = array * layer.out_scale
    return array


def gemma4_vision_pool(
    hidden: ArrayLike,
    grid_size: Sequence[int],
    kernel: int,
) -> np.ndarray:
    """Average-pool the patch grid in ``kernel x kernel`` blocks, then scale by
    ``sqrt(hidden_size)``.  Blocks are emitted row-major: y outer, x inner."""

    array = _as_float("hidden", hidden)
    if array.ndim != 2:
        raise ValueError(f"hidden must have shape (patches, hidden_size), got {array.shape}")
    grid_height, grid_width = (int(dim) for dim in grid_size)
    kernel = _positive_int("kernel", kernel)
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if grid_height * grid_width != array.shape[0]:
        raise ValueError(
            f"grid {grid_height}x{grid_width} does not cover {array.shape[0]} patches"
        )
    if grid_height % kernel or grid_width % kernel:
        raise ValueError(
            f"grid {grid_height}x{grid_width} is not divisible by the pooling kernel {kernel}"
        )
    blocks = array.reshape(
        grid_height // kernel, kernel, grid_width // kernel, kernel, array.shape[1]
    )
    pooled = blocks.mean(axis=(1, 3)).reshape(-1, array.shape[1])
    return pooled * np.sqrt(array.dtype.type(array.shape[1]))


def gemma4_vision_standardize(
    hidden: ArrayLike,
    std_bias: ArrayLike,
    std_scale: ArrayLike,
) -> np.ndarray:
    """``(hidden - std_bias) * std_scale`` per channel."""

    array = _as_float("hidden", hidden)
    bias = _as_float("std_bias", std_bias)
    scale = _as_float("std_scale", std_scale)
    for name, value in (("std_bias", bias), ("std_scale", scale)):
        if value.shape != (array.shape[-1],):
            raise ValueError(
                f"{name} expected shape ({array.shape[-1]},), got {value.shape}"
            )
    return (array - bias) * scale


# --- Full forward --------------------------------------------------------------


def gemma4_vision_forward(
    weights: Gemma4VisionWeights,
    patches: ArrayLike,
    pos_x: ArrayLike,
    pos_y: ArrayLike,
    geometry: Gemma4VisionGeometry,
    *,
    grid_size: Sequence[int] | None = None,
    patch_order: str = PATCH_ORDER_CHANNEL_LAST,
) -> Gemma4VisionForwardResult:
    """Encode one image's patches into soft tokens.

    ``grid_size`` defaults to the grid the positions imply; passing the image's
    own grid is the check that positions and patches describe the same image.
    """

    if len(weights.layers) != geometry.block_count:
        raise ValueError(
            f"expected {geometry.block_count} encoder layers, got {len(weights.layers)}"
        )
    patch_array = _as_float("patches", patches)
    if patch_array.ndim == 4:
        num_patches = patch_array.shape[0]
    elif patch_array.ndim == 2:
        num_patches = patch_array.shape[0]
    else:
        raise ValueError(
            "patches must have shape (num_patches, patch_values) or "
            f"(num_patches, patch, patch, 3), got {patch_array.shape}"
        )
    if num_patches <= 0:
        raise ValueError("patches must hold at least one patch")
    x = _positions("pos_x", pos_x, num_patches)
    y = _positions("pos_y", pos_y, num_patches)
    grid = _resolve_grid(grid_size, x, y, num_patches)

    hidden = gemma4_vision_patch_embed(
        patch_array,
        weights.patch_embd,
        weights.position_embd,
        x,
        y,
        patch_order=patch_order,
    )
    for layer in weights.layers:
        hidden = gemma4_vision_encoder_layer(hidden, layer, geometry, x, y)

    pooled = gemma4_vision_pool(hidden, grid, geometry.pooling_kernel_size)
    num_soft_tokens = pooled.shape[0]
    features = pooled
    if weights.std_bias is not None and weights.std_scale is not None:
        features = gemma4_vision_standardize(pooled, weights.std_bias, weights.std_scale)
    features = gemma4_vision_rms_norm(features, None, eps=geometry.rms_norm_eps)
    embeddings = gemma4_vision_linear(
        features, weights.input_projection, clamp=weights.input_projection_clamp
    )
    if embeddings.shape != (num_soft_tokens, geometry.output_size):
        raise ValueError(
            f"projected embeddings have shape {embeddings.shape}, expected "
            f"({num_soft_tokens}, {geometry.output_size})"
        )
    return Gemma4VisionForwardResult(
        embeddings=embeddings,
        hidden_states=features,
        pooled=pooled,
        num_soft_tokens=num_soft_tokens,
        grid_size=grid,
    )


def gemma4_vision_forward_positions(
    weights: Gemma4VisionWeights,
    patches: ArrayLike,
    positions: ArrayLike,
    geometry: Gemma4VisionGeometry,
    *,
    grid_size: Sequence[int] | None = None,
    patch_order: str = PATCH_ORDER_CHANNEL_FIRST,
) -> Gemma4VisionForwardResult:
    """Forward for an ``(num_patches, 2)`` x/y position array.

    ``positions[:, 0]`` is x, ``positions[:, 1]`` is y (the reference processor's
    layout), and patches default to the GGUF kernel's channel-first order.
    """

    array = np.asarray(positions)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            f"positions must have shape (num_patches, 2) holding (x, y), got {array.shape}"
        )
    return gemma4_vision_forward(
        weights,
        patches,
        array[:, 0],
        array[:, 1],
        geometry,
        grid_size=grid_size,
        patch_order=patch_order,
    )


def _resolve_grid(
    grid_size: Sequence[int] | None,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    num_patches: int,
) -> tuple[int, int]:
    if grid_size is None:
        grid = (int(pos_y.max()) + 1, int(pos_x.max()) + 1)
    else:
        height, width = (int(dim) for dim in grid_size)
        grid = (height, width)
    if grid[0] * grid[1] != num_patches:
        raise ValueError(
            f"grid {grid[0]}x{grid[1]} does not cover {num_patches} patches"
        )
    expected_x = np.tile(np.arange(grid[1], dtype=np.int64), grid[0])
    expected_y = np.repeat(np.arange(grid[0], dtype=np.int64), grid[1])
    if not np.array_equal(pos_x, expected_x) or not np.array_equal(pos_y, expected_y):
        raise ValueError(
            "patch positions must be a row-major grid: x cycles fastest, then y"
        )
    return grid


__all__ = [
    "PATCH_ORDER_CHANNEL_FIRST",
    "PATCH_ORDER_CHANNEL_LAST",
    "Gemma4VisionForwardResult",
    "Gemma4VisionGeometry",
    "Gemma4VisionLayerWeights",
    "Gemma4VisionWeights",
    "gemma4_vision_apply_rope",
    "gemma4_vision_attention",
    "gemma4_vision_encoder_layer",
    "gemma4_vision_forward",
    "gemma4_vision_forward_positions",
    "gemma4_vision_gelu_tanh",
    "gemma4_vision_geometry_from_config",
    "gemma4_vision_linear",
    "gemma4_vision_mlp",
    "gemma4_vision_patch_embed",
    "gemma4_vision_pool",
    "gemma4_vision_rms_norm",
    "gemma4_vision_rope_inv_freq",
    "gemma4_vision_standardize",
    "gemma4_vision_weight_shapes",
    "gemma4_vision_weights_from_slots",
]
