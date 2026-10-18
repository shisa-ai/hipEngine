"""RoPE tables in the layout the gfx1100 rotary kernel indexes.

Gemma 4 rotates with ``rotate_half`` over the full head width: element ``i``
pairs with element ``i + head_dim // 2``, and HuggingFace builds the table as
``emb = cat(freqs, freqs)`` so both halves of a pair read the same angle. Global
layers use proportional RoPE, where only the first ``rope_angles`` pairs carry a
real inverse frequency and every remaining pair gets exactly zero — which makes
its cosine one and its sine zero, leaving those elements untouched.

``qwen35_partial_rotary_kernel`` rotates pairs ``(i, i + rotary_dim // 2)`` for
``i < rotary_dim // 2`` and reads ``cos[dim]``/``sin[dim]`` by element index. So
calling it with ``rotary_dim = head_dim`` and the doubled table below reproduces
Gemma 4's rotation exactly, and no new rotary kernel is needed. The rotated
elements are not a contiguous prefix — they are ``[0, rope_angles)`` together
with ``[head_dim // 2, head_dim // 2 + rope_angles)`` — so a table truncated to
``rope_angles`` would be wrong.

The frequency math itself lives in the CPU reference, which HuggingFace's
oracle already pins. This module only changes the layout.
"""

from __future__ import annotations

import numpy as np

from hipengine.kernels.cpu_reference.gemma4 import (
    Gemma4RopeConfig,
    gemma4_rope_tables,
)

# ``rope_parameters`` types Gemma 4 uses. Sliding layers take the default
# frequency schedule over the full width; global layers take the proportional
# schedule over a quarter of it.
GEMMA4_ROPE_DEFAULT_TYPE = "default"
GEMMA4_ROPE_PROPORTIONAL_TYPE = "proportional"


def gemma4_rope_angles(
    *,
    head_dim: int,
    partial_rotary_factor: float,
    rope_type: str = GEMMA4_ROPE_PROPORTIONAL_TYPE,
) -> int:
    """Return the number of rotated pairs for one layer kind.

    This mirrors ``_compute_proportional_rope_parameters``: the rotated span
    counts pairs, and the proportional form keeps the exponent scale of
    ``head_dim`` rather than of the rotated width.
    """

    if head_dim <= 0 or head_dim % 2:
        raise ValueError("head_dim must be a positive even number")
    if rope_type == GEMMA4_ROPE_DEFAULT_TYPE:
        return head_dim // 2
    if rope_type != GEMMA4_ROPE_PROPORTIONAL_TYPE:
        raise ValueError(f"unsupported Gemma 4 rope type {rope_type!r}")
    factor = float(partial_rotary_factor)
    if not 0.0 <= factor <= 1.0:
        raise ValueError("partial_rotary_factor must be within [0, 1]")
    return int(factor * head_dim // 2)


def gemma4_rope_inverse_frequencies(
    *,
    head_dim: int,
    rope_theta: float,
    rope_angles: int,
) -> np.ndarray:
    """Return ``head_dim // 2`` inverse frequencies, zero past the rotated span."""

    return Gemma4RopeConfig(
        rope_theta=float(rope_theta),
        head_dim=int(head_dim),
        rope_angles=int(rope_angles),
    ).inverse_frequencies


def gemma4_rope_cos_sin_tables(
    rope: Gemma4RopeConfig,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cos, sin)`` of shape ``(len(positions), head_dim)``.

    The half-width tables from the reference are doubled rather than
    recomputed, so there is one definition of the frequency schedule and this
    function only decides how the kernel indexes it.
    """

    half_cos, half_sin = gemma4_rope_tables(rope, positions)
    return (
        np.concatenate((half_cos, half_cos), axis=-1).astype(np.float32),
        np.concatenate((half_sin, half_sin), axis=-1).astype(np.float32),
    )


def gemma4_rotate_split_half(
    value: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> np.ndarray:
    """Rotate ``(..., head_dim)`` with a doubled table, as the kernel does.

    This is the host-side twin of ``qwen35_partial_rotary_kernel`` with
    ``rotary_dim = head_dim``. It exists so the table layout can be checked
    against HuggingFace's ``rotate_half`` form without a GPU.
    """

    width = np.asarray(value).shape[-1]
    half = width // 2
    if cos.shape[-1] != width or sin.shape[-1] != width:
        raise ValueError("cos/sin must carry one entry per rotated element")
    first = value[..., :half]
    second = value[..., half:]
    out_first = first * cos[..., :half] - second * sin[..., :half]
    out_second = second * cos[..., half:] + first * sin[..., half:]
    return np.concatenate((out_first, out_second), axis=-1).astype(np.float32)


__all__ = [
    "GEMMA4_ROPE_DEFAULT_TYPE",
    "GEMMA4_ROPE_PROPORTIONAL_TYPE",
    "gemma4_rope_angles",
    "gemma4_rope_cos_sin_tables",
    "gemma4_rope_inverse_frequencies",
    "gemma4_rotate_split_half",
]
