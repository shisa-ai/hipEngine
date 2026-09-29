"""Tiled flash prefill attention for the Gemma 4 global layers (head_dim 512).

This is the llama.cpp ``fattn-tile`` kernel behind a hipEngine-shaped entry
point. It is a *different association* from :mod:`gemma4_attention`: an online
softmax over key tiles instead of the exact kernel's three passes (max,
denominator, weighted sum), and its device body comes from upstream rather than
being written here. Two consequences follow and neither is optional:

* It is compared numerically, not bitwise, against the exact kernel -- see
  ``tests/test_gpu_gemma4_attention_tiled_parity.py``.
* Importing this module registers nothing and builds nothing. Dispatch
  registration is a separate, deliberate step; this module only exposes the
  launch so the path can be exercised directly.

The entry point takes the same twelve arguments as
:func:`gemma4_attention_prefill_bf16` in the same order, so a caller that
already has the exact kernel's buffer layout can switch symbol and nothing else.
The dtype staging (bf16 -> f32/f16, u8 -> f16, f32 -> bf16) happens inside the
HIP source, which is why no staging argument appears here.

Geometry this kernel can execute, each enforced by a return code in the HIP
source and re-checked here so the failure is a named Python error rather than a
silent mislaunch:

* ``head_dim == 512`` -- the template is specialized ``DKQ = DV = 512``.
* ``keys`` a multiple of 128 -- ``ncols2 > 1`` selects the KV loop branch that
  has no out-of-bounds check, so a partial final tile would read past K/V/mask.
* ``num_heads`` a multiple of 8, and ``num_heads / num_kv_heads`` a multiple of
  8 -- the grid's z decoding only agrees with the launch when ``ncols2``
  divides the GQA ratio.

Anything else raises :class:`Gemma4AttentionTiledUnsupported` naming the
constraint. That is a capability statement, not a quality ranking: shapes the
kernel cannot execute are refused loudly, and shapes it can execute run.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HipRuntime, get_hip_runtime

_SOURCE = Path(__file__).with_name("gemma4_attention_tiled.hip")
_OUTPUT_NAME = "gemma4_attention_tiled.so"
_SYMBOL_PREFILL_BF16 = "hipengine_gemma4_attention_tiled_prefill_bf16"

# Same twelve-argument form and order as gemma4_attention._ARGTYPES_PREFILL, so
# the two launchers are interchangeable at the call site.
_ARGTYPES_PREFILL = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_void_p,
    ctypes.c_int64,
)

# Return codes from the HIP entry point, mirrored so the Python error can name
# the constraint instead of reporting a bare integer.
_RETURN_CODES = {
    1: "head_dim must be 512 (the kernel template is specialized DKQ = DV = 512)",
    2: "tokens and keys must be positive",
    3: "keys must be a multiple of 128 (the KV loop branch taken at ncols2 > 1 "
       "has no out-of-bounds check; pad the key range)",
    4: "num_heads must be a multiple of 8 (the head tile width)",
    5: "num_heads must be a multiple of num_kv_heads",
    6: "num_heads / num_kv_heads must be a multiple of 8 (the grid z decoding "
       "only agrees with the launch when the head tile divides the GQA ratio)",
    7: "staging allocation failed",
    8: "kernel launch failed",
    9: "output conversion failed",
}

# The geometry the kernel is specialized for. Not a support list: it documents
# the head_dim the template is compiled for, and every other dimension is
# checked arithmetically by the rules above.
SUPPORTED_HEAD_DIM = 512
HEAD_TILE = 8
KEY_TILE = 128


class Gemma4AttentionTiledUnsupported(Exception):
    """The requested shape is outside what this kernel's template can execute.

    Raised before launch, naming the constraint. Callers that want a different
    shape route to another kernel; they do not get a partial or padded result.
    """


def plan_gemma4_attention_tiled_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gemma4_attention_tiled",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gemma4_attention_tiled(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family="gemma4_attention_tiled",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gemma4_attention_tiled_admits(
    *,
    tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    keys: int,
) -> None:
    """Raise :class:`Gemma4AttentionTiledUnsupported` if the shape cannot run.

    Mirrors the HIP entry point's return codes so a caller can ask *before*
    allocating buffers. A shape that passes here is a shape the kernel executes;
    nothing about qualification or measurement enters into it.
    """

    for name, value in (
        ("tokens", tokens),
        ("num_heads", num_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
        ("keys", keys),
    ):
        if int(value) <= 0:
            raise Gemma4AttentionTiledUnsupported(
                f"{name} must be positive, got {value!r}"
            )
    if int(head_dim) != SUPPORTED_HEAD_DIM:
        raise Gemma4AttentionTiledUnsupported(_RETURN_CODES[1])
    if int(keys) % KEY_TILE:
        raise Gemma4AttentionTiledUnsupported(_RETURN_CODES[3])
    if int(num_heads) < HEAD_TILE or int(num_heads) % HEAD_TILE:
        raise Gemma4AttentionTiledUnsupported(_RETURN_CODES[4])
    if int(num_heads) % int(num_kv_heads):
        raise Gemma4AttentionTiledUnsupported(_RETURN_CODES[5])
    if (int(num_heads) // int(num_kv_heads)) % HEAD_TILE:
        raise Gemma4AttentionTiledUnsupported(_RETURN_CODES[6])


def gemma4_attention_prefill_tiled(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    keep_mask_ptr: int,
    out_ptr: int,
    *,
    tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    keys: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch the tiled flash kernel over the same tensors as the exact kernel.

    ``query``/``key``/``value``/``out`` are bf16 and ``keep_mask`` is uint8 with
    1 meaning "attend", all in hipEngine's row-major
    ``(tokens, num_heads, head_dim)`` / ``(keys, num_kv_heads, head_dim)`` /
    ``(tokens, keys)`` layout -- identical to
    :func:`gemma4_attention_prefill_bf16`. The kernel is handed the caller's
    mask rather than deriving causality, so a sliding-window or prefix mask
    behaves exactly as it does on the exact path.

    Raises :class:`Gemma4AttentionTiledUnsupported` before launch when the shape
    is outside the template's capability.
    """

    key_count = tokens if keys is None else int(keys)
    _check_prefill_shape(
        tokens=tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        keys=key_count,
    )
    library = library or build_gemma4_attention_tiled(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_PREFILL_BF16, _ARGTYPES_PREFILL, ctypes.c_int)
    err = fn(
        query_ptr,
        key_ptr,
        value_ptr,
        keep_mask_ptr,
        out_ptr,
        int(tokens),
        int(num_heads),
        int(num_kv_heads),
        int(head_dim),
        ctypes.c_float(scale),
        ctypes.c_void_p(stream),
        int(key_count),
    )
    if int(err) != 0:
        detail = _RETURN_CODES.get(int(err), "unknown error")
        raise Gemma4AttentionTiledUnsupported(
            f"tiled prefill launch returned {int(err)}: {detail}"
        )


def _check_prefill_shape(
    *,
    tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    keys: int,
) -> None:
    gemma4_attention_tiled_admits(
        tokens=tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        keys=keys,
    )


__all__ = [
    "HEAD_TILE",
    "KEY_TILE",
    "SUPPORTED_HEAD_DIM",
    "Gemma4AttentionTiledUnsupported",
    "build_gemma4_attention_tiled",
    "gemma4_attention_prefill_tiled",
    "gemma4_attention_tiled_admits",
    "plan_gemma4_attention_tiled_build",
]