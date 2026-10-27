"""Tiled causal flash prefill attention for the dense Llama-family path.

The exact block kernel (:mod:`hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention`)
stages one FP32 logit row per key in workgroup LDS and runs one block per
(token, head). Its shared memory is therefore sized by the context -- which is
what capped a prompt at 15856 tokens -- and it re-reads the whole K/V row for
every query row, which is why a long prompt cost hours.

This kernel tiles query rows and keys instead. One block owns 8 or 16 query rows
of one head, stages a 64-key K/V slice in LDS, keeps an 8x64 or 16x64 score tile,
and folds each slice in with an online softmax. Shared memory does not mention
``keys``, and every K/V slice serves every row in the block.

Geometry this kernel executes, each a return code in the HIP source re-checked
here so the failure is a named Python error:

* ``head_dim == 128`` -- the register tile is specialized for it.
* ``num_heads % num_kv_heads == 0`` -- grouped-query attention, any ratio.
* ``keys >= tokens`` -- rows occupy the last ``tokens`` positions of the key
  frame, which is the contiguous-prompt case.

The mask must be causal in that frame: no key past a row's own position
contributes. That is the contract of
:func:`~hipengine.kernels.hip_gfx1100.attention.dense_prefill_attention.dense_causal_prefill_bf16`
and of :func:`causal_keep_mask`, and the walk is bounded by it.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HipRuntime, get_hip_runtime

_SOURCE = Path(__file__).with_name("dense_flash_prefill.hip")
_OUTPUT_NAME = "dense_flash_prefill.so"
_SYMBOL = "hipengine_dense_flash_prefill_bf16"

SUPPORTED_HEAD_DIM = 128
KEY_TILE = 64

_RETURN_CODES = {
    1: "tokens, keys, heads, and head_dim must be positive with keys >= tokens",
    2: "num_heads must be a multiple of num_kv_heads",
    3: f"head_dim must be {SUPPORTED_HEAD_DIM} (the register tile is specialized for it)",
    4: "kernel launch failed",
}


class DenseFlashPrefillUnsupported(Exception):
    """The dense flash prefill kernel cannot execute this shape."""


def plan_dense_flash_prefill_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="dense_flash_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_dense_flash_prefill(
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
        family="dense_flash_prefill",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def dense_flash_prefill_admits(
    *,
    tokens: int,
    keys: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> str | None:
    """Whether this kernel serves the shape; the reason if not."""

    parsed = (int(tokens), int(keys), int(num_heads), int(num_kv_heads), int(head_dim))
    if min(parsed) <= 0 or int(keys) < int(tokens):
        return _RETURN_CODES[1]
    if int(num_heads) % int(num_kv_heads):
        return _RETURN_CODES[2]
    if int(head_dim) != SUPPORTED_HEAD_DIM:
        return _RETURN_CODES[3]
    return None


def dense_flash_prefill_bf16(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    keep_mask_ptr: int,
    out_ptr: int,
    *,
    tokens: int,
    keys: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run tiled causal prefill attention over one contiguous prompt."""

    miss = dense_flash_prefill_admits(
        tokens=tokens,
        keys=keys,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    if miss is not None:
        raise DenseFlashPrefillUnsupported(miss)
    for name, pointer in (
        ("query_ptr", query_ptr),
        ("key_ptr", key_ptr),
        ("value_ptr", value_ptr),
        ("out_ptr", out_ptr),
    ):
        if int(pointer) <= 0:
            raise ValueError(f"{name} must be non-zero")
    # A zero mask pointer means the causal mask, which the walk already
    # enforces. Anything else must be a real (tokens, keys) uint8 mask.
    if int(keep_mask_ptr) < 0:
        raise ValueError("keep_mask_ptr must be non-negative")

    library = library or build_dense_flash_prefill(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL)
    fn.argtypes = (
        [ctypes.c_void_p] * 5
        + [ctypes.c_int64] * 5
        + [ctypes.c_float, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    code = fn(
        int(query_ptr),
        int(key_ptr),
        int(value_ptr),
        int(keep_mask_ptr),
        int(out_ptr),
        int(tokens),
        int(keys),
        int(num_heads),
        int(num_kv_heads),
        int(head_dim),
        float(scale),
        int(stream),
    )
    if code != 0:
        raise DenseFlashPrefillUnsupported(
            f"dense flash prefill refused the launch: {_RETURN_CODES.get(code, code)}"
        )
