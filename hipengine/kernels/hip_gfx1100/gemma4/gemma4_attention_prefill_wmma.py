"""Raw-pointer wrapper for the BF16 WMMA Gemma 4 sliding-window prefill kernel.

This is a candidate, not a replacement. The strict
:func:`hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention.gemma4_attention_prefill_bf16`
kernel stays the default and nothing here is selected by
``gemma4_attention.attention_symbol``, the layer wrapper, or the runtime; the
only callers are the benchmark row and the correctness test. Promotion needs the
execution-profile gate for a changed-arithmetic path, which has not been run.

The kernel's ABI is the strict kernel's, argument for argument, so the two can
be driven on the same device buffers and compared without an adapter:

* ``query`` is ``(tokens, num_heads, head_dim)`` BF16, ``key`` and ``value`` are
  ``(keys, num_kv_heads, head_dim)`` BF16, ``keep_mask`` is a ``(tokens, keys)``
  uint8 **keep** mask, and ``out`` is ``(tokens, num_heads, head_dim)`` BF16.
* ``keys`` is the mask's column count and column ``j`` is the caller's key
  ``key_begin + j``; ``row_offset`` is the first query row's position in that
  same frame, so row ``t`` sits at ``row_offset + t``. ``window > 0`` is the
  promise that the mask is sliding-causal, which is what licenses trimming the
  walk to ``[row_offset + tile - window + 1, row_offset + tile + QUERY_ROWS)``.
  ``window == 0`` promises nothing and the walk covers every column.
* ``scale`` is applied to the FP32 dot, as the strict kernel applies it.

Geometry is fixed at the sliding-window layers' shape: head_dim 256, GQA ratio
2. The launcher raises for anything else rather than running a different
geometry, and the head_dim-512 full layers are a separate unit.

Importing this module registers a ctypes launch wrapper but does not build or
load ROCm until the wrapper is called.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gemma4_attention_prefill_wmma.hip")
_OUTPUT_NAME = "gemma4_attention_prefill_wmma.so"

SYMBOL_PREFILL_WMMA_BF16 = "hipengine_gemma4_attention_prefill_wmma_bf16"

# The kernel's launch geometry, mirrored from the .hip so the launcher can name
# a geometry miss before a launch rather than after one.
QUERY_ROWS = 16
GQA_HEADS = 2
HEAD_DIM = 256
K_BATCH = 32
THREADS = 64

# Identical to the strict kernel's prefill signature, including argument order.
_ARGTYPES = (
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
    ctypes.c_int64,
    ctypes.c_int64,
)


def plan_gemma4_attention_prefill_wmma_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gemma4_attention_prefill_wmma",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gemma4_attention_prefill_wmma(
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
        family="gemma4_attention_prefill_wmma",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gemma4_attention_prefill_wmma_supported(
    *, num_heads: int, num_kv_heads: int, head_dim: int
) -> bool:
    """Whether the WMMA kernel implements this head geometry.

    Head_dim 256 with GQA ratio 2 is what the sliding layers of Gemma 4
    26B-A4B use. A different ratio is a capability miss, not a slow path.
    """

    if head_dim != HEAD_DIM or num_kv_heads <= 0 or num_heads <= 0:
        return False
    if num_heads % num_kv_heads:
        return False
    return num_heads // num_kv_heads == GQA_HEADS


def gemma4_attention_prefill_wmma_bf16(
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
    window: int = 0,
    row_offset: int = 0,
    scratch: Any = None,
) -> None:
    """Launch the WMMA prefill candidate. Signature matches the strict kernel.

    ``tokens`` is the query block's row count and ``keys`` the mask's column
    count, which may exceed ``tokens`` for a chunked prefill. See the module
    docstring for the mask and offset frame.

    ``scratch`` is accepted and ignored: the strict kernel needs a caller-owned
    scratch buffer because it materialises logits, and this one stages its K/V
    tile in LDS instead. It is in the signature so the two launchers are
    interchangeable at a call site that passes the strict kernel's arguments.
    """

    tokens = int(tokens)
    num_heads = int(num_heads)
    num_kv_heads = int(num_kv_heads)
    head_dim = int(head_dim)
    key_count = tokens if keys is None else int(keys)
    if tokens <= 0 or key_count <= 0:
        raise ValueError("tokens and keys must be positive")
    if not gemma4_attention_prefill_wmma_supported(
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    ):
        raise NotImplementedError(
            f"gemma4 WMMA prefill implements head_dim={HEAD_DIM} with a GQA ratio of "
            f"{GQA_HEADS}; got head_dim={head_dim}, num_heads={num_heads}, "
            f"num_kv_heads={num_kv_heads}"
        )

    library = library or build_gemma4_attention_prefill_wmma(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, SYMBOL_PREFILL_WMMA_BF16, _ARGTYPES, ctypes.c_int)
    err = fn(
        query_ptr,
        key_ptr,
        value_ptr,
        keep_mask_ptr,
        out_ptr,
        tokens,
        num_heads,
        num_kv_heads,
        head_dim,
        ctypes.c_float(scale),
        stream,
        key_count,
        window,
        row_offset,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_gemma4_attention_prefill_wmma_kernels(*, replace: bool = False) -> None:
    """Register the candidate under its own variant name.

    Registration is additive: the variant string differs from the strict
    kernel's ``gemma4_plain``, so no existing resolution can pick this up. It
    exists so the kernel has a four-axis identity when it is promoted, not to
    route anything to it.
    """

    for quant in ("gguf_q4_k_m", "gguf_q4_k_xl", "gguf_q8_0", "gguf"):
        register(
            KernelKey("hip_gfx1100", "prefill_attention", quant, "gemma4_wmma_flash"),
            gemma4_attention_prefill_wmma_bf16,
            replace=replace,
        )
