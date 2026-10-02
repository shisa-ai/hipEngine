"""Raw-pointer wrapper for the BF16 WMMA Gemma 4 full-layer prefill kernel.

This is the head_dim-512 half of the Gemma 4 attention family: the five full
layers (indices 5, 11, 17, 23, 29 of 30), which are 16 query heads against 2 KV
heads with no sliding window. The sliding layers are head_dim 256 with a GQA
ratio of 2 and are served by
:mod:`hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_prefill_wmma`; the
two kernels share a calling convention and nothing else.

The kernel's ABI is the strict kernel's, argument for argument, so the two can
be driven on the same device buffers and compared without an adapter:

* ``query`` is ``(tokens, num_heads, head_dim)`` BF16, ``key`` and ``value`` are
  ``(keys, num_kv_heads, head_dim)`` BF16, ``keep_mask`` is a ``(tokens, keys)``
  uint8 **keep** mask, and ``out`` is ``(tokens, num_heads, head_dim)`` BF16.
* ``keys`` is the mask's column count and column ``j`` is the caller's key
  ``key_begin + j``; ``row_offset`` is the first query row's position in that
  same frame, so row ``t`` sits at ``row_offset + t``. ``window > 0`` is the
  promise that the mask is zero outside ``[row_offset + tile - window + 1,
  row_offset + tile + QUERY_ROWS)``, which is what licenses trimming the walk.
  A full layer passes ``window = start + rows - key_begin`` -- the whole context
  -- so the leading bound is inactive and the trailing one is the causal trim.
  ``window == 0`` promises nothing and the walk covers every column.
* ``scale`` is applied to the FP32 dot, as the strict kernel applies it.

Geometry is fixed at the full layers' shape: head_dim 512, GQA ratio 8. The
launcher raises for anything else rather than running a different geometry.

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

_SOURCE = Path(__file__).with_name("gemma4_attention_prefill_wmma_full.hip")
_OUTPUT_NAME = "gemma4_attention_prefill_wmma_full.so"

SYMBOL_PREFILL_WMMA_FULL_BF16 = "hipengine_gemma4_attention_prefill_wmma_full_bf16"

# The kernel's launch geometry, mirrored from the .hip so the launcher can name
# a geometry miss before a launch rather than after one.
QUERY_ROWS = 16
GQA_HEADS = 2
GQA_RATIO = 8
HEAD_DIM = 512
K_BATCH = 16
DIM_GROUPS = 2
THREADS = 128
COLUMNS = QUERY_ROWS * GQA_HEADS

# How many blocks a single-query-tile launch aims for before it stops dividing
# the key walk. Measured at 262,144 keys on gfx1151: 8 blocks reach 75 GB/s, 64
# reach 225, 256 reach 252, so the rate is found somewhere between 8 and 64 and
# there is nothing to gain by asking for more than this.
TARGET_BLOCKS = 128


def plan_gemma4_attention_wmma_full_slices(
    *, tokens: int, keys: int, num_heads: int, num_kv_heads: int
) -> int:
    """How many key slices the WMMA full-layer walk should be divided into.

    The unsplit grid is ``query_tiles * num_kv_heads * gqa_tiles``. At this
    geometry a one-tile query block -- a decode step -- presents 8 blocks
    against the 64 or more the hardware needs to reach its own memory rate, and
    each of those blocks walks every key, so the kernel waits on itself rather
    than on memory. Dividing the walk gives it something to overlap and the
    per-slice softmax state is combined afterwards.

    Only the single-query-tile case is divided. A prefill block of 512 rows
    already presents 256 blocks, so splitting it would reassociate its softmax
    for no gain; leaving it whole also keeps every prefill shape on exactly the
    arithmetic it has always run. The division is also capped by the walk
    needing at least one K batch per slice, since a slice shorter than a tile
    has no work to overlap.
    """

    query_rows = QUERY_ROWS
    query_tiles = (int(tokens) + query_rows - 1) // query_rows
    gqa_tiles = (int(num_heads) // int(num_kv_heads) + GQA_HEADS - 1) // GQA_HEADS
    base_blocks = query_tiles * int(num_kv_heads) * gqa_tiles
    if query_tiles != 1 or base_blocks >= TARGET_BLOCKS:
        return 1
    slices = (TARGET_BLOCKS + base_blocks - 1) // base_blocks
    by_walk = int(keys) // K_BATCH
    if slices > by_walk:
        slices = by_walk
    return max(1, slices)


def plan_gemma4_attention_wmma_full_scratch_bytes(slices: int, *, num_heads: int, num_kv_heads: int, head_dim: int) -> int:
    """Bytes the split workspace needs for `slices` slices at this geometry.

    One float per (column, output dimension) of accumulator, plus two floats of
    softmax state per column, for every slice of every (kv head, GQA tile)
    plane.
    """

    gqa_tiles = (int(num_heads) // int(num_kv_heads) + GQA_HEADS - 1) // GQA_HEADS
    planes = int(num_kv_heads) * gqa_tiles
    columns = QUERY_ROWS * GQA_HEADS
    return int(slices) * planes * (columns * int(head_dim) + columns * 2) * 4

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
    ctypes.c_void_p,
    ctypes.c_int64,
)


def plan_gemma4_attention_prefill_wmma_full_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gemma4_attention_prefill_wmma_full",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gemma4_attention_prefill_wmma_full(
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
        family="gemma4_attention_prefill_wmma_full",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gemma4_attention_prefill_wmma_full_supported(
    *, num_heads: int, num_kv_heads: int, head_dim: int
) -> bool:
    """Whether the WMMA full-layer kernel implements this head geometry.

    Head_dim 512 with GQA ratio 8 is what the full layers of Gemma 4 26B-A4B
    use. A different ratio or a different head width is a capability miss, not a
    slow path.
    """

    if head_dim != HEAD_DIM or num_kv_heads <= 0 or num_heads <= 0:
        return False
    if num_heads % num_kv_heads:
        return False
    return num_heads // num_kv_heads == GQA_RATIO


def gemma4_attention_prefill_wmma_full_bf16(
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
    """Launch the WMMA full-layer prefill kernel. Signature matches the strict kernel.

    ``tokens`` is the query block's row count and ``keys`` the mask's column
    count, which may exceed ``tokens`` for a chunked prefill. See the module
    docstring for the mask and offset frame.

    ``scratch`` is the layer's attention workspace. The strict kernel needs one
    because it materialises logits; this one stages its K/V tile in LDS instead
    and runs without it. What it uses scratch for is the key split: a decode
    step presents one query tile against the whole context, so the unsplit grid
    is 8 blocks where the hardware needs about 128 to reach its own memory rate,
    and the walk is divided and recombined when there is somewhere to put the
    per-slice softmax state.

    No scratch therefore means the walk is taken whole. That is the correct
    answer either way and it is what a caller with no workspace is asking for --
    the layer always has one, so a production decode always splits, and the
    unsplit path stays reachable for the tests that compare the two.
    """

    tokens = int(tokens)
    num_heads = int(num_heads)
    num_kv_heads = int(num_kv_heads)
    head_dim = int(head_dim)
    key_count = tokens if keys is None else int(keys)
    if tokens <= 0 or key_count <= 0:
        raise ValueError("tokens and keys must be positive")
    if not gemma4_attention_prefill_wmma_full_supported(
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    ):
        raise NotImplementedError(
            f"gemma4 WMMA full-layer prefill implements head_dim={HEAD_DIM} with a GQA "
            f"ratio of {GQA_RATIO}; got head_dim={head_dim}, num_heads={num_heads}, "
            f"num_kv_heads={num_kv_heads}"
        )

    library = library or build_gemma4_attention_prefill_wmma_full(load=True)
    runtime = runtime or get_hip_runtime()
    # The split is planned here rather than in the launcher so the decision is a
    # pure function of the geometry and can be tested without a device.
    slices = plan_gemma4_attention_wmma_full_slices(
        tokens=tokens, keys=key_count, num_heads=num_heads, num_kv_heads=num_kv_heads
    )
    scratch_ptr = 0
    if slices > 1 and scratch is None:
        slices = 1
    if slices > 1:
        nbytes = plan_gemma4_attention_wmma_full_scratch_bytes(
            slices, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        scratch_ptr = scratch.buffer(
            nbytes, stream=int(stream), runtime=runtime
        ).ptr
    fn = signed_kernel_fn(library, SYMBOL_PREFILL_WMMA_FULL_BF16, _ARGTYPES, ctypes.c_int)
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
        scratch_ptr,
        slices,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_gemma4_attention_prefill_wmma_full_kernels(*, replace: bool = False) -> None:
    """Register the full-layer variant under its own variant name.

    Registration is additive: the variant string differs from the strict
    kernel's ``gemma4_plain`` and from the sliding candidate's
    ``gemma4_wmma_flash``, so no existing resolution can pick this up by
    accident. It exists so the kernel has a four-axis identity when the profile
    selects it.
    """

    for quant in ("gguf_q4_k_m", "gguf_q4_k_xl", "gguf_q8_0", "gguf"):
        register(
            KernelKey("hip_gfx1100", "prefill_attention", quant, "gemma4_wmma_flash_full"),
            gemma4_attention_prefill_wmma_full_bf16,
            replace=replace,
        )
