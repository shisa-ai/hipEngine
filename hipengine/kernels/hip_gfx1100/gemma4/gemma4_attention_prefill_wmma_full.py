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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gemma4_attention_prefill_wmma_full.hip")
_OUTPUT_NAME = "gemma4_attention_prefill_wmma_full.so"
# The decode shape is the same source built with a different query block, so it
# is a second object rather than a second kernel. See the .hip for why the pair
# is 16/2 for prefill and 2/8 for decode.
_SOURCE_DECODE = Path(__file__).with_name("gemma4_attention_decode_wmma_full.hip")
_OUTPUT_NAME_DECODE = "gemma4_attention_decode_wmma_full.so"

SYMBOL_PREFILL_WMMA_FULL_BF16 = "hipengine_gemma4_attention_prefill_wmma_full_bf16"
SYMBOL_DECODE_WMMA_FULL_BF16 = "hipengine_gemma4_attention_decode_wmma_full_bf16"

# The kernel's launch geometry, mirrored from the .hip so the launcher can name
# a geometry miss before a launch rather than after one.
QUERY_ROWS = 16
GQA_HEADS = 2
DECODE_QUERY_ROWS = 2
DECODE_GQA_HEADS = 8
GQA_RATIO = 8
HEAD_DIM = 512
K_BATCH = 16
# How the head dimension is split across waves. `waves` is `column_groups *
# dim_groups`, so this is what sets the thread count -- and the thread count is
# what sets how many loads the staging loop keeps in flight. It is per shape
# because the two shapes have different column groups and want the same threads.
DIM_GROUPS = 4
DECODE_DIM_GROUPS = 4
THREADS = 128
COLUMNS = QUERY_ROWS * GQA_HEADS

# How many blocks a single-query-tile launch aims for before it stops dividing
# the key walk. Measured at 262,144 keys on gfx1151: 8 blocks reach 75 GB/s, 64
# reach 225, 256 reach 252, so the rate is found somewhere between 8 and 64 and
# there is nothing to gain by asking for more than this.
#
# This is the *prefill* shape's number, and the decode shape does not share it.
# The two shapes read different amounts per block -- the prefill shape re-reads
# each KV band once per GQA tile, the decode shape reads each band once -- so a
# block count that suits one does not suit the other, and a shape whose blocks
# each move a quarter of the traffic wants a quarter of the blocks, not four
# times the slices to reach the same grid. See ``split_target_blocks`` on each
# shape for the two measurements.
TARGET_BLOCKS = 128

# The decode shape's own target, measured by sweeping the slice count directly
# (the planner monkeypatched, one full layer, tokens=1, median of 9, gfx1151).
# The walk is 1.074 GB at 262,144 keys, against the 238.5 GB/s roofline the
# roofline kernel measured on this device:
#
#   keys     slices=8   slices=16   slices=32   slices=64   slices=128
#   16,384     36.5%       51.3%       39.7%       32.5%       22.1%
#   65,536     41.9%       66.4%       55.7%       57.4%       49.2%
#   131,072    42.8%       73.0%       60.8%       63.3%       60.7%
#   262,144    43.1%       74.7%       61.9%       70.1%       70.4%
#
# Sixteen slices is the fastest at every key count, by 1.066x at 262,144 and
# 1.578x at 16,384. More is worse, not better: the walk is a fixed amount of
# work, so past the point where the blocks cover the device, extra slices only
# add wave quantization and a longer combine. The previous value of 64 came from
# applying the prefill shape's block target to a shape with a quarter of the
# traffic per block.
DECODE_SPLIT_TARGET_BLOCKS = 32


@dataclass(frozen=True)
class WmmaFullShape:
    """One query-block shape of the full-layer kernel, and how to launch it.

    ``query_rows`` and ``gqa_heads`` are the two numbers the .hip is built
    with; everything else follows. They are a property of the build, not of a
    call, so a caller cannot ask for a shape the loaded object does not contain
    and the LDS and thread counts below are the ones that object was compiled
    with rather than a model of them.
    """

    query_rows: int
    gqa_heads: int
    dim_groups: int
    symbol: str
    output_name: str
    split_target_blocks: int

    @property
    def columns(self) -> int:
        return self.query_rows * self.gqa_heads

    @property
    def column_groups(self) -> int:
        return self.columns // 16

    @property
    def waves(self) -> int:
        return self.column_groups * self.dim_groups

    @property
    def threads(self) -> int:
        return 32 * self.waves

    @property
    def gqa_tiles(self) -> int:
        return (GQA_RATIO + self.gqa_heads - 1) // self.gqa_heads

    @property
    def shared_bytes(self) -> int:
        """Q tile, K/V tile, and the per-dimension-group partial score tiles.

        `(kColumns * kQStride + kKBatch * kKvStride) * sizeof(bf16_t)` plus
        `kDimGroups * kColumnGroups * 32 * kKeyTiles * 8 * sizeof(float)` for
        the partials the dimension groups sum through.
        """

        return (
            (self.columns * (HEAD_DIM + 8) + K_BATCH * (HEAD_DIM + 8)) * 2
            + self.dim_groups * (self.columns // 16) * 32 * (K_BATCH // 16) * 8 * 4
        )


PREFILL_SHAPE = WmmaFullShape(
    QUERY_ROWS,
    GQA_HEADS,
    DIM_GROUPS,
    SYMBOL_PREFILL_WMMA_FULL_BF16,
    _OUTPUT_NAME,
    TARGET_BLOCKS,
)
DECODE_SHAPE = WmmaFullShape(
    DECODE_QUERY_ROWS,
    DECODE_GQA_HEADS,
    DECODE_DIM_GROUPS,
    SYMBOL_DECODE_WMMA_FULL_BF16,
    _OUTPUT_NAME_DECODE,
    DECODE_SPLIT_TARGET_BLOCKS,
)


def shape_for_tokens(tokens: int) -> WmmaFullShape:
    """Which shape a query block of ``tokens`` rows is launched with.

    A decode is one query row against the whole context, which is the case the
    decode shape exists for; anything wider is a prefill block, which already
    presents enough blocks that its 4x KV re-read is not what it waits on and
    which must keep the arithmetic it has always run. Two rows is the decode
    shape's block width, so a block that fits in it takes it.
    """

    return DECODE_SHAPE if int(tokens) <= DECODE_SHAPE.query_rows else PREFILL_SHAPE


def plan_gemma4_attention_wmma_full_slices(
    *, tokens: int, keys: int, num_heads: int, num_kv_heads: int
) -> int:
    """How many key slices the WMMA full-layer walk should be divided into.

    The unsplit grid is ``query_tiles * num_kv_heads * gqa_tiles``. A one-tile
    query block -- a decode step -- presents few blocks against the 64 or more
    the hardware needs to reach its own memory rate, and each of those blocks
    walks every key, so the kernel waits on itself rather than on memory.
    Dividing the walk gives it something to overlap and the per-slice softmax
    state is combined afterwards.

    Only the single-query-tile case is divided. A prefill block of 512 rows
    already presents 256 blocks, so splitting it would reassociate its softmax
    for no gain; leaving it whole also keeps every prefill shape on exactly the
    arithmetic it has always run. The division is also capped by the walk
    needing at least one K batch per slice, since a slice shorter than a tile
    has no work to overlap.

    The shape matters to the count as well as to the kernel: the two shapes
    move different amounts of traffic per block, because the prefill shape
    re-reads each KV band once per GQA tile and the decode shape reads each band
    once. Each shape therefore carries its own ``split_target_blocks``, and a
    shape that moves a quarter of the traffic per block wants a quarter of the
    blocks -- not four times the slices to reach the same grid, which is what
    this rule used to do and which measured 1.066x to 1.578x slower.
    """

    shape = shape_for_tokens(tokens)
    query_tiles = (int(tokens) + shape.query_rows - 1) // shape.query_rows
    base_blocks = query_tiles * int(num_kv_heads) * shape.gqa_tiles
    target = shape.split_target_blocks
    if query_tiles != 1 or base_blocks >= target:
        return 1
    slices = (target + base_blocks - 1) // base_blocks
    by_walk = int(keys) // K_BATCH
    if slices > by_walk:
        slices = by_walk
    return max(1, slices)


def plan_gemma4_attention_wmma_full_scratch_bytes(
    slices: int, *, tokens: int, num_heads: int, num_kv_heads: int, head_dim: int
) -> int:
    """Bytes the split workspace needs for `slices` slices at this geometry.

    One float per (column, output dimension) of accumulator, plus two floats of
    softmax state per column, for every slice of every (kv head, GQA tile)
    plane. The column count is the shape's, so the decode shape's workspace is
    half the prefill's at the same slice count.
    """

    shape = shape_for_tokens(tokens)
    planes = int(num_kv_heads) * shape.gqa_tiles
    return int(slices) * planes * (shape.columns * int(head_dim) + shape.columns * 2) * 4

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


def build_gemma4_attention_decode_wmma_full(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    """Build the decode query-block shape of the same kernel source.

    A separate object rather than a separate kernel: the .hip is included with
    the decode shape's macros defined. It is a separate cache entry for the
    same reason, so a device that has only ever run a prefill has not built it
    and one that has only ever run a decode has not built the prefill object.
    """

    return build_hip(
        sources=[_SOURCE_DECODE],
        family="gemma4_attention_decode_wmma_full",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        include_dirs=[_SOURCE.parent],
        output_name=_OUTPUT_NAME_DECODE,
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

    The query block's width also picks the shape, and so the compiled object:
    see :func:`shape_for_tokens`. A one-row block is a decode step, which is
    short of blocks and re-reads each KV band once per GQA tile, so it runs the
    shape that reads each band once.

    ``scratch`` is the layer's attention workspace. The strict kernel needs one
    because it materialises logits; this one stages its K/V tile in LDS instead
    and runs without it. What it uses scratch for is the key split: a decode
    step presents one query tile against the whole context, so the unsplit grid
    is a handful of blocks where the hardware needs about 128 to reach its own
    memory rate, and the walk is divided and recombined when there is somewhere
    to put the per-slice softmax state.

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

    shape = shape_for_tokens(tokens)
    if library is None:
        library = (
            build_gemma4_attention_decode_wmma_full(load=True)
            if shape is DECODE_SHAPE
            else build_gemma4_attention_prefill_wmma_full(load=True)
        )
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
            slices,
            tokens=tokens,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        scratch_ptr = scratch.buffer(
            nbytes, stream=int(stream), runtime=runtime
        ).ptr
    fn = signed_kernel_fn(library, shape.symbol, _ARGTYPES, ctypes.c_int)
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
