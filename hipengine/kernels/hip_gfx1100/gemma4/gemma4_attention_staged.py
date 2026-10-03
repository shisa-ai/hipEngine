"""Raw-pointer wrapper for the staged Gemma 4 attention candidate on gfx1100.

The strict family (:mod:`hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention`)
holds one FP32 logit per live key in shared memory, so its footprint grows with
the context and its launch refuses above 15,616 keys at head_dim 512. This
candidate computes the same arithmetic in three kernels over a caller-owned
global workspace, with a shared footprint that is a constant 8 KB and does not
mention ``keys``.

The arithmetic is the strict family's, not an approximation of it, and
``tests/test_gpu_gemma4_attention_staged.py`` asserts the outputs match the
shipped strict wrapper bitwise on identical device buffers.
``gemma4_attention_staged.hip`` states the three stages, the exactness argument,
the workspace layout and the launch shape; this module states the host side of
the same contract.

The workspace is explicit and stream/device-safe:

* :func:`staged_workspace_bytes` and :class:`StagedWorkspaceLayout` give its
  exact size and layout, and ``hipengine_gemma4_attention_staged_workspace_bytes``
  exports the same definition from the kernel so the two cannot drift.
* :func:`staged_workspace_buffer` takes the bytes from a
  :class:`~hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention.Gemma4AttentionScratch`,
  the strict family's own ownership contract. Pass ``scratch=`` to keep the
  allocation across steps; leave it out and the launcher allocates and releases
  a temporary one around the launch.
* The workspace scales as ``tokens * num_heads * keys`` floats, so a caller that
  cannot afford it must decide that before launching -- :func:`staged_plan`
  reports the number without touching a device, and refuses a shape whose grid
  dimensions or workspace products the launch cannot represent, naming the
  limit it hit rather than wrapping an index.

The module itself registers nothing in the four-axis kernel registry: the
Gemma 4 GGUF execution profiles register :func:`gemma4_attention_staged_bf16`
as the ``gemma4_staged`` prefill-attention variant for both head geometries, and
the family's selector reaches it through
:func:`~hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention.select_prefill_attention`.
The launcher is also reachable directly, as the tests do.

Importing this module registers ctypes launch wrappers but does not build or
load ROCm until a wrapper is called.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.core.memory import DeviceBuffer
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import Gemma4AttentionScratch

_SOURCE = Path(__file__).with_name("gemma4_attention_staged.hip")
_OUTPUT_NAME = "gemma4_attention_staged.so"

SYMBOL_STAGED_BF16 = "hipengine_gemma4_attention_staged_bf16"
SYMBOL_STAGED_F32 = "hipengine_gemma4_attention_staged_f32"
_SYMBOL_WORKSPACE_BYTES = "hipengine_gemma4_attention_staged_workspace_bytes"
_SYMBOL_LDS_BYTES = "hipengine_gemma4_attention_staged_lds_bytes"

# Five buffers, then (tokens, num_heads, num_kv_heads, head_dim, scale, stream,
# keys, window, row_offset) and the workspace pointer. The first thirteen match
# the strict prefill ABI argument for argument -- same order, same meanings --
# so a caller can drive both from one set of device buffers; the workspace
# pointer is this family's own tail. The stream slot is ``hipStream_t``, a
# pointer, so it is declared ``c_void_p`` exactly as the strict wrapper declares
# it rather than as an integer of the same width.
_ARGTYPES_STAGED = (
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
)

# Launch geometry, mirrored from the .hip so the planner can answer without a
# device. A drift between these and the kernel's own constants would be a memory
# corruption bug rather than a slow path, which is why
# `test_staged_workspace_bytes_matches_the_kernel_export` compares them against
# the kernel's exports on a device.
THREADS = 256
SCORE_CHUNK_KEYS = 1024
MAX_ROWS_PER_BLOCK = 8
PV_TILE_KEYS = 256
LDS_BUDGET_BYTES = 64 * 1024

# gfx11 grid X allows 2**31-1 CTAs; Y and Z allow 65535. Limits describe
# the actual launch dimensions, not a context-admission policy.
MAX_GRID_X = 2**31 - 1
MAX_GRID_DIM = 65535  # Y/Z
MAX_KEY_CHUNKS = MAX_GRID_DIM
MAX_KEYS = MAX_KEY_CHUNKS * SCORE_CHUNK_KEYS

# The workspace is indexed and sized with int64 in the kernel and allocated from
# a size_t byte count here. The launch limits bound every product below this, so
# the arithmetic in the two places cannot disagree about a representable size.
_MAX_WORKSPACE_FLOATS = (2**63 - 1) // 4

# The warp-key score tree is specialised for exactly these widths. The strict
# family falls back to a per-key block-sum kernel for any other width; this
# candidate has no such fallback, so it names the miss instead of approximating.
SCORE_TREE_HEAD_DIMS = (256, 512)


def staged_score_chunks(keys: int) -> int:
    """Number of score CTAs per row, and the workspace's chunk-max array width."""

    return (int(keys) + SCORE_CHUNK_KEYS - 1) // SCORE_CHUNK_KEYS


@dataclass(frozen=True, slots=True)
class StagedWorkspaceLayout:
    """The workspace's three regions, in FP32 elements.

    ``scores`` holds the logits after stage 1 and the weights after stage 2;
    ``chunk_max`` holds one running maximum per (row, score chunk); ``denominator``
    holds one value per row. Every region is FP32 and 4-byte aligned, and the
    whole allocation is one contiguous buffer.
    """

    scores_floats: int
    chunk_max_floats: int
    denominator_floats: int

    @property
    def total_floats(self) -> int:
        return self.scores_floats + self.chunk_max_floats + self.denominator_floats

    @property
    def total_bytes(self) -> int:
        return self.total_floats * 4


@dataclass(frozen=True, slots=True)
class StagedAttentionPlan:
    """One resolved launch: the grid, the shared footprint and the workspace.

    Returned by the launcher so a caller can confirm which decomposition ran
    rather than inferring it from timings -- the same reason the strict family
    exports its decode selection. ``workspace_bytes`` is what the caller must
    own; ``lds_bytes`` is what the kernel reserves, and it does not depend on
    ``keys``.
    """

    tokens: int
    keys: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rows: int
    chunks: int
    rows_per_head: int
    row_groups: int
    threads: int
    score_chunk_keys: int
    pv_tile_keys: int
    score_lds_bytes: int
    softmax_lds_bytes: int
    pv_lds_bytes: int
    workspace: StagedWorkspaceLayout

    @property
    def lds_bytes(self) -> int:
        return max(self.score_lds_bytes, self.softmax_lds_bytes, self.pv_lds_bytes)

    @property
    def workspace_bytes(self) -> int:
        return self.workspace.total_bytes

    @property
    def score_grid(self) -> tuple[int, int]:
        return (self.rows, self.chunks)

    @property
    def softmax_grid(self) -> tuple[int, int]:
        return (self.rows, 1)

    @property
    def pv_grid(self) -> tuple[int, int, int]:
        return (self.tokens, self.num_kv_heads, self.row_groups)

    def describe(self) -> str:
        """One line naming the decomposition, for diagnostics."""

        return (
            f"gemma4_attention_staged rows={self.rows} keys={self.keys} "
            f"score_grid={self.score_grid[0]}x{self.score_grid[1]} "
            f"pv_grid={self.pv_grid[0]}x{self.pv_grid[1]}x{self.pv_grid[2]} "
            f"rows_per_head={self.rows_per_head} lds={self.lds_bytes}B "
            f"workspace={self.workspace_bytes}B"
        )


def staged_workspace_bytes(tokens: int, num_heads: int, keys: int) -> int:
    """Bytes of workspace one launch of this shape needs.

    ``(tokens * num_heads * keys + tokens * num_heads * chunks + tokens *
    num_heads) * 4``, where ``chunks`` is :func:`staged_score_chunks`. Computed
    without a device; ``hipengine_gemma4_attention_staged_workspace_bytes``
    exports the same formula from the kernel and returns 0 for the shapes this
    refuses.

    Raises ``ValueError`` for a non-positive dimension or a region that is not
    representable, and ``NotImplementedError`` for a shape past one of the
    launch's grid dimensions -- the same refusal :func:`staged_plan` makes. The
    caller decides whether it can afford the result; the launch limits bound how
    large the result can be.
    """

    for name, value in (("tokens", tokens), ("num_heads", num_heads), ("keys", keys)):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    tokens, num_heads, keys = int(tokens), int(num_heads), int(keys)
    _staged_launch_shape(tokens, num_heads, keys)
    layout = _staged_workspace_layout(tokens, num_heads, keys)
    return layout.total_bytes


def _checked_floats(name: str, value: int) -> int:
    """``value``, or a named refusal if the kernel could not index it."""

    if value > _MAX_WORKSPACE_FLOATS:
        raise ValueError(
            f"staged attention workspace {name} is {value} FP32 elements, past the "
            f"{_MAX_WORKSPACE_FLOATS} that are representable in the kernel's int64 "
            f"index arithmetic and size_t byte count"
        )
    return value


def _staged_workspace_layout(tokens: int, num_heads: int, keys: int) -> StagedWorkspaceLayout:
    """The three regions for one shape, or a named refusal if it has no size.

    Every region is a product of the same row count, so this is where the
    workspace arithmetic is formed and where its representability is checked.
    :func:`staged_plan` and :func:`staged_workspace_bytes` bound the shape before
    they reach here, which is what keeps the products small; the checks are for a
    direct caller, and they are the definition the kernel's export mirrors when
    it returns its invalid sentinel instead.
    """

    rows = _checked_floats("row count (tokens * num_heads)", tokens * num_heads)
    chunks = staged_score_chunks(keys)
    layout = StagedWorkspaceLayout(
        scores_floats=_checked_floats("scores region (rows * keys)", rows * keys),
        chunk_max_floats=_checked_floats(
            "chunk-maximum region (rows * chunks)", rows * chunks
        ),
        denominator_floats=rows,
    )
    _checked_floats("size", layout.total_floats)
    return layout


def _staged_launch_shape(tokens: int, num_heads: int, keys: int) -> int:
    """Validate the dimensions that drive this launch's grids; return the rows.

    The score launch is ``(rows, chunks)`` CTAs; PV is
    ``(tokens, num_kv_heads, row_groups)`` CTAs. X allows ``MAX_GRID_X`` CTAs;
    Y/Z allow ``MAX_GRID_DIM``. KV heads and
    row groups are checked by the full planner, which knows those dimensions.
    """

    if tokens > MAX_GRID_X:
        raise NotImplementedError(
            f"gemma4_attention_staged launches one PV CTA per token on a "
            f"{MAX_GRID_X}-wide grid X dimension, so it serves at most "
            f"{MAX_GRID_X} tokens; {tokens} would need a flattened PV grid"
        )
    if num_heads > MAX_GRID_X:
        raise NotImplementedError(
            f"gemma4_attention_staged launches one score CTA per (token, head) row, "
            f"so its row count is tokens * num_heads on a {MAX_GRID_X}-wide grid X "
            f"dimension; num_heads {num_heads} already exceeds it"
        )
    chunks = staged_score_chunks(keys)
    if chunks > MAX_KEY_CHUNKS:
        raise NotImplementedError(
            f"gemma4_attention_staged launches one score CTA per {SCORE_CHUNK_KEYS} keys "
            f"on a {MAX_GRID_DIM}-wide grid dimension, so it serves at most {MAX_KEYS} "
            f"keys; {keys} would need a flattened score grid"
        )
    rows = tokens * num_heads
    if rows > MAX_GRID_X:
        raise NotImplementedError(
            f"gemma4_attention_staged launches one score CTA per (token, head) row on "
            f"a {MAX_GRID_X}-wide grid X dimension, so it serves at most "
            f"{MAX_GRID_X} rows; {tokens} tokens x {num_heads} heads = {rows} rows "
            f"would need a flattened score grid"
        )
    return rows


def staged_score_lds_bytes(head_dim: int) -> int:
    """Shared bytes the score stage reserves: the query row and eight warp maxima."""

    return (int(head_dim) + (THREADS // 32)) * 4


def staged_softmax_lds_bytes() -> int:
    """Shared bytes the softmax stage reserves: the 256-lane tree and the maximum."""

    return (THREADS + 1) * 4


def staged_pv_lds_bytes() -> int:
    """Shared bytes the PV stage reserves: one weight tile per resident row."""

    return MAX_ROWS_PER_BLOCK * PV_TILE_KEYS * 4


def staged_lds_bytes(head_dim: int) -> int:
    """Largest shared allocation any of the three stages asks for.

    The bounded-layout property: this does not mention ``keys``, where
    ``gemma4_attention_shared_bytes(head_dim=512, keys=n)`` grows with ``n`` and
    refuses above 15,616.
    """

    return max(
        staged_score_lds_bytes(head_dim),
        staged_softmax_lds_bytes(),
        staged_pv_lds_bytes(),
    )


def staged_plan(
    *,
    tokens: int,
    keys: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> StagedAttentionPlan:
    """Resolve the launch for one shape, or name the capability it lacks.

    Host-only: no build, no device, no allocation. Shape errors are
    ``ValueError`` (the caller asked for something meaningless); a geometry the
    candidate has no kernel for, or a shape past one of the launch's grid
    dimensions, is ``NotImplementedError`` naming what is missing, never a silent
    fallback to an approximation.
    """

    for name, value in (
        ("tokens", tokens),
        ("keys", keys),
        ("num_heads", num_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    tokens, keys = int(tokens), int(keys)
    num_heads, num_kv_heads, head_dim = int(num_heads), int(num_kv_heads), int(head_dim)
    if num_heads % num_kv_heads:
        raise ValueError(
            f"num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
        )
    if head_dim not in SCORE_TREE_HEAD_DIMS:
        raise NotImplementedError(
            f"gemma4_attention_staged implements the warp-key score tree for head_dim "
            f"{' and '.join(str(width) for width in SCORE_TREE_HEAD_DIMS)}; head_dim "
            f"{head_dim} would need a staged per-key block-sum score stage, which this "
            f"candidate does not have"
        )
    rows = _staged_launch_shape(tokens, num_heads, keys)
    chunks = staged_score_chunks(keys)
    if num_kv_heads > MAX_GRID_DIM:
        raise NotImplementedError(
            f"gemma4_attention_staged launches one PV CTA per KV head on a "
            f"{MAX_GRID_DIM}-wide grid dimension, so it serves at most {MAX_GRID_DIM} "
            f"KV heads; {num_kv_heads} would need a flattened PV grid"
        )
    rows_per_head = num_heads // num_kv_heads
    row_groups = (rows_per_head + MAX_ROWS_PER_BLOCK - 1) // MAX_ROWS_PER_BLOCK
    if row_groups > MAX_GRID_DIM:
        raise NotImplementedError(
            f"gemma4_attention_staged launches one PV CTA per {MAX_ROWS_PER_BLOCK} query "
            f"rows that share a KV head on a {MAX_GRID_DIM}-wide grid dimension, so it "
            f"serves a GQA ratio of at most {MAX_GRID_DIM * MAX_ROWS_PER_BLOCK}; "
            f"{num_heads}q/{num_kv_heads}kv is a ratio of {rows_per_head}"
        )
    return StagedAttentionPlan(
        tokens=tokens,
        keys=keys,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rows=rows,
        chunks=chunks,
        rows_per_head=rows_per_head,
        row_groups=row_groups,
        threads=THREADS,
        score_chunk_keys=SCORE_CHUNK_KEYS,
        pv_tile_keys=PV_TILE_KEYS,
        score_lds_bytes=staged_score_lds_bytes(head_dim),
        softmax_lds_bytes=staged_softmax_lds_bytes(),
        pv_lds_bytes=staged_pv_lds_bytes(),
        workspace=_staged_workspace_layout(tokens, num_heads, keys),
    )


def staged_workspace_buffer(
    scratch: Gemma4AttentionScratch,
    plan: StagedAttentionPlan,
    *,
    stream: int,
    runtime: HipRuntime,
) -> DeviceBuffer:
    """The workspace buffer for ``plan``, owned by ``scratch`` on this stream.

    Split out from the launcher so the ownership contract is testable without a
    device: ``scratch`` is the strict family's
    :class:`~hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention.Gemma4AttentionScratch`,
    which keeps one buffer per stream, doubles on growth, and frees only after
    every used stream is quiescent.
    """

    return scratch.buffer(plan.workspace_bytes, stream=stream, runtime=runtime)


def plan_gemma4_attention_staged_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "prefill",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gemma4_attention_staged",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gemma4_attention_staged(
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
        family="gemma4_attention_staged",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def staged_workspace_bytes_from_kernel(
    tokens: int, num_heads: int, keys: int, *, library: ctypes.CDLL | None = None
) -> int:
    """The kernel's own workspace size, for cross-checking the Python formula."""

    library = library or build_gemma4_attention_staged(load=True)
    fn = signed_kernel_fn(
        library, _SYMBOL_WORKSPACE_BYTES, (ctypes.c_int64, ctypes.c_int64, ctypes.c_int64),
        ctypes.c_size_t,
    )
    return int(fn(int(tokens), int(num_heads), int(keys)))


def staged_lds_bytes_from_kernel(
    head_dim: int, *, library: ctypes.CDLL | None = None
) -> int:
    """The kernel's own shared-memory reservation, for cross-checking the formula."""

    library = library or build_gemma4_attention_staged(load=True)
    fn = signed_kernel_fn(library, _SYMBOL_LDS_BYTES, (ctypes.c_int64,), ctypes.c_size_t)
    return int(fn(int(head_dim)))


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def _launch_staged(
    symbol: str,
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    keep_mask_ptr: int,
    out_ptr: int,
    *,
    tokens: int,
    keys: int | None,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    window: int,
    row_offset: int,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
    scratch: Gemma4AttentionScratch | None,
) -> StagedAttentionPlan:
    for name, coordinate in (("window", window), ("row_offset", row_offset)):
        if not -(2**63) <= int(coordinate) <= 2**63 - 1:
            raise ValueError(f"{name} must be representable in the raw int64 ABI")
    key_count = int(tokens) if keys is None else int(keys)
    plan = staged_plan(
        tokens=tokens,
        keys=key_count,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    library = library or build_gemma4_attention_staged(load=True)
    runtime = runtime or get_hip_runtime()
    temporary = Gemma4AttentionScratch() if scratch is None else None
    owner = scratch if scratch is not None else temporary
    try:
        workspace = staged_workspace_buffer(owner, plan, stream=stream, runtime=runtime)
        fn = signed_kernel_fn(library, symbol, _ARGTYPES_STAGED, ctypes.c_int)
        err = fn(
            query_ptr,
            key_ptr,
            value_ptr,
            keep_mask_ptr,
            out_ptr,
            plan.tokens,
            plan.num_heads,
            plan.num_kv_heads,
            plan.head_dim,
            ctypes.c_float(scale),
            stream,
            plan.keys,
            window,
            row_offset,
            ctypes.c_void_p(workspace.ptr),
        )
        _check_launch(runtime, err)
    finally:
        # A caller without reusable ownership releases only after the stream is
        # quiescent, including on a partial launch failure.
        if temporary is not None:
            temporary.close()
    return plan


def gemma4_attention_staged_bf16(
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
    scratch: Gemma4AttentionScratch | None = None,
    window: int = 0,
    row_offset: int = 0,
) -> StagedAttentionPlan:
    """Masked, ungated staged prefill attention over ``tokens`` queries.

    Same buffers and same arguments as
    :func:`~hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention.gemma4_attention_prefill_bf16`:
    ``query`` is ``(tokens, num_heads, head_dim)`` BF16, ``key`` and ``value`` are
    ``(keys, num_kv_heads, head_dim)`` BF16, ``keep_mask`` is a ``(tokens, keys)``
    uint8 **keep** mask, and ``out`` is ``(tokens, num_heads, head_dim)`` BF16.
    ``keys`` defaults to ``tokens``. Pass ``scale=1.0`` for Gemma 4, and see the
    strict wrapper for ``window``/``row_offset``: the trim they license is the
    same one, and it only drops columns whose weight is zero.

    The output is bit-identical to the strict wrapper's for the same inputs,
    including its ``tokens == 1`` decode routing. Pass ``scratch=`` to own the
    workspace across calls; without it the launcher allocates and frees a
    temporary one. Returns the :class:`StagedAttentionPlan` it launched, so the
    decomposition that ran is observable rather than inferred.
    """

    return _launch_staged(
        SYMBOL_STAGED_BF16,
        query_ptr,
        key_ptr,
        value_ptr,
        keep_mask_ptr,
        out_ptr,
        tokens=tokens,
        keys=keys,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
        window=window,
        row_offset=row_offset,
        stream=stream,
        library=library,
        runtime=runtime,
        scratch=scratch,
    )


def gemma4_attention_staged_f32(
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
    scratch: Gemma4AttentionScratch | None = None,
    window: int = 0,
    row_offset: int = 0,
) -> StagedAttentionPlan:
    """F32 entry point, for validating against the f32 CPU reference.

    The reference works in f32, so comparing through a BF16 round trip would
    measure the rounding rather than the kernel. Identical arithmetic and
    identical arguments to :func:`gemma4_attention_staged_bf16`.
    """

    return _launch_staged(
        SYMBOL_STAGED_F32,
        query_ptr,
        key_ptr,
        value_ptr,
        keep_mask_ptr,
        out_ptr,
        tokens=tokens,
        keys=keys,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
        window=window,
        row_offset=row_offset,
        stream=stream,
        library=library,
        runtime=runtime,
        scratch=scratch,
    )
