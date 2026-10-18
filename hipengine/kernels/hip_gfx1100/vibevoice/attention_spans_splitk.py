"""Split-K span attention for a decode step (attention_spans_splitk.hip).

A production sibling of ``vv_attention_spans``: the same span ABI and the same
per-chunk softmax, reorganized so the grid is ``kv_heads * splits`` instead of the
query-head count, each KV row is one coalesced run shared by the whole query group,
and the split's scores are staged in shared memory. Arithmetic differs only in the
reduction order (per-split maxima and sums combined afterwards), which is the standard
split-softmax reassociation, so it is registered as a ``production`` variant and keeps
``vv_attention_spans`` as its strict parent.

The split count is a function of the device as well as the context: a block covers one
(KV head, split), so the grid has to be large enough to fill however many compute units
this part has, and no larger than the context can usefully divide. ``plan_split`` owns
that decision and the kernel receives the resulting ``splits`` and ``chunk`` from it, so
the block's serial run and its shared-memory footprint are host-known quantities.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

from hipengine.core.build import BuildArtifact, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime

_SOURCE = Path(__file__).with_name("attention_spans_splitk.hip")
_OUTPUT_NAME = "vibevoice_attention_spans_splitk"

_P = ctypes.c_void_p
_F = ctypes.c_float
_I = ctypes.c_int64
_S = ctypes.c_void_p

_ARGTYPES_SPLITK = (_P, _P, _P, _P, _P) + (_P, _P, _P, _P, _P) + (
    _I,
    _I,
    _I,
    _I,
    _I,
    _I,
    _I,
    _F,
    _S,
)
_ARGTYPES_WORKSPACE = (_I, _I, _I, _I)

# The kernel is specialized for head_dim 128 with one instantiation per query-heads-
# per-KV-head ratio. Anything else takes the strict span kernel.
SUPPORTED_HEAD_DIM = 128
SUPPORTED_Q_PER_KV = (1, 2, 4, 8)
# The kernel stages one split's scores in shared memory and bounds the grid by this,
# and the runtime sizes its workspace for it, so it is the ceiling for every plan.
MAX_SPLITS = 64
# A split is worth cutting while it still holds enough tokens to keep the block's
# eight warps busy; below this the split adds a combine pass and a block launch for
# almost no parallel work.
MIN_CHUNK = 8
# Splits aim for this many tokens each. Longer runs than this serialize one block's
# three passes over the KV rows while other blocks sit idle.
CHUNK_TARGET = 32
# Hard bound on a block's staged scores: ``chunk`` floats per query head in the group.
# At the target geometry that is 2 * 256 * 4 = 2 KiB, and it is what keeps a
# mis-sized plan from overrunning shared memory.
MAX_CHUNK = 256
# Blocks wanted per compute unit. An RDNA3 compute unit holds 32 wave32 slots, so a
# 256-thread block is a quarter of one and four blocks per CU is full occupancy. The
# measured floor is two: half occupancy, because the KV scan is limited by memory
# latency and the extra waves contend for the same rows instead of hiding more of it.
# Measured on a 96-CU gfx1100 at 998 tokens of context, 0.026 ms per call at 24-32
# splits (2.0-2.67 blocks/CU) against 0.032 ms at 48 (full occupancy).
BLOCKS_PER_UNIT = 2
# Used only when the device probe cannot answer. 40 is the smallest gfx11 part we
# target, so an unknown device is treated as one rather than as the largest.
DEFAULT_COMPUTE_UNITS = 40


def plan_attention_spans_splitk_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: str = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="vibevoice_attention_spans_splitk",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_attention_spans_splitk(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: str = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | None:
    return build_hip(
        sources=[_SOURCE],
        family="vibevoice_attention_spans_splitk",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _library() -> ctypes.CDLL:
    library = build_attention_spans_splitk()
    if library is None:
        raise RuntimeError("vibevoice_attention_spans_splitk build returned no library")
    return library


@dataclass(frozen=True)
class SplitPlan:
    """How a decode step's KV range is cut: ``splits`` blocks of ``chunk`` tokens."""

    splits: int
    chunk: int

    @property
    def covers(self) -> int:
        return self.splits * self.chunk


def plan_split(
    context: int,
    kv_heads: int,
    *,
    compute_units: int | None = None,
    blocks_per_unit: int = BLOCKS_PER_UNIT,
) -> SplitPlan:
    """Splits and chunk for a decode step with ``context`` live tokens.

    Two lower bounds decide the split count, and the plan takes whichever is larger:

    * ``context / CHUNK_TARGET`` keeps each block's serial run short, which is what
      makes the step nearly context-independent;
    * ``blocks_per_unit * compute_units / kv_heads`` keeps the grid wide enough to fill
      *this* device. One block covers one (KV head, split), so the grid is
      ``kv_heads * splits`` and a part with more compute units needs more splits for
      the same fill. This bound only decides the answer when the context is short
      enough that the chunk target alone would under-fill the device.

    ``MIN_CHUNK`` and ``MAX_SPLITS`` bound the result, so a short context on a wide
    device does not split into blocks that each hold a token or two. ``compute_units``
    is the device's CU count from :func:`hipengine.core.hip.device_compute_units`;
    ``None`` falls back to :data:`DEFAULT_COMPUTE_UNITS`, the smallest target part.
    """

    context = max(1, int(context))
    kv_heads = max(1, int(kv_heads))
    units = int(compute_units) if compute_units else DEFAULT_COMPUTE_UNITS
    units = max(1, units)
    per_unit = max(1, int(blocks_per_unit))

    by_chunk = -(-context // CHUNK_TARGET)
    by_device = -(-(per_unit * units) // kv_heads)
    splits = min(MAX_SPLITS, max(by_chunk, by_device))
    splits = max(1, min(splits, -(-context // MIN_CHUNK)))
    chunk = -(-context // splits)
    return SplitPlan(splits=splits, chunk=chunk)


def supports(q_heads: int, kv_heads: int, head_dim: int) -> bool:
    """Whether this kernel covers the geometry, so the caller can select it."""

    return (
        q_heads > 0
        and kv_heads > 0
        and head_dim == SUPPORTED_HEAD_DIM
        and q_heads % kv_heads == 0
        and q_heads // kv_heads in SUPPORTED_Q_PER_KV
    )


def workspace_floats(rows: int, q_heads: int, head_dim: int, splits: int) -> int:
    return int(rows) * int(q_heads) * int(splits) * (int(head_dim) + 2)


def vv_attention_spans_splitk(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    workspace_ptr: int,
    spans,
    rows: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float,
    context: int,
    *,
    compute_units: int | None = None,
    plan: SplitPlan | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    from hipengine.kernels.hip_gfx1100.vibevoice.encoder import _span_arguments

    if not supports(q_heads, kv_heads, head_dim):
        raise ValueError(
            f"split-K span attention covers head_dim {SUPPORTED_HEAD_DIM} with "
            f"{SUPPORTED_Q_PER_KV} query heads per KV head; got q_heads={q_heads}, "
            f"kv_heads={kv_heads}, head_dim={head_dim}"
        )
    # The kernel clamps each row's context by the span bound, so plan against the same
    # value: a plan that covers more than the kernel can read is wasted work, and one
    # that covers less would leave the tail of a long row unattended.
    live = max(1, int(spans.max_live_count))
    effective = min(max(1, int(context)), live)
    plan = plan or plan_split(effective, kv_heads, compute_units=compute_units)
    if not 0 < plan.splits <= MAX_SPLITS:
        raise ValueError(f"splits must be in [1, {MAX_SPLITS}], got {plan.splits}")
    if not 0 < plan.chunk <= MAX_CHUNK:
        raise ValueError(f"chunk must be in [1, {MAX_CHUNK}], got {plan.chunk}")
    if plan.covers < effective:
        raise ValueError(
            f"plan covers {plan.covers} tokens ({plan.splits} splits of {plan.chunk}) "
            f"but the step has {effective} live tokens"
        )
    metadata = _span_arguments(spans, rows)
    base_offsets, live_counts, token_positions, evict_mask, row_positions = metadata
    library = library or _library()
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library, "hipengine_vv_attention_spans_splitk", _ARGTYPES_SPLITK, ctypes.c_int
    )
    err = fn(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        out_ptr,
        workspace_ptr,
        base_offsets,
        live_counts,
        token_positions,
        evict_mask,
        row_positions,
        rows,
        q_heads,
        kv_heads,
        head_dim,
        spans.max_live_count,
        plan.splits,
        plan.chunk,
        scale,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))
