"""Split-K span attention for a decode step (attention_spans_splitk.hip).

A production sibling of ``vv_attention_spans``: the same span ABI and the same
per-chunk softmax, reorganized so the grid is ``kv_heads * splits`` instead of the
query-head count, each KV row is one coalesced run shared by the whole query group,
and the split's scores are staged in shared memory. Arithmetic differs only in the
reduction order (per-split maxima and sums combined afterwards), which is the standard
split-softmax reassociation, so it is registered as a ``production`` variant and keeps
``vv_attention_spans`` as its strict parent.
"""

from __future__ import annotations

import ctypes
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

_ARGTYPES_SPLITK = (_P, _P, _P, _P, _P) + (_P, _P, _P, _P, _P) + (_I, _I, _I, _I, _I, _I, _F, _S)
_ARGTYPES_WORKSPACE = (_I, _I, _I, _I)

# The kernel is specialized for head_dim 128 with one instantiation per query-heads-
# per-KV-head ratio. Anything else takes the strict span kernel.
SUPPORTED_HEAD_DIM = 128
SUPPORTED_Q_PER_KV = (1, 2, 4, 8)
MAX_SPLITS = 64
SPLIT_TARGET = 32


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


def split_count(context: int, *, target: int = SPLIT_TARGET) -> int:
    """Splits for a decode step whose context is ``context`` tokens.

    One block covers one KV head's whole query group, so the grid is
    ``kv_heads * splits``; this keeps a split near ``target`` tokens and the split
    count inside the kernel's bound.
    """

    if context <= 0:
        return 1
    return max(1, min(MAX_SPLITS, (int(context) + target - 1) // target))


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
    splits: int,
    *,
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
    if not 0 < splits <= MAX_SPLITS:
        raise ValueError(f"splits must be in [1, {MAX_SPLITS}]")
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
        splits,
        scale,
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))
