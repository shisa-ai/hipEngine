"""Raw-pointer wrappers for the Gemma 4 attention family on gfx1100.

Gemma 4 attention is *ungated*. Every Qwen3.5 prefill variant reads an attention
gate and multiplies by ``sigmoid(gate)``; Laguna's ungated kernel hard-codes
Laguna's head geometry. Neither can serve Gemma 4, so this family exists.

Two kernels serve the family and :func:`attention_symbol` routes between them:
the original correctness-first block kernel for multi-token blocks, and a
decode kernel for ``tokens == 1`` that reproduces the block kernel's arithmetic
bit-for-bit while batching barrier rounds across keys.

Two Gemma 4 specifics the caller must respect, both documented on the kernel:

* ``scale`` is always ``1.0``, never ``head_dim**-0.5``. Gemma 4 folds the softmax
  scaling into the query norm weight, so pass ``geometry.scale`` rather than
  assuming the usual reciprocal square root.
* ``keep_mask`` is a caller-supplied **keep** mask, not an implied causal test.
  Sliding-window layers need ``key > query - window`` in addition to
  ``key <= query``, and which layers those are is a config decision.

Importing this module registers ctypes launch wrappers but does not build or load
ROCm until a wrapper is called.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, malloc, free
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gemma4_attention.hip")
_OUTPUT_NAME = "gemma4_attention.so"

_SYMBOL_PREFILL_BF16 = "hipengine_gemma4_attention_prefill_bf16"
_SYMBOL_PREFILL_F32 = "hipengine_gemma4_attention_prefill_f32"
_SYMBOL_DECODE_BF16 = "hipengine_gemma4_attention_decode_bf16"
_SYMBOL_DECODE_F32 = "hipengine_gemma4_attention_decode_f32"
_SYMBOL_DECODE_VARIANT = "hipengine_gemma4_attention_decode_variant"

# The five buffers and the geometry scalars both prefill and decode share.
_ARGTYPES_COMMON = (
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

# Prefill adds the sliding walk's window and this block's start in the mask's
# column frame; decode adds the two-phase split's scratch pointer and slice
# count instead. The two tails are independent, so they cannot share one base.
_ARGTYPES_PREFILL = _ARGTYPES_COMMON + (ctypes.c_int64, ctypes.c_int64)
_ARGTYPES_DECODE = _ARGTYPES_COMMON + (ctypes.c_void_p, ctypes.c_int)

_DECODE_SYMBOLS = (_SYMBOL_DECODE_BF16, _SYMBOL_DECODE_F32)
_SYMBOL_SPLIT_WORKSPACE_BYTES = "hipengine_gemma4_decode_split_workspace_bytes"
_SYMBOL_DECODE_SELECTION = "hipengine_gemma4_decode_selection"
class Gemma4AttentionScratch:
    """Split workspace owned by one caller on one runtime/device.

    Streams have separate buffers. Growth doubles capacity and retains old
    allocations until close synchronizes every used stream, so queued kernels
    keep valid pointers. Construction is host-only.
    """

    def __init__(self) -> None:
        self._runtime: HipRuntime | None = None
        self._device: int | None = None
        self._current: dict[int, DeviceBuffer] = {}
        self._owned: list[DeviceBuffer] = []
        self._closed = False

    def buffer(self, nbytes: int, *, stream: int, runtime: HipRuntime) -> DeviceBuffer:
        if self._closed:
            raise RuntimeError("attention scratch is closed")
        if nbytes <= 0:
            raise ValueError("attention scratch size must be positive")
        if self._runtime is not None and runtime is not self._runtime:
            raise ValueError("attention scratch cannot change runtime")
        device = runtime.current_device()
        if self._device is not None and device != self._device:
            raise ValueError("attention scratch cannot change device")
        previous = self._current.get(stream)
        if previous is not None and previous.nbytes >= nbytes:
            return previous
        capacity = max(nbytes, 2 * previous.nbytes) if previous else nbytes
        buffer = malloc(capacity, runtime=runtime)
        self._runtime, self._device = runtime, device
        self._owned.append(buffer)
        self._current[stream] = buffer
        return buffer

    def close(self) -> None:
        if self._closed:
            return
        if self._runtime is not None:
            if self._runtime.current_device() != self._device:
                raise ValueError("attention scratch must close on its owning device")
            for stream in self._current:
                self._runtime.stream_synchronize(stream)
            while self._owned:
                free(self._owned[-1], runtime=self._runtime)
                self._owned.pop()
        self._current.clear()
        self._closed = True


def decode_slices(keys: int, head_dim: int) -> int:
    """Return the legacy split-policy value used by the launch ABI.

    One selects the single kernel; values greater than one select separate
    weight and value passes. The value pass partitions output dimensions, not
    keys, and preserves ascending-key accumulation. The exact value above one
    no longer controls its grid. Keep the existing selection policy while
    callers migrate away from the key-slice terminology.
    """

    if keys < 1024:
        return 1
    if head_dim <= 256:
        return 4
    slices = 1
    while slices < 4 and keys > slices * 512:
        slices <<= 1
    return slices


def decode_selection(library: ctypes.CDLL | None = None) -> int:
    """Which decode kernel the launcher last selected.

    0 = block kernel, 1 = key-class single kernel, 2 = key-class two-phase split.
    Introspection for confirming that the intended path ran: the three are
    numerically close by design, so a timing or a parity test cannot tell them
    apart, and a caller who needs to know which one a real request took needs
    this rather than an inference.
    """

    library = library or build_gemma4_attention(load=True)
    fn = signed_kernel_fn(library, _SYMBOL_DECODE_SELECTION, (), ctypes.c_int)
    return int(fn())


def split_workspace_bytes(
    tokens: int, num_heads: int, head_dim: int, keys: int, slices: int, *,
    library: ctypes.CDLL | None = None,
) -> int:
    """Bytes of scratch the split needs, from the kernel's own definition."""

    library = library or build_gemma4_attention(load=True)
    fn = signed_kernel_fn(
        library,
        _SYMBOL_SPLIT_WORKSPACE_BYTES,
        (ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int),
        ctypes.c_size_t,
    )
    return int(fn(int(tokens), int(num_heads), int(head_dim), int(keys), int(slices)))


def plan_gemma4_attention_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gemma4_attention",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gemma4_attention(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family="gemma4_attention",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def gemma4_attention_decode_variant(library: ctypes.CDLL | None = None) -> str:
    """Which decode kernel the launcher last selected.

    ``"class"`` is the single key-class kernel, ``"split"`` separates its
    weights and dimension-partitioned value passes, and ``"block"`` is the
    original block kernel. Introspection only; selection is not a quality gate.
    """

    library = library or build_gemma4_attention(load=True)
    fn = signed_kernel_fn(library, _SYMBOL_DECODE_VARIANT, [], ctypes.c_int)
    return {0: "block", 1: "class", 2: "split"}[int(fn())]


def gemma4_attention_shared_bytes(*, head_dim: int, keys: int) -> int:
    """Validate the resident-logit kernel's gfx1100 shared-memory requirement.

    This implementation stores one logit per live key in LDS. Larger contexts
    need a tiled attention implementation, not a larger prefill scratch block.
    """

    if head_dim <= 0 or keys <= 0:
        raise ValueError("head_dim and keys must be positive")
    threads = min(256, 1 << (int(head_dim) - 1).bit_length())
    required = (int(head_dim) + int(keys) + threads) * 4
    if required > 64 * 1024:
        raise NotImplementedError(
            f"Gemma 4 gfx1100 attention requires {required} bytes of shared memory "
            f"for head_dim={head_dim}, keys={keys}; this kernel supports at most "
            "65536 bytes and needs tiled attention for this context"
        )
    return required


def _check_prefill_shape(tokens: int, num_heads: int, num_kv_heads: int, head_dim: int) -> None:
    for name, value in (
        ("tokens", tokens),
        ("num_heads", num_heads),
        ("num_kv_heads", num_kv_heads),
        ("head_dim", head_dim),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if num_heads % num_kv_heads:
        raise ValueError(
            f"num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
        )


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def attention_symbol(dtype: str, *, tokens: int, head_dim: int) -> str:
    """Select the attention entry point for one launch.

    A decode step (``tokens == 1``) runs the batched-barrier decode kernel,
    which matches the block kernel's outputs bit-for-bit at about half its
    barrier cost. Multi-token blocks keep the original block kernel.
    ``head_dim`` only has to be positive (shape validation happens in the
    launcher).
    """

    if dtype not in ("bf16", "f32"):
        raise ValueError(f"unsupported attention dtype: {dtype!r}")
    if tokens == 1:
        return _SYMBOL_DECODE_BF16 if dtype == "bf16" else _SYMBOL_DECODE_F32
    return _SYMBOL_PREFILL_BF16 if dtype == "bf16" else _SYMBOL_PREFILL_F32


def _launch_prefill(
    symbol: str,
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
    keys: int | None,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: HipRuntime | None,
    scratch: Gemma4AttentionScratch | None = None,
    window: int = 0,
    row_offset: int = 0,
) -> None:
    _check_prefill_shape(tokens, num_heads, num_kv_heads, head_dim)
    key_count = tokens if keys is None else int(keys)
    if key_count <= 0:
        raise ValueError("keys must be positive")
    gemma4_attention_shared_bytes(head_dim=head_dim, keys=key_count)
    library = library or build_gemma4_attention(load=True)
    runtime = runtime or get_hip_runtime()
    if symbol in _DECODE_SYMBOLS:
        # Two-phase split: phase 1 is this same kernel stopping after pass 2 and
        # writing its weights, phase 2 accumulates the weighted V sum over a
        # slice of the key range, and a combine sums the slices. Selected by
        # context length, because the kernel's parallelism is structurally low
        # (grid = tokens * num_heads is 16 blocks on a 96-CU GPU).
        slices = decode_slices(key_count, head_dim)
        workspace = 0
        temporary = Gemma4AttentionScratch() if slices > 1 and scratch is None else None
        owner = scratch if scratch is not None else temporary
        try:
            if slices > 1:
                workspace = owner.buffer(
                    split_workspace_bytes(
                        tokens, num_heads, head_dim, key_count, slices, library=library
                    ),
                    stream=stream, runtime=runtime,
                ).ptr
            fn = signed_kernel_fn(library, symbol, _ARGTYPES_DECODE, ctypes.c_int)
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
                ctypes.c_void_p(workspace),
                ctypes.c_int(slices),
            )
            _check_launch(runtime, err)
        finally:
            # Convenience callers without reusable ownership release only after
            # the stream is quiescent, including a partial launch failure.
            if temporary is not None:
                temporary.close()
        return
    fn = signed_kernel_fn(library, symbol, _ARGTYPES_PREFILL, ctypes.c_int)
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
    _check_launch(runtime, err)


def gemma4_attention_prefill_bf16(
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
) -> None:
    """Masked, ungated prefill attention over ``tokens`` queries.

    ``query`` is ``(tokens, num_heads, head_dim)`` BF16, ``key`` and ``value`` are
    ``(keys, num_kv_heads, head_dim)`` BF16, ``keep_mask`` is a
    ``(tokens, keys)`` uint8 keep-mask, and ``out`` is
    ``(tokens, num_heads, head_dim)`` BF16. Pass ``scale=1.0`` for Gemma 4.

    ``keys`` defaults to ``tokens``, which is the dense prefill case where the
    block being attended is the block of queries. Passing a larger ``keys`` with
    ``key``/``value`` pointing at a KV cache gives the decode step: one query row
    attending over the live context. At ``tokens == 1`` this routes to the
    decode kernel (:func:`attention_symbol`); its output is bit-identical to the
    block kernel's.

    ``window`` and ``row_offset`` let a row skip its leading masked columns.
    ``row_offset`` is the block's start in the mask's column frame, so row
    ``t``'s own position is ``row_offset + t``; with ``window > 0`` the walk
    starts at column ``max(0, row_offset + t - window + 1)``. That is only
    sound when the mask is zero below that column, which a sliding causal mask
    guarantees and an arbitrary caller-supplied mask does not -- so ``window``
    defaults to 0, meaning no promise and a walk from column 0.
    """

    _launch_prefill(
        attention_symbol("bf16", tokens=tokens, head_dim=head_dim),
        query_ptr,
        key_ptr,
        value_ptr,
        keep_mask_ptr,
        out_ptr,
        tokens=tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
        keys=keys,
        stream=stream,
        library=library,
        runtime=runtime,
        scratch=scratch,
        window=window,
        row_offset=row_offset,
    )


def gemma4_attention_prefill_f32(
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
) -> None:
    """F32 entry point, for validating against the f32 CPU reference.

    The reference works in f32, so comparing through a BF16 round trip would
    measure the rounding rather than the kernel. Routed like the BF16 wrapper:
    ``tokens == 1`` selects the decode kernel. ``window``/``row_offset`` carry
    the same meaning as in :func:`gemma4_attention_prefill_bf16`.
    """

    _launch_prefill(
        attention_symbol("f32", tokens=tokens, head_dim=head_dim),
        query_ptr,
        key_ptr,
        value_ptr,
        keep_mask_ptr,
        out_ptr,
        tokens=tokens,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        scale=scale,
        keys=keys,
        stream=stream,
        library=library,
        runtime=runtime,
        scratch=scratch,
        window=window,
        row_offset=row_offset,
    )


def register_gemma4_attention_kernels(*, replace: bool = False) -> None:
    """Register the Gemma 4 prefill attention family against the four-axis registry."""

    for quant in PREFILL_ATTENTION_QUANTS:
        register(
            KernelKey("hip_gfx1100", "prefill_attention", quant, PREFILL_ATTENTION_PLAIN),
            gemma4_attention_prefill_bf16,
            replace=replace,
        )


# --- prefill-attention variant selection -----------------------------------
#
# The strict kernel is the family's only unconditional entry point. A second
# variant exists (``gemma4_wmma_flash``, a BF16 WMMA flash prefill for the
# sliding geometry) and an execution profile may select it. Selection is a
# capability match against what the variant declares it implements -- never a
# model name, artifact path, hash, or an enumerated list of known-good inputs --
# and a miss falls back to the strict kernel with a reason rather than raising,
# because a profile is a performance decision and not a licence to fail a
# request.

PREFILL_ATTENTION_PLAIN = "gemma4_plain"
PREFILL_ATTENTION_WMMA_FLASH = "gemma4_wmma_flash"
PREFILL_ATTENTION_QUANTS = ("gguf_q4_k_m", "gguf_q4_k_xl", "gguf_q8_0", "gguf")


@dataclass(frozen=True, slots=True)
class PrefillAttentionSelection:
    """One resolved prefill-attention launcher plus why it was chosen."""

    variant: str
    launcher: Callable[..., int]
    requested_variant: str | None
    reason: str

    @property
    def is_strict(self) -> bool:
        return self.variant == PREFILL_ATTENTION_PLAIN

    def describe(self) -> str:
        """One line naming the variant and the reason, for diagnostics."""

        requested = self.requested_variant or PREFILL_ATTENTION_PLAIN
        return f"prefill_attention={self.variant} requested={requested} ({self.reason})"


def _select_prefill_attention(
    *,
    requested_variant: str | None = None,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> PrefillAttentionSelection:
    """Resolve a prefill-attention variant for one head geometry.

    ``requested_variant`` is what an execution profile selected, or ``None``
    for the strict kernel. The variant is admitted only where it declares the
    geometry implemented; every other input keeps the strict kernel, which
    covers every geometry the family serves.
    """

    if requested_variant in (None, "", PREFILL_ATTENTION_PLAIN):
        return PrefillAttentionSelection(
            variant=PREFILL_ATTENTION_PLAIN,
            launcher=gemma4_attention_prefill_bf16,
            requested_variant=requested_variant,
            reason="strict",
        )
    if requested_variant != PREFILL_ATTENTION_WMMA_FLASH:
        return PrefillAttentionSelection(
            variant=PREFILL_ATTENTION_PLAIN,
            launcher=gemma4_attention_prefill_bf16,
            requested_variant=requested_variant,
            reason=f"unknown variant {requested_variant!r}",
        )
    # Imported here so the candidate's module -- and the build of its .so -- is
    # only reached once something actually asks for it.
    from .gemma4_attention_prefill_wmma import (
        gemma4_attention_prefill_wmma_bf16,
        gemma4_attention_prefill_wmma_supported,
    )

    if not gemma4_attention_prefill_wmma_supported(
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    ):
        return PrefillAttentionSelection(
            variant=PREFILL_ATTENTION_PLAIN,
            launcher=gemma4_attention_prefill_bf16,
            requested_variant=requested_variant,
            reason=(
                f"capability miss: {PREFILL_ATTENTION_WMMA_FLASH} implements head_dim "
                f"{HEAD_DIM_WMMA} with GQA ratio {GQA_RATIO_WMMA}, got head_dim {head_dim} "
                f"with {num_heads}q/{num_kv_heads}kv"
            ),
        )
    return PrefillAttentionSelection(
        variant=PREFILL_ATTENTION_WMMA_FLASH,
        launcher=gemma4_attention_prefill_wmma_bf16,
        requested_variant=requested_variant,
        reason="capability match",
    )


# The candidate's declared geometry, mirrored so the refusal above can name it
# without importing the candidate's module.
HEAD_DIM_WMMA = 256
GQA_RATIO_WMMA = 2

# One line per distinct (variant, head_dim), on stderr, when asked. The profile
# resolves to a *request*; only the layer knows the geometry, so this is the
# only place that can report what actually ran. A path that silently falls back
# while passing its own targeted test is a defect, and this is what makes the
# fallback visible.
PREFILL_ATTENTION_LOG_ENV = "HIPENGINE_GEMMA4_PREFILL_ATTENTION_LOG"
_logged_selections: set[tuple[str, int]] = set()


def select_prefill_attention(
    *,
    requested_variant: str | None = None,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> PrefillAttentionSelection:
    """Resolve a prefill-attention variant for one head geometry, and report it.

    The selection itself is :func:`_select_prefill_attention`; this wrapper adds
    the one-shot diagnostic the log env var turns on.
    """

    selection = _select_prefill_attention(
        requested_variant=requested_variant,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    if os.environ.get(PREFILL_ATTENTION_LOG_ENV, "").strip():
        marker = (selection.variant, int(head_dim))
        if marker not in _logged_selections:
            _logged_selections.add(marker)
            print(
                f"[gemma4-attention] {selection.describe()} "
                f"{num_heads}q/{num_kv_heads}kv head_dim={head_dim}",
                file=sys.stderr,
                flush=True,
            )
    return selection
