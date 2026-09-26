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
from pathlib import Path

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

# Decode adds the two-phase split's scratch pointer and slice count; the prefill
# symbols keep the twelve-argument form.
_ARGTYPES_DECODE = _ARGTYPES_PREFILL + (ctypes.c_void_p, ctypes.c_int)

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
    """Key slices the two-phase decode split uses at this context length.

    Two measured effects, and they point in different directions.

    *Whether* to split at all is decided by the split's fixed cost - a weights
    round trip through global memory, two extra kernel launches and a combine -
    which needs a long enough context to be paid back. Paired interleaved runs
    put a 512-key context 3.0% *slower* than the single kernel (46.57 vs 48.01
    tok/s, two pairs each), while 1024 keys is 17% faster per launch. So the
    entry threshold is 1024 keys.

    *How many* slices, once splitting, is a pure parallelism question, and the
    answer depends on the head width, because the key tile the kernel stages
    holds half as many keys when ``head_dim`` is 512 as when it is 256:

    - ``head_dim`` 256 (Gemma 4's sliding layers): 4 slices as soon as the split
      is entered. At 1024 keys, 4 slices measure 192.2 us against 244.4 for 2
      (paired, two passes per arm) - 21% faster. The 512-keys-per-slice floor
      the wide geometry needs is the wrong rule here: a narrow head leaves each
      slice short of work long before the key count is short.
    - ``head_dim`` 512 (the ``attention_k_eq_v`` layers): the doubling rule
      below, which keeps more than 512 keys per slice. At a 1024-token prompt
      the row measures 45.33 tok/s with 4 slices over keys 1025-1151 against
      41.79 with 2.

    The narrow-head case became load-bearing when the caller began handing a
    sliding layer only the keys its window can keep: those layers now pass
    exactly 1024 keys where they used to pass the whole live context, and the
    difference between 2 and 4 slices at that length is the whole 1.3 ms per
    step the read range wins back.

    Its execution-profile gate passed on 2026-09-25 - kl_max 0.0067 against the
    0.05 bar with zero top-1 flips over 1023 teacher-forced rows.
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

    ``"class"`` is the key-class shuffle-tree kernel (``head_dim`` 256 or 512),
    ``"block"`` is the original 256-thread block kernel. Introspection only: the
    two paths are bit-identical, so this is how callers and tests confirm which
    one ran rather than inferring it from timings.
    """

    library = library or build_gemma4_attention(load=True)
    fn = signed_kernel_fn(library, _SYMBOL_DECODE_VARIANT, [], ctypes.c_int)
    return "class" if int(fn()) == 1 else "block"


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
) -> None:
    """F32 entry point, for validating against the f32 CPU reference.

    The reference works in f32, so comparing through a BF16 round trip would
    measure the rounding rather than the kernel. Routed like the BF16 wrapper:
    ``tokens == 1`` selects the decode kernel.
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
    )


def register_gemma4_attention_kernels(*, replace: bool = False) -> None:
    """Register the Gemma 4 prefill attention family against the four-axis registry."""

    for quant in ("gguf_q4_k_m", "gguf_q4_k_xl", "gguf_q8_0", "gguf"):
        register(
            KernelKey("hip_gfx1100", "prefill_attention", quant, "gemma4_plain"),
            gemma4_attention_prefill_bf16,
            replace=replace,
        )
