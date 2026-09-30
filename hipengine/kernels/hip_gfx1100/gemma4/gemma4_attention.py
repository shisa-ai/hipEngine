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

A third path serves the sliding layers on the default route:
:func:`gemma4_attention_prefill_aotriton` runs the vendored AOTriton flash kernel
when the keep-mask the exact kernel would read is *exactly causal* and the head
dim has a vendored image. It is a different association (online softmax over
tiles) and its own numerical contract, so callers opt in explicitly rather than
having it inferred from the geometry.

Importing this module registers ctypes launch wrappers but does not build or load
ROCm until a wrapper is called.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
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
_SYMBOL_FLASH_WORKSPACE_BYTES = "hipengine_gemma4_decode_flash_workspace_bytes"
_SYMBOL_DECODE_SELECTION = "hipengine_gemma4_decode_selection"

# Head dims the committed AOTriton runtime carries images for. The vendored tree
# is pruned to BF16 head_dim=256 gfx11xx forward-attention images, which is the
# sliding-layer geometry; the head_dim=512 global layers have no image and stay
# on the exact kernel.
_AOTRITON_PREFILL_HEAD_DIMS = (256,)
_AOTRITON_LIBRARY: ctypes.CDLL | None = None
_AOTRITON_RUNTIME_AVAILABLE: bool | None = None


def aotriton_prefill_head_dims() -> tuple[int, ...]:
    """Head dims the vendored AOTriton flash-attention image set covers."""

    return _AOTRITON_PREFILL_HEAD_DIMS


def aotriton_prefill_available(head_dim: int) -> bool:
    """True when the vendored AOTriton runtime can serve ``head_dim``.

    Availability is a property of the runtime tree, not of the request, so it is
    resolved once and remembered. A missing tree is a missing optional dependency
    for this one path, and the caller's fallback is the exact kernel.
    """

    global _AOTRITON_RUNTIME_AVAILABLE
    if int(head_dim) not in _AOTRITON_PREFILL_HEAD_DIMS:
        return False
    if _AOTRITON_RUNTIME_AVAILABLE is None:
        from hipengine.kernels.hip_gfx1100.attention.aotriton import (
            AotritonNotInstalledError,
            aotriton_runtime_tree,
        )

        try:
            aotriton_runtime_tree()
        except AotritonNotInstalledError:
            _AOTRITON_RUNTIME_AVAILABLE = False
        else:
            _AOTRITON_RUNTIME_AVAILABLE = True
    return _AOTRITON_RUNTIME_AVAILABLE


def aotriton_prefill_library() -> ctypes.CDLL:
    """The loaded AOTriton C shim, built from the vendored tree on first use."""

    global _AOTRITON_LIBRARY
    if _AOTRITON_LIBRARY is None:
        from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import (
            build_aotriton_wrap,
        )

        _AOTRITON_LIBRARY = build_aotriton_wrap(load=True)
    return _AOTRITON_LIBRARY


def aotriton_prefill_admits(
    *,
    rows: int,
    keys: int,
    head_dim: int,
    sliding_window: int | None,
    mask_is_causal: bool,
    available: Callable[[int], bool] | None = None,
) -> bool:
    """True when the flash path may stand in for the exact kernel here.

    Four separate questions, and the answer is yes only when all four hold:

    * **The mask is causal.** ``mask_is_causal`` is the caller's assertion about
      the mask it built. Without it a window bound or an eviction mask would be
      silently dropped, because the flash kernel reads no mask at all.
    * **There is more than one query row.** A single row is a decode step, which
      the decode kernel already serves; the flash path's per-call scratch and
      persistent atomic counter are not worth paying for one row.
    * **The window is not binding.** A sliding mask is exactly causal while the
      attended range is no wider than the window, so the bound has to be
      vacuous for the whole block -- ``keys <= sliding_window``.
    * **The runtime has an image for this head dim.** The vendored set is BF16
      head_dim 256 only.

    ``available`` overrides the runtime probe, which is what lets the policy be
    exercised without a GPU.
    """

    if not mask_is_causal:
        return False
    if int(rows) <= 1:
        return False
    if int(keys) < int(rows):
        return False
    if sliding_window is not None and int(keys) > int(sliding_window):
        return False
    probe = aotriton_prefill_available if available is None else available
    return bool(probe(int(head_dim)))
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
        self._named: dict[tuple[int, str], DeviceBuffer] = {}
        self._u32: dict[tuple[int, str], tuple[int, int]] = {}
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

    def named_buffer(
        self, name: str, nbytes: int, *, stream: int, runtime: HipRuntime
    ) -> DeviceBuffer:
        """A buffer for one named purpose on one stream, grown in place.

        :meth:`buffer` hands out a single allocation per stream because the split
        path reuses one workspace across a call. The AOTriton path needs several
        live at once -- the log-sum-exp row, both cumulative-sequence tables, and
        the persistent atomic counter -- so those are keyed by purpose instead.
        """

        if self._closed:
            raise RuntimeError("attention scratch is closed")
        if nbytes <= 0:
            raise ValueError("attention scratch size must be positive")
        if self._runtime is not None and runtime is not self._runtime:
            raise ValueError("attention scratch cannot change runtime")
        device = runtime.current_device()
        if self._device is not None and device != self._device:
            raise ValueError("attention scratch cannot change device")
        key = (stream, name)
        previous = self._named.get(key)
        if previous is not None and previous.nbytes >= nbytes:
            return previous
        capacity = max(nbytes, 2 * previous.nbytes) if previous else nbytes
        buffer = malloc(capacity, runtime=runtime)
        self._runtime, self._device = runtime, device
        self._owned.append(buffer)
        self._named[key] = buffer
        return buffer

    def upload_u32_pair(
        self, name: str, values: tuple[int, int], *, stream: int, runtime: HipRuntime
    ) -> DeviceBuffer:
        """Two int32 values in a named buffer, re-uploaded only when they change.

        ``copy_host_to_device`` is a synchronous ``hipMemcpy``, and the flash
        path needs the same two cumulative-sequence entries for every layer of a
        block. Re-uploading them per layer would put a device sync inside the
        layer loop, so the buffer remembers the pair it already holds. The
        remembered value is per (stream, name), which is the same key the buffer
        itself uses, so two streams never share an assumption.
        """

        from hipengine.core.memory import copy_host_to_device, host_buffer_ptr

        key = (stream, name)
        buffer = self.named_buffer(name, 2 * 4, stream=stream, runtime=runtime)
        if self._u32.get(key) == values:
            return buffer
        payload = (ctypes.c_int32 * 2)(*values)
        copy_host_to_device(buffer, host_buffer_ptr(payload), 8, runtime=runtime)
        self._u32[key] = values
        return buffer

    def close(self) -> None:
        if self._closed:
            return
        if self._runtime is not None:
            if self._runtime.current_device() != self._device:
                raise ValueError("attention scratch must close on its owning device")
            for stream in self._current:
                self._runtime.stream_synchronize(stream)
            for stream, _ in self._named:
                self._runtime.stream_synchronize(stream)
            while self._owned:
                free(self._owned[-1], runtime=self._runtime)
                self._owned.pop()
        self._current.clear()
        self._named.clear()
        self._u32.clear()
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

    0 = block kernel, 1 = key-class single kernel, 2 = key-class two-phase
    split, 3 = flash-decoding. Introspection for confirming that the intended
    path ran: the four are numerically close by design, so a timing or a parity
    test cannot tell them apart, and a caller who needs to know which one a
    real request took needs this rather than an inference.
    """

    library = library or build_gemma4_attention(load=True)
    fn = signed_kernel_fn(library, _SYMBOL_DECODE_SELECTION, (), ctypes.c_int)
    return int(fn())


def flash_slices(keys: int) -> int:
    """KV slices the flash-decoding route asks for.

    Single source of truth: the host wrapper trusts this count through the
    negative-slices request and lays its partials out from it, so a mismatch
    would be memory corruption rather than a slow path -- the unit battery
    pins the values. One slice per ~32 keys keeps the block count at the
    occupancy target (256 blocks at the 1024 entry threshold) while the cap
    bounds the per-block LDS logits at context maximum.
    """

    if keys <= 0:
        raise ValueError("keys must be positive")
    per_slice = 32
    return min(64, max(4, -(-int(keys) // per_slice)))


def flash_admits(
    *, tokens: int, head_dim: int, num_heads: int, num_kv_heads: int
) -> bool:
    """Capability admission for the flash decode route.

    The kernel is instantiated for exactly the two packings the model has:
    head_dim 256 with GQA ratio 2 (sliding layers, one 256-lane warp tree
    over the row, two query heads sharing K and V) and head_dim 512 with
    GQA ratio 8 (global layers, two 256-dim halves folded into the same
    tree). Anything else -- another head_dim, another ratio, a multi-token
    block -- keeps the incumbent split/class chain, which is the strict
    fallback flash rejects into as well; the C-side dispatcher re-checks the
    same two shapes and returns the capability miss on anything else. No
    identity, path, or measured-performance terms: if the kernels can
    execute the shape, it is admitted.
    """

    if tokens != 1:
        return False
    if num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        return False
    ratio = num_heads // num_kv_heads
    return (head_dim, ratio) in ((256, 2), (512, 8))


def flash_workspace_bytes(
    tokens: int,
    num_heads: int,
    head_dim: int,
    slices: int,
    *,
    library: ctypes.CDLL | None = None,
) -> int:
    """Bytes of scratch the flash partials need, from the kernel's definition."""

    library = library or build_gemma4_attention(load=True)
    fn = signed_kernel_fn(
        library,
        _SYMBOL_FLASH_WORKSPACE_BYTES,
        (ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int),
        ctypes.c_size_t,
    )
    return int(fn(int(tokens), int(num_heads), int(head_dim), int(slices)))


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
    weights and dimension-partitioned value passes, ``"flash"`` is the
    KV-sliced flash route with its combine, and ``"block"`` is the original
    block kernel. Introspection only; selection is not a quality gate.
    """

    library = library or build_gemma4_attention(load=True)
    fn = signed_kernel_fn(library, _SYMBOL_DECODE_VARIANT, [], ctypes.c_int)
    return {0: "block", 1: "class", 2: "split", 3: "flash"}[int(fn())]


def gemma4_attention_shared_bytes(*, head_dim: int, keys: int) -> int:
    """Validate the resident-logit kernel's gfx1100 shared-memory requirement.

    This implementation stores one logit per live key in LDS. Larger contexts
    need a tiled attention implementation, not a larger prefill scratch block.
    """

    if head_dim <= 0 or keys <= 0:
        raise ValueError("head_dim and keys must be positive")
    threads = min(256, 1 << (int(head_dim) - 1).bit_length())
    # The multi-token route runs the decode family, whose key-class kernel holds
    # the logits plus a 256-lane partial per 256-thread group plus one max slot
    # per warp: keys + 512 + 16 floats at its 512-thread block. Take the larger
    # of that and the block kernel's own requirement.
    required = max(
        (int(head_dim) + int(keys) + threads) * 4,
        (int(keys) + 256 * 2 + 16) * 4,
    )
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
        # Flash-decoding (D2) replaces the split for the sliding geometry when
        # admitted: a negative slice count is the flash request, and the
        # workspace is sized for whichever route is larger so a host-side
        # rejection can fall through to the split without reallocating.
        flash = slices > 1 and flash_admits(
            tokens=tokens,
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        request = -flash_slices(key_count) if flash else slices
        workspace = 0
        temporary = Gemma4AttentionScratch() if slices > 1 and scratch is None else None
        owner = scratch if scratch is not None else temporary
        try:
            if slices > 1:
                need = split_workspace_bytes(
                    tokens, num_heads, head_dim, key_count, slices, library=library
                )
                if flash:
                    need = max(
                        need,
                        flash_workspace_bytes(
                            tokens, num_heads, head_dim, flash_slices(key_count),
                            library=library,
                        ),
                    )
                workspace = owner.buffer(
                    need,
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
                ctypes.c_int(request),
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


def gemma4_attention_prefill_aotriton(
    query_ptr: int,
    key_ptr: int,
    value_ptr: int,
    out_ptr: int,
    *,
    tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    keys: int | None = None,
    window_left: int | None = None,
    window_right: int | None = None,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    scratch: Gemma4AttentionScratch | None = None,
) -> None:
    """Causal prefill attention through the vendored AOTriton flash kernel.

    The same tensors as :func:`gemma4_attention_prefill_bf16` except that there
    is no ``keep_mask``: this path takes only the causal geometry. **The caller
    is asserting that the mask the exact kernel would read is exactly
    ``key <= query``.** A sliding-window mask is causal only while the window is
    at least as wide as the attended range, which is why the layer computes that
    condition rather than inferring it here.

    The arithmetic is not the exact kernel's: AOTriton runs the standard flash
    online softmax over key tiles, so the row maximum, the denominator and the
    value accumulation are associated differently. It is a numerically
    equivalent formulation, not a bit-identical one, and it carries its own
    execution-profile gate.

    ``query`` is ``(tokens, num_heads, head_dim)`` BF16, ``key``/``value`` are
    ``(keys, num_kv_heads, head_dim)`` BF16, and ``out`` is
    ``(tokens, num_heads, head_dim)`` BF16. ``scale`` is Gemma 4's
    ``geometry.scale`` (1.0).
    """

    from hipengine.core.dtype import DType
    from hipengine.kernels.hip_gfx1100.attention.aotriton_wrap import (
        aotriton_attn_fwd_v3_compact_varlen,
        tensor1,
        tensor2,
        tensor4,
    )

    _check_prefill_shape(tokens, num_heads, num_kv_heads, head_dim)
    if int(head_dim) not in _AOTRITON_PREFILL_HEAD_DIMS:
        raise ValueError(
            f"AOTriton prefill has no vendored image for head_dim {head_dim}; "
            f"available: {list(_AOTRITON_PREFILL_HEAD_DIMS)}"
        )
    if tokens <= 1:
        raise ValueError("AOTriton prefill requires more than one query row")
    key_count = tokens if keys is None else int(keys)
    if key_count <= 0:
        raise ValueError("keys must be positive")
    if key_count < tokens:
        raise ValueError("keys must cover the query rows")
    runtime = runtime or get_hip_runtime()
    library = library or aotriton_prefill_library()

    temporary = Gemma4AttentionScratch() if scratch is None else None
    owner = scratch if scratch is not None else temporary
    try:
        lse = owner.named_buffer(
            "aotriton_lse", num_heads * tokens * 4, stream=stream, runtime=runtime
        )
        atomic = owner.named_buffer(
            "aotriton_atomic", 4, stream=stream, runtime=runtime
        )
        cu_q = owner.upload_u32_pair(
            "aotriton_cu_q", (0, tokens), stream=stream, runtime=runtime
        )
        cu_k = owner.upload_u32_pair(
            "aotriton_cu_k", (0, key_count), stream=stream, runtime=runtime
        )

        # Q and out are both ``(tokens, heads, head_dim)`` contiguous, and K/V are
        # ``(keys, kv_heads, head_dim)`` contiguous. AOTriton wants rank-4 views,
        # so the leading batch axis is a size-1 axis over the same bytes.
        q_tensor = tensor4(
            query_ptr,
            (1, num_heads, tokens, head_dim),
            (num_heads * head_dim * tokens, head_dim, num_heads * head_dim, 1),
            DType.BF16,
        )
        kv_strides = (
            num_kv_heads * head_dim * key_count,
            head_dim,
            num_kv_heads * head_dim,
            1,
        )
        k_tensor = tensor4(
            key_ptr, (1, num_kv_heads, key_count, head_dim), kv_strides, DType.BF16
        )
        v_tensor = tensor4(
            value_ptr, (1, num_kv_heads, key_count, head_dim), kv_strides, DType.BF16
        )
        out_tensor = tensor4(
            out_ptr,
            (1, num_heads, tokens, head_dim),
            (num_heads * head_dim * tokens, head_dim, num_heads * head_dim, 1),
            DType.BF16,
        )
        lse_tensor = tensor2(lse.ptr, (num_heads, tokens), (tokens, 1), DType.FP32)
        cu_q_tensor = tensor1(cu_q.ptr, (2,), (1,), DType.INT32)
        cu_k_tensor = tensor1(cu_k.ptr, (2,), (1,), DType.INT32)

        # V3 is the wrapper that carries bottom-right-aligned causal semantics,
        # which is what a query block that is a suffix of the attended range
        # needs. The persistent atomic counter is zeroed by the shim before the
        # launch.
        # Window bounds are forwarded only when the caller supplies them. With
        # neither set nothing extra is passed and the wrapper's own defaults
        # apply, so the production path is unchanged by this signature.
        window_kwargs: dict[str, int] = {}
        if window_left is not None:
            window_kwargs["window_left"] = window_left
        if window_right is not None:
            window_kwargs["window_right"] = window_right

        aotriton_attn_fwd_v3_compact_varlen(
            q_tensor,
            k_tensor,
            v_tensor,
            cu_q_tensor,
            cu_k_tensor,
            lse_tensor,
            out_tensor,
            persistent_atomic_counter_ptr=atomic.ptr,
            max_seqlen_q=tokens,
            max_seqlen_k=key_count,
            sm_scale=float(scale),
            is_causal=True,
            **window_kwargs,
            stream=stream,
            library=library,
            runtime=runtime,
        )
    finally:
        if temporary is not None:
            temporary.close()


def register_gemma4_attention_kernels(*, replace: bool = False) -> None:
    """Register the Gemma 4 prefill attention family against the four-axis registry.

    ``gemma4_plain`` is the correctness-first block kernel, which accepts any
    shape; ``gemma4_tiled`` is the head_dim 512 flash path. The two are
    disjoint by shape, so registration order never decides between them -- the
    layer routes by capability and falls back to ``gemma4_plain`` for anything
    the tiled kernel cannot execute.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_tiled import (
        gemma4_attention_prefill_tiled,
    )

    for quant in ("gguf_q4_k_m", "gguf_q4_k_xl", "gguf_q8_0", "gguf"):
        register(
            KernelKey("hip_gfx1100", "prefill_attention", quant, "gemma4_plain"),
            gemma4_attention_prefill_bf16,
            replace=replace,
        )
        register(
            KernelKey("hip_gfx1100", "prefill_attention", quant, "gemma4_tiled"),
            gemma4_attention_prefill_tiled,
            replace=replace,
        )
