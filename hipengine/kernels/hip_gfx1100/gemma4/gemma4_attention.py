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
# Split scratch, grown and reused rather than freed per launch: freeing under an
# in-flight stream is a use-after-free and a fresh hipMalloc would become a
# per-launch cost, the same convention the runtime's staging buffers use. Keyed
# by stream, because two streams sharing one buffer would interleave phase 1 of
# one launch with phase 2 of another. The owner objects are held so the arena
# cannot collect them.
_SPLIT_WORKSPACES: dict = {}


def decode_slices(keys: int) -> int:
    """Key slices the two-phase decode split uses at this context length.

    The split exists because the decode kernel's parallelism is structurally low:
    ``grid = tokens * num_heads`` launches 16 blocks on a 96-CU GPU, and
    ``scripts/gemma4_attention_scale_probe.py`` measures per-head time improving
    6.9x going from 16 to 64 blocks while 128 blocks is worse in total time than
    64. One slice per ~512 keys puts a 1024-key row at 2 slices (32 blocks) and a
    2048-key row at 4 (64 blocks, the measured optimum). Below 512 keys the
    single-kernel path already has enough work per block to amortise its launch,
    and the split would pay two extra kernels and a workspace round trip for
    nothing.

    Its execution-profile gate passed on 2026-09-25 - kl_max 0.0067 against the
    0.05 bar with zero top-1 flips over 1023 teacher-forced rows - so this is
    the default decode path above 512 keys.
    """

    if keys < 512:
        return 1
    slices = 1
    while slices < 4 and keys > slices * 512:
        slices <<= 1
    return slices


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


def _split_workspace_ptr(nbytes: int, stream: int) -> int:
    """Return a scratch pointer of at least ``nbytes`` for this stream."""

    entry = _SPLIT_WORKSPACES.get(stream)
    if entry is not None and entry[1] >= nbytes:
        return entry[0].ptr
    from hipengine.core.memory import malloc

    buffer = malloc(int(nbytes))
    _SPLIT_WORKSPACES[stream] = (buffer, int(nbytes))
    return buffer.ptr


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
        slices = decode_slices(key_count)
        workspace = 0
        if slices > 1:
            workspace = _split_workspace_ptr(
                split_workspace_bytes(
                    tokens, num_heads, head_dim, key_count, slices, library=library
                ),
                stream,
            )
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
    )


def register_gemma4_attention_kernels(*, replace: bool = False) -> None:
    """Register the Gemma 4 prefill attention family against the four-axis registry."""

    for quant in ("gguf_q4_k_m", "gguf_q4_k_xl", "gguf_q8_0", "gguf"):
        register(
            KernelKey("hip_gfx1100", "prefill_attention", quant, "gemma4_plain"),
            gemma4_attention_prefill_bf16,
            replace=replace,
        )
