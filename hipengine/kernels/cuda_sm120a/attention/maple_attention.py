"""Raw-pointer Maple QK/RoPE/KV/attention wrappers over KVLiveSpans."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_cuda, plan_cuda_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.cuda import CUDA_SUCCESS, CudaRuntime, get_cuda_runtime
from hipengine.kernels.backends import cuda_target_arch_for_backend
from hipengine.kernels.registry import KernelKey, register
from hipengine.kvcache import KVLiveSpans

_SOURCE = Path(__file__).with_name("maple_attention.cu")
_OUTPUT_NAME = "maple_attention.so"
_QUANT = "maple_ternary2"
_BACKEND = "cuda_sm120a"
_TARGET_ARCH = cuda_target_arch_for_backend(_BACKEND)
_PTR = ctypes.c_void_p
_I64 = ctypes.c_int64
_F32 = ctypes.c_float


def plan_maple_attention_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_cuda_build(
        sources=[_SOURCE],
        family="cuda_sm120a_maple_attention",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
    )


def build_maple_attention(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_cuda(
        sources=[_SOURCE],
        family="cuda_sm120a_maple_attention",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        target_arch=_TARGET_ARCH,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def maple_kv_span_update(
    spans: KVLiveSpans,
    *,
    position: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    _, live, token_positions, evict_mask, row_positions, capacity = _span_pointers(spans)
    _launch(
        "hipengine_maple_kv_span_update",
        (_PTR, _PTR, _PTR, _PTR, _I64, _I64, _PTR),
        (live, token_positions, evict_mask, row_positions, position, capacity),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_kv_span_update_batched(
    spans: KVLiveSpans,
    *,
    start: int,
    rows: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Publish positions [start, start+rows) into the ring (P4 prefill)."""

    _, live, token_positions, evict_mask, row_positions, capacity = _span_pointers(spans)
    _launch(
        "hipengine_maple_kv_span_update_batched",
        (_PTR, _PTR, _PTR, _PTR, _I64, _I64, _I64, _PTR),
        (live, token_positions, evict_mask, row_positions, start, rows, capacity),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_qknorm_rope_kv_write_bf16(
    qkv_ptr: int,
    q_norm_weight_ptr: int,
    k_norm_weight_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
    rope_theta: float,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    base, live, token_positions, evict_mask, row_positions, capacity = _span_pointers(spans)
    _launch(
        "hipengine_maple_qknorm_rope_kv_write_bf16",
        (_PTR,) * 10 + (_I64, _I64, _I64, _I64, _F32, _F32, _I64, _PTR),
        (
            qkv_ptr,
            q_norm_weight_ptr,
            k_norm_weight_ptr,
            key_cache_ptr,
            value_cache_ptr,
            base,
            live,
            token_positions,
            evict_mask,
            row_positions,
            q_heads,
            kv_heads,
            head_dim,
            rope_dim,
            eps,
            rope_theta,
            capacity,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_qknorm_rope_kv_write_batched_bf16(
    qkv_ptr: int,
    q_norm_weight_ptr: int,
    k_norm_weight_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
    rope_theta: float,
    start: int,
    rows: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Batched qknorm+RoPE+KV write for T prompt rows into the shared ring (P2)."""

    base, live, token_positions, evict_mask, _row_positions, capacity = _span_pointers(spans)
    _launch(
        "hipengine_maple_qknorm_rope_kv_write_batched_bf16",
        (_PTR,) * 9 + (_I64, _I64, _I64, _I64, _F32, _F32, _I64, _I64, _I64, _PTR),
        (
            qkv_ptr,
            q_norm_weight_ptr,
            k_norm_weight_ptr,
            key_cache_ptr,
            value_cache_ptr,
            base,
            live,
            token_positions,
            evict_mask,
            q_heads,
            kv_heads,
            head_dim,
            rope_dim,
            eps,
            rope_theta,
            start,
            rows,
            capacity,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_qknorm_rope_kv_write_batched_decode_bf16(
    qkv_ptr: int,
    q_norm_weight_ptr: int,
    k_norm_weight_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    spans: KVLiveSpans,
    *,
    row_base_offsets: int,
    rows: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
    rope_theta: float,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Batched per-request QK-norm+RoPE+KV write (D5 batch decode).

    Each row uses its own row_positions[row] (local RoPE position) and
    row_base_offsets[row] (arena base) to place K/V into a shared arena while
    keeping per-request local position encoding.
    """

    base, live, token_positions, evict_mask, row_positions, capacity = _span_pointers(spans)
    _launch(
        "hipengine_maple_qknorm_rope_kv_write_batched_decode_bf16",
        (_PTR,) * 11 + (_I64, _I64, _I64, _I64, _F32, _F32, _I64, _I64, _PTR),
        (
            qkv_ptr,
            q_norm_weight_ptr,
            k_norm_weight_ptr,
            key_cache_ptr,
            value_cache_ptr,
            base,
            live,
            token_positions,
            evict_mask,
            row_positions,
            row_base_offsets,
            q_heads,
            kv_heads,
            head_dim,
            rope_dim,
            eps,
            rope_theta,
            rows,
            capacity,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_attention_decode_bf16(
    qkv_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    base, live, token_positions, evict_mask, row_positions, capacity = _span_pointers(spans)
    _launch(
        "hipengine_maple_attention_decode_bf16",
        (_PTR,) * 9 + (_I64, _I64, _I64, _F32, _I64, _PTR),
        (
            qkv_ptr,
            key_cache_ptr,
            value_cache_ptr,
            out_ptr,
            base,
            live,
            token_positions,
            evict_mask,
            row_positions,
            q_heads,
            kv_heads,
            head_dim,
            scale,
            capacity,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_attention_decode_wave32_exact_bf16(
    qkv_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Exact D128 decode using 32 physical lanes per local128 query head."""

    base, live, token_positions, evict_mask, row_positions, capacity = _span_pointers(spans)
    _launch(
        "hipengine_maple_attention_decode_wave32_exact_bf16",
        (_PTR,) * 9 + (_I64, _I64, _I64, _F32, _I64, _PTR),
        (
            qkv_ptr,
            key_cache_ptr,
            value_cache_ptr,
            out_ptr,
            base,
            live,
            token_positions,
            evict_mask,
            row_positions,
            q_heads,
            kv_heads,
            head_dim,
            scale,
            capacity,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_attention_decode_batched_bf16(
    qkv_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    *,
    row_base_offsets: int,
    rows: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Batched per-request attention decode (D5).

    Each row attends its own live span via per-row row_positions/live_counts,
    with its K/V placed at arena slots row_base_offsets[row] + local position.
    """

    base, live, token_positions, evict_mask, row_positions, capacity = _span_pointers(spans)
    _launch(
        "hipengine_maple_attention_decode_batched_bf16",
        (_PTR,) * 10 + (_I64, _I64, _I64, _I64, _F32, _I64, _PTR),
        (
            qkv_ptr,
            key_cache_ptr,
            value_cache_ptr,
            out_ptr,
            base,
            live,
            token_positions,
            evict_mask,
            row_positions,
            row_base_offsets,
            rows,
            q_heads,
            kv_heads,
            head_dim,
            scale,
            capacity,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_attention_fused_qknorm_decode_bf16(
    qkv_ptr: int,
    q_norm_weight_ptr: int,
    k_norm_weight_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    rope_dim: int,
    eps: float,
    rope_theta: float,
    scale: float,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Fused QK-norm+RoPE+KV-write + online-softmax attention decode (D2).

    One kernel per q-head block. Bit-exact with the unfused qknorm_rope_kv_write
    + attention_decode chain; intended to cut the decode launch count.
    """

    base, live, token_positions, evict_mask, row_positions, capacity = _span_pointers(spans)
    _launch(
        "hipengine_maple_attention_fused_qknorm_decode_bf16",
        (_PTR,) * 11 + (_I64, _I64, _I64, _I64, _F32, _F32, _F32, _I64, _PTR),
        (
            qkv_ptr,
            q_norm_weight_ptr,
            k_norm_weight_ptr,
            key_cache_ptr,
            value_cache_ptr,
            out_ptr,
            base,
            live,
            token_positions,
            evict_mask,
            row_positions,
            q_heads,
            kv_heads,
            head_dim,
            rope_dim,
            eps,
            rope_theta,
            scale,
            capacity,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_attention_prefill_bf16(
    qkv_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    *,
    rows: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Batched causal prefill attention over a dense prefix KV cache (P2)."""

    _launch(
        "hipengine_maple_attention_prefill_bf16",
        (_PTR,) * 4 + (_I64, _I64, _I64, _I64, _F32, _PTR),
        (
            qkv_ptr,
            key_cache_ptr,
            value_cache_ptr,
            out_ptr,
            rows,
            q_heads,
            kv_heads,
            head_dim,
            scale,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_attention_prefill_ring_bf16(
    qkv_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    *,
    rows: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float,
    start: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Batched causal prefill attention over the shared KVLiveSpans ring (P2)."""

    base, _live, _tp, _ev, _rp, capacity = _span_pointers(spans)
    _launch(
        "hipengine_maple_attention_prefill_ring_bf16",
        (_PTR,) * 5 + (_I64, _I64, _I64, _I64, _F32, _I64, _I64, _PTR),
        (
            qkv_ptr,
            key_cache_ptr,
            value_cache_ptr,
            out_ptr,
            base,
            rows,
            q_heads,
            kv_heads,
            head_dim,
            scale,
            start,
            capacity,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def maple_attention_prefill_ring_gqa4_bf16(
    qkv_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    out_ptr: int,
    spans: KVLiveSpans,
    *,
    rows: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    scale: float,
    start: int,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: CudaRuntime | None = None,
) -> None:
    """Exact wave32 GQA4 causal prefill over the complete span ABI (P2)."""

    base, live, token_positions, evict_mask, row_positions, capacity = _span_pointers(
        spans
    )
    _launch(
        "hipengine_maple_attention_prefill_ring_gqa4_bf16",
        (_PTR,) * 9 + (_I64, _I64, _I64, _I64, _F32, _I64, _I64, _PTR),
        (
            qkv_ptr,
            key_cache_ptr,
            value_cache_ptr,
            out_ptr,
            base,
            live,
            token_positions,
            evict_mask,
            row_positions,
            rows,
            q_heads,
            kv_heads,
            head_dim,
            scale,
            start,
            capacity,
        ),
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_maple_attention_kernels(
    *,
    backend: str = _BACKEND,
    replace: bool = True,
) -> None:
    kernels = {
        ("maple_kv_span_update", "sliding_ring"): maple_kv_span_update,
        ("maple_kv_span_update", "sliding_ring_batched"): maple_kv_span_update_batched,
        (
            "maple_qknorm_rope_kv_write",
            "partial_rotate_half_bf16",
        ): maple_qknorm_rope_kv_write_bf16,
        (
            "maple_qknorm_rope_kv_write",
            "partial_rotate_half_batched_bf16",
        ): maple_qknorm_rope_kv_write_batched_bf16,
        (
            "maple_qknorm_rope_kv_write",
            "partial_rotate_half_batched_decode_bf16",
        ): maple_qknorm_rope_kv_write_batched_decode_bf16,
        ("maple_attention_decode", "gqa_spans_bf16"): maple_attention_decode_bf16,
        (
            "maple_attention_decode",
            "gqa_spans_wave32_exact_bf16",
        ): maple_attention_decode_wave32_exact_bf16,
        (
            "maple_attention_decode",
            "gqa_spans_batched_bf16",
        ): maple_attention_decode_batched_bf16,
        ("maple_attention_prefill", "gqa_causal_bf16"): maple_attention_prefill_bf16,
        (
            "maple_attention_prefill",
            "gqa_causal_ring_bf16",
        ): maple_attention_prefill_ring_bf16,
        (
            "maple_attention_prefill",
            "gqa4_wave32_causal_ring_bf16",
        ): maple_attention_prefill_ring_gqa4_bf16,
    }
    for (layer, variant), kernel in kernels.items():
        register(
            KernelKey(backend, layer, _QUANT, variant),
            kernel,
            replace=replace,
        )


def _span_pointers(spans: KVLiveSpans) -> tuple[int, int, int, int, int, int]:
    if spans.token_positions is None or spans.evict_mask is None or spans.row_positions is None:
        raise ValueError("Maple attention requires token_positions, evict_mask, and row_positions")
    return (
        spans.base_offsets.ptr,
        spans.live_counts.ptr,
        spans.token_positions.ptr,
        spans.evict_mask.ptr,
        spans.row_positions.ptr,
        spans.max_live_count,
    )


def _launch(
    symbol: str,
    argtypes: tuple,
    args: tuple,
    *,
    stream: int,
    library: ctypes.CDLL | None,
    runtime: CudaRuntime | None,
) -> None:
    library = library or build_maple_attention(load=True)
    runtime = runtime or get_cuda_runtime()
    fn = signed_kernel_fn(library, symbol, argtypes, ctypes.c_int)
    err = fn(*args, stream)
    if int(err) != CUDA_SUCCESS:
        runtime.check(int(err))


register_maple_attention_kernels()


__all__ = [
    "build_maple_attention",
    "maple_attention_decode_bf16",
    "maple_attention_decode_wave32_exact_bf16",
    "maple_attention_prefill_bf16",
    "maple_attention_prefill_ring_bf16",
    "maple_attention_prefill_ring_gqa4_bf16",
    "maple_kv_span_update",
    "maple_kv_span_update_batched",
    "maple_qknorm_rope_kv_write_batched_bf16",
    "maple_qknorm_rope_kv_write_bf16",
    "plan_maple_attention_build",
    "register_maple_attention_kernels",
]
