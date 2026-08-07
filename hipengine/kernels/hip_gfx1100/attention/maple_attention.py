"""Raw-pointer Maple QK/RoPE/KV/attention wrappers over KVLiveSpans."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register
from hipengine.kvcache import KVLiveSpans

_SOURCE = Path(__file__).with_name("maple_attention.hip")
_OUTPUT_NAME = "maple_attention.so"
_QUANT = "maple_ternary2"
_PTR = ctypes.c_void_p
_I64 = ctypes.c_int64
_F32 = ctypes.c_float


def plan_maple_attention_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="maple_attention",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
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
    return build_hip(
        sources=[_SOURCE],
        family="maple_attention",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
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
    runtime: HipRuntime | None = None,
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
    runtime: HipRuntime | None = None,
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
    runtime: HipRuntime | None = None,
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
    runtime: HipRuntime | None = None,
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
    runtime: HipRuntime | None = None,
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
    runtime: HipRuntime | None = None,
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


def register_maple_attention_kernels(
    *,
    backend: str = "hip_gfx1100",
    replace: bool = True,
) -> None:
    kernels = {
        ("maple_kv_span_update", "sliding_ring"): maple_kv_span_update,
        (
            "maple_qknorm_rope_kv_write",
            "partial_rotate_half_bf16",
        ): maple_qknorm_rope_kv_write_bf16,
        (
            "maple_qknorm_rope_kv_write",
            "partial_rotate_half_batched_bf16",
        ): maple_qknorm_rope_kv_write_batched_bf16,
        ("maple_attention_decode", "gqa_spans_bf16"): maple_attention_decode_bf16,
        ("maple_attention_prefill", "gqa_causal_bf16"): maple_attention_prefill_bf16,
        ("maple_attention_prefill", "gqa_causal_ring_bf16"): maple_attention_prefill_ring_bf16,
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
    runtime: HipRuntime | None,
) -> None:
    library = library or build_maple_attention(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, symbol, argtypes, ctypes.c_int)
    err = fn(*args, stream)
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


register_maple_attention_kernels()


__all__ = [
    "build_maple_attention",
    "maple_attention_decode_bf16",
    "maple_attention_prefill_bf16",
    "maple_attention_prefill_ring_bf16",
    "maple_kv_span_update",
    "maple_qknorm_rope_kv_write_batched_bf16",
    "maple_qknorm_rope_kv_write_bf16",
    "plan_maple_attention_build",
    "register_maple_attention_kernels",
]
