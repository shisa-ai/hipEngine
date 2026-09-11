"""QSA dense-attention flash prefill kernel host wrapper.

Two-kernel design (gather + flash) specialized for the Qwen4Exp QSA
geometry: 24 q-heads x 256 dim, 2 kv-heads (GQA 12:1), paged bf16 KV with
256-token blocks, contiguous c1 chunks inside the dense-equivalent limit.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime

_SOURCE = Path(__file__).with_name("qwen4_exp_qsa_flash.hip")
_OUTPUT_NAME = "qwen4_exp_qsa_flash"

_ARGS = (
    ctypes.c_void_p,  # query
    ctypes.c_void_p,  # key_cache
    ctypes.c_void_p,  # value_cache
    ctypes.c_void_p,  # block_table
    ctypes.c_void_p,  # positions
    ctypes.c_void_p,  # k_scratch
    ctypes.c_void_p,  # v_scratch
    ctypes.c_void_p,  # out
    ctypes.c_int64,   # rows
    ctypes.c_int64,   # q_heads
    ctypes.c_int64,   # kv_heads
    ctypes.c_int64,   # head_dim
    ctypes.c_int64,   # block_size
    ctypes.c_int64,   # block_table_len
    ctypes.c_int64,   # context_len
    ctypes.c_float,   # scale
    ctypes.c_void_p,  # stream
)


def build_qwen4_exp_qsa_flash(
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
        family="qwen4_exp_qsa_flash",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def qwen4_exp_qsa_flash_prefill(
    query_ptr: int,
    key_cache_ptr: int,
    value_cache_ptr: int,
    block_table_ptr: int,
    positions_ptr: int,
    k_scratch_ptr: int,
    v_scratch_ptr: int,
    out_ptr: int,
    rows: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    block_size: int,
    block_table_len: int,
    context_len: int,
    scale: float,
    *,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Run the gather + flash QSA dense prefill for one contiguous chunk."""

    if rows <= 0 or q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads:
        raise ValueError("invalid QSA flash head geometry")
    if head_dim != 256:
        raise ValueError("QSA flash prefill requires head_dim == 256")
    if context_len <= 0:
        raise ValueError("context_len must be positive")
    library = library or build_qwen4_exp_qsa_flash(load=True)
    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(
        library,
        "hipengine_qwen4_exp_qsa_flash_prefill",
        _ARGS,
        ctypes.c_int,
    )
    error = fn(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_table_ptr,
        positions_ptr,
        k_scratch_ptr,
        v_scratch_ptr,
        out_ptr,
        rows,
        q_heads,
        kv_heads,
        head_dim,
        block_size,
        block_table_len,
        context_len,
        scale,
        stream,
    )
    if int(error) != HIP_SUCCESS:
        runtime.check(int(error))


__all__ = [
    "build_qwen4_exp_qsa_flash",
    "qwen4_exp_qsa_flash_prefill",
]
