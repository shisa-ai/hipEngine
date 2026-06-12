"""Python-side caller for the MoE C1 C dispatcher (M14.dispatch.1-beta).

This module is paired with ``moe_c1_dispatch.hip``.  It defines:

- ``MoeC1Fns`` and ``MoeC1Args`` ctypes Structures that mirror the C structs.
- ``build_moe_c1_dispatch()`` to JIT-build the dispatcher .so via the existing
  ``build_hip`` plumbing.
- ``dispatch_linear_fp16(fns, args)`` / ``dispatch_full_fp16(fns, args)`` thin
  wrappers around the extern-C entry points.

Building the ``MoeC1Fns`` and ``MoeC1Args`` instances + integrating them into
``Qwen35ParoLayerRuntime.run_moe_c1_fp16`` lives in
``hipengine/runtime/moe_c1_dispatch.py`` so the kernel package stays a pure
launcher boundary.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime

_SOURCE = Path(__file__).with_name("moe_c1_dispatch.hip")
_OUTPUT_NAME = "moe_c1_dispatch.so"
_SYMBOL_LINEAR = "hipengine_moe_c1_dispatch_linear_fp16"
_SYMBOL_FULL = "hipengine_moe_c1_dispatch_full_fp16"


def plan_moe_c1_dispatch_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="moe_c1_dispatch",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_moe_c1_dispatch(
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
        family="moe_c1_dispatch",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


# ---- ctypes Structures mirroring moe_c1_dispatch.hip ----

class MoeC1Fns(ctypes.Structure):
    """Function-pointer table.  Field order MUST match the C struct exactly."""

    _fields_ = [
        ("router_topk_shared_out_fp16", ctypes.c_void_p),
        ("paro_rotate1_fp16", ctypes.c_void_p),
        ("paro_rotate2_fp16", ctypes.c_void_p),
        ("gemv_awq_selected_dual_pack8_transposed_fp16", ctypes.c_void_p),
        ("gemv_awq_selected_dual_pack8_transposed_rotate_staged_keyed_fp16", ctypes.c_void_p),
        ("gemv_awq_selected_pack8_transposed_fp16", ctypes.c_void_p),
        ("gemv_awq_selected_pack8_transposed_silu_rotate_staged_keyed_fp16", ctypes.c_void_p),
        ("silu_mul_dual_rotate_out_fp16", ctypes.c_void_p),
        ("silu_mul_separate_out_fp16", ctypes.c_void_p),
        ("silu_mul_pair_rotate_out_fp16", ctypes.c_void_p),  # linear-attn reduced-DAG default
        ("awq_fusedw4_prefill_dual_fp16", ctypes.c_void_p),    # linear-attn only
        ("gemv_awq_dual_pack8_output_tiled_split_transposed_fp16", ctypes.c_void_p),  # linear-attn opt-in
        ("awq_fusedw4_prefill_fp16", ctypes.c_void_p),         # linear-attn only
        ("gemv_awq_pack8_multi_row_transposed_fp16", ctypes.c_void_p),         # linear-attn shared down opt-in
        ("gemv_awq_pack8_multi_row_decode_transposed_fp16", ctypes.c_void_p),  # linear-attn shared down opt-in
        ("gemv_awq_pack8_output_tiled_transposed_fp16", ctypes.c_void_p),      # linear-attn shared down opt-in
        ("gemv_awq_pack8_output_tiled_combine_residual_transposed_fp16", ctypes.c_void_p),  # linear-attn reduced-DAG opt-in
        ("gemv_awq_pack8_output_tiled_combine_residual_full_transposed_fp16", ctypes.c_void_p),  # full-attn reduced-DAG default
        ("gemv_awq_dual_pack8_transposed_fp16", ctypes.c_void_p),  # full-attn only
        ("gemv_awq_pack8_transposed_fp16", ctypes.c_void_p),       # full-attn only
        ("combine_batch_fp16", ctypes.c_void_p),
    ]


class MoeC1Args(ctypes.Structure):
    """Per-layer + per-call args.  Field order MUST match the C struct."""

    _fields_ = [
        # ---- Per-call variables ----
        ("hidden", ctypes.c_void_p),
        ("residual", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("tokens", ctypes.c_int64),
        ("stream", ctypes.c_void_p),
        ("selected_rotate_barrier_target", ctypes.c_int64),
        ("selected_rotate_barrier_epoch", ctypes.c_int64),
        ("selected_down_barrier_target", ctypes.c_int64),
        ("selected_down_barrier_epoch", ctypes.c_int64),
        # ---- Layer-constant scratch pointers ----
        ("router_logits", ctypes.c_void_p),
        ("selected_experts", ctypes.c_void_p),
        ("routing_weights", ctypes.c_void_p),
        ("gate_up_input", ctypes.c_void_p),
        ("selected_rotate_fuse_barrier", ctypes.c_void_p),
        ("gate_up", ctypes.c_void_p),
        ("down_input", ctypes.c_void_p),
        ("down_out", ctypes.c_void_p),
        ("shared_gate_input", ctypes.c_void_p),
        ("shared_up_input", ctypes.c_void_p),
        ("shared_gate_out", ctypes.c_void_p),     # linear-attn only
        ("shared_up_out", ctypes.c_void_p),       # linear-attn only
        ("shared_up_packed", ctypes.c_void_p),    # full-attn only
        ("shared_intermediate", ctypes.c_void_p), # linear-attn only
        ("shared_down_input", ctypes.c_void_p),
        ("shared_out", ctypes.c_void_p),
        # ---- Layer-constant weight pointers ----
        ("router_combined_weight", ctypes.c_void_p),
        # Selected gate-up (per-expert stacked)
        ("selected_gate_up_pairs", ctypes.c_void_p),
        ("selected_gate_up_theta", ctypes.c_void_p),
        ("selected_gate_up_channel_scales", ctypes.c_void_p),
        ("selected_stacked_gate_qweight", ctypes.c_void_p),
        ("selected_stacked_gate_qzeros", ctypes.c_void_p),
        ("selected_stacked_gate_scales", ctypes.c_void_p),
        ("selected_stacked_up_qweight", ctypes.c_void_p),
        ("selected_stacked_up_qzeros", ctypes.c_void_p),
        ("selected_stacked_up_scales", ctypes.c_void_p),
        # Selected down (per-expert stacked)
        ("selected_down_pairs", ctypes.c_void_p),
        ("selected_down_theta", ctypes.c_void_p),
        ("selected_down_channel_scales", ctypes.c_void_p),
        ("selected_stacked_down_qweight", ctypes.c_void_p),
        ("selected_stacked_down_qzeros", ctypes.c_void_p),
        ("selected_stacked_down_scales", ctypes.c_void_p),
        # Shared gate
        ("shared_gate_pairs", ctypes.c_void_p),
        ("shared_gate_theta", ctypes.c_void_p),
        ("shared_gate_channel_scales", ctypes.c_void_p),
        ("shared_gate_qweight", ctypes.c_void_p),
        ("shared_gate_qzeros", ctypes.c_void_p),
        ("shared_gate_scales", ctypes.c_void_p),
        # Shared up
        ("shared_up_pairs", ctypes.c_void_p),
        ("shared_up_theta", ctypes.c_void_p),
        ("shared_up_channel_scales", ctypes.c_void_p),
        ("shared_up_qweight", ctypes.c_void_p),
        ("shared_up_qzeros", ctypes.c_void_p),
        ("shared_up_scales", ctypes.c_void_p),
        # Shared down
        ("shared_down_pairs", ctypes.c_void_p),
        ("shared_down_theta", ctypes.c_void_p),
        ("shared_down_channel_scales", ctypes.c_void_p),
        ("shared_down_qweight", ctypes.c_void_p),
        ("shared_down_qzeros", ctypes.c_void_p),
        ("shared_down_scales", ctypes.c_void_p),
        # ---- Layer-constant dims ----
        ("hidden_size", ctypes.c_int64),
        ("num_experts", ctypes.c_int64),
        ("num_experts_per_tok", ctypes.c_int64),
        ("moe_intermediate_size", ctypes.c_int64),
        ("shared_expert_intermediate_size", ctypes.c_int64),
        ("group_size", ctypes.c_int64),
        ("krot_selected_gate_up", ctypes.c_int64),
        ("krot_selected_down", ctypes.c_int64),
        ("krot_shared_gate", ctypes.c_int64),
        ("krot_shared_up", ctypes.c_int64),
        ("krot_shared_down", ctypes.c_int64),
        ("out_packed_selected_gate", ctypes.c_int64),
        ("out_packed_selected_up", ctypes.c_int64),
        ("out_packed_selected_down", ctypes.c_int64),
        ("out_packed_shared_gate", ctypes.c_int64),
        ("out_packed_shared_up", ctypes.c_int64),
        ("out_packed_shared_down", ctypes.c_int64),
        # ---- Threads / tile params ----
        ("router_threads", ctypes.c_int64),
        ("selected_threads", ctypes.c_int64),
        ("shared_threads", ctypes.c_int64),
        ("shared_prefill_tile_m", ctypes.c_int64),  # linear-attn only
        ("shared_prefill_tile_n", ctypes.c_int64),  # linear-attn only
        ("shared_down_mode", ctypes.c_int64),        # linear-attn: 0=prefill, 1=multi-row decode, 2=output-tiled auto, 3=gemv, 4=multi-row
        ("combine_threads", ctypes.c_int64),
    ]


# ---- Argtypes for the two dispatcher entry points ----

_ARGTYPES_DISPATCH = (ctypes.POINTER(MoeC1Fns), ctypes.POINTER(MoeC1Args))


def dispatch_linear_fp16(
    library: ctypes.CDLL,
    fns: MoeC1Fns,
    args: MoeC1Args,
    *,
    runtime: HipRuntime | None = None,
) -> None:
    """Call ``hipengine_moe_c1_dispatch_linear_fp16``.  fns + args must already
    be filled in by ``MoeC1DispatchCache``.
    """

    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_LINEAR, _ARGTYPES_DISPATCH, ctypes.c_int32)
    err = fn(ctypes.byref(fns), ctypes.byref(args))
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def dispatch_full_fp16(
    library: ctypes.CDLL,
    fns: MoeC1Fns,
    args: MoeC1Args,
    *,
    runtime: HipRuntime | None = None,
) -> None:
    """Call ``hipengine_moe_c1_dispatch_full_fp16``.  fns + args must already
    be filled in by ``MoeC1DispatchCache``.
    """

    runtime = runtime or get_hip_runtime()
    fn = signed_kernel_fn(library, _SYMBOL_FULL, _ARGTYPES_DISPATCH, ctypes.c_int32)
    err = fn(ctypes.byref(fns), ctypes.byref(args))
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


__all__ = [
    "MoeC1Args",
    "MoeC1Fns",
    "build_moe_c1_dispatch",
    "dispatch_full_fp16",
    "dispatch_linear_fp16",
    "plan_moe_c1_dispatch_build",
]
