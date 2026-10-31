"""Post-attention feed-forward execution for one Qwen3.5 GGUF layer.

Four entry points cover the feed-forward half of a layer:

* :func:`run_post_attention_ffn_rows` - the dispatch that picks the dense
  schedule or one of the two MoE bodies.
* :func:`run_post_attention_moe_c1` - the MoE c=1 body.
* :func:`run_post_attention_moe_c1_unfused_selected_ffn` - the selected-expert
  FFN the c=1 body calls per expert.
* :func:`run_post_attention_moe_rows` - the bulk MoE body.

Inputs
------
Every value the bodies need is an argument. Nothing here imports the runner
module, holds a session or runner object, proxies attributes, or resolves a
global at call time.

``layer`` and ``config`` are the resolved layer weights and model config
(``weights.layer(layer_id)`` / ``weights.config``); ``layer_id`` survives only
where the body forwards it to ``launch_gguf_linear_pair_silu``. ``geometry``
carries ``hidden_size``, ``ffn_size`` and ``backend``; ``ffn_size`` is read by
the dense schedule alone, so a MoE-only caller may pass ``None``. ``runtime`` is
used as given, with the bodies' own ``runtime or get_hip_runtime()`` fallback.

Callable tables
---------------
``kernels``
    Kernel and launch callables. Defaults are the true-module imports, so the
    module runs without an adapter. CPU tests patch these names on the runner
    module (for example ``tests/test_unit_gguf_ud_ffn_callers.py`` patches
    ``silu_mul_separate_out_bf16`` and ``gguf_bf16_add``), so an adapter passes
    the caller module's current attribute for each name and rebuilds the table
    per call; those patches keep intercepting the launch.

``ops``
    The helpers that stay with the caller (policy resolvers, selected MoE
    launchers, the optional-workspace helper, the stage-timing marker, the cast
    library). They are required - the component has no fallback - and the
    helpers that took a runner are called without one.

``flags``
    The env-gated predicates. Each is a thunk so an environment change between
    calls is still observed, exactly like the ``_gguf_*_enabled()`` helpers.

``dense_down_residual_decode_c1``
    ``None`` keeps the body's resolve-at-use fallback. A caller that memoizes the
    decision passes the stored value; the memo read and write stay with the
    caller, which is what keeps it a single read per process.

Launch order, branches, scalar arguments, scratch accesses, stage marks and
stage-timing marks are the bodies' own. The MoE c=1 and MoE rows bodies stay
separate: their router launch counts, selected-expert launchers and combine
kernels differ.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.kernels.hip_gfx1100.convert import f32_to_bf16
from hipengine.kernels.hip_gfx1100.fused import (
    gguf_add_rmsnorm_f32_bf16_f32_weight,
    gguf_add_rmsnorm_f32_f32_f32_weight,
    gguf_bf16_add,
    gguf_f32_bf16_add_out_f32,
    gguf_rmsnorm_bf16_f32_weight,
    gguf_rmsnorm_f32_f32_weight_out_f32,
    shared_gate_combine_residual_batch_out_bf16,
    shared_gate_combine_residual_out_bf16,
    shared_gate_combine_residual_rmsnorm_gguf_bf16_out,
    silu_mul_dual_out_bf16,
    silu_mul_separate_out_bf16,
    silu_mul_separate_out_f32,
)
from hipengine.kernels.hip_gfx1100.fused.paro_combine import (
    weighted_sum_f32_shared_f32_gate_combine_residual_batch_out_f32_accum_f32w,
    weighted_sum_f32_shared_f32_gate_combine_residual_out_f32_accum_f32w,
    weighted_sum_f32_shared_gate_combine_residual_batch_out_f32_accum_f32w,
    weighted_sum_f32_shared_gate_combine_residual_out_f32_accum_f32w,
    weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w,
    weighted_sum_shared_gate_combine_residual_out_bf16_f32w,
)
from hipengine.kernels.hip_gfx1100.moe.router import qwen35_router_select
from hipengine.loading.gguf_selected_contract import selected_ffn_modes
from hipengine.loading.qwen35_gguf import Qwen35GGUFConfig
from hipengine.loading.qwen35_gguf_materialize import (
    Qwen35GGUFDeviceWeight,
    Qwen35GGUFResidentLayerWeights,
)
from hipengine.runtime.gguf_linear import (
    launch_gguf_linear,
    launch_gguf_linear_pair,
    launch_gguf_linear_pair_concat,
    launch_gguf_linear_pair_silu,
    launch_gguf_linear_residual,
)

__all__ = [
    "FfnFlags",
    "FfnGeometry",
    "FfnKernels",
    "FfnOps",
    "ExpertSidecar",
    "StageRecorder",
    "run_post_attention_ffn_rows",
    "run_post_attention_moe_c1",
    "run_post_attention_moe_c1_unfused_selected_ffn",
    "run_post_attention_moe_rows",
]

Fn = Callable[..., Any]


class StageRecorder(Protocol):
    """The ``_HipEventStageRecorder`` surface the FFN bodies mark."""

    def mark(self, name: str) -> None: ...


class ExpertSidecar(Protocol):
    """The ``_DeviceExpertLayerSidecar`` surface the bulk MoE body reads."""

    def tensor(self, name: str) -> Any: ...


@dataclass(frozen=True)
class FfnGeometry:
    """Layer widths and backend the FFN bodies read."""

    hidden_size: int
    #: Read by the MoE bodies' router/launch calls; a dense-only caller may pass
    #: ``None`` because the dense schedule never reads it.
    backend: str | None = None
    #: Dense FFN width. Required by the dense schedule; the MoE bodies never
    #: read it, so a MoE-only caller may pass ``None``.
    ffn_size: int | None = None


@dataclass(frozen=True)
class FfnKernels:
    """Kernel and launch callables used by the FFN bodies.

    Defaults are the true-module imports. A runner-module adapter overrides each
    field with the runner module's global of the same name so the existing
    ``monkeypatch.setattr(qwen35_gguf_runner, <name>, ...)`` seams keep working.
    """

    gguf_add_rmsnorm_f32_f32_f32_weight: Fn = gguf_add_rmsnorm_f32_f32_f32_weight
    gguf_add_rmsnorm_f32_bf16_f32_weight: Fn = gguf_add_rmsnorm_f32_bf16_f32_weight
    gguf_rmsnorm_f32_f32_weight_out_f32: Fn = gguf_rmsnorm_f32_f32_weight_out_f32
    gguf_rmsnorm_bf16_f32_weight: Fn = gguf_rmsnorm_bf16_f32_weight
    gguf_bf16_add: Fn = gguf_bf16_add
    gguf_f32_bf16_add_out_f32: Fn = gguf_f32_bf16_add_out_f32
    f32_to_bf16: Fn = f32_to_bf16
    launch_gguf_linear: Fn = launch_gguf_linear
    launch_gguf_linear_pair: Fn = launch_gguf_linear_pair
    launch_gguf_linear_pair_concat: Fn = launch_gguf_linear_pair_concat
    launch_gguf_linear_pair_silu: Fn = launch_gguf_linear_pair_silu
    launch_gguf_linear_residual: Fn = launch_gguf_linear_residual
    silu_mul_separate_out_bf16: Fn = silu_mul_separate_out_bf16
    silu_mul_separate_out_f32: Fn = silu_mul_separate_out_f32
    silu_mul_dual_out_bf16: Fn = silu_mul_dual_out_bf16
    qwen35_router_select: Fn = qwen35_router_select
    shared_gate_combine_residual_out_bf16: Fn = shared_gate_combine_residual_out_bf16
    shared_gate_combine_residual_rmsnorm_gguf_bf16_out: Fn = (
        shared_gate_combine_residual_rmsnorm_gguf_bf16_out
    )
    shared_gate_combine_residual_batch_out_bf16: Fn = (
        shared_gate_combine_residual_batch_out_bf16
    )
    weighted_sum_f32_shared_f32_gate_combine_residual_out_f32_accum_f32w: Fn = (
        weighted_sum_f32_shared_f32_gate_combine_residual_out_f32_accum_f32w
    )
    weighted_sum_f32_shared_gate_combine_residual_out_f32_accum_f32w: Fn = (
        weighted_sum_f32_shared_gate_combine_residual_out_f32_accum_f32w
    )
    weighted_sum_shared_gate_combine_residual_out_bf16_f32w: Fn = (
        weighted_sum_shared_gate_combine_residual_out_bf16_f32w
    )
    weighted_sum_f32_shared_f32_gate_combine_residual_batch_out_f32_accum_f32w: Fn = (
        weighted_sum_f32_shared_f32_gate_combine_residual_batch_out_f32_accum_f32w
    )
    weighted_sum_f32_shared_gate_combine_residual_batch_out_f32_accum_f32w: Fn = (
        weighted_sum_f32_shared_gate_combine_residual_batch_out_f32_accum_f32w
    )
    weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w: Fn = (
        weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w
    )
    selected_ffn_modes: Fn = selected_ffn_modes


@dataclass(frozen=True)
class FfnOps:
    """Runner-owned helpers the FFN bodies call.

    The resolver/launcher helpers that take ``runner`` as their first
    parameter are supplied with that argument already bound, so the calls
    below carry no runner.
    """

    # --- selected primitive / policy resolution ---------------------------
    #: ``_gguf_norm_residual_decode_kernel(layer=..., rows=..., hidden_size=...)``
    norm_residual_kernel: Fn
    #: ``_gguf_dense_pair_silu_decode_variant(rows=, in_features=, out_features=)``
    dense_pair_silu_variant: Fn
    #: ``_gguf_dense_down_residual_decode_fused(rows=, in_features=, out_features=)``
    dense_down_residual_fused: Fn
    #: ``runner._rounded_add_rmsnorm_fn(rows=rows)``
    rounded_next_rmsnorm: Fn
    #: ``runner._cast_library()``
    cast_library: Fn
    # --- shared helpers ----------------------------------------------------
    #: ``_mark_sync_stage(runtime, timings, sync, name, t0)``
    mark_sync_stage: Fn
    #: ``_optional_q8_1_workspace_ptr(scratch, rows, in_features, enabled=, planes=)``
    optional_q8_1_workspace_ptr: Fn
    #: ``_gguf_use_f32_selected_down(weight, scratch, f32_residual)``
    use_f32_selected_down: Fn
    #: ``_gguf_use_f32_shared_down(scratch, f32_residual, selected_down_is_f32)``
    use_f32_shared_down: Fn
    #: ``_gguf_use_f32_selected_intermediate(scratch, prefer_f32_selected_down)``
    use_f32_selected_intermediate: Fn
    #: ``_gguf_prefill_router_select_threads(backend)``
    prefill_router_select_threads: Fn
    #: ``_selected_gemv_requires_q8_1_input(weight)``
    selected_gemv_requires_q8_1_input: Fn
    #: ``_selected_pair_requires_q8_1_input(gate_weight, up_weight)``
    selected_pair_requires_q8_1_input: Fn
    #: ``_gguf_f32_moe_combine_out_fn()``
    f32_moe_combine_out_fn: Fn
    #: ``_gguf_f32_moe_combine_batch_out_fn()``
    f32_moe_combine_batch_out_fn: Fn
    # --- router launches ---------------------------------------------------
    #: ``_launch_qwen35_router_logits_f32_hidden``
    router_logits_f32_hidden: Fn
    #: ``_launch_qwen35_router_logits_bf16_hidden``
    router_logits_bf16_hidden: Fn
    #: ``_try_launch_qwen35_router_topk_split_shared_bf16_f32w``
    router_topk_split_shared_bf16_f32w: Fn
    # --- dense / shared-expert launch wrappers -----------------------------
    #: ``_try_launch_shared_gate_up_from_f32_post_norm``
    shared_gate_up_from_f32_post_norm: Fn
    #: ``_try_launch_gguf_linear_bf16_f32_output``
    gguf_linear_bf16_f32_output: Fn
    # --- selected-expert launches ------------------------------------------
    #: ``_launch_selected_raw_gguf_moe_pair_silu``
    selected_raw_moe_pair_silu: Fn
    #: ``_launch_selected_raw_gguf_moe_dual``
    selected_raw_moe_dual: Fn
    #: ``_launch_selected_raw_gguf_moe_linear``
    selected_raw_moe_linear: Fn
    #: ``_launch_weighted_selected_raw_gguf_moe_linear``
    weighted_selected_raw_moe_linear: Fn
    #: ``_launch_selected_expert_pack8_moe_pair``
    selected_expert_pack8_moe_pair: Fn
    #: ``_launch_selected_expert_pack8_moe_linear``
    selected_expert_pack8_moe_linear: Fn
    # --- opt-in compact / fused MoE schedules ------------------------------
    #: ``_try_run_post_attention_moe_c1_compact_gemv(layer, ...)``
    moe_c1_compact_gemv: Fn
    #: ``_try_run_post_attention_moe_c1_fused_ffn(layer, ...)``
    moe_c1_fused_ffn: Fn
    #: ``_try_run_post_attention_moe_rows_compact_wmma(layer, ...)``
    moe_rows_compact_wmma: Fn


@dataclass(frozen=True)
class FfnFlags:
    """Env-gated predicates, read per call like the runner's ``_gguf_*`` helpers."""

    #: ``_gguf_verify_f32_attn_out_enabled()``
    verify_f32_attn_out_enabled: Fn
    #: ``_gguf_verify_f32_post_norm_enabled()``
    verify_f32_post_norm_enabled: Fn
    #: ``_gguf_verify_f32_post_norm_router_enabled()``
    verify_f32_post_norm_router_enabled: Fn
    #: ``_gguf_verify_f32_post_norm_selected_q8_enabled()``
    verify_f32_post_norm_selected_q8_enabled: Fn
    #: ``_gguf_verify_f32_post_norm_shared_q8_enabled()``
    verify_f32_post_norm_shared_q8_enabled: Fn
    #: resolved cooperative F32 router policy (owner-bound or ambient)
    router_f32w_coop_enabled: Fn
    #: resolved router persistent-counter policy (owner-bound or ambient)
    router_f32w_persistent_counter_enabled: Fn
    #: ``_env_flag(_GGUF_COMPACT_MOE_C1_ENV, False)``
    compact_moe_c1_enabled: Fn
    #: ``_env_flag(_GGUF_FUSED_MOE_FFN_ENV, False)``
    fused_moe_ffn_enabled: Fn
    #: ``_gguf_q4k_selected_dual_dp4a_enabled()``
    q4k_selected_dual_dp4a_enabled: Fn
    #: ``_gguf_t16_selected_dp4a_enabled()``
    t16_selected_dp4a_enabled: Fn
    #: ``_gguf_raw_selected_dp4a_enabled()``
    raw_selected_dp4a_enabled: Fn


def run_post_attention_ffn_rows(
    layer: Qwen35GGUFResidentLayerWeights,
    config: Qwen35GGUFConfig,
    geometry: FfnGeometry,
    kernels: FfnKernels,
    ops: FfnOps,
    flags: FfnFlags,
    layer_id: int,
    hidden_ptr: int,
    attn_out_ptr: int,
    out_ptr: int,
    scratch,
    *,
    rows: int,
    dense_down_residual_decode_c1: bool | None = None,
    runtime: HipRuntime | None = None,
    next_norm_weight_ptr: int | None = None,
    next_norm_out_ptr: int | None = None,
    stream: int = 0,
    expert_sidecar: ExpertSidecar | None = None,
    expert_pack8_library: object | None = None,
    hidden_f32_ptr: int | None = None,
    out_f32_ptr: int | None = None,
    attn_out_f32_ptr: int | None = None,
    stage_timings: dict[str, float] | None = None,
    sync_stage_timings: bool = False,
    stage_prefix: str = "target_block_ffn",
    gpu_stage_recorder: StageRecorder | None = None,
    force_bulk_rows: bool = False,
    q8_t16_rowtile_all: bool | None = None,
) -> None:
    """Residual + post-norm, dense or MoE feed-forward, down + residual.

    ``force_bulk_rows`` keeps a one-row PREFILL on the bulk schedule.

    The ``rows == 1`` specialisations below exist for decode, where a single
    row is the natural shape. A prefill chunk that happens to contain one
    row is a different thing: taking the decode schedule makes a chunk's
    arithmetic depend on how the prompt was split, so prefilling N tokens as
    one call and as ``N-1 + 1`` calls disagree. That breaks the contract a
    prefix-cache hit owes a miss. Prefill callers therefore pass True;
    decode callers keep the specialised route.

    ``dense_down_residual_decode_c1`` is the resolved
    ``runner._dense_down_residual_decode_c1`` memo. ``None`` re-resolves it at
    the point the runner resolved it (the adapter owns the memo write).
    """

    runtime = runtime or get_hip_runtime()
    sync_stages = bool(sync_stage_timings and stage_timings is not None)
    t_stage = time.perf_counter() if sync_stages else 0.0
    f32_residual = hidden_f32_ptr is not None or out_f32_ptr is not None
    if (next_norm_weight_ptr is None) != (next_norm_out_ptr is None):
        raise ValueError("next norm weight and output pointers must be provided together")
    post_norm_f32_ptr: int | None = None
    if f32_residual:
        if hidden_f32_ptr is None or out_f32_ptr is None:
            raise ValueError("hidden_f32_ptr and out_f32_ptr must be provided together")
        post_attention_norm_ptr = layer.weight("post_attention_norm").allocation().tensor.ptr
        if attn_out_f32_ptr is not None and flags.verify_f32_attn_out_enabled():
            kernels.gguf_add_rmsnorm_f32_f32_f32_weight(
                int(hidden_f32_ptr),
                int(attn_out_f32_ptr),
                post_attention_norm_ptr,
                scratch.post_norm.ptr,
                int(out_f32_ptr),
                rows=rows,
                hidden_size=geometry.hidden_size,
                eps=config.rms_norm_eps,
                stream=stream,
                runtime=runtime,
            )
        else:
            kernels.gguf_add_rmsnorm_f32_bf16_f32_weight(
                int(hidden_f32_ptr),
                attn_out_ptr,
                post_attention_norm_ptr,
                scratch.post_norm.ptr,
                int(out_f32_ptr),
                rows=rows,
                hidden_size=geometry.hidden_size,
                eps=config.rms_norm_eps,
                stream=stream,
                runtime=runtime,
            )
        if flags.verify_f32_post_norm_enabled():
            if not hasattr(scratch, "post_norm_f32"):
                raise ValueError("scratch is missing post_norm_f32 for verifier F32 post-norm diagnostic")
            kernels.gguf_rmsnorm_f32_f32_weight_out_f32(
                int(out_f32_ptr),
                post_attention_norm_ptr,
                scratch.post_norm_f32.ptr,
                rows=rows,
                hidden_size=geometry.hidden_size,
                eps=config.rms_norm_eps,
                stream=stream,
                runtime=runtime,
            )
            post_norm_f32_ptr = scratch.post_norm_f32.ptr
    else:
        add_norm_kernel = ops.norm_residual_kernel(
            layer="add_rmsnorm",
            rows=rows,
            hidden_size=geometry.hidden_size,
        )
        add_norm_kernel(
            hidden_ptr,
            attn_out_ptr,
            layer.weight("post_attention_norm").allocation().tensor.ptr,
            scratch.post_norm.ptr,
            scratch.residual.ptr,
            rows=rows,
            hidden_size=geometry.hidden_size,
            eps=config.rms_norm_eps,
            stream=stream,
            runtime=runtime,
        )
    t_stage = ops.mark_sync_stage(
        runtime,
        stage_timings,
        sync_stages,
        f"{stage_prefix}_post_norm_residual",
        t_stage,
    )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_post_norm_residual")
    if config.is_moe:
        # MoE has no bulk rows==1 schedule (`run_post_attention_moe_rows`
        # refuses rows <= 1), so force_bulk_rows cannot redirect it. A
        # one-row MoE prefill is avoided upstream instead: the prefix-cache
        # admission never leaves a single-token suffix.
        if rows == 1:
            run_post_attention_moe_c1(
                layer,
                config,
                geometry,
                kernels,
                ops,
                flags,
                out_ptr,
                scratch,
                runtime=runtime,
                stream=stream,
                residual_f32_ptr=out_f32_ptr if f32_residual else None,
                out_f32_ptr=out_f32_ptr if f32_residual else None,
                post_norm_f32_ptr=post_norm_f32_ptr,
                next_norm_weight_ptr=next_norm_weight_ptr,
                next_norm_out_ptr=next_norm_out_ptr,
                gpu_stage_recorder=gpu_stage_recorder,
                stage_prefix=f"{stage_prefix}_moe",
            )
        else:
            run_post_attention_moe_rows(
                layer,
                config,
                geometry,
                kernels,
                ops,
                flags,
                out_ptr,
                scratch,
                rows=rows,
                runtime=runtime,
                stream=stream,
                expert_sidecar=expert_sidecar,
                expert_pack8_library=expert_pack8_library,
                residual_f32_ptr=out_f32_ptr if f32_residual else None,
                out_f32_ptr=out_f32_ptr if f32_residual else None,
                post_norm_f32_ptr=post_norm_f32_ptr,
                stage_timings=stage_timings,
                sync_stage_timings=sync_stage_timings,
                stage_prefix=f"{stage_prefix}_moe",
                gpu_stage_recorder=gpu_stage_recorder,
                q8_t16_rowtile_all=q8_t16_rowtile_all,
            )
        return
    if next_norm_weight_ptr is not None and f32_residual:
        raise ValueError("dense next RMSNorm fusion requires a BF16 residual")
    rounded_next_rms_fn = (
        ops.rounded_next_rmsnorm(rows=rows)
        if next_norm_weight_ptr is not None and rows <= 8
        else None
    )
    dense_decode_variant = (
        None
        if force_bulk_rows
        # A registered decode variant swaps the fused gate/up SiLU onto the
        # GEMV decode schedule, which is a different arithmetic route than
        # the GEMM the same tokens take inside a wider prefill chunk.
        else ops.dense_pair_silu_variant(
            rows=rows,
            in_features=geometry.hidden_size,
            out_features=geometry.ffn_size,
        )
    )
    dense_q8_1_workspace_ptr = ops.optional_q8_1_workspace_ptr(
        scratch,
        rows,
        geometry.hidden_size,
        enabled=(
            dense_decode_variant
            in {
                "dense_dual_q8_1x2_dp4a_bf16_bf16_out",
                "dense_dual_q8_1x2_split_weight_dp4a_bf16_bf16_out",
                "dense_dual_q8_1x2_rowtile8_dp4a_bf16_bf16_out",
            }
        ),
        planes=2,
    )
    dense_silu_fused = (
        rows > 1 or dense_decode_variant is not None
    ) and kernels.launch_gguf_linear_pair_silu(
        layer.weight("ffn_gate"),
        layer.weight("ffn_up"),
        scratch.post_norm.ptr,
        scratch.ffn_intermediate.ptr,
        rows=rows,
        in_features=geometry.hidden_size,
        out_features=geometry.ffn_size,
        stream=stream,
        runtime=runtime,
        use_gemv_decode=(True if dense_decode_variant is not None else None),
        registered_decode_variant=dense_decode_variant,
        layer_id=layer_id,
        q8_1_workspace_ptr=dense_q8_1_workspace_ptr,
        pair_workspace_ptr=scratch.ffn_gate_up.ptr,
        pair_workspace_nbytes=int(
            getattr(scratch.ffn_gate_up, "nbytes", 0)
        ),
    )
    if not dense_silu_fused:
        if not kernels.launch_gguf_linear_pair(
            layer.weight("ffn_gate"),
            layer.weight("ffn_up"),
            scratch.post_norm.ptr,
            scratch.ffn_gate_up.ptr,
            scratch.ffn_gate_up.ptr + geometry.ffn_size * rows * 2,
            rows=rows,
            in_features=geometry.hidden_size,
            out_features=geometry.ffn_size,
            stream=stream,
            runtime=runtime,
        ):
            kernels.launch_gguf_linear(
                layer.weight("ffn_gate"),
                scratch.post_norm.ptr,
                scratch.ffn_gate_up.ptr,
                rows=rows,
                in_features=geometry.hidden_size,
                out_features=geometry.ffn_size,
                stream=stream,
                runtime=runtime,
            )
            kernels.launch_gguf_linear(
                layer.weight("ffn_up"),
                scratch.post_norm.ptr,
                scratch.ffn_gate_up.ptr + geometry.ffn_size * rows * 2,
                rows=rows,
                in_features=geometry.hidden_size,
                out_features=geometry.ffn_size,
                stream=stream,
                runtime=runtime,
            )
    t_stage = ops.mark_sync_stage(
        runtime,
        stage_timings,
        sync_stages,
        f"{stage_prefix}_gate_up",
        t_stage,
    )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_gate_up")
    if not dense_silu_fused:
        kernels.silu_mul_separate_out_bf16(
            scratch.ffn_gate_up.ptr,
            scratch.ffn_gate_up.ptr + geometry.ffn_size * rows * 2,
            scratch.ffn_intermediate.ptr,
            rows=rows,
            features=geometry.ffn_size,
            stream=stream,
            runtime=runtime,
        )
    t_stage = ops.mark_sync_stage(
        runtime,
        stage_timings,
        sync_stages,
        f"{stage_prefix}_silu",
        t_stage,
    )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_silu")
    dense_down_decode_c1 = dense_down_residual_decode_c1
    if dense_down_decode_c1 is None:
        # Compatibility/probe runners constructed without __post_init__
        # resolve once here; resident production runners precompute this and
        # pass the memo in. The memo write belongs to the caller's facade.
        dense_down_decode_c1 = ops.dense_down_residual_fused(
            rows=1,
            in_features=geometry.ffn_size,
            out_features=geometry.hidden_size,
        )
    dense_down_decode_fused = (
        rows == 1 and not force_bulk_rows and bool(dense_down_decode_c1)
    )
    down_residual_fused = (
        not f32_residual
        and (
            (next_norm_weight_ptr is None and rows > 1)
            or (rounded_next_rms_fn is None and rows == 1)
        )
        and kernels.launch_gguf_linear_residual(
            layer.weight("ffn_down"),
            scratch.ffn_intermediate.ptr,
            scratch.residual.ptr,
            out_ptr,
            rows,
            geometry.ffn_size,
            geometry.hidden_size,
            stream=stream,
            runtime=runtime,
            registered_decode=dense_down_decode_fused,
        )
    )
    if down_residual_fused and next_norm_weight_ptr is not None:
        # E6c: the composite replaced residual = add(residual, down);
        # the trailing next-input RMSNorm stays exactly the launch the
        # unfused else-branch ran after that add.
        kernels.gguf_rmsnorm_bf16_f32_weight(
            out_ptr,
            int(next_norm_weight_ptr),
            int(next_norm_out_ptr),
            rows=rows,
            hidden_size=geometry.hidden_size,
            eps=config.rms_norm_eps,
            stream=stream,
            runtime=runtime,
        )
    if not down_residual_fused:
        kernels.launch_gguf_linear(
            layer.weight("ffn_down"),
            scratch.ffn_intermediate.ptr,
            scratch.ffn_down.ptr,
            rows=rows,
            in_features=geometry.ffn_size,
            out_features=geometry.hidden_size,
            stream=stream,
            runtime=runtime,
        )
        if f32_residual:
            count = rows * geometry.hidden_size
            kernels.gguf_f32_bf16_add_out_f32(
                int(out_f32_ptr),
                scratch.ffn_down.ptr,
                int(out_f32_ptr),
                count,
                stream=stream,
                runtime=runtime,
            )
            kernels.f32_to_bf16(
                int(out_f32_ptr),
                out_ptr,
                count,
                stream=stream,
                library=ops.cast_library(),
                runtime=runtime,
            )
        elif rounded_next_rms_fn is not None:
            rounded_next_rms_fn(
                scratch.residual.ptr,
                scratch.ffn_down.ptr,
                int(next_norm_weight_ptr),
                int(next_norm_out_ptr),
                out_ptr,
                rows,
                geometry.hidden_size,
                config.rms_norm_eps,
                stream=stream,
                runtime=runtime,
            )
        else:
            kernels.gguf_bf16_add(
                scratch.residual.ptr,
                scratch.ffn_down.ptr,
                out_ptr,
                rows * geometry.hidden_size,
                stream=stream,
                runtime=runtime,
            )
            if next_norm_weight_ptr is not None:
                kernels.gguf_rmsnorm_bf16_f32_weight(
                    out_ptr,
                    int(next_norm_weight_ptr),
                    int(next_norm_out_ptr),
                    rows=rows,
                    hidden_size=geometry.hidden_size,
                    eps=config.rms_norm_eps,
                    stream=stream,
                    runtime=runtime,
                )
    ops.mark_sync_stage(
        runtime,
        stage_timings,
        sync_stages,
        f"{stage_prefix}_down_residual",
        t_stage,
    )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_down_residual")


def run_post_attention_moe_c1(
    layer: Qwen35GGUFResidentLayerWeights,
    config: Qwen35GGUFConfig,
    geometry: FfnGeometry,
    kernels: FfnKernels,
    ops: FfnOps,
    flags: FfnFlags,
    out_ptr: int,
    scratch,
    *,
    runtime: HipRuntime | None = None,
    stream: int = 0,
    residual_f32_ptr: int | None = None,
    out_f32_ptr: int | None = None,
    post_norm_f32_ptr: int | None = None,
    next_norm_weight_ptr: int | None = None,
    next_norm_out_ptr: int | None = None,
    gpu_stage_recorder: StageRecorder | None = None,
    stage_prefix: str = "decode_ffn_moe",
) -> None:
    """One-row MoE feed-forward: router, selected experts, shared expert, combine."""

    cfg = config
    if (next_norm_weight_ptr is None) != (next_norm_out_ptr is None):
        raise ValueError("next norm weight and output pointers must be provided together")
    if not cfg.is_moe:
        raise ValueError("MoE path requires qwen35moe GGUF config")
    runtime = runtime or get_hip_runtime()
    top_k = int(cfg.expert_used_count)
    if top_k <= 0:
        raise ValueError("qwen35moe GGUF expert_used_count must be positive")
    if top_k > scratch.moe_selected_host.shape[0]:
        raise ValueError("qwen35moe scratch top-k capacity is too small")

    # llama.cpp keeps the qwen35moe router and shared-gate tensors in F32.
    # Compute expert logits and the adjacent shared-gate logit separately via
    # the registry-resolved router adapter so GGUF F32 weights do not get
    # silently contracted to BF16 on the correctness-first decode path.
    router_f32_ptr = (
        post_norm_f32_ptr
        if post_norm_f32_ptr is not None and flags.verify_f32_post_norm_router_enabled()
        else None
    )
    selected_f32_ptr = (
        post_norm_f32_ptr
        if post_norm_f32_ptr is not None and flags.verify_f32_post_norm_selected_q8_enabled()
        else None
    )
    shared_f32_ptr = (
        post_norm_f32_ptr
        if post_norm_f32_ptr is not None and flags.verify_f32_post_norm_shared_q8_enabled()
        else None
    )
    router_fn = (
        ops.router_logits_f32_hidden
        if router_f32_ptr is not None
        else ops.router_logits_bf16_hidden
    )
    router_hidden_ptr = int(router_f32_ptr) if router_f32_ptr is not None else scratch.post_norm.ptr
    router_fused = (
        router_f32_ptr is None
        and flags.router_f32w_coop_enabled()
        and ops.router_topk_split_shared_bf16_f32w(
            router_hidden_ptr,
            layer.weight("ffn_gate_inp"),
            layer.weight("ffn_gate_inp_shexp"),
            scratch.moe_router_logits.ptr,
            scratch.moe_selected_experts.ptr,
            scratch.moe_routing_weights.ptr,
            scratch.moe_router_counter.ptr,
            hidden_size=geometry.hidden_size,
            num_experts=cfg.expert_count,
            top_k=top_k,
            persistent_counter=flags.router_f32w_persistent_counter_enabled(),
            stream=stream,
            runtime=runtime,
        )
    )
    if not router_fused:
        router_fn(
            router_hidden_ptr,
            layer.weight("ffn_gate_inp"),
            scratch.moe_router_logits.ptr,
            1,
            geometry.hidden_size,
            cfg.expert_count,
            stream=stream,
            runtime=runtime,
        )
        router_fn(
            router_hidden_ptr,
            layer.weight("ffn_gate_inp_shexp"),
            scratch.moe_router_logits.ptr + cfg.expert_count * DType.FP32.itemsize,
            1,
            geometry.hidden_size,
            1,
            stream=stream,
            runtime=runtime,
        )
        kernels.qwen35_router_select(
            scratch.moe_router_logits.ptr,
            scratch.moe_selected_experts.ptr,
            scratch.moe_routing_weights.ptr,
            1,
            cfg.expert_count,
            cfg.expert_count,
            top_k,
            threads=256,
            stream=stream,
            runtime=runtime,
        )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_router")

    gate_weight = layer.weight("ffn_gate_exps")
    up_weight = layer.weight("ffn_up_exps")
    down_weight = layer.weight("ffn_down_exps")
    f32_residual = residual_f32_ptr is not None or out_f32_ptr is not None
    if f32_residual and (residual_f32_ptr is None or out_f32_ptr is None):
        raise ValueError("residual_f32_ptr and out_f32_ptr must be provided together")

    compact_moe = bool(
        not f32_residual
        and flags.compact_moe_c1_enabled()
        and ops.moe_c1_compact_gemv(
            layer,
            gate_weight,
            up_weight,
            down_weight,
            out_ptr,
            scratch,
            top_k=top_k,
            next_norm_weight_ptr=next_norm_weight_ptr,
            next_norm_out_ptr=next_norm_out_ptr,
            stream=stream,
            runtime=runtime,
        )
    )
    if compact_moe:
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_compact")
        return
    selected_rows = top_k
    selected_down_is_f32 = False
    expert_down_weighted = False
    prefer_f32_selected_down = ops.use_f32_selected_down(down_weight, scratch, f32_residual)
    if not (
        (not prefer_f32_selected_down)
        and flags.fused_moe_ffn_enabled()
        and ops.moe_c1_fused_ffn(
            layer,
            gate_weight,
            up_weight,
            down_weight,
            scratch,
            top_k=top_k,
            stream=stream,
            runtime=runtime,
        )
    ):
        selected_down_is_f32, expert_down_weighted = run_post_attention_moe_c1_unfused_selected_ffn(
            config,
            geometry,
            kernels,
            ops,
            flags,
            gate_weight,
            up_weight,
            down_weight,
            scratch,
            selected_rows=selected_rows,
            post_norm_f32_ptr=selected_f32_ptr,
            prefer_f32_selected_down=prefer_f32_selected_down,
            stream=stream,
            runtime=runtime,
        )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_selected")

    if ops.shared_gate_up_from_f32_post_norm(
        layer.weight("ffn_gate_shexp"),
        layer.weight("ffn_up_shexp"),
        shared_f32_ptr,
        scratch.moe_shared_gate.ptr,
        scratch.moe_shared_up.ptr,
        rows=1,
        hidden_size=geometry.hidden_size,
        shared_ffn=cfg.expert_shared_feed_forward_length,
        stream=stream,
        runtime=runtime,
    ):
        kernels.silu_mul_separate_out_bf16(
            scratch.moe_shared_gate.ptr,
            scratch.moe_shared_up.ptr,
            scratch.moe_shared_intermediate.ptr,
            rows=1,
            features=cfg.expert_shared_feed_forward_length,
            stream=stream,
            runtime=runtime,
        )
    elif kernels.launch_gguf_linear_pair_concat(
        layer.weight("ffn_gate_shexp"),
        layer.weight("ffn_up_shexp"),
        scratch.post_norm.ptr,
        scratch.ffn_gate_up.ptr,
        rows=1,
        in_features=geometry.hidden_size,
        out_features=cfg.expert_shared_feed_forward_length,
        stream=stream,
        runtime=runtime,
    ):
        kernels.silu_mul_dual_out_bf16(
            scratch.ffn_gate_up.ptr,
            scratch.moe_shared_intermediate.ptr,
            rows=1,
            features=cfg.expert_shared_feed_forward_length,
            stream=stream,
            runtime=runtime,
        )
    else:
        if not kernels.launch_gguf_linear_pair(
            layer.weight("ffn_gate_shexp"),
            layer.weight("ffn_up_shexp"),
            scratch.post_norm.ptr,
            scratch.moe_shared_gate.ptr,
            scratch.moe_shared_up.ptr,
            rows=1,
            in_features=geometry.hidden_size,
            out_features=cfg.expert_shared_feed_forward_length,
            stream=stream,
            runtime=runtime,
        ):
            kernels.launch_gguf_linear(
                layer.weight("ffn_gate_shexp"),
                scratch.post_norm.ptr,
                scratch.moe_shared_gate.ptr,
                rows=1,
                in_features=geometry.hidden_size,
                out_features=cfg.expert_shared_feed_forward_length,
                stream=stream,
                runtime=runtime,
            )
            kernels.launch_gguf_linear(
                layer.weight("ffn_up_shexp"),
                scratch.post_norm.ptr,
                scratch.moe_shared_up.ptr,
                rows=1,
                in_features=geometry.hidden_size,
                out_features=cfg.expert_shared_feed_forward_length,
                stream=stream,
                runtime=runtime,
            )
        kernels.silu_mul_separate_out_bf16(
            scratch.moe_shared_gate.ptr,
            scratch.moe_shared_up.ptr,
            scratch.moe_shared_intermediate.ptr,
            rows=1,
            features=cfg.expert_shared_feed_forward_length,
            stream=stream,
            runtime=runtime,
        )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_shared_gate_up")
    shared_down_is_f32 = False
    if (
        ops.use_f32_shared_down(scratch, f32_residual, selected_down_is_f32)
        and ops.gguf_linear_bf16_f32_output(
            layer.weight("ffn_down_shexp"),
            scratch.moe_shared_intermediate.ptr,
            scratch.moe_shared_out_f32.ptr,
            rows=1,
            in_features=cfg.expert_shared_feed_forward_length,
            out_features=geometry.hidden_size,
            stream=stream,
            runtime=runtime,
        )
    ):
        kernels.f32_to_bf16(
            scratch.moe_shared_out_f32.ptr,
            scratch.moe_shared_out.ptr,
            geometry.hidden_size,
            stream=stream,
            library=ops.cast_library(),
            runtime=runtime,
        )
        shared_down_is_f32 = True
    else:
        kernels.launch_gguf_linear(
            layer.weight("ffn_down_shexp"),
            scratch.moe_shared_intermediate.ptr,
            scratch.moe_shared_out.ptr,
            rows=1,
            in_features=cfg.expert_shared_feed_forward_length,
            out_features=geometry.hidden_size,
            stream=stream,
            runtime=runtime,
        )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_shared_down")
    if f32_residual:
        if selected_down_is_f32 and shared_down_is_f32:
            kernels.weighted_sum_f32_shared_f32_gate_combine_residual_out_f32_accum_f32w(
                scratch.moe_down_out_f32.ptr,
                scratch.moe_routing_weights.ptr,
                scratch.moe_shared_out_f32.ptr,
                scratch.moe_router_logits.ptr + cfg.expert_count * 4,
                int(residual_f32_ptr),
                int(out_f32_ptr),
                top_k,
                geometry.hidden_size,
                stream=stream,
                runtime=runtime,
            )
        elif selected_down_is_f32:
            kernels.weighted_sum_f32_shared_gate_combine_residual_out_f32_accum_f32w(
                scratch.moe_down_out_f32.ptr,
                scratch.moe_routing_weights.ptr,
                scratch.moe_shared_out.ptr,
                scratch.moe_router_logits.ptr + cfg.expert_count * 4,
                int(residual_f32_ptr),
                int(out_f32_ptr),
                top_k,
                geometry.hidden_size,
                stream=stream,
                runtime=runtime,
            )
        else:
            ops.f32_moe_combine_out_fn()(
                scratch.moe_down_out.ptr,
                scratch.moe_routing_weights.ptr,
                scratch.moe_shared_out.ptr,
                scratch.moe_router_logits.ptr + cfg.expert_count * 4,
                int(residual_f32_ptr),
                int(out_f32_ptr),
                top_k,
                geometry.hidden_size,
                stream=stream,
                runtime=runtime,
            )
        kernels.f32_to_bf16(
            int(out_f32_ptr),
            out_ptr,
            geometry.hidden_size,
            stream=stream,
            library=ops.cast_library(),
            runtime=runtime,
        )
        if next_norm_weight_ptr is not None:
            kernels.gguf_rmsnorm_bf16_f32_weight(
                out_ptr,
                next_norm_weight_ptr,
                next_norm_out_ptr,
                rows=1,
                hidden_size=geometry.hidden_size,
                eps=cfg.rms_norm_eps,
                stream=stream,
                runtime=runtime,
            )
    else:
        if expert_down_weighted:
            if next_norm_weight_ptr is None:
                kernels.shared_gate_combine_residual_out_bf16(
                    scratch.moe_down_out.ptr,
                    scratch.moe_shared_out.ptr,
                    scratch.moe_router_logits.ptr + cfg.expert_count * 4,
                    scratch.residual.ptr,
                    out_ptr,
                    geometry.hidden_size,
                    stream=stream,
                    runtime=runtime,
                )
            else:
                kernels.shared_gate_combine_residual_rmsnorm_gguf_bf16_out(
                    scratch.moe_down_out.ptr,
                    scratch.moe_shared_out.ptr,
                    scratch.moe_router_logits.ptr + cfg.expert_count * 4,
                    scratch.residual.ptr,
                    next_norm_weight_ptr,
                    next_norm_out_ptr,
                    out_ptr,
                    1,
                    geometry.hidden_size,
                    1,
                    eps=cfg.rms_norm_eps,
                    stream=stream,
                    runtime=runtime,
                )
        else:
            kernels.weighted_sum_shared_gate_combine_residual_out_bf16_f32w(
                scratch.moe_down_out.ptr,
                scratch.moe_routing_weights.ptr,
                scratch.moe_shared_out.ptr,
                scratch.moe_router_logits.ptr + cfg.expert_count * 4,
                scratch.residual.ptr,
                out_ptr,
                top_k,
                geometry.hidden_size,
                stream=stream,
                runtime=runtime,
            )
            if next_norm_weight_ptr is not None:
                kernels.gguf_rmsnorm_bf16_f32_weight(
                    out_ptr,
                    next_norm_weight_ptr,
                    next_norm_out_ptr,
                    rows=1,
                    hidden_size=geometry.hidden_size,
                    eps=cfg.rms_norm_eps,
                    stream=stream,
                    runtime=runtime,
                )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_combine")


def run_post_attention_moe_c1_unfused_selected_ffn(
    config: Qwen35GGUFConfig,
    geometry: FfnGeometry,
    kernels: FfnKernels,
    ops: FfnOps,
    flags: FfnFlags,
    gate_weight: Qwen35GGUFDeviceWeight,
    up_weight: Qwen35GGUFDeviceWeight,
    down_weight: Qwen35GGUFDeviceWeight,
    scratch,
    *,
    selected_rows: int,
    post_norm_f32_ptr: int | None = None,
    prefer_f32_selected_down: bool = False,
    stream: int,
    runtime: HipRuntime,
) -> tuple[bool, bool]:
    """Unfused selected-expert FFN: gate_up GEMV -> silu*mul -> down GEMV into
    ``scratch.moe_down_out``. Numerically-equivalent fallback for the fused B1
    megakernel (architectural invariant), and the default rows==1 path."""

    cfg = config
    gate_rows_nbytes = selected_rows * cfg.expert_feed_forward_length * DType.BF16.itemsize
    f32_selected_intermediate = ops.use_f32_selected_intermediate(
        scratch,
        bool(prefer_f32_selected_down),
    )
    gate_call, down_call = kernels.selected_ffn_modes(
        gate_weight.spec.quant_key, up_weight.spec.quant_key, down_weight.spec.quant_key,
        allow_legacy_silu=True, f32_gate=post_norm_f32_ptr is not None,
        f32_intermediate=f32_selected_intermediate, f32_down=prefer_f32_selected_down,
    )
    expert_silu_ready = False
    if gate_call == "dual_silu":
        expert_silu_ready = ops.selected_raw_moe_pair_silu(
            gate_weight,
            up_weight,
            scratch.post_norm.ptr,
            scratch.moe_selected_experts.ptr,
            scratch.ffn_intermediate.ptr,
            x_rows=1,
            rows=selected_rows,
            num_experts=cfg.expert_count,
            in_features=geometry.hidden_size,
            out_features=cfg.expert_feed_forward_length,
            stream=stream,
            runtime=runtime,
        )
    if not expert_silu_ready:
        q8_1_workspace_ptr = ops.optional_q8_1_workspace_ptr(
            scratch,
            1,
            geometry.hidden_size,
            enabled=(
                flags.q4k_selected_dual_dp4a_enabled()
                or flags.t16_selected_dp4a_enabled()
                or flags.raw_selected_dp4a_enabled()
                or ops.selected_pair_requires_q8_1_input(gate_weight, up_weight)
            ),
        )
        ops.selected_raw_moe_dual(
            gate_weight, up_weight, scratch.post_norm.ptr,
            scratch.moe_selected_experts.ptr, scratch.ffn_gate_up.ptr,
            scratch.ffn_gate_up.ptr + gate_rows_nbytes,
            x_rows=1, rows=selected_rows, num_experts=cfg.expert_count,
            in_features=geometry.hidden_size, out_features=cfg.expert_feed_forward_length,
            q8_1_workspace_ptr=q8_1_workspace_ptr, x_f32_ptr=post_norm_f32_ptr,
            stream=stream, runtime=runtime,
        )
        if f32_selected_intermediate:
            kernels.silu_mul_separate_out_f32(
                scratch.ffn_gate_up.ptr,
                scratch.ffn_gate_up.ptr + gate_rows_nbytes,
                scratch.ffn_intermediate_f32.ptr,
                rows=selected_rows,
                features=cfg.expert_feed_forward_length,
                stream=stream,
                runtime=runtime,
            )
            kernels.f32_to_bf16(
                scratch.ffn_intermediate_f32.ptr,
                scratch.ffn_intermediate.ptr,
                selected_rows * cfg.expert_feed_forward_length,
                stream=stream,
                library=ops.cast_library(),
                runtime=runtime,
            )
        else:
            kernels.silu_mul_separate_out_bf16(
                scratch.ffn_gate_up.ptr,
                scratch.ffn_gate_up.ptr + gate_rows_nbytes,
                scratch.ffn_intermediate.ptr,
                rows=selected_rows,
                features=cfg.expert_feed_forward_length,
                stream=stream,
                runtime=runtime,
            )
    f32_selected_down = bool(prefer_f32_selected_down)
    expert_down_weighted = False
    if down_call == "weighted_down":
        expert_down_weighted = ops.weighted_selected_raw_moe_linear(
            down_weight,
            scratch.ffn_intermediate.ptr,
            scratch.moe_selected_experts.ptr,
            scratch.moe_routing_weights.ptr,
            scratch.moe_down_out.ptr,
            tokens=1,
            top_k=selected_rows,
            num_experts=cfg.expert_count,
            in_features=cfg.expert_feed_forward_length,
            out_features=geometry.hidden_size,
            stream=stream,
            runtime=runtime,
        )
    if not expert_down_weighted:
        ops.selected_raw_moe_linear(
            down_weight,
            scratch.ffn_intermediate.ptr,
            scratch.moe_selected_experts.ptr,
            scratch.moe_down_out_f32.ptr if f32_selected_down else scratch.moe_down_out.ptr,
            x_rows=selected_rows,
            rows=selected_rows,
            num_experts=cfg.expert_count,
            in_features=cfg.expert_feed_forward_length,
            out_features=geometry.hidden_size,
            q8_1_workspace_ptr=ops.optional_q8_1_workspace_ptr(
                scratch,
                selected_rows,
                cfg.expert_feed_forward_length,
                enabled=(
                    flags.t16_selected_dp4a_enabled()
                    or flags.raw_selected_dp4a_enabled()
                    or ops.selected_gemv_requires_q8_1_input(down_weight)
                ),
            ),
            x_f32_ptr=scratch.ffn_intermediate_f32.ptr if f32_selected_intermediate else None,
            prefer_f32_out=f32_selected_down,
            backend=geometry.backend,
            stream=stream,
            runtime=runtime,
        )
    return f32_selected_down, expert_down_weighted


def run_post_attention_moe_rows(
    layer: Qwen35GGUFResidentLayerWeights,
    config: Qwen35GGUFConfig,
    geometry: FfnGeometry,
    kernels: FfnKernels,
    ops: FfnOps,
    flags: FfnFlags,
    out_ptr: int,
    scratch,
    *,
    rows: int,
    runtime: HipRuntime | None = None,
    stream: int = 0,
    expert_sidecar: ExpertSidecar | None = None,
    expert_pack8_library: object | None = None,
    residual_f32_ptr: int | None = None,
    out_f32_ptr: int | None = None,
    post_norm_f32_ptr: int | None = None,
    stage_timings: dict[str, float] | None = None,
    sync_stage_timings: bool = False,
    stage_prefix: str = "target_block_ffn_moe",
    gpu_stage_recorder: StageRecorder | None = None,
    q8_t16_rowtile_all: bool | None = None,
) -> None:
    """Bulk MoE feed-forward for ``rows > 1``.

    ``expert_pack8_library`` is the explicit form of
    ``getattr(runner, "_expert_pack8_library", None)``.
    """

    cfg = config
    if not cfg.is_moe:
        raise ValueError("MoE path requires qwen35moe GGUF config")
    if rows <= 1:
        raise ValueError("bulk MoE rows path requires rows > 1")
    if not hasattr(scratch, "moe_shared_gate_logits"):
        raise ValueError("qwen35moe bulk MoE scratch is missing shared-gate logits")
    runtime = runtime or get_hip_runtime()
    sync_stages = bool(sync_stage_timings and stage_timings is not None)
    t_stage = time.perf_counter() if sync_stages else 0.0
    top_k = int(cfg.expert_used_count)
    if top_k <= 0:
        raise ValueError("qwen35moe GGUF expert_used_count must be positive")
    selected_rows = rows * top_k
    f32_residual = residual_f32_ptr is not None or out_f32_ptr is not None
    if f32_residual and (residual_f32_ptr is None or out_f32_ptr is None):
        raise ValueError("residual_f32_ptr and out_f32_ptr must be provided together")

    router_f32_ptr = (
        post_norm_f32_ptr
        if post_norm_f32_ptr is not None and flags.verify_f32_post_norm_router_enabled()
        else None
    )
    selected_f32_ptr = (
        post_norm_f32_ptr
        if post_norm_f32_ptr is not None and flags.verify_f32_post_norm_selected_q8_enabled()
        else None
    )
    shared_f32_ptr = (
        post_norm_f32_ptr
        if post_norm_f32_ptr is not None and flags.verify_f32_post_norm_shared_q8_enabled()
        else None
    )
    router_fn = (
        ops.router_logits_f32_hidden
        if router_f32_ptr is not None
        else ops.router_logits_bf16_hidden
    )
    router_hidden_ptr = int(router_f32_ptr) if router_f32_ptr is not None else scratch.post_norm.ptr
    router_fn(
        router_hidden_ptr,
        layer.weight("ffn_gate_inp"),
        scratch.moe_router_logits.ptr,
        rows,
        geometry.hidden_size,
        cfg.expert_count,
        stream=stream,
        runtime=runtime,
    )
    router_fn(
        router_hidden_ptr,
        layer.weight("ffn_gate_inp_shexp"),
        scratch.moe_shared_gate_logits.ptr,
        rows,
        geometry.hidden_size,
        1,
        stream=stream,
        runtime=runtime,
    )
    kernels.qwen35_router_select(
        scratch.moe_router_logits.ptr,
        scratch.moe_selected_experts.ptr,
        scratch.moe_routing_weights.ptr,
        rows,
        cfg.expert_count,
        cfg.expert_count,
        top_k,
        threads=ops.prefill_router_select_threads(geometry.backend),
        stream=stream,
        runtime=runtime,
    )
    t_stage = ops.mark_sync_stage(
        runtime,
        stage_timings,
        sync_stages,
        f"{stage_prefix}_router",
        t_stage,
    )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_router")

    gate_weight = layer.weight("ffn_gate_exps")
    up_weight = layer.weight("ffn_up_exps")
    down_weight = layer.weight("ffn_down_exps")
    prefer_f32_selected_down = (
        False
        if expert_sidecar is not None
        else ops.use_f32_selected_down(down_weight, scratch, f32_residual)
    )
    f32_selected_intermediate = ops.use_f32_selected_intermediate(
        scratch,
        prefer_f32_selected_down,
    )
    if (not f32_residual) and ops.moe_rows_compact_wmma(
        layer,
        gate_weight,
        up_weight,
        down_weight,
        out_ptr,
        scratch,
        rows=rows,
        selected_rows=selected_rows,
        top_k=top_k,
        stream=stream,
        runtime=runtime,
        gpu_stage_recorder=gpu_stage_recorder,
        stage_prefix=f"{stage_prefix}_compact_wmma",
        q8_t16_rowtile_all=q8_t16_rowtile_all,
    ):
        ops.mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_compact_wmma",
            t_stage,
        )
        return
    gate_rows_nbytes = selected_rows * cfg.expert_feed_forward_length * DType.BF16.itemsize
    expert_silu_ready = False
    if expert_sidecar is not None and ops.selected_expert_pack8_moe_pair(
        expert_sidecar.tensor("ffn_gate_exps"),
        expert_sidecar.tensor("ffn_up_exps"),
        scratch.post_norm.ptr,
        scratch.moe_selected_experts.ptr,
        scratch.ffn_gate_up.ptr,
        scratch.ffn_gate_up.ptr + gate_rows_nbytes,
        backend=geometry.backend,
        x_rows=rows,
        rows=selected_rows,
        num_experts=cfg.expert_count,
        in_features=geometry.hidden_size,
        out_features=cfg.expert_feed_forward_length,
        stream=stream,
        runtime=runtime,
        library=expert_pack8_library,
    ):
        pass
    elif expert_sidecar is not None:
        ops.selected_expert_pack8_moe_linear(
            expert_sidecar.tensor("ffn_gate_exps"),
            scratch.post_norm.ptr,
            scratch.moe_selected_experts.ptr,
            scratch.ffn_gate_up.ptr,
            backend=geometry.backend,
            x_rows=rows,
            rows=selected_rows,
            num_experts=cfg.expert_count,
            in_features=geometry.hidden_size,
            out_features=cfg.expert_feed_forward_length,
            stream=stream,
            runtime=runtime,
            library=expert_pack8_library,
        )
        ops.selected_expert_pack8_moe_linear(
            expert_sidecar.tensor("ffn_up_exps"),
            scratch.post_norm.ptr,
            scratch.moe_selected_experts.ptr,
            scratch.ffn_gate_up.ptr + gate_rows_nbytes,
            backend=geometry.backend,
            x_rows=rows,
            rows=selected_rows,
            num_experts=cfg.expert_count,
            in_features=geometry.hidden_size,
            out_features=cfg.expert_feed_forward_length,
            stream=stream,
            runtime=runtime,
            library=expert_pack8_library,
        )
    elif kernels.selected_ffn_modes(
        gate_weight.spec.quant_key, up_weight.spec.quant_key, down_weight.spec.quant_key,
        allow_legacy_silu=False,
    )[0] == "dual_silu" and ops.selected_raw_moe_pair_silu(
        gate_weight,
        up_weight,
        scratch.post_norm.ptr,
        scratch.moe_selected_experts.ptr,
        scratch.ffn_intermediate.ptr,
        x_rows=rows,
        rows=selected_rows,
        num_experts=cfg.expert_count,
        in_features=geometry.hidden_size,
        out_features=cfg.expert_feed_forward_length,
        stream=stream,
        runtime=runtime,
        allow_legacy=False,
    ):
        expert_silu_ready = True
    else:
        # The Q4T16 dual+SiLU fusion is decode-only for now.  In rows>1
        # bulk prefill the extra exp/rounding work in the GEMV accumulator
        # did not pay for the removed SiLU launch, so keep the split path.
        q8_1_workspace_ptr = ops.optional_q8_1_workspace_ptr(
            scratch,
            rows,
            geometry.hidden_size,
            enabled=(
                flags.q4k_selected_dual_dp4a_enabled()
                or flags.t16_selected_dp4a_enabled()
                or flags.raw_selected_dp4a_enabled()
                or ops.selected_pair_requires_q8_1_input(gate_weight, up_weight)
            ),
        )
        ops.selected_raw_moe_dual(
            gate_weight, up_weight, scratch.post_norm.ptr,
            scratch.moe_selected_experts.ptr, scratch.ffn_gate_up.ptr,
            scratch.ffn_gate_up.ptr + gate_rows_nbytes,
            x_rows=rows, rows=selected_rows, num_experts=cfg.expert_count,
            in_features=geometry.hidden_size, out_features=cfg.expert_feed_forward_length,
            q8_1_workspace_ptr=q8_1_workspace_ptr, x_f32_ptr=selected_f32_ptr,
            stream=stream, runtime=runtime, stage_timings=stage_timings,
            sync_stage_timings=sync_stages, stage_prefix=f"{stage_prefix}_expert_gate_up",
        )
    t_stage = ops.mark_sync_stage(
        runtime,
        stage_timings,
        sync_stages,
        f"{stage_prefix}_expert_gate_up",
        t_stage,
    )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_expert_gate_up")
    if not expert_silu_ready:
        if f32_selected_intermediate:
            kernels.silu_mul_separate_out_f32(
                scratch.ffn_gate_up.ptr,
                scratch.ffn_gate_up.ptr + gate_rows_nbytes,
                scratch.ffn_intermediate_f32.ptr,
                rows=selected_rows,
                features=cfg.expert_feed_forward_length,
                stream=stream,
                runtime=runtime,
            )
            kernels.f32_to_bf16(
                scratch.ffn_intermediate_f32.ptr,
                scratch.ffn_intermediate.ptr,
                selected_rows * cfg.expert_feed_forward_length,
                stream=stream,
                library=ops.cast_library(),
                runtime=runtime,
            )
        else:
            kernels.silu_mul_separate_out_bf16(
                scratch.ffn_gate_up.ptr,
                scratch.ffn_gate_up.ptr + gate_rows_nbytes,
                scratch.ffn_intermediate.ptr,
                rows=selected_rows,
                features=cfg.expert_feed_forward_length,
                stream=stream,
                runtime=runtime,
            )
        t_stage = ops.mark_sync_stage(
            runtime,
            stage_timings,
            sync_stages,
            f"{stage_prefix}_expert_silu",
            t_stage,
        )
        if gpu_stage_recorder is not None:
            gpu_stage_recorder.mark(f"{stage_prefix}_expert_silu")
    selected_down_is_f32 = False
    expert_down_weighted = False
    if expert_sidecar is not None:
        ops.selected_expert_pack8_moe_linear(
            expert_sidecar.tensor("ffn_down_exps"),
            scratch.ffn_intermediate.ptr,
            scratch.moe_selected_experts.ptr,
            scratch.moe_down_out.ptr,
            backend=geometry.backend,
            x_rows=selected_rows,
            rows=selected_rows,
            num_experts=cfg.expert_count,
            in_features=cfg.expert_feed_forward_length,
            out_features=geometry.hidden_size,
            stream=stream,
            runtime=runtime,
            library=expert_pack8_library,
        )
    else:
        selected_down_is_f32 = prefer_f32_selected_down
        if kernels.selected_ffn_modes(
            gate_weight.spec.quant_key, up_weight.spec.quant_key, down_weight.spec.quant_key,
            allow_legacy_silu=False, f32_down=selected_down_is_f32,
            weighted_down=not f32_residual,
        )[1] == "weighted_down":
            expert_down_weighted = ops.weighted_selected_raw_moe_linear(
                down_weight,
                scratch.ffn_intermediate.ptr,
                scratch.moe_selected_experts.ptr,
                scratch.moe_routing_weights.ptr,
                scratch.moe_down_out.ptr,
                tokens=rows,
                top_k=top_k,
                num_experts=cfg.expert_count,
                in_features=cfg.expert_feed_forward_length,
                out_features=geometry.hidden_size,
                stream=stream,
                runtime=runtime,
            )
        if not expert_down_weighted:
            ops.selected_raw_moe_linear(
                down_weight,
                scratch.ffn_intermediate.ptr,
                scratch.moe_selected_experts.ptr,
                scratch.moe_down_out_f32.ptr if selected_down_is_f32 else scratch.moe_down_out.ptr,
                x_rows=selected_rows,
                rows=selected_rows,
                num_experts=cfg.expert_count,
                in_features=cfg.expert_feed_forward_length,
                out_features=geometry.hidden_size,
                q8_1_workspace_ptr=ops.optional_q8_1_workspace_ptr(
                    scratch,
                    selected_rows,
                    cfg.expert_feed_forward_length,
                    enabled=(
                        flags.t16_selected_dp4a_enabled()
                        or flags.raw_selected_dp4a_enabled()
                        or ops.selected_gemv_requires_q8_1_input(down_weight)
                    ),
                ),
                x_f32_ptr=scratch.ffn_intermediate_f32.ptr if f32_selected_intermediate else None,
                prefer_f32_out=selected_down_is_f32,
                backend=geometry.backend,
                stream=stream,
                runtime=runtime,
                stage_timings=stage_timings,
                sync_stage_timings=sync_stages,
                stage_prefix=f"{stage_prefix}_expert_down",
            )
    t_stage = ops.mark_sync_stage(
        runtime,
        stage_timings,
        sync_stages,
        f"{stage_prefix}_expert_down",
        t_stage,
    )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_expert_down")

    if ops.shared_gate_up_from_f32_post_norm(
        layer.weight("ffn_gate_shexp"),
        layer.weight("ffn_up_shexp"),
        shared_f32_ptr,
        scratch.moe_shared_gate.ptr,
        scratch.moe_shared_up.ptr,
        rows=rows,
        hidden_size=geometry.hidden_size,
        shared_ffn=cfg.expert_shared_feed_forward_length,
        stream=stream,
        runtime=runtime,
    ):
        kernels.silu_mul_separate_out_bf16(
            scratch.moe_shared_gate.ptr,
            scratch.moe_shared_up.ptr,
            scratch.moe_shared_intermediate.ptr,
            rows=rows,
            features=cfg.expert_shared_feed_forward_length,
            stream=stream,
            runtime=runtime,
        )
    elif kernels.launch_gguf_linear_pair_concat(
        layer.weight("ffn_gate_shexp"),
        layer.weight("ffn_up_shexp"),
        scratch.post_norm.ptr,
        scratch.ffn_gate_up.ptr,
        rows=rows,
        in_features=geometry.hidden_size,
        out_features=cfg.expert_shared_feed_forward_length,
        stream=stream,
        runtime=runtime,
    ):
        kernels.silu_mul_dual_out_bf16(
            scratch.ffn_gate_up.ptr,
            scratch.moe_shared_intermediate.ptr,
            rows=rows,
            features=cfg.expert_shared_feed_forward_length,
            stream=stream,
            runtime=runtime,
        )
    else:
        if not kernels.launch_gguf_linear_pair(
            layer.weight("ffn_gate_shexp"),
            layer.weight("ffn_up_shexp"),
            scratch.post_norm.ptr,
            scratch.moe_shared_gate.ptr,
            scratch.moe_shared_up.ptr,
            rows=rows,
            in_features=geometry.hidden_size,
            out_features=cfg.expert_shared_feed_forward_length,
            stream=stream,
            runtime=runtime,
        ):
            kernels.launch_gguf_linear(
                layer.weight("ffn_gate_shexp"),
                scratch.post_norm.ptr,
                scratch.moe_shared_gate.ptr,
                rows=rows,
                in_features=geometry.hidden_size,
                out_features=cfg.expert_shared_feed_forward_length,
                stream=stream,
                runtime=runtime,
            )
            kernels.launch_gguf_linear(
                layer.weight("ffn_up_shexp"),
                scratch.post_norm.ptr,
                scratch.moe_shared_up.ptr,
                rows=rows,
                in_features=geometry.hidden_size,
                out_features=cfg.expert_shared_feed_forward_length,
                stream=stream,
                runtime=runtime,
            )
        kernels.silu_mul_separate_out_bf16(
            scratch.moe_shared_gate.ptr,
            scratch.moe_shared_up.ptr,
            scratch.moe_shared_intermediate.ptr,
            rows=rows,
            features=cfg.expert_shared_feed_forward_length,
            stream=stream,
            runtime=runtime,
        )
    t_stage = ops.mark_sync_stage(
        runtime,
        stage_timings,
        sync_stages,
        f"{stage_prefix}_shared_gate_up_silu",
        t_stage,
    )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_shared_gate_up_silu")
    shared_down_is_f32 = False
    if (
        ops.use_f32_shared_down(scratch, f32_residual, selected_down_is_f32)
        and ops.gguf_linear_bf16_f32_output(
            layer.weight("ffn_down_shexp"),
            scratch.moe_shared_intermediate.ptr,
            scratch.moe_shared_out_f32.ptr,
            rows=rows,
            in_features=cfg.expert_shared_feed_forward_length,
            out_features=geometry.hidden_size,
            stream=stream,
            runtime=runtime,
        )
    ):
        kernels.f32_to_bf16(
            scratch.moe_shared_out_f32.ptr,
            scratch.moe_shared_out.ptr,
            rows * geometry.hidden_size,
            stream=stream,
            library=ops.cast_library(),
            runtime=runtime,
        )
        shared_down_is_f32 = True
    else:
        kernels.launch_gguf_linear(
            layer.weight("ffn_down_shexp"),
            scratch.moe_shared_intermediate.ptr,
            scratch.moe_shared_out.ptr,
            rows=rows,
            in_features=cfg.expert_shared_feed_forward_length,
            out_features=geometry.hidden_size,
            stream=stream,
            runtime=runtime,
        )
    t_stage = ops.mark_sync_stage(
        runtime,
        stage_timings,
        sync_stages,
        f"{stage_prefix}_shared_down",
        t_stage,
    )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_shared_down")
    if f32_residual:
        if selected_down_is_f32 and shared_down_is_f32:
            kernels.weighted_sum_f32_shared_f32_gate_combine_residual_batch_out_f32_accum_f32w(
                scratch.moe_down_out_f32.ptr,
                scratch.moe_routing_weights.ptr,
                scratch.moe_shared_out_f32.ptr,
                scratch.moe_shared_gate_logits.ptr,
                int(residual_f32_ptr),
                int(out_f32_ptr),
                rows,
                top_k,
                geometry.hidden_size,
                1,
                stream=stream,
                runtime=runtime,
            )
        elif selected_down_is_f32:
            kernels.weighted_sum_f32_shared_gate_combine_residual_batch_out_f32_accum_f32w(
                scratch.moe_down_out_f32.ptr,
                scratch.moe_routing_weights.ptr,
                scratch.moe_shared_out.ptr,
                scratch.moe_shared_gate_logits.ptr,
                int(residual_f32_ptr),
                int(out_f32_ptr),
                rows,
                top_k,
                geometry.hidden_size,
                1,
                stream=stream,
                runtime=runtime,
            )
        else:
            ops.f32_moe_combine_batch_out_fn()(
                scratch.moe_down_out.ptr,
                scratch.moe_routing_weights.ptr,
                scratch.moe_shared_out.ptr,
                scratch.moe_shared_gate_logits.ptr,
                int(residual_f32_ptr),
                int(out_f32_ptr),
                rows,
                top_k,
                geometry.hidden_size,
                1,
                stream=stream,
                runtime=runtime,
            )
        kernels.f32_to_bf16(
            int(out_f32_ptr),
            out_ptr,
            rows * geometry.hidden_size,
            stream=stream,
            library=ops.cast_library(),
            runtime=runtime,
        )
    else:
        if expert_down_weighted:
            kernels.shared_gate_combine_residual_batch_out_bf16(
                scratch.moe_down_out.ptr,
                scratch.moe_shared_out.ptr,
                scratch.moe_shared_gate_logits.ptr,
                scratch.residual.ptr,
                out_ptr,
                rows,
                geometry.hidden_size,
                1,
                stream=stream,
                runtime=runtime,
            )
        else:
            kernels.weighted_sum_shared_gate_combine_residual_batch_out_bf16_f32w(
                scratch.moe_down_out.ptr,
                scratch.moe_routing_weights.ptr,
                scratch.moe_shared_out.ptr,
                scratch.moe_shared_gate_logits.ptr,
                scratch.residual.ptr,
                out_ptr,
                rows,
                top_k,
                geometry.hidden_size,
                1,
                stream=stream,
                runtime=runtime,
            )
    ops.mark_sync_stage(
        runtime,
        stage_timings,
        sync_stages,
        f"{stage_prefix}_combine_residual",
        t_stage,
    )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_combine_residual")
