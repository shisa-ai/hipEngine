"""Standalone attention execution for one Qwen hybrid GGUF layer.

This module owns the two shared attention leaves and the two c=1 decode
attention bodies that use them:

* :func:`run_attention_norm_rows` - the fused attention RMSNorm leaf.
* :func:`run_linear_attention_alpha_beta_rows` - the alpha/beta projection leaf.
* :func:`run_linear_attention_attn_only` - the GDN attention body.
* :func:`run_full_attention_attn_only` - the paged full-attention body.

Layer weights, model config, geometry, scratch/state views, the runtime and
every lazily selected operation arrive as explicit arguments. Nothing here
reads a runner, a session, or module globals for policy: the selectors in
:class:`AttentionSelectors` and the predicates in :class:`AttentionPolicy` are
narrow callbacks supplied by the runner adapter, which keeps each library,
short-batch, GDN variant, KV-write and split-gate choice as lazy as it is today.
Kernels are reached through this module's own imports, so a
``monkeypatch.setattr`` on this module still intercepts the launch the body
issues; nothing caches a kernel between calls.

Two scratch responsibilities stay on the explicit workspace object because the
body mutates them mid-execution: publishing the decode position
(``set_full_attention_position``) and growing the split-K partials. The grown
split buffers and the split count are therefore read from the workspace after
the growth callback, exactly as before.

Bulk prefill attention bodies and the FFN half of a layer are out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime
from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    qwen35_full_attn_gate_mul_bf16,
    qwen35_paged_attn_decode_int8_key_bf16_value_gqa_splitk_gate_bf16_spans,
    qwen35_paged_full_attn_decode_context_bf16_spans,
)
from hipengine.kernels.hip_gfx1100.attention.paged_kv_write import (
    qwen35_write_paged_kv_int8_key_bf16_value_spans,
    qwen35_write_paged_kv_mixed_value_bf16_spans,
)
from hipengine.kernels.hip_gfx1100.convert import bf16_to_f32, f32_to_bf16
from hipengine.kernels.hip_gfx1100.fused import (
    gguf_qwen35_head_rmsnorm_partial_rotary_position_f32_weight,
    gguf_rmsnorm_f32_f32_weight,
    gguf_rmsnorm_f32_f32_weight_out_f32,
)
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    qwen35_linear_attn_conv_decode_bf16,
)
from hipengine.kernels.hip_gfx1100.rotary.qwen35_rotary import qwen35_split_qgate_bf16
from hipengine.runtime.gguf_linear import (
    GGUF_ACTIVATION_BF16,
    GGUF_ACTIVATION_F32,
    GGUF_OUTPUT_F32,
    launch_gguf_linear,
    launch_gguf_linear_pair,
    launch_gguf_linear_triple,
)


class StageRecorder(Protocol):
    """The GPU stage marker seam (``_HipEventStageRecorder.mark``)."""

    def mark(self, name: str, *aliases: str) -> None: ...


@dataclass(frozen=True)
class AttentionGeometry:
    """Runner geometry properties the attention bodies read.

    Only the body that reads a field supplies it. The shared norm and alpha/beta
    leaves read ``hidden_size`` alone, so a caller that runs only those leaves
    leaves the rest ``None`` and never evaluates an unused runner property; a
    body that needs an unsupplied field fails on the ``None`` instead of using a
    placeholder value.
    """

    hidden_size: int
    q_width: int | None = None
    kv_width: int | None = None
    linear_qkv_width: int | None = None
    ssm_value_dim: int | None = None
    fp16_recurrent_state: bool | None = None


@dataclass(frozen=True)
class AttentionSelectors:
    """Lazily resolved attention operations, supplied by the runner adapter.

    Every field is called at the point the body needs it, so a library build,
    a variant lookup or a monkeypatched module attribute resolves exactly when
    it does today.
    """

    # Attention norm leaf and its shared cast handle.
    norm_residual_kernel: Callable[..., Callable[..., None]]
    cast_library: Callable[[], object]
    # Alpha/beta projection leaf routes.
    linear_supports_f32_activation: Callable[[object], bool]
    linear_supports_f32_activation_f32_output: Callable[..., bool]
    try_launch_dense_q8_pair_dp4a_f32: Callable[..., bool]
    try_launch_dense_q8_pair_dp4a_f32_out: Callable[..., bool]
    # Linear-attention decode leaves.
    try_launch_dense_f32_alpha_beta_conv_decode: Callable[..., bool]
    gdn_decode_gate_kernel: Callable[[bool | None], Callable[..., None]]
    gdn_decode_output_fusion_for_weight: Callable[[object], Callable[..., None] | None]
    gdn_decode_output_cast_for_weight: Callable[[object], Callable[..., None] | None]
    # Full-attention decode leaves.
    paged_attn_decode_library: Callable[[], object]
    paged_kv_write_library: Callable[[], object]
    full_attn_qk_postprocess_fn: Callable[[], Callable[..., None] | None]
    full_attn_decode_short_batch_fn: Callable[[object], Callable[..., None]]
    int8_kv_append_write_fn: Callable[[object], Callable[..., None]]
    int8_kv_decode_gate_fn: Callable[[object], Callable[..., None]]


@dataclass(frozen=True)
class AttentionPolicy:
    """Environment and backend-policy predicates, resolved per call."""

    verify_f32_attention_norm: Callable[[], bool]
    verify_f32_linear_projections: Callable[[], bool]
    verify_f32_alpha_beta: Callable[[], bool]
    use_short_full_attention_split_decode: Callable[..., bool]
    use_full_attention_split_decode: Callable[[int], bool]
    full_attention_split_gate_fn: Callable[..., Callable[..., None]]


@dataclass(frozen=True)
class LinearAttentionViews:
    """Linear-attention activation buffers and the layer's GDN state.

    The alpha/beta projection leaf reads the ``linear_alpha*`` / ``linear_beta*``
    pair and ``q8_1_workspace_owner`` alone, and its caller holds a layer and no
    layer id, so every other field defaults to ``None`` and that caller supplies
    the pair and nothing else. The attention body reads all of them and rejects a
    missing GDN state pair with a named error.
    """

    linear_alpha: object
    linear_beta: object
    linear_alpha_f32: object | None = None
    linear_beta_f32: object | None = None
    q8_1_workspace_owner: object | None = None
    norm: object | None = None
    norm_f32: object | None = None
    linear_qkv: object | None = None
    linear_z: object | None = None
    conv_out: object | None = None
    recurrent_out: object | None = None
    recurrent_bf16: object | None = None
    conv_state: object | None = None
    recurrent_state: object | None = None


@dataclass(frozen=True)
class FullAttentionViews:
    """Full-attention activation buffers, KV state and per-layer spans."""

    norm: object
    full_q: object
    full_k: object
    full_v: object
    full_query_raw: object
    full_key_raw: object
    full_query: object
    full_key: object
    full_gate: object
    full_gated: object
    full_attn_context: object
    cos_table: object
    sin_table: object
    position_tensor: object
    max_positions: int
    block_size: int
    kv_storage_dtype: object
    int8_kv_value_bf16: bool
    append_spans: object
    decode_spans: object
    key_cache: object
    value_cache: object
    layer_append_spans: object
    layer_decode_spans: object
    bf16_mirror_cache: tuple[object, object] | None


def run_attention_norm_rows(
    *,
    hidden_ptr: int,
    hidden_f32_ptr: int | None,
    weight_ptr: int,
    out_ptr: int,
    out_f32_ptr: int | None = None,
    rows: int,
    eps: float,
    stream: int,
    runtime: HipRuntime,
    geometry: AttentionGeometry,
    selectors: AttentionSelectors,
    policy: AttentionPolicy,
) -> int | None:
    if hidden_f32_ptr is None:
        norm_kernel = selectors.norm_residual_kernel(
            layer="rmsnorm",
            rows=rows,
            hidden_size=geometry.hidden_size,
        )
        norm_kernel(
            hidden_ptr,
            weight_ptr,
            out_ptr,
            rows=rows,
            hidden_size=geometry.hidden_size,
            eps=eps,
            stream=stream,
            runtime=runtime,
        )
        return None
    if policy.verify_f32_attention_norm() and out_f32_ptr is not None:
        gguf_rmsnorm_f32_f32_weight_out_f32(
            int(hidden_f32_ptr),
            weight_ptr,
            int(out_f32_ptr),
            rows=rows,
            hidden_size=geometry.hidden_size,
            eps=eps,
            stream=stream,
            runtime=runtime,
        )
        f32_to_bf16(
            int(out_f32_ptr),
            out_ptr,
            rows * geometry.hidden_size,
            stream=stream,
            library=selectors.cast_library(),
            runtime=runtime,
        )
        return int(out_f32_ptr)
    else:
        gguf_rmsnorm_f32_f32_weight(
            int(hidden_f32_ptr),
            weight_ptr,
            out_ptr,
            rows=rows,
            hidden_size=geometry.hidden_size,
            eps=eps,
            stream=stream,
            runtime=runtime,
        )
        return None


def run_linear_attention_alpha_beta_rows(
    layer,
    norm_ptr: int,
    norm_f32_ptr: int | None,
    views: LinearAttentionViews,
    *,
    cfg,
    geometry: AttentionGeometry,
    rows: int,
    stream: int,
    runtime: HipRuntime,
    selectors: AttentionSelectors,
    policy: AttentionPolicy,
) -> str:
    if (
        norm_f32_ptr is not None
        and policy.verify_f32_linear_projections()
        and views.linear_alpha_f32 is not None
        and views.linear_beta_f32 is not None
    ):
        if (
            selectors.linear_supports_f32_activation_f32_output(
                layer.weight("ssm_alpha"), rows=rows
            )
            and selectors.linear_supports_f32_activation_f32_output(
                layer.weight("ssm_beta"), rows=rows
            )
        ):
            launch_gguf_linear(
                layer.weight("ssm_alpha"),
                int(norm_f32_ptr),
                views.linear_alpha_f32.ptr,
                rows=rows,
                in_features=geometry.hidden_size,
                out_features=cfg.ssm_time_step_rank,
                activation_dtype=GGUF_ACTIVATION_F32,
                output_dtype=GGUF_OUTPUT_F32,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                layer.weight("ssm_beta"),
                int(norm_f32_ptr),
                views.linear_beta_f32.ptr,
                rows=rows,
                in_features=geometry.hidden_size,
                out_features=cfg.ssm_time_step_rank,
                activation_dtype=GGUF_ACTIVATION_F32,
                output_dtype=GGUF_OUTPUT_F32,
                stream=stream,
                runtime=runtime,
            )
            cast_library = selectors.cast_library()
            f32_to_bf16(
                views.linear_alpha_f32.ptr,
                views.linear_alpha.ptr,
                rows * cfg.ssm_time_step_rank,
                stream=stream,
                library=cast_library,
                runtime=runtime,
            )
            f32_to_bf16(
                views.linear_beta_f32.ptr,
                views.linear_beta.ptr,
                rows * cfg.ssm_time_step_rank,
                stream=stream,
                library=cast_library,
                runtime=runtime,
            )
            return "f32_singletons_f32_out"
        if selectors.try_launch_dense_q8_pair_dp4a_f32_out(
            layer.weight("ssm_alpha"),
            layer.weight("ssm_beta"),
            int(norm_f32_ptr),
            views.linear_alpha_f32.ptr,
            views.linear_beta_f32.ptr,
            views.q8_1_workspace_owner,
            rows=rows,
            in_features=geometry.hidden_size,
            out_features_a=cfg.ssm_time_step_rank,
            out_features_b=cfg.ssm_time_step_rank,
            stream=stream,
            runtime=runtime,
        ):
            cast_library = selectors.cast_library()
            f32_to_bf16(
                views.linear_alpha_f32.ptr,
                views.linear_alpha.ptr,
                rows * cfg.ssm_time_step_rank,
                stream=stream,
                library=cast_library,
                runtime=runtime,
            )
            f32_to_bf16(
                views.linear_beta_f32.ptr,
                views.linear_beta.ptr,
                rows * cfg.ssm_time_step_rank,
                stream=stream,
                library=cast_library,
                runtime=runtime,
            )
            return "dense_q8_dp4a_f32_out"
    if norm_f32_ptr is not None and policy.verify_f32_alpha_beta():
        if selectors.try_launch_dense_q8_pair_dp4a_f32(
            layer.weight("ssm_alpha"),
            layer.weight("ssm_beta"),
            int(norm_f32_ptr),
            views.linear_alpha.ptr,
            views.linear_beta.ptr,
            views.q8_1_workspace_owner,
            rows=rows,
            in_features=geometry.hidden_size,
            out_features_a=cfg.ssm_time_step_rank,
            out_features_b=cfg.ssm_time_step_rank,
            stream=stream,
            runtime=runtime,
        ):
            return "dense_q8_dp4a_f32"
        if (
            selectors.linear_supports_f32_activation(layer.weight("ssm_alpha"))
            and selectors.linear_supports_f32_activation(layer.weight("ssm_beta"))
        ):
            launch_gguf_linear(
                layer.weight("ssm_alpha"),
                int(norm_f32_ptr),
                views.linear_alpha.ptr,
                rows=rows,
                in_features=geometry.hidden_size,
                out_features=cfg.ssm_time_step_rank,
                activation_dtype=GGUF_ACTIVATION_F32,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                layer.weight("ssm_beta"),
                int(norm_f32_ptr),
                views.linear_beta.ptr,
                rows=rows,
                in_features=geometry.hidden_size,
                out_features=cfg.ssm_time_step_rank,
                activation_dtype=GGUF_ACTIVATION_F32,
                stream=stream,
                runtime=runtime,
            )
            return "f32_singletons"
    if cfg.is_moe:
        launch_gguf_linear(
            layer.weight("ssm_alpha"),
            norm_ptr,
            views.linear_alpha.ptr,
            rows=rows,
            in_features=geometry.hidden_size,
            out_features=cfg.ssm_time_step_rank,
            stream=stream,
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("ssm_beta"),
            norm_ptr,
            views.linear_beta.ptr,
            rows=rows,
            in_features=geometry.hidden_size,
            out_features=cfg.ssm_time_step_rank,
            stream=stream,
            runtime=runtime,
        )
        return "singletons"
    if launch_gguf_linear_pair(
        layer.weight("ssm_alpha"),
        layer.weight("ssm_beta"),
        norm_ptr,
        views.linear_alpha.ptr,
        views.linear_beta.ptr,
        rows=rows,
        in_features=geometry.hidden_size,
        out_features=cfg.ssm_time_step_rank,
        stream=stream,
        runtime=runtime,
    ):
        return "pair"
    launch_gguf_linear(
        layer.weight("ssm_alpha"),
        norm_ptr,
        views.linear_alpha.ptr,
        rows=rows,
        in_features=geometry.hidden_size,
        out_features=cfg.ssm_time_step_rank,
        stream=stream,
        runtime=runtime,
    )
    launch_gguf_linear(
        layer.weight("ssm_beta"),
        norm_ptr,
        views.linear_beta.ptr,
        rows=rows,
        in_features=geometry.hidden_size,
        out_features=cfg.ssm_time_step_rank,
        stream=stream,
        runtime=runtime,
    )
    return "fallback"


def run_linear_attention_attn_only(
    layer_id: int,
    layer,
    hidden_ptr: int,
    attn_out_ptr: int,
    views: LinearAttentionViews,
    *,
    cfg,
    geometry: AttentionGeometry,
    backend: str,
    runtime: HipRuntime,
    selectors: AttentionSelectors,
    policy: AttentionPolicy,
    hidden_f32_ptr: int | None = None,
    input_norm_ptr: int | None = None,
    stream: int = 0,
    stage_prefix: str = "decode_linear_attn",
    gpu_stage_recorder: StageRecorder | None = None,
) -> None:
    conv_state = views.conv_state
    recurrent_state = views.recurrent_state
    if conv_state is None or recurrent_state is None:
        raise ValueError(f"layer {layer_id} has no linear-attention state")
    if input_norm_ptr is None:
        attn_norm_f32_ptr = run_attention_norm_rows(
            hidden_ptr=hidden_ptr,
            hidden_f32_ptr=hidden_f32_ptr,
            weight_ptr=layer.weight("attn_norm").allocation().tensor.ptr,
            out_ptr=views.norm.ptr,
            out_f32_ptr=(views.norm_f32.ptr if views.norm_f32 is not None else None),
            rows=1,
            eps=cfg.rms_norm_eps,
            stream=stream,
            runtime=runtime,
            geometry=geometry,
            selectors=selectors,
            policy=policy,
        )
    else:
        if int(input_norm_ptr) != int(views.norm.ptr):
            raise ValueError("prefused linear-attention input norm must use scratch.norm")
        attn_norm_f32_ptr = None
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_norm")
    pair_fused = launch_gguf_linear_pair(
        layer.weight("attn_qkv"),
        layer.weight("attn_gate"),
        views.norm.ptr,
        views.linear_qkv.ptr,
        views.linear_z.ptr,
        rows=1,
        in_features=geometry.hidden_size,
        out_features=geometry.linear_qkv_width,
        out_features_b=cfg.ssm_inner_size,
        stream=stream,
        runtime=runtime,
    )
    if not pair_fused:
        launch_gguf_linear(
            layer.weight("attn_qkv"),
            views.norm.ptr,
            views.linear_qkv.ptr,
            rows=1,
            in_features=geometry.hidden_size,
            out_features=geometry.linear_qkv_width,
            stream=stream,
            runtime=runtime,
        )
        launch_gguf_linear(
            layer.weight("attn_gate"),
            views.norm.ptr,
            views.linear_z.ptr,
            rows=1,
            in_features=geometry.hidden_size,
            out_features=cfg.ssm_inner_size,
            stream=stream,
            runtime=runtime,
        )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_qkv_gate")
    linear_alpha_ptr = views.linear_alpha.ptr
    linear_beta_ptr = views.linear_beta.ptr
    alpha_beta_conv_fused = (
        attn_norm_f32_ptr is None
        and selectors.try_launch_dense_f32_alpha_beta_conv_decode(
            layer.weight("ssm_alpha"),
            layer.weight("ssm_beta"),
            views.norm.ptr,
            linear_alpha_ptr,
            linear_beta_ptr,
            views.linear_qkv.ptr,
            conv_state.ptr,
            layer.weight("ssm_conv1d").allocation().tensor.ptr,
            views.conv_out.ptr,
            rows=1,
            in_features=geometry.hidden_size,
            out_features=cfg.ssm_time_step_rank,
            channels=geometry.linear_qkv_width,
            kernel_size=cfg.ssm_conv_kernel,
            backend=backend,
            stream=stream,
            runtime=runtime,
        )
    )
    if not alpha_beta_conv_fused:
        run_linear_attention_alpha_beta_rows(
            layer,
            views.norm.ptr,
            attn_norm_f32_ptr,
            views,
            cfg=cfg,
            geometry=geometry,
            rows=1,
            stream=stream,
            runtime=runtime,
            selectors=selectors,
            policy=policy,
        )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_alpha_beta")
    if not alpha_beta_conv_fused:
        qwen35_linear_attn_conv_decode_bf16(
            views.linear_qkv.ptr,
            conv_state.ptr,
            layer.weight("ssm_conv1d").allocation().tensor.ptr,
            views.conv_out.ptr,
            geometry.linear_qkv_width,
            cfg.ssm_conv_kernel,
            stream=stream,
            runtime=runtime,
        )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_conv")
    ssm_out_weight = layer.weight("ssm_out")
    gdn_output_fusion = selectors.gdn_decode_output_fusion_for_weight(
        ssm_out_weight
    )
    gdn_args = (
        views.conv_out.ptr,
        views.linear_z.ptr,
        linear_alpha_ptr,
        linear_beta_ptr,
        layer.weight("ssm_dt_bias").allocation().tensor.ptr,
        layer.weight("ssm_a").allocation().tensor.ptr,
        layer.weight("ssm_norm").allocation().tensor.ptr,
        recurrent_state.ptr,
        views.recurrent_out.ptr,
    )
    if gdn_output_fusion is None:
        selectors.gdn_decode_gate_kernel(geometry.fp16_recurrent_state)(
            *gdn_args,
            cfg.rms_norm_eps,
            cfg.ssm_group_count,
            cfg.ssm_time_step_rank,
            cfg.ssm_state_size,
            geometry.ssm_value_dim,
            stream=stream,
            runtime=runtime,
        )
    else:
        gdn_output_fusion(
            *gdn_args,
            views.recurrent_bf16.ptr,
            cfg.rms_norm_eps,
            cfg.ssm_group_count,
            cfg.ssm_time_step_rank,
            cfg.ssm_state_size,
            geometry.ssm_value_dim,
            stream=stream,
            runtime=runtime,
        )
    ssm_out_input_ptr = views.recurrent_out.ptr
    ssm_out_activation_dtype = GGUF_ACTIVATION_F32
    output_cast = selectors.gdn_decode_output_cast_for_weight(ssm_out_weight)
    if gdn_output_fusion is not None:
        ssm_out_input_ptr = views.recurrent_bf16.ptr
        ssm_out_activation_dtype = GGUF_ACTIVATION_BF16
    elif output_cast is not None:
        output_cast(
            views.recurrent_out.ptr,
            views.recurrent_bf16.ptr,
            cfg.ssm_inner_size,
            stream=stream,
            library=selectors.cast_library(),
            runtime=runtime,
        )
        ssm_out_input_ptr = views.recurrent_bf16.ptr
        ssm_out_activation_dtype = GGUF_ACTIVATION_BF16
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_gdn")
    launch_gguf_linear(
        ssm_out_weight,
        ssm_out_input_ptr,
        attn_out_ptr,
        rows=1,
        in_features=cfg.ssm_inner_size,
        out_features=geometry.hidden_size,
        activation_dtype=ssm_out_activation_dtype,
        stream=stream,
        runtime=runtime,
    )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_ssm_out")


def run_full_attention_attn_only(
    layer_id: int,
    layer,
    hidden_ptr: int,
    attn_out_ptr: int,
    views: FullAttentionViews,
    scratch,
    *,
    cfg,
    geometry: AttentionGeometry,
    backend: str,
    runtime: HipRuntime,
    selectors: AttentionSelectors,
    policy: AttentionPolicy,
    ensure_full_attn_split_rows: Callable[[int], None],
    external_dms_full_attention: Callable[..., None] | None,
    position: int,
    hidden_f32_ptr: int | None = None,
    input_norm_ptr: int | None = None,
    stream: int = 0,
    attention_max_context_len: int | None = None,
    stage_prefix: str = "decode_full_attn",
    gpu_stage_recorder: StageRecorder | None = None,
    kv_write_only: bool = False,
) -> None:
    if external_dms_full_attention is not None:
        return external_dms_full_attention(
            layer_id=layer_id,
            hidden_ptr=hidden_ptr,
            attn_out_ptr=attn_out_ptr,
            scratch=scratch,
            position=position,
            hidden_f32_ptr=hidden_f32_ptr,
            input_norm_ptr=input_norm_ptr,
            stream=stream,
            stage_prefix=stage_prefix,
            gpu_stage_recorder=gpu_stage_recorder,
        )
    cast_library = selectors.cast_library()
    kv_write_library = selectors.paged_kv_write_library()
    if int(scratch.position_host[0]) != int(position):
        scratch.set_full_attention_position(position, runtime)
    if input_norm_ptr is None:
        run_attention_norm_rows(
            hidden_ptr=hidden_ptr,
            hidden_f32_ptr=hidden_f32_ptr,
            weight_ptr=layer.weight("attn_norm").allocation().tensor.ptr,
            out_ptr=views.norm.ptr,
            rows=1,
            eps=cfg.rms_norm_eps,
            stream=stream,
            runtime=runtime,
            geometry=geometry,
            selectors=selectors,
            policy=policy,
        )
    elif int(input_norm_ptr) != int(views.norm.ptr):
        raise ValueError("prefused full-attention input norm must use scratch.norm")
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_norm")
    qkv_complete = False
    if not kv_write_only:
        qkv_complete = launch_gguf_linear_triple(
            layer.weight("attn_q"),
            layer.weight("attn_k"),
            layer.weight("attn_v"),
            views.norm.ptr,
            views.full_q.ptr,
            views.full_k.ptr,
            views.full_v.ptr,
            rows=1,
            in_features=geometry.hidden_size,
            out_features=2 * geometry.q_width,
            out_features_b=geometry.kv_width,
            out_features_c=geometry.kv_width,
            stream=stream,
            runtime=runtime,
        )
    if not qkv_complete:
        if not kv_write_only:
            launch_gguf_linear(
                layer.weight("attn_q"),
                views.norm.ptr,
                views.full_q.ptr,
                rows=1,
                in_features=geometry.hidden_size,
                out_features=2 * geometry.q_width,
                stream=stream,
                runtime=runtime,
            )
        if not launch_gguf_linear_pair(
            layer.weight("attn_k"),
            layer.weight("attn_v"),
            views.norm.ptr,
            views.full_k.ptr,
            views.full_v.ptr,
            rows=1,
            in_features=geometry.hidden_size,
            out_features=geometry.kv_width,
            stream=stream,
            runtime=runtime,
        ):
            launch_gguf_linear(
                layer.weight("attn_k"),
                views.norm.ptr,
                views.full_k.ptr,
                rows=1,
                in_features=geometry.hidden_size,
                out_features=geometry.kv_width,
                stream=stream,
                runtime=runtime,
            )
            launch_gguf_linear(
                layer.weight("attn_v"),
                views.norm.ptr,
                views.full_v.ptr,
                rows=1,
                in_features=geometry.hidden_size,
                out_features=geometry.kv_width,
                stream=stream,
                runtime=runtime,
            )
    qk_postprocess = selectors.full_attn_qk_postprocess_fn()
    if qk_postprocess is None:
        qwen35_split_qgate_bf16(
            views.full_q.ptr,
            views.full_query_raw.ptr,
            views.full_gate.ptr,
            1,
            cfg.head_count,
            cfg.key_length,
            stream=stream,
            runtime=runtime,
        )
        bf16_to_f32(
            views.full_k.ptr,
            views.full_key_raw.ptr,
            geometry.kv_width,
            stream=stream,
            library=cast_library,
            runtime=runtime,
        )
        gguf_qwen35_head_rmsnorm_partial_rotary_position_f32_weight(
            views.full_query_raw.ptr,
            views.full_key_raw.ptr,
            layer.weight("attn_q_norm").allocation().tensor.ptr,
            layer.weight("attn_k_norm").allocation().tensor.ptr,
            views.cos_table.ptr,
            views.sin_table.ptr,
            views.position_tensor.ptr,
            views.full_query.ptr,
            views.full_key.ptr,
            cfg.rms_norm_eps,
            cfg.head_count,
            cfg.head_count_kv,
            cfg.key_length,
            cfg.rope_dimension_count,
            views.max_positions,
            stream=stream,
            runtime=runtime,
        )
    else:
        qk_postprocess(
            views.full_q.ptr,
            views.full_k.ptr,
            layer.weight("attn_q_norm").allocation().tensor.ptr,
            layer.weight("attn_k_norm").allocation().tensor.ptr,
            views.cos_table.ptr,
            views.sin_table.ptr,
            views.position_tensor.ptr,
            views.full_query.ptr,
            views.full_key.ptr,
            views.full_gate.ptr,
            cfg.rms_norm_eps,
            cfg.head_count,
            cfg.head_count_kv,
            cfg.key_length,
            cfg.rope_dimension_count,
            views.max_positions,
            stream=stream,
            runtime=runtime,
        )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_qkv_head_norm_rope")
    key_cache, value_cache = views.key_cache, views.value_cache
    append_spans = views.layer_append_spans
    decode_spans = views.layer_decode_spans
    paged_attn_library = selectors.paged_attn_decode_library()
    bf16_mirror_cache = views.bf16_mirror_cache
    layer_uses_int8_kv = (
        views.kv_storage_dtype == DType.INT8_PER_TOKEN_HEAD
        and append_spans.scale_metadata is not None
    )
    if layer_uses_int8_kv:
        metadata = append_spans.scale_metadata
        bf16_to_f32(
            views.full_v.ptr,
            views.full_key_raw.ptr,
            geometry.kv_width,
            stream=stream,
            library=cast_library,
            runtime=runtime,
        )
        if views.int8_kv_value_bf16:
            qwen35_write_paged_kv_int8_key_bf16_value_spans(
                views.full_key.ptr,
                views.full_key_raw.ptr,
                key_cache.ptr,
                value_cache.ptr,
                metadata.k_scale.ptr,
                append_spans,
                views.block_size,
                cfg.head_count_kv,
                cfg.key_length,
                stream=stream,
                library=kv_write_library,
                runtime=runtime,
            )
        else:
            int8_append_write_fn = selectors.int8_kv_append_write_fn(metadata)
            int8_append_write_fn(
                views.full_key.ptr,
                views.full_key_raw.ptr,
                key_cache.ptr,
                value_cache.ptr,
                metadata.k_scale.ptr,
                metadata.v_scale.ptr,
                append_spans,
                views.block_size,
                cfg.head_count_kv,
                cfg.key_length,
                stream=stream,
                library=kv_write_library,
                runtime=runtime,
            )
        if bf16_mirror_cache is not None:
            mirror_key_cache, mirror_value_cache = bf16_mirror_cache
            qwen35_write_paged_kv_mixed_value_bf16_spans(
                views.full_key.ptr,
                views.full_v.ptr,
                mirror_key_cache.ptr,
                mirror_value_cache.ptr,
                views.append_spans,
                views.block_size,
                cfg.head_count_kv,
                cfg.key_length,
                stream=stream,
                library=kv_write_library,
                runtime=runtime,
            )
    else:
        qwen35_write_paged_kv_mixed_value_bf16_spans(
            views.full_key.ptr,
            views.full_v.ptr,
            key_cache.ptr,
            value_cache.ptr,
            append_spans,
            views.block_size,
            cfg.head_count_kv,
            cfg.key_length,
            stream=stream,
            library=kv_write_library,
            runtime=runtime,
        )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_kv_write")
    if kv_write_only:
        return
    active_context = int(position) + 1
    attention_context_cap = (
        active_context if attention_max_context_len is None else int(attention_max_context_len)
    )
    if attention_context_cap < active_context:
        raise ValueError("attention_max_context_len must cover the current decode position")
    if attention_context_cap > int(views.max_positions):
        raise ValueError("attention_max_context_len exceeds GGUF resident cache capacity")
    if layer_uses_int8_kv and bf16_mirror_cache is None:
        metadata = decode_spans.scale_metadata
        if metadata is None:
            raise RuntimeError("GGUF INT8 full-attention decode requires scale metadata")
        chunk_size = int(views.block_size)
        num_splits = min(
            int(scratch.full_attn_split_count),
            max(1, (attention_context_cap + chunk_size - 1) // chunk_size),
        )
        ensure_full_attn_split_rows(1)
        if views.int8_kv_value_bf16:
            qwen35_paged_attn_decode_int8_key_bf16_value_gqa_splitk_gate_bf16_spans(
                views.full_query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                metadata.k_scale.ptr,
                views.full_gate.ptr,
                views.full_gated.ptr,
                scratch.full_attn_split_partial.ptr,
                scratch.full_attn_split_m.ptr,
                scratch.full_attn_split_l.ptr,
                decode_spans,
                chunk_size,
                num_splits,
                views.block_size,
                cfg.head_count,
                cfg.head_count_kv,
                cfg.key_length,
                cfg.key_length,
                1,
                cfg.key_length ** -0.5,
                stream=stream,
                library=paged_attn_library,
                runtime=runtime,
            )
        else:
            int8_decode_gate_fn = selectors.int8_kv_decode_gate_fn(metadata)
            int8_decode_gate_fn(
                views.full_query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                metadata.k_scale.ptr,
                metadata.v_scale.ptr,
                views.full_gate.ptr,
                views.full_gated.ptr,
                scratch.full_attn_split_partial.ptr,
                scratch.full_attn_split_m.ptr,
                scratch.full_attn_split_l.ptr,
                decode_spans,
                chunk_size,
                num_splits,
                views.block_size,
                cfg.head_count,
                cfg.head_count_kv,
                cfg.key_length,
                cfg.key_length,
                1,
                cfg.key_length ** -0.5,
                stream=stream,
                library=paged_attn_library,
                runtime=runtime,
            )
    else:
        if bf16_mirror_cache is not None:
            key_cache, value_cache = bf16_mirror_cache
            decode_spans = views.decode_spans
        use_short_split = (
            bf16_mirror_cache is None
            and policy.use_short_full_attention_split_decode(
                cfg,
                backend=backend,
                block_size=views.block_size,
                active_context=attention_context_cap,
            )
        )
        if use_short_split or policy.use_full_attention_split_decode(
            attention_context_cap
        ):
            chunk_size = int(views.block_size)
            num_splits = min(
                int(scratch.full_attn_split_count),
                max(1, (attention_context_cap + chunk_size - 1) // chunk_size),
            )
            ensure_full_attn_split_rows(1)
            split_gate_fn = policy.full_attention_split_gate_fn(
                cfg,
                backend=backend,
                block_size=views.block_size,
                num_splits=num_splits,
                active_context=attention_context_cap,
            )
            split_gate_fn(
                views.full_query.ptr,
                key_cache.ptr,
                value_cache.ptr,
                views.full_gate.ptr,
                views.full_gated.ptr,
                scratch.full_attn_split_partial.ptr,
                scratch.full_attn_split_m.ptr,
                scratch.full_attn_split_l.ptr,
                decode_spans,
                chunk_size,
                num_splits,
                views.block_size,
                cfg.head_count,
                cfg.head_count_kv,
                cfg.key_length,
                cfg.key_length,
                1,
                cfg.key_length ** -0.5,
                stream=stream,
                library=paged_attn_library,
                runtime=runtime,
            )
        else:
            short_batch_max_context = max(
                0,
                int(
                    backend_package_capability(
                        backend,
                        "GGUF_SHORT_C1_BATCH_ATTN_MAX_CONTEXT",
                        0,
                    )
                ),
            )
            if 0 < attention_context_cap <= short_batch_max_context:
                selectors.full_attn_decode_short_batch_fn(decode_spans)(
                    views.full_query.ptr,
                    key_cache.ptr,
                    value_cache.ptr,
                    views.full_attn_context.ptr,
                    decode_spans,
                    1,
                    attention_context_cap,
                    views.block_size,
                    cfg.head_count,
                    cfg.head_count_kv,
                    cfg.key_length,
                    cfg.key_length ** -0.5,
                    stream=stream,
                    library=paged_attn_library,
                    runtime=runtime,
                )
            else:
                qwen35_paged_full_attn_decode_context_bf16_spans(
                    views.full_query.ptr,
                    key_cache.ptr,
                    value_cache.ptr,
                    views.full_attn_context.ptr,
                    decode_spans,
                    attention_context_cap,
                    views.block_size,
                    cfg.head_count,
                    cfg.head_count_kv,
                    cfg.key_length,
                    cfg.key_length ** -0.5,
                    stream=stream,
                    library=paged_attn_library,
                    runtime=runtime,
                )
            qwen35_full_attn_gate_mul_bf16(
                views.full_attn_context.ptr,
                views.full_gate.ptr,
                views.full_gated.ptr,
                geometry.q_width,
                stream=stream,
                library=paged_attn_library,
                runtime=runtime,
            )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_core_gate")
    launch_gguf_linear(
        layer.weight("attn_output"),
        views.full_gated.ptr,
        attn_out_ptr,
        rows=1,
        in_features=geometry.q_width,
        out_features=geometry.hidden_size,
        stream=stream,
        runtime=runtime,
    )
    if gpu_stage_recorder is not None:
        gpu_stage_recorder.mark(f"{stage_prefix}_output")
