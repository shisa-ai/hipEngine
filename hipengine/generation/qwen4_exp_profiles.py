"""Qwen4Exp gfx1151 strict and certified production prefill/decode plans."""

from __future__ import annotations

import os
import warnings
from dataclasses import replace
from typing import Any

from hipengine.execution_profiles import (
    ExecutionProfile,
    ResolvedRuntimeProfile,
    RuntimeProfileKey,
    RuntimeProfilePlan,
    VariantSelection,
    register_runtime_profile_plan,
    registered_runtime_profile_keys,
)

from hipengine.runtime.gguf_linear import (
    DENSE_WIDE_TAIL_VARIANT_ENV,
    NORM_F16_PUBLICATION_ENV,
)

QWEN4_EXP_MODEL = "qwen4_exp_gguf"
QWEN4_EXP_BACKEND = "hip_gfx1151"
QWEN4_EXP_QUANTS = ("gguf_q4_k_m", "gguf_ud_q4_k_xl")
PRODUCTION_MOE_PREFILL_ENV = "HIPENGINE_QWEN4_EXP_PRODUCTION_MOE_PREFILL"
PREBINDER_CONFLICT_ENV = "HIPENGINE_QWEN4_EXP_PREBINDER_CONFLICT"
_last_prebinder_conflicts: dict[str, tuple[str, str]] = {}
# Values this module has written, so a later bind can tell its own previous
# write apart from a value the caller set. Strict-then-production binding in one
# process is normal and is not a conflict.
_binder_written: dict[str, str] = {}
# Fast-profile route selectors this module currently has published.
_fast_published: set[str] = set()


PROFILE_Q5_1_DOWN_M1_ENV = "HIPENGINE_QWEN4_EXP_PROFILE_Q5_1_DOWN_M1"
# The opt-in fast profile is registered for the quant whose production plan is
# in-envelope; it layers reduced-precision operand routes on that plan. Each
# route passes the outer safety floor and keeps its strict fallback, and is
# not yet certified against the calibrated production envelope.
FAST_QUANTS = ("gguf_ud_q4_k_xl",)
_T16_DOWN_ENV = "HIPENGINE_QWEN4_EXP_T16_DOWN"
_FAST_T16_DOWN_VARIANT = "selected_t16_expert_major_wmma_comp_bf16_bf16_out"
_FAST_T16_DOWN_SCOPES = (
    "prefill_rows_ge2_lt64_exact_grouped_q5_1_down",
    "prefill_rows_ge64_lt512_exact_grouped_q5_1_down",
    "prefill_rows_ge512_k640_exact_grouped_q5_1_down",
)
_FAST_T16_DOWN_EVIDENCE = (
    "benchmarks/results/2026-10-09-qwen4exp-t16-down-engine-ab/engine_ab.json"
)
# Routed-expert slot roles whose T16 tiles the fast profile makes resident at
# bind time. The tiles sit beside the raw weights the decode owners read.
_FAST_T16_RESIDENT_SLOTS = ("expert_down",)
PRODUCTION_GDN_PEER_PREFILL_LAYERS = tuple(range(35, 48))
PRODUCTION_GDN_COLWARPS_PREFILL_LAYERS = tuple(range(27, 48))
PRODUCTION_QSA_FLASH_PREFILL_LAYERS = tuple(range(35, 48))
PRODUCTION_WMMA_MOE_PREFILL_LAYERS = tuple(range(27, 48))
# Deepest scope certified against the calibrated production envelope for the
# f16 WMMA dense Q8_0 prefill route: mean KL 3.796e-4 against the 1e-3 limit,
# p95 1.762e-3, top-1 1538/1548, 3/3 deterministic, no scope failures.
# 12-47 and 8-47 screen inconclusive and are not certified.
#
# Retired as the production default at this scope on 2026-09-17: the wide-row
# route below holds the identical envelope (byte-identical strict and candidate
# logits digests) and is 5.4% faster at 1K/4K. The selector and its env var stay
# registered for explicit opt-in and for re-gating.
PRODUCTION_Q8_WMMA_PREFILL_LAYERS = tuple(range(16, 48))
# Deepest scope certified for the wide-row f16 dense Q8_0 prefill route:
# the same envelope as the WMMA route at 16-47 (mean KL 3.80e-4, p95 1.76e-3,
# p99 5.42e-3, max 1.53e-2, top-1 1538/1548, 3/3 deterministic, no scope
# failures), with `strict_logits_sha256` and `candidate_logits_sha256` equal to
# that route's gate artifact byte for byte.
PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS = tuple(range(13, 48))
PRODUCTION_Q4_IU8_PREFILL_LAYERS = tuple(range(35, 48))
PRODUCTION_Q5_1_MMQ_PREFILL_LAYERS = tuple(range(32, 48))
PRODUCTION_Q4_K_MMQ_PREFILL_LAYERS = tuple(range(35, 48))
PRODUCTION_Q4_DP4A_DECODE_LAYERS = (
    0, 2, 5, 6, 8, 9, 10, 11
) + tuple(range(13, 48))
PRODUCTION_ARITHMETIC_RECOVERY_FLAGS = (
    "PRODUCTION_MOE_PREFILL", "Q4_IU8_PREFILL", "Q8_0_SELECTED_WMMA_DOWN",
    "Q8_MMQ_PREFILL", "GR_IU8", "GR_IU8_DOWN",
    "GDN_PEER_PREFILL", "Q4_DP4A64",
    "QSA_H256_WAVE_PREFILL", "QSA_HEAD_PAIR", "QSA_FLASH_PREFILL",
    "QSA_ORDERED_DECODE", "QSA_ORDERED_DECODE_V2", "GR_WIDE",
)
_ARITHMETIC_RECOVERY_EVIDENCE = (
    "benchmarks/results/2026-09-14-q8-prefill-numerics/artifact.json"
)
PRODUCTION_Q8_QSA_RESTORED_FLAGS = {
    "Q4_DP4A64": "1",
    "Q8_0_SELECTED_WMMA_DOWN": "1",
    "Q8_DOWN_VARIANT": "selected_grouped_blockscale_guarded_prefill_bf16_bf16_out",
    "QSA_H256_WAVE_PREFILL": "page256",
    "QSA_HEAD_PAIR": "quad",
    "QSA_ORDERED_DECODE": "1",
    "QSA_ORDERED_DECODE_V2": "1",
}
_Q8_QSA_RESTORATION_EVIDENCE = (
    "benchmarks/results/2026-09-15-q8-blockscale-restoration/artifact.json"
)
_MOE_WMMA_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-29-gfx1151-qwen38-flash-next-wmma-moe27-production.json"
)
_STACK_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-29-gfx1151-qwen38-flash-next-production-mmq-prefill-dp4a43-stack.json"
)
_DECODE_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-29-gfx1151-qwen38-flash-next-production-dp4a-safe43-decode.json"
)
_GDN_PEER_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-29-gfx1151-qwen38-flash-next-production-gdn-peer35.json"
)
_GDN_COLWARPS_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps27-production.json"
)
_GR_SIGMOID_MEAN_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-31-gfx1151-qwen38-flash-next-p3-gr-sigmoid-mean.json"
)
_ROUTER_F32_TILE4_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-31-gfx1151-qwen38-flash-next-p3-router-f32-tile4.json"
)
_ROUTER_WARP_REDUCE_EVIDENCE = (
    "benchmarks/results/2026-09-27-router-warp-reduce/"
)
_ROUTER_DPP_EVIDENCE = (
    "benchmarks/results/2026-10-05-qwen4exp-router-dpp/README.md"
)
# Named router producer and temporary rollback switch. Strict/Q4_K_M keep
# tile4; the measured UD production profile selects ordered DPP by default.
_ROUTER_VARIANT_ENV = "HIPENGINE_QWEN4_EXP_ROUTER_VARIANT"
_ROUTER_TILE4_VARIANT = "f32_hidden_token_tile4_dense_exact"
_ROUTER_WARP_REDUCE_VARIANT = "f32_hidden_warp_reduce_dense_exact"
_ROUTER_DPP_VARIANT = "f32_hidden_token_tile4_dpp_exact"
# One-barrier ordered DPP passed the natural-input/full-state canonical gate
# and improves prefill in the same-residency twelve-case model-wall campaign.
# Keep the exact tile4 parent as the strict fallback and temporary rollback.
_ROUTER_VARIANTS = (
    _ROUTER_TILE4_VARIANT,
    _ROUTER_WARP_REDUCE_VARIANT,
    _ROUTER_DPP_VARIANT,
)


def _selected_router_producer_variant(*, recovery: bool = False) -> str:
    """Return the router producer for the named quant/profile plan.

    UD production defaults to ordered DPP; Q4_K_M remains on tile4. The
    binder publishes the manifest selection for the runner's launch-time
    reader, including the strict tile4 fallback and explicit rollback.
    """

    default = _ROUTER_DPP_VARIANT if recovery else _ROUTER_TILE4_VARIANT
    requested = os.environ.get(_ROUTER_VARIANT_ENV, "").strip() or default
    if requested not in _ROUTER_VARIANTS:
        raise ValueError(f"{_ROUTER_VARIANT_ENV} must be one of {_ROUTER_VARIANTS}; got {requested!r}")
    return requested


_GDN_PREFILL_VARIANT_ENV = "HIPENGINE_QWEN4_EXP_GDN_PREFILL_VARIANT"
_GDN_WIDE_VARIANT = "qwen4exp_sigmoid_wave_norm_prefill"
_GDN_NARROW_VARIANT = "qwen4exp_sigmoid_wave_norm_narrow_prefill"
_GDN_NARROW_SCOPE = "prefill_rows_2_4096_serial_gdn_hk16_hv48_d128"
_GDN_NARROW_EVIDENCE = (
    "benchmarks/results/2026-10-06-qwen4exp-gdn-narrow-index/README.md"
)


def _selected_gdn_prefill_variant() -> str:
    """Resolve the UD production plan, including explicit parent rollback."""

    requested = os.environ.get(_GDN_PREFILL_VARIANT_ENV, "").strip() or _GDN_NARROW_VARIANT
    if requested not in (_GDN_WIDE_VARIANT, _GDN_NARROW_VARIANT):
        raise ValueError(f"{_GDN_PREFILL_VARIANT_ENV} must name the wide or narrow wave-norm variant")
    return requested


_DENSE_WIDE_TAIL_PARENT = "dense_wide256_f16in_f32_f32_out"
_DENSE_WIDE_TAIL_PADDED = "dense_wide256_padded_f16in_f32_f32_out"
_DENSE_WIDE_TAIL_SCOPE = "prefill_rows_512_4096_k10240_n320_dense_wide_layers13_47"
_DENSE_WIDE_TAIL_EVIDENCE = (
    "benchmarks/results/2026-10-06-qwen4exp-dense-wide-padded-gate/README.md"
)


def _dense_wide_tail_selection(*, production: bool) -> VariantSelection:
    selected = "coltile8_rowbatch4_f32_f32_out"
    if production:
        selected = os.environ.get(DENSE_WIDE_TAIL_VARIANT_ENV, "").strip() or _DENSE_WIDE_TAIL_PADDED
        if selected not in (_DENSE_WIDE_TAIL_PARENT, _DENSE_WIDE_TAIL_PADDED):
            raise ValueError(f"{DENSE_WIDE_TAIL_VARIANT_ENV} must name the parent or padded F16 variant")
    return _selection(
        "linear", _DENSE_WIDE_TAIL_SCOPE, selected,
        "coltile8_rowbatch4_f32_f32_out", "gguf_q8_0",
        evidence=_DENSE_WIDE_TAIL_EVIDENCE if production else None,
    )


_NORM_F16_SCOPE = "prefill_rows4096_branches4_h2560_hc_down_layers13_47"
_NORM_F16_EVIDENCE = "benchmarks/results/2026-10-06-qwen4exp-norm-f16-publish/README.md"


def _norm_f16_selection(*, production: bool) -> VariantSelection:
    enabled = production and os.environ.get(NORM_F16_PUBLICATION_ENV, "1").strip().lower() in {
        "1", "true", "yes", "on",
    }
    return _selection(
        "gr_grouped_rmsnorm", _NORM_F16_SCOPE,
        "f32_f16_publish" if enabled else "strict",
        "strict", "bf16_f32",
        evidence=_NORM_F16_EVIDENCE if enabled else None,
    )


_ROUTER_LOGITS_SELECTED_VARIANT = _selected_router_producer_variant()
_ROUTER_UD_LOGITS_SELECTED_VARIANT = _selected_router_producer_variant(recovery=True)
_ROUTER_VARIANT_EVIDENCE = {
    _ROUTER_TILE4_VARIANT: _ROUTER_F32_TILE4_EVIDENCE,
    _ROUTER_WARP_REDUCE_VARIANT: _ROUTER_WARP_REDUCE_EVIDENCE,
    _ROUTER_DPP_VARIANT: _ROUTER_DPP_EVIDENCE,
}
_ROUTER_LOGITS_EVIDENCE = _ROUTER_VARIANT_EVIDENCE[_ROUTER_LOGITS_SELECTED_VARIANT]
_GR_UP_SIGMOID_MEAN_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-31-gfx1151-qwen38-flash-next-p3-gr-up-sigmoid-mean.json"
)
_QSA_DENSE_FIXED256_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-31-gfx1151-qwen38-flash-next-p4-qsa-dense-fixed256.json"
)
_DEVICE_ARGMAX_EVIDENCE = (
    "benchmarks/results/"
    "2026-08-31-gfx1151-qwen38-flash-next-p5-device-argmax.json"
)
_QSA_ORDERED_DECODE_EVIDENCE = (
    "benchmarks/results/"
    "2026-09-02-gfx1151-qwen38-flash-next-p6-qsa-ordered-decode.json"
)
_QSA_ORDERED_DECODE_V2_EVIDENCE = (
    "benchmarks/results/"
    "2026-09-08-framework-qwen4exp-qsa-ordered-v2-production.json"
)
_HALO_PF13_EVIDENCE = (
    "benchmarks/results/"
    "2026-09-04-gfx1151-qwen38-flash-next-halo-pf13-production-refresh.json"
)
_GR_UP_DPP_ENGINE_EVIDENCE = (
    "benchmarks/results/2026-10-05-qwen4exp-gr-dpp-reduction/README.md"
)


def _selection(
    layer: str,
    scope: str,
    selected: str,
    fallback: str,
    quant: str,
    *,
    evidence: str | None = None,
) -> VariantSelection:
    return VariantSelection(
        layer=layer,
        scope=scope,
        selected_variant=selected,
        strict_fallback_variant=fallback,
        registry_quant=quant,
        evidence_artifact=evidence,
    )


def _strict_selections(*, recovery: bool = False) -> tuple[VariantSelection, ...]:
    selections = (
        _selection(
            "gdn_recurrence_norm_gate", "prefill_rows_ge16_hk16_hv32_48_d128_layers27_47",
            "qwen4exp_sigmoid_strict_prefill", "qwen4exp_sigmoid_strict_prefill", "f32_state",
        ),
        _selection("linear","prefill_rows_ge512_k2560_n12288_raw_q",
                   "coltile8_rowbatch4_f32_f32_out","coltile8_rowbatch4_f32_f32_out","gguf_q8_0"),
        _selection(
            "linear","prefill_rows_ge512_token_major_mapped_q8_down",
            "selected_gemv_bf16_bf16_out","selected_gemv_bf16_bf16_out","gguf_q8_0",
        ),
        _selection(
            "linear", "prefill_gdn_qkv_ssm_prepacked_mmq",
            "coltile8_rowbatch4_f32_f32_out", "coltile8_rowbatch4_f32_f32_out",
            "gguf_q8_0",
        ),
        _selection(
            "linear", "prefill_exact_q8_f32_coltile",
            "coltile8_rowbatch4_f32_f32_out", "coltile8_rowbatch4_f32_f32_out",
            "gguf_q8_0",
        ),
        _selection(
            "gdn_recurrence_norm_gate", "prefill_rows_ge2_serial_gdn_hk16_hv48_d128",
            "qwen4exp_sigmoid_strict_prefill", "qwen4exp_sigmoid_strict_prefill",
            "f32_state",
        ),
        _selection(
            "moe_linear", "prefill_rows_ge64_lt512_exact_grouped_q5_1_down",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "gguf_q5_1",
        ),
        _selection(
            "moe_linear", "prefill_rows_ge512_k640_exact_grouped_q5_1_down",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "gguf_q5_1",
        ),
        _selection(
            "moe_linear", "prefill_rows_ge2_lt64_exact_grouped_q4_gate_up",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "gguf_q4_k",
        ),
        _selection(
            "moe_linear", "prefill_rows_ge64_exact_grouped_q4_gate_up",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "gguf_q4_k",
        ),
        _selection(
            "qsa_sparse_attention", "prefill_h256_page256_sparse_rows_ge16",
            "strict_rows_spans", "strict_rows_spans", "bf16_kv",
        ),
        _selection(
            "linear",
            "ungrouped_prefill_rows_ge64_q5k_gate_up",
            "selected_gemv_bf16_bf16_out",
            "selected_gemv_bf16_bf16_out",
            "gguf_q5_k",
        ),
        _selection(
            "argmax",
            "qwen4exp_normal_greedy_output",
            "top1_i64",
            "top1_i64",
            "f32",
            evidence=_DEVICE_ARGMAX_EVIDENCE,
        ),
        _selection(
            "paged_attn_decode",
            "qwen4exp_multirow_dense_qsa",
            "bf16_context_batch_paged_c1_exact_spans",
            "bf16_context_batch_spans",
            "w4_paro",
            evidence=_QSA_DENSE_FIXED256_EVIDENCE,
        ),
        _selection(
            "qsa_sparse_attention",
            "qwen4exp_c1_h256_indexed_sparse_decode",
            "strict_spans",
            "strict_spans",
            "bf16_kv",
        ),
        _selection(
            "qsa_sparse_attention",
            "qwen4exp_c1_h256_indexed_sparse_decode_v2",
            "strict_spans",
            "strict_spans",
            "bf16_kv",
        ),
        _selection(
            "linear+gr_gated_mean",
            "qwen4exp_rows_gt256_gr_up",
            "coltile2_branch4_rowbatch4_f32_exact",
            "coltile8_rowbatch4_f32_f32_out",
            "gguf_q8_0",
            evidence=_GR_UP_SIGMOID_MEAN_EVIDENCE,
        ),
        _selection(
            "router_logits",
            "qwen4exp_multirow_f32_router",
            "f32_hidden_token_tile4_dense_exact",
            "f32_hidden",
            "f32",
            evidence=_ROUTER_F32_TILE4_EVIDENCE,
        ),
        _selection(
            "gr_gated_mean_sigmoid",
            "all_gr_reads_rows_le256",
            "strict",
            "strict_unfused",
            "f32",
            evidence=_GR_SIGMOID_MEAN_EVIDENCE,
        ),
        _selection(
            "moe_linear",
            "prefill_rows_ge2_layers27_47_gate_up",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "gguf_q4_k",
        ),
        _selection(
            "moe_linear",
            "prefill_rows_ge2_layers27_47_down",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "gguf_q5_1",
        ),
        _selection(
            "moe_linear",
            "prefill_rows_ge2_lt64_exact_grouped_q5_1_down",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "gguf_q5_1",
        ),
        _selection(
            "linear",
            "grouped_prefill_rows_lt512_q8_0_expert_down",
            "selected_gemv_bf16_bf16_out",
            "selected_gemv_bf16_bf16_out",
            "gguf_q8_0",
        ),
        _selection(
            "linear", "grouped_prefill_rows_ge512_q8_0_expert_down",
            "selected_gemv_bf16_bf16_out", "selected_gemv_bf16_bf16_out",
            "gguf_q8_0",
        ),
        _selection(
            "linear",
            "prefill_policy_qwen4exp_raw_q8_shapes",
            "coltile8_rowbatch4_f32_f32_out",
            "coltile8_rowbatch4_f32_f32_out",
            "gguf_q8_0",
        ),
        _selection(
            "gdn_recurrence_norm_gate",
            "prefill_rows_ge2_layers27_47_gdn",
            "qwen4exp_sigmoid_strict_prefill",
            "qwen4exp_sigmoid_strict_prefill",
            "f32_state",
        ),
        _selection(
            "linear",
            "decode_c1_calibrated_q4_dp4a_43_layers",
            "selected_dual_silu_logical128_t64_gemv_bf16_bf16_out",
            "selected_dual_silu_logical128_t64_gemv_bf16_bf16_out",
            "gguf_q4_k",
        ),
    )
    if recovery:
        return _recovery_gdn_scopes(
            selections, variant="qwen4exp_sigmoid_strict_prefill", evidence=None,
        ) + (_dense_wide_tail_selection(production=False), _norm_f16_selection(production=False))
    return selections


def _recovery_gdn_scopes(
    selections: tuple[VariantSelection, ...], *, variant: str,
    evidence: str | None,
) -> tuple[VariantSelection, ...]:
    """Split UD's serial scope equally in strict and production plans."""

    result = []
    for selection in selections:
        if selection.layer != "gdn_recurrence_norm_gate":
            result.append(selection)
        elif selection.scope == "prefill_rows_ge2_serial_gdn_hk16_hv48_d128":
            result.append(replace(
                selection, scope=_GDN_NARROW_SCOPE, selected_variant=variant,
                evidence_artifact=evidence or selection.evidence_artifact,
            ))
            result.append(replace(
                selection, scope="prefill_rows_gt4096_serial_gdn_hk16_hv48_d128",
            ))
        # UD disables all peer/tiled routes. Their historical layer scopes
        # overlap this serial scope and do not describe actual dispatch.
    return tuple(result)


def _production_selections(*, recovery: bool = False) -> tuple[VariantSelection, ...]:
    selections = (
        _selection(
            "gdn_recurrence_norm_gate", "prefill_rows_ge16_hk16_hv32_48_d128_layers27_47",
            # Multi-column supersedes the DPP route: same kernel template with an 8-wide
            # per-warp column tile instead of 1, so one k-load feeds eight columns. Measured
            # 1.34x-1.68x faster than the DPP incumbent across 512/1024/2048/4096 tokens at
            # Hk16/Hv48/Dk=Dv=128 on gfx1151, at NMSE ~1.8e-14 (fp32 round-off). Production
            # numerical gates pass (mean KL 8.92e-05, max 4.45e-03, top-1 777/780); the one
            # task finding is shared with the strict arm, not a regression. The DPP route
            # stays registered as the strict fallback.
            #
            # Unconditional on purpose: the declaration has to name the route the binder
            # dispatches. The runner reads HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT, and that
            # binding names the multi-column route for every profile that keeps the
            # column-warp gate open. The recovery profile closes that gate and deactivates
            # tiled variants BY NAME below, so its own declaration is replaced with its exact
            # owner before the manifest is built - an unconditional row therefore leaves the
            # recovery manifest byte-identical while fixing the profile that dispatches it.
            # A conditional row is what let the q4_k_m manifest declare the tiled16 route
            # while 1350 dispatches of the multi-column route ran underneath it.
            "qwen4exp_gdn_tiled16_multi_prefill",
            "qwen4exp_sigmoid_strict_prefill", "f32_state",
            evidence="benchmarks/results/2026-09-30-qwen4exp-gdn-multicolumn-ab/artifact.json",
        ),
        _selection("linear","prefill_rows_ge512_k2560_n12288_raw_q",
                   "mmq128_token64_q8_1_d4x3_guarded_f32_f32_out",
                   "coltile8_rowbatch4_f32_f32_out","gguf_q8_0",
                   evidence="benchmarks/results/2026-09-07-framework-qwen4exp-mmq-token64-production.json"),
        _selection(
            "linear","prefill_rows_ge512_token_major_mapped_q8_down",
            "selected_grouped_row4_register_gemv_bf16_bf16_out",
            "selected_gemv_bf16_bf16_out","gguf_q8_0",
            evidence="benchmarks/results/2026-09-07-framework-qwen4exp-q8-down-register-production.json",
        ),
        _selection(
            "linear", "prefill_exact_q8_f32_coltile",
            "coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out",
            "coltile8_rowbatch4_f32_f32_out",
            "gguf_q8_0",
            evidence="benchmarks/results/2026-09-29-qwen4exp-q8-coltile-pipeline/artifact.json",
        ),
        _selection(
            "gdn_recurrence_norm_gate", "prefill_rows_ge2_serial_gdn_hk16_hv48_d128",
            "qwen4exp_sigmoid_wave_norm_prefill", "qwen4exp_sigmoid_strict_prefill",
            "f32_state",
            evidence="benchmarks/results/2026-09-07-framework-qwen4exp-gdn-wave-norm-owner.json",
        ),
        _selection(
            "moe_linear", "prefill_rows_ge64_lt512_exact_grouped_q5_1_down",
            "selected_grouped_prefill_pair2_fold128_bf16_bf16_out",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "gguf_q5_1",
            evidence="benchmarks/results/2026-09-06-framework-qwen4exp-q51-fold128-production.json",
        ),
        _selection(
            "moe_linear", "prefill_rows_ge512_k640_exact_grouped_q5_1_down",
            "selected_grouped_prefill_pair2_row_publish_bf16_bf16_out",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "gguf_q5_1",
            evidence="benchmarks/results/2026-09-07-framework-qwen4exp-q51-row-publish-production.json",
        ),
        _selection(
            "moe_linear", "prefill_rows_ge2_lt64_exact_grouped_q4_gate_up",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bundle_bf16_bf16_out",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "gguf_q4_k",
            evidence="benchmarks/results/2026-09-05-framework-qwen4exp-prefill-promotion.json",
        ),
        _selection(
            "moe_linear", "prefill_rows_ge64_exact_grouped_q4_gate_up",
            "selected_dual_wmma_iu8_risk_prefill_bf16_bf16_out",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "gguf_q4_k",
            evidence="benchmarks/results/2026-09-07-framework-qwen4exp-q4-iu8-exact-production.json",
        ),
        _selection(
            "qsa_sparse_attention", "prefill_h256_page256_sparse_rows_ge16",
            "strict_h256_head_quad_rows_spans", "strict_rows_spans", "bf16_kv",
            evidence="benchmarks/results/2026-09-07-framework-qwen4exp-qsa-head-quad-production.json",
        ),
        _selection(
            "linear",
            "ungrouped_prefill_rows_ge64_q5k_gate_up",
            "selected_grouped_row4_gemv_bf16_bf16_out",
            "selected_gemv_bf16_bf16_out",
            "gguf_q5_k",
            evidence="benchmarks/results/2026-09-05-framework-qwen4exp-row4-production.json",
        ),
        _selection(
            "argmax",
            "qwen4exp_normal_greedy_output",
            "top1_i64",
            "top1_i64",
            "f32",
            evidence=_DEVICE_ARGMAX_EVIDENCE,
        ),
        _selection(
            "paged_attn_decode",
            "qwen4exp_multirow_dense_qsa",
            "bf16_context_batch_paged_c1_exact_spans",
            "bf16_context_batch_paged_c1_exact_spans",
            "w4_paro",
            evidence=_QSA_DENSE_FIXED256_EVIDENCE,
        ),
        _selection(
            "qsa_sparse_attention",
            "qwen4exp_c1_h256_indexed_sparse_decode",
            "strict_ordered_three_pass_spans",
            "strict_spans",
            "bf16_kv",
            evidence=_QSA_ORDERED_DECODE_EVIDENCE,
        ),
        _selection(
            "qsa_sparse_attention",
            "qwen4exp_c1_h256_indexed_sparse_decode_v2",
            "strict_ordered_three_pass_v2_spans",
            "strict_spans",
            "bf16_kv",
            evidence=_QSA_ORDERED_DECODE_V2_EVIDENCE,
        ),
        _selection(
            "linear+gr_gated_mean",
            "qwen4exp_rows_gt256_gr_up",
            (
                "coltile2_branch4_rowbatch4_wave_scale_dpp_f32_exact"
                if recovery
                else "coltile2_branch4_rowbatch4_wave_scale_f32_exact"
            ),
            "coltile2_branch4_rowbatch4_f32_exact",
            "gguf_q8_0",
            evidence=(
                _GR_UP_DPP_ENGINE_EVIDENCE
                if recovery
                else "benchmarks/results/2026-09-06-framework-qwen4exp-gr-wave-production.json"
            ),
        ),
        _selection(
            "router_logits",
            "qwen4exp_multirow_f32_router",
            _ROUTER_UD_LOGITS_SELECTED_VARIANT if recovery else _ROUTER_LOGITS_SELECTED_VARIANT,
            "f32_hidden_token_tile4_dense_exact",
            "f32",
            evidence=(
                _ROUTER_VARIANT_EVIDENCE[_ROUTER_UD_LOGITS_SELECTED_VARIANT]
                if recovery else _ROUTER_LOGITS_EVIDENCE
            ),
        ),
        _selection(
            "gr_gated_mean_sigmoid",
            "all_gr_reads_rows_le256",
            "strict",
            "strict",
            "f32",
            evidence=_GR_SIGMOID_MEAN_EVIDENCE,
        ),
        _selection(
            "moe_linear",
            "prefill_rows_ge2_layers27_47_gate_up",
            "selected_dual_wmma_prefill_compact_bf16_bf16_out",
            "selected_dual_grouped_rowbatch8_out4_expertgrid64_bf16_bf16_out",
            "gguf_q4_k",
            evidence=_MOE_WMMA_EVIDENCE,
        ),
        _selection(
            "moe_linear",
            "prefill_rows_ge2_layers27_47_down",
            "selected_grouped_wmma_prefill_compact_bf16_bf16_out",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "gguf_q5_1",
            evidence=_MOE_WMMA_EVIDENCE,
        ),
        _selection(
            "moe_linear",
            "prefill_rows_ge2_lt64_exact_grouped_q5_1_down",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_m1_bf16_bf16_out",
            "selected_grouped_prefill_compact_rowbatch8_out8_expertgrid64_bf16_bf16_out",
            "gguf_q5_1",
            evidence=_HALO_PF13_EVIDENCE,
        ),
        _selection(
            "linear",
            "grouped_prefill_rows_lt512_q8_0_expert_down",
            "selected_grouped_gemv_bf16_bf16_out",
            "selected_gemv_bf16_bf16_out",
            "gguf_q8_0",
            evidence=_HALO_PF13_EVIDENCE,
        ),
        _selection(
            "linear", "grouped_prefill_rows_ge512_q8_0_expert_down",
            "selected_grouped_row4_register_gemv_bf16_bf16_out", "selected_gemv_bf16_bf16_out",
            "gguf_q8_0",
            evidence="benchmarks/results/2026-09-07-framework-qwen4exp-q8-down-register-production.json",
        ),
        _selection(
            "linear",
            "prefill_policy_qwen4exp_raw_q8_shapes",
            "mmq128_raw_vec4_q8_1_d4x3_guarded_f32_f32_out",
            "coltile8_rowbatch4_f32_f32_out",
            "gguf_q8_0",
            evidence="benchmarks/results/2026-09-06-framework-qwen4exp-mmq-raw-vector-production.json",
        ),
        _selection(
            "linear",
            "prefill_gdn_qkv_ssm_prepacked_mmq",
            "mmq128_prepacked_vec4_q8_1_d4x3_guarded_f32_f32_out",
            "coltile8_rowbatch4_f32_f32_out",
            "gguf_q8_0",
            evidence="benchmarks/results/2026-09-06-framework-qwen4exp-mmq-vec4-production.json",
        ),
        _selection(
            "gdn_recurrence_norm_gate",
            "prefill_rows_ge2_layers27_47_gdn",
            # tiled16_multi supersedes the columnwarps owner on the 27-47 window
            # after the full layer2 production-envelope admission on the UD lane
            # (2026-10-08: all aggregates and scopes, prefill_to_c1 top-1 18/18,
            # eligible for automatic admission; engine A/B +0.68% prefill with
            # 12/12 cases positive). The columnwarps/wave_norm chain stays
            # registered as the runtime fallback and HIPENGINE_QWEN4_EXP_GDN_TILE16_PREFILL=0
            # is the opt-out.
            "qwen4exp_gdn_tiled16_multi_prefill",
            "qwen4exp_sigmoid_strict_prefill",
            "f32_state",
            evidence="benchmarks/results/2026-10-08-qwen4exp-gdn-multicolumn-admission/gate.json",
        ),
        _selection(
            "linear",
            "decode_c1_calibrated_q4_dp4a_43_layers",
            "selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out",
            "selected_dual_silu_logical128_t64_gemv_bf16_bf16_out",
            "gguf_q4_k",
            evidence=_DECODE_EVIDENCE,
        ),
    )
    if not recovery:
        return selections
    # Scope identifiers are shared with strict and remain stable. Select the
    # recovered owners while preserving the registered rollback contracts.
    inactive = {
        "qwen4exp_gdn_tiled16_dpp_prefill",
        "qwen4exp_gdn_tiled16_prefill",
        "qwen4exp_gdn_columnwarps_prefill",
        "mmq128_token64_q8_1_d4x3_guarded_f32_f32_out",
        "mmq128_raw_vec4_q8_1_d4x3_guarded_f32_f32_out",
        "mmq128_prepacked_vec4_q8_1_d4x3_guarded_f32_f32_out",
        "selected_dual_wmma_prefill_compact_bf16_bf16_out",
        "selected_grouped_wmma_prefill_compact_bf16_bf16_out",
    }
    replacements = {
        "selected_dual_q8_1_dp4a_silu_logical128_t64_gemv_bf16_bf16_out":
            "selected_dual_silu_logical128_t64_gemv_bf16_bf16_out",
    }
    for selection in selections:
        if selection.selected_variant not in inactive:
            continue
        if selection.layer == "gdn_recurrence_norm_gate":
            replacements[selection.selected_variant] = "qwen4exp_sigmoid_wave_norm_prefill"
        elif selection.selected_variant.startswith("mmq128_"):
            replacements[selection.selected_variant] = "coltile8_rowbatch4_wave_scale_f32_f32_out"
        else:
            replacements[selection.selected_variant] = selection.strict_fallback_variant
    restored_q8_scopes = {
        "grouped_prefill_rows_lt512_q8_0_expert_down",
        "grouped_prefill_rows_ge512_q8_0_expert_down",
    }
    restored_qsa = {
        "strict_h256_head_quad_rows_spans", "strict_ordered_three_pass_spans",
        "strict_ordered_three_pass_v2_spans",
    }
    gdn_variant = _selected_gdn_prefill_variant()
    selections = _recovery_gdn_scopes(
        selections, variant=gdn_variant,
        evidence=_GDN_NARROW_EVIDENCE if gdn_variant == _GDN_NARROW_VARIANT else None,
    )
    result = []
    for selection in selections:
        if selection.layer == "gdn_recurrence_norm_gate":
            result.append(selection)
            continue
        variant = replacements.get(selection.selected_variant, selection.selected_variant)
        restored = selection.selected_variant in restored_qsa
        if selection.scope in restored_q8_scopes:
            variant = "selected_grouped_blockscale_guarded_prefill_bf16_bf16_out"
            restored = True
        result.append(replace(
            selection,
            selected_variant=variant,
            evidence_artifact=(
                _GR_UP_DPP_ENGINE_EVIDENCE
                if (
                    selection.layer == "linear+gr_gated_mean"
                    and selection.scope == "qwen4exp_rows_gt256_gr_up"
                )
                else selection.evidence_artifact
                if selection.layer == "router_logits"
                else _Q8_QSA_RESTORATION_EVIDENCE
                if restored
                else _ARITHMETIC_RECOVERY_EVIDENCE
            ),
        ))
    return tuple(result) + (_dense_wide_tail_selection(production=True), _norm_f16_selection(production=True))


def last_prebinder_conflicts() -> dict[str, tuple[str, str]]:
    """Return env keys whose caller-set value the last bind discarded.

    Maps the variable to ``(value_before_bind, value_written_by_bind)``.  The
    binder owns every variable it writes, so a value set before generator
    construction is silently discarded: the caller's route never runs and the
    run reports a clean null.  Gates that flip a profile-owned variable must set
    it *after* the binder, and this getter is how they assert they did.
    """

    return dict(_last_prebinder_conflicts)


def _fast_selections() -> tuple[VariantSelection, ...]:
    """The in-envelope production plan with the fast operand-class overrides."""

    result = []
    replaced = set()
    for selection in _production_selections(recovery=True):
        if (
            selection.layer == "moe_linear"
            and selection.registry_quant == "gguf_q5_1"
            and selection.scope in _FAST_T16_DOWN_SCOPES
        ):
            replaced.add(selection.scope)
            selection = replace(
                selection,
                selected_variant=_FAST_T16_DOWN_VARIANT,
                evidence_artifact=_FAST_T16_DOWN_EVIDENCE,
            )
        result.append(selection)
    missing = set(_FAST_T16_DOWN_SCOPES) - replaced
    if missing:
        raise RuntimeError(
            f"Qwen4Exp fast plan lost routed down scopes: {sorted(missing)!r}"
        )
    return tuple(result)


def _apply_profile_env(values: dict[str, str]) -> None:
    """Write binder-owned env values, recording any caller value it discards."""

    for key, value in values.items():
        previous = os.environ.get(key)
        if (
            previous is not None
            and previous != value
            and _binder_written.get(key, object()) != previous
        ):
            _last_prebinder_conflicts[key] = (previous, value)
        os.environ[key] = value
        _binder_written[key] = value


def _report_prebinder_conflicts() -> None:
    """Warn, or raise, when a bind discarded a caller-set route variable."""

    if not _last_prebinder_conflicts:
        return
    detail = ", ".join(
        f"{key}={before!r} -> {after!r}"
        for key, (before, after) in sorted(_last_prebinder_conflicts.items())
    )
    message = (
        "Qwen4Exp profile bind discarded pre-binder values for "
        f"{len(_last_prebinder_conflicts)} profile-owned variable(s): {detail}. "
        "The binder owns these, so a value set before generator construction "
        "is inert and the intended route did not run. Set them after the "
        "binder instead."
    )
    if os.environ.get(PREBINDER_CONFLICT_ENV, "warn").strip().lower() == "error":
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _bind_default_chunk(generator: Any, *, production: bool) -> None:
    if getattr(generator,"_implicit_profile_chunk_size",False):
        # The qualified factory allocated the caller's chunk before profile
        # binding; the profile owns the production default.
        #
        # PRODUCTION CHUNK (2026-10-04): the generator's capacity guard picks
        # the largest chunk the device can afford at the requested context and
        # records it as `admitted_prefill_chunk_size` (ledger RT-1/RT-3). The
        # binder binds THAT value rather than a fixed constant, so a device that
        # cannot afford 4096 falls back to 2048 or 1024 instead of
        # under-reserving. Evidence for the 1024 -> 4096 step:
        # benchmarks/results/2026-10-04-qwen4exp-prefill-chunk-4096/ - +8.43%
        # prefill and +1.68% decode at code-p4096, bit-identical token ids
        # across strict and production at both chunk sizes, and the layer2 gate
        # at 4096 reproducing the 1024 gate numbers digit for digit.
        admitted = getattr(generator,"admitted_prefill_chunk_size",None)
        if admitted:
            generator.runner.prefill_chunk_size = int(admitted)
        else:
            generator.runner.prefill_chunk_size = 1024 if production else 512


def _bind_fast_routes(*, fast: bool) -> None:
    """Publish the fast profile's route selectors, or retract this module's own.

    Strict and production leave a caller-set selector alone (the A/B and gate
    harnesses set these directly); they only clear a value an earlier fast
    bind in this process wrote.
    """

    for key in (_T16_DOWN_ENV,):
        if fast:
            _apply_profile_env({key: "1"})
            _fast_published.add(key)
        elif key in _fast_published:
            _fast_published.discard(key)
            _apply_profile_env({key: "0"})


def _bind(
    generator: Any,
    resolved: ResolvedRuntimeProfile,
    *,
    production: bool,
    fast: bool = False,
) -> None:
    _last_prebinder_conflicts.clear()
    _bind_default_chunk(generator,production=production)
    table = getattr(getattr(generator, "_resident", None), "ple_table", None)
    if table is not None:
        # Sparse PLE mmap reads must not trigger sequential readahead. The
        # mapping owns the hint across remaps; explicit full-table warming
        # temporarily restores sequential access.
        qualified = production and resolved.manifest.get("quant") == "gguf_ud_q4_k_xl"
        mode = os.environ.get(
            "HIPENGINE_QWEN4_EXP_PLE_MAPPING_ACCESS",
            "random" if qualified else "normal",
        )
        table.configure_mapping_access(mode)
    _apply_profile_env({PRODUCTION_MOE_PREFILL_ENV: "1" if production else "0"})
    # Preserve the existing Q4_K_M plan and freeze neighboring experiments.
    # UD-Q4_K_XL overrides its unqualified arithmetic below, before allocating
    # profile resources; the exact optimized owners remain enabled.
    # The WMMA route covers MoE layers 27-47 via the backend capability
    # constant; the ds4-MMQ envs stay off so they cannot preempt it, and the
    # exact-grouped guards (`not production_grouped_moe`) keep layers 0-26 on
    # their separately manifested exact grouped owners.
    profile_env: dict[str, str] = {}
    for key, value in {
        "HIPENGINE_GGUF_WMMA_PREFILL": "0",
        # Publish the router producer the manifest declares, so the runner's
        # launch-time switch cannot disagree with the resolved selection.
        _ROUTER_VARIANT_ENV: next(
            row["selected_variant"] for row in resolved.manifest["selections"]
            if row["layer"] == "router_logits"
        ),
        "HIPENGINE_QWEN4_EXP_GROUPED_MOE_PREFILL": "0",
        "HIPENGINE_QWEN4_EXP_Q5_1_WMMA": "0",
        "HIPENGINE_QWEN4_EXP_Q8_0_GROUPED_WMMA": "0",
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE": "0",
        # PF-3 Q5_1 M1 and PF-1 grouped Q8_0 down are T0 exact production
        # owners after the one-process/one-residency canonical gate. Strict
        # keeps the preceding owners as registered fallbacks.
        PROFILE_Q5_1_DOWN_M1_ENV: "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_FORKB_GROUPED_DOWN": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q8_0_SELECTED_WMMA_DOWN": (
            "1" if production else "0"
        ),
        "HIPENGINE_QWEN4_EXP_Q8_DOWN_VARIANT": "selected_grouped_wmma_prefill_bf16_bf16_out",
        "HIPENGINE_QWEN4_EXP_GROUPED_ROW4_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q4_BUNDLE_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q4_PAIR_PREFILL": "1" if production else "0",
        # Exact iu8-risk+repair gate/up: bit-identical to the pair2 owner
        # (state gate + 72 exact trajectories); production-only binding.
        "HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT": "1" if production else "0",
        # Weight-exact iu8-WMMA Q5_1 down with risk+repair: bit-identical to
        # the pair2 row-publish parent in practice at multiplier 16 (state
        # gate + 72 exact trajectories); production-only binding.
        "HIPENGINE_QWEN4_EXP_Q51_IU8_EXACT": "1" if production else "0",
        # Exact iu8-risk+repair Q5_K gate/up (layer 2, the sole Q5_K MoE
        # layer): bit-identical to the strict row4 parent (kernel screen
        # 5.67x/6.00x/6.68x at rows 512/1024/4096, repair share 5.4%; full
        # 12-case same-residency A/B +4.3% prefill geo-mean with every case
        # winning and identical output digests); production-only binding.
        "HIPENGINE_QWEN4_EXP_Q5_K_IU8_EXACT": "1" if production else "0",
        # iu8-WMMA GR up (T1, three-plane fp32 staging): teacher-forced
        # logits KL <= 7.4e-8 versus the incumbent with 100% top-1 and
        # 10-100x argmax margins, bit-deterministic; full-suite A/B
        # +9.2% prefill geo-mean with all 12 cases winning; production-only
        # binding, exact fused parent stays the strict fallback.
        "HIPENGINE_QWEN4_EXP_GR_IU8": "1" if production else "0",
        # iu8-WMMA GR down (10240->320, T1, same chain): KL <= 5.5e-8 vs
        # incumbent, top-1 100%, deterministic; full-suite A/B +2.0%
        # prefill geo-mean on top of the promoted up leg; production-only
        # binding, exact coltile parent stays the fallback.
        "HIPENGINE_QWEN4_EXP_GR_IU8_DOWN": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q8_WAVE_SCALE": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_GR_WAVE_SCALE": "1" if production else "0",
        # Exact ordered DPP replacement for the wave-scale GR-up owner. The
        # registered wave-scale owner remains the strict fallback through the
        # runtime resolver, while the profile's declared strict fallback is
        # the unscaled fused parent above.
        "HIPENGINE_QWEN4_EXP_GR_DPP_REDUCTION": (
            "1"
            if production and resolved.manifest.get("quant") == "gguf_ud_q4_k_xl"
            else "0"
        ),
        # K320/full-row GR-up DPP sibling: promoted production default for the
        # qualified lane (2026-10-08). Exact-class (bit-identical outputs); the
        # complete qualification chain is natural-input owner (576 captures,
        # 1.093-1.106x), full model-state gate (12 cases, 3,456 calls, full
        # logits/state/KV exact) and complete-engine A/B (prefill 1.0111x,
        # client wall 0.9975x, cross-mode exact 12/12). The selector admits only
        # rows>256 in complete groups of 4 with four branches and low rank 320;
        # every other shape keeps the generic DPP owner, which remains the
        # registered strict fallback. HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS=0
        # is the bisection opt-out.
        "HIPENGINE_QWEN4_EXP_GR_K320_FULL_ROWS": (
            "1"
            if production and resolved.manifest.get("quant") == "gguf_ud_q4_k_xl"
            else "0"
        ),
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_VEC4": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_RAW_VECTOR": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q8_DOWN_ROW4_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q8_DOWN_BUNDLE_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q8_MAPPED_DOWN": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q8_DOWN_REGISTER": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q51_FOLD128_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q51_FOLD_PAIR_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q51_REGISTER_CACHE": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q51_ROW_PUBLISH": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_GDN_WAVE_NORM": "1" if production else "0",
        # Plan registration captures explicit parent rollback. The binder
        # publishes that same selection; strict and Q4_K_M stay wide-index.
        _GDN_PREFILL_VARIANT_ENV: (
            next(
                (row["selected_variant"] for row in resolved.manifest["selections"]
                 if row["layer"] == "gdn_recurrence_norm_gate"
                 and row["scope"] == _GDN_NARROW_SCOPE),
                _GDN_WIDE_VARIANT,
            ) if production else _GDN_WIDE_VARIANT
        ),
        "HIPENGINE_QWEN4_EXP_MMQ_TOKEN64": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL": "page256" if production else "0",
        "HIPENGINE_QWEN4_EXP_QSA_HEAD_PAIR": "quad" if production else "0",
        # ds4-MMQ MoE suffixes are superseded by the certified WMMA-MoE27
        # routing on layers 27-47.
        "HIPENGINE_QWEN4_EXP_Q5_1_MMQ_PREFILL": "0",
        "HIPENGINE_QWEN4_EXP_Q5_1_MMQ_LAYERS": "",
        "HIPENGINE_QWEN4_EXP_Q4_K_MMQ_PREFILL": "0",
        "HIPENGINE_QWEN4_EXP_Q4_K_MMQ_LAYERS": "",
        "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL_LAYERS": (
            ",".join(map(str, PRODUCTION_GDN_PEER_PREFILL_LAYERS))
            if production
            else ""
        ),
        "HIPENGINE_QWEN4_EXP_Q4_DP4A64": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q4_DP4A64_LAYERS": (
            ",".join(map(str, PRODUCTION_Q4_DP4A_DECODE_LAYERS))
            if production
            else ""
        ),
        # Certified 2026-09-17 at layers 16-47 and measured 5.4% below the f16
        # WMMA route at 1K/4K, with that route's own envelope (bit-identical
        # strict and candidate logits digests). The wide route replaces it here
        # rather than sitting beside it: both families selectable for the same
        # layers is the contention the 2026-09-17 route fix had to resolve. The
        # exact coltile chain stays the registered strict fallback and keeps
        # every layer below 16.
        "HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS": "",
        "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE": (
            "1"
            if production and resolved.manifest.get("quant") == "gguf_ud_q4_k_xl"
            else "0"
        ),
        "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS": (
            ",".join(map(str, PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS))
            if production and resolved.manifest.get("quant") == "gguf_ud_q4_k_xl"
            else ""
        ),
        # Publish the same bounded tail leaf declared in the UD manifest.
        # Strict and Q4_K_M cannot inherit a preceding UD binder's selection.
        DENSE_WIDE_TAIL_VARIANT_ENV: (
            next(row["selected_variant"] for row in resolved.manifest["selections"]
                 if row["scope"] == _DENSE_WIDE_TAIL_SCOPE)
            if production and resolved.manifest.get("quant") == "gguf_ud_q4_k_xl"
            else _DENSE_WIDE_TAIL_PARENT
        ),
        # Publish the manifest's bounded producer selection. Other quants and
        # strict cannot inherit a preceding UD binder's publisher or opt-out.
        NORM_F16_PUBLICATION_ENV: (
            "1" if any(
                row["scope"] == _NORM_F16_SCOPE
                and row["selected_variant"] == "f32_f16_publish"
                for row in resolved.manifest["selections"]
            ) else "0"
        ),
        "HIPENGINE_GGUF_Q8_0_WMMA_TILE_M": "64",
        "HIPENGINE_GGUF_Q8_0_WMMA_TILE_N": "32",
        "HIPENGINE_QWEN4_EXP_Q4_TILE_M": "16",
        "HIPENGINE_QWEN4_EXP_Q4_TILE_N": "16",
        # Certified iu8-WMMA gate/up suffix within the WMMA-MoE27 route.
        "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS": (
            ",".join(map(str, PRODUCTION_Q4_IU8_PREFILL_LAYERS))
            if production
            else ""
        ),
        # `HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT` is bound to the gated
        # alternative, 1.0.
        #
        # Tightening the Kahan risk bound to 1.0 - the top of the range the
        # multiplier's own screening calls safe ("between 0.5 and 1 (below it,
        # BF16 flips escape the repair)") - cuts the repaired share of rows
        # **0.7119% -> 0.2092%** (3.40x fewer, over 1504 calls and 9.86e9
        # outputs, with zero over-capacity calls) and passes the production
        # numerical gate with 3.3-6.4x aggregate margin: mean KL 3.039e-4
        # against 1e-3, p99 4.271e-3 against 2e-2, max 7.771e-3 against 5e-2,
        # top-1 0.99556, `scope_failures: null`, determinism and isolation clean,
        # including a 274x improvement on the `prefill_last` scope that rejected
        # the repair-free sibling.
        #
        # This binding previously sat at 4.0, on the recorded grounds that "the
        # wall benefit does not survive the canonical protocol" and that "the
        # repair kernel launches over `risk_capacity` regardless of
        # `risk_count`, so the real lever is the pass's launch geometry, not its
        # threshold".  BOTH GROUNDS ARE REFUTED by measurement.
        #
        # The binder writes this key unconditionally, so the canonical harness -
        # which reads this profile - could never move it: every canonical reading
        # of "flat" compared 4.0 against 4.0.  Applied through the post-binder
        # override, the queue moves 3.403x (70,167,220 -> 20,621,412 slots) and
        # the repair kernel's own traced time falls 4808.346 -> 3386.063 ms, a
        # 1318.0 ms window reduction on `code-p4096` (64981.3 -> 63663.3 ms of
        # kernel time), i.e. 2.03%.  The kernel does not launch over
        # `risk_capacity`: it launches 1024 blocks with a grid-stride loop bounded
        # by `min(risk_count, max_risks)`.
        #
        # There is a real fixed component - fitting time = F + k*queue over the
        # two arms gives F = 2794.1 ms and k = 2.871e-5 ms/slot, so 58.1% of the
        # repair at 4.0 is independent of the queue.  That is a genuine
        # launch-geometry question worth attacking on its own, but it does not
        # make the threshold free of value: the proportional part is what the
        # 1.0 binding collects, and it is collected now.
        #
        # Task gate: `requires_review` with 15/18 strict-exact and three
        # divergences, all cosmetic - an extra `Deque,` in an import list and two
        # reworded clauses that are semantically equivalent.  The candidate is
        # repeat-deterministic with no repeat mismatches.  Per AGENTS.md,
        # cross-width generated-ID equality is diagnostic rather than binding;
        # the numerical envelope above is the binding gate and it passes.
        #
        # Rollback: set this to "4.0".  The strict fallback is registered and
        # unchanged.
        #
        # Evidence: benchmarks/results/2026-10-04-qwen4exp-q4-iu8-risk-mult-1/
        # and benchmarks/results/2026-10-04-qwen4exp-risk-mult-override-ab/;
        # worklog entries 20261003T225747, 20261003T234213,
        # 20261004T000542 (the binder pin) and 20261004T001404 (this promotion).
        "HIPENGINE_QWEN4_EXP_Q4_IU8_RISK_MULT": "1.0",
        # Certified column-warp GDN prefill suffix (supersedes peer-GDN).
        # Decode retains its separately tuned rows==1 owner.
        "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_PREFILL": "1" if production else "0",
        # The tile16 variant is consulted only inside the colwarps gate above, so
        # it must be bound exactly where that gate is open. Since the tiled16_multi
        # admission passed the full layer2 production envelope on the UD lane
        # (2026-10-08, all aggregates and scopes, prefill_to_c1 top-1 18/18,
        # eligible for automatic admission; wave_norm remains the registered
        # strict fallback and HIPENGINE_QWEN4_EXP_GDN_TILE16_PREFILL=0 is the
        # opt-out), GDN_COLWARPS_PREFILL is no longer recovery-zeroed for UD and
        # the multi variant is named for every production quant.
        "HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT": (
            "qwen4exp_gdn_tiled16_multi_prefill"
            if production
            else ""
        ),
        "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS": (
            ",".join(map(str, PRODUCTION_GDN_COLWARPS_PREFILL_LAYERS))
            if production
            else ""
        ),
        "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_DECODE_LAYERS": "",
        # Certified QSA flash prefill suffix.
        "HIPENGINE_QWEN4_EXP_QSA_FLASH_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_QSA_FLASH_LAYERS": (
            ",".join(map(str, PRODUCTION_QSA_FLASH_PREFILL_LAYERS))
            if production
            else ""
        ),
        "HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_QSA_ORDERED_DECODE_V2": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_DOWN": "1",
        "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4": "1",
        "HIPENGINE_QWEN4_EXP_EXACT_GROUPED_Q4_ALL": "1",
        "HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256": resolved.manifest_sha256,
    }.items():
        profile_env[key] = value
    _apply_profile_env(profile_env)
    if production and resolved.manifest.get("quant") == "gguf_ud_q4_k_xl":
        _apply_profile_env(
            {
                "HIPENGINE_QWEN4_EXP_" + flag: "0"
                for flag in PRODUCTION_ARITHMETIC_RECOVERY_FLAGS
            }
        )
        _apply_profile_env(
            {
                "HIPENGINE_QWEN4_EXP_" + flag: value
                for flag, value in PRODUCTION_Q8_QSA_RESTORED_FLAGS.items()
            }
        )
    _bind_fast_routes(fast=fast)
    _report_prebinder_conflicts()
    if production:
        configure = getattr(
            getattr(generator, "runner", None),
            "configure_mmq_prefill_resources",
            None,
        )
        if callable(configure):
            configure()


def _strict_binder(generator: Any, resolved: ResolvedRuntimeProfile) -> None:
    _bind(generator, resolved, production=False)


def _production_binder(generator: Any, resolved: ResolvedRuntimeProfile) -> None:
    _bind(generator, resolved, production=True)


def _fast_binder(generator: Any, resolved: ResolvedRuntimeProfile) -> None:
    # Tiles first: a route selector published without its tiles would fall
    # through to the production chain and misreport what ran.
    populate = getattr(generator, "populate_t16_resident_stages", None)
    if callable(populate):
        populate(_FAST_T16_RESIDENT_SLOTS)
    _bind(generator, resolved, production=True, fast=True)


def _wanted_keys() -> set[RuntimeProfileKey]:
    return {
        _key(quant, profile)
        for quant in QWEN4_EXP_QUANTS
        for profile in (ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION)
    } | {_key(quant, ExecutionProfile.FAST) for quant in FAST_QUANTS}


def _key(quant: str, profile: ExecutionProfile) -> RuntimeProfileKey:
    return RuntimeProfileKey(QWEN4_EXP_MODEL, QWEN4_EXP_BACKEND, quant, profile)


def register_qwen4_exp_gfx1151_profiles() -> bool:
    wanted = _wanted_keys()
    existing = set(registered_runtime_profile_keys())
    if wanted <= existing:
        return False
    if existing & wanted:
        raise RuntimeError("Qwen4Exp gfx1151 profile registry is partially populated")
    for quant in QWEN4_EXP_QUANTS:
        register_runtime_profile_plan(
            model=QWEN4_EXP_MODEL,
            backend=QWEN4_EXP_BACKEND,
            quant=quant,
            profile=ExecutionProfile.STRICT,
            plan=RuntimeProfilePlan(
                selections=_strict_selections(
                    recovery=quant == "gguf_ud_q4_k_xl",
                ),
                kv_policy="paged_bf16_qsa_index_f32",
                graph_policy="request_owned_exact_moe_graph_c1",
                binder=_strict_binder,
            ),
        )
        register_runtime_profile_plan(
            model=QWEN4_EXP_MODEL,
            backend=QWEN4_EXP_BACKEND,
            quant=quant,
            profile=ExecutionProfile.PRODUCTION,
            plan=RuntimeProfilePlan(
                selections=_production_selections(
                    recovery=quant == "gguf_ud_q4_k_xl",
                ),
                kv_policy="paged_bf16_qsa_index_f32",
                graph_policy="request_owned_exact_moe_graph_c1",
                binder=_production_binder,
            ),
        )
    for quant in FAST_QUANTS:
        register_runtime_profile_plan(
            model=QWEN4_EXP_MODEL,
            backend=QWEN4_EXP_BACKEND,
            quant=quant,
            profile=ExecutionProfile.FAST,
            plan=RuntimeProfilePlan(
                selections=_fast_selections(),
                kv_policy="paged_bf16_qsa_index_f32",
                graph_policy="request_owned_exact_moe_graph_c1",
                binder=_fast_binder,
            ),
        )
    return True


def qwen4_exp_gfx1151_profiles_registered() -> bool:
    wanted = _wanted_keys()
    return wanted <= set(registered_runtime_profile_keys())


__all__ = [
    "FAST_QUANTS",
    "PRODUCTION_GDN_PEER_PREFILL_LAYERS",
    "PRODUCTION_MOE_PREFILL_ENV",
    "PROFILE_Q5_1_DOWN_M1_ENV",
    "PRODUCTION_Q4_DP4A_DECODE_LAYERS",
    "PRODUCTION_Q4_K_MMQ_PREFILL_LAYERS",
    "PRODUCTION_Q5_1_MMQ_PREFILL_LAYERS",
    "QWEN4_EXP_BACKEND",
    "QWEN4_EXP_MODEL",
    "QWEN4_EXP_QUANTS",
    "qwen4_exp_gfx1151_profiles_registered",
    "register_qwen4_exp_gfx1151_profiles",
]
