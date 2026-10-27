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
    resolve_runtime_profile,
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


# Quants whose production plan runs the F16-operand route composition: F16
# operands with F32 accumulation and no exact repair, layered on the
# arithmetic-recovery plan. The complete composition passes the calibrated
# production envelope against strict, and every route keeps its strict
# fallback.
F16_ROUTE_QUANTS = ("gguf_ud_q4_k_xl",)
# The opt-in fast profile is where a reduced-precision route lands before its
# production gate. It carries no route beyond the production plan at present.
FAST_QUANTS = F16_ROUTE_QUANTS
_ROUTED_F16_ENV = "HIPENGINE_QWEN4_EXP_ROUTED_F16"
_ROUTED_F16_GATE_UP = "selected_routed_f16_dual_silu_wmma_prefill_bf16_bf16_out"
_ROUTED_F16_LINEAR = "selected_routed_f16_wmma_prefill_bf16_bf16_out"
_QSA_WMMA_VARIANT = "wmma_f16_masked_prefill_f32"
# Routed-expert scopes the plan moves to the wide-tile F16 GEMMs that
# read the raw expert weights: (layer, scope, registry quant) -> variant.
# Layers whose gate/up is Q5_K route from 64 rows; the rest from 2.
_F16_ROUTE_SCOPES = {
    ("moe_linear", "prefill_rows_ge2_lt64_exact_grouped_q4_gate_up", "gguf_q4_k"):
        _ROUTED_F16_GATE_UP,
    ("moe_linear", "prefill_rows_ge64_exact_grouped_q4_gate_up", "gguf_q4_k"):
        _ROUTED_F16_GATE_UP,
    ("linear", "ungrouped_prefill_rows_ge64_q5k_gate_up", "gguf_q5_k"):
        _ROUTED_F16_GATE_UP,
    ("moe_linear", "prefill_rows_ge2_lt64_exact_grouped_q5_1_down", "gguf_q5_1"):
        _ROUTED_F16_LINEAR,
    ("moe_linear", "prefill_rows_ge64_lt512_exact_grouped_q5_1_down", "gguf_q5_1"):
        _ROUTED_F16_LINEAR,
    ("moe_linear", "prefill_rows_ge512_k640_exact_grouped_q5_1_down", "gguf_q5_1"):
        _ROUTED_F16_LINEAR,
    ("linear", "grouped_prefill_rows_lt512_q8_0_expert_down", "gguf_q8_0"):
        _ROUTED_F16_LINEAR,
    ("linear", "grouped_prefill_rows_ge512_q8_0_expert_down", "gguf_q8_0"):
        _ROUTED_F16_LINEAR,
    # QSA prefill attention, masked sparse window and dense causal window, on
    # the WMMA kernel over F16 K/V rows gathered from the paged BF16 cache.
    ("qsa_sparse_attention", "prefill_h256_page256_sparse_rows_ge16", "bf16_kv"):
        _QSA_WMMA_VARIANT,
    ("paged_attn_decode", "qwen4exp_multirow_dense_qsa", "w4_paro"):
        _QSA_WMMA_VARIANT,
}
# Binder-owned route windows of the composition: the F16 dense wide route runs
# at every layer (13-47 on a quant without the composition), which also
# carries the hyper-connection down projection.
F16_ROUTE_Q8_DENSE_WIDE_PREFILL_LAYERS = tuple(range(48))
_F16_ROUTE_BINDER_ENV = {
    "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS": ",".join(
        map(str, F16_ROUTE_Q8_DENSE_WIDE_PREFILL_LAYERS)
    ),
    # The tiled GDN recurrence at every GDN layer (27-47, with the
    # register-state recurrence below, on a quant without the composition).
    "HIPENGINE_QWEN4_EXP_GDN_COLWARPS_LAYERS": ",".join(map(str, range(48))),
    # The row-split recurrence in place of the multi-column tile-16 body: the
    # same recurrence with the scales applied to the two dot products, a
    # different FP32 summation order, 128-thread blocks.
    "HIPENGINE_QWEN4_EXP_GDN_TILE16_VARIANT": "qwen4exp_gdn_rowsplit_prefill",
}
# Profile-owned composition selectors: profiles without the composition disable
# them even when a caller set them before binding, keeping dispatch and manifest aligned.
_F16_ROUTE_SELECTORS = {
    _ROUTED_F16_ENV: "1",
    # Hyper-connection up projection on the F16 wide route.
    "HIPENGINE_QWEN4_EXP_GR_WIDE": "1",
    # Hyper-connection read projections launched directly on F16 activations:
    # bit-identical to the dispatch they replace under the windows above.
    "HIPENGINE_QWEN4_EXP_GR_F16_DIRECT": "1",
    # Decode: the row-split GDN step and the warp-per-expert routed-expert
    # owners (FP32 or integer operands as before; reduction order differs).
    "HIPENGINE_QWEN4_EXP_GDN_DECODE_ROWSPLIT": "1",
    "HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_GATE_UP": "1",
    "HIPENGINE_QWEN4_EXP_MOE_DECODE_WARP_DOWN": "1",
    # Multi-row FP32 projections over uncompressed weights (the indexer
    # projections) on the small-matrix F16 WMMA GEMM.
    "HIPENGINE_QWEN4_EXP_DENSE_F16_GEMM": "1",
    # QSA prefill attention on the WMMA kernel.
    "HIPENGINE_QWEN4_EXP_QSA_WMMA": "1",
}
_F16_ROUTE_EVIDENCE = (
    "benchmarks/results/2026-10-14-qwen4exp-fast-composition-gate-2/README.md"
)
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
PRODUCTION_Q4_DP4A_DECODE_LAYERS = (
    0, 2, 5, 6, 8, 9, 10, 11
) + tuple(range(13, 48))
PRODUCTION_ARITHMETIC_RECOVERY_FLAGS = (
    "PRODUCTION_MOE_PREFILL", "Q4_IU8_PREFILL", "Q8_0_SELECTED_WMMA_DOWN",
    "Q8_MMQ_PREFILL", "GR_IU8", "GR_IU8_DOWN",
    "Q4_DP4A64",
    "QSA_H256_WAVE_PREFILL", "QSA_FLASH_PREFILL",
    "QSA_ORDERED_DECODE", "GR_WIDE",
)
# The QSA prefill head grouping and the ordered-v2 sparse decode leaf are
# request-local runner state published from the plan's own selection below, so
# they are not environment flags the arithmetic-recovery or restoration passes
# can rewrite.
_ARITHMETIC_RECOVERY_EVIDENCE = (
    "benchmarks/results/2026-09-14-q8-prefill-numerics/artifact.json"
)
_Q4_DECODE_RECOVERY_EVIDENCE = "benchmarks/results/20261022-qwen38fn-q4-decode-repair.json"
# Do not restore Q4_DP4A64: the October 20 long-context capture isolates a
# decode error to that operation. Keep the strict variant already named in
# the recovery manifest. The clearing gate is recorded in docs/REFACTOR.md.
PRODUCTION_Q8_QSA_RESTORED_FLAGS = {
    "Q8_0_SELECTED_WMMA_DOWN": "1",
    "Q8_DOWN_VARIANT": "selected_grouped_blockscale_guarded_prefill_bf16_bf16_out",
    "QSA_H256_WAVE_PREFILL": "page256",
    "QSA_ORDERED_DECODE": "1",
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
# Named router producer and rollback switch. Strict/Q4_K_M keep tile4; the UD
# production plan selects the F16-operand GEMM below, and ordered DPP is the
# arithmetic-recovery default and the named rollback.
_ROUTER_VARIANT_ENV = "HIPENGINE_QWEN4_EXP_ROUTER_VARIANT"
_ROUTER_TILE4_VARIANT = "f32_hidden_token_tile4_dense_exact"
_ROUTER_WARP_REDUCE_VARIANT = "f32_hidden_warp_reduce_dense_exact"
_ROUTER_DPP_VARIANT = "f32_hidden_token_tile4_dpp_exact"
# One-barrier ordered DPP passed the natural-input/full-state canonical gate
# and improves prefill in the same-residency twelve-case model-wall campaign.
# Keep the exact tile4 parent as the strict fallback and temporary rollback.
# The F16-operand WMMA GEMM is the UD production plan's router producer; see
# _f16_route_selections for the replacement it performs.
_ROUTER_F16_GEMM_VARIANT = "f32_hidden_f16_wmma_gemm"
_ROUTER_VARIANTS = (
    _ROUTER_TILE4_VARIANT,
    _ROUTER_WARP_REDUCE_VARIANT,
    _ROUTER_DPP_VARIANT,
    _ROUTER_F16_GEMM_VARIANT,
)


def _selected_router_producer_variant(*, recovery: bool = False) -> str:
    """Return the named router producer, or the plan default when none is named.

    The default is ordered DPP for the arithmetic-recovery plan and the tile4
    parent otherwise, so strict and Q4_K_M production keep tile4. The UD
    production plan then replaces the DPP default with the F16-operand GEMM
    unless a producer was named before import. The binder publishes the
    resolved manifest selection for the runner's launch-time reader, including
    the strict tile4 fallback and explicit rollback.
    """

    default = _ROUTER_DPP_VARIANT if recovery else _ROUTER_TILE4_VARIANT
    requested = os.environ.get(_ROUTER_VARIANT_ENV, "").strip() or default
    if requested not in _ROUTER_VARIANTS:
        raise ValueError(f"{_ROUTER_VARIANT_ENV} must be one of {_ROUTER_VARIANTS}; got {requested!r}")
    return requested


_GDN_WIDE_VARIANT = "qwen4exp_sigmoid_wave_norm_prefill"
_GDN_NARROW_VARIANT = "qwen4exp_sigmoid_wave_norm_narrow_prefill"
_GDN_NARROW_SCOPE = "prefill_rows_2_4096_serial_gdn_hk16_hv48_d128"
_GDN_NARROW_EVIDENCE = (
    "benchmarks/results/2026-10-06-qwen4exp-gdn-narrow-index/README.md"
)


def _selected_gdn_prefill_variant(requested: str | None = None) -> str:
    """Resolve the UD production plan's serial owner, including an explicit rollback.

    ``requested`` is a named, scoped plan-construction argument: the plan is
    registered with it and the binder republishes the resulting manifest row as
    request-local runner state. No process-wide switch selects this leaf.
    """

    value = str(requested or "").strip() or _GDN_NARROW_VARIANT
    if value not in (_GDN_WIDE_VARIANT, _GDN_NARROW_VARIANT):
        raise ValueError(
            "gdn_prefill_variant must name the wide or narrow wave-norm "
            f"variant; got {value!r}"
        )
    return value


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
    _ROUTER_F16_GEMM_VARIANT: _F16_ROUTE_EVIDENCE,
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


def _production_selections(
    *, recovery: bool = False, gdn_prefill_variant: str | None = None,
) -> tuple[VariantSelection, ...]:
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
            # registered as the runtime fallback, and a diagnostic arm pins the
            # columnwarp parent on the runner it measures
            # (``gdn_tile16_prefill(runner, False)``); no process switch carries
            # the arm. The 2026-10-15 five-pair same-thermal 12-case
            # leave-one-out found no non-regressive win for removing the owner.
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
    gdn_variant = _selected_gdn_prefill_variant(gdn_prefill_variant)
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
                _Q4_DECODE_RECOVERY_EVIDENCE
                if selection.scope == "decode_c1_calibrated_q4_dp4a_43_layers"
                else _GR_UP_DPP_ENGINE_EVIDENCE
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


def _f16_route_selections(
    *, gdn_prefill_variant: str | None = None,
) -> tuple[VariantSelection, ...]:
    """The arithmetic-recovery plan with the F16-operand route overrides."""

    result = []
    replaced = set()
    # A router producer the caller named before import is an explicit
    # rollback and is kept; the binder's own published value is not.
    requested = os.environ.get(_ROUTER_VARIANT_ENV, "").strip()
    router_named = bool(requested) and _binder_written.get(_ROUTER_VARIANT_ENV) != requested
    for selection in _production_selections(
        recovery=True, gdn_prefill_variant=gdn_prefill_variant,
    ):
        key = (selection.layer, selection.scope, selection.registry_quant)
        variant = _F16_ROUTE_SCOPES.get(key)
        if (
            selection.layer == "router_logits"
            and selection.selected_variant == _ROUTER_DPP_VARIANT
            and not router_named
        ):
            # The F16-operand WMMA GEMM producer: about 295 ms per 4096-token
            # prefill on the fastest FP32 producer against about 15 ms.
            selection = replace(
                selection,
                selected_variant=_ROUTER_F16_GEMM_VARIANT,
                evidence_artifact=_F16_ROUTE_EVIDENCE,
            )
        if variant is not None:
            replaced.add(key)
            selection = replace(
                selection,
                selected_variant=variant,
                evidence_artifact=_F16_ROUTE_EVIDENCE,
            )
        result.append(selection)
    missing = set(_F16_ROUTE_SCOPES) - replaced
    if missing:
        raise RuntimeError(
            f"Qwen4Exp F16 route plan lost scopes: {sorted(missing)!r}"
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


def _bind_f16_routes(*, enabled: bool) -> None:
    """Bind composition selectors, including strict disablement.

    A/B harnesses must set experimental overrides after binding, as for other
    profile-owned routes. Pre-binder conflicts use the normal warning/error policy.
    """

    _apply_profile_env({
        key: value if enabled else "0"
        for key, value in _F16_ROUTE_SELECTORS.items()
    })


def _bind(
    generator: Any,
    resolved: ResolvedRuntimeProfile,
    *,
    production: bool,
) -> None:
    _last_prebinder_conflicts.clear()
    # The prefill chunk is not a profile decision: the generator's constructor
    # sizes the PLE staging ring, the residency plan and the resident session
    # from one value (the explicit argument, or the capacity guard's admitted
    # rung when the caller left it unset) and hands that same value to the
    # runner. Nothing here restates it, so a caller-supplied runner keeps its
    # own chunk.
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
    # DP-2 loans one F16 staging plane to two immediate consumers of the wide
    # route. It selects no variant, so the manifest needs no new row: the wide
    # route and its F16-input sibling are already dispatch-time owners of the
    # lane this profile binds, and only the cast ownership between two launches
    # changes. The decision therefore follows that lane - UD production, no
    # strict fallback - while the gfx1151 `GGUF_DENSE_WIDE_F16_PAIR_POLICIES`
    # entry keeps owning which rows, layers and ordered pairs may use it.
    # Strict, fast, batch-invariant and Q4_K_M publish False and keep the parent.
    # It is request-local state on the runner, never a process-wide switch, and
    # every bind publishes its own value so a later one cannot inherit a lane.
    dp2_pair_enabled = (
        production
        and resolved.profile is ExecutionProfile.PRODUCTION
        and not resolved.fell_back_to_strict
        and resolved.manifest.get("quant") == "gguf_ud_q4_k_xl"
    )
    # The token64 dense leg follows the Q8 raw-MMQ prefill lane the profile
    # already binds: production (and the fast route that shares this binder)
    # may remap the raw-vec4 parent, strict and batch-invariant may not. It is
    # request-local state on the owning runner, never a process-wide switch.
    dense_mmq_token64_enabled = bool(production)
    # The two Q8 MMQ vector children ride the same lane: production (and the
    # fast route that shares this binder) selects them, strict and
    # batch-invariant keep the scalar parents. Request-local on the owning
    # runner, never a process-wide switch. Publication is deliberately
    # independent of HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL: the UD recovery zeroes
    # that parent after this table is published, and the supported diagnostic -
    # set the parent back to 1 after the bind, then
    # configure_mmq_prefill_resources() - must still get the vectorized children
    # the production plan selected. The dispatch-time selectors keep the
    # rows>=64, parent-variant and registration guards, the prepacked child stays
    # reachable only where a prepared sidecar is published, and the token64 leg
    # stays behind the raw-vec4 parent it builds on.
    dense_mmq_raw_vector_enabled = bool(production)
    dense_mmq_vec4_enabled = bool(production)
    # H256 prefill head grouping for the scalar sparse-prefill rows owner, as
    # request-local state on the owning runner rather than a process-wide
    # switch. Same condition the retired switch published: the production
    # binder's lanes group four heads, and every plan that falls back to the
    # strict binder - Q4_K_M, strict and batch-invariant - keeps the parent.
    # The dispatch-time selector still guards the leaf on the page256 wave
    # variant and Hq24/Hkv2. Note this is the fallback selection, not the owner
    # that runs: the production profiles also bind
    # HIPENGINE_QWEN4_EXP_QSA_WMMA=1, and the mixer launches the head-grouped
    # rows owner only when WMMA attention does not own the chunk. The resolved
    # manifest names that WMMA owner for the same scope, which is why this value
    # follows the binder rather than the manifest row.
    qsa_prefill_head_mode = "quad" if production else "parent"
    # QSA ordered-v2 sparse decode, published as request-local state on the
    # owning runner rather than a process-wide switch. It is the c1 H256
    # indexed sparse decode leaf of the ordered route: the v1 ordered switch
    # above it stays the prerequisite that decides whether any ordered decode
    # runs, the fused coefficients+values sibling stays a separate diagnostic,
    # and the dispatch-time selector still guards the leaf on head_dim 256 and a
    # selection of 1..4096 positions with the registered v1 parent as fallback.
    # The binder publishes what the retired switch bound - on for every
    # production plan, the v1 parent on the strict binder that every plan
    # falling back to it runs.
    qsa_ordered_decode_v2 = bool(production)
    # Q5_1 down M1 sibling of the expertgrid64 exact grouped down owner,
    # published as request-local state on the owning runner rather than a
    # process-wide switch. It is the parent rung of the pair2/fold ladder that
    # preempts it at rows>=64, so it publishes the value the retired switch
    # bound, production-only, for every production plan. The dispatch-time
    # selector still guards it on the exact expert-grid gate and the chain that
    # reaches it, and the strict binder (which every plan that falls back to it
    # runs) publishes False.
    q51_down_m1 = bool(production)
    # Q5_1 fold128 sibling of the grouped pair2 prefill owner, published as
    # request-local state on the owning runner rather than a process-wide
    # switch. It is the parent of the folded-pair, register-cache and
    # row-publish siblings, so it publishes the value the retired switch bound,
    # production-only, for every production plan. The dispatch-time selector
    # still guards the leaf on rows, the pair2 parent variant and the registry
    # carrying it, and the strict binder (which every plan that falls back to it
    # runs) publishes False.
    q51_fold128_prefill = bool(production)
    # Q5_1 folded-pair sibling of the fold128 prefill owner, published as
    # request-local state on the owning runner rather than a process-wide
    # switch. The fold128 rung it replaces publishes alongside it, so a plan
    # that publishes folded-pair also publishes its parent. The dispatch-time
    # selector still guards the leaf on rows, the fold128 parent variant and the
    # registry carrying it, and the strict binder publishes False.
    q51_fold_pair_prefill = bool(production)
    # Q5_1 register-cache fallback sibling of the folded-pair grouped prefill
    # owner, published as request-local state on the owning runner rather than a
    # process-wide switch. It is the fallback selection for its scope, not the
    # owner that runs: the row-publish sibling below replaces it wherever a plan
    # publishes that decision, and the routed F16 WMMA down, the exact
    # iu8-risk+repair down and the grouped-WMMA/MoE/MMQ/T16 owners preempt the
    # pair2 chain that contains it. The manifest row therefore cannot carry this
    # decision, so the binder publishes the value the retired switch bound,
    # production-only, for every production plan. The dispatch-time selector
    # still guards the leaf on rows, K640, the folded-pair parent variant and the
    # registry carrying it, and the strict binder (which every plan that falls
    # back to it runs) publishes False.
    q51_register_cache = bool(production)
    # Q5_1 row-publish sibling of the register-cache grouped prefill owner,
    # published as request-local state on the owning runner rather than a
    # process-wide switch. It is the fallback selection for its scope, not the
    # owner that runs: the shipped F16 plan replaced that manifest row with the
    # routed F16 WMMA owner, and the routed F16 down, the exact iu8-risk+repair
    # down and the grouped-WMMA/MoE/MMQ/T16 owners all preempt the pair2 chain
    # that contains it. The manifest row therefore cannot carry this decision -
    # it now names the preemptor - so the binder publishes the value the retired
    # switch bound, production-only, for every production plan. Narrowing it to
    # the row's selected variant would make the pair2 chain's row-publish
    # sibling unreachable. The dispatch-time selector still guards the leaf on
    # rows, K640, the register-cache parent variant and the registry carrying
    # it, and the strict binder (which every plan that falls back to it runs)
    # publishes False.
    q51_row_publish = bool(production)
    # PF-1 fork-b grouped Q8_0 down owner, published as request-local state on
    # the owning runner rather than a process-wide switch. It publishes the
    # value the retired switch bound, production-only, for every production
    # plan; a plan that binds the Q8_0 WMMA down owner preempts it, and the
    # strict binder (which every plan that falls back to it runs) publishes
    # False so the strict per-expert selected gemv stays the fallback.
    forkb_grouped_down = bool(production)
    # Q8_0 down row4 sibling of the sorted grouped expert-down owner, published
    # as request-local state on the owning runner rather than a process-wide
    # switch. It is the parent of the bundle and register siblings, so it
    # publishes the value the retired switch bound, production-only, for every
    # production plan. The dispatch-time selector still guards the leaf on rows,
    # the sorted parent variant and the registry carrying it, and the strict
    # binder publishes False.
    q8_down_row4_prefill = bool(production)
    # Q8_0 down bundle sibling of the row4 expert-down owner, published as
    # request-local state on the owning runner rather than a process-wide
    # switch. The row4 rung it replaces publishes alongside it. The dispatch-time
    # selector still guards the leaf on rows, the row4 parent variant and the
    # registry carrying it, and the strict binder publishes False.
    q8_down_bundle_prefill = bool(production)
    # Q8_0 down register sibling of the row4 bundle expert-down owner, published
    # as request-local state on the owning runner rather than a process-wide
    # switch. It is the fallback selection for its scope, not the owner that
    # runs: the shipped plan replaced the sorted chain's manifest row
    # (``grouped_prefill_rows_ge512_q8_0_expert_down``) with the routed F16 WMMA
    # owner, while the token-major mapped row still names the register leaf, and
    # the routed F16 down plus the Q8_0 WMMA down preempt both chains that
    # contain it. One row therefore cannot carry this decision for both paths, so
    # the binder publishes the value the retired switch bound, production-only,
    # for every production plan. The dispatch-time selector still guards the leaf
    # on rows, K640, the bundle parent variant and the registry carrying it, and
    # the strict binder (which every plan that falls back to it runs) publishes
    # False.
    q8_down_register = bool(production)
    # Q8_0 token-major mapped expert-down owner, published as request-local
    # state on the owning runner rather than a process-wide switch. It reuses
    # the row4 bundle leaf only when this invocation built the Q5_K row4 map and
    # rows>=512; the dispatch-time selector still guards the current call's map,
    # rows and the registry carrying the leaf. The mapped manifest row names the
    # register selection's leaf rather than this admission, so the binder
    # publishes the value the retired switch bound, production-only, and the
    # strict binder (which every plan that falls back to it runs) publishes
    # False.
    q8_mapped_down = bool(production)
    # GDN serial-prefill owner, published as request-local state on the owning
    # runner rather than a process-wide switch. The production binder publishes
    # the manifest row the plan already declares for the checked-index scope -
    # the narrow leaf on the UD lane, the wide leaf on a rolled-back or
    # non-recovered plan - and the strict binder (which every plan that falls
    # back to it runs) keeps the register parent. The dispatch-time selector
    # still guards the narrow leaf on its Hk16/Hv48/D128 2..4096-row bounds and
    # on the registry carrying it.
    gdn_prefill_variant = (
        next(
            (row["selected_variant"] for row in resolved.manifest["selections"]
             if row["layer"] == "gdn_recurrence_norm_gate"
             and row["scope"] == _GDN_NARROW_SCOPE),
            _GDN_WIDE_VARIANT,
        )
        if production
        else None
    )
    # GDN peer prefill owner's layer scope, published as request-local state on
    # the owning runner rather than the retired
    # ``HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL{,_LAYERS}`` pair. The production
    # binder publishes the certified global layers 35-47 for every production
    # plan except the UD lane, which the arithmetic-recovery sweep below keeps on
    # the serial owners; the strict binder (which every plan that falls back to
    # it runs) publishes no scope. The dispatch-time selector still guards the
    # owner on head_dim 128 and on the layer's own slot path, and the columnwarp
    # parent stays ahead of it inside its own gate.
    gdn_peer_prefill_layers = (
        ",".join(map(str, PRODUCTION_GDN_PEER_PREFILL_LAYERS))
        if production and resolved.manifest.get("quant") != "gguf_ud_q4_k_xl"
        else None
    )
    for key, value in {
        "HIPENGINE_GGUF_WMMA_PREFILL": "0",
        # Publish the router producer the manifest declares, so the runner's
        # launch-time switch cannot disagree with the resolved selection.
        _ROUTER_VARIANT_ENV: next(
            row["selected_variant"] for row in resolved.manifest["selections"]
            if row["layer"] == "router_logits"
        ),
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_ATTN_GATE": "0",
        # PF-3 Q5_1 M1 and PF-1 grouped Q8_0 down are T0 exact production
        # owners after the one-process/one-residency canonical gate, and their
        # selections are request-local runner state now (``q51_down_m1`` /
        # ``forkb_grouped_down`` above), so no process key names them. Strict
        # keeps the preceding owners as registered fallbacks.
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
        # Historical GR-up intent; UD recovery below disables this route after
        # its standalone depth-gate failure. The exact fused parent stays the
        # fallback pending correction; see docs/REFACTOR.md.
        "HIPENGINE_QWEN4_EXP_GR_IU8": "1" if production else "0",
        # Historical GR-down intent; UD recovery below disables this independent
        # depth-gate failure. Its projection repair condition remains open.
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
        "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK": "1" if production else "0",
        # The four expert-down ladder rungs are request-local runner state now
        # (see ``q51_fold128_prefill`` and its siblings below), so no process
        # key names them. ``HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL`` stays: it
        # still decides whether the pair2 parent variant exists at all.
        "HIPENGINE_QWEN4_EXP_Q51_PAIR_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_GDN_REGISTER_PREFILL": "1" if production else "0",
        "HIPENGINE_QWEN4_EXP_QSA_H256_WAVE_PREFILL": "page256" if production else "0",
        # The ds4-MMQ MoE suffixes (Q5_1/Q4_K selected prefill) were superseded
        # by the certified WMMA-MoE27 routing on layers 27-47 and retired, so no
        # profile binds them and no runner reads them.
        #
        # The GDN peer prefill owner's gate and layer scope were the pair
        # HIPENGINE_QWEN4_EXP_GDN_PEER_PREFILL{,_LAYERS}. Both are the owning
        # runner's state now (``gdn_peer_prefill_layers``), published below, so
        # no process key names them and the arithmetic-recovery sweep cannot
        # rewrite them.
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
        # strict fallback and a diagnostic arm pins the columnwarp parent on the
        # runner it measures), GDN_COLWARPS_PREFILL is no longer recovery-zeroed
        # for UD and the multi variant is named for every production quant.
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
        "HIPENGINE_EXECUTION_PROFILE_MANIFEST_SHA256": resolved.manifest_sha256,
    }.items():
        profile_env[key] = value
    _apply_profile_env(profile_env)
    # Publish on the generator as the durable carrier and on the runner it has
    # already built. A generator that builds its runner later still receives the
    # decision, and any c1 runner it clones inherits it; an unbound one keeps the
    # parent route.
    generator.dense_f16_pair_enabled = dp2_pair_enabled
    generator.dense_mmq_token64_enabled = dense_mmq_token64_enabled
    generator.dense_mmq_raw_vector_enabled = dense_mmq_raw_vector_enabled
    generator.dense_mmq_vec4_enabled = dense_mmq_vec4_enabled
    generator.qsa_prefill_head_mode = qsa_prefill_head_mode
    generator.qsa_ordered_decode_v2 = qsa_ordered_decode_v2
    generator.gdn_prefill_variant = gdn_prefill_variant
    generator.gdn_peer_prefill_layers = gdn_peer_prefill_layers
    generator.q51_down_m1 = q51_down_m1
    generator.q51_fold128_prefill = q51_fold128_prefill
    generator.q51_fold_pair_prefill = q51_fold_pair_prefill
    generator.q51_register_cache = q51_register_cache
    generator.q51_row_publish = q51_row_publish
    # The exact grouped Q5_1 down chain's own gate: every plan runs it, and
    # a diagnostic arm closes it on the runner it measures.
    generator.q51_exact_grouped_down = True
    # The F16 WMMA down owner below that chain: every plan publishes it off,
    # because the production grouped MoE route admits it on the Q5K gate/up
    # window by itself and the owner is otherwise a down-only diagnostic. A
    # diagnostic arm admits it on the runner it measures.
    generator.q51_wmma_down = False
    # The chain's non-expertgrid fallback rung: every plan keeps the out8
    # owner, and a diagnostic arm pins the output1 base on its own runner.
    generator.q51_out8 = True
    # The Q4 gate/up chain's own gate and its `_ALL` joint term: every plan runs
    # the chain on every Q4_K multi-row chunk, and a diagnostic arm closes or
    # narrows it on its own runner.
    generator.q4_exact_grouped_gate = True
    generator.q4_exact_grouped_all = True
    # The Q4 gate/up ladder's non-expertgrid fallback rung: every plan keeps the
    # out4 owner, and a diagnostic arm pins the strict base on its own runner.
    generator.q4_out4 = True
    # The compact F16 Q4_K dual gate/up tile pair: every plan publishes 16/16,
    # which is what the retired process keys were pinned to. ``None`` on either
    # side is the unbound heuristic, so a runner built without this bind keeps
    # it; a diagnostic arm binds a pair on the runner it measures.
    generator.q4_tile_m = 16
    generator.q4_tile_n = 16
    # The Q8_0 dense WMMA prefill tile pair for this plan. It is execution-owned
    # runner state, not a process key: the runner binds it around the scopes
    # that can launch that family, so a co-resident model keeps the arch
    # heuristic. ``None`` is the unbound heuristic; a diagnostic arm binds a
    # pair on the runner it measures (see ``q8_wmma_tiles`` in the runner).
    generator.q8_wmma_tiles = (64, 32)
    generator.forkb_grouped_down = forkb_grouped_down
    generator.q8_down_row4_prefill = q8_down_row4_prefill
    generator.q8_down_bundle_prefill = q8_down_bundle_prefill
    generator.q8_down_register = q8_down_register
    generator.q8_mapped_down = q8_mapped_down
    runner = getattr(generator, "runner", None)
    if runner is not None:
        runner.dense_f16_pair_enabled = dp2_pair_enabled
        runner.dense_mmq_token64_enabled = dense_mmq_token64_enabled
        runner.dense_mmq_raw_vector_enabled = dense_mmq_raw_vector_enabled
        runner.dense_mmq_vec4_enabled = dense_mmq_vec4_enabled
        runner.qsa_prefill_head_mode = qsa_prefill_head_mode
        runner.qsa_ordered_decode_v2 = qsa_ordered_decode_v2
        runner.gdn_prefill_variant = gdn_prefill_variant
        runner.gdn_peer_prefill_layers = gdn_peer_prefill_layers
        runner.q51_down_m1 = q51_down_m1
        runner.q51_fold128_prefill = q51_fold128_prefill
        runner.q51_fold_pair_prefill = q51_fold_pair_prefill
        runner.q51_register_cache = q51_register_cache
        runner.q51_row_publish = q51_row_publish
        runner.q51_exact_grouped_down = True
        runner.q51_wmma_down = False
        runner.q51_out8 = True
        runner.q4_exact_grouped_gate = True
        runner.q4_exact_grouped_all = True
        runner.q4_out4 = True
        runner.q4_tile_m = 16
        runner.q4_tile_n = 16
        runner.q8_wmma_tiles = (64, 32)
        runner.forkb_grouped_down = forkb_grouped_down
        runner.q8_down_row4_prefill = q8_down_row4_prefill
        runner.q8_down_bundle_prefill = q8_down_bundle_prefill
        runner.q8_down_register = q8_down_register
        runner.q8_mapped_down = q8_mapped_down
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
    f16_routes = production and resolved.manifest.get("quant") in F16_ROUTE_QUANTS
    if f16_routes:
        # Every key here is written by each bind, so a later bind without the
        # composition in the same process restores its own value.
        _apply_profile_env(_F16_ROUTE_BINDER_ENV)
    _bind_f16_routes(enabled=f16_routes)
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


def _plan_selections(
    quant: str, *, gdn_prefill_variant: str | None = None,
) -> tuple[VariantSelection, ...]:
    if quant in F16_ROUTE_QUANTS:
        return _f16_route_selections(gdn_prefill_variant=gdn_prefill_variant)
    return _production_selections(
        recovery=quant == "gguf_ud_q4_k_xl",
        gdn_prefill_variant=gdn_prefill_variant,
    )


def _wanted_keys() -> set[RuntimeProfileKey]:
    return {
        _key(quant, profile)
        for quant in QWEN4_EXP_QUANTS
        for profile in (ExecutionProfile.STRICT, ExecutionProfile.PRODUCTION)
    } | {_key(quant, ExecutionProfile.FAST) for quant in FAST_QUANTS}


def _key(quant: str, profile: ExecutionProfile) -> RuntimeProfileKey:
    return RuntimeProfileKey(QWEN4_EXP_MODEL, QWEN4_EXP_BACKEND, quant, profile)


def _registered_gdn_prefill_variant() -> str | None:
    """The serial owner the registered UD production plan declares."""

    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
        quant="gguf_ud_q4_k_xl", profile=ExecutionProfile.PRODUCTION,
    )
    return next(
        (row["selected_variant"] for row in resolved.manifest["selections"]
         if row["layer"] == "gdn_recurrence_norm_gate"
         and row["scope"] == _GDN_NARROW_SCOPE),
        None,
    )


def register_qwen4_exp_gfx1151_profiles(
    *, gdn_prefill_variant: str | None = None,
) -> bool:
    """Register the Qwen4Exp gfx1151 plans.

    ``gdn_prefill_variant`` is the named plan-construction control for the UD
    serial GDN owner: ``None`` keeps the qualified narrow selection and an
    explicit wide name registers the parent-rollback plan. It is registry
    configuration, not a request-scoped override: the plan is fixed when it is
    registered, the binder republishes that plan's row as request-local runner
    state, and a later conflicting explicit request fails instead of silently
    reusing the registered plan. A no-argument call stays idempotent.
    """

    # Validate an explicit request before the idempotent return, so a caller
    # cannot get ``False`` back for a value that names no leaf, or a plan that
    # contradicts the one it asked for.
    requested = (
        None if gdn_prefill_variant is None
        else _selected_gdn_prefill_variant(gdn_prefill_variant)
    )
    wanted = _wanted_keys()
    existing = set(registered_runtime_profile_keys())
    if wanted <= existing:
        if requested is not None:
            registered = _registered_gdn_prefill_variant()
            if registered != requested:
                raise RuntimeError(
                    "the registered Qwen4Exp UD production plan selects "
                    f"{registered!r}, not the requested {requested!r}; the plan "
                    "is fixed at registration, so clear the registry (tests) "
                    "or register in a new process"
                )
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
                selections=_plan_selections(
                    quant, gdn_prefill_variant=gdn_prefill_variant,
                ),
                kv_policy="paged_bf16_qsa_index_f32",
                graph_policy="request_owned_exact_moe_graph_c1",
                binder=_production_binder,
                qualification_failure=(
                    "Numerical envelope failed: "
                    "benchmarks/results/2026-09-30-qwen4exp-gdn-multicolumn-q4km/gate.json"
                    if quant == "gguf_q4_k_m" else None
                ),
            ),
        )
    for quant in FAST_QUANTS:
        register_runtime_profile_plan(
            model=QWEN4_EXP_MODEL,
            backend=QWEN4_EXP_BACKEND,
            quant=quant,
            profile=ExecutionProfile.FAST,
            plan=RuntimeProfilePlan(
                selections=_plan_selections(
                    quant, gdn_prefill_variant=gdn_prefill_variant,
                ),
                kv_policy="paged_bf16_qsa_index_f32",
                graph_policy="request_owned_exact_moe_graph_c1",
                binder=_production_binder,
            ),
        )
    return True


def qwen4_exp_gfx1151_profiles_registered() -> bool:
    wanted = _wanted_keys()
    return wanted <= set(registered_runtime_profile_keys())


__all__ = [
    "F16_ROUTE_Q8_DENSE_WIDE_PREFILL_LAYERS",
    "F16_ROUTE_QUANTS",
    "FAST_QUANTS",
    "PRODUCTION_GDN_PEER_PREFILL_LAYERS",
    "PRODUCTION_MOE_PREFILL_ENV",
    "PRODUCTION_Q4_DP4A_DECODE_LAYERS",
    "QWEN4_EXP_BACKEND",
    "QWEN4_EXP_MODEL",
    "QWEN4_EXP_QUANTS",
    "qwen4_exp_gfx1151_profiles_registered",
    "register_qwen4_exp_gfx1151_profiles",
]
