"""GGUF quant, layout, prefill, and decode policy declarations for hip_gfx1151.

This module owns the gfx1151 GGUF policy data: quant/shape rowtile limits,
selected-prefill and decode policies, target-verifier layouts, scratch and graph
bounds, speculative-decode policy, and paged/short-C1 attention policy. It is a
move-only extraction from ``hipengine.kernels.hip_gfx1151``: declaration values,
provenance comments, ordering dependencies, and default/fallback semantics are
unchanged. The backend package re-exports every declaration explicitly for
import compatibility.

The declarations are data only. They import no kernel bodies, touch no registry,
and select no variant at import time; backend registration remains owned by
``hipengine.kernels.hip_gfx1151``. The three geometry identities come from the
shared ``hipengine.kernels.policy`` module rather than a second geometry type.
"""

from __future__ import annotations

from hipengine.kernels.policy import (
    QWEN35_DENSE_H1024_GEOMETRY,
    QWEN35_DENSE_H5120_GEOMETRY,
    QWEN35_MOE_H2048_E256_GEOMETRY,
)


# D08-X (2026-08-15): on Qwen3.5-0.8B dense-FFN pack8 shapes the registered
# pack8 WMMA bulk consumer measures 1.97x ([3584,1024], 16x32) and 2.33x
# ([1024,3584], 64x16) versus the exact tile8x8 leaf at rows=512, within one
# BF16 ULP. The complete p512 A/B also exercised the attention pack8 shapes.
# Fail closed beyond that measured row/shape matrix until it is expanded.
GGUF_Q4_PACK8_WMMA_BULK_PREFILL = True
# Q4T16 gate/up rows33-48: exact 18-prompt gate and +9.2-9.6% complete
# prefill throughput on the gfx1151 Qwen3.8 Q4_K_M artifact.
GGUF_Q4_DUAL_SILU_PREFILL_ROW48_MAX_ROWS = 48
GGUF_Q4_PACK8_WMMA_BULK_PREFILL_SHAPES = frozenset(
    {
        (512, 1_024, 512),
        (512, 1_024, 2_048),
        (512, 1_024, 3_584),
        (512, 2_048, 1_024),
        (512, 3_584, 1_024),
    }
)
# D08-X3 retained: reuse the Q4T16 operation-complete dual-WMMA dataflow over
# the sole resident pack8 gate/up pair. The exact-model-shape route improves
# core/public pp512 by 13.81%/13.85%; two singleton WMMAs plus standalone SiLU
# remain the rollback and every other shape fails closed.
GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL = True
GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL_SHAPES = frozenset(
    {(512, 1_024, 3_584)}
)
GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL_POLICIES = {
    (QWEN35_DENSE_H1024_GEOMETRY, "MOSTLY_Q4_K_M"): frozenset({512}),
}
# D08-X2-K5 (2026-08-15): dense-BF16 bulk prefill (expanded Q6_K down owners)
# prefers the registered LDS-staged 128x64 WMMA consumer over the naive
# 32x8 scalar tile. Admit only the two Qwen3.5-0.8B p512 shapes covered by the
# complete-state A/B; other rows/shapes retain the exact fallback.
GGUF_DENSE_BF16_WMMA_BULK_PREFILL = True
GGUF_DENSE_BF16_WMMA_BULK_PREFILL_SHAPES = frozenset(
    {
        (512, 1_024, 512),
        (512, 3_584, 1_024),
    }
)
# D08-X6: exact rounded-boundary down+residual fusion is admitted only through
# the already-qualified dense-BF16 WMMA shape above. Existing small-row
# residual limits remain unchanged for their quant families.
GGUF_LINEAR_RESIDUAL_MAX_ROWS_BY_QUANT = {
    "gguf_q4_k_t16_v1": 4,
    "gguf_q5_k_t16_v1": 1,
    "gguf_q6_k_t16_qmicro_planar_v1": 3,
    # E6c-2: the raw dense-IQ residual siblings are rows-1 decode
    # composites (their parents are rows-1 owners too); the verifier
    # rows 2-4 local32/strict owners have no residual variant, so cap
    # these quants before the rows 2-4 residual path could try one.
    "gguf_iq4_xs": 1,
    "gguf_iq4_nl": 1,
    "gguf_iq3_s": 1,
    "gguf_q3_k": 1,
    "gguf_iq3_xxs": 1,
    "gguf_iq2_s": 1,
    "gguf_iq2_xs": 1,
    "bf16": 512,
}
# Clean SOL-G5 p512/d128 evidence admits the state-bound composite GGUF graph
# only when at least 128 decode transitions amortize capture/instantiate/close.
GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS = 128
# No per-KV-policy override: the backend floor above is already the horizon the
# INT8 uniform/per_token_head capture was screened at, and that screen was flat
# against eager decode, so a storage-specific floor would only restate 128.
# Declaring nothing keeps BF16, tail4_hadamard_group32, and the INT8 production
# layout on the same 128-transition gate.
GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS_BY_KV_STORAGE = {}
# gfx1100 PM4 evidence does not admit architecture-specific packets on gfx1151.
GGUF_DECODE_GRAPH_SUBMISSION_POLICIES = {
    (QWEN35_MOE_H2048_E256_GEOMETRY, "MOSTLY_Q4_K_M"): {
        "transport": "hipgraph"
    },
}
# Qwen3.8 dense Q4_K packed C2 has 880 eager launches per transition. The exact
# ten-prompt D24 gate admits HIP graph capture at all 23 remaining transitions;
# scalar C1 and every unlisted model/quant/width retain the global floor above.
GGUF_PACKED_DECODE_GRAPH_MIN_REPLAY_STEPS_BY_POLICY = {
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {2: 23},
}
# SH3-M1 admits loader-time host ownership only for private c1 sessions. Q8_0
# retains its CPU-copy route. Qwen3.8 Q4_K uses an anonymous immutable host
# copy: directly registering the file-backed mmap corrupted complete-model
# trajectories on gfx1151, while the copied mapping is graph-safe and exact.
GGUF_HOST_TOKEN_EMBEDDING_C1 = True
GGUF_HOST_TOKEN_EMBEDDING_C1_GGML_TYPES = ("Q8_0", "Q4_K")
GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1 = True
GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1_GGML_TYPES = ("Q4_K",)
GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1_COPY = True
# SH16-M2 retains one bounded private-c1 owner for allocations <=16 MiB.
# The environment selector is a temporary explicit opt-out for rollback.
GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA = True
# Qwen3.8 Q4_K_S packs every non-root immutable owner through the first complete
# inventory crossover. The matched 4K gate collapses 371 physical weight owners
# to three while preserving exact output and complete wall.
GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA_POLICIES = {
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_S"): {
        "enabled": True,
        "max_allocation_bytes": 80 * 1024 * 1024,
    },
}
# The same private-c1 geometry exposes its 188 logical state/KV ranges through
# one physical owner. Shared runners and wider batches retain dedicated owners.
GGUF_PRIVATE_C1_DECODE_SCRATCH_ARENA_POLICIES = {
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_S"): {"enabled": True},
}
# Clean LCP-2A six-case exactness, balanced-wall, and 250-transition natural
# gates admit compiler-cacheable compact-scale direct LDS32 GDN on gfx1151.
GGUF_GDN_PREFILL_AUTO_MODE = "chain_lds32_direct_nonvolatile"
# Qwen3.5-0.8B has one V head per K head and only 64 exact-LDS32 blocks at
# pp512. Complete 18-prompt semantic/decode gates and paired full-engine screens
# admit the existing Vulkan-shaped cluster8 recurrence only for the listed
# quant and (K heads, V heads, K dim, V dim) keys. P2 initially kept Q8_0 exact
# after a 0.0108% guard miss; X2-K2's fresh five-block gate superseded it.
GGUF_GDN_PREFILL_AUTO_MODES_BY_QUANT_SHAPE = {
    ("MOSTLY_Q4_K_M", 16, 16, 128, 128): "chain_peer_cluster8",
    # Qwen3.8-27B P3: compact per-K-head Q/K plus wave32/XOR is bit exact to
    # the peer-wave oracle and 1.42-1.52x faster than direct LDS32 at
    # 512/1K/4K when recurrence chunks are capped at 1K rows.
    ("MOSTLY_Q4_K_M", 16, 48, 128, 128): "chain_compact_peer_wave32",
    # Qwen3.8-27B Q4_K_S has the same GDN geometry and unchanged recurrent
    # state math; only projection tensor quant assignments differ.
    ("MOSTLY_Q4_K_S", 16, 48, 128, 128): "chain_compact_peer_wave32",
    # D08-X2-K2 (2026-08-15): the exact-core gate measures the Vulkan-shaped
    # cluster8 recurrence at +16.70% Q8_0 pp512 with neutral core-graph tg128,
    # so the same one-V-head-per-K-head geometry now uses cluster8 for Q8_0 too.
    # HIPENGINE_GGUF_GDN_PREFILL_MODE remains the explicit override.
    ("MOSTLY_Q8_0", 16, 16, 128, 128): "chain_peer_cluster8",
}
# The 4K unchunked compact recurrence loses 8.26% to direct LDS32. Four
# state-carrying 1K recurrence launches are peer-bit-exact and win 1.422x;
# prepare and RMSNorm remain one complete-chain launch each.
GGUF_GDN_PREFILL_COMPACT_PEER_CHUNK_ROWS = 1024
# The architecture-scoped strict-exact selector resolves to the same proven
# nonvolatile direct route as gfx1151 production.
GGUF_GDN_PREFILL_EXACT_MODE = "chain_lds32_direct_nonvolatile"
# SH-M2 transfers the existing route/stage liveness admission to gfx1151's
# proven compact exact GDN route. SH2-M3 extends the same independent owner-slot
# topology to the right-sized 768-row class; diagnostics retain independently
# owned fields and capability denial retains the existing dedicated fallback.
GGUF_PREFILL_SCRATCH_LIVENESS_ALIAS = True
GGUF_PREFILL_SCRATCH_LIVENESS_MIN_ROWS = 768
GGUF_PREFILL_SCRATCH_ARENA_GROUPING = "owner_slots"
# Dense Qwen3.8 Q4_K_S keeps every bulk-prefill field dedicated: both owner-slot
# and single-arena liveness aliases changed logits despite preserving repeated-
# token top-1. Bulk-prefill scratch rows are capacity-conditional: 4K-class
# requests grow to the natural full-attention 4K query-chunk plateau (which
# admits the Q5 source-F16 route, +2.97% measured at 4K), while 8K and larger
# keep 1,024-row chunks because 4,096-row chunks measured -2.3% slower at 8K
# regardless of source-F16. Memory therefore stays flat as context grows past
# 8K; on 24GB-class cards the autotuner drops the full-attn query chunk to
# 1024/768 at 52K+/128K+ anyway, and gfx1100 has no K_S source-F16 admission,
# so this policy is gfx1151-only.
GGUF_DENSE_PREFILL_SCRATCH_ROW_CAP_POLICIES = {
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_S"): {
        "min_capacity": 4_096,
        "max_rows_by_capacity": {
            4_096: 4_096,
            8_192: 1_024,
        },
    },
}
# Nathan-review P0 exactness, copy-inclusive sub-window, and right-sized
# 512/4K/32K/64K gates admit one reusable head-major BF16 K/V pair before
# AOTriton. Runtime capacity remains capped at the validated 64K allocation
# class, and env=0 or any allocation denial restores strided AOTriton exactly.
GGUF_AOTRITON_HEAD_MAJOR_KV = True
# R2's complete packed numerical/isolation gate plus counterbalanced engine and
# real-Uvicorn serving A/B admit FP16 recurrent-state storage for dense Q4_K_S.
# The environment remains an explicit rollback: =0 restores FP32 storage.
GGUF_FP16_RECURRENT_STATE_DEFAULT_FILE_TYPES = frozenset({"mostly_q4_k_s"})
# Dense linear-attention verifier rows above the full-attention split threshold
# stage the batched projection/head-norm/rotary/KV-write/output-projection chain
# instead of the per-row scalar c1 owner. The staged chain reassociates BF16
# projections, so the default binds the loader-qualified plain artifact identity
# (a UD/unknown-manifest preset never inherits it) and the file types whose
# eager-route teacher parity, straddle-band and task packets were measured:
# Qwen3.8-27B Q4_K_M on 2026-09-19 is byte-exact against the serial-exact
# teacher at the 1024/3530/8192 straddles with the staged owner confirmed by
# route counters, and the committed long-context task packet binds task,
# AR-id and eager-ownership equality at a 4,096-token context. The row-wise c1
# route remains the registered strict fallback behind
# ``HIPENGINE_GGUF_STAGED_LINEAR_ROWS_LONG=0``. Other file types keep the
# row-wise route until their own packet is measured.
GGUF_STAGED_LINEAR_ROWS_LONG_DEFAULT_FILE_TYPES = frozenset({"mostly_q4_k_m"})
# Default AOTriton ON for gfx1151. An earlier 2026-08-20 slice measurement
# (64..2048) claimed native causal_gqa_gate_bf16 was ~2-5% faster with no
# crossover, which set this to False; but the native full-attention path scales
# badly in bulk prefill and collapses at mid/long context. Measured 2026-08-21
# (35B-A3B @ 8060S): native 1K=828, 2K=362, 4K=234 tok/s (and can hang/fail at
# 2K+) while AOTriton holds flat ~1300-1330 tok/s at 1K/2K/4K (1274 at 512).
# Below the 512-token crossover native is still used regardless. This restores
# the retained 4K prefill (~1430 tok/s) that predated the False default.
GGUF_AOTRITON_PREFILL = True
# Measured 2026-10-03 (35B-A3B UD-Q4_K_M, 8060S, accelerator-performance
# profile, shipping AR route, 4 reps, 4 s settle, one session per size): the
# 512-row override this table previously carried is NOT the fastest value. Per
# shape, tok/s at 512 / 1024 / 4096 linear/MoE chunks:
#   1K:    1454.8 / 1486.4 (+2.17%) / 1488.2 (+2.29%)
#   4K:    1459.3 / 1487.7 (+1.95%) / 1503.9 (+3.06%)
#   16K:   1358.5 / 1358.5 (0.00%)  / 1358.3 (-0.01%)
#   64K:    969.9 /  969.5 (-0.04%) /  969.2 (-0.07%)
# so the lever is confined to short prompts and 1024 is the retained value.
# 1024 is also **bit-exact** against 512 on this path (KL 0.00000000,
# max_abs_logit 0.00000000, top-1 identical, 4096 tokens) because row chunking
# partitions tokens rather than reductions - see
# scripts/gguf_prefill_chunk_boundary_gate.py. 4096 is faster still at 1K/4K but
# is NOT bit-exact (KL 3.1e-07, max_abs_logit 1.279 on a low-probability token)
# and costs more at 64K, so it is recorded and not promoted.
# This entry is load-bearing as a shield: the gfx1151 arch profile below still
# carries (256, 256), which measures as the WORST value for this geometry
# (1363.7 tok/s at 4K), so removing this key would regress the lane.
GGUF_PREFILL_CHUNK_SIZES_BY_GEOMETRY = {
    (QWEN35_MOE_H2048_E256_GEOMETRY, "MOSTLY_Q4_K_M"): (1024, 1024),  # (linear, moe)
}
# F3's independent-c1 and physical-width gates admit the one-token-per-row
# indexed GDN sibling for packed AR while retaining segmented GDN as fallback.
GGUF_GDN_INDEXED_SINGLETON_DECODE = True
# The unrestricted historical policy remains false because current
# ZBook-local counterbalanced p512/d128 evidence rejects c2; its absolute rates
# are independent of the other gfx1151 host. Production free-running IDs are
# diagnostic, not binding; the complete strict-teacher gate admits exact c4/c8
# logits, and width-scoped timing wins both shapes. Keep c2 on the direct owner.
GGUF_Q8_T16_DECODE_ROWTILE_ALL = False
# Preserve the shared singleton shape policy independently of the broad floor.
GGUF_Q8_T16_SINGLE_ROWTILE_VARIANTS_BY_SHAPE = {
    (5120, 1024): "t16_gemv_decode_rowtile4_bf16_bf16_out",
}
GGUF_Q8_T16_DECODE_ROWTILE_MIN_ROWS = 4
# The repaired 128-thread pair-only route preserves production reduction order.
# Its independent fallback floor remains physical c8; the all-projection c4
# policy above may still select the same rowtile pair under that broader scope.
GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS = 8
# Exact dynamic expert-ID pairing removes duplicate C8 Q4T16 gate/up weight
# reads while keeping each row's production 128-thread reduction order.
# Physical widths below C8 remain on the established kernel.
GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS = 8
# Exact Qwen c1/top8 Q5T16 selected-down splits each T16 tile into two
# eight-column owners after the gfx1151 leaf clears SH-D1 admission.
GGUF_Q5_T16_SELECTED_QWEN_TILE8 = True
# The same exact dynamic expert-ID pairing is retained for Q5T16 selected-down
# only at physical C8; lower widths preserve the established kernel.
GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS = 8
# Three Q6T16 down layers use the independently gated exact sibling at C8.
GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS = 8
# Qwen3.8-27B P1 candidate: dense H5120 sole-Q4 ownership reuses the existing
# operation-complete T16 family. Architecture-local primitive, actual-weight,
# full-state, natural-suite, memory, and performance gates decide retention.
GGUF_DENSE_Q4_T16 = True
# Qwen3.8 P5 replaces only the Q4_K_S H5120/N17408 gate/up pair with the
# byte-neutral qmicro payload. Direct-metadata c1/native leaves and the bounded
# 4K metadata expansion route are exact and operation-complete; Q4_K_M keeps
# the later requalified standard-T16 owner and every other Q4 role keeps its
# independently qualified T16 owner.
GGUF_DENSE_Q4_QMICRO_T16_GATE_UP = True
GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES = ("MOSTLY_Q4_K_S",)
# Qwen3.8-27B P2 retains the 48 K6144/N5120 recurrent outputs as sole
# Q5T16 after architecture-local actual-weight, GDN-handoff, full-state,
# natural-suite, memory, and performance gates.
GGUF_DENSE_Q5_T16_SSM_OUT = True
# Qwen3.8-27B Q4_K_S candidate: compact the exact H5120 Q5 FFN-down,
# recurrent-QKV, and full-attention-V roles through the operation-complete
# direct/rowtile/WMMA T16 family. Q4_K_M has no Q5 tensors at these shapes.
GGUF_DENSE_Q5_T16_H5120 = True
# D08-P6 admits the same sole-resident family independently for the exact
# Qwen3.5-0.8B K2,048/N1,024 recurrent-output role.
GGUF_DENSE_Q5_T16_SSM_OUT_08B = True
# D08-P4 admits sole compact Q4T16 only for the six Qwen3.5-0.8B full-attention
# K1,024/N4,096 Q projections; other Q4 roles and 27B keep their prior owners.
GGUF_DENSE_Q4_T16_ATTN_Q_08B = True
# D08-D3 keeps every Qwen3.5-0.8B Q4 gate/up pair in its sole pack8 layout and
# selects the existing operation-complete fused-SiLU leaf at t128 only for c1.
# Qwen3.8 Q4_K_M serial c1 uses exact local32 after the formerly retained
# residual-Q8_1x2 split-weight owner lost a ZBook-local counterbalanced rebase;
# native B1 keeps its independently qualified Q8_1x2 owner. Q4_K_S serial c1
# independently keeps the exact split-weight owner. Its native rows2-4 require
# the row-independent Q8_1x2 rowtile because direct-BF16 association can change
# greedy trajectories. Peer geometries and policy misses retain their owners.
GGUF_DENSE_PAIR_SILU_DECODE_POLICIES = {
    (QWEN35_DENSE_H1024_GEOMETRY, "MOSTLY_Q4_K_M"): {
        (1, 1_024, 3_584): "pack8_dual_decode_t128_bf16_bf16_out",
    },
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {
        (1, 5_120, 17_408): "dense_dual_local32_bf16_bf16_out",
    },
    # 2026-09-24, UD-GFX1151-OPTIMIZE2 E2c: a bound UD admission preset
    # extends the policy identity to (geometry, file_type, preset), so the
    # plain rows above never apply to a UD artifact that merely shares the
    # stamp (see _gguf_policy_identity). Without this explicit row the
    # rows==1 pair launcher was never entered for the UD artifact: gate/up
    # pairs ran as two projections plus the separate SiLU, and the registered
    # IQ4_XS local32 dual owner - measured bit-exact with the
    # single/single/silu chain on the real blk.1 pair (E2b') - could not
    # fire. The value mirrors the plain Q4_K_M row so shared-quant pairs take
    # the same qualified owner in both arms; the IQ4_XS dual branch selects
    # its own key and does not read this value.
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M", "gguf_ud_q4_k_m"): {
        (1, 5_120, 17_408): "dense_dual_local32_bf16_bf16_out",
    },
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_S"): {
        (1, 5_120, 17_408): (
            "dense_dual_q8_1x2_split_weight_dp4a_bf16_bf16_out"
        ),
    },
}
GGUF_DENSE_PAIR_SILU_NATIVE_DECODE_POLICIES = {
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {
        (1, 5_120, 17_408): "dense_dual_q8_1x2_dp4a_bf16_bf16_out",
    },
    # 2026-09-24, E2c: native-session mirror of the explicit UD row in the
    # serial table above, so the UD stamp resolves the same qualified owner
    # regardless of which capability table is consulted first.
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M", "gguf_ud_q4_k_m"): {
        (1, 5_120, 17_408): "dense_dual_q8_1x2_dp4a_bf16_bf16_out",
    },
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_S"): {
        (1, 5_120, 17_408): (
            "dense_dual_q8_1x2_split_weight_dp4a_bf16_bf16_out"
        ),
        **{
            (rows, 5_120, 17_408): (
                "dense_dual_q8_1x2_rowtile8_dp4a_bf16_bf16_out"
            )
            # rowtile8 instantiates ROW_TILE 2..8 in one launch; c>8 chunks
            # into <=8-row groups at the dispatch site. The decode regime goes
            # to rows 511 so no gate/up concurrency silently falls to WMMA;
            # c>=512 routes to the existing WMMA prefill owner before this
            # policy is consulted.
            for rows in range(2, 512)
        },
    },
}
# Qwen3.8 P5 independently qualifies the exact same-input F32 alpha/beta pair
# for scalar recurrent layers. Native rows and every other shape retain two
# singleton dense-F32 projections.
GGUF_DENSE_F32_ALPHA_BETA_PAIR_DECODE_SHAPES = frozenset(
    {(1, 5_120, 48, 48)}
)
# Qwen3.8's 24 standard-Q6 recurrent QKV owners and Q4 gates consume the same
# BF16 norm row. One local128 mixed grid preserves both singleton arithmetic
# trees while removing their serial launch boundary. Shape/backend misses and
# native rows retain the two registered primitive projections.
GGUF_Q6_Q4_T16_MIXED_GRID_DECODE_SHAPES = frozenset(
    {(1, 5_120, 10_240, 6_144)}
)
# The 16 full-attention K/V pairs share one BF16 norm row. K is compact Q4T16
# and V is either Q4T16 or byte-neutral planar-Q6; one local128 block-parallel
# grid preserves each qualified singleton arithmetic tree while removing the
# serial launch boundary. Native rows, other shapes, and peers retain the two
# primitive projections.
GGUF_NARROW_KV_PAIR_DECODE_SHAPES = frozenset(
    {(1, 5_120, 1_024, 1_024)}
)
# The graph-safe serial composite joins those independent projections with the
# current in-place Conv channel blocks. Wider/native rows and peer backends keep
# the separately registered pair plus Conv fallback.
GGUF_DENSE_F32_ALPHA_BETA_CONV_DECODE_SHAPES = frozenset(
    {(1, 5_120, 48, 10_240, 4)}
)
# Qwen3.8 serial full-attention consumes packed BF16 Q/gate and BF16 K in one
# exact head RMSNorm/RoPE launch, removing the split and K-cast graph nodes.
GGUF_FULL_ATTN_QK_POSTPROCESS_DECODE_POLICIES = {
    (1, 24, 4, 256): "qwen35_position_qk_bf16_f32",
}
# D08-D3B keeps the current Q4-pack8 and dense-BF16 residents and selects their
# exact rounded-BF16 residual siblings only for the 24 c1 dense-down owners.
# Qwen3.8 P5 independently admits same-resident direct Q4/Q6 c1 siblings at
# H5120/FFN17408 after the complete graph-node gate. Q4_K_S transfers only its
# Q4 down owners; its Q5 down owners retain the registered unfused fallback.
# Native rows, Q8, other models/shapes, and peer backends remain unchanged.
GGUF_DENSE_DOWN_RESIDUAL_DECODE_POLICIES = {
    (QWEN35_DENSE_H1024_GEOMETRY, "MOSTLY_Q4_K_M"): {
        (1, 3_584, 1_024): True,
    },
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {
        (1, 17_408, 5_120): True,
    },
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_S"): {
        (1, 17_408, 5_120): True,
    },
    # E6c down-residual fold, unqualified-manifest lane (kept for unpinned
    # Q4_K_M-stamped H5120 artifacts) and the certificate-bound UD preset
    # lane the resident Qwen3.8-27B UD-Q4_K_M artifact resolves to at
    # runtime. Both rows are capability-shaped: dense H5120 geometry +
    # plain-lane stamp + rows-1 down shape. Kernels resolve per quant -
    # registered exact siblings (Q5T16/Q4T16/planar Q6) fold, raw-IQ down
    # owners fall back to the unfused chain because they have no residual
    # sibling or raw-ABI launcher yet.
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M", "gguf-unqualified-manifest"): {
        (1, 17_408, 5_120): True,
    },
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M", "gguf_ud_q4_k_m"): {
        (1, 17_408, 5_120): True,
    },
}
# D08-D5 admits both fixed-hidden wave-reduction leaves as one inseparable C
# route for the exact dense-0.8B Q4 decode owner. The dense-H5120 sibling
# preserves the generic local256 reduction tree exactly while caching 20 values
# per lane; all 128 actual Qwen3.8 leaves are bit exact and clear the P5 package
# gate. The reduction is quant-independent, so the qualified Q4_K_S geometry
# shares it. Q8, native batches, output norm, other shapes/models, and peer
# backends keep the generic primitives.
GGUF_NORM_RESIDUAL_DECODE_POLICIES = {
    (QWEN35_DENSE_H1024_GEOMETRY, "MOSTLY_Q4_K_M"): {
        (1, 1_024): "bf16_out_fixed1024_wave256",
    },
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {
        (1, 5_120): "bf16_out_fixed5120_wave256",
    },
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_S"): {
        (1, 5_120): "bf16_out_fixed5120_wave256",
    },
}
# The measured physical-c8 owner is two exact c4 rowtiles, not generic WMMA.
GGUF_T16_NATIVE_SPLIT_ROW_CHUNKS_BY_QUANT_SHAPE = {
    "gguf_q4_k_t16_v1": {(8, 1_024, 4_096): 4},
}
# D08-P1 admits the existing direct/rowtile/WMMA Q5T16 family only for the
# exact Qwen3.5-0.8B linear-attention QKV role selected by the materializer.
GGUF_DENSE_Q5_T16_QKV = True
# Strict one-wave/one-16-row-tile WMMA ownership for the six actual Qwen3.8
# standard-Q4 shapes that beat their padded parent at every physical width.
# Narrow K/V K5120/N1024 loses to shared-B and remains on that fallback; all
# peer backends, rows, and shape misses retain their source registrations.
GGUF_Q4_T16_PHYSICAL_SMALLM_ROWS = frozenset({6, 8, 12, 16})
# Supersession bound (scaling-campaign M2j): physical rows at or below this
# ceiling route to the measured low-VGPR/shared-B2W2 siblings below instead of
# the one-row-tile smallm owner. The smallm launcher remains registered.
GGUF_Q4_T16_PHYSICAL_SMALLM_MAX_ROWS = 16
GGUF_Q4_T16_PHYSICAL_SMALLM_SHAPES = frozenset(
    {
        (5_120, 6_144),
        (5_120, 10_240),
        (5_120, 12_288),
        (5_120, 17_408),
        (6_144, 5_120),
        (17_408, 5_120),
    }
)
# Measured 2026-08-29 low-M dense prefill bands (parity campaign P2.3/P2.1).
# Outputs are bit-exact strict siblings with the same per-tile K16 WMMA/BF16
# association. Seven-point rows52-80 screens extend the original rows17-48
# low-VGPR policy through 80 without prompt-specific row checks; unadmitted
# shapes and rows >80 retain shared-B fail-closed.
GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS = 80
GGUF_Q4_T16_DENSE_LOWM_SHAPES = frozenset(
    {
        (5_120, 6_144),
        (5_120, 10_240),
        (5_120, 12_288),
        (5_120, 17_408),
        (6_144, 5_120),
        (17_408, 5_120),
    }
)
GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS = 32
GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS = 48
GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS = 64
GGUF_Q4_T16_DENSE_LOWVGPR48_SHAPES = frozenset(
    {
        (5_120, 10_240),
        (5_120, 12_288),
        (5_120, 17_408),
    }
)
GGUF_Q4_T16_DENSE_LOWVGPR64_SHAPES = frozenset(
    GGUF_Q4_T16_DENSE_LOWM_SHAPES - {(5_120, 17_408)}
)
GGUF_Q4_T16_DENSE_LOWVGPR80_SHAPES = frozenset({(17_408, 5_120)})
# At rows65-80, Q6 planar keeps `<2,1>` on these four shapes and uses
# `<3,1>` on the other two. Rows49-64 use `<2,1>` on all six.
GGUF_Q6_PLANAR_LOWVGPR80_SHAPES = frozenset(
    {
        (17_408, 5_120),
        (5_120, 12_288),
        (6_144, 5_120),
        (5_120, 17_408),
    }
)
# Q5 uses separately measured per-band shape sets. The 96/112-VGPR owners
# preserve the plain owner's per-tile order; row/shape misses retain plain.
GGUF_Q5_T16_DENSE_LOWM_SHAPES = GGUF_Q4_T16_DENSE_LOWM_SHAPES
# Y2: exact shared-weight one-sweep owner for the sole physical Qwen3.8 Q5
# recurrent output shape. Every row/shape miss retains the prior exact owner.
GGUF_Q5_T16_DENSE_SHARED8R3_MIN_ROWS = 256
GGUF_Q5_T16_DENSE_SHARED8R3_MAX_ROWS = 384
GGUF_Q5_T16_DENSE_SHARED8R3_SHAPES = frozenset({(6_144, 5_120)})
GGUF_Q5_T16_DENSE_LOWVGPR_MAX_ROWS = 32
GGUF_Q5_T16_DENSE_LOWVGPR48_MAX_ROWS = 48
GGUF_Q5_T16_DENSE_LOWVGPR64_MAX_ROWS = 64
GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS = 80
GGUF_Q5_T16_DENSE_LOWVGPR_SHAPES = frozenset({(17_408, 5_120)})
GGUF_Q5_T16_DENSE_LOWVGPR48_SHAPES = frozenset(
    {
        (5_120, 6_144),
        (5_120, 10_240),
        (5_120, 12_288),
        (6_144, 5_120),
    }
)
GGUF_Q5_T16_DENSE_LOWVGPR64_SHAPES = frozenset(
    {
        (5_120, 6_144),
        (17_408, 5_120),
        (5_120, 10_240),
        (6_144, 5_120),
    }
)
GGUF_Q5_T16_DENSE_LOWVGPR80_SHAPES = frozenset({(5_120, 12_288)})
# High-row periodic bands (rows81-144) from the rows96/120/134 screen. The
# cut points follow 32/48/64-row owner capacities rather than benchmark prompt
# lengths. Q4 rows145+ retain shared-B; Q5 rows145+ retain plain.
GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS = 96
GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS = 128
GGUF_Q4_T16_DENSE_LOWVGPR144_MAX_ROWS = 144
GGUF_Q4_T16_DENSE_LOWVGPR96_SHAPES = frozenset(
    {(17_408, 5_120), (5_120, 12_288)}
)
GGUF_Q4_T16_DENSE_PLAIN128_SHAPES = frozenset({(5_120, 6_144)})
GGUF_Q4_T16_DENSE_LOWVGPR128_SHAPES = frozenset(
    {(17_408, 5_120), (5_120, 10_240)}
)
GGUF_Q4_T16_DENSE_LOWVGPR48_128_SHAPES = frozenset(
    {(5_120, 12_288), (6_144, 5_120)}
)
GGUF_Q4_T16_DENSE_LOWVGPR48_144_SHAPES = frozenset(
    {(5_120, 6_144), (17_408, 5_120), (5_120, 12_288), (6_144, 5_120)}
)
# Reduced-accumulator shared-B variants preserve the same per-output K16
# schedule. Capacity-periodic screens at rows192/256/320/384 admit only these
# shape bands; rows385+ retain the 48-column/4-wave parent.
GGUF_Q4_T16_DENSE_SHARED2_192_MAX_ROWS = 192
GGUF_Q4_T16_DENSE_SHARED2_256_MAX_ROWS = 256
GGUF_Q4_T16_DENSE_SHARED2_384_MAX_ROWS = 384
GGUF_Q4_T16_DENSE_SHARED2W4_192_SHAPES = frozenset({(17_408, 5_120)})
GGUF_Q4_T16_DENSE_SHARED2W2_192_SHAPES = frozenset({(6_144, 5_120)})
GGUF_Q4_T16_DENSE_SHARED2W2_256_SHAPES = frozenset(
    {(17_408, 5_120), (6_144, 5_120)}
)
GGUF_Q4_T16_DENSE_SHARED2W2_384_SHAPES = frozenset(
    GGUF_Q4_T16_DENSE_LOWM_SHAPES - {(5_120, 12_288)}
)
# Y1 exact one-sweep row band. Eight waves x three row tiles cover 384 rows
# while retaining the parent's 48-column ownership. Rows288/320/384 actual-
# weight screens admit only these three consistently positive shapes.
GGUF_Q4_T16_DENSE_SHARED3W8R3_384_MIN_ROWS = 288
GGUF_Q4_T16_DENSE_SHARED3W8R3_384_SHAPES = frozenset(
    {(17_408, 5_120), (5_120, 12_288), (5_120, 17_408)}
)
GGUF_Q5_T16_DENSE_LOWVGPR96_MAX_ROWS = 96
GGUF_Q5_T16_DENSE_LOWVGPR128_MAX_ROWS = 128
GGUF_Q5_T16_DENSE_LOWVGPR144_MAX_ROWS = 144
GGUF_Q5_T16_DENSE_LOWVGPR96_SHAPES = frozenset(
    {(17_408, 5_120), (5_120, 12_288)}
)
GGUF_Q5_T16_DENSE_LOWVGPR48_128_SHAPES = frozenset(
    {(17_408, 5_120), (5_120, 10_240), (5_120, 12_288), (6_144, 5_120)}
)
GGUF_Q6_PLANAR_LOWVGPR96_MAX_ROWS = 96
GGUF_Q6_PLANAR_LOWVGPR128_MAX_ROWS = 128
GGUF_Q6_PLANAR_LOWVGPR144_MAX_ROWS = 144
GGUF_Q6_PLANAR_LOWVGPR96_SHAPES = frozenset(
    {(17_408, 5_120), (5_120, 12_288)}
)
GGUF_Q6_PLANAR_LOWVGPR128_SHAPES = frozenset({(17_408, 5_120)})
GGUF_Q6_PLANAR_LOWVGPR48_128_SHAPES = frozenset({(6_144, 5_120)})
GGUF_Q6_PLANAR_SHARED4_128_SHAPES = frozenset(
    {(5_120, 10_240), (5_120, 17_408)}
)
GGUF_Q6_PLANAR_LOWVGPR48_144_SHAPES = frozenset(
    {(17_408, 5_120), (6_144, 5_120)}
)
GGUF_Q6_PLANAR_SHARED4_144_SHAPES = frozenset(
    GGUF_Q4_T16_DENSE_LOWM_SHAPES - GGUF_Q6_PLANAR_LOWVGPR48_144_SHAPES
)
# Exact standard-Q4 two-wave/16-column output ownership. The shape map is the
# independently qualified physical-row8 scope. ``rows_by_shape`` narrows the
# OI-1 small-M extension to actual-weight winners; unspecified shapes therefore
# remain row8-only. The WG32/eight-column rowtile is the registered fallback.
GGUF_T16_NATIVE_ROWTILE_VARIANTS_BY_QUANT = {
    "gguf_q4_k_t16_v1": {
        "shapes": {
            (5_120, 1_024): "dense_rowtile16_w2_bf16_bf16_out",
            (5_120, 5_120): "dense_rowtile16_w2_bf16_bf16_out",
            (5_120, 6_144): "dense_rowtile16_w2_bf16_bf16_out",
            (5_120, 10_240): "dense_rowtile16_w2_bf16_bf16_out",
            (5_120, 12_288): "dense_rowtile16_w2_bf16_bf16_out",
            (17_408, 5_120): "dense_rowtile16_w2_bf16_bf16_out",
        },
        "rows_by_shape": {
            (5_120, 10_240): (3, 4, 8),
            (5_120, 12_288): (2, 3, 4),
        },
    },
}
# Packed target verification may reuse only these exact actual-weight Q5/Q6
# rowtiles. Shape-scoped ownership prevents unrelated gfx1151 models and Q4
# projections from inheriting an unmeasured verifier policy.
GGUF_T16_TARGET_VERIFIER_ROWTILE_SHAPES_BY_QUANT = {
    "gguf_q5_k_t16_v1": frozenset({(6_144, 5_120)}),
    "gguf_q6_k_t16_v1": frozenset({(5_120, 10_240)}),
    "gguf_q6_k_t16_qmicro_planar_v1": frozenset(
        {(5_120, 1_024), (17_408, 5_120)}
    ),
}
# Physical target rows above the native rowtile limit must be admitted
# separately. Keep this bounded to observed C3 K2/K3 R9/R12 cells; C4+ and
# arbitrary verifier widths retain their prior owners even when the shape is
# listed. C3/K1 R6 already fits the exact native Q5/Q6 rowtile scope.
GGUF_T16_TARGET_VERIFIER_ROWTILE_CHUNK_ROWS_BY_QUANT = {
    "gguf_q5_k_t16_v1": frozenset({9, 12}),
    # M1 measured the R20-R32 chunk classes under the single-group wide cycle;
    # they stay unengaged at the certified width-4 default and return when a
    # profile re-lists a bound >= 5.
    "gguf_q6_k_t16_v1": frozenset({9, 12, 16}),
    "gguf_q6_k_t16_qmicro_planar_v1": frozenset({9, 12, 16}),
}
# E2 standard-Q6 true-R12: exact one-sweep col8 wins its actual K5120/N10240
# target shape. Planar K5120/N1024 and K17408/N5120 lose their all-shape leaf
# screen and retain R8+R4.
GGUF_T16_TARGET_VERIFIER_TRUE_ROWTILE_VARIANTS = {
    ("gguf_q4_k_t16_v1", 16, 5_120, 1_024): (
        "t16_wmma_prefill_shared_b2r1_bf16_bf16_out"
    ),
    ("gguf_q4_k_t16_v1", 16, 17_408, 5_120): (
        "t16_wmma_prefill_shared_b2r1_bf16_bf16_out"
    ),
    ("gguf_q6_k_t16_v1", 12, 5_120, 10_240): (
        "t16_gemv_rowtile12_col8_bf16_bf16_out"
    ),
    ("gguf_q5_k_t16_v1", 12, 6_144, 5_120): (
        "t16_gemv_rowtile12_col8_bf16_bf16_out"
    ),
    ("gguf_q5_k_t16_v1", 16, 6_144, 5_120): (
        "t16_gemv_rowtile16_col8_bf16_bf16_out"
    ),
}
# B5 changed-arithmetic candidate. Generic dispatch reads this backend-owned
# map only while a caller-owned integer-MMQ workspace context is active.
# Standard Q6 QKV and all Q4/Q5 shapes remain on their current owners.
# UD item 3: dense raw-IQ integer-MMQ prefill. The selected/MoE kernel reads
# weights as qweight + expert*expert_bytes + out_row*weight_row_bytes, so at
# expert 0 it consumes the dense raw GGUF layout unchanged - the dense route is
# the degenerate single-expert case, with no repack and no weight sidecar. The
# strict per-row GEMV stays the registered fallback and the rows=1 decode owner;
# this route is inert unless an execution owner binds its workspace.
#
# min_rows is a measured crossover, not a capability. The GEMV re-reads the
# weight column once per 8-row slab so it grows with rows, while the MMQ pads to
# 128 rows so a short prompt pays for work it does not use. Swept over the six
# distinct IQ4_XS dense shapes in the published UD file
# (scripts/gguf_iq_dense_mmq_crossover.py, 2026-09-08, gfx1151), min-of-5
# interleaved, worst case over shapes:
#
#   rows=2  0.59x   rows=4  0.68x   rows=8  1.58x   rows=16  2.62x   rows=32  4.34x
#
# so 8 is the first row count where the MMQ wins on every measured shape; 4
# still loses on attn_gate (6144x5120). Agreement held across all 96
# (shape, rows) cells at max relative error 0.0025-0.0099.
# Dense raw-IQ prefill route selection. Scored the way
# EXECUTION-PROFILES.md section 6 specifies - candidate production against
# hipEngine strict, not against an external engine - over 162 teacher-forced
# rows:
#
#   route                     prefill      mean       p95       p99       max  >5e-2
#   two-quant integer MMQ      127.2   0.001038  0.003479  0.014385  0.049730    0
#   four-quant integer MMQ     171.9   0.002110  0.006797  0.024044  0.170390    1
#   four-quant W4A16 (below)   150.9   0.000827  0.004547  0.012475  0.023513    0
#
# The integer-MMQ route breaches the binding 5e-2 absolute maximum-row ceiling
# by 3.4x at one position; W4A16 passes every threshold in the calibrated
# envelope. W4A16 is therefore the default at a 12% throughput cost, and the
# integer-MMQ variant name is kept below so a route swap is a one-word edit.
#
# Route coverage differs. The integer-MMQ route needs each K32 group to be
# expressible as scale*int8, which excludes Q3_K, IQ2_S and IQ2_XS: they carry
# a scale per 16 elements, so under a per-32 scale their residuals reach +/-128
# and +/-1333. W4A16 expands per element and has no such contract, so it serves
# every dense IQ/Q3_K quant. Measured against the strict GEMV at 512 rows:
# Q3_K 86.7 -> 14.8 ms (5.9x), IQ2_S 82.3 -> 14.9 (5.5x), IQ2_XS 80.8 -> 13.3
# (6.1x). Swapping the policy back to the integer variant must therefore also
# drop these three, or they will find no registered owner and fall through to
# the strict GEMV - correct, but silently slower.
_IQ_DENSE_W4A16_VARIANT = "dense_wmma_w4a16_prefill_bf16_bf16_out"
_IQ_DENSE_INTEGER_MMQ_VARIANT = (
    "dense_mmq_i128_j128_k256_q8_1_ds4_prefill_bf16_bf16_out"
)
# min_rows is a measured crossover for the integer-MMQ route (worst case over
# shapes: 0.59x at 2 rows, 0.68x at 4, 1.58x at 8). W4A16 has no 128-row
# padding so it does not need a floor, but the shared value is kept because it
# is safe for both and keeps a route swap free of a second variable.
GGUF_IQ_DENSE_PREFILL_POLICY = {
    quant: {
        "min_rows": 8,
        "max_rows": 131072,
        "variant": _IQ_DENSE_W4A16_VARIANT,
    }
    for quant in ("gguf_iq4_xs", "gguf_iq3_xxs", "gguf_iq3_s", "gguf_iq4_nl",
                 "gguf_q3_k", "gguf_iq2_s", "gguf_iq2_xs")
}
# Q3_K, IQ2_S and IQ2_XS history: the integer-MMQ contract could never
# express them (per-16 scales put their residuals at +/-128 and +/-1333),
# but W4A16 expands per element and has no such constraint. Routed unsplit
# they measured 176.5 tok/s on the zbook - faster than the integer-MMQ route
# ever was - but moved the strict-referenced mean 0.000827 -> 0.001061, 6%
# over the calibrated 1e-3 limit, which held them out (2026-09-09 record:
# the entire delta was Q3_K alone, +0.000234 mean for +26 tok/s).
#
# 2026-09-10 unblock (gfx1100 lane, shared source): the W4A16 kernel now
# expands these three through the hi+lo split path - the WMMA products of
# fp16 operands are exact in the f32 accumulator, so a second pass over the
# fp16 rounding residual restores the weight to ~22 mantissa bits and the
# route's intrinsic error drops under the bf16 output floor (bf16-ULP flip
# rate 1.1% vs the unsplit 10.4%; verified against an exact f64 dot). On
# natural self-generated prompts the split route's delta vs the four-quant
# incumbent measures mean 4.8-6.7e-5 and max <= 1.4e-3 (top-1 100%), 20x
# under the envelope, at +7.5% leaf cost on the affected quants only. The
# originally routed four quants keep the single-pass path. The definitive
# 162-row production-referenced gate on this backend's prompts is owed as
# the confirming measurement.

# Dense raw-IQ decode and verifier owners (2026-09-21). register_gfx1151_kernels
# aliases the whole gfx1100 key space into this backend, so local32_gemv and its
# rows sibling below are the *same kernel objects* gfx1100 routes - the
# arithmetic is identical by construction, not merely similar. The routing
# tables are not aliased, and this package declared only the prefill policy, so
# every dense raw-IQ projection on gfx1151 kept the strict per-row GEMV: a zbook
# kernel census measured 61.64 ms/token, 53.42% of UD-Q4_K_M's 115.39 ms pure
# decode, in gguf_iq_dense_strict_kernel at 130.78 launches/token and
# 442.1 us/launch on the dominant template, against zero launches for plain
# Q4_K_M (benchmarks/results/
# 2026-09-21-zbook-ud-vs-plain-q4km-decode-attribution.json).
#
# These two tables mirror gfx1100's exactly, and
# tests/test_unit_gfx1151_iq_dense_policy_parity.py holds them there. The pins
# below are not an optimization knob: they are the per-artifact admission
# gfx1100's tokenized category suite required, and because the owner kernels are
# shared the requirement transfers by construction. Do not drop a pin to gain a
# route.
GGUF_IQ_DENSE_DECODE_POLICY = {
    quant: {"variant": "local32_gemv_bf16_bf16_out"}
    for quant in ("gguf_iq4_xs", "gguf_iq4_nl", "gguf_iq3_s", "gguf_iq3_xxs",
                  "gguf_iq2_s", "gguf_iq2_xs")
}
# Verifier sibling at rows 2-4. Each row's output is bit-identical to the
# rows==1 owner's output for that row, so it adds no arithmetic of its own; it
# keeps the same per-slot strict pin as the decode policy below.
GGUF_IQ_DENSE_VERIFY_POLICY = {
    quant: {
        "min_rows": 2,
        "max_rows": 4,
        "variant": "local32_rows_gemv_bf16_bf16_out",
    }
    for quant in ("gguf_iq4_xs", "gguf_iq4_nl", "gguf_iq3_s", "gguf_iq3_xxs",
                  "gguf_iq2_s", "gguf_iq2_xs")
}
# Per-artifact strict pins, keyed by (file_type, artifact preset). UD-Q4_K_M's
# layers.0.ffn_up is its only Q3_K ffn_up, and it is the slot whose W4A16 hi+lo
# split reached a 5.3e-2 max-row KL against the incumbent on natural tokenized
# prompts (one position of the 18-prompt category suite) - above the 5e-2
# pooled ceiling - so it keeps the strict per-row GEMV in prefill. The decode
# pins keep the same artifact's IQ3_S population (0.82 ms/token, 4 launches) on
# the strict owner: unpinned the family's accumulation-order tail measured
# 5.19e-2 max, and with three of the four slots pinned, 1.785e-1.
GGUF_IQ_DENSE_PREFILL_STRICT_SLOTS = {
    ("MOSTLY_Q4_K_M", "gguf_ud_q4_k_m"): ("layers.0.ffn_up",),
}
GGUF_IQ_DENSE_DECODE_STRICT_SLOTS = {
    ("MOSTLY_Q4_K_M", "gguf_ud_q4_k_m"): (
        "layers.11.ffn_gate", "layers.14.ffn_down", "layers.15.ffn_down",
        "layers.17.ffn_down"),
}

GGUF_Q6_DENSE_INTEGER_MMQ_PREFILL_POLICY = {
    "gguf_q6_k_t16_qmicro_planar_v1": {
        "min_rows": 17,
        "max_rows": 48,
        "shapes": frozenset({(17_408, 5_120), (5_120, 1_024)}),
        "variant": "t16_q8_1_planar_integer_mmq64x64_bf16_bf16_out",
    }
}

# W1 candidate: C8/R32 packed verification emits mixed physical R20/R24/R32
# subshapes, so all three must share the candidate transaction. This table is
# inert unless the explicit outer logical-width context is active.
GGUF_T16_TARGET_VERIFIER_WIDE_Q6_SHARED4_VARIANTS = {
    ("gguf_q4_k_t16_v1", rows, in_features, out_features): (
        "t16_wmma_prefill_shared_b2w2_bf16_bf16_out"
    )
    for rows in (20, 24, 32)
    for in_features, out_features in ((5_120, 1_024), (17_408, 5_120))
} | {
    ("gguf_q6_k_t16_v1", rows, 5_120, 10_240): (
        "t16_wmma_prefill_shared4_bf16_bf16_out"
    )
    for rows in (20, 24, 32)
} | {
    ("gguf_q6_k_t16_qmicro_planar_v1", rows, in_features, out_features): (
        "t16_wmma_prefill_shared4_bf16_bf16_out"
    )
    for rows in (20, 24, 32)
    for in_features, out_features in ((5_120, 1_024), (17_408, 5_120))
}
# Profile-qualified T2 production owner: use per-row-direct-equivalent Q4
# rowtiles for C2/K3 R8 and bounded C3/K1-K3 R6/R9/R12 physical targets on
# actual shapes. Narrow K5120/N1024 is excluded because the historical broad
# native-verify divergence localized there; strict small-M/shared-B WMMA
# remains the manifest fallback.
GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ROWS = frozenset({6, 8, 9, 12})
# Successor C6/K1 screen: one exact two-wave R12 gate/up owner replaces the
# R8+R4 rowtile chain. R16 regresses and deliberately remains absent.
GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_PAIR_VARIANTS = {
    (12, 5_120, 17_408): "dense_dual_wmma_smallm_bf16_bf16_out",
}
GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_SHAPES = frozenset(
    {
        (5_120, 6_144),
        (5_120, 10_240),
        (5_120, 12_288),
        (5_120, 17_408),
        (6_144, 5_120),
        (17_408, 5_120),
    }
)
GGUF_T16_NATIVE_ROWTILE_MAX_ROWS_BY_QUANT = {
    # Standard and planar Q6 col8 rowtiles are exact and qualified through R8.
    "gguf_q6_k_t16_v1": 8,
    "gguf_q6_k_t16_qmicro_planar_v1": 8,
    # Q5: the 27B ssm_out/ffn_down/qkv/v shapes rowtile to c8; the narrow
    # 0.8B SSM-out shape keeps cap 4 so its measured direct leaf wins at c5-c8.
    "gguf_q5_k_t16_v1": {
        "default": 4,
        "shapes": {
            (6_144, 5_120): 8,  # 27B ssm_out
            (17_408, 5_120): 8,  # 27B ffn_down
            (5_120, 10_240): 8,  # 27B attn_qkv
            (5_120, 1_024): 8,  # 27B attn_v
        },
    },
}
# The narrow 0.8B SSM-out shape wins with the direct leaf at c5-c8; QKV and
# bulk rows retain the independently measured WMMA route.
GGUF_T16_NATIVE_DIRECT_SHAPES_BY_QUANT = {
    "gguf_q5_k_t16_v1": frozenset({(2_048, 1_024)}),
}
_GGUF_Q5_T16_ROWTILE_COL8_SHAPES = frozenset(
    {
        (6_144, 5_120),
        (17_408, 5_120),
        (5_120, 10_240),
    }
)


# Qwen3.8-27B P5 exact serial-c1 output subdivisions. Full-attention K/V
# K5120/N1024 Q4 projections use four-column ownership after the actual-weight
# pool improves 1.03169x with 14/15 wins. Recurrent-output Q5 uses two
# eight-column workgroups after all five actual layers improve. Native rows/MTP
# and policy misses retain their independently qualified owners.
GGUF_T16_C1_VARIANTS_BY_QUANT_SHAPE = {
    "gguf_q4_k_t16_v1": {
        (5_120, 1_024): "dense_single_col4_bf16_bf16_out",
    },
    # E4a (UD-GFX1151-OPTIMIZE2): rows=1 Q5_T16 decode shapes routed to the
    # tile8 owner. Per-shape screen (bit-exact vs direct and the gguf_quant_gemv
    # reference at rows=1): ffn_up 1.16x, ffn_down 1.28x, attn_gate 1.36x,
    # attn_qkv 1.39x, attn_q 1.18x, ssm_out 1.56x. attn_v (5_120, 1_024) is
    # deliberately absent: 0.99x, no win over the direct owner.
    "gguf_q5_k_t16_v1": {
        (5_120, 6_144): "t16_gemv_decode_tile8_bf16_bf16_out",
        (5_120, 10_240): "t16_gemv_decode_tile8_bf16_bf16_out",
        (5_120, 12_288): "t16_gemv_decode_tile8_bf16_bf16_out",
        (5_120, 17_408): "t16_gemv_decode_tile8_bf16_bf16_out",
        (6_144, 5_120): "t16_gemv_decode_tile8_bf16_bf16_out",
        (17_408, 5_120): "t16_gemv_decode_tile8_bf16_bf16_out",
    },
}
# Qwen3.8-27B P2: use byte-neutral planar-qmicro Q6 where architecture-local
# exactness and speed gates retain it; slot exclusions below keep one alternate
# layout only where gfx1151 measurements reject planar ownership.
GGUF_DENSE_Q6_T16_QMICRO_PLANAR = True
# The K5120/N10240 recurrent QKV shape is 8.72% slower under planar c1 on
# gfx1151 (0/11 paired wins). Keep exactly one standard-T16 resident for those
# 24 tensors while planar remains the sole owner for down, narrow V, and root.
GGUF_DENSE_Q6_T16_QMICRO_PLANAR_EXCLUDED_SLOTS = ("attn_qkv",)
# Qwen3.8-27B P4: four waves preserve the exact standard-Q6 48x64 sequence
# while sharing one decoded 48x256 slab. The rows96-536 screen is bit-exact
# and positive at every point; shape misses and rows<96 retain 16x16.
GGUF_Q6_PREFILL_SHARED3R1_MIN_ROWS = 33
GGUF_Q6_PREFILL_SHARED3R1_MAX_ROWS = 48
GGUF_Q6_STANDARD_PREFILL_SHARED3R1_SHAPES = frozenset({(5_120, 10_240)})
GGUF_Q6_STANDARD_PREFILL_SHARED6R1_MIN_ROWS = 49
GGUF_Q6_STANDARD_PREFILL_SHARED6R1_MAX_ROWS = 96
GGUF_Q6_PLANAR_PREFILL_SHARED3R1_SHAPES = frozenset(
    {(5_120, 1_024), (17_408, 5_120)}
)
GGUF_Q6_STANDARD_PREFILL_SHARED4_MIN_ROWS = 96
GGUF_Q6_STANDARD_PREFILL_SHARED4_SHAPES = frozenset({(5_120, 10_240)})
GGUF_Q6_STANDARD_PREFILL_SHARED8R3_MIN_ROWS = 256
GGUF_Q6_STANDARD_PREFILL_SHARED8R3_MAX_ROWS = 384
GGUF_Q6_STANDARD_PREFILL_SHARED8R3_HIGH_MIN_ROWS = 385
GGUF_Q6_STANDARD_PREFILL_SHARED8R3_HIGH_MAX_ROWS = 1_024
# The planar sibling uses the same exact shared schedule. The six-shape
# rows256/384/480/536 screen admits shared4 from row256; rows145-255 retain
# plain, and the periodic rows81-144 bands above select separately.
GGUF_Q6_PLANAR_PREFILL_SHARED4R9_ROWS = 536
GGUF_Q6_PLANAR_PREFILL_SHARED4R9_SHAPE = (17_408, 5_120)
GGUF_Q6_PLANAR_PREFILL_SHARED4R6_MIN_ROWS = 288
GGUF_Q6_PLANAR_PREFILL_SHARED4R6_MAX_ROWS_BY_SHAPE = {
    (5_120, 1_024): 536,
}
# The 2026-09-13 wide-down row screen (rows 256-4096, bit-exact to the plain
# parent at every screened row) supersedes the rows256/384/480/536 screen for
# (17_408, 5_120): the 4x4 shared tile owns rows288-1024 and the 48-column
# four-wave tile owns rows1025 and above. The 32-row shared4 owner that shape
# previously used from row256 is 1.4-1.7x slower inside the band and 2.4x
# slower above it. Evidence:
# scripts/qwen38_q6_planar_prefill_large_row_screen.py and
# benchmarks/results/2026-09-13-q6-planar-prefill-large-row-screen/artifact.json
GGUF_Q6_PLANAR_PREFILL_SHARED4R4_MIN_ROWS = 288
GGUF_Q6_PLANAR_PREFILL_SHARED4R4_MAX_ROWS = 1_024
GGUF_Q6_PLANAR_PREFILL_SHARED4R4_ROWS = 256
GGUF_Q6_PLANAR_PREFILL_SHARED4R4_SHAPES = frozenset({(17_408, 5_120)})
GGUF_Q6_PLANAR_PREFILL_SHARED4_GFX1100_MIN_ROWS = 1_025
GGUF_Q6_PLANAR_PREFILL_SHARED4_GFX1100_SHAPES = frozenset({(17_408, 5_120)})
GGUF_Q6_PLANAR_PREFILL_SHARED4R3_ROWS = 256
GGUF_Q6_PLANAR_PREFILL_SHARED4R3_SHAPES = frozenset(
    {(5_120, 1_024), (17_408, 5_120)}
)
GGUF_Q6_PLANAR_PREFILL_SHARED4_MIN_ROWS = 256
GGUF_Q6_PLANAR_PREFILL_SHARED4_SHAPES = GGUF_Q4_T16_DENSE_LOWM_SHAPES
GGUF_DENSE_T16_F16_ROCBLAS_PREFILL_POLICIES = {
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): True,
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_S"): True,
}
# P4 retains changed-arithmetic source-F16 only for the sole-Q5T16 recurrent
# output. Q6 and Q4 complete pp512 screens lose wall and/or memory, so their
# exact T16/WMMA owners remain production. The Q5 octet producer consumes one
# bounded dead-input cast and temporary tile; sole T16 residency is unchanged.
# Q4_K_S shares the same 48 byte-identical Q5T16 K6144/N5120 recurrent outputs
# and therefore admits the same source-F16 route (MOSTLY_Q4_K_S added 2026-08-17);
# the K_S qualification had not extended this K_M-derived prefill policy, leaving
# all 60 Q5 prefill tensors on the single-wave exact WMMA owner.
GGUF_Q5_T16_F16_ROCBLAS_PREFILL_POLICIES = {
    (6_144, 5_120): {512: 1_280, 1_024: 1_280, 4_096: 1_024},
}
GGUF_T16_F16_ROCBLAS_MAX_ROWS_BY_QUANT_SHAPE = {}
GGUF_T16_F16_ROCBLAS_VARIANT_POLICIES = {
    "gguf_q5_k_t16_v1": {
        (6_144, 5_120): {
            (512, 4_096): "f16_rocblas_t16_octet_bf16_bf16_out",
        },
    },
}
# Physical-C8 Q6T16 lm-head now has a native rows-8 rowtile owner, so c8 runs
# as one launch instead of the previous 5+3 partition. rows > 8 still chunk.
GGUF_Q6_LM_HEAD_MAX_CHUNK = 8
# Generation-2 gfx1151 Qwen3.8 qualification (2026-08-22): physical widths
# 1..8 pass the numerical-profile-backed c13 lifecycle with cancellation,
# refill, compaction, no scalar fallback, and exact memory recovery.
GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8)
# Exact fused pointer-table copies contract thousands of per-plane D2D state
# transfers when the production owner switches physical groups. Qualified on
# c13 lifecycle plus counterbalanced c17/c32 serving and marked c17 profiling.
GGUF_FUSED_LINEAR_STATE_TRANSFER = True
# The resident batch owner allocates one contiguous Conv/GDN state slab across
# all scheduler slots. Packed rows may index that canonical slab directly,
# avoiding the secondary packed-state round trip at physical-group boundaries.
GGUF_DIRECT_RESIDENT_LINEAR_STATE = True
# Same-length full-prompt rows may enter one native prefill call. This is scoped
# independently from decode widths and falls back before mutation on misses.
GGUF_C2_PACKED_PREFILL_MAX_ROWS = 8
# Long packed-prefill rounds sample only each slot's final tail. Intermediate
# target-hidden rows and host hidden-seed capture remain intact; C5-C8 512-row
# performance and lifecycle evidence own this gfx1151-only default.
GGUF_PACKED_PREFILL_FINAL_OUTPUT_MASK = True
# SPECDEC2 S3 admits construction of the dense NextN c1 staged adapter on
# gfx1151. S4 additionally admits the physical c2/c4 adapter; arithmetic/default
# promotion remains independently gated by each phase's correctness and
# complete-wall packet. These capabilities expose adapters and AR fallback only.
GGUF_SPECDEC2_MTP2_C1 = True
# Production MTP admission is intentionally non-monotonic. C1-C4 retain their
# certified K1-K3 cells; C5-C8 stay on one full-width AR step, so a scalar
# maximum cannot silently broaden the measured-losing widths.
# The wide C8-K3 cell admitted on 2026-09-05 is withdrawn. This backend's
# prompt-streaming policy admits widths (1,2,3,4), so a pending set of eight rows
# is refused (`physical_streaming_category_rejected`) and the provider opens
# without prompt state, while the route still ran its cycles. Measured on the
# instrument that qualified the cell (one synchronized 8-row batch per cell, 10
# prompts, 24-token horizon, same-run true-AR baseline) it is 46.88 tok/s against
# 50.10 true AR (0.936x) with 0 of 10 cells reproducing the AR output, against
# the retained 1.0015x with 40 of 40 cells AR-equal. gfx1100's prompt-streaming
# policy admits width 8, so its own C8-K3 cell is unaffected.
GGUF_SPECDEC2_MTP2_PHYSICAL = True
GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS: dict[
    str, tuple[tuple[int, int], ...]
] = {
    "production": (
        *((width, depth) for width in range(1, 5) for depth in range(1, 4)),
    )
}
# M5 whole-batch routing (scaling campaign, 2026-08-31): measured at the
# current head, MTP sub-group interleaving reaches only 0.74-0.80x of own AR
# at physical widths 5-8 (C5-C8 28.0/32.7/33.1/35.5 vs AR 36.1/40.8/43.7/47.8
# tok/s), so a due batch wider than the production bound must fall through to
# one full-batch AR decode instead of chaining MTP sub-groups. Widths <= 4
# keep the certified MTP cycle (1.19-1.56x AR).
GGUF_SPECDEC2_MTP2_BATCH_ROUTE_ABOVE_REQUESTS: dict[str, int] = {"production": 4}
# E1a/E7 admit the exact shifted prompt-streaming path for measured Qwen3.8
# standard-Q4 production physical-C2/C3 groups. Scaling-campaign screens
# (2026-08-31): width 1 engages the same exact path (C1 +24.1% screen,
# IDs/acceptance/route/budget identical); width 4 now repeats that result at
# the current head (C4 34.182->35.618, gate 34.596 PASS, every category >=
# own AR, per-cell IDs exact, acceptance 92/121 vs 93/120 baseline). The
# historical 628/796->624/800 C4 drift did not reproduce; the binding frozen
# contract is per-cell output self-exactness (see the M4 decision entry),
# with acceptance trajectory an observational diagnostic. Strict C1,
# other models/quants/profiles, and peers retain replay.
GGUF_SPECDEC2_PHYSICAL_PROMPT_STREAMING_POLICIES = {
    (QWEN35_DENSE_H5120_GEOMETRY, None, "production"): (1, 2, 3, 4),
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M", "production"): (1, 2, 3, 4),
    (QWEN35_MOE_H2048_E256_GEOMETRY, None, "production"): (1, 2),
}
# E1b reuses the exact Q6 F32 small-B rowtile only for physical proposal-head
# dimensions/rows2-4 that have actual Qwen3.8 evidence. Wide request groups
# already lower proposals into rows4 subgroups, so rows5-8 are not runtime keys.
# NextN adapts the source
# model to a one-block geometry and does not carry its file-type label, so this
# key uses the immutable H/N head shape; primitive resolution still requires
# Q6 T16. The direct producer remains the strict policy-miss fallback. The
# rowtile wrapper selects its col8 body at rows3/4 and 16-column body at rows2.
GGUF_SPECDEC2_PROPOSAL_LM_HEAD_ROWTILE_POLICIES = frozenset(
    {
        (5120, 248320, 2),
        (5120, 248320, 3),
        (5120, 248320, 4),
        # Scaling-campaign M2: the single wide proposal carries rows5-8.
        # The planar T16 rowtile body is bit-identical to the direct parent
        # across rows 2-8 (tests/test_live_qwen38_nextn_proposal_head_rowtile.py);
        # without these keys the wide proposal head fell to the per-row
        # direct gemv (measured 36.7 ms/launch vs ~4.8 ms chunked sweeps).
        (5120, 248320, 5),
        (5120, 248320, 6),
        (5120, 248320, 7),
        (5120, 248320, 8),
    }
)
GGUF_SPECDEC2_NATIVE_TARGET_GRAPH_MAX_CONTEXT = 65544
GGUF_SPECDEC2_NATIVE_TARGET_MAX_CONTEXT = 65544
# F4's clean all-candidate, all-workload production gate selects fair:256 at
# +5.90% exact mixed-load SLO goodput over fair:128. Scope the default to the
# measured Q4_K_M generator registry entry; other quants/backends retain their
# prior engine-loop defaults until independently gated.
GGUF_Q4_K_M_PREFILL_DECODE_POLICY = "fair"
GGUF_Q4_K_M_MAX_PREFILL_CHUNK_TOKENS = 256
# Bound fair scheduling to two consecutive 256-token chunks so one p512 row
# becomes decode-ready per interruption instead of paying two partial-width
# decode ticks. The package selector keeps other quants/backends at one chunk.
GGUF_Q4_K_M_FAIR_PREFILL_BURST_CHUNKS = 2
# F3/F2 prove true physical-c8 GGUF AR and exact live ownership. The OpenAI
# coalescer may therefore submit eight plain-AR requests to this registry entry;
# speculative MTP keeps its separately certified four-request cap.
GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS = 8
# Clean LCP-M2 512/1K/4K full-state and balanced-wall gates admit stream-ordered
# device metadata through 4K. Explicit opt-in remains available for diagnosis;
# the 128K one-queue escalation still enters the low-power GPU-active state.
GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS = 4096
# Clean LCP-1 primitive/full-state, same-stream trace, and fresh-process wall
# gates admit the exact 32-token shared-memory convolution schedule on gfx1151.
GGUF_LINEAR_ATTN_CONV_PREFILL_AUTO_MODE = "tile32x128"
# GPF-3A shared-X is byte-exact and wins at 512/1K/4K, but the controlled
# repeated-128K split reproduces the gfx1151 queue no-progress state only when
# this route is enabled. Keep the explicit diagnostic, and use the established
# baseline body automatically until shared-X passes the long-context gate.
#
# R12 ladder step 1: the paired multi-wave LDS-staged bodies for BOTH
# selected-expert legs (`pair32`) are raw-bit identical to the compact baseline
# and clear the gate that shared-X failed. Three independent conservative-set
# 128K warmup+3 processes passed 12/12 (319.25/319.21/319.30 tok/s, stdev
# <= 0.11, finite logits, IDs [9707,9707,9707]) and three independent
# production-shaped processes passed 12/12 at 620.61/619.99/619.57 tok/s, above
# the 575.180-580.626 baseline-only figure on the same workload. The compact
# `baseline` body stays registered as the release-safety rollback and `shared_x`
# stays blocked. See docs/REFACTOR.md and
# worklog/entries/20261003T112247.855338Z-lhl-gfx1151-prefill-r12-design-basis-c1eb0a.md.
GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE = "pair32"
# LCP-4's exact router primitive and full-model gates admit the 256-thread
# reduction geometry for BF16-hidden/F32-weight GGUF router logits on gfx1151.
GGUF_ROUTER_F32_BF16_HIDDEN_THREADS = 256
# Post-LCP-4B profile and full-state gates admit 128 threads for bulk-prefill
# top-k selection. Decode keeps its independently selected 256-thread launch.
GGUF_PREFILL_ROUTER_SELECT_THREADS = 128
# Clean LCP-3 exactness plus balanced 512/4K wall admits four-wave activation
# sharing for covered dense Q8T16 WMMA prefill shapes on gfx1151. Two-wave stays
# available as the first rollback schedule during its release window.
GGUF_Q8_T16_PREFILL_FOUR_WAVE = True
GGUF_Q8_T16_PREFILL_TWO_WAVE = True
# D08-X8: the 18 alpha/beta pairs are same-shape Q8T16 N16 projections.
# One two-wave block shares each activation tile and preserves singleton order.
GGUF_Q8_T16_DUAL_WMMA_PREFILL = True
GGUF_Q8_T16_DUAL_WMMA_PREFILL_SHAPES = frozenset({(512, 1_024, 16, 16)})
GGUF_Q8_T16_DUAL_WMMA_PREFILL_POLICIES = {
    (QWEN35_DENSE_H1024_GEOMETRY, "MOSTLY_Q4_K_M"): frozenset({512}),
    (QWEN35_DENSE_H1024_GEOMETRY, "MOSTLY_Q8_0"): frozenset({512}),
}
# Same-commit production-protocol 128K A/B rejects predecessor two-wave
# (382.041 vs 392.219 tok/s), so LCP-3 conservatively inherits its 64K ceiling.
GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS = 65536
# WPF-1/WPF-1T are W7900 raw-Q5/Q6 candidates. gfx1151 retains its
# independently admitted Q4_K_M/T16 matrix schedules and must not inherit their
# rowbatch or output-column selectors.
GGUF_RAW_K_PREFILL_ROWBATCH_SUPPORTED = False
# Qwen4Exp independently qualifies the exact F32-input raw-Q8 rowbatch family;
# this does not admit the W7900 Q5/Q6 policies above.
GGUF_RAW_Q8_F32_PREFILL_ROWBATCH_SUPPORTED = True
GGUF_RAW_Q8_F32_PREFILL_COLTILE_SUPPORTED = True
# Complete native-owner timing is positive only at rows=4096. The p1024
# owner is slower after scope admission/dispatch bookkeeping and remains on the
# parent. Runtime consumes registry-axis keys; peer backends expose no admission.
GGUF_DENSE_WIDE_F16_PAIR_POLICIES = {
    ("linear", "gguf_q8_0", "dense_wide256_f32_f32_out"): {
        "rows": frozenset({4096}),
        "layers": frozenset(range(13, 48)),
        "pairs": {
            ("attn_qkv", "attn_gate"): (2560, 10240, 6144),
            ("shared_gate", "shared_up"): (2560, 640, 640),
        },
    },
}
GGUF_DENSE_WIDE_F16_TAIL_POLICIES = {
    ("linear", "gguf_q8_0", "dense_wide256_f16in_f32_f32_out"): {
        "rows": (512, 4096),
        "shape": (10240, 320),
        "variant": "dense_wide256_padded_f16in_f32_f32_out",
    },
}


GGUF_RAW_K_PREFILL_ROWBATCH = 0
GGUF_RAW_K_PREFILL_COLTILE_SUPPORTED = False
GGUF_RAW_K_PREFILL_COLTILE2_SHAPES = frozenset()
GGUF_RAW_K_PREFILL_VARIANT = "rowbatch"
# WPF-H5I exact-F32 ordered raw-K ownership is W7900-only until an
# independently measured gfx1151 package policy exists.
GGUF_Q6_F32_ORDERED_PREFILL = False
GGUF_Q6_F32_ORDERED_PREFILL_POLICY = {}
GGUF_F32_ORDERED_PREFILL_QUANTS = frozenset()
GGUF_F32_ORDERED_PREFILL_POLICIES = {}
# SH9-D1 independently admits LCP-2B's exact routing-independent compact-WMMA
# row bound on gfx1151 through 4,096 selected rows. Larger shapes and explicit
# opt-out retain the scalar total-row read.
GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS = 4096
# SH7-A1 retains LCP-D2's registered prepare-plus-coalesced split reducer on
# gfx1151 from 32K onward after exact primitive/semantic admission and measured
# 32K/64K wall wins. Shorter contexts and explicit opt-out keep the serial path.
GGUF_PAGED_ATTN_PARALLEL_REDUCE = True
GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT = 32768
# P5 groups Qwen3.8 dense-27B's 24 query heads by its four K/V heads from
# 4K onward. The 4K actual-shape gate is BF16-bit exact and 4.70x faster than
# the generic split producer; shorter contexts and peer packages keep fallback.
GGUF_PAGED_ATTN_GROUPED_GQA_MIN_CONTEXTS = {
    (5_120, 64, 24, 4, 256, 256, 256): 4_096,
}
# SH10-A1 reuses the existing exact fixed256 compact-row leaf for private-c1
# BF16 attention below the split threshold. The rows=1 actual-shape screen is
# F32 byte-exact and 1.56-1.65x faster at contexts 513/576/640. Context 1024+
# keeps the established direct/split routes.
GGUF_SHORT_C1_BATCH_ATTN_MAX_CONTEXT = 1023
# The c1 short-batch leaf block width. 256 is the exact fixed256 default.
# 1024 runs the same body at a wider block width (split value reduction +
# different warp reduction tree -> T2 non-exact production probe). The 1024
# variant passed the calibrated execution-profile c1 threads gate (full
# mtp-bench category suite, teacher-forced KL/top-1 envelope, 3 repeats) and is
# retained as the gfx1151 default; the exact 256-thread leaf stays registered
# as the strict fallback. HIPENGINE_GGUF_SHORT_C1_ATTN_THREADS overrides.
GGUF_SHORT_C1_BATCH_ATTN_THREADS = 1024
# D08-D4 independently qualifies the existing generic split-K3 plus fused-gate
# chain for Qwen3.5-0.8B's private-c1 8Q/2KV/D256 graph cap. The exact model/
# attention shape and measured 514-641 window preserve fixed256 at the 513 warm
# boundary and for every unqualified model, context, backend, and batch route.
GGUF_SHORT_C1_SPLIT_ATTN_POLICIES = {
    (1_024, 24, 8, 2, 256, 256, 256): (514, 641),
}


__all__ = [
    "GGUF_AOTRITON_HEAD_MAJOR_KV",
    "GGUF_AOTRITON_PREFILL",
    "GGUF_C2_PACKED_PREFILL_MAX_ROWS",
    "GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS",
    "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS",
    "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS_BY_KV_STORAGE",
    "GGUF_DECODE_GRAPH_SUBMISSION_POLICIES",
    "GGUF_DENSE_BF16_WMMA_BULK_PREFILL",
    "GGUF_DENSE_BF16_WMMA_BULK_PREFILL_SHAPES",
    "GGUF_DENSE_DOWN_RESIDUAL_DECODE_POLICIES",
    "GGUF_DENSE_F32_ALPHA_BETA_CONV_DECODE_SHAPES",
    "GGUF_DENSE_F32_ALPHA_BETA_PAIR_DECODE_SHAPES",
    "GGUF_DENSE_PAIR_SILU_DECODE_POLICIES",
    "GGUF_DENSE_PAIR_SILU_NATIVE_DECODE_POLICIES",
    "GGUF_DENSE_PREFILL_SCRATCH_ROW_CAP_POLICIES",
    "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP",
    "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES",
    "GGUF_DENSE_Q4_T16",
    "GGUF_DENSE_Q4_T16_ATTN_Q_08B",
    "GGUF_DENSE_Q5_T16_H5120",
    "GGUF_DENSE_Q5_T16_QKV",
    "GGUF_DENSE_Q5_T16_SSM_OUT",
    "GGUF_DENSE_Q5_T16_SSM_OUT_08B",
    "GGUF_DENSE_Q6_T16_QMICRO_PLANAR",
    "GGUF_DENSE_Q6_T16_QMICRO_PLANAR_EXCLUDED_SLOTS",
    "GGUF_DENSE_T16_F16_ROCBLAS_PREFILL_POLICIES",
    "GGUF_DENSE_WIDE_F16_PAIR_POLICIES",
    "GGUF_DENSE_WIDE_F16_TAIL_POLICIES",
    "GGUF_DIRECT_RESIDENT_LINEAR_STATE",
    "GGUF_F32_ORDERED_PREFILL_POLICIES",
    "GGUF_F32_ORDERED_PREFILL_QUANTS",
    "GGUF_FP16_RECURRENT_STATE_DEFAULT_FILE_TYPES",
    "GGUF_FULL_ATTN_QK_POSTPROCESS_DECODE_POLICIES",
    "GGUF_FUSED_LINEAR_STATE_TRANSFER",
    "GGUF_GDN_INDEXED_SINGLETON_DECODE",
    "GGUF_GDN_PREFILL_AUTO_MODE",
    "GGUF_GDN_PREFILL_AUTO_MODES_BY_QUANT_SHAPE",
    "GGUF_GDN_PREFILL_COMPACT_PEER_CHUNK_ROWS",
    "GGUF_GDN_PREFILL_EXACT_MODE",
    "GGUF_HOST_TOKEN_EMBEDDING_C1",
    "GGUF_HOST_TOKEN_EMBEDDING_C1_GGML_TYPES",
    "GGUF_IQ_DENSE_DECODE_POLICY",
    "GGUF_IQ_DENSE_DECODE_STRICT_SLOTS",
    "GGUF_IQ_DENSE_PREFILL_POLICY",
    "GGUF_IQ_DENSE_PREFILL_STRICT_SLOTS",
    "GGUF_IQ_DENSE_VERIFY_POLICY",
    "GGUF_LINEAR_ATTN_CONV_PREFILL_AUTO_MODE",
    "GGUF_LINEAR_RESIDUAL_MAX_ROWS_BY_QUANT",
    "GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1",
    "GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1_COPY",
    "GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1_GGML_TYPES",
    "GGUF_NARROW_KV_PAIR_DECODE_SHAPES",
    "GGUF_NORM_RESIDUAL_DECODE_POLICIES",
    "GGUF_PACKED_DECODE_GRAPH_MIN_REPLAY_STEPS_BY_POLICY",
    "GGUF_PACKED_PREFILL_FINAL_OUTPUT_MASK",
    "GGUF_PAGED_ATTN_GROUPED_GQA_MIN_CONTEXTS",
    "GGUF_PAGED_ATTN_PARALLEL_REDUCE",
    "GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT",
    "GGUF_PREFILL_CHUNK_SIZES_BY_GEOMETRY",
    "GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS",
    "GGUF_PREFILL_ROUTER_SELECT_THREADS",
    "GGUF_PREFILL_SCRATCH_ARENA_GROUPING",
    "GGUF_PREFILL_SCRATCH_LIVENESS_ALIAS",
    "GGUF_PREFILL_SCRATCH_LIVENESS_MIN_ROWS",
    "GGUF_PRIVATE_C1_DECODE_SCRATCH_ARENA_POLICIES",
    "GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA",
    "GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA_POLICIES",
    "GGUF_Q4_DUAL_SILU_PREFILL_ROW48_MAX_ROWS",
    "GGUF_Q4_K_M_FAIR_PREFILL_BURST_CHUNKS",
    "GGUF_Q4_K_M_MAX_PREFILL_CHUNK_TOKENS",
    "GGUF_Q4_K_M_PREFILL_DECODE_POLICY",
    "GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS",
    "GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL",
    "GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL_POLICIES",
    "GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL_SHAPES",
    "GGUF_Q4_PACK8_WMMA_BULK_PREFILL",
    "GGUF_Q4_PACK8_WMMA_BULK_PREFILL_SHAPES",
    "GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS",
    "GGUF_Q4_T16_DENSE_LOWM_SHAPES",
    "GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS",
    "GGUF_Q4_T16_DENSE_LOWVGPR128_SHAPES",
    "GGUF_Q4_T16_DENSE_LOWVGPR144_MAX_ROWS",
    "GGUF_Q4_T16_DENSE_LOWVGPR48_128_SHAPES",
    "GGUF_Q4_T16_DENSE_LOWVGPR48_144_SHAPES",
    "GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS",
    "GGUF_Q4_T16_DENSE_LOWVGPR48_SHAPES",
    "GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS",
    "GGUF_Q4_T16_DENSE_LOWVGPR64_SHAPES",
    "GGUF_Q4_T16_DENSE_LOWVGPR80_SHAPES",
    "GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS",
    "GGUF_Q4_T16_DENSE_LOWVGPR96_SHAPES",
    "GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS",
    "GGUF_Q4_T16_DENSE_PLAIN128_SHAPES",
    "GGUF_Q4_T16_DENSE_SHARED2W2_192_SHAPES",
    "GGUF_Q4_T16_DENSE_SHARED2W2_256_SHAPES",
    "GGUF_Q4_T16_DENSE_SHARED2W2_384_SHAPES",
    "GGUF_Q4_T16_DENSE_SHARED2W4_192_SHAPES",
    "GGUF_Q4_T16_DENSE_SHARED2_192_MAX_ROWS",
    "GGUF_Q4_T16_DENSE_SHARED2_256_MAX_ROWS",
    "GGUF_Q4_T16_DENSE_SHARED2_384_MAX_ROWS",
    "GGUF_Q4_T16_DENSE_SHARED3W8R3_384_MIN_ROWS",
    "GGUF_Q4_T16_DENSE_SHARED3W8R3_384_SHAPES",
    "GGUF_Q4_T16_PHYSICAL_SMALLM_MAX_ROWS",
    "GGUF_Q4_T16_PHYSICAL_SMALLM_ROWS",
    "GGUF_Q4_T16_PHYSICAL_SMALLM_SHAPES",
    "GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE",
    "GGUF_Q5_T16_DENSE_LOWM_SHAPES",
    "GGUF_Q5_T16_DENSE_LOWVGPR128_MAX_ROWS",
    "GGUF_Q5_T16_DENSE_LOWVGPR144_MAX_ROWS",
    "GGUF_Q5_T16_DENSE_LOWVGPR48_128_SHAPES",
    "GGUF_Q5_T16_DENSE_LOWVGPR48_MAX_ROWS",
    "GGUF_Q5_T16_DENSE_LOWVGPR48_SHAPES",
    "GGUF_Q5_T16_DENSE_LOWVGPR64_MAX_ROWS",
    "GGUF_Q5_T16_DENSE_LOWVGPR64_SHAPES",
    "GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS",
    "GGUF_Q5_T16_DENSE_LOWVGPR80_SHAPES",
    "GGUF_Q5_T16_DENSE_LOWVGPR96_MAX_ROWS",
    "GGUF_Q5_T16_DENSE_LOWVGPR96_SHAPES",
    "GGUF_Q5_T16_DENSE_LOWVGPR_MAX_ROWS",
    "GGUF_Q5_T16_DENSE_LOWVGPR_SHAPES",
    "GGUF_Q5_T16_DENSE_SHARED8R3_MAX_ROWS",
    "GGUF_Q5_T16_DENSE_SHARED8R3_MIN_ROWS",
    "GGUF_Q5_T16_DENSE_SHARED8R3_SHAPES",
    "GGUF_Q5_T16_F16_ROCBLAS_PREFILL_POLICIES",
    "GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q5_T16_SELECTED_QWEN_TILE8",
    "GGUF_Q6_DENSE_INTEGER_MMQ_PREFILL_POLICY",
    "GGUF_Q6_F32_ORDERED_PREFILL",
    "GGUF_Q6_F32_ORDERED_PREFILL_POLICY",
    "GGUF_Q6_LM_HEAD_MAX_CHUNK",
    "GGUF_Q6_PLANAR_LOWVGPR128_MAX_ROWS",
    "GGUF_Q6_PLANAR_LOWVGPR128_SHAPES",
    "GGUF_Q6_PLANAR_LOWVGPR144_MAX_ROWS",
    "GGUF_Q6_PLANAR_LOWVGPR48_128_SHAPES",
    "GGUF_Q6_PLANAR_LOWVGPR48_144_SHAPES",
    "GGUF_Q6_PLANAR_LOWVGPR80_SHAPES",
    "GGUF_Q6_PLANAR_LOWVGPR96_MAX_ROWS",
    "GGUF_Q6_PLANAR_LOWVGPR96_SHAPES",
    "GGUF_Q6_PLANAR_PREFILL_SHARED3R1_SHAPES",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R3_ROWS",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R3_SHAPES",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R4_MAX_ROWS",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R4_MIN_ROWS",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R4_ROWS",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R4_SHAPES",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R6_MAX_ROWS_BY_SHAPE",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R6_MIN_ROWS",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R9_ROWS",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R9_SHAPE",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4_GFX1100_MIN_ROWS",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4_GFX1100_SHAPES",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4_MIN_ROWS",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4_SHAPES",
    "GGUF_Q6_PLANAR_SHARED4_128_SHAPES",
    "GGUF_Q6_PLANAR_SHARED4_144_SHAPES",
    "GGUF_Q6_PREFILL_SHARED3R1_MAX_ROWS",
    "GGUF_Q6_PREFILL_SHARED3R1_MIN_ROWS",
    "GGUF_Q6_Q4_T16_MIXED_GRID_DECODE_SHAPES",
    "GGUF_Q6_STANDARD_PREFILL_SHARED3R1_SHAPES",
    "GGUF_Q6_STANDARD_PREFILL_SHARED4_MIN_ROWS",
    "GGUF_Q6_STANDARD_PREFILL_SHARED4_SHAPES",
    "GGUF_Q6_STANDARD_PREFILL_SHARED6R1_MAX_ROWS",
    "GGUF_Q6_STANDARD_PREFILL_SHARED6R1_MIN_ROWS",
    "GGUF_Q6_STANDARD_PREFILL_SHARED8R3_HIGH_MAX_ROWS",
    "GGUF_Q6_STANDARD_PREFILL_SHARED8R3_HIGH_MIN_ROWS",
    "GGUF_Q6_STANDARD_PREFILL_SHARED8R3_MAX_ROWS",
    "GGUF_Q6_STANDARD_PREFILL_SHARED8R3_MIN_ROWS",
    "GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS",
    "GGUF_Q8_T16_DECODE_ROWTILE_ALL",
    "GGUF_Q8_T16_DECODE_ROWTILE_MIN_ROWS",
    "GGUF_Q8_T16_DUAL_WMMA_PREFILL",
    "GGUF_Q8_T16_DUAL_WMMA_PREFILL_POLICIES",
    "GGUF_Q8_T16_DUAL_WMMA_PREFILL_SHAPES",
    "GGUF_Q8_T16_PREFILL_FOUR_WAVE",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS",
    "GGUF_Q8_T16_SINGLE_ROWTILE_VARIANTS_BY_SHAPE",
    "GGUF_RAW_K_PREFILL_COLTILE2_SHAPES",
    "GGUF_RAW_K_PREFILL_COLTILE_SUPPORTED",
    "GGUF_RAW_K_PREFILL_ROWBATCH",
    "GGUF_RAW_K_PREFILL_ROWBATCH_SUPPORTED",
    "GGUF_RAW_K_PREFILL_VARIANT",
    "GGUF_RAW_Q8_F32_PREFILL_COLTILE_SUPPORTED",
    "GGUF_RAW_Q8_F32_PREFILL_ROWBATCH_SUPPORTED",
    "GGUF_ROUTER_F32_BF16_HIDDEN_THREADS",
    "GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS",
    "GGUF_SHORT_C1_BATCH_ATTN_MAX_CONTEXT",
    "GGUF_SHORT_C1_BATCH_ATTN_THREADS",
    "GGUF_SHORT_C1_SPLIT_ATTN_POLICIES",
    "GGUF_SPECDEC2_MTP2_BATCH_ROUTE_ABOVE_REQUESTS",
    "GGUF_SPECDEC2_MTP2_C1",
    "GGUF_SPECDEC2_MTP2_PHYSICAL",
    "GGUF_SPECDEC2_MTP2_PHYSICAL_WIDTH_DEPTHS",
    "GGUF_SPECDEC2_NATIVE_TARGET_GRAPH_MAX_CONTEXT",
    "GGUF_SPECDEC2_NATIVE_TARGET_MAX_CONTEXT",
    "GGUF_SPECDEC2_PHYSICAL_PROMPT_STREAMING_POLICIES",
    "GGUF_SPECDEC2_PROPOSAL_LM_HEAD_ROWTILE_POLICIES",
    "GGUF_STAGED_LINEAR_ROWS_LONG_DEFAULT_FILE_TYPES",
    "GGUF_T16_C1_VARIANTS_BY_QUANT_SHAPE",
    "GGUF_T16_F16_ROCBLAS_MAX_ROWS_BY_QUANT_SHAPE",
    "GGUF_T16_F16_ROCBLAS_VARIANT_POLICIES",
    "GGUF_T16_NATIVE_DIRECT_SHAPES_BY_QUANT",
    "GGUF_T16_NATIVE_ROWTILE_MAX_ROWS_BY_QUANT",
    "GGUF_T16_NATIVE_ROWTILE_VARIANTS_BY_QUANT",
    "GGUF_T16_NATIVE_SPLIT_ROW_CHUNKS_BY_QUANT_SHAPE",
    "GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_PAIR_VARIANTS",
    "GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ROWS",
    "GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_SHAPES",
    "GGUF_T16_TARGET_VERIFIER_ROWTILE_CHUNK_ROWS_BY_QUANT",
    "GGUF_T16_TARGET_VERIFIER_ROWTILE_SHAPES_BY_QUANT",
    "GGUF_T16_TARGET_VERIFIER_TRUE_ROWTILE_VARIANTS",
    "GGUF_T16_TARGET_VERIFIER_WIDE_Q6_SHARED4_VARIANTS",
]
