"""gfx1100 / RDNA3 backend capabilities."""

from hipengine.kernels.policy import (
    QWEN35_DENSE_H5120_GEOMETRY,
    QWEN35_MOE_H2048_E256_GEOMETRY,
)

# Clean W7900 context/category gates retain the exact token4 score-parallel SWA
# decode default. The wider token8 screen failed the every-category h16 gate and
# was removed; other backends retain the separately registered baseline.
LAGUNA_SWA_DECODE_VARIANT = "swa_context_token4_exact_spans"
# WPF-3's exact adjacent-row SWA policy keeps wave32 below the measured
# crossover and selects qrow4 only for complete M128 tiles at position 256+.
# Complete M512 state is KL0 and both 512/1K same-weight gates are positive;
# explicit local128/wave32/qrow2/qrow4 variants remain registered rollbacks.
LAGUNA_SWA_PREFILL_VARIANT = "swa_context_rows_qrow4_m128_c256_exact_spans"
# WPF-H5M promotes exact source-qualified qrow4 loads after KL0 complete state,
# all 72 integrated role calls, and positive clean 512/1K/4K timing. Every
# explicit route plus shape, registration, and backend misses retain WPF-3.
LAGUNA_SWA_PREFILL_ROLE_VARIANTS = {
    "qrow4_m128_c256_exact": (
        "swa_context_rows_qrow4_sourcequal_exact_spans"
    ),
}
# WPF-H6N keeps exact fixed-512 global dense-initial source ownership. WPF-H6W
# promotes only late-start SWA global-score replay after complete state/topology
# and positive fixed/512/1K/4K gates. H6Z is a separate default-off capability
# for bounded late-start global qualification. H6A remains explicit complete
# rollback; starts0/128 and shape/metadata/registration/backend misses fail
# closed to it.
LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6A_ROLE_VARIANTS = {
    "global_m128_c4096_first_fill_exact": (
        "global_context_rows_dense_initial_fixed512_cached_exact_spans"
    ),
    "swa_qrow4_m128_c512_no_wrap_exact": (
        "swa_context_rows_qrow4_dense_initial_cached_exact_spans"
    ),
}
LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6W_ROLE_VARIANTS = {
    "global_m128_c4096_first_fill_exact": (
        "global_context_rows_dense_initial_fixed512_cached_exact_spans"
    ),
    "swa_qrow4_m128_c512_no_wrap_exact": (
        "swa_context_rows_qrow4_dense_initial_global_score_replay_exact_spans"
    ),
}
LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6Z_ROLE_VARIANTS = {
    "global_m128_c4096_first_fill_exact": (
        "global_context_rows_qrow4_dense_initial_global_score_weight_replay_"
        "exact_spans"
    ),
    "swa_qrow4_m128_c512_no_wrap_exact": (
        "swa_context_rows_qrow4_dense_initial_global_score_replay_exact_spans"
    ),
}
# WPF-H7Y's bounded 72-MiB mirror owner and fused writer are qualified but
# remain default-off pending source promotion. H6Z/H6W stay the source rollback.
LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H7Y_ROLE_VARIANTS = {
    "global_m128_c4096_first_fill_exact": (
        "global_context_rows_qrow4_dense_initial_global_score_weight_replay_"
        "exact_spans"
    ),
    "swa_qrow4_m128_c512_no_wrap_exact": (
        "swa_context_rows_qrow4_dense_initial_lane_major_"
        "global_score_replay_exact_spans"
    ),
}
LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS = dict(
    LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6Z_ROLE_VARIANTS
)
# WPF-H5R promotes exact cached-only SWA attention after KL0 complete state,
# all 144 physical append-before-attention calls, and positive default-off clean
# 512/1K/4K timing. The package-only role map restricts schedule reordering to
# complete no-wrap M128 starts; global, partial, wrapped, explicit, missing, and
# unsupported routes retain attend-before-append fallbacks.
LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS = {
    "swa_qrow4_m128_c512_no_wrap_exact": (
        "swa_context_rows_qrow4_cached_exact_spans"
    ),
}
LAGUNA_PREFILL_KV_PREAPPEND = True
# Clean W7900 D12 leaf/profile/category evidence admits the exact local32
# two-output Q5 schedules for c=1 attention output and query/gate projection.
# Other backends retain the separately registered pack8 fallbacks.
LAGUNA_Q5_WAVE32X2_OUTPUT = True
LAGUNA_Q5_WAVE32X2_QUERY_GATE = True
# Exact wave-uniform Q5 metadata loads pass first/last actual layers, full state,
# both clean context orders, and both complete 18-prompt category orders.
# Explicit role-scoped disables retain the registered coefficient-publication path.
LAGUNA_Q5_FIXED_METADATA = True
# The exact local32 BF16 pair for shared-Q5 gate/up passes first/last actual
# weights, full state, both clean context orders, and both complete category
# orders. Explicit disable restores the registered local128 pack8 pair.
LAGUNA_Q5_SHARED_FIXED_METADATA = True
# WPF-H8A builds exact F32 tile-K-col planes once for the complete global
# attn_q/attn_output class. Complete-plane/state/topology plus fixed, clean-
# length, and committed-production gates admit source ownership; explicit
# disable retains transient H7G.
LAGUNA_Q5_F32_RESIDENT_GLOBAL_CACHE = True
# WPF-H8B reuses an exact H5Y/H6U activation pack only within one explicit
# immutable projection group. Complete state/topology, fixed/length, and clean
# committed-production gates admit source ownership; explicit disable retains H8A.
LAGUNA_ACTIVATION_PACK_REUSE = True
# Exact mixed Q5/Q6 and corrected Q6/Q8 projection quads pass actual layers,
# full state, both clean context orders, and both complete category orders.
# Explicit disable retains the registered Q5/Q6 pair and Q8 singleton chain.
LAGUNA_MIXED_ATTENTION_PROJECTIONS = True
# Cooperative Q6 metadata publication inside the retained mixed quad is exact,
# improves all clean contexts, and passes both complete category process orders.
# Explicit disable restores the generic-Q6 mixed projection variant.
LAGUNA_MIXED_Q6_FIXED_METADATA = True
# Exact all-local32 Q5/Q6 ownership passes production bits, full state, both
# clean context orders, and both complete category orders. Explicit disable
# restores the retained local128 fixed-Q6 mixed projection; layer 47 stays there.
LAGUNA_MIXED_LOCAL32_FIXED_METADATA = True
# The exact Q4_K c=1 BF16-to-F32 LM-head sibling passes production bits,
# full state, cached one-call tracing, both clean context orders, and both
# complete category process orders. Explicit False restores registered local128.
LAGUNA_Q4_LM_HEAD_LOCAL32_FIXED_METADATA = True
# The registered wave4 producer plus slot-order reducer remains the explicit
# rollback and exact-key-miss fallback for the retained fused schedule below.
# Other backends retain their independently selected schedules.
LAGUNA_IQ3_C1_DOWN_SCHEDULE = "wave4_reduce"
# Exact local320 ten-wave IQ3 producer/reducer fusion passes full-state,
# clean-context, and both complete 18-prompt category orders. Explicit wave4
# or serial requests and exact-key misses retain the registered unfused chain.
LAGUNA_IQ3_WAVE10_FUSED = True
# The exact expanded-magnitude IQ2 grid contracts selector reconstruction and
# passes actual-layer, full-state, clean-context, and complete-category gates.
# Rows>1 and other backends retain the compact-grid tile2 fallback.
LAGUNA_IQ2_GRID64 = True
# Clean P2.1 exact split profiles and the complete category/heldout gate admit
# independent global/SWA crossovers. Registered single-block readers remain the
# below-threshold and explicit-disable fallback on gfx1100; other backends do
# not inherit these architecture-local thresholds.
LAGUNA_GLOBAL_SPLIT_MIN_LIVE = 127
LAGUNA_SWA_SPLIT_MIN_LIVE = 65
# P2's exact post-online refinement reduces the retained SWA score-producer
# sub-window at 257/511/512 without changing dispatches, memory, or arithmetic.
LAGUNA_SWA_SPLIT_TILE16_MIN_LIVE = 257
# P4.1 exact actual-layer screens admit folding the existing softplus gate and
# BF16 store into each retained split reducer. The unfused registered chain is
# the explicit rollback and remains the below-threshold fallback.
LAGUNA_SPLIT_GATE_FUSION = True
# The retained SWA reducer replays exact scalar softmax statistics independently
# in each logical wave. This removes all block barriers/LDS and improves every
# clean context plus the complete two-order category suite.
LAGUNA_SWA_SPLIT_WAVE_LOCAL = True
# Current-P4 first/last-layer, clean context, full-state, and complete two-order
# category evidence admits exact c=1 head RMSNorm+RoPE+BF16 KV append fusion.
# Rows/prefill and other backends retain the registered two-launch fallback.
LAGUNA_HEAD_KV_FUSION = True

# Keep the conservative physical-C4 default that prevents the measured 4K/C8
# resident-session OOM. The independently retained p512/d128 physical-C8 and
# live C8->C13 membership gates admit thirteen logical resident rows, lowered
# through masked physical groups no wider than C8, only within the 768-position
# envelope.
GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS = 4
GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS_BY_MAX_SEQUENCE_LENGTH = {
    768: 13,
}
# gfx1151 F4 retains the scoped fair:256 default and W7900 Qwen3.8-27B
# Q4_K_M/BF16-KV at 16K context measured width-4 packed AR decode under fair
# with the burst-1 window (40 packed width-4 steps, zero serial fallback) while
# the protect_decode default serialized every request's decode before the next
# prefill (zero packed steps at c4/c8 bursts). The env override
# HIPENGINE_PREFILL_DECODE_POLICY remains the explicit pin for configurations
# that must stay on protect_decode (e.g. the A4 frozen UD-Q4_K_M gates).
GGUF_Q4_K_M_PREFILL_DECODE_POLICY = "fair"
GGUF_Q4_K_M_MAX_PREFILL_CHUNK_TOKENS = 256
GGUF_Q4_K_M_FAIR_PREFILL_BURST_CHUNKS = 1
# Clean W7900 SOL-G5 p512/d24 evidence admits the state-bound composite GGUF
# graph when at least 24 decode transitions amortize capture/instantiate/close.
GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS = 24
# Full PM4 attribution admits only measured model/quant/physical-row horizons.
# The 35B-A3B c1 floor includes >10% margin over its clean p4096 break-even at
# 143 transitions; packed-width floors retain their independent margins. The
# dense 27B private-c1 row is separately qualified at the measured 128-token
# campaign horizon; wider speculative rows remain HIP graph until measured.
GGUF_DECODE_GRAPH_SUBMISSION_POLICIES = {
    (QWEN35_MOE_H2048_E256_GEOMETRY, "MOSTLY_Q4_K_M"): {
        "transport": "pm4",
        "min_replay_steps_by_physical_rows": {1: 160, 2: 64, 4: 96, 8: 80},
    },
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {
        "transport": "pm4",
        "min_replay_steps_by_physical_rows": {1: 128},
    },
}
# LCP-5A's clean peer-aligned semantic/decode contract and 512/4K floors admit
# the llama.cpp-HIP-shaped normalized-Q/K wave32 recurrence on gfx1100.  The
# compact producer/consumer retains those bits while materializing shared Q/K
# once per K head; cross-board M512/M1024 complete-chain gates admit it.
# Scalar-exact direct LDS32 remains available through the explicit selector.
GGUF_GDN_PREFILL_AUTO_MODE = "chain_compact_peer_wave32"
# No narrower GDN shape overrides are admitted on gfx1100; model families use
# the cross-board compact peer default above.
GGUF_GDN_PREFILL_AUTO_MODES_BY_QUANT_SHAPE: dict[
    tuple[str, int, int, int, int], str
] = {}
# Strict-exact rollback/oracle stays architecture-scoped and does not replace
# the quality-admitted peer-wave production default.
GGUF_GDN_PREFILL_EXACT_MODE = "chain_lds32_direct_nonvolatile"
# The singleton-indexed packed-AR recurrence is retained only on independently
# measured backends. gfx1100 keeps the arbitrary-segment fallback by default.
GGUF_GDN_INDEXED_SINGLETON_DECODE = False
# Exact Q8T16 row-amortized decode remains explicit on gfx1100 until an
# independent native-AR width gate passes on W7900.
GGUF_Q8_T16_DECODE_ROWTILE_ALL = False
# Zero keeps the width-scoped all-projection policy disabled on gfx1100.
GGUF_Q8_T16_DECODE_ROWTILE_MIN_ROWS = 0
# The exact 128-thread c8 pair schedule is admitted only on independently
# measured backends. Zero disables automatic pair rowtiling on gfx1100.
GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS = 0
# Dynamic selected-expert pair reuse remains disabled until independently
# measured on W7900. Zero preserves the existing gfx1100 route.
GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS = 0
# Qwen Q5T16 tile8 selected-down requires an independent W7900 gate.
GGUF_Q5_T16_SELECTED_QWEN_TILE8 = False
# Q5T16 selected-down pair reuse also requires an independent W7900 gate.
GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS = 0
GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS = 0
# The qualified dense H5120 geometry uses one compact T16 owner for rank-2 Q4 c1,
# verifier rows, and bulk prefill. This removes the prior 13.037 GiB pack8
# payload rather than retaining T16 as a 10.049 GiB sidecar.
GGUF_DENSE_Q4_T16 = True
# W7900 physical SPECDEC2 R6 reuses retained C1 rowtile arithmetic for five
# standard-Q4 target shapes. One rows6 launch is BF16-bit exact to two
# independent rows3 owners while avoiding the shared-B kernel's padded 256-row
# tile. The dominant gate/up shape must preserve shared-B bits after its rowtile
# caused one heldout strict-ID divergence, so it uses the bit-exact measured
# single-wave sibling instead. Other rows/shapes retain registered shared-B.
GGUF_Q4_T16_PHYSICAL_C1_ROWTILE_ROWS = frozenset({6})
GGUF_Q4_T16_PHYSICAL_C1_ROWTILE_SHAPES = frozenset(
    {
        (5_120, 1_024),
        (5_120, 6_144),
        (5_120, 10_240),
        (5_120, 12_288),
        (17_408, 5_120),
    }
)
GGUF_Q4_T16_PHYSICAL_SINGLE_WAVE_SHAPES = frozenset(
    {
        (5_120, 17_408),
        (5_120, 10_240),
        (5_120, 12_288),
    }
)
# Exact source-grounded two-wave geometry: adjacent wave32 owners share one
# WG64/16-column block without changing per-output K/FMA/reduction order. The
# W7900 actual-weight R6 screen is positive for all five standard-Q4 role
# shapes (1.039x-1.330x, 16-20/20 pair wins). Every row/shape miss retains the
# registered WG32/eight-column parent.
_Q4_T16_ROWTILE16_W2_R6_SHAPES = {
    (5_120, 1_024),
    (5_120, 6_144),
    (5_120, 10_240),
    (5_120, 12_288),
    (5_120, 17_408),
    (6_144, 5_120),
    (17_408, 5_120),
}
GGUF_SPECDEC2_Q4_DUAL_SILU_ROWTILE_POLICY = {
    "enabled_env": "HIPENGINE_GGUF_Q4_T16_DUAL_SILU_ROW48",
    "enabled_default": True,
    "rows_to_variant": {
        36: "dense_dual_wmma_prefill_row48_bf16_bf16_out",
    },
}
GGUF_T16_NATIVE_ROWTILE_VARIANTS_BY_QUANT = {
    "gguf_q4_k_t16_v1": {
        "canonical": True,
        "enabled_env": "HIPENGINE_GGUF_Q4_T16_ROWTILE16_W2",
        "enabled_default": True,
        "shapes": {
            shape: "dense_rowtile16_w2_bf16_bf16_out"
            for shape in _Q4_T16_ROWTILE16_W2_R6_SHAPES
        },
        "rows_by_shape": {
            shape: (6,) for shape in _Q4_T16_ROWTILE16_W2_R6_SHAPES
        },
    }
}
# Rows for which the single-wave leaf owns a single-wave shape instead of the
# 256-row shared-B tile. The shared-B kernel launches on a
# ``Q4_DENSE_TILE_ROWS = 4*4*16 = 256`` row tile, so small rows pay nearly the
# full 256-row cost; a W7900 row sweep measured single-wave bit-identical and
# faster for rows 2..128 on (5120,17408) and (5120,10240). A repaired
# forward/reverse sweep retained (5120,12288) only through row 112 (1.10x in
# both orders); rows 120/124/128 were 0.994x/0.995x/0.999x in the higher-repeat
# run, so that shape uses the narrower cap below. The shapes it does **not**
# contain are
# measured losses, not unmeasured gaps: single-wave is 0.77x-0.83x at
# (17408, 5120), 0.75x-0.85x at (5120, 6144) and (6144, 5120), and 0.63x-0.92x at
# (5120, 1024). Strict shared-B stays the registered sibling and the fallback.
# Evidence:
# benchmarks/results/2026-08-30-w7900-q4km-t16-single-wave-rows-accepted.json and
# benchmarks/results/2026-08-30-w7900-q4km-t16-single-wave-shapes-accepted.json.
GGUF_Q4_T16_PHYSICAL_SINGLE_WAVE_MAX_ROWS = 128
GGUF_Q4_T16_PHYSICAL_SINGLE_WAVE_MAX_ROWS_BY_SHAPE = {(5_120, 12_288): 112}
# Exact four-wave shared-B sibling with one 16-row tile per wave. On the W7900
# down projection it reduces 256-row padding to 64 rows while retaining the
# parent's 48-column weight-sharing block: 1.10x-1.85x through rows 33-192;
# row 193 adds a fourth row group and falls to 0.90x, setting the boundary.
GGUF_Q4_T16_PHYSICAL_SHARED_B_ROW64_SHAPES = frozenset({(17_408, 5_120)})
GGUF_Q4_T16_PHYSICAL_SHARED_B_ROW64_ROWS = range(33, 193)
# Production C2 reuses C1-equivalent rows6 rowtiles for the remaining Q4
# gate/up and full-attention output shapes. Complete strict-teacher, repeat,
# permutation-isolation, task, and lifecycle packets qualify this arithmetic
# only through the request-local SPECDEC2 scope; strict retains single-wave /
# shared-B siblings.
GGUF_SPECDEC2_PRODUCTION_PHYSICAL_PROMPT_STREAMING = True
GGUF_SPECDEC2_PRODUCTION_PHYSICAL_EXTRA_ROWTILE_SHAPES = frozenset(
    {(5_120, 17_408), (6_144, 5_120)}
)
GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q5_ROWTILE_ROWS = frozenset({6})
GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q6_ROWTILE_ROWS = frozenset({6})
# Default-off Q6 launch-composition screen for larger physical target groups.
# Actual Qwen3.8 weights select only R24/R30/R36: every recurrent-QKV,
# full-attention-V, and FFN-down leaf is BF16-bit exact and 1.079x-1.234x
# faster. R18 remains repeated R6 after recurrent QKV lost and FFN-down was
# noise-flat.
GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q6_MIXED_ROWTILE_CHUNKS = {
    24: (8, 8, 8),
    30: (8, 8, 8, 6),
    36: (8, 8, 8, 6, 6),
}
GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q6_MIXED_ROWTILE_SHAPES = frozenset(
    {(5_120, 10_240), (5_120, 1_024), (17_408, 5_120)}
)
# Exact C2/K3 R8 is the retained production default. It eliminates the four
# inactive rows required by the rows6-multiple fallback; every Q4/Q5/Q6
# actual-weight leaf is BF16-bit exact to the active rows of R12 and
# 1.31x-1.84x faster. The adapter enables this capability only inside its
# request-local production scope and retains explicit padded-R12 rollback.
GGUF_SPECDEC2_PRODUCTION_PHYSICAL_EXACT_ROWTILE_ROWS = frozenset({8})
# The production dense adapter owns one physical request group through C8. The
# adapter derives frontier/accept workspaces from this package capability and
# retains an environment rollback to the previous C4 ceiling. Strict remains
# on its existing C4/exact-fallback route.
GGUF_SPECDEC2_MTP2_MAX_REQUESTS = 8
# Pad physical SPECDEC2 target verify groups up to the admitted rows6
# production rowtile when the root+candidate total falls below it (K1 R4 and
# ragged 2-5-row cycles otherwise ride the shared-B 256-row padded tile at a
# measured ~5.1x cycle cost on W7900). Pad rows are inactive candidates owned
# by the last request; accept/commit stay driven by active rows only. Strict
# profiles do not read this capability.
GGUF_SPECDEC2_TARGET_VERIFY_PAD_ROW_COUNTS = (6,)
# W7900 retains the expanded-metadata T16 gate/up payload. The qmicro replacement
# is qualified independently for gfx1151 and must fail closed on this backend.
GGUF_DENSE_Q4_QMICRO_T16_GATE_UP = False
# Private-c1 token lookup reads one compressed row from a pinned GGUF mmap.
# The mapping is device-visible but does not create a VRAM weight shadow.
GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1 = True
# Auto placement must fail closed before deferring unsupported root tables. The
# explicit host override remains strict and reports the unsupported tensor.
GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1_GGML_TYPES = ("Q4_K",)
# Pack bounded immutable weights into one private-c1 allocation owner only for
# measured geometry/quant identities. Each policy freezes its screened cutoff;
# larger tensors remain dedicated and the planner fails closed before upload.
GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA_POLICIES = {
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {
        "enabled": True,
        "max_allocation_bytes": 80 * 1024 * 1024,
    },
}
# The dense private-c1 decode workspace exposes 188 logical state/KV/temporary
# views through one physical owner. The environment rollback remains available
# only while the complete four-shape memory/performance gate accumulates evidence.
GGUF_PRIVATE_C1_DECODE_SCRATCH_ARENA_POLICIES = {
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {"enabled": True},
}
# The sole-Q4T16 dense gate/up owners already expose an exact c1 dual+SiLU
# sibling. The complete Qwen3.8 512/128 and natural25 gates admit it for the
# validated H5120 geometry, removing 128 decode-graph nodes without a sidecar.
GGUF_DENSE_PAIR_SILU_DECODE_POLICIES = {
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {
        (1, 5_120, 17_408): "dense_dual_local32_bf16_bf16_out",
    },
}
# Production-cache rotation admits sole-resident Q5T16 for the measured dense
# H5120 K6,144/N5,120 recurrent output projections. The materializer remains
# shape/role qualified; peer backends keep dense BF16 until independently gated.
GGUF_DENSE_Q5_T16_SSM_OUT = True
# Q5T16 and planar-qmicro Q6T16 true rowtile primitives were extended and
# validated to rows 5-8 (strict bit-parity vs the per-row producer), so native
# batch decode rewrites rows 2-8 to the true rowtile instead of padded WMMA.
# Other T16 quants retain the generic rows-through-6 behavior unless their
# backend package overrides it.
GGUF_T16_NATIVE_ROWTILE_MAX_ROWS_BY_QUANT = {
    "gguf_q5_k_t16_v1": 8,
    "gguf_q6_k_t16_qmicro_planar_v1": 8,
}
# Exact c1 sibling selection is architecture/shape qualified. W7900 retains
# the established direct owners until an independent device gate admits one.
GGUF_T16_C1_VARIANTS_BY_QUANT_SHAPE = {}
# Full-suite row policy for exact FFN-down plus residual composites. Rotating
# row-4 planar-Q6 loses despite positive isolated leaves, while compact Q4 wins;
# rows 2-3 retain both independently qualified owners.
GGUF_LINEAR_RESIDUAL_MAX_ROWS_BY_QUANT = {
    "gguf_q4_k_t16_v1": 4,
    "gguf_q6_k_t16_qmicro_planar_v1": 3,
}
# Byte-neutral planar qmicro owns the measured dense-H5120 rank-2 Q6 family
# and K5,120/N248,320 root head on W7900. Exact actual-weight c1, rows2-4,
# top-1, and M64/M512 gates improve; peer backends and unmeasured shapes keep
# legacy T16 until independently admitted.
GGUF_DENSE_Q6_T16_QMICRO_PLANAR = True
# The wide planar-Q6 FFN-down prefill owner uses exact cooperative siblings to
# avoid one-wave underfill on W7900. Rows33-128 use four waves over one 16-row
# tile each; rows129-511 use the existing four-wave 256-row owner. Rows<=32 keep
# verifier ownership unchanged, and rows>=512 retain the independently gated
# source-F16 route. The one-wave parent remains a registered strict fallback.
GGUF_Q6_PLANAR_EXACT_PREFILL_VARIANTS = {
    (17_408, 5_120): (
        (33, 128, "t16_wmma_prefill_shared4_row64_bf16_bf16_out"),
        (129, 511, "t16_wmma_prefill_shared4_bf16_bf16_out"),
    ),
}
# Production-shape Q4 changed-arithmetic screen admits FFN-down plus
# full-attention K/V/output. All win at M512/1K/4K; only FFN-down can consume its
# dead BF16 activation in place. Gate/up remain exact at the projection boundary
# but bulk rows now use the operation-complete dual-WMMA+SiLU owner; wider input
# projections retain this bounded source-F16 policy and exact fallbacks.
GGUF_Q4_T16_UNEQUAL_PAIR_PREFILL_POLICIES = {
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): True,
}
GGUF_DENSE_T16_F16_ROCBLAS_PREFILL_POLICIES = {
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): True,
}
GGUF_Q4_T16_F16_ROCBLAS_PREFILL_POLICIES = {
    (17_408, 5_120): {512: 1_024, 768: 512, 1_024: 512, 4_096: 512},
    (5_120, 1_024): {512: 1_024, 768: 512, 1_024: 512, 1_280: 512, 4_096: 256},
    (6_144, 5_120): {512: 1_024, 768: 1_024, 1_024: 512, 1_280: 512, 4_096: 512},
    (5_120, 12_288): {512: 2_048, 1_024: 512},
}
# Row ceilings prevent nearest-anchor extrapolation through known losing shape
# boundaries. Full-attention Q wins with the pair producer at 512/1K but keeps
# exact Q4T16 at 2K/4K and above.
GGUF_T16_F16_ROCBLAS_MAX_ROWS_BY_QUANT_SHAPE = {
    "gguf_q4_k_t16_v1": {(5_120, 12_288): 2_047},
}
# Ordered pair-only admission lets one otherwise-exact second operand reuse an
# already-admitted first operand's FP16 activation without exposing the second
# shape to singleton or unrelated pair dispatch. Values are inclusive row
# intervals -> (second-operand tile, registered variant, activation-in-place).
GGUF_T16_F16_ROCBLAS_PAIR_ONLY_POLICIES = {
    (
        "gguf_q6_k_t16_qmicro_planar_v1",
        5_120,
        10_240,
        "gguf_q4_k_t16_v1",
        6_144,
    ): {
        (512, 1_023): (2_048, "f16_rocblas_t16_pair_bf16_bf16_out", False),
        (1_024, 2_047): (512, "f16_rocblas_t16_pair_bf16_bf16_out", False),
    },
}
# Registered linear-variant overrides for source-F16 chains. Shape and inclusive
# row intervals are model-scoped by the runner policy; absent intervals retain
# each quant's ordinary registered composite. The disjoint K/V M4096 singleton
# deliberately does not extrapolate through the mixed M1280 boundary.
GGUF_T16_F16_ROCBLAS_VARIANT_POLICIES = {
    "gguf_q4_k_t16_v1": {
        (17_408, 5_120): {
            (512, 4_096): "f16_rocblas_t16_pair_bf16_bf16_out",
        },
        (5_120, 1_024): {
            (512, 1_024): "f16_rocblas_t16_pair_bf16_bf16_out",
            (4_096, 4_096): "f16_rocblas_t16_pair_bf16_bf16_out",
        },
        (6_144, 5_120): {
            (512, 768): "f16_rocblas_t16_pair_bf16_bf16_out",
        },
        (5_120, 12_288): {
            (512, 2_047): "f16_rocblas_t16_pair_bf16_bf16_out",
        },
    },
    "gguf_q5_k_t16_v1": {
        (6_144, 5_120): {
            (512, 4_096): "f16_rocblas_t16_octet_bf16_bf16_out",
        },
    },
}
# Quality-gated Q5T16 recurrent-output policy. The K6,144 activation is cast in
# its dead input, preserving the sole resident T16 payload and bounded workspace.
GGUF_Q5_T16_F16_ROCBLAS_PREFILL_POLICIES = {
    (6_144, 5_120): {512: 1_280, 1_024: 1_280, 4_096: 1_024},
}
# Full-category changed-arithmetic admission for sole-planar Q6 dense prefill.
# Rows below 512 (decode/MTP) and the Q6 lm-head remain on exact owners.
GGUF_Q6_T16_F16_ROCBLAS_PREFILL_POLICIES = {
    (17_408, 5_120): {512: 1_024, 768: 1_024, 1_024: 512, 1_280: 512, 4_096: 512},
    (5_120, 10_240): {512: 2_048, 768: 2_048, 1_024: 512, 1_280: 512, 4_096: 256},
    (5_120, 1_024): {512: 1_024, 768: 1_024, 1_024: 512, 1_280: 512, 4_096: 512},
}
# Zero-workspace rocBLAS solution indices selected by exact effective GEMM
# shape. Each preserves every FP16 output bit versus standard dispatch and wins
# on both gfx1100 boards; absent shapes deliberately keep rocBLAS standard.
GGUF_T16_F16_ROCBLAS_SOLUTION_VERSION_PREFIX = "5.2.0.dabb6df2b98"
GGUF_T16_F16_ROCBLAS_SOLUTION_INDICES = {
    (512, 5_120, 1_024): -1_140_856_081,
    (512, 5_120, 2_048): -1_140_856_092,
    (4_096, 5_120, 512): -1_140_855_996,
    (4_096, 17_408, 512): -1_140_855_997,
    (4_096, 6_144, 512): -1_140_855_996,
}
GGUF_Q6_LM_HEAD_MAX_CHUNK = 6
# Concurrency2 C2-6 W7900 qualification (re-qualified 2026-08-17 after the
# conv_out arena-aliasing fix 422209168 and the state-oracle checkout fix
# 46466a86e). Shared-slot packed AR decode is byte-exact through physical c8:
# steady, masked-lane shrink-sparse, fixed-width graph, and p128/p512
# state-oracle gates all exact on tokens, Conv/GDN state, live KV, and every
# layer hidden versus independent c1, with resolution provenance recorded.
# Promoted 2026-08-20 after direct c3/c5/c6/c7 lifecycle certification (#36).
GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8)
# SPECDEC2 exposes the dense GGUF C1 adapter plus the independently qualified
# explicit physical C2 owner on gfx1100. The C4 construction seam is shared,
# but model-plugin evidence still admits only exact C2 keys; C4 and automatic
# physical policy remain fail-closed.
GGUF_SPECDEC2_MTP2_C1 = True
GGUF_SPECDEC2_MTP2_C4 = True
# W7900 P2 p128 found deterministic native target-graph NaN/sentinel output;
# eager/serial target verification remains exact above the locally-qualified
# natural25 context envelope.  This is graph admission, not model policy.
GGUF_SPECDEC2_NATIVE_TARGET_GRAPH_MAX_CONTEXT = 95
GGUF_SPECDEC2_NATIVE_TARGET_MAX_CONTEXT = 95
# Packed-PARO S7 starts with the independently-qualified singleton K1/R2
# frontier only. C2/C4 remains absent until physical multi-request kernels pass.
PARO_SPECDEC2_MTP2_C1 = True
PARO_SPECDEC2_MTP2_C4 = False
# Clean W7900 GPF-3A full-model 512/4K evidence admits byte-exact shared-X
# selected-dual Q4T16 prefill after the predeclared borderline-decode repeat.
GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE = "shared_x"
# Clean W7900 GPF-5A 512/4K transfer evidence admits the exact two-wave dense
# Q8T16 schedule only through the independently measured request scope.
GGUF_Q8_T16_PREFILL_TWO_WAVE = True
GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS = 4096
# WPF-H7U promotes exact stable parallel active-route compaction after full
# standalone, bounded-runtime, fixed, length, and source-trace qualification.
# Explicit serial remains the registered rollback; peer backends stay local.
LAGUNA_MOE_GROUP_COMPACT_MODE = "parallel"
# WPF-2b's exact expert-major IQ2 gate/up route compounds grouped IQ3/IQ4 down
# at M512 with both attention capacities fixed at 128. Complete state is KL0 and
# dirty paired 512/1K admission improves 19.65%/18.01%. Explicit grouped_exact
# gate/up preserves the preceding route-major rollback; direct/M128/M256 remain
# exact fallbacks for unsupported quant/shape keys.
LAGUNA_PREFILL_MATRIX_ROWS = 512
LAGUNA_SELECTED_GATE_UP_MODE = "grouped_pair16"
LAGUNA_SELECTED_DOWN_MODE = "grouped_exact"
# WPF-H6L promotes exact K3072/N1024/E256 rowbatch16 ownership after complete
# default-off state/topology and positive 512/1K/4K gates. Rowbatch8 remains the
# registered same-ABI rollback; shape/registration/backend misses fail closed.
_WPF2B_IQ2_PAIR16_VARIANT = (
    "selected_dual_silu_grouped_prefill_compact_"
    "pair16_rowbatch8_bf16_bf16_out"
)
_H6L_IQ2_PAIR16_ROWBATCH16_VARIANT = (
    "selected_dual_silu_grouped_prefill_compact_"
    "k3072_n1024_e256_pair16_rowbatch16_bf16_bf16_out"
)
LAGUNA_GROUPED_PAIR16_GATE_UP_VARIANTS = {
    "gguf_iq2_xs": _H6L_IQ2_PAIR16_ROWBATCH16_VARIANT
}
LAGUNA_GROUPED_PAIR16_GATE_UP_VARIANT_ABIS = {
    _WPF2B_IQ2_PAIR16_VARIANT: "grouped_raw_iq_dual_silu",
    _H6L_IQ2_PAIR16_ROWBATCH16_VARIANT: "grouped_raw_iq_dual_silu",
}
# WPF-H6C promotes the exact K3072/N1024/E256 fused-SiLU rowbatch4 leaf for
# only the qualified layer-47 IQ3 role after exact state, physical topology,
# clean 512/1K/4K, and matched C4096/M512 wins. An empty role map remains the
# scoped rollback. Wrong role/shape/quant/registration/backend and c=1 retain
# the exact grouped route-major chain, while IQ2 keeps promoted pair16.
_H6C_IQ3_GATE_UP_VARIANT = (
    "selected_dual_silu_grouped_prefill_compact_"
    "k3072_n1024_e256_rowbatch4_bf16_bf16_out"
)
LAGUNA_GROUPED_GATE_UP_ROLE_VARIANTS = {
    "layer47_iq3_k3072_n1024_e256": _H6C_IQ3_GATE_UP_VARIANT
}
LAGUNA_GROUPED_GATE_UP_VARIANT_ABIS = {
    _H6C_IQ3_GATE_UP_VARIANT: "grouped_raw_iq_dual_silu"
}
# WPF-H5Z promotes exact K1024/N3072 activation-resident P256 IQ3 ownership
# after KL0 complete state, exact 45-call integration, and positive selector-
# unset 512/1K/4K timing. H5Q remains registered rollback through the same
# bounded active-expert ABI. H6D retains exact row-interleaved VOPD rollback.
# H6F promotes paired-output ownership through the unchanged ABI after complete
# state, exact topology, and positive short/matched gates. H6I then promotes
# triple-output source ownership through that same raw allocation and library;
# H6F remains the immediate registered rollback and H5J remains unchanged.
# H6P is a qualified default-off same-ABI capability; source stays H6I until
# complete-state, integrated-topology, and clean request gates adjudicate it.
# H6T is the retained source through the same raw allocation and active-expert
# ABI after complete-state, topology, fixed, and 512/1K/4K publication gates.
_H5Q_IQ3_ACTIVE_EXPERT_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "resident_rowbatch8_bf16_bf16_out"
)
_H5Z_IQ3_ACTIVATION_RESIDENT_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_rowbatch8_bf16_bf16_out"
)
_H6D_IQ3_ROW_INTERLEAVED_VOPD_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_rowbatch8_bf16_bf16_out"
)
_H6F_IQ3_PAIRED_OUTPUT_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_paired_output_"
    "rowbatch8_bf16_bf16_out"
)
_H6I_IQ3_TRIPLE_OUTPUT_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_triple_output_"
    "rowbatch8_bf16_bf16_out"
)
_H6P_IQ3_STAGED_WAVE_PUBLICATION_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_"
    "staged_wave_publication_triple_output_rowbatch8_bf16_bf16_out"
)
_H6Q_IQ3_COMPACT_SHUFFLE_LOOP_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_"
    "staged_wave_publication_compact_shuffle_loop_triple_output_"
    "rowbatch8_bf16_bf16_out"
)
_H6R_IQ3_DPP_PEER_EXCHANGE_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_"
    "staged_wave_publication_dpp_peer_exchange_triple_output_"
    "rowbatch8_bf16_bf16_out"
)
_H6T_IQ3_FUSED_DPP_ADD_VARIANT = (
    "selected_grouped_prefill_compact_k1024_active_expert_p64_"
    "activation_resident_out_p256_row_interleaved_vopd_staged_wave_"
    "publication_dpp_peer_exchange_fused_add_triple_output_rowbatch8_"
    "bf16_bf16_out"
)
LAGUNA_GROUPED_IQ_DOWN_VARIANTS = {
    "gguf_iq3_xxs": _H6T_IQ3_FUSED_DPP_ADD_VARIANT,
    "gguf_iq4_xs": (
        "selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out"
    ),
}
LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS = {
    _H5Q_IQ3_ACTIVE_EXPERT_VARIANT: "grouped_raw_iq_active_experts",
    _H5Z_IQ3_ACTIVATION_RESIDENT_VARIANT: "grouped_raw_iq_active_experts",
    _H6D_IQ3_ROW_INTERLEAVED_VOPD_VARIANT: "grouped_raw_iq_active_experts",
    _H6F_IQ3_PAIRED_OUTPUT_VARIANT: "grouped_raw_iq_active_experts",
    _H6I_IQ3_TRIPLE_OUTPUT_VARIANT: "grouped_raw_iq_active_experts",
    _H6P_IQ3_STAGED_WAVE_PUBLICATION_VARIANT: (
        "grouped_raw_iq_active_experts"
    ),
    _H6Q_IQ3_COMPACT_SHUFFLE_LOOP_VARIANT: (
        "grouped_raw_iq_active_experts"
    ),
    _H6R_IQ3_DPP_PEER_EXCHANGE_VARIANT: (
        "grouped_raw_iq_active_experts"
    ),
    _H6T_IQ3_FUSED_DPP_ADD_VARIANT: "grouped_raw_iq_active_experts",
}
# WPF-1 established exact Q5/Q6 rowbatch8 after bit-exact full-state and short
# admission. WPF-1W promotes rowbatch32 after clean paired gains at both short
# shapes. WPF-1T's exact constant-accumulator screen admits four adjacent output
# columns for full RB32 slabs, then qualifies `(2,16)` only for four measured
# `(quant, output variant, K, N)` configurations. Explicit rowbatch preserves
# the preceding exact owner; zero/4/8/16/32 and unsupported widths remain
# fallbacks.
GGUF_RAW_K_PREFILL_ROWBATCH_SUPPORTED = True
GGUF_RAW_K_PREFILL_ROWBATCH = 32
GGUF_RAW_K_PREFILL_COLTILE_SUPPORTED = True
GGUF_RAW_K_PREFILL_COLTILE2_SHAPES = frozenset(
    {
        ("gguf_q5_k", "bf16_bf16_out", 3072, 12288),
        ("gguf_q5_k", "bf16_f32_out", 3072, 6144),
        ("gguf_q5_k", "bf16_f32_out", 3072, 9216),
        ("gguf_q6_k", "bf16_f32_out", 3072, 9216),
    }
)
GGUF_RAW_K_PREFILL_VARIANT = "coltile"
# Grouped full-prompt prefill: admit up to eight same-wave rows into one native prefill
# call. Declared for gfx1100 after the W7900 / Qwen3.8-27B Q4_K_M canonical-suite
# measurement that kept every generated id identical (432 cross-packet row comparisons,
# 0 mismatches; 80/80 correctness cells), left physical width 1 unchanged (-0.4% AR,
# acceptance identical), lifted C8 AR 45.68 -> 78.67 tok/s, and removed the width-
# dependent draft-acceptance collapse (0.467-0.614 at C2-C8 -> 0.789 at every width).
# Scoped independently from decode widths; misses fall back before mutation.
GGUF_C2_PACKED_PREFILL_MAX_ROWS = 8
# WPF-H7C is the retained exact raw-Q6 source for only the three physically and
# actually timed M512 roles below. The named empty map remains the explicit
# generic rollback. gguf_linear consumes this package map generically; no
# model/backend/quant branch owns these variants.
GGUF_RAW_K_PREFILL_GENERIC_ROLE_VARIANTS = {}
GGUF_RAW_K_PREFILL_H7C_ROLE_VARIANTS = {
    (
        "gguf_q6_k",
        "bf16_bf16_out",
        512,
        12_288,
        3_072,
    ): "dpp_wave_reduction_coltile4_rowbatch8_bf16_bf16_out",
    (
        "gguf_q6_k",
        "bf16_f32_out",
        512,
        3_072,
        9_216,
    ): "dpp_wave_reduction_coltile2_rowbatch16_bf16_f32_out",
    (
        "gguf_q6_k",
        "bf16_bf16_out",
        512,
        9_216,
        3_072,
    ): "dpp_wave_reduction_coltile4_rowbatch8_bf16_bf16_out",
}
# WPF-H7I is a separately qualified, default-off capability for the same exact
# three M512 roles. It removes only H7C's redundant inner live-row predicate;
# the live source map remains H7C until the source-default gate passes.
GGUF_RAW_K_PREFILL_H7I_ROLE_VARIANTS = {
    (
        "gguf_q6_k",
        "bf16_bf16_out",
        512,
        12_288,
        3_072,
    ): (
        "dpp_wave_reduction_full_group_compute_"
        "coltile4_rowbatch8_bf16_bf16_out"
    ),
    (
        "gguf_q6_k",
        "bf16_f32_out",
        512,
        3_072,
        9_216,
    ): (
        "dpp_wave_reduction_full_group_compute_"
        "coltile2_rowbatch16_bf16_f32_out"
    ),
    (
        "gguf_q6_k",
        "bf16_bf16_out",
        512,
        9_216,
        3_072,
    ): (
        "dpp_wave_reduction_full_group_compute_"
        "coltile4_rowbatch8_bf16_bf16_out"
    ),
}
GGUF_RAW_K_PREFILL_ROLE_VARIANTS = dict(GGUF_RAW_K_PREFILL_H7I_ROLE_VARIANTS)
# WPF-H5D promotes the exact H5C producer/ordered-consumer chain after KL0,
# byte-identical complete state, and clean +7.235%/+6.519% M512/M1024 gates.
# H5E doubles output ownership on six role-qualified geometries, reducing the
# producer-inclusive weighted Q5 family another 12.32% by events / 7.52% by
# synchronized wall while remaining byte-exact. H5F's exact 12x4 N48 role saves
# another 4.224/1.989 us per M512 request. H5G's exact constant-80/96 tiles
# improve clean 512/1K/4K another 2.192%/2.055%/1.329% on five roles. H5L's
# exact weight-major traversal cuts the six-role/235-call leaf 44.857%/46.544%
# by event/wall. H5X promotes tile-K-col layout on 151 calls after KL0 complete
# state. H5Y then adds an exact tile-K-row BF16 activation plane on all 188
# ordered-Q5 calls, preserving H5X/H5L weight layouts. H7G promotes exact
# padded-row compute for the four natural-M512 roles with a real padded tail
# after complete-state, integrated-topology, and positive fixed/512/1K/4K
# gates. Divisible r4/r8, N48/N72, and every miss retain H5Y/H5G/raw rollback.
GGUF_Q5_F32_ORDERED_PREFILL = True
GGUF_Q5_F32_ORDERED_PREFILL_H5Y_POLICY = {
    ("bf16", 3072, 1024): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile8_rowbatch4"
    ),
    ("bf16", 3072, 12288): (
        "weight_major_row_major_activation_tile_k_row_coltile8_rowbatch12"
    ),
    ("bf16", 6144, 3072): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile16_rowbatch5"
    ),
    ("bf16", 9216, 3072): (
        "weight_major_row_major_activation_tile_k_row_coltile12_rowbatch8"
    ),
    ("f32", 3072, 48): "coltile12_rowbatch4",
    ("f32", 3072, 72): "coltile8_rowbatch4",
    ("f32", 3072, 6144): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile16_rowbatch5"
    ),
    ("f32", 3072, 9216): (
        "weight_major_tile_k_col_activation_tile_k_row_coltile8_rowbatch10"
    ),
}
# WPF-H7G is the retained Q5 source after complete-state, integrated-topology,
# and selector-unset fixed/512/1K/4K publication; H5Y remains exact rollback.
GGUF_Q5_F32_ORDERED_PREFILL_H7G_POLICY = {
    **GGUF_Q5_F32_ORDERED_PREFILL_H5Y_POLICY,
    ("bf16", 3072, 12288): (
        "weight_major_row_major_activation_tile_k_row_"
        "padded_compute_coltile8_rowbatch12"
    ),
    ("bf16", 6144, 3072): (
        "weight_major_tile_k_col_activation_tile_k_row_"
        "padded_compute_coltile16_rowbatch5"
    ),
    ("f32", 3072, 6144): (
        "weight_major_tile_k_col_activation_tile_k_row_"
        "padded_compute_coltile16_rowbatch5"
    ),
    ("f32", 3072, 9216): (
        "weight_major_tile_k_col_activation_tile_k_row_"
        "padded_compute_coltile8_rowbatch10"
    ),
}
# WPF-H7H is the retained Q5 source for the two divisible natural-M512 roles
# after complete-state, topology, and fixed/512/1K/4K qualification. H7G remains
# the exact complete-map rollback.
GGUF_Q5_F32_ORDERED_PREFILL_H7H_POLICY = {
    **GGUF_Q5_F32_ORDERED_PREFILL_H7G_POLICY,
    ("bf16", 3072, 1024): (
        "weight_major_tile_k_col_activation_tile_k_row_"
        "full_group_compute_coltile8_rowbatch4"
    ),
    ("bf16", 9216, 3072): (
        "weight_major_row_major_activation_tile_k_row_"
        "full_group_compute_coltile12_rowbatch8"
    ),
}
GGUF_Q5_F32_ORDERED_PREFILL_POLICY = dict(
    GGUF_Q5_F32_ORDERED_PREFILL_H7H_POLICY
)
# WPF-H5I introduced the shared serial F32 plane and ordered Q6 consumers.
# WPF-H5W retains exact weight-major rollback. WPF-H6E retains exact generic-
# shuffle activation-row rollback. WPF-H6U promotes DPP wave reduction on
# 142/143 selected calls after KL0 complete state, exact integrated topology,
# and positive one-queue fixed/512/1K/4K gates at unchanged workspace. F32 N72
# and long-K/wide-N misses retain the H5I/raw exact fallbacks.
GGUF_Q6_F32_ORDERED_PREFILL = True
GGUF_Q6_F32_ORDERED_PREFILL_POLICY = {
    ("bf16", 3072, 1024): (
        "weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch5"
    ),
    ("bf16", 1024, 3072): (
        "weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch4"
    ),
    ("f32", 3072, 72): "coltile8_rowbatch4",
    ("f32", 3072, 1024): (
        "weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch5"
    ),
}
GGUF_Q6_F32_ORDERED_PREFILL_H6E_POLICY = {
    ("bf16", 3072, 1024): (
        "weight_major_row_major_activation_tile_k_row_"
        "coltile16_rowbatch5"
    ),
    ("bf16", 1024, 3072): (
        "weight_major_row_major_activation_tile_k_row_"
        "coltile16_rowbatch4"
    ),
    ("f32", 3072, 72): "coltile8_rowbatch4",
    ("f32", 3072, 1024): (
        "weight_major_row_major_activation_tile_k_row_"
        "coltile16_rowbatch5"
    ),
}
# WPF-H6U is the retained Q6 source after complete-state, integrated-topology,
# and selector-unset fixed/512/1K/4K publication; H6E remains exact rollback.
GGUF_Q6_F32_ORDERED_PREFILL_H6U_POLICY = {
    ("bf16", 3072, 1024): (
        "weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch5"
    ),
    ("bf16", 1024, 3072): (
        "weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch4"
    ),
    ("f32", 3072, 72): "coltile8_rowbatch4",
    ("f32", 3072, 1024): (
        "weight_major_row_major_activation_tile_k_row_"
        "dpp_wave_reduction_coltile16_rowbatch5"
    ),
}
GGUF_F32_ORDERED_PREFILL_QUANTS = frozenset(("gguf_q5_k", "gguf_q6_k"))
GGUF_F32_ORDERED_PREFILL_POLICIES = {
    "gguf_q5_k": GGUF_Q5_F32_ORDERED_PREFILL_POLICY,
    "gguf_q6_k": GGUF_Q6_F32_ORDERED_PREFILL_POLICY,
}
# LCP-2B removes the 512-token compact-MoE scheduler's per-layer scalar D2H
# boundary using a routing-independent tight padded-row upper bound. Larger
# selected-row shapes keep the exact scalar read until independently measured.
GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS = 4096
# Clean W7900 LCP-4A transfer evidence admits the exact 256-thread
# BF16-hidden/F32-weight router-logits geometry for bulk prefill and decode.
GGUF_ROUTER_F32_BF16_HIDDEN_THREADS = 256
# Clean W7900 LCP-4B transfer evidence admits 128 threads for the bulk-prefill
# top-k selector. Decode keeps its independently selected launch geometry.
GGUF_PREFILL_ROUTER_SELECT_THREADS = 128
# Clean W7900 LCP-M2 transfer evidence admits one stream-ordered metadata
# preparation kernel in place of six synchronous H2D copies through 4K.
GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS = 4096
# Clean W7900 LCP-D2 correctness and 32K/64K/128K graph-decode gates admit the
# split-parallel gated reduction from 32K onward. Shorter contexts retain the
# single-launch serial reducer because the extra prepare launch is neutral/down.
GGUF_PAGED_ATTN_PARALLEL_REDUCE = True
GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT = 32768
# LCP-M1 validates an aligned phase-liveness arena for the production Qwen35
# MoE prefill route. Diagnostic/F32 layouts retain dedicated allocations.
GGUF_PREFILL_SCRATCH_LIVENESS_ALIAS = True
# The qualified dense H5120 geometry has no MoE/shared-expert route, so those
# fields are omitted and the remaining phases reuse the proven arena plan.
GGUF_DENSE_PREFILL_SCRATCH_LIVENESS_POLICIES = {
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M"): {
        "min_rows": 1,
        # Short verifier/NextN arenas retain size-first coloring: moving the
        # long-row priority fields there corrupts the second target-logit row.
        "priority_min_rows": 4_096,
        # Long-row bulk layers consume the source hidden plane before writing
        # final FFN output, so one physical plane can serve both roles.
        "hidden_inplace_min_rows": 4_096,
        # Prefix-8 production oracle: duration>=5 reaches 372.375 MiB while
        # preserving root+128 IDs. Moving the next field (attn_out) diverges.
        "priority_min_live_stages": 5,
    },
}
# LCP-1 remains a separately registered diagnostic on gfx1100 because its
# architecture-local full-state and wall gate rejected automatic promotion.
GGUF_LINEAR_ATTN_CONV_PREFILL_AUTO_MODE = "baseline"

__all__ = [
    "LAGUNA_ACTIVATION_PACK_REUSE",
    "LAGUNA_GLOBAL_SPLIT_MIN_LIVE",
    "LAGUNA_HEAD_KV_FUSION",
    "LAGUNA_GROUPED_GATE_UP_ROLE_VARIANTS",
    "LAGUNA_GROUPED_GATE_UP_VARIANT_ABIS",
    "LAGUNA_GROUPED_PAIR16_GATE_UP_VARIANTS",
    "LAGUNA_GROUPED_PAIR16_GATE_UP_VARIANT_ABIS",
    "LAGUNA_GROUPED_IQ_DOWN_VARIANTS",
    "LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS",
    "LAGUNA_IQ2_GRID64",
    "LAGUNA_IQ3_C1_DOWN_SCHEDULE",
    "LAGUNA_IQ3_WAVE10_FUSED",
    "LAGUNA_MIXED_ATTENTION_PROJECTIONS",
    "LAGUNA_MIXED_LOCAL32_FIXED_METADATA",
    "LAGUNA_MIXED_Q6_FIXED_METADATA",
    "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6A_ROLE_VARIANTS",
    "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6W_ROLE_VARIANTS",
    "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H6Z_ROLE_VARIANTS",
    "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_H7Y_ROLE_VARIANTS",
    "LAGUNA_PREFILL_DENSE_INITIAL_PREAPPEND_ROLE_VARIANTS",
    "LAGUNA_PREFILL_KV_PREAPPEND",
    "LAGUNA_PREFILL_MATRIX_ROWS",
    "LAGUNA_PREFILL_PREAPPEND_ROLE_VARIANTS",
    "LAGUNA_Q4_LM_HEAD_LOCAL32_FIXED_METADATA",
    "LAGUNA_Q5_FIXED_METADATA",
    "LAGUNA_Q5_F32_RESIDENT_GLOBAL_CACHE",
    "LAGUNA_Q5_SHARED_FIXED_METADATA",
    "LAGUNA_Q5_WAVE32X2_OUTPUT",
    "LAGUNA_Q5_WAVE32X2_QUERY_GATE",
    "LAGUNA_SELECTED_DOWN_MODE",
    "LAGUNA_SELECTED_GATE_UP_MODE",
    "LAGUNA_SPLIT_GATE_FUSION",
    "LAGUNA_SWA_DECODE_VARIANT",
    "LAGUNA_SWA_PREFILL_ROLE_VARIANTS",
    "LAGUNA_SWA_PREFILL_VARIANT",
    "LAGUNA_SWA_SPLIT_MIN_LIVE",
    "LAGUNA_SWA_SPLIT_TILE16_MIN_LIVE",
    "LAGUNA_SWA_SPLIT_WAVE_LOCAL",
    "GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS",
    "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS",
    "GGUF_DECODE_GRAPH_SUBMISSION_POLICIES",
    "GGUF_GDN_INDEXED_SINGLETON_DECODE",
    "GGUF_GDN_PREFILL_AUTO_MODE",
    "GGUF_GDN_PREFILL_AUTO_MODES_BY_QUANT_SHAPE",
    "GGUF_GDN_PREFILL_EXACT_MODE",
    "GGUF_LINEAR_ATTN_CONV_PREFILL_AUTO_MODE",
    "GGUF_PAGED_ATTN_PARALLEL_REDUCE",
    "GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT",
    "GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS",
    "GGUF_PREFILL_ROUTER_SELECT_THREADS",
    "GGUF_PREFILL_SCRATCH_LIVENESS_ALIAS",
    "GGUF_DENSE_PREFILL_SCRATCH_LIVENESS_POLICIES",
    "GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS",
    "GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS_BY_MAX_SEQUENCE_LENGTH",
    "GGUF_Q4_K_M_PREFILL_DECODE_POLICY",
    "GGUF_Q4_K_M_MAX_PREFILL_CHUNK_TOKENS",
    "GGUF_Q4_K_M_FAIR_PREFILL_BURST_CHUNKS",
    "GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE",
    "GGUF_F32_ORDERED_PREFILL_POLICIES",
    "GGUF_F32_ORDERED_PREFILL_QUANTS",
    "GGUF_Q5_F32_ORDERED_PREFILL",
    "GGUF_Q5_F32_ORDERED_PREFILL_H5Y_POLICY",
    "GGUF_Q5_F32_ORDERED_PREFILL_H7G_POLICY",
    "GGUF_Q5_F32_ORDERED_PREFILL_H7H_POLICY",
    "GGUF_Q5_F32_ORDERED_PREFILL_POLICY",
    "GGUF_Q6_F32_ORDERED_PREFILL",
    "GGUF_Q6_F32_ORDERED_PREFILL_H6E_POLICY",
    "GGUF_Q6_F32_ORDERED_PREFILL_H6U_POLICY",
    "GGUF_Q6_F32_ORDERED_PREFILL_POLICY",
    "GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP",
    "GGUF_DENSE_Q4_T16",
    "GGUF_Q4_T16_PHYSICAL_C1_ROWTILE_ROWS",
    "GGUF_Q4_T16_PHYSICAL_C1_ROWTILE_SHAPES",
    "GGUF_Q4_T16_PHYSICAL_SINGLE_WAVE_SHAPES",
    "GGUF_T16_NATIVE_ROWTILE_VARIANTS_BY_QUANT",
    "GGUF_Q4_T16_PHYSICAL_SINGLE_WAVE_MAX_ROWS",
    "GGUF_Q4_T16_PHYSICAL_SINGLE_WAVE_MAX_ROWS_BY_SHAPE",
    "GGUF_Q4_T16_PHYSICAL_SHARED_B_ROW64_SHAPES",
    "GGUF_Q4_T16_PHYSICAL_SHARED_B_ROW64_ROWS",
    "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_PROMPT_STREAMING",
    "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_EXTRA_ROWTILE_SHAPES",
    "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q5_ROWTILE_ROWS",
    "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q6_ROWTILE_ROWS",
    "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q6_MIXED_ROWTILE_CHUNKS",
    "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_Q6_MIXED_ROWTILE_SHAPES",
    "GGUF_SPECDEC2_PRODUCTION_PHYSICAL_EXACT_ROWTILE_ROWS",
    "GGUF_SPECDEC2_MTP2_MAX_REQUESTS",
    "GGUF_SPECDEC2_TARGET_VERIFY_PAD_ROW_COUNTS",
    "GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1",
    "GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1_GGML_TYPES",
    "GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA_POLICIES",
    "GGUF_PRIVATE_C1_DECODE_SCRATCH_ARENA_POLICIES",
    "GGUF_DENSE_PAIR_SILU_DECODE_POLICIES",
    "GGUF_DENSE_Q5_T16_SSM_OUT",
    "GGUF_DENSE_Q6_T16_QMICRO_PLANAR",
    "GGUF_DENSE_T16_F16_ROCBLAS_PREFILL_POLICIES",
    "GGUF_Q4_T16_F16_ROCBLAS_PREFILL_POLICIES",
    "GGUF_Q4_T16_UNEQUAL_PAIR_PREFILL_POLICIES",
    "GGUF_Q5_T16_F16_ROCBLAS_PREFILL_POLICIES",
    "GGUF_Q6_PLANAR_EXACT_PREFILL_VARIANTS",
    "GGUF_Q6_T16_F16_ROCBLAS_PREFILL_POLICIES",
    "GGUF_T16_F16_ROCBLAS_MAX_ROWS_BY_QUANT_SHAPE",
    "GGUF_T16_F16_ROCBLAS_PAIR_ONLY_POLICIES",
    "GGUF_T16_F16_ROCBLAS_SOLUTION_INDICES",
    "GGUF_T16_F16_ROCBLAS_SOLUTION_VERSION_PREFIX",
    "GGUF_T16_F16_ROCBLAS_VARIANT_POLICIES",
    "GGUF_T16_NATIVE_ROWTILE_MAX_ROWS_BY_QUANT",
    "GGUF_T16_C1_VARIANTS_BY_QUANT_SHAPE",
    "GGUF_LINEAR_RESIDUAL_MAX_ROWS_BY_QUANT",
    "GGUF_Q5_T16_SELECTED_QWEN_TILE8",
    "GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q6_LM_HEAD_MAX_CHUNK",
    "GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS",
    "GGUF_SPECDEC2_MTP2_C1",
    "GGUF_SPECDEC2_MTP2_C4",
    "GGUF_SPECDEC2_NATIVE_TARGET_GRAPH_MAX_CONTEXT",
    "GGUF_SPECDEC2_NATIVE_TARGET_MAX_CONTEXT",
    "PARO_SPECDEC2_MTP2_C1",
    "PARO_SPECDEC2_MTP2_C4",
    "GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS",
    "GGUF_Q8_T16_DECODE_ROWTILE_ALL",
    "GGUF_Q8_T16_DECODE_ROWTILE_MIN_ROWS",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS",
    "GGUF_RAW_K_PREFILL_COLTILE2_SHAPES",
    "GGUF_RAW_K_PREFILL_COLTILE_SUPPORTED",
    "GGUF_RAW_K_PREFILL_GENERIC_ROLE_VARIANTS",
    "GGUF_RAW_K_PREFILL_H7C_ROLE_VARIANTS",
    "GGUF_RAW_K_PREFILL_H7I_ROLE_VARIANTS",
    "GGUF_RAW_K_PREFILL_ROLE_VARIANTS",
    "GGUF_RAW_K_PREFILL_ROWBATCH",
    "GGUF_RAW_K_PREFILL_ROWBATCH_SUPPORTED",
    "GGUF_RAW_K_PREFILL_VARIANT",
    "GGUF_C2_PACKED_PREFILL_MAX_ROWS",
    "GGUF_ROUTER_F32_BF16_HIDDEN_THREADS",
]
