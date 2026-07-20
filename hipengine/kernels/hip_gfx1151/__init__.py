"""gfx1151 / Strix Halo backend registration.

The initial gfx1151 backend intentionally reuses the proven gfx11 HIP kernel
bodies from ``hip_gfx1100`` and compiles them as native ``gfx1151`` code objects
through ``HIPENGINE_HIP_ARCH=gfx1151`` / ``--offload-arch=gfx1151``.  This gives
Strix Halo a peer backend key while keeping tuning changes separate from the
source-lineage port.
"""

from __future__ import annotations

from importlib import import_module

from hipengine.kernels.backends import hip_target_arch_for_backend
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    qwen35_router_logits_bf16_f32w_auto_256,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
    gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out,
)
from hipengine.kernels.registry import (
    KernelKey,
    is_registered,
    register,
    registered_keys,
    resolve,
)

BACKEND = "hip_gfx1151"
TARGET_ARCH = hip_target_arch_for_backend(BACKEND)
# Clean SOL-G5 p512/d128 evidence admits the state-bound composite GGUF graph
# only when at least 128 decode transitions amortize capture/instantiate/close.
GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS = 128
# Clean LCP-2A six-case exactness, balanced-wall, and 250-transition natural
# gates admit compiler-cacheable compact-scale direct LDS32 GDN on gfx1151.
GGUF_GDN_PREFILL_AUTO_MODE = "chain_lds32_direct_nonvolatile"
# The architecture-scoped strict-exact selector resolves to the same proven
# nonvolatile direct route as gfx1151 production.
GGUF_GDN_PREFILL_EXACT_MODE = "chain_lds32_direct_nonvolatile"
# F3's independent-c1 and physical-width gates admit the one-token-per-row
# indexed GDN sibling for packed AR while retaining segmented GDN as fallback.
GGUF_GDN_INDEXED_SINGLETON_DECODE = True
# F3's canonical p512/d128 gate rejects automatic Q8T16 row amortization:
# one non-repeated prompt trajectory diverges consistently at c2/c4/c8 even
# though the shorter d64 screen passed. Keep the env-only diagnostic available.
GGUF_Q8_T16_DECODE_ROWTILE_ALL = False
# The repaired 128-thread pair-only route preserves production reduction order.
# Scope its small repeatable win to the independently gated physical-c8 shape;
# c2/c4 stay on their faster per-row schedule.
GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS = 8
# Exact dynamic expert-ID pairing removes duplicate C8 Q4T16 gate/up weight
# reads while keeping each row's production 128-thread reduction order.
# Physical widths below C8 remain on the established kernel.
GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS = 8
# The same exact dynamic expert-ID pairing is retained for Q5T16 selected-down
# only at physical C8; lower widths preserve the established kernel.
GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS = 8
# Three Q6T16 down layers use the independently gated exact sibling at C8.
GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS = 8
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
# Clean GPF-3A full-model 512/1K/4K evidence admits the byte-exact shared-X
# selected-dual Q4T16 prefill schedule on gfx1151.
GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE = "shared_x"
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
# Same-commit production-protocol 128K A/B rejects predecessor two-wave
# (382.041 vs 392.219 tok/s), so LCP-3 conservatively inherits its 64K ceiling.
GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS = 65536
# LCP-2B is admitted only on W7900/gfx1100. gfx1151 keeps the exact scalar
# compact-WMMA row read until its independent post-merge transfer gate.
GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS = 0
# LCP-D2 is admitted only on W7900/gfx1100. gfx1151 keeps the serial reduction
# until it receives an independent long-context correctness/performance gate.
GGUF_PAGED_ATTN_PARALLEL_REDUCE = False
GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT = 32768
# Clean PARO G3/G5 physical-width and server gates certify c4/c8 with whole-row
# full-attention execution. Diagnostic c2 row chunking changes row-local
# numerics at these widths and must therefore remain an explicit override.
PARO_FULL_ATTN_NATIVE_EXACT_WIDTHS = frozenset({4, 8})
# G5 retains p512/d128 blocking and SSE c1/c2/c4/c8 scaling, delayed c4->c8
# admission, serial-c8 control, and repeated c8 exactness. Package capabilities
# select those identity-matched widths by default without branching in model or
# engine code; the legacy env flags remain explicit rollback opt-outs.
PARO_RETAINED_BATCH_DEFAULTS = True
PARO_NATIVE_BATCH_DECODE_DEFAULT = True
_SOURCE_BACKEND = "hip_gfx1100"
# Native speculative-cycle providers use dedicated backend registrations rather
# than this generic shared-body alias refresh. The GGUF target launcher has an
# independent gfx1151 parity gate; the proposal graph remains unadmitted here.
_GFX1151_ALIAS_EXCLUSIONS = frozenset(
    {
        (
            "speculative_cycle",
            "w4_gguf",
            "native_v1_b2_target_graph",
        ),
        (
            "speculative_cycle",
            "w4_gguf",
            "native_v1_b2_proposal_graph",
        ),
    }
)
_GFX1151_OVERRIDES = {
    # The scalar-tree c1-exact kernel retained for gfx1100/PARO diverges from
    # gfx1151's established paged-c1 arithmetic at model scale. Keep gfx1151 on
    # the generic reduction, but pin its geometry to the c4/c8-proven 256-thread
    # shape: the generic rows<=2 1024-thread fast path diverges over p512/d128.
    (
        "paged_attn_decode",
        "w4_paro",
        "bf16_context_batch_c1_exact_spans",
    ): qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans,
    (
        "router_logits",
        "f32",
        "bf16_hidden",
    ): qwen35_router_logits_bf16_f32w_auto_256,
    (
        "linear",
        "gguf_q8_0_t16_v1",
        "wmma_prefill_bf16_bf16_out",
    ): gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out,
    (
        "linear",
        "gguf_q8_0_t16_v1",
        "t16_wmma_prefill_bf16_bf16_out",
    ): gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out,
}
_GFX1100_MODULES = (
    "hipengine.kernels.hip_gfx1100.attention",
    "hipengine.kernels.hip_gfx1100.convert",
    "hipengine.kernels.hip_gfx1100.fused",
    "hipengine.kernels.hip_gfx1100.linear",
    "hipengine.kernels.hip_gfx1100.linear_attn",
    "hipengine.kernels.hip_gfx1100.moe",
    "hipengine.kernels.hip_gfx1100.norm",
    "hipengine.kernels.hip_gfx1100.quant",
    "hipengine.kernels.hip_gfx1100.rotary",
    "hipengine.kernels.hip_gfx1100.runtime",
    "hipengine.kernels.hip_gfx1100.sampling",
    "hipengine.kernels.hip_gfx1100.smoke",
    "hipengine.kernels.hip_gfx1100.speculative",
    "hipengine.kernels.hip_gfx1100.wmma",
)


def register_gfx1151_kernels(*, replace: bool = False) -> None:
    """Register gfx1151 aliases for the current gfx1100 kernel key space."""

    for module_name in _GFX1100_MODULES:
        import_module(module_name)
    source_keys = [key for key in registered_keys() if key.backend == _SOURCE_BACKEND]
    for key in source_keys:
        if (key.layer, key.quant, key.variant) in _GFX1151_ALIAS_EXCLUSIONS:
            continue
        target_key = KernelKey(BACKEND, key.layer, key.quant, key.variant)
        if not replace and is_registered(target_key):
            continue
        source_fn = resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        register(
            target_key,
            _GFX1151_OVERRIDES.get((key.layer, key.quant, key.variant), source_fn),
            replace=replace,
        )


register_gfx1151_kernels()
register_backend_kernels = register_gfx1151_kernels

__all__ = [
    "BACKEND",
    "GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS",
    "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS",
    "GGUF_GDN_INDEXED_SINGLETON_DECODE",
    "GGUF_GDN_PREFILL_AUTO_MODE",
    "GGUF_GDN_PREFILL_EXACT_MODE",
    "GGUF_LINEAR_ATTN_CONV_PREFILL_AUTO_MODE",
    "GGUF_PAGED_ATTN_PARALLEL_REDUCE",
    "GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT",
    "GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS",
    "GGUF_PREFILL_ROUTER_SELECT_THREADS",
    "GGUF_Q4_K_M_FAIR_PREFILL_BURST_CHUNKS",
    "GGUF_Q4_K_M_MAX_PREFILL_CHUNK_TOKENS",
    "GGUF_Q4_K_M_PREFILL_DECODE_POLICY",
    "GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS",
    "GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE",
    "GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS",
    "GGUF_Q8_T16_DECODE_ROWTILE_ALL",
    "GGUF_Q8_T16_PREFILL_FOUR_WAVE",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS",
    "GGUF_ROUTER_F32_BF16_HIDDEN_THREADS",
    "PARO_FULL_ATTN_NATIVE_EXACT_WIDTHS",
    "PARO_NATIVE_BATCH_DECODE_DEFAULT",
    "PARO_RETAINED_BATCH_DEFAULTS",
    "TARGET_ARCH",
    "register_backend_kernels",
    "register_gfx1151_kernels",
]
