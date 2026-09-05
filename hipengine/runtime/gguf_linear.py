"""Registry-driven GGUF linear dispatch helpers."""

from __future__ import annotations

import contextlib
import ctypes
import os
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterator, Mapping

from hipengine.core.dtype import DType
from hipengine.core.hip import get_hip_runtime
from hipengine.core.specdec2_scope import (
    physical_exact_rowtiles_enabled,
    q4_t16_physical_extra_rowtiles_enabled,
)
from hipengine.kernels.backends import (
    backend_package_capability,
    load_backend_kernel_package,
)
from hipengine.kernels.hip_gfx1100.linear.dense_gemv import register_dense_gemv_kernels
from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
    gguf_q8_0_dual_gemv_bf16_bf16_out,
    register_gguf_k_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    gguf_q4_k_pack8_dual_prefill_bf16_bf16_out,
    gguf_q4_k_pack8_rowtile_bf16_bf16_out,
    gguf_q4_k_quantize_bf16_q8_1,
    gguf_q4_k_quantize_bf16_q8_1x2,
    register_gguf_q4_k_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill import (
    gguf_q4_k_wmma_prefill_dual_bf16_bf16_out,
    register_gguf_q4_k_prefill_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_pack8_gemv import (
    register_gguf_q4_k_pack8_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q5_k_f32_rocblas_prefill import (
    q5_k_f32_activation_tile_k_row_nbytes,
    q5_k_f32_ordered_workspace_nbytes,
    register_gguf_q5_k_f32_rocblas_prefill_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_f16_rocblas_prefill import (
    q6_k_f16_input_nbytes,
    q6_k_f16_output_nbytes,
    q6_k_f16_weight_nbytes,
    register_gguf_q6_k_f16_rocblas_prefill_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv import (
    register_gguf_q6_k_pack8_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
    register_gguf_q6_k_t16_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
    q6_dense_integer_mmq_workspace,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    register_gguf_k_t16_selected_prefill_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q5_k_qmicro_planar_gemv import (
    gguf_q5_k_qmicro_planar_q8_1_dp4a_grouped_bf16_bf16_out,
    register_gguf_q5_k_qmicro_planar_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_mmq_prefill import (
    gguf_q8_1_d4s4_f32_quantize_bf16,
    gguf_q8_1_d4s4_f32_quantize_bf16_kmajor,
    q8_1_d4s4_f32_kmajor_nbytes,
    q8_1_d4s4_f32_nbytes,
    register_gguf_k_mmq_prefill_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import (
    Q8MMQPrefillPolicy,
    gguf_q8_0_mmq128_quantize_bf16_d4x3,
    gguf_q8_0_mmq128_sparse_exact_correct_bf16,
    q8_mmq_d4x3_nbytes,
    register_gguf_q8_0_mmq_prefill_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_pack8_gemv import (
    register_gguf_q8_0_pack8_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
    gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out,
    register_gguf_q8_0_prefill_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv import (
    gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out,
    gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out,
    gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out,
    gguf_q8_0_t16_dual_gemv_decode_rowtile4_col8_bf16_bf16_out,
    gguf_q8_0_t16_gemv_decode_rowtile4_bf16_bf16_out,
    gguf_q8_0_t16_triple_gemv_decode_rowtile4_bf16_bf16_out,
    gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out,
    register_gguf_q8_0_t16_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
    register_gguf_q8_0_t16_prefill_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    register_gguf_t16_selected_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.runtime.laguna_launch_batch import (
    register_laguna_launch_batch_kernels,
)
from hipengine.kernels.registry import KernelKey, generation, is_registered, resolve
from hipengine.loading.qwen35_gguf_materialize import (
    LAYOUT_DENSE_BF16,
    LAYOUT_DENSE_F32,
    LAYOUT_GGUF_Q4_K_QMICRO_T16,
    LAYOUT_GGUF_Q4_K_T16,
    LAYOUT_GGUF_Q5_K_T16,
    LAYOUT_GGUF_Q6_K_T16,
    LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
    LAYOUT_GGUF_Q8_0_T16,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
    Q4_T16_DECODE_TILES,
    Q4_T16_DECODE_TILES_R3PLUS,
)
from hipengine.runtime.gguf_weight import GGUFDeviceWeight

GGUF_ACTIVATION_BF16 = "bf16"
GGUF_ACTIVATION_F32 = "f32"
GGUF_OUTPUT_BF16 = "bf16"
GGUF_OUTPUT_FP16 = "fp16"
GGUF_OUTPUT_F32 = "f32"

# Opt-in env var for the GGUF WMMA batched prefill family (P8). See
# docs/GGUF.md "P8: real batched prefill GEMM" for the wider plan.
_WMMA_PREFILL_ENV = "HIPENGINE_GGUF_WMMA_PREFILL"

# Session-scoped override; runners can flip this on entry to their bulk
# prefill paths (e.g. from ``PrefillConfig.use_wmma_prefill``). Stays
# ``None`` until set, so the env var still controls the default for plain
# bench/diagnostic invocations.
_wmma_prefill_session_enabled: bool | None = None

# Opt-in env var for the GGUF pack8 GEMV decode family (P9.B). See
# docs/GGUF.md "P9: closing the qwen35moe gap to PARO" for the wider
# plan. This toggles the ``rows == 1`` decode rewrite that routes single-
# token projections through the new ``pack8_gemv_decode_*`` kernels
# (P9.B1-P9.B4b) instead of the legacy ``pack8_gemv_*`` decoders.
_GEMV_DECODE_ENV = "HIPENGINE_GGUF_GEMV_DECODE"
_gemv_decode_session_enabled: bool | None = None
_native_batch_decode_session_enabled = False
_target_verifier_rowtile_session_enabled: ContextVar[bool] = ContextVar(
    "gguf_target_verifier_rowtile_session_enabled",
    default=False,
)
_target_verifier_production_q4_rowtile_session_enabled: ContextVar[bool] = (
    ContextVar(
        "gguf_target_verifier_production_q4_rowtile_session_enabled",
        default=False,
    )
)
_target_verifier_rowtile_chunk_child_enabled: ContextVar[bool] = ContextVar(
    "gguf_target_verifier_rowtile_chunk_child_enabled",
    default=False,
)
_target_verifier_wide_q6_shared4_policy_enabled: ContextVar[bool] = ContextVar(
    "gguf_target_verifier_wide_q6_shared4_policy_enabled",
    default=False,
)
_target_verifier_wide_q6_shared4_leaf_enabled: ContextVar[bool] = ContextVar(
    "gguf_target_verifier_wide_q6_shared4_leaf_enabled",
    default=False,
)
TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ENV = (
    "HIPENGINE_GGUF_VERIFY_PRODUCTION_Q4_ROWTILE"
)
TARGET_VERIFIER_WIDE_Q6_SHARED4_ENV = "HIPENGINE_GGUF_VERIFY_WIDE_Q6_SHARED4"

# B1 mechanism-A transfer (docs/QWEN38-GFX1151-BUILD-CAMPAIGN.md): route the
# packed MTP serving target verifier's rows>1 projections through the same
# retained exact prefill band owners the prefill path uses, instead of the
# July-2026 small-B per-row GEMV owners (9cceedbcc). Default ON for the
# production execution profile (retained 2026-09-02 after the B1 gates:
# one-group suite +56.3/+64.0/+69.5/+72.7 pct C5-C8 with 40/40 exact and
# identical IDs; sec-6 teacher-forced logits gate top-1 100 pct, max KL
# 6.5e-4; production-admission measured inert). Strict (and any profile
# fallback) keeps the GEMV verifier oracle unchanged. The env remains an
# explicit override for bisection and diagnostics: 1/on forces the transfer,
# 0/off restores the GEMV owners everywhere.
MTP_SERVING_TARGET_WMMA_PREFILL_ENV = (
    "HIPENGINE_GGUF_MTP_SERVING_TARGET_WMMA_PREFILL"
)


# B2 P1: prefill F16-staging route (docs/QWEN38-GFX1151-BUILD-CAMPAIGN.md).
# Default OFF: the BF16 owners remain the selected strict fallback. When
# enabled (env or prefill session), Q4/Q5-T16 dense prefill launches stage
# the activation operand into a bounded session-owned IEEE-half workspace,
# then dispatch
# the registered fp16-in siblings (measured 0.69-0.89x their BF16 owners at
# the kernel level). The cast kernel and siblings are exact-family; outputs
# are T1 and require the complete B2 item-3 gates before any default flip.
PREFILL_F16_STAGING_ENV = "HIPENGINE_GGUF_PREFILL_F16_STAGING"
_prefill_f16_staging_session_enabled: ContextVar[bool] = ContextVar(
    "gguf_prefill_f16_staging_session_enabled", default=False
)
_PREFILL_F16_STAGING_MIN_ROWS = 17
PREFILL_F16_STAGING_MAX_ROWS = 1024
_PREFILL_F16_STAGING_QUANTS = frozenset(
    {"gguf_q4_k_t16_v1", "gguf_q5_k_t16_v1"}
)
_PREFILL_F16_STAGING_SOURCE_VARIANT = "t16_wmma_prefill_bf16_bf16_out"
PREFILL_F16_STAGING_VARIANT = "t16_wmma_prefill_fp16_in_bf16_out"

# B5: production-default changed-arithmetic planar-Q6 integer MMQ. The registered
# gfx1151 variant is selected only inside a caller-owned workspace context and
# the backend package's exact row/shape policy. A remains the strict fallback.
Q6_INTEGER_MMQ_PREFILL_ENV = "HIPENGINE_GGUF_Q6_INTEGER_MMQ_PREFILL"


def q6_integer_mmq_for(
    profile: object = None,
    *,
    profile_fell_back_to_strict: bool = False,
) -> bool:
    """Resolve the retained production default with an explicit env override."""

    override = os.environ.get(Q6_INTEGER_MMQ_PREFILL_ENV, "").strip().lower()
    if override:
        return override in {"1", "true", "yes", "on"}
    profile_value = getattr(profile, "value", profile)
    if profile_value is None or str(profile_value) == "":
        return False
    return str(profile_value) == "production" and not bool(
        profile_fell_back_to_strict
    )


@dataclass(frozen=True)
class PrefillF16StagingWorkspace:
    """Bounded device workspace owned by the active resident session."""

    ptr: int
    nbytes: int

    def __post_init__(self) -> None:
        if int(self.ptr) <= 0 or int(self.nbytes) <= 0:
            raise ValueError("prefill F16 staging workspace must be non-empty")


_prefill_f16_staging_workspace: ContextVar[
    PrefillF16StagingWorkspace | None
] = ContextVar("gguf_prefill_f16_staging_workspace", default=None)


@contextlib.contextmanager
def prefill_f16_staging_session(
    enabled: bool = True,
    *,
    workspace_ptr: int = 0,
    workspace_nbytes: int = 0,
) -> Iterator[None]:
    """Enable F16 staging with one caller-owned bounded device workspace."""

    workspace = None
    if enabled and (int(workspace_ptr) or int(workspace_nbytes)):
        workspace = PrefillF16StagingWorkspace(
            ptr=int(workspace_ptr),
            nbytes=int(workspace_nbytes),
        )
    enabled_token = _prefill_f16_staging_session_enabled.set(bool(enabled))
    workspace_token = _prefill_f16_staging_workspace.set(workspace)
    try:
        yield
    finally:
        _prefill_f16_staging_workspace.reset(workspace_token)
        _prefill_f16_staging_session_enabled.reset(enabled_token)


def prefill_f16_staging_enabled(default: bool = False) -> bool:
    """Whether the prefill F16-staging route is active for this launch."""

    if _prefill_f16_staging_session_enabled.get():
        return True
    override = os.environ.get(PREFILL_F16_STAGING_ENV, "").strip().lower()
    if override:
        return override in {"1", "true", "yes", "on"}
    return bool(default)


def prefill_f16_staging_for(
    profile: object = None,
    *,
    profile_fell_back_to_strict: bool = False,
) -> bool:
    """Resolve the profile default while preserving explicit env overrides."""

    override = os.environ.get(PREFILL_F16_STAGING_ENV, "").strip().lower()
    if override:
        return override in {"1", "true", "yes", "on"}
    profile_value = getattr(profile, "value", profile)
    if profile_value is None or str(profile_value) == "":
        return False
    return str(profile_value) == "production" and not bool(
        profile_fell_back_to_strict
    )


def prefill_f16_staging_workspace() -> PrefillF16StagingWorkspace | None:
    """Return the active caller-owned workspace, if one was supplied."""

    return _prefill_f16_staging_workspace.get()


def _prefill_f16_stage_ptr(count: int) -> int:
    """Return the bounded owner pointer, or zero to keep the strict fallback."""

    workspace = prefill_f16_staging_workspace()
    required_nbytes = int(count) * 2
    if workspace is None or int(workspace.nbytes) < required_nbytes:
        return 0
    return int(workspace.ptr)


def _launch_prefill_f16_staged(
    fn,
    weight: GGUFDeviceWeight,
    x_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    kwargs,
    *,
    backend: str,
    quant: str,
    layer: str,
    runtime,
) -> bool:
    """Stage x as IEEE half and dispatch the fp16-in sibling; False = skip."""

    from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
        gguf_cast_bf16_to_f16,
    )

    key = KernelKey(
        backend, layer, quant, PREFILL_F16_STAGING_VARIANT
    )
    if not is_registered(key):
        return False
    sibling = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    count = int(rows) * int(in_features)
    stage_ptr = _prefill_f16_stage_ptr(count)
    if stage_ptr <= 0:
        return False
    gguf_cast_bf16_to_f16(
        x_ptr,
        stage_ptr,
        count,
        stream=kwargs.get("stream", 0),
        runtime=runtime,
    )
    stage_kwargs = dict(kwargs)
    stage_kwargs.pop("library", None)
    _LAUNCH_ABI["t16"](
        sibling,
        weight,
        stage_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stage_kwargs,
    )
    return True


def mtp_serving_target_use_wmma_prefill(
    profile: object = None,
    *,
    profile_fell_back_to_strict: bool = False,
) -> bool:
    """Whether MTP serving target verify passes use the prefill band owners.

    ``profile`` accepts the generator's execution-profile value (or its
    string). Resolution order: explicit env override, then the production
    profile (without strict fallback), then off. Without profile context the
    answer is off so unrelated callers never drift onto the transferred
    owners.
    """

    override = os.environ.get(
        MTP_SERVING_TARGET_WMMA_PREFILL_ENV, ""
    ).strip().lower()
    if override:
        return override in {"1", "true", "yes", "on"}
    profile_value = getattr(profile, "value", profile)
    if profile_value is None or str(profile_value) == "":
        return False
    return str(profile_value) == "production" and not bool(
        profile_fell_back_to_strict
    )

# Small-B weight-amortized row-tile GEMV for raw K-quants and resident-pack8
# Q4_K verifier continuation blocks. Default ON: every specialization preserves
# the corresponding per-row arithmetic. The opt-out exists only for bisection;
# set HIPENGINE_GGUF_Q4K_ROWTILE=0 to disable.
_Q4K_ROWTILE_ENV = "HIPENGINE_GGUF_Q4K_ROWTILE"
_q4k_rowtile_session_enabled: bool | None = None
_ROWTILE_MIN_ROWS = 2
_ROWTILE_MAX_ROWS = 8
# Decode-regime upper bound for native rowtile chunking: any native-session
# concurrency below this is decomposed into <=8-row rowtile8 groups so no
# single Q4/Q5 projection silently falls back to WMMA prefill. rows >= 512 is
# the bulk-prefill regime and stays on WMMA.
_NATIVE_ROWTILE_CHUNK_MAX_ROWS = 512
_DENSE_BF16_ROWTILE_MIN_ROWS = 2
_DENSE_BF16_ROWTILE_MAX_ROWS = 4
# The exact local128 virtual-partition schedule wins through K=10,240 on
# gfx1100, then loses to the retained local256 block from K=12,288 onward.
_DENSE_BF16_VIRTUAL256_MAX_IN_FEATURES = 10_240
# The small-row local128 screen is positive at every row only for the exact
# Qwen3.6 linear-attention output projection shape.
_DENSE_BF16_VIRTUAL256_ROWTILE_IN_FEATURES = 6_144
_DENSE_BF16_VIRTUAL256_ROWTILE_OUT_FEATURES = 5_120
_PACK8_ROWTILE_MIN_ROWS = 2
_PACK8_ROWTILE_MAX_ROWS = 4
_PACK8_DUAL_ROWTILE_SILU_IN_FEATURES = 5_120
_PACK8_DUAL_ROWTILE_SILU_OUT_FEATURES = 17_408
# Fused dual+SiLU gate/up owner floor. 512 was a perf-qualification floor, not a
# correctness boundary: the fused kernel is bit-identical to the
# two-singleton+silu_mul chain at the only shape this predicate admits
# (5120 -> 17408) for rows 45/96/192/511/512 and on the fixture down to rows 2
# (tests/test_gguf_q4_k_t16_dense.py). W7900 Qwen3.8-27B-Q4_K_M measured +4.2%
# (45 rows), +4.2% (96), +4.9% (192) prefill and unchanged 512, so the floor came
# down to 33 - one above the largest row count measured SLOWER for this owner.
# Target verification runs the same shared FFN stage inside captured physical
# groups of 16/32 rows (C4-C8 at K3), where the fused owner measured 8.2% slower
# per cycle across three replications even though the identical-shape prefill A/B
# is faster; rows<=32 therefore keep the unfused chain until a prefill-context
# selector exists (docs/REFACTOR.md). The floor also stays clear of the
# dedicated small-B rowtile/GEMV domain (rows<=8).
_Q4_T16_DUAL_WMMA_SILU_MIN_ROWS = 33
_Q4_T16_DUAL_SILU_RETILE_ENV = "HIPENGINE_GGUF_Q4_T16_DUAL_SILU_RETILE"
_Q4_T16_DUAL_SILU_RETILE_RESOLVED: bool | None = None
_rowtile_variant_policy_env_cache: dict[tuple[str, bool], bool] = {}
_Q4_QMICRO_T16_EXPANDED_META_MIN_ROWS = 4_096
_Q4_T16_DENSE_QUANTS = frozenset(
    {"gguf_q4_k_t16_v1", "gguf_q4_k_qmicro_t16_v1"}
)
_Q4_T16_DENSE_ROWTILE_MAX_ROWS_BY_QUANT = MappingProxyType(
    {
        "gguf_q4_k_t16_v1": 8,
        "gguf_q4_k_qmicro_t16_v1": 4,
    }
)
# Unequal dual-WMMA prefill owner floor (QKV shape 5120 -> 10240/6144). Unlike the
# shared linear_pair_silu gate, this route is ContextVar-scoped to the resident
# prefill entry (q4_t16_unequal_pair_prefill_session, opened only by
# Qwen35GGUFResidentSession.prefill), so the floor does not have to hold back
# captured target verification. 512 was a perf-qualification floor: outputs are
# bit-identical to the two singletons at the dispatched shape for rows 16/24/32/
# 45/96/512, and the W7900 Qwen3.8-27B-Q4_K_M shipping route measured +1.7/+1.9/
# +1.8/+2.1/+1.8/+1.7% prefill at rows 16/24/32/45/96/192 with identical generated
# ids in every cell. The floor stays above the rows<=8 GEMV/rowtile decode owners.
# The constant is global, but the route is additionally capability-gated by
# GGUF_Q4_T16_UNEQUAL_PAIR_PREFILL_POLICIES (currently only the gfx1100 dense
# H5120 MOSTLY_Q4_K_M identity), so other backends keep the legacy singletons.
_Q4_T16_UNEQUAL_DUAL_WMMA_MIN_ROWS = 16
_Q4_T16_UNEQUAL_DUAL_WMMA_SHAPE = (5_120, 10_240, 6_144)
_Q4_T16_COL4_ALL_ROWS_SHAPES = frozenset({(5_120, 1_024)})
_Q4_T16_DENSE_PAIR_SILU_Q8_1X2_VARIANT = (
    "dense_dual_q8_1x2_dp4a_bf16_bf16_out"
)
_Q4_T16_DENSE_PAIR_SILU_SPLIT_WEIGHT_VARIANT = (
    "dense_dual_q8_1x2_split_weight_dp4a_bf16_bf16_out"
)
_Q4_T16_DENSE_PAIR_SILU_Q8_1X2_ROWTILE8_VARIANT = (
    "dense_dual_q8_1x2_rowtile8_dp4a_bf16_bf16_out"
)
_Q4_T16_DENSE_PAIR_SILU_Q8_1X2_VARIANTS = frozenset(
    {
        _Q4_T16_DENSE_PAIR_SILU_Q8_1X2_VARIANT,
        _Q4_T16_DENSE_PAIR_SILU_SPLIT_WEIGHT_VARIANT,
        _Q4_T16_DENSE_PAIR_SILU_Q8_1X2_ROWTILE8_VARIANT,
    }
)
_PACK8_EXACT_PREFILL_MIN_ROWS = 512
_ROWTILE_SUPPORTED_PREFILL_VARIANTS = frozenset(
    {"prefill_bf16_bf16_out", "prefill_bf16_f32_out", "prefill_f32_f32_out"}
)
# Raw-layout quants that ship a ``rowtile_*`` family, with their K-block
# alignment (Q8_0 is 32-wide; the K-quants are 256-wide). Q8_0 is the dense
# projection quant for qwen35moe (attn_qkv/gate, ssm_out); the K-quants cover
# other GGUF dense weights.
_ROWTILE_QUANT_BLOCKS: Mapping[str, int] = {
    "gguf_q4_k": 256,
    "gguf_q5_k": 256,
    "gguf_q6_k": 256,
    "gguf_q8_0": 32,
}

# Large-prefill exact row reuse for raw Q5_K/Q6_K. Execution owners select a
# fixed 4/8/16/32-row slab through the context-local policy below; zero is
# the scalar fallback. Keeping the policy context-local avoids backend/quant
# branches in model code and remains safe for concurrent request owners.
_RAW_K_PREFILL_ROWBATCHES = frozenset({0, 4, 8, 16, 32})
_RAW_K_PREFILL_ROWBATCH_QUANTS = frozenset({"gguf_q5_k", "gguf_q6_k"})
_RAW_K_PREFILL_ROWBATCH_VARIANTS = frozenset(
    {"prefill_bf16_bf16_out", "prefill_bf16_f32_out"}
)
_RAW_K_PREFILL_VARIANTS = frozenset({"rowbatch", "coltile"})
_raw_k_prefill_rowbatch: ContextVar[int] = ContextVar(
    "raw_k_prefill_rowbatch",
    default=0,
)
_raw_k_prefill_variant: ContextVar[str] = ContextVar(
    "raw_k_prefill_variant",
    default="rowbatch",
)

# Quants currently shipping a batched ``wmma_prefill_*`` family. Values are
# the raw GGUF K-block alignment constraints enforced before dispatching to
# the WMMA wrappers. Q4_K is raw-layout only for now: dense 2D Q4_K resident
# weights are still materialized as the lossless pack8 fallback layout, so
# they never reach the raw WMMA ABI unless a caller explicitly has raw bytes.
_WMMA_PREFILL_QUANT_BLOCKS: Mapping[str, int] = {
    "gguf_q8_0": 32,
    "gguf_q4_k": 256,
    "gguf_q6_k": 256,
    # Dense T16 WMMA consumers keep the source quant's K-block alignment.
    "gguf_q5_k_t16_v1": 256,
    "gguf_q6_k_t16_v1": 256,
    "gguf_q6_k_t16_qmicro_planar_v1": 256,
    "gguf_q8_0_t16_v1": 32,
}

# Q8_0 T16 decode scheduling. Backend packages select independently retained
# defaults; explicit env values remain diagnostic/rollback overrides. The
# thread-width hook stays separate from selected-MoE T16 dp4a scheduling.
_Q8_T16_THREADS_ENV = "HIPENGINE_GGUF_Q8_T16_THREADS"
_Q8_T16_ALLOWED_THREADS = frozenset({64, 128})
# Match the production Q8T16 per-output reduction partition. The earlier
# 64-thread rowtile rounded synthetic fixtures identically but diverged by one
# BF16 ULP on real packed-AR activations.
_Q8_T16_ROWTILE_THREADS = 128
_Q8_T16_PAIR_ROWTILE_ENV = "HIPENGINE_GGUF_Q8_T16_PAIR_ROWTILE"
_Q8_T16_PAIR_COL8_ENV = "HIPENGINE_GGUF_Q8_T16_PAIR_COL8"
_Q8_T16_ROWTILE_ALL_ENV = "HIPENGINE_GGUF_Q8_T16_ROWTILE_ALL"
_q8_t16_pair_rowtile_min_rows_session: int | None = None
_q8_t16_rowtile_all_session_enabled: bool | None = None
_Q8_T16_QWEN35_ATTN_QKV_OUT = 8192
_Q8_T16_QWEN35_ATTN_GATE_OUT = 4096
_Q8_T16_QWEN35_ATTN_IN = 2048


@dataclass(frozen=True)
class GGUFLinearDispatch:
    """Resolved kernel key and ABI family for one GGUF linear launch."""

    key: KernelKey
    abi: str


@dataclass(frozen=True)
class _Q8MMQPrefillSession:
    workspace_ptr: int
    workspace_nbytes: int
    risk_count_ptr: int
    risk_count_nbytes: int
    risk_indices_ptr: int
    risk_indices_nbytes: int
    library: ctypes.CDLL | None
    policy: Q8MMQPrefillPolicy


_q8_mmq_prefill_session: ContextVar[_Q8MMQPrefillSession | None] = ContextVar(
    "q8_mmq_prefill_session",
    default=None,
)


@dataclass(frozen=True)
class _Q5RawMMQTargetSession:
    workspace_ptr: int
    workspace_nbytes: int
    library: ctypes.CDLL | None
    quant_library: ctypes.CDLL | None
    source_layout: bool
    planar_dp4a: bool = False
    planar_library: ctypes.CDLL | None = None


_q5_raw_mmq_target_session: ContextVar[_Q5RawMMQTargetSession | None] = (
    ContextVar("q5_raw_mmq_target_session", default=None)
)


@dataclass(frozen=True)
class Q5F32ResidentPlane:
    """Immutable exact F32 plane selected by its resident raw-Q5 pointer."""

    raw_weight_ptr: int
    weight_f32_ptr: int
    weight_f32_nbytes: int
    in_features: int
    out_features: int
    output_dtype: str
    weight_layout: str
    col_tile: int
    row_batch: int

    def __post_init__(self) -> None:
        raw_ptr = int(self.raw_weight_ptr)
        plane_ptr = int(self.weight_f32_ptr)
        hidden = int(self.in_features)
        outputs = int(self.out_features)
        col_tile = int(self.col_tile)
        row_batch = int(self.row_batch)
        if raw_ptr <= 0:
            raise ValueError("resident Q5 raw-weight pointer must be positive")
        if plane_ptr <= 0 or plane_ptr % 16:
            raise ValueError(
                "resident Q5 F32 plane must be a positive 16-byte-aligned pointer"
            )
        if hidden <= 0 or hidden % 256:
            raise ValueError("resident Q5 in_features must be a positive multiple of 256")
        if outputs <= 0 or col_tile <= 0 or outputs % col_tile:
            raise ValueError("resident Q5 out_features must fit its positive col tile")
        if row_batch <= 0:
            raise ValueError("resident Q5 row batch must be positive")
        if self.output_dtype not in {GGUF_OUTPUT_BF16, GGUF_OUTPUT_F32}:
            raise ValueError("resident Q5 output dtype must be bf16 or f32")
        if self.weight_layout != "tile_k_col":
            raise ValueError("resident Q5 weight layout must be tile_k_col")
        expected_nbytes = hidden * outputs * DType.FP32.itemsize
        if int(self.weight_f32_nbytes) != expected_nbytes:
            raise ValueError(
                "resident Q5 F32 plane extent must exactly match K*N*4 bytes"
            )

    @property
    def ordered_geometry(self) -> str:
        return (
            f"weight_major_{self.weight_layout}_activation_tile_k_row_"
            f"padded_compute_coltile{self.col_tile}_rowbatch{self.row_batch}"
        )

    def matches(
        self,
        *,
        raw_weight_ptr: int,
        in_features: int,
        out_features: int,
        output_dtype: str,
        ordered_geometry: str,
    ) -> bool:
        return (
            int(self.raw_weight_ptr) == int(raw_weight_ptr)
            and int(self.in_features) == int(in_features)
            and int(self.out_features) == int(out_features)
            and self.output_dtype == str(output_dtype)
            and self.ordered_geometry == str(ordered_geometry)
        )


@dataclass(frozen=True)
class Q5F32OrderedPrefillSession:
    """Caller-owned transient and optional immutable resident Q5 F32 planes."""

    max_rows: int
    weight_f32_ptr: int
    weight_f32_nbytes: int
    library: object
    min_rows: int = 512
    activation_bf16_ptr: int = 0
    activation_bf16_nbytes: int = 0
    resident_weight_f32_planes: Mapping[int, Q5F32ResidentPlane] | None = None

    def __post_init__(self) -> None:
        if not 0 < int(self.min_rows) <= int(self.max_rows):
            raise ValueError(
                "Q5 F32 ordered min_rows must be positive and fit max_rows"
            )
        if int(self.weight_f32_ptr) <= 0 or int(self.weight_f32_nbytes) <= 0:
            raise ValueError(
                "Q5 F32 ordered weight plane must be a non-empty device buffer"
            )
        has_activation_ptr = int(self.activation_bf16_ptr) > 0
        has_activation_nbytes = int(self.activation_bf16_nbytes) > 0
        if has_activation_ptr != has_activation_nbytes:
            raise ValueError(
                "Q5 F32 ordered activation plane pointer/bytes must be both set or zero"
            )
        normalized: dict[int, Q5F32ResidentPlane] = {}
        for raw_ptr, plane in (self.resident_weight_f32_planes or {}).items():
            key = int(raw_ptr)
            if not isinstance(plane, Q5F32ResidentPlane):
                raise TypeError("resident Q5 plane map values must be Q5F32ResidentPlane")
            if key != int(plane.raw_weight_ptr):
                raise ValueError("resident Q5 plane map key must equal raw-weight pointer")
            if key in normalized:
                raise ValueError("resident Q5 plane map contains a duplicate raw pointer")
            normalized[key] = plane
        object.__setattr__(
            self,
            "resident_weight_f32_planes",
            MappingProxyType(normalized),
        )

    def resident_plane(
        self,
        raw_weight_ptr: int,
        *,
        in_features: int,
        out_features: int,
        output_dtype: str,
        ordered_geometry: str,
    ) -> Q5F32ResidentPlane | None:
        planes = self.resident_weight_f32_planes
        assert planes is not None
        plane = planes.get(int(raw_weight_ptr))
        if plane is None or not plane.matches(
            raw_weight_ptr=raw_weight_ptr,
            in_features=in_features,
            out_features=out_features,
            output_dtype=output_dtype,
            ordered_geometry=ordered_geometry,
        ):
            return None
        return plane


_q5_f32_ordered_prefill_session: ContextVar[
    Q5F32OrderedPrefillSession | None
] = ContextVar("q5_f32_ordered_prefill_session", default=None)


@dataclass(frozen=True)
class T16F16RocblasPrefillSession:
    """Caller-owned transient planes for changed-arithmetic T16 prefill.

    This context never changes resident weight ownership: admitted candidates
    read canonical Q4T16/Q5T16 or planar Q6T16 allocations and dequantize one
    bounded output tile immediately before rocBLAS consumes it.
    """

    min_rows: int
    max_rows: int
    x_f16_ptr: int
    x_f16_nbytes: int
    weight_f16_ptr: int
    weight_f16_nbytes: int
    out_f16_ptr: int
    out_f16_nbytes: int
    tile_out_features_by_shape: Mapping[tuple[int, ...], int]
    dequant_library: object
    cast_library: object
    rocblas: object
    solution_indices_by_gemm_shape: Mapping[tuple[int, int, int], int] | None = None
    q4_tile_out_features_by_shape: Mapping[tuple[int, ...], int] | None = None
    q5_tile_out_features_by_shape: Mapping[tuple[int, ...], int] | None = None
    q4_x_inplace_shapes: frozenset[tuple[int, ...]] = frozenset()
    q5_x_inplace_shapes: frozenset[tuple[int, ...]] = frozenset()
    x_inplace_shapes: frozenset[tuple[int, ...]] = frozenset()
    max_rows_by_quant_shape: Mapping[
        str, Mapping[tuple[int, int], int]
    ] | None = None
    linear_variant_intervals_by_quant: Mapping[
        str, Mapping[tuple[int, int], Mapping[tuple[int, int], str]]
    ] | None = None
    pair_only_second_operand_policies: Mapping[
        tuple[str, int, int, str, int],
        Mapping[tuple[int, int], tuple[int, str, bool]],
    ] | None = None

    def __post_init__(self) -> None:
        if not 1 < int(self.min_rows) <= int(self.max_rows):
            raise ValueError(
                "T16 F16/rocBLAS min_rows must exceed one and fit max_rows"
            )
        for label, ptr, nbytes in (
            ("activation", self.x_f16_ptr, self.x_f16_nbytes),
            ("weight", self.weight_f16_ptr, self.weight_f16_nbytes),
            ("output", self.out_f16_ptr, self.out_f16_nbytes),
        ):
            if int(ptr) <= 0 or int(nbytes) <= 0:
                raise ValueError(f"T16 F16/rocBLAS {label} plane must be non-empty")

        def normalize_policy(
            raw_policy: Mapping[tuple[int, ...], int], *, required: bool
        ) -> Mapping[tuple[int, ...], int]:
            normalized: dict[tuple[int, ...], int] = {}
            for raw_shape, raw_tile in raw_policy.items():
                if len(raw_shape) not in {2, 3}:
                    raise ValueError(
                        "T16 F16/rocBLAS policies require (K, N) or (M, K, N) shapes"
                    )
                shape = tuple(int(value) for value in raw_shape)
                rows, hidden, outputs = (
                    (None, shape[0], shape[1])
                    if len(shape) == 2
                    else (shape[0], shape[1], shape[2])
                )
                tile = int(raw_tile)
                if (
                    (rows is not None and rows <= 0)
                    or hidden <= 0
                    or hidden % 256
                    or outputs <= 0
                ):
                    raise ValueError("T16 F16/rocBLAS policy shapes must be valid")
                if tile <= 0 or tile % 16 or tile > outputs or outputs % tile:
                    raise ValueError(
                        "T16 F16/rocBLAS output tiles must positively divide N"
                    )
                normalized[shape] = tile
            if required and not normalized:
                raise ValueError("T16 F16/rocBLAS requires at least one shape policy")
            return MappingProxyType(normalized)

        normalized = normalize_policy(self.tile_out_features_by_shape, required=False)
        q4_normalized = normalize_policy(
            self.q4_tile_out_features_by_shape or {}, required=False
        )
        q5_normalized = normalize_policy(
            self.q5_tile_out_features_by_shape or {}, required=False
        )
        if not (normalized or q4_normalized or q5_normalized):
            raise ValueError("T16 F16/rocBLAS requires at least one shape policy")
        object.__setattr__(self, "tile_out_features_by_shape", normalized)
        object.__setattr__(self, "q4_tile_out_features_by_shape", q4_normalized)
        object.__setattr__(self, "q5_tile_out_features_by_shape", q5_normalized)
        solution_indices: dict[tuple[int, int, int], int] = {}
        for raw_shape, raw_index in (self.solution_indices_by_gemm_shape or {}).items():
            if len(raw_shape) != 3:
                raise ValueError(
                    "T16 F16/rocBLAS solution policies require (M, K, N) shapes"
                )
            shape = tuple(int(value) for value in raw_shape)
            index = int(raw_index)
            if any(value <= 0 for value in shape) or shape[1] % 256:
                raise ValueError("T16 F16/rocBLAS solution shapes must be valid")
            if not -(1 << 31) <= index < (1 << 31):
                raise ValueError("T16 F16/rocBLAS solution indices must fit int32")
            solution_indices[shape] = index
        object.__setattr__(
            self,
            "solution_indices_by_gemm_shape",
            MappingProxyType(solution_indices),
        )
        def normalize_inplace(
            raw_shapes: frozenset[tuple[int, ...]],
            policy: Mapping[tuple[int, ...], int],
        ) -> frozenset[tuple[int, ...]]:
            inplace = frozenset(
                tuple(int(value) for value in shape) for shape in raw_shapes
            )
            if any(len(shape) not in {2, 3} for shape in inplace):
                raise ValueError(
                    "T16 F16/rocBLAS in-place policies require "
                    "(K, N) or (M, K, N) shapes"
                )
            if not inplace.issubset(policy):
                raise ValueError(
                    "T16 F16/rocBLAS in-place shapes must have tile policies"
                )
            return inplace

        object.__setattr__(
            self,
            "x_inplace_shapes",
            normalize_inplace(self.x_inplace_shapes, normalized),
        )
        object.__setattr__(
            self,
            "q4_x_inplace_shapes",
            normalize_inplace(self.q4_x_inplace_shapes, q4_normalized),
        )
        object.__setattr__(
            self,
            "q5_x_inplace_shapes",
            normalize_inplace(self.q5_x_inplace_shapes, q5_normalized),
        )

        quant_policies = {
            "gguf_q4_k_t16_v1": q4_normalized,
            "gguf_q5_k_t16_v1": q5_normalized,
            "gguf_q6_k_t16_qmicro_planar_v1": normalized,
        }
        normalized_max_rows: dict[str, Mapping[tuple[int, int], int]] = {}
        for raw_quant, raw_shapes in (self.max_rows_by_quant_shape or {}).items():
            quant = str(raw_quant)
            quant_policy = quant_policies.get(quant)
            if quant_policy is None or not isinstance(raw_shapes, Mapping):
                raise ValueError(
                    "T16 F16/rocBLAS max-row policies require a known quant"
                )
            shape_limits: dict[tuple[int, int], int] = {}
            for raw_shape, raw_maximum in raw_shapes.items():
                if len(raw_shape) != 2:
                    raise ValueError(
                        "T16 F16/rocBLAS max-row policies require (K, N) shapes"
                    )
                shape = (int(raw_shape[0]), int(raw_shape[1]))
                maximum = int(raw_maximum)
                matching_anchors = tuple(
                    policy_shape[0]
                    for policy_shape in quant_policy
                    if len(policy_shape) == 3 and policy_shape[1:] == shape
                )
                if not any(
                    policy_shape[-2:] == shape for policy_shape in quant_policy
                ):
                    raise ValueError(
                        "T16 F16/rocBLAS max-row shapes require a tile policy"
                    )
                if maximum <= 0 or (
                    matching_anchors and max(matching_anchors) > maximum
                ):
                    raise ValueError(
                        "T16 F16/rocBLAS max rows must include every row anchor"
                    )
                shape_limits[shape] = maximum
            normalized_max_rows[quant] = MappingProxyType(shape_limits)
        object.__setattr__(
            self,
            "max_rows_by_quant_shape",
            MappingProxyType(normalized_max_rows),
        )

        normalized_variants: dict[
            str,
            Mapping[tuple[int, int], Mapping[tuple[int, int], str]],
        ] = {}
        for raw_quant, raw_shapes in (
            self.linear_variant_intervals_by_quant or {}
        ).items():
            quant = str(raw_quant)
            quant_policy = quant_policies.get(quant)
            if quant_policy is None or not isinstance(raw_shapes, Mapping):
                raise ValueError(
                    "T16 F16/rocBLAS variant policies require a known quant"
                )
            shape_variants: dict[
                tuple[int, int], Mapping[tuple[int, int], str]
            ] = {}
            for raw_shape, raw_intervals in raw_shapes.items():
                if len(raw_shape) != 2 or not isinstance(raw_intervals, Mapping):
                    raise ValueError(
                        "T16 F16/rocBLAS variant policies require (K, N) shapes"
                    )
                shape = (int(raw_shape[0]), int(raw_shape[1]))
                if not any(
                    policy_shape[-2:] == shape for policy_shape in quant_policy
                ):
                    raise ValueError(
                        "T16 F16/rocBLAS variant shapes require a tile policy"
                    )
                intervals: dict[tuple[int, int], str] = {}
                for raw_interval, raw_variant in raw_intervals.items():
                    if len(raw_interval) != 2:
                        raise ValueError(
                            "T16 F16/rocBLAS variant intervals require "
                            "inclusive (min_rows, max_rows) bounds"
                        )
                    interval = (int(raw_interval[0]), int(raw_interval[1]))
                    variant = str(raw_variant).strip()
                    if interval[0] <= 0 or interval[1] < interval[0] or not variant:
                        raise ValueError(
                            "T16 F16/rocBLAS variant intervals must be valid"
                        )
                    if any(
                        max(interval[0], prior[0])
                        <= min(interval[1], prior[1])
                        for prior in intervals
                    ):
                        raise ValueError(
                            "T16 F16/rocBLAS variant intervals must not overlap"
                        )
                    intervals[interval] = variant
                shape_variants[shape] = MappingProxyType(intervals)
            normalized_variants[quant] = MappingProxyType(shape_variants)
        object.__setattr__(
            self,
            "linear_variant_intervals_by_quant",
            MappingProxyType(normalized_variants),
        )

        normalized_pair_only: dict[
            tuple[str, int, int, str, int],
            Mapping[tuple[int, int], tuple[int, str, bool]],
        ] = {}
        for raw_key, raw_intervals in (
            self.pair_only_second_operand_policies or {}
        ).items():
            if len(raw_key) != 5 or not isinstance(raw_intervals, Mapping):
                raise ValueError(
                    "T16 F16/rocBLAS pair-only policies require "
                    "(first_quant, K, first_N, second_quant, second_N) keys"
                )
            key = (
                str(raw_key[0]),
                int(raw_key[1]),
                int(raw_key[2]),
                str(raw_key[3]),
                int(raw_key[4]),
            )
            first_policy = quant_policies.get(key[0])
            if (
                first_policy is None
                or key[3] not in quant_policies
                or key[1] <= 0
                or key[1] % 256
                or key[2] <= 0
                or key[4] <= 0
                or not any(
                    shape[-2:] == (key[1], key[2]) for shape in first_policy
                )
            ):
                raise ValueError(
                    "T16 F16/rocBLAS pair-only policies require an admitted "
                    "first operand and known quant/shape axes"
                )
            intervals: dict[tuple[int, int], tuple[int, str, bool]] = {}
            for raw_interval, raw_spec in raw_intervals.items():
                if len(raw_interval) != 2 or len(raw_spec) != 3:
                    raise ValueError(
                        "T16 F16/rocBLAS pair-only entries require inclusive "
                        "row bounds and (tile, variant, in_place) values"
                    )
                interval = (int(raw_interval[0]), int(raw_interval[1]))
                tile = int(raw_spec[0])
                variant = str(raw_spec[1]).strip()
                in_place = raw_spec[2]
                if (
                    interval[0] <= 0
                    or interval[1] < interval[0]
                    or tile <= 0
                    or tile % 16
                    or tile > key[4]
                    or key[4] % tile
                    or not variant
                    or not isinstance(in_place, bool)
                ):
                    raise ValueError(
                        "T16 F16/rocBLAS pair-only entries must be valid"
                    )
                if any(
                    max(interval[0], prior[0]) <= min(interval[1], prior[1])
                    for prior in intervals
                ):
                    raise ValueError(
                        "T16 F16/rocBLAS pair-only intervals must not overlap"
                    )
                intervals[interval] = (tile, variant, in_place)
            normalized_pair_only[key] = MappingProxyType(intervals)
        object.__setattr__(
            self,
            "pair_only_second_operand_policies",
            MappingProxyType(normalized_pair_only),
        )

    def linear_variant(
        self,
        rows: int,
        in_features: int,
        out_features: int,
        *,
        quant: str,
        default: str,
    ) -> str:
        """Resolve an admitted registered composite variant or its fallback."""

        by_quant = self.linear_variant_intervals_by_quant
        assert by_quant is not None
        intervals = by_quant.get(str(quant), {}).get(
            (int(in_features), int(out_features)), {}
        )
        row_count = int(rows)
        return next(
            (
                variant
                for (minimum, maximum), variant in intervals.items()
                if minimum <= row_count <= maximum
            ),
            str(default),
        )

    def pair_only_second_operand(
        self,
        rows: int,
        in_features: int,
        first_out_features: int,
        second_out_features: int,
        *,
        first_quant: str,
        second_quant: str,
    ) -> tuple[int, str, bool] | None:
        """Resolve an ordered pair-only second operand or fail closed."""

        policies = self.pair_only_second_operand_policies
        assert policies is not None
        intervals = policies.get(
            (
                str(first_quant),
                int(in_features),
                int(first_out_features),
                str(second_quant),
                int(second_out_features),
            ),
            {},
        )
        row_count = int(rows)
        return next(
            (
                spec
                for (minimum, maximum), spec in intervals.items()
                if minimum <= row_count <= maximum
            ),
            None,
        )

    def tile_out_features(
        self,
        rows: int,
        in_features: int,
        out_features: int,
        *,
        quant: str = "gguf_q6_k_t16_qmicro_planar_v1",
    ) -> int | None:
        exact = (int(rows), int(in_features), int(out_features))
        policy = {
            "gguf_q4_k_t16_v1": self.q4_tile_out_features_by_shape,
            "gguf_q5_k_t16_v1": self.q5_tile_out_features_by_shape,
            "gguf_q6_k_t16_qmicro_planar_v1": self.tile_out_features_by_shape,
        }.get(quant, {})
        assert policy is not None
        max_rows = self.max_rows_by_quant_shape
        assert max_rows is not None
        shape_maximum = max_rows.get(quant, {}).get(exact[1:])
        if shape_maximum is not None and exact[0] > shape_maximum:
            return None
        direct = policy.get(exact, policy.get(exact[1:]))
        if direct is not None:
            return direct
        # Three-axis policy rows are measured lower-bound anchors. Use the
        # nearest admitted anchor so ordinary prompt lengths between benchmark
        # powers of two do not silently fall back to a different arithmetic
        # path. ``max_rows`` remains the hard upper admission bound.
        anchored = (
            (shape[0], tile)
            for shape, tile in policy.items()
            if len(shape) == 3
            and shape[0] <= exact[0]
            and shape[1:] == exact[1:]
        )
        return max(anchored, default=(0, None), key=lambda item: item[0])[1]

    def solution_index(
        self,
        rows: int,
        in_features: int,
        tile_out_features: int,
    ) -> int | None:
        """Return the backend-qualified index for one effective tile GEMM."""

        policies = self.solution_indices_by_gemm_shape
        assert policies is not None
        return policies.get(
            (int(rows), int(in_features), int(tile_out_features))
        )

    def activation_is_inplace(
        self,
        rows: int,
        in_features: int,
        out_features: int,
        *,
        quant: str = "gguf_q6_k_t16_qmicro_planar_v1",
    ) -> bool:
        exact = (int(rows), int(in_features), int(out_features))
        policy = {
            "gguf_q4_k_t16_v1": self.q4_x_inplace_shapes,
            "gguf_q5_k_t16_v1": self.q5_x_inplace_shapes,
            "gguf_q6_k_t16_qmicro_planar_v1": self.x_inplace_shapes,
        }.get(quant, ())
        assert policy is not None
        return exact in policy or exact[1:] in policy or any(
            len(shape) == 3
            and shape[0] <= exact[0]
            and shape[1:] == exact[1:]
            for shape in policy
        )


_t16_f16_rocblas_prefill_session: ContextVar[
    T16F16RocblasPrefillSession | None
] = ContextVar("t16_f16_rocblas_prefill_session", default=None)
_q4_t16_unequal_pair_prefill_enabled: ContextVar[bool] = ContextVar(
    "q4_t16_unequal_pair_prefill_enabled", default=False
)
_q4_pack8_dual_wmma_silu_prefill_enabled: ContextVar[bool] = ContextVar(
    "q4_pack8_dual_wmma_silu_prefill_enabled", default=False
)
_q8_t16_dual_wmma_prefill_enabled: ContextVar[bool] = ContextVar(
    "q8_t16_dual_wmma_prefill_enabled", default=False
)


@contextlib.contextmanager
def q4_pack8_dual_wmma_silu_prefill_session(enabled: bool) -> Iterator[None]:
    """Admit the model-qualified operation-complete pack8 owner for one request."""

    token = _q4_pack8_dual_wmma_silu_prefill_enabled.set(bool(enabled))
    try:
        yield
    finally:
        _q4_pack8_dual_wmma_silu_prefill_enabled.reset(token)


@contextlib.contextmanager
def q8_t16_dual_wmma_prefill_session(enabled: bool) -> Iterator[None]:
    """Admit the model-qualified narrow Q8T16 pair for one request."""

    token = _q8_t16_dual_wmma_prefill_enabled.set(bool(enabled))
    try:
        yield
    finally:
        _q8_t16_dual_wmma_prefill_enabled.reset(token)


@contextlib.contextmanager
def q4_t16_unequal_pair_prefill_session(enabled: bool) -> Iterator[None]:
    """Admit the model-qualified unequal Q4T16 bulk pair for one request."""

    token = _q4_t16_unequal_pair_prefill_enabled.set(bool(enabled))
    try:
        yield
    finally:
        _q4_t16_unequal_pair_prefill_enabled.reset(token)


@contextlib.contextmanager
def t16_f16_rocblas_prefill_session(
    session: T16F16RocblasPrefillSession | None,
) -> Iterator[None]:
    """Expose bounded Q4/Q5/Q6 source-F16 planes for one owner-controlled pass."""

    token = _t16_f16_rocblas_prefill_session.set(session)
    try:
        yield
    finally:
        _t16_f16_rocblas_prefill_session.reset(token)


# Compatibility names for callers written when this owner covered only Q6.
Q6T16F16RocblasPrefillSession = T16F16RocblasPrefillSession
q6_t16_f16_rocblas_prefill_session = t16_f16_rocblas_prefill_session


@contextlib.contextmanager
def q5_f32_ordered_prefill_session(
    session: Q5F32OrderedPrefillSession | None,
) -> Iterator[None]:
    """Expose one bounded exact-value raw-K plane during an owner row pass."""

    token = _q5_f32_ordered_prefill_session.set(session)
    try:
        yield
    finally:
        _q5_f32_ordered_prefill_session.reset(token)


@contextlib.contextmanager
def q5_raw_mmq_target_session(
    *,
    workspace_ptr: int = 0,
    workspace_nbytes: int = 0,
    library: ctypes.CDLL | None = None,
    quant_library: ctypes.CDLL | None = None,
    enabled: bool = True,
    source_layout: bool = False,
    planar_dp4a: bool = False,
    planar_library: ctypes.CDLL | None = None,
) -> Iterator[None]:
    """Expose bounded Q8_1 storage for the C8 recurrent-Q5 owner."""

    selected = None
    if enabled:
        if int(workspace_ptr) <= 0 or int(workspace_nbytes) <= 0:
            raise ValueError("Q5 raw MMQ target session requires a device workspace")
        if planar_dp4a and planar_library is None:
            raise ValueError(
                "Q5 planar dp4a target session requires the planar library"
            )
        selected = _Q5RawMMQTargetSession(
            workspace_ptr=int(workspace_ptr),
            workspace_nbytes=int(workspace_nbytes),
            library=library,
            quant_library=quant_library,
            source_layout=bool(source_layout),
            planar_dp4a=bool(planar_dp4a),
            planar_library=planar_library,
        )
    token = _q5_raw_mmq_target_session.set(selected)
    try:
        yield
    finally:
        _q5_raw_mmq_target_session.reset(token)


@contextlib.contextmanager
def q8_mmq_prefill_session(
    *,
    workspace_ptr: int,
    workspace_nbytes: int,
    risk_count_ptr: int = 0,
    risk_count_nbytes: int = 0,
    risk_indices_ptr: int = 0,
    risk_indices_nbytes: int = 0,
    policy: Q8MMQPrefillPolicy | None,
    library: ctypes.CDLL | None = None,
) -> Iterator[None]:
    """Expose a bounded D4 workspace only while a plugin-selected prefill runs."""

    if policy is None:
        selected = None
    else:
        if int(workspace_ptr) <= 0 or int(workspace_nbytes) <= 0:
            raise ValueError("Q8 MMQ prefill requires a non-empty device workspace")
        if int(risk_count_ptr) <= 0 or int(risk_count_nbytes) < ctypes.sizeof(ctypes.c_int32):
            raise ValueError("Q8 MMQ prefill requires a bounded risk counter")
        if int(risk_indices_ptr) <= 0 or int(risk_indices_nbytes) <= 0:
            raise ValueError("Q8 MMQ prefill requires a bounded risk-index queue")
        selected = _Q8MMQPrefillSession(
            workspace_ptr=int(workspace_ptr),
            workspace_nbytes=int(workspace_nbytes),
            risk_count_ptr=int(risk_count_ptr),
            risk_count_nbytes=int(risk_count_nbytes),
            risk_indices_ptr=int(risk_indices_ptr),
            risk_indices_nbytes=int(risk_indices_nbytes),
            library=library,
            policy=policy,
        )
    token = _q8_mmq_prefill_session.set(selected)
    try:
        yield
    finally:
        _q8_mmq_prefill_session.reset(token)


def resolve_q8_mmq_prefill_policy(
    quant: str,
    *,
    backend: str = "hip_gfx1100",
) -> Q8MMQPrefillPolicy | None:
    """Resolve the optional raw-Q8 MMQ policy on the model quant axis."""

    register_gguf_q8_0_mmq_prefill_kernels()
    return resolve(
        backend=backend,
        layer="linear_prefill_policy",
        quant=str(quant),
        variant="raw_q8_mmq128",
        missing="none",
    )


_DISPATCH_TABLE: Mapping[tuple[str, str, str], GGUFLinearDispatch] = {
    (LAYOUT_Q4_K_PACK8, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_bf16_bf16_out"),
        "pack8",
    ),
    (LAYOUT_Q4_K_PACK8, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_FP16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_bf16_fp16_out"),
        "pack8",
    ),
    (LAYOUT_Q4_K_PACK8, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_F32): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "pack8_bf16_f32_out"),
        "pack8",
    ),
    (LAYOUT_RAW_GGUF, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "<from-weight>", "gemv_bf16_bf16_out"),
        "raw",
    ),
    (LAYOUT_RAW_GGUF, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_FP16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "<from-weight>", "gemv_bf16_fp16_out"),
        "raw",
    ),
    (LAYOUT_RAW_GGUF, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_F32): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "<from-weight>", "gemv_bf16_f32_out"),
        "raw",
    ),
    (LAYOUT_DENSE_BF16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "dense_gemv", "bf16", "out"),
        "dense_bf16",
    ),
    (LAYOUT_DENSE_BF16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_F32): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "dense_gemv", "bf16", "f32_out"),
        "dense_bf16",
    ),
    (LAYOUT_DENSE_F32, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "dense_gemv", "f32", "bf16_hidden_bf16_out"),
        "dense_bf16",
    ),
    (LAYOUT_DENSE_F32, GGUF_ACTIVATION_F32, GGUF_OUTPUT_F32): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "dense_gemv", "f32", "f32_hidden_f32_out"),
        "dense_bf16",
    ),
    (LAYOUT_GGUF_Q4_K_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16): GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_single_local32_bf16_bf16_out",
        ),
        "t16",
    ),
    (
        LAYOUT_GGUF_Q4_K_QMICRO_T16,
        GGUF_ACTIVATION_BF16,
        GGUF_OUTPUT_BF16,
    ): GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q4_k_qmicro_t16_v1",
            "dense_single_local32_bf16_bf16_out",
        ),
        "t16",
    ),
    (LAYOUT_GGUF_Q5_K_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q5_k_t16_v1", "t16_gemv_decode_bf16_bf16_out"),
        "t16",
    ),
    (LAYOUT_GGUF_Q6_K_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k_t16_v1", "t16_gemv_decode_bf16_bf16_out"),
        "t16",
    ),
    (LAYOUT_GGUF_Q6_K_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_F32): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k_t16_v1", "t16_gemv_decode_bf16_f32_out"),
        "t16",
    ),
    (
        LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        GGUF_ACTIVATION_BF16,
        GGUF_OUTPUT_BF16,
    ): GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_decode_bf16_bf16_out",
        ),
        "t16",
    ),
    (
        LAYOUT_GGUF_Q6_K_T16_QMICRO_PLANAR,
        GGUF_ACTIVATION_BF16,
        GGUF_OUTPUT_F32,
    ): GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_decode_bf16_f32_out",
        ),
        "t16",
    ),
    (LAYOUT_GGUF_Q8_0_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_bf16_bf16_out"),
        "t16",
    ),
    (LAYOUT_GGUF_Q8_0_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_FP16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_fp16_fp16_out"),
        "t16",
    ),
    (LAYOUT_GGUF_Q8_0_T16, GGUF_ACTIVATION_F32, GGUF_OUTPUT_BF16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q8_0_t16_v1", "t16_gemv_decode_f32_bf16_out"),
        "t16",
    ),
}


def _weight_backend(
    *weights: GGUFDeviceWeight,
    backend: str | None = None,
) -> str:
    """Return one explicit or resident-weight backend for a dispatch group."""

    resident_backends = {
        str(value)
        for weight in weights
        if (value := getattr(weight, "backend", None)) is not None
    }
    if len(resident_backends) > 1:
        raise ValueError(
            "GGUF fused dispatch requires weights from one backend; got "
            + ", ".join(sorted(resident_backends))
        )
    if backend is not None:
        if resident_backends and backend not in resident_backends:
            resident = next(iter(resident_backends))
            raise ValueError(
                f"GGUF dispatch backend {backend!r} does not match resident weight backend {resident!r}"
            )
        return backend
    if resident_backends:
        return next(iter(resident_backends))
    # Compatibility for lightweight dispatch fixtures that predate backend-tagged
    # resident weights. Production materialization always supplies the backend.
    return "hip_gfx1100"


def set_gemv_decode_enabled(enabled: bool | None) -> None:
    """Set the session-scoped opt-in for the GGUF pack8 GEMV decode family.

    Pass ``True`` / ``False`` to override env + per-call kwargs for this
    process. Pass ``None`` to clear the override and fall back to the env
    var (``HIPENGINE_GGUF_GEMV_DECODE``). Intended to be called once by a
    runner that drives ``Qwen35GGUFResidentSession.use_gemv_decode`` from
    its public API. The kwarg path remains available for ad-hoc bisects.
    """

    global _gemv_decode_session_enabled
    _gemv_decode_session_enabled = None if enabled is None else bool(enabled)


@contextlib.contextmanager
def gemv_decode_session(enabled: bool | None) -> Iterator[None]:
    """Context manager wrapper around :func:`set_gemv_decode_enabled`."""

    previous = _gemv_decode_session_enabled
    set_gemv_decode_enabled(enabled)
    try:
        yield
    finally:
        set_gemv_decode_enabled(previous)


@contextlib.contextmanager
def native_batch_decode_session(enabled: bool = True) -> Iterator[None]:
    """Select exact small-row native projection families for c=2/4/8."""

    global _native_batch_decode_session_enabled
    previous = _native_batch_decode_session_enabled
    _native_batch_decode_session_enabled = bool(enabled)
    try:
        yield
    finally:
        _native_batch_decode_session_enabled = previous


def gguf_native_batch_decode_enabled() -> bool:
    """Return whether the current execution context owns native batch decode."""

    return _native_batch_decode_session_enabled


@contextlib.contextmanager
def target_verifier_rowtile_session(enabled: bool = True) -> Iterator[None]:
    """Allow backend-admitted T16 rowtiles inside a packed target verifier.

    This scope is intentionally narrower than :func:`native_batch_decode_session`:
    the backend capability names the eligible quant plugins, and every unrelated
    projection keeps its verifier owner.
    """

    token = _target_verifier_rowtile_session_enabled.set(bool(enabled))
    try:
        yield
    finally:
        _target_verifier_rowtile_session_enabled.reset(token)


@contextlib.contextmanager
def target_verifier_production_q4_rowtile_session(
    enabled: bool = True,
) -> Iterator[None]:
    """Enable a production-numerics Q4 verifier candidate in one context."""

    token = _target_verifier_production_q4_rowtile_session_enabled.set(
        bool(enabled)
    )
    try:
        yield
    finally:
        _target_verifier_production_q4_rowtile_session_enabled.reset(token)


@contextlib.contextmanager
def target_verifier_wide_q6_shared4_session(
    enabled: bool = True,
) -> Iterator[None]:
    """Enable the W1 B-stationary Q6 verifier candidate in one context."""

    token = _target_verifier_wide_q6_shared4_policy_enabled.set(bool(enabled))
    try:
        yield
    finally:
        _target_verifier_wide_q6_shared4_policy_enabled.reset(token)


def target_verifier_wide_q6_shared4_policy_enabled() -> bool:
    """Return whether the outer W1 logical-width policy is enabled."""

    if _target_verifier_wide_q6_shared4_policy_enabled.get():
        return True
    return os.environ.get(TARGET_VERIFIER_WIDE_Q6_SHARED4_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@contextlib.contextmanager
def target_verifier_wide_q6_shared4_leaf_session(
    enabled: bool = True,
) -> Iterator[None]:
    """Activate W1 Q6 leaves only inside one packed verifier transaction."""

    token = _target_verifier_wide_q6_shared4_leaf_enabled.set(bool(enabled))
    try:
        yield
    finally:
        _target_verifier_wide_q6_shared4_leaf_enabled.reset(token)


def _env_gemv_decode_enabled() -> bool:
    raw = os.environ.get(_GEMV_DECODE_ENV, "")
    if not raw:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def gguf_gemv_decode_enabled(use_gemv_decode: bool | None = None) -> bool:
    """Return the resolved GGUF pack8 GEMV decode opt-in state.

    Precedence (highest first): explicit kwarg, session toggle, env var.
    Mirrors :func:`gguf_wmma_prefill_enabled` for the decode-side rewrite.
    """

    return _resolve_use_gemv_decode(use_gemv_decode)


def _resolve_use_gemv_decode(kwarg: bool | None) -> bool:
    if kwarg is not None:
        return bool(kwarg)
    if _gemv_decode_session_enabled is not None:
        return _gemv_decode_session_enabled
    return _env_gemv_decode_enabled()


def _resolve_q8_t16_threads(threads: int = 0) -> int:
    """Resolve the Q8_0 T16 GEMV launch width.

    Returns ``0`` when no override is active so the wrapper keeps its default
    128-thread launch. Explicit kwargs take precedence over the env var.
    """

    if int(threads) != 0:
        value = int(threads)
    else:
        raw = os.environ.get(_Q8_T16_THREADS_ENV, "").strip()
        if not raw:
            return 0
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{_Q8_T16_THREADS_ENV} must be one of 64 or 128") from exc
    if value not in _Q8_T16_ALLOWED_THREADS:
        raise ValueError(f"{_Q8_T16_THREADS_ENV} must be one of 64 or 128")
    return value


def _q8_t16_threads_override_active(threads: int = 0) -> bool:
    return int(threads) != 0 or bool(os.environ.get(_Q8_T16_THREADS_ENV, "").strip())


def set_q8_t16_pair_rowtile_min_rows(min_rows: int | None) -> None:
    """Set the owner-scoped minimum width for exact Q8T16 pair rowtiling."""

    global _q8_t16_pair_rowtile_min_rows_session
    if min_rows is not None and int(min_rows) < 0:
        raise ValueError("Q8T16 pair rowtile minimum rows must be non-negative")
    _q8_t16_pair_rowtile_min_rows_session = None if min_rows is None else int(min_rows)


@contextlib.contextmanager
def q8_t16_pair_rowtile_min_rows_session(min_rows: int | None) -> Iterator[None]:
    """Temporarily apply a backend-certified Q8T16 pair-rowtile width floor."""

    previous = _q8_t16_pair_rowtile_min_rows_session
    set_q8_t16_pair_rowtile_min_rows(min_rows)
    try:
        yield
    finally:
        set_q8_t16_pair_rowtile_min_rows(previous)


def set_q8_t16_rowtile_all_enabled(enabled: bool | None) -> None:
    """Set the packed-AR scoped Q8T16 row-amortized decode policy."""

    global _q8_t16_rowtile_all_session_enabled
    _q8_t16_rowtile_all_session_enabled = None if enabled is None else bool(enabled)


@contextlib.contextmanager
def q8_t16_rowtile_all_session(enabled: bool | None) -> Iterator[None]:
    """Temporarily select Q8T16 row amortization for one execution owner."""

    previous = _q8_t16_rowtile_all_session_enabled
    set_q8_t16_rowtile_all_enabled(enabled)
    try:
        yield
    finally:
        set_q8_t16_rowtile_all_enabled(previous)


def _resolve_use_q8_t16_all_rowtile() -> bool:
    raw = os.environ.get(_Q8_T16_ROWTILE_ALL_ENV, "")
    if raw:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if _q8_t16_rowtile_all_session_enabled is not None:
        return _q8_t16_rowtile_all_session_enabled
    return False


def _resolve_use_q8_t16_pair_rowtile(*, rows: int) -> bool:
    raw = os.environ.get(_Q8_T16_PAIR_ROWTILE_ENV, "")
    if raw:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    min_rows = _q8_t16_pair_rowtile_min_rows_session
    if min_rows is not None and min_rows > 0 and rows >= min_rows:
        return True
    return _resolve_use_q8_t16_all_rowtile()


def _resolve_use_q8_t16_pair_col8(*, rows: int) -> bool:
    raw = os.environ.get(_Q8_T16_PAIR_COL8_ENV, "")
    if raw:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    min_rows = _q8_t16_pair_rowtile_min_rows_session
    return min_rows is not None and min_rows > 0 and rows >= min_rows


def _use_q8_t16_all_rowtile(
    *,
    rows: int,
    in_features: int,
    threads: int = 0,
) -> bool:
    return (
        rows > 1
        and in_features == _Q8_T16_QWEN35_ATTN_IN
        and not _q8_t16_threads_override_active(threads)
        and _resolve_use_q8_t16_all_rowtile()
    )


def _use_q8_t16_pair_rowtile(
    *,
    rows: int,
    in_features: int,
    out_features_a: int,
    out_features_b: int,
    threads: int = 0,
) -> bool:
    return (
        rows > 1
        and in_features == _Q8_T16_QWEN35_ATTN_IN
        and out_features_a == _Q8_T16_QWEN35_ATTN_QKV_OUT
        and out_features_b == _Q8_T16_QWEN35_ATTN_GATE_OUT
        and not _q8_t16_threads_override_active(threads)
        and _resolve_use_q8_t16_pair_rowtile(rows=rows)
    )


def set_wmma_prefill_enabled(enabled: bool | None) -> None:
    """Set the session-scoped opt-in for the GGUF WMMA prefill family.

    Pass ``True`` / ``False`` to override env + per-call kwargs for this
    process. Pass ``None`` to clear the override and fall back to the env
    var (``HIPENGINE_GGUF_WMMA_PREFILL``). Intended to be called once by a
    runner that drives ``PrefillConfig.use_wmma_prefill`` from its public
    API. The kwarg path remains available for ad-hoc bisects.
    """

    global _wmma_prefill_session_enabled
    _wmma_prefill_session_enabled = None if enabled is None else bool(enabled)


@contextlib.contextmanager
def wmma_prefill_session(enabled: bool | None) -> Iterator[None]:
    """Context manager wrapper around :func:`set_wmma_prefill_enabled`."""

    previous = _wmma_prefill_session_enabled
    set_wmma_prefill_enabled(enabled)
    try:
        yield
    finally:
        set_wmma_prefill_enabled(previous)


def _env_wmma_prefill_enabled() -> bool:
    raw = os.environ.get(_WMMA_PREFILL_ENV, "")
    if not raw:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def gguf_wmma_prefill_enabled(use_wmma_prefill: bool | None = None) -> bool:
    """Return the resolved GGUF WMMA prefill opt-in state.

    This exposes the same precedence used by :func:`launch_gguf_linear` so
    higher-level runners can route composite GGUF prefill paths without
    duplicating env-var or session-toggle checks.
    """

    return _resolve_use_wmma_prefill(use_wmma_prefill)


def _resolve_use_wmma_prefill(kwarg: bool | None) -> bool:
    """Combine per-call kwarg + session toggle + env var.

    Precedence (highest first): explicit kwarg, session toggle, env var.
    """

    if kwarg is not None:
        return bool(kwarg)
    if _wmma_prefill_session_enabled is not None:
        return _wmma_prefill_session_enabled
    return _env_wmma_prefill_enabled()


def set_q4k_rowtile_enabled(enabled: bool | None) -> None:
    """Set the session-scoped opt-out for exact small-B row-tile GEMVs.

    Pass ``False`` to force the legacy per-row prefill alias (bisection only);
    ``None`` clears the override and falls back to the env var, which itself
    defaults to ON.
    """

    global _q4k_rowtile_session_enabled
    _q4k_rowtile_session_enabled = None if enabled is None else bool(enabled)


@contextlib.contextmanager
def q4k_rowtile_session(enabled: bool | None) -> Iterator[None]:
    """Context manager wrapper around :func:`set_q4k_rowtile_enabled`."""

    previous = _q4k_rowtile_session_enabled
    set_q4k_rowtile_enabled(enabled)
    try:
        yield
    finally:
        set_q4k_rowtile_enabled(previous)


def _resolve_use_q4k_rowtile(kwarg: bool | None) -> bool:
    if kwarg is not None:
        return bool(kwarg)
    if _q4k_rowtile_session_enabled is not None:
        return _q4k_rowtile_session_enabled
    raw = os.environ.get(_Q4K_ROWTILE_ENV, "")
    if not raw:
        return True  # default ON
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def raw_k_prefill_rowbatch() -> int:
    """Return the execution owner's exact raw-Q5/Q6 prefill row slab."""

    return int(_raw_k_prefill_rowbatch.get())


@contextlib.contextmanager
def raw_k_prefill_rowbatch_session(row_batch: int) -> Iterator[None]:
    """Temporarily select fixed row reuse for one bulk-prefill execution owner."""

    selected = int(row_batch)
    if selected not in _RAW_K_PREFILL_ROWBATCHES:
        raise ValueError(
            "raw-K prefill row batch must be one of 0, 4, 8, 16, or 32"
        )
    token = _raw_k_prefill_rowbatch.set(selected)
    try:
        yield
    finally:
        _raw_k_prefill_rowbatch.reset(token)


def raw_k_prefill_variant() -> str:
    """Return the execution owner's exact raw-Q5/Q6 prefill geometry."""

    return str(_raw_k_prefill_variant.get())


@contextlib.contextmanager
def raw_k_prefill_variant_session(variant: str) -> Iterator[None]:
    """Temporarily select output tiling or the explicit rowbatch rollback."""

    selected = str(variant).strip().lower()
    if selected not in _RAW_K_PREFILL_VARIANTS:
        raise ValueError("raw-K prefill variant must be 'rowbatch' or 'coltile'")
    token = _raw_k_prefill_variant.set(selected)
    try:
        yield
    finally:
        _raw_k_prefill_variant.reset(token)


def _backend_prefill_shape_is_qualified(
    backend: str,
    capability: str,
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> bool:
    """Admit a backend optimization only for its measured shape contracts."""

    shapes = backend_package_capability(backend, capability, frozenset())
    return (int(rows), int(in_features), int(out_features)) in shapes


def _dense_bf16_wmma_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    enabled: bool,
) -> GGUFLinearDispatch:
    """Prefer the D08-X2-K5 LDS-staged WMMA dense-BF16 bulk consumer."""

    if (
        not enabled
        or rows < 16
        or in_features <= 0
        or in_features % 32
        or out_features <= 0
        or out_features % 128
        or dispatch.abi != "dense_bf16"
        or dispatch.key.variant != "prefill_out"
        or dispatch.key.layer != "dense_gemv"
    ):
        return dispatch
    key = KernelKey(
        dispatch.key.backend,
        dispatch.key.layer,
        dispatch.key.quant,
        "prefill_wmma_out",
    )
    if not is_registered(key):
        return dispatch
    return GGUFLinearDispatch(key, dispatch.abi)


def _dense_bf16_rowtile_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    enabled: bool,
    native_batch: bool,
) -> GGUFLinearDispatch:
    """Select exact dense-BF16 native-c1 or small-row arithmetic reuse."""

    if not enabled or dispatch.abi != "dense_bf16":
        return dispatch
    if rows == 1:
        if (
            not native_batch
            or in_features > _DENSE_BF16_VIRTUAL256_MAX_IN_FEATURES
            or dispatch.key.variant != "out"
        ):
            return dispatch
        variant = "virtual256_out"
    elif (
        _DENSE_BF16_ROWTILE_MIN_ROWS <= rows <= _DENSE_BF16_ROWTILE_MAX_ROWS
        and dispatch.key.variant == "prefill_out"
    ):
        if (
            native_batch
            and in_features == _DENSE_BF16_VIRTUAL256_ROWTILE_IN_FEATURES
            and out_features == _DENSE_BF16_VIRTUAL256_ROWTILE_OUT_FEATURES
        ):
            candidate_key = KernelKey(
                dispatch.key.backend,
                dispatch.key.layer,
                dispatch.key.quant,
                "virtual256_rowtile_out",
            )
            if is_registered(candidate_key):
                return GGUFLinearDispatch(candidate_key, dispatch.abi)
        variant = "rowtile_out"
    else:
        return dispatch
    key = KernelKey(
        dispatch.key.backend,
        dispatch.key.layer,
        dispatch.key.quant,
        variant,
    )
    if not is_registered(key):
        return dispatch
    return GGUFLinearDispatch(key, dispatch.abi)


def _pack8_exact_prefill_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    enabled: bool,
) -> GGUFLinearDispatch:
    """Select exact resident-pack8 row reuse for populated prefill chunks."""

    if (
        not enabled
        or rows < _PACK8_EXACT_PREFILL_MIN_ROWS
        or dispatch.abi != "pack8"
        or dispatch.key.variant != "pack8_prefill_bf16_bf16_out"
    ):
        return dispatch
    key = KernelKey(
        dispatch.key.backend,
        dispatch.key.layer,
        dispatch.key.quant,
        "pack8_exact_prefill_tile8x8_bf16_bf16_out",
    )
    if not is_registered(key):
        return dispatch
    return GGUFLinearDispatch(key, dispatch.abi)


def _pack8_rowtile_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    use_rowtile: bool,
    native_batch: bool,
) -> GGUFLinearDispatch:
    """Select exact resident-pack8 weight reuse for native B1-B3 rows."""

    if (
        not use_rowtile
        or not native_batch
        or rows < _PACK8_ROWTILE_MIN_ROWS
        or rows > _PACK8_ROWTILE_MAX_ROWS
        or dispatch.abi != "pack8"
        or dispatch.key.quant != "gguf_q4_k"
        or dispatch.key.variant != "pack8_prefill_bf16_bf16_out"
    ):
        return dispatch
    key = KernelKey(
        dispatch.key.backend,
        dispatch.key.layer,
        dispatch.key.quant,
        "pack8_rowtile_bf16_bf16_out",
    )
    if not is_registered(key):
        return dispatch
    return GGUFLinearDispatch(key, dispatch.abi)


def _pack8_dual_rowtile_silu_dispatch(
    dispatch_a: GGUFLinearDispatch,
    dispatch_b: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    use_rowtile: bool,
    native_batch: bool,
) -> KernelKey | None:
    """Resolve the exact dense-FFN pair fusion or fail closed."""

    if (
        in_features != _PACK8_DUAL_ROWTILE_SILU_IN_FEATURES
        or out_features != _PACK8_DUAL_ROWTILE_SILU_OUT_FEATURES
    ):
        return None
    rowtile_a = _pack8_rowtile_dispatch(
        dispatch_a,
        rows=rows,
        use_rowtile=use_rowtile,
        native_batch=native_batch,
    )
    rowtile_b = _pack8_rowtile_dispatch(
        dispatch_b,
        rows=rows,
        use_rowtile=use_rowtile,
        native_batch=native_batch,
    )
    expected = KernelKey(
        dispatch_a.key.backend,
        "linear",
        "gguf_q4_k",
        "pack8_rowtile_bf16_bf16_out",
    )
    if (
        rowtile_a.abi != "pack8"
        or rowtile_b.abi != "pack8"
        or rowtile_a.key != expected
        or rowtile_b.key != expected
    ):
        return None
    candidate = KernelKey(
        dispatch_a.key.backend,
        "linear_pair_silu",
        "gguf_q4_k",
        "pack8_dual_rowtile_bf16_bf16_out",
    )
    return candidate if is_registered(candidate) else None


def _pack8_dual_wmma_silu_dispatch(
    dispatch_a: GGUFLinearDispatch,
    dispatch_b: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    use_wmma: bool,
) -> KernelKey | None:
    """Resolve the shape-qualified operation-complete pack8 bulk FFN owner."""

    expected = KernelKey(
        dispatch_a.key.backend,
        "linear",
        "gguf_q4_k",
        "pack8_prefill_bf16_bf16_out",
    )
    if (
        not use_wmma
        or not _q4_pack8_dual_wmma_silu_prefill_enabled.get()
        or os.environ.get(
            "HIPENGINE_GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL",
            "1",
        )
        == "0"
        or dispatch_a.abi != "pack8"
        or dispatch_b.abi != "pack8"
        or dispatch_a.key != expected
        or dispatch_b.key != expected
        or not backend_package_capability(
            dispatch_a.key.backend,
            "GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL",
            False,
        )
        or not _backend_prefill_shape_is_qualified(
            dispatch_a.key.backend,
            "GGUF_Q4_PACK8_DUAL_WMMA_SILU_PREFILL_SHAPES",
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
    ):
        return None
    candidate = KernelKey(
        dispatch_a.key.backend,
        "linear_pair_silu",
        "gguf_q4_k",
        "pack8_dual_wmma_prefill_bf16_bf16_out",
    )
    _ensure_linear_kernel_registered(candidate)
    return candidate if is_registered(candidate) else None


def _q4_t16_dual_silu_retile_enabled() -> bool:
    """Return the same-build rollback state for exact short-prefill retiles."""

    global _Q4_T16_DUAL_SILU_RETILE_RESOLVED
    if _Q4_T16_DUAL_SILU_RETILE_RESOLVED is None:
        raw = os.environ.get(_Q4_T16_DUAL_SILU_RETILE_ENV, "1").strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            _Q4_T16_DUAL_SILU_RETILE_RESOLVED = True
        elif raw in {"0", "false", "no", "off"}:
            _Q4_T16_DUAL_SILU_RETILE_RESOLVED = False
        else:
            raise ValueError(
                f"{_Q4_T16_DUAL_SILU_RETILE_ENV} must be a boolean value"
            )
    return _Q4_T16_DUAL_SILU_RETILE_RESOLVED


def _q4_t16_physical_dual_silu_variant(
    backend: str,
    rows: int,
    *,
    layer_id: int | None = None,
) -> str | None:
    """Return a backend-qualified physical fused-FFN retile, if enabled."""

    if not q4_t16_physical_extra_rowtiles_enabled():
        return None
    for capability in (
        "GGUF_SPECDEC2_Q4_DUAL_SILU_ROWTILE_POLICY",
        "GGUF_SPECDEC2_Q4_DUAL_SILU_PRODUCTION_R28_POLICY",
    ):
        policy = backend_package_capability(backend, capability, {})
        if not isinstance(policy, Mapping):
            continue
        rows_to_variant = policy.get("rows_to_variant", {})
        if not isinstance(rows_to_variant, Mapping):
            continue
        variant = rows_to_variant.get(int(rows))
        if not isinstance(variant, str) or not variant:
            continue
        strict_layer_modulus = policy.get("strict_layer_modulus")
        if strict_layer_modulus is not None:
            modulus = int(strict_layer_modulus)
            remainder = int(policy.get("strict_layer_remainder", 0))
            if modulus <= 0 or remainder < 0 or remainder >= modulus:
                raise ValueError(
                    f"{capability} has an invalid strict-layer schedule"
                )
            if layer_id is None or int(layer_id) % modulus == remainder:
                continue
        enabled_env = policy.get("enabled_env")
        if not isinstance(enabled_env, str) or not enabled_env:
            continue
        enabled_default = bool(policy.get("enabled_default", False))
        cache_key = (enabled_env, enabled_default)
        enabled = _rowtile_variant_policy_env_cache.get(cache_key)
        if enabled is None:
            raw = os.environ.get(
                enabled_env,
                "1" if enabled_default else "0",
            ).strip().lower()
            if raw in {"1", "true", "yes", "on"}:
                enabled = True
            elif raw in {"0", "false", "no", "off"}:
                enabled = False
            else:
                raise ValueError(f"{enabled_env} must be a boolean value")
            _rowtile_variant_policy_env_cache[cache_key] = enabled
        if enabled:
            return variant
    return None


def _q4_t16_grouped_pair_rows6_variant(
    backend: str,
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> str | None:
    """Return the backend-qualified grouped physical-pair sibling."""

    if int(rows) < 12 or int(rows) % 6:
        return None
    policy = backend_package_capability(
        backend,
        "GGUF_Q4_T16_GROUPED_PAIR_ROWS6_POLICY",
        {},
    )
    if not isinstance(policy, Mapping):
        return None
    shapes = policy.get("shapes", ())
    try:
        if (int(in_features), int(out_features)) not in shapes:
            return None
    except TypeError:
        return None
    variant = policy.get("variant")
    enabled_env = policy.get("enabled_env")
    if (
        not isinstance(variant, str)
        or not variant
        or not isinstance(enabled_env, str)
        or not enabled_env
    ):
        return None
    enabled_default = bool(policy.get("enabled_default", False))
    cache_key = (enabled_env, enabled_default)
    enabled = _rowtile_variant_policy_env_cache.get(cache_key)
    if enabled is None:
        raw = os.environ.get(
            enabled_env,
            "1" if enabled_default else "0",
        ).strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            enabled = True
        elif raw in {"0", "false", "no", "off"}:
            enabled = False
        else:
            raise ValueError(f"{enabled_env} must be a boolean value")
        _rowtile_variant_policy_env_cache[cache_key] = enabled
    if not enabled:
        return None
    key = KernelKey(backend, "linear", "gguf_q4_k_t16_v1", variant)
    _ensure_linear_kernel_registered(key)
    return variant if is_registered(key) else None


def _q4_t16_dual_wmma_silu_dispatch(
    dispatch_a: GGUFLinearDispatch,
    dispatch_b: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    layer_id: int | None = None,
    expanded_metadata: bool = False,
) -> KernelKey | None:
    """Resolve the operation-complete dense Q4T16 bulk FFN owner."""

    physical_variant = None
    if (
        not expanded_metadata
        and dispatch_a.key.quant == dispatch_b.key.quant
        == "gguf_q4_k_t16_v1"
        and physical_exact_rowtiles_enabled()
    ):
        physical_variant = _q4_t16_physical_dual_silu_variant(
            dispatch_a.key.backend,
            rows,
            layer_id=layer_id,
        )
    if (
        (rows < _Q4_T16_DUAL_WMMA_SILU_MIN_ROWS and physical_variant is None)
        or in_features != _PACK8_DUAL_ROWTILE_SILU_IN_FEATURES
        or out_features != _PACK8_DUAL_ROWTILE_SILU_OUT_FEATURES
        or dispatch_a.abi != "t16"
        or dispatch_b.abi != "t16"
        or dispatch_a.key.quant not in _Q4_T16_DENSE_QUANTS
        or dispatch_a.key.quant != dispatch_b.key.quant
        or (expanded_metadata and dispatch_a.key.quant != "gguf_q4_k_qmicro_t16_v1")
    ):
        return None
    parent_variant = (
        "dense_dual_wmma_prefill_expanded_meta_bf16_bf16_out"
        if expanded_metadata
        else "dense_dual_wmma_prefill_bf16_bf16_out"
    )
    variants = []
    if (
        not expanded_metadata
        and dispatch_a.key.quant == "gguf_q4_k_t16_v1"
        and _q4_t16_dual_silu_retile_enabled()
    ):
        if physical_variant is None:
            physical_variant = _q4_t16_physical_dual_silu_variant(
                dispatch_a.key.backend,
                rows,
                layer_id=layer_id,
            )
        if physical_variant is not None:
            variants.append(physical_variant)
        if rows <= 64:
            variants.append("dense_dual_wmma_prefill_row64_bf16_bf16_out")
        elif rows <= 128:
            variants.append("dense_dual_wmma_prefill_row128_bf16_bf16_out")
    variants.append(parent_variant)
    for variant in variants:
        candidate = KernelKey(
            dispatch_a.key.backend,
            "linear_pair_silu",
            dispatch_a.key.quant,
            variant,
        )
        _ensure_linear_kernel_registered(candidate)
        if is_registered(candidate):
            return candidate
    return None


def _q4_t16_dual_rowtile_silu_dispatch(
    dispatch_a: GGUFLinearDispatch,
    dispatch_b: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    use_sidecar: bool,
    native_batch: bool,
) -> KernelKey | None:
    """Resolve the exact compact-T16 dense-FFN owner or fail closed."""

    sole_t16 = (
        dispatch_a.abi == "t16"
        and dispatch_b.abi == "t16"
        and dispatch_a.key.quant in _Q4_T16_DENSE_QUANTS
        and dispatch_a.key.quant == dispatch_b.key.quant
    )
    pack8_sidecars = (
        use_sidecar
        and dispatch_a.abi == "pack8"
        and dispatch_b.abi == "pack8"
        and dispatch_a.key.quant == "gguf_q4_k"
        and dispatch_b.key.quant == "gguf_q4_k"
    )
    sole_t16_max_rows = int(
        _Q4_T16_DENSE_ROWTILE_MAX_ROWS_BY_QUANT.get(
            dispatch_a.key.quant,
            0,
        )
    )
    if (
        not native_batch
        or rows < _PACK8_ROWTILE_MIN_ROWS
        or (
            rows > _PACK8_ROWTILE_MAX_ROWS
            and not (sole_t16 and rows <= sole_t16_max_rows)
        )
        or in_features != _PACK8_DUAL_ROWTILE_SILU_IN_FEATURES
        or out_features != _PACK8_DUAL_ROWTILE_SILU_OUT_FEATURES
        or not (sole_t16 or pack8_sidecars)
    ):
        return None
    candidate = KernelKey(
        dispatch_a.key.backend,
        "linear_pair_silu",
        dispatch_a.key.quant if sole_t16 else "gguf_q4_k_t16_v1",
        "dense_dual_rowtile_bf16_bf16_out",
    )
    _ensure_linear_kernel_registered(candidate)
    return candidate if is_registered(candidate) else None


def _rowtile_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    use_rowtile: bool,
) -> GGUFLinearDispatch:
    """Rewrite the per-row raw quantized prefill alias -> weight-amortized rowtile.

    No-op unless ``use_rowtile`` and: ``rows`` in [2, 8], ``dispatch.abi`` is
    ``"raw"`` for a quant in ``_ROWTILE_QUANT_BLOCKS`` (Q4_K/Q5_K/Q6_K/Q8_0),
    the variant is one of the supported rows>1 ``prefill_*`` aliases, and
    ``in_features`` is K-block aligned for that quant. The
    rowtile wrapper shares the ``"raw"`` launch ABI, so only the variant name
    changes. Takes priority over ``_wmma_prefill_dispatch`` for small B.
    """

    if not use_rowtile or rows < _ROWTILE_MIN_ROWS or rows > _ROWTILE_MAX_ROWS:
        return dispatch
    block = _ROWTILE_QUANT_BLOCKS.get(dispatch.key.quant)
    if dispatch.abi != "raw" or block is None:
        return dispatch
    variant = dispatch.key.variant
    if variant not in _ROWTILE_SUPPORTED_PREFILL_VARIANTS:
        return dispatch
    if in_features % block != 0:
        return dispatch
    rowtile_variant = "rowtile_" + variant[len("prefill_") :]
    return GGUFLinearDispatch(
        KernelKey(
            dispatch.key.backend,
            dispatch.key.layer,
            dispatch.key.quant,
            rowtile_variant,
        ),
        "raw",
    )


def _raw_k_f32_ordered_prefill_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    weight: GGUFDeviceWeight | None = None,
    in_features: int,
    out_features: int,
) -> GGUFLinearDispatch:
    """Select an exact ordered raw-K route only inside its bounded owner."""

    session = _q5_f32_ordered_prefill_session.get()
    if (
        session is None
        or int(rows) < int(session.min_rows)
        or int(rows) > int(session.max_rows)
    ):
        return dispatch
    output_variants = {
        "prefill_bf16_bf16_out": "bf16",
        "prefill_bf16_f32_out": "f32",
    }
    output_dtype = output_variants.get(dispatch.key.variant)
    if output_dtype is None or dispatch.abi != "raw":
        return dispatch
    enabled_quants = backend_package_capability(
        dispatch.key.backend,
        "GGUF_F32_ORDERED_PREFILL_QUANTS",
        frozenset(),
    )
    if not isinstance(enabled_quants, (set, frozenset, tuple, list)):
        return dispatch
    if dispatch.key.quant not in enabled_quants:
        return dispatch
    policies = backend_package_capability(
        dispatch.key.backend,
        "GGUF_F32_ORDERED_PREFILL_POLICIES",
        {},
    )
    if not isinstance(policies, Mapping):
        return dispatch
    policy = policies.get(dispatch.key.quant, {})
    if not isinstance(policy, Mapping):
        return dispatch
    geometry = policy.get((output_dtype, int(in_features), int(out_features)))
    if geometry is None:
        return dispatch
    try:
        required = q5_k_f32_ordered_workspace_nbytes(
            in_features,
            out_features,
        )
    except ValueError:
        return dispatch
    if required > int(session.weight_f32_nbytes):
        return dispatch
    activation_marker = "_activation_tile_k_row_"
    uses_activation_tile_k_row = activation_marker in geometry
    if uses_activation_tile_k_row:
        try:
            row_batch = int(geometry.rsplit("rowbatch", 1)[1])
            activation_required = q5_k_f32_activation_tile_k_row_nbytes(
                rows,
                in_features,
                row_batch,
            )
        except (IndexError, ValueError):
            return dispatch
        if (
            int(session.activation_bf16_ptr) <= 0
            or activation_required > int(session.activation_bf16_nbytes)
        ):
            return dispatch
    raw_weight_ptr = (
        int(weight.allocation("raw").tensor.ptr)
        if weight is not None
        else 0
    )
    resident_plane = (
        session.resident_plane(
            raw_weight_ptr,
            in_features=in_features,
            out_features=out_features,
            output_dtype=output_dtype,
            ordered_geometry=geometry,
        )
        if raw_weight_ptr > 0
        else None
    )
    if resident_plane is not None:
        resident_key = KernelKey(
            dispatch.key.backend,
            dispatch.key.layer,
            dispatch.key.quant,
            f"f32_resident_ordered_{geometry}_bf16_{output_dtype}_out",
        )
        if is_registered(resident_key):
            return GGUFLinearDispatch(
                resident_key,
                "raw_k_f32_resident_activation_tile_k_row",
            )
    key = KernelKey(
        dispatch.key.backend,
        dispatch.key.layer,
        dispatch.key.quant,
        f"f32_ordered_{geometry}_bf16_{output_dtype}_out",
    )
    if not is_registered(key):
        return dispatch
    abi = (
        "raw_k_f32_ordered_activation_tile_k_row"
        if uses_activation_tile_k_row
        else "raw_k_f32_ordered"
    )
    return GGUFLinearDispatch(key, abi)


def _raw_k_prefill_rowbatch_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    row_batch: int,
    variant: str,
) -> GGUFLinearDispatch:
    """Select exact fixed-grid-Y Q5/Q6 row/output reuse above small-B."""

    selected = int(row_batch)
    geometry = str(variant)
    if (
        selected not in (_RAW_K_PREFILL_ROWBATCHES - {0})
        or geometry not in _RAW_K_PREFILL_VARIANTS
        or rows <= _ROWTILE_MAX_ROWS
    ):
        return dispatch
    if (
        not backend_package_capability(
            dispatch.key.backend,
            "GGUF_RAW_K_PREFILL_ROWBATCH_SUPPORTED",
            False,
        )
        or dispatch.abi != "raw"
        or dispatch.key.quant not in _RAW_K_PREFILL_ROWBATCH_QUANTS
        or dispatch.key.variant not in _RAW_K_PREFILL_ROWBATCH_VARIANTS
        or in_features % 256 != 0
    ):
        return dispatch
    output_variant = dispatch.key.variant[len("prefill_") :]
    selected_variant = f"rowbatch{selected}_{output_variant}"
    if (
        geometry == "coltile"
        and selected == 32
        and out_features % 4 == 0
        and backend_package_capability(
            dispatch.key.backend,
            "GGUF_RAW_K_PREFILL_COLTILE_SUPPORTED",
            False,
        )
    ):
        shape_key = (
            dispatch.key.quant,
            output_variant,
            int(in_features),
            int(out_features),
        )
        coltile2_shapes = backend_package_capability(
            dispatch.key.backend,
            "GGUF_RAW_K_PREFILL_COLTILE2_SHAPES",
            frozenset(),
        )
        geometry = (
            "coltile2_rowbatch16"
            if shape_key in coltile2_shapes
            else "coltile4_rowbatch8"
        )
        selected_variant = f"{geometry}_{output_variant}"
        role_variants = backend_package_capability(
            dispatch.key.backend,
            "GGUF_RAW_K_PREFILL_ROLE_VARIANTS",
            {},
        )
        role_key = (
            dispatch.key.quant,
            output_variant,
            int(rows),
            int(in_features),
            int(out_features),
        )
        if isinstance(role_variants, Mapping):
            role_variant = role_variants.get(role_key)
            if isinstance(role_variant, str) and role_variant:
                role_dispatch_key = KernelKey(
                    dispatch.key.backend,
                    dispatch.key.layer,
                    dispatch.key.quant,
                    role_variant,
                )
                if is_registered(role_dispatch_key):
                    selected_variant = role_variant
    return GGUFLinearDispatch(
        KernelKey(
            dispatch.key.backend,
            dispatch.key.layer,
            dispatch.key.quant,
            selected_variant,
        ),
        "raw",
    )


def resolve_gguf_linear_dispatch(
    weight: GGUFDeviceWeight,
    *,
    activation_dtype: str = GGUF_ACTIVATION_BF16,
    output_dtype: str = GGUF_OUTPUT_BF16,
    backend: str | None = None,
    rows: int = 1,
) -> GGUFLinearDispatch:
    """Resolve a GGUF linear launch without model/engine quant branches."""

    resolved_backend = _weight_backend(weight, backend=backend)
    table_key = (weight.spec.layout, activation_dtype, output_dtype)
    try:
        dispatch = _DISPATCH_TABLE[table_key]
    except KeyError as exc:
        raise ValueError(
            "unsupported GGUF linear dispatch: "
            f"layout={weight.spec.layout!r}, activation={activation_dtype!r}, output={output_dtype!r}"
        ) from exc
    quant = weight.spec.quant_key if dispatch.key.quant == "<from-weight>" else dispatch.key.quant
    if weight.spec.layout in {
        LAYOUT_GGUF_Q4_K_T16,
        LAYOUT_GGUF_Q4_K_QMICRO_T16,
    } and rows > 1:
        variant = "t16_wmma_prefill_bf16_bf16_out"
    else:
        variant = _variant_for_rows(dispatch.key.variant, rows=rows)
    return GGUFLinearDispatch(
        KernelKey(resolved_backend, dispatch.key.layer, quant, variant),
        dispatch.abi,
    )


# Memoized launch_gguf_linear dispatch resolution. The resolved (abi, fn) is a
# pure function of the cache key below plus the registry contents; the registry
# generation is part of the key, so any register/unregister invalidates stale
# entries automatically. In production the registry is stable after import, so
# this collapses the ~18us-per-launch dispatch-resolve chain to a dict lookup.
_DISPATCH_RESOLVE_CACHE: dict[tuple, tuple] = {}
_PAIR_DISPATCH_RESOLVE_CACHE: dict[tuple, str] = {}
_Q8_1_DISPATCH_RESOLVE_CACHE: dict[tuple, tuple | bool] = {}


def clear_gguf_linear_dispatch_cache() -> None:
    """Drop all memoized GGUF linear dispatch resolutions.

    Not normally needed (the registry generation in the cache key invalidates
    stale entries automatically); exposed for tests and defensive callers.
    """

    _DISPATCH_RESOLVE_CACHE.clear()
    _PAIR_DISPATCH_RESOLVE_CACHE.clear()
    _Q8_1_DISPATCH_RESOLVE_CACHE.clear()


def _native_split_row_chunk(
    weight: GGUFDeviceWeight,
    *,
    backend: str,
    rows: int,
    in_features: int,
    out_features: int,
) -> int | None:
    """Resolve one backend-qualified exact-row split or fail closed."""

    if not _native_batch_decode_session_enabled:
        return None
    policies = backend_package_capability(
        backend,
        "GGUF_T16_NATIVE_SPLIT_ROW_CHUNKS_BY_QUANT_SHAPE",
        {},
    )
    quant_policy = (
        policies.get(weight.spec.quant_key, {})
        if isinstance(policies, Mapping)
        else {}
    )
    if not isinstance(quant_policy, Mapping):
        return None
    raw_chunk = quant_policy.get((int(rows), int(in_features), int(out_features)))
    try:
        chunk = int(raw_chunk)
    except (TypeError, ValueError):
        return None
    if chunk <= 1 or rows <= chunk or rows % chunk:
        return None
    return chunk


def _target_verifier_true_rowtile_variant(
    weight: GGUFDeviceWeight,
    *,
    backend: str,
    rows: int,
    in_features: int,
    out_features: int,
) -> GGUFLinearDispatch | None:
    """Resolve one package-qualified operation-complete verifier rowtile."""

    if not (
        _target_verifier_rowtile_session_enabled.get()
        and _target_verifier_production_q4_rowtile_session_enabled.get()
    ):
        return None
    policies = backend_package_capability(
        backend,
        "GGUF_T16_TARGET_VERIFIER_TRUE_ROWTILE_VARIANTS",
        {},
    )
    if not isinstance(policies, Mapping):
        raise RuntimeError("target verifier true-rowtile policies must be a mapping")
    variant = policies.get(
        (
            str(weight.spec.quant_key),
            int(rows),
            int(in_features),
            int(out_features),
        )
    )
    if variant is None:
        return None
    variant = str(variant)
    parent = resolve_gguf_linear_dispatch(weight, backend=backend, rows=rows)
    if parent.abi != "t16":
        return None
    key = KernelKey(backend, parent.key.layer, str(weight.spec.quant_key), variant)
    _ensure_linear_kernel_registered(key)
    return GGUFLinearDispatch(key, parent.abi) if is_registered(key) else None


def _target_verifier_wide_q6_shared4_variant(
    weight: GGUFDeviceWeight,
    *,
    backend: str,
    rows: int,
    in_features: int,
    out_features: int,
) -> GGUFLinearDispatch | None:
    """Resolve the default-off W1 B-stationary Q6 verifier candidate."""

    if not (
        _target_verifier_rowtile_session_enabled.get()
        and _target_verifier_production_q4_rowtile_session_enabled.get()
        and _target_verifier_wide_q6_shared4_leaf_enabled.get()
    ):
        return None
    policies = backend_package_capability(
        backend,
        "GGUF_T16_TARGET_VERIFIER_WIDE_Q6_SHARED4_VARIANTS",
        {},
    )
    if not isinstance(policies, Mapping):
        raise RuntimeError("wide Q6 shared4 policies must be a mapping")
    variant = policies.get(
        (
            str(weight.spec.quant_key),
            int(rows),
            int(in_features),
            int(out_features),
        )
    )
    if variant is None:
        return None
    parent = resolve_gguf_linear_dispatch(weight, backend=backend, rows=rows)
    if parent.abi != "t16":
        return None
    key = KernelKey(
        backend,
        parent.key.layer,
        str(weight.spec.quant_key),
        str(variant),
    )
    _ensure_linear_kernel_registered(key)
    return GGUFLinearDispatch(key, parent.abi) if is_registered(key) else None


def _native_rowtile_chunk_groups(
    weight: GGUFDeviceWeight,
    *,
    backend: str,
    rows: int,
    in_features: int,
    out_features: int,
) -> list[tuple[int, int]] | None:
    """Chunk qualified decode/verifier rows 9..511 into rowtile8 groups.

    Native batch decode keeps its prior rule: only projections that would
    otherwise resolve to WMMA are chunked. Packed target verification has a
    narrower production-profile, shape-qualified rule: exact Q5/Q6 verifier
    rowtiles and an explicitly admitted production-Q4 physical row may reuse
    the same row-independent leaves. The original physical row count owns admission;
    child R8/R4 chunks cannot independently broaden production scope.
    """

    rows = int(rows)
    if _q5_raw_mmq_target_eligible(
        weight,
        rows=rows,
        in_features=in_features,
        out_features=out_features,
    ):
        return None
    if not _ROWTILE_MAX_ROWS < rows < _NATIVE_ROWTILE_CHUNK_MAX_ROWS:
        return None
    resolved = resolve_gguf_linear_dispatch(weight, backend=backend, rows=rows)
    native_scope = bool(_native_batch_decode_session_enabled)
    if native_scope and not resolved.key.variant.startswith("t16_wmma_prefill"):
        native_scope = False

    verifier_rowtile_shapes = backend_package_capability(
        backend,
        "GGUF_T16_TARGET_VERIFIER_ROWTILE_SHAPES_BY_QUANT",
        {},
    )
    quant_shapes = (
        verifier_rowtile_shapes.get(weight.spec.quant_key, ())
        if isinstance(verifier_rowtile_shapes, Mapping)
        else ()
    )
    verifier_chunk_rows = backend_package_capability(
        backend,
        "GGUF_T16_TARGET_VERIFIER_ROWTILE_CHUNK_ROWS_BY_QUANT",
        {},
    )
    quant_chunk_rows = (
        verifier_chunk_rows.get(weight.spec.quant_key, ())
        if isinstance(verifier_chunk_rows, Mapping)
        else ()
    )
    verifier_scope = bool(
        _target_verifier_rowtile_session_enabled.get()
        and _target_verifier_production_q4_rowtile_session_enabled.get()
        and rows in quant_chunk_rows
        and (int(in_features), int(out_features)) in quant_shapes
    )
    production_q4_scope = _target_verifier_production_q4_rowtile_scope_enabled(
        resolved,
        rows=rows,
        in_features=in_features,
        out_features=out_features,
    )
    if not (native_scope or verifier_scope or production_q4_scope):
        return None

    candidates = (
        "dense_rowtile_bf16_bf16_out",
        "dense_rowtile_col4_bf16_bf16_out",
        "t16_gemv_rowtile_bf16_bf16_out",
    )
    quant = weight.spec.quant_key
    if not any(
        is_registered(KernelKey(backend, "linear", quant, variant))
        for variant in candidates
    ):
        return None
    return _rowtile8_row_chunks(rows)


def launch_gguf_linear(
    weight: GGUFDeviceWeight,
    x_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    activation_dtype: str = GGUF_ACTIVATION_BF16,
    output_dtype: str = GGUF_OUTPUT_BF16,
    backend: str | None = None,
    threads: int = 0,
    stream: int = 0,
    libraries: Mapping[str, ctypes.CDLL] | None = None,
    runtime=None,
    use_wmma_prefill: bool | None = None,
    use_gemv_decode: bool | None = None,
    use_q4_pack8_wmma: bool = False,
    registered_variant: str | None = None,
) -> None:
    """Launch a GGUF resident linear projection through the kernel registry.

    Hidden projections use ``output_dtype='bf16'``. The tied Q6_K lm-head path
    uses ``output_dtype='f32'`` to produce logits.

    When ``rows > 1`` and the raw-layout quant has a WMMA prefill kernel
    registered (currently ``gguf_q8_0`` and raw ``gguf_q4_k``), the dispatch
    rewrites to the ``wmma_prefill_*`` family if any of these is true. The same
    opt-in selects the registered arithmetic-preserving resident-pack8 tile8x8
    leaf for populated Q4_K chunks (rows >= 512):

    * ``use_wmma_prefill=True`` is passed explicitly,
    * a runner has called :func:`set_wmma_prefill_enabled` with ``True``,
    * the env var ``HIPENGINE_GGUF_WMMA_PREFILL`` is set.

    Otherwise aligned raw-Q8 BF16 projections use the exact pack8/row-tiled
    schedule selected for their row and output shape; other inputs retain the
    existing decode-shaped ``prefill_*`` aliases.
    """

    resolved_backend = _weight_backend(weight, backend=backend)
    wide_q6_dispatch = (
        _target_verifier_wide_q6_shared4_variant(
            weight,
            backend=resolved_backend,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        if activation_dtype == GGUF_ACTIVATION_BF16
        and output_dtype == GGUF_OUTPUT_BF16
        and threads == 0
        and not use_q4_pack8_wmma
        and registered_variant is None
        else None
    )
    split_row_chunk = (
        _native_split_row_chunk(
            weight,
            backend=resolved_backend,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        if activation_dtype == GGUF_ACTIVATION_BF16
        and output_dtype == GGUF_OUTPUT_BF16
        and threads == 0
        and not use_q4_pack8_wmma
        and registered_variant is None
        and wide_q6_dispatch is None
        else None
    )
    if split_row_chunk is not None:
        element_nbytes = DType.BF16.itemsize
        for row_start in range(0, rows, split_row_chunk):
            launch_gguf_linear(
                weight,
                x_ptr + row_start * in_features * element_nbytes,
                out_ptr + row_start * out_features * element_nbytes,
                split_row_chunk,
                in_features,
                out_features,
                activation_dtype=activation_dtype,
                output_dtype=output_dtype,
                backend=resolved_backend,
                stream=stream,
                libraries=libraries,
                runtime=runtime,
                use_wmma_prefill=use_wmma_prefill,
                use_gemv_decode=use_gemv_decode,
            )
        return
    if wide_q6_dispatch is not None:
        fn = resolve(
            backend=wide_q6_dispatch.key.backend,
            layer=wide_q6_dispatch.key.layer,
            quant=wide_q6_dispatch.key.quant,
            variant=wide_q6_dispatch.key.variant,
        )
        library = None
        if libraries is not None:
            library = libraries.get(
                f"{wide_q6_dispatch.key.quant}:{wide_q6_dispatch.key.variant}",
                libraries.get(wide_q6_dispatch.key.quant),
            )
        kwargs = {"stream": stream, "runtime": runtime}
        if library is not None:
            kwargs["library"] = library
        _LAUNCH_ABI[wide_q6_dispatch.abi](
            fn,
            weight,
            x_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            kwargs,
        )
        return
    true_rowtile_dispatch = (
        _target_verifier_true_rowtile_variant(
            weight,
            backend=resolved_backend,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        if activation_dtype == GGUF_ACTIVATION_BF16
        and output_dtype == GGUF_OUTPUT_BF16
        and threads == 0
        and not use_q4_pack8_wmma
        and registered_variant is None
        else None
    )
    if true_rowtile_dispatch is not None:
        fn = resolve(
            backend=true_rowtile_dispatch.key.backend,
            layer=true_rowtile_dispatch.key.layer,
            quant=true_rowtile_dispatch.key.quant,
            variant=true_rowtile_dispatch.key.variant,
        )
        library = None
        if libraries is not None:
            library = libraries.get(
                f"{true_rowtile_dispatch.key.quant}:"
                f"{true_rowtile_dispatch.key.variant}",
                libraries.get(true_rowtile_dispatch.key.quant),
            )
        kwargs = {"stream": stream, "runtime": runtime}
        if library is not None:
            kwargs["library"] = library
        _LAUNCH_ABI[true_rowtile_dispatch.abi](
            fn,
            weight,
            x_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            kwargs,
        )
        return
    native_rowtile_groups = (
        _native_rowtile_chunk_groups(
            weight,
            backend=resolved_backend,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        if activation_dtype == GGUF_ACTIVATION_BF16
        and output_dtype == GGUF_OUTPUT_BF16
        and threads == 0
        and not use_q4_pack8_wmma
        and registered_variant is None
        else None
    )
    if native_rowtile_groups is not None:
        element_nbytes = DType.BF16.itemsize
        token = _target_verifier_rowtile_chunk_child_enabled.set(True)
        try:
            for chunk_rows, row_base in native_rowtile_groups:
                launch_gguf_linear(
                    weight,
                    x_ptr + row_base * in_features * element_nbytes,
                    out_ptr + row_base * out_features * element_nbytes,
                    chunk_rows,
                    in_features,
                    out_features,
                    activation_dtype=activation_dtype,
                    output_dtype=output_dtype,
                    backend=resolved_backend,
                    stream=stream,
                    libraries=libraries,
                    runtime=runtime,
                    use_wmma_prefill=use_wmma_prefill,
                    use_gemv_decode=use_gemv_decode,
                )
        finally:
            _target_verifier_rowtile_chunk_child_enabled.reset(token)
        return
    f_gemv = _resolve_use_gemv_decode(use_gemv_decode)
    use_wmma = _resolve_use_wmma_prefill(use_wmma_prefill)
    f_rowtile = (not use_wmma) and _resolve_use_q4k_rowtile(None)
    if (
        _native_batch_decode_session_enabled
        and (2 <= rows <= 8)
        and not use_wmma
        and not use_q4_pack8_wmma
        and registered_variant is None
        and threads == 0
        and activation_dtype == GGUF_ACTIVATION_BF16
        and output_dtype == GGUF_OUTPUT_BF16
        and launch_gguf_q4_t16_sidecar_decode(
            weight,
            x_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            backend=resolved_backend,
            stream=stream,
            libraries=libraries,
            runtime=runtime,
        )
    ):
        return
    raw_k_rowbatch = raw_k_prefill_rowbatch()
    raw_k_variant = raw_k_prefill_variant()
    mmq_session = _q8_mmq_prefill_session.get()
    q5_raw_mmq_session = _q5_raw_mmq_target_session.get()
    q5_f32_ordered_session = _q5_f32_ordered_prefill_session.get()
    try:
        has_raw_weight_sidecar = int(weight.allocation("raw").tensor.ptr) > 0
    except KeyError:
        has_raw_weight_sidecar = False
    raw_weight_ptr = (
        int(weight.allocation("raw").tensor.ptr)
        if weight.spec.layout == LAYOUT_RAW_GGUF
        else None
    )
    cache_key = (
        generation(),
        weight.spec.layout,
        weight.spec.quant_key,
        rows,
        in_features,
        out_features,
        activation_dtype,
        output_dtype,
        resolved_backend,
        f_gemv,
        use_wmma,
        f_rowtile,
        raw_k_rowbatch,
        raw_k_variant,
        bool(use_q4_pack8_wmma),
        os.environ.get("HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK", "1") != "0",
        registered_variant,
        bool(_native_batch_decode_session_enabled),
        None if mmq_session is None else id(mmq_session),
        None if q5_raw_mmq_session is None else id(q5_raw_mmq_session),
        (
            None
            if q5_f32_ordered_session is None
            else id(q5_f32_ordered_session)
        ),
        (
            None
            if (q6_f16_session := _t16_f16_rocblas_prefill_session.get())
            is None
            else id(q6_f16_session)
        ),
        (
            None
            if (q6_integer_workspace := q6_dense_integer_mmq_workspace())
            is None
            else id(q6_integer_workspace)
        ),
        raw_weight_ptr,
        has_raw_weight_sidecar,
    )
    cached = _DISPATCH_RESOLVE_CACHE.get(cache_key)
    if cached is None:
        dispatch = resolve_gguf_linear_dispatch(
            weight,
            activation_dtype=activation_dtype,
            output_dtype=output_dtype,
            backend=resolved_backend,
            rows=rows,
        )
        dispatch = _pack8_decode_dispatch(dispatch, rows=rows, out_features=out_features)
        dispatch = _gemv_decode_dispatch(dispatch, rows=rows, use_gemv_decode=f_gemv)
        dispatch = _registered_variant_dispatch(
            dispatch,
            rows=rows,
            variant=registered_variant,
        )
        dispatch = _q4_t16_dense_native_dispatch(
            dispatch,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        dispatch = _t16_c1_variant_dispatch(
            dispatch,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        dispatch = _native_batch_decode_dispatch(
            dispatch,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        dispatch = _q6_planar_rowtile_dispatch(
            dispatch,
            rows=rows,
            use_wmma=use_wmma,
        )
        # The small-B row-tile path is the weight-amortized replacement for the
        # per-row (non-WMMA) prefill alias. It does not override an explicit WMMA
        # opt-in: only fires when WMMA is off (e.g. the small-B target verifier).
        dispatch = _wmma_prefill_dispatch(
            dispatch,
            rows=rows,
            in_features=in_features,
            use_wmma=use_wmma,
        )
        dispatch = _q6_t16_f16_rocblas_prefill_dispatch(
            dispatch,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        dispatch = _q6_integer_mmq_prefill_dispatch(
            dispatch,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        dispatch = _pack8_exact_prefill_dispatch(
            dispatch,
            rows=rows,
            enabled=use_wmma and not use_q4_pack8_wmma,
        )
        dispatch = _dense_bf16_rowtile_dispatch(
            dispatch,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            enabled=not use_wmma,
            native_batch=_native_batch_decode_session_enabled,
        )
        dispatch = _dense_bf16_wmma_dispatch(
            dispatch,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            enabled=(
                use_wmma
                and os.environ.get("HIPENGINE_GGUF_DENSE_WMMA_BULK", "1") != "0"
                and bool(
                    backend_package_capability(
                        resolved_backend,
                        "GGUF_DENSE_BF16_WMMA_BULK_PREFILL",
                        False,
                    )
                )
                and _backend_prefill_shape_is_qualified(
                    resolved_backend,
                    "GGUF_DENSE_BF16_WMMA_BULK_PREFILL_SHAPES",
                    rows=rows,
                    in_features=in_features,
                    out_features=out_features,
                )
            ),
        )
        dispatch = _pack8_rowtile_dispatch(
            dispatch,
            rows=rows,
            use_rowtile=f_rowtile and not use_q4_pack8_wmma,
            native_batch=_native_batch_decode_session_enabled,
        )
        dispatch = _q8_mmq_prefill_dispatch(
            dispatch,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        dispatch = _exact_q8_prefill_dispatch(
            dispatch,
            rows=rows,
            out_features=out_features,
        )
        dispatch = _rowtile_dispatch(
            dispatch,
            rows=rows,
            in_features=in_features,
            use_rowtile=f_rowtile,
        )
        dispatch = _raw_k_f32_ordered_prefill_dispatch(
            dispatch,
            rows=rows,
            weight=weight,
            in_features=in_features,
            out_features=out_features,
        )
        dispatch = _raw_k_prefill_rowbatch_dispatch(
            dispatch,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            row_batch=raw_k_rowbatch,
            variant=raw_k_variant,
        )
        dispatch = _q4_pack8_wmma_dispatch(
            dispatch,
            rows=rows,
            enabled=(
                use_q4_pack8_wmma
                or (
                    use_wmma
                    and bool(
                        backend_package_capability(
                            resolved_backend,
                            "GGUF_Q4_PACK8_WMMA_BULK_PREFILL",
                            False,
                        )
                    )
                    and os.environ.get(
                        "HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK", "1"
                    )
                    != "0"
                    and _backend_prefill_shape_is_qualified(
                        resolved_backend,
                        "GGUF_Q4_PACK8_WMMA_BULK_PREFILL_SHAPES",
                        rows=rows,
                        in_features=in_features,
                        out_features=out_features,
                    )
                )
            ),
        )
        dispatch = _q5_raw_mmq_target_dispatch(
            dispatch,
            weight=weight,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        _ensure_linear_kernel_registered(dispatch.key)
        fn = resolve(
            backend=dispatch.key.backend,
            layer=dispatch.key.layer,
            quant=dispatch.key.quant,
            variant=dispatch.key.variant,
        )
        cached = (
            dispatch.abi,
            fn,
            dispatch.key.layer,
            dispatch.key.quant,
            dispatch.key.variant,
        )
        _DISPATCH_RESOLVE_CACHE[cache_key] = cached
    abi, fn, layer, quant, variant = cached
    library = None
    if libraries is not None:
        library = libraries.get(f"{quant}:{variant}", libraries.get(quant))
    kwargs = {"stream": stream, "runtime": runtime}
    if abi == "t16" and quant == "gguf_q8_0_t16_v1":
        q8_t16_threads = _resolve_q8_t16_threads(threads)
        if q8_t16_threads:
            kwargs["threads"] = q8_t16_threads
    elif threads:
        kwargs["threads"] = threads
    if library is not None:
        kwargs["library"] = library
    if (
        abi == "t16"
        and quant == "gguf_q8_0_t16_v1"
        and activation_dtype == GGUF_ACTIVATION_BF16
        and output_dtype == GGUF_OUTPUT_BF16
        and _use_q8_t16_all_rowtile(
            rows=rows,
            in_features=in_features,
            threads=threads,
        )
    ):
        gguf_q8_0_t16_gemv_decode_rowtile4_bf16_bf16_out(
            x_ptr,
            weight.allocation("tiles").tensor.ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            threads=_Q8_T16_ROWTILE_THREADS,
            **kwargs,
        )
        return
    if (
        prefill_f16_staging_enabled()
        and abi == "t16"
        and quant in _PREFILL_F16_STAGING_QUANTS
        and variant == _PREFILL_F16_STAGING_SOURCE_VARIANT
        and _PREFILL_F16_STAGING_MIN_ROWS <= int(rows)
        <= PREFILL_F16_STAGING_MAX_ROWS
        and activation_dtype == GGUF_ACTIVATION_BF16
        and output_dtype == GGUF_OUTPUT_BF16
    ):
        if _launch_prefill_f16_staged(
            fn,
            weight,
            x_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            kwargs,
            backend=resolved_backend,
            quant=quant,
            layer=layer,
            runtime=runtime,
        ):
            return
    _LAUNCH_ABI[abi](fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs)


def _q4_t16_sidecar_allocation(
    weight: GGUFDeviceWeight,
    *,
    rows: int,
    allow_r3plus_at_row2: bool = False,
):
    """Resolve canonical Q4T16 tiles or a legacy sidecar at its row floor."""

    if weight.spec.quant_key == "gguf_q4_k_t16_v1":
        try:
            return weight.allocation("tiles")
        except KeyError:
            return None
    for allocation_name, min_rows in (
        (Q4_T16_DECODE_TILES, 1),
        (Q4_T16_DECODE_TILES_R3PLUS, 3),
    ):
        if (
            rows < min_rows
            and not (
                allow_r3plus_at_row2
                and rows == 2
                and allocation_name == Q4_T16_DECODE_TILES_R3PLUS
            )
        ):
            continue
        try:
            return weight.allocation(allocation_name)
        except KeyError:
            pass
    return None


def _q4_t16_sidecar_decode_variants(
    *,
    rows: int,
    in_features: int,
    out_features: int,
    backend: str | None = None,
    canonical: bool = False,
) -> tuple[str, ...]:
    """Rank exact rowtile variants by measured backend shape policy."""

    if rows == 1:
        return ("dense_single_local32_bf16_bf16_out",)
    if not 2 <= rows <= 8:
        return ()
    shape = (in_features, out_features)
    if rows <= 4 and shape in _Q4_T16_COL4_ALL_ROWS_SHAPES:
        variants = (
            "dense_rowtile_col4_bf16_bf16_out",
            "dense_rowtile_bf16_bf16_out",
        )
    else:
        variants = ("dense_rowtile_bf16_bf16_out",)
    if backend is None:
        return variants
    rowtile_variants = backend_package_capability(
        backend,
        "GGUF_T16_NATIVE_ROWTILE_VARIANTS_BY_QUANT",
        {},
    )
    variant_policy = (
        rowtile_variants.get("gguf_q4_k_t16_v1")
        if isinstance(rowtile_variants, Mapping)
        else None
    )
    if canonical and not (
        isinstance(variant_policy, Mapping)
        and bool(variant_policy.get("canonical", False))
    ):
        return variants
    if isinstance(variant_policy, Mapping):
        enabled_env = variant_policy.get("enabled_env")
        if isinstance(enabled_env, str) and enabled_env:
            enabled_default = bool(variant_policy.get("enabled_default", False))
            cache_key = (enabled_env, enabled_default)
            enabled = _rowtile_variant_policy_env_cache.get(cache_key)
            if enabled is None:
                raw = os.environ.get(
                    enabled_env,
                    "1" if enabled_default else "0",
                ).strip().lower()
                if raw in {"1", "true", "yes", "on"}:
                    enabled = True
                elif raw in {"0", "false", "no", "off"}:
                    enabled = False
                else:
                    raise ValueError(f"{enabled_env} must be a boolean value")
                _rowtile_variant_policy_env_cache[cache_key] = enabled
            if not enabled:
                return variants
    shapes = (
        variant_policy.get("shapes", {})
        if isinstance(variant_policy, Mapping)
        else {}
    )
    rows_by_shape = (
        variant_policy.get("rows_by_shape", {})
        if isinstance(variant_policy, Mapping)
        else {}
    )
    preferred = shapes.get(shape) if isinstance(shapes, Mapping) else None
    allowed_rows = (
        rows_by_shape.get(shape, (8,))
        if isinstance(rows_by_shape, Mapping)
        else (8,)
    )
    if isinstance(preferred, str) and preferred and int(rows) in allowed_rows:
        return (preferred, *variants)
    return variants


def _target_verifier_production_q4_rowtile_scope_enabled(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> bool:
    if (
        not _target_verifier_production_q4_rowtile_session_enabled.get()
        or dispatch.abi != "t16"
        or dispatch.key.quant != "gguf_q4_k_t16_v1"
    ):
        return False
    admitted_rows = backend_package_capability(
        dispatch.key.backend,
        "GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ROWS",
        (),
    )
    shapes = backend_package_capability(
        dispatch.key.backend,
        "GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_SHAPES",
        (),
    )
    try:
        row_admitted = int(rows) in admitted_rows or bool(
            _target_verifier_rowtile_chunk_child_enabled.get()
            and 2 <= int(rows) <= _ROWTILE_MAX_ROWS
        )
        return row_admitted and (
            int(in_features),
            int(out_features),
        ) in shapes
    except TypeError:
        return False


def _target_verifier_production_q4_pair_key(
    dispatch_a: GGUFLinearDispatch,
    dispatch_b: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> KernelKey | None:
    """Resolve a package-qualified operation-complete verifier pair."""

    if not (
        _target_verifier_production_q4_rowtile_scope_enabled(
            dispatch_a,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        and _target_verifier_production_q4_rowtile_scope_enabled(
            dispatch_b,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        and dispatch_a.key.quant == dispatch_b.key.quant
    ):
        return None
    table = backend_package_capability(
        dispatch_a.key.backend,
        "GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_PAIR_VARIANTS",
        {},
    )
    if not isinstance(table, Mapping):
        raise RuntimeError("production Q4 verifier pair variants must be a mapping")
    variant = table.get((int(rows), int(in_features), int(out_features)))
    if variant is None:
        return None
    key = KernelKey(
        dispatch_a.key.backend,
        "linear_pair_silu",
        dispatch_a.key.quant,
        str(variant),
    )
    _ensure_linear_kernel_registered(key)
    return key if is_registered(key) else None


def _q4_t16_dense_native_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> GGUFLinearDispatch:
    """Select the quant-qualified small-row leaf for a Q4T16 owner."""

    max_rows = int(
        _Q4_T16_DENSE_ROWTILE_MAX_ROWS_BY_QUANT.get(
            dispatch.key.quant,
            0,
        )
    )
    verifier_candidate = _target_verifier_production_q4_rowtile_scope_enabled(
        dispatch,
        rows=rows,
        in_features=in_features,
        out_features=out_features,
    )
    if (
        not (_native_batch_decode_session_enabled or verifier_candidate)
        or dispatch.abi != "t16"
        or not 2 <= rows <= max_rows
    ):
        return dispatch
    for variant in _q4_t16_sidecar_decode_variants(
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        backend=dispatch.key.backend,
        canonical=True,
    ):
        key = KernelKey(
            dispatch.key.backend,
            dispatch.key.layer,
            dispatch.key.quant,
            variant,
        )
        _ensure_linear_kernel_registered(key)
        if is_registered(key):
            return GGUFLinearDispatch(key, dispatch.abi)
    return dispatch


def launch_gguf_q4_t16_sidecar_decode(
    weight: GGUFDeviceWeight,
    x_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    backend: str | None = None,
    stream: int = 0,
    libraries: Mapping[str, ctypes.CDLL] | None = None,
    runtime=None,
    enabled: bool = True,
) -> bool:
    """Launch an exact Q4T16 c1 or measured rows-2-4 resident owner."""

    if not enabled or weight.spec.quant_key not in {
        "gguf_q4_k",
        "gguf_q4_k_t16_v1",
    }:
        return False
    resolved_backend = _weight_backend(weight, backend=backend)
    variants = _q4_t16_sidecar_decode_variants(
        rows=rows,
        in_features=in_features,
        out_features=out_features,
        backend=resolved_backend,
    )
    if not variants:
        return False
    for variant in variants:
        key = KernelKey(
            resolved_backend,
            "linear",
            "gguf_q4_k_t16_v1",
            variant,
        )
        _ensure_linear_kernel_registered(key)
        if not is_registered(key):
            continue
        sidecar = _q4_t16_sidecar_allocation(
            weight,
            rows=rows,
            allow_r3plus_at_row2=(
                variant == "dense_rowtile_col4_bf16_bf16_out"
            ),
        )
        if sidecar is None:
            continue
        fn = resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        kwargs = {"stream": stream, "runtime": runtime}
        if libraries is not None:
            library = libraries.get(
                f"{key.quant}:{key.variant}",
                libraries.get(key.quant),
            )
            if library is not None:
                kwargs["library"] = library
        fn(
            x_ptr,
            sidecar.tensor.ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            **kwargs,
        )
        return True
    return False


def _linear_residual_variant(variant: str) -> str | None:
    """Return the same-ABI rounded-BF16 residual sibling name."""

    if variant == "out":
        return "out_bf16_residual_bf16_out"
    if variant == "prefill_wmma_out":
        return "prefill_wmma_out_bf16_residual_bf16_out"
    suffix = "_bf16_bf16_out"
    if not variant.endswith(suffix):
        return None
    return f"{variant[: -len(suffix)]}_bf16_residual_bf16_out"


def _resolve_registered_linear_residual(
    normal_key: KernelKey,
    *,
    rows: int,
):
    """Resolve one composite only when its exact primitive sibling exists."""

    fused_variant = _linear_residual_variant(normal_key.variant)
    if fused_variant is None:
        return None
    residual_max_rows = 4
    residual_limits = backend_package_capability(
        normal_key.backend,
        "GGUF_LINEAR_RESIDUAL_MAX_ROWS_BY_QUANT",
        {},
    )
    if isinstance(residual_limits, Mapping):
        try:
            residual_max_rows = int(
                residual_limits.get(normal_key.quant, residual_max_rows)
            )
        except (TypeError, ValueError):
            residual_max_rows = 0
    if rows > residual_max_rows:
        return None
    _ensure_linear_kernel_registered(normal_key)
    if not is_registered(normal_key):
        return None
    fused_key = KernelKey(
        normal_key.backend,
        "linear+residual",
        normal_key.quant,
        fused_variant,
    )
    # Ensuring the primitive restores its whole owning module after registry
    # tests clear global state, including any supported composite siblings.
    # Do not bootstrap again for an unsupported composite: dense models with a
    # different quant must fail closed without re-registering every kernel.
    if not is_registered(fused_key):
        return None
    fn = resolve(
        backend=fused_key.backend,
        layer=fused_key.layer,
        quant=fused_key.quant,
        variant=fused_key.variant,
    )
    return fused_key, fn


def _launch_registered_linear_residual(
    normal_key: KernelKey,
    weight_ptr: int,
    x_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    stream: int,
    libraries: Mapping[str, ctypes.CDLL] | None,
    runtime,
) -> bool:
    """Launch one exact composite only when its primitive owner also exists."""

    resolved = _resolve_registered_linear_residual(normal_key, rows=rows)
    if resolved is None:
        return False
    fused_key, fn = resolved
    kwargs = {"stream": stream, "runtime": runtime}
    if libraries is not None:
        library = libraries.get(
            f"{fused_key.quant}:{fused_key.variant}",
            libraries.get(
                f"{normal_key.quant}:{normal_key.variant}",
                libraries.get(fused_key.quant),
            ),
        )
        if library is not None:
            kwargs["library"] = library
    fn(
        x_ptr,
        int(weight_ptr),
        residual_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )
    return True


def launch_gguf_linear_q8_1(
    weight: GGUFDeviceWeight,
    x_ptr: int,
    q8_1_workspace_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    residual_ptr: int | None = None,
    backend: str | None = None,
    stream: int = 0,
    libraries: Mapping[str, ctypes.CDLL] | None = None,
    runtime=None,
    enabled: bool = True,
) -> bool:
    """Launch a registered BF16->Q8_1 plus compressed linear alternative.

    The caller supplies reusable workspace and retains the ordinary registered
    BF16 linear (plus BF16 add, when requested) as the fail-closed fallback.
    No quant or backend identity is selected here: the weight's four-axis key
    must independently register the q8-input primitive.
    """

    if (
        not enabled
        or not q8_1_workspace_ptr
        or rows < 1
        or rows > 4
        or (residual_ptr is not None and rows < 2)
    ):
        return False
    resolved_backend = _weight_backend(weight, backend=backend)
    has_residual = residual_ptr is not None
    cache_key = (
        generation(),
        weight.spec.layout,
        weight.spec.quant_key,
        resolved_backend,
        has_residual,
        rows,
        in_features,
        out_features,
    )
    cached = _Q8_1_DISPATCH_RESOLVE_CACHE.get(cache_key)
    if cached is None:
        layer = "linear_q8_1+residual" if has_residual else "linear_q8_1"
        variant = (
            "t16_q8_1_dp4a_gemv_bf16_residual_bf16_out"
            if has_residual
            else "t16_q8_1_dp4a_gemv_bf16_bf16_out"
        )
        key = KernelKey(
            resolved_backend,
            layer,
            weight.spec.quant_key,
            variant,
        )
        if not is_registered(key):
            _Q8_1_DISPATCH_RESOLVE_CACHE[cache_key] = False
            return False
        dispatch = resolve_gguf_linear_dispatch(
            weight,
            backend=resolved_backend,
            rows=1,
        )
        allocation_name = {"t16": "tiles"}.get(dispatch.abi)
        if allocation_name is None:
            _Q8_1_DISPATCH_RESOLVE_CACHE[cache_key] = False
            return False
        fn = resolve(
            backend=key.backend,
            layer=key.layer,
            quant=key.quant,
            variant=key.variant,
        )
        supports = getattr(fn, "_hipengine_supports", None)
        if callable(supports) and not bool(
            supports(rows, in_features, out_features)
        ):
            _Q8_1_DISPATCH_RESOLVE_CACHE[cache_key] = False
            return False
        cached = (fn, allocation_name, key.quant, key.variant)
        _Q8_1_DISPATCH_RESOLVE_CACHE[cache_key] = cached
    if cached is False:
        return False
    fn, allocation_name, quant, variant = cached
    try:
        allocation = weight.allocation(allocation_name)
    except KeyError:
        return False
    gguf_q4_k_quantize_bf16_q8_1(
        x_ptr,
        q8_1_workspace_ptr,
        rows,
        in_features,
        stream=stream,
        runtime=runtime,
    )
    kwargs = {"stream": stream, "runtime": runtime}
    if libraries is not None:
        library = libraries.get(
            f"{quant}:{variant}",
            libraries.get(quant),
        )
        if library is not None:
            kwargs["library"] = library
    if residual_ptr is None:
        fn(
            q8_1_workspace_ptr,
            allocation.tensor.ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            **kwargs,
        )
    else:
        fn(
            q8_1_workspace_ptr,
            allocation.tensor.ptr,
            residual_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            **kwargs,
        )
    return True


def launch_gguf_linear_residual(
    weight: GGUFDeviceWeight,
    x_ptr: int,
    residual_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    backend: str | None = None,
    stream: int = 0,
    libraries: Mapping[str, ctypes.CDLL] | None = None,
    runtime=None,
    registered_decode: bool = False,
) -> bool:
    """Launch an exact projection plus rounded-BF16 residual composite.

    Registry-qualified native rows 2-4 use their existing session policy. A
    caller may independently admit one model/shape-qualified c1 owner through
    ``registered_decode``. Qualified bulk dense-BF16 WMMA shapes may own the
    same rounded boundary; every ABI, shape, registry, and policy miss fails
    closed to the ordinary projection plus ``gguf_bf16_add`` chain.
    """

    if rows < 1:
        return False
    resolved_backend = _weight_backend(weight, backend=backend)
    if rows > 4:
        # Bulk residual fusion is currently a dense-BF16 WMMA composite. Fail
        # closed before registry/capability work for pack8/raw/T16 owners; their
        # primitive chain remains the route and pays no candidate dispatch tax.
        if weight.spec.layout != LAYOUT_DENSE_BF16:
            return False
        if (
            not _resolve_use_wmma_prefill(None)
            or os.environ.get("HIPENGINE_GGUF_DENSE_WMMA_RESIDUAL", "1") == "0"
        ):
            return False
        dispatch = resolve_gguf_linear_dispatch(
            weight,
            backend=resolved_backend,
            rows=rows,
        )
        dispatch = _dense_bf16_wmma_dispatch(
            dispatch,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            enabled=(
                os.environ.get("HIPENGINE_GGUF_DENSE_WMMA_BULK", "1") != "0"
                and bool(
                    backend_package_capability(
                        resolved_backend,
                        "GGUF_DENSE_BF16_WMMA_BULK_PREFILL",
                        False,
                    )
                )
                and _backend_prefill_shape_is_qualified(
                    resolved_backend,
                    "GGUF_DENSE_BF16_WMMA_BULK_PREFILL_SHAPES",
                    rows=rows,
                    in_features=in_features,
                    out_features=out_features,
                )
            ),
        )
        resolved = _resolve_registered_linear_residual(
            dispatch.key,
            rows=rows,
        )
        launcher = _LAUNCH_RESIDUAL_ABI.get(dispatch.abi)
        if resolved is None or launcher is None:
            return False
        fused_key, fn = resolved
        kwargs = {"stream": stream, "runtime": runtime}
        if libraries is not None:
            library = libraries.get(
                f"{fused_key.quant}:{fused_key.variant}",
                libraries.get(
                    f"{dispatch.key.quant}:{dispatch.key.variant}",
                    libraries.get(fused_key.quant),
                ),
            )
            if library is not None:
                kwargs["library"] = library
        launcher(
            fn,
            weight,
            x_ptr,
            residual_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            kwargs,
        )
        return True
    if rows == 1:
        # ``registered_decode`` is an explicit model/backend/shape admission.
        # A session's independent WMMA-prefill axis must not suppress that
        # rows-1 GEMV composite in mixed prefill+decode workloads.
        if not registered_decode:
            return False
        dispatch = resolve_gguf_linear_dispatch(
            weight,
            backend=resolved_backend,
            rows=rows,
        )
        resolved = _resolve_registered_linear_residual(
            dispatch.key,
            rows=rows,
        )
        launcher = _LAUNCH_RESIDUAL_ABI.get(dispatch.abi)
        if resolved is None or launcher is None:
            return False
        fused_key, fn = resolved
        kwargs = {"stream": stream, "runtime": runtime}
        if libraries is not None:
            library = libraries.get(
                f"{fused_key.quant}:{fused_key.variant}",
                libraries.get(
                    f"{dispatch.key.quant}:{dispatch.key.variant}",
                    libraries.get(fused_key.quant),
                ),
            )
            if library is not None:
                kwargs["library"] = library
        launcher(
            fn,
            weight,
            x_ptr,
            residual_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            kwargs,
        )
        return True
    if _resolve_use_wmma_prefill(None):
        return False
    if not _native_batch_decode_session_enabled:
        return False

    # Legacy compact-Q4 weights expose T16 as a sidecar. Try those registered
    # sibling keys first; sole-T16 owners continue through the canonical route.
    for variant in _q4_t16_sidecar_decode_variants(
        rows=rows,
        in_features=in_features,
        out_features=out_features,
    ):
        sidecar = _q4_t16_sidecar_allocation(
            weight,
            rows=rows,
            allow_r3plus_at_row2=(
                variant == "dense_rowtile_col4_bf16_bf16_out"
            ),
        )
        if sidecar is None:
            continue
        normal_key = KernelKey(
            resolved_backend,
            "linear",
            "gguf_q4_k_t16_v1",
            variant,
        )
        if _launch_registered_linear_residual(
            normal_key,
            sidecar.tensor.ptr,
            x_ptr,
            residual_ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            stream=stream,
            libraries=libraries,
            runtime=runtime,
        ):
            return True

    # Sole-resident T16 owners derive the composite from the same registry key
    # selected by native small-row linear dispatch.
    dispatch = resolve_gguf_linear_dispatch(
        weight,
        backend=resolved_backend,
        rows=rows,
    )
    dispatch = _q4_t16_dense_native_dispatch(
        dispatch,
        rows=rows,
        in_features=in_features,
        out_features=out_features,
    )
    dispatch = _native_batch_decode_dispatch(
        dispatch,
        rows=rows,
        in_features=in_features,
        out_features=out_features,
    )
    allocation_name = {"t16": "tiles"}.get(dispatch.abi)
    if allocation_name is None:
        return False
    try:
        allocation = weight.allocation(allocation_name)
    except KeyError:
        return False
    return _launch_registered_linear_residual(
        dispatch.key,
        allocation.tensor.ptr,
        x_ptr,
        residual_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        libraries=libraries,
        runtime=runtime,
    )


def launch_gguf_linear_moe_tail_host_batch(
    weight: GGUFDeviceWeight,
    x_ptr: int,
    shared_out_ptr: int,
    routed_ptr: int,
    post_attention_ptr: int,
    norm_weight_ptr: int,
    norm_out_ptr: int,
    hidden_out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    eps: float,
    backend: str | None = None,
    stream: int = 0,
    libraries: Mapping[str, ctypes.CDLL] | None = None,
    runtime=None,
    use_gemv_decode: bool | None = None,
    use_q4_t16_sidecar: bool = False,
) -> bool:
    """Enqueue existing shared-down and D9 kernels through one native host call."""

    if rows != 1 or libraries is None:
        return False
    resolved_backend = _weight_backend(weight, backend=backend)
    if (
        use_q4_t16_sidecar
        and weight.spec.quant_key == "gguf_q4_k"
        and "decode_tiles" in weight.allocations
    ):
        t16_key = KernelKey(
            resolved_backend,
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_single_local32_bf16_bf16_out",
        )
        _ensure_linear_kernel_registered(t16_key)
        batch_key = KernelKey(
            resolved_backend,
            "linear+moe_tail+next_rmsnorm_host_batch",
            t16_key.quant,
            t16_key.variant,
        )
        if not is_registered(batch_key):
            return False
        batch_fn = resolve(
            backend=batch_key.backend,
            layer=batch_key.layer,
            quant=batch_key.quant,
            variant=batch_key.variant,
        )
        linear_library = libraries.get(
            f"{t16_key.quant}:{t16_key.variant}",
            libraries.get(t16_key.quant),
        )
        batch_library = libraries.get("launch_batch")
        tail_library = libraries.get("moe_tail")
        if (
            linear_library is None
            or batch_library is None
            or tail_library is None
        ):
            return False
        batch_fn(
            getattr(linear_library, batch_fn.projection_symbol),
            getattr(tail_library, batch_fn.tail_symbol),
            x_ptr,
            weight.allocation("decode_tiles").tensor.ptr,
            shared_out_ptr,
            routed_ptr,
            post_attention_ptr,
            norm_weight_ptr,
            norm_out_ptr,
            hidden_out_ptr,
            rows,
            in_features,
            out_features,
            eps=eps,
            stream=stream,
            library=batch_library,
            runtime=runtime,
        )
        return True
    dispatch = resolve_gguf_linear_dispatch(
        weight,
        backend=resolved_backend,
        rows=rows,
    )
    dispatch = _pack8_decode_dispatch(
        dispatch,
        rows=rows,
        out_features=out_features,
    )
    dispatch = _gemv_decode_dispatch(
        dispatch,
        rows=rows,
        use_gemv_decode=_resolve_use_gemv_decode(use_gemv_decode),
    )
    _ensure_linear_kernel_registered(dispatch.key)
    batch_key = KernelKey(
        dispatch.key.backend,
        "linear+moe_tail+next_rmsnorm_host_batch",
        dispatch.key.quant,
        dispatch.key.variant,
    )
    if not is_registered(batch_key):
        return False
    batch_fn = resolve(
        backend=batch_key.backend,
        layer=batch_key.layer,
        quant=batch_key.quant,
        variant=batch_key.variant,
    )
    linear_library = libraries.get(
        f"{dispatch.key.quant}:{dispatch.key.variant}",
        libraries.get(dispatch.key.quant),
    )
    batch_library = libraries.get("launch_batch")
    tail_library = libraries.get("moe_tail")
    if linear_library is None or batch_library is None or tail_library is None:
        return False
    projection_function = getattr(
        linear_library,
        batch_fn.projection_symbol,
    )
    tail_function = getattr(tail_library, batch_fn.tail_symbol)
    common = (
        projection_function,
        tail_function,
        x_ptr,
    )
    tail = (
        shared_out_ptr,
        routed_ptr,
        post_attention_ptr,
        norm_weight_ptr,
        norm_out_ptr,
        hidden_out_ptr,
        rows,
        in_features,
        out_features,
    )
    kwargs = {
        "eps": eps,
        "stream": stream,
        "library": batch_library,
        "runtime": runtime,
    }
    if dispatch.abi == "pack8":
        batch_fn(
            *common,
            weight.allocation("qweight").tensor.ptr,
            weight.allocation("scales").tensor.ptr,
            weight.allocation("mins").tensor.ptr,
            *tail,
            **kwargs,
        )
        return True
    if dispatch.abi == "raw":
        batch_fn(
            *common,
            weight.allocation("raw").tensor.ptr,
            *tail,
            **kwargs,
        )
        return True
    return False


def launch_gguf_linear_raw_ptr(
    weight: GGUFDeviceWeight,
    qweight_ptr: int,
    x_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    activation_dtype: str = GGUF_ACTIVATION_BF16,
    output_dtype: str = GGUF_OUTPUT_BF16,
    backend: str | None = None,
    threads: int = 0,
    stream: int = 0,
    libraries: Mapping[str, ctypes.CDLL] | None = None,
    runtime=None,
    use_wmma_prefill: bool | None = None,
) -> None:
    """Launch a raw GGUF linear using an already offset qweight pointer.

    Rank-3 MoE expert tensors are materialized as one contiguous raw GGUF
    allocation.  The caller selects an expert by offsetting into that allocation,
    while dispatch still resolves from the original logical weight spec.
    """

    resolved_backend = _weight_backend(weight, backend=backend)
    dispatch = resolve_gguf_linear_dispatch(
        weight,
        activation_dtype=activation_dtype,
        output_dtype=output_dtype,
        backend=resolved_backend,
        rows=rows,
    )
    if dispatch.abi != "raw":
        raise ValueError(f"raw-pointer GGUF launch requires raw layout, got {weight.spec.layout!r}")
    dispatch = _wmma_prefill_dispatch(
        dispatch,
        rows=rows,
        in_features=in_features,
        use_wmma=_resolve_use_wmma_prefill(use_wmma_prefill),
    )
    _ensure_linear_kernel_registered(dispatch.key)
    fn = resolve(
        backend=dispatch.key.backend,
        layer=dispatch.key.layer,
        quant=dispatch.key.quant,
        variant=dispatch.key.variant,
    )
    library = None if libraries is None else libraries.get(dispatch.key.quant)
    kwargs = {"stream": stream, "runtime": runtime}
    if threads and dispatch.abi != "wmma_raw":
        # The WMMA wrapper takes (tile_m, tile_n) instead of (threads); the
        # caller-supplied ``threads`` value applies to the decode-shaped path
        # only and is silently dropped on the WMMA path.
        kwargs["threads"] = threads
    if library is not None:
        kwargs["library"] = library
    fn(x_ptr, int(qweight_ptr), out_ptr, rows, in_features, out_features, **kwargs)


def launch_gguf_linear_pair(
    weight_a: GGUFDeviceWeight,
    weight_b: GGUFDeviceWeight,
    x_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    out_features_b: int | None = None,
    activation_dtype: str = GGUF_ACTIVATION_BF16,
    output_dtype: str = GGUF_OUTPUT_BF16,
    backend: str | None = None,
    stream: int = 0,
    libraries: Mapping[str, ctypes.CDLL] | None = None,
    runtime=None,
    use_wmma_prefill: bool | None = None,
    use_gemv_decode: bool | None = None,
    threads: int = 0,
    registered_decode_only: bool = False,
    registered_decode_variant: str | None = None,
) -> bool:
    """Launch a supported pair of GGUF projections, returning True when fused.

    The pair fast paths cover registered exact raw decode pairs (including
    unequal-width F32 output), architecture-qualified dense-F32 alpha/beta,
    narrow compact K/V and heterogeneous recurrent pairs, Q8_0 dual decode
    GEMV, Q4_K pack8 dual prefill, and raw-Q4/Q8T16 dual WMMA prefill.
    Populated resident-pack8 pairs decline
    the legacy dual owner when the exact tile8x8 singleton is registered.
    Q8T16 WMMA pairing is architecture/shape-qualified; every miss falls back
    to two singleton WMMA projections via :func:`launch_gguf_linear`.

    When ``use_gemv_decode`` is enabled (kwarg / session / env opt-in) and
    ``rows == 1`` with a registered Q8_0 dual gate+up GEMV decode kernel,
    the pair is fused through :func:`gguf_q8_0_pack8_dual_gate_up_gemv_decode_bf16_bf16_out`
    (P9.B3); the output layout matches the legacy ``gguf_q8_0_dual_gemv``
    concatenated layout that ``silu_mul_dual_out_*`` consumes downstream.
    ``registered_decode_only`` restricts resolution to the four-axis
    ``linear_pair`` key and otherwise returns ``False`` for explicit singleton
    fallback.
    """

    resolved_backend = _weight_backend(weight_a, weight_b, backend=backend)
    use_wmma = _resolve_use_wmma_prefill(use_wmma_prefill)
    use_gemv = _resolve_use_gemv_decode(use_gemv_decode)
    out_features_b = out_features if out_features_b is None else int(out_features_b)

    # The gfx1100 physical verifier qualifies the dense Q4T16 projection at
    # exactly rows6. A padded 12/18/24-row group therefore preserves that
    # owner by decomposing the normal unfused gate/up fallback into rows6
    # groups. Per-chunk pair misses intentionally fall through to two single
    # projections; the caller keeps the existing full-row SiLU stage.
    if (
        not registered_decode_only
        and activation_dtype == GGUF_ACTIVATION_BF16
        and output_dtype == GGUF_OUTPUT_BF16
        and int(out_features_b) == int(out_features)
        and weight_a.spec.quant_key == weight_b.spec.quant_key
        == "gguf_q4_k_t16_v1"
        and int(rows) >= 12
        and q4_t16_physical_extra_rowtiles_enabled()
    ):
        wide_exact_rows: set[int] = set()
        for capability in (
            "GGUF_SPECDEC2_EXACT_C7_TARGET_ROWS_POLICY",
            "GGUF_SPECDEC2_EXACT_C8_TARGET_ROWS_POLICY",
        ):
            wide_exact_policy = backend_package_capability(
                resolved_backend, capability, {}
            )
            wide_exact_rows.update(
                int(value) for value in wide_exact_policy.get("rows", ())
            )
        if physical_exact_rowtiles_enabled() and int(rows) in wide_exact_rows:
            for weight, output in (
                (weight_a, out_a_ptr),
                (weight_b, out_b_ptr),
            ):
                launch_gguf_linear(
                    weight,
                    x_ptr,
                    output,
                    rows,
                    in_features,
                    out_features,
                    backend=resolved_backend,
                    stream=stream,
                    libraries=libraries,
                    runtime=runtime,
                    use_wmma_prefill=use_wmma_prefill,
                    use_gemv_decode=use_gemv_decode,
                    threads=threads,
                )
            return True
        physical_pad_counts = backend_package_capability(
            resolved_backend,
            "GGUF_SPECDEC2_TARGET_VERIFY_PAD_ROW_COUNTS",
            (),
        )
        if physical_pad_counts:
            chunk = min(int(value) for value in physical_pad_counts)
            if chunk > 0 and int(rows) % chunk == 0:
                grouped_variant = (
                    _q4_t16_grouped_pair_rows6_variant(
                        resolved_backend,
                        rows=rows,
                        in_features=in_features,
                        out_features=out_features,
                    )
                    if chunk == 6
                    else None
                )
                if grouped_variant is not None:
                    grouped_key = KernelKey(
                        resolved_backend,
                        "linear",
                        "gguf_q4_k_t16_v1",
                        grouped_variant,
                    )
                    grouped_fn = resolve(
                        backend=grouped_key.backend,
                        layer=grouped_key.layer,
                        quant=grouped_key.quant,
                        variant=grouped_key.variant,
                    )
                    library = None
                    if libraries is not None:
                        library = libraries.get(
                            f"{grouped_key.quant}:{grouped_key.variant}",
                            libraries.get(grouped_key.quant),
                        )
                    grouped_kwargs = {"stream": stream, "runtime": runtime}
                    if library is not None:
                        grouped_kwargs["library"] = library
                    for weight, output in (
                        (weight_a, out_a_ptr),
                        (weight_b, out_b_ptr),
                    ):
                        _LAUNCH_ABI["t16"](
                            grouped_fn,
                            weight,
                            x_ptr,
                            output,
                            rows,
                            in_features,
                            out_features,
                            grouped_kwargs,
                        )
                    return True
                element = DType.BF16.itemsize
                for row_base in range(0, int(rows), chunk):
                    x_chunk = int(x_ptr) + row_base * int(in_features) * element
                    out_a_chunk = (
                        int(out_a_ptr) + row_base * int(out_features) * element
                    )
                    out_b_chunk = (
                        int(out_b_ptr) + row_base * int(out_features_b) * element
                    )
                    paired = launch_gguf_linear_pair(
                        weight_a,
                        weight_b,
                        x_chunk,
                        out_a_chunk,
                        out_b_chunk,
                        chunk,
                        in_features,
                        out_features,
                        out_features_b=out_features_b,
                        activation_dtype=activation_dtype,
                        output_dtype=output_dtype,
                        backend=resolved_backend,
                        stream=stream,
                        libraries=libraries,
                        runtime=runtime,
                        use_wmma_prefill=use_wmma_prefill,
                        use_gemv_decode=use_gemv_decode,
                        threads=threads,
                        registered_decode_variant=registered_decode_variant,
                    )
                    if paired:
                        continue
                    for weight, out_chunk, features in (
                        (weight_a, out_a_chunk, out_features),
                        (weight_b, out_b_chunk, out_features_b),
                    ):
                        launch_gguf_linear(
                            weight,
                            x_chunk,
                            out_chunk,
                            chunk,
                            in_features,
                            features,
                            activation_dtype=activation_dtype,
                            output_dtype=output_dtype,
                            backend=resolved_backend,
                            threads=threads,
                            stream=stream,
                            libraries=libraries,
                            runtime=runtime,
                            use_wmma_prefill=use_wmma_prefill,
                            use_gemv_decode=use_gemv_decode,
                        )
                return True

    cache_key = (
        generation(),
        weight_a.spec.layout,
        weight_a.spec.quant_key,
        weight_b.spec.layout,
        weight_b.spec.quant_key,
        rows,
        in_features,
        out_features,
        out_features_b,
        activation_dtype,
        output_dtype,
        resolved_backend,
        use_wmma,
        use_gemv,
        os.environ.get("HIPENGINE_GGUF_Q8_T16_DUAL_WMMA_PREFILL", "1"),
        _q8_t16_dual_wmma_prefill_enabled.get(),
        _q4_t16_unequal_pair_prefill_enabled.get(),
        (
            None
            if (source_f16_session := _t16_f16_rocblas_prefill_session.get())
            is None
            else id(source_f16_session)
        ),
        bool(_native_batch_decode_session_enabled),
        bool(registered_decode_only),
        registered_decode_variant,
    )
    pair_kind = _PAIR_DISPATCH_RESOLVE_CACHE.get(cache_key)
    if pair_kind is None:
        pair_kind = _resolve_gguf_linear_pair_kind(
            weight_a,
            weight_b,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            out_features_b=out_features_b,
            activation_dtype=activation_dtype,
            output_dtype=output_dtype,
            backend=resolved_backend,
            use_wmma=use_wmma,
            use_gemv=use_gemv,
            registered_decode_only=bool(registered_decode_only),
            registered_decode_variant=registered_decode_variant,
        )
        _PAIR_DISPATCH_RESOLVE_CACHE[cache_key] = pair_kind

    if pair_kind == "q6_q4_t16_mixed_grid":
        pair_key = KernelKey(
            resolved_backend,
            "linear_pair",
            "gguf_q6_k_t16_v1+gguf_q4_k_t16_v1",
            "mixed_grid_bf16_bf16_out",
        )
        pair_fn = resolve(
            backend=pair_key.backend,
            layer=pair_key.layer,
            quant=pair_key.quant,
            variant=pair_key.variant,
        )
        pair_kwargs = {"stream": stream, "runtime": runtime}
        pair_library = None if libraries is None else libraries.get(pair_key.quant)
        if pair_library is not None:
            pair_kwargs["library"] = pair_library
        pair_fn(
            x_ptr,
            weight_a.allocation("tiles").tensor.ptr,
            weight_b.allocation("tiles").tensor.ptr,
            out_a_ptr,
            out_b_ptr,
            rows,
            in_features,
            out_features,
            out_features_b,
            **pair_kwargs,
        )
        return True

    if pair_kind in {
        "q4_q4_t16_narrow_col4",
        "q4_q6_t16_narrow_col4_planar",
    }:
        if pair_kind == "q4_q4_t16_narrow_col4":
            pair_quant = "gguf_q4_k_t16_v1"
            pair_variant = "narrow_col4_pair_bf16_bf16_out"
        else:
            pair_quant = (
                "gguf_q4_k_t16_v1+gguf_q6_k_t16_qmicro_planar_v1"
            )
            pair_variant = "narrow_col4_planar_pair_bf16_bf16_out"
        pair_key = KernelKey(
            resolved_backend,
            "linear_pair",
            pair_quant,
            pair_variant,
        )
        pair_fn = resolve(
            backend=pair_key.backend,
            layer=pair_key.layer,
            quant=pair_key.quant,
            variant=pair_key.variant,
        )
        pair_kwargs = {"stream": stream, "runtime": runtime}
        pair_library = None if libraries is None else libraries.get(pair_key.quant)
        if pair_library is not None:
            pair_kwargs["library"] = pair_library
        pair_fn(
            x_ptr,
            weight_a.allocation("tiles").tensor.ptr,
            weight_b.allocation("tiles").tensor.ptr,
            out_a_ptr,
            out_b_ptr,
            rows,
            in_features,
            out_features,
            out_features_b,
            **pair_kwargs,
        )
        return True

    if pair_kind == "dense_f32_alpha_beta":
        pair_key = KernelKey(
            resolved_backend,
            "linear_pair",
            "f32",
            "bf16_hidden_bf16_out",
        )
        pair_fn = resolve(
            backend=pair_key.backend,
            layer=pair_key.layer,
            quant=pair_key.quant,
            variant=pair_key.variant,
        )
        pair_kwargs = {"stream": stream, "runtime": runtime}
        pair_library = None if libraries is None else libraries.get(pair_key.quant)
        if pair_library is not None:
            pair_kwargs["library"] = pair_library
        pair_fn(
            x_ptr,
            weight_a.allocation("raw").tensor.ptr,
            weight_b.allocation("raw").tensor.ptr,
            out_a_ptr,
            out_b_ptr,
            rows,
            in_features,
            out_features,
            **pair_kwargs,
        )
        return True

    if pair_kind == "q8_t16_dual_wmma":
        pair_key = KernelKey(
            resolved_backend,
            "linear_pair",
            "gguf_q8_0_t16_v1",
            "t16_dual_wmma_prefill_bf16_bf16_out",
        )
        pair_fn = resolve(
            backend=pair_key.backend,
            layer=pair_key.layer,
            quant=pair_key.quant,
            variant=pair_key.variant,
        )
        pair_kwargs = {"stream": stream, "runtime": runtime}
        pair_library = None if libraries is None else libraries.get(pair_key.quant)
        if pair_library is not None:
            pair_kwargs["library"] = pair_library
        pair_fn(
            x_ptr,
            weight_a.allocation("tiles").tensor.ptr,
            weight_b.allocation("tiles").tensor.ptr,
            out_a_ptr,
            out_b_ptr,
            rows,
            in_features,
            out_features,
            **pair_kwargs,
        )
        return True

    if pair_kind == "q4_t16_unequal_dual_wmma":
        pair_key = KernelKey(
            resolved_backend,
            "linear_pair",
            "gguf_q4_k_t16_v1",
            "dense_unequal_dual_wmma_prefill_bf16_bf16_out",
        )
        pair_fn = resolve(
            backend=pair_key.backend,
            layer=pair_key.layer,
            quant=pair_key.quant,
            variant=pair_key.variant,
        )
        pair_kwargs = {"stream": stream, "runtime": runtime}
        pair_library = None if libraries is None else libraries.get(pair_key.quant)
        if pair_library is not None:
            pair_kwargs["library"] = pair_library
        pair_fn(
            x_ptr,
            weight_a.allocation("tiles").tensor.ptr,
            weight_b.allocation("tiles").tensor.ptr,
            out_a_ptr,
            out_b_ptr,
            rows,
            in_features,
            out_features,
            out_features_b,
            **pair_kwargs,
        )
        return True

    if pair_kind in {
        "t16_f16_rocblas_shared_activation",
        "t16_f16_rocblas_pair_only_second",
    }:
        session = _t16_f16_rocblas_prefill_session.get()
        assert session is not None
        pair_only_spec = (
            session.pair_only_second_operand(
                rows,
                in_features,
                out_features,
                out_features_b,
                first_quant=weight_a.spec.quant_key,
                second_quant=weight_b.spec.quant_key,
            )
            if pair_kind == "t16_f16_rocblas_pair_only_second"
            else None
        )
        for index, (weight, out_ptr, outputs) in enumerate(
            (
                (weight_a, out_a_ptr, out_features),
                (weight_b, out_b_ptr, out_features_b),
            )
        ):
            dispatch = _pack8_decode_dispatch(
                resolve_gguf_linear_dispatch(
                    weight,
                    activation_dtype=activation_dtype,
                    output_dtype=output_dtype,
                    backend=resolved_backend,
                    rows=rows,
                ),
                rows=rows,
                out_features=outputs,
            )
            tile_override = None
            activation_inplace_override = None
            if index == 1 and pair_only_spec is not None:
                tile_override, variant, activation_inplace_override = pair_only_spec
                dispatch = GGUFLinearDispatch(
                    KernelKey(
                        dispatch.key.backend,
                        "linear",
                        dispatch.key.quant,
                        variant,
                    ),
                    _T16_F16_ROCBLAS_ROUTE_BY_QUANT[dispatch.key.quant][1],
                )
            else:
                dispatch = _q6_t16_f16_rocblas_prefill_dispatch(
                    dispatch,
                    rows=rows,
                    in_features=in_features,
                    out_features=outputs,
                )
            _ensure_linear_kernel_registered(dispatch.key)
            fn = resolve(
                backend=dispatch.key.backend,
                layer=dispatch.key.layer,
                quant=dispatch.key.quant,
                variant=dispatch.key.variant,
            )
            _launch_t16_f16_rocblas(
                dispatch.key.quant,
                fn,
                weight,
                x_ptr,
                out_ptr,
                rows,
                in_features,
                outputs,
                {"stream": stream, "runtime": runtime},
                cast_activation=index == 0,
                tile_override=tile_override,
                activation_inplace_override=activation_inplace_override,
            )
        return True

    if pair_kind == "q4_raw_dual_wmma":
        gguf_q4_k_wmma_prefill_dual_bf16_bf16_out(
            x_ptr,
            weight_a.allocation("raw").tensor.ptr,
            weight_b.allocation("raw").tensor.ptr,
            out_a_ptr,
            out_b_ptr,
            rows,
            in_features,
            out_features,
            stream=stream,
            runtime=runtime,
        )
        return True

    if pair_kind == "q8_t16_dual_split":
        if _use_q8_t16_pair_rowtile(
            rows=rows,
            in_features=in_features,
            out_features_a=out_features,
            out_features_b=out_features_b,
            threads=threads,
        ):
            pair_rowtile_fn = (
                gguf_q8_0_t16_dual_gemv_decode_rowtile4_col8_bf16_bf16_out
                if _resolve_use_q8_t16_pair_col8(rows=rows)
                else gguf_q8_0_t16_dual_gemv_decode_rowtile4_bf16_bf16_out
            )
            pair_rowtile_fn(
                x_ptr,
                weight_a.allocation("tiles").tensor.ptr,
                weight_b.allocation("tiles").tensor.ptr,
                out_a_ptr,
                out_b_ptr,
                rows,
                in_features,
                out_features,
                out_features_b,
                threads=_Q8_T16_ROWTILE_THREADS,
                stream=stream,
                runtime=runtime,
            )
            return True
        gguf_q8_0_t16_dual_gemv_decode_bf16_bf16_out(
            x_ptr,
            weight_a.allocation("tiles").tensor.ptr,
            weight_b.allocation("tiles").tensor.ptr,
            out_a_ptr,
            out_b_ptr,
            rows,
            in_features,
            out_features,
            out_features_b,
            threads=_resolve_q8_t16_threads(threads),
            stream=stream,
            runtime=runtime,
        )
        return True

    if pair_kind in {
        "registered_raw_decode_pair_equal",
        "registered_raw_decode_pair_unequal",
        "registered_raw_decode_pair_custom",
    }:
        pair_variant = (
            registered_decode_variant
            if pair_kind == "registered_raw_decode_pair_custom"
            else f"pack8_gemv_decode_{activation_dtype}_{output_dtype}_out"
        )
        pair_key = KernelKey(
            resolved_backend,
            "linear_pair",
            weight_a.spec.quant_key,
            pair_variant,
        )
        pair_fn = resolve(
            backend=pair_key.backend,
            layer=pair_key.layer,
            quant=pair_key.quant,
            variant=pair_key.variant,
        )
        pair_kwargs = {"stream": stream, "runtime": runtime}
        pair_library = None if libraries is None else libraries.get(pair_key.quant)
        if pair_library is not None:
            pair_kwargs["library"] = pair_library
        pair_args = (
            x_ptr,
            weight_a.allocation("raw").tensor.ptr,
            weight_b.allocation("raw").tensor.ptr,
            out_a_ptr,
            out_b_ptr,
            rows,
            in_features,
            out_features,
        )
        if pair_kind in {
            "registered_raw_decode_pair_unequal",
            "registered_raw_decode_pair_custom",
        }:
            pair_args = (*pair_args, out_features_b)
        pair_fn(*pair_args, **pair_kwargs)
        return True

    if pair_kind == "q8_raw_dual":
        gguf_q8_0_dual_gemv_bf16_bf16_out(
            x_ptr,
            weight_a.allocation("raw").tensor.ptr,
            weight_b.allocation("raw").tensor.ptr,
            out_a_ptr,
            out_b_ptr,
            rows,
            in_features,
            out_features,
            stream=stream,
            runtime=runtime,
        )
        return True

    if pair_kind == "q4_pack8_dual_prefill":
        if (
            _native_batch_decode_session_enabled
            and not use_wmma
            and _resolve_use_q4k_rowtile(None)
            and _PACK8_ROWTILE_MIN_ROWS <= rows <= _PACK8_ROWTILE_MAX_ROWS
        ):
            pair_library = None if libraries is None else libraries.get("gguf_q4_k")
            common_kwargs = {
                "stream": stream,
                "runtime": runtime,
                "library": pair_library,
            }
            gguf_q4_k_pack8_rowtile_bf16_bf16_out(
                x_ptr,
                weight_a.allocation("qweight").tensor.ptr,
                weight_a.allocation("scales").tensor.ptr,
                weight_a.allocation("mins").tensor.ptr,
                out_a_ptr,
                rows,
                in_features,
                out_features,
                **common_kwargs,
            )
            gguf_q4_k_pack8_rowtile_bf16_bf16_out(
                x_ptr,
                weight_b.allocation("qweight").tensor.ptr,
                weight_b.allocation("scales").tensor.ptr,
                weight_b.allocation("mins").tensor.ptr,
                out_b_ptr,
                rows,
                in_features,
                out_features,
                **common_kwargs,
            )
            return True
        # D08-X2-K5a: bulk rows prefer the routed pack8 WMMA leaf per side
        # over the per-row base dual kernel (same registry route the single
        # projection path takes; the base dual stays the fallback).
        wmma_key = KernelKey(
            resolved_backend,
            "linear",
            "gguf_q4_k",
            "pack8_wmma_prefill_bf16_bf16_out",
        )
        if (
            use_wmma
            and rows >= 16
            and os.environ.get("HIPENGINE_GGUF_Q4_PACK8_WMMA_BULK", "1") != "0"
            and backend_package_capability(
                resolved_backend,
                "GGUF_Q4_PACK8_WMMA_BULK_PREFILL",
                False,
            )
            and _backend_prefill_shape_is_qualified(
                resolved_backend,
                "GGUF_Q4_PACK8_WMMA_BULK_PREFILL_SHAPES",
                rows=rows,
                in_features=in_features,
                out_features=out_features,
            )
            and is_registered(wmma_key)
        ):
            wmma_fn = resolve(
                backend=wmma_key.backend,
                layer=wmma_key.layer,
                quant=wmma_key.quant,
                variant=wmma_key.variant,
            )
            pair_library = None if libraries is None else libraries.get(
                f"{wmma_key.quant}:{wmma_key.variant}",
                libraries.get(wmma_key.quant),
            )
            common_kwargs = {
                "stream": stream,
                "runtime": runtime,
                "library": pair_library,
            }
            for weight, out_ptr in ((weight_a, out_a_ptr), (weight_b, out_b_ptr)):
                wmma_fn(
                    x_ptr,
                    weight.allocation("qweight").tensor.ptr,
                    weight.allocation("scales").tensor.ptr,
                    weight.allocation("mins").tensor.ptr,
                    out_ptr,
                    rows,
                    in_features,
                    out_features,
                    **common_kwargs,
                )
            return True
        gguf_q4_k_pack8_dual_prefill_bf16_bf16_out(
            x_ptr,
            weight_a.allocation("qweight").tensor.ptr,
            weight_a.allocation("scales").tensor.ptr,
            weight_a.allocation("mins").tensor.ptr,
            weight_b.allocation("qweight").tensor.ptr,
            weight_b.allocation("scales").tensor.ptr,
            weight_b.allocation("mins").tensor.ptr,
            out_a_ptr,
            out_b_ptr,
            rows,
            in_features,
            out_features,
            stream=stream,
            runtime=runtime,
        )
        return True
    if pair_kind == "q4_pack8_dual_decode":
        pair_key = KernelKey(
            resolved_backend,
            "linear_pair",
            "gguf_q4_k",
            "pack8_dual_decode_bf16_bf16_out",
        )
        pair_fn = resolve(
            backend=pair_key.backend,
            layer=pair_key.layer,
            quant=pair_key.quant,
            variant=pair_key.variant,
        )
        pair_kwargs = {"stream": stream, "runtime": runtime}
        pair_library = None if libraries is None else libraries.get(pair_key.quant)
        if pair_library is not None:
            pair_kwargs["library"] = pair_library
        pair_fn(
            x_ptr,
            weight_a.allocation("qweight").tensor.ptr,
            weight_a.allocation("scales").tensor.ptr,
            weight_a.allocation("mins").tensor.ptr,
            weight_b.allocation("qweight").tensor.ptr,
            weight_b.allocation("scales").tensor.ptr,
            weight_b.allocation("mins").tensor.ptr,
            out_a_ptr,
            out_b_ptr,
            rows,
            in_features,
            out_features,
            **pair_kwargs,
        )
        return True
    return False


def _rowtile8_row_chunks(rows: int) -> list[tuple[int, int]]:
    """Split a decode batch into <=8-row rowtile8 groups, all groups >= 2.

    The qmicro Q8_1x2 rowtile8 owner instantiates ROW_TILE 2..8, so any
    concurrency c>8 is decomposed into contiguous groups of 2..8 rows. The
    tail-1 case is folded into the prior group (e.g. c=9 -> (7, 2)) so no
    single-row group is ever produced. Returns ``(group_rows, row_base)`` pairs.
    """
    rows = int(rows)
    if rows < 2:
        raise ValueError("rowtile8 chunking requires rows >= 2")
    chunks: list[tuple[int, int]] = []
    remaining = rows
    row_base = 0
    while remaining > 0:
        take = min(8, remaining)
        if remaining - take == 1:
            take -= 1
        chunks.append((take, row_base))
        row_base += take
        remaining -= take
    return chunks


def launch_gguf_linear_pair_silu(
    weight_a: GGUFDeviceWeight,
    weight_b: GGUFDeviceWeight,
    x_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    backend: str | None = None,
    stream: int = 0,
    libraries: Mapping[str, ctypes.CDLL] | None = None,
    runtime=None,
    use_gemv_decode: bool | None = None,
    use_q4_t16_sidecar: bool = True,
    use_q4_t16_dual_interleaved: bool = True,
    registered_decode_variant: str | None = None,
    layer_id: int | None = None,
    q8_1_workspace_ptr: int | None = None,
    pair_workspace_ptr: int | None = None,
    pair_workspace_nbytes: int = 0,
) -> bool:
    """Launch a registered gate/up pair plus SiLU, or return False."""

    resolved_backend = _weight_backend(weight_a, weight_b, backend=backend)
    dispatch_a = resolve_gguf_linear_dispatch(
        weight_a,
        backend=resolved_backend,
        rows=rows,
    )
    dispatch_b = resolve_gguf_linear_dispatch(
        weight_b,
        backend=resolved_backend,
        rows=rows,
    )
    if rows != 1:
        dense_pair_quant = (
            dispatch_a.key.quant
            if dispatch_a.key.quant == dispatch_b.key.quant
            and dispatch_a.key.quant in _Q4_T16_DENSE_QUANTS
            else None
        )
        production_q4_pair_key = _target_verifier_production_q4_pair_key(
            dispatch_a,
            dispatch_b,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        if production_q4_pair_key is not None:
            fn = resolve(
                backend=production_q4_pair_key.backend,
                layer=production_q4_pair_key.layer,
                quant=production_q4_pair_key.quant,
                variant=production_q4_pair_key.variant,
            )
            kwargs = {"stream": stream, "runtime": runtime}
            library = (
                None
                if libraries is None
                else libraries.get(production_q4_pair_key.quant)
            )
            if library is not None:
                kwargs["library"] = library
            fn(
                x_ptr,
                weight_a.allocation("tiles").tensor.ptr,
                weight_b.allocation("tiles").tensor.ptr,
                out_ptr,
                rows,
                in_features,
                out_features,
                **kwargs,
            )
            return True
        production_q4_chunk_groups = (
            _rowtile8_row_chunks(rows)
            if rows > _ROWTILE_MAX_ROWS
            and dense_pair_quant == "gguf_q4_k_t16_v1"
            and _target_verifier_production_q4_rowtile_scope_enabled(
                dispatch_a,
                rows=rows,
                in_features=in_features,
                out_features=out_features,
            )
            and _target_verifier_production_q4_rowtile_scope_enabled(
                dispatch_b,
                rows=rows,
                in_features=in_features,
                out_features=out_features,
            )
            else None
        )
        if production_q4_chunk_groups is not None:
            token = _target_verifier_rowtile_chunk_child_enabled.set(True)
            try:
                for chunk_rows, row_base in production_q4_chunk_groups:
                    launched = launch_gguf_linear_pair_silu(
                        weight_a,
                        weight_b,
                        int(x_ptr) + row_base * in_features * DType.BF16.itemsize,
                        int(out_ptr)
                        + row_base * out_features * DType.BF16.itemsize,
                        chunk_rows,
                        in_features,
                        out_features,
                        backend=resolved_backend,
                        stream=stream,
                        libraries=libraries,
                        runtime=runtime,
                        use_gemv_decode=use_gemv_decode,
                        use_q4_t16_sidecar=use_q4_t16_sidecar,
                        use_q4_t16_dual_interleaved=use_q4_t16_dual_interleaved,
                        registered_decode_variant=registered_decode_variant,
                        layer_id=layer_id,
                        q8_1_workspace_ptr=q8_1_workspace_ptr,
                        pair_workspace_ptr=pair_workspace_ptr,
                        pair_workspace_nbytes=pair_workspace_nbytes,
                    )
                    if not launched:
                        raise RuntimeError(
                            "admitted production Q4 verifier chunk lost its "
                            "rowtile owner"
                        )
            finally:
                _target_verifier_rowtile_chunk_child_enabled.reset(token)
            return True
        qmicro_q8x2_rowbatch = (
            _native_batch_decode_session_enabled
            and _resolve_use_gemv_decode(use_gemv_decode)
            and dense_pair_quant == "gguf_q4_k_qmicro_t16_v1"
            and registered_decode_variant
            == _Q4_T16_DENSE_PAIR_SILU_Q8_1X2_ROWTILE8_VARIANT
        )
        if qmicro_q8x2_rowbatch:
            fused_key = KernelKey(
                resolved_backend,
                "linear_pair_silu",
                dense_pair_quant,
                registered_decode_variant,
            )
            _ensure_linear_kernel_registered(fused_key)
            if not is_registered(fused_key):
                return False
            if q8_1_workspace_ptr is None or int(q8_1_workspace_ptr) <= 0:
                return False
            fn = resolve(
                backend=fused_key.backend,
                layer=fused_key.layer,
                quant=fused_key.quant,
                variant=fused_key.variant,
            )
            kwargs = {"stream": stream, "runtime": runtime}
            library = None if libraries is None else libraries.get(fused_key.quant)
            if library is not None:
                kwargs["library"] = library
            tiles_a_ptr = weight_a.allocation("tiles").tensor.ptr
            tiles_b_ptr = weight_b.allocation("tiles").tensor.ptr
            q4_library = None if libraries is None else libraries.get("gguf_q4_k")
            # The rowtile8 owner instantiates ROW_TILE 2..8; any c>8 chunks
            # into <=8-row groups (all groups >= 2) so every decode concurrency
            # shares one weight traversal per group. Groups are row-independent
            # (per-row Q8_1x2 quantization and per-row SiLU), so chunking is
            # exact: each row keeps the same dp4a/FMA/BF16 association as c1.
            for chunk_rows, row_base in _rowtile8_row_chunks(rows):
                x_chunk = int(x_ptr) + row_base * in_features * 2
                gguf_q4_k_quantize_bf16_q8_1x2(
                    x_chunk,
                    int(q8_1_workspace_ptr),
                    chunk_rows,
                    in_features,
                    stream=stream,
                    library=q4_library,
                    runtime=runtime,
                )
                fn(
                    int(q8_1_workspace_ptr),
                    tiles_a_ptr,
                    tiles_b_ptr,
                    int(out_ptr) + row_base * out_features * 2,
                    chunk_rows,
                    in_features,
                    out_features,
                    **kwargs,
                )
            return True
        pack8_wmma_key = _pack8_dual_wmma_silu_dispatch(
            dispatch_a,
            dispatch_b,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            use_wmma=_resolve_use_wmma_prefill(None),
        )
        if pack8_wmma_key is not None:
            fn = resolve(
                backend=pack8_wmma_key.backend,
                layer=pack8_wmma_key.layer,
                quant=pack8_wmma_key.quant,
                variant=pack8_wmma_key.variant,
            )
            kwargs = {"stream": stream, "runtime": runtime}
            library = (
                None if libraries is None else libraries.get(pack8_wmma_key.quant)
            )
            if library is not None:
                kwargs["library"] = library
            fn(
                x_ptr,
                weight_a.allocation("qweight").tensor.ptr,
                weight_a.allocation("scales").tensor.ptr,
                weight_a.allocation("mins").tensor.ptr,
                weight_b.allocation("qweight").tensor.ptr,
                weight_b.allocation("scales").tensor.ptr,
                weight_b.allocation("mins").tensor.ptr,
                out_ptr,
                rows,
                in_features,
                out_features,
                **kwargs,
            )
            return True
        qmicro_metadata_nbytes = (
            (out_features // 16) * (in_features // 256) * 256
        )
        use_expanded_qmicro_metadata = (
            rows >= _Q4_QMICRO_T16_EXPANDED_META_MIN_ROWS
            and dispatch_a.key.quant == dispatch_b.key.quant
            == "gguf_q4_k_qmicro_t16_v1"
            and pair_workspace_ptr is not None
            and int(pair_workspace_ptr) > 0
            and int(pair_workspace_nbytes) >= 2 * qmicro_metadata_nbytes
        )
        t16_wmma_key = _q4_t16_dual_wmma_silu_dispatch(
            dispatch_a,
            dispatch_b,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            layer_id=layer_id,
            expanded_metadata=use_expanded_qmicro_metadata,
        )
        if t16_wmma_key is not None:
            fn = resolve(
                backend=t16_wmma_key.backend,
                layer=t16_wmma_key.layer,
                quant=t16_wmma_key.quant,
                variant=t16_wmma_key.variant,
            )
            kwargs = {"stream": stream, "runtime": runtime}
            library = (
                None if libraries is None else libraries.get(t16_wmma_key.quant)
            )
            if library is not None:
                kwargs["library"] = library
            common = (
                x_ptr,
                weight_a.allocation("tiles").tensor.ptr,
                weight_b.allocation("tiles").tensor.ptr,
            )
            if use_expanded_qmicro_metadata:
                assert pair_workspace_ptr is not None
                fn(
                    *common,
                    int(pair_workspace_ptr),
                    int(pair_workspace_ptr) + qmicro_metadata_nbytes,
                    out_ptr,
                    rows,
                    in_features,
                    out_features,
                    **kwargs,
                )
            else:
                fn(
                    *common,
                    out_ptr,
                    rows,
                    in_features,
                    out_features,
                    **kwargs,
                )
            return True
        t16_rowtile_key = _q4_t16_dual_rowtile_silu_dispatch(
            dispatch_a,
            dispatch_b,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            use_sidecar=use_q4_t16_sidecar,
            native_batch=(
                _native_batch_decode_session_enabled
                or _target_verifier_production_q4_rowtile_scope_enabled(
                    dispatch_a,
                    rows=rows,
                    in_features=in_features,
                    out_features=out_features,
                )
            ),
        )
        decode_tiles_a = None
        decode_tiles_b = None
        allocation_name = (
            "tiles"
            if dispatch_a.key.quant == dispatch_b.key.quant
            and dispatch_a.key.quant in _Q4_T16_DENSE_QUANTS
            else "decode_tiles"
        )
        try:
            decode_tiles_a = weight_a.allocation(allocation_name)
            decode_tiles_b = weight_b.allocation(allocation_name)
        except KeyError:
            pass
        if (
            t16_rowtile_key is not None
            and decode_tiles_a is not None
            and decode_tiles_b is not None
        ):
            fn = resolve(
                backend=t16_rowtile_key.backend,
                layer=t16_rowtile_key.layer,
                quant=t16_rowtile_key.quant,
                variant=t16_rowtile_key.variant,
            )
            kwargs = {"stream": stream, "runtime": runtime}
            library = (
                None
                if libraries is None
                else libraries.get(t16_rowtile_key.quant)
            )
            if library is not None:
                kwargs["library"] = library
            fn(
                x_ptr,
                decode_tiles_a.tensor.ptr,
                decode_tiles_b.tensor.ptr,
                out_ptr,
                rows,
                in_features,
                out_features,
                **kwargs,
            )
            return True
        fused_rowtile_key = _pack8_dual_rowtile_silu_dispatch(
            dispatch_a,
            dispatch_b,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            use_rowtile=_resolve_use_q4k_rowtile(None),
            native_batch=_native_batch_decode_session_enabled,
        )
        if fused_rowtile_key is None:
            return False
        fn = resolve(
            backend=fused_rowtile_key.backend,
            layer=fused_rowtile_key.layer,
            quant=fused_rowtile_key.quant,
            variant=fused_rowtile_key.variant,
        )
        kwargs = {"stream": stream, "runtime": runtime}
        library = None if libraries is None else libraries.get(fused_rowtile_key.quant)
        if library is not None:
            kwargs["library"] = library
        fn(
            x_ptr,
            weight_a.allocation("qweight").tensor.ptr,
            weight_a.allocation("scales").tensor.ptr,
            weight_a.allocation("mins").tensor.ptr,
            weight_b.allocation("qweight").tensor.ptr,
            weight_b.allocation("scales").tensor.ptr,
            weight_b.allocation("mins").tensor.ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            **kwargs,
        )
        return True
    if not _resolve_use_gemv_decode(use_gemv_decode):
        return False
    dense_pair_quant = (
        dispatch_a.key.quant
        if dispatch_a.key.quant == dispatch_b.key.quant
        and dispatch_a.key.quant in _Q4_T16_DENSE_QUANTS
        else None
    )
    q4_t16_decode = (
        None
        if dense_pair_quant is None
        else KernelKey(
            resolved_backend,
            "linear",
            dense_pair_quant,
            "dense_single_local32_bf16_bf16_out",
        )
    )
    q4_t16_pair_variant = (
        registered_decode_variant or "dense_dual_local32_bf16_bf16_out"
    )
    if (
        _native_batch_decode_session_enabled
        and dense_pair_quant == "gguf_q4_k_t16_v1"
        and q4_t16_pair_variant
        == _Q4_T16_DENSE_PAIR_SILU_SPLIT_WEIGHT_VARIANT
    ):
        q4_t16_pair_variant = _Q4_T16_DENSE_PAIR_SILU_Q8_1X2_VARIANT
    q4_t16_pair_silu = (
        None
        if dense_pair_quant is None
        else KernelKey(
            resolved_backend,
            "linear_pair_silu",
            dense_pair_quant,
            q4_t16_pair_variant,
        )
    )
    if q4_t16_pair_silu is not None:
        _ensure_linear_kernel_registered(q4_t16_pair_silu)
    if (
        q4_t16_decode is not None
        and q4_t16_pair_silu is not None
        and dispatch_a.key == q4_t16_decode
        and dispatch_b.key == q4_t16_decode
        and is_registered(q4_t16_pair_silu)
    ):
        fn = resolve(
            backend=q4_t16_pair_silu.backend,
            layer=q4_t16_pair_silu.layer,
            quant=q4_t16_pair_silu.quant,
            variant=q4_t16_pair_silu.variant,
        )
        kwargs = {"stream": stream, "runtime": runtime}
        library = (
            None
            if libraries is None
            else libraries.get(q4_t16_pair_silu.quant)
        )
        if library is not None:
            kwargs["library"] = library
        launch_x_ptr = int(x_ptr)
        if q4_t16_pair_silu.variant in _Q4_T16_DENSE_PAIR_SILU_Q8_1X2_VARIANTS:
            if q8_1_workspace_ptr is None or int(q8_1_workspace_ptr) <= 0:
                return False
            q4_library = (
                None if libraries is None else libraries.get("gguf_q4_k")
            )
            gguf_q4_k_quantize_bf16_q8_1x2(
                x_ptr,
                int(q8_1_workspace_ptr),
                rows,
                in_features,
                stream=stream,
                library=q4_library,
                runtime=runtime,
            )
            launch_x_ptr = int(q8_1_workspace_ptr)
        fn(
            launch_x_ptr,
            weight_a.allocation("tiles").tensor.ptr,
            weight_b.allocation("tiles").tensor.ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            **kwargs,
        )
        return True
    q4_decode = KernelKey(
        resolved_backend,
        "linear",
        "gguf_q4_k",
        "pack8_bf16_bf16_out",
    )
    fused_key = KernelKey(
        resolved_backend,
        "linear_pair_silu",
        "gguf_q4_k",
        registered_decode_variant or "pack8_dual_decode_bf16_bf16_out",
    )
    t16_sidecar_key = KernelKey(
        resolved_backend,
        "linear_pair_silu",
        "gguf_q4_k",
        "t16_sidecar_dual_decode_bf16_bf16_out",
    )
    t16_dual_interleaved_key = KernelKey(
        resolved_backend,
        "linear_pair_silu",
        "gguf_q4_k",
        "t16_dual_interleaved_sidecar_decode_bf16_bf16_out",
    )
    _ensure_linear_kernel_registered(fused_key)
    decode_tiles_dual = None
    try:
        decode_tiles_dual = weight_a.allocation("decode_tiles_dual")
    except KeyError:
        pass
    if (
        dispatch_a.key == q4_decode
        and dispatch_b.key == q4_decode
        and use_q4_t16_sidecar
        and use_q4_t16_dual_interleaved
        and decode_tiles_dual is not None
        and is_registered(t16_dual_interleaved_key)
    ):
        fn = resolve(
            backend=t16_dual_interleaved_key.backend,
            layer=t16_dual_interleaved_key.layer,
            quant=t16_dual_interleaved_key.quant,
            variant=t16_dual_interleaved_key.variant,
        )
        kwargs = {"stream": stream, "runtime": runtime}
        library = (
            None
            if libraries is None
            else libraries.get("gguf_q4_k_t16_v1")
        )
        if library is not None:
            kwargs["library"] = library
        fn(
            x_ptr,
            decode_tiles_dual.tensor.ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            **kwargs,
        )
        return True
    decode_tiles_a = None
    decode_tiles_b = None
    try:
        decode_tiles_a = weight_a.allocation("decode_tiles")
        decode_tiles_b = weight_b.allocation("decode_tiles")
    except KeyError:
        pass
    if (
        dispatch_a.key == q4_decode
        and dispatch_b.key == q4_decode
        and use_q4_t16_sidecar
        and decode_tiles_a is not None
        and decode_tiles_b is not None
        and is_registered(t16_sidecar_key)
    ):
        fn = resolve(
            backend=t16_sidecar_key.backend,
            layer=t16_sidecar_key.layer,
            quant=t16_sidecar_key.quant,
            variant=t16_sidecar_key.variant,
        )
        kwargs = {"stream": stream, "runtime": runtime}
        library = (
            None
            if libraries is None
            else libraries.get("gguf_q4_k_t16_v1")
        )
        if library is not None:
            kwargs["library"] = library
        fn(
            x_ptr,
            decode_tiles_a.tensor.ptr,
            decode_tiles_b.tensor.ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            **kwargs,
        )
        return True
    if (
        dispatch_a.key != q4_decode
        or dispatch_b.key != q4_decode
        or not is_registered(fused_key)
    ):
        return False
    fn = resolve(
        backend=fused_key.backend,
        layer=fused_key.layer,
        quant=fused_key.quant,
        variant=fused_key.variant,
    )
    kwargs = {"stream": stream, "runtime": runtime}
    library = None if libraries is None else libraries.get(fused_key.quant)
    if library is not None:
        kwargs["library"] = library
    fn(
        x_ptr,
        weight_a.allocation("qweight").tensor.ptr,
        weight_a.allocation("scales").tensor.ptr,
        weight_a.allocation("mins").tensor.ptr,
        weight_b.allocation("qweight").tensor.ptr,
        weight_b.allocation("scales").tensor.ptr,
        weight_b.allocation("mins").tensor.ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )
    return True


def _resolve_gguf_linear_pair_kind(
    weight_a: GGUFDeviceWeight,
    weight_b: GGUFDeviceWeight,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    out_features_b: int,
    activation_dtype: str,
    output_dtype: str,
    backend: str,
    use_wmma: bool,
    use_gemv: bool,
    registered_decode_only: bool,
    registered_decode_variant: str | None = None,
) -> str:
    dispatch_a = _pack8_decode_dispatch(
        resolve_gguf_linear_dispatch(
            weight_a,
            activation_dtype=activation_dtype,
            output_dtype=output_dtype,
            backend=backend,
            rows=rows,
        ),
        rows=rows,
        out_features=out_features,
    )
    dispatch_b = _pack8_decode_dispatch(
        resolve_gguf_linear_dispatch(
            weight_b,
            activation_dtype=activation_dtype,
            output_dtype=output_dtype,
            backend=backend,
            rows=rows,
        ),
        rows=rows,
        out_features=out_features_b,
    )
    if _t16_f16_rocblas_prefill_session.get() is not None:
        rewritten = tuple(
            _q6_t16_f16_rocblas_prefill_dispatch(
                dispatch,
                rows=rows,
                in_features=in_features,
                out_features=outputs,
            )
            for dispatch, outputs in (
                (dispatch_a, out_features),
                (dispatch_b, out_features_b),
            )
        )
        rewritten_count = sum(
            candidate is not original
            for candidate, original in zip(rewritten, (dispatch_a, dispatch_b))
        )
        if rewritten_count == 2:
            session = _t16_f16_rocblas_prefill_session.get()
            assert session is not None
            inplace = tuple(
                session.activation_is_inplace(
                    rows,
                    in_features,
                    outputs,
                    quant=dispatch.key.quant,
                )
                for dispatch, outputs in (
                    (dispatch_a, out_features),
                    (dispatch_b, out_features_b),
                )
            )
            if inplace[0] == inplace[1]:
                return "t16_f16_rocblas_shared_activation"
            return "none"
        if (
            rewritten_count == 1
            and rewritten[0] is not dispatch_a
            and rewritten[1] is dispatch_b
        ):
            session = _t16_f16_rocblas_prefill_session.get()
            assert session is not None
            pair_only_spec = session.pair_only_second_operand(
                rows,
                in_features,
                out_features,
                out_features_b,
                first_quant=dispatch_a.key.quant,
                second_quant=dispatch_b.key.quant,
            )
            if pair_only_spec is not None:
                _tile, variant, second_inplace = pair_only_spec
                first_inplace = session.activation_is_inplace(
                    rows,
                    in_features,
                    out_features,
                    quant=dispatch_a.key.quant,
                )
                pair_key = KernelKey(
                    dispatch_b.key.backend,
                    "linear",
                    dispatch_b.key.quant,
                    variant,
                )
                if first_inplace == second_inplace and is_registered(pair_key):
                    return "t16_f16_rocblas_pair_only_second"
        if rewritten_count:
            # A mixed candidate/exact pair has no ordinary shared-activation
            # ABI. Decline unless an explicit ordered pair-only policy matched.
            return "none"
    q6_q4_pair_key = KernelKey(
        backend,
        "linear_pair",
        "gguf_q6_k_t16_v1+gguf_q4_k_t16_v1",
        "mixed_grid_bf16_bf16_out",
    )
    q6_q4_shapes = backend_package_capability(
        backend,
        "GGUF_Q6_Q4_T16_MIXED_GRID_DECODE_SHAPES",
        (),
    )
    if (
        not _native_batch_decode_session_enabled
        and (rows, in_features, out_features, out_features_b) in q6_q4_shapes
        and activation_dtype == GGUF_ACTIVATION_BF16
        and output_dtype == GGUF_OUTPUT_BF16
        and dispatch_a.abi == dispatch_b.abi == "t16"
        and dispatch_a.key.quant == "gguf_q6_k_t16_v1"
        and dispatch_a.key.variant == "t16_gemv_decode_bf16_bf16_out"
        and dispatch_b.key.quant == "gguf_q4_k_t16_v1"
        and dispatch_b.key.variant == "dense_single_local32_bf16_bf16_out"
        and is_registered(q6_q4_pair_key)
    ):
        return "q6_q4_t16_mixed_grid"

    narrow_shapes = backend_package_capability(
        backend,
        "GGUF_NARROW_KV_PAIR_DECODE_SHAPES",
        (),
    )
    if (
        not _native_batch_decode_session_enabled
        and (rows, in_features, out_features, out_features_b) in narrow_shapes
        and activation_dtype == GGUF_ACTIVATION_BF16
        and output_dtype == GGUF_OUTPUT_BF16
        and dispatch_a.abi == dispatch_b.abi == "t16"
    ):
        narrow_a = _t16_c1_variant_dispatch(
            dispatch_a,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
        narrow_b = _t16_c1_variant_dispatch(
            dispatch_b,
            rows=rows,
            in_features=in_features,
            out_features=out_features_b,
        )
        q4_key = KernelKey(
            backend,
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_single_col4_bf16_bf16_out",
        )
        q6_key = KernelKey(
            backend,
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_decode_bf16_bf16_out",
        )
        q4_q4_pair_key = KernelKey(
            backend,
            "linear_pair",
            "gguf_q4_k_t16_v1",
            "narrow_col4_pair_bf16_bf16_out",
        )
        q4_q6_pair_key = KernelKey(
            backend,
            "linear_pair",
            "gguf_q4_k_t16_v1+gguf_q6_k_t16_qmicro_planar_v1",
            "narrow_col4_planar_pair_bf16_bf16_out",
        )
        if (
            narrow_a.key == q4_key
            and narrow_b.key == q4_key
            and is_registered(q4_q4_pair_key)
        ):
            return "q4_q4_t16_narrow_col4"
        if (
            narrow_a.key == q4_key
            and narrow_b.key == q6_key
            and is_registered(q4_q6_pair_key)
        ):
            return "q4_q6_t16_narrow_col4_planar"

    dense_f32_pair_key = KernelKey(
        backend,
        "linear_pair",
        "f32",
        "bf16_hidden_bf16_out",
    )
    dense_f32_pair_shapes = backend_package_capability(
        backend,
        "GGUF_DENSE_F32_ALPHA_BETA_PAIR_DECODE_SHAPES",
        (),
    )
    dense_f32_single_key = KernelKey(
        backend,
        "dense_gemv",
        "f32",
        "bf16_hidden_bf16_out",
    )
    if (
        (rows, in_features, out_features, out_features_b)
        in dense_f32_pair_shapes
        and activation_dtype == GGUF_ACTIVATION_BF16
        and output_dtype == GGUF_OUTPUT_BF16
        and dispatch_a.abi == dispatch_b.abi == "dense_bf16"
        and dispatch_a.key == dispatch_b.key == dense_f32_single_key
        and is_registered(dense_f32_pair_key)
    ):
        return "dense_f32_alpha_beta"

    if use_wmma and rows > 1:
        q8_t16_dual_wmma = KernelKey(
            backend,
            "linear_pair",
            "gguf_q8_0_t16_v1",
            "t16_dual_wmma_prefill_bf16_bf16_out",
        )
        q8_t16_shapes = backend_package_capability(
            backend,
            "GGUF_Q8_T16_DUAL_WMMA_PREFILL_SHAPES",
            (),
        )
        if (
            os.environ.get("HIPENGINE_GGUF_Q8_T16_DUAL_WMMA_PREFILL", "1")
            != "0"
            and _q8_t16_dual_wmma_prefill_enabled.get()
            and bool(
                backend_package_capability(
                    backend,
                    "GGUF_Q8_T16_DUAL_WMMA_PREFILL",
                    False,
                )
            )
            and (rows, in_features, out_features, out_features_b)
            in q8_t16_shapes
            and activation_dtype == GGUF_ACTIVATION_BF16
            and output_dtype == GGUF_OUTPUT_BF16
            and dispatch_a.abi == dispatch_b.abi == "t16"
            and dispatch_a.key.quant == dispatch_b.key.quant == "gguf_q8_0_t16_v1"
            and is_registered(q8_t16_dual_wmma)
        ):
            return "q8_t16_dual_wmma"

        q4_t16_unequal_dual = KernelKey(
            backend,
            "linear_pair",
            "gguf_q4_k_t16_v1",
            "dense_unequal_dual_wmma_prefill_bf16_bf16_out",
        )
        if (
            _q4_t16_unequal_pair_prefill_enabled.get()
            and rows >= _Q4_T16_UNEQUAL_DUAL_WMMA_MIN_ROWS
            and (in_features, out_features, out_features_b)
            == _Q4_T16_UNEQUAL_DUAL_WMMA_SHAPE
            and dispatch_a.abi == "t16"
            and dispatch_b.abi == "t16"
            and dispatch_a.key.quant == "gguf_q4_k_t16_v1"
            and dispatch_b.key.quant == "gguf_q4_k_t16_v1"
        ):
            _ensure_linear_kernel_registered(q4_t16_unequal_dual)
            if is_registered(q4_t16_unequal_dual):
                return "q4_t16_unequal_dual_wmma"

        # A populated resident-pack8 pair has no exact fused tile owner. Decline
        # only when the registered singleton rewrite is available; callers then
        # issue two tile8x8 leaves. Missing keys retain the legacy dual owner.
        if any(
            _pack8_exact_prefill_dispatch(d, rows=rows, enabled=True) is not d
            for d in (dispatch_a, dispatch_b)
        ):
            return "none"

        q4_prefill_raw = KernelKey(
            backend, "linear", "gguf_q4_k", "prefill_bf16_bf16_out"
        )
        if (
            out_features_b == out_features
            and dispatch_a.abi == "raw"
            and dispatch_b.abi == "raw"
            and dispatch_a.key == q4_prefill_raw
            and dispatch_b.key == q4_prefill_raw
            and _wmma_prefill_shape_supported("gguf_q4_k", in_features)
        ):
            return "q4_raw_dual_wmma"

        # If either side would be routed to a WMMA prefill singleton that does
        # not have a dual pair path here (currently Q8_0), decline the pair
        # fusion so the caller falls back to two singletons (each picks up the
        # WMMA family via launch_gguf_linear).
        for d in (dispatch_a, dispatch_b):
            if _dispatch_can_use_wmma_prefill(d, rows=rows, in_features=in_features):
                return "none"
    q8_t16_dual = KernelKey(
        backend,
        "linear",
        "gguf_q8_0_t16_v1",
        "t16_dual_gemv_decode_bf16_bf16_out",
    )
    if (
        activation_dtype == GGUF_ACTIVATION_BF16
        and output_dtype == GGUF_OUTPUT_BF16
        and dispatch_a.abi == "t16"
        and dispatch_b.abi == "t16"
        and dispatch_a.key.quant == "gguf_q8_0_t16_v1"
        and dispatch_b.key.quant == "gguf_q8_0_t16_v1"
        and is_registered(q8_t16_dual)
    ):
        # P10.B4: decline the Q8T16 dual GEMV fusion at rows>1 when WMMA
        # prefill is opted in, so the caller falls back to two singletons
        # that each take the dense Q8T16 WMMA prefill path.
        if use_wmma and rows > 1 and (
            _dispatch_can_use_t16_wmma_prefill(dispatch_a, rows=rows, in_features=in_features)
            or _dispatch_can_use_t16_wmma_prefill(dispatch_b, rows=rows, in_features=in_features)
        ):
            return "none"
        return "q8_t16_dual_split"

    custom_pair_key = KernelKey(
        backend,
        "linear_pair",
        dispatch_a.key.quant,
        registered_decode_variant or "",
    )
    if (
        registered_decode_variant is not None
        and output_dtype in {GGUF_OUTPUT_BF16, GGUF_OUTPUT_F32}
        and use_gemv
        and rows == 1
        and dispatch_a.abi == "raw"
        and dispatch_b.abi == "raw"
        and dispatch_a.key == dispatch_b.key
        and is_registered(custom_pair_key)
    ):
        return "registered_raw_decode_pair_custom"

    registered_pair_key = KernelKey(
        backend,
        "linear_pair",
        dispatch_a.key.quant,
        f"pack8_gemv_decode_{activation_dtype}_{output_dtype}_out",
    )
    if (
        use_gemv
        and rows == 1
        and dispatch_a.abi == "raw"
        and dispatch_b.abi == "raw"
        and dispatch_a.key == dispatch_b.key
        and is_registered(registered_pair_key)
    ):
        if output_dtype == GGUF_OUTPUT_F32:
            return "registered_raw_decode_pair_unequal"
        if out_features_b == out_features:
            return "registered_raw_decode_pair_equal"

    q4_decode = KernelKey(backend, "linear", "gguf_q4_k", "pack8_bf16_bf16_out")
    q4_decode_pair = KernelKey(
        backend,
        "linear_pair",
        "gguf_q4_k",
        "pack8_dual_decode_bf16_bf16_out",
    )
    if (
        rows == 1
        and out_features_b == out_features
        and dispatch_a.key == q4_decode
        and dispatch_b.key == q4_decode
        and is_registered(q4_decode_pair)
    ):
        return "q4_pack8_dual_decode"

    if registered_decode_only:
        return "none"

    q8_decode = KernelKey(backend, "linear", "gguf_q8_0", "pack8_gemv_bf16_bf16_out")
    if rows == 1 and out_features_b == out_features and dispatch_a.key == q8_decode and dispatch_b.key == q8_decode:
        return "q8_raw_dual"

    q4_prefill = KernelKey(backend, "linear", "gguf_q4_k", "pack8_prefill_bf16_bf16_out")
    if rows > 1 and out_features_b == out_features and dispatch_a.key == q4_prefill and dispatch_b.key == q4_prefill:
        return "q4_pack8_dual_prefill"
    return "none"


def launch_gguf_linear_triple(
    weight_a: GGUFDeviceWeight,
    weight_b: GGUFDeviceWeight,
    weight_c: GGUFDeviceWeight,
    x_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    out_c_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    out_features_b: int | None = None,
    out_features_c: int | None = None,
    backend: str | None = None,
    stream: int = 0,
    runtime=None,
    threads: int = 0,
) -> bool:
    """Launch a supported same-input triple of GGUF projections."""

    resolved_backend = _weight_backend(weight_a, weight_b, weight_c, backend=backend)
    out_features_b = out_features if out_features_b is None else int(out_features_b)
    out_features_c = out_features if out_features_c is None else int(out_features_c)
    dispatch_a = _pack8_decode_dispatch(
        resolve_gguf_linear_dispatch(weight_a, backend=resolved_backend, rows=rows),
        rows=rows,
        out_features=out_features,
    )
    dispatch_b = _pack8_decode_dispatch(
        resolve_gguf_linear_dispatch(weight_b, backend=resolved_backend, rows=rows),
        rows=rows,
        out_features=out_features_b,
    )
    dispatch_c = _pack8_decode_dispatch(
        resolve_gguf_linear_dispatch(weight_c, backend=resolved_backend, rows=rows),
        rows=rows,
        out_features=out_features_c,
    )
    use_wmma = _resolve_use_wmma_prefill(None)
    q8_t16_triple = KernelKey(
        resolved_backend,
        "linear",
        "gguf_q8_0_t16_v1",
        "t16_triple_gemv_decode_bf16_bf16_out",
    )
    if use_wmma and rows > 1 and (
        _dispatch_can_use_t16_wmma_prefill(dispatch_a, rows=rows, in_features=in_features)
        or _dispatch_can_use_t16_wmma_prefill(dispatch_b, rows=rows, in_features=in_features)
        or _dispatch_can_use_t16_wmma_prefill(dispatch_c, rows=rows, in_features=in_features)
    ):
        # P10.B4: decline Q8T16 triple GEMV fusion at rows>1 when WMMA
        # prefill is opted in, so the caller falls back to singletons that
        # each take the dense Q8T16 WMMA prefill path.
        return False
    if (
        dispatch_a.abi == "t16"
        and dispatch_b.abi == "t16"
        and dispatch_c.abi == "t16"
        and dispatch_a.key.quant == "gguf_q8_0_t16_v1"
        and dispatch_b.key.quant == "gguf_q8_0_t16_v1"
        and dispatch_c.key.quant == "gguf_q8_0_t16_v1"
        and is_registered(q8_t16_triple)
    ):
        if _use_q8_t16_all_rowtile(
            rows=rows,
            in_features=in_features,
            threads=threads,
        ):
            gguf_q8_0_t16_triple_gemv_decode_rowtile4_bf16_bf16_out(
                x_ptr,
                weight_a.allocation("tiles").tensor.ptr,
                weight_b.allocation("tiles").tensor.ptr,
                weight_c.allocation("tiles").tensor.ptr,
                out_a_ptr,
                out_b_ptr,
                out_c_ptr,
                rows,
                in_features,
                out_features,
                out_features_b,
                out_features_c,
                threads=_Q8_T16_ROWTILE_THREADS,
                stream=stream,
                runtime=runtime,
            )
            return True
        gguf_q8_0_t16_triple_gemv_decode_bf16_bf16_out(
            x_ptr,
            weight_a.allocation("tiles").tensor.ptr,
            weight_b.allocation("tiles").tensor.ptr,
            weight_c.allocation("tiles").tensor.ptr,
            out_a_ptr,
            out_b_ptr,
            out_c_ptr,
            rows,
            in_features,
            out_features,
            out_features_b,
            out_features_c,
            threads=_resolve_q8_t16_threads(threads),
            stream=stream,
            runtime=runtime,
        )
        return True
    return False


def launch_gguf_linear_pair_concat(
    weight_a: GGUFDeviceWeight,
    weight_b: GGUFDeviceWeight,
    x_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    backend: str | None = None,
    stream: int = 0,
    runtime=None,
    use_wmma_prefill: bool | None = None,
    use_gemv_decode: bool | None = None,
    threads: int = 0,
) -> bool:
    """Launch a supported projection pair into one concatenated output buffer.

    This is the prefill-side companion to :func:`launch_gguf_linear_pair` for
    kernels whose natural ABI is ``[rows, out_a + out_b]``. P9.C1 uses it for
    the Q8_0 shared-expert gate+up WMMA prefill path so the downstream
    ``silu_mul_dual_out_*`` kernel can consume the same layout as the selected
    MoE gate+up path. P9.H3 also uses it for resident Q8T16 shared gate/up
    decode so the two Q8_0 projections share one T16 kernel launch family.
    """

    resolved_backend = _weight_backend(weight_a, weight_b, backend=backend)
    use_wmma = _resolve_use_wmma_prefill(use_wmma_prefill)
    dispatch_a = resolve_gguf_linear_dispatch(weight_a, backend=resolved_backend, rows=rows)
    dispatch_b = resolve_gguf_linear_dispatch(weight_b, backend=resolved_backend, rows=rows)
    q8_t16_dual = KernelKey(
        resolved_backend,
        "linear",
        "gguf_q8_0_t16_v1",
        "t16_dual_gate_up_gemv_decode_bf16_bf16_out",
    )
    if (
        dispatch_a.abi == "t16"
        and dispatch_b.abi == "t16"
        and dispatch_a.key.quant == "gguf_q8_0_t16_v1"
        and dispatch_b.key.quant == "gguf_q8_0_t16_v1"
        and is_registered(q8_t16_dual)
    ):
        # P10.B4: decline the Q8T16 dual-gate-up GEMV fusion at rows>1 when
        # WMMA prefill is opted in, so the caller falls back through
        # ``launch_gguf_linear_pair`` (which itself declines T16 fusion when
        # WMMA prefill is on) all the way down to two singletons that each
        # take the dense Q8T16 WMMA prefill path.
        if use_wmma and rows > 1 and (
            _dispatch_can_use_t16_wmma_prefill(dispatch_a, rows=rows, in_features=in_features)
            or _dispatch_can_use_t16_wmma_prefill(dispatch_b, rows=rows, in_features=in_features)
        ):
            return False
        gguf_q8_0_t16_dual_gate_up_gemv_decode_bf16_bf16_out(
            x_ptr,
            weight_a.allocation("tiles").tensor.ptr,
            weight_b.allocation("tiles").tensor.ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            out_features,
            threads=_resolve_q8_t16_threads(threads),
            stream=stream,
            runtime=runtime,
        )
        return True

    if not use_wmma or rows <= 1:
        return False
    q8_prefill_raw = KernelKey(
        resolved_backend, "linear", "gguf_q8_0", "prefill_bf16_bf16_out"
    )
    q8_dual = KernelKey(
        resolved_backend,
        "linear",
        "gguf_q8_0",
        "wmma_prefill_dual_gate_up_bf16_bf16_out",
    )
    if (
        dispatch_a.abi == "raw"
        and dispatch_b.abi == "raw"
        and dispatch_a.key == q8_prefill_raw
        and dispatch_b.key == q8_prefill_raw
        and _wmma_prefill_shape_supported("gguf_q8_0", in_features)
        and is_registered(q8_dual)
    ):
        gguf_q8_0_wmma_prefill_dual_gate_up_bf16_bf16_out(
            x_ptr,
            weight_a.allocation("raw").tensor.ptr,
            weight_b.allocation("raw").tensor.ptr,
            out_ptr,
            rows,
            in_features,
            out_features,
            out_features,
            tile_m=16,
            tile_n=32,
            stream=stream,
            runtime=runtime,
        )
        return True
    return False


def _launch_pack8(fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs) -> None:
    fn(
        x_ptr,
        weight.allocation("qweight").tensor.ptr,
        weight.allocation("scales").tensor.ptr,
        weight.allocation("mins").tensor.ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


def _launch_t16_residual(
    fn,
    weight,
    x_ptr,
    residual_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    kwargs,
) -> None:
    fn(
        x_ptr,
        weight.allocation("tiles").tensor.ptr,
        residual_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


def _launch_pack8_residual(
    fn,
    weight,
    x_ptr,
    residual_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    kwargs,
) -> None:
    fn(
        x_ptr,
        weight.allocation("qweight").tensor.ptr,
        weight.allocation("scales").tensor.ptr,
        weight.allocation("mins").tensor.ptr,
        residual_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


def _launch_dense_bf16_residual(
    fn,
    weight,
    x_ptr,
    residual_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    kwargs,
) -> None:
    fn(
        x_ptr,
        weight.allocation("raw").tensor.ptr,
        residual_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


def _launch_raw(fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs) -> None:
    fn(
        x_ptr,
        weight.allocation("raw").tensor.ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


def _launch_dense_bf16(fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs) -> None:
    fn(
        x_ptr,
        weight.allocation("raw").tensor.ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


def _launch_t16(fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs) -> None:
    fn(
        x_ptr,
        weight.allocation("tiles").tensor.ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


def _device_ranges_overlap(
    left_ptr: int,
    left_nbytes: int,
    right_ptr: int,
    right_nbytes: int,
) -> bool:
    return int(left_ptr) < int(right_ptr) + int(right_nbytes) and int(
        right_ptr
    ) < int(left_ptr) + int(left_nbytes)


def _launch_t16_f16_rocblas(
    quant,
    fn,
    weight,
    x_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    kwargs,
    *,
    cast_activation: bool = True,
    tile_override: int | None = None,
    activation_inplace_override: bool | None = None,
) -> None:
    session = _t16_f16_rocblas_prefill_session.get()
    if session is None:
        raise RuntimeError("T16 F16/rocBLAS launch escaped its owner session")
    tile = (
        int(tile_override)
        if tile_override is not None
        else session.tile_out_features(rows, in_features, out_features, quant=quant)
    )
    if tile is None:
        raise RuntimeError("T16 F16/rocBLAS dispatch escaped its shape policy")
    activation_inplace = (
        bool(activation_inplace_override)
        if activation_inplace_override is not None
        else session.activation_is_inplace(
            rows, in_features, out_features, quant=quant
        )
    )
    required = {
        "activation": q6_k_f16_input_nbytes(rows, in_features),
        "weight": q6_k_f16_weight_nbytes(in_features, tile),
        "output": q6_k_f16_output_nbytes(rows, tile),
    }
    available = {
        "activation": (
            int(rows) * int(in_features) * 2
            if activation_inplace
            else int(session.x_f16_nbytes)
        ),
        "weight": int(session.weight_f16_nbytes),
        "output": int(session.out_f16_nbytes),
    }
    for name, nbytes in required.items():
        if nbytes > available[name]:
            raise ValueError(
                f"T16 F16/rocBLAS {name} plane is too small: "
                f"required={nbytes}, available={available[name]}"
            )
    tiles_ptr = int(weight.allocation("tiles").tensor.ptr)
    block_bytes_per_output = {
        "gguf_q4_k_t16_v1": 148,
        "gguf_q5_k_t16_v1": 180,
        "gguf_q6_k_t16_qmicro_planar_v1": 210,
    }[quant]
    live_regions = {
        "activation input": (int(x_ptr), int(rows) * int(in_features) * 2),
        "resident tiles": (
            tiles_ptr,
            int(out_features)
            * (int(in_features) // 256)
            * block_bytes_per_output,
        ),
        "output": (int(out_ptr), int(rows) * int(out_features) * 2),
        "weight workspace": (
            int(session.weight_f16_ptr),
            required["weight"],
        ),
        "output workspace": (
            int(session.out_f16_ptr),
            required["output"],
        ),
    }
    if not activation_inplace:
        live_regions["activation workspace"] = (
            int(session.x_f16_ptr),
            required["activation"],
        )
    names = tuple(live_regions)
    for index, left_name in enumerate(names):
        left_ptr, left_nbytes = live_regions[left_name]
        for right_name in names[index + 1 :]:
            right_ptr, right_nbytes = live_regions[right_name]
            if _device_ranges_overlap(
                left_ptr,
                left_nbytes,
                right_ptr,
                right_nbytes,
            ):
                raise ValueError(
                    "T16 F16/rocBLAS live regions overlap: "
                    f"{left_name} and {right_name}"
                )
    fn(
        int(x_ptr),
        tiles_ptr,
        int(out_ptr),
        int(x_ptr) if activation_inplace else int(session.x_f16_ptr),
        int(session.weight_f16_ptr),
        int(session.out_f16_ptr),
        int(rows),
        int(in_features),
        int(out_features),
        tile_out_features=tile,
        stream=int(kwargs.get("stream", 0)),
        dequant_library=session.dequant_library,
        cast_library=session.cast_library,
        rocblas=session.rocblas,
        solution_index=session.solution_index(rows, in_features, tile),
        cast_activation=cast_activation,
        runtime=kwargs.get("runtime"),
    )


def _launch_t16_q5_raw_mmq(
    fn,
    weight,
    x_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    kwargs,
) -> None:
    session = _q5_raw_mmq_target_session.get()
    if session is None:
        raise RuntimeError("Q5 raw MMQ launch escaped its target session")
    source_layout = bool(session.source_layout and rows == 32)
    required = (
        q8_1_d4s4_f32_kmajor_nbytes(rows, in_features)
        if source_layout
        else q8_1_d4s4_f32_nbytes(rows, in_features)
    )
    if required > session.workspace_nbytes:
        raise ValueError(
            "Q5 raw MMQ workspace is too small: "
            f"required={required}, available={session.workspace_nbytes}"
        )
    raw_weight_ptr = int(weight.allocation("raw").tensor.ptr)
    runtime = kwargs.get("runtime") or get_hip_runtime()
    stream = int(kwargs.get("stream", 0))
    mmq_kwargs = {
        "stream": stream,
        "runtime": runtime,
        "library": (
            session.library
            if source_layout
            else session.quant_library or session.library
        ),
    }
    quantize_kwargs = {
        **mmq_kwargs,
        "library": session.quant_library or session.library,
    }
    quantize = (
        gguf_q8_1_d4s4_f32_quantize_bf16_kmajor
        if source_layout
        else gguf_q8_1_d4s4_f32_quantize_bf16
    )
    quantize(
        x_ptr,
        session.workspace_ptr,
        rows,
        in_features,
        **quantize_kwargs,
    )
    fn(
        session.workspace_ptr,
        raw_weight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **mmq_kwargs,
    )


def _launch_t16_q5_planar_dp4a(
    fn,
    weight,
    x_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    kwargs,
) -> None:
    """Shared d4s4 producer + grouped integer-dp4a planar-Q5 decode (R24)."""

    del fn
    session = _q5_raw_mmq_target_session.get()
    if session is None:
        raise RuntimeError("Q5 planar dp4a launch escaped its target session")
    if session.planar_library is None:
        raise RuntimeError("Q5 planar dp4a session is missing its library")
    required = q8_1_d4s4_f32_nbytes(rows, in_features)
    if required > session.workspace_nbytes:
        raise ValueError(
            "Q5 planar dp4a workspace is too small: "
            f"required={required}, available={session.workspace_nbytes}"
        )
    planar_ptr = int(weight.allocation("qmicro_planar").tensor.ptr)
    runtime = kwargs.get("runtime") or get_hip_runtime()
    stream = int(kwargs.get("stream", 0))
    quant_kwargs = {
        "stream": stream,
        "runtime": runtime,
        "library": session.quant_library or session.library,
    }
    gguf_q8_1_d4s4_f32_quantize_bf16(
        x_ptr,
        session.workspace_ptr,
        rows,
        in_features,
        **quant_kwargs,
    )
    gguf_q5_k_qmicro_planar_q8_1_dp4a_grouped_bf16_bf16_out(
        session.workspace_ptr,
        planar_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        stream=stream,
        runtime=runtime,
        library=session.planar_library,
    )


def _launch_raw_mmq_d4x3(fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs) -> None:
    session = _q8_mmq_prefill_session.get()
    if session is None:
        raise RuntimeError("Q8 MMQ launch escaped its prefill workspace session")

    max_risks = int(rows) * int(out_features)
    required_risk_bytes = max_risks * ctypes.sizeof(ctypes.c_int32)
    if required_risk_bytes > session.risk_indices_nbytes:
        raise ValueError(
            "Q8 MMQ risk-index queue is too small: "
            f"required={required_risk_bytes}, available={session.risk_indices_nbytes}"
        )

    regions = {
        "workspace": (session.workspace_ptr, session.workspace_nbytes),
        "risk counter": (session.risk_count_ptr, session.risk_count_nbytes),
        "risk-index queue": (session.risk_indices_ptr, session.risk_indices_nbytes),
        "BF16 activation input": (int(x_ptr), int(rows) * int(in_features) * 2),
        "BF16 output": (int(out_ptr), int(rows) * int(out_features) * 2),
    }
    names = tuple(regions)
    for index, left_name in enumerate(names):
        left_ptr, left_nbytes = regions[left_name]
        for right_name in names[index + 1 :]:
            if {left_name, right_name} == {"BF16 activation input", "BF16 output"}:
                continue
            right_ptr, right_nbytes = regions[right_name]
            if max(left_ptr, right_ptr) < min(
                left_ptr + left_nbytes,
                right_ptr + right_nbytes,
            ):
                raise ValueError(f"Q8 MMQ {left_name} overlaps {right_name}")

    runtime = kwargs.get("runtime") or get_hip_runtime()
    stream = int(kwargs.get("stream", 0))
    runtime.memset_async(
        session.risk_count_ptr,
        0,
        ctypes.sizeof(ctypes.c_int32),
        stream,
    )
    mmq_kwargs = {
        "stream": stream,
        "runtime": runtime,
        "library": session.library,
    }
    qweight_ptr = weight.allocation("raw").tensor.ptr
    gguf_q8_0_mmq128_quantize_bf16_d4x3(
        x_ptr,
        session.workspace_ptr,
        rows,
        in_features,
        **mmq_kwargs,
    )
    fn(
        session.workspace_ptr,
        qweight_ptr,
        out_ptr,
        session.risk_count_ptr,
        session.risk_indices_ptr,
        max_risks,
        session.policy.risk_threshold,
        rows,
        in_features,
        out_features,
        **mmq_kwargs,
    )
    gguf_q8_0_mmq128_sparse_exact_correct_bf16(
        x_ptr,
        qweight_ptr,
        out_ptr,
        session.risk_count_ptr,
        session.risk_indices_ptr,
        max_risks,
        rows,
        in_features,
        out_features,
        **mmq_kwargs,
    )


def _pack8_decode_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    out_features: int,
) -> GGUFLinearDispatch:
    if (
        dispatch.abi == "raw"
        and rows == 1
        and out_features % 8 == 0
        and dispatch.key.quant in {"gguf_q8_0", "gguf_q5_k", "gguf_q6_k"}
        and dispatch.key.variant in {"gemv_bf16_bf16_out", "gemv_bf16_f32_out"}
    ):
        return GGUFLinearDispatch(
            KernelKey(
                dispatch.key.backend,
                dispatch.key.layer,
                dispatch.key.quant,
                f"pack8_{dispatch.key.variant}",
            ),
            dispatch.abi,
        )
    return dispatch


_Q5_PLANAR_DP4A_TARGET_ENV = "HIPENGINE_C8_Q5_PLANAR_DP4A"
_Q5_PLANAR_DP4A_VARIANT = "q8_1_dp4a_grouped_bf16_bf16_out"


def _q5_planar_dp4a_target_enabled(session: _Q5RawMMQTargetSession) -> bool:
    """Opt-in C8-P2 planar-dp4a owner pass at R24; default-off (L4 pending)."""

    if not session.planar_dp4a:
        return False
    raw = os.environ.get(_Q5_PLANAR_DP4A_TARGET_ENV, "0").strip().lower()
    if raw in {"", "1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{_Q5_PLANAR_DP4A_TARGET_ENV} must be a boolean value")


def _q5_raw_mmq_target_eligible(
    weight: GGUFDeviceWeight,
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> bool:
    if _q5_raw_mmq_target_session.get() is None or not (
        weight.spec.layout == LAYOUT_GGUF_Q5_K_T16
        and weight.spec.quant_key == "gguf_q5_k_t16_v1"
        and int(rows) in {24, 32}
        and (int(in_features), int(out_features)) == (6_144, 5_120)
    ):
        return False
    try:
        weight.allocation("raw")
    except KeyError:
        return False
    return True


def _q5_raw_mmq_target_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    weight: GGUFDeviceWeight,
    rows: int,
    in_features: int,
    out_features: int,
) -> GGUFLinearDispatch:
    """Select the operation-complete raw-Q5 MMQ target owner."""

    session = _q5_raw_mmq_target_session.get()
    if session is None or not (
        dispatch.abi == "t16"
        and dispatch.key.quant == "gguf_q5_k_t16_v1"
        and _q5_raw_mmq_target_eligible(
            weight,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
        )
    ):
        return dispatch
    source_layout = bool(session.source_layout and rows == 32)
    if _q5_planar_dp4a_target_enabled(session) and rows == 24:
        # The planar branch is R24-only by shape; the source-layout owner
        # keeps R32 because this branch never matches rows == 32
        # (source_layout only applies at rows == 32 downstream).
        try:
            planar_allocation = weight.allocation("qmicro_planar")
        except KeyError:
            planar_allocation = None
        if planar_allocation is not None:
            key = KernelKey(
                dispatch.key.backend,
                dispatch.key.layer,
                "gguf_q5_k",
                _Q5_PLANAR_DP4A_VARIANT,
            )
            if is_registered(key):
                return GGUFLinearDispatch(key, "t16_q5_planar_dp4a")
    required = (
        q8_1_d4s4_f32_kmajor_nbytes(rows, in_features)
        if source_layout
        else q8_1_d4s4_f32_nbytes(rows, in_features)
    )
    if required > session.workspace_nbytes:
        raise ValueError(
            "Q5 raw MMQ workspace is too small: "
            f"required={required}, available={session.workspace_nbytes}"
        )
    variant = (
        "mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out"
        if source_layout
        else "mmq32_q8_1_d4s4_f32_bf16_bf16_out"
    )
    key = KernelKey(
        dispatch.key.backend,
        dispatch.key.layer,
        "gguf_q5_k",
        variant,
    )
    if not is_registered(key):
        return dispatch
    return GGUFLinearDispatch(key, "t16_q5_raw_mmq")


def _q8_mmq_prefill_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> GGUFLinearDispatch:
    """Select D4-Q8_1 MMQ only inside a model-plugin workspace session."""

    session = _q8_mmq_prefill_session.get()
    if session is None or not session.policy(rows, in_features, out_features):
        return dispatch
    if not (
        dispatch.abi == "raw"
        and dispatch.key.quant == "gguf_q8_0"
        and dispatch.key.variant == "prefill_bf16_bf16_out"
        and in_features % 256 == 0
        and out_features % 16 == 0
    ):
        return dispatch
    required = q8_mmq_d4x3_nbytes(rows, in_features)
    if required > session.workspace_nbytes:
        raise ValueError(
            "Q8 MMQ D4 workspace is too small: "
            f"required={required}, available={session.workspace_nbytes}"
        )
    return GGUFLinearDispatch(
        KernelKey(
            dispatch.key.backend,
            dispatch.key.layer,
            dispatch.key.quant,
            "mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out",
        ),
        "raw_mmq_d4x3",
    )


def _exact_q8_prefill_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    out_features: int,
) -> GGUFLinearDispatch:
    """Reuse activations and raw-Q8 weights without changing dot association."""

    if (
        dispatch.abi == "raw"
        and rows > 1
        and out_features % 8 == 0
        and dispatch.key.quant == "gguf_q8_0"
        and dispatch.key.variant == "prefill_bf16_bf16_out"
    ):
        variant = "pack8_gemv_bf16_bf16_out"
        if rows >= 8:
            # Keep enough column blocks to fill the device at short/narrow
            # shapes. Once the measured grid is large enough, 16x4 halves
            # activation reloads while preserving every dot association.
            tile16_row_threshold = (
                512 if out_features <= 512 else 64 if out_features <= 2048 else 32
            )
            if out_features % 16 == 0 and rows >= tile16_row_threshold:
                variant = "exact_prefill_tile16x4_bf16_bf16_out"
            else:
                variant = (
                    "exact_prefill_tile8x2_bf16_bf16_out"
                    if rows < 32 and out_features <= 512
                    else "exact_prefill_tile8x4_bf16_bf16_out"
                )
        return GGUFLinearDispatch(
            KernelKey(
                dispatch.key.backend,
                dispatch.key.layer,
                dispatch.key.quant,
                variant,
            ),
            dispatch.abi,
        )
    return dispatch


def _gemv_decode_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    use_gemv_decode: bool,
) -> GGUFLinearDispatch:
    """Rewrite ``pack8_gemv_*`` -> ``pack8_gemv_decode_*`` for supported quants.

    A no-op unless all of the following hold:

    * ``use_gemv_decode`` is ``True`` (kwarg / session / env opt-in resolved).
    * ``rows == 1`` (prefill / bulk paths are not affected).
    * ``dispatch.abi == "raw"`` (the new GEMV decode kernel reads raw GGUF
      bytes via the same single ``raw`` allocation as the legacy decoder).
    * ``dispatch.key.quant`` ships a registered ``pack8_gemv_decode_*`` family
      (currently P9.B3 ``gguf_q8_0``; the Q5_K/Q6_K dense decode variants
      added in P9.B4b cover the lm-head case via separate runner wiring).
    * ``dispatch.key.variant`` is one of the ``pack8_gemv_*`` aliases
      (i.e. ``_pack8_decode_dispatch`` already rewrote the raw decoder).
    * The rewritten registry key is actually registered. If not, the
      function returns the original ``dispatch`` unchanged so the runtime
      transparently falls back to the legacy decoder.
    """

    if not use_gemv_decode or rows != 1:
        return dispatch
    if dispatch.abi != "raw":
        return dispatch
    variant = dispatch.key.variant
    if variant.startswith("pack8_gemv_") and not variant.startswith("pack8_gemv_decode_"):
        suffix = variant[len("pack8_gemv_") :]
    elif dispatch.key.quant == "gguf_q4_k" and variant.startswith("gemv_"):
        # Raw Q4_K has no generic raw-pack8 registry key: jump directly from
        # its scalar fallback to the separately registered decode family.
        suffix = variant[len("gemv_") :]
    else:
        return dispatch
    rewritten_variant = f"pack8_gemv_decode_{suffix}"
    rewritten_key = KernelKey(
        dispatch.key.backend,
        dispatch.key.layer,
        dispatch.key.quant,
        rewritten_variant,
    )
    if not is_registered(rewritten_key):
        # Registry miss: fall back to the legacy decoder without raising.
        # ``is_registered`` is an exact-key check so the cpu_reference fp16
        # ``linear`` catch-all does not silently route to a kernel whose
        # ABI does not match the GGUF launcher.
        return dispatch
    return GGUFLinearDispatch(rewritten_key, dispatch.abi)


def _registered_variant_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    variant: str | None,
) -> GGUFLinearDispatch:
    """Resolve an explicit same-ABI four-axis sibling or retain the default."""

    if variant is None or rows != 1 or dispatch.abi != "raw":
        return dispatch
    key = KernelKey(
        dispatch.key.backend,
        dispatch.key.layer,
        dispatch.key.quant,
        variant,
    )
    if not is_registered(key):
        return dispatch
    return GGUFLinearDispatch(key, dispatch.abi)


def _t16_c1_variant_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> GGUFLinearDispatch:
    """Select an architecture-qualified exact serial-c1 sibling by shape."""

    if (
        rows != 1
        or dispatch.abi != "t16"
        or _native_batch_decode_session_enabled
    ):
        return dispatch
    policies = backend_package_capability(
        dispatch.key.backend,
        "GGUF_T16_C1_VARIANTS_BY_QUANT_SHAPE",
        {},
    )
    quant_policies = (
        policies.get(dispatch.key.quant, {})
        if isinstance(policies, Mapping)
        else {}
    )
    if not isinstance(quant_policies, Mapping):
        return dispatch
    variant = quant_policies.get((int(in_features), int(out_features)))
    if not isinstance(variant, str) or not variant:
        return dispatch
    key = KernelKey(
        dispatch.key.backend,
        dispatch.key.layer,
        dispatch.key.quant,
        variant,
    )
    _ensure_linear_kernel_registered(key)
    if not is_registered(key):
        return dispatch
    return GGUFLinearDispatch(key, dispatch.abi)


def _q6_planar_rowtile_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    use_wmma: bool,
) -> GGUFLinearDispatch:
    """Weight-amortized small-B rowtile leaf for the planar-Q6 t16 quant.

    The per-row decode kernel re-reads the full tile set once per row
    (grid_y = rows); the registered planar rowtile reads each tile once for all
    rows and is bit-identical to it
    (tests/test_gguf_q6_planar_rowtile_dispatch_route.py). Fires only for rows
    2-8 on the decode leaves and never overrides an explicit WMMA opt-in —
    ``_wmma_prefill_dispatch`` runs after this step for that precedence.
    """

    if use_wmma or not 2 <= int(rows) <= 8:
        return dispatch
    if (
        dispatch.abi != "t16"
        or dispatch.key.quant != "gguf_q6_k_t16_qmicro_planar_v1"
    ):
        return dispatch
    rowtile_variant = {
        "t16_gemv_decode_bf16_f32_out": "t16_gemv_rowtile_bf16_f32_out",
        "t16_gemv_decode_bf16_bf16_out": "t16_gemv_rowtile_bf16_bf16_out",
    }.get(dispatch.key.variant)
    if rowtile_variant is None:
        return dispatch
    key = KernelKey(
        dispatch.key.backend,
        dispatch.key.layer,
        dispatch.key.quant,
        rowtile_variant,
    )
    if not is_registered(key):
        return dispatch
    return GGUFLinearDispatch(key, dispatch.abi)


def _native_batch_decode_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> GGUFLinearDispatch:
    """Select registered compact c=2..8 native projection families."""

    verifier_rowtile_shapes = backend_package_capability(
        dispatch.key.backend,
        "GGUF_T16_TARGET_VERIFIER_ROWTILE_SHAPES_BY_QUANT",
        {},
    )
    quant_verifier_shapes = (
        verifier_rowtile_shapes.get(dispatch.key.quant, ())
        if isinstance(verifier_rowtile_shapes, Mapping)
        else ()
    )
    verifier_rowtile_enabled = bool(
        _target_verifier_rowtile_session_enabled.get()
        and (int(in_features), int(out_features)) in quant_verifier_shapes
    )
    if (
        not _native_batch_decode_session_enabled
        and not verifier_rowtile_enabled
    ) or rows <= 1 or rows > 8:
        return dispatch
    t16_rowtile_max_rows = 6
    t16_rowtile_limits = backend_package_capability(
        dispatch.key.backend,
        "GGUF_T16_NATIVE_ROWTILE_MAX_ROWS_BY_QUANT",
        {},
    )
    if isinstance(t16_rowtile_limits, Mapping):
        raw = t16_rowtile_limits.get(dispatch.key.quant, 6)
        if isinstance(raw, Mapping):
            # Per-shape caps: {"default": N, "shapes": {(in_features,
            # out_features): N}}. Used when a backend measures different
            # rowtile bounds for different shapes of the same quant (e.g.
            # Q5 27B ssm_out/ffn_down/qkv rowtile to c8 while the narrow
            # 0.8B shape keeps the direct leaf above c4).
            try:
                t16_rowtile_max_rows = int(raw.get("default", 6))
            except (TypeError, ValueError):
                t16_rowtile_max_rows = 0
            shape_caps = raw.get("shapes", {})
            if isinstance(shape_caps, Mapping):
                try:
                    t16_rowtile_max_rows = max(
                        t16_rowtile_max_rows,
                        int(
                            shape_caps.get(
                                (int(in_features), int(out_features)),
                                0,
                            )
                        ),
                    )
                except (TypeError, ValueError):
                    pass
        else:
            try:
                t16_rowtile_max_rows = int(raw)
            except (TypeError, ValueError):
                t16_rowtile_max_rows = 0
    if (
        dispatch.abi == "t16"
        and dispatch.key.variant == "t16_gemv_decode_bf16_bf16_out"
    ):
        if rows <= t16_rowtile_max_rows:
            rewritten_key = KernelKey(
                dispatch.key.backend,
                dispatch.key.layer,
                dispatch.key.quant,
                "t16_gemv_rowtile_bf16_bf16_out",
            )
            if is_registered(rewritten_key):
                return GGUFLinearDispatch(rewritten_key, dispatch.abi)
        direct_shapes = backend_package_capability(
            dispatch.key.backend,
            "GGUF_T16_NATIVE_DIRECT_SHAPES_BY_QUANT",
            {},
        )
        quant_direct_shapes = (
            direct_shapes.get(dispatch.key.quant, ())
            if isinstance(direct_shapes, Mapping)
            else ()
        )
        try:
            use_direct = (
                int(in_features),
                int(out_features),
            ) in quant_direct_shapes
        except TypeError:
            use_direct = False
        if use_direct:
            return dispatch
        # Native widths above a backend's measured rowtile bound normally use
        # the same-ABI WMMA sibling. Backends can retain the direct leaf for
        # independently measured exact shapes through the policy above.
        rewritten_key = KernelKey(
            dispatch.key.backend,
            dispatch.key.layer,
            dispatch.key.quant,
            "t16_wmma_prefill_bf16_bf16_out",
        )
        if is_registered(rewritten_key):
            return GGUFLinearDispatch(rewritten_key, dispatch.abi)
        return dispatch
    if dispatch.abi != "raw" or dispatch.key.quant != "gguf_q6_k":
        return dispatch
    variant = dispatch.key.variant
    if not variant.startswith("prefill_"):
        return dispatch
    rewritten_key = KernelKey(
        dispatch.key.backend,
        dispatch.key.layer,
        dispatch.key.quant,
        f"pack8_gemv_{variant[len('prefill_') :]}",
    )
    return GGUFLinearDispatch(rewritten_key, dispatch.abi)


def _q6_integer_mmq_prefill_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> GGUFLinearDispatch:
    """Select the backend-declared dense integer route inside its owner context."""

    if q6_dense_integer_mmq_workspace() is None or dispatch.abi != "t16":
        return dispatch
    policy = backend_package_capability(
        dispatch.key.backend,
        "GGUF_Q6_DENSE_INTEGER_MMQ_PREFILL_POLICY",
        {},
    )
    if not isinstance(policy, Mapping):
        return dispatch
    entry = policy.get(dispatch.key.quant)
    if not isinstance(entry, Mapping):
        return dispatch
    try:
        admitted = (
            int(entry["min_rows"]) <= int(rows) <= int(entry["max_rows"])
            and (int(in_features), int(out_features)) in entry["shapes"]
        )
        variant = str(entry["variant"])
    except (KeyError, TypeError, ValueError):
        return dispatch
    if not admitted:
        return dispatch
    key = KernelKey(
        dispatch.key.backend,
        dispatch.key.layer,
        dispatch.key.quant,
        variant,
    )
    return GGUFLinearDispatch(key, dispatch.abi) if is_registered(key) else dispatch


_T16_F16_ROCBLAS_ROUTE_BY_QUANT = MappingProxyType(
    {
        "gguf_q4_k_t16_v1": (
            "f16_rocblas_t16_bf16_bf16_out",
            "t16_q4_f16_rocblas",
        ),
        "gguf_q5_k_t16_v1": (
            "f16_rocblas_t16_bf16_bf16_out",
            "t16_q5_f16_rocblas",
        ),
        "gguf_q6_k_t16_qmicro_planar_v1": (
            "f16_rocblas_t16_qmicro_planar_bf16_bf16_out",
            "t16_q6_f16_rocblas",
        ),
    }
)


def _q6_t16_f16_rocblas_prefill_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> GGUFLinearDispatch:
    """Select source-F16 arithmetic only inside its bounded owner context."""

    session = _t16_f16_rocblas_prefill_session.get()
    route = _T16_F16_ROCBLAS_ROUTE_BY_QUANT.get(dispatch.key.quant)
    if (
        session is None
        or int(rows) < int(session.min_rows)
        or int(rows) > int(session.max_rows)
        or dispatch.abi != "t16"
        or route is None
        or not dispatch.key.variant.endswith("_bf16_bf16_out")
        or session.tile_out_features(
            rows,
            in_features,
            out_features,
            quant=dispatch.key.quant,
        )
        is None
    ):
        return dispatch
    assert route is not None
    default_variant, launch_abi = route
    variant = session.linear_variant(
        rows,
        in_features,
        out_features,
        quant=dispatch.key.quant,
        default=default_variant,
    )
    key = KernelKey(
        dispatch.key.backend,
        "linear",
        dispatch.key.quant,
        variant,
    )
    if not is_registered(key) and variant != default_variant:
        key = KernelKey(
            dispatch.key.backend,
            "linear",
            dispatch.key.quant,
            default_variant,
        )
    return GGUFLinearDispatch(key, launch_abi) if is_registered(key) else dispatch


def _wmma_prefill_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
    use_wmma: bool,
) -> GGUFLinearDispatch:
    """Rewrite decode-shape variants -> ``wmma_prefill_*`` for supported quants.

    A no-op unless all of the following hold:

    * ``use_wmma`` is ``True`` (kwarg / session / env opt-in resolved).
    * ``rows > 1`` (decode is not affected).
    * ``dispatch.abi`` is one of the supported source ABIs:
      - ``"raw"`` -> ``"wmma_raw"`` (the legacy raw-GGUF WMMA prefill
        family for ``gguf_q8_0`` and raw-layout ``gguf_q4_k``).
      - ``"t16"`` -> ``"t16"`` (P10.B4: ``gguf_q8_0_t16_v1`` rewrites the
        ``t16_gemv_decode_*`` variant to ``t16_wmma_prefill_*`` and keeps
        the same allocation name + launch signature, so the existing
        ``_launch_t16`` ABI helper is reused).
    * ``dispatch.key.quant`` ships a registered WMMA prefill family.
    * ``dispatch.key.variant`` is the rows>1 alias produced by
      ``_variant_for_rows``.
    * ``in_features`` satisfies the quant's K-block alignment constraint.
    """

    if not use_wmma or rows <= 1:
        return dispatch
    if dispatch.abi == "raw":
        if not _dispatch_can_use_wmma_prefill(dispatch, rows=rows, in_features=in_features):
            return dispatch
        variant = dispatch.key.variant
        return GGUFLinearDispatch(
            KernelKey(
                dispatch.key.backend,
                dispatch.key.layer,
                dispatch.key.quant,
                f"wmma_{variant}",
            ),
            "wmma_raw",
        )
    if dispatch.abi == "t16":
        if not _dispatch_can_use_t16_wmma_prefill(dispatch, rows=rows, in_features=in_features):
            return dispatch
        # The T16 decode variant is named ``t16_gemv_decode_<in>_<out>_out``;
        # the rewrite swaps that for ``t16_wmma_prefill_<in>_<out>_out`` while
        # keeping the ``t16`` ABI (same (x, tiles, out, rows, in_f, out_f)
        # signature, additional (tile_m, tile_n) kwargs).
        variant = dispatch.key.variant
        if not variant.startswith("t16_gemv_decode_"):
            return dispatch
        suffix = variant[len("t16_gemv_decode_") :]
        return GGUFLinearDispatch(
            KernelKey(
                dispatch.key.backend,
                dispatch.key.layer,
                dispatch.key.quant,
                f"t16_wmma_prefill_{suffix}",
            ),
            "t16",
        )
    return dispatch


def _dispatch_can_use_t16_wmma_prefill(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
) -> bool:
    """P10.B4 gate: T16 dense rows>1 only rewrites when the kernel is wired."""

    return (
        rows > 1
        and dispatch.abi == "t16"
        and dispatch.key.variant.startswith("t16_gemv_decode_")
        and dispatch.key.quant in _WMMA_PREFILL_QUANT_BLOCKS
        and _wmma_prefill_shape_supported(dispatch.key.quant, in_features)
    )


def _wmma_prefill_shape_supported(quant: str, in_features: int) -> bool:
    block = _WMMA_PREFILL_QUANT_BLOCKS.get(quant)
    return block is not None and in_features % block == 0


def _dispatch_can_use_wmma_prefill(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    in_features: int,
) -> bool:
    return (
        rows > 1
        and dispatch.abi == "raw"
        and dispatch.key.variant.startswith("prefill_")
        and _wmma_prefill_shape_supported(dispatch.key.quant, in_features)
    )


def _variant_for_rows(variant: str, *, rows: int) -> str:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if rows == 1:
        return variant
    if variant.startswith("pack8_"):
        return f"pack8_prefill_{variant[len('pack8_') :]}"
    if variant.startswith("gemv_"):
        return f"prefill_{variant[len('gemv_') :]}"
    if variant == "out":
        return "prefill_out"
    return variant


def _q4_pack8_wmma_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
    enabled: bool,
) -> GGUFLinearDispatch:
    """Select Laguna's wide dense/shared K-quant WMMA prefill leaves."""

    if not enabled or rows < 16:
        return dispatch
    if (
        dispatch.abi == "pack8"
        and dispatch.key.quant == "gguf_q4_k"
        and dispatch.key.variant
        in (
            "pack8_prefill_bf16_bf16_out",
            "pack8_exact_prefill_tile8x8_bf16_bf16_out",
        )
    ):
        key = KernelKey(
            dispatch.key.backend,
            dispatch.key.layer,
            dispatch.key.quant,
            "pack8_wmma_prefill_bf16_bf16_out",
        )
        abi = dispatch.abi
    elif (
        dispatch.abi == "raw"
        and dispatch.key.quant == "gguf_q6_k"
        and dispatch.key.variant == "prefill_bf16_bf16_out"
    ):
        key = KernelKey(
            dispatch.key.backend,
            dispatch.key.layer,
            dispatch.key.quant,
            "wmma_prefill_bf16_bf16_out",
        )
        abi = "wmma_raw"
    else:
        return dispatch
    if not is_registered(key):
        return dispatch
    return GGUFLinearDispatch(key, abi)


def _launch_raw_k_f32_ordered(
    fn,
    weight,
    x_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    kwargs,
) -> None:
    session = _q5_f32_ordered_prefill_session.get()
    if session is None:
        raise RuntimeError("raw-K F32 ordered dispatch escaped its owner session")
    fn(
        x_ptr,
        weight.allocation("raw").tensor.ptr,
        out_ptr,
        session.weight_f32_ptr,
        rows,
        in_features,
        out_features,
        stream=kwargs.get("stream", 0),
        library=session.library,
        runtime=kwargs.get("runtime"),
    )


def _launch_raw_k_f32_ordered_activation_tile_k_row(
    fn,
    weight,
    x_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    kwargs,
) -> None:
    session = _q5_f32_ordered_prefill_session.get()
    if session is None or int(session.activation_bf16_ptr) <= 0:
        raise RuntimeError(
            "raw-K F32 ordered activation-tile-K-row dispatch escaped its owner session"
        )
    fn(
        x_ptr,
        weight.allocation("raw").tensor.ptr,
        out_ptr,
        session.weight_f32_ptr,
        session.activation_bf16_ptr,
        rows,
        in_features,
        out_features,
        stream=kwargs.get("stream", 0),
        library=session.library,
        runtime=kwargs.get("runtime"),
    )


def _launch_raw_k_f32_resident_activation_tile_k_row(
    fn,
    weight,
    x_ptr,
    out_ptr,
    rows,
    in_features,
    out_features,
    kwargs,
) -> None:
    session = _q5_f32_ordered_prefill_session.get()
    if session is None or int(session.activation_bf16_ptr) <= 0:
        raise RuntimeError(
            "resident Q5 activation-tile-K-row dispatch escaped its owner session"
        )
    raw_weight_ptr = int(weight.allocation("raw").tensor.ptr)
    planes = session.resident_weight_f32_planes
    assert planes is not None
    plane = planes.get(raw_weight_ptr)
    if (
        plane is None
        or int(plane.in_features) != int(in_features)
        or int(plane.out_features) != int(out_features)
    ):
        raise RuntimeError("resident Q5 dispatch has no exact raw-pointer plane")
    fn(
        x_ptr,
        plane.weight_f32_ptr,
        out_ptr,
        session.activation_bf16_ptr,
        rows,
        in_features,
        out_features,
        stream=kwargs.get("stream", 0),
        library=session.library,
        runtime=kwargs.get("runtime"),
    )


def _launch_wmma_raw(fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs) -> None:
    # The WMMA prefill wrapper has the same (x, qweight, out, rows, in_f, out_f)
    # raw-pointer signature as _launch_raw, but accepts (tile_m, tile_n, stream)
    # in place of (threads, stream). Strip ``threads`` if the caller set it.
    wmma_kwargs = {k: v for k, v in kwargs.items() if k != "threads"}
    fn(
        x_ptr,
        weight.allocation("raw").tensor.ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **wmma_kwargs,
    )


def _ensure_linear_kernel_registered(key: KernelKey) -> None:
    # Registry plan tests clear global registrations; keep GGUF runtime dispatch
    # independent of previous test/import order without overwriting tests that
    # deliberately replace one dispatch key with a fixture kernel.
    if is_registered(key):
        return
    register_dense_gemv_kernels()
    register_gguf_k_gemv_kernels()
    register_gguf_k_t16_selected_prefill_kernels()
    register_gguf_k_mmq_prefill_kernels()
    register_gguf_q4_k_gemv_kernels()
    register_gguf_q4_k_prefill_kernels()
    register_gguf_q4_k_pack8_gemv_kernels()
    register_gguf_q5_k_f32_rocblas_prefill_kernels()
    register_gguf_q5_k_qmicro_planar_gemv_kernels()
    register_gguf_q6_k_f16_rocblas_prefill_kernels()
    register_gguf_q6_k_pack8_gemv_kernels()
    register_gguf_q6_k_t16_gemv_kernels()
    register_gguf_q8_0_mmq_prefill_kernels()
    register_gguf_q8_0_pack8_gemv_kernels()
    register_gguf_q8_0_prefill_kernels()
    register_gguf_q8_0_t16_gemv_kernels()
    register_gguf_q8_0_t16_prefill_kernels()
    register_gguf_t16_selected_gemv_kernels()
    register_laguna_launch_batch_kernels()
    load_backend_kernel_package(key.backend)


_LAUNCH_RESIDUAL_ABI = {
    "dense_bf16": _launch_dense_bf16_residual,
    "pack8": _launch_pack8_residual,
    "t16": _launch_t16_residual,
}


_LAUNCH_ABI = {
    "dense_bf16": _launch_dense_bf16,
    "pack8": _launch_pack8,
    "raw": _launch_raw,
    "raw_mmq_d4x3": _launch_raw_mmq_d4x3,
    "t16_q5_raw_mmq": _launch_t16_q5_raw_mmq,
    "t16_q5_planar_dp4a": _launch_t16_q5_planar_dp4a,
    "raw_k_f32_ordered": _launch_raw_k_f32_ordered,
    "raw_k_f32_ordered_activation_tile_k_row": (
        _launch_raw_k_f32_ordered_activation_tile_k_row
    ),
    "raw_k_f32_resident_activation_tile_k_row": (
        _launch_raw_k_f32_resident_activation_tile_k_row
    ),
    "t16": _launch_t16,
    "t16_q4_f16_rocblas": lambda *args: _launch_t16_f16_rocblas(
        "gguf_q4_k_t16_v1", *args
    ),
    "t16_q5_f16_rocblas": lambda *args: _launch_t16_f16_rocblas(
        "gguf_q5_k_t16_v1", *args
    ),
    "t16_q6_f16_rocblas": lambda *args: _launch_t16_f16_rocblas(
        "gguf_q6_k_t16_qmicro_planar_v1", *args
    ),
    "wmma_raw": _launch_wmma_raw,
}


__all__ = [
    "GGUF_ACTIVATION_BF16",
    "GGUF_ACTIVATION_F32",
    "GGUF_OUTPUT_BF16",
    "GGUF_OUTPUT_FP16",
    "GGUF_OUTPUT_F32",
    "PREFILL_F16_STAGING_MAX_ROWS",
    "PREFILL_F16_STAGING_VARIANT",
    "PrefillF16StagingWorkspace",
    "launch_gguf_q4_t16_sidecar_decode",
    "GGUFLinearDispatch",
    "Q5F32OrderedPrefillSession",
    "Q5F32ResidentPlane",
    "Q6T16F16RocblasPrefillSession",
    "T16F16RocblasPrefillSession",
    "gguf_wmma_prefill_enabled",
    "launch_gguf_linear",
    "launch_gguf_linear_q8_1",
    "launch_gguf_linear_residual",
    "launch_gguf_linear_moe_tail_host_batch",
    "launch_gguf_linear_pair",
    "launch_gguf_linear_pair_silu",
    "launch_gguf_linear_pair_concat",
    "launch_gguf_linear_raw_ptr",
    "launch_gguf_linear_triple",
    "gguf_native_batch_decode_enabled",
    "native_batch_decode_session",
    "prefill_f16_staging_enabled",
    "prefill_f16_staging_for",
    "prefill_f16_staging_session",
    "prefill_f16_staging_workspace",
    "q6_integer_mmq_for",
    "target_verifier_rowtile_session",
    "target_verifier_production_q4_rowtile_session",
    "target_verifier_wide_q6_shared4_session",
    "target_verifier_wide_q6_shared4_policy_enabled",
    "target_verifier_wide_q6_shared4_leaf_session",
    "TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ENV",
    "TARGET_VERIFIER_WIDE_Q6_SHARED4_ENV",
    "q4_pack8_dual_wmma_silu_prefill_session",
    "q4_t16_unequal_pair_prefill_session",
    "q5_f32_ordered_prefill_session",
    "q5_raw_mmq_target_session",
    "q6_t16_f16_rocblas_prefill_session",
    "t16_f16_rocblas_prefill_session",
    "q8_mmq_prefill_session",
    "q8_t16_dual_wmma_prefill_session",
    "raw_k_prefill_rowbatch",
    "raw_k_prefill_rowbatch_session",
    "raw_k_prefill_variant",
    "raw_k_prefill_variant_session",
    "resolve_gguf_linear_dispatch",
    "resolve_q8_mmq_prefill_policy",
    "set_wmma_prefill_enabled",
    "wmma_prefill_session",
]
