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
from hipengine.kernels.policy import (
    QWEN35_DENSE_H1024_GEOMETRY,
    QWEN35_DENSE_H5120_GEOMETRY,
    QWEN35_MOE_H2048_E256_GEOMETRY,
)
from hipengine.kernels.hip_gfx1100.attention.laguna_kv import (
    laguna_global_f16_projection_head_kv_nontemporal_tile2_bf16_spans,
    laguna_swa_f16_projection_head_kv_nontemporal_tile2_bf16_spans,
)
from hipengine.kernels.hip_gfx1100.attention.paged_attn_decode import (
    qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans,
)
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
    qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16,
    qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16_fp16state,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    qwen35_router_logits_bf16_f32w_auto_256,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
    gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_prefill import (
    gguf_q4_k_pack8_wmma_prefill_gfx1151_bf16_bf16_out,
    gguf_q6_k_wmma_prefill_16x32_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_q8_1_selected_prefill import (
    gguf_q6_k_t16_qmicro_planar_dense_q8_1_mmq64x64_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
    gguf_q4_k_pack8_dual_prefill_bf16_bf16_out,
    gguf_q4_k_pack8_dual_silu_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_t16_selected_prefill import (
    gguf_q4_k_t16_wmma_prefill_bf16_bf16_out,
    gguf_q4_k_t16_wmma_prefill_fp16_in_bf16_out,
    gguf_q4_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out,
    gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out,
    gguf_q4_k_t16_wmma_prefill_smallm_bf16_bf16_out,
    gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out,
    gguf_q4_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out,
    gguf_q4_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out,
    gguf_q4_k_t16_wmma_prefill_shared_b_fp16_in_bf16_out,
    gguf_q4_k_t16_wmma_prefill_shared_b2r1_fp16_in_bf16_out,
    gguf_q4_k_t16_wmma_prefill_shared_b2w2_fp16_in_bf16_out,
    gguf_q4_k_t16_wmma_prefill_shared_b2w4_fp16_in_bf16_out,
    gguf_q4_k_t16_wmma_prefill_shared_b3w8r3_fp16_in_bf16_out,
    gguf_q4_k_t16_wmma_prefill_shared_b2w2_bf16_bf16_out,
    gguf_q4_k_t16_wmma_prefill_shared_b2w4_bf16_bf16_out,
    gguf_q4_k_t16_wmma_prefill_shared_b3w8r3_bf16_bf16_out,
    gguf_q5_k_t16_wmma_prefill_bf16_bf16_out,
    gguf_q5_k_t16_wmma_prefill_fp16_in_bf16_out,
    gguf_q5_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out,
    gguf_q5_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out,
    gguf_q5_k_t16_wmma_prefill_shared8r3_fp16_in_bf16_out,
    gguf_q5_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out,
    gguf_q5_k_t16_wmma_prefill_shared8r3_bf16_bf16_out,
    gguf_q5_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
    gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr48_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared3r1_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r3_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r6_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r9_bf16_bf16_out,
    gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r4_bf16_bf16_out,
    gguf_q6_k_t16_wmma_prefill_bf16_bf16_out,
    gguf_q6_k_t16_wmma_prefill_shared3r1_bf16_bf16_out,
    gguf_q6_k_t16_wmma_prefill_shared4_bf16_bf16_out,
    gguf_q6_k_t16_wmma_prefill_shared6r1_bf16_bf16_out,
    gguf_q6_k_t16_wmma_prefill_shared8r3_bf16_bf16_out,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_t16_selected_gemv import (
    gguf_q4_k_t16_dense_dual_interleaved_tile2_local32_silu_bf16_bf16_out,
    gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out,
    gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out,
    gguf_q5_k_t16_gemv_rowtile_col8_bf16_bf16_out,
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


def _qwen35_08b_q4_pack8_dual_silu_t128(*args, **kwargs):
    """Bind the qualified 0.8B fused gate/up schedule to 128 threads."""

    kwargs["threads"] = 128
    return gguf_q4_k_pack8_dual_silu_bf16_bf16_out(*args, **kwargs)


def gguf_q4_k_t16_wmma_prefill_gfx1151_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    **kwargs,
):
    """Select the admitted strict one-row-tile WMMA owner for physical rows.

    Owner bands follow measured owner crossovers on the six physical Qwen3.8
    dense prefill shapes (synthetic Q4T16 microbench, bit-exact siblings):
    the low-VGPR 16-column owners (one out tile per 32-thread block, 16/24-
    float accumulators) roughly double effective weight bandwidth at rows
    17-144 where the 48-column owners are latency-bound at ~2 waves/SIMD
    (248 VGPRs). Per-shape periodic bands select 32-row/48-row low-VGPR,
    48-column, or shared-B owners; rows145+ keep shared-B. Unadmitted shapes
    retain the shared-B fail-closed fallback.
    """

    row_count = int(rows)
    shape = (int(in_features), int(out_features))
    if (
        row_count <= GGUF_Q4_T16_PHYSICAL_SMALLM_MAX_ROWS
        and shape in GGUF_Q4_T16_PHYSICAL_SMALLM_SHAPES
    ):
        # Scaling-campaign M2j (2026-08-31 rows2-20 sibling screens, all six
        # physical shapes, bit-exact vs shared-B): the 16-column low-VGPR
        # owner beats the one-row-tile smallm at every physical verify row
        # (1.19-1.65x at rows2-16), and the 32-thread shared-B sibling beats
        # even low-VGPR on the N5120 down-projection (0.716-0.727 ms vs
        # 0.772-0.775 ms). The former smallm band {6,8,12,16} is superseded
        # here; the smallm launcher stays registered under its explicit
        # variant for inventory and rollback.
        fn = (
            gguf_q4_k_t16_wmma_prefill_shared_b2w2_bf16_bf16_out
            if shape == (17_408, 5_120)
            else gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
        )
    elif (
        17 <= row_count <= GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWVGPR48_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWVGPR64_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWVGPR80_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWVGPR96_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_PLAIN128_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWVGPR128_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWVGPR48_128_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR144_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWVGPR48_144_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR144_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_SHARED2_192_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_SHARED2W4_192_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b2w4_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR144_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_SHARED2_192_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_SHARED2W2_192_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b2w2_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_SHARED2_192_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_SHARED2_256_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_SHARED2W2_256_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b2w2_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_SHARED3W8R3_384_MIN_ROWS
        <= row_count
        <= GGUF_Q4_T16_DENSE_SHARED2_384_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_SHARED3W8R3_384_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b3w8r3_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_SHARED2_256_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_SHARED2_384_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_SHARED2W2_384_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b2w2_bf16_bf16_out
    else:
        fn = gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out
    return fn(
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


def gguf_q5_k_t16_wmma_prefill_gfx1151_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    **kwargs,
):
    """Select measured low-VGPR Q5 owners on physical low-M shapes.

    Rows 17-32 take the 32-row/16-column owner on all six shapes. Later
    periodic bands through row144 select 32-row, 48-row, or plain owners by
    measured shape crossover. Shape and row misses retain plain. This
    selector is registered only for
    the dense linear prefill key; compact MoE and verifier aliases keep their
    independently qualified owners.
    """

    row_count = int(rows)
    shape = (int(in_features), int(out_features))
    if (
        GGUF_Q5_T16_DENSE_SHARED8R3_MIN_ROWS <= row_count
        <= GGUF_Q5_T16_DENSE_SHARED8R3_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_SHARED8R3_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_shared8r3_bf16_bf16_out
    elif (
        17 <= row_count <= GGUF_Q5_T16_DENSE_LOWVGPR_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR48_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWVGPR_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR48_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWVGPR48_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR48_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR64_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWVGPR64_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR64_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWVGPR80_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR64_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR96_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWVGPR96_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR96_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR96_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR128_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWVGPR48_128_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR128_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR144_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    else:
        fn = gguf_q5_k_t16_wmma_prefill_bf16_bf16_out
    return fn(
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


def gguf_q4_k_t16_wmma_prefill_gfx1151_fp16_in_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    **kwargs,
):
    """F16-staged twin of the Q4 dense prefill router (B2 P1).

    Same band ladder as the BF16 router; every band body is the registered
    input-F16 sibling. The BF16 router remains the selected strict fallback.
    """

    row_count = int(rows)
    shape = (int(in_features), int(out_features))
    if (
        row_count <= GGUF_Q4_T16_PHYSICAL_SMALLM_MAX_ROWS
        and shape in GGUF_Q4_T16_PHYSICAL_SMALLM_SHAPES
    ):
        fn = (
            gguf_q4_k_t16_wmma_prefill_shared_b2w2_fp16_in_bf16_out
            if shape == (17_408, 5_120)
            else gguf_q4_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
        )
    elif (
        17 <= row_count <= GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWVGPR48_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWVGPR64_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_fp16_in_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWVGPR80_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWVGPR96_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_PLAIN128_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_fp16_in_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWVGPR128_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWVGPR48_128_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR144_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWVGPR48_144_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR144_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_SHARED2_192_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_SHARED2W4_192_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b2w4_fp16_in_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR144_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_SHARED2_192_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_SHARED2W2_192_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b2w2_fp16_in_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_SHARED2_192_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_SHARED2_256_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_SHARED2W2_256_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b2w2_fp16_in_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_SHARED3W8R3_384_MIN_ROWS
        <= row_count
        <= GGUF_Q4_T16_DENSE_SHARED2_384_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_SHARED3W8R3_384_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b3w8r3_fp16_in_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_SHARED2_256_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_SHARED2_384_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_SHARED2W2_384_SHAPES
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b2w2_fp16_in_bf16_out
    else:
        fn = gguf_q4_k_t16_wmma_prefill_shared_b_fp16_in_bf16_out
    return fn(
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


def gguf_q5_k_t16_wmma_prefill_gfx1151_fp16_in_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    **kwargs,
):
    """F16-staged twin of the Q5 dense prefill router (B2 P1)."""

    row_count = int(rows)
    shape = (int(in_features), int(out_features))
    if (
        GGUF_Q5_T16_DENSE_SHARED8R3_MIN_ROWS <= row_count
        <= GGUF_Q5_T16_DENSE_SHARED8R3_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_SHARED8R3_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_shared8r3_fp16_in_bf16_out
    elif (
        17 <= row_count <= GGUF_Q5_T16_DENSE_LOWVGPR_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR48_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWVGPR_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR48_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWVGPR48_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR48_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR64_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWVGPR64_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR64_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWVGPR80_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR64_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR96_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWVGPR96_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR96_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR96_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR128_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWVGPR48_128_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        GGUF_Q5_T16_DENSE_LOWVGPR128_MAX_ROWS
        < row_count
        <= GGUF_Q5_T16_DENSE_LOWVGPR144_MAX_ROWS
        and shape in GGUF_Q5_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    else:
        fn = gguf_q5_k_t16_wmma_prefill_fp16_in_bf16_out
    return fn(
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


def gguf_q6_k_t16_wmma_prefill_gfx1151_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    **kwargs,
):
    """Select shared-weight WMMA only for the admitted standard-Q6 QKV."""

    row_count = int(rows)
    shape = (int(in_features), int(out_features))
    fn = (
        gguf_q6_k_t16_wmma_prefill_shared3r1_bf16_bf16_out
        if GGUF_Q6_PREFILL_SHARED3R1_MIN_ROWS <= row_count
        <= GGUF_Q6_PREFILL_SHARED3R1_MAX_ROWS
        and shape in GGUF_Q6_STANDARD_PREFILL_SHARED3R1_SHAPES
        else gguf_q6_k_t16_wmma_prefill_shared6r1_bf16_bf16_out
        if GGUF_Q6_STANDARD_PREFILL_SHARED6R1_MIN_ROWS <= row_count
        <= GGUF_Q6_STANDARD_PREFILL_SHARED6R1_MAX_ROWS
        and shape in GGUF_Q6_STANDARD_PREFILL_SHARED3R1_SHAPES
        else gguf_q6_k_t16_wmma_prefill_shared8r3_bf16_bf16_out
        if GGUF_Q6_STANDARD_PREFILL_SHARED8R3_MIN_ROWS <= row_count
        <= GGUF_Q6_STANDARD_PREFILL_SHARED8R3_MAX_ROWS
        and shape in GGUF_Q6_STANDARD_PREFILL_SHARED4_SHAPES
        else gguf_q6_k_t16_wmma_prefill_shared8r3_bf16_bf16_out
        if GGUF_Q6_STANDARD_PREFILL_SHARED8R3_HIGH_MIN_ROWS <= row_count
        <= GGUF_Q6_STANDARD_PREFILL_SHARED8R3_HIGH_MAX_ROWS
        and shape in GGUF_Q6_STANDARD_PREFILL_SHARED4_SHAPES
        else gguf_q6_k_t16_wmma_prefill_shared4_bf16_bf16_out
        if row_count >= GGUF_Q6_STANDARD_PREFILL_SHARED4_MIN_ROWS
        and shape in GGUF_Q6_STANDARD_PREFILL_SHARED4_SHAPES
        else gguf_q6_k_t16_wmma_prefill_bf16_bf16_out
    )
    return fn(
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


def gguf_q6_k_t16_qmicro_planar_wmma_prefill_gfx1151_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    **kwargs,
):
    """Select shared-weight WMMA only for the admitted wide Q6 down shape.

    Rows17-144 use measured periodic low-VGPR/shared4 bands (bit-exact
    siblings; low-VGPR cuts 184 -> 88 VGPR). Rows145-255 keep plain, and the
    six physical shapes use shared4 from row256. Shape misses keep plain.
    """

    row_count = int(rows)
    shape = (int(in_features), int(out_features))
    if (
        row_count == GGUF_Q6_PLANAR_PREFILL_SHARED4R9_ROWS
        and shape == GGUF_Q6_PLANAR_PREFILL_SHARED4R9_SHAPE
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r9_bf16_bf16_out
    elif (
        GGUF_Q6_PLANAR_PREFILL_SHARED4R6_MIN_ROWS <= row_count
        <= GGUF_Q6_PLANAR_PREFILL_SHARED4R6_MAX_ROWS_BY_SHAPE.get(shape, -1)
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r6_bf16_bf16_out
    elif (
        GGUF_Q6_PREFILL_SHARED3R1_MIN_ROWS <= row_count
        <= GGUF_Q6_PREFILL_SHARED3R1_MAX_ROWS
        and shape in GGUF_Q6_PLANAR_PREFILL_SHARED3R1_SHAPES
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared3r1_bf16_bf16_out
    elif (
        row_count == GGUF_Q6_PLANAR_PREFILL_SHARED4R4_ROWS
        and shape in GGUF_Q6_PLANAR_PREFILL_SHARED4R4_SHAPES
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r4_bf16_bf16_out
    elif (
        row_count == GGUF_Q6_PLANAR_PREFILL_SHARED4R3_ROWS
        and shape in GGUF_Q6_PLANAR_PREFILL_SHARED4R3_SHAPES
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r3_bf16_bf16_out
    elif (
        17 <= row_count <= GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS
        and shape == (5_120, 17_408)
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS
        and shape in GGUF_Q6_PLANAR_LOWVGPR80_SHAPES
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS
        < row_count
        <= GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS
        < row_count
        <= GGUF_Q6_PLANAR_LOWVGPR96_MAX_ROWS
        and shape in GGUF_Q6_PLANAR_LOWVGPR96_SHAPES
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS
        < row_count
        <= GGUF_Q6_PLANAR_LOWVGPR96_MAX_ROWS
        and shape in GGUF_Q4_T16_DENSE_LOWM_SHAPES
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        GGUF_Q6_PLANAR_LOWVGPR96_MAX_ROWS
        < row_count
        <= GGUF_Q6_PLANAR_LOWVGPR128_MAX_ROWS
        and shape in GGUF_Q6_PLANAR_LOWVGPR128_SHAPES
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        GGUF_Q6_PLANAR_LOWVGPR96_MAX_ROWS
        < row_count
        <= GGUF_Q6_PLANAR_LOWVGPR128_MAX_ROWS
        and shape in GGUF_Q6_PLANAR_LOWVGPR48_128_SHAPES
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        GGUF_Q6_PLANAR_LOWVGPR96_MAX_ROWS
        < row_count
        <= GGUF_Q6_PLANAR_LOWVGPR128_MAX_ROWS
        and shape in GGUF_Q6_PLANAR_SHARED4_128_SHAPES
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out
    elif (
        GGUF_Q6_PLANAR_LOWVGPR128_MAX_ROWS
        < row_count
        <= GGUF_Q6_PLANAR_LOWVGPR144_MAX_ROWS
        and shape in GGUF_Q6_PLANAR_LOWVGPR48_144_SHAPES
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        GGUF_Q6_PLANAR_LOWVGPR128_MAX_ROWS
        < row_count
        <= GGUF_Q6_PLANAR_LOWVGPR144_MAX_ROWS
        and shape in GGUF_Q6_PLANAR_SHARED4_144_SHAPES
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out
    elif (
        row_count >= GGUF_Q6_PLANAR_PREFILL_SHARED4_MIN_ROWS
        and shape in GGUF_Q6_PLANAR_PREFILL_SHARED4_SHAPES
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out
    else:
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_bf16_bf16_out
    return fn(
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )


# Clean AR-O2 three-repeat category/quality gates admit compensated source-F16
# WMMA only for SWA QKV/gate/O from 16 rows. Full-attention layers and M2-15
# retain the exact LPF-1 tile; decode retains the separately registered GEMV.
LAGUNA_F16_PREFILL_STRATEGY = "wmma_comp_swa"
LAGUNA_F16_PREFILL_MIN_ROWS = 16
# Exact rows==1 source-F16 single/triple siblings keep the local256 grid and
# reduction order while removing the generic reducer's second broadcast
# barrier. All six natural roles improve at the gfx1151 leaf; exact production
# A/B and cache-only tracing admit automatic selection.
LAGUNA_F16_DECODE_ONEBARRIER = True
# Compile-time K3072/K6144/K9216 specializations retain the one-barrier
# arithmetic/grid while removing dynamic loop/address machinery. Seven exact
# p512/d128 pairs and cache-only tracing admit them over the generic
# one-barrier owner; the latter remains the explicit rollback.
LAGUNA_F16_DECODE_FIXEDK = True
# Exact c=1 source-F16 Q/K/V/gate launch contraction preserves every
# output column's fixed-K reduction order. Seven same-resident p512/d128 pairs
# are exact and all positive; triple-plus-single remains the unfused fallback.
LAGUNA_F16_ATTENTION_QUAD_DECODE = True
# Exact last-producer projection/head/KV fusion keeps every fixed-K dot and
# head RMSNorm/RoPE association, removes another 48 launches/token, and is
# byte-exact on global/SWA state. The seven-pair gate is throughput-flat but
# mechanically positive; the quad plus registered head/KV remains rollback.
LAGUNA_F16_PROJECTION_HEAD_KV_DECODE = True
# Exact last-producer attention-output fusion preserves the fixed-K projection
# and standalone local256 add/RMSNorm trees while removing 48 launches/token.
# Seven exact same-resident pairs are all positive.
LAGUNA_F16_OUTPUT_ADD_RMSNORM_DECODE = True
# Exact source-F16 decode streams each weight once. Cache-bypassing loads cut
# the complete six-role leaf family 3.36%, and all seven same-resident
# p512/d128 pairs improve without changing arithmetic or resident bytes.
# Explicit false and peer backends retain the cached-load owner.
LAGUNA_F16_NONTEMPORAL_DECODE = True
# The greedy argmax winner also owns the next embedding token and scratch/KV
# positions. The following exact autoregressive step validates the host token
# and consumes those device-published scalars without three synchronous H2D
# copies; forced-token and peer-backend paths retain ordinary publication.
LAGUNA_ARGMAX_CONTROL_PUBLISH = True
# Exact Q6T16 LM-head producer tile maxima remove the full-logit argmax scan
# and one model launch. The actual-weight leaf, seven-pair p512/d128 gate, and
# cache-only trace admit it on gfx1151; constructor false keeps exact rollback.
LAGUNA_Q6_T16_LM_HEAD_TOP1_STAGE1 = True
# Exact c=1 router projection wave-level reduction. Seven same-resident
# p512/d128 pairs are exact and win 6/7; the scalar local256 tree remains the
# registered rollback.
LAGUNA_ROUTER_PROJECTION_WAVE0_TREE = True
# gfx11 any-order dispatch lets the exact router projection wait on the ordered
# output/add/RMSNorm producer without holding the queue at that boundary.
LAGUNA_OUTPUT_ROUTER_ANYORDER_DECODE_AVAILABLE = True
LAGUNA_OUTPUT_ROUTER_ANYORDER_DECODE = True
# Exact D9 wave-0 RMS reduction preserves the local256 partials and complete
# FP32 addition tree while removing seven workgroup barriers. The gfx1151
# production-shape leaf improves 3.66%; the scalar tree remains rollback.
LAGUNA_MOE_TAIL_WAVE0_TREE = True
# Exact K3072/N1024 gate/up and K1024/N3072 down siblings preserve the
# production local128 grid and reduction order while compile-time-specializing
# only Laguna's c=1/top-10 shape. All three actual-weight roles improve, and
# seven exact p512/d128 pairs admit the combined owner.
LAGUNA_SELECTED_NATURAL_DECODE = True
# The exact selected-down sibling distributes the final 16 ordered wave sums
# across lanes 0..15. All seven resident p512/d128 pairs are exact and
# positive; the serial owner remains registered rollback.
LAGUNA_SELECTED_DOWN_NATURAL_PARALLEL_DECODE = True
# Exact tile-local completion preserves all ten route-parallel weight streams,
# every BF16 projection boundary, and the slot-order weighted FMA chain while
# removing the standalone reducer launch. Seven same-resident pairs all win.
LAGUNA_SELECTED_DOWN_NATURAL_PARALLEL_WEIGHTED_DECODE = True
# Reuse each adjacent Q4 selected-down nibble byte and aligned coefficient
# pair. Exact p512/d128 wins 5/7 pairs at +0.1365%; explicit false retains the
# preceding scalar-payload weighted route for rollback.
LAGUNA_SELECTED_DOWN_Q4_PAIRCOEFF_WEIGHTED_DECODE = True
# Preserve the production Q4/Q6 shared-down grids and D9 local256 tree while
# enqueueing both unchanged launch wrappers inside one native host call. Seven
# exact same-resident p512/d128 pairs win 7/7; peer backends retain separate
# Python calls until independently measured.
LAGUNA_SHARED_DOWN_MOE_TAIL_HOST_BATCH = True
# Exact gate/up owner that splits each resident T16 tile across two 8-column
# workgroups, halving live accumulators. The actual-weight leaf improves
# 5.35-7.13%; seven exact p512/d128 pairs are all positive.
LAGUNA_SELECTED_NATURAL_TILE8_DECODE = True
# The exact tile8 parallel-tail sibling preserves every column's arithmetic.
# All seven resident p512/d128 pairs are exact and positive.
LAGUNA_SELECTED_NATURAL_TILE8_PARALLEL_DECODE = True
# Promoted exact fusion: the qualified parallel tile8 owner materializes the
# BF16 SiLU intermediate directly; all seven resident pairs are positive.
LAGUNA_SELECTED_NATURAL_TILE8_PARALLEL_SILU_DECODE = True
# Convert BF16 activations and dequantized Q4 gate/up pairs to FP16 in
# registers, then use gfx1151's native adjacent-K dot2 with FP32 accumulation.
# The byte-neutral interleaved resident layout is unchanged. The actual-weight
# leaf improves 17.99%, all seven p512/d128 pairs improve, and recurrent
# candidate-vs-exact quality stays at max KL 0.00820 / top-1 93.75%.
# Constructor false retains the exact pair-coefficient owner.
LAGUNA_SELECTED_HALFDOT_DECODE = True
# Exact dense/shared Q4 pair fusion preserves the two BF16 projection
# boundaries in registers before applying the existing SiLU-product
# expression. Seven resident p512/d128 pairs are exact and all positive.
LAGUNA_Q4_PACK8_DUAL_SILU_DECODE = True
# Exact decode-only T16 consumers stream compact Q4 coefficients beside the
# retained pack8 prefill layout. Actual shared/dense leaves are BF16 bit-exact
# and improve 19.0%/60.2%; admission accounts for the bounded auxiliary bytes.
LAGUNA_Q4_DENSE_DECODE_T16_SIDECAR = True
# Exact byte-neutral gate/up interleave plus two-column wave ownership cuts
# the actual dense/shared leaf family by 6.50% before the resident gate.
LAGUNA_Q4_DENSE_DECODE_T16_DUAL_INTERLEAVED = True
# Exact standalone Q4T16 shared-down owner streams 22.9% fewer resident bytes
# than expanded pack8 and is bit-identical across all 24 actual matrices.
LAGUNA_Q4_SHARED_DOWN_T16_DECODE = True
# The selected gate/up decoder and the production D8 prefill owner now share
# one exact byte-neutral paired T16 allocation. Natural prefill leaves improve
# at M55+ and are effectively flat at M32; c=1 decode improves 5.70%.
LAGUNA_Q4_EXPERT_T16_DUAL_INTERLEAVED = True
# Clean post-350 repeated M512/M1024/M2048 timing and full-logit quality admit
# 2048-row projection/MoE transactions while attention and physical KV writes
# remain independently tiled at 128. M2048 is byte-identical at pp512, keeps
# top-1 at 512/1K/4K, and has max relative KL 1.25e-5 versus M512.
# Other backends retain the 128-row runtime fallback until measured independently.
LAGUNA_PREFILL_MATRIX_ROWS = 2048
# The post-350 LAP-7 screen reuses each streamed BF16 K/V row across four
# adjacent queries. It is byte-identical to the admitted online-qrow2 arithmetic
# on the wrap/eviction oracle and improves matched pp512 production by 3.23%.
# Qrow2/exact variants remain explicit rollback; unmeasured backends are unchanged.
LAGUNA_GLOBAL_PREFILL_VARIANT = "global_context_rows_qrow4_m128_online_spans"
LAGUNA_SWA_PREFILL_VARIANT = "swa_context_rows_qrow4_m128_online_spans"
# WPF-H5M source-qualified exact qrow4 ownership is W7900-only pending an
# independent gfx1151 exactness/resource/performance gate.
LAGUNA_SWA_PREFILL_ROLE_VARIANTS = {}
# Exact pre-append scheduling lets complete M128 global tiles and pre-wrap SWA
# tiles consume one BF16 cache source. Wrapped SWA, residual rows, verifier
# transactions, and other backends retain attend-then-append.
LAGUNA_PREFILL_KV_PREAPPEND = True
# Once a safe tile is pre-appended, complete KVLiveSpans metadata can decide
# visibility without selecting between current-row and cache sources inside
# the dot-product loop. Measured admission keeps the slower global start-0
# slice on the source-qualified cached kernel while enabling metadata-only SWA
# and global tiles beginning at position 128.
LAGUNA_PREFILL_CACHED_META = True
# Exact global-only qrow6 reuses each streamed BF16 K/V row across six adjacent
# queries. Leaf admission is limited to complete preappended global M128 tiles
# beginning at position 128; global start 0 and every SWA tile retain qrow4.
LAGUNA_PREFILL_GLOBAL_QROW6 = True
# Complete initial no-wrap preappended tiles have identity token positions and
# no eviction. The separately registered dense-initial attention variants
# preserve the full KVLiveSpans ABI while skipping per-token metadata loads.
# Partial, wrapped, explicitly evicted, and verifier routes retain exact
# cached-metadata/current-source fallbacks.
LAGUNA_PREFILL_DENSE_INITIAL = True
# Dense-initial M128 tiles beginning at position 128 widen the resident BF16
# K/V prefix exactly once, then use zero-workspace F32 hipBLASLt QK/PV
# contractions around a KVLiveSpans-qualified causal softmax. The complete
# pp512 route wins 6/7 paired runs (602.52 versus 576.08 tok/s median), keeps
# top-1 2930, and lowers all-exact full-logit KL from 0.003246 to 0.002214.
# Start-0, partial, wrapped, evicted, verifier, and decode routes stay on the
# established attention kernels.
LAGUNA_PREFILL_ATTENTION_HIPBLASLT = True
# Packing the 4.7-MB query/output tile into head-major order allows one
# eight-way wide QK and one wide PV batch without replicating K/V. It improves
# the qualified 48-layer leaf model 5.08% and the seven-pair pp512 median
# 0.87%, while all-exact KL improves from 0.002214 to 0.002097.
LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERIES = True
# Write the three qualified M128 query tiles directly in head-major order from
# the fused RMSNorm/RoPE producer. This removes 144 standalone query-transpose
# launches at pp512; eleven complete-state pairs improve the median 0.532% and
# every token/logit/hidden/KV hash remains exact.
LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERY_PRODUCER = True
# One wave32 owns one causal-score row, replacing the former 256-thread
# block reduction and its LDS barriers. The qualified 48-layer attention leaf
# improves 13.72%; paired pp512 improves 0.574% and all-exact KL falls from
# 0.002097 to 0.001796.
LAGUNA_PREFILL_ATTENTION_HIPBLASLT_WAVE_ROWS_SOFTMAX = True
# Keep the three qualified library-attention output tiles in their native
# head-major order and consume that mixed layout directly in the exact
# softplus gate. This removes 144 standalone output-transpose launches at
# pp512. Eleven complete-state pairs improve the median 0.338%; the stronger
# admission signal is the exact removal of the traced transpose sub-window.
LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_OUTPUT_GATE = True
# Contexts above 4K route only the 12 global-attention layers through a
# capacity-sized 48-head packed-F32 owner. Same-session complete-model gates
# preserve the 4K path and improve 16K/64K/128K by 7.93%/16.94%/22.09%;
# SWA, decode, partial, wrapped, evicted, and verifier paths remain unchanged.
LAGUNA_PREFILL_LONG_ATTENTION_HIPBLASLT = True
# Exact 4K key blocks carry online row max/sum/output state across tensorized
# QK/PV calls. This cuts 128K scratch 4.298 GB -> 143.753 MB and improves the
# complete-model long route another 12.52% while preserving its >4K gate.
LAGUNA_PREFILL_BLOCK_ATTENTION_HIPBLASLT = True
# Dense-initial global cache blocks are allocated in identity physical order.
# Direct addressing removes per-element span checks/remaps from the exact 4K
# BF16 widen; the full KVLiveSpans route remains the rollback/fallback.
LAGUNA_PREFILL_DENSE_CONTIGUOUS_CACHE = True
# Reuse each exact 4K global K/V block across a complete M2048 matrix chunk.
# SWA and partial matrix tails remain on the independently retained M128
# routes. LC-3 complete-model gates improve 4K/16K/64K/128K by
# 4.89%/20.48%/39.11%/44.59%.
LAGUNA_PREFILL_GLOBAL_ATTENTION_ROWS = 2_048
# Rolling M128 SWA gathers the exact 511 historical BF16 ring rows plus 128
# current BF16-rounded rows into one 639-key tensorized QK/PV union. The
# complete 4K/16K/64K/128K gate improves every shape and remains bounded.
LAGUNA_PREFILL_SWA_ATTENTION_HIPBLASLT = True
# Clean LAP-3/LAP-4 full-category admission quantizes gate/up in same-byte
# 16-value groups and uses the resident-T16 128x32 integer-dot consumer.
# The post-350 wave-column screen keeps row-vector D8 activation staging, maps
# one 32-column output slice to each wave, holds decoded T16 weights in
# registers, and ping-pongs the activation tile to remove one barrier per K32.
# For chunks of at least 512 rows, prefetch the next K32's eight raw nibble
# words while current packed dots execute; smaller chunks keep the rollback.
# Packed-dot arithmetic and K order remain bit-for-bit unchanged.
# Other backends retain exact.
LAGUNA_SELECTED_GATE_UP_MODE = (
    "mmq128x32_d8_f32_wavecols_direct_doublebuf_rawprefetch_ge512"
)
# Exact eight-token router tiling preserves every token/expert's K traversal
# and reduction tree while reusing each F32 weight row twice as long.
LAGUNA_ROUTER_LOGITS_MODE = "token_tile_8"
# The post-350 down screen maps Q4 output columns across two wave32s and lets
# the Q6 row-vector consumer reuse one decoded tile across 64 routed rows.
# Range-safe D4 resident-T16 integer-dot arithmetic is unchanged; 32-row Q6,
# scalar-staged, and exact routes remain rollbacks. At producer rows >=512,
# Q4 down also carries the next K32 raw nibble payload in registers.
LAGUNA_SELECTED_DOWN_MODE = (
    "mmq64x64_d4_f32_q6_wavecols_direct_rawprefetch_q4_ge512"
)
# Exact scratch reuse writes packed gate/up into the larger selected-down
# output allocation, then folds the standalone BF16 SiLU boundary into the
# range-safe down pack. Seven paired pp512 runs are exact and win 7/7; the
# standalone SiLU plus ordinary pack remain the explicit rollback chain.
LAGUNA_FUSED_SELECTED_SILU_PACK = True
# Byte-neutral Q6 qmicro keeps the resident T16 metadata but groups each
# four-column K4 quant quartet into one aligned 12-byte record. Exact c1 and
# selected-prefill gates both improve on gfx1151; peer backends retain legacy
# Q6 T16 bytes until independently measured.
LAGUNA_Q6_QMICRO = True
# Q6 has no minimum term, so selected-down never consumes the Q8_1 activation
# sum metadata. The compact activation tile also narrows each bounded K16
# quant sum to int16, reducing the production 64-row kernel's LDS footprint
# from 5,632 to 5,120 bytes without changing dot or accumulation order.
LAGUNA_Q6_COMPACT_ACTIVATION = True
# Split each compact Q6 activation row across two threads so all 128 threads
# stage one 16-byte half and one K16 quant sum. The exact all-layer screen
# improves 21/23 real Q6 layers without changing resources or output bytes.
LAGUNA_Q6_HALF_ROW_ACTIVATION = True
# Padded rows are never consumed by the guarded dot/store loops. Avoid writing
# zero Q8 bytes and recomputing zero K16 sums for those slots.
LAGUNA_Q6_SKIP_PADDED_ACTIVATION = True
# Two byte-permute gathers replace the scalar four-column qmicro unpack while
# preserving the byte-neutral resident layout and exact integer-dot order.
# The actual-weight leaf improves 2.67%, the complete model wins 5/7 matched
# pairs, and cached tracing reduces the selected Q6 body with no spills.
LAGUNA_Q6_QMICRO_PERMUTE = True
# Byte-neutral planar qmicro removes two prefill byte gathers and lowers exact
# decode register pressure. The actual leaf wins, while two owner-order
# full-model blocks are aggregate-neutral with complete state exact.
LAGUNA_Q6_QMICRO_PLANAR = True
# Prefetch the next planar-qmicro weight record into registers while the
# current integer-WMMA fragment consumes LDS, then recycle the same 5,120-byte
# shared tile. This preserves every Q6 dot/FP32 boundary and adds no sidecar.
LAGUNA_Q6_WMMA_PREFETCH_WEIGHT = True
# Pipeline the next compact Q8 activation half-row beside the retained next
# weight record. The current K32 WMMA hides the global read, and the following
# iteration publishes the exact bytes into the unchanged shared activation
# tile. The complete-state pp512 A/B is exact and improves the median.
LAGUNA_Q6_WMMA_PREFETCH_ACTIVATION = True
# Reuse Q6's otherwise-unused D4 sum field for two exact K16 quant sums.
# Computing them once in the packer removes repeated sum dots from every
# selected-down output-column workgroup without changing activation bytes.
LAGUNA_Q6_PRECOMPUTED_ACTIVATION_SUMS = True
# D8 gate/up stores exact K16 activation sums in a bounded scratch sidecar so
# each output-column workgroup can skip rebuilding them with integer dots.
# Resident weights, D8 bytes/scales, arithmetic, and BF16 output are unchanged.
LAGUNA_Q4_PRECOMPUTED_ACTIVATION_SUMS = True
# Stable expert-major count/prefix/scatter uses one workgroup per expert instead
# of one workgroup serially scanning all 5,120 routed lanes twice.
LAGUNA_MOE_GROUP_COMPACT_MODE = "parallel"
# The always-on shared expert is independent of router selection and routed
# gate/up/down until the final combine. A nonblocking secondary stream plus
# two dependency events overlaps 99.16% of its measured pp512 kernel time;
# complete-state A/B is exact and wins all seven queue-matched pairs.
LAGUNA_MOE_BRANCH_CONCURRENCY = True
# The same exact event schedule is profitable at c=1 once the specialized
# T16 selected and shared decode kernels are active. Seven counterbalanced
# p512/d128 pairs win 7/7, and tracing observes 94 secondary kernels/token.
LAGUNA_MOE_DECODE_BRANCH_CONCURRENCY = True
# Protect router logits/selection before releasing the concurrent shared
# branch. Matched complete-state pp512 is +0.073% with 5/7 wins, and cached
# tracing verifies a 0.310-ms kernel-span reduction.
LAGUNA_MOE_SHARED_AFTER_ROUTER = True
# gfx1151 exposes least/greatest HIP stream priorities +1/-1. Running the
# after-router shared branch at +1 improves exact pp512 0.494% (6/7 wins) and
# cuts cached kernel span 7.255 ms while keeping 99.75% of shared work hidden.
LAGUNA_MOE_SHARED_LOW_PRIORITY = True
# Decode benefits from equal scheduling priority after its exact selected and
# shared paths became closely balanced. Keep prefill at +1, but use a separate
# priority-0 shared stream for c=1; the same-session gate wins all seven pairs.
LAGUNA_MOE_DECODE_SHARED_NORMAL_PRIORITY = True
# Clean LAP-5 admission selects resident pack8-Q4/raw-Q6 64x16 WMMA consumers
# for dense/shared rows while preserving the exact low-row fallback.
LAGUNA_DENSE_Q4_PREFILL_MODE = "wmma_pack8"
# D08-X (2026-08-15): on Qwen3.5-0.8B dense-FFN pack8 shapes the registered
# pack8 WMMA bulk consumer measures 1.97x ([3584,1024], 16x32) and 2.33x
# ([1024,3584], 64x16) versus the exact tile8x8 leaf at rows=512, within one
# BF16 ULP. The complete p512 A/B also exercised the attention pack8 shapes.
# Fail closed beyond that measured row/shape matrix until it is expanded.
GGUF_Q4_PACK8_WMMA_BULK_PREFILL = True
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
    "gguf_q6_k_t16_qmicro_planar_v1": 3,
    "bf16": 512,
}
# The attention-RMSNorm source range is statically bounded from resident F32
# norm weights, so Q/K/V/gate use direct BF16-to-FP16 and omit identity output
# restores. Attention output retains power-of-two row scaling; decode is
# unchanged.
LAGUNA_F16_PREFILL_MODE = "hipblaslt_range_direct"
# Exact producer-boundary variants write FP16(BF16(value)) directly from the
# attention RMSNorm and softplus-gate kernels. This removes the two standalone
# BF16-to-FP16 casts per layer while preserving the established source-F16
# input bits; the runtime setter remains the explicit rollback.
LAGUNA_F16_BOUNDARY_FUSION = True
# The gfx1100 current-P4 body is shape-identical for Laguna S 2.1 and compiles
# from the shared gfx11 source as a native gfx1151 code object. The
# architecture-local bit-exact and p512/d128 gates admit automatic selection;
# explicit False retains the registered fallback chain for rollback.
LAGUNA_HEAD_KV_FUSION = True
# The complete gfx11 exact split-attention bundle wins the clean gfx1151
# p512/d128 gate. Keep the thresholds and reducer capabilities inseparable:
# explicit use_split_attention=False retains serial global/SWA attention.
LAGUNA_GLOBAL_SPLIT_MIN_LIVE = 127
# The exact natural 48Q/8KV/D128/capacity-4096 reducer preserves the retained
# dynamic-live score ABI and local256 arithmetic. Three production live points
# and seven exact p512/d128 pairs admit it on gfx1151 only.
LAGUNA_GLOBAL_SPLIT_FIXEDSHAPE_REDUCE = True
# Above the fused kernel's LDS-resident score limit, replace the per-query score
# producer and scalar sixfold-V reducer with an exact GQA6 producer, exp32
# normalizer, and dimension-sharded shared-V owner. The live16,448 leaf is
# byte-exact and 80.94% faster than the generic split path.
LAGUNA_GLOBAL_SPLIT_GQA6_DIM32_VSTAGE64 = True
# Defer the exact normalization product into each shared-probability load,
# removing one full score-plane pass without changing its F32 bits.
LAGUNA_GLOBAL_SPLIT_GQA6_DEFERREDNORM_DIM32_VSTAGE64 = True
# Keep six query vectors resident while one exact wave scores four consecutive
# KV tokens. The natural-depth leaf is byte-exact and 8.6-9.6% faster from
# 4K through 128K on gfx1151.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE64 = True
# V80 keeps the same chronological FMA chain while reducing exact-PV staging
# barriers by 20%. It wins all four natural depths; V92/V96 lose at long
# context, so the V64 specialization remains the explicit rollback.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80 = True
# Dense initial caches do not need the physical-slot plane in the exact V80
# owner. Preserve the complete spans ABI and fall back after explicit eviction.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX = True
# Keep the old cache-policy crossover for the prefetch4 rollback. Deeper
# operand-prefetch owners hide the bypass latency and select combined
# non-temporal K/V across the dense V80 band above the 6,000-live fused owner.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX_NONTEMPORAL_MIN_LIVE = 65_536
# Once the long-context cache bypass is active, bypass K as well as V. The
# incremental exact leaf wins 3.26-3.58% at the admitted 64K/128K shapes.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE = True
# Deeper ordered operand prefetch becomes a 6.84-6.94% exact leaf win after
# non-temporal K/V increases the latency each output wave must cover.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_PREFETCH8_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE = True
# Prefetch16 preserves the chronological FMA chain and wins a further
# 1.69-2.25% over prefetch8 from 4K through 128K on gfx1151.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE = True
# V128 keeps two 512-thread workgroups resident while reducing exact-PV stage
# rounds by 37.5%. It wins 2.15-2.37% over V80 at every natural long depth;
# V160 loses the gain at 64K/128K and is not retained.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE128_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE = True
# Vectorize contiguous FP32 probability loads/stores into the shared V128
# stage. This remains byte-exact and improves every natural long depth.
LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE128_PROBABILITY_VEC4_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE = True
LAGUNA_GLOBAL_SPLIT_GQA6_ALLWAVE_TILE1024_DENSE_PREFIX = True
LAGUNA_GLOBAL_SPLIT_GQA6_ALLWAVE_TILE1024_DENSE_PREFIX_MIN_LIVE = 32_768
# Ordered-prefetch4 moved the exact/context-PV crossover above 64K. Keep the
# quality-scoped layer schedule only at the measured deep-context band; shorter
# long contexts use exact deferred-normalization GQA6.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_MIN_LAYER = 32
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_COMPENSATED_LAYER = 28
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_MIN_LIVE = 98_304
# Share each staged probability/value tile across two output-dimension waves.
# D64 is byte-identical to D32 and reduces the active long-context leaf by
# 17.3%/10.7%/10.9% at 16K/64K/128K on gfx1151.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DIM_TILE = 64
# The quality-scoped D64 context-split owners reuse the same exact score plane
# as the V80 path. Four-token ownership cuts their score grid by 4x while the
# admitted partial-PV/merge arithmetic remains unchanged.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_TOKENLOOP4 = True
# Keep the exact exp32 reciprocal but apply it in the context-PV probability
# loader, avoiding one normalized-score write/read pass.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DEFERREDNORM = True
# Dense initial caches have identity logical-to-physical metadata. Let the
# long-context token-loop owner bypass those metadata streams while retaining
# the full KVLiveSpans ABI and the generic fallback after any explicit eviction.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_SCORE = True
# The deep ctx4096 route streams each K/V line once. Bypass the cache only for
# that dense-prefix owner; the formal 128K leaf wins 8.47% ordinary and 6.13%
# compensated without changing F32 or BF16 output bits.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE = True
# Ordinary PV reaches its 128K minimum with four ordered operands in flight;
# deeper prefetch is tied but consumes more registers.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE_PREFETCH = 4
# Kahan PV has a longer dependent chain and continues improving through p16.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_COMPENSATED_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE_PREFETCH = 16
# Reuse the exact 1,024-token all-wave QK owner in the five quality-scoped
# ctx4096 layers. The partial-PV and merge association remain unchanged; the
# serial formal leaf improves 128K by 2.16% ordinary / 2.07% compensated.
LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_ALLWAVE_SCORE = True
# The exact dynamic-scan fused one-head owner keeps all 48 workgroups and
# removes the score plane/launch while preserving reduction association.
LAGUNA_GLOBAL_FUSED_FIXEDSHAPE = True
# Pair adjacent global query heads and reuse each staged 64-slot V tile.
# The exact natural-live leaf and seven resident-model pairs admit it.
LAGUNA_GLOBAL_GQA2_VSTAGE64_FIXEDSHAPE = True
# Preserve global GQA2 arithmetic while widening each padded V-stage copy to
# one aligned 16-byte transaction.
LAGUNA_GLOBAL_GQA2_VSTAGE64_VEC16_FIXEDSHAPE = True
# Avoid the compiler-generated 32-byte private scratch aggregate used by the
# retained vec16 copy and write each valid vector directly into the V tile.
LAGUNA_GLOBAL_GQA2_VSTAGE64_VEC16_DIRECT_FIXEDSHAPE = True
# Exact score-domain sibling passes the seven-pair resident-model wall gate.
LAGUNA_GLOBAL_GQA2_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Issue each wave's independent exact global-softmax exponentials across
# wave32 while retaining lane-0 token-order summation.
LAGUNA_GLOBAL_GQA2_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Raise the exact global-attention grid from 24 to 32 workgroups by assigning
# each 6-query GQA group as 2+2+1+1 owners. Singleton-owner idle waves retain
# staged-V barrier participation while active heads preserve every operation.
LAGUNA_GLOBAL_MIXED32_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Exact score-producer maxima remove the global score reread and one barrier.
# Seven resident p512/d128 pairs admit the qualified gfx1151 route.
LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Preserve the producer-max QK tree while replacing ds_bpermute transport with
# permlanex16 plus DPP moves. Seven resident pairs admit the exact sibling.
LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Exact aligned float4 replay of the normalized global probability plane.
# All seven resident p512/d128 pairs win with exact generated state.
LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Normalize each global probability once in LDS before the exact PV replay.
# Seven resident p512/d128 pairs admit the exact gfx1151 sibling.
LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Preserve the local256 eight-wave denominator tree while all sixteen waves
# share independent QK and value transport. Every natural global leaf wins.
LAGUNA_GLOBAL_MIXED32_LOCAL512_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Use all 40 gfx1151 CUs by assigning each six-query GQA group as
# 2+1+1+1+1 exact owners. All three natural global leaves and all seven
# resident p512/d128 model pairs win against mixed32-local512.
LAGUNA_GLOBAL_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE = True
# Sequential global decode before an explicit eviction has a visible identity
# prefix. Compile out token/base/eviction metadata and the physical-slot LDS
# plane while preserving the exact mixed40 arithmetic and the full span ABI.
# All natural leaves and seven resident p512/d128 pairs win byte-exactly.
LAGUNA_GLOBAL_DENSE_PREFIX = True
# Overlap the next dense-prefix global V64 load with exact scalar PV work by
# assigning it to otherwise-idle query-owner waves. All natural leaves and
# seven resident p512/d128 pairs win byte-exactly.
LAGUNA_GLOBAL_DENSE_PREFIX_IDLE_DOUBLE_BUFFER = True
# Preserve the eight-wave denominator while widening dense-prefix QK/value
# transport. All seven exact same-resident p512/d128 candidate runs win.
LAGUNA_GLOBAL_LOCAL1024 = True
LAGUNA_SWA_SPLIT_MIN_LIVE = 65
LAGUNA_SWA_SPLIT_TILE16_MIN_LIVE = 257
LAGUNA_SPLIT_GATE_FUSION = True
LAGUNA_SWA_SPLIT_WAVE_LOCAL = True
# Exact GQA3 score ownership reuses each streamed SWA K row across three query
# heads while retaining the 72-head score plane and wave-local value reducer.
# The 24 x token/tile grid preserves gfx1151 breadth; peer backends keep the
# one-query score owners until independently measured.
LAGUNA_SWA_SPLIT_GQA3_SCORES = True
# The saturated 512-slot SWA reducer preserves the retained 72-workgroup /
# 288-wave grid and every scalar/FMA operation while specializing the natural
# 72Q/8KV/D128 ring. Exact leaf and seven-pair production gates admit it.
LAGUNA_SWA_SPLIT_FIXED512_REDUCE = True
# The exact local256 GQA2 fused owner keeps 320 waves, reuses each K row across
# adjacent query heads, and removes the global score plane. Seven resident
# p512/d128 pairs admit it only at the saturated natural 512-slot shape.
LAGUNA_SWA_FUSED_FIXED512 = True
# The exact local384 GQA3 sibling keeps all 288 query/dimension waves active
# while reducing saturated K-cache owners per KV head from five to three.
# Seven resident p512/d128 pairs admit it only for the natural gfx1151 shape.
LAGUNA_SWA_GQA3_LOCAL384_FIXED512 = True
# The exact local384 sibling stages 64 contiguous V rows in LDS and reuses
# each load across the three owned query heads. The seven-pair p512/d128 gate
# is bit-identical and promotes it only at the saturated natural shape.
LAGUNA_SWA_GQA3_VSTAGE64_FIXED512 = True
# Replace scalar BF16 staging copies with aligned 16-byte transactions while
# preserving the retained local384 compute and every output operation.
LAGUNA_SWA_GQA3_VSTAGE64_VEC16_FIXED512 = True
# Avoid the compiler-generated per-thread LDS aggregate used by the retained
# vec16 copy and write each valid vector directly into the real V tile.
LAGUNA_SWA_GQA3_VSTAGE64_VEC16_DIRECT_FIXED512 = True
# The exact compiler-expf sibling exposes the finite non-positive
# score-minus-maximum domain and removes generic exponential guard work.
LAGUNA_SWA_GQA3_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = True
# Balance each KV head's nine queries as 2+2+2+3 across 32 local384 blocks.
# Exact seven-pair resident decode admits the one-phase mixed owner.
LAGUNA_SWA_MIXED32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = True
# Issue each four-slot exact softmax batch across lanes 0..3, then shuffle the
# weights back into the unchanged ordered denominator/PV chains. Seven
# resident pairs admit the resource-neutral sibling at saturated SWA512.
LAGUNA_SWA_MIXED32_EXP4_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = True
# Extend the same exact lane-parallel issue schedule to eight softmax weights.
# The leaf, cached trace, and all seven resident pairs improve without a
# resource or arithmetic-order change.
LAGUNA_SWA_MIXED32_EXP8_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = True
# Extend the exact lane-parallel schedule to sixteen softmax weights. The
# default remains subject to the resident p512/d128 gate.
LAGUNA_SWA_MIXED32_EXP16_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = True
# Complete the bounded issue-width screen with one exact softmax weight per
# wave32 lane. Resident decode decides whether this becomes the final owner.
LAGUNA_SWA_MIXED32_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = True
# Exact score-producer partial maxima remove four redundant 512-score scans
# per query. Seven exact resident p512/d128 pairs admit the specialization.
LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Compute each owned query's softplus gate once. All seven byte-exact resident
# p512/d128 pairs improve, so gfx1151 promotes the specialization.
LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Reuse each exact softmax weight across all four V-output waves through the
# V-stage publication barrier already paid by production. Seven exact resident
# pairs improve with unchanged VGPRs, so gfx1151 promotes the specialization.
LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Replay the published K64 probability tile through sixteen aligned float4 LDS
# reads while preserving the 64 ordered denominator adds. All seven exact
# resident p512/d128 pairs improve at unchanged kernel resources.
LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VEC4_DENOM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Read each published K64 probability row through sixteen aligned float4 LDS
# vectors while preserving the 64 ordered PV FMAs. All seven resident pairs
# improve at unchanged kernel resources.
LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# On pair-owner blocks, move the unchanged vectorized denominator replay onto
# idle waves 8/9 so all eight active output waves can execute PV concurrently.
# Seven exact resident p512/d128 pairs improve with complete separation.
LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Fill all 40 gfx1151 CUs with one 2+2+2+2+1 owner grid. Seven exact resident
# pairs improve with complete separation despite 25% more K/V-owner traffic.
LAGUNA_SWA_MIXED40_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Separate mixed40 tail exp producers from idle denominator and active PV
# waves. Six of seven exact resident pairs improve and the sole loss is
# smaller than the median paired gain.
LAGUNA_SWA_MIXED40_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Raise the exact mixed40 workgroup from 12 to 16 wave32s while retaining all
# 40 owners. All seven resident p512/d128 pairs improve with identical
# 128-token trajectories; the kernel also drops from 104 to 32 VGPRs.
LAGUNA_SWA_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Let the two exact tail-probability waves copy the final 64 staged-V vectors.
# The local512 combination wins all seven resident p512/d128 pairs while
# preserving the complete generated trajectory and allocation lifecycle.
LAGUNA_SWA_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_VALUE_TAIL_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512 = (
    True
)
# Replace the retained wave32 QK shuffle transport with the association-
# identical permlanex16/DPP sequence inside the final local512/V128 tile.
# The leaf improves 5.35% and all seven resident p512/d128 pairs win.
LAGUNA_SWA_OUTPUT_SHARDED_PROBABILITY_DPP_QK = True
# Saturated sequential SWA has an identity physical ring with every slot
# visible. The exact dense-ring sibling compiles out token/base/eviction
# metadata traffic and its 2-KiB LDS physical-slot plane. Explicit eviction,
# pre-saturation, and non-standard states retain the generic DPP owner.
# The byte-exact leaf improves 25.55% and all seven resident pairs win.
LAGUNA_SWA_DENSE_RING = True
# gfx1151 exposes 32 wave32 slots and 1024 work-items per CU. The exact
# local1024 dense-ring owner fills that natural limit while preserving all 40
# workgroups; all seven same-resident p512/d128 pairs improve.
LAGUNA_SWA_LOCAL1024 = True
# Clean SOL-G5 p512/d128 evidence admits the state-bound composite GGUF graph
# only when at least 128 decode transitions amortize capture/instantiate/close.
GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS = 128
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
# Default AOTriton ON for gfx1151. An earlier 2026-08-20 slice measurement
# (64..2048) claimed native causal_gqa_gate_bf16 was ~2-5% faster with no
# crossover, which set this to False; but the native full-attention path scales
# badly in bulk prefill and collapses at mid/long context. Measured 2026-08-21
# (35B-A3B @ 8060S): native 1K=828, 2K=362, 4K=234 tok/s (and can hang/fail at
# 2K+) while AOTriton holds flat ~1300-1330 tok/s at 1K/2K/4K (1274 at 512).
# Below the 512-token crossover native is still used regardless. This restores
# the retained 4K prefill (~1430 tok/s) that predated the False default.
GGUF_AOTRITON_PREFILL = True
# Measured 2026-08-20 (35B-A3B @ 2048/4096, 8060S): 512-row linear/MoE prefill
# chunks are ~1.2% (2048 tok) / ~2.3% (4096 tok) faster than the 1024 default,
# and chunk-boundary-correct (KL 0.00013 at 2048, far below the 0.034 run-noise
# floor). The 27B dense (H5120) is inconclusive within 60W-lane variance, so the
# override is keyed to the H2048-MoE geometry only. Smaller chunks reduce
# transient scratch and improve pipelining on the 40-CU APU.
GGUF_PREFILL_CHUNK_SIZES_BY_GEOMETRY = {
    (QWEN35_MOE_H2048_E256_GEOMETRY, "MOSTLY_Q4_K_M"): (512, 512),  # (linear, moe)
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


def _gguf_q5_k_t16_gemv_rowtile_gfx1151_bf16_bf16_out(
    x_ptr: int,
    tiles_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    **kwargs,
) -> None:
    fn = (
        gguf_q5_k_t16_gemv_rowtile_col8_bf16_bf16_out
        if (int(in_features), int(out_features))
        in _GGUF_Q5_T16_ROWTILE_COL8_SHAPES
        else gguf_q5_k_t16_gemv_rowtile_bf16_bf16_out
    )
    fn(
        x_ptr,
        tiles_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
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
    "gguf_q5_k_t16_v1": {
        (6_144, 5_120): "t16_gemv_decode_tile8_bf16_bf16_out",
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
    (17_408, 5_120): 1_024,
    (5_120, 1_024): 536,
}
GGUF_Q6_PLANAR_PREFILL_SHARED4R4_ROWS = 256
GGUF_Q6_PLANAR_PREFILL_SHARED4R4_SHAPES = frozenset({(17_408, 5_120)})
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
GGUF_Q6_T16_F16_ROCBLAS_PREFILL_POLICIES = {}
GGUF_Q4_T16_F16_ROCBLAS_PREFILL_POLICIES = {}
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
# SPECDEC2 S3 admits construction of the dense NextN c1 staged adapter on
# gfx1151. S4 additionally admits the physical c2/c4 adapter; arithmetic/default
# promotion remains independently gated by each phase's correctness and
# complete-wall packet. These capabilities expose adapters and AR fallback only.
GGUF_SPECDEC2_MTP2_C1 = True
# Scaling-campaign M1 measured the production single-group wide cycle (bound 8)
# exact at 80/80 with unchanged acceptance, but C6-C8 regressed versus the
# two-subgroup default because rows>16 target verification falls to direct
# Q6-planar/Q4-selected owners and the single accept interval scales with
# width (see the M1 blocker artifact). Production retains the certified
# width-4 default; an admitted profile re-lists its bound to lift the cap.
GGUF_SPECDEC2_MTP2_PHYSICAL = True
GGUF_SPECDEC2_MTP2_PHYSICAL_MAX_REQUESTS: dict[str, int] = {}
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
    (QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M", "production"): (1, 2, 3, 4),
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
        # across rows 2-8 (tests/test_qwen38_nextn_proposal_head_rowtile.py);
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
GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE = "baseline"
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
# WPF-H5J/H5Q's K1024 IQ row owners are W7900-only until an independent gfx1151
# actual-layer and complete-runtime transfer gate exists.
LAGUNA_GROUPED_IQ_DOWN_VARIANTS = {}
LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS = {}
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
        # Dense-H5120 narrow/selective Q4T16 col4 rowtiles are W7900-only until
        # gfx1151 receives an independent shape crossover and full-model gate.
        (
            "linear",
            "gguf_q4_k_t16_v1",
            "dense_rowtile_col4_bf16_bf16_out",
        ),
        # The exact scalar F32 alpha/beta pair is independently admitted on
        # gfx1151; wider/native rows retain singleton projections.
        # The cross-family alpha/beta plus snapshot-Conv owner has only been
        # screened on gfx1100; gfx1151 retains three independent leaves.
        (
            "linear_attn_alpha_beta+chain_conv+snapshot",
            "f32",
            "bf16_k5120_n48_c10240_k4_exact_state_rows_tloop",
        ),
        # The dependent alpha/beta-to-GDN owner is screened only on gfx1100;
        # gfx1151 retains scalar projections plus snapshot chain GDN.
        (
            "linear_attn_alpha_beta+gdn_chain_recurrent_rmsnorm_gate+cast+snapshot",
            "f32+gguf_q5_k_t16_v1",
            "bf16_k5120_n48_hk16_hv48_d128_exact_state_rows_tloop_f32_bf16_out",
        ),
        # Dense-H5120 down+residual and rounded next-input RMSNorm fusions
        # are W7900-only pending independent gfx1151 boundary/model gates.
        (
            "linear+residual",
            "gguf_q4_k_t16_v1",
            "dense_rowtile_bf16_residual_bf16_out",
        ),
        (
            "add+rmsnorm",
            "gguf_f32_weight",
            "rounded_bf16_out",
        ),
        # Shared-cache verifier KV batching is qualified only for the W7900
        # dense-H5120 N1 graph; gfx1151 retains scalar append aliases.
        (
            "paged_kv_write",
            "gguf_q4_k_m",
            "mixed_bf16_shared_batch_spans",
        ),
        # The scalar Q5T16 ssm_out GDN+cast handoff is independently admitted
        # on gfx1151; verifier-chain ownership remains W7900-only. D08-P1
        # aliases the generic direct/rowtile/WMMA leaves for its QKV role.
        (
            "gdn_chain_recurrent_rmsnorm_gate+cast",
            "gguf_q5_k_t16_v1",
            "bf16_c1_exact_state_rows_tloop_f32_bf16_out",
        ),
        # GGUF rollback snapshot collapse and producer-folded capture are
        # W7900-only until gfx1151 receives independent transaction and
        # launch-overhead gates.
        (
            "linear_state_pair_copy",
            "f32",
            "chunked_i32",
        ),
        (
            "linear_attn_chain_conv_decode+snapshot",
            "gguf_qwen35",
            "bf16_c1_exact_state_rows_tloop",
        ),
        (
            "gdn_chain_recurrent_rmsnorm_gate+snapshot",
            "gguf_qwen35",
            "bf16_c1_exact_state_rows_tloop",
        ),
        (
            "gdn_chain_recurrent_rmsnorm_gate+cast+snapshot",
            "gguf_q5_k_t16_v1",
            "bf16_c1_exact_state_rows_tloop_f32_bf16_out",
        ),
        # P2 admits the exact native planar-qmicro Q6 leaves on gfx1151. The
        # changed-math source-F16 library route remains independently excluded.
        (
            "linear",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "f16_rocblas_t16_qmicro_planar_bf16_bf16_out",
        ),
        # Qwen3.8-27B actual down weights reject the exact fused rowtile at
        # rows2/3/4 by 17.35%/11.44%/11.15%; retain planar projection plus
        # primitive BF16 add as the exact, faster chain on gfx1151.
        (
            "linear+residual",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_gemv_rowtile_bf16_residual_bf16_out",
        ),
        (
            "linear_q8_1",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_q8_1_dp4a_gemv_bf16_bf16_out",
        ),
        (
            "linear_q8_1+residual",
            "gguf_q6_k_t16_qmicro_planar_v1",
            "t16_q8_1_dp4a_gemv_bf16_residual_bf16_out",
        ),
        # Exact single-page and P2 split attention are W7900-only until gfx1151
        # receives independent crossover, full-state, and performance gates.
        (
            "laguna_attention_decode",
            "bf16",
            "global_context_single_page_spans",
        ),
        (
            "laguna_attention_decode+attention_gate",
            "bf16",
            "global_single_page_softplus_bf16_spans",
        ),
        # The exact split-attention producer/reducer bundle is registered for
        # an independent gfx1151 threshold and full-model screen. Automatic
        # selection remains off until that architecture-local gate passes.
        # The global-only wave-0 tree remains W7900-only. The retained scalar
        # current-P4 global/SWA bodies are independently gated on gfx1151.
        (
            "head_rmsnorm+partial_rotary+kv_write",
            "laguna_f32_weight",
            "global_wave0_tree_f32_bf16_spans",
        ),
        # D9's wave-0 RMS tree has an independent gfx1151 gate. The exact
        # top-10 split sibling remains W7900-only.
        (
            "weighted_sum+moe_tail",
            "bf16",
            "laguna_top10_routed_hidden_out",
        ),
        # Staged unrounded-F32 Laguna add+RMSNorm is W7900-only until an
        # independent gfx1151 correctness and performance gate.
        (
            "add_rmsnorm",
            "gguf_f32_weight",
            "bf16_out_staged_f32_local256",
        ),
        # IQ2 fixed-local64 DPP reduction is W7900-only pending an independent gate.
        (
            "moe_linear",
            "gguf_iq2_xs",
            "selected_dual_silu_gemv_decode_tile2_grid64_local64_reduce_bf16_bf16_out",
        ),
        # IQ3 selected-down tiling is gfx1100-only: gfx1151 tile4 was
        # trajectory-equivalent to tile1 but noise-flat/regressive at complete
        # DFlash wall, while the shared route also failed true-AR equality.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_gemv_decode_tile4_bf16_bf16_out",
        ),
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_gemv_decode_k1024_wave4_signbit_bf16_bf16_out",
        ),
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_weighted_down_gemv_decode_k1024_wave10_bf16_bf16_out",
        ),
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_weighted_down_gemv_decode_k1024_wave10_signbit_bf16_bf16_out",
        ),
        # Laguna top-10/K1024 IQ4 weighted ownership is gfx1100-only pending an
        # independent gfx1151 correctness and performance gate.
        (
            "moe_linear",
            "gguf_iq4_xs",
            "selected_weighted_down_gemv_decode_bf16_bf16_out",
        ),
        # WPF-1 fixed-grid-Y raw Q5/Q6 row reuse is W7900-only pending an
        # independent gfx1151 gate. Keep every output/slab key unaliased.
        *(
            ("linear", quant, f"rowbatch{row_batch}_bf16_{output_dtype}_out")
            for quant in ("gguf_q5_k", "gguf_q6_k")
            for row_batch in (4, 8, 16, 32)
            for output_dtype in ("bf16", "f32")
        ),
        *(
            (
                "linear",
                quant,
                f"coltile{col_tile}_rowbatch{row_batch}_bf16_{output_dtype}_out",
            )
            for quant in ("gguf_q5_k", "gguf_q6_k")
            for col_tile, row_batch in ((2, 16), (4, 8))
            for output_dtype in ("bf16", "f32")
        ),
        # WPF-H7C transfers the exact H6U reduction instruction form to two
        # W7900 raw-Q6 leaves and remains absent without a gfx1151 screen.
        (
            "linear",
            "gguf_q6_k",
            "dpp_wave_reduction_coltile4_rowbatch8_bf16_bf16_out",
        ),
        (
            "linear",
            "gguf_q6_k",
            "dpp_wave_reduction_coltile2_rowbatch16_bf16_f32_out",
        ),
        # WPF-H7I removes H7C's inner live-row predicate only for exactly-full
        # W7900 natural-M512 role groups; gfx1151 remains unscreened.
        (
            "linear",
            "gguf_q6_k",
            "dpp_wave_reduction_full_group_compute_"
            "coltile4_rowbatch8_bf16_bf16_out",
        ),
        (
            "linear",
            "gguf_q6_k",
            "dpp_wave_reduction_full_group_compute_"
            "coltile2_rowbatch16_bf16_f32_out",
        ),
        # WPF-H5U and H6A local256 cached-only global leaves are W7900-only
        # pending independent gfx1151 resource/performance gates.
        (
            "laguna_attention_prefill",
            "bf16",
            "global_context_rows_cached_exact_spans",
        ),
        (
            "laguna_attention_prefill",
            "bf16",
            "global_context_rows_dense_initial_cached_exact_spans",
        ),
        # WPF-H6N's retained fixed-arena leaf is W7900-only pending independent
        # gfx1151 resource/performance evidence and any bounded runtime owner.
        (
            "laguna_attention_prefill",
            "bf16",
            "global_context_rows_dense_initial_fixed512_cached_exact_spans",
        ),
        # WPF-H6Z exact late-start global qrow4 score/weight replay is W7900-
        # only pending independent gfx1151 resource/performance evidence.
        (
            "laguna_attention_prefill",
            "bf16",
            "global_context_rows_qrow4_dense_initial_global_score_weight_replay_exact_spans",
        ),
        # WPF-H5R and H6A exact cached-only two-pass SWA qrow4 leaves are
        # W7900-only pending independent gfx1151 resource/performance gates.
        (
            "laguna_attention_prefill",
            "bf16",
            "swa_context_rows_qrow4_cached_exact_spans",
        ),
        (
            "laguna_attention_prefill",
            "bf16",
            "swa_context_rows_qrow4_dense_initial_cached_exact_spans",
        ),
        # WPF-H6W exact late-start global score replay is W7900-only pending
        # independent gfx1151 resource/performance evidence.
        (
            "laguna_attention_prefill",
            "bf16",
            "swa_context_rows_qrow4_dense_initial_global_score_replay_exact_spans",
        ),
        # WPF-H7Y's lane-major SWA cache consumer and fused mirror writer are
        # W7900-only pending an independent gfx1151 resource/performance gate.
        (
            "laguna_attention_prefill",
            "bf16",
            "swa_context_rows_qrow4_dense_initial_lane_major_global_score_replay_exact_spans",
        ),
        (
            "laguna_kv_write",
            "bf16",
            "swa_f32_rows_natural_lane_major_spans",
        ),
        # WPF-H5M exact source-qualified qrow4 is W7900-only pending an
        # independent gfx1151 resource/performance gate.
        (
            "laguna_attention_prefill",
            "bf16",
            "swa_context_rows_qrow4_sourcequal_exact_spans",
        ),
        # WPF-H5N's identity/no-wrap exact specialization is likewise scoped to
        # gfx1100 until it has independent gfx1151 evidence.
        (
            "laguna_attention_prefill",
            "bf16",
            "swa_context_rows_qrow4_dense_first_fill_exact_spans",
        ),
        # WPF-H2 copies llama.cpp's gfx1100 F16-WMMA FlashAttention geometry
        # and remains excluded until gfx1151 receives an independent gate.
        (
            "laguna_attention_prefill",
            "bf16",
            "source_f16_wmma_q8_gqa8_spans",
        ),
        # WPF-H1 copies the gfx1100/RDNA3 source geometry and remains excluded
        # until gfx1151 receives an independent resource/correctness gate.
        ("activation_quant", "q8_1_ds4", "bf16_kmajor"),
        (
            "linear",
            "gguf_q5_k",
            "mmq_i128_j128_k256_q8_1_ds4_bf16_bf16_out",
        ),
        (
            "linear",
            "gguf_q5_k",
            "mmq_i128_j128_k256_q8_1_ds4_bf16_f32_out",
        ),
        # WPF-H6L's exact K3072/N1024/E256 pair16 rowbatch16 leaf is
        # W7900-only pending independent gfx1151 resource/performance gates.
        (
            "moe_linear",
            "gguf_iq2_xs",
            "selected_dual_silu_grouped_prefill_compact_"
            "k3072_n1024_e256_pair16_rowbatch16_bf16_bf16_out",
        ),
        # WPF-H6C's K3072/N1024/E256 expert-major fused-SiLU leaf is
        # W7900-only pending independent gfx1151 resource/performance gates.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_dual_silu_grouped_prefill_compact_"
            "k3072_n1024_e256_rowbatch4_bf16_bf16_out",
        ),
        # WPF-H5J's K1024 resident-segment IQ3 and one-wave IQ4 schedules are
        # W7900-only pending independent gfx1151 resource/performance gates.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_resident_rowbatch8_bf16_bf16_out",
        ),
        (
            "moe_linear",
            "gguf_iq4_xs",
            "selected_grouped_prefill_compact_k1024_wave32_bf16_bf16_out",
        ),
        # H5Q's active-expert persistent traversal is likewise gfx1100-only
        # until gfx1151 receives independent resource/performance evidence.
        *(
            (
                "moe_linear",
                "gguf_iq3_xxs",
                "selected_grouped_prefill_compact_k1024_active_expert_p"
                f"{partition}_resident_rowbatch8_bf16_bf16_out",
            )
            for partition in (64,)
        ),
        # H5Z's activation-resident output sweep keeps H5Q expert P64 but is
        # gfx1100-only until independently screened on gfx1151.
        *(
            (
                "moe_linear",
                "gguf_iq3_xxs",
                "selected_grouped_prefill_compact_k1024_active_expert_p64_"
                "activation_resident_out_p"
                f"{output_partition}_rowbatch8_bf16_bf16_out",
            )
            for output_partition in (256,)
        ),
        # H6D interleaves independent H5Z row accumulators specifically for
        # gfx1100 VOPD and remains absent without a gfx1151 screen.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_active_expert_p64_"
            "activation_resident_out_p256_row_interleaved_vopd_rowbatch8_"
            "bf16_bf16_out",
        ),
        # H6F pairs two independent H6D output reductions to amortize gfx1100
        # barriers and remains absent without an independent gfx1151 screen.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_active_expert_p64_"
            "activation_resident_out_p256_row_interleaved_vopd_paired_output_"
            "rowbatch8_bf16_bf16_out",
        ),
        # H6I groups three independent H6F output reductions and remains
        # gfx1100-only until an independent gfx1151 screen.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_active_expert_p64_"
            "activation_resident_out_p256_row_interleaved_vopd_triple_output_"
            "rowbatch8_bf16_bf16_out",
        ),
        # H6P changes H6I accumulator liveness with gfx1100 wave publication
        # and remains absent without an independent gfx1151 screen.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_active_expert_p64_"
            "activation_resident_out_p256_row_interleaved_vopd_"
            "staged_wave_publication_triple_output_rowbatch8_bf16_bf16_out",
        ),
        # H6Q changes only H6P's gfx1100 shuffle-loop code footprint and
        # remains absent without an independent gfx1151 screen.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_active_expert_p64_"
            "activation_resident_out_p256_row_interleaved_vopd_"
            "staged_wave_publication_compact_shuffle_loop_triple_output_"
            "rowbatch8_bf16_bf16_out",
        ),
        # H6R changes only H6Q's gfx1100 wave peer-exchange instructions and
        # remains absent without an independent gfx1151 screen.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_active_expert_p64_"
            "activation_resident_out_p256_row_interleaved_vopd_"
            "staged_wave_publication_dpp_peer_exchange_triple_output_"
            "rowbatch8_bf16_bf16_out",
        ),
        # H6T changes only H6R's final gfx1100 DPP move/add instruction form
        # and remains absent without an independent gfx1151 screen.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_grouped_prefill_compact_k1024_active_expert_p64_"
            "activation_resident_out_p256_row_interleaved_vopd_staged_wave_"
            "publication_dpp_peer_exchange_fused_add_triple_output_"
            "rowbatch8_bf16_bf16_out",
        ),
        # WPF-H3 reuses the DS4 producer but has independently qualified raw-IQ
        # consumers. Both remain gfx1100-only pending a gfx1151 gate.
        *(
            (
                "moe_linear",
                quant,
                "selected_mmq_i128_j128_k256_q8_1_ds4_prefill_compact_bf16_bf16_out",
            )
            for quant in ("gguf_iq3_xxs", "gguf_iq4_xs")
        ),
        # H7E's residual-D4x2 IQ3 consumer remains gfx1100-only until an
        # independently qualified gfx1151 residual-plane gate exists.
        (
            "moe_linear",
            "gguf_iq3_xxs",
            "selected_mmq_i128_j128_k256_q8_1_ds4x2_"
            "prefill_compact_bf16_bf16_out",
        ),
        # WPF-H4 copies llama.cpp's gfx1100 Q6-to-F16/rocBLAS ownership and
        # remains excluded until gfx1151 receives an independent gate.
        ("dequant", "gguf_q6_k", "raw_f16_source_local64"),
        (
            "dequant_cast",
            "gguf_q6_k",
            "raw_f16_bf16_input_source_local64",
        ),
        (
            "linear",
            "gguf_q6_k",
            "f16_rocblas_source_bf16_bf16_out",
        ),
        (
            "linear",
            "gguf_q6_k",
            "f16_rocblas_source_bf16_f32_out",
        ),
        # WPF-H5A/H5I transient exact-F32 Q5/Q6 producers are gfx1100-only
        # until gfx1151 receives independent resource, timing, and quality gates.
        ("dequant", "gguf_q5_k", "raw_f32_exact_local64"),
        ("dequant", "gguf_q6_k", "raw_f32_exact_local64"),
        (
            "dequant_cast",
            "gguf_q5_k",
            "raw_f32_bf16_input_exact_local64",
        ),
        (
            "linear",
            "gguf_q5_k",
            "f32_rocblas_exact_values_bf16_bf16_out",
        ),
        (
            "linear",
            "gguf_q5_k",
            "f32_rocblas_exact_values_bf16_f32_out",
        ),
        # WPF-H5C/H5I production-ordered and H5L weight-major F32 consumers
        # plus raw-Q5/Q6 composites stay gfx1100-only pending a gfx1151 gate.
        *(
            (
                "linear",
                quant,
                f"{prefix}coltile{col_tile}_rowbatch{row_batch}_"
                f"bf16_{output_dtype}_out",
            )
            for quant, prefix in (
                ("f32_weight", "ordered_"),
                ("gguf_q5_k", "f32_ordered_"),
                ("gguf_q6_k", "f32_ordered_"),
            )
            for col_tile, row_batch in (
                (4, 8),
                (8, 4),
                (4, 16),
                (8, 8),
                (16, 4),
                (12, 4),
                (8, 10),
                (16, 5),
                (8, 12),
                (12, 8),
            )
            if quant != "gguf_q6_k"
            or (col_tile, row_batch) in {(8, 4), (16, 4), (16, 5)}
            for output_dtype in ("bf16", "f32")
        ),
        *(
            (
                "linear",
                quant,
                f"{prefix}coltile{col_tile}_rowbatch{row_batch}_"
                f"bf16_{output_dtype}_out",
            )
            for quant, prefix in (
                ("f32_weight", "ordered_weight_major_"),
                ("gguf_q5_k", "f32_ordered_weight_major_"),
            )
            for col_tile, row_batch, output_dtype in (
                (8, 4, "bf16"),
                (8, 12, "bf16"),
                (16, 5, "bf16"),
                (12, 8, "bf16"),
                (16, 4, "bf16"),
                (16, 5, "f32"),
                (8, 10, "f32"),
            )
        ),
        *(
            (
                "linear",
                "gguf_q6_k",
                f"f32_ordered_weight_major_coltile{col_tile}_"
                f"rowbatch{row_batch}_bf16_{output_dtype}_out",
            )
            for col_tile, row_batch, output_dtype in (
                (16, 5, "bf16"),
                (16, 4, "bf16"),
                (16, 5, "f32"),
            )
        ),
        # H5X exact tile-K-col Q5 producers/consumers are W7900-only until an
        # independent gfx1151 layout/resource/performance gate qualifies them.
        *(
            (
                layer,
                quant,
                f"{prefix}coltile{col_tile}_rowbatch{row_batch}_"
                f"bf16_{output_dtype}_out",
            )
            for layer, quant, prefix in (
                ("dequant", "gguf_q5_k", "raw_f32_exact_tile_k_col_"),
                (
                    "linear",
                    "f32_weight",
                    "ordered_weight_major_tile_k_col_",
                ),
                (
                    "linear",
                    "gguf_q5_k",
                    "f32_ordered_weight_major_tile_k_col_",
                ),
            )
            for col_tile, row_batch, output_dtype in (
                (8, 4, "bf16"),
                (16, 5, "bf16"),
                (16, 5, "f32"),
                (8, 10, "f32"),
            )
        ),
        # H5Y exact activation-tile-K-row packs/consumers are likewise
        # W7900-only pending an independent gfx1151 transfer gate.
        *(
            (
                "activation_pack",
                "bf16",
                f"tile_k_row_coltile{col_tile}_rowbatch{row_batch}_"
                f"bf16_{output_dtype}_out",
            )
            for col_tile, row_batch, output_dtype, _weight_layout in (
                (8, 4, "bf16", "tile_k_col"),
                (8, 12, "bf16", "row_major"),
                (16, 5, "bf16", "tile_k_col"),
                (12, 8, "bf16", "row_major"),
                (16, 5, "f32", "tile_k_col"),
                (8, 10, "f32", "tile_k_col"),
            )
        ),
        *(
            (
                "linear",
                quant,
                f"{prefix}{weight_layout}_activation_tile_k_row_"
                f"coltile{col_tile}_rowbatch{row_batch}_"
                f"bf16_{output_dtype}_out",
            )
            for quant, prefix in (
                ("f32_weight", "ordered_weight_major_"),
                ("gguf_q5_k", "f32_ordered_weight_major_"),
            )
            for col_tile, row_batch, output_dtype, weight_layout in (
                (8, 4, "bf16", "tile_k_col"),
                (8, 12, "bf16", "row_major"),
                (16, 5, "bf16", "tile_k_col"),
                (12, 8, "bf16", "row_major"),
                (16, 5, "f32", "tile_k_col"),
                (8, 10, "f32", "tile_k_col"),
            )
        ),
        # H7G exact padded-row Q5 consumers are gfx1100-only pending an
        # independent gfx1151 physical/performance transfer gate.
        *(
            (
                "linear",
                quant,
                f"{prefix}{weight_layout}_activation_tile_k_row_"
                f"padded_compute_coltile{col_tile}_rowbatch{row_batch}_"
                f"bf16_{output_dtype}_out",
            )
            for quant, prefix in (
                ("f32_weight", "ordered_weight_major_"),
                ("gguf_q5_k", "f32_ordered_weight_major_"),
            )
            for col_tile, row_batch, output_dtype, weight_layout in (
                (8, 12, "bf16", "row_major"),
                (16, 5, "bf16", "tile_k_col"),
                (16, 5, "f32", "tile_k_col"),
                (8, 10, "f32", "tile_k_col"),
            )
        ),
        # H8A's resident-plane composites remain gfx1100-only; gfx1151 has no
        # package capability or independently qualified owner/cache policy.
        *(
            (
                "linear",
                "gguf_q5_k",
                "f32_resident_ordered_weight_major_tile_k_col_"
                "activation_tile_k_row_padded_compute_coltile16_rowbatch5_"
                f"bf16_{output_dtype}_out",
            )
            for output_dtype in ("bf16", "f32")
        ),
        # H7H exact full-group Q5 consumers are separately scoped to gfx1100.
        *(
            (
                "linear",
                quant,
                f"{prefix}{weight_layout}_activation_tile_k_row_"
                f"full_group_compute_coltile{col_tile}_rowbatch{row_batch}_"
                f"bf16_{output_dtype}_out",
            )
            for quant, prefix in (
                ("f32_weight", "ordered_weight_major_"),
                ("gguf_q5_k", "f32_ordered_weight_major_"),
            )
            for col_tile, row_batch, output_dtype, weight_layout in (
                (8, 4, "bf16", "tile_k_col"),
                (12, 8, "bf16", "row_major"),
            )
        ),
        # H6E exact Q6 activation-row consumers are separately scoped to
        # gfx1100 pending their standalone physical/performance gate.
        *(
            (
                "linear",
                quant,
                f"{prefix}weight_major_row_major_activation_tile_k_row_"
                f"coltile{col_tile}_rowbatch{row_batch}_"
                f"bf16_{output_dtype}_out",
            )
            for quant, prefix in (
                ("f32_weight", "ordered_"),
                ("gguf_q6_k", "f32_ordered_"),
            )
            for col_tile, row_batch, output_dtype in (
                (16, 5, "bf16"),
                (16, 4, "bf16"),
                (16, 5, "f32"),
            )
        ),
        # H6U exact DPP-reduction candidates remain gfx1100-only unless their
        # standalone W7900 screen and a later gfx1151 transfer both pass.
        *(
            (
                "linear",
                quant,
                f"{prefix}weight_major_row_major_activation_tile_k_row_"
                f"dpp_wave_reduction_coltile{col_tile}_"
                f"rowbatch{row_batch}_bf16_{output_dtype}_out",
            )
            for quant, prefix in (
                ("f32_weight", "ordered_"),
                ("gguf_q6_k", "f32_ordered_"),
            )
            for col_tile, row_batch, output_dtype in (
                (16, 5, "bf16"),
                (16, 4, "bf16"),
                (16, 5, "f32"),
            )
        ),
        # Rejected WPF-1B producer/MMQ primitives remain gfx1100-only
        # diagnostic evidence, with no runtime policy owner on either backend.
        ("activation_quant", "q8_1_d4s4_f32", "bf16"),
        ("activation_quant", "q8_1_d8s8_f32", "bf16"),
        ("activation_quant", "q8_1_d8r8s8_f32", "bf16"),
        *(
            (
                "linear",
                quant,
                f"mmq32_q8_1_{producer}_f32_bf16_{output_dtype}_out",
            )
            for quant in ("gguf_q5_k", "gguf_q6_k")
            for producer in ("d4s4", "d8s8", "d8r8s8")
            for output_dtype in ("bf16", "f32")
        ),
        # Q4 local32 LM-head ownership is W7900-only pending an independent gate.
        (
            "linear",
            "gguf_q4_k",
            "local32_fixed_meta_gemv_decode_bf16_f32_out",
        ),
        # Q6 local32 standalone ownership is likewise W7900-only pending a gate.
        (
            "linear",
            "gguf_q6_k",
            "standalone_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out",
        ),
        # Paired-output SWAR Q5 reconstruction is W7900-only pending an independent gate.
        (
            "linear",
            "gguf_q5_k",
            "wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out",
        ),
        (
            "linear_pair",
            "gguf_q5_k",
            "wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out",
        ),
        (
            "attention_projection_quad",
            "gguf_q5_k+gguf_q6_k+gguf_q6_k+gguf_q5_k",
            "mixed_local32_q5_swar_pair_fixed_meta_gemv_decode_bf16_f32_out",
        ),
        # Heterogeneous Q5/Q6 pair reuse is W7900-only pending an independent gate.
        (
            "attention_projection_quad",
            "gguf_q5_k+gguf_q6_k+gguf_q6_k+gguf_q5_k",
            "mixed_pair_reuse_local32_fixed_meta_pack8_gemv_decode_bf16_f32_out",
        ),
        # Laguna compact/persistent routing is W7900-only pending independent gates.
        (
            "laguna_sigmoid_router_topk",
            "f32",
            "correction_bias_compact_wave32",
        ),
        (
            "laguna_router_topk",
            "f32",
            "bf16_hidden_correction_bias_persistent_wave_top10",
        ),
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
    # Source-faithful Vulkan DMMV output-row reuse, adapted without Vulkan's
    # rejected wave64/local64 geometry: one retained local256 block owns two
    # adjacent F16 columns, shares the BF16 activation load/conversion, and
    # reduces both exact accumulator trees through one barrier.
    (
        "attention_projection+head_rmsnorm+partial_rotary+kv_write",
        "fp16_weight+laguna_f32_weight",
        "global_fixedk_nontemporal_bf16_f32_spans",
    ): laguna_global_f16_projection_head_kv_nontemporal_tile2_bf16_spans,
    (
        "attention_projection+head_rmsnorm+partial_rotary+kv_write",
        "fp16_weight+laguna_f32_weight",
        "swa_fixedk_nontemporal_bf16_f32_spans",
    ): laguna_swa_f16_projection_head_kv_nontemporal_tile2_bf16_spans,
    # F3Q caches 24 of 128 FP32 state rows across the GDN dependency barrier.
    # Its 15 KiB LDS footprint preserves four resident blocks on gfx1151.
    (
        "gdn_recurrent_rmsnorm_gate",
        "gguf_qwen35",
        "bf16_indexed_singleton",
    ): qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16,
    # fp16-state sibling for the production route (half-sized per-slot state);
    # rows < 8 delegate to the plain fp16 indexed wrapper inside the wrapper.
    (
        "gdn_recurrent_rmsnorm_gate",
        "gguf_qwen35",
        "bf16_indexed_singleton_fp16state",
    ): qwen35_gdn_recurrent_rmsnorm_gate_indexed_shared_statecache24_lowp_bf16_fp16state,
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
        "gguf_q5_k_t16_v1",
        "t16_gemv_rowtile_bf16_bf16_out",
    ): _gguf_q5_k_t16_gemv_rowtile_gfx1151_bf16_bf16_out,
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
    (
        "linear",
        "gguf_q4_k",
        "pack8_wmma_prefill_bf16_bf16_out",
    ): gguf_q4_k_pack8_wmma_prefill_gfx1151_bf16_bf16_out,
    (
        "linear",
        "gguf_q6_k",
        "wmma_prefill_bf16_bf16_out",
    ): gguf_q6_k_wmma_prefill_16x32_bf16_bf16_out,
    (
        "linear",
        "gguf_q4_k_t16_v1",
        "t16_wmma_prefill_bf16_bf16_out",
    ): gguf_q4_k_t16_wmma_prefill_gfx1151_bf16_bf16_out,
    (
        "linear",
        "gguf_q5_k_t16_v1",
        "t16_wmma_prefill_bf16_bf16_out",
    ): gguf_q5_k_t16_wmma_prefill_gfx1151_bf16_bf16_out,
    # B2 P1: F16-staged activation siblings, admitted unselected; the
    # bf16 routers above stay the selected strict fallback.
    (
        "linear",
        "gguf_q4_k_t16_v1",
        "t16_wmma_prefill_fp16_in_bf16_out",
    ): gguf_q4_k_t16_wmma_prefill_gfx1151_fp16_in_bf16_out,
    (
        "linear",
        "gguf_q4_k_t16_v1",
        "t16_wmma_prefill_shared_b_fp16_in_bf16_out",
    ): gguf_q4_k_t16_wmma_prefill_shared_b_fp16_in_bf16_out,
    (
        "linear",
        "gguf_q5_k_t16_v1",
        "t16_wmma_prefill_fp16_in_bf16_out",
    ): gguf_q5_k_t16_wmma_prefill_gfx1151_fp16_in_bf16_out,
    (
        "linear",
        "gguf_q6_k_t16_v1",
        "t16_wmma_prefill_bf16_bf16_out",
    ): gguf_q6_k_t16_wmma_prefill_gfx1151_bf16_bf16_out,
    (
        "linear",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "t16_wmma_prefill_bf16_bf16_out",
    ): gguf_q6_k_t16_qmicro_planar_wmma_prefill_gfx1151_bf16_bf16_out,
}
_GFX1100_MODULES = (
    "hipengine.kernels.hip_gfx1100.attention",
    "hipengine.kernels.hip_gfx1100.attention.maple_attention",
    "hipengine.kernels.hip_gfx1100.convert",
    "hipengine.kernels.hip_gfx1100.fused",
    "hipengine.kernels.hip_gfx1100.linear",
    "hipengine.kernels.hip_gfx1100.linear_attn",
    "hipengine.kernels.hip_gfx1100.moe",
    "hipengine.kernels.hip_gfx1100.moe.maple_moe",
    "hipengine.kernels.hip_gfx1100.norm",
    "hipengine.kernels.hip_gfx1100.quant",
    "hipengine.kernels.hip_gfx1100.quant.maple_ternary",
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
    q6_integer_mmq_key = KernelKey(
        BACKEND,
        "linear",
        "gguf_q6_k_t16_qmicro_planar_v1",
        "t16_q8_1_planar_integer_mmq64x64_bf16_bf16_out",
    )
    if replace or not is_registered(q6_integer_mmq_key):
        register(
            q6_integer_mmq_key,
            gguf_q6_k_t16_qmicro_planar_dense_q8_1_mmq64x64_bf16_bf16_out,
            replace=replace,
        )
    for variant, fn in (
        (
            "t16_wmma_prefill_single_wave_bf16_bf16_out",
            gguf_q4_k_t16_wmma_prefill_bf16_bf16_out,
        ),
        (
            "t16_wmma_prefill_smallm_bf16_bf16_out",
            gguf_q4_k_t16_wmma_prefill_smallm_bf16_bf16_out,
        ),
        (
            "t16_wmma_prefill_shared_b_bf16_bf16_out",
            gguf_q4_k_t16_wmma_prefill_shared_b_bf16_bf16_out,
        ),
        (
            "t16_wmma_prefill_shared_b3w8r3_bf16_bf16_out",
            gguf_q4_k_t16_wmma_prefill_shared_b3w8r3_bf16_bf16_out,
        ),
    ):
        key = KernelKey(BACKEND, "linear", "gguf_q4_k_t16_v1", variant)
        if replace or not is_registered(key):
            register(key, fn, replace=replace)
    q4_pack8_decode_pair_key = KernelKey(
        BACKEND,
        "linear_pair",
        "gguf_q4_k",
        "pack8_dual_decode_bf16_bf16_out",
    )
    if replace or not is_registered(q4_pack8_decode_pair_key):
        register(
            q4_pack8_decode_pair_key,
            gguf_q4_k_pack8_dual_prefill_bf16_bf16_out,
            replace=replace,
        )
    q4_pack8_decode_pair_silu_key = KernelKey(
        BACKEND,
        "linear_pair_silu",
        "gguf_q4_k",
        "pack8_dual_decode_bf16_bf16_out",
    )
    if replace or not is_registered(q4_pack8_decode_pair_silu_key):
        register(
            q4_pack8_decode_pair_silu_key,
            gguf_q4_k_pack8_dual_silu_bf16_bf16_out,
            replace=replace,
        )
    q4_pack8_decode_pair_silu_t128_key = KernelKey(
        BACKEND,
        "linear_pair_silu",
        "gguf_q4_k",
        "pack8_dual_decode_t128_bf16_bf16_out",
    )
    if replace or not is_registered(q4_pack8_decode_pair_silu_t128_key):
        register(
            q4_pack8_decode_pair_silu_t128_key,
            _qwen35_08b_q4_pack8_dual_silu_t128,
            replace=replace,
        )
    q4_t16_sidecar_pair_silu_key = KernelKey(
        BACKEND,
        "linear_pair_silu",
        "gguf_q4_k",
        "t16_sidecar_dual_decode_bf16_bf16_out",
    )
    if replace or not is_registered(q4_t16_sidecar_pair_silu_key):
        register(
            q4_t16_sidecar_pair_silu_key,
            gguf_q4_k_t16_dense_dual_local32_silu_bf16_bf16_out,
            replace=replace,
        )
    q4_t16_dual_interleaved_pair_silu_key = KernelKey(
        BACKEND,
        "linear_pair_silu",
        "gguf_q4_k",
        "t16_dual_interleaved_sidecar_decode_bf16_bf16_out",
    )
    if replace or not is_registered(
        q4_t16_dual_interleaved_pair_silu_key
    ):
        register(
            q4_t16_dual_interleaved_pair_silu_key,
            gguf_q4_k_t16_dense_dual_interleaved_tile2_local32_silu_bf16_bf16_out,
            replace=replace,
        )


register_gfx1151_kernels()
register_backend_kernels = register_gfx1151_kernels

__all__ = [
    "BACKEND",
    "GGUF_AOTRITON_HEAD_MAJOR_KV",
    "GGUF_AOTRITON_PREFILL",
    "GGUF_PREFILL_CHUNK_SIZES_BY_GEOMETRY",
    "GGUF_COMPACT_WMMA_NO_READ_MAX_SELECTED_ROWS",
    "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS",
    "GGUF_DECODE_GRAPH_SUBMISSION_POLICIES",
    "GGUF_PACKED_DECODE_GRAPH_MIN_REPLAY_STEPS_BY_POLICY",
    "GGUF_FP16_RECURRENT_STATE_DEFAULT_FILE_TYPES",
    "GGUF_GDN_INDEXED_SINGLETON_DECODE",
    "GGUF_GDN_PREFILL_AUTO_MODE",
    "GGUF_GDN_PREFILL_AUTO_MODES_BY_QUANT_SHAPE",
    "GGUF_GDN_PREFILL_COMPACT_PEER_CHUNK_ROWS",
    "GGUF_GDN_PREFILL_EXACT_MODE",
    "GGUF_FULL_ATTN_QK_POSTPROCESS_DECODE_POLICIES",
    "GGUF_HOST_TOKEN_EMBEDDING_C1",
    "GGUF_HOST_TOKEN_EMBEDDING_C1_GGML_TYPES",
    "GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1",
    "GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1_GGML_TYPES",
    "GGUF_MAPPED_HOST_TOKEN_EMBEDDING_C1_COPY",
    "GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA",
    "GGUF_PRIVATE_C1_SMALL_WEIGHT_ARENA_POLICIES",
    "GGUF_PRIVATE_C1_DECODE_SCRATCH_ARENA_POLICIES",
    "GGUF_LINEAR_ATTN_CONV_PREFILL_AUTO_MODE",
    "GGUF_PAGED_ATTN_GROUPED_GQA_MIN_CONTEXTS",
    "GGUF_PAGED_ATTN_PARALLEL_REDUCE",
    "GGUF_PAGED_ATTN_PARALLEL_REDUCE_MIN_CONTEXT",
    "GGUF_SHORT_C1_BATCH_ATTN_MAX_CONTEXT",
    "GGUF_SHORT_C1_BATCH_ATTN_THREADS",
    "GGUF_SHORT_C1_SPLIT_ATTN_POLICIES",
    "GGUF_PREFILL_DEVICE_METADATA_MAX_TOKENS",
    "GGUF_PREFILL_ROUTER_SELECT_THREADS",
    "GGUF_PREFILL_SCRATCH_ARENA_GROUPING",
    "GGUF_PREFILL_SCRATCH_LIVENESS_ALIAS",
    "GGUF_PREFILL_SCRATCH_LIVENESS_MIN_ROWS",
    "GGUF_Q4_K_M_FAIR_PREFILL_BURST_CHUNKS",
    "GGUF_Q4_K_M_MAX_PREFILL_CHUNK_TOKENS",
    "GGUF_Q4_K_M_PREFILL_DECODE_POLICY",
    "GGUF_Q4_K_M_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS",
    "GGUF_Q4_T16_PHYSICAL_SMALLM_ROWS",
    "GGUF_Q4_T16_PHYSICAL_SMALLM_MAX_ROWS",
    "GGUF_Q4_T16_PHYSICAL_SMALLM_SHAPES",
    "GGUF_Q4_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q4_T16_SELECTED_PREFILL_AUTO_MODE",
    "GGUF_Q5_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_DENSE_PAIR_SILU_DECODE_POLICIES",
    "GGUF_DENSE_PAIR_SILU_NATIVE_DECODE_POLICIES",
    "GGUF_DENSE_F32_ALPHA_BETA_PAIR_DECODE_SHAPES",
    "GGUF_DENSE_F32_ALPHA_BETA_CONV_DECODE_SHAPES",
    "GGUF_DENSE_PREFILL_SCRATCH_ROW_CAP_POLICIES",
    "GGUF_DENSE_DOWN_RESIDUAL_DECODE_POLICIES",
    "GGUF_NORM_RESIDUAL_DECODE_POLICIES",
    "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP",
    "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES",
    "GGUF_DENSE_Q4_T16",
    "GGUF_DENSE_Q4_T16_ATTN_Q_08B",
    "GGUF_DENSE_Q5_T16_SSM_OUT",
    "GGUF_DENSE_Q5_T16_SSM_OUT_08B",
    "GGUF_DENSE_Q5_T16_QKV",
    "GGUF_DENSE_Q5_T16_H5120",
    "GGUF_DENSE_Q6_T16_QMICRO_PLANAR",
    "GGUF_DENSE_Q6_T16_QMICRO_PLANAR_EXCLUDED_SLOTS",
    "GGUF_DENSE_T16_F16_ROCBLAS_PREFILL_POLICIES",
    "GGUF_Q4_T16_F16_ROCBLAS_PREFILL_POLICIES",
    "GGUF_Q5_T16_F16_ROCBLAS_PREFILL_POLICIES",
    "GGUF_Q6_T16_F16_ROCBLAS_PREFILL_POLICIES",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R9_ROWS",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R9_SHAPE",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R6_MIN_ROWS",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R6_MAX_ROWS_BY_SHAPE",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R4_ROWS",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R4_SHAPES",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R3_ROWS",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4R3_SHAPES",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4_MIN_ROWS",
    "GGUF_Q6_PLANAR_PREFILL_SHARED4_SHAPES",
    "GGUF_Q5_T16_DENSE_SHARED8R3_MIN_ROWS",
    "GGUF_Q5_T16_DENSE_SHARED8R3_MAX_ROWS",
    "GGUF_Q5_T16_DENSE_SHARED8R3_SHAPES",
    "GGUF_Q6_PREFILL_SHARED3R1_MIN_ROWS",
    "GGUF_Q6_PREFILL_SHARED3R1_MAX_ROWS",
    "GGUF_Q6_STANDARD_PREFILL_SHARED3R1_SHAPES",
    "GGUF_Q6_STANDARD_PREFILL_SHARED6R1_MIN_ROWS",
    "GGUF_Q6_STANDARD_PREFILL_SHARED6R1_MAX_ROWS",
    "GGUF_Q6_DENSE_INTEGER_MMQ_PREFILL_POLICY",
    "GGUF_Q6_PLANAR_PREFILL_SHARED3R1_SHAPES",
    "GGUF_Q6_STANDARD_PREFILL_SHARED4_MIN_ROWS",
    "GGUF_Q6_STANDARD_PREFILL_SHARED4_SHAPES",
    "GGUF_Q6_STANDARD_PREFILL_SHARED8R3_MIN_ROWS",
    "GGUF_Q6_STANDARD_PREFILL_SHARED8R3_MAX_ROWS",
    "GGUF_Q6_STANDARD_PREFILL_SHARED8R3_HIGH_MIN_ROWS",
    "GGUF_Q6_STANDARD_PREFILL_SHARED8R3_HIGH_MAX_ROWS",
    "GGUF_Q6_Q4_T16_MIXED_GRID_DECODE_SHAPES",
    "GGUF_NARROW_KV_PAIR_DECODE_SHAPES",
    "GGUF_T16_F16_ROCBLAS_MAX_ROWS_BY_QUANT_SHAPE",
    "GGUF_T16_F16_ROCBLAS_VARIANT_POLICIES",
    "GGUF_T16_NATIVE_DIRECT_SHAPES_BY_QUANT",
    "GGUF_T16_NATIVE_ROWTILE_MAX_ROWS_BY_QUANT",
    "GGUF_T16_NATIVE_ROWTILE_VARIANTS_BY_QUANT",
    "GGUF_T16_TARGET_VERIFIER_ROWTILE_SHAPES_BY_QUANT",
    "GGUF_T16_TARGET_VERIFIER_ROWTILE_CHUNK_ROWS_BY_QUANT",
    "GGUF_T16_TARGET_VERIFIER_TRUE_ROWTILE_VARIANTS",
    "GGUF_T16_TARGET_VERIFIER_WIDE_Q6_SHARED4_VARIANTS",
    "GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_PAIR_VARIANTS",
    "GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_ROWS",
    "GGUF_T16_TARGET_VERIFIER_PRODUCTION_Q4_ROWTILE_SHAPES",
    "GGUF_T16_C1_VARIANTS_BY_QUANT_SHAPE",
    "GGUF_T16_NATIVE_SPLIT_ROW_CHUNKS_BY_QUANT_SHAPE",
    "GGUF_Q5_T16_SELECTED_QWEN_TILE8",
    "GGUF_Q6_T16_SELECTED_PAIRREUSE_MIN_ROWS",
    "GGUF_Q6_LM_HEAD_MAX_CHUNK",
    "GGUF_SHARED_SLOT_AR_PHYSICAL_WIDTHS",
    "GGUF_SPECDEC2_MTP2_C1",
    "GGUF_SPECDEC2_MTP2_PHYSICAL",
    "GGUF_SPECDEC2_MTP2_PHYSICAL_MAX_REQUESTS",
    "GGUF_SPECDEC2_MTP2_BATCH_ROUTE_ABOVE_REQUESTS",
    "GGUF_SPECDEC2_PHYSICAL_PROMPT_STREAMING_POLICIES",
    "GGUF_SPECDEC2_PROPOSAL_LM_HEAD_ROWTILE_POLICIES",
    "GGUF_SPECDEC2_NATIVE_TARGET_GRAPH_MAX_CONTEXT",
    "GGUF_SPECDEC2_NATIVE_TARGET_MAX_CONTEXT",
    "GGUF_FUSED_LINEAR_STATE_TRANSFER",
    "GGUF_DIRECT_RESIDENT_LINEAR_STATE",
    "GGUF_C2_PACKED_PREFILL_MAX_ROWS",
    "GGUF_Q8_T16_DECODE_PAIR_ROWTILE_MIN_ROWS",
    "GGUF_Q8_T16_DECODE_ROWTILE_ALL",
    "GGUF_Q8_T16_DECODE_ROWTILE_MIN_ROWS",
    "GGUF_Q8_T16_DUAL_WMMA_PREFILL",
    "GGUF_Q8_T16_DUAL_WMMA_PREFILL_SHAPES",
    "GGUF_Q8_T16_DUAL_WMMA_PREFILL_POLICIES",
    "GGUF_Q8_T16_PREFILL_FOUR_WAVE",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE",
    "GGUF_Q8_T16_PREFILL_TWO_WAVE_MAX_TOKENS",
    "GGUF_F32_ORDERED_PREFILL_POLICIES",
    "GGUF_F32_ORDERED_PREFILL_QUANTS",
    "GGUF_Q6_F32_ORDERED_PREFILL",
    "GGUF_Q6_F32_ORDERED_PREFILL_POLICY",
    "GGUF_RAW_K_PREFILL_COLTILE2_SHAPES",
    "GGUF_RAW_K_PREFILL_COLTILE_SUPPORTED",
    "GGUF_RAW_K_PREFILL_ROWBATCH",
    "GGUF_RAW_K_PREFILL_ROWBATCH_SUPPORTED",
    "GGUF_RAW_K_PREFILL_VARIANT",
    "GGUF_ROUTER_F32_BF16_HIDDEN_THREADS",
    "LAGUNA_DENSE_Q4_PREFILL_MODE",
    "LAGUNA_F16_BOUNDARY_FUSION",
    "LAGUNA_F16_ATTENTION_QUAD_DECODE",
    "LAGUNA_F16_NONTEMPORAL_DECODE",
    "LAGUNA_F16_OUTPUT_ADD_RMSNORM_DECODE",
    "LAGUNA_F16_PROJECTION_HEAD_KV_DECODE",
    "LAGUNA_F16_DECODE_FIXEDK",
    "LAGUNA_F16_DECODE_ONEBARRIER",
    "LAGUNA_Q4_PACK8_DUAL_SILU_DECODE",
    "LAGUNA_Q4_DENSE_DECODE_T16_SIDECAR",
    "LAGUNA_Q4_DENSE_DECODE_T16_DUAL_INTERLEAVED",
    "LAGUNA_Q4_SHARED_DOWN_T16_DECODE",
    "LAGUNA_Q4_EXPERT_T16_DUAL_INTERLEAVED",
    "LAGUNA_SELECTED_NATURAL_DECODE",
    "LAGUNA_SELECTED_DOWN_NATURAL_PARALLEL_DECODE",
    "LAGUNA_SELECTED_DOWN_NATURAL_PARALLEL_WEIGHTED_DECODE",
    "LAGUNA_SELECTED_DOWN_Q4_PAIRCOEFF_WEIGHTED_DECODE",
    "LAGUNA_SELECTED_NATURAL_TILE8_DECODE",
    "LAGUNA_SELECTED_NATURAL_TILE8_PARALLEL_DECODE",
    "LAGUNA_SELECTED_NATURAL_TILE8_PARALLEL_SILU_DECODE",
    "LAGUNA_SELECTED_HALFDOT_DECODE",
    "LAGUNA_F16_PREFILL_MIN_ROWS",
    "LAGUNA_F16_PREFILL_MODE",
    "LAGUNA_F16_PREFILL_STRATEGY",
    "LAGUNA_GLOBAL_PREFILL_VARIANT",
    "LAGUNA_GLOBAL_SPLIT_MIN_LIVE",
    "LAGUNA_GLOBAL_SPLIT_FIXEDSHAPE_REDUCE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_DIM32_VSTAGE64",
    "LAGUNA_GLOBAL_SPLIT_GQA6_DEFERREDNORM_DIM32_VSTAGE64",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE64",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX_NONTEMPORAL_MIN_LIVE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_PREFETCH8_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE80_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE128_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_TOKENLOOP4_DEFERREDNORM_DIM32_VSTAGE128_PROBABILITY_VEC4_PREFETCH16_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_ALLWAVE_TILE1024_DENSE_PREFIX",
    "LAGUNA_GLOBAL_SPLIT_GQA6_ALLWAVE_TILE1024_DENSE_PREFIX_MIN_LIVE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_COMPENSATED_LAYER",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_COMPENSATED_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE_PREFETCH",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_ALLWAVE_SCORE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DIM_TILE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DEFERREDNORM",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_NONTEMPORAL_KEY_VALUE_PREFETCH",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_DENSE_PREFIX_SCORE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_MIN_LIVE",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_MIN_LAYER",
    "LAGUNA_GLOBAL_SPLIT_GQA6_CTX4096_TOKENLOOP4",
    "LAGUNA_HEAD_KV_FUSION",
    "LAGUNA_GROUPED_IQ_DOWN_VARIANTS",
    "LAGUNA_GROUPED_IQ_DOWN_VARIANT_ABIS",
    "LAGUNA_MOE_BRANCH_CONCURRENCY",
    "LAGUNA_MOE_DECODE_BRANCH_CONCURRENCY",
    "LAGUNA_MOE_DECODE_SHARED_NORMAL_PRIORITY",
    "LAGUNA_MOE_TAIL_WAVE0_TREE",
    "LAGUNA_MOE_GROUP_COMPACT_MODE",
    "LAGUNA_MOE_SHARED_AFTER_ROUTER",
    "LAGUNA_MOE_SHARED_LOW_PRIORITY",
    "LAGUNA_GLOBAL_FUSED_FIXEDSHAPE",
    "LAGUNA_GLOBAL_GQA2_VSTAGE64_FIXEDSHAPE",
    "LAGUNA_GLOBAL_GQA2_VSTAGE64_VEC16_FIXEDSHAPE",
    "LAGUNA_GLOBAL_GQA2_VSTAGE64_VEC16_DIRECT_FIXEDSHAPE",
    "LAGUNA_GLOBAL_GQA2_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_GQA2_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_MIXED32_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_MIXED32_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_MIXED32_LOCAL512_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_DPP_QK_PROBABILITY_VEC4_PRENORM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXEDSHAPE",
    "LAGUNA_GLOBAL_DENSE_PREFIX",
    "LAGUNA_GLOBAL_DENSE_PREFIX_IDLE_DOUBLE_BUFFER",
    "LAGUNA_GLOBAL_LOCAL1024",
    "LAGUNA_PREFILL_ATTENTION_HIPBLASLT",
    "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_OUTPUT_GATE",
    "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERY_PRODUCER",
    "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_PACKED_QUERIES",
    "LAGUNA_PREFILL_ATTENTION_HIPBLASLT_WAVE_ROWS_SOFTMAX",
    "LAGUNA_PREFILL_BLOCK_ATTENTION_HIPBLASLT",
    "LAGUNA_PREFILL_DENSE_CONTIGUOUS_CACHE",
    "LAGUNA_PREFILL_GLOBAL_ATTENTION_ROWS",
    "LAGUNA_PREFILL_LONG_ATTENTION_HIPBLASLT",
    "LAGUNA_PREFILL_SWA_ATTENTION_HIPBLASLT",
    "LAGUNA_PREFILL_CACHED_META",
    "LAGUNA_PREFILL_KV_PREAPPEND",
    "LAGUNA_PREFILL_MATRIX_ROWS",
    "LAGUNA_Q6_QMICRO",
    "LAGUNA_Q6_QMICRO_PLANAR",
    "LAGUNA_Q6_QMICRO_PERMUTE",
    "LAGUNA_ROUTER_PROJECTION_WAVE0_TREE",
    "LAGUNA_ROUTER_LOGITS_MODE",
    "LAGUNA_SELECTED_DOWN_MODE",
    "LAGUNA_SELECTED_GATE_UP_MODE",
    "LAGUNA_SPLIT_GATE_FUSION",
    "LAGUNA_SWA_SPLIT_MIN_LIVE",
    "LAGUNA_SWA_SPLIT_GQA3_SCORES",
    "LAGUNA_SWA_FUSED_FIXED512",
    "LAGUNA_SWA_GQA3_LOCAL384_FIXED512",
    "LAGUNA_SWA_GQA3_VSTAGE64_FIXED512",
    "LAGUNA_SWA_GQA3_VSTAGE64_VEC16_FIXED512",
    "LAGUNA_SWA_GQA3_VSTAGE64_VEC16_DIRECT_FIXED512",
    "LAGUNA_SWA_GQA3_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP4_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP8_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP16_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP32_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VEC4_DENOM_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED32_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED40_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED40_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_MIXED40_LOCAL512_EXP32_PRODUCER_MAX_GATE_STAGE_PCACHE_TAIL_PRODUCER_VALUE_TAIL_IDLE_VEC4_DENOM_PROBABILITY_VSTAGE64_VEC16_DIRECT_ASSUME_EXP_FIXED512",
    "LAGUNA_SWA_DENSE_RING",
    "LAGUNA_SWA_LOCAL1024",
    "LAGUNA_SWA_OUTPUT_SHARDED_PROBABILITY_DPP_QK",
    "LAGUNA_SWA_SPLIT_FIXED512_REDUCE",
    "LAGUNA_SWA_SPLIT_TILE16_MIN_LIVE",
    "LAGUNA_SWA_SPLIT_WAVE_LOCAL",
    "LAGUNA_SWA_PREFILL_ROLE_VARIANTS",
    "LAGUNA_SWA_PREFILL_VARIANT",
    "PARO_FULL_ATTN_NATIVE_EXACT_WIDTHS",
    "PARO_NATIVE_BATCH_DECODE_DEFAULT",
    "PARO_RETAINED_BATCH_DEFAULTS",
    "TARGET_ARCH",
    "register_backend_kernels",
    "register_gfx1151_kernels",
]
