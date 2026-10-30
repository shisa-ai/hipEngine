"""gfx1151 / Strix Halo backend registration.

The initial gfx1151 backend intentionally reuses the proven gfx11 HIP kernel
bodies from ``hip_gfx1100`` and compiles them as native ``gfx1151`` code objects
through ``HIPENGINE_HIP_ARCH=gfx1151`` / ``--offload-arch=gfx1151``.  This gives
Strix Halo a peer backend key while keeping tuning changes separate from the
source-lineage port.
"""

from __future__ import annotations

import sys

from importlib import import_module

from hipengine.kernels.backends import (
    backend_package_capability_from_module,
    hip_target_arch_for_backend,
)
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
    qwen35_router_logits_bf16_f32w,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_prefill import (
    gguf_q8_0_t16_wmma_prefill_auto_4wave_bf16_bf16_out,
    gguf_q8_0_t16_wmma_prefill_auto_4wave_f16w_bf16_bf16_out,
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
    gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_gfx1100_bf16_bf16_out,
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
from hipengine.kernels.hip_gfx1151 import policy_laguna as _policy_laguna
from hipengine.kernels.hip_gfx1151 import policy_capability as _policy_capability
from hipengine.kernels.hip_gfx1151 import policy_gguf as _policy_gguf

# Ordered owners of this backend's capability declarations, most specific
# first. This is the only place the owner list is written down: the module
# references below are derived from it, and the cold source reader reads the
# same declaration, so runtime and source ownership cannot drift apart.
# A name the package binds itself always wins over its owner.
POLICY_MODULE_NAMES = ("policy_capability", "policy_gguf", "policy_laguna")
_POLICY_MODULES = tuple(
    import_module(f"{__name__}.{_owner}") for _owner in POLICY_MODULE_NAMES
)

# The live package module object (not a snapshot or proxy). Backend functions
# resolve policy settings at each reached operand, so a package reassignment is
# honored and the declared owner is consulted only while the binding is absent.
_SELF = sys.modules[__name__]


def _capability(name):
    """Read one policy setting: package binding first, then its declared owner."""

    missing = object()
    value = backend_package_capability_from_module(_SELF, name, missing)
    if value is missing:
        raise NameError(name)
    return value

# Architecture identity is part of the backend contract and stays on the
# package. Every other policy value is read from its declared owner through
# ``backend_package_capability`` / ``_capability``; there are no policy
# re-export aliases here.
TARGET_ARCH = _policy_capability.TARGET_ARCH


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
        row_count <= _capability("GGUF_Q4_T16_PHYSICAL_SMALLM_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_PHYSICAL_SMALLM_SHAPES")
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
        17 <= row_count <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWVGPR48_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWVGPR64_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWVGPR80_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWVGPR96_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_PLAIN128_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWVGPR128_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWVGPR48_128_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR144_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWVGPR48_144_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR144_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_SHARED2_192_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_SHARED2W4_192_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b2w4_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR144_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_SHARED2_192_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_SHARED2W2_192_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b2w2_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_SHARED2_192_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_SHARED2_256_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_SHARED2W2_256_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b2w2_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_SHARED3W8R3_384_MIN_ROWS")
        <= row_count
        <= _capability("GGUF_Q4_T16_DENSE_SHARED2_384_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_SHARED3W8R3_384_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b3w8r3_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_SHARED2_256_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_SHARED2_384_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_SHARED2W2_384_SHAPES")
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
        _capability("GGUF_Q5_T16_DENSE_SHARED8R3_MIN_ROWS") <= row_count
        <= _capability("GGUF_Q5_T16_DENSE_SHARED8R3_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_SHARED8R3_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_shared8r3_bf16_bf16_out
    elif (
        17 <= row_count <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR48_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWVGPR_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR48_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWVGPR48_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR48_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR64_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWVGPR64_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR64_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWVGPR80_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR64_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR96_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWVGPR96_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR96_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR96_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR128_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWVGPR48_128_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR128_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR144_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWM_SHAPES")
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
        row_count <= _capability("GGUF_Q4_T16_PHYSICAL_SMALLM_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_PHYSICAL_SMALLM_SHAPES")
    ):
        fn = (
            gguf_q4_k_t16_wmma_prefill_shared_b2w2_fp16_in_bf16_out
            if shape == (17_408, 5_120)
            else gguf_q4_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
        )
    elif (
        17 <= row_count <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWVGPR48_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWVGPR64_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWVGPR80_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWVGPR96_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_PLAIN128_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWVGPR128_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR96_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWVGPR48_128_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR128_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR144_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWVGPR48_144_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR144_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_SHARED2_192_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_SHARED2W4_192_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b2w4_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR144_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_SHARED2_192_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_SHARED2W2_192_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b2w2_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_SHARED2_192_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_SHARED2_256_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_SHARED2W2_256_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b2w2_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_SHARED3W8R3_384_MIN_ROWS")
        <= row_count
        <= _capability("GGUF_Q4_T16_DENSE_SHARED2_384_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_SHARED3W8R3_384_SHAPES")
    ):
        fn = gguf_q4_k_t16_wmma_prefill_shared_b3w8r3_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_SHARED2_256_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_SHARED2_384_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_SHARED2W2_384_SHAPES")
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
        _capability("GGUF_Q5_T16_DENSE_SHARED8R3_MIN_ROWS") <= row_count
        <= _capability("GGUF_Q5_T16_DENSE_SHARED8R3_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_SHARED8R3_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_shared8r3_fp16_in_bf16_out
    elif (
        17 <= row_count <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR48_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWVGPR_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR48_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWVGPR48_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR48_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR64_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWVGPR64_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR64_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWVGPR80_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR64_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR96_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWVGPR96_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR80_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR96_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR96_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR128_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWVGPR48_128_SHAPES")
    ):
        fn = gguf_q5_k_t16_wmma_prefill_lowvgpr48_fp16_in_bf16_out
    elif (
        _capability("GGUF_Q5_T16_DENSE_LOWVGPR128_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q5_T16_DENSE_LOWVGPR144_MAX_ROWS")
        and shape in _capability("GGUF_Q5_T16_DENSE_LOWM_SHAPES")
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
        if _capability("GGUF_Q6_PREFILL_SHARED3R1_MIN_ROWS") <= row_count
        <= _capability("GGUF_Q6_PREFILL_SHARED3R1_MAX_ROWS")
        and shape in _capability("GGUF_Q6_STANDARD_PREFILL_SHARED3R1_SHAPES")
        else gguf_q6_k_t16_wmma_prefill_shared6r1_bf16_bf16_out
        if _capability("GGUF_Q6_STANDARD_PREFILL_SHARED6R1_MIN_ROWS") <= row_count
        <= _capability("GGUF_Q6_STANDARD_PREFILL_SHARED6R1_MAX_ROWS")
        and shape in _capability("GGUF_Q6_STANDARD_PREFILL_SHARED3R1_SHAPES")
        else gguf_q6_k_t16_wmma_prefill_shared8r3_bf16_bf16_out
        if _capability("GGUF_Q6_STANDARD_PREFILL_SHARED8R3_MIN_ROWS") <= row_count
        <= _capability("GGUF_Q6_STANDARD_PREFILL_SHARED8R3_MAX_ROWS")
        and shape in _capability("GGUF_Q6_STANDARD_PREFILL_SHARED4_SHAPES")
        else gguf_q6_k_t16_wmma_prefill_shared8r3_bf16_bf16_out
        if _capability("GGUF_Q6_STANDARD_PREFILL_SHARED8R3_HIGH_MIN_ROWS") <= row_count
        <= _capability("GGUF_Q6_STANDARD_PREFILL_SHARED8R3_HIGH_MAX_ROWS")
        and shape in _capability("GGUF_Q6_STANDARD_PREFILL_SHARED4_SHAPES")
        else gguf_q6_k_t16_wmma_prefill_shared4_bf16_bf16_out
        if row_count >= _capability("GGUF_Q6_STANDARD_PREFILL_SHARED4_MIN_ROWS")
        and shape in _capability("GGUF_Q6_STANDARD_PREFILL_SHARED4_SHAPES")
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
    six physical shapes use shared4 from row256, except the wide down shape,
    whose rows288+ bands follow its own 256-4096 row screen. Shape misses
    keep plain.
    """

    row_count = int(rows)
    shape = (int(in_features), int(out_features))
    if (
        row_count == _capability("GGUF_Q6_PLANAR_PREFILL_SHARED4R9_ROWS")
        and shape == _capability("GGUF_Q6_PLANAR_PREFILL_SHARED4R9_SHAPE")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r9_bf16_bf16_out
    elif (
        _capability("GGUF_Q6_PLANAR_PREFILL_SHARED4R6_MIN_ROWS") <= row_count
        <= _capability("GGUF_Q6_PLANAR_PREFILL_SHARED4R6_MAX_ROWS_BY_SHAPE").get(shape, -1)
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r6_bf16_bf16_out
    elif (
        _capability("GGUF_Q6_PLANAR_PREFILL_SHARED4R4_MIN_ROWS") <= row_count
        <= _capability("GGUF_Q6_PLANAR_PREFILL_SHARED4R4_MAX_ROWS")
        and shape in _capability("GGUF_Q6_PLANAR_PREFILL_SHARED4R4_SHAPES")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r4_bf16_bf16_out
    elif (
        row_count >= _capability("GGUF_Q6_PLANAR_PREFILL_SHARED4_GFX1100_MIN_ROWS")
        and shape in _capability("GGUF_Q6_PLANAR_PREFILL_SHARED4_GFX1100_SHAPES")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_gfx1100_bf16_bf16_out
    elif (
        _capability("GGUF_Q6_PREFILL_SHARED3R1_MIN_ROWS") <= row_count
        <= _capability("GGUF_Q6_PREFILL_SHARED3R1_MAX_ROWS")
        and shape in _capability("GGUF_Q6_PLANAR_PREFILL_SHARED3R1_SHAPES")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared3r1_bf16_bf16_out
    elif (
        row_count == _capability("GGUF_Q6_PLANAR_PREFILL_SHARED4R4_ROWS")
        and shape in _capability("GGUF_Q6_PLANAR_PREFILL_SHARED4R4_SHAPES")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r4_bf16_bf16_out
    elif (
        row_count == _capability("GGUF_Q6_PLANAR_PREFILL_SHARED4R3_ROWS")
        and shape in _capability("GGUF_Q6_PLANAR_PREFILL_SHARED4R3_SHAPES")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4r3_bf16_bf16_out
    elif (
        17 <= row_count <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS")
        and shape == (5_120, 17_408)
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR48_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS")
        and shape in _capability("GGUF_Q6_PLANAR_LOWVGPR80_SHAPES")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWVGPR64_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q6_PLANAR_LOWVGPR96_MAX_ROWS")
        and shape in _capability("GGUF_Q6_PLANAR_LOWVGPR96_SHAPES")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        _capability("GGUF_Q4_T16_DENSE_LOWM_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q6_PLANAR_LOWVGPR96_MAX_ROWS")
        and shape in _capability("GGUF_Q4_T16_DENSE_LOWM_SHAPES")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        _capability("GGUF_Q6_PLANAR_LOWVGPR96_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q6_PLANAR_LOWVGPR128_MAX_ROWS")
        and shape in _capability("GGUF_Q6_PLANAR_LOWVGPR128_SHAPES")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr_bf16_bf16_out
    elif (
        _capability("GGUF_Q6_PLANAR_LOWVGPR96_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q6_PLANAR_LOWVGPR128_MAX_ROWS")
        and shape in _capability("GGUF_Q6_PLANAR_LOWVGPR48_128_SHAPES")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        _capability("GGUF_Q6_PLANAR_LOWVGPR96_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q6_PLANAR_LOWVGPR128_MAX_ROWS")
        and shape in _capability("GGUF_Q6_PLANAR_SHARED4_128_SHAPES")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out
    elif (
        _capability("GGUF_Q6_PLANAR_LOWVGPR128_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q6_PLANAR_LOWVGPR144_MAX_ROWS")
        and shape in _capability("GGUF_Q6_PLANAR_LOWVGPR48_144_SHAPES")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_lowvgpr48_bf16_bf16_out
    elif (
        _capability("GGUF_Q6_PLANAR_LOWVGPR128_MAX_ROWS")
        < row_count
        <= _capability("GGUF_Q6_PLANAR_LOWVGPR144_MAX_ROWS")
        and shape in _capability("GGUF_Q6_PLANAR_SHARED4_144_SHAPES")
    ):
        fn = gguf_q6_k_t16_qmicro_planar_wmma_prefill_shared4_bf16_bf16_out
    elif (
        row_count >= _capability("GGUF_Q6_PLANAR_PREFILL_SHARED4_MIN_ROWS")
        and shape in _capability("GGUF_Q6_PLANAR_PREFILL_SHARED4_SHAPES")
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
        in _capability("_GGUF_Q5_T16_ROWTILE_COL8_SHAPES")
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


_SOURCE_BACKEND = "hip_gfx1100"
# Native speculative-cycle providers use dedicated backend registrations rather
# than this generic shared-body alias refresh. The GGUF target launcher has an
# independent gfx1151 parity gate; the proposal graph remains unadmitted here.
_GFX1151_ALIAS_EXCLUSIONS = frozenset(
    {
        # The dense-H5120 narrow Q4T16 col4 rowtile and the dense-H5120
        # down+residual rowtile were admitted on 2026-09-12: independent gfx1151
        # shape crossovers (rows 2-4, bit-exact) and complete-model gates now
        # exist. See
        # benchmarks/results/2026-09-12-gfx1151-qwen38-27b-q4km-dense-rowtile-withheld-variants-qualified.json.
        # The exact scalar F32 alpha/beta pair is independently admitted on
        # gfx1151; wider/native rows retain singleton projections.
        # The dependent alpha/beta-to-GDN owner is screened only on gfx1100;
        # gfx1151 retains scalar projections plus snapshot chain GDN.
        (
            "linear_attn_alpha_beta+gdn_chain_recurrent_rmsnorm_gate+cast+snapshot",
            "f32+gguf_q5_k_t16_v1",
            "bf16_k5120_n48_hk16_hv48_d128_exact_state_rows_tloop_f32_bf16_out",
        ),
        # The rounded next-input RMSNorm fusion is W7900-only pending an
        # independent gfx1151 boundary/model gate. Its sibling dense-H5120
        # down+residual rowtile is admitted (2026-09-12 qualification).
        (
            "add+rmsnorm",
            "gguf_f32_weight",
            "rounded_bf16_out",
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
        ("activation_quant", "q8_1_d4s4_f32", "bf16_kmajor"),
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
        (
            "linear",
            "gguf_q5_k",
            "mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_bf16_out",
        ),
        (
            "linear",
            "gguf_q5_k",
            "mmq_i64_j16_j32_k256_q8_1_d4s4_f32_kmajor_bf16_f32_out",
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
    ): qwen35_router_logits_bf16_f32w,
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
    # The explicit two-operand form of the same launch: the schedule must stay
    # the four-wave one on this backend, or the f16w route would silently fall
    # back to the gfx1100 two-wave default.
    (
        "linear",
        "gguf_q8_0_t16_v1",
        "t16_wmma_prefill_f16w_bf16_bf16_out",
    ): gguf_q8_0_t16_wmma_prefill_auto_4wave_f16w_bf16_bf16_out,
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
    "hipengine.kernels.hip_gfx1100.attention.qwen4_exp_qsa_wmma",
    "hipengine.kernels.hip_gfx1100.convert",
    "hipengine.kernels.hip_gfx1100.fused",
    "hipengine.kernels.hip_gfx1100.linear",
    "hipengine.kernels.hip_gfx1100.linear.dense_f16_gemm",
    "hipengine.kernels.hip_gfx1100.linear_attn",
    "hipengine.kernels.hip_gfx1100.moe",
    "hipengine.kernels.hip_gfx1100.moe.maple_moe",
    "hipengine.kernels.hip_gfx1100.norm",
    "hipengine.kernels.hip_gfx1100.quant",
    "hipengine.kernels.hip_gfx1100.quant.gguf_k_selected_prefill",
    "hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill",
    "hipengine.kernels.hip_gfx1100.quant.gguf_q5_k_q8_1_selected_prefill",
    "hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_dense_wide",
    "hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill",
    "hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill",
    "hipengine.kernels.hip_gfx1100.quant.qwen4_exp_routed_f16",
    "hipengine.kernels.hip_gfx1100.quant.maple_ternary",
    "hipengine.kernels.hip_gfx1100.rotary",
    "hipengine.kernels.hip_gfx1100.runtime",
    "hipengine.kernels.hip_gfx1100.sampling",
    "hipengine.kernels.hip_gfx1100.smoke",
    "hipengine.kernels.hip_gfx1100.speculative",
    "hipengine.kernels.hip_gfx1100.wmma",
)


def import_gfx1100_kernel_modules() -> None:
    """Import every gfx1100 kernel module so import-time registrations run.

    Several gfx1100 kernel modules register their kernels at module import
    time. Importing a module is a once-per-process side effect, so callers that
    need those registrations to exist (the alias pass below, the unit-test
    conftest baseline snapshot) must trigger the imports through this helper
    instead of assuming a previous caller already did.
    """

    for module_name in _GFX1100_MODULES:
        import_module(module_name)


def register_gfx1151_kernels(*, replace: bool = False) -> None:
    """Register gfx1151 aliases for the current gfx1100 kernel key space."""

    import_gfx1100_kernel_modules()

    # LDS-staged f32 dense Q8_0 tile. A production-gate candidate, not a default:
    # it is measured 1.23x to 2.22x faster than the exact coltile leaf across all
    # six Q8_0 dense projections of a layer step (see
    # ``benchmarks/results/2026-09-29-dp8-staged-f32-dense-family/``), but its
    # output is NOT bit-identical to that leaf, so selecting it requires the
    # ``docs/EXECUTION-PROFILES.md`` production gate rather than strict parity.
    # Nothing selects it by default; see ``_raw_k_staged_f32_dispatch`` in
    # ``hipengine.runtime.gguf_linear``.
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_gr_staged_f32 import (
        gr_staged_f32_up,
    )

    staged_key = KernelKey(_capability("BACKEND"), "linear", "gguf_q8_0", _capability("STAGED_F32_VARIANT"))
    if replace or not is_registered(staged_key):
        register(staged_key, gr_staged_f32_up, replace=replace)

    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_prefill import (
        gguf_q8_0_selected_grouped_blockscale_prefill_bf16_bf16_out,
        gguf_q8_0_selected_grouped_blockscale_guarded_prefill_bf16_bf16_out,
    )
    blockscale_key = KernelKey(
        _capability("BACKEND"), "linear", "gguf_q8_0",
        "selected_grouped_blockscale_prefill_bf16_bf16_out",
    )
    if replace or not is_registered(blockscale_key):
        register(blockscale_key, gguf_q8_0_selected_grouped_blockscale_prefill_bf16_bf16_out,
                 replace=replace)
    guarded_key = KernelKey(
        _capability("BACKEND"), "linear", "gguf_q8_0",
        "selected_grouped_blockscale_guarded_prefill_bf16_bf16_out",
    )
    if replace or not is_registered(guarded_key):
        register(guarded_key, gguf_q8_0_selected_grouped_blockscale_guarded_prefill_bf16_bf16_out,
                 replace=replace)
    from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import (
        qwen4_exp_gdn_decode_rowsplit_f32,
        qwen4_exp_gdn_prefill_rowsplit_f32,
        qwen4_exp_gdn_prefill_tiled16_dpp_f32,
        qwen4_exp_gdn_prefill_tiled16_multi_f32,
    )
    from hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_gemv import (
        gguf_q4_k_selected_dual_q8_1_dp4a_silu_warp256_gemv_bf16_bf16_out,
    )
    from hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 import (
        qwen4_exp_q5_1_selected_weighted_sum_warp256_fast_bf16_bf16_out,
    )
    for warp_key, warp_fn in (
        (KernelKey(_capability("BACKEND"), "linear", "gguf_q4_k",
                   "selected_dual_q8_1_dp4a_silu_warp256_gemv_bf16_bf16_out"),
         gguf_q4_k_selected_dual_q8_1_dp4a_silu_warp256_gemv_bf16_bf16_out),
        (KernelKey(_capability("BACKEND"), "linear", "gguf_q5_1",
                   "selected_weighted_sum_warp256_fast_bf16_bf16_out"),
         qwen4_exp_q5_1_selected_weighted_sum_warp256_fast_bf16_bf16_out),
    ):
        if replace or not is_registered(warp_key):
            register(warp_key, warp_fn, replace=replace)
    rowsplit_key = KernelKey(
        _capability("BACKEND"), "gdn_recurrence_norm_gate", "f32_state",
        "qwen4exp_gdn_rowsplit_prefill",
    )
    if replace or not is_registered(rowsplit_key):
        register(rowsplit_key, qwen4_exp_gdn_prefill_rowsplit_f32, replace=replace)
    rowsplit_decode_key = KernelKey(
        _capability("BACKEND"), "gdn_recurrence_norm_gate", "f32_state",
        "qwen4exp_gdn_rowsplit_decode",
    )
    if replace or not is_registered(rowsplit_decode_key):
        register(
            rowsplit_decode_key, qwen4_exp_gdn_decode_rowsplit_f32, replace=replace
        )
    dpp_key = KernelKey(
        _capability("BACKEND"), "gdn_recurrence_norm_gate", "f32_state",
        "qwen4exp_gdn_tiled16_dpp_prefill",
    )
    if replace or not is_registered(dpp_key):
        register(dpp_key, qwen4_exp_gdn_prefill_tiled16_dpp_f32, replace=replace)
    multi_key = KernelKey(
        _capability("BACKEND"), "gdn_recurrence_norm_gate", "f32_state",
        "qwen4exp_gdn_tiled16_multi_prefill",
    )
    if replace or not is_registered(multi_key):
        register(multi_key, qwen4_exp_gdn_prefill_tiled16_multi_f32, replace=replace)
    from hipengine.kernels.hip_gfx1100.fused.qwen4_exp_gr import (
        qwen4_exp_grouped_rmsnorm_bf16_f32_f16,
    )
    publish_key = KernelKey(
        _capability("BACKEND"), "gr_grouped_rmsnorm", "bf16_f32", "f32_f16_publish",
    )
    if replace or not is_registered(publish_key):
        register(publish_key, qwen4_exp_grouped_rmsnorm_bf16_f32_f16, replace=replace)

    from hipengine.kernels.hip_gfx1100.timesfm.registered import (
        register_timesfm_kernels as register_timesfm_shared,
    )
    timesfm_keys = register_timesfm_shared(_capability("BACKEND"), replace=replace)
    source_keys = [key for key in registered_keys() if key.backend == _SOURCE_BACKEND]
    for key in source_keys:
        if (key.layer, key.quant, key.variant) in _GFX1151_ALIAS_EXCLUSIONS:
            continue
        target_key = KernelKey(_capability("BACKEND"), key.layer, key.quant, key.variant)
        # TimesFM launchers bind an architecture; do not alias gfx1100 closures.
        if target_key in timesfm_keys:
            continue
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
        _capability("BACKEND"),
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
        key = KernelKey(_capability("BACKEND"), "linear", "gguf_q4_k_t16_v1", variant)
        if replace or not is_registered(key):
            register(key, fn, replace=replace)
    q4_pack8_decode_pair_key = KernelKey(
        _capability("BACKEND"),
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
        _capability("BACKEND"),
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
        _capability("BACKEND"),
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
        _capability("BACKEND"),
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
        _capability("BACKEND"),
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
    "TARGET_ARCH",
    "register_backend_kernels",
    "register_gfx1151_kernels",
]


def register_vibevoice_kernels():
    from hipengine.kernels.hip_gfx1100.vibevoice.registered import register_vibevoice_kernels as register_shared
    register_shared('hip_gfx1151')


def register_vibevoice_q4_kernels():
    from hipengine.kernels.hip_gfx1100.vibevoice.q4 import register_vibevoice_q4_kernels as register_q4
    register_q4('hip_gfx1151')


def register_timesfm_kernels():
    from hipengine.kernels.hip_gfx1100.timesfm.registered import register_timesfm_kernels as register_shared
    register_shared("hip_gfx1151")


register_timesfm_kernels()
register_vibevoice_kernels()
