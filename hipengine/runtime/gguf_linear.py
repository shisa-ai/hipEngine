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
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_pack8_gemv import (
    register_gguf_q6_k_pack8_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_q6_k_t16_gemv import (
    register_gguf_q6_k_t16_gemv_kernels,
)
from hipengine.kernels.hip_gfx1100.quant.gguf_k_mmq_prefill import (
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
    LAYOUT_GGUF_Q6_K_T16,
    LAYOUT_GGUF_Q8_0_T16,
    LAYOUT_Q4_K_PACK8,
    LAYOUT_RAW_GGUF,
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

# Small-B weight-amortized row-tile GEMV for raw K-quants and resident-pack8
# Q4_K verifier continuation blocks. Default ON: every specialization preserves
# the corresponding per-row arithmetic. The opt-out exists only for bisection;
# set HIPENGINE_GGUF_Q4K_ROWTILE=0 to disable.
_Q4K_ROWTILE_ENV = "HIPENGINE_GGUF_Q4K_ROWTILE"
_q4k_rowtile_session_enabled: bool | None = None
_ROWTILE_MIN_ROWS = 2
_ROWTILE_MAX_ROWS = 8
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
    # P10.B4: Q8T16 dense WMMA prefill consumes T16 tiles with 32 K-values
    # per tile slab. Same block alignment as raw Q8_0.
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
    (LAYOUT_DENSE_F32, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_BF16): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "dense_gemv", "f32", "bf16_hidden_bf16_out"),
        "dense_bf16",
    ),
    (LAYOUT_DENSE_F32, GGUF_ACTIVATION_F32, GGUF_OUTPUT_F32): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "dense_gemv", "f32", "f32_hidden_f32_out"),
        "dense_bf16",
    ),
    (LAYOUT_GGUF_Q6_K_T16, GGUF_ACTIVATION_BF16, GGUF_OUTPUT_F32): GGUFLinearDispatch(
        KernelKey("hip_gfx1100", "linear", "gguf_q6_k_t16_v1", "t16_gemv_decode_bf16_f32_out"),
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


def clear_gguf_linear_dispatch_cache() -> None:
    """Drop all memoized GGUF linear dispatch resolutions.

    Not normally needed (the registry generation in the cache key invalidates
    stale entries automatically); exposed for tests and defensive callers.
    """

    _DISPATCH_RESOLVE_CACHE.clear()
    _PAIR_DISPATCH_RESOLVE_CACHE.clear()


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
    f_gemv = _resolve_use_gemv_decode(use_gemv_decode)
    use_wmma = _resolve_use_wmma_prefill(use_wmma_prefill)
    f_rowtile = (not use_wmma) and _resolve_use_q4k_rowtile(None)
    raw_k_rowbatch = raw_k_prefill_rowbatch()
    raw_k_variant = raw_k_prefill_variant()
    mmq_session = _q8_mmq_prefill_session.get()
    q5_f32_ordered_session = _q5_f32_ordered_prefill_session.get()
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
        registered_variant,
        bool(_native_batch_decode_session_enabled),
        None if mmq_session is None else id(mmq_session),
        (
            None
            if q5_f32_ordered_session is None
            else id(q5_f32_ordered_session)
        ),
        raw_weight_ptr,
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
        dispatch = _native_batch_decode_dispatch(dispatch, rows=rows)
        # The small-B row-tile path is the weight-amortized replacement for the
        # per-row (non-WMMA) prefill alias. It does not override an explicit WMMA
        # opt-in: only fires when WMMA is off (e.g. the small-B target verifier).
        dispatch = _wmma_prefill_dispatch(
            dispatch,
            rows=rows,
            in_features=in_features,
            use_wmma=use_wmma,
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
            enabled=use_q4_pack8_wmma,
        )
        _ensure_linear_kernel_registered(dispatch.key)
        fn = resolve(
            backend=dispatch.key.backend,
            layer=dispatch.key.layer,
            quant=dispatch.key.quant,
            variant=dispatch.key.variant,
        )
        cached = (dispatch.abi, fn, dispatch.key.quant, dispatch.key.variant)
        _DISPATCH_RESOLVE_CACHE[cache_key] = cached
    abi, fn, quant, variant = cached
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
    _LAUNCH_ABI[abi](fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs)


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
    """Launch an exact Q4T16 decode sidecar when one is resident."""

    if (
        not enabled
        or rows != 1
        or weight.spec.quant_key != "gguf_q4_k"
        or "decode_tiles" not in weight.allocations
    ):
        return False
    resolved_backend = _weight_backend(weight, backend=backend)
    key = KernelKey(
        resolved_backend,
        "linear",
        "gguf_q4_k_t16_v1",
        "dense_single_local32_bf16_bf16_out",
    )
    _ensure_linear_kernel_registered(key)
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
        weight.allocation("decode_tiles").tensor.ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        **kwargs,
    )
    return True


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
    unequal-width F32 output), Q8_0 dual decode GEMV, Q4_K pack8 dual prefill,
    and the P8.2 raw-Q4_K dual WMMA prefill. Populated resident-pack8 pairs
    decline the legacy dual owner when the exact tile8x8 singleton is registered.
    There is still no Q8_0 dual WMMA
    prefill; when ``use_wmma_prefill`` would otherwise route Q8_0 rows>1 to
    the WMMA family, the pair function returns ``False`` so the caller falls
    back to two singletons that each take the WMMA path via
    :func:`launch_gguf_linear`.

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
) -> bool:
    """Launch an exact registered gate/up pair plus SiLU, or return False."""

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
        "pack8_dual_decode_bf16_bf16_out",
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
    if use_wmma and rows > 1:
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


def _native_batch_decode_dispatch(
    dispatch: GGUFLinearDispatch,
    *,
    rows: int,
) -> GGUFLinearDispatch:
    """Select the exact raw pack8 GEMV family for compact c=2/4/8 decode."""

    if not _native_batch_decode_session_enabled or rows <= 1 or rows > 8:
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
        and dispatch.key.quant.endswith("_t16_v1")
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
        and dispatch.key.variant == "pack8_prefill_bf16_bf16_out"
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
    if resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
        missing="none",
    ) is not None:
        return
    register_dense_gemv_kernels()
    register_gguf_k_gemv_kernels()
    register_gguf_k_mmq_prefill_kernels()
    register_gguf_q4_k_gemv_kernels()
    register_gguf_q4_k_prefill_kernels()
    register_gguf_q4_k_pack8_gemv_kernels()
    register_gguf_q5_k_f32_rocblas_prefill_kernels()
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


_LAUNCH_ABI = {
    "dense_bf16": _launch_dense_bf16,
    "pack8": _launch_pack8,
    "raw": _launch_raw,
    "raw_mmq_d4x3": _launch_raw_mmq_d4x3,
    "raw_k_f32_ordered": _launch_raw_k_f32_ordered,
    "raw_k_f32_ordered_activation_tile_k_row": (
        _launch_raw_k_f32_ordered_activation_tile_k_row
    ),
    "raw_k_f32_resident_activation_tile_k_row": (
        _launch_raw_k_f32_resident_activation_tile_k_row
    ),
    "t16": _launch_t16,
    "wmma_raw": _launch_wmma_raw,
}


__all__ = [
    "GGUF_ACTIVATION_BF16",
    "GGUF_ACTIVATION_F32",
    "GGUF_OUTPUT_BF16",
    "GGUF_OUTPUT_FP16",
    "GGUF_OUTPUT_F32",
    "launch_gguf_q4_t16_sidecar_decode",
    "GGUFLinearDispatch",
    "Q5F32OrderedPrefillSession",
    "Q5F32ResidentPlane",
    "gguf_wmma_prefill_enabled",
    "launch_gguf_linear",
    "launch_gguf_linear_moe_tail_host_batch",
    "launch_gguf_linear_pair",
    "launch_gguf_linear_pair_silu",
    "launch_gguf_linear_pair_concat",
    "launch_gguf_linear_raw_ptr",
    "launch_gguf_linear_triple",
    "native_batch_decode_session",
    "q5_f32_ordered_prefill_session",
    "q8_mmq_prefill_session",
    "raw_k_prefill_rowbatch",
    "raw_k_prefill_rowbatch_session",
    "raw_k_prefill_variant",
    "raw_k_prefill_variant_session",
    "resolve_gguf_linear_dispatch",
    "resolve_q8_mmq_prefill_policy",
    "set_wmma_prefill_enabled",
    "wmma_prefill_session",
]
