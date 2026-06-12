"""Runtime glue for the MoE C1 C dispatcher (M14.dispatch.1-beta).

Bridges the ``Qwen35ParoLayerRuntime.run_moe_c1_fp16`` Python path to the
two extern-C dispatcher entry points in
``hipengine/kernels/hip_gfx1100/dispatch/moe_c1_dispatch.hip``.  Resolves all
14 underlying kernel-launcher function pointers once at warmup, snapshots all
layer-constant weight pointers and dims into a ``MoeC1Args`` struct, and at
runtime updates only the per-call variables before making a single ctypes
call.

Cache strategy: one ``MoeC1DispatchCache`` per (LayerRuntime, scratch)
shape.  The shapes that share a scratch are stable across the model's
lifetime, so a per-layer dict on the ``LayerRuntime`` is enough.

Gate: enabled by default; set ``HIPENGINE_MOE_C1_C_DISPATCH={0,off,no,false}`` to disable.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hipengine.core.tensor import Tensor
from hipengine.kernels.hip_gfx1100.dispatch.moe_c1 import (
    MoeC1Args,
    MoeC1Fns,
    build_moe_c1_dispatch,
    dispatch_full_fp16,
    dispatch_linear_fp16,
)

if TYPE_CHECKING:
    from hipengine.runtime.qwen35_paro import (
        Qwen35ParoLayerRuntime,
        Qwen35ParoMoeScratch,
    )


# ---------- Process-global singletons ----------
#
# These are shared across every layer and every model.  The dispatcher
# library, the 5 underlying kernel libraries, and the resolved function-
# pointer table are all the same for every call.  Caching them at module
# level avoids the ~180 ms/layer cost of re-running ``build_X(load=True)``
# at every MoeC1DispatchCache construction (the warmup hot path).
_DISPATCH_LIBRARY: ctypes.CDLL | None = None
_FNS_TABLE: MoeC1Fns | None = None


def _get_dispatch_library() -> ctypes.CDLL:
    global _DISPATCH_LIBRARY
    if _DISPATCH_LIBRARY is None:
        _DISPATCH_LIBRARY = build_moe_c1_dispatch(load=True)
    return _DISPATCH_LIBRARY


def _get_fns_table() -> MoeC1Fns:
    global _FNS_TABLE
    if _FNS_TABLE is None:
        _FNS_TABLE = _build_fns_table()
    return _FNS_TABLE


def moe_c1_c_dispatch_enabled() -> bool:
    """Master gate.  Default on after M14.dispatch.1 prewarm validation."""

    value = os.environ.get("HIPENGINE_MOE_C1_C_DISPATCH")
    if value is None or value.strip() == "":
        return True
    return value.strip().lower() not in {"0", "off", "no", "false", "disable", "disabled"}


def _w4_dual_output_tiled_split_prefill_enabled() -> bool:
    """M16.4 split-output shared gate/up prefill route.

    Default-on after the 2026-06-11 W7900 D32 9-prompt gate proved the stacked
    P1 route exact and non-regressive. Keep the opt-out for bisection.
    """

    value = os.environ.get("HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL")
    if value is None or value.strip() == "":
        return True
    return value.strip().lower() in {"1", "on", "yes", "true", "enable", "enabled"}


def _w4_output_tiled_prefill_enabled() -> bool:
    value = os.environ.get("HIPENGINE_W4_OUTPUT_TILED_PREFILL")
    if value is None or value.strip() == "":
        return True
    return value.strip().lower() not in {"0", "off", "no", "false", "disable", "disabled"}


def _linear_shared_down_combine_fused_enabled() -> bool:
    """Reduced-DAG experiment: fuse linear shared-down output-tiled W4 with combine.

    Default-on after the 2026-06-12 W7900 D32 9-prompt gate proved that removing
    the 30 linear-attn combine launches beats the parallel epilogue cost. Opt
    out with ``HIPENGINE_LINEAR_SHARED_DOWN_COMBINE_FUSED=0``.
    """

    value = os.environ.get("HIPENGINE_LINEAR_SHARED_DOWN_COMBINE_FUSED")
    if value is None or value.strip() == "":
        return True
    return value.strip().lower() not in {"0", "off", "no", "false", "disable", "disabled"}


def _full_shared_down_combine_fused_enabled() -> bool:
    """Reduced-DAG full-attn shared-down output-tiled W4 + combine.

    Default-on after the 2026-06-12 W7900 D32 9-prompt gate proved that
    removing the remaining 10 full-attn combine launches is exact and
    same-suite positive. Opt out with
    ``HIPENGINE_FULL_SHARED_DOWN_COMBINE_FUSED=0``.
    """

    value = os.environ.get("HIPENGINE_FULL_SHARED_DOWN_COMBINE_FUSED")
    if value is None or value.strip() == "":
        return True
    return value.strip().lower() not in {"0", "off", "no", "false", "disable", "disabled"}


def _linear_shared_silu_rotate_fused_enabled() -> bool:
    """Fuse linear shared SiLU and down-rotate in the C dispatcher.

    Default-on after the 2026-06-12 W7900 D32 9-prompt gate proved this small
    reduced-DAG slice exact and same-suite positive. Opt out with
    ``HIPENGINE_LINEAR_SHARED_SILU_ROTATE_FUSED=0``.
    """

    value = os.environ.get("HIPENGINE_LINEAR_SHARED_SILU_ROTATE_FUSED")
    if value is None or value.strip() == "":
        return True
    return value.strip().lower() not in {"0", "off", "no", "false", "disable", "disabled"}


def _linear_shared_down_mode() -> int:
    value = os.environ.get("HIPENGINE_W4_DOWN_PROJ_SMALL_BATCH")
    if value is None or value.strip() == "":
        return 2 if _w4_output_tiled_prefill_enabled() else 1
    mode = value.strip().lower().replace("-", "_")
    aliases = {
        "0": "prefill",
        "off": "prefill",
        "false": "prefill",
        "no": "prefill",
        "decode": "gemv",
        "decode_gemv": "gemv",
        "single": "gemv",
        "single_gemv": "gemv",
        "1": "multi_row_decode",
        "on": "multi_row_decode",
        "true": "multi_row_decode",
        "yes": "multi_row_decode",
        "multi_row_exact": "multi_row_decode",
        "decode_multi_row": "multi_row_decode",
        "multi_row_gemv": "multi_row_decode",
    }
    mode = aliases.get(mode, mode)
    if mode == "prefill":
        return 0
    if mode == "gemv":
        return 3
    if mode == "multi_row":
        return 4
    if mode == "multi_row_decode":
        return 1
    raise ValueError("HIPENGINE_W4_DOWN_PROJ_SMALL_BATCH must be prefill, gemv, multi_row, or multi_row_decode")


def prewarm_moe_c1_c_dispatch() -> None:
    """Resolve the dispatcher .so and shared function-pointer table eagerly.

    M14.dispatch.1-beta originally built these singletons lazily from the first
    verifier-layer call.  That kept steady-state fast but charged ~200 ms of
    one-time ctypes/build-cache work to verifier cycle 1, which the economics
    harness then amortized as an apparent +8 ms/cycle regression.  Resident
    sessions call this during build when the env gate is enabled, moving the
    one-time setup out of the measured decode/verify window.
    """

    _get_dispatch_library()
    _get_fns_table()


@dataclass(frozen=True)
class MoeC1DispatchSupport:
    """Which call patterns the C dispatcher can handle.

    The dispatcher mirrors a specific subset of ``run_moe_c1_fp16`` paths.
    When a call falls outside this subset, the layer must fall back to the
    existing Python pipeline.
    """

    layer_kind: str            # "linear_attention" | "full_attention"
    tokens_min: int = 2         # tokens=1 has coop router; dispatcher path is tokens>1
    paro_w4_shared: bool = True  # legacy w8a16 not supported here


class MoeC1DispatchCache:
    """One cache instance per (layer, layer-type) combination.

    Construction (warmup) resolves all 13 kernel-launcher function pointers
    via the existing build_X() functions and pre-fills the layer-constant
    fields of a ``MoeC1Args`` instance.  At runtime, ``dispatch(...)`` only
    updates the per-call fields (hidden, residual, out, tokens, stream,
    scratch ptrs) and invokes the C entry point.
    """

    __slots__ = (
        "_layer",
        "_layer_kind",
        "_runtime",
        "_dispatch_library",
        "_fns",
        "_args",
        "_small_batch_threshold",
    )

    def __init__(
        self,
        layer: "Qwen35ParoLayerRuntime",
        *,
        layer_kind: str,
        small_batch_threshold: int,
    ) -> None:
        if layer_kind not in {"linear_attention", "full_attention"}:
            raise ValueError(f"unsupported layer_kind: {layer_kind!r}")
        self._layer = layer
        self._layer_kind = layer_kind
        self._runtime = layer.runtime
        self._small_batch_threshold = small_batch_threshold
        # Use module-global singletons; first cache build pays the warmup
        # cost (~200 ms total for library resolves), subsequent caches reuse.
        self._dispatch_library = _get_dispatch_library()
        self._fns = _get_fns_table()
        self._args = MoeC1Args()
        _fill_layer_constant_args(self._args, layer, layer_kind)

    @property
    def layer_kind(self) -> str:
        return self._layer_kind

    def supports_call(self, *, tokens: int) -> bool:
        """Return True iff a `run_moe_c1_fp16` call with these args matches the
        dispatcher's contract.
        """

        if tokens < 2:
            return False
        if self._layer_kind == "full_attention":
            # Full-attn dispatcher uses small-batch GEMV; only valid when
            # `small_batch = tokens <= _small_batch_decode_threshold()`.
            if tokens > self._small_batch_threshold:
                return False
        # Linear-attn dispatcher uses the prefill (non-small-batch) shared
        # expert path.  Always small_batch=False for linear-attn layers in
        # the current verifier config, so no extra check needed.
        return True

    def dispatch(
        self,
        *,
        hidden: Tensor,
        residual: Tensor,
        out: Tensor,
        scratch: "Qwen35ParoMoeScratch",
        tokens: int,
        group_size: int,
        stream: int,
        selected_rotate_fuse_barrier: Tensor | None = None,
        selected_rotate_barrier_target: int = 0,
        selected_rotate_barrier_epoch: int = 0,
        selected_down_barrier_target: int = 0,
        selected_down_barrier_epoch: int = 0,
    ) -> Tensor:
        """Update per-call args and invoke the C dispatcher.  Returns ``out``."""

        args = self._args
        # ---- Per-call variables ----
        args.hidden = hidden.ptr
        args.residual = residual.ptr
        args.out = out.ptr
        args.tokens = tokens
        args.stream = stream
        args.selected_rotate_barrier_target = int(selected_rotate_barrier_target)
        args.selected_rotate_barrier_epoch = int(selected_rotate_barrier_epoch)
        args.selected_down_barrier_target = int(selected_down_barrier_target)
        args.selected_down_barrier_epoch = int(selected_down_barrier_epoch)
        # ---- Scratch pointers (may change if scratch instance differs across
        # calls; the verifier reuses a persistent scratch so this is usually
        # the same address each time, but we refresh defensively) ----
        args.router_logits = scratch.router_logits.ptr
        args.selected_experts = scratch.selected_experts.ptr
        args.routing_weights = scratch.routing_weights.ptr
        args.gate_up_input = scratch.gate_up_input.ptr
        args.selected_rotate_fuse_barrier = 0 if selected_rotate_fuse_barrier is None else selected_rotate_fuse_barrier.ptr
        args.gate_up = scratch.gate_up.ptr
        args.down_input = scratch.down_input.ptr
        args.down_out = scratch.down_out.ptr
        args.shared_gate_input = scratch.shared_gate_input.ptr
        args.shared_up_input = scratch.shared_up_input.ptr
        if self._layer_kind == "linear_attention":
            args.shared_gate_out = scratch.shared_gate_out.ptr
            args.shared_up_out = scratch.shared_up_out.ptr
            args.shared_intermediate = scratch.shared_intermediate.ptr
        else:
            args.shared_up_packed = scratch.shared_up.ptr
        args.shared_down_input = scratch.shared_down_input.ptr
        args.shared_out = scratch.shared_out.ptr
        # ---- group_size can vary per call site (rare) ----
        args.group_size = group_size
        # ---- Invoke C ----
        if self._layer_kind == "linear_attention":
            dispatch_linear_fp16(
                self._dispatch_library, self._fns, args, runtime=self._runtime,
            )
        else:
            dispatch_full_fp16(
                self._dispatch_library, self._fns, args, runtime=self._runtime,
            )
        return out


# ---------- Internal helpers ----------


def _build_fns_table() -> MoeC1Fns:
    """Resolve all 13 kernel-launcher function pointers and pack into struct.

    Uses raw ``getattr(lib, symbol)`` for the function pointer; argtypes are
    not needed on the Python wrapper side because the C dispatcher casts each
    ``void*`` to the correct C signature via ``reinterpret_cast``.
    """

    from hipengine.kernels.hip_gfx1100.fused.paro_combine import build_paro_combine
    from hipengine.kernels.hip_gfx1100.fused.paro_silu import build_paro_silu
    from hipengine.kernels.hip_gfx1100.moe.router import build_qwen35_router
    from hipengine.kernels.hip_gfx1100.quant.paro_awq_gemv import build_paro_awq_gemv
    from hipengine.kernels.hip_gfx1100.rotary.paro_rotate import build_paro_rotate

    router_lib = build_qwen35_router(load=True)
    rotate_lib = build_paro_rotate(load=True)
    awq_lib = build_paro_awq_gemv(load=True)
    silu_lib = build_paro_silu(load=True)
    combine_lib = build_paro_combine(load=True)

    fns = MoeC1Fns()

    def addr(library: ctypes.CDLL, symbol: str) -> int:
        fn = getattr(library, symbol)
        return ctypes.cast(fn, ctypes.c_void_p).value or 0

    fns.router_topk_shared_out_fp16 = addr(
        router_lib, "hipengine_qwen35_router_topk_shared_out_fp16",
    )
    fns.paro_rotate1_fp16 = addr(rotate_lib, "hipengine_paro_rotate1_fp16")
    fns.paro_rotate2_fp16 = addr(rotate_lib, "hipengine_paro_rotate2_fp16")
    fns.gemv_awq_selected_dual_pack8_transposed_fp16 = addr(
        awq_lib, "hipengine_gemv_awq_selected_dual_pack8_transposed_fp16",
    )
    fns.gemv_awq_selected_dual_pack8_transposed_rotate_staged_keyed_fp16 = addr(
        awq_lib, "hipengine_gemv_awq_selected_dual_pack8_transposed_rotate_staged_keyed_fp16",
    )
    fns.gemv_awq_selected_pack8_transposed_fp16 = addr(
        awq_lib, "hipengine_gemv_awq_selected_pack8_transposed_fp16",
    )
    fns.gemv_awq_selected_pack8_transposed_silu_rotate_staged_keyed_fp16 = addr(
        awq_lib, "hipengine_gemv_awq_selected_pack8_transposed_silu_rotate_staged_keyed_fp16",
    )
    fns.silu_mul_dual_rotate_out_fp16 = addr(
        silu_lib, "hipengine_silu_mul_dual_rotate_out_fp16",
    )
    fns.silu_mul_separate_out_fp16 = addr(
        silu_lib, "hipengine_silu_mul_separate_out_fp16",
    )
    if _linear_shared_silu_rotate_fused_enabled():
        fns.silu_mul_pair_rotate_out_fp16 = addr(
            silu_lib, "hipengine_silu_mul_pair_rotate_out_fp16",
        )
    fns.awq_fusedw4_prefill_dual_fp16 = addr(
        awq_lib, "hipengine_awq_fusedw4_prefill_dual_fp16",
    )
    if _w4_dual_output_tiled_split_prefill_enabled():
        fns.gemv_awq_dual_pack8_output_tiled_split_transposed_fp16 = addr(
            awq_lib, "hipengine_gemv_awq_dual_pack8_output_tiled_split_transposed_fp16",
        )
    fns.awq_fusedw4_prefill_fp16 = addr(
        awq_lib, "hipengine_awq_fusedw4_prefill_fp16",
    )
    fns.gemv_awq_pack8_multi_row_transposed_fp16 = addr(
        awq_lib, "hipengine_gemv_awq_pack8_multi_row_transposed_fp16",
    )
    fns.gemv_awq_pack8_multi_row_decode_transposed_fp16 = addr(
        awq_lib, "hipengine_gemv_awq_pack8_multi_row_decode_transposed_fp16",
    )
    fns.gemv_awq_pack8_output_tiled_transposed_fp16 = addr(
        awq_lib, "hipengine_gemv_awq_pack8_output_tiled_transposed_fp16",
    )
    if _linear_shared_down_combine_fused_enabled():
        fns.gemv_awq_pack8_output_tiled_combine_residual_transposed_fp16 = addr(
            awq_lib, "hipengine_gemv_awq_pack8_output_tiled_combine_residual_transposed_fp16",
        )
    if _full_shared_down_combine_fused_enabled():
        fns.gemv_awq_pack8_output_tiled_combine_residual_full_transposed_fp16 = addr(
            awq_lib, "hipengine_gemv_awq_pack8_output_tiled_combine_residual_transposed_fp16",
        )
    fns.gemv_awq_dual_pack8_transposed_fp16 = addr(
        awq_lib, "hipengine_gemv_awq_dual_pack8_transposed_fp16",
    )
    fns.gemv_awq_pack8_transposed_fp16 = addr(
        awq_lib, "hipengine_gemv_awq_pack8_transposed_fp16",
    )
    fns.combine_batch_fp16 = addr(
        combine_lib,
        "hipengine_weighted_sum_shared_gate_combine_residual_batch_out_fp16_f32w",
    )
    return fns


def _fill_layer_constant_args(
    args: MoeC1Args,
    layer: "Qwen35ParoLayerRuntime",
    layer_kind: str,
) -> None:
    """Snapshot all layer-constant weight pointers + dims into ``args``.

    Called once at cache construction.  Reads ``layer.tensor(name)`` for each
    weight (which is a frozen address after model load).
    """

    cfg = layer.config
    layer_id = layer.layer_weights.layer_id
    moe_prefix = f"layers.{layer_id}.mlp"

    # Router combined weight
    args.router_combined_weight = layer.tensor(f"{moe_prefix}.router_shared_gate.weight").ptr

    # Selected gate-up (per-expert stacked W4)
    experts = f"{moe_prefix}.experts"
    gate_up_pairs = layer.tensor(f"{experts}.gate_up_weight_pairs")
    args.selected_gate_up_pairs = gate_up_pairs.ptr
    args.selected_gate_up_theta = layer.tensor(f"{experts}.gate_up_weight_theta").ptr
    args.selected_gate_up_channel_scales = layer.tensor(
        f"{experts}.gate_up_weight_channel_scales"
    ).ptr
    gate_qweight = layer.tensor(f"{experts}.stacked_gate_qweight_pack8_decode")
    args.selected_stacked_gate_qweight = gate_qweight.ptr
    args.selected_stacked_gate_qzeros = layer.tensor(f"{experts}.stacked_gate_qzeros").ptr
    args.selected_stacked_gate_scales = layer.tensor(f"{experts}.stacked_gate_scales").ptr
    up_qweight = layer.tensor(f"{experts}.stacked_up_qweight_pack8_decode")
    args.selected_stacked_up_qweight = up_qweight.ptr
    args.selected_stacked_up_qzeros = layer.tensor(f"{experts}.stacked_up_qzeros").ptr
    args.selected_stacked_up_scales = layer.tensor(f"{experts}.stacked_up_scales").ptr

    # Selected down (per-expert stacked W4)
    down_pairs = layer.tensor(f"{experts}.down_weight_pairs")
    args.selected_down_pairs = down_pairs.ptr
    args.selected_down_theta = layer.tensor(f"{experts}.down_weight_theta").ptr
    args.selected_down_channel_scales = layer.tensor(
        f"{experts}.down_weight_channel_scales"
    ).ptr
    down_qweight = layer.tensor(f"{experts}.stacked_down_qweight_pack8_decode")
    args.selected_stacked_down_qweight = down_qweight.ptr
    args.selected_stacked_down_qzeros = layer.tensor(f"{experts}.stacked_down_qzeros").ptr
    args.selected_stacked_down_scales = layer.tensor(f"{experts}.stacked_down_scales").ptr

    # Shared expert (PARO W4: 3 separate dense layers, each with its own
    # pairs/theta/channel_scales/qweight/qzeros/scales)
    shared = f"{moe_prefix}.shared_expert"
    shared_gate = f"{shared}.gate_proj"
    shared_up = f"{shared}.up_proj"
    shared_down = f"{shared}.down_proj"

    sg_pairs = layer.tensor(f"{shared_gate}.pairs")
    su_pairs = layer.tensor(f"{shared_up}.pairs")
    sd_pairs = layer.tensor(f"{shared_down}.pairs")
    args.shared_gate_pairs = sg_pairs.ptr
    args.shared_gate_theta = layer.tensor(f"{shared_gate}.theta").ptr
    args.shared_gate_channel_scales = layer.tensor(f"{shared_gate}.channel_scales").ptr
    sg_qweight = layer.tensor(f"{shared_gate}.qweight_pack8_decode")
    args.shared_gate_qweight = sg_qweight.ptr
    args.shared_gate_qzeros = layer.tensor(f"{shared_gate}.qzeros").ptr
    args.shared_gate_scales = layer.tensor(f"{shared_gate}.scales").ptr

    args.shared_up_pairs = su_pairs.ptr
    args.shared_up_theta = layer.tensor(f"{shared_up}.theta").ptr
    args.shared_up_channel_scales = layer.tensor(f"{shared_up}.channel_scales").ptr
    su_qweight = layer.tensor(f"{shared_up}.qweight_pack8_decode")
    args.shared_up_qweight = su_qweight.ptr
    args.shared_up_qzeros = layer.tensor(f"{shared_up}.qzeros").ptr
    args.shared_up_scales = layer.tensor(f"{shared_up}.scales").ptr

    args.shared_down_pairs = sd_pairs.ptr
    args.shared_down_theta = layer.tensor(f"{shared_down}.theta").ptr
    args.shared_down_channel_scales = layer.tensor(f"{shared_down}.channel_scales").ptr
    sd_qweight = layer.tensor(f"{shared_down}.qweight_pack8_decode")
    args.shared_down_qweight = sd_qweight.ptr
    args.shared_down_qzeros = layer.tensor(f"{shared_down}.qzeros").ptr
    args.shared_down_scales = layer.tensor(f"{shared_down}.scales").ptr

    # Layer-constant dims
    args.hidden_size = int(cfg.hidden_size)
    args.num_experts = int(cfg.num_experts)
    args.num_experts_per_tok = int(cfg.num_experts_per_tok)
    args.moe_intermediate_size = int(cfg.moe_intermediate_size)
    args.shared_expert_intermediate_size = int(cfg.shared_expert_intermediate_size)
    args.group_size = 128  # default; overridden per-call in dispatch()

    # krots (rotation count from pairs shape[0])
    args.krot_selected_gate_up = int(gate_up_pairs.shape[0])
    args.krot_selected_down = int(down_pairs.shape[0])
    args.krot_shared_gate = int(sg_pairs.shape[0])
    args.krot_shared_up = int(su_pairs.shape[0])
    args.krot_shared_down = int(sd_pairs.shape[0])

    # out_packed = qweight.shape[1] for transposed stacked, shape[0] for generic
    # transposed.  Selected (stacked) and shared (generic) differ.
    def stacked_out_packed(qw: Tensor) -> int:
        if len(qw.shape) < 3:
            raise ValueError("stacked transposed qweight must be 3D")
        return int(qw.shape[1])

    def generic_out_packed(qw: Tensor) -> int:
        if len(qw.shape) != 2:
            raise ValueError("generic transposed qweight must be 2D")
        return int(qw.shape[0])

    args.out_packed_selected_gate = stacked_out_packed(gate_qweight)
    args.out_packed_selected_up = stacked_out_packed(up_qweight)
    args.out_packed_selected_down = stacked_out_packed(down_qweight)
    args.out_packed_shared_gate = generic_out_packed(sg_qweight)
    args.out_packed_shared_up = generic_out_packed(su_qweight)
    args.out_packed_shared_down = generic_out_packed(sd_qweight)

    # Threads / tile defaults.  Match the Python wrapper defaults for the
    # verifier batched path (tokens > 1).
    args.router_threads = 256             # prefill_threads at tokens>1
    args.selected_threads = 64            # verifier-selected GEMV profile (B+1/top-k rows)
    args.shared_threads = 128             # small-batch shared expert (full-attn) default
    args.shared_prefill_tile_m = 16        # B+1 verifier small-batch tile (HIPENGINE_W4_PREFILL_SMALLBATCH_TILE_M default)
    args.shared_prefill_tile_n = 16        # rows < 32 → 16
    args.shared_down_mode = _linear_shared_down_mode()
    args.combine_threads = 256             # weighted_sum_shared_gate_combine_residual_batch default
