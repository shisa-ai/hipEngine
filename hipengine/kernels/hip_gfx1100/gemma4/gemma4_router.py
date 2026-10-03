"""Orchestrated Gemma 4 router on gfx1100.

Assembles the router from kernels that are each tested on their own:

    weightless RMSNorm * scale * hidden_size**-0.5   gemma4_router_prescale_bf16
    logits (BF16 hidden, F32 weights)                >= 1024 rows: bf16 -> f32
                                                     upcast + rocBLAS SGEMM over
                                                     the F32 weights, else
                                                     qwen35_router_logits_bf16_f32w_token_tile_16
    top-k + softmax + renormalise                    qwen35_router_select
    per-expert scale                                 gemma4_expert_weight_scale_f32

At ``tokens == 1`` (decode) all four stages take one fused launch instead --
``gemma4_router_topk_fused_bf16`` recomputes the prescaled row per block with
the same bf16 rounding, projects with an f32 reduction in a different order,
and runs the identical select + per-expert-scale math in the last block. The
chain below stays the strict unfused fallback under that composite: it is
what runs for ``tokens > 1``, and its four primitives are the registered
chain the fused launch can fall back to.

The reference order is pinned in ``gemma4_router_topk``: normalise without a
weight, apply the learned ``router.scale`` and ``hidden_size**-0.5``, project to
expert logits, softmax over **all** experts, keep the top ``k``, renormalise
those to sum to one, then multiply by ``per_expert_scale`` of the selected
experts.

Two details of that contract are easy to get wrong and are deliberate here:

* **The router norm is weightless.** Gemma 4 has no router-norm weight tensor —
  ``ffn_gate_inp`` and ``ffn_gate_inp_scale`` are the only router parameters in
  the GGUF, and ``_gemma4_gguf_names`` maps no norm weight for it. So
  ``gemma4_router_prescale_bf16``, which folds the norm, the learned scale, and
  ``root_size`` into one launch, is the whole pre-projection step.

* **``qwen35_router_select`` softmaxes over the selected set, not all experts.**
  That is equivalent, not a shortcut: the reference computes
  ``p_i = exp(l_i) / Z`` over every expert and then divides by the sum of the
  selected ``p``, so the global normaliser ``Z`` cancels exactly and both forms
  reduce to ``exp(l_i) / sum_selected exp(l_j)``. They differ only in rounding.

``logits_stride`` is ``num_experts``. The sibling
``qwen35_router_select_sigmoid_shared_kernel`` reads an extra shared-expert gate
column at ``logits[token * logits_stride + num_experts]``, but it only sigmoids
that column *after* selection — the selection itself reads ``[0, num_experts)``.
Gemma 4 has no shared expert gate, so the plain kernel is the right one and no
dummy column is needed.

The router projection is F32 while the activation is BF16. **The prescale
output is BF16, not F32** — ``gemma4_router_prescale_kernel`` is templated on
``scalar_t`` for both its input and its output, so the ``_bf16`` symbol writes
BF16. That makes the ``qwen35_router_logits_bf16_f32w`` family the matching
logits variant; the route takes its ``token_tile_16`` specialization at
``threads=128``, which is the same arithmetic class with a tile that fits this
shape -- see the call site for the measurement. Pairing a BF16 prescale buffer
with the F32-hidden variant reads the bf16 row as
f32 pairs and produces uncorrelated logits. Rounding the prescaled row to BF16
costs about 0.4% relative on values that are O(0.01) after ``root_size``, which
is the standard low-precision router path here.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field

from hipengine.core.ctypes_cache import signed_kernel_fn
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.core.memory import DeviceBuffer, free as hip_free, malloc
from hipengine.core.rocblas import get_rocblas
from hipengine.kernels.hip_gfx1100.convert.cast import bf16_to_f32
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
    build_gemma4_norm,
    gemma4_expert_weight_scale_f32,
    gemma4_router_prescale_bf16,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    qwen35_router_logits_bf16_f32w,
    qwen35_router_logits_bf16_f32w_token_tile_8,
    qwen35_router_logits_bf16_f32w_token_tile_16,
    qwen35_router_select,
)
from hipengine.kernels.registry import KernelKey, register

_BF16_BYTES = 2
_F32_BYTES = 4
_I64_BYTES = 8

_SYMBOL_ROUTER_TOPK_FUSED_BF16 = "hipengine_gemma4_router_topk_fused_bf16"
_ARGTYPES_ROUTER_TOPK_FUSED = (
    ctypes.c_void_p,  # hidden (tokens=1, BF16)
    ctypes.c_void_p,  # scale
    ctypes.c_void_p,  # proj
    ctypes.c_void_p,  # per_expert_scale
    ctypes.c_void_p,  # logits (scratch)
    ctypes.c_void_p,  # selected (int64, zeroed for the arrival counter)
    ctypes.c_void_p,  # routing weights
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_void_p,  # stream
)

# Size-based logits variant selection, the same shape of dispatch the C layer
# already does inside ``launch_qwen35_router_logits`` (``tokens >= 4`` picks
# the token tile over the untiled kernel). Measured on the production shape
# (hidden 2816, 128 experts) with ``scripts/
# gemma4_router_logits_variant_bench.py`` and the token sweep in the P8 worklog
# entry:
#
#   tokens   bf16_f32w   token_tile_16   winner
#        1     5.16 us      6.98 us      base
#       16     8.88 us      9.94 us      base
#       20    16.60 us     10.30 us      token_tile_16
#      512   183.68 us    94.06 us      token_tile_16
#
# Decode therefore takes the untiled path and prefill takes the tile. Choosing
# the tile unconditionally costs about 0.6% of decode throughput (30 layers x
# 1.8 us per step) for the prefill win; the crossover sits between 16 and 20
# tokens and 32 keeps clear of both sides of it.
_TOKEN_TILE_16_MIN_TOKENS = 32

# Thread width for the token tile at this shape, and it is not the 256 the
# binding defaults to. Grid and tile are fixed, so width only sets the K
# stride (blockDim.x * 8) and the reduction shape -- but the effect on time is
# large and non-monotone. Three warmed passes over hidden 2816 / 128 experts /
# 512 tokens, in milliseconds:
#
#   threads   1st      steady   TFLOP/s
#       64   0.0927   0.0927     3.98
#      128   0.0938   0.0666     5.43
#      256   0.0991   0.0943     3.72   <- the binding default
#      512   0.1546   0.1546     2.39
#
# 128 is the peak: every thread still owns a full K slice (all 128 are within
# the 352 vectors hidden 2816 needs) while each carries 2.75 vector steps
# instead of 1.4. At 64 the blocks starve, at 256 the threads go thin, at 512
# a third of them have no K range at all.
_TOKEN_TILE_16_THREADS = 128

# Row count where the projection leaves the hand-written tile for a bit-exact
# bf16 -> f32 upcast of the prescaled row plus rocBLAS SGEMM over the F32
# weights. Measured on the production shape (RX 7900 XTX, screen artifact
# benchmarks/results/2026-09-30-gemma4-p8-router-gemm-screen.json): the tile
# takes 0.0671 ms at 512 rows and 0.1835 ms at 1024, 0.7363 ms at 4096; the
# upcast + SGEMM pair takes 0.1265 / 0.1394 / 0.2444 -- so the pair wins from
# about 896 rows up and loses below about 768. 1024 is DEFAULT_PREFILL_BLOCK,
# the production full-block width, and a measured win point (1.32x at 1024,
# 3.01x at 4096); narrower blocks keep the tile, which wins beneath it. The
# weights stay F32 -- no downcast (the named F16 route was slower at 4096,
# 0.3030 ms, and about 100x less accurate, maxabs 5.0e-03). The accumulation
# order changes, so the teacher-forced gate gates this tier exactly as it
# gated the token-tile tier.
_ROUTER_SGEMM_MIN_TOKENS = 1024


@dataclass
class Gemma4RouterScratch:
    """Reusable device scratch for one router shape.

    Sized once from ``(tokens, hidden_size, num_experts, top_k)`` and reused
    across layers and decode steps.
    """

    tokens: int
    hidden_size: int
    num_experts: int
    top_k: int
    _buffers: list[DeviceBuffer] = field(default_factory=list, repr=False)
    _by_name: dict[str, DeviceBuffer] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("tokens", self.tokens),
            ("hidden_size", self.hidden_size),
            ("num_experts", self.num_experts),
            ("top_k", self.top_k),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")

    def buffer(self, name: str) -> DeviceBuffer:
        """Allocate ``name`` on first use, then reuse it."""

        existing = self._by_name.get(name)
        if existing is not None:
            return existing
        buf = malloc(self._size_of(name))
        self._by_name[name] = buf
        self._buffers.append(buf)
        return buf

    def free(self) -> None:
        for buf in self._buffers:
            hip_free(buf)
        self._buffers.clear()
        self._by_name.clear()

    def _sizes(self) -> dict[str, int]:
        """Byte size of every buffer this scratch can hold.

        One table shared by ``_size_of`` and ``resident_bytes`` so the
        allocator and the memory planner cannot drift apart.
        """
        rows = self.tokens
        sizes = {
            # BF16, not F32: the prescale kernel's output dtype is templated on
            # its input dtype, so the _bf16 symbol writes bf16.
            "prescaled": rows * self.hidden_size * _BF16_BYTES,
            # F32 upcast of ``prescaled`` for the >= 1024-row SGEMM tier.
            # buffer() is lazy, so a decode-only runner never allocates it;
            # resident_bytes still counts it, because the whole-table upper
            # bound is the point of _sizes().
            "prescaled_f32": rows * self.hidden_size * _F32_BYTES,
            "logits": rows * self.num_experts * _F32_BYTES,
        }
        return sizes

    def _size_of(self, name: str) -> int:
        try:
            return self._sizes()[name]
        except KeyError:
            raise ValueError(f"unknown scratch buffer {name!r}") from None

    def resident_bytes(self) -> int:
        """Upper bound on every device buffer this scratch can take."""
        return sum(self._sizes().values())


def gemma4_router_topk_fused_bf16(
    hidden_ptr: int,
    scale_ptr: int,
    proj_ptr: int,
    per_expert_scale_ptr: int,
    selected_ptr: int,
    weights_ptr: int,
    *,
    tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    scratch: Gemma4RouterScratch,
    eps: float = 1e-6,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Route one token through the fused prescale/projection/select/scale launch.

    ``tokens`` must be 1: this is the rows=1 decode kernel, and the caller
    keeps the four-launch chain for wider blocks (see
    ``gemma4_router_topk_bf16``). Argument layouts are the orchestrator's.

    ``selected`` is zeroed on ``stream`` before the launch because the kernel
    uses its first bytes as an arrival counter; the selection overwrites
    every int64 the token writes, and ``top_k >= 1`` guarantees no counter
    bytes survive the launch.
    """

    if int(tokens) != 1:
        raise ValueError("the fused router runs exactly one token")
    for name, value in (
        ("hidden_size", hidden_size),
        ("num_experts", num_experts),
        ("top_k", top_k),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")
    if top_k > 16:
        raise ValueError("fused router selection supports top_k up to 16")
    if (hidden_size, num_experts, top_k) != (
        scratch.hidden_size,
        scratch.num_experts,
        scratch.top_k,
    ):
        raise ValueError("router scratch shape does not match this call")
    if tokens > scratch.tokens:
        raise ValueError(f"tokens={tokens} exceeds scratch capacity {scratch.tokens}")

    logits = scratch.buffer("logits")
    library = library or build_gemma4_norm(load=True)
    runtime = runtime or get_hip_runtime()
    runtime.memset_async(selected_ptr, 0, top_k * _I64_BYTES, stream)
    fn = signed_kernel_fn(
        library,
        _SYMBOL_ROUTER_TOPK_FUSED_BF16,
        _ARGTYPES_ROUTER_TOPK_FUSED,
        ctypes.c_int,
    )
    err = fn(
        hidden_ptr,
        scale_ptr,
        proj_ptr,
        per_expert_scale_ptr,
        logits.ptr,
        selected_ptr,
        weights_ptr,
        hidden_size,
        num_experts,
        top_k,
        float(eps),
        float(hidden_size**-0.5),
        stream,
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def gemma4_router_topk_bf16(
    hidden_ptr: int,
    scale_ptr: int,
    proj_ptr: int,
    per_expert_scale_ptr: int,
    selected_ptr: int,
    weights_ptr: int,
    *,
    tokens: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    scratch: Gemma4RouterScratch,
    eps: float = 1e-6,
    stream: int = 0,
    backend: str | None = None,
) -> None:
    """Route ``tokens`` rows of BF16 hidden state to ``top_k`` of ``num_experts``.

    ``hidden_ptr`` is ``(tokens, hidden_size)`` BF16, ``scale_ptr`` is
    ``(hidden_size,)`` F32, ``proj_ptr`` is ``(num_experts, hidden_size)`` F32,
    and ``per_expert_scale_ptr`` is ``(num_experts,)`` F32. ``selected_ptr`` is
    ``(tokens, top_k)`` **int64** and ``weights_ptr`` is ``(tokens, top_k)`` F32
    — both token-major, and the int64 width is the router-select ABI, which is
    also what the expert forward's group-scatter chain consumes. Both are
    written directly; the scratch holds the ``prescaled``, ``prescaled_f32``,
    and ``logits`` intermediates.

    ``scratch`` must have been built for the same ``(tokens, hidden_size,
    num_experts, top_k)``; a mismatch is a programming error rather than a
    recoverable condition, so it raises.
    """

    for name, value in (
        ("tokens", tokens),
        ("hidden_size", hidden_size),
        ("num_experts", num_experts),
        ("top_k", top_k),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) cannot exceed num_experts ({num_experts})")
    # ``scratch`` is a capacity, not an identity: a caller sizes it for the
    # widest block it will ever route and then routes narrower blocks through it
    # (a prefill of N tokens, then N one-token decode steps). The shape
    # parameters must still match exactly, since they describe the weights.
    if (hidden_size, num_experts, top_k) != (
        scratch.hidden_size,
        scratch.num_experts,
        scratch.top_k,
    ):
        raise ValueError(
            f"scratch was built for (hidden_size, num_experts, top_k)="
            f"{(scratch.hidden_size, scratch.num_experts, scratch.top_k)} but this "
            f"call needs {(hidden_size, num_experts, top_k)}"
        )
    if tokens > scratch.tokens:
        raise ValueError(f"tokens={tokens} exceeds scratch capacity {scratch.tokens}")

    prescaled = scratch.buffer("prescaled")
    logits = scratch.buffer("logits")

    if tokens == 1:
        # Decode's single-token route takes the fused launch: all four
        # stages in one kernel with the same bf16 prescale rounding and the
        # same select math, only the projection's reduction order differs.
        # Selection is by shape, exactly like the token-tile logits variant
        # below; the chain stays the strict unfused fallback for wider
        # blocks and for anything the composite cannot serve.
        gemma4_router_topk_fused_bf16(
            hidden_ptr,
            scale_ptr,
            proj_ptr,
            per_expert_scale_ptr,
            selected_ptr,
            weights_ptr,
            tokens=tokens,
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            scratch=scratch,
            eps=eps,
            stream=stream,
        )
        return

    # Weightless norm, learned scale, and hidden_size**-0.5 in one launch.
    gemma4_router_prescale_bf16(
        hidden_ptr,
        scale_ptr,
        prescaled.ptr,
        tokens,
        hidden_size,
        eps,
        root_size=hidden_size**-0.5,
        stream=stream,
    )
    # Which logits schedule runs is a registered backend capability, not a
    # correctness question. The untiled kernel launches one block per
    # (expert-row, token), so it reads each F32 weight row once per token; an
    # N-token tiling launches one block per (expert-row, N tokens) and reads each
    # weight row once per N. gfx1151 declares "token_tile_8", so resolving the
    # capability turns it on without a backend branch. The router is 17.5 ms of
    # the prefill (worklog/entries/20260929T083000).
    #
    # EXACTNESS IS PER SCHEDULE, MEASURED, AND NOT UNIFORM
    # (scripts/gemma4_router_tile_equivalence.py):
    #   token_tile_8   bit-identical to the untiled kernel at tokens
    #                  1/512/777/4096 and an unrelated geometry.
    #   token_tile_16  NOT bit-identical -- 7.2e-07 max absolute delta on the
    #                  same inputs, so it reassociates. It runs when asked for by
    #                  name, and being asked for by name is the whole reason it is
    #                  reachable here; no backend declares it as a default.
    #   token_tile_4   the baseline, and has no tiled kernel -- the untiled
    #                  launch is what it denotes.
    # The logits feed top-k selection, so a changed logit can change which experts
    # a token routes to. That is why each schedule's exactness is recorded here
    # rather than assumed from the family.
    from hipengine.runtime.laguna_moe import resolve_laguna_router_logits_mode

    # The resolver rejects an unknown mode and the tiled schedules are named
    # explicitly, so a mode this cannot serve is a named miss rather than a
    # silent fall back to the untiled kernel.
    mode = resolve_laguna_router_logits_mode(
        backend if backend is not None else "hip_gfx1100"
    )
    if mode != "token_tile_4":
        # The backend names a tiled schedule, so run exactly that one at the
        # binding's default width.
        logits_kernel = {
            "token_tile_8": qwen35_router_logits_bf16_f32w_token_tile_8,
            "token_tile_16": qwen35_router_logits_bf16_f32w_token_tile_16,
        }[mode]
        logits_kernel(
            prescaled.ptr,
            proj_ptr,
            logits.ptr,
            tokens,
            hidden_size,
            num_experts,
            threads=512,
            stream=stream,
        )
    elif tokens >= _ROUTER_SGEMM_MIN_TOKENS:
        # The backend declares the baseline schedule (gfx1100 declares no tiled
        # mode), so the projection is size-tiered by measurement. A full
        # 1024-row block routes through the F32 SGEMM (see
        # ``_ROUTER_SGEMM_MIN_TOKENS``); narrower blocks take token_tile_16 at
        # 128 threads rather than the generic bf16_f32w entry point, which
        # defaults to threads=512 with a four-token tile. At this shape (hidden
        # 2816) that leaves threads 352..511 with no K range at all, so 31% of
        # every block idles behind a nine-round barrier tree for 64 FLOPs of
        # work per useful thread. Measured on the production shape: 0.2309 ms
        # -> 0.0666 ms per launch (1.60 -> 5.43 TFLOP/s), 3.5x, and both land
        # within 4e-06 of a float64 reference. Smaller token counts keep the
        # untiled path -- see ``_TOKEN_TILE_16_MIN_TOKENS`` -- and the width is
        # not the binding's default of 256 -- see ``_TOKEN_TILE_16_THREADS``.
        # Neither choice is bit-identical to what it replaces: the tiling and
        # the width both change the reduction, so the teacher-forced gate gates
        # this.
        #
        # Full prefill block: F32 weights through rocBLAS SGEMM. The cast is
        # a bit-exact bf16 -> f32 of the prescaled row; only the accumulation
        # order changes versus the tile (maxabs 4.7e-05 against a float64
        # reference, tile 3.8e-06), and the F32 weights survive -- the P8
        # row's downcast proposal would have lost precision to go fast.
        prescaled_f32 = scratch.buffer("prescaled_f32")
        bf16_to_f32(
            prescaled.ptr,
            prescaled_f32.ptr,
            tokens * hidden_size,
            stream=stream,
        )
        get_rocblas().sgemm_rowmajor_nt(
            prescaled_f32.ptr,
            proj_ptr,
            logits.ptr,
            rows=tokens,
            in_features=hidden_size,
            out_features=num_experts,
            stream=stream,
        )
    elif tokens >= _TOKEN_TILE_16_MIN_TOKENS:
        qwen35_router_logits_bf16_f32w_token_tile_16(
            prescaled.ptr,
            proj_ptr,
            logits.ptr,
            tokens,
            hidden_size,
            num_experts,
            threads=_TOKEN_TILE_16_THREADS,
            stream=stream,
        )
    else:
        qwen35_router_logits_bf16_f32w(
            prescaled.ptr,
            proj_ptr,
            logits.ptr,
            tokens,
            hidden_size,
            num_experts,
            stream=stream,
        )
    # logits_stride is num_experts: Gemma 4 has no shared-expert gate column.
    qwen35_router_select(
        logits.ptr,
        selected_ptr,
        weights_ptr,
        tokens,
        num_experts,
        num_experts,
        top_k,
        stream=stream,
    )
    gemma4_expert_weight_scale_f32(
        weights_ptr,
        selected_ptr,
        per_expert_scale_ptr,
        tokens,
        top_k,
        stream=stream,
    )


def register_gemma4_router_kernels(*, replace: bool = False) -> None:
    """Register the Gemma 4 router family against the four-axis registry.

    The fused rows=1 launch is the registered composite; the four chain
    primitives it composes are registered with their own keys
    (``router_prescale`` in the norm family, ``router_logits`` and
    ``router_select`` in the MoE family), which is the strict unfused chain
    under the composite.
    """

    for quant in ("gguf_q4_k_m", "gguf_q4_k_xl", "gguf_q8_0", "gguf"):
        register(
            KernelKey("hip_gfx1100", "router_topk", quant, "gemma4_fused_c1"),
            gemma4_router_topk_fused_bf16,
            replace=replace,
        )


register_gemma4_router_kernels()
