"""Orchestrated Gemma 4 router on gfx1100.

Assembles the router from kernels that are each tested on their own:

    weightless RMSNorm * scale * hidden_size**-0.5   gemma4_router_prescale_bf16
    logits (BF16 hidden, F32 weights)                qwen35_router_logits_bf16_f32w
    top-k + softmax + renormalise                    qwen35_router_select
    per-expert scale                                 gemma4_expert_weight_scale_f32

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
BF16. That makes ``qwen35_router_logits_bf16_f32w`` the matching logits variant;
pairing a BF16 prescale buffer with the F32-hidden variant reads the bf16 row as
f32 pairs and produces uncorrelated logits. Rounding the prescaled row to BF16
costs about 0.4% relative on values that are O(0.01) after ``root_size``, which
is the standard low-precision router path here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hipengine.core.memory import DeviceBuffer, free as hip_free, malloc
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_norm import (
    gemma4_expert_weight_scale_f32,
    gemma4_router_prescale_bf16,
)
from hipengine.kernels.hip_gfx1100.moe.router import (
    qwen35_router_logits_bf16_f32w,
    qwen35_router_logits_bf16_f32w_token_tile_8,
    qwen35_router_logits_bf16_f32w_token_tile_16,
    qwen35_router_select,
)

_BF16_BYTES = 2
_F32_BYTES = 4


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

    def _size_of(self, name: str) -> int:
        rows = self.tokens
        sizes = {
            # BF16, not F32: the prescale kernel's output dtype is templated on
            # its input dtype, so the _bf16 symbol writes bf16.
            "prescaled": rows * self.hidden_size * _BF16_BYTES,
            "logits": rows * self.num_experts * _F32_BYTES,
        }
        try:
            return sizes[name]
        except KeyError:
            raise ValueError(f"unknown scratch buffer {name!r}") from None


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
    written directly; the scratch holds only the ``prescaled`` and ``logits``
    intermediates.

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
    logits_kernel = {
        "token_tile_4": qwen35_router_logits_bf16_f32w,
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
    )    # logits_stride is num_experts: Gemma 4 has no shared-expert gate column.
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
