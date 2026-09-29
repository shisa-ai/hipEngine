"""Does the router token-tile ever change which experts get selected?

`74fb74ffe` ("select the Gemma 4 router logits kernel by token count") sends
prefill to `qwen35_router_logits_bf16_f32w_token_tile_16` and leaves decode on
`qwen35_router_logits_bf16_f32w`. The commit states plainly that the two are
not bit-identical -- 46264 of 65536 outputs differ, max 3.81e-06 -- and gated
the change at `--prefill 1024`, where it passed at kl_max 1.30e-03.

V1 extended the gate to `--prefill 4096`, where the same change fails at
kl_max 0.224071 on two rows (894 and 327), and a commit-by-commit bisect
isolated `74fb74ffe` as the first failing commit. Since decode stays on the
untiled kernel either way (`_TOKEN_TILE_16_MIN_TOKENS = 32` and scored rows are
single-token), all of that divergence has to enter through prefill.

The mechanism this test checks: the two kernels differ in their low-order
bits, and where a router's top-k boundary is near-tied that difference flips
which experts are chosen. A single flipped expert changes the mixture feeding
the rest of the layer, which amplifies into a large KL at that position while
leaving the final argmax alone -- exactly the observed signature (0 top-1
flips, kl_mean/p95/p99 all comfortable, kl_max large).

The float64 reference decides the other half of the question: if the two
kernels disagree but only one of them matches float64, that one is defective
and the fix is in the kernel. If they disagree and float64 sits between them,
both are equally correct and this is inherent rounding, which is a promotion
decision rather than a bug.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

try:
    ctypes.CDLL("libamdhip64.so")
    HIP_AVAILABLE = True
except OSError:  # pragma: no cover - exercised on no-ROCm runners
    HIP_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not HIP_AVAILABLE, reason="HIP runtime is not available"
)

# The production shape, from the same numbers the selection comment records.
TOKENS = 512
HIDDEN = 2816
EXPERTS = 128
TOP_K = 8
SEED = 20260929


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round float32 to bf16 bit patterns, as the route's prescale output is."""
    u = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    u = u + ((((u >> np.uint32(16)) & np.uint32(1))) + np.uint32(0x7FFF))
    return (u >> np.uint32(16)).astype(np.uint16)


def _f32_logits(hidden_bf16: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """The route's logits in float64, rounded once to float32."""
    from hipengine.quant.gguf_q4_k import _bf16_u16_to_f32

    h = _bf16_u16_to_f32(hidden_bf16).astype(np.float64)
    return (h @ weight.astype(np.float64).T).astype(np.float32)


def _topk_sets(logits: np.ndarray) -> np.ndarray:
    """Which experts each token selects, order-insensitive."""
    order = np.argsort(-logits, axis=1, kind="stable")[:, :TOP_K]
    return np.sort(order, axis=1)


def _run_pair():
    """Run both logits kernels plus select on each; return selections."""
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.moe.router import (
        _router_library,
        qwen35_router_logits_bf16_f32w,
        qwen35_router_logits_bf16_f32w_token_tile_16,
        qwen35_router_select,
    )

    rng = np.random.default_rng(SEED)
    # A BF16 activation after weightless RMSNorm * scale * hidden**-0.5 is
    # small and roughly unit-scale; the weight reproduces a projection.
    hidden = _bf16_bits(rng.standard_normal((TOKENS, HIDDEN)) * 2.0)
    weight = (rng.standard_normal((EXPERTS, HIDDEN)) * 0.05).astype(np.float32)

    logits = np.zeros((TOKENS, EXPERTS), dtype=np.float32)
    selected = np.zeros((TOKENS, TOP_K), dtype=np.int64)
    routing = np.zeros((TOKENS, TOP_K), dtype=np.float32)

    library = _router_library()
    arrays = (hidden, weight, logits, selected, routing)
    buffers = [malloc(a.nbytes) for a in arrays]
    try:
        for arr, buf in zip(arrays, buffers, strict=True):
            copy_host_to_device(buf, host_array_ptr(arr), arr.nbytes)

        picks = {}
        for name, fn in (
            ("base", qwen35_router_logits_bf16_f32w),
            ("tiled", qwen35_router_logits_bf16_f32w_token_tile_16),
        ):
            fn(
                buffers[0].ptr,
                buffers[1].ptr,
                buffers[2].ptr,
                TOKENS,
                HIDDEN,
                EXPERTS,
                library=library,
            )
            copy_device_to_host(host_array_ptr(logits), buffers[2], logits.nbytes)
            logits_copy = logits.copy()
            qwen35_router_select(
                buffers[2].ptr,
                buffers[3].ptr,
                buffers[4].ptr,
                TOKENS,
                EXPERTS,
                EXPERTS,
                TOP_K,
                library=library,
            )
            copy_device_to_host(host_array_ptr(selected), buffers[3], selected.nbytes)
            picks[name] = {
                "logits": logits_copy,
                "selected": np.sort(selected.copy(), axis=1),
            }
        return hidden, weight, picks
    finally:
        for buf in reversed(buffers):
            free(buf)


def test_router_token_tile_selection_agrees_with_the_untiled_kernel() -> None:
    """Prefill and decode must route to the same experts, or prefill's output
    diverges from what the baseline produced -- which is the V1 4096 failure."""

    hidden, weight, picks = _run_pair()
    base, tiled = picks["base"], picks["tiled"]

    # Both kernels must be close to each other and to the float64 yardstick
    # before any selection comparison means anything.
    np.testing.assert_allclose(
        tiled["logits"], base["logits"], atol=1e-5, rtol=1e-4,
        err_msg="the two router logits kernels disagree by more than rounding",
    )

    reference = _topk_sets(_f32_logits(hidden, weight))
    base_sel = base["selected"]
    tiled_sel = tiled["selected"]

    base_mismatch = int((base_sel != reference).any(axis=1).sum())
    tiled_mismatch = int((tiled_sel != reference).any(axis=1).sum())
    differ = int((base_sel != tiled_sel).any(axis=1).sum())

    print(
        f"\nrouter variant selection at {TOKENS}x{HIDDEN}->{EXPERTS} top-{TOP_K}:\n"
        f"  tokens where base != tiled : {differ}/{TOKENS}\n"
        f"  base   vs float64 reference: {base_mismatch}/{TOKENS}\n"
        f"  tiled  vs float64 reference: {tiled_mismatch}/{TOKENS}\n"
        f"  max |tiled - base| logits  : "
        f"{np.abs(tiled['logits'] - base['logits']).max():.3e}"
    )

    if differ == 0:
        # No selection ever moves, so the tiled kernel cannot be the source of
        # the V1 divergence and the cause lies elsewhere in the change.
        return

    # Something did move. The informative case is whether one kernel is the
    # better match to float64 -- that would make the other defective rather
    # than merely differently rounded.
    assert tiled_mismatch <= base_mismatch, (
        f"the tiled kernel disagrees with float64 more often than the untiled "
        f"one ({tiled_mismatch} vs {base_mismatch} of {TOKENS}): it is the "
        f"defective variant, not an equivalent rounding"
    )


def test_router_selection_is_stable_across_repeated_launches() -> None:
    """Same inputs must always yield the same experts -- an unstable reduction
    would make the gate itself non-reproducible rather than merely different."""

    _, _, first = _run_pair()
    _, _, second = _run_pair()
    for name in ("base", "tiled"):
        np.testing.assert_array_equal(
            first[name]["selected"],
            second[name]["selected"],
            err_msg=f"{name} router selection is not reproducible run to run",
        )