"""The BF16 WMMA prefill candidate against the strict kernel it is meant to replace.

Every case drives both kernels through ctypes on the same device buffers with
the same arguments, so the only difference between the two outputs is the
arithmetic. The candidate is a changed-arithmetic production candidate, not a
strict variant: WMMA accumulates each 16-wide dot in the matrix unit's own
order, and the softmax weights reach the P*V dot as FP16 rather than FP32. The
tolerance below is declared, not derived from a strictness claim.

What the cases cover, and why each is here:

* The model's real sliding geometry (head_dim 256, 16 query heads, 8 KV heads,
  window 1024, causal) at 512 and 1024 tokens. 25 of the artifact's 30 layers
  are sliding layers, so this is the shape the kernel exists for.
* Chunked prefill: ``row_offset > 0`` with ``keys > tokens``, which is what a
  4096-token prompt produces (eight 512-token launches per layer, row_offset
  0, 512, ... and keys growing to 4096). A kernel that only works at
  ``row_offset == 0`` would pass the first case and fail here.
* ``window == 0`` with an arbitrary mask. ``window > 0`` is the caller's promise
  that the mask is sliding-causal and is what licenses the walk's block-level
  trim; ``window == 0`` promises nothing, so the walk must cover every column
  and only the mask may decide visibility. Random and blocky masks, including a
  fully masked row, exercise that.
* ``tokens < QUERY_ROWS``, where most of the query tile is past the end of the
  block and the kernel must not read or write those rows.
* The frame ``gemma4_layer`` actually passes: ``row_offset = window - 1`` and
  ``keys = tokens + window - 1``, with column 0 at ``key_begin``.

Skipped rather than failed when the process HIP runtime is unavailable, per the
repository's GPU-test guard.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")

# The sliding layers of gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf.
SLIDING_HEADS = 16
SLIDING_KV_HEADS = 8
SLIDING_HEAD_DIM = 256
SLIDING_WINDOW = 1024

# Declared tolerance for a changed-arithmetic candidate.
#
# The strict kernel's contract is bit-exactness against the decode twin, which
# the refuted tiled scalar variant reproduced. This kernel does not have that
# contract and does not claim it: it reassociates the score dot through the
# matrix unit and rounds the softmax weights to FP16 before the P*V dot.
#
# The measured bound is that the two kernels agree to within one BF16 unit in
# the last place on the great majority of elements and never more than a few --
# measured max abs 1.56e-2 at 4096 tokens, which is exactly one BF16 ulp at the
# largest output magnitude, against 99.0 percent of BF16 bit patterns equal.
# The assertions below are set at four BF16 ulps (2**-7 = 7.8e-3 relative on
# elements above five percent of the output peak), which is loose enough not to
# flake on a reordering and tight enough that a layout or masking error cannot
# hide inside it: every wrong-answer failure mode this kernel has produces a
# difference orders of magnitude larger.
RTOL = 7.9e-3
ATOL = 0.0


def _bf16(array: np.ndarray) -> np.ndarray:
    """Round an F32 array to BF16 bits, the same way the kernel's inputs are made."""

    bits = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return (rounded >> 16).astype(np.uint16)


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _sliding_keep_mask(tokens: int, keys: int, window: int, row_offset: int) -> np.ndarray:
    """The sliding-causal keep mask ``_keep_mask`` builds, in the kernel's frame.

    Column 0 is the caller's ``key_begin``, so row ``t``'s own position is
    ``row_offset + t`` and it keeps column ``c`` while ``c <= row_offset + t``
    and ``row_offset + t - c < window``.
    """

    rows = np.arange(row_offset, row_offset + tokens, dtype=np.int64)[:, None]
    columns = np.arange(keys, dtype=np.int64)[None, :]
    keep = columns <= rows
    if window > 0:
        keep &= (rows - columns) < window
    return np.ascontiguousarray(keep.astype(np.uint8))


def _run_both(
    *,
    tokens: int,
    keys: int,
    mask: np.ndarray,
    window: int,
    row_offset: int,
    num_heads: int = SLIDING_HEADS,
    num_kv_heads: int = SLIDING_KV_HEADS,
    head_dim: int = SLIDING_HEAD_DIM,
    scale: float = 1.0,
    seed: int = 20260930,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the strict kernel and the candidate on identical buffers."""

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        gemma4_attention_prefill_bf16 as strict,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_prefill_wmma import (
        gemma4_attention_prefill_wmma_bf16 as candidate,
    )

    rng = np.random.default_rng(seed)
    query = _bf16(rng.standard_normal((tokens, num_heads, head_dim)).astype(np.float32))
    key = _bf16(rng.standard_normal((keys, num_kv_heads, head_dim)).astype(np.float32))
    value = _bf16(rng.standard_normal((keys, num_kv_heads, head_dim)).astype(np.float32))
    out_strict = np.zeros((tokens, num_heads, head_dim), dtype=np.uint16)
    out_candidate = np.zeros_like(out_strict)

    buffers = []
    try:
        for array in (query, key, value, mask, out_strict, out_candidate):
            buffer = malloc(array.nbytes)
            buffers.append(buffer)
            copy_host_array_to_device(buffer, array)
        common = dict(
            tokens=tokens,
            keys=keys,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=scale,
            window=window,
            row_offset=row_offset,
        )
        strict(buffers[0].ptr, buffers[1].ptr, buffers[2].ptr, buffers[3].ptr, buffers[4].ptr, **common)
        candidate(
            buffers[0].ptr, buffers[1].ptr, buffers[2].ptr, buffers[3].ptr, buffers[5].ptr, **common
        )
        get_hip_runtime().device_synchronize()
        copy_device_to_host(host_array_ptr(out_strict), buffers[4])
        copy_device_to_host(host_array_ptr(out_candidate), buffers[5])
    finally:
        for buffer in buffers:
            free(buffer)
    return _bf16_to_f32(out_strict), _bf16_to_f32(out_candidate)


def _assert_matches(strict: np.ndarray, candidate: np.ndarray, *, context: str) -> None:
    """Compare two outputs, ignoring NaN placement but not NaN presence."""

    assert strict.shape == candidate.shape
    strict_nan = np.isnan(strict)
    candidate_nan = np.isnan(candidate)
    # A fully masked row is 0/0 in both kernels. Neither may invent or drop one.
    np.testing.assert_array_equal(
        strict_nan, candidate_nan, err_msg=f"{context}: NaN placement differs"
    )
    finite = ~strict_nan
    assert finite.any(), f"{context}: every element was NaN"
    a = strict[finite]
    b = candidate[finite]
    peak = float(np.abs(a).max())
    # Relative to the block's own peak rather than per element: an output that
    # is near zero carries no information at this tolerance, and dividing by it
    # would turn a one-ulp difference into a large ratio.
    assert peak > 0.0, f"{context}: all-zero output"
    worst = float(np.abs(a - b).max())
    assert worst <= RTOL * peak, (
        f"{context}: max abs diff {worst:.3e} exceeds {RTOL:.1e} of the output peak "
        f"{peak:.3e}"
    )


@pytest.mark.parametrize("tokens", [512, 1024])
def test_real_sliding_geometry_matches_the_strict_kernel(tokens):
    """The shape 25 of the artifact's 30 layers run at, at prefill block sizes."""

    mask = _sliding_keep_mask(tokens, tokens, SLIDING_WINDOW, 0)
    strict, candidate = _run_both(
        tokens=tokens, keys=tokens, mask=mask, window=SLIDING_WINDOW, row_offset=0
    )
    _assert_matches(strict, candidate, context=f"sliding tokens={tokens}")


@pytest.mark.parametrize("tokens", [512, 1024])
def test_production_mask_frame_matches_the_strict_kernel(tokens):
    """The frame ``gemma4_layer`` passes: keys is the mask width, not the context.

    ``_sliding_read_range`` moves the key pointer to ``start - window + 1`` and
    shortens ``keys`` to match, so the layer hands the kernel
    ``row_offset = window - 1`` and ``keys = tokens + window - 1`` with column 0
    at ``key_begin``. A kernel that assumes ``keys == tokens`` is correct in the
    benchmark and wrong here.
    """

    keys = tokens + SLIDING_WINDOW - 1
    row_offset = SLIDING_WINDOW - 1
    mask = _sliding_keep_mask(tokens, keys, SLIDING_WINDOW, row_offset)
    strict, candidate = _run_both(
        tokens=tokens, keys=keys, mask=mask, window=SLIDING_WINDOW, row_offset=row_offset
    )
    _assert_matches(strict, candidate, context=f"production frame tokens={tokens}")


@pytest.mark.parametrize("tokens,keys,row_offset", [(512, 1024, 512), (256, 4096, 3840)])
def test_chunked_prefill_matches_the_strict_kernel(tokens, keys, row_offset):
    """``row_offset > 0`` with ``keys > tokens``: the tail chunk of a long prompt.

    The second case is the extreme of the shape: the block sits at the end of a
    4096-token context, so its whole walk is behind it and the leading trim has
    to be exact rather than merely close.
    """

    mask = _sliding_keep_mask(tokens, keys, SLIDING_WINDOW, row_offset)
    strict, candidate = _run_both(
        tokens=tokens, keys=keys, mask=mask, window=SLIDING_WINDOW, row_offset=row_offset
    )
    _assert_matches(strict, candidate, context=f"chunked row_offset={row_offset}")


@pytest.mark.parametrize("window", [1, 2, 4, 17])
def test_a_narrow_window_pins_the_walk_trim(window):
    """A window where each row keeps only a few keys, so the trim is load-bearing.

    At the artifact's window of 1024 a single wrongly dropped column moves the
    output by about one part in 1024, which is inside the declared tolerance
    because it is genuinely that small. With a narrow window each row keeps
    ``window`` keys, so dropping one changes the weighted average by roughly
    ``1/window`` -- an order of magnitude above the tolerance. This is the case
    that makes the block-level ``first_kept``/``walk_end`` bounds an assertion
    rather than an argument, and the negative control for it is an off-by-one in
    either bound.
    """

    tokens, keys = 64, 96
    mask = _sliding_keep_mask(tokens, keys, window, 0)
    strict, candidate = _run_both(
        tokens=tokens, keys=keys, mask=mask, window=window, row_offset=0
    )
    _assert_matches(strict, candidate, context=f"narrow window={window}")


@pytest.mark.parametrize("mode", ["random", "blocky", "fully_masked_row"])
def test_window_zero_with_an_arbitrary_mask_matches_the_strict_kernel(mode):
    """``window == 0`` is the no-promise case, so only the mask decides.

    ``window > 0`` is what licenses the kernel's block-level walk trim; passing
    an arbitrary mask with a window set would let it drop a visible column.
    These masks are deliberately not sliding-causal, and one row is fully masked
    so the 0/0 both kernels produce is compared too.
    """

    tokens, keys = 128, 192
    rng = np.random.default_rng(4242)
    if mode == "random":
        mask = (rng.random((tokens, keys)) < 0.5).astype(np.uint8)
    elif mode == "blocky":
        mask = np.zeros((tokens, keys), dtype=np.uint8)
        for t in range(tokens):
            if t % 3 != 0:
                lo = (7 * t) % (keys // 2)
                mask[t, lo : lo + keys // 3] = 1
    else:
        mask = _sliding_keep_mask(tokens, keys, SLIDING_WINDOW, 0)
    mask = mask.copy()
    mask[-1, :] = 0

    strict, candidate = _run_both(
        tokens=tokens, keys=keys, mask=mask, window=0, row_offset=0
    )
    _assert_matches(strict, candidate, context=f"window=0 mask={mode}")


@pytest.mark.parametrize("tokens,keys", [(1, 1), (5, 8), (15, 16), (17, 32)])
def test_short_blocks_match_the_strict_kernel(tokens, keys):
    """``tokens < QUERY_ROWS``: the query tile is mostly past the end of the block.

    The kernel must not read a mask row or write an output row for a query past
    ``tokens``, and the padded columns of its Q tile must not reach a result.
    ``tokens = 17`` is one row into a second tile, which is the other edge.
    """

    mask = _sliding_keep_mask(tokens, keys, SLIDING_WINDOW, 0)
    strict, candidate = _run_both(
        tokens=tokens, keys=keys, mask=mask, window=SLIDING_WINDOW, row_offset=0
    )
    _assert_matches(strict, candidate, context=f"short tokens={tokens} keys={keys}")


def test_scale_is_applied_where_the_strict_kernel_applies_it():
    """Gemma 4 passes ``scale=1.0``, but the kernel must multiply by what it is given.

    A candidate that folded the scale into the staged Q instead would still be
    right at 1.0 and wrong here.
    """

    tokens, keys = 64, 64
    mask = _sliding_keep_mask(tokens, keys, SLIDING_WINDOW, 0)
    strict, candidate = _run_both(
        tokens=tokens,
        keys=keys,
        mask=mask,
        window=SLIDING_WINDOW,
        row_offset=0,
        scale=0.25,
    )
    _assert_matches(strict, candidate, context="scale=0.25")


def test_the_launcher_refuses_a_geometry_the_kernel_does_not_implement():
    """The head_dim-512 full layers are a capability miss, not a silent fallback."""

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_prefill_wmma import (
        gemma4_attention_prefill_wmma_bf16,
    )

    with pytest.raises(NotImplementedError, match="head_dim"):
        gemma4_attention_prefill_wmma_bf16(
            0, 0, 0, 0, 0,
            tokens=8,
            num_heads=16,
            num_kv_heads=2,
            head_dim=512,
            scale=1.0,
        )


def test_the_strict_kernel_is_still_what_the_production_path_selects():
    """Nothing routes a real request to the candidate.

    The unit is a candidate, not a promotion: the strict kernel stays the
    default, and this pins that rather than describing it.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import attention_symbol

    assert attention_symbol("bf16", tokens=512, head_dim=256) == (
        "hipengine_gemma4_attention_prefill_bf16"
    )
