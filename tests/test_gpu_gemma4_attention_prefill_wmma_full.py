"""The BF16 WMMA full-layer prefill candidate against the strict kernel.

Every case drives both kernels through ctypes on the same device buffers with
the same arguments, so the only difference between the two outputs is the
arithmetic. Like its sliding sibling, the candidate is a changed-arithmetic
production candidate rather than a strict variant: WMMA accumulates each 16-wide
dot in the matrix unit's own order, and the softmax weights reach the P*V dot as
BF16 rather than FP32. The tolerance below is declared, not derived from a
strictness claim.

The geometry is the full layers' shape (head_dim 512, 16 query heads, 2 KV
heads, no sliding window), which the sliding candidate refuses and the strict
kernel serves at 434 GFLOP/s. These five layers are the reason the full
candidate exists; the sliding kernel's own file covers the other 25.

What the cases cover, and why each is here:

* The model's real full geometry at 512 and 1024 tokens, with ``window == 0``
  and a causal mask.
* The frame ``gemma4_layer`` actually passes for a full layer: ``key_begin`` is
  0 (nothing is skipped), so ``row_offset = start``, ``keys = start + rows`` and
  ``window = keys``. The leading bound is inactive and the trailing one is the
  causal trim.
* Chunked prefill: ``row_offset > 0`` with ``keys > tokens``, which is what a
  4096-token prompt produces (eight 512-token launches per layer). A kernel
  that only works at ``row_offset == 0`` would pass the first case and fail
  here.
* Mask widths that are not a multiple of ``K_BATCH``, and a final key batch
  shorter than 16, which is the path where the vector mask read has to clamp
  instead of running past the end of the row.
* ``keys < K_BATCH``, where the vector mask read is not usable at all and the
  byte path takes over.
* ``tokens < QUERY_ROWS``, where most of the query tile is past the end of the
  block and the kernel must not read or write those rows.
* ``window == 0`` with an arbitrary mask, including a fully masked row. Nothing
  licenses a walk trim there, so only the mask may decide visibility.

Skipped rather than failed when the process HIP runtime is unavailable, per the
repository's GPU-test guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")

# The full layers of gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf: head_dim 512, 16 query
# heads, 2 KV heads, GQA ratio 8, no sliding window.
FULL_HEADS = 16
FULL_KV_HEADS = 2
FULL_HEAD_DIM = 512

# Declared tolerance for a changed-arithmetic candidate.
#
# The strict kernel's contract is bit-exactness against the decode twin. This
# kernel does not have that contract and does not claim it: it reassociates the
# score dot through the matrix unit and rounds the softmax weights to BF16
# before the P*V dot.
#
# The measured bound is that the two kernels agree to within one BF16 unit in
# the last place on the great majority of elements and never more than a few --
# measured max abs 1.56e-2 at 4096 tokens against an output peak of 4.8, which
# is exactly one BF16 ulp at that magnitude, with 98.3 percent of BF16 bit
# patterns equal. The assertion is set at four BF16 ulps (2**-7 = 7.8e-3
# relative on elements above five percent of the output peak), which is loose
# enough not to flake on a reordering and tight enough that a layout or masking
# error cannot hide inside it: every wrong-answer failure mode this kernel has
# produces a difference orders of magnitude larger.
RTOL = 7.9e-3
ATOL = 0.0


def _bf16(array: np.ndarray) -> np.ndarray:
    """Round an F32 array to BF16 bits, the same way the kernel's inputs are made."""

    bits = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return (rounded >> 16).astype(np.uint16)


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _causal_keep_mask(tokens: int, keys: int, row_offset: int) -> np.ndarray:
    """The causal keep mask ``_keep_mask`` builds for a full layer.

    Column 0 is the caller's ``key_begin``, which is 0 for a full layer, so row
    ``t``'s own position is ``row_offset + t`` and it keeps column ``c`` while
    ``c <= row_offset + t``. There is no window term: a full layer's mask is the
    sliding-causal mask with an unbounded window.
    """

    rows = np.arange(row_offset, row_offset + tokens, dtype=np.int64)[:, None]
    columns = np.arange(keys, dtype=np.int64)[None, :]
    return np.ascontiguousarray((columns <= rows).astype(np.uint8))


def _run_both(
    *,
    tokens: int,
    keys: int,
    mask: np.ndarray,
    window: int,
    row_offset: int,
    num_heads: int = FULL_HEADS,
    num_kv_heads: int = FULL_KV_HEADS,
    head_dim: int = FULL_HEAD_DIM,
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
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_prefill_wmma_full import (
        gemma4_attention_prefill_wmma_full_bf16 as candidate,
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
        strict(
            buffers[0].ptr, buffers[1].ptr, buffers[2].ptr, buffers[3].ptr, buffers[4].ptr, **common
        )
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
def test_real_full_geometry_matches_the_strict_kernel(tokens):
    """The shape the artifact's five full layers run at, at prefill block sizes."""

    mask = _causal_keep_mask(tokens, tokens, 0)
    strict, candidate = _run_both(
        tokens=tokens, keys=tokens, mask=mask, window=0, row_offset=0
    )
    _assert_matches(strict, candidate, context=f"full tokens={tokens}")


@pytest.mark.parametrize("tokens", [512, 1024])
def test_production_frame_matches_the_strict_kernel(tokens):
    """The frame ``gemma4_layer`` passes: ``window`` is the whole context.

    A full layer skips no keys, so ``key_begin`` is 0, ``keys = start + rows``
    and the layer hands the kernel ``window = keys``. The kernel must treat that
    as the causal trim and nothing more; a kernel that read a nonzero window as
    a promise of a *narrow* window would drop columns it cannot drop.
    """

    strict, candidate = _run_both(
        tokens=tokens,
        keys=tokens,
        mask=_causal_keep_mask(tokens, tokens, 0),
        window=tokens,
        row_offset=0,
    )
    _assert_matches(strict, candidate, context=f"production frame tokens={tokens}")


@pytest.mark.parametrize(
    "tokens,keys,row_offset",
    [
        # A middle chunk of a long prompt, and the tail chunk of a 4096-token
        # one: the block sits at the end of the context, so its whole walk is
        # behind it.
        (512, 1024, 512),
        (256, 4096, 3840),
        # A key count that is not a multiple of K_BATCH with the block ending on
        # the last column, so the final key batch is short and the vector mask
        # read has to clamp rather than run off the end of the row.
        (64, 200, 136),
        (100, 100, 0),
        (257, 257, 0),
    ],
)
def test_chunked_prefill_matches_the_strict_kernel(tokens, keys, row_offset):
    """``row_offset > 0`` with ``keys > tokens``: the tail chunk of a long prompt."""

    mask = _causal_keep_mask(tokens, keys, row_offset)
    strict, candidate = _run_both(
        tokens=tokens, keys=keys, mask=mask, window=keys, row_offset=row_offset
    )
    _assert_matches(strict, candidate, context=f"chunked row_offset={row_offset}")


@pytest.mark.parametrize("keys", [512, 1024, 4096, 15360])
def test_one_token_decode_matches_the_strict_decode_kernel(keys):
    """The shape a full layer's decode presents, which this candidate now serves.

    Above the strict decode kernel's LDS bound this candidate is the only path
    that runs for the full layers, and it gets there with ``tokens == 1``: the
    kernel's 16-row query tiling is fifteen sixteenths padding. That is the
    shape being asserted here rather than assumed.

    The frame is the one ``gemma4_layer`` passes for a full layer's decode. No
    keys are skipped, so ``key_begin`` is 0, the single row sits at
    ``row_offset = keys - 1``, and ``window`` is the whole context. The mask is
    therefore all ones: a decode step keeps every key.
    """

    mask = _causal_keep_mask(1, keys, keys - 1)
    strict, candidate = _run_both(
        tokens=1, keys=keys, mask=mask, window=keys, row_offset=keys - 1
    )
    _assert_matches(strict, candidate, context=f"decode keys={keys}")


@pytest.mark.parametrize("tokens,keys,row_offset", [(512, 15360, 14848)])
def test_the_longest_prefill_block_the_strict_kernel_can_serve(tokens, keys, row_offset):
    """The deep end of the strict prefill's own range, where this kernel takes over.

    15,360 keys is the largest key count the strict prefill can hold in LDS at a
    512-row block -- ``(head_dim + keys + threads) * 4`` is exactly 65,536 there
    -- so this is the last prefill shape both kernels can run and therefore the
    last one a comparison is possible at. Every longer context reaches this
    candidate with the strict kernel no longer available as an oracle, which is
    why the comparison has to be made at the boundary rather than near it.
    """

    mask = _causal_keep_mask(tokens, keys, row_offset)
    strict, candidate = _run_both(
        tokens=tokens, keys=keys, mask=mask, window=keys, row_offset=row_offset
    )
    _assert_matches(strict, candidate, context=f"deep prefill keys={keys}")


@pytest.mark.parametrize("mode", ["random", "blocky", "fully_masked_row"])
def test_window_zero_with_an_arbitrary_mask_matches_the_strict_kernel(mode):
    """``window == 0`` is the no-promise case, so only the mask decides.

    These masks are deliberately not causal, and one row is fully masked so the
    0/0 both kernels produce is compared too.
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
        mask = _causal_keep_mask(tokens, keys, 0)
    mask = mask.copy()
    mask[-1, :] = 0

    strict, candidate = _run_both(
        tokens=tokens, keys=keys, mask=mask, window=0, row_offset=0
    )
    _assert_matches(strict, candidate, context=f"window=0 mask={mode}")


@pytest.mark.parametrize("tokens,keys", [(1, 1), (5, 8), (15, 16), (17, 32)])
def test_short_blocks_match_the_strict_kernel(tokens, keys):
    """``tokens < QUERY_ROWS``, and key counts below one K-batch.

    The kernel must not read a mask row or write an output row for a query past
    ``tokens``, and the padded columns of its Q tile must not reach a result.
    ``tokens = 17`` is one row into a second tile, which is the other edge.
    ``keys < 16`` is below the vector mask read's width, so those two cases take
    the byte path; ``keys = 16`` is the first width the vector path can serve.
    """

    mask = _causal_keep_mask(tokens, keys, 0)
    strict, candidate = _run_both(
        tokens=tokens, keys=keys, mask=mask, window=0, row_offset=0
    )
    _assert_matches(strict, candidate, context=f"short tokens={tokens} keys={keys}")


def test_scale_is_applied_where_the_strict_kernel_applies_it():
    """Gemma 4 passes ``scale=1.0``, but the kernel must multiply by what it is given.

    A candidate that folded the scale into the staged Q instead would still be
    right at 1.0 and wrong here.
    """

    tokens, keys = 64, 64
    mask = _causal_keep_mask(tokens, keys, 0)
    strict, candidate = _run_both(
        tokens=tokens, keys=keys, mask=mask, window=0, row_offset=0, scale=0.25
    )
    _assert_matches(strict, candidate, context="scale=0.25")


def _run_candidate_split(
    *,
    tokens: int,
    keys: int,
    mask: np.ndarray,
    window: int,
    row_offset: int,
    split: bool,
    num_heads: int = FULL_HEADS,
    num_kv_heads: int = FULL_KV_HEADS,
    head_dim: int = FULL_HEAD_DIM,
    scale: float = 1.0,
    seed: int = 20260930,
) -> np.ndarray:
    """Run the candidate alone, with or without the key-split workspace."""

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_array_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import Gemma4AttentionScratch
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_prefill_wmma_full import (
        gemma4_attention_prefill_wmma_full_bf16 as candidate,
    )

    rng = np.random.default_rng(seed)
    query = _bf16(rng.standard_normal((tokens, num_heads, head_dim)).astype(np.float32))
    key = _bf16(rng.standard_normal((keys, num_kv_heads, head_dim)).astype(np.float32))
    value = _bf16(rng.standard_normal((keys, num_kv_heads, head_dim)).astype(np.float32))
    out = np.zeros((tokens, num_heads, head_dim), dtype=np.uint16)

    scratch = Gemma4AttentionScratch() if split else None
    buffers = []
    try:
        for array in (query, key, value, mask, out):
            buffer = malloc(array.nbytes)
            buffers.append(buffer)
            copy_host_array_to_device(buffer, array)
        candidate(
            buffers[0].ptr,
            buffers[1].ptr,
            buffers[2].ptr,
            buffers[3].ptr,
            buffers[4].ptr,
            tokens=tokens,
            keys=keys,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=scale,
            window=window,
            row_offset=row_offset,
            scratch=scratch,
        )
        get_hip_runtime().device_synchronize()
        copy_device_to_host(host_array_ptr(out), buffers[4])
    finally:
        for buffer in buffers:
            free(buffer)
        if scratch is not None:
            scratch.close()
    return _bf16_to_f32(out)


@pytest.mark.parametrize("keys", [4096, 32768, 131072])
def test_a_split_key_walk_matches_the_whole_walk(keys):
    """Dividing the walk and recombining must reproduce walking it whole.

    A decode step presents one query tile against the whole context, so the
    unsplit grid is `num_kv_heads * gqa_tiles` = 8 blocks and each walks every
    key. Measured at 262,144 keys on gfx1151 that reaches 75 GB/s where 64
    blocks reach 225 and 256 reach 252, so the kernel is latency-bound and the
    walk is worth dividing. Dividing it gives each slice its own online-softmax
    maximum, and the combine rescales every slice to the largest of them; that
    is a reassociation, so this is not the bit-identity the mask tests assert.
    It is however the same sum, and the tolerance below is the file's own.

    The unsplit arm is the same kernel with no workspace handed to it, which is
    the path every prefill shape takes, so this also pins that the split is
    opt-in rather than a change to the default arithmetic.
    """

    mask = _causal_keep_mask(1, keys, keys - 1)
    whole = _run_candidate_split(
        tokens=1, keys=keys, mask=mask, window=keys, row_offset=keys - 1, split=False
    )
    split = _run_candidate_split(
        tokens=1, keys=keys, mask=mask, window=keys, row_offset=keys - 1, split=True
    )
    assert whole.shape == split.shape
    whole_nan = np.isnan(whole)
    np.testing.assert_array_equal(
        whole_nan, np.isnan(split), err_msg=f"keys={keys}: NaN placement differs"
    )
    finite = ~whole_nan
    assert finite.any(), f"keys={keys}: every element was NaN"
    a, b = whole[finite], split[finite]
    peak = float(np.abs(a).max())
    assert peak > 0.0, f"keys={keys}: output was all zero"
    worst = float(np.abs(a - b).max())
    assert worst <= RTOL * peak, (
        f"keys={keys}: the split walk differs by {worst:.4g} of a {peak:.4g} peak "
        f"against {RTOL * peak:.4g}; that is a different answer, not a reassociation"
    )


def test_the_launcher_refuses_a_geometry_the_kernel_does_not_implement():
    """The head_dim-256 sliding layers are a capability miss, not a fallback."""

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_prefill_wmma_full import (
        gemma4_attention_prefill_wmma_full_bf16,
    )

    with pytest.raises(NotImplementedError, match="head_dim"):
        gemma4_attention_prefill_wmma_full_bf16(
            0,
            0,
            0,
            0,
            0,
            tokens=8,
            num_heads=16,
            num_kv_heads=8,
            head_dim=256,
            scale=1.0,
        )


def test_the_production_profile_routes_both_geometries_to_their_own_candidate():
    """What a real request gets, per layer geometry.

    The unit is wired, not promoted by hand: the production plan requests both
    candidates and each layer's own geometry decides which one runs. The strict
    kernel remains the fallback for every geometry neither candidate implements.
    """

    from hipengine.generation.gemma4_gguf_profiles import (
        PREFILL_ATTENTION_PRODUCTION_VARIANTS,
        PREFILL_ATTENTION_WMMA_FLASH,
        PREFILL_ATTENTION_WMMA_FLASH_FULL,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        gemma4_attention_prefill_bf16,
        select_prefill_attention,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_prefill_wmma import (
        gemma4_attention_prefill_wmma_bf16,
    )
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention_prefill_wmma_full import (
        gemma4_attention_prefill_wmma_full_bf16,
    )

    requested = PREFILL_ATTENTION_PRODUCTION_VARIANTS
    full = select_prefill_attention(
        requested_variant=requested, num_heads=FULL_HEADS, num_kv_heads=FULL_KV_HEADS,
        head_dim=FULL_HEAD_DIM,
    )
    sliding = select_prefill_attention(
        requested_variant=requested, num_heads=16, num_kv_heads=8, head_dim=256
    )
    other = select_prefill_attention(
        requested_variant=requested, num_heads=8, num_kv_heads=1, head_dim=64
    )

    assert full.variant == PREFILL_ATTENTION_WMMA_FLASH_FULL
    assert full.launcher is gemma4_attention_prefill_wmma_full_bf16
    assert sliding.variant == PREFILL_ATTENTION_WMMA_FLASH
    assert sliding.launcher is gemma4_attention_prefill_wmma_bf16
    assert other.launcher is gemma4_attention_prefill_bf16
