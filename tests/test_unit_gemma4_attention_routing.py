"""Decode-vs-prefill symbol selection for Gemma 4 attention wrappers (unit tier).

The public wrappers are the registry entry point for the ``prefill_attention``
family. A decode step (``tokens == 1``) must select the warp-32 decode kernel;
multi-token blocks keep the original block kernel. ``head_dim > 512`` has no
decode-kernel register layout, so it stays on the block kernel.
"""

from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import attention_symbol


def test_single_token_selects_decode_kernel():
    assert attention_symbol("bf16", tokens=1, head_dim=128) == (
        "hipengine_gemma4_attention_decode_bf16"
    )
    assert attention_symbol("f32", tokens=1, head_dim=128) == (
        "hipengine_gemma4_attention_decode_f32"
    )


def test_multi_token_selects_prefill_kernel():
    assert attention_symbol("bf16", tokens=2, head_dim=128) == (
        "hipengine_gemma4_attention_prefill_bf16"
    )
    assert attention_symbol("f32", tokens=64, head_dim=128) == (
        "hipengine_gemma4_attention_prefill_f32"
    )


def test_large_head_dim_still_selects_decode_kernel():
    # The decode kernel mirrors the block kernel's capped thread width, so no
    # head_dim needs the block kernel at tokens == 1.
    assert attention_symbol("bf16", tokens=1, head_dim=1024) == (
        "hipengine_gemma4_attention_decode_bf16"
    )
    assert attention_symbol("bf16", tokens=1, head_dim=512) == (
        "hipengine_gemma4_attention_decode_bf16"
    )


def test_small_head_dim_stays_on_decode_kernel():
    # head_dim < 32 collapses to the single-chain (shfl-only) layout.
    assert attention_symbol("bf16", tokens=1, head_dim=6) == (
        "hipengine_gemma4_attention_decode_bf16"
    )


def test_the_dimension_partition_entry_covers_both_sides_of_the_threshold():
    """The decode route's context threshold, exercised off the threshold too.

    ``decode_split_engages`` decides whether a decode step takes the
    single-kernel path or the two-pass dimension partition, so its boundary
    matters. One measured fact sets it: the partition's fixed cost needs 1024
    keys to be paid back (a 512-key context is 3.0% slower partitioned than
    unpartitioned). Values unrelated to any boundary (300, 700, 3000, 7000) are
    included so the test does not only prove the threshold. The head width is
    not a parameter: the partition's grid is derived from it at launch, so no
    geometry moves the entry.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        DECODE_SPLIT_MIN_KEYS,
        decode_split_engages,
    )

    assert DECODE_SPLIT_MIN_KEYS == 1024
    assert not decode_split_engages(keys=1)
    assert not decode_split_engages(keys=300)  # unrelated to any boundary
    assert not decode_split_engages(keys=700)  # unrelated, inside the single-kernel range
    assert not decode_split_engages(keys=1023)  # just below the first splittable length
    assert decode_split_engages(keys=1024)  # exactly on it
    assert decode_split_engages(keys=1025)  # just above it
    assert decode_split_engages(keys=1152)  # the 1024p row's decode range
    assert decode_split_engages(keys=2048)
    assert decode_split_engages(keys=3000)  # unrelated, inside the plateau
    assert decode_split_engages(keys=4096)  # the campaign's long-context row
    assert decode_split_engages(keys=7000)  # unrelated, inside the plateau
    assert decode_split_engages(keys=8192)  # the artifact's context cap


def test_the_partition_entry_is_the_default_decode_policy():
    """The partition entry ships on, and this is what keeps it that way.

    It changes arithmetic, so it needed its execution-profile gate before it
    could be the default: that gate passed on 2026-09-25 (kl_max 0.0067 against
    the 0.05 bar, zero top-1 flips over 1023 teacher-forced rows). A change that
    reverts the entry back to the single kernel for ordinary decode lengths is a
    regression, not a safe default, and this test is what catches it. Which
    route a launch finally takes above the entry is the launcher's decision: an
    admitted flash packing takes flash from the same threshold.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        decode_split_engages,
    )

    assert decode_split_engages(keys=1024)
    assert decode_split_engages(keys=4096)
    assert decode_split_engages(keys=8192)


def test_the_signed_slice_count_abi_is_gone():
    """The route is named, not encoded in a count.

    ``decode_slices`` was the last consumer of the signed ``split_slices``
    argument: one selected the single kernel, above one the dimension
    partition, and a negative value was the flash request. All three are route
    values now, so the helper must not come back as a second selection path.
    """

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention

    assert not hasattr(gemma4_attention, "decode_slices")
    assert gemma4_attention.DECODE_ROUTE_STRICT == 0
    assert gemma4_attention.DECODE_ROUTE_SPLIT == 1
    assert gemma4_attention.DECODE_ROUTE_FLASH == 2
