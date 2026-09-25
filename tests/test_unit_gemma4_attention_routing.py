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

def test_split_slice_policy_covers_both_sides_of_every_threshold():
    """The split's context thresholds, exercised off the threshold points too.

    ``decode_slices`` decides whether a decode step takes the single-kernel path
    or the two-phase split, so its boundaries matter. Two measured facts set
    them: the split's fixed cost needs 1024 keys to be paid back (a 512-key
    context is 3.0% slower split than unsplit), and above that threshold more
    slices is better (4 slices beat 2 at a 1024-token prompt). Values unrelated
    to any boundary (300, 700, 3000, 7000) are included so the test does not only
    prove the thresholds.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import decode_slices

    policy = decode_slices
    assert policy(1, 512) == 1
    assert policy(300, 512) == 1  # unrelated to any boundary
    assert policy(700, 512) == 1  # unrelated, inside the single-kernel range
    assert policy(1023, 512) == 1  # just below the first splittable length
    assert policy(1024, 512) == 2  # exactly on it: 2 slices beat the single kernel
    assert policy(1025, 512) == 4  # just above it: doubling resumes
    assert policy(1152, 512) == 4  # the 1024p row's decode range
    assert policy(2048, 512) == 4  # the 4-slice plateau
    assert policy(3000, 512) == 4  # unrelated, inside the plateau
    assert policy(4096, 512) == 4  # the campaign's long-context row
    assert policy(7000, 512) == 4  # unrelated, inside the plateau
    assert policy(8192, 512) == 4  # the artifact's context cap


def test_narrow_heads_split_as_soon_as_the_split_is_entered():
    """A 256-wide head wants 4 slices at the entry threshold, not 2.

    The wide geometry's 512-keys-per-slice floor does not transfer: the key
    tile holds twice as many keys at head_dim 256, so a slice is short of work
    long before the key count is short. Measured on the sliding geometry at
    1024 keys, 4 slices are 192.2 us against 244.4 for 2 (paired, two passes
    each). These are Gemma 4's 25 sliding layers, and since the read range
    began handing them exactly 1024 keys, this case is on the primary row's
    hot path rather than a corner.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import decode_slices

    assert decode_slices(1, 256) == 1
    assert decode_slices(512, 256) == 1  # the 512p row: no split, as before
    assert decode_slices(1023, 256) == 1  # just below the entry threshold
    assert decode_slices(1024, 256) == 4  # exactly on it, and 4 not 2
    assert decode_slices(1152, 256) == 4  # what the 1024p row used to pass
    assert decode_slices(4096, 256) == 4
    assert decode_slices(8192, 256) == 4


def test_split_is_the_default_decode_path():
    """The split ships on, and this is what keeps it that way.

    It changes arithmetic, so it needed its execution-profile gate before it
    could be the default: that gate passed on 2026-09-25 (kl_max 0.0067 against
    the 0.05 bar, zero top-1 flips over 1023 teacher-forced rows). A change that
    reverts the selection back to the single kernel for ordinary decode lengths
    is a regression, not a safe default, and this test is what catches it.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import decode_slices

    assert decode_slices(1024, 512) == 2
    assert decode_slices(4096, 512) == 4
    assert decode_slices(8192, 512) == 4
    assert decode_slices(1024, 256) == 4
