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

    ``decode_slices`` is the gate that decides whether a decode step takes the
    single-kernel path or the two-phase split, so its boundaries matter: 512 is
    the first length that can be split, 1024 the last 2-slice length, and 2048
    and beyond the 4-slice plateau. Values unrelated to any boundary (3000,
    7000) are included so the test does not only prove the thresholds.
    """

    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
        _split_slices_policy,
    )

    policy = _split_slices_policy
    assert policy(1) == 1
    assert policy(256) == 1  # unrelated to any boundary
    assert policy(511) == 1  # just below the first splittable length
    assert policy(512) == 1  # exactly on it: the split is not yet worth it
    assert policy(513) == 2  # just above it
    assert policy(768) == 2  # unrelated
    assert policy(1024) == 2  # the last 2-slice length
    assert policy(1025) == 4  # just above it
    assert policy(2048) == 4  # exactly on the 4-slice boundary
    assert policy(3000) == 4  # unrelated, inside the plateau
    assert policy(7000) == 4  # unrelated, inside the plateau
    assert policy(8192) == 4  # the artifact's context cap


def test_split_stays_off_the_default_path_until_its_gate_passes():
    """The precondition is the reason the default path is unchanged.

    The split changes arithmetic, so docs/EXECUTION-PROFILES.md requires its
    execution-profile gate first. Until that verdict exists the launcher must
    report the single-kernel path for every length - which is what keeps the
    incumbent's bit-exactness the production behaviour.
    """

    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as module

    assert module._SPLIT_GATE_PASSED is False, (
        "the split may only be the default path once its gate passes; if the gate "
        "has passed, retire the constant and the docs/REFACTOR.md entry instead of "
        "deleting this test"
    )
    for keys in (256, 1024, 8192):
        assert module.decode_slices(keys) == 1
