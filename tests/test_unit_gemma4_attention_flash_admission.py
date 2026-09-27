"""Admission policy for the Gemma 4 flash-attention prefill path.

The policy is the whole correctness story for routing a sliding-window layer to
a kernel that reads no mask, so the boundary is exercised on both sides of every
term rather than at the configured defaults.
"""

import pytest

from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    aotriton_prefill_admits,
    aotriton_prefill_head_dims,
)


def _admits(**overrides) -> bool:
    fields = {
        "rows": 512,
        "keys": 512,
        "head_dim": 256,
        "sliding_window": 1024,
        "mask_is_causal": True,
        "available": lambda head_dim: head_dim == 256,
    }
    fields.update(overrides)
    return aotriton_prefill_admits(**fields)


def test_vendored_images_cover_only_the_sliding_head_dim() -> None:
    # The runtime tree is pruned to BF16 head_dim 256 forward-attention images;
    # the 5 global layers run head_dim 512 and must not be admitted.
    assert aotriton_prefill_head_dims() == (256,)
    assert _admits(head_dim=512) is False
    assert _admits(head_dim=128) is False


def test_a_mask_that_is_not_asserted_causal_is_never_admitted() -> None:
    # The flash kernel reads no mask. A caller that cannot assert causality --
    # eviction, or a window that binds -- must keep the exact kernel even when
    # every other term holds.
    assert _admits(mask_is_causal=False) is False
    assert _admits(mask_is_causal=False, sliding_window=None, keys=4096) is False


def test_window_is_vacuous_up_to_and_including_its_width() -> None:
    # A sliding mask is exactly causal while the attended range is no wider than
    # the window, so ``keys <= window`` admits and one key past it does not.
    assert _admits(keys=1023, sliding_window=1024) is True
    assert _admits(keys=1024, sliding_window=1024) is True
    assert _admits(keys=1025, sliding_window=1024) is False
    # A width unrelated to the window and the block sizes, to catch a policy
    # that happens to be right at the configured points.
    assert _admits(keys=7777, sliding_window=1024) is False
    assert _admits(keys=7777, sliding_window=None) is True


def test_single_row_blocks_are_decode_and_stay_on_the_decode_kernel() -> None:
    assert _admits(rows=1) is False
    assert _admits(rows=2, keys=2) is True


def test_a_key_range_shorter_than_the_query_block_is_refused() -> None:
    assert _admits(rows=512, keys=511) is False
    assert _admits(rows=512, keys=512) is True


def test_admission_follows_the_runtime_probe() -> None:
    assert _admits(available=lambda head_dim: False) is False
    assert _admits(available=lambda head_dim: head_dim == 256) is True


def test_the_real_probe_refuses_head_dims_without_an_image() -> None:
    # No override: the policy must consult the vendored image set rather than a
    # caller's belief about it. head_dim 512 is refused whatever is installed.
    assert (
        aotriton_prefill_admits(
            rows=512,
            keys=512,
            head_dim=512,
            sliding_window=None,
            mask_is_causal=True,
        )
        is False
    )


@pytest.mark.parametrize("rows", [2, 16, 64, 512])
def test_admitted_blocks_are_not_special_to_one_block_size(rows: int) -> None:
    assert _admits(rows=rows, keys=rows) is True
