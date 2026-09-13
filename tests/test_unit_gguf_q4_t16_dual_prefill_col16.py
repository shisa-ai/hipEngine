"""The 16-column dense Q4T16 dual-SiLU prefill candidates are registry-reachable.

Why this exists: the gfx1151 Q4_K_M prefill profile
(``benchmarks/results/2026-09-13-qwen38-gfx1151-prefill-kernel-profile``) put
33-36% of prefill kernel time on
``gguf_q4_t16_dense_dual_wmma_prefill_silu_bf16_kernel<false,false,4,4>``, and
the tile-knob screen
(``benchmarks/results/2026-09-13-q4-dual-prefill-tile-knob-screen``) showed the
reachable row-tile shrinks are strictly worse while the column axis (32 -> 16)
is 1.27-1.39x faster on the single-matrix siblings.  The dual kernel could not
express a 16-column block without a new instantiation, so the candidate arms
live in new ``out_tiles_per_block``/``min_blocks_per_cu`` template parameters.

Those arms are registered but deliberately *unselected*: no dispatch or engine
code branches on them, and the strict parent owner stays the fallback.  This
test pins the three properties that make that safe, on CPU only:

* the arms resolve through the same four-axis key as the parent
  (``hip_gfx1100``/``linear_pair_silu``/``gguf_q4_k_t16_v1``);
* the gfx1151 backend alias covers them, so the campaign target can select
  them by variant name without touching backend dispatch; and
* the strict parent owner remains registered for every arm, and the 16-column
  arms validate a 16-column (not 32-column) output contract.
"""

from __future__ import annotations

import pytest

# Importing the family and the gfx1151 package registers both backend key spaces
# at import (collection) time, so the shared ``tests/conftest.py`` teardown
# baseline snapshot includes these variants and their gfx1151 aliases. The
# profile-resolution and dispatch gates resolve registered keys in every test, so
# the registration cannot be per-test here.
from hipengine.kernels.hip_gfx1100.quant import gguf_k_t16_selected_prefill as prefill
from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
from hipengine.kernels.registry import KernelKey, is_registered, resolve

register_gfx1151_kernels()

PARENT_VARIANT = "dense_dual_wmma_prefill_bf16_bf16_out"
# (registry variant, launcher attribute). The fused variant spelling drops the
# `silu` that the launcher and its exported C symbol keep, so both are pinned.
COL16_VARIANTS = (
    (
        "dense_dual_wmma_prefill_col16_row256_bf16_bf16_out",
        "gguf_q4_k_t16_dense_dual_wmma_prefill_col16_row256_silu_bf16_bf16_out",
    ),
    (
        "dense_dual_wmma_prefill_col16_row512_bf16_bf16_out",
        "gguf_q4_k_t16_dense_dual_wmma_prefill_col16_row512_silu_bf16_bf16_out",
    ),
)
COL16_VARIANT_NAMES = tuple(variant for variant, _ in COL16_VARIANTS)
LAYER = "linear_pair_silu"
QUANT = "gguf_q4_k_t16_v1"


def _key(variant: str, backend: str = "hip_gfx1100") -> KernelKey:
    return KernelKey(backend, LAYER, QUANT, variant)


@pytest.mark.parametrize("variant", COL16_VARIANT_NAMES)
def test_col16_dual_prefill_arms_resolve_on_gfx1100(variant: str) -> None:
    assert is_registered(_key(variant))
    assert callable(resolve(
        backend="hip_gfx1100",
        layer=LAYER,
        quant=QUANT,
        variant=variant,
    ))


@pytest.mark.parametrize("variant", COL16_VARIANT_NAMES)
def test_col16_dual_prefill_arms_keep_the_strict_parent_fallback(variant: str) -> None:
    """Every candidate arm keeps the exact parent owner registered beside it."""

    assert is_registered(_key(PARENT_VARIANT))
    parent = resolve(
        backend="hip_gfx1100",
        layer=LAYER,
        quant=QUANT,
        variant=PARENT_VARIANT,
    )
    assert (
        parent
        is prefill.gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out
    )
    assert resolve(
        backend="hip_gfx1100",
        layer=LAYER,
        quant=QUANT,
        variant=variant,
    ) is not parent


@pytest.mark.parametrize("variant", COL16_VARIANT_NAMES)
def test_col16_dual_prefill_arms_are_aliased_on_gfx1151(variant: str) -> None:
    """gfx1151 reaches the arms through the shared-body alias, not dispatch."""

    assert is_registered(_key(variant, backend="hip_gfx1151"))


class _RecordingLibrary:
    """Minimal ctypes-library stand-in that records one launch."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, symbol: str):
        def _entry(*args):
            self.calls.append((symbol, args))
            return 0

        _entry.argtypes = None
        _entry.restype = None
        return _entry


def _launch(variant: str, launcher: str, *, out_features: int):
    library = _RecordingLibrary()
    getattr(prefill, launcher)(
        0x1000,
        0x2000,
        0x3000,
        0x4000,
        64,
        5_120,
        out_features,
        library=library,
        runtime=object(),
    )
    return library.calls


def test_col16_arms_accept_a_sixteen_column_output_contract() -> None:
    """16 columns is the whole point of the arm; the parent still needs 32."""

    for variant, launcher in COL16_VARIANTS:
        calls = _launch(variant, launcher, out_features=16)
        assert len(calls) == 1
        assert calls[0][0] == f"hipengine_{launcher}"
        assert [arg.value for arg in calls[0][1][4:7]] == [64, 5_120, 16]

    with pytest.raises(ValueError, match="multiple of 32"):
        _launch(
            PARENT_VARIANT,
            "gguf_q4_k_t16_dense_dual_wmma_prefill_silu_bf16_bf16_out",
            out_features=16,
        )


def test_col16_arms_reject_an_odd_column_count() -> None:
    for variant, launcher in COL16_VARIANTS:
        with pytest.raises(ValueError, match="multiple of 16"):
            _launch(variant, launcher, out_features=8)
