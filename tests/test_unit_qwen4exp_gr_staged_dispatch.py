"""Unit coverage for the HC-1 GR staged-f32 selector and its registration.

The staged-f32 GR-up tile keeps the ``dense_wide256`` tiling and LDS staging but
does the multiply in f32 FMA on f32 operands, so its only arithmetic difference
from the fused exact parent is the summation order. It is screened at 2.388x
(``benchmarks/results/2026-09-29-gr-staged-f32-screen/``) and is **default-off**:
the variant is not bit-exact with the parent, so under
``docs/EXECUTION-PROFILES.md`` it needs the production numerical gate before it
can be selected, and that gate has not been run for this layer.

These tests pin the selection contract, not the kernel's arithmetic. The
arithmetic contract lives in ``tests/test_gpu_qwen4exp_gr_staged_f32.py``.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_gr_staged_f32 import (
    GR_STAGED_F32_VARIANT,
)

GR_LAYER = "linear+gr_gated_mean"
GR_QUANT = "gguf_q8_0"


def _key() -> KernelKey:
    return KernelKey("hip_gfx1100", GR_LAYER, GR_QUANT, GR_STAGED_F32_VARIANT)


class GrStagedRegistrationTest(unittest.TestCase):
    def test_variant_is_registered_under_the_gr_layer(self) -> None:
        self.assertTrue(is_registered(_key()))

    def test_variant_key_is_the_screened_tile_name(self) -> None:
        # The screen and the gate script both name this tile by its shape, so a
        # rename here would silently disconnect them from the dispatch.
        self.assertEqual(
            GR_STAGED_F32_VARIANT, "coltile2_branch4_rowbatch4_staged_f32"
        )


class GrStagedSelectorTest(unittest.TestCase):
    """The selector is default-off and declines shapes the tile cannot serve."""

    def setUp(self) -> None:
        from hipengine.runtime.qwen4_exp_runner import _qwen4_exp_gr_staged_enabled

        self.enabled = _qwen4_exp_gr_staged_enabled
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("HIPENGINE_QWEN4_EXP_GR_STAGED", None)

    def test_default_is_off(self) -> None:
        self.assertFalse(self.enabled(1024, 320))

    def test_enabled_at_prefill_rows_with_a_tile_aligned_k(self) -> None:
        with mock.patch.dict(os.environ, {"HIPENGINE_QWEN4_EXP_GR_STAGED": "1"}):
            self.assertTrue(self.enabled(1024, 320))

    def test_declines_rows_at_or_below_the_fused_floor(self) -> None:
        # rows <= 256 stays on the fused exact parent, matching the wide and iu8
        # routes: the staged tile only pays once the projection is prefill-sized.
        with mock.patch.dict(os.environ, {"HIPENGINE_QWEN4_EXP_GR_STAGED": "1"}):
            self.assertFalse(self.enabled(256, 320))
            self.assertFalse(self.enabled(16, 320))

    def test_declines_a_k_extent_the_tile_cannot_serve(self) -> None:
        # BM=128/BN=64/BK=32 with a fixed 256-thread block: a K extent that is
        # not a multiple of the K-tile cannot be served by this tile, and the
        # launcher rejects a mismatched thread count rather than retiling.
        with mock.patch.dict(os.environ, {"HIPENGINE_QWEN4_EXP_GR_STAGED": "1"}):
            self.assertFalse(self.enabled(1024, 321))
            self.assertTrue(self.enabled(1024, 320))
            self.assertTrue(self.enabled(1024, 384))

    def test_explicit_falsey_values_stay_off(self) -> None:
        for value in ("", "0", "false", "False"):
            with mock.patch.dict(os.environ, {"HIPENGINE_QWEN4_EXP_GR_STAGED": value}):
                self.assertFalse(
                    self.enabled(1024, 320), f"{value!r} must not enable the route"
                )


if __name__ == "__main__":
    unittest.main()
