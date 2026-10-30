"""Regression: gfx1151 aliases must survive collection-time baseline restore.

``hipengine.kernels.hip_gfx1151`` registers aliases for the gfx1100 kernels
that exist when the package is imported. A test module that imports the
backend before the gfx1100 Q8T16 GEMV source module therefore leaves that
source key without an alias. The shared collection baseline used to snapshot
the incomplete alias set and restore it after every test, which made the
backend-before-``test_unit_gguf_linear_dispatch`` collection order fail.

Keep the imports in this order: importing the backend first is what makes the
missing alias observable without depending on another test file.
"""

from __future__ import annotations

import hipengine.kernels.hip_gfx1151  # noqa: F401
import hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_t16_gemv  # noqa: F401

from hipengine.kernels.registry import (
    KernelKey,
    is_registered,
    restore_registry_for_tests,
)
from tests import conftest as tests_conftest


_GFX1151_Q8_T16_DUAL = KernelKey(
    "hip_gfx1151",
    "linear",
    "gguf_q8_0_t16_v1",
    "t16_dual_gemv_decode_bf16_bf16_out",
)


def test_collection_baseline_keeps_aliases_for_late_gfx1100_imports() -> None:
    baseline = tests_conftest._BASELINE_KERNELS
    assert baseline is not None
    assert _GFX1151_Q8_T16_DUAL in baseline

    restore_registry_for_tests(baseline)
    assert is_registered(_GFX1151_Q8_T16_DUAL)
