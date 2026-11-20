"""Physical prompt streaming is bounded by geometry and backend, not labels."""
from types import SimpleNamespace

import pytest

from hipengine.generation.qwen35_gguf_mtp2 import _physical_prompt_streaming_widths
from hipengine.kernels.policy import QWEN35_DENSE_H5120_GEOMETRY, QWEN35_MOE_H2048_E256_GEOMETRY


@pytest.mark.parametrize("backend,geometry,expected", [
    ("hip_gfx1100", QWEN35_DENSE_H5120_GEOMETRY, (1, 2, 8)),
    ("hip_gfx1100", QWEN35_MOE_H2048_E256_GEOMETRY, (1, 2)),
    ("hip_gfx1151", QWEN35_DENSE_H5120_GEOMETRY, (1, 2, 3, 4)),
    ("hip_gfx1151", QWEN35_MOE_H2048_E256_GEOMETRY, (1, 2)),
])
@pytest.mark.parametrize("stamp", ["MOSTLY_Q4_K_M", "MOSTLY_Q8_0", "NEW_QUANT_LABEL"])
def test_prompt_streaming_capability_is_independent_of_weight_label(backend, geometry, expected, stamp):
    owner = SimpleNamespace(_shared_runner=SimpleNamespace(weights=SimpleNamespace(
        geometry=geometry, file_type_name=stamp,
    )))
    generator = SimpleNamespace(backend=backend, execution_profile="production")
    assert _physical_prompt_streaming_widths(owner, generator) == expected
