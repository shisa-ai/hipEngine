from __future__ import annotations

import numpy as np
import pytest

from hipengine.loading.qwen35_gguf_materialize import _gguf_ssm_a_to_kernel_a_log


def test_gguf_ssm_a_materialization_converts_decay_coefficients_to_kernel_log() -> None:
    coeff = np.asarray([-1.0, -0.25, -72.0], dtype=np.float32)
    converted = _gguf_ssm_a_to_kernel_a_log(coeff)

    assert converted.dtype == np.float32
    np.testing.assert_allclose(-np.exp(converted), coeff, rtol=1.0e-6, atol=1.0e-6)

    with pytest.raises(ValueError, match="negative decay coefficients"):
        _gguf_ssm_a_to_kernel_a_log(np.asarray([-1.0, 0.0], dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        _gguf_ssm_a_to_kernel_a_log(np.asarray([-1.0, np.nan], dtype=np.float32))
