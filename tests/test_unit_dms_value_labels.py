from __future__ import annotations

import numpy as np
import pytest

from hipengine.kvcache.dms_labels import future_attention_mass_cpu
from hipengine.kvcache.dms_value_labels import continuation_mass_cpu, value_perturbation_norm_cpu


def fixture():
    rng = np.random.default_rng(3)
    return rng.normal(size=(8, 4, 3)), rng.normal(size=(8, 2, 3)), rng.normal(size=(8, 2, 3))


def test_h0_h1_mass_shapes_and_causal_prefix() -> None:
    q, k, _ = fixture()
    h0 = future_attention_mass_cpu(q, k, window_size=2)
    h1 = continuation_mass_cpu(q[4:], k[:4], prefix_length=4, window_size=1)
    assert h0.shape == (8, 2)
    assert h1.shape == (4, 2)
    assert np.all(h0 >= 0) and np.all(h1 >= 0)
    assert np.count_nonzero(h1) > 0


def test_h2_value_perturbation_is_finite_and_protected_rows_zero() -> None:
    q, k, v = fixture()
    h2 = value_perturbation_norm_cpu(q, k, v, window_size=2)
    assert h2.shape == (8, 2)
    assert np.isfinite(h2).all()
    assert np.all(h2[-3:] == 0)


def test_h2_rejects_bad_values_and_nonfinite_input() -> None:
    q, k, v = fixture()
    with pytest.raises(ValueError, match="V"):
        value_perturbation_norm_cpu(q, k, v[:, :1], window_size=2)
    q[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        continuation_mass_cpu(q, k, prefix_length=4, window_size=1)
