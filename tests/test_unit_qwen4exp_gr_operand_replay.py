import numpy as np
import pytest

from scripts.qwen4exp_gr_operand_replay import error_metrics, reconstruct_planes, sample_indices


def test_geometry_sampling_is_bounded_and_includes_endpoints():
    assert sample_indices(1, 3).tolist() == [0]
    assert sample_indices(8, 3).tolist() == [0, 3, 7]
    with pytest.raises(ValueError):
        sample_indices(0, 3)


def test_three_planes_preserve_zero_and_reduce_reconstruction_error():
    values = np.linspace(-0.97, 1.13, 64, dtype=np.float32).reshape(2, 32)
    one = reconstruct_planes(values, planes=1)
    three = reconstruct_planes(values)
    assert np.max(np.abs(values - three)) < np.max(np.abs(values - one)) * 0.001
    np.testing.assert_array_equal(reconstruct_planes(np.zeros((1, 32), np.float32)), 0)
    with pytest.raises(ValueError):
        reconstruct_planes(np.zeros((1, 33), np.float32))


def test_metrics_separate_absolute_and_relative_error():
    result = error_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.25]))
    assert result["changed"] == 1
    assert result["max_abs"] == 0.25
    assert result["mse"] == 0.03125
    with pytest.raises(ValueError):
        error_metrics(np.array([np.nan]), np.array([0.0]))
