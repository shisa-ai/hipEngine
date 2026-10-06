"""CPU contracts for canonical routed ownership in the natural Q4 screen."""
import numpy as np

from scripts.qwen4exp_q4_narrow_owner import canonical_risks


def test_membership_is_independent_of_compact_scatter_order():
    cols = 4
    first_lanes = np.array([2, 0, 1], np.int64)
    second_lanes = np.array([1, 2, 0], np.int64)
    first_queue = np.array([1, 6, 11], np.int32)
    second_queue = np.array([5, 10, 3], np.int32)
    np.testing.assert_array_equal(canonical_risks(first_queue, first_lanes, cols),
                                  canonical_risks(second_queue, second_lanes, cols))
    np.testing.assert_array_equal(canonical_risks(first_queue, first_lanes, cols),
                                  np.array([2, 7, 9]))


def test_empty_queue_is_valid():
    actual = canonical_risks(np.array([], np.int32), np.array([0], np.int64), 1280)
    assert actual.shape == (0,)
    assert actual.dtype == np.int64
