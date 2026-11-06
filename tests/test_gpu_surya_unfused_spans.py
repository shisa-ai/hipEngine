"""Arbitrary-shape Surya unfused fallback honors all span visibility fields."""
import ctypes

import numpy as np
import pytest


def _hip_available():
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.parametrize("live,position,empty", [(17, 11, False), (301, 213, False), (0, 0, True)])
def test_unfused_nonstandard_geometry(monkeypatch, live, position, empty):
    import test_gpu_surya_kv_spans as fixture

    monkeypatch.setattr(fixture, "_NQ", 6)
    monkeypatch.setattr(fixture, "_NK", 2)
    monkeypatch.setattr(fixture, "_HD", 32)
    capacity = fixture._MAX_SEQ
    positions = np.arange(capacity, dtype=np.int64)
    positions[7] = -1
    evicted = np.zeros(capacity, dtype=np.bool_)
    evicted[3] = True
    actual, _, expected = fixture._run_decode_case(
        page_table=fixture._permuted_page_table(capacity // fixture._BLOCK_SIZE),
        token_positions=positions, evict_mask=evicted, live_count=live,
        row_position=position, with_parent=False, unfused_only=True, poison_masked=True,
    )
    np.testing.assert_allclose(actual, expected, atol=fixture._CPU_REFERENCE_ATOL, rtol=0)
    if empty:
        assert np.count_nonzero(actual) == 0
