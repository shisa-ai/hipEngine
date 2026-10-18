"""Tiny recurrent-prefill tails must not overflow fixed-width scratch."""
import pytest
from hipengine.runtime.qwen35_gguf_runner import _chunk_ranges


@pytest.mark.parametrize('total', [4095, 4096, 4097, 4098, 4099, 4100, 6145, 8193, 12001])
def test_prefill_partitions_fit_scratch(total):
    spans = _chunk_ranges(total, 4096, min_chunk_size=4)
    assert spans[0][0] == 0
    assert spans[-1][1] == total
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert all(4 <= end - start <= 4096 for start, end in spans)


def test_borrow_only_needed_rows_for_tiny_tail():
    assert _chunk_ranges(4097, 4096, min_chunk_size=4) == ((0, 4093), (4093, 4097))
    assert _chunk_ranges(8193, 4096, min_chunk_size=4) == ((0, 4096), (4096, 8189), (8189, 8193))
