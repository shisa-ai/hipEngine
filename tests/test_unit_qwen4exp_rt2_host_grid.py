"""CPU contracts for the RT-2 host-sized tile-grid screen."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.qwen4exp_rt2_host_grid import (
    checked_total_rows,
    launch_rows,
    static_tile_bound,
)


def test_static_bound_covers_every_small_count_partition():
    for rows in range(1, 34):
        for experts in range(1, 6):
            def partitions(total, slots):
                if slots == 1:
                    yield (total,)
                else:
                    for count in range(total + 1):
                        for rest in partitions(total - count, slots - 1):
                            yield (count, *rest)
            bound = static_tile_bound(rows, experts)
            for counts in partitions(rows, experts):
                actual = sum((count + 15) // 16 for count in counts)
                assert actual <= bound, (rows, experts, counts)


@pytest.mark.parametrize("total", [0, 15, 17, 48, 144])
def test_checked_total_rejects_empty_unaligned_undercoverage_or_overflow(total):
    with pytest.raises(ValueError):
        checked_total_rows(total, compact_rows=64, static_rows=128)


def test_checked_total_accepts_exact_and_static_edges():
    assert checked_total_rows(64, compact_rows=64, static_rows=128) == 64
    assert checked_total_rows(128, compact_rows=64, static_rows=128) == 128


def test_candidate_charges_readback_but_parent_does_not(monkeypatch):
    import scripts.qwen4exp_rt2_host_grid as probe
    calls = []
    runtime = SimpleNamespace(stream_synchronize=lambda stream: calls.append(("sync", stream)))
    def download(host_ptr, buffer, nbytes, *, runtime):
        calls.append(("D2H", nbytes))
        host[:] = 80
    monkeypatch.setattr(probe, "copy_device_to_host", download)
    host = np.empty(1, np.int64)
    kwargs = dict(runtime=runtime, stream=7, total_buffer=object(), host=host,
                  compact_rows=64, static_rows=128)
    assert launch_rows("static", **kwargs) == 128
    assert calls == []
    assert launch_rows("host", **kwargs) == 80
    assert calls == [("sync", 7), ("D2H", 8)]


def test_unknown_arm_fails_closed():
    with pytest.raises(ValueError, match="arm"):
        launch_rows("bogus", runtime=None, stream=0, total_buffer=None,
                    host=None, compact_rows=64, static_rows=128)
